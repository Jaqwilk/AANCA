"""Outcome-blind checkpoint data plane for the unchanged original confirmatory study.

This module deliberately owns no authority, CLI, lifecycle, terminal-seal, outcome,
or predecessor-discovery behavior.  Fresh execution is built from the frozen
180-checkpoint contract without filesystem reads.  Successor execution can be built
only from a same-process preparation that performs the frozen predecessor inspection,
physical copy, and verification itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointExecutionContract,
    ConfirmatoryCheckpointFileIdentity,
    ConfirmatoryCheckpointPhysicalIdentity,
    _hold_private_checkpoint_snapshot,
    _is_reparse,
    _lease_checkpoint_execution_contract_single_use,
    _named_streams,
    _read_private_checkpoint_bytes,
    _register_checkpoint_execution_contract,
)
from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    ConfirmatoryFrozenBlocker,
    ConfirmatoryMatrixArtifacts,
    ConfirmatoryRotationInputs,
    execute_confirmatory_matrix,
    run_confirmatory_frozen_feature_oof,
    run_confirmatory_image_oof,
)
from histo_audit.experiment.original_confirmatory_resume import (
    ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256,
    ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256,
    ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256,
    ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD,
    BoundOriginalConfirmatoryResumeRequest,
    OriginalConfirmatoryPredecessorSnapshot,
    OriginalConfirmatoryResumeContract,
    OriginalConfirmatoryResumeCopyReceipt,
    OriginalConfirmatorySuccessorExecutionContract,
    build_original_confirmatory_resume_evidence,
    build_original_confirmatory_successor_execution_contract,
    copy_original_confirmatory_checkpoints,
    inspect_original_confirmatory_predecessor,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryObservedLabelSet,
)
from histo_audit.experiment.study_contracts import ConfirmatoryMatrixPlan
from histo_audit.models.cnn import validate_confirmatory_checkpoint_artifact
from histo_audit.pannuke.publication import (
    PublishedPath,
    anchored_physical_copy_session,
)
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    PrimaryExecutionGateEvidence,
)

ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY = "exact_all_180_predeclared_fit_directives_v1"
ORIGINAL_CONFIRMATORY_AUTHORITY_PROJECTION_POLICY = (
    "original_confirmatory_lossless_checkpoint_authority_projection_v1"
)
ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY = (
    "original_confirmatory_successor_precopy_projection_v1"
)
ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY = "anchored_physical_no_overwrite_no_link_v1"
ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_RECEIPT_POLICY = (
    "original_confirmatory_successor_post_e_copy_receipt_v1"
)
ORIGINAL_CONFIRMATORY_CHECKPOINT_EXECUTION_CONTRACT_FILENAME = (
    "original_confirmatory_checkpoint_execution_contract.json"
)
ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT = 108
ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT = 36
ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT = 5
_HEX = frozenset("0123456789abcdef")

type OriginalConfirmatoryExecutionMode = Literal["fresh", "successor_resume"]


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


def _expectation_payload(contract: OriginalConfirmatoryResumeContract) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": value.relative_path,
            "cell_id": value.cell_id,
            "fold_id": value.fold_id,
            "expected_configuration": dict(value.expected_configuration),
            "expected_model_metadata": dict(value.expected_model_metadata),
            "expected_data_and_split_sha256": dict(value.expected_data_and_split_sha256),
        }
        for value in contract.checkpoint_expectations
    ]


def _require_exact_original_contract(contract: OriginalConfirmatoryResumeContract) -> None:
    if not isinstance(contract, OriginalConfirmatoryResumeContract):
        raise ConfirmatoryCheckpointContractError(
            "original confirmatory execution requires the typed frozen resume contract"
        )
    if (
        contract.plan_semantic_sha256 != ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256
        or contract.config_semantic_sha256 != ORIGINAL_CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or contract.controls_binding_sha256 != ORIGINAL_CONFIRMATORY_CONTROLS_BINDING_SHA256
        or len(contract.all_cell_ids) != ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT
        or len(set(contract.all_cell_ids)) != ORIGINAL_CONFIRMATORY_PLANNED_CELL_COUNT
        or len(contract.cnn_cell_ids) != ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT
        or len(set(contract.cnn_cell_ids)) != ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT
        or tuple(sorted(contract.all_cell_ids)) != contract.all_cell_ids
        or tuple(sorted(contract.cnn_cell_ids)) != contract.cnn_cell_ids
        or not set(contract.cnn_cell_ids).issubset(contract.all_cell_ids)
        or len(contract.checkpoint_expectations) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or canonical_sha256(_expectation_payload(contract)) != contract.checkpoint_allowlist_sha256
    ):
        raise ConfirmatoryCheckpointContractError(
            "original confirmatory frozen contract is incomplete or altered"
        )
    expected_paths = {
        f"cells/{cell_id}/checkpoints/fold_{fold_id:02d}.pt"
        for cell_id in contract.cnn_cell_ids
        for fold_id in range(ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT)
    }
    observed_paths = {value.relative_path for value in contract.checkpoint_expectations}
    observed_identities = {
        (value.cell_id, value.fold_id) for value in contract.checkpoint_expectations
    }
    expected_identities = {
        (cell_id, fold_id)
        for cell_id in contract.cnn_cell_ids
        for fold_id in range(ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT)
    }
    if (
        observed_paths != expected_paths
        or observed_identities != expected_identities
        or len({value.casefold() for value in observed_paths})
        != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
    ):
        raise ConfirmatoryCheckpointContractError(
            "original confirmatory checkpoint allowlist is not exact 36x5 coverage"
        )


def _directive_from_expectation(
    *,
    execution_mode: OriginalConfirmatoryExecutionMode,
    expectation: Any,
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
    configuration = dict(expectation.expected_configuration)
    metadata = dict(expectation.expected_model_metadata)
    data_and_split = dict(expectation.expected_data_and_split_sha256)
    configuration_json = _canonical_json(configuration)
    metadata_json = _canonical_json(metadata)
    data_and_split_json = _canonical_json(data_and_split)
    maximum_epochs = configuration.get("epochs")
    if type(maximum_epochs) is not int:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint expectation lacks an exact maximum epoch count"
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


def _build_execution_contract(
    *,
    execution_mode: OriginalConfirmatoryExecutionMode,
    retry_of_run_id: str | None,
    directives: tuple[ConfirmatoryCheckpointDirective, ...],
    predecessor_checkpoint_read_performed: bool,
    predecessor_checkpoint_copy_performed: bool,
    predecessor_snapshot_sha256: str | None,
    predecessor_copy_receipt_sha256: str | None,
) -> ConfirmatoryCheckpointExecutionContract:
    ordered = tuple(sorted(directives, key=lambda value: (value.cell_id, value.fold_id)))
    contract = ConfirmatoryCheckpointExecutionContract(
        execution_mode=execution_mode,
        contract_profile="original_confirmatory_exact_180",
        retry_of_run_id=retry_of_run_id,
        directives=ordered,
        directives_sha256=canonical_sha256([value.as_dict() for value in ordered]),
        predecessor_checkpoint_read_performed=(predecessor_checkpoint_read_performed),
        predecessor_checkpoint_copy_performed=(predecessor_checkpoint_copy_performed),
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        predecessor_snapshot_sha256=predecessor_snapshot_sha256,
        predecessor_copy_receipt_sha256=predecessor_copy_receipt_sha256,
    )
    return _register_checkpoint_execution_contract(
        contract,
        expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    )


def build_original_confirmatory_fresh_checkpoint_execution_contract(
    contract: OriginalConfirmatoryResumeContract,
) -> ConfirmatoryCheckpointExecutionContract:
    """Build exact fresh 180-fold execution without reading any predecessor or outcome."""

    _require_exact_original_contract(contract)
    directives = tuple(
        _directive_from_expectation(
            execution_mode="fresh",
            expectation=expectation,
            action="fresh_fit",
            checkpoint_sha256=None,
            checkpoint_size_bytes=None,
            source_predecessor_checkpoint=None,
            destination_imported_checkpoint=None,
            completed_epochs_before_fit=0,
            stopped_early_before_fit=None,
        )
        for expectation in contract.checkpoint_expectations
    )
    return _build_execution_contract(
        execution_mode="fresh",
        retry_of_run_id=None,
        directives=directives,
        predecessor_checkpoint_read_performed=False,
        predecessor_checkpoint_copy_performed=False,
        predecessor_snapshot_sha256=None,
        predecessor_copy_receipt_sha256=None,
    )


@dataclass(frozen=True, slots=True)
class PreparedOriginalConfirmatorySuccessor:
    """Opaque same-process result of inspect -> copy -> verify."""

    snapshot: OriginalConfirmatoryPredecessorSnapshot
    copy_receipt: OriginalConfirmatoryResumeCopyReceipt
    successor_execution: OriginalConfirmatorySuccessorExecutionContract
    resume_evidence_sha256: str
    preparation_sha256: str


_PREPARED_SUCCESSORS: dict[
    int,
    tuple[PreparedOriginalConfirmatorySuccessor, str],
] = {}
_PREPARED_SUCCESSORS_LOCK = threading.RLock()


def _prepared_digest(value: PreparedOriginalConfirmatorySuccessor) -> str:
    return canonical_sha256(
        {
            "snapshot_sha256": value.snapshot.snapshot_sha256,
            "destination_checkpoint_tree_sha256": (
                value.copy_receipt.destination_checkpoint_tree_sha256
            ),
            "successor_directives_sha256": (value.successor_execution.directives_sha256),
            "resume_evidence_sha256": value.resume_evidence_sha256,
            "preparation_sha256": value.preparation_sha256,
            "retry_of_run_id": value.snapshot.request.retry_of_run_id,
            "successor_run_id": value.snapshot.request.successor_run_id,
        }
    )


def prepare_original_confirmatory_successor_checkpoint_execution(
    *,
    request: BoundOriginalConfirmatoryResumeRequest,
    contract: OriginalConfirmatoryResumeContract,
) -> PreparedOriginalConfirmatorySuccessor:
    """Inspect and copy once using only the frozen production resume implementation."""

    _require_exact_original_contract(contract)
    snapshot = inspect_original_confirmatory_predecessor(request, contract)
    copy_receipt = copy_original_confirmatory_checkpoints(snapshot)
    resume_evidence = build_original_confirmatory_resume_evidence(
        snapshot,
        copy_receipt,
    )
    successor_execution = build_original_confirmatory_successor_execution_contract(
        snapshot,
        copy_receipt,
    )
    evidence_sha = resume_evidence.get(ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD)
    if (
        not isinstance(evidence_sha, str)
        or len(evidence_sha) != 64
        or not set(evidence_sha).issubset(_HEX)
        or resume_evidence.get("outcome_artifacts_read") is not False
        or resume_evidence.get("automatic_retry_allowed") is not False
    ):
        raise ConfirmatoryCheckpointContractError(
            "frozen successor resume evidence lacks its exact outcome-blind/no-retry hash"
        )
    preparation_sha = canonical_sha256(
        {
            "snapshot_sha256": snapshot.snapshot_sha256,
            "destination_checkpoint_tree_sha256": (copy_receipt.destination_checkpoint_tree_sha256),
            "successor_directives_sha256": successor_execution.directives_sha256,
            "resume_evidence_sha256": evidence_sha,
            "outcome_artifacts_read": resume_evidence.get("outcome_artifacts_read"),
            "automatic_retry_allowed": resume_evidence.get("automatic_retry_allowed"),
        }
    )
    prepared = PreparedOriginalConfirmatorySuccessor(
        snapshot=snapshot,
        copy_receipt=copy_receipt,
        successor_execution=successor_execution,
        resume_evidence_sha256=evidence_sha,
        preparation_sha256=preparation_sha,
    )
    digest = _prepared_digest(prepared)
    with _PREPARED_SUCCESSORS_LOCK:
        _PREPARED_SUCCESSORS[id(prepared)] = (prepared, digest)
    return prepared


def build_original_confirmatory_successor_checkpoint_execution_contract(
    prepared: PreparedOriginalConfirmatorySuccessor,
) -> ConfirmatoryCheckpointExecutionContract:
    """Derive actions only from a registered, freshly reverified preparation."""

    if not isinstance(prepared, PreparedOriginalConfirmatorySuccessor):
        raise ConfirmatoryCheckpointContractError(
            "successor execution requires an opaque prepared successor"
        )
    with _PREPARED_SUCCESSORS_LOCK:
        registered = _PREPARED_SUCCESSORS.get(id(prepared))
    if (
        registered is None
        or registered[0] is not prepared
        or registered[1] != _prepared_digest(prepared)
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor preparation is caller-supplied, changed, or not registered"
        )
    snapshot = prepared.snapshot
    receipt = prepared.copy_receipt
    _require_exact_original_contract(snapshot.contract)
    # These calls re-open and revalidate the destination copy immediately before
    # deriving the neutral data-plane action map.
    resume_evidence = build_original_confirmatory_resume_evidence(snapshot, receipt)
    derived_execution = build_original_confirmatory_successor_execution_contract(
        snapshot,
        receipt,
    )
    if (
        derived_execution != prepared.successor_execution
        or resume_evidence.get(ORIGINAL_CONFIRMATORY_RESUME_EVIDENCE_HASH_FIELD)
        != prepared.resume_evidence_sha256
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor snapshot/copy/execution changed after preparation"
        )
    expectations = {
        value.relative_path: value for value in snapshot.contract.checkpoint_expectations
    }
    states = {value.relative_path: value for value in snapshot.checkpoint_states}
    derived = {value.relative_path: value for value in derived_execution.directives}
    copied = {value.relative_path: value for value in receipt.base_receipt.copied_records}
    if (
        set(expectations) != set(states)
        or set(expectations) != set(derived)
        or set(copied)
        != {path for path, state in states.items() if state.source_sha256 is not None}
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor snapshot/copy/directive universes differ"
        )
    directives: list[ConfirmatoryCheckpointDirective] = []
    for relative_path in sorted(expectations):
        expectation = expectations[relative_path]
        state = states[relative_path]
        execution = derived[relative_path]
        if (
            state.cell_id != expectation.cell_id
            or state.fold_id != expectation.fold_id
            or execution.cell_id != expectation.cell_id
            or execution.fold_id != expectation.fold_id
            or execution.relative_path != relative_path
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor action identity differs from the frozen expectation"
            )
        destination = snapshot.request.successor_run_directory / relative_path
        if state.action == "fresh_fit":
            if (
                execution.action != "fresh_fit"
                or state.source_sha256 is not None
                or state.source_size_bytes is not None
                or os.path.lexists(destination)
            ):
                raise ConfirmatoryCheckpointContractError(
                    "fresh successor action is not derived from one absent checkpoint"
                )
            directives.append(
                _directive_from_expectation(
                    execution_mode="successor_resume",
                    expectation=expectation,
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
        copied_record = copied.get(relative_path)
        if (
            copied_record is None
            or copied_record.sha256 != state.source_sha256
            or copied_record.size_bytes != state.source_size_bytes
            or execution.checkpoint_sha256 != state.source_sha256
            or execution.next_epoch_index != state.completed_epochs
            or execution.maximum_epochs != state.maximum_epochs
            or execution.action != state.action
            or type(state.completed_epochs) is not int
            or type(state.stopped_early) is not bool
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor action was not derived from the copied checkpoint state"
            )
        directives.append(
            _directive_from_expectation(
                execution_mode="successor_resume",
                expectation=expectation,
                action=state.action,
                checkpoint_sha256=state.source_sha256,
                checkpoint_size_bytes=state.source_size_bytes,
                source_predecessor_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=Path(snapshot.request.predecessor_run_directory / relative_path),
                    physical_identity=_physical_identity(copied_record.source_identity),
                    size_bytes=copied_record.size_bytes,
                    sha256=copied_record.sha256,
                ),
                destination_imported_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=destination,
                    physical_identity=_physical_identity(copied_record.destination_identity),
                    size_bytes=copied_record.size_bytes,
                    sha256=copied_record.sha256,
                ),
                completed_epochs_before_fit=state.completed_epochs,
                stopped_early_before_fit=state.stopped_early,
            )
        )
    copy_receipt_sha = canonical_sha256(
        {
            "snapshot_sha256": snapshot.snapshot_sha256,
            "destination_checkpoint_tree_sha256": (receipt.destination_checkpoint_tree_sha256),
            "base_destination_inventory_sha256": (
                receipt.base_receipt.destination_inventory_sha256
            ),
            "resume_evidence_sha256": prepared.resume_evidence_sha256,
        }
    )
    return _build_execution_contract(
        execution_mode="successor_resume",
        retry_of_run_id=snapshot.request.retry_of_run_id,
        directives=tuple(directives),
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=True,
        predecessor_snapshot_sha256=snapshot.snapshot_sha256,
        predecessor_copy_receipt_sha256=copy_receipt_sha,
    )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCanonicalECheckpointIdentity:
    """Exact outcome-blind source/destination identity authorized by E."""

    path: Path
    physical_identity: ConfirmatoryCheckpointPhysicalIdentity
    size_bytes: int
    sha256: str

    @property
    def file_id_128(self) -> str:
        return self.physical_identity.file_id_128

    @property
    def device(self) -> int:
        return self.physical_identity.device

    @property
    def inode(self) -> int:
        return self.physical_identity.inode

    @property
    def mode(self) -> int:
        return self.physical_identity.mode

    @property
    def link_count(self) -> int:
        return self.physical_identity.link_count

    @property
    def modified_time_ns(self) -> int:
        return self.physical_identity.modified_time_ns

    @property
    def changed_time_ns(self) -> int:
        return self.physical_identity.changed_time_ns

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_id_128": self.file_id_128,
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
            "link_count": self.link_count,
            "modified_time_ns": self.modified_time_ns,
            "changed_time_ns": self.changed_time_ns,
            "sha256": self.sha256,
        }

    def validate(self) -> None:
        if (
            not self.path.is_absolute()
            or not isinstance(
                self.physical_identity,
                ConfirmatoryCheckpointPhysicalIdentity,
            )
            or len(self.file_id_128) != 32
            or not set(self.file_id_128).issubset(_HEX)
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or len(self.sha256) != 64
            or not set(self.sha256).issubset(_HEX)
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E checkpoint identity is incomplete or altered"
            )
        self.physical_identity.validate(expected_size_bytes=self.size_bytes)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCanonicalEDirectoryIdentity:
    """Exact lexical identity of the already-qualified predecessor run root."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mode: int
    link_count: int
    modified_time_ns: int
    changed_time_ns: int

    @property
    def file_id_128(self) -> str:
        mask = (1 << 64) - 1
        return f"{self.device & mask:016x}{self.inode & mask:016x}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_id_128": self.file_id_128,
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
            "link_count": self.link_count,
            "modified_time_ns": self.modified_time_ns,
            "changed_time_ns": self.changed_time_ns,
        }

    def validate(self) -> None:
        values = (
            self.device,
            self.inode,
            self.size_bytes,
            self.mode,
            self.link_count,
            self.modified_time_ns,
            self.changed_time_ns,
        )
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or any(type(value) is not int for value in values)
            or self.device < 0
            or self.inode < 0
            or self.size_bytes < 0
            or not stat.S_ISDIR(self.mode)
            or self.link_count < 1
            or len(self.file_id_128) != 32
            or not set(self.file_id_128).issubset(_HEX)
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E predecessor-root identity is incomplete or altered"
            )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatorySuccessorPrecopyFitDirective:
    """One source-bound action authorized before the successor run exists."""

    fit_id: str
    execution_mode: Literal["successor_resume"]
    cell_id: str
    fold_id: int
    action: Literal[
        "fresh_fit",
        "resume_incomplete_fit",
        "restore_terminal_checkpoint_without_fit",
    ]
    source_predecessor_checkpoint: OriginalConfirmatoryCanonicalECheckpointIdentity | None
    destination_checkpoint_relative_path: str
    versioned_checkpoint_output_directory_relative_path: str | None
    checkpoint_execution_manifest_relative_path: str
    completed_epochs_before_fit: int
    stopped_early_before_fit: bool | None
    next_epoch_index: int
    maximum_epochs: int
    expected_configuration: dict[str, Any]
    expected_configuration_sha256: str
    expected_model_metadata: dict[str, Any]
    expected_model_metadata_sha256: str
    expected_data_and_split: dict[str, Any]
    expected_data_and_split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "execution_mode": self.execution_mode,
            "cell_id": self.cell_id,
            "fold_id": self.fold_id,
            "action": self.action,
            "source_predecessor_checkpoint": (
                self.source_predecessor_checkpoint.as_dict()
                if self.source_predecessor_checkpoint is not None
                else None
            ),
            "destination_checkpoint_relative_path": (self.destination_checkpoint_relative_path),
            "versioned_checkpoint_output_directory_relative_path": (
                self.versioned_checkpoint_output_directory_relative_path
            ),
            "checkpoint_execution_manifest_relative_path": (
                self.checkpoint_execution_manifest_relative_path
            ),
            "completed_epochs_before_fit": self.completed_epochs_before_fit,
            "stopped_early_before_fit": self.stopped_early_before_fit,
            "next_epoch_index": self.next_epoch_index,
            "maximum_epochs": self.maximum_epochs,
            "expected_configuration": dict(self.expected_configuration),
            "expected_configuration_sha256": self.expected_configuration_sha256,
            "expected_model_metadata": dict(self.expected_model_metadata),
            "expected_model_metadata_sha256": self.expected_model_metadata_sha256,
            "expected_data_and_split": dict(self.expected_data_and_split),
            "expected_data_and_split_sha256": self.expected_data_and_split_sha256,
        }

    def validate(self) -> None:
        expected_fit_id = f"{self.cell_id}::fold_{self.fold_id:02d}"
        expected_checkpoint = f"cells/{self.cell_id}/checkpoints/fold_{self.fold_id:02d}.pt"
        expected_output = f"cells/{self.cell_id}/checkpoint_versions/fold_{self.fold_id:02d}"
        expected_manifest = (
            f"cells/{self.cell_id}/checkpoint_execution/fold_{self.fold_id:02d}.json"
        )
        if (
            self.fit_id != expected_fit_id
            or self.execution_mode != "successor_resume"
            or not self.cell_id
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT
            or self.action
            not in {
                "fresh_fit",
                "resume_incomplete_fit",
                "restore_terminal_checkpoint_without_fit",
            }
            or self.destination_checkpoint_relative_path != expected_checkpoint
            or self.checkpoint_execution_manifest_relative_path != expected_manifest
            or (
                self.action == "restore_terminal_checkpoint_without_fit"
                and self.versioned_checkpoint_output_directory_relative_path is not None
            )
            or (
                self.action in {"fresh_fit", "resume_incomplete_fit"}
                and self.versioned_checkpoint_output_directory_relative_path != expected_output
            )
            or type(self.completed_epochs_before_fit) is not int
            or type(self.next_epoch_index) is not int
            or type(self.maximum_epochs) is not int
            or self.maximum_epochs <= 0
            or self.next_epoch_index != self.completed_epochs_before_fit
            or self.expected_configuration.get("epochs") != self.maximum_epochs
            or canonical_sha256(self.expected_configuration) != self.expected_configuration_sha256
            or canonical_sha256(self.expected_model_metadata) != self.expected_model_metadata_sha256
            or canonical_sha256(self.expected_data_and_split) != self.expected_data_and_split_sha256
            or set(self.expected_data_and_split)
            != {
                "training_data_sha256",
                "reference_validation_data_sha256",
                "training_split_sha256",
                "reference_validation_split_sha256",
            }
            or any(
                not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX)
                for value in self.expected_data_and_split.values()
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy directive is incomplete or altered"
            )
        source = self.source_predecessor_checkpoint
        if self.action == "fresh_fit":
            if (
                source is not None
                or self.completed_epochs_before_fit != 0
                or self.stopped_early_before_fit is not None
            ):
                raise ConfirmatoryCheckpointContractError(
                    "successor pre-copy fresh fit carries predecessor state"
                )
            return
        if (
            not isinstance(source, OriginalConfirmatoryCanonicalECheckpointIdentity)
            or type(self.stopped_early_before_fit) is not bool
            or not 1 <= self.completed_epochs_before_fit <= self.maximum_epochs
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy resume action lacks exact source state"
            )
        source.validate()
        if stat.S_IMODE(source.mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy source checkpoint remains writable"
            )
        terminal = (
            self.stopped_early_before_fit or self.completed_epochs_before_fit == self.maximum_epochs
        )
        if terminal != (self.action == "restore_terminal_checkpoint_without_fit"):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy action differs from terminal state"
            )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
    """Acyclic E authority: exact sources and future relative destinations only."""

    schema_version: int
    policy: str
    execution_mode: Literal["successor_resume"]
    contract_profile: str
    retry_of_run_id: str
    successor_run_id: str
    predecessor_run_directory: Path
    predecessor_class: Literal[
        "sealed_failed_demoted",
        "unsealed_interrupted_orphan",
    ]
    predecessor_process_id: int | None
    predecessor_process_create_time_unix_us: int | None
    orphan_manual_diagnosis: bool
    resume_authority_binding_sha256: str
    bound_resume_request_sha256: str
    predecessor_qualification: dict[str, Any]
    predecessor_qualification_sha256: str
    predecessor_root_identity: OriginalConfirmatoryCanonicalEDirectoryIdentity
    predecessor_snapshot_sha256: str
    predecessor_checkpoint_tree_sha256: str
    copy_policy: str
    predecessor_checkpoint_read_performed: bool
    predecessor_checkpoint_copy_performed: bool
    outcome_artifacts_read: bool
    automatic_retry_allowed: bool
    predecessor_autodiscovery_allowed: bool
    checkpoint_fallback_allowed: bool
    max_attempt_count: int
    base_template_directives_sha256: str
    source_checkpoint_count: int
    source_checkpoint_total_bytes: int
    source_checkpoint_inventory_sha256: str
    destination_relative_paths_sha256: str
    directives: tuple[OriginalConfirmatorySuccessorPrecopyFitDirective, ...]
    directives_sha256: str
    projection_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "execution_mode": self.execution_mode,
            "contract_profile": self.contract_profile,
            "retry_of_run_id": self.retry_of_run_id,
            "successor_run_id": self.successor_run_id,
            "predecessor_run_directory": str(self.predecessor_run_directory),
            "predecessor_class": self.predecessor_class,
            "predecessor_process_id": self.predecessor_process_id,
            "predecessor_process_create_time_unix_us": (
                self.predecessor_process_create_time_unix_us
            ),
            "orphan_manual_diagnosis": self.orphan_manual_diagnosis,
            "resume_authority_binding_sha256": self.resume_authority_binding_sha256,
            "bound_resume_request_sha256": self.bound_resume_request_sha256,
            "predecessor_qualification": dict(self.predecessor_qualification),
            "predecessor_qualification_sha256": (self.predecessor_qualification_sha256),
            "predecessor_root_identity": self.predecessor_root_identity.as_dict(),
            "predecessor_snapshot_sha256": self.predecessor_snapshot_sha256,
            "predecessor_checkpoint_tree_sha256": (self.predecessor_checkpoint_tree_sha256),
            "copy_policy": self.copy_policy,
            "predecessor_checkpoint_read_performed": (self.predecessor_checkpoint_read_performed),
            "predecessor_checkpoint_copy_performed": (self.predecessor_checkpoint_copy_performed),
            "outcome_artifacts_read": self.outcome_artifacts_read,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "predecessor_autodiscovery_allowed": (self.predecessor_autodiscovery_allowed),
            "checkpoint_fallback_allowed": self.checkpoint_fallback_allowed,
            "max_attempt_count": self.max_attempt_count,
            "base_template_directives_sha256": (self.base_template_directives_sha256),
            "source_checkpoint_count": self.source_checkpoint_count,
            "source_checkpoint_total_bytes": self.source_checkpoint_total_bytes,
            "source_checkpoint_inventory_sha256": (self.source_checkpoint_inventory_sha256),
            "destination_relative_paths_sha256": (self.destination_relative_paths_sha256),
            "directives": [value.as_dict() for value in self.directives],
            "directives_sha256": self.directives_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "projection_sha256": self.projection_sha256,
        }

    def validate(self) -> None:
        if type(self) is not OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
            raise TypeError("successor pre-copy authority requires its exact type")
        self.predecessor_root_identity.validate()
        qualification = dict(self.predecessor_qualification)
        qualification_fields = {
            "predecessor_class",
            "policy",
            "terminal_status",
            "sealed_integrity_valid",
            "integrity_registry_record_present",
            "artifact_root_sha256",
            "artifact_manifest_sha256",
            "immutable_marker_sha256",
            "status_sha256",
            "completion_evidence_sha256",
            "process_receipt",
            "active_lock_paths_before",
            "active_lock_paths_after",
            "stage_attestation_count",
            "disposition_record_count",
            "root_inventory_before_sha256",
            "root_inventory_after_sha256",
            "qualification_sha256",
        }
        process_fields = {
            "expected_process_id",
            "expected_create_time_unix_us",
            "expected_identity_state",
            "inspector_process_id",
            "inspector_create_time_unix_us",
            "exact_inspector_match_excluded",
            "exact_authorized_supervisor_match_excluded",
            "matching_predecessor_process_ids",
            "inspected_process_count",
            "receipt_sha256",
        }
        process = qualification.get("process_receipt")
        if (
            set(qualification) != qualification_fields
            or not isinstance(process, Mapping)
            or set(process) != process_fields
            or qualification.get("predecessor_class") != self.predecessor_class
            or qualification.get("active_lock_paths_before") != []
            or qualification.get("active_lock_paths_after") != []
            or qualification.get("stage_attestation_count") != 0
            or qualification.get("disposition_record_count") != 0
            or not _valid_sha256(qualification.get("qualification_sha256"))
            or canonical_sha256(qualification) != self.predecessor_qualification_sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy predecessor qualification is incomplete or altered"
            )
        if self.predecessor_class == "sealed_failed_demoted":
            qualifying_terminal = (
                qualification.get("terminal_status") == "failed"
                and qualification.get("sealed_integrity_valid") is True
                and qualification.get("integrity_registry_record_present") is True
                and all(
                    _valid_sha256(qualification.get(field))
                    for field in (
                        "artifact_root_sha256",
                        "artifact_manifest_sha256",
                        "immutable_marker_sha256",
                        "status_sha256",
                        "completion_evidence_sha256",
                    )
                )
                and self.predecessor_process_id is None
                and self.predecessor_process_create_time_unix_us is None
                and self.orphan_manual_diagnosis is False
            )
        else:
            qualifying_terminal = (
                qualification.get("terminal_status") == "unsealed_interrupted_orphan"
                and qualification.get("sealed_integrity_valid") is False
                and qualification.get("integrity_registry_record_present") is False
                and type(self.predecessor_process_id) is int
                and self.predecessor_process_id > 0
                and type(self.predecessor_process_create_time_unix_us) is int
                and self.predecessor_process_create_time_unix_us > 0
                and self.orphan_manual_diagnosis is True
            )
        for directive in self.directives:
            if type(directive) is not OriginalConfirmatorySuccessorPrecopyFitDirective:
                raise ConfirmatoryCheckpointContractError(
                    "successor pre-copy projection contains an untyped directive"
                )
            directive.validate()
        source_records = [
            {
                "fit_id": value.fit_id,
                "source_predecessor_checkpoint": (value.source_predecessor_checkpoint.as_dict()),
            }
            for value in self.directives
            if value.source_predecessor_checkpoint is not None
        ]
        sources = [
            value.source_predecessor_checkpoint
            for value in self.directives
            if value.source_predecessor_checkpoint is not None
        ]
        destination_paths = [
            value.destination_checkpoint_relative_path for value in self.directives
        ]
        if (
            not qualifying_terminal
            or self.schema_version != 1
            or self.policy != ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY
            or self.execution_mode != "successor_resume"
            or self.contract_profile != "original_confirmatory_exact_180"
            or not self.retry_of_run_id
            or Path(self.retry_of_run_id).name != self.retry_of_run_id
            or not self.successor_run_id
            or Path(self.successor_run_id).name != self.successor_run_id
            or self.retry_of_run_id.casefold() == self.successor_run_id.casefold()
            or self.predecessor_run_directory.name != self.retry_of_run_id
            or self.predecessor_root_identity.path != self.predecessor_run_directory
            or any(
                not _valid_sha256(value)
                for value in (
                    self.resume_authority_binding_sha256,
                    self.bound_resume_request_sha256,
                    self.predecessor_qualification_sha256,
                    self.predecessor_snapshot_sha256,
                    self.predecessor_checkpoint_tree_sha256,
                    self.base_template_directives_sha256,
                    self.source_checkpoint_inventory_sha256,
                    self.destination_relative_paths_sha256,
                    self.directives_sha256,
                    self.projection_sha256,
                )
            )
            or self.copy_policy != ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY
            or self.predecessor_checkpoint_read_performed is not True
            or self.predecessor_checkpoint_copy_performed is not False
            or self.outcome_artifacts_read is not False
            or self.automatic_retry_allowed is not False
            or self.predecessor_autodiscovery_allowed is not False
            or self.checkpoint_fallback_allowed is not False
            or self.max_attempt_count != 1
            or len(self.directives) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or tuple(sorted(self.directives, key=lambda item: (item.cell_id, item.fold_id)))
            != self.directives
            or len({(value.cell_id, value.fold_id) for value in self.directives})
            != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or len(set(destination_paths)) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or len({value.casefold() for value in destination_paths})
            != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or self.source_checkpoint_count != len(sources)
            or self.source_checkpoint_count <= 0
            or self.source_checkpoint_total_bytes != sum(value.size_bytes for value in sources)
            or len({value.file_id_128 for value in sources}) != len(sources)
            or any(
                value.path.parent.parent.parent.parent != self.predecessor_run_directory
                for value in sources
            )
            or canonical_sha256(source_records) != self.source_checkpoint_inventory_sha256
            or canonical_sha256(destination_paths) != self.destination_relative_paths_sha256
            or canonical_sha256([value.as_dict() for value in self.directives])
            != self.directives_sha256
            or canonical_sha256(self.payload_without_self_hash()) != self.projection_sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy checkpoint projection is incomplete or altered"
            )


type OriginalConfirmatoryScientificCheckpointAuthority = (
    OriginalConfirmatoryCanonicalECheckpointProjection
    | OriginalConfirmatorySuccessorPrecopyCheckpointProjection
)


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCanonicalEFitDirective:
    """One lossless canonical-E authorization record for an exact CNN fold fit."""

    fit_id: str
    execution_mode: OriginalConfirmatoryExecutionMode
    cell_id: str
    fold_id: int
    action: Literal[
        "fresh_fit",
        "resume_incomplete_fit",
        "restore_terminal_checkpoint_without_fit",
    ]
    source_predecessor_checkpoint: OriginalConfirmatoryCanonicalECheckpointIdentity | None
    destination_imported_checkpoint: OriginalConfirmatoryCanonicalECheckpointIdentity | None
    versioned_checkpoint_output_directory_relative_path: str | None
    checkpoint_execution_manifest_relative_path: str
    completed_epochs_before_fit: int
    stopped_early_before_fit: bool | None
    next_epoch_index: int
    maximum_epochs: int
    expected_configuration: dict[str, Any]
    expected_configuration_sha256: str
    expected_model_metadata: dict[str, Any]
    expected_model_metadata_sha256: str
    expected_data_and_split: dict[str, Any]
    expected_data_and_split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "execution_mode": self.execution_mode,
            "cell_id": self.cell_id,
            "fold_id": self.fold_id,
            "action": self.action,
            "source_predecessor_checkpoint": (
                self.source_predecessor_checkpoint.as_dict()
                if self.source_predecessor_checkpoint is not None
                else None
            ),
            "destination_imported_checkpoint": (
                self.destination_imported_checkpoint.as_dict()
                if self.destination_imported_checkpoint is not None
                else None
            ),
            "versioned_checkpoint_output_directory_relative_path": (
                self.versioned_checkpoint_output_directory_relative_path
            ),
            "checkpoint_execution_manifest_relative_path": (
                self.checkpoint_execution_manifest_relative_path
            ),
            "completed_epochs_before_fit": self.completed_epochs_before_fit,
            "stopped_early_before_fit": self.stopped_early_before_fit,
            "next_epoch_index": self.next_epoch_index,
            "maximum_epochs": self.maximum_epochs,
            "expected_configuration": dict(self.expected_configuration),
            "expected_configuration_sha256": self.expected_configuration_sha256,
            "expected_model_metadata": dict(self.expected_model_metadata),
            "expected_model_metadata_sha256": self.expected_model_metadata_sha256,
            "expected_data_and_split": dict(self.expected_data_and_split),
            "expected_data_and_split_sha256": (self.expected_data_and_split_sha256),
        }

    def validate(self) -> None:
        expected_fit_id = f"{self.cell_id}::fold_{self.fold_id:02d}"
        expected_output_directory = (
            f"cells/{self.cell_id}/checkpoint_versions/fold_{self.fold_id:02d}"
        )
        expected_execution_manifest = (
            f"cells/{self.cell_id}/checkpoint_execution/fold_{self.fold_id:02d}.json"
        )
        if (
            self.fit_id != expected_fit_id
            or self.execution_mode not in {"fresh", "successor_resume"}
            or not self.cell_id
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < ORIGINAL_CONFIRMATORY_OOF_FOLD_COUNT
            or self.action
            not in {
                "fresh_fit",
                "resume_incomplete_fit",
                "restore_terminal_checkpoint_without_fit",
            }
            or self.checkpoint_execution_manifest_relative_path != expected_execution_manifest
            or (
                self.action == "restore_terminal_checkpoint_without_fit"
                and self.versioned_checkpoint_output_directory_relative_path is not None
            )
            or (
                self.action in {"fresh_fit", "resume_incomplete_fit"}
                and self.versioned_checkpoint_output_directory_relative_path
                != expected_output_directory
            )
            or type(self.completed_epochs_before_fit) is not int
            or type(self.next_epoch_index) is not int
            or type(self.maximum_epochs) is not int
            or self.maximum_epochs <= 0
            or self.next_epoch_index != self.completed_epochs_before_fit
            or self.expected_configuration.get("epochs") != self.maximum_epochs
            or canonical_sha256(self.expected_configuration) != self.expected_configuration_sha256
            or canonical_sha256(self.expected_model_metadata) != self.expected_model_metadata_sha256
            or canonical_sha256(self.expected_data_and_split) != self.expected_data_and_split_sha256
            or set(self.expected_data_and_split)
            != {
                "training_data_sha256",
                "reference_validation_data_sha256",
                "training_split_sha256",
                "reference_validation_split_sha256",
            }
            or any(
                not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX)
                for value in self.expected_data_and_split.values()
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E fit directive is incomplete or altered"
            )
        if self.action == "fresh_fit":
            if (
                self.source_predecessor_checkpoint is not None
                or self.destination_imported_checkpoint is not None
                or self.completed_epochs_before_fit != 0
                or self.stopped_early_before_fit is not None
            ):
                raise ConfirmatoryCheckpointContractError(
                    "canonical E fresh fit carries predecessor state"
                )
            return
        if (
            self.execution_mode != "successor_resume"
            or not isinstance(
                self.source_predecessor_checkpoint,
                OriginalConfirmatoryCanonicalECheckpointIdentity,
            )
            or not isinstance(
                self.destination_imported_checkpoint,
                OriginalConfirmatoryCanonicalECheckpointIdentity,
            )
            or type(self.stopped_early_before_fit) is not bool
            or not 1 <= self.completed_epochs_before_fit <= self.maximum_epochs
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E resume fit lacks exact predecessor state"
            )
        self.source_predecessor_checkpoint.validate()
        self.destination_imported_checkpoint.validate()
        if (
            os.path.normcase(str(self.source_predecessor_checkpoint.path))
            == os.path.normcase(str(self.destination_imported_checkpoint.path))
            or self.source_predecessor_checkpoint.file_id_128
            == self.destination_imported_checkpoint.file_id_128
            or self.source_predecessor_checkpoint.size_bytes
            != self.destination_imported_checkpoint.size_bytes
            or self.source_predecessor_checkpoint.sha256
            != self.destination_imported_checkpoint.sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E source/destination identities are aliased or mismatched"
            )
        terminal = (
            self.stopped_early_before_fit or self.completed_epochs_before_fit == self.maximum_epochs
        )
        if terminal != (self.action == "restore_terminal_checkpoint_without_fit"):
            raise ConfirmatoryCheckpointContractError(
                "canonical E resume action differs from terminal state"
            )


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryCanonicalECheckpointProjection:
    """Lossless typed E projection of one registered exact-180 contract."""

    schema_version: int
    policy: str
    execution_mode: str
    contract_profile: str
    retry_of_run_id: str | None
    predecessor_checkpoint_read_performed: bool
    predecessor_checkpoint_copy_performed: bool
    outcome_artifacts_read: bool
    automatic_retry_allowed: bool
    predecessor_snapshot_sha256: str | None
    predecessor_copy_receipt_sha256: str | None
    directives: tuple[OriginalConfirmatoryCanonicalEFitDirective, ...]
    directives_sha256: str
    checkpoint_contract_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "execution_mode": self.execution_mode,
            "contract_profile": self.contract_profile,
            "retry_of_run_id": self.retry_of_run_id,
            "predecessor_checkpoint_read_performed": (self.predecessor_checkpoint_read_performed),
            "predecessor_checkpoint_copy_performed": (self.predecessor_checkpoint_copy_performed),
            "outcome_artifacts_read": self.outcome_artifacts_read,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "predecessor_snapshot_sha256": self.predecessor_snapshot_sha256,
            "predecessor_copy_receipt_sha256": (self.predecessor_copy_receipt_sha256),
            "directives": [value.as_dict() for value in self.directives],
            "directives_sha256": self.directives_sha256,
            "checkpoint_contract_sha256": self.checkpoint_contract_sha256,
        }

    def validate(self) -> None:
        for directive in self.directives:
            if not isinstance(
                directive,
                OriginalConfirmatoryCanonicalEFitDirective,
            ):
                raise ConfirmatoryCheckpointContractError(
                    "canonical E checkpoint projection contains an untyped directive"
                )
            directive.validate()
        if (
            self.schema_version != 1
            or self.policy != ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY
            or self.execution_mode not in {"fresh", "successor_resume"}
            or self.contract_profile != "original_confirmatory_exact_180"
            or self.outcome_artifacts_read
            or self.automatic_retry_allowed
            or len(self.directives) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or len({(value.cell_id, value.fold_id) for value in self.directives})
            != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or len(
                {
                    value.checkpoint_execution_manifest_relative_path.casefold()
                    for value in self.directives
                }
            )
            != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or len(
                {
                    value.versioned_checkpoint_output_directory_relative_path.casefold()
                    for value in self.directives
                    if value.versioned_checkpoint_output_directory_relative_path is not None
                }
            )
            != sum(
                value.versioned_checkpoint_output_directory_relative_path is not None
                for value in self.directives
            )
            or any(value.execution_mode != self.execution_mode for value in self.directives)
            or canonical_sha256([value.as_dict() for value in self.directives])
            != self.directives_sha256
            or not isinstance(self.checkpoint_contract_sha256, str)
            or len(self.checkpoint_contract_sha256) != 64
            or not set(self.checkpoint_contract_sha256).issubset(_HEX)
        ):
            raise ConfirmatoryCheckpointContractError(
                "canonical E checkpoint projection is incomplete or altered"
            )
        if self.execution_mode == "fresh":
            if (
                self.retry_of_run_id is not None
                or self.predecessor_checkpoint_read_performed
                or self.predecessor_checkpoint_copy_performed
                or self.predecessor_snapshot_sha256 is not None
                or self.predecessor_copy_receipt_sha256 is not None
                or any(value.action != "fresh_fit" for value in self.directives)
            ):
                raise ConfirmatoryCheckpointContractError(
                    "fresh canonical E projection carries predecessor state"
                )
        elif (
            self.retry_of_run_id is None
            or not self.predecessor_checkpoint_read_performed
            or not self.predecessor_checkpoint_copy_performed
            or self.predecessor_snapshot_sha256 is None
            or self.predecessor_copy_receipt_sha256 is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor canonical E projection lacks exact lineage"
            )


def build_original_confirmatory_canonical_e_checkpoint_projection(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> OriginalConfirmatoryCanonicalECheckpointProjection:
    """Build canonical E's lossless typed authorization from a registered contract."""

    contract.validate(expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT)
    cell_ids = tuple(sorted({value.cell_id for value in contract.directives}))
    if len(cell_ids) != ORIGINAL_CONFIRMATORY_CNN_CELL_COUNT:
        raise ConfirmatoryCheckpointContractError(
            "fit-directive authority requires exactly 36 CNN cells"
        )
    records: list[OriginalConfirmatoryCanonicalEFitDirective] = []
    for directive in contract.directives:
        records.append(
            OriginalConfirmatoryCanonicalEFitDirective(
                fit_id=f"{directive.cell_id}::fold_{directive.fold_id:02d}",
                execution_mode=directive.execution_mode,
                cell_id=directive.cell_id,
                fold_id=directive.fold_id,
                action=directive.action,
                source_predecessor_checkpoint=(
                    OriginalConfirmatoryCanonicalECheckpointIdentity(
                        path=directive.source_predecessor_checkpoint.path,
                        physical_identity=(
                            directive.source_predecessor_checkpoint.physical_identity
                        ),
                        size_bytes=directive.source_predecessor_checkpoint.size_bytes,
                        sha256=directive.source_predecessor_checkpoint.sha256,
                    )
                    if directive.source_predecessor_checkpoint is not None
                    else None
                ),
                destination_imported_checkpoint=(
                    OriginalConfirmatoryCanonicalECheckpointIdentity(
                        path=directive.destination_imported_checkpoint.path,
                        physical_identity=(
                            directive.destination_imported_checkpoint.physical_identity
                        ),
                        size_bytes=directive.destination_imported_checkpoint.size_bytes,
                        sha256=directive.destination_imported_checkpoint.sha256,
                    )
                    if directive.destination_imported_checkpoint is not None
                    else None
                ),
                versioned_checkpoint_output_directory_relative_path=(
                    directive.versioned_checkpoint_output_directory_relative_path
                ),
                checkpoint_execution_manifest_relative_path=(
                    directive.checkpoint_execution_manifest_relative_path
                ),
                completed_epochs_before_fit=(directive.completed_epochs_before_fit),
                stopped_early_before_fit=directive.stopped_early_before_fit,
                next_epoch_index=directive.next_epoch_index,
                maximum_epochs=directive.maximum_epochs,
                expected_configuration=json.loads(directive.expected_configuration_json),
                expected_configuration_sha256=(directive.expected_configuration_sha256),
                expected_model_metadata=json.loads(directive.expected_model_metadata_json),
                expected_model_metadata_sha256=(directive.expected_model_metadata_sha256),
                expected_data_and_split=json.loads(directive.expected_data_and_split_json),
                expected_data_and_split_sha256=(directive.expected_data_and_split_sha256),
            )
        )
    output = tuple(records)
    if len(output) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT:
        raise ConfirmatoryCheckpointContractError(
            "fit-directive authority record count is not exact"
        )
    directives_sha256 = canonical_sha256([value.as_dict() for value in output])
    contract_payload = {
        "checkpoint_execution_contract": contract.as_dict(),
        "canonical_e_directives": [value.as_dict() for value in output],
        "canonical_e_directives_sha256": directives_sha256,
    }
    projection = OriginalConfirmatoryCanonicalECheckpointProjection(
        schema_version=1,
        policy=ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY,
        execution_mode=contract.execution_mode,
        contract_profile=contract.contract_profile,
        retry_of_run_id=contract.retry_of_run_id,
        predecessor_checkpoint_read_performed=(contract.predecessor_checkpoint_read_performed),
        predecessor_checkpoint_copy_performed=(contract.predecessor_checkpoint_copy_performed),
        outcome_artifacts_read=contract.outcome_artifacts_read,
        automatic_retry_allowed=contract.automatic_retry_allowed,
        predecessor_snapshot_sha256=contract.predecessor_snapshot_sha256,
        predecessor_copy_receipt_sha256=(contract.predecessor_copy_receipt_sha256),
        directives=output,
        directives_sha256=directives_sha256,
        checkpoint_contract_sha256=canonical_sha256(contract_payload),
    )
    projection.validate()
    return projection


def _fresh_directive_from_precopy(
    value: OriginalConfirmatorySuccessorPrecopyFitDirective,
) -> ConfirmatoryCheckpointDirective:
    return ConfirmatoryCheckpointDirective(
        execution_mode="fresh",
        cell_id=value.cell_id,
        fold_id=value.fold_id,
        action="fresh_fit",
        source_predecessor_checkpoint=None,
        destination_imported_checkpoint=None,
        versioned_checkpoint_output_directory_relative_path=(
            f"cells/{value.cell_id}/checkpoint_versions/fold_{value.fold_id:02d}"
        ),
        checkpoint_execution_manifest_relative_path=(
            value.checkpoint_execution_manifest_relative_path
        ),
        checkpoint_sha256=None,
        checkpoint_size_bytes=None,
        completed_epochs_before_fit=0,
        stopped_early_before_fit=None,
        next_epoch_index=0,
        maximum_epochs=value.maximum_epochs,
        expected_configuration_json=_canonical_json(value.expected_configuration),
        expected_configuration_sha256=value.expected_configuration_sha256,
        expected_model_metadata_json=_canonical_json(value.expected_model_metadata),
        expected_model_metadata_sha256=value.expected_model_metadata_sha256,
        expected_data_and_split_json=_canonical_json(value.expected_data_and_split),
        expected_data_and_split_sha256=value.expected_data_and_split_sha256,
    )


def _fresh_contract_from_precopy(
    projection: OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
) -> ConfirmatoryCheckpointExecutionContract:
    projection.validate()
    directives = tuple(_fresh_directive_from_precopy(value) for value in projection.directives)
    contract = _build_execution_contract(
        execution_mode="fresh",
        retry_of_run_id=None,
        directives=directives,
        predecessor_checkpoint_read_performed=False,
        predecessor_checkpoint_copy_performed=False,
        predecessor_snapshot_sha256=None,
        predecessor_copy_receipt_sha256=None,
    )
    if contract.directives_sha256 != projection.base_template_directives_sha256:
        raise ConfirmatoryCheckpointContractError(
            "successor pre-copy authority differs from its frozen fresh template"
        )
    return contract


def build_original_confirmatory_successor_precopy_checkpoint_projection(
    snapshot: OriginalConfirmatoryPredecessorSnapshot,
) -> OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
    """Project one qualified predecessor before RunTracker creation or copy."""

    if type(snapshot) is not OriginalConfirmatoryPredecessorSnapshot:
        raise TypeError("successor pre-copy authority requires the exact typed snapshot")
    _require_exact_original_contract(snapshot.contract)
    request = snapshot.request
    states = {value.relative_path: value for value in snapshot.checkpoint_states}
    records = {value.relative_path: value for value in snapshot.base_snapshot.records}
    expectations = {
        value.relative_path: value for value in snapshot.contract.checkpoint_expectations
    }
    if (
        set(states) != set(expectations)
        or set(records) != set(expectations)
        or snapshot.checkpoint_tree_before_sha256 != snapshot.checkpoint_tree_after_sha256
        or snapshot.base_snapshot.predecessor_directory != request.predecessor_run_directory
        or snapshot.base_snapshot.retry_of_run_id != request.retry_of_run_id
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor pre-copy snapshot universes or roots differ"
        )
    directives: list[OriginalConfirmatorySuccessorPrecopyFitDirective] = []
    for relative_path in sorted(expectations):
        expectation = expectations[relative_path]
        state = states[relative_path]
        record = records[relative_path]
        if (
            state.cell_id != expectation.cell_id
            or state.fold_id != expectation.fold_id
            or record.cell_id != expectation.cell_id
            or record.fold_id != expectation.fold_id
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor pre-copy checkpoint identity differs from its frozen expectation"
            )
        source: OriginalConfirmatoryCanonicalECheckpointIdentity | None = None
        if state.action == "fresh_fit":
            if (
                record.decision != "missing_fresh"
                or record.identity is not None
                or state.source_sha256 is not None
            ):
                raise ConfirmatoryCheckpointContractError(
                    "successor pre-copy fresh action differs from the qualified snapshot"
                )
            completed_epochs = 0
        else:
            if (
                record.decision != "resume"
                or record.identity is None
                or record.size_bytes is None
                or record.sha256 is None
                or state.source_sha256 != record.sha256
                or state.source_size_bytes != record.size_bytes
                or state.completed_epochs is None
            ):
                raise ConfirmatoryCheckpointContractError(
                    "successor pre-copy resume action lacks its qualified source"
                )
            source = OriginalConfirmatoryCanonicalECheckpointIdentity(
                path=request.predecessor_run_directory / relative_path,
                physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
                    device=record.identity.device,
                    inode=record.identity.inode,
                    size_bytes=record.identity.size_bytes,
                    mode=record.identity.mode,
                    link_count=record.identity.link_count,
                    modified_time_ns=record.identity.modified_time_ns,
                    changed_time_ns=record.identity.changed_time_ns,
                ),
                size_bytes=record.size_bytes,
                sha256=record.sha256,
            )
            completed_epochs = state.completed_epochs
        configuration = dict(expectation.expected_configuration)
        metadata = dict(expectation.expected_model_metadata)
        data_and_split = dict(expectation.expected_data_and_split_sha256)
        directives.append(
            OriginalConfirmatorySuccessorPrecopyFitDirective(
                fit_id=f"{expectation.cell_id}::fold_{expectation.fold_id:02d}",
                execution_mode="successor_resume",
                cell_id=expectation.cell_id,
                fold_id=expectation.fold_id,
                action=state.action,
                source_predecessor_checkpoint=source,
                destination_checkpoint_relative_path=relative_path,
                versioned_checkpoint_output_directory_relative_path=(
                    None
                    if state.action == "restore_terminal_checkpoint_without_fit"
                    else (
                        f"cells/{expectation.cell_id}/checkpoint_versions/"
                        f"fold_{expectation.fold_id:02d}"
                    )
                ),
                checkpoint_execution_manifest_relative_path=(
                    f"cells/{expectation.cell_id}/checkpoint_execution/"
                    f"fold_{expectation.fold_id:02d}.json"
                ),
                completed_epochs_before_fit=completed_epochs,
                stopped_early_before_fit=state.stopped_early,
                next_epoch_index=completed_epochs,
                maximum_epochs=state.maximum_epochs,
                expected_configuration=configuration,
                expected_configuration_sha256=canonical_sha256(configuration),
                expected_model_metadata=metadata,
                expected_model_metadata_sha256=canonical_sha256(metadata),
                expected_data_and_split=data_and_split,
                expected_data_and_split_sha256=canonical_sha256(data_and_split),
            )
        )
    output = tuple(directives)
    base_contract = build_original_confirmatory_fresh_checkpoint_execution_contract(
        snapshot.contract
    )
    root = snapshot.base_snapshot.predecessor_root_identity
    source_records = [
        {
            "fit_id": value.fit_id,
            "source_predecessor_checkpoint": value.source_predecessor_checkpoint.as_dict(),
        }
        for value in output
        if value.source_predecessor_checkpoint is not None
    ]
    destination_paths = [value.destination_checkpoint_relative_path for value in output]
    qualification = snapshot.qualification.as_dict()
    provisional = OriginalConfirmatorySuccessorPrecopyCheckpointProjection(
        schema_version=1,
        policy=ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY,
        execution_mode="successor_resume",
        contract_profile="original_confirmatory_exact_180",
        retry_of_run_id=request.retry_of_run_id,
        successor_run_id=request.successor_run_id,
        predecessor_run_directory=request.predecessor_run_directory,
        predecessor_class=request.predecessor_class,
        predecessor_process_id=request.predecessor_process_id,
        predecessor_process_create_time_unix_us=(request.predecessor_process_create_time_unix_us),
        orphan_manual_diagnosis=request.orphan_manual_diagnosis,
        resume_authority_binding_sha256=request.authority.authorization_binding_sha256,
        bound_resume_request_sha256=request.request_sha256,
        predecessor_qualification=qualification,
        predecessor_qualification_sha256=canonical_sha256(qualification),
        predecessor_root_identity=OriginalConfirmatoryCanonicalEDirectoryIdentity(
            path=request.predecessor_run_directory,
            device=root.device,
            inode=root.inode,
            size_bytes=root.size_bytes,
            mode=root.mode,
            link_count=root.link_count,
            modified_time_ns=root.modified_time_ns,
            changed_time_ns=root.changed_time_ns,
        ),
        predecessor_snapshot_sha256=snapshot.snapshot_sha256,
        predecessor_checkpoint_tree_sha256=snapshot.checkpoint_tree_after_sha256,
        copy_policy=ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY,
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=False,
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        predecessor_autodiscovery_allowed=False,
        checkpoint_fallback_allowed=False,
        max_attempt_count=1,
        base_template_directives_sha256=base_contract.directives_sha256,
        source_checkpoint_count=len(source_records),
        source_checkpoint_total_bytes=sum(
            value.source_predecessor_checkpoint.size_bytes
            for value in output
            if value.source_predecessor_checkpoint is not None
        ),
        source_checkpoint_inventory_sha256=canonical_sha256(source_records),
        destination_relative_paths_sha256=canonical_sha256(destination_paths),
        directives=output,
        directives_sha256=canonical_sha256([value.as_dict() for value in output]),
        projection_sha256="0" * 64,
    )
    projection = replace(
        provisional,
        projection_sha256=canonical_sha256(provisional.payload_without_self_hash()),
    )
    projection.validate()
    return projection


def _directory_identity_from_path(
    path: Path,
    *,
    role: str,
) -> OriginalConfirmatoryCanonicalEDirectoryIdentity:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(f"{role} is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _is_reparse(observed)
        or path.is_symlink()
        or int(observed.st_nlink) < 1
        or _named_streams(path)
    ):
        raise ConfirmatoryCheckpointContractError(
            f"{role} is linked, reparse, streamed, or not a directory"
        )
    identity = OriginalConfirmatoryCanonicalEDirectoryIdentity(
        path=path,
        device=int(observed.st_dev),
        inode=int(observed.st_ino),
        size_bytes=int(observed.st_size),
        mode=int(observed.st_mode),
        link_count=int(observed.st_nlink),
        modified_time_ns=int(observed.st_mtime_ns),
        changed_time_ns=int(observed.st_ctime_ns),
    )
    identity.validate()
    return identity


@dataclass(frozen=True, slots=True)
class OriginalConfirmatorySuccessorCopyReceipt:
    """Post-E receipt for one exact O_EXCL checkpoint import."""

    schema_version: int
    policy: str
    scientific_authority_projection_sha256: str
    retry_of_run_id: str
    run_id: str
    predecessor_run_directory: Path
    destination_run_directory: Path
    copy_policy: str
    predecessor_root_identity_before: OriginalConfirmatoryCanonicalEDirectoryIdentity
    predecessor_root_identity_after: OriginalConfirmatoryCanonicalEDirectoryIdentity
    copied_checkpoints: tuple[ConfirmatoryCheckpointFileIdentity, ...]
    fresh_checkpoint_relative_paths: tuple[str, ...]
    copied_checkpoint_count: int
    copied_total_bytes: int
    destination_inventory_sha256: str
    predecessor_source_inventory_sha256: str
    predecessor_unchanged: bool
    outcome_artifacts_read: bool
    automatic_retry_allowed: bool
    checkpoint_fallback_used: bool
    receipt_sha256: str

    def payload_without_self_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "scientific_authority_projection_sha256": (self.scientific_authority_projection_sha256),
            "retry_of_run_id": self.retry_of_run_id,
            "run_id": self.run_id,
            "predecessor_run_directory": str(self.predecessor_run_directory),
            "destination_run_directory": str(self.destination_run_directory),
            "copy_policy": self.copy_policy,
            "predecessor_root_identity_before": (self.predecessor_root_identity_before.as_dict()),
            "predecessor_root_identity_after": (self.predecessor_root_identity_after.as_dict()),
            "copied_checkpoints": [value.as_dict() for value in self.copied_checkpoints],
            "fresh_checkpoint_relative_paths": list(self.fresh_checkpoint_relative_paths),
            "copied_checkpoint_count": self.copied_checkpoint_count,
            "copied_total_bytes": self.copied_total_bytes,
            "destination_inventory_sha256": self.destination_inventory_sha256,
            "predecessor_source_inventory_sha256": (self.predecessor_source_inventory_sha256),
            "predecessor_unchanged": self.predecessor_unchanged,
            "outcome_artifacts_read": self.outcome_artifacts_read,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "checkpoint_fallback_used": self.checkpoint_fallback_used,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload_without_self_hash(),
            "receipt_sha256": self.receipt_sha256,
        }

    def validate(self) -> None:
        if (
            type(self) is not OriginalConfirmatorySuccessorCopyReceipt
            or type(self.predecessor_root_identity_before)
            is not OriginalConfirmatoryCanonicalEDirectoryIdentity
            or type(self.predecessor_root_identity_after)
            is not OriginalConfirmatoryCanonicalEDirectoryIdentity
            or type(self.copied_checkpoints) is not tuple
            or type(self.fresh_checkpoint_relative_paths) is not tuple
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor post-E copy receipt is not its exact typed record"
            )
        self.predecessor_root_identity_before.validate()
        self.predecessor_root_identity_after.validate()
        copied_relative_paths: list[str] = []
        for value in self.copied_checkpoints:
            if type(value) is not ConfirmatoryCheckpointFileIdentity:
                raise ConfirmatoryCheckpointContractError(
                    "successor copy receipt contains an untyped destination"
                )
            value.validate()
            try:
                relative = value.path.relative_to(self.destination_run_directory)
            except ValueError as exc:
                raise ConfirmatoryCheckpointContractError(
                    "successor copy receipt destination escapes its exact run"
                ) from exc
            relative_text = relative.as_posix()
            if value.path != self.destination_run_directory / relative_text:
                raise ConfirmatoryCheckpointContractError(
                    "successor copy receipt destination is not lexically canonical"
                )
            copied_relative_paths.append(relative_text)
        all_relative_paths = [
            *copied_relative_paths,
            *self.fresh_checkpoint_relative_paths,
        ]
        canonical_checkpoint_paths = all(
            type(value) is str
            and value == Path(value).as_posix()
            and not Path(value).is_absolute()
            and len(Path(value).parts) == 4
            and Path(value).parts[0] == "cells"
            and Path(value).parts[2] == "checkpoints"
            and Path(value).parts[3].startswith("fold_")
            and Path(value).parts[3].endswith(".pt")
            and Path(value).parts[3][5:-3].isdigit()
            and len(Path(value).parts[3][5:-3]) == 2
            for value in all_relative_paths
        )
        inventory = [value.as_dict() for value in self.copied_checkpoints]
        if (
            self.schema_version != 1
            or self.policy != ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_RECEIPT_POLICY
            or not _valid_sha256(self.scientific_authority_projection_sha256)
            or not self.retry_of_run_id
            or not self.run_id
            or Path(self.retry_of_run_id).name != self.retry_of_run_id
            or Path(self.run_id).name != self.run_id
            or self.retry_of_run_id.casefold() == self.run_id.casefold()
            or not self.predecessor_run_directory.is_absolute()
            or not self.destination_run_directory.is_absolute()
            or self.predecessor_run_directory == self.destination_run_directory
            or self.predecessor_run_directory.name != self.retry_of_run_id
            or self.destination_run_directory.name != self.run_id
            or self.copy_policy != ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY
            or self.predecessor_root_identity_before != self.predecessor_root_identity_after
            or self.predecessor_root_identity_before.path != self.predecessor_run_directory
            or self.copied_checkpoint_count != len(self.copied_checkpoints)
            or self.copied_checkpoint_count <= 0
            or len(all_relative_paths) != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or not canonical_checkpoint_paths
            or len({value.casefold() for value in all_relative_paths})
            != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
            or self.copied_total_bytes != sum(value.size_bytes for value in self.copied_checkpoints)
            or len({os.path.normcase(str(value.path)) for value in self.copied_checkpoints})
            != len(self.copied_checkpoints)
            or len({value.file_id_128 for value in self.copied_checkpoints})
            != len(self.copied_checkpoints)
            or canonical_sha256(inventory) != self.destination_inventory_sha256
            or not _valid_sha256(self.predecessor_source_inventory_sha256)
            or self.predecessor_unchanged is not True
            or self.outcome_artifacts_read is not False
            or self.automatic_retry_allowed is not False
            or self.checkpoint_fallback_used is not False
            or canonical_sha256(self.payload_without_self_hash()) != self.receipt_sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor post-E copy receipt is incomplete or altered"
            )


@dataclass(frozen=True, slots=True)
class MaterializedOriginalConfirmatorySuccessor:
    """Exact post-copy refinement consumed once by the common matrix lifecycle."""

    execution_contract: ConfirmatoryCheckpointExecutionContract
    checkpoint_projection: OriginalConfirmatoryCanonicalECheckpointProjection
    copy_receipt: OriginalConfirmatorySuccessorCopyReceipt

    def validate(
        self,
        authority: OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
    ) -> None:
        if type(self) is not MaterializedOriginalConfirmatorySuccessor:
            raise TypeError("materialized successor requires its exact typed result")
        authority.validate()
        self.copy_receipt.validate()
        self.execution_contract.validate(
            expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        )
        require_original_confirmatory_canonical_e_projection(
            self.checkpoint_projection,
            self.execution_contract,
        )
        if (
            self.execution_contract.execution_mode != "successor_resume"
            or self.execution_contract.retry_of_run_id != authority.retry_of_run_id
            or self.execution_contract.predecessor_snapshot_sha256
            != authority.predecessor_snapshot_sha256
            or self.execution_contract.predecessor_copy_receipt_sha256
            != self.copy_receipt.receipt_sha256
            or self.copy_receipt.scientific_authority_projection_sha256
            != authority.projection_sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "materialized successor differs from its exact pre-copy authority"
            )


def materialize_original_confirmatory_successor_checkpoint_execution(
    authority: OriginalConfirmatorySuccessorPrecopyCheckpointProjection,
    *,
    destination_run_directory: Path,
    fresh_template_contract: ConfirmatoryCheckpointExecutionContract,
) -> MaterializedOriginalConfirmatorySuccessor:
    """Copy after E ACK and RunTracker creation; never discover, retry, or fall back."""

    if type(authority) is not OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
        raise TypeError("successor materialization requires its exact pre-copy authority")
    authority.validate()
    fresh_template_contract.validate(
        expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
    )
    if (
        destination_run_directory != destination_run_directory.absolute()
        or destination_run_directory.name != authority.successor_run_id
        or destination_run_directory == authority.predecessor_run_directory
        or fresh_template_contract != _fresh_contract_from_precopy(authority)
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor materialization paths or Q science differ from E"
        )
    try:
        destination_value = destination_run_directory.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(
            "successor RunTracker directory is unavailable after exact creation"
        ) from exc
    if (
        not stat.S_ISDIR(destination_value.st_mode)
        or _is_reparse(destination_value)
        or destination_run_directory.is_symlink()
        or _named_streams(destination_run_directory)
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor RunTracker directory is linked/reparse/streamed"
        )
    root_before = _directory_identity_from_path(
        authority.predecessor_run_directory,
        role="authorized predecessor run directory",
    )
    if root_before != authority.predecessor_root_identity:
        raise ConfirmatoryCheckpointContractError(
            "qualified predecessor root changed before post-E import"
        )
    destination_identities: list[ConfirmatoryCheckpointFileIdentity] = []
    postcopy_directives: list[ConfirmatoryCheckpointDirective] = []
    fresh_paths: list[str] = []
    published_by_fit: dict[str, PublishedPath] = {}
    try:
        with anchored_physical_copy_session(
            authority.predecessor_run_directory,
            destination_run_directory,
        ) as copy_session:
            for value in authority.directives:
                source = value.source_predecessor_checkpoint
                destination = destination_run_directory / value.destination_checkpoint_relative_path
                if source is None:
                    if os.path.lexists(destination):
                        raise ConfirmatoryCheckpointContractError(
                            "fresh successor checkpoint appeared before import"
                        )
                    fresh_paths.append(value.destination_checkpoint_relative_path)
                    postcopy_directives.append(
                        replace(
                            _fresh_directive_from_precopy(value),
                            execution_mode="successor_resume",
                        )
                    )
                    continue
                expected_source = (
                    authority.predecessor_run_directory / value.destination_checkpoint_relative_path
                )
                if source.path != expected_source:
                    raise ConfirmatoryCheckpointContractError(
                        "successor source path differs from its exact relative destination"
                    )
                validate_confirmatory_checkpoint_artifact(
                    source.path,
                    expected_configuration=value.expected_configuration,
                    expected_model_metadata=value.expected_model_metadata,
                    expected_data_and_split_sha256=value.expected_data_and_split,
                )
                with _hold_private_checkpoint_snapshot(
                    source.path,
                    role="E-authorized predecessor checkpoint",
                ) as source_snapshot:
                    if (
                        source_snapshot.sha256 != source.sha256
                        or source_snapshot.size_bytes != source.size_bytes
                        or source_snapshot.identity != source.physical_identity
                        or stat.S_IMODE(source_snapshot.identity.mode)
                        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    ):
                        raise ConfirmatoryCheckpointContractError(
                            "E-authorized predecessor checkpoint changed before copy"
                        )
                    published = copy_session.copy_file_no_overwrite(
                        value.destination_checkpoint_relative_path,
                        expected_size_bytes=source.size_bytes,
                        expected_sha256=source.sha256,
                    )
                    if (
                        published.path != destination
                        or published.kind != "file"
                        or published.sha256 != source.sha256
                        or published.identity[:2]
                        == (
                            source.physical_identity.device,
                            source.physical_identity.inode,
                        )
                    ):
                        raise ConfirmatoryCheckpointContractError(
                            "successor physical-copy result differs from E"
                        )
                published_by_fit[value.fit_id] = published
            copy_session.assert_roots_current()
    except ConfirmatoryCheckpointContractError:
        raise
    except BaseException as exc:
        raise ConfirmatoryCheckpointContractError(
            "post-E successor checkpoint import failed closed; partial run is retained"
        ) from exc
    for value in authority.directives:
        source = value.source_predecessor_checkpoint
        if source is None:
            continue
        destination = destination_run_directory / value.destination_checkpoint_relative_path
        publication = published_by_fit.get(value.fit_id)
        if publication is None:
            raise ConfirmatoryCheckpointContractError(
                "post-E successor copy lacks one authorized publication receipt"
            )
        validate_confirmatory_checkpoint_artifact(
            destination,
            expected_configuration=value.expected_configuration,
            expected_model_metadata=value.expected_model_metadata,
            expected_data_and_split_sha256=value.expected_data_and_split,
        )
        destination_snapshot = _read_private_checkpoint_bytes(
            destination,
            role="post-E imported successor checkpoint",
        )
        destination_identity = ConfirmatoryCheckpointFileIdentity(
            path=destination,
            physical_identity=destination_snapshot.identity,
            size_bytes=destination_snapshot.size_bytes,
            sha256=destination_snapshot.sha256,
        )
        destination_identity.validate()
        if (
            destination_identity.sha256 != source.sha256
            or destination_identity.size_bytes != source.size_bytes
            or destination_identity.file_id_128 == source.file_id_128
            or publication.identity[:3]
            != (
                destination_identity.physical_identity.device,
                destination_identity.physical_identity.inode,
                destination_identity.size_bytes,
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "post-E destination identity is aliased or hash-mismatched"
            )
        destination_identities.append(destination_identity)
        postcopy_directives.append(
            ConfirmatoryCheckpointDirective(
                execution_mode="successor_resume",
                cell_id=value.cell_id,
                fold_id=value.fold_id,
                action=value.action,
                source_predecessor_checkpoint=ConfirmatoryCheckpointFileIdentity(
                    path=source.path,
                    physical_identity=source.physical_identity,
                    size_bytes=source.size_bytes,
                    sha256=source.sha256,
                ),
                destination_imported_checkpoint=destination_identity,
                versioned_checkpoint_output_directory_relative_path=(
                    value.versioned_checkpoint_output_directory_relative_path
                ),
                checkpoint_execution_manifest_relative_path=(
                    value.checkpoint_execution_manifest_relative_path
                ),
                checkpoint_sha256=source.sha256,
                checkpoint_size_bytes=source.size_bytes,
                completed_epochs_before_fit=value.completed_epochs_before_fit,
                stopped_early_before_fit=value.stopped_early_before_fit,
                next_epoch_index=value.next_epoch_index,
                maximum_epochs=value.maximum_epochs,
                expected_configuration_json=_canonical_json(value.expected_configuration),
                expected_configuration_sha256=value.expected_configuration_sha256,
                expected_model_metadata_json=_canonical_json(value.expected_model_metadata),
                expected_model_metadata_sha256=(value.expected_model_metadata_sha256),
                expected_data_and_split_json=_canonical_json(value.expected_data_and_split),
                expected_data_and_split_sha256=(value.expected_data_and_split_sha256),
            )
        )
    root_after = _directory_identity_from_path(
        authority.predecessor_run_directory,
        role="authorized predecessor run directory after import",
    )
    expected_existing = {
        value.destination_checkpoint_relative_path
        for value in authority.directives
        if value.source_predecessor_checkpoint is not None
    }
    observed_existing = {
        value.relative_to(destination_run_directory).as_posix()
        for value in (destination_run_directory / "cells").glob("*/checkpoints/fold_*.pt")
        if value.is_file()
    }
    if (
        root_after != root_before
        or observed_existing != expected_existing
        or any(os.path.lexists(destination_run_directory / relative) for relative in fresh_paths)
    ):
        raise ConfirmatoryCheckpointContractError(
            "successor source changed, destination inventory drifted, or fallback occurred"
        )
    destination_tuple = tuple(destination_identities)
    receipt_provisional = OriginalConfirmatorySuccessorCopyReceipt(
        schema_version=1,
        policy=ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_RECEIPT_POLICY,
        scientific_authority_projection_sha256=authority.projection_sha256,
        retry_of_run_id=authority.retry_of_run_id,
        run_id=authority.successor_run_id,
        predecessor_run_directory=authority.predecessor_run_directory,
        destination_run_directory=destination_run_directory,
        copy_policy=authority.copy_policy,
        predecessor_root_identity_before=root_before,
        predecessor_root_identity_after=root_after,
        copied_checkpoints=destination_tuple,
        fresh_checkpoint_relative_paths=tuple(fresh_paths),
        copied_checkpoint_count=len(destination_tuple),
        copied_total_bytes=sum(value.size_bytes for value in destination_tuple),
        destination_inventory_sha256=canonical_sha256(
            [value.as_dict() for value in destination_tuple]
        ),
        predecessor_source_inventory_sha256=(authority.source_checkpoint_inventory_sha256),
        predecessor_unchanged=True,
        outcome_artifacts_read=False,
        automatic_retry_allowed=False,
        checkpoint_fallback_used=False,
        receipt_sha256="0" * 64,
    )
    receipt = replace(
        receipt_provisional,
        receipt_sha256=canonical_sha256(receipt_provisional.payload_without_self_hash()),
    )
    receipt.validate()
    contract = _build_execution_contract(
        execution_mode="successor_resume",
        retry_of_run_id=authority.retry_of_run_id,
        directives=tuple(postcopy_directives),
        predecessor_checkpoint_read_performed=True,
        predecessor_checkpoint_copy_performed=True,
        predecessor_snapshot_sha256=authority.predecessor_snapshot_sha256,
        predecessor_copy_receipt_sha256=receipt.receipt_sha256,
    )
    projection = build_original_confirmatory_canonical_e_checkpoint_projection(contract)
    result = MaterializedOriginalConfirmatorySuccessor(
        execution_contract=contract,
        checkpoint_projection=projection,
        copy_receipt=receipt,
    )
    result.validate(authority)
    return result


def require_original_confirmatory_canonical_e_projection(
    projection: OriginalConfirmatoryCanonicalECheckpointProjection,
    contract: ConfirmatoryCheckpointExecutionContract,
) -> ConfirmatoryCheckpointExecutionContract:
    """Require typed lossless canonical-E authorization for this exact contract."""

    if not isinstance(
        projection,
        OriginalConfirmatoryCanonicalECheckpointProjection,
    ):
        raise TypeError("canonical E authorization must use its typed checkpoint projection")
    projection.validate()
    expected = build_original_confirmatory_canonical_e_checkpoint_projection(contract)
    if projection != expected:
        raise ConfirmatoryCheckpointContractError(
            "canonical E checkpoint projection differs from the exact runner contract"
        )
    return contract


def fit_directive_authority_records(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> tuple[dict[str, Any], ...]:
    """Return the lossless canonical E records for compatibility/reporting."""

    projection = build_original_confirmatory_canonical_e_checkpoint_projection(contract)
    return tuple(value.as_dict() for value in projection.directives)


def fit_directives_root_sha256(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> str:
    return build_original_confirmatory_canonical_e_checkpoint_projection(contract).directives_sha256


@dataclass(frozen=True, slots=True, eq=False)
class _OriginalConfirmatoryPreparedMatrixRequest:
    """Prepared matrix-only state created inside the guarded full lifecycle."""

    rotations: tuple[ConfirmatoryRotationInputs, ...]
    plan: ConfirmatoryMatrixPlan
    controls: ConfirmatoryExecutionControls
    output_directory: Path
    frozen_blockers: tuple[tuple[str, ConfirmatoryFrozenBlocker], ...]
    artifact_scope: str
    gate_evidence: ConfirmatoryExecutionGateEvidence
    canonical_e_projection: OriginalConfirmatoryCanonicalECheckpointProjection
    scientific_authority_projection_sha256: str
    draft_checkpoint_contract: ConfirmatoryCheckpointExecutionContract

    def validate(self) -> None:
        """Reject any request that is not the exact verified Q/E projection."""

        if type(self) is not _OriginalConfirmatoryPreparedMatrixRequest:
            raise TypeError("run-confirmatory requires its exact prepared matrix request type")
        if (
            type(self.rotations) is not tuple
            or len(self.rotations) != 3
            or any(type(value) is not ConfirmatoryRotationInputs for value in self.rotations)
            or type(self.plan) is not ConfirmatoryMatrixPlan
            or type(self.controls) is not ConfirmatoryExecutionControls
            or not isinstance(self.output_directory, Path)
            or not self.output_directory.is_absolute()
            or type(self.frozen_blockers) is not tuple
            or self.artifact_scope != REAL_CONFIRMATORY_ARTIFACT_SCOPE
            or type(self.gate_evidence) is not ConfirmatoryExecutionGateEvidence
            or type(self.canonical_e_projection)
            is not OriginalConfirmatoryCanonicalECheckpointProjection
            or not _valid_sha256(self.scientific_authority_projection_sha256)
            or type(self.draft_checkpoint_contract) is not ConfirmatoryCheckpointExecutionContract
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory capsule request is incomplete or noncanonical"
            )
        try:
            resolved_output = self.output_directory.resolve(strict=False)
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory output path cannot be resolved"
            ) from exc
        if resolved_output != self.output_directory:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory output path is not absolute and canonical"
            )
        blocker_keys: list[str] = []
        for item in self.frozen_blockers:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not ConfirmatoryFrozenBlocker
            ):
                raise ConfirmatoryCheckpointContractError(
                    "run-confirmatory blockers are not exact immutable typed records"
                )
            blocker_keys.append(item[0])
        if (
            self.frozen_blockers != tuple(sorted(self.frozen_blockers, key=lambda item: item[0]))
            or len(set(blocker_keys)) != len(blocker_keys)
            or len({value.casefold() for value in blocker_keys}) != len(blocker_keys)
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory blockers are duplicated or noncanonical"
            )
        self.controls.validate_for_plan(self.plan)
        if self.plan != self.controls.plan:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory plan differs from its frozen controls"
            )
        for rotation in self.rotations:
            rotation.validate(self.controls)
        if {value.outer_fold for value in self.rotations} != set(self.controls.official_folds):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory rotations differ from the three official folds"
            )
        draft = self.draft_checkpoint_contract
        draft.validate(expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT)
        if (
            draft.contract_profile != "original_confirmatory_exact_180"
            or draft.authority_projection_sha256 is not None
            or draft.authority_predecessor_binding_sha256 is not None
            or draft.authority_resume_adapter_sha256 is not None
            or draft.authority_fit_directives_root_sha256 is not None
            or draft.authority_checkpoint_allowlist_sha256 is not None
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory requires one unbound exact-180 Q-derived draft"
            )
        require_original_confirmatory_canonical_e_projection(
            self.canonical_e_projection,
            draft,
        )


def _valid_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value).issubset(_HEX)


def _require_closed_absolute_path(path: object, *, role: str) -> Path:
    """Validate an already-canonical absolute projection without filesystem access."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ConfirmatoryCheckpointContractError(f"{role} is not an exact absolute Path")
    if Path(os.path.normpath(str(path))) != path or any(part == ".." for part in path.parts):
        raise ConfirmatoryCheckpointContractError(f"{role} is not lexically canonical")
    return path


@dataclass(frozen=True, slots=True, eq=False)
class OriginalConfirmatoryCapsuleExecutionRequest:
    """Closed Q/E-bound input for the complete original-confirmatory lifecycle.

    The handler may construct this object only after the E-consumption custody ACK.
    It contains paths and immutable typed projections, never loaded arrays, callbacks,
    dependency injection, a raw CLI tail, or an ambient-discovery capability.
    """

    gate_evidence: ConfirmatoryExecutionGateEvidence
    primary_run_directory: Path
    project_root: Path
    freeze_directory: Path
    technical_authority_directory: Path
    technical_authority_artifact_root_sha256: str
    technical_authorization_sha256: str
    technical_authority_namespace_directory: Path
    technical_authority_namespace_claim_sha256: str
    published_technical_authority_lifecycle_binding_sha256: str
    dataset_path: Path
    manifest_path: Path
    duplicate_audit_path: Path
    pathology_encoder_audit_path: Path
    frozen_primary_config_path: Path
    frozen_confirmatory_config_path: Path
    crop_cache_path: Path
    expected_crop_cache_sha256: str
    expected_crop_metadata_sha256: str
    expected_raw_inventory_sha256: str
    frozen_feature_caches: tuple[ConfirmatoryFrozenFeatureCacheSpec, ...]
    observed_label_sets: tuple[ConfirmatoryObservedLabelSet, ...]
    runs_root: Path
    supervisor_job_id: str
    attempt_id: str
    run_id: str
    expected_run_directory: Path
    retry_of_run_id: str | None
    resume_run_directory: Path | None
    lifecycle_readiness_run_directory: Path
    artifact_scope: str
    execution_mode: OriginalConfirmatoryExecutionMode
    q_static_runner_binding_sha256: str
    e_intent_core_sha256: str
    expected_plan_sha256: str
    expected_controls_binding_sha256: str
    expected_bridge_binding_sha256: str
    expected_gate_evidence_sha256: str
    expected_cli_input_binding_sha256: str
    scientific_request_projection_sha256: str
    canonical_e_projection: OriginalConfirmatoryScientificCheckpointAuthority
    draft_checkpoint_contract: ConfirmatoryCheckpointExecutionContract

    def validate(self) -> None:
        if type(self) is not OriginalConfirmatoryCapsuleExecutionRequest:
            raise TypeError("run-confirmatory requires its exact high-level capsule request type")
        if (
            type(self.gate_evidence) is not ConfirmatoryExecutionGateEvidence
            or type(self.frozen_feature_caches) is not tuple
            or any(
                type(value) is not ConfirmatoryFrozenFeatureCacheSpec
                for value in self.frozen_feature_caches
            )
            or type(self.observed_label_sets) is not tuple
            or any(
                type(value) is not ConfirmatoryObservedLabelSet
                for value in self.observed_label_sets
            )
            or type(self.supervisor_job_id) is not str
            or not self.supervisor_job_id
            or Path(self.supervisor_job_id).name != self.supervisor_job_id
            or type(self.attempt_id) is not str
            or not self.attempt_id
            or Path(self.attempt_id).name != self.attempt_id
            or type(self.run_id) is not str
            or not self.run_id
            or self.artifact_scope != REAL_CONFIRMATORY_ARTIFACT_SCOPE
            or self.execution_mode not in {"fresh", "successor_resume"}
            or not _valid_sha256(self.expected_crop_cache_sha256)
            or not _valid_sha256(self.expected_crop_metadata_sha256)
            or not _valid_sha256(self.expected_raw_inventory_sha256)
            or not _valid_sha256(self.technical_authority_artifact_root_sha256)
            or not _valid_sha256(self.technical_authorization_sha256)
            or not _valid_sha256(self.technical_authority_namespace_claim_sha256)
            or not _valid_sha256(self.published_technical_authority_lifecycle_binding_sha256)
            or not _valid_sha256(self.q_static_runner_binding_sha256)
            or not _valid_sha256(self.e_intent_core_sha256)
            or not _valid_sha256(self.expected_plan_sha256)
            or not _valid_sha256(self.expected_controls_binding_sha256)
            or not _valid_sha256(self.expected_bridge_binding_sha256)
            or not _valid_sha256(self.expected_gate_evidence_sha256)
            or not _valid_sha256(self.expected_cli_input_binding_sha256)
            or not _valid_sha256(self.scientific_request_projection_sha256)
            or type(self.draft_checkpoint_contract) is not ConfirmatoryCheckpointExecutionContract
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory high-level capsule request is incomplete or noncanonical"
            )
        for role in (
            "primary_run_directory",
            "project_root",
            "freeze_directory",
            "technical_authority_directory",
            "technical_authority_namespace_directory",
            "dataset_path",
            "manifest_path",
            "duplicate_audit_path",
            "pathology_encoder_audit_path",
            "frozen_primary_config_path",
            "frozen_confirmatory_config_path",
            "crop_cache_path",
            "runs_root",
            "expected_run_directory",
            "lifecycle_readiness_run_directory",
        ):
            _require_closed_absolute_path(getattr(self, role), role=role)
        if self.resume_run_directory is not None:
            _require_closed_absolute_path(
                self.resume_run_directory,
                role="resume_run_directory",
            )
        if (
            self.expected_run_directory != self.runs_root / self.run_id
            or Path(self.run_id).name != self.run_id
            or self.run_id in {".", ".."}
            or self.technical_authority_directory.parent
            != self.technical_authority_namespace_directory
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory run identity differs from its exact run directory"
            )
        cache_ids = tuple(value.scenario_id for value in self.frozen_feature_caches)
        if cache_ids != tuple(sorted(cache_ids)) or len(set(cache_ids)) != len(cache_ids):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory feature-cache projection is duplicated or noncanonical"
            )
        observed_folds = tuple(value.outer_fold for value in self.observed_label_sets)
        if observed_folds != tuple(sorted(observed_folds)) or len(set(observed_folds)) != len(
            observed_folds
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory observed-label projection is duplicated or noncanonical"
            )
        draft = self.draft_checkpoint_contract
        draft.validate(expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT)
        if (
            draft.execution_mode != "fresh"
            or draft.retry_of_run_id is not None
            or draft.contract_profile != "original_confirmatory_exact_180"
            or any(
                value is not None
                for value in (
                    draft.authority_projection_sha256,
                    draft.authority_predecessor_binding_sha256,
                    draft.authority_resume_adapter_sha256,
                    draft.authority_fit_directives_root_sha256,
                    draft.authority_checkpoint_allowlist_sha256,
                )
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory requires one unbound exact-180 mode-neutral fresh Q template"
            )
        projection = self.canonical_e_projection
        if self.execution_mode == "fresh":
            if (
                self.retry_of_run_id is not None
                or self.resume_run_directory is not None
                or type(projection) is not OriginalConfirmatoryCanonicalECheckpointProjection
                or projection.execution_mode != "fresh"
            ):
                raise ConfirmatoryCheckpointContractError(
                    "fresh run-confirmatory request carries successor lineage"
                )
            require_original_confirmatory_canonical_e_projection(
                projection,
                draft,
            )
            return
        if (
            type(self.retry_of_run_id) is not str
            or not self.retry_of_run_id
            or Path(self.retry_of_run_id).name != self.retry_of_run_id
            or not isinstance(self.resume_run_directory, Path)
            or self.resume_run_directory != self.runs_root / self.retry_of_run_id
            or self.retry_of_run_id.casefold() == self.run_id.casefold()
            or type(projection) is not OriginalConfirmatorySuccessorPrecopyCheckpointProjection
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor run-confirmatory request lacks exact E-selected lineage"
            )
        projection.validate()
        if (
            projection.retry_of_run_id != self.retry_of_run_id
            or projection.successor_run_id != self.run_id
            or projection.predecessor_run_directory != self.resume_run_directory
            or projection.base_template_directives_sha256 != draft.directives_sha256
            or _fresh_contract_from_precopy(projection) != draft
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor E authority differs from Q science or exact run lineage"
            )


def prepare_original_confirmatory_capsule_request(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
) -> OriginalConfirmatoryCapsuleExecutionRequest:
    """Pure fail-closed constructor boundary for a held Q/E projection."""

    if type(request) is not OriginalConfirmatoryCapsuleExecutionRequest:
        raise TypeError("capsule preparation requires its exact closed request type")
    request.validate()
    return request


_STATIC_RUNNER_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "project_root",
    "primary_run_directory",
    "freeze_directory",
    "technical_authority_directory",
    "technical_authority_artifact_root_sha256",
    "technical_authorization_sha256",
    "published_technical_authority_lifecycle_binding",
    "lifecycle_readiness_run_directory",
    "dataset_path",
    "manifest_path",
    "duplicate_audit_path",
    "pathology_encoder_audit_path",
    "frozen_primary_config_path",
    "frozen_confirmatory_config_path",
    "runs_root",
    "expected_confirmatory_gate",
    "expected_confirmatory_gate_sha256",
    "expected_cli_input_binding",
    "expected_cli_input_binding_sha256",
    "artifact_scope",
    "semantic_outcome_read_scope",
    "binding_sha256",
}
_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "namespace_directory",
    "namespace_claim_sha256",
    "review_attempt_claim_sha256",
    "technical_authority",
    "automatic_retry_allowed",
    "adoption_allowed",
    "cleanup_allowed",
    "binding_sha256",
}
_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "authority_directory",
    "chain_depth",
    "artifact_root_sha256",
    "sha256_manifest_sha256",
    "execution_source_manifest_sha256",
    "execution_source_root_sha256",
    "parent_authority_directory",
    "parent_artifact_root_sha256",
    "parent_sha256_manifest_sha256",
    "technical_authorization_sha256",
    "independent_review_receipt_sha256",
    "immutable_marker_sha256",
    "publication_attempt_sha256",
    "publication_success_sha256",
    "primary_outcomes_inspected",
    "confirmatory_outcomes_inspected",
    "confirmatory_outcome_values_read",
    "scientific_definition_changed",
    "automatic_retry_allowed",
    "binding_sha256",
}
_STATIC_RUNNER_BINDING_POLICY = "original_confirmatory_static_runner_binding_v3"
_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY = (
    "published_original_confirmatory_technical_authority_lifecycle_binding_v1"
)
_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY = (
    "original_confirmatory_technical_authority_lifecycle_binding_v1"
)
_CLI_INPUT_BINDING_FIELDS = {
    "schema_version",
    "policy",
    "crop_cache_path",
    "expected_crop_cache_sha256",
    "expected_crop_metadata_sha256",
    "expected_raw_inventory_sha256",
    "frozen_feature_caches",
    "frozen_feature_caches_sha256",
    "observed_label_sets",
    "observed_label_sets_sha256",
    "draft_checkpoint_contract",
    "draft_checkpoint_contract_sha256",
    "bridge_binding_sha256",
    "scientific_outcomes_read",
    "automatic_retry_allowed",
    "binding_sha256",
}
_SCIENTIFIC_REQUEST_PROJECTION_FIELDS = {
    "schema_version",
    "policy",
    "q_static_runner_binding_sha256",
    "job_id",
    "attempt_id",
    "run_id",
    "execution_mode",
    "retry_of_run_id",
    "runs_root",
    "expected_run_directory",
    "plan_sha256",
    "controls_binding_sha256",
    "bridge_binding_sha256",
    "gate_evidence_sha256",
    "cli_input_binding_sha256",
    "checkpoint_authority_projection",
    "checkpoint_authority_projection_sha256",
    "checkpoint_contract_profile",
    "checkpoint_directive_count",
    "artifact_scope",
    "scientific_outcomes_read",
    "selection_or_tuning_performed",
    "publication_performed",
    "automatic_retry_allowed",
    "projection_sha256",
}
_OBSERVED_LABEL_PROJECTION_FIELDS = {
    "outer_fold",
    "sample_ids",
    "pre_corruption_labels",
    "observed_labels",
    "is_injected_corruption",
    "corruption_types",
    "configuration_sha256",
}
_CHECKPOINT_IDENTITY_FIELDS = {
    "path",
    "file_id_128",
    "device",
    "inode",
    "size_bytes",
    "mode",
    "link_count",
    "modified_time_ns",
    "changed_time_ns",
    "sha256",
}
_DIRECTORY_IDENTITY_FIELDS = _CHECKPOINT_IDENTITY_FIELDS - {"sha256"}


def _exact_mapping(value: object, *, fields: set[str], role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmatoryCheckpointContractError(f"{role} must be an exact mapping")
    raw = dict(value)
    if set(raw) != fields or any(type(key) is not str for key in raw):
        raise ConfirmatoryCheckpointContractError(f"{role} has an unexpected field set")
    return raw


def _exact_sequence(value: object, *, role: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfirmatoryCheckpointContractError(f"{role} must be an exact sequence")
    return tuple(value)


def _path_projection(value: object, *, role: str) -> Path:
    if type(value) is not str or not value:
        raise ConfirmatoryCheckpointContractError(f"{role} must be an absolute path string")
    return _require_closed_absolute_path(Path(value), role=role)


def _decode_published_technical_authority_lifecycle_binding(
    value: object,
    *,
    project_root: Path,
) -> dict[str, Any]:
    published = _exact_mapping(
        value,
        fields=_PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS,
        role="published technical authority lifecycle binding",
    )
    technical = _exact_mapping(
        published["technical_authority"],
        fields=_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS,
        role="published technical authority nested lifecycle binding",
    )
    namespace = _path_projection(
        published["namespace_directory"],
        role="published technical authority namespace directory",
    )
    authority_directory = _path_projection(
        technical["authority_directory"],
        role="published technical authority directory",
    )
    parent_authority_directory = _path_projection(
        technical["parent_authority_directory"],
        role="published technical authority parent directory",
    )
    for field in (
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "execution_source_manifest_sha256",
        "execution_source_root_sha256",
        "parent_artifact_root_sha256",
        "parent_sha256_manifest_sha256",
        "technical_authorization_sha256",
        "independent_review_receipt_sha256",
        "immutable_marker_sha256",
        "publication_attempt_sha256",
        "publication_success_sha256",
    ):
        if not _valid_sha256(technical[field]):
            raise ConfirmatoryCheckpointContractError(
                f"published technical authority {field} is not SHA-256"
            )
    if not _valid_sha256(published["namespace_claim_sha256"]):
        raise ConfirmatoryCheckpointContractError(
            "published technical authority namespace claim is not SHA-256"
        )
    if not _valid_sha256(published["review_attempt_claim_sha256"]):
        raise ConfirmatoryCheckpointContractError(
            "published technical authority review-attempt claim is not SHA-256"
        )
    technical_unsigned = {key: item for key, item in technical.items() if key != "binding_sha256"}
    published_unsigned = {key: item for key, item in published.items() if key != "binding_sha256"}
    if (
        type(technical["schema_version"]) is not int
        or technical["schema_version"] != 1
        or technical["policy"] != _TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or type(technical["chain_depth"]) is not int
        or technical["chain_depth"] < 1
        or technical["primary_outcomes_inspected"] is not True
        or technical["confirmatory_outcomes_inspected"] is not False
        or technical["confirmatory_outcome_values_read"] is not False
        or technical["scientific_definition_changed"] is not False
        or technical["automatic_retry_allowed"] is not False
        or technical["binding_sha256"] != canonical_sha256(technical_unsigned)
        or type(published["schema_version"]) is not int
        or published["schema_version"] != 1
        or published["policy"] != _PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_POLICY
        or published["automatic_retry_allowed"] is not False
        or published["adoption_allowed"] is not False
        or published["cleanup_allowed"] is not False
        or published["binding_sha256"] != canonical_sha256(published_unsigned)
        or namespace != project_root / "artifacts" / "original_confirmatory_technical_authorities"
        or authority_directory.parent != namespace
        or project_root not in parent_authority_directory.parents
    ):
        raise ConfirmatoryCheckpointContractError(
            "published technical authority lifecycle binding violates its exact one-use policy"
        )
    return published


def _decode_primary_gate(value: object) -> PrimaryExecutionGateEvidence:
    fields = set(PrimaryExecutionGateEvidence.__dataclass_fields__)
    raw = _exact_mapping(value, fields=fields, role="Q primary gate projection")
    for field in (
        "freeze_directory",
        "base_freeze_directory",
    ):
        raw[field] = _path_projection(raw[field], role=f"Q primary gate {field}")
    for field in (
        "primary_matrix_cell_count",
        "primary_required_cell_count",
        "confirmatory_matrix_cell_count",
        "registration_authority_chain_depth",
    ):
        if type(raw[field]) is not int:
            raise ConfirmatoryCheckpointContractError(f"Q primary gate {field} must be an integer")
    for field in (
        "original_unamended_primary_claim_allowed",
        "amended_primary_claim_allowed",
    ):
        if type(raw[field]) is not bool:
            raise ConfirmatoryCheckpointContractError(f"Q primary gate {field} must be boolean")
    for field, item in raw.items():
        if field.endswith("_sha256") and not _valid_sha256(item):
            raise ConfirmatoryCheckpointContractError(f"Q primary gate {field} is not SHA-256")
    try:
        gate = PrimaryExecutionGateEvidence(**raw)
    except TypeError as exc:
        raise ConfirmatoryCheckpointContractError("Q primary gate is not typed") from exc
    if gate.as_dict() != {
        **raw,
        "freeze_directory": str(raw["freeze_directory"]),
        "base_freeze_directory": str(raw["base_freeze_directory"]),
    }:
        raise ConfirmatoryCheckpointContractError("Q primary gate does not round-trip exactly")
    return gate


def _decode_confirmatory_gate(value: object) -> ConfirmatoryExecutionGateEvidence:
    if not isinstance(value, Mapping):
        raise ConfirmatoryCheckpointContractError("Q confirmatory gate must be an exact mapping")
    raw = dict(value)
    allowed = set(ConfirmatoryExecutionGateEvidence.__dataclass_fields__)
    if not set(raw).issubset(allowed) or "primary_gate" not in raw:
        raise ConfirmatoryCheckpointContractError("Q confirmatory gate has an unexpected field set")
    raw["primary_gate"] = _decode_primary_gate(raw["primary_gate"])
    raw["primary_run_directory"] = _path_projection(
        raw.get("primary_run_directory"),
        role="Q confirmatory primary run",
    )
    for field in (
        "completed_required_cell_count",
        "primary_statistics_comparison_count",
    ):
        if field in raw and raw[field] is not None and type(raw[field]) is not int:
            raise ConfirmatoryCheckpointContractError(
                f"Q confirmatory gate {field} must be an integer or null"
            )
    for field in (
        "primary_finalization_only_successor",
        "primary_orphan_recovery",
    ):
        if field in raw and type(raw[field]) is not bool:
            raise ConfirmatoryCheckpointContractError(
                f"Q confirmatory gate {field} must be boolean"
            )
    for field, item in raw.items():
        if field.endswith("_sha256") and item is not None and not _valid_sha256(item):
            raise ConfirmatoryCheckpointContractError(
                f"Q confirmatory gate {field} is not SHA-256 or null"
            )
    try:
        gate = ConfirmatoryExecutionGateEvidence(**raw)
    except TypeError as exc:
        raise ConfirmatoryCheckpointContractError("Q confirmatory gate is not typed") from exc
    expected = dict(value)
    if gate.as_dict() != expected:
        raise ConfirmatoryCheckpointContractError("Q confirmatory gate does not round-trip exactly")
    return gate


def _cache_projection(
    value: ConfirmatoryFrozenFeatureCacheSpec,
) -> dict[str, Any]:
    return {
        "scenario_id": value.scenario_id,
        "cache_path": str(value.cache_path),
        "expected_cache_sha256": value.expected_cache_sha256,
        "expected_metadata_sha256": value.expected_metadata_sha256,
        "expected_weight_sha256": value.expected_weight_sha256,
    }


def _decode_feature_cache(value: object) -> ConfirmatoryFrozenFeatureCacheSpec:
    raw = _exact_mapping(
        value,
        fields=set(ConfirmatoryFrozenFeatureCacheSpec.__dataclass_fields__),
        role="Q frozen feature-cache projection",
    )
    spec = ConfirmatoryFrozenFeatureCacheSpec(
        scenario_id=str(raw["scenario_id"]),
        cache_path=_path_projection(raw["cache_path"], role="Q frozen feature cache path"),
        expected_cache_sha256=str(raw["expected_cache_sha256"]),
        expected_metadata_sha256=str(raw["expected_metadata_sha256"]),
        expected_weight_sha256=(
            str(raw["expected_weight_sha256"])
            if raw["expected_weight_sha256"] is not None
            else None
        ),
    )
    if (
        type(raw["scenario_id"]) is not str
        or not raw["scenario_id"]
        or not _valid_sha256(spec.expected_cache_sha256)
        or not _valid_sha256(spec.expected_metadata_sha256)
        or (
            spec.expected_weight_sha256 is not None
            and not _valid_sha256(spec.expected_weight_sha256)
        )
        or _cache_projection(spec) != raw
    ):
        raise ConfirmatoryCheckpointContractError(
            "Q frozen feature-cache projection is incomplete or altered"
        )
    return spec


def _observed_label_projection(value: ConfirmatoryObservedLabelSet) -> dict[str, Any]:
    return {
        "outer_fold": value.outer_fold,
        "sample_ids": list(value.sample_ids),
        "pre_corruption_labels": value.pre_corruption_labels.tolist(),
        "observed_labels": value.observed_labels.tolist(),
        "is_injected_corruption": value.is_injected_corruption.tolist(),
        "corruption_types": list(value.corruption_types),
        "configuration_sha256": value.configuration_sha256,
    }


def _decode_observed_labels(value: object) -> ConfirmatoryObservedLabelSet:
    raw = _exact_mapping(
        value,
        fields=_OBSERVED_LABEL_PROJECTION_FIELDS,
        role="Q observed-label projection",
    )
    sample_ids = _exact_sequence(raw["sample_ids"], role="Q observed-label sample IDs")
    pre = _exact_sequence(
        raw["pre_corruption_labels"],
        role="Q pre-corruption labels",
    )
    observed = _exact_sequence(raw["observed_labels"], role="Q observed labels")
    injected = _exact_sequence(
        raw["is_injected_corruption"],
        role="Q injected-corruption flags",
    )
    corruption_types = _exact_sequence(
        raw["corruption_types"],
        role="Q corruption types",
    )
    if (
        type(raw["outer_fold"]) is not int
        or raw["outer_fold"] not in {1, 2, 3}
        or not sample_ids
        or len({len(sample_ids), len(pre), len(observed), len(injected), len(corruption_types)})
        != 1
        or any(type(item) is not str or not item for item in sample_ids)
        or any(type(item) is not int for item in (*pre, *observed))
        or any(type(item) is not bool for item in injected)
        or any(type(item) is not str for item in corruption_types)
        or not _valid_sha256(raw["configuration_sha256"])
    ):
        raise ConfirmatoryCheckpointContractError(
            "Q observed-label projection is incomplete or noncanonical"
        )
    pre_array = np.asarray(pre, dtype=np.int64)
    observed_array = np.asarray(observed, dtype=np.int64)
    injected_array = np.asarray(injected, dtype=np.bool_)
    pre_array.setflags(write=False)
    observed_array.setflags(write=False)
    injected_array.setflags(write=False)
    decoded = ConfirmatoryObservedLabelSet(
        outer_fold=raw["outer_fold"],
        sample_ids=tuple(sample_ids),
        pre_corruption_labels=pre_array,
        observed_labels=observed_array,
        is_injected_corruption=injected_array,
        corruption_types=tuple(corruption_types),
        configuration_sha256=raw["configuration_sha256"],
    )
    if _observed_label_projection(decoded) != raw:
        raise ConfirmatoryCheckpointContractError(
            "Q observed-label projection does not round-trip exactly"
        )
    return decoded


def _decode_checkpoint_identity(
    value: object,
    *,
    role: str,
) -> OriginalConfirmatoryCanonicalECheckpointIdentity:
    raw = _exact_mapping(value, fields=_CHECKPOINT_IDENTITY_FIELDS, role=role)
    integer_fields = (
        "device",
        "inode",
        "size_bytes",
        "mode",
        "link_count",
        "modified_time_ns",
        "changed_time_ns",
    )
    if (
        any(type(raw[field]) is not int for field in integer_fields)
        or not _valid_sha256(raw["sha256"])
        or type(raw["file_id_128"]) is not str
    ):
        raise ConfirmatoryCheckpointContractError(f"{role} fields are not exact")
    identity = OriginalConfirmatoryCanonicalECheckpointIdentity(
        path=_path_projection(raw["path"], role=f"{role} path"),
        physical_identity=ConfirmatoryCheckpointPhysicalIdentity(
            device=raw["device"],
            inode=raw["inode"],
            size_bytes=raw["size_bytes"],
            mode=raw["mode"],
            link_count=raw["link_count"],
            modified_time_ns=raw["modified_time_ns"],
            changed_time_ns=raw["changed_time_ns"],
        ),
        size_bytes=raw["size_bytes"],
        sha256=raw["sha256"],
    )
    identity.validate()
    if identity.as_dict() != raw:
        raise ConfirmatoryCheckpointContractError(f"{role} does not round-trip exactly")
    return identity


def _decode_directory_identity(
    value: object,
    *,
    role: str,
) -> OriginalConfirmatoryCanonicalEDirectoryIdentity:
    raw = _exact_mapping(value, fields=_DIRECTORY_IDENTITY_FIELDS, role=role)
    integer_fields = (
        "device",
        "inode",
        "size_bytes",
        "mode",
        "link_count",
        "modified_time_ns",
        "changed_time_ns",
    )
    if (
        any(type(raw[field]) is not int for field in integer_fields)
        or type(raw["file_id_128"]) is not str
    ):
        raise ConfirmatoryCheckpointContractError(f"{role} fields are not exact")
    identity = OriginalConfirmatoryCanonicalEDirectoryIdentity(
        path=_path_projection(raw["path"], role=f"{role} path"),
        device=raw["device"],
        inode=raw["inode"],
        size_bytes=raw["size_bytes"],
        mode=raw["mode"],
        link_count=raw["link_count"],
        modified_time_ns=raw["modified_time_ns"],
        changed_time_ns=raw["changed_time_ns"],
    )
    identity.validate()
    if identity.as_dict() != raw:
        raise ConfirmatoryCheckpointContractError(f"{role} does not round-trip exactly")
    return identity


def _decode_canonical_e_projection(
    value: object,
) -> OriginalConfirmatoryCanonicalECheckpointProjection:
    raw = _exact_mapping(
        value,
        fields=set(OriginalConfirmatoryCanonicalECheckpointProjection.__dataclass_fields__),
        role="E checkpoint authority projection",
    )
    directives_raw = _exact_sequence(
        raw["directives"],
        role="E checkpoint fit directives",
    )
    directives: list[OriginalConfirmatoryCanonicalEFitDirective] = []
    directive_fields = set(OriginalConfirmatoryCanonicalEFitDirective.__dataclass_fields__)
    for item in directives_raw:
        row = _exact_mapping(
            item,
            fields=directive_fields,
            role="E checkpoint fit directive",
        )
        if (
            row["source_predecessor_checkpoint"] is not None
            or row["destination_imported_checkpoint"] is not None
        ):
            raise ConfirmatoryCheckpointContractError(
                "fresh E checkpoint directive carries predecessor identity"
            )
        directive = OriginalConfirmatoryCanonicalEFitDirective(
            **{
                **row,
                "expected_configuration": dict(
                    _exact_mapping(
                        row["expected_configuration"],
                        fields=set(row["expected_configuration"]),
                        role="E expected checkpoint configuration",
                    )
                ),
                "expected_model_metadata": dict(
                    _exact_mapping(
                        row["expected_model_metadata"],
                        fields=set(row["expected_model_metadata"]),
                        role="E expected model metadata",
                    )
                ),
                "expected_data_and_split": dict(
                    _exact_mapping(
                        row["expected_data_and_split"],
                        fields=set(row["expected_data_and_split"]),
                        role="E expected data/split",
                    )
                ),
            }
        )
        directive.validate()
        if directive.as_dict() != row:
            raise ConfirmatoryCheckpointContractError(
                "E checkpoint fit directive does not round-trip exactly"
            )
        directives.append(directive)
    projection = OriginalConfirmatoryCanonicalECheckpointProjection(
        **{
            **raw,
            "directives": tuple(directives),
        }
    )
    projection.validate()
    if projection.execution_mode != "fresh" or projection.as_dict() != raw:
        raise ConfirmatoryCheckpointContractError(
            "fresh E checkpoint authority projection does not round-trip exactly"
        )
    return projection


def _decode_successor_precopy_projection(
    value: object,
) -> OriginalConfirmatorySuccessorPrecopyCheckpointProjection:
    raw = _exact_mapping(
        value,
        fields=set(OriginalConfirmatorySuccessorPrecopyCheckpointProjection.__dataclass_fields__),
        role="successor E pre-copy checkpoint authority",
    )
    directives_raw = _exact_sequence(
        raw["directives"],
        role="successor E pre-copy directives",
    )
    directive_fields = set(OriginalConfirmatorySuccessorPrecopyFitDirective.__dataclass_fields__)
    directives: list[OriginalConfirmatorySuccessorPrecopyFitDirective] = []
    for item in directives_raw:
        row = _exact_mapping(
            item,
            fields=directive_fields,
            role="successor E pre-copy directive",
        )
        source_raw = row["source_predecessor_checkpoint"]
        source = (
            _decode_checkpoint_identity(
                source_raw,
                role="successor E source checkpoint",
            )
            if source_raw is not None
            else None
        )
        directive = OriginalConfirmatorySuccessorPrecopyFitDirective(
            **{
                **row,
                "source_predecessor_checkpoint": source,
                "expected_configuration": dict(
                    _exact_mapping(
                        row["expected_configuration"],
                        fields=set(row["expected_configuration"]),
                        role="successor E expected checkpoint configuration",
                    )
                ),
                "expected_model_metadata": dict(
                    _exact_mapping(
                        row["expected_model_metadata"],
                        fields=set(row["expected_model_metadata"]),
                        role="successor E expected model metadata",
                    )
                ),
                "expected_data_and_split": dict(
                    _exact_mapping(
                        row["expected_data_and_split"],
                        fields=set(row["expected_data_and_split"]),
                        role="successor E expected data/split",
                    )
                ),
            }
        )
        directive.validate()
        if directive.as_dict() != row:
            raise ConfirmatoryCheckpointContractError(
                "successor E pre-copy directive does not round-trip exactly"
            )
        directives.append(directive)
    qualification = _exact_mapping(
        raw["predecessor_qualification"],
        fields={
            "predecessor_class",
            "policy",
            "terminal_status",
            "sealed_integrity_valid",
            "integrity_registry_record_present",
            "artifact_root_sha256",
            "artifact_manifest_sha256",
            "immutable_marker_sha256",
            "status_sha256",
            "completion_evidence_sha256",
            "process_receipt",
            "active_lock_paths_before",
            "active_lock_paths_after",
            "stage_attestation_count",
            "disposition_record_count",
            "root_inventory_before_sha256",
            "root_inventory_after_sha256",
            "qualification_sha256",
        },
        role="successor E predecessor qualification",
    )
    projection = OriginalConfirmatorySuccessorPrecopyCheckpointProjection(
        **{
            **raw,
            "predecessor_run_directory": _path_projection(
                raw["predecessor_run_directory"],
                role="successor E predecessor run directory",
            ),
            "predecessor_qualification": qualification,
            "predecessor_root_identity": _decode_directory_identity(
                raw["predecessor_root_identity"],
                role="successor E predecessor root identity",
            ),
            "directives": tuple(directives),
        }
    )
    projection.validate()
    if projection.as_dict() != raw:
        raise ConfirmatoryCheckpointContractError(
            "successor E pre-copy projection does not round-trip exactly"
        )
    return projection


def _decode_scientific_checkpoint_authority(
    value: object,
) -> OriginalConfirmatoryScientificCheckpointAuthority:
    if not isinstance(value, Mapping):
        raise ConfirmatoryCheckpointContractError(
            "E checkpoint authority must be an exact tagged mapping"
        )
    policy = value.get("policy")
    if policy == ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY:
        return _decode_canonical_e_projection(value)
    if policy == ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY:
        return _decode_successor_precopy_projection(value)
    raise ConfirmatoryCheckpointContractError(
        "E checkpoint authority policy is outside the closed fresh/successor union"
    )


def _reconstruct_draft_checkpoint_contract(
    summary: object,
    projection: OriginalConfirmatoryScientificCheckpointAuthority,
) -> ConfirmatoryCheckpointExecutionContract:
    raw = _exact_mapping(
        summary,
        fields=(set(ConfirmatoryCheckpointExecutionContract.__dataclass_fields__) - {"directives"})
        | {"directive_count"},
        role="Q draft checkpoint contract",
    )
    if raw["directive_count"] != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT:
        raise ConfirmatoryCheckpointContractError(
            "Q draft checkpoint directive count is not exact 180"
        )
    directives = (
        tuple(_fresh_directive_from_precopy(value) for value in projection.directives)
        if type(projection) is OriginalConfirmatorySuccessorPrecopyCheckpointProjection
        else tuple(
            ConfirmatoryCheckpointDirective(
                execution_mode=value.execution_mode,
                cell_id=value.cell_id,
                fold_id=value.fold_id,
                action=value.action,
                source_predecessor_checkpoint=None,
                destination_imported_checkpoint=None,
                versioned_checkpoint_output_directory_relative_path=(
                    value.versioned_checkpoint_output_directory_relative_path
                ),
                checkpoint_execution_manifest_relative_path=(
                    value.checkpoint_execution_manifest_relative_path
                ),
                checkpoint_sha256=None,
                checkpoint_size_bytes=None,
                completed_epochs_before_fit=value.completed_epochs_before_fit,
                stopped_early_before_fit=value.stopped_early_before_fit,
                next_epoch_index=value.next_epoch_index,
                maximum_epochs=value.maximum_epochs,
                expected_configuration_json=_canonical_json(value.expected_configuration),
                expected_configuration_sha256=value.expected_configuration_sha256,
                expected_model_metadata_json=_canonical_json(value.expected_model_metadata),
                expected_model_metadata_sha256=value.expected_model_metadata_sha256,
                expected_data_and_split_json=_canonical_json(value.expected_data_and_split),
                expected_data_and_split_sha256=value.expected_data_and_split_sha256,
            )
            for value in projection.directives
        )
    )
    draft = ConfirmatoryCheckpointExecutionContract(
        **{key: item for key, item in raw.items() if key != "directive_count"},
        directives=directives,
    )
    registered = _register_checkpoint_execution_contract(
        draft,
        expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    )
    if (
        registered.as_dict() != raw
        or (
            type(projection) is OriginalConfirmatoryCanonicalECheckpointProjection
            and build_original_confirmatory_canonical_e_checkpoint_projection(registered)
            != projection
        )
        or (
            type(projection) is OriginalConfirmatorySuccessorPrecopyCheckpointProjection
            and registered.directives_sha256 != projection.base_template_directives_sha256
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "Q draft contract differs from the exact E checkpoint projection"
        )
    return registered


def build_original_confirmatory_capsule_request_from_authority(
    *,
    q_static_runner_binding: Mapping[str, Any],
    e_scientific_request_projection: Mapping[str, Any],
    e_intent_core_sha256: str,
) -> OriginalConfirmatoryCapsuleExecutionRequest:
    """Decode one exact held Q/E mapping pair without filesystem discovery."""

    q = _exact_mapping(
        q_static_runner_binding,
        fields=_STATIC_RUNNER_BINDING_FIELDS,
        role="Q static runner binding",
    )
    e = _exact_mapping(
        e_scientific_request_projection,
        fields=_SCIENTIFIC_REQUEST_PROJECTION_FIELDS,
        role="E scientific request projection",
    )
    cli = _exact_mapping(
        q["expected_cli_input_binding"],
        fields=_CLI_INPUT_BINDING_FIELDS,
        role="Q CLI input binding",
    )
    project_root = _path_projection(q["project_root"], role="Q project root")
    published_binding = _decode_published_technical_authority_lifecycle_binding(
        q["published_technical_authority_lifecycle_binding"],
        project_root=project_root,
    )
    published_technical = published_binding["technical_authority"]
    gate_raw = q["expected_confirmatory_gate"]
    if (
        q["schema_version"] != 3
        or q["policy"] != _STATIC_RUNNER_BINDING_POLICY
        or not _valid_sha256(q["technical_authority_artifact_root_sha256"])
        or not _valid_sha256(q["technical_authorization_sha256"])
        or q["technical_authority_directory"] != published_technical["authority_directory"]
        or q["freeze_directory"] != published_technical["parent_authority_directory"]
        or q["technical_authority_artifact_root_sha256"]
        != published_technical["artifact_root_sha256"]
        or q["technical_authorization_sha256"]
        != published_technical["technical_authorization_sha256"]
        or q["artifact_scope"] != REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or q["semantic_outcome_read_scope"] != "integrity/control_only_no_scientific_outcomes"
        or q["expected_confirmatory_gate_sha256"] != canonical_sha256(gate_raw)
        or q["expected_cli_input_binding_sha256"] != canonical_sha256(cli)
        or q["binding_sha256"]
        != canonical_sha256({key: item for key, item in q.items() if key != "binding_sha256"})
    ):
        raise ConfirmatoryCheckpointContractError(
            "Q static runner binding violates its exact policy"
        )
    if (
        cli["schema_version"] != 1
        or cli["policy"] != "original_confirmatory_cli_input_binding_v1"
        or cli["scientific_outcomes_read"] is not False
        or cli["automatic_retry_allowed"] is not False
        or cli["frozen_feature_caches_sha256"] != canonical_sha256(cli["frozen_feature_caches"])
        or cli["observed_label_sets_sha256"] != canonical_sha256(cli["observed_label_sets"])
        or cli["draft_checkpoint_contract_sha256"]
        != canonical_sha256(cli["draft_checkpoint_contract"])
        or cli["binding_sha256"]
        != canonical_sha256({key: item for key, item in cli.items() if key != "binding_sha256"})
    ):
        raise ConfirmatoryCheckpointContractError("Q CLI input binding violates its exact policy")
    checkpoint_raw = e["checkpoint_authority_projection"]
    if (
        e["schema_version"] != 1
        or e["policy"] != "original_confirmatory_capsule_request_projection_v1"
        or e["q_static_runner_binding_sha256"] != q["binding_sha256"]
        or e["runs_root"] != q["runs_root"]
        or e["expected_run_directory"] != str(Path(q["runs_root"]) / str(e["run_id"]))
        or e["execution_mode"] not in {"fresh", "successor_resume"}
        or (e["execution_mode"] == "fresh" and e["retry_of_run_id"] is not None)
        or (
            e["execution_mode"] == "successor_resume"
            and (
                type(e["retry_of_run_id"]) is not str
                or not e["retry_of_run_id"]
                or Path(e["retry_of_run_id"]).name != e["retry_of_run_id"]
            )
        )
        or e["gate_evidence_sha256"] != q["expected_confirmatory_gate_sha256"]
        or e["cli_input_binding_sha256"] != q["expected_cli_input_binding_sha256"]
        or e["bridge_binding_sha256"] != cli["bridge_binding_sha256"]
        or e["checkpoint_authority_projection_sha256"] != canonical_sha256(checkpoint_raw)
        or e["checkpoint_contract_profile"] != "original_confirmatory_exact_180"
        or e["checkpoint_directive_count"] != ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT
        or e["artifact_scope"] != REAL_CONFIRMATORY_ARTIFACT_SCOPE
        or e["scientific_outcomes_read"] is not False
        or e["selection_or_tuning_performed"] is not False
        or e["publication_performed"] is not False
        or e["automatic_retry_allowed"] is not False
        or e["projection_sha256"]
        != canonical_sha256({key: item for key, item in e.items() if key != "projection_sha256"})
        or not _valid_sha256(e_intent_core_sha256)
    ):
        raise ConfirmatoryCheckpointContractError(
            "E scientific request projection violates its exact no-retry policy"
        )
    for role in (
        "plan_sha256",
        "controls_binding_sha256",
        "bridge_binding_sha256",
        "gate_evidence_sha256",
        "cli_input_binding_sha256",
        "checkpoint_authority_projection_sha256",
        "projection_sha256",
    ):
        if not _valid_sha256(e[role]):
            raise ConfirmatoryCheckpointContractError(f"E {role} is not SHA-256")
    gate = _decode_confirmatory_gate(gate_raw)
    cache_rows = _exact_sequence(
        cli["frozen_feature_caches"],
        role="Q frozen feature-cache projections",
    )
    caches = tuple(_decode_feature_cache(value) for value in cache_rows)
    observed_rows = _exact_sequence(
        cli["observed_label_sets"],
        role="Q observed-label projections",
    )
    observed = tuple(_decode_observed_labels(value) for value in observed_rows)
    projection = _decode_scientific_checkpoint_authority(checkpoint_raw)
    if (
        projection.execution_mode != e["execution_mode"]
        or (
            type(projection) is OriginalConfirmatorySuccessorPrecopyCheckpointProjection
            and (
                projection.retry_of_run_id != e["retry_of_run_id"]
                or projection.successor_run_id != e["run_id"]
            )
        )
        or (
            type(projection) is OriginalConfirmatoryCanonicalECheckpointProjection
            and (e["execution_mode"] != "fresh" or projection.retry_of_run_id is not None)
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "E outer lineage differs from its tagged checkpoint authority"
        )
    draft = _reconstruct_draft_checkpoint_contract(
        cli["draft_checkpoint_contract"],
        projection,
    )
    request = OriginalConfirmatoryCapsuleExecutionRequest(
        gate_evidence=gate,
        primary_run_directory=_path_projection(
            q["primary_run_directory"],
            role="Q primary run directory",
        ),
        project_root=project_root,
        freeze_directory=_path_projection(q["freeze_directory"], role="Q freeze directory"),
        technical_authority_directory=_path_projection(
            q["technical_authority_directory"],
            role="Q technical authority directory",
        ),
        technical_authority_artifact_root_sha256=str(q["technical_authority_artifact_root_sha256"]),
        technical_authorization_sha256=str(q["technical_authorization_sha256"]),
        technical_authority_namespace_directory=_path_projection(
            published_binding["namespace_directory"],
            role="published technical authority namespace directory",
        ),
        technical_authority_namespace_claim_sha256=str(published_binding["namespace_claim_sha256"]),
        published_technical_authority_lifecycle_binding_sha256=str(
            published_binding["binding_sha256"]
        ),
        dataset_path=_path_projection(q["dataset_path"], role="Q dataset path"),
        manifest_path=_path_projection(q["manifest_path"], role="Q manifest path"),
        duplicate_audit_path=_path_projection(
            q["duplicate_audit_path"],
            role="Q duplicate audit path",
        ),
        pathology_encoder_audit_path=_path_projection(
            q["pathology_encoder_audit_path"],
            role="Q pathology encoder audit path",
        ),
        frozen_primary_config_path=_path_projection(
            q["frozen_primary_config_path"],
            role="Q frozen primary config path",
        ),
        frozen_confirmatory_config_path=_path_projection(
            q["frozen_confirmatory_config_path"],
            role="Q frozen confirmatory config path",
        ),
        crop_cache_path=_path_projection(
            cli["crop_cache_path"],
            role="Q crop cache path",
        ),
        expected_crop_cache_sha256=str(cli["expected_crop_cache_sha256"]),
        expected_crop_metadata_sha256=str(cli["expected_crop_metadata_sha256"]),
        expected_raw_inventory_sha256=str(cli["expected_raw_inventory_sha256"]),
        frozen_feature_caches=caches,
        observed_label_sets=observed,
        runs_root=_path_projection(q["runs_root"], role="Q runs root"),
        supervisor_job_id=str(e["job_id"]),
        attempt_id=str(e["attempt_id"]),
        run_id=str(e["run_id"]),
        expected_run_directory=_path_projection(
            e["expected_run_directory"],
            role="E expected run directory",
        ),
        retry_of_run_id=(str(e["retry_of_run_id"]) if e["retry_of_run_id"] is not None else None),
        resume_run_directory=(
            _require_closed_absolute_path(
                projection.predecessor_run_directory,
                role="E predecessor run directory",
            )
            if type(projection) is OriginalConfirmatorySuccessorPrecopyCheckpointProjection
            else None
        ),
        lifecycle_readiness_run_directory=_path_projection(
            q["lifecycle_readiness_run_directory"],
            role="Q lifecycle readiness directory",
        ),
        artifact_scope=str(e["artifact_scope"]),
        execution_mode=e["execution_mode"],
        q_static_runner_binding_sha256=str(q["binding_sha256"]),
        e_intent_core_sha256=e_intent_core_sha256,
        expected_plan_sha256=str(e["plan_sha256"]),
        expected_controls_binding_sha256=str(e["controls_binding_sha256"]),
        expected_bridge_binding_sha256=str(e["bridge_binding_sha256"]),
        expected_gate_evidence_sha256=str(e["gate_evidence_sha256"]),
        expected_cli_input_binding_sha256=str(e["cli_input_binding_sha256"]),
        scientific_request_projection_sha256=str(e["projection_sha256"]),
        canonical_e_projection=projection,
        draft_checkpoint_contract=draft,
    )
    return prepare_original_confirmatory_capsule_request(request)


_CAPSULE_REQUEST_TOMBSTONES: dict[int, OriginalConfirmatoryCapsuleExecutionRequest] = {}
_CAPSULE_REQUEST_TOMBSTONES_LOCK = threading.RLock()


def _claim_original_confirmatory_capsule_request(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
) -> None:
    """Burn one exact in-process request before any output can be written."""

    key = id(request)
    with _CAPSULE_REQUEST_TOMBSTONES_LOCK:
        existing = _CAPSULE_REQUEST_TOMBSTONES.get(key)
        if existing is request:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory capsule request was already consumed"
            )
        if existing is not None:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory capsule request identity collision is fail-closed"
            )
        _CAPSULE_REQUEST_TOMBSTONES[key] = request


def _run_original_confirmatory_capsule_request(
    request: OriginalConfirmatoryCapsuleExecutionRequest,
) -> dict[str, Any]:
    """Consume one Q/E request and execute the complete guarded study lifecycle."""

    if type(request) is not OriginalConfirmatoryCapsuleExecutionRequest:
        raise TypeError("run-confirmatory requires its exact capsule request type")
    request.validate()
    _claim_original_confirmatory_capsule_request(request)
    # Lazy import preserves the checkpoint data-plane dependency direction while
    # ensuring the handler has only this one high-level production entry.
    from histo_audit.experiment.confirmatory_runner import (
        _execute_original_confirmatory_capsule_lifecycle,
    )

    result = _execute_original_confirmatory_capsule_lifecycle(request)
    if type(result) is not dict:
        raise ConfirmatoryCheckpointContractError(
            "run-confirmatory lifecycle returned a noncanonical completion record"
        )
    return result


def _execute_original_confirmatory_prepared_matrix(
    request: _OriginalConfirmatoryPreparedMatrixRequest,
) -> ConfirmatoryMatrixArtifacts:
    """Execute the fixed matrix point prepared inside the full guarded lifecycle."""

    if type(request) is not _OriginalConfirmatoryPreparedMatrixRequest:
        raise TypeError("matrix execution requires its exact prepared request type")
    request.validate()
    draft = request.draft_checkpoint_contract
    projection = request.canonical_e_projection
    checkpoint_paths = sorted(
        f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
        for value in draft.directives
    )
    bound = replace(
        draft,
        authority_projection_sha256=(request.scientific_authority_projection_sha256),
        authority_predecessor_binding_sha256=(
            projection.predecessor_snapshot_sha256
            if projection.execution_mode == "successor_resume"
            else None
        ),
        authority_resume_adapter_sha256=(
            projection.predecessor_copy_receipt_sha256
            if projection.execution_mode == "successor_resume"
            else None
        ),
        authority_fit_directives_root_sha256=projection.directives_sha256,
        authority_checkpoint_allowlist_sha256=canonical_sha256(checkpoint_paths),
    )
    registered = _register_checkpoint_execution_contract(
        bound,
        expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    )
    lease = _lease_checkpoint_execution_contract_single_use(
        registered,
        expected_directive_count=ORIGINAL_CONFIRMATORY_CHECKPOINT_COUNT,
    )

    def execute_bound_contract(
        contract: ConfirmatoryCheckpointExecutionContract,
    ) -> ConfirmatoryMatrixArtifacts:
        result = execute_confirmatory_matrix(
            request.rotations,
            request.plan,
            request.controls,
            output_directory=request.output_directory,
            frozen_oof_runner=run_confirmatory_frozen_feature_oof,
            image_oof_runner=run_confirmatory_image_oof,
            frozen_blockers=dict(request.frozen_blockers),
            artifact_scope=request.artifact_scope,
            cpu_test_only=False,
            checkpoint_execution_contract=contract,
            gate_evidence=request.gate_evidence,
            progress_callback=None,
        )
        if type(result) is not ConfirmatoryMatrixArtifacts:
            raise ConfirmatoryCheckpointContractError(
                "run-confirmatory returned a noncanonical artifact record"
            )
        return result

    return lease.execute(execute_bound_contract)


__all__ = [
    "ORIGINAL_CONFIRMATORY_AUTHORITY_PROJECTION_POLICY",
    "ORIGINAL_CONFIRMATORY_CHECKPOINT_EXECUTION_CONTRACT_FILENAME",
    "ORIGINAL_CONFIRMATORY_FIT_DIRECTIVE_POLICY",
    "ORIGINAL_CONFIRMATORY_SUCCESSOR_COPY_POLICY",
    "ORIGINAL_CONFIRMATORY_SUCCESSOR_PRECOPY_POLICY",
    "MaterializedOriginalConfirmatorySuccessor",
    "OriginalConfirmatoryCanonicalECheckpointIdentity",
    "OriginalConfirmatoryCanonicalECheckpointProjection",
    "OriginalConfirmatoryCanonicalEDirectoryIdentity",
    "OriginalConfirmatoryCanonicalEFitDirective",
    "OriginalConfirmatoryCapsuleExecutionRequest",
    "OriginalConfirmatoryExecutionMode",
    "OriginalConfirmatoryScientificCheckpointAuthority",
    "OriginalConfirmatorySuccessorCopyReceipt",
    "OriginalConfirmatorySuccessorPrecopyCheckpointProjection",
    "OriginalConfirmatorySuccessorPrecopyFitDirective",
    "PreparedOriginalConfirmatorySuccessor",
    "build_original_confirmatory_canonical_e_checkpoint_projection",
    "build_original_confirmatory_capsule_request_from_authority",
    "build_original_confirmatory_fresh_checkpoint_execution_contract",
    "build_original_confirmatory_successor_checkpoint_execution_contract",
    "build_original_confirmatory_successor_precopy_checkpoint_projection",
    "fit_directive_authority_records",
    "fit_directives_root_sha256",
    "materialize_original_confirmatory_successor_checkpoint_execution",
    "prepare_original_confirmatory_capsule_request",
    "prepare_original_confirmatory_successor_checkpoint_execution",
    "require_original_confirmatory_canonical_e_projection",
]
