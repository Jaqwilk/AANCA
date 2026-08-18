"""Group-safe image OOF predictions for the confirmatory ResNet-18.

The wrapper deliberately accepts only audit-pool and clean
reference-validation images.  The untouched final-reference partition is
represented by group identifiers alone, so its pixels and labels cannot enter
training, early stopping, checkpointing, or prediction through this API.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import pickle
import re
import secrets
import stat
import threading
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from histo_audit.cross_validation.oof import (
    OOFFoldProvenance,
    OOFResult,
    make_group_stratified_folds,
)
from histo_audit.models.cnn import (
    CLASS_ORDER,
    CPU_TEST_ONLY_WEIGHT_IDENTIFIER,
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
    ConfirmatoryCNNCPUTestOnlyAdapter,
    ConfirmatoryResNet18Classifier,
    _IndexedRows,
    _json_sha256,
    _validate_adamw_state,
    _validate_checkpoint_state_dict,
    _validate_grad_scaler_state,
    _validate_history_and_telemetry,
    _validate_rng_state,
)

type ImageArray = NDArray[np.generic] | _IndexedRows
_LeaseResultT = TypeVar("_LeaseResultT")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FILE_ID_128 = re.compile(r"[0-9a-f]{32}")
_SAFE_CELL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DATA_AND_SPLIT_KEYS = {
    "training_data_sha256",
    "reference_validation_data_sha256",
    "training_split_sha256",
    "reference_validation_split_sha256",
}

CheckpointExecutionMode = Literal["fresh", "successor_resume"]
CheckpointContractProfile = Literal[
    "original_confirmatory_exact_180",
    "resource_bounded_confirmatory_exact_30",
    "cpu_test_only",
]
CheckpointFitAction = Literal[
    "fresh_fit",
    "resume_incomplete_fit",
    "restore_terminal_checkpoint_without_fit",
]


class ConfirmatoryCheckpointContractError(RuntimeError):
    """A run-level checkpoint directive violation that must never become a cell result."""


def _file_id_128(value: os.stat_result) -> str:
    """Return the frozen stat-derived 128-bit identity surrogate."""

    mask = (1 << 64) - 1
    return f"{int(value.st_dev) & mask:016x}{int(value.st_ino) & mask:016x}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_object(encoded: str, *, role: str) -> dict[str, Any]:
    if not isinstance(encoded, str) or not encoded:
        raise ConfirmatoryCheckpointContractError(f"{role} must be canonical JSON")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfirmatoryCheckpointContractError(f"{role} contains a duplicate key")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            encoded,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ConfirmatoryCheckpointContractError(f"{role} contains non-finite JSON: {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfirmatoryCheckpointContractError(f"{role} is not valid canonical JSON") from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != encoded:
        raise ConfirmatoryCheckpointContractError(f"{role} is not one canonical JSON object")
    return cast(dict[str, Any], decoded)


@dataclass(frozen=True, slots=True)
class ConfirmatoryCheckpointPhysicalIdentity:
    """Full no-follow identity used by the resume/copy boundary."""

    device: int
    inode: int
    size_bytes: int
    mode: int
    link_count: int
    modified_time_ns: int
    changed_time_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> ConfirmatoryCheckpointPhysicalIdentity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size_bytes=int(value.st_size),
            mode=int(value.st_mode),
            link_count=int(value.st_nlink),
            modified_time_ns=int(value.st_mtime_ns),
            changed_time_ns=int(value.st_ctime_ns),
        )

    @property
    def file_id_128(self) -> str:
        mask = (1 << 64) - 1
        return f"{self.device & mask:016x}{self.inode & mask:016x}"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def validate(self, *, expected_size_bytes: int | None = None) -> None:
        if (
            any(
                type(value) is not int
                for value in (
                    self.device,
                    self.inode,
                    self.size_bytes,
                    self.mode,
                    self.link_count,
                    self.modified_time_ns,
                    self.changed_time_ns,
                )
            )
            or self.device < 0
            or self.inode < 0
            or self.size_bytes <= 0
            or self.link_count != 1
            or not stat.S_ISREG(self.mode)
            or (expected_size_bytes is not None and self.size_bytes != expected_size_bytes)
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint physical identity is incomplete or non-private"
            )


@dataclass(frozen=True, slots=True)
class ConfirmatoryCheckpointFileIdentity:
    """One exact source or destination checkpoint identity."""

    path: Path
    physical_identity: ConfirmatoryCheckpointPhysicalIdentity
    size_bytes: int
    sha256: str

    @property
    def file_id_128(self) -> str:
        return self.physical_identity.file_id_128

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_id_128": self.file_id_128,
            "device": self.physical_identity.device,
            "inode": self.physical_identity.inode,
            "size_bytes": self.size_bytes,
            "mode": self.physical_identity.mode,
            "link_count": self.physical_identity.link_count,
            "modified_time_ns": self.physical_identity.modified_time_ns,
            "changed_time_ns": self.physical_identity.changed_time_ns,
            "sha256": self.sha256,
        }

    def validate(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or not isinstance(
                self.physical_identity,
                ConfirmatoryCheckpointPhysicalIdentity,
            )
            or _FILE_ID_128.fullmatch(self.file_id_128) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint file identity is incomplete or noncanonical"
            )
        self.physical_identity.validate(expected_size_bytes=self.size_bytes)


@dataclass(frozen=True, slots=True)
class ConfirmatoryCheckpointDirective:
    """One immutable cell/fold checkpoint action fixed before matrix execution."""

    execution_mode: CheckpointExecutionMode
    cell_id: str
    fold_id: int
    action: CheckpointFitAction
    source_predecessor_checkpoint: ConfirmatoryCheckpointFileIdentity | None
    destination_imported_checkpoint: ConfirmatoryCheckpointFileIdentity | None
    versioned_checkpoint_output_directory_relative_path: str | None
    checkpoint_execution_manifest_relative_path: str
    checkpoint_sha256: str | None
    checkpoint_size_bytes: int | None
    completed_epochs_before_fit: int
    stopped_early_before_fit: bool | None
    next_epoch_index: int
    maximum_epochs: int
    expected_configuration_json: str
    expected_configuration_sha256: str
    expected_model_metadata_json: str
    expected_model_metadata_sha256: str
    expected_data_and_split_json: str
    expected_data_and_split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_predecessor_checkpoint"] = (
            self.source_predecessor_checkpoint.as_dict()
            if self.source_predecessor_checkpoint is not None
            else None
        )
        payload["destination_imported_checkpoint"] = (
            self.destination_imported_checkpoint.as_dict()
            if self.destination_imported_checkpoint is not None
            else None
        )
        return payload

    @property
    def directive_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def validate(self) -> None:
        if self.execution_mode not in {"fresh", "successor_resume"}:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution mode is outside the closed union"
            )
        if (
            not isinstance(self.cell_id, str)
            or _SAFE_CELL_ID.fullmatch(self.cell_id) is None
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < 5
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive cell/fold identity is invalid"
            )
        expected_output_directory = (
            f"cells/{self.cell_id}/checkpoint_versions/fold_{self.fold_id:02d}"
        )
        expected_execution_manifest = (
            f"cells/{self.cell_id}/checkpoint_execution/fold_{self.fold_id:02d}.json"
        )
        if (
            self.checkpoint_execution_manifest_relative_path != expected_execution_manifest
            or (
                self.action == "restore_terminal_checkpoint_without_fit"
                and self.versioned_checkpoint_output_directory_relative_path is not None
            )
            or (
                self.action in {"fresh_fit", "resume_incomplete_fit"}
                and self.versioned_checkpoint_output_directory_relative_path
                != expected_output_directory
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive output/manifest path policy is noncanonical"
            )
        if self.action not in {
            "fresh_fit",
            "resume_incomplete_fit",
            "restore_terminal_checkpoint_without_fit",
        }:
            raise ConfirmatoryCheckpointContractError("checkpoint directive action is unsupported")
        if (
            type(self.maximum_epochs) is not int
            or self.maximum_epochs <= 0
            or type(self.completed_epochs_before_fit) is not int
            or type(self.next_epoch_index) is not int
            or (
                self.stopped_early_before_fit is not None
                and type(self.stopped_early_before_fit) is not bool
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive epoch state is not exact"
            )
        configuration = _strict_json_object(
            self.expected_configuration_json,
            role="checkpoint expected configuration",
        )
        metadata = _strict_json_object(
            self.expected_model_metadata_json,
            role="checkpoint expected model metadata",
        )
        data_and_split = _strict_json_object(
            self.expected_data_and_split_json,
            role="checkpoint expected data/split binding",
        )
        hashes = (
            (self.expected_configuration_sha256, configuration),
            (self.expected_model_metadata_sha256, metadata),
            (self.expected_data_and_split_sha256, data_and_split),
        )
        if any(
            _SHA256.fullmatch(digest) is None or _canonical_sha256(value) != digest
            for digest, value in hashes
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive canonical object hash is invalid"
            )
        if set(data_and_split) != _DATA_AND_SPLIT_KEYS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in data_and_split.values()
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive data/split binding is invalid"
            )
        if configuration.get("epochs") != self.maximum_epochs:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive maximum epochs differs from its configuration"
            )
        if self.action == "fresh_fit":
            if (
                self.checkpoint_sha256 is not None
                or self.checkpoint_size_bytes is not None
                or self.source_predecessor_checkpoint is not None
                or self.destination_imported_checkpoint is not None
                or self.completed_epochs_before_fit != 0
                or self.stopped_early_before_fit is not None
                or self.next_epoch_index != 0
            ):
                raise ConfirmatoryCheckpointContractError(
                    "fresh checkpoint directive carries predecessor state"
                )
            return
        if self.execution_mode != "successor_resume":
            raise ConfirmatoryCheckpointContractError(
                "resume/restore action is forbidden in fresh execution mode"
            )
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256.fullmatch(self.checkpoint_sha256) is None
            or type(self.checkpoint_size_bytes) is not int
            or self.checkpoint_size_bytes <= 0
            or not isinstance(
                self.source_predecessor_checkpoint,
                ConfirmatoryCheckpointFileIdentity,
            )
            or not isinstance(
                self.destination_imported_checkpoint,
                ConfirmatoryCheckpointFileIdentity,
            )
            or type(self.stopped_early_before_fit) is not bool
            or not 1 <= self.completed_epochs_before_fit <= self.maximum_epochs
            or self.next_epoch_index != self.completed_epochs_before_fit
        ):
            raise ConfirmatoryCheckpointContractError(
                "resume checkpoint directive lacks exact checkpoint/epoch state"
            )
        self.source_predecessor_checkpoint.validate()
        self.destination_imported_checkpoint.validate()
        if (
            self.source_predecessor_checkpoint.size_bytes != self.checkpoint_size_bytes
            or self.destination_imported_checkpoint.size_bytes != self.checkpoint_size_bytes
            or self.source_predecessor_checkpoint.sha256 != self.checkpoint_sha256
            or self.destination_imported_checkpoint.sha256 != self.checkpoint_sha256
            or os.path.normcase(str(self.source_predecessor_checkpoint.path))
            == os.path.normcase(str(self.destination_imported_checkpoint.path))
            or self.source_predecessor_checkpoint.file_id_128
            == self.destination_imported_checkpoint.file_id_128
        ):
            raise ConfirmatoryCheckpointContractError(
                "source/destination checkpoint identities are aliased or hash-mismatched"
            )
        terminal = (
            bool(self.stopped_early_before_fit)
            or self.completed_epochs_before_fit == self.maximum_epochs
        )
        if terminal != (self.action == "restore_terminal_checkpoint_without_fit"):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint directive terminal state differs from its action"
            )


_CHECKPOINT_EXECUTION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "policy",
        "fit_id",
        "fit_attempt",
        "action",
        "directive_sha256",
        "source_predecessor_checkpoint",
        "destination_imported_checkpoint",
        "imported_checkpoint_observed",
        "canonical_working_checkpoint",
        "canonical_working_checkpoint_read_only",
        "versioned_checkpoint_output_directory_relative_path",
        "checkpoint_execution_manifest_relative_path",
        "completed_epochs_before_fit",
        "completed_epochs_after_fit",
        "trained_epochs",
        "publication_boundary",
        "versioned_outputs",
        "final_checkpoint",
        "automatic_retry_allowed",
        "imported_checkpoint_modified",
        "hardlink_or_replace_used_for_immutable_publication",
        "mutable_latest_path_created",
    }
)
_CHECKPOINT_PUBLICATION_RECORD_KEYS = frozenset(
    {
        "publication_index",
        "completed_epochs",
        "checkpoint_relative_path",
        "checkpoint",
        "commit_manifest_relative_path",
        "commit_manifest_sha256",
        "commit_manifest_size_bytes",
        "commit_manifest_physical_identity",
    }
)
_CHECKPOINT_FILE_IDENTITY_KEYS = frozenset(
    {
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
)
_CHECKPOINT_PHYSICAL_IDENTITY_KEYS = frozenset(
    {
        "device",
        "inode",
        "size_bytes",
        "mode",
        "link_count",
        "modified_time_ns",
        "changed_time_ns",
    }
)


def _require_exact_mapping_keys(
    value: object,
    expected_keys: Collection[str],
    *,
    role: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ConfirmatoryCheckpointContractError(f"{role} schema is not exact")
    return cast(Mapping[str, Any], value)


def _checkpoint_physical_identity_from_exact_manifest(
    value: object,
    *,
    role: str,
) -> ConfirmatoryCheckpointPhysicalIdentity:
    payload = _require_exact_mapping_keys(
        value,
        _CHECKPOINT_PHYSICAL_IDENTITY_KEYS,
        role=role,
    )
    fields = (
        "device",
        "inode",
        "size_bytes",
        "mode",
        "link_count",
        "modified_time_ns",
        "changed_time_ns",
    )
    if any(type(payload[field]) is not int for field in fields):
        raise ConfirmatoryCheckpointContractError(f"{role} fields are not exact integers")
    identity = ConfirmatoryCheckpointPhysicalIdentity(
        device=payload["device"],
        inode=payload["inode"],
        size_bytes=payload["size_bytes"],
        mode=payload["mode"],
        link_count=payload["link_count"],
        modified_time_ns=payload["modified_time_ns"],
        changed_time_ns=payload["changed_time_ns"],
    )
    identity.validate()
    return identity


def _checkpoint_file_identity_from_exact_manifest(
    value: object,
    *,
    role: str,
) -> ConfirmatoryCheckpointFileIdentity:
    payload = _require_exact_mapping_keys(
        value,
        _CHECKPOINT_FILE_IDENTITY_KEYS,
        role=role,
    )
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
        not isinstance(payload["path"], str)
        or not payload["path"]
        or any(type(payload[field]) is not int for field in integer_fields)
        or not isinstance(payload["file_id_128"], str)
        or not isinstance(payload["sha256"], str)
    ):
        raise ConfirmatoryCheckpointContractError(f"{role} fields are not exact")
    identity = ConfirmatoryCheckpointPhysicalIdentity(
        device=payload["device"],
        inode=payload["inode"],
        size_bytes=payload["size_bytes"],
        mode=payload["mode"],
        link_count=payload["link_count"],
        modified_time_ns=payload["modified_time_ns"],
        changed_time_ns=payload["changed_time_ns"],
    )
    result = ConfirmatoryCheckpointFileIdentity(
        path=Path(payload["path"]),
        physical_identity=identity,
        size_bytes=payload["size_bytes"],
        sha256=payload["sha256"],
    )
    result.validate()
    if payload["file_id_128"] != result.file_id_128:
        raise ConfirmatoryCheckpointContractError(f"{role} file identity is inconsistent")
    return result


def _require_absolute_path_suffix(
    absolute_path: Path,
    relative_path: str,
    *,
    run_directory: Path,
    role: str,
) -> None:
    relative = Path(relative_path)
    expected = (run_directory / relative).resolve()
    if (
        not run_directory.is_absolute()
        or not absolute_path.is_absolute()
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or os.path.normcase(str(absolute_path.resolve())) != os.path.normcase(str(expected))
    ):
        raise ConfirmatoryCheckpointContractError(
            f"{role} path differs from its authorized relative path"
        )


def _require_exact_checkpoint_execution_manifest_payload(
    value: object,
    directive: ConfirmatoryCheckpointDirective,
    *,
    run_directory: Path,
) -> None:
    """Validate the complete immutable schema-v3 execution record against Q/E."""

    directive.validate()
    payload = _require_exact_mapping_keys(
        value,
        _CHECKPOINT_EXECUTION_MANIFEST_KEYS,
        role="checkpoint execution manifest",
    )
    if (
        payload["schema_version"] != 3
        or payload["policy"] != "aanca_fold_boundary_checkpoint_execution_v3"
        or payload["fit_id"] != f"{directive.cell_id}::fold_{directive.fold_id:02d}"
        or type(payload["fit_attempt"]) is not int
        or payload["fit_attempt"] != 1
        or payload["action"] != directive.action
        or payload["directive_sha256"] != directive.directive_sha256
        or payload["source_predecessor_checkpoint"]
        != (
            directive.source_predecessor_checkpoint.as_dict()
            if directive.source_predecessor_checkpoint is not None
            else None
        )
        or payload["destination_imported_checkpoint"]
        != (
            directive.destination_imported_checkpoint.as_dict()
            if directive.destination_imported_checkpoint is not None
            else None
        )
        or payload["versioned_checkpoint_output_directory_relative_path"]
        != directive.versioned_checkpoint_output_directory_relative_path
        or payload["checkpoint_execution_manifest_relative_path"]
        != directive.checkpoint_execution_manifest_relative_path
        or type(payload["completed_epochs_before_fit"]) is not int
        or payload["completed_epochs_before_fit"] != directive.completed_epochs_before_fit
        or type(payload["completed_epochs_after_fit"]) is not int
        or type(payload["trained_epochs"]) is not int
        or payload["automatic_retry_allowed"] is not False
        or payload["imported_checkpoint_modified"]
        is not (directive.action == "resume_incomplete_fit")
        or payload["hardlink_or_replace_used_for_immutable_publication"] is not False
        or payload["canonical_working_checkpoint_read_only"] is not True
        or payload["publication_boundary"] != "successful_fold_completion"
        or payload["mutable_latest_path_created"] is not False
        or not isinstance(payload["versioned_outputs"], list)
    ):
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution manifest differs from its exact directive"
        )

    imported = payload["imported_checkpoint_observed"]
    destination = directive.destination_imported_checkpoint
    if destination is None:
        if imported is not None:
            raise ConfirmatoryCheckpointContractError(
                "fresh checkpoint execution manifest reports an imported checkpoint"
            )
    else:
        _checkpoint_file_identity_from_exact_manifest(
            imported,
            role="checkpoint execution imported checkpoint",
        )
        if imported != destination.as_dict():
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution imported identity differs from its directive"
            )

    canonical_checkpoint = _checkpoint_file_identity_from_exact_manifest(
        payload["canonical_working_checkpoint"],
        role="checkpoint execution canonical working checkpoint",
    )
    canonical_relative = f"cells/{directive.cell_id}/checkpoints/fold_{directive.fold_id:02d}.pt"
    _require_absolute_path_suffix(
        canonical_checkpoint.path,
        canonical_relative,
        run_directory=run_directory,
        role="checkpoint execution canonical working checkpoint",
    )
    if stat.S_IMODE(canonical_checkpoint.physical_identity.mode) & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution canonical working checkpoint remains writable"
        )
    final_checkpoint = _checkpoint_file_identity_from_exact_manifest(
        payload["final_checkpoint"],
        role="checkpoint execution final checkpoint",
    )
    completed_after = payload["completed_epochs_after_fit"]
    trained_epochs = payload["trained_epochs"]
    publications = cast(list[object], payload["versioned_outputs"])
    if directive.action == "restore_terminal_checkpoint_without_fit":
        if (
            destination is None
            or completed_after != directive.completed_epochs_before_fit
            or trained_epochs != 0
            or publications
            or payload["versioned_checkpoint_output_directory_relative_path"] is not None
            or payload["final_checkpoint"] != payload["canonical_working_checkpoint"]
            or canonical_checkpoint.path != destination.path
            or canonical_checkpoint.sha256 != destination.sha256
            or canonical_checkpoint.size_bytes != destination.size_bytes
            or imported is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "terminal checkpoint execution manifest describes mutable work"
            )
        return

    if (
        trained_epochs <= 0
        or completed_after != directive.completed_epochs_before_fit + trained_epochs
        or len(publications) != 1
        or directive.versioned_checkpoint_output_directory_relative_path is None
    ):
        raise ConfirmatoryCheckpointContractError(
            "training checkpoint execution manifest has inconsistent epochs"
        )
    expected_final: Mapping[str, Any] | None = None
    for publication_index, raw_publication in enumerate(publications, start=1):
        publication = _require_exact_mapping_keys(
            raw_publication,
            _CHECKPOINT_PUBLICATION_RECORD_KEYS,
            role="checkpoint execution publication",
        )
        completed_epochs = completed_after
        checkpoint_relative = (
            f"{directive.versioned_checkpoint_output_directory_relative_path}/"
            f"epoch_{completed_epochs:04d}.pt"
        )
        commit_relative = (
            f"{directive.versioned_checkpoint_output_directory_relative_path}/"
            f"epoch_{completed_epochs:04d}.commit.json"
        )
        if (
            type(publication["publication_index"]) is not int
            or publication["publication_index"] != publication_index
            or type(publication["completed_epochs"]) is not int
            or publication["completed_epochs"] != completed_epochs
            or publication["checkpoint_relative_path"] != checkpoint_relative
            or publication["commit_manifest_relative_path"] != commit_relative
            or not isinstance(publication["commit_manifest_sha256"], str)
            or _SHA256.fullmatch(publication["commit_manifest_sha256"]) is None
            or type(publication["commit_manifest_size_bytes"]) is not int
            or publication["commit_manifest_size_bytes"] <= 0
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution publication differs from its exact epoch/path"
            )
        checkpoint_identity = _checkpoint_file_identity_from_exact_manifest(
            publication["checkpoint"],
            role="checkpoint execution versioned checkpoint",
        )
        _require_absolute_path_suffix(
            checkpoint_identity.path,
            checkpoint_relative,
            run_directory=run_directory,
            role="checkpoint execution versioned checkpoint",
        )
        commit_identity = _checkpoint_physical_identity_from_exact_manifest(
            publication["commit_manifest_physical_identity"],
            role="checkpoint execution commit manifest",
        )
        if commit_identity.size_bytes != publication["commit_manifest_size_bytes"]:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution commit-manifest size differs from its identity"
            )
        expected_final = publication["checkpoint"]
    if expected_final is None or payload["final_checkpoint"] != expected_final:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution final checkpoint is not the last versioned output"
        )
    expected_final_relative = (
        f"{directive.versioned_checkpoint_output_directory_relative_path}/"
        f"epoch_{completed_after:04d}.pt"
    )
    _require_absolute_path_suffix(
        final_checkpoint.path,
        expected_final_relative,
        run_directory=run_directory,
        role="checkpoint execution final checkpoint",
    )
    if (
        canonical_checkpoint.sha256 != final_checkpoint.sha256
        or canonical_checkpoint.size_bytes != final_checkpoint.size_bytes
        or canonical_checkpoint.file_id_128 == final_checkpoint.file_id_128
        or stat.S_IMODE(final_checkpoint.physical_identity.mode)
        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ConfirmatoryCheckpointContractError(
            "canonical and versioned final checkpoints differ, alias, or remain writable"
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ConfirmatoryCheckpointExecutionContract:
    """Closed original-confirmatory checkpoint data-plane contract."""

    execution_mode: CheckpointExecutionMode
    contract_profile: CheckpointContractProfile
    retry_of_run_id: str | None
    directives: tuple[ConfirmatoryCheckpointDirective, ...]
    directives_sha256: str
    predecessor_checkpoint_read_performed: bool
    predecessor_checkpoint_copy_performed: bool
    outcome_artifacts_read: bool
    automatic_retry_allowed: bool
    predecessor_snapshot_sha256: str | None = None
    predecessor_copy_receipt_sha256: str | None = None
    authority_projection_sha256: str | None = None
    authority_predecessor_binding_sha256: str | None = None
    authority_resume_adapter_sha256: str | None = None
    authority_fit_directives_root_sha256: str | None = None
    authority_checkpoint_allowlist_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "contract_profile": self.contract_profile,
            "retry_of_run_id": self.retry_of_run_id,
            "directive_count": len(self.directives),
            "directives_sha256": self.directives_sha256,
            "predecessor_checkpoint_read_performed": (self.predecessor_checkpoint_read_performed),
            "predecessor_checkpoint_copy_performed": (self.predecessor_checkpoint_copy_performed),
            "outcome_artifacts_read": self.outcome_artifacts_read,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "predecessor_snapshot_sha256": self.predecessor_snapshot_sha256,
            "predecessor_copy_receipt_sha256": (self.predecessor_copy_receipt_sha256),
            "authority_projection_sha256": self.authority_projection_sha256,
            "authority_predecessor_binding_sha256": (self.authority_predecessor_binding_sha256),
            "authority_resume_adapter_sha256": (self.authority_resume_adapter_sha256),
            "authority_fit_directives_root_sha256": (self.authority_fit_directives_root_sha256),
            "authority_checkpoint_allowlist_sha256": (self.authority_checkpoint_allowlist_sha256),
        }

    def _validate_payload(self, *, expected_directive_count: int) -> None:
        if self.execution_mode not in {"fresh", "successor_resume"}:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution mode is outside the closed union"
            )
        fixed_profile_count = {
            "original_confirmatory_exact_180": 180,
            "resource_bounded_confirmatory_exact_30": 30,
        }.get(self.contract_profile)
        if self.contract_profile not in {
            "original_confirmatory_exact_180",
            "resource_bounded_confirmatory_exact_30",
            "cpu_test_only",
        } or (fixed_profile_count is not None and expected_directive_count != fixed_profile_count):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint contract profile/count binding is invalid"
            )
        if (
            type(expected_directive_count) is not int
            or expected_directive_count <= 0
            or len(self.directives) != expected_directive_count
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution contract has incomplete directive coverage"
            )
        for directive in self.directives:
            if not isinstance(directive, ConfirmatoryCheckpointDirective):
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint execution contract contains an untyped directive"
                )
            directive.validate()
            if directive.execution_mode != self.execution_mode:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint directive execution mode differs from the contract"
                )
        identities = [(value.cell_id, value.fold_id) for value in self.directives]
        execution_manifest_paths = [
            value.checkpoint_execution_manifest_relative_path for value in self.directives
        ]
        output_directory_paths = [
            value.versioned_checkpoint_output_directory_relative_path
            for value in self.directives
            if value.versioned_checkpoint_output_directory_relative_path is not None
        ]
        if (
            len(set(identities)) != len(identities)
            or len(set(execution_manifest_paths)) != len(execution_manifest_paths)
            or len({path.casefold() for path in execution_manifest_paths})
            != len(execution_manifest_paths)
            or len(set(output_directory_paths)) != len(output_directory_paths)
            or len({path.casefold() for path in output_directory_paths})
            != len(output_directory_paths)
            or _canonical_sha256([value.as_dict() for value in self.directives])
            != self.directives_sha256
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution directives are duplicated or hash-mismatched"
            )
        if (
            self.outcome_artifacts_read
            or self.automatic_retry_allowed
            or type(self.predecessor_checkpoint_read_performed) is not bool
            or type(self.predecessor_checkpoint_copy_performed) is not bool
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution contract violates outcome-blind/no-retry policy"
            )
        authority_core_bindings = (
            self.authority_projection_sha256,
            self.authority_fit_directives_root_sha256,
            self.authority_checkpoint_allowlist_sha256,
        )
        authority_lineage_bindings = (
            self.authority_predecessor_binding_sha256,
            self.authority_resume_adapter_sha256,
        )
        if any(
            value is not None for value in authority_core_bindings + authority_lineage_bindings
        ) and (
            self.contract_profile != "original_confirmatory_exact_180"
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in authority_core_bindings
            )
            or (self.execution_mode == "fresh" and authority_lineage_bindings != (None, None))
            or (
                self.execution_mode == "successor_resume"
                and any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in authority_lineage_bindings
                )
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint authority binding is partial, malformed, or profile-mismatched"
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
                    "fresh execution carries predecessor lineage or a resume action"
                )
            return
        if (
            not isinstance(self.retry_of_run_id, str)
            or not self.retry_of_run_id
            or Path(self.retry_of_run_id).name != self.retry_of_run_id
            or not self.predecessor_checkpoint_read_performed
            or not self.predecessor_checkpoint_copy_performed
            or not isinstance(self.predecessor_snapshot_sha256, str)
            or _SHA256.fullmatch(self.predecessor_snapshot_sha256) is None
            or not isinstance(self.predecessor_copy_receipt_sha256, str)
            or _SHA256.fullmatch(self.predecessor_copy_receipt_sha256) is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "successor-resume execution lacks exact predecessor lineage"
            )

    def validate(self, *, expected_directive_count: int = 180) -> None:
        self._validate_payload(expected_directive_count=expected_directive_count)
        digest = _checkpoint_execution_contract_digest(self)
        with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
            registration = _CHECKPOINT_CONTRACT_REGISTRY.get(id(self))
        if registration is None or registration[0]() is not self or registration[1] != digest:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution contract was not derived by the sealed builder"
            )
        _require_checkpoint_execution_contract_lease(
            self,
            required=False,
        )

    def directives_for_cell(
        self,
        cell_id: str,
        *,
        expected_fold_count: int = 5,
    ) -> tuple[ConfirmatoryCheckpointDirective, ...]:
        values = tuple(value for value in self.directives if value.cell_id == cell_id)
        if len(values) != expected_fold_count or {value.fold_id for value in values} != set(
            range(expected_fold_count)
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint directives do not cover every fold for {cell_id}"
            )
        return tuple(sorted(values, key=lambda value: value.fold_id))


_CHECKPOINT_CONTRACT_REGISTRY: dict[
    int,
    tuple[weakref.ReferenceType[ConfirmatoryCheckpointExecutionContract], str],
] = {}
_CHECKPOINT_CONTRACT_REGISTRY_LOCK = threading.RLock()


@dataclass(slots=True)
class _CheckpointExecutionContractLeaseState:
    """Process-local one-use lease state; never serialized into scientific evidence."""

    lock: Any = field(default_factory=threading.RLock, repr=False)
    used: bool = False
    active: bool = False
    invalidated: bool = False
    owner_thread_id: int | None = None

    def activate(self) -> None:
        with self.lock:
            if self.used or self.active or self.invalidated:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint execution lease is already consumed or invalidated"
                )
            self.used = True
            self.active = True
            self.owner_thread_id = threading.get_ident()

    def require_active(self) -> None:
        with self.lock:
            if not self.active or self.invalidated or self.owner_thread_id != threading.get_ident():
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint execution contract lacks its active same-thread lease"
                )

    def close_after_execution(self) -> None:
        with self.lock:
            self.active = False
            self.invalidated = True
            self.owner_thread_id = None

    def invalidate(self) -> None:
        with self.lock:
            if self.active:
                raise ConfirmatoryCheckpointContractError(
                    "active checkpoint execution lease cannot be invalidated out of band"
                )
            self.used = True
            self.invalidated = True
            self.owner_thread_id = None


@dataclass(frozen=True, slots=True)
class _CheckpointExecutionContractLeaseRecord:
    """Module-owned lease record retained for the lifetime of this process."""

    contract: ConfirmatoryCheckpointExecutionContract
    state: _CheckpointExecutionContractLeaseState
    expected_directive_count: int


_CHECKPOINT_CONTRACT_LEASE_RECORDS: dict[
    str,
    _CheckpointExecutionContractLeaseRecord,
] = {}
_CHECKPOINT_CONTRACT_LEASE_REGISTRY: dict[
    int,
    _CheckpointExecutionContractLeaseRecord,
] = {}


class _CheckpointExecutionContractLease:
    """Token-only one-use callback surface with no raw contract/state attribute."""

    __token: str
    __slots__ = ("__token",)

    def __init__(self, token: str) -> None:
        object.__setattr__(self, "_CheckpointExecutionContractLease__token", token)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("checkpoint execution lease facade is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("checkpoint execution lease facade is immutable")

    def execute(
        self,
        callback: Callable[[ConfirmatoryCheckpointExecutionContract], _LeaseResultT],
    ) -> _LeaseResultT:
        """Run exactly one same-thread consumer while the contract lease is active."""

        return _execute_checkpoint_execution_contract_lease(self.__token, callback)

    def invalidate(self) -> None:
        """Permanently close an unused lease after a pre-execution STOP."""

        _invalidate_checkpoint_execution_contract_lease(self.__token)


def _execute_checkpoint_execution_contract_lease[CheckpointLeaseResultT](
    token: str,
    callback: Callable[
        [ConfirmatoryCheckpointExecutionContract],
        CheckpointLeaseResultT,
    ],
) -> CheckpointLeaseResultT:
    """Execute a token-bound lease without projecting its record onto the facade."""

    if not callable(callback):
        raise TypeError("checkpoint execution lease callback must be callable")
    with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
        record = _CHECKPOINT_CONTRACT_LEASE_RECORDS.get(token)
    if record is None:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution lease token is unknown or invalid"
        )
    record.state.activate()
    try:
        record.contract.validate(
            expected_directive_count=record.expected_directive_count,
        )
        return callback(record.contract)
    finally:
        record.state.close_after_execution()


def _invalidate_checkpoint_execution_contract_lease(token: str) -> None:
    """Invalidate an unused token-bound lease without exposing its mutable state."""

    with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
        record = _CHECKPOINT_CONTRACT_LEASE_RECORDS.get(token)
    if record is None:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution lease token is unknown or invalid"
        )
    record.state.invalidate()


def _require_checkpoint_execution_contract_lease(
    contract: ConfirmatoryCheckpointExecutionContract,
    *,
    required: bool,
) -> None:
    with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
        record = _CHECKPOINT_CONTRACT_LEASE_REGISTRY.get(id(contract))
    if record is None or record.contract is not contract:
        if required:
            raise ConfirmatoryCheckpointContractError(
                "original confirmatory checkpoint contract lacks its single-use execution lease"
            )
        return
    record.state.require_active()


def _lease_checkpoint_execution_contract_single_use(
    contract: ConfirmatoryCheckpointExecutionContract,
    *,
    expected_directive_count: int = 180,
) -> _CheckpointExecutionContractLease:
    """Attach one inactive process-local lease to an exact authority-bound contract."""

    contract.validate(expected_directive_count=expected_directive_count)
    core_bindings = (
        contract.authority_projection_sha256,
        contract.authority_fit_directives_root_sha256,
        contract.authority_checkpoint_allowlist_sha256,
    )
    lineage_bindings = (
        contract.authority_predecessor_binding_sha256,
        contract.authority_resume_adapter_sha256,
    )
    if (
        contract.contract_profile != "original_confirmatory_exact_180"
        or expected_directive_count != 180
        or any(value is None for value in core_bindings)
        or (contract.execution_mode == "fresh" and lineage_bindings != (None, None))
        or (
            contract.execution_mode == "successor_resume"
            and any(value is None for value in lineage_bindings)
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "single-use checkpoint execution lease requires one exact authority-bound contract"
        )
    state = _CheckpointExecutionContractLeaseState()
    with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
        existing = _CHECKPOINT_CONTRACT_LEASE_REGISTRY.get(id(contract))
        if existing is not None:
            if existing.contract is contract:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint execution contract already has a single-use lease"
                )
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution contract identity collision is fail-closed"
            )
        token = secrets.token_hex(32)
        while token in _CHECKPOINT_CONTRACT_LEASE_RECORDS:
            token = secrets.token_hex(32)
        record = _CheckpointExecutionContractLeaseRecord(
            contract=contract,
            state=state,
            expected_directive_count=expected_directive_count,
        )
        _CHECKPOINT_CONTRACT_LEASE_REGISTRY[id(contract)] = record
        _CHECKPOINT_CONTRACT_LEASE_RECORDS[token] = record
    return _CheckpointExecutionContractLease(token)


def _checkpoint_execution_contract_digest(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> str:
    return _canonical_sha256(
        {
            "contract": contract.as_dict(),
            "directives": [value.as_dict() for value in contract.directives],
        }
    )


def _register_checkpoint_execution_contract(
    contract: ConfirmatoryCheckpointExecutionContract,
    *,
    expected_directive_count: int = 180,
) -> ConfirmatoryCheckpointExecutionContract:
    """Register one builder-derived immutable contract for same-process consumption."""

    if not isinstance(contract, ConfirmatoryCheckpointExecutionContract):
        raise ConfirmatoryCheckpointContractError(
            "checkpoint execution builder returned an untyped contract"
        )
    contract._validate_payload(expected_directive_count=expected_directive_count)
    digest = _checkpoint_execution_contract_digest(contract)
    with _CHECKPOINT_CONTRACT_REGISTRY_LOCK:
        _CHECKPOINT_CONTRACT_REGISTRY[id(contract)] = (weakref.ref(contract), digest)
    contract.validate(expected_directive_count=expected_directive_count)
    return contract


def require_original_confirmatory_checkpoint_authority_binding(
    contract: ConfirmatoryCheckpointExecutionContract,
) -> ConfirmatoryCheckpointExecutionContract:
    """Reject an exact-180 draft before any confirmatory matrix work begins."""

    if not isinstance(contract, ConfirmatoryCheckpointExecutionContract):
        raise ConfirmatoryCheckpointContractError(
            "original confirmatory execution requires a typed checkpoint contract"
        )
    contract.validate(expected_directive_count=180)
    core_bindings = (
        contract.authority_projection_sha256,
        contract.authority_fit_directives_root_sha256,
        contract.authority_checkpoint_allowlist_sha256,
    )
    lineage_bindings = (
        contract.authority_predecessor_binding_sha256,
        contract.authority_resume_adapter_sha256,
    )
    if (
        contract.contract_profile != "original_confirmatory_exact_180"
        or any(value is None for value in core_bindings)
        or (contract.execution_mode == "fresh" and lineage_bindings != (None, None))
        or (
            contract.execution_mode == "successor_resume"
            and any(value is None for value in lineage_bindings)
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "original confirmatory execution rejects an unbound checkpoint-authority draft"
        )
    _require_checkpoint_execution_contract_lease(
        contract,
        required=True,
    )
    return contract


@dataclass(frozen=True, slots=True)
class ConfirmatoryImageOOFFoldEvidence:
    """Checkpoint, split, and execution evidence for one image OOF fold."""

    fold_id: int
    model_seed: int
    training_sample_ids: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]
    training_groups: tuple[str, ...]
    held_out_groups: tuple[str, ...]
    reference_validation_sample_ids: tuple[str, ...]
    reference_validation_groups: tuple[str, ...]
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_physical_identity: ConfirmatoryCheckpointPhysicalIdentity
    checkpoint_execution_manifest_path: str
    checkpoint_execution_manifest_sha256: str
    checkpoint_execution_manifest_physical_identity: ConfirmatoryCheckpointPhysicalIdentity
    configuration_sha256: str
    resumed_from_checkpoint: bool
    checkpoint_execution_mode: CheckpointExecutionMode
    checkpoint_action: CheckpointFitAction
    checkpoint_sha256_before_fit: str | None
    completed_epochs_before_fit: int
    trained_epochs_this_invocation: int
    successful_optimiser_steps_before_fit: int
    successful_optimiser_steps_after_fit: int
    successful_optimiser_steps_this_invocation: int
    execution_mode: str
    study_outcome_eligible: bool
    completed_epochs: int
    best_epoch: int | None
    best_reference_validation_loss: float | None
    telemetry: dict[str, Any]
    model_metadata: dict[str, Any]
    data_and_split_sha256: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfirmatoryImageOOFResult:
    """Image OOF probabilities plus auditable per-fold training evidence."""

    oof_result: OOFResult
    fold_evidence: tuple[ConfirmatoryImageOOFFoldEvidence, ...]
    study_outcome_eligible: bool
    execution_mode: str

    def validate(self) -> None:
        """Reject inconsistent OOF, checkpoint, or eligibility evidence."""

        self.oof_result.validate()
        if len(self.fold_evidence) != len(self.oof_result.folds):
            raise ValueError("image OOF fold evidence does not align with OOF folds")
        evidence_by_fold = {value.fold_id: value for value in self.fold_evidence}
        if len(evidence_by_fold) != len(self.fold_evidence):
            raise ValueError("image OOF fold evidence contains duplicate fold IDs")
        for fold in self.oof_result.folds:
            evidence = evidence_by_fold.get(fold.fold_id)
            if evidence is None:
                raise ValueError(f"missing image OOF evidence for fold {fold.fold_id}")
            if evidence.training_groups != fold.training_groups:
                raise ValueError(f"training-group evidence differs for fold {fold.fold_id}")
            if evidence.held_out_groups != fold.held_out_groups:
                raise ValueError(f"holdout-group evidence differs for fold {fold.fold_id}")
            if evidence.held_out_sample_ids != fold.held_out_sample_ids:
                raise ValueError(f"holdout-sample evidence differs for fold {fold.fold_id}")
            if set(evidence.reference_validation_groups).intersection(
                evidence.training_groups + evidence.held_out_groups
            ):
                raise ValueError(
                    f"reference-validation group leakage in image OOF fold {fold.fold_id}"
                )
            if len(evidence.checkpoint_sha256) != 64:
                raise ValueError(f"invalid checkpoint SHA-256 for fold {fold.fold_id}")
            if (
                type(evidence.checkpoint_size_bytes) is not int
                or evidence.checkpoint_size_bytes <= 0
            ):
                raise ValueError(f"invalid checkpoint size for fold {fold.fold_id}")
            try:
                evidence.checkpoint_physical_identity.validate(
                    expected_size_bytes=evidence.checkpoint_size_bytes
                )
                evidence.checkpoint_execution_manifest_physical_identity.validate()
            except ConfirmatoryCheckpointContractError as exc:
                raise ValueError(
                    f"invalid checkpoint physical identity for fold {fold.fold_id}"
                ) from exc
            if (
                not isinstance(
                    evidence.checkpoint_execution_manifest_path,
                    str,
                )
                or not evidence.checkpoint_execution_manifest_path
                or not isinstance(
                    evidence.checkpoint_execution_manifest_sha256,
                    str,
                )
                or _SHA256.fullmatch(evidence.checkpoint_execution_manifest_sha256) is None
            ):
                raise ValueError(
                    f"missing immutable checkpoint commit manifest for fold {fold.fold_id}"
                )
            if len(evidence.configuration_sha256) != 64:
                raise ValueError(f"invalid configuration SHA-256 for fold {fold.fold_id}")
            if evidence.checkpoint_execution_mode not in {"fresh", "successor_resume"}:
                raise ValueError(f"invalid checkpoint execution mode for fold {fold.fold_id}")
            if evidence.checkpoint_action not in {
                "fresh_fit",
                "resume_incomplete_fit",
                "restore_terminal_checkpoint_without_fit",
            }:
                raise ValueError(f"invalid checkpoint action for fold {fold.fold_id}")
            if (
                type(evidence.completed_epochs_before_fit) is not int
                or evidence.completed_epochs_before_fit < 0
                or type(evidence.trained_epochs_this_invocation) is not int
                or evidence.trained_epochs_this_invocation < 0
                or evidence.completed_epochs
                != (evidence.completed_epochs_before_fit + evidence.trained_epochs_this_invocation)
                or type(evidence.successful_optimiser_steps_before_fit) is not int
                or type(evidence.successful_optimiser_steps_after_fit) is not int
                or type(evidence.successful_optimiser_steps_this_invocation) is not int
                or evidence.successful_optimiser_steps_this_invocation < 0
                or evidence.successful_optimiser_steps_after_fit
                != (
                    evidence.successful_optimiser_steps_before_fit
                    + evidence.successful_optimiser_steps_this_invocation
                )
            ):
                raise ValueError(
                    f"invalid checkpoint continuation evidence for fold {fold.fold_id}"
                )
            if evidence.checkpoint_action == "fresh_fit":
                if (
                    evidence.resumed_from_checkpoint
                    or evidence.checkpoint_sha256_before_fit is not None
                    or evidence.completed_epochs_before_fit != 0
                    or (
                        evidence.checkpoint_execution_mode == "fresh"
                        and evidence.trained_epochs_this_invocation == 0
                    )
                ):
                    raise ValueError(f"invalid fresh checkpoint evidence for fold {fold.fold_id}")
            else:
                if (
                    not evidence.resumed_from_checkpoint
                    or not isinstance(evidence.checkpoint_sha256_before_fit, str)
                    or _SHA256.fullmatch(evidence.checkpoint_sha256_before_fit) is None
                ):
                    raise ValueError(f"invalid resume checkpoint evidence for fold {fold.fold_id}")
            if evidence.checkpoint_action == "resume_incomplete_fit" and (
                evidence.trained_epochs_this_invocation <= 0
                or evidence.successful_optimiser_steps_this_invocation <= 0
            ):
                raise ValueError(
                    f"incomplete checkpoint did not continue training for fold {fold.fold_id}"
                )
            if evidence.checkpoint_action == "restore_terminal_checkpoint_without_fit" and (
                evidence.trained_epochs_this_invocation != 0
                or evidence.successful_optimiser_steps_this_invocation != 0
                or evidence.checkpoint_sha256_before_fit != evidence.checkpoint_sha256
            ):
                raise ValueError(f"terminal checkpoint changed or trained for fold {fold.fold_id}")
            hashes = evidence.data_and_split_sha256
            if set(hashes) != {
                "training_data_sha256",
                "reference_validation_data_sha256",
                "training_split_sha256",
                "reference_validation_split_sha256",
            } or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            ):
                raise ValueError(f"invalid data/split checkpoint bindings for fold {fold.fold_id}")
        eligible = all(value.study_outcome_eligible for value in self.fold_evidence)
        if self.study_outcome_eligible != eligible:
            raise ValueError("overall image OOF eligibility differs from fold evidence")
        if self.execution_mode == "cpu_test_only_non_evidence" and self.study_outcome_eligible:
            raise ValueError("CPU test-only image OOF output cannot be study-outcome eligible")


def _integer_vector(
    values: Sequence[int] | NDArray[np.integer],
    n: int,
    *,
    name: str,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.shape != (n,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an aligned one-dimensional integer vector")
    result = raw.astype(np.int64, copy=True)
    outside = sorted(set(int(value) for value in result).difference(CLASS_ORDER))
    if outside:
        raise ValueError(f"{name} contains labels outside fixed class order: {outside}")
    return result


def _identifier_vector(
    values: Sequence[str] | NDArray[np.str_],
    n: int,
    *,
    name: str,
    unique: bool,
) -> NDArray[np.str_]:
    result = np.asarray(values, dtype=np.str_)
    if result.shape != (n,) or any(not str(value) for value in result):
        raise ValueError(f"{name} must be non-empty strings aligned with images")
    if unique and len(set(result.tolist())) != n:
        raise ValueError(f"{name} must be unique")
    return result


def _image_count(images: ImageArray, *, name: str) -> int:
    if isinstance(images, _IndexedRows):
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] != 3:
            raise ValueError(f"{name} must have non-empty shape (n, height, width, 3)")
        return int(images.shape[0])
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[0] == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have non-empty shape (n, height, width, 3)")
    return int(array.shape[0])


def _select_image_rows(
    values: ImageArray,
    indices: NDArray[np.int64],
) -> ImageArray:
    if isinstance(values, _IndexedRows):
        return values.select_rows(indices)
    return np.asarray(values)[indices]


def _configuration_sha256(config: ConfirmatoryCNNConfig) -> str:
    payload = json.dumps(
        asdict(config),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment_sha256(labels: NDArray[np.int64]) -> str:
    payload = json.dumps(
        [int(value) for value in labels],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_reparse(value: os.stat_result) -> bool:
    return bool(
        int(getattr(value, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


if os.name == "nt":

    class _Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * (260 + 36)),
        ]

    class _Win32FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _Win32FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _Win32FileId128),
        ]

    class _Win32FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", ctypes.c_uint32),
            ("reparse_tag", ctypes.c_uint32),
        ]

    class _Win32FileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("creation_time", ctypes.c_longlong),
            ("last_access_time", ctypes.c_longlong),
            ("last_write_time", ctypes.c_longlong),
            ("change_time", ctypes.c_longlong),
            ("file_attributes", ctypes.c_uint32),
        ]


@dataclass(frozen=True, slots=True)
class _NativeCheckpointIdentity:
    """Native Windows volume plus the complete 128-bit FILE_ID_INFO identity."""

    volume_serial_number: int
    file_id_128: str
    file_attributes: int
    reparse_tag: int


_WIN_GENERIC_READ = 0x80000000
_WIN_GENERIC_WRITE = 0x40000000
_WIN_DELETE = 0x00010000
_WIN_FILE_WRITE_ATTRIBUTES = 0x00000100
_WIN_FILE_SHARE_READ = 0x00000001
_WIN_FILE_SHARE_WRITE = 0x00000002
_WIN_FILE_SHARE_DELETE = 0x00000004
_WIN_CREATE_NEW = 1
_WIN_OPEN_EXISTING = 3
_WIN_FILE_ATTRIBUTE_READONLY = 0x00000001
_WIN_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_WIN_FILE_ATTRIBUTE_NORMAL = 0x00000080
_WIN_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_WIN_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WIN_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WIN_FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
_WIN_FILE_FLAG_WRITE_THROUGH = 0x80000000
_WIN_FILE_BASIC_INFO = 0
_WIN_FILE_ATTRIBUTE_TAG_INFO = 9
_WIN_FILE_ID_INFO = 18


def _win32_close_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def _win32_native_identity(handle: int) -> _NativeCheckpointIdentity:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    file_id = _Win32FileIdInfo()
    if not get_information(
        ctypes.c_void_p(handle),
        _WIN_FILE_ID_INFO,
        ctypes.byref(file_id),
        ctypes.sizeof(file_id),
    ):
        raise OSError(ctypes.get_last_error(), "FileIdInfo readback failed")
    attributes = _Win32FileAttributeTagInfo()
    if not get_information(
        ctypes.c_void_p(handle),
        _WIN_FILE_ATTRIBUTE_TAG_INFO,
        ctypes.byref(attributes),
        ctypes.sizeof(attributes),
    ):
        raise OSError(ctypes.get_last_error(), "FileAttributeTagInfo readback failed")
    return _NativeCheckpointIdentity(
        volume_serial_number=int(file_id.volume_serial_number),
        file_id_128=bytes(file_id.file_id.identifier).hex(),
        file_attributes=int(attributes.file_attributes),
        reparse_tag=int(attributes.reparse_tag),
    )


def _win32_open_handle(
    path: Path,
    *,
    desired_access: int,
    creation_disposition: int,
    flags_and_attributes: int,
    share_mode: int = _WIN_FILE_SHARE_READ,
) -> int:
    """Open one leaf with no reparse traversal and no WRITE/DELETE sharing."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        desired_access,
        share_mode,
        None,
        creation_disposition,
        flags_and_attributes | _WIN_FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in {None, invalid}:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed: {path}")
    return int(handle)


def _win32_descriptor_from_handle(handle: int, *, writable: bool) -> int:
    msvcrt = __import__("msvcrt")
    flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
    return int(msvcrt.open_osfhandle(handle, flags))


def _win32_open_existing_checkpoint_descriptor(
    path: Path,
    *,
    write_attributes: bool,
) -> tuple[int, _NativeCheckpointIdentity]:
    access = _WIN_GENERIC_READ
    if write_attributes:
        access |= _WIN_FILE_WRITE_ATTRIBUTES
    handle = _win32_open_handle(
        path,
        desired_access=access,
        creation_disposition=_WIN_OPEN_EXISTING,
        flags_and_attributes=0,
    )
    try:
        native = _win32_native_identity(handle)
        if native.file_attributes & (
            _WIN_FILE_ATTRIBUTE_REPARSE_POINT | _WIN_FILE_ATTRIBUTE_DIRECTORY
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint leaf is a reparse point or directory: {path}"
            )
        return (
            _win32_descriptor_from_handle(handle, writable=False),
            native,
        )
    except BaseException:
        with suppress(OSError):
            _win32_close_handle(handle)
        raise


def _win32_create_checkpoint_descriptor(
    path: Path,
) -> tuple[int, _NativeCheckpointIdentity]:
    handle = _win32_open_handle(
        path,
        desired_access=(_WIN_GENERIC_READ | _WIN_GENERIC_WRITE | _WIN_FILE_WRITE_ATTRIBUTES),
        creation_disposition=_WIN_CREATE_NEW,
        flags_and_attributes=(_WIN_FILE_ATTRIBUTE_NORMAL | _WIN_FILE_FLAG_WRITE_THROUGH),
    )
    try:
        native = _win32_native_identity(handle)
        if native.file_attributes & (
            _WIN_FILE_ATTRIBUTE_REPARSE_POINT | _WIN_FILE_ATTRIBUTE_DIRECTORY
        ):
            raise ConfirmatoryCheckpointContractError(
                f"new checkpoint leaf is a reparse point or directory: {path}"
            )
        return _win32_descriptor_from_handle(handle, writable=True), native
    except BaseException:
        with suppress(OSError):
            _win32_close_handle(handle)
        raise


def _win32_create_owner_lock_descriptor(
    path: Path,
) -> tuple[int, _NativeCheckpointIdentity]:
    handle = _win32_open_handle(
        path,
        desired_access=_WIN_GENERIC_READ | _WIN_GENERIC_WRITE | _WIN_DELETE,
        creation_disposition=_WIN_CREATE_NEW,
        flags_and_attributes=(
            _WIN_FILE_ATTRIBUTE_NORMAL
            | _WIN_FILE_FLAG_WRITE_THROUGH
            | _WIN_FILE_FLAG_DELETE_ON_CLOSE
        ),
        share_mode=_WIN_FILE_SHARE_READ,
    )
    try:
        native = _win32_native_identity(handle)
        if native.file_attributes & (
            _WIN_FILE_ATTRIBUTE_REPARSE_POINT | _WIN_FILE_ATTRIBUTE_DIRECTORY
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint owner lock is a reparse point or directory: {path}"
            )
        return _win32_descriptor_from_handle(handle, writable=True), native
    except BaseException:
        with suppress(OSError):
            _win32_close_handle(handle)
        raise


def _win32_open_directory_handle(path: Path) -> tuple[int, _NativeCheckpointIdentity]:
    handle = _win32_open_handle(
        path,
        desired_access=_WIN_GENERIC_READ,
        creation_disposition=_WIN_OPEN_EXISTING,
        flags_and_attributes=_WIN_FILE_FLAG_BACKUP_SEMANTICS,
        share_mode=_WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE,
    )
    try:
        native = _win32_native_identity(handle)
        if (
            not native.file_attributes & _WIN_FILE_ATTRIBUTE_DIRECTORY
            or native.file_attributes & _WIN_FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint ancestor is not a plain directory: {path}"
            )
        return handle, native
    except BaseException:
        with suppress(OSError):
            _win32_close_handle(handle)
        raise


def _win32_native_directory_identity_from_path(
    path: Path,
) -> _NativeCheckpointIdentity:
    handle, native = _win32_open_directory_handle(path)
    try:
        return native
    finally:
        _win32_close_handle(handle)


def _win32_native_identity_with_share_mode(
    path: Path,
    *,
    share_mode: int,
) -> _NativeCheckpointIdentity:
    handle = _win32_open_handle(
        path,
        desired_access=_WIN_GENERIC_READ,
        creation_disposition=_WIN_OPEN_EXISTING,
        flags_and_attributes=0,
        share_mode=share_mode,
    )
    try:
        return _win32_native_identity(handle)
    finally:
        _win32_close_handle(handle)


def _win32_native_identity_from_path(path: Path) -> _NativeCheckpointIdentity:
    """Inspect an ordinary path without allowing a writer or deleter."""

    return _win32_native_identity_with_share_mode(
        path,
        share_mode=_WIN_FILE_SHARE_READ,
    )


def _win32_native_identity_from_live_writer_path(
    path: Path,
) -> _NativeCheckpointIdentity:
    """Inspect a path while its retained creator/writer handle remains live."""

    return _win32_native_identity_with_share_mode(
        path,
        share_mode=_WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE,
    )


def _win32_native_identity_from_owner_lock_path(
    path: Path,
) -> _NativeCheckpointIdentity:
    """Inspect only the role-bound DELETE_ON_CLOSE checkpoint owner lock."""

    return _win32_native_identity_with_share_mode(
        path,
        share_mode=(_WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE | _WIN_FILE_SHARE_DELETE),
    )


def _win32_native_identity_from_descriptor(
    descriptor: int,
) -> _NativeCheckpointIdentity:
    msvcrt = __import__("msvcrt")
    return _win32_native_identity(int(msvcrt.get_osfhandle(descriptor)))


def _win32_set_read_only_on_descriptor(descriptor: int) -> None:
    msvcrt = __import__("msvcrt")
    handle = int(msvcrt.get_osfhandle(descriptor))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    basic = _Win32FileBasicInfo()
    if not get_information(
        ctypes.c_void_p(handle),
        _WIN_FILE_BASIC_INFO,
        ctypes.byref(basic),
        ctypes.sizeof(basic),
    ):
        raise OSError(ctypes.get_last_error(), "FileBasicInfo readback failed")
    basic.file_attributes = (
        int(basic.file_attributes) | _WIN_FILE_ATTRIBUTE_READONLY
    ) & ~_WIN_FILE_ATTRIBUTE_NORMAL
    if not set_information(
        ctypes.c_void_p(handle),
        _WIN_FILE_BASIC_INFO,
        ctypes.byref(basic),
        ctypes.sizeof(basic),
    ):
        raise OSError(ctypes.get_last_error(), "read-only FileBasicInfo update failed")


@dataclass(slots=True)
class _DirectoryCustody:
    """A retained no-reparse ancestor handle that denies directory deletion."""

    path: Path
    role: str
    lexical_identity: ConfirmatoryCheckpointPhysicalIdentity
    native_identity: _NativeCheckpointIdentity | None
    native_handle: int | None = None
    descriptor: int | None = None

    @classmethod
    def acquire(cls, path: Path, *, role: str) -> _DirectoryCustody:
        absolute = Path(os.path.abspath(path))
        native_handle: int | None = None
        descriptor: int | None = None
        try:
            lexical = absolute.lstat()
            if (
                not stat.S_ISDIR(lexical.st_mode)
                or _is_reparse(lexical)
                or int(lexical.st_nlink) < 1
                or _named_streams(absolute)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} is not one plain stream-free directory"
                )
            if os.name == "nt":
                native_handle, native = _win32_open_directory_handle(absolute)
                path_native = _win32_native_directory_identity_from_path(absolute)
                if not _same_native_file(native, path_native):
                    raise ConfirmatoryCheckpointContractError(
                        f"{role} changed while native custody was acquired"
                    )
            else:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                )
                descriptor = os.open(absolute, flags)
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or int(opened.st_dev) != int(lexical.st_dev)
                    or int(opened.st_ino) != int(lexical.st_ino)
                ):
                    raise ConfirmatoryCheckpointContractError(
                        f"{role} changed while descriptor custody was acquired"
                    )
                native = None
            custody = cls(
                path=absolute,
                role=role,
                lexical_identity=_stat_identity(lexical),
                native_identity=native,
                native_handle=native_handle,
                descriptor=descriptor,
            )
            custody.verify()
            return custody
        except BaseException:
            if native_handle is not None:
                with suppress(OSError):
                    _win32_close_handle(native_handle)
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            raise

    def verify(self) -> None:
        try:
            lexical = self.path.lstat()
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                f"{self.role} disappeared while under retained custody"
            ) from exc
        observed = _stat_identity(lexical)
        if (
            not stat.S_ISDIR(lexical.st_mode)
            or _is_reparse(lexical)
            or int(lexical.st_nlink) < 1
            or (
                observed.device,
                observed.inode,
                stat.S_IMODE(observed.mode),
            )
            != (
                self.lexical_identity.device,
                self.lexical_identity.inode,
                stat.S_IMODE(self.lexical_identity.mode),
            )
            or _named_streams(self.path)
        ):
            raise ConfirmatoryCheckpointContractError(
                f"{self.role} changed while under retained custody"
            )
        if os.name == "nt":
            if self.native_handle is None or self.native_identity is None:
                raise ConfirmatoryCheckpointContractError(
                    f"{self.role} lacks its native custody handle"
                )
            handle_native = _win32_native_identity(self.native_handle)
            path_native = _win32_native_directory_identity_from_path(self.path)
            if (
                not _same_native_file(handle_native, self.native_identity)
                or not _same_native_file(path_native, self.native_identity)
                or not handle_native.file_attributes & _WIN_FILE_ATTRIBUTE_DIRECTORY
                or handle_native.file_attributes & _WIN_FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{self.role} native FileId/volume custody changed"
                )
        else:
            if self.descriptor is None:
                raise ConfirmatoryCheckpointContractError(
                    f"{self.role} lacks its directory descriptor"
                )
            opened = os.fstat(self.descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or int(opened.st_dev) != self.lexical_identity.device
                or int(opened.st_ino) != self.lexical_identity.inode
            ):
                raise ConfirmatoryCheckpointContractError(f"{self.role} descriptor custody changed")

    def close(self) -> None:
        verification_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as exc:
            verification_error = exc
        finally:
            native_handle = self.native_handle
            descriptor = self.descriptor
            self.native_handle = None
            self.descriptor = None
            if native_handle is not None:
                _win32_close_handle(native_handle)
            if descriptor is not None:
                os.close(descriptor)
        if verification_error is not None:
            raise verification_error


def _named_streams(path: Path) -> tuple[str, ...]:
    if os.name != "nt":
        return ()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(_Win32FindStreamData),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Win32FindStreamData),
    ]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    data = _Win32FindStreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error in {1, 38}:
            return ()
        raise ConfirmatoryCheckpointContractError(
            f"cannot enumerate checkpoint streams: winerror={error}"
        )
    streams: list[str] = []
    try:
        while True:
            name = str(data.stream_name)
            if name != "::$DATA":
                streams.append(name)
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise ConfirmatoryCheckpointContractError(
                    f"checkpoint stream enumeration failed: winerror={error}"
                )
    finally:
        find_close(handle)
    return tuple(sorted(streams))


def _stat_identity(value: os.stat_result) -> ConfirmatoryCheckpointPhysicalIdentity:
    return ConfirmatoryCheckpointPhysicalIdentity.from_stat(value)


def _same_open_file_object(
    first: ConfirmatoryCheckpointPhysicalIdentity,
    second: ConfirmatoryCheckpointPhysicalIdentity,
) -> bool:
    """Compare handle/path observations while retaining full lexical identity.

    On Windows CPython reports a distinct ``st_ctime_ns`` for the same NTFS
    object through ``fstat`` and ``lstat``.  File-ID, size, mode, link count,
    and mtime still bind the handle to the lexical object; ctime remains part
    of the full lexical identity and is compared path-to-path.
    """

    stable = (
        first.device,
        first.inode,
        first.size_bytes,
        first.mode,
        first.link_count,
        first.modified_time_ns,
    ) == (
        second.device,
        second.inode,
        second.size_bytes,
        second.mode,
        second.link_count,
        second.modified_time_ns,
    )
    return stable and (os.name == "nt" or first.changed_time_ns == second.changed_time_ns)


def _require_plain_checkpoint_directory(path: Path) -> None:
    checkpoint_directory = Path(os.path.abspath(path))
    if checkpoint_directory.name != "checkpoints":
        raise ConfirmatoryCheckpointContractError(
            "checkpoint directory is not the canonical per-cell checkpoints directory"
        )
    ancestors = (
        checkpoint_directory.parents[2],
        checkpoint_directory.parents[1],
        checkpoint_directory.parent,
        checkpoint_directory,
    )
    for ancestor in ancestors:
        try:
            observed = ancestor.lstat()
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint directory ancestor is unavailable: {ancestor}"
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or _is_reparse(observed)
            or int(observed.st_nlink) < 1
            or _named_streams(ancestor)
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint directory ancestor is linked/reparse/streamed/invalid: {ancestor}"
            )


def _require_plain_checkpoint_output_directory(
    path: Path,
    *,
    expected_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None,
) -> ConfirmatoryCheckpointPhysicalIdentity:
    """Verify one private versioned-output directory without following links."""

    output_directory = Path(os.path.abspath(path))
    _ensure_plain_checkpoint_artifact_directory(
        output_directory.parent,
        expected_name="checkpoint_versions",
    )
    try:
        observed = output_directory.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint output directory is unavailable: {output_directory}"
        ) from exc
    identity = _stat_identity(observed)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _is_reparse(observed)
        or int(observed.st_nlink) < 1
        or _named_streams(output_directory)
        or (
            expected_identity is not None
            and (
                identity.device,
                identity.inode,
                identity.mode,
            )
            != (
                expected_identity.device,
                expected_identity.inode,
                expected_identity.mode,
            )
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint output directory is replaced/reparse/streamed/invalid: {output_directory}"
        )
    return identity


def _ensure_plain_checkpoint_artifact_directory(
    path: Path,
    *,
    expected_name: str,
) -> ConfirmatoryCheckpointPhysicalIdentity:
    """Create-once or verify one plain sibling directory below the cell root."""

    artifact_directory = Path(os.path.abspath(path))
    checkpoint_directory = artifact_directory.parent / "checkpoints"
    _require_plain_checkpoint_directory(checkpoint_directory)
    if artifact_directory.name != expected_name:
        raise ConfirmatoryCheckpointContractError(
            "checkpoint artifact directory has a noncanonical role name"
        )
    try:
        artifact_directory.mkdir()
    except FileExistsError:
        pass
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint artifact directory cannot be established: {artifact_directory}"
        ) from exc
    try:
        observed = artifact_directory.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint artifact directory is unavailable: {artifact_directory}"
        ) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _is_reparse(observed)
        or int(observed.st_nlink) < 1
        or _named_streams(artifact_directory)
    ):
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint artifact directory is linked/reparse/streamed/invalid: "
            f"{artifact_directory}"
        )
    return _stat_identity(observed)


def _hash_private_checkpoint(path: Path, *, role: str) -> tuple[str, int]:
    with _hold_private_checkpoint_snapshot(path, role=role) as snapshot:
        return snapshot.sha256, snapshot.size_bytes


@dataclass(frozen=True, slots=True)
class _PrivateCheckpointSnapshot:
    payload: bytes
    sha256: str
    size_bytes: int
    identity: ConfirmatoryCheckpointPhysicalIdentity

    @property
    def file_id_128(self) -> str:
        return self.identity.file_id_128


def _open_existing_checkpoint_descriptor(
    path: Path,
    *,
    write_attributes: bool,
) -> tuple[int, _NativeCheckpointIdentity | None]:
    if os.name == "nt":
        return _win32_open_existing_checkpoint_descriptor(
            path,
            write_attributes=write_attributes,
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path, flags), None


def _create_checkpoint_descriptor(
    path: Path,
) -> tuple[int, _NativeCheckpointIdentity | None]:
    if os.name == "nt":
        return _win32_create_checkpoint_descriptor(path)
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_RDWR
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path, flags, 0o600), None


def _create_owner_lock_descriptor(
    path: Path,
) -> tuple[int, _NativeCheckpointIdentity | None]:
    if os.name == "nt":
        return _win32_create_owner_lock_descriptor(path)
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_RDWR
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path, flags, 0o600), None


def _native_identity_from_descriptor(
    descriptor: int,
) -> _NativeCheckpointIdentity | None:
    if os.name != "nt":
        return None
    return _win32_native_identity_from_descriptor(descriptor)


def _set_posix_descriptor_mode(descriptor: int, mode: int) -> None:
    fchmod = cast(Callable[[int, int], None], vars(os)["fchmod"])
    fchmod(descriptor, mode)


def _native_identity_from_path(path: Path) -> _NativeCheckpointIdentity | None:
    if os.name != "nt":
        return None
    return _win32_native_identity_from_path(path)


def _native_identity_from_live_writer_path(
    path: Path,
) -> _NativeCheckpointIdentity | None:
    if os.name != "nt":
        return None
    return _win32_native_identity_from_live_writer_path(path)


def _native_identity_from_owner_lock_path(
    path: Path,
) -> _NativeCheckpointIdentity | None:
    if os.name != "nt":
        return None
    return _win32_native_identity_from_owner_lock_path(path)


def _native_identity_is_plain_file(
    value: _NativeCheckpointIdentity | None,
) -> bool:
    return value is None or not value.file_attributes & (
        _WIN_FILE_ATTRIBUTE_REPARSE_POINT | _WIN_FILE_ATTRIBUTE_DIRECTORY
    )


def _same_native_file(
    first: _NativeCheckpointIdentity | None,
    second: _NativeCheckpointIdentity | None,
    *,
    allow_read_only_transition: bool = False,
) -> bool:
    if first is None or second is None:
        return first is second
    if (
        first.volume_serial_number != second.volume_serial_number
        or first.file_id_128 != second.file_id_128
        or first.reparse_tag != second.reparse_tag
    ):
        return False
    if not allow_read_only_transition:
        return True
    return bool(second.file_attributes & _WIN_FILE_ATTRIBUTE_READONLY)


def _same_file_except_read_only_transition(
    before: ConfirmatoryCheckpointPhysicalIdentity,
    after: ConfirmatoryCheckpointPhysicalIdentity,
) -> bool:
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return (
        before.device,
        before.inode,
        before.size_bytes,
        before.link_count,
        before.modified_time_ns,
        before.mode & ~write_bits,
    ) == (
        after.device,
        after.inode,
        after.size_bytes,
        after.link_count,
        after.modified_time_ns,
        after.mode & ~write_bits,
    ) and after.mode & write_bits == 0


@contextmanager
def _hold_private_checkpoint_descriptor(
    path: Path,
    *,
    role: str,
    write_attributes: bool = False,
    allow_read_only_transition: bool = False,
) -> Iterator[tuple[int, _PrivateCheckpointSnapshot]]:
    """Hold one no-follow descriptor through consumer adoption.

    The yielded bytes, digest, size, and full lexical identity all come from
    one descriptor-bound read.  The descriptor remains open until the
    consumer leaves the context.  On Windows it is a native ``CreateFileW``
    handle opened with ``FILE_FLAG_OPEN_REPARSE_POINT`` and share READ only,
    so WRITE and DELETE are denied while the consumer adopts the evidence.
    Both native volume/FILE_ID_128 and lexical identity are checked before
    and after adoption.  A same-byte pathname replacement is therefore fatal.
    """

    try:
        initial = path.lstat()
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(f"{role} is unavailable: {path}") from exc
    if (
        not stat.S_ISREG(initial.st_mode)
        or _is_reparse(initial)
        or int(initial.st_nlink) != 1
        or _named_streams(path)
    ):
        raise ConfirmatoryCheckpointContractError(
            f"{role} is not one private regular stream-free file: {path}"
        )
    descriptor: int | None = None
    try:
        descriptor, native_initial = _open_existing_checkpoint_descriptor(
            path,
            write_attributes=write_attributes,
        )
        native_path_reader = (
            _native_identity_from_live_writer_path
            if write_attributes
            else _native_identity_from_path
        )
        opened = os.fstat(descriptor)
        native_opened = _native_identity_from_descriptor(descriptor)
        native_path = native_path_reader(path)
        if (
            not _same_open_file_object(
                _stat_identity(opened),
                _stat_identity(initial),
            )
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or not _same_native_file(native_opened, native_initial)
            or not _same_native_file(native_path, native_initial)
            or not _native_identity_is_plain_file(native_initial)
        ):
            raise ConfirmatoryCheckpointContractError(f"{role} changed while opening")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        opened_after = os.fstat(descriptor)
        lexical_after = path.lstat()
        native_after = _native_identity_from_descriptor(descriptor)
        native_path_after = native_path_reader(path)
        identity = _stat_identity(lexical_after)
        if (
            _stat_identity(opened_after) != _stat_identity(opened)
            or identity != _stat_identity(initial)
            or not _same_open_file_object(
                _stat_identity(opened_after),
                identity,
            )
            or _is_reparse(lexical_after)
            or _named_streams(path)
            or size != int(opened_after.st_size)
            or not _same_native_file(native_after, native_initial)
            or not _same_native_file(native_path_after, native_initial)
            or not _native_identity_is_plain_file(native_after)
        ):
            raise ConfirmatoryCheckpointContractError(f"{role} changed while reading")
        snapshot = _PrivateCheckpointSnapshot(
            payload=b"".join(chunks),
            sha256=digest.hexdigest(),
            size_bytes=size,
            identity=identity,
        )
        try:
            yield descriptor, snapshot
        finally:
            opened_final = os.fstat(descriptor)
            lexical_final = path.lstat()
            opened_final_identity = _stat_identity(opened_final)
            lexical_final_identity = _stat_identity(lexical_final)
            native_final = _native_identity_from_descriptor(descriptor)
            native_path_final = native_path_reader(path)
            stable_transition = (
                _same_file_except_read_only_transition(
                    _stat_identity(opened_after),
                    opened_final_identity,
                )
                and _same_file_except_read_only_transition(
                    identity,
                    lexical_final_identity,
                )
                if allow_read_only_transition
                else (
                    opened_final_identity == _stat_identity(opened_after)
                    and lexical_final_identity == identity
                )
            )
            if (
                not stable_transition
                or not _same_open_file_object(
                    opened_final_identity,
                    lexical_final_identity,
                )
                or _is_reparse(lexical_final)
                or _named_streams(path)
                or not _same_native_file(
                    native_initial,
                    native_final,
                    allow_read_only_transition=allow_read_only_transition,
                )
                or not _same_native_file(
                    native_initial,
                    native_path_final,
                    allow_read_only_transition=allow_read_only_transition,
                )
                or not _native_identity_is_plain_file(native_final)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} changed before consumer adoption completed"
                )
    except OSError as exc:
        raise ConfirmatoryCheckpointContractError(f"{role} is unreadable: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _hold_private_checkpoint_snapshot(
    path: Path,
    *,
    role: str,
) -> Iterator[_PrivateCheckpointSnapshot]:
    with _hold_private_checkpoint_descriptor(path, role=role) as (_, snapshot):
        yield snapshot


def _read_private_checkpoint_bytes(
    path: Path,
    *,
    role: str,
) -> _PrivateCheckpointSnapshot:
    """Read and verify one private file for immediate, non-retained use."""

    with _hold_private_checkpoint_snapshot(path, role=role) as snapshot:
        return snapshot


def _validate_confirmatory_checkpoint_bytes(
    payload_bytes: bytes,
    *,
    expected_configuration: Mapping[str, Any],
    expected_model_metadata: Mapping[str, Any],
    expected_data_and_split_sha256: Mapping[str, str],
) -> None:
    """Validate one held checkpoint snapshot without reopening its pathname."""

    try:
        payload = torch.load(
            io.BytesIO(payload_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("confirmatory checkpoint is not a safe Torch payload") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("confirmatory checkpoint payload must be a mapping")
    required_fields = {
        "schema_version",
        "model_kind",
        "execution_mode",
        "study_outcome_eligible",
        "configuration",
        "configuration_sha256",
        "resume_contract_sha256",
        "data_and_split_sha256",
        "model_metadata",
        "class_order",
        "completed_epochs",
        "network_state_dict",
        "optimiser_state_dict",
        "scaler_state_dict",
        "history",
        "effective_batch_size",
        "rng_state",
        "early_stopping_state",
        "telemetry",
    }
    if set(payload) != required_fields:
        raise ValueError("confirmatory checkpoint has an invalid exact schema")
    configuration = dict(expected_configuration)
    stored_configuration = payload.get("configuration")
    if (
        payload.get("schema_version") != 1
        or payload.get("model_kind") != "confirmatory_resnet18_five_class"
        or payload.get("execution_mode") != "real_study_cuda"
        or payload.get("study_outcome_eligible") is not True
        or stored_configuration != configuration
        or payload.get("configuration_sha256") != _json_sha256(configuration)
    ):
        raise ValueError("confirmatory checkpoint mode/configuration is invalid")
    resume_configuration = dict(configuration)
    resume_configuration.pop("epochs", None)
    if payload.get("resume_contract_sha256") != _json_sha256(resume_configuration):
        raise ValueError("confirmatory checkpoint resume contract is invalid")
    hashes = payload.get("data_and_split_sha256")
    expected_hashes = dict(expected_data_and_split_sha256)
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != _DATA_AND_SPLIT_KEYS
        or dict(hashes) != expected_hashes
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in hashes.values()
        )
    ):
        raise ValueError("confirmatory checkpoint data/split fingerprints are invalid")
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, Mapping) or dict(metadata) != dict(expected_model_metadata):
        raise ValueError("confirmatory checkpoint model metadata differs from fold evidence")
    class_order = payload.get("class_order")
    if (
        not isinstance(class_order, torch.Tensor)
        or class_order.dtype != torch.int64
        or class_order.shape != (len(CLASS_ORDER),)
        or not np.array_equal(
            class_order.cpu().numpy(),
            np.asarray(CLASS_ORDER, dtype=np.int64),
        )
    ):
        raise ValueError("confirmatory checkpoint class order is invalid")
    input_channels = int(expected_model_metadata["input_channels"])
    _validate_checkpoint_state_dict(
        payload.get("network_state_dict"),
        input_channels=input_channels,
        role="network_state_dict",
    )
    early = payload.get("early_stopping_state")
    if not isinstance(early, Mapping) or set(early) != {
        "best_epoch",
        "best_validation_loss",
        "epochs_without_improvement",
        "stopped_early",
        "best_network_state_dict",
    }:
        raise ValueError("confirmatory checkpoint early-stopping state is invalid")
    _validate_checkpoint_state_dict(
        early.get("best_network_state_dict"),
        input_channels=input_channels,
        role="best_network_state_dict",
    )
    completed_epochs = payload.get("completed_epochs")
    effective_batch_size = payload.get("effective_batch_size")
    if (
        type(completed_epochs) is not int
        or not 1 <= completed_epochs <= int(configuration["epochs"])
        or type(effective_batch_size) is not int
        or not int(configuration["minimum_batch_size"])
        <= effective_batch_size
        <= int(configuration["batch_size"])
    ):
        raise ValueError("confirmatory checkpoint training state is invalid")
    telemetry = payload.get("telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or type(telemetry.get("successful_optimiser_steps")) is not int
        or type(telemetry.get("skipped_optimiser_steps")) is not int
    ):
        raise ValueError("confirmatory checkpoint optimiser-step telemetry is invalid")
    _validate_adamw_state(
        payload.get("optimiser_state_dict"),
        input_channels=input_channels,
        configuration=configuration,
        successful_optimiser_steps=int(telemetry["successful_optimiser_steps"]),
        skipped_optimiser_steps=int(telemetry["skipped_optimiser_steps"]),
    )
    _validate_grad_scaler_state(payload.get("scaler_state_dict"))
    _validate_rng_state(payload.get("rng_state"))
    _validate_history_and_telemetry(
        payload.get("history"),
        early,
        telemetry,
        completed_epochs=completed_epochs,
        effective_batch_size=effective_batch_size,
        configuration=configuration,
    )


class _OwnedCheckpointIO:
    """One fail-closed checkpoint ownership session for a single fold.

    The pinned scientific model retains its frozen ``checkpoint_path``/``resume``
    contract and overwrites that private canonical working file after each
    completed epoch.  At a successful fold boundary this operational owner holds
    and validates the exact working bytes, publishes one no-overwrite immutable
    version plus its commit record, and makes the canonical working checkpoint
    itself read-only.  The predecessor source is never modified or hardlinked.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        directive: ConfirmatoryCheckpointDirective,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._run_directory = checkpoint_path.parents[3]
        output_relative = directive.versioned_checkpoint_output_directory_relative_path
        self._output_directory = (
            self._run_directory / output_relative if output_relative is not None else None
        )
        self._execution_manifest_path = (
            self._run_directory / directive.checkpoint_execution_manifest_relative_path
        )
        self._current_checkpoint_path: Path | None = None
        self._current_manifest_path: Path | None = None
        self._current_manifest_snapshot: _PrivateCheckpointSnapshot | None = None
        self._directive = directive
        self._lock_path = checkpoint_path.with_name(f".{checkpoint_path.name}.aanca-owner.lock")
        self._lock_descriptor: int | None = None
        self._lock_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None
        self._lock_native_identity: _NativeCheckpointIdentity | None = None
        self._lock_payload = (
            _canonical_json(
                {
                    "schema_version": 1,
                    "policy": "aanca_checkpoint_single_owner_v1",
                    "pid": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "token": secrets.token_hex(32),
                }
            ).encode("ascii")
            + b"\n"
        )
        self._resume_bytes: bytes | None = None
        self._expected_sha256: str | None = None
        self._expected_size_bytes: int | None = None
        self._expected_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None
        self._import_snapshot: _PrivateCheckpointSnapshot | None = None
        self._source_snapshot: _PrivateCheckpointSnapshot | None = None
        self._canonical_final_snapshot: _PrivateCheckpointSnapshot | None = None
        self._output_directory_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None
        self._output_parent_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None
        self._manifest_parent_identity: ConfirmatoryCheckpointPhysicalIdentity | None = None
        self._publication_records: list[dict[str, Any]] = []
        self._publication_count = 0
        self._published = False
        self._directory_custodies: list[_DirectoryCustody] = []
        self._retained_source_stack = ExitStack()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def published_checkpoint_path(self) -> Path:
        if self._current_checkpoint_path is None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner has no current immutable checkpoint"
            )
        return self._current_checkpoint_path

    @property
    def current_manifest_path(self) -> Path | None:
        return self._current_manifest_path

    @property
    def resume_bytes(self) -> bytes | None:
        return self._resume_bytes

    @property
    def should_resume(self) -> bool:
        return self._resume_bytes is not None

    @property
    def sha256_before(self) -> str | None:
        return self._expected_sha256 if self._resume_bytes is not None else None

    @property
    def size_before(self) -> int | None:
        return self._expected_size_bytes if self._resume_bytes is not None else None

    @classmethod
    def acquire(
        cls,
        checkpoint_path: Path,
        directive: ConfirmatoryCheckpointDirective,
    ) -> _OwnedCheckpointIO:
        session = cls(checkpoint_path, directive)
        try:
            session._acquire()
        except BaseException:
            session._close_after_failed_acquire()
            raise
        return session

    def _acquire(self) -> None:
        self._directive.validate()
        for path, role in (
            (self._run_directory, "checkpoint run directory"),
            (self._run_directory / "cells", "checkpoint cells directory"),
            (self._checkpoint_path.parents[1], "checkpoint cell directory"),
            (self._checkpoint_path.parent, "canonical checkpoints directory"),
        ):
            self._hold_directory_custody(path, role=role)
        _require_plain_checkpoint_directory(self._checkpoint_path.parent)
        expected_import_path = (
            self._run_directory / f"cells/{self._directive.cell_id}/checkpoints/"
            f"fold_{self._directive.fold_id:02d}.pt"
        )
        if os.path.normcase(str(expected_import_path)) != os.path.normcase(
            str(self._checkpoint_path)
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner was given a noncanonical working slot"
            )
        try:
            descriptor, native_identity = _create_owner_lock_descriptor(self._lock_path)
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint already has an owner or unsafe lock: {self._checkpoint_path}"
            ) from exc
        self._lock_descriptor = descriptor
        self._lock_native_identity = native_identity
        try:
            written = 0
            while written < len(self._lock_payload):
                count = os.write(descriptor, self._lock_payload[written:])
                if count <= 0:
                    raise OSError("checkpoint owner lock write made no progress")
                written += count
            os.fsync(descriptor)
            observed = os.fstat(descriptor)
            lexical = self._lock_path.lstat()
            native_path_identity = _native_identity_from_owner_lock_path(self._lock_path)
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner lock could not be durably established"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or _is_reparse(lexical)
            or not _same_open_file_object(
                _stat_identity(observed),
                _stat_identity(lexical),
            )
            or int(observed.st_size) != len(self._lock_payload)
            or _named_streams(self._lock_path)
            or not _same_native_file(
                _native_identity_from_descriptor(descriptor),
                native_identity,
            )
            or not _same_native_file(native_path_identity, native_identity)
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner lock is linked, replaced, or incomplete"
            )
        self._lock_identity = _stat_identity(lexical)

        if os.path.lexists(self._execution_manifest_path):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution manifest already exists; overwrite is forbidden"
            )
        self._manifest_parent_identity = _ensure_plain_checkpoint_artifact_directory(
            self._execution_manifest_path.parent,
            expected_name="checkpoint_execution",
        )
        self._hold_directory_custody(
            self._execution_manifest_path.parent,
            role="checkpoint execution-manifest directory",
        )
        if self._directive.action in {"fresh_fit", "resume_incomplete_fit"}:
            if self._output_directory is None or os.path.lexists(self._output_directory):
                raise ConfirmatoryCheckpointContractError(
                    "versioned checkpoint output directory must be a new absent path"
                )
            self._output_parent_identity = _ensure_plain_checkpoint_artifact_directory(
                self._output_directory.parent,
                expected_name="checkpoint_versions",
            )
            self._hold_directory_custody(
                self._output_directory.parent,
                role="checkpoint versions directory",
            )
            try:
                self._output_directory.mkdir()
            except OSError as exc:
                raise ConfirmatoryCheckpointContractError(
                    "versioned checkpoint output directory could not be created exactly once"
                ) from exc
            self._output_directory_identity = _require_plain_checkpoint_output_directory(
                self._output_directory
            )
            self._hold_directory_custody(
                self._output_directory,
                role="checkpoint fold-version directory",
            )
        elif self._output_directory is not None:
            raise ConfirmatoryCheckpointContractError(
                "terminal restoration forbids a versioned checkpoint output directory"
            )

        if self._directive.action == "fresh_fit":
            if os.path.lexists(self._checkpoint_path):
                raise ConfirmatoryCheckpointContractError(
                    f"fresh fit canonical working checkpoint is not absent: {self._checkpoint_path}"
                )
            return
        source = self._directive.source_predecessor_checkpoint
        if source is None:
            raise ConfirmatoryCheckpointContractError(
                "successor checkpoint lacks its predecessor source identity"
            )
        expected_source_suffix = Path(
            f"cells/{self._directive.cell_id}/checkpoints/fold_{self._directive.fold_id:02d}.pt"
        )
        if tuple(source.path.parts[-4:]) != tuple(expected_source_suffix.parts):
            raise ConfirmatoryCheckpointContractError(
                "predecessor source checkpoint has a noncanonical cell/fold path"
            )
        source_run_directory = source.path.parents[3]
        for path, role in (
            (source_run_directory, "predecessor checkpoint run directory"),
            (source_run_directory / "cells", "predecessor checkpoint cells directory"),
            (source.path.parents[1], "predecessor checkpoint cell directory"),
            (source.path.parent, "predecessor checkpoints directory"),
        ):
            self._hold_directory_custody(path, role=role)
        source_snapshot = self._retained_source_stack.enter_context(
            _hold_private_checkpoint_snapshot(
                source.path,
                role="retained immutable predecessor source checkpoint",
            )
        )
        if (
            source_snapshot.sha256 != source.sha256
            or source_snapshot.size_bytes != source.size_bytes
            or source_snapshot.identity != source.physical_identity
            or stat.S_IMODE(source_snapshot.identity.mode)
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ConfirmatoryCheckpointContractError(
                "predecessor source checkpoint changed or remains writable before successor fitting"
            )
        snapshot = _read_private_checkpoint_bytes(
            self._checkpoint_path,
            role="authorized successor working checkpoint",
        )
        if (
            snapshot.sha256 != self._directive.checkpoint_sha256
            or snapshot.size_bytes != self._directive.checkpoint_size_bytes
            or self._directive.destination_imported_checkpoint is None
            or os.path.normcase(str(self._directive.destination_imported_checkpoint.path))
            != os.path.normcase(str(self._checkpoint_path))
            or snapshot.identity
            != self._directive.destination_imported_checkpoint.physical_identity
        ):
            raise ConfirmatoryCheckpointContractError(
                "authorized checkpoint disappeared or changed; fresh fallback is forbidden"
            )
        self._resume_bytes = snapshot.payload
        self._expected_sha256 = snapshot.sha256
        self._expected_size_bytes = snapshot.size_bytes
        self._expected_identity = snapshot.identity
        self._import_snapshot = snapshot
        self._source_snapshot = source_snapshot
        if self._directive.action == "restore_terminal_checkpoint_without_fit":
            self._current_checkpoint_path = self._checkpoint_path

    def _hold_directory_custody(self, path: Path, *, role: str) -> None:
        absolute = Path(os.path.abspath(path))
        if any(
            os.path.normcase(str(value.path)) == os.path.normcase(str(absolute))
            for value in self._directory_custodies
        ):
            return
        self._directory_custodies.append(_DirectoryCustody.acquire(absolute, role=role))

    def _verify_directory_custodies(self) -> None:
        if not self._directory_custodies:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner lacks retained ancestor custody"
            )
        for custody in self._directory_custodies:
            custody.verify()

    def _close_directory_custodies(self) -> None:
        first_error: BaseException | None = None
        for custody in reversed(self._directory_custodies):
            try:
                custody.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._directory_custodies.clear()
        if first_error is not None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint ancestor custody could not be released exactly"
            ) from first_error

    def _close_after_failed_acquire(self) -> None:
        descriptor = self._lock_descriptor
        self._lock_descriptor = None
        if descriptor is not None:
            owned_identity = self._lock_identity
            if owned_identity is None:
                with suppress(OSError):
                    owned_identity = _stat_identity(os.fstat(descriptor))
            with suppress(OSError):
                os.close(descriptor)
            if os.name != "nt":
                with suppress(OSError):
                    if (
                        owned_identity is not None
                        and self._lock_path.is_file()
                        and (
                            _stat_identity(self._lock_path.lstat()) == owned_identity
                            or _same_open_file_object(
                                _stat_identity(self._lock_path.lstat()),
                                owned_identity,
                            )
                        )
                    ):
                        self._lock_path.unlink()
        with suppress(Exception):
            self._retained_source_stack.close()
        with suppress(Exception):
            self._close_directory_custodies()
        # The lock is preclaim-only ephemeral ownership state.  Any created
        # scientific output namespace is deliberately retained: removing it
        # would silently rearm a one-use destination after a failed acquire.

    def _require_live_owner(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is None or self._lock_identity is None:
            raise ConfirmatoryCheckpointContractError("checkpoint owner session is not live")
        self._verify_directory_custodies()
        try:
            observed = os.fstat(descriptor)
            lexical = self._lock_path.lstat()
            native_path_identity = _native_identity_from_owner_lock_path(self._lock_path)
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError("checkpoint owner lock disappeared") from exc
        if (
            not _same_open_file_object(
                _stat_identity(observed),
                self._lock_identity,
            )
            or _stat_identity(lexical) != self._lock_identity
            or _is_reparse(lexical)
            or int(lexical.st_nlink) != 1
            or not _same_native_file(
                _native_identity_from_descriptor(descriptor),
                self._lock_native_identity,
            )
            or not _same_native_file(
                native_path_identity,
                self._lock_native_identity,
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner lock was replaced or aliased"
            )
        if self._output_directory is not None:
            if self._output_directory_identity is None:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint output directory has no held creation identity"
                )
            _require_plain_checkpoint_output_directory(
                self._output_directory,
                expected_identity=self._output_directory_identity,
            )
        if self._manifest_parent_identity is None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution-manifest directory has no held identity"
            )
        observed_manifest_parent = _ensure_plain_checkpoint_artifact_directory(
            self._execution_manifest_path.parent,
            expected_name="checkpoint_execution",
        )
        if (
            observed_manifest_parent.device,
            observed_manifest_parent.inode,
            observed_manifest_parent.mode,
        ) != (
            self._manifest_parent_identity.device,
            self._manifest_parent_identity.inode,
            self._manifest_parent_identity.mode,
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution-manifest directory was replaced"
            )
        if self._output_directory is not None:
            if self._output_parent_identity is None:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint versioned-output parent has no held identity"
                )
            observed_output_parent = _ensure_plain_checkpoint_artifact_directory(
                self._output_directory.parent,
                expected_name="checkpoint_versions",
            )
            if (
                observed_output_parent.device,
                observed_output_parent.inode,
                observed_output_parent.mode,
            ) != (
                self._output_parent_identity.device,
                self._output_parent_identity.inode,
                self._output_parent_identity.mode,
            ):
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint versioned-output parent was replaced"
                )

    def _verify_source_unchanged(self, *, role: str) -> None:
        if self._source_snapshot is None:
            return
        source = self._directive.source_predecessor_checkpoint
        if source is None:
            raise ConfirmatoryCheckpointContractError(
                "predecessor source identity disappeared from the directive"
            )
        observed = _read_private_checkpoint_bytes(source.path, role=role)
        if observed != self._source_snapshot:
            raise ConfirmatoryCheckpointContractError(
                "immutable predecessor source checkpoint changed during execution"
            )

    def _verify_current(self, *, role: str) -> _PrivateCheckpointSnapshot:
        self._require_live_owner()
        if (
            self._current_checkpoint_path is None
            or self._expected_sha256 is None
            or self._expected_size_bytes is None
            or self._expected_identity is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint has not yet been published by this owner"
            )
        snapshot = _read_private_checkpoint_bytes(
            self._current_checkpoint_path,
            role=role,
        )
        if (
            snapshot.sha256 != self._expected_sha256
            or snapshot.size_bytes != self._expected_size_bytes
            or snapshot.identity != self._expected_identity
        ):
            raise ConfirmatoryCheckpointContractError(
                f"{role} differs from the exact bytes owned by this session"
            )
        return snapshot

    def _make_read_only(
        self,
        path: Path,
        *,
        expected: _PrivateCheckpointSnapshot,
        role: str,
    ) -> _PrivateCheckpointSnapshot:
        """Remove write access through one retained native/descriptor-bound handle."""

        try:
            with _hold_private_checkpoint_descriptor(
                path,
                role=f"read-only adoption {role}",
                write_attributes=True,
                allow_read_only_transition=True,
            ) as (descriptor, before):
                if before != expected:
                    raise ConfirmatoryCheckpointContractError(
                        f"{role} changed before read-only adoption"
                    )
                if os.name == "nt":
                    _win32_set_read_only_on_descriptor(descriptor)
                else:
                    read_only_mode = stat.S_IMODE(before.identity.mode) & ~(
                        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                    )
                    _set_posix_descriptor_mode(descriptor, read_only_mode)
                    os.fsync(descriptor)
                opened_after = os.fstat(descriptor)
                lexical_after = path.lstat()
                native_after = _native_identity_from_descriptor(descriptor)
                native_path_after = _native_identity_from_live_writer_path(path)
                if (
                    not _same_open_file_object(
                        _stat_identity(opened_after),
                        _stat_identity(lexical_after),
                    )
                    or native_after != native_path_after
                    or not _native_identity_is_plain_file(native_after)
                    or _named_streams(path)
                ):
                    raise ConfirmatoryCheckpointContractError(
                        f"{role} changed while becoming read-only"
                    )
                after = _PrivateCheckpointSnapshot(
                    payload=before.payload,
                    sha256=before.sha256,
                    size_bytes=before.size_bytes,
                    identity=_stat_identity(lexical_after),
                )
                if (
                    not _same_file_except_read_only_transition(
                        before.identity,
                        after.identity,
                    )
                    or after.identity.link_count != 1
                ):
                    raise ConfirmatoryCheckpointContractError(
                        f"{role} read-only adoption changed bytes or ownership"
                    )
                self._verify_directory_custodies()
                return after
        except ConfirmatoryCheckpointContractError:
            raise
        except OSError as exc:
            raise ConfirmatoryCheckpointContractError(
                f"{role} could not be made read-only"
            ) from exc

    def _commit_immutable_bytes(
        self,
        destination: Path,
        payload: bytes,
        *,
        role: str,
    ) -> _PrivateCheckpointSnapshot:
        """Write one O_EXCL destination directly; never link, replace, or clean it."""

        if not payload:
            raise ConfirmatoryCheckpointContractError(f"empty {role} is forbidden")
        self._require_live_owner()
        if os.path.lexists(destination):
            raise ConfirmatoryCheckpointContractError(
                f"{role} destination already exists; overwrite is forbidden"
            )
        descriptor: int | None = None
        committed: _PrivateCheckpointSnapshot | None = None
        try:
            descriptor, native_created = _create_checkpoint_descriptor(destination)
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError(f"{role} write made no progress")
                written += count
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            readback = bytearray()
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                readback.extend(chunk)
            held_before = _stat_identity(os.fstat(descriptor))
            lexical_before = destination.lstat()
            lexical_before_identity = _stat_identity(lexical_before)
            native_before = _native_identity_from_descriptor(descriptor)
            native_path_before = _native_identity_from_live_writer_path(destination)
            if (
                bytes(readback) != payload
                or not _same_open_file_object(held_before, lexical_before_identity)
                or held_before.link_count != 1
                or held_before.size_bytes != len(payload)
                or not stat.S_ISREG(held_before.mode)
                or _is_reparse(lexical_before)
                or _named_streams(destination)
                or not _same_native_file(native_before, native_created)
                or not _same_native_file(native_path_before, native_created)
                or not _native_identity_is_plain_file(native_before)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} O_EXCL destination is not exact and private"
                )
            if os.name == "nt":
                _win32_set_read_only_on_descriptor(descriptor)
            else:
                _set_posix_descriptor_mode(
                    descriptor,
                    stat.S_IMODE(held_before.mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
                )
                os.fsync(descriptor)
            held_after = _stat_identity(os.fstat(descriptor))
            lexical_after = destination.lstat()
            lexical_after_identity = _stat_identity(lexical_after)
            native_after = _native_identity_from_descriptor(descriptor)
            native_path_after = _native_identity_from_live_writer_path(destination)
            if (
                not _same_file_except_read_only_transition(
                    held_before,
                    held_after,
                )
                or not _same_file_except_read_only_transition(
                    lexical_before_identity,
                    lexical_after_identity,
                )
                or not _same_open_file_object(held_after, lexical_after_identity)
                or not _same_native_file(
                    native_created,
                    native_after,
                    allow_read_only_transition=True,
                )
                or not _same_native_file(
                    native_created,
                    native_path_after,
                    allow_read_only_transition=True,
                )
                or _is_reparse(lexical_after)
                or _named_streams(destination)
            ):
                raise ConfirmatoryCheckpointContractError(
                    f"{role} changed while being frozen on its creation handle"
                )
            committed = _PrivateCheckpointSnapshot(
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                identity=lexical_after_identity,
            )
        except ConfirmatoryCheckpointContractError:
            raise
        except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
            raise ConfirmatoryCheckpointContractError(
                f"{role} immutable no-overwrite publication failed"
            ) from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)

        if committed is None:
            raise ConfirmatoryCheckpointContractError(
                f"{role} immutable publication produced no adopted evidence"
            )
        readback_snapshot = _read_private_checkpoint_bytes(
            destination,
            role=f"committed {role}",
        )
        if (
            readback_snapshot != committed
            or readback_snapshot.payload != payload
            or readback_snapshot.sha256 != hashlib.sha256(payload).hexdigest()
            or readback_snapshot.size_bytes != len(payload)
        ):
            raise ConfirmatoryCheckpointContractError(
                f"committed {role} differs from its exact written bytes"
            )
        self._verify_directory_custodies()
        return readback_snapshot

    def read_working_checkpoint(self) -> _PrivateCheckpointSnapshot:
        """Hold and read the exact canonical per-epoch working checkpoint."""

        self._require_live_owner()
        self._verify_source_unchanged(role="pre-boundary immutable predecessor source checkpoint")
        return _read_private_checkpoint_bytes(
            self._checkpoint_path,
            role="post-fit canonical working checkpoint",
        )

    def _publication_paths(self, *, completed_epochs: int) -> tuple[Path, Path]:
        if (
            type(completed_epochs) is not int
            or completed_epochs <= self._directive.completed_epochs_before_fit
            or completed_epochs > self._directive.maximum_epochs
            or self._publication_count != 0
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint boundary publication epoch is invalid or duplicate"
            )
        if self._output_directory is None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint publication lacks its predeclared output directory"
            )
        basename = f"epoch_{completed_epochs:04d}"
        return (
            self._output_directory / f"{basename}.pt",
            self._output_directory / f"{basename}.commit.json",
        )

    def publish_completed_working_checkpoint(
        self,
        expected_working: _PrivateCheckpointSnapshot,
        *,
        completed_epochs: int,
        trained_epochs: int,
    ) -> None:
        """Freeze the canonical working file and publish one immutable fold boundary."""

        if (
            self._directive.action == "restore_terminal_checkpoint_without_fit"
            or type(trained_epochs) is not int
            or trained_epochs <= 0
            or completed_epochs != self._directive.completed_epochs_before_fit + trained_epochs
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint boundary publication does not describe completed training"
            )
        self._require_live_owner()
        self._verify_source_unchanged(
            role="pre-publication immutable predecessor source checkpoint"
        )
        observed_working = _read_private_checkpoint_bytes(
            self._checkpoint_path,
            role="pre-publication canonical working checkpoint",
        )
        if observed_working != expected_working:
            raise ConfirmatoryCheckpointContractError(
                "canonical working checkpoint changed before boundary publication"
            )
        canonical_frozen = self._make_read_only(
            self._checkpoint_path,
            expected=observed_working,
            role="canonical working checkpoint",
        )
        checkpoint_path, manifest_path = self._publication_paths(completed_epochs=completed_epochs)
        published = self._commit_immutable_bytes(
            checkpoint_path,
            canonical_frozen.payload,
            role="versioned checkpoint",
        )
        destination_identity = ConfirmatoryCheckpointFileIdentity(
            path=checkpoint_path.resolve(),
            physical_identity=published.identity,
            size_bytes=published.size_bytes,
            sha256=published.sha256,
        )
        canonical_identity = ConfirmatoryCheckpointFileIdentity(
            path=self._checkpoint_path.resolve(),
            physical_identity=canonical_frozen.identity,
            size_bytes=canonical_frozen.size_bytes,
            sha256=canonical_frozen.sha256,
        )
        destination_identity.validate()
        canonical_identity.validate()
        previous_identity = (
            ConfirmatoryCheckpointFileIdentity(
                path=self._checkpoint_path.resolve(),
                physical_identity=self._import_snapshot.identity,
                size_bytes=self._import_snapshot.size_bytes,
                sha256=self._import_snapshot.sha256,
            )
            if self._import_snapshot is not None
            else None
        )
        manifest_payload = (
            _canonical_json(
                {
                    "schema_version": 2,
                    "policy": "aanca_fold_boundary_checkpoint_commit_v2",
                    "fit_id": (f"{self._directive.cell_id}::fold_{self._directive.fold_id:02d}"),
                    "fit_attempt": 1,
                    "publication_index": 1,
                    "publication_boundary": "successful_fold_completion",
                    "completed_epochs": completed_epochs,
                    "trained_epochs": trained_epochs,
                    "versioned_checkpoint_output_directory_relative_path": (
                        self._directive.versioned_checkpoint_output_directory_relative_path
                    ),
                    "checkpoint_execution_manifest_relative_path": (
                        self._directive.checkpoint_execution_manifest_relative_path
                    ),
                    "directive_sha256": self._directive.directive_sha256,
                    "source_predecessor_checkpoint": (
                        self._directive.source_predecessor_checkpoint.as_dict()
                        if self._directive.source_predecessor_checkpoint is not None
                        else None
                    ),
                    "destination_imported_checkpoint": (
                        self._directive.destination_imported_checkpoint.as_dict()
                        if self._directive.destination_imported_checkpoint is not None
                        else None
                    ),
                    "previous_checkpoint": (
                        previous_identity.as_dict() if previous_identity is not None else None
                    ),
                    "canonical_working_checkpoint": canonical_identity.as_dict(),
                    "versioned_checkpoint": destination_identity.as_dict(),
                    "automatic_retry_allowed": False,
                    "hardlink_or_replace_used": False,
                    "canonical_working_checkpoint_read_only": True,
                    "mutable_latest_path_created": False,
                }
            ).encode("ascii")
            + b"\n"
        )
        manifest = self._commit_immutable_bytes(
            manifest_path,
            manifest_payload,
            role="checkpoint commit manifest",
        )

        self._verify_source_unchanged(
            role="post-publication immutable predecessor source checkpoint"
        )
        canonical_after = _read_private_checkpoint_bytes(
            self._checkpoint_path,
            role="post-publication canonical working checkpoint",
        )
        published_after = _read_private_checkpoint_bytes(
            checkpoint_path,
            role="post-publication immutable checkpoint",
        )
        manifest_after = _read_private_checkpoint_bytes(
            manifest_path,
            role="post-publication checkpoint commit manifest",
        )
        if (
            canonical_after != canonical_frozen
            or published_after != published
            or manifest_after != manifest
            or canonical_after.payload != published_after.payload
            or canonical_after.sha256 != published_after.sha256
            or canonical_after.identity.file_id_128 == published_after.identity.file_id_128
        ):
            raise ConfirmatoryCheckpointContractError(
                "fold-boundary checkpoint copies changed, differ, or alias"
            )

        self._publication_records.append(
            {
                "publication_index": 1,
                "completed_epochs": completed_epochs,
                "checkpoint_relative_path": checkpoint_path.relative_to(
                    self._run_directory
                ).as_posix(),
                "checkpoint": destination_identity.as_dict(),
                "commit_manifest_relative_path": manifest_path.relative_to(
                    self._run_directory
                ).as_posix(),
                "commit_manifest_sha256": manifest.sha256,
                "commit_manifest_size_bytes": manifest.size_bytes,
                "commit_manifest_physical_identity": manifest.identity.as_dict(),
            }
        )
        self._canonical_final_snapshot = canonical_frozen
        self._current_checkpoint_path = checkpoint_path
        self._expected_sha256 = published.sha256
        self._expected_size_bytes = published.size_bytes
        self._expected_identity = published.identity
        self._publication_count = 1
        self._published = True

    def adopt_terminal_working_checkpoint(
        self,
        expected_working: _PrivateCheckpointSnapshot,
    ) -> None:
        """Freeze an unchanged terminal successor copy without publishing a new version."""

        self._require_live_owner()
        if (
            self._directive.action != "restore_terminal_checkpoint_without_fit"
            or self._import_snapshot is None
            or expected_working != self._import_snapshot
            or self._published
        ):
            raise ConfirmatoryCheckpointContractError(
                "terminal working checkpoint differs from its exact imported copy"
            )
        self._verify_source_unchanged(role="terminal immutable predecessor source checkpoint")
        canonical_frozen = self._make_read_only(
            self._checkpoint_path,
            expected=expected_working,
            role="terminal canonical working checkpoint",
        )
        self._canonical_final_snapshot = canonical_frozen
        self._current_checkpoint_path = self._checkpoint_path
        self._expected_sha256 = canonical_frozen.sha256
        self._expected_size_bytes = canonical_frozen.size_bytes
        self._expected_identity = canonical_frozen.identity
        self._verify_directory_custodies()

    def record_execution_manifest(
        self,
        *,
        completed_epochs_after: int,
        trained_epochs: int,
    ) -> None:
        """Commit one exact schema-v3 record for the successful fold boundary."""

        if (
            self._current_checkpoint_path is None
            or self._canonical_final_snapshot is None
            or self._current_manifest_path is not None
            or os.path.lexists(self._execution_manifest_path)
            or type(completed_epochs_after) is not int
            or type(trained_epochs) is not int
            or trained_epochs < 0
        ):
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution manifest request is duplicate or incomplete"
            )
        terminal_restore = self._directive.action == "restore_terminal_checkpoint_without_fit"
        if terminal_restore:
            if (
                trained_epochs != 0
                or completed_epochs_after != self._directive.completed_epochs_before_fit
                or self._published
                or self._publication_records
                or self._output_directory is not None
                or self._import_snapshot is None
            ):
                raise ConfirmatoryCheckpointContractError(
                    "terminal restoration execution manifest would describe training"
                )
        elif (
            trained_epochs <= 0
            or completed_epochs_after
            != self._directive.completed_epochs_before_fit + trained_epochs
            or not self._published
            or self._publication_count != 1
            or len(self._publication_records) != 1
            or self._publication_records[0].get("completed_epochs") != completed_epochs_after
            or self._output_directory is None
        ):
            raise ConfirmatoryCheckpointContractError(
                "fresh/resume execution manifest lacks its one fold-boundary publication"
            )

        adopted_manifest: _PrivateCheckpointSnapshot | None = None
        with ExitStack() as held_sources:
            if self._source_snapshot is not None:
                source_identity = self._directive.source_predecessor_checkpoint
                if source_identity is None:
                    raise ConfirmatoryCheckpointContractError(
                        "execution manifest lost predecessor source identity"
                    )
                source_snapshot = held_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        source_identity.path,
                        role="execution-manifest predecessor source checkpoint",
                    )
                )
                if source_snapshot != self._source_snapshot:
                    raise ConfirmatoryCheckpointContractError(
                        "predecessor source changed before manifest adoption"
                    )

            canonical_snapshot = held_sources.enter_context(
                _hold_private_checkpoint_snapshot(
                    self._checkpoint_path,
                    role="execution-manifest canonical working checkpoint",
                )
            )
            if canonical_snapshot != self._canonical_final_snapshot or stat.S_IMODE(
                canonical_snapshot.identity.mode
            ) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise ConfirmatoryCheckpointContractError(
                    "canonical working checkpoint is changed or writable"
                )
            canonical_identity = ConfirmatoryCheckpointFileIdentity(
                path=self._checkpoint_path.resolve(),
                physical_identity=canonical_snapshot.identity,
                size_bytes=canonical_snapshot.size_bytes,
                sha256=canonical_snapshot.sha256,
            )
            imported_identity = (
                ConfirmatoryCheckpointFileIdentity(
                    path=self._checkpoint_path.resolve(),
                    physical_identity=self._import_snapshot.identity,
                    size_bytes=self._import_snapshot.size_bytes,
                    sha256=self._import_snapshot.sha256,
                )
                if self._import_snapshot is not None
                else None
            )

            final_snapshot = held_sources.enter_context(
                _hold_private_checkpoint_snapshot(
                    self._current_checkpoint_path,
                    role="execution-manifest final checkpoint",
                )
            )
            if (
                final_snapshot.sha256 != self._expected_sha256
                or final_snapshot.size_bytes != self._expected_size_bytes
                or final_snapshot.identity != self._expected_identity
                or stat.S_IMODE(final_snapshot.identity.mode)
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise ConfirmatoryCheckpointContractError(
                    "execution-manifest final checkpoint differs from this owner"
                )
            final_identity = ConfirmatoryCheckpointFileIdentity(
                path=self._current_checkpoint_path.resolve(),
                physical_identity=final_snapshot.identity,
                size_bytes=final_snapshot.size_bytes,
                sha256=final_snapshot.sha256,
            )
            if terminal_restore:
                if final_snapshot != canonical_snapshot:
                    raise ConfirmatoryCheckpointContractError(
                        "terminal final checkpoint differs from its canonical copy"
                    )
            elif (
                final_snapshot.payload != canonical_snapshot.payload
                or final_snapshot.sha256 != canonical_snapshot.sha256
                or final_snapshot.identity.file_id_128 == canonical_snapshot.identity.file_id_128
            ):
                raise ConfirmatoryCheckpointContractError(
                    "versioned final and canonical working checkpoints differ or alias"
                )

            for publication in self._publication_records:
                checkpoint_relative = publication.get("checkpoint_relative_path")
                commit_relative = publication.get("commit_manifest_relative_path")
                if not isinstance(checkpoint_relative, str) or not isinstance(
                    commit_relative,
                    str,
                ):
                    raise ConfirmatoryCheckpointContractError(
                        "execution-manifest publication path is not exact"
                    )
                checkpoint_path = (self._run_directory / checkpoint_relative).resolve()
                commit_path = (self._run_directory / commit_relative).resolve()
                checkpoint = held_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        checkpoint_path,
                        role="execution-manifest versioned checkpoint",
                    )
                )
                commit = held_sources.enter_context(
                    _hold_private_checkpoint_snapshot(
                        commit_path,
                        role="execution-manifest versioned commit sidecar",
                    )
                )
                expected_checkpoint = _checkpoint_file_identity_from_exact_manifest(
                    publication.get("checkpoint"),
                    role="execution-manifest versioned checkpoint record",
                )
                expected_commit = _checkpoint_physical_identity_from_exact_manifest(
                    publication.get("commit_manifest_physical_identity"),
                    role="execution-manifest commit-sidecar record",
                )
                if (
                    os.path.normcase(str(expected_checkpoint.path))
                    != os.path.normcase(str(checkpoint_path))
                    or checkpoint.sha256 != expected_checkpoint.sha256
                    or checkpoint.size_bytes != expected_checkpoint.size_bytes
                    or checkpoint.identity != expected_checkpoint.physical_identity
                    or commit.sha256 != publication.get("commit_manifest_sha256")
                    or commit.size_bytes != publication.get("commit_manifest_size_bytes")
                    or commit.identity != expected_commit
                    or stat.S_IMODE(checkpoint.identity.mode)
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                    or stat.S_IMODE(commit.identity.mode)
                    & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                ):
                    raise ConfirmatoryCheckpointContractError(
                        "execution manifest versioned output changed or is writable"
                    )

            manifest_payload = {
                "schema_version": 3,
                "policy": "aanca_fold_boundary_checkpoint_execution_v3",
                "fit_id": (f"{self._directive.cell_id}::fold_{self._directive.fold_id:02d}"),
                "fit_attempt": 1,
                "action": self._directive.action,
                "directive_sha256": self._directive.directive_sha256,
                "source_predecessor_checkpoint": (
                    self._directive.source_predecessor_checkpoint.as_dict()
                    if self._directive.source_predecessor_checkpoint is not None
                    else None
                ),
                "destination_imported_checkpoint": (
                    self._directive.destination_imported_checkpoint.as_dict()
                    if self._directive.destination_imported_checkpoint is not None
                    else None
                ),
                "imported_checkpoint_observed": (
                    imported_identity.as_dict() if imported_identity is not None else None
                ),
                "canonical_working_checkpoint": canonical_identity.as_dict(),
                "canonical_working_checkpoint_read_only": True,
                "versioned_checkpoint_output_directory_relative_path": (
                    self._directive.versioned_checkpoint_output_directory_relative_path
                ),
                "checkpoint_execution_manifest_relative_path": (
                    self._directive.checkpoint_execution_manifest_relative_path
                ),
                "completed_epochs_before_fit": self._directive.completed_epochs_before_fit,
                "completed_epochs_after_fit": completed_epochs_after,
                "trained_epochs": trained_epochs,
                "publication_boundary": "successful_fold_completion",
                "versioned_outputs": list(self._publication_records),
                "final_checkpoint": final_identity.as_dict(),
                "automatic_retry_allowed": False,
                "imported_checkpoint_modified": (self._directive.action == "resume_incomplete_fit"),
                "hardlink_or_replace_used_for_immutable_publication": False,
                "mutable_latest_path_created": False,
            }
            _require_exact_checkpoint_execution_manifest_payload(
                manifest_payload,
                self._directive,
                run_directory=self._run_directory,
            )
            payload = _canonical_json(manifest_payload).encode("ascii") + b"\n"
            committed_manifest = self._commit_immutable_bytes(
                self._execution_manifest_path,
                payload,
                role="checkpoint execution manifest",
            )
            adopted_manifest = held_sources.enter_context(
                _hold_private_checkpoint_snapshot(
                    self._execution_manifest_path,
                    role="post-publication checkpoint execution manifest",
                )
            )
            if adopted_manifest != committed_manifest or stat.S_IMODE(
                adopted_manifest.identity.mode
            ) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint execution manifest changed or remained writable"
                )

        if adopted_manifest is None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint execution manifest was not adopted"
            )
        self._current_manifest_snapshot = adopted_manifest
        self._current_manifest_path = self._execution_manifest_path

    def read_current_bytes(self) -> _PrivateCheckpointSnapshot:
        return self._verify_current(role="post-fit owned checkpoint")

    def read_current_manifest_bytes(self) -> _PrivateCheckpointSnapshot:
        """Read back only the exact execution manifest adopted by this owner."""

        self._require_live_owner()
        if self._current_manifest_path is None or self._current_manifest_snapshot is None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner has no adopted execution manifest"
            )
        self._verify_source_unchanged(role="post-fit immutable predecessor source checkpoint")
        canonical = _read_private_checkpoint_bytes(
            self._checkpoint_path,
            role="post-fit canonical working checkpoint",
        )
        if canonical != self._canonical_final_snapshot:
            raise ConfirmatoryCheckpointContractError(
                "post-fit canonical working checkpoint changed after adoption"
            )
        self._verify_current(role="post-fit checkpoint paired with execution manifest")
        observed = _read_private_checkpoint_bytes(
            self._current_manifest_path,
            role="post-fit owned checkpoint execution manifest",
        )
        if observed != self._current_manifest_snapshot:
            raise ConfirmatoryCheckpointContractError(
                "post-fit execution manifest differs from the exact owned publication"
            )
        return observed

    def close(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is None and not self._directory_custodies:
            return
        first_error: BaseException | None = None
        try:
            if descriptor is None:
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint owner lost its singleton lock before ancestor release"
                )
            self._require_live_owner()
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed_payload = bytearray()
            while chunk := os.read(descriptor, 4096):
                observed_payload.extend(chunk)
            if bytes(observed_payload) != self._lock_payload:
                raise ConfirmatoryCheckpointContractError("checkpoint owner lock payload changed")
        except BaseException as exc:
            first_error = exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
                self._lock_descriptor = None

        if os.name == "nt":
            if os.path.lexists(self._lock_path) and first_error is None:
                first_error = ConfirmatoryCheckpointContractError(
                    "checkpoint owner lock survived native delete-on-close"
                )
        elif descriptor is not None:
            try:
                lexical = self._lock_path.lstat()
                if _stat_identity(lexical) != self._lock_identity:
                    raise ConfirmatoryCheckpointContractError(
                        "checkpoint owner lock changed before release"
                    )
                self._lock_path.unlink()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        try:
            self._retained_source_stack.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            self._close_directory_custodies()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint owner or ancestor custody failed during release"
            ) from first_error


@dataclass(frozen=True, slots=True)
class _CheckpointState:
    completed_epochs: int
    stopped_early: bool
    successful_optimiser_steps: int
    skipped_optimiser_steps: int


def _read_checkpoint_state(
    checkpoint_bytes: bytes,
    directive: ConfirmatoryCheckpointDirective,
    *,
    cpu_test_only: bool,
) -> _CheckpointState:
    configuration = _strict_json_object(
        directive.expected_configuration_json,
        role="fit-time expected configuration",
    )
    metadata = _strict_json_object(
        directive.expected_model_metadata_json,
        role="fit-time expected model metadata",
    )
    data_and_split = _strict_json_object(
        directive.expected_data_and_split_json,
        role="fit-time expected data/split binding",
    )
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ConfirmatoryCheckpointContractError(
            "fit-time checkpoint is not a safe Torch payload"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ConfirmatoryCheckpointContractError("fit-time checkpoint payload is not a mapping")
    stored_configuration = payload.get("configuration")
    early = payload.get("early_stopping_state")
    telemetry = payload.get("telemetry")
    completed = payload.get("completed_epochs")
    history = payload.get("history")
    if (
        stored_configuration != configuration
        or payload.get("configuration_sha256") != _canonical_sha256(configuration)
        or payload.get("data_and_split_sha256") != data_and_split
        or type(completed) is not int
        or not 1 <= completed <= directive.maximum_epochs
        or not isinstance(early, Mapping)
        or type(early.get("stopped_early")) is not bool
        or not isinstance(telemetry, Mapping)
        or type(telemetry.get("successful_optimiser_steps")) is not int
        or type(telemetry.get("skipped_optimiser_steps")) is not int
        or not isinstance(history, (list, tuple))
        or len(history) != completed
        or any(
            not isinstance(row, Mapping) or row.get("epoch") != index
            for index, row in enumerate(history, start=1)
        )
    ):
        raise ConfirmatoryCheckpointContractError(
            "fit-time checkpoint configuration/data/epoch state is invalid"
        )
    if not cpu_test_only:
        try:
            _validate_confirmatory_checkpoint_bytes(
                checkpoint_bytes,
                expected_configuration=configuration,
                expected_model_metadata=metadata,
                expected_data_and_split_sha256=cast(
                    Mapping[str, str],
                    data_and_split,
                ),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ConfirmatoryCheckpointContractError(
                "fit-time checkpoint failed strict production validation"
            ) from exc
    return _CheckpointState(
        completed_epochs=completed,
        stopped_early=bool(early["stopped_early"]),
        successful_optimiser_steps=int(telemetry["successful_optimiser_steps"]),
        skipped_optimiser_steps=int(telemetry["skipped_optimiser_steps"]),
    )


def _require_fold_directives(
    directives: Sequence[ConfirmatoryCheckpointDirective],
    *,
    cell_id: str,
    n_splits: int,
    checkpoint_directory: Path,
    base_config: ConfirmatoryCNNConfig,
) -> dict[int, ConfirmatoryCheckpointDirective]:
    values = tuple(directives)
    if len(values) != n_splits:
        raise ConfirmatoryCheckpointContractError(
            f"{cell_id} requires exactly {n_splits} checkpoint directives"
        )
    output: dict[int, ConfirmatoryCheckpointDirective] = {}
    run_directory = checkpoint_directory.parents[2]
    for directive in values:
        directive.validate()
        if (
            directive.cell_id != cell_id
            or directive.fold_id in output
            or directive.maximum_epochs != base_config.epochs
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint directives are duplicated or misbound for {cell_id}"
            )
        actual_path = checkpoint_directory / f"fold_{directive.fold_id:02d}.pt"
        expected_manifest_path = (
            run_directory / directive.checkpoint_execution_manifest_relative_path
        )
        canonical_manifest_path = (
            checkpoint_directory.parent
            / "checkpoint_execution"
            / (f"fold_{directive.fold_id:02d}.json")
        )
        expected_output_path = (
            run_directory / directive.versioned_checkpoint_output_directory_relative_path
            if directive.versioned_checkpoint_output_directory_relative_path is not None
            else None
        )
        canonical_output_path = (
            checkpoint_directory.parent / "checkpoint_versions" / (f"fold_{directive.fold_id:02d}")
        )
        if (
            os.path.normcase(str(expected_manifest_path))
            != os.path.normcase(str(canonical_manifest_path))
            or (
                expected_output_path is not None
                and os.path.normcase(str(expected_output_path))
                != os.path.normcase(str(canonical_output_path))
            )
            or directive.expected_configuration_sha256
            != _configuration_sha256(
                replace(base_config, seed=base_config.seed + directive.fold_id)
            )
        ):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint directive path/config differs for {cell_id}/fold {directive.fold_id}"
            )
        if directive.destination_imported_checkpoint is not None and os.path.normcase(
            str(directive.destination_imported_checkpoint.path)
        ) != os.path.normcase(str(actual_path)):
            raise ConfirmatoryCheckpointContractError(
                f"checkpoint destination identity differs for {cell_id}/fold {directive.fold_id}"
            )
        output[directive.fold_id] = directive
    if set(output) != set(range(n_splits)):
        raise ConfirmatoryCheckpointContractError(
            f"checkpoint directives omit a fold for {cell_id}"
        )
    return output


@dataclass(frozen=True, slots=True)
class _CheckpointBoundFitResult:
    classifier: Any
    checkpoint_path_after: Path
    checkpoint_execution_manifest_path: Path
    checkpoint_execution_manifest_sha256: str
    checkpoint_sha256_before: str | None
    checkpoint_size_before: int | None
    checkpoint_sha256_after: str
    checkpoint_size_after: int
    checkpoint_physical_identity_after: ConfirmatoryCheckpointPhysicalIdentity
    checkpoint_execution_manifest_physical_identity: ConfirmatoryCheckpointPhysicalIdentity
    completed_epochs_before: int
    stopped_early_before: bool
    successful_steps_before: int
    completed_epochs_after: int
    trained_epochs: int
    successful_steps_after: int
    successful_steps_this_invocation: int
    resumed: bool
    telemetry: dict[str, Any]


def _fit_checkpoint_bound_fold(
    *,
    classifier_type: Any,
    fold_config: ConfirmatoryCNNConfig,
    checkpoint_path: Path,
    directive: ConfirmatoryCheckpointDirective,
    cpu_test_only: bool,
    training_images: ImageArray,
    observed_training_labels: NDArray[np.int64],
    fit_arguments: Mapping[str, Any],
) -> _CheckpointBoundFitResult:
    """Consume, fit, publish, and validate one fold under one ownership lock."""

    checkpoint_io = _OwnedCheckpointIO.acquire(checkpoint_path, directive)
    try:
        checkpoint_sha256_before = checkpoint_io.sha256_before
        checkpoint_size_before = checkpoint_io.size_before
        completed_epochs_before = 0
        stopped_early_before = False
        successful_steps_before = 0
        resume_bytes = checkpoint_io.resume_bytes
        if directive.action != "fresh_fit":
            if resume_bytes is None:
                raise ConfirmatoryCheckpointContractError(
                    "successor directive lacks its pinned checkpoint bytes"
                )
            state_before = _read_checkpoint_state(
                resume_bytes,
                directive,
                cpu_test_only=cpu_test_only,
            )
            if (
                state_before.completed_epochs != directive.completed_epochs_before_fit
                or state_before.stopped_early != directive.stopped_early_before_fit
            ):
                raise ConfirmatoryCheckpointContractError(
                    "authorized checkpoint epoch state differs from its directive"
                )
            completed_epochs_before = state_before.completed_epochs
            stopped_early_before = state_before.stopped_early
            successful_steps_before = state_before.successful_optimiser_steps

        classifier = classifier_type(fold_config)
        try:
            classifier.fit(
                training_images,
                observed_training_labels,
                **dict(fit_arguments),
                checkpoint_path=checkpoint_io.checkpoint_path,
                resume=checkpoint_io.should_resume,
            )
        except ConfirmatoryCheckpointContractError:
            raise
        except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
            if resume_bytes is not None:
                raise ConfirmatoryCheckpointContractError(
                    "classifier failed while consuming the exact pinned resume checkpoint"
                ) from exc
            raise

        working_checkpoint = checkpoint_io.read_working_checkpoint()
        completed_epochs_after = int(classifier.completed_epochs_)
        trained_epochs = completed_epochs_after - completed_epochs_before
        telemetry = dict(classifier.telemetry_)
        successful_steps_after = telemetry.get("successful_optimiser_steps", 0)
        if type(successful_steps_after) is not int:
            raise ConfirmatoryCheckpointContractError(
                "post-fit optimiser-step evidence is not an exact integer"
            )
        successful_steps_this_invocation = successful_steps_after - successful_steps_before
        if trained_epochs < 0 or successful_steps_this_invocation < 0:
            raise ConfirmatoryCheckpointContractError("checkpoint continuation counters regressed")
        expected_data_and_split = _strict_json_object(
            directive.expected_data_and_split_json,
            role="post-fit expected data/split binding",
        )
        if dict(classifier.data_and_split_sha256_) != expected_data_and_split:
            raise ConfirmatoryCheckpointContractError(
                "fit-time data/split fingerprints differ from the directive"
            )
        if completed_epochs_after > directive.maximum_epochs:
            raise ConfirmatoryCheckpointContractError(
                "checkpoint continuation exceeded the predeclared maximum epoch"
            )
        if not cpu_test_only:
            expected_configuration = _strict_json_object(
                directive.expected_configuration_json,
                role="post-fit expected configuration",
            )
            expected_metadata = _strict_json_object(
                directive.expected_model_metadata_json,
                role="post-fit expected model metadata",
            )
            try:
                _validate_confirmatory_checkpoint_bytes(
                    working_checkpoint.payload,
                    expected_configuration=expected_configuration,
                    expected_model_metadata=expected_metadata,
                    expected_data_and_split_sha256=cast(
                        Mapping[str, str],
                        expected_data_and_split,
                    ),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ConfirmatoryCheckpointContractError(
                    "post-fit checkpoint failed strict production validation"
                ) from exc

        if directive.action == "fresh_fit":
            if (
                completed_epochs_before != 0
                or trained_epochs <= 0
                or successful_steps_this_invocation <= 0
            ):
                raise ConfirmatoryCheckpointContractError(
                    "fresh directive did not start at epoch zero and train"
                )
        elif directive.action == "resume_incomplete_fit":
            history = tuple(getattr(classifier, "history_", ()))
            if (
                completed_epochs_before != directive.next_epoch_index
                or trained_epochs <= 0
                or successful_steps_this_invocation <= 0
                or len(history) < completed_epochs_before + 1
                or not isinstance(history[completed_epochs_before], Mapping)
                or history[completed_epochs_before].get("epoch") != directive.next_epoch_index + 1
            ):
                raise ConfirmatoryCheckpointContractError(
                    "incomplete checkpoint did not continue at the exact next epoch"
                )
        elif (
            trained_epochs != 0
            or successful_steps_this_invocation != 0
            or checkpoint_sha256_before != working_checkpoint.sha256
            or checkpoint_size_before != working_checkpoint.size_bytes
            or completed_epochs_after != directive.completed_epochs_before_fit
            or bool(getattr(classifier, "stopped_early_", False)) != stopped_early_before
        ):
            raise ConfirmatoryCheckpointContractError(
                "terminal checkpoint trained or changed instead of zero-step restoration"
            )

        if directive.action == "restore_terminal_checkpoint_without_fit":
            checkpoint_io.adopt_terminal_working_checkpoint(working_checkpoint)
        else:
            checkpoint_io.publish_completed_working_checkpoint(
                working_checkpoint,
                completed_epochs=completed_epochs_after,
                trained_epochs=trained_epochs,
            )
        checkpoint_after = checkpoint_io.read_current_bytes()
        checkpoint_io.record_execution_manifest(
            completed_epochs_after=completed_epochs_after,
            trained_epochs=trained_epochs,
        )
        checkpoint_execution_manifest_path = checkpoint_io.current_manifest_path
        if checkpoint_execution_manifest_path is None:
            raise ConfirmatoryCheckpointContractError(
                "fit lacks its immutable checkpoint execution manifest"
            )
        checkpoint_execution_manifest = checkpoint_io.read_current_manifest_bytes()

        return _CheckpointBoundFitResult(
            classifier=classifier,
            checkpoint_path_after=checkpoint_io.published_checkpoint_path,
            checkpoint_execution_manifest_path=checkpoint_execution_manifest_path,
            checkpoint_execution_manifest_sha256=(checkpoint_execution_manifest.sha256),
            checkpoint_sha256_before=checkpoint_sha256_before,
            checkpoint_size_before=checkpoint_size_before,
            checkpoint_sha256_after=checkpoint_after.sha256,
            checkpoint_size_after=checkpoint_after.size_bytes,
            checkpoint_physical_identity_after=checkpoint_after.identity,
            checkpoint_execution_manifest_physical_identity=(
                checkpoint_execution_manifest.identity
            ),
            completed_epochs_before=completed_epochs_before,
            stopped_early_before=stopped_early_before,
            successful_steps_before=successful_steps_before,
            completed_epochs_after=completed_epochs_after,
            trained_epochs=trained_epochs,
            successful_steps_after=successful_steps_after,
            successful_steps_this_invocation=successful_steps_this_invocation,
            resumed=resume_bytes is not None,
            telemetry=telemetry,
        )
    finally:
        checkpoint_io.close()


def grouped_oof_confirmatory_cnn(
    audit_rgb: ImageArray,
    observed_labels: Sequence[int] | NDArray[np.integer],
    pre_corruption_labels: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str] | NDArray[np.str_],
    *,
    sample_ids: Sequence[str] | NDArray[np.str_],
    audit_target_masks: ImageArray | None,
    reference_validation_rgb: ImageArray,
    reference_validation_labels: Sequence[int] | NDArray[np.integer],
    reference_validation_sample_ids: Sequence[str] | NDArray[np.str_],
    reference_validation_group_ids: Sequence[str] | NDArray[np.str_],
    reference_validation_target_masks: ImageArray | None,
    final_reference_group_ids: Collection[str],
    base_config: ConfirmatoryCNNConfig,
    cell_id: str,
    checkpoint_directory: str | Path,
    checkpoint_execution_contract: ConfirmatoryCheckpointExecutionContract,
    cpu_test_only: bool,
    n_splits: int = 5,
    split_seed: int = 23,
) -> ConfirmatoryImageOOFResult:
    """Train one fresh confirmatory CNN per group-disjoint audit fold.

    ``observed_labels`` are used for fitting. ``pre_corruption_labels`` are
    used only to deterministically freeze fold membership.  The only accepted
    evidence about the final reference test is its non-empty group-ID set.
    """

    base_config.validate()
    if not isinstance(checkpoint_execution_contract, ConfirmatoryCheckpointExecutionContract):
        raise ConfirmatoryCheckpointContractError(
            "image OOF requires one typed checkpoint execution contract"
        )
    if cpu_test_only:
        if checkpoint_execution_contract.contract_profile != "cpu_test_only":
            raise ConfirmatoryCheckpointContractError(
                "CPU test execution requires the explicit test-only checkpoint profile"
            )
        expected_contract_count = len(checkpoint_execution_contract.directives)
    elif checkpoint_execution_contract.contract_profile == ("original_confirmatory_exact_180"):
        expected_contract_count = 180
    elif checkpoint_execution_contract.contract_profile == (
        "resource_bounded_confirmatory_exact_30"
    ):
        expected_contract_count = 30
    else:
        raise ConfirmatoryCheckpointContractError(
            "production image OOF requires a production checkpoint profile"
        )
    checkpoint_execution_contract.validate(expected_directive_count=expected_contract_count)
    if checkpoint_execution_contract.contract_profile == ("original_confirmatory_exact_180"):
        require_original_confirmatory_checkpoint_authority_binding(checkpoint_execution_contract)
    audit_count = _image_count(audit_rgb, name="audit_rgb")
    validation_count = _image_count(
        reference_validation_rgb,
        name="reference_validation_rgb",
    )
    observed = _integer_vector(
        observed_labels,
        audit_count,
        name="observed_labels",
    )
    assignment_labels = _integer_vector(
        pre_corruption_labels,
        audit_count,
        name="pre_corruption_labels",
    )
    identifiers = _identifier_vector(
        sample_ids,
        audit_count,
        name="sample_ids",
        unique=True,
    )
    groups = _identifier_vector(
        group_ids,
        audit_count,
        name="group_ids",
        unique=False,
    )
    validation_labels = _integer_vector(
        reference_validation_labels,
        validation_count,
        name="reference_validation_labels",
    )
    validation_sample_ids = _identifier_vector(
        reference_validation_sample_ids,
        validation_count,
        name="reference_validation_sample_ids",
        unique=True,
    )
    validation_groups = _identifier_vector(
        reference_validation_group_ids,
        validation_count,
        name="reference_validation_group_ids",
        unique=False,
    )

    audit_group_set = set(groups.tolist())
    validation_group_set = set(validation_groups.tolist())
    final_groups = {str(value) for value in final_reference_group_ids}
    if not final_groups or any(not value for value in final_groups):
        raise ValueError("non-empty final-reference group evidence is mandatory")
    if overlap := audit_group_set.intersection(validation_group_set):
        raise ValueError(f"audit/reference-validation group leakage detected: {sorted(overlap)}")
    if overlap := audit_group_set.intersection(final_groups):
        raise ValueError(f"final-reference groups present in audit pool: {sorted(overlap)}")
    if overlap := validation_group_set.intersection(final_groups):
        raise ValueError(
            f"reference-validation/final-reference group leakage detected: {sorted(overlap)}"
        )
    if overlap := set(identifiers.tolist()).intersection(validation_sample_ids.tolist()):
        raise ValueError(f"audit/reference-validation sample IDs overlap: {sorted(overlap)}")
    if set(assignment_labels.tolist()) != set(CLASS_ORDER):
        raise ValueError("pre_corruption_labels must contain the fixed five-class order")
    if set(observed.tolist()) != set(CLASS_ORDER):
        raise ValueError("observed_labels must contain the fixed five-class order")

    if cpu_test_only:
        if base_config.weight_identifier != CPU_TEST_ONLY_WEIGHT_IDENTIFIER:
            raise ValueError(
                "cpu_test_only=True requires the explicit TEST_ONLY_RANDOM_SEEDED weight marker"
            )
        classifier_type: (
            type[ConfirmatoryCNNCPUTestOnlyAdapter] | type[ConfirmatoryResNet18Classifier]
        ) = ConfirmatoryCNNCPUTestOnlyAdapter
        execution_mode = "cpu_test_only_non_evidence"
    else:
        if base_config.weight_identifier != OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER:
            raise ValueError(
                "production image OOF requires the explicit official ImageNet weight identifier"
            )
        classifier_type = ConfirmatoryResNet18Classifier
        execution_mode = "real_study_cuda"

    folds = make_group_stratified_folds(
        assignment_labels,
        tuple(str(value) for value in groups),
        n_splits=n_splits,
        class_order=CLASS_ORDER,
        seed=split_seed,
    )
    required_classes = set(CLASS_ORDER)
    for fold in folds:
        training_classes = set(int(value) for value in observed[fold.train_indices])
        missing = sorted(required_classes.difference(training_classes))
        if missing:
            raise ValueError(
                f"image OOF fold {fold.fold_id} observed-label training partition is "
                f"missing fixed classes: {missing}"
            )

    checkpoint_root = Path(os.path.abspath(checkpoint_directory))
    _require_plain_checkpoint_directory(checkpoint_root)
    directives_by_fold = _require_fold_directives(
        checkpoint_execution_contract.directives_for_cell(
            cell_id,
            expected_fold_count=n_splits,
        ),
        cell_id=cell_id,
        n_splits=n_splits,
        checkpoint_directory=checkpoint_root,
        base_config=base_config,
    )
    probabilities = np.full((audit_count, len(CLASS_ORDER)), np.nan, dtype=np.float64)
    fold_ids = np.full(audit_count, -1, dtype=np.int64)
    coverage = np.zeros(audit_count, dtype=np.int64)
    provenance: list[OOFFoldProvenance] = []
    evidence_rows: list[ConfirmatoryImageOOFFoldEvidence] = []

    audit_rgb_array: ImageArray = (
        audit_rgb if isinstance(audit_rgb, _IndexedRows) else np.asarray(audit_rgb)
    )
    validation_rgb_array: ImageArray = (
        reference_validation_rgb
        if isinstance(reference_validation_rgb, _IndexedRows)
        else np.asarray(reference_validation_rgb)
    )
    audit_masks: ImageArray | None = (
        audit_target_masks
        if isinstance(audit_target_masks, _IndexedRows)
        else np.asarray(audit_target_masks)
        if audit_target_masks is not None
        else None
    )
    validation_masks: ImageArray | None = (
        reference_validation_target_masks
        if isinstance(reference_validation_target_masks, _IndexedRows)
        else np.asarray(reference_validation_target_masks)
        if reference_validation_target_masks is not None
        else None
    )
    validation_id_tuple = tuple(str(value) for value in validation_sample_ids)
    validation_group_tuple = tuple(sorted(str(value) for value in np.unique(validation_groups)))

    for fold in folds:
        fold_config = replace(base_config, seed=base_config.seed + fold.fold_id)
        checkpoint_path = checkpoint_root / f"fold_{fold.fold_id:02d}.pt"
        directive = directives_by_fold[fold.fold_id]
        fit_result = _fit_checkpoint_bound_fold(
            classifier_type=classifier_type,
            fold_config=fold_config,
            checkpoint_path=checkpoint_path,
            directive=directive,
            cpu_test_only=cpu_test_only,
            training_images=_select_image_rows(audit_rgb_array, fold.train_indices),
            observed_training_labels=observed[fold.train_indices],
            fit_arguments={
                "training_sample_ids": identifiers[fold.train_indices],
                "training_group_ids": groups[fold.train_indices],
                "training_target_masks": (
                    _select_image_rows(audit_masks, fold.train_indices)
                    if audit_masks is not None
                    else None
                ),
                "reference_validation_images": validation_rgb_array,
                "reference_validation_labels": validation_labels,
                "reference_validation_sample_ids": validation_sample_ids,
                "reference_validation_group_ids": validation_groups,
                "reference_validation_target_masks": validation_masks,
                "reference_validation_role": "reference_validation",
            },
        )
        classifier = fit_result.classifier
        classifier_classes = np.asarray(classifier.classes_, dtype=np.int64)
        if not np.array_equal(classifier_classes, np.asarray(CLASS_ORDER, dtype=np.int64)):
            raise ValueError(
                f"image OOF fold {fold.fold_id} classifier does not expose the fixed "
                "class order (0, 1, 2, 3, 4)"
            )
        fold_probabilities = np.asarray(
            classifier.predict_proba(
                _select_image_rows(audit_rgb_array, fold.holdout_indices),
                target_masks=(
                    _select_image_rows(audit_masks, fold.holdout_indices)
                    if audit_masks is not None
                    else None
                ),
            ),
            dtype=np.float64,
        )
        expected_shape = (len(fold.holdout_indices), len(CLASS_ORDER))
        if fold_probabilities.shape != expected_shape:
            raise ValueError(
                f"image OOF fold {fold.fold_id} returned {fold_probabilities.shape}; "
                f"expected {expected_shape}"
            )
        if not np.isfinite(fold_probabilities).all():
            raise ValueError(f"image OOF fold {fold.fold_id} returned non-finite probabilities")
        if np.any(fold_probabilities < 0.0) or np.any(fold_probabilities > 1.0):
            raise ValueError(f"image OOF fold {fold.fold_id} probabilities lie outside [0, 1]")
        row_sums = fold_probabilities.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"image OOF fold {fold.fold_id} probability rows do not sum to one")
        # CUDA softmax is evaluated in float32; normalise once in float64 so
        # the shared OOFResult's stricter persisted-probability contract holds.
        fold_probabilities = fold_probabilities / row_sums
        telemetry = fit_result.telemetry

        probabilities[fold.holdout_indices] = fold_probabilities
        fold_ids[fold.holdout_indices] = fold.fold_id
        coverage[fold.holdout_indices] += 1
        held_out_ids = tuple(str(identifiers[index]) for index in fold.holdout_indices)
        training_ids = tuple(str(identifiers[index]) for index in fold.train_indices)
        provenance.append(
            OOFFoldProvenance(
                fold_id=fold.fold_id,
                training_groups=fold.training_groups,
                held_out_groups=fold.held_out_groups,
                held_out_sample_ids=held_out_ids,
            )
        )
        if telemetry.get("execution_mode") != execution_mode:
            raise ValueError(
                f"image OOF fold {fold.fold_id} telemetry execution mode is inconsistent"
            )
        fold_eligible = bool(telemetry.get("study_outcome_eligible", False))
        if cpu_test_only and fold_eligible:
            raise ValueError(
                f"image OOF fold {fold.fold_id} incorrectly marks CPU test output eligible"
            )
        with (
            _hold_private_checkpoint_snapshot(
                fit_result.checkpoint_path_after,
                role="post-prediction final checkpoint evidence",
            ) as held_checkpoint,
            _hold_private_checkpoint_snapshot(
                fit_result.checkpoint_execution_manifest_path,
                role="post-prediction checkpoint execution manifest",
            ) as held_manifest,
        ):
            if (
                held_checkpoint.identity != fit_result.checkpoint_physical_identity_after
                or held_checkpoint.sha256 != fit_result.checkpoint_sha256_after
                or held_checkpoint.size_bytes != fit_result.checkpoint_size_after
                or held_manifest.identity
                != fit_result.checkpoint_execution_manifest_physical_identity
                or held_manifest.sha256 != fit_result.checkpoint_execution_manifest_sha256
            ):
                raise ConfirmatoryCheckpointContractError(
                    "checkpoint evidence changed after owner-session close"
                )
            evidence_rows.append(
                ConfirmatoryImageOOFFoldEvidence(
                    fold_id=fold.fold_id,
                    model_seed=fold_config.seed,
                    training_sample_ids=training_ids,
                    held_out_sample_ids=held_out_ids,
                    training_groups=fold.training_groups,
                    held_out_groups=fold.held_out_groups,
                    reference_validation_sample_ids=validation_id_tuple,
                    reference_validation_groups=validation_group_tuple,
                    checkpoint_path=str(fit_result.checkpoint_path_after),
                    checkpoint_sha256=fit_result.checkpoint_sha256_after,
                    checkpoint_size_bytes=fit_result.checkpoint_size_after,
                    checkpoint_physical_identity=(fit_result.checkpoint_physical_identity_after),
                    checkpoint_execution_manifest_path=(
                        str(fit_result.checkpoint_execution_manifest_path)
                    ),
                    checkpoint_execution_manifest_sha256=(
                        fit_result.checkpoint_execution_manifest_sha256
                    ),
                    checkpoint_execution_manifest_physical_identity=(
                        fit_result.checkpoint_execution_manifest_physical_identity
                    ),
                    configuration_sha256=_configuration_sha256(fold_config),
                    resumed_from_checkpoint=fit_result.resumed,
                    checkpoint_execution_mode=(checkpoint_execution_contract.execution_mode),
                    checkpoint_action=directive.action,
                    checkpoint_sha256_before_fit=(fit_result.checkpoint_sha256_before),
                    completed_epochs_before_fit=fit_result.completed_epochs_before,
                    trained_epochs_this_invocation=fit_result.trained_epochs,
                    successful_optimiser_steps_before_fit=(fit_result.successful_steps_before),
                    successful_optimiser_steps_after_fit=(fit_result.successful_steps_after),
                    successful_optimiser_steps_this_invocation=(
                        fit_result.successful_steps_this_invocation
                    ),
                    execution_mode=execution_mode,
                    study_outcome_eligible=fold_eligible,
                    completed_epochs=int(classifier.completed_epochs_),
                    best_epoch=classifier.best_epoch_,
                    best_reference_validation_loss=(classifier.best_validation_loss_),
                    telemetry=telemetry,
                    model_metadata=dict(classifier.model_metadata_),
                    data_and_split_sha256=dict(classifier.data_and_split_sha256_),
                )
            )

    predicted = np.asarray(CLASS_ORDER, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    oof_result = OOFResult(
        probabilities=probabilities,
        predicted_class=predicted,
        fold_id=fold_ids,
        coverage_count=coverage,
        sample_ids=tuple(str(value) for value in identifiers),
        group_ids=tuple(str(value) for value in groups),
        final_reference_groups=tuple(sorted(final_groups)),
        class_order=CLASS_ORDER,
        folds=tuple(provenance),
        model_name="confirmatory_resnet18_five_class",
        representation=base_config.input_variant,
        model_seed=base_config.seed,
        split_seed=split_seed,
        fold_assignment_labels=assignment_labels,
        fold_assignment_label_source="pre_corruption_label",
        fold_assignment_labels_sha256=_assignment_sha256(assignment_labels),
    )
    output = ConfirmatoryImageOOFResult(
        oof_result=oof_result,
        fold_evidence=tuple(evidence_rows),
        study_outcome_eligible=all(row.study_outcome_eligible for row in evidence_rows),
        execution_mode=execution_mode,
    )
    output.validate()
    return output


__all__ = [
    "CheckpointContractProfile",
    "CheckpointExecutionMode",
    "CheckpointFitAction",
    "ConfirmatoryCheckpointContractError",
    "ConfirmatoryCheckpointDirective",
    "ConfirmatoryCheckpointExecutionContract",
    "ConfirmatoryCheckpointFileIdentity",
    "ConfirmatoryImageOOFFoldEvidence",
    "ConfirmatoryImageOOFResult",
    "grouped_oof_confirmatory_cnn",
    "require_original_confirmatory_checkpoint_authority_binding",
]
