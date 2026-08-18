"""Bounded checkpoint import from one explicit read-only predecessor.

This module deliberately has no run discovery, training, OOF, metric, ranking,
retry, or fallback path.  A caller supplies the complete canonical checkpoint
allowlist and the exact validator inputs.  Existing checkpoints are validated and
physically copied; absent checkpoints remain explicit fresh-training decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.pannuke.publication import anchored_physical_copy_session

RESOURCE_BOUNDED_RESUME_POLICY = "explicit_read_only_checkpoint_successor_v1"
RESOURCE_BOUNDED_COPY_POLICY = "anchored_physical_no_overwrite_no_link_v1"
RESOURCE_BOUNDED_READ_SCOPE = "caller_allowlisted_canonical_cell_checkpoints_only"
RESOURCE_BOUNDED_RESUME_EVIDENCE_SCHEMA_VERSION = 1
RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD = "evidence_without_self_hash_sha256"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPARSE_POINT = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
_WINDOWS_RESERVED_SEGMENTS = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_DATA_AND_SPLIT_KEYS = frozenset(
    {
        "training_data_sha256",
        "reference_validation_data_sha256",
        "training_split_sha256",
        "reference_validation_split_sha256",
    }
)
_DEFAULT_COPY_CHUNK_SIZE_BYTES = 8 * 1024 * 1024


def _is_safe_segment(value: object) -> bool:
    if not isinstance(value, str) or _SAFE_SEGMENT.fullmatch(value) is None:
        return False
    basename = value.split(".", 1)[0].rstrip(" .").casefold()
    return not value.endswith(".") and basename not in _WINDOWS_RESERVED_SEGMENTS


class ResourceBoundedResumeError(RuntimeError):
    """Fail-closed predecessor qualification or checkpoint-copy error."""


class StrictCheckpointValidator(Protocol):
    """Exact callback contract implemented by the confirmatory checkpoint validator."""

    def __call__(
        self,
        path: str | Path,
        *,
        expected_configuration: Mapping[str, Any],
        expected_model_metadata: Mapping[str, Any],
        expected_data_and_split_sha256: Mapping[str, str],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ResumeCheckpointExpectation:
    """One caller-authorized canonical cell/fold checkpoint and its exact bindings."""

    relative_path: str
    cell_id: str
    fold_id: int
    expected_configuration: Mapping[str, Any]
    expected_model_metadata: Mapping[str, Any]
    expected_data_and_split_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class BoundResumeCheckpointExpectation:
    """Immutable JSON encoding of one validated caller expectation."""

    relative_path: str
    cell_id: str
    fold_id: int
    configuration_json: str
    configuration_sha256: str
    model_metadata_json: str
    model_metadata_sha256: str
    data_and_split_json: str
    data_and_split_binding_sha256: str

    def validator_arguments(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        configuration = json.loads(self.configuration_json)
        model_metadata = json.loads(self.model_metadata_json)
        data_and_split = json.loads(self.data_and_split_json)
        if not all(
            isinstance(value, dict) for value in (configuration, model_metadata, data_and_split)
        ):
            raise RuntimeError("bound resume metadata no longer decodes to JSON objects")
        return (
            cast(dict[str, Any], configuration),
            cast(dict[str, Any], model_metadata),
            cast(dict[str, str], data_and_split),
        )

    def evidence_dict(self) -> dict[str, Any]:
        _, _, data_and_split = self.validator_arguments()
        return {
            "relative_path": self.relative_path,
            "cell_id": self.cell_id,
            "fold_id": self.fold_id,
            "configuration_sha256": self.configuration_sha256,
            "model_metadata_sha256": self.model_metadata_sha256,
            "data_and_split_sha256": data_and_split,
            "data_and_split_binding_sha256": self.data_and_split_binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResumePathIdentity:
    """Stable identity fields used to detect replacement or in-place mutation."""

    device: int
    inode: int
    size_bytes: int
    mode: int
    link_count: int
    modified_time_ns: int
    changed_time_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> ResumePathIdentity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size_bytes=int(value.st_size),
            mode=int(value.st_mode),
            link_count=int(value.st_nlink),
            modified_time_ns=int(value.st_mtime_ns),
            changed_time_ns=int(value.st_ctime_ns),
        )


@dataclass(frozen=True, slots=True)
class ResumeCheckpointSourceRecord:
    """One available resume checkpoint or one explicit fresh-fold decision."""

    relative_path: str
    cell_id: str
    fold_id: int
    decision: Literal["resume", "missing_fresh"]
    size_bytes: int | None
    sha256: str | None
    identity: ResumePathIdentity | None
    configuration_sha256: str
    model_metadata_sha256: str
    data_and_split_binding_sha256: str

    def evidence_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True, slots=True)
class ReadOnlyPredecessorSnapshot:
    """Content- and identity-bound snapshot of only the caller allowlist."""

    predecessor_directory: Path
    retry_of_run_id: str
    expectations: tuple[BoundResumeCheckpointExpectation, ...]
    records: tuple[ResumeCheckpointSourceRecord, ...]
    predecessor_root_identity: ResumePathIdentity
    allowlist_sha256: str
    inventory_sha256: str
    snapshot_sha256: str

    @property
    def resume_records(self) -> tuple[ResumeCheckpointSourceRecord, ...]:
        return tuple(record for record in self.records if record.decision == "resume")

    @property
    def fresh_records(self) -> tuple[ResumeCheckpointSourceRecord, ...]:
        return tuple(record for record in self.records if record.decision == "missing_fresh")

    @property
    def total_copy_bytes(self) -> int:
        return sum(cast(int, record.size_bytes) for record in self.resume_records)


@dataclass(frozen=True, slots=True)
class ResumeCheckpointCopyRecord:
    """One independently created destination checkpoint."""

    relative_path: str
    cell_id: str
    fold_id: int
    size_bytes: int
    sha256: str
    source_identity: ResumePathIdentity
    destination_identity: ResumePathIdentity

    def evidence_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceBoundedResumeCopyReceipt:
    """Receipt proving an exact physical import and unchanged allowlisted source."""

    predecessor_directory: Path
    destination_directory: Path
    retry_of_run_id: str
    copy_policy: str
    allowlist_sha256: str
    source_snapshot_before_sha256: str
    source_snapshot_after_sha256: str
    source_unchanged: bool
    copied_records: tuple[ResumeCheckpointCopyRecord, ...]
    fresh_relative_paths: tuple[str, ...]
    destination_inventory_sha256: str

    @property
    def copied_checkpoint_count(self) -> int:
        return len(self.copied_records)

    @property
    def copied_bytes(self) -> int:
        return sum(record.size_bytes for record in self.copied_records)


def _canonical_json_object(value: Mapping[str, Any], *, role: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ResourceBoundedResumeError(f"{role} must be a JSON mapping")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResourceBoundedResumeError(f"{role} is not strict JSON metadata") from exc
    if not isinstance(decoded, dict):
        raise ResourceBoundedResumeError(f"{role} must encode one JSON object")
    return encoded, cast(dict[str, Any], decoded)


def _bind_expectation(value: ResumeCheckpointExpectation) -> BoundResumeCheckpointExpectation:
    if not isinstance(value, ResumeCheckpointExpectation):
        raise TypeError("checkpoint allowlist entries must be ResumeCheckpointExpectation values")
    if (
        not _is_safe_segment(value.cell_id)
        or type(value.fold_id) is not int
        or not 0 <= value.fold_id <= 99
    ):
        raise ResourceBoundedResumeError("checkpoint cell/fold identity is unsafe")
    canonical_path = f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
    if value.relative_path != canonical_path:
        raise ResourceBoundedResumeError(
            "checkpoint allowlist path is not the exact canonical cell/fold path"
        )
    configuration_json, configuration = _canonical_json_object(
        value.expected_configuration,
        role=f"{canonical_path} expected configuration",
    )
    model_metadata_json, model_metadata = _canonical_json_object(
        value.expected_model_metadata,
        role=f"{canonical_path} expected model metadata",
    )
    data_and_split_json, data_and_split = _canonical_json_object(
        value.expected_data_and_split_sha256,
        role=f"{canonical_path} expected data/split fingerprints",
    )
    if (
        type(configuration.get("epochs")) is not int
        or cast(int, configuration["epochs"]) < 1
        or not model_metadata
        or set(data_and_split) != _DATA_AND_SPLIT_KEYS
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in data_and_split.values()
        )
    ):
        raise ResourceBoundedResumeError(
            f"{canonical_path} lacks exact configuration/model/data-split metadata"
        )
    return BoundResumeCheckpointExpectation(
        relative_path=canonical_path,
        cell_id=value.cell_id,
        fold_id=value.fold_id,
        configuration_json=configuration_json,
        configuration_sha256=hashlib.sha256(configuration_json.encode("utf-8")).hexdigest(),
        model_metadata_json=model_metadata_json,
        model_metadata_sha256=hashlib.sha256(model_metadata_json.encode("utf-8")).hexdigest(),
        data_and_split_json=data_and_split_json,
        data_and_split_binding_sha256=hashlib.sha256(
            data_and_split_json.encode("utf-8")
        ).hexdigest(),
    )


def _bind_allowlist(
    values: Sequence[ResumeCheckpointExpectation],
) -> tuple[BoundResumeCheckpointExpectation, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ResourceBoundedResumeError("checkpoint allowlist must be one non-empty sequence")
    bound = tuple(
        sorted((_bind_expectation(value) for value in values), key=lambda item: item.relative_path)
    )
    paths = [item.relative_path for item in bound]
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise ResourceBoundedResumeError("checkpoint allowlist contains duplicate canonical paths")
    return bound


def _validate_bound_expectation(value: BoundResumeCheckpointExpectation) -> None:
    if not isinstance(value, BoundResumeCheckpointExpectation):
        raise ResourceBoundedResumeError("bound checkpoint allowlist entry has an invalid type")
    if (
        not _is_safe_segment(value.cell_id)
        or type(value.fold_id) is not int
        or not 0 <= value.fold_id <= 99
        or value.relative_path != f"cells/{value.cell_id}/checkpoints/fold_{value.fold_id:02d}.pt"
    ):
        raise ResourceBoundedResumeError("bound checkpoint path/cell/fold is not canonical")
    try:
        configuration = json.loads(value.configuration_json)
        model_metadata = json.loads(value.model_metadata_json)
        data_and_split = json.loads(value.data_and_split_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResourceBoundedResumeError("bound checkpoint metadata JSON is invalid") from exc
    if not all(isinstance(item, dict) for item in (configuration, model_metadata, data_and_split)):
        raise ResourceBoundedResumeError("bound checkpoint metadata is not three JSON objects")
    canonical_configuration, decoded_configuration = _canonical_json_object(
        cast(dict[str, Any], configuration),
        role=f"{value.relative_path} bound configuration",
    )
    canonical_model_metadata, decoded_model_metadata = _canonical_json_object(
        cast(dict[str, Any], model_metadata),
        role=f"{value.relative_path} bound model metadata",
    )
    canonical_data_and_split, decoded_data_and_split = _canonical_json_object(
        cast(dict[str, Any], data_and_split),
        role=f"{value.relative_path} bound data/split fingerprints",
    )
    if (
        value.configuration_json != canonical_configuration
        or value.model_metadata_json != canonical_model_metadata
        or value.data_and_split_json != canonical_data_and_split
        or value.configuration_sha256
        != hashlib.sha256(canonical_configuration.encode("utf-8")).hexdigest()
        or value.model_metadata_sha256
        != hashlib.sha256(canonical_model_metadata.encode("utf-8")).hexdigest()
        or value.data_and_split_binding_sha256
        != hashlib.sha256(canonical_data_and_split.encode("utf-8")).hexdigest()
        or type(decoded_configuration.get("epochs")) is not int
        or cast(int, decoded_configuration["epochs"]) < 1
        or not decoded_model_metadata
        or set(decoded_data_and_split) != _DATA_AND_SPLIT_KEYS
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in decoded_data_and_split.values()
        )
    ):
        raise ResourceBoundedResumeError("bound checkpoint metadata/hash binding is invalid")


def _validate_bound_allowlist(
    values: tuple[BoundResumeCheckpointExpectation, ...],
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ResourceBoundedResumeError("bound checkpoint allowlist must be a non-empty tuple")
    for value in values:
        _validate_bound_expectation(value)
    paths = [value.relative_path for value in values]
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or len(paths) != len({path.casefold() for path in paths})
    ):
        raise ResourceBoundedResumeError(
            "bound checkpoint allowlist is unsorted or contains duplicate paths"
        )


def _is_link_or_reparse(path: Path, value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return path.is_symlink() or stat.S_ISLNK(value.st_mode) or bool(attributes & _REPARSE_POINT)


def _plain_directory_identity(path: Path, *, role: str) -> ResumePathIdentity:
    if not path.is_absolute() or not path.anchor:
        raise ResourceBoundedResumeError(f"{role} must be one absolute lexical path")
    components = path.parts
    current = Path(components[0])
    final_value: os.stat_result | None = None
    for index, component in enumerate(components):
        if index > 0:
            current /= component
        try:
            value = current.lstat()
        except OSError as exc:
            raise ResourceBoundedResumeError(
                f"{role} has a missing or inaccessible component: {current}"
            ) from exc
        if not stat.S_ISDIR(value.st_mode) or _is_link_or_reparse(current, value):
            raise ResourceBoundedResumeError(
                f"{role} contains a non-directory link or reparse component: {current}"
            )
        final_value = value
    if final_value is None:
        raise RuntimeError("absolute directory path unexpectedly had no components")
    return ResumePathIdentity.from_stat(final_value)


def _checkpoint_path_state(
    root: Path,
    expectation: BoundResumeCheckpointExpectation,
    *,
    role: str,
) -> tuple[Path, os.stat_result | None]:
    current = root
    parts = expectation.relative_path.split("/")
    for index, component in enumerate(parts):
        current /= component
        try:
            value = current.lstat()
        except FileNotFoundError:
            return current, None
        except OSError as exc:
            raise ResourceBoundedResumeError(
                f"{role} canonical path is inaccessible: {current}"
            ) from exc
        if _is_link_or_reparse(current, value):
            raise ResourceBoundedResumeError(
                f"{role} canonical path contains a link or reparse point: {current}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(value.st_mode):
            raise ResourceBoundedResumeError(
                f"{role} canonical parent is not a directory: {current}"
            )
        if index == len(parts) - 1:
            return current, value
    raise RuntimeError("canonical checkpoint path unexpectedly had no components")


def _stable_file_identity(value: os.stat_result) -> ResumePathIdentity:
    return ResumePathIdentity.from_stat(value)


def _same_file_object(left: ResumePathIdentity, right: ResumePathIdentity) -> bool:
    return (
        left.device,
        left.inode,
        left.size_bytes,
        stat.S_IFMT(left.mode),
        left.link_count,
    ) == (
        right.device,
        right.inode,
        right.size_bytes,
        stat.S_IFMT(right.mode),
        right.link_count,
    )


def _hash_plain_checkpoint(
    path: Path,
    initial: os.stat_result,
    *,
    role: str,
    chunk_size_bytes: int = _DEFAULT_COPY_CHUNK_SIZE_BYTES,
) -> tuple[ResumePathIdentity, str]:
    if (
        not stat.S_ISREG(initial.st_mode)
        or _is_link_or_reparse(path, initial)
        or initial.st_nlink != 1
    ):
        raise ResourceBoundedResumeError(f"{role} is not one private regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        initial_identity = _stable_file_identity(initial)
        opened_identity = _stable_file_identity(opened)
        if (
            not _same_file_object(opened_identity, initial_identity)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise ResourceBoundedResumeError(f"{role} changed while it was opened: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, chunk_size_bytes):
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        try:
            lexical_after = path.lstat()
        except OSError as exc:
            raise ResourceBoundedResumeError(f"{role} disappeared while hashing: {path}") from exc
        opened_after_identity = _stable_file_identity(opened_after)
        lexical_after_identity = _stable_file_identity(lexical_after)
        if (
            opened_after_identity != opened_identity
            or lexical_after_identity != initial_identity
            or not _same_file_object(lexical_after_identity, opened_after_identity)
            or _is_link_or_reparse(path, lexical_after)
        ):
            raise ResourceBoundedResumeError(f"{role} changed while hashing: {path}")
        return initial_identity, digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _run_strict_validator(
    validator: StrictCheckpointValidator,
    path: Path,
    expectation: BoundResumeCheckpointExpectation,
) -> None:
    configuration, model_metadata, data_and_split = expectation.validator_arguments()
    try:
        validator(
            path,
            expected_configuration=configuration,
            expected_model_metadata=model_metadata,
            expected_data_and_split_sha256=data_and_split,
        )
    except Exception as exc:
        raise ResourceBoundedResumeError(
            f"strict checkpoint validation failed: {expectation.relative_path}"
        ) from exc


def _capture_record(
    root: Path,
    expectation: BoundResumeCheckpointExpectation,
    *,
    validator: StrictCheckpointValidator | None,
    role: str,
) -> ResumeCheckpointSourceRecord:
    path, observed = _checkpoint_path_state(root, expectation, role=role)
    if observed is None:
        return ResumeCheckpointSourceRecord(
            relative_path=expectation.relative_path,
            cell_id=expectation.cell_id,
            fold_id=expectation.fold_id,
            decision="missing_fresh",
            size_bytes=None,
            sha256=None,
            identity=None,
            configuration_sha256=expectation.configuration_sha256,
            model_metadata_sha256=expectation.model_metadata_sha256,
            data_and_split_binding_sha256=expectation.data_and_split_binding_sha256,
        )
    identity, digest = _hash_plain_checkpoint(path, observed, role=role)
    if validator is not None:
        _run_strict_validator(validator, path, expectation)
        path_after, observed_after = _checkpoint_path_state(root, expectation, role=role)
        if path_after != path or observed_after is None:
            raise ResourceBoundedResumeError(
                f"{role} changed during strict validation: {expectation.relative_path}"
            )
        identity_after, digest_after = _hash_plain_checkpoint(path, observed_after, role=role)
        if identity_after != identity or digest_after != digest:
            raise ResourceBoundedResumeError(
                f"{role} changed during strict validation: {expectation.relative_path}"
            )
    return ResumeCheckpointSourceRecord(
        relative_path=expectation.relative_path,
        cell_id=expectation.cell_id,
        fold_id=expectation.fold_id,
        decision="resume",
        size_bytes=identity.size_bytes,
        sha256=digest,
        identity=identity,
        configuration_sha256=expectation.configuration_sha256,
        model_metadata_sha256=expectation.model_metadata_sha256,
        data_and_split_binding_sha256=expectation.data_and_split_binding_sha256,
    )


def _validate_source_record(
    record: ResumeCheckpointSourceRecord,
    expectation: BoundResumeCheckpointExpectation,
) -> None:
    if (
        not isinstance(record, ResumeCheckpointSourceRecord)
        or record.relative_path != expectation.relative_path
        or record.cell_id != expectation.cell_id
        or record.fold_id != expectation.fold_id
        or record.configuration_sha256 != expectation.configuration_sha256
        or record.model_metadata_sha256 != expectation.model_metadata_sha256
        or record.data_and_split_binding_sha256 != expectation.data_and_split_binding_sha256
    ):
        raise ResourceBoundedResumeError("resume source record differs from its allowlist entry")
    if record.decision == "missing_fresh":
        if any(value is not None for value in (record.size_bytes, record.sha256, record.identity)):
            raise ResourceBoundedResumeError("fresh resume decision contains file evidence")
        return
    if (
        record.decision != "resume"
        or type(record.size_bytes) is not int
        or record.size_bytes < 0
        or not isinstance(record.sha256, str)
        or _SHA256.fullmatch(record.sha256) is None
        or not isinstance(record.identity, ResumePathIdentity)
        or record.identity.size_bytes != record.size_bytes
        or not stat.S_ISREG(record.identity.mode)
        or record.identity.link_count != 1
    ):
        raise ResourceBoundedResumeError("resume source record has invalid file evidence")


def _snapshot_semantic_sha256(
    *,
    predecessor_directory: Path,
    retry_of_run_id: str,
    predecessor_root_identity: ResumePathIdentity,
    allowlist_sha256: str,
    inventory_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "policy": RESOURCE_BOUNDED_RESUME_POLICY,
            "predecessor_directory": str(predecessor_directory),
            "retry_of_run_id": retry_of_run_id,
            "predecessor_root_identity": asdict(predecessor_root_identity),
            "allowlist_sha256": allowlist_sha256,
            "inventory_sha256": inventory_sha256,
        }
    )


def _validate_snapshot_structure(snapshot: ReadOnlyPredecessorSnapshot) -> None:
    if not isinstance(snapshot, ReadOnlyPredecessorSnapshot):
        raise TypeError("resume operation requires a typed predecessor snapshot")
    predecessor = Path(os.path.abspath(snapshot.predecessor_directory.expanduser()))
    if (
        predecessor != snapshot.predecessor_directory
        or not _is_safe_segment(snapshot.retry_of_run_id)
        or predecessor.name != snapshot.retry_of_run_id
        or not isinstance(snapshot.predecessor_root_identity, ResumePathIdentity)
        or not stat.S_ISDIR(snapshot.predecessor_root_identity.mode)
    ):
        raise ResourceBoundedResumeError("resume snapshot predecessor identity is invalid")
    _validate_bound_allowlist(snapshot.expectations)
    if not isinstance(snapshot.records, tuple) or len(snapshot.records) != len(
        snapshot.expectations
    ):
        raise ResourceBoundedResumeError("resume snapshot inventory length is invalid")
    for record, expectation in zip(snapshot.records, snapshot.expectations, strict=True):
        _validate_source_record(record, expectation)
    allowlist_sha = canonical_sha256(
        [expectation.evidence_dict() for expectation in snapshot.expectations]
    )
    inventory_sha = canonical_sha256([record.evidence_dict() for record in snapshot.records])
    if (
        snapshot.allowlist_sha256 != allowlist_sha
        or snapshot.inventory_sha256 != inventory_sha
        or snapshot.snapshot_sha256
        != _snapshot_semantic_sha256(
            predecessor_directory=predecessor,
            retry_of_run_id=snapshot.retry_of_run_id,
            predecessor_root_identity=snapshot.predecessor_root_identity,
            allowlist_sha256=allowlist_sha,
            inventory_sha256=inventory_sha,
        )
    ):
        raise ResourceBoundedResumeError("resume snapshot hash binding is invalid")


def _snapshot_from_bound_allowlist(
    predecessor_directory: Path,
    *,
    retry_of_run_id: str,
    expectations: tuple[BoundResumeCheckpointExpectation, ...],
    validator: StrictCheckpointValidator | None,
) -> ReadOnlyPredecessorSnapshot:
    _validate_bound_allowlist(expectations)
    root_before = _plain_directory_identity(
        predecessor_directory,
        role="resume predecessor directory",
    )
    records = tuple(
        _capture_record(
            predecessor_directory,
            expectation,
            validator=validator,
            role="resume predecessor checkpoint",
        )
        for expectation in expectations
    )
    root_after = _plain_directory_identity(
        predecessor_directory,
        role="resume predecessor directory",
    )
    if root_after != root_before:
        raise ResourceBoundedResumeError("resume predecessor root changed during snapshot")
    allowlist_payload = [expectation.evidence_dict() for expectation in expectations]
    inventory_payload = [record.evidence_dict() for record in records]
    allowlist_sha = canonical_sha256(allowlist_payload)
    inventory_sha = canonical_sha256(inventory_payload)
    return ReadOnlyPredecessorSnapshot(
        predecessor_directory=predecessor_directory,
        retry_of_run_id=retry_of_run_id,
        expectations=expectations,
        records=records,
        predecessor_root_identity=root_before,
        allowlist_sha256=allowlist_sha,
        inventory_sha256=inventory_sha,
        snapshot_sha256=_snapshot_semantic_sha256(
            predecessor_directory=predecessor_directory,
            retry_of_run_id=retry_of_run_id,
            predecessor_root_identity=root_before,
            allowlist_sha256=allowlist_sha,
            inventory_sha256=inventory_sha,
        ),
    )


def inspect_read_only_resume_predecessor(
    predecessor_directory: str | Path,
    *,
    retry_of_run_id: str,
    checkpoint_allowlist: Sequence[ResumeCheckpointExpectation],
    validator: StrictCheckpointValidator,
) -> ReadOnlyPredecessorSnapshot:
    """Validate one explicit predecessor and only the supplied canonical allowlist."""

    if not callable(validator):
        raise TypeError("strict checkpoint validator must be callable")
    if not _is_safe_segment(retry_of_run_id):
        raise ResourceBoundedResumeError("retry_of_run_id must be one safe non-empty run ID")
    predecessor = Path(os.path.abspath(Path(predecessor_directory).expanduser()))
    if predecessor.name != retry_of_run_id:
        raise ResourceBoundedResumeError(
            "retry_of_run_id must exactly equal the predecessor directory name"
        )
    expectations = _bind_allowlist(checkpoint_allowlist)
    return _snapshot_from_bound_allowlist(
        predecessor,
        retry_of_run_id=retry_of_run_id,
        expectations=expectations,
        validator=validator,
    )


def _require_same_source_snapshot(
    expected: ReadOnlyPredecessorSnapshot,
    observed: ReadOnlyPredecessorSnapshot,
    *,
    phase: str,
) -> None:
    if (
        observed.predecessor_directory != expected.predecessor_directory
        or observed.retry_of_run_id != expected.retry_of_run_id
        or observed.allowlist_sha256 != expected.allowlist_sha256
        or observed.inventory_sha256 != expected.inventory_sha256
        or observed.snapshot_sha256 != expected.snapshot_sha256
        or observed.records != expected.records
    ):
        raise ResourceBoundedResumeError(f"resume predecessor changed {phase}")


def _require_destination_allowlist_absent(
    destination: Path,
    expectations: tuple[BoundResumeCheckpointExpectation, ...],
) -> None:
    _plain_directory_identity(destination, role="resume destination directory")
    for expectation in expectations:
        path, observed = _checkpoint_path_state(
            destination,
            expectation,
            role="resume destination checkpoint",
        )
        if observed is not None:
            raise ResourceBoundedResumeError(
                f"resume destination checkpoint already exists: {path}"
            )


def copy_validated_resume_checkpoints(
    snapshot: ReadOnlyPredecessorSnapshot,
    destination_directory: str | Path,
    *,
    validator: StrictCheckpointValidator,
    chunk_size_bytes: int = _DEFAULT_COPY_CHUNK_SIZE_BYTES,
) -> ResourceBoundedResumeCopyReceipt:
    """Physically import existing allowlisted checkpoints into one new destination."""

    _validate_snapshot_structure(snapshot)
    if not callable(validator):
        raise TypeError("strict checkpoint validator must be callable")
    if type(chunk_size_bytes) is not int or chunk_size_bytes < 1:
        raise ValueError("resume copy chunk_size_bytes must be a positive integer")
    destination = Path(os.path.abspath(Path(destination_directory).expanduser()))
    if destination == snapshot.predecessor_directory:
        raise ResourceBoundedResumeError("resume destination must differ from predecessor")
    try:
        common = Path(os.path.commonpath((snapshot.predecessor_directory, destination)))
    except ValueError:
        common = None
    if common is not None and common in {snapshot.predecessor_directory, destination}:
        raise ResourceBoundedResumeError(
            "resume predecessor and destination must not contain one another"
        )

    current_before_copy = _snapshot_from_bound_allowlist(
        snapshot.predecessor_directory,
        retry_of_run_id=snapshot.retry_of_run_id,
        expectations=snapshot.expectations,
        validator=None,
    )
    _require_same_source_snapshot(snapshot, current_before_copy, phase="before copy")
    _require_destination_allowlist_absent(destination, snapshot.expectations)

    expectations_by_path = {
        expectation.relative_path: expectation for expectation in snapshot.expectations
    }
    copied: list[ResumeCheckpointCopyRecord] = []
    published_identity_by_path: dict[str, tuple[int, int, int, int]] = {}
    try:
        with anchored_physical_copy_session(
            snapshot.predecessor_directory,
            destination,
            chunk_size_bytes=chunk_size_bytes,
        ) as session:
            for source_record in snapshot.resume_records:
                assert source_record.size_bytes is not None
                assert source_record.sha256 is not None
                assert source_record.identity is not None
                published = session.copy_file_no_overwrite(
                    source_record.relative_path,
                    expected_size_bytes=source_record.size_bytes,
                    expected_sha256=source_record.sha256,
                )
                destination_path = destination / Path(source_record.relative_path)
                if (
                    published.path != destination_path
                    or published.kind != "file"
                    or published.sha256 != source_record.sha256
                    or published.identity[2] != source_record.size_bytes
                    or published.identity[:2]
                    == (
                        source_record.identity.device,
                        source_record.identity.inode,
                    )
                ):
                    raise ResourceBoundedResumeError(
                        f"physical-copy receipt differs: {source_record.relative_path}"
                    )
                published_identity_by_path[source_record.relative_path] = published.identity
            source_after_copy = _snapshot_from_bound_allowlist(
                snapshot.predecessor_directory,
                retry_of_run_id=snapshot.retry_of_run_id,
                expectations=snapshot.expectations,
                validator=None,
            )
            _require_same_source_snapshot(snapshot, source_after_copy, phase="during copy")

        for source_record in snapshot.resume_records:
            expectation = expectations_by_path[source_record.relative_path]
            assert source_record.size_bytes is not None
            assert source_record.sha256 is not None
            assert source_record.identity is not None
            destination_record = _capture_record(
                destination,
                expectation,
                validator=validator,
                role="resume destination checkpoint",
            )
            published_identity = published_identity_by_path[source_record.relative_path]
            if (
                destination_record.decision != "resume"
                or destination_record.size_bytes != source_record.size_bytes
                or destination_record.sha256 != source_record.sha256
                or destination_record.identity is None
                or (
                    destination_record.identity.device,
                    destination_record.identity.inode,
                    destination_record.identity.size_bytes,
                )
                != published_identity[:3]
                or (
                    destination_record.identity.device,
                    destination_record.identity.inode,
                )
                == (
                    source_record.identity.device,
                    source_record.identity.inode,
                )
                or destination_record.identity.link_count != 1
            ):
                raise ResourceBoundedResumeError(
                    f"physical-copy identity/readback differs: {source_record.relative_path}"
                )
            copied.append(
                ResumeCheckpointCopyRecord(
                    relative_path=source_record.relative_path,
                    cell_id=source_record.cell_id,
                    fold_id=source_record.fold_id,
                    size_bytes=source_record.size_bytes,
                    sha256=source_record.sha256,
                    source_identity=source_record.identity,
                    destination_identity=destination_record.identity,
                )
            )
        source_after = _snapshot_from_bound_allowlist(
            snapshot.predecessor_directory,
            retry_of_run_id=snapshot.retry_of_run_id,
            expectations=snapshot.expectations,
            validator=None,
        )
        _require_same_source_snapshot(snapshot, source_after, phase="after copy")
    except ResourceBoundedResumeError:
        raise
    except Exception as exc:
        raise ResourceBoundedResumeError("anchored resume checkpoint copy failed closed") from exc

    copied_tuple = tuple(copied)
    destination_inventory_sha = canonical_sha256(
        [record.evidence_dict() for record in copied_tuple]
    )
    return ResourceBoundedResumeCopyReceipt(
        predecessor_directory=snapshot.predecessor_directory,
        destination_directory=destination,
        retry_of_run_id=snapshot.retry_of_run_id,
        copy_policy=RESOURCE_BOUNDED_COPY_POLICY,
        allowlist_sha256=snapshot.allowlist_sha256,
        source_snapshot_before_sha256=snapshot.snapshot_sha256,
        source_snapshot_after_sha256=source_after.snapshot_sha256,
        source_unchanged=True,
        copied_records=copied_tuple,
        fresh_relative_paths=tuple(record.relative_path for record in snapshot.fresh_records),
        destination_inventory_sha256=destination_inventory_sha,
    )


def _validate_copy_receipt_structure(
    snapshot: ReadOnlyPredecessorSnapshot,
    receipt: ResourceBoundedResumeCopyReceipt,
) -> None:
    if not isinstance(receipt, ResourceBoundedResumeCopyReceipt):
        raise TypeError("resume evidence requires a typed copy receipt")
    destination = Path(os.path.abspath(receipt.destination_directory.expanduser()))
    source_records = snapshot.resume_records
    if (
        receipt.predecessor_directory != snapshot.predecessor_directory
        or destination != receipt.destination_directory
        or receipt.retry_of_run_id != snapshot.retry_of_run_id
        or receipt.copy_policy != RESOURCE_BOUNDED_COPY_POLICY
        or receipt.allowlist_sha256 != snapshot.allowlist_sha256
        or receipt.source_snapshot_before_sha256 != snapshot.snapshot_sha256
        or receipt.source_snapshot_after_sha256 != snapshot.snapshot_sha256
        or receipt.source_unchanged is not True
        or not isinstance(receipt.copied_records, tuple)
        or len(receipt.copied_records) != len(source_records)
        or receipt.fresh_relative_paths
        != tuple(record.relative_path for record in snapshot.fresh_records)
    ):
        raise ResourceBoundedResumeError("resume snapshot/copy receipt lineage differs")
    for copied, source in zip(receipt.copied_records, source_records, strict=True):
        if (
            not isinstance(copied, ResumeCheckpointCopyRecord)
            or copied.relative_path != source.relative_path
            or copied.cell_id != source.cell_id
            or copied.fold_id != source.fold_id
            or copied.size_bytes != source.size_bytes
            or copied.sha256 != source.sha256
            or copied.source_identity != source.identity
            or not isinstance(copied.destination_identity, ResumePathIdentity)
            or copied.destination_identity.size_bytes != copied.size_bytes
            or copied.destination_identity.link_count != 1
            or not stat.S_ISREG(copied.destination_identity.mode)
            or (
                copied.destination_identity.device,
                copied.destination_identity.inode,
            )
            == (
                copied.source_identity.device,
                copied.source_identity.inode,
            )
        ):
            raise ResourceBoundedResumeError("resume copy receipt has invalid checkpoint evidence")
    recomputed_destination_inventory_sha = canonical_sha256(
        [record.evidence_dict() for record in receipt.copied_records]
    )
    if receipt.destination_inventory_sha256 != recomputed_destination_inventory_sha:
        raise ResourceBoundedResumeError("resume copy receipt destination inventory hash differs")


def _resume_evidence_payload(
    *,
    run_mode: Literal["fresh", "successor_resume"],
    predecessor_directory: str | None,
    destination_directory: str | None,
    retry_of_run_id: str | None,
    copy_policy: str,
    allowlist: list[dict[str, Any]],
    allowlist_sha256: str,
    source_inventory: list[dict[str, Any]],
    source_inventory_sha256: str,
    source_snapshot_before_sha256: str | None,
    source_snapshot_after_sha256: str | None,
    imported_inventory: list[dict[str, Any]],
    imported_inventory_sha256: str,
    fresh_checkpoint_paths: list[str],
    copied_bytes: int,
    predecessor_read_performed: bool,
    strict_checkpoint_validator_invoked: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESOURCE_BOUNDED_RESUME_EVIDENCE_SCHEMA_VERSION,
        "policy": RESOURCE_BOUNDED_RESUME_POLICY,
        "run_mode": run_mode,
        "copy_policy": copy_policy,
        "read_scope": RESOURCE_BOUNDED_READ_SCOPE,
        "predecessor_directory": predecessor_directory,
        "destination_directory": destination_directory,
        "retry_of_run_id": retry_of_run_id,
        "predecessor_read_performed": predecessor_read_performed,
        "strict_checkpoint_validator_invoked": strict_checkpoint_validator_invoked,
        "allowlist": allowlist,
        "allowlist_sha256": allowlist_sha256,
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_inventory_sha256,
        "source_snapshot_before_sha256": source_snapshot_before_sha256,
        "source_snapshot_after_sha256": source_snapshot_after_sha256,
        "source_unchanged": True,
        "imported_inventory": imported_inventory,
        "imported_inventory_sha256": imported_inventory_sha256,
        "fresh_checkpoint_paths": fresh_checkpoint_paths,
        "expected_checkpoint_count": len(source_inventory),
        "imported_checkpoint_count": len(imported_inventory),
        "fresh_checkpoint_count": len(fresh_checkpoint_paths),
        "copied_bytes": copied_bytes,
        "automatic_retry_allowed": False,
        "auto_discovery_used": False,
        "oof_artifacts_read": False,
        "metrics_artifacts_read": False,
        "ranking_artifacts_read": False,
    }
    payload[RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD] = canonical_sha256(payload)
    return payload


def build_resource_bounded_resume_evidence(
    snapshot: ReadOnlyPredecessorSnapshot,
    receipt: ResourceBoundedResumeCopyReceipt,
    *,
    validator: StrictCheckpointValidator,
) -> dict[str, Any]:
    """Re-read only canonical checkpoints and build exact successor evidence."""

    if not callable(validator):
        raise TypeError("strict checkpoint validator must be callable")
    _validate_snapshot_structure(snapshot)
    _validate_copy_receipt_structure(snapshot, receipt)
    source_before_evidence = _snapshot_from_bound_allowlist(
        snapshot.predecessor_directory,
        retry_of_run_id=snapshot.retry_of_run_id,
        expectations=snapshot.expectations,
        validator=None,
    )
    _require_same_source_snapshot(
        snapshot,
        source_before_evidence,
        phase="before evidence readback",
    )
    _plain_directory_identity(
        receipt.destination_directory,
        role="resume evidence destination directory",
    )

    copied_by_path = {record.relative_path: record for record in receipt.copied_records}
    destination_records: list[ResumeCheckpointCopyRecord] = []
    for expectation, source_record in zip(
        snapshot.expectations,
        snapshot.records,
        strict=True,
    ):
        destination_record = _capture_record(
            receipt.destination_directory,
            expectation,
            validator=validator if source_record.decision == "resume" else None,
            role="resume evidence destination checkpoint",
        )
        if source_record.decision == "missing_fresh":
            if destination_record.decision != "missing_fresh":
                raise ResourceBoundedResumeError(
                    f"fresh checkpoint destination unexpectedly exists: {source_record.relative_path}"
                )
            continue
        copied = copied_by_path[source_record.relative_path]
        assert source_record.identity is not None
        assert source_record.size_bytes is not None
        assert source_record.sha256 is not None
        if (
            destination_record.decision != "resume"
            or destination_record.identity is None
            or destination_record.size_bytes != source_record.size_bytes
            or destination_record.sha256 != source_record.sha256
        ):
            raise ResourceBoundedResumeError(
                f"resume evidence destination differs: {source_record.relative_path}"
            )
        observed_copy = ResumeCheckpointCopyRecord(
            relative_path=source_record.relative_path,
            cell_id=source_record.cell_id,
            fold_id=source_record.fold_id,
            size_bytes=source_record.size_bytes,
            sha256=source_record.sha256,
            source_identity=source_record.identity,
            destination_identity=destination_record.identity,
        )
        if observed_copy != copied:
            raise ResourceBoundedResumeError(
                f"resume copy receipt differs from filesystem: {source_record.relative_path}"
            )
        destination_records.append(observed_copy)

    source_after_evidence = _snapshot_from_bound_allowlist(
        snapshot.predecessor_directory,
        retry_of_run_id=snapshot.retry_of_run_id,
        expectations=snapshot.expectations,
        validator=None,
    )
    _require_same_source_snapshot(
        snapshot,
        source_after_evidence,
        phase="after evidence readback",
    )

    expected_inventory = [expectation.evidence_dict() for expectation in snapshot.expectations]
    source_inventory = [record.evidence_dict() for record in snapshot.records]
    imported_inventory = [record.evidence_dict() for record in destination_records]
    imported_inventory_sha = canonical_sha256(imported_inventory)
    if imported_inventory_sha != receipt.destination_inventory_sha256:
        raise ResourceBoundedResumeError("resume evidence destination inventory hash differs")
    return _resume_evidence_payload(
        run_mode="successor_resume",
        predecessor_directory=str(snapshot.predecessor_directory),
        destination_directory=str(receipt.destination_directory),
        retry_of_run_id=snapshot.retry_of_run_id,
        copy_policy=receipt.copy_policy,
        allowlist=expected_inventory,
        allowlist_sha256=snapshot.allowlist_sha256,
        source_inventory=source_inventory,
        source_inventory_sha256=snapshot.inventory_sha256,
        source_snapshot_before_sha256=receipt.source_snapshot_before_sha256,
        source_snapshot_after_sha256=receipt.source_snapshot_after_sha256,
        imported_inventory=imported_inventory,
        imported_inventory_sha256=imported_inventory_sha,
        fresh_checkpoint_paths=list(receipt.fresh_relative_paths),
        copied_bytes=receipt.copied_bytes,
        predecessor_read_performed=True,
        strict_checkpoint_validator_invoked=True,
    )


def build_fresh_resource_resume_evidence(
    checkpoint_allowlist: Sequence[ResumeCheckpointExpectation],
) -> dict[str, Any]:
    """Build the same schema for a fresh run without touching any predecessor path."""

    expectations = _bind_allowlist(checkpoint_allowlist)
    _validate_bound_allowlist(expectations)
    records = [
        ResumeCheckpointSourceRecord(
            relative_path=expectation.relative_path,
            cell_id=expectation.cell_id,
            fold_id=expectation.fold_id,
            decision="missing_fresh",
            size_bytes=None,
            sha256=None,
            identity=None,
            configuration_sha256=expectation.configuration_sha256,
            model_metadata_sha256=expectation.model_metadata_sha256,
            data_and_split_binding_sha256=expectation.data_and_split_binding_sha256,
        )
        for expectation in expectations
    ]
    allowlist = [expectation.evidence_dict() for expectation in expectations]
    source_inventory = [record.evidence_dict() for record in records]
    imported_inventory: list[dict[str, Any]] = []
    return _resume_evidence_payload(
        run_mode="fresh",
        predecessor_directory=None,
        destination_directory=None,
        retry_of_run_id=None,
        copy_policy="no_copy_fresh_run",
        allowlist=allowlist,
        allowlist_sha256=canonical_sha256(allowlist),
        source_inventory=source_inventory,
        source_inventory_sha256=canonical_sha256(source_inventory),
        source_snapshot_before_sha256=None,
        source_snapshot_after_sha256=None,
        imported_inventory=imported_inventory,
        imported_inventory_sha256=canonical_sha256(imported_inventory),
        fresh_checkpoint_paths=[record.relative_path for record in records],
        copied_bytes=0,
        predecessor_read_performed=False,
        strict_checkpoint_validator_invoked=False,
    )


__all__ = [
    "RESOURCE_BOUNDED_COPY_POLICY",
    "RESOURCE_BOUNDED_READ_SCOPE",
    "RESOURCE_BOUNDED_RESUME_EVIDENCE_HASH_FIELD",
    "RESOURCE_BOUNDED_RESUME_EVIDENCE_SCHEMA_VERSION",
    "RESOURCE_BOUNDED_RESUME_POLICY",
    "ReadOnlyPredecessorSnapshot",
    "ResourceBoundedResumeCopyReceipt",
    "ResourceBoundedResumeError",
    "ResumeCheckpointCopyRecord",
    "ResumeCheckpointExpectation",
    "ResumeCheckpointSourceRecord",
    "ResumePathIdentity",
    "StrictCheckpointValidator",
    "build_fresh_resource_resume_evidence",
    "build_resource_bounded_resume_evidence",
    "copy_validated_resume_checkpoints",
    "inspect_read_only_resume_predecessor",
]
