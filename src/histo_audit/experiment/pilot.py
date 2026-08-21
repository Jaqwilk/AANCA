"""Callable, fail-closed real PanNuke pilot experiment.

This runner consumes an already validated complete release and immutable nucleus
manifest.  It never downloads PanNuke or encoder weights, never changes source
annotations, and never adapts its declared group sample to make a split pass.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray

from histo_audit.auditing.neighbours import (
    NeighbourDisagreementResult,
    fold_safe_neighbour_disagreement,
)
from histo_audit.auditing.scores import CleanlabScoreResult, cleanlab_scores, score_annotations
from histo_audit.config import load_config, resolve_config
from histo_audit.corruption.controlled import apply_controlled_corruption, canonical_sha256
from histo_audit.cross_validation.oof import (
    grouped_oof_logistic,
    make_group_stratified_fold_plan,
)
from histo_audit.evaluation.restoration import evaluate_downstream_restoration
from histo_audit.pannuke.io import inventory_raw_files, sha256_file
from histo_audit.pannuke.manifest import validate_manifest_invariants
from histo_audit.pannuke.models import (
    OFFICIAL_METRICS_CLASS_MAPPING,
    DiscoveredFold,
    ManifestArtifacts,
    PanNukeValidationResult,
    RawFileRecord,
    ReleaseDiscovery,
    ValidationArtifacts,
    VerifiedClassMapping,
)
from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock
from histo_audit.pannuke.validation import validate_discovered_release
from histo_audit.representations.eligibility import (
    DEVELOPMENT_MANIFEST_VIEW_SCOPE,
    ELIGIBILITY_POLICY,
    MANIFEST_VIEW_METADATA_KEY,
    OVERLAP_EXCLUSION_REASON,
    select_manifest_rows,
)
from histo_audit.representations.imagenet import ResNet18EmbeddingConfig
from histo_audit.representations.pannuke import (
    PanNukeCropConfig,
    PanNukeRepresentationArtifacts,
    build_pannuke_representation_cache,
)
from histo_audit.statistics.review import (
    ReviewBudgetResult,
    evaluate_review_budget,
    random_review_baseline,
    rank_indices,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    verify_run_integrity,
)

CLASS_ORDER = (0, 1, 2, 3, 4)
REQUIRED_AUDIT_METHODS = (
    "self_confidence",
    "cleanlab",
    "nearest_neighbour_disagreement",
)
CLASS_FREE_ELIGIBILITY_COLUMNS = (
    "official_fold",
    "group_id",
    "primary_eligible",
    "confirmatory_eligible",
    "cross_class_overlap_touching",
    "qc_exclusion_reason",
)
_FINAL_REFERENCE_SAMPLE_ID = re.compile(
    r"pannuke-f0*3-p\d+-c\d+-i\d+",
    flags=re.IGNORECASE,
)
_FINAL_REFERENCE_SAMPLE_ID_BYTES = re.compile(
    rb"pannuke-f0*3-p\d+-c\d+-i\d+",
    flags=re.IGNORECASE,
)
_PUBLIC_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
_FINAL_SENSITIVE_PATH_TOKENS = (
    "_ids",
    "annotation",
    "area",
    "bbox",
    "centroid",
    "class",
    "coordinate",
    "crop",
    "embedding",
    "feature",
    "geometry",
    "identity",
    "instance_id",
    "label",
    "logit",
    "mask_qc",
    "observed",
    "organ",
    "original",
    "outcome",
    "patch_qc",
    "perimeter",
    "pixel",
    "pre_corruption",
    "prediction",
    "probability",
    "quality_flag",
    "replacement",
    "representation",
    "risk",
    "sample_id",
    "score",
    "target",
    "tissue",
)
_FINAL_SENSITIVE_SCOPE_ALIASES = (
    "final_reference",
    "final_test",
    "test_final",
    "reference_test",
    "test_reference",
    "held_out_test",
    "heldout_test",
    "holdout_test",
)


@dataclass(frozen=True, slots=True)
class PanNukePilotResult:
    """Terminal paths and primary counts for one sealed pilot run."""

    run_id: str
    run_directory: Path
    metrics_path: Path
    report_path: Path
    selected_ids_path: Path
    audit_sample_count: int
    reference_validation_sample_count: int
    final_reference_sample_count: int
    exact_corruption_count: int


@dataclass(frozen=True, slots=True)
class PanNukePilotDevelopmentManifestView:
    """Checksum-bound pre-pilot development-only manifest artifact."""

    parquet_path: Path
    metadata_path: Path
    canonical_manifest_sha256: str
    development_manifest_sha256: str
    development_instance_count: int


@dataclass(frozen=True, slots=True)
class _PilotManifestInputs:
    """Privacy-partitioned canonical manifest inputs for one pilot run."""

    development_table: Any
    development_frame: Any
    development_provenance: dict[str, Any]
    class_free_frame: Any
    final_metadata_frame: Any
    global_provenance: dict[str, Any]
    canonical_manifest_sha256: str
    development_manifest_sha256: str
    view_metadata: dict[str, Any]


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"pilot configuration field {location} must be a mapping")
    return value


def _require_exact_int(value: object, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"pilot configuration field {location} must be an integer")
    return value


def _validate_seed_provenance(
    resolved: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    corruption_config: Mapping[str, Any],
) -> dict[str, int]:
    """Require one registry-compatible seed ledger consistent with execution fields."""

    split_seed = _require_exact_int(model_config.get("split_seed"), "model.split_seed")
    model_seed = _require_exact_int(model_config.get("seed"), "model.seed")
    raw_corruption_seeds = corruption_config.get("seeds")
    if not isinstance(raw_corruption_seeds, Sequence) or isinstance(
        raw_corruption_seeds, (str, bytes)
    ):
        raise ValueError("pilot configuration field corruption.seeds must be a sequence")
    if len(raw_corruption_seeds) != 1:
        raise ValueError("pilot corruption configuration must contain exactly one seed")
    corruption_seed = _require_exact_int(raw_corruption_seeds[0], "corruption.seeds[0]")
    declared = _mapping(resolved.get("seed"), "seed")
    expected = {
        "split": split_seed,
        "model": model_seed,
        "corruption": corruption_seed,
    }
    if set(declared) != set(expected):
        raise ValueError(
            "pilot configuration seed must contain exactly split, model, and corruption"
        )
    observed = {name: _require_exact_int(declared[name], f"seed.{name}") for name in expected}
    if observed != expected:
        raise ValueError(
            "pilot top-level seed provenance must exactly match the nested execution seeds: "
            f"expected {expected}, found {observed}"
        )
    return observed


def _final_reference_metadata_binding(final_frame: Any, *, final_fold: int) -> dict[str, Any]:
    """Bind class-free final-fold metadata without publishing sample identities."""

    group_counts = sorted(
        (str(group_id), int(count))
        for group_id, count in final_frame["group_id"].astype(str).value_counts().items()
    )
    binding_payload = {
        "official_fold": int(final_fold),
        "analysis_eligible_sample_count": len(final_frame),
        "group_sample_counts": group_counts,
    }
    return {
        "schema_version": 1,
        "scope": "analysis_eligible_final_reference_metadata",
        "bound_fields": ["official_fold", "group_id", "group_sample_count"],
        "sample_count": len(final_frame),
        "group_count": len(group_counts),
        "contains_sample_ids": False,
        "contains_class_labels": False,
        "sha256": canonical_sha256(binding_payload),
    }


def _optional_reason(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _class_free_eligibility_records(frame: Any) -> list[dict[str, Any]]:
    """Aggregate eligibility without ever requiring sample identities or class labels."""

    if missing := sorted(set(CLASS_FREE_ELIGIBILITY_COLUMNS).difference(frame.columns)):
        raise ValueError(f"manifest lacks class-free eligibility fields: {missing}")
    counts: dict[tuple[int, str, bool, bool, bool, str | None], int] = {}
    for values in frame.loc[:, list(CLASS_FREE_ELIGIBILITY_COLUMNS)].itertuples(
        index=False, name=None
    ):
        fold, group, primary, confirmatory, touching, raw_reason = values
        key = (
            int(fold),
            str(group),
            bool(primary),
            bool(confirmatory),
            bool(touching),
            _optional_reason(raw_reason),
        )
        if key[0] <= 0 or not key[1]:
            raise ValueError("manifest class-free fold/group metadata is invalid")
        if key[2] != key[3] or (not key[2]) != key[4]:
            raise ValueError("manifest class-free eligibility masks are inconsistent")
        if (key[2] and key[5] is not None) or (not key[2] and key[5] != OVERLAP_EXCLUSION_REASON):
            raise ValueError("manifest class-free eligibility reason is inconsistent")
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("manifest class-free eligibility metadata is empty")
    return [
        {
            "official_fold": key[0],
            "group_id": key[1],
            "primary_eligible": key[2],
            "confirmatory_eligible": key[3],
            "cross_class_overlap_touching": key[4],
            "qc_exclusion_reason": key[5],
            "instance_count": count,
        }
        for key, count in sorted(
            counts.items(), key=lambda item: (*item[0][:-1], item[0][-1] or "")
        )
    ]


def _class_free_eligibility_provenance(
    frame: Any,
    *,
    canonical_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _class_free_eligibility_records(frame)
    primary = np.asarray(frame["primary_eligible"], dtype=np.bool_)
    support_by_fold: list[dict[str, int]] = []
    for fold in sorted(int(value) for value in frame["official_fold"].unique()):
        fold_mask = np.asarray(frame["official_fold"], dtype=np.int64) == fold
        eligible = int(np.sum(primary[fold_mask]))
        total = int(np.sum(fold_mask))
        support_by_fold.append(
            {
                "official_fold": fold,
                "instance_count": total,
                "eligible_instance_count": eligible,
                "excluded_instance_count": total - eligible,
                "group_count": int(frame.loc[fold_mask, "group_id"].astype(str).nunique()),
            }
        )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "scope": "complete_manifest_class_free_eligibility",
        "eligibility_policy": ELIGIBILITY_POLICY,
        "cross_class_overlap_exclusion_reason": OVERLAP_EXCLUSION_REASON,
        "manifest_sha256": canonical_manifest_sha256,
        "manifest_instance_count": len(frame),
        "manifest_eligible_instance_count": int(primary.sum()),
        "manifest_excluded_instance_count": int((~primary).sum()),
        "support_by_official_fold": support_by_fold,
        "bound_fields": [*CLASS_FREE_ELIGIBILITY_COLUMNS, "instance_count"],
        "manifest_class_free_eligibility_sha256": canonical_sha256(records),
        "contains_sample_ids": False,
        "contains_class_labels": False,
        "final_reference_sample_or_class_columns_read": False,
        "source_annotations_modified": False,
    }
    provenance["semantic_sha256"] = canonical_sha256(provenance)
    return provenance, records


def _with_development_manifest_view_metadata(
    development_table: Any,
    *,
    canonical_digest: str,
    global_provenance: Mapping[str, Any],
    development_folds: tuple[int, ...],
    final_fold: int,
) -> tuple[Any, dict[str, Any]]:
    development_all = development_table.to_pandas()
    identifiers = tuple(development_all["sample_id"].astype(str).tolist())
    primary = np.asarray(development_all["primary_eligible"], dtype=np.bool_)
    eligible_ids = tuple(
        sample_id for sample_id, keep in zip(identifiers, primary, strict=True) if keep
    )
    excluded_ids = tuple(
        sample_id for sample_id, keep in zip(identifiers, primary, strict=True) if not keep
    )
    development_records = _class_free_eligibility_records(
        development_all.loc[:, list(CLASS_FREE_ELIGIBILITY_COLUMNS)]
    )
    view_metadata: dict[str, Any] = {
        "schema_version": 1,
        "scope": DEVELOPMENT_MANIFEST_VIEW_SCOPE,
        "derivation_policy": "complete_canonical_development_rows_selected_by_official_fold",
        "included_official_folds": list(development_folds),
        "excluded_official_folds": [final_fold],
        "contains_final_reference_sample_ids": False,
        "contains_final_reference_class_labels": False,
        "canonical_manifest_sha256": canonical_digest,
        "canonical_manifest_class_free_eligibility_sha256": global_provenance[
            "manifest_class_free_eligibility_sha256"
        ],
        "development_class_free_eligibility_sha256": canonical_sha256(development_records),
        "manifest_instance_count": len(identifiers),
        "manifest_eligible_instance_count": len(eligible_ids),
        "manifest_excluded_instance_count": len(excluded_ids),
        "manifest_eligible_sample_ids_sha256": _sample_order_sha256(eligible_ids),
        "manifest_excluded_sample_ids_sha256": _sample_order_sha256(excluded_ids),
        "source_annotations_modified": False,
    }
    view_metadata["semantic_sha256"] = canonical_sha256(view_metadata)
    schema_metadata = dict(development_table.schema.metadata or {})
    schema_metadata[MANIFEST_VIEW_METADATA_KEY] = json.dumps(
        view_metadata,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return development_table.replace_schema_metadata(schema_metadata), view_metadata


def _publication_parent_key(path: Path) -> str:
    return os.path.normcase(str(Path(os.path.abspath(path))))


def _is_reparse_or_symlink(path: Path, value: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    attributes = int(getattr(value, "st_file_attributes", 0))
    return path.is_symlink() or bool(reparse_flag and attributes & reparse_flag)


def _require_publication_leaf(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("publication name must be one non-empty relative leaf")


def _windows_native_functions() -> tuple[Any, Any, Any]:
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    nt_create_file = ntdll.NtCreateFile
    nt_set_information_file = ntdll.NtSetInformationFile
    rtl_status_to_error = ntdll.RtlNtStatusToDosError
    nt_create_file.restype = wintypes.LONG
    nt_set_information_file.restype = wintypes.LONG
    rtl_status_to_error.argtypes = (wintypes.LONG,)
    rtl_status_to_error.restype = wintypes.ULONG
    return nt_create_file, nt_set_information_file, rtl_status_to_error


def _windows_raise_ntstatus(status: int, operation: str, converter: Any) -> None:
    if status >= 0:
        return
    import ctypes

    code = int(converter(status))
    message = f"{operation}: NTSTATUS=0x{ctypes.c_uint32(status).value:08X}"
    if code in {2, 3}:
        raise FileNotFoundError(code, message)
    if code in {80, 183}:
        raise FileExistsError(code, message)
    raise OSError(code, message)


def _windows_open_relative_descriptor(
    directory_handle: int,
    name: str,
    *,
    create: bool,
    write: bool = False,
    delete_access: bool = False,
) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _require_publication_leaf(name)

    class UnicodeString(ctypes.Structure):
        _fields_ = (
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        )

    class ObjectAttributes(ctypes.Structure):
        _fields_ = (
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(UnicodeString)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_quality_of_service", wintypes.LPVOID),
        )

    class IoStatusUnion(ctypes.Union):
        _fields_ = (("status", wintypes.LONG), ("pointer", wintypes.LPVOID))

    class IoStatusBlock(ctypes.Structure):
        _fields_ = (("value", IoStatusUnion), ("information", ctypes.c_size_t))

    encoded = name.encode("utf-16-le")
    name_buffer = ctypes.create_unicode_buffer(name)
    unicode_name = UnicodeString(
        len(encoded),
        len(encoded) + ctypes.sizeof(wintypes.WCHAR),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes),
        directory_handle,
        ctypes.pointer(unicode_name),
        0x40 | 0x1000,
        None,
        None,
    )
    io_status = IoStatusBlock()
    handle = wintypes.HANDLE()
    read_data = 0x0001
    write_data = 0x0002
    read_attributes = 0x0080
    write_attributes = 0x0100
    delete = 0x00010000
    synchronize = 0x00100000
    access = read_data | read_attributes | synchronize
    if write:
        access |= write_data | write_attributes
    if delete_access:
        access |= delete
    nt_create_file, _, converter = _windows_native_functions()
    nt_create_file.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.ULONG,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    )
    status = int(
        nt_create_file(
            ctypes.byref(handle),
            access,
            ctypes.byref(attributes),
            ctypes.byref(io_status),
            None,
            0x80,
            0x1 | 0x2 | 0x4,
            2 if create else 1,
            0x40 | 0x20 | 0x00200000,
            None,
            0,
        )
    )
    _windows_raise_ntstatus(status, f"NtCreateFile({name!r})", converter)
    assert handle.value is not None
    flags = (os.O_RDWR if write else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    try:
        return msvcrt.open_osfhandle(int(handle.value), flags)
    except BaseException:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        raise


def _windows_link_relative(
    source_descriptor: int,
    target_directory_handle: int,
    final_name: str,
) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _require_publication_leaf(final_name)

    class IoStatusUnion(ctypes.Union):
        _fields_ = (("status", wintypes.LONG), ("pointer", wintypes.LPVOID))

    class IoStatusBlock(ctypes.Structure):
        _fields_ = (("value", IoStatusUnion), ("information", ctypes.c_size_t))

    class FileLinkInformation(ctypes.Structure):
        _fields_ = (
            ("replace_if_exists", ctypes.c_ubyte),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.ULONG),
            ("file_name", wintypes.WCHAR * 1),
        )

    encoded = final_name.encode("utf-16-le")
    offset = FileLinkInformation.file_name.offset
    information_length = offset + len(encoded)
    allocation = ctypes.create_string_buffer(
        max(ctypes.sizeof(FileLinkInformation), information_length)
    )
    information = ctypes.cast(
        allocation,
        ctypes.POINTER(FileLinkInformation),
    ).contents
    information.replace_if_exists = 0
    information.root_directory = target_directory_handle
    information.file_name_length = len(encoded)
    ctypes.memmove(ctypes.addressof(allocation) + offset, encoded, len(encoded))
    io_status = IoStatusBlock()
    _, nt_set_information_file, converter = _windows_native_functions()
    nt_set_information_file.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
    )
    status = int(
        nt_set_information_file(
            msvcrt.get_osfhandle(source_descriptor),
            ctypes.byref(io_status),
            ctypes.cast(allocation, wintypes.LPVOID),
            information_length,
            11,
        )
    )
    _windows_raise_ntstatus(
        status,
        f"NtSetInformationFile(FileLinkInformation,{final_name!r})",
        converter,
    )


def _windows_delete_opened_link(descriptor: int) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class IoStatusUnion(ctypes.Union):
        _fields_ = (("status", wintypes.LONG), ("pointer", wintypes.LPVOID))

    class IoStatusBlock(ctypes.Structure):
        _fields_ = (("value", IoStatusUnion), ("information", ctypes.c_size_t))

    delete_file = ctypes.c_ubyte(1)
    io_status = IoStatusBlock()
    _, nt_set_information_file, converter = _windows_native_functions()
    nt_set_information_file.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
    )
    status = int(
        nt_set_information_file(
            msvcrt.get_osfhandle(descriptor),
            ctypes.byref(io_status),
            ctypes.byref(delete_file),
            ctypes.sizeof(delete_file),
            13,
        )
    )
    _windows_raise_ntstatus(
        status,
        "NtSetInformationFile(FileDispositionInformation)",
        converter,
    )


@dataclass(slots=True)
class _LockedPublicationParent:
    path: Path
    identity: tuple[int, int]
    descriptor: int | None
    native_handle: int | None

    def assert_current(self) -> None:
        try:
            value = self.path.stat(follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(f"publication parent changed while locked: {self.path}") from error
        if (
            not stat.S_ISDIR(value.st_mode)
            or _is_reparse_or_symlink(self.path, value)
            or (value.st_dev, value.st_ino) != self.identity
        ):
            raise RuntimeError(f"publication parent changed while locked: {self.path}")


@dataclass(slots=True)
class _LockedPublishedFile:
    parent: _LockedPublicationParent
    path: Path
    name: str
    identity: tuple[int, int, int]
    sha256: str

    def _open_descriptor(self, *, delete_access: bool = False) -> int:
        if self.parent.native_handle is not None:
            return _windows_open_relative_descriptor(
                self.parent.native_handle,
                self.name,
                create=False,
                delete_access=delete_access,
            )
        if self.parent.descriptor is None:
            raise RuntimeError(f"publication parent is not anchored: {self.parent.path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self.name, flags, dir_fd=self.parent.descriptor)

    def _stat(self) -> os.stat_result:
        if self.parent.native_handle is None:
            assert self.parent.descriptor is not None
            return os.stat(
                self.name,
                dir_fd=self.parent.descriptor,
                follow_symlinks=False,
            )
        descriptor = self._open_descriptor()
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def _sha256(self) -> str:
        descriptor = self._open_descriptor()
        digest = hashlib.sha256()
        try:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        return digest.hexdigest()

    def exists(self) -> bool:
        try:
            self._stat()
        except FileNotFoundError:
            return False
        return True

    def still_owned(self) -> bool:
        try:
            value = self._stat()
            return (
                stat.S_ISREG(value.st_mode)
                and (value.st_dev, value.st_ino, value.st_size) == self.identity
                and self._sha256() == self.sha256
            )
        except OSError:
            return False

    def unlink_owned(self) -> None:
        if self.parent.native_handle is not None:
            descriptor = self._open_descriptor(delete_access=True)
            digest = hashlib.sha256()
            try:
                value = os.fstat(descriptor)
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                if (
                    not stat.S_ISREG(value.st_mode)
                    or (value.st_dev, value.st_ino, value.st_size) != self.identity
                    or digest.hexdigest() != self.sha256
                ):
                    raise RuntimeError(f"refused to remove unowned publication: {self.path}")
                _windows_delete_opened_link(descriptor)
            finally:
                os.close(descriptor)
            return
        if not self.still_owned():
            raise RuntimeError(f"refused to remove unowned publication: {self.path}")
        assert self.parent.descriptor is not None
        os.unlink(self.name, dir_fd=self.parent.descriptor)

    def unlink_name_if_same_inode(self) -> None:
        if self.parent.native_handle is not None:
            descriptor = self._open_descriptor(delete_access=True)
            try:
                value = os.fstat(descriptor)
                if (value.st_dev, value.st_ino) != self.identity[:2]:
                    raise RuntimeError(f"refused to remove replaced staging name: {self.path}")
                _windows_delete_opened_link(descriptor)
            finally:
                os.close(descriptor)
            return
        assert self.parent.descriptor is not None
        value = os.stat(
            self.name,
            dir_fd=self.parent.descriptor,
            follow_symlinks=False,
        )
        if (value.st_dev, value.st_ino) != self.identity[:2]:
            raise RuntimeError(f"refused to remove replaced staging name: {self.path}")
        os.unlink(self.name, dir_fd=self.parent.descriptor)


def _rollback_locked_publications(publications: list[_LockedPublishedFile]) -> None:
    errors: list[str] = []
    for published in reversed(publications):
        if not published.exists():
            continue
        try:
            published.unlink_owned()
        except RuntimeError as error:
            errors.append(str(error))
        except OSError as error:
            errors.append(f"{published.path}: {error}")
    if errors:
        raise RuntimeError("publication rollback was incomplete: " + "; ".join(errors))


def _rollback_locked_staging_files(staged_files: list[_LockedPublishedFile]) -> None:
    errors: list[str] = []
    for staged in reversed(staged_files):
        if not staged.exists():
            continue
        try:
            staged.unlink_name_if_same_inode()
        except (RuntimeError, OSError) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("staging cleanup was incomplete: " + "; ".join(errors))


def _locked_path_exists(
    path: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> bool:
    parent = parents[_publication_parent_key(path.parent)]
    parent.assert_current()
    try:
        if parent.native_handle is not None:
            descriptor = _windows_open_relative_descriptor(
                parent.native_handle,
                path.name,
                create=False,
            )
            os.close(descriptor)
        else:
            assert parent.descriptor is not None
            os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_locked_path(
    path: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> tuple[bytes, tuple[int, int, int, int, int], tuple[int, int, int, int, int]]:
    parent = parents[_publication_parent_key(path.parent)]
    parent.assert_current()
    if parent.native_handle is not None:
        descriptor = _windows_open_relative_descriptor(
            parent.native_handle,
            path.name,
            create=False,
        )
    else:
        assert parent.descriptor is not None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent.descriptor)
    try:
        stat_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        observed = b"".join(chunks)
        stat_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(stat_after.st_mode):
        raise RuntimeError(f"publication path is not a regular file: {path}")
    identity_before = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
        stat_before.st_ctime_ns,
    )
    identity_after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
        stat_after.st_ctime_ns,
    )
    parent.assert_current()
    return observed, identity_before, identity_after


def _write_locked_staging_file(
    parent: _LockedPublicationParent,
    *,
    target_name: str,
    payload: bytes,
) -> _LockedPublishedFile:
    if parent.descriptor is None and parent.native_handle is None:
        raise RuntimeError("handle-relative staging is unavailable for this publication parent")
    descriptor: int | None = None
    name = ""
    for _ in range(32):
        name = f".{target_name}.stage-{secrets.token_hex(16)}.tmp"
        try:
            if parent.native_handle is not None:
                descriptor = _windows_open_relative_descriptor(
                    parent.native_handle,
                    name,
                    create=True,
                    write=True,
                    delete_access=True,
                )
            else:
                assert parent.descriptor is not None
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(name, flags, 0o600, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        break
    if descriptor is None:
        raise FileExistsError(f"could not allocate private anchored staging file for {target_name}")
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"failed to write anchored staging file for {target_name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
        value = os.fstat(descriptor)
    except BaseException:
        if parent.native_handle is not None:
            with suppress(OSError):
                _windows_delete_opened_link(descriptor)
        os.close(descriptor)
        if parent.native_handle is None:
            assert parent.descriptor is not None
            with suppress(OSError):
                os.unlink(name, dir_fd=parent.descriptor)
        raise
    os.close(descriptor)
    result = _LockedPublishedFile(
        parent=parent,
        path=parent.path / name,
        name=name,
        identity=(value.st_dev, value.st_ino, value.st_size),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    if not result.still_owned():
        raise OSError(f"anchored staging file failed identity/hash readback: {target_name}")
    return result


@contextmanager
def _stage_development_manifest_bundle(
    parents: Mapping[str, _LockedPublicationParent],
    *,
    destination: Path,
    metadata_path: Path,
    parquet_bytes: bytes,
    metadata_bytes: bytes,
) -> Iterator[tuple[_LockedPublishedFile, _LockedPublishedFile]]:
    staged: list[_LockedPublishedFile] = []
    try:
        parquet_parent = parents[_publication_parent_key(destination.parent)]
        metadata_parent = parents[_publication_parent_key(metadata_path.parent)]
        staged.append(
            _write_locked_staging_file(
                parquet_parent,
                target_name=destination.name,
                payload=parquet_bytes,
            )
        )
        staged.append(
            _write_locked_staging_file(
                metadata_parent,
                target_name=metadata_path.name,
                payload=metadata_bytes,
            )
        )
        yield staged[0], staged[1]
    finally:
        _rollback_locked_staging_files(staged)


def _link_locked_staging_file(
    staged: _LockedPublishedFile,
    target_parent: _LockedPublicationParent,
    target_name: str,
) -> None:
    source_descriptor = staged._open_descriptor()
    try:
        if target_parent.native_handle is not None:
            _windows_link_relative(
                source_descriptor,
                target_parent.native_handle,
                target_name,
            )
        else:
            assert staged.parent.descriptor is not None
            assert target_parent.descriptor is not None
            os.link(
                staged.name,
                target_name,
                src_dir_fd=staged.parent.descriptor,
                dst_dir_fd=target_parent.descriptor,
                follow_symlinks=False,
            )
    finally:
        os.close(source_descriptor)


def _publish_file_to_locked_parent(
    staged: _LockedPublishedFile,
    target: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> _LockedPublishedFile:
    parent = parents[_publication_parent_key(target.parent)]
    parent.assert_current()
    if not staged.still_owned():
        raise RuntimeError(f"anchored staging file changed before publication: {staged.path}")
    source_value = staged._stat()
    _link_locked_staging_file(staged, parent, target.name)
    target_value = _LockedPublishedFile(
        parent=parent,
        path=target,
        name=target.name,
        identity=(0, 0, 0),
        sha256=staged.sha256,
    )._stat()
    result = _LockedPublishedFile(
        parent=parent,
        path=target,
        name=target.name,
        identity=(target_value.st_dev, target_value.st_ino, target_value.st_size),
        sha256=staged.sha256,
    )
    if result.identity != (
        source_value.st_dev,
        source_value.st_ino,
        source_value.st_size,
    ):
        result.unlink_owned()
        raise OSError(f"published file failed hard-link identity readback: {target}")

    try:
        parent.assert_current()
    except BaseException:
        result.unlink_owned()
        raise
    if not result.still_owned():
        raise OSError(f"published file failed identity/hash readback: {target}")
    return result


@contextmanager
def _locked_publication_parents(
    paths: Sequence[Path],
    *,
    final_paths: Sequence[Path],
    raw_root: Path,
) -> Iterator[dict[str, _LockedPublicationParent]]:
    """Anchor parents so a path swap cannot redirect a publication into raw data."""

    resolved_raw = raw_root.resolve()
    for path in final_paths:
        current = (path.parent.resolve() / path.name).resolve()
        if current == resolved_raw or resolved_raw in current.parents:
            raise ValueError("development-manifest bundle cannot be published under the raw root")

    unique_paths = {_publication_parent_key(path): Path(os.path.abspath(path)) for path in paths}
    parents: dict[str, _LockedPublicationParent] = {}
    descriptors: list[int] = []
    windows_chain_handles: list[int] = []
    windows_native_handles: list[int] = []
    close_windows_handle: Any | None = None
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            close_windows_handle = kernel32.CloseHandle
            close_windows_handle.argtypes = (wintypes.HANDLE,)
            close_windows_handle.restype = wintypes.BOOL
            invalid_handle = ctypes.c_void_p(-1).value
            read_attributes = 0x0080
            share_read_write_without_delete = 0x00000001 | 0x00000002
            open_existing = 3
            backup_semantics = 0x02000000
            open_reparse_point = 0x00200000

            for key, parent_path in unique_paths.items():
                parts = parent_path.parts
                candidate = Path(parts[0])
                final_handle: int | None = None
                for index, part in enumerate(parts):
                    if index:
                        candidate /= part
                    if not os.path.lexists(candidate):
                        with suppress(FileExistsError):
                            candidate.mkdir()
                    handle = create_file(
                        str(candidate),
                        read_attributes,
                        share_read_write_without_delete,
                        None,
                        open_existing,
                        backup_semantics | open_reparse_point,
                        None,
                    )
                    if handle == invalid_handle:
                        raise ctypes.WinError(ctypes.get_last_error())
                    numeric_handle = int(handle)
                    windows_chain_handles.append(numeric_handle)
                    value = candidate.stat(follow_symlinks=False)
                    if not stat.S_ISDIR(value.st_mode) or _is_reparse_or_symlink(candidate, value):
                        raise RuntimeError(
                            f"publication parent path contains a reparse point: {candidate}"
                        )
                    final_handle = numeric_handle
                if final_handle is None:
                    raise RuntimeError(f"publication parent is invalid: {parent_path}")
                native_handle = create_file(
                    str(parent_path),
                    read_attributes,
                    0x00000001 | 0x00000002 | 0x00000004,
                    None,
                    open_existing,
                    backup_semantics | open_reparse_point,
                    None,
                )
                if native_handle == invalid_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                numeric_native_handle = int(native_handle)
                windows_native_handles.append(numeric_native_handle)
                value = parent_path.stat(follow_symlinks=False)
                parents[key] = _LockedPublicationParent(
                    path=parent_path,
                    identity=(value.st_dev, value.st_ino),
                    descriptor=None,
                    native_handle=numeric_native_handle,
                )
            assert close_windows_handle is not None
            for handle in reversed(windows_chain_handles):
                close_windows_handle(handle)
            windows_chain_handles.clear()
        else:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            for key, parent_path in unique_paths.items():
                parts = parent_path.parts
                descriptor = os.open(parts[0], directory_flags)
                try:
                    for part in parts[1:]:
                        with suppress(FileExistsError):
                            os.mkdir(part, dir_fd=descriptor)
                        next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
                        os.close(descriptor)
                        descriptor = next_descriptor
                    value = os.fstat(descriptor)
                    if not stat.S_ISDIR(value.st_mode):
                        raise RuntimeError(f"publication parent is not a directory: {parent_path}")
                    descriptors.append(descriptor)
                    parents[key] = _LockedPublicationParent(
                        path=parent_path,
                        identity=(value.st_dev, value.st_ino),
                        descriptor=descriptor,
                        native_handle=None,
                    )
                except BaseException:
                    os.close(descriptor)
                    raise

        for parent in parents.values():
            parent.assert_current()
        yield parents
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        if close_windows_handle is not None:
            for handle in reversed(windows_chain_handles):
                close_windows_handle(handle)
            for handle in reversed(windows_native_handles):
                close_windows_handle(handle)


def _publish_development_manifest_bundle(
    *,
    destination: Path,
    metadata_path: Path,
    parquet_bytes: bytes,
    sidecar: Mapping[str, Any],
    raw_root: Path,
    raw_inventory: object,
    source_hashes: Mapping[Path, str],
) -> None:
    """Publish the Parquet/certificate pair under one ownership-safe transaction."""

    expected_sidecar_bytes = (
        json.dumps(
            sidecar,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    def verify_destinations_outside_raw() -> None:
        resolved_raw = raw_root.resolve()
        for role, path in (
            ("development manifest view", destination),
            ("pre-pilot gate certificate", metadata_path),
        ):
            current = (path.parent.resolve() / path.name).resolve()
            if current == resolved_raw or resolved_raw in current.parents:
                raise ValueError(f"{role} cannot be published under the raw root")

    def verify_publication_parents() -> None:
        for parent in publication_parents.values():
            parent.assert_current()

    def verify_outputs() -> None:
        verify_publication_parents()
        for role, path, expected in (
            ("development manifest view", destination, parquet_bytes),
            ("pre-pilot gate certificate", metadata_path, expected_sidecar_bytes),
        ):
            observed, identity_before, identity_after = _read_locked_path(
                path,
                publication_parents,
            )
            expected_digest = hashlib.sha256(expected).hexdigest()
            if (
                identity_before != identity_after
                or observed != expected
                or hashlib.sha256(observed).hexdigest() != expected_digest
            ):
                raise RuntimeError(f"published {role} failed stable byte/hash readback")
        verify_publication_parents()

    def verify_inputs() -> None:
        verify_publication_parents()
        verify_destinations_outside_raw()
        for path, expected_digest in source_hashes.items():
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise RuntimeError(
                    f"pre-pilot source evidence changed during view materialization: {path}"
                )
        _verified_inventory(raw_root, raw_inventory)
        verify_publication_parents()

    with (
        ExclusiveBundlePublicationLock(
            (destination, metadata_path),
            role="pre-pilot development-manifest bundle",
        ) as publication_lock,
        _locked_publication_parents(
            (destination.parent, metadata_path.parent),
            final_paths=(destination, metadata_path),
            raw_root=raw_root,
        ) as publication_parents,
    ):
        publication_lock.assert_owned()
        verify_destinations_outside_raw()
        parquet_exists = _locked_path_exists(destination, publication_parents)
        metadata_exists = _locked_path_exists(metadata_path, publication_parents)
        if parquet_exists != metadata_exists:
            raise FileExistsError(
                "pre-pilot development-manifest bundle is partial; refusing publication"
            )
        verify_inputs()
        if parquet_exists:
            if (
                _read_locked_path(destination, publication_parents)[0] != parquet_bytes
                or _read_locked_path(metadata_path, publication_parents)[0]
                != expected_sidecar_bytes
            ):
                raise FileExistsError(
                    "refusing to overwrite a different pre-pilot development-manifest bundle"
                )
            verify_outputs()
            verify_inputs()
            verify_outputs()
            publication_lock.assert_owned()
            return

        publications: list[_LockedPublishedFile] = []
        with _stage_development_manifest_bundle(
            publication_parents,
            destination=destination,
            metadata_path=metadata_path,
            parquet_bytes=parquet_bytes,
            metadata_bytes=expected_sidecar_bytes,
        ) as (staged_parquet, staged_metadata):
            verify_inputs()
            try:
                publications.append(
                    _publish_file_to_locked_parent(
                        staged_parquet,
                        destination,
                        publication_parents,
                    )
                )
                publications.append(
                    _publish_file_to_locked_parent(
                        staged_metadata,
                        metadata_path,
                        publication_parents,
                    )
                )
                publication_lock.assert_owned()
                verify_outputs()
                verify_inputs()
                verify_outputs()
                publication_lock.assert_owned()
            except BaseException as publication_error:
                try:
                    _rollback_locked_publications(publications)
                except RuntimeError:
                    raise RuntimeError(
                        "pre-pilot development-manifest publication failed and "
                        "ownership-safe rollback was incomplete"
                    ) from publication_error
                raise


def _validate_complete_real_dataset_evidence(
    dataset_path: Path,
    validation_path: Path,
    duplicate_path: Path,
) -> tuple[str, ...]:
    """Import the external gate lazily to avoid the experiment-package import cycle."""

    from histo_audit.external_validation import validate_real_dataset_evidence

    return validate_real_dataset_evidence(dataset_path, validation_path, duplicate_path)


def build_pannuke_pilot_development_manifest_view(
    validation_source: ValidationArtifacts | str | Path,
    manifest_source: ManifestArtifacts | str | Path,
    duplicate_audit_source: str | Path,
    output_path: str | Path,
    *,
    development_folds: tuple[int, ...] = (1, 2),
    final_fold: int = 3,
) -> PanNukePilotDevelopmentManifestView:
    """Materialize the explicit pre-pilot privacy boundary before a pilot run starts."""

    manifest_path = _manifest_path(manifest_source)
    validation_path = _validation_evidence_path(validation_source)
    duplicate_path = Path(duplicate_audit_source).expanduser().resolve()
    if not duplicate_path.is_file():
        raise FileNotFoundError(f"duplicate-audit evidence is missing: {duplicate_path}")
    destination = Path(output_path).expanduser().resolve()
    if destination == manifest_path:
        raise ValueError("development manifest view cannot overwrite the canonical manifest")
    metadata_path = destination.with_suffix(f"{destination.suffix}.metadata.json")
    canonical_digest = sha256_file(manifest_path)
    validation_digest = sha256_file(validation_path)
    duplicate_digest = sha256_file(duplicate_path)
    validation_payload = _read_json_mapping(validation_path, "PanNuke validation evidence")
    duplicate_payload = _read_json_mapping(duplicate_path, "PanNuke duplicate-audit evidence")
    development_validation_view = _privacy_safe_development_validation_view(
        validation_payload,
        development_folds=development_folds,
        final_fold=final_fold,
    )
    raw_root = Path(str(development_validation_view["root"])).expanduser().resolve()
    if destination == raw_root or raw_root in destination.parents:
        raise ValueError("development manifest view cannot be published under the raw root")
    evidence_errors = _validate_complete_real_dataset_evidence(
        raw_root,
        validation_path,
        duplicate_path,
    )
    if evidence_errors:
        raise ValueError(
            "pre-pilot source evidence failed the complete M5 gate: " + "; ".join(evidence_errors)
        )
    verified_inventory = _verified_inventory(
        raw_root, development_validation_view["raw_file_inventory"]
    )
    if (
        canonical_sha256([record.as_dict() for record in verified_inventory])
        != (development_validation_view["raw_file_inventory_sha256"])
    ):
        raise ValueError("pre-pilot raw inventory binding is invalid")
    if duplicate_payload.get("required_two_signal_near_duplicate_gate_complete") is not True:
        raise ValueError("duplicate-audit evidence does not pass the required two-signal gate")
    canonical_table = pq.read_table(manifest_path)
    validate_manifest_invariants(canonical_table)
    manifest_inventory_payload = (canonical_table.schema.metadata or {}).get(b"raw_file_inventory")
    if manifest_inventory_payload is None:
        raise ValueError("canonical manifest lacks raw-file inventory metadata")
    try:
        manifest_inventory = json.loads(manifest_inventory_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical manifest raw-file inventory metadata is invalid") from error
    if manifest_inventory != validation_payload.get("raw_file_inventory"):
        raise ValueError("canonical manifest raw-file inventory differs from validation evidence")
    metadata_frame = canonical_table.select(list(CLASS_FREE_ELIGIBILITY_COLUMNS)).to_pandas()
    global_provenance, _ = _class_free_eligibility_provenance(
        metadata_frame,
        canonical_manifest_sha256=canonical_digest,
    )
    observed_folds = set(int(value) for value in metadata_frame["official_fold"].unique())
    if observed_folds != {*development_folds, final_fold}:
        raise ValueError(
            "manifest class-free metadata must contain exactly the declared official folds"
        )
    fold_values = np.asarray(canonical_table["official_fold"], dtype=np.int64)
    development_indices = np.flatnonzero(np.isin(fold_values, development_folds))
    development_table = canonical_table.take(pa.array(development_indices, type=pa.int64()))
    validate_manifest_invariants(development_table)
    development_table, view_metadata = _with_development_manifest_view_metadata(
        development_table,
        canonical_digest=canonical_digest,
        global_provenance=global_provenance,
        development_folds=development_folds,
        final_fold=final_fold,
    )
    selected = select_manifest_rows(development_table, sample_ids=None, scope="analysis")
    if set(int(value) for value in selected.frame["official_fold"].unique()) != set(
        development_folds
    ):
        raise RuntimeError("pre-pilot development manifest view has an invalid fold scope")
    stream = io.BytesIO()
    pq.write_table(development_table, stream, compression="zstd")
    encoded = stream.getvalue()
    development_digest = hashlib.sha256(encoded).hexdigest()
    sidecar: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "policy": "pre_pilot_privacy_gate_v1",
        "source_evidence": {
            "validation_json": {
                "path": str(validation_path),
                "sha256": validation_digest,
            },
            "duplicate_audit_json": {
                "path": str(duplicate_path),
                "sha256": duplicate_digest,
            },
            "canonical_manifest": {
                "path": str(manifest_path),
                "sha256": canonical_digest,
            },
            "development_manifest_view": {
                "path": str(destination),
                "sha256": development_digest,
            },
        },
        "development_manifest_view_metadata": view_metadata,
        "development_validation_view": development_validation_view,
        "global_class_free_eligibility": global_provenance,
        "materialization_boundary": "pre_pilot_before_run_creation",
        "privacy_contract": {
            "final_reference_sample_ids_published": False,
            "final_reference_class_labels_published": False,
            "final_reference_patch_or_instance_qc_published": False,
            "pilot_parses_full_validation_json": False,
            "pilot_semantically_decodes_canonical_sample_or_class_columns": False,
            "opaque_source_sha256_verified": True,
            "contains_final_reference_file_paths_and_integrity_hashes": True,
            "final_reference_raw_access": "byte_level_integrity_only",
            "pilot_final_reference_scope": "class_free_integrity_metadata_only",
        },
        "source_annotations_modified": False,
    }
    sidecar["semantic_sha256"] = canonical_sha256(sidecar)
    _publish_development_manifest_bundle(
        destination=destination,
        metadata_path=metadata_path,
        parquet_bytes=encoded,
        sidecar=sidecar,
        raw_root=raw_root,
        raw_inventory=development_validation_view["raw_file_inventory"],
        source_hashes={
            manifest_path: canonical_digest,
            validation_path: validation_digest,
            duplicate_path: duplicate_digest,
        },
    )
    return PanNukePilotDevelopmentManifestView(
        parquet_path=destination,
        metadata_path=metadata_path,
        canonical_manifest_sha256=canonical_digest,
        development_manifest_sha256=development_digest,
        development_instance_count=len(development_table),
    )


def _load_privacy_partitioned_manifest(
    manifest_path: Path,
    development_manifest_path: Path,
    *,
    development_folds: tuple[int, ...],
    final_fold: int,
) -> _PilotManifestInputs:
    """Load only class-free final metadata plus a prebuilt development-only view."""

    canonical_digest = sha256_file(manifest_path)
    metadata_table = pq.read_table(
        manifest_path,
        columns=list(CLASS_FREE_ELIGIBILITY_COLUMNS),
    )
    metadata_frame = metadata_table.to_pandas()
    global_provenance, global_records = _class_free_eligibility_provenance(
        metadata_frame,
        canonical_manifest_sha256=canonical_digest,
    )
    observed_folds = set(int(value) for value in metadata_frame["official_fold"].unique())
    if observed_folds != {*development_folds, final_fold}:
        raise ValueError(
            "manifest class-free metadata must contain exactly the declared official folds"
        )
    development_digest = sha256_file(development_manifest_path)
    development_table = pq.read_table(development_manifest_path)
    if sha256_file(development_manifest_path) != development_digest:
        raise RuntimeError("pre-pilot development manifest view changed during parsing")
    validate_manifest_invariants(development_table)
    development_selection = select_manifest_rows(
        development_table,
        sample_ids=None,
        scope="analysis",
    )
    view_metadata = development_selection.provenance.get("manifest_view")
    if not isinstance(view_metadata, Mapping):
        raise ValueError("development manifest view lacks the privacy-boundary metadata")
    view_metadata = dict(view_metadata)
    if (
        view_metadata.get("canonical_manifest_sha256") != canonical_digest
        or view_metadata.get("canonical_manifest_class_free_eligibility_sha256")
        != global_provenance["manifest_class_free_eligibility_sha256"]
    ):
        raise ValueError("development manifest view is not bound to the canonical manifest")
    development_all = development_table.to_pandas()
    expected_development_metadata = metadata_frame[
        metadata_frame["official_fold"].isin(development_folds)
    ].reset_index(drop=True)
    development_records = _class_free_eligibility_records(
        development_all.loc[:, list(CLASS_FREE_ELIGIBILITY_COLUMNS)]
    )
    expected_records = _class_free_eligibility_records(expected_development_metadata)
    if (
        development_records != expected_records
        or len(development_all) != len(expected_development_metadata)
        or view_metadata.get("development_class_free_eligibility_sha256")
        != canonical_sha256(development_records)
    ):
        raise RuntimeError(
            "development manifest view does not contain every canonical development row"
        )
    final_metadata = metadata_frame[
        (metadata_frame["official_fold"] == final_fold)
        & metadata_frame["primary_eligible"].astype(bool)
    ].reset_index(drop=True)
    if final_metadata.empty:
        raise ValueError("analysis-eligible final-reference metadata is empty")
    if (
        canonical_sha256(global_records)
        != global_provenance["manifest_class_free_eligibility_sha256"]
    ):
        raise RuntimeError("global class-free eligibility binding is inconsistent")
    return _PilotManifestInputs(
        development_table=development_table,
        development_frame=development_selection.frame,
        development_provenance=development_selection.provenance,
        class_free_frame=metadata_frame,
        final_metadata_frame=final_metadata,
        global_provenance=global_provenance,
        canonical_manifest_sha256=canonical_digest,
        development_manifest_sha256=development_digest,
        view_metadata=view_metadata,
    )


def _publish_development_manifest_view(
    tracker: RunTracker,
    inputs: _PilotManifestInputs,
    source_path: Path,
) -> Path:
    destination = tracker.run_directory / "development_manifest_view.parquet"
    if sha256_file(source_path) != inputs.development_manifest_sha256:
        raise RuntimeError("pre-pilot development manifest view changed before publication")
    atomic_write_bytes(destination, source_path.read_bytes())
    readback = pq.read_table(destination)
    validate_manifest_invariants(readback)
    selected = select_manifest_rows(readback, sample_ids=None, scope="analysis")
    if selected.provenance != inputs.development_provenance:
        raise RuntimeError("published development manifest view failed provenance readback")
    view_digest = sha256_file(destination)
    if view_digest != inputs.development_manifest_sha256:
        raise RuntimeError("run-local development manifest view differs from its source")
    tracker.write_json(
        "development_manifest_view.json",
        {
            "schema_version": 1,
            "canonical_manifest_sha256": inputs.canonical_manifest_sha256,
            "development_manifest_view_path": destination.name,
            "development_manifest_view_sha256": view_digest,
            "development_manifest_view_metadata": inputs.view_metadata,
            "global_class_free_eligibility": inputs.global_provenance,
            "final_reference_sample_ids_or_class_labels_in_view": False,
            "source_annotations_modified": False,
        },
    )
    return destination


def _manifest_path(source: ManifestArtifacts | str | Path) -> Path:
    path = source.parquet_path if isinstance(source, ManifestArtifacts) else Path(source)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"validated PanNuke nucleus manifest is missing: {resolved}")
    return resolved


def _validation_evidence_path(source: ValidationArtifacts | str | Path) -> Path:
    path = source.json_path if isinstance(source, ValidationArtifacts) else Path(source)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PanNuke validation evidence is missing: {resolved}")
    return resolved


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return dict(value)


def _privacy_safe_development_validation_view(
    payload: Mapping[str, Any],
    *,
    development_folds: tuple[int, ...],
    final_fold: int,
) -> dict[str, Any]:
    if (
        payload.get("status") != "valid"
        or payload.get("validation_scope") != "full_semantic_scan"
        or payload.get("release_complete") is not True
        or tuple(payload.get("expected_fold_ids", ())) != (*development_folds, final_fold)
    ):
        raise ValueError("saved PanNuke validation evidence is not a complete release gate")
    root = payload.get("root")
    mapping = payload.get("class_mapping")
    folds = payload.get("folds")
    fold_validation = payload.get("fold_validation")
    inventory = payload.get("raw_file_inventory")
    qc_policy = payload.get("qc_policy")
    if (
        not isinstance(root, str)
        or not root
        or not isinstance(mapping, Mapping)
        or not isinstance(folds, list)
        or not isinstance(fold_validation, list)
        or not isinstance(inventory, list)
        or not isinstance(qc_policy, Mapping)
    ):
        raise ValueError("saved PanNuke validation evidence has an invalid structure")
    expected_mapping = OFFICIAL_METRICS_CLASS_MAPPING.as_dict()
    expected_mapping["class_names"] = list(expected_mapping["class_names"])
    if dict(mapping) != expected_mapping:
        raise ValueError("saved PanNuke class mapping differs from pinned evidence")
    expected_fold_fields = {
        "fold_id",
        "image_path",
        "mask_path",
        "tissue_path",
        "image_channel_axis",
        "mask_channel_axis",
    }
    if any(
        not isinstance(item, Mapping)
        or set(item) != expected_fold_fields
        or type(item.get("fold_id")) is not int
        or type(item.get("image_channel_axis")) is not int
        or type(item.get("mask_channel_axis")) is not int
        or any(
            not isinstance(item.get(field), str) or not str(item[field])
            for field in ("image_path", "mask_path", "tissue_path")
        )
        for item in folds
    ):
        raise ValueError("saved PanNuke fold descriptors are invalid")
    all_fold_ids = {int(item["fold_id"]) for item in folds if isinstance(item, Mapping)}
    all_fact_ids = {int(item["fold_id"]) for item in fold_validation if isinstance(item, Mapping)}
    expected = {*development_folds, final_fold}
    if all_fold_ids != expected or all_fact_ids != expected:
        raise ValueError("saved PanNuke validation evidence lacks an official fold")
    development_fold_records = [
        dict(item)
        for item in folds
        if isinstance(item, Mapping) and int(item["fold_id"]) in development_folds
    ]
    development_fact_records: list[dict[str, Any]] = []
    for item in fold_validation:
        if not isinstance(item, Mapping) or int(item["fold_id"]) not in development_folds:
            continue
        if item.get("validation_scope") != "full_semantic_scan":
            raise ValueError("development fold lacks full-semantic-scan evidence")
        development_fact_records.append(
            {
                "fold_id": int(item["fold_id"]),
                "n_patches": int(item["n_patches"]),
                "positive_channel_indices": [
                    int(value) for value in item["positive_channel_indices"]
                ],
                "validation_scope": str(item["validation_scope"]),
                "disconnected_instance_count_full_scan": int(
                    item["disconnected_instance_count_full_scan"]
                ),
                "disconnected_patch_count_full_scan": int(
                    item["disconnected_patch_count_full_scan"]
                ),
            }
        )
    if {item["fold_id"] for item in development_fold_records} != set(development_folds) or {
        item["fold_id"] for item in development_fact_records
    } != set(development_folds):
        raise ValueError("privacy-safe validation view lacks a development fold")
    inventory_records = [dict(item) for item in inventory if isinstance(item, Mapping)]
    if len(inventory_records) != len(inventory):
        raise ValueError("saved PanNuke raw inventory is invalid")
    inventory_by_path = {
        str(item.get("relative_path", "")).replace("\\", "/").casefold(): item
        for item in inventory_records
    }
    bound_paths: set[str] = set()
    for item in development_fold_records:
        fold_id = int(item["fold_id"])
        for field in ("image_path", "mask_path", "tissue_path"):
            relative = Path(str(item[field]))
            canonical = relative.as_posix()
            record = inventory_by_path.get(canonical.casefold())
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or record is None
                or record.get("relative_path") != canonical
                or record.get("fold_id") != fold_id
                or record.get("file_kind") != "npy"
                or canonical.casefold() in bound_paths
            ):
                raise ValueError(
                    "saved PanNuke development descriptor is not bound to its raw-inventory fold"
                )
            bound_paths.add(canonical.casefold())
    view: dict[str, Any] = {
        "schema_version": 1,
        "root": root,
        "class_mapping": dict(mapping),
        "folds": development_fold_records,
        "raw_file_inventory": inventory_records,
        "raw_file_inventory_sha256": canonical_sha256(inventory_records),
        "qc_policy": {
            "source_masks_modified": qc_policy.get("source_masks_modified"),
            "disconnected_instance_ids_are_fatal": qc_policy.get(
                "disconnected_instance_ids_are_fatal"
            ),
            "complete_policy_sha256": canonical_sha256(qc_policy),
        },
        "expected_official_fold_ids": [*development_folds, final_fold],
        "development_official_fold_ids": list(development_folds),
        "final_reference_official_fold_id": final_fold,
        "release_complete": True,
        "contains_final_reference_semantic_fold_descriptors": False,
        "contains_final_reference_file_paths_and_integrity_hashes": True,
        "final_reference_raw_access": "byte_level_integrity_only",
        "contains_final_reference_patch_or_instance_qc": False,
        "contains_final_reference_sample_ids": False,
        "contains_final_reference_class_labels": False,
    }
    view["semantic_sha256"] = canonical_sha256(view)
    return view


def _verified_inventory(
    root: Path,
    raw_records: object,
) -> tuple[RawFileRecord, ...]:
    resolved_root = root.resolve()
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("pilot gate certificate raw inventory is invalid")
    records: list[RawFileRecord] = []
    seen_relative_paths: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "size_bytes",
            "sha256",
            "fold_id",
            "file_kind",
        }:
            raise ValueError("pilot gate certificate raw inventory row is invalid")
        raw_relative = raw.get("relative_path")
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError("pilot gate certificate raw inventory path is invalid")
        relative = Path(raw_relative)
        canonical_relative = relative.as_posix()
        folded_relative = canonical_relative.casefold()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or canonical_relative != raw_relative.replace("\\", "/")
            or folded_relative in seen_relative_paths
        ):
            raise ValueError("pilot gate certificate raw inventory path is unsafe")
        seen_relative_paths.add(folded_relative)
        path = (resolved_root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("pilot gate certificate raw inventory path escapes root") from error
        expected_size = raw.get("size_bytes")
        expected_sha = raw.get("sha256")
        fold_id = raw.get("fold_id")
        file_kind = raw.get("file_kind")
        if (
            type(expected_size) is not int
            or expected_size < 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or expected_sha != expected_sha.lower()
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or (fold_id is not None and type(fold_id) is not int)
            or not isinstance(file_kind, str)
            or not file_kind
        ):
            raise ValueError("pilot gate certificate raw inventory row has invalid values")
        records.append(
            RawFileRecord(
                relative_path=canonical_relative,
                size_bytes=expected_size,
                sha256=expected_sha,
                fold_id=fold_id,
                file_kind=file_kind,
            )
        )

    observed_records = inventory_raw_files(
        resolved_root,
        include_temporary_files=True,
    )
    observed_by_path = {record.relative_path.casefold(): record for record in observed_records}
    if set(observed_by_path) != seen_relative_paths:
        expected_by_path = {
            record.relative_path.casefold(): record.relative_path for record in records
        }
        added = sorted(
            observed_by_path[key].relative_path
            for key in set(observed_by_path).difference(seen_relative_paths)
        )
        removed = sorted(
            expected_by_path[key] for key in seen_relative_paths.difference(observed_by_path)
        )
        raise RuntimeError(f"raw inventory path set changed (added={added!r}, removed={removed!r})")

    for expected in records:
        observed = observed_by_path[expected.relative_path.casefold()]
        if (
            observed.relative_path != expected.relative_path
            or observed.size_bytes != expected.size_bytes
            or observed.sha256 != expected.sha256
            or observed.file_kind != expected.file_kind
        ):
            raise RuntimeError(f"raw inventory binding failed for {expected.relative_path}")
    return tuple(records)


def _load_pilot_gate_certificate(
    certificate_path: Path,
    *,
    manifest_path: Path,
    development_manifest_path: Path,
    expected_duplicate_sha256: str,
    expected_data_root: Path,
) -> tuple[PanNukeValidationResult, dict[str, Any], str]:
    certificate_digest = sha256_file(certificate_path)
    certificate = _read_json_mapping(certificate_path, "pre-pilot gate certificate")
    semantic = certificate.pop("semantic_sha256", None)
    if semantic != canonical_sha256(certificate):
        raise ValueError("pre-pilot gate certificate semantic hash is invalid")
    certificate["semantic_sha256"] = semantic
    privacy = certificate.get("privacy_contract")
    source = certificate.get("source_evidence")
    validation_view = certificate.get("development_validation_view")
    if (
        certificate.get("schema_version") != 1
        or certificate.get("status") != "complete"
        or certificate.get("policy") != "pre_pilot_privacy_gate_v1"
        or not isinstance(privacy, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(validation_view, Mapping)
    ):
        raise ValueError("pre-pilot gate certificate structure is invalid")
    expected_certificate_fields = {
        "schema_version",
        "status",
        "policy",
        "source_evidence",
        "development_manifest_view_metadata",
        "development_validation_view",
        "global_class_free_eligibility",
        "materialization_boundary",
        "privacy_contract",
        "source_annotations_modified",
        "semantic_sha256",
    }
    if set(certificate) != expected_certificate_fields or set(source) != {
        "validation_json",
        "duplicate_audit_json",
        "canonical_manifest",
        "development_manifest_view",
    }:
        raise ValueError("pre-pilot gate certificate contains unexpected fields")
    if (
        certificate.get("materialization_boundary") != "pre_pilot_before_run_creation"
        or certificate.get("source_annotations_modified") is not False
    ):
        raise ValueError("pre-pilot gate certificate boundary policy is invalid")
    if re.search(
        r"pannuke-f3-p\d+-c\d+-i\d+",
        json.dumps(certificate, ensure_ascii=True, sort_keys=True),
    ):
        raise ValueError("pre-pilot gate certificate contains a final-reference sample ID")
    required_privacy = {
        "final_reference_sample_ids_published": False,
        "final_reference_class_labels_published": False,
        "final_reference_patch_or_instance_qc_published": False,
        "pilot_parses_full_validation_json": False,
        "pilot_semantically_decodes_canonical_sample_or_class_columns": False,
        "opaque_source_sha256_verified": True,
        "contains_final_reference_file_paths_and_integrity_hashes": True,
        "final_reference_raw_access": "byte_level_integrity_only",
        "pilot_final_reference_scope": "class_free_integrity_metadata_only",
    }
    if set(privacy) != set(required_privacy) or any(
        privacy.get(field) != value for field, value in required_privacy.items()
    ):
        raise ValueError("pre-pilot gate certificate privacy contract is invalid")

    def source_record(name: str) -> tuple[Path, str]:
        value = source.get(name)
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError(f"pre-pilot gate certificate lacks {name}")
        path = Path(str(value.get("path", ""))).expanduser().resolve()
        digest = value.get("sha256")
        if (
            not isinstance(value.get("path"), str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise RuntimeError(f"pre-pilot source binding failed for {name}")
        return path, digest

    _, validation_digest = source_record("validation_json")
    _, duplicate_digest = source_record("duplicate_audit_json")
    bound_manifest, manifest_digest = source_record("canonical_manifest")
    bound_development, development_digest = source_record("development_manifest_view")
    if (
        validation_digest == ""
        or duplicate_digest != expected_duplicate_sha256
        or bound_manifest != manifest_path
        or manifest_digest != sha256_file(manifest_path)
        or bound_development != development_manifest_path
        or development_digest != sha256_file(development_manifest_path)
    ):
        raise ValueError("pre-pilot gate certificate source bindings differ from execution")
    validation_semantic = dict(validation_view)
    view_semantic = validation_semantic.pop("semantic_sha256", None)
    if view_semantic != canonical_sha256(validation_semantic):
        raise ValueError("privacy-safe development validation view hash is invalid")
    expected_validation_view_fields = {
        "schema_version",
        "root",
        "class_mapping",
        "folds",
        "raw_file_inventory",
        "raw_file_inventory_sha256",
        "qc_policy",
        "expected_official_fold_ids",
        "development_official_fold_ids",
        "final_reference_official_fold_id",
        "release_complete",
        "contains_final_reference_semantic_fold_descriptors",
        "contains_final_reference_file_paths_and_integrity_hashes",
        "final_reference_raw_access",
        "contains_final_reference_patch_or_instance_qc",
        "contains_final_reference_sample_ids",
        "contains_final_reference_class_labels",
        "semantic_sha256",
    }
    if set(validation_view) != expected_validation_view_fields:
        raise ValueError("privacy-safe development validation view contains unexpected fields")
    if (
        validation_view.get("schema_version") != 1
        or validation_view.get("contains_final_reference_semantic_fold_descriptors") is not False
        or validation_view.get("contains_final_reference_file_paths_and_integrity_hashes")
        is not True
        or validation_view.get("final_reference_raw_access") != "byte_level_integrity_only"
        or validation_view.get("contains_final_reference_patch_or_instance_qc") is not False
        or validation_view.get("contains_final_reference_sample_ids") is not False
        or validation_view.get("contains_final_reference_class_labels") is not False
    ):
        raise ValueError("privacy-safe development validation view is not final-blind")
    raw_policy = validation_view.get("qc_policy")
    if (
        not isinstance(raw_policy, Mapping)
        or set(raw_policy)
        != {
            "source_masks_modified",
            "disconnected_instance_ids_are_fatal",
            "complete_policy_sha256",
        }
        or type(raw_policy.get("source_masks_modified")) is not bool
        or type(raw_policy.get("disconnected_instance_ids_are_fatal")) is not bool
        or not isinstance(raw_policy.get("complete_policy_sha256"), str)
        or len(str(raw_policy["complete_policy_sha256"])) != 64
        or validation_view.get("expected_official_fold_ids") != [1, 2, 3]
        or validation_view.get("development_official_fold_ids") != [1, 2]
        or validation_view.get("final_reference_official_fold_id") != 3
        or validation_view.get("release_complete") is not True
    ):
        raise ValueError("privacy-safe development validation policy is invalid")
    root = Path(str(validation_view.get("root", ""))).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"certified PanNuke raw root is missing: {root}")
    if root != expected_data_root.resolve():
        raise ValueError(
            "pre-pilot gate certificate raw root differs from the explicitly supplied dataset"
        )
    inventory = _verified_inventory(root, validation_view.get("raw_file_inventory"))
    if canonical_sha256([record.as_dict() for record in inventory]) != validation_view.get(
        "raw_file_inventory_sha256"
    ):
        raise ValueError("pilot gate certificate raw inventory hash is invalid")
    raw_mapping = validation_view.get("class_mapping")
    if not isinstance(raw_mapping, Mapping) or set(raw_mapping) != {
        "class_names",
        "source",
        "source_revision",
        "verified",
        "source_note",
    }:
        raise ValueError("pilot gate certificate class mapping is invalid")
    expected_mapping = OFFICIAL_METRICS_CLASS_MAPPING.as_dict()
    expected_mapping["class_names"] = list(expected_mapping["class_names"])
    if dict(raw_mapping) != expected_mapping:
        raise ValueError("pilot gate certificate class mapping differs from pinned evidence")
    mapping = VerifiedClassMapping(
        class_names=tuple(str(value) for value in raw_mapping.get("class_names", ())),
        source=str(raw_mapping.get("source", "")),
        source_revision=str(raw_mapping.get("source_revision", "")),
        verified=bool(raw_mapping.get("verified")),
        source_note=str(raw_mapping.get("source_note", "")),
    )
    raw_folds = validation_view.get("folds")
    if not isinstance(raw_folds, list) or any(
        not isinstance(item, Mapping)
        or set(item)
        != {
            "fold_id",
            "image_path",
            "mask_path",
            "tissue_path",
            "image_channel_axis",
            "mask_channel_axis",
        }
        or type(item.get("fold_id")) is not int
        or type(item.get("image_channel_axis")) is not int
        or type(item.get("mask_channel_axis")) is not int
        or any(
            not isinstance(item.get(field), str) or not str(item[field])
            for field in ("image_path", "mask_path", "tissue_path")
        )
        for item in raw_folds
    ):
        raise ValueError("pilot gate certificate development-fold evidence is invalid")
    inventory_by_path = {record.relative_path.casefold(): record for record in inventory}
    bound_descriptor_paths: set[str] = set()
    resolved_descriptor_paths: dict[tuple[int, str], Path] = {}
    for item in raw_folds:
        assert isinstance(item, Mapping)
        fold_id = int(item["fold_id"])
        for role, field in (
            ("image", "image_path"),
            ("mask", "mask_path"),
            ("tissue", "tissue_path"),
        ):
            declared_path = Path(str(item[field]))
            if declared_path.is_absolute() or ".." in declared_path.parts:
                raise ValueError("certified development array path is unsafe")
            resolved_path = (root / declared_path).resolve()
            try:
                relative_path = resolved_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError("certified development array path escapes raw root") from error
            folded_path = relative_path.casefold()
            inventory_record = inventory_by_path.get(folded_path)
            if (
                inventory_record is None
                or inventory_record.relative_path != relative_path
                or inventory_record.fold_id != fold_id
                or inventory_record.file_kind != "npy"
                or resolved_path.suffix.casefold() != ".npy"
                or folded_path in bound_descriptor_paths
            ):
                raise ValueError(
                    "certified development array descriptor is not exactly bound to its "
                    f"raw-inventory fold/role: fold={fold_id}, role={role}"
                )
            bound_descriptor_paths.add(folded_path)
            resolved_descriptor_paths[(fold_id, role)] = resolved_path
    folds = tuple(
        DiscoveredFold(
            fold_id=int(item["fold_id"]),
            image_path=resolved_descriptor_paths[(int(item["fold_id"]), "image")],
            mask_path=resolved_descriptor_paths[(int(item["fold_id"]), "mask")],
            tissue_path=resolved_descriptor_paths[(int(item["fold_id"]), "tissue")],
            image_channel_axis=int(item["image_channel_axis"]),
            mask_channel_axis=int(item["mask_channel_axis"]),
        )
        for item in raw_folds
        if isinstance(item, Mapping)
    )
    if {fold.fold_id for fold in folds} != {1, 2}:
        raise ValueError("pilot gate certificate exposes a non-development fold")
    for fold in folds:
        for path in (fold.image_path, fold.mask_path, fold.tissue_path):
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("certified development array path escapes raw root") from error
            if not path.is_file():
                raise FileNotFoundError(f"certified development array is missing: {path}")
    discovery = ReleaseDiscovery(
        root=root,
        fold_ids=tuple(sorted(fold.fold_id for fold in folds)),
        npy_files=tuple(
            sorted(
                (
                    path
                    for fold in folds
                    for path in (fold.image_path, fold.mask_path, fold.tissue_path)
                ),
                key=lambda path: path.as_posix(),
            )
        ),
        archives=tuple(
            sorted(
                (
                    (root / record.relative_path).resolve()
                    for record in inventory
                    if Path(record.relative_path).suffix.lower() == ".zip"
                ),
                key=lambda path: path.as_posix(),
            )
        ),
        inspections=(),
        folds=folds,
    )
    context = validate_discovered_release(
        discovery,
        class_mapping=mapping,
        use_documented_default_mapping=False,
        max_samples_per_fold=100000,
        expected_fold_ids=(1, 2),
        max_qc_overlay_patches=24,
    )
    if tuple(record.as_dict() for record in context.inventory) != tuple(
        record.as_dict() for record in inventory
    ):
        raise RuntimeError("development semantic validation raw inventory differs from certificate")
    if sha256_file(certificate_path) != certificate_digest:
        raise RuntimeError("pre-pilot gate certificate changed during verification")
    return context, certificate, certificate_digest


def _validate_gate(
    validation: PanNukeValidationResult,
    class_free_frame: Any,
    data_config: Mapping[str, Any],
) -> tuple[tuple[int, ...], int, int, float]:
    if not validation.release_complete:
        raise ValueError("the real pilot requires a complete validated PanNuke release")
    if validation.expected_fold_ids != (1, 2):
        raise ValueError("the pilot semantic validator must expose development folds 1 and 2 only")
    if any(facts.validation_scope != "full_semantic_scan" for facts in validation.fold_validation):
        raise ValueError("every official fold must have full-semantic-scan evidence")
    if validation.mapping.class_names != (
        "neoplastic",
        "inflammatory",
        "connective_soft_tissue",
        "dead",
        "non_neoplastic_epithelial",
    ):
        raise ValueError("the verified five-class mapping does not match the declared pilot")
    if set(int(value) for value in class_free_frame["official_fold"].unique()) != {1, 2, 3}:
        raise ValueError("the manifest must contain all and only official folds 1, 2, and 3")
    development_folds = tuple(int(value) for value in data_config["development_official_folds"])
    final_fold = int(data_config["final_test_fold"])
    group_limit = int(data_config["development_group_limit"])
    reference_fraction = float(data_config["reference_validation_fraction_groups"])
    if development_folds != (1, 2) or final_fold != 3:
        raise ValueError("pilot outer structure is fixed to development folds 1/2 and final fold 3")
    if (
        data_config.get("final_reference_access_policy")
        != "metadata_only_until_preregistration_freeze"
    ):
        raise ValueError(
            "pilot final-reference access must remain metadata-only until preregistration freeze"
        )
    if data_config.get("final_group_limit") is not None:
        raise ValueError("final_group_limit must be null: the complete final fold is mandatory")
    if group_limit <= 0:
        raise ValueError("development_group_limit must be a positive explicit fixed count")
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference-validation group fraction must lie in (0, 1)")
    return development_folds, final_fold, group_limit, reference_fraction


def _select_groups(
    development_frame: Any,
    final_metadata_frame: Any,
    *,
    development_folds: tuple[int, ...],
    final_fold: int,
    development_group_limit: int,
    reference_fraction: float,
    selection_seed: int,
    oof_splits: int,
) -> dict[str, Any]:
    development = development_frame[
        development_frame["official_fold"].isin(development_folds)
    ].copy()
    final = final_metadata_frame[final_metadata_frame["official_fold"] == final_fold].copy()
    available_groups = np.asarray(sorted(development["group_id"].astype(str).unique()))
    if len(available_groups) < development_group_limit:
        raise ValueError(
            "declared development_group_limit cannot be met: "
            f"requested {development_group_limit}, found {len(available_groups)}"
        )
    rng = np.random.default_rng(selection_seed)
    selected_groups = tuple(
        str(value)
        for value in available_groups[
            rng.permutation(len(available_groups))[:development_group_limit]
        ]
    )
    selected = development[development["group_id"].astype(str).isin(selected_groups)].copy()
    selected_fold_ids = set(int(value) for value in selected["official_fold"].unique())
    if selected_fold_ids != set(development_folds):
        raise ValueError("the fixed development group sample does not cover both development folds")
    reference_count = int(np.ceil(development_group_limit * reference_fraction))
    reference_order = rng.permutation(np.asarray(selected_groups, dtype=np.str_))
    reference_groups = tuple(str(value) for value in reference_order[:reference_count])
    audit_groups = tuple(str(value) for value in reference_order[reference_count:])
    if len(audit_groups) < oof_splits:
        raise ValueError("the fixed group sample leaves too few audit groups for grouped OOF")
    audit = selected[selected["group_id"].astype(str).isin(audit_groups)].copy()
    reference = selected[selected["group_id"].astype(str).isin(reference_groups)].copy()
    final_groups = tuple(sorted(final["group_id"].astype(str).unique()))
    if not final_groups:
        raise ValueError("the complete final official fold is empty")
    group_sets = {
        "audit": set(audit_groups),
        "reference_validation": set(reference_groups),
        "final_reference": set(final_groups),
    }
    pairwise_group_overlaps = {
        "audit/reference_validation": sorted(
            group_sets["audit"].intersection(group_sets["reference_validation"])
        ),
        "audit/final_reference": sorted(
            group_sets["audit"].intersection(group_sets["final_reference"])
        ),
        "reference_validation/final_reference": sorted(
            group_sets["reference_validation"].intersection(group_sets["final_reference"])
        ),
    }
    nonempty_overlaps = {name: values for name, values in pairwise_group_overlaps.items() if values}
    if nonempty_overlaps:
        counts = "; ".join(f"{name}={len(values)}" for name, values in nonempty_overlaps.items())
        raise RuntimeError(
            "source-group overlap occurred during deterministic pilot selection: " + counts
        )
    for partition_name, partition in (
        ("audit pool", audit),
        ("reference-validation partition", reference),
    ):
        classes = set(int(value) for value in partition["pre_corruption_label"].unique())
        if classes != set(CLASS_ORDER):
            raise ValueError(
                f"the fixed {partition_name} does not contain all five positive classes: "
                f"found {sorted(classes)}"
            )
    return {
        "selected_development_groups": selected_groups,
        "audit_groups": audit_groups,
        "reference_validation_groups": reference_groups,
        "final_reference_groups": final_groups,
        "pairwise_group_overlap_counts": {
            name: len(values) for name, values in pairwise_group_overlaps.items()
        },
        "audit_frame": audit,
        "reference_frame": reference,
        "final_frame": final,
    }


def _require_official_embedding_provenance(
    representations: PanNukeRepresentationArtifacts,
    expected_sample_ids: tuple[str, ...],
) -> NDArray[np.float64]:
    result = representations.embeddings
    result.validate()
    if tuple(result.sample_ids.tolist()) != expected_sample_ids:
        raise RuntimeError("ResNet-18 cache order differs from the selected manifest sample order")
    metadata = result.metadata
    required = {
        "encoder_name": "torchvision.resnet18",
        "encoder_frozen": True,
        "input_variant": "target_highlighted_rgb",
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise RuntimeError(f"ResNet-18 provenance field {field} is not {expected!r}")
    weight_identifier = str(metadata.get("weight_identifier", ""))
    weight_sha256 = str(metadata.get("weight_sha256", ""))
    if "IMAGENET1K_V1" not in weight_identifier:
        raise RuntimeError("pilot embeddings do not use official IMAGENET1K_V1 weights")
    if len(weight_sha256) != 64:
        raise RuntimeError("pilot embedding provenance lacks the official weight SHA-256")
    return np.asarray(result.embeddings, dtype=np.float64)


def _review_payload(result: ReviewBudgetResult) -> dict[str, Any]:
    return {
        "budget_fraction": result.budget_fraction,
        "total_examples": result.total_examples,
        "reviewed_count": result.reviewed_count,
        "injected_total": result.injected_total,
        "injected_reviewed": result.injected_reviewed,
        "precision": result.precision,
        "recall": result.recall,
        "expected_random_recall": result.expected_random_recall,
        "lift_over_random": result.lift_over_random,
        "average_precision": result.average_precision,
        "reviewed_indices": result.reviewed_indices.tolist(),
    }


def _sample_order_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in sample_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_npz(path: Path, **arrays: Any) -> Path:
    return atomic_write_npz(path, arrays)


def _is_final_sensitive_path(path: str) -> bool:
    normalised = re.sub(r"[^a-z0-9]+", "_", path.casefold())
    return any(scope in normalised for scope in _FINAL_SENSITIVE_SCOPE_ALIASES) and any(
        token in normalised for token in _FINAL_SENSITIVE_PATH_TOKENS
    )


def _privacy_scan_value(value: object, *, path: str, artifact: Path) -> None:
    """Reject final-reference identities/outcomes in structured public values."""

    if _FINAL_REFERENCE_SAMPLE_ID.search(path):
        raise RuntimeError(
            f"pilot privacy reconciliation found a final-reference sample ID in {artifact}"
        )
    if _is_final_sensitive_path(path) and value is not False:
        raise RuntimeError(
            "pilot privacy reconciliation found a populated final-sensitive "
            f"field at {artifact}:{path}"
        )
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            _privacy_scan_value(child, path=child_path, artifact=artifact)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _privacy_scan_value(child, path=f"{path}[{index}]", artifact=artifact)
        return
    if isinstance(value, bytes):
        if _FINAL_REFERENCE_SAMPLE_ID_BYTES.search(value):
            raise RuntimeError(
                f"pilot privacy reconciliation found a final-reference sample ID in {artifact}"
            )
        try:
            text_value = value.decode("utf-8")
        except UnicodeDecodeError:
            return
    elif isinstance(value, str):
        text_value = value
    else:
        return
    if _FINAL_REFERENCE_SAMPLE_ID.search(text_value):
        raise RuntimeError(
            f"pilot privacy reconciliation found a final-reference sample ID in {artifact}"
        )
    stripped = text_value.strip()
    if stripped.startswith(("{", "[")):
        try:
            embedded = json.loads(stripped)
        except json.JSONDecodeError:
            return
        _privacy_scan_value(embedded, path=f"{path}.embedded_json", artifact=artifact)


def _scan_numpy_array_privacy(
    values: NDArray[np.generic],
    *,
    path: str,
    artifact: Path,
) -> None:
    dtype = values.dtype
    if dtype.hasobject or dtype.kind == "O":
        raise RuntimeError(f"pilot NPZ contains a forbidden object dtype: {artifact}:{path}")
    if dtype.names is not None:
        structured_values: Any = values
        canonical_names = set(dtype.names)
        assert dtype.fields is not None
        for raw_alias in dtype.fields:
            alias = str(raw_alias)
            if alias not in canonical_names:
                _privacy_scan_value(
                    alias,
                    path=f"{path}.dtype.aliases.{alias}",
                    artifact=artifact,
                )
        for field_name in dtype.names:
            field_path = f"{path}.{field_name}"
            _privacy_scan_value(field_name, path=field_path, artifact=artifact)
            field_definition = dtype.fields[field_name]
            if len(field_definition) >= 3 and field_definition[2] is not None:
                title = str(field_definition[2])
                _privacy_scan_value(
                    title,
                    path=f"{field_path}.dtype.title",
                    artifact=artifact,
                )
            _scan_numpy_array_privacy(
                np.asarray(structured_values[field_name]),
                path=field_path,
                artifact=artifact,
            )
        return
    if dtype.subdtype is not None:
        base_dtype, _ = dtype.subdtype
        _scan_numpy_array_privacy(
            values.view(base_dtype),
            path=f"{path}.subarray",
            artifact=artifact,
        )
        return
    if dtype.kind == "V":
        raise RuntimeError(f"pilot NPZ contains an opaque unscannable dtype: {artifact}:{path}")
    if dtype.kind in {"U", "S"}:
        for index, item in enumerate(values.reshape(-1).tolist()):
            _privacy_scan_value(
                item,
                path=f"{path}[{index}]",
                artifact=artifact,
            )


def _scan_encoded_metadata(
    metadata: Mapping[bytes, bytes] | None,
    *,
    path: str,
    artifact: Path,
) -> None:
    for raw_key, raw_value in (metadata or {}).items():
        key = raw_key.decode("utf-8", errors="ignore")
        decoded_value = raw_value.decode("utf-8", errors="ignore")
        try:
            structured_value: object = json.loads(decoded_value)
        except json.JSONDecodeError:
            structured_value = decoded_value
        _privacy_scan_value(
            structured_value,
            path=f"{path}.{key}",
            artifact=artifact,
        )


def _scan_arrow_field_privacy(field: Any, *, path: str, artifact: Path) -> None:
    field_path = f"{path}.{field.name}"
    _privacy_scan_value(field.name, path=field_path, artifact=artifact)
    _scan_encoded_metadata(
        field.metadata,
        path=f"{field_path}.metadata",
        artifact=artifact,
    )
    field_type = field.type
    _privacy_scan_value(
        str(field_type),
        path=f"{field_path}.type",
        artifact=artifact,
    )
    if isinstance(field_type, pa.ExtensionType):
        _privacy_scan_value(
            field_type.extension_name,
            path=f"{field_path}.type.extension_name",
            artifact=artifact,
        )
        try:
            extension_metadata = field_type.__arrow_ext_serialize__()
        except Exception as error:
            raise RuntimeError(
                f"pilot Parquet extension metadata is unreadable: {artifact}:{field_path}"
            ) from error
        _privacy_scan_value(
            extension_metadata,
            path=f"{field_path}.type.extension_metadata",
            artifact=artifact,
        )
        _scan_arrow_field_privacy(
            pa.field("storage", field_type.storage_type),
            path=f"{field_path}.type",
            artifact=artifact,
        )
    if hasattr(field_type, "field"):
        for index in range(int(field_type.num_fields)):
            _scan_arrow_field_privacy(
                field_type.field(index),
                path=f"{field_path}.children[{index}]",
                artifact=artifact,
            )


def _scan_npz_privacy(
    path: Path,
    *,
    final_fold: int,
    allowed_development_sample_ids: set[str] | None,
) -> int:
    with np.load(path, allow_pickle=False) as payload:
        try:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        except ValueError as error:
            raise RuntimeError(
                f"pilot NPZ contains a forbidden or unreadable dtype: {path}"
            ) from error
        for name, values in arrays.items():
            _privacy_scan_value(values, path=name, artifact=path)
            _scan_numpy_array_privacy(values, path=name, artifact=path)
            lowered = name.casefold()
            if lowered in {
                "official_fold",
                "official_folds",
                "source_official_fold",
                "source_official_folds",
            }:
                if values.dtype.kind not in {"i", "u"}:
                    raise RuntimeError(
                        f"representation fold array does not contain exact integers: {path}:{name}"
                    )
                observed_folds = {int(value) for value in np.asarray(values).reshape(-1).tolist()}
                if final_fold in observed_folds:
                    raise RuntimeError(
                        "pilot privacy reconciliation found final-fold rows in a "
                        f"representation array: {path}:{name}"
                    )
        if "sample_ids" not in payload.files:
            raise RuntimeError(f"pilot NPZ lacks development sample-order identity binding: {path}")
        raw_sample_ids = arrays["sample_ids"]
        if raw_sample_ids.dtype.kind not in {"U", "S"}:
            raise RuntimeError(f"pilot NPZ sample_ids are not strings: {path}")
        sample_ids = raw_sample_ids.astype(str).reshape(-1).tolist()
        if (
            allowed_development_sample_ids is None
            or len(sample_ids) != len(set(sample_ids))
            or not set(sample_ids).issubset(allowed_development_sample_ids)
        ):
            raise RuntimeError(
                f"pilot NPZ sample identities are not bound to the development view: {path}"
            )
        assert allowed_development_sample_ids is not None
        for name, values in arrays.items():
            lowered = name.casefold()
            if values.dtype.kind in {"U", "S"}:
                string_values = values.astype(str).reshape(-1).tolist()
                if (
                    lowered
                    in {
                        "sample_ids",
                        "held_out_sample_ids",
                        "neighbour_ids",
                    }
                    or lowered.endswith("_sample_ids")
                ) and not {value for value in string_values if value}.issubset(
                    allowed_development_sample_ids
                ):
                    raise RuntimeError(
                        f"pilot NPZ identity array is outside the development view: {path}:{name}"
                    )
    return len(arrays)


def _scan_parquet_privacy(
    path: Path,
    *,
    final_fold: int,
    require_complete_development_scope: bool = False,
) -> tuple[int, set[str]]:
    table = pq.read_table(path)
    _privacy_scan_value(
        table.schema.serialize().to_pybytes(),
        path="schema.serialized",
        artifact=path,
    )
    _scan_encoded_metadata(
        table.schema.metadata,
        path="schema.metadata",
        artifact=path,
    )
    for index, field in enumerate(table.schema):
        _scan_arrow_field_privacy(
            field,
            path=f"schema.fields[{index}]",
            artifact=path,
        )
    if "official_fold" in table.column_names:
        raw_folds = table["official_fold"].to_pylist()
        if any(type(value) is not int for value in raw_folds):
            raise RuntimeError(f"Parquet official_fold values are not exact integers: {path}")
        folds = set(raw_folds)
        if final_fold in folds:
            raise RuntimeError(
                f"pilot privacy reconciliation found final-fold rows in Parquet: {path}"
            )
        if require_complete_development_scope and folds != {1, 2}:
            raise RuntimeError(
                "run-local development manifest view does not contain exactly official "
                f"folds 1 and 2: {path}"
            )
    elif require_complete_development_scope:
        raise RuntimeError(f"development manifest view lacks official_fold: {path}")
    for name in table.column_names:
        _privacy_scan_value(table[name], path=name, artifact=path)
        for index, value in enumerate(table[name].to_pylist()):
            _privacy_scan_value(value, path=f"{name}[{index}]", artifact=path)
    sample_ids: set[str] = set()
    if "sample_id" in table.column_names:
        raw_sample_ids = table["sample_id"].to_pylist()
        if any(not isinstance(value, str) for value in raw_sample_ids):
            raise RuntimeError(f"Parquet sample_id column is not string-valued: {path}")
        sample_ids = {str(value) for value in raw_sample_ids}
        if len(sample_ids) != len(raw_sample_ids):
            raise RuntimeError(f"Parquet sample_id values are duplicated: {path}")
    elif require_complete_development_scope:
        raise RuntimeError(f"development manifest view lacks sample_id: {path}")
    return len(table), sample_ids


def _reconcile_pilot_final_reference_privacy(
    run_directory: Path,
    *,
    final_fold: int,
) -> dict[str, Any]:
    """Scan every public text/NPZ/Parquet artifact before run sealing."""

    text_files = 0
    npz_files = 0
    npz_arrays = 0
    parquet_files = 0
    parquet_rows = 0
    scanned_paths: list[str] = []
    development_manifest_path = run_directory / "development_manifest_view.parquet"
    allowed_development_sample_ids: set[str] | None = None
    if development_manifest_path.is_file():
        development_rows, allowed_development_sample_ids = _scan_parquet_privacy(
            development_manifest_path,
            final_fold=final_fold,
            require_complete_development_scope=True,
        )
        parquet_files += 1
        parquet_rows += development_rows
        scanned_paths.append(development_manifest_path.name)
    for path in sorted(run_directory.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        if path == development_manifest_path:
            continue
        suffix = path.suffix.casefold()
        relative = path.relative_to(run_directory).as_posix()
        _privacy_scan_value(
            relative,
            path="artifact.relative_path",
            artifact=path,
        )
        for index, component in enumerate(relative.split("/")):
            _privacy_scan_value(
                component,
                path=f"artifact.path_components[{index}]",
                artifact=path,
            )
        if suffix in _PUBLIC_TEXT_SUFFIXES:
            text_files += 1
            scanned_paths.append(relative)
            try:
                contents = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(
                    f"public pilot text artifact is not valid UTF-8: {path}"
                ) from error
            if _FINAL_REFERENCE_SAMPLE_ID.search(contents):
                raise RuntimeError(
                    f"pilot privacy reconciliation found a final-reference sample ID in {path}"
                )
            if suffix == ".json":
                try:
                    structured = json.loads(contents)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"public pilot JSON is invalid: {path}") from error
                _privacy_scan_value(structured, path="", artifact=path)
            elif suffix == ".jsonl":
                for line_number, line in enumerate(contents.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        structured = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            f"public pilot JSONL is invalid: {path}:{line_number}"
                        ) from error
                    _privacy_scan_value(
                        structured,
                        path=f"line[{line_number}]",
                        artifact=path,
                    )
            elif suffix == ".csv":
                reader = csv.reader(io.StringIO(contents))
                header = next(reader, [])
                for column in header:
                    if _is_final_sensitive_path(column):
                        raise RuntimeError(
                            "pilot privacy reconciliation found a forbidden final-sensitive "
                            f"CSV column at {path}:{column}"
                        )
            elif suffix in {".yaml", ".yml"}:
                try:
                    structured = yaml.safe_load(contents)
                except yaml.YAMLError as error:
                    raise RuntimeError(f"public pilot YAML is invalid: {path}") from error
                _privacy_scan_value(structured, path="", artifact=path)
        elif suffix == ".npz":
            npz_files += 1
            scanned_paths.append(relative)
            npz_arrays += _scan_npz_privacy(
                path,
                final_fold=final_fold,
                allowed_development_sample_ids=allowed_development_sample_ids,
            )
        elif suffix == ".parquet":
            parquet_files += 1
            scanned_paths.append(relative)
            rows, _ = _scan_parquet_privacy(path, final_fold=final_fold)
            parquet_rows += rows
    if not scanned_paths:
        raise RuntimeError("pilot privacy reconciliation found no public artifacts to scan")
    return {
        "schema_version": 1,
        "status": "passed",
        "policy": "final_reference_identity_and_outcome_nonpublication_v1",
        "final_reference_official_fold": final_fold,
        "final_fold_identity_pattern_absent": True,
        "final_sensitive_fields_unpopulated": True,
        "final_fold_representation_rows_absent": True,
        "scanned_file_count": len(scanned_paths),
        "text_file_count": text_files,
        "npz_file_count": npz_files,
        "npz_array_count": npz_arrays,
        "parquet_file_count": parquet_files,
        "parquet_row_count": parquet_rows,
        "scanned_paths": scanned_paths,
    }


def _write_cleanlab_audit_evidence(
    run_directory: Path,
    result: CleanlabScoreResult,
    *,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    observed_labels: NDArray[np.int64],
) -> dict[str, Any]:
    identifiers = tuple(str(value) for value in sample_ids)
    groups = tuple(str(value) for value in group_ids)
    n_samples = len(identifiers)
    if (
        not result.available
        or result.quality_scores is None
        or result.risk_scores is None
        or result.issue_mask is None
    ):
        raise ValueError("available Cleanlab evidence requires quality, risk, and issue arrays")
    quality = np.asarray(result.quality_scores, dtype=np.float64)
    risk = np.asarray(result.risk_scores, dtype=np.float64)
    issues = np.asarray(result.issue_mask, dtype=np.bool_)
    observed = np.asarray(observed_labels, dtype=np.int64)
    if (
        len(groups) != n_samples
        or len(set(identifiers)) != n_samples
        or quality.shape != (n_samples,)
        or risk.shape != (n_samples,)
        or issues.shape != (n_samples,)
        or observed.shape != (n_samples,)
        or not np.isfinite(quality).all()
        or not np.isfinite(risk).all()
        or not np.allclose(risk, 1.0 - quality, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Cleanlab per-sample evidence is not finite and exactly aligned")

    # The stable APIs called by cleanlab_scores return quality values and issue flags,
    # not suggested labels.  Its convenience argmax is deliberately not persisted as
    # a Cleanlab suggestion because that would misstate API provenance.
    suggestion_status = "unavailable_api_did_not_return_suggested_labels"
    npz_path = _write_npz(
        run_directory / "cleanlab_evidence.npz",
        schema_version=np.asarray(1, dtype=np.int64),
        sample_ids=np.asarray(identifiers, dtype=np.str_),
        group_ids=np.asarray(groups, dtype=np.str_),
        observed_label=observed,
        quality_scores=quality,
        risk_scores=risk,
        issue_mask=issues,
        suggested_labels_available=np.asarray(False, dtype=np.bool_),
        suggested_labels_status=np.asarray(suggestion_status, dtype=np.str_),
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        (
            "sample_id",
            "group_id",
            "observed_label",
            "quality_score",
            "risk_score",
            "is_label_issue",
            "suggested_label",
            "suggested_label_status",
        )
    )
    for index, sample_id in enumerate(identifiers):
        writer.writerow(
            (
                sample_id,
                groups[index],
                int(observed[index]),
                float(quality[index]),
                float(risk[index]),
                bool(issues[index]),
                "",
                suggestion_status,
            )
        )
    csv_path = atomic_write_text(run_directory / "cleanlab_evidence.csv", stream.getvalue())
    metadata = {
        "schema_version": 1,
        "available": True,
        "package_name": "cleanlab",
        "package_version": result.package_version,
        "api_path": result.api_path,
        "error": result.error,
        "sample_count": n_samples,
        "sample_order_sha256": _sample_order_sha256(identifiers),
        "quality_scores_available": True,
        "issue_flags_available": True,
        "suggested_labels": {
            "available": False,
            "status": suggestion_status,
            "reason": (
                "The invoked Cleanlab quality-score and issue-filter APIs do not return "
                "suggested labels; no substitute was fabricated."
            ),
        },
        "npz": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
        "csv": {"path": csv_path.name, "sha256": sha256_file(csv_path)},
    }
    metadata_path = atomic_write_json(run_directory / "cleanlab_evidence.json", metadata)
    return {**metadata, "metadata_path": metadata_path.name}


def _write_neighbour_audit_evidence(
    run_directory: Path,
    result: NeighbourDisagreementResult,
    *,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    observed_labels: NDArray[np.int64],
    fold_ids: NDArray[np.int64],
    class_order: Sequence[int],
) -> dict[str, Any]:
    identifiers = tuple(str(value) for value in sample_ids)
    groups = tuple(str(value) for value in group_ids)
    observed = np.asarray(observed_labels, dtype=np.int64)
    folds = np.asarray(fold_ids, dtype=np.int64)
    classes = tuple(int(value) for value in class_order)
    n_samples = len(identifiers)
    result.validate(identifiers, groups)
    if observed.shape != (n_samples,) or folds.shape != (n_samples,):
        raise ValueError("neighbour labels/folds do not align with query samples")
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError("neighbour evidence class order must be non-empty and unique")
    counts = np.asarray([len(row) for row in result.neighbour_ids], dtype=np.int64)
    if counts.shape != (n_samples,) or np.any(counts <= 0) or np.any(counts > result.k):
        raise ValueError("neighbour evidence counts must lie in [1, k]")
    width = int(counts.max())
    id_width = max(
        1,
        *(len(value) for value in identifiers),
        *(len(value) for row in result.neighbour_ids for value in row),
    )
    group_width = max(
        1,
        *(len(value) for value in groups),
        *(len(value) for row in result.neighbour_groups for value in row),
    )
    neighbour_ids = np.full((n_samples, width), "", dtype=f"<U{id_width}")
    neighbour_groups = np.full((n_samples, width), "", dtype=f"<U{group_width}")
    distances = np.full((n_samples, width), np.nan, dtype=np.float64)
    weights = np.full((n_samples, width), np.nan, dtype=np.float64)
    neighbour_labels = np.full((n_samples, width), -1, dtype=np.int64)
    support = np.zeros((n_samples, len(classes)), dtype=np.float64)
    sample_to_index = {sample_id: index for index, sample_id in enumerate(identifiers)}
    class_to_column = {label: index for index, label in enumerate(classes)}
    same_group_excluded = np.ones(n_samples, dtype=np.bool_)

    for query_index, (ids, row_groups, row_distances) in enumerate(
        zip(
            result.neighbour_ids,
            result.neighbour_groups,
            result.neighbour_distances,
            strict=True,
        )
    ):
        count = int(counts[query_index])
        if len(row_groups) != count or len(row_distances) != count:
            raise ValueError("neighbour identity/group/distance rows do not align")
        row_distance_array = np.asarray(row_distances, dtype=np.float64)
        if not np.isfinite(row_distance_array).all() or np.any(row_distance_array < 0.0):
            raise ValueError("neighbour distances must be finite and non-negative")
        row_labels = np.empty(count, dtype=np.int64)
        for position, (neighbour_id, neighbour_group) in enumerate(
            zip(ids, row_groups, strict=True)
        ):
            neighbour_index = sample_to_index.get(str(neighbour_id))
            if neighbour_index is None:
                raise ValueError("neighbour evidence references an unknown audit sample")
            if groups[neighbour_index] != str(neighbour_group):
                raise ValueError("neighbour sample/group provenance differs from audit order")
            row_labels[position] = int(observed[neighbour_index])
        raw_weights = 1.0 / np.maximum(row_distance_array, 1e-8)
        normalised_weights = raw_weights / raw_weights.sum()
        row_support = np.zeros(len(classes), dtype=np.float64)
        for label, weight in zip(row_labels, normalised_weights, strict=True):
            if int(label) not in class_to_column:
                raise ValueError("neighbour label is absent from class order")
            row_support[class_to_column[int(label)]] += float(weight)
        observed_column = class_to_column[int(observed[query_index])]
        alternative = row_support.copy()
        alternative[observed_column] = -np.inf
        if (
            not np.isclose(
                result.risk_scores[query_index],
                1.0 - row_support[observed_column],
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                result.alternative_class_support[query_index],
                alternative.max(),
                rtol=0.0,
                atol=1e-12,
            )
            or int(result.suggested_class[query_index]) != classes[int(np.argmax(row_support))]
        ):
            raise ValueError("saved neighbour identities do not reconstruct derived decisions")
        neighbour_ids[query_index, :count] = ids
        neighbour_groups[query_index, :count] = row_groups
        distances[query_index, :count] = row_distance_array
        weights[query_index, :count] = normalised_weights
        neighbour_labels[query_index, :count] = row_labels
        support[query_index] = row_support
        same_group_excluded[query_index] = all(
            str(value) != groups[query_index] for value in row_groups
        )
    if not bool(same_group_excluded.all()):
        raise ValueError("neighbour evidence contains a query-group neighbour")

    npz_path = _write_npz(
        run_directory / "neighbour_evidence.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        sample_ids=np.asarray(identifiers, dtype=np.str_),
        group_ids=np.asarray(groups, dtype=np.str_),
        observed_label=observed,
        fold_id=folds,
        class_order=np.asarray(classes, dtype=np.int64),
        risk_scores=np.asarray(result.risk_scores, dtype=np.float64),
        alternative_class_support=np.asarray(result.alternative_class_support, dtype=np.float64),
        suggested_class=np.asarray(result.suggested_class, dtype=np.int64),
        neighbour_count=counts,
        neighbour_ids=neighbour_ids,
        neighbour_groups=neighbour_groups,
        neighbour_distances=distances,
        neighbour_weights=weights,
        neighbour_observed_labels=neighbour_labels,
        class_support=support,
        same_group_exclusion_verified=same_group_excluded,
        k=np.asarray(result.k, dtype=np.int64),
        metric=np.asarray(result.metric, dtype=np.str_),
    )
    stream = io.StringIO(newline="")
    fieldnames = [
        "query_sample_id",
        "query_group_id",
        "fold_id",
        "query_observed_label",
        "neighbour_position",
        "neighbour_sample_id",
        "neighbour_group_id",
        "neighbour_distance",
        "neighbour_weight",
        "neighbour_observed_label",
        "risk_score",
        "alternative_class_support",
        "suggested_class",
        *(f"support_class_{label}" for label in classes),
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for query_index, sample_id in enumerate(identifiers):
        for position in range(int(counts[query_index])):
            writer.writerow(
                {
                    "query_sample_id": sample_id,
                    "query_group_id": groups[query_index],
                    "fold_id": int(folds[query_index]),
                    "query_observed_label": int(observed[query_index]),
                    "neighbour_position": position + 1,
                    "neighbour_sample_id": neighbour_ids[query_index, position],
                    "neighbour_group_id": neighbour_groups[query_index, position],
                    "neighbour_distance": float(distances[query_index, position]),
                    "neighbour_weight": float(weights[query_index, position]),
                    "neighbour_observed_label": int(neighbour_labels[query_index, position]),
                    "risk_score": float(result.risk_scores[query_index]),
                    "alternative_class_support": float(
                        result.alternative_class_support[query_index]
                    ),
                    "suggested_class": int(result.suggested_class[query_index]),
                    **{
                        f"support_class_{label}": float(support[query_index, column])
                        for column, label in enumerate(classes)
                    },
                }
            )
    csv_path = atomic_write_text(run_directory / "neighbour_evidence.csv", stream.getvalue())
    metadata = {
        "schema_version": 2,
        "sample_count": n_samples,
        "row_count": int(counts.sum()),
        "sample_order_sha256": _sample_order_sha256(identifiers),
        "k": result.k,
        "metric": result.metric,
        "weight_definition": "inverse_distance_then_row_normalised",
        "class_order": classes,
        "same_group_exclusion_verified": True,
        "same_group_exclusion_verified_count": int(same_group_excluded.sum()),
        "npz": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
        "csv": {"path": csv_path.name, "sha256": sha256_file(csv_path)},
    }
    metadata_path = atomic_write_json(run_directory / "neighbour_evidence.json", metadata)
    return {**metadata, "metadata_path": metadata_path.name}


def _read_json_object(path: Path, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return payload


def _metadata_bound_path(run_directory: Path, metadata: Mapping[str, Any], field: str) -> Path:
    record = metadata.get(field)
    if not isinstance(record, Mapping):
        raise ValueError(f"audit evidence metadata lacks {field} binding")
    filename = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"audit evidence {field} path is not a local artifact name")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"audit evidence {field} binding lacks a valid SHA-256")
    path = run_directory / filename
    if not path.is_file():
        raise ValueError(f"required pilot audit evidence is missing: {filename}")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"pilot audit evidence hash mismatch: {filename}")
    return path


def reconcile_pilot_audit_evidence(
    run_directory: str | Path,
    *,
    require_sealed_integrity: bool = True,
) -> dict[str, Any]:
    """Reconstruct Cleanlab and fold-safe-neighbour evidence across pilot artifacts."""

    run_path = Path(run_directory).resolve()
    required = (
        "selected_groups_and_samples.json",
        "corruption_manifest.json",
        "oof_predictions.npz",
        "oof_provenance.json",
        "ranking.csv",
        "cleanlab_evidence.npz",
        "cleanlab_evidence.csv",
        "cleanlab_evidence.json",
        "neighbour_evidence.npz",
        "neighbour_evidence.csv",
        "neighbour_evidence.json",
    )
    missing = tuple(name for name in required if not (run_path / name).is_file())
    if missing:
        raise ValueError(f"required pilot audit evidence is missing: {missing}")
    if require_sealed_integrity:
        integrity = verify_run_integrity(run_path)
        if not integrity.valid or not integrity.registry_record_present:
            raise ValueError(
                "pilot audit evidence is not sealed and registry-backed or failed integrity: "
                f"{integrity.errors}"
            )

    selected = _read_json_object(
        run_path / "selected_groups_and_samples.json", "pilot selection evidence"
    )
    raw_audit_ids = selected.get("audit_sample_ids")
    if not isinstance(raw_audit_ids, list) or not raw_audit_ids:
        raise ValueError("pilot selection evidence lacks ordered audit sample IDs")
    audit_ids = tuple(str(value) for value in raw_audit_ids)
    if len(set(audit_ids)) != len(audit_ids):
        raise ValueError("pilot selection audit sample IDs are not unique")

    corruption_payload = _read_json_object(
        run_path / "corruption_manifest.json", "pilot corruption evidence"
    )
    raw_rows = corruption_payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(audit_ids):
        raise ValueError("pilot corruption rows do not align with the selected audit pool")
    corruption_rows: list[Mapping[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ValueError("pilot corruption evidence contains a non-object row")
        corruption_rows.append(row)
    corruption_ids = tuple(str(row.get("sample_id")) for row in corruption_rows)
    corruption_groups = tuple(str(row.get("group_id")) for row in corruption_rows)
    corruption_pre = np.asarray(
        [row.get("pre_corruption_label") for row in corruption_rows], dtype=np.int64
    )
    corruption_observed = np.asarray(
        [row.get("observed_label") for row in corruption_rows], dtype=np.int64
    )
    corruption_injected = np.asarray(
        [row.get("is_injected_corruption") for row in corruption_rows], dtype=np.bool_
    )
    if corruption_ids != audit_ids:
        raise ValueError("pilot corruption sample order differs from selected audit order")

    with np.load(run_path / "oof_predictions.npz", allow_pickle=False) as payload:
        oof_required = {
            "sample_ids",
            "group_ids",
            "pre_corruption_label",
            "observed_label",
            "is_injected_corruption",
            "fold_id",
            "cleanlab_risk",
            "nearest_neighbour_disagreement",
        }
        absent = oof_required.difference(payload.files)
        if absent:
            raise ValueError(f"pilot OOF evidence lacks arrays: {sorted(absent)}")
        oof_ids = tuple(str(value) for value in np.asarray(payload["sample_ids"]).tolist())
        oof_groups = tuple(str(value) for value in np.asarray(payload["group_ids"]).tolist())
        oof_pre = np.asarray(payload["pre_corruption_label"], dtype=np.int64)
        oof_observed = np.asarray(payload["observed_label"], dtype=np.int64)
        oof_injected = np.asarray(payload["is_injected_corruption"], dtype=np.bool_)
        oof_folds = np.asarray(payload["fold_id"], dtype=np.int64)
        oof_cleanlab_risk = np.asarray(payload["cleanlab_risk"], dtype=np.float64)
        oof_neighbour_risk = np.asarray(payload["nearest_neighbour_disagreement"], dtype=np.float64)
    if (
        oof_ids != audit_ids
        or oof_groups != corruption_groups
        or not np.array_equal(oof_pre, corruption_pre)
        or not np.array_equal(oof_observed, corruption_observed)
        or not np.array_equal(oof_injected, corruption_injected)
    ):
        raise ValueError("pilot OOF sample/group/label order differs from selection/corruption")
    n_samples = len(audit_ids)
    for name, array in {
        "fold_id": oof_folds,
        "cleanlab_risk": oof_cleanlab_risk,
        "nearest_neighbour_disagreement": oof_neighbour_risk,
    }.items():
        if array.shape != (n_samples,):
            raise ValueError(f"pilot OOF {name} does not align with audit samples")

    provenance = _read_json_object(run_path / "oof_provenance.json", "pilot OOF provenance")
    raw_class_order = provenance.get("class_order")
    raw_folds = provenance.get("folds")
    if not isinstance(raw_class_order, list) or not isinstance(raw_folds, list):
        raise ValueError("pilot OOF provenance lacks class order or folds")
    classes = tuple(int(value) for value in raw_class_order)
    split_seed = provenance.get("split_seed")
    if type(split_seed) is not int:
        raise ValueError("pilot OOF provenance lacks an exact integer split seed")
    if provenance.get("fold_assignment_label_source") != "pre_corruption_label":
        raise ValueError("pilot OOF folds must be assigned from pre_corruption_label")
    expected_plan = make_group_stratified_fold_plan(
        oof_pre,
        oof_groups,
        n_splits=len(raw_folds),
        class_order=classes,
        seed=split_seed,
    )
    if (
        provenance.get("splitter_class_name") != expected_plan.splitter_class_name
        or provenance.get("splitter_fallback_status") != expected_plan.splitter_fallback_status
        or provenance.get("splitter_fallback_reason") != expected_plan.splitter_fallback_reason
    ):
        raise ValueError("pilot OOF splitter provenance is not reproducible")
    training_groups: dict[int, set[str]] = {}
    expected_fold_ids = np.full(n_samples, -1, dtype=np.int64)
    for fold, expected_fold in zip(raw_folds, expected_plan.folds, strict=True):
        if not isinstance(fold, Mapping) or not isinstance(fold.get("training_groups"), list):
            raise ValueError("pilot OOF provenance contains an invalid fold")
        fold_id = int(fold["fold_id"])
        training = tuple(str(value) for value in fold["training_groups"])
        held_out = tuple(str(value) for value in fold.get("held_out_groups", ()))
        held_out_ids = tuple(str(value) for value in fold.get("held_out_sample_ids", ()))
        expected_ids = tuple(oof_ids[index] for index in expected_fold.holdout_indices)
        if (
            fold_id != expected_fold.fold_id
            or training != expected_fold.training_groups
            or held_out != expected_fold.held_out_groups
            or held_out_ids != expected_ids
        ):
            raise ValueError("pilot OOF fold provenance differs from the recreated splitter")
        expected_fold_ids[expected_fold.holdout_indices] = expected_fold.fold_id
        training_groups[fold_id] = set(training)
    if not np.array_equal(oof_folds, expected_fold_ids):
        raise ValueError("pilot OOF fold IDs differ from the recreated splitter")

    cleanlab_metadata = _read_json_object(run_path / "cleanlab_evidence.json", "Cleanlab metadata")
    cleanlab_npz_path = _metadata_bound_path(run_path, cleanlab_metadata, "npz")
    cleanlab_csv_path = _metadata_bound_path(run_path, cleanlab_metadata, "csv")
    if (
        cleanlab_metadata.get("available") is not True
        or cleanlab_metadata.get("package_name") != "cleanlab"
        or not isinstance(cleanlab_metadata.get("package_version"), str)
        or not isinstance(cleanlab_metadata.get("api_path"), str)
        or cleanlab_metadata.get("error") is not None
    ):
        raise ValueError("Cleanlab package/API/version/error provenance is incomplete")
    if cleanlab_metadata.get("sample_count") != n_samples or cleanlab_metadata.get(
        "sample_order_sha256"
    ) != _sample_order_sha256(audit_ids):
        raise ValueError("Cleanlab metadata does not bind the exact audit sample order")
    suggestion_record = cleanlab_metadata.get("suggested_labels")
    if (
        not isinstance(suggestion_record, Mapping)
        or suggestion_record.get("available") is not False
    ):
        raise ValueError("Cleanlab suggested-label unavailability is not explicit")
    with np.load(cleanlab_npz_path, allow_pickle=False) as payload:
        clean_required = {
            "sample_ids",
            "group_ids",
            "observed_label",
            "quality_scores",
            "risk_scores",
            "issue_mask",
            "suggested_labels_available",
            "suggested_labels_status",
        }
        absent = clean_required.difference(payload.files)
        if absent:
            raise ValueError(f"Cleanlab evidence lacks arrays: {sorted(absent)}")
        if "suggested_class" in payload.files or "suggested_labels" in payload.files:
            raise ValueError("Cleanlab evidence contains an unproven suggested-label array")
        clean_ids = tuple(str(value) for value in np.asarray(payload["sample_ids"]).tolist())
        clean_groups = tuple(str(value) for value in np.asarray(payload["group_ids"]).tolist())
        clean_observed = np.asarray(payload["observed_label"], dtype=np.int64)
        clean_quality = np.asarray(payload["quality_scores"], dtype=np.float64)
        clean_risk = np.asarray(payload["risk_scores"], dtype=np.float64)
        clean_issues = np.asarray(payload["issue_mask"], dtype=np.bool_)
        suggestions_available = bool(np.asarray(payload["suggested_labels_available"]).item())
        suggestion_status = str(np.asarray(payload["suggested_labels_status"]).item())
    if (
        clean_ids != audit_ids
        or clean_groups != oof_groups
        or not np.array_equal(clean_observed, oof_observed)
        or clean_quality.shape != (n_samples,)
        or clean_risk.shape != (n_samples,)
        or clean_issues.shape != (n_samples,)
        or suggestions_available
        or suggestion_status != suggestion_record.get("status")
        or not np.isfinite(clean_quality).all()
        or not np.isfinite(clean_risk).all()
        or not np.allclose(clean_risk, 1.0 - clean_quality, rtol=0.0, atol=1e-12)
        or not np.allclose(clean_risk, oof_cleanlab_risk, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Cleanlab evidence is not aligned with OOF/corruption evidence")
    with cleanlab_csv_path.open("r", encoding="utf-8", newline="") as handle:
        clean_csv_rows = list(csv.DictReader(handle))
    if tuple(row.get("sample_id", "") for row in clean_csv_rows) != audit_ids:
        raise ValueError("Cleanlab CSV sample order differs from the selected audit order")
    for index, row in enumerate(clean_csv_rows):
        if (
            row.get("group_id") != oof_groups[index]
            or int(row["observed_label"]) != int(oof_observed[index])
            or not np.isclose(float(row["quality_score"]), clean_quality[index])
            or not np.isclose(float(row["risk_score"]), clean_risk[index])
            or (row.get("is_label_issue") == "True") != bool(clean_issues[index])
            or row.get("suggested_label") not in {None, ""}
            or row.get("suggested_label_status") != suggestion_status
        ):
            raise ValueError("Cleanlab CSV values differ from the bound NPZ evidence")

    neighbour_metadata = _read_json_object(
        run_path / "neighbour_evidence.json", "neighbour metadata"
    )
    neighbour_npz_path = _metadata_bound_path(run_path, neighbour_metadata, "npz")
    neighbour_csv_path = _metadata_bound_path(run_path, neighbour_metadata, "csv")
    with np.load(neighbour_npz_path, allow_pickle=False) as payload:
        neighbour_required = {
            "sample_ids",
            "group_ids",
            "observed_label",
            "fold_id",
            "class_order",
            "risk_scores",
            "alternative_class_support",
            "suggested_class",
            "neighbour_count",
            "neighbour_ids",
            "neighbour_groups",
            "neighbour_distances",
            "neighbour_weights",
            "neighbour_observed_labels",
            "class_support",
            "same_group_exclusion_verified",
        }
        absent = neighbour_required.difference(payload.files)
        if absent:
            raise ValueError(f"neighbour evidence lacks arrays: {sorted(absent)}")
        neighbour_ids_order = tuple(
            str(value) for value in np.asarray(payload["sample_ids"]).tolist()
        )
        neighbour_query_groups = tuple(
            str(value) for value in np.asarray(payload["group_ids"]).tolist()
        )
        neighbour_query_observed = np.asarray(payload["observed_label"], dtype=np.int64)
        neighbour_fold = np.asarray(payload["fold_id"], dtype=np.int64)
        neighbour_class_order = tuple(
            int(value) for value in np.asarray(payload["class_order"]).tolist()
        )
        neighbour_risk = np.asarray(payload["risk_scores"], dtype=np.float64)
        neighbour_alternative = np.asarray(payload["alternative_class_support"], dtype=np.float64)
        neighbour_suggested = np.asarray(payload["suggested_class"], dtype=np.int64)
        neighbour_count = np.asarray(payload["neighbour_count"], dtype=np.int64)
        neighbour_sample_ids = np.asarray(payload["neighbour_ids"])
        neighbour_group_ids = np.asarray(payload["neighbour_groups"])
        neighbour_distances = np.asarray(payload["neighbour_distances"], dtype=np.float64)
        neighbour_weights = np.asarray(payload["neighbour_weights"], dtype=np.float64)
        neighbour_labels = np.asarray(payload["neighbour_observed_labels"], dtype=np.int64)
        class_support = np.asarray(payload["class_support"], dtype=np.float64)
        same_group_verified = np.asarray(payload["same_group_exclusion_verified"], dtype=np.bool_)
    if (
        neighbour_ids_order != audit_ids
        or neighbour_query_groups != oof_groups
        or not np.array_equal(neighbour_query_observed, oof_observed)
        or not np.array_equal(neighbour_fold, oof_folds)
        or neighbour_class_order != classes
        or not np.allclose(neighbour_risk, oof_neighbour_risk, rtol=0.0, atol=1e-12)
        or neighbour_metadata.get("sample_order_sha256") != _sample_order_sha256(audit_ids)
        or neighbour_metadata.get("same_group_exclusion_verified") is not True
    ):
        raise ValueError("neighbour evidence query order differs from OOF/corruption evidence")
    if neighbour_metadata.get("sample_count") != n_samples:
        raise ValueError("neighbour metadata sample count differs from the audit pool")
    if (
        neighbour_count.shape != (n_samples,)
        or neighbour_risk.shape != (n_samples,)
        or neighbour_alternative.shape != (n_samples,)
        or neighbour_suggested.shape != (n_samples,)
        or same_group_verified.shape != (n_samples,)
        or class_support.shape != (n_samples, len(classes))
        or neighbour_sample_ids.ndim != 2
        or neighbour_group_ids.shape != neighbour_sample_ids.shape
        or neighbour_distances.shape != neighbour_sample_ids.shape
        or neighbour_weights.shape != neighbour_sample_ids.shape
        or neighbour_labels.shape != neighbour_sample_ids.shape
        or neighbour_sample_ids.shape[0] != n_samples
        or not bool(same_group_verified.all())
    ):
        raise ValueError("neighbour evidence arrays are not aligned")
    sample_to_index = {sample_id: index for index, sample_id in enumerate(audit_ids)}
    class_to_column = {label: column for column, label in enumerate(classes)}
    expected_csv_rows: list[tuple[int, int]] = []
    for query_index in range(n_samples):
        count = int(neighbour_count[query_index])
        if count <= 0 or count > neighbour_sample_ids.shape[1]:
            raise ValueError("neighbour evidence contains an invalid active count")
        active_ids = tuple(str(value) for value in neighbour_sample_ids[query_index, :count])
        active_groups = tuple(str(value) for value in neighbour_group_ids[query_index, :count])
        active_distances = neighbour_distances[query_index, :count]
        active_weights = neighbour_weights[query_index, :count]
        active_labels = neighbour_labels[query_index, :count]
        if (
            any(not value for value in (*active_ids, *active_groups))
            or audit_ids[query_index] in active_ids
            or oof_groups[query_index] in active_groups
            or any(value not in sample_to_index for value in active_ids)
            or not np.isfinite(active_distances).all()
            or np.any(active_distances < 0.0)
            or not np.isfinite(active_weights).all()
            or np.any(active_weights < 0.0)
            or not np.isclose(active_weights.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("neighbour evidence violates identity/group/distance/weight rules")
        allowed_groups = training_groups.get(int(oof_folds[query_index]))
        if allowed_groups is None or any(group not in allowed_groups for group in active_groups):
            raise ValueError("neighbour evidence contains a non-training-fold reference")
        for position, (sample_id, group_id, label) in enumerate(
            zip(active_ids, active_groups, active_labels, strict=True)
        ):
            reference_index = sample_to_index[sample_id]
            if oof_groups[reference_index] != group_id or int(oof_observed[reference_index]) != int(
                label
            ):
                raise ValueError("neighbour ID/group/label evidence is inconsistent")
            expected_csv_rows.append((query_index, position))
        recomputed_weights = 1.0 / np.maximum(active_distances, 1e-8)
        recomputed_weights /= recomputed_weights.sum()
        recomputed_support = np.zeros(len(classes), dtype=np.float64)
        for label, weight in zip(active_labels, recomputed_weights, strict=True):
            if int(label) not in class_to_column:
                raise ValueError("neighbour label is absent from class order")
            recomputed_support[class_to_column[int(label)]] += float(weight)
        observed_column = class_to_column[int(oof_observed[query_index])]
        alternative = recomputed_support.copy()
        alternative[observed_column] = -np.inf
        if (
            not np.allclose(active_weights, recomputed_weights, rtol=0.0, atol=1e-12)
            or not np.allclose(class_support[query_index], recomputed_support, rtol=0.0, atol=1e-12)
            or not np.isclose(
                neighbour_risk[query_index],
                1.0 - recomputed_support[observed_column],
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                neighbour_alternative[query_index],
                alternative.max(),
                rtol=0.0,
                atol=1e-12,
            )
            or int(neighbour_suggested[query_index]) != classes[int(np.argmax(recomputed_support))]
        ):
            raise ValueError("neighbour weights/support do not reconstruct saved decisions")
        if (
            np.any(neighbour_sample_ids[query_index, count:] != "")
            or np.any(neighbour_group_ids[query_index, count:] != "")
            or not np.isnan(neighbour_distances[query_index, count:]).all()
            or not np.isnan(neighbour_weights[query_index, count:]).all()
            or np.any(neighbour_labels[query_index, count:] != -1)
        ):
            raise ValueError("neighbour evidence padding is not canonical")

    with neighbour_csv_path.open("r", encoding="utf-8", newline="") as handle:
        neighbour_csv_rows = list(csv.DictReader(handle))
    if len(neighbour_csv_rows) != len(expected_csv_rows):
        raise ValueError("neighbour CSV row count differs from the bound NPZ")
    if neighbour_metadata.get("row_count") != len(expected_csv_rows):
        raise ValueError("neighbour metadata row count differs from the bound evidence")
    for row, (query_index, position) in zip(neighbour_csv_rows, expected_csv_rows, strict=True):
        if (
            row.get("query_sample_id") != audit_ids[query_index]
            or row.get("query_group_id") != oof_groups[query_index]
            or int(row["fold_id"]) != int(oof_folds[query_index])
            or int(row["neighbour_position"]) != position + 1
            or row.get("neighbour_sample_id") != str(neighbour_sample_ids[query_index, position])
            or row.get("neighbour_group_id") != str(neighbour_group_ids[query_index, position])
            or not np.isclose(
                float(row["neighbour_distance"]),
                neighbour_distances[query_index, position],
            )
            or not np.isclose(
                float(row["neighbour_weight"]), neighbour_weights[query_index, position]
            )
            or int(row["neighbour_observed_label"]) != int(neighbour_labels[query_index, position])
        ):
            raise ValueError("neighbour CSV differs from the bound NPZ evidence")

    with (run_path / "ranking.csv").open("r", encoding="utf-8", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))
    if len(ranking_rows) != n_samples or {str(row.get("sample_id")) for row in ranking_rows} != set(
        audit_ids
    ):
        raise ValueError("pilot ranking does not contain the exact audit sample set")
    if [int(row["rank"]) for row in ranking_rows] != list(range(1, n_samples + 1)):
        raise ValueError("pilot ranking ranks are not contiguous")
    for row in ranking_rows:
        index = sample_to_index[str(row["sample_id"])]
        if (
            str(row.get("group_id")) != oof_groups[index]
            or int(row["pre_corruption_label"]) != int(oof_pre[index])
            or int(row["observed_label"]) != int(oof_observed[index])
            or (row.get("is_injected_corruption") == "True") != bool(oof_injected[index])
            or not np.isclose(float(row["cleanlab"]), clean_risk[index])
            or not np.isclose(float(row["nearest_neighbour_disagreement"]), neighbour_risk[index])
        ):
            raise ValueError("pilot ranking values differ from per-sample audit evidence")

    return {
        "status": "passed",
        "sample_count": n_samples,
        "sample_order_sha256": _sample_order_sha256(audit_ids),
        "cleanlab": {
            "available": True,
            "package_version": cleanlab_metadata["package_version"],
            "api_path": cleanlab_metadata["api_path"],
            "error": None,
            "issue_count": int(clean_issues.sum()),
            "suggested_labels_available": False,
            "npz_sha256": sha256_file(cleanlab_npz_path),
            "csv_sha256": sha256_file(cleanlab_csv_path),
        },
        "neighbours": {
            "same_group_exclusion_verified": True,
            "query_count": n_samples,
            "edge_count": len(expected_csv_rows),
            "npz_sha256": sha256_file(neighbour_npz_path),
            "csv_sha256": sha256_file(neighbour_csv_path),
        },
    }


def _report_markdown(metrics: Mapping[str, Any], run_id: str) -> str:
    ranking = _mapping(metrics["ranking"], "metrics.ranking")
    downstream = _mapping(metrics["downstream_restoration"], "metrics.downstream")
    random_review = _mapping(metrics["random_review"], "metrics.random_review")
    random_restoration = _mapping(
        downstream["random_review_restoration"], "downstream.random_review_restoration"
    )
    random_restoration_runs = random_restoration.get("runs")
    if not isinstance(random_restoration_runs, Sequence) or isinstance(
        random_restoration_runs, (str, bytes)
    ):
        raise ValueError("downstream random-restoration runs must be a sequence")
    ranking_random_repeats = int(random_review["repeats"])
    downstream_random_repeats = len(random_restoration_runs)
    guided = _mapping(downstream["audit_guided_restoration"], "guided")
    guided_metrics = _mapping(guided["metrics"], "guided.metrics")
    corrupted = _mapping(downstream["corrupted_observed_baseline"], "corrupted")
    corrupted_metrics = _mapping(corrupted["metrics"], "corrupted.metrics")
    lines = [
        "# Real PanNuke controlled-corruption pilot",
        "",
        f"Run ID: `{run_id}`",
        "",
        (
            "This research prototype ranks each potentially inconsistent annotation. "
            "High-ranked items are recommended for expert review. It is not a diagnostic "
            "system, and no source annotation was modified."
        ),
        "",
        "## Fixed split and corruption",
        "",
        f"- Audit-pool nuclei: {metrics['sample_counts']['audit_pool']}",
        f"- Reference-validation nuclei: {metrics['sample_counts']['reference_validation']}",
        f"- Untouched final-reference nuclei: {metrics['sample_counts']['final_reference_test']}",
        f"- Injected corruptions: {metrics['corruption']['exact_count']}",
        "- Grouping unit: source patch; stronger patient/WSI independence is not claimed.",
        "",
        "## Ranking evidence",
        "",
        "| Method | Average precision |",
        "|---|---:|",
    ]
    for method in REQUIRED_AUDIT_METHODS:
        method_metrics = _mapping(ranking[method], f"ranking.{method}")
        lines.append(f"| {method} | {float(method_metrics['average_precision']):.6f} |")
    lines.extend(
        [
            "",
            "## Controlled restoration on reference validation",
            "",
            (
                "The restoration experiment uses the fixed self-confidence ranking and restores "
                "only reviewed injected corruptions. Unreviewed observations remain unchanged. "
                "Downstream pilot metrics are evaluated on the clean development "
                "reference-validation partition."
            ),
            "",
            f"- Ranking-only random-review baseline repeats: {ranking_random_repeats}.",
            f"- Downstream random-restoration model refits: {downstream_random_repeats}.",
            (
                "- Explicit pilot reduction: downstream random restoration uses "
                f"{downstream_random_repeats} refitted models versus "
                f"{ranking_random_repeats} inexpensive ranking-only random-review repeats. "
                "This lower refit count limits pilot precision and is not hidden or treated "
                "as confirmatory evidence."
            ),
            "",
            f"- `corrupted_observed_baseline` macro F1: {float(corrupted_metrics['macro_f1']):.6f}",
            f"- `audit_guided_restoration` macro F1: {float(guided_metrics['macro_f1']):.6f}",
            "",
            (
                "These results measure the declared injected-corruption process and must not be "
                "interpreted as proof that any original annotation or pathologist was wrong."
            ),
            (
                "Official fold 3 remains the untouched final reference: its labels were not used "
                "for pilot outcome evaluation and no representations were extracted for it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_pannuke_pilot(
    gate_certificate_source: str | Path,
    manifest_source: ManifestArtifacts | str | Path,
    *,
    development_manifest_source: str | Path,
    project_root: str | Path,
    expected_data_root: str | Path,
    config: Mapping[str, Any] | str | Path | None = None,
    allow_weight_download: bool = False,
    device: str = "auto",
    duplicate_audit_status: str = "not_supplied",
) -> PanNukePilotResult:
    """Run and seal the fixed all-class PanNuke pilot or fail with evidence."""

    root = Path(project_root).resolve()
    duplicate_prefix = "complete_sha256:"
    if not duplicate_audit_status.startswith(duplicate_prefix):
        raise ValueError(
            "pilot requires a completed full-coverage two-signal duplicate audit with "
            "status complete_sha256:<artifact SHA-256>"
        )
    duplicate_digest = duplicate_audit_status.removeprefix(duplicate_prefix)
    if len(duplicate_digest) != 64 or any(
        character not in "0123456789abcdef" for character in duplicate_digest.lower()
    ):
        raise ValueError("duplicate-audit status contains an invalid SHA-256 digest")
    if config is None:
        resolved = load_config(root / "configs" / "pilot.yaml")
    elif isinstance(config, (str, Path)):
        resolved = load_config(config)
    else:
        resolved = resolve_config(config)
    if resolved.get("experiment_name") != "pannuke_pilot":
        raise ValueError("pilot experiment_name must be pannuke_pilot")
    if resolved.get("configuration_role") != "fixed_pilot_protocol":
        raise ValueError("pilot configuration_role must be fixed_pilot_protocol")
    if "status" in resolved:
        raise ValueError(
            "pilot configuration must not contain a run-status field; terminal status is "
            "recorded by the run tracker"
        )
    gate_certificate_path = Path(gate_certificate_source).expanduser().resolve()
    if not gate_certificate_path.is_file():
        raise FileNotFoundError(
            f"pre-pilot privacy gate certificate is missing: {gate_certificate_path}"
        )
    manifest_path = _manifest_path(manifest_source)
    development_manifest_source_path = Path(development_manifest_source).expanduser().resolve()
    if not development_manifest_source_path.is_file():
        raise FileNotFoundError(
            "pre-pilot development-only manifest view is missing: "
            f"{development_manifest_source_path}"
        )
    data_config = _mapping(resolved.get("data"), "data")
    corruption_config = _mapping(resolved.get("corruption"), "corruption")
    representation_config = _mapping(resolved.get("representation"), "representation")
    model_config = _mapping(resolved.get("model"), "model")
    audit_config = _mapping(resolved.get("audit"), "audit")
    evaluation_config = _mapping(resolved.get("evaluation"), "evaluation")
    if data_config.get("source") != "pannuke":
        raise ValueError("pilot data source must be PanNuke")
    if (
        representation_config.get("encoder") != "official_imagenet_resnet18"
        or representation_config.get("input") != "target_highlighted_rgb"
    ):
        raise ValueError("pilot representation must be official ResNet-18 target-highlighted RGB")
    if representation_config.get("weight_identifier") != "IMAGENET1K_V1":
        raise ValueError("pilot official ResNet-18 weight identifier must be IMAGENET1K_V1")
    if model_config.get("classifier") != "multinomial_logistic_regression":
        raise ValueError("pilot classifier must be multinomial logistic regression")
    if evaluation_config.get("restoration_ranking") != "self_confidence":
        raise ValueError("pilot restoration ranking must be fixed to self_confidence")
    if evaluation_config.get("downstream_evaluation_partition") != "reference_validation":
        raise ValueError("pilot downstream restoration must be evaluated on reference_validation")
    raw_development_folds = data_config.get("development_official_folds")
    if not isinstance(raw_development_folds, Sequence) or isinstance(
        raw_development_folds, (str, bytes)
    ):
        raise ValueError("pilot development_official_folds must be a sequence")
    declared_development_folds = tuple(
        _require_exact_int(value, "data.development_official_folds")
        for value in raw_development_folds
    )
    declared_final_fold = _require_exact_int(
        data_config.get("final_test_fold"), "data.final_test_fold"
    )
    validation, gate_certificate, gate_certificate_sha256 = _load_pilot_gate_certificate(
        gate_certificate_path,
        manifest_path=manifest_path,
        development_manifest_path=development_manifest_source_path,
        expected_duplicate_sha256=duplicate_digest,
        expected_data_root=Path(expected_data_root).expanduser().resolve(),
    )
    manifest_inputs = _load_privacy_partitioned_manifest(
        manifest_path,
        development_manifest_source_path,
        development_folds=declared_development_folds,
        final_fold=declared_final_fold,
    )
    if (
        gate_certificate.get("global_class_free_eligibility") != manifest_inputs.global_provenance
        or gate_certificate.get("development_manifest_view_metadata")
        != manifest_inputs.view_metadata
    ):
        raise ValueError("pre-pilot gate certificate differs from the class-free manifest bindings")
    selection_seed = _require_exact_int(data_config.get("selection_seed"), "data.selection_seed")
    random_review_seed = _require_exact_int(
        evaluation_config.get("random_review_seed"), "evaluation.random_review_seed"
    )
    ranking_random_repeats = _require_exact_int(
        evaluation_config.get("random_review_repeats"), "evaluation.random_review_repeats"
    )
    downstream_random_repeats = _require_exact_int(
        evaluation_config.get("downstream_random_repeats"),
        "evaluation.downstream_random_repeats",
    )
    if ranking_random_repeats < 100:
        raise ValueError("pilot random review requires at least 100 deterministic repetitions")
    if not 0 < downstream_random_repeats < ranking_random_repeats:
        raise ValueError(
            "pilot downstream_random_repeats must be positive and smaller than the "
            "ranking-only random_review_repeats so the declared pilot reduction is true"
        )
    development_folds, final_fold, group_limit, reference_fraction = _validate_gate(
        validation, manifest_inputs.class_free_frame, data_config
    )
    oof_splits = _require_exact_int(model_config.get("oof_splits"), "model.oof_splits")
    selection = _select_groups(
        manifest_inputs.development_frame,
        manifest_inputs.final_metadata_frame,
        development_folds=development_folds,
        final_fold=final_fold,
        development_group_limit=group_limit,
        reference_fraction=reference_fraction,
        selection_seed=selection_seed,
        oof_splits=oof_splits,
    )
    methods = tuple(str(value) for value in audit_config["methods"])
    if methods != REQUIRED_AUDIT_METHODS:
        raise ValueError(f"pilot auditing methods must be exactly {REQUIRED_AUDIT_METHODS}")
    seed_provenance = _validate_seed_provenance(
        resolved,
        model_config=model_config,
        corruption_config=corruption_config,
    )
    split_seed = seed_provenance["split"]
    model_seed = seed_provenance["model"]
    corruption_seed = seed_provenance["corruption"]
    if corruption_seed != 404:
        raise ValueError("pilot corruption seed is fixed at 404")
    if str(corruption_config["mechanism"]) != "symmetric_random_corruption":
        raise ValueError("pilot corruption mechanism must be symmetric_random_corruption")
    if float(corruption_config["rate"]) != 0.10:
        raise ValueError("pilot corruption rate must be exactly 10%")
    audit_frame = selection["audit_frame"]
    reference_frame = selection["reference_frame"]
    final_metadata_frame = selection["final_frame"]
    audit_ids = tuple(audit_frame["sample_id"].astype(str).tolist())
    reference_ids = tuple(reference_frame["sample_id"].astype(str).tolist())
    extraction_ids = (*audit_ids, *reference_ids)
    audit_count = len(audit_ids)
    tracker = RunTracker.start(
        experiment_name="pannuke_pilot",
        config=resolved,
        project_root=root,
        dataset_path=validation.root,
        manifest_path=manifest_path,
        duplicate_audit_status=duplicate_audit_status,
    )
    with tracker:
        run_gate_certificate_path = tracker.run_directory / "pre_pilot_gate_certificate.json"
        atomic_write_bytes(run_gate_certificate_path, gate_certificate_path.read_bytes())
        if sha256_file(run_gate_certificate_path) != gate_certificate_sha256:
            raise RuntimeError("run-local pre-pilot gate certificate differs from its source")
        development_manifest_path = _publish_development_manifest_view(
            tracker, manifest_inputs, development_manifest_source_path
        )
        eligibility_provenance = manifest_inputs.global_provenance
        tracker.log_event(
            "fixed_group_selection_complete",
            selected_development_groups=group_limit,
            audit_groups=len(selection["audit_groups"]),
            reference_validation_groups=len(selection["reference_validation_groups"]),
            final_groups=len(selection["final_reference_groups"]),
            manifest_instances=eligibility_provenance["manifest_instance_count"],
            analysis_eligible_instances=eligibility_provenance["manifest_eligible_instance_count"],
            overlap_excluded_instances=eligibility_provenance["manifest_excluded_instance_count"],
        )
        selected_ids_payload = {
            "selection_seed": selection_seed,
            "selection_policy": data_config["subset_policy"],
            "selected_development_group_limit": group_limit,
            "final_group_limit": None,
            "development_official_folds": development_folds,
            "final_test_fold": final_fold,
            "selected_development_groups": selection["selected_development_groups"],
            "audit_groups": selection["audit_groups"],
            "reference_validation_groups": selection["reference_validation_groups"],
            "final_reference_groups": selection["final_reference_groups"],
            "pairwise_group_overlap_counts": selection["pairwise_group_overlap_counts"],
            "audit_sample_ids": audit_ids,
            "reference_validation_sample_ids": tuple(
                reference_frame["sample_id"].astype(str).tolist()
            ),
            "final_reference_sample_count": len(final_metadata_frame),
            "final_reference_metadata_binding": _final_reference_metadata_binding(
                final_metadata_frame, final_fold=final_fold
            ),
            "final_reference_outcomes_used": False,
            "final_reference_representations_extracted": False,
            "final_reference_sample_ids_read": False,
            "final_reference_class_labels_read": False,
            "no_group_overlap_verified": True,
            "final_fold_complete": len(final_metadata_frame)
            == next(
                item["eligible_instance_count"]
                for item in eligibility_provenance["support_by_official_fold"]
                if item["official_fold"] == final_fold
            ),
            "final_fold_complete_scope": "all analysis-eligible instances in the official fold",
            "analysis_eligibility": eligibility_provenance,
            "pre_pilot_gate_certificate_sha256": gate_certificate_sha256,
        }
        selected_ids_path = tracker.write_json(
            "selected_groups_and_samples.json", selected_ids_payload
        )
        representations = build_pannuke_representation_cache(
            validation,
            development_manifest_path,
            tracker.run_directory / "representations",
            sample_ids=extraction_ids,
            crop_config=PanNukeCropConfig(
                output_size=int(representation_config["crop_output_size"]),
                padding=int(representation_config["crop_padding"]),
                context_brightness=float(representation_config["context_brightness"]),
            ),
            resnet_config=ResNet18EmbeddingConfig(
                weight_identifier=str(representation_config["weight_identifier"]),
                input_variant="target_highlighted_rgb",
                context_brightness=float(representation_config["context_brightness"]),
                device=device,
                batch_size=int(representation_config["batch_size"]),
                allow_weight_download=allow_weight_download,
            ),
        )
        embeddings = _require_official_embedding_provenance(representations, extraction_ids)
        audit_embeddings = embeddings[:audit_count]
        reference_embeddings = embeddings[audit_count:]
        pre_corruption = np.asarray(audit_frame["pre_corruption_label"], dtype=np.int64)
        audit_groups = tuple(audit_frame["group_id"].astype(str).tolist())
        reference_groups = tuple(reference_frame["group_id"].astype(str).tolist())
        corruption = apply_controlled_corruption(
            pre_corruption,
            sample_ids=audit_ids,
            group_ids=audit_groups,
            rate=float(corruption_config["rate"]),
            mechanism=str(corruption_config["mechanism"]),
            seed=corruption_seed,
            n_classes=len(CLASS_ORDER),
            generator_representation=None,
            auditor_representation="official_imagenet_resnet18_target_highlighted_rgb",
            upstream_manifest_hash=sha256_file(manifest_path),
        )
        if set(corruption.observed_labels.tolist()) != set(CLASS_ORDER):
            raise ValueError("the fixed 10% corruption removed a class from observed audit labels")
        oof = grouped_oof_logistic(
            audit_embeddings,
            corruption.observed_labels,
            audit_groups,
            final_reference_group_ids=selection["final_reference_groups"],
            sample_ids=audit_ids,
            fold_assignment_labels=corruption.pre_corruption_labels,
            fold_assignment_label_source="pre_corruption_label",
            n_splits=oof_splits,
            class_order=CLASS_ORDER,
            split_seed=split_seed,
            model_seed=model_seed,
            representation="official_imagenet_resnet18_target_highlighted_rgb",
            l2=float(model_config["l2"]),
            max_iter=int(model_config["max_iter"]),
        )
        self_confidence = score_annotations(
            corruption.observed_labels,
            oof.probabilities,
            method="self_confidence",
            class_order=CLASS_ORDER,
        )
        cleanlab = cleanlab_scores(corruption.observed_labels, oof.probabilities)
        if not cleanlab.available or cleanlab.risk_scores is None:
            raise RuntimeError(f"Cleanlab is mandatory for the pilot but failed: {cleanlab.error}")
        neighbours = fold_safe_neighbour_disagreement(
            audit_embeddings,
            corruption.observed_labels,
            audit_groups,
            oof.fold_id,
            oof.training_groups_by_fold,
            sample_ids=audit_ids,
            class_order=CLASS_ORDER,
            k=int(audit_config["neighbour_k"]),
            metric=str(audit_config["neighbour_metric"]),
        )
        risk_by_method = {
            "self_confidence": self_confidence,
            "cleanlab": cleanlab.risk_scores,
            "nearest_neighbour_disagreement": neighbours.risk_scores,
        }
        review_budgets = tuple(float(value) for value in evaluation_config["review_budgets"])
        ranking: dict[str, Any] = {}
        for method, risks in risk_by_method.items():
            budget_results = {
                str(budget): _review_payload(
                    evaluate_review_budget(
                        corruption.is_injected_corruption,
                        risks,
                        budget=budget,
                        tie_break_ids=audit_ids,
                    )
                )
                for budget in review_budgets
            }
            ranking[method] = {
                "average_precision": budget_results[str(review_budgets[0])]["average_precision"],
                "review_budgets": budget_results,
                "score_min": float(risks.min()),
                "score_median": float(np.median(risks)),
                "score_max": float(risks.max()),
            }
        random_review = random_review_baseline(
            corruption.is_injected_corruption,
            budget=float(evaluation_config["restoration_budget"]),
            repeats=ranking_random_repeats,
            seed=random_review_seed,
        )
        downstream = evaluate_downstream_restoration(
            audit_embeddings,
            corruption.pre_corruption_labels,
            corruption.observed_labels,
            corruption.is_injected_corruption,
            reference_embeddings,
            np.asarray(reference_frame["pre_corruption_label"], dtype=np.int64),
            self_confidence,
            development_group_ids=audit_groups,
            final_test_group_ids=reference_groups,
            final_test_is_injected_corruption=tuple(False for _ in reference_ids),
            review_budget=float(evaluation_config["restoration_budget"]),
            sample_ids=audit_ids,
            class_order=CLASS_ORDER,
            random_repeats=downstream_random_repeats,
            random_seed=random_review_seed,
            model_seed=model_seed,
            l2=float(model_config["l2"]),
            max_iter=int(model_config["max_iter"]),
        )
        downstream_payload = downstream.as_dict()
        downstream_payload["partition_evidence"] = {
            "development_groups": downstream.development_groups,
            "evaluation_partition": "reference_validation",
            "evaluation_groups": downstream.final_reference_groups,
            "evaluation_partition_sample_count": len(reference_frame),
            "evaluation_partition_uncorrupted_verified": (
                downstream.final_test_uncorrupted_verified
            ),
            "final_reference_groups": selection["final_reference_groups"],
            "final_reference_outcomes_used": False,
            "final_reference_representations_extracted": False,
            "group_overlap_count": 0,
        }
        metrics: dict[str, Any] = {
            "artifact_scope": "real_pannuke_controlled_corruption_pilot",
            "completion_stage_if_sealed": "PILOT_COMPLETE",
            "source_annotations_modified": False,
            "diagnostic_claim": False,
            "final_reference_access": {
                "official_fold": final_fold,
                "policy": "metadata_only_until_preregistration_freeze",
                "outcomes_used": False,
                "representations_extracted": False,
                "sample_ids_read": False,
                "class_labels_read": False,
            },
            "sample_counts": {
                "manifest_total_before_qc_exclusion": eligibility_provenance[
                    "manifest_instance_count"
                ],
                "manifest_analysis_eligible": eligibility_provenance[
                    "manifest_eligible_instance_count"
                ],
                "manifest_overlap_touching_excluded": eligibility_provenance[
                    "manifest_excluded_instance_count"
                ],
                "audit_pool": audit_count,
                "reference_validation": len(reference_frame),
                "final_reference_test": len(final_metadata_frame),
                "selected_development_groups": group_limit,
                "audit_groups": len(selection["audit_groups"]),
                "reference_validation_groups": len(selection["reference_validation_groups"]),
                "final_reference_groups": len(selection["final_reference_groups"]),
            },
            "analysis_eligibility": {
                "eligibility_policy": eligibility_provenance["eligibility_policy"],
                "cross_class_overlap_exclusion_reason": eligibility_provenance[
                    "cross_class_overlap_exclusion_reason"
                ],
                "canonical_manifest_sha256": eligibility_provenance["manifest_sha256"],
                "manifest_class_free_eligibility_sha256": eligibility_provenance[
                    "manifest_class_free_eligibility_sha256"
                ],
                "contains_final_reference_sample_ids": False,
                "contains_final_reference_class_labels": False,
                "development_manifest_eligible_sample_ids_sha256": (
                    manifest_inputs.development_provenance["manifest_eligible_sample_ids_sha256"]
                ),
                "development_manifest_excluded_sample_ids_sha256": (
                    manifest_inputs.development_provenance["manifest_excluded_sample_ids_sha256"]
                ),
                "applied_before_group_selection_split_and_model": True,
            },
            "corruption": {
                "mechanism": corruption.mechanism,
                "requested_rate": corruption.requested_rate,
                "exact_count": corruption.exact_count,
                "configuration_hash": corruption.configuration_hash,
                "only_audit_pool_corrupted": True,
                "final_reference_fold_uncorrupted": True,
            },
            "oof": {
                "fold_count": len(oof.folds),
                "complete_once_coverage": bool(np.all(oof.coverage_count == 1)),
                "group_overlap_count": 0,
                "final_reference_groups_excluded": True,
                "probability_sum_maximum_error": float(
                    np.max(np.abs(oof.probabilities.sum(axis=1) - 1.0))
                ),
            },
            "ranking": ranking,
            "cleanlab": {
                "available": cleanlab.available,
                "package_version": cleanlab.package_version,
                "api_path": cleanlab.api_path,
                "error": cleanlab.error,
                "issue_count": int(cleanlab.issue_mask.sum())
                if cleanlab.issue_mask is not None
                else None,
                "suggested_labels_available": False,
                "suggested_labels_status": ("unavailable_api_did_not_return_suggested_labels"),
            },
            "random_review": {
                "reviewed_count": random_review.reviewed_count,
                "repeats": len(random_review.seeds),
                "seeds": random_review.seeds,
                "mean_precision": random_review.mean_precision,
                "mean_recall": random_review.mean_recall,
                "recall_interval_95": random_review.recall_interval_95,
            },
            "pilot_reductions": [
                {
                    "component": "downstream_random_restoration",
                    "status": "declared_pilot_reduction",
                    "ranking_random_review_repeats": len(random_review.seeds),
                    "downstream_random_restoration_repeats": len(
                        downstream.random_review_restoration
                    ),
                    "reason": (
                        "each downstream random-restoration repeat refits the classifier; "
                        "ranking-only random review does not"
                    ),
                    "confirmatory_evidence": False,
                }
            ],
            "downstream_restoration": downstream_payload,
            "representation": representations.embeddings.metadata,
            "limitations": [
                "Controlled injected corruption is not evidence of naturally occurring errors.",
                "Separation is at source-patch level; patient/WSI independence is not claimed.",
                "A high-ranked potentially inconsistent annotation is recommended for expert "
                "review and is not a diagnostic output.",
            ],
        }
        tracker.write_json(
            "corruption_manifest.json",
            {
                "configuration_payload": corruption.configuration_payload_json,
                "rows": corruption.manifest_rows(audit_ids, audit_groups),
            },
        )
        tracker.write_json(
            "oof_provenance.json",
            {
                "class_order": oof.class_order,
                "model_name": oof.model_name,
                "representation": oof.representation,
                "split_seed": oof.split_seed,
                "model_seed": oof.model_seed,
                "splitter_class_name": oof.splitter_class_name,
                "splitter_fallback_status": oof.splitter_fallback_status,
                "splitter_fallback_reason": oof.splitter_fallback_reason,
                "fold_assignment_label_source": oof.fold_assignment_label_source,
                "fold_assignment_labels_sha256": oof.fold_assignment_labels_sha256,
                "folds": [
                    {
                        "fold_id": fold.fold_id,
                        "model_seed": oof.model_seed + fold.fold_id,
                        "training_groups": fold.training_groups,
                        "held_out_groups": fold.held_out_groups,
                        "held_out_sample_ids": fold.held_out_sample_ids,
                    }
                    for fold in oof.folds
                ],
                "final_reference_groups": oof.final_reference_groups,
            },
        )
        atomic_write_npz(
            tracker.run_directory / "oof_predictions.npz",
            {
                "sample_ids": np.asarray(audit_ids, dtype=np.str_),
                "group_ids": np.asarray(audit_groups, dtype=np.str_),
                "pre_corruption_label": corruption.pre_corruption_labels,
                "observed_label": corruption.observed_labels,
                "is_injected_corruption": corruption.is_injected_corruption,
                "probabilities": oof.probabilities,
                "predicted_class": oof.predicted_class,
                "fold_id": oof.fold_id,
                "self_confidence": self_confidence,
                "cleanlab_risk": cleanlab.risk_scores,
                "nearest_neighbour_disagreement": neighbours.risk_scores,
            },
        )
        cleanlab_evidence = _write_cleanlab_audit_evidence(
            tracker.run_directory,
            cleanlab,
            sample_ids=audit_ids,
            group_ids=audit_groups,
            observed_labels=corruption.observed_labels,
        )
        neighbour_evidence = _write_neighbour_audit_evidence(
            tracker.run_directory,
            neighbours,
            sample_ids=audit_ids,
            group_ids=audit_groups,
            observed_labels=corruption.observed_labels,
            fold_ids=oof.fold_id,
            class_order=oof.class_order,
        )
        order = rank_indices(self_confidence, tie_break_ids=audit_ids)
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(
            (
                "rank",
                "sample_id",
                "group_id",
                "pre_corruption_label",
                "observed_label",
                "is_injected_corruption",
                "predicted_class",
                *risk_by_method,
            )
        )
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                (
                    rank,
                    audit_ids[index],
                    audit_groups[index],
                    int(corruption.pre_corruption_labels[index]),
                    int(corruption.observed_labels[index]),
                    bool(corruption.is_injected_corruption[index]),
                    int(oof.predicted_class[index]),
                    *(float(risks[index]) for risks in risk_by_method.values()),
                )
            )
        tracker.write_text("ranking.csv", stream.getvalue())
        audit_evidence_reconciliation = reconcile_pilot_audit_evidence(
            tracker.run_directory,
            require_sealed_integrity=False,
        )
        tracker.write_json("audit_evidence_reconciliation.json", audit_evidence_reconciliation)
        metrics["audit_evidence"] = {
            "reconciliation_status": audit_evidence_reconciliation["status"],
            "sample_order_sha256": audit_evidence_reconciliation["sample_order_sha256"],
            "cleanlab": cleanlab_evidence,
            "neighbours": neighbour_evidence,
            "reconciliation_path": "audit_evidence_reconciliation.json",
        }
        metrics_path = tracker.write_metrics(metrics)
        report_path = tracker.write_text("report.md", _report_markdown(metrics, tracker.run_id))
        tracker.write_provenance(
            split_policy="official folds 1/2 development; complete fold 3 untouched final reference",
            selected_ids_artifact=selected_ids_path.name,
            seed_provenance=seed_provenance,
            official_resnet18_weight_sha256=representations.embeddings.metadata["weight_sha256"],
            automatic_source_annotation_modification=False,
        )
        privacy_reconciliation = _reconcile_pilot_final_reference_privacy(
            tracker.run_directory,
            final_fold=final_fold,
        )
        tracker.write_json(
            "final_reference_privacy_reconciliation.json",
            privacy_reconciliation,
        )
        privacy_reconciliation = _reconcile_pilot_final_reference_privacy(
            tracker.run_directory,
            final_fold=final_fold,
        )
        tracker.write_json(
            "final_reference_privacy_reconciliation.json",
            privacy_reconciliation,
        )
        if (
            _reconcile_pilot_final_reference_privacy(
                tracker.run_directory,
                final_fold=final_fold,
            )
            != privacy_reconciliation
        ):
            raise RuntimeError(
                "pilot privacy reconciliation evidence changed during final readback"
            )

    integrity = verify_run_integrity(tracker.run_directory)
    if not integrity.valid:
        raise RuntimeError(
            f"sealed PanNuke pilot failed integrity verification: {integrity.errors}"
        )
    _reconcile_pilot_final_reference_privacy(
        tracker.run_directory,
        final_fold=final_fold,
    )
    reconcile_pilot_audit_evidence(tracker.run_directory, require_sealed_integrity=True)
    return PanNukePilotResult(
        run_id=tracker.run_id,
        run_directory=tracker.run_directory,
        metrics_path=metrics_path,
        report_path=report_path,
        selected_ids_path=selected_ids_path,
        audit_sample_count=audit_count,
        reference_validation_sample_count=len(reference_frame),
        final_reference_sample_count=len(final_metadata_frame),
        exact_corruption_count=corruption.exact_count,
    )


__all__ = [
    "PanNukePilotDevelopmentManifestView",
    "PanNukePilotResult",
    "build_pannuke_pilot_development_manifest_view",
    "reconcile_pilot_audit_evidence",
    "run_pannuke_pilot",
]
