"""One-use CREATE_NEW publisher and independent CLI verifier for T0.

The scientific/schema module owns every published byte.  This module only
implements the filesystem transaction:

1. rebuild the supplied typed bundle with the live schema builder;
2. claim a never-before-existing directory;
3. write attempt, core artifacts, manifest, and immutable marker with
   same-handle readback;
4. recheck the exact preterminal inventory;
5. write ``publication_success.json`` as the final fallible publication.

Any failure after the directory claim and before success attempts one permanent
STOP write.  No path is removed, replaced, adopted, or retried.  Terminal live
verification is deliberately a separate CLI command/process.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import psutil  # type: ignore[import-untyped]
import typer

from . import original_confirmatory_technical_authority_v1 as authority_schema

CORE_FILENAMES = authority_schema.CORE_FILENAMES
MANIFEST_FILENAME = authority_schema.MANIFEST_FILENAME
IMMUTABLE_MARKER_FILENAME = authority_schema.IMMUTABLE_MARKER_FILENAME
ATTEMPT_FILENAME = authority_schema.ATTEMPT_FILENAME
SUCCESS_FILENAME = authority_schema.SUCCESS_FILENAME
STOP_FILENAME = authority_schema.STOP_FILENAME
QUALIFYING_FILENAMES = authority_schema.QUALIFYING_FILENAMES

PRETERMINAL_FILENAMES = CORE_FILENAMES | {
    MANIFEST_FILENAME,
    IMMUTABLE_MARKER_FILENAME,
    ATTEMPT_FILENAME,
}
AUTHORITY_NAMESPACE_DIRECTORY_NAME = "original_confirmatory_technical_authorities"
NAMESPACE_CLAIM_FILENAME = "original_confirmatory_technical_authority_v1.claim.json"
NAMESPACE_STOP_FILENAME = "original_confirmatory_technical_authority_v1.stop.json"
NAMESPACE_CLAIM_POLICY = "original_confirmatory_technical_authority_namespace_claim_v1"
AUTHORITY_REQUEST_DIRECTORY_NAME = "original_confirmatory_technical_authority_requests"
INTENT_REQUEST_FILENAME = "original_confirmatory_technical_authority_v1.intent.json"
REVIEW_ATTEMPT_FILENAME = "original_confirmatory_technical_authority_v1.review_attempt.json"
REVIEW_REQUEST_FILENAME = "original_confirmatory_technical_authority_v1.review.json"
REVIEW_ATTEMPT_POLICY = "original_confirmatory_technical_authority_review_attempt_v1"
REVIEWER_MODULE_NAME = (
    "histo_audit.workflows.original_confirmatory_technical_authority_review_producer_v1"
)
_UTC_CLOCK_FOR_TESTS_ONLY: Callable[[], datetime] | None = None

_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_STARTF_USESTDHANDLES = 0x00000100
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_HANDLE_FLAG_INHERIT = 0x00000001
_INVALID_DWORD = 0xFFFFFFFF
_WAIT_OBJECT_0 = 0x00000000
_INFINITE = 0xFFFFFFFF
_STILL_ACTIVE = 259
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_HANDLE_EOF = 38
_ERROR_BROKEN_PIPE = 109
_REVIEWER_PIPE_READ_SIZE = 65_536


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", ctypes.c_ulong),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", ctypes.c_int),
    ]


class _StartupInfoW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_ulong),
        ("dwY", ctypes.c_ulong),
        ("dwXSize", ctypes.c_ulong),
        ("dwYSize", ctypes.c_ulong),
        ("dwXCountChars", ctypes.c_ulong),
        ("dwYCountChars", ctypes.c_ulong),
        ("dwFillAttribute", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("wShowWindow", ctypes.c_ushort),
        ("cbReserved2", ctypes.c_ushort),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class _StartupInfoExW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _StartupInfoW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_ulong),
        ("dwThreadId", ctypes.c_ulong),
    ]


class _OwnedWinHandleV1(ctypes.c_void_p):
    """One idempotent owner for a native Windows HANDLE.

    Using this type directly as a ``ctypes`` ``restype`` makes the temporary
    value on the Python evaluation stack own the returned HANDLE.  Therefore an
    asynchronous exception before the next STORE cannot orphan the native
    resource: normal reference destruction closes it.
    """

    def value_int(self) -> int:
        return int(self.value or 0)

    def is_valid(self) -> bool:
        return self.value_int() not in {0, int(ctypes.c_void_p(-1).value or -1)}

    def close(self, *, role: str) -> None:
        handle = self.value_int()
        self.value = None
        if handle in {0, int(ctypes.c_void_p(-1).value or -1)}:
            return
        _close_windows_handle_v1(handle, role=role)

    def close_noexcept(self) -> bool:
        handle = self.value_int()
        self.value = None
        if handle in {0, int(ctypes.c_void_p(-1).value or -1)}:
            return True
        return _close_windows_handle_noexcept_v1(handle)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close_noexcept()
        # Interpreter shutdown and hard process death are outside the
        # recoverable one-crash-cut model.  Never emit from a finalizer.


@dataclass(slots=True)
class _ReviewerRawLaunchHandleOwnerV1:
    """Preallocated owner for every raw Job/stdin/pipe launch HANDLE."""

    job: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)
    stdin: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)
    stdout_read: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)
    stdout_write: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)
    stderr_read: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)
    stderr_write: _OwnedWinHandleV1 = field(default_factory=_OwnedWinHandleV1)

    def require_empty(self) -> None:
        if any(
            handle.is_valid()
            for handle in (
                self.job,
                self.stdin,
                self.stdout_read,
                self.stdout_write,
                self.stderr_read,
                self.stderr_write,
            )
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer raw launch-HANDLE owner was not empty"
            )

    def close_noexcept(self) -> None:
        # The Job is closed last so a suspended child remains contained while
        # all ordinary stdio custody is released.
        for handle in (
            self.stdin,
            self.stdout_read,
            self.stdout_write,
            self.stderr_read,
            self.stderr_write,
            self.job,
        ):
            handle.close_noexcept()


@dataclass(slots=True)
class _ReviewerAttributeInitializationContextV1:
    """Storage retained by the scalar result token returned from WinAPI."""

    buffer: Any
    attribute_list: ctypes.c_void_p
    closed: bool = False


class _OwnedReviewerAttributeInitializationResultV1(ctypes.c_int):
    """RAII BOOL whose truth value says whether Delete is legally required."""

    _context: Any = None

    def succeeded(self) -> bool:
        return bool(self.value)

    def close_noexcept(self) -> None:
        context = self._context
        if context is None or context.closed:
            return
        context.closed = True
        attribute_list = context.attribute_list
        # Retain the backing allocation through Delete; the raw pointer becomes
        # invalid if this is the last strong reference and it is cleared first.
        buffer = context.buffer
        context.attribute_list = ctypes.c_void_p()
        context.buffer = None
        try:
            if self.succeeded():
                _delete_reviewer_attribute_list_noexcept_v1(attribute_list)
        finally:
            del buffer

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close_noexcept()


def _bound_reviewer_attribute_initialization_result_type_v1(
    context: _ReviewerAttributeInitializationContextV1,
) -> type[_OwnedReviewerAttributeInitializationResultV1]:
    """Bind one ctypes scalar result type to exactly one initialization."""

    class _BoundReviewerAttributeInitializationResultV1(
        _OwnedReviewerAttributeInitializationResultV1
    ):
        _context = context

    return _BoundReviewerAttributeInitializationResultV1


@dataclass(slots=True)
class _ReviewerAttributeListOwnerV1:
    """Preallocated adopter for one RAII STARTUPINFOEX result token."""

    initialization: _OwnedReviewerAttributeInitializationResultV1 | None = None
    handle_array: Any = None
    job_array: Any = None

    @property
    def buffer(self) -> Any:
        if self.initialization is None:
            return None
        return self.initialization._context.buffer

    @property
    def attribute_list(self) -> ctypes.c_void_p:
        if self.initialization is None:
            return ctypes.c_void_p()
        context = cast(
            _ReviewerAttributeInitializationContextV1,
            self.initialization._context,
        )
        return context.attribute_list

    @property
    def delete_armed(self) -> bool:
        return (
            self.initialization is not None
            and self.initialization.succeeded()
            and not self.initialization._context.closed
        )

    def adopt(
        self,
        initialization: _OwnedReviewerAttributeInitializationResultV1,
    ) -> None:
        if self.initialization is not None or not initialization.succeeded():
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer attribute-list initialization adoption was invalid"
            )
        # Store the same RAII token first.  A single async cut before this STORE
        # leaves the caller-local token to delete; a cut after it leaves this
        # owner able to delete.  No raw pointer ownership is copied.
        self.initialization = initialization

    def close_noexcept(self) -> None:
        initialization = self.initialization
        self.initialization = None
        self.handle_array = None
        self.job_array = None
        if initialization is not None:
            initialization.close_noexcept()


class OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(RuntimeError):
    """The one-use T0 directory transaction failed closed."""


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryTechnicalAuthorityPublicationV1Result:
    """Identity known before the final success write."""

    authority_directory: Path
    namespace_directory: Path
    namespace_claim_sha256: str
    review_attempt_claim_sha256: str
    artifact_count: int
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    technical_authorization_sha256: str
    immutable_marker_sha256: str
    publication_attempt_sha256: str
    publication_success_sha256: str
    terminal_disposition: str = "success"
    independent_verification_required: bool = True
    automatic_retry_allowed: bool = False
    adoption_allowed: bool = False
    cleanup_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True, slots=True)
class VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1:
    """Terminal schema verification plus the exact singleton namespace claim."""

    authority: authority_schema.VerifiedOriginalConfirmatoryTechnicalAuthority
    namespace_directory: Path
    namespace_claim_sha256: str
    review_attempt_claim_sha256: str

    def lifecycle_binding(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": 1,
            "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
            "namespace_directory": str(self.namespace_directory),
            "namespace_claim_sha256": self.namespace_claim_sha256,
            "review_attempt_claim_sha256": self.review_attempt_claim_sha256,
            "technical_authority": self.authority.lifecycle_binding(),
            "automatic_retry_allowed": False,
            "adoption_allowed": False,
            "cleanup_allowed": False,
        }
        return {
            **unsigned,
            "binding_sha256": authority_schema.canonical_json_sha256(unsigned),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "authority": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.authority).items()
            },
            "namespace_directory": str(self.namespace_directory),
            "namespace_claim_sha256": self.namespace_claim_sha256,
            "review_attempt_claim_sha256": self.review_attempt_claim_sha256,
            "lifecycle_binding": self.lifecycle_binding(),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_json_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token is forbidden: {token}")


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _strict_json_line_bytes(payload: bytes, *, role: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be one non-empty canonical JSON line"
        )
    if b"\n" in payload[:-1]:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be one canonical JSON line"
        )
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} is not strict finite UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or authority_schema.canonical_json_line_bytes(value) != payload:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} differs from canonical JSON"
        )
    return value


def _require_exact_builder_bundle(
    value: Any,
) -> authority_schema.OriginalConfirmatoryTechnicalAuthorityBundle:
    """Rebuild through the live schema and compare every byte and identity."""

    bundle_type = authority_schema.OriginalConfirmatoryTechnicalAuthorityBundle
    if type(value) is not bundle_type:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "publisher requires the exact typed bundle returned by the live T0 builder"
        )
    bundle = value
    try:
        artifacts = dict(bundle.artifacts)
    except (TypeError, ValueError) as exc:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "typed bundle artifacts are not one finite mapping"
        ) from exc
    if set(artifacts) != CORE_FILENAMES or any(
        type(name) is not str or type(payload) is not bytes or not payload
        for name, payload in artifacts.items()
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "typed bundle core artifact inventory is not exact"
        )
    try:
        intent = _strict_json_line_bytes(
            artifacts[authority_schema.INTENT_FILENAME],
            role="technical authority intent",
        )
        review = _strict_json_line_bytes(
            artifacts[authority_schema.REVIEW_FILENAME],
            role="independent review",
        )
        evidence = _strict_json_line_bytes(
            artifacts[authority_schema.EVIDENCE_FILENAME],
            role="technical authority evidence",
        )
        source_inventory = _strict_json_line_bytes(
            artifacts[authority_schema.SOURCE_INVENTORY_FILENAME],
            role="execution source inventory",
        )
        rebuilt = authority_schema.build_original_confirmatory_technical_authority_bundle_v1(
            authority_directory=bundle.authority_directory,
            intent=intent,
            independent_review=review,
            publication_timestamp_utc=evidence["publication_timestamp_utc"],
            preregistration_bytes=artifacts[authority_schema.PREREGISTRATION_FILENAME],
            primary_config_bytes=artifacts[authority_schema.PRIMARY_CONFIG_FILENAME],
            confirmatory_config_bytes=artifacts[authority_schema.CONFIRMATORY_CONFIG_FILENAME],
            source_inventory=source_inventory,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        authority_schema.OriginalConfirmatoryTechnicalAuthorityError,
    ) as exc:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "typed bundle cannot be reproduced by the live T0 builder"
        ) from exc
    fields_match = (
        rebuilt.authority_directory == bundle.authority_directory
        and dict(rebuilt.artifacts) == artifacts
        and rebuilt.sha256_manifest_bytes == bundle.sha256_manifest_bytes
        and rebuilt.immutable_marker_bytes == bundle.immutable_marker_bytes
        and rebuilt.publication_attempt_bytes == bundle.publication_attempt_bytes
        and rebuilt.publication_success_bytes == bundle.publication_success_bytes
        and rebuilt.publication_stop_bytes == bundle.publication_stop_bytes
        and rebuilt.artifact_root_sha256 == bundle.artifact_root_sha256
        and rebuilt.sha256_manifest_sha256 == bundle.sha256_manifest_sha256
        and rebuilt.technical_authorization_sha256 == bundle.technical_authorization_sha256
    )
    if not fields_match:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "typed bundle differs byte-for-byte from the live T0 builder"
        )
    return rebuilt


def _namespace_claim_bytes(
    *,
    authority_directory: Path,
    intent: Mapping[str, Any],
    publication_timestamp_utc: str,
    artifact_root_sha256: str,
    sha256_manifest_sha256: str,
    technical_authorization_sha256: str,
    review_attempt_claim_sha256: str,
) -> bytes:
    source = intent["execution_source"]
    capsule = intent["execution_capsule"]
    capacity = intent["capacity_v2"]
    unsigned = {
        "schema_version": 1,
        "policy": NAMESPACE_CLAIM_POLICY,
        "namespace_directory": str(authority_directory.parent),
        "authority_directory": str(authority_directory),
        "publication_timestamp_utc": publication_timestamp_utc,
        "intent_root_sha256": intent["intent_root_sha256"],
        "execution_source_manifest_sha256": source["manifest_sha256"],
        "execution_source_root_sha256": source["root_sha256"],
        "capsule_sha256": capsule["sha256"],
        "capsule_internal_manifest_sha256": capsule["internal_manifest_sha256"],
        "capsule_source_records_root_sha256": capsule["source_records_root_sha256"],
        "capsule_publication_receipt_sha256": capsule["publication_receipt_sha256"],
        "capacity_receipt_sha256": capacity["receipt_sha256"],
        "artifact_root_sha256": artifact_root_sha256,
        "sha256_manifest_sha256": sha256_manifest_sha256,
        "technical_authorization_sha256": technical_authorization_sha256,
        "review_attempt_claim_sha256": review_attempt_claim_sha256,
        "creation_disposition": "CREATE_NEW",
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    return authority_schema.canonical_json_line_bytes(
        {
            **unsigned,
            "claim_root_sha256": authority_schema.canonical_json_sha256(unsigned),
        }
    )


def _namespace_claim_bytes_from_bundle(
    bundle: authority_schema.OriginalConfirmatoryTechnicalAuthorityBundle,
) -> bytes:
    intent = _strict_json_line_bytes(
        bundle.artifacts[authority_schema.INTENT_FILENAME],
        role="technical authority intent",
    )
    evidence = _strict_json_line_bytes(
        bundle.artifacts[authority_schema.EVIDENCE_FILENAME],
        role="technical authority evidence",
    )
    project_root = _authority_namespace_from_intent(intent).parents[1]
    request_anchor = _open_authority_request_namespace(
        project_root,
        create=False,
    )
    try:
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="namespace-claim construction",
        )
    finally:
        request_anchor.close_noexcept()
    request_directory = project_root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME
    if (
        _stable_regular_file_bytes(
            request_directory / INTENT_REQUEST_FILENAME,
            role="request intent for namespace publication",
        )
        != bundle.artifacts[authority_schema.INTENT_FILENAME]
        or _stable_regular_file_bytes(
            request_directory / REVIEW_REQUEST_FILENAME,
            role="request review for namespace publication",
        )
        != bundle.artifacts[authority_schema.REVIEW_FILENAME]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "request intent/review differ from the typed T0 bundle"
        )
    attempt_path = request_directory / REVIEW_ATTEMPT_FILENAME
    verified_attempt = verify_original_confirmatory_technical_review_attempt_claim_v1(
        attempt_path,
        intent=intent,
        project_root=project_root,
    )
    review = authority_schema.canonical_original_confirmatory_technical_authority_review_v1(
        _strict_json_line_bytes(
            bundle.artifacts[authority_schema.REVIEW_FILENAME],
            role="typed bundle independent review",
        ),
        intent=intent,
    )
    if authority_schema._utc(
        verified_attempt["attempt_created_at_utc"],
        role="review-attempt creation",
    ) > authority_schema._utc(
        review["review_started_at_utc"],
        role="review start",
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "typed review predates its permanent attempt claim"
        )
    review_attempt_claim_sha256 = _sha256_bytes(
        _stable_regular_file_bytes(
            attempt_path,
            role="review-attempt claim for namespace publication",
        )
    )
    return _namespace_claim_bytes(
        authority_directory=bundle.authority_directory,
        intent=intent,
        publication_timestamp_utc=evidence["publication_timestamp_utc"],
        artifact_root_sha256=bundle.artifact_root_sha256,
        sha256_manifest_sha256=bundle.sha256_manifest_sha256,
        technical_authorization_sha256=bundle.technical_authorization_sha256,
        review_attempt_claim_sha256=review_attempt_claim_sha256,
    )


def _authority_namespace_from_intent(intent: Mapping[str, Any]) -> Path:
    parent = Path(intent["parent"]["authority_directory"])
    try:
        project_root = parent.parents[2]
    except IndexError as exc:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "parent P cannot derive the live project root"
        ) from exc
    expected_parent = (
        project_root / "artifacts" / "preregistration_amendments" / "20260727T133947.089370Z"
    )
    if parent != expected_parent:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "parent P does not derive the exact live project root"
        )
    return project_root / "artifacts" / AUTHORITY_NAMESPACE_DIRECTORY_NAME


def _is_reparse(path: Path, value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & reparse)


def _is_read_only(value: os.stat_result) -> bool:
    return not bool(value.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _windows_named_data_streams(path: Path) -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    class Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(Win32FindStreamData),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Win32FindStreamData),
    ]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    data = Win32FindStreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error in {1, 38}:
            return ()
        raise OSError(error, f"cannot enumerate named streams: {path}")
    streams: list[str] = []
    try:
        while True:
            stream = str(data.stream_name)
            if stream != "::$DATA":
                streams.append(stream)
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise OSError(error, f"named-stream enumeration failed: {path}")
    finally:
        find_close(handle)
    return tuple(sorted(streams))


def _windows_final_path_from_fd(descriptor: int) -> str | None:
    if os.name != "nt":
        return None
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    get_final_path.restype = ctypes.c_uint32
    handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
    required = get_final_path(handle, None, 0, 0)
    if required == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = get_final_path(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _flush_descriptor(descriptor: int) -> None:
    if os.name != "nt":
        os.fsync(descriptor)
        return
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
    if not flush(handle):
        raise ctypes.WinError(ctypes.get_last_error())


@dataclass(slots=True)
class _RetainedDirectoryAnchor:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    final_path: str | None

    @classmethod
    def open(
        cls,
        raw: Path,
        *,
        write_access: bool = True,
    ) -> _RetainedDirectoryAnchor:
        path = raw.absolute()
        if os.name == "nt":
            import msvcrt

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
                ((0x40000000 if write_access else 0) | 0x00000080 | 0x00100000),
                0x00000001 | 0x00000002,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle),
                    os.O_RDONLY | getattr(os, "O_NOINHERIT", 0),
                )
            except BaseException:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
                raise
        else:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        try:
            opened = os.fstat(descriptor)
            logical = path.stat(follow_symlinks=False)
            final_path = _windows_final_path_from_fd(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(logical.st_mode)
                or _is_reparse(path, logical)
                or (opened.st_dev, opened.st_ino) != (logical.st_dev, logical.st_ino)
                or _windows_named_data_streams(path)
                or (
                    final_path is not None
                    and os.path.normcase(os.path.normpath(final_path))
                    != os.path.normcase(os.path.normpath(str(path)))
                )
            ):
                raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                    f"directory anchor is linked, streamed, replaced, or invalid: {path}"
                )
            return cls(
                path=path,
                descriptor=descriptor,
                identity=(opened.st_dev, opened.st_ino),
                final_path=final_path,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def assert_current(self) -> None:
        opened = os.fstat(self.descriptor)
        logical = self.path.stat(follow_symlinks=False)
        final_path = _windows_final_path_from_fd(self.descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(logical.st_mode)
            or _is_reparse(self.path, logical)
            or (opened.st_dev, opened.st_ino) != self.identity
            or (logical.st_dev, logical.st_ino) != self.identity
            or final_path != self.final_path
            or _windows_named_data_streams(self.path)
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                f"retained directory anchor changed: {self.path}"
            )

    def flush_and_assert(self) -> None:
        self.assert_current()
        _flush_descriptor(self.descriptor)
        self.assert_current()

    def close_noexcept(self) -> None:
        with suppress(OSError):
            os.close(self.descriptor)


def _require_safe_new_directory_path(raw: Path, *, expected_namespace: Path) -> Path:
    if not raw.is_absolute() or raw != Path(os.path.abspath(raw)):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "authority directory must be one canonical absolute path"
        )
    parent = raw.parent.resolve(strict=True)
    parent_value = raw.parent.stat(follow_symlinks=False)
    if (
        _is_reparse(raw.parent, parent_value)
        or not stat.S_ISDIR(parent_value.st_mode)
        or parent != raw.parent
        or raw.parent != expected_namespace
        or raw.parent.name != AUTHORITY_NAMESPACE_DIRECTORY_NAME
        or raw.parent.parent.name != "artifacts"
        or raw.name in {"", ".", ".."}
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "authority destination must be a direct child of the canonical "
            "artifacts/original_confirmatory_technical_authorities namespace"
        )
    if _lexists(raw):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"STOP: authority directory already exists; adoption/retry forbidden: {raw}"
        )
    return raw


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("CREATE_NEW write made no progress")
        offset += written


def _read_exact_from_start(descriptor: int, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    observed = b"".join(chunks)
    if os.read(descriptor, 1):
        raise OSError("same-handle readback exceeded expected size")
    return observed


def _create_new_descriptor(path: Path) -> int:
    if os.name != "nt":
        return os.open(
            path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IREAD,
        )
    import msvcrt

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
        0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
        0x00000001,  # FILE_SHARE_READ; deny write and delete until terminal
        None,
        1,  # CREATE_NEW
        0x00000001 | 0x80000000,  # READONLY | WRITE_THROUGH
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            int(handle),
            os.O_RDWR | os.O_BINARY | getattr(os, "O_NOINHERIT", 0),
        )
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise


def _open_existing_descriptor(path: Path) -> int:
    if os.name != "nt":
        return os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    import msvcrt

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
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ; deny write and delete while verified
        None,
        3,  # OPEN_EXISTING
        0x00200000,  # OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(
            int(handle),
            os.O_RDONLY | os.O_BINARY | getattr(os, "O_NOINHERIT", 0),
        )
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise


@dataclass(slots=True)
class _RetainedPublishedFile:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    final_path: str | None
    payload: bytes
    sha256: str

    @classmethod
    def open_existing(cls, path: Path) -> _RetainedPublishedFile:
        descriptor: int | None = None
        try:
            descriptor = _open_existing_descriptor(path)
            opened = os.fstat(descriptor)
            payload = _read_exact_from_start(descriptor, opened.st_size)
            retained = cls(
                path=path,
                descriptor=descriptor,
                identity=(opened.st_dev, opened.st_ino),
                final_path=_windows_final_path_from_fd(descriptor),
                payload=payload,
                sha256=_sha256_bytes(payload),
            )
            retained.assert_current()
            descriptor = None
            return retained
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @classmethod
    def create(
        cls,
        path: Path,
        payload: bytes,
        *,
        on_created: Callable[[], None] | None = None,
    ) -> _RetainedPublishedFile:
        if type(payload) is not bytes or not payload:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                f"CREATE_NEW payload must be exact non-empty bytes: {path.name}"
            )
        descriptor: int | None = None
        try:
            descriptor = _create_new_descriptor(path)
            if on_created is not None:
                on_created()
            _write_all(descriptor, payload)
            _flush_descriptor(descriptor)
            opened = os.fstat(descriptor)
            retained = cls(
                path=path,
                descriptor=descriptor,
                identity=(opened.st_dev, opened.st_ino),
                final_path=_windows_final_path_from_fd(descriptor),
                payload=payload,
                sha256=_sha256_bytes(payload),
            )
            retained.assert_current()
            descriptor = None
            return retained
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def assert_current(self) -> None:
        opened = os.fstat(self.descriptor)
        logical = self.path.stat(follow_symlinks=False)
        observed = _read_exact_from_start(self.descriptor, len(self.payload))
        final_path = _windows_final_path_from_fd(self.descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(logical.st_mode)
            or _is_reparse(self.path, logical)
            or opened.st_nlink != 1
            or logical.st_nlink != 1
            or not _is_read_only(logical)
            or opened.st_size != len(self.payload)
            or logical.st_size != len(self.payload)
            or (opened.st_dev, opened.st_ino) != self.identity
            or (logical.st_dev, logical.st_ino) != self.identity
            or final_path != self.final_path
            or _windows_named_data_streams(self.path)
            or observed != self.payload
            or _sha256_bytes(observed) != self.sha256
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                f"retained CREATE_NEW handle differs: {self.path}"
            )

    def close_noexcept(self) -> None:
        with suppress(OSError):
            os.close(self.descriptor)


def _create_new_file_no_cleanup(
    path: Path,
    payload: bytes,
    *,
    on_created: Callable[[], None] | None = None,
) -> _RetainedPublishedFile:
    return _RetainedPublishedFile.create(
        path,
        payload,
        on_created=on_created,
    )


def _require_exact_file(path: Path, payload: bytes, expected_sha256: str) -> None:
    value = path.stat(follow_symlinks=False)
    observed = path.read_bytes()
    if (
        _is_reparse(path, value)
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or not _is_read_only(value)
        or value.st_size != len(payload)
        or _windows_named_data_streams(path)
        or observed != payload
        or _sha256_bytes(observed) != expected_sha256
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"retained publication file differs: {path}"
        )


def _create_and_require(
    path: Path,
    payload: bytes,
    *,
    on_created: Callable[[], None] | None = None,
) -> _RetainedPublishedFile:
    retained = (
        _create_new_file_no_cleanup(path, payload)
        if on_created is None
        else _create_new_file_no_cleanup(
            path,
            payload,
            on_created=on_created,
        )
    )
    retained.assert_current()
    return retained


def _write_stop_if_possible(
    directory: Path,
    filename: str,
    payload: bytes,
) -> _RetainedPublishedFile | None:
    path = directory / filename
    if _lexists(path):
        _require_exact_file(path, payload, _sha256_bytes(payload))
        return None
    return _create_and_require(path, payload)


def publish_original_confirmatory_technical_authority_v1_once(
    bundle: authority_schema.OriginalConfirmatoryTechnicalAuthorityBundle,
) -> OriginalConfirmatoryTechnicalAuthorityPublicationV1Result:
    """Publish one exact live-builder bundle without adoption, cleanup, or retry."""

    exact = _require_exact_builder_bundle(bundle)
    exact_intent = _strict_json_line_bytes(
        exact.artifacts[authority_schema.INTENT_FILENAME],
        role="technical authority intent",
    )
    destination = _require_safe_new_directory_path(
        exact.authority_directory,
        expected_namespace=_authority_namespace_from_intent(exact_intent),
    )
    namespace = destination.parent
    claim_bytes = _namespace_claim_bytes_from_bundle(exact)
    claim_sha256 = _sha256_bytes(claim_bytes)
    review_attempt_claim_sha256 = _strict_json_line_bytes(
        claim_bytes,
        role="permanent T0 namespace claim",
    )["review_attempt_claim_sha256"]
    artifacts = dict(exact.artifacts)
    manifest_sha256 = _sha256_bytes(exact.sha256_manifest_bytes)
    marker_sha256 = _sha256_bytes(exact.immutable_marker_bytes)
    attempt_sha256 = _sha256_bytes(exact.publication_attempt_bytes)
    success_sha256 = _sha256_bytes(exact.publication_success_bytes)
    result = OriginalConfirmatoryTechnicalAuthorityPublicationV1Result(
        authority_directory=destination,
        namespace_directory=namespace,
        namespace_claim_sha256=claim_sha256,
        review_attempt_claim_sha256=review_attempt_claim_sha256,
        artifact_count=len(artifacts),
        artifact_root_sha256=exact.artifact_root_sha256,
        sha256_manifest_sha256=manifest_sha256,
        technical_authorization_sha256=exact.technical_authorization_sha256,
        immutable_marker_sha256=marker_sha256,
        publication_attempt_sha256=attempt_sha256,
        publication_success_sha256=success_sha256,
    )
    claim_created = False
    directory_created = False
    success_created = False
    namespace_anchor: _RetainedDirectoryAnchor | None = None
    authority_anchor: _RetainedDirectoryAnchor | None = None
    retained_files: list[_RetainedPublishedFile] = []

    def require_namespace() -> _RetainedDirectoryAnchor:
        if namespace_anchor is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "publication namespace anchor is unavailable"
            )
        namespace_anchor.assert_current()
        return namespace_anchor

    def require_anchors() -> tuple[
        _RetainedDirectoryAnchor,
        _RetainedDirectoryAnchor,
    ]:
        current_namespace = require_namespace()
        if authority_anchor is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "publication authority anchor is unavailable"
            )
        authority_anchor.assert_current()
        return current_namespace, authority_anchor

    def retain_created(
        path: Path,
        payload: bytes,
        *,
        child: bool,
        on_created: Callable[[], None] | None = None,
    ) -> _RetainedPublishedFile:
        if child:
            require_anchors()
        else:
            require_namespace()
        retained = _create_and_require(
            path,
            payload,
            on_created=on_created,
        )
        retained_files.append(retained)
        if child:
            require_anchors()
        else:
            require_namespace()
        return retained

    def create_child(path: Path, payload: bytes) -> _RetainedPublishedFile:
        require_anchors()
        return retain_created(path, payload, child=True)

    try:
        namespace_anchor = _RetainedDirectoryAnchor.open(namespace)
        namespace_anchor.assert_current()
        if any(namespace.iterdir()):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "STOP: global T0 namespace is not pristine; a second authority "
                "or adoption is forbidden"
            )

        def mark_claim_created() -> None:
            nonlocal claim_created
            claim_created = True

        retain_created(
            namespace / NAMESPACE_CLAIM_FILENAME,
            claim_bytes,
            child=False,
            on_created=mark_claim_created,
        )
        namespace_anchor.flush_and_assert()

        os.mkdir(destination)
        directory_created = True
        namespace_anchor.flush_and_assert()
        authority_anchor = _RetainedDirectoryAnchor.open(destination)
        require_anchors()
        create_child(
            destination / ATTEMPT_FILENAME,
            exact.publication_attempt_bytes,
        )
        authority_anchor.flush_and_assert()
        for name, payload in sorted(artifacts.items()):
            create_child(destination / name, payload)
        create_child(
            destination / MANIFEST_FILENAME,
            exact.sha256_manifest_bytes,
        )
        create_child(
            destination / IMMUTABLE_MARKER_FILENAME,
            exact.immutable_marker_bytes,
        )
        authority_anchor.flush_and_assert()

        require_anchors()
        if {path.name for path in destination.iterdir()} != PRETERMINAL_FILENAMES:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "preterminal T0 inventory is not exact"
            )
        if {path.name for path in namespace.iterdir()} != {
            NAMESPACE_CLAIM_FILENAME,
            destination.name,
        }:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "global T0 namespace changed before terminal success"
            )
        for retained in retained_files:
            retained.assert_current()
        require_anchors()

        # CREATE_NEW, file flush, directory flush, and retained-handle readback
        # together form the final fallible publication.  After they return,
        # only in-memory assignment/return and noexcept handle closes remain.
        success_file = create_child(
            destination / SUCCESS_FILENAME,
            exact.publication_success_bytes,
        )
        authority_anchor.flush_and_assert()
        success_file.assert_current()
        success_created = True
        return result
    except BaseException as exc:
        child_stop_sha256: str | None = None
        namespace_stop_sha256: str | None = None
        stop_errors: list[str] = []
        if claim_created and not success_created:
            if directory_created:
                try:
                    if authority_anchor is None:
                        authority_anchor = _RetainedDirectoryAnchor.open(destination)
                    require_anchors()
                    child_stop = _write_stop_if_possible(
                        destination,
                        STOP_FILENAME,
                        exact.publication_stop_bytes,
                    )
                    if child_stop is not None:
                        retained_files.append(child_stop)
                    child_stop_sha256 = _sha256_bytes(exact.publication_stop_bytes)
                    authority_anchor.flush_and_assert()
                except BaseException as nested:
                    stop_errors.append(f"child_STOP_failed={type(nested).__name__}: {nested}")
            try:
                current_namespace = require_namespace()
                namespace_stop = _write_stop_if_possible(
                    namespace,
                    NAMESPACE_STOP_FILENAME,
                    exact.publication_stop_bytes,
                )
                if namespace_stop is not None:
                    retained_files.append(namespace_stop)
                namespace_stop_sha256 = _sha256_bytes(exact.publication_stop_bytes)
                current_namespace.flush_and_assert()
            except BaseException as nested:
                stop_errors.append(f"namespace_STOP_failed={type(nested).__name__}: {nested}")
        stop_detail = "; ".join(
            [
                f"durable_child_stop_sha256={child_stop_sha256}",
                f"durable_namespace_stop_sha256={namespace_stop_sha256}",
                *stop_errors,
            ]
        )
        if not stop_detail:
            stop_detail = "no STOP written because the global CREATE_NEW claim was not acquired"
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "STOP: T0 publication failed; retain all state; adoption, cleanup, "
            f"and retry are forbidden: {type(exc).__name__}: {exc}; {stop_detail}"
        ) from exc
    finally:
        for retained in reversed(retained_files):
            retained.close_noexcept()
        if authority_anchor is not None:
            authority_anchor.close_noexcept()
        if namespace_anchor is not None:
            namespace_anchor.close_noexcept()


def _stable_regular_file_bytes(path: Path, *, role: str) -> bytes:
    supplied = path.stat(follow_symlinks=False)
    if _is_reparse(path, supplied) or not stat.S_ISREG(supplied.st_mode):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be one canonical regular non-link file"
        )
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be one canonical regular non-link file"
        )
    logical = canonical.stat(follow_symlinks=False)
    if (
        _is_reparse(canonical, logical)
        or not stat.S_ISREG(logical.st_mode)
        or logical.st_nlink != 1
        or _windows_named_data_streams(canonical)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be one regular non-link file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    try:
        opened_before = os.fstat(descriptor)
        payload = _read_exact_from_start(descriptor, opened_before.st_size)
        opened_after = os.fstat(descriptor)
        logical_after = canonical.stat(follow_symlinks=False)
        final_path = _windows_final_path_from_fd(descriptor)
        stable_fields_before = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_mode,
            opened_before.st_nlink,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
        )
        stable_fields_after = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_mode,
            opened_after.st_nlink,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
        )
        if (
            (supplied.st_dev, supplied.st_ino) != (opened_before.st_dev, opened_before.st_ino)
            or not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or stable_fields_before != stable_fields_after
            or opened_before.st_size != len(payload)
            or (opened_before.st_dev, opened_before.st_ino)
            != (logical_after.st_dev, logical_after.st_ino)
            or _is_reparse(canonical, logical_after)
            or _windows_named_data_streams(canonical)
            or (
                final_path is not None
                and os.path.normcase(os.path.normpath(final_path))
                != os.path.normcase(os.path.normpath(str(canonical)))
            )
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                f"{role} changed during same-handle readback"
            )
        return payload
    finally:
        os.close(descriptor)


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    supplied = path if path.is_absolute() else project_root / path
    return Path(os.path.abspath(supplied))


def _require_authority_request_leaf(
    *,
    project_root: Path,
    path: Path,
    expected_filename: str,
    role: str,
) -> Path:
    expected_parent = project_root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME
    if path.parent != expected_parent or path.name != expected_filename:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} must be the fixed {expected_filename} child of {expected_parent}"
        )
    return path


def _open_authority_request_namespace(
    project_root: Path,
    *,
    create: bool,
) -> _RetainedDirectoryAnchor:
    artifacts = project_root / "artifacts"
    artifacts_anchor = _RetainedDirectoryAnchor.open(artifacts)
    request_directory = artifacts / AUTHORITY_REQUEST_DIRECTORY_NAME
    try:
        artifacts_anchor.assert_current()
        if not _lexists(request_directory):
            if not create:
                raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                    "fixed technical-authority request namespace does not exist"
                )
            with suppress(FileExistsError):
                os.mkdir(request_directory)
            artifacts_anchor.flush_and_assert()
        request_anchor = _RetainedDirectoryAnchor.open(request_directory)
        request_anchor.assert_current()
        return request_anchor
    finally:
        artifacts_anchor.close_noexcept()


def _require_authority_request_inventory(
    request_anchor: _RetainedDirectoryAnchor,
    *,
    expected_names: frozenset[str],
    phase: str,
) -> None:
    request_anchor.assert_current()
    children = sorted(request_anchor.path.iterdir(), key=lambda item: item.name)
    observed_names = frozenset(path.name for path in children)
    if observed_names != expected_names:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{phase} request inventory is not exact: "
            f"expected={sorted(expected_names)}, observed={sorted(observed_names)}"
        )
    for path in children:
        value = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(value.st_mode)
            or _is_reparse(path, value)
            or value.st_nlink != 1
            or not _is_read_only(value)
            or _windows_named_data_streams(path)
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                f"{phase} request leaf is linked, streamed, writable, or invalid: {path}"
            )
    request_anchor.assert_current()


@dataclass(slots=True)
class _RetainedReviewMutex:
    handle: int

    @classmethod
    def acquire(cls, project_root: Path) -> _RetainedReviewMutex:
        if os.name != "nt":
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "fresh-child review mutex requires Windows"
            )
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        create_mutex.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        name = "Local\\AANCA-original-confirmatory-T0-review-" + _sha256_bytes(
            str(project_root).casefold().encode("utf-8")
        )
        ctypes.set_last_error(0)
        handle = create_mutex(None, 1, name)
        if handle in {None, 0, ctypes.c_void_p(-1).value}:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            close_handle(ctypes.c_void_p(handle))
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "one fresh-child review controller is already active"
            )
        return cls(handle=int(handle))

    def close_noexcept(self) -> None:
        if not self.handle:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        release_mutex = kernel32.ReleaseMutex
        release_mutex.argtypes = [ctypes.c_void_p]
        release_mutex.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        with suppress(Exception):
            release_mutex(ctypes.c_void_p(self.handle))
        with suppress(Exception):
            close_handle(ctypes.c_void_p(self.handle))
        self.handle = 0


def _read_json_input(path: Path, *, role: str) -> dict[str, Any]:
    return _strict_json_line_bytes(
        _stable_regular_file_bytes(path, role=role),
        role=role,
    )


def _canonical_utc_now() -> str:
    clock = _UTC_CLOCK_FOR_TESTS_ONLY
    value = datetime.now(UTC) if clock is None else clock()
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "internal UTC clock returned a non-UTC timestamp"
        )
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _capture_process_identity_v1(
    process_id: int,
    implementation_path: str | Path,
) -> dict[str, Any]:
    """Capture one live process plus its fixed implementation identity."""

    process = psutil.Process(process_id)
    executable = Path(process.exe()).resolve(strict=True)
    implementation = Path(implementation_path)
    if not implementation.is_absolute() or implementation != Path(os.path.abspath(implementation)):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "producer implementation path must be canonical and absolute"
        )
    implementation = implementation.resolve(strict=True)
    executable_bytes = _stable_regular_file_bytes(
        executable,
        role="producer executable",
    )
    implementation_bytes = _stable_regular_file_bytes(
        implementation,
        role="producer implementation",
    )
    created = (
        datetime.fromtimestamp(process.create_time(), tz=UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    return {
        "process_id": process.pid,
        "process_created_at_utc": created,
        "executable_path": str(executable),
        "executable_size_bytes": len(executable_bytes),
        "executable_sha256": _sha256_bytes(executable_bytes),
        "implementation_path": str(implementation),
        "implementation_sha256": _sha256_bytes(implementation_bytes),
    }


def capture_current_process_identity_v1(
    implementation_path: str | Path,
) -> dict[str, Any]:
    """Capture non-caller-declared identity for the current producer process."""

    return _capture_process_identity_v1(os.getpid(), implementation_path)


def _canonical_review_attempt_claim_v1(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    canonical_intent = (
        authority_schema.canonical_original_confirmatory_technical_authority_intent_v1(intent)
    )
    fields = {
        "schema_version",
        "policy",
        "request_namespace",
        "review_attempt_path",
        "intent_path",
        "intent_file_sha256",
        "intent_root_sha256",
        "review_output_path",
        "attempt_created_at_utc",
        "controller_process",
        "reviewer_implementation_path",
        "reviewer_implementation_size_bytes",
        "reviewer_implementation_sha256",
        "creation_disposition",
        "attempt_count",
        "max_attempt_count",
        "outcome_values_read",
        "scientific_execution_performed",
        "automatic_retry_allowed",
        "adoption_allowed",
        "cleanup_allowed",
        "overwrite_allowed",
        "attempt_root_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "review-attempt claim fields are not exact"
        )
    request_namespace = project_root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME
    intent_path = request_namespace / INTENT_REQUEST_FILENAME
    attempt_path = request_namespace / REVIEW_ATTEMPT_FILENAME
    review_path = request_namespace / REVIEW_REQUEST_FILENAME
    controller = authority_schema._process(
        value["controller_process"],
        role="review-attempt controller",
    )
    created = authority_schema._utc(
        value["attempt_created_at_utc"],
        role="review-attempt creation",
    )
    reviewer_path = Path(value["reviewer_implementation_path"])
    reviewer_spec = importlib.util.find_spec(REVIEWER_MODULE_NAME)
    if reviewer_spec is None or reviewer_spec.origin is None:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "review-attempt reviewer implementation is unavailable"
        )
    expected_reviewer_path = Path(reviewer_spec.origin).resolve(strict=True)
    reviewer_bytes = _stable_regular_file_bytes(
        expected_reviewer_path,
        role="review-attempt reviewer implementation",
    )
    controller_implementation = Path(__file__).resolve(strict=True)
    controller_bytes = _stable_regular_file_bytes(
        controller_implementation,
        role="review-attempt controller implementation",
    )
    intent_bytes = authority_schema.canonical_json_line_bytes(canonical_intent)
    unsigned = {
        "schema_version": 1,
        "policy": REVIEW_ATTEMPT_POLICY,
        "request_namespace": str(request_namespace),
        "review_attempt_path": str(attempt_path),
        "intent_path": str(intent_path),
        "intent_file_sha256": _sha256_bytes(intent_bytes),
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "review_output_path": str(review_path),
        "attempt_created_at_utc": value["attempt_created_at_utc"],
        "controller_process": controller,
        "reviewer_implementation_path": str(expected_reviewer_path),
        "reviewer_implementation_size_bytes": len(reviewer_bytes),
        "reviewer_implementation_sha256": _sha256_bytes(reviewer_bytes),
        "creation_disposition": "CREATE_NEW",
        "attempt_count": 1,
        "max_attempt_count": 1,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    expected = {
        **unsigned,
        "attempt_root_sha256": authority_schema.canonical_json_sha256(unsigned),
    }
    if (
        authority_schema.canonical_json_line_bytes(value)
        != authority_schema.canonical_json_line_bytes(expected)
        or reviewer_path != expected_reviewer_path
        or controller["implementation_path"] != str(controller_implementation)
        or controller["implementation_sha256"] != _sha256_bytes(controller_bytes)
        or controller["implementation_path"] == str(expected_reviewer_path)
        or controller["implementation_sha256"] == _sha256_bytes(reviewer_bytes)
        or authority_schema._utc(
            controller["process_created_at_utc"],
            role="review-attempt controller process creation",
        )
        > created
        or _stable_regular_file_bytes(intent_path, role="review-attempt intent") != intent_bytes
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "review-attempt claim is contradictory or not fixed"
        )
    authority_schema._verify_process_files(
        controller,
        role="review-attempt controller",
    )
    return expected


def _build_original_confirmatory_technical_review_attempt_claim_at_v1(
    *,
    intent: Mapping[str, Any],
    project_root: str | Path,
    controller_process: Mapping[str, Any],
    reviewer_implementation_path: str | Path,
    attempt_created_at_utc: str,
) -> dict[str, Any]:
    """Build the one fixed pre-CreateProcessW review-attempt claim."""

    root = Path(project_root).resolve(strict=True)
    reviewer_path = Path(reviewer_implementation_path).resolve(strict=True)
    reviewer_bytes = _stable_regular_file_bytes(
        reviewer_path,
        role="review-attempt reviewer implementation",
    )
    canonical_intent = (
        authority_schema.canonical_original_confirmatory_technical_authority_intent_v1(intent)
    )
    request_namespace = root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME
    unsigned = {
        "schema_version": 1,
        "policy": REVIEW_ATTEMPT_POLICY,
        "request_namespace": str(request_namespace),
        "review_attempt_path": str(request_namespace / REVIEW_ATTEMPT_FILENAME),
        "intent_path": str(request_namespace / INTENT_REQUEST_FILENAME),
        "intent_file_sha256": _sha256_bytes(
            authority_schema.canonical_json_line_bytes(canonical_intent)
        ),
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "review_output_path": str(request_namespace / REVIEW_REQUEST_FILENAME),
        "attempt_created_at_utc": attempt_created_at_utc,
        "controller_process": dict(controller_process),
        "reviewer_implementation_path": str(reviewer_path),
        "reviewer_implementation_size_bytes": len(reviewer_bytes),
        "reviewer_implementation_sha256": _sha256_bytes(reviewer_bytes),
        "creation_disposition": "CREATE_NEW",
        "attempt_count": 1,
        "max_attempt_count": 1,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    candidate = {
        **unsigned,
        "attempt_root_sha256": authority_schema.canonical_json_sha256(unsigned),
    }
    return _canonical_review_attempt_claim_v1(
        candidate,
        intent=canonical_intent,
        project_root=root,
    )


def build_original_confirmatory_technical_review_attempt_claim_v1(
    *,
    intent: Mapping[str, Any],
    project_root: str | Path,
    controller_process: Mapping[str, Any],
    reviewer_implementation_path: str | Path,
) -> dict[str, Any]:
    """Build the one fixed claim with a non-caller-declared UTC timestamp."""

    return _build_original_confirmatory_technical_review_attempt_claim_at_v1(
        intent=intent,
        project_root=project_root,
        controller_process=controller_process,
        reviewer_implementation_path=reviewer_implementation_path,
        attempt_created_at_utc=_canonical_utc_now(),
    )


def verify_original_confirmatory_technical_review_attempt_claim_v1(
    claim_path: str | Path,
    *,
    intent: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Read and fail-closed verify the permanent fixed review-attempt claim."""

    root = Path(project_root).resolve(strict=True)
    expected_path = root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME / REVIEW_ATTEMPT_FILENAME
    path = Path(claim_path)
    if path != expected_path:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "review-attempt claim path is not fixed"
        )
    value = _strict_json_line_bytes(
        _stable_regular_file_bytes(path, role="review-attempt claim"),
        role="review-attempt claim",
    )
    return _canonical_review_attempt_claim_v1(
        value,
        intent=intent,
        project_root=root,
    )


def _close_windows_handle_v1(handle: int, *, role: str) -> None:
    if not handle:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    ctypes.set_last_error(0)
    if not close_handle(ctypes.c_void_p(handle)):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} CloseHandle failed: {ctypes.WinError(ctypes.get_last_error())}"
        )


def _close_windows_handle_noexcept_v1(handle: int) -> bool:
    try:
        _close_windows_handle_v1(handle, role="reviewer custody")
    except Exception:
        return False
    return True


def _require_non_inheritable_windows_handle_v1(handle: int, *, role: str) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_handle_information = kernel32.GetHandleInformation
    get_handle_information.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    get_handle_information.restype = ctypes.c_int
    flags = ctypes.c_ulong()
    ctypes.set_last_error(0)
    if not get_handle_information(ctypes.c_void_p(handle), ctypes.byref(flags)):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} handle inheritance query failed: {ctypes.WinError(ctypes.get_last_error())}"
        )
    if flags.value & _HANDLE_FLAG_INHERIT:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"{role} handle is inheritable"
        )


def _create_reviewer_kill_job_v1(
    *,
    owner: _ReviewerRawLaunchHandleOwnerV1,
) -> None:
    if os.name != "nt":
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "fresh-child reviewer Job Object custody requires Windows"
        )
    owner.require_empty()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    create_job.restype = _OwnedWinHandleV1
    set_job_information = kernel32.SetInformationJobObject
    set_job_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    set_job_information.restype = ctypes.c_int
    query_job_information = kernel32.QueryInformationJobObject
    query_job_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    query_job_information.restype = ctypes.c_int

    try:
        ctypes.set_last_error(0)
        raw_handle = create_job(None, None)
        if not raw_handle.is_valid():
            raw_handle.close_noexcept()
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "CREATE_NEW unnamed reviewer Job Object failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        # Store the same owning wrapper; never copy its numeric HANDLE into a
        # second owner.  Its finalizer covers the native CALL-to-STORE cut.
        owner.job = raw_handle
        handle = owner.job.value_int()
        _set_windows_handle_inheritance_v1(handle, inheritable=False)
        _require_non_inheritable_windows_handle_v1(
            handle,
            role="reviewer Job Object",
        )
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_ACTIVE_PROCESS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        ctypes.set_last_error(0)
        if not set_job_information(
            ctypes.c_void_p(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer Job Object KILL_ON_JOB_CLOSE configuration failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        observed = _JobObjectExtendedLimitInformation()
        returned_length = ctypes.c_ulong()
        ctypes.set_last_error(0)
        if not query_job_information(
            ctypes.c_void_p(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(observed),
            ctypes.sizeof(observed),
            ctypes.byref(returned_length),
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer Job Object limit readback failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        flags = int(observed.BasicLimitInformation.LimitFlags)
        required = _JOB_OBJECT_LIMIT_ACTIVE_PROCESS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        forbidden = _JOB_OBJECT_LIMIT_BREAKAWAY_OK | _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        if (
            returned_length.value != ctypes.sizeof(observed)
            or flags & required != required
            or flags & forbidden
            or observed.BasicLimitInformation.ActiveProcessLimit != 1
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer Job Object custody limits did not read back exactly"
            )
        return
    except BaseException:
        owner.job.close_noexcept()
        raise


def _resume_reviewer_initial_thread_v1(thread_handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    resume_thread = kernel32.ResumeThread
    resume_thread.argtypes = [ctypes.c_void_p]
    resume_thread.restype = ctypes.c_ulong
    ctypes.set_last_error(0)
    previous_suspend_count = int(resume_thread(ctypes.c_void_p(thread_handle)))
    if previous_suspend_count != 1:
        detail = (
            str(ctypes.WinError(ctypes.get_last_error()))
            if previous_suspend_count == _INVALID_DWORD
            else f"unexpected previous suspend count {previous_suspend_count}"
        )
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"suspended reviewer initial-thread resume failed: {detail}"
        )


def _set_windows_handle_inheritance_v1(handle: int, *, inheritable: bool) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_handle_information = kernel32.SetHandleInformation
    set_handle_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    set_handle_information.restype = ctypes.c_int
    ctypes.set_last_error(0)
    if not set_handle_information(
        ctypes.c_void_p(handle),
        _HANDLE_FLAG_INHERIT,
        _HANDLE_FLAG_INHERIT if inheritable else 0,
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer Windows handle inheritance configuration failed: "
            f"{ctypes.WinError(ctypes.get_last_error())}"
        )


def _create_reviewer_pipe_v1(
    *,
    owner: _ReviewerRawLaunchHandleOwnerV1,
    stream: str,
) -> None:
    if stream == "stdout":
        read_handle = owner.stdout_read
        write_handle = owner.stdout_write
    elif stream == "stderr":
        read_handle = owner.stderr_read
        write_handle = owner.stderr_write
    else:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer pipe stream is not exact"
        )
    if read_handle.is_valid() or write_handle.is_valid():
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"reviewer {stream} pipe owner was not empty"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_pipe = kernel32.CreatePipe
    create_pipe.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_SecurityAttributes),
        ctypes.c_ulong,
    ]
    create_pipe.restype = ctypes.c_int
    security = _SecurityAttributes(
        nLength=ctypes.sizeof(_SecurityAttributes),
        lpSecurityDescriptor=None,
        bInheritHandle=1,
    )
    ctypes.set_last_error(0)
    if not create_pipe(
        ctypes.byref(read_handle),
        ctypes.byref(write_handle),
        ctypes.byref(security),
        0,
    ):
        read_handle.close_noexcept()
        write_handle.close_noexcept()
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"reviewer stdio pipe creation failed: {ctypes.WinError(ctypes.get_last_error())}"
        )
    read_value = read_handle.value_int()
    try:
        _set_windows_handle_inheritance_v1(read_value, inheritable=False)
        _require_non_inheritable_windows_handle_v1(
            read_value,
            role="reviewer parent pipe",
        )
        return
    except BaseException:
        read_handle.close_noexcept()
        write_handle.close_noexcept()
        raise


def _create_reviewer_null_input_v1(
    *,
    owner: _ReviewerRawLaunchHandleOwnerV1,
) -> None:
    if owner.stdin.is_valid():
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer NUL stdin owner was not empty"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(_SecurityAttributes),
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    create_file.restype = _OwnedWinHandleV1
    security = _SecurityAttributes(
        nLength=ctypes.sizeof(_SecurityAttributes),
        lpSecurityDescriptor=None,
        bInheritHandle=1,
    )
    ctypes.set_last_error(0)
    try:
        raw_handle = create_file(
            "NUL",
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            ctypes.byref(security),
            _OPEN_EXISTING,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if not raw_handle.is_valid():
            raw_handle.close_noexcept()
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer NUL stdin handle creation failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        # Preserve the same RAII object across the creator return boundary.
        owner.stdin = raw_handle
    except BaseException:
        owner.stdin.close_noexcept()
        raise


def _canonical_windows_environment_buffer_v1(
    environment: Mapping[str, str],
) -> Any:
    records: list[tuple[str, str]] = []
    folded: set[str] = set()
    for key, value in environment.items():
        if (
            type(key) is not str
            or type(value) is not str
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            or key.casefold() in folded
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer Unicode environment is not canonical"
            )
        folded.add(key.casefold())
        records.append((key, value))
    records.sort(key=lambda item: (item[0].casefold(), item[0]))
    # create_unicode_buffer adds the second terminal NUL required by CreateProcessW.
    text = "\0".join(f"{key}={value}" for key, value in records) + "\0"
    return ctypes.create_unicode_buffer(text)


def _initialize_reviewer_attribute_list_call_v1(
    attribute_list: ctypes.c_void_p | None,
    size: ctypes.c_size_t,
) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    initialize = kernel32.InitializeProcThreadAttributeList
    initialize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    initialize.restype = ctypes.c_int
    ctypes.set_last_error(0)
    return bool(initialize(attribute_list, 2, 0, ctypes.byref(size)))


def _initialize_reviewer_attribute_list_owned_call_v1(
    *,
    attribute_list: ctypes.c_void_p,
    size: ctypes.c_size_t,
    buffer: Any,
) -> _OwnedReviewerAttributeInitializationResultV1:
    """Initialize with a success-aware RAII token on the native return stack."""

    context = _ReviewerAttributeInitializationContextV1(
        buffer=buffer,
        attribute_list=attribute_list,
    )
    result_type = _bound_reviewer_attribute_initialization_result_type_v1(context)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    initialize = kernel32.InitializeProcThreadAttributeList
    initialize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    initialize.restype = result_type
    ctypes.set_last_error(0)
    return cast(
        _OwnedReviewerAttributeInitializationResultV1,
        initialize(attribute_list, 2, 0, ctypes.byref(size)),
    )


def _build_reviewer_attribute_value_arrays_v1(
    *,
    job_handle: int,
    inherited_handles: tuple[int, int, int],
) -> tuple[Any, Any]:
    return (
        (ctypes.c_void_p * len(inherited_handles))(*inherited_handles),
        (ctypes.c_void_p * 1)(job_handle),
    )


def _create_reviewer_attribute_list_v1(
    *,
    job_handle: int,
    inherited_handles: tuple[int, int, int],
    owner: _ReviewerAttributeListOwnerV1,
) -> None:
    if (
        owner.delete_armed
        or owner.attribute_list
        or owner.buffer is not None
        or owner.handle_array is not None
        or owner.job_array is not None
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer attribute-list owner was not empty before initialization"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    update = kernel32.UpdateProcThreadAttribute
    update.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    update.restype = ctypes.c_int
    size = ctypes.c_size_t()
    sizing_succeeded = _initialize_reviewer_attribute_list_call_v1(None, size)
    sizing_error = ctypes.get_last_error()
    if sizing_succeeded or sizing_error != _ERROR_INSUFFICIENT_BUFFER or not size.value:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer STARTUPINFOEX attribute-list sizing failed exact contract: "
            f"result={sizing_succeeded}; size={size.value}; "
            f"error={ctypes.WinError(sizing_error)}"
        )
    buffer = ctypes.create_string_buffer(size.value)
    attribute_list = ctypes.cast(buffer, ctypes.c_void_p)
    try:
        initialization = _initialize_reviewer_attribute_list_owned_call_v1(
            attribute_list=attribute_list,
            size=size,
            buffer=buffer,
        )
        if not initialization.succeeded():
            initialization.close_noexcept()
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer STARTUPINFOEX attribute-list initialization failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        owner.adopt(initialization)
        (
            owner.handle_array,
            owner.job_array,
        ) = _build_reviewer_attribute_value_arrays_v1(
            job_handle=job_handle,
            inherited_handles=inherited_handles,
        )
        ctypes.set_last_error(0)
        if not update(
            attribute_list,
            0,
            _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.cast(owner.handle_array, ctypes.c_void_p),
            ctypes.sizeof(owner.handle_array),
            None,
            None,
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer exact HANDLE_LIST installation failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        ctypes.set_last_error(0)
        if not update(
            attribute_list,
            0,
            _PROC_THREAD_ATTRIBUTE_JOB_LIST,
            ctypes.cast(owner.job_array, ctypes.c_void_p),
            ctypes.sizeof(owner.job_array),
            None,
            None,
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer atomic JOB_LIST assignment installation failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
    except BaseException:
        owner.close_noexcept()
        raise


def _delete_reviewer_attribute_list_noexcept_v1(
    attribute_list: ctypes.c_void_p,
) -> None:
    if not attribute_list:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    delete = kernel32.DeleteProcThreadAttributeList
    delete.argtypes = [ctypes.c_void_p]
    delete.restype = None
    with suppress(Exception):
        delete(attribute_list)


def _create_atomic_job_bound_process_v1(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    job_handle: int,
    stdin_handle: int,
    stdout_handle: int,
    stderr_handle: int,
    process_information: _ProcessInformation,
    attribute_owner: _ReviewerAttributeListOwnerV1,
) -> None:
    if (
        not argv
        or not Path(argv[0]).is_absolute()
        or Path(argv[0]) != Path(os.path.abspath(argv[0]))
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer application path is not exact and absolute"
        )
    if any(
        (
            process_information.hProcess,
            process_information.hThread,
            process_information.dwProcessId,
            process_information.dwThreadId,
        )
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer PROCESS_INFORMATION owner was not empty before CreateProcessW"
        )
    try:
        _create_reviewer_attribute_list_v1(
            job_handle=job_handle,
            inherited_handles=(stdin_handle, stdout_handle, stderr_handle),
            owner=attribute_owner,
        )
        attribute_list = attribute_owner.attribute_list
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
        environment_buffer = _canonical_windows_environment_buffer_v1(environment)
        startup = _StartupInfoExW()
        startup.StartupInfo.cb = ctypes.sizeof(_StartupInfoExW)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = ctypes.c_void_p(stdin_handle)
        startup.StartupInfo.hStdOutput = ctypes.c_void_p(stdout_handle)
        startup.StartupInfo.hStdError = ctypes.c_void_p(stderr_handle)
        startup.lpAttributeList = attribute_list
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_process = kernel32.CreateProcessW
        create_process.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.POINTER(_StartupInfoW),
            ctypes.POINTER(_ProcessInformation),
        ]
        create_process.restype = ctypes.c_int
        ctypes.set_last_error(0)
        if not create_process(
            argv[0],
            command_line,
            None,
            None,
            1,
            (_CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT | _EXTENDED_STARTUPINFO_PRESENT),
            environment_buffer,
            str(cwd),
            ctypes.byref(startup.StartupInfo),
            ctypes.byref(process_information),
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "atomic CREATE_SUSPENDED reviewer creation/JOB_LIST assignment failed: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
    finally:
        attribute_owner.close_noexcept()


def _require_atomic_reviewer_process_identity_v1(
    process_information: _ProcessInformation,
    *,
    job_handle: int,
) -> None:
    process_handle = int(process_information.hProcess or 0)
    thread_handle = int(process_information.hThread or 0)
    _require_non_inheritable_windows_handle_v1(
        process_handle,
        role="reviewer process",
    )
    _require_non_inheritable_windows_handle_v1(
        thread_handle,
        role="reviewer initial thread",
    )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_process_id = kernel32.GetProcessId
    get_process_id.argtypes = [ctypes.c_void_p]
    get_process_id.restype = ctypes.c_ulong
    get_thread_id = kernel32.GetThreadId
    get_thread_id.argtypes = [ctypes.c_void_p]
    get_thread_id.restype = ctypes.c_ulong
    get_process_id_of_thread = kernel32.GetProcessIdOfThread
    get_process_id_of_thread.argtypes = [ctypes.c_void_p]
    get_process_id_of_thread.restype = ctypes.c_ulong
    is_process_in_job = kernel32.IsProcessInJob
    is_process_in_job.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    is_process_in_job.restype = ctypes.c_int
    if (
        int(get_process_id(ctypes.c_void_p(process_handle))) != int(process_information.dwProcessId)
        or int(get_thread_id(ctypes.c_void_p(thread_handle))) != int(process_information.dwThreadId)
        or int(get_process_id_of_thread(ctypes.c_void_p(thread_handle)))
        != int(process_information.dwProcessId)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "atomic reviewer retained process/thread identity is contradictory"
        )
    in_job = ctypes.c_int()
    ctypes.set_last_error(0)
    if (
        not is_process_in_job(
            ctypes.c_void_p(process_handle),
            ctypes.c_void_p(job_handle),
            ctypes.byref(in_job),
        )
        or not in_job.value
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "atomic reviewer JOB_LIST membership readback failed: "
            f"{ctypes.WinError(ctypes.get_last_error())}"
        )


def _wait_for_reviewer_process_v1(process_handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    wait = kernel32.WaitForSingleObject
    wait.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    wait.restype = ctypes.c_ulong
    ctypes.set_last_error(0)
    result = int(wait(ctypes.c_void_p(process_handle), _INFINITE))
    if result != _WAIT_OBJECT_0:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer WaitForExit failed: "
            f"result={result}; error={ctypes.WinError(ctypes.get_last_error())}"
        )


def _reviewer_process_exit_code_v1(process_handle: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    get_exit_code.restype = ctypes.c_int
    exit_code = ctypes.c_ulong()
    ctypes.set_last_error(0)
    if not get_exit_code(ctypes.c_void_p(process_handle), ctypes.byref(exit_code)):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"reviewer exit-code readback failed: {ctypes.WinError(ctypes.get_last_error())}"
        )
    if exit_code.value == _STILL_ACTIVE:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer remains active after WaitForExit"
        )
    return int(exit_code.value)


def _terminate_reviewer_job_noexcept_v1(job_handle: int) -> None:
    if not job_handle:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    terminate_job = kernel32.TerminateJobObject
    terminate_job.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    terminate_job.restype = ctypes.c_int
    with suppress(Exception):
        terminate_job(ctypes.c_void_p(job_handle), 1)


def _terminate_reviewer_process_noexcept_v1(process_handle: int) -> None:
    if not process_handle:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    terminate_process = kernel32.TerminateProcess
    terminate_process.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    terminate_process.restype = ctypes.c_int
    with suppress(Exception):
        terminate_process(ctypes.c_void_p(process_handle), 1)


@dataclass(slots=True)
class _ReviewerPipeCaptureV1:
    handle: _OwnedWinHandleV1
    chunks: list[bytes] = field(default_factory=list)
    error: BaseException | None = None
    thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer pipe drain was started twice"
            )
        self.thread = threading.Thread(
            target=self._drain,
            name="AANCA-T0-reviewer-pipe-drain",
            daemon=False,
        )
        self.thread.start()

    def _drain(self) -> None:
        try:
            while chunk := _read_reviewer_pipe_chunk_v1(self.handle.value_int()):
                self.chunks.append(chunk)
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.handle.close(role="reviewer parent pipe")
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def finish(self) -> bytes:
        if self.thread is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer pipe drain was never started"
            )
        self.thread.join()
        if self.error is not None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer pipe drain failed"
            ) from self.error
        return b"".join(self.chunks)

    def close_noexcept(self) -> None:
        # Thread.start may raise after ``self.thread`` is assigned but before
        # the native thread exists.  Such an object cannot be joined and still
        # owns the raw pipe HANDLE in this controller thread.
        if self.thread is None or self.thread.ident is None:
            self.handle.close_noexcept()
            return
        with suppress(Exception):
            self.finish()


def _read_reviewer_pipe_chunk_v1(handle: int) -> bytes:
    if not handle:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer pipe read has no retained handle"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    read_file = kernel32.ReadFile
    read_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    ]
    read_file.restype = ctypes.c_int
    buffer = (ctypes.c_ubyte * _REVIEWER_PIPE_READ_SIZE)()
    count = ctypes.c_ulong()
    ctypes.set_last_error(0)
    if not read_file(
        ctypes.c_void_p(handle),
        ctypes.byref(buffer),
        _REVIEWER_PIPE_READ_SIZE,
        ctypes.byref(count),
        None,
    ):
        error = ctypes.get_last_error()
        if error in {_ERROR_HANDLE_EOF, _ERROR_BROKEN_PIPE}:
            return b""
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            f"reviewer native pipe read failed: {ctypes.WinError(error)}"
        )
    return bytes(buffer[: count.value])


def _reviewer_pipe_capture_from_handle_v1(
    handle: _OwnedWinHandleV1,
) -> _ReviewerPipeCaptureV1:
    if not handle.is_valid():
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer pipe capture cannot consume an invalid handle"
        )
    return _ReviewerPipeCaptureV1(handle=handle)


@dataclass(slots=True)
class _RetainedReviewerChildV1:
    process_handle: int
    thread_handle: int
    job_handle: _OwnedWinHandleV1
    pid: int
    initial_thread_id: int
    stdout_capture: _ReviewerPipeCaptureV1
    stderr_capture: _ReviewerPipeCaptureV1
    returncode: int | None = None
    resumed: bool = False
    waited: bool = False
    custody_closed: bool = False

    def resume_exactly_once(self) -> None:
        if self.resumed or self.waited or self.custody_closed:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer initial thread cannot be resumed in its current state"
            )
        _resume_reviewer_initial_thread_v1(self.thread_handle)
        self.resumed = True

    def communicate(self) -> tuple[bytes, bytes]:
        if not self.resumed or self.waited or self.custody_closed:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer WaitForExit cannot run in its current state"
            )
        _wait_for_reviewer_process_v1(self.process_handle)
        self.waited = True
        self.returncode = _reviewer_process_exit_code_v1(self.process_handle)
        stdout = self.stdout_capture.finish()
        stderr = self.stderr_capture.finish()
        return stdout, stderr

    def close_after_wait(self) -> None:
        if self.custody_closed:
            return
        if not self.waited or self.returncode is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer custody cannot close before WaitForExit"
            )
        handles = [
            ("reviewer initial thread", self.thread_handle),
            ("reviewer process", self.process_handle),
        ]
        self.thread_handle = 0
        self.process_handle = 0
        self.custody_closed = True
        errors: list[str] = []
        for role, handle in handles:
            try:
                _close_windows_handle_v1(handle, role=role)
            except Exception as exc:
                errors.append(f"{role}: {type(exc).__name__}: {exc}")
        try:
            self.job_handle.close(role="reviewer Job Object")
        except Exception as exc:
            errors.append(f"reviewer Job Object: {type(exc).__name__}: {exc}")
        if errors:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer custody handle close failed after WaitForExit: " + "; ".join(errors)
            )

    def close_job_then_wait_noexcept(self) -> None:
        if self.custody_closed:
            return
        job_handle = self.job_handle.value_int()
        job_closed = self.job_handle.close_noexcept()
        if not job_closed:
            _terminate_reviewer_job_noexcept_v1(job_handle)
            _terminate_reviewer_process_noexcept_v1(self.process_handle)
        with suppress(Exception):
            _wait_for_reviewer_process_v1(self.process_handle)
            self.waited = True
            self.returncode = _reviewer_process_exit_code_v1(self.process_handle)
        with suppress(Exception):
            self.stdout_capture.finish()
        with suppress(Exception):
            self.stderr_capture.finish()
        _close_windows_handle_noexcept_v1(self.thread_handle)
        _close_windows_handle_noexcept_v1(self.process_handle)
        self.thread_handle = 0
        self.process_handle = 0
        self.custody_closed = True


@dataclass(slots=True)
class _RetainedReviewerChildOwnerV1:
    """Preallocated caller owner spanning the launch CALL/STORE boundary."""

    child: _RetainedReviewerChildV1 | None = None

    def close_noexcept(self) -> None:
        child = self.child
        self.child = None
        if child is not None:
            child.close_job_then_wait_noexcept()


def _launch_retained_reviewer_child_v1(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    child_owner: _RetainedReviewerChildOwnerV1,
) -> None:
    """Atomically create one suspended reviewer inside a controller-only Job."""

    if os.name != "nt":
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "fresh-child reviewer atomic Job Object custody requires Windows"
        )
    if child_owner.child is not None:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "reviewer child owner was not empty before launch"
        )
    raw_handles = _ReviewerRawLaunchHandleOwnerV1()
    process_handle = 0
    thread_handle = 0
    process_information = _ProcessInformation()
    attribute_owner = _ReviewerAttributeListOwnerV1()
    stdout_capture: _ReviewerPipeCaptureV1 | None = None
    stderr_capture: _ReviewerPipeCaptureV1 | None = None
    try:
        _create_reviewer_kill_job_v1(owner=raw_handles)
        _create_reviewer_null_input_v1(owner=raw_handles)
        _create_reviewer_pipe_v1(owner=raw_handles, stream="stdout")
        _create_reviewer_pipe_v1(owner=raw_handles, stream="stderr")
        _create_atomic_job_bound_process_v1(
            argv,
            cwd=cwd,
            environment=env,
            job_handle=raw_handles.job.value_int(),
            stdin_handle=raw_handles.stdin.value_int(),
            stdout_handle=raw_handles.stdout_write.value_int(),
            stderr_handle=raw_handles.stderr_write.value_int(),
            process_information=process_information,
            attribute_owner=attribute_owner,
        )
        process_handle = int(process_information.hProcess or 0)
        thread_handle = int(process_information.hThread or 0)
        raw_handles.stdin.close(role="reviewer child stdin")
        raw_handles.stdout_write.close(role="reviewer child stdout")
        raw_handles.stderr_write.close(role="reviewer child stderr")
        _require_atomic_reviewer_process_identity_v1(
            process_information,
            job_handle=raw_handles.job.value_int(),
        )
        process_id = int(process_information.dwProcessId)
        initial_thread_id = int(process_information.dwThreadId)
        process_information.hProcess = None
        process_information.hThread = None
        stdout_capture = _reviewer_pipe_capture_from_handle_v1(raw_handles.stdout_read)
        stderr_capture = _reviewer_pipe_capture_from_handle_v1(raw_handles.stderr_read)
        stdout_capture.start()
        stderr_capture.start()
        child_owner.child = _RetainedReviewerChildV1(
            process_handle=process_handle,
            thread_handle=thread_handle,
            job_handle=raw_handles.job,
            pid=process_id,
            initial_thread_id=initial_thread_id,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
        )
        process_handle = 0
        thread_handle = 0
        stdout_capture = None
        stderr_capture = None
    except BaseException as exc:
        attribute_owner.close_noexcept()
        if child_owner.child is not None:
            process_handle = 0
            thread_handle = 0
            stdout_capture = None
            stderr_capture = None
            child_owner.close_noexcept()
        if not process_handle:
            process_handle = int(process_information.hProcess or 0)
        if not thread_handle:
            thread_handle = int(process_information.hThread or 0)
        process_information.hProcess = None
        process_information.hThread = None
        _terminate_reviewer_job_noexcept_v1(raw_handles.job.value_int())
        _terminate_reviewer_process_noexcept_v1(process_handle)
        if process_handle:
            with suppress(Exception):
                _wait_for_reviewer_process_v1(process_handle)
        if stdout_capture is not None:
            stdout_capture.close_noexcept()
        if stderr_capture is not None:
            stderr_capture.close_noexcept()
        raw_handles.close_noexcept()
        for handle in (thread_handle, process_handle):
            _close_windows_handle_noexcept_v1(handle)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "STOP: atomic one-shot reviewer creation/Job custody failed; "
            "attempt remains permanent and retry/adoption are forbidden: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _capture_and_wait_for_reviewer_child_v1(
    child: _RetainedReviewerChildV1,
    reviewer_implementation: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Authenticate one job-custodied child and retain handles through WaitForExit."""

    try:
        identity = _capture_process_identity_v1(
            child.pid,
            reviewer_implementation,
        )
        child.resume_exactly_once()
        # This is the sole authorized reviewer.  Wait on its retained process
        # handle without a timeout so the controller cannot abandon an
        # authenticated child that might later CREATE_NEW its receipt.
        stdout, stderr = child.communicate()
        child.close_after_wait()
        return identity, stdout, stderr
    except BaseException:
        # Closing the controller-only Job handle kills the reviewer (and any
        # possible descendant) before this retained process handle is waited.
        child.close_job_then_wait_noexcept()
        raise


def _launch_capture_and_wait_for_reviewer_v1(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    reviewer_implementation: Path,
) -> tuple[_RetainedReviewerChildV1, dict[str, Any], bytes, bytes]:
    """Launch and enter retained custody without an unguarded caller gap."""

    child_owner = _RetainedReviewerChildOwnerV1()
    child: _RetainedReviewerChildV1 | None = None
    completed = False
    try:
        _launch_retained_reviewer_child_v1(
            argv,
            cwd=cwd,
            env=env,
            child_owner=child_owner,
        )
        child = child_owner.child
        if child is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer launch returned without retained child ownership"
            )
        identity, stdout, stderr = _capture_and_wait_for_reviewer_child_v1(
            child,
            reviewer_implementation,
        )
        completed = True
        return child, identity, stdout, stderr
    finally:
        if completed:
            child_owner.child = None
        else:
            child_owner.close_noexcept()


def publish_canonical_control_leaf_create_new_v1(
    destination: str | Path,
    payload: bytes,
) -> str:
    """Durably CREATE_NEW one immutable control leaf without cleanup or retry."""

    path = Path(destination)
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or type(payload) is not bytes
        or not payload
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "control-leaf destination/payload is not canonical"
        )
    parent = path.parent.resolve(strict=True)
    parent_value = path.parent.stat(follow_symlinks=False)
    if (
        parent != path.parent
        or _is_reparse(path.parent, parent_value)
        or not stat.S_ISDIR(parent_value.st_mode)
        or _lexists(path)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "control-leaf parent is unsafe or destination already exists; "
            "overwrite/adoption/retry are forbidden"
        )
    parent_anchor: _RetainedDirectoryAnchor | None = None
    retained: _RetainedPublishedFile | None = None
    try:
        parent_anchor = _RetainedDirectoryAnchor.open(parent)
        parent_anchor.assert_current()
        retained = _create_and_require(path, payload)
        parent_anchor.flush_and_assert()
        retained.assert_current()
        return retained.sha256
    except BaseException as exc:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "STOP: immutable control-leaf CREATE_NEW failed; retain all state; "
            "overwrite, adoption, cleanup, and retry are forbidden: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if retained is not None:
            retained.close_noexcept()
        if parent_anchor is not None:
            parent_anchor.close_noexcept()


def verify_original_confirmatory_technical_intent_source_binding_v1(
    *,
    intent: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize intent and cross-bind the exact supplied source inventory."""

    canonical_intent = (
        authority_schema.canonical_original_confirmatory_technical_authority_intent_v1(intent)
    )
    source = canonical_intent["execution_source"]
    inventory_bytes = authority_schema.canonical_json_line_bytes(source_inventory)
    inventory_path = Path(source["manifest_path"])
    if (
        _sha256_bytes(inventory_bytes) != source["manifest_sha256"]
        or source_inventory.get("root_sha256") != source["root_sha256"]
        or not isinstance(source_inventory.get("artifacts"), list)
        or len(source_inventory["artifacts"]) != source["record_count"]
        or _stable_regular_file_bytes(
            inventory_path,
            role="bound execution source inventory",
        )
        != inventory_bytes
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "execution source inventory differs from canonical intent"
        )
    return canonical_intent


def verify_original_confirmatory_technical_intent_live_bindings_v1(
    *,
    intent: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    project_root: str | Path,
    reviewer_process: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform one full outcome-blind live pass for the actual reviewer."""

    canonical_intent = verify_original_confirmatory_technical_intent_source_binding_v1(
        intent=intent,
        source_inventory=source_inventory,
    )
    canonical_reviewer = authority_schema._process(
        reviewer_process,
        role="technical authority independent reviewer",
    )
    root = Path(project_root).resolve(strict=True)
    authority_schema._verify_live_bindings(
        project_root=root,
        intent=canonical_intent,
        review={"reviewer_process": canonical_reviewer},
        source_inventory=source_inventory,
    )
    return canonical_intent


def _directory_identity_snapshot(directory: Path) -> tuple[tuple[Any, ...], ...]:
    records: list[tuple[Any, ...]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        value = path.stat(follow_symlinks=False)
        records.append(
            (
                path.name,
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                int(getattr(value, "st_file_attributes", 0)),
            )
        )
    return tuple(records)


def verify_original_confirmatory_technical_authority_namespace_claim_v1(
    authority_directory: str | Path,
) -> str:
    """Read-only verify the permanent singleton claim around qualified T0."""

    supplied = Path(authority_directory)
    if (
        not supplied.is_absolute()
        or supplied != Path(os.path.abspath(supplied))
        or not _lexists(supplied)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 directory must be one existing canonical absolute path"
        )
    supplied_value = supplied.stat(follow_symlinks=False)
    if _is_reparse(supplied, supplied_value) or not stat.S_ISDIR(supplied_value.st_mode):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 directory must be one regular non-link directory"
        )
    directory = supplied.resolve(strict=True)
    if directory != supplied:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 directory resolves through a link or non-canonical ancestor"
        )
    namespace = directory.parent
    if (
        namespace.name != AUTHORITY_NAMESPACE_DIRECTORY_NAME
        or namespace.parent.name != "artifacts"
        or namespace.resolve(strict=True) != namespace
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 is outside the exact singleton namespace"
        )
    namespace_value = namespace.stat(follow_symlinks=False)
    directory_value = directory.stat(follow_symlinks=False)
    if (
        _is_reparse(namespace, namespace_value)
        or not stat.S_ISDIR(namespace_value.st_mode)
        or _is_reparse(directory, directory_value)
        or not stat.S_ISDIR(directory_value.st_mode)
        or {path.name for path in namespace.iterdir()} != {NAMESPACE_CLAIM_FILENAME, directory.name}
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 singleton namespace inventory is not exact terminal success"
        )
    before = _directory_identity_snapshot(namespace)
    intent = _read_json_input(
        directory / authority_schema.INTENT_FILENAME,
        role="published technical authority intent",
    )
    if _authority_namespace_from_intent(intent) != namespace:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 namespace is not derived from the exact parent-P project root"
        )
    project_root = namespace.parents[1]
    request_anchor = _open_authority_request_namespace(
        project_root,
        create=False,
    )
    try:
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="terminal namespace verification",
        )
    finally:
        request_anchor.close_noexcept()
    attempt_path = (
        project_root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME / REVIEW_ATTEMPT_FILENAME
    )
    verified_attempt = verify_original_confirmatory_technical_review_attempt_claim_v1(
        attempt_path,
        intent=intent,
        project_root=project_root,
    )
    review = authority_schema.canonical_original_confirmatory_technical_authority_review_v1(
        _read_json_input(
            directory / authority_schema.REVIEW_FILENAME,
            role="published independent review",
        ),
        intent=intent,
    )
    if _stable_regular_file_bytes(
        project_root / "artifacts" / AUTHORITY_REQUEST_DIRECTORY_NAME / REVIEW_REQUEST_FILENAME,
        role="terminal request review",
    ) != authority_schema.canonical_json_line_bytes(review) or authority_schema._utc(
        verified_attempt["attempt_created_at_utc"],
        role="review-attempt creation",
    ) > authority_schema._utc(
        review["review_started_at_utc"],
        role="review start",
    ):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "published review differs from or predates its permanent request chain"
        )
    review_attempt_claim_sha256 = _sha256_bytes(
        _stable_regular_file_bytes(
            attempt_path,
            role="published review-attempt claim",
        )
    )
    evidence = _read_json_input(
        directory / authority_schema.EVIDENCE_FILENAME,
        role="published technical authority evidence",
    )
    manifest_path = directory / MANIFEST_FILENAME
    manifest_bytes = _stable_regular_file_bytes(
        manifest_path,
        role="published technical authority manifest",
    )
    manifest = _strict_json_line_bytes(
        manifest_bytes,
        role="published technical authority manifest",
    )
    expected = _namespace_claim_bytes(
        authority_directory=directory,
        intent=intent,
        publication_timestamp_utc=evidence["publication_timestamp_utc"],
        artifact_root_sha256=manifest["artifact_root_sha256"],
        sha256_manifest_sha256=_sha256_bytes(manifest_bytes),
        technical_authorization_sha256=evidence["technical_authorization_sha256"],
        review_attempt_claim_sha256=review_attempt_claim_sha256,
    )
    claim_path = namespace / NAMESPACE_CLAIM_FILENAME
    claim_value = claim_path.stat(follow_symlinks=False)
    if not _is_read_only(claim_value):
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "permanent T0 namespace claim is writable"
        )
    observed = _stable_regular_file_bytes(
        claim_path,
        role="permanent T0 namespace claim",
    )
    _strict_json_line_bytes(observed, role="permanent T0 namespace claim")
    if observed != expected:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "permanent T0 namespace claim differs from terminal authority"
        )
    if _directory_identity_snapshot(namespace) != before:
        raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
            "T0 singleton namespace changed during read-only verification"
        )
    return _sha256_bytes(observed)


def verify_published_original_confirmatory_technical_authority_v1(
    authority_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_live: bool = True,
) -> VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1:
    """Qualify both the live T0 schema and singleton claim without writing."""

    directory = Path(authority_directory)
    claim_before = verify_original_confirmatory_technical_authority_namespace_claim_v1(directory)
    directory = directory.resolve(strict=True)
    namespace_anchor = _RetainedDirectoryAnchor.open(
        directory.parent,
        write_access=False,
    )
    authority_anchor: _RetainedDirectoryAnchor | None = None
    request_anchor: _RetainedDirectoryAnchor | None = None
    retained: list[_RetainedPublishedFile] = []
    try:
        authority_anchor = _RetainedDirectoryAnchor.open(
            directory,
            write_access=False,
        )
        derived_project_root = directory.parent.parents[1]
        request_anchor = _open_authority_request_namespace(
            derived_project_root,
            create=False,
        )
        namespace_anchor.assert_current()
        authority_anchor.assert_current()
        request_anchor.assert_current()
        retained.append(
            _RetainedPublishedFile.open_existing(directory.parent / NAMESPACE_CLAIM_FILENAME)
        )
        for name in sorted(QUALIFYING_FILENAMES):
            retained.append(_RetainedPublishedFile.open_existing(directory / name))
        review_attempt_retained: _RetainedPublishedFile | None = None
        for name in (
            INTENT_REQUEST_FILENAME,
            REVIEW_ATTEMPT_FILENAME,
            REVIEW_REQUEST_FILENAME,
        ):
            item = _RetainedPublishedFile.open_existing(request_anchor.path / name)
            retained.append(item)
            if name == REVIEW_ATTEMPT_FILENAME:
                review_attempt_retained = item
        if review_attempt_retained is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "retained review-attempt claim is absent"
            )
        namespace_anchor.assert_current()
        authority_anchor.assert_current()
        request_anchor.assert_current()
        if {path.name for path in directory.iterdir()} != QUALIFYING_FILENAMES:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "retained T0 terminal inventory is not exact"
            )
        if {path.name for path in directory.parent.iterdir()} != {
            NAMESPACE_CLAIM_FILENAME,
            directory.name,
        }:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "retained T0 namespace inventory is not exact"
            )
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="retained terminal verification",
        )
        claim_before = verify_original_confirmatory_technical_authority_namespace_claim_v1(
            directory
        )
        verified = authority_schema.verify_original_confirmatory_technical_authority_v1(
            directory,
            project_root=project_root,
            verify_live=verify_live,
        )
        claim_after = verify_original_confirmatory_technical_authority_namespace_claim_v1(directory)
        for item in retained:
            item.assert_current()
        namespace_anchor.assert_current()
        authority_anchor.assert_current()
        request_anchor.assert_current()
        if claim_before != claim_after:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "T0 singleton claim changed across live schema verification"
            )
        return VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
            authority=verified,
            namespace_directory=verified.authority_directory.parent,
            namespace_claim_sha256=claim_after,
            review_attempt_claim_sha256=review_attempt_retained.sha256,
        )
    finally:
        for item in reversed(retained):
            item.close_noexcept()
        if authority_anchor is not None:
            authority_anchor.close_noexcept()
        if request_anchor is not None:
            request_anchor.close_noexcept()
        namespace_anchor.close_noexcept()


def _cli_failure(message: str) -> NoReturn:
    typer.echo(f"ERROR: {message}", err=True)
    raise typer.Exit(code=1)


original_confirmatory_technical_authority_app = typer.Typer(
    help=(
        "Build, review, CREATE_NEW-publish, or independently verify the "
        "sealed original-confirmatory T0 authority."
    ),
    no_args_is_help=True,
)


@original_confirmatory_technical_authority_app.command("build-intent")
def build_original_confirmatory_technical_authority_intent_v1_command(
    parent_json: Annotated[Path, typer.Option("--parent-json", dir_okay=False)],
    frozen_science_json: Annotated[
        Path,
        typer.Option("--frozen-science-json", dir_okay=False),
    ],
    historical_primary_json: Annotated[
        Path,
        typer.Option("--historical-primary-json", dir_okay=False),
    ],
    execution_source_json: Annotated[
        Path,
        typer.Option("--execution-source-json", dir_okay=False),
    ],
    source_inventory_json: Annotated[
        Path,
        typer.Option("--source-inventory-json", dir_okay=False),
    ],
    execution_capsule_json: Annotated[
        Path,
        typer.Option("--execution-capsule-json", dir_okay=False),
    ],
    capacity_v2_json: Annotated[
        Path,
        typer.Option("--capacity-v2-json", dir_okay=False),
    ],
    outcome_scope_json: Annotated[
        Path,
        typer.Option("--outcome-scope-json", dir_okay=False),
    ],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False),
    ] = Path("."),
) -> None:
    """Canonical-build, source-cross-bind, and CREATE_NEW one intent."""

    request_anchor: _RetainedDirectoryAnchor | None = None
    try:
        root = project_root.resolve(strict=True)
        request_anchor = _open_authority_request_namespace(root, create=True)
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(),
            phase="pre-build",
        )
        output_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / INTENT_REQUEST_FILENAME,
            expected_filename=INTENT_REQUEST_FILENAME,
            role="technical intent output",
        )
        inputs = {
            "parent": _read_json_input(
                _resolve_from_root(root, parent_json),
                role="parent-P binding",
            ),
            "frozen_science": _read_json_input(
                _resolve_from_root(root, frozen_science_json),
                role="frozen-science binding",
            ),
            "historical_primary": _read_json_input(
                _resolve_from_root(root, historical_primary_json),
                role="historical-primary binding",
            ),
            "execution_source": _read_json_input(
                _resolve_from_root(root, execution_source_json),
                role="execution-source binding",
            ),
            "execution_capsule": _read_json_input(
                _resolve_from_root(root, execution_capsule_json),
                role="execution-capsule binding",
            ),
            "capacity_v2": _read_json_input(
                _resolve_from_root(root, capacity_v2_json),
                role="capacity-v2 binding",
            ),
            "outcome_scope": _read_json_input(
                _resolve_from_root(root, outcome_scope_json),
                role="outcome-split scope",
            ),
        }
        source_inventory_path = _resolve_from_root(root, source_inventory_json)
        source_inventory = _read_json_input(
            source_inventory_path,
            role="execution source inventory",
        )
        if Path(inputs["execution_source"]["manifest_path"]) != source_inventory_path:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "source-inventory input path differs from execution-source binding"
            )
        builder_process = capture_current_process_identity_v1(Path(__file__).resolve())
        intent = authority_schema.build_original_confirmatory_technical_authority_intent_v1(
            created_at_utc=_canonical_utc_now(),
            builder_process=builder_process,
            parent=inputs["parent"],
            frozen_science=inputs["frozen_science"],
            historical_primary=inputs["historical_primary"],
            execution_source=inputs["execution_source"],
            execution_capsule=inputs["execution_capsule"],
            capacity_v2=inputs["capacity_v2"],
            outcome_scope=inputs["outcome_scope"],
        )
        verified_intent = verify_original_confirmatory_technical_intent_source_binding_v1(
            intent=intent,
            source_inventory=source_inventory,
        )
        payload = authority_schema.canonical_json_line_bytes(verified_intent)
        output_sha256 = publish_canonical_control_leaf_create_new_v1(
            output_path,
            payload,
        )
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset({INTENT_REQUEST_FILENAME}),
            phase="post-build",
        )
    except Exception as exc:
        _cli_failure(
            f"original-confirmatory technical-intent build failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if request_anchor is not None:
            request_anchor.close_noexcept()
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": ("original_confirmatory_technical_authority_v1_build_intent"),
                "decision": "passed",
                "output": str(output_path),
                "output_sha256": output_sha256,
                "intent_root_sha256": verified_intent["intent_root_sha256"],
                "builder_process": builder_process,
                "outcome_values_read": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
                "automatic_retry_allowed": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


@original_confirmatory_technical_authority_app.command("review-intent")
def review_original_confirmatory_technical_authority_intent_v1_command(
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False),
    ] = Path("."),
) -> None:
    """Spawn exactly one fresh reviewer child and verify its immutable receipt."""

    retained_intent: _RetainedPublishedFile | None = None
    retained_attempt: _RetainedPublishedFile | None = None
    retained_review: _RetainedPublishedFile | None = None
    request_anchor: _RetainedDirectoryAnchor | None = None
    review_mutex: _RetainedReviewMutex | None = None
    try:
        root = project_root.resolve(strict=True)
        review_mutex = _RetainedReviewMutex.acquire(root)
        request_anchor = _open_authority_request_namespace(root, create=False)
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset({INTENT_REQUEST_FILENAME}),
            phase="pre-review",
        )
        intent_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / INTENT_REQUEST_FILENAME,
            expected_filename=INTENT_REQUEST_FILENAME,
            role="technical intent input",
        )
        output_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / REVIEW_REQUEST_FILENAME,
            expected_filename=REVIEW_REQUEST_FILENAME,
            role="independent-review output",
        )
        attempt_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / REVIEW_ATTEMPT_FILENAME,
            expected_filename=REVIEW_ATTEMPT_FILENAME,
            role="review-attempt claim",
        )
        retained_intent = _RetainedPublishedFile.open_existing(intent_path)
        intent = authority_schema.canonical_original_confirmatory_technical_authority_intent_v1(
            _strict_json_line_bytes(
                retained_intent.payload,
                role="technical intent",
            )
        )
        source_inventory_path = Path(intent["execution_source"]["manifest_path"])
        source_inventory = _read_json_input(
            source_inventory_path,
            role="execution source inventory",
        )
        verify_original_confirmatory_technical_intent_source_binding_v1(
            intent=intent,
            source_inventory=source_inventory,
        )
        if _lexists(output_path):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "review output already exists; adoption/retry are forbidden"
            )
        module_spec = importlib.util.find_spec(REVIEWER_MODULE_NAME)
        if module_spec is None or module_spec.origin is None:
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "fresh-child reviewer implementation is unavailable"
            )
        reviewer_implementation = Path(module_spec.origin).resolve(strict=True)
        reviewer_implementation_sha256 = _sha256_bytes(
            _stable_regular_file_bytes(
                reviewer_implementation,
                role="fresh-child reviewer implementation",
            )
        )
        builder = intent["builder_process"]
        if (
            reviewer_implementation == Path(builder["implementation_path"])
            or reviewer_implementation_sha256 == builder["implementation_sha256"]
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "reviewer and builder implementation spaces are not independent"
            )
        controller_process = capture_current_process_identity_v1(Path(__file__).resolve())
        attempt = build_original_confirmatory_technical_review_attempt_claim_v1(
            intent=intent,
            project_root=root,
            controller_process=controller_process,
            reviewer_implementation_path=reviewer_implementation,
        )
        attempt_sha256 = publish_canonical_control_leaf_create_new_v1(
            attempt_path,
            authority_schema.canonical_json_line_bytes(attempt),
        )
        retained_attempt = _RetainedPublishedFile.open_existing(attempt_path)
        verified_attempt = verify_original_confirmatory_technical_review_attempt_claim_v1(
            attempt_path,
            intent=intent,
            project_root=root,
        )
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                }
            ),
            phase="pre-CreateProcessW review",
        )
        child_environment = os.environ.copy()
        explicit_child_environment: dict[str, str] = {}
        if sys.prefix != sys.base_prefix:
            # Windows venv python.exe is a redirector that spawns the base
            # interpreter and exits, breaking exact retained-handle custody.
            # Launch the already-attested live interpreter directly while
            # preserving the venv identity derived from this explicit marker.
            explicit_child_environment["__PYVENV_LAUNCHER__"] = sys.executable
            child_environment.update(explicit_child_environment)
        argv = [
            controller_process["executable_path"],
            "-B",
            "-m",
            REVIEWER_MODULE_NAME,
            "--intent-json",
            str(intent_path),
            "--output",
            str(output_path),
            "--project-root",
            str(root),
        ]
        (
            child,
            spawned_process_identity,
            stdout,
            stderr_bytes,
        ) = _launch_capture_and_wait_for_reviewer_v1(
            argv,
            cwd=root,
            env=child_environment,
            reviewer_implementation=reviewer_implementation,
        )
        if child.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace")[-4000:]
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "fresh-child reviewer failed; this controller will not retry: "
                f"exit_code={child.returncode}; stderr={stderr}"
            )
        retained_review = _RetainedPublishedFile.open_existing(output_path)
        review = authority_schema.canonical_original_confirmatory_technical_authority_review_v1(
            _strict_json_line_bytes(
                retained_review.payload,
                role="independent review receipt",
            ),
            intent=intent,
        )
        reviewer = review["reviewer_process"]
        if (
            reviewer != spawned_process_identity
            or child.pid != reviewer["process_id"]
            or reviewer["process_id"]
            in {
                controller_process["process_id"],
                builder["process_id"],
            }
            or (
                reviewer["process_id"],
                reviewer["process_created_at_utc"],
            )
            == (
                builder["process_id"],
                builder["process_created_at_utc"],
            )
            or Path(reviewer["implementation_path"]) != reviewer_implementation
            or reviewer["implementation_sha256"] != reviewer_implementation_sha256
            or reviewer["implementation_path"] == builder["implementation_path"]
            or reviewer["implementation_sha256"] == builder["implementation_sha256"]
            or reviewer["executable_path"] != controller_process["executable_path"]
            or reviewer["executable_sha256"] != controller_process["executable_sha256"]
            or authority_schema._utc(
                verified_attempt["attempt_created_at_utc"],
                role="review-attempt creation",
            )
            > authority_schema._utc(
                review["review_started_at_utc"],
                role="review start",
            )
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "fresh-child review process/source independence is invalid"
            )
        retained_intent.assert_current()
        retained_attempt.assert_current()
        retained_review.assert_current()
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="post-review",
        )
        review_sha256 = _sha256_bytes(authority_schema.canonical_json_line_bytes(review))
        child_stdout_sha256 = _sha256_bytes(stdout)
    except Exception as exc:
        _cli_failure(f"original-confirmatory intent review failed: {type(exc).__name__}: {exc}")
    finally:
        if retained_review is not None:
            retained_review.close_noexcept()
        if retained_attempt is not None:
            retained_attempt.close_noexcept()
        if retained_intent is not None:
            retained_intent.close_noexcept()
        if request_anchor is not None:
            request_anchor.close_noexcept()
        if review_mutex is not None:
            review_mutex.close_noexcept()
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": ("original_confirmatory_technical_authority_v1_review_intent"),
                "decision": "passed",
                "output": str(output_path),
                "output_sha256": review_sha256,
                "review_root_sha256": review["review_root_sha256"],
                "review_attempt_claim_sha256": attempt_sha256,
                "review_attempt_root_sha256": verified_attempt["attempt_root_sha256"],
                "fresh_child_argv_sha256": (authority_schema.canonical_json_sha256(argv)),
                "fresh_child_explicit_environment_sha256": (
                    authority_schema.canonical_json_sha256(explicit_child_environment)
                ),
                "fresh_child_stdout_sha256": child_stdout_sha256,
                "reviewer_process": review["reviewer_process"],
                "outcome_values_read": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
                "automatic_retry_allowed": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


@original_confirmatory_technical_authority_app.command("publish")
def publish_original_confirmatory_technical_authority_v1_command(
    destination: Annotated[Path, typer.Option("--destination")],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False),
    ] = Path("."),
) -> None:
    """Build and publish once; do not run terminal verification in this process."""

    retained_intent: _RetainedPublishedFile | None = None
    retained_attempt: _RetainedPublishedFile | None = None
    retained_review: _RetainedPublishedFile | None = None
    request_anchor: _RetainedDirectoryAnchor | None = None
    try:
        root = project_root.resolve(strict=True)
        destination_path = _resolve_from_root(root, destination)
        request_anchor = _open_authority_request_namespace(root, create=False)
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="pre-publication",
        )
        intent_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / INTENT_REQUEST_FILENAME,
            expected_filename=INTENT_REQUEST_FILENAME,
            role="technical authority intent",
        )
        review_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / REVIEW_REQUEST_FILENAME,
            expected_filename=REVIEW_REQUEST_FILENAME,
            role="independent review",
        )
        attempt_path = _require_authority_request_leaf(
            project_root=root,
            path=request_anchor.path / REVIEW_ATTEMPT_FILENAME,
            expected_filename=REVIEW_ATTEMPT_FILENAME,
            role="review-attempt claim",
        )
        retained_intent = _RetainedPublishedFile.open_existing(intent_path)
        retained_attempt = _RetainedPublishedFile.open_existing(attempt_path)
        retained_review = _RetainedPublishedFile.open_existing(review_path)
        intent = _strict_json_line_bytes(
            retained_intent.payload,
            role="technical authority intent",
        )
        review = _strict_json_line_bytes(
            retained_review.payload,
            role="independent review",
        )
        verified_attempt = verify_original_confirmatory_technical_review_attempt_claim_v1(
            attempt_path,
            intent=intent,
            project_root=root,
        )
        canonical_review = (
            authority_schema.canonical_original_confirmatory_technical_authority_review_v1(
                review,
                intent=intent,
            )
        )
        if authority_schema._utc(
            verified_attempt["attempt_created_at_utc"],
            role="review-attempt creation",
        ) > authority_schema._utc(
            canonical_review["review_started_at_utc"],
            role="review start",
        ):
            raise OriginalConfirmatoryTechnicalAuthorityPublicationV1Error(
                "review-attempt claim is later than the independent review"
            )
        source_inventory_path = Path(intent["execution_source"]["manifest_path"])
        source_inventory = _read_json_input(
            source_inventory_path,
            role="execution source inventory",
        )
        publication_timestamp_utc = _canonical_utc_now()
        bundle = authority_schema.build_original_confirmatory_technical_authority_bundle_v1(
            authority_directory=destination_path,
            intent=intent,
            independent_review=review,
            publication_timestamp_utc=publication_timestamp_utc,
            preregistration_bytes=_stable_regular_file_bytes(
                root / "PRE_REGISTRATION.md",
                role="frozen preregistration",
            ),
            primary_config_bytes=_stable_regular_file_bytes(
                root / "configs" / "primary_frozen.yaml",
                role="frozen primary config",
            ),
            confirmatory_config_bytes=_stable_regular_file_bytes(
                root / "configs" / "confirmatory_frozen.yaml",
                role="frozen confirmatory config",
            ),
            source_inventory=source_inventory,
        )
        published = publish_original_confirmatory_technical_authority_v1_once(bundle)
        review_attempt_claim_sha256 = retained_attempt.sha256
        retained_intent.assert_current()
        retained_attempt.assert_current()
        retained_review.assert_current()
        _require_authority_request_inventory(
            request_anchor,
            expected_names=frozenset(
                {
                    INTENT_REQUEST_FILENAME,
                    REVIEW_ATTEMPT_FILENAME,
                    REVIEW_REQUEST_FILENAME,
                }
            ),
            phase="post-publication",
        )
    except Exception as exc:
        _cli_failure(f"original-confirmatory T0 publication failed: {type(exc).__name__}: {exc}")
    finally:
        if retained_review is not None:
            retained_review.close_noexcept()
        if retained_attempt is not None:
            retained_attempt.close_noexcept()
        if retained_intent is not None:
            retained_intent.close_noexcept()
        if request_anchor is not None:
            request_anchor.close_noexcept()
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "original_confirmatory_technical_authority_v1_publication",
                "terminal_disposition": "success",
                "scientific_execution_performed": False,
                "independent_verification_performed": False,
                "automatic_retry_allowed": False,
                "publication_timestamp_utc": publication_timestamp_utc,
                "review_attempt_claim_sha256": review_attempt_claim_sha256,
                "publication": published.as_dict(),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


@original_confirmatory_technical_authority_app.command("verify")
def verify_original_confirmatory_technical_authority_v1_command(
    authority_directory: Annotated[
        Path,
        typer.Option("--authority-directory", file_okay=False),
    ],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False),
    ] = Path("."),
) -> None:
    """Independently and read-only verify terminal success plus live bindings."""

    try:
        root = project_root.resolve(strict=True)
        directory = _resolve_from_root(root, authority_directory)
        verified = verify_published_original_confirmatory_technical_authority_v1(
            directory,
            project_root=root,
            verify_live=True,
        )
    except Exception as exc:
        _cli_failure(f"original-confirmatory T0 verification failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "original_confirmatory_technical_authority_v1_verification",
                "decision": "passed",
                "read_only": True,
                "scientific_execution_performed": False,
                "namespace_claim_sha256": verified.namespace_claim_sha256,
                "review_attempt_claim_sha256": verified.review_attempt_claim_sha256,
                "authority": verified.as_dict()["authority"],
                "lifecycle_binding": verified.lifecycle_binding(),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "AUTHORITY_NAMESPACE_DIRECTORY_NAME",
    "AUTHORITY_REQUEST_DIRECTORY_NAME",
    "INTENT_REQUEST_FILENAME",
    "NAMESPACE_CLAIM_FILENAME",
    "NAMESPACE_STOP_FILENAME",
    "PRETERMINAL_FILENAMES",
    "REVIEWER_MODULE_NAME",
    "REVIEW_ATTEMPT_FILENAME",
    "REVIEW_ATTEMPT_POLICY",
    "REVIEW_REQUEST_FILENAME",
    "OriginalConfirmatoryTechnicalAuthorityPublicationV1Error",
    "OriginalConfirmatoryTechnicalAuthorityPublicationV1Result",
    "VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1",
    "build_original_confirmatory_technical_authority_intent_v1_command",
    "build_original_confirmatory_technical_review_attempt_claim_v1",
    "capture_current_process_identity_v1",
    "original_confirmatory_technical_authority_app",
    "publish_canonical_control_leaf_create_new_v1",
    "publish_original_confirmatory_technical_authority_v1_command",
    "publish_original_confirmatory_technical_authority_v1_once",
    "review_original_confirmatory_technical_authority_intent_v1_command",
    "verify_original_confirmatory_technical_authority_namespace_claim_v1",
    "verify_original_confirmatory_technical_authority_v1_command",
    "verify_original_confirmatory_technical_intent_live_bindings_v1",
    "verify_original_confirmatory_technical_intent_source_binding_v1",
    "verify_original_confirmatory_technical_review_attempt_claim_v1",
    "verify_published_original_confirmatory_technical_authority_v1",
]


if __name__ == "__main__":
    original_confirmatory_technical_authority_app()
