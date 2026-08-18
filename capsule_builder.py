from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import re
import stat
import struct
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

MANIFEST_NAME: Final = "AANCA_CAPSULE_MANIFEST.json"
CAPSULE_FILENAME: Final = "original_confirmatory.pyz"
MANIFEST_POLICY: Final = "original_confirmatory_execution_capsule_manifest_v1"
SOURCE_INVENTORY_POLICY: Final = "original_confirmatory_execution_capsule_source_inventory_v1"
FIXED_ZIP_DATETIME: Final = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR: Final = (stat.S_IFREG | 0o444) << 16
MAX_MEMBER_BYTES: Final = 512 * 1024 * 1024
_MEMBER_PATH = re.compile(r"^[a-z0-9_][a-z0-9_.-]*(/[a-z0-9_][a-z0-9_.-]*)*$")
_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMITTED_ROLES: Final = frozenset(
    {
        "capsule_authority",
        "capsule_bootstrap",
        "capsule_contract",
        "capsule_dispatcher",
        "capsule_policy",
        "capsule_terminal",
        "package_initializer",
        "project_source",
        "scientific_completion",
        "scientific_entry",
    }
)
_WINDOWS_RESERVED = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
_ARCHIVE_POLICY: Final = {
    "format": "zip",
    "compression": "stored",
    "payload_entry_order": "ordinal_then_manifest_last",
    "fixed_dos_datetime": "1980-01-01T00:00:00",
    "create_system": 3,
    "unix_mode": "100444",
    "zip64": False,
    "archive_comment_empty": True,
    "directory_entries": False,
    "manifest_self_entry": False,
}
_SPECIAL_PROJECT_ROLES: Final = {
    "histo_audit/experiment/confirmatory_completion.py": "scientific_completion",
    "histo_audit/experiment/original_confirmatory_runner_core.py": "scientific_entry",
    "histo_audit/workflows/original_confirmatory_capsule_authority.py": ("capsule_authority"),
    "histo_audit/workflows/original_confirmatory_capsule_entry.py": "capsule_dispatcher",
    "histo_audit/workflows/original_confirmatory_capsule_terminal.py": "capsule_terminal",
}
_REQUIRED_PROJECT_MEMBERS: Final = frozenset(_SPECIAL_PROJECT_ROLES)


class CapsuleBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    link_count: int
    file_attributes: int


@dataclass(frozen=True, slots=True)
class PayloadMember:
    source_path: Path
    relative_path: str
    role: str
    size_bytes: int
    sha256: str
    payload: bytes
    identity: FileIdentity

    def manifest_record(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceInventory:
    entries: tuple[dict[str, Any], ...]
    records_root_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": SOURCE_INVENTORY_POLICY,
            "entries": [dict(entry) for entry in self.entries],
            "entry_count": len(self.entries),
            "records_root_sha256": self.records_root_sha256,
        }


@dataclass(frozen=True, slots=True)
class CapsuleBuildResult:
    output_path: Path
    size_bytes: int
    sha256: str
    internal_manifest_sha256: str
    records_root_sha256: str
    entry_count: int
    payload_size_bytes: int


@dataclass(frozen=True, slots=True)
class CapsuleByteBuild:
    archive_bytes: bytes
    manifest_bytes: bytes
    members: tuple[PayloadMember, ...]
    size_bytes: int
    sha256: str
    internal_manifest_sha256: str
    records_root_sha256: str
    entry_count: int
    payload_size_bytes: int


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _records_preimage(records: Sequence[Mapping[str, Any]]) -> bytes:
    chunks: list[bytes] = []
    for record in records:
        relative_path = record["relative_path"]
        role = record["role"]
        size_bytes = record["size_bytes"]
        sha256 = record["sha256"]
        chunks.append(f"{relative_path}\0{role}\0{size_bytes}\0{sha256}\n".encode("ascii"))
    return b"".join(chunks)


def _records_root(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_records_preimage(records)).hexdigest()


def _file_attributes(value: os.stat_result) -> int:
    return int(getattr(value, "st_file_attributes", 0))


def _is_reparse(value: os.stat_result) -> bool:
    return bool(_file_attributes(value) & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _windows_api_path(path: Path) -> str:
    value = str(path)
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _lstat(path: Path) -> os.stat_result:
    if os.name == "nt":
        return os.lstat(_windows_api_path(path))
    return os.lstat(path)


def _identity(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size_bytes=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
        link_count=int(value.st_nlink),
        file_attributes=_file_attributes(value),
    )


def _same_file_object(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare stable Windows/POSIX fields shared by path and handle stat.

    On Windows, CPython can expose path `st_ctime_ns` and handle `st_ctime_ns`
    with different creation-time precision for a file copied with metadata.
    Identity, content length, modification time, mode, link count, and file
    attributes remain exact; descriptor-before/after comparison still includes
    the full `FileIdentity`, including `ctime_ns`.
    """

    return (
        int(left.st_dev),
        int(left.st_ino),
        int(left.st_mode),
        int(left.st_size),
        int(left.st_mtime_ns),
        int(left.st_nlink),
        _file_attributes(left),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        int(right.st_mode),
        int(right.st_size),
        int(right.st_mtime_ns),
        int(right.st_nlink),
        _file_attributes(right),
    )


def _validate_member_path(relative_path: str) -> str:
    if (
        type(relative_path) is not str
        or relative_path == MANIFEST_NAME
        or relative_path.casefold() == MANIFEST_NAME.casefold()
        or _MEMBER_PATH.fullmatch(relative_path) is None
        or str(PurePosixPath(relative_path)) != relative_path
        or "\\" in relative_path
        or ":" in relative_path
        or "\x00" in relative_path
    ):
        raise CapsuleBuildError(f"unsafe capsule member path: {relative_path!r}")
    for segment in relative_path.split("/"):
        stem = segment.split(".", 1)[0].casefold()
        if (
            segment in {".", ".."}
            or segment.endswith((".", " "))
            or stem in _WINDOWS_RESERVED
            or any(ord(character) < 32 or ord(character) > 126 for character in segment)
        ):
            raise CapsuleBuildError(f"unsafe capsule path segment: {segment!r}")
    return relative_path


def _validate_role(role: str) -> str:
    if type(role) is not str or _ROLE.fullmatch(role) is None or role not in _EMITTED_ROLES:
        raise CapsuleBuildError(f"unsafe capsule member role: {role!r}")
    return role


def _require_plain_ancestor_chain(path: Path) -> None:
    """Inspect the supplied lexical path from its anchor without following aliases."""

    if not path.is_absolute():
        raise CapsuleBuildError(f"path is not absolute: {path}")
    parts = path.parts
    if not parts or not path.anchor:
        raise CapsuleBuildError(f"path has no absolute anchor: {path}")
    component = Path(parts[0])
    chain = [component]
    for part in parts[1:]:
        component = component / part
        chain.append(component)
    for component in chain:
        value = _lstat(component)
        if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
            raise CapsuleBuildError(f"source ancestor is a link/reparse point: {component}")


def _require_canonical_existing_path(path: str | Path, *, label: str) -> Path:
    try:
        supplied_text = os.fspath(path)
    except TypeError as error:
        raise CapsuleBuildError(f"{label} is not path-like") from error
    if type(supplied_text) is not str or not supplied_text or "\x00" in supplied_text:
        raise CapsuleBuildError(f"{label} is not a non-empty text path")
    supplied = Path(supplied_text)
    if not supplied.is_absolute():
        raise CapsuleBuildError(f"{label} must be supplied as an absolute path")
    lexical_normal = os.path.abspath(supplied_text)
    if supplied_text != str(supplied) or supplied_text != lexical_normal:
        raise CapsuleBuildError(f"{label} is not in canonical lexical form: {supplied_text}")

    # This traversal deliberately precedes resolve(). A resolve-first check would
    # erase the evidence that a supplied path component was a symlink/junction.
    _require_plain_ancestor_chain(supplied)
    if os.name == "nt":
        # Every lexical component was inspected with long-path-aware lstat and no
        # component is a reparse point. Path.resolve() is MAX_PATH-limited on some
        # supported Python/Windows combinations and adds no authority here.
        return supplied
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise CapsuleBuildError(f"{label} does not resolve to an existing object") from error
    if str(resolved) != supplied_text:
        raise CapsuleBuildError(f"{label} is not the canonical filesystem path: {supplied}")
    return supplied


def _named_streams(path: Path) -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    class WIN32_FIND_STREAM_DATA(ctypes.Structure):
        _fields_ = [
            ("StreamSize", ctypes.c_longlong),
            ("cStreamName", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.POINTER(WIN32_FIND_STREAM_DATA),
        ctypes.c_uint32,
    ]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(WIN32_FIND_STREAM_DATA)]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int

    data = WIN32_FIND_STREAM_DATA()
    handle = find_first(_windows_api_path(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error == 38:
            return ()
        raise CapsuleBuildError(f"cannot enumerate named streams for {path}: {error}")
    values: list[str] = []
    try:
        while True:
            values.append(str(data.cStreamName))
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise CapsuleBuildError(
                    f"cannot continue named-stream enumeration for {path}: {error}"
                )
    finally:
        find_close(handle)
    return tuple(sorted(value for value in values if value != "::$DATA"))


def _open_read_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_CLOEXEC", 0))
    if os.name != "nt":
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        return os.open(path, flags)

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
        _windows_api_path(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00200000 | 0x08000000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
    try:
        return msvcrt.open_osfhandle(int(handle), flags)
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise


def _read_stable_regular_file(
    source_path: Path,
    *,
    relative_path: str,
    role: str,
) -> PayloadMember:
    source = _require_canonical_existing_path(source_path, label="capsule source path")
    first_path = _lstat(source)
    if (
        not stat.S_ISREG(first_path.st_mode)
        or stat.S_ISLNK(first_path.st_mode)
        or _is_reparse(first_path)
        or int(first_path.st_nlink) != 1
    ):
        raise CapsuleBuildError(f"capsule source is not a plain single-link file: {source}")
    streams_before = _named_streams(source)
    if streams_before:
        raise CapsuleBuildError(f"capsule source has alternate streams: {source}")

    descriptor = _open_read_no_follow(source)
    try:
        first_fd = os.fstat(descriptor)
        if (
            not stat.S_ISREG(first_fd.st_mode)
            or _is_reparse(first_fd)
            or int(first_fd.st_nlink) != 1
            or (int(first_fd.st_dev), int(first_fd.st_ino))
            != (int(first_path.st_dev), int(first_path.st_ino))
        ):
            raise CapsuleBuildError(f"capsule source changed during open: {source}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MEMBER_BYTES:
                raise CapsuleBuildError(f"capsule source exceeds byte limit: {source}")
            chunks.append(chunk)
        second_fd = os.fstat(descriptor)
        second_path = _lstat(source)
        streams_after = _named_streams(source)
        final_path = _lstat(source)
        final_fd = os.fstat(descriptor)
        if (
            _identity(first_fd) != _identity(second_fd)
            or _identity(first_fd) != _identity(final_fd)
            or not _same_file_object(first_fd, second_path)
            or not _same_file_object(first_fd, final_path)
            or streams_after
        ):
            raise CapsuleBuildError(f"capsule source changed during stable read: {source}")
        payload = b"".join(chunks)
        if len(payload) != int(first_fd.st_size):
            raise CapsuleBuildError(f"capsule source read length changed: {source}")
        return PayloadMember(
            source_path=source,
            relative_path=_validate_member_path(relative_path),
            role=_validate_role(role),
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=payload,
            identity=_identity(first_fd),
        )
    finally:
        os.close(descriptor)


def _project_role(relative_path: str) -> str:
    if relative_path in _SPECIAL_PROJECT_ROLES:
        return _SPECIAL_PROJECT_ROLES[relative_path]
    if relative_path.endswith("/__init__.py") or relative_path == "histo_audit/__init__.py":
        return "package_initializer"
    return "project_source"


def _scan_python_paths(package_root: Path) -> tuple[Path, ...]:
    root = _require_canonical_existing_path(package_root, label="package root")
    if root.name != "histo_audit" or not root.is_dir():
        raise CapsuleBuildError("package root must be the exact histo_audit directory")
    discovered: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        directory_stat = _lstat(directory)
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or _is_reparse(directory_stat)
        ):
            raise CapsuleBuildError(f"package directory is not plain: {directory}")
        if _named_streams(directory):
            raise CapsuleBuildError(f"package directory has alternate streams: {directory}")
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            path = Path(entry.path)
            value = _lstat(path)
            if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
                raise CapsuleBuildError(f"package tree contains a link/reparse point: {path}")
            if stat.S_ISDIR(value.st_mode):
                if entry.name == "__pycache__":
                    continue
                pending.append(path)
            elif entry.name.endswith(".py"):
                if not stat.S_ISREG(value.st_mode):
                    raise CapsuleBuildError(f"Python source is not regular: {path}")
                discovered.append(path)
    return tuple(sorted(discovered, key=lambda path: path.relative_to(root).as_posix()))


def discover_project_payload(
    *,
    package_root: str | Path,
    bootstrap_path: str | Path,
    policy_path: str | Path,
    entry_contract_path: str | Path,
) -> tuple[PayloadMember, ...]:
    root = _require_canonical_existing_path(package_root, label="package root")
    members: list[PayloadMember] = []
    for source in _scan_python_paths(root):
        relative = f"histo_audit/{source.relative_to(root).as_posix()}"
        members.append(
            _read_stable_regular_file(
                source,
                relative_path=relative,
                role=_project_role(relative),
            )
        )
    present = {member.relative_path for member in members}
    missing = sorted(_REQUIRED_PROJECT_MEMBERS - present)
    if missing:
        raise CapsuleBuildError(f"required capsule project members are absent: {missing}")
    members.extend(
        (
            _read_stable_regular_file(
                Path(bootstrap_path),
                relative_path="__main__.py",
                role="capsule_bootstrap",
            ),
            _read_stable_regular_file(
                Path(policy_path),
                relative_path="aanca_capsule/capsule_policy.json",
                role="capsule_policy",
            ),
            _read_stable_regular_file(
                Path(entry_contract_path),
                relative_path="aanca_capsule/entry_contract.json",
                role="capsule_contract",
            ),
        )
    )
    return _validate_member_set(members)


def _validate_member_set(members: Sequence[PayloadMember]) -> tuple[PayloadMember, ...]:
    ordered = tuple(sorted(members, key=lambda item: item.relative_path))
    if not ordered:
        raise CapsuleBuildError("capsule payload is empty")
    exact: set[str] = set()
    folded: set[str] = set()
    identities: set[tuple[int, int]] = set()
    for member in ordered:
        _validate_member_path(member.relative_path)
        _validate_role(member.role)
        if (
            type(member.size_bytes) is not int
            or member.size_bytes < 0
            or member.size_bytes != len(member.payload)
            or _SHA256.fullmatch(member.sha256) is None
            or hashlib.sha256(member.payload).hexdigest() != member.sha256
        ):
            raise CapsuleBuildError(f"invalid payload record: {member.relative_path}")
        folded_path = member.relative_path.casefold()
        identity = (member.identity.device, member.identity.inode)
        if member.relative_path in exact or folded_path in folded or identity in identities:
            raise CapsuleBuildError(
                f"duplicate capsule path or physical identity: {member.relative_path}"
            )
        exact.add(member.relative_path)
        folded.add(folded_path)
        identities.add(identity)
    return ordered


def source_inventory(members: Sequence[PayloadMember]) -> SourceInventory:
    ordered = _validate_member_set(members)
    records = tuple(member.manifest_record() for member in ordered)
    return SourceInventory(entries=records, records_root_sha256=_records_root(records))


def _require_inventory(
    members: Sequence[PayloadMember],
    expected_inventory: SourceInventory,
) -> tuple[PayloadMember, ...]:
    ordered = _validate_member_set(members)
    actual = source_inventory(ordered)
    if actual != expected_inventory:
        raise CapsuleBuildError("current source inventory differs from the frozen inventory")
    return ordered


def _zip_info(relative_path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative_path, FIXED_ZIP_DATETIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = FIXED_EXTERNAL_ATTR
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _manifest_bytes(members: Sequence[PayloadMember]) -> tuple[bytes, str, int]:
    records = [member.manifest_record() for member in members]
    root = _records_root(records)
    payload_size = sum(member.size_bytes for member in members)
    manifest = {
        "schema_version": 1,
        "policy": MANIFEST_POLICY,
        "archive_policy": dict(_ARCHIVE_POLICY),
        "entries": records,
        "entry_count": len(records),
        "payload_size_bytes": payload_size,
        "records_root_sha256": root,
    }
    return _canonical_json_line(manifest), root, payload_size


def _write_archive_bytes(
    members: Sequence[PayloadMember],
    manifest: bytes,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for member in members:
            archive.writestr(
                _zip_info(member.relative_path),
                member.payload,
                compress_type=zipfile.ZIP_STORED,
            )
        archive.writestr(
            _zip_info(MANIFEST_NAME),
            manifest,
            compress_type=zipfile.ZIP_STORED,
        )
    return stream.getvalue()


def _verify_archive_bytes(
    archive_bytes: bytes,
    *,
    members: Sequence[PayloadMember],
    manifest: bytes,
) -> None:
    expected_names = [member.relative_path for member in members] + [MANIFEST_NAME]
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r", allowZip64=False) as archive:
        if archive.comment != b"" or archive.namelist() != expected_names:
            raise CapsuleBuildError("archive order/comment differs from the capsule policy")
        infos = archive.infolist()
        for index, info in enumerate(infos):
            expected_name = expected_names[index]
            expected_bytes = manifest if expected_name == MANIFEST_NAME else members[index].payload
            if (
                info.filename != expected_name
                or info.date_time != FIXED_ZIP_DATETIME
                or info.compress_type != zipfile.ZIP_STORED
                or info.create_system != 3
                or info.external_attr != FIXED_EXTERNAL_ATTR
                or info.extra != b""
                or info.comment != b""
                or info.file_size != len(expected_bytes)
                or info.compress_size != len(expected_bytes)
                or archive.read(info) != expected_bytes
            ):
                raise CapsuleBuildError(f"archive entry violates fixed policy: {expected_name}")
        if archive.testzip() is not None:
            raise CapsuleBuildError("archive CRC verification failed")
        central_offset = int(archive.start_dir)
    expected_payloads = [member.payload for member in members] + [manifest]
    _verify_local_headers(
        archive_bytes,
        expected_names=expected_names,
        expected_payloads=expected_payloads,
        infos=infos,
        central_offset=central_offset,
    )


_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_HEADER_SIGNATURE: Final = 0x04034B50
_ZIP_VERSION_NEEDED: Final = 20
_FIXED_DOS_TIME: Final = 0
_FIXED_DOS_DATE: Final = 33


def _verify_local_headers(
    archive_bytes: bytes,
    *,
    expected_names: Sequence[str],
    expected_payloads: Sequence[bytes],
    infos: Sequence[zipfile.ZipInfo],
    central_offset: int,
) -> None:
    if not (len(expected_names) == len(expected_payloads) == len(infos) and central_offset >= 0):
        raise CapsuleBuildError("invalid local-header verification inputs")
    offset = 0
    for index, (expected_name, expected_payload, info) in enumerate(
        zip(expected_names, expected_payloads, infos, strict=True)
    ):
        header_end = offset + _LOCAL_HEADER.size
        if header_end > len(archive_bytes):
            raise CapsuleBuildError(f"truncated local header: {expected_name}")
        (
            signature,
            version_needed,
            flag_bits,
            compression,
            dos_time,
            dos_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
        ) = _LOCAL_HEADER.unpack_from(archive_bytes, offset)
        expected_filename = expected_name.encode("ascii")
        expected_crc32 = zlib.crc32(expected_payload) & 0xFFFFFFFF
        if (
            signature != _LOCAL_FILE_HEADER_SIGNATURE
            or version_needed != _ZIP_VERSION_NEEDED
            or flag_bits != 0
            or compression != zipfile.ZIP_STORED
            or dos_time != _FIXED_DOS_TIME
            or dos_date != _FIXED_DOS_DATE
            or crc32 != expected_crc32
            or compressed_size != len(expected_payload)
            or uncompressed_size != len(expected_payload)
            or filename_size != len(expected_filename)
            or extra_size != 0
            or int(info.header_offset) != offset
        ):
            raise CapsuleBuildError(f"archive local header violates fixed policy: {expected_name}")
        filename_start = header_end
        payload_start = filename_start + filename_size
        payload_end = payload_start + compressed_size
        if (
            payload_end > len(archive_bytes)
            or archive_bytes[filename_start:payload_start] != expected_filename
            or archive_bytes[payload_start:payload_end] != expected_payload
        ):
            raise CapsuleBuildError(
                f"archive local entry bytes violate fixed policy: {expected_name}"
            )
        offset = payload_end
        if index + 1 < len(expected_names) and offset >= central_offset:
            raise CapsuleBuildError("local entries overlap the central directory")
    if offset != central_offset:
        raise CapsuleBuildError("local-entry extent does not exactly meet the central directory")


def _validated_byte_build(build: CapsuleByteBuild) -> tuple[PayloadMember, ...]:
    if type(build) is not CapsuleByteBuild:
        raise CapsuleBuildError("capsule byte build has an unexpected type")
    ordered = _validate_member_set(build.members)
    manifest, records_root_sha256, payload_size_bytes = _manifest_bytes(ordered)
    expected_archive = _write_archive_bytes(ordered, manifest)
    if (
        build.manifest_bytes != manifest
        or build.archive_bytes != expected_archive
        or build.size_bytes != len(expected_archive)
        or build.sha256 != hashlib.sha256(expected_archive).hexdigest()
        or build.internal_manifest_sha256 != hashlib.sha256(manifest).hexdigest()
        or build.records_root_sha256 != records_root_sha256
        or build.entry_count != len(ordered)
        or build.payload_size_bytes != payload_size_bytes
    ):
        raise CapsuleBuildError("capsule byte build differs from its exact derivation")
    _verify_archive_bytes(expected_archive, members=ordered, manifest=manifest)
    return ordered


def build_capsule_bytes(
    *,
    members: Sequence[PayloadMember],
    expected_inventory: SourceInventory,
) -> CapsuleByteBuild:
    ordered = _require_inventory(members, expected_inventory)
    manifest, records_root_sha256, payload_size_bytes = _manifest_bytes(ordered)
    archive_bytes = _write_archive_bytes(ordered, manifest)
    _verify_archive_bytes(archive_bytes, members=ordered, manifest=manifest)
    result = CapsuleByteBuild(
        archive_bytes=archive_bytes,
        manifest_bytes=manifest,
        members=ordered,
        size_bytes=len(archive_bytes),
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        internal_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        records_root_sha256=records_root_sha256,
        entry_count=len(ordered),
        payload_size_bytes=payload_size_bytes,
    )
    _validated_byte_build(result)
    return result


def build_project_capsule_bytes(
    *,
    package_root: str | Path,
    bootstrap_path: str | Path,
    policy_path: str | Path,
    entry_contract_path: str | Path,
    expected_inventory: SourceInventory,
) -> CapsuleByteBuild:
    members = discover_project_payload(
        package_root=package_root,
        bootstrap_path=bootstrap_path,
        policy_path=policy_path,
        entry_contract_path=entry_contract_path,
    )
    return build_capsule_bytes(
        members=members,
        expected_inventory=expected_inventory,
    )


@dataclass(frozen=True, slots=True)
class _RetainedAncestor:
    path: Path
    token: int
    identity: tuple[int, str] | tuple[int, int]
    file_attributes: int
    windows_native: bool


def _windows_handle_facts(handle: int) -> tuple[int, str, int]:
    class _FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
        ]

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", ctypes.c_uint32),
            ("reparse_tag", ctypes.c_uint32),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    file_id = _FileIdInfo()
    attributes = _FileAttributeTagInfo()
    for information_class, target in ((18, file_id), (9, attributes)):
        if not get_information(
            ctypes.c_void_p(handle),
            information_class,
            ctypes.byref(target),
            ctypes.sizeof(target),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "GetFileInformationByHandleEx failed",
            )
    return (
        int(file_id.volume_serial_number),
        bytes(file_id.file_id.identifier).hex(),
        int(attributes.file_attributes),
    )


def _windows_open_plain_path(
    path: Path,
    *,
    directory: bool,
    desired_access: int,
    share_access: int,
    creation_disposition: int,
    create_read_only: bool,
) -> int:
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
    flags = 0x00200000
    if directory:
        flags |= 0x02000000
    if create_read_only:
        flags |= 0x00000001 | 0x80000000
    handle = create_file(
        _windows_api_path(path),
        desired_access,
        share_access,
        None,
        creation_disposition,
        flags,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(
            ctypes.get_last_error(),
            f"CreateFileW failed for {path}",
        )
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def _retained_ancestor_chain(parent: Path) -> tuple[_RetainedAncestor, ...]:
    canonical_parent = _require_canonical_existing_path(
        parent,
        label="capsule destination parent",
    )
    parent_value = _lstat(canonical_parent)
    if (
        not stat.S_ISDIR(parent_value.st_mode)
        or stat.S_ISLNK(parent_value.st_mode)
        or _is_reparse(parent_value)
    ):
        raise CapsuleBuildError("capsule destination parent is not one plain directory")
    paths = (*reversed(canonical_parent.parents), canonical_parent)
    retained: list[_RetainedAncestor] = []
    try:
        for path in paths:
            path_value = _lstat(path)
            if (
                not stat.S_ISDIR(path_value.st_mode)
                or stat.S_ISLNK(path_value.st_mode)
                or _is_reparse(path_value)
            ):
                raise CapsuleBuildError(f"capsule destination ancestor is not plain: {path}")
            if os.name == "nt":
                token = _windows_open_plain_path(
                    path,
                    directory=True,
                    desired_access=0x00000080,
                    share_access=0x00000001 | 0x00000002,
                    creation_disposition=3,
                    create_read_only=False,
                )
                try:
                    volume, file_id, attributes = _windows_handle_facts(token)
                    if (
                        not attributes & 0x00000010
                        or attributes & 0x00000400
                        or attributes != _file_attributes(_lstat(path))
                    ):
                        raise CapsuleBuildError(
                            f"capsule destination ancestor changed during lease: {path}"
                        )
                except BaseException:
                    _close_windows_handle(token)
                    raise
                retained.append(
                    _RetainedAncestor(
                        path=path,
                        token=token,
                        identity=(volume, file_id),
                        file_attributes=attributes,
                        windows_native=True,
                    )
                )
            else:
                flags = (
                    os.O_RDONLY
                    | int(getattr(os, "O_CLOEXEC", 0))
                    | int(getattr(os, "O_DIRECTORY", 0))
                    | int(getattr(os, "O_NOFOLLOW", 0))
                )
                token = os.open(path, flags)
                try:
                    value = os.fstat(token)
                    if not stat.S_ISDIR(value.st_mode) or _is_reparse(value):
                        raise CapsuleBuildError(
                            f"capsule destination ancestor changed during lease: {path}"
                        )
                except BaseException:
                    os.close(token)
                    raise
                retained.append(
                    _RetainedAncestor(
                        path=path,
                        token=token,
                        identity=(int(value.st_dev), int(value.st_ino)),
                        file_attributes=_file_attributes(value),
                        windows_native=False,
                    )
                )
        return tuple(retained)
    except BaseException:
        _close_retained_ancestors(tuple(retained))
        raise


def _require_retained_ancestors_unchanged(
    retained: Sequence[_RetainedAncestor],
) -> None:
    for item in retained:
        path_value = _lstat(item.path)
        if (
            not stat.S_ISDIR(path_value.st_mode)
            or stat.S_ISLNK(path_value.st_mode)
            or _is_reparse(path_value)
        ):
            raise CapsuleBuildError(f"capsule destination ancestor became non-plain: {item.path}")
        if item.windows_native:
            volume, file_id, attributes = _windows_handle_facts(item.token)
            observed_identity: tuple[int, str] | tuple[int, int] = (volume, file_id)
        else:
            value = os.fstat(item.token)
            observed_identity = (int(value.st_dev), int(value.st_ino))
            attributes = _file_attributes(value)
        if (
            observed_identity != item.identity
            or attributes != item.file_attributes
            or attributes != _file_attributes(path_value)
        ):
            raise CapsuleBuildError(f"capsule destination ancestor lease changed: {item.path}")


def _close_retained_ancestors(retained: Sequence[_RetainedAncestor]) -> None:
    first_error: BaseException | None = None
    for item in reversed(retained):
        try:
            if item.windows_native:
                _close_windows_handle(item.token)
            else:
                os.close(item.token)
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _require_exact_capsule_destination(
    output_path: str | Path,
    *,
    capsule_sha256: str,
) -> Path:
    try:
        supplied_text = os.fspath(output_path)
    except TypeError as error:
        raise CapsuleBuildError("capsule destination is not path-like") from error
    if type(supplied_text) is not str or not supplied_text or "\x00" in supplied_text:
        raise CapsuleBuildError("capsule destination is not one non-empty text path")
    destination = Path(supplied_text)
    if (
        not destination.is_absolute()
        or supplied_text != str(destination)
        or supplied_text != os.path.abspath(os.path.normpath(supplied_text))
    ):
        raise CapsuleBuildError("capsule destination is not exact absolute lexical canonical form")
    if (
        destination.name != CAPSULE_FILENAME
        or destination.parent.name != capsule_sha256
        or _SHA256.fullmatch(destination.parent.name) is None
        or destination.parent.parent.name != "execution_capsules"
        or destination.parent.parent.parent.name != "artifacts"
    ):
        raise CapsuleBuildError(
            "capsule destination is not the exact content-addressed project layout"
        )
    _require_canonical_existing_path(
        destination.parent,
        label="capsule content-address parent",
    )
    return destination


def _create_new_capsule_descriptor(destination: Path) -> int:
    flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_CLOEXEC", 0))
    if os.name != "nt":
        flags |= os.O_CREAT | os.O_EXCL | int(getattr(os, "O_NOFOLLOW", 0))
        return os.open(destination, flags, 0o444)

    import msvcrt

    handle = _windows_open_plain_path(
        destination,
        directory=False,
        desired_access=0x80000000 | 0x40000000,
        share_access=0x00000001,
        creation_disposition=1,
        create_read_only=True,
    )
    try:
        return msvcrt.open_osfhandle(handle, flags & ~(os.O_CREAT | os.O_EXCL))
    except BaseException:
        _close_windows_handle(handle)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset : offset + 1024 * 1024])
        if written <= 0:
            raise CapsuleBuildError("capsule CREATE_NEW write made no progress")
        offset += written


def _read_exact_descriptor(descriptor: int, *, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, expected_size - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > expected_size:
            raise CapsuleBuildError("published capsule exceeds its exact byte build")
    if total != expected_size:
        raise CapsuleBuildError("published capsule is shorter than its exact byte build")
    return b"".join(chunks)


def _descriptor_native_identity(descriptor: int) -> tuple[int, str] | tuple[int, int]:
    if os.name == "nt":
        import msvcrt

        handle = msvcrt.get_osfhandle(descriptor)
        volume, file_id, _attributes = _windows_handle_facts(handle)
        return volume, file_id
    value = os.fstat(descriptor)
    return int(value.st_dev), int(value.st_ino)


def _path_native_identity(path: Path) -> tuple[int, str] | tuple[int, int]:
    if os.name == "nt":
        handle = _windows_open_plain_path(
            path,
            directory=False,
            desired_access=0x80000000,
            share_access=0x00000001 | 0x00000002,
            creation_disposition=3,
            create_read_only=False,
        )
        try:
            volume, file_id, _attributes = _windows_handle_facts(handle)
            return volume, file_id
        finally:
            _close_windows_handle(handle)
    descriptor = _open_read_no_follow(path)
    try:
        return _descriptor_native_identity(descriptor)
    finally:
        os.close(descriptor)


def _is_read_only_file(value: os.stat_result) -> bool:
    if os.name == "nt":
        return bool(_file_attributes(value) & 0x00000001)
    return not bool(value.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def publish_capsule_create_new(
    *,
    build: CapsuleByteBuild,
    output_path: str | Path,
) -> CapsuleBuildResult:
    ordered = _validated_byte_build(build)
    destination = _require_exact_capsule_destination(
        output_path,
        capsule_sha256=build.sha256,
    )
    retained = _retained_ancestor_chain(destination.parent)
    descriptor = -1
    try:
        _require_retained_ancestors_unchanged(retained)
        try:
            descriptor = _create_new_capsule_descriptor(destination)
        except OSError as error:
            raise CapsuleBuildError(
                f"capsule CREATE_NEW publication failed: {destination}"
            ) from error
        created_identity = _descriptor_native_identity(descriptor)
        created_value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created_value.st_mode)
            or _is_reparse(created_value)
            or int(created_value.st_nlink) != 1
            or int(created_value.st_size) != 0
        ):
            raise CapsuleBuildError("new capsule leaf is not one empty plain file")
        _write_all(descriptor, build.archive_bytes)
        os.fsync(descriptor)
        readback = _read_exact_descriptor(
            descriptor,
            expected_size=build.size_bytes,
        )
        if readback != build.archive_bytes or hashlib.sha256(readback).hexdigest() != build.sha256:
            raise CapsuleBuildError("same-handle capsule readback differs from its build")
        _verify_archive_bytes(
            readback,
            members=ordered,
            manifest=build.manifest_bytes,
        )
        _require_retained_ancestors_unchanged(retained)
        canonical_destination = _require_canonical_existing_path(
            destination,
            label="published capsule",
        )
        path_value = _lstat(canonical_destination)
        descriptor_value = os.fstat(descriptor)
        streams = _named_streams(canonical_destination)
        if (
            created_identity != _descriptor_native_identity(descriptor)
            or created_identity != _path_native_identity(canonical_destination)
            or not stat.S_ISREG(path_value.st_mode)
            or stat.S_ISLNK(path_value.st_mode)
            or _is_reparse(path_value)
            or not stat.S_ISREG(descriptor_value.st_mode)
            or _is_reparse(descriptor_value)
            or int(path_value.st_nlink) != 1
            or int(descriptor_value.st_nlink) != 1
            or int(path_value.st_size) != build.size_bytes
            or int(descriptor_value.st_size) != build.size_bytes
            or not _same_file_object(path_value, descriptor_value)
            or not _is_read_only_file(path_value)
            or streams
        ):
            raise CapsuleBuildError(
                "published capsule path/handle identity violates its closed policy"
            )
        final_readback = _read_exact_descriptor(
            descriptor,
            expected_size=build.size_bytes,
        )
        if final_readback != readback:
            raise CapsuleBuildError("published capsule changed during final readback")
        _require_retained_ancestors_unchanged(retained)
        return CapsuleBuildResult(
            output_path=destination,
            size_bytes=build.size_bytes,
            sha256=build.sha256,
            internal_manifest_sha256=build.internal_manifest_sha256,
            records_root_sha256=build.records_root_sha256,
            entry_count=build.entry_count,
            payload_size_bytes=build.payload_size_bytes,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_retained_ancestors(retained)


__all__ = [
    "CAPSULE_FILENAME",
    "MANIFEST_NAME",
    "CapsuleBuildError",
    "CapsuleBuildResult",
    "CapsuleByteBuild",
    "FileIdentity",
    "PayloadMember",
    "SourceInventory",
    "build_capsule_bytes",
    "build_project_capsule_bytes",
    "discover_project_payload",
    "publish_capsule_create_new",
    "source_inventory",
]
