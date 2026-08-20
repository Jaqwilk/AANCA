"""Cross-process, never-overwrite publication primitives for PanNuke artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import TracebackType
from typing import Any, Literal, Self

from histo_audit.utils.run_tracking import (
    ARTIFACT_MANIFEST_FILENAME,
    IMMUTABLE_MARKER,
)

_WINDOWS_RESERVED_LEAVES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}

ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY = "anchored_physical_copy_no_overwrite_wof_lzx_v1"
WOF_LZX_MIN_FREE_MARGIN_BYTES = 10 * 1024**3

PhysicalCopyCompressor = Callable[[Path], None]
PhysicalCopyFreeSpaceProbe = Callable[[Path], int]


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_wof_lzx_compress_file(path: Path) -> None:
    """Apply one non-retrying WOF LZX transform to an already copied file."""

    if os.name != "nt":
        raise OSError("WOF LZX physical-copy compression is available only on Windows")
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root:
        raise OSError("SystemRoot is unavailable for the trusted compact.exe path")
    executable = Path(system_root) / "System32" / "compact.exe"
    try:
        executable_value = executable.stat(follow_symlinks=False)
        destination_value = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError(f"WOF LZX compressor prerequisite is unavailable: {exc}") from exc
    if (
        not stat.S_ISREG(executable_value.st_mode)
        or _is_link_or_reparse(executable, executable_value)
        or not stat.S_ISREG(destination_value.st_mode)
        or _is_link_or_reparse(path, destination_value)
    ):
        raise OSError("WOF LZX compressor executable/destination is not a regular plain file")
    completed = subprocess.run(
        [
            str(executable),
            "/c",
            "/exe:lzx",
            "/f",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise OSError(f"compact.exe WOF LZX failed with exit code {completed.returncode}{suffix}")


def _physical_copy_free_space(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size, value.st_ctime_ns)


def _lexical_final_path(path: str | Path) -> Path:
    """Resolve the parent only, preserving an existing or broken final symlink."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _absolute_lexical_path(path: str | Path) -> Path:
    """Return an absolute path without following any source component."""

    return Path(os.path.abspath(Path(path).expanduser()))


def _is_link_or_reparse(path: Path, value: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    attributes = int(getattr(value, "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return (
        path.is_symlink()
        or bool(reparse_flag and attributes & reparse_flag)
        or bool(is_junction is not None and is_junction())
    )


def assert_mutable_publication_destination(
    destination: str | Path,
    *,
    role: str = "publication destination",
) -> Path:
    """Reject reparse traversal and every sealed/immutable destination ancestor.

    The check deliberately uses the lexical absolute parent chain with
    ``follow_symlinks=False`` before resolving it.  Callers repeat it after parent
    creation and immediately around no-overwrite publication so an existing
    symlink/junction cannot redirect writes and a newly sealed ancestor fails
    closed.
    """

    supplied = Path(destination).expanduser()
    lexical = Path(os.path.abspath(supplied))
    parent = lexical.parent
    chain = (*reversed(parent.parents), parent)
    for candidate in chain:
        if not os.path.lexists(candidate):
            continue
        value = candidate.stat(follow_symlinks=False)
        if _is_link_or_reparse(candidate, value):
            raise PermissionError(
                f"{role} parent must not be a symlink/junction/reparse point: {candidate}"
            )
        if not stat.S_ISDIR(value.st_mode):
            raise NotADirectoryError(f"{role} parent is not a directory: {candidate}")
        for marker_name in (IMMUTABLE_MARKER, ARTIFACT_MANIFEST_FILENAME):
            marker = candidate / marker_name
            if os.path.lexists(marker):
                raise PermissionError(f"{role} is inside a sealed/immutable ancestor: {candidate}")
    return parent.resolve(strict=False) / lexical.name


def _publication_parent_key(path: Path) -> str:
    return os.path.normcase(str(Path(os.path.abspath(path))))


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


def _physical_copy_relative_parts(relative_path: str) -> tuple[str, ...]:
    """Return one canonical, Windows-safe relative path for anchored copying."""

    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError("physical-copy path must be canonical POSIX text")
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.as_posix() != relative_path
        or len(posix.parts) == 0
        or any(part in {".", ".."} for part in posix.parts)
        or ":" in relative_path
    ):
        raise ValueError(f"unsafe physical-copy relative path: {relative_path!r}")
    for part in posix.parts:
        _require_publication_leaf(part)
        basename = part.split(".", 1)[0].rstrip(" .").casefold()
        if (
            part[-1] in {" ", "."}
            or any(ord(character) < 32 for character in part)
            or any(character in '<>"|?*' for character in part)
            or basename in _WINDOWS_RESERVED_LEAVES
        ):
            raise ValueError(f"physical-copy path is not Windows-safe: {relative_path!r}")
    return tuple(posix.parts)


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
    create: bool = False,
    write: bool = False,
    delete_access: bool = False,
    share_write: bool = True,
    share_delete: bool = True,
    directory: bool = False,
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
    access = 0x0001 | 0x0080 | 0x00100000
    if write:
        access |= 0x0002 | 0x0100
        if directory:
            access |= 0x0004
    if delete_access:
        access |= 0x00010000
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
            0x10 if directory else 0x80,
            0x1 | (0x2 if share_write else 0) | (0x4 if share_delete else 0),
            2 if create else 1,
            (0x1 if directory else 0x40)
            | 0x20
            | (0x00200000 if (not directory or not create) else 0),
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


def _windows_final_path_for_handle(native_handle: int) -> str:
    """Return one normalised DOS/UNC final path for an already-open handle."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    get_final_path.restype = wintypes.DWORD
    required = int(get_final_path(native_handle, None, 0, 0))
    if required <= 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = int(get_final_path(native_handle, buffer, len(buffer), 0))
    if written <= 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _windows_final_path_for_descriptor(descriptor: int) -> str:
    import msvcrt

    return _windows_final_path_for_handle(msvcrt.get_osfhandle(descriptor))


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
    information_length = max(
        ctypes.sizeof(FileLinkInformation),
        offset + len(encoded),
    )
    allocation = ctypes.create_string_buffer(information_length)
    information = ctypes.cast(allocation, ctypes.POINTER(FileLinkInformation)).contents
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


def _windows_rename_relative_no_overwrite(
    source_descriptor: int,
    target_directory_handle: int,
    final_name: str,
) -> None:
    """Atomically rename an opened file relative to a pinned directory handle."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    _require_publication_leaf(final_name)

    class IoStatusUnion(ctypes.Union):
        _fields_ = (("status", wintypes.LONG), ("pointer", wintypes.LPVOID))

    class IoStatusBlock(ctypes.Structure):
        _fields_ = (("value", IoStatusUnion), ("information", ctypes.c_size_t))

    class FileRenameInformation(ctypes.Structure):
        _fields_ = (
            ("replace_if_exists", ctypes.c_ubyte),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.ULONG),
            ("file_name", wintypes.WCHAR * 1),
        )

    encoded = final_name.encode("utf-16-le")
    offset = FileRenameInformation.file_name.offset
    information_length = max(
        ctypes.sizeof(FileRenameInformation),
        offset + len(encoded),
    )
    allocation = ctypes.create_string_buffer(information_length)
    information = ctypes.cast(allocation, ctypes.POINTER(FileRenameInformation)).contents
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
            10,
        )
    )
    _windows_raise_ntstatus(
        status,
        f"NtSetInformationFile(FileRenameInformation,{final_name!r})",
        converter,
    )


def _windows_flush_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (wintypes.HANDLE,)
    flush_file_buffers.restype = wintypes.BOOL
    if not flush_file_buffers(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_flush_descriptor(descriptor: int) -> None:
    import msvcrt

    _windows_flush_handle(msvcrt.get_osfhandle(descriptor))


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


def _posix_link_open_descriptor(
    source_descriptor: int,
    target_directory_descriptor: int,
    target_name: str,
) -> None:
    """Link the already-open source inode without resolving its mutable name."""

    import ctypes

    _require_publication_leaf(target_name)
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    if (
        linkat(
            source_descriptor,
            b"",
            target_directory_descriptor,
            os.fsencode(target_name),
            0x1000,
        )
        == 0
    ):
        return
    first_code = ctypes.get_errno()
    proc_descriptor = f"/proc/self/fd/{source_descriptor}".encode()
    if (
        linkat(
            -100,
            proc_descriptor,
            target_directory_descriptor,
            os.fsencode(target_name),
            0x400,
        )
        != 0
    ):
        code = ctypes.get_errno()
        raise OSError(
            code,
            "descriptor-relative hard-link publication failed closed; "
            f"AT_EMPTY_PATH errno={first_code} and procfs fallback errno={code}",
            target_name,
        )


def _posix_rename_noreplace(
    source_directory_descriptor: int,
    source_name: str,
    target_directory_descriptor: int,
    target_name: str,
) -> None:
    """Atomically rename one anchored leaf while refusing an existing target."""

    import ctypes

    _require_publication_leaf(source_name)
    _require_publication_leaf(target_name)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        status = renameat2(
            source_directory_descriptor,
            os.fsencode(source_name),
            target_directory_descriptor,
            os.fsencode(target_name),
            1,  # RENAME_NOREPLACE
        )
    else:
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise RuntimeError(
                "atomic no-replace directory adoption is unavailable on this POSIX platform"
            )
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        status = renameatx_np(
            source_directory_descriptor,
            os.fsencode(source_name),
            target_directory_descriptor,
            os.fsencode(target_name),
            0x4,  # Darwin RENAME_EXCL
        )
    if status == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(code, os.strerror(code), target_name)
    if code == errno.ENOENT:
        raise FileNotFoundError(code, os.strerror(code), source_name)
    raise OSError(code, os.strerror(code), target_name)


def _posix_create_directory_noreplace(
    parent_descriptor: int,
    final_name: str,
) -> int:
    """Create/open a private directory, then atomically adopt its exact fd."""

    _require_publication_leaf(final_name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    private_name = f".histo-audit-publish-{secrets.token_hex(16)}"
    os.mkdir(private_name, mode=0o700, dir_fd=parent_descriptor)
    descriptor: int | None = None
    expected: os.stat_result | None = None
    published = False
    try:
        descriptor = os.open(private_name, flags, dir_fd=parent_descriptor)
        expected = os.fstat(descriptor)
        if not stat.S_ISDIR(expected.st_mode):
            raise RuntimeError("private publication root is not a directory")
        _posix_rename_noreplace(
            parent_descriptor,
            private_name,
            parent_descriptor,
            final_name,
        )
        published = True
        logical = os.stat(final_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(logical.st_mode) or (logical.st_dev, logical.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise RuntimeError(
                "atomically adopted publication directory changed during identity readback"
            )
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        cleanup_name = final_name if published else private_name
        try:
            value = os.stat(cleanup_name, dir_fd=parent_descriptor, follow_symlinks=False)
            if expected is not None and (
                value.st_dev,
                value.st_ino,
            ) == (expected.st_dev, expected.st_ino):
                os.rmdir(cleanup_name, dir_fd=parent_descriptor)
        except (FileNotFoundError, OSError):
            pass
        raise


def _posix_restore_quarantined_leaf(
    parent_descriptor: int,
    quarantine_name: str,
    final_name: str,
) -> None:
    try:
        _posix_rename_noreplace(
            parent_descriptor,
            quarantine_name,
            parent_descriptor,
            final_name,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "ownership mismatch was preserved in quarantine because the logical "
            f"publication path is occupied: {final_name}"
        ) from error


def _posix_quarantine_and_delete_file(
    parent_descriptor: int,
    final_name: str,
    expected_identity: tuple[int, int, int],
    expected_sha256: str,
) -> None:
    """Move a name aside atomically, verify its fd, then delete the quarantine.

    POSIX does not provide unlink-by-file-descriptor.  A random same-directory
    quarantine plus ``RENAME_NOREPLACE`` removes the verify/unlink window from the
    public name and makes a pre-quarantine replacement detectable/restorable.
    """

    quarantine_name = f".histo-audit-rollback-{secrets.token_hex(16)}"
    descriptor: int | None = None
    quarantined = False
    deleted = False
    try:
        _posix_rename_noreplace(
            parent_descriptor,
            final_name,
            parent_descriptor,
            quarantine_name,
        )
        quarantined = True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(quarantine_name, flags, dir_fd=parent_descriptor)
        value = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if (
            not stat.S_ISREG(value.st_mode)
            or (value.st_dev, value.st_ino, value.st_size) != expected_identity
            or digest.hexdigest() != expected_sha256
        ):
            raise RuntimeError(f"refused to remove unowned publication: {final_name}")
        logical = os.stat(
            quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (logical.st_dev, logical.st_ino, logical.st_size) != expected_identity:
            raise RuntimeError(f"publication quarantine changed before deletion: {final_name}")
        os.unlink(quarantine_name, dir_fd=parent_descriptor)
        deleted = True
        os.fsync(parent_descriptor)
    except BaseException as error:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
            descriptor = None
        if quarantined and not deleted:
            try:
                _posix_restore_quarantined_leaf(
                    parent_descriptor,
                    quarantine_name,
                    final_name,
                )
            except BaseException as restore_error:
                raise RuntimeError(
                    "publication rollback failed before ownership-safe quarantine "
                    f"restoration completed for {final_name}: {restore_error}"
                ) from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _posix_quarantine_and_delete_directory(
    parent_descriptor: int,
    final_name: str,
    expected_identity: tuple[int, int],
) -> None:
    quarantine_name = f".histo-audit-rollback-{secrets.token_hex(16)}"
    descriptor: int | None = None
    quarantined = False
    deleted = False
    try:
        _posix_rename_noreplace(
            parent_descriptor,
            final_name,
            parent_descriptor,
            quarantine_name,
        )
        quarantined = True
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(quarantine_name, flags, dir_fd=parent_descriptor)
        value = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(value.st_mode)
            or (value.st_dev, value.st_ino) != expected_identity
            or os.listdir(descriptor)
        ):
            raise RuntimeError(
                f"refused to remove unowned or non-empty publication directory: {final_name}"
            )
        logical = os.stat(
            quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (logical.st_dev, logical.st_ino) != expected_identity:
            raise RuntimeError(
                f"publication directory quarantine changed before deletion: {final_name}"
            )
        os.rmdir(quarantine_name, dir_fd=parent_descriptor)
        deleted = True
        os.fsync(parent_descriptor)
    except BaseException as error:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
            descriptor = None
        if quarantined and not deleted:
            try:
                _posix_restore_quarantined_leaf(
                    parent_descriptor,
                    quarantine_name,
                    final_name,
                )
            except BaseException as restore_error:
                raise RuntimeError(
                    "publication-directory rollback failed before ownership-safe quarantine "
                    f"restoration completed for {final_name}: {restore_error}"
                ) from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(slots=True)
class _RetainedPublicationHandle:
    descriptor: int | None
    identity: tuple[int, int, int]
    kind: Literal["file", "directory"]
    sha256: str | None

    def _sha256(self) -> str:
        if self.descriptor is None:
            raise RuntimeError("publication ownership handle is closed")
        digest = hashlib.sha256()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        while chunk := os.read(self.descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()

    def still_owned_object(self) -> bool:
        if self.descriptor is None:
            return False
        try:
            value = os.fstat(self.descriptor)
            if (value.st_dev, value.st_ino, value.st_size) != self.identity:
                return False
            if self.kind == "directory":
                return stat.S_ISDIR(value.st_mode)
            return stat.S_ISREG(value.st_mode) and self._sha256() == self.sha256
        except OSError:
            return False

    def delete_exact(self) -> None:
        if self.descriptor is None or not self.still_owned_object():
            raise RuntimeError("refused to remove publication without its exact owned handle")
        descriptor = self.descriptor
        try:
            if os.name != "nt":
                raise RuntimeError(
                    "exact handle deletion is unavailable on this POSIX publication record"
                )
            _windows_delete_opened_link(descriptor)
        finally:
            os.close(descriptor)
            self.descriptor = None

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None

    def __del__(self) -> None:
        with suppress(OSError):
            self.close()


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
            or _is_link_or_reparse(self.path, value)
            or (value.st_dev, value.st_ino) != self.identity
        ):
            raise RuntimeError(f"publication parent changed while locked: {self.path}")


def _create_locked_directory(
    parent: _LockedPublicationParent,
    target: Path,
) -> tuple[_LockedPublicationParent, _RetainedPublicationHandle | None, os.stat_result]:
    """Atomically create and retain the exact directory object, never reopening by name."""

    _require_publication_leaf(target.name)
    if parent.native_handle is not None:
        import msvcrt

        descriptor = _windows_open_relative_descriptor(
            parent.native_handle,
            target.name,
            create=True,
            write=True,
            # Omitting FILE_SHARE_DELETE is sufficient to pin the created
            # directory against rename/deletion.  Do not also request DELETE:
            # ordinary Python path readers do not share delete access and would
            # be locked out for the complete flat-publication transaction.
            delete_access=False,
            share_delete=False,
            directory=True,
        )
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            os.close(descriptor)
            raise RuntimeError(f"created publication root is not a directory: {target}")
        child = _LockedPublicationParent(
            path=target,
            identity=(value.st_dev, value.st_ino),
            descriptor=None,
            native_handle=msvcrt.get_osfhandle(descriptor),
        )
        retained = _RetainedPublicationHandle(
            descriptor=descriptor,
            identity=(value.st_dev, value.st_ino, value.st_size),
            kind="directory",
            sha256=None,
        )
        try:
            child.assert_current()
            _windows_flush_descriptor(descriptor)
            _windows_flush_handle(parent.native_handle)
        except BaseException as create_error:
            try:
                _rollback_created_directory(parent, target, retained)
            except (OSError, RuntimeError) as rollback_error:
                raise RuntimeError(
                    "created-directory durability check failed and stable rollback was "
                    f"incomplete: {target}: {rollback_error}"
                ) from create_error
            raise
        return child, retained, value

    assert parent.descriptor is not None
    descriptor = _posix_create_directory_noreplace(parent.descriptor, target.name)
    value = os.fstat(descriptor)
    child = _LockedPublicationParent(
        path=target,
        identity=(value.st_dev, value.st_ino),
        descriptor=descriptor,
        native_handle=None,
    )
    try:
        child.assert_current()
        os.fsync(descriptor)
        os.fsync(parent.descriptor)
    except BaseException:
        try:
            _posix_quarantine_and_delete_directory(
                parent.descriptor,
                target.name,
                child.identity,
            )
        finally:
            os.close(descriptor)
        raise
    return child, None, value


def _rollback_created_directory(
    parent: _LockedPublicationParent,
    target: Path,
    retained: _RetainedPublicationHandle,
) -> None:
    """Close a root guard, then verify and delete one stable reopened directory."""

    if parent.native_handle is None:
        raise RuntimeError("stable created-directory rollback is Windows-only")
    if not retained.still_owned_object():
        raise RuntimeError(f"created directory ownership changed before rollback: {target}")
    expected_identity = retained.identity[:2]
    retained.close()
    descriptor: int | None = None
    try:
        descriptor = _windows_open_relative_descriptor(
            parent.native_handle,
            target.name,
            delete_access=True,
            share_delete=False,
            directory=True,
        )
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode) or (value.st_dev, value.st_ino) != expected_identity:
            raise RuntimeError(f"refused to remove unowned publication directory: {target}")
        if os.listdir(target):
            raise RuntimeError(f"refused to remove non-empty publication directory: {target}")
        _windows_delete_opened_link(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if os.path.lexists(target):
        raise RuntimeError(
            f"exact owned directory was removed but a foreign logical publication remains: {target}"
        )


@dataclass(slots=True)
class _LockedPublishedFile:
    parent: _LockedPublicationParent
    path: Path
    name: str
    identity: tuple[int, int, int]
    sha256: str
    retained_descriptor: int | None = None
    retained_descriptor_delete_capable: bool = True

    def _open_descriptor(
        self,
        *,
        delete_access: bool = False,
        share_write: bool = True,
        share_delete: bool = True,
        write: bool = False,
    ) -> int:
        if self.retained_descriptor is not None:
            return os.dup(self.retained_descriptor)
        if self.parent.native_handle is not None:
            return _windows_open_relative_descriptor(
                self.parent.native_handle,
                self.name,
                write=write,
                delete_access=delete_access,
                share_write=share_write,
                share_delete=share_delete,
            )
        if self.parent.descriptor is None:
            raise RuntimeError(f"publication parent is not anchored: {self.parent.path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self.name, flags, dir_fd=self.parent.descriptor)

    def _stat(self) -> os.stat_result:
        if self.parent.native_handle is None:
            assert self.parent.descriptor is not None
            return os.stat(self.name, dir_fd=self.parent.descriptor, follow_symlinks=False)
        descriptor = self._open_descriptor()
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def _sha256(self) -> str:
        descriptor = self._open_descriptor()
        digest = hashlib.sha256()
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
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
            if self.retained_descriptor is not None and not self.retained_descriptor_delete_capable:
                retained = self.retained_descriptor
                digest = hashlib.sha256()
                os.lseek(retained, 0, os.SEEK_SET)
                retained_value = os.fstat(retained)
                while chunk := os.read(retained, 1024 * 1024):
                    digest.update(chunk)
                if (
                    not stat.S_ISREG(retained_value.st_mode)
                    or (
                        retained_value.st_dev,
                        retained_value.st_ino,
                        retained_value.st_size,
                    )
                    != self.identity
                    or digest.hexdigest() != self.sha256
                ):
                    raise RuntimeError(
                        f"refused to reopen changed guarded publication: {self.path}"
                    )
                os.close(retained)
                self.retained_descriptor = None
                descriptor = self._open_descriptor(
                    delete_access=True,
                    share_write=False,
                    share_delete=False,
                )
            else:
                descriptor = (
                    self.retained_descriptor
                    if self.retained_descriptor is not None
                    else self._open_descriptor(
                        delete_access=True,
                        share_write=False,
                        share_delete=False,
                    )
                )
            digest = hashlib.sha256()
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
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
                if descriptor == self.retained_descriptor:
                    self.retained_descriptor = None
            return
        assert self.parent.descriptor is not None
        _posix_quarantine_and_delete_file(
            self.parent.descriptor,
            self.name,
            self.identity,
            self.sha256,
        )

    def close_retained_descriptor(self) -> None:
        """Release the transaction-local descriptor before returning to callers.

        A Windows handle granted ``DELETE`` access prevents ordinary Python readers
        (whose share mask omits ``FILE_SHARE_DELETE``) from reopening the published
        file.  Retaining it beyond the publication transaction therefore breaks the
        public artifact contract.  Rollback obtains a fresh, no-share delete handle
        and verifies that same opened object instead.
        """

        if self.retained_descriptor is not None:
            os.close(self.retained_descriptor)
            self.retained_descriptor = None


def _locked_path_exists(path: Path, parents: Mapping[str, _LockedPublicationParent]) -> bool:
    parent = parents[_publication_parent_key(path.parent)]
    parent.assert_current()
    try:
        if parent.native_handle is not None:
            # The complete parent chain is held without FILE_SHARE_DELETE, so
            # this lexical lookup cannot be redirected.  ``lexists`` also
            # recognises directories and reparse entries, whereas the native
            # file descriptor helper intentionally accepts regular files only.
            return os.path.lexists(path)
        else:
            assert parent.descriptor is not None
            os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _capture_locked_file(
    path: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> _LockedPublishedFile:
    parent = parents[_publication_parent_key(path.parent)]
    provisional = _LockedPublishedFile(parent, path, path.name, (0, 0, 0), "")
    value = provisional._stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"staged publication source is not a regular file: {path}")
    captured = _LockedPublishedFile(
        parent=parent,
        path=path,
        name=path.name,
        identity=(value.st_dev, value.st_ino, value.st_size),
        sha256=provisional._sha256(),
    )
    if not captured.still_owned():
        raise RuntimeError(f"staged publication source changed during capture: {path}")
    return captured


def _link_locked_file(
    staged: _LockedPublishedFile,
    target_parent: _LockedPublicationParent,
    target_name: str,
) -> None:
    source_descriptor = staged._open_descriptor(
        delete_access=target_parent.native_handle is not None,
        share_delete=True,
    )
    try:
        source_value = os.fstat(source_descriptor)
        digest = hashlib.sha256()
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
        if (
            not stat.S_ISREG(source_value.st_mode)
            or (source_value.st_dev, source_value.st_ino, source_value.st_size) != staged.identity
            or digest.hexdigest() != staged.sha256
        ):
            raise RuntimeError(
                f"anchored staging descriptor changed before publication: {staged.path}"
            )
        if target_parent.native_handle is not None:
            _windows_link_relative(
                source_descriptor,
                target_parent.native_handle,
                target_name,
            )
        else:
            assert target_parent.descriptor is not None
            _posix_link_open_descriptor(
                source_descriptor,
                target_parent.descriptor,
                target_name,
            )
    except OSError as error:
        if error.errno == errno.EXDEV or getattr(error, "winerror", None) == 17:
            raise OSError(
                errno.EXDEV,
                "anchored publication is fail-closed across volumes; staged and "
                "destination parents must be on the same volume",
            ) from error
        raise
    finally:
        os.close(source_descriptor)


def _publish_file_to_locked_parent(
    staged: _LockedPublishedFile,
    target: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> _LockedPublishedFile:
    parent = parents[_publication_parent_key(target.parent)]
    parent.assert_current()
    if _locked_path_exists(target, parents):
        raise FileExistsError(f"refusing to overwrite publication path: {target}")
    if not staged.still_owned():
        raise RuntimeError(f"anchored staging file changed before publication: {staged.path}")
    source_value = staged._stat()
    _link_locked_file(staged, parent, target.name)
    if parent.native_handle is not None:
        retained_descriptor = _windows_open_relative_descriptor(
            parent.native_handle,
            target.name,
            write=True,
            delete_access=True,
            share_delete=True,
        )
    else:
        assert parent.descriptor is not None
        retained_descriptor = os.open(
            target.name,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent.descriptor,
        )
    target_value = os.fstat(retained_descriptor)
    result = _LockedPublishedFile(
        parent=parent,
        path=target,
        name=target.name,
        identity=(target_value.st_dev, target_value.st_ino, target_value.st_size),
        sha256=staged.sha256,
        retained_descriptor=retained_descriptor,
    )
    source_identity = (source_value.st_dev, source_value.st_ino, source_value.st_size)
    if result.identity != source_identity:
        os.close(retained_descriptor)
        result.retained_descriptor = None
        raise RuntimeError(
            "published name no longer identifies the linked staging inode; foreign path "
            f"preserved: {target}"
        )
    try:
        parent.assert_current()
        if not result.still_owned():
            raise OSError(f"published file failed identity/hash readback: {target}")
        if parent.native_handle is not None:
            logical_value = target.stat(follow_symlinks=False)
            _windows_flush_descriptor(retained_descriptor)
            _windows_flush_handle(parent.native_handle)
            logical_after = target.stat(follow_symlinks=False)
        else:
            assert parent.descriptor is not None
            logical_value = os.stat(
                target.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            os.fsync(retained_descriptor)
            os.fsync(parent.descriptor)
            logical_after = os.stat(
                target.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        if (
            logical_value.st_dev,
            logical_value.st_ino,
            logical_value.st_size,
        ) != source_identity or (
            logical_after.st_dev,
            logical_after.st_ino,
            logical_after.st_size,
        ) != source_identity:
            raise RuntimeError(f"published logical name changed during durable readback: {target}")
        return result
    except BaseException as publish_error:
        try:
            result.unlink_owned()
        except (OSError, RuntimeError) as rollback_error:
            raise RuntimeError(
                "post-link publication failed and exact-handle rollback was incomplete: "
                f"{target}: {rollback_error}"
            ) from publish_error
        if os.path.lexists(target):
            raise RuntimeError(
                "post-link publication failed; exact owned object was removed but a foreign "
                f"logical destination remains: {target}"
            ) from publish_error
        raise


@contextmanager
def _locked_publication_parents(
    paths: Sequence[Path],
    *,
    final_paths: Sequence[Path],
    create_missing: bool = True,
    read_only_paths: Sequence[Path] = (),
) -> Iterator[dict[str, _LockedPublicationParent]]:
    for path in final_paths:
        assert_mutable_publication_destination(path)
    unique_paths = {_publication_parent_key(path): Path(os.path.abspath(path)) for path in paths}
    read_only_keys = {_publication_parent_key(path) for path in read_only_paths}
    if not read_only_keys.issubset(unique_paths):
        raise ValueError("read-only publication anchors must be included in paths")
    parents: dict[str, _LockedPublicationParent] = {}
    descriptors: list[int] = []
    windows_chain_descriptors: list[int] = []
    windows_chain_handles: list[int] = []
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
                import msvcrt

                request_write = key not in read_only_keys
                parts = parent_path.parts
                candidate = Path(parts[0])
                root_handle = create_file(
                    str(candidate),
                    read_attributes,
                    share_read_write_without_delete,
                    None,
                    open_existing,
                    backup_semantics | open_reparse_point,
                    None,
                )
                if root_handle == invalid_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                numeric_root_handle = int(root_handle)
                windows_chain_handles.append(numeric_root_handle)
                current_handle = numeric_root_handle
                value = candidate.stat(follow_symlinks=False)
                if not stat.S_ISDIR(value.st_mode) or _is_link_or_reparse(candidate, value):
                    raise RuntimeError(
                        f"publication parent path contains a reparse point: {candidate}"
                    )

                for part in parts[1:]:
                    candidate /= part
                    created = False
                    parent_handle = current_handle
                    try:
                        descriptor = _windows_open_relative_descriptor(
                            parent_handle,
                            part,
                            write=request_write,
                            share_delete=False,
                            directory=True,
                        )
                    except FileNotFoundError:
                        if not create_missing or key in read_only_keys:
                            raise FileNotFoundError(
                                f"publication parent is missing: {candidate}"
                            ) from None
                        try:
                            descriptor = _windows_open_relative_descriptor(
                                parent_handle,
                                part,
                                create=True,
                                write=request_write,
                                share_delete=False,
                                directory=True,
                            )
                            created = True
                        except FileExistsError as error:
                            raise RuntimeError(
                                "publication parent appeared concurrently after the "
                                f"anchored absence check; refusing adoption: {candidate}"
                            ) from error
                    windows_chain_descriptors.append(descriptor)
                    value = os.fstat(descriptor)
                    if not stat.S_ISDIR(value.st_mode) or _is_link_or_reparse(candidate, value):
                        raise RuntimeError(
                            f"publication parent path contains a reparse point: {candidate}"
                        )
                    if created:
                        _windows_flush_descriptor(descriptor)
                        _windows_flush_handle(parent_handle)
                    current_handle = msvcrt.get_osfhandle(descriptor)

                parents[key] = _LockedPublicationParent(
                    path=parent_path,
                    identity=(value.st_dev, value.st_ino),
                    descriptor=None,
                    native_handle=current_handle,
                )
            # Keep every no-delete-sharing chain handle alive until the complete
            # publication transaction exits.  Closing these here would reopen a
            # path-redirection window around later name-based Windows APIs.
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
                        try:
                            next_descriptor = os.open(
                                part,
                                directory_flags,
                                dir_fd=descriptor,
                            )
                        except FileNotFoundError:
                            if not create_missing or key in read_only_keys:
                                raise
                            try:
                                next_descriptor = _posix_create_directory_noreplace(
                                    descriptor,
                                    part,
                                )
                            except FileExistsError as error:
                                raise RuntimeError(
                                    "publication parent appeared concurrently after the "
                                    f"anchored absence check; refusing adoption: {parent_path}"
                                ) from error
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
        for path in final_paths:
            assert_mutable_publication_destination(path)
        yield parents
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        for descriptor in reversed(windows_chain_descriptors):
            with suppress(OSError):
                os.close(descriptor)
        if close_windows_handle is not None:
            for handle in reversed(windows_chain_handles):
                close_windows_handle(handle)


class ExclusivePublicationLock:
    """One non-blocking OS-held lock; crashes release it without stale-lock deletion."""

    def __init__(self, path: str | Path, *, role: str) -> None:
        self.logical_path = Path(path).resolve()
        self.role = role
        key = hashlib.sha256(str(self.logical_path).casefold().encode("utf-8")).hexdigest()
        self.path = Path(tempfile.gettempdir()) / "histo-audit-publication-locks" / f"{key}.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl: Any = __import__("fcntl")

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            raise FileExistsError(
                f"another {self.role} publication is active: {self.logical_path}"
            ) from error
        self._descriptor = descriptor
        return self

    def assert_owned(self) -> None:
        """Fail without mutation if this process no longer holds a valid descriptor."""

        if self._descriptor is None:
            raise RuntimeError(f"{self.role} publication lock is not held")
        try:
            os.fstat(self._descriptor)
        except OSError as error:
            raise RuntimeError(f"{self.role} publication lock descriptor is invalid") from error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc, traceback
        self.assert_owned()
        assert self._descriptor is not None
        descriptor = self._descriptor
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl: Any = __import__("fcntl")

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            self._descriptor = None
        return False


class ExclusiveBundlePublicationLock:
    """Create order-independent O_EXCL locks for a complete artifact bundle.

    The lock file is deliberately never reclaimed automatically.  A process crash
    therefore leaves a visible blocker that must be investigated instead of letting
    a later publisher guess that the in-flight transaction was safe to supersede.
    One aggregate lock serialises identical bundles, while one lock per canonical
    destination makes partially overlapping bundles conflict as well.  Every
    descriptor remains open for the complete context.  Cleanup removes a lock only
    after its path, inode identity, and random ownership token still match this
    context.
    """

    def __init__(self, paths: Sequence[str | Path], *, role: str) -> None:
        if not paths:
            raise ValueError("a bundle publication lock requires at least one path")
        canonical_paths = tuple(
            sorted(
                {str(_lexical_final_path(path)).casefold() for path in paths},
            )
        )
        self.logical_paths = canonical_paths
        self.role = role
        key_payload = "\0".join(canonical_paths).encode("utf-8")
        key = hashlib.sha256(key_payload).hexdigest()
        registry = Path(tempfile.gettempdir()) / "histo-audit-publication-locks"
        self.path = registry / f"bundle-{key}.lock"
        constituent_paths = tuple(
            registry / ("target-" + hashlib.sha256(path.encode("utf-8")).hexdigest() + ".lock")
            for path in canonical_paths
        )
        self.lock_paths = (self.path, *constituent_paths)
        self._token = secrets.token_hex(32)
        self._payload = f"{self._token}\n".encode("ascii")
        self._records: list[tuple[Path, int, tuple[int, int]]] = []

    def _path_owned(self, path: Path, identity: tuple[int, int]) -> bool:
        if not os.path.lexists(path):
            return False
        try:
            value = path.stat(follow_symlinks=False)
            if (value.st_dev, value.st_ino) != identity:
                return False
            return path.read_bytes() == self._payload
        except OSError:
            return False

    def _close_and_cleanup(self) -> list[str]:
        records = self._records
        self._records = []
        if os.name == "nt" and records:
            return self._close_and_cleanup_windows(records)
        errors: list[str] = []
        for _, descriptor, _ in records:
            with suppress(OSError):
                os.close(descriptor)
        if not records:
            return errors
        parent_path = records[0][0].parent
        try:
            with _locked_publication_parents(
                (parent_path,),
                final_paths=(),
                create_missing=False,
            ) as parents:
                parent = parents[_publication_parent_key(parent_path)]
                assert parent.descriptor is not None
                payload_sha256 = hashlib.sha256(self._payload).hexdigest()
                for path, _, identity in reversed(records):
                    try:
                        _posix_quarantine_and_delete_file(
                            parent.descriptor,
                            path.name,
                            (identity[0], identity[1], len(self._payload)),
                            payload_sha256,
                        )
                    except (OSError, RuntimeError) as error:
                        errors.append(f"{path}: {error}")
        except (FileNotFoundError, RuntimeError) as error:
            errors.append(str(error))
        return errors

    def _close_and_cleanup_windows(
        self,
        records: list[tuple[Path, int, tuple[int, int]]],
    ) -> list[str]:
        """Delete each lock through the exact stable object verified after reopen."""

        errors: list[str] = []
        open_descriptors = {descriptor for _, descriptor, _ in records}
        parent_path = records[0][0].parent
        try:
            with _locked_publication_parents(
                (parent_path,),
                final_paths=(),
                create_missing=False,
            ) as parents:
                parent = parents[_publication_parent_key(parent_path)]
                assert parent.native_handle is not None
                for path, original_descriptor, identity in reversed(records):
                    with suppress(OSError):
                        os.close(original_descriptor)
                    open_descriptors.discard(original_descriptor)
                    stable_descriptor: int | None = None
                    try:
                        stable_descriptor = _windows_open_relative_descriptor(
                            parent.native_handle,
                            path.name,
                            delete_access=True,
                            share_delete=False,
                        )
                        value = os.fstat(stable_descriptor)
                        payload = bytearray()
                        os.lseek(stable_descriptor, 0, os.SEEK_SET)
                        while chunk := os.read(stable_descriptor, 4096):
                            payload.extend(chunk)
                        if (
                            not stat.S_ISREG(value.st_mode)
                            or (value.st_dev, value.st_ino) != identity
                            or bytes(payload) != self._payload
                        ):
                            errors.append(f"refused to remove unowned bundle lock: {path}")
                            continue
                        _windows_delete_opened_link(stable_descriptor)
                    except (OSError, RuntimeError) as error:
                        errors.append(f"{path}: {error}")
                        continue
                    finally:
                        if stable_descriptor is not None:
                            with suppress(OSError):
                                os.close(stable_descriptor)
                    if os.path.lexists(path):
                        errors.append(
                            "exact owned bundle lock was removed but a foreign logical "
                            f"lock remains: {path}"
                        )
        except (FileNotFoundError, RuntimeError) as error:
            errors.append(str(error))
        finally:
            for descriptor in open_descriptors:
                with suppress(OSError):
                    os.close(descriptor)
        return errors

    def __enter__(self) -> Self:
        if self._records:
            raise RuntimeError(f"{self.role} bundle publication lock is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        try:
            for lock_path in self.lock_paths:
                try:
                    descriptor = os.open(lock_path, flags, 0o600)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"another {self.role} publication is active or requires "
                        f"stale-lock review: {lock_path}"
                    ) from error
                value = os.fstat(descriptor)
                identity = (value.st_dev, value.st_ino)
                self._records.append((lock_path, descriptor, identity))
                remaining = memoryview(self._payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError(f"failed to initialise bundle lock: {lock_path}")
                    remaining = remaining[written:]
                os.fsync(descriptor)
        except FileExistsError as error:
            cleanup_errors = self._close_and_cleanup()
            if cleanup_errors:
                raise RuntimeError(
                    "bundle lock acquisition failed and ownership-safe cleanup was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from error
            raise
        except BaseException as error:
            cleanup_errors = self._close_and_cleanup()
            if cleanup_errors:
                raise RuntimeError(
                    "bundle lock initialisation failed and ownership-safe cleanup was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from error
            raise
        return self

    def assert_owned(self) -> None:
        """Require every open descriptor and lock path to retain this ownership."""

        if len(self._records) != len(self.lock_paths):
            raise RuntimeError(f"{self.role} bundle publication lock is not held")
        for path, descriptor, identity in self._records:
            try:
                descriptor_value = os.fstat(descriptor)
            except OSError as error:
                raise RuntimeError(
                    f"{self.role} bundle publication lock descriptor is invalid: {path}"
                ) from error
            descriptor_identity = (descriptor_value.st_dev, descriptor_value.st_ino)
            if descriptor_identity != identity or not self._path_owned(path, identity):
                raise RuntimeError(f"{self.role} bundle publication lock ownership changed: {path}")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, traceback
        cleanup_errors = self._close_and_cleanup()
        if cleanup_errors:
            cleanup_error = RuntimeError(
                f"{self.role} bundle publication lock cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            )
            if exc is not None:
                raise cleanup_error from exc
            raise cleanup_error
        return False


@dataclass(frozen=True, slots=True)
class PublishedPath:
    """Identity proof for one final path created by the current transaction."""

    path: Path
    identity: tuple[int, int, int, int]
    kind: Literal["file", "directory"]
    sha256: str | None = None
    required_nlink: int | None = None

    def still_owned(self) -> bool:
        try:
            value = self.path.stat(follow_symlinks=False)
            current = (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_ctime_ns,
            )
            if self.required_nlink is not None and value.st_nlink != self.required_nlink:
                return False
            if self.kind == "directory":
                return stat.S_ISDIR(value.st_mode) and current[:2] == self.identity[:2]
            if current[:3] != self.identity[:3]:
                return False
            if self.kind == "file":
                return stat.S_ISREG(value.st_mode) and _sha256(self.path) == self.sha256
            return False
        except OSError:
            return False


class AnchoredPhysicalCopyBoundaryError(RuntimeError):
    """Final copy-boundary failure that forbids pathname-based recovery.

    The destination path may no longer name the directory held by the copy
    session.  Callers must therefore treat the expected destination as unsealed
    and default-deny instead of trying to demote, fail, or seal it by pathname.
    """

    def __init__(
        self,
        *,
        source_tree_current: bool,
        destination_tree_current: bool,
        rollback_complete: bool,
        expected_destination_root: Path,
        boundary_errors: Sequence[str] = (),
        rollback_errors: Sequence[str] = (),
    ) -> None:
        self.source_tree_current = source_tree_current
        self.destination_tree_current = destination_tree_current
        self.rollback_complete = rollback_complete
        self.expected_destination_root = expected_destination_root
        self.boundary_errors = tuple(boundary_errors)
        self.rollback_errors = tuple(rollback_errors)
        state = (
            f"source_tree_current={source_tree_current}, "
            f"destination_tree_current={destination_tree_current}, "
            f"rollback_complete={rollback_complete}"
        )
        details = "; ".join((*self.boundary_errors, *self.rollback_errors))
        suffix = f": {details}" if details else ""
        super().__init__(
            "anchored physical-copy boundary failed; pathname-based recovery is unsafe "
            f"for {expected_destination_root} ({state}){suffix}"
        )


@dataclass(slots=True)
class _CreatedPhysicalCopyDirectory:
    parent: _LockedPublicationParent
    child: _LockedPublicationParent
    path: Path
    retained: _RetainedPublicationHandle | None


def _descriptor_sha256(descriptor: int, *, chunk_size_bytes: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, chunk_size_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def _stream_physical_copy(
    source_descriptor: int,
    destination_descriptor: int,
    *,
    chunk_size_bytes: int,
) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(source_descriptor, chunk_size_bytes):
        digest.update(chunk)
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(destination_descriptor, remaining)
            if written <= 0:
                raise OSError("physical-copy destination made no forward progress")
            remaining = remaining[written:]
    return digest.hexdigest()


def _stable_source_snapshot(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _flush_physical_copy_destination(
    destination_descriptor: int,
    parent: _LockedPublicationParent,
) -> None:
    os.fsync(destination_descriptor)
    if parent.native_handle is not None:
        _windows_flush_handle(parent.native_handle)
    else:
        assert parent.descriptor is not None
        os.fsync(parent.descriptor)


class AnchoredPhysicalCopySession:
    """Physically copy relative files while retaining both root handle chains."""

    def __init__(
        self,
        source_root: Path,
        destination_root: Path,
        source_anchor: _LockedPublicationParent,
        destination_anchor: _LockedPublicationParent,
        *,
        chunk_size_bytes: int,
        compression_policy: str | None,
        compressor: PhysicalCopyCompressor | None,
        free_space_probe: PhysicalCopyFreeSpaceProbe | None,
    ) -> None:
        self.source_root = source_root
        self.destination_root = destination_root
        self.chunk_size_bytes = chunk_size_bytes
        self.compression_policy = compression_policy
        self._compressor = compressor
        self._free_space_probe = free_space_probe
        self._source_parents: dict[tuple[str, ...], _LockedPublicationParent] = {(): source_anchor}
        self._destination_parents: dict[tuple[str, ...], _LockedPublicationParent] = {
            (): destination_anchor
        }
        self._owned_directory_descriptors: list[int] = []
        self._retained_directory_handles: list[_RetainedPublicationHandle] = []
        self._completed_files: list[_LockedPublishedFile] = []
        self._created_directories: list[_CreatedPhysicalCopyDirectory] = []
        self._closed = False
        self._source_final_path: str | None = None
        self._destination_final_path: str | None = None
        if os.name == "nt":
            if source_anchor.native_handle is None or destination_anchor.native_handle is None:
                raise RuntimeError("Windows physical-copy roots lack native handles")
            self._source_final_path = _windows_final_path_for_handle(source_anchor.native_handle)
            self._destination_final_path = _windows_final_path_for_handle(
                destination_anchor.native_handle
            )
            if self._source_final_path != os.path.normcase(os.path.normpath(str(source_root))):
                raise RuntimeError("physical-copy source root final path differs from its anchor")
            if self._destination_final_path != os.path.normcase(
                os.path.normpath(str(destination_root))
            ):
                raise RuntimeError(
                    "physical-copy destination root final path differs from its anchor"
                )
        if source_anchor.identity == destination_anchor.identity:
            raise ValueError("physical-copy source and destination roots must be distinct")
        self._assert_all_current()

    def _require_compression_space(
        self,
        *,
        next_logical_size_bytes: int,
        phase: str,
    ) -> int:
        if self._free_space_probe is None:
            raise RuntimeError("WOF LZX copy lacks its mandatory free-space probe")
        observed = self._free_space_probe(self.destination_root)
        if type(observed) is not int or observed < 0:
            raise RuntimeError("WOF LZX free-space probe returned an invalid byte count")
        required = WOF_LZX_MIN_FREE_MARGIN_BYTES + next_logical_size_bytes
        if observed < required:
            raise OSError(
                f"WOF LZX free-space guard failed {phase}: "
                f"observed={observed}, required={required}, "
                f"next_logical_size={next_logical_size_bytes}, "
                f"margin={WOF_LZX_MIN_FREE_MARGIN_BYTES}"
            )
        return observed

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("anchored physical-copy session is closed")

    def _expected_windows_path(self, *, source: bool, parts: Sequence[str]) -> str:
        root = self._source_final_path if source else self._destination_final_path
        if root is None:
            raise RuntimeError("Windows final-path evidence is unavailable")
        return os.path.normcase(os.path.normpath(str(Path(root).joinpath(*parts))))

    def _assert_windows_descriptor_path(
        self,
        descriptor: int,
        *,
        source: bool,
        parts: Sequence[str],
    ) -> None:
        if os.name == "nt" and _windows_final_path_for_descriptor(
            descriptor
        ) != self._expected_windows_path(source=source, parts=parts):
            raise RuntimeError("opened physical-copy object has an unexpected final path")

    def _assert_root_current(self, *, source: bool) -> None:
        self._assert_open()
        parent = self._source_parents[()] if source else self._destination_parents[()]
        parent.assert_current()
        if os.name == "nt":
            assert parent.native_handle is not None
            expected = self._source_final_path if source else self._destination_final_path
            if _windows_final_path_for_handle(parent.native_handle) != expected:
                role = "source" if source else "destination"
                raise RuntimeError(f"physical-copy {role} root final path changed while anchored")

    def _assert_tree_current(self, *, source: bool) -> None:
        self._assert_root_current(source=source)
        parents = self._source_parents if source else self._destination_parents
        root = parents[()]
        seen: set[int] = set()
        for parent in parents.values():
            if parent is root or id(parent) in seen:
                continue
            seen.add(id(parent))
            parent.assert_current()
        if not source:
            for completed in self._completed_files:
                if not completed.still_owned():
                    raise RuntimeError(
                        "completed physical-copy destination changed before final "
                        f"boundary readback: {completed.path}"
                    )

    def _assert_roots_current(self) -> None:
        self._assert_root_current(source=True)
        self._assert_root_current(source=False)

    def _assert_all_current(self) -> None:
        self._assert_tree_current(source=True)
        self._assert_tree_current(source=False)

    def _open_existing_directory(
        self,
        parent: _LockedPublicationParent,
        path: Path,
        *,
        source: bool,
        parts: tuple[str, ...],
    ) -> _LockedPublicationParent:
        if parent.native_handle is not None:
            import msvcrt

            descriptor = _windows_open_relative_descriptor(
                parent.native_handle,
                path.name,
                write=not source,
                share_delete=False,
                directory=True,
            )
            try:
                value = os.fstat(descriptor)
                if not stat.S_ISDIR(value.st_mode) or _is_link_or_reparse(path, value):
                    raise RuntimeError(f"physical-copy path contains a reparse point: {path}")
                self._assert_windows_descriptor_path(
                    descriptor,
                    source=source,
                    parts=parts,
                )
                child = _LockedPublicationParent(
                    path=path,
                    identity=(value.st_dev, value.st_ino),
                    descriptor=None,
                    native_handle=msvcrt.get_osfhandle(descriptor),
                )
                child.assert_current()
            except BaseException:
                os.close(descriptor)
                raise
            self._owned_directory_descriptors.append(descriptor)
            return child

        if parent.descriptor is None:
            raise RuntimeError(f"physical-copy parent is not anchored: {parent.path}")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path.name, flags, dir_fd=parent.descriptor)
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISDIR(value.st_mode):
                raise RuntimeError(f"physical-copy component is not a directory: {path}")
            child = _LockedPublicationParent(
                path=path,
                identity=(value.st_dev, value.st_ino),
                descriptor=descriptor,
                native_handle=None,
            )
            child.assert_current()
        except BaseException:
            os.close(descriptor)
            raise
        self._owned_directory_descriptors.append(descriptor)
        return child

    def _directory(
        self,
        parts: Sequence[str],
        *,
        source: bool,
        create: bool,
    ) -> _LockedPublicationParent:
        cache = self._source_parents if source else self._destination_parents
        root = self.source_root if source else self.destination_root
        for index in range(1, len(parts) + 1):
            key = tuple(parts[:index])
            if key in cache:
                continue
            parent_key = tuple(parts[: index - 1])
            parent = cache[parent_key]
            path = root.joinpath(*key)
            try:
                child = self._open_existing_directory(
                    parent,
                    path,
                    source=source,
                    parts=key,
                )
            except FileNotFoundError:
                if source or not create:
                    raise
                try:
                    child, retained, _ = _create_locked_directory(parent, path)
                except FileExistsError as error:
                    raise RuntimeError(
                        "physical-copy destination directory appeared concurrently; "
                        f"refusing adoption: {path}"
                    ) from error
                if child.native_handle is not None:
                    assert retained is not None and retained.descriptor is not None
                    try:
                        self._assert_windows_descriptor_path(
                            retained.descriptor,
                            source=False,
                            parts=key,
                        )
                    except BaseException as path_error:
                        try:
                            _rollback_created_directory(parent, path, retained)
                        except BaseException as rollback_error:
                            raise RuntimeError(
                                "created physical-copy directory failed final-path "
                                "validation and exact-handle rollback was incomplete: "
                                f"{path}: {rollback_error}"
                            ) from path_error
                        raise
                    self._retained_directory_handles.append(retained)
                elif child.descriptor is not None:
                    self._owned_directory_descriptors.append(child.descriptor)
                self._created_directories.append(
                    _CreatedPhysicalCopyDirectory(parent, child, path, retained)
                )
            cache[key] = child
        return cache[tuple(parts)]

    def ensure_directory(self, relative_path: str) -> Path:
        """Require a source directory and create its anchored destination peer."""

        self._assert_open()
        parts = _physical_copy_relative_parts(relative_path)
        source = self._directory(parts, source=True, create=False)
        destination = self._directory(parts, source=False, create=True)
        source.assert_current()
        destination.assert_current()
        self._assert_roots_current()
        return self.destination_root.joinpath(*parts)

    def _open_source_file(
        self,
        parent: _LockedPublicationParent,
        parts: tuple[str, ...],
    ) -> int:
        if parent.native_handle is not None:
            descriptor = _windows_open_relative_descriptor(
                parent.native_handle,
                parts[-1],
                share_write=False,
                share_delete=False,
            )
            try:
                self._assert_windows_descriptor_path(descriptor, source=True, parts=parts)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        if parent.descriptor is None:
            raise RuntimeError("physical-copy source parent is not anchored")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(parts[-1], flags, dir_fd=parent.descriptor)

    def _create_destination_file(
        self,
        parent: _LockedPublicationParent,
        parts: tuple[str, ...],
    ) -> int:
        if parent.native_handle is not None:
            delete_capable = self._compressor is None
            descriptor = _windows_open_relative_descriptor(
                parent.native_handle,
                parts[-1],
                create=True,
                write=True,
                delete_access=delete_capable,
                share_write=False,
                share_delete=False,
            )
            try:
                self._assert_windows_descriptor_path(descriptor, source=False, parts=parts)
            except BaseException as path_error:
                cleanup: _LockedPublishedFile | None = None
                try:
                    if delete_capable:
                        _windows_delete_opened_link(descriptor)
                    else:
                        value = os.fstat(descriptor)
                        cleanup = _LockedPublishedFile(
                            parent=parent,
                            path=self.destination_root.joinpath(*parts),
                            name=parts[-1],
                            identity=(value.st_dev, value.st_ino, value.st_size),
                            sha256=_descriptor_sha256(
                                descriptor,
                                chunk_size_bytes=self.chunk_size_bytes,
                            ),
                            retained_descriptor=descriptor,
                            retained_descriptor_delete_capable=False,
                        )
                        descriptor = -1
                        cleanup.unlink_owned()
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "destination final-path validation failed and exact-handle rollback "
                        f"was incomplete: {rollback_error}"
                    ) from path_error
                finally:
                    if cleanup is not None:
                        cleanup.close_retained_descriptor()
                    if descriptor >= 0:
                        os.close(descriptor)
                raise
            return descriptor
        if parent.descriptor is None:
            raise RuntimeError("physical-copy destination parent is not anchored")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        return os.open(parts[-1], flags, 0o600, dir_fd=parent.descriptor)

    def _logical_destination_stat(
        self,
        parent: _LockedPublicationParent,
        path: Path,
    ) -> os.stat_result:
        if parent.native_handle is not None:
            return path.stat(follow_symlinks=False)
        if parent.descriptor is None:
            raise RuntimeError("physical-copy destination parent is not anchored")
        return os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)

    def _rollback_partial_destination(
        self,
        parent: _LockedPublicationParent,
        path: Path,
        descriptor: int,
        *,
        descriptor_delete_capable: bool,
    ) -> None:
        if parent.native_handle is not None:
            if descriptor_delete_capable:
                _windows_delete_opened_link(descriptor)
                return
            value = os.fstat(descriptor)
            cleanup = _LockedPublishedFile(
                parent=parent,
                path=path,
                name=path.name,
                identity=(value.st_dev, value.st_ino, value.st_size),
                sha256=_descriptor_sha256(
                    descriptor,
                    chunk_size_bytes=self.chunk_size_bytes,
                ),
                retained_descriptor=descriptor,
                retained_descriptor_delete_capable=False,
            )
            cleanup.unlink_owned()
            return
        if parent.descriptor is None:
            raise RuntimeError("physical-copy destination parent is not anchored")
        value = os.fstat(descriptor)
        digest = _descriptor_sha256(descriptor, chunk_size_bytes=self.chunk_size_bytes)
        try:
            _posix_quarantine_and_delete_file(
                parent.descriptor,
                path.name,
                (value.st_dev, value.st_ino, value.st_size),
                digest,
            )
        except FileNotFoundError:
            if os.fstat(descriptor).st_nlink != 0:
                raise RuntimeError(
                    "owned partial destination lost its logical name but remains linked"
                ) from None

    def copy_file_no_overwrite(
        self,
        relative_path: str,
        *,
        expected_size_bytes: int,
        expected_sha256: str,
    ) -> PublishedPath:
        """Stream one exact source file into one new, physically independent file."""

        self._assert_open()
        parts = _physical_copy_relative_parts(relative_path)
        if (
            type(expected_size_bytes) is not int
            or expected_size_bytes < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("physical-copy expected size/SHA-256 is invalid")
        if self._compressor is not None:
            self._require_compression_space(
                next_logical_size_bytes=expected_size_bytes,
                phase="before copy",
            )
        source_parent = self._directory(parts[:-1], source=True, create=False)
        destination_parent = self._directory(parts[:-1], source=False, create=True)
        source_path = self.source_root.joinpath(*parts)
        destination_path = self.destination_root.joinpath(*parts)
        source_parent.assert_current()
        destination_parent.assert_current()
        self._assert_roots_current()
        assert_mutable_publication_destination(
            self.destination_root / ".histo-audit-copy-boundary",
            role="physical-copy destination root",
        )
        source_parent.assert_current()
        destination_parent.assert_current()
        self._assert_roots_current()

        source_descriptor: int | None = None
        destination_descriptor: int | None = None
        compression_guard_descriptor: int | None = None
        completed_file: _LockedPublishedFile | None = None
        try:
            source_descriptor = self._open_source_file(source_parent, parts)
            source_before = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(source_before.st_mode)
                or source_before.st_size != expected_size_bytes
            ):
                raise RuntimeError(
                    f"physical-copy source size/type differs from evidence: {source_path}"
                )
            destination_descriptor = self._create_destination_file(destination_parent, parts)
            destination_before = os.fstat(destination_descriptor)
            if (
                not stat.S_ISREG(destination_before.st_mode)
                or destination_before.st_size != 0
                or destination_before.st_nlink != 1
                or (destination_before.st_dev, destination_before.st_ino)
                == (source_before.st_dev, source_before.st_ino)
            ):
                raise RuntimeError(
                    f"physical-copy destination is not a new independent file: {destination_path}"
                )
            source_digest = _stream_physical_copy(
                source_descriptor,
                destination_descriptor,
                chunk_size_bytes=self.chunk_size_bytes,
            )
            source_after = os.fstat(source_descriptor)
            if (
                _stable_source_snapshot(source_after) != _stable_source_snapshot(source_before)
                or source_digest != expected_sha256
            ):
                raise RuntimeError(
                    f"physical-copy source changed or failed its expected digest: {source_path}"
                )
            _flush_physical_copy_destination(
                destination_descriptor,
                destination_parent,
            )
            destination_digest = _descriptor_sha256(
                destination_descriptor,
                chunk_size_bytes=self.chunk_size_bytes,
            )
            destination_after = os.fstat(destination_descriptor)
            logical = self._logical_destination_stat(destination_parent, destination_path)
            destination_identity = (
                destination_after.st_dev,
                destination_after.st_ino,
                destination_after.st_size,
            )
            if (
                not stat.S_ISREG(destination_after.st_mode)
                or destination_after.st_size != expected_size_bytes
                or destination_after.st_nlink != 1
                or destination_digest != expected_sha256
                or (logical.st_dev, logical.st_ino, logical.st_size) != destination_identity
            ):
                raise RuntimeError(
                    f"physical-copy destination failed durable exact readback: {destination_path}"
                )
            self._assert_windows_descriptor_path(
                destination_descriptor,
                source=False,
                parts=parts,
            )
            retained_descriptor_delete_capable = True
            if self._compressor is not None:
                if destination_parent.native_handle is not None:
                    compression_guard_descriptor = _windows_open_relative_descriptor(
                        destination_parent.native_handle,
                        parts[-1],
                        share_write=True,
                        share_delete=False,
                    )
                    self._assert_windows_descriptor_path(
                        compression_guard_descriptor,
                        source=False,
                        parts=parts,
                    )
                    guarded_before = os.fstat(compression_guard_descriptor)
                    if (
                        not stat.S_ISREG(guarded_before.st_mode)
                        or (
                            guarded_before.st_dev,
                            guarded_before.st_ino,
                            guarded_before.st_size,
                        )
                        != destination_identity
                        or guarded_before.st_nlink != 1
                    ):
                        raise RuntimeError(
                            "WOF LZX guard does not retain the exact copied destination"
                        )
                    os.close(destination_descriptor)
                    destination_descriptor = None
                    retained_descriptor_delete_capable = False
                else:
                    compression_guard_descriptor = destination_descriptor
                    destination_descriptor = None

                compression_result = self._compressor(destination_path)
                if compression_result is not None:
                    raise RuntimeError("WOF LZX compressor must return None on success")
                assert compression_guard_descriptor is not None
                if destination_parent.native_handle is None:
                    _flush_physical_copy_destination(
                        compression_guard_descriptor,
                        destination_parent,
                    )
                destination_digest = _descriptor_sha256(
                    compression_guard_descriptor,
                    chunk_size_bytes=self.chunk_size_bytes,
                )
                destination_after = os.fstat(compression_guard_descriptor)
                logical = self._logical_destination_stat(
                    destination_parent,
                    destination_path,
                )
                destination_identity = (
                    destination_after.st_dev,
                    destination_after.st_ino,
                    destination_after.st_size,
                )
                source_after_compression = os.fstat(source_descriptor)
                if (
                    not stat.S_ISREG(destination_after.st_mode)
                    or destination_after.st_size != expected_size_bytes
                    or destination_after.st_nlink != 1
                    or destination_digest != expected_sha256
                    or (logical.st_dev, logical.st_ino, logical.st_size) != destination_identity
                    or _stable_source_snapshot(source_after_compression)
                    != _stable_source_snapshot(source_before)
                ):
                    raise RuntimeError(
                        "WOF LZX destination/source failed exact post-compression readback: "
                        f"{destination_path}"
                    )
                self._assert_windows_descriptor_path(
                    compression_guard_descriptor,
                    source=False,
                    parts=parts,
                )
                self._require_compression_space(
                    next_logical_size_bytes=0,
                    phase="after compression",
                )
            source_parent.assert_current()
            destination_parent.assert_current()
            self._assert_roots_current()
            retained_descriptor = (
                compression_guard_descriptor
                if compression_guard_descriptor is not None
                else destination_descriptor
            )
            if retained_descriptor is None:
                raise RuntimeError("physical-copy destination ownership descriptor is unavailable")
            published = PublishedPath(
                path=destination_path,
                identity=(
                    destination_after.st_dev,
                    destination_after.st_ino,
                    destination_after.st_size,
                    destination_after.st_ctime_ns,
                ),
                kind="file",
                sha256=destination_digest,
            )
            completed_file = _LockedPublishedFile(
                parent=destination_parent,
                path=destination_path,
                name=parts[-1],
                identity=destination_identity,
                sha256=destination_digest,
                retained_descriptor=retained_descriptor,
                retained_descriptor_delete_capable=(retained_descriptor_delete_capable),
            )
        except BaseException as copy_error:
            rollback_error: BaseException | None = None
            destination_was_created = (
                destination_descriptor is not None or compression_guard_descriptor is not None
            )
            if destination_descriptor is not None and compression_guard_descriptor is not None:
                with suppress(OSError):
                    os.close(compression_guard_descriptor)
                compression_guard_descriptor = None
            if destination_descriptor is not None:
                try:
                    self._rollback_partial_destination(
                        destination_parent,
                        destination_path,
                        destination_descriptor,
                        descriptor_delete_capable=self._compressor is None,
                    )
                except BaseException as error:
                    rollback_error = error
            elif compression_guard_descriptor is not None:
                guarded_cleanup: _LockedPublishedFile | None = None
                try:
                    guarded_value = os.fstat(compression_guard_descriptor)
                    guarded_digest = _descriptor_sha256(
                        compression_guard_descriptor,
                        chunk_size_bytes=self.chunk_size_bytes,
                    )
                    guarded_cleanup = _LockedPublishedFile(
                        parent=destination_parent,
                        path=destination_path,
                        name=parts[-1],
                        identity=(
                            guarded_value.st_dev,
                            guarded_value.st_ino,
                            guarded_value.st_size,
                        ),
                        sha256=guarded_digest,
                        retained_descriptor=compression_guard_descriptor,
                        retained_descriptor_delete_capable=(
                            destination_parent.native_handle is None
                        ),
                    )
                    compression_guard_descriptor = None
                    guarded_cleanup.unlink_owned()
                except BaseException as error:
                    rollback_error = error
                finally:
                    if guarded_cleanup is not None:
                        guarded_cleanup.close_retained_descriptor()
            if destination_descriptor is not None:
                with suppress(OSError):
                    os.close(destination_descriptor)
                destination_descriptor = None
            if compression_guard_descriptor is not None:
                with suppress(OSError):
                    os.close(compression_guard_descriptor)
                compression_guard_descriptor = None
            if source_descriptor is not None:
                with suppress(OSError):
                    os.close(source_descriptor)
                source_descriptor = None
            if rollback_error is not None or (
                destination_was_created and os.path.lexists(destination_path)
            ):
                detail = (
                    f": {type(rollback_error).__name__}: {rollback_error}"
                    if rollback_error is not None
                    else "; a foreign logical destination remains"
                )
                raise RuntimeError(
                    "physical copy failed and ownership-safe rollback was incomplete "
                    f"for {destination_path}{detail}"
                ) from copy_error
            raise
        else:
            assert source_descriptor is not None and completed_file is not None
            self._completed_files.append(completed_file)
            if compression_guard_descriptor is not None:
                compression_guard_descriptor = None
            else:
                destination_descriptor = None
            os.close(source_descriptor)
            source_descriptor = None
            source_parent.assert_current()
            destination_parent.assert_current()
            self._assert_roots_current()
            return published

    def assert_roots_current(self) -> None:
        """Public final boundary used by the owning context manager."""

        self._assert_all_current()

    def _boundary_state(self) -> tuple[bool, bool, list[str]]:
        errors: list[str] = []
        states: list[bool] = []
        for source, role in ((True, "source"), (False, "destination")):
            try:
                self._assert_tree_current(source=source)
            except BaseException as error:
                states.append(False)
                errors.append(f"{role}: {type(error).__name__}: {error}")
            else:
                states.append(True)
        return states[0], states[1], errors

    def _release_created_posix_directory(
        self,
        created: _CreatedPhysicalCopyDirectory,
    ) -> None:
        descriptor = created.child.descriptor
        if descriptor is None:
            return
        created.child.descriptor = None
        with suppress(ValueError):
            self._owned_directory_descriptors.remove(descriptor)
        os.close(descriptor)

    def _rollback_completed_destination(self) -> list[str]:
        errors: list[str] = []
        for completed in reversed(self._completed_files):
            try:
                completed.unlink_owned()
            except BaseException as error:
                errors.append(f"file {completed.path}: {type(error).__name__}: {error}")
        for created in reversed(self._created_directories):
            try:
                if created.parent.native_handle is not None:
                    if created.retained is None:
                        raise RuntimeError(
                            "created Windows directory lacks its retained ownership handle"
                        )
                    _rollback_created_directory(
                        created.parent,
                        created.path,
                        created.retained,
                    )
                    with suppress(ValueError):
                        self._retained_directory_handles.remove(created.retained)
                else:
                    self._release_created_posix_directory(created)
                    if created.parent.descriptor is None:
                        raise RuntimeError("created POSIX directory parent anchor is unavailable")
                    _posix_quarantine_and_delete_directory(
                        created.parent.descriptor,
                        created.path.name,
                        created.child.identity,
                    )
            except BaseException as error:
                errors.append(f"directory {created.path}: {type(error).__name__}: {error}")
        return errors

    def _raise_for_boundary_failure(
        self,
        *,
        trigger: BaseException | None = None,
    ) -> None:
        source_current, destination_current, boundary_errors = self._boundary_state()
        if source_current and destination_current:
            return
        rollback_errors = self._rollback_completed_destination()
        boundary_error = AnchoredPhysicalCopyBoundaryError(
            source_tree_current=source_current,
            destination_tree_current=destination_current,
            rollback_complete=not rollback_errors,
            expected_destination_root=self.destination_root,
            boundary_errors=boundary_errors,
            rollback_errors=rollback_errors,
        )
        if trigger is not None:
            raise boundary_error from trigger
        raise boundary_error

    def close(self) -> None:
        if self._closed:
            return
        for completed in reversed(self._completed_files):
            with suppress(OSError):
                completed.close_retained_descriptor()
        for retained in reversed(self._retained_directory_handles):
            with suppress(OSError):
                retained.close()
        for descriptor in reversed(self._owned_directory_descriptors):
            with suppress(OSError):
                os.close(descriptor)
        self._closed = True


@contextmanager
def anchored_physical_copy_session(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    chunk_size_bytes: int = 8 * 1024 * 1024,
    compression_policy: str | None = None,
    compressor: PhysicalCopyCompressor | None = None,
    free_space_probe: PhysicalCopyFreeSpaceProbe | None = None,
) -> Iterator[AnchoredPhysicalCopySession]:
    """Retain both complete root handle chains for one multi-file physical import."""

    if type(chunk_size_bytes) is not int or chunk_size_bytes <= 0:
        raise ValueError("physical-copy chunk_size_bytes must be a positive integer")
    if compression_policy not in {None, ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY}:
        raise ValueError("physical-copy compression policy is unsupported")
    if compression_policy is None and (compressor is not None or free_space_probe is not None):
        raise ValueError("physical-copy compressor/free-space probe requires the WOF LZX policy")
    if compressor is not None and not callable(compressor):
        raise TypeError("physical-copy compressor must be callable")
    if free_space_probe is not None and not callable(free_space_probe):
        raise TypeError("physical-copy free-space probe must be callable")
    selected_compressor: PhysicalCopyCompressor | None = None
    selected_free_space_probe: PhysicalCopyFreeSpaceProbe | None = None
    if compression_policy == ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY:
        if compressor is None and os.name != "nt":
            raise OSError("default WOF LZX physical-copy compression requires Windows")
        selected_compressor = compressor or _windows_wof_lzx_compress_file
        selected_free_space_probe = free_space_probe or _physical_copy_free_space
    source = Path(os.path.abspath(Path(source_root).expanduser()))
    destination = Path(os.path.abspath(Path(destination_root).expanduser()))
    if _publication_parent_key(source) == _publication_parent_key(destination):
        raise ValueError("physical-copy source and destination roots must be different paths")
    try:
        common = Path(os.path.commonpath((source, destination)))
    except ValueError:
        common = None
    if common is not None and _publication_parent_key(common) in {
        _publication_parent_key(source),
        _publication_parent_key(destination),
    }:
        raise ValueError("physical-copy roots must not contain one another")
    session: AnchoredPhysicalCopySession | None = None
    session_created = False
    yielded = False
    safe_untyped_body_error = False
    try:
        assert_mutable_publication_destination(
            destination / ".histo-audit-copy-boundary",
            role="physical-copy destination root",
        )
        with _locked_publication_parents(
            (source, destination),
            final_paths=(),
            create_missing=False,
            read_only_paths=(source,),
        ) as parents:
            session = AnchoredPhysicalCopySession(
                source,
                destination,
                parents[_publication_parent_key(source)],
                parents[_publication_parent_key(destination)],
                chunk_size_bytes=chunk_size_bytes,
                compression_policy=compression_policy,
                compressor=selected_compressor,
                free_space_probe=selected_free_space_probe,
            )
            session_created = True
            try:
                yielded = True
                yield session
            except BaseException as error:
                session._raise_for_boundary_failure(trigger=error)
                safe_untyped_body_error = True
                raise
            else:
                session._raise_for_boundary_failure()
            finally:
                session.close()
    except AnchoredPhysicalCopyBoundaryError:
        raise
    except BaseException as error:
        if safe_untyped_body_error:
            raise
        phase = (
            "before anchored session creation"
            if not session_created
            else (
                "before the first yielded operation"
                if not yielded
                else "during final anchored boundary verification"
            )
        )
        raise AnchoredPhysicalCopyBoundaryError(
            source_tree_current=False,
            destination_tree_current=False,
            rollback_complete=False,
            expected_destination_root=destination,
            boundary_errors=(f"{phase}: {type(error).__name__}: {error}",),
        ) from error


def _capture_physical_publication_source(
    path: Path,
    parents: Mapping[str, _LockedPublicationParent],
    *,
    max_bytes: int | None = None,
) -> _LockedPublishedFile:
    """Capture one lexical regular source through a retained no-follow descriptor."""

    if max_bytes is not None and (
        type(max_bytes) is not int  # bool is deliberately rejected
        or max_bytes < 0
    ):
        raise ValueError("anchored read max_bytes must be a non-negative exact integer")
    parent = parents[_publication_parent_key(path.parent)]
    parent.assert_current()
    provisional = _LockedPublishedFile(parent, path, path.name, (0, 0, 0), "")
    descriptor: int | None = None
    try:
        try:
            descriptor = provisional._open_descriptor(
                share_write=False,
                share_delete=False,
            )
        except OSError as exc:
            raise ValueError(
                f"physical publication source is not a lexical regular file: {path}"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _is_link_or_reparse(path, before):
            raise ValueError(f"physical publication source is not a lexical regular file: {path}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(
                f"anchored read source exceeds the bounded size limit of {max_bytes} bytes: {path}"
            )
        digest = _descriptor_sha256(descriptor, chunk_size_bytes=1024 * 1024)
        after = os.fstat(descriptor)
        if _stable_source_snapshot(after) != _stable_source_snapshot(before):
            raise RuntimeError(f"physical publication source changed during capture: {path}")
        captured = _LockedPublishedFile(
            parent=parent,
            path=path,
            name=path.name,
            identity=(before.st_dev, before.st_ino, before.st_size),
            sha256=digest,
            retained_descriptor=descriptor,
        )
        descriptor = None
        parent.assert_current()
        return captured
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_file_anchored(
    path: str | Path,
    *,
    allow_empty: bool = False,
    require_single_link: bool = True,
    max_bytes: int | None = None,
) -> bytes:
    """Read one lexical file through a no-follow, parent-anchored descriptor.

    On Windows the complete parent chain and the opened file deny delete sharing,
    while the file itself also denies write sharing. On POSIX the file is opened
    relative to a retained parent descriptor with ``O_NOFOLLOW``. The payload,
    identity, metadata, and digest are all checked on that same opened object.
    """

    if max_bytes is not None and (
        type(max_bytes) is not int  # bool is deliberately rejected
        or max_bytes < 0
    ):
        raise ValueError("anchored read max_bytes must be a non-negative exact integer")
    source = _absolute_lexical_path(path)
    _require_publication_leaf(source.name)
    if not os.path.lexists(source):
        raise FileNotFoundError(f"anchored read source is missing: {source}")
    captured: _LockedPublishedFile | None = None
    with _locked_publication_parents(
        (source.parent,),
        final_paths=(),
        create_missing=False,
        read_only_paths=(source.parent,),
    ) as parents:
        parent = parents[_publication_parent_key(source.parent)]
        parent.assert_current()
        captured = _capture_physical_publication_source(
            source,
            parents,
            max_bytes=max_bytes,
        )
        descriptor: int | None = None
        try:
            descriptor = captured._open_descriptor(
                share_write=False,
                share_delete=False,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or _is_link_or_reparse(source, before)
                or (require_single_link and before.st_nlink != 1)
            ):
                raise ValueError(
                    "anchored read requires a regular, non-reparse"
                    + (", single-link" if require_single_link else "")
                    + f" source: {source}"
                )
            payload = bytearray()
            digest = hashlib.sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while chunk := os.read(descriptor, 1024 * 1024):
                if max_bytes is not None and len(payload) + len(chunk) > max_bytes:
                    raise ValueError(
                        "anchored read source grew beyond the bounded size limit of "
                        f"{max_bytes} bytes: {source}"
                    )
                payload.extend(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                _stable_source_snapshot(after) != _stable_source_snapshot(before)
                or len(payload) != after.st_size
                or digest.hexdigest() != captured.sha256
                or (require_single_link and after.st_nlink != 1)
            ):
                raise RuntimeError(f"anchored read source changed during capture: {source}")
            logical_descriptor: int | None = None
            try:
                if parent.native_handle is not None:
                    logical_descriptor = _windows_open_relative_descriptor(
                        parent.native_handle,
                        source.name,
                        share_write=False,
                        share_delete=False,
                    )
                else:
                    assert parent.descriptor is not None
                    logical_descriptor = os.open(
                        source.name,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent.descriptor,
                    )
                logical_before = os.fstat(logical_descriptor)
                logical_digest = _descriptor_sha256(
                    logical_descriptor,
                    chunk_size_bytes=1024 * 1024,
                )
                logical_after = os.fstat(logical_descriptor)
                if parent.native_handle is None:
                    assert parent.descriptor is not None
                    logical_name_after = os.stat(
                        source.name,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                else:
                    logical_name_after = source.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(logical_before.st_mode)
                    or _is_link_or_reparse(source, logical_before)
                    or (
                        logical_before.st_dev,
                        logical_before.st_ino,
                        logical_before.st_size,
                    )
                    != captured.identity
                    or _stable_source_snapshot(logical_after)
                    != _stable_source_snapshot(logical_before)
                    or (
                        logical_name_after.st_dev,
                        logical_name_after.st_ino,
                        logical_name_after.st_size,
                    )
                    != captured.identity
                    or _is_link_or_reparse(source, logical_name_after)
                    or logical_digest != captured.sha256
                    or (require_single_link and logical_after.st_nlink != 1)
                    or (require_single_link and logical_name_after.st_nlink != 1)
                ):
                    raise RuntimeError(
                        f"anchored read logical source changed during capture: {source}"
                    )
            finally:
                if logical_descriptor is not None:
                    os.close(logical_descriptor)
            parent.assert_current()
            if not payload and not allow_empty:
                raise ValueError(f"anchored read source must not be empty: {source}")
            return bytes(payload)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if captured is not None:
                captured.close_retained_descriptor()


def _create_physical_publication_file(
    parent: _LockedPublicationParent,
    path: Path,
) -> int:
    """Create one exact private file relative to an anchored destination parent."""

    _require_publication_leaf(path.name)
    if parent.native_handle is not None:
        descriptor = _windows_open_relative_descriptor(
            parent.native_handle,
            path.name,
            create=True,
            write=True,
            delete_access=True,
            share_write=False,
            share_delete=False,
        )
        try:
            expected = os.path.normcase(os.path.normpath(str(path)))
            if _windows_final_path_for_descriptor(descriptor) != expected:
                raise RuntimeError(
                    f"physical publication destination has an unexpected final path: {path}"
                )
        except BaseException as path_error:
            try:
                _windows_delete_opened_link(descriptor)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "physical publication destination path validation failed and exact-handle "
                    f"rollback was incomplete: {path}: {rollback_error}"
                ) from path_error
            finally:
                os.close(descriptor)
            raise
        return descriptor
    if parent.descriptor is None:
        raise RuntimeError(f"physical publication parent is not anchored: {parent.path}")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path.name, flags, 0o600, dir_fd=parent.descriptor)


def _physical_publication_logical_stat(
    parent: _LockedPublicationParent,
    path: Path,
) -> os.stat_result:
    if parent.native_handle is not None:
        return path.stat(follow_symlinks=False)
    if parent.descriptor is None:
        raise RuntimeError(f"physical publication parent is not anchored: {parent.path}")
    return os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)


def _rollback_physical_publication_file(
    parent: _LockedPublicationParent,
    path: Path,
    descriptor: int,
) -> None:
    """Remove only the exact private physical-copy inode held by ``descriptor``."""

    if parent.native_handle is not None:
        _windows_delete_opened_link(descriptor)
        return
    if parent.descriptor is None:
        raise RuntimeError(f"physical publication parent is not anchored: {parent.path}")
    value = os.fstat(descriptor)
    digest = _descriptor_sha256(descriptor, chunk_size_bytes=1024 * 1024)
    try:
        _posix_quarantine_and_delete_file(
            parent.descriptor,
            path.name,
            (value.st_dev, value.st_ino, value.st_size),
            digest,
        )
    except FileNotFoundError:
        if os.fstat(descriptor).st_nlink != 0:
            raise RuntimeError(
                "owned private physical-publication file lost its name but remains linked"
            ) from None


def _copy_locked_source_to_private_file(
    source: _LockedPublishedFile,
    private_path: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> _LockedPublishedFile:
    """Stream a retained source into one fresh, same-parent private inode."""

    parent = parents[_publication_parent_key(private_path.parent)]
    parent.assert_current()
    source.parent.assert_current()
    if _locked_path_exists(private_path, parents):
        raise FileExistsError(f"private physical-publication path exists: {private_path}")

    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = source._open_descriptor(
            share_write=False,
            share_delete=False,
        )
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or (source_before.st_dev, source_before.st_ino, source_before.st_size)
            != source.identity
        ):
            raise RuntimeError(
                f"physical publication source differs from captured identity: {source.path}"
            )
        destination_descriptor = _create_physical_publication_file(parent, private_path)
        destination_before = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(destination_before.st_mode)
            or destination_before.st_size != 0
            or destination_before.st_nlink != 1
            or (destination_before.st_dev, destination_before.st_ino)
            == (source_before.st_dev, source_before.st_ino)
        ):
            raise RuntimeError(
                f"physical publication did not create an independent inode: {private_path}"
            )

        os.lseek(source_descriptor, 0, os.SEEK_SET)
        source_digest = _stream_physical_copy(
            source_descriptor,
            destination_descriptor,
            chunk_size_bytes=8 * 1024 * 1024,
        )
        source_after = os.fstat(source_descriptor)
        if (
            _stable_source_snapshot(source_after) != _stable_source_snapshot(source_before)
            or source_digest != source.sha256
        ):
            raise RuntimeError(
                f"physical publication source changed or failed digest: {source.path}"
            )

        _flush_physical_copy_destination(destination_descriptor, parent)
        destination_digest = _descriptor_sha256(
            destination_descriptor,
            chunk_size_bytes=8 * 1024 * 1024,
        )
        destination_after = os.fstat(destination_descriptor)
        logical = _physical_publication_logical_stat(parent, private_path)
        destination_identity = (
            destination_after.st_dev,
            destination_after.st_ino,
            destination_after.st_size,
        )
        if (
            not stat.S_ISREG(destination_after.st_mode)
            or destination_after.st_size != source_before.st_size
            or destination_after.st_nlink != 1
            or destination_digest != source.sha256
            or (logical.st_dev, logical.st_ino, logical.st_size) != destination_identity
            or logical.st_nlink != 1
            or destination_identity[:2] == source.identity[:2]
        ):
            raise RuntimeError(
                f"private physical-publication file failed exact readback: {private_path}"
            )
        parent.assert_current()
        source.parent.assert_current()
        completed = _LockedPublishedFile(
            parent=parent,
            path=private_path,
            name=private_path.name,
            identity=destination_identity,
            sha256=destination_digest,
            retained_descriptor=destination_descriptor,
        )
        destination_descriptor = None
        return completed
    except BaseException as copy_error:
        rollback_error: BaseException | None = None
        destination_was_created = destination_descriptor is not None
        if destination_descriptor is not None:
            try:
                _rollback_physical_publication_file(
                    parent,
                    private_path,
                    destination_descriptor,
                )
            except BaseException as error:
                rollback_error = error
            finally:
                with suppress(OSError):
                    os.close(destination_descriptor)
                destination_descriptor = None
        if rollback_error is not None or (
            destination_was_created and os.path.lexists(private_path)
        ):
            detail = (
                f": {type(rollback_error).__name__}: {rollback_error}"
                if rollback_error is not None
                else "; a foreign logical private path remains"
            )
            raise RuntimeError(
                "physical publication copy failed and exact rollback was incomplete for "
                f"{private_path}{detail}"
            ) from copy_error
        raise
    finally:
        if destination_descriptor is not None:
            with suppress(OSError):
                os.close(destination_descriptor)
        if source_descriptor is not None:
            with suppress(OSError):
                os.close(source_descriptor)


def _physically_publish_locked_file(
    source: _LockedPublishedFile,
    target: Path,
    parents: Mapping[str, _LockedPublicationParent],
) -> _LockedPublishedFile:
    """Copy to a private inode and atomically rename it to the final leaf."""

    parent = parents[_publication_parent_key(target.parent)]
    parent.assert_current()
    if _locked_path_exists(target, parents):
        raise FileExistsError(f"refusing to overwrite publication path: {target}")
    private_path = target.with_name(f".histo-audit-physical-{secrets.token_hex(16)}")
    private = _copy_locked_source_to_private_file(source, private_path, parents)
    descriptor = private.retained_descriptor
    candidate: _LockedPublishedFile | None = None
    published: _LockedPublishedFile | None = None
    renamed = False
    try:
        if descriptor is None:
            raise RuntimeError(
                f"private physical-publication descriptor was not retained: {private_path}"
            )
        candidate = _LockedPublishedFile(
            parent=parent,
            path=target,
            name=target.name,
            identity=private.identity,
            sha256=private.sha256,
            retained_descriptor=descriptor,
        )
        if parent.native_handle is not None:
            _windows_rename_relative_no_overwrite(
                descriptor,
                parent.native_handle,
                target.name,
            )
        else:
            if parent.descriptor is None:
                raise RuntimeError(f"physical publication parent is not anchored: {parent.path}")
            _posix_rename_noreplace(
                parent.descriptor,
                private.name,
                parent.descriptor,
                target.name,
            )
        renamed = True
        published = candidate
        private.retained_descriptor = None

        if parent.native_handle is not None:
            expected = os.path.normcase(os.path.normpath(str(target)))
            if _windows_final_path_for_descriptor(descriptor) != expected:
                raise RuntimeError(
                    f"physical publication destination has an unexpected final path: {target}"
                )
            _windows_flush_handle(parent.native_handle)
        else:
            assert parent.descriptor is not None
            os.fsync(parent.descriptor)
        if _locked_path_exists(private_path, parents):
            raise RuntimeError(
                f"private physical-publication name remains after promotion: {private_path}"
            )
        value = os.fstat(descriptor)
        logical = _physical_publication_logical_stat(parent, target)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or (logical.st_dev, logical.st_ino, logical.st_size) != published.identity
            or logical.st_nlink != 1
            or published.identity[:2] == source.identity[:2]
            or not published.still_owned()
        ):
            raise RuntimeError(
                f"physical publication is not a single-link independent file: {target}"
            )
        parent.assert_current()
        return published
    except BaseException as publish_error:
        cleanup_errors: list[str] = []
        if renamed:
            if candidate is None:
                cleanup_errors.append("published target ownership proof was not retained")
                rollback_target = None
            else:
                rollback_target = published if published is not None else candidate
            if private.retained_descriptor == descriptor:
                private.retained_descriptor = None
            if rollback_target is not None:
                try:
                    rollback_target.unlink_owned()
                except (OSError, RuntimeError) as error:
                    cleanup_errors.append(f"published target: {error}")
        else:
            # Before rename, ``private`` is the sole descriptor owner.  The
            # target-shaped candidate merely carries the same integer so that
            # an immediately completed rename can be rolled back by handle.
            if candidate is not None:
                candidate.retained_descriptor = None
            try:
                private.unlink_owned()
            except (OSError, RuntimeError) as error:
                cleanup_errors.append(f"private source: {error}")
        if candidate is not None:
            candidate.close_retained_descriptor()
        private.close_retained_descriptor()
        remaining_path = target if renamed else private_path
        if os.path.lexists(remaining_path):
            cleanup_errors.append(f"owned logical path remains: {remaining_path}")
        if cleanup_errors:
            raise RuntimeError(
                "physical publication failed and ownership-safe rollback was incomplete: "
                + "; ".join(cleanup_errors)
            ) from publish_error
        raise


def publish_file_physical_copy_no_overwrite(
    staged: str | Path,
    destination: str | Path,
) -> PublishedPath:
    """Publish a lexical source as a physically independent, single-link file."""

    source = _absolute_lexical_path(staged)
    if not os.path.lexists(source):
        raise FileNotFoundError(f"physical publication source is missing: {source}")
    source_value = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(source_value.st_mode) or _is_link_or_reparse(source, source_value):
        raise ValueError(f"physical publication source must be a lexical regular file: {source}")
    final = assert_mutable_publication_destination(
        destination,
        role="physical file publication destination",
    )
    read_only_paths = (
        (source.parent,)
        if _publication_parent_key(source.parent) != _publication_parent_key(final.parent)
        else ()
    )
    locked_publication: _LockedPublishedFile | None = None
    with _locked_publication_parents(
        (source.parent, final.parent),
        final_paths=(final,),
        read_only_paths=read_only_paths,
    ) as parents:
        captured = _capture_physical_publication_source(source, parents)
        try:
            assert_mutable_publication_destination(
                final,
                role="physical file publication destination",
            )
            locked_publication = _physically_publish_locked_file(captured, final, parents)
            assert_mutable_publication_destination(
                final,
                role="physical file publication destination",
            )
            value = locked_publication._stat()
            if value.st_nlink != 1 or locked_publication.identity[:2] == captured.identity[:2]:
                raise RuntimeError(
                    f"physical file publication failed independence readback: {final}"
                )
            result = PublishedPath(
                path=final,
                identity=(
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_ctime_ns,
                ),
                kind="file",
                sha256=locked_publication.sha256,
                required_nlink=1,
            )
            locked_publication.close_retained_descriptor()
            locked_publication = None
            return result
        except BaseException as publish_error:
            if locked_publication is not None and locked_publication.exists():
                try:
                    locked_publication.unlink_owned()
                except (OSError, RuntimeError) as rollback_error:
                    raise RuntimeError(
                        "physical file publication failed and ownership-safe rollback was "
                        f"incomplete: {final}: {rollback_error}"
                    ) from publish_error
            raise
        finally:
            if locked_publication is not None:
                locked_publication.close_retained_descriptor()
            captured.close_retained_descriptor()


def _publish_file_no_overwrite(
    staged: str | Path,
    destination: str | Path,
    *,
    owned_success_marker_parent: PublishedPath | None,
) -> PublishedPath:
    """Internal publication with a narrow owned marker-last exception."""

    source = Path(staged).resolve()
    lexical_final = _lexical_final_path(destination)
    if owned_success_marker_parent is not None:
        if lexical_final.name not in {IMMUTABLE_MARKER, ARTIFACT_MANIFEST_FILENAME}:
            raise ValueError(
                "owned success-marker publication is limited to a recognised seal marker"
            )
        if (
            owned_success_marker_parent.kind != "directory"
            or _lexical_final_path(owned_success_marker_parent.path) != lexical_final.parent
            or not owned_success_marker_parent.still_owned()
        ):
            raise RuntimeError(
                "success-marker publication requires the exact transaction-owned parent"
            )
    final = assert_mutable_publication_destination(
        lexical_final,
        role="file publication destination",
    )
    if not source.is_file():
        raise FileNotFoundError(f"staged publication file is missing: {source}")

    locked_publication: _LockedPublishedFile | None = None
    with _locked_publication_parents(
        (source.parent, final.parent),
        final_paths=(final,),
    ) as parents:
        staged_file = _capture_locked_file(source, parents)
        if _locked_path_exists(final, parents):
            raise FileExistsError(f"refusing to overwrite publication path: {final}")
        assert_mutable_publication_destination(
            final,
            role="file publication destination",
        )
        try:
            locked_publication = _publish_file_to_locked_parent(
                staged_file,
                final,
                parents,
            )
            value = locked_publication._stat()
            if owned_success_marker_parent is None:
                assert_mutable_publication_destination(
                    final,
                    role="file publication destination",
                )
            else:
                if (
                    _lexical_final_path(owned_success_marker_parent.path) != final.parent
                    or not owned_success_marker_parent.still_owned()
                    or not locked_publication.still_owned()
                ):
                    raise RuntimeError(
                        "success marker or its transaction-owned parent changed during publication"
                    )
                # Recheck every ancestor above the newly sealed root with the
                # ordinary guard. Within the owned root, only this exact owned
                # marker may exist.
                assert_mutable_publication_destination(
                    final.parent,
                    role="success-marker publication root",
                )
                for marker_name in (IMMUTABLE_MARKER, ARTIFACT_MANIFEST_FILENAME):
                    marker = final.parent / marker_name
                    if os.path.lexists(marker) and marker != final:
                        raise PermissionError(
                            "success-marker publication found another seal marker in its "
                            f"transaction-owned root: {marker}"
                        )
            if not locked_publication.still_owned():
                raise OSError(f"published file failed identity/hash readback: {final}")
            published = PublishedPath(
                path=final,
                identity=(
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_ctime_ns,
                ),
                kind="file",
                sha256=locked_publication.sha256,
            )
            locked_publication.close_retained_descriptor()
            return published
        except BaseException as publish_error:
            if locked_publication is not None and locked_publication.exists():
                try:
                    locked_publication.unlink_owned()
                except (OSError, RuntimeError):
                    raise RuntimeError(
                        "file publication failed and anchored ownership-safe rollback was "
                        f"incomplete: {final}"
                    ) from publish_error
            raise


def publish_file_no_overwrite(staged: str | Path, destination: str | Path) -> PublishedPath:
    """Publish one staged regular file atomically without replacing any path."""

    return _publish_file_no_overwrite(
        staged,
        destination,
        owned_success_marker_parent=None,
    )


def publish_bytes_no_overwrite(payload: bytes, destination: str | Path) -> PublishedPath:
    """Durably create one exact single-link file from in-memory bytes.

    The final leaf is created with no-overwrite semantics relative to a retained
    parent handle. A failed write/readback removes only that exact opened object;
    a foreign replacement is never deleted.
    """

    if type(payload) is not bytes or not payload:
        raise ValueError("in-memory publication payload must be non-empty exact bytes")
    final = assert_mutable_publication_destination(
        destination,
        role="in-memory file publication destination",
    )
    descriptor: int | None = None
    created = False
    with _locked_publication_parents(
        (final.parent,),
        final_paths=(final,),
    ) as parents:
        parent = parents[_publication_parent_key(final.parent)]
        if _locked_path_exists(final, parents):
            raise FileExistsError(f"refusing to overwrite publication path: {final}")
        try:
            descriptor = _create_physical_publication_file(parent, final)
            created = True
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError(f"in-memory publication write made no progress: {final}")
                offset += written
            if parent.native_handle is not None:
                _windows_flush_descriptor(descriptor)
            else:
                os.fsync(descriptor)
            observed_sha256 = _descriptor_sha256(
                descriptor,
                chunk_size_bytes=1024 * 1024,
            )
            value = os.fstat(descriptor)
            logical = _physical_publication_logical_stat(parent, final)
            expected_sha256 = hashlib.sha256(payload).hexdigest()
            if (
                not stat.S_ISREG(value.st_mode)
                or value.st_nlink != 1
                or value.st_size != len(payload)
                or observed_sha256 != expected_sha256
                or (
                    logical.st_dev,
                    logical.st_ino,
                    logical.st_size,
                )
                != (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                )
                or logical.st_nlink != 1
                or _is_link_or_reparse(final, logical)
            ):
                raise RuntimeError(f"in-memory publication failed exact object readback: {final}")
            parent.assert_current()
            if parent.native_handle is not None:
                _windows_flush_handle(parent.native_handle)
            else:
                assert parent.descriptor is not None
                os.fsync(parent.descriptor)
            return PublishedPath(
                path=final,
                identity=(
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_ctime_ns,
                ),
                kind="file",
                sha256=expected_sha256,
                required_nlink=1,
            )
        except BaseException as publish_error:
            if descriptor is not None and created:
                try:
                    _rollback_physical_publication_file(
                        parent,
                        final,
                        descriptor,
                    )
                except (OSError, RuntimeError) as rollback_error:
                    raise RuntimeError(
                        "in-memory file publication failed and exact rollback was "
                        f"incomplete: {final}: {rollback_error}"
                    ) from publish_error
                if os.path.lexists(final):
                    raise RuntimeError(
                        "in-memory file publication rolled back its owned object "
                        f"but a foreign logical path remains: {final}"
                    ) from publish_error
            raise
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)


def publish_success_marker_no_overwrite(
    staged: str | Path,
    destination: str | Path,
    *,
    owned_parent: PublishedPath,
) -> PublishedPath:
    """Publish one recognised seal marker last into its transaction-owned root."""

    return _publish_file_no_overwrite(
        staged,
        destination,
        owned_success_marker_parent=owned_parent,
    )


def create_directory_no_overwrite(destination: str | Path) -> PublishedPath:
    """Create one final directory only if its name is absent."""

    final = assert_mutable_publication_destination(
        destination,
        role="directory publication destination",
    )
    with _locked_publication_parents(
        (final.parent,),
        final_paths=(final,),
    ) as parents:
        parent = parents[_publication_parent_key(final.parent)]
        if _locked_path_exists(final, parents):
            raise FileExistsError(f"refusing to overwrite publication path: {final}")
        assert_mutable_publication_destination(
            final,
            role="directory publication destination",
        )
        child: _LockedPublicationParent | None = None
        retained: _RetainedPublicationHandle | None = None
        try:
            child, retained, value = _create_locked_directory(parent, final)
            if (
                os.listdir(final)
                if child.native_handle is not None
                else os.listdir(child.descriptor)
            ):
                raise RuntimeError(f"new publication directory is unexpectedly non-empty: {final}")
            parent.assert_current()
            child.assert_current()
            assert_mutable_publication_destination(
                final,
                role="directory publication destination",
            )
            created = PublishedPath(
                path=final,
                identity=(
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_ctime_ns,
                ),
                kind="directory",
            )
            if not created.still_owned():
                raise OSError(f"published directory failed identity readback: {final}")
            if child.descriptor is not None:
                os.close(child.descriptor)
                child.descriptor = None
            if retained is not None:
                retained.close()
                retained = None
            return created
        except BaseException as create_error:
            if retained is not None:
                try:
                    _rollback_created_directory(parent, final, retained)
                except (OSError, RuntimeError):
                    raise RuntimeError(
                        "directory publication failed and ownership-safe rollback was "
                        f"incomplete: {final}"
                    ) from create_error
            elif child is not None and child.descriptor is not None:
                try:
                    assert parent.descriptor is not None
                    _posix_quarantine_and_delete_directory(
                        parent.descriptor,
                        final.name,
                        child.identity,
                    )
                except (OSError, RuntimeError):
                    raise RuntimeError(
                        "directory publication failed and ownership-safe rollback was "
                        f"incomplete: {final}"
                    ) from create_error
                finally:
                    os.close(child.descriptor)
                    child.descriptor = None
            raise


def publish_flat_directory_no_overwrite(
    staged_directory: str | Path,
    destination: str | Path,
    *,
    success_marker_name: str | None = None,
) -> list[PublishedPath]:
    """Publish a flat directory with an optional marker file linked last."""

    staged = Path(staged_directory).resolve()
    if not staged.is_dir():
        raise FileNotFoundError(f"staged publication directory is missing: {staged}")
    sources = tuple(sorted(staged.iterdir(), key=lambda value: value.name))
    if not sources or any(not source.is_file() for source in sources):
        raise ValueError("staged publication directory must be a non-empty flat file set")
    if success_marker_name is not None:
        marker = staged / success_marker_name
        if marker not in sources:
            raise FileNotFoundError(f"staged success marker is missing: {marker}")
        non_markers = tuple(
            source
            for source in sources
            if source != marker
            and not (
                success_marker_name == IMMUTABLE_MARKER
                and source.name == ARTIFACT_MANIFEST_FILENAME
            )
        )
        owned_predecessor_markers = tuple(
            source
            for source in sources
            if source != marker and source.name == ARTIFACT_MANIFEST_FILENAME
        )
        sources = (*non_markers, *owned_predecessor_markers, marker)

    final = assert_mutable_publication_destination(
        destination,
        role="flat-directory publication destination",
    )
    targets = tuple(final / source.name for source in sources)
    locked_publications: list[_LockedPublishedFile] = []

    with _locked_publication_parents(
        (staged, final.parent),
        final_paths=(final, *targets),
    ) as outer_parents:
        target_parent = outer_parents[_publication_parent_key(final.parent)]
        if _locked_path_exists(final, outer_parents):
            raise FileExistsError(f"refusing to overwrite publication path: {final}")
        output_parent: _LockedPublicationParent | None = None
        retained_directory: _RetainedPublicationHandle | None = None
        try:
            output_parent, retained_directory, directory_value = _create_locked_directory(
                target_parent,
                final,
            )
            assert_mutable_publication_destination(
                final,
                role="flat-directory publication destination",
            )
            parents = {
                **outer_parents,
                _publication_parent_key(final): output_parent,
            }
            for source, target in zip(sources, targets, strict=True):
                staged_file = _capture_locked_file(source, parents)
                recognised_success_marker = source.name in {
                    IMMUTABLE_MARKER,
                    ARTIFACT_MANIFEST_FILENAME,
                } and (
                    source.name == success_marker_name
                    or (
                        success_marker_name == IMMUTABLE_MARKER
                        and source.name == ARTIFACT_MANIFEST_FILENAME
                    )
                )
                if recognised_success_marker:
                    assert_mutable_publication_destination(
                        final,
                        role="success-marker publication root",
                    )
                    for marker_name in (IMMUTABLE_MARKER, ARTIFACT_MANIFEST_FILENAME):
                        marker_path = final / marker_name
                        if not os.path.lexists(marker_path):
                            continue
                        if not any(
                            value.path == marker_path and value.still_owned()
                            for value in locked_publications
                        ):
                            raise PermissionError(
                                "success-marker publication found an unowned seal marker in "
                                f"its transaction root: {marker_path}"
                            )
                else:
                    assert_mutable_publication_destination(
                        target,
                        role="flat-directory file destination",
                    )
                locked = _publish_file_to_locked_parent(
                    staged_file,
                    target,
                    parents,
                )
                locked_publications.append(locked)
                if recognised_success_marker:
                    assert_mutable_publication_destination(
                        final,
                        role="success-marker publication root",
                    )
                else:
                    assert_mutable_publication_destination(
                        target,
                        role="flat-directory file destination",
                    )

            actual_names = set(
                os.listdir(final)
                if output_parent.native_handle is not None
                else os.listdir(output_parent.descriptor)
            )
            expected_names = {target.name for target in targets}
            if actual_names != expected_names:
                raise RuntimeError(
                    "flat-directory publication exact-set readback failed: "
                    f"expected={sorted(expected_names)}, observed={sorted(actual_names)}"
                )
            target_parent.assert_current()
            output_parent.assert_current()
            if not all(value.still_owned() for value in locked_publications):
                raise OSError(f"flat-directory publication failed final readback: {final}")
            assert_mutable_publication_destination(
                final,
                role="flat-directory publication destination",
            )

            file_publications: list[PublishedPath] = []
            for locked in locked_publications:
                value = locked._stat()
                file_publications.append(
                    PublishedPath(
                        path=locked.path,
                        identity=(
                            value.st_dev,
                            value.st_ino,
                            value.st_size,
                            value.st_ctime_ns,
                        ),
                        kind="file",
                        sha256=locked.sha256,
                    )
                )
            directory_publication = PublishedPath(
                path=final,
                identity=(
                    directory_value.st_dev,
                    directory_value.st_ino,
                    directory_value.st_size,
                    directory_value.st_ctime_ns,
                ),
                kind="directory",
            )
            for locked in locked_publications:
                locked.close_retained_descriptor()
            if retained_directory is not None:
                retained_directory.close()
                retained_directory = None
            if output_parent.descriptor is not None:
                os.close(output_parent.descriptor)
                output_parent.descriptor = None
            return [directory_publication, *file_publications]
        except BaseException as publish_error:
            rollback_errors: list[str] = []
            for published in reversed(locked_publications):
                try:
                    published.unlink_owned()
                except FileNotFoundError:
                    continue
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
            if retained_directory is not None:
                try:
                    _rollback_created_directory(
                        target_parent,
                        final,
                        retained_directory,
                    )
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
            elif output_parent is not None and output_parent.descriptor is not None:
                try:
                    assert target_parent.descriptor is not None
                    _posix_quarantine_and_delete_directory(
                        target_parent.descriptor,
                        final.name,
                        output_parent.identity,
                    )
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
                finally:
                    os.close(output_parent.descriptor)
                    output_parent.descriptor = None
            if rollback_errors:
                raise RuntimeError(
                    "flat-directory publication failed and ownership-safe rollback was "
                    "incomplete: " + "; ".join(rollback_errors)
                ) from publish_error
            raise


def publish_flat_directory_physical_copy_no_overwrite(
    staged_directory: str | Path,
    destination: str | Path,
    *,
    success_marker_name: str | None = None,
) -> list[PublishedPath]:
    """Physically publish a flat authority bundle with its marker copied last."""

    staged = _absolute_lexical_path(staged_directory)
    if not os.path.lexists(staged):
        raise FileNotFoundError(f"physical bundle staging directory is missing: {staged}")
    staged_value = staged.stat(follow_symlinks=False)
    if not stat.S_ISDIR(staged_value.st_mode) or _is_link_or_reparse(staged, staged_value):
        raise ValueError(
            f"physical bundle staging source must be a lexical regular directory: {staged}"
        )
    if success_marker_name is not None:
        _require_publication_leaf(success_marker_name)

    final = assert_mutable_publication_destination(
        destination,
        role="physical flat-directory publication destination",
    )
    if staged == final or staged in final.parents or final in staged.parents:
        raise ValueError("physical bundle source and destination must not contain one another")

    locked_publications: list[_LockedPublishedFile] = []
    source_identities: dict[str, tuple[int, int, int]] = {}
    with _locked_publication_parents(
        (staged, final.parent),
        final_paths=(final,),
        read_only_paths=(staged,),
    ) as outer_parents:
        staged_parent = outer_parents[_publication_parent_key(staged)]
        staged_parent.assert_current()
        names = tuple(
            sorted(
                os.listdir(
                    staged if staged_parent.native_handle is not None else staged_parent.descriptor
                )
            )
        )
        if not names:
            raise ValueError("physical bundle staging directory must be non-empty")
        for name in names:
            _require_publication_leaf(name)
        if len(set(names)) != len(names):
            raise RuntimeError("physical bundle staging directory has duplicate names")
        if success_marker_name is not None and success_marker_name not in names:
            raise FileNotFoundError(
                f"staged physical success marker is missing: {staged / success_marker_name}"
            )

        ordered_names = names
        if success_marker_name is not None:
            non_markers = tuple(
                name
                for name in names
                if name != success_marker_name
                and not (
                    success_marker_name == IMMUTABLE_MARKER and name == ARTIFACT_MANIFEST_FILENAME
                )
            )
            owned_predecessor_markers = tuple(
                name
                for name in names
                if name != success_marker_name and name == ARTIFACT_MANIFEST_FILENAME
            )
            ordered_names = (*non_markers, *owned_predecessor_markers, success_marker_name)

        sources = tuple(staged / name for name in ordered_names)
        targets = tuple(final / name for name in ordered_names)
        target_parent = outer_parents[_publication_parent_key(final.parent)]
        if _locked_path_exists(final, outer_parents):
            raise FileExistsError(f"refusing to overwrite publication path: {final}")
        output_parent: _LockedPublicationParent | None = None
        retained_directory: _RetainedPublicationHandle | None = None
        try:
            output_parent, retained_directory, directory_value = _create_locked_directory(
                target_parent,
                final,
            )
            assert_mutable_publication_destination(
                final,
                role="physical flat-directory publication destination",
            )
            parents = {
                **outer_parents,
                _publication_parent_key(final): output_parent,
            }
            for source, target in zip(sources, targets, strict=True):
                captured = _capture_physical_publication_source(source, parents)
                try:
                    source_identities[source.name] = captured.identity
                    recognised_success_marker = source.name in {
                        IMMUTABLE_MARKER,
                        ARTIFACT_MANIFEST_FILENAME,
                    } and (
                        source.name == success_marker_name
                        or (
                            success_marker_name == IMMUTABLE_MARKER
                            and source.name == ARTIFACT_MANIFEST_FILENAME
                        )
                    )
                    if recognised_success_marker:
                        assert_mutable_publication_destination(
                            final,
                            role="physical success-marker publication root",
                        )
                        for marker_name in (IMMUTABLE_MARKER, ARTIFACT_MANIFEST_FILENAME):
                            marker_path = final / marker_name
                            if not os.path.lexists(marker_path):
                                continue
                            if not any(
                                value.path == marker_path
                                and value.still_owned()
                                and value._stat().st_nlink == 1
                                for value in locked_publications
                            ):
                                raise PermissionError(
                                    "physical success-marker publication found an unowned "
                                    f"seal marker in its transaction root: {marker_path}"
                                )
                    else:
                        assert_mutable_publication_destination(
                            target,
                            role="physical flat-directory file destination",
                        )
                    locked = _physically_publish_locked_file(captured, target, parents)
                    locked_publications.append(locked)
                    if recognised_success_marker:
                        assert_mutable_publication_destination(
                            final,
                            role="physical success-marker publication root",
                        )
                    else:
                        assert_mutable_publication_destination(
                            target,
                            role="physical flat-directory file destination",
                        )
                finally:
                    captured.close_retained_descriptor()

            actual_names = set(
                os.listdir(final)
                if output_parent.native_handle is not None
                else os.listdir(output_parent.descriptor)
            )
            expected_names = {target.name for target in targets}
            if actual_names != expected_names:
                raise RuntimeError(
                    "physical flat-directory exact-set readback failed: "
                    f"expected={sorted(expected_names)}, observed={sorted(actual_names)}"
                )
            target_parent.assert_current()
            output_parent.assert_current()
            for locked in locked_publications:
                value = locked._stat()
                source_identity = source_identities[locked.name]
                if (
                    not locked.still_owned()
                    or value.st_nlink != 1
                    or locked.identity[:2] == source_identity[:2]
                ):
                    raise OSError(
                        f"physical flat-directory file failed independence readback: {locked.path}"
                    )
            assert_mutable_publication_destination(
                final,
                role="physical flat-directory publication destination",
            )

            file_publications: list[PublishedPath] = []
            for locked in locked_publications:
                value = locked._stat()
                file_publications.append(
                    PublishedPath(
                        path=locked.path,
                        identity=(
                            value.st_dev,
                            value.st_ino,
                            value.st_size,
                            value.st_ctime_ns,
                        ),
                        kind="file",
                        sha256=locked.sha256,
                        required_nlink=1,
                    )
                )
            directory_publication = PublishedPath(
                path=final,
                identity=(
                    directory_value.st_dev,
                    directory_value.st_ino,
                    directory_value.st_size,
                    directory_value.st_ctime_ns,
                ),
                kind="directory",
            )
            for locked in locked_publications:
                locked.close_retained_descriptor()
            if retained_directory is not None:
                retained_directory.close()
                retained_directory = None
            if output_parent.descriptor is not None:
                os.close(output_parent.descriptor)
                output_parent.descriptor = None
            return [directory_publication, *file_publications]
        except BaseException as publish_error:
            rollback_errors: list[str] = []
            for published in reversed(locked_publications):
                try:
                    published.unlink_owned()
                except FileNotFoundError:
                    continue
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
            if retained_directory is not None:
                try:
                    _rollback_created_directory(
                        target_parent,
                        final,
                        retained_directory,
                    )
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
            elif output_parent is not None and output_parent.descriptor is not None:
                try:
                    assert target_parent.descriptor is not None
                    _posix_quarantine_and_delete_directory(
                        target_parent.descriptor,
                        final.name,
                        output_parent.identity,
                    )
                except (OSError, RuntimeError) as error:
                    rollback_errors.append(str(error))
                finally:
                    os.close(output_parent.descriptor)
                    output_parent.descriptor = None
            for published in locked_publications:
                published.close_retained_descriptor()
            if rollback_errors:
                raise RuntimeError(
                    "physical flat-directory publication failed and ownership-safe rollback "
                    "was incomplete: " + "; ".join(rollback_errors)
                ) from publish_error
            raise


def _rollback_one_owned_publication(published: PublishedPath) -> None:
    if not os.path.lexists(published.path):
        return
    with _locked_publication_parents(
        (published.path.parent,),
        final_paths=(),
        create_missing=False,
    ) as parents:
        parent = parents[_publication_parent_key(published.path.parent)]
        if parent.native_handle is not None:
            descriptor: int | None = None
            try:
                descriptor = _windows_open_relative_descriptor(
                    parent.native_handle,
                    published.path.name,
                    delete_access=True,
                    share_delete=False,
                    directory=published.kind == "directory",
                )
                value = os.fstat(descriptor)
                if published.kind == "file":
                    digest = hashlib.sha256()
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    while chunk := os.read(descriptor, 1024 * 1024):
                        digest.update(chunk)
                    if (
                        not stat.S_ISREG(value.st_mode)
                        or (value.st_dev, value.st_ino, value.st_size) != published.identity[:3]
                        or digest.hexdigest() != published.sha256
                    ):
                        raise RuntimeError(
                            f"refused to remove unowned publication: {published.path}"
                        )
                else:
                    if (
                        not stat.S_ISDIR(value.st_mode)
                        or (value.st_dev, value.st_ino) != published.identity[:2]
                    ):
                        raise RuntimeError(
                            f"refused to remove unowned publication: {published.path}"
                        )
                    if os.listdir(published.path):
                        raise RuntimeError(
                            "refused to remove non-empty owned publication directory: "
                            f"{published.path}"
                        )
                _windows_delete_opened_link(descriptor)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        elif published.kind == "file":
            locked = _capture_locked_file(published.path, parents)
            if locked.identity != published.identity[:3] or locked.sha256 != published.sha256:
                raise RuntimeError(f"refused to remove unowned publication: {published.path}")
            locked.unlink_owned()
        else:
            assert parent.descriptor is not None
            _posix_quarantine_and_delete_directory(
                parent.descriptor,
                published.path.name,
                published.identity[:2],
            )

    if os.path.lexists(published.path):
        raise RuntimeError(
            "exact owned object was removed but a foreign logical publication remains: "
            f"{published.path}"
        )


def rollback_owned_publications(publications: list[PublishedPath]) -> None:
    """Remove only paths still proven to belong to the current transaction.

    Each target receives its own parent-anchor transaction.  This matters for a
    flat bundle: handles that stabilise the bundle while child files are removed
    must close before the bundle directory itself can receive delete disposition.
    On Windows ownership verification and deletion use one no-share handle.
    """

    errors: list[str] = []
    for published in reversed(publications):
        try:
            _rollback_one_owned_publication(published)
        except FileNotFoundError:
            continue
        except (OSError, RuntimeError) as error:
            errors.append(f"{published.path}: {error}")
    if errors:
        raise RuntimeError("publication rollback was incomplete: " + "; ".join(errors))


__all__ = [
    "ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY",
    "WOF_LZX_MIN_FREE_MARGIN_BYTES",
    "AnchoredPhysicalCopyBoundaryError",
    "AnchoredPhysicalCopySession",
    "ExclusiveBundlePublicationLock",
    "ExclusivePublicationLock",
    "PublishedPath",
    "anchored_physical_copy_session",
    "assert_mutable_publication_destination",
    "create_directory_no_overwrite",
    "publish_bytes_no_overwrite",
    "publish_file_no_overwrite",
    "publish_file_physical_copy_no_overwrite",
    "publish_flat_directory_no_overwrite",
    "publish_flat_directory_physical_copy_no_overwrite",
    "publish_success_marker_no_overwrite",
    "read_file_anchored",
    "rollback_owned_publications",
]
