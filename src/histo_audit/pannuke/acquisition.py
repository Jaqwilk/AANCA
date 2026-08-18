"""Read-only PanNuke acquisition provenance and Git-safety evidence.

This module never downloads or extracts the dataset.  It inventories the exact
local release, verifies archive integrity and safe member paths, and writes a
small machine-readable provenance record outside the ignored raw-data tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
import zlib
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from histo_audit.utils.run_tracking import sha256_file

from .publication import (
    ExclusiveBundlePublicationLock,
    ExclusivePublicationLock,
    PublishedPath,
    publish_file_no_overwrite,
)

OFFICIAL_SOURCE_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/"
OFFICIAL_SOURCE_PUBLISHER = "University of Warwick Tissue Image Analytics"
LICENSE_SPDX = "CC-BY-NC-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
ACQUISITION_METHOD = "manual_download_from_official_warwick_page"
VERIFICATION_METHOD = (
    "local_streaming_sha256_zip_crc32_safe_member_paths_and_extracted_npy_inventory"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class PanNukeAcquisitionError(ValueError):
    """Raised when local acquisition evidence is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class ArchiveExpectation:
    """Locally recorded identity of one manually acquired official archive."""

    fold: int
    relative_path: str
    size_bytes: int
    sha256: str


DEFAULT_ARCHIVE_EXPECTATIONS = (
    ArchiveExpectation(
        fold=1,
        relative_path="data/raw/pannuke/fold_1.zip",
        size_bytes=700_275_281,
        sha256="6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
    ),
    ArchiveExpectation(
        fold=2,
        relative_path="data/raw/pannuke/fold_2.zip",
        size_bytes=658_842_552,
        sha256="5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
    ),
    ArchiveExpectation(
        fold=3,
        relative_path="data/raw/pannuke/fold_3.zip",
        size_bytes=717_969_882,
        sha256="c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d",
    ),
)

_EXPECTED_NPY_PATHS = {
    fold: {
        "images": f"Fold {fold}/images/fold{fold}/images.npy",
        "types": f"Fold {fold}/images/fold{fold}/types.npy",
        "masks": f"Fold {fold}/masks/fold{fold}/masks.npy",
    }
    for fold in (1, 2, 3)
}

_EXPECTED_DOCUMENT_PATHS = {
    fold: {
        "fold_readme": f"Fold {fold}/README.md",
        "masks_readme": f"Fold {fold}/masks/README.md",
        "masks_license_text": f"Fold {fold}/masks/by-nc-sa.md",
    }
    for fold in (1, 2, 3)
}

_LICENSE_MARKER = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
_LICENSE_README_MARKER = "Attribution-NonCommercial-ShareAlike 4.0 International"
_CITATION_MARKERS = ("gamper2020pannuke", "gamper2019pannuke")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_and_crc32(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> tuple[str, str]:
    """Hash one extracted file once with both required reconciliation digests."""

    sha256 = hashlib.sha256()
    crc32 = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            sha256.update(chunk)
            crc32 = zlib.crc32(chunk, crc32)
    return sha256.hexdigest(), f"{crc32 & 0xFFFFFFFF:08x}"


def _raw_metadata_snapshot(raw_root: Path) -> list[dict[str, Any]]:
    """Capture non-content metadata used to detect writes during verification."""

    return [
        {
            "path": path.relative_to(raw_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "last_write_time_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(
            (candidate for candidate in raw_root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(raw_root).as_posix().casefold(),
        )
    ]


def _require_sha256(value: object, role: str) -> str:
    candidate = str(value)
    if _SHA256.fullmatch(candidate) is None:
        raise PanNukeAcquisitionError(f"{role} must be a lowercase SHA-256")
    return candidate


def _project_relative(path: Path, project_root: Path, role: str) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise PanNukeAcquisitionError(f"{role} lies outside the project root") from error


def _utc_timestamp(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: object) -> str:
    candidate = str(value)
    if _UTC_TIMESTAMP.fullmatch(candidate) is None:
        raise PanNukeAcquisitionError("verification timestamp must be an explicit UTC timestamp")
    try:
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as error:
        raise PanNukeAcquisitionError("verification timestamp is invalid") from error
    return candidate


def validate_zip_member_path(name: str) -> str:
    """Return a canonical ZIP member path or reject extraction-unsafe names."""

    if not name or "\x00" in name or any(ord(character) < 32 for character in name):
        raise PanNukeAcquisitionError("ZIP member path is empty or contains control bytes")
    windows = PureWindowsPath(name)
    if windows.drive or windows.root:
        raise PanNukeAcquisitionError(f"ZIP member path is absolute: {name!r}")
    normalised = name.replace("\\", "/")
    posix = PurePosixPath(normalised)
    if posix.is_absolute():
        raise PanNukeAcquisitionError(f"ZIP member path is absolute: {name!r}")
    parts = posix.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise PanNukeAcquisitionError(f"ZIP member path traverses or aliases: {name!r}")
    for part in parts:
        if ":" in part or part.endswith((" ", ".")):
            raise PanNukeAcquisitionError(f"ZIP member path is unsafe on Windows: {name!r}")
        stem = part.split(".", maxsplit=1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise PanNukeAcquisitionError(f"ZIP member uses a reserved path: {name!r}")
    return "/".join(parts)


def _is_symlink(member: ZipInfo) -> bool:
    mode = (member.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _inspect_zip(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            members: list[dict[str, Any]] = []
            destinations: set[str] = set()
            for member in archive.infolist():
                canonical = validate_zip_member_path(member.filename)
                collision_key = canonical.casefold()
                if collision_key in destinations:
                    raise PanNukeAcquisitionError(
                        f"ZIP contains duplicate/case-alias member path: {canonical!r}"
                    )
                destinations.add(collision_key)
                if _is_symlink(member):
                    raise PanNukeAcquisitionError(
                        f"ZIP contains a symbolic-link member: {canonical!r}"
                    )
                members.append(
                    {
                        "path": canonical,
                        "is_directory": member.is_dir(),
                        "size_bytes": member.file_size,
                        "compressed_size_bytes": member.compress_size,
                        "crc32": f"{member.CRC:08x}",
                    }
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise PanNukeAcquisitionError(f"ZIP CRC failed for member {bad_member!r}")
    except BadZipFile as error:
        raise PanNukeAcquisitionError(f"archive is not a valid ZIP: {path}") from error
    return {
        "zip_crc_status": "passed",
        "zip_crc_failed_member_count": 0,
        "path_safety_status": "passed",
        "rejected_unsafe_member_path_count": 0,
        "rejected_symbolic_link_count": 0,
        "rejected_duplicate_or_case_alias_count": 0,
        "member_count": len(members),
        "uncompressed_size_bytes": sum(int(value["size_bytes"]) for value in members),
        "member_inventory_sha256": _canonical_sha256(members),
        "member_inventory": members,
    }


def _expected_npy_inventory(raw_root: Path) -> list[tuple[int, str, Path]]:
    expected: list[tuple[int, str, Path]] = []
    for fold, roles in _EXPECTED_NPY_PATHS.items():
        for role, relative in roles.items():
            expected.append((fold, role, raw_root / PurePosixPath(relative)))
    return expected


def _expected_document_inventory(raw_root: Path) -> list[tuple[int, str, Path]]:
    expected: list[tuple[int, str, Path]] = []
    for fold, roles in _EXPECTED_DOCUMENT_PATHS.items():
        for role, relative in roles.items():
            expected.append((fold, role, raw_root / PurePosixPath(relative)))
    return expected


def _license_evidence(project_root: Path, raw_root: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        path = raw_root / f"Fold {fold}" / "masks" / "by-nc-sa.md"
        if not path.is_file() or _LICENSE_MARKER not in path.read_text(encoding="utf-8"):
            raise PanNukeAcquisitionError(f"fold {fold} lacks the expected local licence text")
        evidence.append(
            {
                "fold": fold,
                "path": _project_relative(path, project_root, "licence evidence"),
                "sha256": sha256_file(path),
            }
        )
    return evidence


def _license_scope_readme_evidence(project_root: Path, raw_root: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        for role, path in (
            ("fold_readme", raw_root / f"Fold {fold}" / "README.md"),
            ("masks_readme", raw_root / f"Fold {fold}" / "masks" / "README.md"),
        ):
            if not path.is_file() or _LICENSE_README_MARKER not in path.read_text(encoding="utf-8"):
                raise PanNukeAcquisitionError(
                    f"fold {fold} {role} lacks the expected masks licence statement"
                )
            evidence.append(
                {
                    "fold": fold,
                    "role": role,
                    "path": _project_relative(path, project_root, "licence scope evidence"),
                    "sha256": sha256_file(path),
                }
            )
    return evidence


def _citation_evidence(project_root: Path, raw_root: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        for role, path in (
            ("fold_readme", raw_root / f"Fold {fold}" / "README.md"),
            ("masks_readme", raw_root / f"Fold {fold}" / "masks" / "README.md"),
        ):
            if not path.is_file():
                raise PanNukeAcquisitionError(f"fold {fold} lacks {role} citation evidence")
            text = path.read_text(encoding="utf-8")
            if any(marker not in text for marker in _CITATION_MARKERS):
                raise PanNukeAcquisitionError(
                    f"fold {fold} {role} lacks both required PanNuke citation keys"
                )
            evidence.append(
                {
                    "fold": fold,
                    "role": role,
                    "path": _project_relative(path, project_root, "citation evidence"),
                    "sha256": sha256_file(path),
                }
            )
    return evidence


def build_pannuke_acquisition_manifest(
    project_root: str | Path,
    raw_root: str | Path,
    *,
    verification_timestamp_utc: str,
    archive_expectations: Sequence[ArchiveExpectation] = DEFAULT_ARCHIVE_EXPECTATIONS,
) -> dict[str, Any]:
    """Verify the local release read-only and return its strict provenance manifest."""

    project = Path(project_root).resolve()
    raw = Path(raw_root).resolve()
    if not project.is_dir() or not raw.is_dir():
        raise PanNukeAcquisitionError("project and PanNuke raw roots must exist")
    verification_started_monotonic = time.perf_counter()
    raw_snapshot_before = _raw_metadata_snapshot(raw)
    raw_relative = _project_relative(raw, project, "PanNuke raw root")
    timestamp = _validate_timestamp(verification_timestamp_utc)
    expectations = tuple(archive_expectations)
    if {value.fold for value in expectations} != {1, 2, 3} or len(expectations) != 3:
        raise PanNukeAcquisitionError("archive expectations must cover folds 1, 2, and 3 exactly")
    expected_archive_names = {Path(value.relative_path).name for value in expectations}
    actual_archive_names = {value.name for value in raw.glob("*.zip") if value.is_file()}
    if actual_archive_names != expected_archive_names:
        raise PanNukeAcquisitionError("local ZIP inventory differs from the three recorded folds")
    unexpected_top_level_files = {
        value.name
        for value in raw.iterdir()
        if value.is_file() and value.name not in expected_archive_names and value.name != ".gitkeep"
    }
    if unexpected_top_level_files:
        raise PanNukeAcquisitionError(
            "unexpected top-level files exist in the raw release: "
            f"{sorted(unexpected_top_level_files)}"
        )

    archives: list[dict[str, Any]] = []
    for expectation in sorted(expectations, key=lambda value: value.fold):
        _require_sha256(expectation.sha256, f"fold {expectation.fold} expected archive hash")
        archive_path = (project / PurePosixPath(expectation.relative_path)).resolve()
        try:
            archive_path.relative_to(raw)
        except ValueError as error:
            raise PanNukeAcquisitionError(
                "expected archive path lies outside the raw root"
            ) from error
        if not archive_path.is_file():
            raise PanNukeAcquisitionError(f"missing fold {expectation.fold} archive")
        actual_size = archive_path.stat().st_size
        actual_sha256 = sha256_file(archive_path)
        if actual_size != expectation.size_bytes or actual_sha256 != expectation.sha256:
            raise PanNukeAcquisitionError(
                f"fold {expectation.fold} archive differs from recorded local identity"
            )
        inspection = _inspect_zip(archive_path)
        archives.append(
            {
                "fold": expectation.fold,
                "path": _project_relative(archive_path, project, "archive"),
                "size_bytes": actual_size,
                "sha256": actual_sha256,
                "checksum_provenance": "locally_computed_not_publisher_provided",
                "local_last_write_time_utc": _utc_timestamp(archive_path.stat().st_mtime),
                **inspection,
            }
        )

    expected_npy = _expected_npy_inventory(raw)
    expected_documents = _expected_document_inventory(raw)
    expected_archive_members = {
        fold: {
            path.relative_to(raw).as_posix()
            for candidate_fold, _, path in (*expected_npy, *expected_documents)
            if candidate_fold == fold
        }
        for fold in (1, 2, 3)
    }
    for archive in archives:
        actual_members = {
            str(member["path"])
            for member in archive["member_inventory"]
            if not bool(member["is_directory"])
        }
        if actual_members != expected_archive_members[int(archive["fold"])]:
            raise PanNukeAcquisitionError(
                f"fold {archive['fold']} ZIP file inventory differs from official layout"
            )
    expected_extracted_paths = {
        path.resolve() for _, _, path in (*expected_npy, *expected_documents)
    }
    actual_extracted_paths = {
        path.resolve() for path in raw.rglob("*") if path.is_file() and path.parent.resolve() != raw
    }
    if actual_extracted_paths != expected_extracted_paths:
        raise PanNukeAcquisitionError(
            "extracted NPY/README/licence inventory differs from the official layout"
        )

    archive_member_lookup = {
        int(archive["fold"]): {
            str(member["path"]): member
            for member in archive["member_inventory"]
            if not bool(member["is_directory"])
        }
        for archive in archives
    }

    def extracted_record(fold: int, role: str, path: Path, kind: str) -> dict[str, Any]:
        if not path.is_file():
            raise PanNukeAcquisitionError(f"missing extracted fold {fold} {role} file")
        member_path = path.relative_to(raw).as_posix()
        member = archive_member_lookup[fold].get(member_path)
        if member is None:
            raise PanNukeAcquisitionError(
                f"fold {fold} extracted {role} is absent from its source archive"
            )
        sha256, crc32 = _sha256_and_crc32(path)
        if path.stat().st_size != int(member["size_bytes"]) or crc32 != member["crc32"]:
            raise PanNukeAcquisitionError(
                f"fold {fold} extracted {role} differs from its ZIP member"
            )
        return {
            "fold": fold,
            "role": role,
            "kind": kind,
            "path": _project_relative(path, project, f"extracted {kind}"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256,
            "crc32": crc32,
            "archive_member_path": member_path,
            "archive_member_crc32_match": True,
        }

    npy_inventory = [extracted_record(fold, role, path, "npy") for fold, role, path in expected_npy]
    npy_inventory.sort(key=lambda value: (int(value["fold"]), str(value["role"])))
    document_inventory = [
        extracted_record(fold, role, path, "documentation")
        for fold, role, path in expected_documents
    ]
    document_inventory.sort(key=lambda value: (int(value["fold"]), str(value["role"])))

    licence_evidence = _license_evidence(project, raw)
    licence_scope_evidence = _license_scope_readme_evidence(project, raw)
    citation_evidence = _citation_evidence(project, raw)
    raw_snapshot_after = _raw_metadata_snapshot(raw)
    if raw_snapshot_after != raw_snapshot_before:
        raise PanNukeAcquisitionError("raw release changed during read-only verification")
    verification_completed_at = (
        datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
    verification_duration_seconds = round(time.perf_counter() - verification_started_monotonic, 6)

    manifest = {
        "schema_version": 2,
        "dataset": "PanNuke",
        "release": "official_three_fold_release",
        "source": {
            "official_url": OFFICIAL_SOURCE_URL,
            "publisher": OFFICIAL_SOURCE_PUBLISHER,
            "source_kind": "official_publisher_download_page",
        },
        "license": {
            "spdx_id": LICENSE_SPDX,
            "name": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
            "url": LICENSE_URL,
            "scope_statement": (
                "each release README explicitly applies CC BY-NC-SA 4.0 to the "
                "masks directory and its contents"
            ),
            "project_use": "research_noncommercial",
            "local_evidence": licence_evidence,
            "scope_readme_evidence": licence_scope_evidence,
        },
        "citation_requirement": {
            "required": True,
            "reference_keys": ["gamper2019pannuke", "gamper2020pannukeextension"],
            "evidence": "release README requests citation of both PanNuke publications",
            "local_readme_evidence": citation_evidence,
        },
        "acquisition": {
            "method": ACQUISITION_METHOD,
            "verification_timestamp_utc": timestamp,
            "verification_completed_at_utc": verification_completed_at,
            "verification_duration_seconds": verification_duration_seconds,
            "verification_method": VERIFICATION_METHOD,
            "raw_root": raw_relative,
            "download_performed_by_this_software": False,
            "extraction_performed_by_this_software": False,
        },
        "immutable_raw_policy": {
            "archives_retained_unchanged": True,
            "extracted_arrays_retained_unchanged": True,
            "automatic_source_annotation_modification_forbidden": True,
            "derived_outputs_must_be_outside_raw_root": True,
            "git_tracking_for_raw_release_forbidden": True,
        },
        "archives": archives,
        "extracted_npy_inventory": npy_inventory,
        "extracted_npy_inventory_sha256": _canonical_sha256(npy_inventory),
        "extracted_document_inventory": document_inventory,
        "extracted_document_inventory_sha256": _canonical_sha256(document_inventory),
        "extracted_file_inventory_sha256": _canonical_sha256([*npy_inventory, *document_inventory]),
        "raw_release_read_only_verification": {
            "status": "passed",
            "pre_post_path_size_mtime_match": True,
            "regular_file_count": len(raw_snapshot_after),
            "metadata_snapshot_sha256": _canonical_sha256(raw_snapshot_after),
        },
    }
    validate_acquisition_manifest(manifest)
    return manifest


def _require_mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PanNukeAcquisitionError(f"{role} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise PanNukeAcquisitionError(f"{role} keys differ from schema")


def validate_acquisition_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the complete acquisition-manifest schema and semantic bindings."""

    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "dataset",
            "release",
            "source",
            "license",
            "citation_requirement",
            "acquisition",
            "immutable_raw_policy",
            "archives",
            "extracted_npy_inventory",
            "extracted_npy_inventory_sha256",
            "extracted_document_inventory",
            "extracted_document_inventory_sha256",
            "extracted_file_inventory_sha256",
            "raw_release_read_only_verification",
        },
        "acquisition manifest",
    )
    if (
        manifest["schema_version"] != 2
        or manifest["dataset"] != "PanNuke"
        or manifest["release"] != "official_three_fold_release"
    ):
        raise PanNukeAcquisitionError("acquisition manifest identity is invalid")
    source = _require_mapping(manifest["source"], "source")
    if source != {
        "official_url": OFFICIAL_SOURCE_URL,
        "publisher": OFFICIAL_SOURCE_PUBLISHER,
        "source_kind": "official_publisher_download_page",
    }:
        raise PanNukeAcquisitionError("source provenance differs from the verified official source")
    licence = _require_mapping(manifest["license"], "license")
    if (
        licence.get("spdx_id") != LICENSE_SPDX
        or licence.get("url") != LICENSE_URL
        or licence.get("project_use") != "research_noncommercial"
        or "masks directory and its contents" not in str(licence.get("scope_statement"))
    ):
        raise PanNukeAcquisitionError("licence provenance is invalid")
    local_evidence = licence.get("local_evidence")
    if not isinstance(local_evidence, list) or len(local_evidence) != 3:
        raise PanNukeAcquisitionError("licence evidence must cover all three folds")
    if {int(value["fold"]) for value in local_evidence if isinstance(value, Mapping)} != {
        1,
        2,
        3,
    }:
        raise PanNukeAcquisitionError("licence evidence fold coverage is invalid")
    for value in local_evidence:
        record = _require_mapping(value, "licence evidence record")
        _require_sha256(record.get("sha256"), "licence evidence hash")
        if not str(record.get("path", "")).startswith("data/raw/pannuke/"):
            raise PanNukeAcquisitionError("licence evidence path is outside the raw release")
    scope_evidence = licence.get("scope_readme_evidence")
    if not isinstance(scope_evidence, list) or len(scope_evidence) != 6:
        raise PanNukeAcquisitionError("licence scope evidence must cover both READMEs per fold")
    if {
        (int(value["fold"]), str(value["role"]))
        for value in scope_evidence
        if isinstance(value, Mapping)
    } != {(fold, role) for fold in (1, 2, 3) for role in ("fold_readme", "masks_readme")}:
        raise PanNukeAcquisitionError("licence scope evidence fold/README coverage is invalid")
    for value in scope_evidence:
        record = _require_mapping(value, "licence scope evidence record")
        _require_exact_keys(record, {"fold", "role", "path", "sha256"}, "licence scope evidence")
        _require_sha256(record["sha256"], "licence scope evidence hash")
        if not str(record["path"]).startswith("data/raw/pannuke/"):
            raise PanNukeAcquisitionError("licence scope evidence path is outside raw release")
    citation = _require_mapping(manifest["citation_requirement"], "citation requirement")
    if citation.get("required") is not True or citation.get("reference_keys") != [
        "gamper2019pannuke",
        "gamper2020pannukeextension",
    ]:
        raise PanNukeAcquisitionError("citation requirement is invalid")
    citation_evidence = citation.get("local_readme_evidence")
    if not isinstance(citation_evidence, list) or len(citation_evidence) != 6:
        raise PanNukeAcquisitionError("citation evidence must cover both READMEs in every fold")
    if {
        (int(value["fold"]), str(value["role"]))
        for value in citation_evidence
        if isinstance(value, Mapping)
    } != {(fold, role) for fold in (1, 2, 3) for role in ("fold_readme", "masks_readme")}:
        raise PanNukeAcquisitionError("citation evidence fold/README coverage is invalid")
    for value in citation_evidence:
        record = _require_mapping(value, "citation evidence record")
        _require_exact_keys(record, {"fold", "role", "path", "sha256"}, "citation evidence")
        _require_sha256(record["sha256"], "citation evidence hash")
        if not str(record["path"]).startswith("data/raw/pannuke/"):
            raise PanNukeAcquisitionError("citation evidence path is outside the raw release")
    acquisition = _require_mapping(manifest["acquisition"], "acquisition")
    started_at = _validate_timestamp(acquisition.get("verification_timestamp_utc"))
    completed_at = _validate_timestamp(acquisition.get("verification_completed_at_utc"))
    if (
        acquisition.get("method") != ACQUISITION_METHOD
        or acquisition.get("verification_method") != VERIFICATION_METHOD
        or acquisition.get("raw_root") != "data/raw/pannuke"
        or acquisition.get("download_performed_by_this_software") is not False
        or acquisition.get("extraction_performed_by_this_software") is not False
        or float(acquisition.get("verification_duration_seconds", -1.0)) < 0.0
        or datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        < datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ):
        raise PanNukeAcquisitionError("acquisition method/root evidence is invalid")
    policy = _require_mapping(manifest["immutable_raw_policy"], "immutable raw policy")
    if not policy or any(value is not True for value in policy.values()):
        raise PanNukeAcquisitionError("immutable raw policy must be wholly enabled")

    archives = manifest["archives"]
    if not isinstance(archives, list) or len(archives) != 3:
        raise PanNukeAcquisitionError("archive inventory must contain three records")
    if {int(value["fold"]) for value in archives if isinstance(value, Mapping)} != {1, 2, 3}:
        raise PanNukeAcquisitionError("archive fold coverage is invalid")
    for value in archives:
        record = _require_mapping(value, "archive record")
        _require_exact_keys(
            record,
            {
                "fold",
                "path",
                "size_bytes",
                "sha256",
                "checksum_provenance",
                "local_last_write_time_utc",
                "zip_crc_status",
                "zip_crc_failed_member_count",
                "path_safety_status",
                "rejected_unsafe_member_path_count",
                "rejected_symbolic_link_count",
                "rejected_duplicate_or_case_alias_count",
                "member_count",
                "uncompressed_size_bytes",
                "member_inventory_sha256",
                "member_inventory",
            },
            "archive record",
        )
        _require_sha256(record["sha256"], "archive hash")
        _require_sha256(record["member_inventory_sha256"], "archive member inventory hash")
        if (
            record["zip_crc_status"] != "passed"
            or record["path_safety_status"] != "passed"
            or any(
                int(record[key]) != 0
                for key in (
                    "zip_crc_failed_member_count",
                    "rejected_unsafe_member_path_count",
                    "rejected_symbolic_link_count",
                    "rejected_duplicate_or_case_alias_count",
                )
            )
            or record["checksum_provenance"] != "locally_computed_not_publisher_provided"
            or int(record["size_bytes"]) <= 0
            or int(record["member_count"]) <= 0
            or int(record["uncompressed_size_bytes"]) <= 0
        ):
            raise PanNukeAcquisitionError("archive integrity evidence is invalid")
        _validate_timestamp(record["local_last_write_time_utc"])
        if not str(record["path"]).startswith("data/raw/pannuke/"):
            raise PanNukeAcquisitionError("archive path is outside the raw release")
        members = record["member_inventory"]
        if not isinstance(members, list) or len(members) != int(record["member_count"]):
            raise PanNukeAcquisitionError("archive member inventory count is invalid")
        if record["member_inventory_sha256"] != _canonical_sha256(members):
            raise PanNukeAcquisitionError("archive member inventory checksum is invalid")
        for member_value in members:
            member = _require_mapping(member_value, "archive member")
            _require_exact_keys(
                member,
                {
                    "path",
                    "is_directory",
                    "size_bytes",
                    "compressed_size_bytes",
                    "crc32",
                },
                "archive member",
            )
            validate_zip_member_path(str(member["path"]))
            if re.fullmatch(r"[0-9a-f]{8}", str(member["crc32"])) is None:
                raise PanNukeAcquisitionError("archive member CRC32 is invalid")
            if int(member["size_bytes"]) < 0 or int(member["compressed_size_bytes"]) < 0:
                raise PanNukeAcquisitionError("archive member size is invalid")

    inventory = manifest["extracted_npy_inventory"]
    if not isinstance(inventory, list) or len(inventory) != 9:
        raise PanNukeAcquisitionError("extracted NPY inventory must contain nine records")
    identities = {
        (int(value["fold"]), str(value["role"]))
        for value in inventory
        if isinstance(value, Mapping)
    }
    if identities != {(fold, role) for fold in (1, 2, 3) for role in ("images", "masks", "types")}:
        raise PanNukeAcquisitionError("extracted NPY fold/role coverage is invalid")
    archive_member_by_fold = {
        int(archive["fold"]): {
            str(member["path"]): member
            for member in archive["member_inventory"]
            if not bool(member["is_directory"])
        }
        for archive in archives
    }

    def validate_extracted_record(value: object, expected_kind: str) -> Mapping[str, Any]:
        record = _require_mapping(value, f"extracted {expected_kind} record")
        _require_exact_keys(
            record,
            {
                "fold",
                "role",
                "kind",
                "path",
                "size_bytes",
                "sha256",
                "crc32",
                "archive_member_path",
                "archive_member_crc32_match",
            },
            f"extracted {expected_kind} record",
        )
        _require_sha256(record["sha256"], f"extracted {expected_kind} hash")
        if (
            record["kind"] != expected_kind
            or int(record["size_bytes"]) <= 0
            or not str(record["path"]).startswith("data/raw/pannuke/")
            or record["archive_member_crc32_match"] is not True
            or re.fullmatch(r"[0-9a-f]{8}", str(record["crc32"])) is None
        ):
            raise PanNukeAcquisitionError(f"extracted {expected_kind} record is invalid")
        member = archive_member_by_fold[int(record["fold"])].get(str(record["archive_member_path"]))
        if (
            member is None
            or int(member["size_bytes"]) != int(record["size_bytes"])
            or member["crc32"] != record["crc32"]
        ):
            raise PanNukeAcquisitionError(
                f"extracted {expected_kind} record is not bound to its archive member"
            )
        return record

    for value in inventory:
        validate_extracted_record(value, "npy")
    if manifest["extracted_npy_inventory_sha256"] != _canonical_sha256(inventory):
        raise PanNukeAcquisitionError("extracted NPY inventory checksum is invalid")

    documents = manifest["extracted_document_inventory"]
    if not isinstance(documents, list) or len(documents) != 9:
        raise PanNukeAcquisitionError("extracted document inventory must contain nine records")
    document_identities = {
        (int(value["fold"]), str(value["role"]))
        for value in documents
        if isinstance(value, Mapping)
    }
    if document_identities != {
        (fold, role)
        for fold in (1, 2, 3)
        for role in ("fold_readme", "masks_readme", "masks_license_text")
    }:
        raise PanNukeAcquisitionError("extracted document fold/role coverage is invalid")
    for value in documents:
        validate_extracted_record(value, "documentation")
    if manifest["extracted_document_inventory_sha256"] != _canonical_sha256(documents):
        raise PanNukeAcquisitionError("extracted document inventory checksum is invalid")
    if manifest["extracted_file_inventory_sha256"] != _canonical_sha256([*inventory, *documents]):
        raise PanNukeAcquisitionError("combined extracted-file inventory checksum is invalid")

    read_only = _require_mapping(
        manifest["raw_release_read_only_verification"], "raw read-only verification"
    )
    _require_exact_keys(
        read_only,
        {
            "status",
            "pre_post_path_size_mtime_match",
            "regular_file_count",
            "metadata_snapshot_sha256",
        },
        "raw read-only verification",
    )
    _require_sha256(read_only["metadata_snapshot_sha256"], "raw metadata snapshot hash")
    if (
        read_only["status"] != "passed"
        or read_only["pre_post_path_size_mtime_match"] is not True
        or int(read_only["regular_file_count"]) < 21
    ):
        raise PanNukeAcquisitionError("raw read-only verification is invalid")


def verify_acquisition_raw_metadata_unchanged(
    manifest: Mapping[str, Any], project_root: str | Path
) -> list[dict[str, Any]]:
    """Require the complete raw path/size/mtime snapshot bound by the manifest."""

    validate_acquisition_manifest(manifest)
    project = Path(project_root).resolve()
    raw = (project / "data" / "raw" / "pannuke").resolve()
    if not raw.is_dir():
        raise PanNukeAcquisitionError("acquisition raw root is missing")
    first = _raw_metadata_snapshot(raw)
    second = _raw_metadata_snapshot(raw)
    if first != second:
        raise PanNukeAcquisitionError(
            "raw release changed while acquisition publication inventory was captured"
        )
    expected = _require_mapping(
        manifest["raw_release_read_only_verification"], "raw read-only verification"
    )
    if (
        len(second) != int(expected["regular_file_count"])
        or _canonical_sha256(second) != expected["metadata_snapshot_sha256"]
    ):
        raise PanNukeAcquisitionError(
            "raw release path/size/mtime inventory changed after acquisition verification"
        )
    return second


def write_json_compare_and_swap(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None = None,
) -> Path:
    """Atomically create JSON or update it only against an exact previous SHA."""

    destination = _lexical_json_destination(path)
    with (
        ExclusiveBundlePublicationLock((destination,), role="provenance JSON"),
        # Retain interoperability with older publication callers while every
        # PanNuke bundle migrates to the shared O_EXCL target-lock scheme.
        ExclusivePublicationLock(destination, role="provenance JSON"),
    ):
        write_required, previous = _preflight_json_destination(
            destination,
            payload,
            expected_previous_sha256=expected_previous_sha256,
        )
        if not write_required:
            return destination
        publication = _publish_json_cas_locked(destination, payload, previous)
        try:
            if json.loads(destination.read_text(encoding="utf-8")) != payload:
                raise PanNukeAcquisitionError("persisted provenance JSON is invalid")
        except BaseException as error:
            _rollback_json_publication(publication, previous, error)
            raise
        return destination


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{content}\n".encode()


def _lexical_json_destination(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _stage_json_payload(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.cas-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _anticipated_json_publication(staged: Path, destination: Path) -> PublishedPath:
    """Capture the staged file's identity before it can become externally visible."""

    value = staged.stat(follow_symlinks=False)
    return PublishedPath(
        path=destination,
        identity=(value.st_dev, value.st_ino, value.st_size, value.st_ctime_ns),
        kind="file",
        sha256=sha256_file(staged),
    )


def _publish_json_cas_locked(
    destination: Path,
    payload: Mapping[str, Any],
    previous: bytes | None,
) -> PublishedPath:
    staged = _stage_json_payload(destination, payload)
    try:
        if previous is None:
            return publish_file_no_overwrite(staged, destination)
        if not os.path.lexists(destination) or destination.read_bytes() != previous:
            raise PanNukeAcquisitionError(
                f"existing provenance artifact changed immediately before update: {destination}"
            )
        publication = _anticipated_json_publication(staged, destination)
        os.replace(staged, destination)
        if not publication.still_owned():
            raise RuntimeError(
                "updated provenance artifact ownership changed during CAS publication; "
                f"foreign destination preserved: {destination}"
            )
        return publication
    finally:
        staged.unlink(missing_ok=True)


def _rollback_json_publication(
    publication: PublishedPath,
    previous: bytes | None,
    publish_error: BaseException,
) -> None:
    if not os.path.lexists(publication.path):
        return
    if not publication.still_owned():
        raise RuntimeError(
            "provenance rollback refused to mutate a destination whose ownership changed"
        ) from publish_error
    if previous is None:
        publication.path.unlink()
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{publication.path.name}.rollback-",
        suffix=".tmp",
        dir=publication.path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous)
            handle.flush()
            os.fsync(handle.fileno())
        if not publication.still_owned():
            raise RuntimeError(
                "provenance rollback ownership changed before restoration"
            ) from publish_error
        os.replace(temporary, publication.path)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_json_destination(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None,
) -> tuple[bool, bytes | None]:
    """Return whether a CAS write is needed and retain rollback bytes."""

    if not os.path.lexists(path):
        if expected_previous_sha256 is not None:
            raise PanNukeAcquisitionError(
                f"compare-and-swap expected an existing provenance artifact: {path}"
            )
        return True, None
    previous = path.read_bytes()
    try:
        current = json.loads(previous.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PanNukeAcquisitionError(
            f"existing provenance artifact is unreadable: {path}"
        ) from error
    if current == payload:
        return False, previous
    if expected_previous_sha256 is None:
        raise FileExistsError(
            f"refusing to overwrite provenance artifact without compare-and-swap: {path}"
        )
    expected = _require_sha256(expected_previous_sha256, "expected previous artifact hash")
    if hashlib.sha256(previous).hexdigest() != expected:
        raise PanNukeAcquisitionError(f"existing provenance artifact changed before update: {path}")
    return True, previous


def write_acquisition_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None = None,
) -> Path:
    """Validate and atomically persist one acquisition manifest."""

    validate_acquisition_manifest(manifest)
    return write_json_compare_and_swap(
        path,
        manifest,
        expected_previous_sha256=expected_previous_sha256,
    )


def git_ignore_evidence(project_root: str | Path) -> dict[str, Any]:
    """Prove that the complete raw release is ignored and evidence stays trackable."""

    project = Path(project_root).resolve()
    raw_root = project / "data" / "raw" / "pannuke"
    ignored_candidates = tuple(
        path.relative_to(project).as_posix()
        for path in sorted(
            (
                candidate
                for candidate in raw_root.rglob("*")
                if candidate.is_file() and candidate.name != ".gitkeep"
            ),
            key=lambda candidate: candidate.relative_to(project).as_posix().casefold(),
        )
    )
    if not ignored_candidates:
        raise PanNukeAcquisitionError("no local raw release files were found for Git-safety proof")
    ignored: list[dict[str, str]] = []
    for candidate in ignored_candidates:
        result = subprocess.run(
            ["git", "check-ignore", "-v", "--", candidate],
            cwd=project,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise PanNukeAcquisitionError(f"raw data path is not ignored by Git: {candidate}")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", candidate],
            cwd=project,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode == 0:
            raise PanNukeAcquisitionError(f"raw data path is already tracked by Git: {candidate}")
        if tracked.returncode != 1:
            raise PanNukeAcquisitionError("git ls-files failed during raw-data safety proof")
        ignored.append({"path": candidate, "git_check_ignore": result.stdout.strip()})
    trackable = (
        "data/manifests/pannuke_acquisition.json",
        "reports/pannuke_acquisition_verification.json",
    )
    for candidate in trackable:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", candidate],
            cwd=project,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise PanNukeAcquisitionError(
                f"provenance artifact is unexpectedly ignored: {candidate}"
            )
        if result.returncode not in {1}:
            raise PanNukeAcquisitionError("git check-ignore failed")
    status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "data/raw/pannuke",
        ],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise PanNukeAcquisitionError("git status failed during raw-data safety proof")
    visible = [line for line in status.stdout.splitlines() if line.strip()]
    if any(not line.endswith("data/raw/pannuke/.gitkeep") for line in visible):
        raise PanNukeAcquisitionError("a raw release file is visible in Git status")
    return {
        "status": "passed",
        "raw_release_file_count": len(ignored_candidates),
        "ignored_raw_file_count": len(ignored),
        "tracked_raw_file_count": 0,
        "ignored_raw_paths": ignored,
        "trackable_provenance_paths": list(trackable),
        "git_status_visible_raw_paths": visible,
    }


def build_acquisition_verification_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a compact report binding acquisition evidence to Git safety."""

    validate_acquisition_manifest(manifest)
    path = Path(manifest_path)
    supplied_sha = (
        _require_sha256(manifest_sha256, "acquisition manifest hash")
        if manifest_sha256 is not None
        else None
    )
    staged_sha = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    if supplied_sha is not None and supplied_sha != staged_sha:
        raise PanNukeAcquisitionError("supplied acquisition manifest hash is not payload-bound")
    if supplied_sha is not None:
        bound_sha = supplied_sha
    elif path.is_file():
        bound_sha = sha256_file(path)
        if bound_sha != staged_sha:
            raise PanNukeAcquisitionError("existing acquisition manifest differs from payload")
    else:
        raise PanNukeAcquisitionError(
            "acquisition manifest must exist or have a staged SHA before report creation"
        )
    acquisition = _require_mapping(manifest["acquisition"], "acquisition")
    return {
        "schema_version": 2,
        "status": "passed",
        "scope": "acquisition_provenance_only",
        "dataset": "PanNuke",
        "verification_timestamp_utc": acquisition["verification_timestamp_utc"],
        "verification_completed_at_utc": acquisition["verification_completed_at_utc"],
        "verification_duration_seconds": acquisition["verification_duration_seconds"],
        "acquisition_manifest_path": _project_relative(
            path, Path(project_root), "acquisition manifest"
        ),
        "acquisition_manifest_sha256": bound_sha,
        "archive_count": len(manifest["archives"]),
        "extracted_npy_count": len(manifest["extracted_npy_inventory"]),
        "extracted_document_count": len(manifest["extracted_document_inventory"]),
        "all_archive_crc_checks_passed": True,
        "archive_crc_failed_member_count": sum(
            int(value["zip_crc_failed_member_count"]) for value in manifest["archives"]
        ),
        "all_archive_path_safety_checks_passed": True,
        "rejected_unsafe_archive_member_path_count": sum(
            int(value["rejected_unsafe_member_path_count"]) for value in manifest["archives"]
        ),
        "all_extracted_files_match_archive_member_crc32": all(
            bool(value["archive_member_crc32_match"])
            for value in [
                *manifest["extracted_npy_inventory"],
                *manifest["extracted_document_inventory"],
            ]
        ),
        "raw_release_is_immutable": True,
        "git_safety": git_ignore_evidence(project_root),
        "scientific_stage_advanced": False,
    }


def write_acquisition_artifact_bundle(
    manifest_path: str | Path,
    report_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    project_root: str | Path,
    expected_previous_manifest_sha256: str | None = None,
    expected_previous_report_sha256: str | None = None,
) -> tuple[Path, Path]:
    """CAS-preflight and write the bound manifest/report pair fail-closed."""

    validate_acquisition_manifest(manifest)
    manifest_destination = _lexical_json_destination(manifest_path)
    report_destination = _lexical_json_destination(report_path)
    if manifest_destination.resolve() == report_destination.resolve():
        raise PanNukeAcquisitionError("manifest and report destinations must differ")
    destinations = (manifest_destination, report_destination)
    with ExclusiveBundlePublicationLock(
        destinations,
        role="acquisition provenance bundle",
    ) as bundle_lock:
        # The one O_EXCL lock spans input verification, payload construction,
        # CAS preflight, staging/publication, readback, and any rollback.
        bundle_lock.assert_owned()
        verify_acquisition_raw_metadata_unchanged(manifest, project_root)
        manifest_sha = hashlib.sha256(_json_bytes(manifest)).hexdigest()
        report = build_acquisition_verification_report(
            manifest,
            manifest_path=manifest_destination,
            project_root=project_root,
            manifest_sha256=manifest_sha,
        )
        verify_acquisition_raw_metadata_unchanged(manifest, project_root)
        publications: list[tuple[PublishedPath, bytes | None]] = []
        with ExitStack() as locks:
            for destination in sorted(destinations, key=lambda value: str(value).casefold()):
                locks.enter_context(
                    ExclusivePublicationLock(destination, role="acquisition provenance bundle")
                )
            bundle_lock.assert_owned()
            write_manifest, previous_manifest = _preflight_json_destination(
                manifest_destination,
                manifest,
                expected_previous_sha256=expected_previous_manifest_sha256,
            )
            write_report, previous_report = _preflight_json_destination(
                report_destination,
                report,
                expected_previous_sha256=expected_previous_report_sha256,
            )
            try:
                # The bound report is ancillary; canonical manifest JSON is the
                # success marker and is deliberately published last.
                if write_report:
                    bundle_lock.assert_owned()
                    publications.append(
                        (
                            _publish_json_cas_locked(report_destination, report, previous_report),
                            previous_report,
                        )
                    )
                if write_manifest:
                    bundle_lock.assert_owned()
                    publications.append(
                        (
                            _publish_json_cas_locked(
                                manifest_destination, manifest, previous_manifest
                            ),
                            previous_manifest,
                        )
                    )
                bundle_lock.assert_owned()
                if sha256_file(manifest_destination) != manifest_sha:
                    raise PanNukeAcquisitionError("persisted acquisition manifest hash is invalid")
                persisted_report = json.loads(report_destination.read_text(encoding="utf-8"))
                if persisted_report != report:
                    raise PanNukeAcquisitionError("persisted acquisition report is invalid")
                verify_acquisition_raw_metadata_unchanged(manifest, project_root)
            except BaseException as publish_error:
                rollback_errors: list[str] = []
                for publication, previous in reversed(publications):
                    try:
                        _rollback_json_publication(publication, previous, publish_error)
                    except RuntimeError as rollback_error:
                        rollback_errors.append(str(rollback_error))
                if rollback_errors:
                    raise RuntimeError(
                        "acquisition bundle rollback was incomplete: " + "; ".join(rollback_errors)
                    ) from publish_error
                raise
    return manifest_destination, report_destination


__all__ = [
    "ACQUISITION_METHOD",
    "DEFAULT_ARCHIVE_EXPECTATIONS",
    "LICENSE_SPDX",
    "LICENSE_URL",
    "OFFICIAL_SOURCE_PUBLISHER",
    "OFFICIAL_SOURCE_URL",
    "VERIFICATION_METHOD",
    "ArchiveExpectation",
    "PanNukeAcquisitionError",
    "build_acquisition_verification_report",
    "build_pannuke_acquisition_manifest",
    "git_ignore_evidence",
    "validate_acquisition_manifest",
    "validate_zip_member_path",
    "verify_acquisition_raw_metadata_unchanged",
    "write_acquisition_artifact_bundle",
    "write_acquisition_manifest",
    "write_json_compare_and_swap",
]
