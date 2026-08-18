"""Fail-closed preregistration freezing after a verified real-data pilot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from histo_audit.config import config_sha256, load_config
from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    create_directory_no_overwrite,
    publish_file_no_overwrite,
    publish_success_marker_no_overwrite,
    rollback_owned_publications,
)
from histo_audit.utils.run_tracking import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    capture_git_state,
    capture_governance_tree,
    capture_source_tree,
    require_run_stage_eligible,
    sha256_file,
    verify_run_integrity,
    windows_compatible_relative_path_sort_key,
)

BASE_FREEZE_EVIDENCE_SCHEMA_VERSION = 3

_REQUIRED_PREREGISTRATION_SECTIONS = (
    "primary design",
    "dataset and split",
    "corruption",
    "representations and models",
    "audit methods",
    "metrics and statistics",
    "exclusions and missing data",
    "final-test policy",
    "amendments",
)
_INCOMPLETE_TOKEN = re.compile(
    r"\b(?:draft|unresolved|unset|tbd|todo|placeholder|not[ -]frozen)\b", re.IGNORECASE
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PreregistrationFreezeResult:
    """Paths and cryptographic evidence for one immutable freeze snapshot."""

    freeze_directory: Path
    frozen_preregistration_path: Path
    frozen_primary_config_path: Path
    timestamped_primary_config_path: Path
    frozen_confirmatory_config_path: Path
    timestamped_confirmatory_config_path: Path
    duplicate_audit_snapshot_path: Path
    pathology_encoder_audit_snapshot_path: Path
    representation_independence_snapshot_path: Path
    pilot_derived_parameters_snapshot_path: Path
    raw_checksum_snapshot_path: Path
    pilot_post_seal_verification_path: Path
    sha256_manifest_path: Path
    immutable_marker_path: Path
    freeze_timestamp_utc: str
    pilot_run_id: str
    artifact_root_sha256: str
    sha256_manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible freeze evidence."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class PreregistrationFreezeVerification:
    """Independent checksum verification for a frozen preregistration directory."""

    valid: bool
    artifact_root_sha256: str | None
    expected_artifact_root_sha256: str | None
    missing_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    errors: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True, slots=True)
class _ChecksumRecord:
    relative_path: str
    sha256: str
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class _DatasetSnapshot:
    dataset_sha256: str
    records: tuple[dict[str, Any], ...]
    raw_inventory_records: tuple[dict[str, Any], ...]
    raw_inventory_sha256: str


def _is_reparse_or_link(path: Path, value: os.stat_result | None = None) -> bool:
    observed = value or path.stat(follow_symlinks=False)
    attributes = int(getattr(observed, "st_file_attributes", 0))
    return path.is_symlink() or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _stable_file_record(path: Path, *, relative_path: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"dataset inventory entry is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while its checksum was being captured: {path}")
    return {
        "relative_path": relative_path,
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def _dataset_tree_records(dataset: Path) -> tuple[dict[str, Any], ...]:
    if dataset.is_file():
        if _is_reparse_or_link(dataset):
            raise ValueError(
                f"dataset path must not be a symbolic link or reparse point: {dataset}"
            )
        return (_stable_file_record(dataset, relative_path=dataset.name),)
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {dataset}")
    if _is_reparse_or_link(dataset):
        raise ValueError(f"dataset root must not be a symbolic link or reparse point: {dataset}")
    records: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for current, directory_names, file_names in os.walk(dataset, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            child = current_path / name
            value = child.stat(follow_symlinks=False)
            if not stat.S_ISDIR(value.st_mode) or _is_reparse_or_link(child, value):
                raise ValueError(
                    f"dataset tree contains a non-directory link or reparse point: {child}"
                )
        directory_names.sort()
        for name in sorted(file_names):
            child = current_path / name
            value = child.stat(follow_symlinks=False)
            if not stat.S_ISREG(value.st_mode) or _is_reparse_or_link(child, value):
                raise ValueError(f"dataset tree contains a non-regular file: {child}")
            relative = child.relative_to(dataset).as_posix()
            alias = relative.casefold()
            previous = aliases.get(alias)
            if previous is not None and previous != relative:
                raise ValueError(
                    "dataset inventory contains a duplicate or case-alias path: "
                    f"{previous!r}, {relative!r}"
                )
            aliases[alias] = relative
            records.append(_stable_file_record(child, relative_path=relative))
    return tuple(sorted(records, key=lambda record: str(record["relative_path"])))


def _dataset_sha256(dataset: Path, records: Sequence[Mapping[str, Any]]) -> str:
    if dataset.is_file():
        if len(records) != 1:
            raise RuntimeError("file dataset checksum capture did not contain exactly one record")
        return str(records[0]["sha256"])
    digest = hashlib.sha256()
    # RunTracker seals dataset directories through ``sha256_path()``.  Reorder
    # only this digest view with its shared, platform-independent key so freeze
    # verification reproduces the sealed authority without changing the stable,
    # case-sensitive POSIX inventory records or their semantic hash.
    digest_records = sorted(
        records,
        key=lambda record: windows_compatible_relative_path_sort_key(str(record["relative_path"])),
    )
    for record in digest_records:
        relative = str(record["relative_path"]).encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(record["sha256"])))
    return digest.hexdigest()


def _normalise_inventory_path(
    value: object,
    *,
    dataset: Path,
    project_root: Path,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("raw checksum inventory contains an empty or non-string path")
    rendered = value.strip().replace("\\", "/")
    windows_absolute = bool(re.fullmatch(r"[A-Za-z]:/.*", rendered))
    if PurePosixPath(rendered).is_absolute() or windows_absolute:
        absolute = Path(rendered).resolve()
        try:
            return absolute.relative_to(dataset).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"raw checksum inventory path is outside the exact dataset: {value}"
            ) from exc
    relative = PurePosixPath(rendered)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"raw checksum inventory path is unsafe: {value}")
    try:
        dataset_from_project = dataset.relative_to(project_root).as_posix()
    except ValueError:
        dataset_from_project = ""
    if dataset_from_project and (
        rendered == dataset_from_project or rendered.startswith(f"{dataset_from_project}/")
    ):
        rendered = rendered[len(dataset_from_project) :].lstrip("/")
    if not rendered:
        raise ValueError(f"raw checksum inventory path names the dataset root, not a file: {value}")
    return PurePosixPath(rendered).as_posix()


def _checksum_records_from_values(
    values: object,
    *,
    dataset: Path,
    project_root: Path,
) -> tuple[_ChecksumRecord, ...]:
    raw_records: list[tuple[object, object, object]] = []
    if isinstance(values, Mapping):
        for path, evidence in values.items():
            if isinstance(evidence, str):
                raw_records.append((path, evidence, None))
            elif isinstance(evidence, Mapping):
                raw_records.append((path, evidence.get("sha256"), evidence.get("size_bytes")))
            else:
                raise ValueError("raw checksum mapping values must be SHA-256 strings or objects")
    elif isinstance(values, list):
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("raw checksum inventory records must be objects")
            raw_records.append(
                (
                    value.get("relative_path", value.get("path")),
                    value.get("sha256"),
                    value.get("size_bytes"),
                )
            )
    else:
        raise ValueError("raw checksum inventory must be a path-to-hash mapping or record list")
    if not raw_records:
        raise ValueError("raw checksum inventory is empty")
    parsed: list[_ChecksumRecord] = []
    aliases: dict[str, str] = {}
    for raw_path, raw_sha, raw_size in raw_records:
        relative = _normalise_inventory_path(
            raw_path,
            dataset=dataset,
            project_root=project_root,
        )
        digest = str(raw_sha).casefold() if isinstance(raw_sha, str) else ""
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"raw checksum inventory has an invalid SHA-256 for {relative}")
        if isinstance(raw_size, str) and raw_size.strip().isdigit():
            raw_size = int(raw_size.strip())
        if raw_size in (None, ""):
            size = None
        elif isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(f"raw checksum inventory has an invalid size for {relative}")
        else:
            size = raw_size
        alias = relative.casefold()
        previous = aliases.get(alias)
        if previous is not None:
            raise ValueError(
                "raw checksum inventory contains a duplicate or case-alias path: "
                f"{previous!r}, {relative!r}"
            )
        aliases[alias] = relative
        parsed.append(_ChecksumRecord(relative, digest, size))
    return tuple(sorted(parsed, key=lambda record: record.relative_path))


def _parse_raw_checksum_manifest(
    path: Path,
    *,
    dataset: Path,
    project_root: Path,
) -> tuple[_ChecksumRecord, ...]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"raw checksum manifest cannot be read: {path}: {exc}") from exc
    if not content:
        raise ValueError(f"raw checksum manifest is empty: {path}")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"raw checksum manifest must be UTF-8 JSON/CSV: {path}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
            rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
        except (csv.Error, UnicodeError) as exc:
            raise ValueError(
                f"raw checksum manifest is neither valid JSON nor CSV: {path}"
            ) from exc
        return _checksum_records_from_values(rows, dataset=dataset, project_root=project_root)
    if isinstance(payload, Mapping):
        if "raw_file_inventory" in payload:
            values = payload["raw_file_inventory"]
        elif "files" in payload:
            values = payload["files"]
        elif "inventory" in payload:
            values = payload["inventory"]
        elif all(isinstance(value, (str, Mapping)) for value in payload.values()):
            values = payload
        else:
            raise ValueError(
                "raw checksum JSON lacks an exact raw_file_inventory/files/inventory collection"
            )
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError("raw checksum JSON must be an object or list")
    return _checksum_records_from_values(values, dataset=dataset, project_root=project_root)


def _relative_if_within(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _capture_and_reconcile_dataset(
    dataset: Path,
    *,
    expected: Sequence[_ChecksumRecord],
    evidence_paths: Sequence[Path],
) -> _DatasetSnapshot:
    records = _dataset_tree_records(dataset)
    excluded = {
        relative
        for path in evidence_paths
        if (relative := _relative_if_within(path, dataset)) is not None
    }
    raw_records = tuple(
        record for record in records if str(record["relative_path"]) not in excluded
    )
    expected_by_path = {record.relative_path: record for record in expected}
    actual_by_path = {str(record["relative_path"]): record for record in raw_records}
    missing = sorted(set(expected_by_path).difference(actual_by_path))
    added = sorted(set(actual_by_path).difference(expected_by_path))
    changed: list[str] = []
    for relative in sorted(set(expected_by_path).intersection(actual_by_path)):
        wanted = expected_by_path[relative]
        observed = actual_by_path[relative]
        if wanted.sha256 != observed["sha256"] or (
            wanted.size_bytes is not None and wanted.size_bytes != observed["size_bytes"]
        ):
            changed.append(relative)
    if missing or added or changed:
        raise ValueError(
            "raw checksum manifest does not exactly reconcile to the dataset: "
            f"missing={missing}, added={added}, changed={changed}"
        )
    canonical = json.dumps(
        list(raw_records),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _DatasetSnapshot(
        dataset_sha256=_dataset_sha256(dataset, records),
        records=records,
        raw_inventory_records=raw_records,
        raw_inventory_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _read_json_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _normalise_timestamp(timestamp: datetime | None) -> tuple[str, str]:
    moment = timestamp or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("freeze timestamp must be timezone-aware")
    utc = moment.astimezone(UTC)
    rendered = utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    component = utc.strftime("%Y%m%dT%H%M%S.%fZ")
    return rendered, component


def _validate_preregistration(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"preregistration document does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if _INCOMPLETE_TOKEN.search(text):
        raise ValueError(
            "PRE_REGISTRATION.md is still draft/incomplete; resolve every draft, unresolved, "
            "unset, TBD, TODO, placeholder, and not-frozen marker before freezing"
        )
    lowered = text.casefold()
    missing = [section for section in _REQUIRED_PREREGISTRATION_SECTIONS if section not in lowered]
    if missing:
        raise ValueError(f"PRE_REGISTRATION.md lacks required completed sections: {missing}")
    state_match = re.search(r"\*\*state:\*\*\s*([^\n]+)", text, flags=re.IGNORECASE)
    if state_match is None or "ready_for_freeze" not in state_match.group(1).casefold():
        raise ValueError("PRE_REGISTRATION.md state must be READY_FOR_FREEZE before freezing")
    if len(text.strip()) < 500:
        raise ValueError(
            "PRE_REGISTRATION.md is implausibly short for a complete primary definition"
        )
    return text.replace("\r\n", "\n")


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"primary configuration field {location} must be a mapping")
    return value


def _validate_primary_config(config: Mapping[str, Any]) -> dict[str, Any]:
    # Import lazily so a clean ``histo_audit.workflows`` import can finish before
    # ``histo_audit.experiment.__init__`` loads primary_runner -> study_gates.
    from histo_audit.experiment.study_contracts import validate_frozen_primary_config

    return validate_frozen_primary_config(config)


def _validate_confirmatory_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a freeze candidate without creating a package import cycle."""

    from histo_audit.experiment.study_contracts import validate_frozen_confirmatory_config

    return validate_frozen_confirmatory_config(config)


def _validate_primary_confirmatory_cross_config(
    primary_config: Mapping[str, Any],
    confirmatory_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the neutral shared-config contract without creating an import cycle."""

    from histo_audit.experiment.study_contracts import (
        validate_primary_confirmatory_cross_config,
    )

    return validate_primary_confirmatory_cross_config(primary_config, confirmatory_config)


def _validate_completed_pilot(
    pilot_run_directory: Path,
    *,
    dataset_sha256: str,
    manifest_sha256: str,
    duplicate_audit_sha256: str,
    development_manifest_source: Path | None = None,
    gate_certificate_source: Path | None = None,
) -> dict[str, Any]:
    # Import lazily to preserve the workflows/experiment import boundary.  A seal
    # authenticates bytes, but it does not by itself establish the scientific M6
    # semantics required to authorize M7.
    from histo_audit.experiment.pilot_postseal import verify_pilot_post_seal

    integrity = verify_run_integrity(pilot_run_directory)
    if not integrity.valid:
        raise ValueError(f"pilot run failed integrity verification: {integrity.errors}")
    require_run_stage_eligible(pilot_run_directory, integrity=integrity)
    if development_manifest_source is None or gate_certificate_source is None:
        raise ValueError(
            "freeze requires explicit external pilot development-manifest and gate-certificate "
            "byte-identity authorities"
        )
    post_seal_verification = verify_pilot_post_seal(
        pilot_run_directory,
        development_manifest_source=development_manifest_source,
        gate_certificate_source=gate_certificate_source,
    )
    if (
        post_seal_verification.get("status") != "passed"
        or post_seal_verification.get("scientific_stage_eligible") is not True
    ):
        raise ValueError("pilot failed mandatory independent post-seal semantic verification")
    marker = _read_json_mapping(pilot_run_directory / ".immutable.json", role="pilot marker")
    status = _read_json_mapping(pilot_run_directory / "status.json", role="pilot status")
    config = load_config(pilot_run_directory / "resolved_config.yaml")
    checksums = _read_json_mapping(pilot_run_directory / "checksums.json", role="pilot checksums")
    if marker.get("status") != "completed" or status.get("status") != "completed":
        raise ValueError("pilot run must have completed terminal status")
    if config.get("experiment_name") != "pannuke_pilot":
        raise ValueError("freeze requires a completed pannuke_pilot run, not another experiment")
    for artifact in (
        "metrics.json",
        "report.md",
        "selected_groups_and_samples.json",
        "corruption_manifest.json",
        "oof_provenance.json",
        "oof_predictions.npz",
        "ranking.csv",
        "cleanlab_evidence.json",
        "cleanlab_evidence.csv",
        "cleanlab_evidence.npz",
        "neighbour_evidence.json",
        "neighbour_evidence.csv",
        "neighbour_evidence.npz",
        "audit_evidence_reconciliation.json",
        "development_manifest_view.parquet",
        "pre_pilot_gate_certificate.json",
        "final_reference_privacy_reconciliation.json",
    ):
        if not (pilot_run_directory / artifact).is_file():
            raise ValueError(f"completed pilot evidence lacks required artifact: {artifact}")
    metrics = _read_json_mapping(pilot_run_directory / "metrics.json", role="pilot metrics")
    if metrics.get("artifact_scope") != "real_pannuke_controlled_corruption_pilot":
        raise ValueError("pilot artifact scope is not eligible real PanNuke evidence")
    if metrics.get("completion_stage_if_sealed") != "PILOT_COMPLETE":
        raise ValueError("pilot metrics do not declare the sealed PILOT_COMPLETE gate")
    report_text = (pilot_run_directory / "report.md").read_text(encoding="utf-8").casefold()
    metrics_text = json.dumps(metrics, ensure_ascii=False, sort_keys=True).casefold()
    for location, text in (("report", report_text), ("metrics", metrics_text)):
        for phrase in ("potentially inconsistent annotation", "recommended for expert review"):
            if phrase not in text:
                raise ValueError(f"pilot {location} lacks mandatory terminology: {phrase}")
        if "not a diagnostic" not in text and "not diagnostic" not in text:
            raise ValueError(f"pilot {location} lacks its non-diagnostic limitation")
    selection = _read_json_mapping(
        pilot_run_directory / "selected_groups_and_samples.json",
        role="pilot selection evidence",
    )
    for field in (
        "final_reference_outcomes_used",
        "final_reference_representations_extracted",
        "final_reference_sample_ids_read",
        "final_reference_class_labels_read",
    ):
        if selection.get(field) is not False:
            raise ValueError(f"pilot selection evidence does not forbid {field}")
    saved_privacy = _read_json_mapping(
        pilot_run_directory / "final_reference_privacy_reconciliation.json",
        role="pilot final-reference privacy reconciliation",
    )
    if (
        saved_privacy.get("status") != "passed"
        or saved_privacy.get("policy") != "final_reference_identity_and_outcome_nonpublication_v1"
        or saved_privacy.get("final_reference_official_fold") != 3
    ):
        raise ValueError("pilot final-reference privacy evidence is not the required passed policy")
    dataset_record = _mapping(checksums.get("dataset"), location="pilot checksums.dataset")
    manifest_record = _mapping(checksums.get("manifest"), location="pilot checksums.manifest")
    if dataset_record.get("sha256") != dataset_sha256:
        raise ValueError("freeze dataset hash differs from the verified completed pilot")
    if manifest_record.get("sha256") != manifest_sha256:
        raise ValueError("freeze manifest hash differs from the verified completed pilot")
    if checksums.get("duplicate_audit_status") != (f"complete_sha256:{duplicate_audit_sha256}"):
        raise ValueError("freeze duplicate-audit hash differs from the verified completed pilot")
    if not integrity.registry_record_present:
        raise ValueError("pilot integrity evidence lacks its append-only registry record")
    return {
        "run_id": integrity.run_id,
        "run_directory": str(pilot_run_directory),
        "status": "completed",
        "artifact_root_sha256": integrity.expected_root_sha256,
        "integrity_registry_record_present": integrity.registry_record_present,
        "resolved_config_sha256": config_sha256(config),
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "duplicate_audit_sha256": duplicate_audit_sha256,
        "audit_evidence_reconciliation_sha256": sha256_file(
            pilot_run_directory / "audit_evidence_reconciliation.json"
        ),
        "final_reference_privacy_reconciliation_sha256": sha256_file(
            pilot_run_directory / "final_reference_privacy_reconciliation.json"
        ),
        "post_seal_verification": post_seal_verification,
        "post_seal_verification_semantic_sha256": hashlib.sha256(
            json.dumps(
                post_seal_verification,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _validate_duplicate_audit(path: Path) -> Mapping[str, Any]:
    payload = _read_json_mapping(path, role="duplicate audit")
    if payload.get("status") != "completed":
        raise ValueError("duplicate audit must have completed status")
    if payload.get("required_two_signal_near_duplicate_gate_complete") is not True:
        raise ValueError("duplicate audit lacks complete perceptual and embedding coverage")
    policy = _mapping(payload.get("policy"), location="duplicate audit policy")
    if policy.get("automatic_deletion") is not False or policy.get("cross_fold_only") is not True:
        raise ValueError("duplicate audit policy must be cross-fold review-only without deletion")
    coverage = _mapping(payload.get("coverage"), location="duplicate audit coverage")
    counts = (
        coverage.get("total_source_patches"),
        coverage.get("patches_with_full_hash_provenance"),
        coverage.get("perceptual_comparison_patch_count"),
        coverage.get("embedding_patch_count"),
    )
    if any(not isinstance(value, int) or value <= 0 for value in counts) or len(set(counts)) != 1:
        raise ValueError("duplicate audit does not cover every source patch with both signals")
    return payload


def validate_pathology_encoder_audit(path: Path) -> Mapping[str, Any]:
    """Apply the exact freeze-time semantic gate to a pathology audit."""

    payload = _read_json_mapping(path, role="pathology encoder availability audit")
    if payload.get("status") not in {"completed", "blocked"}:
        raise ValueError("pathology encoder audit must be completed or blocked with evidence")
    if not isinstance(payload.get("selection_rule"), str) or not payload.get("selection_rule"):
        raise ValueError("pathology encoder audit lacks its frozen priority selection rule")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("pathology encoder audit lacks candidate records")
    selected = payload.get("selected_encoder")
    if payload.get("status") == "blocked" and selected is not None:
        raise ValueError("blocked pathology encoder audit cannot identify a selected encoder")
    return payload


def _validate_bound_evidence_hashes(
    primary: Mapping[str, Any],
    confirmatory: Mapping[str, Any],
    *,
    project_root: Path,
    pathology_encoder_audit_sha256: str,
) -> Path:
    from histo_audit.experiment.pannuke_primary_inputs import (
        _load_independence_evidence_matrix,
    )

    representations = primary.get("representations")
    assert isinstance(representations, list)  # enforced by the strict contract
    pathology_hashes = {
        str(item["availability_audit_sha256"])
        for item in representations
        if isinstance(item, Mapping) and item.get("family") == "pathology"
    }
    scenarios = confirmatory.get("scenarios")
    assert isinstance(scenarios, list)  # enforced by the strict contract
    pathology_hashes.update(
        str(item["availability_audit_sha256"])
        for item in scenarios
        if isinstance(item, Mapping) and item.get("family") == "pathology_frozen"
    )
    if pathology_hashes != {pathology_encoder_audit_sha256}:
        raise ValueError(
            "primary/confirmatory pathology availability hashes must match the supplied audit"
        )
    mechanisms = _mapping(primary.get("corruption"), location="primary corruption").get(
        "mechanisms"
    )
    instance = _mapping(
        _mapping(mechanisms, location="primary mechanisms").get("instance_dependent_corruption"),
        location="instance-dependent corruption",
    )
    independence_path = Path(str(instance["independence_matrix_path"]))
    if not independence_path.is_absolute():
        independence_path = project_root / independence_path
    if not independence_path.is_file():
        raise FileNotFoundError(
            f"representation-independence evidence does not exist: {independence_path}"
        )
    if sha256_file(independence_path) != instance["independence_matrix_sha256"]:
        raise ValueError("representation-independence evidence hash differs from primary config")
    evidence_by_representation = _load_independence_evidence_matrix(independence_path)
    representation_by_id = {
        str(item["id"]): item for item in representations if isinstance(item, Mapping)
    }
    available_ids = {
        identifier
        for identifier, item in representation_by_id.items()
        if isinstance(item.get("cache_provenance"), Mapping)
        and item["cache_provenance"].get("status") == "available"
    }
    if set(evidence_by_representation) != available_ids:
        raise ValueError(
            "representation-independence matrix must contain exactly every available "
            f"frozen representation: expected={sorted(available_ids)}, "
            f"actual={sorted(evidence_by_representation)}"
        )
    generators = {evidence.generator for evidence in evidence_by_representation.values()}
    if len(generators) != 1:
        raise ValueError("representation-independence entries do not bind one generator space")
    generator = next(iter(generators))
    if (
        generator.representation_name != instance["generator_representation"]
        or generator.family != "morphology"
    ):
        raise ValueError("representation-independence matrix binds the wrong generator space")
    for identifier, evidence in evidence_by_representation.items():
        representation = representation_by_id[identifier]
        if (
            evidence.auditor.representation_name != identifier
            or evidence.auditor.family != representation["family"]
        ):
            raise ValueError(
                f"representation-independence auditor identity differs for {identifier}"
            )
        expected_status = (
            "verified_independent"
            if evidence.matrix_decision == "verified_independent"
            else "circularity_risk"
        )
        binding = _mapping(
            representation.get("generator_independence"),
            location=f"representation {identifier} generator independence",
        )
        if binding.get("status") != expected_status or binding.get(
            "independence_matrix_sha256"
        ) != sha256_file(independence_path):
            raise ValueError(f"frozen generator-independence status/hash differs for {identifier}")
    return independence_path.resolve()


def _assert_semantically_equal(
    actual: Any,
    expected: Any,
    *,
    location: str,
) -> None:
    """Compare derivation payload values, tolerating JSON/YAML float round-off only."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"pilot-derived parameters differ at {location}")
        for key, expected_value in expected.items():
            _assert_semantically_equal(
                actual[key],
                expected_value,
                location=f"{location}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if (
            not isinstance(actual, Sequence)
            or isinstance(actual, (str, bytes, bytearray))
            or len(actual) != len(expected)
        ):
            raise ValueError(f"pilot-derived parameters differ at {location}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
            _assert_semantically_equal(
                actual_value,
                expected_value,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(expected, bool):
        matches = actual is expected
    elif isinstance(expected, (int, float)):
        matches = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"pilot-derived parameters differ at {location}")


def validate_pilot_derived_parameters(
    primary: Mapping[str, Any],
    *,
    project_root: Path,
    pilot_evidence: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Apply the exact freeze-time gate to the pilot-derived parameter record."""

    binding = _mapping(
        primary.get("pilot_derived_parameters"),
        location="pilot_derived_parameters",
    )
    configured_path = str(binding.get("path"))
    if configured_path != "reports/pilot_derived_primary_parameters.json":
        raise ValueError("pilot-derived parameters must use their canonical reports path")
    path = (project_root / configured_path).resolve()
    if not _path_within(path, project_root):
        raise ValueError("pilot-derived parameters resolve outside the project root")
    if not path.is_file():
        raise FileNotFoundError(f"pilot-derived parameter evidence does not exist: {path}")
    lexical_value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(lexical_value.st_mode) or _is_reparse_or_link(path, lexical_value):
        raise ValueError(
            f"pilot-derived parameter evidence must be a regular non-link file: {path}"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    stable_record = _stable_file_record(path, relative_path=configured_path)
    if stable_record["sha256"] != digest:
        raise RuntimeError("pilot-derived parameter evidence changed while being validated")
    if binding.get("sha256") != digest:
        raise ValueError("pilot-derived parameter evidence hash differs from primary config")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"pilot-derived parameter evidence is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("pilot-derived parameter evidence must be a JSON object")

    schema_version = binding.get("schema_version")
    producer_id = binding.get("producer_id")
    if payload.get("schema_version") != schema_version or schema_version != 1:
        raise ValueError("pilot-derived parameter schema differs from primary config")
    if (
        payload.get("producer_id") != producer_id
        or producer_id != "pilot_derived_primary_parameters_v1"
    ):
        raise ValueError("pilot-derived parameter producer differs from primary config")
    producer_source_sha256 = payload.get("producer_source_sha256")
    producer_source_path = (
        Path(__file__).resolve().parents[1] / "experiment" / "pilot_derived_parameters.py"
    )
    if (
        not isinstance(producer_source_sha256, str)
        or _SHA256_PATTERN.fullmatch(producer_source_sha256) is None
        or producer_source_sha256 != sha256_file(producer_source_path)
    ):
        raise ValueError("pilot-derived parameter producer source is not the current frozen source")

    source_pilot = _mapping(
        payload.get("source_pilot"),
        location="pilot-derived parameters source_pilot",
    )
    canonical_run_id = pilot_evidence.get("run_id")
    canonical_root = pilot_evidence.get("artifact_root_sha256")
    if (
        binding.get("source_pilot_run_id") != canonical_run_id
        or source_pilot.get("run_id") != canonical_run_id
    ):
        raise ValueError("pilot-derived parameters do not bind the canonical M6 pilot run_id")
    if (
        binding.get("source_pilot_artifact_root_sha256") != canonical_root
        or source_pilot.get("artifact_root_sha256") != canonical_root
        or not isinstance(canonical_root, str)
        or _SHA256_PATTERN.fullmatch(canonical_root) is None
    ):
        raise ValueError("pilot-derived parameters do not bind the canonical M6 artifact root")

    data = _mapping(primary.get("data"), location="primary data")
    if source_pilot.get("development_official_folds") != data.get("development_official_folds"):
        raise ValueError("pilot-derived parameters use the wrong development official folds")
    if source_pilot.get("final_reference_policy") != (
        "no final-fold sample identifier, label, representation, or outcome read"
    ):
        raise ValueError("pilot-derived parameters lack the frozen final-reference privacy policy")
    if payload.get("class_order") != data.get("class_order"):
        raise ValueError("pilot-derived parameter class order differs from primary config")

    mechanisms = _mapping(
        _mapping(primary.get("corruption"), location="primary corruption").get("mechanisms"),
        location="primary corruption mechanisms",
    )
    derived_confusion = _mapping(
        payload.get("confusion_targeted_corruption"),
        location="pilot-derived confusion parameters",
    )
    expected_confusion = _mapping(
        mechanisms.get("confusion_targeted_corruption"),
        location="primary confusion parameters",
    )
    _assert_semantically_equal(
        derived_confusion.get("transition_matrix"),
        expected_confusion.get("transition_matrix"),
        location="confusion_targeted_corruption.transition_matrix",
    )
    derived_group = _mapping(
        payload.get("group_conditional_corruption"),
        location="pilot-derived group-conditional parameters",
    )
    expected_group = _mapping(
        mechanisms.get("group_conditional_corruption"),
        location="primary group-conditional parameters",
    )
    for field in ("grouping_field", "weights_by_value", "default_weight"):
        _assert_semantically_equal(
            derived_group.get(field),
            expected_group.get(field),
            location=f"group_conditional_corruption.{field}",
        )

    return path, {
        "schema_version": schema_version,
        "producer_id": producer_id,
        "producer_source_sha256": producer_source_sha256,
        "canonical_path": configured_path,
        "sha256": digest,
        "source_pilot_run_id": canonical_run_id,
        "source_pilot_artifact_root_sha256": canonical_root,
        "strict_semantic_validation_passed": True,
    }


# Backward-compatible private aliases for callers/tests predating the shared
# freeze-gate API.  New code should import the public names above.
_validate_pathology_encoder_audit = validate_pathology_encoder_audit
_validate_pilot_derived_parameters = validate_pilot_derived_parameters


def _canonical_artifact_root(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        list(records), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _snapshot_records(directory: Path) -> list[dict[str, Any]]:
    excluded = {"sha256_manifest.json", ".immutable.json"}
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative not in excluded:
            records.append(
                {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return records


@dataclass(frozen=True, slots=True)
class _ParentAnchor:
    path: Path
    identity: tuple[int, int]

    def assert_current(self) -> None:
        try:
            value = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"freeze publication parent disappeared: {self.path}") from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or _is_reparse_or_link(self.path, value)
            or (value.st_dev, value.st_ino) != self.identity
            or self.path.resolve() != self.path
        ):
            raise RuntimeError(f"freeze publication parent identity changed: {self.path}")


def _lexical_output_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _path_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolved_output_path(path: Path) -> Path:
    return (path.parent.resolve() / path.name).resolve(strict=False)


def _assert_safe_freeze_destinations(
    *,
    project_root: Path,
    dataset: Path,
    pilot_run_directory: Path,
    outputs: Sequence[Path],
) -> None:
    raw_protected = dataset.resolve()
    pilot_protected = pilot_run_directory.resolve()
    runs_root = (project_root / "artifacts" / "runs").resolve()
    source_roots = ((project_root / "src").resolve(), (project_root / "configs").resolve())
    for output in outputs:
        current = _resolved_output_path(output)
        inside_raw = (
            current == raw_protected
            if raw_protected.is_file()
            else _path_within(current, raw_protected)
        )
        if inside_raw:
            raise ValueError(f"preregistration freeze output cannot be inside raw data: {output}")
        if _path_within(current, pilot_protected) or _path_within(current, runs_root):
            raise ValueError(
                f"preregistration freeze output cannot be inside a sealed experiment run: {output}"
            )
        if output.name not in {"primary_frozen.yaml", "confirmatory_frozen.yaml"} and any(
            _path_within(current, source_root) for source_root in source_roots
        ):
            raise ValueError(
                f"timestamped preregistration evidence cannot be published in src/configs: {output}"
            )


def _anchor_parent(path: Path) -> _ParentAnchor:
    lexical_value = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(lexical_value.st_mode) or _is_reparse_or_link(path, lexical_value):
        raise ValueError(f"freeze publication parent must not be a link/reparse point: {path}")
    parent = path.resolve()
    value = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(value.st_mode) or _is_reparse_or_link(parent, value):
        raise ValueError(f"freeze publication parent must be a real directory: {parent}")
    return _ParentAnchor(parent, (value.st_dev, value.st_ino))


def _source_hashes(paths: Sequence[Path]) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for path in paths:
        lexical_value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(lexical_value.st_mode) or _is_reparse_or_link(path, lexical_value):
            raise ValueError(f"freeze source evidence must be a regular non-link file: {path}")
        resolved = path.resolve()
        record = _stable_file_record(resolved, relative_path=resolved.name)
        result[resolved] = str(record["sha256"])
    return result


def _assert_source_hashes(expected: Mapping[Path, str]) -> None:
    for path, digest in expected.items():
        if not path.is_file():
            raise RuntimeError(f"freeze source evidence disappeared before publication: {path}")
        observed = _stable_file_record(path, relative_path=path.name)
        if observed["sha256"] != digest:
            raise RuntimeError(f"freeze source evidence changed before publication: {path}")


def _assert_source_tree_unchanged(
    project_root: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = capture_source_tree(project_root)
    actual_records = actual.get("artifacts")
    expected_records = expected.get("artifacts")
    if not isinstance(actual_records, list) or not isinstance(expected_records, list):
        raise RuntimeError("source-tree capture returned malformed artifact records")
    if actual_records != expected_records:
        raise RuntimeError("generating source tree changed during preregistration freeze")


def _assert_governance_tree_unchanged(
    project_root: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = capture_governance_tree(project_root)
    if actual != expected:
        raise RuntimeError("project governance tree changed during preregistration freeze")


def _stable_git_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every Git-state field except its observation timestamp."""

    available = state.get("available")
    if not isinstance(available, bool):
        raise RuntimeError("Git state capture available must be a boolean")
    if available:
        status = state.get("status_porcelain")
        dirty = state.get("dirty")
        branch = state.get("branch")
        commit = state.get("commit")
        if not isinstance(status, str):
            raise RuntimeError("available Git state capture status_porcelain must be a string")
        if not isinstance(dirty, bool):
            raise RuntimeError("available Git state capture dirty must be a boolean")
        if dirty != bool(status):
            raise RuntimeError("available Git state capture dirty differs from status_porcelain")
        if not isinstance(branch, str):
            raise RuntimeError("available Git state capture branch must be a string")
        if commit is not None and (not isinstance(commit, str) or not commit):
            raise RuntimeError(
                "available Git state capture commit must be null or a non-empty string"
            )
    else:
        reason = state.get("reason")
        if not isinstance(reason, str) or not reason:
            raise RuntimeError("unavailable Git state capture reason must be a non-empty string")
    stable = dict(state)
    stable.pop("captured_at_utc", None)
    return stable


def _assert_parent_anchors(anchors: Sequence[_ParentAnchor]) -> None:
    for anchor in anchors:
        anchor.assert_current()


def _publish_freeze_transaction(
    *,
    project_root: Path,
    dataset: Path,
    pilot_path: Path,
    destination: Path,
    frozen_primary_config: Path,
    frozen_confirmatory_config: Path,
    snapshot_staging: Path,
    canonical_primary_staging: Path,
    canonical_confirmatory_staging: Path,
    parent_anchors: Sequence[_ParentAnchor],
    verify_sources: Any,
) -> list[PublishedPath]:
    marker_source = snapshot_staging / ".immutable.json"
    snapshot_sources = tuple(
        sorted(
            (path for path in snapshot_staging.iterdir() if path != marker_source),
            key=lambda path: path.name,
        )
    )
    if not marker_source.is_file() or any(not path.is_file() for path in snapshot_sources):
        raise RuntimeError("freeze staging directory is incomplete or non-flat")
    lock_paths = (destination, frozen_primary_config, frozen_confirmatory_config)
    publications: list[PublishedPath] = []
    with ExclusiveBundlePublicationLock(lock_paths, role="preregistration freeze bundle") as lock:
        lock.assert_owned()
        _assert_parent_anchors(parent_anchors)
        _assert_safe_freeze_destinations(
            project_root=project_root,
            dataset=dataset,
            pilot_run_directory=pilot_path,
            outputs=lock_paths,
        )
        occupied = [str(path) for path in lock_paths if os.path.lexists(path)]
        if occupied:
            raise FileExistsError(
                f"preregistration freeze is one-shot and refuses occupied destinations: {occupied}"
            )
        verify_sources(require_git_state_unchanged=True)
        try:
            publications.append(create_directory_no_overwrite(destination))
            for source in snapshot_sources:
                _assert_parent_anchors(parent_anchors)
                if not publications[0].still_owned():
                    raise RuntimeError("freeze destination directory ownership changed")
                publications.append(publish_file_no_overwrite(source, destination / source.name))
            publications.append(
                publish_file_no_overwrite(canonical_primary_staging, frozen_primary_config)
            )
            publications.append(
                publish_file_no_overwrite(
                    canonical_confirmatory_staging,
                    frozen_confirmatory_config,
                )
            )
            lock.assert_owned()
            _assert_parent_anchors(parent_anchors)
            verify_sources(require_git_state_unchanged=False)
            if not all(publication.still_owned() for publication in publications):
                raise RuntimeError("freeze publication ownership changed before success marker")
            publications.append(
                publish_success_marker_no_overwrite(
                    marker_source,
                    destination / marker_source.name,
                    owned_parent=publications[0],
                )
            )
            lock.assert_owned()
            _assert_parent_anchors(parent_anchors)
            if not all(publication.still_owned() for publication in publications):
                raise RuntimeError("freeze publication failed final identity/hash readback")
            verify_sources(require_git_state_unchanged=False)
            verification = verify_preregistration_freeze(
                destination,
                frozen_primary_config_path=frozen_primary_config,
                frozen_confirmatory_config_path=frozen_confirmatory_config,
            )
            if not verification.valid:
                raise RuntimeError(
                    "published preregistration freeze failed independent verification: "
                    f"errors={verification.errors}, missing={verification.missing_paths}, "
                    f"added={verification.added_paths}, changed={verification.changed_paths}"
                )
            lock.assert_owned()
            _assert_parent_anchors(parent_anchors)
        except BaseException as publication_error:
            try:
                rollback_owned_publications(publications)
            except RuntimeError as rollback_error:
                raise RuntimeError(
                    "preregistration freeze failed and ownership-safe rollback was incomplete"
                ) from rollback_error
            raise publication_error
    return publications


def _amendment_policy(freeze_timestamp: str) -> str:
    return f"""# Preregistration amendment policy

Frozen at `{freeze_timestamp}`. The files in this directory and
`configs/primary_frozen.yaml` and `configs/confirmatory_frozen.yaml` must never be
edited or replaced.

Any change requires a new, timestamped amendment artifact. It must record the date,
reason, affected hypotheses and analyses, whether outcomes had been inspected, and the
before/after SHA-256 values. An amendment cannot retroactively redefine a primary result;
affected analyses must be labelled amended or exploratory.
"""


def freeze_preregistration(
    *,
    project_root: str | Path,
    pilot_run_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    raw_checksum_manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    pilot_development_manifest_path: str | Path | None = None,
    pilot_gate_certificate_path: str | Path | None = None,
    preregistration_path: str | Path | None = None,
    primary_config_path: str | Path | None = None,
    confirmatory_config_path: str | Path | None = None,
    freeze_root: str | Path | None = None,
    timestamp: datetime | None = None,
) -> PreregistrationFreezeResult:
    """Cryptographically freeze a complete primary plan after a verified pilot.

    This operation is deliberately one-shot. It refuses an existing
    either canonical frozen configuration or the timestamp directory and never
    overwrites any of them.  The exact confirmatory matrix is frozen at the same time,
    before primary outcomes can be inspected.
    """

    root = Path(project_root).resolve()
    prereg_path = _lexical_output_path(preregistration_path or root / "PRE_REGISTRATION.md")
    primary_path = _lexical_output_path(primary_config_path or root / "configs" / "primary.yaml")
    confirmatory_path = _lexical_output_path(
        confirmatory_config_path or root / "configs" / "confirmatory.yaml"
    )
    pilot_path = _lexical_output_path(pilot_run_directory)
    dataset = _lexical_output_path(dataset_path)
    manifest = _lexical_output_path(manifest_path)
    raw_checksums = _lexical_output_path(raw_checksum_manifest_path)
    duplicate_audit = _lexical_output_path(duplicate_audit_path)
    pathology_encoder_audit = _lexical_output_path(pathology_encoder_audit_path)
    pilot_development_manifest = (
        _lexical_output_path(pilot_development_manifest_path)
        if pilot_development_manifest_path is not None
        else None
    )
    pilot_gate_certificate = (
        _lexical_output_path(pilot_gate_certificate_path)
        if pilot_gate_certificate_path is not None
        else None
    )
    frozen_primary_config = _lexical_output_path(root / "configs" / "primary_frozen.yaml")
    frozen_confirmatory_config = _lexical_output_path(root / "configs" / "confirmatory_frozen.yaml")
    freeze_parent = _lexical_output_path(freeze_root or root / "artifacts" / "preregistrations")
    freeze_timestamp, timestamp_component = _normalise_timestamp(timestamp)
    destination = _lexical_output_path(freeze_parent / timestamp_component)

    _assert_safe_freeze_destinations(
        project_root=root,
        dataset=dataset,
        pilot_run_directory=pilot_path,
        outputs=(destination, frozen_primary_config, frozen_confirmatory_config),
    )
    if os.path.lexists(frozen_primary_config):
        raise FileExistsError(
            f"frozen primary configuration already exists: {frozen_primary_config}"
        )
    if os.path.lexists(frozen_confirmatory_config):
        raise FileExistsError(
            f"frozen confirmatory configuration already exists: {frozen_confirmatory_config}"
        )
    if os.path.lexists(destination):
        raise FileExistsError(f"preregistration freeze destination already exists: {destination}")
    if not dataset.exists():
        raise FileNotFoundError(f"dataset path does not exist: {dataset}")
    if not manifest.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest}")
    if not raw_checksums.is_file() or raw_checksums.stat().st_size == 0:
        raise FileNotFoundError(
            f"raw dataset checksum manifest is missing or empty: {raw_checksums}"
        )
    if not duplicate_audit.is_file():
        raise FileNotFoundError(f"duplicate audit does not exist: {duplicate_audit}")
    if not pathology_encoder_audit.is_file():
        raise FileNotFoundError(
            f"pathology encoder availability audit does not exist: {pathology_encoder_audit}"
        )
    _validate_duplicate_audit(duplicate_audit)
    _validate_pathology_encoder_audit(pathology_encoder_audit)
    expected_raw_checksums = _parse_raw_checksum_manifest(
        raw_checksums,
        dataset=dataset,
        project_root=root,
    )
    dataset_snapshot = _capture_and_reconcile_dataset(
        dataset,
        expected=expected_raw_checksums,
        evidence_paths=(raw_checksums,),
    )
    dataset_sha = dataset_snapshot.dataset_sha256
    manifest_sha = sha256_file(manifest)
    raw_checksum_sha = sha256_file(raw_checksums)
    duplicate_audit_sha = sha256_file(duplicate_audit)
    pathology_encoder_audit_sha = sha256_file(pathology_encoder_audit)
    pilot_evidence = _validate_completed_pilot(
        pilot_path,
        dataset_sha256=dataset_sha,
        manifest_sha256=manifest_sha,
        duplicate_audit_sha256=duplicate_audit_sha,
        development_manifest_source=pilot_development_manifest,
        gate_certificate_source=pilot_gate_certificate,
    )
    if pilot_development_manifest is None or pilot_gate_certificate is None:  # pragma: no cover
        raise RuntimeError("pilot authority paths vanished after mandatory validation")
    preregistration_text = _validate_preregistration(prereg_path)
    if not primary_path.is_file():
        raise FileNotFoundError(f"primary configuration does not exist: {primary_path}")
    primary = _validate_primary_config(load_config(primary_path))
    if not confirmatory_path.is_file():
        raise FileNotFoundError(f"confirmatory configuration does not exist: {confirmatory_path}")
    confirmatory = _validate_confirmatory_config(load_config(confirmatory_path))
    primary, confirmatory = _validate_primary_confirmatory_cross_config(primary, confirmatory)
    independence_path = _validate_bound_evidence_hashes(
        primary,
        confirmatory,
        project_root=root,
        pathology_encoder_audit_sha256=pathology_encoder_audit_sha,
    )
    pilot_derived_path, pilot_derived_evidence = _validate_pilot_derived_parameters(
        primary,
        project_root=root,
        pilot_evidence=pilot_evidence,
    )
    primary_bytes = yaml.safe_dump(
        primary, allow_unicode=True, default_flow_style=False, sort_keys=True
    ).encode("utf-8")
    confirmatory_bytes = yaml.safe_dump(
        confirmatory, allow_unicode=True, default_flow_style=False, sort_keys=True
    ).encode("utf-8")

    source_hashes = _source_hashes(
        (
            prereg_path,
            primary_path,
            confirmatory_path,
            manifest,
            raw_checksums,
            duplicate_audit,
            pathology_encoder_audit,
            independence_path,
            pilot_derived_path,
            pilot_development_manifest,
            pilot_gate_certificate,
        )
    )
    git_state = capture_git_state(root)
    stable_git_state = _stable_git_state(git_state)
    source_tree = capture_source_tree(root)
    governance_tree = capture_governance_tree(root)
    freeze_parent.mkdir(parents=True, exist_ok=True)
    frozen_primary_config.parent.mkdir(parents=True, exist_ok=True)
    parent_anchors = (
        _anchor_parent(freeze_parent),
        _anchor_parent(frozen_primary_config.parent),
    )
    staging = Path(tempfile.mkdtemp(prefix=f"histo-audit-freeze-{timestamp_component}."))
    snapshot_staging = staging / "snapshot"
    snapshot_staging.mkdir()
    canonical_primary_staging = staging / "canonical-primary.tmp"
    canonical_confirmatory_staging = staging / "canonical-confirmatory.tmp"
    try:
        atomic_write_text(
            snapshot_staging / "PRE_REGISTRATION_FROZEN.md",
            preregistration_text,
        )
        atomic_write_bytes(snapshot_staging / "primary_frozen.yaml", primary_bytes)
        atomic_write_bytes(snapshot_staging / "confirmatory_frozen.yaml", confirmatory_bytes)
        atomic_write_bytes(canonical_primary_staging, primary_bytes)
        atomic_write_bytes(canonical_confirmatory_staging, confirmatory_bytes)
        raw_snapshot_name = f"raw_dataset_checksums{raw_checksums.suffix or '.txt'}"
        duplicate_snapshot_name = f"duplicate_audit{duplicate_audit.suffix or '.json'}"
        pathology_snapshot_name = (
            f"pathology_encoder_availability{pathology_encoder_audit.suffix or '.json'}"
        )
        independence_snapshot_name = (
            f"representation_independence{independence_path.suffix or '.json'}"
        )
        pilot_derived_snapshot_name = "pilot_derived_primary_parameters.json"
        atomic_write_bytes(snapshot_staging / raw_snapshot_name, raw_checksums.read_bytes())
        atomic_write_bytes(
            snapshot_staging / duplicate_snapshot_name,
            duplicate_audit.read_bytes(),
        )
        atomic_write_bytes(
            snapshot_staging / pathology_snapshot_name,
            pathology_encoder_audit.read_bytes(),
        )
        atomic_write_bytes(
            snapshot_staging / independence_snapshot_name,
            independence_path.read_bytes(),
        )
        atomic_write_bytes(
            snapshot_staging / pilot_derived_snapshot_name,
            pilot_derived_path.read_bytes(),
        )
        atomic_write_json(snapshot_staging / "git_state.json", git_state)
        atomic_write_json(snapshot_staging / "source_tree_manifest.json", source_tree)
        atomic_write_json(snapshot_staging / "governance_tree_manifest.json", governance_tree)
        atomic_write_json(snapshot_staging / "pilot_integrity_evidence.json", pilot_evidence)
        post_seal_snapshot_path = atomic_write_json(
            snapshot_staging / "pilot_post_seal_verification.json",
            pilot_evidence["post_seal_verification"],
        )
        post_seal_snapshot_sha256 = sha256_file(post_seal_snapshot_path)
        atomic_write_text(
            snapshot_staging / "AMENDMENT_POLICY.md",
            _amendment_policy(freeze_timestamp),
        )
        atomic_write_json(
            snapshot_staging / "freeze_evidence.json",
            {
                "schema_version": BASE_FREEZE_EVIDENCE_SCHEMA_VERSION,
                "completion_stage_enabled": "PRE_REGISTRATION_FROZEN",
                "freeze_timestamp_utc": freeze_timestamp,
                "pilot": pilot_evidence,
                "pilot_external_byte_identity_authorities": {
                    "development_manifest_path": str(pilot_development_manifest),
                    "development_manifest_sha256": source_hashes[pilot_development_manifest],
                    "gate_certificate_path": str(pilot_gate_certificate),
                    "gate_certificate_sha256": source_hashes[pilot_gate_certificate],
                    "post_seal_verification_snapshot": "pilot_post_seal_verification.json",
                    "post_seal_verification_snapshot_sha256": post_seal_snapshot_sha256,
                    "post_seal_verification_semantic_sha256": pilot_evidence[
                        "post_seal_verification_semantic_sha256"
                    ],
                },
                "preregistration": {
                    "source_path": str(prereg_path),
                    "sha256": sha256_file(prereg_path),
                },
                "primary_config": {
                    "source_path": str(primary_path),
                    "semantic_sha256": config_sha256(primary),
                    "frozen_path": str(frozen_primary_config),
                    "frozen_file_sha256": hashlib.sha256(primary_bytes).hexdigest(),
                },
                "confirmatory_config": {
                    "source_path": str(confirmatory_path),
                    "semantic_sha256": config_sha256(confirmatory),
                    "frozen_path": str(frozen_confirmatory_config),
                    "frozen_file_sha256": hashlib.sha256(confirmatory_bytes).hexdigest(),
                    "frozen_before_primary_outcomes": True,
                },
                "dataset": {"path": str(dataset), "sha256": dataset_sha},
                "manifest": {"path": str(manifest), "sha256": manifest_sha},
                "raw_dataset_checksum_manifest": {
                    "path": str(raw_checksums),
                    "sha256": raw_checksum_sha,
                    "snapshot": raw_snapshot_name,
                    "parsed_record_count": len(expected_raw_checksums),
                    "exact_dataset_reconciliation": True,
                    "reconciled_inventory_record_count": len(
                        dataset_snapshot.raw_inventory_records
                    ),
                    "reconciled_inventory_sha256": dataset_snapshot.raw_inventory_sha256,
                },
                "duplicate_audit": {
                    "path": str(duplicate_audit),
                    "sha256": duplicate_audit_sha,
                    "snapshot": duplicate_snapshot_name,
                    "required_two_signal_gate_complete": True,
                },
                "pathology_encoder_availability_audit": {
                    "path": str(pathology_encoder_audit),
                    "sha256": pathology_encoder_audit_sha,
                    "snapshot": pathology_snapshot_name,
                },
                "representation_independence": {
                    "path": str(independence_path),
                    "sha256": source_hashes[independence_path],
                    "snapshot": independence_snapshot_name,
                    "schema_version": 2,
                    "scope": "every available frozen primary representation",
                    "strict_semantic_validation_passed": True,
                },
                "pilot_derived_parameters": {
                    **pilot_derived_evidence,
                    "resolved_path": str(pilot_derived_path),
                    "snapshot": pilot_derived_snapshot_name,
                    "snapshot_sha256": sha256_file(snapshot_staging / pilot_derived_snapshot_name),
                },
                "source_tree_root_sha256": source_tree["root_sha256"],
                "governance_tree_root_sha256": governance_tree["root_sha256"],
                "execution_source_tree": {
                    "schema_version": source_tree["schema_version"],
                    "scope_kind": source_tree["scope_kind"],
                    "scope": source_tree["scope"],
                    "artifact_count": source_tree["artifact_count"],
                    "root_sha256": source_tree["root_sha256"],
                    "snapshot": "source_tree_manifest.json",
                },
                "governance_tree": {
                    "schema_version": governance_tree["schema_version"],
                    "scope_kind": governance_tree["scope_kind"],
                    "scope": governance_tree["scope"],
                    "artifact_count": governance_tree["artifact_count"],
                    "root_sha256": governance_tree["root_sha256"],
                    "snapshot": "governance_tree_manifest.json",
                },
                "git": git_state,
                "amendment_policy": "AMENDMENT_POLICY.md",
                "overwrite_policy": "never overwrite; amendments require new timestamped artifacts",
            },
        )
        records = _snapshot_records(snapshot_staging)
        artifact_root = _canonical_artifact_root(records)
        sha_manifest_path = atomic_write_json(
            snapshot_staging / "sha256_manifest.json",
            {
                "schema_version": 1,
                "freeze_timestamp_utc": freeze_timestamp,
                "artifact_count": len(records),
                "artifact_root_sha256": artifact_root,
                "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
                "excluded_paths": [".immutable.json", "sha256_manifest.json"],
                "artifacts": records,
            },
        )
        manifest_digest = sha256_file(sha_manifest_path)
        atomic_write_json(
            snapshot_staging / ".immutable.json",
            {
                "schema_version": 1,
                "status": "frozen",
                "freeze_timestamp_utc": freeze_timestamp,
                "artifact_root_sha256": artifact_root,
                "sha256_manifest_sha256": manifest_digest,
                "amendment_only": True,
            },
        )

        def verify_sources(*, require_git_state_unchanged: bool) -> None:
            _assert_safe_freeze_destinations(
                project_root=root,
                dataset=dataset,
                pilot_run_directory=pilot_path,
                outputs=(destination, frozen_primary_config, frozen_confirmatory_config),
            )
            _assert_source_hashes(source_hashes)
            observed_checksum_records = _parse_raw_checksum_manifest(
                raw_checksums,
                dataset=dataset,
                project_root=root,
            )
            if observed_checksum_records != expected_raw_checksums:
                raise RuntimeError("parsed raw checksum inventory changed during freeze")
            observed_dataset = _capture_and_reconcile_dataset(
                dataset,
                expected=expected_raw_checksums,
                evidence_paths=(raw_checksums,),
            )
            if observed_dataset != dataset_snapshot:
                raise RuntimeError("exact dataset inventory changed during preregistration freeze")
            observed_pilot = _validate_completed_pilot(
                pilot_path,
                dataset_sha256=dataset_sha,
                manifest_sha256=manifest_sha,
                duplicate_audit_sha256=duplicate_audit_sha,
                development_manifest_source=pilot_development_manifest,
                gate_certificate_source=pilot_gate_certificate,
            )
            if observed_pilot != pilot_evidence:
                raise RuntimeError("pilot integrity/eligibility evidence changed during freeze")
            observed_derived_path, observed_derived_evidence = _validate_pilot_derived_parameters(
                primary,
                project_root=root,
                pilot_evidence=observed_pilot,
            )
            if (
                observed_derived_path != pilot_derived_path
                or observed_derived_evidence != pilot_derived_evidence
            ):
                raise RuntimeError("pilot-derived parameter evidence changed during freeze")
            _assert_source_tree_unchanged(root, source_tree)
            _assert_governance_tree_unchanged(root, governance_tree)
            if require_git_state_unchanged:
                observed_git_state = _stable_git_state(capture_git_state(root))
                if observed_git_state != stable_git_state:
                    raise RuntimeError("Git state changed before preregistration publication")

        _publish_freeze_transaction(
            project_root=root,
            dataset=dataset,
            pilot_path=pilot_path,
            destination=destination,
            frozen_primary_config=frozen_primary_config,
            frozen_confirmatory_config=frozen_confirmatory_config,
            snapshot_staging=snapshot_staging,
            canonical_primary_staging=canonical_primary_staging,
            canonical_confirmatory_staging=canonical_confirmatory_staging,
            parent_anchors=parent_anchors,
            verify_sources=verify_sources,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return PreregistrationFreezeResult(
        freeze_directory=destination,
        frozen_preregistration_path=destination / "PRE_REGISTRATION_FROZEN.md",
        frozen_primary_config_path=frozen_primary_config,
        timestamped_primary_config_path=destination / "primary_frozen.yaml",
        frozen_confirmatory_config_path=frozen_confirmatory_config,
        timestamped_confirmatory_config_path=destination / "confirmatory_frozen.yaml",
        duplicate_audit_snapshot_path=destination / duplicate_snapshot_name,
        pathology_encoder_audit_snapshot_path=destination / pathology_snapshot_name,
        representation_independence_snapshot_path=destination / independence_snapshot_name,
        pilot_derived_parameters_snapshot_path=destination / pilot_derived_snapshot_name,
        raw_checksum_snapshot_path=destination / raw_snapshot_name,
        pilot_post_seal_verification_path=destination / "pilot_post_seal_verification.json",
        sha256_manifest_path=destination / "sha256_manifest.json",
        immutable_marker_path=destination / ".immutable.json",
        freeze_timestamp_utc=freeze_timestamp,
        pilot_run_id=str(pilot_evidence["run_id"]),
        artifact_root_sha256=artifact_root,
        sha256_manifest_sha256=manifest_digest,
    )


def verify_preregistration_freeze(
    freeze_directory: str | Path,
    *,
    frozen_primary_config_path: str | Path | None = None,
    frozen_confirmatory_config_path: str | Path | None = None,
) -> PreregistrationFreezeVerification:
    """Recompute a freeze snapshot and optionally verify both canonical config copies."""

    directory = Path(freeze_directory).resolve()
    errors: list[str] = []
    if not directory.is_dir():
        return PreregistrationFreezeVerification(
            False, None, None, (), (), (), (f"freeze directory does not exist: {directory}",)
        )
    try:
        manifest = _read_json_mapping(directory / "sha256_manifest.json", role="freeze manifest")
        marker = _read_json_mapping(directory / ".immutable.json", role="freeze marker")
    except ValueError as exc:
        return PreregistrationFreezeVerification(False, None, None, (), (), (), (str(exc),))
    raw_records = manifest.get("artifacts")
    expected_records = (
        [dict(record) for record in raw_records if isinstance(record, Mapping)]
        if isinstance(raw_records, list)
        else []
    )
    if not isinstance(raw_records, list) or len(expected_records) != len(raw_records):
        errors.append("freeze manifest contains malformed artifact records")
    actual_records = _snapshot_records(directory)
    expected_by_path = {str(record.get("path")): record for record in expected_records}
    actual_by_path = {str(record.get("path")): record for record in actual_records}
    missing = tuple(sorted(set(expected_by_path).difference(actual_by_path)))
    added = tuple(sorted(set(actual_by_path).difference(expected_by_path)))
    changed = tuple(
        sorted(
            path
            for path in set(expected_by_path).intersection(actual_by_path)
            if expected_by_path[path].get("size_bytes") != actual_by_path[path].get("size_bytes")
            or expected_by_path[path].get("sha256") != actual_by_path[path].get("sha256")
        )
    )
    actual_root = _canonical_artifact_root(actual_records)
    expected_root_value = manifest.get("artifact_root_sha256")
    expected_root = str(expected_root_value) if expected_root_value is not None else None
    manifest_sha = sha256_file(directory / "sha256_manifest.json")
    if marker.get("status") != "frozen":
        errors.append("immutable marker status is not frozen")
    if marker.get("artifact_root_sha256") != expected_root:
        errors.append("immutable marker root differs from freeze manifest")
    if marker.get("sha256_manifest_sha256") != manifest_sha:
        errors.append("immutable marker does not authenticate the freeze manifest")
    if expected_root is None or not _SHA256_PATTERN.fullmatch(expected_root):
        errors.append("freeze manifest lacks a valid artifact root SHA-256")
    if frozen_primary_config_path is not None:
        canonical = Path(frozen_primary_config_path).resolve()
        snapshot = directory / "primary_frozen.yaml"
        if not canonical.is_file() or sha256_file(canonical) != sha256_file(snapshot):
            errors.append("canonical configs/primary_frozen.yaml differs from the timestamped copy")
    if frozen_confirmatory_config_path is not None:
        canonical = Path(frozen_confirmatory_config_path).resolve()
        snapshot = directory / "confirmatory_frozen.yaml"
        if not canonical.is_file() or sha256_file(canonical) != sha256_file(snapshot):
            errors.append(
                "canonical configs/confirmatory_frozen.yaml differs from the timestamped copy"
            )
    valid = (
        not errors and not missing and not added and not changed and actual_root == expected_root
    )
    return PreregistrationFreezeVerification(
        valid=valid,
        artifact_root_sha256=actual_root,
        expected_artifact_root_sha256=expected_root,
        missing_paths=missing,
        added_paths=added,
        changed_paths=changed,
        errors=tuple(errors),
    )


__all__ = [
    "BASE_FREEZE_EVIDENCE_SCHEMA_VERSION",
    "PreregistrationFreezeResult",
    "PreregistrationFreezeVerification",
    "freeze_preregistration",
    "validate_pathology_encoder_audit",
    "validate_pilot_derived_parameters",
    "verify_preregistration_freeze",
]
