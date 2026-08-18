"""Bounded, zero-training recovery primitives for one interrupted primary orphan.

The module deliberately contains no trainer, matrix executor, statistics aggregator,
recursive invocation, or automatic retry.  It qualifies one read-only source tree,
physically copies an exact content-addressed allowlist, and independently verifies the
destination.  Run creation, sealing, and downstream stage attestation are integrated
at a higher workflow layer.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, cast

import numpy as np

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    PrimaryFilesystemReadbackEvidence,
    PrimaryRestorationReadbackEvidence,
    read_primary_filesystem_evidence,
    read_primary_restoration_evidence,
)
from histo_audit.experiment.primary_core import PrimaryExecutionControls
from histo_audit.experiment.study_contracts import PrimaryMatrixPlan
from histo_audit.pannuke.publication import (
    ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY,
    anchored_physical_copy_session,
)
from histo_audit.utils.run_tracking import sha256_file

RECOVERY_POLICY = "interrupted_unsealed_primary_recovery_v1"
RECOVERY_EXPERIMENT_NAME = "pannuke_primary_orphan_recovery"
SOURCE_EXPERIMENT_NAME = "pannuke_primary_frozen_feature_benchmark"
RECOVERY_EVIDENCE_FILENAME = "primary_recovery_evidence.json"
RECOVERY_REGISTRATION_STATUS = "amended_or_exploratory"
RECOVERY_COPY_POLICY = ANCHORED_PHYSICAL_COPY_WOF_LZX_POLICY

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT = 0x400
_SOURCE_ROOT_FILES = frozenset(
    {
        "cell_index.csv",
        "execution_controls.json",
        "matrix_plan.json",
        "primary_input_bindings.json",
        "reconciliation.json",
        "restoration_index.json",
    }
)
_STATISTICS_FILES = frozenset(
    {
        "primary_bootstrap_evidence.npz",
        "primary_statistics.json",
        "primary_statistics_manifest.json",
        "primary_subgroups.csv",
    }
)
_CELL_ARTIFACT_FILES = frozenset(
    {
        "bootstrap_evidence.npz",
        "cleanlab_evidence.json",
        "cleanlab_evidence.npz",
        "corruption_manifest.json",
        "independence_evidence.json",
        "metrics.json",
        "neighbour_evidence.npz",
        "oof_predictions.npz",
        "oof_provenance.json",
        "ranking.csv",
        "risk_scores.npz",
    }
)
_CELL_DIRECTORY_FILES = _CELL_ARTIFACT_FILES | {"artifact_manifest.json"}
_RESTORATION_FILES = frozenset(
    {"restoration.json", "restoration_evidence.npz", "restoration_manifest.json"}
)
_AUTHORIZATION_KEYS = {
    "schema_version",
    "policy",
    "source_run_id",
    "source_run_directory",
    "interruption_evidence",
    "outcomes_inspected",
    "outcome_inspection_at_utc",
    "analysis_disposition",
    "scientific_method_changes",
    "expected_status_sha256",
    "expected_primary_execution_gate_sha256",
    "expected_source_tree_manifest_sha256",
    "expected_source_tree_root_sha256",
    "expected_source_snapshot_root_sha256",
    "expected_source_filesystem_readback_root_sha256",
    "expected_restoration_readback_root_sha256",
    "expected_statistics_manifest_sha256",
    "trust_assumption",
    "limitation",
    "reason",
}
_INTERRUPTION_KEYS = {
    "kind",
    "observed_at_utc",
    "last_boot_at_utc",
    "event_id",
    "source_process_id",
    "process_checked_at_utc",
    "process_active",
    "receipt_path",
    "receipt_sha256",
}


class PrimaryRecoveryError(RuntimeError):
    """Fail-closed recovery qualification or verification error."""


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_sha(value: object, role: str) -> str:
    if not _valid_sha(value):
        raise PrimaryRecoveryError(f"{role} must be one lowercase SHA-256")
    return cast(str, value)


def _require_utc(value: object, role: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PrimaryRecoveryError(f"{role} must be an explicit UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PrimaryRecoveryError(f"{role} is not valid ISO-8601 UTC") from exc
    if parsed.utcoffset() is None:
        raise PrimaryRecoveryError(f"{role} lacks a UTC offset")
    return value


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrimaryRecoveryError(f"{role} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrimaryRecoveryError(f"{role} must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _assert_plain_path(path: Path, *, boundary: Path, role: str) -> None:
    lexical = Path(os.path.abspath(path))
    root = Path(os.path.abspath(boundary))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise PrimaryRecoveryError(f"{role} escapes its required boundary") from exc
    current = root
    for component in lexical.relative_to(root).parts:
        current /= component
        if not current.exists():
            raise PrimaryRecoveryError(f"{role} path component is missing: {current}")
        value = current.lstat()
        if current.is_symlink() or _is_reparse(value):
            raise PrimaryRecoveryError(f"{role} contains a link or reparse point: {current}")


def _require_regular_file(path: Path, role: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise PrimaryRecoveryError(f"{role} is missing: {path}: {exc}") from exc
    if (
        not stat.S_ISREG(value.st_mode)
        or path.is_symlink()
        or _is_reparse(value)
        or value.st_nlink != 1
    ):
        raise PrimaryRecoveryError(f"{role} is not one unlinked regular file: {path}")
    return value


def _default_pid_probe(pid: int) -> bool:
    psutil = importlib.import_module("psutil")
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


@dataclass(frozen=True, slots=True)
class RecoveryInterruptionEvidence:
    """Sealed host-interruption facts used only to authorize the orphan path."""

    kind: str
    observed_at_utc: str
    last_boot_at_utc: str
    event_id: int
    source_process_id: int
    process_checked_at_utc: str
    process_active: bool
    receipt_path: Path
    receipt_sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping) or set(value) != _INTERRUPTION_KEYS:
            raise PrimaryRecoveryError("interruption evidence has missing or extra fields")
        receipt = Path(str(value.get("receipt_path", "")))
        if not receipt.is_absolute():
            raise PrimaryRecoveryError("interruption receipt path must be absolute")
        event_id = value.get("event_id")
        process_id = value.get("source_process_id")
        if (
            value.get("kind") != "host_reboot"
            or type(event_id) is not int
            or int(event_id) < 1
            or type(process_id) is not int
            or int(process_id) < 1
            or value.get("process_active") is not False
        ):
            raise PrimaryRecoveryError("interruption evidence is not an inactive reboot case")
        return cls(
            kind="host_reboot",
            observed_at_utc=_require_utc(value.get("observed_at_utc"), "observation timestamp"),
            last_boot_at_utc=_require_utc(value.get("last_boot_at_utc"), "last boot timestamp"),
            event_id=int(event_id),
            source_process_id=int(process_id),
            process_checked_at_utc=_require_utc(
                value.get("process_checked_at_utc"), "process-check timestamp"
            ),
            process_active=False,
            receipt_path=receipt.resolve(),
            receipt_sha256=_require_sha(
                value.get("receipt_sha256"), "interruption receipt SHA-256"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["receipt_path"] = str(self.receipt_path)
        return payload


@dataclass(frozen=True, slots=True)
class RecoveryAuthorization:
    """Typed recovery authorization embedded in one verified amendment."""

    source_run_id: str
    source_run_directory: Path
    interruption: RecoveryInterruptionEvidence
    outcomes_inspected: bool
    outcome_inspection_at_utc: str
    analysis_disposition: str
    expected_status_sha256: str
    expected_primary_execution_gate_sha256: str
    expected_source_tree_manifest_sha256: str
    expected_source_tree_root_sha256: str
    expected_source_snapshot_root_sha256: str
    expected_source_filesystem_readback_root_sha256: str
    expected_restoration_readback_root_sha256: str
    expected_statistics_manifest_sha256: str
    trust_assumption: str
    limitation: str
    reason: str
    canonical_sha256: str
    authority_directory: Path | None = None
    authority_artifact_root_sha256: str | None = None
    authority_manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        authority_directory: str | Path | None = None,
        authority_artifact_root_sha256: str | None = None,
        authority_manifest_sha256: str | None = None,
    ) -> Self:
        if set(value) != _AUTHORIZATION_KEYS:
            raise PrimaryRecoveryError("recovery authorization has missing or extra fields")
        source_run_id = value.get("source_run_id")
        source_directory = Path(str(value.get("source_run_directory", "")))
        if (
            value.get("schema_version") != 1
            or value.get("policy") != RECOVERY_POLICY
            or not isinstance(source_run_id, str)
            or not source_run_id.strip()
            or Path(source_run_id).name != source_run_id
            or not source_directory.is_absolute()
            or source_directory.name != source_run_id
            or value.get("outcomes_inspected") is not True
            or value.get("analysis_disposition") != RECOVERY_REGISTRATION_STATUS
            or value.get("scientific_method_changes") != []
        ):
            raise PrimaryRecoveryError("recovery authorization policy or disposition is invalid")
        trust = value.get("trust_assumption")
        limitation = value.get("limitation")
        reason = value.get("reason")
        if not all(isinstance(item, str) and item.strip() for item in (trust, limitation, reason)):
            raise PrimaryRecoveryError("recovery trust, limitation, and reason must be explicit")
        authority_path = Path(authority_directory).resolve() if authority_directory else None
        if (authority_artifact_root_sha256 is None) != (authority_manifest_sha256 is None):
            raise PrimaryRecoveryError("outer amendment hash bindings are incomplete")
        outer_root = (
            _require_sha(authority_artifact_root_sha256, "authority artifact root")
            if authority_artifact_root_sha256 is not None
            else None
        )
        outer_manifest = (
            _require_sha(authority_manifest_sha256, "authority manifest SHA-256")
            if authority_manifest_sha256 is not None
            else None
        )
        return cls(
            source_run_id=source_run_id,
            source_run_directory=source_directory.resolve(),
            interruption=RecoveryInterruptionEvidence.from_mapping(
                value.get("interruption_evidence")
            ),
            outcomes_inspected=True,
            outcome_inspection_at_utc=_require_utc(
                value.get("outcome_inspection_at_utc"), "outcome-inspection timestamp"
            ),
            analysis_disposition=RECOVERY_REGISTRATION_STATUS,
            expected_status_sha256=_require_sha(
                value.get("expected_status_sha256"), "source status SHA-256"
            ),
            expected_primary_execution_gate_sha256=_require_sha(
                value.get("expected_primary_execution_gate_sha256"),
                "source primary gate SHA-256",
            ),
            expected_source_tree_manifest_sha256=_require_sha(
                value.get("expected_source_tree_manifest_sha256"),
                "source-tree manifest SHA-256",
            ),
            expected_source_tree_root_sha256=_require_sha(
                value.get("expected_source_tree_root_sha256"), "source-tree root SHA-256"
            ),
            expected_source_snapshot_root_sha256=_require_sha(
                value.get("expected_source_snapshot_root_sha256"),
                "source snapshot root SHA-256",
            ),
            expected_source_filesystem_readback_root_sha256=_require_sha(
                value.get("expected_source_filesystem_readback_root_sha256"),
                "source filesystem readback root",
            ),
            expected_restoration_readback_root_sha256=_require_sha(
                value.get("expected_restoration_readback_root_sha256"),
                "source restoration readback root",
            ),
            expected_statistics_manifest_sha256=_require_sha(
                value.get("expected_statistics_manifest_sha256"),
                "source statistics manifest SHA-256",
            ),
            trust_assumption=cast(str, trust),
            limitation=cast(str, limitation),
            reason=cast(str, reason),
            canonical_sha256=canonical_sha256(dict(value)),
            authority_directory=authority_path,
            authority_artifact_root_sha256=outer_root,
            authority_manifest_sha256=outer_manifest,
        )


@dataclass(frozen=True, slots=True, order=True)
class RecoveryArtifact:
    """One exact allowlisted file copied from the orphan."""

    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrphanSourceSnapshot:
    """Typed, content-addressed source inventory without a terminal-run claim."""

    run_directory: Path
    artifacts: tuple[RecoveryArtifact, ...]
    snapshot_root_sha256: str
    filesystem_readback: PrimaryFilesystemReadbackEvidence
    restoration_readback: PrimaryRestorationReadbackEvidence
    statistics_manifest_sha256: str
    completed_required_cell_count: int
    skipped_optional_cell_count: int
    all_cell_ids: tuple[str, ...]
    restoration_cell_ids: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return sum(record.size_bytes for record in self.artifacts)


@dataclass(frozen=True, slots=True)
class OrphanSourceInspection:
    """One authorization-matched, inactive, read-only orphan."""

    authorization: RecoveryAuthorization
    snapshot: OrphanSourceSnapshot
    source_status_sha256: str
    source_primary_gate_sha256: str
    source_tree_manifest_sha256: str
    source_tree_root_sha256: str


@dataclass(frozen=True, slots=True)
class RecoveryCopyReceipt:
    """Receipt for one non-retrying physical copy."""

    source_run_directory: Path
    destination_directory: Path
    copy_policy: str
    artifact_count: int
    total_bytes: int
    snapshot_root_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_run_directory"] = str(self.source_run_directory)
        payload["destination_directory"] = str(self.destination_directory)
        return payload


@dataclass(frozen=True, slots=True)
class RecoveryDestinationVerification:
    """Independent typed verification of the copied destination."""

    destination_directory: Path
    snapshot_root_sha256: str
    filesystem_readback: PrimaryFilesystemReadbackEvidence
    restoration_readback: PrimaryRestorationReadbackEvidence
    statistics_manifest_sha256: str
    statistics_comparison_count: int
    bootstrap_saved_draw_count: int


def _artifact(
    path: Path, relative_path: str, *, expected_sha: str | None = None
) -> RecoveryArtifact:
    value = _require_regular_file(path, f"recovery artifact {relative_path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise PrimaryRecoveryError(f"recovery artifact hash differs: {relative_path}")
    return RecoveryArtifact(relative_path, value.st_size, digest)


def _manifest_artifacts(cell_directory: Path, cell_id: str) -> tuple[RecoveryArtifact, ...]:
    manifest_path = cell_directory / "artifact_manifest.json"
    manifest = _read_json_object(manifest_path, f"cell {cell_id} artifact manifest")
    rows = manifest.get("artifacts")
    if manifest.get("schema_version") != 1 or not isinstance(rows, list):
        raise PrimaryRecoveryError(f"cell {cell_id} artifact manifest schema is invalid")
    records: list[RecoveryArtifact] = []
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise PrimaryRecoveryError(f"cell {cell_id} artifact record is invalid")
        name = raw.get("path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name not in _CELL_ARTIFACT_FILES
            or name in names
            or type(size) is not int
            or int(size) < 0
            or not _valid_sha(digest)
        ):
            raise PrimaryRecoveryError(f"cell {cell_id} artifact record is unsafe")
        path = cell_directory / name
        value = _require_regular_file(path, f"cell {cell_id} artifact {name}")
        # The immediately preceding typed filesystem readback already rehashed every
        # cell artifact against this exact manifest.  Repeating that 43-GiB pass here
        # would add no independent boundary: the anchored copy below streams and
        # rechecks each expected digest again.
        if value.st_size != size:
            raise PrimaryRecoveryError(f"cell {cell_id} artifact size differs: {name}")
        names.add(name)
        records.append(RecoveryArtifact(f"cells/{cell_id}/{name}", int(size), cast(str, digest)))
    if names != _CELL_ARTIFACT_FILES:
        raise PrimaryRecoveryError(f"cell {cell_id} artifact manifest is incomplete")
    return tuple(records)


def _statistics_artifacts(run_path: Path) -> tuple[tuple[RecoveryArtifact, ...], dict[str, Any]]:
    manifest_path = run_path / "primary_statistics_manifest.json"
    manifest = _read_json_object(manifest_path, "primary statistics manifest")
    rows = manifest.get("artifacts")
    if manifest.get("schema_version") != 1 or not isinstance(rows, list):
        raise PrimaryRecoveryError("primary statistics manifest schema is invalid")
    records = [
        _artifact(manifest_path, "primary_statistics_manifest.json"),
    ]
    expected_outputs = _STATISTICS_FILES - {"primary_statistics_manifest.json"}
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise PrimaryRecoveryError("statistics artifact record is invalid")
        name = raw.get("path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or name not in expected_outputs
            or name in names
            or type(size) is not int
            or int(size) < 0
            or not _valid_sha(digest)
        ):
            raise PrimaryRecoveryError("statistics artifact record is unsafe")
        path = run_path / name
        value = _require_regular_file(path, f"statistics artifact {name}")
        if value.st_size != size or sha256_file(path) != digest:
            raise PrimaryRecoveryError(f"statistics artifact differs from its manifest: {name}")
        names.add(name)
        records.append(RecoveryArtifact(name, int(size), cast(str, digest)))
    if names != expected_outputs:
        raise PrimaryRecoveryError("statistics artifact set is incomplete")
    return tuple(records), manifest


def collect_orphan_source_snapshot(
    run_directory: str | Path,
    *,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
) -> OrphanSourceSnapshot:
    """Perform the bounded typed readbacks and derive the exact copy allowlist."""

    run_path = Path(run_directory).resolve()
    if not run_path.is_dir():
        raise PrimaryRecoveryError(f"orphan source directory does not exist: {run_path}")
    readback = read_primary_filesystem_evidence(plan, run_path)
    restoration = read_primary_restoration_evidence(run_path, controls)
    if (
        not readback.passed
        or not restoration.passed
        or readback.run_directory.resolve() != run_path
        or restoration.run_directory.resolve() != run_path
        or restoration.source_readback_root_sha256 != readback.readback_root_sha256
        or readback.completed_required_cell_count != plan.required_cell_count
        or readback.completed_cell_count != plan.required_cell_count
        or readback.skipped_optional_cell_count != plan.optional_cell_count
    ):
        raise PrimaryRecoveryError("orphan source failed exact matrix/restoration qualification")

    records: list[RecoveryArtifact] = [
        _artifact(run_path / name, name) for name in sorted(_SOURCE_ROOT_FILES)
    ]
    completed_manifests = dict(readback.cell_artifact_manifest_sha256)
    if len(completed_manifests) != plan.required_cell_count:
        raise PrimaryRecoveryError("orphan source completed-cell manifest count is invalid")
    all_cell_ids: list[str] = []
    for cell in plan.cells:
        all_cell_ids.append(cell.cell_id)
        cell_path = run_path / "cells" / cell.cell_id
        if not cell_path.is_dir():
            raise PrimaryRecoveryError(f"orphan source lacks cell directory: {cell.cell_id}")
        entries = {entry.name for entry in cell_path.iterdir()}
        if cell.cell_id in completed_manifests:
            if not cell.required or entries != _CELL_DIRECTORY_FILES:
                raise PrimaryRecoveryError(
                    f"completed cell is optional or has a non-exact file set: {cell.cell_id}"
                )
            records.append(
                _artifact(
                    cell_path / "artifact_manifest.json",
                    f"cells/{cell.cell_id}/artifact_manifest.json",
                    expected_sha=completed_manifests[cell.cell_id],
                )
            )
            records.extend(_manifest_artifacts(cell_path, cell.cell_id))
        elif cell.required or entries:
            raise PrimaryRecoveryError(
                f"required or non-empty cell is absent from completed evidence: {cell.cell_id}"
            )

    scenario_hashes = dict(readback.scenario_artifact_sha256)
    if set(scenario_hashes) != {scenario.scenario_id for scenario in plan.scenarios}:
        raise PrimaryRecoveryError("orphan scenario evidence is incomplete")
    for scenario_id, digest in sorted(scenario_hashes.items()):
        records.append(
            _artifact(
                run_path / "corruption_scenarios" / f"{scenario_id}.json",
                f"corruption_scenarios/{scenario_id}.json",
                expected_sha=digest,
            )
        )

    restoration_hashes = {
        "restoration.json": dict(restoration.cell_json_sha256),
        "restoration_evidence.npz": dict(restoration.cell_evidence_sha256),
        "restoration_manifest.json": dict(restoration.cell_manifest_sha256),
    }
    restoration_ids = tuple(cell_id for cell_id, _ in restoration.cell_json_sha256)
    if any(set(values) != set(restoration_ids) for values in restoration_hashes.values()):
        raise PrimaryRecoveryError("orphan restoration hash sets differ")
    for cell_id in restoration_ids:
        for name in sorted(_RESTORATION_FILES):
            records.append(
                _artifact(
                    run_path / "restorations" / cell_id / name,
                    f"restorations/{cell_id}/{name}",
                    expected_sha=restoration_hashes[name][cell_id],
                )
            )

    statistics_records, _ = _statistics_artifacts(run_path)
    records.extend(statistics_records)
    records_tuple = tuple(sorted(records))
    paths = [record.path for record in records_tuple]
    if len(paths) != len(set(paths)):
        raise PrimaryRecoveryError("orphan recovery allowlist contains duplicate paths")
    snapshot_root = canonical_sha256([record.as_dict() for record in records_tuple])
    return OrphanSourceSnapshot(
        run_directory=run_path,
        artifacts=records_tuple,
        snapshot_root_sha256=snapshot_root,
        filesystem_readback=readback,
        restoration_readback=restoration,
        statistics_manifest_sha256=sha256_file(run_path / "primary_statistics_manifest.json"),
        completed_required_cell_count=readback.completed_required_cell_count,
        skipped_optional_cell_count=readback.skipped_optional_cell_count,
        all_cell_ids=tuple(all_cell_ids),
        restoration_cell_ids=restoration_ids,
    )


def build_primary_recovery_authorization(
    snapshot: OrphanSourceSnapshot,
    *,
    interruption_receipt_path: str | Path,
    interruption_observed_at_utc: str,
    last_boot_at_utc: str,
    event_id: int,
    source_process_id: int,
    process_checked_at_utc: str,
    outcome_inspection_at_utc: str,
    trust_assumption: str,
    limitation: str,
    reason: str,
    pid_probe: Callable[[int], bool] = _default_pid_probe,
) -> dict[str, Any]:
    """Build the plain mapping that must subsequently be sealed by an amendment."""

    if not isinstance(snapshot, OrphanSourceSnapshot):
        raise TypeError("recovery authorization requires a typed orphan snapshot")
    source = snapshot.run_directory
    status_path = source / "status.json"
    gate_path = source / "primary_execution_gate.json"
    source_tree_path = source / "source_tree_manifest.json"
    status = _read_json_object(status_path, "orphan status")
    source_tree = _read_json_object(source_tree_path, "orphan source-tree manifest")
    source_tree_root = source_tree.get("root_sha256")
    receipt = Path(interruption_receipt_path).resolve()
    if (
        status.get("status") != "running"
        or status.get("run_id") != source.name
        or status.get("experiment_name") != SOURCE_EXPERIMENT_NAME
        or not _valid_sha(source_tree_root)
        or type(event_id) is not int
        or event_id < 1
        or type(source_process_id) is not int
        or source_process_id < 1
        or pid_probe(source_process_id)
    ):
        raise PrimaryRecoveryError("source is not one inactive running primary orphan")
    for forbidden in (".immutable.json", "artifact_manifest.json", "failure.json"):
        if (source / forbidden).exists():
            raise PrimaryRecoveryError(f"source is not an unsealed orphan: {forbidden}")
    _require_regular_file(receipt, "interruption receipt")
    if not all(
        isinstance(value, str) and value.strip() for value in (trust_assumption, limitation, reason)
    ):
        raise PrimaryRecoveryError("recovery trust, limitation, and reason must be explicit")
    mapping: dict[str, Any] = {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "source_run_id": source.name,
        "source_run_directory": str(source),
        "interruption_evidence": {
            "kind": "host_reboot",
            "observed_at_utc": _require_utc(
                interruption_observed_at_utc, "interruption observation timestamp"
            ),
            "last_boot_at_utc": _require_utc(last_boot_at_utc, "last boot timestamp"),
            "event_id": event_id,
            "source_process_id": source_process_id,
            "process_checked_at_utc": _require_utc(
                process_checked_at_utc, "process-check timestamp"
            ),
            "process_active": False,
            "receipt_path": str(receipt),
            "receipt_sha256": sha256_file(receipt),
        },
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": _require_utc(
            outcome_inspection_at_utc, "outcome-inspection timestamp"
        ),
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "scientific_method_changes": [],
        "expected_status_sha256": sha256_file(status_path),
        "expected_primary_execution_gate_sha256": sha256_file(gate_path),
        "expected_source_tree_manifest_sha256": sha256_file(source_tree_path),
        "expected_source_tree_root_sha256": source_tree_root,
        "expected_source_snapshot_root_sha256": snapshot.snapshot_root_sha256,
        "expected_source_filesystem_readback_root_sha256": (
            snapshot.filesystem_readback.readback_root_sha256
        ),
        "expected_restoration_readback_root_sha256": (
            snapshot.restoration_readback.readback_root_sha256
        ),
        "expected_statistics_manifest_sha256": snapshot.statistics_manifest_sha256,
        "trust_assumption": trust_assumption,
        "limitation": limitation,
        "reason": reason,
    }
    RecoveryAuthorization.from_mapping(mapping)
    return mapping


def inspect_orphan_source(
    *,
    runs_root: str | Path,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
    pid_probe: Callable[[int], bool] = _default_pid_probe,
) -> OrphanSourceInspection:
    """Qualify one exact inactive orphan without mutating it or creating a new run."""

    run_root = Path(runs_root).resolve()
    source = authorization.source_run_directory.resolve()
    if source != run_root / authorization.source_run_id or source.parent != run_root:
        raise PrimaryRecoveryError("authorized orphan is not the exact runs-root child")
    _assert_plain_path(source, boundary=run_root, role="orphan source")
    if pid_probe(authorization.interruption.source_process_id):
        raise PrimaryRecoveryError("authorized orphan process is still active")
    receipt = authorization.interruption.receipt_path
    _require_regular_file(receipt, "interruption receipt")
    if sha256_file(receipt) != authorization.interruption.receipt_sha256:
        raise PrimaryRecoveryError("interruption receipt differs from the amendment")
    for forbidden in (".immutable.json", "artifact_manifest.json", "failure.json"):
        if (source / forbidden).exists():
            raise PrimaryRecoveryError(f"source is not an unsealed running orphan: {forbidden}")
    status_path = source / "status.json"
    status = _read_json_object(status_path, "orphan status")
    if (
        status.get("status") != "running"
        or status.get("run_id") != authorization.source_run_id
        or status.get("experiment_name") != SOURCE_EXPERIMENT_NAME
        or sha256_file(status_path) != authorization.expected_status_sha256
    ):
        raise PrimaryRecoveryError("orphan status differs from its authorization")
    gate_path = source / "primary_execution_gate.json"
    source_tree_path = source / "source_tree_manifest.json"
    gate_sha = sha256_file(gate_path)
    source_manifest_sha = sha256_file(source_tree_path)
    source_tree = _read_json_object(source_tree_path, "orphan source-tree manifest")
    source_root = source_tree.get("root_sha256")
    if (
        gate_sha != authorization.expected_primary_execution_gate_sha256
        or source_manifest_sha != authorization.expected_source_tree_manifest_sha256
        or source_root != authorization.expected_source_tree_root_sha256
    ):
        raise PrimaryRecoveryError("orphan gate/source-tree identity differs")
    snapshot = collect_orphan_source_snapshot(source, plan=plan, controls=controls)
    if (
        snapshot.snapshot_root_sha256 != authorization.expected_source_snapshot_root_sha256
        or snapshot.filesystem_readback.readback_root_sha256
        != authorization.expected_source_filesystem_readback_root_sha256
        or snapshot.restoration_readback.readback_root_sha256
        != authorization.expected_restoration_readback_root_sha256
        or snapshot.statistics_manifest_sha256 != authorization.expected_statistics_manifest_sha256
    ):
        raise PrimaryRecoveryError("orphan typed snapshot differs from its sealed authorization")
    return OrphanSourceInspection(
        authorization=authorization,
        snapshot=snapshot,
        source_status_sha256=authorization.expected_status_sha256,
        source_primary_gate_sha256=gate_sha,
        source_tree_manifest_sha256=source_manifest_sha,
        source_tree_root_sha256=cast(str, source_root),
    )


def copy_authorized_orphan_artifacts(
    inspection: OrphanSourceInspection,
    destination_directory: str | Path,
) -> RecoveryCopyReceipt:
    """Perform exactly one anchored physical copy of the authorized allowlist."""

    if not isinstance(inspection, OrphanSourceInspection):
        raise TypeError("recovery copy requires a genuine orphan inspection")
    destination = Path(destination_directory).resolve()
    if not destination.is_dir() or destination == inspection.snapshot.run_directory:
        raise PrimaryRecoveryError("recovery destination must be a distinct existing directory")
    with anchored_physical_copy_session(
        inspection.snapshot.run_directory,
        destination,
        compression_policy=RECOVERY_COPY_POLICY,
    ) as session:
        for record in inspection.snapshot.artifacts:
            published = session.copy_file_no_overwrite(
                record.path,
                expected_size_bytes=record.size_bytes,
                expected_sha256=record.sha256,
            )
            if (
                published.path != destination / Path(record.path)
                or published.kind != "file"
                or published.sha256 != record.sha256
                or published.identity[2] != record.size_bytes
            ):
                raise PrimaryRecoveryError(f"physical-copy receipt differs: {record.path}")
        for cell_id in inspection.snapshot.all_cell_ids:
            session.ensure_directory(f"cells/{cell_id}")
        session.ensure_directory("corruption_scenarios")
        for cell_id in inspection.snapshot.restoration_cell_ids:
            session.ensure_directory(f"restorations/{cell_id}")
    return RecoveryCopyReceipt(
        source_run_directory=inspection.snapshot.run_directory,
        destination_directory=destination,
        copy_policy=RECOVERY_COPY_POLICY,
        artifact_count=len(inspection.snapshot.artifacts),
        total_bytes=inspection.snapshot.total_bytes,
        snapshot_root_sha256=inspection.snapshot.snapshot_root_sha256,
    )


def _verify_statistics_closure(
    run_path: Path,
    controls: PrimaryExecutionControls,
    *,
    source_readback_root_sha256: str,
) -> tuple[int, int]:
    statistics = _read_json_object(run_path / "primary_statistics.json", "primary statistics")
    manifest = _read_json_object(
        run_path / "primary_statistics_manifest.json", "primary statistics manifest"
    )
    expected_ids = [
        *(definition.comparison_id for definition in controls.within_cell_comparisons),
        *(definition.comparison_id for definition in controls.method_vs_random_comparisons),
        *(definition.comparison_id for definition in controls.cross_cell_comparisons),
    ]
    comparisons = statistics.get("comparisons")
    bootstrap = statistics.get("bootstrap")
    if (
        statistics.get("schema_version") != 1
        or statistics.get("analysis_scope") != "real_pannuke_primary_statistics"
        or statistics.get("execution_controls_binding_sha256") != controls.binding_sha256
        or statistics.get("matrix_plan_sha256") != controls.plan_sha256
        or statistics.get("source_filesystem_readback_root_sha256") != source_readback_root_sha256
        or not isinstance(comparisons, list)
        or [item.get("comparison_id") for item in comparisons if isinstance(item, Mapping)]
        != expected_ids
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("requested_iterations") != controls.bootstrap_iterations
        or bootstrap.get("saved_draw_count") != controls.bootstrap_iterations
        or bootstrap.get("seed") != controls.bootstrap_seed
        or manifest.get("execution_controls_binding_sha256") != controls.binding_sha256
        or manifest.get("source_filesystem_readback_root_sha256") != source_readback_root_sha256
        or manifest.get("statistics_payload_sha256") != canonical_sha256(statistics)
    ):
        raise PrimaryRecoveryError("statistics quartet fails lightweight semantic closure")

    expected_npz_keys = {
        "comparison_ids",
        "comparison_kinds",
        "draw_indices",
        "draw_offsets",
        "group_ids",
        "random_review_seeds",
        "sample_ids",
        "tissue_types",
    }
    for index in range(len(expected_ids)):
        prefix = f"comparison_{index:03d}"
        expected_npz_keys.update(
            {
                f"{prefix}_valid_draw_indices",
                f"{prefix}_metric_a",
                f"{prefix}_metric_b",
                f"{prefix}_differences",
            }
        )
    with np.load(run_path / "primary_bootstrap_evidence.npz", allow_pickle=False) as archive:
        if set(archive.files) != expected_npz_keys:
            raise PrimaryRecoveryError("bootstrap evidence has missing or extra arrays")
        comparison_ids = np.asarray(archive["comparison_ids"])
        comparison_kinds = np.asarray(archive["comparison_kinds"])
        random_seeds = np.asarray(archive["random_review_seeds"])
        draw_offsets = np.asarray(archive["draw_offsets"])
        if (
            comparison_ids.ndim != 1
            or comparison_ids.astype(str).tolist() != expected_ids
            or comparison_kinds.shape != comparison_ids.shape
            or random_seeds.shape != (controls.random_review_repeats,)
            or draw_offsets.ndim != 1
            or len(draw_offsets) != controls.bootstrap_iterations + 1
        ):
            raise PrimaryRecoveryError("bootstrap evidence structural dimensions are invalid")
        for index in range(len(expected_ids)):
            prefix = f"comparison_{index:03d}"
            valid = np.asarray(archive[f"{prefix}_valid_draw_indices"])
            method_a = np.asarray(archive[f"{prefix}_metric_a"])
            method_b = np.asarray(archive[f"{prefix}_metric_b"])
            differences = np.asarray(archive[f"{prefix}_differences"])
            if (
                valid.ndim != 1
                or method_a.ndim != 1
                or method_b.ndim != 1
                or differences.ndim != 1
                or not (len(valid) == len(method_a) == len(method_b) == len(differences))
                or len(valid) > controls.bootstrap_iterations
                or not np.array_equal(differences, method_a - method_b, equal_nan=True)
            ):
                raise PrimaryRecoveryError("bootstrap comparison arrays fail closure")
    return len(expected_ids), int(bootstrap["saved_draw_count"])


def verify_recovery_destination(
    inspection: OrphanSourceInspection,
    destination_directory: str | Path,
    *,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
) -> RecoveryDestinationVerification:
    """Independently verify the destination without training or bootstrap recomputation."""

    if not isinstance(inspection, OrphanSourceInspection):
        raise TypeError("destination verification requires a genuine orphan inspection")
    destination = Path(destination_directory).resolve()
    snapshot = collect_orphan_source_snapshot(destination, plan=plan, controls=controls)
    if (
        snapshot.snapshot_root_sha256 != inspection.snapshot.snapshot_root_sha256
        or snapshot.filesystem_readback.readback_root_sha256
        != inspection.snapshot.filesystem_readback.readback_root_sha256
        or snapshot.restoration_readback.readback_root_sha256
        != inspection.snapshot.restoration_readback.readback_root_sha256
        or snapshot.statistics_manifest_sha256 != inspection.snapshot.statistics_manifest_sha256
    ):
        raise PrimaryRecoveryError("recovery destination differs from the authorized source")
    comparison_count, saved_draw_count = _verify_statistics_closure(
        destination,
        controls,
        source_readback_root_sha256=snapshot.filesystem_readback.readback_root_sha256,
    )
    return RecoveryDestinationVerification(
        destination_directory=destination,
        snapshot_root_sha256=snapshot.snapshot_root_sha256,
        filesystem_readback=snapshot.filesystem_readback,
        restoration_readback=snapshot.restoration_readback,
        statistics_manifest_sha256=snapshot.statistics_manifest_sha256,
        statistics_comparison_count=comparison_count,
        bootstrap_saved_draw_count=saved_draw_count,
    )


__all__ = [
    "RECOVERY_COPY_POLICY",
    "RECOVERY_EVIDENCE_FILENAME",
    "RECOVERY_EXPERIMENT_NAME",
    "RECOVERY_POLICY",
    "RECOVERY_REGISTRATION_STATUS",
    "OrphanSourceInspection",
    "OrphanSourceSnapshot",
    "PrimaryRecoveryError",
    "RecoveryArtifact",
    "RecoveryAuthorization",
    "RecoveryCopyReceipt",
    "RecoveryDestinationVerification",
    "RecoveryInterruptionEvidence",
    "build_primary_recovery_authorization",
    "collect_orphan_source_snapshot",
    "copy_authorized_orphan_artifacts",
    "inspect_orphan_source",
    "verify_recovery_destination",
]
