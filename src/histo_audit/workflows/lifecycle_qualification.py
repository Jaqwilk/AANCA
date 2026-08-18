"""Operational lifecycle qualification for costly study executions.

The rehearsal is deliberately synthetic and carries no scientific completion
stage.  It exercises the real persistence, physical-copy publication, sealing,
registry, integrity, and fresh-process verification boundaries used by costly
study runs.  A separate sealed verification run is the durable readiness token.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from histo_audit.config import config_sha256, load_config
from histo_audit.models.mlp import _atomic_torch_save
from histo_audit.pannuke.publication import (
    anchored_physical_copy_session,
)
from histo_audit.utils.run_tracking import (
    ARTIFACT_MANIFEST_FILENAME,
    IMMUTABLE_MARKER,
    INTEGRITY_REGISTRY_FILENAME,
    REGISTRY_COLUMNS,
    STATUS_FILENAME,
    RunTracker,
    append_registry_row,
    atomic_write_json,
    attest_lifecycle_run_qualification,
    build_lifecycle_qualification_attestation_verification,
    capture_source_tree,
    require_lifecycle_run_qualified,
    require_run_stage_eligible,
    sha256_file,
    verify_run_integrity,
)

from .original_confirmatory_technical_authority_publication_v1 import (
    VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1,
    verify_published_original_confirmatory_technical_authority_v1,
)
from .preregistration import verify_preregistration_freeze
from .preregistration_amendment import (
    RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE,
    RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
    _require_sealed_effective_resource_bounded_confirmatory_authorization,
    verify_preregistration_amendment,
)

LIFECYCLE_REHEARSAL_EXPERIMENT = "lifecycle_qualification_rehearsal"
LIFECYCLE_READINESS_EXPERIMENT = "lifecycle_qualification_verification"
LIFECYCLE_SCHEMA_VERSION = 1
LIFECYCLE_COMPLETION_FILENAME = "completion_evidence.json"
LIFECYCLE_READINESS_FILENAME = "lifecycle_readiness_evidence.json"
LIFECYCLE_PLAN_FILENAME = "lifecycle_plan.json"
LIFECYCLE_AUTHORITY_FILENAME = "authority_binding.json"
LIFECYCLE_PUBLICATION_FILENAME = "publication_receipt.json"
ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND = "original_confirmatory_technical_authority_v1"
_SHA256_FIELDS = "0123456789abcdef"

_RUN_TRACKER_TERMINAL_FILES = frozenset(
    {
        ".immutable.json",
        "artifact_manifest.json",
        "checksums.json",
        "environment.json",
        "events.jsonl",
        "git_state.json",
        "resolved_config.yaml",
        "run.log",
        "run_provenance.json",
        "runtime.json",
        "source_tree_manifest.json",
        "status.json",
    }
)
_REHEARSAL_FILES = _RUN_TRACKER_TERMINAL_FILES | {
    LIFECYCLE_AUTHORITY_FILENAME,
    LIFECYCLE_COMPLETION_FILENAME,
    LIFECYCLE_PLAN_FILENAME,
    LIFECYCLE_PUBLICATION_FILENAME,
    "checkpoint.pt",
    "checkpoint_attestation.json",
    "restoration.json",
    "restoration_attestation.json",
    "statistics.json",
    "statistics_attestation.json",
}
_READINESS_FILES = _RUN_TRACKER_TERMINAL_FILES | {LIFECYCLE_READINESS_FILENAME}

_REHEARSAL_CONFIG: dict[str, Any] = {
    "schema_version": LIFECYCLE_SCHEMA_VERSION,
    "experiment_name": LIFECYCLE_REHEARSAL_EXPERIMENT,
    "purpose": "operational_lifecycle_qualification_non_scientific",
    "seed": 1701,
    "synthetic_group_count": 2,
    "synthetic_sample_count": 10,
    "class_count": 5,
    "checkpoint_format": "torch_weights_only_safe_payload_v1",
    "physical_publication": "anchored_physical_copy_no_overwrite_single_link_v1",
    "fresh_process_verification_required": True,
    "scientific_outcome": False,
}

_REHEARSAL_PLAN: dict[str, Any] = {
    "schema_version": LIFECYCLE_SCHEMA_VERSION,
    "policy": "costly_study_lifecycle_qualification_v1",
    "steps": [
        "runner",
        "checkpoint",
        "completion",
        "physical_publisher",
        "seal",
        "integrity",
        "independent_fresh_process_verifier",
        "registry",
    ],
    "required_attestations": ["checkpoint", "statistics", "restoration"],
    "scientific_completion_stage_claimed": False,
}


class LifecycleQualificationError(RuntimeError):
    """The operational readiness evidence failed closed."""


@dataclass(frozen=True, slots=True)
class LifecycleRehearsalResult:
    run_directory: Path
    run_id: str
    artifact_root_sha256: str
    completion_evidence_sha256: str
    config_semantic_sha256: str
    plan_sha256: str
    authority_binding_sha256: str
    fresh_process_verification_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class LifecycleReadinessVerification:
    valid: bool
    readiness_run_directory: Path
    rehearsal_run_directory: Path | None
    qualification_binding_sha256: str | None
    readiness_record_sha256: str | None
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class LifecycleReadinessResult:
    readiness_run_directory: Path
    readiness_run_id: str
    rehearsal_run_directory: Path
    artifact_root_sha256: str
    qualification_binding_sha256: str
    readiness_record_sha256: str
    decision: str = "passed"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins:
    """Closed identity pins derived by the strict published-T0 lifecycle gate."""

    namespace_directory: Path
    namespace_claim_sha256: str
    technical_authority_directory: Path
    technical_authority_artifact_root_sha256: str
    technical_authorization_sha256: str
    published_technical_authority_lifecycle_binding_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.namespace_claim_sha256,
            self.technical_authority_artifact_root_sha256,
            self.technical_authorization_sha256,
            self.published_technical_authority_lifecycle_binding_sha256,
        )
        if (
            self.namespace_directory != self.namespace_directory.resolve()
            or self.technical_authority_directory != self.technical_authority_directory.resolve()
            or self.technical_authority_directory.parent != self.namespace_directory
            or not all(_valid_sha256(value) for value in hashes)
        ):
            raise LifecycleQualificationError(
                "published original-confirmatory lifecycle pins are invalid"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryPublishedT0LifecycleReadinessVerification:
    """Readiness plus the exact published-T0 carrier and its six derived pins."""

    readiness: LifecycleReadinessVerification
    verified_published_technical_authority: (
        VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1
    )
    published_t0_pins: PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins

    def __post_init__(self) -> None:
        try:
            binding = dict(self.verified_published_technical_authority.lifecycle_binding())
            authority = self.verified_published_technical_authority.authority
            binding_sha256 = binding.get("binding_sha256")
            binding_unsigned = {
                key: value for key, value in binding.items() if key != "binding_sha256"
            }
            binding_hash_matches = _valid_sha256(
                binding_sha256
            ) and binding_sha256 == _canonical_sha256(binding_unsigned)
        except (AttributeError, TypeError, ValueError) as exc:
            raise LifecycleQualificationError(
                "published original-confirmatory readiness carrier is invalid"
            ) from exc
        pins = self.published_t0_pins
        if (
            type(self.readiness) is not LifecycleReadinessVerification
            or not self.readiness.valid
            or self.readiness.errors
            or type(self.verified_published_technical_authority)
            is not VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1
            or type(pins) is not PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins
            or pins.namespace_directory
            != self.verified_published_technical_authority.namespace_directory
            or pins.namespace_claim_sha256
            != self.verified_published_technical_authority.namespace_claim_sha256
            or pins.technical_authority_directory != authority.authority_directory
            or pins.technical_authority_artifact_root_sha256 != authority.artifact_root_sha256
            or pins.technical_authorization_sha256 != authority.technical_authorization_sha256
            or pins.published_technical_authority_lifecycle_binding_sha256 != binding_sha256
            or not binding_hash_matches
        ):
            raise LifecycleQualificationError(
                "published original-confirmatory readiness envelope is inconsistent"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "readiness": self.readiness.as_dict(),
            "verified_published_technical_authority": (
                self.verified_published_technical_authority.as_dict()
            ),
            "published_t0_pins": self.published_t0_pins.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _PublishedT0LifecycleContext:
    verified: VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1
    authority_binding: dict[str, Any]
    pins: PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _qualification_binding_payload(verified: Mapping[str, Any]) -> dict[str, Any]:
    stable_verified = {
        key: value for key, value in verified.items() if key != "producer_process_id"
    }
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "policy": "fresh_process_lifecycle_readiness_v1",
        "decision": "passed",
        "scientific_outcome": False,
        "project_completion_status_changed": False,
        **stable_verified,
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_FIELDS for character in value)
    )


def _json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LifecycleQualificationError(f"{role} is unavailable or invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise LifecycleQualificationError(f"{role} must be a JSON object: {path}")
    return dict(value)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise LifecycleQualificationError(f"rehearsal artifact is not a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": value.st_size,
        "sha256": sha256_file(path),
    }


def _require_exact_file_set(run_directory: Path, expected: frozenset[str]) -> None:
    actual: set[str] = set()
    for path in run_directory.rglob("*"):
        relative = path.relative_to(run_directory).as_posix()
        value = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise LifecycleQualificationError(
                f"sealed qualification run contains a non-regular or non-single-link path: {relative}"
            )
        actual.add(relative)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        added = sorted(actual.difference(expected))
        raise LifecycleQualificationError(
            f"sealed qualification run file set differs: missing={missing}, added={added}"
        )


def _require_resolved_config(run_directory: Path, expected: Mapping[str, Any]) -> None:
    resolved = load_config(run_directory / "resolved_config.yaml")
    if resolved != dict(expected) or config_sha256(resolved) != config_sha256(expected):
        raise LifecycleQualificationError(
            "resolved configuration differs from qualification intent"
        )


def _require_provenance(
    run_directory: Path,
    *,
    experiment_name: str,
    expected_config: Mapping[str, Any],
    expected_source_root_sha256: str,
    details: Mapping[str, Any],
) -> None:
    source_manifest_path = run_directory / "source_tree_manifest.json"
    source_manifest = _json_object(source_manifest_path, "run execution-source manifest")
    if source_manifest.get("root_sha256") != expected_source_root_sha256:
        raise LifecycleQualificationError("run execution-source root differs from authority")
    provenance = _json_object(run_directory / "run_provenance.json", "run provenance")
    expected = {
        "run_id": run_directory.name,
        "experiment_name": experiment_name,
        "config_sha256": config_sha256(expected_config),
        "split_seed": "",
        "model_seed": "",
        "corruption_seed": "",
        "source_tree": {
            "manifest": "source_tree_manifest.json",
            "manifest_sha256": sha256_file(source_manifest_path),
            "root_sha256": expected_source_root_sha256,
        },
        **dict(details),
    }
    if set(provenance) != {*expected, "started_at_utc"}:
        raise LifecycleQualificationError(
            "run provenance schema differs from qualification contract"
        )
    started_at = provenance.pop("started_at_utc", None)
    if not isinstance(started_at, str) or not started_at.strip() or provenance != expected:
        raise LifecycleQualificationError("run provenance bindings are stale or invalid")


def _require_single_link_file(path: Path, record: Mapping[str, Any]) -> None:
    value = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(value.st_mode)
        or path.is_symlink()
        or value.st_nlink != 1
        or value.st_size != record.get("size_bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise LifecycleQualificationError(
            f"physical rehearsal artifact differs or is not single-copy: {path}"
        )


def _verified_published_original_confirmatory_technical_authority(
    project_root: Path,
    authority_directory: Path,
    *,
    verify_live: bool,
) -> tuple[
    VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1,
    dict[str, Any],
]:
    if type(verify_live) is not bool:
        raise LifecycleQualificationError(
            "published original-confirmatory verification mode must be explicit"
        )
    try:
        verified = verify_published_original_confirmatory_technical_authority_v1(
            authority_directory,
            project_root=project_root,
            verify_live=verify_live,
        )
        lifecycle_binding = dict(verified.lifecycle_binding())
        technical_binding = dict(verified.authority.lifecycle_binding())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LifecycleQualificationError(
            "published original-confirmatory technical authority verification failed"
        ) from exc

    binding_sha256 = lifecycle_binding.get("binding_sha256")
    unsigned = {key: value for key, value in lifecycle_binding.items() if key != "binding_sha256"}
    technical_binding_sha256 = technical_binding.get("binding_sha256")
    technical_unsigned = {
        key: value for key, value in technical_binding.items() if key != "binding_sha256"
    }
    try:
        technical = authority_directory.resolve()
        namespace = technical.parent
        binding_hash_matches = _valid_sha256(
            binding_sha256
        ) and binding_sha256 == _canonical_sha256(unsigned)
        technical_binding_hash_matches = _valid_sha256(
            technical_binding_sha256
        ) and technical_binding_sha256 == _canonical_sha256(technical_unsigned)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise LifecycleQualificationError(
            "published original-confirmatory T0 lifecycle carrier is invalid"
        ) from exc
    expected_fields = {
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
    if (
        type(verified) is not VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1
        or set(lifecycle_binding) != expected_fields
        or verified.authority.authority_directory.resolve() != technical
        or verified.namespace_directory.resolve() != namespace
        or verified.namespace_directory != namespace
        or lifecycle_binding.get("schema_version") != 1
        or lifecycle_binding.get("namespace_directory") != str(namespace)
        or lifecycle_binding.get("namespace_claim_sha256") != verified.namespace_claim_sha256
        or not _valid_sha256(verified.namespace_claim_sha256)
        or lifecycle_binding.get("review_attempt_claim_sha256")
        != getattr(verified, "review_attempt_claim_sha256", None)
        or not _valid_sha256(getattr(verified, "review_attempt_claim_sha256", None))
        or not binding_hash_matches
        or lifecycle_binding.get("policy")
        != "published_original_confirmatory_technical_authority_lifecycle_binding_v1"
        or lifecycle_binding.get("technical_authority") != technical_binding
        or lifecycle_binding.get("automatic_retry_allowed") is not False
        or lifecycle_binding.get("adoption_allowed") is not False
        or lifecycle_binding.get("cleanup_allowed") is not False
        or not technical_binding_hash_matches
        or technical_binding.get("authority_directory") != str(technical)
        or technical_binding.get("artifact_root_sha256") != verified.authority.artifact_root_sha256
        or technical_binding.get("technical_authorization_sha256")
        != verified.authority.technical_authorization_sha256
        or technical_binding.get("policy")
        != "original_confirmatory_technical_authority_lifecycle_binding_v1"
        or technical_binding.get("primary_outcomes_inspected") is not True
        or technical_binding.get("confirmatory_outcomes_inspected") is not False
        or technical_binding.get("confirmatory_outcome_values_read") is not False
        or technical_binding.get("scientific_definition_changed") is not False
        or technical_binding.get("automatic_retry_allowed") is not False
    ):
        raise LifecycleQualificationError(
            "published original-confirmatory T0 lifecycle carrier is invalid"
        )
    return verified, lifecycle_binding


def _published_t0_authority_binding(
    verified: VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1,
    lifecycle_binding: Mapping[str, Any],
) -> dict[str, Any]:
    authority = verified.authority
    return {
        "schema_version": 3,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "authority_directory": str(authority.authority_directory),
        "chain_depth": authority.chain_depth,
        "artifact_root_sha256": authority.artifact_root_sha256,
        "sha256_manifest_sha256": authority.sha256_manifest_sha256,
        "execution_source_manifest_sha256": (authority.execution_source_manifest_sha256),
        "execution_source_root_sha256": authority.execution_source_root_sha256,
        "historical_parent_authority_directory": str(authority.parent_authority_directory),
        "historical_parent_chain_depth": authority.chain_depth - 1,
        "historical_parent_artifact_root_sha256": (authority.parent_artifact_root_sha256),
        "historical_parent_sha256_manifest_sha256": (authority.parent_sha256_manifest_sha256),
        "technical_authorization_sha256": authority.technical_authorization_sha256,
        "published_technical_authority_lifecycle_binding": dict(lifecycle_binding),
    }


def _authority_binding(
    project_root: Path,
    authority_directory: Path,
    *,
    _require_live_execution_source: bool = True,
) -> dict[str, Any]:
    marker = _json_object(authority_directory / IMMUTABLE_MARKER, "authority marker")
    if marker.get("authority_kind") == ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND:
        raise LifecycleQualificationError(
            "published original-confirmatory T0 is forbidden on the generic "
            "lifecycle path; use the strict published-T0 lifecycle entrypoint"
        )
    if marker.get("authority_kind") == "preregistration_amendment":
        amendment_verification = verify_preregistration_amendment(authority_directory)
        if not amendment_verification.valid:
            raise LifecycleQualificationError(
                f"registration amendment verification failed: {amendment_verification.errors}"
            )
        authority_kind = "preregistration_amendment"
        chain_depth = amendment_verification.chain_depth
        artifact_root = amendment_verification.artifact_root_sha256
        manifest_sha = amendment_verification.sha256_manifest_sha256
        amendment_evidence = _json_object(
            authority_directory / "amendment_evidence.json",
            "registration amendment evidence",
        )
        if amendment_evidence.get("amendment_purpose") in {
            RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE,
            RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        }:
            try:
                _require_sealed_effective_resource_bounded_confirmatory_authorization(
                    authority_directory
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise LifecycleQualificationError(
                    "resource-bounded registration authority is not its unique "
                    "effective execution leaf"
                ) from error
    elif marker.get("status") == "frozen":
        freeze_verification = verify_preregistration_freeze(authority_directory)
        if not freeze_verification.valid:
            raise LifecycleQualificationError(
                f"base preregistration verification failed: {freeze_verification.errors}"
            )
        authority_kind = "base_freeze"
        chain_depth = 0
        artifact_root = freeze_verification.artifact_root_sha256
        manifest_sha = marker.get("sha256_manifest_sha256")
    else:
        raise LifecycleQualificationError("unsupported registration authority marker")
    source_manifest_path = authority_directory / "source_tree_manifest.json"
    source_manifest = _json_object(source_manifest_path, "authority execution-source manifest")
    live_source = capture_source_tree(project_root)
    if _require_live_execution_source and source_manifest != live_source:
        raise LifecycleQualificationError(
            "live execution source differs from the verified registration authority"
        )
    source_root = source_manifest.get("root_sha256")
    if not all(
        _valid_sha256(value)
        for value in (
            artifact_root,
            manifest_sha,
            source_root,
            sha256_file(source_manifest_path),
        )
    ):
        raise LifecycleQualificationError("registration authority contains invalid hash bindings")
    return {
        "schema_version": 1,
        "authority_kind": authority_kind,
        "authority_directory": str(authority_directory),
        "chain_depth": chain_depth,
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "execution_source_manifest_sha256": sha256_file(source_manifest_path),
        "execution_source_root_sha256": source_root,
    }


def _dependency_lock_record(project_root: Path) -> dict[str, Any]:
    path = project_root / "uv.lock"
    if path.is_symlink() or not path.is_file():
        raise LifecycleQualificationError("exact dependency lock is unavailable")
    return {
        "relative_path": "uv.lock",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verifier_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "module": "histo_audit.workflows.lifecycle_qualification",
        "callable": "require_current_lifecycle_readiness",
        "module_sha256": sha256_file(path),
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
    }


def _exact_registry_row(
    run_directory: Path,
    *,
    expected_status: str,
    expected_experiment_name: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, str]:
    registry = run_directory.parent / "registry.csv"
    try:
        with registry.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise LifecycleQualificationError("run registry is unavailable") from exc
    matches = [row for row in rows if row.get("run_id") == run_directory.name]
    if len(matches) != 1:
        raise LifecycleQualificationError("run registry must contain exactly one matching row")
    row = matches[0]
    if (
        row.get("status") != expected_status
        or not row.get("run_path")
        or Path(str(row["run_path"])).resolve() != run_directory
        or (
            expected_experiment_name is not None
            and row.get("experiment_name") != expected_experiment_name
        )
        or (
            expected_config_sha256 is not None
            and row.get("config_sha256") != expected_config_sha256
        )
    ):
        raise LifecycleQualificationError("run registry row differs from the sealed run")
    return row


def _registry_git_state(state: Mapping[str, Any]) -> str:
    if not state.get("available"):
        return "unavailable"
    commit = state.get("commit") or "unborn"
    return f"{commit}+{'dirty' if state.get('dirty') else 'clean'}"


def _lifecycle_registry_row(
    tracker: RunTracker,
    *,
    status: Literal["completed", "failed"],
    completed_at: str,
) -> dict[str, Any]:
    seeds = tracker.config.get("seed")
    dataset = tracker.checksums.get("dataset")
    manifest = tracker.checksums.get("manifest")
    row = {
        "run_id": tracker.run_id,
        "experiment_name": tracker.experiment_name,
        "status": status,
        "started_at": tracker.timer.started_at_utc,
        "completed_at": completed_at,
        "config_sha256": tracker.config_hash,
        "git_state": _registry_git_state(tracker.git_state),
        "dataset_sha256": (dataset.get("sha256") if isinstance(dataset, Mapping) else "") or "",
        "manifest_sha256": (manifest.get("sha256") if isinstance(manifest, Mapping) else "") or "",
        "split_seed": seeds.get("split", "") if isinstance(seeds, Mapping) else "",
        "model_seed": seeds.get("model", "") if isinstance(seeds, Mapping) else "",
        "corruption_seed": (seeds.get("corruption", "") if isinstance(seeds, Mapping) else ""),
        "run_path": str(tracker.run_directory.resolve()),
    }
    if set(row) != set(REGISTRY_COLUMNS):
        raise LifecycleQualificationError("lifecycle recovery registry schema drifted")
    return row


def _registry_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(REGISTRY_COLUMNS):
                raise LifecycleQualificationError("run registry schema is invalid")
            return list(reader)
    except OSError as exc:
        raise LifecycleQualificationError("run registry is unavailable") from exc


def _integrity_registry_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LifecycleQualificationError("integrity registry is unavailable") from exc
    if content and not content.endswith("\n"):
        raise LifecycleQualificationError("integrity registry has a truncated final record")
    records: list[dict[str, Any]] = []
    for line in content.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleQualificationError("integrity registry contains invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise LifecycleQualificationError("integrity registry record must be an object")
        records.append(dict(value))
    return records


def _manifest_payload_records(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_directory.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(run_directory).as_posix()
        if relative in {ARTIFACT_MANIFEST_FILENAME, IMMUTABLE_MARKER}:
            continue
        if path.is_symlink():
            raise LifecycleQualificationError(
                f"interrupted lifecycle finalization contains a symbolic link: {relative}"
            )
        if path.is_file():
            records.append(_file_record(path, run_directory))
    return records


def _recover_interrupted_finalization(
    tracker: RunTracker,
    *,
    expected_status: Literal["completed", "failed"],
) -> bool:
    """Finish only an exact, already committed terminal lifecycle transaction.

    RunTracker publishes its integrity record and CSV row before its immutable marker.
    If one of those durable writes succeeds and the following operation raises, retrying
    ``fail()`` would append a second terminal history.  This recovery path independently
    verifies the manifest and the single append-only integrity record, fills only a
    missing exact CSV row, and finally publishes the marker.  With no integrity record,
    callers may still perform the ordinary failed finalization.
    """

    if tracker.experiment_name not in {
        LIFECYCLE_REHEARSAL_EXPERIMENT,
        LIFECYCLE_READINESS_EXPERIMENT,
    }:
        raise LifecycleQualificationError("refusing non-lifecycle finalization recovery")
    run = tracker.run_directory.resolve()
    marker_path = run / IMMUTABLE_MARKER
    if marker_path.exists():
        integrity = verify_run_integrity(run)
        _exact_registry_row(
            run,
            expected_status=expected_status,
            expected_experiment_name=tracker.experiment_name,
            expected_config_sha256=tracker.config_hash,
        )
        if not integrity.valid or not integrity.registry_record_present:
            raise LifecycleQualificationError(
                f"published lifecycle marker is not recoverable: {integrity.errors}"
            )
        return True

    status_path = run / STATUS_FILENAME
    manifest_path = run / ARTIFACT_MANIFEST_FILENAME
    registry_rows = [
        row for row in _registry_rows(tracker.registry_path) if row.get("run_id") == tracker.run_id
    ]
    integrity_registry = tracker.registry_path.parent / INTEGRITY_REGISTRY_FILENAME
    integrity_records = [
        record
        for record in _integrity_registry_records(integrity_registry)
        if record.get("run_id") == tracker.run_id
    ]
    if not integrity_records:
        if registry_rows:
            raise LifecycleQualificationError(
                "lifecycle finalization published a CSV row without an integrity record"
            )
        return False
    if len(integrity_records) != 1 or len(registry_rows) > 1:
        raise LifecycleQualificationError(
            "lifecycle finalization has duplicate terminal registry records"
        )
    if not status_path.is_file() or not manifest_path.is_file():
        raise LifecycleQualificationError(
            "lifecycle finalization registry exists without terminal run evidence"
        )
    status = _json_object(status_path, "interrupted lifecycle status")
    completed_at = status.get("completed_at_utc")
    if (
        status.get("status") != expected_status
        or status.get("run_id") != tracker.run_id
        or status.get("experiment_name") != tracker.experiment_name
        or not isinstance(completed_at, str)
        or not completed_at.strip()
    ):
        raise LifecycleQualificationError("interrupted lifecycle status is inconsistent")
    manifest = _json_object(manifest_path, "interrupted lifecycle artifact manifest")
    records = _manifest_payload_records(run)
    expected_manifest = {
        "schema_version": 1,
        "run_id": tracker.run_id,
        "status": expected_status,
        "created_at_utc": manifest.get("created_at_utc"),
        "artifact_count": len(records),
        "artifact_root_sha256": _canonical_sha256(records),
        "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
        "excluded_paths": sorted({ARTIFACT_MANIFEST_FILENAME, IMMUTABLE_MARKER}),
        "artifacts": records,
    }
    if (
        not isinstance(manifest.get("created_at_utc"), str)
        or not str(manifest["created_at_utc"]).strip()
        or manifest != expected_manifest
    ):
        raise LifecycleQualificationError("interrupted lifecycle artifact manifest is inconsistent")
    integrity_record = integrity_records[0]
    expected_integrity = {
        "run_id": tracker.run_id,
        "status": expected_status,
        "sealed_at_utc": integrity_record.get("sealed_at_utc"),
        "run_path": str(run),
        "artifact_count": len(records),
        "artifact_root_sha256": expected_manifest["artifact_root_sha256"],
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }
    if (
        not isinstance(integrity_record.get("sealed_at_utc"), str)
        or not str(integrity_record["sealed_at_utc"]).strip()
        or integrity_record != expected_integrity
    ):
        raise LifecycleQualificationError("interrupted lifecycle integrity record is inconsistent")
    expected_registry = _lifecycle_registry_row(
        tracker,
        status=expected_status,
        completed_at=completed_at,
    )
    if registry_rows:
        if registry_rows[0] != {key: str(value) for key, value in expected_registry.items()}:
            raise LifecycleQualificationError(
                "interrupted lifecycle CSV registry row is inconsistent"
            )
    else:
        append_registry_row(tracker.registry_path, expected_registry)
    if marker_path.exists():
        raise LifecycleQualificationError("lifecycle immutable marker appeared during recovery")
    atomic_write_json(
        marker_path,
        {
            **integrity_record,
            "integrity_registry": str(integrity_registry.resolve()),
        },
    )
    integrity = verify_run_integrity(run)
    _exact_registry_row(
        run,
        expected_status=expected_status,
        expected_experiment_name=tracker.experiment_name,
        expected_config_sha256=tracker.config_hash,
    )
    if not integrity.valid or not integrity.registry_record_present:
        raise LifecycleQualificationError(
            f"recovered lifecycle finalization failed integrity: {integrity.errors}"
        )
    return True


def _finalize_failed_lifecycle_run(tracker: RunTracker, error: BaseException) -> None:
    if tracker.finalized:
        return
    try:
        tracker.fail(error)
    except BaseException:
        if not _recover_interrupted_finalization(tracker, expected_status="failed"):
            raise


def _retry_binding(run_root: Path, retry_of_run_id: str | None) -> dict[str, Any] | None:
    if retry_of_run_id is None:
        return None
    if Path(retry_of_run_id).name != retry_of_run_id or not retry_of_run_id.strip():
        raise LifecycleQualificationError("retry_of_run_id must be one safe run ID")
    predecessor = (run_root / retry_of_run_id).resolve()
    if predecessor.parent != run_root or not predecessor.is_dir():
        raise LifecycleQualificationError("rehearsal retry predecessor is unavailable")
    integrity = verify_run_integrity(predecessor)
    status = _json_object(predecessor / STATUS_FILENAME, "retry predecessor status")
    _exact_registry_row(
        predecessor,
        expected_status="failed",
        expected_experiment_name=LIFECYCLE_REHEARSAL_EXPERIMENT,
    )
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or status.get("status") != "failed"
    ):
        raise LifecycleQualificationError("rehearsal retry predecessor is not a sealed failed run")
    payload = {
        "run_id": retry_of_run_id,
        "run_directory": str(predecessor),
        "artifact_root_sha256": integrity.expected_root_sha256,
    }
    return {**payload, "lineage_binding_sha256": _canonical_sha256(payload)}


def _write_synthetic_staging(
    staging: Path,
    *,
    authority: Mapping[str, Any],
    dependency_lock: Mapping[str, Any],
    retry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_sha = config_sha256(_REHEARSAL_CONFIG)
    plan_sha = _canonical_sha256(_REHEARSAL_PLAN)
    authority_sha = _canonical_sha256(authority)
    verifier = _verifier_record()
    checkpoint_path = staging / "checkpoint.pt"
    tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    checkpoint_payload = {
        "schema_version": 1,
        "kind": "lifecycle_rehearsal_checkpoint",
        "study_outcome_eligible": False,
        "seed": _REHEARSAL_CONFIG["seed"],
        "configuration_sha256": config_sha,
        "plan_sha256": plan_sha,
        "weights": tensor,
    }
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    safe_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(safe_checkpoint, Mapping) or not torch.equal(
        cast(torch.Tensor, safe_checkpoint.get("weights")), tensor
    ):
        raise LifecycleQualificationError("synthetic checkpoint safe readback failed")
    checkpoint_record = _file_record(checkpoint_path, staging)
    atomic_write_json(
        staging / "checkpoint_attestation.json",
        {
            "schema_version": 1,
            "policy": "weights_only_checkpoint_readback_v1",
            "checkpoint": checkpoint_record,
            "weights_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            "verification_status": "passed",
        },
    )
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    statistics = {
        "schema_version": 1,
        "group_count": 2,
        "sample_count": 10,
        "values": values,
        "mean": sum(values) / len(values),
    }
    atomic_write_json(staging / "statistics.json", statistics)
    statistics_record = _file_record(staging / "statistics.json", staging)
    atomic_write_json(
        staging / "statistics_attestation.json",
        {
            "schema_version": 1,
            "policy": "deterministic_statistics_recomputation_v1",
            "statistics": statistics_record,
            "recomputed_mean": sum(values) / len(values),
            "verification_status": "passed",
        },
    )
    observed = [0, 2, 2, 3, 4]
    reviewed = [False, True, False, False, False]
    reference = [0, 1, 2, 3, 4]
    restored = [reference[i] if reviewed[i] else observed[i] for i in range(len(observed))]
    restoration = {
        "schema_version": 1,
        "observed_labels": observed,
        "reviewed_mask": reviewed,
        "reference_labels": reference,
        "restored_labels": restored,
        "restored_count": sum(
            before != after for before, after in zip(observed, restored, strict=True)
        ),
    }
    atomic_write_json(staging / "restoration.json", restoration)
    restoration_record = _file_record(staging / "restoration.json", staging)
    atomic_write_json(
        staging / "restoration_attestation.json",
        {
            "schema_version": 1,
            "policy": "reviewed_only_restoration_replay_v1",
            "restoration": restoration_record,
            "restored_count": restoration["restored_count"],
            "verification_status": "passed",
        },
    )
    base_names = (
        "checkpoint.pt",
        "checkpoint_attestation.json",
        "statistics.json",
        "statistics_attestation.json",
        "restoration.json",
        "restoration_attestation.json",
    )
    records = [_file_record(staging / name, staging) for name in base_names]
    completion = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "qualification_kind": "operational_lifecycle_readiness",
        "scientific_outcome": False,
        "scientific_completion_stage_claimed": False,
        "fresh_process_verification_required": True,
        "producer_process_id": os.getpid(),
        "configuration_semantic_sha256": config_sha,
        "plan_sha256": plan_sha,
        "authority_binding_sha256": authority_sha,
        "execution_source_root_sha256": authority["execution_source_root_sha256"],
        "dependency_lock": dict(dependency_lock),
        "verifier": verifier,
        "retry_of_run_id": retry["run_id"] if retry is not None else None,
        "retry_lineage_binding_sha256": (
            retry["lineage_binding_sha256"] if retry is not None else None
        ),
        "required_artifacts": records,
        "required_attestations": [
            "checkpoint_attestation.json",
            "statistics_attestation.json",
            "restoration_attestation.json",
        ],
    }
    atomic_write_json(staging / LIFECYCLE_COMPLETION_FILENAME, completion)
    records.append(_file_record(staging / LIFECYCLE_COMPLETION_FILENAME, staging))
    return completion, records


def execute_lifecycle_rehearsal(
    *,
    project_root: str | Path,
    authority_directory: str | Path,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
    retry_of_run_id: str | None = None,
) -> LifecycleRehearsalResult:
    """Execute and seal one non-scientific production-boundary rehearsal."""

    root = Path(project_root).resolve()
    authority_dir = Path(authority_directory).resolve()
    authority = _authority_binding(root, authority_dir)
    return _execute_lifecycle_rehearsal_with_authority(
        project_root=root,
        authority=authority,
        runs_root=runs_root,
        run_id=run_id,
        retry_of_run_id=retry_of_run_id,
    )


def _execute_lifecycle_rehearsal_with_authority(
    *,
    project_root: Path,
    authority: Mapping[str, Any],
    runs_root: str | Path | None,
    run_id: str | None,
    retry_of_run_id: str | None,
) -> LifecycleRehearsalResult:
    root = project_root
    authority = dict(authority)
    dependency_lock = _dependency_lock_record(root)
    run_root = Path(runs_root).resolve() if runs_root is not None else root / "artifacts" / "runs"
    retry = _retry_binding(run_root, retry_of_run_id)
    tracker = RunTracker.start(
        experiment_name=LIFECYCLE_REHEARSAL_EXPERIMENT,
        config=_REHEARSAL_CONFIG,
        project_root=root,
        runs_root=run_root,
        run_id=run_id,
        environment={
            "qualification_kind": "operational_lifecycle_readiness",
            "scientific_outcome": False,
        },
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{tracker.run_id}.rehearsal-", dir=run_root.parent))
    try:
        if tracker.source_tree.get("root_sha256") != authority["execution_source_root_sha256"]:
            raise LifecycleQualificationError(
                "source changed between authority gate and RunTracker"
            )
        completion, records = _write_synthetic_staging(
            staging,
            authority=authority,
            dependency_lock=dependency_lock,
            retry=retry,
        )
        published: list[dict[str, Any]] = []
        with anchored_physical_copy_session(staging, tracker.run_directory) as session:
            for record in records:
                result = session.copy_file_no_overwrite(
                    str(record["path"]),
                    expected_size_bytes=int(record["size_bytes"]),
                    expected_sha256=str(record["sha256"]),
                )
                destination_nlink = result.path.stat(follow_symlinks=False).st_nlink
                if destination_nlink != 1:
                    raise LifecycleQualificationError(
                        f"physical rehearsal publication has nlink={destination_nlink}: {result.path}"
                    )
                published.append(
                    {
                        **record,
                        "destination": str(result.path),
                        "destination_nlink": destination_nlink,
                    }
                )
            session.assert_roots_current()
        for record in records:
            _require_single_link_file(tracker.run_directory / str(record["path"]), record)
        tracker.write_json(LIFECYCLE_AUTHORITY_FILENAME, authority)
        tracker.write_json(LIFECYCLE_PLAN_FILENAME, _REHEARSAL_PLAN)
        tracker.write_json(
            LIFECYCLE_PUBLICATION_FILENAME,
            {
                "schema_version": 1,
                "policy": _REHEARSAL_CONFIG["physical_publication"],
                "physical_copy_verified": True,
                "published": published,
            },
        )
        tracker.write_provenance(
            qualification_kind="operational_lifecycle_readiness",
            scientific_outcome=False,
            producer_process_id=os.getpid(),
            authority_binding_sha256=_canonical_sha256(authority),
            lifecycle_plan_sha256=_canonical_sha256(_REHEARSAL_PLAN),
            dependency_lock_sha256=dependency_lock["sha256"],
            verifier_implementation=_verifier_record(),
            retry_of_run_id=retry_of_run_id,
            retry_lineage_binding_sha256=(
                retry["lineage_binding_sha256"] if retry is not None else None
            ),
            fresh_process_verification_required=True,
        )
        tracker.complete()
    except BaseException as error:
        if not _recover_interrupted_finalization(tracker, expected_status="completed"):
            _finalize_failed_lifecycle_run(tracker, error)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    integrity = verify_run_integrity(tracker.run_directory)
    _exact_registry_row(
        tracker.run_directory,
        expected_status="completed",
        expected_experiment_name=LIFECYCLE_REHEARSAL_EXPERIMENT,
        expected_config_sha256=config_sha256(_REHEARSAL_CONFIG),
    )
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != tracker.run_id
    ):
        raise LifecycleQualificationError(
            f"sealed lifecycle rehearsal failed integrity: {integrity.errors}"
        )
    completion_path = tracker.run_directory / LIFECYCLE_COMPLETION_FILENAME
    return LifecycleRehearsalResult(
        run_directory=tracker.run_directory,
        run_id=tracker.run_id,
        artifact_root_sha256=str(integrity.expected_root_sha256),
        completion_evidence_sha256=sha256_file(completion_path),
        config_semantic_sha256=str(completion["configuration_semantic_sha256"]),
        plan_sha256=str(completion["plan_sha256"]),
        authority_binding_sha256=str(completion["authority_binding_sha256"]),
    )


def _verify_rehearsal_read_only(
    *,
    project_root: Path,
    authority: Mapping[str, Any],
    rehearsal_run_directory: Path,
    enforce_fresh_process: bool = False,
) -> dict[str, Any]:
    authority = dict(authority)
    dependency_lock = _dependency_lock_record(project_root)
    run = rehearsal_run_directory.resolve()
    integrity = verify_run_integrity(run)
    _exact_registry_row(
        run,
        expected_status="completed",
        expected_experiment_name=LIFECYCLE_REHEARSAL_EXPERIMENT,
        expected_config_sha256=config_sha256(_REHEARSAL_CONFIG),
    )
    if not integrity.valid or not integrity.registry_record_present or integrity.run_id != run.name:
        raise LifecycleQualificationError(f"rehearsal integrity failed: {integrity.errors}")
    try:
        require_run_stage_eligible(run, integrity=integrity)
    except (OSError, ValueError, TypeError) as exc:
        raise LifecycleQualificationError(
            "rehearsal stage eligibility is withdrawn or invalid"
        ) from exc
    _require_exact_file_set(run, _REHEARSAL_FILES)
    status = _json_object(run / STATUS_FILENAME, "rehearsal status")
    if (
        status.get("status") != "completed"
        or status.get("experiment_name") != LIFECYCLE_REHEARSAL_EXPERIMENT
    ):
        raise LifecycleQualificationError("sealed run is not a completed lifecycle rehearsal")
    completion = _json_object(run / LIFECYCLE_COMPLETION_FILENAME, "rehearsal completion")
    producer_process_id = completion.get("producer_process_id")
    expected_completion_keys = {
        "schema_version",
        "qualification_kind",
        "scientific_outcome",
        "scientific_completion_stage_claimed",
        "fresh_process_verification_required",
        "producer_process_id",
        "configuration_semantic_sha256",
        "plan_sha256",
        "authority_binding_sha256",
        "execution_source_root_sha256",
        "dependency_lock",
        "verifier",
        "retry_of_run_id",
        "retry_lineage_binding_sha256",
        "required_artifacts",
        "required_attestations",
    }
    if (
        set(completion) != expected_completion_keys
        or completion.get("schema_version") != LIFECYCLE_SCHEMA_VERSION
        or completion.get("qualification_kind") != "operational_lifecycle_readiness"
        or completion.get("scientific_outcome") is not False
        or completion.get("scientific_completion_stage_claimed") is not False
        or completion.get("fresh_process_verification_required") is not True
        or type(producer_process_id) is not int
        or producer_process_id <= 0
        or (enforce_fresh_process and producer_process_id == os.getpid())
        or completion.get("configuration_semantic_sha256") != config_sha256(_REHEARSAL_CONFIG)
        or completion.get("plan_sha256") != _canonical_sha256(_REHEARSAL_PLAN)
        or completion.get("authority_binding_sha256") != _canonical_sha256(authority)
        or completion.get("execution_source_root_sha256")
        != authority["execution_source_root_sha256"]
        or completion.get("dependency_lock") != dependency_lock
        or completion.get("verifier") != _verifier_record()
    ):
        raise LifecycleQualificationError("rehearsal completion bindings are stale or invalid")
    authority_evidence = _json_object(run / LIFECYCLE_AUTHORITY_FILENAME, "authority binding")
    plan_evidence = _json_object(run / LIFECYCLE_PLAN_FILENAME, "lifecycle plan")
    if authority_evidence != authority or plan_evidence != _REHEARSAL_PLAN:
        raise LifecycleQualificationError("authority or lifecycle plan evidence is stale")
    retry_of_run_id = completion.get("retry_of_run_id")
    retry_lineage_sha256 = completion.get("retry_lineage_binding_sha256")
    if retry_of_run_id is None:
        if retry_lineage_sha256 is not None:
            raise LifecycleQualificationError(
                "rehearsal retry lineage exists without a predecessor"
            )
    elif not isinstance(retry_of_run_id, str):
        raise LifecycleQualificationError("rehearsal retry predecessor ID is malformed")
    else:
        retry = _retry_binding(run.parent, retry_of_run_id)
        if retry is None or retry_lineage_sha256 != retry["lineage_binding_sha256"]:
            raise LifecycleQualificationError("rehearsal retry lineage binding is invalid")
    _require_resolved_config(run, _REHEARSAL_CONFIG)
    _require_provenance(
        run,
        experiment_name=LIFECYCLE_REHEARSAL_EXPERIMENT,
        expected_config=_REHEARSAL_CONFIG,
        expected_source_root_sha256=str(authority["execution_source_root_sha256"]),
        details={
            "qualification_kind": "operational_lifecycle_readiness",
            "scientific_outcome": False,
            "producer_process_id": producer_process_id,
            "authority_binding_sha256": _canonical_sha256(authority),
            "lifecycle_plan_sha256": _canonical_sha256(_REHEARSAL_PLAN),
            "dependency_lock_sha256": dependency_lock["sha256"],
            "verifier_implementation": _verifier_record(),
            "retry_of_run_id": retry_of_run_id,
            "retry_lineage_binding_sha256": retry_lineage_sha256,
            "fresh_process_verification_required": True,
        },
    )
    raw_records = completion.get("required_artifacts")
    if not isinstance(raw_records, list) or not raw_records:
        raise LifecycleQualificationError("rehearsal completion lacks required artifacts")
    records = [dict(value) for value in raw_records if isinstance(value, Mapping)]
    expected_required_paths = {
        "checkpoint.pt",
        "checkpoint_attestation.json",
        "statistics.json",
        "statistics_attestation.json",
        "restoration.json",
        "restoration_attestation.json",
    }
    if (
        len(records) != len(raw_records)
        or len({str(value.get("path")) for value in records}) != len(records)
        or {str(value.get("path")) for value in records} != expected_required_paths
    ):
        raise LifecycleQualificationError("rehearsal artifact records are malformed or duplicate")
    expected_attestations = {
        "checkpoint_attestation.json",
        "statistics_attestation.json",
        "restoration_attestation.json",
    }
    raw_attestations = completion.get("required_attestations")
    if (
        not isinstance(raw_attestations, list)
        or len(raw_attestations) != len(expected_attestations)
        or set(raw_attestations) != expected_attestations
    ):
        raise LifecycleQualificationError("rehearsal attestation set is missing or stale")
    for record in records:
        relative = str(record.get("path", ""))
        path = run / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise LifecycleQualificationError("rehearsal artifact path is unsafe")
        _require_single_link_file(path, record)
    checkpoint = run / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema_version",
            "kind",
            "study_outcome_eligible",
            "seed",
            "configuration_sha256",
            "plan_sha256",
            "weights",
        }
        or payload.get("schema_version") != 1
        or payload.get("kind") != "lifecycle_rehearsal_checkpoint"
        or payload.get("study_outcome_eligible") is not False
        or payload.get("configuration_sha256") != config_sha256(_REHEARSAL_CONFIG)
        or payload.get("plan_sha256") != _canonical_sha256(_REHEARSAL_PLAN)
    ):
        raise LifecycleQualificationError("checkpoint semantic readback failed")
    tensor = payload.get("weights")
    if not isinstance(tensor, torch.Tensor):
        raise LifecycleQualificationError("checkpoint lacks its deterministic tensor")
    checkpoint_attestation = _json_object(
        run / "checkpoint_attestation.json", "checkpoint attestation"
    )
    if (
        set(checkpoint_attestation)
        != {
            "schema_version",
            "policy",
            "checkpoint",
            "weights_sha256",
            "verification_status",
        }
        or checkpoint_attestation.get("schema_version") != 1
        or checkpoint_attestation.get("policy") != "weights_only_checkpoint_readback_v1"
        or checkpoint_attestation.get("verification_status") != "passed"
        or checkpoint_attestation.get("checkpoint") != _file_record(checkpoint, run)
        or checkpoint_attestation.get("weights_sha256")
        != hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    ):
        raise LifecycleQualificationError("checkpoint attestation verification failed")
    statistics = _json_object(run / "statistics.json", "statistics evidence")
    values = statistics.get("values")
    statistics_attestation = _json_object(
        run / "statistics_attestation.json", "statistics attestation"
    )
    if (
        set(statistics) != {"schema_version", "group_count", "sample_count", "values", "mean"}
        or statistics.get("schema_version") != 1
        or statistics.get("group_count") != 2
        or statistics.get("sample_count") != 10
        or not isinstance(values, list)
        or not values
        or statistics.get("mean") != sum(float(value) for value in values) / len(values)
        or set(statistics_attestation)
        != {
            "schema_version",
            "policy",
            "statistics",
            "recomputed_mean",
            "verification_status",
        }
        or statistics_attestation.get("schema_version") != 1
        or statistics_attestation.get("policy") != "deterministic_statistics_recomputation_v1"
        or statistics_attestation.get("verification_status") != "passed"
        or statistics_attestation.get("statistics") != _file_record(run / "statistics.json", run)
        or statistics_attestation.get("recomputed_mean") != statistics.get("mean")
    ):
        raise LifecycleQualificationError("statistics attestation verification failed")
    restoration = _json_object(run / "restoration.json", "restoration evidence")
    observed = restoration.get("observed_labels")
    reviewed = restoration.get("reviewed_mask")
    reference = restoration.get("reference_labels")
    restored = restoration.get("restored_labels")
    if not all(isinstance(value, list) for value in (observed, reviewed, reference, restored)):
        raise LifecycleQualificationError("restoration evidence arrays are malformed")
    observed_values = cast(list[Any], observed)
    reviewed_values = cast(list[Any], reviewed)
    reference_values = cast(list[Any], reference)
    restored_values = cast(list[Any], restored)
    if not (
        len(observed_values)
        == len(reviewed_values)
        == len(reference_values)
        == len(restored_values)
    ):
        raise LifecycleQualificationError("restoration evidence array lengths differ")
    expected_restored = [
        reference_values[index] if reviewed_values[index] else observed_values[index]
        for index in range(len(observed_values))
    ]
    restoration_attestation = _json_object(
        run / "restoration_attestation.json", "restoration attestation"
    )
    if (
        set(restoration)
        != {
            "schema_version",
            "observed_labels",
            "reviewed_mask",
            "reference_labels",
            "restored_labels",
            "restored_count",
        }
        or restoration.get("schema_version") != 1
        or restored != expected_restored
        or restoration.get("restored_count")
        != sum(
            before != after for before, after in zip(observed_values, restored_values, strict=True)
        )
        or set(restoration_attestation)
        != {
            "schema_version",
            "policy",
            "restoration",
            "restored_count",
            "verification_status",
        }
        or restoration_attestation.get("schema_version") != 1
        or restoration_attestation.get("policy") != "reviewed_only_restoration_replay_v1"
        or restoration_attestation.get("verification_status") != "passed"
        or restoration_attestation.get("restoration") != _file_record(run / "restoration.json", run)
        or restoration_attestation.get("restored_count") != restoration.get("restored_count")
    ):
        raise LifecycleQualificationError("restoration attestation verification failed")
    publication = _json_object(run / LIFECYCLE_PUBLICATION_FILENAME, "publication receipt")
    published_records = [
        *records,
        _file_record(run / LIFECYCLE_COMPLETION_FILENAME, run),
    ]
    expected_publication = {
        "schema_version": 1,
        "policy": _REHEARSAL_CONFIG["physical_publication"],
        "physical_copy_verified": True,
        "published": [
            {
                **record,
                "destination": str(run / str(record["path"])),
                "destination_nlink": 1,
            }
            for record in published_records
        ],
    }
    if publication != expected_publication:
        raise LifecycleQualificationError("physical publication receipt is invalid")
    return {
        "rehearsal_run_id": run.name,
        "rehearsal_run_directory": str(run),
        "rehearsal_artifact_root_sha256": integrity.expected_root_sha256,
        "rehearsal_artifact_manifest_sha256": sha256_file(run / ARTIFACT_MANIFEST_FILENAME),
        "completion_evidence_sha256": sha256_file(run / LIFECYCLE_COMPLETION_FILENAME),
        "configuration_semantic_sha256": config_sha256(_REHEARSAL_CONFIG),
        "plan_sha256": _canonical_sha256(_REHEARSAL_PLAN),
        "authority": authority,
        "dependency_lock": dependency_lock,
        "verifier": _verifier_record(),
        "producer_process_id": producer_process_id,
        "retry_of_run_id": retry_of_run_id,
        "retry_lineage_binding_sha256": retry_lineage_sha256,
    }


def verify_lifecycle_rehearsal_fresh_process(
    *,
    project_root: str | Path,
    authority_directory: str | Path,
    rehearsal_run_directory: str | Path,
    runs_root: str | Path | None = None,
) -> LifecycleReadinessResult:
    """Reopen a sealed rehearsal and publish one separately sealed readiness run."""

    root = Path(project_root).resolve()
    authority_directory_path = Path(authority_directory).resolve()
    authority = _authority_binding(root, authority_directory_path)
    return _verify_lifecycle_rehearsal_fresh_process_with_authority(
        project_root=root,
        authority=authority,
        rehearsal_run_directory=Path(rehearsal_run_directory).resolve(),
        runs_root=runs_root,
    )


def _verify_lifecycle_rehearsal_fresh_process_with_authority(
    *,
    project_root: Path,
    authority: Mapping[str, Any],
    rehearsal_run_directory: Path,
    runs_root: str | Path | None,
) -> LifecycleReadinessResult:
    root = project_root
    authority = dict(authority)
    rehearsal = Path(rehearsal_run_directory).resolve()
    verified = _verify_rehearsal_read_only(
        project_root=root,
        authority=authority,
        rehearsal_run_directory=rehearsal,
        enforce_fresh_process=True,
    )
    fresh_verifier_process_id = os.getpid()
    qualification_binding = _qualification_binding_payload(verified)
    qualification_sha256 = _canonical_sha256(qualification_binding)
    readiness_record = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "policy": "fresh_process_lifecycle_readiness_v1",
        "decision": "passed",
        "scientific_outcome": False,
        "project_completion_status_changed": False,
        "fresh_verifier_process_id": fresh_verifier_process_id,
        **verified,
        "qualification_binding_sha256": qualification_sha256,
    }
    readiness_record_sha256 = _canonical_sha256(readiness_record)
    run_root = Path(runs_root).resolve() if runs_root is not None else root / "artifacts" / "runs"
    rehearsal_id_sha256 = hashlib.sha256(rehearsal.name.encode("utf-8")).hexdigest()
    deterministic_id = f"lifecycle_ready_{rehearsal_id_sha256[:12]}_{qualification_sha256[:16]}"
    readiness_config = {
        "schema_version": 1,
        "experiment_name": LIFECYCLE_READINESS_EXPERIMENT,
        "rehearsal_run_id": rehearsal.name,
        "qualification_binding_sha256": qualification_sha256,
        "readiness_record_sha256": readiness_record_sha256,
        "fresh_verifier_process_id": fresh_verifier_process_id,
        "scientific_outcome": False,
    }
    tracker = RunTracker.start(
        experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
        config=readiness_config,
        project_root=root,
        runs_root=run_root,
        run_id=deterministic_id,
        environment={
            "fresh_process_verifier": True,
            "fresh_verifier_process_id": fresh_verifier_process_id,
            "scientific_outcome": False,
        },
    )
    try:
        if (
            tracker.source_tree.get("root_sha256")
            != verified["authority"]["execution_source_root_sha256"]
        ):
            raise LifecycleQualificationError(
                "source changed between fresh verification and readiness RunTracker"
            )
        tracker.write_json(
            LIFECYCLE_READINESS_FILENAME,
            {
                **readiness_record,
                "readiness_record_sha256": readiness_record_sha256,
            },
        )
        tracker.write_provenance(
            fresh_process_verifier=True,
            fresh_verifier_process_id=fresh_verifier_process_id,
            decision="passed",
            scientific_outcome=False,
            rehearsal_run_id=rehearsal.name,
            qualification_binding_sha256=qualification_sha256,
            readiness_record_sha256=readiness_record_sha256,
        )
        tracker.complete()
    except BaseException as error:
        if not _recover_interrupted_finalization(tracker, expected_status="completed"):
            _finalize_failed_lifecycle_run(tracker, error)
            raise
    integrity = verify_run_integrity(tracker.run_directory)
    _exact_registry_row(
        tracker.run_directory,
        expected_status="completed",
        expected_experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
        expected_config_sha256=config_sha256(readiness_config),
    )
    if not integrity.valid or not integrity.registry_record_present:
        raise LifecycleQualificationError(
            f"readiness verification run failed integrity: {integrity.errors}"
        )
    _require_exact_file_set(tracker.run_directory, _READINESS_FILES)
    _require_resolved_config(tracker.run_directory, readiness_config)
    _require_provenance(
        tracker.run_directory,
        experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
        expected_config=readiness_config,
        expected_source_root_sha256=str(verified["authority"]["execution_source_root_sha256"]),
        details={
            "fresh_process_verifier": True,
            "fresh_verifier_process_id": fresh_verifier_process_id,
            "decision": "passed",
            "scientific_outcome": False,
            "rehearsal_run_id": rehearsal.name,
            "qualification_binding_sha256": qualification_sha256,
            "readiness_record_sha256": readiness_record_sha256,
        },
    )
    typed_verification = build_lifecycle_qualification_attestation_verification(
        tracker.run_directory,
        integrity=integrity,
    )
    appended_attestation = attest_lifecycle_run_qualification(
        tracker.run_directory,
        verification=typed_verification,
    )
    required_attestation = require_lifecycle_run_qualified(
        tracker.run_directory,
        integrity=integrity,
    )
    if appended_attestation != required_attestation:
        raise LifecycleQualificationError(
            "lifecycle qualification attestation failed exact readback"
        )
    final = _require_current_lifecycle_readiness_with_authority(
        project_root=root,
        authority=authority,
        readiness_run_directory=tracker.run_directory,
    )
    if (
        not final.valid
        or final.qualification_binding_sha256 != qualification_sha256
        or final.readiness_record_sha256 != readiness_record_sha256
    ):
        raise LifecycleQualificationError(f"fresh readiness readback failed: {final.errors}")
    return LifecycleReadinessResult(
        readiness_run_directory=tracker.run_directory,
        readiness_run_id=tracker.run_id,
        rehearsal_run_directory=rehearsal,
        artifact_root_sha256=str(integrity.expected_root_sha256),
        qualification_binding_sha256=qualification_sha256,
        readiness_record_sha256=readiness_record_sha256,
    )


def require_current_lifecycle_readiness(
    *,
    project_root: str | Path,
    authority_directory: str | Path,
    readiness_run_directory: str | Path,
) -> LifecycleReadinessVerification:
    """Fail closed unless a fresh sealed readiness run matches current execution."""

    root = Path(project_root).resolve()
    authority_directory_path = Path(authority_directory).resolve()
    try:
        authority = _authority_binding(root, authority_directory_path)
    except (OSError, ValueError, TypeError, LifecycleQualificationError) as exc:
        raise LifecycleQualificationError(
            f"lifecycle readiness gate failed closed: {type(exc).__name__}: {exc}"
        ) from exc
    return _require_current_lifecycle_readiness_with_authority(
        project_root=root,
        authority=authority,
        readiness_run_directory=Path(readiness_run_directory).resolve(),
    )


def _require_current_lifecycle_readiness_with_authority(
    *,
    project_root: Path,
    authority: Mapping[str, Any],
    readiness_run_directory: Path,
) -> LifecycleReadinessVerification:
    root = project_root
    authority = dict(authority)
    readiness = Path(readiness_run_directory).resolve()
    rehearsal: Path | None = None
    errors: list[str] = []
    qualification_sha256: str | None = None
    readiness_record_sha256: str | None = None
    try:
        integrity = verify_run_integrity(readiness)
        _exact_registry_row(
            readiness,
            expected_status="completed",
            expected_experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
        )
        if not integrity.valid or not integrity.registry_record_present:
            raise LifecycleQualificationError(f"readiness run integrity failed: {integrity.errors}")
        try:
            require_run_stage_eligible(readiness, integrity=integrity)
        except (OSError, ValueError, TypeError) as exc:
            raise LifecycleQualificationError(
                "readiness stage eligibility is withdrawn or invalid"
            ) from exc
        try:
            require_lifecycle_run_qualified(readiness, integrity=integrity)
        except (OSError, ValueError, TypeError) as exc:
            raise LifecycleQualificationError(
                "readiness lifecycle qualification attestation is missing or invalid"
            ) from exc
        _require_exact_file_set(readiness, _READINESS_FILES)
        status = _json_object(readiness / STATUS_FILENAME, "readiness run status")
        if (
            status.get("status") != "completed"
            or status.get("experiment_name") != LIFECYCLE_READINESS_EXPERIMENT
        ):
            raise LifecycleQualificationError("sealed run is not a lifecycle readiness verifier")
        evidence = _json_object(
            readiness / LIFECYCLE_READINESS_FILENAME, "lifecycle readiness evidence"
        )
        readiness_record_sha_value = evidence.pop("readiness_record_sha256", None)
        if (
            not _valid_sha256(readiness_record_sha_value)
            or _canonical_sha256(evidence) != readiness_record_sha_value
        ):
            raise LifecycleQualificationError("readiness record SHA-256 is invalid")
        readiness_record_sha256 = str(readiness_record_sha_value)
        qualification_sha_value = evidence.get("qualification_binding_sha256")
        if not _valid_sha256(qualification_sha_value):
            raise LifecycleQualificationError("qualification binding SHA-256 is invalid")
        qualification_sha256 = str(qualification_sha_value)
        if (
            evidence.get("decision") != "passed"
            or evidence.get("scientific_outcome") is not False
            or evidence.get("project_completion_status_changed") is not False
        ):
            raise LifecycleQualificationError(
                "readiness evidence lacks a positive non-scientific decision"
            )
        fresh_verifier_process_id = evidence.get("fresh_verifier_process_id")
        if type(fresh_verifier_process_id) is not int or fresh_verifier_process_id <= 0:
            raise LifecycleQualificationError("readiness evidence lacks a fresh verifier process")
        rehearsal_value = evidence.get("rehearsal_run_directory")
        if not isinstance(rehearsal_value, str):
            raise LifecycleQualificationError("readiness evidence lacks its rehearsal path")
        rehearsal = Path(rehearsal_value).resolve()
        rehearsal_id_sha256 = hashlib.sha256(rehearsal.name.encode("utf-8")).hexdigest()
        expected_readiness_run_id = (
            f"lifecycle_ready_{rehearsal_id_sha256[:12]}_{qualification_sha256[:16]}"
        )
        if readiness.name != expected_readiness_run_id:
            raise LifecycleQualificationError("readiness run ID is not canonical")
        current = _verify_rehearsal_read_only(
            project_root=root,
            authority=authority,
            rehearsal_run_directory=rehearsal,
        )
        if fresh_verifier_process_id == current.get("producer_process_id"):
            raise LifecycleQualificationError(
                "readiness verification was not executed in a fresh process"
            )
        expected_qualification = _qualification_binding_payload(current)
        if _canonical_sha256(expected_qualification) != qualification_sha256:
            raise LifecycleQualificationError("qualification binding is stale or mismatched")
        expected = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "policy": "fresh_process_lifecycle_readiness_v1",
            "decision": "passed",
            "scientific_outcome": False,
            "project_completion_status_changed": False,
            "fresh_verifier_process_id": fresh_verifier_process_id,
            **current,
            "qualification_binding_sha256": qualification_sha256,
        }
        if evidence != expected:
            raise LifecycleQualificationError("readiness evidence is stale or mismatched")
        readiness_config = {
            "schema_version": 1,
            "experiment_name": LIFECYCLE_READINESS_EXPERIMENT,
            "rehearsal_run_id": rehearsal.name,
            "qualification_binding_sha256": qualification_sha256,
            "readiness_record_sha256": readiness_record_sha256,
            "fresh_verifier_process_id": fresh_verifier_process_id,
            "scientific_outcome": False,
        }
        _require_resolved_config(readiness, readiness_config)
        _exact_registry_row(
            readiness,
            expected_status="completed",
            expected_experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
            expected_config_sha256=config_sha256(readiness_config),
        )
        _require_provenance(
            readiness,
            experiment_name=LIFECYCLE_READINESS_EXPERIMENT,
            expected_config=readiness_config,
            expected_source_root_sha256=str(current["authority"]["execution_source_root_sha256"]),
            details={
                "fresh_process_verifier": True,
                "fresh_verifier_process_id": fresh_verifier_process_id,
                "decision": "passed",
                "scientific_outcome": False,
                "rehearsal_run_id": rehearsal.name,
                "qualification_binding_sha256": qualification_sha256,
                "readiness_record_sha256": readiness_record_sha256,
            },
        )
    except (OSError, ValueError, TypeError, LifecycleQualificationError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result = LifecycleReadinessVerification(
        valid=not errors,
        readiness_run_directory=readiness,
        rehearsal_run_directory=rehearsal,
        qualification_binding_sha256=qualification_sha256,
        readiness_record_sha256=readiness_record_sha256,
        errors=tuple(errors),
    )
    if errors:
        raise LifecycleQualificationError(
            "lifecycle readiness gate failed closed: " + "; ".join(errors)
        )
    return result


def _require_original_confirmatory_authority_chain(
    *,
    project_root: Path,
    historical_authority_directory: Path,
    technical_authority_directory: Path,
    verify_live: bool,
) -> _PublishedT0LifecycleContext:
    verified, lifecycle_binding = _verified_published_original_confirmatory_technical_authority(
        project_root,
        technical_authority_directory,
        verify_live=verify_live,
    )
    technical = _published_t0_authority_binding(verified, lifecycle_binding)
    nested = technical.get("published_technical_authority_lifecycle_binding")
    nested_technical = nested.get("technical_authority") if isinstance(nested, Mapping) else None
    authority = verified.authority
    if not isinstance(nested, Mapping) or not isinstance(nested_technical, Mapping):
        raise LifecycleQualificationError(
            "original-confirmatory lifecycle authority lacks its published T0 carrier"
        )
    if (
        technical.get("authority_kind") != ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND
        or technical.get("authority_directory") != str(technical_authority_directory)
        or technical.get("artifact_root_sha256") != authority.artifact_root_sha256
        or technical.get("technical_authorization_sha256")
        != authority.technical_authorization_sha256
        or technical.get("historical_parent_authority_directory")
        != str(historical_authority_directory)
        or technical.get("historical_parent_artifact_root_sha256")
        != authority.parent_artifact_root_sha256
        or technical.get("historical_parent_sha256_manifest_sha256")
        != authority.parent_sha256_manifest_sha256
        or authority.parent_authority_directory.resolve() != historical_authority_directory
        or nested.get("namespace_directory") != str(verified.namespace_directory)
        or nested.get("namespace_claim_sha256") != verified.namespace_claim_sha256
        or nested.get("binding_sha256") != lifecycle_binding.get("binding_sha256")
        or nested_technical.get("authority_directory") != str(technical_authority_directory)
        or nested_technical.get("artifact_root_sha256") != authority.artifact_root_sha256
        or nested_technical.get("technical_authorization_sha256")
        != authority.technical_authorization_sha256
    ):
        raise LifecycleQualificationError(
            "published original-confirmatory T0 namespace/path/root/authorization "
            "or historical parent differs"
        )
    binding_sha256 = lifecycle_binding.get("binding_sha256")
    if not _valid_sha256(binding_sha256):
        raise LifecycleQualificationError(
            "published original-confirmatory T0 binding SHA-256 is invalid"
        )
    return _PublishedT0LifecycleContext(
        verified=verified,
        authority_binding=technical,
        pins=PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins(
            namespace_directory=verified.namespace_directory,
            namespace_claim_sha256=verified.namespace_claim_sha256,
            technical_authority_directory=authority.authority_directory,
            technical_authority_artifact_root_sha256=(authority.artifact_root_sha256),
            technical_authorization_sha256=(authority.technical_authorization_sha256),
            published_technical_authority_lifecycle_binding_sha256=str(binding_sha256),
        ),
    )


def _require_published_t0_context_unchanged(
    before: _PublishedT0LifecycleContext,
    after: _PublishedT0LifecycleContext,
    *,
    operation: str,
) -> None:
    if (
        after.verified != before.verified
        or after.authority_binding != before.authority_binding
        or after.pins != before.pins
    ):
        raise LifecycleQualificationError(
            f"published original-confirmatory authority carrier changed during {operation}"
        )


def execute_original_confirmatory_lifecycle_rehearsal(
    *,
    project_root: str | Path,
    historical_authority_directory: str | Path,
    technical_authority_directory: str | Path,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
) -> LifecycleRehearsalResult:
    """Create a rehearsal only under the exact published P -> T0 carrier."""

    root = Path(project_root).resolve()
    historical = Path(historical_authority_directory).resolve()
    technical = Path(technical_authority_directory).resolve()
    before = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=True,
    )
    result = _execute_lifecycle_rehearsal_with_authority(
        project_root=root,
        authority=before.authority_binding,
        runs_root=runs_root,
        run_id=run_id,
        retry_of_run_id=None,
    )
    after = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=False,
    )
    _require_published_t0_context_unchanged(
        before,
        after,
        operation="rehearsal",
    )
    return result


def verify_original_confirmatory_lifecycle_rehearsal_fresh_process(
    *,
    project_root: str | Path,
    historical_authority_directory: str | Path,
    technical_authority_directory: str | Path,
    rehearsal_run_directory: str | Path,
    runs_root: str | Path | None = None,
) -> LifecycleReadinessResult:
    """Publish readiness only for a rehearsal under the exact published T0."""

    root = Path(project_root).resolve()
    historical = Path(historical_authority_directory).resolve()
    technical = Path(technical_authority_directory).resolve()
    before = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=True,
    )
    result = _verify_lifecycle_rehearsal_fresh_process_with_authority(
        project_root=root,
        authority=before.authority_binding,
        rehearsal_run_directory=Path(rehearsal_run_directory).resolve(),
        runs_root=runs_root,
    )
    after = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=False,
    )
    _require_published_t0_context_unchanged(
        before,
        after,
        operation="readiness publication",
    )
    return result


def require_current_original_confirmatory_lifecycle_readiness(
    *,
    project_root: str | Path,
    historical_authority_directory: str | Path,
    technical_authority_directory: str | Path,
    readiness_run_directory: str | Path,
) -> OriginalConfirmatoryPublishedT0LifecycleReadinessVerification:
    """Require exact P -> published T0 -> readiness without caller-supplied pins."""

    root = Path(project_root).resolve()
    historical = Path(historical_authority_directory).resolve()
    technical = Path(technical_authority_directory).resolve()
    before = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=True,
    )
    readiness = _require_current_lifecycle_readiness_with_authority(
        project_root=root,
        authority=before.authority_binding,
        readiness_run_directory=Path(readiness_run_directory).resolve(),
    )
    after = _require_original_confirmatory_authority_chain(
        project_root=root,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        verify_live=False,
    )
    _require_published_t0_context_unchanged(
        before,
        after,
        operation="readiness verification",
    )
    return OriginalConfirmatoryPublishedT0LifecycleReadinessVerification(
        readiness=readiness,
        verified_published_technical_authority=before.verified,
        published_t0_pins=before.pins,
    )


__all__ = [
    "LIFECYCLE_AUTHORITY_FILENAME",
    "LIFECYCLE_COMPLETION_FILENAME",
    "LIFECYCLE_PLAN_FILENAME",
    "LIFECYCLE_PUBLICATION_FILENAME",
    "LIFECYCLE_READINESS_EXPERIMENT",
    "LIFECYCLE_READINESS_FILENAME",
    "LIFECYCLE_REHEARSAL_EXPERIMENT",
    "LifecycleQualificationError",
    "LifecycleReadinessResult",
    "LifecycleReadinessVerification",
    "LifecycleRehearsalResult",
    "OriginalConfirmatoryPublishedT0LifecycleReadinessVerification",
    "PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins",
    "execute_lifecycle_rehearsal",
    "execute_original_confirmatory_lifecycle_rehearsal",
    "require_current_lifecycle_readiness",
    "require_current_original_confirmatory_lifecycle_readiness",
    "verify_lifecycle_rehearsal_fresh_process",
    "verify_original_confirmatory_lifecycle_rehearsal_fresh_process",
]
