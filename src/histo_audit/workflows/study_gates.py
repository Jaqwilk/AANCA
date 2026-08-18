"""Cryptographic execution gates for real primary and confirmatory studies."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    read_primary_filesystem_evidence,
    read_primary_restoration_evidence,
)
from histo_audit.experiment.primary_core import primary_execution_controls_from_frozen_config
from histo_audit.experiment.primary_recovery import RECOVERY_COPY_POLICY
from histo_audit.experiment.study_contracts import (
    build_confirmatory_matrix_plan,
    build_primary_matrix_plan,
)
from histo_audit.utils.run_tracking import (
    SOURCE_GOVERNANCE_FILENAMES,
    RunStageEligibilityReceipt,
    capture_source_tree,
    require_run_stage_eligibility_receipt,
    require_run_stage_eligible,
    sha256_file,
    sha256_path,
    verify_run_integrity,
)
from histo_audit.workflows.preregistration import (
    BASE_FREEZE_EVIDENCE_SCHEMA_VERSION,
    verify_preregistration_freeze,
)
from histo_audit.workflows.preregistration_amendment import (
    RESOURCE_BOUNDED_CAPACITY_POLICY_V3,
    RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE,
    ConfirmatoryStoragePolicy,
    _require_sealed_effective_resource_bounded_confirmatory_authorization,
    require_confirmatory_storage_policy,
    require_primary_recovery_authorization,
    validate_resource_bounded_capacity_v3,
    verify_preregistration_amendment,
)

_require_sealed_resource_bounded_confirmatory_authorization = (
    _require_sealed_effective_resource_bounded_confirmatory_authorization
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_SOURCE_SCHEMA_VERSION = 3
_EXECUTION_SOURCE_SCOPE = ("src/**", "configs/**", "pyproject.toml", "uv.lock")
_EXECUTION_SOURCE_EXCLUDED_ROOTS = (".git", ".venv", "artifacts", "data")
_EXECUTION_SOURCE_EXCLUDED_PATHS = (
    "configs/confirmatory_frozen.yaml",
    "configs/primary_frozen.yaml",
)
_GOVERNANCE_SCHEMA_VERSION = 1
_MAX_AUTHORITY_CHAIN_DEPTH = 64
_PRIMARY_EXPERIMENT_NAME = "pannuke_primary_frozen_feature_benchmark"
PRIMARY_RECOVERY_EXPERIMENT_NAME = "pannuke_primary_orphan_recovery"
_ORIGINAL_REGISTRATION_STATUS = "original_unamended"
_PRE_OUTCOME_AMENDED_REGISTRATION_STATUS = "amended_before_outcome_inspection"
_POST_OUTCOME_RECOVERY_REGISTRATION_STATUS = "amended_or_exploratory"
_PRIMARY_RECOVERY_POLICY = "interrupted_unsealed_primary_recovery_v1"
_PRIMARY_RECOVERY_EVIDENCE_FILENAME = "primary_recovery_evidence.json"
_PRIMARY_RECOVERY_STAGE_POLICY = "primary_orphan_recovery_postseal_attestation_v1"
_PRIMARY_RECOVERY_REQUIRED_CELL_COUNT = 185
_PRIMARY_RECOVERY_SKIPPED_OPTIONAL_CELL_COUNT = 37
_PRIMARY_RECOVERY_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "experiment_name",
        "source_run_id",
        "destination_run_id",
        "recovery_authorization_sha256",
        "source_snapshot_root_sha256",
        "destination_snapshot_root_sha256",
        "reused_required_cell_count",
        "skipped_optional_cell_count",
        "retrained_cell_count",
        "copy_policy",
        "copied_artifact_count",
        "copied_total_bytes",
        "verification_mode",
        "prior_numeric_verification_proof_sha256",
        "analysis_disposition",
        "outcomes_inspected",
        "training_invoked",
        "matrix_executor_invoked",
        "fallback_invoked",
        "automatic_retry_allowed",
    }
)
_PRIMARY_FULL_RETRY_PREDECESSOR_POLICY = "sealed_failed_full_primary_retry_v1"
_PRIMARY_RETRY_SCIENTIFIC_GATE_BINDINGS = (
    "base_freeze_directory",
    "preregistration_sha256",
    "frozen_primary_config_sha256",
    "frozen_confirmatory_config_sha256",
    "primary_config_semantic_sha256",
    "confirmatory_config_semantic_sha256",
    "primary_matrix_cell_count",
    "primary_required_cell_count",
    "confirmatory_matrix_cell_count",
    "pilot_run_id",
    "pilot_artifact_root_sha256",
    "dataset_sha256",
    "manifest_sha256",
    "duplicate_audit_sha256",
    "pathology_encoder_audit_sha256",
)
_PRIMARY_FINALIZATION_SUCCESSOR_EXPERIMENT_NAME = "pannuke_primary_finalization_successor"
_PRIMARY_FINALIZATION_SUCCESSOR_FILENAME = "primary_finalization_successor_evidence.json"
_PRIMARY_FINALIZATION_SUCCESSOR_POLICY = "sealed_failed_primary_finalization_successor_v1"
_PRIMARY_FINALIZATION_AUTHORIZATION_POLICY_V1 = "sealed_failed_primary_finalization_successor_v1"
_PRIMARY_FINALIZATION_AUTHORIZATION_POLICY_V2 = "sealed_failed_primary_finalization_successor_v2"
_INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE = "inherited_prior_numeric_verification_v1"
_PRIMARY_STATISTICS_STAGE_READBACK_ATTESTATION = object()
_PRIMARY_STATISTICS_MANIFEST_FIELDS = {
    "schema_version",
    "execution_controls_binding_sha256",
    "source_filesystem_readback_root_sha256",
    "source_cell_artifact_manifest_sha256",
    "primary_input_bindings_sha256",
    "crop_cache_sha256",
    "artifacts",
    "statistics_payload_sha256",
    "subgroup_rows_sha256",
}


@dataclass(frozen=True, slots=True)
class PrimaryStatisticsStageReadback:
    """Hash-only readback of statistics covered by a positive post-seal attestation.

    This deliberately does not claim a fresh semantic recomputation. The primary
    producer or finalization successor must have completed that independent verifier
    before its immutable run could receive the attestation consumed here.
    """

    status: str
    output_directory: Path
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    subgroups_sha256: str
    manifest_sha256: str
    source_readback_root_sha256: str
    comparison_count: int
    stage_attestation_record_sha256: str
    stage_attestation_verification_sha256: str
    _attestation: object | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def valid(self) -> bool:
        return (
            self.status == "passed_postseal_attestation_readback"
            and self._attestation is _PRIMARY_STATISTICS_STAGE_READBACK_ATTESTATION
        )


@dataclass(frozen=True, slots=True)
class PrimaryExecutionGateEvidence:
    """Verified immutable dependencies required before a real primary run starts."""

    freeze_directory: Path
    freeze_artifact_root_sha256: str
    freeze_manifest_sha256: str
    preregistration_sha256: str
    frozen_primary_config_sha256: str
    frozen_confirmatory_config_sha256: str
    primary_config_semantic_sha256: str
    confirmatory_config_semantic_sha256: str
    primary_matrix_cell_count: int
    primary_required_cell_count: int
    confirmatory_matrix_cell_count: int
    pilot_run_id: str
    pilot_artifact_root_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    duplicate_audit_sha256: str
    pathology_encoder_audit_sha256: str
    source_tree_root_sha256: str
    base_freeze_directory: Path
    registration_authority_kind: str = "base_freeze"
    registration_status: str = _ORIGINAL_REGISTRATION_STATUS
    registration_authority_chain_depth: int = 0
    original_unamended_primary_claim_allowed: bool = True
    amended_primary_claim_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryExecutionGateEvidence:
    """Verified primary dependency required before confirmatory execution starts."""

    primary_gate: PrimaryExecutionGateEvidence
    primary_run_directory: Path
    primary_run_id: str
    primary_artifact_root_sha256: str
    primary_completion_evidence_sha256: str
    primary_reconciliation_sha256: str
    completed_required_cell_count: int
    primary_statistics_manifest_sha256: str | None = None
    primary_statistics_sha256: str | None = None
    primary_bootstrap_evidence_sha256: str | None = None
    primary_subgroups_sha256: str | None = None
    primary_statistics_source_readback_root_sha256: str | None = None
    primary_statistics_comparison_count: int | None = None
    primary_stage_attestation_record_sha256: str | None = None
    primary_stage_attestation_verification_sha256: str | None = None
    primary_restoration_readback_root_sha256: str | None = None
    primary_finalization_only_successor: bool = False
    primary_finalization_successor_evidence_sha256: str | None = None
    primary_predecessor_run_id: str | None = None
    primary_predecessor_artifact_root_sha256: str | None = None
    confirmatory_storage_policy_sha256: str | None = None
    primary_orphan_recovery: bool = False
    primary_recovery_evidence_sha256: str | None = None
    primary_recovery_authorization_sha256: str | None = None
    primary_recovery_source_run_id: str | None = None
    primary_recovery_source_snapshot_root_sha256: str | None = None
    primary_recovery_analysis_disposition: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "primary_gate": self.primary_gate.as_dict(),
            "primary_run_directory": str(self.primary_run_directory),
            "primary_run_id": self.primary_run_id,
            "primary_artifact_root_sha256": self.primary_artifact_root_sha256,
            "primary_completion_evidence_sha256": self.primary_completion_evidence_sha256,
            "primary_reconciliation_sha256": self.primary_reconciliation_sha256,
            "completed_required_cell_count": self.completed_required_cell_count,
            "primary_statistics_manifest_sha256": self.primary_statistics_manifest_sha256,
            "primary_statistics_sha256": self.primary_statistics_sha256,
            "primary_bootstrap_evidence_sha256": self.primary_bootstrap_evidence_sha256,
            "primary_subgroups_sha256": self.primary_subgroups_sha256,
            "primary_statistics_source_readback_root_sha256": (
                self.primary_statistics_source_readback_root_sha256
            ),
            "primary_statistics_comparison_count": self.primary_statistics_comparison_count,
            "primary_stage_attestation_record_sha256": (
                self.primary_stage_attestation_record_sha256
            ),
            "primary_stage_attestation_verification_sha256": (
                self.primary_stage_attestation_verification_sha256
            ),
            "primary_restoration_readback_root_sha256": (
                self.primary_restoration_readback_root_sha256
            ),
            "primary_finalization_only_successor": self.primary_finalization_only_successor,
            "primary_finalization_successor_evidence_sha256": (
                self.primary_finalization_successor_evidence_sha256
            ),
            "primary_predecessor_run_id": self.primary_predecessor_run_id,
            "primary_predecessor_artifact_root_sha256": (
                self.primary_predecessor_artifact_root_sha256
            ),
            "confirmatory_storage_policy_sha256": (self.confirmatory_storage_policy_sha256),
        }
        if self.primary_orphan_recovery:
            payload.update(
                {
                    "primary_orphan_recovery": True,
                    "primary_recovery_evidence_sha256": (self.primary_recovery_evidence_sha256),
                    "primary_recovery_authorization_sha256": (
                        self.primary_recovery_authorization_sha256
                    ),
                    "primary_recovery_source_run_id": self.primary_recovery_source_run_id,
                    "primary_recovery_source_snapshot_root_sha256": (
                        self.primary_recovery_source_snapshot_root_sha256
                    ),
                    "primary_recovery_analysis_disposition": (
                        self.primary_recovery_analysis_disposition
                    ),
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class HistoricalPrimaryDependencyEvidence:
    """Completed recovery primary authenticated under its historical authority P."""

    primary_gate: PrimaryExecutionGateEvidence
    primary_run_directory: Path
    primary_run_id: str
    primary_artifact_root_sha256: str
    primary_artifact_manifest_sha256: str
    primary_completion_evidence_sha256: str
    primary_execution_gate_sha256: str
    primary_reconciliation_sha256: str
    completed_required_cell_count: int
    primary_statistics_manifest_sha256: str
    primary_statistics_sha256: str
    primary_bootstrap_evidence_sha256: str
    primary_subgroups_sha256: str
    primary_statistics_source_readback_root_sha256: str
    primary_statistics_comparison_count: int
    primary_stage_attestation_record_sha256: str
    primary_stage_attestation_verification_sha256: str
    primary_restoration_readback_root_sha256: str
    primary_recovery_evidence_sha256: str
    primary_recovery_authorization_sha256: str
    primary_recovery_source_run_id: str
    primary_recovery_source_snapshot_root_sha256: str
    primary_recovery_analysis_disposition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_gate": self.primary_gate.as_dict(),
            "primary_run_directory": str(self.primary_run_directory),
            "primary_run_id": self.primary_run_id,
            "primary_artifact_root_sha256": self.primary_artifact_root_sha256,
            "primary_artifact_manifest_sha256": (self.primary_artifact_manifest_sha256),
            "primary_completion_evidence_sha256": (self.primary_completion_evidence_sha256),
            "primary_execution_gate_sha256": self.primary_execution_gate_sha256,
            "primary_reconciliation_sha256": self.primary_reconciliation_sha256,
            "completed_required_cell_count": self.completed_required_cell_count,
            "primary_statistics_manifest_sha256": (self.primary_statistics_manifest_sha256),
            "primary_statistics_sha256": self.primary_statistics_sha256,
            "primary_bootstrap_evidence_sha256": (self.primary_bootstrap_evidence_sha256),
            "primary_subgroups_sha256": self.primary_subgroups_sha256,
            "primary_statistics_source_readback_root_sha256": (
                self.primary_statistics_source_readback_root_sha256
            ),
            "primary_statistics_comparison_count": (self.primary_statistics_comparison_count),
            "primary_stage_attestation_record_sha256": (
                self.primary_stage_attestation_record_sha256
            ),
            "primary_stage_attestation_verification_sha256": (
                self.primary_stage_attestation_verification_sha256
            ),
            "primary_restoration_readback_root_sha256": (
                self.primary_restoration_readback_root_sha256
            ),
            "primary_recovery_evidence_sha256": (self.primary_recovery_evidence_sha256),
            "primary_recovery_authorization_sha256": (self.primary_recovery_authorization_sha256),
            "primary_recovery_source_run_id": self.primary_recovery_source_run_id,
            "primary_recovery_source_snapshot_root_sha256": (
                self.primary_recovery_source_snapshot_root_sha256
            ),
            "primary_recovery_analysis_disposition": (self.primary_recovery_analysis_disposition),
        }


@dataclass(frozen=True, slots=True)
class ResourceBoundedExecutionAuthorityEvidence:
    """Current resource execution authority C, separate from historical primary P."""

    authority_directory: Path
    authority_artifact_root_sha256: str
    authority_manifest_sha256: str
    authority_chain_depth: int
    authorization_sha256: str
    resource_profile_id: str
    resource_confirmatory_config_file_sha256: str
    resource_confirmatory_config_semantic_sha256: str
    resource_execution_source_root_sha256: str
    resource_execution_source_manifest_sha256: str
    resource_source_delta_sha256: str
    confirmatory_storage_policy_sha256: str
    resource_capacity_policy: Mapping[str, Any]
    resource_input_workspace_plan: Mapping[str, Any] | None = None
    resource_input_workspace_plan_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "authority_directory": str(self.authority_directory),
            "authority_artifact_root_sha256": (self.authority_artifact_root_sha256),
            "authority_manifest_sha256": self.authority_manifest_sha256,
            "authority_chain_depth": self.authority_chain_depth,
            "authorization_sha256": self.authorization_sha256,
            "resource_profile_id": self.resource_profile_id,
            "resource_confirmatory_config_file_sha256": (
                self.resource_confirmatory_config_file_sha256
            ),
            "resource_confirmatory_config_semantic_sha256": (
                self.resource_confirmatory_config_semantic_sha256
            ),
            "resource_execution_source_root_sha256": (self.resource_execution_source_root_sha256),
            "resource_execution_source_manifest_sha256": (
                self.resource_execution_source_manifest_sha256
            ),
            "resource_source_delta_sha256": self.resource_source_delta_sha256,
            "confirmatory_storage_policy_sha256": (self.confirmatory_storage_policy_sha256),
            "resource_capacity_policy": dict(self.resource_capacity_policy),
        }
        if self.resource_input_workspace_plan is not None:
            if self.resource_input_workspace_plan_sha256 is None:
                raise ValueError("resource input-workspace plan requires its exact SHA-256")
            payload["resource_input_workspace_plan"] = dict(self.resource_input_workspace_plan)
            payload["resource_input_workspace_plan_sha256"] = (
                self.resource_input_workspace_plan_sha256
            )
        elif self.resource_input_workspace_plan_sha256 is not None:
            raise ValueError("resource input-workspace plan SHA-256 cannot exist without its plan")
        return payload


@dataclass(frozen=True, slots=True)
class ResourceBoundedExecutionGateEvidence:
    """Dual-authority P+C evidence for permanently exploratory execution."""

    historical_primary: HistoricalPrimaryDependencyEvidence
    execution_authority: ResourceBoundedExecutionAuthorityEvidence
    purpose: str = RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
    outcomes_inspected: bool = True
    analysis_disposition: str = _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
    original_confirmatory_claim_allowed: bool = False
    study_outcome_eligible: bool = False
    completion_stage: None = None
    primary_rebinding_allowed: bool = False
    primary_mutation_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "historical_primary": self.historical_primary.as_dict(),
            "execution_authority": self.execution_authority.as_dict(),
            "purpose": self.purpose,
            "outcomes_inspected": self.outcomes_inspected,
            "analysis_disposition": self.analysis_disposition,
            "original_confirmatory_claim_allowed": (self.original_confirmatory_claim_allowed),
            "study_outcome_eligible": self.study_outcome_eligible,
            "completion_stage": self.completion_stage,
            "primary_rebinding_allowed": self.primary_rebinding_allowed,
            "primary_mutation_allowed": self.primary_mutation_allowed,
        }


@dataclass(frozen=True, slots=True)
class _PrimaryRecoveryDownstreamIdentity:
    """Exact sealed recovery lineage admitted to the confirmatory gate."""

    evidence_sha256: str
    authorization_sha256: str
    source_run_id: str
    source_snapshot_root_sha256: str
    analysis_disposition: str


@dataclass(frozen=True, slots=True)
class _ResolvedRegistrationAuthority:
    """Verified latest authority plus the immutable base dependency authority."""

    directory: Path
    base_freeze_directory: Path
    kind: str
    chain_depth: int
    artifact_root_sha256: str
    manifest_sha256: str
    preregistration_path: Path
    primary_config_path: Path
    confirmatory_config_path: Path
    source_tree_manifest_path: Path
    registration_status: str
    original_unamended_primary_claim_allowed: bool
    amended_primary_claim_allowed: bool


def _read_mapping(path: Path, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return payload


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _expected_hash(record: Mapping[str, Any], role: str) -> str:
    value = record.get("sha256")
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"freeze evidence lacks a valid {role} SHA-256")
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase 64-character SHA-256")
    return value


def _require_exact_integer(value: Any, role: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{role} must be an exact integer >= {minimum}")
    return value


def _canonical_artifact_root(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalise_tree_records(raw: Any, role: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{role} artifacts must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"{role} artifact {index} must contain exactly path/size_bytes/sha256")
        path = value.get("path")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != path
            or path in seen
        ):
            raise ValueError(f"{role} contains an invalid or duplicate path: {path!r}")
        size = _require_exact_integer(value.get("size_bytes"), f"{role} artifact {path!r} size")
        digest = _require_sha256(value.get("sha256"), f"{role} artifact {path!r} hash")
        seen.add(path)
        records.append({"path": path, "size_bytes": size, "sha256": digest})
    paths = [record["path"] for record in records]
    if paths != sorted(paths):
        raise ValueError(f"{role} artifact records must be sorted by path")
    return records


def _validate_execution_source_manifest(path: Path, role: str) -> Mapping[str, Any]:
    payload = _read_mapping(path, role)
    if set(payload) != {
        "schema_version",
        "scope_kind",
        "scope",
        "excluded_roots",
        "excluded_paths",
        "artifact_count",
        "root_sha256",
        "artifacts",
    }:
        raise ValueError(f"{role} must contain exactly the execution-source schema-v3 fields")
    if payload.get("schema_version") != _EXECUTION_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"{role} does not use execution-source schema v3")
    if payload.get("scope_kind") != "execution_source":
        raise ValueError(f"{role} scope kind is not execution_source")
    if payload.get("scope") != list(_EXECUTION_SOURCE_SCOPE):
        raise ValueError(f"{role} has an unexpected execution scope")
    if payload.get("excluded_roots") != list(_EXECUTION_SOURCE_EXCLUDED_ROOTS):
        raise ValueError(f"{role} has unexpected excluded roots")
    if payload.get("excluded_paths") != list(_EXECUTION_SOURCE_EXCLUDED_PATHS):
        raise ValueError(f"{role} has unexpected excluded paths")
    records = _normalise_tree_records(payload.get("artifacts"), role)
    invalid_paths = [
        record["path"]
        for record in records
        if record["path"] in _EXECUTION_SOURCE_EXCLUDED_PATHS
        or not (
            record["path"] in {"pyproject.toml", "uv.lock"}
            or record["path"].startswith("src/")
            or record["path"].startswith("configs/")
        )
    ]
    if invalid_paths:
        raise ValueError(f"{role} contains out-of-scope paths: {invalid_paths}")
    count = _require_exact_integer(payload.get("artifact_count"), f"{role} artifact count")
    if count != len(records):
        raise ValueError(f"{role} artifact count differs from its records")
    root = _require_sha256(payload.get("root_sha256"), f"{role} root")
    if root != _canonical_artifact_root(records):
        raise ValueError(f"{role} root differs from its canonical artifact records")
    return payload


def _validate_governance_manifest(path: Path, role: str) -> Mapping[str, Any]:
    payload = _read_mapping(path, role)
    if set(payload) != {
        "schema_version",
        "scope_kind",
        "scope",
        "governance_files",
        "artifact_count",
        "root_sha256",
        "artifacts",
    }:
        raise ValueError(f"{role} must contain exactly the governance schema-v1 fields")
    expected_scope = list(SOURCE_GOVERNANCE_FILENAMES)
    if payload.get("schema_version") != _GOVERNANCE_SCHEMA_VERSION:
        raise ValueError(f"{role} does not use governance schema v1")
    if payload.get("scope_kind") != "governance_snapshot":
        raise ValueError(f"{role} scope kind is not governance_snapshot")
    if payload.get("scope") != expected_scope or payload.get("governance_files") != expected_scope:
        raise ValueError(f"{role} has an unexpected governance scope")
    records = _normalise_tree_records(payload.get("artifacts"), role)
    invalid_paths = [
        record["path"] for record in records if record["path"] not in SOURCE_GOVERNANCE_FILENAMES
    ]
    if invalid_paths:
        raise ValueError(f"{role} contains out-of-scope governance paths: {invalid_paths}")
    count = _require_exact_integer(payload.get("artifact_count"), f"{role} artifact count")
    if count != len(records):
        raise ValueError(f"{role} artifact count differs from its records")
    root = _require_sha256(payload.get("root_sha256"), f"{role} root")
    if root != _canonical_artifact_root(records):
        raise ValueError(f"{role} root differs from its canonical artifact records")
    return payload


def _cross_check_tree_evidence(
    *,
    freeze: Path,
    evidence: Mapping[str, Any],
    evidence_key: str,
    snapshot_name: str,
    validator: Any,
    role: str,
) -> Mapping[str, Any]:
    record = _mapping(evidence.get(evidence_key), f"{role} freeze evidence")
    if set(record) != {
        "schema_version",
        "scope_kind",
        "scope",
        "artifact_count",
        "root_sha256",
        "snapshot",
    }:
        raise ValueError(f"{role} freeze evidence has an unexpected field set")
    if record.get("snapshot") != snapshot_name:
        raise ValueError(f"{role} freeze evidence names an unexpected snapshot")
    snapshot = validator(freeze / snapshot_name, f"frozen {role} manifest")
    for field in ("schema_version", "scope_kind", "scope", "artifact_count", "root_sha256"):
        if record.get(field) != snapshot.get(field):
            raise ValueError(
                f"{role} freeze evidence {field} differs from its authenticated snapshot"
            )
    return snapshot


def _resolve_from_project(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _raise_invalid_base_freeze(verification: Any) -> NoReturn:
    raise ValueError(
        "preregistration freeze failed integrity verification: "
        f"errors={verification.errors}, missing={verification.missing_paths}, "
        f"added={verification.added_paths}, changed={verification.changed_paths}"
    )


def _verify_amendment_dispositions(
    directory: Path,
    *,
    experiment_name: str,
) -> tuple[Path, str]:
    """Return the base and conservative status for one purpose-bound amendment chain."""

    current = directory
    visited: set[Path] = set()
    registration_status = _PRE_OUTCOME_AMENDED_REGISTRATION_STATUS
    for _ in range(_MAX_AUTHORITY_CHAIN_DEPTH):
        if current in visited:
            raise ValueError("preregistration amendment parent cycle detected")
        visited.add(current)
        evidence = _read_mapping(current / "amendment_evidence.json", "amendment evidence")
        outcomes_inspected = evidence.get("outcomes_inspected")
        if outcomes_inspected is True:
            if experiment_name != PRIMARY_RECOVERY_EXPERIMENT_NAME:
                raise ValueError(
                    "post-outcome amendment cannot authorize an original or amended primary claim"
                )
            try:
                require_primary_recovery_authorization(current)
            except ValueError as exc:
                raise ValueError(
                    "post-outcome recovery amendment lacks a typed primary recovery "
                    f"authorization: {exc}"
                ) from exc
            expected_registration_status = _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
            expected_amended_claim = False
            registration_status = _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
        elif outcomes_inspected is False:
            expected_registration_status = _PRE_OUTCOME_AMENDED_REGISTRATION_STATUS
            expected_amended_claim = True
        else:
            raise ValueError("amendment outcomes_inspected must be an exact boolean")
        dispositions = evidence.get("analysis_dispositions")
        if not isinstance(dispositions, list) or not dispositions:
            raise ValueError("amendment lacks explicit affected-analysis dispositions")
        for disposition in dispositions:
            if (
                not isinstance(disposition, Mapping)
                or disposition.get("registration_status") != expected_registration_status
                or disposition.get("original_unamended_primary_claim_allowed") is not False
                or disposition.get("amended_primary_claim_allowed") is not expected_amended_claim
            ):
                if outcomes_inspected is True:
                    raise ValueError(
                        "post-outcome recovery dispositions must all be "
                        "amended_or_exploratory with both primary claim booleans false"
                    )
                raise ValueError(
                    "amendment analysis dispositions do not authorize a pre-outcome amended primary"
                )
        parent = _mapping(evidence.get("parent"), "amendment parent evidence")
        parent_path_value = parent.get("authority_directory")
        if not isinstance(parent_path_value, str) or not Path(parent_path_value).is_absolute():
            raise ValueError("amendment parent authority path must be explicit and absolute")
        parent_path = Path(parent_path_value).resolve()
        parent_kind = parent.get("authority_kind")
        if parent_kind == "base_freeze":
            return parent_path, registration_status
        if parent_kind != "preregistration_amendment":
            raise ValueError("amendment parent authority kind is invalid")
        current = parent_path
    raise ValueError(
        f"preregistration amendment chain exceeds {_MAX_AUTHORITY_CHAIN_DEPTH} authorities"
    )


def _resolve_registration_authority(
    *,
    root: Path,
    authority_directory: Path,
    requested_primary_config: str | Path | None,
    requested_confirmatory_config: str | Path | None,
    experiment_name: str,
) -> _ResolvedRegistrationAuthority:
    if not isinstance(experiment_name, str) or experiment_name not in {
        _PRIMARY_EXPERIMENT_NAME,
        PRIMARY_RECOVERY_EXPERIMENT_NAME,
    }:
        raise ValueError(f"unsupported primary execution purpose: {experiment_name!r}")
    marker = _read_mapping(authority_directory / ".immutable.json", "authority marker")
    status = marker.get("status")
    if status == "frozen":
        if experiment_name == PRIMARY_RECOVERY_EXPERIMENT_NAME:
            raise ValueError(
                "primary orphan recovery requires a verified post-outcome recovery amendment"
            )
        primary = _resolve_from_project(
            root, requested_primary_config or root / "configs" / "primary_frozen.yaml"
        )
        confirmatory = _resolve_from_project(
            root,
            requested_confirmatory_config or root / "configs" / "confirmatory_frozen.yaml",
        )
        base_verification = verify_preregistration_freeze(
            authority_directory,
            frozen_primary_config_path=primary,
            frozen_confirmatory_config_path=confirmatory,
        )
        if not base_verification.valid or base_verification.expected_artifact_root_sha256 is None:
            _raise_invalid_base_freeze(base_verification)
        return _ResolvedRegistrationAuthority(
            directory=authority_directory,
            base_freeze_directory=authority_directory,
            kind="base_freeze",
            chain_depth=0,
            artifact_root_sha256=base_verification.expected_artifact_root_sha256,
            manifest_sha256=sha256_file(authority_directory / "sha256_manifest.json"),
            preregistration_path=authority_directory / "PRE_REGISTRATION_FROZEN.md",
            primary_config_path=primary,
            confirmatory_config_path=confirmatory,
            source_tree_manifest_path=authority_directory / "source_tree_manifest.json",
            registration_status=_ORIGINAL_REGISTRATION_STATUS,
            original_unamended_primary_claim_allowed=True,
            amended_primary_claim_allowed=False,
        )
    if status != "amended" or marker.get("authority_kind") != "preregistration_amendment":
        raise ValueError("registration authority is neither a base freeze nor an amendment")

    from histo_audit.workflows.preregistration_amendment import (
        verify_preregistration_amendment,
    )

    amendment_verification = verify_preregistration_amendment(
        authority_directory, max_chain_depth=_MAX_AUTHORITY_CHAIN_DEPTH
    )
    if (
        not amendment_verification.valid
        or amendment_verification.artifact_root_sha256 is None
        or amendment_verification.sha256_manifest_sha256 is None
        or amendment_verification.chain_depth is None
    ):
        raise ValueError(
            "preregistration amendment failed chain/integrity verification: "
            f"errors={amendment_verification.errors}"
        )
    base, registration_status = _verify_amendment_dispositions(
        authority_directory,
        experiment_name=experiment_name,
    )
    if (
        experiment_name == PRIMARY_RECOVERY_EXPERIMENT_NAME
        and registration_status != _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
    ):
        raise ValueError(
            "primary orphan recovery requires a verified post-outcome recovery amendment"
        )
    base_verification = verify_preregistration_freeze(base)
    if not base_verification.valid or base_verification.expected_artifact_root_sha256 is None:
        _raise_invalid_base_freeze(base_verification)

    primary_snapshot = (authority_directory / "primary_frozen.yaml").resolve()
    confirmatory_snapshot = (authority_directory / "confirmatory_frozen.yaml").resolve()
    primary = (
        primary_snapshot
        if requested_primary_config is None
        else _resolve_from_project(root, requested_primary_config)
    )
    confirmatory = (
        confirmatory_snapshot
        if requested_confirmatory_config is None
        else _resolve_from_project(root, requested_confirmatory_config)
    )
    if primary != primary_snapshot or confirmatory != confirmatory_snapshot:
        raise ValueError(
            "amended execution must load the primary and confirmatory snapshots from the latest "
            "verified amendment bundle"
        )
    return _ResolvedRegistrationAuthority(
        directory=authority_directory,
        base_freeze_directory=base,
        kind="preregistration_amendment",
        chain_depth=amendment_verification.chain_depth,
        artifact_root_sha256=amendment_verification.artifact_root_sha256,
        manifest_sha256=amendment_verification.sha256_manifest_sha256,
        preregistration_path=authority_directory / "PRE_REGISTRATION_FROZEN.md",
        primary_config_path=primary,
        confirmatory_config_path=confirmatory,
        source_tree_manifest_path=authority_directory / "source_tree_manifest.json",
        registration_status=registration_status,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=(
            registration_status == _PRE_OUTCOME_AMENDED_REGISTRATION_STATUS
        ),
    )


def _require_matching_path_hash(
    path: Path, expected: str, role: str, *, tree: bool = False
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    actual = sha256_path(path) if tree else sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} hash differs from the immutable preregistration freeze")


def _validate_base_config_record(
    *,
    record: Mapping[str, Any],
    snapshot_path: Path,
    canonical_path: Path | None,
    role: str,
) -> Mapping[str, Any]:
    frozen_hash = _require_sha256(record.get("frozen_file_sha256"), f"{role} frozen-file hash")
    semantic_hash = _require_sha256(record.get("semantic_sha256"), f"{role} semantic hash")
    if sha256_file(snapshot_path) != frozen_hash:
        raise ValueError(f"timestamped {role} config differs from freeze evidence")
    snapshot_config = load_config(snapshot_path)
    if config_sha256(snapshot_config) != semantic_hash:
        raise ValueError(f"timestamped {role} config semantic hash differs from freeze evidence")
    if canonical_path is not None:
        declared = record.get("frozen_path")
        if not isinstance(declared, str) or Path(declared).resolve() != canonical_path:
            raise ValueError(f"canonical {role} config path differs from freeze evidence")
        if sha256_file(canonical_path) != frozen_hash:
            raise ValueError(f"canonical {role} config differs from freeze evidence")
    return snapshot_config


def _validate_amended_config_record(
    *,
    record: Mapping[str, Any],
    config_path: Path,
    expected_snapshot: str,
    role: str,
) -> Mapping[str, Any]:
    if record.get("snapshot") != expected_snapshot:
        raise ValueError(f"amended {role} config record names an unexpected snapshot")
    file_hash = _require_sha256(record.get("file_sha256"), f"amended {role} file hash")
    semantic_hash = _require_sha256(record.get("semantic_sha256"), f"amended {role} semantic hash")
    if sha256_file(config_path) != file_hash:
        raise ValueError(f"amended {role} config differs from amendment evidence")
    config = load_config(config_path)
    if config_sha256(config) != semantic_hash:
        raise ValueError(f"amended {role} config semantic hash differs from amendment evidence")
    return config


def _reverify_registration_authority(authority: _ResolvedRegistrationAuthority) -> None:
    if authority.kind == "base_freeze":
        base_verification = verify_preregistration_freeze(
            authority.directory,
            frozen_primary_config_path=authority.primary_config_path,
            frozen_confirmatory_config_path=authority.confirmatory_config_path,
        )
        if (
            not base_verification.valid
            or base_verification.expected_artifact_root_sha256 != authority.artifact_root_sha256
            or sha256_file(authority.directory / "sha256_manifest.json")
            != authority.manifest_sha256
        ):
            raise ValueError("base preregistration authority changed during execution-gate checks")
        return

    from histo_audit.workflows.preregistration_amendment import (
        verify_preregistration_amendment,
    )

    amendment_verification = verify_preregistration_amendment(
        authority.directory, max_chain_depth=_MAX_AUTHORITY_CHAIN_DEPTH
    )
    if (
        not amendment_verification.valid
        or amendment_verification.artifact_root_sha256 != authority.artifact_root_sha256
        or amendment_verification.sha256_manifest_sha256 != authority.manifest_sha256
        or amendment_verification.chain_depth != authority.chain_depth
    ):
        raise ValueError("preregistration amendment authority changed during execution-gate checks")
    if sha256_file(authority.primary_config_path) != sha256_file(
        authority.directory / "primary_frozen.yaml"
    ) or sha256_file(authority.confirmatory_config_path) != sha256_file(
        authority.directory / "confirmatory_frozen.yaml"
    ):
        raise ValueError("amended config snapshots changed during execution-gate checks")


def _validate_registered_primary_dependencies(
    *,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path | None = None,
    frozen_confirmatory_config_path: str | Path | None = None,
    experiment_name: str = _PRIMARY_EXPERIMENT_NAME,
) -> PrimaryExecutionGateEvidence:
    """Reconstruct immutable registered dependencies without consulting live source.

    This private boundary is also used to authenticate a historical completed primary
    under its original recovery authority.  It verifies the authority snapshots,
    dataset, pilot, and matrix contracts, but deliberately makes no claim that the
    current checkout still equals that historical source snapshot.
    """

    root = Path(project_root).resolve()
    freeze = Path(freeze_directory).resolve()
    authority = _resolve_registration_authority(
        root=root,
        authority_directory=freeze,
        requested_primary_config=frozen_primary_config_path,
        requested_confirmatory_config=frozen_confirmatory_config_path,
        experiment_name=experiment_name,
    )
    base_freeze = authority.base_freeze_directory
    evidence = _read_mapping(base_freeze / "freeze_evidence.json", "freeze evidence")
    if evidence.get("schema_version") != BASE_FREEZE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "freeze evidence does not use the required schema version "
            f"{BASE_FREEZE_EVIDENCE_SCHEMA_VERSION}"
        )
    if evidence.get("completion_stage_enabled") != "PRE_REGISTRATION_FROZEN":
        raise ValueError("freeze evidence does not enable PRE_REGISTRATION_FROZEN")

    base_source_manifest = _cross_check_tree_evidence(
        freeze=base_freeze,
        evidence=evidence,
        evidence_key="execution_source_tree",
        snapshot_name="source_tree_manifest.json",
        validator=_validate_execution_source_manifest,
        role="execution source tree",
    )
    governance_manifest = _cross_check_tree_evidence(
        freeze=base_freeze,
        evidence=evidence,
        evidence_key="governance_tree",
        snapshot_name="governance_tree_manifest.json",
        validator=_validate_governance_manifest,
        role="governance tree",
    )
    if evidence.get("source_tree_root_sha256") != base_source_manifest.get("root_sha256"):
        raise ValueError("legacy source-tree root alias differs from execution-source evidence")
    if evidence.get("governance_tree_root_sha256") != governance_manifest.get("root_sha256"):
        raise ValueError("governance-tree root alias differs from governance evidence")

    primary_record = _mapping(evidence.get("primary_config"), "primary config evidence")
    confirmatory_record = _mapping(
        evidence.get("confirmatory_config"), "confirmatory config evidence"
    )
    if confirmatory_record.get("frozen_before_primary_outcomes") is not True:
        raise ValueError("confirmatory matrix was not frozen before primary outcomes")
    _validate_base_config_record(
        record=primary_record,
        snapshot_path=base_freeze / "primary_frozen.yaml",
        canonical_path=(authority.primary_config_path if authority.kind == "base_freeze" else None),
        role="primary",
    )
    _validate_base_config_record(
        record=confirmatory_record,
        snapshot_path=base_freeze / "confirmatory_frozen.yaml",
        canonical_path=(
            authority.confirmatory_config_path if authority.kind == "base_freeze" else None
        ),
        role="confirmatory",
    )

    frozen_source_hash = str(base_source_manifest["root_sha256"])
    if authority.kind == "preregistration_amendment":
        amendment_evidence = _read_mapping(
            authority.directory / "amendment_evidence.json", "amendment evidence"
        )
        after = _mapping(amendment_evidence.get("after"), "amendment after evidence")
        primary_config = _validate_amended_config_record(
            record=_mapping(after.get("primary_config"), "amended primary config evidence"),
            config_path=authority.primary_config_path,
            expected_snapshot="primary_frozen.yaml",
            role="primary",
        )
        confirmatory_config = _validate_amended_config_record(
            record=_mapping(
                after.get("confirmatory_config"), "amended confirmatory config evidence"
            ),
            config_path=authority.confirmatory_config_path,
            expected_snapshot="confirmatory_frozen.yaml",
            role="confirmatory",
        )
        latest_source = _validate_execution_source_manifest(
            authority.source_tree_manifest_path, "amended execution-source manifest"
        )
        amended_source_record = _mapping(
            after.get("execution_source"), "amended execution-source evidence"
        )
        if amended_source_record.get("snapshot") != "source_tree_manifest.json":
            raise ValueError("amended execution-source evidence names an unexpected snapshot")
        if amended_source_record.get("manifest_sha256") != sha256_file(
            authority.source_tree_manifest_path
        ):
            raise ValueError("amended execution-source manifest hash differs from its evidence")
        if amended_source_record.get("root_sha256") != latest_source.get("root_sha256"):
            raise ValueError("amended execution-source root differs from its evidence")
        frozen_source_hash = str(latest_source["root_sha256"])
    else:
        primary_config = load_config(authority.primary_config_path)
        confirmatory_config = load_config(authority.confirmatory_config_path)

    primary_plan = build_primary_matrix_plan(primary_config)
    confirmatory_plan = build_confirmatory_matrix_plan(confirmatory_config)

    dataset_record = _mapping(evidence.get("dataset"), "dataset evidence")
    manifest_record = _mapping(evidence.get("manifest"), "manifest evidence")
    duplicate_record = _mapping(evidence.get("duplicate_audit"), "duplicate-audit evidence")
    pathology_record = _mapping(
        evidence.get("pathology_encoder_availability_audit"), "pathology-audit evidence"
    )
    dataset = _resolve_from_project(root, dataset_path)
    manifest = _resolve_from_project(root, manifest_path)
    duplicate_audit = _resolve_from_project(root, duplicate_audit_path)
    pathology_audit = _resolve_from_project(root, pathology_encoder_audit_path)
    dataset_sha = _expected_hash(dataset_record, "dataset")
    manifest_sha = _expected_hash(manifest_record, "manifest")
    duplicate_sha = _expected_hash(duplicate_record, "duplicate audit")
    pathology_sha = _expected_hash(pathology_record, "pathology encoder audit")
    _require_matching_path_hash(dataset, dataset_sha, "dataset", tree=True)
    _require_matching_path_hash(manifest, manifest_sha, "manifest")
    _require_matching_path_hash(duplicate_audit, duplicate_sha, "duplicate audit")
    _require_matching_path_hash(pathology_audit, pathology_sha, "pathology encoder audit")

    pilot_record = _mapping(evidence.get("pilot"), "pilot evidence")
    pilot_directory_value = pilot_record.get("run_directory")
    if not isinstance(pilot_directory_value, str) or not pilot_directory_value:
        raise ValueError("freeze pilot evidence lacks an explicit run directory")
    pilot_path = _resolve_from_project(root, pilot_directory_value)
    pilot_integrity = verify_run_integrity(pilot_path)
    if not pilot_integrity.valid or not pilot_integrity.registry_record_present:
        raise ValueError(f"frozen pilot failed integrity verification: {pilot_integrity.errors}")
    require_run_stage_eligible(pilot_path, integrity=pilot_integrity)
    if pilot_integrity.run_id != pilot_record.get("run_id"):
        raise ValueError("frozen pilot run ID differs from live integrity evidence")
    if pilot_integrity.expected_root_sha256 != pilot_record.get("artifact_root_sha256"):
        raise ValueError("frozen pilot artifact root differs from live integrity evidence")
    pilot_checksums = _read_mapping(pilot_path / "checksums.json", "pilot checksums")
    if pilot_checksums.get("duplicate_audit_status") != f"complete_sha256:{duplicate_sha}":
        raise ValueError("pilot does not bind the frozen duplicate-audit artifact")

    _reverify_registration_authority(authority)
    return PrimaryExecutionGateEvidence(
        freeze_directory=authority.directory,
        freeze_artifact_root_sha256=authority.artifact_root_sha256,
        freeze_manifest_sha256=authority.manifest_sha256,
        preregistration_sha256=sha256_file(authority.preregistration_path),
        frozen_primary_config_sha256=sha256_file(authority.primary_config_path),
        frozen_confirmatory_config_sha256=sha256_file(authority.confirmatory_config_path),
        primary_config_semantic_sha256=primary_plan.config_sha256,
        confirmatory_config_semantic_sha256=confirmatory_plan.config_sha256,
        primary_matrix_cell_count=len(primary_plan.cells),
        primary_required_cell_count=primary_plan.required_cell_count,
        confirmatory_matrix_cell_count=len(confirmatory_plan.cells),
        pilot_run_id=str(pilot_integrity.run_id),
        pilot_artifact_root_sha256=str(pilot_integrity.expected_root_sha256),
        dataset_sha256=dataset_sha,
        manifest_sha256=manifest_sha,
        duplicate_audit_sha256=duplicate_sha,
        pathology_encoder_audit_sha256=pathology_sha,
        source_tree_root_sha256=frozen_source_hash,
        base_freeze_directory=base_freeze,
        registration_authority_kind=authority.kind,
        registration_status=authority.registration_status,
        registration_authority_chain_depth=authority.chain_depth,
        original_unamended_primary_claim_allowed=(
            authority.original_unamended_primary_claim_allowed
        ),
        amended_primary_claim_allowed=authority.amended_primary_claim_allowed,
    )


def validate_primary_execution_gate(
    *,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path | None = None,
    frozen_confirmatory_config_path: str | Path | None = None,
    experiment_name: str = _PRIMARY_EXPERIMENT_NAME,
) -> PrimaryExecutionGateEvidence:
    """Verify every immutable real-data dependency before creating a primary run.

    The public primary gate retains its original fail-closed live-source comparison.
    Historical consumers use a separate private reconstruction boundary and therefore
    cannot accidentally turn this API into a source-check bypass.
    """

    evidence = _validate_registered_primary_dependencies(
        project_root=project_root,
        freeze_directory=freeze_directory,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        frozen_primary_config_path=frozen_primary_config_path,
        frozen_confirmatory_config_path=frozen_confirmatory_config_path,
        experiment_name=experiment_name,
    )
    current_source = capture_source_tree(Path(project_root).resolve())
    if current_source.get("root_sha256") != evidence.source_tree_root_sha256:
        raise ValueError(
            "current execution source tree differs from the registration authority; "
            "create a dated amendment before execution"
        )
    return evidence


def _read_primary_retry_registry_row(
    *,
    registry_path: Path,
    retry_of_run_id: str,
    predecessor: Path,
) -> dict[str, Any]:
    try:
        with registry_path.open("r", encoding="utf-8", newline="") as stream:
            rows: list[dict[str, Any]] = []
            for raw_row in csv.DictReader(stream):
                if None in raw_row:
                    raise ValueError("primary run registry contains an invalid row")
                row: dict[str, Any] = {str(key): value for key, value in raw_row.items()}
                if row.get("run_id") == retry_of_run_id:
                    rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(
            f"primary retry predecessor registry is unavailable or invalid: {registry_path}: {exc}"
        ) from exc
    if (
        len(rows) != 1
        or rows[0].get("status") != "failed"
        or rows[0].get("experiment_name") != _PRIMARY_EXPERIMENT_NAME
        or not isinstance(rows[0].get("run_path"), str)
        or Path(str(rows[0]["run_path"])).resolve() != predecessor
    ):
        raise ValueError(
            "primary retry predecessor lacks exactly one matching failed run-registry record"
        )
    return rows[0]


def _reconstruct_primary_retry_predecessor_evidence(
    *,
    run_directory: Path,
    retry_of_run_id: str,
    primary_gate: PrimaryExecutionGateEvidence,
) -> dict[str, Any]:
    """Reconstruct the producer's binding from one live sealed failed predecessor."""

    if Path(retry_of_run_id).name != retry_of_run_id or not retry_of_run_id.strip():
        raise ValueError("primary retry_of_run_id must be one safe non-empty run ID")
    run_root = run_directory.parent.resolve()
    predecessor = (run_root / retry_of_run_id).resolve()
    if (
        predecessor == run_directory.resolve()
        or predecessor.parent != run_root
        or not predecessor.is_dir()
    ):
        raise ValueError("primary retry predecessor is not an exact sibling run")

    integrity = verify_run_integrity(predecessor)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != retry_of_run_id
        or not isinstance(integrity.expected_root_sha256, str)
        or _SHA256_PATTERN.fullmatch(integrity.expected_root_sha256) is None
        or integrity.actual_root_sha256 != integrity.expected_root_sha256
    ):
        raise ValueError(
            "primary retry predecessor is not an integrity-valid registry-backed sealed run: "
            f"{integrity.errors}"
        )
    artifact_root_sha256 = integrity.expected_root_sha256

    required_files = {
        "artifact_manifest": predecessor / "artifact_manifest.json",
        "immutable_marker": predecessor / ".immutable.json",
        "status": predecessor / "status.json",
        "environment": predecessor / "environment.json",
        "completion": predecessor / "completion_evidence.json",
        "primary_gate": predecessor / "primary_execution_gate.json",
        "primary_input_bindings": predecessor / "primary_input_bindings.json",
    }
    missing = sorted(role for role, path in required_files.items() if not path.is_file())
    if missing:
        raise ValueError(f"primary retry predecessor lacks sealed evidence: {missing}")

    artifact_manifest = _read_mapping(
        required_files["artifact_manifest"], "primary retry predecessor artifact manifest"
    )
    immutable_marker = _read_mapping(
        required_files["immutable_marker"], "primary retry predecessor immutable marker"
    )
    status = _read_mapping(required_files["status"], "primary retry predecessor status")
    environment = _read_mapping(
        required_files["environment"], "primary retry predecessor environment"
    )
    completion = _read_mapping(required_files["completion"], "primary retry predecessor completion")
    predecessor_gate = _read_mapping(
        required_files["primary_gate"], "primary retry predecessor execution gate"
    )
    predecessor_inputs = _read_mapping(
        required_files["primary_input_bindings"],
        "primary retry predecessor input bindings",
    )
    environment_gate = _mapping(
        environment.get("primary_gate"), "primary retry predecessor environment gate"
    )
    marker_run_path = immutable_marker.get("run_path")
    if (
        status.get("status") != "failed"
        or status.get("run_id") != retry_of_run_id
        or status.get("experiment_name") != _PRIMARY_EXPERIMENT_NAME
        or artifact_manifest.get("status") != "failed"
        or artifact_manifest.get("run_id") != retry_of_run_id
        or artifact_manifest.get("artifact_root_sha256") != artifact_root_sha256
        or immutable_marker.get("status") != "failed"
        or immutable_marker.get("run_id") != retry_of_run_id
        or immutable_marker.get("artifact_root_sha256") != artifact_root_sha256
        or immutable_marker.get("artifact_manifest_sha256")
        != sha256_file(required_files["artifact_manifest"])
        or not isinstance(marker_run_path, str)
        or Path(marker_run_path).resolve() != predecessor
        or environment.get("artifact_scope") != "real_pannuke_primary_study"
        or completion.get("completion_stage") is not None
        or completion.get("study_outcome_eligible") is not False
        or completion.get("valid_completion_claim") is not False
        or completion.get("artifact_scope") != "real_pannuke_primary_study"
        or not str(completion.get("runner_failure", "")).strip()
        or dict(predecessor_gate) != dict(environment_gate)
        or set(predecessor_gate) != set(primary_gate.as_dict())
        or predecessor_inputs.get("schema_version") != 1
        or not isinstance(predecessor_inputs.get("execution_controls_binding_sha256"), str)
        or _SHA256_PATTERN.fullmatch(
            str(predecessor_inputs.get("execution_controls_binding_sha256"))
        )
        is None
    ):
        raise ValueError("primary retry predecessor is not an exact failed full-primary run")

    predecessor_hashes = {
        field: value for field, value in predecessor_gate.items() if field.endswith("_sha256")
    }
    predecessor_counts = {
        field: predecessor_gate.get(field)
        for field in (
            "primary_matrix_cell_count",
            "primary_required_cell_count",
            "confirmatory_matrix_cell_count",
        )
    }
    if (
        not predecessor_hashes
        or any(
            not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            for value in predecessor_hashes.values()
        )
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in predecessor_counts.values()
        )
        or not all(
            isinstance(predecessor_gate.get(field), str)
            and Path(str(predecessor_gate[field])).is_absolute()
            for field in ("freeze_directory", "base_freeze_directory")
        )
    ):
        raise ValueError("primary retry predecessor gate contains invalid typed bindings")

    current_gate = primary_gate.as_dict()
    scientific_bindings = {
        field: current_gate[field] for field in _PRIMARY_RETRY_SCIENTIFIC_GATE_BINDINGS
    }
    if any(predecessor_gate.get(field) != value for field, value in scientific_bindings.items()):
        raise ValueError(
            "primary retry predecessor scientific bindings differ from the current authority"
        )
    registry_row = _read_primary_retry_registry_row(
        registry_path=run_root / "registry.csv",
        retry_of_run_id=retry_of_run_id,
        predecessor=predecessor,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": _PRIMARY_FULL_RETRY_PREDECESSOR_POLICY,
        "run_id": retry_of_run_id,
        "run_directory": str(predecessor),
        "experiment_name": _PRIMARY_EXPERIMENT_NAME,
        "terminal_status": "failed",
        "integrity_verified": True,
        "integrity_registry_record_present": True,
        "artifact_root_sha256": artifact_root_sha256,
        "artifact_manifest_sha256": sha256_file(required_files["artifact_manifest"]),
        "immutable_marker_sha256": sha256_file(required_files["immutable_marker"]),
        "status_sha256": sha256_file(required_files["status"]),
        "environment_sha256": sha256_file(required_files["environment"]),
        "completion_evidence_sha256": sha256_file(required_files["completion"]),
        "primary_execution_gate_sha256": sha256_file(required_files["primary_gate"]),
        "primary_input_bindings_sha256": sha256_file(required_files["primary_input_bindings"]),
        "registry_row_sha256": canonical_sha256(registry_row),
        "predecessor_freeze_directory": predecessor_gate.get("freeze_directory"),
        "predecessor_freeze_artifact_root_sha256": predecessor_gate.get(
            "freeze_artifact_root_sha256"
        ),
        "predecessor_source_tree_root_sha256": predecessor_gate.get("source_tree_root_sha256"),
        "scientific_gate_bindings": scientific_bindings,
        "scientific_gate_binding_sha256": canonical_sha256(scientific_bindings),
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return payload


def _validate_primary_full_retry_binding(
    *,
    run_directory: Path,
    primary_gate: PrimaryExecutionGateEvidence,
    completion: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> None:
    """Fail closed unless ordinary-primary retry lineage matches the producer contract."""

    legacy_path = run_directory / "primary_retry_predecessor.json"
    if legacy_path.exists() or legacy_path.is_symlink():
        raise ValueError("ordinary primary run carries ambiguous legacy retry evidence")
    required_fields = (
        (completion, {"retry_of_run_id", "retry_predecessor_binding_sha256"}),
        (
            input_bindings,
            {
                "retry_of_run_id",
                "retry_predecessor",
                "retry_predecessor_binding_sha256",
            },
        ),
        (provenance, {"retry_of_run_id", "retry_predecessor_binding_sha256"}),
        (metrics, {"retry_of_run_id", "retry_predecessor_binding_sha256"}),
    )
    if input_bindings.get("schema_version") != 1 or any(
        not fields.issubset(record) for record, fields in required_fields
    ):
        raise ValueError("ordinary primary retry binding fields are absent or invalid")

    retry_of_run_id = completion.get("retry_of_run_id")
    retry_ids = (
        retry_of_run_id,
        input_bindings.get("retry_of_run_id"),
        provenance.get("retry_of_run_id"),
        metrics.get("retry_of_run_id"),
    )
    bindings = (
        completion.get("retry_predecessor_binding_sha256"),
        input_bindings.get("retry_predecessor_binding_sha256"),
        provenance.get("retry_predecessor_binding_sha256"),
        metrics.get("retry_predecessor_binding_sha256"),
    )
    predecessor_evidence = input_bindings.get("retry_predecessor")
    if retry_of_run_id is None:
        if (
            any(value is not None for value in retry_ids)
            or any(value is not None for value in bindings)
            or predecessor_evidence is not None
        ):
            raise ValueError("ordinary non-retry primary carries inconsistent retry evidence")
        return

    if (
        not isinstance(retry_of_run_id, str)
        or not retry_of_run_id.strip()
        or any(value != retry_of_run_id for value in retry_ids)
    ):
        raise ValueError("ordinary primary retry run IDs are absent or inconsistent")
    binding_sha256 = _require_sha256(bindings[0], "primary retry predecessor binding")
    if any(value != binding_sha256 for value in bindings):
        raise ValueError("ordinary primary retry predecessor binding SHA-256 is inconsistent")
    evidence = _mapping(predecessor_evidence, "primary retry predecessor evidence")
    if evidence.get("binding_sha256") != binding_sha256:
        raise ValueError("ordinary primary retry evidence does not bind the claimed SHA-256")

    reconstructed = _reconstruct_primary_retry_predecessor_evidence(
        run_directory=run_directory,
        retry_of_run_id=retry_of_run_id,
        primary_gate=primary_gate,
    )
    if dict(evidence) != reconstructed:
        raise ValueError(
            "ordinary primary retry evidence differs from the exact sealed failed predecessor"
        )


def _validate_primary_run_identity(
    *,
    run_directory: Path,
    integrity: Any,
    primary_gate: PrimaryExecutionGateEvidence,
    completion: Mapping[str, Any],
    metrics: Mapping[str, Any],
    controls: Any,
) -> None:
    """Bind a sealed primary claim to its exact run, source, gate, and inputs."""

    run_id = integrity.run_id
    artifact_root = integrity.expected_root_sha256
    if (
        not isinstance(run_id, str)
        or not run_id
        or not _SHA256_PATTERN.fullmatch(str(artifact_root))
    ):
        raise ValueError("primary integrity evidence lacks an exact run/root identity")

    status = _read_mapping(run_directory / "status.json", "primary status")
    artifact_manifest = _read_mapping(
        run_directory / "artifact_manifest.json", "primary artifact manifest"
    )
    immutable_marker = _read_mapping(run_directory / ".immutable.json", "primary immutable marker")
    if any(
        record.get("run_id") != run_id for record in (status, artifact_manifest, immutable_marker)
    ):
        raise ValueError("primary sealed run identity is inconsistent")
    finalization_only_successor = completion.get("finalization_only_successor") is True
    recovery_only = completion.get("recovery_only") is True
    if finalization_only_successor and recovery_only:
        raise ValueError("primary run cannot claim finalization and orphan recovery together")
    expected_experiment = (
        _PRIMARY_FINALIZATION_SUCCESSOR_EXPERIMENT_NAME
        if finalization_only_successor
        else PRIMARY_RECOVERY_EXPERIMENT_NAME
        if recovery_only
        else _PRIMARY_EXPERIMENT_NAME
    )
    if status.get("experiment_name") != expected_experiment:
        raise ValueError("primary status names an unexpected experiment")
    if any(
        record.get("status") != "completed"
        for record in (status, artifact_manifest, immutable_marker)
    ):
        raise ValueError("primary confirmatory dependency is not a sealed completed run")
    if (
        artifact_manifest.get("artifact_root_sha256") != artifact_root
        or immutable_marker.get("artifact_root_sha256") != artifact_root
        or immutable_marker.get("artifact_manifest_sha256")
        != sha256_file(run_directory / "artifact_manifest.json")
    ):
        raise ValueError("primary sealed artifact identities differ from integrity evidence")
    marker_path = immutable_marker.get("run_path")
    if not isinstance(marker_path, str) or Path(marker_path).resolve() != run_directory:
        raise ValueError("primary immutable marker names a different run path")

    source_manifest_path = run_directory / "source_tree_manifest.json"
    source_manifest = _validate_execution_source_manifest(
        source_manifest_path, "primary run source manifest"
    )
    provenance = _read_mapping(run_directory / "run_provenance.json", "primary provenance")
    provenance_source = _mapping(provenance.get("source_tree"), "primary provenance source tree")
    if (
        source_manifest.get("root_sha256") != primary_gate.source_tree_root_sha256
        or completion.get("source_tree_root_sha256") != primary_gate.source_tree_root_sha256
        or provenance_source.get("root_sha256") != primary_gate.source_tree_root_sha256
        or provenance_source.get("manifest") != "source_tree_manifest.json"
        or provenance_source.get("manifest_sha256") != sha256_file(source_manifest_path)
    ):
        raise ValueError("primary source identity differs from the gated execution source")
    if (
        provenance.get("run_id") != run_id
        or provenance.get("experiment_name") != expected_experiment
        or completion.get("run_id") != run_id
        or metrics.get("run_id") != run_id
    ):
        raise ValueError("primary completion/provenance run identity is inconsistent")

    execution_gate_path = run_directory / "primary_execution_gate.json"
    input_bindings_path = run_directory / "primary_input_bindings.json"
    execution_gate = _read_mapping(execution_gate_path, "primary execution gate artifact")
    input_bindings = _read_mapping(input_bindings_path, "primary input bindings")
    if execution_gate != primary_gate.as_dict():
        raise ValueError("primary run does not bind the exact live execution gate")
    if recovery_only:
        recovery_input_path = run_directory / "primary_recovery_input_bindings.json"
        recovery_input = _read_mapping(
            recovery_input_path,
            "primary orphan-recovery input bindings",
        )
        expected_recovery_input = {
            "schema_version": 1,
            "policy": _PRIMARY_RECOVERY_POLICY,
            "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
            "source_run_id": completion.get("retry_of_run_id"),
            "recovery_authorization_sha256": completion.get("recovery_authorization_sha256"),
            "authority_directory": str(primary_gate.freeze_directory),
            "authority_artifact_root_sha256": primary_gate.freeze_artifact_root_sha256,
            "authority_manifest_sha256": primary_gate.freeze_manifest_sha256,
            "source_snapshot_root_sha256": completion.get("recovery_source_snapshot_root_sha256"),
            "matrix_plan_sha256": controls.plan_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "primary_gate_sha256": canonical_sha256(primary_gate.as_dict()),
            "analysis_disposition": _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS,
            "outcomes_inspected": True,
            "training_invoked": False,
            "matrix_executor_invoked": False,
            "fallback_invoked": False,
            "automatic_retry_allowed": False,
            "source_annotations_modified": False,
            "planned_cell_count": len(controls.plan.cells),
            "required_cell_count": controls.plan.required_cell_count,
        }
        if (
            provenance.get("primary_execution_gate_sha256") != sha256_file(execution_gate_path)
            or provenance.get("primary_recovery_input_bindings_sha256")
            != sha256_file(recovery_input_path)
            or any(
                recovery_input.get(field) != expected
                for field, expected in expected_recovery_input.items()
            )
        ):
            raise ValueError("primary orphan-recovery inputs differ from the frozen execution")
    elif provenance.get("primary_gate") != dict(execution_gate):
        raise ValueError("primary run does not bind the exact live execution gate")
    if (
        (
            not finalization_only_successor
            and not recovery_only
            and provenance.get("primary_input_bindings_sha256") != sha256_file(input_bindings_path)
        )
        or input_bindings.get("config_semantic_sha256")
        != primary_gate.primary_config_semantic_sha256
        or input_bindings.get("plan_semantic_sha256") != controls.plan_sha256
        or input_bindings.get("execution_controls_binding_sha256") != controls.binding_sha256
        or provenance.get("retry_of_run_id") != completion.get("retry_of_run_id")
        or (
            not finalization_only_successor
            and not recovery_only
            and input_bindings.get("retry_of_run_id") != completion.get("retry_of_run_id")
        )
    ):
        raise ValueError("primary input/provenance bindings differ from the frozen execution")
    if not finalization_only_successor and not recovery_only:
        _validate_primary_full_retry_binding(
            run_directory=run_directory,
            primary_gate=primary_gate,
            completion=completion,
            metrics=metrics,
            provenance=provenance,
            input_bindings=input_bindings,
        )


def _validate_primary_statistics_stage_metadata(
    *,
    run_directory: Path,
    controls: Any,
    matrix_readback: Any,
    completion: Mapping[str, Any],
    metrics: Mapping[str, Any],
    primary_stage_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate untrusted metadata but never mint typed stage authority."""

    attestation = _mapping(primary_stage_attestation, "primary post-seal attestation")
    verification = _mapping(
        attestation.get("verification"), "primary post-seal statistics verification"
    )
    try:
        verification_sha256 = hashlib.sha256(
            json.dumps(
                dict(verification),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("primary post-seal verification is not canonical JSON") from exc
    record_sha256 = attestation.get("record_sha256")
    if (
        attestation.get("scientific_stage_eligible") is not True
        or attestation.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or verification.get("semantic_verification_status") != "passed"
        or verification.get("run_id") != run_directory.name
        or verification.get("run_path") != str(run_directory)
        or attestation.get("verification_sha256") != verification_sha256
        or not isinstance(record_sha256, str)
        or _SHA256_PATTERN.fullmatch(record_sha256) is None
    ):
        raise ValueError("primary post-seal statistics attestation is invalid")

    comparison_families = (
        getattr(controls, "within_cell_comparisons", None),
        getattr(controls, "method_vs_random_comparisons", None),
        getattr(controls, "cross_cell_comparisons", None),
    )
    comparison_count = 0
    for family in comparison_families:
        if not isinstance(family, tuple):
            raise ValueError("frozen primary controls lack exact statistics comparison families")
        comparison_count += len(family)
    source_root = matrix_readback.readback_root_sha256
    if (
        verification.get("filesystem_readback_root_sha256") != source_root
        or verification.get("primary_statistics_source_readback_root_sha256") != source_root
        or verification.get("primary_statistics_comparison_count") != comparison_count
    ):
        raise ValueError("primary post-seal statistics attestation differs from frozen controls")

    files = {
        "primary_statistics.json": (
            "primary_statistics_sha256",
            "primary_statistics_size_bytes",
        ),
        "primary_bootstrap_evidence.npz": (
            "primary_bootstrap_evidence_sha256",
            "primary_bootstrap_evidence_size_bytes",
        ),
        "primary_subgroups.csv": (
            "primary_subgroups_sha256",
            "primary_subgroups_size_bytes",
        ),
        "primary_statistics_manifest.json": (
            "primary_statistics_manifest_sha256",
            "primary_statistics_manifest_size_bytes",
        ),
    }
    for name, (sha_field, size_field) in files.items():
        path = run_directory / name
        expected_sha = verification.get(sha_field)
        expected_size = verification.get(size_field)
        if (
            not isinstance(expected_sha, str)
            or _SHA256_PATTERN.fullmatch(expected_sha) is None
            or type(expected_size) is not int
            or int(expected_size) < 0
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"primary post-seal statistics file binding changed: {name}")

    expected_completion = {
        "primary_statistics_verification_status": "passed",
        "primary_statistics_sha256": verification["primary_statistics_sha256"],
        "primary_bootstrap_evidence_sha256": verification["primary_bootstrap_evidence_sha256"],
        "primary_subgroups_sha256": verification["primary_subgroups_sha256"],
        "primary_statistics_manifest_sha256": verification["primary_statistics_manifest_sha256"],
        "primary_statistics_source_readback_root_sha256": source_root,
        "primary_statistics_comparison_count": comparison_count,
    }
    if any(
        completion.get(field_name) != value for field_name, value in expected_completion.items()
    ):
        raise ValueError("primary completion differs from post-seal statistics attestation")
    if (
        metrics.get("primary_statistics_manifest_sha256")
        != verification["primary_statistics_manifest_sha256"]
        or metrics.get("primary_statistics_comparison_count") != comparison_count
    ):
        raise ValueError("primary metrics differ from post-seal statistics attestation")

    manifest = _read_mapping(
        run_directory / "primary_statistics_manifest.json", "primary statistics manifest"
    )
    if set(manifest) != _PRIMARY_STATISTICS_MANIFEST_FIELDS:
        raise ValueError("primary statistics manifest has missing or extra fields")
    source_cell_hashes = manifest.get("source_cell_artifact_manifest_sha256")
    expected_source_cell_hashes = dict(
        sorted(dict(matrix_readback.cell_artifact_manifest_sha256).items())
    )
    input_bindings_path = run_directory / "primary_input_bindings.json"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("execution_controls_binding_sha256") != controls.binding_sha256
        or manifest.get("source_filesystem_readback_root_sha256") != source_root
        or source_cell_hashes != expected_source_cell_hashes
        or input_bindings_path.is_symlink()
        or not input_bindings_path.is_file()
        or manifest.get("primary_input_bindings_sha256") != sha256_file(input_bindings_path)
    ):
        raise ValueError("primary statistics manifest differs from its sealed inputs")
    for hash_field in (
        "crop_cache_sha256",
        "statistics_payload_sha256",
        "subgroup_rows_sha256",
    ):
        value = manifest.get(hash_field)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"primary statistics manifest has invalid {hash_field}")

    records = manifest.get("artifacts")
    expected_record_names = set(files).difference({"primary_statistics_manifest.json"})
    seen: set[str] = set()
    if not isinstance(records, list) or len(records) != len(expected_record_names):
        raise ValueError("primary statistics manifest has an invalid artifact list")
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("primary statistics manifest has an invalid artifact record")
        artifact_name = raw_record.get("path")
        if (
            not isinstance(artifact_name, str)
            or artifact_name not in expected_record_names
            or artifact_name in seen
        ):
            raise ValueError("primary statistics manifest has an unexpected artifact")
        sha_field, size_field = files[artifact_name]
        if (
            raw_record.get("sha256") != verification[sha_field]
            or raw_record.get("size_bytes") != verification[size_field]
        ):
            raise ValueError(f"primary statistics manifest record differs for {artifact_name}")
        seen.add(artifact_name)
    if seen != expected_record_names:
        raise ValueError("primary statistics manifest lacks a required artifact")

    return {
        "status": "passed_postseal_attestation_metadata",
        "output_directory": run_directory,
        "statistics_sha256": str(verification["primary_statistics_sha256"]),
        "bootstrap_evidence_sha256": str(verification["primary_bootstrap_evidence_sha256"]),
        "subgroups_sha256": str(verification["primary_subgroups_sha256"]),
        "manifest_sha256": str(verification["primary_statistics_manifest_sha256"]),
        "source_readback_root_sha256": str(source_root),
        "comparison_count": comparison_count,
        "stage_attestation_record_sha256": record_sha256,
        "stage_attestation_verification_sha256": verification_sha256,
    }


def _primary_statistics_stage_readback(
    *,
    run_directory: Path,
    controls: Any,
    matrix_readback: Any,
    completion: Mapping[str, Any],
    metrics: Mapping[str, Any],
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt,
) -> PrimaryStatisticsStageReadback:
    """Mint typed hash-only readback solely from genuine run-tracking authority."""

    if (
        not isinstance(primary_stage_eligibility_receipt, RunStageEligibilityReceipt)
        or not primary_stage_eligibility_receipt.valid
    ):
        raise ValueError("primary statistics require a genuine run-tracking eligibility receipt")
    if (
        primary_stage_eligibility_receipt.run_directory.resolve() != run_directory
        or primary_stage_eligibility_receipt.run_id != run_directory.name
        or primary_stage_eligibility_receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
    ):
        raise ValueError("primary eligibility receipt is bound to a different sealed run")
    metadata = _validate_primary_statistics_stage_metadata(
        run_directory=run_directory,
        controls=controls,
        matrix_readback=matrix_readback,
        completion=completion,
        metrics=metrics,
        primary_stage_attestation=primary_stage_eligibility_receipt.attestation_record(),
    )
    if (
        metadata["stage_attestation_record_sha256"]
        != primary_stage_eligibility_receipt.record_sha256
        or metadata["stage_attestation_verification_sha256"]
        != primary_stage_eligibility_receipt.verification_sha256
    ):
        raise ValueError("primary statistics metadata differs from its eligibility receipt")
    readback = PrimaryStatisticsStageReadback(
        status="passed_postseal_attestation_readback",
        output_directory=metadata["output_directory"],
        statistics_sha256=metadata["statistics_sha256"],
        bootstrap_evidence_sha256=metadata["bootstrap_evidence_sha256"],
        subgroups_sha256=metadata["subgroups_sha256"],
        manifest_sha256=metadata["manifest_sha256"],
        source_readback_root_sha256=metadata["source_readback_root_sha256"],
        comparison_count=metadata["comparison_count"],
        stage_attestation_record_sha256=metadata["stage_attestation_record_sha256"],
        stage_attestation_verification_sha256=metadata["stage_attestation_verification_sha256"],
    )
    object.__setattr__(readback, "_attestation", _PRIMARY_STATISTICS_STAGE_READBACK_ATTESTATION)
    return readback


def _validate_primary_completion_attestations(
    *,
    run_directory: Path,
    primary_gate: PrimaryExecutionGateEvidence,
    completion: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    integrity: Any,
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt,
) -> tuple[Any, Any, Any]:
    """Bind matrix/restoration readbacks and the exact post-seal statistics receipt."""

    config_path = primary_gate.freeze_directory / "primary_frozen.yaml"
    if (
        not config_path.is_file()
        or sha256_file(config_path) != primary_gate.frozen_primary_config_sha256
    ):
        raise ValueError("primary run gate no longer resolves its exact frozen config")
    controls = primary_execution_controls_from_frozen_config(load_config(config_path))
    if (
        controls.config_semantic_sha256 != primary_gate.primary_config_semantic_sha256
        or len(controls.plan.cells) != primary_gate.primary_matrix_cell_count
        or controls.plan.required_cell_count != primary_gate.primary_required_cell_count
    ):
        raise ValueError("primary controls differ from the gated frozen matrix")

    matrix_readback = read_primary_filesystem_evidence(controls.plan, run_directory)
    if not matrix_readback.passed:
        raise ValueError("primary matrix lacks a passed typed filesystem attestation")
    if canonical_sha256(reconciliation) != canonical_sha256(
        matrix_readback.reconciliation.as_dict()
    ):
        raise ValueError("primary reconciliation differs from strict matrix readback")

    statistics = _primary_statistics_stage_readback(
        run_directory=run_directory,
        controls=controls,
        matrix_readback=matrix_readback,
        completion=completion,
        metrics=metrics,
        primary_stage_eligibility_receipt=primary_stage_eligibility_receipt,
    )
    if not statistics.valid:
        raise ValueError("primary statistics lack a passed typed verification")
    if (
        statistics.output_directory.resolve() != run_directory
        or statistics.source_readback_root_sha256 != matrix_readback.readback_root_sha256
    ):
        raise ValueError("primary statistics are bound to a different matrix readback")

    restoration = read_primary_restoration_evidence(run_directory, controls)
    if not restoration.passed:
        raise ValueError("primary restoration lacks a passed typed readback")
    if (
        restoration.run_directory.resolve() != run_directory
        or restoration.source_readback_root_sha256 != matrix_readback.readback_root_sha256
    ):
        raise ValueError("primary restoration is bound to a different matrix readback")

    expected_completion = {
        "matrix_config_sha256": controls.config_semantic_sha256,
        "planned_cell_count": len(controls.plan.cells),
        "required_cell_count": controls.plan.required_cell_count,
        "completed_required_cell_count": matrix_readback.completed_required_cell_count,
        "skipped_optional_cell_count": matrix_readback.skipped_optional_cell_count,
        "failed_required_cell_count": 0,
        "reconciliation_status": "passed",
        "filesystem_run_directory": str(run_directory),
        "filesystem_matrix_plan_sha256": matrix_readback.matrix_plan_sha256,
        "filesystem_execution_controls_sha256": matrix_readback.execution_controls_sha256,
        "filesystem_execution_controls_binding_sha256": (
            matrix_readback.execution_controls_binding_sha256
        ),
        "execution_controls_binding_sha256": controls.binding_sha256,
        "filesystem_cell_index_sha256": matrix_readback.cell_index_sha256,
        "filesystem_readback_root_sha256": matrix_readback.readback_root_sha256,
        "filesystem_completed_cell_count": matrix_readback.completed_cell_count,
        "circularity_excluded_cell_count": len(matrix_readback.circularity_excluded_cell_ids),
        "circularity_excluded_cell_ids": list(matrix_readback.circularity_excluded_cell_ids),
        "primary_confirmatory_claims_require_exclusion_of_these_cells": bool(
            matrix_readback.circularity_excluded_cell_ids
        ),
        "filesystem_scenario_corruption_sha256": dict(matrix_readback.scenario_corruption_sha256),
        "primary_statistics_verification_status": "passed",
        "primary_statistics_sha256": statistics.statistics_sha256,
        "primary_bootstrap_evidence_sha256": statistics.bootstrap_evidence_sha256,
        "primary_subgroups_sha256": statistics.subgroups_sha256,
        "primary_statistics_manifest_sha256": statistics.manifest_sha256,
        "primary_statistics_comparison_count": statistics.comparison_count,
        "primary_statistics_source_readback_root_sha256": (statistics.source_readback_root_sha256),
        "primary_restoration_verification_status": "passed",
        "primary_restoration_index_sha256": restoration.restoration_index_sha256,
        "primary_restoration_readback_root_sha256": restoration.readback_root_sha256,
        "primary_restoration_source_readback_root_sha256": (
            restoration.source_readback_root_sha256
        ),
        "primary_restoration_cell_count": restoration.restoration_cell_count,
        "primary_restoration_downstream_comparison_count": (
            restoration.downstream_comparison_count
        ),
    }
    for field, expected in expected_completion.items():
        if completion.get(field) != expected:
            raise ValueError(f"primary completion attestation {field} is absent or stale")

    restoration_bindings: dict[str, str] = {}
    for role, hashes in (
        ("json", restoration.cell_json_sha256),
        ("evidence", restoration.cell_evidence_sha256),
        ("manifest", restoration.cell_manifest_sha256),
    ):
        for cell_id, digest in hashes:
            restoration_bindings[f"primary_restoration_{role}_sha256::{cell_id}"] = digest
    actual_restoration_binding_keys = {
        key
        for key in completion
        if isinstance(key, str) and key.startswith("primary_restoration_") and "_sha256::" in key
    }
    if actual_restoration_binding_keys != set(restoration_bindings) or any(
        completion.get(field) != expected for field, expected in restoration_bindings.items()
    ):
        raise ValueError("primary completion lacks the exact per-cell restoration bindings")

    if completion.get("schema_version") not in {1, 2}:
        raise ValueError("primary completion uses an unsupported schema version")
    if (
        completion.get("completion_stage_enabled_only_after_run_seal_and_integrity_verification")
        is not True
        or completion.get("post_seal_integrity_verification_required") is not True
    ):
        raise ValueError("primary completion lacks mandatory seal/integrity attestations")

    _validate_primary_run_identity(
        run_directory=run_directory,
        integrity=integrity,
        primary_gate=primary_gate,
        completion=completion,
        metrics=metrics,
        controls=controls,
    )
    if (
        metrics.get("matrix_config_sha256") != controls.config_semantic_sha256
        or metrics.get("matrix_plan_sha256") != controls.plan_sha256
        or metrics.get("execution_controls_binding_sha256") != controls.binding_sha256
        or metrics.get("filesystem_readback_root_sha256") != matrix_readback.readback_root_sha256
        or metrics.get("primary_statistics_manifest_sha256") != statistics.manifest_sha256
        or metrics.get("completion_evidence_sha256")
        != sha256_file(run_directory / "completion_evidence.json")
    ):
        raise ValueError("primary metrics do not bind the exact attested completion")
    return matrix_readback, statistics, restoration


def _validate_primary_recovery_downstream_identity(
    *,
    run_directory: Path,
    primary_gate: PrimaryExecutionGateEvidence,
    completion: Mapping[str, Any],
    metrics: Mapping[str, Any],
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt,
    recovery_experiment: bool,
) -> _PrimaryRecoveryDownstreamIdentity | None:
    """Admit only the exact post-outcome recovery lineage to confirmatory use."""

    evidence_path = run_directory / _PRIMARY_RECOVERY_EVIDENCE_FILENAME
    recovery_completion_fields = {
        "recovery_only",
        "recovery_policy",
        "primary_recovery_evidence_sha256",
        "recovery_evidence_sha256",
        "recovery_authorization_sha256",
        "recovery_source_snapshot_root_sha256",
    }
    if not recovery_experiment:
        if evidence_path.exists() or recovery_completion_fields.intersection(completion):
            raise ValueError("ordinary primary cannot carry orphan-recovery lineage")
        return None

    optional_cell_count = (
        primary_gate.primary_matrix_cell_count - primary_gate.primary_required_cell_count
    )
    if (
        primary_gate.registration_authority_kind != "preregistration_amendment"
        or primary_gate.registration_status != _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
        or primary_gate.original_unamended_primary_claim_allowed
        or primary_gate.amended_primary_claim_allowed
        or primary_gate.primary_required_cell_count != _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT
        or optional_cell_count != _PRIMARY_RECOVERY_SKIPPED_OPTIONAL_CELL_COUNT
    ):
        raise ValueError(
            "primary orphan recovery lacks its exact exploratory amendment and 185/37 matrix"
        )
    if (
        not isinstance(primary_stage_eligibility_receipt, RunStageEligibilityReceipt)
        or not primary_stage_eligibility_receipt.valid
        or primary_stage_eligibility_receipt.run_directory.resolve() != run_directory
        or primary_stage_eligibility_receipt.run_id != run_directory.name
        or primary_stage_eligibility_receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
    ):
        raise ValueError("primary orphan recovery lacks a genuine positive stage receipt")

    from histo_audit.experiment.primary_recovery import RecoveryAuthorization

    authorization_mapping = require_primary_recovery_authorization(primary_gate.freeze_directory)
    authorization = RecoveryAuthorization.from_mapping(
        authorization_mapping,
        authority_directory=primary_gate.freeze_directory,
        authority_artifact_root_sha256=primary_gate.freeze_artifact_root_sha256,
        authority_manifest_sha256=primary_gate.freeze_manifest_sha256,
    )
    if (
        authorization.authority_directory != primary_gate.freeze_directory.resolve()
        or authorization.authority_artifact_root_sha256 != primary_gate.freeze_artifact_root_sha256
        or authorization.authority_manifest_sha256 != primary_gate.freeze_manifest_sha256
        or authorization.analysis_disposition != _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
        or authorization.outcomes_inspected is not True
    ):
        raise ValueError("primary orphan recovery authorization differs from its amendment")

    evidence = _read_mapping(evidence_path, "primary orphan-recovery evidence")
    provenance = _read_mapping(
        run_directory / "run_provenance.json",
        "primary orphan-recovery provenance",
    )
    evidence_sha256 = sha256_file(evidence_path)
    source_run_id = authorization.source_run_id
    source_snapshot_root = authorization.expected_source_snapshot_root_sha256
    proof_sha256 = evidence.get("prior_numeric_verification_proof_sha256")
    copied_artifact_count = evidence.get("copied_artifact_count")
    copied_total_bytes = evidence.get("copied_total_bytes")
    if (
        set(evidence) != _PRIMARY_RECOVERY_EVIDENCE_FIELDS
        or not isinstance(proof_sha256, str)
        or _SHA256_PATTERN.fullmatch(proof_sha256) is None
        or type(copied_artifact_count) is not int
        or copied_artifact_count < 1
        or type(copied_total_bytes) is not int
        or copied_total_bytes < 1
    ):
        raise ValueError("primary orphan-recovery evidence has an invalid closed schema")

    expected_evidence = {
        "schema_version": 1,
        "policy": _PRIMARY_RECOVERY_POLICY,
        "experiment_name": PRIMARY_RECOVERY_EXPERIMENT_NAME,
        "source_run_id": source_run_id,
        "destination_run_id": run_directory.name,
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "source_snapshot_root_sha256": source_snapshot_root,
        "destination_snapshot_root_sha256": source_snapshot_root,
        "reused_required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "skipped_optional_cell_count": _PRIMARY_RECOVERY_SKIPPED_OPTIONAL_CELL_COUNT,
        "retrained_cell_count": 0,
        "copy_policy": RECOVERY_COPY_POLICY,
        "copied_artifact_count": copied_artifact_count,
        "copied_total_bytes": copied_total_bytes,
        "verification_mode": _INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "analysis_disposition": _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if dict(evidence) != expected_evidence:
        raise ValueError("primary orphan-recovery evidence differs from its exact authorization")

    expected_completion = {
        "recovery_only": True,
        "recovery_policy": _PRIMARY_RECOVERY_POLICY,
        "primary_recovery_evidence_sha256": evidence_sha256,
        "recovery_evidence_sha256": evidence_sha256,
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "recovery_source_snapshot_root_sha256": source_snapshot_root,
        "retry_of_run_id": source_run_id,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": authorization.outcome_inspection_at_utc,
        "verification_mode": _INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "completed_required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "reused_required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "skipped_optional_cell_count": _PRIMARY_RECOVERY_SKIPPED_OPTIONAL_CELL_COUNT,
        "retrained_cell_count": 0,
        "physical_copy_verified": True,
        "copy_policy": RECOVERY_COPY_POLICY,
        "copied_artifact_count": copied_artifact_count,
        "copied_total_bytes": copied_total_bytes,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if any(completion.get(field) != value for field, value in expected_completion.items()):
        raise ValueError("primary orphan recovery completion differs from its sealed lineage")

    expected_provenance = {
        key: value
        for key, value in expected_completion.items()
        if key
        not in {
            "required_cell_count",
            "completed_required_cell_count",
        }
    }
    if any(provenance.get(field) != value for field, value in expected_provenance.items()):
        raise ValueError("primary orphan recovery provenance differs from its sealed lineage")

    expected_metrics = {
        "recovery_only": True,
        "recovery_policy": _PRIMARY_RECOVERY_POLICY,
        "retry_of_run_id": source_run_id,
        "retry_predecessor_binding_sha256": evidence_sha256,
        "analysis_disposition": _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "completed_required_cell_count": _PRIMARY_RECOVERY_REQUIRED_CELL_COUNT,
        "skipped_optional_cell_count": _PRIMARY_RECOVERY_SKIPPED_OPTIONAL_CELL_COUNT,
        "physical_copy_verified": True,
        "copy_policy": RECOVERY_COPY_POLICY,
        "copied_artifact_count": copied_artifact_count,
        "copied_total_bytes": copied_total_bytes,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if any(metrics.get(field) != value for field, value in expected_metrics.items()):
        raise ValueError("primary orphan recovery metrics differ from its sealed lineage")

    forbidden_predecessor_fields = {
        "finalization_only_successor",
        "finalization_successor_authorization_sha256",
        "finalization_successor_evidence_sha256",
        "predecessor_artifact_root_sha256",
        "predecessor_artifact_manifest_sha256",
        "predecessor_source_tree_root_sha256",
    }
    if (
        forbidden_predecessor_fields.intersection(completion)
        or forbidden_predecessor_fields.intersection(provenance)
        or forbidden_predecessor_fields.intersection(evidence)
    ):
        raise ValueError("primary orphan recovery cannot claim a sealed predecessor")

    stage_record = primary_stage_eligibility_receipt.attestation_record()
    stage_verification = _mapping(
        stage_record.get("verification"),
        "primary orphan-recovery stage verification",
    )
    if (
        stage_record.get("run_id") != run_directory.name
        or stage_record.get("run_path") != str(run_directory)
        or stage_record.get("terminal_status") != "completed"
        or stage_record.get("scientific_stage_eligible") is not True
        or stage_record.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or stage_record.get("completion_evidence_sha256")
        != sha256_file(run_directory / "completion_evidence.json")
        or stage_verification.get("policy") != _PRIMARY_RECOVERY_STAGE_POLICY
        or stage_verification.get("experiment_name") != PRIMARY_RECOVERY_EXPERIMENT_NAME
        or stage_verification.get("run_id") != run_directory.name
        or stage_verification.get("run_path") != str(run_directory)
        or stage_verification.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or stage_verification.get("retry_of_run_id") != source_run_id
        or stage_verification.get("lineage_binding_sha256") != evidence_sha256
        or stage_verification.get("authorization_binding_sha256") != authorization.canonical_sha256
        or stage_verification.get("semantic_verification_status") != "passed"
    ):
        raise ValueError("primary orphan recovery lacks its exact positive stage attestation")

    return _PrimaryRecoveryDownstreamIdentity(
        evidence_sha256=evidence_sha256,
        authorization_sha256=authorization.canonical_sha256,
        source_run_id=source_run_id,
        source_snapshot_root_sha256=source_snapshot_root,
        analysis_disposition=_POST_OUTCOME_RECOVERY_REGISTRATION_STATUS,
    )


def _validate_primary_finalization_successor(
    *,
    run_directory: Path,
    completion: Mapping[str, Any],
    primary_gate: PrimaryExecutionGateEvidence,
) -> tuple[bool, str | None, str | None, str | None]:
    """Require the one canonical sealed-failed-predecessor lineage for finalization only."""

    flag = completion.get("finalization_only_successor")
    evidence_path = run_directory / _PRIMARY_FINALIZATION_SUCCESSOR_FILENAME
    legacy_evidence_path = run_directory / "primary_retry_predecessor.json"
    if flag is None or flag is False:
        if evidence_path.exists():
            raise ValueError(
                "ordinary primary completion cannot carry unclaimed finalization-successor evidence"
            )
        return False, None, None, None
    if flag is not True:
        raise ValueError("primary finalization_only_successor must be an exact boolean")
    if legacy_evidence_path.exists():
        raise ValueError("finalization-only primary successor uses ambiguous legacy retry evidence")
    if not evidence_path.is_file():
        raise ValueError("finalization-only primary successor lacks its sealed lineage evidence")
    evidence_sha = sha256_file(evidence_path)
    if completion.get(
        "finalization_successor_evidence_sha256"
    ) != evidence_sha or not _SHA256_PATTERN.fullmatch(evidence_sha):
        raise ValueError("primary successor completion does not bind its exact lineage evidence")

    # Imported lazily: the producer imports this gate for its own mandatory live
    # revalidation, so a module-level import would create a circular dependency.
    verify_primary_finalization_successor_evidence = getattr(
        importlib.import_module("histo_audit.experiment.primary_finalization_successor"),
        "verify_primary_finalization_successor_evidence",
        None,
    )
    if not callable(verify_primary_finalization_successor_evidence):
        raise ValueError("the retired finalization-successor verifier is unavailable")

    evidence = _read_mapping(evidence_path, "primary finalization-successor evidence")
    verified = verify_primary_finalization_successor_evidence(run_directory, completion)
    if not isinstance(verified, Mapping):
        raise ValueError("primary finalization-successor verifier returned invalid evidence")
    inherited_evidence = (
        completion.get("verification_mode") == _INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
    )
    if (
        evidence.get("schema_version") != (2 if inherited_evidence else 1)
        or evidence.get("policy") != _PRIMARY_FINALIZATION_SUCCESSOR_POLICY
        or dict(verified) != dict(evidence)
    ):
        raise ValueError("primary finalization-successor evidence uses an invalid policy")
    predecessor = _mapping(
        evidence.get("predecessor"), "primary finalization-successor predecessor"
    )
    predecessor_run_id = predecessor.get("run_id")
    predecessor_root = predecessor.get("artifact_root_sha256")
    if (
        not isinstance(predecessor_run_id, str)
        or not predecessor_run_id
        or completion.get("retry_of_run_id") != predecessor_run_id
        or completion.get("predecessor_artifact_root_sha256") != predecessor_root
        or not isinstance(predecessor_root, str)
        or _SHA256_PATTERN.fullmatch(predecessor_root) is None
    ):
        raise ValueError("primary successor completion has inconsistent predecessor identity")

    from histo_audit.workflows.preregistration_amendment import (
        require_finalization_successor_authorization,
    )

    authorization = require_finalization_successor_authorization(primary_gate.freeze_directory)
    authorization_sha = canonical_sha256(dict(authorization))
    successor_provenance = _read_mapping(
        run_directory / "run_provenance.json", "primary successor provenance"
    )
    authorized_predecessor = _mapping(
        authorization.get("predecessor"), "finalization-successor authorization predecessor"
    )
    authorized_reuse = _mapping(
        authorization.get("reuse"), "finalization-successor authorization reuse"
    )
    verification_mode = completion.get("verification_mode")
    inherited_mode = verification_mode == _INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
    expected_authorization_policy = (
        _PRIMARY_FINALIZATION_AUTHORIZATION_POLICY_V2
        if inherited_mode
        else _PRIMARY_FINALIZATION_AUTHORIZATION_POLICY_V1
    )
    authorized_numeric = (
        _mapping(
            authorization.get("numeric_verification"),
            "finalization-successor numeric verification",
        )
        if inherited_mode
        else None
    )
    optional_count = (
        primary_gate.primary_matrix_cell_count - primary_gate.primary_required_cell_count
    )
    if (
        primary_gate.registration_authority_kind != "preregistration_amendment"
        or authorization.get("policy") != expected_authorization_policy
        or authorization.get("outcomes_inspected") is not False
        or authorized_predecessor.get("run_id") != predecessor_run_id
        or authorized_predecessor.get("artifact_root_sha256") != predecessor_root
        or authorized_predecessor.get("artifact_manifest_sha256")
        != predecessor.get("artifact_manifest_sha256")
        or authorized_predecessor.get("execution_source_root_sha256")
        != predecessor.get("source_tree_root_sha256")
        or authorized_reuse.get("reused_required_cell_count")
        != primary_gate.primary_required_cell_count
        or authorized_reuse.get("reused_optional_cell_count") != optional_count
        or authorized_reuse.get("retrained_cell_count") != 0
        or completion.get("finalization_successor_authorization_sha256") != authorization_sha
        or successor_provenance.get("finalization_successor_authorization_sha256")
        != authorization_sha
        or (
            inherited_mode
            and (
                evidence.get("verification_mode") != verification_mode
                or evidence.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
                or authorized_numeric is None
                or authorized_numeric.get("mode") != verification_mode
                or authorized_numeric.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
                or successor_provenance.get("verification_mode") != verification_mode
                or successor_provenance.get("prior_numeric_verification_proof_sha256")
                != completion.get("prior_numeric_verification_proof_sha256")
            )
        )
        or (
            not inherited_mode
            and (
                "numeric_verification" in authorization
                or "verification_mode" in evidence
                or "verification_mode" in completion
            )
        )
    ):
        raise ValueError(
            "primary finalization successor differs from its sealed amendment authorization"
        )
    return True, evidence_sha, predecessor_run_id, predecessor_root


def validate_historical_primary_dependency(
    *,
    primary_run_directory: str | Path,
    project_root: str | Path,
    recovery_authority_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt | None = None,
) -> HistoricalPrimaryDependencyEvidence:
    """Authenticate a completed recovery primary under historical authority P.

    P is used only to reconstruct immutable primary dependencies and to validate the
    sealed run's own source manifest.  This function intentionally never compares P's
    source root with the current project checkout.
    """

    run_directory = Path(primary_run_directory).resolve()
    integrity = verify_run_integrity(run_directory)
    if not integrity.valid or not integrity.registry_record_present:
        raise ValueError(f"historical primary failed integrity verification: {integrity.errors}")
    if primary_stage_eligibility_receipt is None:
        eligibility_receipt = require_run_stage_eligibility_receipt(
            run_directory,
            integrity=integrity,
        )
    else:
        eligibility_receipt = primary_stage_eligibility_receipt
        if not isinstance(eligibility_receipt, RunStageEligibilityReceipt):
            raise ValueError(
                "supplied historical-primary receipt is not active under its mutation guard"
            )
        try:
            eligibility_receipt.require_active_authority()
        except ValueError as exc:
            raise ValueError(
                "supplied historical-primary receipt is not active under its mutation guard"
            ) from exc
    if (
        not isinstance(eligibility_receipt, RunStageEligibilityReceipt)
        or not eligibility_receipt.valid
        or eligibility_receipt.run_directory.resolve() != run_directory
        or eligibility_receipt.run_id != run_directory.name
        or eligibility_receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
    ):
        raise ValueError(
            "historical primary lacks its exact positive PRIMARY_STUDY_COMPLETE "
            "post-seal eligibility receipt"
        )
    status = _read_mapping(run_directory / "status.json", "historical primary status")
    if status.get("experiment_name") != PRIMARY_RECOVERY_EXPERIMENT_NAME:
        raise ValueError("resource-bounded execution requires the exact completed recovery primary")
    primary_gate = _validate_registered_primary_dependencies(
        project_root=project_root,
        freeze_directory=recovery_authority_directory,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        experiment_name=PRIMARY_RECOVERY_EXPERIMENT_NAME,
    )
    for artifact in (
        "completion_evidence.json",
        "reconciliation.json",
        "matrix_plan.json",
        "metrics.json",
        "report.md",
    ):
        if not (run_directory / artifact).is_file():
            raise ValueError(
                f"historical primary completion evidence lacks required artifact: {artifact}"
            )
    completion = _read_mapping(
        run_directory / "completion_evidence.json",
        "historical primary completion",
    )
    reconciliation = _read_mapping(
        run_directory / "reconciliation.json",
        "historical primary reconciliation",
    )
    metrics = _read_mapping(
        run_directory / "metrics.json",
        "historical primary metrics",
    )
    if (
        completion.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or completion.get("study_outcome_eligible") is not True
        or completion.get("artifact_scope") != "real_pannuke_primary_study"
    ):
        raise ValueError(
            "historical primary completion is not eligible PRIMARY_STUDY_COMPLETE evidence"
        )
    if (
        metrics.get("artifact_scope") != "real_pannuke_primary_study"
        or metrics.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or metrics.get("study_outcome_eligible") is not True
    ):
        raise ValueError("historical primary metrics carry an ineligible claim")
    if reconciliation.get("status") != "passed":
        raise ValueError("historical primary artifact reconciliation did not pass")
    required_count = completion.get("required_cell_count")
    completed_count = completion.get("completed_required_cell_count")
    if (
        type(required_count) is not int
        or required_count != primary_gate.primary_required_cell_count
        or completed_count != required_count
        or completion.get("failed_required_cell_count") != 0
    ):
        raise ValueError("historical primary completion differs from the frozen required-cell set")
    expected_bindings = {
        "freeze_artifact_root_sha256": primary_gate.freeze_artifact_root_sha256,
        "frozen_primary_config_sha256": primary_gate.frozen_primary_config_sha256,
        "frozen_confirmatory_config_sha256": (primary_gate.frozen_confirmatory_config_sha256),
        "dataset_sha256": primary_gate.dataset_sha256,
        "manifest_sha256": primary_gate.manifest_sha256,
        "duplicate_audit_sha256": primary_gate.duplicate_audit_sha256,
        "pathology_encoder_audit_sha256": (primary_gate.pathology_encoder_audit_sha256),
        "source_tree_root_sha256": primary_gate.source_tree_root_sha256,
    }
    for field, expected in expected_bindings.items():
        if completion.get(field) != expected:
            raise ValueError(f"historical primary completion binding {field} differs from P")
    for artifact in (
        ".immutable.json",
        "artifact_manifest.json",
        "status.json",
        "source_tree_manifest.json",
        "run_provenance.json",
        "primary_execution_gate.json",
        "primary_input_bindings.json",
        "execution_controls.json",
        "cell_index.csv",
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
        "restoration_index.json",
    ):
        if not (run_directory / artifact).is_file():
            raise ValueError(
                f"historical primary completion evidence lacks required artifact: {artifact}"
            )
    recovery_identity = _validate_primary_recovery_downstream_identity(
        run_directory=run_directory,
        primary_gate=primary_gate,
        completion=completion,
        metrics=metrics,
        primary_stage_eligibility_receipt=eligibility_receipt,
        recovery_experiment=True,
    )
    if recovery_identity is None:
        raise ValueError("historical primary lacks exact orphan-recovery lineage under P")
    _, statistics, restoration = _validate_primary_completion_attestations(
        run_directory=run_directory,
        primary_gate=primary_gate,
        completion=completion,
        reconciliation=reconciliation,
        metrics=metrics,
        integrity=integrity,
        primary_stage_eligibility_receipt=eligibility_receipt,
    )
    (
        finalization_only_successor,
        _,
        _,
        _,
    ) = _validate_primary_finalization_successor(
        run_directory=run_directory,
        completion=completion,
        primary_gate=primary_gate,
    )
    if finalization_only_successor:
        raise ValueError(
            "resource-bounded historical primary cannot be a finalization-only successor"
        )
    run_id = integrity.run_id
    artifact_root = integrity.expected_root_sha256
    if (
        not isinstance(run_id, str)
        or not isinstance(artifact_root, str)
        or _SHA256_PATTERN.fullmatch(artifact_root) is None
    ):
        raise ValueError("historical primary integrity lacks exact run/root identity")
    return HistoricalPrimaryDependencyEvidence(
        primary_gate=primary_gate,
        primary_run_directory=run_directory,
        primary_run_id=run_id,
        primary_artifact_root_sha256=artifact_root,
        primary_artifact_manifest_sha256=sha256_file(run_directory / "artifact_manifest.json"),
        primary_completion_evidence_sha256=sha256_file(run_directory / "completion_evidence.json"),
        primary_execution_gate_sha256=sha256_file(run_directory / "primary_execution_gate.json"),
        primary_reconciliation_sha256=sha256_file(run_directory / "reconciliation.json"),
        completed_required_cell_count=int(completed_count),
        primary_statistics_manifest_sha256=statistics.manifest_sha256,
        primary_statistics_sha256=statistics.statistics_sha256,
        primary_bootstrap_evidence_sha256=statistics.bootstrap_evidence_sha256,
        primary_subgroups_sha256=statistics.subgroups_sha256,
        primary_statistics_source_readback_root_sha256=(statistics.source_readback_root_sha256),
        primary_statistics_comparison_count=statistics.comparison_count,
        primary_stage_attestation_record_sha256=(statistics.stage_attestation_record_sha256),
        primary_stage_attestation_verification_sha256=(
            statistics.stage_attestation_verification_sha256
        ),
        primary_restoration_readback_root_sha256=(restoration.readback_root_sha256),
        primary_recovery_evidence_sha256=recovery_identity.evidence_sha256,
        primary_recovery_authorization_sha256=(recovery_identity.authorization_sha256),
        primary_recovery_source_run_id=recovery_identity.source_run_id,
        primary_recovery_source_snapshot_root_sha256=(
            recovery_identity.source_snapshot_root_sha256
        ),
        primary_recovery_analysis_disposition=(recovery_identity.analysis_disposition),
    )


def validate_confirmatory_execution_gate(
    *,
    primary_run_directory: str | Path,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path | None = None,
    frozen_confirmatory_config_path: str | Path | None = None,
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt | None = None,
) -> ConfirmatoryExecutionGateEvidence:
    """Return evidence for a complete, sealed, outcome-eligible primary study.

    An ordinary call is preliminary evidence only.  Execution authorization must
    pass an active receipt yielded by ``guard_run_stage_eligibility`` so the final
    validation and dependent run creation share one mutation-lock boundary.
    """

    run_directory = Path(primary_run_directory).resolve()
    integrity = verify_run_integrity(run_directory)
    if not integrity.valid or not integrity.registry_record_present:
        raise ValueError(f"primary run failed integrity verification: {integrity.errors}")
    if primary_stage_eligibility_receipt is None:
        eligibility_receipt = require_run_stage_eligibility_receipt(
            run_directory,
            integrity=integrity,
        )
    else:
        eligibility_receipt = primary_stage_eligibility_receipt
        if not isinstance(eligibility_receipt, RunStageEligibilityReceipt):
            raise ValueError(
                "supplied primary eligibility receipt is not active under its mutation guard"
            )
        try:
            eligibility_receipt.require_active_authority()
        except ValueError as exc:
            raise ValueError(
                "supplied primary eligibility receipt is not active under its mutation guard"
            ) from exc
    if (
        not isinstance(eligibility_receipt, RunStageEligibilityReceipt)
        or not eligibility_receipt.valid
        or eligibility_receipt.run_directory.resolve() != run_directory
        or eligibility_receipt.run_id != run_directory.name
        or eligibility_receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
    ):
        raise ValueError(
            "primary run lacks its exact positive PRIMARY_STUDY_COMPLETE post-seal "
            "eligibility receipt"
        )
    status = _read_mapping(run_directory / "status.json", "primary run status")
    recovery_experiment = status.get("experiment_name") == PRIMARY_RECOVERY_EXPERIMENT_NAME
    primary_gate_arguments: dict[str, Any] = {
        "project_root": project_root,
        "freeze_directory": freeze_directory,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "duplicate_audit_path": duplicate_audit_path,
        "pathology_encoder_audit_path": pathology_encoder_audit_path,
        "frozen_primary_config_path": frozen_primary_config_path,
        "frozen_confirmatory_config_path": frozen_confirmatory_config_path,
    }
    if recovery_experiment:
        primary_gate_arguments["experiment_name"] = PRIMARY_RECOVERY_EXPERIMENT_NAME
    primary_gate = validate_primary_execution_gate(**primary_gate_arguments)
    for artifact in (
        "completion_evidence.json",
        "reconciliation.json",
        "matrix_plan.json",
        "metrics.json",
        "report.md",
    ):
        if not (run_directory / artifact).is_file():
            raise ValueError(f"primary completion evidence lacks required artifact: {artifact}")
    completion = _read_mapping(run_directory / "completion_evidence.json", "primary completion")
    reconciliation = _read_mapping(run_directory / "reconciliation.json", "primary reconciliation")
    metrics = _read_mapping(run_directory / "metrics.json", "primary metrics")
    if completion.get("completion_stage") != "PRIMARY_STUDY_COMPLETE":
        raise ValueError("primary completion stage is not PRIMARY_STUDY_COMPLETE")
    if completion.get("study_outcome_eligible") is not True:
        raise ValueError("primary run is not eligible study-outcome evidence")
    if completion.get("artifact_scope") != "real_pannuke_primary_study":
        raise ValueError("primary artifact scope is not real PanNuke primary evidence")
    if metrics.get("artifact_scope") != "real_pannuke_primary_study":
        raise ValueError("primary metrics artifact scope is ineligible")
    if (
        metrics.get("completion_stage") != "PRIMARY_STUDY_COMPLETE"
        or metrics.get("study_outcome_eligible") is not True
    ):
        raise ValueError("primary metrics do not carry an eligible completion claim")
    if reconciliation.get("status") != "passed":
        raise ValueError("primary artifact reconciliation did not pass")
    required_count = completion.get("required_cell_count")
    completed_count = completion.get("completed_required_cell_count")
    if (
        not isinstance(required_count, int)
        or required_count != primary_gate.primary_required_cell_count
    ):
        raise ValueError("primary completion required-cell count differs from the frozen matrix")
    if completed_count != required_count or completion.get("failed_required_cell_count") != 0:
        raise ValueError("primary completion has missing or failed required cells")
    expected_bindings = {
        "freeze_artifact_root_sha256": primary_gate.freeze_artifact_root_sha256,
        "frozen_primary_config_sha256": primary_gate.frozen_primary_config_sha256,
        "frozen_confirmatory_config_sha256": primary_gate.frozen_confirmatory_config_sha256,
        "dataset_sha256": primary_gate.dataset_sha256,
        "manifest_sha256": primary_gate.manifest_sha256,
        "duplicate_audit_sha256": primary_gate.duplicate_audit_sha256,
        "pathology_encoder_audit_sha256": primary_gate.pathology_encoder_audit_sha256,
        "source_tree_root_sha256": primary_gate.source_tree_root_sha256,
    }
    for field, expected in expected_bindings.items():
        if completion.get(field) != expected:
            raise ValueError(f"primary completion binding {field} differs from frozen evidence")
    for artifact in (
        ".immutable.json",
        "artifact_manifest.json",
        "status.json",
        "source_tree_manifest.json",
        "run_provenance.json",
        "primary_execution_gate.json",
        "primary_input_bindings.json",
        "execution_controls.json",
        "cell_index.csv",
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
        "restoration_index.json",
    ):
        if not (run_directory / artifact).is_file():
            raise ValueError(f"primary completion evidence lacks required artifact: {artifact}")
    recovery_identity = _validate_primary_recovery_downstream_identity(
        run_directory=run_directory,
        primary_gate=primary_gate,
        completion=completion,
        metrics=metrics,
        primary_stage_eligibility_receipt=eligibility_receipt,
        recovery_experiment=recovery_experiment,
    )
    _, statistics, restoration = _validate_primary_completion_attestations(
        run_directory=run_directory,
        primary_gate=primary_gate,
        completion=completion,
        reconciliation=reconciliation,
        metrics=metrics,
        integrity=integrity,
        primary_stage_eligibility_receipt=eligibility_receipt,
    )
    (
        finalization_only_successor,
        successor_evidence_sha,
        predecessor_run_id,
        predecessor_artifact_root,
    ) = _validate_primary_finalization_successor(
        run_directory=run_directory,
        completion=completion,
        primary_gate=primary_gate,
    )
    confirmatory_storage_policy_sha256 = canonical_sha256(
        require_confirmatory_storage_policy(primary_gate.freeze_directory)
    )
    if _SHA256_PATTERN.fullmatch(confirmatory_storage_policy_sha256) is None:
        raise ValueError("confirmatory storage policy did not produce a canonical SHA-256")
    return ConfirmatoryExecutionGateEvidence(
        primary_gate=primary_gate,
        primary_run_directory=run_directory,
        primary_run_id=str(integrity.run_id),
        primary_artifact_root_sha256=str(integrity.expected_root_sha256),
        primary_completion_evidence_sha256=sha256_file(run_directory / "completion_evidence.json"),
        primary_reconciliation_sha256=sha256_file(run_directory / "reconciliation.json"),
        completed_required_cell_count=int(completed_count),
        primary_statistics_manifest_sha256=statistics.manifest_sha256,
        primary_statistics_sha256=statistics.statistics_sha256,
        primary_bootstrap_evidence_sha256=statistics.bootstrap_evidence_sha256,
        primary_subgroups_sha256=statistics.subgroups_sha256,
        primary_statistics_source_readback_root_sha256=(statistics.source_readback_root_sha256),
        primary_statistics_comparison_count=statistics.comparison_count,
        primary_stage_attestation_record_sha256=(statistics.stage_attestation_record_sha256),
        primary_stage_attestation_verification_sha256=(
            statistics.stage_attestation_verification_sha256
        ),
        primary_restoration_readback_root_sha256=restoration.readback_root_sha256,
        primary_finalization_only_successor=finalization_only_successor,
        primary_finalization_successor_evidence_sha256=successor_evidence_sha,
        primary_predecessor_run_id=predecessor_run_id,
        primary_predecessor_artifact_root_sha256=predecessor_artifact_root,
        confirmatory_storage_policy_sha256=confirmatory_storage_policy_sha256,
        primary_orphan_recovery=recovery_identity is not None,
        primary_recovery_evidence_sha256=(
            recovery_identity.evidence_sha256 if recovery_identity is not None else None
        ),
        primary_recovery_authorization_sha256=(
            recovery_identity.authorization_sha256 if recovery_identity is not None else None
        ),
        primary_recovery_source_run_id=(
            recovery_identity.source_run_id if recovery_identity is not None else None
        ),
        primary_recovery_source_snapshot_root_sha256=(
            recovery_identity.source_snapshot_root_sha256 if recovery_identity is not None else None
        ),
        primary_recovery_analysis_disposition=(
            recovery_identity.analysis_disposition if recovery_identity is not None else None
        ),
    )


def validate_resource_bounded_execution_gate(
    *,
    primary_run_directory: str | Path,
    project_root: str | Path,
    resource_authority_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    resource_confirmatory_config_path: str | Path | None = None,
    primary_stage_eligibility_receipt: RunStageEligibilityReceipt | None = None,
) -> ResourceBoundedExecutionGateEvidence:
    """Validate dual authority P+(effective C/D) without rebinding primary.

    P authenticates only the completed recovery primary and its sealed source.
    C, or its unique direct technical successor D, authenticates only the current
    resource profile, source, inherited storage policy, and fixed capacity bounds.
    The returned evidence is permanently post-outcome and cannot represent a
    confirmatory completion claim.
    """

    root = Path(project_root).resolve()
    resource_authority = Path(resource_authority_directory).resolve()
    authorization = _require_sealed_resource_bounded_confirmatory_authorization(resource_authority)
    authority_verification = verify_preregistration_amendment(
        resource_authority,
        max_chain_depth=_MAX_AUTHORITY_CHAIN_DEPTH,
    )
    if (
        not authority_verification.valid
        or authority_verification.artifact_root_sha256 is None
        or authority_verification.sha256_manifest_sha256 is None
        or authority_verification.chain_depth is None
    ):
        raise ValueError(
            "effective resource-bounded execution authority C/D failed immutable "
            "verification: "
            f"{authority_verification.errors}"
        )
    historical_authorization = _mapping(
        authorization.get("historical_primary"),
        "resource-bounded historical-primary authorization",
    )
    registration = _mapping(
        historical_authorization.get("registration_authority"),
        "resource-bounded historical registration authority",
    )
    historical_authority_value = registration.get("directory")
    if (
        not isinstance(historical_authority_value, str)
        or not Path(historical_authority_value).is_absolute()
    ):
        raise ValueError("resource-bounded historical authority P must be explicit and absolute")
    historical_authority = Path(historical_authority_value).resolve()
    requested_primary = Path(primary_run_directory).resolve()
    authorized_primary_value = historical_authorization.get("run_directory")
    if (
        not isinstance(authorized_primary_value, str)
        or not Path(authorized_primary_value).is_absolute()
        or requested_primary != Path(authorized_primary_value).resolve()
    ):
        raise ValueError("resource-bounded primary run differs from the exact historical binding")

    historical = validate_historical_primary_dependency(
        primary_run_directory=requested_primary,
        project_root=root,
        recovery_authority_directory=historical_authority,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        duplicate_audit_path=duplicate_audit_path,
        pathology_encoder_audit_path=pathology_encoder_audit_path,
        primary_stage_eligibility_receipt=primary_stage_eligibility_receipt,
    )
    expected_historical_bindings = {
        "run_id": historical.primary_run_id,
        "run_directory": str(historical.primary_run_directory),
        "artifact_root_sha256": historical.primary_artifact_root_sha256,
        "artifact_manifest_sha256": historical.primary_artifact_manifest_sha256,
        "completion_evidence_sha256": (historical.primary_completion_evidence_sha256),
        "primary_execution_gate_sha256": (historical.primary_execution_gate_sha256),
        "stage_attestation_record_sha256": (historical.primary_stage_attestation_record_sha256),
        "stage_attestation_verification_sha256": (
            historical.primary_stage_attestation_verification_sha256
        ),
        "recovery_evidence_sha256": (historical.primary_recovery_evidence_sha256),
        "recovery_authorization_sha256": (historical.primary_recovery_authorization_sha256),
    }
    for field, expected in expected_historical_bindings.items():
        if historical_authorization.get(field) != expected:
            raise ValueError(
                f"resource-bounded historical-primary authorization differs at {field}"
            )
    expected_registration = {
        "directory": str(historical.primary_gate.freeze_directory),
        "kind": "preregistration_amendment",
        "artifact_root_sha256": (historical.primary_gate.freeze_artifact_root_sha256),
        "sha256_manifest_sha256": historical.primary_gate.freeze_manifest_sha256,
        "chain_depth": (historical.primary_gate.registration_authority_chain_depth),
    }
    if dict(registration) != expected_registration:
        raise ValueError(
            "resource-bounded historical-primary authority P differs from its reconstructed gate"
        )

    resource_profile = _mapping(
        authorization.get("resource_profile"),
        "resource-bounded execution profile",
    )
    config_snapshot = (resource_authority / "confirmatory_frozen.yaml").resolve()
    requested_config = (
        config_snapshot
        if resource_confirmatory_config_path is None
        else _resolve_from_project(root, resource_confirmatory_config_path)
    )
    if requested_config != config_snapshot:
        raise ValueError(
            "resource-bounded execution must load the effective authority's exact "
            "confirmatory snapshot"
        )
    resource_config = load_config(requested_config)
    if (
        sha256_file(requested_config)
        != resource_profile.get("resource_confirmatory_config_file_sha256")
        or config_sha256(resource_config)
        != resource_profile.get("resource_confirmatory_config_semantic_sha256")
        or resource_config.get("execution_profile") != resource_profile.get("profile_id")
        or resource_config.get("analysis_disposition") != _POST_OUTCOME_RECOVERY_REGISTRATION_STATUS
        or resource_config.get("original_confirmatory_claim_allowed") is not False
        or resource_config.get("completion_stage") is not None
    ):
        raise ValueError(
            "resource-bounded current config differs from effective execution authority"
        )

    child_source_path = resource_authority / "source_tree_manifest.json"
    child_source = _validate_execution_source_manifest(
        child_source_path,
        "effective resource-bounded authority execution source",
    )
    source_delta = _mapping(
        authorization.get("execution_source_delta"),
        "resource-bounded execution-source delta",
    )
    if child_source.get("root_sha256") != source_delta.get("resource_root_sha256") or sha256_file(
        child_source_path
    ) != source_delta.get("resource_manifest_sha256"):
        raise ValueError(
            "resource-bounded source snapshot differs from effective execution authority"
        )
    current_source = capture_source_tree(root)
    if current_source != child_source:
        raise ValueError(
            "current execution source differs from resource-bounded authority C "
            "or its effective successor D"
        )
    for snapshot_name in ("PRE_REGISTRATION_FROZEN.md", "primary_frozen.yaml"):
        if sha256_file(resource_authority / snapshot_name) != sha256_file(
            historical_authority / snapshot_name
        ):
            raise ValueError(
                "effective resource-bounded authority changed preregistration or primary "
                f"identity: {snapshot_name}"
            )

    resource_amendment_evidence = _read_mapping(
        resource_authority / "amendment_evidence.json",
        "resource-bounded amendment evidence",
    )
    storage_policy = _mapping(
        resource_amendment_evidence.get("confirmatory_storage_policy"),
        "resource-bounded confirmatory storage policy",
    )
    expected_storage_policy = ConfirmatoryStoragePolicy().as_dict()
    if dict(storage_policy) != expected_storage_policy:
        raise ValueError("resource-bounded storage policy differs from the exact inherited policy")
    storage_policy_sha256 = canonical_sha256(expected_storage_policy)
    if _SHA256_PATTERN.fullmatch(storage_policy_sha256) is None:
        raise ValueError("resource-bounded storage policy did not produce a canonical SHA-256")
    capacity_policy = _mapping(
        authorization.get("resource_capacity_policy"),
        "resource-bounded capacity policy",
    )
    raw_workspace_plan = authorization.get("resource_input_workspace_plan")
    workspace_plan: Mapping[str, Any] | None
    workspace_plan_sha256: str | None
    if raw_workspace_plan is None:
        if capacity_policy.get("policy") == RESOURCE_BOUNDED_CAPACITY_POLICY_V3:
            raise ValueError(
                "resource-bounded capacity-v3 authority lacks its exact input-workspace plan"
            )
        workspace_plan = None
        workspace_plan_sha256 = None
    else:
        workspace_plan = _mapping(
            raw_workspace_plan,
            "resource-bounded input-workspace plan",
        )
        canonical_capacity, canonical_workspace_plan = validate_resource_bounded_capacity_v3(
            capacity_policy,
            workspace_plan,
        )
        capacity_policy = canonical_capacity
        workspace_plan = canonical_workspace_plan
        workspace_plan_sha256 = _require_sha256(
            workspace_plan.get("plan_without_self_hash_sha256"),
            "resource-bounded input-workspace plan",
        )
    execution_authority = ResourceBoundedExecutionAuthorityEvidence(
        authority_directory=resource_authority,
        authority_artifact_root_sha256=(authority_verification.artifact_root_sha256),
        authority_manifest_sha256=(authority_verification.sha256_manifest_sha256),
        authority_chain_depth=authority_verification.chain_depth,
        authorization_sha256=canonical_sha256(authorization),
        resource_profile_id=str(resource_profile["profile_id"]),
        resource_confirmatory_config_file_sha256=str(
            resource_profile["resource_confirmatory_config_file_sha256"]
        ),
        resource_confirmatory_config_semantic_sha256=str(
            resource_profile["resource_confirmatory_config_semantic_sha256"]
        ),
        resource_execution_source_root_sha256=str(source_delta["resource_root_sha256"]),
        resource_execution_source_manifest_sha256=str(source_delta["resource_manifest_sha256"]),
        resource_source_delta_sha256=str(source_delta["delta_sha256"]),
        confirmatory_storage_policy_sha256=storage_policy_sha256,
        resource_capacity_policy=dict(capacity_policy),
        resource_input_workspace_plan=dict(workspace_plan) if workspace_plan else None,
        resource_input_workspace_plan_sha256=workspace_plan_sha256,
    )
    final_authority_verification = verify_preregistration_amendment(
        resource_authority,
        max_chain_depth=_MAX_AUTHORITY_CHAIN_DEPTH,
    )
    if (
        not final_authority_verification.valid
        or final_authority_verification.artifact_root_sha256
        != authority_verification.artifact_root_sha256
        or final_authority_verification.sha256_manifest_sha256
        != authority_verification.sha256_manifest_sha256
        or final_authority_verification.chain_depth != authority_verification.chain_depth
    ):
        raise ValueError(
            "effective resource-bounded execution authority changed during gate validation"
        )
    return ResourceBoundedExecutionGateEvidence(
        historical_primary=historical,
        execution_authority=execution_authority,
    )


__all__ = [
    "PRIMARY_RECOVERY_EXPERIMENT_NAME",
    "ConfirmatoryExecutionGateEvidence",
    "HistoricalPrimaryDependencyEvidence",
    "PrimaryExecutionGateEvidence",
    "PrimaryStatisticsStageReadback",
    "ResourceBoundedExecutionAuthorityEvidence",
    "ResourceBoundedExecutionGateEvidence",
    "validate_confirmatory_execution_gate",
    "validate_historical_primary_dependency",
    "validate_primary_execution_gate",
    "validate_resource_bounded_execution_gate",
]
