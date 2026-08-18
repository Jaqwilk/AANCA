"""One-shot tracked recovery of an authorized interrupted primary orphan.

This workflow has no trainer, matrix executor, fallback, retry loop, or source-run
mutation.  It qualifies the orphan before creating a run, performs one physical
copy, verifies the destination, writes one completion candidate, seals once, and
requests one post-seal stage attestation.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    PrimaryFilesystemReadbackEvidence,
    PrimaryRestorationReadbackEvidence,
    build_primary_completion_evidence,
)
from histo_audit.experiment.primary_core import PrimaryExecutionControls
from histo_audit.experiment.primary_recovery import (
    RECOVERY_COPY_POLICY,
    RECOVERY_EVIDENCE_FILENAME,
    RECOVERY_EXPERIMENT_NAME,
    RECOVERY_POLICY,
    RECOVERY_REGISTRATION_STATUS,
    OrphanSourceInspection,
    RecoveryAuthorization,
    RecoveryCopyReceipt,
    RecoveryDestinationVerification,
    copy_authorized_orphan_artifacts,
    inspect_orphan_source,
    verify_recovery_destination,
)
from histo_audit.experiment.primary_statistics import InheritedPrimaryStatisticsVerification
from histo_audit.experiment.study_contracts import PrimaryMatrixPlan
from histo_audit.pannuke.publication import AnchoredPhysicalCopyBoundaryError
from histo_audit.utils.run_tracking import (
    IntegrityVerification,
    RunTracker,
    _build_primary_stage_attestation_verification,
    attest_primary_run_stage_eligibility,
    require_run_stage_eligible,
    sha256_file,
    verify_run_integrity,
    withdraw_run_eligibility,
)
from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence

_COMPLETION_STAGE = "PRIMARY_STUDY_COMPLETE"
_DISK_MARGIN_BYTES = 10 * 1024**3
_NUMERIC_PROOF_POLICY = "orphan_recovery_content_addressed_numeric_inheritance_v1"
_POSTSEAL_WITHDRAWAL_REASON_CODE = "primary_orphan_recovery.postseal_failure"
_POSTSEAL_WITHDRAWAL_REASON = "Mandatory post-seal primary orphan-recovery verification failed."
_STATISTICS_FILES = frozenset(
    {
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
    }
)
_SCIENTIFIC_GATE_BINDINGS = (
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
_GATE_COMPLETION_BINDINGS = (
    "freeze_artifact_root_sha256",
    "frozen_primary_config_sha256",
    "frozen_confirmatory_config_sha256",
    "dataset_sha256",
    "manifest_sha256",
    "duplicate_audit_sha256",
    "pathology_encoder_audit_sha256",
    "source_tree_root_sha256",
)
_RECOVERY_EVIDENCE_KEYS = {
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


class PrimaryRecoveryRunnerError(RuntimeError):
    """One fail-closed recovery execution or verification error."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        run_directory: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.run_directory = run_directory


@dataclass(frozen=True, slots=True)
class RecoveryDiskPreflight:
    """Free-space receipt established before the sole RunTracker starts."""

    volume_path: str
    copy_policy: str
    capacity_basis: str
    compressor_path: str
    copy_bytes: int
    largest_artifact_bytes: int
    margin_bytes: int
    required_free_bytes: int
    observed_free_bytes: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrimaryRecoveryRunnerError(f"{role} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrimaryRecoveryRunnerError(f"{role} must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _require_authorization_matches_immutable_amendment(
    authorization: RecoveryAuthorization,
) -> None:
    """Reload and exactly match the canonical authority before source scanning."""

    if (
        authorization.authority_directory is None
        or authorization.authority_artifact_root_sha256 is None
        or authorization.authority_manifest_sha256 is None
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery authorization lacks immutable amendment bindings"
        )
    from histo_audit.workflows.preregistration_amendment import (
        require_primary_recovery_authorization,
    )

    canonical = require_primary_recovery_authorization(
        authorization.authority_directory,
    )
    reloaded = RecoveryAuthorization.from_mapping(
        canonical,
        authority_directory=authorization.authority_directory,
        authority_artifact_root_sha256=(authorization.authority_artifact_root_sha256),
        authority_manifest_sha256=authorization.authority_manifest_sha256,
    )
    if reloaded != authorization:
        raise PrimaryRecoveryRunnerError(
            "supplied recovery authorization differs from its immutable amendment"
        )


def _require_recovery_statistics_api() -> ModuleType:
    """Fail before RunTracker when the recovery-specific capability is unavailable."""

    module = importlib.import_module("histo_audit.experiment.primary_statistics")
    required = (
        "AuthorizedOrphanNumericVerificationProof",
        "OrphanRecoveryNumericVerificationProvenance",
        "_issue_authorized_orphan_numeric_verification_proof",
        "attest_inherited_primary_statistics_artifacts",
        "INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION",
        "INHERITED_PRIOR_NUMERIC_LIMITATION",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise PrimaryRecoveryRunnerError(
            "recovery-specific inherited-statistics capability is unavailable; "
            f"refusing to fabricate finalization-only proof fields: {missing}"
        )
    return module


def _require_gate_and_authority(
    *,
    gate: PrimaryExecutionGateEvidence,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
) -> None:
    if not isinstance(gate, PrimaryExecutionGateEvidence):
        raise TypeError("recovery requires a genuine PrimaryExecutionGateEvidence")
    if not isinstance(plan, PrimaryMatrixPlan):
        raise TypeError("recovery requires a genuine PrimaryMatrixPlan")
    if not isinstance(controls, PrimaryExecutionControls):
        raise TypeError("recovery requires genuine PrimaryExecutionControls")
    if not isinstance(authorization, RecoveryAuthorization):
        raise TypeError("recovery requires a typed RecoveryAuthorization")
    controls.validate_for_plan(plan)
    if (
        controls.plan_sha256 != canonical_sha256(plan.as_dict())
        or plan.config_sha256 != gate.primary_config_semantic_sha256
        or len(plan.cells) != gate.primary_matrix_cell_count
        or plan.required_cell_count != gate.primary_required_cell_count
    ):
        raise PrimaryRecoveryRunnerError("recovery plan/controls differ from the current gate")
    authority_directory = authorization.authority_directory
    if (
        authority_directory is None
        or authorization.authority_artifact_root_sha256 is None
        or authorization.authority_manifest_sha256 is None
        or gate.freeze_directory.resolve() != authority_directory.resolve()
        or gate.freeze_artifact_root_sha256 != authorization.authority_artifact_root_sha256
        or gate.freeze_manifest_sha256 != authorization.authority_manifest_sha256
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery authorization lacks its exact verified immutable authority binding"
        )
    _require_authorization_matches_immutable_amendment(authorization)
    registration_status = getattr(gate, "registration_status", RECOVERY_REGISTRATION_STATUS)
    if (
        registration_status != RECOVERY_REGISTRATION_STATUS
        or gate.original_unamended_primary_claim_allowed
        or gate.amended_primary_claim_allowed
        or gate.registration_authority_kind == "base_freeze"
    ):
        raise PrimaryRecoveryRunnerError(
            "current gate does not authorize the amended-or-exploratory recovery disposition"
        )


def _require_source_gate_compatibility(
    inspection: OrphanSourceInspection,
    current_gate: PrimaryExecutionGateEvidence,
) -> None:
    source_gate = _read_json_object(
        inspection.snapshot.run_directory / "primary_execution_gate.json",
        "orphan primary execution gate",
    )
    current = current_gate.as_dict()
    differing = [
        field for field in _SCIENTIFIC_GATE_BINDINGS if source_gate.get(field) != current.get(field)
    ]
    if differing:
        raise PrimaryRecoveryRunnerError(
            f"orphan scientific bindings differ from the recovery gate: {differing}"
        )


def _validate_recovery_downstream_attestations(
    *,
    run_directory: Path,
    readback: PrimaryFilesystemReadbackEvidence,
    statistics_verification: InheritedPrimaryStatisticsVerification,
    restoration_readback: PrimaryRestorationReadbackEvidence,
) -> None:
    if (
        not isinstance(readback, PrimaryFilesystemReadbackEvidence)
        or not readback.passed
        or readback.run_directory.resolve() != run_directory
        or not isinstance(
            statistics_verification,
            InheritedPrimaryStatisticsVerification,
        )
        or not statistics_verification.valid
        or statistics_verification.authorization_kind != "orphan_recovery"
        or statistics_verification.output_directory.resolve() != run_directory
        or statistics_verification.source_readback_root_sha256 != readback.readback_root_sha256
        or not isinstance(restoration_readback, PrimaryRestorationReadbackEvidence)
        or not restoration_readback.passed
        or restoration_readback.run_directory.resolve() != run_directory
        or restoration_readback.source_readback_root_sha256 != readback.readback_root_sha256
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery typed downstream attestations do not bind this destination"
        )


def _validate_recovery_completion_candidate(
    candidate: Mapping[str, Any],
    *,
    plan: PrimaryMatrixPlan,
    gate: PrimaryExecutionGateEvidence,
    readback: PrimaryFilesystemReadbackEvidence,
    controls: PrimaryExecutionControls,
    statistics_verification: InheritedPrimaryStatisticsVerification,
    restoration_readback: PrimaryRestorationReadbackEvidence,
    run_id: str,
    source_run_id: str,
    lineage_sha256: str,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "schema_version": 2,
        "completion_stage": _COMPLETION_STAGE,
        "study_outcome_eligible": True,
        "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
        "matrix_config_sha256": plan.config_sha256,
        "planned_cell_count": len(plan.cells),
        "required_cell_count": plan.required_cell_count,
        "completed_required_cell_count": plan.required_cell_count,
        "skipped_optional_cell_count": readback.skipped_optional_cell_count,
        "failed_required_cell_count": 0,
        "reconciliation_status": "passed",
        "completion_stage_enabled_only_after_run_seal_and_integrity_verification": True,
        "filesystem_run_directory": str(readback.run_directory),
        "filesystem_matrix_plan_sha256": readback.matrix_plan_sha256,
        "filesystem_execution_controls_sha256": readback.execution_controls_sha256,
        "filesystem_execution_controls_binding_sha256": (
            readback.execution_controls_binding_sha256
        ),
        "filesystem_cell_index_sha256": readback.cell_index_sha256,
        "filesystem_readback_root_sha256": readback.readback_root_sha256,
        "filesystem_completed_cell_count": readback.completed_cell_count,
        "filesystem_scenario_corruption_sha256": dict(readback.scenario_corruption_sha256),
        "execution_controls_binding_sha256": controls.binding_sha256,
        "circularity_excluded_cell_count": len(readback.circularity_excluded_cell_ids),
        "circularity_excluded_cell_ids": list(readback.circularity_excluded_cell_ids),
        "primary_confirmatory_claims_require_exclusion_of_these_cells": bool(
            readback.circularity_excluded_cell_ids
        ),
        "primary_statistics_verification_status": "passed",
        "primary_statistics_sha256": statistics_verification.statistics_sha256,
        "primary_bootstrap_evidence_sha256": (statistics_verification.bootstrap_evidence_sha256),
        "primary_subgroups_sha256": statistics_verification.subgroups_sha256,
        "primary_statistics_manifest_sha256": statistics_verification.manifest_sha256,
        "primary_statistics_comparison_count": statistics_verification.comparison_count,
        "primary_statistics_source_readback_root_sha256": (
            statistics_verification.source_readback_root_sha256
        ),
        "primary_restoration_verification_status": "passed",
        "primary_restoration_index_sha256": (restoration_readback.restoration_index_sha256),
        "primary_restoration_readback_root_sha256": (restoration_readback.readback_root_sha256),
        "primary_restoration_source_readback_root_sha256": (
            restoration_readback.source_readback_root_sha256
        ),
        "primary_restoration_cell_count": restoration_readback.restoration_cell_count,
        "primary_restoration_downstream_comparison_count": (
            restoration_readback.downstream_comparison_count
        ),
        "run_id": run_id,
        "retry_of_run_id": source_run_id,
        "retry_predecessor_binding_sha256": lineage_sha256,
        "post_seal_integrity_verification_required": True,
        "post_seal_attestation_required": True,
    }
    differing = [field for field, value in expected.items() if candidate.get(field) != value]
    if differing:
        raise PrimaryRecoveryRunnerError(
            f"completion builder produced a non-exact recovery candidate: {differing}"
        )
    for field in _GATE_COMPLETION_BINDINGS:
        if candidate.get(field) != getattr(gate, field):
            raise PrimaryRecoveryRunnerError(
                f"completion candidate differs from recovery gate binding {field}"
            )
    expected_restoration_hashes: dict[str, str] = {}
    for role, hashes in (
        ("json", restoration_readback.cell_json_sha256),
        ("evidence", restoration_readback.cell_evidence_sha256),
        ("manifest", restoration_readback.cell_manifest_sha256),
    ):
        for cell_id, digest in hashes:
            expected_restoration_hashes[f"primary_restoration_{role}_sha256::{cell_id}"] = digest
    actual_restoration_keys = {
        key
        for key in candidate
        if isinstance(key, str) and key.startswith("primary_restoration_") and "_sha256::" in key
    }
    if actual_restoration_keys != set(expected_restoration_hashes) or any(
        candidate.get(key) != value for key, value in expected_restoration_hashes.items()
    ):
        raise PrimaryRecoveryRunnerError(
            "completion candidate lacks exact per-cell restoration hashes"
        )
    expected_keys = (
        set(expected) | set(_GATE_COMPLETION_BINDINGS) | set(expected_restoration_hashes)
    )
    if set(candidate) != expected_keys:
        raise PrimaryRecoveryRunnerError(
            "completion builder produced missing or unrecognized recovery fields"
        )
    return dict(candidate)


def _require_disk_preflight(
    *,
    runs_root: Path,
    copy_bytes: int,
    largest_artifact_bytes: int,
) -> RecoveryDiskPreflight:
    if (
        type(copy_bytes) is not int
        or copy_bytes < 0
        or type(largest_artifact_bytes) is not int
        or largest_artifact_bytes < 0
        or largest_artifact_bytes > copy_bytes
    ):
        raise PrimaryRecoveryRunnerError("recovery copy sizes are not exact non-negative integers")
    system_root = os.environ.get("SYSTEMROOT")
    compressor = (
        Path(system_root) / "System32" / "compact.exe" if os.name == "nt" and system_root else None
    )
    if compressor is None or not compressor.is_file():
        raise PrimaryRecoveryRunnerError(
            "the exact Windows WOF/LZX compressor is unavailable before recovery"
        )
    volume_probe = Path(os.path.abspath(runs_root))
    while not volume_probe.exists() and volume_probe != volume_probe.parent:
        volume_probe = volume_probe.parent
    try:
        observed_free = shutil.disk_usage(volume_probe).free
    except OSError as exc:
        raise PrimaryRecoveryRunnerError(
            f"could not establish recovery destination free space: {exc}"
        ) from exc
    # The copier checks this same invariant before every file and immediately
    # compresses that file before advancing.  Safety therefore depends on the
    # largest next logical file plus the fixed margin, never on an estimated
    # compression ratio for the whole tree.
    required_free = largest_artifact_bytes + _DISK_MARGIN_BYTES
    receipt = RecoveryDiskPreflight(
        volume_path=str(volume_probe.resolve()),
        copy_policy=RECOVERY_COPY_POLICY,
        capacity_basis="streaming_largest_artifact_plus_margin_v1",
        compressor_path=str(compressor.resolve()),
        copy_bytes=copy_bytes,
        largest_artifact_bytes=largest_artifact_bytes,
        margin_bytes=_DISK_MARGIN_BYTES,
        required_free_bytes=required_free,
        observed_free_bytes=observed_free,
    )
    if observed_free < required_free:
        raise PrimaryRecoveryRunnerError(
            "insufficient space before recovery RunTracker creation: "
            f"free={observed_free}, largest_artifact={largest_artifact_bytes}, "
            f"margin={_DISK_MARGIN_BYTES}, required={required_free}"
        )
    return receipt


def _statistics_quartet(
    inspection: OrphanSourceInspection,
) -> list[dict[str, Any]]:
    records = [
        record.as_dict()
        for record in inspection.snapshot.artifacts
        if record.path in _STATISTICS_FILES
    ]
    if len(records) != 4 or {record["path"] for record in records} != _STATISTICS_FILES:
        raise PrimaryRecoveryRunnerError("authorized recovery statistics quartet is incomplete")
    return sorted(records, key=lambda record: str(record["path"]))


def _issue_recovery_numeric_proof(
    statistics_api: ModuleType,
    *,
    inspection: OrphanSourceInspection,
    destination: RecoveryDestinationVerification,
) -> Any:
    authorization = inspection.authorization
    expected_trust = statistics_api.INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
    expected_limitation = statistics_api.INHERITED_PRIOR_NUMERIC_LIMITATION
    if (
        authorization.trust_assumption != expected_trust
        or authorization.limitation != expected_limitation
        or authorization.authority_directory is None
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery amendment does not carry the exact inherited-statistics trust contract"
        )
    quartet = _statistics_quartet(inspection)
    proof_payload = {
        "schema_version": 1,
        "policy": _NUMERIC_PROOF_POLICY,
        "authorization_sha256": authorization.canonical_sha256,
        "source_run_id": authorization.source_run_id,
        "source_snapshot_root_sha256": inspection.snapshot.snapshot_root_sha256,
        "source_status_sha256": inspection.source_status_sha256,
        "source_tree_root_sha256": inspection.source_tree_root_sha256,
        "source_tree_manifest_sha256": inspection.source_tree_manifest_sha256,
        "source_readback_root_sha256": (
            inspection.snapshot.filesystem_readback.readback_root_sha256
        ),
        "statistics_quartet": quartet,
        "comparison_count": destination.statistics_comparison_count,
        "trust_assumption": authorization.trust_assumption,
        "limitation": authorization.limitation,
    }
    proof_sha256 = canonical_sha256(proof_payload)
    issuer = statistics_api._issue_authorized_orphan_numeric_verification_proof
    proof = issuer(
        amendment_directory=authorization.authority_directory,
        authorization_sha256=authorization.canonical_sha256,
        source_run_id=authorization.source_run_id,
        source_snapshot_root_sha256=inspection.snapshot.snapshot_root_sha256,
        source_status_sha256=inspection.source_status_sha256,
        source_tree_root_sha256=inspection.source_tree_root_sha256,
        source_tree_manifest_sha256=inspection.source_tree_manifest_sha256,
        source_readback_root_sha256=(inspection.snapshot.filesystem_readback.readback_root_sha256),
        prior_numeric_verification_proof_sha256=proof_sha256,
        trust_assumption=authorization.trust_assumption,
        limitation=authorization.limitation,
        statistics_quartet=quartet,
        comparison_count=destination.statistics_comparison_count,
    )
    proof_type = statistics_api.AuthorizedOrphanNumericVerificationProof
    if not isinstance(proof, proof_type) or not proof.valid:
        raise PrimaryRecoveryRunnerError(
            "recovery-specific numeric issuer did not return a genuine capability"
        )
    return proof


def _recovery_input_bindings(
    *,
    gate: PrimaryExecutionGateEvidence,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    inspection: OrphanSourceInspection,
    disk: RecoveryDiskPreflight,
) -> dict[str, Any]:
    authorization = inspection.authorization
    return {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "experiment_name": RECOVERY_EXPERIMENT_NAME,
        "source_run_id": authorization.source_run_id,
        "source_run_directory": str(authorization.source_run_directory),
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "authority_directory": str(authorization.authority_directory),
        "authority_artifact_root_sha256": authorization.authority_artifact_root_sha256,
        "authority_manifest_sha256": authorization.authority_manifest_sha256,
        "source_snapshot_root_sha256": inspection.snapshot.snapshot_root_sha256,
        "matrix_plan_sha256": controls.plan_sha256,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "primary_gate_sha256": canonical_sha256(gate.as_dict()),
        "disk_preflight": disk.as_dict(),
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "reused_required_cell_count": plan.required_cell_count,
        "skipped_optional_cell_count": len(plan.cells) - plan.required_cell_count,
        "retrained_cell_count": 0,
        "copy_policy": RECOVERY_COPY_POLICY,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
        "source_annotations_modified": False,
        "planned_cell_count": len(plan.cells),
        "required_cell_count": plan.required_cell_count,
    }


def _prepare_recovery(
    *,
    gate_evidence: PrimaryExecutionGateEvidence,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
    runs_root: Path,
    run_id: str | None,
) -> tuple[OrphanSourceInspection, RecoveryDiskPreflight, ModuleType]:
    """Freshly qualify every pre-RunTracker boundary for one invocation."""

    _require_gate_and_authority(
        gate=gate_evidence,
        plan=plan,
        controls=controls,
        authorization=authorization,
    )
    if run_id is not None and (
        not run_id.strip() or Path(run_id).name != run_id or run_id == authorization.source_run_id
    ):
        raise PrimaryRecoveryRunnerError("recovery run_id is unsafe or reuses the orphan ID")
    statistics_api = _require_recovery_statistics_api()
    if (
        authorization.trust_assumption != statistics_api.INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
        or authorization.limitation != statistics_api.INHERITED_PRIOR_NUMERIC_LIMITATION
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery amendment does not carry the exact inherited-statistics trust contract"
        )
    inspection = inspect_orphan_source(
        runs_root=runs_root,
        plan=plan,
        controls=controls,
        authorization=authorization,
    )
    _require_source_gate_compatibility(inspection, gate_evidence)
    disk = _require_disk_preflight(
        runs_root=runs_root,
        copy_bytes=inspection.snapshot.total_bytes,
        largest_artifact_bytes=max(
            (artifact.size_bytes for artifact in inspection.snapshot.artifacts),
            default=0,
        ),
    )
    return inspection, disk, statistics_api


def preflight_primary_orphan_recovery(
    *,
    gate_evidence: PrimaryExecutionGateEvidence,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
    runs_root: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Qualify recovery without creating a tracker or copying artifacts."""

    try:
        inspection, disk, _statistics_api = _prepare_recovery(
            gate_evidence=gate_evidence,
            plan=plan,
            controls=controls,
            authorization=authorization,
            runs_root=Path(runs_root).resolve(),
            run_id=run_id,
        )
    except PrimaryRecoveryRunnerError:
        raise
    except Exception as exc:
        raise PrimaryRecoveryRunnerError(
            f"primary orphan recovery preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "status": "passed",
        "policy": RECOVERY_POLICY,
        "experiment_name": RECOVERY_EXPERIMENT_NAME,
        "source_run_id": authorization.source_run_id,
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "source_snapshot_root_sha256": inspection.snapshot.snapshot_root_sha256,
        "reused_required_cell_count": inspection.snapshot.completed_required_cell_count,
        "skipped_optional_cell_count": inspection.snapshot.skipped_optional_cell_count,
        "retrained_cell_count": 0,
        "copy_artifact_count": len(inspection.snapshot.artifacts),
        "copy_total_bytes": inspection.snapshot.total_bytes,
        "disk_preflight": disk.as_dict(),
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
        "run_tracker_created": False,
        "copy_invoked": False,
    }


def _require_source_metadata_unchanged(inspection: OrphanSourceInspection) -> None:
    """Cheap post-copy check of the mutable source/control metadata boundary."""

    source = inspection.snapshot.run_directory
    authorization = inspection.authorization
    observed = {
        "status": sha256_file(source / "status.json"),
        "primary_gate": sha256_file(source / "primary_execution_gate.json"),
        "source_tree_manifest": sha256_file(source / "source_tree_manifest.json"),
        "interruption_receipt": sha256_file(authorization.interruption.receipt_path),
    }
    expected = {
        "status": authorization.expected_status_sha256,
        "primary_gate": authorization.expected_primary_execution_gate_sha256,
        "source_tree_manifest": authorization.expected_source_tree_manifest_sha256,
        "interruption_receipt": authorization.interruption.receipt_sha256,
    }
    if observed != expected:
        raise PrimaryRecoveryRunnerError(
            "orphan control metadata changed during the anchored physical copy"
        )


def _build_recovery_evidence(
    *,
    tracker: RunTracker,
    inspection: OrphanSourceInspection,
    destination: RecoveryDestinationVerification,
    copy_receipt: RecoveryCopyReceipt,
    statistics_verification: Any,
) -> dict[str, Any]:
    authorization = inspection.authorization
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "experiment_name": RECOVERY_EXPERIMENT_NAME,
        "source_run_id": authorization.source_run_id,
        "destination_run_id": tracker.run_id,
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "source_snapshot_root_sha256": inspection.snapshot.snapshot_root_sha256,
        "destination_snapshot_root_sha256": destination.snapshot_root_sha256,
        "reused_required_cell_count": (
            destination.filesystem_readback.completed_required_cell_count
        ),
        "skipped_optional_cell_count": (
            destination.filesystem_readback.skipped_optional_cell_count
        ),
        "retrained_cell_count": 0,
        "copy_policy": copy_receipt.copy_policy,
        "copied_artifact_count": copy_receipt.artifact_count,
        "copied_total_bytes": copy_receipt.total_bytes,
        "verification_mode": statistics_verification.verification_mode,
        "prior_numeric_verification_proof_sha256": (
            statistics_verification.prior_numeric_verification_proof_sha256
        ),
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    if set(evidence) != _RECOVERY_EVIDENCE_KEYS:
        raise PrimaryRecoveryRunnerError("internal recovery evidence schema is incomplete")
    return evidence


def _validate_recovery_evidence_payload(
    evidence: Mapping[str, Any],
    *,
    run_path: Path,
    plan: PrimaryMatrixPlan,
    authorization: RecoveryAuthorization,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    if set(evidence) != _RECOVERY_EVIDENCE_KEYS:
        raise PrimaryRecoveryRunnerError("recovery evidence has missing or extra fields")
    evidence_path = run_path / RECOVERY_EVIDENCE_FILENAME
    expected = {
        "schema_version": 1,
        "policy": RECOVERY_POLICY,
        "experiment_name": RECOVERY_EXPERIMENT_NAME,
        "source_run_id": authorization.source_run_id,
        "destination_run_id": run_path.name,
        "recovery_authorization_sha256": authorization.canonical_sha256,
        "source_snapshot_root_sha256": authorization.expected_source_snapshot_root_sha256,
        "destination_snapshot_root_sha256": (authorization.expected_source_snapshot_root_sha256),
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "reused_required_cell_count": plan.required_cell_count,
        "skipped_optional_cell_count": len(plan.cells) - plan.required_cell_count,
        "retrained_cell_count": 0,
        "copy_policy": RECOVERY_COPY_POLICY,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    differing = [field for field, value in expected.items() if evidence.get(field) != value]
    if differing:
        raise PrimaryRecoveryRunnerError(
            f"recovery evidence differs from its authority or destination: {differing}"
        )
    if (
        type(evidence.get("copied_artifact_count")) is not int
        or int(evidence["copied_artifact_count"]) < 1
        or type(evidence.get("copied_total_bytes")) is not int
        or int(evidence["copied_total_bytes"]) < 1
        or completion.get("retry_of_run_id") != authorization.source_run_id
        or completion.get("copy_policy") != evidence.get("copy_policy")
        or completion.get("copied_artifact_count") != evidence.get("copied_artifact_count")
        or completion.get("copied_total_bytes") != evidence.get("copied_total_bytes")
        or completion.get("retry_predecessor_binding_sha256") != sha256_file(evidence_path)
        or completion.get("primary_recovery_evidence_sha256") != sha256_file(evidence_path)
        or completion.get("recovery_evidence_sha256") != sha256_file(evidence_path)
        or completion.get("recovery_authorization_sha256") != authorization.canonical_sha256
        or completion.get("recovery_source_snapshot_root_sha256")
        != authorization.expected_source_snapshot_root_sha256
        or completion.get("completion_stage") != _COMPLETION_STAGE
        or completion.get("study_outcome_eligible") is not True
        or completion.get("post_seal_attestation_required") is not True
    ):
        raise PrimaryRecoveryRunnerError(
            "recovery statistics or completion lineage is not self-consistent"
        )
    return dict(evidence)


def _require_sealed_copy_manifest_matches_inspection(
    run_path: Path,
    inspection: OrphanSourceInspection,
) -> None:
    """Bind every imported byte to the final sealed artifact manifest.

    The destination typed readback happens before sealing.  This cheap manifest
    comparison closes the intervening mutation window without another full-tree
    hash pass: ``verify_run_integrity`` has already rehashed the sealed tree, while
    the source inspection carries the exact authorized allowlist digests.
    """

    manifest = _read_json_object(
        run_path / "artifact_manifest.json",
        "sealed recovery artifact manifest",
    )
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise PrimaryRecoveryRunnerError(
            "sealed recovery artifact manifest lacks its artifact records"
        )
    observed: dict[str, tuple[int, str]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or type(record.get("size_bytes")) is not int
            or not isinstance(record.get("sha256"), str)
        ):
            raise PrimaryRecoveryRunnerError(
                "sealed recovery artifact manifest has a malformed record"
            )
        relative_path = str(record["path"])
        if relative_path in observed:
            raise PrimaryRecoveryRunnerError(
                "sealed recovery artifact manifest repeats an artifact path"
            )
        observed[relative_path] = (
            int(record["size_bytes"]),
            str(record["sha256"]),
        )
    mismatched = [
        record.path
        for record in inspection.snapshot.artifacts
        if observed.get(record.path) != (record.size_bytes, record.sha256)
    ]
    if mismatched:
        raise PrimaryRecoveryRunnerError(
            f"sealed recovery copy differs from its authorized source snapshot: {mismatched[:5]}"
        )


def verify_primary_recovery_evidence(
    run_directory: str | Path,
    *,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
    completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the sealed recovery lineage without reopening outcome values."""

    run_path = Path(run_directory).resolve()
    controls.validate_for_plan(plan)
    integrity = verify_run_integrity(run_path)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_path.name
        or integrity.expected_root_sha256 != integrity.actual_root_sha256
    ):
        raise PrimaryRecoveryRunnerError(
            f"recovery run failed immutable integrity verification: {integrity.errors}"
        )
    status = _read_json_object(run_path / "status.json", "recovery status")
    sealed_completion = _read_json_object(
        run_path / "completion_evidence.json", "recovery completion"
    )
    if (
        status.get("status") != "completed"
        or status.get("experiment_name") != RECOVERY_EXPERIMENT_NAME
        or status.get("run_id") != run_path.name
        or (completion is not None and dict(completion) != sealed_completion)
    ):
        raise PrimaryRecoveryRunnerError("sealed directory is not the exact completed recovery")
    evidence = _read_json_object(
        run_path / RECOVERY_EVIDENCE_FILENAME,
        "primary recovery evidence",
    )
    verified = _validate_recovery_evidence_payload(
        evidence,
        run_path=run_path,
        plan=plan,
        authorization=authorization,
        completion=sealed_completion,
    )
    require_run_stage_eligible(run_path)
    return verified


def _failure_completion(
    tracker: RunTracker,
    error: BaseException,
    *,
    source_run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "completion_stage": None,
        "study_outcome_eligible": False,
        "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
        "run_id": tracker.run_id,
        "retry_of_run_id": source_run_id,
        "recovery_only": True,
        "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
        "outcomes_inspected": True,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
        "valid_completion_claim": False,
        "recovery_failure": f"{type(error).__name__}: {error}",
    }


def _fail_seal(
    tracker: RunTracker,
    error: BaseException,
    *,
    source_run_id: str,
) -> None:
    if tracker.finalized:
        return
    try:
        completion = _failure_completion(
            tracker,
            error,
            source_run_id=source_run_id,
        )
        tracker.write_json("completion_evidence.json", completion)
        tracker.write_metrics(
            {
                "schema_version": 1,
                "completion_stage": None,
                "study_outcome_eligible": False,
                "run_id": tracker.run_id,
                "retry_of_run_id": source_run_id,
                "recovery_only": True,
                "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
                "training_invoked": False,
                "matrix_executor_invoked": False,
                "fallback_invoked": False,
                "automatic_retry_allowed": False,
                "valid_completion_claim": False,
            }
        )
        tracker.write_text(
            "report.md",
            "# Failed primary orphan recovery\n\n"
            "No completion stage is claimed. No source annotation was modified.\n",
        )
        tracker.fail(error)
    except BaseException as seal_error:
        error.add_note(
            "recovery could not be safely failure-sealed and remains default-deny: "
            f"{type(seal_error).__name__}: {seal_error}"
        )


def _withdraw_postseal_failure(tracker: RunTracker, error: BaseException) -> None:
    try:
        withdraw_run_eligibility(
            tracker.run_directory,
            reason_code=_POSTSEAL_WITHDRAWAL_REASON_CODE,
            reason=_POSTSEAL_WITHDRAWAL_REASON,
        )
    except BaseException as withdrawal_error:
        error.add_note(
            "post-seal recovery withdrawal also failed; downstream gates must remain "
            f"default-deny: {type(withdrawal_error).__name__}: {withdrawal_error}"
        )


def _report_text(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# Recovered real PanNuke primary study",
            "",
            "This controlled benchmark ranks potentially inconsistent annotations "
            "recommended for expert review. It does not modify source annotations.",
            "",
            f"- Recovery disposition: `{metrics['analysis_disposition']}`",
            f"- Source run: `{metrics['retry_of_run_id']}`",
            f"- Required cells recovered: {metrics['completed_required_cell_count']}",
            f"- Optional cells skipped: {metrics['skipped_optional_cell_count']}",
            f"- Filesystem readback: `{metrics['filesystem_readback_status']}`",
            f"- Completion candidate: `{metrics['completion_stage']}`",
            "- No training, matrix execution, fallback, or automatic retry was invoked.",
            "- The candidate is valid only after seal, integrity verification, and "
            "positive post-seal stage attestation.",
            "",
        )
    )


def execute_primary_orphan_recovery(
    *,
    gate_evidence: PrimaryExecutionGateEvidence,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
    authorization: RecoveryAuthorization,
    project_root: str | Path | None = None,
    runs_root: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Recover one exact interrupted orphan under one new immutable tracked run."""

    run_root = Path(runs_root).resolve()
    root = Path(project_root or Path.cwd()).resolve()
    tracker: RunTracker | None = None
    proof: Any = None
    candidate: dict[str, Any] | None = None
    seal_attempted = False
    try:
        inspection, disk, statistics_api = _prepare_recovery(
            gate_evidence=gate_evidence,
            plan=plan,
            controls=controls,
            authorization=authorization,
            runs_root=run_root,
            run_id=run_id,
        )
        if controls.frozen_config_canonical_json is None:
            raise PrimaryRecoveryRunnerError("recovery controls lack canonical frozen config")
        frozen_config = json.loads(controls.frozen_config_canonical_json)
        if not isinstance(frozen_config, dict):
            raise PrimaryRecoveryRunnerError("canonical frozen config is not a JSON object")

        tracker = RunTracker.start(
            experiment_name=RECOVERY_EXPERIMENT_NAME,
            config=frozen_config,
            project_root=root,
            runs_root=run_root,
            run_id=run_id,
            duplicate_audit_status="passed",
        )
        gate_path = tracker.write_json(
            "primary_execution_gate.json",
            gate_evidence.as_dict(),
        )
        input_bindings = _recovery_input_bindings(
            gate=gate_evidence,
            plan=plan,
            controls=controls,
            inspection=inspection,
            disk=disk,
        )
        input_path = tracker.write_json(
            "primary_recovery_input_bindings.json",
            input_bindings,
        )
        tracker.write_provenance(
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            recovery_only=True,
            recovery_policy=RECOVERY_POLICY,
            retry_of_run_id=authorization.source_run_id,
            recovery_authorization_sha256=authorization.canonical_sha256,
            recovery_source_snapshot_root_sha256=inspection.snapshot.snapshot_root_sha256,
            analysis_disposition=RECOVERY_REGISTRATION_STATUS,
            outcomes_inspected=True,
            training_invoked=False,
            matrix_executor_invoked=False,
            fallback_invoked=False,
            automatic_retry_allowed=False,
            primary_execution_gate_sha256=sha256_file(gate_path),
            primary_recovery_input_bindings_sha256=sha256_file(input_path),
        )
        copy_receipt = copy_authorized_orphan_artifacts(
            inspection,
            tracker.run_directory,
        )
        _require_source_metadata_unchanged(inspection)
        destination = verify_recovery_destination(
            inspection,
            tracker.run_directory,
            plan=plan,
            controls=controls,
        )
        proof = _issue_recovery_numeric_proof(
            statistics_api,
            inspection=inspection,
            destination=destination,
        )
        statistics_verification, statistics_provenance = (
            statistics_api.attest_inherited_primary_statistics_artifacts(
                tracker.run_directory,
                controls,
                authorization=proof,
            )
        )
        _validate_recovery_downstream_attestations(
            run_directory=tracker.run_directory,
            readback=destination.filesystem_readback,
            statistics_verification=statistics_verification,
            restoration_readback=destination.restoration_readback,
        )
        statistics_provenance_payload = {
            "schema_version": 1,
            "policy": _NUMERIC_PROOF_POLICY,
            "recovery_authorization_sha256": authorization.canonical_sha256,
            "source_run_id": authorization.source_run_id,
            "source_snapshot_root_sha256": inspection.snapshot.snapshot_root_sha256,
            "verification": statistics_verification.as_dict(),
            "provenance": statistics_provenance.as_dict(),
        }
        statistics_provenance_path = tracker.write_json(
            "primary_recovery_statistics_verification.json",
            statistics_provenance_payload,
        )
        evidence = _build_recovery_evidence(
            tracker=tracker,
            inspection=inspection,
            destination=destination,
            copy_receipt=copy_receipt,
            statistics_verification=statistics_verification,
        )
        evidence_path = tracker.write_json(RECOVERY_EVIDENCE_FILENAME, evidence)
        evidence_sha256 = sha256_file(evidence_path)

        raw_candidate = build_primary_completion_evidence(
            plan=plan,
            reconciliation=destination.filesystem_readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=gate_evidence,
            filesystem_readback=destination.filesystem_readback,
            statistics_verification=statistics_verification,
            restoration_readback=destination.restoration_readback,
            _inherited_authorization=proof,
        )
        raw_candidate.update(
            {
                "execution_controls_binding_sha256": controls.binding_sha256,
                "run_id": tracker.run_id,
                "retry_of_run_id": authorization.source_run_id,
                "retry_predecessor_binding_sha256": evidence_sha256,
                "post_seal_integrity_verification_required": True,
                "post_seal_attestation_required": True,
            }
        )
        candidate = _validate_recovery_completion_candidate(
            raw_candidate,
            plan=plan,
            gate=gate_evidence,
            readback=destination.filesystem_readback,
            controls=controls,
            statistics_verification=statistics_verification,
            restoration_readback=destination.restoration_readback,
            run_id=tracker.run_id,
            source_run_id=authorization.source_run_id,
            lineage_sha256=evidence_sha256,
        )
        candidate.update(
            {
                "recovery_only": True,
                "recovery_policy": RECOVERY_POLICY,
                "primary_recovery_evidence_sha256": evidence_sha256,
                "recovery_evidence_sha256": evidence_sha256,
                "recovery_authorization_sha256": authorization.canonical_sha256,
                "recovery_source_snapshot_root_sha256": (inspection.snapshot.snapshot_root_sha256),
                "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
                "outcomes_inspected": True,
                "outcome_inspection_at_utc": authorization.outcome_inspection_at_utc,
                "verification_mode": statistics_verification.verification_mode,
                "prior_numeric_verification_proof_sha256": (
                    statistics_verification.prior_numeric_verification_proof_sha256
                ),
                "training_invoked": False,
                "matrix_executor_invoked": False,
                "fallback_invoked": False,
                "automatic_retry_allowed": False,
                "physical_copy_verified": True,
                "copy_policy": copy_receipt.copy_policy,
                "copied_artifact_count": copy_receipt.artifact_count,
                "copied_total_bytes": copy_receipt.total_bytes,
                "reused_required_cell_count": (
                    destination.filesystem_readback.completed_required_cell_count
                ),
                "retrained_cell_count": 0,
            }
        )
        core_completion = build_primary_completion_evidence(
            plan=plan,
            reconciliation=destination.filesystem_readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
        )
        core_completion.update(
            {
                "execution_controls_binding_sha256": controls.binding_sha256,
                "run_id": tracker.run_id,
                "retry_of_run_id": authorization.source_run_id,
                "recovery_only": True,
                "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
            }
        )
        tracker.write_json("core_completion_evidence.json", core_completion)
        completion_path = tracker.write_json("completion_evidence.json", candidate)
        metrics = {
            "schema_version": 1,
            "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
            "study_outcome_eligible": True,
            "completion_stage": _COMPLETION_STAGE,
            "run_id": tracker.run_id,
            "retry_of_run_id": authorization.source_run_id,
            "retry_predecessor_binding_sha256": evidence_sha256,
            "recovery_only": True,
            "recovery_policy": RECOVERY_POLICY,
            "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
            "outcomes_inspected": True,
            "training_invoked": False,
            "matrix_executor_invoked": False,
            "fallback_invoked": False,
            "automatic_retry_allowed": False,
            "physical_copy_verified": True,
            "copy_policy": copy_receipt.copy_policy,
            "copied_artifact_count": copy_receipt.artifact_count,
            "copied_total_bytes": copy_receipt.total_bytes,
            "matrix_config_sha256": plan.config_sha256,
            "matrix_plan_sha256": controls.plan_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "planned_cell_count": len(plan.cells),
            "required_cell_count": plan.required_cell_count,
            "completed_required_cell_count": (
                destination.filesystem_readback.completed_required_cell_count
            ),
            "skipped_optional_cell_count": (
                destination.filesystem_readback.skipped_optional_cell_count
            ),
            "filesystem_readback_status": destination.filesystem_readback.status,
            "filesystem_readback_root_sha256": (
                destination.filesystem_readback.readback_root_sha256
            ),
            "primary_statistics_manifest_sha256": statistics_verification.manifest_sha256,
            "primary_statistics_comparison_count": statistics_verification.comparison_count,
            "primary_restoration_readback_root_sha256": (
                destination.restoration_readback.readback_root_sha256
            ),
            "verification_mode": statistics_verification.verification_mode,
            "prior_numeric_verification_proof_sha256": (
                statistics_verification.prior_numeric_verification_proof_sha256
            ),
            "completion_evidence_sha256": sha256_file(completion_path),
            "post_seal_integrity_verification_required": True,
            "post_seal_attestation_required": True,
            "valid_completion_claim": "pending_post_seal_verification",
        }
        tracker.write_metrics(metrics)
        tracker.write_text("report.md", _report_text(metrics))
        tracker.write_provenance(
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            recovery_only=True,
            recovery_policy=RECOVERY_POLICY,
            retry_of_run_id=authorization.source_run_id,
            retry_predecessor_binding_sha256=evidence_sha256,
            primary_recovery_evidence_sha256=evidence_sha256,
            recovery_evidence_sha256=evidence_sha256,
            recovery_authorization_sha256=authorization.canonical_sha256,
            recovery_source_snapshot_root_sha256=inspection.snapshot.snapshot_root_sha256,
            source_run_directory=str(authorization.source_run_directory),
            source_status_sha256=inspection.source_status_sha256,
            source_primary_execution_gate_sha256=inspection.source_primary_gate_sha256,
            source_tree_manifest_sha256=inspection.source_tree_manifest_sha256,
            source_tree_root_sha256=inspection.source_tree_root_sha256,
            source_filesystem_readback_root_sha256=(
                inspection.snapshot.filesystem_readback.readback_root_sha256
            ),
            source_restoration_readback_root_sha256=(
                inspection.snapshot.restoration_readback.readback_root_sha256
            ),
            source_statistics_manifest_sha256=(inspection.snapshot.statistics_manifest_sha256),
            destination_filesystem_readback_root_sha256=(
                destination.filesystem_readback.readback_root_sha256
            ),
            destination_restoration_readback_root_sha256=(
                destination.restoration_readback.readback_root_sha256
            ),
            destination_statistics_manifest_sha256=(destination.statistics_manifest_sha256),
            authority_directory=str(authorization.authority_directory),
            authority_artifact_root_sha256=authorization.authority_artifact_root_sha256,
            authority_manifest_sha256=authorization.authority_manifest_sha256,
            primary_execution_gate_sha256=sha256_file(gate_path),
            primary_recovery_input_bindings_sha256=sha256_file(input_path),
            disk_preflight=disk.as_dict(),
            analysis_disposition=RECOVERY_REGISTRATION_STATUS,
            outcomes_inspected=True,
            outcome_inspection_at_utc=authorization.outcome_inspection_at_utc,
            verification_mode=statistics_verification.verification_mode,
            prior_numeric_verification_proof_sha256=(
                statistics_verification.prior_numeric_verification_proof_sha256
            ),
            statistics_verification_provenance_sha256=canonical_sha256(
                statistics_provenance.as_dict()
            ),
            primary_recovery_statistics_verification_sha256=sha256_file(statistics_provenance_path),
            training_invoked=False,
            matrix_executor_invoked=False,
            fallback_invoked=False,
            automatic_retry_allowed=False,
            physical_copy_verified=True,
            reused_required_cell_count=(
                destination.filesystem_readback.completed_required_cell_count
            ),
            skipped_optional_cell_count=(
                destination.filesystem_readback.skipped_optional_cell_count
            ),
            retrained_cell_count=0,
            copy_policy=copy_receipt.copy_policy,
            copied_artifact_count=copy_receipt.artifact_count,
            copied_total_bytes=copy_receipt.total_bytes,
            post_seal_attestation_required=True,
        )
        tracker.log_event(
            "primary_orphan_recovery_candidate_written",
            retry_of_run_id=authorization.source_run_id,
            primary_recovery_evidence_sha256=evidence_sha256,
            recovery_authorization_sha256=authorization.canonical_sha256,
            training_invoked=False,
            matrix_executor_invoked=False,
            fallback_invoked=False,
            automatic_retry_allowed=False,
        )
        seal_attempted = True
        tracker.complete()
    except BaseException as exc:
        if isinstance(exc, AnchoredPhysicalCopyBoundaryError):
            expected = Path(os.path.abspath(exc.expected_destination_root))
            raise PrimaryRecoveryRunnerError(
                "anchored recovery copy boundary failed; no pathname-based demotion, "
                "failure seal, withdrawal, cleanup, or retry was attempted. The exact "
                f"expected path remains a default-deny ambiguous orphan: {expected}: {exc}",
                run_id=tracker.run_id if tracker is not None else run_id,
                run_directory=expected,
            ) from exc
        if tracker is not None:
            if tracker.finalized:
                _withdraw_postseal_failure(tracker, exc)
            elif not seal_attempted:
                _fail_seal(
                    tracker,
                    exc,
                    source_run_id=authorization.source_run_id,
                )
        if isinstance(exc, Exception):
            raise PrimaryRecoveryRunnerError(
                f"primary orphan recovery failed: {type(exc).__name__}: {exc}",
                run_id=tracker.run_id if tracker is not None else None,
                run_directory=tracker.run_directory if tracker is not None else None,
            ) from exc
        raise

    assert tracker is not None
    assert candidate is not None
    assert proof is not None
    try:
        integrity: IntegrityVerification = verify_run_integrity(tracker.run_directory)
        if (
            not integrity.valid
            or not integrity.registry_record_present
            or integrity.run_id != tracker.run_id
            or integrity.expected_root_sha256 != integrity.actual_root_sha256
        ):
            raise PrimaryRecoveryRunnerError(
                f"sealed recovery failed post-seal integrity: {integrity.errors}"
            )
        _require_sealed_copy_manifest_matches_inspection(
            tracker.run_directory,
            inspection,
        )
        sealed_completion = _read_json_object(
            tracker.run_directory / "completion_evidence.json",
            "sealed recovery completion",
        )
        if sealed_completion != candidate:
            raise PrimaryRecoveryRunnerError(
                "sealed recovery completion differs from the verified candidate"
            )
        lineage = _validate_recovery_evidence_payload(
            _read_json_object(
                tracker.run_directory / RECOVERY_EVIDENCE_FILENAME,
                "sealed primary recovery evidence",
            ),
            run_path=tracker.run_directory,
            plan=plan,
            authorization=authorization,
            completion=sealed_completion,
        )
        stage_verification = _build_primary_stage_attestation_verification(
            tracker.run_directory,
            integrity=integrity,
            completion=sealed_completion,
            filesystem_readback=destination.filesystem_readback,
            statistics_verification=statistics_verification,
            restoration_readback=destination.restoration_readback,
            lineage_verification=lineage,
        )
        completion_evidence_sha256 = sha256_file(tracker.run_directory / "completion_evidence.json")
        recovery_evidence_sha256 = sha256_file(tracker.run_directory / RECOVERY_EVIDENCE_FILENAME)
        result = {
            "status": "completed",
            "completion_stage": _COMPLETION_STAGE,
            "study_outcome_eligible": True,
            "analysis_disposition": RECOVERY_REGISTRATION_STATUS,
            "outcomes_inspected": True,
            "run_id": tracker.run_id,
            "run_directory": str(tracker.run_directory),
            "retry_of_run_id": authorization.source_run_id,
            "artifact_root_sha256": integrity.expected_root_sha256,
            "registry_record_present": integrity.registry_record_present,
            "completion_evidence_path": str(tracker.run_directory / "completion_evidence.json"),
            "completion_evidence_sha256": completion_evidence_sha256,
            "primary_recovery_evidence_path": str(
                tracker.run_directory / RECOVERY_EVIDENCE_FILENAME
            ),
            "primary_recovery_evidence_sha256": recovery_evidence_sha256,
            "source_snapshot_root_sha256": (authorization.expected_source_snapshot_root_sha256),
            "planned_cell_count": len(plan.cells),
            "completed_required_cell_count": plan.required_cell_count,
            "training_invoked": False,
            "matrix_executor_invoked": False,
            "fallback_invoked": False,
            "automatic_retry_allowed": False,
        }
        # This is the terminal commit.  Every fallible file read and validation
        # above has already completed, so a positive stage attestation is never
        # followed by another verifier that could turn success into failure.
        stage_attestation = attest_primary_run_stage_eligibility(
            tracker.run_directory,
            verification=stage_verification,
        )
        result["stage_attestation_record_sha256"] = stage_attestation["record_sha256"]
        return result
    except BaseException as exc:
        _withdraw_postseal_failure(tracker, exc)
        if isinstance(exc, Exception):
            raise PrimaryRecoveryRunnerError(
                "sealed primary orphan recovery was permanently withdrawn after "
                f"post-seal failure: {type(exc).__name__}: {exc}",
                run_id=tracker.run_id,
                run_directory=tracker.run_directory,
            ) from exc
        raise


__all__ = (
    "PrimaryRecoveryRunnerError",
    "RecoveryDiskPreflight",
    "execute_primary_orphan_recovery",
    "preflight_primary_orphan_recovery",
    "verify_primary_recovery_evidence",
)
