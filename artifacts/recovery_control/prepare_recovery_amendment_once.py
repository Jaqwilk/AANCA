"""Publish the one authorized post-outcome orphan-recovery amendment.

This control script is deliberately outside the frozen execution-source scope. It
prints only provenance identities, never scientific outcome values.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from histo_audit.config import load_config
from histo_audit.experiment.primary_core import (
    primary_execution_controls_from_frozen_config,
)
from histo_audit.experiment.primary_recovery import (
    build_primary_recovery_authorization,
    collect_orphan_source_snapshot,
)
from histo_audit.experiment.primary_statistics import (
    INHERITED_PRIOR_NUMERIC_LIMITATION,
    INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
)
from histo_audit.experiment.study_contracts import build_primary_matrix_plan
from histo_audit.workflows.preregistration_amendment import (
    ConfirmatoryStoragePolicy,
    create_preregistration_amendment,
    verify_preregistration_amendment,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = (ROOT / "artifacts" / "preregistration_amendments" / "20260719T011146.248393Z").resolve()
SOURCE = (
    ROOT
    / "artifacts"
    / "runs"
    / "20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
).resolve()
RECEIPT = (
    ROOT
    / "artifacts"
    / "process_observations"
    / "20260727T110704.4225163Z_primary_orphan_boot_receipt.json"
).resolve()
AMENDMENT_ROOT = (ROOT / "artifacts" / "preregistration_amendments").resolve()
REASON = (
    "Authorize zero-training recovery of the host-reboot-interrupted unsealed "
    "primary after accidental outcome exposure, preserving every frozen scientific "
    "method and classifying the recovered analyses as amended_or_exploratory."
)


def _reject_existing_recovery_authority() -> None:
    for evidence_path in AMENDMENT_ROOT.glob("*/amendment_evidence.json"):
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "primary_recovery_authorization" in payload:
            raise RuntimeError(
                "a primary-recovery amendment already exists; automatic duplication is forbidden: "
                f"{evidence_path.parent.resolve()}"
            )


def main() -> None:
    _reject_existing_recovery_authority()
    primary_path = PARENT / "primary_frozen.yaml"
    confirmatory_path = PARENT / "confirmatory_frozen.yaml"
    preregistration_path = PARENT / "PRE_REGISTRATION_FROZEN.md"

    config = load_config(primary_path)
    plan = build_primary_matrix_plan(config)
    controls = primary_execution_controls_from_frozen_config(config)
    controls.validate_for_plan(plan)

    snapshot = collect_orphan_source_snapshot(
        SOURCE,
        plan=plan,
        controls=controls,
    )
    authorization = build_primary_recovery_authorization(
        snapshot,
        interruption_receipt_path=RECEIPT,
        interruption_observed_at_utc="2026-07-27T11:07:04.4225163Z",
        last_boot_at_utc="2026-07-27T10:37:04.5000000Z",
        event_id=12,
        source_process_id=20792,
        process_checked_at_utc="2026-07-27T11:07:04.4225163Z",
        outcome_inspection_at_utc="2026-07-27T10:57:07.000000Z",
        trust_assumption=INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
        limitation=INHERITED_PRIOR_NUMERIC_LIMITATION,
        reason=REASON,
    )

    result = create_preregistration_amendment(
        project_root=ROOT,
        parent_authority_directory=PARENT,
        amendment_root=AMENDMENT_ROOT,
        preregistration_path=preregistration_path,
        primary_config_path=primary_path,
        confirmatory_config_path=confirmatory_path,
        reason=REASON,
        affected_hypotheses=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
        affected_analyses=(
            "primary_frozen_feature_benchmark_all_registered_cells",
            "primary_restoration_analysis",
            "primary_registered_statistics",
            "confirmatory_primary_dependency_gate",
            "confirmatory_single_copy_checkpoint_storage",
            "primary_orphan_recovery_execution_lineage",
        ),
        outcomes_inspected=True,
        outcomes_inspected_at=datetime.fromisoformat("2026-07-27T10:57:07+00:00"),
        primary_recovery_authorization=authorization,
        confirmatory_storage_policy=ConfirmatoryStoragePolicy(),
        timestamp=datetime.now(UTC),
    )
    verification = verify_preregistration_amendment(result.amendment_directory)
    if not verification.valid:
        raise RuntimeError(
            "new recovery amendment failed immediate independent verification: "
            + "; ".join(verification.errors)
        )
    print(
        json.dumps(
            {
                "status": "published_and_verified",
                "amendment_directory": str(result.amendment_directory),
                "artifact_root_sha256": result.artifact_root_sha256,
                "sha256_manifest_sha256": result.sha256_manifest_sha256,
                "recovery_source_run_id": SOURCE.name,
                "source_snapshot_root_sha256": snapshot.snapshot_root_sha256,
                "completed_required_cell_count": snapshot.completed_required_cell_count,
                "skipped_optional_cell_count": snapshot.skipped_optional_cell_count,
                "analysis_disposition": "amended_or_exploratory",
                "confirmatory_storage_policy_bound": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
