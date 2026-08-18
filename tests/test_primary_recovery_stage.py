"""Focused fail-closed stage tests for interrupted-primary orphan recovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.experiment.primary_completion as completion_module
import histo_audit.experiment.primary_statistics as statistics_module
import histo_audit.utils.run_tracking as run_tracking
import histo_audit.workflows.preregistration_amendment as amendment_module
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import (
    PrimaryFilesystemReadbackEvidence,
    PrimaryMatrixReconciliation,
    PrimaryRestorationReadbackEvidence,
)
from histo_audit.experiment.primary_recovery import RECOVERY_COPY_POLICY
from histo_audit.experiment.primary_statistics import (
    InheritedPrimaryStatisticsVerification,
    PrimaryStatisticsVerification,
)
from histo_audit.utils.run_tracking import (
    PrimaryStageAttestationVerification,
    RunTracker,
    attest_primary_run_stage_eligibility,
    require_run_stage_eligible,
    sha256_file,
    verify_run_integrity,
)

_EXPERIMENT = "pannuke_primary_orphan_recovery"
_POLICY = "interrupted_unsealed_primary_recovery_v1"
_COPY_POLICY = RECOVERY_COPY_POLICY
_ATTESTATION_POLICY = "primary_orphan_recovery_postseal_attestation_v1"
_DISPOSITION = "amended_or_exploratory"
_VERIFICATION_MODE = "inherited_prior_numeric_verification_v1"
_EVIDENCE_FIELDS = {
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
    "analysis_disposition",
    "outcomes_inspected",
    "verification_mode",
    "prior_numeric_verification_proof_sha256",
    "training_invoked",
    "matrix_executor_invoked",
    "fallback_invoked",
    "automatic_retry_allowed",
}


def _sealed_recovery_candidate(
    root: Path,
    *,
    evidence_overrides: dict[str, Any] | None = None,
    execution_overrides: dict[str, Any] | None = None,
) -> tuple[
    RunTracker,
    PrimaryStageAttestationVerification,
    dict[str, Any],
]:
    source_run_id = "interrupted-unsealed-primary"
    source_snapshot_root = "a" * 64
    proof_sha256 = "b" * 64
    amendment = (root / "recovery-amendment").resolve()
    amendment.mkdir()
    authorization = {
        "schema_version": 1,
        "policy": _POLICY,
        "source_run_id": source_run_id,
        "expected_source_snapshot_root_sha256": source_snapshot_root,
    }
    authorization_sha256 = canonical_sha256(authorization)
    execution_fields: dict[str, Any] = {
        "recovery_policy": _POLICY,
        "retry_of_run_id": source_run_id,
        "recovery_source_snapshot_root_sha256": source_snapshot_root,
        "recovery_authorization_sha256": authorization_sha256,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    execution_fields.update(execution_overrides or {})

    tracker = RunTracker.start(
        experiment_name=_EXPERIMENT,
        config={"experiment_name": _EXPERIMENT, "seed": {}},
        project_root=root,
        runs_root=root / "artifacts" / "runs",
        environment={},
    )
    files = {
        "matrix_plan.json": "{}\n",
        "execution_controls.json": "{}\n",
        "cell_index.csv": "cell_id,status\n",
        "primary_statistics.json": "{}\n",
        "primary_bootstrap_evidence.npz": "bootstrap",
        "primary_subgroups.csv": "cell_id,status\n",
        "primary_statistics_manifest.json": "{}\n",
        "restoration_index.json": "{}\n",
    }
    for relative_path, content in files.items():
        tracker.write_text(relative_path, content)
    tracker.write_json(
        "primary_execution_gate.json",
        {"freeze_directory": str(amendment)},
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "policy": _POLICY,
        "experiment_name": _EXPERIMENT,
        "source_run_id": source_run_id,
        "destination_run_id": tracker.run_id,
        "recovery_authorization_sha256": authorization_sha256,
        "source_snapshot_root_sha256": source_snapshot_root,
        "destination_snapshot_root_sha256": source_snapshot_root,
        "analysis_disposition": _DISPOSITION,
        "outcomes_inspected": True,
        "verification_mode": _VERIFICATION_MODE,
        "prior_numeric_verification_proof_sha256": proof_sha256,
        "reused_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "retrained_cell_count": 0,
        "copy_policy": _COPY_POLICY,
        "copied_artifact_count": 2_270,
        "copied_total_bytes": 46_291_340_622,
        "training_invoked": False,
        "matrix_executor_invoked": False,
        "fallback_invoked": False,
        "automatic_retry_allowed": False,
    }
    evidence.update(evidence_overrides or {})
    evidence_path = tracker.write_json("primary_recovery_evidence.json", evidence)
    evidence_sha256 = sha256_file(evidence_path)
    execution_fields["primary_recovery_evidence_sha256"] = evidence_sha256
    execution_fields["recovery_evidence_sha256"] = evidence_sha256
    tracker.write_json(
        "run_provenance.json",
        execution_fields,
    )
    tracker.write_json(
        "completion_evidence.json",
        {
            "schema_version": 1,
            "run_id": tracker.run_id,
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "study_outcome_eligible": True,
            "post_seal_attestation_required": True,
            "recovery_only": True,
            "retry_predecessor_binding_sha256": evidence_sha256,
            **execution_fields,
        },
    )
    tracker.complete()

    integrity = verify_run_integrity(tracker.run_directory)
    assert integrity.valid and integrity.expected_root_sha256 is not None
    verification = PrimaryStageAttestationVerification(
        policy=_ATTESTATION_POLICY,
        experiment_name=_EXPERIMENT,
        run_id=tracker.run_id,
        run_path=str(tracker.run_directory),
        completion_stage="PRIMARY_STUDY_COMPLETE",
        first_integrity_root_sha256=integrity.expected_root_sha256,
        final_integrity_root_sha256=integrity.expected_root_sha256,
        artifact_manifest_sha256=sha256_file(tracker.run_directory / "artifact_manifest.json"),
        completion_evidence_sha256=sha256_file(tracker.run_directory / "completion_evidence.json"),
        matrix_plan_sha256=sha256_file(tracker.run_directory / "matrix_plan.json"),
        execution_controls_sha256=sha256_file(tracker.run_directory / "execution_controls.json"),
        cell_index_sha256=sha256_file(tracker.run_directory / "cell_index.csv"),
        filesystem_readback_root_sha256="c" * 64,
        primary_statistics_sha256=sha256_file(tracker.run_directory / "primary_statistics.json"),
        primary_statistics_size_bytes=(tracker.run_directory / "primary_statistics.json")
        .stat()
        .st_size,
        primary_bootstrap_evidence_sha256=sha256_file(
            tracker.run_directory / "primary_bootstrap_evidence.npz"
        ),
        primary_bootstrap_evidence_size_bytes=(
            tracker.run_directory / "primary_bootstrap_evidence.npz"
        )
        .stat()
        .st_size,
        primary_subgroups_sha256=sha256_file(tracker.run_directory / "primary_subgroups.csv"),
        primary_subgroups_size_bytes=(tracker.run_directory / "primary_subgroups.csv")
        .stat()
        .st_size,
        primary_statistics_manifest_sha256=sha256_file(
            tracker.run_directory / "primary_statistics_manifest.json"
        ),
        primary_statistics_manifest_size_bytes=(
            tracker.run_directory / "primary_statistics_manifest.json"
        )
        .stat()
        .st_size,
        primary_statistics_source_readback_root_sha256="c" * 64,
        primary_statistics_comparison_count=1,
        primary_restoration_index_sha256=sha256_file(
            tracker.run_directory / "restoration_index.json"
        ),
        primary_restoration_readback_root_sha256="d" * 64,
        retry_of_run_id=source_run_id,
        lineage_binding_sha256=evidence_sha256,
        authorization_binding_sha256=authorization_sha256,
    )
    object.__setattr__(
        verification,
        "_attestation",
        run_tracking._PRIMARY_STAGE_ATTESTATION_TOKEN,
    )
    return tracker, verification, authorization


def _patch_recovery_authority(
    monkeypatch: pytest.MonkeyPatch,
    authorization: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        amendment_module,
        "require_primary_recovery_authorization",
        lambda _: dict(authorization),
    )


def test_runner_emits_the_exact_closed_stage_lineage_schema() -> None:
    from histo_audit.experiment.primary_recovery_runner import (
        _build_recovery_evidence,
    )

    evidence = _build_recovery_evidence(
        tracker=SimpleNamespace(run_id="recovery-run"),
        inspection=SimpleNamespace(
            authorization=SimpleNamespace(
                source_run_id="orphan-run",
                canonical_sha256="1" * 64,
            ),
            snapshot=SimpleNamespace(snapshot_root_sha256="2" * 64),
        ),
        destination=SimpleNamespace(
            snapshot_root_sha256="2" * 64,
            filesystem_readback=SimpleNamespace(
                completed_required_cell_count=185,
                skipped_optional_cell_count=37,
            ),
        ),
        copy_receipt=SimpleNamespace(
            copy_policy=_COPY_POLICY,
            artifact_count=2_270,
            total_bytes=46_291_340_622,
        ),
        statistics_verification=SimpleNamespace(
            verification_mode=_VERIFICATION_MODE,
            prior_numeric_verification_proof_sha256="3" * 64,
        ),
    )

    assert set(evidence) == _EVIDENCE_FIELDS
    assert evidence["reused_required_cell_count"] == 185
    assert evidence["skipped_optional_cell_count"] == 37
    assert evidence["retrained_cell_count"] == 0


def test_recovery_gets_exactly_one_completed_postseal_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, verification, authorization = _sealed_recovery_candidate(tmp_path)
    _patch_recovery_authority(monkeypatch, authorization)

    record = attest_primary_run_stage_eligibility(
        tracker.run_directory,
        verification=verification,
    )

    assert record["completion_stage"] == "PRIMARY_STUDY_COMPLETE"
    assert record["verification"]["policy"] == _ATTESTATION_POLICY
    assert record["verification"]["retry_of_run_id"] == authorization["source_run_id"]
    assert record["verification"]["lineage_binding_sha256"] == sha256_file(
        tracker.run_directory / "primary_recovery_evidence.json"
    )
    assert require_run_stage_eligible(tracker.run_directory) == record
    with pytest.raises(ValueError, match="already has"):
        attest_primary_run_stage_eligibility(
            tracker.run_directory,
            verification=verification,
        )


def test_recovery_attestation_reconciles_exact_commit_then_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, verification, authorization = _sealed_recovery_candidate(tmp_path)
    _patch_recovery_authority(monkeypatch, authorization)
    real_atomic_write_json = run_tracking.atomic_write_json
    injected = 0

    def commit_then_raise(path: Any, payload: Any, **kwargs: Any) -> Path:
        nonlocal injected
        result = real_atomic_write_json(path, payload, **kwargs)
        if Path(path).name == run_tracking.RUN_STAGE_ATTESTATION_ANCHOR_FILENAME:
            injected += 1
            raise OSError("injected post-commit durability report")
        return result

    monkeypatch.setattr(run_tracking, "atomic_write_json", commit_then_raise)
    record = attest_primary_run_stage_eligibility(
        tracker.run_directory,
        verification=verification,
    )
    assert injected == 1
    assert record["run_id"] == tracker.run_id
    assert require_run_stage_eligible(tracker.run_directory) == record


@pytest.mark.parametrize(
    ("evidence_overrides", "execution_overrides"),
    [
        ({"unexpected_field": "forbidden"}, {}),
        ({"destination_snapshot_root_sha256": "f" * 64}, {}),
        ({"training_invoked": True}, {"training_invoked": True}),
        ({"fallback_invoked": True}, {"fallback_invoked": True}),
        (
            {"reused_required_cell_count": 184},
            {"reused_required_cell_count": 184},
        ),
        (
            {"skipped_optional_cell_count": 36},
            {"skipped_optional_cell_count": 36},
        ),
        ({"retrained_cell_count": 1}, {"retrained_cell_count": 1}),
        ({"copy_policy": "copy_fallback"}, {"copy_policy": "copy_fallback"}),
        ({"copied_artifact_count": 0}, {"copied_artifact_count": 0}),
        ({"copied_total_bytes": 0}, {"copied_total_bytes": 0}),
        ({}, {"predecessor_artifact_root_sha256": "e" * 64}),
        (
            {"automatic_retry_allowed": True},
            {"automatic_retry_allowed": True},
        ),
    ],
)
def test_recovery_attestation_rejects_tampered_or_executing_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_overrides: dict[str, Any],
    execution_overrides: dict[str, Any],
) -> None:
    tracker, verification, authorization = _sealed_recovery_candidate(
        tmp_path,
        evidence_overrides=evidence_overrides,
        execution_overrides=execution_overrides,
    )
    _patch_recovery_authority(monkeypatch, authorization)

    with pytest.raises(ValueError, match=r"orphan.?recover"):
        attest_primary_run_stage_eligibility(
            tracker.run_directory,
            verification=verification,
        )


def test_recovery_attestation_rejects_changed_amendment_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, verification, authorization = _sealed_recovery_candidate(tmp_path)
    changed_authorization = {
        **authorization,
        "expected_source_snapshot_root_sha256": "e" * 64,
    }
    _patch_recovery_authority(monkeypatch, changed_authorization)

    with pytest.raises(ValueError, match=r"orphan.?recover"):
        attest_primary_run_stage_eligibility(
            tracker.run_directory,
            verification=verification,
        )


def test_recovery_stage_builder_rejects_fresh_statistics_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, _, _ = _sealed_recovery_candidate(tmp_path)
    run_path = tracker.run_directory
    reconciliation = PrimaryMatrixReconciliation(
        status="passed",
        planned_cell_count=0,
        planned_required_cell_count=0,
        completed_cell_count=0,
        completed_required_cell_count=0,
        skipped_optional_cell_count=0,
        failed_cell_count=0,
        missing_cell_ids=(),
        extra_cell_ids=(),
        duplicate_cell_ids=(),
        invalid_cell_ids=(),
        errors=(),
    )
    filesystem = PrimaryFilesystemReadbackEvidence(
        run_directory=run_path,
        status="passed",
        matrix_plan_sha256="a" * 64,
        execution_controls_sha256="b" * 64,
        execution_controls_binding_sha256="c" * 64,
        cell_index_sha256="d" * 64,
        readback_root_sha256="e" * 64,
        planned_cell_count=0,
        completed_cell_count=0,
        completed_required_cell_count=0,
        skipped_optional_cell_count=0,
        circularity_excluded_cell_ids=(),
        cell_artifact_manifest_sha256=(),
        scenario_artifact_sha256=(),
        scenario_corruption_sha256=(),
        reconciliation=reconciliation,
    )
    object.__setattr__(
        filesystem,
        "_attestation",
        completion_module._FILESYSTEM_READBACK_ATTESTATION,
    )
    restoration = PrimaryRestorationReadbackEvidence(
        run_directory=run_path,
        status="passed",
        restoration_index_sha256="f" * 64,
        readback_root_sha256="1" * 64,
        source_readback_root_sha256=filesystem.readback_root_sha256,
        restoration_cell_count=0,
        downstream_comparison_count=0,
        cell_json_sha256=(),
        cell_evidence_sha256=(),
        cell_manifest_sha256=(),
    )
    object.__setattr__(
        restoration,
        "_attestation",
        completion_module._RESTORATION_READBACK_ATTESTATION,
    )
    fresh_statistics = PrimaryStatisticsVerification(
        status="passed",
        output_directory=run_path,
        statistics_sha256="2" * 64,
        bootstrap_evidence_sha256="3" * 64,
        subgroups_sha256="4" * 64,
        manifest_sha256="5" * 64,
        source_readback_root_sha256=filesystem.readback_root_sha256,
        comparison_count=1,
    )
    object.__setattr__(
        fresh_statistics,
        "_attestation",
        statistics_module._STATISTICS_VERIFICATION_ATTESTATION,
    )
    integrity = verify_run_integrity(run_path)
    monkeypatch.setattr(
        run_tracking,
        "_read_sealed_json_object",
        lambda _path, _role: {
            "status": "completed",
            "experiment_name": _EXPERIMENT,
        },
    )

    with pytest.raises(
        ValueError,
        match="requires inherited statistics verification only",
    ):
        run_tracking._build_primary_stage_attestation_verification(
            run_path,
            integrity=integrity,
            completion={},
            filesystem_readback=filesystem,
            statistics_verification=fresh_statistics,
            restoration_readback=restoration,
        )

    wrong_inherited = InheritedPrimaryStatisticsVerification(
        status="passed",
        output_directory=run_path,
        statistics_sha256="2" * 64,
        bootstrap_evidence_sha256="3" * 64,
        subgroups_sha256="4" * 64,
        manifest_sha256="5" * 64,
        source_readback_root_sha256=filesystem.readback_root_sha256,
        comparison_count=1,
        verification_mode=_VERIFICATION_MODE,
        prior_numeric_verification_proof_sha256="6" * 64,
    )
    object.__setattr__(
        wrong_inherited,
        "_attestation",
        statistics_module._INHERITED_STATISTICS_VERIFICATION_ATTESTATION,
    )
    object.__setattr__(
        wrong_inherited,
        "_authorization_kind",
        "finalization_successor",
    )
    assert wrong_inherited.valid
    with pytest.raises(
        ValueError,
        match="requires an orphan-recovery numeric proof",
    ):
        run_tracking._build_primary_stage_attestation_verification(
            run_path,
            integrity=integrity,
            completion={},
            filesystem_readback=filesystem,
            statistics_verification=wrong_inherited,
            restoration_readback=restoration,
        )
