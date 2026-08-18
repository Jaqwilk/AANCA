"""Fast orchestration tests for the one-shot primary orphan recovery runner."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.experiment.primary_core as primary_core_module
import histo_audit.experiment.primary_recovery_runner as runner_module
import histo_audit.experiment.primary_statistics as primary_statistics_module
from histo_audit.experiment.primary_core import PrimaryExecutionControls
from histo_audit.experiment.primary_recovery import (
    RECOVERY_EVIDENCE_FILENAME,
    RECOVERY_REGISTRATION_STATUS,
    RecoveryAuthorization,
    inspect_orphan_source,
)
from histo_audit.experiment.primary_recovery_runner import (
    PrimaryRecoveryRunnerError,
    execute_primary_orphan_recovery,
    preflight_primary_orphan_recovery,
)
from histo_audit.experiment.primary_statistics import (
    INHERITED_PRIOR_NUMERIC_LIMITATION,
    INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
)
from histo_audit.pannuke.publication import AnchoredPhysicalCopyBoundaryError
from histo_audit.utils.run_tracking import (
    _ensure_run_disposition_anchor,
    _ensure_run_stage_attestation_anchor,
    read_run_dispositions,
    sha256_file,
)
from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence
from tests.test_primary_recovery import _make_fixture, _tree_state, _write_json


@dataclass(frozen=True, slots=True)
class _RunnerFixture:
    runs_root: Path
    source: Path
    plan: Any
    controls: PrimaryExecutionControls
    gate: PrimaryExecutionGateEvidence
    authorization: RecoveryAuthorization


def _runner_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _RunnerFixture:
    runs_root = tmp_path / "runs"
    _ensure_run_disposition_anchor(runs_root)
    _ensure_run_stage_attestation_anchor(runs_root)
    base = _make_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        PrimaryExecutionControls,
        "validate_for_plan",
        lambda self, supplied: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_require_authorization_matches_immutable_amendment",
        lambda _authorization: None,
    )
    controls = replace(base.controls, frozen_config_canonical_json="{}")
    authority = tmp_path / "recovery-authority"
    base_freeze = tmp_path / "base-freeze"
    authority.mkdir()
    base_freeze.mkdir()
    gate = PrimaryExecutionGateEvidence(
        freeze_directory=authority.resolve(),
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256=base.plan.config_sha256,
        confirmatory_config_semantic_sha256="6" * 64,
        primary_matrix_cell_count=len(base.plan.cells),
        primary_required_cell_count=base.plan.required_cell_count,
        confirmatory_matrix_cell_count=1,
        pilot_run_id="tiny-pilot",
        pilot_artifact_root_sha256="7" * 64,
        dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        duplicate_audit_sha256="a" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="d" * 64,
        base_freeze_directory=base_freeze.resolve(),
        registration_authority_kind="amendment",
        registration_status=RECOVERY_REGISTRATION_STATUS,
        registration_authority_chain_depth=1,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=False,
    )
    _write_json(base.source / "primary_execution_gate.json", gate.as_dict())
    mapping = copy.deepcopy(base.authorization_mapping)
    mapping["expected_primary_execution_gate_sha256"] = sha256_file(
        base.source / "primary_execution_gate.json"
    )
    mapping["interruption_evidence"]["source_process_id"] = 999_999_937
    mapping["trust_assumption"] = INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
    mapping["limitation"] = INHERITED_PRIOR_NUMERIC_LIMITATION
    authorization = RecoveryAuthorization.from_mapping(
        mapping,
        authority_directory=authority,
        authority_artifact_root_sha256=gate.freeze_artifact_root_sha256,
        authority_manifest_sha256=gate.freeze_manifest_sha256,
    )
    real_inspect = inspect_orphan_source

    def inactive_inspect(**kwargs: Any) -> Any:
        return real_inspect(**kwargs, pid_probe=lambda _pid: False)

    monkeypatch.setattr(runner_module, "inspect_orphan_source", inactive_inspect)
    monkeypatch.setattr(
        vars(runner_module)["shutil"],
        "disk_usage",
        lambda _path: SimpleNamespace(total=2**61, used=2**60, free=2**60),
    )
    return _RunnerFixture(
        runs_root=base.runs_root,
        source=base.source,
        plan=base.plan,
        controls=controls,
        gate=gate,
        authorization=authorization,
    )


def _forbid_tracker_and_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("failed preflight created a tracker or invoked copy")

    monkeypatch.setattr(vars(runner_module)["RunTracker"], "start", forbidden)
    monkeypatch.setattr(runner_module, "copy_authorized_orphan_artifacts", forbidden)


def _forbid_training_and_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("orphan recovery invoked training or statistics recomputation")

    monkeypatch.setattr(primary_core_module, "execute_primary_matrix", forbidden)
    monkeypatch.setattr(primary_statistics_module, "aggregate_primary_statistics", forbidden)
    monkeypatch.setattr(primary_statistics_module, "_compute_statistics", forbidden)


def _stub_positive_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_build_primary_stage_attestation_verification",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner_module,
        "attest_primary_run_stage_eligibility",
        lambda *_args, **_kwargs: {"record_sha256": "e" * 64},
    )


def test_read_only_preflight_and_actual_runner_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    source_before = _tree_state(fixture.source)
    preflight = preflight_primary_orphan_recovery(
        gate_evidence=fixture.gate,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        runs_root=fixture.runs_root,
        run_id="tiny-recovery",
    )
    assert preflight["status"] == "passed"
    assert preflight["run_tracker_created"] is False
    assert preflight["copy_invoked"] is False
    assert _tree_state(fixture.source) == source_before
    assert not (fixture.runs_root / "tiny-recovery").exists()

    _forbid_training_and_recomputation(monkeypatch)
    _stub_positive_stage(monkeypatch)
    result = execute_primary_orphan_recovery(
        gate_evidence=fixture.gate,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        project_root=tmp_path,
        runs_root=fixture.runs_root,
        run_id="tiny-recovery",
    )
    run = Path(result["run_directory"])
    evidence = json.loads((run / RECOVERY_EVIDENCE_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["training_invoked"] is False
    assert result["matrix_executor_invoked"] is False
    assert result["fallback_invoked"] is False
    assert result["automatic_retry_allowed"] is False
    assert set(evidence) == runner_module._RECOVERY_EVIDENCE_KEYS
    assert evidence["reused_required_cell_count"] == 1
    assert evidence["skipped_optional_cell_count"] == 0
    assert evidence["retrained_cell_count"] == 0
    assert (run / "artifact_manifest.json").is_file()
    assert (run / ".immutable.json").is_file()
    assert _tree_state(fixture.source) == source_before
    source = inspect.getsource(runner_module)
    assert "execute_primary_matrix" not in source
    assert "aggregate_primary_statistics" not in source
    assert "_compute_statistics" not in source


def test_bad_authority_fails_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_tracker_and_copy(monkeypatch)
    mismatched = replace(
        fixture.authorization,
        authority_manifest_sha256="0" * 64,
    )
    with pytest.raises(PrimaryRecoveryRunnerError, match="authority binding"):
        preflight_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=mismatched,
            runs_root=fixture.runs_root,
        )


def test_stale_amendment_authorization_fails_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_tracker_and_copy(monkeypatch)

    def reject(_authorization: Any) -> None:
        raise PrimaryRecoveryRunnerError(
            "supplied recovery authorization differs from its immutable amendment"
        )

    monkeypatch.setattr(
        runner_module,
        "_require_authorization_matches_immutable_amendment",
        reject,
    )
    with pytest.raises(PrimaryRecoveryRunnerError, match="immutable amendment"):
        preflight_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            runs_root=fixture.runs_root,
        )


def test_bad_numeric_trust_contract_fails_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_tracker_and_copy(monkeypatch)
    mismatched = replace(
        fixture.authorization,
        trust_assumption="different trust assumption",
    )
    with pytest.raises(PrimaryRecoveryRunnerError, match="trust contract"):
        preflight_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=mismatched,
            runs_root=fixture.runs_root,
        )


def test_active_pid_fails_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_tracker_and_copy(monkeypatch)

    def active_inspect(**kwargs: Any) -> Any:
        return inspect_orphan_source(**kwargs, pid_probe=lambda _pid: True)

    monkeypatch.setattr(runner_module, "inspect_orphan_source", active_inspect)
    with pytest.raises(PrimaryRecoveryRunnerError, match="process is still active"):
        preflight_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            runs_root=fixture.runs_root,
        )


def test_low_disk_fails_before_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_tracker_and_copy(monkeypatch)
    monkeypatch.setattr(
        vars(runner_module)["shutil"],
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )
    with pytest.raises(PrimaryRecoveryRunnerError, match="insufficient space"):
        preflight_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            runs_root=fixture.runs_root,
        )


def test_streaming_disk_preflight_uses_largest_file_not_whole_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    first = preflight_primary_orphan_recovery(
        gate_evidence=fixture.gate,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        runs_root=fixture.runs_root,
    )
    disk = first["disk_preflight"]
    assert disk["capacity_basis"] == "streaming_largest_artifact_plus_margin_v1"
    assert disk["copy_policy"].endswith("_wof_lzx_v1")
    streaming_free = int(disk["required_free_bytes"])
    assert streaming_free < int(disk["copy_bytes"]) + int(disk["margin_bytes"])
    monkeypatch.setattr(
        vars(runner_module)["shutil"],
        "disk_usage",
        lambda _path: SimpleNamespace(total=2**61, used=1, free=streaming_free),
    )
    second = preflight_primary_orphan_recovery(
        gate_evidence=fixture.gate,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        runs_root=fixture.runs_root,
    )
    assert second["status"] == "passed"
    assert second["run_tracker_created"] is False
    assert second["copy_invoked"] is False


def test_copy_failure_creates_one_failed_run_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_training_and_recomputation(monkeypatch)
    calls = 0

    def fail_copy(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise OSError("injected copy failure")

    monkeypatch.setattr(runner_module, "copy_authorized_orphan_artifacts", fail_copy)
    run = fixture.runs_root / "copy-failure-recovery"
    with pytest.raises(PrimaryRecoveryRunnerError, match="injected copy failure"):
        execute_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            project_root=tmp_path,
            runs_root=fixture.runs_root,
            run_id=run.name,
        )
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert status["status"] == "failed"
    assert (run / "artifact_manifest.json").is_file()
    assert (run / ".immutable.json").is_file()
    assert (
        json.loads((run / "completion_evidence.json").read_text(encoding="utf-8"))[
            "study_outcome_eligible"
        ]
        is False
    )


def test_anchored_boundary_error_never_uses_pathname_fail_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    calls = 0
    run = fixture.runs_root / "ambiguous-boundary-recovery"

    def boundary_failure(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AnchoredPhysicalCopyBoundaryError(
            source_tree_current=True,
            destination_tree_current=False,
            rollback_complete=False,
            expected_destination_root=run,
            boundary_errors=("injected destination root swap",),
        )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("boundary failure attempted pathname fail-seal or withdrawal")

    monkeypatch.setattr(
        runner_module,
        "copy_authorized_orphan_artifacts",
        boundary_failure,
    )
    monkeypatch.setattr(runner_module, "_fail_seal", forbidden)
    monkeypatch.setattr(runner_module, "_withdraw_postseal_failure", forbidden)
    with pytest.raises(PrimaryRecoveryRunnerError, match="no pathname-based demotion"):
        execute_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            project_root=tmp_path,
            runs_root=fixture.runs_root,
            run_id=run.name,
        )
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert status["status"] == "running"
    assert not (run / "artifact_manifest.json").exists()
    assert not (run / ".immutable.json").exists()


def test_partial_completion_seal_never_attempts_failure_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_training_and_recomputation(monkeypatch)
    run = fixture.runs_root / "partial-seal-recovery"
    complete_calls = 0

    def fail_complete(_tracker: Any) -> None:
        nonlocal complete_calls
        complete_calls += 1
        raise OSError("injected partial completion seal failure")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("partial completion seal attempted a second failure seal")

    monkeypatch.setattr(vars(runner_module)["RunTracker"], "complete", fail_complete)
    monkeypatch.setattr(runner_module, "_fail_seal", forbidden)
    with pytest.raises(PrimaryRecoveryRunnerError, match="partial completion seal failure"):
        execute_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            project_root=tmp_path,
            runs_root=fixture.runs_root,
            run_id=run.name,
        )
    assert complete_calls == 1
    assert not (run / "artifact_manifest.json").exists()
    assert not (run / ".immutable.json").exists()


def test_positive_attestation_is_terminal_without_later_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_training_and_recomputation(monkeypatch)
    committed = False
    real_sha256_file = runner_module.sha256_file

    def guarded_sha256_file(path: Any, *args: Any, **kwargs: Any) -> str:
        if committed:
            raise AssertionError("positive attestation was followed by a file read")
        return real_sha256_file(path, *args, **kwargs)

    def terminal_attestation(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        nonlocal committed
        committed = True
        return {"record_sha256": "e" * 64}

    monkeypatch.setattr(runner_module, "sha256_file", guarded_sha256_file)
    monkeypatch.setattr(
        runner_module,
        "_build_primary_stage_attestation_verification",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner_module,
        "attest_primary_run_stage_eligibility",
        terminal_attestation,
    )
    result = execute_primary_orphan_recovery(
        gate_evidence=fixture.gate,
        plan=fixture.plan,
        controls=fixture.controls,
        authorization=fixture.authorization,
        project_root=tmp_path,
        runs_root=fixture.runs_root,
        run_id="terminal-attestation-recovery",
    )
    assert committed is True
    assert result["stage_attestation_record_sha256"] == "e" * 64


def test_preseal_copy_mutation_is_sealed_but_permanently_withdrawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_training_and_recomputation(monkeypatch)
    run = fixture.runs_root / "preseal-copy-mutation-recovery"
    real_complete = vars(runner_module)["RunTracker"].complete

    def mutate_copy_then_complete(tracker: Any) -> None:
        (tracker.run_directory / "matrix_plan.json").write_text(
            '{"mutated_before_seal":true}\n',
            encoding="utf-8",
        )
        real_complete(tracker)

    monkeypatch.setattr(
        vars(runner_module)["RunTracker"],
        "complete",
        mutate_copy_then_complete,
    )
    with pytest.raises(
        PrimaryRecoveryRunnerError,
        match=r"permanently withdrawn.*authorized source snapshot",
    ):
        execute_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            project_root=tmp_path,
            runs_root=fixture.runs_root,
            run_id=run.name,
        )
    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    dispositions = read_run_dispositions(fixture.runs_root / "run_dispositions.jsonl")
    assert any(
        record["run_id"] == run.name and record["event_type"] == "eligibility_withdrawn"
        for record in dispositions
    )


def test_postseal_failure_withdraws_completed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    _forbid_training_and_recomputation(monkeypatch)
    run = fixture.runs_root / "postseal-failure-recovery"
    withdrawal_calls = 0
    real_withdraw = vars(runner_module)["withdraw_run_eligibility"]

    def fail_stage(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected post-seal stage failure")

    def spy_withdraw(*args: Any, **kwargs: Any) -> Any:
        nonlocal withdrawal_calls
        withdrawal_calls += 1
        return real_withdraw(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "_build_primary_stage_attestation_verification",
        fail_stage,
    )
    monkeypatch.setattr(runner_module, "withdraw_run_eligibility", spy_withdraw)
    with pytest.raises(PrimaryRecoveryRunnerError, match="permanently withdrawn"):
        execute_primary_orphan_recovery(
            gate_evidence=fixture.gate,
            plan=fixture.plan,
            controls=fixture.controls,
            authorization=fixture.authorization,
            project_root=tmp_path,
            runs_root=fixture.runs_root,
            run_id=run.name,
        )
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    dispositions = read_run_dispositions(fixture.runs_root / "run_dispositions.jsonl")
    assert withdrawal_calls == 1
    assert status["status"] == "completed"
    assert any(
        record["run_id"] == run.name and record["event_type"] == "eligibility_withdrawn"
        for record in dispositions
    )
