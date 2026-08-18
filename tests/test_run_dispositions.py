"""Focused tests for irreversible scientific run-eligibility withdrawals."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import histo_audit.utils.run_tracking as run_tracking
from histo_audit.cli import app
from histo_audit.utils.run_tracking import (
    RUN_DISPOSITION_ANCHOR_FILENAME,
    RUN_DISPOSITION_REGISTRY_FILENAME,
    RunTracker,
    attest_run_stage_eligibility,
    read_run_dispositions,
    require_run_stage_eligible,
    sha256_file,
    sha256_path,
    verify_run_integrity,
    withdraw_run_eligibility,
)


def _completed_run(root: Path, experiment_name: str) -> RunTracker:
    tracker = RunTracker.start(
        experiment_name=experiment_name,
        config={"experiment_name": experiment_name, "seed": {}},
        project_root=root,
        runs_root=root / "artifacts" / "runs",
        environment={},
    )
    tracker.write_metrics({"artifact_scope": "test_fixture"})
    tracker.complete()
    return tracker


def _sealed_confirmatory_candidate(
    root: Path,
    *,
    write_completion: bool,
) -> RunTracker:
    tracker = RunTracker.start(
        experiment_name="pannuke_confirmatory_study",
        config={"experiment_name": "pannuke_confirmatory_study", "seed": {}},
        project_root=root,
        runs_root=root / "artifacts" / "runs",
        environment={},
    )
    if write_completion:
        tracker.write_json(
            "completion_evidence.json",
            {
                "schema_version": 1,
                "run_id": tracker.run_id,
                "completion_stage": "CONFIRMATORY_COMPLETE",
                "study_outcome_eligible": True,
                "post_seal_attestation_required": True,
            },
        )
    tracker.complete()
    return tracker


def test_sealed_confirmatory_run_without_completion_fails_closed(tmp_path: Path) -> None:
    tracker = _sealed_confirmatory_candidate(tmp_path, write_completion=False)

    with pytest.raises(ValueError, match="lacks completion evidence"):
        require_run_stage_eligible(tracker.run_directory)


def test_sealed_confirmatory_candidate_without_positive_attestation_fails_closed(
    tmp_path: Path,
) -> None:
    tracker = _sealed_confirmatory_candidate(tmp_path, write_completion=True)

    with pytest.raises(ValueError, match="lacks one durable positive"):
        require_run_stage_eligible(tracker.run_directory)


def test_arbitrary_full_checklist_cannot_mint_confirmatory_attestation(
    tmp_path: Path,
) -> None:
    tracker = _sealed_confirmatory_candidate(tmp_path, write_completion=True)
    integrity = verify_run_integrity(tracker.run_directory)
    assert integrity.valid and integrity.expected_root_sha256 is not None
    verification = {
        "schema_version": 1,
        "policy": "confirmatory_postseal_attestation_v1",
        "run_id": tracker.run_id,
        "completion_stage": "CONFIRMATORY_COMPLETE",
        "first_integrity_root_sha256": integrity.expected_root_sha256,
        "final_integrity_root_sha256": integrity.expected_root_sha256,
        "matrix_plan_sha256": "1" * 64,
        "cell_index_sha256": "2" * 64,
        "scientific_artifact_manifest_sha256": "3" * 64,
        "reconciliation_sha256": "4" * 64,
        "semantic_readback_status": "passed",
        "semantic_checked_artifact_count": 999,
    }

    with pytest.raises(ValueError, match="gate evidence is unavailable"):
        attest_run_stage_eligibility(
            tracker.run_directory,
            completion_stage="CONFIRMATORY_COMPLETE",
            verification=verification,
        )
    with pytest.raises(ValueError, match="lacks one durable positive"):
        require_run_stage_eligible(tracker.run_directory)


def test_stale_integrity_object_cannot_authorize_artifact_tamper(tmp_path: Path) -> None:
    tracker = _completed_run(tmp_path, "stale-integrity")
    stale = verify_run_integrity(tracker.run_directory)
    assert stale.valid
    metrics_path = tracker.run_directory / "metrics.json"
    metrics_path.write_bytes(metrics_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="integrity-valid"):
        require_run_stage_eligible(tracker.run_directory, integrity=stale)


def _canonical_record_sha256(record: dict[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rewrite_anchor_for_ledger(registry_path: Path) -> None:
    content = registry_path.read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8").split("\n") if line]
    anchor = {
        "schema_version": 1,
        "ledger_filename": RUN_DISPOSITION_REGISTRY_FILENAME,
        "chain_algorithm": "sha256(canonical-json-record-with-previous-head)",
        "record_count": len(records),
        "head_record_sha256": records[-1]["record_sha256"] if records else None,
        "ledger_sha256": hashlib.sha256(content).hexdigest(),
    }
    registry_path.with_name(RUN_DISPOSITION_ANCHOR_FILENAME).write_text(
        json.dumps(anchor, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_withdrawals_are_hash_chained_without_modifying_sealed_runs(tmp_path: Path) -> None:
    first_run = _completed_run(tmp_path, "first-completed")
    second_run = _completed_run(tmp_path, "second-completed")
    runs_root = first_run.run_directory.parent
    registry_path = runs_root / RUN_DISPOSITION_REGISTRY_FILENAME
    standard_registry_before = (runs_root / "registry.csv").read_bytes()
    integrity_registry_before = (runs_root / "integrity_registry.jsonl").read_bytes()
    first_tree_before = sha256_path(first_run.run_directory)
    second_tree_before = sha256_path(second_run.run_directory)

    first = withdraw_run_eligibility(
        first_run.run_directory,
        reason_code="post_seal_policy_violation",
        reason="A post-seal audit found that this run cannot support a scientific stage.",
    )
    second = withdraw_run_eligibility(
        second_run.run_directory,
        reason_code="superseded_protocol_fixture",
        reason="A corrected protocol requires this completed fixture to remain historical only.",
    )

    records = read_run_dispositions(registry_path)
    assert records == (first, second)
    assert first["sequence"] == 1
    assert first["previous_record_sha256"] is None
    assert second["sequence"] == 2
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert all(record["event_type"] == "eligibility_withdrawn" for record in records)
    assert all(record["terminal_status"] == "completed" for record in records)
    assert all(record["scientific_stage_eligible"] is False for record in records)
    assert first["artifact_manifest_sha256"] == sha256_file(
        first_run.run_directory / "artifact_manifest.json"
    )
    assert (
        first["artifact_root_sha256"]
        == verify_run_integrity(first_run.run_directory).expected_root_sha256
    )
    assert sha256_path(first_run.run_directory) == first_tree_before
    assert sha256_path(second_run.run_directory) == second_tree_before
    assert (runs_root / "registry.csv").read_bytes() == standard_registry_before
    assert (runs_root / "integrity_registry.jsonl").read_bytes() == integrity_registry_before
    assert verify_run_integrity(first_run.run_directory).valid
    assert verify_run_integrity(second_run.run_directory).valid


def test_withdrawal_is_irreversible_and_requires_completed_integrity(tmp_path: Path) -> None:
    completed = _completed_run(tmp_path, "completed")
    withdraw_run_eligibility(
        completed.run_directory,
        reason_code="withdrawn_once",
        reason="This completed fixture is permanently ineligible for scientific stages.",
    )
    with pytest.raises(ValueError, match="already withdrawn"):
        withdraw_run_eligibility(
            completed.run_directory,
            reason_code="attempted_reinstatement",
            reason="A second disposition cannot reverse the first eligibility withdrawal.",
        )
    with pytest.raises(ValueError, match="permanently withdrawn"):
        require_run_stage_eligible(completed.run_directory)

    failed = RunTracker.start(
        experiment_name="failed",
        config={"experiment_name": "failed"},
        project_root=tmp_path,
        runs_root=tmp_path / "failed-runs",
        environment={},
    )
    failed.fail(RuntimeError("expected fixture failure"))
    with pytest.raises(ValueError, match="terminal completed"):
        withdraw_run_eligibility(
            failed.run_directory,
            reason_code="failed_fixture",
            reason="Failed executions retain their truthful failed terminal status.",
        )


def test_disposition_reader_detects_hash_chain_tampering(tmp_path: Path) -> None:
    first_run = _completed_run(tmp_path, "chain-first")
    second_run = _completed_run(tmp_path, "chain-second")
    registry_path = first_run.run_directory.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    withdraw_run_eligibility(
        first_run.run_directory,
        reason_code="chain_first",
        reason="The first historical fixture is withdrawn for a recorded audit reason.",
    )
    withdraw_run_eligibility(
        second_run.run_directory,
        reason_code="chain_second",
        reason="The second historical fixture is withdrawn for a recorded audit reason.",
    )
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["reason"] = "Tampered after append."
    first["record_sha256"] = _canonical_record_sha256(first)
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash chain"):
        read_run_dispositions(registry_path)
    with pytest.raises(ValueError, match="hash chain"):
        require_run_stage_eligible(second_run.run_directory)


def test_disposition_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    tracker = _completed_run(tmp_path, "binding")
    registry_path = tracker.run_directory.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    withdraw_run_eligibility(
        tracker.run_directory,
        reason_code="binding_fixture",
        reason="This fixture verifies exact binding to the immutable artifact identity.",
    )
    record = json.loads(registry_path.read_text(encoding="utf-8"))
    record["artifact_root_sha256"] = "0" * 64
    record["record_sha256"] = _canonical_record_sha256(record)
    registry_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_anchor_for_ledger(registry_path)

    assert len(read_run_dispositions(registry_path)) == 1
    with pytest.raises(ValueError, match="artifact_root_sha256 does not match"):
        require_run_stage_eligible(tracker.run_directory)


def test_anchor_detects_tail_truncation_and_complete_ledger_deletion(tmp_path: Path) -> None:
    first_run = _completed_run(tmp_path, "anchored-first")
    second_run = _completed_run(tmp_path, "anchored-second")
    registry_path = first_run.run_directory.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    anchor_path = registry_path.with_name(RUN_DISPOSITION_ANCHOR_FILENAME)
    withdraw_run_eligibility(
        first_run.run_directory,
        reason_code="anchored_first",
        reason="The first event establishes an anchored disposition history.",
    )
    withdraw_run_eligibility(
        second_run.run_directory,
        reason_code="anchored_second",
        reason="The second event must not disappear through valid-prefix truncation.",
    )
    original_registry = registry_path.read_bytes()
    original_anchor = anchor_path.read_bytes()
    first_line = original_registry.split(b"\n", maxsplit=1)[0] + b"\n"

    registry_path.write_bytes(first_line)
    with pytest.raises(ValueError, match="anchor does not match"):
        require_run_stage_eligible(second_run.run_directory)

    registry_path.write_bytes(original_registry)
    registry_path.unlink()
    with pytest.raises(ValueError, match="anchor does not match"):
        require_run_stage_eligible(second_run.run_directory)

    anchor_path.unlink()
    with pytest.raises(ValueError, match="anchor is missing"):
        require_run_stage_eligible(second_run.run_directory)
    with pytest.raises(ValueError, match="automatic reinitialization"):
        RunTracker.start(
            experiment_name="must-not-reinitialize",
            config={"experiment_name": "must-not-reinitialize"},
            project_root=tmp_path,
            runs_root=second_run.run_directory.parent,
            environment={},
        )

    registry_path.write_bytes(original_registry)
    anchor_path.write_bytes(original_anchor)
    with pytest.raises(ValueError, match="permanently withdrawn"):
        require_run_stage_eligible(second_run.run_directory)


def test_copied_run_cannot_escape_original_integrity_registry_path_binding(tmp_path: Path) -> None:
    tracker = _completed_run(tmp_path / "original", "copy-bound")
    original_runs = tracker.run_directory.parent
    copied_runs = tmp_path / "copied" / "artifacts" / "runs"
    copied_runs.mkdir(parents=True)
    copied_run = copied_runs / tracker.run_directory.name
    shutil.copytree(tracker.run_directory, copied_run)
    shutil.copy2(
        original_runs / "integrity_registry.jsonl",
        copied_runs / "integrity_registry.jsonl",
    )
    shutil.copy2(
        original_runs / RUN_DISPOSITION_ANCHOR_FILENAME,
        copied_runs / RUN_DISPOSITION_ANCHOR_FILENAME,
    )
    marker_path = copied_run / ".immutable.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["run_path"] = str(copied_run.resolve())
    marker["integrity_registry"] = str((copied_runs / "integrity_registry.jsonl").resolve())
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    copied_integrity = verify_run_integrity(copied_run)
    assert not copied_integrity.valid
    assert not copied_integrity.registry_record_present
    assert "matching append-only integrity registry record is absent" in copied_integrity.errors
    with pytest.raises(ValueError, match="integrity-valid"):
        require_run_stage_eligible(copied_run)


def test_unicode_line_separator_reason_round_trips_without_poisoning_ledger(
    tmp_path: Path,
) -> None:
    tracker = _completed_run(tmp_path, "unicode-reason")
    reason = "A valid Unicode reason contains a line separator:\u2028and remains one JSONL record."
    record = withdraw_run_eligibility(
        tracker.run_directory,
        reason_code="unicode_separator",
        reason=reason,
    )
    registry_path = tracker.run_directory.parent / RUN_DISPOSITION_REGISTRY_FILENAME

    records = read_run_dispositions(registry_path)
    assert records == (record,)
    assert records[0]["reason"] == reason
    assert b"\\u2028" in registry_path.read_bytes()


def test_old_lock_is_never_removed_by_age_only_recovery(tmp_path: Path) -> None:
    registry_path = tmp_path / RUN_DISPOSITION_REGISTRY_FILENAME
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    lock_path.write_text("999999 2000-01-01T00:00:00.000Z\n", encoding="utf-8")
    os.utime(lock_path, (1, 1))

    with (
        pytest.raises(TimeoutError, match="timed out"),
        run_tracking._registry_lock(registry_path, timeout_seconds=0.05),
    ):
        raise AssertionError("an existing lock must not be stolen")
    assert lock_path.is_file()


def test_write_started_before_finalize_is_serialized_into_valid_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = RunTracker.start(
        experiment_name="write-seal-race",
        config={"experiment_name": "write-seal-race"},
        project_root=tmp_path,
        runs_root=tmp_path / "artifacts" / "runs",
        environment={},
    )
    entered_write = threading.Event()
    release_write = threading.Event()
    original_atomic_write_text = run_tracking.atomic_write_text
    errors: list[BaseException] = []

    def _delayed_write(path: str | Path, content: str) -> Path:
        if Path(path).name == "late.txt":
            entered_write.set()
            assert release_write.wait(timeout=5)
        return original_atomic_write_text(path, content)

    monkeypatch.setattr(run_tracking, "atomic_write_text", _delayed_write)

    def _capture(action: object) -> None:
        try:
            assert callable(action)
            action()
        except BaseException as exc:  # pragma: no cover - diagnostic preservation
            errors.append(exc)

    writer = threading.Thread(target=_capture, args=(lambda: tracker.write_text("late.txt", "x"),))
    writer.start()
    assert entered_write.wait(timeout=5)
    finalizer = threading.Thread(target=_capture, args=(tracker.complete,))
    finalizer.start()
    time.sleep(0.05)
    assert finalizer.is_alive()
    release_write.set()
    writer.join(timeout=5)
    finalizer.join(timeout=5)

    assert not writer.is_alive()
    assert not finalizer.is_alive()
    assert errors == []
    assert verify_run_integrity(tracker.run_directory).valid
    manifest = json.loads(
        (tracker.run_directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert "late.txt" in {artifact["path"] for artifact in manifest["artifacts"]}


def test_withdraw_run_eligibility_cli_appends_one_auditable_event(tmp_path: Path) -> None:
    tracker = _completed_run(tmp_path, "cli-withdrawal")
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "withdraw-run-eligibility",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            str(tracker.run_directory),
            "--reason-code",
            "cli_post_seal_audit",
            "--reason",
            "A post-seal audit made this completed run ineligible for scientific stages.",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event_type"] == "eligibility_withdrawn"
    assert payload["run_id"] == tracker.run_id
    assert payload["run_terminal_status"] == "completed"
    assert payload["scientific_stage_eligible"] is False
    assert len(payload["record_sha256"]) == 64
    assert Path(payload["disposition_registry"]).is_file()

    duplicate = CliRunner().invoke(
        app,
        [
            "experiment",
            "withdraw-run-eligibility",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            str(tracker.run_directory),
            "--reason-code",
            "duplicate_attempt",
            "--reason",
            "A duplicate event must not append or imply reinstatement of eligibility.",
        ],
    )
    assert duplicate.exit_code == 1
    assert "already withdrawn" in duplicate.output
    registry_path = tracker.run_directory.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    assert len(read_run_dispositions(registry_path)) == 1
