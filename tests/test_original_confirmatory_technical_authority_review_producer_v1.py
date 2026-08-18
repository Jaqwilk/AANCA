"""Focused tests for the fresh-process outcome-blind T0 review producer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import pytest

from histo_audit.workflows import (
    original_confirmatory_technical_authority_publication_v1 as publication,
)
from histo_audit.workflows import (
    original_confirmatory_technical_authority_review_producer_v1 as producer,
)
from histo_audit.workflows import (
    original_confirmatory_technical_authority_v1 as schema,
)

LIVE_PROJECT_ROOT = Path(
    os.environ.get("AANCA_PROJECT_ROOT", r"C:\Users\NATAN\Documents\AANCA")
).resolve()
WIP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _direct_call_parent_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model controller custody when focused tests call the producer in-process."""

    controller = publication._capture_process_identity_v1(
        os.getpid(),
        Path(publication.__file__).resolve(),
    )
    monkeypatch.setattr(
        producer,
        "_capture_parent_controller_process_v1",
        lambda: dict(controller),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _process_created_at(process_id: int) -> str:
    return (
        datetime.fromtimestamp(psutil.Process(process_id).create_time(), tz=UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _intent_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path.resolve()
    request_root = root / "artifacts" / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
    request_root.mkdir(parents=True)

    source_inventory = {
        "schema_version": 3,
        "policy": "synthetic_runtracker_capture_source_tree",
        "root_sha256": "1" * 64,
        "artifacts": [
            {
                "path": "src/histo_audit/synthetic.py",
                "size_bytes": 3,
                "sha256": "2" * 64,
            }
        ],
    }
    source_bytes = schema.canonical_json_line_bytes(source_inventory)
    source_path = (root / "source_inventory.json").resolve()
    source_path.write_bytes(source_bytes)

    builder_implementation = (root / "synthetic_intent_builder.py").resolve()
    builder_implementation.write_bytes(b"# distinct synthetic intent builder\n")
    builder_process = publication.capture_current_process_identity_v1(builder_implementation)
    capsule_sha256 = "3" * 64
    parent = {
        "schema_version": 1,
        "authority_kind": "preregistration_amendment",
        "authority_directory": str(
            (
                root / "artifacts" / "preregistration_amendments" / "20260727T133947.089370Z"
            ).resolve()
        ),
        "chain_depth": schema.PARENT_CHAIN_DEPTH,
        "artifact_root_sha256": schema.PARENT_ARTIFACT_ROOT_SHA256,
        "sha256_manifest_sha256": schema.PARENT_MANIFEST_SHA256,
        "execution_source_root_sha256": schema.PARENT_SOURCE_ROOT_SHA256,
        "execution_source_manifest_sha256": schema.PARENT_SOURCE_MANIFEST_SHA256,
    }
    frozen_science = {
        "schema_version": 1,
        "preregistration_path": str((LIVE_PROJECT_ROOT / "PRE_REGISTRATION.md").resolve()),
        "preregistration_sha256": schema.PREREGISTRATION_SHA256,
        "primary_config_path": str(
            (LIVE_PROJECT_ROOT / "configs" / "primary_frozen.yaml").resolve()
        ),
        "primary_config_sha256": schema.PRIMARY_CONFIG_SHA256,
        "primary_config_semantic_sha256": schema.PRIMARY_CONFIG_SEMANTIC_SHA256,
        "confirmatory_config_path": str(
            (LIVE_PROJECT_ROOT / "configs" / "confirmatory_frozen.yaml").resolve()
        ),
        "confirmatory_config_sha256": schema.CONFIRMATORY_CONFIG_SHA256,
        "confirmatory_config_semantic_sha256": (schema.CONFIRMATORY_CONFIG_SEMANTIC_SHA256),
        "scientific_definition_changed": False,
    }
    historical_primary = {
        "schema_version": 1,
        "run_directory": str(
            (root / "artifacts" / "runs" / schema.HISTORICAL_PRIMARY_RUN_ID).resolve()
        ),
        "run_id": schema.HISTORICAL_PRIMARY_RUN_ID,
        "terminal_status": "completed",
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "artifact_root_sha256": schema.HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256,
        "artifact_manifest_sha256": schema.HISTORICAL_PRIMARY_MANIFEST_SHA256,
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "skipped_optional_cell_count": 37,
        "failed_required_cell_count": 0,
        "retrained_cell_count": 0,
        "verification_scope": ("integrity_and_control_metadata_only_no_scientific_outcome_values"),
        "outcome_values_read": False,
    }
    execution_source = {
        "schema_version": 1,
        "policy": "runtracker_capture_source_tree_exact_object_v1",
        "manifest_path": str(source_path),
        "manifest_sha256": _sha256_bytes(source_bytes),
        "root_sha256": source_inventory["root_sha256"],
        "record_count": len(source_inventory["artifacts"]),
    }
    execution_capsule = {
        "schema_version": 1,
        "policy": "content_addressed_original_confirmatory_execution_capsule_v1",
        "path": str((root / "capsules" / capsule_sha256 / "original_confirmatory.pyz").resolve()),
        "size_bytes": 10,
        "sha256": capsule_sha256,
        "internal_manifest_sha256": "4" * 64,
        "source_records_root_sha256": "5" * 64,
        "publication_receipt_path": str((root / "capsule_publication.json").resolve()),
        "publication_receipt_sha256": "6" * 64,
        "independent_readback_path": str((root / "capsule_readback.json").resolve()),
        "independent_readback_sha256": "7" * 64,
        "content_addressed_create_new_verified": True,
        "scientific_execution_performed": False,
    }
    capacity_v2 = {
        "schema_version": schema.CAPACITY_SCHEMA_VERSION,
        "policy": schema.CAPACITY_POLICY_NAME,
        "policy_sha256": schema.CAPACITY_POLICY_SHA256,
        "receipt_path": str((root / "capacity_v2.json").resolve()),
        "receipt_sha256": "8" * 64,
        "required_free_bytes": schema.CAPACITY_REQUIRED_FREE_BYTES,
        "observed_free_bytes": schema.CAPACITY_REQUIRED_FREE_BYTES + 1,
        "passed": True,
        "capsule_sha256": capsule_sha256,
        "execution_source_root_sha256": source_inventory["root_sha256"],
        "outcome_values_read": False,
        "scientific_execution_performed": False,
    }
    outcome_scope = {
        "schema_version": 1,
        "primary_outcomes_inspected": True,
        "primary_outcomes_inspected_at_utc": (schema.PRIMARY_OUTCOME_INSPECTION_AT_UTC),
        "primary_analysis_disposition": "amended_or_exploratory",
        "confirmatory_outcomes_inspected": False,
        "confirmatory_outcome_values_read": False,
        "confirmatory_registration_status": ("original_frozen_confirmatory_unchanged"),
        "selection_performed": False,
        "tuning_performed": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    intent = schema.build_original_confirmatory_technical_authority_intent_v1(
        created_at_utc=_utc_now(),
        builder_process=builder_process,
        parent=parent,
        frozen_science=frozen_science,
        historical_primary=historical_primary,
        execution_source=execution_source,
        execution_capsule=execution_capsule,
        capacity_v2=capacity_v2,
        outcome_scope=outcome_scope,
    )
    intent_path = request_root / publication.INTENT_REQUEST_FILENAME
    review_path = request_root / publication.REVIEW_REQUEST_FILENAME
    intent_path.write_bytes(schema.canonical_json_line_bytes(intent))
    intent_path.chmod(stat.S_IREAD)
    reviewer_spec = publication.importlib.util.find_spec(publication.REVIEWER_MODULE_NAME)
    assert reviewer_spec is not None and reviewer_spec.origin is not None
    attempt = publication._build_original_confirmatory_technical_review_attempt_claim_at_v1(
        intent=intent,
        project_root=root,
        controller_process=publication.capture_current_process_identity_v1(
            Path(publication.__file__).resolve()
        ),
        reviewer_implementation_path=Path(reviewer_spec.origin).resolve(),
        attempt_created_at_utc=_utc_now(),
    )
    attempt_path = request_root / publication.REVIEW_ATTEMPT_FILENAME
    attempt_path.write_bytes(schema.canonical_json_line_bytes(attempt))
    attempt_path.chmod(stat.S_IREAD)
    return {
        "root": root,
        "request_root": request_root,
        "intent": intent,
        "intent_path": intent_path,
        "review_path": review_path,
        "attempt_path": attempt_path,
        "source_inventory": source_inventory,
        "source_path": source_path,
        "builder_process": builder_process,
    }


def _synthetic_reviewer_process(fixture: dict[str, Any]) -> dict[str, Any]:
    identity = publication.capture_current_process_identity_v1(Path(producer.__file__).resolve())
    identity["process_id"] = fixture["builder_process"]["process_id"] + 1_000_000
    return identity


def test_producer_builds_exact_review_and_create_new_read_only_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    captured: list[dict[str, Any]] = []
    reviewer = _synthetic_reviewer_process(fixture)
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )

    def verify_live(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return dict(kwargs["intent"])

    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        verify_live,
    )
    result = producer.produce_original_confirmatory_technical_authority_review_v1(
        intent_json=fixture["intent_path"],
        output=fixture["review_path"],
        project_root=fixture["root"],
    )

    payload = fixture["review_path"].read_bytes()
    review = json.loads(payload)
    assert payload == schema.canonical_json_line_bytes(review)
    assert result["output_sha256"] == _sha256_bytes(payload)
    assert result["review_root_sha256"] == review["review_root_sha256"]
    assert review["reviewer_process"] == reviewer
    assert review["reviewer_process"]["implementation_path"] == str(
        Path(producer.__file__).resolve()
    )
    assert (
        review["reviewer_process"]["implementation_sha256"]
        != fixture["builder_process"]["implementation_sha256"]
    )
    assert review["outcome_values_read"] is False
    assert review["scientific_execution_performed"] is False
    assert review["publication_performed"] is False
    assert review["selection_performed"] is False
    assert review["tuning_performed"] is False
    assert len(captured) == 1
    assert captured[0]["reviewer_process"] == reviewer
    assert captured[0]["source_inventory"] == fixture["source_inventory"]
    assert not fixture["review_path"].stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_existing_review_is_never_overwritten_adopted_or_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    fixture["review_path"].write_bytes(b"retain exact foreign bytes")
    calls = 0

    def forbidden_live(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("existing receipt must stop before live verification")

    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        forbidden_live,
    )
    with pytest.raises(
        producer.OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error,
        match="overwrite, adoption, cleanup, and retry are forbidden",
    ):
        producer.produce_original_confirmatory_technical_authority_review_v1(
            intent_json=fixture["intent_path"],
            output=fixture["review_path"],
            project_root=fixture["root"],
        )

    assert calls == 0
    assert fixture["review_path"].read_bytes() == b"retain exact foreign bytes"


def test_unpaired_or_outside_request_output_is_rejected_before_read(
    tmp_path: Path,
) -> None:
    fixture = _intent_fixture(tmp_path)
    for output in (
        fixture["request_root"] / "other.review.json",
        fixture["root"] / "outside.review.json",
    ):
        with pytest.raises(Exception, match="review output"):
            producer.produce_original_confirmatory_technical_authority_review_v1(
                intent_json=fixture["intent_path"],
                output=output,
                project_root=fixture["root"],
            )
        assert not output.exists()


def test_bound_source_inventory_is_read_from_intent_not_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    reviewer = _synthetic_reviewer_process(fixture)
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )

    def capture(**kwargs: Any) -> dict[str, Any]:
        seen.append(dict(kwargs["source_inventory"]))
        return dict(kwargs["intent"])

    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        capture,
    )
    producer.produce_original_confirmatory_technical_authority_review_v1(
        intent_json=fixture["intent_path"],
        output=fixture["review_path"],
        project_root=fixture["root"],
    )

    assert seen == [fixture["source_inventory"]]
    assert Path(fixture["intent"]["execution_source"]["manifest_path"]) == fixture["source_path"]


def test_internal_timestamps_bracket_verification_before_review_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    reviewer = _synthetic_reviewer_process(fixture)
    timeline: list[str] = []
    attempt = json.loads(fixture["attempt_path"].read_bytes())
    attempt_created_at = datetime.fromisoformat(
        attempt["attempt_created_at_utc"].replace("Z", "+00:00")
    )
    review_started_at = attempt_created_at + timedelta(seconds=1, microseconds=1)
    review_completed_at = review_started_at + timedelta(seconds=1, microseconds=1)
    timestamps = iter((review_started_at, review_completed_at))
    expected_started_at = review_started_at.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    expected_completed_at = review_completed_at.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )

    def clock() -> datetime:
        value = next(timestamps)
        timeline.append("clock:" + value.isoformat(timespec="microseconds").replace("+00:00", "Z"))
        return value

    real_verify_attempt = publication.verify_original_confirmatory_technical_review_attempt_claim_v1

    def verify_attempt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        timeline.append("verify:permanent-attempt")
        return real_verify_attempt(*args, **kwargs)

    def verify(**kwargs: Any) -> dict[str, Any]:
        timeline.append("verify:upstream-and-reviewer")
        assert kwargs["reviewer_process"] == reviewer
        return dict(kwargs["intent"])

    real_publish = publication.publish_canonical_control_leaf_create_new_v1

    def publish(destination: Path, payload: bytes) -> str:
        timeline.append("publish")
        return real_publish(destination, payload)

    monkeypatch.setattr(producer, "_UTC_CLOCK_FOR_TESTS_ONLY", clock)
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )
    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_review_attempt_claim_v1",
        verify_attempt,
    )
    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        verify,
    )
    monkeypatch.setattr(
        publication,
        "publish_canonical_control_leaf_create_new_v1",
        publish,
    )
    producer.produce_original_confirmatory_technical_authority_review_v1(
        intent_json=fixture["intent_path"],
        output=fixture["review_path"],
        project_root=fixture["root"],
    )

    review = json.loads(fixture["review_path"].read_bytes())
    assert review["review_started_at_utc"] == expected_started_at
    assert review["review_completed_at_utc"] == expected_completed_at
    assert timeline == [
        "verify:permanent-attempt",
        f"clock:{expected_started_at}",
        "verify:upstream-and-reviewer",
        f"clock:{expected_completed_at}",
        "publish",
    ]


def test_same_builder_reviewer_process_or_source_fails_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(fixture["builder_process"]),
    )
    with pytest.raises(
        producer.OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error,
        match="not process/source independent",
    ):
        producer.produce_original_confirmatory_technical_authority_review_v1(
            intent_json=fixture["intent_path"],
            output=fixture["review_path"],
            project_root=fixture["root"],
        )
    assert not fixture["review_path"].exists()


def test_reviewer_rejects_parent_that_does_not_match_attempt_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    reviewer = _synthetic_reviewer_process(fixture)
    attempt = json.loads(fixture["attempt_path"].read_bytes())
    unexpected_parent = dict(attempt["controller_process"])
    unexpected_parent["process_id"] += 1
    live_calls = 0

    def forbidden_live(**_kwargs: Any) -> dict[str, Any]:
        nonlocal live_calls
        live_calls += 1
        raise AssertionError("parent mismatch must fail before the full live pass")

    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )
    monkeypatch.setattr(
        producer,
        "_capture_parent_controller_process_v1",
        lambda: unexpected_parent,
    )
    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        forbidden_live,
    )

    with pytest.raises(
        producer.OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error,
        match="parent custody",
    ):
        producer.produce_original_confirmatory_technical_authority_review_v1(
            intent_json=fixture["intent_path"],
            output=fixture["review_path"],
            project_root=fixture["root"],
        )

    assert live_calls == 0
    assert not fixture["review_path"].exists()


def test_live_verification_failure_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    reviewer = _synthetic_reviewer_process(fixture)
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )

    def fail_closed(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic live mismatch")

    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        fail_closed,
    )
    with pytest.raises(RuntimeError, match="synthetic live mismatch"):
        producer.produce_original_confirmatory_technical_authority_review_v1(
            intent_json=fixture["intent_path"],
            output=fixture["review_path"],
            project_root=fixture["root"],
        )
    assert not fixture["review_path"].exists()


def test_create_new_failure_is_called_once_without_cleanup_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _intent_fixture(tmp_path)
    reviewer = _synthetic_reviewer_process(fixture)
    monkeypatch.setattr(
        publication,
        "capture_current_process_identity_v1",
        lambda _implementation: dict(reviewer),
    )
    monkeypatch.setattr(
        publication,
        "verify_original_confirmatory_technical_intent_live_bindings_v1",
        lambda **kwargs: dict(kwargs["intent"]),
    )
    calls = 0

    def fail_once(_destination: Path, _payload: bytes) -> str:
        nonlocal calls
        calls += 1
        raise OSError("synthetic CREATE_NEW failure")

    monkeypatch.setattr(
        publication,
        "publish_canonical_control_leaf_create_new_v1",
        fail_once,
    )
    with pytest.raises(OSError, match="synthetic CREATE_NEW failure"):
        producer.produce_original_confirmatory_technical_authority_review_v1(
            intent_json=fixture["intent_path"],
            output=fixture["review_path"],
            project_root=fixture["root"],
        )
    assert calls == 1
    assert not fixture["review_path"].exists()


def test_real_fresh_child_process_records_actual_pid_creation_executable_and_module(
    tmp_path: Path,
) -> None:
    fixture = _intent_fixture(tmp_path)
    driver = (tmp_path / "fresh_reviewer_driver.py").resolve()
    driver.write_text(
        "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {str((WIP_ROOT / 'src').resolve())!r})",
                f"sys.path.insert(1, {str((LIVE_PROJECT_ROOT / 'src').resolve())!r})",
                "from histo_audit.workflows import "
                "original_confirmatory_technical_authority_review_producer_v1 as review",
                "review.publication."
                "verify_original_confirmatory_technical_intent_live_bindings_v1 = "
                "lambda **kwargs: dict(kwargs['intent'])",
                "raise SystemExit(review.main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    exact_interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    command = [
        str(exact_interpreter),
        "-I",
        "-B",
        str(driver),
        "--intent-json",
        str(fixture["intent_path"]),
        "--output",
        str(fixture["review_path"]),
        "--project-root",
        str(fixture["root"]),
    ]
    child_environment = dict(os.environ)
    child_environment["__PYVENV_LAUNCHER__"] = sys.executable
    child = subprocess.Popen(
        command,
        cwd=fixture["root"],
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_created_at = _process_created_at(child.pid)
    stdout, stderr = child.communicate(timeout=30)

    assert child.returncode == 0, stderr.decode("utf-8", errors="replace")
    assert stderr == b""
    summary = json.loads(stdout)
    assert stdout == schema.canonical_json_line_bytes(summary)
    review = json.loads(fixture["review_path"].read_bytes())
    reviewer = review["reviewer_process"]
    assert reviewer["process_id"] == child.pid
    assert reviewer["process_created_at_utc"] == child_created_at
    assert Path(reviewer["executable_path"]).resolve() == exact_interpreter
    assert reviewer["executable_sha256"] == _sha256_bytes(
        Path(reviewer["executable_path"]).read_bytes()
    )
    assert Path(reviewer["implementation_path"]).resolve() == Path(producer.__file__).resolve()
    assert reviewer["implementation_sha256"] == _sha256_bytes(Path(producer.__file__).read_bytes())
    assert reviewer["process_id"] != fixture["builder_process"]["process_id"]
    assert reviewer["implementation_sha256"] != fixture["builder_process"]["implementation_sha256"]
    assert summary["outcome_values_read"] is False
    assert summary["scientific_execution_performed"] is False
    assert summary["publication_performed"] is False
    assert summary["automatic_retry_allowed"] is False


def test_cli_failure_is_nonzero_and_does_not_create_receipt(
    tmp_path: Path,
) -> None:
    fixture = _intent_fixture(tmp_path)
    missing = fixture["request_root"] / "missing.intent.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str((WIP_ROOT / 'src').resolve())!r});"
                f"sys.path.insert(1,{str((LIVE_PROJECT_ROOT / 'src').resolve())!r});"
                "from histo_audit.workflows."
                "original_confirmatory_technical_authority_review_producer_v1 "
                "import main;"
                "raise SystemExit(main())"
            ),
            "--intent-json",
            str(missing),
            "--output",
            str(fixture["review_path"]),
            "--project-root",
            str(fixture["root"]),
        ],
        cwd=fixture["root"],
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert result.stdout == b""
    assert b"ERROR:" in result.stderr
    assert not fixture["review_path"].exists()
