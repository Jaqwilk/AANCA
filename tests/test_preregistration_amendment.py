"""Fail-closed tests for immutable preregistration successor authorities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from histo_audit.utils.run_tracking import (
    capture_source_tree,
    sha256_file,
    sha256_path,
)
from histo_audit.workflows.preregistration import verify_preregistration_freeze
from histo_audit.workflows.preregistration_amendment import (
    create_preregistration_amendment,
    verify_preregistration_amendment,
)

BASE_TIME = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)


def _require_local_prior_publication_failure_evidence() -> Path:
    """Return Authority C or skip when its ignored terminal history is unavailable."""

    project_root = Path(__file__).resolve().parents[1]
    control_root = project_root / "artifacts" / "resource_control"
    run_state_root = project_root / "artifacts" / "runs"
    frozen_input_root = control_root / "authority_d_inputs_20260727Tfinal_source_v2"
    required_paths = (
        control_root / "resource_authority_d_publication_attempt.json",
        control_root / "resource_authority_d_publication_failure.json",
        control_root / "authority_d_v2_publish_20260727T212335.140Z.stdout.log",
        control_root / "authority_d_v2_publish_20260727T212335.140Z.stderr.log",
        control_root / "prepare_resource_authority_d_once.py",
        frozen_input_root / "authority_d_workspace_plan.json",
        frozen_input_root / "authority_d_source_allowlist.json",
        frozen_input_root / "authority_d_frozen_source_receipt.json",
        frozen_input_root / "authority_d_cnn_correction_receipt.json",
        run_state_root / "integrity_registry.jsonl",
        run_state_root / "registry.csv",
        run_state_root / "run_dispositions.anchor.json",
        run_state_root / "run_dispositions.jsonl",
        run_state_root / "run_stage_attestations.anchor.json",
        run_state_root / "run_stage_attestations.jsonl",
    )
    missing_paths = [path for path in required_paths if not os.path.lexists(path)]
    if missing_paths:
        missing = ", ".join(path.relative_to(project_root).as_posix() for path in missing_paths)
        pytest.skip(
            f"requires ignored local terminal Authority-D publication evidence; missing: {missing}"
        )
    return project_root / "artifacts" / "preregistration_amendments" / "20260727T170413.080954Z"


def _local_replacement_terminal_contract() -> tuple[Any, Any, dict[str, Any], Path]:
    """Build or read Q without publishing it, then return amendment-side inputs."""

    from histo_audit.workflows import preregistration_amendment as amendment
    from histo_audit.workflows import (
        resource_authority_d_replacement_v2_controller as controller,
    )

    project_root = Path(__file__).resolve().parents[1]
    authority_c = _require_local_prior_publication_failure_evidence().resolve()
    namespace = controller.Namespace.for_project(project_root)
    required_paths = (
        project_root
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_controller.py",
        project_root
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_v2_controller.py",
        namespace.control_root
        / "resource_authority_d_replacement_publication_authorization_v1.json",
        namespace.control_root / "resource_authority_d_replacement_v1_publication_attempt.json",
        namespace.control_root / "resource_authority_d_replacement_v1_publication_failure.json",
        namespace.control_root / "authority_d_replacement_inputs_v1.invalidation.json",
        namespace.control_root / "resource_authority_d_prior_publication_failure_receipt_v1.json",
        namespace.control_root / "failed_resource_preflight_20260727T173054.689Z.json",
        *(
            namespace.control_root / "authority_d_replacement_inputs_v2" / filename
            for filename in controller.INPUT_V3_FILENAMES.values()
        ),
    )
    missing = [path for path in required_paths if not os.path.lexists(path)]
    if missing:
        rendered = ", ".join(path.relative_to(project_root).as_posix() for path in missing)
        pytest.skip(
            f"requires ignored local replacement-publication terminal evidence; missing: {rendered}"
        )
    if os.path.lexists(namespace.terminal_qualification):
        receipt = json.loads(namespace.terminal_qualification.read_text(encoding="utf-8"))
    else:
        (
            project,
            parent,
            paths,
            _input_payloads,
            input_records,
            authorization,
            attempt,
            failure,
        ) = controller._historical_terminal_payloads(
            namespace,
            parent_authority_directory=authority_c,
            pins=controller.DEFAULT_HISTORICAL_PINS,
        )
        _support, _authority, _aggregate, reads = controller._historical_support_records(
            project=project,
            parent=parent,
            paths=paths,
            pins=controller.DEFAULT_HISTORICAL_PINS,
            input_records=input_records,
            authorization_record=authorization["authorization_record"],
            attempt_record=attempt["attempt_record"],
            failure_record=failure["failure_record"],
        )
        receipt = controller._build_terminal_receipt(
            namespace,
            parent_authority_directory=authority_c,
            qualified_at=datetime(2026, 7, 28, 19, 1, tzinfo=UTC),
            process_quiescence={
                "query_method": "windows_cim_process_command_line_query_v1",
                "observer_pid": 12345,
                "observed_at_utc": "2026-07-28T19:00:00.000000Z",
                "matches": [],
                "historical_pid_inference_performed": False,
            },
            lock_quiescence={
                "scan_method": "two_pass_scoped_lock_path_scan_v1",
                "first_scan_paths": [],
                "second_scan_paths": [],
                "reads_between_scans": reads,
            },
            pins=controller.DEFAULT_HISTORICAL_PINS,
        )
    state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    return amendment, state, receipt, namespace.terminal_qualification


@dataclass(frozen=True, slots=True)
class AmendmentFixture:
    project: Path
    base: Path
    amendment_root: Path
    preregistration: Path
    primary: Path
    confirmatory: Path
    module: Path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_root(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_mapping_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in {".immutable.json", "sha256_manifest.json"}
    ]


def _seal_base(directory: Path, *, timestamp: datetime) -> None:
    rendered = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    records = _records(directory)
    root = _canonical_root(records)
    manifest = {
        "schema_version": 1,
        "freeze_timestamp_utc": rendered,
        "artifact_count": len(records),
        "artifact_root_sha256": root,
        "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
        "excluded_paths": [".immutable.json", "sha256_manifest.json"],
        "artifacts": records,
    }
    _write_json(directory / "sha256_manifest.json", manifest)
    _write_json(
        directory / ".immutable.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "freeze_timestamp_utc": rendered,
            "artifact_root_sha256": root,
            "sha256_manifest_sha256": sha256_file(directory / "sha256_manifest.json"),
            "amendment_only": True,
        },
    )


def _reseal_amendment(directory: Path) -> None:
    marker = json.loads((directory / ".immutable.json").read_text(encoding="utf-8"))
    records = _records(directory)
    root = _canonical_root(records)
    _write_json(
        directory / "sha256_manifest.json",
        {
            "schema_version": 1,
            "authority_kind": "preregistration_amendment",
            "amendment_timestamp_utc": marker["amendment_timestamp_utc"],
            "artifact_count": len(records),
            "artifact_root_sha256": root,
            "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
            "excluded_paths": [".immutable.json", "sha256_manifest.json"],
            "artifacts": records,
        },
    )
    marker["artifact_root_sha256"] = root
    marker["sha256_manifest_sha256"] = sha256_file(directory / "sha256_manifest.json")
    _write_json(directory / ".immutable.json", marker)


@pytest.fixture
def amendment_fixture(tmp_path: Path) -> AmendmentFixture:
    project = tmp_path / "project"
    module = project / "src" / "histo_audit" / "module.py"
    primary = project / "configs" / "primary.yaml"
    confirmatory = project / "configs" / "confirmatory.yaml"
    preregistration = project / "PRE_REGISTRATION.md"
    module.parent.mkdir(parents=True)
    primary.parent.mkdir()
    module.write_text("VALUE = 1\n", encoding="utf-8")
    primary.write_text("schema_version: 1\nmodel_seed: 11\n", encoding="utf-8")
    confirmatory.write_text("schema_version: 1\nmodel_seed: 12\n", encoding="utf-8")
    preregistration.write_text(
        "# Frozen preregistration\n\nPrimary analysis definition version 1.\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        "[project]\nname='amendment-test'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    base = tmp_path / "authorities" / "base"
    base.mkdir(parents=True)
    (base / "PRE_REGISTRATION_FROZEN.md").write_bytes(preregistration.read_bytes())
    (base / "primary_frozen.yaml").write_bytes(primary.read_bytes())
    (base / "confirmatory_frozen.yaml").write_bytes(confirmatory.read_bytes())
    _write_json(base / "source_tree_manifest.json", capture_source_tree(project))
    _write_json(
        base / "freeze_evidence.json",
        {
            "schema_version": 2,
            "completion_stage_enabled": "PRE_REGISTRATION_FROZEN",
        },
    )
    _seal_base(base, timestamp=BASE_TIME)
    assert verify_preregistration_freeze(base).valid
    return AmendmentFixture(
        project=project,
        base=base,
        amendment_root=tmp_path / "authorities" / "amendments",
        preregistration=preregistration,
        primary=primary,
        confirmatory=confirmatory,
        module=module,
    )


def _change_inputs(fixture: AmendmentFixture, version: int) -> None:
    fixture.preregistration.write_text(
        f"# Amended preregistration\n\nPrimary analysis definition version {version}.\n",
        encoding="utf-8",
    )
    fixture.primary.write_text(
        f"schema_version: 1\nmodel_seed: {10 + version}\n",
        encoding="utf-8",
    )
    fixture.confirmatory.write_text(
        f"schema_version: 1\nmodel_seed: {20 + version}\n",
        encoding="utf-8",
    )
    fixture.module.write_text(f"VALUE = {version}\n", encoding="utf-8")


def _create(
    fixture: AmendmentFixture,
    *,
    parent: Path | None = None,
    amendment_root: Path | None = None,
    timestamp: datetime = BASE_TIME + timedelta(hours=1),
    outcomes_inspected: bool = False,
    post_publication_check: Any = None,
) -> Any:
    return create_preregistration_amendment(
        project_root=fixture.project,
        parent_authority_directory=parent or fixture.base,
        amendment_root=amendment_root or fixture.amendment_root,
        preregistration_path=fixture.preregistration,
        primary_config_path=fixture.primary,
        confirmatory_config_path=fixture.confirmatory,
        reason="Correct a prespecified execution definition.",
        affected_hypotheses=["H1"],
        affected_analyses=["primary_ranking"],
        outcomes_inspected=outcomes_inspected,
        outcomes_inspected_at=(timestamp - timedelta(minutes=15) if outcomes_inspected else None),
        post_publication_check=post_publication_check,
        timestamp=timestamp,
    )


def test_create_successor_binds_full_before_after_snapshots_and_preserves_base(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    base_before = sha256_path(fixture.base)
    _change_inputs(fixture, 2)

    result = _create(fixture)
    verification = verify_preregistration_amendment(result.amendment_directory)

    assert verification.valid
    assert verification.chain_depth == 1
    assert verification.parent_authority_directory == fixture.base.resolve()
    assert sha256_path(fixture.base) == base_before
    assert verify_preregistration_freeze(fixture.base).valid
    assert result.amended_preregistration_path.read_bytes() == fixture.preregistration.read_bytes()
    assert result.amended_primary_config_path.read_bytes() == fixture.primary.read_bytes()
    assert result.amended_confirmatory_config_path.read_bytes() == fixture.confirmatory.read_bytes()

    evidence = json.loads(result.amendment_evidence_path.read_text(encoding="utf-8"))
    assert evidence["parent"]["authority_kind"] == "base_freeze"
    assert evidence["before"]["preregistration"]["file_sha256"] == sha256_file(
        fixture.base / "PRE_REGISTRATION_FROZEN.md"
    )
    assert evidence["after"]["preregistration"]["file_sha256"] == sha256_file(
        result.amended_preregistration_path
    )
    assert (
        evidence["before"]["execution_source"]["root_sha256"]
        != evidence["after"]["execution_source"]["root_sha256"]
    )
    assert evidence["analysis_dispositions"] == [
        {
            "analysis": "primary_ranking",
            "registration_status": "amended_before_outcome_inspection",
            "original_unamended_primary_claim_allowed": False,
            "amended_primary_claim_allowed": True,
        }
    ]


def test_generic_verifier_rejects_symlinked_ancestor_of_amendment(
    amendment_fixture: AmendmentFixture,
    tmp_path: Path,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture)
    alias_parent = tmp_path / "amendment-parent-alias"
    try:
        alias_parent.symlink_to(
            result.amendment_directory.parent,
            target_is_directory=True,
        )
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    aliased_amendment = alias_parent / result.amendment_directory.name
    verification = verify_preregistration_amendment(aliased_amendment)

    assert not verification.valid
    assert any(
        "must not traverse a symlink or reparse point" in error for error in verification.errors
    )
    assert verify_preregistration_amendment(result.amendment_directory).valid


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_generic_verifier_rejects_junction_ancestor_of_amendment(
    amendment_fixture: AmendmentFixture,
    tmp_path: Path,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture)
    alias_parent = tmp_path / "amendment-parent-junction"
    completed = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(alias_parent),
            str(result.amendment_directory.parent),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(
            "directory junctions are unavailable: " + (completed.stderr or completed.stdout).strip()
        )

    try:
        aliased_amendment = alias_parent / result.amendment_directory.name
        verification = verify_preregistration_amendment(aliased_amendment)

        assert not verification.valid
        assert any(
            "must not traverse a symlink or reparse point" in error for error in verification.errors
        )
        assert verify_preregistration_amendment(result.amendment_directory).valid
    finally:
        os.rmdir(alias_parent)


def test_transaction_scoped_post_publication_check_runs_before_commit_and_rolls_back(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    base_before = sha256_path(fixture.base)
    intended = fixture.amendment_root / "20260718T110000.000000Z"
    observations: list[tuple[Path, bool]] = []

    def reject_after_independent_readback(result: Any) -> None:
        observations.append(
            (
                result.amendment_directory,
                verify_preregistration_amendment(result.amendment_directory).valid,
            )
        )
        raise RuntimeError("injected transaction-scoped verifier failure")

    with pytest.raises(
        RuntimeError,
        match="injected transaction-scoped verifier failure",
    ):
        create_preregistration_amendment(
            project_root=fixture.project,
            parent_authority_directory=fixture.base,
            amendment_root=fixture.amendment_root,
            preregistration_path=fixture.preregistration,
            primary_config_path=fixture.primary,
            confirmatory_config_path=fixture.confirmatory,
            reason="Correct a prespecified execution definition.",
            affected_hypotheses=["H1"],
            affected_analyses=["primary_ranking"],
            outcomes_inspected=False,
            outcomes_inspected_at=None,
            post_publication_check=reject_after_independent_readback,
            timestamp=BASE_TIME + timedelta(hours=1),
        )

    assert observations == [(intended, True)]
    assert not intended.exists()
    assert sha256_path(fixture.base) == base_before
    assert verify_preregistration_freeze(fixture.base).valid
    assert not fixture.amendment_root.exists() or not tuple(fixture.amendment_root.iterdir())


def test_transaction_scoped_post_publication_check_returns_the_committed_result(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    observations: list[Any] = []

    result = create_preregistration_amendment(
        project_root=fixture.project,
        parent_authority_directory=fixture.base,
        amendment_root=fixture.amendment_root,
        preregistration_path=fixture.preregistration,
        primary_config_path=fixture.primary,
        confirmatory_config_path=fixture.confirmatory,
        reason="Correct a prespecified execution definition.",
        affected_hypotheses=["H1"],
        affected_analyses=["primary_ranking"],
        outcomes_inspected=False,
        outcomes_inspected_at=None,
        post_publication_check=observations.append,
        timestamp=BASE_TIME + timedelta(hours=1),
    )

    assert observations == [result]
    assert verify_preregistration_amendment(result.amendment_directory).valid


def test_transaction_scoped_post_publication_check_rejects_non_none_return(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    intended = fixture.amendment_root / "20260718T110000.000000Z"

    def invalid_verifier_return(_result: Any) -> Any:
        return {"invalid": "read-only verifier returned data"}

    with pytest.raises(TypeError, match="must return None"):
        create_preregistration_amendment(
            project_root=fixture.project,
            parent_authority_directory=fixture.base,
            amendment_root=fixture.amendment_root,
            preregistration_path=fixture.preregistration,
            primary_config_path=fixture.primary,
            confirmatory_config_path=fixture.confirmatory,
            reason="Correct a prespecified execution definition.",
            affected_hypotheses=["H1"],
            affected_analyses=["primary_ranking"],
            outcomes_inspected=False,
            outcomes_inspected_at=None,
            post_publication_check=invalid_verifier_return,
            timestamp=BASE_TIME + timedelta(hours=1),
        )

    assert not intended.exists()


def test_transaction_scoped_post_publication_check_rejects_recursive_creation_and_resets(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    first_intended = fixture.amendment_root / "20260718T110000.000000Z"
    recursive_intended = fixture.amendment_root / "20260718T120000.000000Z"

    def recurse_from_verifier(_result: Any) -> None:
        _create(
            fixture,
            timestamp=BASE_TIME + timedelta(hours=2),
        )

    with pytest.raises(RuntimeError, match="concurrent or recursive amendment creation"):
        create_preregistration_amendment(
            project_root=fixture.project,
            parent_authority_directory=fixture.base,
            amendment_root=fixture.amendment_root,
            preregistration_path=fixture.preregistration,
            primary_config_path=fixture.primary,
            confirmatory_config_path=fixture.confirmatory,
            reason="Correct a prespecified execution definition.",
            affected_hypotheses=["H1"],
            affected_analyses=["primary_ranking"],
            outcomes_inspected=False,
            outcomes_inspected_at=None,
            post_publication_check=recurse_from_verifier,
            timestamp=BASE_TIME + timedelta(hours=1),
        )

    assert not first_intended.exists()
    assert not recursive_intended.exists()
    result = _create(fixture)
    assert result.amendment_directory == first_intended
    assert verify_preregistration_amendment(first_intended).valid


def test_process_guard_rejects_callback_creation_from_fresh_thread(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    nested_errors: list[BaseException] = []

    def callback(_result: Any) -> None:
        def worker() -> None:
            try:
                _create(
                    fixture,
                    timestamp=BASE_TIME + timedelta(hours=2),
                )
            except BaseException as exc:
                nested_errors.append(exc)

        thread = Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert len(nested_errors) == 1
        assert "concurrent or recursive amendment creation" in str(nested_errors[0])

    result = _create(fixture, post_publication_check=callback)

    assert verify_preregistration_amendment(result.amendment_directory).valid
    assert not (fixture.amendment_root / "20260718T120000.000000Z").exists()


def test_final_inventory_check_rejects_foreign_ninth_file(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    intended = fixture.amendment_root / "20260718T110000.000000Z"

    def inject_foreign_file(result: Any) -> None:
        (result.amendment_directory / "FOREIGN.txt").write_text(
            "not owned by the amendment publication\n",
            encoding="utf-8",
        )

    with pytest.raises(
        RuntimeError,
        match="ownership-safe rollback was incomplete",
    ):
        _create(fixture, post_publication_check=inject_foreign_file)

    assert intended.exists()
    assert (intended / "FOREIGN.txt").read_text(encoding="utf-8").startswith("not owned by")
    assert not verify_preregistration_amendment(intended).valid


def test_resource_successor_verifier_rejects_same_process_before_filesystem_read(
    tmp_path: Path,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    with pytest.raises(ValueError, match="must run in a fresh process"):
        amendment.verify_resource_bounded_technical_successor(
            tmp_path / "missing-successor",
            expected_parent_authority_directory=tmp_path / "missing-parent",
            expected_artifact_root_sha256="a" * 64,
            expected_sha256_manifest_sha256="b" * 64,
            expected_authorization_sha256="c" * 64,
            expected_intent_sha256="d" * 64,
            expected_controller_process_id=os.getpid(),
            verification_nonce="e" * 64,
        )


def test_resource_successor_verifier_rejects_spoofed_parent_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    claimed_controller_pid = os.getpid() + 1
    monkeypatch.setattr(
        amendment.os,
        "getppid",
        lambda: claimed_controller_pid + 1,
    )
    with pytest.raises(ValueError, match="must be a direct child"):
        amendment.verify_resource_bounded_technical_successor(
            tmp_path / "missing-successor",
            expected_parent_authority_directory=tmp_path / "missing-parent",
            expected_artifact_root_sha256="a" * 64,
            expected_sha256_manifest_sha256="b" * 64,
            expected_authorization_sha256="c" * 64,
            expected_intent_sha256="d" * 64,
            expected_controller_process_id=claimed_controller_pid,
            verification_nonce="e" * 64,
        )


def test_prior_publication_failure_evidence_binds_exact_terminal_attempt() -> None:
    from histo_audit.workflows import (
        build_resource_bounded_prior_publication_failure_evidence,
    )

    authority_c = _require_local_prior_publication_failure_evidence()
    c_sha256_before = sha256_path(authority_c)

    evidence = build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=authority_c,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
    )

    assert evidence["attempt_marker"]["sha256"] == (
        "8c93e65eca0bb4d64af4e94012004d74178448941cc746675d0e8e72ac5e90e2"
    )
    assert evidence["failure_marker"]["sha256"] == (
        "de123683a56ab0349c44536e969f536843ed0c557bae8573187664cab7fc8615"
    )
    assert evidence["frozen_v2_inputs"]["records_sha256"] == (
        "beba860f18ef656c5f9a3e874b41d1ae9a55919fcce78213317d256c6775e76d"
    )
    assert evidence["run_state"]["canonical_sha256"] == (
        "5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4"
    )
    assert evidence["disposition"] == {
        "prior_attempt_consumed": True,
        "prior_authority_published": False,
        "failed_intended_authority_absent": True,
        "prior_success_marker_absent": True,
        "replacement_mode": "manual_new_one_shot_after_rolled_back_publication",
        "automatic_retry_allowed": False,
        "scientific_outcome_values_read": False,
        "scientific_profile_changed": False,
    }
    assert sha256_path(authority_c) == c_sha256_before


def _mock_prior_failure_receipt_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, dict[str, Any]]:
    from histo_audit.workflows import preregistration_amendment as amendment

    parent = tmp_path / "project" / "artifacts" / "preregistration_amendments" / "authority_c"
    parent.mkdir(parents=True)
    control_root = parent.parent.parent / "resource_control"
    control_root.mkdir()
    receipt = control_root / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    evidence = {
        "observed_at_utc": BASE_TIME.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "policy": "synthetic_prior_failure_test",
        "schema_version": 1,
    }

    def build_evidence(
        *,
        superseded_resource_authority_directory: str | Path,
        observed_at: datetime,
    ) -> dict[str, Any]:
        assert Path(superseded_resource_authority_directory) == parent
        assert observed_at == BASE_TIME
        return dict(evidence)

    monkeypatch.setattr(
        amendment,
        "build_resource_bounded_prior_publication_failure_evidence",
        build_evidence,
    )
    monkeypatch.setattr(amendment, "_authority_state", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        amendment,
        "_canonical_prior_resource_authority_d_publication_failure",
        lambda value, **kwargs: dict(value),
    )
    return amendment, parent, receipt, evidence


def test_prior_publication_failure_receipt_publish_verify_and_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent, receipt, evidence = _mock_prior_failure_receipt_authority(
        tmp_path,
        monkeypatch,
    )

    verification = amendment.publish_resource_bounded_prior_publication_failure_receipt(
        superseded_resource_authority_directory=parent,
        observed_at=BASE_TIME,
    )
    expected = amendment._atomic_json_bytes(evidence)

    assert receipt.read_bytes() == expected
    assert verification == {
        "receipt_path": str(receipt),
        "receipt_sha256": hashlib.sha256(expected).hexdigest(),
        "evidence": evidence,
    }
    assert (
        amendment.verify_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent,
        )
        == verification
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        amendment.publish_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent,
            observed_at=BASE_TIME,
        )
    assert receipt.read_bytes() == expected


def test_prior_publication_failure_receipt_rolls_back_post_write_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent, receipt, evidence = _mock_prior_failure_receipt_authority(
        tmp_path,
        monkeypatch,
    )

    def drifting_builder(
        *,
        superseded_resource_authority_directory: str | Path,
        observed_at: datetime,
    ) -> dict[str, Any]:
        assert Path(superseded_resource_authority_directory) == parent
        assert observed_at == BASE_TIME
        current = dict(evidence)
        if receipt.exists():
            current["policy"] = "changed_after_publication"
        return current

    monkeypatch.setattr(
        amendment,
        "build_resource_bounded_prior_publication_failure_evidence",
        drifting_builder,
    )

    with pytest.raises(ValueError, match="canonical live evidence"):
        amendment.publish_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent,
            observed_at=BASE_TIME,
        )
    assert not os.path.lexists(receipt)


def test_prior_publication_failure_receipt_rejects_noncanonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent, receipt, evidence = _mock_prior_failure_receipt_authority(
        tmp_path,
        monkeypatch,
    )
    compact = json.dumps(evidence, sort_keys=True).encode("utf-8")
    receipt.write_bytes(compact)

    with pytest.raises(ValueError, match="canonical live evidence"):
        amendment.verify_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent,
        )
    assert receipt.read_bytes() == compact


def test_prior_publication_failure_receipt_lock_exit_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent, receipt, _ = _mock_prior_failure_receipt_authority(
        tmp_path,
        monkeypatch,
    )
    original_exit = amendment.ExclusiveBundlePublicationLock.__exit__

    def fail_after_clean_lock_exit(
        self: Any,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        original_exit(self, exc_type, exc, traceback)
        raise RuntimeError("injected lock exit failure")

    monkeypatch.setattr(
        amendment.ExclusiveBundlePublicationLock,
        "__exit__",
        fail_after_clean_lock_exit,
    )

    with pytest.raises(RuntimeError, match="injected lock exit failure"):
        amendment.publish_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent,
            observed_at=BASE_TIME,
        )
    assert not os.path.lexists(receipt)


def test_prior_publication_failure_rejects_stable_frozen_input_tamper_between_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    authority_c = _require_local_prior_publication_failure_evidence().resolve()
    project_root = Path(__file__).resolve().parents[1]
    evidence = amendment.build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=authority_c,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
    )
    parent_state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    receipt_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    ).resolve()
    target = (
        project_root
        / "artifacts"
        / "resource_control"
        / "authority_d_inputs_20260727Tfinal_source_v2"
        / "authority_d_workspace_plan.json"
    ).resolve()
    real_read = amendment._read_stable_single_link_file
    original = real_read(
        target,
        role="test frozen-v2 workspace plan",
    )
    tampered = bytes([original[0] ^ 1]) + original[1:]
    target_read_count = 0

    def read_with_stable_tamper(
        path: Path,
        *,
        role: str,
        allow_empty: bool = False,
    ) -> bytes:
        nonlocal target_read_count
        if Path(path) == target:
            target_read_count += 1
            if target_read_count >= 2:
                return tampered
        return real_read(path, role=role, allow_empty=allow_empty)

    monkeypatch.setattr(
        amendment,
        "_read_stable_single_link_file",
        read_with_stable_tamper,
    )

    with pytest.raises(ValueError, match="changed from its exact historical bytes"):
        amendment._canonical_prior_resource_authority_d_publication_failure(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": amendment._atomic_json_sha256(evidence),
                "evidence": evidence,
            },
            resource_parent=parent_state,
            verify_live_receipt=True,
        )

    assert target_read_count == 2
    assert target.read_bytes() == original


def test_prior_publication_failure_canonicalizer_rejects_disposition_tamper() -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    authority_c = _require_local_prior_publication_failure_evidence().resolve()
    project_root = Path(__file__).resolve().parents[1]
    evidence = amendment.build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=authority_c,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
    )
    tampered = json.loads(json.dumps(evidence))
    tampered["disposition"]["automatic_retry_allowed"] = True
    parent_state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    receipt_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    ).resolve()

    with pytest.raises(ValueError, match="disposition is not fail-closed"):
        amendment._canonical_prior_resource_authority_d_publication_failure(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": amendment._atomic_json_sha256(tampered),
                "evidence": tampered,
            },
            resource_parent=parent_state,
            verify_live_receipt=False,
        )


@pytest.mark.parametrize(
    "target",
    [
        "disposition",
        "terminal_stdout",
    ],
)
def test_prior_publication_failure_rejects_int_bool_alias(target: str) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    authority_c = _require_local_prior_publication_failure_evidence().resolve()
    project_root = Path(__file__).resolve().parents[1]
    evidence = amendment.build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=authority_c,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
    )
    if target == "disposition":
        assert evidence["disposition"]["automatic_retry_allowed"] is False
        evidence["disposition"]["automatic_retry_allowed"] = 0
    else:
        assert evidence["terminal_stdout"]["payload"]["automatic_retry_allowed"] is False
        evidence["terminal_stdout"]["payload"]["automatic_retry_allowed"] = 0
    parent_state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    receipt_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    ).resolve()

    with pytest.raises(ValueError):
        amendment._canonical_prior_resource_authority_d_publication_failure(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": amendment._atomic_json_sha256(evidence),
                "evidence": evidence,
            },
            resource_parent=parent_state,
            verify_live_receipt=False,
        )


@pytest.mark.parametrize("schema_version", [1, 2.0, True])
def test_resource_successor_authorization_rejects_schema_downgrade_and_non_int(
    schema_version: Any,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    authorization = {
        "schema_version": schema_version,
        "policy": "post_outcome_resource_bounded_confirmatory_technical_successor_v2",
        "purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "supersedes": {},
        "prior_publication_failure": {},
        "failed_preflight": {},
        "historical_primary": {},
        "resource_profile": {},
        "execution_source_delta": {},
        "cnn_provenance_correction": {},
        "resource_capacity_policy": {},
        "resource_input_workspace_plan": {},
        "expected_successor_config_semantic_sha256": "a" * 64,
        "resource_profile_shape": {},
        "outcomes_inspected": True,
        "analysis_disposition": "amended_or_exploratory",
        "outcome_use_policy": (
            "resource_constraints_only_no_outcome_value_selection_tuning_or_exclusion"
        ),
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "primary_rebinding_allowed": False,
        "primary_mutation_allowed": False,
        "automatic_retry_allowed": False,
        "scientific_profile_change_allowed": False,
    }

    with pytest.raises(ValueError, match="unchanged post-outcome policy"):
        amendment._canonical_resource_bounded_technical_successor_authorization(
            authorization,
            parent_state=None,
            parent_resource_authorization={},
            successor_source={},
            successor_source_manifest_sha256="b" * 64,
            successor_config={},
            successor_config_file_sha256="c" * 64,
            successor_config_semantic_sha256="d" * 64,
            verify_live_receipt=False,
        )


def test_resource_successor_authorization_serializes_v2_and_v3_compatibly() -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    arguments = {
        "superseded_authority": {"authority": "C"},
        "prior_publication_failure": {"receipt": "prior"},
        "failed_preflight": {"receipt": "preflight"},
        "historical_primary": {"run_id": "primary"},
        "resource_profile": {"profile_id": "bounded"},
        "execution_source_delta": {"root": "source"},
        "cnn_provenance_correction": {"status": "corrected"},
        "resource_capacity_policy": {"capacity": "fixed"},
        "resource_input_workspace_plan": {"workspace": "typed"},
        "expected_successor_config_semantic_sha256": "a" * 64,
    }
    schema_v2 = amendment.ResourceBoundedTechnicalSuccessorAuthorization(
        **arguments,
    ).as_dict()
    lineage = {
        "terminal_qualification_receipt_path": r"C:\exact\Q.json",
        "terminal_qualification_receipt_sha256": "b" * 64,
        "terminal_qualification_receipt": {"status": "qualified"},
    }
    schema_v3 = amendment.ResourceBoundedTechnicalSuccessorAuthorization(
        **arguments,
        replacement_publication_failure_lineage=lineage,
    ).as_dict()

    assert schema_v2["schema_version"] == 2
    assert schema_v2["policy"] == (
        "post_outcome_resource_bounded_confirmatory_technical_successor_v2"
    )
    assert "replacement_publication_failure_lineage" not in schema_v2
    assert schema_v3["schema_version"] == 3
    assert schema_v3["policy"] == (
        "post_outcome_resource_bounded_confirmatory_technical_successor_v3"
    )
    assert schema_v3["replacement_publication_failure_lineage"] == lineage
    comparable_v3 = dict(schema_v3)
    comparable_v3["schema_version"] = 2
    comparable_v3["policy"] = schema_v2["policy"]
    comparable_v3.pop("replacement_publication_failure_lineage")
    assert comparable_v3 == schema_v2


def test_schema_v3_terminal_qualification_matches_controller_contract() -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()

    canonical_receipt = amendment._canonical_replacement_publication_terminal_qualification(
        receipt,
        resource_parent=parent_state,
        verify_live_records=False,
    )
    lineage = {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": amendment._atomic_json_sha256(canonical_receipt),
        "terminal_qualification_receipt": canonical_receipt,
    }

    assert canonical_receipt == receipt
    assert (
        amendment._canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=False,
        )
        == lineage
    )
    assert list(canonical_receipt["controller_identities"]) == [
        "consumed_attempt_controller",
        "diagnosed_fixed_legacy_controller",
        "qualifying_live_controller",
    ]
    assert [
        record["role"] for record in canonical_receipt["lock_quiescence"]["reads_between_scans"]
    ] == [
        "publication_authorization_receipt",
        "publication_attempt_marker",
        "publication_failure_marker",
        "retired_v1_invalidation_receipt",
        "prior_publication_failure_receipt",
        "failed_preflight_receipt",
        "v2_cnn_correction_receipt",
        "v2_frozen_source_receipt",
        "v2_source_allowlist",
        "v2_workspace_plan",
        "protected_specification",
        "protected_pre_registration",
        "protected_primary_config",
        "protected_confirmatory_config",
        "authority_c_amendment_evidence",
        "authority_c_amendment_report",
        "authority_c_confirmatory_config",
        "authority_c_immutable_marker",
        "authority_c_preregistration",
        "authority_c_primary_config",
        "authority_c_sha256_manifest",
        "authority_c_source_tree_manifest",
        "run_state_integrity_registry.jsonl",
        "run_state_registry.csv",
        "run_state_run_dispositions.anchor.json",
        "run_state_run_dispositions.jsonl",
        "run_state_run_stage_attestations.anchor.json",
        "run_state_run_stage_attestations.jsonl",
    ]
    assert canonical_receipt["run_state"]["sha256"] == _canonical_mapping_sha256(
        {name: record["sha256"] for name, record in canonical_receipt["run_state"]["files"].items()}
    )


@pytest.mark.parametrize(
    "path_keys",
    [
        ("project_root",),
        ("authority_c", "directory"),
        ("authority_c", "files", "amendment_evidence", "path"),
        ("terminal_namespace", "success_marker_absence", "path"),
        ("frozen_v2_inputs", "directory"),
        ("frozen_v2_inputs", "files", "workspace_plan", "path"),
        ("controller_identities", "consumed_attempt_controller", "path"),
        ("controller_identities", "diagnosed_fixed_legacy_controller", "path"),
        ("controller_identities", "qualifying_live_controller", "path"),
        ("run_state", "root"),
        ("run_state", "files", "registry.csv", "path"),
        ("protected_bindings", "specification", "path"),
        ("lock_quiescence", "reads_between_scans", 0, "path"),
    ],
)
def test_schema_v3_terminal_qualification_rejects_lexical_path_alias(
    path_keys: tuple[str | int, ...],
) -> None:
    amendment, parent_state, receipt, _receipt_path = _local_replacement_terminal_contract()
    tampered = json.loads(json.dumps(receipt))
    current: Any = tampered
    for key in path_keys[:-1]:
        current = current[key]
    leaf = path_keys[-1]
    original = Path(current[leaf])
    current[leaf] = str(original.parent / ".." / original.parent.name / original.name)

    with pytest.raises(ValueError):
        amendment._canonical_replacement_publication_terminal_qualification(
            tampered,
            resource_parent=parent_state,
            verify_live_records=False,
        )


def test_schema_v3_failure_lineage_rejects_receipt_path_alias_and_hash_tamper() -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()
    canonical_receipt = amendment._canonical_replacement_publication_terminal_qualification(
        receipt,
        resource_parent=parent_state,
        verify_live_records=False,
    )
    lineage = {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": amendment._atomic_json_sha256(canonical_receipt),
        "terminal_qualification_receipt": canonical_receipt,
    }
    aliased = json.loads(json.dumps(lineage))
    aliased["terminal_qualification_receipt_path"] = str(
        receipt_path.parent / ".." / receipt_path.parent.name / receipt_path.name
    )
    with pytest.raises(ValueError, match="exact canonical absolute path"):
        amendment._canonical_replacement_publication_failure_lineage(
            aliased,
            resource_parent=parent_state,
            verify_live_receipt=False,
        )

    hash_tampered = json.loads(json.dumps(lineage))
    hash_tampered["terminal_qualification_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt is not canonical"):
        amendment._canonical_replacement_publication_failure_lineage(
            hash_tampered,
            resource_parent=parent_state,
            verify_live_receipt=False,
        )


def test_schema_v3_public_terminal_verifier_rejects_lexical_receipt_alias() -> None:
    amendment, parent_state, _receipt, receipt_path = _local_replacement_terminal_contract()
    alias = receipt_path.parent / ".." / receipt_path.parent.name / receipt_path.name

    with pytest.raises(ValueError, match="exact canonical lexical path"):
        amendment.verify_resource_bounded_replacement_terminal_qualification_receipt(
            alias,
            project_root=parent_state.directory.parent.parent.parent,
            parent_authority_directory=parent_state.directory,
        )


def test_schema_v3_terminal_qualification_requires_exact_authority_c_pin() -> None:
    amendment, parent_state, receipt, _receipt_path = _local_replacement_terminal_contract()
    wrong_parent = amendment._AuthorityState(
        directory=parent_state.directory,
        kind=parent_state.kind,
        timestamp_utc=parent_state.timestamp_utc,
        timestamp=parent_state.timestamp,
        chain_depth=parent_state.chain_depth,
        artifact_root_sha256="0" * 64,
        sha256_manifest_sha256=parent_state.sha256_manifest_sha256,
        snapshot_hashes=parent_state.snapshot_hashes,
        parent_directory=parent_state.parent_directory,
    )

    with pytest.raises(ValueError, match="wrong Authority C"):
        amendment._canonical_replacement_publication_terminal_qualification(
            receipt,
            resource_parent=wrong_parent,
            verify_live_records=False,
        )


def test_schema_v3_terminal_qualification_rejects_numeric_aliases_and_duplicate_reads() -> None:
    amendment, parent_state, receipt, _receipt_path = _local_replacement_terminal_contract()
    mutations: list[tuple[tuple[str | int, ...], Any]] = [
        (("authority_c", "chain_depth"), float(receipt["authority_c"]["chain_depth"])),
        (("authority_c", "flat_file_count"), 8.0),
        (("terminal_namespace", "candidate_count"), 0.0),
        (("terminal_links", "max_attempt_count"), True),
        (
            (
                "controller_identities",
                "consumed_attempt_controller",
                "size_bytes",
            ),
            216288.0,
        ),
        (
            (
                "controller_identities",
                "diagnosed_fixed_legacy_controller",
                "size_bytes",
            ),
            218766.0,
        ),
        (("process_quiescence", "observer_pid"), 12345.0),
        (("disposition", "v1_retry_allowed"), 0),
        (("failure_cause", "scientific_or_evidence_corruption"), 0),
        (("scientific_execution_performed",), 0),
    ]
    for path_keys, replacement in mutations:
        tampered = json.loads(json.dumps(receipt))
        current: Any = tampered
        for key in path_keys[:-1]:
            current = current[key]
        current[path_keys[-1]] = replacement
        with pytest.raises(ValueError):
            amendment._canonical_replacement_publication_terminal_qualification(
                tampered,
                resource_parent=parent_state,
                verify_live_records=False,
            )

    duplicated_reads = json.loads(json.dumps(receipt))
    first_read = duplicated_reads["lock_quiescence"]["reads_between_scans"][0]
    duplicated_reads["lock_quiescence"]["reads_between_scans"] = [
        dict(first_read) for _ in range(28)
    ]
    with pytest.raises(ValueError, match="lock-quiescence"):
        amendment._canonical_replacement_publication_terminal_qualification(
            duplicated_reads,
            resource_parent=parent_state,
            verify_live_records=False,
        )

    swapped_reads = json.loads(json.dumps(receipt))
    reads = swapped_reads["lock_quiescence"]["reads_between_scans"]
    reads[0], reads[1] = reads[1], reads[0]
    with pytest.raises(ValueError, match="lock-quiescence"):
        amendment._canonical_replacement_publication_terminal_qualification(
            swapped_reads,
            resource_parent=parent_state,
            verify_live_records=False,
        )

    stale_process_probe = json.loads(json.dumps(receipt))
    qualified_at = datetime.fromisoformat(stale_process_probe["qualified_at_utc"][:-1] + "+00:00")
    stale_process_probe["process_quiescence"]["observed_at_utc"] = (
        (qualified_at - timedelta(seconds=61))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    with pytest.raises(ValueError, match="process-quiescence"):
        amendment._canonical_replacement_publication_terminal_qualification(
            stale_process_probe,
            resource_parent=parent_state,
            verify_live_records=False,
        )

    arbitrary_run_record = json.loads(json.dumps(receipt))
    arbitrary_run_record["run_state"]["files"]["registry.csv"] = dict(
        arbitrary_run_record["run_state"]["files"]["integrity_registry.jsonl"]
    )
    with pytest.raises(ValueError, match=r"run-state registry\.csv"):
        amendment._canonical_replacement_publication_terminal_qualification(
            arbitrary_run_record,
            resource_parent=parent_state,
            verify_live_records=False,
        )


def test_schema_v3_live_terminal_qualification_rejects_future_timestamp() -> None:
    amendment, parent_state, receipt, _receipt_path = _local_replacement_terminal_contract()
    tampered = json.loads(json.dumps(receipt))
    future = datetime.now(UTC) + timedelta(days=1)
    observed = future - timedelta(minutes=1)
    tampered["qualified_at_utc"] = future.isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )
    tampered["process_quiescence"]["observed_at_utc"] = observed.isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")

    with pytest.raises(ValueError, match="fixed policy"):
        amendment._canonical_replacement_publication_terminal_qualification(
            tampered,
            resource_parent=parent_state,
            verify_live_records=True,
        )


def test_schema_v3_sealed_lineage_survives_permitted_run_state_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()
    lineage = {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": amendment._atomic_json_sha256(receipt),
        "terminal_qualification_receipt": receipt,
    }
    run_paths = {Path(record["path"]) for record in receipt["run_state"]["files"].values()}
    real_read = amendment._read_stable_single_link_file
    run_state_reads = 0

    def read_after_lifecycle_advance(
        path: Path,
        *,
        role: str,
        allow_empty: bool = False,
        max_bytes: int | None = None,
    ) -> bytes:
        nonlocal run_state_reads
        lexical = Path(path)
        if lexical == receipt_path:
            return amendment._atomic_json_bytes(receipt)
        payload = real_read(
            lexical,
            role=role,
            allow_empty=allow_empty,
            max_bytes=max_bytes,
        )
        if lexical in run_paths:
            run_state_reads += 1
            return payload + b"\n"
        return payload

    monkeypatch.setattr(
        amendment,
        "_read_stable_single_link_file",
        read_after_lifecycle_advance,
    )
    sealed = amendment._canonical_replacement_publication_failure_lineage(
        lineage,
        resource_parent=parent_state,
        verify_live_receipt=True,
        verify_live_run_state=False,
    )
    assert sealed == lineage
    assert run_state_reads == 0

    with pytest.raises(ValueError, match="live bytes changed"):
        amendment._canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=True,
            verify_live_run_state=True,
        )
    assert run_state_reads > 0


def test_schema_v3_intent_hash_dispatches_and_binds_terminal_qualification() -> None:
    amendment, parent_state, receipt, receipt_path = _local_replacement_terminal_contract()
    canonical_receipt = amendment._canonical_replacement_publication_terminal_qualification(
        receipt,
        resource_parent=parent_state,
        verify_live_records=False,
    )
    lineage = {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": amendment._atomic_json_sha256(canonical_receipt),
        "terminal_qualification_receipt": canonical_receipt,
    }
    qualification_at = datetime.fromisoformat(canonical_receipt["qualified_at_utc"][:-1] + "+00:00")
    prior_failure_at = qualification_at - timedelta(minutes=1)
    failed_preflight_at = parent_state.timestamp + timedelta(minutes=1)
    publication_at = qualification_at + timedelta(minutes=1)

    def rendered(value: datetime) -> str:
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    authorization_v3 = {
        "schema_version": 3,
        "policy": "post_outcome_resource_bounded_confirmatory_technical_successor_v3",
        "purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "prior_publication_failure": {"evidence": {"observed_at_utc": rendered(prior_failure_at)}},
        "failed_preflight": {"evidence": {"observed_at_utc": rendered(failed_preflight_at)}},
        "replacement_publication_failure_lineage": lineage,
        "outcomes_inspected": True,
        "analysis_disposition": "amended_or_exploratory",
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "automatic_retry_allowed": False,
        "scientific_profile_change_allowed": False,
    }
    common = {
        "parent_authority_directory": parent_state.directory,
        "amendment_timestamp_utc": rendered(publication_at),
        "reason": "Bind the qualified replacement-publication terminal failure.",
        "affected_hypotheses": ["H1"],
        "affected_analyses": ["resource_sensitivity"],
        "outcomes_inspected_at_utc": rendered(prior_failure_at),
        "confirmatory_storage_policy": amendment.ConfirmatoryStoragePolicy().as_dict(),
    }
    schema_v3_hash = amendment.resource_bounded_technical_successor_intent_sha256(
        authorization=authorization_v3,
        **common,
    )
    authorization_v2 = dict(authorization_v3)
    authorization_v2["schema_version"] = 2
    authorization_v2["policy"] = "post_outcome_resource_bounded_confirmatory_technical_successor_v2"
    authorization_v2.pop("replacement_publication_failure_lineage")
    schema_v2_hash = amendment.resource_bounded_technical_successor_intent_sha256(
        authorization=authorization_v2,
        **common,
    )

    assert len(schema_v3_hash) == 64
    assert len(schema_v2_hash) == 64
    assert schema_v3_hash != schema_v2_hash
    tampered = json.loads(json.dumps(authorization_v3))
    tampered["replacement_publication_failure_lineage"]["terminal_qualification_receipt_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="receipt is not canonical"):
        amendment.resource_bounded_technical_successor_intent_sha256(
            authorization=tampered,
            **common,
        )


def test_schema_v3_root_inventory_rejects_files_and_extra_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    amendment_root = tmp_path / "artifacts" / "preregistration_amendments"
    paths = {
        role: amendment_root / role
        for role in ("authority_a", "authority_p", "authority_c", "authority_d")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "artifacts" / "preregistration_freeze"

    def state(
        directory: Path,
        *,
        kind: str,
        parent: Path | None,
        depth: int,
    ) -> Any:
        return amendment._AuthorityState(
            directory=directory,
            kind=kind,
            timestamp_utc="2026-07-28T18:00:00.000000Z",
            timestamp=datetime(2026, 7, 28, 18, 0, tzinfo=UTC),
            chain_depth=depth,
            artifact_root_sha256="a" * 64,
            sha256_manifest_sha256="b" * 64,
            snapshot_hashes={},
            parent_directory=parent,
        )

    states = {
        paths["authority_c"]: state(
            paths["authority_c"],
            kind="preregistration_amendment",
            parent=paths["authority_p"],
            depth=3,
        ),
        paths["authority_p"]: state(
            paths["authority_p"],
            kind="preregistration_amendment",
            parent=paths["authority_a"],
            depth=2,
        ),
        paths["authority_a"]: state(
            paths["authority_a"],
            kind="preregistration_amendment",
            parent=base,
            depth=1,
        ),
        base: state(base, kind="base_freeze", parent=None, depth=0),
    }
    monkeypatch.setattr(
        amendment,
        "_authority_state",
        lambda directory, **_kwargs: states[Path(directory)],
    )

    inventory = amendment._strict_schema_v3_resource_amendment_root_inventory(
        paths["authority_c"],
        paths["authority_d"],
    )
    assert set(inventory) == set(paths.values())

    foreign_file = amendment_root / "foreign.txt"
    foreign_file.write_text("ambiguous peer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains a file"):
        amendment._strict_schema_v3_resource_amendment_root_inventory(
            paths["authority_c"],
            paths["authority_d"],
        )
    foreign_file.unlink()

    (amendment_root / "foreign_directory").mkdir()
    with pytest.raises(ValueError, match="exactly baseline A/P/C plus successor D"):
        amendment._strict_schema_v3_resource_amendment_root_inventory(
            paths["authority_c"],
            paths["authority_d"],
        )


def test_prior_publication_failure_rejects_non_integer_schema() -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    authority_c = _require_local_prior_publication_failure_evidence().resolve()
    project_root = Path(__file__).resolve().parents[1]
    evidence = amendment.build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=authority_c,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
    )
    evidence["schema_version"] = 1.0
    parent_state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    receipt_path = (
        project_root
        / "artifacts"
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    ).resolve()

    with pytest.raises(ValueError, match="receipt identity is invalid"):
        amendment._canonical_prior_resource_authority_d_publication_failure(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": amendment._atomic_json_sha256(evidence),
                "evidence": evidence,
            },
            resource_parent=parent_state,
            verify_live_receipt=False,
        )


def test_lock_exit_failure_after_publication_still_rolls_back_owned_bundle(
    amendment_fixture: AmendmentFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    intended = fixture.amendment_root / "20260718T110000.000000Z"
    original_exit = amendment.ExclusiveBundlePublicationLock.__exit__

    def fail_after_clean_lock_exit(
        self: Any,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> Any:
        result = original_exit(self, exc_type, exc, traceback)
        if exc_type is None:
            raise RuntimeError("injected lock-exit failure")
        return result

    monkeypatch.setattr(
        amendment.ExclusiveBundlePublicationLock,
        "__exit__",
        fail_after_clean_lock_exit,
    )

    with pytest.raises(RuntimeError, match="injected lock-exit failure"):
        _create(fixture)

    assert not intended.exists()
    assert verify_preregistration_freeze(fixture.base).valid


def test_schema_v5_d_uses_historical_c_policy_but_c_is_not_effective_leaf(
    amendment_fixture: AmendmentFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histo_audit.workflows import preregistration_amendment as amendment

    fixture = amendment_fixture
    policy = amendment.ConfirmatoryStoragePolicy().as_dict()

    # Exercise physical bundle creation, publication, chain verification, leaf
    # selection, and rollback primitives. Stub only the unrelated large
    # scientific/live-primary authorization payload canonicalizers.
    monkeypatch.setattr(
        amendment,
        "_canonical_primary_recovery_authorization",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        amendment,
        "_canonical_resource_bounded_confirmatory_authorization",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        amendment,
        "_canonical_resource_bounded_technical_successor_authorization",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        amendment,
        "_amendment_markdown",
        lambda **_kwargs: "# Synthetic C-to-D authority regression\n",
    )

    inspected_at = BASE_TIME + timedelta(minutes=15)
    fixture.module.write_text("VALUE = 2\n", encoding="utf-8")
    recovery_authorization = {
        "interruption_evidence": {
            "observed_at_utc": "2026-07-18T10:30:00.000000Z",
        }
    }
    recovery = create_preregistration_amendment(
        project_root=fixture.project,
        parent_authority_directory=fixture.base,
        amendment_root=fixture.amendment_root,
        preregistration_path=fixture.preregistration,
        primary_config_path=fixture.primary,
        confirmatory_config_path=fixture.confirmatory,
        reason="Synthetic recovery parent.",
        affected_hypotheses=["H1"],
        affected_analyses=["primary_ranking"],
        outcomes_inspected=True,
        outcomes_inspected_at=inspected_at,
        primary_recovery_authorization=recovery_authorization,
        confirmatory_storage_policy=policy,
        timestamp=BASE_TIME + timedelta(hours=1),
    )
    assert verify_preregistration_amendment(recovery.amendment_directory).valid

    config_c = fixture.project / "configs" / "resource_c.yaml"
    config_c.write_text("schema_version: 1\nmodel_seed: 13\n", encoding="utf-8")
    fixture.module.write_text("VALUE = 3\n", encoding="utf-8")
    authorization_c = {"fixture_authority": "C"}
    authority_c = create_preregistration_amendment(
        project_root=fixture.project,
        parent_authority_directory=recovery.amendment_directory,
        amendment_root=fixture.amendment_root,
        preregistration_path=recovery.amended_preregistration_path,
        primary_config_path=recovery.amended_primary_config_path,
        confirmatory_config_path=config_c,
        reason="Synthetic resource authority C.",
        affected_hypotheses=["H1"],
        affected_analyses=["resource_sensitivity"],
        outcomes_inspected=True,
        outcomes_inspected_at=inspected_at,
        resource_bounded_confirmatory_authorization=authorization_c,
        confirmatory_storage_policy=policy,
        timestamp=BASE_TIME + timedelta(hours=2),
    )
    authority_c_verification = verify_preregistration_amendment(authority_c.amendment_directory)
    assert authority_c_verification.valid
    c_sha256_before_d = sha256_path(authority_c.amendment_directory)

    config_d = fixture.project / "configs" / "resource_d.yaml"
    config_d.write_text("schema_version: 1\nmodel_seed: 14\n", encoding="utf-8")
    fixture.module.write_text("VALUE = 4\n", encoding="utf-8")
    authorization_d = {
        "schema_version": 2,
        "policy": "post_outcome_resource_bounded_confirmatory_technical_successor_v2",
        "purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "supersedes": {
            "authority_directory": str(authority_c.amendment_directory),
            "authority_schema_version": 4,
            "artifact_root_sha256": authority_c_verification.artifact_root_sha256,
            "sha256_manifest_sha256": (authority_c_verification.sha256_manifest_sha256),
            "chain_depth": authority_c_verification.chain_depth,
            "amendment_evidence_sha256": sha256_file(authority_c.amendment_evidence_path),
            "authorization_sha256": _canonical_mapping_sha256(authorization_c),
            "effective_execution_leaf": False,
            "historical_verification_retained": True,
        },
        "failed_preflight": {
            "evidence": {
                "observed_at_utc": "2026-07-18T12:30:00.000000Z",
            }
        },
        "prior_publication_failure": {
            "receipt_path": str(fixture.project / "synthetic-prior-failure.json"),
            "receipt_sha256": "e" * 64,
            "evidence": {
                "fixture": "prior failed publication",
                "observed_at_utc": "2026-07-18T12:45:00.000000Z",
            },
        },
        "fixture_authority": "D",
        "outcomes_inspected": True,
        "analysis_disposition": "amended_or_exploratory",
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "primary_rebinding_allowed": False,
        "primary_mutation_allowed": False,
        "automatic_retry_allowed": False,
        "scientific_profile_change_allowed": False,
    }
    d_timestamp = BASE_TIME + timedelta(hours=3)
    d_timestamp_text = d_timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    legacy_authorization_d = dict(authorization_d)
    legacy_authorization_d["schema_version"] = 1
    legacy_authorization_d.pop("prior_publication_failure")
    with pytest.raises(ValueError, match="authorization is not non-claiming"):
        amendment.resource_bounded_technical_successor_intent_sha256(
            parent_authority_directory=authority_c.amendment_directory,
            amendment_timestamp_utc=d_timestamp_text,
            reason="Synthetic technical successor D.",
            affected_hypotheses=["H1"],
            affected_analyses=["resource_sensitivity"],
            outcomes_inspected_at_utc=inspected_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            authorization=legacy_authorization_d,
            confirmatory_storage_policy=policy,
        )
    with pytest.raises(ValueError, match="timestamps do not follow"):
        amendment.resource_bounded_technical_successor_intent_sha256(
            parent_authority_directory=authority_c.amendment_directory,
            amendment_timestamp_utc="2026-07-18T12:40:00.000000Z",
            reason="Synthetic technical successor D.",
            affected_hypotheses=["H1"],
            affected_analyses=["resource_sensitivity"],
            outcomes_inspected_at_utc=inspected_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            authorization=authorization_d,
            confirmatory_storage_policy=policy,
        )
    expected_d_intent_sha256 = amendment.resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=authority_c.amendment_directory,
        amendment_timestamp_utc=d_timestamp_text,
        reason="Synthetic technical successor D.",
        affected_hypotheses=["H1"],
        affected_analyses=["resource_sensitivity"],
        outcomes_inspected_at_utc=inspected_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        authorization=authorization_d,
        confirmatory_storage_policy=policy,
    )
    expected_d_authorization_sha256 = _canonical_mapping_sha256(authorization_d)

    def create_d(
        post_publication_check: Any = None,
    ) -> Any:
        return create_preregistration_amendment(
            project_root=fixture.project,
            parent_authority_directory=authority_c.amendment_directory,
            amendment_root=fixture.amendment_root,
            preregistration_path=authority_c.amended_preregistration_path,
            primary_config_path=authority_c.amended_primary_config_path,
            confirmatory_config_path=config_d,
            reason="Synthetic technical successor D.",
            affected_hypotheses=["H1"],
            affected_analyses=["resource_sensitivity"],
            outcomes_inspected=True,
            outcomes_inspected_at=inspected_at,
            resource_bounded_technical_successor_authorization=authorization_d,
            confirmatory_storage_policy=policy,
            post_publication_check=post_publication_check,
            timestamp=d_timestamp,
        )

    intended_d = fixture.amendment_root / "20260718T130000.000000Z"
    historical_reader = amendment._require_historical_resource_bounded_confirmatory_storage_policy
    monkeypatch.setattr(
        amendment,
        "_require_historical_resource_bounded_confirmatory_storage_policy",
        amendment.require_confirmatory_storage_policy,
    )
    peers_before = tuple(sorted(path.name for path in fixture.amendment_root.iterdir()))
    with pytest.raises(
        RuntimeError,
        match="historically valid but no longer the effective execution leaf",
    ):
        create_d()
    assert not intended_d.exists()
    assert sha256_path(authority_c.amendment_directory) == c_sha256_before_d
    assert tuple(sorted(path.name for path in fixture.amendment_root.iterdir())) == peers_before
    assert (
        amendment._resource_technical_successor_candidate_directories(
            authority_c.amendment_directory
        )
        == ()
    )
    monkeypatch.setattr(
        amendment,
        "_require_historical_resource_bounded_confirmatory_storage_policy",
        historical_reader,
    )

    callback_observations: list[Path] = []

    def reject_after_typed_d_readback(result: Any) -> None:
        assert (
            amendment.require_resource_bounded_technical_successor_authorization(
                result.amendment_directory
            )
            == authorization_d
        )
        callback_observations.append(result.amendment_directory)
        raise RuntimeError("injected schema-v5 transaction verifier failure")

    with pytest.raises(
        RuntimeError,
        match="injected schema-v5 transaction verifier failure",
    ):
        create_d(reject_after_typed_d_readback)
    assert callback_observations == [intended_d]
    assert not intended_d.exists()
    assert sha256_path(authority_c.amendment_directory) == c_sha256_before_d
    assert (
        amendment._resource_technical_successor_candidate_directories(
            authority_c.amendment_directory
        )
        == ()
    )

    accepted_observations: list[Path] = []
    authority_d = create_d(lambda result: accepted_observations.append(result.amendment_directory))

    assert verify_preregistration_amendment(authority_d.amendment_directory).valid
    assert accepted_observations == [authority_d.amendment_directory]
    synthetic_controller_process_id = os.getpid() + 1
    monkeypatch.setattr(
        amendment.os,
        "getppid",
        lambda: synthetic_controller_process_id,
    )
    fresh_verification = amendment.verify_resource_bounded_technical_successor(
        authority_d.amendment_directory,
        expected_parent_authority_directory=authority_c.amendment_directory,
        expected_artifact_root_sha256=authority_d.artifact_root_sha256,
        expected_sha256_manifest_sha256=authority_d.sha256_manifest_sha256,
        expected_authorization_sha256=expected_d_authorization_sha256,
        expected_intent_sha256=expected_d_intent_sha256,
        expected_controller_process_id=synthetic_controller_process_id,
        verification_nonce="f" * 64,
    )
    assert fresh_verification.successor_directory == authority_d.amendment_directory
    assert fresh_verification.parent_authority_directory == (authority_c.amendment_directory)
    assert fresh_verification.flat_file_count == 8
    assert fresh_verification.manifest_artifact_count == 6
    assert fresh_verification.as_dict()["checks"] == {
        "generic_chain_integrity": True,
        "typed_successor_authorization": True,
        "effective_execution_leaf": True,
        "historical_c_integrity": True,
        "historical_c_typed_authorization": True,
        "c_superseded_for_execution": True,
        "unique_direct_successor": True,
        "storage_policy_inherited_unchanged": True,
        "flat_exact_file_set": True,
        "external_intent_binding": True,
        "fresh_process_boundary": True,
        "live_prior_publication_failure": True,
        "live_failed_preflight_receipt": True,
        "live_historical_primary": True,
    }
    assert fresh_verification.controller_process_id == synthetic_controller_process_id
    assert fresh_verification.verifier_process_id == os.getpid()
    assert fresh_verification.verifier_parent_process_id == synthetic_controller_process_id
    assert fresh_verification.verification_nonce == "f" * 64
    assert (
        amendment.require_resource_bounded_technical_successor_authorization(
            authority_d.amendment_directory
        )
        == authorization_d
    )
    assert (
        amendment.require_effective_resource_bounded_confirmatory_authorization(
            authority_d.amendment_directory
        )
        == authorization_d
    )
    assert sha256_path(authority_c.amendment_directory) == c_sha256_before_d
    assert verify_preregistration_amendment(authority_c.amendment_directory).valid
    assert (
        amendment.require_resource_bounded_confirmatory_authorization(
            authority_c.amendment_directory
        )
        == authorization_c
    )
    assert (
        amendment._require_historical_resource_bounded_confirmatory_storage_policy(
            authority_c.amendment_directory
        )
        == policy
    )
    with pytest.raises(
        ValueError,
        match="no longer the effective execution leaf",
    ):
        amendment.require_confirmatory_storage_policy(authority_c.amendment_directory)
    assert amendment.require_confirmatory_storage_policy(authority_d.amendment_directory) == policy
    with pytest.raises(
        ValueError,
        match="historically valid but superseded by effective successor D",
    ):
        amendment.require_effective_resource_bounded_confirmatory_authorization(
            authority_c.amendment_directory
        )
    assert amendment._resource_technical_successor_candidate_directories(
        authority_c.amendment_directory
    ) == (authority_d.amendment_directory.resolve(),)

    d_sha256_before_fork = sha256_path(authority_d.amendment_directory)
    fork = fixture.amendment_root / "20260718T140000.000000Z"
    shutil.copytree(authority_d.amendment_directory, fork)
    assert amendment._resource_technical_successor_candidate_directories(
        authority_c.amendment_directory
    ) == (authority_d.amendment_directory.resolve(), fork.resolve())
    with pytest.raises(ValueError, match="forbidden technical-successor fork"):
        amendment.require_effective_resource_bounded_confirmatory_authorization(
            authority_c.amendment_directory
        )
    with pytest.raises(ValueError, match="must have exactly one technical successor D"):
        amendment.require_resource_bounded_technical_successor_authorization(
            authority_d.amendment_directory
        )
    with pytest.raises(ValueError, match="must have exactly one technical successor D"):
        amendment.require_effective_resource_bounded_confirmatory_authorization(
            authority_d.amendment_directory
        )
    assert sha256_path(authority_c.amendment_directory) == c_sha256_before_d
    assert sha256_path(authority_d.amendment_directory) == d_sha256_before_fork


def test_amendment_can_explicitly_name_an_amendment_parent(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    first = _create(fixture)
    first_before = sha256_path(first.amendment_directory)
    _change_inputs(fixture, 3)

    second = _create(
        fixture,
        parent=first.amendment_directory,
        timestamp=BASE_TIME + timedelta(hours=2),
    )
    verification = verify_preregistration_amendment(second.amendment_directory)

    assert verification.valid
    assert verification.chain_depth == 2
    assert verification.parent_authority_directory == first.amendment_directory
    assert sha256_path(first.amendment_directory) == first_before
    evidence = json.loads(second.amendment_evidence_path.read_text(encoding="utf-8"))
    assert evidence["parent"]["authority_kind"] == "preregistration_amendment"
    assert evidence["parent"]["chain_depth"] == 1


def test_post_outcome_amendment_forces_amended_or_exploratory_reporting(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture, outcomes_inspected=True)

    evidence = json.loads(result.amendment_evidence_path.read_text(encoding="utf-8"))
    disposition = evidence["analysis_dispositions"][0]
    assert evidence["outcomes_inspected"] is True
    assert disposition["registration_status"] == "amended_or_exploratory"
    assert disposition["original_unamended_primary_claim_allowed"] is False
    assert disposition["amended_primary_claim_allowed"] is False
    assert "never be reported as the original unamended primary" in (
        result.amendment_directory / "AMENDMENT.md"
    ).read_text(encoding="utf-8")
    assert verify_preregistration_amendment(result.amendment_directory).valid


def test_outcome_timestamp_contract_fails_closed(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    common = {
        "project_root": fixture.project,
        "parent_authority_directory": fixture.base,
        "amendment_root": fixture.amendment_root,
        "preregistration_path": fixture.preregistration,
        "primary_config_path": fixture.primary,
        "confirmatory_config_path": fixture.confirmatory,
        "reason": "Correct a prespecified execution definition.",
        "affected_hypotheses": ["H1"],
        "affected_analyses": ["primary_ranking"],
        "timestamp": BASE_TIME + timedelta(hours=1),
    }
    with pytest.raises(ValueError, match="required"):
        create_preregistration_amendment(
            **common,
            outcomes_inspected=True,
            outcomes_inspected_at=None,
        )
    with pytest.raises(ValueError, match="must be omitted"):
        create_preregistration_amendment(
            **common,
            outcomes_inspected=False,
            outcomes_inspected_at=BASE_TIME,
        )
    assert not fixture.amendment_root.exists()


def test_no_overwrite_and_snapshot_tamper_are_detected(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture)
    amendment_before = sha256_path(result.amendment_directory)

    with pytest.raises(FileExistsError, match="already exists"):
        _create(fixture)
    assert sha256_path(result.amendment_directory) == amendment_before

    result.amended_primary_config_path.write_text("schema_version: 1\nmodel_seed: 999\n")
    verification = verify_preregistration_amendment(result.amendment_directory)
    assert not verification.valid
    assert "changed=['primary_frozen.yaml']" in verification.errors[0]


def test_cycle_in_an_internally_resealed_forged_bundle_is_detected(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture)
    evidence = json.loads(result.amendment_evidence_path.read_text(encoding="utf-8"))
    evidence["parent"]["authority_directory"] = str(result.amendment_directory)
    _write_json(result.amendment_evidence_path, evidence)
    _reseal_amendment(result.amendment_directory)

    verification = verify_preregistration_amendment(result.amendment_directory)
    assert not verification.valid
    assert "cycle detected" in verification.errors[0]


def test_amendment_requires_an_actual_change(amendment_fixture: AmendmentFixture) -> None:
    fixture = amendment_fixture
    with pytest.raises(ValueError, match="actual frozen change"):
        _create(fixture)
    assert not fixture.amendment_root.exists() or not any(fixture.amendment_root.iterdir())


def test_rejecting_output_inside_parent_does_not_mutate_base(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    base_before = sha256_path(fixture.base)
    forbidden_root = fixture.base / "amendments"
    _change_inputs(fixture, 2)

    with pytest.raises(ValueError, match="immutable parent"):
        _create(fixture, amendment_root=forbidden_root)

    assert sha256_path(fixture.base) == base_before
    assert not forbidden_root.exists()
    assert verify_preregistration_freeze(fixture.base).valid


def test_resealed_manifest_cannot_change_execution_exclusion_policy(
    amendment_fixture: AmendmentFixture,
) -> None:
    fixture = amendment_fixture
    _change_inputs(fixture, 2)
    result = _create(fixture)
    source_manifest = json.loads(result.source_tree_manifest_path.read_text(encoding="utf-8"))
    source_manifest["excluded_paths"] = []
    _write_json(result.source_tree_manifest_path, source_manifest)
    _reseal_amendment(result.amendment_directory)

    verification = verify_preregistration_amendment(result.amendment_directory)

    assert not verification.valid
    assert "unexpected excluded paths" in verification.errors[0]
