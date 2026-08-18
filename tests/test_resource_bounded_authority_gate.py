from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.utils.run_tracking as run_tracking
import histo_audit.workflows.lifecycle_qualification as lifecycle_qualification
import histo_audit.workflows.preregistration_amendment as amendment
import histo_audit.workflows.study_gates as study_gates
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.utils.run_tracking import IntegrityVerification, RunStageEligibilityReceipt
from histo_audit.workflows.preregistration_amendment import (
    ResourceBoundedConfirmatoryAuthorization,
    require_resource_bounded_confirmatory_authorization,
)
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    HistoricalPrimaryDependencyEvidence,
    PrimaryExecutionGateEvidence,
    validate_historical_primary_dependency,
    validate_resource_bounded_execution_gate,
)

_DISPOSITION = "amended_or_exploratory"
_RUN_ID = "sealed-recovery"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: str(record["path"]))
    return {
        "schema_version": 3,
        "scope_kind": "execution_source",
        "scope": list(amendment._EXECUTION_SCOPE),
        "excluded_roots": list(amendment._EXECUTION_EXCLUDED_ROOTS),
        "excluded_paths": list(amendment._EXECUTION_EXCLUDED_PATHS),
        "artifact_count": len(ordered),
        "root_sha256": amendment._canonical_root(ordered),
        "artifacts": ordered,
    }


def _closed_source_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    parent_records: list[dict[str, Any]] = []
    resource_records: list[dict[str, Any]] = []
    for path, change_kind in amendment._RESOURCE_BOUNDED_SOURCE_DELTA_KINDS.items():
        if change_kind != "added":
            parent_records.append(
                {
                    "path": path,
                    "size_bytes": 10,
                    "sha256": _digest(f"parent:{path}"),
                }
            )
        if change_kind != "removed":
            resource_records.append(
                {
                    "path": path,
                    "size_bytes": 20,
                    "sha256": _digest(f"resource:{path}"),
                }
            )
    return _source_manifest(parent_records), _source_manifest(resource_records)


def _primary_gate(tmp_path: Path, *, authority: Path | None = None) -> PrimaryExecutionGateEvidence:
    recovery_authority = (
        authority.resolve()
        if authority is not None
        else (tmp_path / "recovery-authority").resolve()
    )
    recovery_authority.mkdir(parents=True, exist_ok=True)
    base = (tmp_path / "base-freeze").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return PrimaryExecutionGateEvidence(
        freeze_directory=recovery_authority,
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256="6" * 64,
        confirmatory_config_semantic_sha256="7" * 64,
        primary_matrix_cell_count=222,
        primary_required_cell_count=185,
        confirmatory_matrix_cell_count=108,
        pilot_run_id="pilot",
        pilot_artifact_root_sha256="8" * 64,
        dataset_sha256="9" * 64,
        manifest_sha256="a" * 64,
        duplicate_audit_sha256="b" * 64,
        pathology_encoder_audit_sha256="c" * 64,
        source_tree_root_sha256="d" * 64,
        base_freeze_directory=base,
        registration_authority_kind="preregistration_amendment",
        registration_status=_DISPOSITION,
        registration_authority_chain_depth=1,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=False,
    )


def _receipt(run: Path) -> RunStageEligibilityReceipt:
    record = {
        "run_id": run.name,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
    }
    receipt = RunStageEligibilityReceipt(
        run_directory=run,
        run_id=run.name,
        completion_stage="PRIMARY_STUDY_COMPLETE",
        record_sha256="e" * 64,
        verification_sha256="f" * 64,
        _canonical_record_json=json.dumps(record, sort_keys=True),
    )
    object.__setattr__(
        receipt,
        "_attestation",
        run_tracking._RUN_STAGE_ELIGIBILITY_RECEIPT_TOKEN,
    )
    return receipt


def _historical_primary(
    tmp_path: Path,
    *,
    gate: PrimaryExecutionGateEvidence | None = None,
) -> HistoricalPrimaryDependencyEvidence:
    primary_gate = gate or _primary_gate(tmp_path)
    run = (tmp_path / "runs" / _RUN_ID).resolve()
    run.mkdir(parents=True, exist_ok=True)
    return HistoricalPrimaryDependencyEvidence(
        primary_gate=primary_gate,
        primary_run_directory=run,
        primary_run_id=run.name,
        primary_artifact_root_sha256="0" * 64,
        primary_artifact_manifest_sha256="1" * 64,
        primary_completion_evidence_sha256="2" * 64,
        primary_execution_gate_sha256="3" * 64,
        primary_reconciliation_sha256="4" * 64,
        completed_required_cell_count=185,
        primary_statistics_manifest_sha256="5" * 64,
        primary_statistics_sha256="6" * 64,
        primary_bootstrap_evidence_sha256="7" * 64,
        primary_subgroups_sha256="8" * 64,
        primary_statistics_source_readback_root_sha256="9" * 64,
        primary_statistics_comparison_count=36,
        primary_stage_attestation_record_sha256="a" * 64,
        primary_stage_attestation_verification_sha256="b" * 64,
        primary_restoration_readback_root_sha256="c" * 64,
        primary_recovery_evidence_sha256="d" * 64,
        primary_recovery_authorization_sha256="e" * 64,
        primary_recovery_source_run_id="interrupted-primary",
        primary_recovery_source_snapshot_root_sha256="f" * 64,
        primary_recovery_analysis_disposition=_DISPOSITION,
    )


def _schema_v4_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    parent = (tmp_path / "recovery-authority").resolve()
    parent.mkdir()
    child = (tmp_path / "resource-authority").resolve()
    child.mkdir()
    run = (tmp_path / "runs" / _RUN_ID).resolve()
    recovery_authorization = {"schema_version": 1, "sealed": True}
    recovery_authorization_sha256 = amendment._canonical_mapping_sha256(recovery_authorization)
    source_delta_records = (
        {
            "path": "src/histo_audit/workflows/study_gates.py",
            "change_kind": "modified",
            "before": {"size_bytes": 1, "sha256": "1" * 64},
            "after": {"size_bytes": 2, "sha256": "2" * 64},
        },
    )
    source_delta_sha256 = amendment._canonical_value_sha256(list(source_delta_records))
    parent_state = amendment._AuthorityState(
        directory=parent,
        kind="preregistration_amendment",
        timestamp_utc="2026-07-27T12:00:00.000000Z",
        timestamp=datetime(2026, 7, 27, 12, tzinfo=UTC),
        chain_depth=1,
        artifact_root_sha256="3" * 64,
        sha256_manifest_sha256="4" * 64,
        snapshot_hashes={
            "confirmatory_config": {
                "file_sha256": "5" * 64,
                "semantic_sha256": "6" * 64,
            },
            "execution_source": {
                "root_sha256": "7" * 64,
                "manifest_sha256": "8" * 64,
            },
        },
        parent_directory=(tmp_path / "base-freeze").resolve(),
    )
    resource_source = {"root_sha256": "9" * 64}
    _write_json(parent / "source_tree_manifest.json", {"parent": True})
    _write_json(child / "source_tree_manifest.json", resource_source)
    authorization = ResourceBoundedConfirmatoryAuthorization(
        primary_run_id=run.name,
        primary_run_directory=run,
        primary_artifact_root_sha256="a" * 64,
        primary_artifact_manifest_sha256="b" * 64,
        primary_completion_evidence_sha256="c" * 64,
        primary_execution_gate_sha256="d" * 64,
        primary_stage_attestation_record_sha256="e" * 64,
        primary_stage_attestation_verification_sha256="f" * 64,
        primary_recovery_evidence_sha256="0" * 64,
        primary_recovery_authorization_sha256=recovery_authorization_sha256,
        recovery_authority_directory=parent,
        recovery_authority_artifact_root_sha256=parent_state.artifact_root_sha256,
        recovery_authority_manifest_sha256=parent_state.sha256_manifest_sha256,
        recovery_authority_chain_depth=parent_state.chain_depth,
        resource_profile_id=amendment._RESOURCE_BOUNDED_PROFILE_ID,
        parent_confirmatory_config_file_sha256="5" * 64,
        parent_confirmatory_config_semantic_sha256="6" * 64,
        resource_confirmatory_config_file_sha256="1" * 64,
        resource_confirmatory_config_semantic_sha256=(
            amendment._RESOURCE_BOUNDED_CONFIG_SEMANTIC_SHA256
        ),
        parent_execution_source_root_sha256="7" * 64,
        parent_execution_source_manifest_sha256="8" * 64,
        resource_execution_source_root_sha256="9" * 64,
        resource_execution_source_manifest_sha256="2" * 64,
        source_delta_records=source_delta_records,
        source_delta_sha256=source_delta_sha256,
    ).as_dict()
    evidence = {
        "schema_version": 4,
        "amendment_purpose": (amendment.RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE),
        "parent": {"authority_directory": str(parent)},
        "resource_bounded_confirmatory_authorization": authorization,
        "confirmatory_storage_policy": amendment.ConfirmatoryStoragePolicy().as_dict(),
    }
    _write_json(child / "amendment_evidence.json", evidence)
    monkeypatch.setattr(
        amendment,
        "verify_preregistration_amendment",
        lambda *_args, **_kwargs: SimpleNamespace(valid=True, errors=()),
    )
    monkeypatch.setattr(amendment, "_authority_state", lambda *_args, **_kwargs: parent_state)
    monkeypatch.setattr(
        amendment,
        "_snapshot_hashes",
        lambda _directory: {
            "confirmatory_config": {
                "file_sha256": "1" * 64,
                "semantic_sha256": amendment._RESOURCE_BOUNDED_CONFIG_SEMANTIC_SHA256,
            },
            "execution_source": {
                "root_sha256": "9" * 64,
                "manifest_sha256": "2" * 64,
            },
        },
    )
    monkeypatch.setattr(
        amendment,
        "_resource_parent_recovery_authorization",
        lambda _state: recovery_authorization,
    )
    monkeypatch.setattr(
        amendment,
        "_canonical_resource_source_delta",
        lambda _parent, _resource: (source_delta_records, source_delta_sha256),
    )
    monkeypatch.setattr(
        amendment,
        "_validate_live_resource_primary_binding",
        lambda *_args, **_kwargs: None,
    )
    return child, evidence


def test_closed_resource_source_delta_accepts_exact_allowlist() -> None:
    parent, resource = _closed_source_pair()

    delta, delta_sha256 = amendment._canonical_resource_source_delta(parent, resource)

    assert {record["path"]: record["change_kind"] for record in delta} == (
        amendment._RESOURCE_BOUNDED_SOURCE_DELTA_KINDS
    )
    assert delta_sha256 == amendment._canonical_value_sha256(list(delta))


def test_real_parent_live_source_delta_matches_independent_registered_contract() -> None:
    """Regress the real P->live delta without deriving expectations from production."""

    project_root = Path(__file__).resolve().parents[1]
    parent_manifest_path = (
        project_root
        / "artifacts"
        / "preregistration_amendments"
        / "20260727T133947.089370Z"
        / "source_tree_manifest.json"
    )
    if not parent_manifest_path.is_file():
        pytest.skip("real recovery authority P is unavailable in this checkout")
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    assert parent["root_sha256"] == (
        "ba7fb4c8336c4f9ba138fcda16019dc31bec7e5cc3e8b846e643d6dd0332601b"
    )
    assert run_tracking.sha256_file(parent_manifest_path) == (
        "0f5e33259962c5b8f2bf5e3c11a776bdfbf280f6e3cded906e1222e89e1a4df2"
    )
    current = run_tracking.capture_source_tree(project_root)
    parent_by_path = {record["path"]: record for record in parent["artifacts"]}
    current_by_path = {record["path"]: record for record in current["artifacts"]}
    observed: dict[str, str] = {}
    for path in sorted(set(parent_by_path).union(current_by_path)):
        before = parent_by_path.get(path)
        after = current_by_path.get(path)
        if before == after:
            continue
        observed[path] = "added" if before is None else "removed" if after is None else "modified"
        expected = {
            "configs/confirmatory_resource_bounded_amended.yaml": "added",
            "pyproject.toml": "modified",
            "src/histo_audit/cli.py": "modified",
            "src/histo_audit/cross_validation/image_oof.py": "modified",
            "src/histo_audit/experiment/__init__.py": "modified",
            "src/histo_audit/experiment/confirmatory_cli_inputs.py": "added",
            "src/histo_audit/experiment/confirmatory_completion.py": "modified",
            "src/histo_audit/experiment/confirmatory_core.py": "modified",
            "src/histo_audit/experiment/confirmatory_memory_workspace.py": "added",
            "src/histo_audit/experiment/confirmatory_runner.py": "modified",
            "src/histo_audit/experiment/original_confirmatory_preflight.py": "added",
            "src/histo_audit/experiment/original_confirmatory_resume.py": "added",
            "src/histo_audit/experiment/original_confirmatory_runner_core.py": "added",
            "src/histo_audit/experiment/pannuke_confirmatory_inputs.py": "modified",
            "src/histo_audit/experiment/resource_bounded_checkpoint_execution.py": "added",
            "src/histo_audit/experiment/resource_bounded_resume.py": "added",
            "src/histo_audit/experiment/resource_bounded_runner.py": "added",
            "src/histo_audit/experiment/study_contracts.py": "modified",
            "src/histo_audit/models/cnn.py": "modified",
            "src/histo_audit/mvp_demo.py": "added",
            "src/histo_audit/pannuke/publication.py": "modified",
            "src/histo_audit/workflows/__init__.py": "modified",
            "src/histo_audit/workflows/lifecycle_qualification.py": "added",
            "src/histo_audit/workflows/original_confirmatory_capsule_authority.py": "added",
            "src/histo_audit/workflows/original_confirmatory_capsule_entry.py": "added",
            "src/histo_audit/workflows/original_confirmatory_capsule_terminal.py": "added",
            "src/histo_audit/workflows/original_confirmatory_technical_authority_publication_v1.py": (
                "added"
            ),
            "src/histo_audit/workflows/original_confirmatory_technical_authority_review_producer_v1.py": (
                "added"
            ),
            "src/histo_audit/workflows/original_confirmatory_technical_authority_v1.py": "added",
            "src/histo_audit/workflows/preregistration_amendment.py": "modified",
            "src/histo_audit/workflows/resource_authority_d_replacement_controller.py": "added",
            "src/histo_audit/workflows/resource_authority_d_replacement_v2_controller.py": "added",
            "src/histo_audit/workflows/study_gates.py": "modified",
        }
    assert observed == expected
    delta, _ = amendment._canonical_source_delta_with_allowlist(
        parent,
        current,
        allowlisted_change_kinds=expected,
        role="live resource integration",
    )
    assert {record["path"]: record["change_kind"] for record in delta} == expected


@pytest.mark.parametrize(
    ("tamper_kind", "message"),
    [
        ("unknown", r"unknown=\['src/unregistered.py'\]"),
        ("missing", r"missing=\['src/histo_audit/cli.py'\]"),
        (
            "wrong_kind",
            r"wrong_kind=\['configs/confirmatory_resource_bounded_amended.yaml'\]",
        ),
    ],
)
def test_closed_resource_source_delta_rejects_unknown_missing_and_wrong_kind(
    tamper_kind: str,
    message: str,
) -> None:
    parent, resource = _closed_source_pair()
    parent_records = list(parent["artifacts"])
    resource_records = list(resource["artifacts"])
    if tamper_kind == "unknown":
        resource_records.append(
            {
                "path": "src/unregistered.py",
                "size_bytes": 1,
                "sha256": _digest("unknown"),
            }
        )
    elif tamper_kind == "missing":
        path = "src/histo_audit/cli.py"
        parent_record = next(record for record in parent_records if record["path"] == path)
        resource_records = [
            dict(parent_record) if record["path"] == path else record for record in resource_records
        ]
    else:
        path = "configs/confirmatory_resource_bounded_amended.yaml"
        resource_records = [record for record in resource_records if record["path"] != path]
        parent_records.append(
            {
                "path": path,
                "size_bytes": 1,
                "sha256": _digest("wrong-direction"),
            }
        )

    with pytest.raises(ValueError, match=message):
        amendment._canonical_resource_source_delta(
            _source_manifest(parent_records),
            _source_manifest(resource_records),
        )


def test_schema_v4_authority_accepts_only_exact_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _ = _schema_v4_fixture(tmp_path, monkeypatch)

    canonical = require_resource_bounded_confirmatory_authorization(authority)

    assert canonical["resource_capacity_policy"] == amendment._RESOURCE_BOUNDED_CAPACITY
    evidence_path = authority / "amendment_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["resource_bounded_confirmatory_authorization"]["resource_capacity_policy"][
        "planned_required_cells"
    ] += 1
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="capacity policy differs"):
        require_resource_bounded_confirmatory_authorization(authority)


def test_historical_c_storage_readback_survives_d_while_execution_c_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_c, evidence = _schema_v4_fixture(tmp_path, monkeypatch)
    successor_d = (tmp_path / "resource-successor-d").resolve()
    successor_d.mkdir()
    _write_json(
        successor_d / "amendment_evidence.json",
        {
            "schema_version": 5,
            "amendment_purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
            "resource_bounded_technical_successor_authorization": {},
        },
    )
    _write_json(
        successor_d / ".immutable.json",
        {
            "schema_version": 1,
            "status": "amended",
            "authority_kind": "preregistration_amendment",
        },
    )

    historical = amendment._require_historical_resource_bounded_confirmatory_storage_policy(
        authority_c
    )

    assert historical == evidence["confirmatory_storage_policy"]
    with pytest.raises(ValueError, match="historically valid but no longer the effective"):
        amendment.require_confirmatory_storage_policy(authority_c)


def test_historical_c_storage_readback_rejects_tampered_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_c, evidence = _schema_v4_fixture(tmp_path, monkeypatch)
    evidence["confirmatory_storage_policy"]["retained_copy_count"] = 2
    _write_json(authority_c / "amendment_evidence.json", evidence)

    with pytest.raises(
        ValueError,
        match="confirmatory storage policy field 'retained_copy_count' is invalid",
    ):
        amendment._require_historical_resource_bounded_confirmatory_storage_policy(authority_c)


def test_historical_c_storage_readback_rejects_evidence_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_c, _ = _schema_v4_fixture(tmp_path, monkeypatch)
    evidence_path = authority_c / "amendment_evidence.json"

    def mutate_evidence_during_sealed_validation(_directory: str | Path) -> dict[str, Any]:
        changed = json.loads(evidence_path.read_text(encoding="utf-8"))
        changed["toctou_marker"] = "changed-after-initial-read"
        _write_json(evidence_path, changed)
        return changed["resource_bounded_confirmatory_authorization"]

    monkeypatch.setattr(
        amendment,
        "_require_sealed_resource_bounded_confirmatory_authorization",
        mutate_evidence_during_sealed_validation,
    )

    with pytest.raises(ValueError, match="changed during storage-policy readback"):
        amendment._require_historical_resource_bounded_confirmatory_storage_policy(authority_c)


def test_partial_d_candidate_blocks_c_execution_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_c, _ = _schema_v4_fixture(tmp_path, monkeypatch)
    partial_d = (tmp_path / "partial-resource-successor-d").resolve()
    partial_d.mkdir()
    _write_json(
        partial_d / "amendment_evidence.json",
        {
            "schema_version": 5,
            "amendment_purpose": amendment.RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        },
    )

    with pytest.raises(ValueError, match="no longer the effective execution leaf"):
        amendment.require_confirmatory_storage_policy(authority_c)
    with pytest.raises(ValueError, match="lacks its direct parent or authorization"):
        amendment.require_resource_bounded_technical_successor_authorization(partial_d)


def test_execution_modules_never_import_historical_c_storage_reader() -> None:
    project_root = Path(__file__).resolve().parents[1]
    forbidden = "_require_historical_resource_bounded_confirmatory_storage_policy"
    authority_module = (
        project_root / "src" / "histo_audit" / "workflows" / "preregistration_amendment.py"
    )

    for module in (project_root / "src" / "histo_audit").rglob("*.py"):
        if module == authority_module:
            continue
        assert forbidden not in module.read_text(encoding="utf-8")


def test_resource_and_lifecycle_gates_bind_the_effective_authority_reader() -> None:
    effective = amendment._require_sealed_effective_resource_bounded_confirmatory_authorization

    assert study_gates._require_sealed_resource_bounded_confirmatory_authorization is effective
    assert (
        lifecycle_qualification._require_sealed_effective_resource_bounded_confirmatory_authorization
        is effective
    )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("outcomes_inspected", False),
        ("analysis_disposition", "confirmatory"),
        ("outcome_use_policy", "outcomes_used_for_selection"),
        ("original_confirmatory_claim_allowed", True),
        ("study_outcome_eligible", True),
        ("completion_stage", "CONFIRMATORY_COMPLETE"),
        ("primary_rebinding_allowed", True),
        ("primary_mutation_allowed", True),
    ],
)
def test_schema_v4_authority_rejects_tampered_fixed_outcome_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered: Any,
) -> None:
    authority, evidence = _schema_v4_fixture(tmp_path, monkeypatch)
    evidence["resource_bounded_confirmatory_authorization"][field] = tampered
    _write_json(authority / "amendment_evidence.json", evidence)

    with pytest.raises(ValueError, match="fixed post-outcome exploratory policy"):
        require_resource_bounded_confirmatory_authorization(authority)


def test_historical_gate_never_calls_public_primary_gate_or_live_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _primary_gate(tmp_path)
    run = (tmp_path / "runs" / _RUN_ID).resolve()
    run.mkdir(parents=True)
    receipt = _receipt(run)
    completion = {
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "study_outcome_eligible": True,
        "artifact_scope": "real_pannuke_primary_study",
        "required_cell_count": 185,
        "completed_required_cell_count": 185,
        "failed_required_cell_count": 0,
        "freeze_artifact_root_sha256": gate.freeze_artifact_root_sha256,
        "frozen_primary_config_sha256": gate.frozen_primary_config_sha256,
        "frozen_confirmatory_config_sha256": gate.frozen_confirmatory_config_sha256,
        "dataset_sha256": gate.dataset_sha256,
        "manifest_sha256": gate.manifest_sha256,
        "duplicate_audit_sha256": gate.duplicate_audit_sha256,
        "pathology_encoder_audit_sha256": gate.pathology_encoder_audit_sha256,
        "source_tree_root_sha256": gate.source_tree_root_sha256,
    }
    _write_json(
        run / "status.json",
        {"experiment_name": study_gates.PRIMARY_RECOVERY_EXPERIMENT_NAME},
    )
    _write_json(run / "completion_evidence.json", completion)
    _write_json(run / "reconciliation.json", {"status": "passed"})
    _write_json(
        run / "metrics.json",
        {
            "artifact_scope": "real_pannuke_primary_study",
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "study_outcome_eligible": True,
        },
    )
    for name in (
        "matrix_plan.json",
        "report.md",
        ".immutable.json",
        "artifact_manifest.json",
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
        "primary_recovery_evidence.json",
    ):
        path = run / name
        if not path.exists():
            path.write_bytes(b"fixture")
    integrity = IntegrityVerification(
        valid=True,
        run_id=run.name,
        expected_root_sha256="0" * 64,
        actual_root_sha256="0" * 64,
        missing_paths=(),
        added_paths=(),
        changed_paths=(),
        registry_record_present=True,
        errors=(),
    )
    recovery = SimpleNamespace(
        evidence_sha256="d" * 64,
        authorization_sha256="e" * 64,
        source_run_id="interrupted-primary",
        source_snapshot_root_sha256="f" * 64,
        analysis_disposition=_DISPOSITION,
    )
    statistics = SimpleNamespace(
        manifest_sha256="5" * 64,
        statistics_sha256="6" * 64,
        bootstrap_evidence_sha256="7" * 64,
        subgroups_sha256="8" * 64,
        source_readback_root_sha256="9" * 64,
        comparison_count=36,
        stage_attestation_record_sha256=receipt.record_sha256,
        stage_attestation_verification_sha256=receipt.verification_sha256,
    )
    monkeypatch.setattr(study_gates, "verify_run_integrity", lambda _run: integrity)
    monkeypatch.setattr(
        study_gates,
        "require_run_stage_eligibility_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        study_gates,
        "_validate_registered_primary_dependencies",
        lambda **_kwargs: gate,
    )
    monkeypatch.setattr(
        study_gates,
        "_validate_primary_recovery_downstream_identity",
        lambda **_kwargs: recovery,
    )
    monkeypatch.setattr(
        study_gates,
        "_validate_primary_completion_attestations",
        lambda **_kwargs: (
            SimpleNamespace(),
            statistics,
            SimpleNamespace(readback_root_sha256="c" * 64),
        ),
    )
    monkeypatch.setattr(
        study_gates,
        "_validate_primary_finalization_successor",
        lambda **_kwargs: (False, None, None, None),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("historical P consulted a live/public primary source gate")

    monkeypatch.setattr(study_gates, "validate_primary_execution_gate", forbidden)
    monkeypatch.setattr(study_gates, "capture_source_tree", forbidden)

    result = validate_historical_primary_dependency(
        primary_run_directory=run,
        project_root=tmp_path,
        recovery_authority_directory=gate.freeze_directory,
        dataset_path=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest",
        duplicate_audit_path=tmp_path / "duplicates",
        pathology_encoder_audit_path=tmp_path / "pathology",
    )

    assert result.primary_gate is gate
    assert result.primary_run_id == run.name
    assert result.primary_recovery_authorization_sha256 == recovery.authorization_sha256


def _patch_resource_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_matches: bool,
    historical_tamper: bool = False,
) -> tuple[Path, HistoricalPrimaryDependencyEvidence]:
    historical_authority = (tmp_path / "authority-p").resolve()
    gate = _primary_gate(tmp_path, authority=historical_authority)
    historical = _historical_primary(tmp_path, gate=gate)
    resource_authority = (tmp_path / "authority-c").resolve()
    resource_authority.mkdir()
    for directory in (historical_authority, resource_authority):
        (directory / "PRE_REGISTRATION_FROZEN.md").write_text(
            "frozen preregistration\n", encoding="utf-8"
        )
        (directory / "primary_frozen.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    config_path = resource_authority / "confirmatory_frozen.yaml"
    config_path.write_text("resource: bounded\n", encoding="utf-8")
    source_path = resource_authority / "source_tree_manifest.json"
    _write_json(source_path, {"fixture": "authority-c"})
    resource_config = {
        "execution_profile": "resource_bounded_confirmatory_v1",
        "analysis_disposition": _DISPOSITION,
        "original_confirmatory_claim_allowed": False,
        "completion_stage": None,
    }
    child_source = {
        "schema_version": 3,
        "root_sha256": "1" * 64,
        "artifacts": [],
    }
    historical_record = {
        "run_id": historical.primary_run_id,
        "run_directory": str(historical.primary_run_directory),
        "artifact_root_sha256": historical.primary_artifact_root_sha256,
        "artifact_manifest_sha256": historical.primary_artifact_manifest_sha256,
        "completion_evidence_sha256": historical.primary_completion_evidence_sha256,
        "primary_execution_gate_sha256": historical.primary_execution_gate_sha256,
        "stage_attestation_record_sha256": (historical.primary_stage_attestation_record_sha256),
        "stage_attestation_verification_sha256": (
            historical.primary_stage_attestation_verification_sha256
        ),
        "recovery_evidence_sha256": historical.primary_recovery_evidence_sha256,
        "recovery_authorization_sha256": (historical.primary_recovery_authorization_sha256),
        "registration_authority": {
            "directory": str(gate.freeze_directory),
            "kind": "preregistration_amendment",
            "artifact_root_sha256": gate.freeze_artifact_root_sha256,
            "sha256_manifest_sha256": gate.freeze_manifest_sha256,
            "chain_depth": gate.registration_authority_chain_depth,
        },
    }
    if historical_tamper:
        historical_record["artifact_root_sha256"] = "f" * 64
    authorization = {
        "historical_primary": historical_record,
        "resource_profile": {
            "profile_id": "resource_bounded_confirmatory_v1",
            "resource_confirmatory_config_file_sha256": study_gates.sha256_file(config_path),
            "resource_confirmatory_config_semantic_sha256": (
                study_gates.config_sha256(resource_config)
            ),
        },
        "execution_source_delta": {
            "resource_root_sha256": child_source["root_sha256"],
            "resource_manifest_sha256": study_gates.sha256_file(source_path),
            "delta_sha256": "2" * 64,
        },
        "resource_capacity_policy": {
            "schema_version": 1,
            "planned_required_cells": 24,
        },
    }
    _write_json(
        resource_authority / "amendment_evidence.json",
        {"confirmatory_storage_policy": (amendment.ConfirmatoryStoragePolicy().as_dict())},
    )
    monkeypatch.setattr(
        study_gates,
        "_require_sealed_resource_bounded_confirmatory_authorization",
        lambda _directory: authorization,
    )
    monkeypatch.setattr(
        study_gates,
        "verify_preregistration_amendment",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=True,
            artifact_root_sha256="3" * 64,
            sha256_manifest_sha256="4" * 64,
            chain_depth=2,
            errors=(),
        ),
    )
    monkeypatch.setattr(
        study_gates,
        "validate_historical_primary_dependency",
        lambda **_kwargs: historical,
    )
    monkeypatch.setattr(study_gates, "load_config", lambda _path: resource_config)
    monkeypatch.setattr(
        study_gates,
        "_validate_execution_source_manifest",
        lambda _path, _role: child_source,
    )
    monkeypatch.setattr(
        study_gates,
        "capture_source_tree",
        lambda _root: child_source if source_matches else {"root_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        study_gates,
        "require_confirmatory_storage_policy",
        lambda _directory: {"policy": "single_canonical_checkpoint_copy_v1"},
    )
    return resource_authority, historical


def test_resource_gate_binds_exact_historical_p_and_current_c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_authority, historical = _patch_resource_gate(
        tmp_path,
        monkeypatch,
        source_matches=True,
    )

    result = validate_resource_bounded_execution_gate(
        primary_run_directory=historical.primary_run_directory,
        project_root=tmp_path,
        resource_authority_directory=resource_authority,
        dataset_path=tmp_path / "dataset",
        manifest_path=tmp_path / "manifest",
        duplicate_audit_path=tmp_path / "duplicates",
        pathology_encoder_audit_path=tmp_path / "pathology",
    )

    assert result.historical_primary is historical
    assert result.execution_authority.authority_directory == resource_authority
    assert result.execution_authority.resource_execution_source_root_sha256 == "1" * 64
    assert result.outcomes_inspected is True
    assert result.analysis_disposition == _DISPOSITION
    assert result.original_confirmatory_claim_allowed is False
    assert result.study_outcome_eligible is False
    assert result.completion_stage is None
    assert result.primary_rebinding_allowed is False
    assert result.primary_mutation_allowed is False


def test_resource_gate_rejects_current_source_mismatch_against_c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_authority, historical = _patch_resource_gate(
        tmp_path,
        monkeypatch,
        source_matches=False,
    )

    with pytest.raises(
        ValueError,
        match="current execution source differs from resource-bounded authority C",
    ):
        validate_resource_bounded_execution_gate(
            primary_run_directory=historical.primary_run_directory,
            project_root=tmp_path,
            resource_authority_directory=resource_authority,
            dataset_path=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest",
            duplicate_audit_path=tmp_path / "duplicates",
            pathology_encoder_audit_path=tmp_path / "pathology",
        )


def test_resource_gate_rejects_tampered_historical_p_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_authority, historical = _patch_resource_gate(
        tmp_path,
        monkeypatch,
        source_matches=True,
        historical_tamper=True,
    )

    with pytest.raises(ValueError, match="differs at artifact_root_sha256"):
        validate_resource_bounded_execution_gate(
            primary_run_directory=historical.primary_run_directory,
            project_root=tmp_path,
            resource_authority_directory=resource_authority,
            dataset_path=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest",
            duplicate_audit_path=tmp_path / "duplicates",
            pathology_encoder_audit_path=tmp_path / "pathology",
        )


def test_legacy_confirmatory_gate_serialization_remains_unchanged(
    tmp_path: Path,
) -> None:
    gate = _primary_gate(tmp_path)
    legacy = ConfirmatoryExecutionGateEvidence(
        primary_gate=gate,
        primary_run_directory=(tmp_path / "legacy-run").resolve(),
        primary_run_id="legacy-run",
        primary_artifact_root_sha256="1" * 64,
        primary_completion_evidence_sha256="2" * 64,
        primary_reconciliation_sha256="3" * 64,
        completed_required_cell_count=185,
    )

    payload = legacy.as_dict()

    assert set(payload) == {
        "primary_gate",
        "primary_run_directory",
        "primary_run_id",
        "primary_artifact_root_sha256",
        "primary_completion_evidence_sha256",
        "primary_reconciliation_sha256",
        "completed_required_cell_count",
        "primary_statistics_manifest_sha256",
        "primary_statistics_sha256",
        "primary_bootstrap_evidence_sha256",
        "primary_subgroups_sha256",
        "primary_statistics_source_readback_root_sha256",
        "primary_statistics_comparison_count",
        "primary_stage_attestation_record_sha256",
        "primary_stage_attestation_verification_sha256",
        "primary_restoration_readback_root_sha256",
        "primary_finalization_only_successor",
        "primary_finalization_successor_evidence_sha256",
        "primary_predecessor_run_id",
        "primary_predecessor_artifact_root_sha256",
        "confirmatory_storage_policy_sha256",
    }
    assert payload["primary_gate"] == gate.as_dict()
    assert "primary_orphan_recovery" not in payload
    assert all("resource_bounded" not in key for key in payload)
    assert canonical_sha256(payload) == canonical_sha256(legacy.as_dict())
