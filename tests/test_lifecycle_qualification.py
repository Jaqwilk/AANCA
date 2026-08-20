"""Focused production-boundary tests for the M8 lifecycle qualification."""

from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import histo_audit.workflows as workflows_api
from histo_audit.cli import app
from histo_audit.experiment.confirmatory_runner import (
    ConfirmatoryStudyRunnerError,
    execute_confirmatory_study,
)
from histo_audit.pannuke.publication import AnchoredPhysicalCopySession
from histo_audit.utils import run_tracking as run_tracking_module
from histo_audit.utils.run_tracking import (
    RUN_STAGE_ATTESTATION_ANCHOR_FILENAME,
    RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
    capture_source_tree,
    require_lifecycle_run_qualified,
    sha256_file,
    verify_run_integrity,
    withdraw_run_eligibility,
)
from histo_audit.workflows import lifecycle_qualification as lifecycle_module
from histo_audit.workflows.lifecycle_qualification import (
    LifecycleQualificationError,
    execute_lifecycle_rehearsal,
    require_current_lifecycle_readiness,
)
from histo_audit.workflows.original_confirmatory_technical_authority_v1 import (
    VerifiedOriginalConfirmatoryTechnicalAuthority,
)
from histo_audit.workflows.preregistration import verify_preregistration_freeze
from tests.cli_contracts import cli_options


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_root(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attestation_record_sha256(record: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attestation_anchor(
    records: list[dict[str, Any]],
    ledger_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ledger_filename": RUN_STAGE_ATTESTATION_REGISTRY_FILENAME,
        "chain_algorithm": "sha256(canonical-json-record-with-previous-head)",
        "record_count": len(records),
        "head_record_sha256": records[-1]["record_sha256"] if records else None,
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }


def _sealed_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _refresh_rehearsal_semantic_receipts(run_directory: Path) -> None:
    completion_path = run_directory / "completion_evidence.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["required_artifacts"] = [
        _sealed_file_record(run_directory / str(record["path"]), run_directory)
        for record in completion["required_artifacts"]
    ]
    _write_json(completion_path, completion)

    receipt_path = run_directory / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["published"] = [
        {
            **_sealed_file_record(run_directory / str(record["path"]), run_directory),
            "destination": str(run_directory / str(record["path"])),
            "destination_nlink": 1,
        }
        for record in receipt["published"]
    ]
    _write_json(receipt_path, receipt)


def _make_authority(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "src" / "qualification_fixture").mkdir(parents=True)
    (project / "src" / "qualification_fixture" / "module.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (project / "configs").mkdir()
    (project / "configs" / "fixture.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[project]\nname='lifecycle-fixture'\nversion='0.1.0'\n", encoding="utf-8"
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    authority = tmp_path / "authority"
    authority.mkdir()
    (authority / "PRE_REGISTRATION_FROZEN.md").write_text("# Frozen\n", encoding="utf-8")
    (authority / "primary_frozen.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (authority / "confirmatory_frozen.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    _write_json(authority / "source_tree_manifest.json", capture_source_tree(project))
    _write_json(
        authority / "freeze_evidence.json",
        {"schema_version": 2, "completion_stage_enabled": "PRE_REGISTRATION_FROZEN"},
    )
    excluded = {".immutable.json", "sha256_manifest.json"}
    records = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(authority.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in excluded
    ]
    root_sha256 = _canonical_root(records)
    timestamp = datetime(2026, 7, 21, 10, 0, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    _write_json(
        authority / "sha256_manifest.json",
        {
            "schema_version": 1,
            "freeze_timestamp_utc": timestamp,
            "artifact_count": len(records),
            "artifact_root_sha256": root_sha256,
            "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
            "excluded_paths": sorted(excluded),
            "artifacts": records,
        },
    )
    _write_json(
        authority / ".immutable.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "freeze_timestamp_utc": timestamp,
            "artifact_root_sha256": root_sha256,
            "sha256_manifest_sha256": sha256_file(authority / "sha256_manifest.json"),
            "amendment_only": True,
        },
    )
    assert verify_preregistration_freeze(authority).valid
    return project, authority


def _install_t0_verifier_double(
    *,
    project: Path,
    historical_authority: Path,
    technical_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[bool]]:
    historical = lifecycle_module._authority_binding(
        project.resolve(),
        historical_authority.resolve(),
    )
    technical_authority.mkdir()
    _write_json(
        technical_authority / ".immutable.json",
        {
            "schema_version": 1,
            "authority_kind": (lifecycle_module.ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND),
        },
    )
    authority = VerifiedOriginalConfirmatoryTechnicalAuthority(
        authority_directory=technical_authority.resolve(),
        chain_depth=int(historical["chain_depth"]) + 1,
        artifact_root_sha256="a" * 64,
        sha256_manifest_sha256="b" * 64,
        execution_source_manifest_sha256="c" * 64,
        execution_source_root_sha256=str(capture_source_tree(project)["root_sha256"]),
        parent_authority_directory=historical_authority.resolve(),
        parent_artifact_root_sha256=str(historical["artifact_root_sha256"]),
        parent_sha256_manifest_sha256=str(historical["sha256_manifest_sha256"]),
        technical_authorization_sha256="d" * 64,
        independent_review_receipt_sha256="e" * 64,
        immutable_marker_sha256="1" * 64,
        publication_attempt_sha256="2" * 64,
        publication_success_sha256="f" * 64,
    )
    verified = lifecycle_module.VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
        authority=authority,
        namespace_directory=technical_authority.resolve().parent,
        namespace_claim_sha256="9" * 64,
        review_attempt_claim_sha256="8" * 64,
    )
    verify_modes: list[bool] = []

    def verify(
        authority_directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> Any:
        assert Path(authority_directory).resolve() == technical_authority.resolve()
        assert project_root is not None
        assert Path(project_root).resolve() == project.resolve()
        assert type(verify_live) is bool
        verify_modes.append(verify_live)
        return verified

    monkeypatch.setattr(
        lifecycle_module,
        "verify_published_original_confirmatory_technical_authority_v1",
        verify,
    )
    return verified, verify_modes


def test_lifecycle_published_t0_binding_is_full_grouped_and_exact_parent_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical-authority"
    verified, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )

    context = lifecycle_module._require_original_confirmatory_authority_chain(
        project_root=project.resolve(),
        historical_authority_directory=historical.resolve(),
        technical_authority_directory=technical.resolve(),
        verify_live=True,
    )
    authority = context.authority_binding
    assert verify_modes == [True]
    assert authority["schema_version"] == 3
    assert authority["authority_kind"] == (
        lifecycle_module.ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND
    )
    assert authority["chain_depth"] == verified.authority.chain_depth
    assert authority["historical_parent_authority_directory"] == str(historical.resolve())
    nested = authority["published_technical_authority_lifecycle_binding"]
    assert nested == verified.lifecycle_binding()
    assert nested["namespace_directory"] == str(technical.resolve().parent)
    assert nested["namespace_claim_sha256"] == verified.namespace_claim_sha256
    assert nested["automatic_retry_allowed"] is False
    assert nested["adoption_allowed"] is False
    assert nested["cleanup_allowed"] is False
    assert nested["technical_authority"] == verified.authority.lifecycle_binding()
    assert context.pins.as_dict() == {
        "namespace_directory": str(technical.resolve().parent),
        "namespace_claim_sha256": verified.namespace_claim_sha256,
        "technical_authority_directory": str(technical.resolve()),
        "technical_authority_artifact_root_sha256": (verified.authority.artifact_root_sha256),
        "technical_authorization_sha256": (verified.authority.technical_authorization_sha256),
        "published_technical_authority_lifecycle_binding_sha256": (nested["binding_sha256"]),
    }

    with pytest.raises(LifecycleQualificationError, match="generic lifecycle path"):
        lifecycle_module._authority_binding(
            project.resolve(),
            technical.resolve(),
        )
    assert verify_modes == [True]
    assert "verify_original_confirmatory_technical_authority_v1" not in inspect.getsource(
        lifecycle_module
    )


def test_lifecycle_published_t0_chain_rejects_wrong_historical_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical-authority"
    _, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    supplied_parent = tmp_path / "alternate-valid-parent"
    shutil.copytree(historical, supplied_parent)
    assert verify_preregistration_freeze(supplied_parent).valid

    with pytest.raises(
        LifecycleQualificationError,
        match="historical parent differs",
    ):
        lifecycle_module._require_original_confirmatory_authority_chain(
            project_root=project.resolve(),
            historical_authority_directory=supplied_parent.resolve(),
            technical_authority_directory=technical.resolve(),
            verify_live=True,
        )
    assert verify_modes == [True]


def test_strict_published_t0_entrypoints_own_exactly_one_live_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    readiness = tmp_path / "readiness"
    rehearsal = tmp_path / "rehearsal"
    verified, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    rehearsal_result = lifecycle_module.LifecycleRehearsalResult(
        run_directory=rehearsal,
        run_id="rehearsal",
        artifact_root_sha256="1" * 64,
        completion_evidence_sha256="2" * 64,
        config_semantic_sha256="3" * 64,
        plan_sha256="4" * 64,
        authority_binding_sha256="5" * 64,
    )
    readiness_result = lifecycle_module.LifecycleReadinessResult(
        readiness_run_directory=readiness,
        readiness_run_id="readiness",
        rehearsal_run_directory=rehearsal,
        artifact_root_sha256="6" * 64,
        qualification_binding_sha256="7" * 64,
        readiness_record_sha256="8" * 64,
    )
    readiness_verification = lifecycle_module.LifecycleReadinessVerification(
        valid=True,
        readiness_run_directory=readiness.resolve(),
        rehearsal_run_directory=rehearsal.resolve(),
        qualification_binding_sha256="7" * 64,
        readiness_record_sha256="8" * 64,
        errors=(),
    )
    bound_calls: list[str] = []

    def execute_bound(**kwargs: Any) -> Any:
        bound_calls.append("execute")
        assert kwargs["authority"]["schema_version"] == 3
        return rehearsal_result

    def verify_bound(**kwargs: Any) -> Any:
        bound_calls.append("verify")
        assert kwargs["authority"]["schema_version"] == 3
        return readiness_result

    def require_bound(**kwargs: Any) -> Any:
        bound_calls.append("require")
        assert kwargs["authority"]["schema_version"] == 3
        return readiness_verification

    def forbidden_generic(**kwargs: Any) -> Any:
        raise AssertionError(f"strict path re-entered a generic lifecycle API: {kwargs}")

    monkeypatch.setattr(
        lifecycle_module,
        "_execute_lifecycle_rehearsal_with_authority",
        execute_bound,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "_verify_lifecycle_rehearsal_fresh_process_with_authority",
        verify_bound,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "_require_current_lifecycle_readiness_with_authority",
        require_bound,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "execute_lifecycle_rehearsal",
        forbidden_generic,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "verify_lifecycle_rehearsal_fresh_process",
        forbidden_generic,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "require_current_lifecycle_readiness",
        forbidden_generic,
    )

    assert (
        lifecycle_module.execute_original_confirmatory_lifecycle_rehearsal(
            project_root=project,
            historical_authority_directory=historical,
            technical_authority_directory=technical,
        )
        == rehearsal_result
    )
    assert verify_modes == [True, False]
    verify_modes.clear()

    assert (
        lifecycle_module.verify_original_confirmatory_lifecycle_rehearsal_fresh_process(
            project_root=project,
            historical_authority_directory=historical,
            technical_authority_directory=technical,
            rehearsal_run_directory=rehearsal,
        )
        == readiness_result
    )
    assert verify_modes == [True, False]
    verify_modes.clear()

    result = lifecycle_module.require_current_original_confirmatory_lifecycle_readiness(
        project_root=project,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        readiness_run_directory=readiness,
    )
    assert verify_modes == [True, False]
    assert bound_calls == ["execute", "verify", "require"]
    assert result.readiness is readiness_verification
    assert result.verified_published_technical_authority is verified
    assert result.verified_published_technical_authority.review_attempt_claim_sha256 == "8" * 64
    assert result.published_t0_pins.namespace_claim_sha256 == "9" * 64
    assert result.published_t0_pins.technical_authority_directory == technical.resolve()
    assert (
        result.as_dict()["verified_published_technical_authority"]["review_attempt_claim_sha256"]
        == "8" * 64
    )
    verify_modes.clear()
    repeated = lifecycle_module.require_current_original_confirmatory_lifecycle_readiness(
        project_root=project,
        historical_authority_directory=historical,
        technical_authority_directory=technical,
        readiness_run_directory=readiness,
    )
    assert repeated.published_t0_pins == result.published_t0_pins
    assert verify_modes == [True, False]
    assert bound_calls == ["execute", "verify", "require", "require"]


def test_strict_readiness_api_is_closed_to_preverified_objects_pins_and_modes() -> None:
    expected_parameters = {
        lifecycle_module.execute_original_confirmatory_lifecycle_rehearsal: [
            "project_root",
            "historical_authority_directory",
            "technical_authority_directory",
            "runs_root",
            "run_id",
        ],
        lifecycle_module.verify_original_confirmatory_lifecycle_rehearsal_fresh_process: [
            "project_root",
            "historical_authority_directory",
            "technical_authority_directory",
            "rehearsal_run_directory",
            "runs_root",
        ],
        lifecycle_module.require_current_original_confirmatory_lifecycle_readiness: [
            "project_root",
            "historical_authority_directory",
            "technical_authority_directory",
            "readiness_run_directory",
        ],
    }
    forbidden = {
        "verify_live",
        "verified",
        "preverified",
        "published_t0_pins",
        "expected_technical_authority_artifact_root_sha256",
        "expected_technical_authorization_sha256",
    }
    for function, expected in expected_parameters.items():
        signature = inspect.signature(function)
        assert list(signature.parameters) == expected
        assert forbidden.isdisjoint(signature.parameters)


def test_private_bound_call_graph_has_one_combined_verifier_site_and_no_core_bypass() -> None:
    module_tree = ast.parse(inspect.getsource(lifecycle_module))
    called_names = [
        node.func.id
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert called_names.count("verify_published_original_confirmatory_technical_authority_v1") == 1
    assert "verify_original_confirmatory_technical_authority_v1" not in called_names

    private_bound_functions = (
        lifecycle_module._execute_lifecycle_rehearsal_with_authority,
        lifecycle_module._verify_rehearsal_read_only,
        lifecycle_module._verify_lifecycle_rehearsal_fresh_process_with_authority,
        lifecycle_module._require_current_lifecycle_readiness_with_authority,
    )
    for function in private_bound_functions:
        source = inspect.getsource(function)
        assert "verify_live" not in source
        assert "verify_published_original_confirmatory_technical_authority_v1" not in source
        assert "_authority_binding(" not in source


def test_closed_return_dataclasses_reject_internally_inconsistent_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    verified, _ = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    binding_sha256 = verified.lifecycle_binding()["binding_sha256"]
    readiness = lifecycle_module.LifecycleReadinessVerification(
        valid=True,
        readiness_run_directory=tmp_path / "readiness",
        rehearsal_run_directory=tmp_path / "rehearsal",
        qualification_binding_sha256="a" * 64,
        readiness_record_sha256="b" * 64,
        errors=(),
    )
    with pytest.raises(LifecycleQualificationError, match="pins are invalid"):
        lifecycle_module.PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins(
            namespace_directory=verified.namespace_directory,
            namespace_claim_sha256="not-a-sha256",
            technical_authority_directory=verified.authority.authority_directory,
            technical_authority_artifact_root_sha256=(verified.authority.artifact_root_sha256),
            technical_authorization_sha256=(verified.authority.technical_authorization_sha256),
            published_technical_authority_lifecycle_binding_sha256=binding_sha256,
        )
    individually_valid_but_mismatched = (
        lifecycle_module.PublishedOriginalConfirmatoryTechnicalAuthorityLifecyclePins(
            namespace_directory=verified.namespace_directory,
            namespace_claim_sha256="7" * 64,
            technical_authority_directory=verified.authority.authority_directory,
            technical_authority_artifact_root_sha256=(verified.authority.artifact_root_sha256),
            technical_authorization_sha256=(verified.authority.technical_authorization_sha256),
            published_technical_authority_lifecycle_binding_sha256=binding_sha256,
        )
    )
    with pytest.raises(LifecycleQualificationError, match="envelope is inconsistent"):
        lifecycle_module.OriginalConfirmatoryPublishedT0LifecycleReadinessVerification(
            readiness=readiness,
            verified_published_technical_authority=verified,
            published_t0_pins=individually_valid_but_mismatched,
        )


def test_strict_readiness_rejects_shallow_carrier_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    verified, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    changed = lifecycle_module.VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
        authority=verified.authority,
        namespace_directory=verified.namespace_directory,
        namespace_claim_sha256="8" * 64,
        review_attempt_claim_sha256=verified.review_attempt_claim_sha256,
    )

    def changing_verify(
        authority_directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> Any:
        assert Path(authority_directory).resolve() == technical.resolve()
        assert project_root is not None
        verify_modes.append(verify_live)
        return verified if verify_live else changed

    monkeypatch.setattr(
        lifecycle_module,
        "verify_published_original_confirmatory_technical_authority_v1",
        changing_verify,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "_require_current_lifecycle_readiness_with_authority",
        lambda **_: lifecycle_module.LifecycleReadinessVerification(
            valid=True,
            readiness_run_directory=tmp_path / "readiness",
            rehearsal_run_directory=tmp_path / "rehearsal",
            qualification_binding_sha256="a" * 64,
            readiness_record_sha256="b" * 64,
            errors=(),
        ),
    )

    with pytest.raises(LifecycleQualificationError, match="carrier changed"):
        lifecycle_module.require_current_original_confirmatory_lifecycle_readiness(
            project_root=project,
            historical_authority_directory=historical,
            technical_authority_directory=technical,
            readiness_run_directory=tmp_path / "readiness",
        )
    assert verify_modes[-2:] == [True, False]


def test_published_t0_carrier_rejects_resealed_nested_flat_pin_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    verified, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    original_lifecycle_binding = type(verified).lifecycle_binding

    def mismatched_binding(self: Any) -> dict[str, Any]:
        binding = original_lifecycle_binding(self)
        nested = dict(binding["technical_authority"])
        nested["artifact_root_sha256"] = "0" * 64
        nested_unsigned = {key: value for key, value in nested.items() if key != "binding_sha256"}
        nested["binding_sha256"] = lifecycle_module._canonical_sha256(nested_unsigned)
        binding["technical_authority"] = nested
        unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
        binding["binding_sha256"] = lifecycle_module._canonical_sha256(unsigned)
        return binding

    monkeypatch.setattr(type(verified), "lifecycle_binding", mismatched_binding)
    with pytest.raises(LifecycleQualificationError, match="carrier is invalid"):
        lifecycle_module._require_original_confirmatory_authority_chain(
            project_root=project.resolve(),
            historical_authority_directory=historical.resolve(),
            technical_authority_directory=technical.resolve(),
            verify_live=True,
        )
    assert verify_modes == [True]


@pytest.mark.parametrize(
    "tamper",
    [
        "namespace_directory",
        "namespace_claim",
        "review_attempt_claim",
        "binding_sha256",
    ],
)
def test_published_t0_carrier_rejects_malformed_namespace_and_composite_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    verified, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    candidate = verified
    if tamper == "namespace_directory":
        candidate = lifecycle_module.VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
            authority=verified.authority,
            namespace_directory=tmp_path / "wrong-namespace",
            namespace_claim_sha256=verified.namespace_claim_sha256,
            review_attempt_claim_sha256=verified.review_attempt_claim_sha256,
        )
    elif tamper == "namespace_claim":
        candidate = lifecycle_module.VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
            authority=verified.authority,
            namespace_directory=verified.namespace_directory,
            namespace_claim_sha256="not-a-sha256",
            review_attempt_claim_sha256=verified.review_attempt_claim_sha256,
        )
    elif tamper == "review_attempt_claim":
        candidate = lifecycle_module.VerifiedPublishedOriginalConfirmatoryTechnicalAuthorityV1(
            authority=verified.authority,
            namespace_directory=verified.namespace_directory,
            namespace_claim_sha256=verified.namespace_claim_sha256,
            review_attempt_claim_sha256="not-a-sha256",
        )
    else:
        original_lifecycle_binding = type(verified).lifecycle_binding

        def wrong_root(self: Any) -> dict[str, Any]:
            binding = original_lifecycle_binding(self)
            binding["binding_sha256"] = "0" * 64
            return binding

        monkeypatch.setattr(type(verified), "lifecycle_binding", wrong_root)

    def malformed_verify(
        authority_directory: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_live: bool = True,
    ) -> Any:
        assert Path(authority_directory).resolve() == technical.resolve()
        assert project_root is not None
        verify_modes.append(verify_live)
        return candidate

    monkeypatch.setattr(
        lifecycle_module,
        "verify_published_original_confirmatory_technical_authority_v1",
        malformed_verify,
    )
    with pytest.raises(LifecycleQualificationError, match="carrier is invalid"):
        lifecycle_module._require_original_confirmatory_authority_chain(
            project_root=project.resolve(),
            historical_authority_directory=historical.resolve(),
            technical_authority_directory=technical.resolve(),
            verify_live=True,
        )
    assert verify_modes[-1:] == [True]


def _generic_t0_rejection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[bool], list[dict[str, Any]], list[dict[str, Any]]]:
    project, historical = _make_authority(tmp_path)
    technical = tmp_path / "technical"
    _, verify_modes = _install_t0_verifier_double(
        project=project,
        historical_authority=historical,
        technical_authority=technical,
        monkeypatch=monkeypatch,
    )
    run_starts: list[dict[str, Any]] = []
    registry_writes: list[dict[str, Any]] = []

    def forbidden_start(**kwargs: Any) -> Any:
        run_starts.append(kwargs)
        raise AssertionError("generic T0 rejection occurred after RunTracker.start")

    def forbidden_registry_write(*args: Any, **kwargs: Any) -> Any:
        registry_writes.append({"args": args, "kwargs": kwargs})
        raise AssertionError("generic T0 rejection occurred after a registry write")

    monkeypatch.setattr(lifecycle_module.RunTracker, "start", forbidden_start)
    monkeypatch.setattr(
        lifecycle_module,
        "append_registry_row",
        forbidden_registry_write,
    )
    return project, technical, verify_modes, run_starts, registry_writes


def test_generic_direct_apis_reject_published_t0_before_any_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, technical, verify_modes, run_starts, registry_writes = _generic_t0_rejection_fixture(
        tmp_path, monkeypatch
    )
    common = {
        "project_root": project,
        "authority_directory": technical,
    }
    calls = [
        (
            lifecycle_module.execute_lifecycle_rehearsal,
            {**common, "runs_root": tmp_path / "runs"},
        ),
        (
            lifecycle_module.verify_lifecycle_rehearsal_fresh_process,
            {
                **common,
                "rehearsal_run_directory": tmp_path / "rehearsal",
                "runs_root": tmp_path / "runs",
            },
        ),
        (
            lifecycle_module.require_current_lifecycle_readiness,
            {**common, "readiness_run_directory": tmp_path / "readiness"},
        ),
    ]
    for function, kwargs in calls:
        with pytest.raises(LifecycleQualificationError, match="generic lifecycle path"):
            function(**kwargs)
    assert run_starts == []
    assert registry_writes == []
    assert verify_modes == []
    assert not (tmp_path / "runs").exists()


def test_generic_workflow_aliases_and_cli_reject_published_t0_before_any_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, technical, verify_modes, run_starts, registry_writes = _generic_t0_rejection_fixture(
        tmp_path, monkeypatch
    )
    common = {
        "project_root": project,
        "authority_directory": technical,
    }
    calls = [
        (
            workflows_api.execute_lifecycle_rehearsal,
            {**common, "runs_root": tmp_path / "runs"},
        ),
        (
            workflows_api.verify_lifecycle_rehearsal_fresh_process,
            {
                **common,
                "rehearsal_run_directory": tmp_path / "rehearsal",
                "runs_root": tmp_path / "runs",
            },
        ),
        (
            workflows_api.require_current_lifecycle_readiness,
            {**common, "readiness_run_directory": tmp_path / "readiness"},
        ),
    ]
    for function, kwargs in calls:
        with pytest.raises(LifecycleQualificationError, match="generic lifecycle path"):
            function(**kwargs)

    runner = CliRunner()
    rehearsal_cli = runner.invoke(
        app,
        [
            "experiment",
            "lifecycle-rehearsal",
            "--project-root",
            str(project),
            "--authority-dir",
            str(technical),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert rehearsal_cli.exit_code == 1
    assert "generic lifecycle path" in rehearsal_cli.output
    readiness_cli = runner.invoke(
        app,
        [
            "experiment",
            "verify-lifecycle-rehearsal",
            "--project-root",
            str(project),
            "--authority-dir",
            str(technical),
            "--rehearsal-run-dir",
            str(tmp_path / "rehearsal"),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )
    assert readiness_cli.exit_code == 1
    assert "generic lifecycle path" in readiness_cli.output
    assert run_starts == []
    assert registry_writes == []
    assert verify_modes == []
    assert not (tmp_path / "runs").exists()


def _invoke_cli_subprocess(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "histo_audit", *arguments],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=60,
    )


def _run_cli_subprocess(*arguments: str) -> dict[str, Any]:
    completed = _invoke_cli_subprocess(*arguments)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def _qualify_lifecycle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project, authority = _make_authority(tmp_path)
    runs = project / "artifacts" / "runs"
    rehearsal_payload = _run_cli_subprocess(
        "experiment",
        "lifecycle-rehearsal",
        "--project-root",
        str(project),
        "--authority-dir",
        str(authority),
        "--runs-root",
        str(runs),
    )
    rehearsal = Path(rehearsal_payload["result"]["run_directory"])
    readiness_payload = _run_cli_subprocess(
        "experiment",
        "verify-lifecycle-rehearsal",
        "--project-root",
        str(project),
        "--authority-dir",
        str(authority),
        "--rehearsal-run-dir",
        str(rehearsal),
        "--runs-root",
        str(runs),
    )
    readiness = Path(readiness_payload["result"]["readiness_run_directory"])
    return project, authority, rehearsal, readiness


@pytest.fixture(scope="module")
def qualified_lifecycle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path, Path]:
    return _qualify_lifecycle(tmp_path_factory.mktemp("lifecycle-qualified"))


def test_real_subprocess_rehearsal_and_fresh_readiness_are_idempotently_reopened(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
) -> None:
    project, authority, rehearsal, readiness = qualified_lifecycle
    before = (readiness / ".immutable.json").stat().st_mtime_ns
    first = require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    )
    second = require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    )
    assert first.valid and second == first
    assert first.rehearsal_run_directory == rehearsal.resolve()
    assert (readiness / ".immutable.json").stat().st_mtime_ns == before


def test_second_fresh_verifier_is_no_overwrite_and_creates_no_positive_run(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
) -> None:
    project, authority, rehearsal, readiness = qualified_lifecycle
    runs = readiness.parent
    directories_before = {path.name for path in runs.iterdir() if path.is_dir()}
    registry_before = (runs / "registry.csv").read_bytes()
    repeated = _invoke_cli_subprocess(
        "experiment",
        "verify-lifecycle-rehearsal",
        "--project-root",
        str(project),
        "--authority-dir",
        str(authority),
        "--rehearsal-run-dir",
        str(rehearsal),
        "--runs-root",
        str(runs),
    )
    assert repeated.returncode != 0
    assert "already exists" in (repeated.stderr + repeated.stdout)
    assert {path.name for path in runs.iterdir() if path.is_dir()} == directories_before
    assert (runs / "registry.csv").read_bytes() == registry_before


def test_readiness_rejects_an_extra_sealed_file(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
) -> None:
    project, authority, _, readiness = qualified_lifecycle
    extra = readiness / "unexpected.txt"
    extra.write_text("not part of the seal\n", encoding="utf-8")
    try:
        with pytest.raises(LifecycleQualificationError, match="failed closed"):
            require_current_lifecycle_readiness(
                project_root=project,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
    finally:
        extra.unlink(missing_ok=True)
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid


def test_readiness_rejects_a_duplicate_main_registry_row(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
) -> None:
    project, authority, _, readiness = qualified_lifecycle
    registry = readiness.parent / "registry.csv"
    original = registry.read_text(encoding="utf-8")
    matching = next(line for line in original.splitlines() if readiness.name in line)
    registry.write_text(original + matching + "\n", encoding="utf-8")
    try:
        with pytest.raises(LifecycleQualificationError, match="failed closed"):
            require_current_lifecycle_readiness(
                project_root=project,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
    finally:
        registry.write_text(original, encoding="utf-8")
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid


@pytest.mark.parametrize(
    "tamper",
    ["missing_record", "missing_anchor", "stale_anchor", "duplicate_record"],
)
def test_readiness_rejects_missing_stale_or_duplicate_qualification_attestation(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
    tamper: str,
) -> None:
    project, authority, _, readiness = qualified_lifecycle
    runs = readiness.parent
    registry = runs / RUN_STAGE_ATTESTATION_REGISTRY_FILENAME
    anchor = runs / RUN_STAGE_ATTESTATION_ANCHOR_FILENAME
    original_registry = registry.read_bytes()
    original_anchor = anchor.read_bytes()
    records = [json.loads(line) for line in original_registry.decode("utf-8").splitlines()]
    baseline = require_lifecycle_run_qualified(readiness)
    assert baseline["run_id"] == readiness.name
    assert baseline["scientific_stage_eligible"] is False
    assert baseline["completion_stage"] is None

    if tamper == "missing_record":
        registry.write_bytes(b"")
        _write_json(anchor, _attestation_anchor([], b""))
    elif tamper == "missing_anchor":
        anchor.unlink()
    elif tamper == "stale_anchor":
        stale = json.loads(original_anchor)
        stale["record_count"] = int(stale["record_count"]) + 1
        _write_json(anchor, stale)
    else:
        duplicate = dict(next(record for record in records if record["run_id"] == readiness.name))
        duplicate["sequence"] = len(records) + 1
        duplicate["previous_record_sha256"] = records[-1]["record_sha256"]
        duplicate["record_sha256"] = _attestation_record_sha256(duplicate)
        duplicate_line = json.dumps(
            duplicate,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        changed_registry = original_registry + duplicate_line + b"\n"
        changed_records = [*records, duplicate]
        registry.write_bytes(changed_registry)
        _write_json(anchor, _attestation_anchor(changed_records, changed_registry))

    try:
        with pytest.raises((ValueError, LifecycleQualificationError)):
            require_lifecycle_run_qualified(readiness)
        with pytest.raises(LifecycleQualificationError, match="failed closed"):
            require_current_lifecycle_readiness(
                project_root=project,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
    finally:
        registry.write_bytes(original_registry)
        anchor.write_bytes(original_anchor)

    assert require_lifecycle_run_qualified(readiness) == baseline
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid


def test_readiness_rejects_live_source_drift(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
) -> None:
    project, authority, _, readiness = qualified_lifecycle
    module = project / "src" / "qualification_fixture" / "module.py"
    original = module.read_text(encoding="utf-8")
    module.write_text("VALUE = 2\n", encoding="utf-8")
    try:
        with pytest.raises(LifecycleQualificationError, match="failed closed"):
            require_current_lifecycle_readiness(
                project_root=project,
                authority_directory=authority,
                readiness_run_directory=readiness,
            )
    finally:
        module.write_text(original, encoding="utf-8")
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid


@pytest.mark.parametrize("withdrawn_role", ["rehearsal", "readiness"])
def test_readiness_rejects_a_withdrawn_lifecycle_run(
    tmp_path: Path,
    withdrawn_role: str,
) -> None:
    project, authority, rehearsal, readiness = _qualify_lifecycle(tmp_path)
    withdrawn = rehearsal if withdrawn_role == "rehearsal" else readiness
    before = verify_run_integrity(withdrawn)

    disposition = withdraw_run_eligibility(
        withdrawn,
        reason_code=f"test_{withdrawn_role}_withdrawal",
        reason=f"Adversarial withdrawal of the {withdrawn_role} lifecycle run.",
    )

    assert disposition["run_id"] == withdrawn.name
    assert disposition["scientific_stage_eligible"] is False
    assert verify_run_integrity(withdrawn) == before
    with pytest.raises(LifecycleQualificationError, match="withdrawn"):
        require_current_lifecycle_readiness(
            project_root=project,
            authority_directory=authority,
            readiness_run_directory=readiness,
        )


@pytest.mark.parametrize(
    "binding",
    ["schema", "config", "plan", "uv_lock", "verifier", "authority"],
)
def test_readiness_rejects_each_current_execution_binding_drift(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
) -> None:
    project, authority, _, readiness = qualified_lifecycle
    authority_argument = authority
    lock = project / "uv.lock"
    original_lock = lock.read_bytes()

    if binding == "schema":
        monkeypatch.setattr(
            lifecycle_module,
            "LIFECYCLE_SCHEMA_VERSION",
            lifecycle_module.LIFECYCLE_SCHEMA_VERSION + 1,
        )
    elif binding == "config":
        changed_config = dict(lifecycle_module._REHEARSAL_CONFIG)
        changed_config["synthetic_sample_count"] = 11
        monkeypatch.setattr(lifecycle_module, "_REHEARSAL_CONFIG", changed_config)
    elif binding == "plan":
        changed_plan = dict(lifecycle_module._REHEARSAL_PLAN)
        changed_plan["policy"] = "costly_study_lifecycle_qualification_drifted"
        monkeypatch.setattr(lifecycle_module, "_REHEARSAL_PLAN", changed_plan)
    elif binding == "uv_lock":
        lock.write_text("version = 2\n", encoding="utf-8")
    elif binding == "verifier":
        original_verifier = lifecycle_module._verifier_record

        def drifted_verifier() -> dict[str, Any]:
            record = original_verifier()
            return {**record, "module_sha256": "f" * 64}

        monkeypatch.setattr(lifecycle_module, "_verifier_record", drifted_verifier)
    else:
        authority_argument = tmp_path / "alternate-valid-authority"
        shutil.copytree(authority, authority_argument)
        assert verify_preregistration_freeze(authority_argument).valid

    try:
        with pytest.raises(LifecycleQualificationError, match="failed closed"):
            require_current_lifecycle_readiness(
                project_root=project,
                authority_directory=authority_argument,
                readiness_run_directory=readiness,
            )
    finally:
        if binding == "uv_lock":
            lock.write_bytes(original_lock)


def test_confirmatory_direct_entry_fails_before_any_input_without_readiness(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfirmatoryStudyRunnerError, match="lifecycle readiness is required"):
        execute_confirmatory_study(
            gate_evidence=object(),  # type: ignore[arg-type]
            primary_run_directory=tmp_path / "never-read-primary",
            project_root=tmp_path,
            freeze_directory=tmp_path / "never-read-authority",
            dataset_path=tmp_path / "never-read-dataset",
            manifest_path=tmp_path / "never-read-manifest",
            duplicate_audit_path=tmp_path / "never-read-duplicates",
            pathology_encoder_audit_path=tmp_path / "never-read-pathology",
            frozen_primary_config_path=tmp_path / "never-read-primary-config",
            frozen_confirmatory_config_path=tmp_path / "never-read-confirmatory-config",
            crop_cache_path=tmp_path / "never-read-cache",
            expected_crop_cache_sha256="0" * 64,
            expected_crop_metadata_sha256="0" * 64,
            expected_raw_inventory_sha256="0" * 64,
            frozen_feature_caches=(),
        )
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_public_cli_exposes_lifecycle_commands_and_confirmatory_gate() -> None:
    rehearsal = cli_options(app, ("experiment", "lifecycle-rehearsal"))
    verifier = cli_options(app, ("experiment", "verify-lifecycle-rehearsal"))
    confirmatory = cli_options(app, ("experiment", "confirmatory"))
    assert "--authority-dir" in rehearsal
    assert "--rehearsal-run-dir" in verifier
    assert "--lifecycle-readiness-run-dir" in confirmatory


def test_partial_publication_failure_is_sealed_failed_and_can_be_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, authority = _make_authority(tmp_path)
    runs = project / "artifacts" / "runs"
    failed_run_id = "lifecycle_rehearsal_injected_failure"
    original_copy = AnchoredPhysicalCopySession.copy_file_no_overwrite
    calls = 0

    def fail_third_copy(
        self: AnchoredPhysicalCopySession,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected partial publication failure")
        return original_copy(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(AnchoredPhysicalCopySession, "copy_file_no_overwrite", fail_third_copy)
        with pytest.raises(RuntimeError, match="injected partial publication failure"):
            execute_lifecycle_rehearsal(
                project_root=project,
                authority_directory=authority,
                runs_root=runs,
                run_id=failed_run_id,
            )

    failed = runs / failed_run_id
    failed_status = json.loads((failed / "status.json").read_text(encoding="utf-8"))
    failed_integrity = verify_run_integrity(failed)
    assert failed_status["status"] == "failed"
    assert "completion_stage" not in failed_status
    assert failed_integrity.valid and failed_integrity.registry_record_present
    assert not (failed / "lifecycle_readiness_evidence.json").exists()
    assert not list(project.glob(f".{failed_run_id}.rehearsal-*"))

    retry = execute_lifecycle_rehearsal(
        project_root=project,
        authority_directory=authority,
        runs_root=runs,
        run_id="lifecycle_rehearsal_exact_retry",
        retry_of_run_id=failed_run_id,
    )
    retry_completion = json.loads(
        (retry.run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    assert retry_completion["retry_of_run_id"] == failed_run_id
    assert retry_completion["retry_lineage_binding_sha256"]

    with pytest.raises(LifecycleQualificationError, match="registry row differs"):
        execute_lifecycle_rehearsal(
            project_root=project,
            authority_directory=authority,
            runs_root=runs,
            run_id="retry_of_completed_is_forbidden",
            retry_of_run_id=retry.run_id,
        )
    with pytest.raises(LifecycleQualificationError, match="unavailable"):
        execute_lifecycle_rehearsal(
            project_root=project,
            authority_directory=authority,
            runs_root=runs,
            run_id="retry_of_missing_is_forbidden",
            retry_of_run_id="missing_rehearsal_run",
        )


@pytest.mark.parametrize(
    ("fault_point", "committed_before_fault"),
    [
        ("integrity_registry", False),
        ("main_registry", True),
        ("immutable_marker", True),
    ],
)
def test_interrupted_finalization_is_reconciled_once_and_preserves_retry_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
    committed_before_fault: bool,
) -> None:
    project, authority = _make_authority(tmp_path)
    runs = project / "artifacts" / "runs"
    run_id = f"lifecycle_rehearsal_interrupted_{fault_point}"
    injected = False

    if fault_point == "integrity_registry":
        original = run_tracking_module.append_integrity_record

        def fail_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if not injected:
                injected = True
                raise RuntimeError("injected integrity-registry interruption")
            return original(*args, **kwargs)

        monkeypatch.setattr(run_tracking_module, "append_integrity_record", fail_once)
    elif fault_point == "main_registry":
        original = run_tracking_module.append_registry_row

        def fail_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if not injected:
                injected = True
                raise RuntimeError("injected main-registry interruption")
            return original(*args, **kwargs)

        monkeypatch.setattr(run_tracking_module, "append_registry_row", fail_once)
    else:
        original = run_tracking_module.atomic_write_json

        def fail_once(destination: str | Path, *args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if Path(destination).name == ".immutable.json" and not injected:
                injected = True
                raise RuntimeError("injected immutable-marker interruption")
            return original(destination, *args, **kwargs)

        monkeypatch.setattr(run_tracking_module, "atomic_write_json", fail_once)

    if committed_before_fault:
        rehearsal = execute_lifecycle_rehearsal(
            project_root=project,
            authority_directory=authority,
            runs_root=runs,
            run_id=run_id,
        )
        terminal = rehearsal.run_directory
        expected_status = "completed"
    else:
        with pytest.raises(RuntimeError, match="injected integrity-registry interruption"):
            execute_lifecycle_rehearsal(
                project_root=project,
                authority_directory=authority,
                runs_root=runs,
                run_id=run_id,
            )
        terminal = runs / run_id
        expected_status = "failed"
    assert injected

    with (runs / "registry.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    integrity_rows = [
        json.loads(line)
        for line in (runs / "integrity_registry.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("run_id") == run_id
    ]
    assert len(csv_rows) == len(integrity_rows) == 1
    assert csv_rows[0]["status"] == integrity_rows[0]["status"] == expected_status
    status = json.loads((terminal / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == expected_status
    integrity = verify_run_integrity(terminal)
    assert integrity.valid and integrity.registry_record_present

    monkeypatch.undo()
    if expected_status == "failed":
        retry = execute_lifecycle_rehearsal(
            project_root=project,
            authority_directory=authority,
            runs_root=runs,
            run_id=f"{run_id}_exact_retry",
            retry_of_run_id=run_id,
        )
        retry_completion = json.loads(
            (retry.run_directory / "completion_evidence.json").read_text(encoding="utf-8")
        )
        assert retry_completion["retry_of_run_id"] == run_id
        assert retry_completion["retry_lineage_binding_sha256"]
        rehearsal_for_verifier = retry.run_directory
    else:
        with pytest.raises(LifecycleQualificationError, match="registry row differs"):
            execute_lifecycle_rehearsal(
                project_root=project,
                authority_directory=authority,
                runs_root=runs,
                run_id=f"{run_id}_forbidden_retry",
                retry_of_run_id=run_id,
            )
        rehearsal_for_verifier = terminal

    readiness_payload = _run_cli_subprocess(
        "experiment",
        "verify-lifecycle-rehearsal",
        "--project-root",
        str(project),
        "--authority-dir",
        str(authority),
        "--rehearsal-run-dir",
        str(rehearsal_for_verifier),
        "--runs-root",
        str(runs),
    )
    readiness = Path(readiness_payload["result"]["readiness_run_directory"])
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid


@pytest.mark.parametrize(
    ("attestation_name", "fault"),
    [
        pytest.param("statistics_attestation.json", "missing", id="statistics-missing"),
        pytest.param("statistics_attestation.json", "stale", id="statistics-stale"),
        pytest.param("restoration_attestation.json", "missing", id="restoration-missing"),
        pytest.param("restoration_attestation.json", "stale", id="restoration-stale"),
    ],
)
def test_fresh_verifier_rejects_integrity_valid_semantic_attestation_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attestation_name: str,
    fault: str,
) -> None:
    project, authority = _make_authority(tmp_path)
    runs = project / "artifacts" / "runs"
    original_complete = run_tracking_module.RunTracker.complete
    injected = False

    def complete_with_semantic_fault(tracker: Any) -> None:
        nonlocal injected
        if not injected and tracker.experiment_name == "lifecycle_qualification_rehearsal":
            attestation_path = tracker.run_directory / attestation_name
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            if fault == "missing":
                attestation.pop("verification_status")
            elif attestation_name == "statistics_attestation.json":
                attestation["recomputed_mean"] = float(attestation["recomputed_mean"]) + 1.0
            else:
                attestation["restored_count"] = int(attestation["restored_count"]) + 1
            _write_json(attestation_path, attestation)
            _refresh_rehearsal_semantic_receipts(tracker.run_directory)
            injected = True
        original_complete(tracker)

    monkeypatch.setattr(run_tracking_module.RunTracker, "complete", complete_with_semantic_fault)
    rehearsal = execute_lifecycle_rehearsal(
        project_root=project,
        authority_directory=authority,
        runs_root=runs,
        run_id=f"semantic_{attestation_name.removesuffix('.json')}_{fault}",
    )
    assert injected
    integrity = verify_run_integrity(rehearsal.run_directory)
    assert integrity.valid and integrity.registry_record_present
    directories_before = {path.name for path in runs.iterdir() if path.is_dir()}

    monkeypatch.undo()
    completed = _invoke_cli_subprocess(
        "experiment",
        "verify-lifecycle-rehearsal",
        "--project-root",
        str(project),
        "--authority-dir",
        str(authority),
        "--rehearsal-run-dir",
        str(rehearsal.run_directory),
        "--runs-root",
        str(runs),
    )
    expected_role = "statistics" if attestation_name.startswith("statistics") else "restoration"
    assert completed.returncode != 0
    assert f"{expected_role} attestation verification failed" in (
        completed.stderr + completed.stdout
    )
    assert {path.name for path in runs.iterdir() if path.is_dir()} == directories_before
    assert verify_run_integrity(rehearsal.run_directory) == integrity


def test_readiness_rejects_hardlink_and_corrupt_attestation_then_recovers(
    qualified_lifecycle: tuple[Path, Path, Path, Path],
    tmp_path: Path,
) -> None:
    project, authority, rehearsal, readiness = qualified_lifecycle
    hardlink = tmp_path / "checkpoint-alias.pt"
    try:
        os.link(rehearsal / "checkpoint.pt", hardlink)
    except OSError as error:
        pytest.skip(f"host does not support a test hardlink: {error}")
    with pytest.raises(LifecycleQualificationError, match="single-link"):
        require_current_lifecycle_readiness(
            project_root=project,
            authority_directory=authority,
            readiness_run_directory=readiness,
        )
    hardlink.unlink()
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid

    attestation = rehearsal / "statistics_attestation.json"
    original_bytes = attestation.read_bytes()
    attestation.write_text('{"schema_version": 999}\n', encoding="utf-8")
    with pytest.raises(LifecycleQualificationError, match="integrity failed"):
        require_current_lifecycle_readiness(
            project_root=project,
            authority_directory=authority,
            readiness_run_directory=readiness,
        )
    attestation.write_bytes(original_bytes)
    assert require_current_lifecycle_readiness(
        project_root=project,
        authority_directory=authority,
        readiness_run_directory=readiness,
    ).valid
