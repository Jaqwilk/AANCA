"""Fail-closed execution-gate tests for frozen and amended identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_workflow_stage_apis import (
    _completed_pilot,
    _pilot_authority_kwargs,
    _project_inputs,
)

from histo_audit.utils.run_tracking import sha256_file
from histo_audit.workflows.preregistration import (
    freeze_preregistration,
    verify_preregistration_freeze,
)
from histo_audit.workflows.preregistration_amendment import (
    create_preregistration_amendment,
)
from histo_audit.workflows.study_gates import validate_primary_execution_gate

FREEZE_TIME = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class FrozenGateFixture:
    root: Path
    freeze: Path
    dataset: Path
    manifest: Path
    duplicate_audit: Path
    pathology: Path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative not in {".immutable.json", "sha256_manifest.json"}:
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _canonical_root(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reseal_authority(directory: Path) -> None:
    manifest_path = directory / "sha256_manifest.json"
    marker_path = directory / ".immutable.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    records = _artifact_records(directory)
    root = _canonical_root(records)
    manifest["artifacts"] = records
    manifest["artifact_count"] = len(records)
    manifest["artifact_root_sha256"] = root
    _write_json(manifest_path, manifest)
    marker["artifact_root_sha256"] = root
    marker["sha256_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(marker_path, marker)


def _stub_post_seal(monkeypatch: pytest.MonkeyPatch) -> None:
    from histo_audit.experiment import pilot_postseal

    def verify(
        run_directory: str | Path,
        *,
        development_manifest_source: str | Path,
        gate_certificate_source: str | Path,
    ) -> dict[str, Any]:
        run = Path(run_directory).resolve()
        development = Path(development_manifest_source).resolve()
        certificate = Path(gate_certificate_source).resolve()
        assert development.read_bytes() == (run / "development_manifest_view.parquet").read_bytes()
        assert certificate.read_bytes() == (run / "pre_pilot_gate_certificate.json").read_bytes()
        return {
            "schema_version": 1,
            "status": "passed",
            "policy": "read_only_pilot_post_seal_verification_v1_fixture",
            "run_id": run.name,
            "scientific_stage_eligible": True,
            "sealed_run_unchanged": True,
        }

    monkeypatch.setattr(pilot_postseal, "verify_pilot_post_seal", verify)


def _frozen_gate_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenGateFixture:
    _stub_post_seal(monkeypatch)
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    result = freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        **_pilot_authority_kwargs(root),
        timestamp=FREEZE_TIME,
    )
    return FrozenGateFixture(
        root=root,
        freeze=result.freeze_directory,
        dataset=dataset,
        manifest=manifest,
        duplicate_audit=duplicate_audit,
        pathology=pathology,
    )


def _gate(fixture: FrozenGateFixture, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "project_root": fixture.root,
        "freeze_directory": fixture.freeze,
        "dataset_path": fixture.dataset,
        "manifest_path": fixture.manifest,
        "duplicate_audit_path": fixture.duplicate_audit,
        "pathology_encoder_audit_path": fixture.pathology,
    }
    arguments.update(overrides)
    return validate_primary_execution_gate(**arguments)


def _create_amendment(
    fixture: FrozenGateFixture,
    *,
    outcomes_inspected: bool = False,
) -> Any:
    (fixture.root / "src" / "amended_runtime.py").write_text(
        "AMENDED_EXECUTION = True\n", encoding="utf-8"
    )
    timestamp = FREEZE_TIME + timedelta(hours=1)
    return create_preregistration_amendment(
        project_root=fixture.root,
        parent_authority_directory=fixture.freeze,
        amendment_root=fixture.root / "artifacts" / "preregistration_amendments",
        preregistration_path=fixture.root / "PRE_REGISTRATION.md",
        primary_config_path=fixture.root / "configs" / "primary.yaml",
        confirmatory_config_path=fixture.root / "configs" / "confirmatory.yaml",
        reason="Correct a prespecified execution implementation.",
        affected_hypotheses=["H1"],
        affected_analyses=["primary_ranking"],
        outcomes_inspected=outcomes_inspected,
        outcomes_inspected_at=(timestamp - timedelta(minutes=15) if outcomes_inspected else None),
        timestamp=timestamp,
    )


def test_base_gate_authenticates_frozen_governance_but_allows_live_status_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    initial = _gate(fixture)

    (fixture.root / "STATUS.md").write_text(
        "# Current status\n\nTruthful post-freeze execution status.\n", encoding="utf-8"
    )
    after_governance_update = _gate(fixture)

    assert after_governance_update.source_tree_root_sha256 == initial.source_tree_root_sha256
    assert after_governance_update.registration_authority_kind == "base_freeze"
    (fixture.root / "src" / "outcome_dependent_change.py").write_text(
        "CHANGED = True\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="current execution source tree differs"):
        _gate(fixture)


def test_resealed_base_cannot_desynchronise_tree_evidence_from_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    evidence_path = fixture.freeze / "freeze_evidence.json"
    original = json.loads(evidence_path.read_text(encoding="utf-8"))

    mutated = json.loads(json.dumps(original))
    mutated["execution_source_tree"]["artifact_count"] += 1
    _write_json(evidence_path, mutated)
    _reseal_authority(fixture.freeze)
    verification = verify_preregistration_freeze(
        fixture.freeze,
        frozen_primary_config_path=fixture.root / "configs" / "primary_frozen.yaml",
        frozen_confirmatory_config_path=fixture.root / "configs" / "confirmatory_frozen.yaml",
    )
    assert verification.valid
    with pytest.raises(ValueError, match="artifact_count differs"):
        _gate(fixture)

    _write_json(evidence_path, original)
    _reseal_authority(fixture.freeze)
    mutated = json.loads(json.dumps(original))
    mutated["governance_tree"]["scope"] = ["STATUS.md"]
    _write_json(evidence_path, mutated)
    _reseal_authority(fixture.freeze)
    assert verify_preregistration_freeze(fixture.freeze).valid
    with pytest.raises(ValueError, match="scope differs"):
        _gate(fixture)


def test_base_gate_rejects_mutated_canonical_config_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    (fixture.root / "configs" / "primary_frozen.yaml").write_text(
        "schema_version: 999\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="integrity verification"):
        _gate(fixture)


def test_pre_outcome_amendment_is_the_latest_bound_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    amendment = _create_amendment(fixture)

    gate = _gate(
        fixture,
        freeze_directory=amendment.amendment_directory,
        frozen_primary_config_path=amendment.amended_primary_config_path,
        frozen_confirmatory_config_path=amendment.amended_confirmatory_config_path,
    )

    assert gate.freeze_directory == amendment.amendment_directory
    assert gate.registration_authority_kind == "preregistration_amendment"
    assert gate.registration_authority_chain_depth == 1
    assert gate.original_unamended_primary_claim_allowed is False
    assert gate.amended_primary_claim_allowed is True
    assert gate.freeze_artifact_root_sha256 == amendment.artifact_root_sha256
    with pytest.raises(ValueError, match="latest verified amendment bundle"):
        _gate(
            fixture,
            freeze_directory=amendment.amendment_directory,
            frozen_primary_config_path=fixture.root / "configs" / "primary_frozen.yaml",
            frozen_confirmatory_config_path=fixture.root / "configs" / "confirmatory_frozen.yaml",
        )


def test_post_outcome_amendment_cannot_authorize_primary_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    amendment = _create_amendment(fixture, outcomes_inspected=True)

    with pytest.raises(ValueError, match="post-outcome amendment"):
        _gate(
            fixture,
            freeze_directory=amendment.amendment_directory,
            frozen_primary_config_path=amendment.amended_primary_config_path,
            frozen_confirmatory_config_path=amendment.amended_confirmatory_config_path,
        )


def test_resealed_amendment_parent_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_gate_fixture(tmp_path, monkeypatch)
    amendment = _create_amendment(fixture)
    evidence = json.loads(amendment.amendment_evidence_path.read_text(encoding="utf-8"))
    evidence["parent"]["artifact_root_sha256"] = "0" * 64
    _write_json(amendment.amendment_evidence_path, evidence)
    _reseal_authority(amendment.amendment_directory)

    with pytest.raises(ValueError, match="amendment failed chain/integrity verification"):
        _gate(
            fixture,
            freeze_directory=amendment.amendment_directory,
            frozen_primary_config_path=amendment.amended_primary_config_path,
            frozen_confirmatory_config_path=amendment.amended_confirmatory_config_path,
        )
