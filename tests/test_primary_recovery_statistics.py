"""Recovery-specific statistics capability and provenance boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_primary_completion import _real_gate
from test_primary_recovery import _make_fixture

import histo_audit.experiment.primary_statistics as primary_statistics_module
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    build_primary_completion_evidence,
)
from histo_audit.experiment.primary_statistics import (
    INHERITED_PRIOR_NUMERIC_LIMITATION,
    INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
    AuthorizedOrphanNumericVerificationProof,
    InheritedPrimaryStatisticsVerificationProvenance,
    OrphanRecoveryNumericVerificationProvenance,
    attest_inherited_primary_statistics_artifacts,
)
from histo_audit.utils.run_tracking import sha256_file

_PROOF_SHA256 = "9" * 64
_STATISTICS_FILES = (
    "primary_statistics.json",
    "primary_bootstrap_evidence.npz",
    "primary_subgroups.csv",
    "primary_statistics_manifest.json",
)


def _quartet(run_directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "size_bytes": (run_directory / name).stat().st_size,
            "sha256": sha256_file(run_directory / name),
        }
        for name in _STATISTICS_FILES
    ]


def _comparison_count(fixture: Any) -> int:
    controls = fixture.controls
    return sum(
        len(values)
        for values in (
            controls.within_cell_comparisons,
            controls.method_vs_random_comparisons,
            controls.cross_cell_comparisons,
        )
    )


def _orphan_proof(
    fixture: Any,
    *,
    statistics_quartet: list[dict[str, Any]] | None = None,
) -> AuthorizedOrphanNumericVerificationProof:
    authorization = fixture.authorization
    return primary_statistics_module._issue_authorized_orphan_numeric_verification_proof(
        amendment_directory=fixture.runs_root / "technical-amendment",
        authorization_sha256=authorization.canonical_sha256,
        source_run_id=authorization.source_run_id,
        source_snapshot_root_sha256=authorization.expected_source_snapshot_root_sha256,
        source_status_sha256=authorization.expected_status_sha256,
        source_tree_root_sha256=authorization.expected_source_tree_root_sha256,
        source_tree_manifest_sha256=(authorization.expected_source_tree_manifest_sha256),
        source_readback_root_sha256=(authorization.expected_source_filesystem_readback_root_sha256),
        prior_numeric_verification_proof_sha256=_PROOF_SHA256,
        trust_assumption=authorization.trust_assumption,
        limitation=authorization.limitation,
        statistics_quartet=(
            _quartet(fixture.source) if statistics_quartet is None else statistics_quartet
        ),
        comparison_count=_comparison_count(fixture),
    )


def _finalization_proof(fixture: Any) -> Any:
    return primary_statistics_module._issue_authorized_prior_numeric_verification_proof(
        amendment_directory=fixture.runs_root / "finalization-amendment",
        authorization_sha256="a" * 64,
        predecessor_run_id="sealed-predecessor",
        predecessor_artifact_root_sha256="b" * 64,
        predecessor_artifact_manifest_sha256="c" * 64,
        predecessor_source_tree_root_sha256="d" * 64,
        source_readback_root_sha256=(
            fixture.authorization.expected_source_filesystem_readback_root_sha256
        ),
        prior_numeric_verification_proof_sha256=_PROOF_SHA256,
        trust_assumption=INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
        limitation=INHERITED_PRIOR_NUMERIC_LIMITATION,
        statistics_quartet=_quartet(fixture.source),
        comparison_count=_comparison_count(fixture),
    )


def _completion_kwargs(fixture: Any) -> dict[str, Any]:
    filesystem = fixture.snapshot.filesystem_readback
    return {
        "plan": fixture.plan,
        "reconciliation": filesystem.reconciliation,
        "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
        "study_outcome_eligible": True,
        "gate_evidence": _real_gate(fixture.plan, fixture.runs_root),
        "filesystem_readback": filesystem,
        "restoration_readback": fixture.snapshot.restoration_readback,
    }


def test_orphan_attestation_has_recovery_provenance_without_sealed_predecessor_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    proof = _orphan_proof(fixture)

    verification, provenance = attest_inherited_primary_statistics_artifacts(
        fixture.source,
        fixture.controls,
        authorization=proof,
    )

    assert proof.valid
    assert verification.valid
    assert verification.authorization_kind == "orphan_recovery"
    assert isinstance(provenance, OrphanRecoveryNumericVerificationProvenance)
    payload = provenance.as_dict()
    assert payload["authorization_sha256"] == fixture.authorization.canonical_sha256
    assert payload["source_run_id"] == fixture.source.name
    assert payload["source_snapshot_root_sha256"] == fixture.snapshot.snapshot_root_sha256
    assert payload["source_status_sha256"] == fixture.authorization.expected_status_sha256
    assert (
        payload["source_tree_manifest_sha256"]
        == fixture.authorization.expected_source_tree_manifest_sha256
    )
    assert all(
        forbidden not in key
        for key in payload
        for forbidden in ("predecessor", "artifact_root", "finalization")
    )


def test_finalization_attestation_keeps_finalization_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)

    verification, provenance = attest_inherited_primary_statistics_artifacts(
        fixture.source,
        fixture.controls,
        authorization=_finalization_proof(fixture),
    )

    assert verification.valid
    assert verification.authorization_kind == "finalization_successor"
    assert isinstance(provenance, InheritedPrimaryStatisticsVerificationProvenance)
    assert provenance.predecessor_run_id == "sealed-predecessor"
    assert "source_snapshot_root_sha256" not in provenance.as_dict()


def test_completion_accepts_only_the_matching_recovery_capability_and_exact_quartet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    orphan_proof = _orphan_proof(fixture)
    finalization_proof = _finalization_proof(fixture)
    orphan_verification, _ = attest_inherited_primary_statistics_artifacts(
        fixture.source,
        fixture.controls,
        authorization=orphan_proof,
    )
    finalization_verification, _ = attest_inherited_primary_statistics_artifacts(
        fixture.source,
        fixture.controls,
        authorization=finalization_proof,
    )
    completion_kwargs = _completion_kwargs(fixture)

    completion = build_primary_completion_evidence(
        **completion_kwargs,
        statistics_verification=orphan_verification,
        _inherited_authorization=orphan_proof,
    )
    assert completion["completion_stage"] == "PRIMARY_STUDY_COMPLETE"

    with pytest.raises(ValueError, match="attested primary statistics"):
        build_primary_completion_evidence(
            **completion_kwargs,
            statistics_verification=orphan_verification,
            _inherited_authorization=finalization_proof,
        )
    with pytest.raises(ValueError, match="attested primary statistics"):
        build_primary_completion_evidence(
            **completion_kwargs,
            statistics_verification=finalization_verification,
            _inherited_authorization=orphan_proof,
        )

    wrong_quartet = _quartet(fixture.source)
    wrong_quartet[0] = {**wrong_quartet[0], "sha256": "0" * 64}
    mismatched_orphan_proof = _orphan_proof(
        fixture,
        statistics_quartet=wrong_quartet,
    )
    with pytest.raises(ValueError, match="attested primary statistics"):
        build_primary_completion_evidence(
            **completion_kwargs,
            statistics_verification=orphan_verification,
            _inherited_authorization=mismatched_orphan_proof,
        )
