"""Read-only post-seal PanNuke pilot verifier and CLI tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.experiment import pilot_postseal
from histo_audit.experiment.pilot_postseal import (
    PilotPostSealVerificationError,
    _require_fixed_group_selection,
    verify_pilot_post_seal,
)
from histo_audit.utils.run_tracking import IntegrityVerification


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _privacy_result(run: Path) -> dict[str, Any]:
    paths = sorted(path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file())
    return {
        "schema_version": 1,
        "status": "passed",
        "policy": "final_reference_identity_and_outcome_nonpublication_v1",
        "final_reference_official_fold": 3,
        "final_fold_identity_pattern_absent": True,
        "final_sensitive_fields_unpopulated": True,
        "final_fold_representation_rows_absent": True,
        "scanned_file_count": len(paths),
        "text_file_count": len(paths) - 2,
        "npz_file_count": 1,
        "npz_array_count": 9,
        "parquet_file_count": 1,
        "parquet_row_count": 4,
        "scanned_paths": paths,
    }


def _pilot_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "runs" / "pilot-test"
    run.mkdir(parents=True)
    sources = tmp_path / "sources"
    sources.mkdir()
    development_source = sources / "development.parquet"
    certificate_source = sources / "development.metadata.json"
    development_source.write_bytes(b"development-folds-1-and-2")
    certificate_source.write_bytes(b'{"policy":"pre_pilot_privacy_gate_v1"}')
    (run / "development_manifest_view.parquet").write_bytes(development_source.read_bytes())
    (run / "pre_pilot_gate_certificate.json").write_bytes(certificate_source.read_bytes())

    _write_json(
        run / "status.json",
        {
            "run_id": run.name,
            "experiment_name": "pannuke_pilot",
            "status": "completed",
            "traceback": None,
        },
    )
    _write_json(
        run / "metrics.json",
        {
            "artifact_scope": "real_pannuke_controlled_corruption_pilot",
            "completion_stage_if_sealed": "PILOT_COMPLETE",
            "diagnostic_claim": False,
            "source_annotations_modified": False,
            "limitations": [
                "A potentially inconsistent annotation is recommended for expert review "
                "and is not a diagnostic output."
            ],
            "final_reference_access": {
                "official_fold": 3,
                "class_labels_read": False,
                "outcomes_used": False,
                "representations_extracted": False,
                "sample_ids_read": False,
            },
            "corruption": {
                "requested_rate": 0.25,
                "exact_count": 1,
                "only_audit_pool_corrupted": True,
                "final_reference_fold_uncorrupted": True,
            },
            "oof": {
                "complete_once_coverage": True,
                "final_reference_groups_excluded": True,
                "group_overlap_count": 0,
                "fold_count": 2,
                "probability_sum_maximum_error": 0.0,
            },
            "sample_counts": {
                "audit_pool": 4,
                "audit_groups": 4,
                "reference_validation": 1,
                "final_reference_test": 2,
            },
        },
    )
    (run / "report.md").write_text(
        "A potentially inconsistent annotation is recommended for expert review. "
        "This is not a diagnostic system.\n",
        encoding="utf-8",
    )
    _write_json(
        run / "selected_groups_and_samples.json",
        {
            "audit_sample_ids": ["s1", "s2", "s3", "s4"],
            "audit_groups": ["g1", "g2", "g3", "g4"],
            "reference_validation_sample_ids": ["r1"],
            "reference_validation_groups": ["vr"],
            "final_reference_groups": ["fg"],
            "final_reference_sample_count": 2,
            "final_test_fold": 3,
            "no_group_overlap_verified": True,
            "pairwise_group_overlap_counts": {
                "audit/reference_validation": 0,
                "audit/final_reference": 0,
                "reference_validation/final_reference": 0,
            },
            "final_reference_class_labels_read": False,
            "final_reference_outcomes_used": False,
            "final_reference_representations_extracted": False,
            "final_reference_sample_ids_read": False,
        },
    )
    _write_json(
        run / "oof_provenance.json",
        {
            "class_order": [0, 1],
            "fold_assignment_label_source": "pre_corruption_label",
            "folds": [{"fold_id": 0}, {"fold_id": 1}],
        },
    )
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]], dtype=np.float64)
    np.savez_compressed(
        run / "oof_predictions.npz",
        sample_ids=np.asarray(["s1", "s2", "s3", "s4"]),
        group_ids=np.asarray(["g1", "g2", "g3", "g4"]),
        pre_corruption_label=np.asarray([0, 1, 0, 1], dtype=np.int64),
        observed_label=np.asarray([0, 1, 1, 1], dtype=np.int64),
        is_injected_corruption=np.asarray([False, False, True, False], dtype=np.bool_),
        probabilities=probabilities,
        predicted_class=np.asarray([0, 1, 0, 1], dtype=np.int64),
        fold_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        self_confidence=np.asarray([0.2, 0.4, 0.7, 0.1], dtype=np.float64),
    )
    _write_json(run / "audit_evidence_reconciliation.json", {"status": "passed", "sample_count": 4})
    _write_json(run / "final_reference_privacy_reconciliation.json", {})
    _write_json(run / "final_reference_privacy_reconciliation.json", _privacy_result(run))
    (run.parent / "run_dispositions.jsonl").write_bytes(b"historical-record\n")
    (run.parent / "run_dispositions.anchor.json").write_bytes(b"anchor")
    return run, development_source, certificate_source


def _patch_read_only_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    run: Path,
    *,
    privacy: Callable[[Path], Mapping[str, Any]] | None = None,
) -> None:
    integrity = IntegrityVerification(
        valid=True,
        run_id=run.name,
        expected_root_sha256="a" * 64,
        actual_root_sha256="a" * 64,
        missing_paths=(),
        added_paths=(),
        changed_paths=(),
        registry_record_present=True,
        errors=(),
    )
    monkeypatch.setattr(pilot_postseal, "verify_run_integrity", lambda _run: integrity)
    monkeypatch.setattr(
        pilot_postseal,
        "require_run_stage_eligible",
        lambda _run, *, integrity: None,
    )
    monkeypatch.setattr(
        pilot_postseal,
        "read_run_dispositions",
        lambda _path: ({"run_id": "withdrawn-old-run"},),
    )
    monkeypatch.setattr(
        pilot_postseal,
        "reconcile_pilot_audit_evidence",
        lambda _run, *, require_sealed_integrity: {"status": "passed", "sample_count": 4},
    )
    monkeypatch.setattr(
        pilot_postseal,
        "_require_fixed_m6_protocol",
        lambda **_kwargs: {"status": "passed", "protocol": "fixed_real_pannuke_m6_pilot_v1"},
    )
    privacy_function = privacy or (lambda checked_run: _privacy_result(checked_run))
    monkeypatch.setattr(
        pilot_postseal,
        "_format_aware_privacy_scan",
        lambda checked_run, *, final_fold: dict(privacy_function(checked_run)),
    )


def test_fixed_group_selection_reconciles_full_group_ledger_with_metrics() -> None:
    audit_groups = [f"pannuke-fold-1-patch-{index:06d}" for index in range(225)]
    reference_groups = [f"pannuke-fold-2-patch-{index:06d}" for index in range(25)]
    selected = {
        "audit_groups": audit_groups,
        "reference_validation_groups": reference_groups,
        "final_reference_groups": ["pannuke-fold-3-patch-000001"],
        "selected_development_groups": [*audit_groups, *reference_groups],
        "selected_development_group_limit": 250,
        "development_official_folds": [1, 2],
        "selection_policy": "fixed_deterministic_group_sample_documented_in_run",
        "final_fold_complete": True,
        "final_group_limit": None,
    }
    metrics = {"sample_counts": {"selected_development_groups": 250}}

    audit, reference, final = _require_fixed_group_selection(selected, metrics)

    assert len(audit) == 225
    assert len(reference) == 25
    assert final == ("pannuke-fold-3-patch-000001",)

    malformed = dict(selected)
    malformed["selected_development_groups"] = 250
    with pytest.raises(PilotPostSealVerificationError, match="sequence of strings"):
        _require_fixed_group_selection(malformed, metrics)


def test_post_seal_verification_passes_without_mutating_run_or_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    _patch_read_only_dependencies(monkeypatch, run)
    before_run = _tree_bytes(run)
    before_parent = _tree_bytes(run.parent)

    result = verify_pilot_post_seal(
        run,
        development_manifest_source=development,
        gate_certificate_source=certificate,
    )

    assert result["status"] == "passed"
    assert result["scientific_stage_eligible"] is True
    assert result["sealed_run_unchanged"] is True
    assert result["automatic_withdrawal_performed"] is False
    assert result["byte_identity"]["development_manifest"]["byte_identical"] is True
    assert result["format_aware_privacy"]["exact_sealed_file_set_scanned"] is True
    assert result["oof_and_corruption"]["corruption_label_separation_exact"] is True
    assert _tree_bytes(run) == before_run
    assert _tree_bytes(run.parent) == before_parent


def test_post_seal_verification_rejects_corruption_label_conflation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    _patch_read_only_dependencies(monkeypatch, run)
    with np.load(run / "oof_predictions.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["is_injected_corruption"] = np.asarray([False, False, False, False], dtype=np.bool_)
    np.savez_compressed(run / "oof_predictions.npz", **arrays)

    with pytest.raises(PilotPostSealVerificationError, match="is_injected_corruption"):
        verify_pilot_post_seal(
            run,
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )


def test_post_seal_verification_rejects_nonidentical_external_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    _patch_read_only_dependencies(monkeypatch, run)
    development.write_bytes(b"different-development-view")

    with pytest.raises(PilotPostSealVerificationError, match="not byte-identical"):
        verify_pilot_post_seal(
            run,
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )


def test_post_seal_verification_rejects_missing_exact_terminology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    _patch_read_only_dependencies(monkeypatch, run)
    (run / "report.md").write_text(
        "A potentially inconsistent annotation. This is not a diagnostic system.\n",
        encoding="utf-8",
    )

    with pytest.raises(PilotPostSealVerificationError, match="recommended for expert review"):
        verify_pilot_post_seal(
            run,
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )


def test_post_seal_verification_fails_closed_on_unscanned_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    (run / "opaque.bin").write_bytes(b"unscannable")
    _patch_read_only_dependencies(
        monkeypatch,
        run,
        privacy=lambda checked_run: {
            **_privacy_result(checked_run),
            "scanned_file_count": len(_privacy_result(checked_run)["scanned_paths"]) - 1,
            "scanned_paths": [
                path
                for path in _privacy_result(checked_run)["scanned_paths"]
                if path != "opaque.bin"
            ],
        },
    )

    with pytest.raises(PilotPostSealVerificationError, match="exact sealed file set"):
        verify_pilot_post_seal(
            run,
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )


def test_post_seal_verification_eligibility_failure_does_not_withdraw_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, certificate = _pilot_fixture(tmp_path)
    _patch_read_only_dependencies(monkeypatch, run)
    monkeypatch.setattr(
        pilot_postseal,
        "require_run_stage_eligible",
        lambda _run, *, integrity: (_ for _ in ()).throw(ValueError("withdrawn")),
    )
    before = _tree_bytes(run.parent)

    with pytest.raises(PilotPostSealVerificationError, match="not scientifically stage-eligible"):
        verify_pilot_post_seal(
            run,
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )
    assert _tree_bytes(run.parent) == before


def test_verify_pilot_post_seal_cli_outputs_success_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pilot_postseal,
        "verify_pilot_post_seal",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "passed",
            "automatic_withdrawal_performed": False,
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "verify-pilot-post-seal",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            "run",
            "--development-manifest",
            "development.parquet",
            "--gate-certificate",
            "certificate.json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "schema_version": 1,
        "status": "passed",
        "automatic_withdrawal_performed": False,
    }


def test_verify_pilot_post_seal_cli_outputs_failure_json_without_withdrawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pilot_postseal,
        "verify_pilot_post_seal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("privacy leak")),
    )
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "verify-pilot-post-seal",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            "run",
            "--development-manifest",
            "development.parquet",
            "--gate-certificate",
            "certificate.json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert payload["automatic_withdrawal_performed"] is False
