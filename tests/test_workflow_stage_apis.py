from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_study_contracts import (
    complete_primary_config,
    cross_compatible_confirmatory_config,
)

from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
)
from histo_audit.experiment.study_contracts import (
    _CLEAN_REFERENCE_CELL,
    _PANNUKE_EXCLUSION_POLICY,
)
from histo_audit.external_validation import (
    build_blinded_review_package,
    validate_blinded_review_package,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    sha256_file,
    verify_run_integrity,
    withdraw_run_eligibility,
)
from histo_audit.workflows import (
    audit_original_labels,
    freeze_preregistration,
    validate_confirmatory_execution_gate,
    validate_primary_execution_gate,
    verify_preregistration_freeze,
)


def _complete_preregistration() -> str:
    detail = (
        "The fixed definition records exact choices, preserves source-group separation, and "
        "keeps the untouched final reference fold unavailable for selection. "
    )
    return f"""# Complete primary preregistration

**State:** READY_FOR_FREEZE

## Primary design
{detail * 2}

## Dataset and split
Official fold 3 is the untouched final reference test; folds 1 and 2 are development data.
The grouping unit is the source patch and the split seed is 223. {detail}

## Corruption
Rates, mechanisms, transition matrices, rounding, and seeds 404 through 406 are fixed. {detail}

## Representations and models
Frozen engineered and ImageNet representations use logistic regression and a small MLP. {detail}

## Audit methods
Self-confidence is primary and the fixed equal-weight hybrid is secondary. {detail}

## Metrics and statistics
Average precision and the five-percent budget are primary; 2,000 paired group bootstraps use seed 431.
Random review uses 100 repetitions. {detail}

## Exclusions and missing data
Malformed masks, duplicate policy, borders, unavailable encoders, and subgroup thresholds are fixed. {detail}

## Final-test policy
No tuning, calibration, selection, corruption, or favourable-result inspection uses the final fold. {detail}

## Amendments
Any later change is a dated, reasoned, checksum-linked amendment and cannot redefine primary results.
"""


def _project_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "src").mkdir()
    preregistration = root / "PRE_REGISTRATION.md"
    preregistration.write_text(_complete_preregistration(), encoding="utf-8")
    independence = root / "reports" / "representation_independence.json"
    independence.parent.mkdir(parents=True)
    pathology = root / "reports" / "pathology_encoder_availability.json"
    pathology.write_text(
        json.dumps(
            {
                "status": "blocked",
                "selected_encoder": None,
                "selection_rule": "first candidate satisfying every frozen availability gate",
                "records": [{"name": "candidate", "status": "blocked"}],
            }
        ),
        encoding="utf-8",
    )
    pathology_sha = sha256_file(pathology)
    primary_config = complete_primary_config()
    primary_config["data"]["exclusions"] = dict(_PANNUKE_EXCLUSION_POLICY)
    primary_config["corruption"]["clean_reference"] = dict(_CLEAN_REFERENCE_CELL)
    generator = FeatureSpaceEvidence(
        representation_name="morphology_only_v1",
        feature_artifact_hash="1" * 64,
        family="morphology",
        implementation_hash="2" * 64,
        weights_hash="3" * 64,
        preprocessing_hash="4" * 64,
        fitted_data_hash="5" * 64,
    )
    entries: dict[str, Any] = {}
    for index, representation in enumerate(primary_config["representations"], start=6):
        if representation["cache_provenance"]["status"] != "available":
            continue
        auditor = FeatureSpaceEvidence(
            representation_name=representation["id"],
            feature_artifact_hash=f"{index:x}" * 64,
            family=representation["family"],
            implementation_hash="a" * 64,
            weights_hash="b" * 64,
            preprocessing_hash="c" * 64,
            fitted_data_hash="5" * 64,
        )
        decision = (
            "not_independent"
            if representation["family"] == "engineered"
            else "verified_independent"
        )
        entries[representation["id"]] = FeatureIndependenceEvidence.create(
            matrix_version="fixture_v1",
            matrix_decision=decision,
            matrix_reason="Deterministic workflow-stage fixture with explicit feature spaces.",
            generator=generator,
            auditor=auditor,
        ).as_dict()
    independence.write_text(
        json.dumps({"schema_version": 2, "entries": entries}, sort_keys=True),
        encoding="utf-8",
    )
    independence_sha = sha256_file(independence)
    primary_config["corruption"]["mechanisms"]["instance_dependent_corruption"].update(
        independence_matrix_path="reports/representation_independence.json",
        independence_matrix_sha256=independence_sha,
    )
    for representation in primary_config["representations"]:
        representation["generator_independence"]["independence_matrix_sha256"] = independence_sha
    primary_config["representations"][-1]["availability_audit_sha256"] = pathology_sha
    confirmatory_config = cross_compatible_confirmatory_config(primary_config)
    confirmatory_config["scenarios"][-1]["availability_audit_sha256"] = pathology_sha
    confirmatory_config["cache_provenance"][-1]["blocker_evidence_sha256"] = pathology_sha
    (root / "configs" / "primary.yaml").write_text(
        yaml.safe_dump(primary_config, sort_keys=True), encoding="utf-8"
    )
    (root / "configs" / "confirmatory.yaml").write_text(
        yaml.safe_dump(confirmatory_config, sort_keys=True), encoding="utf-8"
    )
    dataset = root / "data" / "pannuke"
    dataset.mkdir(parents=True)
    (dataset / "fold.npy").write_bytes(b"verified-pannuke-fixture")
    manifest = dataset / "nuclei.csv"
    manifest.write_text("sample_id,group_id,observed_label\nn0,g0,0\n", encoding="utf-8")
    raw_checksums = dataset / "raw_checksums.json"
    raw_checksums.write_text(
        json.dumps(
            {
                "fold.npy": sha256_file(dataset / "fold.npy"),
                "nuclei.csv": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    duplicate_audit = root / "reports" / "pannuke_duplicate_audit.json"
    duplicate_audit.write_text(
        json.dumps(
            {
                "status": "completed",
                "required_two_signal_near_duplicate_gate_complete": True,
                "policy": {
                    "automatic_deletion": False,
                    "cross_fold_only": True,
                    "candidate_action": "review_only",
                },
                "coverage": {
                    "total_source_patches": 1,
                    "patches_with_full_hash_provenance": 1,
                    "perceptual_comparison_patch_count": 1,
                    "embedding_patch_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return root, dataset, manifest, raw_checksums, duplicate_audit, pathology


def _pilot_authority_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "reports" / "pilot_development_manifest.parquet",
        root / "reports" / "pilot_gate_certificate.json",
    )


def _pilot_authority_kwargs(root: Path) -> dict[str, Path]:
    development, certificate = _pilot_authority_paths(root)
    return {
        "pilot_development_manifest_path": development,
        "pilot_gate_certificate_path": certificate,
    }


@pytest.fixture(autouse=True)
def _stub_expensive_post_seal_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _completed_pilot(root: Path, dataset: Path, manifest: Path, duplicate_audit: Path) -> Path:
    external_development, external_certificate = _pilot_authority_paths(root)
    tracker = RunTracker.start(
        experiment_name="pannuke_pilot",
        config={"experiment_name": "pannuke_pilot", "schema_version": 1},
        project_root=root,
        dataset_path=dataset,
        manifest_path=manifest,
        environment={},
        duplicate_audit_status=f"complete_sha256:{sha256_file(duplicate_audit)}",
    )
    with tracker:
        tracker.write_metrics(
            {
                "artifact_scope": "real_pannuke_controlled_corruption_pilot",
                "completion_stage_if_sealed": "PILOT_COMPLETE",
                "limitations": [
                    "Each potentially inconsistent annotation is recommended for expert review "
                    "and is not a diagnostic output."
                ],
            }
        )
        tracker.write_text(
            "report.md",
            "# Verified pilot\n\nEach potentially inconsistent annotation is recommended for "
            "expert review. This research fixture is not a diagnostic output.\n",
        )
        tracker.write_json(
            "selected_groups_and_samples.json",
            {
                "no_group_overlap_verified": True,
                "final_reference_outcomes_used": False,
                "final_reference_representations_extracted": False,
                "final_reference_sample_ids_read": False,
                "final_reference_class_labels_read": False,
            },
        )
        tracker.write_json("corruption_manifest.json", {"rows": []})
        tracker.write_json("oof_provenance.json", {"folds": []})
        np.savez(tracker.run_directory / "oof_predictions.npz", probabilities=np.ones((1, 5)))
        tracker.write_text("ranking.csv", "sample_id,risk_score\nn0,0.1\n")
        tracker.write_text("cleanlab_evidence.csv", "sample_id,risk_score\nn0,0.1\n")
        tracker.write_json("cleanlab_evidence.json", {"status": "fixture"})
        np.savez(tracker.run_directory / "cleanlab_evidence.npz", risk=np.asarray([0.1]))
        tracker.write_text("neighbour_evidence.csv", "sample_id,risk_score\nn0,0.1\n")
        tracker.write_json("neighbour_evidence.json", {"status": "fixture"})
        np.savez(tracker.run_directory / "neighbour_evidence.npz", risk=np.asarray([0.1]))
        tracker.write_json("audit_evidence_reconciliation.json", {"status": "passed"})
        tracker.write_json(
            "final_reference_privacy_reconciliation.json",
            {
                "status": "passed",
                "policy": "final_reference_identity_and_outcome_nonpublication_v1",
                "final_reference_official_fold": 3,
            },
        )
        development = tracker.write_text("development_manifest_view.parquet", "fixture-view\n")
        certificate = tracker.write_json("pre_pilot_gate_certificate.json", {"status": "passed"})
        external_development.write_bytes(development.read_bytes())
        external_certificate.write_bytes(certificate.read_bytes())
    _bind_pilot_derived_parameters(root, tracker.run_directory)
    return tracker.run_directory


def _bind_pilot_derived_parameters(root: Path, pilot: Path) -> None:
    """Create a compact semantic derivation fixture bound to this sealed pilot."""

    from histo_audit.experiment import pilot_derived_parameters as producer

    integrity = verify_run_integrity(pilot)
    assert integrity.valid
    assert integrity.expected_root_sha256 is not None
    primary_path = root / "configs" / "primary.yaml"
    primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
    mechanisms = primary["corruption"]["mechanisms"]
    record = {
        "schema_version": 1,
        "producer_id": "pilot_derived_primary_parameters_v1",
        "producer_source_sha256": sha256_file(Path(producer.__file__).resolve()),
        "class_order": primary["data"]["class_order"],
        "confusion_targeted_corruption": {
            "transition_matrix": mechanisms["confusion_targeted_corruption"]["transition_matrix"]
        },
        "group_conditional_corruption": {
            field: mechanisms["group_conditional_corruption"][field]
            for field in ("grouping_field", "weights_by_value", "default_weight")
        },
        "source_pilot": {
            "run_id": pilot.name,
            "artifact_root_sha256": integrity.expected_root_sha256,
            "development_official_folds": primary["data"]["development_official_folds"],
            "final_reference_policy": (
                "no final-fold sample identifier, label, representation, or outcome read"
            ),
        },
    }
    derived_path = root / "reports" / "pilot_derived_primary_parameters.json"
    derived_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    primary["pilot_derived_parameters"].update(
        sha256=sha256_file(derived_path),
        source_pilot_run_id=pilot.name,
        source_pilot_artifact_root_sha256=integrity.expected_root_sha256,
    )
    primary_path.write_text(yaml.safe_dump(primary, sort_keys=True), encoding="utf-8")


def test_preregistration_freeze_refuses_unsealed_pilot(tmp_path: Path) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    unsealed = root / "artifacts" / "runs" / "pannuke-pilot-running"
    unsealed.mkdir(parents=True)

    with pytest.raises(ValueError, match="integrity verification"):
        freeze_preregistration(
            project_root=root,
            pilot_run_directory=unsealed,
            dataset_path=dataset,
            manifest_path=manifest,
            raw_checksum_manifest_path=raw_checksums,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
            **_pilot_authority_kwargs(root),
        )

    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()


def test_preregistration_freeze_succeeds_once_and_detects_tampering(tmp_path: Path) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    timestamp = datetime(2026, 7, 17, 12, 30, tzinfo=UTC)

    result = freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        **_pilot_authority_kwargs(root),
        timestamp=timestamp,
    )

    verification = verify_preregistration_freeze(
        result.freeze_directory,
        frozen_primary_config_path=result.frozen_primary_config_path,
        frozen_confirmatory_config_path=result.frozen_confirmatory_config_path,
    )
    assert verification.valid
    assert result.pilot_run_id == pilot.name
    assert result.frozen_primary_config_path.is_file()
    assert result.frozen_confirmatory_config_path.is_file()
    assert result.duplicate_audit_snapshot_path.is_file()
    assert result.pathology_encoder_audit_snapshot_path.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_preregistration(
            project_root=root,
            pilot_run_directory=pilot,
            dataset_path=dataset,
            manifest_path=manifest,
            raw_checksum_manifest_path=raw_checksums,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
            **_pilot_authority_kwargs(root),
            timestamp=timestamp,
        )

    result.frozen_preregistration_path.write_text("tampered\n", encoding="utf-8")
    tampered = verify_preregistration_freeze(result.freeze_directory)
    assert not tampered.valid
    assert "PRE_REGISTRATION_FROZEN.md" in tampered.changed_paths


def test_preregistration_freeze_rejects_withdrawn_completed_pilot(tmp_path: Path) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    withdraw_run_eligibility(
        pilot,
        reason_code="post_seal_pilot_audit",
        reason="A post-seal audit made this completed pilot ineligible for stage progression.",
    )

    with pytest.raises(ValueError, match="scientific stage eligibility was permanently withdrawn"):
        freeze_preregistration(
            project_root=root,
            pilot_run_directory=pilot,
            dataset_path=dataset,
            manifest_path=manifest,
            raw_checksum_manifest_path=raw_checksums,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
            **_pilot_authority_kwargs(root),
            timestamp=datetime(2026, 7, 17, 12, 45, tzinfo=UTC),
        )

    assert not (root / "artifacts" / "preregistrations").exists()


def test_primary_gate_binds_both_plans_and_every_real_data_dependency(tmp_path: Path) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    freeze = freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        **_pilot_authority_kwargs(root),
        timestamp=datetime(2026, 7, 17, 13, 0, tzinfo=UTC),
    )

    gate = validate_primary_execution_gate(
        project_root=root,
        freeze_directory=freeze.freeze_directory,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
    )

    assert gate.primary_matrix_cell_count == 222
    assert gate.primary_required_cell_count == 185
    assert gate.confirmatory_matrix_cell_count == 108
    assert gate.pilot_run_id == pilot.name
    assert gate.duplicate_audit_sha256 == sha256_file(duplicate_audit)

    (root / "src" / "outcome_dependent_change.py").write_text("changed = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source tree differs"):
        validate_primary_execution_gate(
            project_root=root,
            freeze_directory=freeze.freeze_directory,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
        )


def test_primary_gate_rejects_pilot_withdrawn_after_freeze(tmp_path: Path) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    freeze = freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        **_pilot_authority_kwargs(root),
        timestamp=datetime(2026, 7, 17, 13, 15, tzinfo=UTC),
    )
    withdraw_run_eligibility(
        pilot,
        reason_code="post_freeze_pilot_audit",
        reason="A later audit permanently withdrew this pilot before primary execution.",
    )

    with pytest.raises(ValueError, match="scientific stage eligibility was permanently withdrawn"):
        validate_primary_execution_gate(
            project_root=root,
            freeze_directory=freeze.freeze_directory,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
        )


def test_confirmatory_gate_rejects_synthetic_or_ineligible_primary_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    freeze = freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        **_pilot_authority_kwargs(root),
        timestamp=datetime(2026, 7, 17, 13, 30, tzinfo=UTC),
    )
    gate = validate_primary_execution_gate(
        project_root=root,
        freeze_directory=freeze.freeze_directory,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
    )
    tracker = RunTracker.start(
        experiment_name="primary_synthetic_integration_fixture",
        config={"experiment_name": "primary_synthetic_integration_fixture"},
        project_root=root,
        environment={},
    )
    with tracker:
        tracker.write_json("matrix_plan.json", {"cell_count": gate.primary_matrix_cell_count})
        tracker.write_json("reconciliation.json", {"status": "passed"})
        tracker.write_metrics({"artifact_scope": "synthetic_primary_orchestrator_fixture"})
        tracker.write_text("report.md", "# Synthetic fixture — not study evidence\n")
        tracker.write_json(
            "completion_evidence.json",
            {
                "completion_stage": None,
                "study_outcome_eligible": False,
                "artifact_scope": "synthetic_primary_orchestrator_fixture",
                "required_cell_count": gate.primary_required_cell_count,
                "completed_required_cell_count": gate.primary_required_cell_count,
                "failed_required_cell_count": 0,
            },
        )

    with pytest.raises(ValueError, match="PRIMARY_STUDY_COMPLETE"):
        validate_confirmatory_execution_gate(
            primary_run_directory=tracker.run_directory,
            project_root=root,
            freeze_directory=freeze.freeze_directory,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
        )

    withdraw_run_eligibility(
        tracker.run_directory,
        reason_code="post_primary_audit",
        reason="A later audit permanently withdrew this primary run before confirmation.",
    )
    from histo_audit.workflows import study_gates

    original_read_mapping = study_gates._read_mapping
    primary_outcome_reads: list[Path] = []

    def _record_primary_outcome_reads(path: Path, role: str) -> Mapping[str, Any]:
        resolved = Path(path).resolve()
        if resolved.parent == tracker.run_directory.resolve():
            primary_outcome_reads.append(resolved)
        return original_read_mapping(path, role)

    monkeypatch.setattr(study_gates, "_read_mapping", _record_primary_outcome_reads)
    with pytest.raises(ValueError, match="scientific stage eligibility was permanently withdrawn"):
        validate_confirmatory_execution_gate(
            primary_run_directory=tracker.run_directory,
            project_root=root,
            freeze_directory=freeze.freeze_directory,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology,
        )
    assert primary_outcome_reads == []


def _original_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_index in range(10):
        for within_group in range(4):
            label = (group_index + within_group) % 2
            rows.append(
                {
                    "sample_id": f"sample-{group_index:02d}-{within_group}",
                    "group_id": f"group-{group_index:02d}",
                    "tissue_type": "tissue-a" if group_index < 5 else "tissue-b",
                    "pre_corruption_label": label,
                    "observed_label": label,
                    "is_injected_corruption": False,
                }
            )
    return pd.DataFrame(rows)


def test_original_label_audit_is_group_safe_ranked_and_read_only(tmp_path: Path) -> None:
    frame = _original_manifest()
    manifest_path = tmp_path / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    before = sha256_file(manifest_path)
    rng = np.random.default_rng(41)
    features = np.column_stack(
        [frame["observed_label"].to_numpy(dtype=float), rng.normal(size=len(frame))]
    )
    permutation = rng.permutation(len(frame))
    feature_ids = frame["sample_id"].to_numpy()[permutation].tolist()

    result = audit_original_labels(
        manifest_path,
        features[permutation],
        feature_ids,
        tmp_path / "original-audit",
        final_reference_group_ids={"final-group-00", "final-group-01"},
        class_order=(0, 1),
        n_splits=5,
        top_count_overall=5,
        top_count_per_class=2,
        top_count_per_tissue=3,
        balanced_top_count=6,
        balanced_max_per_group=1,
        balanced_max_per_class=3,
        balanced_max_per_tissue=3,
        balanced_max_per_transition=3,
        balanced_minimum_cosine_distance=0.0,
    )

    assert sha256_file(manifest_path) == before
    assert result.sample_count == 40
    assert result.top_overall_count == 5
    assert result.top_per_class_count == 4
    assert result.top_per_tissue_count == 6
    assert result.balanced_quality_count == 6
    assert result.balanced_quality_underfilled is False
    assert result.balanced_quality_queue_path is not None
    assert result.balanced_quality_queue_path.is_file()
    assert result.balanced_queue_evidence_path is not None
    balanced_evidence = json.loads(result.balanced_queue_evidence_path.read_text(encoding="utf-8"))
    assert balanced_evidence["model_improvement_queue"]["available"] is False
    assert max(balanced_evidence["group_counts"].values()) == 1
    ranking = pd.read_csv(result.ranking_all_path)
    assert ranking["risk_score"].is_monotonic_decreasing
    assert "tissue_type" in ranking
    assert "tissue" not in ranking
    assert set(ranking["review_recommendation"]) == {
        "recommended for expert review as a potentially inconsistent annotation"
    }
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["automatic_source_annotation_modification"] is False
    assert metadata["injected_corruption_count"] == 0
    assert metadata["group_safe_oof"]["coverage_exactly_once"] is True
    assert "not a diagnostic system" in result.report_path.read_text(encoding="utf-8")


def test_original_label_audit_rejects_injected_corruption(tmp_path: Path) -> None:
    frame = _original_manifest()
    frame.loc[0, "is_injected_corruption"] = True
    features = np.ones((len(frame), 2), dtype=np.float64)

    with pytest.raises(ValueError, match="injected corruption"):
        audit_original_labels(
            frame,
            features,
            frame["sample_id"].tolist(),
            tmp_path / "refused-audit",
            final_reference_group_ids={"final-group"},
            class_order=(0, 1),
            n_splits=5,
        )


def test_original_label_audit_can_persist_fold_safe_neighbour_evidence(tmp_path: Path) -> None:
    frame = _original_manifest()
    rng = np.random.default_rng(53)
    features = np.column_stack(
        [frame["observed_label"].to_numpy(dtype=float), rng.normal(size=len(frame))]
    )

    result = audit_original_labels(
        frame,
        features,
        frame["sample_id"].tolist(),
        tmp_path / "neighbour-audit",
        final_reference_group_ids={"final-group"},
        class_order=(0, 1),
        n_splits=5,
        method="nearest_neighbour_disagreement",
        neighbour_k=3,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    evidence = json.loads(
        (result.output_directory / "neighbour_evidence.json").read_text(encoding="utf-8")
    )
    assert metadata["risk_method"] == "nearest_neighbour_disagreement"
    assert metadata["risk_strategy"]["neighbour_k"] == 3
    assert len(evidence["records"]) == len(frame)
    group_by_sample = dict(zip(frame["sample_id"], frame["group_id"], strict=True))
    for record in evidence["records"]:
        assert group_by_sample[record["sample_id"]] not in record["neighbour_groups"]


def test_original_label_audit_rejects_conflicting_tissue_aliases(tmp_path: Path) -> None:
    frame = _original_manifest()
    frame["tissue"] = frame["tissue_type"]
    frame.loc[0, "tissue"] = "conflicting-tissue"

    with pytest.raises(ValueError, match="tissue_type and legacy tissue columns conflict"):
        audit_original_labels(
            frame,
            np.ones((len(frame), 2), dtype=np.float64),
            frame["sample_id"].tolist(),
            tmp_path / "refused-conflict",
            final_reference_group_ids={"final-group"},
            class_order=(0, 1),
            n_splits=5,
        )


def test_blinded_review_package_validation_checks_private_linkage(tmp_path: Path) -> None:
    image = tmp_path / "asset.png"
    image.write_bytes(b"minimal fixture image bytes")
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(6)],
            "observed_label": [index % 2 for index in range(6)],
            "full_patch_path": [str(image)] * 6,
            "target_crop_path": [str(image)] * 6,
            "target_contour_path": [str(image)] * 6,
        }
    )
    ranking = pd.DataFrame(
        {"sample_id": [f"s{index}" for index in range(6)], "risk_score": np.linspace(1, 0, 6)}
    )
    package = build_blinded_review_package(
        manifest,
        ranking,
        tmp_path / "review-package",
        top_count=2,
        random_count=2,
        seed=17,
    )

    validation = validate_blinded_review_package(
        package.package_directory,
        private_unblinding_key_path=package.private_unblinding_key_csv,
    )
    assert validation.valid
    assert validation.private_linkage_validated
    assert validation.item_count == 4
    assert validation.asset_count == 12

    responses = pd.read_csv(package.response_template_csv, dtype=str).fillna("")
    responses.loc[0, "response"] = "probably_inconsistent"
    responses.to_csv(package.response_template_csv, index=False)
    invalid = validate_blinded_review_package(
        package.package_directory,
        private_unblinding_key_path=package.private_unblinding_key_csv,
    )
    assert not invalid.valid
    assert any("not blank" in error for error in invalid.errors)
