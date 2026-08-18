"""Tests for the small, read-only AANCA presentation package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.mvp_demo import build_mvp_presentation, verify_mvp_presentation


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _make_sources(root: Path) -> tuple[Path, Path]:
    run = root / "artifacts" / "runs" / "accepted_primary_fixture"
    run.mkdir(parents=True)
    comparisons: list[dict[str, Any]] = []
    for family, count in (("h1", 12), ("h3", 6), ("h5", 12), ("h6", 3), ("h7", 3)):
        for index in range(count):
            unavailable = family == "h6"
            difference = 0.02 + index / 1000
            interval = [difference - 0.01, difference + 0.01]
            p_value = 0.01
            if family == "h3" and index == 0:
                interval = [-0.001, difference + 0.01]
            if family == "h7":
                difference = (1 - index) / 1000
                interval = [-0.01, 0.01]
                p_value = 0.8
            comparisons.append(
                {
                    "comparison_id": f"{family}_fixture_{index:02d}",
                    "status": ("not_available_frozen_optional_cell" if unavailable else "reported"),
                    "method_a": "auditor",
                    "method_b": "baseline",
                    "metric": "average_precision",
                    "point_difference": None if unavailable else difference,
                    "interval_95": None if unavailable else interval,
                    "p_value_holm": None if unavailable else p_value,
                    "valid_bootstrap_iterations": 0 if unavailable else 2000,
                }
            )

    instance_cells = [
        {
            "cell": {
                "cell_id": f"primary_instance_fixture_{seed}",
                "classifier_id": "multinomial_logistic_regression",
                "corruption_seed": seed,
                "mechanism": "instance_dependent_corruption",
                "rate": 0.1,
                "representation_id": "imagenet_resnet18_context",
                "required": True,
                "scenario_id": f"instance_fixture_{seed}",
            }
        }
        for seed in (404, 405, 406)
    ]
    restoration_relative = "restorations/primary_0027_fixture/restoration.json"
    restoration_payload = {
        "cell": {
            "cell_id": "primary_0027_fixture",
            "classifier_id": "multinomial_logistic_regression",
            "corruption_seed": 404,
            "mechanism": "symmetric_random_corruption",
            "rate": 0.1,
            "representation_id": "imagenet_resnet18_highlighted",
            "required": True,
            "scenario_id": "restoration_fixture",
        },
        "downstream_comparisons": [
            {
                "comparison_id": "audit_guided_minus_random_macro_f1",
                "status": "reported",
                "metric": "macro_f1",
                "point_difference": -0.002,
                "point_metric_a": 0.524,
                "point_metric_b": 0.526,
                "interval_95": [-0.003, -0.001],
                "probability_positive": 0.0,
                "random_repetitions": 100,
            }
        ],
        "evaluation": {
            "corrupted_observed_baseline": {"metrics": {"macro_f1": 0.5265}},
            "uncorrupted_reference_baseline": {"metrics": {"macro_f1": 0.5375}},
        },
    }
    _write_json(run / restoration_relative, restoration_payload)
    restoration_sha = hashlib.sha256((run / restoration_relative).read_bytes()).hexdigest()
    payloads: dict[str, Any] = {
        "completion_evidence.json": {
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "analysis_disposition": "amended_or_exploratory",
            "completed_required_cell_count": 185,
            "failed_required_cell_count": 0,
            "skipped_optional_cell_count": 37,
            "training_invoked": False,
            "fallback_invoked": False,
            "automatic_retry_allowed": False,
            "primary_statistics_verification_status": "passed",
            "primary_restoration_verification_status": "passed",
            "primary_statistics_comparison_count": 36,
            "outcomes_inspected": True,
        },
        "metrics.json": {
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "analysis_disposition": "amended_or_exploratory",
        },
        "primary_recovery_evidence.json": {"retrained_cell_count": 0},
        "primary_recovery_statistics_verification.json": {"verification": {"status": "passed"}},
        "primary_statistics.json": {
            "comparisons": comparisons,
            "cells": instance_cells,
            "bootstrap": {
                "requested_iterations": 2000,
                "saved_draw_count": 2000,
                "resampling_scope": "whole_groups_only",
                "unit": "source_patch_id",
            },
            "multiple_comparison_correction": {
                "method": "holm",
                "families": [
                    "h1_method_vs_random",
                    "h3_mechanism_hardness",
                    "h5_fixed_hybrid",
                    "h6_encoder_family",
                    "h7_target_indication",
                ],
                "one_sided_p_value_definition": (
                    "(1 + count(bootstrap_difference <= 0)) / (1 + valid_iterations)"
                ),
            },
            "subgroups": {"row_count": 4},
        },
        "restoration_index.json": {
            "schema_version": 1,
            "restoration_cell_count": 1,
            "restoration_cell_ids": ["primary_0027_fixture"],
            "cells": [
                {
                    "cell": restoration_payload["cell"],
                    "json_path": restoration_relative,
                    "json_sha256": restoration_sha,
                    "ranking_method": "self_confidence",
                }
            ],
            "downstream_comparisons": [
                {
                    "comparison_id": "audit_guided_minus_random_macro_f1",
                    "method_a": "audit_guided_restoration",
                    "method_b": "random_review_restoration",
                    "metric": "macro_f1",
                }
            ],
        },
        "reconciliation.json": {"status": "passed"},
        "status.json": {"status": "completed"},
    }
    for relative, payload in payloads.items():
        _write_json(run / relative, payload)
    (run / "report.md").write_text("# Accepted primary fixture\n", encoding="utf-8")
    (run / "primary_subgroups.csv").write_text(
        "cell_id,scenario_id,method,dimension,value,sample_count,"
        "injected_corruption_count,average_precision_status,average_precision,"
        "suppression_reason\n"
        "cell,scenario,self_confidence,class,0,100,10,reported,0.3,\n"
        "cell,scenario,self_confidence,tissue,Breast,100,10,reported,0.4,\n"
        "cell,scenario,self_confidence,mechanism,symmetric,100,10,reported,0.5,\n"
        "cell,scenario,self_confidence,rate,0.1,100,10,reported,0.6,\n",
        encoding="utf-8",
    )
    for seed in (404, 405, 406):
        cell_directory = run / "cells" / f"primary_instance_fixture_{seed}"
        cell_directory.mkdir(parents=True)
        (cell_directory / "ranking.csv").write_bytes(b"identical-ranking-fixture\n")
        (cell_directory / "oof_predictions.npz").write_bytes(b"identical-oof-fixture")
    statistics_record = _file_record(run / "primary_statistics.json", "primary_statistics.json")
    subgroup_record = _file_record(run / "primary_subgroups.csv", "primary_subgroups.csv")
    _write_json(
        run / "primary_statistics_manifest.json",
        {"artifacts": [statistics_record, subgroup_record]},
    )

    seed_evidence = tuple(
        f"cells/primary_instance_fixture_{seed}/{filename}"
        for seed in (404, 405, 406)
        for filename in ("ranking.csv", "oof_predictions.npz")
    )
    selected = sorted(
        (
            *payloads,
            "report.md",
            "primary_statistics_manifest.json",
            "primary_subgroups.csv",
            restoration_relative,
            *seed_evidence,
        )
    )
    records = [_file_record(run / relative, relative) for relative in selected]
    artifact_root = "a" * 64
    manifest = {
        "run_id": run.name,
        "status": "completed",
        "artifact_count": len(records),
        "artifact_root_sha256": artifact_root,
        "artifacts": records,
    }
    _write_json(run / "artifact_manifest.json", manifest)
    manifest_sha = hashlib.sha256((run / "artifact_manifest.json").read_bytes()).hexdigest()
    immutable = {
        "run_id": run.name,
        "status": "completed",
        "artifact_count": len(records),
        "artifact_root_sha256": artifact_root,
        "artifact_manifest_sha256": manifest_sha,
    }
    _write_json(run / ".immutable.json", immutable)

    stage_unsigned = {
        "artifact_manifest_sha256": manifest_sha,
        "artifact_root_sha256": artifact_root,
        "completion_stage": "PRIMARY_STUDY_COMPLETE",
        "event_type": "postseal_stage_eligibility_attested",
        "previous_record_sha256": None,
        "run_id": run.name,
        "scientific_stage_eligible": True,
        "verification_sha256": "b" * 64,
    }
    stage = {**stage_unsigned, "record_sha256": _canonical_sha256(stage_unsigned)}
    ledger = run.parent / "run_stage_attestations.jsonl"
    ledger.write_text(
        json.dumps(stage, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _write_json(
        run.parent / "run_stage_attestations.anchor.json",
        {
            "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "record_count": 1,
            "head_record_sha256": stage["record_sha256"],
        },
    )

    qc = root / "reports" / "pannuke_qc"
    qc.mkdir(parents=True)
    qc_payload = {
        "source_masks_modified": False,
        "selection_sha256": "c" * 64,
        "qc_policy": {
            "no_class_arbitration": True,
            "supplied_background_is_exact_complement_required": False,
        },
        "global_mask_qc": {
            "fold_count": 3,
            "patch_count": 7901,
            "cross_class_overlap_pixel_count": 4318,
            "cross_class_overlap_patch_count": 575,
            "void_pixel_count": 10486091,
            "void_patch_count": 162,
            "overlap_touching_instance_count": 1411,
        },
    }
    _write_json(qc / "pannuke_mask_qc.json", qc_payload)
    (qc / "pannuke_mask_qc_overlays.png").write_bytes(b"synthetic-png-fixture")
    qc_records = {
        relative: {
            "sha256": _file_record(qc / relative, relative)["sha256"],
            "size_bytes": (qc / relative).stat().st_size,
        }
        for relative in ("pannuke_mask_qc.json", "pannuke_mask_qc_overlays.png")
    }
    _write_json(
        qc / "artifact_manifest.json",
        {
            "files": qc_records,
            "overlay_sha256": qc_records["pannuke_mask_qc_overlays.png"]["sha256"],
        },
    )
    return run, qc


def test_build_and_verify_mvp_is_read_only_and_complete(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for base in (run, qc)
        for path in base.rglob("*")
        if path.is_file()
    }

    artifacts = build_mvp_presentation(
        project_root=tmp_path,
        run_directory=run,
        qc_bundle_directory=qc,
        output_directory=Path("artifacts/mvp_demo"),
    )

    evidence = json.loads(artifacts.evidence_path.read_text(encoding="utf-8"))
    html = artifacts.html_path.read_text(encoding="utf-8")
    assert evidence["presentation_status"] == "DEMO_COMPLETE"
    assert evidence["scientific_status"] == "PRIMARY_STUDY_COMPLETE"
    assert evidence["analysis_disposition"] == "amended_or_exploratory"
    assert evidence["confirmatory_completed"] is False
    assert evidence["external_validation_completed"] is False
    assert len(evidence["primary"]["comparisons"]) == 36
    assert evidence["primary"]["h2_subgroups"]["reported_count"] == 4
    assert (
        evidence["primary"]["h4_restoration"]["directional_result"]
        == "adverse_to_registered_hypothesis"
    )
    assert evidence["primary"]["h4_restoration"]["registered_hypothesis_supported"] is False
    assert (
        evidence["primary"]["instance_dependent_seed_audit"]["independent_corruption_realisations"]
        is False
    )
    assert evidence["primary"]["inference"]["p_value_sidedness"] == "one_sided"
    assert "potentially inconsistent annotation" in html
    assert "recommended for expert review" in html
    assert "Better triage did not improve the downstream model" in html
    assert "Holm-adjusted p" in html
    assert "not independent realisations" in html
    assert "Natan Smogór" in html
    assert "18 August 2026" in html
    assert "gsap@3.15.0" in html
    assert "three@0.185.1" in html
    assert "DEMO_COMPLETE" not in html
    assert "PRIMARY_STUDY_COMPLETE" not in html
    assert "amended_or_exploratory" not in html
    assert 'id="hero-canvas"' in html
    assert "Source annotations stay fixed" in html
    assert "Conceptual workflow · not benchmark data" in html
    assert "threejs-review-queue" in html
    assert "immutable-source-ranked-review" in html
    assert "SOURCE PATCH" in html
    assert "REVIEW QUEUE" in html
    assert 'class="story"' in html
    assert 'class="forest-plot"' in html
    assert 'id="filter-hypothesis"' in html
    assert "prefers-reduced-motion" in html
    assert ".story-step { opacity: 1 !important; }" in html
    assert ".workflow-panel [data-stage] { opacity: 1 !important;" in html
    assert verify_mvp_presentation(artifacts.output_directory)["status"] == "valid"
    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes
    }

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        build_mvp_presentation(
            project_root=tmp_path,
            run_directory=run,
            qc_bundle_directory=qc,
            output_directory=artifacts.output_directory,
        )


def test_verify_mvp_rejects_tampering(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    artifacts = build_mvp_presentation(
        project_root=tmp_path,
        run_directory=run,
        qc_bundle_directory=qc,
        output_directory=Path("artifacts/mvp_demo"),
    )
    artifacts.html_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its seal"):
        verify_mvp_presentation(artifacts.output_directory)


def test_mvp_cli_build_and_verify(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "demo",
            "build",
            "--project-root",
            str(tmp_path),
            "--run-dir",
            str(run),
            "--qc-bundle",
            str(qc),
            "--output-dir",
            "artifacts/cli_mvp",
        ],
    )
    assert build.exit_code == 0, build.output
    assert '"status": "built_and_verified"' in build.output

    verify = runner.invoke(
        app,
        ["demo", "verify", "--output-dir", str(tmp_path / "artifacts" / "cli_mvp")],
    )
    assert verify.exit_code == 0, verify.output
    assert '"status": "valid"' in verify.output
