"""Tests for the small, read-only AANCA presentation package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.request import urlopen

import pytest
from typer.testing import CliRunner

from histo_audit.cli import app
from histo_audit.mvp_demo import (
    build_mvp_presentation,
    create_mvp_http_server,
    verify_mvp_presentation,
)


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
    readme = artifacts.readme_path.read_text(encoding="utf-8")
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
    assert 'href="#author">Author</a>' in html
    assert 'id="author" aria-labelledby="author-title"' in html
    assert "Research and implementation by Natan Smogór" in html
    assert "Management and Artificial Intelligence" in html
    assert "Kozminski University" in html
    assert "Current student" in html
    assert "Uniwersytet Młodzieżowy" in html
    assert "80-hour Artificial Intelligence programme" in html
    assert "2024/2025 academic year" in html
    assert "Completion diploma" in html
    assert "do not imply institutional endorsement of AANCA" in html
    assert "18 August 2026" in html
    assert "gsap@3.15.0" in html
    assert "three@0.185.1" in html
    assert "import('https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js')" in html
    assert "DEMO_COMPLETE" not in html
    assert "PRIMARY_STUDY_COMPLETE" not in html
    assert "amended_or_exploratory" not in html
    assert 'id="hero-canvas"' in html
    assert "#hero-canvas[hidden]" in html
    assert ">Results</a>" in html
    assert "Source annotations stay fixed" in html
    assert "Conceptual workflow · not benchmark data" in html
    assert "threejs-review-queue" in html
    assert "immutable-source-ranked-review" in html
    assert "SOURCE PATCH" in html
    assert "REVIEW QUEUE" in html
    assert 'class="journey story"' in html
    assert 'class="journey-connector"' in html
    assert "One controlled question, unfolded step by step" in html
    assert "Saved performance estimates varied across contexts" in html
    assert "What the study actually learned" in html
    assert html.count('<article class="hypothesis-row" data-learned-slide>') == 7
    assert 'aria-roledescription="carousel"' in html
    assert 'aria-roledescription="slide"' not in html
    assert "let learnedSlideThresholds = [0]" in html
    assert "let learnedSettleThresholds = [0]" in html
    assert "learnedSlideStarts.map(start => start / learnedDuration)" in html
    assert "learnedSettleTimes.map(time => time / learnedDuration)" in html
    assert "const activationDistance = beforeFirstAnswer ? 60 : 360" in html
    assert "learnedStory.addEventListener('wheel', onLearnedWheel, {passive: false})" in html
    assert "height: 980vh" in html
    assert "scrub: true" in html
    assert "filter: 'blur(4px)'" not in html
    assert "Editorial surfaces: structure with rhythm and rules" in html
    assert ".repo-card::before { display: none; }" in html
    assert "max-height: min(60svh,620px)" in html
    assert "max-height: min(62svh,680px)" in html
    assert 'aria-label="Scrollable forest plot' in html
    assert 'comparisons" tabindex="0"' in html
    assert "html.motion-enhanced .learned-story" in html
    assert "html.motion-enhanced .learned-word" in html
    assert 'id="learned-current"' not in html
    assert 'class="learned-progress"' not in html
    assert 'class="hypothesis-row-head"' not in html
    assert 'class="learned-answer-label"' not in html
    assert "Move through one registered question at a time" not in html
    assert "12 / 12 positive differences" not in html
    assert "Automated Auditing of Nucleus Class Annotations" in html
    assert "Automated nucleus-annotation auditing" not in html
    assert "--prose: 640px" in html
    assert "--editorial: 640px" in html
    assert "\N{EM DASH}" not in html
    assert "\N{EN DASH}" not in html
    assert "The design limits outcome-informed model selection" in html
    assert "33 reported · 3 unavailable" in html
    assert "github.com/Jaqwilk/AANCA" in html
    assert "View repository" in html
    assert 'class="brand-label"' in html
    assert '<div class="reading-progress"' not in html
    assert "<details" not in html
    assert 'class="forest-plot"' in html
    assert 'role="group" aria-label="Complete registered H4 downstream result"' in html
    assert 'id="filter-hypothesis"' in html
    assert "prefers-reduced-motion" in html
    assert "document.addEventListener('visibilitychange', syncRenderLoop)" in html
    assert "canvas.dataset.animationState = 'paused'" in html
    assert "powerPreference: 'low-power'" in html
    assert 'fetchpriority="low" width="1512" height="3840"' in html
    assert '<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>' in html
    assert "python scripts/present_demo.py" in html
    assert "Repository links require reviewer access" in readme
    assert ".journey-stage-group, .journey-connector, .journey-step" in html
    assert "html:not(.motion-enhanced) .journey" in html
    assert "const journeyTrigger = scrollEngine.create" in html
    assert "Math.min(1, journeyTrigger.progress)" in html
    assert 'role="region" aria-label="Complete comparison results" tabindex="0"' in html
    assert "stroke-dashoffset: 0 !important" in html
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


def test_verify_mvp_rejects_duplicate_manifest_paths(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    artifacts = build_mvp_presentation(
        project_root=tmp_path,
        run_directory=run,
        qc_bundle_directory=qc,
        output_directory=Path("artifacts/mvp_demo"),
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0] = dict(manifest["files"][1])
    manifest["manifest_root_sha256"] = _canonical_sha256(manifest["files"])
    _write_json(artifacts.manifest_path, manifest)

    with pytest.raises(ValueError, match="record allowlist differs"):
        verify_mvp_presentation(artifacts.output_directory)


def test_verified_mvp_http_server_serves_only_the_closed_package(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    artifacts = build_mvp_presentation(
        project_root=tmp_path,
        run_directory=run,
        qc_bundle_directory=qc,
        output_directory=Path("artifacts/mvp_demo"),
    )
    server, verification = create_mvp_http_server(
        artifacts.output_directory,
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            payload = response.read()
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Content-Security-Policy"] == (
                "base-uri 'none'; frame-ancestors 'none'; object-src 'none'"
            )
            assert response.headers["Permissions-Policy"] == (
                "camera=(), geolocation=(), microphone=()"
            )
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["Server"] == "AANCA"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
        assert payload.startswith(b"<!doctype html>")
        assert verification["status"] == "valid"
        assert verification["file_count"] == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_standalone_presentation_launcher_needs_no_project_import(tmp_path: Path) -> None:
    run, qc = _make_sources(tmp_path)
    artifacts = build_mvp_presentation(
        project_root=tmp_path,
        run_directory=run,
        qc_bundle_directory=qc,
        output_directory=Path("artifacts/mvp_demo"),
    )
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "present_demo.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(launcher),
            "--verify-only",
            "--output-dir",
            str(artifacts.output_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "valid"' in result.stdout
    assert artifacts.manifest_root_sha256 in result.stdout


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
