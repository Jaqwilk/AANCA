"""Filesystem-backed primary statistics and tamper-evidence tests."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_study_contracts import complete_primary_config

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_core import (
    PrimaryExecutionControls,
    primary_execution_controls_from_frozen_config,
)
from histo_audit.experiment.primary_statistics import (
    aggregate_primary_statistics,
    verify_primary_statistics_artifacts,
)
from histo_audit.experiment.study_contracts import (
    PrimaryCell,
    PrimaryScenario,
    _expand_primary_matrix_components,
)
from histo_audit.statistics.review import rank_indices
from histo_audit.utils.run_tracking import sha256_file

_ARTIFACT_NAMES = (
    "bootstrap_evidence.npz",
    "cleanlab_evidence.json",
    "cleanlab_evidence.npz",
    "corruption_manifest.json",
    "independence_evidence.json",
    "metrics.json",
    "neighbour_evidence.npz",
    "oof_predictions.npz",
    "oof_provenance.json",
    "ranking.csv",
    "risk_scores.npz",
)

_REPRESENTATION_A = "engineered_target_features"
_REPRESENTATION_B = "imagenet_resnet18_highlighted"
_CLASSIFIER = "multinomial_logistic_regression"
_MECHANISM = "symmetric_random_corruption"
_RATE = 0.2
_SEED = 404


def _selector(
    representation_id: str,
    *,
    rate: float = _RATE,
    seed: int = _SEED,
) -> dict[str, object]:
    return {
        "mechanism": _MECHANISM,
        "rate": rate,
        "seed": seed,
        "representation_id": representation_id,
        "classifier_id": _CLASSIFIER,
    }


def _frozen_config(*, cross_scenario: bool = False) -> dict[str, Any]:
    """Return a minimal, fully valid schema-v2 configuration for statistics tests."""

    config = complete_primary_config()
    config["representations"] = [
        config["representations"][0],
        config["representations"][2],
    ]
    for representation in config["representations"]:
        representation["classifiers"] = [_CLASSIFIER]
    selector_a = _selector(_REPRESENTATION_A)
    zero_event_selector = _selector(_REPRESENTATION_A, rate=0.05)
    selector_b = (
        _selector(_REPRESENTATION_A, rate=0.10) if cross_scenario else _selector(_REPRESENTATION_B)
    )
    config["statistics"] = {
        "paired_group_bootstrap_iterations": 2000,
        "bootstrap_seed": 431,
        "holm_families": ["primary_ranking"],
        "within_cell_comparisons": [
            {
                "comparison_id": "within_self_vs_nll",
                "selector": selector_a,
                "method_a": "self_confidence",
                "method_b": "negative_log_likelihood",
                "metric": "average_precision",
                "direction": "method_a_minus_method_b",
                "holm_family": "primary_ranking",
            },
            {
                "comparison_id": "zero_event_self_vs_nll",
                "selector": zero_event_selector,
                "method_a": "self_confidence",
                "method_b": "negative_log_likelihood",
                "metric": "average_precision",
                "direction": "method_a_minus_method_b",
                "holm_family": "primary_ranking",
            },
        ],
        "method_vs_random_comparisons": [
            {
                "comparison_id": "self_vs_random",
                "selector": selector_a,
                "method_a": "self_confidence",
                "method_b": "random_review",
                "metric": "precision_at_budget",
                "review_budget": 0.05,
                "direction": "method_a_minus_method_b",
                "holm_family": "primary_ranking",
            }
        ],
        "cross_cell_comparisons": [
            {
                "comparison_id": "representation_a_vs_b",
                "selector_a": selector_a,
                "selector_b": selector_b,
                "method_a": "self_confidence",
                "method_b": "self_confidence",
                "metric": "average_precision",
                "direction": "method_a_minus_method_b",
                "holm_family": "primary_ranking",
            }
        ],
        "exploratory_multiple_comparison_correction": "holm",
    }
    _, cells = _expand_primary_matrix_components(config)
    restoration_cell = next(
        cell
        for cell in cells
        if cell.mechanism == _MECHANISM
        and cell.rate == _RATE
        and cell.corruption_seed == _SEED
        and cell.representation_id == _REPRESENTATION_B
        and cell.classifier_id == _CLASSIFIER
    )
    config["restoration"]["enabled_cells"] = [restoration_cell.cell_id]
    return config


def _controls() -> PrimaryExecutionControls:
    return primary_execution_controls_from_frozen_config(_frozen_config())


def _comparison_cell(
    controls: PrimaryExecutionControls,
    representation_id: str,
) -> PrimaryCell:
    return next(
        cell
        for cell in controls.plan.cells
        if cell.mechanism == _MECHANISM
        and cell.rate == _RATE
        and cell.corruption_seed == _SEED
        and cell.representation_id == representation_id
        and cell.classifier_id == _CLASSIFIER
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_manifest(cell_directory: Path) -> Path:
    records = [
        {
            "path": name,
            "size_bytes": (cell_directory / name).stat().st_size,
            "sha256": sha256_file(cell_directory / name),
        }
        for name in _ARTIFACT_NAMES
    ]
    path = cell_directory / "artifact_manifest.json"
    _write_json(path, {"schema_version": 1, "artifacts": records})
    return path


def _fixture_arrays(
    *,
    corruption_rate: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.str_]],
    np.ndarray[Any, np.dtype[np.str_]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
]:
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(5)], dtype=np.str_)
    group_ids = np.asarray([f"group_{index // 2:03d}" for index in range(5)], dtype=np.str_)
    pre = np.arange(5, dtype=np.int64) % 5
    injected = np.zeros(5, dtype=np.bool_)
    selected_count = int(
        (Decimal(len(injected)) * Decimal(str(corruption_rate))).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    injected[:selected_count] = True
    observed = pre.copy()
    observed[injected] = (observed[injected] + 1) % 5
    return sample_ids, group_ids, pre, observed, injected


def _write_cell(
    root: Path,
    cell: PrimaryCell,
    scenario: PrimaryScenario,
    controls: PrimaryExecutionControls,
    *,
    shared_hash: str,
) -> dict[str, Any]:
    directory = root / "cells" / cell.cell_id
    directory.mkdir()
    sample_ids, group_ids, pre, observed, injected = _fixture_arrays(corruption_rate=cell.rate)
    probabilities = np.full((len(sample_ids), 5), 0.2, dtype=np.float64)
    np.savez_compressed(
        directory / "oof_predictions.npz",
        sample_ids=sample_ids,
        group_ids=group_ids,
        pre_corruption_label=pre,
        observed_label=observed,
        is_injected_corruption=injected,
        probabilities=probabilities,
        predicted_class=np.zeros(len(sample_ids), dtype=np.int64),
        fold_id=np.arange(len(sample_ids), dtype=np.int64) % 2,
        coverage_count=np.ones(len(sample_ids), dtype=np.int64),
        fold_assignment_labels=pre,
        fold_assignment_label_source=np.asarray(["pre_corruption_label"], dtype=np.str_),
        fold_assignment_labels_sha256=np.asarray(["a" * 64], dtype=np.str_),
    )
    base = np.linspace(1.0, 0.0, len(sample_ids), dtype=np.float64)
    offset = 0.02 if cell.representation_id == _REPRESENTATION_B else 0.0
    risks = {
        method: np.clip(base + offset + method_index * 0.001, 0.0, 2.0)
        for method_index, method in enumerate(controls.audit_methods)
    }
    np.savez_compressed(directory / "risk_scores.npz", **risks)
    ranking_order = rank_indices(
        risks[controls.primary_ranking_method], tie_break_ids=sample_ids.tolist()
    )
    ranking_fields = (
        "rank",
        "sample_id",
        "group_id",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        *risks,
    )
    with (directory / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranking_fields)
        writer.writeheader()
        for rank, index in enumerate(ranking_order.tolist(), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "sample_id": sample_ids[index],
                    "group_id": group_ids[index],
                    "pre_corruption_label": int(pre[index]),
                    "observed_label": int(observed[index]),
                    "is_injected_corruption": bool(injected[index]),
                    **{method: float(values[index]) for method, values in risks.items()},
                }
            )
    corruption_rows = [
        {
            "sample_id": str(sample_ids[index]),
            "group_id": str(group_ids[index]),
            "pre_corruption_label": int(pre[index]),
            "observed_label": int(observed[index]),
            "is_injected_corruption": bool(injected[index]),
            "corruption_type": cell.mechanism if injected[index] else "none",
            "original_class": int(pre[index]),
            "replacement_class": int(observed[index]) if injected[index] else None,
            "corruption_rate": cell.rate,
            "corruption_seed": cell.corruption_seed,
            "corruption_representation": "generator",
            "auditor_representation": None,
            "feature_space_independent": None,
            "independence_status": "not_applicable",
            "independence_reason": "not instance dependent",
            "independence_evidence": None,
            "circularity_risk": False,
            "dataset_seed": None,
            "upstream_manifest_hash": "b" * 64,
            "configuration_hash": "c" * 64,
            "timestamp_utc": None,
        }
        for index in range(len(sample_ids))
    ]
    identity = asdict(cell)
    scenario_identity = asdict(scenario)
    instance_dependent = cell.mechanism == "instance_dependent_corruption"
    circularity_risk = instance_dependent and cell.representation_id == _REPRESENTATION_A
    independence_status = (
        "circularity_risk"
        if circularity_risk
        else "verified_independent"
        if instance_dependent
        else "not_applicable"
    )
    independence_matrix_sha256 = (
        controls.instance_independence_matrix_sha256 if instance_dependent else None
    )
    independence_evidence_sha256 = "f" * 64 if instance_dependent else None
    primary_confirmatory_eligible = not circularity_risk
    _write_json(
        directory / "corruption_manifest.json",
        {
            "schema_version": 1,
            "cell": identity,
            "scenario": scenario_identity,
            "configuration_hash": shared_hash,
            "shared_scenario_corruption_hash": shared_hash,
            "cell_corruption_provenance_sha256": "c" * 64,
            "rows": corruption_rows,
            "independence_status": independence_status,
            "circularity_risk": circularity_risk,
        },
    )
    _write_json(
        directory / "independence_evidence.json",
        {
            "schema_version": 1,
            "mechanism": cell.mechanism,
            "representation_id": cell.representation_id,
            "status": independence_status,
            "reason": (
                "separate generator and auditor representations"
                if instance_dependent and not circularity_risk
                else "overlapping generator and auditor representations"
                if circularity_risk
                else "not instance dependent"
            ),
            "circularity_risk": circularity_risk,
            "primary_confirmatory_eligible": primary_confirmatory_eligible,
            "matrix_artifact_sha256": independence_matrix_sha256,
            "evidence_sha256": independence_evidence_sha256,
            "evidence": {} if instance_dependent else None,
        },
    )
    _write_json(
        directory / "metrics.json",
        {
            "cell": identity,
            "scenario": scenario_identity,
            "corruption_configuration_hash": shared_hash,
            "independence_status": independence_status,
            "independence_evidence_sha256": independence_evidence_sha256,
            "independence_matrix_artifact_sha256": independence_matrix_sha256,
            "circularity_risk": circularity_risk,
            "primary_confirmatory_eligible": primary_confirmatory_eligible,
            "paired_group_bootstrap": {},
            "comparison_execution_scope": "deferred_exact_frozen_selectors",
        },
    )
    _write_json(
        directory / "cleanlab_evidence.json",
        {
            "schema_version": 1,
            "available": True,
            "package_version": "2.9.0",
            "api_path": "cleanlab.rank",
            "error": None,
            "blocker": None,
            "failure_policy": "missing_with_recorded_blocker",
        },
    )
    _write_json(directory / "oof_provenance.json", {"fixture": True})
    np.savez_compressed(directory / "bootstrap_evidence.npz", draw_indices=np.arange(4))
    np.savez_compressed(directory / "cleanlab_evidence.npz", available=np.asarray(True))
    np.savez_compressed(directory / "neighbour_evidence.npz", risk_scores=base)
    manifest_path = _write_artifact_manifest(directory)
    return {
        **identity,
        "status": "completed",
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(directory / "metrics.json"),
        "corruption_configuration_hash": shared_hash,
        "execution_controls_binding_sha256": controls.binding_sha256,
        "independence_status": independence_status,
        "independence_evidence_sha256": independence_evidence_sha256,
        "independence_matrix_artifact_sha256": independence_matrix_sha256,
        "circularity_risk": circularity_risk,
        "primary_confirmatory_eligible": primary_confirmatory_eligible,
    }


def _build_tree(root: Path, controls: PrimaryExecutionControls) -> None:
    root.mkdir()
    (root / "cells").mkdir()
    (root / "corruption_scenarios").mkdir()
    _write_json(root / "matrix_plan.json", controls.plan.as_dict())
    _write_json(root / "execution_controls.json", controls.as_dict())
    scenario_by_id = {scenario.scenario_id: scenario for scenario in controls.plan.scenarios}
    hashes = {
        scenario.scenario_id: canonical_sha256(asdict(scenario))
        for scenario in controls.plan.scenarios
    }
    rows = [
        _write_cell(
            root,
            cell,
            scenario_by_id[cell.scenario_id],
            controls,
            shared_hash=hashes[cell.scenario_id],
        )
        for cell in controls.plan.cells
    ]
    _write_index(root / "cell_index.csv", rows)
    for scenario in controls.plan.scenarios:
        _write_json(
            root / "corruption_scenarios" / f"{scenario.scenario_id}.json",
            {
                "scenario": asdict(scenario),
                "shared_scenario_corruption_hash": hashes[scenario.scenario_id],
            },
        )
    sample_ids, _, _, _, _ = _fixture_arrays(corruption_rate=_RATE)
    crop_path = root.parent / f"{root.name}_crop.npz"
    np.savez_compressed(
        crop_path,
        sample_ids=sample_ids,
        tissue_types=np.asarray(
            ["tissue_a" if index < 3 else "tissue_b" for index in range(5)],
            dtype=np.str_,
        ),
    )
    crop_sha = sha256_file(crop_path)
    _write_json(
        root / "primary_input_bindings.json",
        {
            "cache_paths": {"crop_cache_path": str(crop_path)},
            "expected_hashes": {"crop_cache_sha256": crop_sha},
            "verified_hashes": {"crop_cache_sha256": crop_sha},
            "sample_order_sha256": canonical_sha256(sample_ids.tolist()),
        },
    )


def _update_statistics_manifest(root: Path, artifact_name: str) -> None:
    path = root / "primary_statistics_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["artifacts"] if item["path"] == artifact_name)
    record["size_bytes"] = (root / artifact_name).stat().st_size
    record["sha256"] = sha256_file(root / artifact_name)
    _write_json(path, manifest)


def test_primary_statistics_aggregate_all_frozen_families_and_zero_event_cell(
    tmp_path: Path,
) -> None:
    controls = _controls()
    run = tmp_path / "primary"
    _build_tree(run, controls)

    artifacts = aggregate_primary_statistics(run, controls)
    verification = verify_primary_statistics_artifacts(run, controls)
    statistics = json.loads(artifacts.statistics_path.read_text(encoding="utf-8"))

    assert verification.valid
    assert statistics["bootstrap"]["unit"] == "source_patch_id"
    assert artifacts.comparison_count == 4
    assert artifacts.reportable_comparison_count == 3
    assert {item["kind"] for item in statistics["comparisons"]} == {
        "within_cell",
        "method_vs_random",
        "cross_cell",
    }
    reportable = [item for item in statistics["comparisons"] if item["status"] == "reported"]
    assert all(item["p_value_holm"] is not None for item in reportable)
    zero_comparison = next(
        item
        for item in statistics["comparisons"]
        if item["comparison_id"] == "zero_event_self_vs_nll"
    )
    assert zero_comparison["status"] == "not_applicable_zero_corruption"
    assert zero_comparison["reason"] == (
        "no injected-corruption events; no random or paired inference performed"
    )
    assert zero_comparison["point_difference"] is None
    assert zero_comparison["p_value_unadjusted"] is None
    assert zero_comparison["p_value_holm"] is None
    assert zero_comparison["valid_bootstrap_iterations"] == 0
    zero = next(
        item
        for item in statistics["cells"]
        if item["cell"]["rate"] == 0.05
        and item["cell"]["corruption_seed"] == _SEED
        and item["cell"]["representation_id"] == _REPRESENTATION_A
        and item["cell"]["mechanism"] == _MECHANISM
    )
    assert zero["sample_count"] == 5
    assert zero["injected_corruption_count"] == 0
    assert zero["zero_corruption_status"] == "not_applicable_for_event_based_inference"
    assert zero["methods"]["self_confidence"]["average_precision"] == {
        "status": "not_applicable",
        "value": None,
        "reason": "no injected-corruption events",
    }
    zero_random = zero["random_review_by_budget"]["0.05"]
    assert zero_random["inference_status"] == "not_applicable_zero_corruption"
    assert zero_random["repeats"] == controls.random_review_repeats
    assert zero_random["seeds"] == list(
        range(
            controls.random_review_seed,
            controls.random_review_seed + controls.random_review_repeats,
        )
    )
    assert zero_random["mean_recall"]["value"] is None
    assert zero_random["recall_interval_95"] is None
    assert (
        zero["methods"]["self_confidence"]["review_budgets"]["0.05"]["tied_random_review"]
        == zero_random
    )
    positive = next(
        item
        for item in statistics["cells"]
        if item["cell"]["rate"] == _RATE
        and item["cell"]["corruption_seed"] == _SEED
        and item["cell"]["representation_id"] == _REPRESENTATION_A
        and item["cell"]["mechanism"] == _MECHANISM
    )
    assert positive["methods"]["self_confidence"]["auroc"]["status"] == "reported"
    assert (
        positive["methods"]["self_confidence"]["review_budgets"]["0.05"]["precision"]["status"]
        == "reported"
    )
    with np.load(artifacts.bootstrap_evidence_path, allow_pickle=False) as evidence:
        assert len(evidence["draw_offsets"]) == controls.bootstrap_iterations + 1
        assert evidence["comparison_ids"].tolist() == [
            "within_self_vs_nll",
            "zero_event_self_vs_nll",
            "self_vs_random",
            "representation_a_vs_b",
        ]
        assert evidence["comparison_001_valid_draw_indices"].size == 0
        assert evidence["comparison_001_metric_a"].size == 0
        assert evidence["comparison_001_metric_b"].size == 0
        assert evidence["comparison_001_differences"].size == 0
    subgroup_text = artifacts.subgroups_path.read_text(encoding="utf-8")
    assert "suppressed_insufficient_support" in subgroup_text
    assert "not_applicable_zero_corruption" in subgroup_text


def test_primary_statistics_verifier_rejects_json_tamper_with_updated_manifest(
    tmp_path: Path,
) -> None:
    controls = _controls()
    run = tmp_path / "tampered-json"
    _build_tree(run, controls)
    aggregate_primary_statistics(run, controls)
    statistics_path = run / "primary_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["comparisons"][0]["point_difference"] = 123.0
    _write_json(statistics_path, statistics)
    _update_statistics_manifest(run, "primary_statistics.json")

    with pytest.raises(ValueError, match="differs from recomputed"):
        verify_primary_statistics_artifacts(run, controls)


def test_primary_statistics_verifier_rejects_bootstrap_tamper_with_updated_manifest(
    tmp_path: Path,
) -> None:
    controls = _controls()
    run = tmp_path / "tampered-bootstrap"
    _build_tree(run, controls)
    aggregate_primary_statistics(run, controls)
    bootstrap_path = run / "primary_bootstrap_evidence.npz"
    with np.load(bootstrap_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["comparison_000_differences"] = arrays["comparison_000_differences"].copy()
    arrays["comparison_000_differences"][0] += 0.5
    np.savez_compressed(bootstrap_path, **arrays)
    _update_statistics_manifest(run, "primary_bootstrap_evidence.npz")

    with pytest.raises(ValueError, match="bootstrap array differs"):
        verify_primary_statistics_artifacts(run, controls)


def test_primary_statistics_accepts_controlled_cross_scenario_frozen_comparison(
    tmp_path: Path,
) -> None:
    del tmp_path

    controls = primary_execution_controls_from_frozen_config(_frozen_config(cross_scenario=True))
    comparison = controls.cross_cell_comparisons[0]

    assert comparison.selector_a.mechanism == comparison.selector_b.mechanism
    assert comparison.selector_a.rate == 0.20
    assert comparison.selector_b.rate == 0.10
    assert comparison.selector_a.seed == comparison.selector_b.seed == 404
    assert comparison.selector_a.representation_id == comparison.selector_b.representation_id


def test_primary_statistics_rejects_same_scenario_corruption_manifest_drift(
    tmp_path: Path,
) -> None:
    controls = _controls()
    run = tmp_path / "manifest-drift"
    _build_tree(run, controls)
    target_cell = _comparison_cell(controls, _REPRESENTATION_B)
    cell_directory = run / "cells" / target_cell.cell_id
    corruption_path = cell_directory / "corruption_manifest.json"
    corruption = json.loads(corruption_path.read_text(encoding="utf-8"))
    corruption["rows"][0]["replacement_class"] = 4
    _write_json(corruption_path, corruption)
    artifact_manifest = _write_artifact_manifest(cell_directory)
    with (run / "cell_index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    target = next(row for row in rows if row["cell_id"] == target_cell.cell_id)
    target["artifact_manifest_sha256"] = sha256_file(artifact_manifest)
    _write_index(run / "cell_index.csv", rows)

    with pytest.raises(ValueError, match="does not share one corruption manifest"):
        aggregate_primary_statistics(run, controls)


def test_primary_statistics_excludes_circularity_risk_from_every_comparison(
    tmp_path: Path,
) -> None:
    controls = _controls()
    run = tmp_path / "circularity"
    _build_tree(run, controls)
    target_cell = _comparison_cell(controls, _REPRESENTATION_A)
    cell_directory = run / "cells" / target_cell.cell_id
    independence_path = cell_directory / "independence_evidence.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    independence.update(
        status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
    )
    _write_json(independence_path, independence)
    corruption_path = cell_directory / "corruption_manifest.json"
    corruption = json.loads(corruption_path.read_text(encoding="utf-8"))
    corruption.update(independence_status="circularity_risk", circularity_risk=True)
    _write_json(corruption_path, corruption)
    metrics_path = cell_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        independence_status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
    )
    _write_json(metrics_path, metrics)
    artifact_manifest = _write_artifact_manifest(cell_directory)
    with (run / "cell_index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    target = next(row for row in rows if row["cell_id"] == target_cell.cell_id)
    target.update(
        independence_status="circularity_risk",
        circularity_risk=True,
        primary_confirmatory_eligible=False,
        metrics_sha256=sha256_file(metrics_path),
        artifact_manifest_sha256=sha256_file(artifact_manifest),
    )
    _write_index(run / "cell_index.csv", rows)

    artifacts = aggregate_primary_statistics(run, controls)
    statistics = json.loads(artifacts.statistics_path.read_text(encoding="utf-8"))

    expected_automatic_exclusions = {
        cell.cell_id
        for cell in controls.plan.cells
        if cell.mechanism == "instance_dependent_corruption"
        and cell.representation_id == _REPRESENTATION_A
    }
    assert set(statistics["circularity_excluded_cell_ids"]) == {
        target_cell.cell_id,
        *expected_automatic_exclusions,
    }
    affected = [
        item
        for item in statistics["comparisons"]
        if item["comparison_id"] != "zero_event_self_vs_nll"
    ]
    assert affected
    assert all(item["status"] == "excluded_circularity_risk" for item in affected)
    assert all(item["p_value_holm"] is None for item in affected)
    zero_event = next(
        item
        for item in statistics["comparisons"]
        if item["comparison_id"] == "zero_event_self_vs_nll"
    )
    assert zero_event["status"] == "not_applicable_zero_corruption"
