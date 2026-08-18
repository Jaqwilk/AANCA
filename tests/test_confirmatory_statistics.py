"""Exact selector-scoped paired statistics for the confirmatory matrix."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from test_study_contracts import complete_confirmatory_config

from histo_audit.experiment import confirmatory_completion as completion_module
from histo_audit.experiment import confirmatory_statistics as statistics_module
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    confirmatory_execution_controls_from_frozen_config,
    run_synthetic_confirmatory_contract_fixture,
)
from histo_audit.experiment.confirmatory_statistics import (
    ConfirmatoryStatisticsArtifacts,
    aggregate_confirmatory_statistics,
    verify_confirmatory_statistics_artifacts,
)
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file


@pytest.fixture(scope="module")
def completed_statistics_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[
    tuple[Path, dict[str, object], ConfirmatoryExecutionControls, ConfirmatoryStatisticsArtifacts]
]:
    root = tmp_path_factory.mktemp("confirmatory-statistics") / "run"
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    run_synthetic_confirmatory_contract_fixture(config, output_directory=root)
    artifacts = aggregate_confirmatory_statistics(root, controls)
    yield root, config, controls, artifacts


def _status_by_id(run: Path) -> dict[str, str]:
    with (run / "cell_index.csv").open(encoding="utf-8", newline="") as stream:
        return {str(row["cell_id"]): str(row["status"]) for row in csv.DictReader(stream)}


def _copy_run(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def test_statistics_execute_only_frozen_operands_and_match_completion_readback(
    completed_statistics_run: tuple[
        Path,
        dict[str, object],
        ConfirmatoryExecutionControls,
        ConfirmatoryStatisticsArtifacts,
    ],
) -> None:
    run, config, controls, artifacts = completed_statistics_run
    payload = json.loads(artifacts.statistics_path.read_text(encoding="utf-8"))
    expected_ids = [value.comparison_id for value in controls.paired_comparisons]

    assert artifacts.comparison_count == len(expected_ids) == 4
    assert artifacts.completed_comparison_count == 3
    assert payload["config_semantic_sha256"] == controls.config_semantic_sha256
    assert payload["execution_controls_binding_sha256"] == controls.binding_sha256
    assert payload["paired_unit"] == controls.statistical_group_unit
    assert [value["comparison_id"] for value in payload["comparisons"]] == expected_ids
    assert [value["operand_a"] for value in payload["comparisons"]] == [
        value.as_dict()["operand_a"] for value in controls.paired_comparisons
    ]

    results = {value["comparison_id"]: value for value in payload["comparisons"]}
    assert results["pathology_minus_imagenet"]["status"] == (
        "not_estimable_frozen_optional_blocker"
    )
    assert results["pathology_minus_imagenet"]["frozen_unavailability"] is True
    assert {
        value["status"] for key, value in results.items() if key != "pathology_minus_imagenet"
    } == {"completed"}
    assert all(value["selected_injected_event_count"] > 0 for value in results.values())
    assert all(
        value["valid_iterations"] > 0
        and value["valid_iterations"] <= controls.paired_group_bootstrap_iterations
        and value["holm_adjusted_p"] is not None
        for value in results.values()
        if value["status"] == "completed"
    )

    with np.load(artifacts.bootstrap_evidence_path, allow_pickle=False) as evidence:
        expected_keys = {"bootstrap_group_universe", "bootstrap_group_draws"}
        for comparison_id in expected_ids:
            expected_keys.update(
                {
                    f"valid_draw_mask__{comparison_id}",
                    f"metric_a__{comparison_id}",
                    f"metric_b__{comparison_id}",
                    f"differences__{comparison_id}",
                }
            )
        assert set(evidence.files) == expected_keys
        universe = evidence["bootstrap_group_universe"]
        draws = evidence["bootstrap_group_draws"]
        assert draws.shape == (
            controls.paired_group_bootstrap_iterations,
            len(universe),
        )
        for comparison_id, result in results.items():
            mask = evidence[f"valid_draw_mask__{comparison_id}"]
            first = evidence[f"metric_a__{comparison_id}"]
            second = evidence[f"metric_b__{comparison_id}"]
            differences = evidence[f"differences__{comparison_id}"]
            assert int(mask.sum()) == result["valid_iterations"]
            np.testing.assert_allclose(differences, first - second, rtol=0.0, atol=1e-12)
            if result["status"] == "completed":
                assert not (np.all(first == 0.0) and np.all(second == 0.0))
            else:
                assert not mask.any() and not len(first) and not len(second)

    verification = verify_confirmatory_statistics_artifacts(run, controls)
    assert verification.status == "passed"
    completion_module._validate_paired_statistics(
        run,
        config=config,
        plan=controls.plan,
        status_by_id=_status_by_id(run),
    )


def test_statistics_never_overwrite_existing_artifacts(
    completed_statistics_run: tuple[
        Path,
        dict[str, object],
        ConfirmatoryExecutionControls,
        ConfirmatoryStatisticsArtifacts,
    ],
) -> None:
    run, _config, controls, _artifacts = completed_statistics_run

    with pytest.raises(FileExistsError, match="never overwrite"):
        aggregate_confirmatory_statistics(run, controls)


def test_verifier_rejects_bootstrap_tamper_even_after_json_sha_update(
    completed_statistics_run: tuple[
        Path,
        dict[str, object],
        ConfirmatoryExecutionControls,
        ConfirmatoryStatisticsArtifacts,
    ],
    tmp_path: Path,
) -> None:
    source, _config, controls, _artifacts = completed_statistics_run
    run = _copy_run(source, tmp_path / "tampered-bootstrap")
    bootstrap = run / "paired_bootstrap_evidence.npz"
    with np.load(bootstrap, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    key = "differences__target_mask_minus_context_rgb"
    arrays[key][0] += 0.125
    np.savez_compressed(bootstrap, **arrays)
    payload = json.loads((run / "paired_statistics.json").read_text(encoding="utf-8"))
    payload["bootstrap_evidence_sha256"] = sha256_file(bootstrap)
    atomic_write_json(run / "paired_statistics.json", payload)

    with pytest.raises(ValueError, match="bootstrap array differs"):
        verify_confirmatory_statistics_artifacts(run, controls)


def test_verifier_rejects_selector_or_summary_tamper(
    completed_statistics_run: tuple[
        Path,
        dict[str, object],
        ConfirmatoryExecutionControls,
        ConfirmatoryStatisticsArtifacts,
    ],
    tmp_path: Path,
) -> None:
    source, _config, controls, _artifacts = completed_statistics_run
    run = _copy_run(source, tmp_path / "tampered-json")
    path = run / "paired_statistics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparisons"][0]["operand_a"]["risk_id"] = "fixed_hybrid"
    payload["comparisons"][0]["observed_delta"] = 999.0
    atomic_write_json(path, payload)

    with pytest.raises(ValueError, match="differs from recomputed"):
        verify_confirmatory_statistics_artifacts(run, controls)


def test_zero_event_statistics_are_structured_not_applicable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    completed: dict[str, statistics_module._CellEvidence] = {}
    status_by_id: dict[str, str] = {}
    blocker_by_scenario: dict[str, dict[str, object]] = {}
    for cell in controls.plan.cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        if not cell.required:
            status_by_id[cell.cell_id] = "skipped_with_frozen_blocker"
            blocker_by_scenario[cell.scenario_id] = {
                "blocker": "frozen optional scenario unavailable in zero-event fixture",
                "blocker_evidence_sha256": scenario.availability_audit_sha256,
            }
            continue
        sample_ids = np.asarray(
            [f"rotation_{cell.outer_fold}_sample_0", f"rotation_{cell.outer_fold}_sample_1"],
            dtype=np.str_,
        )
        group_ids = np.asarray(
            [f"rotation_{cell.outer_fold}_group_0", f"rotation_{cell.outer_fold}_group_1"],
            dtype=np.str_,
        )
        pre_corruption_label = np.asarray([0, 1], dtype=np.int64)
        completed[cell.cell_id] = statistics_module._CellEvidence(
            cell=cell,
            sample_ids=sample_ids,
            group_ids=group_ids,
            group_tokens=np.asarray(
                [f"fold_{cell.outer_fold}::{group}" for group in group_ids],
                dtype=np.str_,
            ),
            pre_corruption_label=pre_corruption_label,
            observed_label=pre_corruption_label.copy(),
            injected=np.zeros(2, dtype=bool),
            fold_id=np.asarray([0, 1], dtype=np.int64),
            fold_assignment_labels=pre_corruption_label.copy(),
            risks={
                risk_id: np.asarray([0.25, 0.75], dtype=np.float64)
                for risk_id in statistics_module._RISK_ARRAY_BY_ID
            },
        )
        status_by_id[cell.cell_id] = "completed"
    matrix = statistics_module._LoadedMatrix(
        completed=completed,
        status_by_id=status_by_id,
        blocker_by_scenario=blocker_by_scenario,
        group_universe=np.asarray(
            [
                f"fold_{outer_fold}::rotation_{outer_fold}_group_{group_index}"
                for outer_fold in controls.official_folds
                for group_index in range(2)
            ],
            dtype=np.str_,
        ),
    )
    monkeypatch.setattr(statistics_module, "_load_matrix", lambda *_args: matrix)

    run = tmp_path / "zero-event"
    artifacts = aggregate_confirmatory_statistics(run, controls)
    payload = json.loads(artifacts.statistics_path.read_text(encoding="utf-8"))

    assert {
        result["status"]
        for result in payload["comparisons"]
        if result["comparison_id"] != "pathology_minus_imagenet"
    } == {"not_applicable_zero_event"}
    assert all(result["selected_injected_event_count"] == 0 for result in payload["comparisons"])
    with np.load(artifacts.bootstrap_evidence_path, allow_pickle=False) as evidence:
        for result in payload["comparisons"]:
            if result["status"] == "not_applicable_zero_event":
                comparison_id = result["comparison_id"]
                assert not evidence[f"valid_draw_mask__{comparison_id}"].any()
                assert not len(evidence[f"differences__{comparison_id}"])
    assert verify_confirmatory_statistics_artifacts(run, controls).status == "passed"
