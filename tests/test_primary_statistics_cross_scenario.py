"""Focused regressions for controlled cross-scenario statistics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from histo_audit.experiment.primary_core import (
    PrimaryExecutionControls,
    _validate_frozen_primary_comparisons,
    primary_execution_controls_from_frozen_config,
)
from histo_audit.experiment.primary_statistics import (
    _CellData,
    _comparison_result,
    _resolve_selector,
    _validate_cross_cell_pair,
)
from histo_audit.experiment.study_contracts import PrimaryCell
from histo_audit.statistics.review import average_precision
from tests.test_study_contracts_strict_m7 import strict_primary_config

_METHOD = "self_confidence"
_CLASSIFIER = "multinomial_logistic_regression"
_REPRESENTATION = "imagenet_resnet18_highlighted"


def _cell(
    suffix: str,
    *,
    scenario_id: str,
    mechanism: str = "symmetric_random_corruption",
    rate: float = 0.10,
    seed: int = 404,
    representation_id: str = _REPRESENTATION,
    classifier_id: str = _CLASSIFIER,
) -> PrimaryCell:
    return PrimaryCell(
        cell_id=f"primary_{suffix}",
        scenario_id=scenario_id,
        mechanism=mechanism,
        rate=rate,
        corruption_seed=seed,
        representation_id=representation_id,
        classifier_id=classifier_id,
        required=True,
    )


def _data(
    cell: PrimaryCell,
    *,
    injected: np.ndarray,
    scores: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
) -> _CellData:
    n_samples = len(injected)
    samples = (
        np.asarray([f"sample_{index}" for index in range(n_samples)], dtype=np.str_)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.str_)
    )
    groups = (
        np.asarray([f"group_{index // 2}" for index in range(n_samples)], dtype=np.str_)
        if group_ids is None
        else np.asarray(group_ids, dtype=np.str_)
    )
    pre = np.arange(n_samples, dtype=np.int64) % 5
    injected_bool = np.asarray(injected, dtype=np.bool_)
    observed = pre.copy()
    observed[injected_bool] = (observed[injected_bool] + 1) % 5
    risk = (
        np.linspace(1.0, 0.0, n_samples, dtype=np.float64)
        if scores is None
        else np.asarray(scores, dtype=np.float64)
    )
    return _CellData(
        cell=cell,
        sample_ids=samples,
        group_ids=groups,
        pre_corruption_label=pre,
        observed_label=observed,
        injected=injected_bool,
        risks={_METHOD: risk},
        shared_corruption_rows=(),
        shared_corruption_sha256="a" * 64,
        circularity_risk=False,
        primary_confirmatory_eligible=True,
    )


def _rate_contrast_controls() -> PrimaryExecutionControls:
    config = strict_primary_config()
    comparison = config["statistics"]["cross_cell_comparisons"][0]
    comparison["comparison_id"] = "rate_sensitivity_effect"
    comparison["selector_a"] = {
        "mechanism": "symmetric_random_corruption",
        "rate": 0.10,
        "seed": 404,
        "representation_id": _REPRESENTATION,
        "classifier_id": _CLASSIFIER,
    }
    comparison["selector_b"] = {
        "mechanism": "symmetric_random_corruption",
        "rate": 0.20,
        "seed": 404,
        "representation_id": _REPRESENTATION,
        "classifier_id": _CLASSIFIER,
    }
    return primary_execution_controls_from_frozen_config(config)


@pytest.mark.parametrize(
    "cell_b",
    [
        _cell("rate_b", scenario_id="scenario_rate_b", rate=0.20),
        _cell(
            "mechanism_b",
            scenario_id="scenario_mechanism_b",
            mechanism="confusion_targeted_corruption",
        ),
    ],
)
def test_cross_scenario_allows_exactly_one_changed_corruption_factor(cell_b: PrimaryCell) -> None:
    cell_a = _cell("a", scenario_id="scenario_a")
    injected_a = np.asarray([True, False, False, True, False, False])
    injected_b = np.asarray([False, True, False, False, True, False])

    _validate_cross_cell_pair(
        comparison_id="controlled_cross_scenario",
        cell_a=cell_a,
        cell_b=cell_b,
        data_a=_data(cell_a, injected=injected_a),
        data_b=_data(cell_b, injected=injected_b),
    )


@pytest.mark.parametrize(
    ("cell_b", "message"),
    [
        (
            _cell("seed_b", scenario_id="scenario_seed_b", rate=0.20, seed=405),
            "hold seed, representation, and classifier fixed",
        ),
        (
            _cell(
                "two_factors_b",
                scenario_id="scenario_two_factors_b",
                mechanism="confusion_targeted_corruption",
                rate=0.20,
            ),
            "vary exactly one of mechanism or rate",
        ),
        (
            _cell("no_factor_b", scenario_id="scenario_no_factor_b"),
            "vary exactly one of mechanism or rate",
        ),
    ],
)
def test_cross_scenario_rejects_seed_drift_and_confounded_factor_changes(
    cell_b: PrimaryCell,
    message: str,
) -> None:
    cell_a = _cell("a", scenario_id="scenario_a")

    with pytest.raises(ValueError, match=message):
        _validate_cross_cell_pair(
            comparison_id="invalid_cross_scenario",
            cell_a=cell_a,
            cell_b=cell_b,
            data_a=None,
            data_b=None,
        )


@pytest.mark.parametrize(
    "mutate_b",
    [
        lambda data: object.__setattr__(
            data,
            "sample_ids",
            np.asarray(["different", *data.sample_ids[1:].tolist()], dtype=np.str_),
        ),
        lambda data: object.__setattr__(
            data,
            "group_ids",
            np.asarray(["different", *data.group_ids[1:].tolist()], dtype=np.str_),
        ),
        lambda data: object.__setattr__(
            data,
            "pre_corruption_label",
            np.asarray([4, *data.pre_corruption_label[1:].tolist()], dtype=np.int64),
        ),
    ],
)
def test_cross_scenario_requires_exact_sample_group_reference_alignment(
    mutate_b: Callable[[_CellData], None],
) -> None:
    cell_a = _cell("a", scenario_id="scenario_a", rate=0.10)
    cell_b = _cell("b", scenario_id="scenario_b", rate=0.20)
    injected = np.asarray([True, False, False, True, False, False])
    data_a = _data(cell_a, injected=injected)
    data_b = _data(cell_b, injected=np.roll(injected, 1))
    mutate_b(data_b)

    with pytest.raises(ValueError, match="sample/group/reference-label alignment"):
        _validate_cross_cell_pair(
            comparison_id="misaligned_cross_scenario",
            cell_a=cell_a,
            cell_b=cell_b,
            data_a=data_a,
            data_b=data_b,
        )


def test_within_scenario_corruption_identity_invariant_is_not_relaxed() -> None:
    cell_a = _cell("a", scenario_id="same_scenario")
    cell_b = _cell(
        "b",
        scenario_id="same_scenario",
        representation_id="imagenet_resnet18_context",
    )
    injected_a = np.asarray([True, False, False, True, False, False])
    injected_b = np.asarray([False, True, False, True, False, False])

    with pytest.raises(ValueError, match=r"within-scenario.*not identical"):
        _validate_cross_cell_pair(
            comparison_id="same_scenario_model_contrast",
            cell_a=cell_a,
            cell_b=cell_b,
            data_a=_data(cell_a, injected=injected_a),
            data_b=_data(cell_b, injected=injected_b),
        )


def test_cross_scenario_metrics_use_each_cells_own_injected_labels_and_shared_draws() -> None:
    cell_a = _cell("a", scenario_id="scenario_rate_010", rate=0.10)
    cell_b = _cell("b", scenario_id="scenario_rate_020", rate=0.20)
    injected_a = np.asarray([True, False, False, True, False, False])
    injected_b = np.asarray([False, True, False, False, True, False])
    scores_a = np.asarray([0.95, 0.60, 0.50, 0.10, 0.40, 0.30])
    scores_b = np.asarray([0.10, 0.95, 0.20, 0.30, 0.90, 0.40])
    data_a = _data(cell_a, injected=injected_a, scores=scores_a)
    data_b = _data(cell_b, injected=injected_b, scores=scores_b)
    draws = (
        np.arange(6, dtype=np.int64),
        np.asarray([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64),
    )

    result, valid_indices, metrics_a, metrics_b = _comparison_result(
        comparison_id="rate_sensitivity_effect",
        kind="cross_cell",
        cell_a=data_a,
        cell_b=data_b,
        method_a=_METHOD,
        method_b=_METHOD,
        metric="average_precision",
        direction="method_a_minus_method_b",
        holm_family="primary_sensitivity",
        review_budget=None,
        draws=draws,
        random_scores=np.empty((0, 6), dtype=np.float64),
    )

    expected_a = np.asarray(
        [average_precision(injected_a[draw], scores_a[draw]) for draw in draws],
        dtype=np.float64,
    )
    expected_b = np.asarray(
        [average_precision(injected_b[draw], scores_b[draw]) for draw in draws],
        dtype=np.float64,
    )
    assert result["status"] == "reported"
    assert result["point_metric_a"] == pytest.approx(average_precision(injected_a, scores_a))
    assert result["point_metric_b"] == pytest.approx(average_precision(injected_b, scores_b))
    assert result["point_metric_b"] != pytest.approx(average_precision(injected_a, scores_b))
    np.testing.assert_array_equal(valid_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(metrics_a, expected_a)
    np.testing.assert_allclose(metrics_b, expected_b)
    np.testing.assert_allclose(metrics_a - metrics_b, expected_a - expected_b)


def test_real_frozen_controls_reach_cross_scenario_statistics() -> None:
    controls = _rate_contrast_controls()
    comparison = controls.cross_cell_comparisons[0]
    planned_a = _resolve_selector(controls, comparison.selector_a, "h2.selector_a")
    planned_b = _resolve_selector(controls, comparison.selector_b, "h2.selector_b")
    injected_a = np.asarray([True, False, False, True, False, False])
    injected_b = np.asarray([False, True, False, False, True, False])
    scores_a = np.asarray([0.95, 0.60, 0.50, 0.10, 0.40, 0.30])
    scores_b = np.asarray([0.10, 0.95, 0.20, 0.30, 0.90, 0.40])
    data_a = _data(planned_a, injected=injected_a, scores=scores_a)
    data_b = _data(planned_b, injected=injected_b, scores=scores_b)

    _validate_cross_cell_pair(
        comparison_id=comparison.comparison_id,
        cell_a=planned_a,
        cell_b=planned_b,
        data_a=data_a,
        data_b=data_b,
    )
    result, _, _, _ = _comparison_result(
        comparison_id=comparison.comparison_id,
        kind="cross_cell",
        cell_a=data_a,
        cell_b=data_b,
        method_a=comparison.method_a,
        method_b=comparison.method_b,
        metric=comparison.metric,
        direction=comparison.direction,
        holm_family=comparison.holm_family,
        review_budget=None,
        draws=(np.arange(6, dtype=np.int64),),
        random_scores=np.empty((0, 6), dtype=np.float64),
    )

    assert planned_a.scenario_id != planned_b.scenario_id
    assert result["status"] == "reported"
    assert result["point_metric_a"] == pytest.approx(average_precision(injected_a, scores_a))
    assert result["point_metric_b"] == pytest.approx(average_precision(injected_b, scores_b))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda comparison: replace(
                comparison,
                selector_b=replace(comparison.selector_b, seed=405),
            ),
            "hold seed, representation, and classifier fixed",
        ),
        (
            lambda comparison: replace(
                comparison,
                selector_b=replace(
                    comparison.selector_b,
                    mechanism="confusion_targeted_corruption",
                ),
            ),
            "vary exactly one of mechanism or rate",
        ),
        (
            lambda comparison: replace(comparison, method_b="predictive_entropy"),
            "hold the audit method fixed",
        ),
    ],
)
def test_runtime_controls_fail_closed_on_confounded_cross_scenario_tamper(
    tamper: Callable,
    message: str,
) -> None:
    controls = _rate_contrast_controls()
    invalid_comparison = tamper(controls.cross_cell_comparisons[0])
    tampered_controls = replace(
        controls,
        cross_cell_comparisons=(invalid_comparison,),
    )

    with pytest.raises(ValueError, match=message):
        _validate_frozen_primary_comparisons(
            tampered_controls,
            (controls.primary_review_budget, *controls.secondary_review_budgets),
        )
