from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from histo_audit.auditing import (
    FixedHybridDropOneResult,
    fixed_hybrid_drop_one_ablations,
)
from histo_audit.auditing.scores import percentile_normalise


@pytest.fixture
def components() -> Mapping[str, NDArray[np.float64]]:
    return {
        "confidence": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "neighbours": np.asarray([3.0, 2.0, 1.0, 0.0]),
        "ensemble": np.asarray([0.0, 2.0, 1.0, 3.0]),
    }


def test_full_hybrid_and_every_drop_one_ablation_use_frozen_weight_ratios(
    components: Mapping[str, NDArray[np.float64]],
) -> None:
    result = fixed_hybrid_drop_one_ablations(
        components,
        components=("confidence", "neighbours", "ensemble"),
        weights=(2.0, 3.0, 5.0),
    )
    confidence = percentile_normalise(components["confidence"])
    neighbours = percentile_normalise(components["neighbours"])
    ensemble = percentile_normalise(components["ensemble"])

    np.testing.assert_allclose(
        result.full_score,
        0.2 * confidence + 0.3 * neighbours + 0.5 * ensemble,
    )
    assert tuple(result.drop_one_scores) == ("confidence", "neighbours", "ensemble")
    np.testing.assert_allclose(
        result.drop_one_scores["confidence"],
        3.0 / 8.0 * neighbours + 5.0 / 8.0 * ensemble,
    )
    np.testing.assert_allclose(
        result.drop_one_scores["neighbours"],
        2.0 / 7.0 * confidence + 5.0 / 7.0 * ensemble,
    )
    np.testing.assert_allclose(
        result.drop_one_scores["ensemble"],
        2.0 / 5.0 * confidence + 3.0 / 5.0 * neighbours,
    )
    assert result.components == ("confidence", "neighbours", "ensemble")
    np.testing.assert_allclose(result.normalised_weights, (0.2, 0.3, 0.5))


def test_result_is_deterministic_immutable_and_has_exactly_one_ablation_per_component(
    components: Mapping[str, NDArray[np.float64]],
) -> None:
    first = fixed_hybrid_drop_one_ablations(
        components,
        components=("confidence", "neighbours", "ensemble"),
        weights=(0.2, 0.3, 0.5),
    )
    second = fixed_hybrid_drop_one_ablations(
        components,
        components=("confidence", "neighbours", "ensemble"),
        weights=(0.2, 0.3, 0.5),
    )

    assert isinstance(first, FixedHybridDropOneResult)
    assert len(first.drop_one_scores) == len(first.components)
    np.testing.assert_array_equal(first.full_score, second.full_score)
    for name in first.components:
        np.testing.assert_array_equal(first.drop_one_scores[name], second.drop_one_scores[name])
        assert not first.drop_one_scores[name].flags.writeable
    assert not first.full_score.flags.writeable
    with pytest.raises(TypeError):
        first.drop_one_scores["new"] = first.full_score  # type: ignore[index]


def test_two_component_design_produces_single_component_ablations(
    components: Mapping[str, NDArray[np.float64]],
) -> None:
    result = fixed_hybrid_drop_one_ablations(
        components,
        components=("confidence", "ensemble"),
        weights=(0.25, 0.75),
    )

    np.testing.assert_array_equal(
        result.drop_one_scores["confidence"],
        percentile_normalise(components["ensemble"]),
    )
    np.testing.assert_array_equal(
        result.drop_one_scores["ensemble"],
        percentile_normalise(components["confidence"]),
    )


def test_larger_component_risks_remain_higher_in_full_and_every_ablation() -> None:
    aligned = {
        "a": np.asarray([0.1, 0.4, 0.9]),
        "b": np.asarray([-2.0, 0.0, 5.0]),
        "c": np.asarray([10.0, 11.0, 100.0]),
    }
    result = fixed_hybrid_drop_one_ablations(
        aligned,
        components=("a", "b", "c"),
        weights=(0.2, 0.3, 0.5),
    )

    assert result.full_score[-1] > result.full_score[0]
    assert all(score[-1] > score[0] for score in result.drop_one_scores.values())


@pytest.mark.parametrize(
    ("component_names", "weights", "message"),
    [
        ((), (), "at least two"),
        (("confidence",), (1.0,), "at least two"),
        (("confidence", "confidence"), (0.5, 0.5), "unique"),
        (("confidence", "unknown"), (0.5, 0.5), "missing"),
        (("confidence", "ensemble"), (1.0,), "align"),
        (("confidence", "ensemble"), (0.5, -0.5), "non-negative"),
        (("confidence", "ensemble"), (0.5, float("nan")), "finite"),
        (("confidence", "ensemble"), (0.0, 0.0), "not all be zero"),
        (("confidence", "ensemble"), (1.0, 0.0), "no positive frozen"),
    ],
)
def test_invalid_frozen_designs_fail_closed(
    components: Mapping[str, NDArray[np.float64]],
    component_names: tuple[str, ...],
    weights: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fixed_hybrid_drop_one_ablations(
            components,
            components=component_names,
            weights=weights,
        )


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (np.asarray([0.0, np.inf, 1.0, 2.0]), "finite"),
        (np.asarray([0.0, 1.0, 2.0]), "different lengths"),
        (np.asarray([[0.0, 1.0], [2.0, 3.0]]), "finite non-empty vector"),
    ],
)
def test_invalid_component_vectors_fail_closed(
    components: Mapping[str, NDArray[np.float64]],
    bad_value: NDArray[np.float64],
    message: str,
) -> None:
    invalid = dict(components)
    invalid["ensemble"] = bad_value
    with pytest.raises(ValueError, match=message):
        fixed_hybrid_drop_one_ablations(
            invalid,
            components=("confidence", "neighbours", "ensemble"),
            weights=(0.2, 0.3, 0.5),
        )
