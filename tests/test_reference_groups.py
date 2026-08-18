from __future__ import annotations

import math

import numpy as np
import pytest

from histo_audit.experiment.reference_groups import (
    ReferenceGroupSelectionError,
    deterministic_group_greedy_class_distribution_v1,
)


def _balanced_group_fixture(group_count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat(
        np.asarray([f"source-patch-{index:02d}" for index in range(group_count)], dtype=np.str_),
        5,
    )
    labels = np.tile(np.arange(5, dtype=np.int64), group_count)
    return labels, groups


def test_shared_reference_selector_is_exact_deterministic_and_class_complete() -> None:
    labels, groups = _balanced_group_fixture()

    first = deterministic_group_greedy_class_distribution_v1(
        labels,
        groups,
        class_order=(0, 1, 2, 3, 4),
        fraction=0.10,
        seed=223,
    )
    second = deterministic_group_greedy_class_distribution_v1(
        labels.copy(),
        groups.copy(),
        class_order=(0, 1, 2, 3, 4),
        fraction=0.10,
        seed=223,
    )

    assert first == second
    assert first == tuple(sorted(first))
    assert len(first) == math.ceil(len(np.unique(groups)) * 0.10)
    reference = np.isin(groups, first)
    assert set(labels[reference].tolist()) == {0, 1, 2, 3, 4}
    assert set(labels[~reference].tolist()) == {0, 1, 2, 3, 4}
    for group in np.unique(groups):
        members = reference[groups == group]
        assert bool(members.all()) or not bool(members.any())


def test_shared_reference_selector_fails_when_exact_group_count_cannot_cover_classes() -> None:
    labels = np.arange(5, dtype=np.int64)
    groups = np.asarray([f"source-patch-{index}" for index in range(5)], dtype=np.str_)

    with pytest.raises(
        ReferenceGroupSelectionError,
        match="reference-validation partition does not contain every frozen class",
    ):
        deterministic_group_greedy_class_distribution_v1(
            labels,
            groups,
            class_order=(0, 1, 2, 3, 4),
            fraction=0.10,
            seed=223,
        )


@pytest.mark.parametrize(
    ("labels", "groups", "message"),
    [
        (np.asarray([0, 1]), np.asarray(["a"]), "aligned vectors"),
        (np.asarray([0, 1]), np.asarray(["a", ""]), "non-empty strings"),
        (np.asarray([0, 1]), np.asarray(["a", "b"]), "every frozen class"),
    ],
)
def test_shared_reference_selector_rejects_invalid_inputs(
    labels: np.ndarray, groups: np.ndarray, message: str
) -> None:
    with pytest.raises(ReferenceGroupSelectionError, match=message):
        deterministic_group_greedy_class_distribution_v1(
            labels,
            groups,
            class_order=(0, 1, 2, 3, 4),
            fraction=0.10,
            seed=223,
        )
