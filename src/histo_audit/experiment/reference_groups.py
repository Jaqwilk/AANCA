"""Deterministic, source-group-safe reference-validation selection.

The selector is shared by the primary and confirmatory PanNuke adapters so both
study stages implement one frozen allocation rule.  It uses only the immutable
pre-corruption class labels and source-patch group identifiers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ReferenceGroupSelectionError(ValueError):
    """Raised when the frozen group-safe allocation cannot be constructed."""


def deterministic_group_greedy_class_distribution_v1(
    pre_corruption_labels: Sequence[int] | NDArray[np.integer[Any]],
    group_ids: Sequence[str] | NDArray[np.str_],
    *,
    class_order: tuple[int, ...],
    fraction: float,
    seed: int,
) -> tuple[str, ...]:
    """Select an exact group count with deterministic class-distribution matching.

    The target is ``ceil(number_of_groups * fraction)`` (clipped to leave a
    non-empty audit partition).  At each step, the candidate key is the L1
    divergence from the global class distribution, then absolute sample-fraction
    error, then a seeded group permutation used only as a tie-break, and finally
    the group identifier.  The completed split fails closed unless every frozen
    class occurs in both partitions.
    """

    labels = np.asarray(pre_corruption_labels, dtype=np.int64)
    raw_groups = np.asarray(group_ids)
    if labels.ndim != 1 or raw_groups.ndim != 1 or raw_groups.shape != labels.shape:
        raise ReferenceGroupSelectionError("labels and groups must be aligned vectors")
    if not len(labels):
        raise ReferenceGroupSelectionError("labels and groups must be non-empty")
    if not class_order or len(set(class_order)) != len(class_order):
        raise ReferenceGroupSelectionError("class_order must contain unique classes")
    if raw_groups.dtype.kind not in {"O", "S", "U"}:
        raise ReferenceGroupSelectionError("group identifiers must be strings")
    raw_values = raw_groups.tolist()
    if any(not isinstance(value, str) or not value for value in raw_values):
        raise ReferenceGroupSelectionError("group identifiers must be non-empty strings")
    groups = np.asarray(raw_values, dtype=np.str_)
    required_classes = set(class_order)
    if set(int(value) for value in labels.tolist()) != required_classes:
        raise ReferenceGroupSelectionError("labels must contain exactly every frozen class")
    if not 0.0 < fraction < 1.0:
        raise ReferenceGroupSelectionError("reference fraction must lie in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReferenceGroupSelectionError("seed must be a nonnegative integer")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ReferenceGroupSelectionError("at least two source groups are required")
    target_count = max(1, min(len(unique_groups) - 1, math.ceil(len(unique_groups) * fraction)))
    class_to_column = {value: index for index, value in enumerate(class_order)}
    counts: dict[str, NDArray[np.float64]] = {}
    sizes: dict[str, int] = {}
    for raw_group in unique_groups:
        group = str(raw_group)
        members = labels[groups == raw_group]
        vector = np.zeros(len(class_order), dtype=np.float64)
        for label in members:
            vector[class_to_column[int(label)]] += 1.0
        counts[group] = vector
        sizes[group] = len(members)

    global_counts = sum(counts.values(), start=np.zeros(len(class_order), dtype=np.float64))
    global_distribution = global_counts / global_counts.sum()
    permutation = np.random.default_rng(seed).permutation(unique_groups)
    tie_rank = {str(group): rank for rank, group in enumerate(permutation)}
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_counts = np.zeros(len(class_order), dtype=np.float64)
    selected_size = 0
    total_size = len(labels)
    while len(selected) < target_count:
        candidates: list[tuple[tuple[float, float, int, str], str]] = []
        for raw_group in unique_groups:
            group = str(raw_group)
            if group in selected_set:
                continue
            candidate_counts = selected_counts + counts[group]
            candidate_distribution = candidate_counts / candidate_counts.sum()
            divergence = float(np.abs(candidate_distribution - global_distribution).sum())
            sample_error = abs((selected_size + sizes[group]) / total_size - fraction)
            candidates.append(((divergence, sample_error, tie_rank[group], group), group))
        _, chosen = min(candidates, key=lambda item: item[0])
        selected.append(chosen)
        selected_set.add(chosen)
        selected_counts += counts[chosen]
        selected_size += sizes[chosen]

    reference_mask = np.isin(groups, selected)
    audit_mask = ~reference_mask
    if set(int(value) for value in labels[reference_mask].tolist()) != required_classes:
        raise ReferenceGroupSelectionError(
            "reference-validation partition does not contain every frozen class"
        )
    if set(int(value) for value in labels[audit_mask].tolist()) != required_classes:
        raise ReferenceGroupSelectionError("audit partition does not contain every frozen class")
    return tuple(sorted(selected))


__all__ = [
    "ReferenceGroupSelectionError",
    "deterministic_group_greedy_class_distribution_v1",
]
