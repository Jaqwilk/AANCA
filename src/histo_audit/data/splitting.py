"""Outer-fold and source-group-safe development partition helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .manifest import assert_group_partitions_disjoint


@dataclass(frozen=True, slots=True)
class OuterAuditSplit:
    """Audit, reference-validation, and untouched final-reference partitions."""

    audit_indices: NDArray[np.int64]
    reference_validation_indices: NDArray[np.int64]
    final_test_indices: NDArray[np.int64]
    audit_groups: tuple[str, ...]
    reference_validation_groups: tuple[str, ...]
    final_test_groups: tuple[str, ...]
    final_test_fold: int
    split_seed: int

    def validate(self, n_samples: int) -> None:
        partitions = (
            self.audit_indices,
            self.reference_validation_indices,
            self.final_test_indices,
        )
        concatenated = np.concatenate(partitions)
        if len(concatenated) != n_samples or not np.array_equal(
            np.sort(concatenated), np.arange(n_samples)
        ):
            raise ValueError("outer split must cover every sample exactly once")
        assert_group_partitions_disjoint(
            (
                self.audit_groups,
                self.reference_validation_groups,
                self.final_test_groups,
            )
        )


def make_outer_audit_split(
    official_folds: Sequence[int] | NDArray[np.integer],
    group_ids: Sequence[str],
    *,
    final_test_fold: int,
    reference_validation_fraction: float = 0.10,
    seed: int = 41,
) -> OuterAuditSplit:
    """Hold out an official fold, then reserve development groups for validation."""

    folds = np.asarray(official_folds, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    if folds.ndim != 1 or groups.shape != folds.shape or not len(folds):
        raise ValueError("official folds and groups must be non-empty aligned vectors")
    if not 0.0 < reference_validation_fraction < 1.0:
        raise ValueError("reference_validation_fraction must lie in (0, 1)")
    if final_test_fold not in set(int(value) for value in folds):
        raise ValueError("requested final test fold is absent")
    for group in np.unique(groups):
        group_folds = np.unique(folds[groups == group])
        if len(group_folds) != 1:
            raise ValueError(f"source group {group!s} spans official folds")
    final_mask = folds == final_test_fold
    development_groups = np.unique(groups[~final_mask])
    if len(development_groups) < 2:
        raise ValueError("at least two development groups are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(development_groups)
    validation_group_count = max(
        1,
        min(
            len(development_groups) - 1,
            int(np.ceil(len(development_groups) * reference_validation_fraction)),
        ),
    )
    validation_groups = set(str(value) for value in shuffled[:validation_group_count])
    validation_mask = (~final_mask) & np.isin(groups, tuple(validation_groups))
    audit_mask = (~final_mask) & (~validation_mask)
    indices = np.arange(len(folds), dtype=np.int64)
    result = OuterAuditSplit(
        audit_indices=indices[audit_mask],
        reference_validation_indices=indices[validation_mask],
        final_test_indices=indices[final_mask],
        audit_groups=tuple(sorted(str(value) for value in np.unique(groups[audit_mask]))),
        reference_validation_groups=tuple(
            sorted(str(value) for value in np.unique(groups[validation_mask]))
        ),
        final_test_groups=tuple(sorted(str(value) for value in np.unique(groups[final_mask]))),
        final_test_fold=final_test_fold,
        split_seed=seed,
    )
    result.validate(len(folds))
    return result


def make_fractional_outer_audit_split(
    group_ids: Sequence[str],
    *,
    final_test_fraction: float = 0.20,
    reference_validation_fraction: float = 0.10,
    seed: int = 41,
) -> OuterAuditSplit:
    """Create three source-group-safe partitions for synthetic validation data."""

    groups = np.asarray(group_ids, dtype=np.str_)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("group IDs must be a non-empty vector")
    if not 0.0 < final_test_fraction < 1.0:
        raise ValueError("final_test_fraction must lie in (0, 1)")
    if not 0.0 < reference_validation_fraction < 1.0:
        raise ValueError("reference_validation_fraction must lie in (0, 1)")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    final_count = max(1, min(len(shuffled) - 2, int(np.ceil(len(shuffled) * final_test_fraction))))
    remaining_count = len(shuffled) - final_count
    validation_count = max(
        1,
        min(remaining_count - 1, int(np.ceil(remaining_count * reference_validation_fraction))),
    )
    final_groups = set(str(value) for value in shuffled[:final_count])
    validation_groups = set(
        str(value) for value in shuffled[final_count : final_count + validation_count]
    )
    final_mask = np.isin(groups, tuple(final_groups))
    validation_mask = np.isin(groups, tuple(validation_groups))
    audit_mask = ~(final_mask | validation_mask)
    indices = np.arange(len(groups), dtype=np.int64)
    result = OuterAuditSplit(
        audit_indices=indices[audit_mask],
        reference_validation_indices=indices[validation_mask],
        final_test_indices=indices[final_mask],
        audit_groups=tuple(sorted(str(value) for value in np.unique(groups[audit_mask]))),
        reference_validation_groups=tuple(
            sorted(str(value) for value in np.unique(groups[validation_mask]))
        ),
        final_test_groups=tuple(sorted(str(value) for value in np.unique(groups[final_mask]))),
        final_test_fold=-1,
        split_seed=seed,
    )
    result.validate(len(groups))
    return result
