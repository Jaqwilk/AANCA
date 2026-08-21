"""Compact, reconstructible machine evidence for audit and restoration runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.auditing.neighbours import NeighbourDisagreementResult
from histo_audit.evaluation.restoration import DownstreamEvaluation
from histo_audit.utils.run_tracking import atomic_write_npz


def _atomic_npz(path: Path, **arrays: Any) -> Path:
    """Write one compressed NPZ and atomically publish it at *path*."""
    return atomic_write_npz(path, arrays)


def write_neighbour_evidence(
    path: str | Path,
    result: NeighbourDisagreementResult,
    *,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
) -> Path:
    """Persist every fold-safe neighbour identity, group, distance, and derived decision."""

    identifiers = tuple(str(value) for value in sample_ids)
    groups = tuple(str(value) for value in group_ids)
    result.validate(identifiers, groups)
    if not identifiers:
        raise ValueError("neighbour evidence requires at least one audit sample")
    counts = np.asarray([len(values) for values in result.neighbour_ids], dtype=np.int64)
    if counts.shape != (len(identifiers),) or np.any(counts <= 0) or np.any(counts > result.k):
        raise ValueError("neighbour evidence counts must lie in [1, k]")
    width = int(counts.max())
    max_identifier_length = max(
        1,
        *(len(value) for value in identifiers),
        *(len(value) for row in result.neighbour_ids for value in row),
    )
    max_group_length = max(
        1,
        *(len(value) for value in groups),
        *(len(value) for row in result.neighbour_groups for value in row),
    )
    neighbour_ids = np.full((len(identifiers), width), "", dtype=f"<U{max_identifier_length}")
    neighbour_groups = np.full((len(identifiers), width), "", dtype=f"<U{max_group_length}")
    neighbour_distances = np.full((len(identifiers), width), np.nan, dtype=np.float64)
    for index, (ids, row_groups, distances) in enumerate(
        zip(
            result.neighbour_ids,
            result.neighbour_groups,
            result.neighbour_distances,
            strict=True,
        )
    ):
        count = len(ids)
        if len(row_groups) != count or len(distances) != count:
            raise ValueError("neighbour identity/group/distance rows must align")
        neighbour_ids[index, :count] = ids
        neighbour_groups[index, :count] = row_groups
        neighbour_distances[index, :count] = distances
    return _atomic_npz(
        Path(path),
        schema_version=np.asarray(1, dtype=np.int64),
        sample_ids=np.asarray(identifiers, dtype=np.str_),
        group_ids=np.asarray(groups, dtype=np.str_),
        risk_scores=np.asarray(result.risk_scores, dtype=np.float64),
        alternative_class_support=np.asarray(result.alternative_class_support, dtype=np.float64),
        suggested_class=np.asarray(result.suggested_class, dtype=np.int64),
        neighbour_count=counts,
        neighbour_ids=neighbour_ids,
        neighbour_groups=neighbour_groups,
        neighbour_distances=neighbour_distances,
        k=np.asarray(result.k, dtype=np.int64),
        metric=np.asarray(result.metric, dtype=np.str_),
    )


def write_restoration_evidence(
    path: str | Path,
    evaluation: DownstreamEvaluation,
    *,
    development_sample_ids: Sequence[str],
    development_group_ids: Sequence[str],
    pre_corruption_label: NDArray[np.integer] | Sequence[int],
    observed_label: NDArray[np.integer] | Sequence[int],
    is_injected_corruption: NDArray[np.bool_] | Sequence[bool],
    final_test_sample_ids: Sequence[str],
    final_test_group_ids: Sequence[str],
    final_test_reference_label: NDArray[np.integer] | Sequence[int],
    final_test_is_injected_corruption: NDArray[np.bool_] | Sequence[bool],
    class_order: Sequence[int],
) -> Path:
    """Persist selections, restored labels/masks, and every final-test prediction."""

    development_ids = tuple(str(value) for value in development_sample_ids)
    development_groups = tuple(str(value) for value in development_group_ids)
    final_ids = tuple(str(value) for value in final_test_sample_ids)
    final_groups = tuple(str(value) for value in final_test_group_ids)
    pre = np.asarray(pre_corruption_label, dtype=np.int64)
    observed = np.asarray(observed_label, dtype=np.int64)
    injected = np.asarray(is_injected_corruption, dtype=bool)
    final_reference = np.asarray(final_test_reference_label, dtype=np.int64)
    final_injected = np.asarray(final_test_is_injected_corruption, dtype=bool)
    n_development = len(development_ids)
    n_final = len(final_ids)
    if (
        len(development_groups) != n_development
        or pre.shape != (n_development,)
        or observed.shape != (n_development,)
        or injected.shape != (n_development,)
        or len(set(development_ids)) != n_development
    ):
        raise ValueError("restoration development evidence is not aligned and unique")
    if (
        len(final_groups) != n_final
        or final_reference.shape != (n_final,)
        or final_injected.shape != (n_final,)
        or len(set(final_ids)) != n_final
        or set(development_ids).intersection(final_ids)
        or final_injected.any()
    ):
        raise ValueError("restoration final-reference evidence is invalid")
    guided_indices = np.asarray(evaluation.audit_reviewed_indices, dtype=np.int64)
    guided = evaluation.audit_guided_restoration_evidence
    random_indices = np.stack(evaluation.random_reviewed_indices, axis=0).astype(
        np.int64, copy=False
    )
    random_evidence = evaluation.random_review_restoration_evidence
    random_runs = evaluation.random_review_restoration
    repeats = len(random_runs)
    if (
        not repeats
        or len(random_evidence) != repeats
        or random_indices.shape != (repeats, evaluation.review_budget_count)
    ):
        raise ValueError("random restoration evidence does not align with repeat count/budget")
    if any(run.review_seed is None for run in random_runs):
        raise ValueError("every random restoration repeat requires a saved seed")
    random_seeds = np.asarray([run.review_seed for run in random_runs], dtype=np.int64)
    random_reviewed_ids = np.asarray(
        [[development_ids[int(index)] for index in row] for row in random_indices],
        dtype=np.str_,
    ).reshape(repeats, evaluation.review_budget_count)
    random_restored_labels = np.stack(
        [item.restored_labels for item in random_evidence], axis=0
    ).astype(np.int64, copy=False)
    random_restored_masks = np.stack(
        [item.restored_mask for item in random_evidence], axis=0
    ).astype(bool, copy=False)
    random_probabilities = np.stack(
        [run.final_test_probabilities for run in random_runs], axis=0
    ).astype(np.float64, copy=False)
    random_predicted = np.stack(
        [run.final_test_predicted_class for run in random_runs], axis=0
    ).astype(np.int64, copy=False)
    guided_ids = np.asarray(
        [development_ids[int(index)] for index in guided_indices], dtype=np.str_
    )
    conditions = {
        "uncorrupted_reference_baseline": evaluation.uncorrupted_reference_baseline,
        "corrupted_observed_baseline": evaluation.corrupted_observed_baseline,
        "audit_guided_restoration": evaluation.audit_guided_restoration,
    }
    condition_arrays: dict[str, object] = {}
    for name, run in conditions.items():
        condition_arrays[f"{name}_final_test_probabilities"] = np.asarray(
            run.final_test_probabilities, dtype=np.float64
        )
        condition_arrays[f"{name}_final_test_predicted_class"] = np.asarray(
            run.final_test_predicted_class, dtype=np.int64
        )
    return _atomic_npz(
        Path(path),
        schema_version=np.asarray(1, dtype=np.int64),
        development_sample_ids=np.asarray(development_ids, dtype=np.str_),
        development_group_ids=np.asarray(development_groups, dtype=np.str_),
        pre_corruption_label=pre,
        observed_label=observed,
        is_injected_corruption=injected,
        final_test_sample_ids=np.asarray(final_ids, dtype=np.str_),
        final_test_group_ids=np.asarray(final_groups, dtype=np.str_),
        final_test_reference_label=final_reference,
        final_test_is_injected_corruption=final_injected,
        class_order=np.asarray(tuple(int(value) for value in class_order), dtype=np.int64),
        review_budget_fraction=np.asarray(evaluation.review_budget_fraction, dtype=np.float64),
        review_budget_count=np.asarray(evaluation.review_budget_count, dtype=np.int64),
        audit_guided_reviewed_indices=guided_indices,
        audit_guided_reviewed_sample_ids=guided_ids,
        audit_guided_restored_label=np.asarray(guided.restored_labels, dtype=np.int64),
        audit_guided_restored_mask=np.asarray(guided.restored_mask, dtype=bool),
        random_review_seeds=random_seeds,
        random_reviewed_indices=random_indices,
        random_reviewed_sample_ids=random_reviewed_ids,
        random_restored_label=random_restored_labels,
        random_restored_mask=random_restored_masks,
        random_review_restoration_final_test_probabilities=random_probabilities,
        random_review_restoration_final_test_predicted_class=random_predicted,
        **condition_arrays,
    )


__all__ = ["write_neighbour_evidence", "write_restoration_evidence"]
