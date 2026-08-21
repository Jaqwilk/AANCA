from __future__ import annotations

from typing import Any

import pytest

from histo_audit.research.autoresearch import AutoresearchCandidate
from histo_audit.research.runtime_amendment import analyse_runtime_amendment


def _record(candidate: AutoresearchCandidate, *, passed: bool, elapsed: float) -> dict[str, Any]:
    gates = {
        "retrieval_lower_bound_gt_zero_vs_exact_matched_random": passed,
        "downstream_lower_bound_gt_zero_vs_uncorrected": passed,
        "downstream_lower_bound_gt_zero_vs_exact_matched_random": passed,
        "every_important_class_recall_lower_bound_gte_minus_0_01": passed,
        "direction_consistent_across_corruption_seeds": passed,
    }
    return {
        "stage": "full_nested",
        "candidate": candidate.as_dict(),
        "candidate_sha256": candidate.candidate_sha256,
        "config_sha256": "config",
        "partition_sha256": "partition",
        "status": "timeout" if passed else "discard",
        "all_success_gates_pass": False,
        "success_gates": gates,
        "objective": 0.2 if passed else -0.1,
        "elapsed_seconds": elapsed,
        "retrieval": {"candidate_minus_matched_random_precision": 0.3},
        "downstream": {"candidate_minus_uncorrected_macro_f1": 0.1},
        "final_external_test_used": False,
        "natural_error_detection_evaluated": False,
        "source_annotations_modified": False,
    }


def _amendment(candidates: list[AutoresearchCandidate]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "amendment_id": "test",
        "parent_study_id": "parent",
        "parent_config_sha256": "config",
        "partition_sha256": "partition",
        "full_finalists_in_frozen_order": [value.candidate_sha256 for value in candidates],
        "runtime_budget": {"amended_seconds_per_candidate": 1000},
    }


def _authority() -> dict[str, Any]:
    return {
        "study_id": "parent",
        "config_sha256": "config",
        "partition_sha256": "partition",
        "final_external_test_used": False,
        "natural_error_detection_evaluated": False,
    }


def test_amendment_recovers_scientific_gates_from_parent_timeout() -> None:
    candidate = AutoresearchCandidate()
    summary = analyse_runtime_amendment(
        _amendment([candidate]),
        _authority(),
        [_record(candidate, passed=True, elapsed=500.0)],
    )
    assert summary["selected_candidate_sha256"] == candidate.candidate_sha256
    assert summary["selected_full_result"]["status"] == "keep"
    assert summary["selected_full_result"]["parent_status"] == "timeout"
    assert summary["selected_candidate_tcga_pretraining_overlap_limitation"] is False
    assert summary["best_overlap_free_candidate_sha256"] == candidate.candidate_sha256
    assert summary["executable_action"] == "retain_uncorrected"


def test_amendment_requires_exact_complete_frozen_set() -> None:
    first = AutoresearchCandidate()
    second = AutoresearchCandidate(review_budget=0.1)
    with pytest.raises(ValueError, match="exact frozen finalist set"):
        analyse_runtime_amendment(
            _amendment([first, second]),
            _authority(),
            [_record(first, passed=False, elapsed=100.0)],
        )


def test_amendment_rejects_candidate_over_new_budget() -> None:
    candidate = AutoresearchCandidate()
    with pytest.raises(ValueError, match="amended runtime budget"):
        analyse_runtime_amendment(
            _amendment([candidate]),
            _authority(),
            [_record(candidate, passed=True, elapsed=1000.1)],
        )


def test_amendment_returns_a_complete_no_winner_summary() -> None:
    candidate = AutoresearchCandidate()
    summary = analyse_runtime_amendment(
        _amendment([candidate]),
        _authority(),
        [_record(candidate, passed=False, elapsed=100.0)],
    )
    assert summary["selected_candidate"] is None
    assert summary["best_overlap_free_candidate"] is None
    assert summary["development_disposition"] == (
        "no_candidate_passed_all_nested_development_gates"
    )
