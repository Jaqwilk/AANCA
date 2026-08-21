from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing import (
    CROSS_FITTED_UTILITY_EVIDENCE,
    GROUP_SAFE_OOF_EVIDENCE,
    UTILITY_PRODUCT_PRIORITY,
    QueueConstraints,
    build_measured_utility_queue,
)


def _inputs() -> dict[str, object]:
    return {
        "annotation_inconsistency_scores": np.asarray([0.99, 0.90, 0.80, 0.70]),
        "expected_downstream_gain": np.asarray([0.03, 0.05, 0.12, 0.25]),
        "downstream_gain_lower_bound": np.asarray([0.02, 0.04, 0.10, 0.20]),
        "group_ids": ["p0", "p1", "p2", "p3"],
        "observed_labels": [0, 0, 1, 1],
        "sample_ids": ["s0", "s1", "s2", "s3"],
        "constraints": QueueConstraints(requested_count=2, max_per_group=1),
        "annotation_evidence_role": GROUP_SAFE_OOF_EVIDENCE,
        "utility_evidence_role": CROSS_FITTED_UTILITY_EVIDENCE,
    }


def test_measured_utility_queue_uses_the_frozen_product_priority() -> None:
    result = build_measured_utility_queue(**_inputs())

    expected_priority = result.annotation_risk_percentiles * np.asarray([0.02, 0.04, 0.10, 0.20])
    assert result.priority_method == UTILITY_PRODUCT_PRIORITY
    assert np.allclose(result.combined_priority_scores, expected_priority)
    assert result.model_improvement.selected_sample_ids == ("s2", "s1")
    assert result.source_annotations_modified is False


def test_measured_utility_queue_fails_closed_for_untrusted_utility() -> None:
    inputs = _inputs()
    inputs["utility_evidence_role"] = "same_sample_estimate"
    result = build_measured_utility_queue(**inputs)

    assert result.model_improvement.available is False
    assert "cross-fitted measured" in str(result.model_improvement.unavailable_reason)


def test_measured_utility_queue_rejects_non_oof_risk_and_invalid_bounds() -> None:
    inputs = _inputs()
    inputs["annotation_evidence_role"] = "in_sample"
    with pytest.raises(ValueError, match="group-safe OOF"):
        build_measured_utility_queue(**inputs)

    inputs = _inputs()
    inputs["downstream_gain_lower_bound"] = np.asarray([0.04, 0.04, 0.10, 0.20])
    with pytest.raises(ValueError, match="finite and aligned"):
        build_measured_utility_queue(**inputs)
