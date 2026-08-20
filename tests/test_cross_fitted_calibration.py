from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing.calibration import (
    NEW_EXPERT_DEVELOPMENT_LABELS,
    cross_fitted_temperature_calibration,
)
from histo_audit.auditing.two_queue import GROUP_SAFE_OOF_EVIDENCE


def test_temperature_calibration_is_group_cross_fitted_and_fail_closed() -> None:
    labels = np.asarray([0, 1] * 4, dtype=np.int64)
    probabilities = np.asarray([[0.6, 0.4], [0.4, 0.6]] * 4, dtype=np.float64)
    groups = [group for group in ("a", "b", "c", "d") for _ in range(2)]
    result = cross_fitted_temperature_calibration(
        probabilities,
        labels,
        groups,
        [f"s{index}" for index in range(len(labels))],
        class_order=(0, 1),
        evidence_role=NEW_EXPERT_DEVELOPMENT_LABELS,
        probability_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
        n_splits=4,
        split_seed=17,
    )

    assert result.group_safe_cross_fitted is True
    assert result.final_external_test_used is False
    assert np.array_equal(np.sort(np.unique(result.fold_id)), np.arange(4))
    assert len(result.fold_temperatures) == 4
    assert result.calibrated_negative_log_likelihood < result.uncalibrated_negative_log_likelihood
    assert result.calibration_adopted is True
    assert result.frozen_scaler is not None
    transformed = result.frozen_scaler.transform(probabilities)
    assert np.allclose(transformed.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(transformed, axis=1), np.argmax(probabilities, axis=1))


def test_temperature_calibration_rejects_nonexpert_or_reused_sample_evidence() -> None:
    probabilities = np.asarray([[0.6, 0.4], [0.4, 0.6]] * 2, dtype=np.float64)
    common = {
        "probabilities": probabilities,
        "expert_reference_labels": [0, 1, 0, 1],
        "group_ids": ["a", "a", "b", "b"],
        "sample_ids": ["s1", "s2", "s3", "s4"],
        "class_order": (0, 1),
        "probability_evidence_role": GROUP_SAFE_OOF_EVIDENCE,
        "n_splits": 2,
    }
    with pytest.raises(ValueError, match="newly collected expert"):
        cross_fitted_temperature_calibration(**common, evidence_role="nucls_final")

    common["probability_evidence_role"] = "training_resubstitution"
    with pytest.raises(ValueError, match="group-safe OOF"):
        cross_fitted_temperature_calibration(
            **common,
            evidence_role=NEW_EXPERT_DEVELOPMENT_LABELS,
        )

    common["probability_evidence_role"] = GROUP_SAFE_OOF_EVIDENCE
    common["sample_ids"] = ["s1", "s1", "s3", "s4"]
    with pytest.raises(ValueError, match="unique sample IDs"):
        cross_fitted_temperature_calibration(
            **common,
            evidence_role=NEW_EXPERT_DEVELOPMENT_LABELS,
        )
