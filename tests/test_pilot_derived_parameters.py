from __future__ import annotations

import numpy as np
import pytest

from histo_audit.experiment.pilot_derived_parameters import (
    PilotParameterDerivationError,
    derive_clean_oof_primary_parameters,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = np.repeat(np.asarray([f"g-{index:02d}" for index in range(10)]), 5)
    labels = np.tile(np.arange(5, dtype=np.int64), 10)
    tissues = np.asarray(["tissue-a" if index < 25 else "tissue-b" for index in range(50)])
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(50)])
    rng = np.random.default_rng(17)
    features = rng.normal(size=(50, 8)).astype(np.float32)
    features[:, 0] += labels * 0.35
    return features, labels, groups, tissues, sample_ids


def test_pilot_parameter_derivation_is_deterministic_and_reconciled() -> None:
    values = _fixture()
    first = derive_clean_oof_primary_parameters(*values, n_splits=5)
    second = derive_clean_oof_primary_parameters(*values, n_splits=5)

    assert first == second
    assert first["audit_pool"]["sample_count"] == 50
    assert first["audit_pool"]["group_count"] == 10
    confusion = np.asarray(first["clean_group_oof"]["confusion_matrix_rows_true_columns_predicted"])
    assert confusion.shape == (5, 5)
    assert int(confusion.sum()) == 50
    transition = np.asarray(first["confusion_targeted_corruption"]["transition_matrix"])
    assert np.all(np.diag(transition) == 0.0)
    assert np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    tissue = first["group_conditional_corruption"]
    assert set(tissue["weights_by_value"]) == {"tissue-a", "tissue-b"}
    assert max(tissue["weights_by_value"].values()) == 1.0
    assert 0.0 < tissue["default_weight"] <= 1.0


def test_pilot_parameter_derivation_rejects_missing_fixed_class() -> None:
    features, labels, groups, tissues, sample_ids = _fixture()
    keep = labels != 4
    with pytest.raises(PilotParameterDerivationError, match="all fixed classes"):
        derive_clean_oof_primary_parameters(
            features[keep],
            labels[keep],
            groups[keep],
            tissues[keep],
            sample_ids[keep],
            n_splits=2,
        )
