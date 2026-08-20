from __future__ import annotations

import numpy as np
import pytest

from histo_audit.auditing.stability import persistent_group_safe_risk
from histo_audit.auditing.two_queue import GROUP_SAFE_OOF_EVIDENCE


def test_persistent_risk_filters_transient_difficulty_across_three_models() -> None:
    history = np.asarray(
        [
            [[0.9, 0.9, 0.1], [0.9, 0.1, 0.2], [0.8, 0.1, 0.2]],
            [[0.8, 0.9, 0.2], [0.9, 0.1, 0.2], [0.8, 0.1, 0.1]],
            [[0.9, 0.9, 0.1], [0.8, 0.1, 0.2], [0.9, 0.1, 0.2]],
        ],
        dtype=np.float64,
    )
    result = persistent_group_safe_risk(
        history,
        evidence_role=GROUP_SAFE_OOF_EVIDENCE,
        risk_threshold=0.7,
        minimum_checkpoint_persistence=2 / 3,
        minimum_stable_model_fraction=2 / 3,
    )

    assert result.persistent_mask.tolist() == [True, False, False]
    assert result.persistence_weighted_priority[0] > 0.0
    assert result.persistence_weighted_priority[1:].tolist() == [0.0, 0.0]
    assert result.model_count == 3
    assert result.interpreted_as_proven_error is False


def test_persistent_risk_requires_three_to_five_group_safe_models() -> None:
    with pytest.raises(ValueError, match="3-to-5"):
        persistent_group_safe_risk(
            np.ones((2, 3, 4)),
            evidence_role=GROUP_SAFE_OOF_EVIDENCE,
            risk_threshold=0.5,
        )
    with pytest.raises(ValueError, match="group-safe OOF"):
        persistent_group_safe_risk(
            np.ones((3, 3, 4)),
            evidence_role="training_predictions",
            risk_threshold=0.5,
        )
