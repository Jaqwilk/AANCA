"""Persistent annotation-risk evidence across models and training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scores import percentile_normalise
from .two_queue import GROUP_SAFE_OOF_EVIDENCE


@dataclass(frozen=True, slots=True)
class PersistentRiskResult:
    """Separate stable disagreement from transient model difficulty."""

    mean_risk: NDArray[np.float64]
    risk_standard_deviation: NDArray[np.float64]
    checkpoint_persistence_by_model: NDArray[np.float64]
    stable_model_fraction: NDArray[np.float64]
    persistent_mask: NDArray[np.bool_]
    persistence_weighted_priority: NDArray[np.float64]
    model_count: int
    checkpoint_count: int
    risk_threshold: float
    minimum_checkpoint_persistence: float
    minimum_stable_model_fraction: float
    evidence_role: str
    interpreted_as_proven_error: bool = False


def persistent_group_safe_risk(
    risk_history: NDArray[np.generic],
    *,
    evidence_role: str,
    risk_threshold: float,
    minimum_checkpoint_persistence: float = 0.6,
    minimum_stable_model_fraction: float = 0.6,
) -> PersistentRiskResult:
    """Summarise 3-5 model histories while rejecting transient high-risk spikes.

    ``risk_history`` has shape ``(models, checkpoints, samples)`` and must contain
    risks generated without fitting on each scored sample or its source group.  This
    is a stability signal for review prioritisation, never a pathologist-error label.
    """

    history = np.asarray(risk_history, dtype=np.float64)
    if evidence_role != GROUP_SAFE_OOF_EVIDENCE:
        raise ValueError("risk histories must be group-safe OOF evidence")
    if (
        history.ndim != 3
        or not 3 <= history.shape[0] <= 5
        or history.shape[1] < 2
        or history.shape[2] < 1
        or not np.isfinite(history).all()
    ):
        raise ValueError(
            "risk_history must be finite with shape (3-to-5 models, >=2 checkpoints, samples)"
        )
    if not np.isfinite(risk_threshold):
        raise ValueError("risk_threshold must be finite")
    if not 0.0 <= minimum_checkpoint_persistence <= 1.0:
        raise ValueError("minimum_checkpoint_persistence must lie in [0, 1]")
    if not 0.0 <= minimum_stable_model_fraction <= 1.0:
        raise ValueError("minimum_stable_model_fraction must lie in [0, 1]")

    checkpoint_persistence = np.mean(history >= risk_threshold, axis=1)
    stable_model_fraction = np.mean(
        checkpoint_persistence >= minimum_checkpoint_persistence, axis=0
    )
    persistent = stable_model_fraction >= minimum_stable_model_fraction
    mean_risk = history.mean(axis=(0, 1))
    priority = percentile_normalise(mean_risk) * stable_model_fraction
    priority[~persistent] = 0.0
    return PersistentRiskResult(
        mean_risk=np.asarray(mean_risk, dtype=np.float64),
        risk_standard_deviation=np.asarray(history.std(axis=(0, 1)), dtype=np.float64),
        checkpoint_persistence_by_model=np.asarray(checkpoint_persistence, dtype=np.float64),
        stable_model_fraction=np.asarray(stable_model_fraction, dtype=np.float64),
        persistent_mask=np.asarray(persistent, dtype=bool),
        persistence_weighted_priority=np.asarray(priority, dtype=np.float64),
        model_count=history.shape[0],
        checkpoint_count=history.shape[1],
        risk_threshold=float(risk_threshold),
        minimum_checkpoint_persistence=float(minimum_checkpoint_persistence),
        minimum_stable_model_fraction=float(minimum_stable_model_fraction),
        evidence_role=evidence_role,
    )


__all__ = ["PersistentRiskResult", "persistent_group_safe_risk"]
