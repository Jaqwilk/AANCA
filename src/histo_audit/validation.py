"""Strict numerical input contracts shared by maintained audit interfaces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def integer_vector(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    """Accept finite integer-valued labels without truncation or int64 overflow."""

    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite integer values")
    if raw.dtype.kind == "f":
        if (
            not np.isfinite(raw).all()
            or np.any(raw != np.floor(raw))
            or np.any(raw < -(2**63))
            or np.any(raw >= 2**63)
        ):
            raise ValueError(f"{name} must contain finite integer values within int64 range")
    elif raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an integer outside int64 range")
    return raw.astype(np.int64, copy=False)


def fixed_class_order(values: Sequence[int], *, minimum_classes: int = 2) -> tuple[int, ...]:
    """Preserve a declared class order while rejecting aliases and duplicates."""

    classes = tuple(int(value) for value in integer_vector(values, name="class_order"))
    if len(classes) < minimum_classes or len(set(classes)) != len(classes):
        raise ValueError(f"class_order must contain at least {minimum_classes} unique values")
    return classes


def probability_matrix(
    values: ArrayLike,
    *,
    n_samples: int | None = None,
    n_classes: int | None = None,
    minimum_classes: int = 2,
    sum_tolerance: float = 5.0e-7,
) -> NDArray[np.float64]:
    """Validate distributions without clipping or renormalising.

    The absolute row-sum tolerance admits a few float32 softmax rounding units,
    including after an adapter promotes those unchanged values to float64.
    """

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] < minimum_classes:
        raise ValueError("probabilities must be a non-empty sample-by-class matrix")
    if (n_samples is not None and matrix.shape[0] != n_samples) or (
        n_classes is not None and matrix.shape[1] != n_classes
    ):
        raise ValueError("probabilities must align with labels and class order")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("probabilities lie outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=sum_tolerance, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    return matrix
