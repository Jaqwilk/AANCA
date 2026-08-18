"""Shared deterministic fixtures for the synthetic scientific core."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synthetic_dataset():
    from histo_audit.data.synthetic import generate_synthetic_dataset

    return generate_synthetic_dataset(n_groups=18, instances_per_group=7, patch_size=48, seed=2027)
