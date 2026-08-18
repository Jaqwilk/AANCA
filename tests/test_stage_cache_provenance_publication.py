"""Producer-side publication tests for primary frozen-cache provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.representations.cache_provenance import (
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    primary_cache_provenance_record,
    verify_frozen_cache_sidecar,
)


def _cache_metadata(sample_ids: np.ndarray[Any, Any], *, scope: str) -> dict[str, Any]:
    return build_frozen_cache_metadata(
        base_metadata={"schema_version": 1},
        sample_ids=sample_ids,
        manifest_sha256="a" * 64,
        raw_inventory_sha256="b" * 64,
        representation_id="test_representation",
        input_variant="test_input",
        encoder_identifier="test_encoder_v1",
        encoder_metadata={"architecture": "test"},
        encoder_implementation={"entrypoint": "test"},
        weight_identifier="unlearned:test_weights",
        weights_sha256="c" * 64,
        preprocessing_identifier="test_preprocessing_v1",
        preprocessing={"normalisation": "none"},
        cache_recipe={"identifier": "test_cache_v1"},
        dtype="float32",
        feature_dimension=2,
        package_versions={"numpy": np.__version__},
        matrix_key="values",
        provenance_scope=scope,
    )


def _arrays() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "sample_ids": np.asarray(["sample-0", "sample-1"], dtype=np.str_),
        "values": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }


def test_stage_cache_publishes_primary_projection_before_post_publish_check(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    destination = tmp_path / "stage-cache.npz"
    sidecar = destination.with_suffix(".npz.metadata.json")
    observed: dict[str, Any] = {}

    def post_publish_check() -> None:
        observed.update(json.loads(sidecar.read_text(encoding="utf-8")))

    _, _, complete = atomic_save_npz_with_sidecar(
        destination,
        arrays=arrays,
        metadata=_cache_metadata(arrays["sample_ids"], scope="stage_eligible"),
        post_publish_check=post_publish_check,
    )

    expected = primary_cache_provenance_record(complete)
    assert complete["primary_cache_provenance"] == expected
    assert observed["primary_cache_provenance"] == expected
    verification = verify_frozen_cache_sidecar(destination)
    assert verification.metadata["primary_cache_provenance"] == expected


def test_non_stage_cache_preserves_fixture_provenance_shape(tmp_path: Path) -> None:
    arrays = _arrays()
    destination = tmp_path / "fixture-cache.npz"
    _, sidecar, complete = atomic_save_npz_with_sidecar(
        destination,
        arrays=arrays,
        metadata=_cache_metadata(arrays["sample_ids"], scope="non_stage_fixture"),
    )

    assert "primary_cache_provenance" not in complete
    assert "primary_cache_provenance" not in json.loads(sidecar.read_text(encoding="utf-8"))
    assert "primary_cache_provenance" not in verify_frozen_cache_sidecar(destination).metadata
