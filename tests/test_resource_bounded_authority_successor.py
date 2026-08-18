from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from histo_audit.config import load_config
from histo_audit.workflows import preregistration_amendment as amendment


def _execution_source(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: str(record["path"]))
    return {
        "schema_version": 3,
        "scope_kind": "execution_source",
        "scope": ["src/**", "configs/**", "pyproject.toml", "uv.lock"],
        "excluded_roots": [".git", ".venv", "artifacts", "data"],
        "excluded_paths": [
            "configs/confirmatory_frozen.yaml",
            "configs/primary_frozen.yaml",
        ],
        "artifact_count": len(ordered),
        "root_sha256": amendment._canonical_root(ordered),
        "artifacts": ordered,
    }


def test_technical_successor_source_allowlist_is_exact_and_glob_free() -> None:
    parent = _execution_source([{"path": "src/a.py", "size_bytes": 1, "sha256": "1" * 64}])
    successor = _execution_source([{"path": "src/a.py", "size_bytes": 2, "sha256": "2" * 64}])

    delta, digest = amendment._canonical_source_delta_with_allowlist(
        parent,
        successor,
        allowlisted_change_kinds={"src/a.py": "modified"},
        role="test successor",
    )

    assert [record["path"] for record in delta] == ["src/a.py"]
    assert digest == amendment._canonical_value_sha256(list(delta))
    with pytest.raises(ValueError, match="exact execution-source path"):
        amendment._canonical_source_delta_with_allowlist(
            parent,
            successor,
            allowlisted_change_kinds={"src/*.py": "modified"},
            role="test successor",
        )
    with pytest.raises(ValueError, match="closed allowlist"):
        amendment._canonical_source_delta_with_allowlist(
            parent,
            successor,
            allowlisted_change_kinds={"src/a.py": "removed"},
            role="test successor",
        )


def _real_resource_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    authority_c = (
        project_root
        / "artifacts"
        / "preregistration_amendments"
        / "20260727T170413.080954Z"
        / "confirmatory_frozen.yaml"
    )
    successor_config = project_root / "configs" / "confirmatory_resource_bounded_amended.yaml"
    if not authority_c.is_file() or not successor_config.is_file():
        pytest.skip("real authority-C/successor config pair is unavailable")
    return load_config(authority_c), load_config(successor_config)


def test_technical_successor_config_changes_only_two_logical_cnn_digests() -> None:
    parent, successor = _real_resource_configs()

    before, after = amendment._require_resource_technical_config_correction(
        parent,
        successor,
    )

    changed = {
        field
        for field in amendment._RESOURCE_BOUNDED_CNN_PROVENANCE_FIELDS
        if before[field] != after[field]
    }
    assert changed == {
        "encoder_metadata_sha256",
        "preprocessing_sha256",
    }


def test_technical_successor_config_rejects_scientific_or_noop_change() -> None:
    parent, successor = _real_resource_configs()
    changed_training = copy.deepcopy(successor)
    changed_training["training"]["max_epochs"] = 5

    with pytest.raises(ValueError, match="outside the exact logical CNN provenance"):
        amendment._require_resource_technical_config_correction(
            parent,
            changed_training,
        )
    with pytest.raises(ValueError, match="must change exactly"):
        amendment._require_resource_technical_config_correction(parent, parent)


def test_capacity_v3_extends_but_does_not_rewrite_authority_c_capacity_v2() -> None:
    parent = amendment._RESOURCE_BOUNDED_CAPACITY
    successor = amendment._RESOURCE_BOUNDED_CAPACITY_V3

    assert parent["schema_version"] == 2
    assert parent["policy"] == "resource_bounded_confirmatory_capacity_v2"
    assert "workspace_layout_policy" not in parent
    assert successor["schema_version"] == 3
    assert successor["policy"] == "resource_bounded_confirmatory_capacity_v3"
    assert successor["planned_required_cells"] == 24
    assert successor["planned_cnn_cells"] == 6
    assert successor["planned_cnn_fold_checkpoints"] == 30
    assert successor["workspace_source_array_count"] == 12
    assert successor["workspace_partition_count"] == 9
    assert successor["workspace_shared_backing_bytes"] == 4_294_182_269
    assert successor["workspace_partition_index_bytes"] == 4_521_144
    assert successor["projected_workspace_bytes"] == 4_298_703_413
    assert successor["maximum_workspace_bytes"] == 4_567_138_869
    assert successor["minimum_free_bytes_before_workspace_build"] == 28_189_458_997
    assert successor["minimum_free_bytes_before_tracker"] == 23_622_320_128
    for field, value in parent.items():
        if field not in {"schema_version", "policy"}:
            assert successor[field] == value


def test_authority_c_retains_historical_registered_config_semantics() -> None:
    assert amendment.RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256 == (
        "1c9a41b92dabbeafbb92b1bc8aced158337046fc1d6e056b011f6a27b98e8298"
    )


def test_real_failed_preflight_receipt_is_exact_and_live_hash_bound() -> None:
    project_root = Path(__file__).resolve().parents[1]
    authority_c = (
        project_root / "artifacts" / "preregistration_amendments" / "20260727T170413.080954Z"
    ).resolve()
    receipt = (
        project_root
        / "artifacts"
        / "resource_control"
        / "failed_resource_preflight_20260727T173054.689Z.json"
    ).resolve()
    if not authority_c.is_dir() or not receipt.is_file():
        pytest.skip("real authority-C failed-preflight receipt is unavailable")
    parent_state = amendment._authority_state(
        authority_c,
        visited=set(),
        depth=0,
        max_chain_depth=64,
    )
    parent_authorization = amendment._require_sealed_resource_bounded_confirmatory_authorization(
        authority_c
    )
    evidence = json.loads(receipt.read_text(encoding="utf-8"))

    canonical = amendment._canonical_failed_resource_preflight(
        {
            "receipt_path": str(receipt),
            "receipt_sha256": amendment.sha256_file(receipt),
            "evidence": evidence,
        },
        resource_parent=parent_state,
        resource_authorization_sha256=amendment._canonical_mapping_sha256(parent_authorization),
        verify_live_receipt=True,
    )

    assert canonical["receipt_sha256"] == (
        "e308aa0089a84caaca3f0722711e623579372636d47e161d9f32ca5a71f8c6eb"
    )
    assert len(canonical["evidence"]) == 35
    assert canonical["evidence"]["run_tracker_created"] is False
    assert canonical["evidence"]["scientific_outcome_values_read"] is False
