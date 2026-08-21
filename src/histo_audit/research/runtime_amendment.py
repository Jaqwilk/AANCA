"""Fail-closed analysis for a pre-outcome autoresearch runtime amendment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from histo_audit.research.autoresearch import (
    AutoresearchCandidate,
    select_passing_winner,
)


def load_runtime_amendment(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the immutable runtime amendment."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime amendment must be a mapping")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported runtime amendment schema")
    finalists = value.get("full_finalists_in_frozen_order")
    if not isinstance(finalists, list) or not finalists:
        raise ValueError("runtime amendment has no frozen finalists")
    hashes = [str(item) for item in finalists]
    if len(hashes) != len(set(hashes)):
        raise ValueError("runtime amendment finalist hashes are not unique")
    if any(len(item) != 64 for item in hashes):
        raise ValueError("runtime amendment contains an invalid candidate hash")
    return value


def validate_parent_authority(amendment: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    """Verify immutable parent evaluator identities used by every result record."""

    expected = {
        "study_id": amendment["parent_study_id"],
        "config_sha256": amendment["parent_config_sha256"],
        "partition_sha256": amendment["partition_sha256"],
    }
    for name, value in expected.items():
        if authority.get(name) != value:
            raise ValueError(f"parent authority {name} does not match amendment")
    if authority.get("final_external_test_used") is not False:
        raise ValueError("parent authority used a forbidden external test")
    if authority.get("natural_error_detection_evaluated") is not False:
        raise ValueError("parent authority makes a natural-error evaluation claim")


def _validated_full_record(
    record: Mapping[str, Any],
    *,
    amendment: Mapping[str, Any],
    amended_budget_seconds: float,
) -> dict[str, Any]:
    if record.get("stage") != "full_nested":
        raise ValueError("amendment record is not a full_nested result")
    if record.get("config_sha256") != amendment["parent_config_sha256"]:
        raise ValueError("full record config hash differs from amendment")
    if record.get("partition_sha256") != amendment["partition_sha256"]:
        raise ValueError("full record partition hash differs from amendment")
    if record.get("final_external_test_used") is not False:
        raise ValueError("full record used a forbidden external test")
    if record.get("natural_error_detection_evaluated") is not False:
        raise ValueError("full record evaluates natural pathologist error")
    if record.get("source_annotations_modified") is not False:
        raise ValueError("full record reports source-annotation modification")
    if record.get("status") == "crash":
        raise ValueError("a frozen full finalist crashed; amended analysis is incomplete")

    candidate_value = record.get("candidate")
    if not isinstance(candidate_value, Mapping):
        raise ValueError("full record has no candidate mapping")
    candidate = AutoresearchCandidate.from_mapping(candidate_value)
    if candidate.candidate_sha256 != record.get("candidate_sha256"):
        raise ValueError("full record candidate hash is inconsistent")
    if not isinstance(record.get("retrieval"), Mapping) or not isinstance(
        record.get("downstream"), Mapping
    ):
        raise ValueError("full record is missing complete result summaries")

    elapsed = float(record.get("elapsed_seconds", math.nan))
    objective = float(record.get("objective", math.nan))
    if not math.isfinite(elapsed) or elapsed < 0.0 or not math.isfinite(objective):
        raise ValueError("full record has a non-finite runtime or objective")
    if elapsed > amended_budget_seconds:
        raise ValueError("a full finalist exceeded the frozen amended runtime budget")

    gates = record.get("success_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("full record has no individual success gates")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("full record success gates are not Boolean")
    amended_pass = all(bool(value) for value in gates.values())

    amended = deepcopy(dict(record))
    amended["parent_status"] = record.get("status")
    amended["parent_all_success_gates_pass"] = record.get("all_success_gates_pass")
    amended["runtime_amendment_applied"] = True
    amended["runtime_amendment_id"] = amendment["amendment_id"]
    amended["all_success_gates_pass"] = amended_pass
    amended["status"] = "keep" if amended_pass else "discard"
    return amended


def analyse_runtime_amendment(
    amendment: Mapping[str, Any],
    authority: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all frozen finalists and select a winner without changing metrics."""

    validate_parent_authority(amendment, authority)
    frozen = [str(value) for value in amendment["full_finalists_in_frozen_order"]]
    full_records = [record for record in records if record.get("stage") == "full_nested"]
    by_hash: dict[str, Mapping[str, Any]] = {}
    for record in full_records:
        candidate_hash = str(record.get("candidate_sha256", ""))
        if candidate_hash in by_hash:
            raise ValueError("parent ledger contains duplicate full finalist records")
        by_hash[candidate_hash] = record
    if set(by_hash) != set(frozen) or len(full_records) != len(frozen):
        missing = sorted(set(frozen) - set(by_hash))
        unexpected = sorted(set(by_hash) - set(frozen))
        raise ValueError(
            "parent full ledger is not the exact frozen finalist set; "
            f"missing={missing}, unexpected={unexpected}"
        )

    amended_budget = float(amendment["runtime_budget"]["amended_seconds_per_candidate"])
    amended_records = [
        _validated_full_record(
            by_hash[candidate_hash],
            amendment=amendment,
            amended_budget_seconds=amended_budget,
        )
        for candidate_hash in frozen
    ]
    winner = select_passing_winner(amended_records)
    selected = None
    if winner is not None:
        selected = next(
            record
            for record in amended_records
            if record["candidate_sha256"] == winner.candidate_sha256
        )
    selected_overlap_limitation = bool(
        selected is not None and str(selected["candidate"]["feature_view"]).startswith("phikon_v2")
    )
    overlap_free_records = [
        record
        for record in amended_records
        if not str(record["candidate"]["feature_view"]).startswith("phikon_v2")
    ]
    overlap_free_winner = select_passing_winner(overlap_free_records)
    overlap_free_selected = None
    if overlap_free_winner is not None:
        overlap_free_selected = next(
            record
            for record in overlap_free_records
            if record["candidate_sha256"] == overlap_free_winner.candidate_sha256
        )
    return {
        "schema_version": 1,
        "amendment_id": amendment["amendment_id"],
        "parent_study_id": amendment["parent_study_id"],
        "parent_config_sha256": amendment["parent_config_sha256"],
        "partition_sha256": amendment["partition_sha256"],
        "frozen_finalist_count": len(frozen),
        "amended_runtime_budget_seconds": amended_budget,
        "all_frozen_finalists_complete": True,
        "selected_candidate": winner.as_dict() if winner else None,
        "selected_candidate_sha256": winner.candidate_sha256 if winner else None,
        "selected_full_result": selected,
        "selected_candidate_tcga_pretraining_overlap_limitation": (selected_overlap_limitation),
        "best_overlap_free_candidate": (
            overlap_free_winner.as_dict() if overlap_free_winner else None
        ),
        "best_overlap_free_candidate_sha256": (
            overlap_free_winner.candidate_sha256 if overlap_free_winner else None
        ),
        "best_overlap_free_full_result": overlap_free_selected,
        "amended_full_results": amended_records,
        "development_disposition": (
            "candidate_frozen_for_genuinely_new_external_validation"
            if winner
            else "no_candidate_passed_all_nested_development_gates"
        ),
        "executable_action": "retain_uncorrected",
        "final_external_test_used": False,
        "natural_error_detection_proven": False,
        "real_use_superiority_proven": False,
        "source_annotations_modified": False,
    }


__all__ = [
    "analyse_runtime_amendment",
    "load_runtime_amendment",
    "validate_parent_authority",
]
