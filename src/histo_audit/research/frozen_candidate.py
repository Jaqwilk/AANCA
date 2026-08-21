"""Validation for an AANCA candidate frozen after development-only selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from histo_audit.utils.run_tracking import sha256_file

from .autoresearch import AutoresearchCandidate

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentCandidate:
    """A selected research candidate that cannot self-authorise natural-data use."""

    record_id: str
    candidate: AutoresearchCandidate
    candidate_sha256: str
    selection_authority: dict[str, str]
    development_evidence: dict[str, Any]
    executable_action_until_new_confirmation: str
    external_confirmation_complete: bool = False
    natural_error_detection_proven: bool = False
    real_use_superiority_proven: bool = False
    automatic_annotation_change_permitted: bool = False
    source_annotations_modified: bool = False

    @property
    def natural_data_activation_permitted(self) -> bool:
        return (
            self.external_confirmation_complete
            and self.natural_error_detection_proven
            and self.real_use_superiority_proven
            and not self.automatic_annotation_change_permitted
        )


def _required_sha256(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"selected-candidate authority {name} is not a SHA-256")
    return value


def load_frozen_development_candidate(
    path: str | Path,
) -> FrozenDevelopmentCandidate:
    """Load a candidate only when its identity and fail-closed boundary are intact."""

    candidate_path = Path(path)
    checksum_path = candidate_path.with_suffix(candidate_path.suffix + ".sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        if sha256_file(candidate_path) != expected:
            raise ValueError("selected development candidate checksum differs")
    value = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("selected development candidate has an unsupported schema")
    if value.get("project") != "AANCA" or value.get("replacement_project_or_v2") is not False:
        raise ValueError("selected development candidate is not the current AANCA project")
    if value.get("role") != "development_candidate_only":
        raise ValueError("selected candidate must remain development-only")
    candidate_value = value.get("candidate")
    if not isinstance(candidate_value, dict):
        raise ValueError("selected candidate mapping is absent")
    candidate = AutoresearchCandidate.from_mapping(candidate_value)
    candidate_sha256 = str(value.get("candidate_sha256", ""))
    if candidate.candidate_sha256 != candidate_sha256:
        raise ValueError("selected candidate identity does not match its configuration")

    authority_value = value.get("selection_authority")
    if not isinstance(authority_value, dict):
        raise ValueError("selected candidate has no selection authority")
    authority = {
        name: _required_sha256(authority_value, name)
        for name in (
            "parent_config_sha256",
            "partition_sha256",
            "runtime_amendment_sha256",
            "parent_authority_sha256",
            "parent_ledger_sha256",
        )
    }
    for name in ("parent_study_id", "runtime_amendment_id"):
        item = authority_value.get(name)
        if not isinstance(item, str) or not item:
            raise ValueError(f"selected-candidate authority {name} is absent")
        authority[name] = item

    evidence = value.get("development_evidence")
    claims = value.get("claim_boundary")
    activation = value.get("activation")
    if (
        not isinstance(evidence, dict)
        or not isinstance(claims, dict)
        or not isinstance(activation, dict)
    ):
        raise ValueError("selected candidate evidence or claim boundary is absent")
    required_false = {
        "final_external_test_used": evidence.get("final_external_test_used"),
        "natural_error_detection_evaluated": evidence.get("natural_error_detection_evaluated"),
        "natural_error_detection_proven": claims.get("natural_error_detection_proven"),
        "real_use_superiority_proven": claims.get("real_use_superiority_proven"),
        "automatic_annotation_change_permitted": claims.get(
            "automatic_annotation_change_permitted"
        ),
        "source_annotations_modified": claims.get("source_annotations_modified"),
        "external_confirmation_complete": activation.get("external_confirmation_complete"),
    }
    if any(item is not False for item in required_false.values()):
        raise ValueError("development candidate overstates evidence or changes source labels")
    action = activation.get("executable_action_until_new_confirmation")
    if action != "retain_uncorrected":
        raise ValueError("development candidate is not fail-closed before confirmation")
    record_id = value.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("selected candidate record ID is absent")
    return FrozenDevelopmentCandidate(
        record_id=record_id,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        selection_authority=authority,
        development_evidence=dict(evidence),
        executable_action_until_new_confirmation=str(action),
    )


__all__ = ["FrozenDevelopmentCandidate", "load_frozen_development_candidate"]
