"""Typed pre-lifecycle authority for the unchanged original confirmatory study.

This authority is deliberately operational and outcome-split.  It never changes
the frozen scientific definition.  It binds the already inspected historical
primary lineage while attesting that confirmatory outcome values remain unread.
Launcher, supervisor, session, Q, E, and terminal-composition identities are
strictly downstream and are forbidden here to keep the dependency graph acyclic.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND: Final = (
    "original_confirmatory_technical_authority_v1"
)
INTENT_POLICY: Final = "original_confirmatory_technical_authority_intent_v1"
REVIEW_POLICY: Final = "original_confirmatory_technical_authority_independent_review_v1"
EVIDENCE_POLICY: Final = "original_confirmatory_technical_authority_evidence_v1"
MANIFEST_POLICY: Final = "original_confirmatory_technical_authority_manifest_v1"
ATTEMPT_POLICY: Final = "original_confirmatory_technical_authority_publication_attempt_v1"
SUCCESS_POLICY: Final = "original_confirmatory_technical_authority_publication_success_v1"
STOP_POLICY: Final = "original_confirmatory_technical_authority_publication_stop_v1"

INTENT_FILENAME: Final = "technical_authority_intent.json"
REVIEW_FILENAME: Final = "independent_review.json"
EVIDENCE_FILENAME: Final = "technical_authority_evidence.json"
PREREGISTRATION_FILENAME: Final = "PRE_REGISTRATION_FROZEN.md"
PRIMARY_CONFIG_FILENAME: Final = "primary_frozen.yaml"
CONFIRMATORY_CONFIG_FILENAME: Final = "confirmatory_frozen.yaml"
SOURCE_INVENTORY_FILENAME: Final = "source_inventory.json"
CAPSULE_BINDING_FILENAME: Final = "capsule_binding.json"
CAPACITY_BINDING_FILENAME: Final = "capacity_binding.json"
MANIFEST_FILENAME: Final = "sha256_manifest.json"
IMMUTABLE_MARKER_FILENAME: Final = ".immutable.json"
ATTEMPT_FILENAME: Final = "publication_attempt.json"
SUCCESS_FILENAME: Final = "publication_success.json"
STOP_FILENAME: Final = "publication_stop.json"

CORE_FILENAMES: Final = frozenset(
    {
        INTENT_FILENAME,
        REVIEW_FILENAME,
        EVIDENCE_FILENAME,
        PREREGISTRATION_FILENAME,
        PRIMARY_CONFIG_FILENAME,
        CONFIRMATORY_CONFIG_FILENAME,
        SOURCE_INVENTORY_FILENAME,
        CAPSULE_BINDING_FILENAME,
        CAPACITY_BINDING_FILENAME,
    }
)
QUALIFYING_FILENAMES: Final = CORE_FILENAMES | {
    MANIFEST_FILENAME,
    IMMUTABLE_MARKER_FILENAME,
    ATTEMPT_FILENAME,
    SUCCESS_FILENAME,
}

PARENT_RELATIVE_DIRECTORY: Final = "artifacts/preregistration_amendments/20260727T133947.089370Z"
PARENT_ARTIFACT_ROOT_SHA256: Final = (
    "4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a"
)
PARENT_MANIFEST_SHA256: Final = "b5efc656f074b2933138b1a623de24e099bea9e7e2b75edc8e484d22ca176d10"
PARENT_SOURCE_ROOT_SHA256: Final = (
    "ba7fb4c8336c4f9ba138fcda16019dc31bec7e5cc3e8b846e643d6dd0332601b"
)
PARENT_SOURCE_MANIFEST_SHA256: Final = (
    "0f5e33259962c5b8f2bf5e3c11a776bdfbf280f6e3cded906e1222e89e1a4df2"
)
PARENT_CHAIN_DEPTH: Final = 2

PREREGISTRATION_SHA256: Final = "7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b"
PRIMARY_CONFIG_SHA256: Final = "0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9"
PRIMARY_CONFIG_SEMANTIC_SHA256: Final = (
    "c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15"
)
CONFIRMATORY_CONFIG_SHA256: Final = (
    "4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009"
)
CONFIRMATORY_CONFIG_SEMANTIC_SHA256: Final = (
    "ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b"
)

HISTORICAL_PRIMARY_RUN_ID: Final = "20260727T133947.089370Z_pannuke_primary_orphan_recovery"
HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256: Final = (
    "8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4"
)
HISTORICAL_PRIMARY_MANIFEST_SHA256: Final = (
    "9abff1b2f0e745a50b3aa1922d3d725bb629276f6090c32e9b423fea82d0e0ce"
)
PRIMARY_OUTCOME_INSPECTION_AT_UTC: Final = "2026-07-27T10:57:07.000000Z"

CAPACITY_SCHEMA_VERSION: Final = 2
CAPACITY_POLICY_NAME: Final = "original_confirmatory_sealed_plan_capacity_v2"
CAPACITY_POLICY_SHA256: Final = "b83d3e8e1693a640b8a306a1e5a4b7722fe323bedeefcff8aa1d29c7927bf284"
CAPACITY_REQUIRED_FREE_BYTES: Final = 75_161_927_680

FORBIDDEN_DOWNSTREAM_FIELDS: Final = (
    "codex_session",
    "e_intent",
    "launcher",
    "q_authority",
    "saved_session",
    "supervisor",
    "terminal_composition",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class OriginalConfirmatoryTechnicalAuthorityError(ValueError):
    """The typed pre-lifecycle authority failed closed."""


@dataclass(frozen=True, slots=True)
class OriginalConfirmatoryTechnicalAuthorityBundle:
    """Deterministic bytes consumed by the one-use directory publisher."""

    authority_directory: Path
    artifacts: Mapping[str, bytes]
    sha256_manifest_bytes: bytes
    immutable_marker_bytes: bytes
    publication_attempt_bytes: bytes
    publication_success_bytes: bytes
    publication_stop_bytes: bytes
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    technical_authorization_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedOriginalConfirmatoryTechnicalAuthority:
    """Stable verified identity consumed by lifecycle and Q construction."""

    authority_directory: Path
    chain_depth: int
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    execution_source_manifest_sha256: str
    execution_source_root_sha256: str
    parent_authority_directory: Path
    parent_artifact_root_sha256: str
    parent_sha256_manifest_sha256: str
    technical_authorization_sha256: str
    independent_review_receipt_sha256: str
    immutable_marker_sha256: str
    publication_attempt_sha256: str
    publication_success_sha256: str

    def lifecycle_binding(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "policy": "original_confirmatory_technical_authority_lifecycle_binding_v1",
            **{
                key: str(item) if isinstance(item, Path) else item
                for key, item in asdict(self).items()
            },
            "primary_outcomes_inspected": True,
            "confirmatory_outcomes_inspected": False,
            "confirmatory_outcome_values_read": False,
            "scientific_definition_changed": False,
            "automatic_retry_allowed": False,
        }
        return {
            **value,
            "binding_sha256": canonical_json_sha256(value),
        }


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "value is not strict canonical JSON"
        ) from exc


def canonical_json_line_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} must be one lowercase SHA-256")
    return value


def _absolute_path(value: Any, *, role: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            f"{role} must be one absolute canonical path"
        )
    path = Path(value)
    _drive, tail = os.path.splitdrive(str(path))
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or (os.name == "nt" and ":" in tail)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            f"{role} must be one absolute canonical path"
        )
    return path


def _mapping(value: Any, *, fields: set[str], role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} must be an object")
    result = dict(value)
    if set(result) != fields or any(type(key) is not str for key in result):
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} has an unexpected field set")
    return result


def _utc(value: Any, *, role: str) -> datetime:
    if type(value) is not str or value != value.strip() or not value.endswith("Z"):
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} must be explicit ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            f"{role} must be explicit ISO-8601 UTC"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} must be UTC")
    rendered = parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value != rendered:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            f"{role} must use canonical microsecond UTC"
        )
    return parsed


def _rooted(value: Mapping[str, Any], root_field: str, *, role: str) -> dict[str, Any]:
    raw = dict(value)
    root = _sha256(raw.pop(root_field, None), role=f"{role} root")
    if canonical_json_sha256(raw) != root:
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} self-root is invalid")
    return {**raw, root_field: root}


_PARENT_FIELDS = {
    "schema_version",
    "authority_kind",
    "authority_directory",
    "chain_depth",
    "artifact_root_sha256",
    "sha256_manifest_sha256",
    "execution_source_root_sha256",
    "execution_source_manifest_sha256",
}
_FROZEN_FIELDS = {
    "schema_version",
    "preregistration_path",
    "preregistration_sha256",
    "primary_config_path",
    "primary_config_sha256",
    "primary_config_semantic_sha256",
    "confirmatory_config_path",
    "confirmatory_config_sha256",
    "confirmatory_config_semantic_sha256",
    "scientific_definition_changed",
}
_PRIMARY_FIELDS = {
    "schema_version",
    "run_directory",
    "run_id",
    "terminal_status",
    "completion_stage",
    "artifact_root_sha256",
    "artifact_manifest_sha256",
    "required_cell_count",
    "completed_required_cell_count",
    "skipped_optional_cell_count",
    "failed_required_cell_count",
    "retrained_cell_count",
    "verification_scope",
    "outcome_values_read",
}
_SOURCE_FIELDS = {
    "schema_version",
    "policy",
    "manifest_path",
    "manifest_sha256",
    "root_sha256",
    "record_count",
}
_CAPSULE_FIELDS = {
    "schema_version",
    "policy",
    "path",
    "size_bytes",
    "sha256",
    "internal_manifest_sha256",
    "source_records_root_sha256",
    "publication_receipt_path",
    "publication_receipt_sha256",
    "independent_readback_path",
    "independent_readback_sha256",
    "content_addressed_create_new_verified",
    "scientific_execution_performed",
}
_CAPACITY_FIELDS = {
    "schema_version",
    "policy",
    "policy_sha256",
    "receipt_path",
    "receipt_sha256",
    "required_free_bytes",
    "observed_free_bytes",
    "passed",
    "capsule_sha256",
    "execution_source_root_sha256",
    "outcome_values_read",
    "scientific_execution_performed",
}
_OUTCOME_FIELDS = {
    "schema_version",
    "primary_outcomes_inspected",
    "primary_outcomes_inspected_at_utc",
    "primary_analysis_disposition",
    "confirmatory_outcomes_inspected",
    "confirmatory_outcome_values_read",
    "confirmatory_registration_status",
    "selection_performed",
    "tuning_performed",
    "scientific_execution_performed",
    "automatic_retry_allowed",
}
_PROCESS_FIELDS = {
    "process_id",
    "process_created_at_utc",
    "executable_path",
    "executable_size_bytes",
    "executable_sha256",
    "implementation_path",
    "implementation_sha256",
}
_EVIDENCE_FIELDS = {
    "schema_version",
    "policy",
    "authority_kind",
    "authority_directory",
    "publication_timestamp_utc",
    "parent",
    "frozen_science",
    "historical_primary",
    "execution_source",
    "execution_capsule",
    "capacity_v2",
    "outcome_scope",
    "intent_root_sha256",
    "intent_file_sha256",
    "independent_review_root_sha256",
    "independent_review_receipt_sha256",
    "scientific_definition_changed",
    "automatic_retry_allowed",
    "technical_authorization_sha256",
}
_ATTEMPT_FIELDS = {
    "schema_version",
    "policy",
    "authority_directory",
    "publication_timestamp_utc",
    "intent_root_sha256",
    "independent_review_root_sha256",
    "artifact_root_sha256",
    "sha256_manifest_sha256",
    "creation_disposition",
    "attempt_count",
    "max_attempt_count",
    "automatic_retry_allowed",
    "adoption_allowed",
    "cleanup_allowed",
    "overwrite_allowed",
    "attempt_root_sha256",
}
_SUCCESS_FIELDS = {
    "schema_version",
    "policy",
    "authority_directory",
    "publication_timestamp_utc",
    "publication_attempt_sha256",
    "immutable_marker_sha256",
    "artifact_root_sha256",
    "sha256_manifest_sha256",
    "technical_authorization_sha256",
    "terminal_disposition",
    "attempt_count",
    "automatic_retry_allowed",
    "adoption_allowed",
    "cleanup_allowed",
    "overwrite_allowed",
    "success_root_sha256",
}


def _parent(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_PARENT_FIELDS, role="technical authority parent")
    directory = _absolute_path(raw["authority_directory"], role="parent authority")
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["authority_kind"] != "preregistration_amendment"
        or not directory.as_posix().casefold().endswith(PARENT_RELATIVE_DIRECTORY.casefold())
        or raw["chain_depth"] != PARENT_CHAIN_DEPTH
        or type(raw["chain_depth"]) is not int
        or raw["artifact_root_sha256"] != PARENT_ARTIFACT_ROOT_SHA256
        or raw["sha256_manifest_sha256"] != PARENT_MANIFEST_SHA256
        or raw["execution_source_root_sha256"] != PARENT_SOURCE_ROOT_SHA256
        or raw["execution_source_manifest_sha256"] != PARENT_SOURCE_MANIFEST_SHA256
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority parent is not exact P"
        )
    return {**raw, "authority_directory": str(directory)}


def _frozen(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_FROZEN_FIELDS, role="frozen science binding")
    paths = {
        "preregistration_path": _absolute_path(
            raw["preregistration_path"], role="frozen preregistration"
        ),
        "primary_config_path": _absolute_path(
            raw["primary_config_path"], role="frozen primary config"
        ),
        "confirmatory_config_path": _absolute_path(
            raw["confirmatory_config_path"], role="frozen confirmatory config"
        ),
    }
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["preregistration_sha256"] != PREREGISTRATION_SHA256
        or raw["primary_config_sha256"] != PRIMARY_CONFIG_SHA256
        or raw["primary_config_semantic_sha256"] != PRIMARY_CONFIG_SEMANTIC_SHA256
        or raw["confirmatory_config_sha256"] != CONFIRMATORY_CONFIG_SHA256
        or raw["confirmatory_config_semantic_sha256"] != CONFIRMATORY_CONFIG_SEMANTIC_SHA256
        or raw["scientific_definition_changed"] is not False
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError("frozen scientific inputs differ from P")
    return {**raw, **{key: str(item) for key, item in paths.items()}}


def _historical_primary(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_PRIMARY_FIELDS, role="historical primary binding")
    directory = _absolute_path(raw["run_directory"], role="historical primary run")
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["run_id"] != HISTORICAL_PRIMARY_RUN_ID
        or directory.name != HISTORICAL_PRIMARY_RUN_ID
        or raw["terminal_status"] != "completed"
        or raw["completion_stage"] != "PRIMARY_STUDY_COMPLETE"
        or raw["artifact_root_sha256"] != HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256
        or raw["artifact_manifest_sha256"] != HISTORICAL_PRIMARY_MANIFEST_SHA256
        or raw["required_cell_count"] != 185
        or raw["completed_required_cell_count"] != 185
        or raw["skipped_optional_cell_count"] != 37
        or raw["failed_required_cell_count"] != 0
        or raw["retrained_cell_count"] != 0
        or any(
            type(raw[field]) is not int
            for field in (
                "required_cell_count",
                "completed_required_cell_count",
                "skipped_optional_cell_count",
                "failed_required_cell_count",
                "retrained_cell_count",
            )
        )
        or raw["verification_scope"]
        != "integrity_and_control_metadata_only_no_scientific_outcome_values"
        or raw["outcome_values_read"] is not False
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "historical primary binding is not the exact sealed 185/37 recovery"
        )
    return {**raw, "run_directory": str(directory)}


def _source(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_SOURCE_FIELDS, role="execution source binding")
    path = _absolute_path(raw["manifest_path"], role="execution source manifest")
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["policy"] != "runtracker_capture_source_tree_exact_object_v1"
        or type(raw["record_count"]) is not int
        or raw["record_count"] < 1
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError("execution source binding is malformed")
    _sha256(raw["manifest_sha256"], role="execution source manifest")
    _sha256(raw["root_sha256"], role="execution source root")
    return {**raw, "manifest_path": str(path)}


def _capsule(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_CAPSULE_FIELDS, role="execution capsule binding")
    path = _absolute_path(raw["path"], role="published execution capsule")
    publication = _absolute_path(
        raw["publication_receipt_path"], role="capsule publication receipt"
    )
    readback = _absolute_path(raw["independent_readback_path"], role="capsule independent readback")
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["policy"] != "content_addressed_original_confirmatory_execution_capsule_v1"
        or type(raw["size_bytes"]) is not int
        or raw["size_bytes"] < 1
        or path.name != "original_confirmatory.pyz"
        or path.parent.name != raw["sha256"]
        or raw["content_addressed_create_new_verified"] is not True
        or raw["scientific_execution_performed"] is not False
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "execution capsule binding is malformed or not content-addressed"
        )
    for field in (
        "sha256",
        "internal_manifest_sha256",
        "source_records_root_sha256",
        "publication_receipt_sha256",
        "independent_readback_sha256",
    ):
        _sha256(raw[field], role=f"execution capsule {field}")
    return {
        **raw,
        "path": str(path),
        "publication_receipt_path": str(publication),
        "independent_readback_path": str(readback),
    }


def _capacity(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_CAPACITY_FIELDS, role="capacity-v2 binding")
    path = _absolute_path(raw["receipt_path"], role="capacity-v2 receipt")
    if (
        raw["schema_version"] != CAPACITY_SCHEMA_VERSION
        or type(raw["schema_version"]) is not int
        or raw["policy"] != CAPACITY_POLICY_NAME
        or raw["policy_sha256"] != CAPACITY_POLICY_SHA256
        or raw["required_free_bytes"] != CAPACITY_REQUIRED_FREE_BYTES
        or type(raw["required_free_bytes"]) is not int
        or type(raw["observed_free_bytes"]) is not int
        or raw["observed_free_bytes"] < CAPACITY_REQUIRED_FREE_BYTES
        or raw["passed"] is not True
        or raw["outcome_values_read"] is not False
        or raw["scientific_execution_performed"] is not False
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capacity-v2 binding does not prove the exact 70-GiB gate"
        )
    for field in ("receipt_sha256", "capsule_sha256", "execution_source_root_sha256"):
        _sha256(raw[field], role=f"capacity-v2 {field}")
    return {**raw, "receipt_path": str(path)}


def _outcome_scope(value: Any) -> dict[str, Any]:
    raw = _mapping(value, fields=_OUTCOME_FIELDS, role="outcome-split scope")
    _utc(raw["primary_outcomes_inspected_at_utc"], role="primary inspection timestamp")
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["primary_outcomes_inspected"] is not True
        or raw["primary_outcomes_inspected_at_utc"] != PRIMARY_OUTCOME_INSPECTION_AT_UTC
        or raw["primary_analysis_disposition"] != "amended_or_exploratory"
        or raw["confirmatory_outcomes_inspected"] is not False
        or raw["confirmatory_outcome_values_read"] is not False
        or raw["confirmatory_registration_status"] != "original_frozen_confirmatory_unchanged"
        or raw["selection_performed"] is not False
        or raw["tuning_performed"] is not False
        or raw["scientific_execution_performed"] is not False
        or raw["automatic_retry_allowed"] is not False
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "outcome scope is not exact primary-inspected/confirmatory-uninspected"
        )
    return raw


def _process(value: Any, *, role: str) -> dict[str, Any]:
    raw = _mapping(value, fields=_PROCESS_FIELDS, role=role)
    executable = _absolute_path(raw["executable_path"], role=f"{role} executable")
    implementation = _absolute_path(raw["implementation_path"], role=f"{role} implementation")
    _utc(raw["process_created_at_utc"], role=f"{role} process creation")
    if (
        type(raw["process_id"]) is not int
        or raw["process_id"] <= 0
        or type(raw["executable_size_bytes"]) is not int
        or raw["executable_size_bytes"] < 1
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} is malformed")
    _sha256(raw["executable_sha256"], role=f"{role} executable")
    _sha256(raw["implementation_sha256"], role=f"{role} implementation")
    return {
        **raw,
        "executable_path": str(executable),
        "implementation_path": str(implementation),
    }


_INTENT_FIELDS = {
    "schema_version",
    "policy",
    "created_at_utc",
    "builder_process",
    "parent",
    "frozen_science",
    "historical_primary",
    "execution_source",
    "execution_capsule",
    "capacity_v2",
    "outcome_scope",
    "forbidden_downstream_fields",
    "downstream_bindings_included",
    "automatic_retry_allowed",
    "intent_root_sha256",
}


def canonical_original_confirmatory_technical_authority_intent_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _mapping(value, fields=_INTENT_FIELDS, role="technical authority intent")
    rooted = _rooted(raw, "intent_root_sha256", role="technical authority intent")
    created = _utc(rooted["created_at_utc"], role="intent creation timestamp")
    builder = _process(rooted["builder_process"], role="intent builder")
    canonical = {
        "schema_version": 1,
        "policy": INTENT_POLICY,
        "created_at_utc": rooted["created_at_utc"],
        "builder_process": builder,
        "parent": _parent(rooted["parent"]),
        "frozen_science": _frozen(rooted["frozen_science"]),
        "historical_primary": _historical_primary(rooted["historical_primary"]),
        "execution_source": _source(rooted["execution_source"]),
        "execution_capsule": _capsule(rooted["execution_capsule"]),
        "capacity_v2": _capacity(rooted["capacity_v2"]),
        "outcome_scope": _outcome_scope(rooted["outcome_scope"]),
        "forbidden_downstream_fields": list(FORBIDDEN_DOWNSTREAM_FIELDS),
        "downstream_bindings_included": False,
        "automatic_retry_allowed": False,
    }
    if (
        rooted["schema_version"] != 1
        or type(rooted["schema_version"]) is not int
        or rooted["policy"] != INTENT_POLICY
        or rooted["forbidden_downstream_fields"] != list(FORBIDDEN_DOWNSTREAM_FIELDS)
        or rooted["downstream_bindings_included"] is not False
        or rooted["automatic_retry_allowed"] is not False
        or _utc(
            builder["process_created_at_utc"],
            role="intent builder process creation",
        )
        > created
        or canonical["capacity_v2"]["capsule_sha256"] != canonical["execution_capsule"]["sha256"]
        or canonical["capacity_v2"]["execution_source_root_sha256"]
        != canonical["execution_source"]["root_sha256"]
        or canonical_json_sha256(canonical) != rooted["intent_root_sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority intent violates its exact acyclic policy"
        )
    return {**canonical, "intent_root_sha256": rooted["intent_root_sha256"]}


def build_original_confirmatory_technical_authority_intent_v1(
    *,
    created_at_utc: str,
    builder_process: Mapping[str, Any],
    parent: Mapping[str, Any],
    frozen_science: Mapping[str, Any],
    historical_primary: Mapping[str, Any],
    execution_source: Mapping[str, Any],
    execution_capsule: Mapping[str, Any],
    capacity_v2: Mapping[str, Any],
    outcome_scope: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "policy": INTENT_POLICY,
        "created_at_utc": created_at_utc,
        "builder_process": dict(builder_process),
        "parent": dict(parent),
        "frozen_science": dict(frozen_science),
        "historical_primary": dict(historical_primary),
        "execution_source": dict(execution_source),
        "execution_capsule": dict(execution_capsule),
        "capacity_v2": dict(capacity_v2),
        "outcome_scope": dict(outcome_scope),
        "forbidden_downstream_fields": list(FORBIDDEN_DOWNSTREAM_FIELDS),
        "downstream_bindings_included": False,
        "automatic_retry_allowed": False,
    }
    return canonical_original_confirmatory_technical_authority_intent_v1(
        {**unsigned, "intent_root_sha256": canonical_json_sha256(unsigned)}
    )


_REVIEW_FIELDS = {
    "schema_version",
    "policy",
    "intent_root_sha256",
    "intent_file_sha256",
    "review_started_at_utc",
    "review_completed_at_utc",
    "reviewer_process",
    "reviewer_independent",
    "outcome_values_read",
    "scientific_execution_performed",
    "publication_performed",
    "selection_performed",
    "tuning_performed",
    "decision",
    "review_root_sha256",
}


def canonical_original_confirmatory_technical_authority_review_v1(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_intent = canonical_original_confirmatory_technical_authority_intent_v1(intent)
    raw = _mapping(value, fields=_REVIEW_FIELDS, role="independent review")
    rooted = _rooted(raw, "review_root_sha256", role="independent review")
    started = _utc(rooted["review_started_at_utc"], role="review start")
    completed = _utc(rooted["review_completed_at_utc"], role="review completion")
    created = _utc(canonical_intent["created_at_utc"], role="intent creation")
    reviewer = _process(rooted["reviewer_process"], role="independent reviewer")
    builder = canonical_intent["builder_process"]
    canonical = {
        "schema_version": 1,
        "policy": REVIEW_POLICY,
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "intent_file_sha256": _sha256(rooted["intent_file_sha256"], role="reviewed intent file"),
        "review_started_at_utc": rooted["review_started_at_utc"],
        "review_completed_at_utc": rooted["review_completed_at_utc"],
        "reviewer_process": reviewer,
        "reviewer_independent": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
        "selection_performed": False,
        "tuning_performed": False,
        "decision": "passed",
    }
    if (
        rooted["schema_version"] != 1
        or type(rooted["schema_version"]) is not int
        or rooted["policy"] != REVIEW_POLICY
        or rooted["intent_root_sha256"] != canonical_intent["intent_root_sha256"]
        or rooted["intent_file_sha256"]
        != _sha256_bytes(canonical_json_line_bytes(canonical_intent))
        or started < created
        or completed < started
        or _utc(
            reviewer["process_created_at_utc"],
            role="independent reviewer process creation",
        )
        > started
        or (
            reviewer["process_id"],
            reviewer["process_created_at_utc"],
        )
        == (
            builder["process_id"],
            builder["process_created_at_utc"],
        )
        or reviewer["implementation_path"] == builder["implementation_path"]
        or reviewer["implementation_sha256"] == builder["implementation_sha256"]
        or rooted["reviewer_independent"] is not True
        or rooted["outcome_values_read"] is not False
        or rooted["scientific_execution_performed"] is not False
        or rooted["publication_performed"] is not False
        or rooted["selection_performed"] is not False
        or rooted["tuning_performed"] is not False
        or rooted["decision"] != "passed"
        or canonical_json_sha256(canonical) != rooted["review_root_sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "independent review is stale, non-independent, or not outcome-blind"
        )
    return {**canonical, "review_root_sha256": rooted["review_root_sha256"]}


def build_original_confirmatory_technical_authority_review_v1(
    *,
    intent: Mapping[str, Any],
    review_started_at_utc: str,
    review_completed_at_utc: str,
    reviewer_process: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_intent = canonical_original_confirmatory_technical_authority_intent_v1(intent)
    unsigned = {
        "schema_version": 1,
        "policy": REVIEW_POLICY,
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "intent_file_sha256": _sha256_bytes(canonical_json_line_bytes(canonical_intent)),
        "review_started_at_utc": review_started_at_utc,
        "review_completed_at_utc": review_completed_at_utc,
        "reviewer_process": dict(reviewer_process),
        "reviewer_independent": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
        "selection_performed": False,
        "tuning_performed": False,
        "decision": "passed",
    }
    return canonical_original_confirmatory_technical_authority_review_v1(
        {**unsigned, "review_root_sha256": canonical_json_sha256(unsigned)},
        intent=canonical_intent,
    )


def _artifact_records(artifacts: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for name, payload in sorted(artifacts.items())
    ]


def build_original_confirmatory_technical_authority_bundle_v1(
    *,
    authority_directory: str | Path,
    intent: Mapping[str, Any],
    independent_review: Mapping[str, Any],
    publication_timestamp_utc: str,
    preregistration_bytes: bytes,
    primary_config_bytes: bytes,
    confirmatory_config_bytes: bytes,
    source_inventory: Mapping[str, Any],
) -> OriginalConfirmatoryTechnicalAuthorityBundle:
    canonical_intent = canonical_original_confirmatory_technical_authority_intent_v1(intent)
    review = canonical_original_confirmatory_technical_authority_review_v1(
        independent_review,
        intent=canonical_intent,
    )
    directory = Path(authority_directory)
    if not directory.is_absolute() or directory != Path(os.path.abspath(directory)):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "authority directory must be canonical and absolute"
        )
    published = _utc(publication_timestamp_utc, role="technical authority publication timestamp")
    reviewed = _utc(review["review_completed_at_utc"], role="review completion")
    if published <= reviewed:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "publication timestamp must be chosen after review completion"
        )
    snapshots = {
        PREREGISTRATION_FILENAME: preregistration_bytes,
        PRIMARY_CONFIG_FILENAME: primary_config_bytes,
        CONFIRMATORY_CONFIG_FILENAME: confirmatory_config_bytes,
    }
    expected_snapshot_hashes = {
        PREREGISTRATION_FILENAME: PREREGISTRATION_SHA256,
        PRIMARY_CONFIG_FILENAME: PRIMARY_CONFIG_SHA256,
        CONFIRMATORY_CONFIG_FILENAME: CONFIRMATORY_CONFIG_SHA256,
    }
    if any(
        type(payload) is not bytes
        or not payload
        or _sha256_bytes(payload) != expected_snapshot_hashes[name]
        for name, payload in snapshots.items()
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError("frozen snapshot bytes differ from P")
    source_bytes = canonical_json_line_bytes(source_inventory)
    source_binding = canonical_intent["execution_source"]
    if (
        source_inventory.get("root_sha256") != source_binding["root_sha256"]
        or not isinstance(source_inventory.get("artifacts"), list)
        or len(source_inventory["artifacts"]) != source_binding["record_count"]
        or _sha256_bytes(source_bytes) != source_binding["manifest_sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "source inventory differs from the RunTracker-compatible binding"
        )
    evidence_unsigned = {
        "schema_version": 1,
        "policy": EVIDENCE_POLICY,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "authority_directory": str(directory),
        "publication_timestamp_utc": publication_timestamp_utc,
        "parent": canonical_intent["parent"],
        "frozen_science": canonical_intent["frozen_science"],
        "historical_primary": canonical_intent["historical_primary"],
        "execution_source": source_binding,
        "execution_capsule": canonical_intent["execution_capsule"],
        "capacity_v2": canonical_intent["capacity_v2"],
        "outcome_scope": canonical_intent["outcome_scope"],
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "intent_file_sha256": _sha256_bytes(canonical_json_line_bytes(canonical_intent)),
        "independent_review_root_sha256": review["review_root_sha256"],
        "independent_review_receipt_sha256": _sha256_bytes(canonical_json_line_bytes(review)),
        "scientific_definition_changed": False,
        "automatic_retry_allowed": False,
    }
    evidence = {
        **evidence_unsigned,
        "technical_authorization_sha256": canonical_json_sha256(evidence_unsigned),
    }
    artifacts: dict[str, bytes] = {
        INTENT_FILENAME: canonical_json_line_bytes(canonical_intent),
        REVIEW_FILENAME: canonical_json_line_bytes(review),
        EVIDENCE_FILENAME: canonical_json_line_bytes(evidence),
        **snapshots,
        SOURCE_INVENTORY_FILENAME: source_bytes,
        CAPSULE_BINDING_FILENAME: canonical_json_line_bytes(canonical_intent["execution_capsule"]),
        CAPACITY_BINDING_FILENAME: canonical_json_line_bytes(canonical_intent["capacity_v2"]),
    }
    records = _artifact_records(artifacts)
    artifact_root = canonical_json_sha256(records)
    manifest = {
        "schema_version": 1,
        "policy": MANIFEST_POLICY,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "publication_timestamp_utc": publication_timestamp_utc,
        "artifact_count": len(records),
        "artifacts": records,
        "artifact_root_sha256": artifact_root,
        "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
        "excluded_paths": [
            IMMUTABLE_MARKER_FILENAME,
            MANIFEST_FILENAME,
            ATTEMPT_FILENAME,
            SUCCESS_FILENAME,
            STOP_FILENAME,
        ],
    }
    manifest_bytes = canonical_json_line_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    marker = {
        "schema_version": 1,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "status": "amended",
        "amendment_only": True,
        "publication_timestamp_utc": publication_timestamp_utc,
        "chain_depth": PARENT_CHAIN_DEPTH + 1,
        "parent_artifact_root_sha256": PARENT_ARTIFACT_ROOT_SHA256,
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "technical_authorization_sha256": evidence["technical_authorization_sha256"],
        "automatic_retry_allowed": False,
    }
    marker_bytes = canonical_json_line_bytes(marker)
    attempt_unsigned = {
        "schema_version": 1,
        "policy": ATTEMPT_POLICY,
        "authority_directory": str(directory),
        "publication_timestamp_utc": publication_timestamp_utc,
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "independent_review_root_sha256": review["review_root_sha256"],
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "creation_disposition": "CREATE_NEW",
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    attempt = {
        **attempt_unsigned,
        "attempt_root_sha256": canonical_json_sha256(attempt_unsigned),
    }
    attempt_bytes = canonical_json_line_bytes(attempt)
    success_unsigned = {
        "schema_version": 1,
        "policy": SUCCESS_POLICY,
        "authority_directory": str(directory),
        "publication_timestamp_utc": publication_timestamp_utc,
        "publication_attempt_sha256": _sha256_bytes(attempt_bytes),
        "immutable_marker_sha256": _sha256_bytes(marker_bytes),
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "technical_authorization_sha256": evidence["technical_authorization_sha256"],
        "terminal_disposition": "success",
        "attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    success = {
        **success_unsigned,
        "success_root_sha256": canonical_json_sha256(success_unsigned),
    }
    stop_unsigned = {
        "schema_version": 1,
        "policy": STOP_POLICY,
        "authority_directory": str(directory),
        "publication_timestamp_utc": publication_timestamp_utc,
        "publication_attempt_sha256": _sha256_bytes(attempt_bytes),
        "intent_root_sha256": canonical_intent["intent_root_sha256"],
        "terminal_disposition": "permanent_stop_no_retry_no_adoption_no_cleanup",
        "attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    stop = {
        **stop_unsigned,
        "stop_root_sha256": canonical_json_sha256(stop_unsigned),
    }
    return OriginalConfirmatoryTechnicalAuthorityBundle(
        authority_directory=directory,
        artifacts=artifacts,
        sha256_manifest_bytes=manifest_bytes,
        immutable_marker_bytes=marker_bytes,
        publication_attempt_bytes=attempt_bytes,
        publication_success_bytes=canonical_json_line_bytes(success),
        publication_stop_bytes=canonical_json_line_bytes(stop),
        artifact_root_sha256=artifact_root,
        sha256_manifest_sha256=manifest_sha,
        technical_authorization_sha256=evidence["technical_authorization_sha256"],
    )


def _strict_json_line_bytes(payload: bytes, *, role: str) -> dict[str, Any]:
    if not payload or not payload.endswith(b"\n") or b"\n" in payload[:-1]:
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} is not one canonical JSON line")
    duplicates: set[str] = set()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicates.add(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} is not strict JSON") from exc
    if duplicates or not isinstance(value, dict) or canonical_json_line_bytes(value) != payload:
        raise OriginalConfirmatoryTechnicalAuthorityError(f"{role} differs from canonical JSON")
    return value


def _strict_json_line(path: Path, *, role: str) -> dict[str, Any]:
    return _strict_json_line_bytes(path.read_bytes(), role=role)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file_set(directory: Path) -> None:
    actual: set[str] = set()
    for path in directory.iterdir():
        value = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"authority contains a non-regular or linked path: {path.name}"
            )
        actual.add(path.name)
    if actual != QUALIFYING_FILENAMES or (directory / STOP_FILENAME).exists():
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "authority inventory is not exact terminal success"
        )


def _directory_snapshot(directory: Path) -> tuple[tuple[str, int, str], ...]:
    _require_file_set(directory)
    return tuple(
        (
            name,
            (directory / name).stat(follow_symlinks=False).st_size,
            _file_sha256(directory / name),
        )
        for name in sorted(QUALIFYING_FILENAMES)
    )


_CAPSULE_PUBLICATION_RECEIPT_FIELDS = {
    "schema_version",
    "policy",
    "published_at_utc",
    "publisher_process",
    "capsule_path",
    "capsule_size_bytes",
    "capsule_sha256",
    "internal_manifest_sha256",
    "source_records_root_sha256",
    "creation_disposition",
    "same_handle_readback_verified",
    "archive_integrity_verified",
    "outcome_values_read",
    "scientific_execution_performed",
    "automatic_retry_allowed",
    "receipt_root_sha256",
}
_CAPSULE_READBACK_FIELDS = {
    "schema_version",
    "policy",
    "verified_at_utc",
    "reviewer_process",
    "publication_receipt_sha256",
    "capsule_path",
    "capsule_size_bytes",
    "capsule_sha256",
    "internal_manifest_sha256",
    "source_records_root_sha256",
    "byte_readback_verified",
    "archive_integrity_verified",
    "outcome_values_read",
    "scientific_execution_performed",
    "automatic_retry_allowed",
    "readback_root_sha256",
}
_CAPACITY_RECEIPT_FIELDS = {
    "schema_version",
    "policy",
    "policy_sha256",
    "checked_at_utc",
    "phase",
    "planned_cell_count",
    "planned_required_cell_count",
    "planned_optional_cell_count",
    "planned_cnn_cell_count",
    "planned_cnn_fold_checkpoint_count",
    "checkpoint_physical_copy_count",
    "projected_checkpoint_bytes_per_physical_copy",
    "projected_all_checkpoint_copies_bytes",
    "safety_margin_bytes",
    "required_free_bytes",
    "observed_free_bytes",
    "passed",
    "capsule_sha256",
    "execution_source_root_sha256",
    "outcome_values_read",
    "scientific_execution_performed",
    "adaptive_execution_changes_allowed",
    "capacity_receipt_root_sha256",
}


def _plain_file_matches(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
) -> bool:
    try:
        value = path.stat(follow_symlinks=False)
        return (
            path.is_absolute()
            and path == Path(os.path.abspath(path))
            and path == path.resolve()
            and stat.S_ISREG(value.st_mode)
            and not path.is_symlink()
            and int(getattr(value, "st_file_attributes", 0)) & 0x400 == 0
            and value.st_nlink == 1
            and (expected_size_bytes is None or value.st_size == expected_size_bytes)
            and _file_sha256(path) == expected_sha256
        )
    except OSError:
        return False


def _verify_process_files(value: Mapping[str, Any], *, role: str) -> None:
    process = _process(value, role=role)
    executable = Path(process["executable_path"])
    implementation = Path(process["implementation_path"])
    if not _plain_file_matches(
        executable,
        expected_sha256=process["executable_sha256"],
        expected_size_bytes=process["executable_size_bytes"],
    ) or not _plain_file_matches(
        implementation,
        expected_sha256=process["implementation_sha256"],
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            f"{role} executable or implementation changed"
        )


def _capsule_records_root(records: list[dict[str, Any]]) -> str:
    preimage = b"".join(
        (
            f"{record['relative_path']}\0{record['role']}\0"
            f"{record['size_bytes']}\0{record['sha256']}\n"
        ).encode("ascii")
        for record in records
    )
    return hashlib.sha256(preimage).hexdigest()


def _verify_capsule_archive(
    capsule_path: Path,
    *,
    capsule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = capsule_path.read_bytes()
    if len(payload) != capsule["size_bytes"] or _sha256_bytes(payload) != capsule["sha256"]:
        raise OriginalConfirmatoryTechnicalAuthorityError("capsule bytes differ from their binding")
    try:
        with zipfile.ZipFile(
            io.BytesIO(payload),
            mode="r",
            allowZip64=False,
        ) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                archive.comment != b""
                or not infos
                or names[-1] != "AANCA_CAPSULE_MANIFEST.json"
                or len(names) != len(set(names))
                or archive.testzip() is not None
            ):
                raise OriginalConfirmatoryTechnicalAuthorityError(
                    "capsule ZIP inventory or CRC is invalid"
                )
            manifest_bytes = archive.read(infos[-1])
            manifest = _strict_json_line_bytes(
                manifest_bytes,
                role="capsule internal manifest",
            )
            entries = manifest.get("entries")
            if (
                set(manifest)
                != {
                    "schema_version",
                    "policy",
                    "archive_policy",
                    "entries",
                    "entry_count",
                    "payload_size_bytes",
                    "records_root_sha256",
                }
                or type(manifest.get("schema_version")) is not int
                or manifest.get("schema_version") != 1
                or manifest.get("policy") != "original_confirmatory_execution_capsule_manifest_v1"
                or manifest.get("archive_policy")
                != {
                    "format": "zip",
                    "compression": "stored",
                    "payload_entry_order": "ordinal_then_manifest_last",
                    "fixed_dos_datetime": "1980-01-01T00:00:00",
                    "create_system": 3,
                    "unix_mode": "100444",
                    "zip64": False,
                    "archive_comment_empty": True,
                    "directory_entries": False,
                    "manifest_self_entry": False,
                }
                or not isinstance(entries, list)
                or type(manifest.get("entry_count")) is not int
                or manifest.get("entry_count") != len(entries)
                or len(infos) != len(entries) + 1
                or _sha256_bytes(manifest_bytes) != capsule["internal_manifest_sha256"]
            ):
                raise OriginalConfirmatoryTechnicalAuthorityError(
                    "capsule internal manifest is invalid"
                )
            records: list[dict[str, Any]] = []
            payload_size = 0
            for index, raw_record in enumerate(entries):
                record = _mapping(
                    raw_record,
                    fields={
                        "relative_path",
                        "role",
                        "size_bytes",
                        "sha256",
                    },
                    role="capsule manifest record",
                )
                relative = record["relative_path"]
                role = record["role"]
                size = record["size_bytes"]
                digest = record["sha256"]
                pure = PurePosixPath(relative) if type(relative) is str else None
                if (
                    pure is None
                    or pure.is_absolute()
                    or str(pure) != relative
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or "\\" in relative
                    or type(role) is not str
                    or not role
                    or type(size) is not int
                    or size < 0
                    or type(digest) is not str
                    or _SHA256.fullmatch(digest) is None
                    or names[index] != relative
                ):
                    raise OriginalConfirmatoryTechnicalAuthorityError(
                        "capsule manifest record is malformed"
                    )
                info = infos[index]
                member = archive.read(info)
                if (
                    info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.create_system != 3
                    or info.external_attr != (stat.S_IFREG | 0o444) << 16
                    or info.extra != b""
                    or info.comment != b""
                    or info.file_size != size
                    or info.compress_size != size
                    or len(member) != size
                    or _sha256_bytes(member) != digest
                ):
                    raise OriginalConfirmatoryTechnicalAuthorityError(
                        "capsule member differs from its fixed record"
                    )
                payload_size += size
                records.append(record)
            manifest_info = infos[-1]
            if (
                manifest_info.date_time != (1980, 1, 1, 0, 0, 0)
                or manifest_info.compress_type != zipfile.ZIP_STORED
                or manifest_info.create_system != 3
                or manifest_info.external_attr != (stat.S_IFREG | 0o444) << 16
                or manifest_info.extra != b""
                or manifest_info.comment != b""
                or manifest_info.file_size != len(manifest_bytes)
                or manifest_info.compress_size != len(manifest_bytes)
                or type(manifest.get("payload_size_bytes")) is not int
                or manifest.get("payload_size_bytes") != payload_size
                or manifest.get("records_root_sha256") != _capsule_records_root(records)
                or manifest.get("records_root_sha256") != capsule["source_records_root_sha256"]
            ):
                raise OriginalConfirmatoryTechnicalAuthorityError(
                    "capsule manifest totals or source root are invalid"
                )
            return records
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        if isinstance(exc, OriginalConfirmatoryTechnicalAuthorityError):
            raise
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule archive could not be verified"
        ) from exc


def _capsule_role(relative_path: str) -> str:
    special = {
        "histo_audit/experiment/confirmatory_completion.py": ("scientific_completion"),
        "histo_audit/experiment/original_confirmatory_runner_core.py": ("scientific_entry"),
        "histo_audit/workflows/original_confirmatory_capsule_authority.py": ("capsule_authority"),
        "histo_audit/workflows/original_confirmatory_capsule_entry.py": ("capsule_dispatcher"),
        "histo_audit/workflows/original_confirmatory_capsule_terminal.py": ("capsule_terminal"),
    }
    if relative_path in special:
        return special[relative_path]
    if relative_path == "histo_audit/__init__.py" or relative_path.endswith("/__init__.py"):
        return "package_initializer"
    return "project_source"


def _verify_capsule_source_alignment(
    *,
    records: list[dict[str, Any]],
    source_inventory: Mapping[str, Any],
    project_root: Path,
) -> None:
    raw_source_records = source_inventory.get("artifacts")
    if not isinstance(raw_source_records, list):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "execution source inventory lacks artifact records"
        )
    expected_package: list[dict[str, Any]] = []
    for raw_record in raw_source_records:
        record = _mapping(
            raw_record,
            fields={"path", "size_bytes", "sha256"},
            role="execution source artifact",
        )
        path = record["path"]
        if type(path) is str and path.startswith("src/histo_audit/") and path.endswith(".py"):
            relative = path.removeprefix("src/")
            expected_package.append(
                {
                    "relative_path": relative,
                    "role": _capsule_role(relative),
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }
            )
    actual_package = [
        record for record in records if record["relative_path"].startswith("histo_audit/")
    ]
    if canonical_json_bytes(actual_package) != canonical_json_bytes(
        sorted(
            expected_package,
            key=lambda item: item["relative_path"],
        )
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule Python payload differs from the authority-bound source tree"
        )
    extras = {
        record["relative_path"]: record
        for record in records
        if not record["relative_path"].startswith("histo_audit/")
    }
    expected_extras = {
        "__main__.py": (
            project_root / "capsule_bootstrap.py",
            "capsule_bootstrap",
        ),
        "aanca_capsule/capsule_policy.json": (
            project_root / "capsule_policy.json",
            "capsule_policy",
        ),
        "aanca_capsule/entry_contract.json": (
            project_root / "entry_contract.json",
            "capsule_contract",
        ),
    }
    if set(extras) != set(expected_extras):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule standalone payload set is not exact"
        )
    for relative, (path, role) in expected_extras.items():
        record = extras[relative]
        if (
            not path.is_file()
            or path.is_symlink()
            or record["role"] != role
            or record["size_bytes"] != path.stat(follow_symlinks=False).st_size
            or record["sha256"] != _file_sha256(path)
        ):
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"capsule standalone payload differs: {relative}"
            )


def _verify_capsule_receipts(
    capsule: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    publication_path = Path(capsule["publication_receipt_path"])
    readback_path = Path(capsule["independent_readback_path"])
    publication = _mapping(
        _strict_json_line(
            publication_path,
            role="capsule publication receipt",
        ),
        fields=_CAPSULE_PUBLICATION_RECEIPT_FIELDS,
        role="capsule publication receipt",
    )
    rooted_publication = _rooted(
        publication,
        "receipt_root_sha256",
        role="capsule publication receipt",
    )
    published = _utc(
        rooted_publication["published_at_utc"],
        role="capsule publication timestamp",
    )
    publisher = _process(
        rooted_publication["publisher_process"],
        role="capsule publisher",
    )
    expected_publication_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_execution_capsule_publication_receipt_v1",
        "published_at_utc": rooted_publication["published_at_utc"],
        "publisher_process": publisher,
        "capsule_path": capsule["path"],
        "capsule_size_bytes": capsule["size_bytes"],
        "capsule_sha256": capsule["sha256"],
        "internal_manifest_sha256": capsule["internal_manifest_sha256"],
        "source_records_root_sha256": capsule["source_records_root_sha256"],
        "creation_disposition": "CREATE_NEW",
        "same_handle_readback_verified": True,
        "archive_integrity_verified": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    expected_publication = {
        **expected_publication_unsigned,
        "receipt_root_sha256": canonical_json_sha256(expected_publication_unsigned),
    }
    if (
        canonical_json_line_bytes(rooted_publication)
        != canonical_json_line_bytes(expected_publication)
        or _utc(
            publisher["process_created_at_utc"],
            role="capsule publisher process creation",
        )
        > published
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule publication receipt is contradictory or stale"
        )
    _verify_process_files(
        publisher,
        role="capsule publisher",
    )
    readback = _mapping(
        _strict_json_line(
            readback_path,
            role="capsule independent readback",
        ),
        fields=_CAPSULE_READBACK_FIELDS,
        role="capsule independent readback",
    )
    rooted_readback = _rooted(
        readback,
        "readback_root_sha256",
        role="capsule independent readback",
    )
    verified = _utc(
        rooted_readback["verified_at_utc"],
        role="capsule independent readback timestamp",
    )
    reviewer = _process(
        rooted_readback["reviewer_process"],
        role="capsule independent readback process",
    )
    expected_readback_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_execution_capsule_independent_readback_v1",
        "verified_at_utc": rooted_readback["verified_at_utc"],
        "reviewer_process": reviewer,
        "publication_receipt_sha256": capsule["publication_receipt_sha256"],
        "capsule_path": capsule["path"],
        "capsule_size_bytes": capsule["size_bytes"],
        "capsule_sha256": capsule["sha256"],
        "internal_manifest_sha256": capsule["internal_manifest_sha256"],
        "source_records_root_sha256": capsule["source_records_root_sha256"],
        "byte_readback_verified": True,
        "archive_integrity_verified": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    expected_readback = {
        **expected_readback_unsigned,
        "readback_root_sha256": canonical_json_sha256(expected_readback_unsigned),
    }
    if (
        canonical_json_line_bytes(rooted_readback) != canonical_json_line_bytes(expected_readback)
        or verified <= published
        or _utc(
            reviewer["process_created_at_utc"],
            role="capsule readback process creation",
        )
        > verified
        or (
            reviewer["process_id"],
            reviewer["process_created_at_utc"],
        )
        == (
            publisher["process_id"],
            publisher["process_created_at_utc"],
        )
        or reviewer["implementation_path"] == publisher["implementation_path"]
        or reviewer["implementation_sha256"] == publisher["implementation_sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule independent readback is not exact and independent"
        )
    _verify_process_files(
        reviewer,
        role="capsule independent readback process",
    )
    return published, verified


def _verify_capacity_receipt(capacity: Mapping[str, Any]) -> datetime:
    receipt = _mapping(
        _strict_json_line(
            Path(capacity["receipt_path"]),
            role="capacity-v2 receipt",
        ),
        fields=_CAPACITY_RECEIPT_FIELDS,
        role="capacity-v2 receipt",
    )
    rooted = _rooted(
        receipt,
        "capacity_receipt_root_sha256",
        role="capacity-v2 receipt",
    )
    checked = _utc(rooted["checked_at_utc"], role="capacity-v2 timestamp")
    expected_unsigned = {
        "schema_version": 2,
        "policy": CAPACITY_POLICY_NAME,
        "policy_sha256": CAPACITY_POLICY_SHA256,
        "checked_at_utc": rooted["checked_at_utc"],
        "phase": "before_technical_authority",
        "planned_cell_count": 108,
        "planned_required_cell_count": 90,
        "planned_optional_cell_count": 18,
        "planned_cnn_cell_count": 36,
        "planned_cnn_fold_checkpoint_count": 180,
        "checkpoint_physical_copy_count": 2,
        "projected_checkpoint_bytes_per_physical_copy": 30 * 1024**3,
        "projected_all_checkpoint_copies_bytes": 60 * 1024**3,
        "safety_margin_bytes": 10 * 1024**3,
        "required_free_bytes": CAPACITY_REQUIRED_FREE_BYTES,
        "observed_free_bytes": capacity["observed_free_bytes"],
        "passed": True,
        "capsule_sha256": capacity["capsule_sha256"],
        "execution_source_root_sha256": capacity["execution_source_root_sha256"],
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "adaptive_execution_changes_allowed": False,
    }
    expected = {
        **expected_unsigned,
        "capacity_receipt_root_sha256": canonical_json_sha256(expected_unsigned),
    }
    if canonical_json_line_bytes(rooted) != canonical_json_line_bytes(expected):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capacity-v2 receipt does not prove the exact sealed arithmetic"
        )
    return checked


def _verify_live_bindings(
    *,
    project_root: Path,
    intent: Mapping[str, Any],
    review: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
) -> None:
    parent = intent["parent"]
    parent_path = Path(parent["authority_directory"])
    expected_parent = project_root / Path(PARENT_RELATIVE_DIRECTORY)
    if parent_path != expected_parent.resolve():
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority parent path is not exact live P"
        )
    from histo_audit.utils.run_tracking import (
        capture_source_tree,
        require_run_stage_eligibility_receipt,
    )

    from .preregistration_amendment import verify_preregistration_amendment

    parent_verification = verify_preregistration_amendment(parent_path)
    if (
        not parent_verification.valid
        or parent_verification.chain_depth != PARENT_CHAIN_DEPTH
        or parent_verification.artifact_root_sha256 != PARENT_ARTIFACT_ROOT_SHA256
        or parent_verification.sha256_manifest_sha256 != PARENT_MANIFEST_SHA256
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "live parent P failed recursive verification"
        )
    frozen = intent["frozen_science"]
    expected_frozen_paths = {
        "preregistration_path": (project_root / "PRE_REGISTRATION.md").resolve(),
        "primary_config_path": (project_root / "configs" / "primary_frozen.yaml").resolve(),
        "confirmatory_config_path": (
            project_root / "configs" / "confirmatory_frozen.yaml"
        ).resolve(),
    }
    for path_field, hash_field in (
        ("preregistration_path", "preregistration_sha256"),
        ("primary_config_path", "primary_config_sha256"),
        ("confirmatory_config_path", "confirmatory_config_sha256"),
    ):
        path = Path(frozen[path_field])
        if (
            path != expected_frozen_paths[path_field]
            or not path.is_file()
            or path.is_symlink()
            or _file_sha256(path) != frozen[hash_field]
        ):
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"live frozen input differs: {path_field}"
            )
    source_path = Path(intent["execution_source"]["manifest_path"])
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or _file_sha256(source_path) != intent["execution_source"]["manifest_sha256"]
        or canonical_json_bytes(
            _strict_json_line(
                source_path,
                role="external execution source inventory",
            )
        )
        != canonical_json_bytes(dict(source_inventory))
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "external execution source inventory differs from its T0 snapshot"
        )
    if canonical_json_bytes(capture_source_tree(project_root)) != canonical_json_bytes(
        dict(source_inventory)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "live execution source differs from T0 source inventory"
        )
    _verify_process_files(
        intent["builder_process"],
        role="technical authority builder",
    )
    _verify_process_files(
        review["reviewer_process"],
        role="technical authority independent reviewer",
    )
    primary = intent["historical_primary"]
    primary_path = Path(primary["run_directory"])
    expected_primary_path = (
        project_root / "artifacts" / "runs" / HISTORICAL_PRIMARY_RUN_ID
    ).resolve()
    if primary_path != expected_primary_path:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "historical primary path differs from the exact sealed run"
        )
    stage = require_run_stage_eligibility_receipt(primary_path)
    if (
        stage is None
        or not stage.valid
        or stage.completion_stage != "PRIMARY_STUDY_COMPLETE"
        or stage.run_id != HISTORICAL_PRIMARY_RUN_ID
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "historical primary lacks its positive typed stage receipt"
        )
    stage_record = stage.attestation_record()
    if (
        stage_record.get("artifact_root_sha256") != HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256
        or stage_record.get("artifact_manifest_sha256") != HISTORICAL_PRIMARY_MANIFEST_SHA256
        or stage_record.get("terminal_status") != "completed"
        or stage_record.get("scientific_stage_eligible") is not True
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "historical primary stage receipt differs from the exact sealed run"
        )
    capsule = intent["execution_capsule"]
    capsule_path = Path(capsule["path"])
    expected_capsule_path = (
        project_root
        / "artifacts"
        / "execution_capsules"
        / capsule["sha256"]
        / "original_confirmatory.pyz"
    ).resolve()
    if (
        capsule_path != expected_capsule_path
        or not capsule_path.is_file()
        or capsule_path.is_symlink()
        or capsule_path.stat(follow_symlinks=False).st_nlink != 1
        or capsule_path.stat().st_size != capsule["size_bytes"]
        or _file_sha256(capsule_path) != capsule["sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "published execution capsule differs from T0"
        )
    for path_field, hash_field in (
        ("publication_receipt_path", "publication_receipt_sha256"),
        ("independent_readback_path", "independent_readback_sha256"),
    ):
        path = Path(capsule[path_field])
        if not path.is_file() or path.is_symlink() or _file_sha256(path) != capsule[hash_field]:
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"capsule evidence differs: {path_field}"
            )
    capsule_records = _verify_capsule_archive(
        capsule_path,
        capsule=capsule,
    )
    _verify_capsule_source_alignment(
        records=capsule_records,
        source_inventory=source_inventory,
        project_root=project_root,
    )
    capsule_published, capsule_verified = _verify_capsule_receipts(capsule)
    capacity = intent["capacity_v2"]
    capacity_path = Path(capacity["receipt_path"])
    if (
        not capacity_path.is_file()
        or capacity_path.is_symlink()
        or _file_sha256(capacity_path) != capacity["receipt_sha256"]
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError("capacity-v2 receipt differs from T0")
    capacity_checked = _verify_capacity_receipt(capacity)
    intent_created = _utc(
        intent["created_at_utc"],
        role="technical authority intent creation",
    )
    if not (capsule_published < capsule_verified <= capacity_checked <= intent_created):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule, capacity, and technical-intent chronology is invalid"
        )
    if shutil.disk_usage(capsule_path.parent).free < CAPACITY_REQUIRED_FREE_BYTES:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "current capsule volume no longer satisfies capacity-v2"
        )

    # Re-read every live upstream authority after the expensive validation pass.
    # This second pass is intentionally outcome-blind and prevents a successful
    # return when an upstream input changes while T0 is being qualified.
    final_parent_verification = verify_preregistration_amendment(parent_path)
    if (
        not final_parent_verification.valid
        or final_parent_verification.chain_depth != parent_verification.chain_depth
        or final_parent_verification.artifact_root_sha256
        != parent_verification.artifact_root_sha256
        or final_parent_verification.sha256_manifest_sha256
        != parent_verification.sha256_manifest_sha256
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "live parent P changed during T0 verification"
        )
    for path_field, hash_field in (
        ("preregistration_path", "preregistration_sha256"),
        ("primary_config_path", "primary_config_sha256"),
        ("confirmatory_config_path", "confirmatory_config_sha256"),
    ):
        path = Path(frozen[path_field])
        if path != expected_frozen_paths[path_field] or not _plain_file_matches(
            path,
            expected_sha256=frozen[hash_field],
        ):
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"live frozen input changed during verification: {path_field}"
            )
    if not _plain_file_matches(
        source_path,
        expected_sha256=intent["execution_source"]["manifest_sha256"],
    ) or canonical_json_bytes(
        _strict_json_line(
            source_path,
            role="external execution source inventory final readback",
        )
    ) != canonical_json_bytes(dict(source_inventory)):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "external execution source inventory changed during T0 verification"
        )
    _verify_process_files(
        intent["builder_process"],
        role="technical authority builder final readback",
    )
    _verify_process_files(
        review["reviewer_process"],
        role="technical authority independent reviewer final readback",
    )
    final_stage = require_run_stage_eligibility_receipt(primary_path)
    if (
        final_stage is None
        or not final_stage.valid
        or canonical_json_bytes(final_stage.attestation_record())
        != canonical_json_bytes(stage_record)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "historical primary changed during T0 verification"
        )
    if not _plain_file_matches(
        capsule_path,
        expected_sha256=capsule["sha256"],
        expected_size_bytes=capsule["size_bytes"],
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "execution capsule changed during T0 verification"
        )
    for path_field, hash_field in (
        ("publication_receipt_path", "publication_receipt_sha256"),
        ("independent_readback_path", "independent_readback_sha256"),
    ):
        path = Path(capsule[path_field])
        if not _plain_file_matches(path, expected_sha256=capsule[hash_field]):
            raise OriginalConfirmatoryTechnicalAuthorityError(
                f"capsule evidence changed during verification: {path_field}"
            )
    final_capsule_records = _verify_capsule_archive(
        capsule_path,
        capsule=capsule,
    )
    _verify_capsule_source_alignment(
        records=final_capsule_records,
        source_inventory=source_inventory,
        project_root=project_root,
    )
    final_capsule_published, final_capsule_verified = _verify_capsule_receipts(capsule)
    if (
        canonical_json_bytes(final_capsule_records) != canonical_json_bytes(capsule_records)
        or final_capsule_published != capsule_published
        or final_capsule_verified != capsule_verified
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "execution capsule readback changed during T0 verification"
        )
    if not _plain_file_matches(
        capacity_path,
        expected_sha256=capacity["receipt_sha256"],
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capacity-v2 receipt changed during T0 verification"
        )
    if _verify_capacity_receipt(capacity) != capacity_checked:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capacity-v2 readback changed during T0 verification"
        )
    if canonical_json_bytes(capture_source_tree(project_root)) != canonical_json_bytes(
        dict(source_inventory)
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "live execution source changed during T0 verification"
        )
    if shutil.disk_usage(capsule_path.parent).free < CAPACITY_REQUIRED_FREE_BYTES:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "capsule volume lost capacity during T0 verification"
        )


def verify_original_confirmatory_technical_authority_v1(
    authority_directory: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_live: bool = True,
) -> VerifiedOriginalConfirmatoryTechnicalAuthority:
    """Verify a sealed T0 directory and optionally all live upstream bindings."""

    directory = Path(authority_directory)
    if (
        not directory.is_absolute()
        or directory != Path(os.path.abspath(directory))
        or directory != directory.resolve()
        or not directory.is_dir()
        or directory.is_symlink()
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority directory is not exact, canonical, plain, and absolute"
        )
    initial_snapshot = _directory_snapshot(directory)
    intent = canonical_original_confirmatory_technical_authority_intent_v1(
        _strict_json_line(directory / INTENT_FILENAME, role="technical intent")
    )
    review = canonical_original_confirmatory_technical_authority_review_v1(
        _strict_json_line(directory / REVIEW_FILENAME, role="independent review"),
        intent=intent,
    )
    evidence = _mapping(
        _strict_json_line(
            directory / EVIDENCE_FILENAME,
            role="technical authority evidence",
        ),
        fields=_EVIDENCE_FIELDS,
        role="technical authority evidence",
    )
    evidence_root = _sha256(
        evidence.get("technical_authorization_sha256"),
        role="technical authorization",
    )
    evidence_unsigned = dict(evidence)
    evidence_unsigned.pop("technical_authorization_sha256")
    expected_evidence_unsigned = {
        "schema_version": 1,
        "policy": EVIDENCE_POLICY,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "authority_directory": str(directory),
        "publication_timestamp_utc": evidence["publication_timestamp_utc"],
        "parent": intent["parent"],
        "frozen_science": intent["frozen_science"],
        "historical_primary": intent["historical_primary"],
        "execution_source": intent["execution_source"],
        "execution_capsule": intent["execution_capsule"],
        "capacity_v2": intent["capacity_v2"],
        "outcome_scope": intent["outcome_scope"],
        "intent_root_sha256": intent["intent_root_sha256"],
        "intent_file_sha256": _file_sha256(directory / INTENT_FILENAME),
        "independent_review_root_sha256": review["review_root_sha256"],
        "independent_review_receipt_sha256": _file_sha256(directory / REVIEW_FILENAME),
        "scientific_definition_changed": False,
        "automatic_retry_allowed": False,
    }
    expected_evidence = {
        **expected_evidence_unsigned,
        "technical_authorization_sha256": canonical_json_sha256(expected_evidence_unsigned),
    }
    if (
        canonical_json_line_bytes(evidence) != canonical_json_line_bytes(expected_evidence)
        or evidence.get("schema_version") != 1
        or evidence.get("policy") != EVIDENCE_POLICY
        or evidence.get("authority_kind") != ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND
        or evidence.get("authority_directory") != str(directory)
        or evidence.get("parent") != intent["parent"]
        or evidence.get("frozen_science") != intent["frozen_science"]
        or evidence.get("historical_primary") != intent["historical_primary"]
        or evidence.get("execution_source") != intent["execution_source"]
        or evidence.get("execution_capsule") != intent["execution_capsule"]
        or evidence.get("capacity_v2") != intent["capacity_v2"]
        or evidence.get("outcome_scope") != intent["outcome_scope"]
        or evidence.get("intent_root_sha256") != intent["intent_root_sha256"]
        or evidence.get("intent_file_sha256") != _file_sha256(directory / INTENT_FILENAME)
        or evidence.get("independent_review_root_sha256") != review["review_root_sha256"]
        or evidence.get("independent_review_receipt_sha256")
        != _file_sha256(directory / REVIEW_FILENAME)
        or evidence.get("scientific_definition_changed") is not False
        or evidence.get("automatic_retry_allowed") is not False
        or canonical_json_sha256(evidence_unsigned) != evidence_root
        or _utc(
            evidence.get("publication_timestamp_utc"),
            role="authority publication timestamp",
        )
        <= _utc(review["review_completed_at_utc"], role="review completion")
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority evidence differs from intent/review"
        )
    source_inventory = _strict_json_line(
        directory / SOURCE_INVENTORY_FILENAME, role="source inventory"
    )
    source = intent["execution_source"]
    if (
        _file_sha256(directory / SOURCE_INVENTORY_FILENAME) != source["manifest_sha256"]
        or source_inventory.get("root_sha256") != source["root_sha256"]
        or not isinstance(source_inventory.get("artifacts"), list)
        or len(source_inventory["artifacts"]) != source["record_count"]
        or _strict_json_line(directory / CAPSULE_BINDING_FILENAME, role="capsule binding")
        != intent["execution_capsule"]
        or _strict_json_line(directory / CAPACITY_BINDING_FILENAME, role="capacity binding")
        != intent["capacity_v2"]
        or _file_sha256(directory / PREREGISTRATION_FILENAME) != PREREGISTRATION_SHA256
        or _file_sha256(directory / PRIMARY_CONFIG_FILENAME) != PRIMARY_CONFIG_SHA256
        or _file_sha256(directory / CONFIRMATORY_CONFIG_FILENAME) != CONFIRMATORY_CONFIG_SHA256
    ):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority snapshots or source/capsule/capacity bindings differ"
        )
    records = _artifact_records({name: (directory / name).read_bytes() for name in CORE_FILENAMES})
    artifact_root = canonical_json_sha256(records)
    manifest = _strict_json_line(directory / MANIFEST_FILENAME, role="authority manifest")
    expected_manifest = {
        "schema_version": 1,
        "policy": MANIFEST_POLICY,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "publication_timestamp_utc": evidence["publication_timestamp_utc"],
        "artifact_count": len(records),
        "artifacts": records,
        "artifact_root_sha256": artifact_root,
        "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
        "excluded_paths": [
            IMMUTABLE_MARKER_FILENAME,
            MANIFEST_FILENAME,
            ATTEMPT_FILENAME,
            SUCCESS_FILENAME,
            STOP_FILENAME,
        ],
    }
    if canonical_json_line_bytes(manifest) != canonical_json_line_bytes(expected_manifest):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority manifest is stale or invalid"
        )
    manifest_sha = _file_sha256(directory / MANIFEST_FILENAME)
    marker = _strict_json_line(directory / IMMUTABLE_MARKER_FILENAME, role="immutable marker")
    expected_marker = {
        "schema_version": 1,
        "authority_kind": ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND,
        "status": "amended",
        "amendment_only": True,
        "publication_timestamp_utc": evidence["publication_timestamp_utc"],
        "chain_depth": PARENT_CHAIN_DEPTH + 1,
        "parent_artifact_root_sha256": PARENT_ARTIFACT_ROOT_SHA256,
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "technical_authorization_sha256": evidence_root,
        "automatic_retry_allowed": False,
    }
    if canonical_json_line_bytes(marker) != canonical_json_line_bytes(expected_marker):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority immutable marker is stale or invalid"
        )
    attempt = _mapping(
        _strict_json_line(
            directory / ATTEMPT_FILENAME,
            role="publication attempt",
        ),
        fields=_ATTEMPT_FIELDS,
        role="publication attempt",
    )
    rooted_attempt = _rooted(attempt, "attempt_root_sha256", role="publication attempt")
    expected_attempt_unsigned = {
        "schema_version": 1,
        "policy": ATTEMPT_POLICY,
        "authority_directory": str(directory),
        "publication_timestamp_utc": evidence["publication_timestamp_utc"],
        "intent_root_sha256": intent["intent_root_sha256"],
        "independent_review_root_sha256": review["review_root_sha256"],
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "creation_disposition": "CREATE_NEW",
        "attempt_count": 1,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    expected_attempt = {
        **expected_attempt_unsigned,
        "attempt_root_sha256": canonical_json_sha256(expected_attempt_unsigned),
    }
    if canonical_json_line_bytes(rooted_attempt) != canonical_json_line_bytes(expected_attempt):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "publication attempt is not the exact one-use claim"
        )
    success = _mapping(
        _strict_json_line(
            directory / SUCCESS_FILENAME,
            role="publication success",
        ),
        fields=_SUCCESS_FIELDS,
        role="publication success",
    )
    rooted_success = _rooted(success, "success_root_sha256", role="publication success")
    expected_success_unsigned = {
        "schema_version": 1,
        "policy": SUCCESS_POLICY,
        "authority_directory": str(directory),
        "publication_timestamp_utc": evidence["publication_timestamp_utc"],
        "publication_attempt_sha256": _file_sha256(directory / ATTEMPT_FILENAME),
        "immutable_marker_sha256": _file_sha256(directory / IMMUTABLE_MARKER_FILENAME),
        "artifact_root_sha256": artifact_root,
        "sha256_manifest_sha256": manifest_sha,
        "technical_authorization_sha256": evidence_root,
        "terminal_disposition": "success",
        "attempt_count": 1,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
        "overwrite_allowed": False,
    }
    expected_success = {
        **expected_success_unsigned,
        "success_root_sha256": canonical_json_sha256(expected_success_unsigned),
    }
    if canonical_json_line_bytes(rooted_success) != canonical_json_line_bytes(expected_success):
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "publication success is stale or ambiguous"
        )
    if verify_live:
        if project_root is None:
            raise OriginalConfirmatoryTechnicalAuthorityError(
                "live T0 verification requires project_root"
            )
        root = Path(project_root).resolve()
        _verify_live_bindings(
            project_root=root,
            intent=intent,
            review=review,
            source_inventory=source_inventory,
        )
    final_review_receipt_sha256 = _file_sha256(directory / REVIEW_FILENAME)
    final_immutable_marker_sha256 = _file_sha256(directory / IMMUTABLE_MARKER_FILENAME)
    final_publication_attempt_sha256 = _file_sha256(directory / ATTEMPT_FILENAME)
    final_publication_success_sha256 = _file_sha256(directory / SUCCESS_FILENAME)
    if _directory_snapshot(directory) != initial_snapshot:
        raise OriginalConfirmatoryTechnicalAuthorityError(
            "technical authority changed during verification"
        )
    return VerifiedOriginalConfirmatoryTechnicalAuthority(
        authority_directory=directory,
        chain_depth=PARENT_CHAIN_DEPTH + 1,
        artifact_root_sha256=artifact_root,
        sha256_manifest_sha256=manifest_sha,
        execution_source_manifest_sha256=source["manifest_sha256"],
        execution_source_root_sha256=source["root_sha256"],
        parent_authority_directory=Path(intent["parent"]["authority_directory"]),
        parent_artifact_root_sha256=PARENT_ARTIFACT_ROOT_SHA256,
        parent_sha256_manifest_sha256=PARENT_MANIFEST_SHA256,
        technical_authorization_sha256=evidence_root,
        independent_review_receipt_sha256=final_review_receipt_sha256,
        immutable_marker_sha256=final_immutable_marker_sha256,
        publication_attempt_sha256=final_publication_attempt_sha256,
        publication_success_sha256=final_publication_success_sha256,
    )


__all__ = [
    "CORE_FILENAMES",
    "IMMUTABLE_MARKER_FILENAME",
    "MANIFEST_FILENAME",
    "ORIGINAL_CONFIRMATORY_TECHNICAL_AUTHORITY_KIND",
    "QUALIFYING_FILENAMES",
    "STOP_FILENAME",
    "OriginalConfirmatoryTechnicalAuthorityBundle",
    "OriginalConfirmatoryTechnicalAuthorityError",
    "VerifiedOriginalConfirmatoryTechnicalAuthority",
    "build_original_confirmatory_technical_authority_bundle_v1",
    "build_original_confirmatory_technical_authority_intent_v1",
    "build_original_confirmatory_technical_authority_review_v1",
    "canonical_json_line_bytes",
    "canonical_json_sha256",
    "canonical_original_confirmatory_technical_authority_intent_v1",
    "canonical_original_confirmatory_technical_authority_review_v1",
    "verify_original_confirmatory_technical_authority_v1",
]
