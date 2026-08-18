"""Fail-closed one-shot controller for replacement Authority-D publication.

The historical controller, its v2 inputs, and its terminal markers are never
imported or reused.  This module reserves a new marker namespace, classifies the
exact terminal-state truth table, and supplies a direct-child fresh verifier plus a
transaction-scoped publication harness.

Its explicit live modes freeze a new four-file input bundle, run a read-only
resource preflight, publish a separate one-attempt authorization receipt, and only
then consume that receipt in one Authority-D transaction. Import has no I/O or
subprocess side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    create_directory_no_overwrite,
    publish_bytes_no_overwrite,
    read_file_anchored,
    rollback_owned_publications,
)

SCHEMA_VERSION = 1
MARKER_PREFIX = "resource_authority_d_replacement_v1_publication_"
ATTEMPT_FILENAME = f"{MARKER_PREFIX}attempt.json"
SUCCESS_FILENAME = f"{MARKER_PREFIX}success.json"
FAILURE_FILENAME = f"{MARKER_PREFIX}failure.json"
ATTEMPT_POLICY = "resource_authority_d_replacement_attempt_v1"
SUCCESS_POLICY = "resource_authority_d_replacement_success_v1"
FAILURE_POLICY = "resource_authority_d_replacement_failure_v1"
_MAX_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_WAIT_SLICE_SECONDS = 0.05
_CLEANUP_GRACE_SECONDS = 2.0
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_AMENDMENT_EVIDENCE_FILENAME = "amendment_evidence.json"
_AMENDMENT_MARKER_FILENAME = ".immutable.json"
_TECHNICAL_SUCCESSOR_PURPOSE = "resource_bounded_confirmatory_technical_successor"
_PRIOR_FAILURE_RECEIPT_FILENAME = "resource_authority_d_prior_publication_failure_receipt_v1.json"
_PUBLICATION_AUTHORIZATION_FILENAME = (
    "resource_authority_d_replacement_publication_authorization_v1.json"
)
_PUBLICATION_AUTHORIZATION_POLICY = "resource_authority_d_replacement_publication_authorization_v1"
_LIVE_PREFLIGHT_POLICY = "resource_authority_d_replacement_live_preflight_v1"
_REPLACEMENT_INPUT_PREFIX = "authority_d_replacement_inputs_"
_RETIRED_INPUT_DIRECTORY_NAME = "authority_d_replacement_inputs_v1"
_RETIRED_INPUT_INVALIDATION_FILENAME = "authority_d_replacement_inputs_v1.invalidation.json"
_RETIRED_INPUT_INVALIDATION_POLICY = "resource_authority_d_replacement_input_bundle_invalidation_v1"
_REPLACEMENT_INPUT_DIRECTORY_NAME = "authority_d_replacement_inputs_v2"
_REPLACEMENT_INPUT_FILENAMES = {
    "frozen_source_receipt": "authority_d_replacement_frozen_source_receipt.json",
    "source_allowlist": "authority_d_replacement_source_allowlist.json",
    "workspace_plan": "authority_d_replacement_workspace_plan.json",
    "cnn_correction_receipt": "authority_d_replacement_cnn_correction_receipt.json",
}
_RUN_STATE_FILENAMES = (
    "registry.csv",
    "integrity_registry.jsonl",
    "run_dispositions.anchor.json",
    "run_dispositions.jsonl",
    "run_stage_attestations.anchor.json",
    "run_stage_attestations.jsonl",
)
_PUBLICATION_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "authorized_at_utc",
        "automatic_retry_allowed",
        "max_attempt_count",
        "authorized_attempt_id",
        "publication",
        "preflight",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_PUBLICATION_FIELDS = frozenset(
    {
        "amendment_timestamp_utc",
        "intended_authority_directory",
        "parent_authority_directory",
        "amendment_schema_version",
        "amendment_purpose",
        "chain_depth",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "contract",
        "preflight_fingerprint_sha256",
        "capacity_observation",
        "compute_observation",
    }
)
_PREFLIGHT_CONTRACT_FIELDS = frozenset(
    {
        "project_root",
        "parent_authority_directory",
        "controller",
        "failed_preflight_receipt",
        "prior_failure_receipt",
        "retired_input_invalidation_receipt",
        "frozen_input_bundle",
        "source",
        "config",
        "manifest",
        "run_state",
        "technical_successor",
        "replacement_state",
        "capacity_contract",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_FILE_RECORD_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_RETIRED_INVALIDATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "invalidated_at_utc",
        "reason_code",
        "reason",
        "retired_bundle",
        "prior_failure_receipt",
        "failed_preflight_evidence",
        "corrected_controller",
        "run_state_sha256",
        "disposition",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_RETIRED_BUNDLE_FIELDS = frozenset(
    {
        "directory",
        "files",
        "records_sha256",
        "controller_path",
        "controller_size_bytes",
        "controller_sha256",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "execution_source_delta_sha256",
        "authorization_sha256",
    }
)
_FAILED_PREFLIGHT_EVIDENCE_FIELDS = frozenset(
    {
        "receipt",
        "stored_observed_at_utc",
        "normalized_observed_at_utc",
        "logs",
        "error_type",
        "error_message",
        "error_sha256",
        "publication_authorization_created",
        "attempt_marker_created",
        "authority_d_created",
        "scientific_run_started",
    }
)
_RETIRED_DISPOSITION_FIELDS = frozenset(
    {
        "v1_preflight_allowed",
        "v1_authorization_allowed",
        "v1_publication_allowed",
        "v1_may_be_modified_moved_or_deleted",
        "prior_failure_receipt_may_be_modified_or_republished",
        "replacement_requires_exact_v2_singleton",
    }
)
_FROZEN_BUNDLE_FIELDS = frozenset(
    {
        "directory",
        "frozen_source_receipt",
        "source_allowlist",
        "workspace_plan",
        "cnn_correction_receipt",
    }
)
_SOURCE_CONTRACT_FIELDS = frozenset(
    {
        "root_sha256",
        "manifest_sha256",
        "delta_sha256",
        "allowlisted_change_count",
    }
)
_CONFIG_CONTRACT_FIELDS = frozenset({"path", "file_sha256", "semantic_sha256"})
_RUN_STATE_CONTRACT_FIELDS = frozenset({"root", "files", "sha256"})
_TECHNICAL_SUCCESSOR_FIELDS = frozenset({"authorization_sha256", "intent_sha256"})
_REPLACEMENT_STATE_FIELDS = frozenset(
    {
        "state",
        "candidate_count",
        "attempt_marker_absent",
        "success_marker_absent",
        "failure_marker_absent",
        "intended_authority_absent",
    }
)
_CAPACITY_CONTRACT_FIELDS = frozenset(
    {
        "resource_capacity_policy_sha256",
        "workspace_plan_sha256",
        "workspace_plan_without_self_hash_sha256",
        "projected_stable_run_bytes",
        "fixed_safety_margin_bytes",
        "minimum_free_bytes_before_tracker",
        "maximum_workspace_bytes",
        "minimum_free_bytes_before_workspace_build",
        "planned_workspace_bytes",
        "required_free_bytes_before",
        "required_free_bytes",
    }
)
_CAPACITY_OBSERVATION_FIELDS = frozenset(
    {
        "observed_at_utc",
        "filesystem_path",
        "observed_free_bytes",
        "required_free_bytes",
        "passed",
    }
)
_COMPUTE_OBSERVATION_FIELDS = frozenset({"evidence", "evidence_sha256"})
_RESOURCE_COMPUTE_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "minimum_available_ram_bytes",
        "policy_sha256",
        "observation",
        "observation_sha256",
        "checked_at_utc",
        "passed",
        "outcome_values_read",
        "prohibited_for_selection_tuning",
        "adaptive_execution_changes_allowed",
    }
)
_AUTHORITY_C_COMPONENT = "20260727T170413.080954Z"
_AUTHORITY_C_PINS = {
    "artifact_root_sha256": ("57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627"),
    "sha256_manifest_sha256": ("4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156"),
    "amendment_evidence_sha256": (
        "c2531787116e125bdb46e862f6803429c72e5d4766d5127f2cefa693e320912a"
    ),
    "immutable_marker_sha256": ("49caed80e2e1c07b14a862767ffd5b674c941a7d5c42f7a950ceb902ecee2821"),
    "chain_depth": 3,
}
_AUTHORITY_C_SOURCE_PINS = {
    "root_sha256": "1179f91725a3027c0397e87691774377bbd4ba5469d588390c72b0b88515547b",
    "manifest_sha256": ("03bcc6020e3be5a22fe257c45820e4e8ebece3ce471c2b6cecff0e3e9419fc66"),
    "artifact_count": 101,
}
_FAILED_PREFLIGHT_FILENAME = "failed_resource_preflight_20260727T173054.689Z.json"
_FAILED_PREFLIGHT_SHA256 = "e308aa0089a84caaca3f0722711e623579372636d47e161d9f32ca5a71f8c6eb"
_FAILED_PREFLIGHT_SIZE_BYTES = 1994
_HISTORICAL_FAILED_AT_UTC = "2026-07-27T17:30:54.689Z"
_PRIOR_FAILURE_RECEIPT_SIZE_BYTES = 11413
_PRIOR_FAILURE_RECEIPT_SHA256 = "2b46f11d1580a6469715a525c0738d39fb3ae0f74f542e142ecd293ae7beed00"
_RETIRED_INPUT_FILE_PINS = {
    "cnn_correction_receipt": {
        "size_bytes": 4452,
        "sha256": "0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e",
    },
    "frozen_source_receipt": {
        "size_bytes": 3663,
        "sha256": "18b573b1b18f2a0fdcefe5f06862a4c900efdf17fcf259567827a6846acf99ea",
    },
    "source_allowlist": {
        "size_bytes": 3903,
        "sha256": "8b90ce20910afef617fc4029c72f2df6cca561e0487844b43d92f3aa94338a70",
    },
    "workspace_plan": {
        "size_bytes": 12186,
        "sha256": "d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b",
    },
}
_RETIRED_CONTROLLER_SIZE_BYTES = 169526
_RETIRED_CONTROLLER_SHA256 = "00320399508e3ab3d8a39b8b5428ebaf84100e0e1c8a20bd913b815d8209dc7c"
_RETIRED_SOURCE_ROOT_SHA256 = "bc1ee2e9f576cdf1628de3c9e5bbbae2356add4040a0fd59dcbc9c7c3d96d552"
_RETIRED_SOURCE_MANIFEST_SHA256 = "bb391be5b0aafcdc1678a02f93af42f5e8e4965b265530c8abac7c2d507fa6b0"
_RETIRED_SOURCE_DELTA_SHA256 = "89cfca6f619a63f327ecf46d60f5b6a7bfe34c3480657d1191b3fa0b3b17cd01"
_RETIRED_AUTHORIZATION_SHA256 = "f8ec5bd94cf2665ede0ded9d1f1d14c94107b1649da2c9c442a97e7765ce8bbe"
_RETIRED_RUN_STATE_SHA256 = "5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4"
_RETIRED_INVALIDATION_REASON_CODE = (
    "external_timestamp_precision_contract_p1_requires_source_change"
)
_RETIRED_INVALIDATION_REASON = (
    "The first read-only preflight of the exact v1 bundle failed closed before "
    "authorization, publication, or scientific execution because the immutable "
    "SHA-pinned failed-preflight receipt uses canonical millisecond UTC precision. "
    "The compatibility correction changes the self-bound controller source, so v1 "
    "is preserved as non-publishable evidence and an exact v2 singleton is required."
)
_RETIRED_FAILURE_ERROR_MESSAGE = "failed-preflight observation is not canonical to microseconds"
_RETIRED_FAILURE_ERROR_SHA256 = "c103cd60f4aba8d6673884180ceb1c168f1f8b1506ae75cabe3311a956fb02e3"
_RETIRED_LOG_PINS = {
    "freeze_stderr": {
        "filename": "option_b_replacement_freeze_20260728T160420.324Z.stderr.log",
        "size_bytes": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "freeze_stdout": {
        "filename": "option_b_replacement_freeze_20260728T160420.324Z.stdout.log",
        "size_bytes": 2564,
        "sha256": "1cb46124ed48800ab5614886b5ecf786e31835ccedf32f30ec62d82b9a9c5a3c",
    },
    "preflight_stderr": {
        "filename": "option_b_replacement_preflight1_20260728T161040.612Z.stderr.log",
        "size_bytes": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "preflight_stdout": {
        "filename": "option_b_replacement_preflight1_20260728T161040.612Z.stdout.log",
        "size_bytes": 259,
        "sha256": "c51dd63b0534b1d2d245709752b64a035f51ab2418385ae4061928e01fac7b1d",
    },
    "diagnostic_stderr": {
        "filename": "option_b_preflight_diagnostic_20260728T161642.776Z.stderr.log",
        "size_bytes": 665,
        "sha256": "163b208d12a2828af832f4e25ea629a6b8e87a8fa2c578437bd83c45ab6cbf6a",
    },
    "diagnostic_stdout": {
        "filename": "option_b_preflight_diagnostic_20260728T161642.776Z.stdout.log",
        "size_bytes": 254,
        "sha256": "1b8916f31e87165480342e8ba432cff57d95eed8e4f8d3f1c2bf41733bd6e4aa",
    },
}
_RESOURCE_CONFIG_FILE_SHA256 = "7acffcf06471554d460e409b543d42ccd16205ad56483205957a63756f75ffb5"
_RESOURCE_CONFIG_SEMANTIC_SHA256 = (
    "af99f0acfe3a075715a2e90d28ae6b896197fc30cf3819303d7b82837a0f6f88"
)
_PANNUKE_MANIFEST_SHA256 = "7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e"
_EXPECTED_SOURCE_CHANGE_KINDS = {
    "configs/confirmatory_resource_bounded_amended.yaml": "modified",
    "src/histo_audit/cli.py": "modified",
    "src/histo_audit/cross_validation/image_oof.py": "modified",
    "src/histo_audit/experiment/confirmatory_completion.py": "modified",
    "src/histo_audit/experiment/confirmatory_core.py": "modified",
    "src/histo_audit/experiment/confirmatory_memory_workspace.py": "added",
    "src/histo_audit/experiment/confirmatory_runner.py": "modified",
    "src/histo_audit/experiment/pannuke_confirmatory_inputs.py": "modified",
    "src/histo_audit/experiment/resource_bounded_runner.py": "modified",
    "src/histo_audit/experiment/study_contracts.py": "modified",
    "src/histo_audit/models/cnn.py": "modified",
    "src/histo_audit/pannuke/publication.py": "modified",
    "src/histo_audit/workflows/__init__.py": "modified",
    "src/histo_audit/workflows/lifecycle_qualification.py": "modified",
    "src/histo_audit/workflows/preregistration_amendment.py": "modified",
    "src/histo_audit/workflows/resource_authority_d_replacement_controller.py": "added",
    "src/histo_audit/workflows/study_gates.py": "modified",
}
_SOURCE_ALLOWLIST_POLICY = "resource_authority_d_replacement_exact_17_path_allowlist_v1"
_FROZEN_SOURCE_RECEIPT_POLICY = "resource_authority_d_replacement_frozen_source_receipt_v2"
_CNN_CORRECTION_RECEIPT_POLICY = "resource_authority_d_replacement_cnn_correction_receipt_v1"
_CNN_SEMANTIC_EQUIVALENCE_POLICY = "resource_authority_d_replacement_cnn_semantic_equivalence_v1"
_OUTCOMES_INSPECTED_AT = datetime.fromisoformat("2026-07-27T10:57:07+00:00")
_AMENDMENT_REASON = (
    "Publish the direct technical successor to resource authority C after its "
    "outcome-blind preflight exposed stale logical CNN provenance and excessive "
    "input duplication; bind only the corrected logical provenance and the exact "
    "capacity-v3 indexed workspace without changing the scientific profile."
)
_AFFECTED_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
_AFFECTED_ANALYSES = (
    "resource_bounded_confirmatory_sensitivity",
    "confirmatory_fold_rotation",
    "confirmatory_ranking_statistics",
    "confirmatory_restoration_analysis",
    "confirmatory_model_representation_comparisons",
    "confirmatory_checkpoint_successor_lineage",
    "confirmatory_completion_and_m9_eligibility",
)

_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "automatic_retry_allowed",
        "attempt_id",
        "intended_authority_directory",
        "parent_authority_directory",
        "controller_path",
        "controller_size_bytes",
        "controller_sha256",
        "failed_preflight_receipt_path",
        "failed_preflight_receipt_sha256",
        "prior_failure_receipt_path",
        "prior_failure_receipt_sha256",
        "retired_input_invalidation_path",
        "retired_input_invalidation_sha256",
        "frozen_source_receipt_path",
        "frozen_source_receipt_sha256",
        "source_allowlist_path",
        "source_allowlist_sha256",
        "workspace_plan_path",
        "workspace_plan_sha256",
        "cnn_correction_receipt_path",
        "cnn_correction_receipt_sha256",
        "publication_authorization_receipt_path",
        "publication_authorization_receipt_sha256",
        "preflight_fingerprint_sha256",
        "max_attempt_count",
        "amendment_timestamp_utc",
        "authorization_sha256",
        "intent_sha256",
        "run_state_sha256",
    }
)
_SUCCESS_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "automatic_retry_allowed",
        "attempt_id",
        "attempt_marker_sha256",
        "authority_directory",
        "parent_authority_directory",
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "authorization_sha256",
        "intent_sha256",
        "verification_nonce",
        "fresh_verifier_payload_sha256",
        "controller_process_id",
        "verifier_process_id",
        "verifier_parent_process_id",
        "chain_depth",
        "run_state_sha256",
    }
)
_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "automatic_retry_allowed",
        "attempt_id",
        "attempt_marker_sha256",
        "intended_authority_directory",
        "parent_authority_directory",
        "error_type_sha256",
        "error_sha256",
        "authority_absent_after_rollback",
        "run_state_sha256",
    }
)
_CHECK_FIELDS = frozenset(
    {
        "generic_chain_integrity",
        "typed_successor_authorization",
        "effective_execution_leaf",
        "historical_c_integrity",
        "historical_c_typed_authorization",
        "c_superseded_for_execution",
        "unique_direct_successor",
        "storage_policy_inherited_unchanged",
        "flat_exact_file_set",
        "external_intent_binding",
        "fresh_process_boundary",
        "live_prior_publication_failure",
        "live_failed_preflight_receipt",
        "live_historical_primary",
    }
)


class ControlError(RuntimeError):
    """Base replacement-controller error."""


class AmbiguousStateError(ControlError):
    """Observed state is outside the exact truth table."""


class FreshVerifierError(ControlError):
    """Fresh direct-child verification failed closed."""


def _fresh_verifier_spawn_executable(python_executable: str) -> str | None:
    """Return the trusted Windows interpreter behind a virtual-env launcher."""

    if os.name != "nt":
        return None
    raw_executable = getattr(sys, "_base_executable", None)
    if type(raw_executable) is not str or not raw_executable:
        raise FreshVerifierError("Windows base Python executable is unavailable")
    raw_base_prefix = sys.base_prefix
    if type(raw_base_prefix) is not str or not raw_base_prefix:
        raise FreshVerifierError("Windows base Python prefix is unavailable")
    candidate = Path(raw_executable).expanduser()
    base_prefix_candidate = Path(raw_base_prefix).expanduser()
    if not candidate.is_absolute():
        raise FreshVerifierError("Windows base Python executable must be absolute")
    if not base_prefix_candidate.is_absolute():
        raise FreshVerifierError("Windows base Python prefix must be absolute")
    try:
        metadata = candidate.lstat()
        base_prefix = _real_directory(
            base_prefix_candidate,
            "Windows base Python prefix",
        )
        parent = _real_directory(
            candidate.parent,
            "Windows base Python executable parent",
        )
        resolved_candidate = candidate.resolve(strict=True)
        expected_candidate = (base_prefix / Path(python_executable).name).resolve(strict=True)
    except (OSError, ControlError, ValueError) as exc:
        raise FreshVerifierError("Windows base Python executable cannot be trusted") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE
    ):
        raise FreshVerifierError("Windows base Python executable is not a real regular file")
    if not _same(parent, base_prefix) or not _same(
        resolved_candidate,
        expected_candidate,
    ):
        raise FreshVerifierError(
            "Windows base Python executable differs from the base-prefix interpreter"
        )
    if resolved_candidate.suffix.casefold() != ".exe":
        raise FreshVerifierError("Windows base Python executable is not an EXE")
    return str(resolved_candidate)


class State(StrEnum):
    READY = "ready"
    ROLLED_BACK_FAILURE = "rolled_back_failure"
    COMMITTED = "committed"
    STOP_AMBIGUOUS = "stop_ambiguous"


@dataclass(frozen=True, slots=True)
class Namespace:
    control_root: Path

    @classmethod
    def for_project(cls, root: str | Path) -> Namespace:
        return cls(_absolute(root) / "artifacts" / "resource_control")

    @property
    def attempt(self) -> Path:
        return self.control_root / ATTEMPT_FILENAME

    @property
    def success(self) -> Path:
        return self.control_root / SUCCESS_FILENAME

    @property
    def failure(self) -> Path:
        return self.control_root / FAILURE_FILENAME

    @property
    def publication_authorization(self) -> Path:
        return self.control_root / _PUBLICATION_AUTHORIZATION_FILENAME


@dataclass(frozen=True, slots=True)
class FrozenInputBindings:
    """Exact immutable inputs consumed by the one-shot replacement attempt."""

    failed_preflight_receipt: Path
    prior_failure_receipt: Path
    retired_input_invalidation: Path
    frozen_source_receipt: Path
    source_allowlist: Path
    workspace_plan: Path
    cnn_correction_receipt: Path

    def paths(self) -> dict[str, Path]:
        return {
            "failed_preflight_receipt": _absolute(self.failed_preflight_receipt),
            "prior_failure_receipt": _absolute(self.prior_failure_receipt),
            "retired_input_invalidation": _absolute(self.retired_input_invalidation),
            "frozen_source_receipt": _absolute(self.frozen_source_receipt),
            "source_allowlist": _absolute(self.source_allowlist),
            "workspace_plan": _absolute(self.workspace_plan),
            "cnn_correction_receipt": _absolute(self.cnn_correction_receipt),
        }


@dataclass(frozen=True, slots=True)
class Classification:
    state: State
    reason: str
    candidates: tuple[Path, ...]
    attempt_sha256: str | None = None
    success_sha256: str | None = None
    failure_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        publication_performed: bool | None
        if self.state is State.COMMITTED:
            publication_performed = True
        elif self.state in {State.READY, State.ROLLED_BACK_FAILURE}:
            publication_performed = False
        else:
            publication_performed = None
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": "resource_authority_d_replacement_state_classifier_v1",
            "state": self.state.value,
            "reason": self.reason,
            "candidate_directories": [str(path) for path in self.candidates],
            "attempt_marker_sha256": self.attempt_sha256,
            "success_marker_sha256": self.success_sha256,
            "failure_marker_sha256": self.failure_sha256,
            "automatic_retry_allowed": False,
            "publication_performed": publication_performed,
        }


@dataclass(frozen=True, slots=True)
class VerifyRequest:
    project_root: Path
    successor_directory: Path
    parent_directory: Path
    artifact_root_sha256: str
    manifest_sha256: str
    authorization_sha256: str
    intent_sha256: str
    nonce: str
    chain_depth: int = 4
    python_executable: str = sys.executable

    def checked(self) -> VerifyRequest:
        if type(self.chain_depth) is not int or self.chain_depth <= 0:
            raise FreshVerifierError("chain depth must be a positive exact integer")
        executable = Path(self.python_executable).expanduser()
        if not executable.is_absolute():
            raise FreshVerifierError("fresh verifier executable must be absolute")
        try:
            executable = executable.resolve(strict=True)
            expected_executable = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise FreshVerifierError("fresh verifier executable cannot be resolved") from exc
        if os.path.normcase(str(executable)) != os.path.normcase(str(expected_executable)):
            raise FreshVerifierError(
                "fresh verifier must use this controller's exact Python executable"
            )
        return VerifyRequest(
            project_root=_absolute(self.project_root),
            successor_directory=_absolute(self.successor_directory),
            parent_directory=_absolute(self.parent_directory),
            artifact_root_sha256=_sha(self.artifact_root_sha256, "artifact root"),
            manifest_sha256=_sha(self.manifest_sha256, "manifest"),
            authorization_sha256=_sha(self.authorization_sha256, "authorization"),
            intent_sha256=_sha(self.intent_sha256, "intent"),
            nonce=_sha(self.nonce, "nonce"),
            chain_depth=self.chain_depth,
            python_executable=str(executable),
        )

    def argv(self, controller_pid: int) -> tuple[str, ...]:
        request = self.checked()
        if type(controller_pid) is not int or controller_pid <= 0:
            raise FreshVerifierError("controller PID must be a positive exact integer")
        return (
            request.python_executable,
            "-I",
            "-B",
            "-m",
            "histo_audit",
            "preregistration",
            "verify-resource-technical-successor",
            "--project-root",
            str(request.project_root),
            "--successor-dir",
            str(request.successor_directory),
            "--expected-parent-authority-dir",
            str(request.parent_directory),
            "--expected-artifact-root-sha256",
            request.artifact_root_sha256,
            "--expected-sha256-manifest-sha256",
            request.manifest_sha256,
            "--expected-authorization-sha256",
            request.authorization_sha256,
            "--expected-intent-sha256",
            request.intent_sha256,
            "--expected-controller-pid",
            str(controller_pid),
            "--verification-nonce",
            request.nonce,
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    request: VerifyRequest
    argv: tuple[str, ...]
    process_id: int
    payload: dict[str, Any]
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class PublicationResult:
    state: State
    marker_path: Path
    marker_sha256: str
    authority_directory: Path | None


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(_absolute(left)))) == os.path.normcase(
        os.path.normpath(str(_absolute(right)))
    )


def _sha(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlError(f"{role} must be one lowercase SHA-256")
    return value


def _canonical_timestamp(value: object, role: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ControlError(f"{role} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ControlError(f"{role} is not a valid timestamp") from exc
    canonical = parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value != canonical:
        raise ControlError(f"{role} is not canonical to microseconds")
    return parsed.astimezone(UTC)


def _canonical_historical_failed_timestamp(value: object, role: str) -> datetime:
    """Parse only the exact SHA-pinned historical millisecond UTC spelling."""

    if value != _HISTORICAL_FAILED_AT_UTC or not isinstance(value, str):
        raise ControlError(f"{role} differs from the exact historical timestamp")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value) is None:
        raise ControlError(f"{role} is not canonical to milliseconds")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(UTC)
    except ValueError as exc:  # pragma: no cover - exact spelling is statically valid
        raise ControlError(f"{role} is not a valid historical timestamp") from exc
    canonical = parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if canonical != value:
        raise ControlError(f"{role} is not canonical to milliseconds")
    return parsed


def _exact_nonnegative_int(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        raise ControlError(f"{role} must be a nonnegative exact integer")
    return value


def _exact_positive_int(value: object, role: str) -> int:
    result = _exact_nonnegative_int(value, role)
    if result == 0:
        raise ControlError(f"{role} must be positive")
    return result


def _canonical_file_record(
    value: object,
    role: str,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    record = _exact_dict(value, _FILE_RECORD_FIELDS, role)
    path = _absolute(record["path"])
    if not Path(str(record["path"])).is_absolute():
        raise ControlError(f"{role} path must be absolute")
    return {
        "path": str(path),
        "size_bytes": (
            _exact_nonnegative_int(record["size_bytes"], f"{role} size")
            if allow_empty
            else _exact_positive_int(record["size_bytes"], f"{role} size")
        ),
        "sha256": _sha(record["sha256"], role),
    }


def _canonical_publication_authorization(
    value: object,
    *,
    namespace: Namespace,
) -> dict[str, Any]:
    """Validate the closed one-attempt decision receipt without mutating state."""

    payload = _exact_dict(
        value,
        _PUBLICATION_AUTHORIZATION_FIELDS,
        "publication authorization receipt",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["policy"] != _PUBLICATION_AUTHORIZATION_POLICY
        or payload["status"] != "authorized_for_one_attempt"
        or payload["automatic_retry_allowed"] is not False
        or type(payload["max_attempt_count"]) is not int
        or payload["max_attempt_count"] != 1
        or payload["outcome_value_interpretation_performed"] is not False
        or payload["scientific_execution_performed"] is not False
        or payload["publication_performed"] is not False
    ):
        raise ControlError("publication authorization fixed policy is invalid")
    attempt_id = _sha(payload["authorized_attempt_id"], "authorized attempt id")
    authorized_at = _canonical_timestamp(
        payload["authorized_at_utc"],
        "publication authorization time",
    )
    publication = _exact_dict(
        payload["publication"],
        _PUBLICATION_FIELDS,
        "authorized publication",
    )
    amendment_at = _canonical_timestamp(
        publication["amendment_timestamp_utc"],
        "authorized amendment timestamp",
    )
    if amendment_at != authorized_at:
        raise ControlError(
            "authorization and amendment timestamps must be the same captured instant"
        )
    parent = _absolute(publication["parent_authority_directory"])
    destination = _absolute(publication["intended_authority_directory"])
    project_root, parent, destination_checked = _require_project_layout(
        namespace,
        parent=parent,
        destination=destination,
    )
    if destination_checked is None:
        raise ControlError("authorized publication destination is missing")
    expected_component = amendment_at.strftime("%Y%m%dT%H%M%S.%fZ")
    if (
        destination.name != expected_component
        or type(publication["amendment_schema_version"]) is not int
        or publication["amendment_schema_version"] != 5
        or publication["amendment_purpose"] != _TECHNICAL_SUCCESSOR_PURPOSE
        or type(publication["chain_depth"]) is not int
        or publication["chain_depth"] != 4
    ):
        raise ControlError("authorized publication identity is invalid")

    preflight = _exact_dict(payload["preflight"], _PREFLIGHT_FIELDS, "live preflight")
    if (
        type(preflight["schema_version"]) is not int
        or preflight["schema_version"] != SCHEMA_VERSION
        or preflight["policy"] != _LIVE_PREFLIGHT_POLICY
        or preflight["status"] != "passed"
    ):
        raise ControlError("live preflight fixed policy is invalid")
    contract = _exact_dict(
        preflight["contract"],
        _PREFLIGHT_CONTRACT_FIELDS,
        "live preflight contract",
    )
    if (
        not _same(contract["project_root"], project_root)
        or not _same(contract["parent_authority_directory"], parent)
        or any(
            contract[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("live preflight project/behavior contract is invalid")
    controller = _canonical_file_record(contract["controller"], "preflight controller")
    if not _same(controller["path"], Path(__file__)):
        raise ControlError("live preflight controller is not this executing module")
    failed = _canonical_file_record(
        contract["failed_preflight_receipt"],
        "failed-preflight receipt",
    )
    prior = _canonical_file_record(
        contract["prior_failure_receipt"],
        "prior-publication failure receipt",
    )
    invalidation = _canonical_file_record(
        contract["retired_input_invalidation_receipt"],
        "retired input invalidation receipt",
    )
    control_root = _absolute(namespace.control_root)
    if (
        Path(failed["path"]) != control_root / _FAILED_PREFLIGHT_FILENAME
        or failed["size_bytes"] != _FAILED_PREFLIGHT_SIZE_BYTES
        or failed["sha256"] != _FAILED_PREFLIGHT_SHA256
        or Path(prior["path"]) != control_root / _PRIOR_FAILURE_RECEIPT_FILENAME
        or prior["size_bytes"] != _PRIOR_FAILURE_RECEIPT_SIZE_BYTES
        or prior["sha256"] != _PRIOR_FAILURE_RECEIPT_SHA256
        or Path(invalidation["path"]) != control_root / _RETIRED_INPUT_INVALIDATION_FILENAME
    ):
        raise ControlError("live preflight receipt paths are outside the exact namespace")

    bundle = _exact_dict(
        contract["frozen_input_bundle"],
        _FROZEN_BUNDLE_FIELDS,
        "frozen replacement bundle",
    )
    bundle_root = _replacement_input_directory(
        control_root,
        bundle["directory"],
    )
    if not Path(str(bundle["directory"])).is_absolute():
        raise ControlError("frozen replacement bundle path is outside its namespace")
    canonical_bundle: dict[str, Any] = {"directory": str(bundle_root)}
    for role, filename in _REPLACEMENT_INPUT_FILENAMES.items():
        record = _canonical_file_record(bundle[role], f"frozen replacement {role}")
        if Path(record["path"]) != bundle_root / filename:
            raise ControlError(f"frozen replacement path differs for {role}")
        canonical_bundle[role] = record

    source = _exact_dict(contract["source"], _SOURCE_CONTRACT_FIELDS, "source contract")
    canonical_source = {
        "root_sha256": _sha(source["root_sha256"], "source root"),
        "manifest_sha256": _sha(source["manifest_sha256"], "source manifest"),
        "delta_sha256": _sha(source["delta_sha256"], "source delta"),
        "allowlisted_change_count": _exact_positive_int(
            source["allowlisted_change_count"],
            "source allowlisted change count",
        ),
    }
    config = _exact_dict(contract["config"], _CONFIG_CONTRACT_FIELDS, "config contract")
    config_path = _absolute(config["path"])
    if (
        not Path(str(config["path"])).is_absolute()
        or config_path != project_root / "configs" / "confirmatory_resource_bounded_amended.yaml"
    ):
        raise ControlError("resource config path is not canonical")
    canonical_config = {
        "path": str(config_path),
        "file_sha256": _sha(config["file_sha256"], "config file"),
        "semantic_sha256": _sha(config["semantic_sha256"], "config semantics"),
    }
    manifest = _canonical_file_record(contract["manifest"], "PanNuke manifest")
    if Path(manifest["path"]) != (
        project_root / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
    ):
        raise ControlError("PanNuke manifest path is not canonical")

    run_state = _exact_dict(
        contract["run_state"],
        _RUN_STATE_CONTRACT_FIELDS,
        "run-state contract",
    )
    run_root = _absolute(run_state["root"])
    files = run_state["files"]
    if (
        not Path(str(run_state["root"])).is_absolute()
        or run_root != project_root / "artifacts" / "runs"
        or type(files) is not dict
        or tuple(sorted(files)) != tuple(sorted(_RUN_STATE_FILENAMES))
    ):
        raise ControlError("run-state contract has the wrong root or file set")
    canonical_run_files = {name: _sha(files[name], f"run-state {name}") for name in sorted(files)}
    run_state_sha256 = _sha(run_state["sha256"], "run-state contract")
    if run_state_sha256 != _compact_sha256(canonical_run_files):
        raise ControlError("run-state contract digest differs from its six files")

    technical = _exact_dict(
        contract["technical_successor"],
        _TECHNICAL_SUCCESSOR_FIELDS,
        "technical successor contract",
    )
    canonical_technical = {
        "authorization_sha256": _sha(
            technical["authorization_sha256"],
            "technical successor authorization",
        ),
        "intent_sha256": _sha(
            technical["intent_sha256"],
            "technical successor intent",
        ),
    }
    replacement = _exact_dict(
        contract["replacement_state"],
        _REPLACEMENT_STATE_FIELDS,
        "replacement state contract",
    )
    expected_replacement = {
        "state": State.READY.value,
        "candidate_count": 0,
        "attempt_marker_absent": True,
        "success_marker_absent": True,
        "failure_marker_absent": True,
        "intended_authority_absent": True,
    }
    if replacement != expected_replacement or any(
        type(replacement[field]) is not int for field in ("candidate_count",)
    ):
        raise ControlError("replacement preflight state is not exact READY")

    capacity = _exact_dict(
        contract["capacity_contract"],
        _CAPACITY_CONTRACT_FIELDS,
        "capacity contract",
    )
    for field_name in (
        "resource_capacity_policy_sha256",
        "workspace_plan_sha256",
        "workspace_plan_without_self_hash_sha256",
    ):
        _sha(capacity[field_name], f"capacity {field_name}")
    integer_fields = (
        "projected_stable_run_bytes",
        "fixed_safety_margin_bytes",
        "minimum_free_bytes_before_tracker",
        "maximum_workspace_bytes",
        "minimum_free_bytes_before_workspace_build",
        "planned_workspace_bytes",
        "required_free_bytes_before",
        "required_free_bytes",
    )
    canonical_capacity = {
        field_name: _exact_positive_int(
            capacity[field_name],
            f"capacity {field_name}",
        )
        for field_name in integer_fields
    }
    canonical_capacity.update(
        {
            field_name: capacity[field_name]
            for field_name in (
                "resource_capacity_policy_sha256",
                "workspace_plan_sha256",
                "workspace_plan_without_self_hash_sha256",
            )
        }
    )
    if (
        canonical_capacity["fixed_safety_margin_bytes"] != 10 * 1024**3
        or canonical_capacity["minimum_free_bytes_before_tracker"]
        != canonical_capacity["projected_stable_run_bytes"]
        + canonical_capacity["fixed_safety_margin_bytes"]
        or canonical_capacity["minimum_free_bytes_before_workspace_build"]
        != canonical_capacity["minimum_free_bytes_before_tracker"]
        + canonical_capacity["maximum_workspace_bytes"]
        or canonical_capacity["required_free_bytes"]
        != max(
            canonical_capacity["minimum_free_bytes_before_workspace_build"],
            canonical_capacity["required_free_bytes_before"],
        )
    ):
        raise ControlError("capacity contract arithmetic is not conservative")

    observation = _exact_dict(
        preflight["capacity_observation"],
        _CAPACITY_OBSERVATION_FIELDS,
        "capacity observation",
    )
    observed_at = _canonical_timestamp(
        observation["observed_at_utc"],
        "capacity observation time",
    )
    filesystem_path = _absolute(observation["filesystem_path"])
    observed_free = _exact_nonnegative_int(
        observation["observed_free_bytes"],
        "observed free bytes",
    )
    if (
        observed_at > authorized_at
        or not _same(filesystem_path, run_root)
        or observation["required_free_bytes"] != canonical_capacity["required_free_bytes"]
        or type(observation["required_free_bytes"]) is not int
        or observation["passed"] is not True
        or observed_free < canonical_capacity["required_free_bytes"]
    ):
        raise ControlError("capacity observation did not pass the exact contract")
    compute = _exact_dict(
        preflight["compute_observation"],
        _COMPUTE_OBSERVATION_FIELDS,
        "compute observation",
    )
    compute_evidence = _exact_dict(
        compute["evidence"],
        _RESOURCE_COMPUTE_EVIDENCE_FIELDS,
        "compute evidence",
    )
    compute_checked_at = _canonical_timestamp(
        compute_evidence["checked_at_utc"],
        "compute observation time",
    )
    if (
        type(compute_evidence["schema_version"]) is not int
        or compute_evidence["schema_version"] != 1
        or compute_evidence["phase"] != "guarded_before_data_loading"
        or _exact_positive_int(
            compute_evidence["minimum_available_ram_bytes"],
            "compute minimum available RAM",
        )
        <= 0
        or compute_evidence["policy_sha256"]
        != canonical_capacity["resource_capacity_policy_sha256"]
        or not isinstance(compute_evidence["observation"], Mapping)
        or compute_evidence["observation_sha256"]
        != _compact_sha256(compute_evidence["observation"])
        or compute_checked_at > authorized_at
        or compute_evidence["passed"] is not True
        or compute_evidence["outcome_values_read"] is not False
        or compute_evidence["prohibited_for_selection_tuning"] is not True
        or compute_evidence["adaptive_execution_changes_allowed"] is not False
        or compute["evidence_sha256"] != _compact_sha256(compute_evidence)
    ):
        raise ControlError("compute observation did not pass the exact public contract")

    canonical_contract = {
        "project_root": str(project_root),
        "parent_authority_directory": str(parent),
        "controller": controller,
        "failed_preflight_receipt": failed,
        "prior_failure_receipt": prior,
        "retired_input_invalidation_receipt": invalidation,
        "frozen_input_bundle": canonical_bundle,
        "source": canonical_source,
        "config": canonical_config,
        "manifest": manifest,
        "run_state": {
            "root": str(run_root),
            "files": canonical_run_files,
            "sha256": run_state_sha256,
        },
        "technical_successor": canonical_technical,
        "replacement_state": expected_replacement,
        "capacity_contract": canonical_capacity,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    fingerprint = _sha(
        preflight["preflight_fingerprint_sha256"],
        "preflight fingerprint",
    )
    if fingerprint != _compact_sha256(canonical_contract):
        raise ControlError("preflight fingerprint differs from its exact stable contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": _PUBLICATION_AUTHORIZATION_POLICY,
        "status": "authorized_for_one_attempt",
        "authorized_at_utc": payload["authorized_at_utc"],
        "automatic_retry_allowed": False,
        "max_attempt_count": 1,
        "authorized_attempt_id": attempt_id,
        "publication": {
            "amendment_timestamp_utc": publication["amendment_timestamp_utc"],
            "intended_authority_directory": str(destination),
            "parent_authority_directory": str(parent),
            "amendment_schema_version": 5,
            "amendment_purpose": _TECHNICAL_SUCCESSOR_PURPOSE,
            "chain_depth": 4,
        },
        "preflight": {
            "schema_version": SCHEMA_VERSION,
            "policy": _LIVE_PREFLIGHT_POLICY,
            "status": "passed",
            "contract": canonical_contract,
            "preflight_fingerprint_sha256": fingerprint,
            "capacity_observation": {
                "observed_at_utc": observation["observed_at_utc"],
                "filesystem_path": str(filesystem_path),
                "observed_free_bytes": observed_free,
                "required_free_bytes": canonical_capacity["required_free_bytes"],
                "passed": True,
            },
            "compute_observation": {
                "evidence": compute_evidence,
                "evidence_sha256": compute["evidence_sha256"],
            },
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }


def _exact_dict(value: object, fields: frozenset[str], role: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        raise ControlError(f"{role} differs from its exact closed schema")
    return dict(value)


def _real_directory(path: str | Path, role: str) -> Path:
    candidate = _absolute(path)
    for component in (*reversed(candidate.parents), candidate):
        try:
            value = os.lstat(component)
        except OSError as exc:
            raise ControlError(f"{role} has a missing lexical ancestor: {component}") from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or getattr(value, "st_file_attributes", 0) & _REPARSE
        ):
            raise ControlError(
                f"{role} must traverse only real non-reparse directories: {component}"
            )
    return candidate


def _namespace_project_root(namespace: Namespace) -> Path:
    control_root = _real_directory(namespace.control_root, "replacement control root")
    if control_root.name != "resource_control" or control_root.parent.name != "artifacts":
        raise ControlError("replacement control root must be PROJECT/artifacts/resource_control")
    project_root = _real_directory(control_root.parent.parent, "replacement project root")
    if not _same(control_root, project_root / "artifacts" / "resource_control"):
        raise ControlError("replacement namespace is not bound to one exact project")
    return project_root


def _replacement_input_directory(
    control_root: str | Path,
    value: str | Path,
) -> Path:
    """Return the one canonical replacement-input directory or fail closed."""

    root = _absolute(value)
    if root.parent != _absolute(control_root) or root.name != _REPLACEMENT_INPUT_DIRECTORY_NAME:
        raise ControlError("replacement input directory must use the one canonical singleton path")
    return root


def _require_project_layout(
    namespace: Namespace,
    *,
    parent: str | Path,
    destination: str | Path | None = None,
    verifier_project_root: str | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    project_root = _namespace_project_root(namespace)
    parent_path = _real_directory(parent, "Authority C")
    expected_amendment_root = project_root / "artifacts" / "preregistration_amendments"
    if not _same(parent_path.parent, expected_amendment_root):
        raise ControlError("Authority C is not a direct authority in the bound project")
    destination_path = _absolute(destination) if destination is not None else None
    if destination_path is not None and (
        not _same(destination_path.parent, expected_amendment_root)
        or _same(destination_path, parent_path)
    ):
        raise ControlError("Authority D is not a distinct direct peer in the bound project")
    if verifier_project_root is not None and not _same(verifier_project_root, project_root):
        raise ControlError("fresh verifier project root differs from the marker namespace")
    return project_root, parent_path, destination_path


def _capture_attempt_inputs(
    namespace: Namespace,
    bindings: FrozenInputBindings,
    *,
    capture_run_state: bool = True,
) -> tuple[dict[str, Path], dict[str, str], str | None]:
    project_root = _namespace_project_root(namespace)
    control_root = _absolute(namespace.control_root)
    paths = bindings.paths()
    failed = paths["failed_preflight_receipt"]
    prior = paths["prior_failure_receipt"]
    invalidation = paths["retired_input_invalidation"]
    if failed != control_root / _FAILED_PREFLIGHT_FILENAME:
        raise ControlError("failed-preflight receipt path is not the exact historical pin")
    if prior != control_root / _PRIOR_FAILURE_RECEIPT_FILENAME:
        raise ControlError("prior-publication failure receipt path is not canonical")
    if invalidation != control_root / _RETIRED_INPUT_INVALIDATION_FILENAME:
        raise ControlError("retired input invalidation receipt path is not canonical")
    replacement_roles = tuple(_REPLACEMENT_INPUT_FILENAMES)
    replacement_parents = {paths[role].parent for role in replacement_roles}
    if len(replacement_parents) != 1:
        raise ControlError("replacement frozen inputs must share one immutable bundle")
    replacement_root = _replacement_input_directory(
        control_root,
        next(iter(replacement_parents)),
    )
    for role, filename in _REPLACEMENT_INPUT_FILENAMES.items():
        if paths[role] != replacement_root / filename:
            raise ControlError(f"replacement frozen input path differs for {role}")
    _real_directory(replacement_root, "replacement frozen input bundle")
    observed_children = tuple(sorted(entry.name for entry in os.scandir(replacement_root)))
    expected_children = tuple(sorted(_REPLACEMENT_INPUT_FILENAMES.values()))
    if observed_children != expected_children:
        raise ControlError("replacement frozen input bundle has a non-closed inventory")
    hashes: dict[str, str] = {}
    encoded_inputs: dict[str, bytes] = {}
    for role, path in paths.items():
        try:
            encoded = read_file_anchored(path, max_bytes=_MAX_INPUT_BYTES)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ControlError(f"{role} failed anchored bounded readback") from exc
        encoded_inputs[role] = encoded
        hashes[role] = hashlib.sha256(encoded).hexdigest()
    if (
        len(encoded_inputs["failed_preflight_receipt"]) != _FAILED_PREFLIGHT_SIZE_BYTES
        or hashes["failed_preflight_receipt"] != _FAILED_PREFLIGHT_SHA256
    ):
        raise ControlError("failed-preflight receipt differs from its exact historical pin")
    if (
        len(encoded_inputs["prior_failure_receipt"]) != _PRIOR_FAILURE_RECEIPT_SIZE_BYTES
        or hashes["prior_failure_receipt"] != _PRIOR_FAILURE_RECEIPT_SHA256
    ):
        raise ControlError("prior-publication failure receipt differs from its exact pin")
    frozen_receipt = _strict_json_object(
        encoded_inputs["frozen_source_receipt"],
        "replacement frozen-source receipt",
    )
    expected_receipt_hashes = {
        "source_allowlist_sha256": hashes["source_allowlist"],
        "workspace_plan_sha256": hashes["workspace_plan"],
        "cnn_correction_receipt_sha256": hashes["cnn_correction_receipt"],
    }
    if any(
        frozen_receipt.get(field_name) != expected
        for field_name, expected in expected_receipt_hashes.items()
    ):
        raise ControlError("frozen-source receipt does not bind the exact input bundle")
    if (
        frozen_receipt.get("retired_input_invalidation_receipt_path") != str(invalidation)
        or frozen_receipt.get("retired_input_invalidation_receipt_sha256")
        != hashes["retired_input_invalidation"]
    ):
        raise ControlError("frozen-source receipt does not bind v1 invalidation")
    _, invalidation_sha256 = _read_retired_input_invalidation(
        namespace,
        verify_live_run_state=capture_run_state,
    )
    if invalidation_sha256 != hashes["retired_input_invalidation"]:
        raise ControlError("retired input invalidation changed during input capture")
    run_state_sha256: str | None = None
    if capture_run_state:
        run_root = _real_directory(
            project_root / "artifacts" / "runs",
            "run-state registry root",
        )
        run_state: dict[str, str] = {}
        for filename in _RUN_STATE_FILENAMES:
            try:
                encoded = read_file_anchored(
                    run_root / filename,
                    max_bytes=_MAX_INPUT_BYTES,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ControlError(f"run-state file failed anchored readback: {filename}") from exc
            run_state[filename] = hashlib.sha256(encoded).hexdigest()
        run_state_sha256 = _compact_sha256(run_state)
    return paths, hashes, run_state_sha256


def _bindings_from_attempt(attempt: Mapping[str, Any]) -> FrozenInputBindings:
    return FrozenInputBindings(
        failed_preflight_receipt=Path(attempt["failed_preflight_receipt_path"]),
        prior_failure_receipt=Path(attempt["prior_failure_receipt_path"]),
        retired_input_invalidation=Path(attempt["retired_input_invalidation_path"]),
        frozen_source_receipt=Path(attempt["frozen_source_receipt_path"]),
        source_allowlist=Path(attempt["source_allowlist_path"]),
        workspace_plan=Path(attempt["workspace_plan_path"]),
        cnn_correction_receipt=Path(attempt["cnn_correction_receipt_path"]),
    )


def _verify_attempt_inputs(
    namespace: Namespace,
    attempt: Mapping[str, Any],
    *,
    verify_live_run_state: bool = True,
) -> None:
    paths, hashes, run_state_sha256 = _capture_attempt_inputs(
        namespace,
        _bindings_from_attempt(attempt),
        capture_run_state=verify_live_run_state,
    )
    for role, path in paths.items():
        if not _same(path, attempt[f"{role}_path"]):
            raise ControlError(f"attempt input path changed for {role}")
        if hashes[role] != attempt[f"{role}_sha256"]:
            raise ControlError(f"attempt input digest changed for {role}")
    if verify_live_run_state and run_state_sha256 != attempt["run_state_sha256"]:
        raise ControlError("exact run-state files changed after attempt capture")
    receipt_path = _absolute(attempt["publication_authorization_receipt_path"])
    if receipt_path != namespace.publication_authorization:
        raise ControlError("attempt publication authorization receipt path changed")
    receipt, receipt_sha256 = _read_publication_authorization(namespace)
    if receipt_sha256 != attempt["publication_authorization_receipt_sha256"]:
        raise ControlError("attempt publication authorization receipt digest changed")
    _verify_publication_authorization_live_files(
        namespace,
        receipt,
        verify_run_state=verify_live_run_state,
    )
    publication = receipt["publication"]
    preflight = receipt["preflight"]
    contract = preflight["contract"]
    technical = contract["technical_successor"]
    controller = contract["controller"]
    if (
        receipt["authorized_attempt_id"] != attempt["attempt_id"]
        or receipt["max_attempt_count"] != attempt["max_attempt_count"]
        or publication["amendment_timestamp_utc"] != attempt["amendment_timestamp_utc"]
        or not _same(
            publication["intended_authority_directory"],
            attempt["intended_authority_directory"],
        )
        or not _same(
            publication["parent_authority_directory"],
            attempt["parent_authority_directory"],
        )
        or preflight["preflight_fingerprint_sha256"] != attempt["preflight_fingerprint_sha256"]
        or technical["authorization_sha256"] != attempt["authorization_sha256"]
        or technical["intent_sha256"] != attempt["intent_sha256"]
        or not _same(controller["path"], attempt["controller_path"])
        or controller["size_bytes"] != attempt["controller_size_bytes"]
        or controller["sha256"] != attempt["controller_sha256"]
        or contract["run_state"]["sha256"] != attempt["run_state_sha256"]
    ):
        raise ControlError("attempt differs from its publication authorization receipt")
    receipt_input_records = {
        "failed_preflight_receipt": contract["failed_preflight_receipt"],
        "prior_failure_receipt": contract["prior_failure_receipt"],
        "retired_input_invalidation": contract["retired_input_invalidation_receipt"],
        **{role: contract["frozen_input_bundle"][role] for role in _REPLACEMENT_INPUT_FILENAMES},
    }
    for role, record in receipt_input_records.items():
        if (
            not _same(record["path"], attempt[f"{role}_path"])
            or record["sha256"] != attempt[f"{role}_sha256"]
        ):
            raise ControlError(f"attempt authorization input binding changed for {role}")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _compact_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(encoded: bytes, role: str) -> dict[str, Any]:
    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(
            encoded.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlError(f"{role} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ControlError(f"{role} must be one JSON object")
    return value


def _read_publication_authorization(
    namespace: Namespace,
) -> tuple[dict[str, Any], str]:
    path = _absolute(namespace.publication_authorization)
    expected = _absolute(namespace.control_root) / _PUBLICATION_AUTHORIZATION_FILENAME
    if path != expected:
        raise ControlError("publication authorization receipt path is not canonical")
    try:
        encoded = read_file_anchored(
            path,
            require_single_link=True,
            max_bytes=_MAX_INPUT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(
            "publication authorization receipt failed anchored bounded readback"
        ) from exc
    raw = _strict_json_object(encoded, "publication authorization receipt")
    canonical = _canonical_publication_authorization(raw, namespace=namespace)
    if encoded != _canonical_bytes(canonical):
        raise ControlError("publication authorization receipt bytes are not canonical")
    return canonical, hashlib.sha256(encoded).hexdigest()


def _verify_publication_authorization_live_files(
    namespace: Namespace,
    receipt: Mapping[str, Any],
    *,
    verify_run_state: bool,
) -> None:
    contract = receipt["preflight"]["contract"]
    bundle = contract["frozen_input_bundle"]
    try:
        observed_children = tuple(
            sorted(entry.name for entry in os.scandir(Path(bundle["directory"])))
        )
    except OSError as exc:
        raise ControlError("publication authorization frozen bundle is unavailable") from exc
    if observed_children != tuple(sorted(_REPLACEMENT_INPUT_FILENAMES.values())):
        raise ControlError("publication authorization frozen bundle inventory changed")
    records = [
        contract["controller"],
        contract["failed_preflight_receipt"],
        contract["prior_failure_receipt"],
        contract["retired_input_invalidation_receipt"],
        contract["manifest"],
        *(contract["frozen_input_bundle"][role] for role in _REPLACEMENT_INPUT_FILENAMES),
    ]
    for record in records:
        try:
            encoded = read_file_anchored(
                Path(record["path"]),
                require_single_link=True,
                max_bytes=_MAX_INPUT_BYTES,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ControlError("publication authorization file binding changed") from exc
        if (
            len(encoded) != record["size_bytes"]
            or hashlib.sha256(encoded).hexdigest() != record["sha256"]
        ):
            raise ControlError("publication authorization file identity changed")
    _, invalidation_sha256 = _read_retired_input_invalidation(
        namespace,
        verify_live_run_state=verify_run_state,
    )
    if invalidation_sha256 != contract["retired_input_invalidation_receipt"]["sha256"]:
        raise ControlError("publication authorization v1 invalidation lineage changed")
    config = contract["config"]
    try:
        config_bytes = read_file_anchored(
            Path(config["path"]),
            require_single_link=True,
            max_bytes=_MAX_INPUT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError("publication authorization config binding changed") from exc
    if hashlib.sha256(config_bytes).hexdigest() != config["file_sha256"]:
        raise ControlError("publication authorization config bytes changed")
    if verify_run_state:
        run_state = contract["run_state"]
        current: dict[str, str] = {}
        for name in _RUN_STATE_FILENAMES:
            try:
                encoded = read_file_anchored(
                    Path(run_state["root"]) / name,
                    require_single_link=True,
                    max_bytes=_MAX_INPUT_BYTES,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ControlError(
                    f"publication authorization run-state binding changed: {name}"
                ) from exc
            current[name] = hashlib.sha256(encoded).hexdigest()
        if current != run_state["files"] or _compact_sha256(current) != run_state["sha256"]:
            raise ControlError("publication authorization run-state files changed")


def _read_marker(path: Path, kind: str) -> tuple[dict[str, Any], str]:
    try:
        encoded = read_file_anchored(path, max_bytes=_MAX_BYTES)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{kind} marker failed anchored readback") from exc
    payload = _strict_json_object(encoded, f"{kind} marker")
    return _record(payload, kind), hashlib.sha256(encoded).hexdigest()


def write_marker(path: Path, payload: Mapping[str, Any]) -> str:
    """O_EXCL-write one marker; never replace or retry."""

    _real_directory(path.parent, "marker parent")
    encoded = _canonical_bytes(payload)
    published = publish_bytes_no_overwrite(encoded, path)
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    if published.sha256 != expected_sha256:
        raise ControlError("published marker digest differs from its canonical payload")
    return expected_sha256


def _record(value: object, kind: str) -> dict[str, Any]:
    fields, policy, status = {
        "attempt": (_ATTEMPT_FIELDS, ATTEMPT_POLICY, "claimed"),
        "success": (_SUCCESS_FIELDS, SUCCESS_POLICY, "committed"),
        "failure": (_FAILURE_FIELDS, FAILURE_POLICY, "rolled_back_failure_no_retry"),
    }[kind]
    payload = _exact_dict(value, fields, f"{kind} marker")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["policy"] != policy
        or payload["status"] != status
        or payload["automatic_retry_allowed"] is not False
    ):
        raise ControlError(f"{kind} marker fixed policy is invalid")
    for field_name, item in payload.items():
        if field_name.endswith("_sha256") or field_name == "attempt_id":
            _sha(item, f"{kind} {field_name}")
    path_fields = {
        "attempt": (
            "intended_authority_directory",
            "parent_authority_directory",
            "controller_path",
            "failed_preflight_receipt_path",
            "prior_failure_receipt_path",
            "retired_input_invalidation_path",
            "frozen_source_receipt_path",
            "source_allowlist_path",
            "workspace_plan_path",
            "cnn_correction_receipt_path",
            "publication_authorization_receipt_path",
        ),
        "success": ("authority_directory", "parent_authority_directory"),
        "failure": ("intended_authority_directory", "parent_authority_directory"),
    }[kind]
    if any(not Path(str(payload[field])).is_absolute() for field in path_fields):
        raise ControlError(f"{kind} marker paths must be absolute")
    if kind == "attempt" and (
        type(payload["controller_size_bytes"]) is not int or payload["controller_size_bytes"] <= 0
    ):
        raise ControlError("attempt controller size must be a positive exact integer")
    if kind == "attempt":
        _canonical_timestamp(
            payload["amendment_timestamp_utc"],
            "attempt amendment timestamp",
        )
        if type(payload["max_attempt_count"]) is not int or payload["max_attempt_count"] != 1:
            raise ControlError("attempt must bind exactly one authorized attempt")
    if kind == "success":
        _sha(payload["verification_nonce"], "success verification nonce")
    if kind == "success" and (
        type(payload["controller_process_id"]) is not int
        or payload["controller_process_id"] <= 0
        or type(payload["verifier_process_id"]) is not int
        or payload["verifier_process_id"] <= 0
        or type(payload["verifier_parent_process_id"]) is not int
        or payload["verifier_parent_process_id"] != payload["controller_process_id"]
        or payload["verifier_process_id"] == payload["controller_process_id"]
        or type(payload["chain_depth"]) is not int
        or payload["chain_depth"] <= 0
    ):
        raise ControlError("success process/chain fields are not exact")
    if kind == "failure" and payload["authority_absent_after_rollback"] is not True:
        raise ControlError("failure marker must attest exact D absence")
    return payload


def attempt_record(
    *,
    attempt_id: str,
    destination: str | Path,
    parent: str | Path,
    controller_path: str | Path,
    project_root: str | Path,
    frozen_inputs: FrozenInputBindings,
    publication_authorization_receipt: str | Path,
    authorization_sha256: str,
    intent_sha256: str,
) -> dict[str, Any]:
    namespace = Namespace.for_project(project_root)
    project, parent_path, destination_path = _require_project_layout(
        namespace,
        parent=parent,
        destination=destination,
        verifier_project_root=project_root,
    )
    if not _same(project, project_root) or destination_path is None:
        raise ControlError("attempt project layout is internally inconsistent")
    controller = _absolute(controller_path)
    try:
        controller_bytes = read_file_anchored(controller, max_bytes=_MAX_INPUT_BYTES)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError("attempt controller failed anchored capture") from exc
    input_paths, input_hashes, run_state_sha256 = _capture_attempt_inputs(
        namespace,
        frozen_inputs,
    )
    if run_state_sha256 is None:  # pragma: no cover - defensive internal invariant
        raise ControlError("attempt capture omitted exact run-state evidence")
    receipt_path = _absolute(publication_authorization_receipt)
    if receipt_path != namespace.publication_authorization:
        raise ControlError("attempt publication authorization receipt path is not canonical")
    receipt, receipt_sha256 = _read_publication_authorization(namespace)
    _verify_publication_authorization_live_files(
        namespace,
        receipt,
        verify_run_state=True,
    )
    publication = receipt["publication"]
    preflight = receipt["preflight"]
    contract = preflight["contract"]
    technical = contract["technical_successor"]
    if (
        receipt["authorized_attempt_id"] != attempt_id
        or not _same(publication["intended_authority_directory"], destination_path)
        or not _same(publication["parent_authority_directory"], parent_path)
        or not _same(contract["project_root"], project)
        or not _same(controller, contract["controller"]["path"])
        or len(controller_bytes) != contract["controller"]["size_bytes"]
        or hashlib.sha256(controller_bytes).hexdigest() != contract["controller"]["sha256"]
        or technical["authorization_sha256"] != authorization_sha256
        or technical["intent_sha256"] != intent_sha256
        or contract["run_state"]["sha256"] != run_state_sha256
    ):
        raise ControlError("attempt differs from its one-shot publication authorization receipt")
    receipt_input_records = {
        "failed_preflight_receipt": contract["failed_preflight_receipt"],
        "prior_failure_receipt": contract["prior_failure_receipt"],
        "retired_input_invalidation": contract["retired_input_invalidation_receipt"],
        **{role: contract["frozen_input_bundle"][role] for role in _REPLACEMENT_INPUT_FILENAMES},
    }
    for role, record in receipt_input_records.items():
        if not _same(record["path"], input_paths[role]) or record["sha256"] != input_hashes[role]:
            raise ControlError(f"attempt input differs from publication authorization for {role}")
    return _record(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": ATTEMPT_POLICY,
            "status": "claimed",
            "automatic_retry_allowed": False,
            "attempt_id": attempt_id,
            "intended_authority_directory": str(destination_path),
            "parent_authority_directory": str(parent_path),
            "controller_path": str(controller),
            "controller_size_bytes": len(controller_bytes),
            "controller_sha256": hashlib.sha256(controller_bytes).hexdigest(),
            **{f"{role}_path": str(path) for role, path in input_paths.items()},
            **{f"{role}_sha256": digest for role, digest in input_hashes.items()},
            "publication_authorization_receipt_path": str(receipt_path),
            "publication_authorization_receipt_sha256": receipt_sha256,
            "preflight_fingerprint_sha256": preflight["preflight_fingerprint_sha256"],
            "max_attempt_count": 1,
            "amendment_timestamp_utc": publication["amendment_timestamp_utc"],
            "authorization_sha256": authorization_sha256,
            "intent_sha256": intent_sha256,
            "run_state_sha256": run_state_sha256,
        },
        "attempt",
    )


def _candidates(
    paths: Sequence[str | Path],
    *,
    parent: Path,
) -> tuple[Path, ...]:
    result: list[Path] = []
    for raw in paths:
        candidate = _real_directory(raw, "Authority-D candidate")
        if candidate.parent != parent.parent or _same(candidate, parent):
            raise ControlError("Authority-D candidate must be one direct peer of Authority C")
        if any(_same(candidate, existing) for existing in result):
            raise ControlError("duplicate Authority-D candidate")
        result.append(candidate)
    return tuple(sorted(result, key=lambda path: os.path.normcase(str(path))))


def discover_candidates(parent_authority_directory: str | Path) -> tuple[Path, ...]:
    """Independently enumerate every schema-v5-shaped direct peer of C."""

    parent = _real_directory(parent_authority_directory, "Authority C")
    root = _real_directory(parent.parent, "preregistration amendment root")

    def inventory() -> tuple[str, ...]:
        return tuple(
            sorted(
                (entry.name for entry in os.scandir(root)),
                key=str.casefold,
            )
        )

    names_before = inventory()
    candidates: list[Path] = []
    for name in names_before:
        peer = root / name
        if _same(peer, parent):
            continue
        evidence_path = peer / _AMENDMENT_EVIDENCE_FILENAME
        marker_path = peer / _AMENDMENT_MARKER_FILENAME
        evidence_exists = os.path.lexists(evidence_path)
        marker_exists = os.path.lexists(marker_path)
        if not evidence_exists and not marker_exists:
            continue
        peer = _real_directory(peer, "preregistration authority peer")
        if not evidence_exists or not marker_exists:
            raise ControlError("authority-shaped peer has an incomplete evidence/marker pair")
        try:
            evidence = _strict_json_object(
                read_file_anchored(evidence_path, max_bytes=_MAX_BYTES),
                "authority peer evidence",
            )
            read_file_anchored(marker_path, max_bytes=_MAX_BYTES)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ControlError("authority peer failed anchored inventory readback") from exc
        if (
            evidence.get("schema_version") == 5
            or evidence.get("amendment_purpose") == _TECHNICAL_SUCCESSOR_PURPOSE
            or "resource_bounded_technical_successor_authorization" in evidence
        ):
            candidates.append(peer)
    if inventory() != names_before:
        raise ControlError("preregistration amendment peer set changed during discovery")
    return _candidates(candidates, parent=parent)


def _attempt_input_json(
    attempt: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    try:
        encoded = read_file_anchored(
            Path(attempt[f"{role}_path"]),
            max_bytes=_MAX_INPUT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"committed input readback failed for {role}") from exc
    if hashlib.sha256(encoded).hexdigest() != attempt[f"{role}_sha256"]:
        raise ControlError(f"committed input digest changed for {role}")
    return _strict_json_object(encoded, f"committed {role}")


def _require_authorization_input_bindings(
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    failed = authorization.get("failed_preflight")
    prior = authorization.get("prior_publication_failure")
    source_delta = authorization.get("execution_source_delta")
    workspace = authorization.get("resource_input_workspace_plan")
    correction = authorization.get("cnn_provenance_correction")
    if not all(
        isinstance(item, Mapping) for item in (failed, prior, source_delta, workspace, correction)
    ):
        raise ControlError("committed authorization lacks exact frozen-input sections")
    assert isinstance(failed, Mapping)
    assert isinstance(prior, Mapping)
    assert isinstance(source_delta, Mapping)
    assert isinstance(workspace, Mapping)
    assert isinstance(correction, Mapping)
    if (
        failed.get("receipt_sha256") != attempt["failed_preflight_receipt_sha256"]
        or not _same(
            failed.get("receipt_path", ""),
            attempt["failed_preflight_receipt_path"],
        )
        or prior.get("receipt_sha256") != attempt["prior_failure_receipt_sha256"]
        or not _same(
            prior.get("receipt_path", ""),
            attempt["prior_failure_receipt_path"],
        )
    ):
        raise ControlError("committed authorization receipt pins differ from the attempt")
    allowlist = _attempt_input_json(attempt, "source_allowlist")
    records = allowlist.get("records")
    if type(records) is not list:
        raise ControlError("frozen source allowlist lacks its exact record list")
    allowlisted_change_kinds: dict[str, str] = {}
    for record in records:
        if (
            type(record) is not dict
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("change_kind"), str)
            or record["path"] in allowlisted_change_kinds
        ):
            raise ControlError("frozen source allowlist record is ambiguous")
        allowlisted_change_kinds[record["path"]] = record["change_kind"]
    if _compact_sha256(allowlisted_change_kinds) != _compact_sha256(
        source_delta.get("allowlisted_change_kinds")
    ):
        raise ControlError("committed authorization source allowlist differs from frozen input")
    frozen_workspace = _attempt_input_json(attempt, "workspace_plan")
    if _compact_sha256(frozen_workspace) != _compact_sha256(workspace):
        raise ControlError("committed authorization workspace plan differs from frozen input")
    correction_receipt = _attempt_input_json(attempt, "cnn_correction_receipt")
    if _compact_sha256(correction_receipt.get("correction")) != _compact_sha256(correction):
        raise ControlError("committed authorization CNN correction differs from frozen input")
    frozen_source = _attempt_input_json(attempt, "frozen_source_receipt")
    source_bindings = {
        "execution_source_root_sha256": "resource_root_sha256",
        "execution_source_manifest_sha256": "resource_manifest_sha256",
        "execution_source_delta_sha256": "delta_sha256",
    }
    if any(
        frozen_source.get(receipt_field) != source_delta.get(authorization_field)
        for receipt_field, authorization_field in source_bindings.items()
    ):
        raise ControlError("committed authorization source identity differs from frozen receipt")
    if (
        frozen_source.get("retired_input_invalidation_receipt_path")
        != attempt["retired_input_invalidation_path"]
        or frozen_source.get("retired_input_invalidation_receipt_sha256")
        != attempt["retired_input_invalidation_sha256"]
    ):
        raise ControlError("committed authorization lacks exact v1 invalidation lineage")


def _verify_committed_candidate(
    candidate: Path,
    success: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    """Require the committed D to retain all externally pinned identities."""

    from histo_audit.workflows.preregistration_amendment import (
        require_confirmatory_storage_policy,
        require_resource_bounded_technical_successor_authorization,
        resource_bounded_technical_successor_intent_sha256,
        verify_preregistration_amendment,
    )

    generic = verify_preregistration_amendment(candidate)
    if (
        not generic.valid
        or generic.parent_authority_directory != _absolute(success["parent_authority_directory"])
        or generic.artifact_root_sha256 != success["artifact_root_sha256"]
        or generic.sha256_manifest_sha256 != success["sha256_manifest_sha256"]
        or generic.chain_depth != success["chain_depth"]
    ):
        raise ControlError("committed D differs from its generic success pins")
    authorization = require_resource_bounded_technical_successor_authorization(candidate)
    if _compact_sha256(authorization) != success["authorization_sha256"]:
        raise ControlError("committed D authorization differs from its success pin")
    _require_authorization_input_bindings(authorization, attempt)
    evidence = _strict_json_object(
        read_file_anchored(
            candidate / _AMENDMENT_EVIDENCE_FILENAME,
            max_bytes=_MAX_BYTES,
        ),
        "committed D amendment evidence",
    )
    required = {
        "amendment_timestamp_utc",
        "reason",
        "affected_hypotheses",
        "affected_analyses",
        "outcomes_inspected_at_utc",
        "confirmatory_storage_policy",
    }
    if not required.issubset(evidence):
        raise ControlError("committed D lacks fields required for intent readback")
    intent_sha256 = resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=success["parent_authority_directory"],
        amendment_timestamp_utc=evidence["amendment_timestamp_utc"],
        reason=evidence["reason"],
        affected_hypotheses=evidence["affected_hypotheses"],
        affected_analyses=evidence["affected_analyses"],
        outcomes_inspected_at_utc=evidence["outcomes_inspected_at_utc"],
        authorization=authorization,
        confirmatory_storage_policy=require_confirmatory_storage_policy(candidate),
    )
    if intent_sha256 != success["intent_sha256"]:
        raise ControlError("committed D intent differs from its success pin")


def classify(
    namespace: Namespace,
    *,
    parent_authority_directory: str | Path,
    candidate_discoverer: Callable[[str | Path], Sequence[str | Path]] = discover_candidates,
    committed_candidate_verifier: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> Classification:
    """Apply the exact state truth table; every other combination is STOP."""

    try:
        root = _real_directory(namespace.control_root, "replacement control root")
        _, parent, _ = _require_project_layout(
            namespace,
            parent=parent_authority_directory,
        )
        candidates = _candidates(
            candidate_discoverer(parent),
            parent=parent,
        )
        expected_names = {ATTEMPT_FILENAME, SUCCESS_FILENAME, FAILURE_FILENAME}
        expected_casefold = {name.casefold() for name in expected_names}
        authorization_casefold = _PUBLICATION_AUTHORIZATION_FILENAME.casefold()
        replacement_prefix_casefold = _REPLACEMENT_INPUT_PREFIX.casefold()
        allowed_replacement_names = {
            _RETIRED_INPUT_DIRECTORY_NAME,
            _RETIRED_INPUT_INVALIDATION_FILENAME,
            _REPLACEMENT_INPUT_DIRECTORY_NAME,
        }
        extras = sorted(
            entry.name
            for entry in os.scandir(root)
            if (
                entry.name.casefold().startswith(MARKER_PREFIX.casefold())
                and (
                    entry.name not in expected_names
                    or entry.name.casefold() not in expected_casefold
                )
            )
            or (
                entry.name.casefold() == authorization_casefold
                and entry.name != _PUBLICATION_AUTHORIZATION_FILENAME
            )
            or (
                entry.name.casefold().startswith(replacement_prefix_casefold)
                and entry.name not in allowed_replacement_names
            )
        )
        if extras:
            raise ControlError(f"unexpected replacement namespace entries: {extras!r}")
        retired_path = root / _RETIRED_INPUT_DIRECTORY_NAME
        invalidation_path = root / _RETIRED_INPUT_INVALIDATION_FILENAME
        if not os.path.lexists(retired_path) or not os.path.lexists(invalidation_path):
            raise ControlError("retired v1 bundle and its invalidation receipt must both exist")
        _read_retired_input_invalidation(
            namespace,
            verify_live_run_state=False,
        )
        active_path = root / _REPLACEMENT_INPUT_DIRECTORY_NAME
        if os.path.lexists(active_path):
            active_root = _real_directory(active_path, "active replacement input bundle")
            observed_active = tuple(sorted(entry.name for entry in os.scandir(active_root)))
            if observed_active != tuple(sorted(_REPLACEMENT_INPUT_FILENAMES.values())):
                raise ControlError("active replacement input bundle inventory is not exact")
        if os.path.lexists(namespace.publication_authorization):
            authorization, _ = _read_publication_authorization(namespace)
            _verify_publication_authorization_live_files(
                namespace,
                authorization,
                verify_run_state=False,
            )
        readbacks: dict[str, tuple[dict[str, Any], str] | None] = {}
        for kind, path in (
            ("attempt", namespace.attempt),
            ("success", namespace.success),
            ("failure", namespace.failure),
        ):
            readbacks[kind] = _read_marker(path, kind) if os.path.lexists(path) else None
        if readbacks["attempt"] is not None:
            _verify_attempt_inputs(
                namespace,
                readbacks["attempt"][0],
                verify_live_run_state=False,
            )
    except (OSError, TypeError, ValueError, ControlError) as exc:
        return Classification(State.STOP_AMBIGUOUS, f"{type(exc).__name__}: {exc}", ())

    attempt = readbacks["attempt"]
    success = readbacks["success"]
    failure = readbacks["failure"]
    hashes = {
        "attempt_sha256": attempt[1] if attempt else None,
        "success_sha256": success[1] if success else None,
        "failure_sha256": failure[1] if failure else None,
    }
    if not any((attempt, success, failure, candidates)):
        return Classification(State.READY, "no replacement marker and no D exist", (), **hashes)
    if attempt and failure and not success and not candidates:
        a, a_sha = attempt
        f, _ = failure
        destination = _absolute(a["intended_authority_directory"])
        if (
            _same(a["parent_authority_directory"], parent)
            and f["attempt_marker_sha256"] == a_sha
            and f["attempt_id"] == a["attempt_id"]
            and _same(f["intended_authority_directory"], destination)
            and _same(f["parent_authority_directory"], a["parent_authority_directory"])
            and f["run_state_sha256"] == a["run_state_sha256"]
            and not os.path.lexists(destination)
        ):
            return Classification(
                State.ROLLED_BACK_FAILURE,
                "exact A+F exist and D is absent",
                (),
                **hashes,
            )
    if attempt and success and not failure and len(candidates) == 1:
        a, a_sha = attempt
        s, _ = success
        destination = _absolute(a["intended_authority_directory"])
        if (
            _same(a["parent_authority_directory"], parent)
            and s["attempt_marker_sha256"] == a_sha
            and s["attempt_id"] == a["attempt_id"]
            and _same(s["authority_directory"], destination)
            and _same(candidates[0], destination)
            and _same(s["parent_authority_directory"], a["parent_authority_directory"])
            and s["authorization_sha256"] == a["authorization_sha256"]
            and s["intent_sha256"] == a["intent_sha256"]
            and s["run_state_sha256"] == a["run_state_sha256"]
        ):
            try:
                if committed_candidate_verifier is None:
                    _verify_committed_candidate(candidates[0], s, a)
                else:
                    committed_candidate_verifier(candidates[0], s)
            except (OSError, TypeError, ValueError, ControlError) as exc:
                return Classification(
                    State.STOP_AMBIGUOUS,
                    f"committed D failed exact readback: {type(exc).__name__}: {exc}",
                    candidates,
                    **hashes,
                )
            return Classification(
                State.COMMITTED,
                "exact A+S and exactly one matching D exist",
                candidates,
                **hashes,
            )
    return Classification(
        State.STOP_AMBIGUOUS,
        "state is outside READY, exact A+F/no-D, or exact A+S+D",
        candidates,
        **hashes,
    )


def _verified_payload(
    value: object,
    *,
    request: VerifyRequest,
    controller_pid: int,
    child_pid: int,
) -> dict[str, Any]:
    """Validate the exact public schema-v2 verifier output."""

    request = request.checked()
    top_fields = frozenset(
        {
            "status",
            "verification_schema_version",
            "verification_kind",
            "process_boundary",
            "successor_authority",
            "superseded_authority",
            "bundle",
            "confirmatory_storage_policy_sha256",
            "successor_candidate_count",
            "checks",
            "outcome_value_interpretation_performed",
            "scientific_execution_performed",
            "publication_performed",
        }
    )
    payload = _exact_dict(value, top_fields, "fresh verifier payload")
    if (
        payload["status"] != "verified"
        or type(payload["verification_schema_version"]) is not int
        or payload["verification_schema_version"] != 2
        or payload["verification_kind"] != "resource_bounded_technical_successor_fresh_process"
    ):
        raise FreshVerifierError("fresh verifier top-level identity is invalid")
    process = _exact_dict(
        payload["process_boundary"],
        frozenset(
            {
                "controller_process_id",
                "verifier_process_id",
                "verifier_parent_process_id",
                "distinct_processes",
                "direct_child_process",
                "verification_nonce",
            }
        ),
        "process boundary",
    )
    expected_process = {
        "controller_process_id": controller_pid,
        "verifier_process_id": child_pid,
        "verifier_parent_process_id": controller_pid,
        "distinct_processes": True,
        "direct_child_process": True,
        "verification_nonce": request.nonce,
    }
    if (
        _canonical_bytes(process) != _canonical_bytes(expected_process)
        or child_pid == controller_pid
    ):
        raise FreshVerifierError("fresh verifier is not the exact direct child")
    successor = _exact_dict(
        payload["successor_authority"],
        frozenset(
            {
                "directory",
                "schema_version",
                "purpose",
                "chain_depth",
                "artifact_root_sha256",
                "sha256_manifest_sha256",
                "authorization_sha256",
                "intent_sha256",
            }
        ),
        "successor authority",
    )
    expected_successor = {
        "directory": str(request.successor_directory),
        "schema_version": 5,
        "purpose": "resource_bounded_confirmatory_technical_successor",
        "chain_depth": request.chain_depth,
        "artifact_root_sha256": request.artifact_root_sha256,
        "sha256_manifest_sha256": request.manifest_sha256,
        "authorization_sha256": request.authorization_sha256,
        "intent_sha256": request.intent_sha256,
    }
    if _canonical_bytes(successor) != _canonical_bytes(expected_successor):
        raise FreshVerifierError("fresh verifier successor pins differ")
    superseded = _exact_dict(
        payload["superseded_authority"],
        frozenset(
            {
                "directory",
                "schema_version",
                "historically_verified",
                "effective_execution_leaf",
            }
        ),
        "superseded authority",
    )
    expected_superseded = {
        "directory": str(request.parent_directory),
        "schema_version": 4,
        "historically_verified": True,
        "effective_execution_leaf": False,
    }
    if _canonical_bytes(superseded) != _canonical_bytes(expected_superseded):
        raise FreshVerifierError("fresh verifier historical-C pins differ")
    bundle = _exact_dict(
        payload["bundle"],
        frozenset(
            {
                "flat_file_count",
                "manifest_artifact_count",
                "flat_file_inventory_sha256",
                "flat_file_hashes_verified",
            }
        ),
        "successor bundle",
    )
    if (
        type(bundle["flat_file_count"]) is not int
        or bundle["flat_file_count"] != 8
        or type(bundle["manifest_artifact_count"]) is not int
        or bundle["manifest_artifact_count"] != 6
        or bundle["flat_file_hashes_verified"] is not True
    ):
        raise FreshVerifierError("fresh verifier bundle contract differs")
    _sha(bundle["flat_file_inventory_sha256"], "flat inventory")
    _sha(payload["confirmatory_storage_policy_sha256"], "storage policy")
    checks = _exact_dict(payload["checks"], _CHECK_FIELDS, "fresh verifier checks")
    if (
        any(checks[field] is not True for field in _CHECK_FIELDS)
        or type(payload["successor_candidate_count"]) is not int
        or payload["successor_candidate_count"] != 1
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise FreshVerifierError("fresh verifier mandatory checks differ")
    return payload


@dataclass(slots=True)
class _BoundedPipeReader:
    stream: Any
    role: str
    limit: int
    payload: bytearray = field(default_factory=bytearray)
    error: BaseException | None = None
    overflow: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._read,
            name=f"aanca-{self.role}-reader",
            daemon=True,
        )
        self.thread.start()

    def _read(self) -> None:
        try:
            while len(self.payload) <= self.limit:
                remaining_with_sentinel = self.limit + 1 - len(self.payload)
                requested = min(_PIPE_CHUNK_BYTES, remaining_with_sentinel)
                chunk = self.stream.read(requested)
                if type(chunk) is not bytes:
                    raise TypeError(f"{self.role} pipe returned non-bytes data")
                if len(chunk) > requested:
                    raise RuntimeError(f"{self.role} pipe exceeded its bounded read request")
                if not chunk:
                    break
                self.payload.extend(chunk)
                if len(self.payload) > self.limit:
                    self.overflow = True
                    break
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


def _reader_failure(readers: Sequence[_BoundedPipeReader]) -> FreshVerifierError | None:
    for reader in readers:
        if reader.overflow:
            return FreshVerifierError(
                f"fresh verifier {reader.role} exceeded its bounded byte limit"
            )
        if reader.error is not None:
            return FreshVerifierError(f"fresh verifier {reader.role} pipe read failed")
    return None


def _close_and_join_readers(
    readers: Sequence[_BoundedPipeReader],
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    deadline = time.monotonic() + timeout_seconds
    errors: list[str] = []
    for reader in readers:
        thread = reader.thread
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
    for reader in readers:
        thread = reader.thread
        if thread is not None and thread.is_alive():
            try:
                descriptor = reader.stream.fileno()
            except (AttributeError, OSError, ValueError):
                descriptor = None
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread is not None and thread.is_alive():
            errors.append(f"{reader.role} reader did not stop within its bound")
        else:
            try:
                reader.stream.close()
            except (OSError, ValueError) as exc:
                errors.append(f"{reader.role} pipe close failed: {exc}")
    return tuple(errors)


def _terminate_and_reap_bounded(
    process: Any,
    readers: Sequence[_BoundedPipeReader],
) -> tuple[str, ...]:
    errors: list[str] = []
    if getattr(process, "returncode", None) is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(f"terminate failed: {type(exc).__name__}: {exc}")
        try:
            process.wait(timeout=_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as exc:
            errors.append(f"post-terminate wait failed: {type(exc).__name__}: {exc}")
    if getattr(process, "returncode", None) is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(f"kill failed: {type(exc).__name__}: {exc}")
        try:
            process.wait(timeout=_CLEANUP_GRACE_SECONDS)
        except BaseException as exc:
            errors.append(f"final reap failed: {type(exc).__name__}: {exc}")
    errors.extend(
        _close_and_join_readers(
            readers,
            timeout_seconds=_CLEANUP_GRACE_SECONDS,
        )
    )
    if getattr(process, "returncode", None) is None:
        errors.append("fresh verifier child was not reaped")
    return tuple(errors)


def _wait_with_bounded_pipes(
    process: Any,
    readers: Sequence[_BoundedPipeReader],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        failure = _reader_failure(readers)
        if failure is not None:
            raise failure
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FreshVerifierError("fresh verifier timed out")
        try:
            process.wait(timeout=min(_WAIT_SLICE_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        except BaseException as exc:
            raise FreshVerifierError("fresh verifier wait failed") from exc
        break
    remaining = max(0.0, deadline - time.monotonic())
    for reader in readers:
        thread = reader.thread
        if thread is not None:
            thread.join(timeout=remaining)
            remaining = max(0.0, deadline - time.monotonic())
    failure = _reader_failure(readers)
    if failure is not None:
        raise failure
    if any(reader.thread is not None and reader.thread.is_alive() for reader in readers):
        raise FreshVerifierError("fresh verifier pipes did not reach bounded EOF")


def run_fresh_verifier(
    request: VerifyRequest,
    *,
    timeout_seconds: float = 900.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> VerifyResult:
    """Run one shell-free direct child; retain stdout only in memory."""

    request = request.checked()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise FreshVerifierError("timeout must be positive")
    controller_pid = os.getpid()
    argv = request.argv(controller_pid)
    environment = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    popen_arguments: dict[str, Any] = {
        "cwd": str(request.project_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "text": False,
        "close_fds": True,
        "env": environment,
    }
    spawn_executable = _fresh_verifier_spawn_executable(request.python_executable)
    if spawn_executable is not None:
        popen_arguments["executable"] = spawn_executable
    process = popen_factory(list(argv), **popen_arguments)
    readers: tuple[_BoundedPipeReader, ...] = ()
    try:
        if type(process.pid) is not int or process.pid <= 0:
            raise FreshVerifierError("fresh verifier has no positive child PID")
        if process.stdout is None or process.stderr is None:
            raise FreshVerifierError("fresh verifier did not expose both bounded pipes")
        readers = (
            _BoundedPipeReader(process.stdout, "stdout", _MAX_BYTES),
            _BoundedPipeReader(process.stderr, "stderr", _MAX_STDERR_BYTES),
        )
        for reader in readers:
            reader.start()
        _wait_with_bounded_pipes(
            process,
            readers,
            timeout_seconds=float(timeout_seconds),
        )
        close_errors = _close_and_join_readers(
            readers,
            timeout_seconds=_CLEANUP_GRACE_SECONDS,
        )
        if close_errors:
            raise FreshVerifierError(
                "fresh verifier pipe finalization failed: " + "; ".join(close_errors)
            )
        stdout = bytes(readers[0].payload)
        stderr = bytes(readers[1].payload)
    except BaseException as exc:
        cleanup_errors = _terminate_and_reap_bounded(process, readers)
        if cleanup_errors:
            raise FreshVerifierError(
                "fresh verifier failed and bounded child cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        if isinstance(exc, FreshVerifierError):
            raise
        raise FreshVerifierError("fresh verifier process interaction failed") from exc
    if process.returncode != 0 or stderr != b"" or type(stdout) is not bytes:
        raise FreshVerifierError("fresh verifier process did not exit cleanly")
    try:
        raw = _strict_json_object(stdout, "fresh verifier stdout")
    except ControlError as exc:
        raise FreshVerifierError("fresh verifier stdout is not one JSON object") from exc
    payload = _verified_payload(
        raw,
        request=request,
        controller_pid=controller_pid,
        child_pid=process.pid,
    )
    return VerifyResult(
        request=request,
        argv=argv,
        process_id=process.pid,
        payload=payload,
        payload_sha256=hashlib.sha256(stdout).hexdigest(),
    )


class TransactionVerifier:
    """One-use creator callback; no callback log is written."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        parent: str | Path,
        destination: str | Path,
        authorization_sha256: str,
        intent_sha256: str,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(32),
    ) -> None:
        self.project_root = _absolute(project_root)
        self.parent = _absolute(parent)
        self.destination = _absolute(destination)
        self.authorization_sha256 = _sha(authorization_sha256, "authorization")
        self.intent_sha256 = _sha(intent_sha256, "intent")
        self.popen_factory = popen_factory
        self.nonce_factory = nonce_factory
        self.result: VerifyResult | None = None
        self.published_result: Any | None = None
        self.invoked = False

    def __call__(self, published: Any) -> None:
        if self.invoked:
            raise FreshVerifierError("transaction verifier callback ran twice")
        self.invoked = True
        if not _same(published.parent_authority_directory, self.parent):
            raise FreshVerifierError("published parent differs from Authority C")
        if not _same(published.amendment_directory, self.destination):
            raise FreshVerifierError("published successor differs from the intended Authority D")
        request = VerifyRequest(
            project_root=self.project_root,
            successor_directory=published.amendment_directory,
            parent_directory=self.parent,
            artifact_root_sha256=published.artifact_root_sha256,
            manifest_sha256=published.sha256_manifest_sha256,
            authorization_sha256=self.authorization_sha256,
            intent_sha256=self.intent_sha256,
            nonce=self.nonce_factory(),
            chain_depth=published.chain_depth,
        )
        self.result = run_fresh_verifier(request, popen_factory=self.popen_factory)
        self.published_result = published
        return None


def _success_record(
    attempt: Mapping[str, Any],
    attempt_sha: str,
    verifier: VerifyResult,
) -> dict[str, Any]:
    request = verifier.request
    process = verifier.payload.get("process_boundary")
    if not isinstance(process, Mapping):
        raise ControlError("fresh verifier result lacks its process boundary")
    return _record(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": SUCCESS_POLICY,
            "status": "committed",
            "automatic_retry_allowed": False,
            "attempt_id": attempt["attempt_id"],
            "attempt_marker_sha256": attempt_sha,
            "authority_directory": str(request.successor_directory),
            "parent_authority_directory": str(request.parent_directory),
            "artifact_root_sha256": request.artifact_root_sha256,
            "sha256_manifest_sha256": request.manifest_sha256,
            "authorization_sha256": request.authorization_sha256,
            "intent_sha256": request.intent_sha256,
            "verification_nonce": request.nonce,
            "fresh_verifier_payload_sha256": verifier.payload_sha256,
            "controller_process_id": process.get("controller_process_id"),
            "verifier_process_id": process.get("verifier_process_id"),
            "verifier_parent_process_id": process.get("verifier_parent_process_id"),
            "chain_depth": request.chain_depth,
            "run_state_sha256": attempt["run_state_sha256"],
        },
        "success",
    )


def _failure_record(
    attempt: Mapping[str, Any],
    attempt_sha: str,
    error: BaseException,
) -> dict[str, Any]:
    error_type = type(error).__name__
    return _record(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": FAILURE_POLICY,
            "status": "rolled_back_failure_no_retry",
            "automatic_retry_allowed": False,
            "attempt_id": attempt["attempt_id"],
            "attempt_marker_sha256": attempt_sha,
            "intended_authority_directory": attempt["intended_authority_directory"],
            "parent_authority_directory": attempt["parent_authority_directory"],
            "error_type_sha256": hashlib.sha256(error_type.encode()).hexdigest(),
            "error_sha256": hashlib.sha256(f"{error_type}: {error}".encode()).hexdigest(),
            "authority_absent_after_rollback": True,
            "run_state_sha256": attempt["run_state_sha256"],
        },
        "failure",
    )


def _require_exact_transaction_result(
    value: Any,
    verifier: TransactionVerifier,
) -> None:
    from histo_audit.workflows.preregistration_amendment import (
        PreregistrationAmendmentResult,
    )

    if (
        verifier.result is None
        or verifier.published_result is None
        or value is not verifier.published_result
        or type(value) is not PreregistrationAmendmentResult
    ):
        raise AmbiguousStateError(
            "creator did not return the exact transaction-verified amendment result"
        )
    request = verifier.result.request
    expected_paths = {
        "amendment_directory": request.successor_directory,
        "parent_authority_directory": request.parent_directory,
        "amendment_evidence_path": request.successor_directory / _AMENDMENT_EVIDENCE_FILENAME,
        "amended_preregistration_path": request.successor_directory / "PRE_REGISTRATION_FROZEN.md",
        "amended_primary_config_path": request.successor_directory / "primary_frozen.yaml",
        "amended_confirmatory_config_path": request.successor_directory
        / "confirmatory_frozen.yaml",
        "source_tree_manifest_path": request.successor_directory / "source_tree_manifest.json",
        "sha256_manifest_path": request.successor_directory / "sha256_manifest.json",
        "immutable_marker_path": request.successor_directory / _AMENDMENT_MARKER_FILENAME,
    }
    if any(not _same(getattr(value, field), path) for field, path in expected_paths.items()):
        raise AmbiguousStateError("creator result paths differ from fresh-verifier pins")
    if (
        type(value.chain_depth) is not int
        or value.chain_depth != request.chain_depth
        or value.artifact_root_sha256 != request.artifact_root_sha256
        or value.sha256_manifest_sha256 != request.manifest_sha256
    ):
        raise AmbiguousStateError("creator result identities differ from fresh-verifier pins")


def execute_once(
    *,
    namespace: Namespace,
    attempt: Mapping[str, Any],
    transaction: Callable[[Callable[[Any], None]], Any],
    verifier: TransactionVerifier,
    candidate_discoverer: (Callable[[str | Path], Sequence[str | Path]] | None) = None,
    committed_candidate_verifier: (Callable[[Path, Mapping[str, Any]], None] | None) = None,
    preclaim_check: Callable[[], None] | None = None,
) -> PublicationResult:
    """Generic one-shot harness; success marker is the final mutation.

    The creator dependency must call ``verifier`` from its rollback scope and return
    that exact typed result.  After the protocol lock is cleanly released, final
    input/candidate readback precedes the terminal O_EXCL marker.  Failure is recorded
    only if the creator did not return and a complete scan proves D absent.
    """

    attempt = _record(attempt, "attempt")
    project_root, parent, destination = _require_project_layout(
        namespace,
        parent=attempt["parent_authority_directory"],
        destination=attempt["intended_authority_directory"],
        verifier_project_root=verifier.project_root,
    )
    if destination is None:
        raise AmbiguousStateError("attempt lacks an Authority-D destination")
    active_input_root = _replacement_input_directory(
        namespace.control_root,
        Path(attempt["frozen_source_receipt_path"]).parent,
    )
    discovery = candidate_discoverer or discover_candidates
    with ExclusiveBundlePublicationLock(
        (
            active_input_root,
            namespace.attempt,
            namespace.success,
            namespace.failure,
            namespace.publication_authorization,
        ),
        role="resource Authority-D replacement protocol",
    ):
        if verifier.invoked or verifier.result is not None:
            raise AmbiguousStateError(
                "transaction verifier must be fresh before the one-shot attempt"
            )
        if (
            not _same(verifier.parent, parent)
            or not _same(verifier.destination, destination)
            or verifier.authorization_sha256 != attempt["authorization_sha256"]
            or verifier.intent_sha256 != attempt["intent_sha256"]
        ):
            raise AmbiguousStateError(
                "transaction verifier differs from the exact attempt bindings"
            )
        if not _same(project_root, verifier.project_root):
            raise AmbiguousStateError("transaction verifier is outside the bound project")
        controller_path = _absolute(attempt["controller_path"])
        try:
            controller_bytes = read_file_anchored(
                controller_path,
                max_bytes=attempt["controller_size_bytes"],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise AmbiguousStateError(
                "attempt controller failed anchored identity readback"
            ) from exc
        if (
            not _same(controller_path, Path(__file__))
            or len(controller_bytes) != attempt["controller_size_bytes"]
            or hashlib.sha256(controller_bytes).hexdigest() != attempt["controller_sha256"]
        ):
            raise AmbiguousStateError("attempt controller differs from the executing source")
        initial = classify(
            namespace,
            parent_authority_directory=parent,
            candidate_discoverer=discovery,
            committed_candidate_verifier=committed_candidate_verifier,
        )
        if initial.state is not State.READY:
            raise AmbiguousStateError(f"initial state is {initial.state.value}, not READY")
        _verify_attempt_inputs(namespace, attempt)
        if preclaim_check is not None:
            if preclaim_check() is not None:
                raise AmbiguousStateError("preclaim check must return exactly None")
            repeated = classify(
                namespace,
                parent_authority_directory=parent,
                candidate_discoverer=discovery,
                committed_candidate_verifier=committed_candidate_verifier,
            )
            if repeated.state is not State.READY:
                raise AmbiguousStateError(
                    f"state changed during preclaim check: {repeated.state.value}"
                )
            _verify_attempt_inputs(namespace, attempt)
        attempt_sha = write_marker(namespace.attempt, attempt)
    try:
        transaction_result = transaction(verifier)
    except BaseException as exc:
        try:
            candidates = _candidates(
                discovery(parent),
                parent=parent,
            )
            destination_exists = os.path.lexists(destination)
            _verify_attempt_inputs(namespace, attempt)
        except BaseException as scan_error:
            raise AmbiguousStateError("post-rollback candidate scan failed") from scan_error
        if candidates or destination_exists:
            raise AmbiguousStateError("creator failed but D is present") from exc
        failure = _failure_record(attempt, attempt_sha, exc)
        failure_sha = write_marker(namespace.failure, failure)
        return PublicationResult(
            State.ROLLED_BACK_FAILURE,
            namespace.failure,
            failure_sha,
            None,
        )
    if not verifier.invoked or verifier.result is None:
        raise AmbiguousStateError("creator returned without transaction-scoped verification")
    _require_exact_transaction_result(transaction_result, verifier)
    _verify_attempt_inputs(namespace, attempt)
    candidates = _candidates(discovery(parent), parent=parent)
    if candidates != (destination,):
        raise AmbiguousStateError("creator returned without exactly one intended Authority D")
    success = _success_record(attempt, attempt_sha, verifier.result)
    if committed_candidate_verifier is None:
        _verify_committed_candidate(destination, success, attempt)
    else:
        committed_candidate_verifier(destination, success)
    request = verifier.result.request
    success_sha = write_marker(namespace.success, success)
    return PublicationResult(
        State.COMMITTED,
        namespace.success,
        success_sha,
        request.successor_directory,
    )


def _live_paths(
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> dict[str, Path]:
    project_root, parent, _ = _require_project_layout(
        namespace,
        parent=parent_authority_directory,
    )
    if parent.name != _AUTHORITY_C_COMPONENT:
        raise ControlError("live replacement must use the exact historical Authority C")
    return {
        "project_root": project_root,
        "parent": parent,
        "amendment_root": parent.parent,
        "control_root": _absolute(namespace.control_root),
        "run_root": project_root / "artifacts" / "runs",
        "config": project_root / "configs" / "confirmatory_resource_bounded_amended.yaml",
        "manifest": project_root
        / "data"
        / "manifests"
        / "pannuke"
        / "pannuke_nucleus_manifest.parquet",
        "failed_preflight": _absolute(namespace.control_root) / _FAILED_PREFLIGHT_FILENAME,
        "prior_failure": _absolute(namespace.control_root) / _PRIOR_FAILURE_RECEIPT_FILENAME,
        "retired_input": _absolute(namespace.control_root) / _RETIRED_INPUT_DIRECTORY_NAME,
        "retired_input_invalidation": (
            _absolute(namespace.control_root) / _RETIRED_INPUT_INVALIDATION_FILENAME
        ),
    }


def _live_file_record(
    path: str | Path,
    role: str,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    candidate = _absolute(path)
    try:
        encoded = read_file_anchored(
            candidate,
            require_single_link=True,
            max_bytes=_MAX_INPUT_BYTES,
            allow_empty=allow_empty,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{role} failed bounded anchored readback") from exc
    return {
        "path": str(candidate),
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _live_json(path: str | Path, role: str) -> dict[str, Any]:
    record = _live_file_record(path, role)
    try:
        encoded = read_file_anchored(
            Path(record["path"]),
            require_single_link=True,
            max_bytes=record["size_bytes"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{role} changed during anchored readback") from exc
    if hashlib.sha256(encoded).hexdigest() != record["sha256"]:
        raise ControlError(f"{role} changed during anchored readback")
    return _strict_json_object(encoded, role)


def _live_run_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    run_root = _real_directory(paths["run_root"], "run-state registry root")
    files = {
        name: _live_file_record(run_root / name, f"run-state {name}")["sha256"]
        for name in _RUN_STATE_FILENAMES
    }
    return {
        "root": str(run_root),
        "files": files,
        "sha256": _compact_sha256(files),
    }


def _retired_input_invalidation_path(namespace: Namespace) -> Path:
    return _absolute(namespace.control_root) / _RETIRED_INPUT_INVALIDATION_FILENAME


def _live_canonical_json_record(
    path: str | Path,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _live_file_record(path, role)
    try:
        encoded = read_file_anchored(
            Path(record["path"]),
            require_single_link=True,
            max_bytes=record["size_bytes"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{role} changed during canonical readback") from exc
    if (
        len(encoded) != record["size_bytes"]
        or hashlib.sha256(encoded).hexdigest() != record["sha256"]
    ):
        raise ControlError(f"{role} changed during canonical readback")
    payload = _strict_json_object(encoded, role)
    if encoded != _canonical_bytes(payload):
        raise ControlError(f"{role} bytes are not canonical")
    return payload, record


def _live_pinned_json_record(
    path: str | Path,
    role: str,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly parse an exact-pinned historical JSON file without newline rewriting."""

    record = _live_file_record(path, role)
    if record["size_bytes"] != expected_size_bytes or record["sha256"] != expected_sha256:
        raise ControlError(f"{role} differs from its exact pin")
    try:
        encoded = read_file_anchored(
            Path(record["path"]),
            require_single_link=True,
            max_bytes=record["size_bytes"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{role} changed during strict readback") from exc
    if (
        len(encoded) != record["size_bytes"]
        or hashlib.sha256(encoded).hexdigest() != record["sha256"]
    ):
        raise ControlError(f"{role} changed during strict readback")
    return _strict_json_object(encoded, role), record


def _retired_bundle_snapshot(namespace: Namespace) -> dict[str, Any]:
    control_root = _absolute(namespace.control_root)
    root = _real_directory(
        control_root / _RETIRED_INPUT_DIRECTORY_NAME,
        "retired replacement input bundle v1",
    )
    if root.parent != control_root or root.name != _RETIRED_INPUT_DIRECTORY_NAME:
        raise ControlError("retired replacement input bundle path is not canonical")
    observed_children = tuple(sorted(entry.name for entry in os.scandir(root)))
    expected_children = tuple(sorted(_REPLACEMENT_INPUT_FILENAMES.values()))
    if observed_children != expected_children:
        raise ControlError("retired replacement input bundle inventory changed")
    files: dict[str, dict[str, Any]] = {}
    for role, filename in _REPLACEMENT_INPUT_FILENAMES.items():
        record = _live_file_record(root / filename, f"retired v1 {role}")
        expected = _RETIRED_INPUT_FILE_PINS[role]
        if record["size_bytes"] != expected["size_bytes"] or record["sha256"] != expected["sha256"]:
            raise ControlError(f"retired replacement input changed for {role}")
        files[role] = record
    source_receipt, source_record = _live_canonical_json_record(
        root / _REPLACEMENT_INPUT_FILENAMES["frozen_source_receipt"],
        "retired v1 frozen-source receipt",
    )
    if source_record != files["frozen_source_receipt"]:
        raise ControlError("retired frozen-source receipt changed during readback")
    expected_source = {
        "policy": "resource_authority_d_replacement_frozen_source_receipt_v1",
        "source_allowlist_sha256": files["source_allowlist"]["sha256"],
        "workspace_plan_sha256": files["workspace_plan"]["sha256"],
        "cnn_correction_receipt_sha256": files["cnn_correction_receipt"]["sha256"],
        "controller_path": str(_absolute(Path(__file__))),
        "controller_size_bytes": _RETIRED_CONTROLLER_SIZE_BYTES,
        "controller_sha256": _RETIRED_CONTROLLER_SHA256,
        "execution_source_root_sha256": _RETIRED_SOURCE_ROOT_SHA256,
        "execution_source_manifest_sha256": _RETIRED_SOURCE_MANIFEST_SHA256,
        "execution_source_delta_sha256": _RETIRED_SOURCE_DELTA_SHA256,
        "authorization_sha256": _RETIRED_AUTHORIZATION_SHA256,
        "failed_preflight_receipt_path": str(control_root / _FAILED_PREFLIGHT_FILENAME),
        "failed_preflight_receipt_sha256": _FAILED_PREFLIGHT_SHA256,
        "prior_failure_receipt_path": str(control_root / _PRIOR_FAILURE_RECEIPT_FILENAME),
        "prior_failure_receipt_sha256": _PRIOR_FAILURE_RECEIPT_SHA256,
        "run_state_sha256": _RETIRED_RUN_STATE_SHA256,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    if any(source_receipt.get(field) != expected for field, expected in expected_source.items()):
        raise ControlError("retired frozen-source receipt no longer binds exact v1 evidence")
    return {
        "directory": str(root),
        "files": {role: files[role] for role in sorted(files)},
        "records_sha256": _compact_sha256({role: files[role] for role in sorted(files)}),
        "controller_path": str(_absolute(Path(__file__))),
        "controller_size_bytes": _RETIRED_CONTROLLER_SIZE_BYTES,
        "controller_sha256": _RETIRED_CONTROLLER_SHA256,
        "execution_source_root_sha256": _RETIRED_SOURCE_ROOT_SHA256,
        "execution_source_manifest_sha256": _RETIRED_SOURCE_MANIFEST_SHA256,
        "execution_source_delta_sha256": _RETIRED_SOURCE_DELTA_SHA256,
        "authorization_sha256": _RETIRED_AUTHORIZATION_SHA256,
    }


def _retired_failure_evidence_snapshot(namespace: Namespace) -> dict[str, Any]:
    control_root = _absolute(namespace.control_root)
    receipt_path = control_root / _FAILED_PREFLIGHT_FILENAME
    receipt, receipt_record = _live_canonical_json_record(
        receipt_path,
        "historical failed-preflight receipt",
    )
    if (
        receipt_record["size_bytes"] != _FAILED_PREFLIGHT_SIZE_BYTES
        or receipt_record["sha256"] != _FAILED_PREFLIGHT_SHA256
        or receipt.get("observed_at_utc") != _HISTORICAL_FAILED_AT_UTC
    ):
        raise ControlError("historical failed-preflight receipt differs from its exact pin")
    _canonical_historical_failed_timestamp(
        receipt["observed_at_utc"],
        "historical failed-preflight observation",
    )
    logs: dict[str, dict[str, Any]] = {}
    for role, pin in _RETIRED_LOG_PINS.items():
        filename = pin["filename"]
        if not isinstance(filename, str):  # pragma: no cover - static constant
            raise RuntimeError("retired log filename constant is invalid")
        record = _live_file_record(
            control_root / filename,
            f"retired {role} log",
            allow_empty=True,
        )
        if record["size_bytes"] != pin["size_bytes"] or record["sha256"] != pin["sha256"]:
            raise ControlError(f"retired {role} log differs from its exact pin")
        logs[role] = record
    preflight_pin = _RETIRED_LOG_PINS["preflight_stdout"]
    preflight_filename = preflight_pin["filename"]
    preflight_size = preflight_pin["size_bytes"]
    preflight_sha256 = preflight_pin["sha256"]
    if (
        not isinstance(preflight_filename, str)
        or type(preflight_size) is not int
        or not isinstance(preflight_sha256, str)
    ):  # pragma: no cover - static constants
        raise RuntimeError("preflight log pin constants are invalid")
    preflight_payload, preflight_record = _live_pinned_json_record(
        control_root / preflight_filename,
        "failed replacement preflight stdout",
        expected_size_bytes=preflight_size,
        expected_sha256=preflight_sha256,
    )
    if (
        preflight_record != logs["preflight_stdout"]
        or preflight_payload.get("status") != "stopped_without_write"
        or preflight_payload.get("replacement_state") != State.READY.value
        or preflight_payload.get("automatic_retry_allowed") is not False
        or preflight_payload.get("publication_performed") is not False
        or preflight_payload.get("error_sha256") != _RETIRED_FAILURE_ERROR_SHA256
    ):
        raise ControlError("failed replacement preflight log does not prove no-write failure")
    parsed = _canonical_historical_failed_timestamp(
        _HISTORICAL_FAILED_AT_UTC,
        "historical failed-preflight observation",
    )
    return {
        "receipt": receipt_record,
        "stored_observed_at_utc": _HISTORICAL_FAILED_AT_UTC,
        "normalized_observed_at_utc": parsed.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "logs": {role: logs[role] for role in sorted(logs)},
        "error_type": "ControlError",
        "error_message": _RETIRED_FAILURE_ERROR_MESSAGE,
        "error_sha256": _RETIRED_FAILURE_ERROR_SHA256,
        "publication_authorization_created": False,
        "attempt_marker_created": False,
        "authority_d_created": False,
        "scientific_run_started": False,
    }


def _canonical_retired_input_invalidation(
    value: object,
    *,
    namespace: Namespace,
) -> dict[str, Any]:
    payload = _exact_dict(
        value,
        _RETIRED_INVALIDATION_FIELDS,
        "retired input invalidation receipt",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["policy"] != _RETIRED_INPUT_INVALIDATION_POLICY
        or payload["status"] != "preserved_invalid_nonpublishable"
        or payload["reason_code"] != _RETIRED_INVALIDATION_REASON_CODE
        or payload["reason"] != _RETIRED_INVALIDATION_REASON
        or payload["outcome_value_interpretation_performed"] is not False
        or payload["scientific_execution_performed"] is not False
        or payload["publication_performed"] is not False
    ):
        raise ControlError("retired input invalidation fixed policy is invalid")
    _canonical_timestamp(
        payload["invalidated_at_utc"],
        "retired input invalidation time",
    )
    control_root = _absolute(namespace.control_root)
    retired = _exact_dict(
        payload["retired_bundle"],
        _RETIRED_BUNDLE_FIELDS,
        "retired replacement bundle",
    )
    retired_root = _absolute(retired["directory"])
    if (
        not Path(str(retired["directory"])).is_absolute()
        or retired_root != control_root / _RETIRED_INPUT_DIRECTORY_NAME
    ):
        raise ControlError("retired replacement bundle path is not canonical")
    raw_files = retired["files"]
    if type(raw_files) is not dict or tuple(sorted(raw_files)) != tuple(
        sorted(_REPLACEMENT_INPUT_FILENAMES)
    ):
        raise ControlError("retired replacement bundle receipt has the wrong file roles")
    canonical_files: dict[str, dict[str, Any]] = {}
    for role, filename in _REPLACEMENT_INPUT_FILENAMES.items():
        record = _canonical_file_record(raw_files[role], f"retired v1 {role}")
        pin = _RETIRED_INPUT_FILE_PINS[role]
        if (
            Path(record["path"]) != retired_root / filename
            or record["size_bytes"] != pin["size_bytes"]
            or record["sha256"] != pin["sha256"]
        ):
            raise ControlError(f"retired replacement receipt pin differs for {role}")
        canonical_files[role] = record
    canonical_files = {role: canonical_files[role] for role in sorted(canonical_files)}
    if (
        retired["records_sha256"] != _compact_sha256(canonical_files)
        or not _same(retired["controller_path"], Path(__file__))
        or retired["controller_size_bytes"] != _RETIRED_CONTROLLER_SIZE_BYTES
        or retired["controller_sha256"] != _RETIRED_CONTROLLER_SHA256
        or retired["execution_source_root_sha256"] != _RETIRED_SOURCE_ROOT_SHA256
        or retired["execution_source_manifest_sha256"] != _RETIRED_SOURCE_MANIFEST_SHA256
        or retired["execution_source_delta_sha256"] != _RETIRED_SOURCE_DELTA_SHA256
        or retired["authorization_sha256"] != _RETIRED_AUTHORIZATION_SHA256
    ):
        raise ControlError("retired replacement bundle identity differs from exact v1")
    prior = _canonical_file_record(
        payload["prior_failure_receipt"],
        "prior-publication failure receipt",
    )
    if (
        Path(prior["path"]) != control_root / _PRIOR_FAILURE_RECEIPT_FILENAME
        or prior["size_bytes"] != _PRIOR_FAILURE_RECEIPT_SIZE_BYTES
        or prior["sha256"] != _PRIOR_FAILURE_RECEIPT_SHA256
    ):
        raise ControlError("prior-publication failure receipt pin changed")
    failed = _exact_dict(
        payload["failed_preflight_evidence"],
        _FAILED_PREFLIGHT_EVIDENCE_FIELDS,
        "failed-preflight invalidation evidence",
    )
    failed_receipt = _canonical_file_record(
        failed["receipt"],
        "historical failed-preflight receipt",
    )
    if (
        Path(failed_receipt["path"]) != control_root / _FAILED_PREFLIGHT_FILENAME
        or failed_receipt["size_bytes"] != _FAILED_PREFLIGHT_SIZE_BYTES
        or failed_receipt["sha256"] != _FAILED_PREFLIGHT_SHA256
        or failed["stored_observed_at_utc"] != _HISTORICAL_FAILED_AT_UTC
        or failed["normalized_observed_at_utc"] != "2026-07-27T17:30:54.689000Z"
        or failed["error_type"] != "ControlError"
        or failed["error_message"] != _RETIRED_FAILURE_ERROR_MESSAGE
        or failed["error_sha256"] != _RETIRED_FAILURE_ERROR_SHA256
        or any(
            failed[field] is not False
            for field in (
                "publication_authorization_created",
                "attempt_marker_created",
                "authority_d_created",
                "scientific_run_started",
            )
        )
    ):
        raise ControlError("failed-preflight invalidation evidence is not exact")
    _canonical_historical_failed_timestamp(
        failed["stored_observed_at_utc"],
        "stored historical failed-preflight observation",
    )
    logs = failed["logs"]
    if type(logs) is not dict or tuple(sorted(logs)) != tuple(sorted(_RETIRED_LOG_PINS)):
        raise ControlError("retired failure log inventory is not exact")
    canonical_logs: dict[str, dict[str, Any]] = {}
    for role, pin in _RETIRED_LOG_PINS.items():
        log_filename = pin["filename"]
        if not isinstance(log_filename, str):  # pragma: no cover - static constant
            raise RuntimeError("retired log filename constant is invalid")
        record = _canonical_file_record(
            logs[role],
            f"retired {role} log",
            allow_empty=True,
        )
        if (
            Path(record["path"]) != control_root / log_filename
            or record["size_bytes"] != pin["size_bytes"]
            or record["sha256"] != pin["sha256"]
        ):
            raise ControlError(f"retired failure log pin differs for {role}")
        canonical_logs[role] = record
    canonical_logs = {role: canonical_logs[role] for role in sorted(canonical_logs)}
    corrected = _canonical_file_record(
        payload["corrected_controller"],
        "corrected replacement controller",
    )
    if Path(corrected["path"]) != _absolute(Path(__file__)):
        raise ControlError("corrected replacement controller path is not canonical")
    run_state_sha256 = _sha(payload["run_state_sha256"], "retired invalidation run-state")
    if run_state_sha256 != _RETIRED_RUN_STATE_SHA256:
        raise ControlError("retired invalidation run-state differs from frozen v1 lineage")
    disposition = _exact_dict(
        payload["disposition"],
        _RETIRED_DISPOSITION_FIELDS,
        "retired input disposition",
    )
    expected_disposition = {
        "v1_preflight_allowed": False,
        "v1_authorization_allowed": False,
        "v1_publication_allowed": False,
        "v1_may_be_modified_moved_or_deleted": False,
        "prior_failure_receipt_may_be_modified_or_republished": False,
        "replacement_requires_exact_v2_singleton": True,
    }
    if disposition != expected_disposition:
        raise ControlError("retired input disposition is not fail-closed")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": _RETIRED_INPUT_INVALIDATION_POLICY,
        "status": "preserved_invalid_nonpublishable",
        "invalidated_at_utc": payload["invalidated_at_utc"],
        "reason_code": _RETIRED_INVALIDATION_REASON_CODE,
        "reason": _RETIRED_INVALIDATION_REASON,
        "retired_bundle": {
            "directory": str(retired_root),
            "files": canonical_files,
            "records_sha256": retired["records_sha256"],
            "controller_path": str(_absolute(retired["controller_path"])),
            "controller_size_bytes": _RETIRED_CONTROLLER_SIZE_BYTES,
            "controller_sha256": _RETIRED_CONTROLLER_SHA256,
            "execution_source_root_sha256": _RETIRED_SOURCE_ROOT_SHA256,
            "execution_source_manifest_sha256": _RETIRED_SOURCE_MANIFEST_SHA256,
            "execution_source_delta_sha256": _RETIRED_SOURCE_DELTA_SHA256,
            "authorization_sha256": _RETIRED_AUTHORIZATION_SHA256,
        },
        "prior_failure_receipt": prior,
        "failed_preflight_evidence": {
            "receipt": failed_receipt,
            "stored_observed_at_utc": _HISTORICAL_FAILED_AT_UTC,
            "normalized_observed_at_utc": "2026-07-27T17:30:54.689000Z",
            "logs": canonical_logs,
            "error_type": "ControlError",
            "error_message": _RETIRED_FAILURE_ERROR_MESSAGE,
            "error_sha256": _RETIRED_FAILURE_ERROR_SHA256,
            "publication_authorization_created": False,
            "attempt_marker_created": False,
            "authority_d_created": False,
            "scientific_run_started": False,
        },
        "corrected_controller": corrected,
        "run_state_sha256": run_state_sha256,
        "disposition": expected_disposition,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }


def _read_retired_input_invalidation(
    namespace: Namespace,
    *,
    verify_live_run_state: bool = True,
) -> tuple[dict[str, Any], str]:
    path = _retired_input_invalidation_path(namespace)
    try:
        encoded = read_file_anchored(
            path,
            require_single_link=True,
            max_bytes=_MAX_INPUT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError("retired input invalidation failed anchored readback") from exc
    raw = _strict_json_object(encoded, "retired input invalidation receipt")
    canonical = _canonical_retired_input_invalidation(raw, namespace=namespace)
    if encoded != _canonical_bytes(canonical):
        raise ControlError("retired input invalidation receipt bytes are not canonical")
    if canonical["retired_bundle"] != _retired_bundle_snapshot(namespace):
        raise ControlError("retired v1 bundle changed after invalidation")
    control_root = _absolute(namespace.control_root)
    prior_payload, prior_record = _live_canonical_json_record(
        control_root / _PRIOR_FAILURE_RECEIPT_FILENAME,
        "prior-publication failure receipt",
    )
    if prior_record != canonical["prior_failure_receipt"]:
        raise ControlError("prior-publication failure receipt changed after invalidation")
    failed_snapshot = _retired_failure_evidence_snapshot(namespace)
    if failed_snapshot != canonical["failed_preflight_evidence"]:
        raise ControlError("failed-preflight evidence changed after invalidation")
    if (
        _live_file_record(Path(__file__), "corrected replacement controller")
        != canonical["corrected_controller"]
    ):
        raise ControlError("corrected replacement controller changed after invalidation")
    paths = {
        "run_root": _namespace_project_root(namespace) / "artifacts" / "runs",
    }
    if verify_live_run_state and _live_run_state(paths)["sha256"] != canonical["run_state_sha256"]:
        raise ControlError("run-state changed after retired-input invalidation")
    failed_at = _canonical_historical_failed_timestamp(
        _HISTORICAL_FAILED_AT_UTC,
        "historical failed-preflight observation",
    )
    prior_at = _canonical_timestamp(
        prior_payload.get("observed_at_utc"),
        "prior-publication failure observation",
    )
    invalidated_at = _canonical_timestamp(
        canonical["invalidated_at_utc"],
        "retired input invalidation time",
    )
    if not failed_at < prior_at <= invalidated_at:
        raise ControlError("retired invalidation timestamps are out of order")
    return canonical, hashlib.sha256(encoded).hexdigest()


def publish_retired_input_invalidation_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    invalidated_at: datetime | None = None,
) -> dict[str, Any]:
    """Permanently preserve v1 as invalid evidence without touching its bytes."""

    from histo_audit.workflows import (
        verify_resource_bounded_prior_publication_failure_receipt,
    )

    project_root, parent, _ = _require_project_layout(
        namespace,
        parent=parent_authority_directory,
    )
    if parent.name != _AUTHORITY_C_COMPONENT:
        raise ControlError("retired-input invalidation requires exact Authority C")
    control_root = _absolute(namespace.control_root)
    destination = _retired_input_invalidation_path(namespace)
    if os.path.lexists(destination):
        raise FileExistsError("retired input invalidation receipt already exists")
    forbidden = (
        control_root / _REPLACEMENT_INPUT_DIRECTORY_NAME,
        namespace.publication_authorization,
        namespace.attempt,
        namespace.success,
        namespace.failure,
    )

    def competing_state_reason() -> str | None:
        if any(os.path.lexists(path) for path in forbidden):
            return "v2 or publication control write exists"
        if discover_candidates(parent):
            return "Authority D candidate exists"
        return None

    initial_competing_state = competing_state_reason()
    if initial_competing_state is not None:
        raise ControlError(f"retired input invalidation requires no {initial_competing_state}")
    prior_path = control_root / _PRIOR_FAILURE_RECEIPT_FILENAME
    verify_resource_bounded_prior_publication_failure_receipt(
        superseded_resource_authority_directory=parent,
        receipt_path=prior_path,
    )
    retired = _retired_bundle_snapshot(namespace)
    prior_record = _live_file_record(prior_path, "prior-publication failure receipt")
    if (
        prior_record["size_bytes"] != _PRIOR_FAILURE_RECEIPT_SIZE_BYTES
        or prior_record["sha256"] != _PRIOR_FAILURE_RECEIPT_SHA256
    ):
        raise ControlError("prior-publication failure receipt differs from exact v1 lineage")
    failed = _retired_failure_evidence_snapshot(namespace)
    controller_record = _live_file_record(Path(__file__), "corrected replacement controller")
    run_state = _live_run_state({"run_root": project_root / "artifacts" / "runs"})
    if run_state["sha256"] != _RETIRED_RUN_STATE_SHA256:
        raise ControlError("run-state differs from the frozen v1 lineage")
    captured_at = datetime.now(UTC)
    moment = invalidated_at or captured_at
    if moment.tzinfo is None:
        raise ControlError("retired input invalidation timestamp must be timezone-aware")
    moment = moment.astimezone(UTC)
    if moment > captured_at:
        raise ControlError("retired input invalidation timestamp must not be in the future")
    timestamp = moment.isoformat(timespec="microseconds").replace("+00:00", "Z")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "policy": _RETIRED_INPUT_INVALIDATION_POLICY,
        "status": "preserved_invalid_nonpublishable",
        "invalidated_at_utc": timestamp,
        "reason_code": _RETIRED_INVALIDATION_REASON_CODE,
        "reason": _RETIRED_INVALIDATION_REASON,
        "retired_bundle": retired,
        "prior_failure_receipt": prior_record,
        "failed_preflight_evidence": failed,
        "corrected_controller": controller_record,
        "run_state_sha256": run_state["sha256"],
        "disposition": {
            "v1_preflight_allowed": False,
            "v1_authorization_allowed": False,
            "v1_publication_allowed": False,
            "v1_may_be_modified_moved_or_deleted": False,
            "prior_failure_receipt_may_be_modified_or_republished": False,
            "replacement_requires_exact_v2_singleton": True,
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    canonical = _canonical_retired_input_invalidation(receipt, namespace=namespace)
    if canonical != receipt:
        raise ControlError("retired input invalidation builder produced noncanonical evidence")
    published: PublishedPath | None = None
    verification: dict[str, Any] | None = None
    preserve_published_on_ambiguity = False
    try:
        with ExclusiveBundlePublicationLock(
            (destination, *forbidden),
            role="resource Authority-D retired v1 invalidation",
        ) as publication_lock:
            try:
                if os.path.lexists(destination):
                    raise FileExistsError("retired input invalidation receipt already exists")
                competing_before_write = competing_state_reason()
                if competing_before_write is not None:
                    raise ControlError(
                        "retired input invalidation competing state appeared before write: "
                        f"{competing_before_write}"
                    )
                if (
                    _retired_bundle_snapshot(namespace) != retired
                    or _live_file_record(prior_path, "prior-publication failure receipt")
                    != prior_record
                    or _retired_failure_evidence_snapshot(namespace) != failed
                    or _live_file_record(Path(__file__), "corrected replacement controller")
                    != controller_record
                    or _live_run_state({"run_root": project_root / "artifacts" / "runs"})
                    != run_state
                ):
                    raise ControlError("retired invalidation inputs changed before publication")
                publication_lock.assert_owned()
                published = publish_bytes_no_overwrite(_canonical_bytes(receipt), destination)
                publication_lock.assert_owned()
                verification, receipt_sha256 = _read_retired_input_invalidation(namespace)
                if published.sha256 != receipt_sha256:
                    raise ControlError("retired input invalidation readback digest changed")
                try:
                    competing_after_readback = competing_state_reason()
                except BaseException as discovery_error:
                    preserve_published_on_ambiguity = True
                    raise AmbiguousStateError(
                        "retired input invalidation cannot prove a stable no-D state "
                        "after exact readback; preserve the receipt and stop"
                    ) from discovery_error
                if competing_after_readback is not None:
                    preserve_published_on_ambiguity = True
                    raise AmbiguousStateError(
                        "retired input invalidation competing state appeared after readback: "
                        f"{competing_after_readback}; preserve the exact receipt and stop"
                    )
                publication_lock.assert_owned()
            except BaseException as locked_error:
                if published is not None and not preserve_published_on_ambiguity:
                    publication_lock.assert_owned()
                    try:
                        rollback_owned_publications([published])
                    except (OSError, RuntimeError) as rollback_error:
                        raise RuntimeError(
                            "retired input invalidation failed and ownership-safe "
                            "rollback was incomplete while exclusion remained held; "
                            f"triggering error was {type(locked_error).__name__}: "
                            f"{locked_error}"
                        ) from rollback_error
                    try:
                        publication_lock.assert_owned()
                    except BaseException as ownership_error:
                        published = None
                        raise AmbiguousStateError(
                            "retired input invalidation rollback completed but "
                            "exclusion ownership could not be revalidated; stop"
                        ) from ownership_error
                    published = None
                raise locked_error
    except BaseException as publication_error:
        if published is not None and not preserve_published_on_ambiguity:
            raise AmbiguousStateError(
                "retired input invalidation exclusion ended before safe rollback; "
                "do not mutate the remaining state; exact disposition is ambiguous; stop"
            ) from publication_error
        raise publication_error
    if verification is None or published is None:  # pragma: no cover - defensive
        raise RuntimeError("retired input invalidation returned without exact readback")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "retired_v1_preserved_invalid_nonpublishable",
        "receipt_path": str(destination),
        "receipt_sha256": published.sha256,
        "retired_bundle_sha256": verification["retired_bundle"]["records_sha256"],
        "corrected_controller_sha256": verification["corrected_controller"]["sha256"],
        "run_state_sha256": verification["run_state_sha256"],
        "active_replacement_input_directory": str(control_root / _REPLACEMENT_INPUT_DIRECTORY_NAME),
        "publication_performed": False,
        "scientific_execution_performed": False,
        "outcome_value_interpretation_performed": False,
    }


def _normalise_source_path(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ControlError("source path must be one non-empty canonical string")
    if "\\" in value or ":" in value:
        raise ControlError("source path must use canonical relative POSIX syntax")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or logical.as_posix() != value
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.parts[0].casefold() in {".git", ".venv", "artifacts", "data"}
        or value in {"configs/confirmatory_frozen.yaml", "configs/primary_frozen.yaml"}
    ):
        raise ControlError(f"source path escapes the execution-source scope: {value!r}")
    return value


def _derive_live_source(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    from histo_audit.utils.run_tracking import capture_source_tree
    from histo_audit.workflows import preregistration_amendment as amendment

    parent_source = _live_json(
        paths["parent"] / "source_tree_manifest.json",
        "Authority-C execution-source snapshot",
    )
    parent_pins = {
        "root_sha256": parent_source.get("root_sha256"),
        "manifest_sha256": hashlib.sha256(_canonical_bytes(parent_source)).hexdigest(),
        "artifact_count": parent_source.get("artifact_count"),
    }
    if parent_pins != _AUTHORITY_C_SOURCE_PINS:
        raise ControlError("Authority-C execution source differs from its exact pins")
    current = capture_source_tree(paths["project_root"])
    parent_records = {
        str(record["path"]): dict(record)
        for record in parent_source.get("artifacts", [])
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    current_records = {
        str(record["path"]): dict(record)
        for record in current.get("artifacts", [])
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    observed: dict[str, str] = {}
    for logical in sorted(set(parent_records) | set(current_records)):
        if logical not in parent_records:
            observed[logical] = "added"
        elif logical not in current_records:
            observed[logical] = "removed"
        elif parent_records[logical] != current_records[logical]:
            observed[logical] = "modified"
    if observed != _EXPECTED_SOURCE_CHANGE_KINDS:
        raise ControlError(
            "live C-to-D execution-source delta differs from the exact 17-path allowlist"
        )
    delta, delta_sha256 = amendment._canonical_source_delta_with_allowlist(
        parent_source,
        current,
        allowlisted_change_kinds=observed,
        role="resource Authority-D replacement",
    )
    records: list[dict[str, Any]] = []
    for logical, change_kind in sorted(observed.items()):
        _normalise_source_path(logical)
        if change_kind == "removed":
            records.append({"path": logical, "change_kind": "removed"})
            continue
        source_record = current_records.get(logical)
        if not isinstance(source_record, Mapping):
            raise ControlError(f"live execution-source capture lacks {logical}")
        size = source_record.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ControlError(f"live execution-source size is invalid for {logical}")
        records.append(
            {
                "path": logical,
                "change_kind": change_kind,
                "size_bytes": size,
                "sha256": _sha(source_record.get("sha256"), f"source {logical}"),
            }
        )
    allowlist = {
        "schema_version": SCHEMA_VERSION,
        "policy": _SOURCE_ALLOWLIST_POLICY,
        "file_count": len(_EXPECTED_SOURCE_CHANGE_KINDS),
        "records": records,
    }
    current_manifest_sha256 = hashlib.sha256(_canonical_bytes(current)).hexdigest()
    return {
        "parent_source": parent_source,
        "current_source": current,
        "current_manifest_sha256": current_manifest_sha256,
        "change_kinds": observed,
        "delta": tuple(dict(record) for record in delta),
        "delta_sha256": delta_sha256,
        "allowlist": allowlist,
    }


def _require_live_authority_and_config(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    from histo_audit.config import config_sha256, load_config
    from histo_audit.experiment.study_contracts import (
        validate_resource_bounded_confirmatory_config,
    )
    from histo_audit.workflows import preregistration_amendment as amendment

    verification = amendment.verify_preregistration_amendment(paths["parent"])
    observed_parent = {
        "artifact_root_sha256": verification.artifact_root_sha256,
        "sha256_manifest_sha256": verification.sha256_manifest_sha256,
        "amendment_evidence_sha256": _live_file_record(
            paths["parent"] / "amendment_evidence.json",
            "Authority-C amendment evidence",
        )["sha256"],
        "immutable_marker_sha256": _live_file_record(
            paths["parent"] / ".immutable.json",
            "Authority-C immutable marker",
        )["sha256"],
        "chain_depth": verification.chain_depth,
    }
    if not verification.valid or observed_parent != _AUTHORITY_C_PINS:
        raise ControlError("Authority C differs from its exact immutable identity")
    failed_record = _live_file_record(
        paths["failed_preflight"],
        "failed resource preflight receipt",
    )
    if failed_record["sha256"] != _FAILED_PREFLIGHT_SHA256:
        raise ControlError("failed resource preflight receipt differs from its exact pin")
    failed = _live_json(paths["failed_preflight"], "failed resource preflight receipt")
    config_record = _live_file_record(paths["config"], "resource confirmatory config")
    if config_record["sha256"] != _RESOURCE_CONFIG_FILE_SHA256:
        raise ControlError("resource confirmatory config differs from its exact file pin")
    config = validate_resource_bounded_confirmatory_config(load_config(paths["config"]))
    if config_sha256(config) != _RESOURCE_CONFIG_SEMANTIC_SHA256:
        raise ControlError("resource confirmatory config differs from its semantic pin")
    manifest_record = _live_file_record(paths["manifest"], "PanNuke nucleus manifest")
    if manifest_record["sha256"] != _PANNUKE_MANIFEST_SHA256:
        raise ControlError("PanNuke nucleus manifest differs from its exact pin")
    parent_authorization = amendment._require_resource_bounded_confirmatory_authorization(
        paths["parent"],
        verify_live_primary=False,
    )
    return {
        "parent_verification": verification,
        "parent_authorization": dict(parent_authorization),
        "failed": failed,
        "failed_record": failed_record,
        "config": config,
        "config_record": config_record,
        "config_semantic_sha256": _RESOURCE_CONFIG_SEMANTIC_SHA256,
        "manifest_record": manifest_record,
    }


def _derive_live_workspace(
    paths: Mapping[str, Path],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    from histo_audit.experiment.confirmatory_cli_inputs import (
        _resolve_bound_confirmatory_inputs,
    )
    from histo_audit.experiment.confirmatory_memory_workspace import (
        build_confirmatory_memory_workspace_plan,
    )
    from histo_audit.experiment.m7_config_finalization import (
        derive_confirmatory_cnn_logical_provenance,
    )
    from histo_audit.experiment.pannuke_confirmatory_inputs import (
        derive_pannuke_confirmatory_workspace_array_specs,
        derive_pannuke_confirmatory_workspace_index_specs,
    )
    from histo_audit.representations.cache_provenance import verify_frozen_cache_sidecar
    from histo_audit.workflows import preregistration_amendment as amendment
    from histo_audit.workflows.preregistration_amendment import (
        validate_resource_bounded_capacity_v3,
    )

    config = authority["config"]
    historical_primary = authority["parent_authorization"].get("historical_primary")
    if not isinstance(historical_primary, Mapping):
        raise ControlError("Authority C lacks its historical-primary binding")
    primary_run = _real_directory(
        historical_primary.get("run_directory", ""),
        "historical primary run",
    )
    resolved = _resolve_bound_confirmatory_inputs(
        primary_run=primary_run,
        config_path=paths["config"].resolve(),
        manifest=paths["manifest"],
        config=config,
        config_file_sha=authority["config_record"]["sha256"],
        semantic_sha=authority["config_semantic_sha256"],
        manifest_sha=authority["manifest_record"]["sha256"],
    )
    array_specs = tuple(
        derive_pannuke_confirmatory_workspace_array_specs(
            resolved.crop_cache_path,
            expected_crop_cache_sha256=resolved.expected_crop_cache_sha256,
            expected_crop_metadata_sha256=resolved.expected_crop_metadata_sha256,
            frozen_feature_caches=resolved.frozen_feature_caches,
        )
    )
    index_specs = tuple(
        derive_pannuke_confirmatory_workspace_index_specs(
            resolved.crop_cache_path,
            confirmatory_config=config,
            expected_config_sha256=authority["config_semantic_sha256"],
            expected_crop_cache_sha256=resolved.expected_crop_cache_sha256,
            expected_crop_metadata_sha256=resolved.expected_crop_metadata_sha256,
            expected_manifest_sha256=authority["manifest_record"]["sha256"],
            expected_raw_inventory_sha256=resolved.expected_raw_inventory_sha256,
        )
    )
    capacity = dict(amendment._RESOURCE_BOUNDED_CAPACITY_V3)
    plan = build_confirmatory_memory_workspace_plan(
        array_specs,
        index_specs,
        minimum_free_bytes_after=int(capacity["minimum_free_bytes_before_tracker"]),
        maximum_workspace_bytes=int(capacity["maximum_workspace_bytes"]),
    )
    canonical_capacity, canonical_plan = validate_resource_bounded_capacity_v3(
        capacity,
        plan,
    )
    if canonical_capacity != capacity or canonical_plan != plan:
        raise ControlError("public capacity-v3 validator changed the reconstructed plan")
    crop = verify_frozen_cache_sidecar(
        resolved.crop_cache_path,
        expected_manifest_sha256=authority["manifest_record"]["sha256"],
        expected_representation_id="pannuke_component_covering_target_crops",
    )
    records = config.get("cache_provenance")
    if not isinstance(records, list):
        raise ControlError("resource config lacks cache-provenance records")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("id") == "cnn_context_rgb_cache"
    ]
    if len(matches) != 1:
        raise ControlError("resource config lacks exactly one context-RGB CNN record")
    cnn_record = dict(matches[0])
    derived = derive_confirmatory_cnn_logical_provenance(
        crop,
        weight_identifier=str(cnn_record["weight_identifier"]),
        weights_sha256=str(cnn_record["weights_sha256"]),
        input_size=int(config["training"]["input_size"]),
    )
    runtime_record = derived.get("cnn_context_rgb_cache")
    if not isinstance(runtime_record, Mapping) or dict(runtime_record) != cnn_record:
        raise ControlError("live CNN logical provenance differs from the resource config")
    return {
        "resolved": resolved,
        "array_specs": array_specs,
        "index_specs": index_specs,
        "capacity_policy": capacity,
        "workspace_plan": plan,
        "crop": crop,
        "runtime_record": dict(runtime_record),
    }


def _source_record_sha(source: Mapping[str, Any], logical_path: str) -> str:
    records = source.get("artifacts")
    if not isinstance(records, list):
        raise ControlError("execution-source evidence lacks artifact records")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("path") == logical_path
    ]
    if len(matches) != 1:
        raise ControlError(f"execution source lacks exactly one {logical_path}")
    return _sha(matches[0].get("sha256"), logical_path)


def _cnn_config_record(config: Mapping[str, Any]) -> dict[str, Any]:
    records = config.get("cache_provenance")
    if not isinstance(records, list):
        raise ControlError("confirmatory config lacks cache-provenance records")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("id") == "cnn_context_rgb_cache"
    ]
    if len(matches) != 1:
        raise ControlError("confirmatory config lacks exactly one context-RGB CNN record")
    return dict(matches[0])


def _derive_context_rgb_record_for_source(
    runtime_record: Mapping[str, Any],
    crop: Any,
    *,
    implementation_source_sha256: str,
    input_size: int,
) -> dict[str, Any]:
    implementation_sha = _sha(
        implementation_source_sha256,
        "CNN logical-provenance implementation source",
    )
    if type(input_size) is not int or input_size < 32:
        raise ControlError("CNN logical-provenance input size is invalid")
    crop_metadata = getattr(crop, "metadata", None)
    if not isinstance(crop_metadata, Mapping):
        raise ControlError("verified crop evidence lacks canonical metadata")
    crop_arrays = crop_metadata.get("cache_array_sha256_by_name")
    if not isinstance(crop_arrays, Mapping) or "context_rgb" not in crop_arrays:
        raise ControlError("verified crop evidence lacks its context_rgb binding")
    weight_identifier = runtime_record.get("weight_identifier")
    if not isinstance(weight_identifier, str) or not weight_identifier:
        raise ControlError("CNN logical-provenance weight identifier is invalid")
    weights_sha256 = _sha(
        runtime_record.get("weights_sha256"),
        "CNN logical-provenance weights",
    )
    input_binding = {
        "schema_version": 1,
        "binding_type": "confirmatory_cnn_logical_crop_view_v1",
        "crop_cache_file_sha256": crop.cache_file_sha256,
        "crop_sidecar_semantic_sha256": crop.sidecar_semantic_sha256,
        "crop_cache_content_sha256": crop_metadata.get("cache_content_sha256"),
        "sample_order_sha256": crop_metadata.get("sample_order_sha256"),
        "manifest_sha256": crop_metadata.get("manifest_sha256"),
        "input_array_sha256_by_name": {
            "context_rgb": str(crop_arrays["context_rgb"]),
        },
    }
    encoder_metadata = {
        "schema_version": 1,
        "identifier": "confirmatory_resnet18_five_class_encoder_v1",
        "implementation_module": "histo_audit.models.cnn",
        "implementation_source_sha256": implementation_sha,
        "architecture": "torchvision.resnet18",
        "class_order": [0, 1, 2, 3, 4],
        "input_variant": "context_rgb",
        "input_channels": 3,
        "output_classes": 5,
        "weight_identifier": weight_identifier,
        "weights_sha256": weights_sha256,
        "fourth_channel_initialisation": None,
    }
    preprocessing = {
        "schema_version": 1,
        "identifier": "confirmatory_context_rgb_224_v1",
        "implementation_module": "histo_audit.models.cnn._batch_tensor",
        "implementation_source_sha256": implementation_sha,
        "input_size": input_size,
        "rgb_dtype": "uint8",
        "rgb_scale": "divide_by_255_to_float32",
        "rgb_resize": "bilinear_align_corners_false_antialias_true",
        "rgb_mean": [0.485, 0.456, 0.406],
        "rgb_standard_deviation": [0.229, 0.224, 0.225],
        "target_mask_resize": None,
        "logical_input_binding": input_binding,
    }
    result = dict(runtime_record)
    result["encoder_metadata_sha256"] = _compact_sha256(encoder_metadata)
    result["preprocessing_sha256"] = _compact_sha256(preprocessing)
    return result


def _derive_live_cnn_correction(
    paths: Mapping[str, Path],
    authority: Mapping[str, Any],
    source: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    from histo_audit.config import load_config
    from histo_audit.workflows.preregistration_amendment import (
        ResourceBoundedCnnProvenanceCorrection,
    )

    current_source = source["current_source"]
    runtime_record = workspace["runtime_record"]
    crop = workspace["crop"]
    parent_config = load_config(paths["parent"] / "confirmatory_frozen.yaml")
    before_config_record = _cnn_config_record(parent_config)
    after_config_record = _cnn_config_record(authority["config"])
    input_size = authority["config"].get("training", {}).get("input_size")
    if type(input_size) is not int:
        raise ControlError("resource config lacks one integer CNN input size")
    parent_cnn_sha = _source_record_sha(
        source["parent_source"],
        "src/histo_audit/models/cnn.py",
    )
    current_cnn_sha = _source_record_sha(
        current_source,
        "src/histo_audit/models/cnn.py",
    )
    historical_logical_sha = _sha(
        authority["failed"].get("logical_provenance_source_sha256"),
        "failed-preflight logical-provenance source",
    )
    historical_record = _derive_context_rgb_record_for_source(
        runtime_record,
        crop,
        implementation_source_sha256=historical_logical_sha,
        input_size=input_size,
    )
    parent_recomputed = _derive_context_rgb_record_for_source(
        runtime_record,
        crop,
        implementation_source_sha256=parent_cnn_sha,
        input_size=input_size,
    )
    current_recomputed = _derive_context_rgb_record_for_source(
        runtime_record,
        crop,
        implementation_source_sha256=current_cnn_sha,
        input_size=input_size,
    )
    if historical_record != before_config_record:
        raise ControlError(
            "Authority-C CNN config does not reconstruct from its historical logical source"
        )
    if (
        parent_recomputed == before_config_record
        or current_recomputed != runtime_record
        or current_recomputed != after_config_record
    ):
        raise ControlError(
            "CNN reconstruction does not prove the exact two-field provenance correction"
        )
    semantic = {
        "schema_version": 1,
        "policy": _CNN_SEMANTIC_EQUIVALENCE_POLICY,
        "scenario_id": "cnn_context_rgb",
        "cache_provenance_id": "cnn_context_rgb_cache",
        "runtime_model_or_preprocessing_behavior_changed": False,
        "scientific_profile_changed": False,
        "cache_bytes_changed": False,
        "sidecar_bytes_changed": False,
        "evidence_note": (
            "Outcome-blind deterministic reconstruction binds the unchanged context-RGB "
            "CNN recipe and verified crop bytes to the current implementation source; "
            "only encoder_metadata_sha256 and preprocessing_sha256 change."
        ),
    }
    correction = ResourceBoundedCnnProvenanceCorrection(
        before_execution_cnn_source_sha256=parent_cnn_sha,
        before_logical_provenance_source_sha256=historical_logical_sha,
        before_recomputed_record_sha256=_compact_sha256(parent_recomputed),
        after_execution_cnn_source_sha256=current_cnn_sha,
        after_logical_provenance_source_sha256=current_cnn_sha,
        after_recomputed_record_sha256=_compact_sha256(current_recomputed),
        semantic_equivalence_evidence_sha256=_compact_sha256(semantic),
        crop_cache_file_sha256=crop.cache_file_sha256,
        crop_sidecar_file_sha256=crop.sidecar_file_sha256,
        crop_sidecar_semantic_sha256=crop.sidecar_semantic_sha256,
    )
    full = correction.as_dict(
        before_config_record=before_config_record,
        after_config_record=after_config_record,
    )
    return {
        "correction": correction,
        "full": full,
        "receipt": {
            "schema_version": SCHEMA_VERSION,
            "policy": _CNN_CORRECTION_RECEIPT_POLICY,
            "correction": full,
            "semantic_equivalence_evidence": semantic,
        },
    }


def _derive_live_foundation(
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> dict[str, Any]:
    state = classify(
        namespace,
        parent_authority_directory=parent_authority_directory,
    )
    if state.state is not State.READY:
        raise ControlError(f"live derivation requires READY, got {state.state.value}")
    paths = _live_paths(namespace, parent_authority_directory)
    invalidation, invalidation_sha256 = _read_retired_input_invalidation(namespace)
    authority = _require_live_authority_and_config(paths)
    source = _derive_live_source(paths)
    workspace = _derive_live_workspace(paths, authority)
    cnn = _derive_live_cnn_correction(
        paths,
        authority,
        source,
        workspace,
    )
    return {
        "namespace": namespace,
        "paths": paths,
        "authority": authority,
        "source": source,
        "workspace": workspace,
        "cnn": cnn,
        "run_state": _live_run_state(paths),
        "retired_input_invalidation": {
            "receipt": invalidation,
            "record": _live_file_record(
                paths["retired_input_invalidation"],
                "retired input invalidation receipt",
            ),
            "sha256": invalidation_sha256,
        },
        "controller_record": _live_file_record(
            Path(__file__),
            "replacement controller",
        ),
    }


def _finalize_live_context(foundation: Mapping[str, Any]) -> dict[str, Any]:
    from histo_audit.workflows import (
        build_resource_bounded_technical_successor_authorization,
        verify_resource_bounded_prior_publication_failure_receipt,
    )

    paths = foundation["paths"]
    authority = foundation["authority"]
    source = foundation["source"]
    workspace = foundation["workspace"]
    cnn = foundation["cnn"]
    prior = verify_resource_bounded_prior_publication_failure_receipt(
        superseded_resource_authority_directory=paths["parent"],
        receipt_path=paths["prior_failure"],
    )
    authorization = build_resource_bounded_technical_successor_authorization(
        project_root=paths["project_root"],
        superseded_resource_authority_directory=paths["parent"],
        resource_confirmatory_config_path=paths["config"],
        failed_preflight_receipt_path=paths["failed_preflight"],
        prior_publication_failure_receipt_path=paths["prior_failure"],
        cnn_provenance_correction=cnn["correction"],
        source_delta_allowlist=source["change_kinds"],
        resource_input_workspace_plan=workspace["workspace_plan"],
        resource_input_workspace_array_specs=workspace["array_specs"],
        resource_input_workspace_index_specs=workspace["index_specs"],
        expected_successor_config_semantic_sha256=_RESOURCE_CONFIG_SEMANTIC_SHA256,
    )
    authorization_dict = authorization.as_dict()
    authorization_source = authorization_dict.get("execution_source_delta")
    expected_authorization_source = {
        "resource_root_sha256": source["current_source"]["root_sha256"],
        "resource_manifest_sha256": source["current_manifest_sha256"],
        "delta_sha256": source["delta_sha256"],
    }
    if not isinstance(authorization_source, Mapping) or any(
        authorization_source.get(field_name) != expected
        for field_name, expected in expected_authorization_source.items()
    ):
        raise ControlError("authorization execution source differs from the live foundation")
    if authorization_dict.get("cnn_provenance_correction") != cnn["full"]:
        raise ControlError("authorization changed the full frozen CNN correction")
    authorization_sha256 = _compact_sha256(authorization_dict)
    allowlist = source["allowlist"]
    workspace_plan = workspace["workspace_plan"]
    cnn_receipt = cnn["receipt"]
    child_sha256 = {
        "source_allowlist_sha256": hashlib.sha256(_canonical_bytes(allowlist)).hexdigest(),
        "workspace_plan_sha256": hashlib.sha256(_canonical_bytes(workspace_plan)).hexdigest(),
        "cnn_correction_receipt_sha256": hashlib.sha256(_canonical_bytes(cnn_receipt)).hexdigest(),
    }
    source_receipt = {
        "schema_version": SCHEMA_VERSION,
        "policy": _FROZEN_SOURCE_RECEIPT_POLICY,
        "file_count": len(_EXPECTED_SOURCE_CHANGE_KINDS),
        **child_sha256,
        "source_allowlist_semantic_sha256": _compact_sha256(allowlist),
        "execution_source_root_sha256": source["current_source"]["root_sha256"],
        "execution_source_manifest_sha256": source["current_manifest_sha256"],
        "execution_source_artifact_count": source["current_source"]["artifact_count"],
        "execution_source_delta_count": len(source["delta"]),
        "execution_source_delta_sha256": source["delta_sha256"],
        "execution_source_change_kinds_sha256": _compact_sha256(source["change_kinds"]),
        "parent_execution_source_root_sha256": _AUTHORITY_C_SOURCE_PINS["root_sha256"],
        "parent_execution_source_manifest_sha256": _AUTHORITY_C_SOURCE_PINS["manifest_sha256"],
        "config_path": str(paths["config"]),
        "config_file_sha256": authority["config_record"]["sha256"],
        "config_semantic_sha256": authority["config_semantic_sha256"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": authority["manifest_record"]["sha256"],
        "failed_preflight_receipt_path": str(paths["failed_preflight"]),
        "failed_preflight_receipt_sha256": authority["failed_record"]["sha256"],
        "prior_failure_receipt_path": prior["receipt_path"],
        "prior_failure_receipt_sha256": prior["receipt_sha256"],
        "retired_input_invalidation_receipt_path": str(paths["retired_input_invalidation"]),
        "retired_input_invalidation_receipt_sha256": foundation["retired_input_invalidation"][
            "sha256"
        ],
        "controller_path": foundation["controller_record"]["path"],
        "controller_size_bytes": foundation["controller_record"]["size_bytes"],
        "controller_sha256": foundation["controller_record"]["sha256"],
        "run_state_root": foundation["run_state"]["root"],
        "run_state_files": foundation["run_state"]["files"],
        "run_state_sha256": foundation["run_state"]["sha256"],
        "authorization_sha256": authorization_sha256,
        "workspace_plan_without_self_hash_sha256": workspace_plan["plan_without_self_hash_sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    result = dict(foundation)
    result.update(
        {
            "prior": prior,
            "authorization": authorization,
            "authorization_dict": authorization_dict,
            "authorization_sha256": authorization_sha256,
            "payloads": {
                "source_allowlist": allowlist,
                "workspace_plan": workspace_plan,
                "cnn_correction_receipt": cnn_receipt,
                "frozen_source_receipt": source_receipt,
            },
        }
    )
    return result


def _canonical_external_timestamp(value: object, role: str) -> str:
    if not isinstance(value, str):
        raise ControlError(f"{role} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ControlError(f"{role} is invalid") from exc
    if parsed.tzinfo is None:
        raise ControlError(f"{role} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_live_capacity_and_compute(context: Mapping[str, Any]) -> dict[str, Any]:
    from histo_audit.experiment.resource_bounded_runner import (
        require_resource_capacity,
        require_resource_compute,
    )

    paths = context["paths"]
    workspace = context["workspace"]
    capacity = require_resource_capacity(
        paths["run_root"],
        capacity_policy=workspace["capacity_policy"],
        phase="guarded_before_workspace_build",
        resource_input_workspace_plan=workspace["workspace_plan"],
    )
    compute = require_resource_compute(
        phase="guarded_before_data_loading",
        capacity_policy=workspace["capacity_policy"],
        resource_input_workspace_plan=workspace["workspace_plan"],
    )
    if not capacity.passed or not compute.passed:
        raise ControlError("public live resource gates did not both pass")
    return {
        "capacity": capacity,
        "capacity_dict": capacity.as_dict(),
        "compute": compute,
        "compute_dict": compute.as_dict(),
    }


def _replacement_bindings(
    paths: Mapping[str, Path],
    input_directory: str | Path,
) -> FrozenInputBindings:
    root = _replacement_input_directory(
        paths["control_root"],
        input_directory,
    )
    return FrozenInputBindings(
        failed_preflight_receipt=paths["failed_preflight"],
        prior_failure_receipt=paths["prior_failure"],
        retired_input_invalidation=paths["retired_input_invalidation"],
        frozen_source_receipt=root / _REPLACEMENT_INPUT_FILENAMES["frozen_source_receipt"],
        source_allowlist=root / _REPLACEMENT_INPUT_FILENAMES["source_allowlist"],
        workspace_plan=root / _REPLACEMENT_INPUT_FILENAMES["workspace_plan"],
        cnn_correction_receipt=root / _REPLACEMENT_INPUT_FILENAMES["cnn_correction_receipt"],
    )


def _verify_frozen_payloads(
    context: Mapping[str, Any],
    bindings: FrozenInputBindings,
) -> dict[str, Any]:
    namespace = context["namespace"]
    paths, hashes, _ = _capture_attempt_inputs(
        namespace,
        bindings,
        capture_run_state=False,
    )
    for role in _REPLACEMENT_INPUT_FILENAMES:
        try:
            encoded = read_file_anchored(
                paths[role],
                require_single_link=True,
                max_bytes=_MAX_INPUT_BYTES,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ControlError(f"frozen {role} failed exact readback") from exc
        observed = _strict_json_object(encoded, f"frozen {role}")
        expected = context["payloads"][role]
        if (
            observed != expected
            or encoded != _canonical_bytes(expected)
            or hashlib.sha256(encoded).hexdigest() != hashes[role]
        ):
            raise ControlError(f"frozen {role} differs from live reconstruction")
    return {
        "paths": paths,
        "hashes": hashes,
        "root": paths["frozen_source_receipt"].parent,
    }


def _require_live_freeze_context_unchanged(context: Mapping[str, Any]) -> None:
    """Recheck every mutable live input used by the singleton freeze."""

    paths = context["paths"]
    if _derive_live_source(paths) != context["source"]:
        raise ControlError("execution source changed during replacement input freeze")
    if _require_live_authority_and_config(paths) != context["authority"]:
        raise ControlError("authority, config, or manifest changed during replacement input freeze")
    if _live_run_state(paths) != context["run_state"]:
        raise ControlError("run-state changed during replacement input freeze")
    if (
        _live_file_record(
            Path(__file__),
            "replacement controller",
        )
        != context["controller_record"]
    ):
        raise ControlError("replacement controller changed during input freeze")
    prior_record = _live_file_record(
        paths["prior_failure"],
        "prior-publication failure receipt",
    )
    if prior_record["sha256"] != context["prior"]["receipt_sha256"]:
        raise ControlError("prior-publication failure receipt changed during input freeze")
    invalidation, invalidation_sha256 = _read_retired_input_invalidation(context["namespace"])
    expected_invalidation = context["retired_input_invalidation"]
    if (
        invalidation != expected_invalidation["receipt"]
        or invalidation_sha256 != expected_invalidation["sha256"]
        or _live_file_record(
            paths["retired_input_invalidation"],
            "retired input invalidation receipt",
        )
        != expected_invalidation["record"]
    ):
        raise ControlError("retired input invalidation changed during input freeze")


def freeze_replacement_inputs_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify preserved lineage and create the exact four-file active v2 bundle."""

    from histo_audit.workflows import (
        verify_resource_bounded_prior_publication_failure_receipt,
    )

    destination = _replacement_input_directory(
        namespace.control_root,
        output_directory,
    )
    if os.path.lexists(destination):
        raise ControlError("canonical replacement freeze destination already exists")

    _read_retired_input_invalidation(namespace)
    initial_paths = _live_paths(namespace, parent_authority_directory)
    verify_resource_bounded_prior_publication_failure_receipt(
        superseded_resource_authority_directory=initial_paths["parent"],
        receipt_path=initial_paths["prior_failure"],
    )
    foundation = _derive_live_foundation(namespace, parent_authority_directory)
    _require_live_capacity_and_compute(foundation)
    paths = foundation["paths"]
    context = _finalize_live_context(foundation)
    final_paths = {
        role: destination / filename for role, filename in _REPLACEMENT_INPUT_FILENAMES.items()
    }
    publications: list[PublishedPath] = []
    resources: dict[str, Any] | None = None
    try:
        with ExclusiveBundlePublicationLock(
            (destination, *final_paths.values()),
            role="resource Authority-D replacement frozen inputs",
        ) as publication_lock:
            try:
                _require_live_freeze_context_unchanged(context)
                resources = _require_live_capacity_and_compute(context)
                publication_lock.assert_owned()
                _require_live_freeze_context_unchanged(context)
                publication_lock.assert_owned()
                publications.append(create_directory_no_overwrite(destination))
                publication_lock.assert_owned()
                for role in (
                    "source_allowlist",
                    "workspace_plan",
                    "cnn_correction_receipt",
                    "frozen_source_receipt",
                ):
                    publication_lock.assert_owned()
                    publications.append(
                        publish_bytes_no_overwrite(
                            _canonical_bytes(context["payloads"][role]),
                            final_paths[role],
                        )
                    )
                    publication_lock.assert_owned()
                publication_lock.assert_owned()
                bindings = _replacement_bindings(paths, destination)
                frozen = _verify_frozen_payloads(context, bindings)
                _require_live_freeze_context_unchanged(context)
                publication_lock.assert_owned()
            except BaseException as locked_error:
                if publications:
                    publication_lock.assert_owned()
                    try:
                        rollback_owned_publications(publications)
                    except (OSError, RuntimeError) as rollback_error:
                        raise RuntimeError(
                            "replacement input freeze failed and ownership-safe "
                            "rollback was incomplete while exclusion remained held; "
                            f"triggering error was {type(locked_error).__name__}: "
                            f"{locked_error}"
                        ) from rollback_error
                    try:
                        publication_lock.assert_owned()
                    except BaseException as ownership_error:
                        publications.clear()
                        raise AmbiguousStateError(
                            "replacement input freeze rollback completed but "
                            "exclusion ownership could not be revalidated; stop"
                        ) from ownership_error
                    publications.clear()
                raise locked_error
    except BaseException as publication_error:
        if publications:
            raise AmbiguousStateError(
                "replacement input freeze exclusion ended before safe rollback; "
                "do not mutate the remaining state; exact disposition is ambiguous; stop"
            ) from publication_error
        raise publication_error
    if resources is None:  # pragma: no cover - defensive postcondition
        raise RuntimeError("replacement input freeze returned without fresh resource evidence")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "replacement_inputs_frozen_and_verified",
        "output_directory": str(destination),
        "files": {
            role: {
                "path": str(frozen["paths"][role]),
                "sha256": frozen["hashes"][role],
            }
            for role in sorted(_REPLACEMENT_INPUT_FILENAMES)
        },
        "prior_failure_receipt_path": str(paths["prior_failure"]),
        "prior_failure_receipt_sha256": context["prior"]["receipt_sha256"],
        "retired_input_invalidation_receipt_path": str(paths["retired_input_invalidation"]),
        "retired_input_invalidation_receipt_sha256": context["retired_input_invalidation"][
            "sha256"
        ],
        "authorization_sha256": context["authorization_sha256"],
        "execution_source_root_sha256": context["source"]["current_source"]["root_sha256"],
        "execution_source_delta_sha256": context["source"]["delta_sha256"],
        "capacity_gate": resources["capacity_dict"],
        "compute_gate_sha256": resources["compute"].evidence_sha256,
        "publication_performed": False,
        "scientific_execution_performed": False,
        "outcome_value_interpretation_performed": False,
    }


def _build_live_preflight(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    input_directory: str | Path,
    amendment_timestamp: datetime | None = None,
) -> dict[str, Any]:
    from histo_audit.workflows.preregistration_amendment import (
        require_confirmatory_storage_policy,
        resource_bounded_technical_successor_intent_sha256,
    )

    context = _finalize_live_context(_derive_live_foundation(namespace, parent_authority_directory))
    paths = context["paths"]
    bindings = _replacement_bindings(paths, input_directory)
    frozen = _verify_frozen_payloads(context, bindings)
    resources = _require_live_capacity_and_compute(context)
    moment = amendment_timestamp or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ControlError("preflight amendment timestamp must be timezone-aware")
    moment = moment.astimezone(UTC)
    timestamp_text = moment.isoformat(timespec="microseconds").replace("+00:00", "Z")
    parent_evidence = _live_json(
        paths["parent"] / "amendment_evidence.json",
        "Authority-C amendment evidence",
    )
    parent_at = _canonical_timestamp(
        parent_evidence.get("amendment_timestamp_utc"),
        "Authority-C amendment timestamp",
    )
    failed_at = _canonical_historical_failed_timestamp(
        context["authority"]["failed"].get("observed_at_utc"),
        "failed-preflight observation",
    )
    prior_at = _canonical_timestamp(
        context["prior"]["evidence"].get("observed_at_utc"),
        "prior-publication failure observation",
    )
    invalidated_at = _canonical_timestamp(
        context["retired_input_invalidation"]["receipt"].get("invalidated_at_utc"),
        "retired input invalidation time",
    )
    if not parent_at < failed_at < prior_at <= invalidated_at <= moment:
        raise ControlError(
            "live preflight timestamps do not follow C < failed < prior <= invalidation <= D"
        )
    destination = paths["amendment_root"] / moment.strftime("%Y%m%dT%H%M%S.%fZ")
    if os.path.lexists(destination):
        raise ControlError("preflight intended Authority-D destination already exists")
    state = classify(
        namespace,
        parent_authority_directory=paths["parent"],
    )
    if state.state is not State.READY or state.candidates:
        raise ControlError("live preflight no longer observes exact READY/no-D state")
    storage_policy = require_confirmatory_storage_policy(paths["parent"])
    intent_sha256 = resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=paths["parent"],
        amendment_timestamp_utc=timestamp_text,
        reason=_AMENDMENT_REASON,
        affected_hypotheses=_AFFECTED_HYPOTHESES,
        affected_analyses=_AFFECTED_ANALYSES,
        outcomes_inspected_at_utc=_OUTCOMES_INSPECTED_AT.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        authorization=context["authorization_dict"],
        confirmatory_storage_policy=storage_policy,
    )
    workspace_plan = context["workspace"]["workspace_plan"]
    capacity_policy = context["workspace"]["capacity_policy"]
    required_before_workspace = int(capacity_policy["minimum_free_bytes_before_workspace_build"])
    required_free = max(
        required_before_workspace,
        int(workspace_plan["required_free_bytes_before"]),
    )
    capacity_contract = {
        "resource_capacity_policy_sha256": _compact_sha256(capacity_policy),
        "workspace_plan_sha256": frozen["hashes"]["workspace_plan"],
        "workspace_plan_without_self_hash_sha256": workspace_plan["plan_without_self_hash_sha256"],
        "projected_stable_run_bytes": int(capacity_policy["projected_stable_run_bytes"]),
        "fixed_safety_margin_bytes": int(capacity_policy["fixed_safety_margin_bytes"]),
        "minimum_free_bytes_before_tracker": int(
            capacity_policy["minimum_free_bytes_before_tracker"]
        ),
        "maximum_workspace_bytes": int(capacity_policy["maximum_workspace_bytes"]),
        "minimum_free_bytes_before_workspace_build": required_before_workspace,
        "planned_workspace_bytes": int(workspace_plan["planned_workspace_bytes"]),
        "required_free_bytes_before": int(workspace_plan["required_free_bytes_before"]),
        "required_free_bytes": required_free,
    }
    run_state = _live_run_state(paths)
    if run_state != context["run_state"]:
        raise ControlError("run-state changed during live preflight")
    contract = {
        "project_root": str(paths["project_root"]),
        "parent_authority_directory": str(paths["parent"]),
        "controller": context["controller_record"],
        "failed_preflight_receipt": _live_file_record(
            paths["failed_preflight"],
            "failed-preflight receipt",
        ),
        "prior_failure_receipt": _live_file_record(
            paths["prior_failure"],
            "prior-publication failure receipt",
        ),
        "retired_input_invalidation_receipt": _live_file_record(
            paths["retired_input_invalidation"],
            "retired input invalidation receipt",
        ),
        "frozen_input_bundle": {
            "directory": str(frozen["root"]),
            **{
                role: _live_file_record(
                    frozen["paths"][role],
                    f"frozen replacement {role}",
                )
                for role in _REPLACEMENT_INPUT_FILENAMES
            },
        },
        "source": {
            "root_sha256": context["source"]["current_source"]["root_sha256"],
            "manifest_sha256": context["source"]["current_manifest_sha256"],
            "delta_sha256": context["source"]["delta_sha256"],
            "allowlisted_change_count": len(context["source"]["change_kinds"]),
        },
        "config": {
            "path": str(paths["config"]),
            "file_sha256": context["authority"]["config_record"]["sha256"],
            "semantic_sha256": context["authority"]["config_semantic_sha256"],
        },
        "manifest": context["authority"]["manifest_record"],
        "run_state": run_state,
        "technical_successor": {
            "authorization_sha256": context["authorization_sha256"],
            "intent_sha256": intent_sha256,
        },
        "replacement_state": {
            "state": State.READY.value,
            "candidate_count": 0,
            "attempt_marker_absent": not os.path.lexists(namespace.attempt),
            "success_marker_absent": not os.path.lexists(namespace.success),
            "failure_marker_absent": not os.path.lexists(namespace.failure),
            "intended_authority_absent": not os.path.lexists(destination),
        },
        "capacity_contract": capacity_contract,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    capacity_dict = resources["capacity_dict"]
    if (
        capacity_dict["minimum_free_bytes"] != required_free
        or capacity_dict["free_bytes"] < required_free
    ):
        raise ControlError("public disk evidence differs from the exact capacity contract")
    compute_evidence = dict(resources["compute_dict"])
    compute_evidence["checked_at_utc"] = _canonical_external_timestamp(
        compute_evidence["checked_at_utc"],
        "public compute observation",
    )
    compute_evidence["observation"] = dict(compute_evidence["observation"])
    if compute_evidence["policy_sha256"] != capacity_contract[
        "resource_capacity_policy_sha256"
    ] or compute_evidence["observation_sha256"] != _compact_sha256(compute_evidence["observation"]):
        raise ControlError("public compute evidence differs from capacity-v3")
    return {
        "context": context,
        "bindings": bindings,
        "frozen": frozen,
        "resources": resources,
        "storage_policy": storage_policy,
        "timestamp": moment,
        "timestamp_text": timestamp_text,
        "destination": destination,
        "intent_sha256": intent_sha256,
        "contract": contract,
        "preflight_fingerprint_sha256": _compact_sha256(contract),
        "capacity_observation": {
            "observed_at_utc": _canonical_external_timestamp(
                capacity_dict["checked_at_utc"],
                "public capacity observation",
            ),
            "filesystem_path": str(paths["run_root"]),
            "observed_free_bytes": capacity_dict["free_bytes"],
            "required_free_bytes": required_free,
            "passed": True,
        },
        "compute_observation": {
            "evidence": compute_evidence,
            "evidence_sha256": _compact_sha256(compute_evidence),
        },
    }


def preflight_replacement_inputs(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    input_directory: str | Path,
) -> dict[str, Any]:
    """Run the full outcome-value-blind live preflight without writing a marker."""

    preflight = _build_live_preflight(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        input_directory=input_directory,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": _LIVE_PREFLIGHT_POLICY,
        "status": "passed",
        "proposed_amendment_timestamp_utc": preflight["timestamp_text"],
        "proposed_authority_directory": str(preflight["destination"]),
        "preflight_fingerprint_sha256": preflight["preflight_fingerprint_sha256"],
        "authorization_sha256": preflight["context"]["authorization_sha256"],
        "intent_sha256": preflight["intent_sha256"],
        "capacity_observation": preflight["capacity_observation"],
        "compute_observation_sha256": preflight["compute_observation"]["evidence_sha256"],
        "automatic_retry_allowed": False,
        "publication_performed": False,
        "scientific_execution_performed": False,
        "outcome_value_interpretation_performed": False,
    }


def authorize_replacement_publication_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    input_directory: str | Path,
) -> dict[str, Any]:
    """Publish one canonical receipt authorizing at most one exact D attempt."""

    active_input_root = _replacement_input_directory(
        namespace.control_root,
        input_directory,
    )
    published: PublishedPath | None = None
    verification: dict[str, Any] | None = None
    try:
        with ExclusiveBundlePublicationLock(
            (
                active_input_root,
                namespace.publication_authorization,
                namespace.attempt,
                namespace.success,
                namespace.failure,
            ),
            role="resource Authority-D replacement publication authorization",
        ) as publication_lock:
            try:
                if os.path.lexists(namespace.publication_authorization):
                    raise FileExistsError(
                        "replacement publication authorization receipt already exists"
                    )
                preflight = _build_live_preflight(
                    namespace=namespace,
                    parent_authority_directory=parent_authority_directory,
                    input_directory=active_input_root,
                )
                receipt = {
                    "schema_version": SCHEMA_VERSION,
                    "policy": _PUBLICATION_AUTHORIZATION_POLICY,
                    "status": "authorized_for_one_attempt",
                    "authorized_at_utc": preflight["timestamp_text"],
                    "automatic_retry_allowed": False,
                    "max_attempt_count": 1,
                    "authorized_attempt_id": secrets.token_hex(32),
                    "publication": {
                        "amendment_timestamp_utc": preflight["timestamp_text"],
                        "intended_authority_directory": str(preflight["destination"]),
                        "parent_authority_directory": str(preflight["context"]["paths"]["parent"]),
                        "amendment_schema_version": 5,
                        "amendment_purpose": _TECHNICAL_SUCCESSOR_PURPOSE,
                        "chain_depth": 4,
                    },
                    "preflight": {
                        "schema_version": SCHEMA_VERSION,
                        "policy": _LIVE_PREFLIGHT_POLICY,
                        "status": "passed",
                        "contract": preflight["contract"],
                        "preflight_fingerprint_sha256": preflight["preflight_fingerprint_sha256"],
                        "capacity_observation": preflight["capacity_observation"],
                        "compute_observation": preflight["compute_observation"],
                    },
                    "outcome_value_interpretation_performed": False,
                    "scientific_execution_performed": False,
                    "publication_performed": False,
                }
                canonical = _canonical_publication_authorization(
                    receipt,
                    namespace=namespace,
                )
                if canonical != receipt:
                    raise ControlError(
                        "publication authorization builder produced noncanonical evidence"
                    )
                publication_lock.assert_owned()
                published = publish_bytes_no_overwrite(
                    _canonical_bytes(receipt),
                    namespace.publication_authorization,
                )
                publication_lock.assert_owned()
                verification, receipt_sha256 = _read_publication_authorization(namespace)
                _verify_publication_authorization_live_files(
                    namespace,
                    verification,
                    verify_run_state=True,
                )
                repeated = _build_live_preflight(
                    namespace=namespace,
                    parent_authority_directory=parent_authority_directory,
                    input_directory=active_input_root,
                    amendment_timestamp=preflight["timestamp"],
                )
                if (
                    repeated["contract"] != preflight["contract"]
                    or repeated["preflight_fingerprint_sha256"]
                    != preflight["preflight_fingerprint_sha256"]
                    or repeated["intent_sha256"] != preflight["intent_sha256"]
                    or repeated["destination"] != preflight["destination"]
                    or published.sha256 != receipt_sha256
                ):
                    raise ControlError("live preflight changed during publication authorization")
                publication_lock.assert_owned()
            except BaseException as locked_error:
                if published is not None:
                    publication_lock.assert_owned()
                    try:
                        rollback_owned_publications([published])
                    except (OSError, RuntimeError) as rollback_error:
                        raise RuntimeError(
                            "publication authorization failed and ownership-safe "
                            "rollback was incomplete while exclusion remained held; "
                            f"triggering error was {type(locked_error).__name__}: "
                            f"{locked_error}"
                        ) from rollback_error
                    try:
                        publication_lock.assert_owned()
                    except BaseException as ownership_error:
                        published = None
                        raise AmbiguousStateError(
                            "publication authorization rollback completed but "
                            "exclusion ownership could not be revalidated; stop"
                        ) from ownership_error
                    published = None
                raise locked_error
    except BaseException as publication_error:
        if published is not None:
            raise AmbiguousStateError(
                "publication authorization exclusion ended before safe rollback; "
                "do not mutate the remaining state; exact disposition is ambiguous; stop"
            ) from publication_error
        raise publication_error
    if verification is None or published is None:  # pragma: no cover - defensive
        raise RuntimeError("publication authorization returned without verification")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "publication_authorized_for_one_attempt",
        "receipt_path": str(namespace.publication_authorization),
        "receipt_sha256": published.sha256,
        "authorized_attempt_id": verification["authorized_attempt_id"],
        "amendment_timestamp_utc": verification["publication"]["amendment_timestamp_utc"],
        "intended_authority_directory": verification["publication"]["intended_authority_directory"],
        "preflight_fingerprint_sha256": verification["preflight"]["preflight_fingerprint_sha256"],
        "authorization_sha256": verification["preflight"]["contract"]["technical_successor"][
            "authorization_sha256"
        ],
        "intent_sha256": verification["preflight"]["contract"]["technical_successor"][
            "intent_sha256"
        ],
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "publication_performed": False,
        "scientific_execution_performed": False,
        "outcome_value_interpretation_performed": False,
    }


def publish_replacement_authority_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    input_directory: str | Path,
) -> PublicationResult:
    """Consume the separate authorization receipt in one atomic D transaction."""

    from histo_audit.workflows.preregistration_amendment import (
        create_preregistration_amendment,
    )

    receipt, _ = _read_publication_authorization(namespace)
    timestamp = _canonical_timestamp(
        receipt["publication"]["amendment_timestamp_utc"],
        "authorized amendment timestamp",
    )
    preflight = _build_live_preflight(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        input_directory=input_directory,
        amendment_timestamp=timestamp,
    )
    if (
        preflight["contract"] != receipt["preflight"]["contract"]
        or preflight["preflight_fingerprint_sha256"]
        != receipt["preflight"]["preflight_fingerprint_sha256"]
        or str(preflight["destination"]) != receipt["publication"]["intended_authority_directory"]
        or preflight["intent_sha256"]
        != receipt["preflight"]["contract"]["technical_successor"]["intent_sha256"]
    ):
        raise ControlError("live publication preflight differs from its authorization receipt")
    context = preflight["context"]
    attempt = attempt_record(
        attempt_id=receipt["authorized_attempt_id"],
        destination=preflight["destination"],
        parent=context["paths"]["parent"],
        controller_path=Path(__file__),
        project_root=context["paths"]["project_root"],
        frozen_inputs=preflight["bindings"],
        publication_authorization_receipt=namespace.publication_authorization,
        authorization_sha256=context["authorization_sha256"],
        intent_sha256=preflight["intent_sha256"],
    )
    verifier = TransactionVerifier(
        project_root=context["paths"]["project_root"],
        parent=context["paths"]["parent"],
        destination=preflight["destination"],
        authorization_sha256=context["authorization_sha256"],
        intent_sha256=preflight["intent_sha256"],
    )

    def preclaim_check() -> None:
        repeated = _build_live_preflight(
            namespace=namespace,
            parent_authority_directory=parent_authority_directory,
            input_directory=input_directory,
            amendment_timestamp=timestamp,
        )
        if (
            repeated["contract"] != receipt["preflight"]["contract"]
            or repeated["preflight_fingerprint_sha256"]
            != receipt["preflight"]["preflight_fingerprint_sha256"]
            or repeated["intent_sha256"] != preflight["intent_sha256"]
            or repeated["destination"] != preflight["destination"]
        ):
            raise ControlError("preclaim live preflight changed after authorization")
        return None

    def transaction(callback: Callable[[Any], None]) -> Any:
        def rollback_scoped_callback(published: Any) -> None:
            _verify_attempt_inputs(namespace, attempt)
            callback(published)
            _verify_attempt_inputs(namespace, attempt)
            return None

        return create_preregistration_amendment(
            project_root=context["paths"]["project_root"],
            parent_authority_directory=context["paths"]["parent"],
            amendment_root=context["paths"]["amendment_root"],
            preregistration_path=context["paths"]["parent"] / "PRE_REGISTRATION_FROZEN.md",
            primary_config_path=context["paths"]["parent"] / "primary_frozen.yaml",
            confirmatory_config_path=context["paths"]["config"],
            reason=_AMENDMENT_REASON,
            affected_hypotheses=_AFFECTED_HYPOTHESES,
            affected_analyses=_AFFECTED_ANALYSES,
            outcomes_inspected=True,
            outcomes_inspected_at=_OUTCOMES_INSPECTED_AT,
            resource_bounded_technical_successor_authorization=context["authorization"],
            confirmatory_storage_policy=preflight["storage_policy"],
            post_publication_check=rollback_scoped_callback,
            timestamp=timestamp,
        )

    return execute_once(
        namespace=namespace,
        attempt=attempt,
        transaction=transaction,
        verifier=verifier,
        preclaim_check=preclaim_check,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--classify", action="store_true")
    mode.add_argument("--invalidate-v1", action="store_true")
    mode.add_argument("--freeze-inputs", type=Path)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--authorize-publication", action="store_true")
    mode.add_argument("--publish-once", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--parent-authority-dir", type=Path, required=True)
    parser.add_argument("--frozen-input-dir", type=Path)
    return parser


def _cli_exception_disposition(
    *,
    namespace: Namespace | None,
    parent_authority_directory: Path | None,
    baseline: Mapping[str, Any] | None,
) -> tuple[str, bool | None, str | None, int]:
    """Classify publication state after a CLI exception without assuming rollback."""

    if namespace is None or parent_authority_directory is None:
        return "stopped_without_write", False, None, 1
    current = _cli_mutation_snapshot(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
    )
    changed_since_entry = (
        baseline is None
        or current is None
        or _canonical_bytes(baseline) != _canonical_bytes(current)
    )
    replacement = classify(
        namespace,
        parent_authority_directory=parent_authority_directory,
    )
    if replacement.state is State.READY:
        if changed_since_entry:
            return "stopped_after_control_write", False, replacement.state.value, 3
        return "stopped_without_write", False, replacement.state.value, 1
    if replacement.state is State.ROLLED_BACK_FAILURE:
        return "stopped_after_control_write", False, replacement.state.value, 3
    if replacement.state is State.COMMITTED:
        return "stopped_after_attempt", True, replacement.state.value, 3
    control_write_observed = (
        changed_since_entry
        or any(
            os.path.lexists(path)
            for path in (namespace.attempt, namespace.success, namespace.failure)
        )
        or bool(replacement.candidates)
    )
    status = "stopped_after_control_write" if control_write_observed else "stopped_ambiguous"
    return status, None, replacement.state.value, 3


def _cli_mutation_snapshot(
    *,
    namespace: Namespace,
    parent_authority_directory: Path,
) -> dict[str, Any] | None:
    """Capture names/existence only so exception reporting can distinguish writes."""

    try:
        parent = _absolute(parent_authority_directory)
        amendment_root = parent.parent
        control_paths = {
            "prior_failure_receipt": namespace.control_root / _PRIOR_FAILURE_RECEIPT_FILENAME,
            "retired_input_bundle_v1": (namespace.control_root / _RETIRED_INPUT_DIRECTORY_NAME),
            "retired_input_invalidation": (
                namespace.control_root / _RETIRED_INPUT_INVALIDATION_FILENAME
            ),
            "frozen_input_bundle_v2": (namespace.control_root / _REPLACEMENT_INPUT_DIRECTORY_NAME),
            "publication_authorization": namespace.publication_authorization,
            "attempt": namespace.attempt,
            "success": namespace.success,
            "failure": namespace.failure,
        }
        return {
            "control_entries": {
                role: os.path.lexists(path) for role, path in sorted(control_paths.items())
            },
            "amendment_entries": tuple(sorted(entry.name for entry in os.scandir(amendment_root))),
        }
    except (OSError, TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    namespace: Namespace | None = None
    parent: Path | None = None
    mutation_baseline: dict[str, Any] | None = None
    try:
        project_root = _absolute(args.project_root)
        parent = args.parent_authority_dir
        if not parent.is_absolute():
            parent = project_root / parent
        namespace = Namespace.for_project(project_root)
        mutation_baseline = _cli_mutation_snapshot(
            namespace=namespace,
            parent_authority_directory=parent,
        )
        if args.invalidate_v1:
            result = publish_retired_input_invalidation_once(
                namespace=namespace,
                parent_authority_directory=parent,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.freeze_inputs is not None:
            output = args.freeze_inputs
            if not output.is_absolute():
                output = project_root / output
            result = freeze_replacement_inputs_once(
                namespace=namespace,
                parent_authority_directory=parent,
                output_directory=output,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.classify:
            state = classify(
                namespace,
                parent_authority_directory=parent,
            )
            print(json.dumps(state.as_dict(), indent=2, sort_keys=True))
            return 0 if state.state is not State.STOP_AMBIGUOUS else 1
        if args.frozen_input_dir is None:
            raise ControlError(
                "live preflight/authorization/publication requires --frozen-input-dir"
            )
        input_directory = args.frozen_input_dir
        if not input_directory.is_absolute():
            input_directory = project_root / input_directory
        if args.preflight_only:
            result = preflight_replacement_inputs(
                namespace=namespace,
                parent_authority_directory=parent,
                input_directory=input_directory,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.authorize_publication:
            result = authorize_replacement_publication_once(
                namespace=namespace,
                parent_authority_directory=parent,
                input_directory=input_directory,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.publish_once:
            publication_result = publish_replacement_authority_once(
                namespace=namespace,
                parent_authority_directory=parent,
                input_directory=input_directory,
            )
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "state": publication_result.state.value,
                        "terminal_marker_path": str(publication_result.marker_path),
                        "terminal_marker_sha256": publication_result.marker_sha256,
                        "authority_directory": (
                            str(publication_result.authority_directory)
                            if publication_result.authority_directory is not None
                            else None
                        ),
                        "automatic_retry_allowed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if publication_result.state is State.COMMITTED else 1
        raise ControlError("unreachable replacement-controller mode")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        status, publication_performed, replacement_state, exit_code = _cli_exception_disposition(
            namespace=namespace,
            parent_authority_directory=parent,
            baseline=mutation_baseline,
        )
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "automatic_retry_allowed": False,
                    "publication_performed": publication_performed,
                    "replacement_state": replacement_state,
                    "error_sha256": hashlib.sha256(
                        f"{type(exc).__name__}: {exc}".encode()
                    ).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
