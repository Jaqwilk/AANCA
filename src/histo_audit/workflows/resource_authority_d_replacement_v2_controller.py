"""Independent replacement-v2 protocol for a future Authority-D publication.

This module deliberately does not import
``resource_authority_d_replacement_controller``.  The consumed replacement-v1
lineage is authenticated from its immutable files and historical identities only.
Importing this module performs no I/O, creates no lock, and starts no subprocess.

The governed progression is:

``QUALIFICATION_REQUIRED -> INPUT_FREEZE_REQUIRED -> AUTHORIZATION_REQUIRED ->
READY -> (ROLLED_BACK_FAILURE | COMMITTED)``.

Every partial, aliased, competing, drifting, or otherwise unrecognised state is
``STOP_AMBIGUOUS``.  State-changing APIs use one overlapping publication-lock
topology, O_EXCL publication, ownership-checked rollback, and no automatic retry.
The schema-v3 amendment API remains constructor-injected for isolated tests; the
public path uses the exact production adapter and fails closed on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import psutil  # type: ignore[import-untyped]

from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    create_directory_no_overwrite,
    publish_bytes_no_overwrite,
    read_file_anchored,
    rollback_owned_publications,
)

PROTOCOL_SCHEMA_VERSION = 2

TERMINAL_QUALIFICATION_FILENAME = (
    "resource_authority_d_replacement_v1_terminal_qualification_v1.json"
)
TERMINAL_QUALIFICATION_POLICY = "resource_authority_d_replacement_v1_terminal_qualification_v1"
TERMINAL_QUALIFICATION_STATUS = "qualified_rolled_back_failure_no_retry"

INPUT_V3_DIRECTORY_NAME = "authority_d_replacement_inputs_v3"
INPUT_V3_FILENAMES = {
    "cnn_correction_receipt": "authority_d_replacement_cnn_correction_receipt.json",
    "frozen_source_receipt": "authority_d_replacement_frozen_source_receipt.json",
    "source_allowlist": "authority_d_replacement_source_allowlist.json",
    "workspace_plan": "authority_d_replacement_workspace_plan.json",
}
INPUT_V3_POLICIES = {
    "cnn_correction_receipt": ("resource_authority_d_replacement_cnn_correction_receipt_v1"),
    "frozen_source_receipt": ("resource_authority_d_replacement_frozen_source_receipt_v3"),
    "source_allowlist": ("resource_authority_d_replacement_exact_18_path_allowlist_v1"),
    "workspace_plan": None,
}

PUBLICATION_AUTHORIZATION_V2_FILENAME = (
    "resource_authority_d_replacement_publication_authorization_v2.json"
)
PUBLICATION_AUTHORIZATION_V2_POLICY = (
    "resource_authority_d_replacement_publication_authorization_v2"
)
MARKER_V2_PREFIX = "resource_authority_d_replacement_v2_publication_"
ATTEMPT_V2_FILENAME = f"{MARKER_V2_PREFIX}attempt.json"
SUCCESS_V2_FILENAME = f"{MARKER_V2_PREFIX}success.json"
FAILURE_V2_FILENAME = f"{MARKER_V2_PREFIX}failure.json"
ATTEMPT_V2_POLICY = "resource_authority_d_replacement_attempt_v2"
SUCCESS_V2_POLICY = "resource_authority_d_replacement_success_v2"
FAILURE_V2_POLICY = "resource_authority_d_replacement_failure_v2"

SCHEMA_V3_AUTHORIZATION_POLICY = "post_outcome_resource_bounded_confirmatory_technical_successor_v3"
FAILURE_LINEAGE_ENVELOPE_FIELDS = frozenset(
    {
        "terminal_qualification_receipt_path",
        "terminal_qualification_receipt_sha256",
        "terminal_qualification_receipt",
    }
)

FRESH_DIAGNOSTIC_POLICY = "resource_authority_d_bounded_fresh_verifier_diagnostic_v1"
FRESH_DIAGNOSTIC_SCHEMA_VERSION = 1
_MAX_STDOUT_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_CONTROL_BYTES = 16 * 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_WAIT_SLICE_SECONDS = 0.05
_CLEANUP_GRACE_SECONDS = 2.0
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

_HISTORICAL_INPUT_V2_DIRECTORY_NAME = "authority_d_replacement_inputs_v2"
_HISTORICAL_INPUT_V2_FILENAMES = dict(INPUT_V3_FILENAMES)
_HISTORICAL_AUTH_V1_FILENAME = "resource_authority_d_replacement_publication_authorization_v1.json"
_HISTORICAL_ATTEMPT_V1_FILENAME = "resource_authority_d_replacement_v1_publication_attempt.json"
_HISTORICAL_SUCCESS_V1_FILENAME = "resource_authority_d_replacement_v1_publication_success.json"
_HISTORICAL_FAILURE_V1_FILENAME = "resource_authority_d_replacement_v1_publication_failure.json"
_HISTORICAL_INVALIDATION_FILENAME = "authority_d_replacement_inputs_v1.invalidation.json"
_HISTORICAL_PRIOR_FAILURE_FILENAME = (
    "resource_authority_d_prior_publication_failure_receipt_v1.json"
)
_HISTORICAL_FAILED_PREFLIGHT_FILENAME = "failed_resource_preflight_20260727T173054.689Z.json"

_AUTHORITY_C_COMPONENT = "20260727T170413.080954Z"
_AMENDMENT_BASELINE = (
    "20260719T011146.248393Z",
    "20260727T133947.089370Z",
    _AUTHORITY_C_COMPONENT,
)
_HISTORICAL_D1_COMPONENT = "20260728T181920.303224Z"

_RUN_STATE_FILENAMES = (
    "integrity_registry.jsonl",
    "registry.csv",
    "run_dispositions.anchor.json",
    "run_dispositions.jsonl",
    "run_stage_attestations.anchor.json",
    "run_stage_attestations.jsonl",
)
_RUN_STATE_PINS = {
    "integrity_registry.jsonl": (
        7370,
        "094d545f7acd0543ac352b7916404c7e2f6b3c44b2ba75f024356efe4ecbd7e3",
    ),
    "registry.csv": (
        8123,
        "c1d8eac3d58f4ea7e42d3cb716c79c007e4ff16109cdf96b71ec722a223f14e3",
    ),
    "run_dispositions.anchor.json": (
        346,
        "ea29ec96e3ab449bb44809114f9c1cc321291dd41c90f1442514f819b5282fed",
    ),
    "run_dispositions.jsonl": (
        2198,
        "c7ee7d5a6a4f27c64b6c8b9f7d3af05d3405e014262f48463ce66b86a6121f07",
    ),
    "run_stage_attestations.anchor.json": (
        352,
        "22732dfa17ff51adf50b7d76633c508111ab26e5ce79654b6237bb9e1d19dd83",
    ),
    "run_stage_attestations.jsonl": (
        5270,
        "eefef3cd970e42a9c39ba5a374d69941e6aac196469c8319ac24483896762553",
    ),
}

_PROTECTED_BINDINGS = {
    "specification": (
        "SPEC.md",
        11275,
        "9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0",
    ),
    "pre_registration": (
        "PRE_REGISTRATION.md",
        32538,
        "7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b",
    ),
    "primary_config": (
        "configs/primary_frozen.yaml",
        28497,
        "0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9",
    ),
    "confirmatory_config": (
        "configs/confirmatory_frozen.yaml",
        16099,
        "4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009",
    ),
}

_AUTHORITY_C_FILE_PINS = {
    "amendment_evidence": (
        "amendment_evidence.json",
        16677,
        "c2531787116e125bdb46e862f6803429c72e5d4766d5127f2cefa693e320912a",
    ),
    "amendment_report": (
        "AMENDMENT.md",
        6834,
        "e2aca95871edaa0fc44abdcabe9ee8b9c1170a50d2ab884860e1d43acf803552",
    ),
    "confirmatory_config": (
        "confirmatory_frozen.yaml",
        13994,
        "783968e8afc132cca0c877aadf953fc68d3c35f606021b3a97ed380478dbad4a",
    ),
    "immutable_marker": (
        ".immutable.json",
        374,
        "49caed80e2e1c07b14a862767ffd5b674c941a7d5c42f7a950ceb902ecee2821",
    ),
    "preregistration": (
        "PRE_REGISTRATION_FROZEN.md",
        32538,
        "7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b",
    ),
    "primary_config": (
        "primary_frozen.yaml",
        28497,
        "0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9",
    ),
    "sha256_manifest": (
        "sha256_manifest.json",
        1394,
        "4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156",
    ),
    "source_tree_manifest": (
        "source_tree_manifest.json",
        18678,
        "03bcc6020e3be5a22fe257c45820e4e8ebece3ce471c2b6cecff0e3e9419fc66",
    ),
}

_AUTHORITY_C_ARTIFACT_ROOT_SHA256 = (
    "57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627"
)
_AUTHORITY_C_MANIFEST_SHA256 = "4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156"
_HISTORICAL_RUN_STATE_SHA256 = "5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4"
_HISTORICAL_CONTROLLER_SIZE_BYTES = 216288
_HISTORICAL_CONTROLLER_SHA256 = "cbea3c3536dbad729383c96e0ef602042c7e3c4e000f9b0cb79e50c13b2ced58"
_DIAGNOSED_FIXED_LEGACY_CONTROLLER_SIZE_BYTES = 218766
_DIAGNOSED_FIXED_LEGACY_CONTROLLER_SHA256 = (
    "e20278105b6ea4e2786713c64d9e8cf7bb06d9e4c8155f35a46861e72cb67b5f"
)
_HISTORICAL_AUTHORIZATION_SHA256 = (
    "4c892f7e518964a46569290e1a486d7f7e193121ed870522895946413dbee565"
)
_HISTORICAL_ATTEMPT_SHA256 = "e602993753949ecbd5bfe3dfd9ba77d1890d63ae6232db9db6d66caff48e3ace"
_HISTORICAL_FAILURE_SHA256 = "e66305dac9a2c1b59d5cb554081470c1947b939d8a07ade3cf77046f0e353b12"
_HISTORICAL_ATTEMPT_ID = "c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86"
_HISTORICAL_TECHNICAL_AUTHORIZATION_SHA256 = (
    "886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8"
)
_HISTORICAL_INTENT_SHA256 = "9c5018d37a4a9f4d26dd40d1b4c3eb902c97601459720085ae12bc91d4e4e347"
_HISTORICAL_PREFLIGHT_FINGERPRINT_SHA256 = (
    "9e828dd7652a2be3c3ecee798fae9f7b1b1167875129e7c3fed581443550270a"
)
_HISTORICAL_FAILURE_ERROR_TYPE = "FreshVerifierError"
_HISTORICAL_FAILURE_ERROR_TEXT = "FreshVerifierError: fresh verifier process did not exit cleanly"
_HISTORICAL_FAILURE_ERROR_TYPE_SHA256 = (
    "431fd4d500d504c9f02a7e5f505eb2065cf1612fc36b6474af72b89e5d3a8ffd"
)
_HISTORICAL_FAILURE_ERROR_SHA256 = (
    "a6a9e199b4080a911e7f07f3243e3982049a85fc18ba712883de9f9cc099e1fb"
)
_HISTORICAL_INPUT_V2_RECORDS_SHA256 = (
    "8de462490b465d639badefe4cfd411773c4071cb028929e8b0faf73755333aa1"
)
_HISTORICAL_V2_SOURCE_ROOT_SHA256 = (
    "c81e7a01bc6949d82d5cb76a206776dde4ceda47c1506a71bd8edf736649bd75"
)
_HISTORICAL_V2_SOURCE_MANIFEST_SHA256 = (
    "49857657c88249999278b4d9f51fa70cc622b9c23fb1e16a800c7e7f9e8a1a0f"
)
_HISTORICAL_V2_SOURCE_DELTA_SHA256 = (
    "82acb5a60100141a2c54f2094b8a438fd725bcfe4227c132e4dff185608d7217"
)

_AUTHORITY_C_SOURCE_PINS = {
    "root_sha256": "1179f91725a3027c0397e87691774377bbd4ba5469d588390c72b0b88515547b",
    "manifest_sha256": ("03bcc6020e3be5a22fe257c45820e4e8ebece3ce471c2b6cecff0e3e9419fc66"),
    "artifact_count": 101,
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
    "src/histo_audit/workflows/resource_authority_d_replacement_v2_controller.py": "added",
    "src/histo_audit/workflows/study_gates.py": "modified",
}
_CNN_SEMANTIC_EQUIVALENCE_POLICY = "resource_authority_d_replacement_cnn_semantic_equivalence_v1"
_LIVE_PREFLIGHT_V2_POLICY = "resource_authority_d_replacement_live_preflight_v2"
_TECHNICAL_SUCCESSOR_PURPOSE = "resource_bounded_confirmatory_technical_successor"
_OUTCOMES_INSPECTED_AT = datetime.fromisoformat("2026-07-27T10:57:07+00:00")
_AMENDMENT_REASON = (
    "Publish one independently authorized schema-v3 technical successor to resource "
    "authority C after the consumed replacement-v1 publication was terminally "
    "qualified as a process-boundary rollback failure; preserve the exact "
    "outcome-blind CNN provenance correction and capacity-v3 indexed workspace "
    "without changing the scientific profile."
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


class ControlError(RuntimeError):
    """Base fail-closed replacement-v2 error."""


class AmbiguousStateError(ControlError):
    """The observed state cannot be safely classified or repaired."""


class SchemaV3UnavailableError(ControlError):
    """The schema-v3 amendment integration is not installed."""


class FreshVerifierError(ControlError):
    """Fresh verifier execution or payload validation failed closed."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic) if diagnostic is not None else None


class State(StrEnum):
    QUALIFICATION_REQUIRED = "qualification_required"
    INPUT_FREEZE_REQUIRED = "input_freeze_required"
    AUTHORIZATION_REQUIRED = "authorization_required"
    READY = "ready"
    ROLLED_BACK_FAILURE = "rolled_back_failure"
    COMMITTED = "committed"
    STOP_AMBIGUOUS = "stop_ambiguous"


@dataclass(frozen=True, slots=True)
class FilePin:
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class HistoricalPins:
    input_v2: Mapping[str, FilePin]
    authorization_v1: FilePin
    attempt_v1: FilePin
    failure_v1: FilePin
    invalidation: FilePin
    prior_failure: FilePin
    failed_preflight: FilePin
    controller: FilePin
    run_state_sha256: str
    attempt_id: str
    technical_authorization_sha256: str
    intent_sha256: str
    preflight_fingerprint_sha256: str
    failure_error_type_sha256: str
    failure_error_sha256: str
    input_v2_records_sha256: str


DEFAULT_HISTORICAL_PINS = HistoricalPins(
    input_v2={
        "cnn_correction_receipt": FilePin(
            4452,
            "0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e",
        ),
        "frozen_source_receipt": FilePin(
            3943,
            "1acbcfd44b3f95d6387d7da573786547a6c1ff5dcd0d05b4198d311fbe813605",
        ),
        "source_allowlist": FilePin(
            3903,
            "397fac0240f36fb598095e7605dae770b55faf114d4c692e555a7101fd47c369",
        ),
        "workspace_plan": FilePin(
            12186,
            "d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b",
        ),
    },
    authorization_v1=FilePin(9396, _HISTORICAL_AUTHORIZATION_SHA256),
    attempt_v1=FilePin(3420, _HISTORICAL_ATTEMPT_SHA256),
    failure_v1=FilePin(924, _HISTORICAL_FAILURE_SHA256),
    invalidation=FilePin(
        6550,
        "0b9af7cdb9ca3fcb60c8dd6c123eda22f13631c1188ff390cd9421998e28e997",
    ),
    prior_failure=FilePin(
        11413,
        "2b46f11d1580a6469715a525c0738d39fb3ae0f74f542e142ecd293ae7beed00",
    ),
    failed_preflight=FilePin(
        1994,
        "e308aa0089a84caaca3f0722711e623579372636d47e161d9f32ca5a71f8c6eb",
    ),
    controller=FilePin(
        _HISTORICAL_CONTROLLER_SIZE_BYTES,
        _HISTORICAL_CONTROLLER_SHA256,
    ),
    run_state_sha256=_HISTORICAL_RUN_STATE_SHA256,
    attempt_id=_HISTORICAL_ATTEMPT_ID,
    technical_authorization_sha256=_HISTORICAL_TECHNICAL_AUTHORIZATION_SHA256,
    intent_sha256=_HISTORICAL_INTENT_SHA256,
    preflight_fingerprint_sha256=_HISTORICAL_PREFLIGHT_FINGERPRINT_SHA256,
    failure_error_type_sha256=_HISTORICAL_FAILURE_ERROR_TYPE_SHA256,
    failure_error_sha256=_HISTORICAL_FAILURE_ERROR_SHA256,
    input_v2_records_sha256=_HISTORICAL_INPUT_V2_RECORDS_SHA256,
)


@dataclass(frozen=True, slots=True)
class Namespace:
    control_root: Path

    @classmethod
    def for_project(cls, project_root: str | Path) -> Namespace:
        return cls(_absolute(project_root) / "artifacts" / "resource_control")

    @property
    def terminal_qualification(self) -> Path:
        return self.control_root / TERMINAL_QUALIFICATION_FILENAME

    @property
    def input_v3(self) -> Path:
        return self.control_root / INPUT_V3_DIRECTORY_NAME

    @property
    def authorization_v2(self) -> Path:
        return self.control_root / PUBLICATION_AUTHORIZATION_V2_FILENAME

    @property
    def attempt_v2(self) -> Path:
        return self.control_root / ATTEMPT_V2_FILENAME

    @property
    def success_v2(self) -> Path:
        return self.control_root / SUCCESS_V2_FILENAME

    @property
    def failure_v2(self) -> Path:
        return self.control_root / FAILURE_V2_FILENAME


@dataclass(frozen=True, slots=True)
class HistoricalSnapshot:
    payload: dict[str, Any]
    authorization: dict[str, Any]
    attempt: dict[str, Any]
    failure: dict[str, Any]
    intended_authority: Path


@dataclass(frozen=True, slots=True)
class Classification:
    state: State
    reason: str
    candidates: tuple[Path, ...] = ()
    terminal_qualification_sha256: str | None = None
    input_v3_sha256: str | None = None
    authorization_v2_sha256: str | None = None
    attempt_v2_sha256: str | None = None
    success_v2_sha256: str | None = None
    failure_v2_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        publication_performed: bool | None
        if self.state is State.COMMITTED:
            publication_performed = True
        elif self.state is State.STOP_AMBIGUOUS:
            publication_performed = None
        else:
            publication_performed = False
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "policy": "resource_authority_d_replacement_state_classifier_v2",
            "state": self.state.value,
            "reason": self.reason,
            "candidate_directories": [str(path) for path in self.candidates],
            "terminal_qualification_receipt_sha256": self.terminal_qualification_sha256,
            "input_v3_manifest_sha256": self.input_v3_sha256,
            "publication_authorization_v2_sha256": self.authorization_v2_sha256,
            "attempt_v2_sha256": self.attempt_v2_sha256,
            "success_v2_sha256": self.success_v2_sha256,
            "failure_v2_sha256": self.failure_v2_sha256,
            "automatic_retry_allowed": False,
            "publication_performed": publication_performed,
        }


@dataclass(frozen=True, slots=True)
class AuthorityPins:
    directory: Path
    parent_directory: Path
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    authorization_sha256: str
    intent_sha256: str
    chain_depth: int


class SchemaV3API(Protocol):
    """Narrow integration boundary supplied by the schema-v3 amendment owner."""

    def canonicalize_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def create_authority(
        self,
        *,
        authorization: Mapping[str, Any],
        post_publication_check: Callable[[Any], None],
    ) -> Any: ...

    def authority_pins(self, published: Any) -> AuthorityPins: ...

    def verify_committed(
        self,
        authority: Path,
        *,
        expected: AuthorityPins,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> None: ...


class UnavailableSchemaV3API:
    """Default adapter: no schema-v3 authorization or publication is possible."""

    @staticmethod
    def _unavailable() -> SchemaV3UnavailableError:
        return SchemaV3UnavailableError(
            "schema-v3 technical-successor API is unavailable; inject a qualified adapter"
        )

    def canonicalize_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> dict[str, Any]:
        del authorization, replacement_publication_failure_lineage
        raise self._unavailable()

    def create_authority(
        self,
        *,
        authorization: Mapping[str, Any],
        post_publication_check: Callable[[Any], None],
    ) -> Any:
        del authorization, post_publication_check
        raise self._unavailable()

    def authority_pins(self, published: Any) -> AuthorityPins:
        del published
        raise self._unavailable()

    def verify_committed(
        self,
        authority: Path,
        *,
        expected: AuthorityPins,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> None:
        del authority, expected, replacement_publication_failure_lineage
        raise self._unavailable()


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(_absolute(left)))) == os.path.normcase(
        os.path.normpath(str(_absolute(right)))
    )


def _exact_path_text(value: object, expected: str | Path, role: str) -> str:
    """Require the exact canonical lexical spelling stored in governed evidence."""

    canonical = str(_absolute(expected))
    if type(value) is not str or value != canonical:
        raise ControlError(f"{role} must be exactly {canonical!r}")
    return value


def _sha(value: object, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControlError(f"{role} must be one lowercase SHA-256")
    return value


def _positive_int(value: object, role: str) -> int:
    if type(value) is not int or value <= 0:
        raise ControlError(f"{role} must be a positive exact integer")
    return value


def _nonnegative_int(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        raise ControlError(f"{role} must be a nonnegative exact integer")
    return value


def _canonical_timestamp(value: object, role: str) -> str:
    if type(value) is not str:
        raise ControlError(f"{role} must be canonical UTC text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ControlError(f"{role} must use six-digit microsecond UTC precision") from exc
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if canonical != value:
        raise ControlError(f"{role} is not canonical")
    return canonical


def _timestamp(moment: datetime | None = None) -> str:
    value = moment or datetime.now(UTC)
    if value.tzinfo is None:
        raise ControlError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


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


def _exact_dict(value: object, fields: frozenset[str], role: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ControlError(f"{role} must contain exactly {sorted(fields)!r}")
    return value


def _real_directory(path: str | Path, role: str) -> Path:
    candidate = _absolute(path)
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ControlError(f"{role} is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE
        or not _same(candidate, resolved)
    ):
        raise ControlError(f"{role} must be one real non-reparse directory")
    return resolved


def _read_bytes(path: Path, role: str, *, max_bytes: int = _MAX_CONTROL_BYTES) -> bytes:
    try:
        return read_file_anchored(
            path,
            require_single_link=True,
            max_bytes=max_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControlError(f"{role} failed anchored bounded readback") from exc


def _file_record(path: Path, role: str) -> dict[str, Any]:
    encoded = _read_bytes(path, role)
    return {
        "path": str(_absolute(path)),
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _read_pinned_json(path: Path, pin: FilePin, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = _read_bytes(path, role)
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) != pin.size_bytes or digest != pin.sha256:
        raise ControlError(f"{role} differs from its immutable historical pin")
    payload = _strict_json_object(encoded, role)
    if encoded != _canonical_bytes(payload):
        raise ControlError(f"{role} bytes are not canonical")
    return payload, {
        "path": str(_absolute(path)),
        "size_bytes": len(encoded),
        "sha256": digest,
    }


def _record_matches(record: object, *, path: Path, pin: FilePin, role: str) -> None:
    payload = _exact_dict(
        record,
        frozenset({"path", "size_bytes", "sha256"}),
        role,
    )
    _exact_path_text(payload["path"], path, f"{role} path")
    if (
        type(payload["size_bytes"]) is not int
        or payload["size_bytes"] != pin.size_bytes
        or _sha(payload["sha256"], f"{role} SHA-256") != pin.sha256
    ):
        raise ControlError(f"{role} differs from its exact historical binding")


def _project_root(namespace: Namespace) -> Path:
    control = _real_directory(namespace.control_root, "replacement-v2 control root")
    expected = control.parent.parent / "artifacts" / "resource_control"
    if not _same(control, expected):
        raise ControlError("replacement-v2 namespace is outside the project layout")
    return control.parent.parent


def _require_parent(namespace: Namespace, parent: str | Path) -> tuple[Path, Path]:
    project = _project_root(namespace)
    authority = _real_directory(parent, "Authority C")
    expected = project / "artifacts" / "preregistration_amendments" / _AUTHORITY_C_COMPONENT
    if not _same(authority, expected):
        raise ControlError("replacement-v2 must use exact Authority C")
    return project, authority


def _historical_paths(namespace: Namespace, parent: Path) -> dict[str, Path]:
    root = _absolute(namespace.control_root)
    input_v2 = root / _HISTORICAL_INPUT_V2_DIRECTORY_NAME
    return {
        "input_v2": input_v2,
        **{
            f"input_v2_{role}": input_v2 / filename
            for role, filename in _HISTORICAL_INPUT_V2_FILENAMES.items()
        },
        "authorization_v1": root / _HISTORICAL_AUTH_V1_FILENAME,
        "attempt_v1": root / _HISTORICAL_ATTEMPT_V1_FILENAME,
        "success_v1": root / _HISTORICAL_SUCCESS_V1_FILENAME,
        "failure_v1": root / _HISTORICAL_FAILURE_V1_FILENAME,
        "invalidation": root / _HISTORICAL_INVALIDATION_FILENAME,
        "prior_failure": root / _HISTORICAL_PRIOR_FAILURE_FILENAME,
        "failed_preflight": root / _HISTORICAL_FAILED_PREFLIGHT_FILENAME,
        "historical_d1": parent.parent / _HISTORICAL_D1_COMPONENT,
    }


def _stable_amendment_inventory(
    amendment_root: Path,
    *,
    allowed_extra: Path | None = None,
) -> tuple[str, ...]:
    expected = set(_AMENDMENT_BASELINE)
    if allowed_extra is not None:
        if allowed_extra.parent != amendment_root:
            raise ControlError("allowed Authority-D peer is outside the amendment root")
        expected.add(allowed_extra.name)

    def scan() -> tuple[str, ...]:
        try:
            entries = tuple(sorted(entry.name for entry in os.scandir(amendment_root)))
        except OSError as exc:
            raise ControlError("amendment-root inventory scan failed") from exc
        return entries

    first = scan()
    if set(first) != expected or len(first) != len(expected):
        raise ControlError("amendment-root inventory is not exact A/P/C[/D2]")
    for name in first:
        _real_directory(amendment_root / name, f"amendment authority {name}")
    second = scan()
    if second != first:
        raise ControlError("amendment-root inventory changed between scans")
    return first


def _authority_c_receipt(parent: Path) -> dict[str, Any]:
    observed_names = tuple(sorted(entry.name for entry in os.scandir(parent)))
    expected_names = tuple(sorted(value[0] for value in _AUTHORITY_C_FILE_PINS.values()))
    if observed_names != expected_names:
        raise ControlError("Authority C does not have the exact eight-file inventory")
    records: dict[str, dict[str, Any]] = {}
    for role, (filename, size_bytes, sha256) in _AUTHORITY_C_FILE_PINS.items():
        record = _file_record(parent / filename, f"Authority C {role}")
        if record["size_bytes"] != size_bytes or record["sha256"] != sha256:
            raise ControlError(f"Authority C {role} differs from its immutable pin")
        records[role] = record
    evidence = _strict_json_object(
        _read_bytes(parent / "amendment_evidence.json", "Authority C amendment evidence"),
        "Authority C amendment evidence",
    )
    if (
        evidence.get("schema_version") != 4
        or evidence.get("chain_depth") != 3
        or records["sha256_manifest"]["sha256"] != _AUTHORITY_C_MANIFEST_SHA256
    ):
        raise ControlError("Authority C schema/depth/manifest identity differs")
    return {
        "directory": str(parent),
        "schema_version": 4,
        "chain_depth": 3,
        "artifact_root_sha256": _AUTHORITY_C_ARTIFACT_ROOT_SHA256,
        "sha256_manifest_sha256": _AUTHORITY_C_MANIFEST_SHA256,
        "flat_file_count": 8,
        "files": records,
        "integrity_verified": True,
    }


def _protected_receipt(project: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role, (relative_path, size_bytes, sha256) in _PROTECTED_BINDINGS.items():
        record = _file_record(project / relative_path, f"protected {role}")
        if record["size_bytes"] != size_bytes or record["sha256"] != sha256:
            raise ControlError(f"protected {role} differs from its immutable pin")
        records[role] = record
    return records


def _run_state_receipt(project: Path, expected_sha256: str) -> dict[str, Any]:
    root = _real_directory(project / "artifacts" / "runs", "run-state root")
    files: dict[str, dict[str, Any]] = {}
    for filename in _RUN_STATE_FILENAMES:
        record = _file_record(root / filename, f"run-state {filename}")
        size_bytes, sha256 = _RUN_STATE_PINS[filename]
        _record_matches(
            record,
            path=root / filename,
            pin=FilePin(size_bytes, sha256),
            role=f"run-state {filename}",
        )
        files[filename] = record
    digest = _compact_sha256(
        {filename: files[filename]["sha256"] for filename in _RUN_STATE_FILENAMES}
    )
    if digest != expected_sha256:
        raise ControlError("run-state aggregate differs from the historical lineage")
    return {"root": str(root), "files": files, "sha256": digest}


def _input_v2_records(
    paths: Mapping[str, Path],
    pins: HistoricalPins,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    input_root = _real_directory(paths["input_v2"], "historical input-v2 bundle")
    observed = tuple(sorted(entry.name for entry in os.scandir(input_root)))
    expected = tuple(sorted(_HISTORICAL_INPUT_V2_FILENAMES.values()))
    if observed != expected:
        raise ControlError("historical input-v2 inventory is not exact")
    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role in _HISTORICAL_INPUT_V2_FILENAMES:
        payload, record = _read_pinned_json(
            paths[f"input_v2_{role}"],
            pins.input_v2[role],
            f"historical input-v2 {role}",
        )
        payloads[role] = payload
        records[role] = record
    if _compact_sha256(records) != pins.input_v2_records_sha256:
        raise ControlError("historical input-v2 role-record root differs")
    return payloads, records


def _require_historical_controller_attestations(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    invalidation: Mapping[str, Any],
    input_payloads: Mapping[str, Mapping[str, Any]],
    expected_controller_path: Path,
    pins: HistoricalPins,
) -> str:
    try:
        controller = authorization["preflight"]["contract"]["controller"]
    except (KeyError, TypeError) as exc:
        raise ControlError("historical authorization lacks its controller binding") from exc
    old_path = _absolute(expected_controller_path)
    _exact_path_text(
        controller.get("path"),
        old_path,
        "historical authorization controller path",
    )
    _exact_path_text(
        attempt.get("controller_path"),
        old_path,
        "historical attempt controller path",
    )
    if (
        controller.get("size_bytes") != pins.controller.size_bytes
        or controller.get("sha256") != pins.controller.sha256
        or attempt.get("controller_size_bytes") != pins.controller.size_bytes
        or attempt.get("controller_sha256") != pins.controller.sha256
    ):
        raise ControlError("historical controller identity is not exact in auth/A1")
    _record_matches(
        invalidation.get("corrected_controller"),
        path=old_path,
        pin=pins.controller,
        role="historical invalidation controller",
    )
    frozen_source = input_payloads["frozen_source_receipt"]
    _exact_path_text(
        frozen_source.get("controller_path"),
        old_path,
        "historical frozen-source controller path",
    )
    if (
        type(frozen_source.get("controller_size_bytes")) is not int
        or frozen_source.get("controller_size_bytes") != pins.controller.size_bytes
        or frozen_source.get("controller_sha256") != pins.controller.sha256
    ):
        raise ControlError("historical frozen-source controller identity differs")
    allowlist = input_payloads["source_allowlist"].get("records")
    if type(allowlist) is not list:
        raise ControlError("historical v2 source allowlist has no exact record list")
    matches = [
        item
        for item in allowlist
        if type(item) is dict
        and item.get("path")
        == "src/histo_audit/workflows/resource_authority_d_replacement_controller.py"
    ]
    if len(matches) != 1:
        raise ControlError("historical source allowlist lacks one old-controller record")
    record = matches[0]
    if (
        record.get("change_kind") != "added"
        or record.get("size_bytes") != pins.controller.size_bytes
        or record.get("sha256") != pins.controller.sha256
    ):
        raise ControlError("historical source allowlist old-controller identity differs")
    current = _file_record(old_path, "live legacy controller")
    if (
        current["size_bytes"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SIZE_BYTES
        or current["sha256"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SHA256
    ):
        raise ControlError("live legacy controller differs from the diagnosed fixed bytes")
    return str(old_path)


def _historical_terminal_payloads(
    namespace: Namespace,
    *,
    parent_authority_directory: str | Path,
    pins: HistoricalPins,
) -> tuple[
    Path,
    Path,
    dict[str, Path],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    project, parent = _require_parent(namespace, parent_authority_directory)
    paths = _historical_paths(namespace, parent)
    input_payloads, input_records = _input_v2_records(paths, pins)
    authorization, authorization_record = _read_pinned_json(
        paths["authorization_v1"],
        pins.authorization_v1,
        "historical publication authorization-v1",
    )
    attempt, attempt_record = _read_pinned_json(
        paths["attempt_v1"],
        pins.attempt_v1,
        "historical replacement-v1 attempt",
    )
    failure, failure_record = _read_pinned_json(
        paths["failure_v1"],
        pins.failure_v1,
        "historical replacement-v1 failure",
    )
    invalidation, _invalidation_record = _read_pinned_json(
        paths["invalidation"],
        pins.invalidation,
        "historical v1-input invalidation",
    )
    if os.path.lexists(paths["success_v1"]):
        raise ControlError("historical replacement-v1 success marker must be absent")
    if os.path.lexists(paths["historical_d1"]):
        raise ControlError("historical intended Authority D1 must be absent")

    if (
        authorization.get("schema_version") != 1
        or authorization.get("policy")
        != "resource_authority_d_replacement_publication_authorization_v1"
        or authorization.get("status") != "authorized_for_one_attempt"
        or authorization.get("max_attempt_count") != 1
        or authorization.get("automatic_retry_allowed") is not False
        or authorization.get("authorized_attempt_id") != pins.attempt_id
    ):
        raise ControlError("historical publication authorization-v1 fixed policy differs")
    publication = authorization.get("publication")
    preflight = authorization.get("preflight")
    if type(publication) is not dict or type(preflight) is not dict:
        raise ControlError("historical authorization lacks publication/preflight")
    contract = preflight.get("contract")
    if type(contract) is not dict:
        raise ControlError("historical authorization lacks its preflight contract")
    technical = contract.get("technical_successor")
    _exact_path_text(
        publication.get("parent_authority_directory"),
        parent,
        "historical authorization parent path",
    )
    _exact_path_text(
        publication.get("intended_authority_directory"),
        paths["historical_d1"],
        "historical authorization destination path",
    )
    if (
        type(technical) is not dict
        or technical.get("authorization_sha256") != pins.technical_authorization_sha256
        or technical.get("intent_sha256") != pins.intent_sha256
        or preflight.get("preflight_fingerprint_sha256") != pins.preflight_fingerprint_sha256
        or contract.get("run_state", {}).get("sha256") != pins.run_state_sha256
    ):
        raise ControlError("historical authorization lineage links differ")
    frozen_bundle = contract.get("frozen_input_bundle")
    if type(frozen_bundle) is not dict:
        raise ControlError("historical authorization does not bind exact input-v2")
    _exact_path_text(
        frozen_bundle.get("directory"),
        paths["input_v2"],
        "historical authorization input-v2 directory",
    )
    for role in _HISTORICAL_INPUT_V2_FILENAMES:
        _record_matches(
            frozen_bundle.get(role),
            path=paths[f"input_v2_{role}"],
            pin=pins.input_v2[role],
            role=f"historical authorization input-v2 {role}",
        )

    _exact_path_text(
        attempt.get("parent_authority_directory"),
        parent,
        "historical attempt parent path",
    )
    _exact_path_text(
        attempt.get("intended_authority_directory"),
        paths["historical_d1"],
        "historical attempt destination path",
    )
    if (
        attempt.get("schema_version") != 1
        or attempt.get("policy") != "resource_authority_d_replacement_attempt_v1"
        or attempt.get("status") != "claimed"
        or attempt.get("attempt_id") != pins.attempt_id
        or attempt.get("max_attempt_count") != 1
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("publication_authorization_receipt_sha256") != pins.authorization_v1.sha256
        or attempt.get("preflight_fingerprint_sha256") != pins.preflight_fingerprint_sha256
        or attempt.get("authorization_sha256") != pins.technical_authorization_sha256
        or attempt.get("intent_sha256") != pins.intent_sha256
        or attempt.get("run_state_sha256") != pins.run_state_sha256
    ):
        raise ControlError("historical replacement-v1 attempt links differ")
    _exact_path_text(
        failure.get("parent_authority_directory"),
        parent,
        "historical failure parent path",
    )
    _exact_path_text(
        failure.get("intended_authority_directory"),
        paths["historical_d1"],
        "historical failure destination path",
    )
    if (
        failure.get("schema_version") != 1
        or failure.get("policy") != "resource_authority_d_replacement_failure_v1"
        or failure.get("status") != "rolled_back_failure_no_retry"
        or failure.get("automatic_retry_allowed") is not False
        or failure.get("attempt_id") != pins.attempt_id
        or failure.get("attempt_marker_sha256") != pins.attempt_v1.sha256
        or failure.get("run_state_sha256") != pins.run_state_sha256
        or failure.get("error_type_sha256") != pins.failure_error_type_sha256
        or failure.get("error_sha256") != pins.failure_error_sha256
        or failure.get("authority_absent_after_rollback") is not True
    ):
        raise ControlError("historical replacement-v1 failure links differ")
    _require_historical_controller_attestations(
        authorization=authorization,
        attempt=attempt,
        invalidation=invalidation,
        input_payloads=input_payloads,
        expected_controller_path=(
            project
            / "src"
            / "histo_audit"
            / "workflows"
            / "resource_authority_d_replacement_controller.py"
        ),
        pins=pins,
    )
    return (
        project,
        parent,
        paths,
        input_payloads,
        input_records,
        {
            "authorization": authorization,
            "authorization_record": authorization_record,
        },
        {"attempt": attempt, "attempt_record": attempt_record},
        {"failure": failure, "failure_record": failure_record},
    )


def _historical_support_records(
    *,
    project: Path,
    parent: Path,
    paths: Mapping[str, Path],
    pins: HistoricalPins,
    input_records: Mapping[str, Mapping[str, Any]],
    authorization_record: Mapping[str, Any],
    attempt_record: Mapping[str, Any],
    failure_record: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    invalidation_payload, invalidation_record = _read_pinned_json(
        paths["invalidation"],
        pins.invalidation,
        "historical v1-input invalidation",
    )
    prior_payload, prior_record = _read_pinned_json(
        paths["prior_failure"],
        pins.prior_failure,
        "historical prior-publication failure",
    )
    failed_payload, failed_record = _read_pinned_json(
        paths["failed_preflight"],
        pins.failed_preflight,
        "historical failed preflight",
    )
    corrected_controller = invalidation_payload.get("corrected_controller")
    _record_matches(
        corrected_controller,
        path=(
            project
            / "src"
            / "histo_audit"
            / "workflows"
            / "resource_authority_d_replacement_controller.py"
        ),
        pin=pins.controller,
        role="historical invalidation consumed controller",
    )
    del prior_payload, failed_payload
    protected = _protected_receipt(project)
    authority_c = _authority_c_receipt(parent)
    run_state = _run_state_receipt(project, pins.run_state_sha256)
    ordered_reads: list[dict[str, Any]] = [
        {"role": "publication_authorization_receipt", **authorization_record},
        {"role": "publication_attempt_marker", **attempt_record},
        {"role": "publication_failure_marker", **failure_record},
        {"role": "retired_v1_invalidation_receipt", **invalidation_record},
        {"role": "prior_publication_failure_receipt", **prior_record},
        {"role": "failed_preflight_receipt", **failed_record},
    ]
    ordered_reads.extend(
        {"role": f"v2_{role}", **input_records[role]}
        for role in (
            "cnn_correction_receipt",
            "frozen_source_receipt",
            "source_allowlist",
            "workspace_plan",
        )
    )
    ordered_reads.extend(
        {"role": f"protected_{role}", **protected[role]}
        for role in (
            "specification",
            "pre_registration",
            "primary_config",
            "confirmatory_config",
        )
    )
    ordered_reads.extend(
        {"role": f"authority_c_{role}", **authority_c["files"][role]}
        for role in (
            "amendment_evidence",
            "amendment_report",
            "confirmatory_config",
            "immutable_marker",
            "preregistration",
            "primary_config",
            "sha256_manifest",
            "source_tree_manifest",
        )
    )
    ordered_reads.extend(
        {"role": f"run_state_{filename}", **run_state["files"][filename]}
        for filename in _RUN_STATE_FILENAMES
    )
    if len(ordered_reads) != 28:
        raise ControlError("terminal qualification read set is not exactly 28 records")
    return (
        {
            "failed_preflight_receipt": failed_record,
            "prior_failure_receipt": prior_record,
            "retired_input_invalidation_receipt": invalidation_record,
        },
        authority_c,
        {"protected": protected, "run_state": run_state},
        ordered_reads,
    )


def _legacy_topology_locks(
    namespace: Namespace,
    *,
    parent: Path,
) -> tuple[ExclusiveBundlePublicationLock, ...]:
    paths = _historical_paths(namespace, parent)
    invalidation = ExclusiveBundlePublicationLock(
        (
            paths["invalidation"],
            paths["input_v2"],
            paths["authorization_v1"],
            paths["attempt_v1"],
            paths["success_v1"],
            paths["failure_v1"],
        ),
        role="historical replacement-v1 invalidation topology",
    )
    freeze = ExclusiveBundlePublicationLock(
        (
            paths["input_v2"],
            *(
                paths[f"input_v2_{role}"]
                for role in (
                    "cnn_correction_receipt",
                    "frozen_source_receipt",
                    "source_allowlist",
                    "workspace_plan",
                )
            ),
        ),
        role="historical replacement-v1 freeze topology",
    )
    protocol = ExclusiveBundlePublicationLock(
        (
            paths["input_v2"],
            paths["authorization_v1"],
            paths["attempt_v1"],
            paths["success_v1"],
            paths["failure_v1"],
        ),
        role="historical replacement-v1 authorization/protocol topology",
    )
    creator = ExclusiveBundlePublicationLock(
        (parent, paths["historical_d1"]),
        role="historical Authority-C-to-D1 amendment-creator topology",
    )
    locks = (invalidation, freeze, protocol, creator)
    if tuple(len(lock.lock_paths) for lock in locks) != (7, 6, 6, 3):
        raise ControlError("historical lock topology is not exactly 7+6+6+3 checks")
    unique = {path for lock in locks for path in lock.lock_paths}
    if len(unique) != 16:
        raise ControlError("historical lock topology is not exactly 16 unique registry paths")
    return locks


def _legacy_scoped_lock_paths(namespace: Namespace, *, parent: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                path
                for lock in _legacy_topology_locks(namespace, parent=parent)
                for path in lock.lock_paths
            },
            key=lambda path: os.path.normcase(str(path)),
        )
    )


def _scan_present_paths(paths: Sequence[Path]) -> list[str]:
    present: list[str] = []
    for path in paths:
        try:
            exists = os.path.lexists(path)
        except OSError as exc:
            raise ControlError("scoped publication-lock scan failed") from exc
        if exists:
            present.append(str(_absolute(path)))
    return present


def _protocol_lock_paths(namespace: Namespace, *, parent: Path) -> tuple[Path, ...]:
    historical = _historical_paths(namespace, parent)
    return (
        namespace.terminal_qualification,
        namespace.input_v3,
        namespace.authorization_v2,
        namespace.attempt_v2,
        namespace.success_v2,
        namespace.failure_v2,
        historical["input_v2"],
        historical["authorization_v1"],
        historical["attempt_v1"],
        historical["success_v1"],
        historical["failure_v1"],
    )


def _protocol_lock(
    namespace: Namespace, *, parent: Path, role: str
) -> ExclusiveBundlePublicationLock:
    return ExclusiveBundlePublicationLock(
        _protocol_lock_paths(namespace, parent=parent),
        role=role,
    )


def _require_legacy_lock_state_under_protocol_lock(
    *,
    legacy_paths: Sequence[Path],
    owned_locks: Sequence[ExclusiveBundlePublicationLock],
) -> None:
    for lock in owned_locks:
        lock.assert_owned()
    owned = {_absolute(path) for lock in owned_locks for path in lock.lock_paths}
    for path in legacy_paths:
        absolute = _absolute(path)
        if absolute in owned:
            if not os.path.lexists(absolute):
                raise AmbiguousStateError("owned overlapping legacy target lock disappeared")
        elif os.path.lexists(absolute):
            raise AmbiguousStateError("foreign historical replacement lock appeared")
    for lock in owned_locks:
        lock.assert_owned()


_PROCESS_QUERY_METHOD = "windows_cim_process_command_line_query_v1"
_LOCK_SCAN_METHOD = "two_pass_scoped_lock_path_scan_v1"
_PROCESS_TOKENS = (
    "resource_authority_d_replacement",
    "resource-bounded-sensitivity",
    "resource_bounded_runner",
    " -m histo_audit experiment primary ",
    " -m histo_audit experiment confirmatory ",
)


def _windows_process_quiescence(observer_pid: int) -> dict[str, Any]:
    if os.name != "nt":
        raise ControlError("terminal qualification process query requires Windows CIM")
    _positive_int(observer_pid, "process-query observer PID")
    token_filter = " -or ".join(f"$_.CommandLine -like '*{token}*'" for token in _PROCESS_TOKENS)
    script = (
        "$ErrorActionPreference='Stop';"
        f"$observer={observer_pid};"
        "$self=$PID;"
        "$items=@(Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.ProcessId -ne $observer -and $_.ProcessId -ne $self -and "
        f"($_.CommandLine) -and ({token_filter}) }} | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine);"
        "$items | ConvertTo-Json -Compress -Depth 3"
    )
    try:
        result = subprocess.run(
            (
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ),
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControlError("Windows CIM process query failed") from exc
    if result.returncode != 0 or result.stderr != b"" or len(result.stdout) > _MAX_STDOUT_BYTES:
        raise ControlError("Windows CIM process query did not exit cleanly")
    raw = result.stdout.strip()
    if not raw:
        matches: list[dict[str, Any]] = []
    else:
        try:
            decoded = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlError("Windows CIM process query returned invalid JSON") from exc
        values = decoded if type(decoded) is list else [decoded]
        matches = []
        for value in values:
            if type(value) is not dict:
                raise ControlError("Windows CIM process record is not one object")
            process_id = value.get("ProcessId")
            parent_id = value.get("ParentProcessId")
            name = value.get("Name")
            command_line = value.get("CommandLine")
            _positive_int(process_id, "matching process ID")
            _nonnegative_int(parent_id, "matching parent process ID")
            if type(name) is not str or not name or type(command_line) is not str:
                raise ControlError("Windows CIM process record has invalid text")
            matches.append(
                {
                    "process_id": process_id,
                    "parent_process_id": parent_id,
                    "name": name,
                    "command_line_sha256": hashlib.sha256(command_line.encode()).hexdigest(),
                }
            )
        matches.sort(key=lambda value: value["process_id"])
    return {
        "query_method": _PROCESS_QUERY_METHOD,
        "observer_pid": observer_pid,
        "observed_at_utc": _timestamp(),
        "matches": matches,
        "historical_pid_inference_performed": False,
    }


def _canonical_process_quiescence(
    value: object,
    *,
    expected_observer_pid: int | None = None,
) -> dict[str, Any]:
    payload = _exact_dict(
        value,
        frozenset(
            {
                "query_method",
                "observer_pid",
                "observed_at_utc",
                "matches",
                "historical_pid_inference_performed",
            }
        ),
        "process quiescence",
    )
    observer_pid = _positive_int(payload["observer_pid"], "process observer PID")
    if expected_observer_pid is not None and (
        type(expected_observer_pid) is not int
        or expected_observer_pid <= 0
        or observer_pid != expected_observer_pid
    ):
        raise ControlError("process observer PID differs from the querying controller")
    if (
        payload["query_method"] != _PROCESS_QUERY_METHOD
        or payload["matches"] != []
        or payload["historical_pid_inference_performed"] is not False
    ):
        raise ControlError("terminal process quiescence is not exact")
    _canonical_timestamp(payload["observed_at_utc"], "process observation timestamp")
    return payload


def _controller_identity() -> dict[str, Any]:
    return _file_record(_absolute(Path(__file__)), "replacement-v2 qualifying controller")


def _build_terminal_receipt(
    namespace: Namespace,
    *,
    parent_authority_directory: str | Path,
    qualified_at: datetime | None,
    process_quiescence: Mapping[str, Any],
    lock_quiescence: Mapping[str, Any],
    pins: HistoricalPins,
) -> dict[str, Any]:
    (
        project,
        parent,
        paths,
        input_payloads,
        input_records,
        auth_item,
        attempt_item,
        failure_item,
    ) = _historical_terminal_payloads(
        namespace,
        parent_authority_directory=parent_authority_directory,
        pins=pins,
    )
    del input_payloads
    _support, authority_c, aggregate, ordered_reads = _historical_support_records(
        project=project,
        parent=parent,
        paths=paths,
        pins=pins,
        input_records=input_records,
        authorization_record=auth_item["authorization_record"],
        attempt_record=attempt_item["attempt_record"],
        failure_record=failure_item["failure_record"],
    )
    process = _canonical_process_quiescence(process_quiescence)
    lock = _exact_dict(
        lock_quiescence,
        frozenset(
            {
                "scan_method",
                "first_scan_paths",
                "second_scan_paths",
                "reads_between_scans",
            }
        ),
        "lock quiescence",
    )
    if (
        lock["scan_method"] != _LOCK_SCAN_METHOD
        or lock["first_scan_paths"] != []
        or lock["second_scan_paths"] != []
        or lock["reads_between_scans"] != ordered_reads
    ):
        raise ControlError("terminal lock quiescence is not exact")
    authorization = auth_item["authorization"]
    attempt = attempt_item["attempt"]
    live_controller = _controller_identity()
    consumed_controller_path = _absolute(attempt["controller_path"])
    diagnosed_controller = _file_record(
        consumed_controller_path,
        "diagnosed fixed legacy controller",
    )
    if (
        diagnosed_controller["size_bytes"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SIZE_BYTES
        or diagnosed_controller["sha256"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SHA256
    ):
        raise ControlError("diagnosed fixed legacy controller differs from its exact pin")
    receipt = {
        "schema_version": 1,
        "policy": TERMINAL_QUALIFICATION_POLICY,
        "status": TERMINAL_QUALIFICATION_STATUS,
        "qualified_at_utc": _timestamp(qualified_at),
        "project_root": str(project),
        "authority_c": authority_c,
        "terminal_namespace": {
            "classification": "rolled_back_failure",
            "classification_reason": "exact A+F exist and S/D/candidates are absent",
            "publication_authorization_receipt": auth_item["authorization_record"],
            "attempt_marker": attempt_item["attempt_record"],
            "failure_marker": failure_item["failure_record"],
            "success_marker_absence": {
                "path": str(_absolute(paths["success_v1"])),
                "absent": True,
            },
            "intended_authority_absence": {
                "path": str(_absolute(paths["historical_d1"])),
                "absent": True,
            },
            "candidate_count": 0,
            "candidate_paths": [],
        },
        "terminal_links": {
            "attempt_id": pins.attempt_id,
            "max_attempt_count": 1,
            "automatic_retry_allowed": False,
            "authorization_receipt_sha256": pins.authorization_v1.sha256,
            "attempt_marker_sha256": pins.attempt_v1.sha256,
            "technical_successor_authorization_sha256": (pins.technical_authorization_sha256),
            "intent_sha256": pins.intent_sha256,
            "preflight_fingerprint_sha256": pins.preflight_fingerprint_sha256,
            "amendment_timestamp_utc": authorization["publication"]["amendment_timestamp_utc"],
            "parent_authority_directory": str(parent),
            "run_state_sha256": pins.run_state_sha256,
        },
        "frozen_v2_inputs": {
            "directory": str(_absolute(paths["input_v2"])),
            "files": input_records,
            "execution_source_root_sha256": _HISTORICAL_V2_SOURCE_ROOT_SHA256,
            "execution_source_manifest_sha256": _HISTORICAL_V2_SOURCE_MANIFEST_SHA256,
            "execution_source_delta_sha256": _HISTORICAL_V2_SOURCE_DELTA_SHA256,
            "records_sha256": pins.input_v2_records_sha256,
        },
        "controller_identities": {
            "consumed_attempt_controller": {
                "path": str(consumed_controller_path),
                "size_bytes": pins.controller.size_bytes,
                "sha256": pins.controller.sha256,
                "live_file_match": False,
                "attested_by": [
                    "retired_v1_invalidation_receipt",
                    "attempt_marker",
                    "publication_authorization_receipt",
                    "v2_frozen_source_receipt",
                    "v2_source_allowlist",
                ],
            },
            "diagnosed_fixed_legacy_controller": {
                **diagnosed_controller,
                "distinct_from_consumed_attempt_controller": True,
                "authorized_to_retry_v1": False,
                "diagnostic_scope": ("future_process_boundary_regression_only_no_v1_retry"),
            },
            "qualifying_live_controller": {
                **live_controller,
                "distinct_from_consumed_attempt_controller": True,
                "authorized_to_retry_v1": False,
            },
        },
        "failure_cause": {
            "error_type": _HISTORICAL_FAILURE_ERROR_TYPE,
            "error_type_sha256": pins.failure_error_type_sha256,
            "error_text": _HISTORICAL_FAILURE_ERROR_TEXT,
            "error_sha256": pins.failure_error_sha256,
            "reason_code": "windows_venv_launcher_breaks_direct_child_ppid_contract",
            "scientific_or_evidence_corruption": False,
        },
        "run_state": aggregate["run_state"],
        "protected_bindings": aggregate["protected"],
        "process_quiescence": process,
        "lock_quiescence": lock,
        "disposition": {
            "v1_attempt_consumed": True,
            "v1_retry_allowed": False,
            "v1_artifacts_may_be_modified_moved_or_deleted": False,
            "successor_requires_new_namespace": True,
            "successor_may_reuse_v2_inputs": False,
            "qualification_authorizes_publication": False,
            "outcome_values_read": False,
        },
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    return _canonical_terminal_receipt(receipt, namespace=namespace, pins=pins)


_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "qualified_at_utc",
        "project_root",
        "authority_c",
        "terminal_namespace",
        "terminal_links",
        "frozen_v2_inputs",
        "controller_identities",
        "failure_cause",
        "run_state",
        "protected_bindings",
        "process_quiescence",
        "lock_quiescence",
        "disposition",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)


def _canonical_terminal_receipt(
    value: object,
    *,
    namespace: Namespace,
    pins: HistoricalPins,
) -> dict[str, Any]:
    payload = _exact_dict(value, _TERMINAL_RECEIPT_FIELDS, "terminal qualification receipt")
    project = _project_root(namespace)
    _exact_path_text(
        payload["project_root"],
        project,
        "terminal qualification project root",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["policy"] != TERMINAL_QUALIFICATION_POLICY
        or payload["status"] != TERMINAL_QUALIFICATION_STATUS
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("terminal qualification fixed policy is invalid")
    qualified_at = datetime.strptime(
        _canonical_timestamp(payload["qualified_at_utc"], "terminal qualification time"),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    consumed_at = datetime.strptime(
        "2026-07-28T18:19:20.303224Z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    if qualified_at <= consumed_at:
        raise ControlError("terminal qualification must postdate consumed attempt")
    if qualified_at > datetime.now(UTC):
        raise ControlError("terminal qualification cannot be future-dated")
    authority = _exact_dict(
        payload["authority_c"],
        frozenset(
            {
                "directory",
                "schema_version",
                "chain_depth",
                "artifact_root_sha256",
                "sha256_manifest_sha256",
                "flat_file_count",
                "files",
                "integrity_verified",
            }
        ),
        "terminal Authority C",
    )
    expected_parent = project / "artifacts" / "preregistration_amendments" / _AUTHORITY_C_COMPONENT
    _exact_path_text(authority["directory"], expected_parent, "terminal Authority C path")
    if (
        type(authority["schema_version"]) is not int
        or authority["schema_version"] != 4
        or type(authority["chain_depth"]) is not int
        or authority["chain_depth"] != 3
        or authority["artifact_root_sha256"] != _AUTHORITY_C_ARTIFACT_ROOT_SHA256
        or authority["sha256_manifest_sha256"] != _AUTHORITY_C_MANIFEST_SHA256
        or type(authority["flat_file_count"]) is not int
        or authority["flat_file_count"] != 8
        or authority["integrity_verified"] is not True
        or type(authority["files"]) is not dict
        or set(authority["files"]) != set(_AUTHORITY_C_FILE_PINS)
    ):
        raise ControlError("terminal Authority C binding differs")
    for role, (filename, size_bytes, sha256) in _AUTHORITY_C_FILE_PINS.items():
        _record_matches(
            authority["files"][role],
            path=expected_parent / filename,
            pin=FilePin(size_bytes, sha256),
            role=f"terminal Authority C {role}",
        )
    terminal = _exact_dict(
        payload["terminal_namespace"],
        frozenset(
            {
                "classification",
                "classification_reason",
                "publication_authorization_receipt",
                "attempt_marker",
                "failure_marker",
                "success_marker_absence",
                "intended_authority_absence",
                "candidate_count",
                "candidate_paths",
            }
        ),
        "terminal namespace",
    )
    if (
        terminal["classification"] != "rolled_back_failure"
        or terminal["classification_reason"] != "exact A+F exist and S/D/candidates are absent"
        or type(terminal["candidate_count"]) is not int
        or terminal["candidate_count"] != 0
        or terminal["candidate_paths"] != []
    ):
        raise ControlError("terminal namespace classification differs")
    for role, pin in (
        ("publication_authorization_receipt", pins.authorization_v1),
        ("attempt_marker", pins.attempt_v1),
        ("failure_marker", pins.failure_v1),
    ):
        terminal_path = {
            "publication_authorization_receipt": (
                namespace.control_root / _HISTORICAL_AUTH_V1_FILENAME
            ),
            "attempt_marker": namespace.control_root / _HISTORICAL_ATTEMPT_V1_FILENAME,
            "failure_marker": namespace.control_root / _HISTORICAL_FAILURE_V1_FILENAME,
        }[role]
        _record_matches(
            terminal[role],
            path=terminal_path,
            pin=pin,
            role=f"terminal namespace {role}",
        )
    absence_paths = {
        "success_marker_absence": (namespace.control_root / _HISTORICAL_SUCCESS_V1_FILENAME),
        "intended_authority_absence": (expected_parent.parent / _HISTORICAL_D1_COMPONENT),
    }
    for role, expected_path in absence_paths.items():
        absence = _exact_dict(
            terminal[role],
            frozenset({"path", "absent"}),
            f"terminal {role}",
        )
        _exact_path_text(absence["path"], expected_path, f"terminal {role} path")
        if absence["absent"] is not True:
            raise ControlError(f"terminal {role} is not exact")
    links = _exact_dict(
        payload["terminal_links"],
        frozenset(
            {
                "attempt_id",
                "max_attempt_count",
                "automatic_retry_allowed",
                "authorization_receipt_sha256",
                "attempt_marker_sha256",
                "technical_successor_authorization_sha256",
                "intent_sha256",
                "preflight_fingerprint_sha256",
                "amendment_timestamp_utc",
                "parent_authority_directory",
                "run_state_sha256",
            }
        ),
        "terminal links",
    )
    _exact_path_text(
        links["parent_authority_directory"],
        expected_parent,
        "terminal links parent path",
    )
    if (
        links["attempt_id"] != pins.attempt_id
        or type(links["max_attempt_count"]) is not int
        or links["max_attempt_count"] != 1
        or links["automatic_retry_allowed"] is not False
        or links["authorization_receipt_sha256"] != pins.authorization_v1.sha256
        or links["attempt_marker_sha256"] != pins.attempt_v1.sha256
        or links["technical_successor_authorization_sha256"] != pins.technical_authorization_sha256
        or links["intent_sha256"] != pins.intent_sha256
        or links["preflight_fingerprint_sha256"] != pins.preflight_fingerprint_sha256
        or links["run_state_sha256"] != pins.run_state_sha256
        or links["amendment_timestamp_utc"] != "2026-07-28T18:19:20.303224Z"
    ):
        raise ControlError("terminal links differ from consumed v1")
    _canonical_timestamp(links["amendment_timestamp_utc"], "historical amendment time")
    frozen = _exact_dict(
        payload["frozen_v2_inputs"],
        frozenset(
            {
                "directory",
                "files",
                "execution_source_root_sha256",
                "execution_source_manifest_sha256",
                "execution_source_delta_sha256",
                "records_sha256",
            }
        ),
        "terminal frozen-v2 inputs",
    )
    expected_v2 = project / "artifacts" / "resource_control" / _HISTORICAL_INPUT_V2_DIRECTORY_NAME
    _exact_path_text(frozen["directory"], expected_v2, "terminal frozen-v2 directory")
    if (
        frozen["execution_source_root_sha256"] != _HISTORICAL_V2_SOURCE_ROOT_SHA256
        or frozen["execution_source_manifest_sha256"] != _HISTORICAL_V2_SOURCE_MANIFEST_SHA256
        or frozen["execution_source_delta_sha256"] != _HISTORICAL_V2_SOURCE_DELTA_SHA256
        or frozen["records_sha256"] != pins.input_v2_records_sha256
        or type(frozen["files"]) is not dict
        or set(frozen["files"]) != set(_HISTORICAL_INPUT_V2_FILENAMES)
        or _compact_sha256(frozen["files"]) != pins.input_v2_records_sha256
    ):
        raise ControlError("terminal frozen-v2 binding differs")
    for role, filename in _HISTORICAL_INPUT_V2_FILENAMES.items():
        _record_matches(
            frozen["files"][role],
            path=expected_v2 / filename,
            pin=pins.input_v2[role],
            role=f"terminal frozen-v2 {role}",
        )
    controllers = _exact_dict(
        payload["controller_identities"],
        frozenset(
            {
                "consumed_attempt_controller",
                "diagnosed_fixed_legacy_controller",
                "qualifying_live_controller",
            }
        ),
        "terminal controller identities",
    )
    consumed_controller = _exact_dict(
        controllers["consumed_attempt_controller"],
        frozenset({"path", "size_bytes", "sha256", "live_file_match", "attested_by"}),
        "consumed attempt controller",
    )
    expected_old_controller = (
        project
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_controller.py"
    )
    _exact_path_text(
        consumed_controller["path"],
        expected_old_controller,
        "consumed attempt controller path",
    )
    if (
        type(consumed_controller["size_bytes"]) is not int
        or consumed_controller["size_bytes"] != pins.controller.size_bytes
        or consumed_controller["sha256"] != pins.controller.sha256
        or consumed_controller["live_file_match"] is not False
        or consumed_controller["attested_by"]
        != [
            "retired_v1_invalidation_receipt",
            "attempt_marker",
            "publication_authorization_receipt",
            "v2_frozen_source_receipt",
            "v2_source_allowlist",
        ]
    ):
        raise ControlError("consumed attempt controller binding differs")
    diagnosed_controller = _exact_dict(
        controllers["diagnosed_fixed_legacy_controller"],
        frozenset(
            {
                "path",
                "size_bytes",
                "sha256",
                "distinct_from_consumed_attempt_controller",
                "authorized_to_retry_v1",
                "diagnostic_scope",
            }
        ),
        "diagnosed fixed legacy controller",
    )
    _exact_path_text(
        diagnosed_controller["path"],
        expected_old_controller,
        "diagnosed fixed legacy controller path",
    )
    if (
        type(diagnosed_controller["size_bytes"]) is not int
        or diagnosed_controller["size_bytes"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SIZE_BYTES
        or diagnosed_controller["sha256"] != _DIAGNOSED_FIXED_LEGACY_CONTROLLER_SHA256
        or diagnosed_controller["distinct_from_consumed_attempt_controller"] is not True
        or diagnosed_controller["authorized_to_retry_v1"] is not False
        or diagnosed_controller["diagnostic_scope"]
        != "future_process_boundary_regression_only_no_v1_retry"
    ):
        raise ControlError("diagnosed fixed legacy controller binding differs")
    qualifying_controller = _exact_dict(
        controllers["qualifying_live_controller"],
        frozenset(
            {
                "path",
                "size_bytes",
                "sha256",
                "distinct_from_consumed_attempt_controller",
                "authorized_to_retry_v1",
            }
        ),
        "qualifying live controller",
    )
    _exact_path_text(
        qualifying_controller["path"],
        Path(__file__),
        "qualifying live controller path",
    )
    if (
        _positive_int(
            qualifying_controller["size_bytes"],
            "qualifying controller size",
        )
        <= 0
        or qualifying_controller["sha256"] == pins.controller.sha256
        or qualifying_controller["distinct_from_consumed_attempt_controller"] is not True
        or qualifying_controller["authorized_to_retry_v1"] is not False
    ):
        raise ControlError("qualifying controller binding differs")
    _sha(qualifying_controller["sha256"], "qualifying controller")
    _sha(payload["failure_cause"].get("error_sha256"), "terminal failure error")
    expected_failure_cause = {
        "error_type": _HISTORICAL_FAILURE_ERROR_TYPE,
        "error_type_sha256": pins.failure_error_type_sha256,
        "error_text": _HISTORICAL_FAILURE_ERROR_TEXT,
        "error_sha256": pins.failure_error_sha256,
        "reason_code": "windows_venv_launcher_breaks_direct_child_ppid_contract",
        "scientific_or_evidence_corruption": False,
    }
    failure_cause = payload["failure_cause"]
    if (
        type(failure_cause) is not dict
        or set(failure_cause) != set(expected_failure_cause)
        or failure_cause["scientific_or_evidence_corruption"] is not False
        or failure_cause != expected_failure_cause
    ):
        raise ControlError("terminal failure cause differs")
    process = _canonical_process_quiescence(payload["process_quiescence"])
    process_at = datetime.strptime(
        process["observed_at_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    process_age_seconds = (qualified_at - process_at).total_seconds()
    if process_at <= consumed_at or process_age_seconds < 0 or process_age_seconds > 60:
        raise ControlError("terminal process observation time is out of order")
    run_state = _exact_dict(
        payload["run_state"],
        frozenset({"root", "files", "sha256"}),
        "terminal run state",
    )
    expected_run_root = project / "artifacts" / "runs"
    _exact_path_text(run_state["root"], expected_run_root, "terminal run-state root")
    if (
        run_state["sha256"] != pins.run_state_sha256
        or type(run_state["files"]) is not dict
        or set(run_state["files"]) != set(_RUN_STATE_FILENAMES)
    ):
        raise ControlError("terminal run-state binding differs")
    for filename in _RUN_STATE_FILENAMES:
        size_bytes, sha256 = _RUN_STATE_PINS[filename]
        _record_matches(
            run_state["files"][filename],
            path=expected_run_root / filename,
            pin=FilePin(size_bytes, sha256),
            role=f"terminal run-state {filename}",
        )
    if (
        _compact_sha256(
            {filename: run_state["files"][filename]["sha256"] for filename in _RUN_STATE_FILENAMES}
        )
        != pins.run_state_sha256
    ):
        raise ControlError("terminal run-state aggregate does not recompute")
    protected = payload["protected_bindings"]
    if type(protected) is not dict or set(protected) != set(_PROTECTED_BINDINGS):
        raise ControlError("terminal protected binding roles differ")
    for role, (relative_path, size_bytes, sha256) in _PROTECTED_BINDINGS.items():
        _record_matches(
            protected[role],
            path=project / relative_path,
            pin=FilePin(size_bytes, sha256),
            role=f"terminal protected {role}",
        )
    lock = _exact_dict(
        payload["lock_quiescence"],
        frozenset(
            {
                "scan_method",
                "first_scan_paths",
                "second_scan_paths",
                "reads_between_scans",
            }
        ),
        "terminal lock quiescence",
    )
    auxiliary = {
        "retired_v1_invalidation_receipt": (
            namespace.control_root / _HISTORICAL_INVALIDATION_FILENAME,
            pins.invalidation,
        ),
        "prior_publication_failure_receipt": (
            namespace.control_root / _HISTORICAL_PRIOR_FAILURE_FILENAME,
            pins.prior_failure,
        ),
        "failed_preflight_receipt": (
            namespace.control_root / _HISTORICAL_FAILED_PREFLIGHT_FILENAME,
            pins.failed_preflight,
        ),
    }
    expected_reads: list[dict[str, Any]] = [
        {
            "role": "publication_authorization_receipt",
            **terminal["publication_authorization_receipt"],
        },
        {"role": "publication_attempt_marker", **terminal["attempt_marker"]},
        {"role": "publication_failure_marker", **terminal["failure_marker"]},
    ]
    expected_reads.extend(
        {
            "role": role,
            "path": str(_absolute(path)),
            "size_bytes": pin.size_bytes,
            "sha256": pin.sha256,
        }
        for role, (path, pin) in auxiliary.items()
    )
    expected_reads.extend(
        {"role": f"v2_{role}", **frozen["files"][role]}
        for role in (
            "cnn_correction_receipt",
            "frozen_source_receipt",
            "source_allowlist",
            "workspace_plan",
        )
    )
    expected_reads.extend(
        {"role": f"protected_{role}", **protected[role]}
        for role in (
            "specification",
            "pre_registration",
            "primary_config",
            "confirmatory_config",
        )
    )
    expected_reads.extend(
        {"role": f"authority_c_{role}", **authority["files"][role]}
        for role in (
            "amendment_evidence",
            "amendment_report",
            "confirmatory_config",
            "immutable_marker",
            "preregistration",
            "primary_config",
            "sha256_manifest",
            "source_tree_manifest",
        )
    )
    expected_reads.extend(
        {"role": f"run_state_{filename}", **run_state["files"][filename]}
        for filename in _RUN_STATE_FILENAMES
    )
    raw_reads = lock["reads_between_scans"]
    if type(raw_reads) is not list or len(raw_reads) != len(expected_reads):
        raise ControlError("terminal lock-quiescence read count differs")
    for position, record in enumerate(raw_reads):
        if (
            type(record) is not dict
            or set(record) != {"role", "path", "size_bytes", "sha256"}
            or type(record["role"]) is not str
            or type(record["path"]) is not str
            or type(record["size_bytes"]) is not int
        ):
            raise ControlError(f"terminal lock-quiescence read {position} has invalid exact types")
        _sha(record["sha256"], f"terminal lock-quiescence read {position}")
    if (
        lock["scan_method"] != _LOCK_SCAN_METHOD
        or lock["first_scan_paths"] != []
        or lock["second_scan_paths"] != []
        or raw_reads != expected_reads
    ):
        raise ControlError("terminal lock quiescence differs")
    expected_disposition = {
        "v1_attempt_consumed": True,
        "v1_retry_allowed": False,
        "v1_artifacts_may_be_modified_moved_or_deleted": False,
        "successor_requires_new_namespace": True,
        "successor_may_reuse_v2_inputs": False,
        "qualification_authorizes_publication": False,
        "outcome_values_read": False,
    }
    disposition = payload["disposition"]
    if (
        type(disposition) is not dict
        or set(disposition) != set(expected_disposition)
        or any(disposition[key] is not value for key, value in expected_disposition.items())
    ):
        raise ControlError("terminal qualification disposition differs")
    return payload


def _read_terminal_qualification(
    namespace: Namespace,
    *,
    pins: HistoricalPins = DEFAULT_HISTORICAL_PINS,
    verify_live_history: bool = True,
) -> tuple[dict[str, Any], str]:
    encoded = _read_bytes(
        namespace.terminal_qualification,
        "terminal qualification receipt",
    )
    payload = _canonical_terminal_receipt(
        _strict_json_object(encoded, "terminal qualification receipt"),
        namespace=namespace,
        pins=pins,
    )
    if encoded != _canonical_bytes(payload):
        raise ControlError("terminal qualification receipt bytes are not canonical")
    if verify_live_history:
        for record in payload["lock_quiescence"]["reads_between_scans"]:
            observed = _file_record(
                Path(record["path"]),
                f"terminal read {record['role']}",
            )
            if any(observed[field] != record[field] for field in ("path", "size_bytes", "sha256")):
                raise ControlError(f"terminal qualification binding changed: {record['role']}")
        controllers = payload["controller_identities"]
        current = _controller_identity()
        live_controller = controllers["qualifying_live_controller"]
        if any(
            current[field] != live_controller[field] for field in ("path", "size_bytes", "sha256")
        ):
            raise ControlError("qualifying replacement-v2 controller changed")
        diagnosed = _file_record(
            Path(controllers["diagnosed_fixed_legacy_controller"]["path"]),
            "diagnosed fixed legacy controller",
        )
        if any(
            diagnosed[field] != controllers["diagnosed_fixed_legacy_controller"][field]
            for field in ("path", "size_bytes", "sha256")
        ):
            raise ControlError("diagnosed fixed legacy controller changed")
        terminal = payload["terminal_namespace"]
        if os.path.lexists(terminal["success_marker_absence"]["path"]) or os.path.lexists(
            terminal["intended_authority_absence"]["path"]
        ):
            raise ControlError("historical terminal absence no longer holds")
        project = _project_root(namespace)
        parent = project / "artifacts" / "preregistration_amendments" / _AUTHORITY_C_COMPONENT
        _stable_amendment_inventory(parent.parent)
        if discover_candidates(parent):
            raise ControlError("terminal qualification no-D condition no longer holds")
        _reserved_family_presence(namespace)
        historical = _historical_paths(namespace, parent)
        _input_v2_records(historical, pins)
        _authority_c_receipt(parent)
    return payload, hashlib.sha256(encoded).hexdigest()


def replacement_publication_failure_lineage(
    namespace: Namespace,
    *,
    pins: HistoricalPins = DEFAULT_HISTORICAL_PINS,
) -> dict[str, Any]:
    """Return the exact schema-v3 envelope around the canonical receipt."""

    receipt, digest = _read_terminal_qualification(namespace, pins=pins)
    envelope = {
        "terminal_qualification_receipt_path": str(_absolute(namespace.terminal_qualification)),
        "terminal_qualification_receipt_sha256": digest,
        "terminal_qualification_receipt": receipt,
    }
    return _exact_dict(
        envelope,
        FAILURE_LINEAGE_ENVELOPE_FIELDS,
        "replacement publication failure lineage",
    )


def discover_candidates(parent_authority_directory: str | Path) -> tuple[Path, ...]:
    """Treat every direct amendment-root peer beyond frozen A/P/C as a candidate."""

    parent = _real_directory(parent_authority_directory, "Authority C")
    root = _real_directory(parent.parent, "amendment root")

    def scan() -> tuple[str, ...]:
        try:
            return tuple(sorted(entry.name for entry in os.scandir(root)))
        except OSError as exc:
            raise ControlError("candidate amendment-root scan failed") from exc

    first = scan()
    baseline = set(_AMENDMENT_BASELINE)
    if not baseline.issubset(first):
        raise ControlError("amendment root no longer contains exact A/P/C baseline")
    extras = tuple(name for name in first if name not in baseline)
    candidates: list[Path] = []
    for name in first:
        authority = _real_directory(root / name, f"amendment-root peer {name}")
        if name in extras:
            candidates.append(authority)
    second = scan()
    if second != first:
        raise ControlError("candidate amendment-root scan changed")
    return tuple(candidates)


_RESERVED_INPUT_PREFIX = "authority_d_replacement_inputs_"
_RESERVED_PROTOCOL_PREFIX = "resource_authority_d_replacement_"


def _reserved_family_presence(namespace: Namespace) -> dict[str, bool]:
    """Reject every noncanonical alias in the two governed control families."""

    control = _real_directory(namespace.control_root, "replacement-v2 control root")
    allowed = {
        "authority_d_replacement_inputs_v1",
        _HISTORICAL_INVALIDATION_FILENAME,
        _HISTORICAL_INPUT_V2_DIRECTORY_NAME,
        INPUT_V3_DIRECTORY_NAME,
        _HISTORICAL_AUTH_V1_FILENAME,
        _HISTORICAL_ATTEMPT_V1_FILENAME,
        _HISTORICAL_SUCCESS_V1_FILENAME,
        _HISTORICAL_FAILURE_V1_FILENAME,
        TERMINAL_QUALIFICATION_FILENAME,
        PUBLICATION_AUTHORIZATION_V2_FILENAME,
        ATTEMPT_V2_FILENAME,
        SUCCESS_V2_FILENAME,
        FAILURE_V2_FILENAME,
    }
    try:
        observed_names = tuple(entry.name for entry in os.scandir(control))
    except OSError as exc:
        raise ControlError("replacement-v2 reserved-family scan failed") from exc
    governed = [
        name
        for name in observed_names
        if name.casefold().startswith(_RESERVED_INPUT_PREFIX.casefold())
        or name.casefold().startswith(_RESERVED_PROTOCOL_PREFIX.casefold())
    ]
    if len({name.casefold() for name in governed}) != len(governed):
        raise AmbiguousStateError("replacement-v2 reserved family contains case aliases")
    for name in governed:
        if name not in allowed:
            raise AmbiguousStateError(
                f"unrecognised replacement-v2 reserved-family entry: {name!r}"
            )
    paths = {
        "qualification": namespace.terminal_qualification,
        "inputs": namespace.input_v3,
        "authorization": namespace.authorization_v2,
        "attempt": namespace.attempt_v2,
        "success": namespace.success_v2,
        "failure": namespace.failure_v2,
    }
    return {role: os.path.lexists(path) for role, path in paths.items()}


def _require_no_preliminary_successor_assets(namespace: Namespace) -> None:
    presence = _reserved_family_presence(namespace)
    assets = (
        namespace.terminal_qualification,
        namespace.input_v3,
        namespace.authorization_v2,
        namespace.attempt_v2,
        namespace.success_v2,
        namespace.failure_v2,
    )
    if any(presence.values()) or any(os.path.lexists(path) for path in assets):
        raise ControlError("replacement-v2 preliminary or terminal asset already exists")


def _rollback_owned_or_ambiguous(
    owned_locks: Sequence[ExclusiveBundlePublicationLock],
    publications: list[PublishedPath],
    cause: BaseException,
) -> None:
    try:
        for lock in owned_locks:
            lock.assert_owned()
        rollback_owned_publications(publications)
        publications.clear()
        for lock in owned_locks:
            lock.assert_owned()
    except (OSError, RuntimeError) as rollback_error:
        raise AmbiguousStateError(
            "owned publication could not be rolled back under intact lock ownership"
        ) from rollback_error
    raise cause


@contextmanager
def _publication_exclusion_boundary(
    publications: list[PublishedPath],
    *,
    role: str,
) -> Iterator[None]:
    """Classify lock-exit failure after a retained O_EXCL write as ambiguous."""

    try:
        yield
    except BaseException as exc:
        if publications:
            raise AmbiguousStateError(
                f"{role} exclusion ended while an owned publication may remain"
            ) from exc
        raise


def _qualify_historical_terminal_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    pins: HistoricalPins,
    clock: Callable[[], datetime],
    process_probe: Callable[[int], Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Private sealed-seam implementation used by production and hermetic tests."""

    _project, parent = _require_parent(namespace, parent_authority_directory)
    _stable_amendment_inventory(parent.parent)
    _require_no_preliminary_successor_assets(namespace)
    if discover_candidates(parent):
        raise ControlError("historical terminal qualification requires zero candidates")
    legacy_lock_paths = _legacy_scoped_lock_paths(namespace, parent=parent)
    if _scan_present_paths(legacy_lock_paths):
        raise ControlError("historical replacement lock topology is not quiescent")
    observer_pid = os.getpid()
    _canonical_process_quiescence(
        dict(process_probe(observer_pid)),
        expected_observer_pid=observer_pid,
    )

    published: list[PublishedPath] = []
    with (
        _publication_exclusion_boundary(
            published,
            role="terminal qualification",
        ),
        _protocol_lock(
            namespace,
            parent=parent,
            role="resource Authority-D replacement-v2 terminal qualification",
        ) as publication_lock,
        ExclusiveBundlePublicationLock(
            (parent,),
            role="resource Authority-D replacement-v2 Authority-C parent guard",
        ) as parent_guard,
    ):
        try:
            owned_locks = (publication_lock, parent_guard)
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            _stable_amendment_inventory(parent.parent)
            _require_no_preliminary_successor_assets(namespace)
            if discover_candidates(parent):
                raise ControlError("Authority-D candidate appeared before terminal qualification")
            (
                project_snapshot,
                parent_snapshot,
                paths,
                _input_payloads,
                input_records,
                auth_item,
                attempt_item,
                failure_item,
            ) = _historical_terminal_payloads(
                namespace,
                parent_authority_directory=parent,
                pins=pins,
            )
            first_scan = _scan_present_paths(
                tuple(
                    path
                    for path in legacy_lock_paths
                    if _absolute(path)
                    not in {_absolute(owned) for lock in owned_locks for owned in lock.lock_paths}
                )
            )
            if first_scan:
                raise ControlError("foreign historical lock appeared before reads")
            _support, _authority_c, _aggregate, ordered_reads = _historical_support_records(
                project=project_snapshot,
                parent=parent_snapshot,
                paths=paths,
                pins=pins,
                input_records=input_records,
                authorization_record=auth_item["authorization_record"],
                attempt_record=attempt_item["attempt_record"],
                failure_record=failure_item["failure_record"],
            )
            second_scan = _scan_present_paths(
                tuple(
                    path
                    for path in legacy_lock_paths
                    if _absolute(path)
                    not in {_absolute(owned) for lock in owned_locks for owned in lock.lock_paths}
                )
            )
            if second_scan:
                raise ControlError("foreign historical lock appeared during reads")
            process_quiescence = _canonical_process_quiescence(
                dict(process_probe(observer_pid)),
                expected_observer_pid=observer_pid,
            )
            qualified_at = clock()
            trusted_now = clock()
            if (
                not isinstance(qualified_at, datetime)
                or not isinstance(trusted_now, datetime)
                or qualified_at.tzinfo is None
                or trusted_now.tzinfo is None
                or qualified_at.astimezone(UTC) > trusted_now.astimezone(UTC)
            ):
                raise ControlError("terminal qualification clock is not trusted UTC order")
            lock_quiescence = {
                "scan_method": _LOCK_SCAN_METHOD,
                "first_scan_paths": first_scan,
                "second_scan_paths": second_scan,
                "reads_between_scans": ordered_reads,
            }
            candidate_receipt = _build_terminal_receipt(
                namespace,
                parent_authority_directory=parent,
                qualified_at=qualified_at,
                process_quiescence=process_quiescence,
                lock_quiescence=lock_quiescence,
                pins=pins,
            )
            for lock in owned_locks:
                lock.assert_owned()
            published.append(
                publish_bytes_no_overwrite(
                    _canonical_bytes(candidate_receipt),
                    namespace.terminal_qualification,
                )
            )
            for lock in owned_locks:
                lock.assert_owned()
            readback, digest = _read_terminal_qualification(namespace, pins=pins)
            if (
                readback != candidate_receipt
                or published[0].sha256 != digest
                or not published[0].still_owned()
            ):
                raise ControlError("terminal qualification failed exact O_EXCL readback")
            presence = _reserved_family_presence(namespace)
            if presence != {
                "qualification": True,
                "inputs": False,
                "authorization": False,
                "attempt": False,
                "success": False,
                "failure": False,
            }:
                raise AmbiguousStateError(
                    "replacement-v2 state changed during terminal qualification"
                )
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise AmbiguousStateError(
                    "Authority-D candidate appeared during terminal qualification"
                )
            _canonical_process_quiescence(
                dict(process_probe(observer_pid)),
                expected_observer_pid=observer_pid,
            )
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            for lock in owned_locks:
                lock.assert_owned()
            return readback, digest
        except BaseException as exc:
            if published:
                _rollback_owned_or_ambiguous(owned_locks, published, exc)
            raise


def qualify_historical_terminal_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> tuple[dict[str, Any], str]:
    """O_EXCL-publish the sole historical A1+F1/no-S1/no-D1 qualification."""

    return _qualify_historical_terminal_once(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        pins=DEFAULT_HISTORICAL_PINS,
        clock=lambda: datetime.now(UTC),
        process_probe=_windows_process_quiescence,
    )


_SOURCE_ALLOWLIST_FIELDS = frozenset({"schema_version", "policy", "file_count", "records"})
_CNN_RECEIPT_FIELDS = frozenset(
    {"schema_version", "policy", "correction", "semantic_equivalence_evidence"}
)
_CNN_CORRECTION_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "scenario_id",
        "cache_provenance_id",
        "before_config_record",
        "before_config_record_sha256",
        "after_config_record",
        "after_config_record_sha256",
        "before",
        "after",
        "unchanged_cache_artifacts",
        "scientific_profile_changed",
    }
)
_CNN_CONFIG_RECORD_FIELDS = frozenset(
    {
        "id",
        "representation_id",
        "status",
        "cache_file_sha256",
        "sample_order_sha256",
        "manifest_sha256",
        "encoder_identifier",
        "encoder_metadata_sha256",
        "input_variant",
        "preprocessing_identifier",
        "preprocessing_sha256",
        "sidecar_semantic_sha256",
        "weight_identifier",
        "weights_sha256",
    }
)
_CNN_SEMANTIC_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "scenario_id",
        "cache_provenance_id",
        "runtime_model_or_preprocessing_behavior_changed",
        "scientific_profile_changed",
        "cache_bytes_changed",
        "sidecar_bytes_changed",
        "evidence_note",
    }
)
_FROZEN_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "file_count",
        "source_allowlist_sha256",
        "workspace_plan_sha256",
        "cnn_correction_receipt_sha256",
        "source_allowlist_semantic_sha256",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "execution_source_artifact_count",
        "execution_source_delta_count",
        "execution_source_delta_sha256",
        "execution_source_change_kinds_sha256",
        "parent_execution_source_root_sha256",
        "parent_execution_source_manifest_sha256",
        "config_path",
        "config_file_sha256",
        "config_semantic_sha256",
        "manifest_path",
        "manifest_sha256",
        "failed_preflight_receipt_path",
        "failed_preflight_receipt_sha256",
        "prior_failure_receipt_path",
        "prior_failure_receipt_sha256",
        "retired_input_invalidation_receipt_path",
        "retired_input_invalidation_receipt_sha256",
        "terminal_qualification_receipt_path",
        "terminal_qualification_receipt_sha256",
        "controller_path",
        "controller_size_bytes",
        "controller_sha256",
        "run_state_root",
        "run_state_files",
        "run_state_sha256",
        "authorization_sha256",
        "workspace_plan_without_self_hash_sha256",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)


class InputV3Reconstructor(Protocol):
    """Outcome-blind source for the four native, role-specific v3 inputs."""

    def reconstruct(
        self,
        *,
        namespace: Namespace,
        project_root: Path,
        parent_authority_directory: Path,
        terminal_qualification: Mapping[str, Any],
        terminal_qualification_sha256: str,
    ) -> Mapping[str, Mapping[str, Any]]: ...


def _normalise_source_path(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
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


def _canonical_source_allowlist(value: object) -> dict[str, Any]:
    payload = _exact_dict(value, _SOURCE_ALLOWLIST_FIELDS, "input-v3 source allowlist")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["policy"] != INPUT_V3_POLICIES["source_allowlist"]
        or type(payload["file_count"]) is not int
        or payload["file_count"] != len(_EXPECTED_SOURCE_CHANGE_KINDS)
        or type(payload["records"]) is not list
        or len(payload["records"]) != len(_EXPECTED_SOURCE_CHANGE_KINDS)
    ):
        raise ControlError("input-v3 source allowlist fixed policy differs")
    canonical_records: list[dict[str, Any]] = []
    for logical, expected_kind in sorted(_EXPECTED_SOURCE_CHANGE_KINDS.items()):
        position = len(canonical_records)
        raw = payload["records"][position]
        expected_fields = (
            frozenset({"path", "change_kind"})
            if expected_kind == "removed"
            else frozenset({"path", "change_kind", "size_bytes", "sha256"})
        )
        record = _exact_dict(
            raw,
            expected_fields,
            f"input-v3 source allowlist record {logical}",
        )
        if (
            _normalise_source_path(record["path"]) != logical
            or record["change_kind"] != expected_kind
        ):
            raise ControlError("input-v3 source allowlist is not exact and sorted")
        canonical_record: dict[str, Any] = {
            "path": logical,
            "change_kind": expected_kind,
        }
        if expected_kind != "removed":
            canonical_record.update(
                {
                    "size_bytes": _positive_int(
                        record["size_bytes"],
                        f"input-v3 source {logical} size",
                    ),
                    "sha256": _sha(
                        record["sha256"],
                        f"input-v3 source {logical}",
                    ),
                }
            )
        canonical_records.append(canonical_record)
    canonical = {
        "schema_version": 1,
        "policy": INPUT_V3_POLICIES["source_allowlist"],
        "file_count": len(canonical_records),
        "records": canonical_records,
    }
    if payload != canonical:
        raise ControlError("input-v3 source allowlist is not canonical")
    return canonical


def _canonical_workspace_plan(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ControlError("input-v3 workspace plan must be one exact mapping")
    try:
        from histo_audit.workflows import preregistration_amendment as amendment

        _, canonical = amendment.validate_resource_bounded_capacity_v3(
            dict(amendment._RESOURCE_BOUNDED_CAPACITY_V3),
            value,
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ControlError("input-v3 workspace plan failed the public validator") from exc
    if type(canonical) is not dict or canonical != value:
        raise ControlError("input-v3 workspace plan is not provider-canonical")
    return canonical


def _canonical_cnn_receipt(value: object) -> dict[str, Any]:
    payload = _exact_dict(value, _CNN_RECEIPT_FIELDS, "input-v3 CNN receipt")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["policy"] != INPUT_V3_POLICIES["cnn_correction_receipt"]
    ):
        raise ControlError("input-v3 CNN receipt fixed policy differs")
    correction = _exact_dict(
        payload["correction"],
        _CNN_CORRECTION_FIELDS,
        "input-v3 CNN correction",
    )
    semantic = _exact_dict(
        payload["semantic_equivalence_evidence"],
        _CNN_SEMANTIC_FIELDS,
        "input-v3 CNN semantic evidence",
    )
    expected_semantic_fixed = {
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
    if semantic != expected_semantic_fixed:
        raise ControlError("input-v3 CNN semantic evidence is not exact")
    if (
        type(correction["schema_version"]) is not int
        or correction["schema_version"] != 1
        or correction["policy"] != "resource_bounded_cnn_logical_provenance_correction_v1"
        or correction["scenario_id"] != "cnn_context_rgb"
        or correction["cache_provenance_id"] != "cnn_context_rgb_cache"
        or correction["scientific_profile_changed"] is not False
    ):
        raise ControlError("input-v3 CNN correction fixed policy differs")
    before_record = _exact_dict(
        correction["before_config_record"],
        _CNN_CONFIG_RECORD_FIELDS,
        "input-v3 CNN before-config record",
    )
    after_record = _exact_dict(
        correction["after_config_record"],
        _CNN_CONFIG_RECORD_FIELDS,
        "input-v3 CNN after-config record",
    )
    before = _exact_dict(
        correction["before"],
        frozenset(
            {
                "execution_cnn_source_sha256",
                "logical_provenance_source_sha256",
                "recomputed_record_sha256",
                "matches_config_record",
            }
        ),
        "input-v3 CNN before evidence",
    )
    after = _exact_dict(
        correction["after"],
        frozenset(
            {
                "execution_cnn_source_sha256",
                "logical_provenance_source_sha256",
                "recomputed_record_sha256",
                "semantic_equivalence_evidence_sha256",
                "matches_config_record",
            }
        ),
        "input-v3 CNN after evidence",
    )
    unchanged = _exact_dict(
        correction["unchanged_cache_artifacts"],
        frozenset(
            {
                "cache_file_sha256",
                "sidecar_file_sha256",
                "sidecar_semantic_sha256",
                "cache_bytes_changed",
                "sidecar_bytes_changed",
            }
        ),
        "input-v3 CNN unchanged-cache evidence",
    )
    for role, item in (
        ("before", before),
        ("after", after),
        ("unchanged cache", unchanged),
    ):
        for key, observed in item.items():
            if key.endswith("_sha256"):
                _sha(observed, f"input-v3 CNN {role} {key}")
    if (
        correction["before_config_record_sha256"] != _compact_sha256(before_record)
        or correction["after_config_record_sha256"] != _compact_sha256(after_record)
        or before["matches_config_record"] is not False
        or after["matches_config_record"] is not True
        or after["execution_cnn_source_sha256"] != after["logical_provenance_source_sha256"]
        or after["recomputed_record_sha256"] != correction["after_config_record_sha256"]
        or after["semantic_equivalence_evidence_sha256"] != _compact_sha256(semantic)
        or unchanged["cache_bytes_changed"] is not False
        or unchanged["sidecar_bytes_changed"] is not False
    ):
        raise ControlError("input-v3 CNN correction cross-links differ")
    canonical = {
        "schema_version": 1,
        "policy": INPUT_V3_POLICIES["cnn_correction_receipt"],
        "correction": correction,
        "semantic_equivalence_evidence": semantic,
    }
    _canonical_bytes(canonical)
    return canonical


def _canonical_frozen_source_receipt(
    value: object,
    *,
    namespace: Namespace,
    qualification_sha256: str,
) -> dict[str, Any]:
    payload = _exact_dict(
        value,
        _FROZEN_SOURCE_FIELDS,
        "input-v3 frozen-source receipt",
    )
    project = _project_root(namespace)
    paths = {
        "config": project / "configs" / "confirmatory_resource_bounded_amended.yaml",
        "manifest": (
            project / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
        ),
        "failed": namespace.control_root / _HISTORICAL_FAILED_PREFLIGHT_FILENAME,
        "prior": namespace.control_root / _HISTORICAL_PRIOR_FAILURE_FILENAME,
        "invalidation": namespace.control_root / _HISTORICAL_INVALIDATION_FILENAME,
        "qualification": namespace.terminal_qualification,
        "controller": Path(__file__),
        "run_state": project / "artifacts" / "runs",
    }
    for field, path in (
        ("config_path", paths["config"]),
        ("manifest_path", paths["manifest"]),
        ("failed_preflight_receipt_path", paths["failed"]),
        ("prior_failure_receipt_path", paths["prior"]),
        ("retired_input_invalidation_receipt_path", paths["invalidation"]),
        ("terminal_qualification_receipt_path", paths["qualification"]),
        ("controller_path", paths["controller"]),
        ("run_state_root", paths["run_state"]),
    ):
        _exact_path_text(payload[field], path, f"input-v3 frozen source {field}")
    integer_fields = (
        "schema_version",
        "file_count",
        "execution_source_artifact_count",
        "execution_source_delta_count",
        "controller_size_bytes",
    )
    if any(type(payload[field]) is not int for field in integer_fields):
        raise ControlError("input-v3 frozen-source integer fields must be exact integers")
    if (
        payload["schema_version"] != 1
        or payload["policy"] != INPUT_V3_POLICIES["frozen_source_receipt"]
        or payload["file_count"] != len(_EXPECTED_SOURCE_CHANGE_KINDS)
        or payload["execution_source_artifact_count"] <= 0
        or payload["execution_source_delta_count"] != len(_EXPECTED_SOURCE_CHANGE_KINDS)
        or payload["controller_size_bytes"] <= 0
        or payload["parent_execution_source_root_sha256"] != _AUTHORITY_C_SOURCE_PINS["root_sha256"]
        or payload["parent_execution_source_manifest_sha256"]
        != _AUTHORITY_C_SOURCE_PINS["manifest_sha256"]
        or payload["config_file_sha256"] != _RESOURCE_CONFIG_FILE_SHA256
        or payload["config_semantic_sha256"] != _RESOURCE_CONFIG_SEMANTIC_SHA256
        or payload["manifest_sha256"] != _PANNUKE_MANIFEST_SHA256
        or payload["failed_preflight_receipt_sha256"]
        != DEFAULT_HISTORICAL_PINS.failed_preflight.sha256
        or payload["prior_failure_receipt_sha256"] != DEFAULT_HISTORICAL_PINS.prior_failure.sha256
        or payload["retired_input_invalidation_receipt_sha256"]
        != DEFAULT_HISTORICAL_PINS.invalidation.sha256
        or payload["terminal_qualification_receipt_sha256"] != qualification_sha256
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("input-v3 frozen-source fixed policy differs")
    sha_fields = (
        "source_allowlist_sha256",
        "workspace_plan_sha256",
        "cnn_correction_receipt_sha256",
        "source_allowlist_semantic_sha256",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "execution_source_delta_sha256",
        "execution_source_change_kinds_sha256",
        "parent_execution_source_root_sha256",
        "parent_execution_source_manifest_sha256",
        "config_file_sha256",
        "config_semantic_sha256",
        "manifest_sha256",
        "failed_preflight_receipt_sha256",
        "prior_failure_receipt_sha256",
        "retired_input_invalidation_receipt_sha256",
        "terminal_qualification_receipt_sha256",
        "controller_sha256",
        "run_state_sha256",
        "authorization_sha256",
        "workspace_plan_without_self_hash_sha256",
    )
    for field in sha_fields:
        _sha(payload[field], f"input-v3 frozen source {field}")
    run_files = payload["run_state_files"]
    if type(run_files) is not dict or set(run_files) != set(_RUN_STATE_FILENAMES):
        raise ControlError("input-v3 frozen source run-state roles differ")
    for filename in _RUN_STATE_FILENAMES:
        _sha(run_files[filename], f"input-v3 frozen run-state {filename}")
    if payload["run_state_sha256"] != _compact_sha256(run_files):
        raise ControlError("input-v3 frozen source run-state root differs")
    return payload


def _canonical_input_v3_payloads(
    values: Mapping[str, Mapping[str, Any]],
    *,
    namespace: Namespace,
    qualification: Mapping[str, Any],
    qualification_sha256: str,
) -> dict[str, dict[str, Any]]:
    if type(values) is not dict or set(values) != set(INPUT_V3_FILENAMES):
        raise ControlError("input-v3 requires exactly four native role documents")
    result = {
        "source_allowlist": _canonical_source_allowlist(values["source_allowlist"]),
        "workspace_plan": _canonical_workspace_plan(values["workspace_plan"]),
        "cnn_correction_receipt": _canonical_cnn_receipt(values["cnn_correction_receipt"]),
        "frozen_source_receipt": _canonical_frozen_source_receipt(
            values["frozen_source_receipt"],
            namespace=namespace,
            qualification_sha256=qualification_sha256,
        ),
    }
    frozen = result["frozen_source_receipt"]
    q_run_state = qualification.get("run_state")
    q_controllers = qualification.get("controller_identities")
    if (
        type(q_run_state) is not dict
        or type(q_run_state.get("files")) is not dict
        or type(q_controllers) is not dict
        or type(q_controllers.get("qualifying_live_controller")) is not dict
        or frozen["run_state_root"] != q_run_state.get("root")
        or frozen["run_state_sha256"] != q_run_state.get("sha256")
        or frozen["run_state_files"]
        != {filename: q_run_state["files"][filename]["sha256"] for filename in _RUN_STATE_FILENAMES}
        or any(
            frozen[frozen_field] != q_controllers["qualifying_live_controller"][q_field]
            for frozen_field, q_field in (
                ("controller_path", "path"),
                ("controller_size_bytes", "size_bytes"),
                ("controller_sha256", "sha256"),
            )
        )
    ):
        raise ControlError("input-v3 frozen source does not cross-bind Q run-state/controller")
    child_fields = {
        "source_allowlist_sha256": "source_allowlist",
        "workspace_plan_sha256": "workspace_plan",
        "cnn_correction_receipt_sha256": "cnn_correction_receipt",
    }
    for field, role in child_fields.items():
        expected = hashlib.sha256(_canonical_bytes(result[role])).hexdigest()
        if frozen[field] != expected:
            raise ControlError(f"input-v3 frozen source does not bind {role}")
    if (
        frozen["source_allowlist_semantic_sha256"] != _compact_sha256(result["source_allowlist"])
        or frozen["execution_source_change_kinds_sha256"]
        != _compact_sha256(_EXPECTED_SOURCE_CHANGE_KINDS)
        or frozen["workspace_plan_without_self_hash_sha256"]
        != result["workspace_plan"]["plan_without_self_hash_sha256"]
    ):
        raise ControlError("input-v3 frozen source semantic child links differ")
    return result


def _input_v3_live_paths(
    namespace: Namespace,
    *,
    project: Path,
    parent: Path,
) -> dict[str, Path]:
    return {
        "project_root": project,
        "parent": parent,
        "amendment_root": parent.parent,
        "control_root": namespace.control_root,
        "run_root": project / "artifacts" / "runs",
        "config": project / "configs" / "confirmatory_resource_bounded_amended.yaml",
        "manifest": (
            project / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
        ),
        "failed_preflight": namespace.control_root / _HISTORICAL_FAILED_PREFLIGHT_FILENAME,
        "prior_failure": namespace.control_root / _HISTORICAL_PRIOR_FAILURE_FILENAME,
        "invalidation": namespace.control_root / _HISTORICAL_INVALIDATION_FILENAME,
        "qualification": namespace.terminal_qualification,
    }


def _live_json(path: Path, role: str) -> dict[str, Any]:
    encoded = _read_bytes(path, role)
    return _strict_json_object(encoded, role)


def _live_run_state_hashes(project: Path) -> dict[str, Any]:
    root = _real_directory(project / "artifacts" / "runs", "live run-state root")
    files = {
        filename: _file_record(root / filename, f"live run-state {filename}")["sha256"]
        for filename in _RUN_STATE_FILENAMES
    }
    return {
        "root": str(root),
        "files": files,
        "sha256": _compact_sha256(files),
    }


def _derive_live_source_v3(paths: Mapping[str, Path]) -> dict[str, Any]:
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
        if type(record) is dict and type(record.get("path")) is str
    }
    current_records = {
        str(record["path"]): dict(record)
        for record in current.get("artifacts", [])
        if type(record) is dict and type(record.get("path")) is str
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
            "live C-to-D2 execution-source delta differs from the exact 18-path allowlist"
        )
    delta, delta_sha256 = amendment._canonical_source_delta_with_allowlist(
        parent_source,
        current,
        allowlisted_change_kinds=observed,
        role="resource Authority-D replacement-v2",
    )
    records: list[dict[str, Any]] = []
    for logical, change_kind in sorted(observed.items()):
        _normalise_source_path(logical)
        if change_kind == "removed":
            records.append({"path": logical, "change_kind": "removed"})
            continue
        source_record = current_records.get(logical)
        if type(source_record) is not dict:
            raise ControlError(f"live execution-source capture lacks {logical}")
        records.append(
            {
                "path": logical,
                "change_kind": change_kind,
                "size_bytes": _positive_int(
                    source_record.get("size_bytes"),
                    f"live source {logical} size",
                ),
                "sha256": _sha(
                    source_record.get("sha256"),
                    f"live source {logical}",
                ),
            }
        )
    allowlist = _canonical_source_allowlist(
        {
            "schema_version": 1,
            "policy": INPUT_V3_POLICIES["source_allowlist"],
            "file_count": len(records),
            "records": records,
        }
    )
    return {
        "parent_source": parent_source,
        "current_source": current,
        "current_manifest_sha256": hashlib.sha256(_canonical_bytes(current)).hexdigest(),
        "change_kinds": observed,
        "delta": tuple(dict(record) for record in delta),
        "delta_sha256": delta_sha256,
        "allowlist": allowlist,
    }


def _require_live_authority_and_config_v3(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    from histo_audit.config import config_sha256, load_config
    from histo_audit.experiment.study_contracts import (
        validate_resource_bounded_confirmatory_config,
    )
    from histo_audit.workflows import preregistration_amendment as amendment

    verification = amendment.verify_preregistration_amendment(paths["parent"])
    if (
        not verification.valid
        or verification.artifact_root_sha256 != _AUTHORITY_C_ARTIFACT_ROOT_SHA256
        or verification.sha256_manifest_sha256 != _AUTHORITY_C_MANIFEST_SHA256
        or type(verification.chain_depth) is not int
        or verification.chain_depth != 3
    ):
        raise ControlError("Authority C differs from its exact immutable identity")
    _authority_c_receipt(paths["parent"])
    failed_record = _file_record(
        paths["failed_preflight"],
        "failed resource preflight receipt",
    )
    if (
        failed_record["size_bytes"] != DEFAULT_HISTORICAL_PINS.failed_preflight.size_bytes
        or failed_record["sha256"] != DEFAULT_HISTORICAL_PINS.failed_preflight.sha256
    ):
        raise ControlError("failed resource preflight receipt differs from its exact pin")
    failed = _live_json(paths["failed_preflight"], "failed resource preflight receipt")
    config_record = _file_record(paths["config"], "resource confirmatory config")
    if config_record["sha256"] != _RESOURCE_CONFIG_FILE_SHA256:
        raise ControlError("resource confirmatory config differs from its exact file pin")
    config = validate_resource_bounded_confirmatory_config(load_config(paths["config"]))
    if config_sha256(config) != _RESOURCE_CONFIG_SEMANTIC_SHA256:
        raise ControlError("resource confirmatory config differs from its semantic pin")
    manifest_record = _file_record(paths["manifest"], "PanNuke nucleus manifest")
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


def _derive_live_workspace_v3(
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
    canonical_capacity, canonical_plan = amendment.validate_resource_bounded_capacity_v3(
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
        "workspace_plan": _canonical_workspace_plan(plan),
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
    if type(weight_identifier) is not str or not weight_identifier:
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


def _derive_live_cnn_correction_v3(
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
    receipt = _canonical_cnn_receipt(
        {
            "schema_version": 1,
            "policy": INPUT_V3_POLICIES["cnn_correction_receipt"],
            "correction": full,
            "semantic_equivalence_evidence": semantic,
        }
    )
    return {
        "correction": correction,
        "full": full,
        "receipt": receipt,
    }


def _require_resource_gates_v3(
    paths: Mapping[str, Path],
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    from histo_audit.experiment.resource_bounded_runner import (
        require_resource_capacity,
        require_resource_compute,
    )

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
        "capacity": capacity.as_dict(),
        "compute": compute.as_dict(),
    }


class ProductionInputV3Reconstructor:
    """Reconstruct the native inputs from current outcome-blind evidence only."""

    def __init__(self) -> None:
        self.last_context: dict[str, Any] | None = None

    def reconstruct(
        self,
        *,
        namespace: Namespace,
        project_root: Path,
        parent_authority_directory: Path,
        terminal_qualification: Mapping[str, Any],
        terminal_qualification_sha256: str,
    ) -> Mapping[str, Mapping[str, Any]]:
        from histo_audit.workflows import preregistration_amendment as amendment

        public_lineage = (
            amendment.verify_resource_bounded_replacement_terminal_qualification_receipt(
                namespace.terminal_qualification,
                project_root=project_root,
                parent_authority_directory=parent_authority_directory,
            )
        )
        if (
            public_lineage.get("terminal_qualification_receipt") != terminal_qualification
            or public_lineage.get("terminal_qualification_receipt_sha256")
            != terminal_qualification_sha256
            or public_lineage.get("terminal_qualification_receipt_path")
            != str(_absolute(namespace.terminal_qualification))
        ):
            raise ControlError("public schema-v3 terminal lineage differs from Q")
        paths = _input_v3_live_paths(
            namespace,
            project=project_root,
            parent=parent_authority_directory,
        )
        authority = _require_live_authority_and_config_v3(paths)
        source = _derive_live_source_v3(paths)
        workspace = _derive_live_workspace_v3(paths, authority)
        resources = _require_resource_gates_v3(paths, workspace)
        cnn = _derive_live_cnn_correction_v3(
            paths,
            authority,
            source,
            workspace,
        )
        prior = amendment.verify_resource_bounded_prior_publication_failure_receipt(
            superseded_resource_authority_directory=parent_authority_directory,
            receipt_path=paths["prior_failure"],
        )
        invalidation_payload, invalidation_record = _read_pinned_json(
            paths["invalidation"],
            DEFAULT_HISTORICAL_PINS.invalidation,
            "retired v1-input invalidation",
        )
        del invalidation_payload
        authorization = amendment.build_resource_bounded_technical_successor_authorization(
            project_root=project_root,
            superseded_resource_authority_directory=parent_authority_directory,
            resource_confirmatory_config_path=paths["config"],
            failed_preflight_receipt_path=paths["failed_preflight"],
            prior_publication_failure_receipt_path=paths["prior_failure"],
            cnn_provenance_correction=cnn["correction"],
            source_delta_allowlist=source["change_kinds"],
            resource_input_workspace_plan=workspace["workspace_plan"],
            resource_input_workspace_array_specs=workspace["array_specs"],
            resource_input_workspace_index_specs=workspace["index_specs"],
            expected_successor_config_semantic_sha256=(_RESOURCE_CONFIG_SEMANTIC_SHA256),
            replacement_publication_terminal_qualification_receipt_path=(
                namespace.terminal_qualification
            ),
        )
        authorization_dict = authorization.as_dict()
        if (
            authorization_dict.get("schema_version") != 3
            or authorization_dict.get("policy") != SCHEMA_V3_AUTHORIZATION_POLICY
            or authorization_dict.get("replacement_publication_failure_lineage") != public_lineage
        ):
            raise ControlError("technical-successor authorization is not exact schema-v3")
        authorization_source = authorization_dict.get("execution_source_delta")
        if (
            not isinstance(authorization_source, Mapping)
            or authorization_source.get("resource_root_sha256")
            != source["current_source"]["root_sha256"]
            or authorization_source.get("resource_manifest_sha256")
            != source["current_manifest_sha256"]
            or authorization_source.get("delta_sha256") != source["delta_sha256"]
            or authorization_dict.get("cnn_provenance_correction") != cnn["full"]
        ):
            raise ControlError("technical-successor authorization changed live inputs")
        allowlist = source["allowlist"]
        workspace_plan = workspace["workspace_plan"]
        cnn_receipt = cnn["receipt"]
        controller_record = _controller_identity()
        run_state = _live_run_state_hashes(project_root)
        frozen_receipt = {
            "schema_version": 1,
            "policy": INPUT_V3_POLICIES["frozen_source_receipt"],
            "file_count": len(_EXPECTED_SOURCE_CHANGE_KINDS),
            "source_allowlist_sha256": hashlib.sha256(_canonical_bytes(allowlist)).hexdigest(),
            "workspace_plan_sha256": hashlib.sha256(_canonical_bytes(workspace_plan)).hexdigest(),
            "cnn_correction_receipt_sha256": hashlib.sha256(
                _canonical_bytes(cnn_receipt)
            ).hexdigest(),
            "source_allowlist_semantic_sha256": _compact_sha256(allowlist),
            "execution_source_root_sha256": source["current_source"]["root_sha256"],
            "execution_source_manifest_sha256": source["current_manifest_sha256"],
            "execution_source_artifact_count": source["current_source"]["artifact_count"],
            "execution_source_delta_count": len(source["delta"]),
            "execution_source_delta_sha256": source["delta_sha256"],
            "execution_source_change_kinds_sha256": _compact_sha256(source["change_kinds"]),
            "parent_execution_source_root_sha256": _AUTHORITY_C_SOURCE_PINS["root_sha256"],
            "parent_execution_source_manifest_sha256": _AUTHORITY_C_SOURCE_PINS["manifest_sha256"],
            "config_path": str(_absolute(paths["config"])),
            "config_file_sha256": authority["config_record"]["sha256"],
            "config_semantic_sha256": authority["config_semantic_sha256"],
            "manifest_path": str(_absolute(paths["manifest"])),
            "manifest_sha256": authority["manifest_record"]["sha256"],
            "failed_preflight_receipt_path": str(_absolute(paths["failed_preflight"])),
            "failed_preflight_receipt_sha256": authority["failed_record"]["sha256"],
            "prior_failure_receipt_path": prior["receipt_path"],
            "prior_failure_receipt_sha256": prior["receipt_sha256"],
            "retired_input_invalidation_receipt_path": invalidation_record["path"],
            "retired_input_invalidation_receipt_sha256": invalidation_record["sha256"],
            "terminal_qualification_receipt_path": str(_absolute(namespace.terminal_qualification)),
            "terminal_qualification_receipt_sha256": terminal_qualification_sha256,
            "controller_path": controller_record["path"],
            "controller_size_bytes": controller_record["size_bytes"],
            "controller_sha256": controller_record["sha256"],
            "run_state_root": run_state["root"],
            "run_state_files": run_state["files"],
            "run_state_sha256": run_state["sha256"],
            "authorization_sha256": _compact_sha256(authorization_dict),
            "workspace_plan_without_self_hash_sha256": workspace_plan[
                "plan_without_self_hash_sha256"
            ],
            "outcome_value_interpretation_performed": False,
            "scientific_execution_performed": False,
            "publication_performed": False,
        }
        payloads = {
            "source_allowlist": allowlist,
            "workspace_plan": workspace_plan,
            "cnn_correction_receipt": cnn_receipt,
            "frozen_source_receipt": frozen_receipt,
        }
        self.last_context = {
            "paths": paths,
            "authority": authority,
            "source": source,
            "workspace": workspace,
            "resources": resources,
            "cnn": cnn,
            "prior": prior,
            "invalidation_record": invalidation_record,
            "terminal_lineage": public_lineage,
            "technical_authorization": authorization,
            "technical_authorization_dict": authorization_dict,
            "technical_authorization_sha256": _compact_sha256(authorization_dict),
            "payloads": payloads,
        }
        return payloads


def _reconstruct_input_v3(
    namespace: Namespace,
    *,
    project: Path,
    parent: Path,
    reconstructor: InputV3Reconstructor,
    pins: HistoricalPins,
) -> dict[str, dict[str, Any]]:
    qualification, qualification_sha256 = _read_terminal_qualification(
        namespace,
        pins=pins,
    )
    raw = reconstructor.reconstruct(
        namespace=namespace,
        project_root=project,
        parent_authority_directory=parent,
        terminal_qualification=qualification,
        terminal_qualification_sha256=qualification_sha256,
    )
    if type(raw) is not dict:
        raise ControlError("input-v3 reconstructor must return one exact dictionary")
    return _canonical_input_v3_payloads(
        raw,
        namespace=namespace,
        qualification=qualification,
        qualification_sha256=qualification_sha256,
    )


def _read_input_v3(
    namespace: Namespace,
    *,
    pins: HistoricalPins = DEFAULT_HISTORICAL_PINS,
    verify_live: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    qualification, qualification_sha256 = _read_terminal_qualification(
        namespace,
        pins=pins,
        verify_live_history=verify_live,
    )
    root = _real_directory(namespace.input_v3, "active input-v3 bundle")
    observed = tuple(sorted(entry.name for entry in os.scandir(root)))
    expected = tuple(sorted(INPUT_V3_FILENAMES.values()))
    if observed != expected:
        raise ControlError("active input-v3 bundle inventory is not exact")
    raw_payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role, filename in INPUT_V3_FILENAMES.items():
        path = root / filename
        encoded = _read_bytes(path, f"input-v3 {role}")
        payload = _strict_json_object(encoded, f"input-v3 {role}")
        if encoded != _canonical_bytes(payload):
            raise ControlError(f"input-v3 {role} bytes are not canonical")
        raw_payloads[role] = payload
        records[role] = {
            "path": str(_absolute(path)),
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    repeated_inventory = tuple(sorted(entry.name for entry in os.scandir(root)))
    if repeated_inventory != expected:
        raise ControlError("active input-v3 bundle inventory changed during readback")
    payloads = _canonical_input_v3_payloads(
        raw_payloads,
        namespace=namespace,
        qualification=qualification,
        qualification_sha256=qualification_sha256,
    )
    if verify_live:
        project = _project_root(namespace)
        frozen = payloads["frozen_source_receipt"]
        current_controller = _controller_identity()
        if any(
            frozen[frozen_field] != current_controller[record_field]
            for frozen_field, record_field in (
                ("controller_path", "path"),
                ("controller_size_bytes", "size_bytes"),
                ("controller_sha256", "sha256"),
            )
        ):
            raise ControlError("input-v3 controller binding changed")
        run_state = _live_run_state_hashes(project)
        if (
            frozen["run_state_root"] != run_state["root"]
            or frozen["run_state_files"] != run_state["files"]
            or frozen["run_state_sha256"] != run_state["sha256"]
        ):
            raise ControlError("input-v3 run-state binding changed")
        live_records = {
            "config_file_sha256": _file_record(
                Path(frozen["config_path"]),
                "input-v3 live config",
            )["sha256"],
            "manifest_sha256": _file_record(
                Path(frozen["manifest_path"]),
                "input-v3 live manifest",
            )["sha256"],
            "failed_preflight_receipt_sha256": _file_record(
                Path(frozen["failed_preflight_receipt_path"]),
                "input-v3 live failed preflight",
            )["sha256"],
            "prior_failure_receipt_sha256": _file_record(
                Path(frozen["prior_failure_receipt_path"]),
                "input-v3 live prior failure",
            )["sha256"],
            "retired_input_invalidation_receipt_sha256": _file_record(
                Path(frozen["retired_input_invalidation_receipt_path"]),
                "input-v3 live invalidation",
            )["sha256"],
        }
        if any(frozen[field] != digest for field, digest in live_records.items()):
            raise ControlError("input-v3 pinned live record changed")
        parent = project / "artifacts" / "preregistration_amendments" / _AUTHORITY_C_COMPONENT
        live_source = _derive_live_source_v3(
            _input_v3_live_paths(
                namespace,
                project=project,
                parent=parent,
            )
        )
        if (
            frozen["execution_source_root_sha256"] != live_source["current_source"]["root_sha256"]
            or frozen["execution_source_manifest_sha256"] != live_source["current_manifest_sha256"]
            or frozen["execution_source_delta_sha256"] != live_source["delta_sha256"]
            or frozen["execution_source_artifact_count"]
            != live_source["current_source"]["artifact_count"]
        ):
            raise ControlError("input-v3 execution-source binding changed")
        reconstructor = ProductionInputV3Reconstructor()
        independently_reconstructed = _reconstruct_input_v3(
            namespace,
            project=project,
            parent=parent,
            reconstructor=reconstructor,
            pins=pins,
        )
        if independently_reconstructed != payloads:
            raise ControlError("input-v3 differs from full independent live reconstruction")
    for role, filename in INPUT_V3_FILENAMES.items():
        final_bytes = _read_bytes(root / filename, f"final input-v3 {role}")
        if (
            len(final_bytes) != records[role]["size_bytes"]
            or hashlib.sha256(final_bytes).hexdigest() != records[role]["sha256"]
            or final_bytes != _canonical_bytes(payloads[role])
        ):
            raise ControlError(f"input-v3 {role} changed after live verification")
    final_inventory = tuple(sorted(entry.name for entry in os.scandir(root)))
    if final_inventory != expected:
        raise ControlError("active input-v3 bundle changed after live verification")
    return payloads, records, _compact_sha256(records)


def _freeze_input_v3_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    reconstructor: InputV3Reconstructor,
    pins: HistoricalPins,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    project, parent = _require_parent(namespace, parent_authority_directory)
    if os.path.lexists(namespace.input_v3):
        raise FileExistsError("active input-v3 bundle already exists")
    if any(
        os.path.lexists(path)
        for path in (
            namespace.authorization_v2,
            namespace.attempt_v2,
            namespace.success_v2,
            namespace.failure_v2,
        )
    ):
        raise ControlError("input-v3 freeze requires no later protocol asset")
    _stable_amendment_inventory(parent.parent)
    if discover_candidates(parent):
        raise ControlError("input-v3 freeze requires exact A/P/C inventory")
    payloads = _reconstruct_input_v3(
        namespace,
        project=project,
        parent=parent,
        reconstructor=reconstructor,
        pins=pins,
    )
    final_paths = {
        role: namespace.input_v3 / filename for role, filename in INPUT_V3_FILENAMES.items()
    }
    publications: list[PublishedPath] = []
    lock_paths = (*_protocol_lock_paths(namespace, parent=parent), *final_paths.values())
    legacy_lock_paths = _legacy_scoped_lock_paths(namespace, parent=parent)
    with (
        _publication_exclusion_boundary(
            publications,
            role="input-v3 freeze",
        ),
        ExclusiveBundlePublicationLock(
            lock_paths,
            role="resource Authority-D replacement-v2 input-v3 freeze",
        ) as publication_lock,
        ExclusiveBundlePublicationLock(
            (parent,),
            role="resource Authority-D replacement-v2 input-v3 Authority-C parent guard",
        ) as parent_guard,
    ):
        try:
            owned_locks = (publication_lock, parent_guard)
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            publication_lock.assert_owned()
            if os.path.lexists(namespace.input_v3):
                raise FileExistsError("active input-v3 bundle already exists")
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise ControlError("Authority-D candidate appeared before input-v3 freeze")
            repeated = _reconstruct_input_v3(
                namespace,
                project=project,
                parent=parent,
                reconstructor=reconstructor,
                pins=pins,
            )
            if repeated != payloads:
                raise ControlError("input-v3 live reconstruction changed before freeze")
            parent_guard.assert_owned()
            publication_lock.assert_owned()
            publications.append(create_directory_no_overwrite(namespace.input_v3))
            publication_lock.assert_owned()
            for role in (
                "source_allowlist",
                "workspace_plan",
                "cnn_correction_receipt",
                "frozen_source_receipt",
            ):
                publications.append(
                    publish_bytes_no_overwrite(
                        _canonical_bytes(payloads[role]),
                        final_paths[role],
                    )
                )
                publication_lock.assert_owned()
            read_payloads, records, digest = _read_input_v3(
                namespace,
                pins=pins,
                verify_live=type(reconstructor) is ProductionInputV3Reconstructor,
            )
            if read_payloads != payloads or any(
                not publication.still_owned() for publication in publications
            ):
                raise ControlError("input-v3 bundle failed exact post-freeze readback")
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise AmbiguousStateError("Authority-D candidate appeared during input-v3 freeze")
            presence = _reserved_family_presence(namespace)
            if presence != {
                "qualification": True,
                "inputs": True,
                "authorization": False,
                "attempt": False,
                "success": False,
                "failure": False,
            }:
                raise AmbiguousStateError("replacement-v2 state changed during input freeze")
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            (
                final_live_payloads,
                final_live_records,
                final_live_digest,
            ) = _read_input_v3(
                namespace,
                pins=pins,
                verify_live=type(reconstructor) is ProductionInputV3Reconstructor,
            )
            if (
                final_live_payloads != read_payloads
                or final_live_records != records
                or final_live_digest != digest
            ):
                raise AmbiguousStateError("input-v3 live evidence changed before freeze return")
            expected_children = tuple(sorted(INPUT_V3_FILENAMES.values()))
            for _scan_index in range(2):
                observed_children = tuple(
                    sorted(entry.name for entry in os.scandir(namespace.input_v3))
                )
                if observed_children != expected_children:
                    raise AmbiguousStateError(
                        "input-v3 child inventory changed before freeze return"
                    )
                for role, filename in INPUT_V3_FILENAMES.items():
                    final_bytes = _read_bytes(
                        namespace.input_v3 / filename,
                        f"final frozen input-v3 {role}",
                    )
                    if (
                        len(final_bytes) != records[role]["size_bytes"]
                        or hashlib.sha256(final_bytes).hexdigest() != records[role]["sha256"]
                        or final_bytes != _canonical_bytes(read_payloads[role])
                    ):
                        raise AmbiguousStateError(f"input-v3 {role} changed before freeze return")
                if any(not publication.still_owned() for publication in publications):
                    raise AmbiguousStateError(
                        "input-v3 publication ownership changed before freeze return"
                    )
                parent_guard.assert_owned()
                publication_lock.assert_owned()
            parent_guard.assert_owned()
            publication_lock.assert_owned()
            return final_live_payloads, final_live_records, final_live_digest
        except BaseException as exc:
            if publications:
                _rollback_owned_or_ambiguous(owned_locks, publications, exc)
            raise


def freeze_input_v3_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    """Freeze one independently reconstructed native input-v3 singleton."""

    return _freeze_input_v3_once(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        reconstructor=ProductionInputV3Reconstructor(),
        pins=DEFAULT_HISTORICAL_PINS,
    )


def _canonical_external_timestamp(value: object, role: str) -> str:
    if type(value) is not str:
        raise ControlError(f"{role} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ControlError(f"{role} is invalid") from exc
    if parsed.tzinfo is None:
        raise ControlError(f"{role} must be timezone-aware")
    canonical = (
        parsed.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )
    if value != canonical:
        raise ControlError(f"{role} is not canonical six-microsecond UTC text")
    return canonical


def _normalize_external_timestamp(value: object, role: str) -> str:
    """Normalize one trusted live producer timestamp before it enters evidence."""

    if type(value) is not str:
        raise ControlError(f"{role} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ControlError(f"{role} is invalid") from exc
    if parsed.tzinfo is None:
        raise ControlError(f"{role} must be timezone-aware")
    return (
        parsed.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _build_live_preflight_v2(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    amendment_timestamp: datetime,
    authorization_present: bool = False,
) -> dict[str, Any]:
    from histo_audit.workflows import preregistration_amendment as amendment

    project, parent = _require_parent(namespace, parent_authority_directory)
    if amendment_timestamp.tzinfo is None:
        raise ControlError("replacement-v2 preflight timestamp must be timezone-aware")
    moment = amendment_timestamp.astimezone(UTC)
    if moment > datetime.now(UTC):
        raise ControlError("replacement-v2 preflight timestamp cannot be future-dated")
    timestamp_text = _timestamp(moment)
    destination = parent.parent / moment.strftime("%Y%m%dT%H%M%S.%fZ")
    if os.path.lexists(destination):
        raise ControlError("replacement-v2 intended Authority D2 already exists")
    presence = _reserved_family_presence(namespace)
    if type(authorization_present) is not bool:
        raise ControlError("authorization-present expectation must be an exact boolean")
    if presence != {
        "qualification": True,
        "inputs": True,
        "authorization": authorization_present,
        "attempt": False,
        "success": False,
        "failure": False,
    }:
        raise ControlError("authorization preflight requires exact Q+I3 state")
    _stable_amendment_inventory(parent.parent)
    if discover_candidates(parent):
        raise ControlError("authorization preflight requires no Authority-D candidate")
    frozen_payloads, frozen_records, frozen_records_sha256 = _read_input_v3(namespace)
    reconstructor = ProductionInputV3Reconstructor()
    reconstructed = _reconstruct_input_v3(
        namespace,
        project=project,
        parent=parent,
        reconstructor=reconstructor,
        pins=DEFAULT_HISTORICAL_PINS,
    )
    context = reconstructor.last_context
    if reconstructed != frozen_payloads or context is None:
        raise ControlError("frozen input-v3 differs from independent live reconstruction")
    terminal_lineage = context["terminal_lineage"]
    qualified_at = datetime.strptime(
        terminal_lineage["terminal_qualification_receipt"]["qualified_at_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    if qualified_at > moment:
        raise ControlError("replacement-v2 timestamp must follow terminal qualification")
    storage_policy = amendment.require_confirmatory_storage_policy(parent)
    authorization_dict = context["technical_authorization_dict"]
    intent_sha256 = amendment.resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=parent,
        amendment_timestamp_utc=timestamp_text,
        reason=_AMENDMENT_REASON,
        affected_hypotheses=_AFFECTED_HYPOTHESES,
        affected_analyses=_AFFECTED_ANALYSES,
        outcomes_inspected_at_utc=_timestamp(_OUTCOMES_INSPECTED_AT),
        authorization=authorization_dict,
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
        "workspace_plan_sha256": frozen_records["workspace_plan"]["sha256"],
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
    controller = _controller_identity()
    frozen_source = frozen_payloads["frozen_source_receipt"]
    contract = {
        "project_root": str(project),
        "parent_authority_directory": str(parent),
        "controller": controller,
        "terminal_qualification": terminal_lineage,
        "frozen_input_bundle": {
            "directory": str(_absolute(namespace.input_v3)),
            "files": frozen_records,
            "records_sha256": frozen_records_sha256,
        },
        "source": {
            "root_sha256": frozen_source["execution_source_root_sha256"],
            "manifest_sha256": frozen_source["execution_source_manifest_sha256"],
            "delta_sha256": frozen_source["execution_source_delta_sha256"],
            "allowlisted_change_count": len(_EXPECTED_SOURCE_CHANGE_KINDS),
        },
        "config": {
            "path": frozen_source["config_path"],
            "file_sha256": frozen_source["config_file_sha256"],
            "semantic_sha256": frozen_source["config_semantic_sha256"],
        },
        "manifest": {
            "path": frozen_source["manifest_path"],
            "sha256": frozen_source["manifest_sha256"],
        },
        "historical_lineage": {
            "failed_preflight_receipt_path": (frozen_source["failed_preflight_receipt_path"]),
            "failed_preflight_receipt_sha256": (frozen_source["failed_preflight_receipt_sha256"]),
            "prior_failure_receipt_path": frozen_source["prior_failure_receipt_path"],
            "prior_failure_receipt_sha256": frozen_source["prior_failure_receipt_sha256"],
            "retired_input_invalidation_receipt_path": frozen_source[
                "retired_input_invalidation_receipt_path"
            ],
            "retired_input_invalidation_receipt_sha256": frozen_source[
                "retired_input_invalidation_receipt_sha256"
            ],
        },
        "run_state": {
            "root": frozen_source["run_state_root"],
            "files": frozen_source["run_state_files"],
            "sha256": frozen_source["run_state_sha256"],
        },
        "technical_successor": {
            "authorization": authorization_dict,
            "authorization_sha256": context["technical_authorization_sha256"],
            "intent_sha256": intent_sha256,
            "storage_policy": storage_policy,
        },
        "publication": {
            "amendment_timestamp_utc": timestamp_text,
            "intended_authority_directory": str(_absolute(destination)),
            "amendment_schema_version": 5,
            "amendment_purpose": _TECHNICAL_SUCCESSOR_PURPOSE,
            "chain_depth": 4,
        },
        "replacement_state": {
            "state": State.AUTHORIZATION_REQUIRED.value,
            "candidate_count": 0,
            "attempt_marker_absent": True,
            "success_marker_absent": True,
            "failure_marker_absent": True,
            "intended_authority_absent": True,
        },
        "capacity_contract": capacity_contract,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    resources = context["resources"]
    capacity_observation = dict(resources["capacity"])
    compute_observation = dict(resources["compute"])
    if (
        capacity_observation.get("minimum_free_bytes") != required_free
        or type(capacity_observation.get("free_bytes")) is not int
        or capacity_observation["free_bytes"] < required_free
        or capacity_observation.get("passed") is not True
        or compute_observation.get("passed") is not True
    ):
        raise ControlError("live resource observations differ from capacity-v3")
    capacity_observation["checked_at_utc"] = _normalize_external_timestamp(
        capacity_observation["checked_at_utc"],
        "replacement-v2 capacity observation",
    )
    compute_observation["checked_at_utc"] = _normalize_external_timestamp(
        compute_observation["checked_at_utc"],
        "replacement-v2 compute observation",
    )
    return {
        "timestamp": moment,
        "timestamp_text": timestamp_text,
        "destination": destination,
        "context": context,
        "contract": contract,
        "preflight_fingerprint_sha256": _compact_sha256(contract),
        "capacity_observation": capacity_observation,
        "compute_observation": compute_observation,
        "intent_sha256": intent_sha256,
    }


_AUTH_V2_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "authorized_at_utc",
        "authorized_attempt_id",
        "max_attempt_count",
        "automatic_retry_allowed",
        "publication",
        "preflight",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_AUTH_V2_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "contract",
        "preflight_fingerprint_sha256",
        "capacity_observations",
        "compute_observations",
    }
)
_SCHEMA_V3_TECHNICAL_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "purpose",
        "supersedes",
        "prior_publication_failure",
        "failed_preflight",
        "historical_primary",
        "resource_profile",
        "execution_source_delta",
        "cnn_provenance_correction",
        "resource_capacity_policy",
        "resource_input_workspace_plan",
        "expected_successor_config_semantic_sha256",
        "resource_profile_shape",
        "outcomes_inspected",
        "analysis_disposition",
        "outcome_use_policy",
        "original_confirmatory_claim_allowed",
        "study_outcome_eligible",
        "completion_stage",
        "primary_rebinding_allowed",
        "primary_mutation_allowed",
        "automatic_retry_allowed",
        "scientific_profile_change_allowed",
        "replacement_publication_failure_lineage",
    }
)
_SCHEMA_V3_TECHNICAL_MAPPING_FIELDS = (
    "supersedes",
    "prior_publication_failure",
    "failed_preflight",
    "historical_primary",
    "resource_profile",
    "execution_source_delta",
    "cnn_provenance_correction",
    "resource_capacity_policy",
    "resource_input_workspace_plan",
    "resource_profile_shape",
    "replacement_publication_failure_lineage",
)


def _canonical_dynamic_record(
    value: object,
    *,
    path: Path,
    role: str,
) -> dict[str, Any]:
    record = _exact_dict(
        value,
        frozenset({"path", "size_bytes", "sha256"}),
        role,
    )
    _exact_path_text(record["path"], path, f"{role} path")
    _positive_int(record["size_bytes"], f"{role} size")
    _sha(record["sha256"], f"{role} SHA-256")
    return record


def _canonical_schema_v3_technical_authorization(
    value: object,
    *,
    terminal_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    from histo_audit.workflows.preregistration_amendment import (
        validate_resource_bounded_capacity_v3,
    )

    payload = _exact_dict(
        value,
        _SCHEMA_V3_TECHNICAL_AUTHORIZATION_FIELDS,
        "authorization-v2 schema-v3 technical authorization",
    )
    if any(type(payload[field]) is not dict for field in _SCHEMA_V3_TECHNICAL_MAPPING_FIELDS):
        raise ControlError(
            "authorization-v2 schema-v3 technical mappings are not exact dictionaries"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 3
        or payload["policy"] != SCHEMA_V3_AUTHORIZATION_POLICY
        or payload["purpose"] != _TECHNICAL_SUCCESSOR_PURPOSE
        or payload["outcomes_inspected"] is not True
        or payload["analysis_disposition"] != "amended_or_exploratory"
        or payload["outcome_use_policy"]
        != "resource_constraints_only_no_outcome_value_selection_tuning_or_exclusion"
        or payload["original_confirmatory_claim_allowed"] is not False
        or payload["study_outcome_eligible"] is not False
        or payload["completion_stage"] is not None
        or payload["primary_rebinding_allowed"] is not False
        or payload["primary_mutation_allowed"] is not False
        or payload["automatic_retry_allowed"] is not False
        or payload["scientific_profile_change_allowed"] is not False
        or payload["resource_profile_shape"]
        != {
            "planned_required_cells": 24,
            "planned_cnn_cells": 6,
            "planned_cnn_fold_checkpoints": 30,
        }
        or payload["replacement_publication_failure_lineage"] != terminal_lineage
    ):
        raise ControlError("authorization-v2 schema-v3 technical fixed policy differs")
    _sha(
        payload["expected_successor_config_semantic_sha256"],
        "authorization-v2 schema-v3 successor config semantics",
    )
    try:
        canonical_capacity, canonical_workspace = validate_resource_bounded_capacity_v3(
            payload["resource_capacity_policy"],
            payload["resource_input_workspace_plan"],
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ControlError("authorization-v2 schema-v3 capacity/workspace is invalid") from exc
    if (
        canonical_capacity != payload["resource_capacity_policy"]
        or canonical_workspace != payload["resource_input_workspace_plan"]
    ):
        raise ControlError("authorization-v2 schema-v3 capacity/workspace is noncanonical")
    _canonical_bytes(payload)
    return payload


_CAPACITY_OBSERVATION_FIELDS = frozenset(
    {
        "phase",
        "probe_path",
        "free_bytes",
        "minimum_free_bytes",
        "policy_sha256",
        "checked_at_utc",
        "passed",
    }
)
_COMPUTE_OBSERVATION_FIELDS = frozenset(
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
_COMPUTE_PROBE_FIELDS = frozenset(
    {
        "total_host_ram_bytes",
        "available_host_ram_bytes",
        "cuda_available",
        "cuda_device_count",
        "selected_cuda_device_index",
        "selected_cuda_device_name",
        "total_vram_bytes",
        "free_vram_bytes",
        "cudnn_available",
        "amp_available",
        "amp_dtype",
        "weight_identifier",
        "weight_path",
        "weight_present",
        "weight_sha256",
        "smoke_attempted",
        "smoke_completed",
        "smoke_input_shape",
        "smoke_forward_finite",
        "smoke_backward_finite",
        "smoke_peak_allocated_bytes",
        "smoke_error",
    }
)
_COMPUTE_PROBE_INTEGER_FIELDS = (
    "total_host_ram_bytes",
    "available_host_ram_bytes",
    "cuda_device_count",
    "selected_cuda_device_index",
    "total_vram_bytes",
    "free_vram_bytes",
    "smoke_peak_allocated_bytes",
)
_COMPUTE_PROBE_BOOLEAN_FIELDS = (
    "cuda_available",
    "cudnn_available",
    "amp_available",
    "weight_present",
    "smoke_attempted",
    "smoke_completed",
    "smoke_forward_finite",
    "smoke_backward_finite",
)
_COMPUTE_LIVE_STABLE_FIELDS = (
    "total_host_ram_bytes",
    "cuda_available",
    "cuda_device_count",
    "selected_cuda_device_index",
    "selected_cuda_device_name",
    "total_vram_bytes",
    "cudnn_available",
    "amp_available",
    "amp_dtype",
    "weight_identifier",
    "weight_path",
    "weight_present",
    "weight_sha256",
    "smoke_attempted",
    "smoke_completed",
    "smoke_input_shape",
    "smoke_forward_finite",
    "smoke_backward_finite",
    "smoke_error",
)


def _observation_time(
    value: object,
    *,
    role: str,
    latest_allowed: datetime | None,
) -> datetime:
    text = _canonical_external_timestamp(value, role)
    observed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    if latest_allowed is not None and observed > latest_allowed:
        raise ControlError(f"{role} follows its authorization")
    return observed


def _canonical_capacity_observation(
    value: object,
    *,
    project: Path,
    capacity_contract: Mapping[str, Any],
    latest_allowed: datetime | None,
    role: str,
) -> tuple[dict[str, Any], datetime]:
    observation = _exact_dict(value, _CAPACITY_OBSERVATION_FIELDS, role)
    _exact_path_text(
        observation["probe_path"],
        project / "artifacts" / "runs",
        f"{role} probe",
    )
    if (
        observation["phase"] != "guarded_before_workspace_build"
        or type(observation["free_bytes"]) is not int
        or observation["free_bytes"] < 0
        or type(observation["minimum_free_bytes"]) is not int
        or observation["minimum_free_bytes"] <= 0
        or observation["minimum_free_bytes"] != capacity_contract["required_free_bytes"]
        or observation["policy_sha256"] != capacity_contract["resource_capacity_policy_sha256"]
        or observation["passed"] is not True
        or observation["free_bytes"] < observation["minimum_free_bytes"]
    ):
        raise ControlError(f"{role} differs from the exact capacity contract")
    _sha(observation["policy_sha256"], f"{role} policy")
    checked_at = _observation_time(
        observation["checked_at_utc"],
        role=f"{role} time",
        latest_allowed=latest_allowed,
    )
    _canonical_bytes(observation)
    return observation, checked_at


def _canonical_compute_observation(
    value: object,
    *,
    capacity_contract: Mapping[str, Any],
    capacity_policy: Mapping[str, Any],
    latest_allowed: datetime | None,
    role: str,
) -> tuple[dict[str, Any], datetime]:
    observation = _exact_dict(value, _COMPUTE_OBSERVATION_FIELDS, role)
    probe = _exact_dict(
        observation["observation"],
        _COMPUTE_PROBE_FIELDS,
        f"{role} probe",
    )
    for field in _COMPUTE_PROBE_INTEGER_FIELDS:
        _nonnegative_int(probe[field], f"{role} probe {field}")
    if any(type(probe[field]) is not bool for field in _COMPUTE_PROBE_BOOLEAN_FIELDS):
        raise ControlError(f"{role} probe boolean evidence differs")
    for field in ("selected_cuda_device_name", "amp_dtype", "weight_identifier"):
        if type(probe[field]) is not str or not probe[field].strip():
            raise ControlError(f"{role} probe {field} is invalid")
    weight_path = probe["weight_path"]
    if type(weight_path) is not str or weight_path != str(_absolute(weight_path)):
        raise ControlError(f"{role} probe weight path is not canonical absolute text")
    _sha(probe["weight_sha256"], f"{role} probe weight")
    smoke_shape = probe["smoke_input_shape"]
    if (
        type(smoke_shape) is not list
        or any(type(item) is not int or item <= 0 for item in smoke_shape)
        or probe["smoke_error"] is not None
    ):
        raise ControlError(f"{role} probe smoke evidence differs")
    policy_sha256 = _compact_sha256(capacity_policy)
    minimum_available = capacity_policy.get("minimum_available_ram_bytes_before_data")
    policy_integer_fields = (
        "minimum_total_ram_bytes",
        "minimum_available_ram_bytes_before_data",
        "cuda_device_index",
        "minimum_total_vram_bytes",
        "minimum_free_vram_bytes",
        "cuda_smoke_max_peak_allocated_bytes",
    )
    if any(
        type(capacity_policy.get(field)) is not int or capacity_policy[field] < 0
        for field in policy_integer_fields
    ):
        raise ControlError(f"{role} capacity policy integer evidence differs")
    if (
        type(observation["schema_version"]) is not int
        or observation["schema_version"] != 1
        or observation["phase"] != "guarded_before_data_loading"
        or type(observation["minimum_available_ram_bytes"]) is not int
        or observation["minimum_available_ram_bytes"] != minimum_available
        or observation["policy_sha256"] != policy_sha256
        or observation["policy_sha256"] != capacity_contract["resource_capacity_policy_sha256"]
        or observation["observation_sha256"] != _compact_sha256(probe)
        or observation["passed"] is not True
        or observation["outcome_values_read"] is not False
        or observation["prohibited_for_selection_tuning"] is not True
        or observation["adaptive_execution_changes_allowed"] is not False
        or probe["total_host_ram_bytes"] < capacity_policy["minimum_total_ram_bytes"]
        or probe["available_host_ram_bytes"] < minimum_available
        or probe["cuda_available"] is not capacity_policy.get("cuda_required")
        or probe["cuda_device_count"] < 1
        or probe["selected_cuda_device_index"] != capacity_policy["cuda_device_index"]
        or probe["total_vram_bytes"] < capacity_policy["minimum_total_vram_bytes"]
        or probe["free_vram_bytes"] < capacity_policy["minimum_free_vram_bytes"]
        or probe["cudnn_available"] is not capacity_policy.get("cudnn_required")
        or probe["amp_available"] is not capacity_policy.get("amp_required")
        or probe["amp_dtype"] != capacity_policy.get("amp_dtype")
        or probe["smoke_attempted"] is not True
        or probe["smoke_completed"] is not True
        or probe["smoke_input_shape"] != capacity_policy.get("cuda_smoke_input_shape")
        or probe["smoke_forward_finite"] is not True
        or probe["smoke_backward_finite"] is not True
        or probe["smoke_peak_allocated_bytes"]
        > capacity_policy["cuda_smoke_max_peak_allocated_bytes"]
        or probe["weight_identifier"] != capacity_policy.get("official_weight_identifier")
        or probe["weight_present"] is not True
        or probe["weight_sha256"] != capacity_policy.get("official_weight_sha256")
        or capacity_policy.get("implicit_weight_download_allowed") is not False
    ):
        raise ControlError(f"{role} differs from the exact compute contract")
    _sha(observation["policy_sha256"], f"{role} policy")
    _sha(observation["observation_sha256"], f"{role} probe")
    checked_at = _observation_time(
        observation["checked_at_utc"],
        role=f"{role} time",
        latest_allowed=latest_allowed,
    )
    _canonical_bytes(observation)
    return observation, checked_at


def _require_live_observation_cross_links(
    *,
    stored_capacity: Sequence[Mapping[str, Any]],
    stored_compute: Sequence[Mapping[str, Any]],
    live_capacity: Mapping[str, Any],
    live_compute: Mapping[str, Any],
) -> None:
    capacity_fields = (
        "phase",
        "probe_path",
        "minimum_free_bytes",
        "policy_sha256",
        "passed",
    )
    compute_fields = (
        "schema_version",
        "phase",
        "minimum_available_ram_bytes",
        "policy_sha256",
        "passed",
        "outcome_values_read",
        "prohibited_for_selection_tuning",
        "adaptive_execution_changes_allowed",
    )
    if any(
        any(item[field] != live_capacity[field] for field in capacity_fields)
        for item in stored_capacity
    ):
        raise ControlError("stored capacity observations differ from the live contract")
    if any(
        any(item[field] != live_compute[field] for field in compute_fields)
        for item in stored_compute
    ):
        raise ControlError("stored compute observations differ from the live contract")
    live_probe = live_compute["observation"]
    if any(
        any(
            item["observation"][field] != live_probe[field] for field in _COMPUTE_LIVE_STABLE_FIELDS
        )
        for item in stored_compute
    ):
        raise ControlError("stored compute observations differ from live stable hardware")


def _canonical_publication_authorization_v2(
    value: object,
    *,
    namespace: Namespace,
    verify_live_controller: bool = True,
) -> dict[str, Any]:
    payload = _exact_dict(value, _AUTH_V2_FIELDS, "publication authorization-v2")
    project = _project_root(namespace)
    parent = project / "artifacts" / "preregistration_amendments" / _AUTHORITY_C_COMPONENT
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["policy"] != PUBLICATION_AUTHORIZATION_V2_POLICY
        or payload["status"] != "authorized_for_one_attempt"
        or type(payload["max_attempt_count"]) is not int
        or payload["max_attempt_count"] != 1
        or payload["automatic_retry_allowed"] is not False
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("publication authorization-v2 fixed policy differs")
    attempt_id = _sha(payload["authorized_attempt_id"], "authorized v2 attempt ID")
    authorized_text = _canonical_timestamp(
        payload["authorized_at_utc"],
        "publication authorization-v2 time",
    )
    authorized_at = datetime.strptime(
        authorized_text,
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    if authorized_at > datetime.now(UTC):
        raise ControlError("publication authorization-v2 cannot be future-dated")
    publication = _exact_dict(
        payload["publication"],
        frozenset(
            {
                "amendment_timestamp_utc",
                "intended_authority_directory",
                "parent_authority_directory",
                "amendment_schema_version",
                "amendment_purpose",
                "chain_depth",
            }
        ),
        "publication authorization-v2 publication",
    )
    amendment_text = _canonical_timestamp(
        publication["amendment_timestamp_utc"],
        "authorized D2 timestamp",
    )
    amendment_at = datetime.strptime(
        amendment_text,
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    destination = parent.parent / amendment_at.strftime("%Y%m%dT%H%M%S.%fZ")
    _exact_path_text(
        publication["parent_authority_directory"],
        parent,
        "authorized D2 parent",
    )
    _exact_path_text(
        publication["intended_authority_directory"],
        destination,
        "authorized D2 destination",
    )
    if (
        type(publication["amendment_schema_version"]) is not int
        or publication["amendment_schema_version"] != 5
        or publication["amendment_purpose"] != _TECHNICAL_SUCCESSOR_PURPOSE
        or type(publication["chain_depth"]) is not int
        or publication["chain_depth"] != 4
        or amendment_at > authorized_at
    ):
        raise ControlError("publication authorization-v2 publication contract differs")
    preflight = _exact_dict(
        payload["preflight"],
        _AUTH_V2_PREFLIGHT_FIELDS,
        "publication authorization-v2 preflight",
    )
    if (
        type(preflight["schema_version"]) is not int
        or preflight["schema_version"] != 2
        or preflight["policy"] != _LIVE_PREFLIGHT_V2_POLICY
        or preflight["status"] != "passed_twice"
        or type(preflight["contract"]) is not dict
        or preflight["preflight_fingerprint_sha256"] != _compact_sha256(preflight["contract"])
        or type(preflight["capacity_observations"]) is not list
        or type(preflight["compute_observations"]) is not list
        or len(preflight["capacity_observations"]) != 2
        or len(preflight["compute_observations"]) != 2
    ):
        raise ControlError("publication authorization-v2 preflight envelope differs")
    _sha(
        preflight["preflight_fingerprint_sha256"],
        "publication authorization-v2 preflight fingerprint",
    )
    contract = _exact_dict(
        preflight["contract"],
        frozenset(
            {
                "project_root",
                "parent_authority_directory",
                "controller",
                "terminal_qualification",
                "frozen_input_bundle",
                "source",
                "config",
                "manifest",
                "historical_lineage",
                "run_state",
                "technical_successor",
                "publication",
                "replacement_state",
                "capacity_contract",
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            }
        ),
        "publication authorization-v2 stable contract",
    )
    _exact_path_text(contract["project_root"], project, "authorization-v2 project")
    _exact_path_text(
        contract["parent_authority_directory"],
        parent,
        "authorization-v2 parent",
    )
    if contract["publication"] != publication:
        raise ControlError("authorization-v2 publication copies differ")
    controller = _canonical_dynamic_record(
        contract["controller"],
        path=Path(__file__),
        role="authorization-v2 controller",
    )
    if verify_live_controller and controller != _controller_identity():
        raise ControlError("authorization-v2 controller changed")
    terminal = _exact_dict(
        contract["terminal_qualification"],
        FAILURE_LINEAGE_ENVELOPE_FIELDS,
        "authorization-v2 terminal lineage",
    )
    _exact_path_text(
        terminal["terminal_qualification_receipt_path"],
        namespace.terminal_qualification,
        "authorization-v2 terminal receipt",
    )
    _sha(
        terminal["terminal_qualification_receipt_sha256"],
        "authorization-v2 terminal receipt",
    )
    terminal_receipt = _canonical_terminal_receipt(
        terminal["terminal_qualification_receipt"],
        namespace=namespace,
        pins=DEFAULT_HISTORICAL_PINS,
    )
    if (
        terminal["terminal_qualification_receipt"] != terminal_receipt
        or terminal["terminal_qualification_receipt_sha256"]
        != hashlib.sha256(_canonical_bytes(terminal_receipt)).hexdigest()
    ):
        raise ControlError("authorization-v2 terminal receipt envelope differs")
    frozen = _exact_dict(
        contract["frozen_input_bundle"],
        frozenset({"directory", "files", "records_sha256"}),
        "authorization-v2 frozen input bundle",
    )
    _exact_path_text(frozen["directory"], namespace.input_v3, "authorization-v2 input-v3")
    if type(frozen["files"]) is not dict or set(frozen["files"]) != set(INPUT_V3_FILENAMES):
        raise ControlError("authorization-v2 frozen input roles differ")
    canonical_frozen_files = {
        role: _canonical_dynamic_record(
            frozen["files"][role],
            path=namespace.input_v3 / INPUT_V3_FILENAMES[role],
            role=f"authorization-v2 input-v3 {role}",
        )
        for role in INPUT_V3_FILENAMES
    }
    if (
        frozen["records_sha256"] != _compact_sha256(canonical_frozen_files)
        or frozen["files"] != canonical_frozen_files
    ):
        raise ControlError("authorization-v2 frozen input root differs")
    static_input_payloads, static_input_records, static_input_root = _read_input_v3(
        namespace,
        verify_live=False,
    )
    if (
        static_input_records != canonical_frozen_files
        or static_input_root != frozen["records_sha256"]
    ):
        raise ControlError("authorization-v2 input-v3 sealed readback differs")
    source = _exact_dict(
        contract["source"],
        frozenset(
            {
                "root_sha256",
                "manifest_sha256",
                "delta_sha256",
                "allowlisted_change_count",
            }
        ),
        "authorization-v2 source",
    )
    for field in ("root_sha256", "manifest_sha256", "delta_sha256"):
        _sha(source[field], f"authorization-v2 source {field}")
    if type(source["allowlisted_change_count"]) is not int or source[
        "allowlisted_change_count"
    ] != len(_EXPECTED_SOURCE_CHANGE_KINDS):
        raise ControlError("authorization-v2 source cardinality differs")
    config = _exact_dict(
        contract["config"],
        frozenset({"path", "file_sha256", "semantic_sha256"}),
        "authorization-v2 config",
    )
    config_path = project / "configs" / "confirmatory_resource_bounded_amended.yaml"
    _exact_path_text(config["path"], config_path, "authorization-v2 config path")
    if (
        config["file_sha256"] != _RESOURCE_CONFIG_FILE_SHA256
        or config["semantic_sha256"] != _RESOURCE_CONFIG_SEMANTIC_SHA256
    ):
        raise ControlError("authorization-v2 config pins differ")
    manifest = _exact_dict(
        contract["manifest"],
        frozenset({"path", "sha256"}),
        "authorization-v2 manifest",
    )
    manifest_path = project / "data" / "manifests" / "pannuke" / "pannuke_nucleus_manifest.parquet"
    _exact_path_text(manifest["path"], manifest_path, "authorization-v2 manifest path")
    if manifest["sha256"] != _PANNUKE_MANIFEST_SHA256:
        raise ControlError("authorization-v2 manifest pin differs")
    history = _exact_dict(
        contract["historical_lineage"],
        frozenset(
            {
                "failed_preflight_receipt_path",
                "failed_preflight_receipt_sha256",
                "prior_failure_receipt_path",
                "prior_failure_receipt_sha256",
                "retired_input_invalidation_receipt_path",
                "retired_input_invalidation_receipt_sha256",
            }
        ),
        "authorization-v2 historical lineage",
    )
    history_specs = (
        (
            "failed_preflight_receipt",
            namespace.control_root / _HISTORICAL_FAILED_PREFLIGHT_FILENAME,
            DEFAULT_HISTORICAL_PINS.failed_preflight,
        ),
        (
            "prior_failure_receipt",
            namespace.control_root / _HISTORICAL_PRIOR_FAILURE_FILENAME,
            DEFAULT_HISTORICAL_PINS.prior_failure,
        ),
        (
            "retired_input_invalidation_receipt",
            namespace.control_root / _HISTORICAL_INVALIDATION_FILENAME,
            DEFAULT_HISTORICAL_PINS.invalidation,
        ),
    )
    for stem, expected_path, pin in history_specs:
        _exact_path_text(
            history[f"{stem}_path"],
            expected_path,
            f"authorization-v2 {stem}",
        )
        if history[f"{stem}_sha256"] != pin.sha256:
            raise ControlError(f"authorization-v2 {stem} pin differs")
    run_state = _exact_dict(
        contract["run_state"],
        frozenset({"root", "files", "sha256"}),
        "authorization-v2 run state",
    )
    _exact_path_text(
        run_state["root"],
        project / "artifacts" / "runs",
        "authorization-v2 run-state root",
    )
    q_run_state = terminal_receipt["run_state"]
    expected_run_hashes = {
        filename: q_run_state["files"][filename]["sha256"] for filename in _RUN_STATE_FILENAMES
    }
    if (
        type(run_state["files"]) is not dict
        or run_state["files"] != expected_run_hashes
        or run_state["sha256"] != q_run_state["sha256"]
        or run_state["sha256"] != _compact_sha256(run_state["files"])
    ):
        raise ControlError("authorization-v2 run-state cross-link differs")
    technical = _exact_dict(
        contract["technical_successor"],
        frozenset(
            {
                "authorization",
                "authorization_sha256",
                "intent_sha256",
                "storage_policy",
            }
        ),
        "authorization-v2 technical successor",
    )
    canonical_technical_authorization = _canonical_schema_v3_technical_authorization(
        technical["authorization"],
        terminal_lineage=terminal,
    )
    if (
        technical["authorization"] != canonical_technical_authorization
        or type(technical["storage_policy"]) is not dict
        or not technical["storage_policy"]
        or technical["authorization_sha256"] != _compact_sha256(canonical_technical_authorization)
        or technical["authorization_sha256"]
        != static_input_payloads["frozen_source_receipt"]["authorization_sha256"]
    ):
        raise ControlError("authorization-v2 technical successor differs")
    technical_source = canonical_technical_authorization["execution_source_delta"]
    if (
        type(technical_source) is not dict
        or technical_source.get("resource_root_sha256") != source["root_sha256"]
        or technical_source.get("resource_manifest_sha256") != source["manifest_sha256"]
        or technical_source.get("delta_sha256") != source["delta_sha256"]
    ):
        raise ControlError("authorization-v2 technical source cross-link differs")
    _canonical_bytes(technical["storage_policy"])
    _sha(technical["authorization_sha256"], "authorization-v2 technical authorization")
    _sha(technical["intent_sha256"], "authorization-v2 intent")
    replacement = _exact_dict(
        contract["replacement_state"],
        frozenset(
            {
                "state",
                "candidate_count",
                "attempt_marker_absent",
                "success_marker_absent",
                "failure_marker_absent",
                "intended_authority_absent",
            }
        ),
        "authorization-v2 replacement state",
    )
    if (
        replacement["state"] != State.AUTHORIZATION_REQUIRED.value
        or type(replacement["candidate_count"]) is not int
        or replacement["candidate_count"] != 0
        or any(
            replacement[field] is not True
            for field in (
                "attempt_marker_absent",
                "success_marker_absent",
                "failure_marker_absent",
                "intended_authority_absent",
            )
        )
    ):
        raise ControlError("authorization-v2 replacement state differs")
    capacity = _exact_dict(
        contract["capacity_contract"],
        frozenset(
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
        ),
        "authorization-v2 capacity contract",
    )
    for field in (
        "resource_capacity_policy_sha256",
        "workspace_plan_sha256",
        "workspace_plan_without_self_hash_sha256",
    ):
        _sha(capacity[field], f"authorization-v2 capacity {field}")
    for field in (
        "projected_stable_run_bytes",
        "fixed_safety_margin_bytes",
        "minimum_free_bytes_before_tracker",
        "maximum_workspace_bytes",
        "minimum_free_bytes_before_workspace_build",
        "planned_workspace_bytes",
        "required_free_bytes_before",
        "required_free_bytes",
    ):
        _positive_int(capacity[field], f"authorization-v2 capacity {field}")
    if (
        capacity["workspace_plan_sha256"] != canonical_frozen_files["workspace_plan"]["sha256"]
        or capacity["fixed_safety_margin_bytes"] != 10 * 1024**3
        or capacity["minimum_free_bytes_before_tracker"]
        != capacity["projected_stable_run_bytes"] + capacity["fixed_safety_margin_bytes"]
        or capacity["minimum_free_bytes_before_workspace_build"]
        != capacity["minimum_free_bytes_before_tracker"] + capacity["maximum_workspace_bytes"]
        or capacity["required_free_bytes"]
        != max(
            capacity["minimum_free_bytes_before_workspace_build"],
            capacity["required_free_bytes_before"],
        )
    ):
        raise ControlError("authorization-v2 capacity arithmetic differs")
    frozen_source = static_input_payloads["frozen_source_receipt"]
    workspace_plan = static_input_payloads["workspace_plan"]
    expected_controller = {
        "path": frozen_source["controller_path"],
        "size_bytes": frozen_source["controller_size_bytes"],
        "sha256": frozen_source["controller_sha256"],
    }
    expected_history = {
        "failed_preflight_receipt_path": frozen_source["failed_preflight_receipt_path"],
        "failed_preflight_receipt_sha256": frozen_source["failed_preflight_receipt_sha256"],
        "prior_failure_receipt_path": frozen_source["prior_failure_receipt_path"],
        "prior_failure_receipt_sha256": frozen_source["prior_failure_receipt_sha256"],
        "retired_input_invalidation_receipt_path": frozen_source[
            "retired_input_invalidation_receipt_path"
        ],
        "retired_input_invalidation_receipt_sha256": frozen_source[
            "retired_input_invalidation_receipt_sha256"
        ],
    }
    expected_run_state = {
        "root": frozen_source["run_state_root"],
        "files": frozen_source["run_state_files"],
        "sha256": frozen_source["run_state_sha256"],
    }
    capacity_policy = canonical_technical_authorization["resource_capacity_policy"]
    if (
        terminal["terminal_qualification_receipt_sha256"]
        != frozen_source["terminal_qualification_receipt_sha256"]
        or controller != expected_controller
        or source["root_sha256"] != frozen_source["execution_source_root_sha256"]
        or source["manifest_sha256"] != frozen_source["execution_source_manifest_sha256"]
        or source["delta_sha256"] != frozen_source["execution_source_delta_sha256"]
        or source["allowlisted_change_count"] != frozen_source["execution_source_delta_count"]
        or config["path"] != frozen_source["config_path"]
        or config["file_sha256"] != frozen_source["config_file_sha256"]
        or config["semantic_sha256"] != frozen_source["config_semantic_sha256"]
        or manifest["path"] != frozen_source["manifest_path"]
        or manifest["sha256"] != frozen_source["manifest_sha256"]
        or history != expected_history
        or run_state != expected_run_state
        or capacity["resource_capacity_policy_sha256"] != _compact_sha256(capacity_policy)
        or capacity["workspace_plan_without_self_hash_sha256"]
        != frozen_source["workspace_plan_without_self_hash_sha256"]
        or capacity["workspace_plan_without_self_hash_sha256"]
        != workspace_plan["plan_without_self_hash_sha256"]
        or capacity["planned_workspace_bytes"] != workspace_plan["planned_workspace_bytes"]
        or capacity["required_free_bytes_before"] != workspace_plan["required_free_bytes_before"]
        or capacity["projected_stable_run_bytes"] != capacity_policy["projected_stable_run_bytes"]
        or capacity["fixed_safety_margin_bytes"] != capacity_policy["fixed_safety_margin_bytes"]
        or capacity["minimum_free_bytes_before_tracker"]
        != capacity_policy["minimum_free_bytes_before_tracker"]
        or capacity["maximum_workspace_bytes"] != capacity_policy["maximum_workspace_bytes"]
        or capacity["minimum_free_bytes_before_workspace_build"]
        != capacity_policy["minimum_free_bytes_before_workspace_build"]
    ):
        raise ControlError("authorization-v2 sealed Q/I3 contract cross-link differs")
    from histo_audit.workflows import preregistration_amendment as amendment

    # Live pre-D qualification independently reads and authenticates C through
    # ``require_confirmatory_storage_policy``.  Static post-D readback must not
    # bypass the effective-authority boundary with the private historical reader;
    # compare the sealed authorization against the public closed policy type.
    sealed_storage_policy = amendment.ConfirmatoryStoragePolicy().as_dict()
    recomputed_intent_sha256 = amendment.resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=parent,
        amendment_timestamp_utc=publication["amendment_timestamp_utc"],
        reason=_AMENDMENT_REASON,
        affected_hypotheses=_AFFECTED_HYPOTHESES,
        affected_analyses=_AFFECTED_ANALYSES,
        outcomes_inspected_at_utc=_timestamp(_OUTCOMES_INSPECTED_AT),
        authorization=canonical_technical_authorization,
        confirmatory_storage_policy=sealed_storage_policy,
    )
    if (
        technical["storage_policy"] != sealed_storage_policy
        or technical["intent_sha256"] != recomputed_intent_sha256
    ):
        raise ControlError("authorization-v2 sealed storage or intent binding differs")
    for flag in (
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    ):
        if contract[flag] is not False:
            raise ControlError("authorization-v2 stable contract execution flags differ")
    capacity_policy = canonical_technical_authorization["resource_capacity_policy"]
    if (
        type(capacity_policy) is not dict
        or _compact_sha256(capacity_policy) != capacity["resource_capacity_policy_sha256"]
    ):
        raise ControlError("authorization-v2 capacity policy cross-link differs")
    capacity_observations: list[dict[str, Any]] = []
    capacity_times: list[datetime] = []
    for index, raw in enumerate(preflight["capacity_observations"]):
        canonical, observed_at = _canonical_capacity_observation(
            raw,
            project=project,
            capacity_contract=capacity,
            latest_allowed=authorized_at,
            role=f"authorization-v2 capacity observation {index + 1}",
        )
        capacity_observations.append(canonical)
        capacity_times.append(observed_at)
    compute_observations: list[dict[str, Any]] = []
    compute_times: list[datetime] = []
    for index, raw in enumerate(preflight["compute_observations"]):
        canonical, observed_at = _canonical_compute_observation(
            raw,
            capacity_contract=capacity,
            capacity_policy=capacity_policy,
            latest_allowed=authorized_at,
            role=f"authorization-v2 compute observation {index + 1}",
        )
        compute_observations.append(canonical)
        compute_times.append(observed_at)
    if (
        preflight["capacity_observations"] != capacity_observations
        or preflight["compute_observations"] != compute_observations
        or not (
            amendment_at
            <= capacity_times[0]
            <= compute_times[0]
            <= capacity_times[1]
            <= compute_times[1]
            <= authorized_at
        )
        or capacity_times[0] == capacity_times[1]
        or compute_times[0] == compute_times[1]
    ):
        raise ControlError(
            "authorization-v2 observations are not two ordered independent preflights"
        )
    del attempt_id
    return payload


def _read_publication_authorization_v2(
    namespace: Namespace,
    *,
    verify_live: bool = True,
) -> tuple[dict[str, Any], str]:
    encoded = _read_bytes(
        namespace.authorization_v2,
        "publication authorization-v2",
    )
    payload = _canonical_publication_authorization_v2(
        _strict_json_object(encoded, "publication authorization-v2"),
        namespace=namespace,
        verify_live_controller=verify_live,
    )
    if encoded != _canonical_bytes(payload):
        raise ControlError("publication authorization-v2 bytes are not canonical")
    if verify_live:
        timestamp = datetime.strptime(
            payload["publication"]["amendment_timestamp_utc"],
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ).replace(tzinfo=UTC)
        repeated = _build_live_preflight_v2(
            namespace=namespace,
            parent_authority_directory=payload["publication"]["parent_authority_directory"],
            amendment_timestamp=timestamp,
            authorization_present=True,
        )
        preflight = payload["preflight"]
        if (
            repeated["contract"] != preflight["contract"]
            or repeated["preflight_fingerprint_sha256"] != preflight["preflight_fingerprint_sha256"]
            or repeated["intent_sha256"]
            != preflight["contract"]["technical_successor"]["intent_sha256"]
            or str(_absolute(repeated["destination"]))
            != payload["publication"]["intended_authority_directory"]
        ):
            raise ControlError("publication authorization-v2 live contract changed")
        _require_live_observation_cross_links(
            stored_capacity=preflight["capacity_observations"],
            stored_compute=preflight["compute_observations"],
            live_capacity=repeated["capacity_observation"],
            live_compute=repeated["compute_observation"],
        )
    return payload, hashlib.sha256(encoded).hexdigest()


def _authorize_publication_v2_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any], str]:
    _project, parent = _require_parent(namespace, parent_authority_directory)
    if _reserved_family_presence(namespace) != {
        "qualification": True,
        "inputs": True,
        "authorization": False,
        "attempt": False,
        "success": False,
        "failure": False,
    }:
        raise ControlError("publication authorization-v2 requires exact Q+I3 state")
    proposed_at = clock()
    if (
        not isinstance(proposed_at, datetime)
        or proposed_at.tzinfo is None
        or proposed_at.astimezone(UTC) > datetime.now(UTC)
    ):
        raise ControlError("publication authorization-v2 clock is not trusted")
    proposed_at = proposed_at.astimezone(UTC)
    first = _build_live_preflight_v2(
        namespace=namespace,
        parent_authority_directory=parent,
        amendment_timestamp=proposed_at,
    )
    publications: list[PublishedPath] = []
    legacy_lock_paths = _legacy_scoped_lock_paths(namespace, parent=parent)
    with (
        _publication_exclusion_boundary(
            publications,
            role="publication authorization-v2",
        ),
        _protocol_lock(
            namespace,
            parent=parent,
            role="resource Authority-D replacement-v2 publication authorization",
        ) as publication_lock,
        ExclusiveBundlePublicationLock(
            (parent,),
            role="resource Authority-D replacement-v2 authorization Authority-C parent guard",
        ) as parent_guard,
    ):
        owned_locks = (publication_lock, parent_guard)
        try:
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            if _reserved_family_presence(namespace) != {
                "qualification": True,
                "inputs": True,
                "authorization": False,
                "attempt": False,
                "success": False,
                "failure": False,
            }:
                raise ControlError("replacement-v2 state changed before authorization")
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise ControlError("Authority-D candidate appeared before authorization")
            second = _build_live_preflight_v2(
                namespace=namespace,
                parent_authority_directory=parent,
                amendment_timestamp=proposed_at,
            )
            for field in (
                "contract",
                "preflight_fingerprint_sha256",
                "intent_sha256",
                "destination",
            ):
                if first[field] != second[field]:
                    raise ControlError(
                        "two authorization preflights have different stable contracts"
                    )
            authorized_at = clock()
            if (
                not isinstance(authorized_at, datetime)
                or authorized_at.tzinfo is None
                or authorized_at.astimezone(UTC) < proposed_at
                or authorized_at.astimezone(UTC) > datetime.now(UTC)
            ):
                raise ControlError("publication authorization-v2 final clock is invalid")
            contract = second["contract"]
            receipt = {
                "schema_version": 2,
                "policy": PUBLICATION_AUTHORIZATION_V2_POLICY,
                "status": "authorized_for_one_attempt",
                "authorized_at_utc": _timestamp(authorized_at),
                "authorized_attempt_id": secrets.token_hex(32),
                "max_attempt_count": 1,
                "automatic_retry_allowed": False,
                "publication": contract["publication"],
                "preflight": {
                    "schema_version": 2,
                    "policy": _LIVE_PREFLIGHT_V2_POLICY,
                    "status": "passed_twice",
                    "contract": contract,
                    "preflight_fingerprint_sha256": second["preflight_fingerprint_sha256"],
                    "capacity_observations": [
                        first["capacity_observation"],
                        second["capacity_observation"],
                    ],
                    "compute_observations": [
                        first["compute_observation"],
                        second["compute_observation"],
                    ],
                },
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            }
            canonical = _canonical_publication_authorization_v2(
                receipt,
                namespace=namespace,
            )
            if canonical != receipt:
                raise ControlError("publication authorization-v2 builder is noncanonical")
            for lock in owned_locks:
                lock.assert_owned()
            publications.append(
                publish_bytes_no_overwrite(
                    _canonical_bytes(receipt),
                    namespace.authorization_v2,
                )
            )
            for lock in owned_locks:
                lock.assert_owned()
            if _reserved_family_presence(namespace) != {
                "qualification": True,
                "inputs": True,
                "authorization": True,
                "attempt": False,
                "success": False,
                "failure": False,
            }:
                raise AmbiguousStateError("replacement-v2 state changed during authorization")
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise AmbiguousStateError("Authority-D candidate appeared during authorization")
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            readback, digest = _read_publication_authorization_v2(
                namespace,
                verify_live=True,
            )
            if _reserved_family_presence(namespace) != {
                "qualification": True,
                "inputs": True,
                "authorization": True,
                "attempt": False,
                "success": False,
                "failure": False,
            }:
                raise AmbiguousStateError(
                    "replacement-v2 state changed during final authorization readback"
                )
            _stable_amendment_inventory(parent.parent)
            if discover_candidates(parent):
                raise AmbiguousStateError(
                    "Authority-D candidate appeared during final authorization readback"
                )
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            final_encoded = _read_bytes(
                namespace.authorization_v2,
                "final publication authorization-v2",
            )
            if any(not publication.still_owned() for publication in publications) or (
                readback != receipt
                or digest != publications[0].sha256
                or final_encoded != _canonical_bytes(receipt)
                or hashlib.sha256(final_encoded).hexdigest() != digest
            ):
                raise AmbiguousStateError("publication authorization-v2 changed before return")
            for lock in owned_locks:
                lock.assert_owned()
            return readback, digest
        except BaseException as exc:
            if publications:
                _rollback_owned_or_ambiguous(owned_locks, publications, exc)
            raise


def authorize_publication_v2_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> tuple[dict[str, Any], str]:
    """Publish one one-attempt authorization after two outcome-blind preflights."""

    return _authorize_publication_v2_once(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        clock=lambda: datetime.now(UTC),
    )


_ATTEMPT_V2_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "claimed_at_utc",
        "attempt_id",
        "max_attempt_count",
        "automatic_retry_allowed",
        "publication_authorization_receipt",
        "terminal_qualification",
        "frozen_input_bundle",
        "parent_authority_directory",
        "intended_authority_directory",
        "amendment_timestamp_utc",
        "controller",
        "source",
        "run_state",
        "technical_authorization_sha256",
        "intent_sha256",
        "verification_nonce",
        "preflight_fingerprint_sha256",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)


def _verification_nonce_v2(
    authorization: Mapping[str, Any],
    authorization_receipt_sha256: str,
) -> str:
    publication = authorization["publication"]
    technical = authorization["preflight"]["contract"]["technical_successor"]
    return _compact_sha256(
        {
            "domain": "aanca-resource-authority-d-replacement-v2-verification-nonce",
            "authorized_attempt_id": authorization["authorized_attempt_id"],
            "publication_authorization_receipt_sha256": _sha(
                authorization_receipt_sha256,
                "replacement-v2 nonce authorization receipt",
            ),
            "intent_sha256": technical["intent_sha256"],
            "intended_authority_directory": publication["intended_authority_directory"],
        }
    )


def _canonical_attempt_v2(
    value: object,
    *,
    namespace: Namespace,
    authorization: Mapping[str, Any],
    authorization_receipt_sha256: str,
    verify_live_controller: bool = True,
) -> dict[str, Any]:
    payload = _exact_dict(value, _ATTEMPT_V2_FIELDS, "replacement-v2 attempt marker")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["policy"] != ATTEMPT_V2_POLICY
        or payload["status"] != "claimed"
        or type(payload["max_attempt_count"]) is not int
        or payload["max_attempt_count"] != 1
        or payload["automatic_retry_allowed"] is not False
        or payload["attempt_id"] != authorization["authorized_attempt_id"]
        or payload["verification_nonce"]
        != _verification_nonce_v2(
            authorization,
            authorization_receipt_sha256,
        )
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("replacement-v2 attempt fixed policy differs")
    _sha(payload["attempt_id"], "replacement-v2 attempt ID")
    claimed_at = datetime.strptime(
        _canonical_timestamp(payload["claimed_at_utc"], "replacement-v2 claim time"),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    authorized_at = datetime.strptime(
        authorization["authorized_at_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    if claimed_at < authorized_at or claimed_at > datetime.now(UTC):
        raise ControlError("replacement-v2 claim timestamp is out of order")
    authorization_record = _canonical_dynamic_record(
        payload["publication_authorization_receipt"],
        path=namespace.authorization_v2,
        role="replacement-v2 authorization receipt",
    )
    if authorization_record["sha256"] != authorization_receipt_sha256:
        raise ControlError("replacement-v2 attempt authorization pin differs")
    preflight = authorization["preflight"]
    contract = preflight["contract"]
    publication = authorization["publication"]
    terminal = _exact_dict(
        payload["terminal_qualification"],
        frozenset({"path", "sha256"}),
        "replacement-v2 attempt terminal qualification",
    )
    _exact_path_text(
        terminal["path"],
        namespace.terminal_qualification,
        "replacement-v2 attempt terminal qualification",
    )
    if (
        terminal["sha256"]
        != contract["terminal_qualification"]["terminal_qualification_receipt_sha256"]
    ):
        raise ControlError("replacement-v2 attempt terminal qualification differs")
    _exact_path_text(
        payload["parent_authority_directory"],
        publication["parent_authority_directory"],
        "replacement-v2 attempt parent",
    )
    _exact_path_text(
        payload["intended_authority_directory"],
        publication["intended_authority_directory"],
        "replacement-v2 attempt destination",
    )
    if payload["amendment_timestamp_utc"] != publication["amendment_timestamp_utc"]:
        raise ControlError("replacement-v2 attempt timestamp differs")
    _canonical_timestamp(
        payload["amendment_timestamp_utc"],
        "replacement-v2 attempt amendment time",
    )
    if (
        payload["frozen_input_bundle"] != contract["frozen_input_bundle"]
        or payload["source"] != contract["source"]
        or payload["run_state"] != contract["run_state"]
        or payload["technical_authorization_sha256"]
        != contract["technical_successor"]["authorization_sha256"]
        or payload["intent_sha256"] != contract["technical_successor"]["intent_sha256"]
        or payload["preflight_fingerprint_sha256"] != preflight["preflight_fingerprint_sha256"]
    ):
        raise ControlError("replacement-v2 attempt stable bindings differ")
    for field in (
        "technical_authorization_sha256",
        "intent_sha256",
        "verification_nonce",
        "preflight_fingerprint_sha256",
    ):
        _sha(payload[field], f"replacement-v2 attempt {field}")
    controller = _canonical_dynamic_record(
        payload["controller"],
        path=Path(__file__),
        role="replacement-v2 attempt controller",
    )
    if controller != contract["controller"]:
        raise ControlError("replacement-v2 attempt controller differs from authorization")
    if verify_live_controller and controller != _controller_identity():
        raise ControlError("replacement-v2 attempt controller changed")
    return payload


def _read_attempt_v2(
    namespace: Namespace,
    *,
    authorization: Mapping[str, Any] | None = None,
    authorization_receipt_sha256: str | None = None,
    verify_live: bool = True,
) -> tuple[dict[str, Any], str]:
    if authorization is None or authorization_receipt_sha256 is None:
        authorization, authorization_receipt_sha256 = _read_publication_authorization_v2(
            namespace, verify_live=verify_live
        )
    encoded = _read_bytes(namespace.attempt_v2, "replacement-v2 attempt marker")
    payload = _canonical_attempt_v2(
        _strict_json_object(encoded, "replacement-v2 attempt marker"),
        namespace=namespace,
        authorization=authorization,
        authorization_receipt_sha256=authorization_receipt_sha256,
        verify_live_controller=verify_live,
    )
    if encoded != _canonical_bytes(payload):
        raise ControlError("replacement-v2 attempt marker bytes are not canonical")
    if verify_live:
        frozen_payloads, frozen_records, frozen_root = _read_input_v3(namespace)
        del frozen_payloads
        if (
            payload["frozen_input_bundle"]["files"] != frozen_records
            or payload["frozen_input_bundle"]["records_sha256"] != frozen_root
        ):
            raise ControlError("replacement-v2 attempt input-v3 binding changed")
        current_run = _live_run_state_hashes(_project_root(namespace))
        if payload["run_state"] != current_run:
            raise ControlError("replacement-v2 attempt run-state binding changed")
    return payload, hashlib.sha256(encoded).hexdigest()


_FRESH_CHECK_FIELDS = frozenset(
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
_FRESH_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "failure_phase",
        "requested_python_executable",
        "effective_spawn_executable",
        "executable_override_used",
        "request",
        "request_sha256",
        "argv_sha256",
        "controller_process_id",
        "verifier_process_id",
        "returncode",
        "timeout_milliseconds",
        "timed_out",
        "stdout",
        "stderr",
        "cleanup",
        "payload_sha256",
        "payload_validation_completed",
        "stdout_content_included",
        "stderr_content_included",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_FRESH_REQUEST_FIELDS = frozenset(
    {
        "project_root",
        "successor_directory",
        "parent_directory",
        "artifact_root_sha256",
        "manifest_sha256",
        "authorization_sha256",
        "intent_sha256",
        "nonce",
        "chain_depth",
        "python_executable",
    }
)
_STREAM_DIAGNOSTIC_FIELDS = frozenset(
    {
        "capture_started",
        "limit_bytes",
        "captured_size_bytes",
        "captured_sha256",
        "overflow",
        "eof_observed",
        "read_error",
        "reader_joined",
        "pipe_closed",
    }
)
_CLEANUP_DIAGNOSTIC_FIELDS = frozenset(
    {
        "terminate_attempted",
        "terminate_succeeded",
        "kill_attempted",
        "kill_succeeded",
        "child_reaped",
        "returncode_observed",
        "stdout_reader_joined",
        "stderr_reader_joined",
        "stdout_pipe_closed",
        "stderr_pipe_closed",
        "descendant_quiescence_proven",
        "descendants_reaped",
        "containment",
        "complete",
        "error_codes",
    }
)
_DIAGNOSTIC_PHASES = frozenset(
    {
        "not_started",
        "spawn",
        "pipe_read",
        "wait",
        "cleanup",
        "returncode",
        "payload_parse",
        "payload_validation",
        "completed",
    }
)
_CLEANUP_ERROR_CODES = frozenset(
    {
        "terminate_failed",
        "kill_failed",
        "reap_failed",
        "stdout_reader_not_joined",
        "stderr_reader_not_joined",
        "stdout_pipe_not_closed",
        "stderr_pipe_not_closed",
        "process_tree_probe_failed",
        "descendant_reap_failed",
    }
)


@dataclass(frozen=True, slots=True)
class VerifyRequestV2:
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

    def checked(self) -> VerifyRequestV2:
        if type(self.chain_depth) is not int or self.chain_depth != 4:
            raise FreshVerifierError("fresh verifier chain depth must be exact integer 4")
        executable = Path(self.python_executable).expanduser()
        if not executable.is_absolute():
            raise FreshVerifierError("fresh verifier executable must be absolute")
        try:
            executable = executable.resolve(strict=True)
            expected = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise FreshVerifierError("fresh verifier executable cannot be resolved") from exc
        if os.path.normcase(str(executable)) != os.path.normcase(str(expected)):
            raise FreshVerifierError("fresh verifier must use this controller's Python executable")
        return VerifyRequestV2(
            project_root=_absolute(self.project_root),
            successor_directory=_absolute(self.successor_directory),
            parent_directory=_absolute(self.parent_directory),
            artifact_root_sha256=_sha(self.artifact_root_sha256, "artifact root"),
            manifest_sha256=_sha(self.manifest_sha256, "manifest"),
            authorization_sha256=_sha(self.authorization_sha256, "authorization"),
            intent_sha256=_sha(self.intent_sha256, "intent"),
            nonce=_sha(self.nonce, "verification nonce"),
            chain_depth=4,
            python_executable=str(executable),
        )

    def argv(self, controller_pid: int) -> tuple[str, ...]:
        request = self.checked()
        _positive_int(controller_pid, "fresh verifier controller PID")
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
class VerifyResultV2:
    request: VerifyRequestV2
    argv: tuple[str, ...]
    process_id: int
    payload: dict[str, Any]
    payload_sha256: str
    diagnostic: dict[str, Any]


def _fresh_request_record(request: VerifyRequestV2) -> dict[str, Any]:
    request = request.checked()
    return {
        "project_root": str(request.project_root),
        "successor_directory": str(request.successor_directory),
        "parent_directory": str(request.parent_directory),
        "artifact_root_sha256": request.artifact_root_sha256,
        "manifest_sha256": request.manifest_sha256,
        "authorization_sha256": request.authorization_sha256,
        "intent_sha256": request.intent_sha256,
        "nonce": request.nonce,
        "chain_depth": request.chain_depth,
        "python_executable": request.python_executable,
    }


def _canonical_fresh_request_record(value: object) -> dict[str, Any]:
    record = _exact_dict(
        value,
        _FRESH_REQUEST_FIELDS,
        "fresh verifier request record",
    )
    for field in ("project_root", "successor_directory", "parent_directory"):
        _exact_path_text(
            record[field],
            _absolute(record[field]),
            f"fresh verifier request {field}",
        )
    for field in (
        "artifact_root_sha256",
        "manifest_sha256",
        "authorization_sha256",
        "intent_sha256",
        "nonce",
    ):
        _sha(record[field], f"fresh verifier request {field}")
    if type(record["chain_depth"]) is not int or record["chain_depth"] != 4:
        raise ControlError("fresh verifier request chain depth differs")
    try:
        executable = Path(record["python_executable"]).resolve(strict=True)
        expected = Path(sys.executable).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ControlError("fresh verifier request executable is unavailable") from exc
    if (
        type(record["python_executable"]) is not str
        or record["python_executable"] != str(expected)
        or not _same(executable, expected)
    ):
        raise ControlError("fresh verifier request executable differs")
    _canonical_bytes(record)
    return record


def _fresh_verifier_spawn_executable(python_executable: str) -> str | None:
    """Use the trusted base EXE while retaining venv Python as argv[0] on Windows."""

    if os.name != "nt":
        return None
    raw_executable = getattr(sys, "_base_executable", None)
    raw_base_prefix = sys.base_prefix
    if (
        type(raw_executable) is not str
        or not raw_executable
        or type(raw_base_prefix) is not str
        or not raw_base_prefix
    ):
        raise FreshVerifierError("Windows base Python identity is unavailable")
    candidate = Path(raw_executable).expanduser()
    base_candidate = Path(raw_base_prefix).expanduser()
    if not candidate.is_absolute() or not base_candidate.is_absolute():
        raise FreshVerifierError("Windows base Python identity must be absolute")
    try:
        metadata = candidate.lstat()
        base = _real_directory(base_candidate, "Windows base Python prefix")
        parent = _real_directory(candidate.parent, "Windows base Python parent")
        resolved = candidate.resolve(strict=True)
        expected = (base / Path(python_executable).name).resolve(strict=True)
    except (OSError, ControlError, ValueError) as exc:
        raise FreshVerifierError("Windows base Python executable is untrusted") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & _REPARSE
        or not _same(parent, base)
        or not _same(resolved, expected)
        or resolved.suffix.casefold() != ".exe"
    ):
        raise FreshVerifierError("Windows base Python executable identity differs")
    return str(resolved)


@dataclass(slots=True)
class _BoundedPipeReader:
    stream: Any
    role: str
    limit: int
    payload: bytearray = dataclass_field(default_factory=bytearray)
    error: bool = False
    overflow: bool = False
    eof_observed: bool = False
    reader_joined: bool = False
    pipe_closed: bool = False
    done: threading.Event = dataclass_field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._read,
            name=f"aanca-v2-{self.role}-reader",
            daemon=True,
        )
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                chunk = self.stream.read(_PIPE_CHUNK_BYTES)
                if type(chunk) is not bytes:
                    raise TypeError("bounded verifier pipe returned non-bytes data")
                if not chunk:
                    self.eof_observed = True
                    break
                available = max(0, self.limit - len(self.payload))
                self.payload.extend(chunk[:available])
                if len(chunk) > available:
                    self.overflow = True
                    break
        except BaseException:
            self.error = True
        finally:
            self.done.set()

    def record(self) -> dict[str, Any]:
        return {
            "capture_started": self.thread is not None,
            "limit_bytes": self.limit,
            "captured_size_bytes": len(self.payload),
            "captured_sha256": hashlib.sha256(bytes(self.payload)).hexdigest(),
            "overflow": self.overflow,
            "eof_observed": self.eof_observed,
            "read_error": self.error,
            "reader_joined": self.reader_joined,
            "pipe_closed": self.pipe_closed,
        }


class _WindowsJobApi(Protocol):
    def create_job(self) -> int: ...

    def set_kill_on_close(self, job_handle: int) -> None: ...

    def assign_process(self, job_handle: int, process_handle: int) -> None: ...

    def enumerate_threads(self, owner_pid: int) -> tuple[int, ...]: ...

    def open_thread(self, thread_id: int) -> int: ...

    def resume_thread(self, thread_handle: int) -> int: ...

    def terminate_job(self, job_handle: int) -> None: ...

    def active_process_ids(self, job_handle: int) -> tuple[int, ...]: ...

    def close_handle(self, handle: int) -> None: ...


class _CtypesWindowsJobApi:
    """Minimal Win32 API surface used by the suspended-child containment gate."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        self.ExtendedLimitInformation = ExtendedLimitInformation
        self.ThreadEntry32 = ThreadEntry32

    def _raise_last_error(self, role: str) -> None:
        code = self.ctypes.get_last_error()
        raise OSError(code, role)

    def create_job(self) -> int:
        function = self.kernel32.CreateJobObjectW
        function.argtypes = [self.ctypes.c_void_p, self.wintypes.LPCWSTR]
        function.restype = self.wintypes.HANDLE
        handle = function(None, None)
        if not handle:
            self._raise_last_error("CreateJobObjectW failed")
        return int(handle)

    def set_kill_on_close(self, job_handle: int) -> None:
        information = self.ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        function = self.kernel32.SetInformationJobObject
        function.argtypes = [
            self.wintypes.HANDLE,
            self.ctypes.c_int,
            self.ctypes.c_void_p,
            self.wintypes.DWORD,
        ]
        function.restype = self.wintypes.BOOL
        if not function(
            job_handle,
            9,
            self.ctypes.byref(information),
            self.ctypes.sizeof(information),
        ):
            self._raise_last_error("SetInformationJobObject failed")

    def assign_process(self, job_handle: int, process_handle: int) -> None:
        function = self.kernel32.AssignProcessToJobObject
        function.argtypes = [self.wintypes.HANDLE, self.wintypes.HANDLE]
        function.restype = self.wintypes.BOOL
        if not function(job_handle, process_handle):
            self._raise_last_error("AssignProcessToJobObject failed")

    def enumerate_threads(self, owner_pid: int) -> tuple[int, ...]:
        snapshot_function = self.kernel32.CreateToolhelp32Snapshot
        snapshot_function.argtypes = [self.wintypes.DWORD, self.wintypes.DWORD]
        snapshot_function.restype = self.wintypes.HANDLE
        snapshot = snapshot_function(0x00000004, 0)
        invalid_handle = self.ctypes.c_void_p(-1).value
        if not snapshot or int(snapshot) == invalid_handle:
            self._raise_last_error("CreateToolhelp32Snapshot failed")
        entry = self.ThreadEntry32()
        entry.dwSize = self.ctypes.sizeof(entry)
        first = self.kernel32.Thread32First
        first.argtypes = [
            self.wintypes.HANDLE,
            self.ctypes.POINTER(self.ThreadEntry32),
        ]
        first.restype = self.wintypes.BOOL
        following = self.kernel32.Thread32Next
        following.argtypes = [
            self.wintypes.HANDLE,
            self.ctypes.POINTER(self.ThreadEntry32),
        ]
        following.restype = self.wintypes.BOOL
        thread_ids: list[int] = []
        error: BaseException | None = None
        try:
            found = bool(first(snapshot, self.ctypes.byref(entry)))
            if not found:
                self._raise_last_error("Thread32First failed")
            while found:
                if int(entry.th32OwnerProcessID) == owner_pid:
                    thread_ids.append(int(entry.th32ThreadID))
                found = bool(following(snapshot, self.ctypes.byref(entry)))
            last_error = self.ctypes.get_last_error()
            if last_error not in (0, 18):
                self._raise_last_error("Thread32Next failed")
        except BaseException as exc:
            error = exc
        try:
            self.close_handle(int(snapshot))
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error
        return tuple(sorted(thread_ids))

    def open_thread(self, thread_id: int) -> int:
        function = self.kernel32.OpenThread
        function.argtypes = [
            self.wintypes.DWORD,
            self.wintypes.BOOL,
            self.wintypes.DWORD,
        ]
        function.restype = self.wintypes.HANDLE
        handle = function(0x0002, False, thread_id)
        if not handle:
            self._raise_last_error("OpenThread failed")
        return int(handle)

    def resume_thread(self, thread_handle: int) -> int:
        function = self.kernel32.ResumeThread
        function.argtypes = [self.wintypes.HANDLE]
        function.restype = self.wintypes.DWORD
        previous = int(function(thread_handle))
        if previous == 0xFFFFFFFF:
            self._raise_last_error("ResumeThread failed")
        return previous

    def terminate_job(self, job_handle: int) -> None:
        function = self.kernel32.TerminateJobObject
        function.argtypes = [self.wintypes.HANDLE, self.wintypes.UINT]
        function.restype = self.wintypes.BOOL
        if not function(job_handle, 1):
            self._raise_last_error("TerminateJobObject failed")

    def active_process_ids(self, job_handle: int) -> tuple[int, ...]:
        function = self.kernel32.QueryInformationJobObject
        function.argtypes = [
            self.wintypes.HANDLE,
            self.ctypes.c_int,
            self.ctypes.c_void_p,
            self.wintypes.DWORD,
            self.ctypes.POINTER(self.wintypes.DWORD),
        ]
        function.restype = self.wintypes.BOOL
        size = 4096
        while size <= 65536:
            buffer = self.ctypes.create_string_buffer(size)
            returned = self.wintypes.DWORD()
            if function(
                job_handle,
                3,
                buffer,
                size,
                self.ctypes.byref(returned),
            ):
                assigned = self.wintypes.DWORD.from_buffer(buffer, 0).value
                listed = self.wintypes.DWORD.from_buffer(
                    buffer,
                    self.ctypes.sizeof(self.wintypes.DWORD),
                ).value
                pointer_offset = self.ctypes.sizeof(self.wintypes.DWORD) * 2
                pointer_alignment = self.ctypes.sizeof(self.ctypes.c_size_t)
                pointer_offset = (
                    (pointer_offset + pointer_alignment - 1)
                    // pointer_alignment
                    * pointer_alignment
                )
                if listed != assigned or pointer_offset + listed * pointer_alignment > size:
                    raise OSError("QueryInformationJobObject returned invalid counts")
                array_type = self.ctypes.c_size_t * listed
                values = array_type.from_buffer(buffer, pointer_offset)
                result = tuple(sorted(int(value) for value in values))
                if len(result) != len(set(result)) or any(
                    type(value) is not int or value <= 0 for value in result
                ):
                    raise OSError("QueryInformationJobObject returned invalid process IDs")
                return result
            if self.ctypes.get_last_error() != 234:
                self._raise_last_error("QueryInformationJobObject failed")
            size *= 2
        raise OSError("QueryInformationJobObject process list exceeded bound")

    def close_handle(self, handle: int) -> None:
        function = self.kernel32.CloseHandle
        function.argtypes = [self.wintypes.HANDLE]
        function.restype = self.wintypes.BOOL
        if not function(handle):
            self._raise_last_error("CloseHandle failed")


_JOB_CONTAINMENT_FIELDS = frozenset(
    {
        "required",
        "created",
        "kill_on_close_configured",
        "child_created_suspended",
        "process_handle_proven",
        "assigned",
        "assignment_membership_proven",
        "thread_enumeration_succeeded",
        "owned_thread_count",
        "thread_opened",
        "thread_resumed",
        "terminate_attempted",
        "terminate_succeeded",
        "job_empty_proven",
        "close_attempted",
        "close_succeeded",
        "complete",
        "error_codes",
    }
)
_JOB_ERROR_CODES = frozenset(
    {
        "job_create_failed",
        "job_limit_failed",
        "process_handle_unavailable",
        "job_assign_failed",
        "thread_snapshot_failed",
        "initial_thread_count_invalid",
        "thread_open_failed",
        "thread_resume_failed",
        "thread_handle_close_failed",
        "job_terminate_failed",
        "job_query_failed",
        "job_not_empty",
        "job_close_failed",
    }
)


@dataclass(slots=True)
class _WindowsJobContainment:
    required: bool
    api: _WindowsJobApi | None = None
    handle: int | None = None
    created: bool = False
    kill_on_close_configured: bool = False
    child_created_suspended: bool = False
    process_handle_proven: bool = False
    assigned: bool = False
    assignment_membership_proven: bool = False
    thread_enumeration_succeeded: bool = False
    owned_thread_count: int = 0
    thread_opened: bool = False
    thread_resumed: bool = False
    terminate_attempted: bool = False
    terminate_succeeded: bool = False
    job_empty_proven: bool = False
    close_attempted: bool = False
    close_succeeded: bool = False
    error_codes: list[str] = dataclass_field(default_factory=list)

    @property
    def ready_before_spawn(self) -> bool:
        return not self.required or (
            self.created
            and self.kill_on_close_configured
            and type(self.handle) is int
            and self.handle > 0
            and not self.error_codes
        )

    def mark_child_created(self, *, suspended: bool) -> None:
        if type(suspended) is not bool:
            raise FreshVerifierError("child suspension evidence is not exact")
        self.child_created_suspended = suspended

    def _fail(self, code: str, message: str, cause: BaseException | None = None) -> None:
        self.error_codes.append(code)
        error = FreshVerifierError(message)
        if cause is None:
            raise error
        raise error from cause

    def assign_and_resume(self, process: Any) -> None:
        if not self.required:
            return
        if not self.ready_before_spawn or not self.child_created_suspended:
            self._fail(
                "job_assign_failed",
                "Windows verifier job was not ready for its suspended child",
            )
        raw_handle = getattr(process, "_handle", None)
        try:
            process_handle = 0 if raw_handle is None else int(raw_handle)
        except (TypeError, ValueError, OverflowError):
            process_handle = 0
        if isinstance(raw_handle, bool) or process_handle <= 0:
            self._fail(
                "process_handle_unavailable",
                "Windows verifier process handle is unavailable",
            )
        self.process_handle_proven = True
        assert self.api is not None and self.handle is not None
        try:
            self.api.assign_process(self.handle, process_handle)
        except BaseException as exc:
            self._fail(
                "job_assign_failed",
                "Windows verifier child could not be assigned to its Job Object",
                exc,
            )
        self.assigned = True
        try:
            active = self.api.active_process_ids(self.handle)
        except BaseException as exc:
            self._fail(
                "job_query_failed",
                "Windows verifier Job membership query failed",
                exc,
            )
        if active != (int(process.pid),):
            self._fail(
                "job_query_failed",
                "Windows verifier Job does not contain exactly its suspended child",
            )
        self.assignment_membership_proven = True
        try:
            thread_ids = self.api.enumerate_threads(int(process.pid))
        except BaseException as exc:
            self._fail(
                "thread_snapshot_failed",
                "Windows verifier initial-thread snapshot failed",
                exc,
            )
        self.thread_enumeration_succeeded = True
        self.owned_thread_count = len(thread_ids)
        if len(thread_ids) != 1:
            self._fail(
                "initial_thread_count_invalid",
                "Windows verifier suspended child does not have exactly one thread",
            )
        try:
            thread_handle = self.api.open_thread(thread_ids[0])
        except BaseException as exc:
            self._fail(
                "thread_open_failed",
                "Windows verifier initial thread could not be opened",
                exc,
            )
        self.thread_opened = True
        resume_error: BaseException | None = None
        try:
            previous = self.api.resume_thread(thread_handle)
            if previous != 1:
                raise OSError("unexpected initial suspend count")
            self.thread_resumed = True
        except BaseException as exc:
            resume_error = exc
            self.error_codes.append("thread_resume_failed")
        try:
            self.api.close_handle(thread_handle)
        except BaseException as exc:
            self.error_codes.append("thread_handle_close_failed")
            resume_error = resume_error or exc
        if resume_error is not None:
            raise FreshVerifierError(
                "Windows verifier initial thread could not be resumed exactly once"
            ) from resume_error

    def terminate(self) -> None:
        if not self.required or not self.assigned or self.handle is None:
            return
        self.terminate_attempted = True
        assert self.api is not None
        try:
            self.api.terminate_job(self.handle)
            self.terminate_succeeded = True
        except BaseException as exc:
            self.error_codes.append("job_terminate_failed")
            raise FreshVerifierError("Windows verifier Job termination failed") from exc

    def prove_empty(self, *, timeout_seconds: float = _CLEANUP_GRACE_SECONDS) -> None:
        if not self.required or not self.assigned or self.handle is None:
            return
        assert self.api is not None
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                active = self.api.active_process_ids(self.handle)
            except BaseException as exc:
                self.error_codes.append("job_query_failed")
                raise FreshVerifierError("Windows verifier Job process query failed") from exc
            if not active:
                self.job_empty_proven = True
                return
            if time.monotonic() >= deadline:
                self.error_codes.append("job_not_empty")
                raise FreshVerifierError(
                    "Windows verifier Job did not become empty within its bound"
                )
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        if not self.required or self.handle is None:
            return
        self.close_attempted = True
        handle = self.handle
        assert self.api is not None
        try:
            self.api.close_handle(handle)
            self.close_succeeded = True
            self.handle = None
        except BaseException as exc:
            self.error_codes.append("job_close_failed")
            raise FreshVerifierError("Windows verifier Job handle close failed") from exc

    def diagnostic(self) -> dict[str, Any]:
        codes = sorted(set(self.error_codes))
        if not self.required:
            complete = not codes
        elif not self.child_created_suspended:
            complete = (
                self.created
                and self.kill_on_close_configured
                and self.close_attempted
                and self.close_succeeded
                and not codes
            )
        else:
            complete = (
                self.created
                and self.kill_on_close_configured
                and self.process_handle_proven
                and self.assigned
                and self.assignment_membership_proven
                and self.thread_enumeration_succeeded
                and self.owned_thread_count == 1
                and self.thread_opened
                and self.thread_resumed
                and self.job_empty_proven
                and (not self.terminate_attempted or self.terminate_succeeded)
                and self.close_attempted
                and self.close_succeeded
                and not codes
            )
        return {
            "required": self.required,
            "created": self.created,
            "kill_on_close_configured": self.kill_on_close_configured,
            "child_created_suspended": self.child_created_suspended,
            "process_handle_proven": self.process_handle_proven,
            "assigned": self.assigned,
            "assignment_membership_proven": self.assignment_membership_proven,
            "thread_enumeration_succeeded": self.thread_enumeration_succeeded,
            "owned_thread_count": self.owned_thread_count,
            "thread_opened": self.thread_opened,
            "thread_resumed": self.thread_resumed,
            "terminate_attempted": self.terminate_attempted,
            "terminate_succeeded": self.terminate_succeeded,
            "job_empty_proven": self.job_empty_proven,
            "close_attempted": self.close_attempted,
            "close_succeeded": self.close_succeeded,
            "complete": complete,
            "error_codes": codes,
        }


def _create_windows_job_containment(
    *,
    api: _WindowsJobApi | None = None,
) -> _WindowsJobContainment:
    if os.name != "nt":
        return _WindowsJobContainment(required=False)
    selected = api or _CtypesWindowsJobApi()
    containment = _WindowsJobContainment(required=True, api=selected)
    try:
        containment.handle = selected.create_job()
        if type(containment.handle) is not int or containment.handle <= 0:
            raise OSError("invalid Windows Job handle")
        containment.created = True
    except BaseException:
        containment.error_codes.append("job_create_failed")
        return containment
    try:
        selected.set_kill_on_close(containment.handle)
        containment.kill_on_close_configured = True
    except BaseException:
        containment.error_codes.append("job_limit_failed")
        with suppress(FreshVerifierError):
            containment.close()
    return containment


def _preflight_windows_job_containment(
    *,
    python_executable: str,
    containment_factory: Callable[[], _WindowsJobContainment],
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    containment = containment_factory()
    if type(containment) is not _WindowsJobContainment:
        raise FreshVerifierError("Windows verifier containment factory is untyped")
    if not containment.ready_before_spawn:
        raise FreshVerifierError("Windows verifier Job capability gate failed")
    if not containment.required:
        diagnostic = containment.diagnostic()
        if diagnostic["complete"] is not True:
            raise FreshVerifierError("non-Windows containment gate is inconsistent")
        return diagnostic
    override = _fresh_verifier_spawn_executable(python_executable)
    arguments: dict[str, Any] = {
        "cwd": str(_absolute(Path.cwd())),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "shell": False,
        "text": False,
        "close_fds": True,
        "creationflags": getattr(subprocess, "CREATE_SUSPENDED", 0x00000004),
        "env": {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
            }
        }
        | {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    }
    if override is not None:
        arguments["executable"] = override
    process: Any | None = None
    tree: _ProcessTreeTracker | None = None
    try:
        process = popen_factory(
            [python_executable, "-I", "-B", "-c", "pass"],
            **arguments,
        )
        if type(getattr(process, "pid", None)) is not int or process.pid <= 0:
            raise FreshVerifierError("Windows verifier canary process identity is unavailable")
        containment.mark_child_created(suspended=True)
        containment.assign_and_resume(process)
        tree = _ProcessTreeTracker(process.pid)
        tree.refresh()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_CLEANUP_GRACE_SECONDS)
        cleanup = _cleanup_verifier_process(
            process,
            (),
            force=getattr(process, "returncode", None) is None,
            tree=tree,
            containment=containment,
        )
    except BaseException as exc:
        if process is not None:
            cleanup = _cleanup_verifier_process(
                process,
                (),
                force=True,
                tree=tree,
                containment=containment,
            )
        else:
            cleanup = _cleanup_without_spawned_process(containment)
        raise FreshVerifierError(
            "Windows verifier full disposable capability canary failed"
        ) from exc
    diagnostic = cleanup["containment"]
    if (
        cleanup["complete"] is not True
        or diagnostic["complete"] is not True
        or getattr(process, "returncode", None) != 0
    ):
        raise FreshVerifierError("Windows verifier full disposable capability cleanup failed")
    return diagnostic


def _close_join_reader(reader: _BoundedPipeReader, deadline: float) -> None:
    thread = reader.thread
    if thread is not None:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        reader.reader_joined = not thread.is_alive()
    try:
        reader.stream.close()
        reader.pipe_closed = True
    except (OSError, ValueError):
        reader.pipe_closed = False


@dataclass(slots=True)
class _ProcessTreeTracker:
    """Track and boundedly reap every verifier descendant, not only Popen's child."""

    root_pid: int
    known: dict[int, psutil.Process] = dataclass_field(default_factory=dict)
    probe_failed: bool = False
    root_suspended: bool = False
    root_identity_proven: bool = False

    def refresh(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
            root.create_time()
            self.root_identity_proven = True
            descendants = root.children(recursive=True)
        except psutil.NoSuchProcess:
            if not self.root_identity_proven:
                self.probe_failed = True
            return
        except (psutil.AccessDenied, OSError):
            self.probe_failed = True
            return
        for descendant in descendants:
            if descendant.pid != self.root_pid:
                self.known.setdefault(descendant.pid, descendant)

    def quiesce(self) -> None:
        """Suspend the tree before its final snapshot so no descendant can escape."""

        try:
            root = psutil.Process(self.root_pid)
            if root.is_running():
                root.suspend()
                self.root_suspended = True
        except psutil.NoSuchProcess:
            return
        except (psutil.AccessDenied, OSError):
            self.probe_failed = True
            return
        for _ in range(4):
            previous = set(self.known)
            self.refresh()
            for descendant in tuple(self.known.values()):
                try:
                    if descendant.is_running():
                        descendant.suspend()
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, OSError):
                    self.probe_failed = True
            if set(self.known) == previous:
                return
        self.refresh()
        self.probe_failed = True

    def reap_descendants(self) -> bool:
        self.refresh()
        descendants = tuple(self.known.values())
        for descendant in descendants:
            try:
                if descendant.is_running():
                    descendant.terminate()
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                self.probe_failed = True
        _gone, alive = psutil.wait_procs(
            list(descendants),
            timeout=_CLEANUP_GRACE_SECONDS,
        )
        for descendant in alive:
            try:
                descendant.kill()
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                self.probe_failed = True
        if alive:
            _gone, alive = psutil.wait_procs(
                alive,
                timeout=_CLEANUP_GRACE_SECONDS,
            )
        still_alive: list[psutil.Process] = []
        for descendant in alive:
            try:
                if descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE:
                    still_alive.append(descendant)
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                self.probe_failed = True
                still_alive.append(descendant)
        return not still_alive


def _cleanup_verifier_process(
    process: Any,
    readers: Sequence[_BoundedPipeReader],
    *,
    force: bool,
    tree: _ProcessTreeTracker | None = None,
    containment: _WindowsJobContainment | None = None,
) -> dict[str, Any]:
    terminate_attempted = False
    terminate_succeeded = False
    kill_attempted = False
    kill_succeeded = False
    error_codes: list[str] = []
    if containment is not None and containment.assigned:
        try:
            containment.terminate()
        except FreshVerifierError:
            error_codes.append("job_terminate_failed")
    if tree is not None:
        tree.refresh()
        if force and getattr(process, "returncode", None) is None:
            tree.quiesce()
    if force and getattr(process, "returncode", None) is None:
        terminate_attempted = True
        try:
            process.terminate()
            terminate_succeeded = True
        except ProcessLookupError:
            terminate_succeeded = True
        except BaseException:
            error_codes.append("terminate_failed")
        try:
            process.wait(timeout=_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        except BaseException:
            error_codes.append("reap_failed")
    if getattr(process, "returncode", None) is None:
        kill_attempted = True
        try:
            process.kill()
            kill_succeeded = True
        except ProcessLookupError:
            kill_succeeded = True
        except BaseException:
            error_codes.append("kill_failed")
        try:
            process.wait(timeout=_CLEANUP_GRACE_SECONDS)
        except BaseException:
            error_codes.append("reap_failed")
    child_reaped = getattr(process, "returncode", None) is not None
    if not child_reaped:
        error_codes.append("reap_failed")
    if containment is not None:
        try:
            containment.prove_empty()
        except FreshVerifierError:
            error_codes.extend(
                code
                for code in containment.error_codes
                if code in {"job_query_failed", "job_not_empty"}
            )
        try:
            containment.close()
        except FreshVerifierError:
            error_codes.append("job_close_failed")
    deadline = time.monotonic() + _CLEANUP_GRACE_SECONDS
    for reader in readers:
        _close_join_reader(reader, deadline)
        if not reader.reader_joined:
            error_codes.append(f"{reader.role}_reader_not_joined")
        if not reader.pipe_closed:
            error_codes.append(f"{reader.role}_pipe_not_closed")
    by_role = {reader.role: reader for reader in readers}
    stdout = by_role.get("stdout")
    stderr = by_role.get("stderr")
    stdout_joined = stdout.reader_joined if stdout is not None else True
    stderr_joined = stderr.reader_joined if stderr is not None else True
    stdout_closed = stdout.pipe_closed if stdout is not None else True
    stderr_closed = stderr.pipe_closed if stderr is not None else True
    containment_record = (
        _empty_job_containment_diagnostic() if containment is None else containment.diagnostic()
    )
    if containment is not None and containment_record["required"]:
        tracked_descendants_reaped = True if tree is None else tree.reap_descendants()
        descendants_reaped = containment_record["complete"] and tracked_descendants_reaped
        tree_proven = containment_record["complete"]
    else:
        descendants_reaped = True if tree is None else tree.reap_descendants()
        tree_proven = tree is not None and tree.root_identity_proven and not tree.probe_failed
    if not tree_proven:
        error_codes.append("process_tree_probe_failed")
    if not descendants_reaped:
        error_codes.append("descendant_reap_failed")
    codes = sorted(set(error_codes))
    return {
        "terminate_attempted": terminate_attempted,
        "terminate_succeeded": terminate_succeeded,
        "kill_attempted": kill_attempted,
        "kill_succeeded": kill_succeeded,
        "child_reaped": child_reaped,
        "returncode_observed": type(getattr(process, "returncode", None)) is int,
        "stdout_reader_joined": stdout_joined,
        "stderr_reader_joined": stderr_joined,
        "stdout_pipe_closed": stdout_closed,
        "stderr_pipe_closed": stderr_closed,
        "descendant_quiescence_proven": tree_proven,
        "descendants_reaped": descendants_reaped,
        "containment": containment_record,
        "complete": (
            child_reaped
            and stdout_joined
            and stderr_joined
            and stdout_closed
            and stderr_closed
            and tree_proven
            and descendants_reaped
            and containment_record["complete"]
            and not codes
        ),
        "error_codes": codes,
    }


def _empty_stream_diagnostic(limit: int) -> dict[str, Any]:
    return {
        "capture_started": False,
        "limit_bytes": limit,
        "captured_size_bytes": 0,
        "captured_sha256": hashlib.sha256(b"").hexdigest(),
        "overflow": False,
        "eof_observed": False,
        "read_error": False,
        "reader_joined": True,
        "pipe_closed": True,
    }


def _empty_job_containment_diagnostic() -> dict[str, Any]:
    return {
        "required": os.name == "nt",
        "created": False,
        "kill_on_close_configured": False,
        "child_created_suspended": False,
        "process_handle_proven": False,
        "assigned": False,
        "assignment_membership_proven": False,
        "thread_enumeration_succeeded": False,
        "owned_thread_count": 0,
        "thread_opened": False,
        "thread_resumed": False,
        "terminate_attempted": False,
        "terminate_succeeded": False,
        "job_empty_proven": False,
        "close_attempted": False,
        "close_succeeded": False,
        "complete": True,
        "error_codes": [],
    }


def _empty_cleanup_diagnostic() -> dict[str, Any]:
    return {
        "terminate_attempted": False,
        "terminate_succeeded": False,
        "kill_attempted": False,
        "kill_succeeded": False,
        "child_reaped": True,
        "returncode_observed": False,
        "stdout_reader_joined": True,
        "stderr_reader_joined": True,
        "stdout_pipe_closed": True,
        "stderr_pipe_closed": True,
        "descendant_quiescence_proven": True,
        "descendants_reaped": True,
        "containment": _empty_job_containment_diagnostic(),
        "complete": True,
        "error_codes": [],
    }


def _cleanup_without_spawned_process(
    containment: _WindowsJobContainment,
) -> dict[str, Any]:
    with suppress(FreshVerifierError):
        containment.close()
    record = _empty_cleanup_diagnostic()
    containment_record = containment.diagnostic()
    record["containment"] = containment_record
    record["descendant_quiescence_proven"] = containment_record["complete"]
    record["descendants_reaped"] = containment_record["complete"]
    record["complete"] = containment_record["complete"]
    record["error_codes"] = list(containment_record["error_codes"])
    return record


def _canonical_fresh_diagnostic(value: object) -> dict[str, Any]:
    payload = _exact_dict(value, _FRESH_DIAGNOSTIC_FIELDS, "fresh verifier diagnostic")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != FRESH_DIAGNOSTIC_SCHEMA_VERSION
        or payload["policy"] != FRESH_DIAGNOSTIC_POLICY
        or payload["status"] not in {"not_invoked", "spawn_failed", "failed", "passed"}
        or payload["failure_phase"] not in _DIAGNOSTIC_PHASES
        or type(payload["requested_python_executable"]) is not str
        or type(payload["effective_spawn_executable"]) is not str
        or type(payload["executable_override_used"]) is not bool
        or type(payload["controller_process_id"]) is not int
        or payload["controller_process_id"] <= 0
        or type(payload["timeout_milliseconds"]) is not int
        or payload["timeout_milliseconds"] <= 0
        or type(payload["timed_out"]) is not bool
        or payload["stdout_content_included"] is not False
        or payload["stderr_content_included"] is not False
        or payload["outcome_value_interpretation_performed"] is not False
        or payload["scientific_execution_performed"] is not False
        or payload["publication_performed"] is not False
    ):
        raise ControlError("fresh verifier diagnostic fixed policy differs")
    try:
        requested = Path(payload["requested_python_executable"]).resolve(strict=True)
        expected_requested = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise ControlError("fresh verifier requested executable is unavailable") from exc
    expected_override = _fresh_verifier_spawn_executable(str(expected_requested))
    expected_effective = str(expected_requested) if expected_override is None else expected_override
    request_record = _canonical_fresh_request_record(payload["request"])
    if (
        not _same(requested, expected_requested)
        or payload["requested_python_executable"] != str(expected_requested)
        or payload["effective_spawn_executable"] != expected_effective
        or payload["executable_override_used"] is not (expected_override is not None)
        or request_record["python_executable"] != payload["requested_python_executable"]
        or payload["request_sha256"] != _compact_sha256(request_record)
    ):
        raise ControlError("fresh verifier executable identity differs")
    _sha(payload["request_sha256"], "fresh verifier request record")
    _sha(payload["argv_sha256"], "fresh verifier argv")
    if payload["status"] == "not_invoked":
        expected_argv_sha256 = _compact_sha256([])
    else:
        request_object = VerifyRequestV2(
            project_root=Path(request_record["project_root"]),
            successor_directory=Path(request_record["successor_directory"]),
            parent_directory=Path(request_record["parent_directory"]),
            artifact_root_sha256=request_record["artifact_root_sha256"],
            manifest_sha256=request_record["manifest_sha256"],
            authorization_sha256=request_record["authorization_sha256"],
            intent_sha256=request_record["intent_sha256"],
            nonce=request_record["nonce"],
            chain_depth=request_record["chain_depth"],
            python_executable=request_record["python_executable"],
        ).checked()
        expected_argv_sha256 = _compact_sha256(
            list(request_object.argv(payload["controller_process_id"]))
        )
    if payload["argv_sha256"] != expected_argv_sha256:
        raise ControlError("fresh verifier argv does not bind its exact request")
    if payload["verifier_process_id"] is not None:
        _positive_int(payload["verifier_process_id"], "fresh verifier child PID")
    if payload["returncode"] is not None and type(payload["returncode"]) is not int:
        raise ControlError("fresh verifier returncode must be exact int or null")
    if type(payload["payload_validation_completed"]) is not bool:
        raise ControlError("fresh verifier payload validation flag must be boolean")
    if payload["payload_sha256"] is not None:
        _sha(payload["payload_sha256"], "fresh verifier payload")
    for role, expected_limit in (
        ("stdout", _MAX_STDOUT_BYTES),
        ("stderr", _MAX_STDERR_BYTES),
    ):
        stream = _exact_dict(
            payload[role],
            _STREAM_DIAGNOSTIC_FIELDS,
            f"fresh verifier {role} diagnostic",
        )
        if (
            type(stream["capture_started"]) is not bool
            or type(stream["limit_bytes"]) is not int
            or stream["limit_bytes"] != expected_limit
            or type(stream["captured_size_bytes"]) is not int
            or not 0 <= stream["captured_size_bytes"] <= expected_limit
            or any(
                type(stream[field]) is not bool
                for field in (
                    "overflow",
                    "eof_observed",
                    "read_error",
                    "reader_joined",
                    "pipe_closed",
                )
            )
        ):
            raise ControlError(f"fresh verifier {role} diagnostic differs")
        _sha(stream["captured_sha256"], f"fresh verifier {role} capture")
    cleanup = _exact_dict(
        payload["cleanup"],
        _CLEANUP_DIAGNOSTIC_FIELDS,
        "fresh verifier cleanup diagnostic",
    )
    for field in _CLEANUP_DIAGNOSTIC_FIELDS - {"containment", "error_codes"}:
        if type(cleanup[field]) is not bool:
            raise ControlError(f"fresh verifier cleanup {field} must be boolean")
    containment = _exact_dict(
        cleanup["containment"],
        _JOB_CONTAINMENT_FIELDS,
        "fresh verifier Windows Job containment",
    )
    if (
        type(containment["required"]) is not bool
        or containment["required"] is not (os.name == "nt")
        or type(containment["owned_thread_count"]) is not int
        or containment["owned_thread_count"] < 0
        or any(
            type(containment[field]) is not bool
            for field in _JOB_CONTAINMENT_FIELDS - {"owned_thread_count", "error_codes"}
        )
        or type(containment["error_codes"]) is not list
        or containment["error_codes"] != sorted(set(containment["error_codes"]))
        or any(code not in _JOB_ERROR_CODES for code in containment["error_codes"])
        or (containment["terminate_succeeded"] and not containment["terminate_attempted"])
        or (containment["close_succeeded"] and not containment["close_attempted"])
    ):
        raise ControlError("fresh verifier Windows Job containment differs")
    if containment["required"] and containment["child_created_suspended"]:
        expected_containment_complete = (
            containment["created"]
            and containment["kill_on_close_configured"]
            and containment["process_handle_proven"]
            and containment["assigned"]
            and containment["assignment_membership_proven"]
            and containment["thread_enumeration_succeeded"]
            and containment["owned_thread_count"] == 1
            and containment["thread_opened"]
            and containment["thread_resumed"]
            and containment["job_empty_proven"]
            and (not containment["terminate_attempted"] or containment["terminate_succeeded"])
            and containment["close_attempted"]
            and containment["close_succeeded"]
            and not containment["error_codes"]
        )
    elif containment["required"] and (containment["created"] or containment["close_attempted"]):
        expected_containment_complete = (
            containment["created"]
            and containment["kill_on_close_configured"]
            and containment["close_attempted"]
            and containment["close_succeeded"]
            and not containment["error_codes"]
        )
    else:
        expected_containment_complete = not containment["error_codes"]
    if containment["complete"] is not expected_containment_complete:
        raise ControlError("fresh verifier Windows Job completion proof differs")
    if (
        type(cleanup["error_codes"]) is not list
        or cleanup["error_codes"] != sorted(set(cleanup["error_codes"]))
        or any(
            code not in (_CLEANUP_ERROR_CODES | _JOB_ERROR_CODES) for code in cleanup["error_codes"]
        )
        or cleanup["complete"]
        is not (
            cleanup["child_reaped"]
            and cleanup["stdout_reader_joined"]
            and cleanup["stderr_reader_joined"]
            and cleanup["stdout_pipe_closed"]
            and cleanup["stderr_pipe_closed"]
            and cleanup["descendant_quiescence_proven"]
            and cleanup["descendants_reaped"]
            and containment["complete"]
            and not cleanup["error_codes"]
        )
    ):
        raise ControlError("fresh verifier cleanup diagnostic differs")
    if (cleanup["terminate_succeeded"] and not cleanup["terminate_attempted"]) or (
        cleanup["kill_succeeded"] and not cleanup["kill_attempted"]
    ):
        raise ControlError("fresh verifier cleanup operation ordering differs")
    if cleanup["returncode_observed"] is not (type(payload["returncode"]) is int):
        raise ControlError("fresh verifier returncode observation differs")
    stdout = payload["stdout"]
    stderr = payload["stderr"]
    if (
        stdout["reader_joined"] is not cleanup["stdout_reader_joined"]
        or stdout["pipe_closed"] is not cleanup["stdout_pipe_closed"]
        or stderr["reader_joined"] is not cleanup["stderr_reader_joined"]
        or stderr["pipe_closed"] is not cleanup["stderr_pipe_closed"]
        or (
            payload["payload_sha256"] is not None
            and payload["payload_sha256"] != stdout["captured_sha256"]
        )
    ):
        raise ControlError("fresh verifier streams differ from cleanup or payload hash")
    status = payload["status"]
    empty_stdout = _empty_stream_diagnostic(_MAX_STDOUT_BYTES)
    empty_stderr = _empty_stream_diagnostic(_MAX_STDERR_BYTES)
    if status == "not_invoked":
        if (
            payload["failure_phase"] != "not_started"
            or payload["verifier_process_id"] is not None
            or payload["returncode"] is not None
            or payload["timed_out"] is not False
            or payload["payload_validation_completed"] is not False
            or payload["payload_sha256"] is not None
            or stdout != empty_stdout
            or stderr != empty_stderr
            or cleanup != _empty_cleanup_diagnostic()
        ):
            raise ControlError("not-invoked fresh verifier diagnostic is inconsistent")
    elif status == "spawn_failed":
        if (
            payload["failure_phase"] != "spawn"
            or payload["verifier_process_id"] is not None
            or payload["returncode"] is not None
            or payload["timed_out"] is not False
            or payload["payload_validation_completed"] is not False
            or payload["payload_sha256"] is not None
            or stdout != empty_stdout
            or stderr != empty_stderr
            or cleanup["terminate_attempted"]
            or cleanup["kill_attempted"]
            or cleanup["child_reaped"] is not True
            or cleanup["stdout_reader_joined"] is not True
            or cleanup["stderr_reader_joined"] is not True
            or cleanup["stdout_pipe_closed"] is not True
            or cleanup["stderr_pipe_closed"] is not True
        ):
            raise ControlError("spawn-failed fresh verifier diagnostic is inconsistent")
    elif status == "failed" and (
        payload["failure_phase"]
        not in {
            "pipe_read",
            "wait",
            "cleanup",
            "returncode",
            "payload_parse",
            "payload_validation",
        }
        or payload["verifier_process_id"] is None
        or payload["verifier_process_id"] == payload["controller_process_id"]
        or payload["payload_validation_completed"] is not False
        or (payload["timed_out"] and payload["failure_phase"] != "wait")
        or (
            payload["payload_sha256"] is not None
            and payload["failure_phase"]
            not in {"payload_parse", "payload_validation", "returncode"}
        )
        or (
            payload["failure_phase"] in {"returncode", "payload_parse", "payload_validation"}
            and payload["payload_sha256"] is None
        )
        or (
            payload["failure_phase"] == "returncode"
            and (
                payload["returncode"] is None
                or (payload["returncode"] == 0 and stderr["captured_size_bytes"] == 0)
            )
        )
        or (
            payload["failure_phase"] in {"payload_parse", "payload_validation"}
            and (
                payload["returncode"] != 0
                or stderr["captured_size_bytes"] != 0
                or cleanup["complete"] is not True
            )
        )
        or (payload["failure_phase"] == "cleanup" and cleanup["complete"] is not False)
    ):
        raise ControlError("failed fresh verifier diagnostic is inconsistent")
    if payload["status"] == "passed" and (
        payload["failure_phase"] != "completed"
        or payload["returncode"] != 0
        or payload["timed_out"] is not False
        or payload["payload_validation_completed"] is not True
        or payload["payload_sha256"] is None
        or payload["verifier_process_id"] is None
        or payload["verifier_process_id"] == payload["controller_process_id"]
        or cleanup["complete"] is not True
        or cleanup["returncode_observed"] is not True
        or cleanup["kill_attempted"] is not False
        or cleanup["kill_succeeded"] is not False
        or cleanup["terminate_attempted"] is not False
        or cleanup["terminate_succeeded"] is not False
        or (
            containment["required"]
            and (
                containment["terminate_attempted"] is not True
                or containment["terminate_succeeded"] is not True
            )
        )
        or (
            not containment["required"]
            and (
                containment["terminate_attempted"] is not False
                or containment["terminate_succeeded"] is not False
            )
        )
        or stdout["capture_started"] is not True
        or stdout["captured_size_bytes"] <= 0
        or stdout["overflow"] is not False
        or stdout["eof_observed"] is not True
        or stdout["read_error"] is not False
        or stderr["capture_started"] is not True
        or stderr["captured_size_bytes"] != 0
        or stderr["captured_sha256"] != hashlib.sha256(b"").hexdigest()
        or stderr["overflow"] is not False
        or stderr["eof_observed"] is not True
        or stderr["read_error"] is not False
    ):
        raise ControlError("passed fresh verifier diagnostic is incomplete")
    _canonical_bytes(payload)
    return payload


def _verified_fresh_payload(
    value: object,
    *,
    request: VerifyRequestV2,
    controller_pid: int,
    child_pid: int,
) -> dict[str, Any]:
    request = request.checked()
    payload = _exact_dict(
        value,
        frozenset(
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
        ),
        "fresh verifier payload",
    )
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
        "fresh verifier process boundary",
    )
    if process != {
        "controller_process_id": controller_pid,
        "verifier_process_id": child_pid,
        "verifier_parent_process_id": controller_pid,
        "distinct_processes": True,
        "direct_child_process": True,
        "verification_nonce": request.nonce,
    }:
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
        "fresh verifier successor",
    )
    _exact_path_text(
        successor["directory"],
        request.successor_directory,
        "fresh verifier successor",
    )
    if (
        type(successor["schema_version"]) is not int
        or successor["schema_version"] != 5
        or successor["purpose"] != _TECHNICAL_SUCCESSOR_PURPOSE
        or type(successor["chain_depth"]) is not int
        or successor["chain_depth"] != 4
        or successor["artifact_root_sha256"] != request.artifact_root_sha256
        or successor["sha256_manifest_sha256"] != request.manifest_sha256
        or successor["authorization_sha256"] != request.authorization_sha256
        or successor["intent_sha256"] != request.intent_sha256
    ):
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
        "fresh verifier superseded authority",
    )
    _exact_path_text(
        superseded["directory"],
        request.parent_directory,
        "fresh verifier superseded authority",
    )
    if (
        type(superseded["schema_version"]) is not int
        or superseded["schema_version"] != 4
        or superseded["historically_verified"] is not True
        or superseded["effective_execution_leaf"] is not False
    ):
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
        "fresh verifier bundle",
    )
    if (
        type(bundle["flat_file_count"]) is not int
        or bundle["flat_file_count"] != 8
        or type(bundle["manifest_artifact_count"]) is not int
        or bundle["manifest_artifact_count"] != 6
        or bundle["flat_file_hashes_verified"] is not True
    ):
        raise FreshVerifierError("fresh verifier bundle contract differs")
    _sha(bundle["flat_file_inventory_sha256"], "fresh verifier inventory")
    _sha(
        payload["confirmatory_storage_policy_sha256"],
        "fresh verifier storage policy",
    )
    checks = _exact_dict(
        payload["checks"],
        _FRESH_CHECK_FIELDS,
        "fresh verifier checks",
    )
    if (
        any(checks[field] is not True for field in _FRESH_CHECK_FIELDS)
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


def _fresh_diagnostic_record(
    *,
    status: str,
    failure_phase: str,
    request: VerifyRequestV2,
    argv: Sequence[str],
    effective_executable: str,
    controller_pid: int,
    verifier_pid: int | None,
    returncode: int | None,
    timeout_milliseconds: int,
    timed_out: bool,
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    payload_sha256: str | None,
    payload_validation_completed: bool,
) -> dict[str, Any]:
    request_record = _fresh_request_record(request)
    payload = {
        "schema_version": FRESH_DIAGNOSTIC_SCHEMA_VERSION,
        "policy": FRESH_DIAGNOSTIC_POLICY,
        "status": status,
        "failure_phase": failure_phase,
        "requested_python_executable": request.python_executable,
        "effective_spawn_executable": effective_executable,
        "executable_override_used": (not _same(request.python_executable, effective_executable)),
        "request": request_record,
        "request_sha256": _compact_sha256(request_record),
        "argv_sha256": _compact_sha256(list(argv)),
        "controller_process_id": controller_pid,
        "verifier_process_id": verifier_pid,
        "returncode": returncode,
        "timeout_milliseconds": timeout_milliseconds,
        "timed_out": timed_out,
        "stdout": dict(stdout),
        "stderr": dict(stderr),
        "cleanup": dict(cleanup),
        "payload_sha256": payload_sha256,
        "payload_validation_completed": payload_validation_completed,
        "stdout_content_included": False,
        "stderr_content_included": False,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    return _canonical_fresh_diagnostic(payload)


def run_fresh_verifier_v2(
    request: VerifyRequestV2,
    *,
    timeout_seconds: float = 900.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    tree_factory: Callable[[int], _ProcessTreeTracker] = _ProcessTreeTracker,
    containment_factory: Callable[
        [],
        _WindowsJobContainment,
    ] = _create_windows_job_containment,
) -> VerifyResultV2:
    """Run one bounded direct child and retain only hashes plus validated JSON."""

    request = request.checked()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise FreshVerifierError("fresh verifier timeout must be positive and finite")
    timeout_milliseconds = max(1, round(float(timeout_seconds) * 1000))
    controller_pid = os.getpid()
    argv = request.argv(controller_pid)
    override = _fresh_verifier_spawn_executable(request.python_executable)
    effective_executable = request.python_executable if override is None else override
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
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
    containment = containment_factory()
    if type(containment) is not _WindowsJobContainment:
        raise FreshVerifierError("fresh verifier containment factory is untyped")
    if not containment.ready_before_spawn:
        cleanup = _cleanup_without_spawned_process(containment)
        diagnostic = _fresh_diagnostic_record(
            status="spawn_failed",
            failure_phase="spawn",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=None,
            returncode=None,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=_empty_stream_diagnostic(_MAX_STDOUT_BYTES),
            stderr=_empty_stream_diagnostic(_MAX_STDERR_BYTES),
            cleanup=cleanup,
            payload_sha256=None,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier Windows Job setup failed",
            diagnostic=diagnostic,
        )
    if containment.required:
        popen_arguments["creationflags"] = getattr(
            subprocess,
            "CREATE_SUSPENDED",
            0x00000004,
        )
    if override is not None:
        popen_arguments["executable"] = override
    try:
        process = popen_factory(list(argv), **popen_arguments)
    except BaseException as exc:
        cleanup = _cleanup_without_spawned_process(containment)
        diagnostic = _fresh_diagnostic_record(
            status="spawn_failed",
            failure_phase="spawn",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=None,
            returncode=None,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=_empty_stream_diagnostic(_MAX_STDOUT_BYTES),
            stderr=_empty_stream_diagnostic(_MAX_STDERR_BYTES),
            cleanup=cleanup,
            payload_sha256=None,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier process could not be created",
            diagnostic=diagnostic,
        ) from exc
    containment.mark_child_created(suspended=containment.required)
    if type(getattr(process, "pid", None)) is not int or process.pid <= 0:
        cleanup = _cleanup_verifier_process(
            process,
            (),
            force=True,
            tree=None,
            containment=containment,
        )
        diagnostic = _fresh_diagnostic_record(
            status="spawn_failed",
            failure_phase="spawn",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=None,
            returncode=None,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=_empty_stream_diagnostic(_MAX_STDOUT_BYTES),
            stderr=_empty_stream_diagnostic(_MAX_STDERR_BYTES),
            cleanup=cleanup,
            payload_sha256=None,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier process identity is unavailable",
            diagnostic=diagnostic,
        )
    verifier_pid = process.pid
    readers: tuple[_BoundedPipeReader, ...] = ()
    phase: str | None = None
    timed_out = False
    tree: _ProcessTreeTracker | None = None
    setup_error: BaseException | None = None
    try:
        containment.assign_and_resume(process)
        tree = tree_factory(verifier_pid)
        tree.refresh()
        if process.stdout is None or process.stderr is None:
            phase = "pipe_read"
        else:
            readers = (
                _BoundedPipeReader(process.stdout, "stdout", _MAX_STDOUT_BYTES),
                _BoundedPipeReader(process.stderr, "stderr", _MAX_STDERR_BYTES),
            )
            for reader in readers:
                reader.start()
    except BaseException as exc:
        setup_error = exc
        phase = "cleanup"
    if setup_error is not None:
        cleanup = _cleanup_verifier_process(
            process,
            readers,
            force=True,
            tree=tree,
            containment=containment,
        )
        by_role = {reader.role: reader for reader in readers}
        stdout_reader = by_role.get("stdout")
        stderr_reader = by_role.get("stderr")
        diagnostic = _fresh_diagnostic_record(
            status="failed",
            failure_phase=("cleanup" if cleanup["complete"] is not True else "wait"),
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
            returncode=(
                process.returncode if type(getattr(process, "returncode", None)) is int else None
            ),
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=(
                _empty_stream_diagnostic(_MAX_STDOUT_BYTES)
                if stdout_reader is None
                else stdout_reader.record()
            ),
            stderr=(
                _empty_stream_diagnostic(_MAX_STDERR_BYTES)
                if stderr_reader is None
                else stderr_reader.record()
            ),
            cleanup=cleanup,
            payload_sha256=None,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier suspended-child setup failed closed",
            diagnostic=diagnostic,
        ) from setup_error
    assert tree is not None
    deadline = time.monotonic() + float(timeout_seconds)
    while phase is None:
        tree.refresh()
        if tree.probe_failed or tree.known:
            phase = "wait"
            break
        if any(reader.error or reader.overflow for reader in readers):
            phase = "pipe_read"
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            phase = "wait"
            break
        try:
            process.wait(timeout=min(_WAIT_SLICE_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        except BaseException:
            phase = "wait"
        else:
            break
    if phase is None:
        while any(reader.thread is not None and reader.thread.is_alive() for reader in readers):
            tree.refresh()
            if tree.probe_failed or tree.known:
                phase = "wait"
                break
            if any(reader.error or reader.overflow for reader in readers):
                phase = "pipe_read"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                phase = "wait"
                break
            for reader in readers:
                if reader.thread is not None:
                    reader.thread.join(timeout=min(_WAIT_SLICE_SECONDS, remaining))
    cleanup = _cleanup_verifier_process(
        process,
        readers,
        force=phase is not None,
        tree=tree,
        containment=containment,
    )
    by_role = {reader.role: reader for reader in readers}
    stdout_reader = by_role.get("stdout")
    stderr_reader = by_role.get("stderr")
    stdout_record = (
        _empty_stream_diagnostic(_MAX_STDOUT_BYTES)
        if stdout_reader is None
        else stdout_reader.record()
    )
    stderr_record = (
        _empty_stream_diagnostic(_MAX_STDERR_BYTES)
        if stderr_reader is None
        else stderr_reader.record()
    )
    stdout_bytes = b"" if stdout_reader is None else bytes(stdout_reader.payload)
    stderr_bytes = b"" if stderr_reader is None else bytes(stderr_reader.payload)
    returncode = process.returncode if type(getattr(process, "returncode", None)) is int else None
    if not cleanup["complete"]:
        phase = "cleanup"
    if phase is not None:
        diagnostic = _fresh_diagnostic_record(
            status="failed",
            failure_phase=phase,
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
            returncode=returncode,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=timed_out,
            stdout=stdout_record,
            stderr=stderr_record,
            cleanup=cleanup,
            payload_sha256=None,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier process interaction failed closed",
            diagnostic=diagnostic,
        )
    payload_sha256 = hashlib.sha256(stdout_bytes).hexdigest()
    if returncode != 0 or stderr_bytes:
        diagnostic = _fresh_diagnostic_record(
            status="failed",
            failure_phase="returncode",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
            returncode=returncode,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=stdout_record,
            stderr=stderr_record,
            cleanup=cleanup,
            payload_sha256=payload_sha256,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier process did not exit cleanly",
            diagnostic=diagnostic,
        )
    try:
        raw = _strict_json_object(stdout_bytes, "fresh verifier stdout")
    except ControlError as exc:
        diagnostic = _fresh_diagnostic_record(
            status="failed",
            failure_phase="payload_parse",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
            returncode=returncode,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=stdout_record,
            stderr=stderr_record,
            cleanup=cleanup,
            payload_sha256=payload_sha256,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier stdout is not one JSON object",
            diagnostic=diagnostic,
        ) from exc
    try:
        payload = _verified_fresh_payload(
            raw,
            request=request,
            controller_pid=controller_pid,
            child_pid=verifier_pid,
        )
    except (ControlError, TypeError, ValueError) as exc:
        diagnostic = _fresh_diagnostic_record(
            status="failed",
            failure_phase="payload_validation",
            request=request,
            argv=argv,
            effective_executable=effective_executable,
            controller_pid=controller_pid,
            verifier_pid=verifier_pid,
            returncode=returncode,
            timeout_milliseconds=timeout_milliseconds,
            timed_out=False,
            stdout=stdout_record,
            stderr=stderr_record,
            cleanup=cleanup,
            payload_sha256=payload_sha256,
            payload_validation_completed=False,
        )
        raise FreshVerifierError(
            "fresh verifier payload failed exact validation",
            diagnostic=diagnostic,
        ) from exc
    diagnostic = _fresh_diagnostic_record(
        status="passed",
        failure_phase="completed",
        request=request,
        argv=argv,
        effective_executable=effective_executable,
        controller_pid=controller_pid,
        verifier_pid=verifier_pid,
        returncode=returncode,
        timeout_milliseconds=timeout_milliseconds,
        timed_out=False,
        stdout=stdout_record,
        stderr=stderr_record,
        cleanup=cleanup,
        payload_sha256=payload_sha256,
        payload_validation_completed=True,
    )
    return VerifyResultV2(
        request=request,
        argv=argv,
        process_id=verifier_pid,
        payload=payload,
        payload_sha256=payload_sha256,
        diagnostic=diagnostic,
    )


def _not_invoked_fresh_diagnostic(
    *,
    timeout_seconds: float,
    project_root: Path,
    successor_directory: Path,
    parent_directory: Path,
    authorization_sha256: str,
    intent_sha256: str,
    verification_nonce: str,
) -> dict[str, Any]:
    request = VerifyRequestV2(
        project_root=project_root,
        successor_directory=successor_directory,
        parent_directory=parent_directory,
        artifact_root_sha256="0" * 64,
        manifest_sha256="0" * 64,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        nonce=verification_nonce,
    ).checked()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise FreshVerifierError("fresh verifier timeout must be positive and finite")
    override = _fresh_verifier_spawn_executable(request.python_executable)
    effective = request.python_executable if override is None else override
    timeout_milliseconds = max(1, round(float(timeout_seconds) * 1000))
    return _fresh_diagnostic_record(
        status="not_invoked",
        failure_phase="not_started",
        request=request,
        argv=(),
        effective_executable=effective,
        controller_pid=os.getpid(),
        verifier_pid=None,
        returncode=None,
        timeout_milliseconds=timeout_milliseconds,
        timed_out=False,
        stdout=_empty_stream_diagnostic(_MAX_STDOUT_BYTES),
        stderr=_empty_stream_diagnostic(_MAX_STDERR_BYTES),
        cleanup=_empty_cleanup_diagnostic(),
        payload_sha256=None,
        payload_validation_completed=False,
    )


class ProductionSchemaV3API:
    """Concrete, narrowly bound adapter around the public amendment owner."""

    def __init__(
        self,
        *,
        project_root: Path,
        parent: Path,
        destination: Path,
        config_path: Path,
        timestamp: datetime,
        storage_policy: Mapping[str, Any],
        terminal_lineage: Mapping[str, Any],
        expected_authorization_sha256: str,
        expected_intent_sha256: str,
    ) -> None:
        self.project_root = _absolute(project_root)
        self.parent = _absolute(parent)
        self.destination = _absolute(destination)
        self.config_path = _absolute(config_path)
        if timestamp.tzinfo is None:
            raise ControlError("schema-v3 adapter timestamp must be timezone-aware")
        self.timestamp = timestamp.astimezone(UTC)
        self.storage_policy = dict(storage_policy)
        self.terminal_lineage = dict(terminal_lineage)
        self.expected_authorization_sha256 = _sha(
            expected_authorization_sha256,
            "schema-v3 adapter authorization",
        )
        self.expected_intent_sha256 = _sha(
            expected_intent_sha256,
            "schema-v3 adapter intent",
        )

    def canonicalize_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> dict[str, Any]:
        if dict(replacement_publication_failure_lineage) != self.terminal_lineage:
            raise ControlError("schema-v3 adapter terminal lineage differs")
        canonical = _canonical_schema_v3_technical_authorization(
            dict(authorization),
            terminal_lineage=self.terminal_lineage,
        )
        if _compact_sha256(canonical) != self.expected_authorization_sha256:
            raise ControlError("schema-v3 adapter authorization hash differs")
        return canonical

    def create_authority(
        self,
        *,
        authorization: Mapping[str, Any],
        post_publication_check: Callable[[Any], None],
    ) -> Any:
        from histo_audit.workflows.preregistration_amendment import (
            create_preregistration_amendment,
        )

        canonical = self.canonicalize_authorization(
            authorization,
            replacement_publication_failure_lineage=self.terminal_lineage,
        )
        return create_preregistration_amendment(
            project_root=self.project_root,
            parent_authority_directory=self.parent,
            amendment_root=self.parent.parent,
            preregistration_path=self.parent / "PRE_REGISTRATION_FROZEN.md",
            primary_config_path=self.parent / "primary_frozen.yaml",
            confirmatory_config_path=self.config_path,
            reason=_AMENDMENT_REASON,
            affected_hypotheses=_AFFECTED_HYPOTHESES,
            affected_analyses=_AFFECTED_ANALYSES,
            outcomes_inspected=True,
            outcomes_inspected_at=_OUTCOMES_INSPECTED_AT,
            resource_bounded_technical_successor_authorization=canonical,
            confirmatory_storage_policy=self.storage_policy,
            post_publication_check=post_publication_check,
            timestamp=self.timestamp,
        )

    def authority_pins(self, published: Any) -> AuthorityPins:
        from histo_audit.workflows.preregistration_amendment import (
            PreregistrationAmendmentResult,
        )

        if type(published) is not PreregistrationAmendmentResult:
            raise AmbiguousStateError("schema-v3 creator returned an untyped result")
        pins = AuthorityPins(
            directory=_absolute(published.amendment_directory),
            parent_directory=_absolute(published.parent_authority_directory),
            artifact_root_sha256=_sha(
                published.artifact_root_sha256,
                "schema-v3 published artifact root",
            ),
            sha256_manifest_sha256=_sha(
                published.sha256_manifest_sha256,
                "schema-v3 published manifest",
            ),
            authorization_sha256=self.expected_authorization_sha256,
            intent_sha256=self.expected_intent_sha256,
            chain_depth=published.chain_depth,
        )
        if (
            not _same(pins.directory, self.destination)
            or not _same(pins.parent_directory, self.parent)
            or type(pins.chain_depth) is not int
            or pins.chain_depth != 4
        ):
            raise AmbiguousStateError("schema-v3 creator result paths or depth differ")
        return pins

    def verify_committed(
        self,
        authority: Path,
        *,
        expected: AuthorityPins,
        replacement_publication_failure_lineage: Mapping[str, Any],
    ) -> None:
        from histo_audit.workflows import preregistration_amendment as amendment

        candidate = _absolute(authority)
        if (
            not _same(candidate, expected.directory)
            or not _same(expected.directory, self.destination)
            or not _same(expected.parent_directory, self.parent)
            or expected.authorization_sha256 != self.expected_authorization_sha256
            or expected.intent_sha256 != self.expected_intent_sha256
            or type(expected.chain_depth) is not int
            or expected.chain_depth != 4
            or dict(replacement_publication_failure_lineage) != self.terminal_lineage
        ):
            raise ControlError("schema-v3 committed verification scope differs")
        generic = amendment.verify_preregistration_amendment(candidate)
        if (
            not generic.valid
            or generic.parent_authority_directory != expected.parent_directory
            or generic.chain_depth != expected.chain_depth
            or generic.artifact_root_sha256 != expected.artifact_root_sha256
            or generic.sha256_manifest_sha256 != expected.sha256_manifest_sha256
        ):
            raise ControlError("schema-v3 committed generic pins differ")
        canonical = amendment._require_resource_bounded_technical_successor_authorization(
            candidate,
            verify_live_primary=False,
            verify_live_receipt=False,
            enforce_unique_leaf=True,
        )
        if (
            _compact_sha256(canonical) != expected.authorization_sha256
            or canonical.get("replacement_publication_failure_lineage") != self.terminal_lineage
        ):
            raise ControlError("schema-v3 committed typed authorization differs")
        evidence = _strict_json_object(
            _read_bytes(
                candidate / "amendment_evidence.json",
                "schema-v3 committed amendment evidence",
            ),
            "schema-v3 committed amendment evidence",
        )
        amendment_timestamp = evidence.get("amendment_timestamp_utc")
        reason = evidence.get("reason")
        outcomes_inspected_at = evidence.get("outcomes_inspected_at_utc")
        if (
            type(amendment_timestamp) is not str
            or type(reason) is not str
            or type(outcomes_inspected_at) is not str
        ):
            raise ControlError("schema-v3 committed intent text fields differ")
        intent_sha256 = amendment.resource_bounded_technical_successor_intent_sha256(
            parent_authority_directory=self.parent,
            amendment_timestamp_utc=amendment_timestamp,
            reason=reason,
            affected_hypotheses=evidence.get("affected_hypotheses", ()),
            affected_analyses=evidence.get("affected_analyses", ()),
            outcomes_inspected_at_utc=outcomes_inspected_at,
            authorization=canonical,
            confirmatory_storage_policy=evidence.get(
                "confirmatory_storage_policy",
                {},
            ),
        )
        if intent_sha256 != expected.intent_sha256:
            raise ControlError("schema-v3 committed intent differs")


class TransactionVerifierV2:
    """One-use rollback-scoped creator callback with in-memory diagnostics only."""

    def __init__(
        self,
        *,
        api: SchemaV3API,
        project_root: Path,
        parent: Path,
        destination: Path,
        authorization_sha256: str,
        intent_sha256: str,
        verification_nonce: str,
        timeout_seconds: float = 900.0,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        containment_factory: Callable[
            [],
            _WindowsJobContainment,
        ] = _create_windows_job_containment,
        canary_popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.api = api
        self.project_root = _absolute(project_root)
        self.parent = _absolute(parent)
        self.destination = _absolute(destination)
        self.authorization_sha256 = _sha(
            authorization_sha256,
            "transaction verifier authorization",
        )
        self.intent_sha256 = _sha(intent_sha256, "transaction verifier intent")
        self.verification_nonce = _sha(
            verification_nonce,
            "transaction verifier precommitted nonce",
        )
        self.timeout_seconds = timeout_seconds
        self.popen_factory = popen_factory
        self.containment_factory = containment_factory
        self.canary_popen_factory = canary_popen_factory
        self.containment_preflight_completed = False
        self.result: VerifyResultV2 | None = None
        self.published_result: Any | None = None
        self.invoked = False
        self.diagnostic = _not_invoked_fresh_diagnostic(
            timeout_seconds=timeout_seconds,
            project_root=self.project_root,
            successor_directory=self.destination,
            parent_directory=self.parent,
            authorization_sha256=self.authorization_sha256,
            intent_sha256=self.intent_sha256,
            verification_nonce=self.verification_nonce,
        )

    def preflight_containment(self) -> None:
        if self.containment_preflight_completed:
            raise FreshVerifierError(
                "transaction verifier containment preflight ran more than once"
            )
        _preflight_windows_job_containment(
            python_executable=sys.executable,
            containment_factory=self.containment_factory,
            popen_factory=self.canary_popen_factory,
        )
        self.containment_preflight_completed = True

    def __call__(self, published: Any) -> None:
        if self.invoked:
            raise FreshVerifierError("transaction verifier callback ran more than once")
        if not self.containment_preflight_completed:
            raise FreshVerifierError(
                "transaction verifier callback lacks its pre-A2 containment gate"
            )
        self.invoked = True
        pins = self.api.authority_pins(published)
        if (
            not _same(pins.directory, self.destination)
            or not _same(pins.parent_directory, self.parent)
            or pins.authorization_sha256 != self.authorization_sha256
            or pins.intent_sha256 != self.intent_sha256
            or pins.chain_depth != 4
        ):
            raise FreshVerifierError("transaction verifier published pins differ")
        request = VerifyRequestV2(
            project_root=self.project_root,
            successor_directory=pins.directory,
            parent_directory=pins.parent_directory,
            artifact_root_sha256=pins.artifact_root_sha256,
            manifest_sha256=pins.sha256_manifest_sha256,
            authorization_sha256=pins.authorization_sha256,
            intent_sha256=pins.intent_sha256,
            nonce=self.verification_nonce,
            chain_depth=pins.chain_depth,
        )
        try:
            result = run_fresh_verifier_v2(
                request,
                timeout_seconds=self.timeout_seconds,
                popen_factory=self.popen_factory,
                containment_factory=self.containment_factory,
            )
        except FreshVerifierError as exc:
            if exc.diagnostic is not None:
                self.diagnostic = _canonical_fresh_diagnostic(exc.diagnostic)
            raise
        self.result = result
        self.diagnostic = result.diagnostic
        self.published_result = published


@dataclass(frozen=True, slots=True)
class PublicationResultV2:
    state: State
    marker_path: Path
    marker_sha256: str
    authority_directory: Path | None


def _build_attempt_v2(
    *,
    namespace: Namespace,
    authorization: Mapping[str, Any],
    authorization_receipt_sha256: str,
    claimed_at: datetime,
) -> dict[str, Any]:
    if claimed_at.tzinfo is None:
        raise ControlError("replacement-v2 claim clock must be timezone-aware")
    contract = authorization["preflight"]["contract"]
    publication = authorization["publication"]
    authorization_bytes = _read_bytes(
        namespace.authorization_v2,
        "replacement-v2 authorization before claim",
    )
    if hashlib.sha256(authorization_bytes).hexdigest() != authorization_receipt_sha256:
        raise ControlError("replacement-v2 authorization changed before claim")
    payload = {
        "schema_version": 2,
        "policy": ATTEMPT_V2_POLICY,
        "status": "claimed",
        "claimed_at_utc": _timestamp(claimed_at),
        "attempt_id": authorization["authorized_attempt_id"],
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "publication_authorization_receipt": {
            "path": str(_absolute(namespace.authorization_v2)),
            "size_bytes": len(authorization_bytes),
            "sha256": authorization_receipt_sha256,
        },
        "terminal_qualification": {
            "path": str(_absolute(namespace.terminal_qualification)),
            "sha256": contract["terminal_qualification"]["terminal_qualification_receipt_sha256"],
        },
        "frozen_input_bundle": contract["frozen_input_bundle"],
        "parent_authority_directory": publication["parent_authority_directory"],
        "intended_authority_directory": publication["intended_authority_directory"],
        "amendment_timestamp_utc": publication["amendment_timestamp_utc"],
        "controller": contract["controller"],
        "source": contract["source"],
        "run_state": contract["run_state"],
        "technical_authorization_sha256": contract["technical_successor"]["authorization_sha256"],
        "intent_sha256": contract["technical_successor"]["intent_sha256"],
        "verification_nonce": _verification_nonce_v2(
            authorization,
            authorization_receipt_sha256,
        ),
        "preflight_fingerprint_sha256": authorization["preflight"]["preflight_fingerprint_sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    canonical = _canonical_attempt_v2(
        payload,
        namespace=namespace,
        authorization=authorization,
        authorization_receipt_sha256=authorization_receipt_sha256,
    )
    if canonical != payload:
        raise ControlError("replacement-v2 attempt builder is noncanonical")
    return payload


_SUCCESS_V2_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "committed_at_utc",
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
        "fresh_verifier_validated_payload",
        "fresh_verifier_validated_payload_sha256",
        "fresh_verifier_payload_sha256",
        "fresh_verifier_diagnostic",
        "fresh_verifier_diagnostic_sha256",
        "controller_process_id",
        "verifier_process_id",
        "verifier_parent_process_id",
        "chain_depth",
        "run_state_sha256",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_FAILURE_V2_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "status",
        "failed_at_utc",
        "failure_phase",
        "automatic_retry_allowed",
        "attempt_id",
        "attempt_marker_sha256",
        "intended_authority_directory",
        "parent_authority_directory",
        "error_type_sha256",
        "error_sha256",
        "fresh_verifier_validated_payload",
        "fresh_verifier_validated_payload_sha256",
        "fresh_verifier_diagnostic",
        "fresh_verifier_diagnostic_sha256",
        "rollback_checked_at_utc",
        "rollback_scan_count",
        "candidate_directories_after_rollback",
        "authority_absent_after_rollback",
        "run_state_sha256",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "publication_performed",
    }
)
_FAILURE_PHASES = frozenset(
    {
        "authority_creation_before_fresh_verifier",
        "fresh_verifier",
        "authority_creation_after_fresh_verifier",
    }
)


def _canonical_success_v2(
    value: object,
    *,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
) -> dict[str, Any]:
    payload = _exact_dict(value, _SUCCESS_V2_FIELDS, "replacement-v2 success marker")
    diagnostic = _canonical_fresh_diagnostic(payload["fresh_verifier_diagnostic"])
    request = diagnostic["request"]
    attempted_parent = _absolute(attempt["parent_authority_directory"])
    try:
        attempted_project = attempted_parent.parents[2]
    except IndexError as exc:
        raise ControlError("replacement-v2 attempted parent cannot bind a project root") from exc
    request_object = VerifyRequestV2(
        project_root=Path(request["project_root"]),
        successor_directory=Path(request["successor_directory"]),
        parent_directory=Path(request["parent_directory"]),
        artifact_root_sha256=request["artifact_root_sha256"],
        manifest_sha256=request["manifest_sha256"],
        authorization_sha256=request["authorization_sha256"],
        intent_sha256=request["intent_sha256"],
        nonce=request["nonce"],
        chain_depth=request["chain_depth"],
        python_executable=request["python_executable"],
    ).checked()
    validated_payload = _verified_fresh_payload(
        payload["fresh_verifier_validated_payload"],
        request=request_object,
        controller_pid=payload["controller_process_id"],
        child_pid=payload["verifier_process_id"],
    )
    claimed_at = datetime.strptime(
        attempt["claimed_at_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    committed_at = datetime.strptime(
        _canonical_timestamp(payload["committed_at_utc"], "replacement-v2 commit time"),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["policy"] != SUCCESS_V2_POLICY
        or payload["status"] != "committed"
        or payload["automatic_retry_allowed"] is not False
        or committed_at < claimed_at
        or committed_at > datetime.now(UTC)
        or payload["attempt_id"] != attempt["attempt_id"]
        or payload["attempt_marker_sha256"] != attempt_sha256
        or payload["authority_directory"] != attempt["intended_authority_directory"]
        or payload["parent_authority_directory"] != attempt["parent_authority_directory"]
        or payload["authorization_sha256"] != attempt["technical_authorization_sha256"]
        or payload["intent_sha256"] != attempt["intent_sha256"]
        or payload["authority_directory"] != request["successor_directory"]
        or payload["parent_authority_directory"] != request["parent_directory"]
        or payload["artifact_root_sha256"] != request["artifact_root_sha256"]
        or payload["sha256_manifest_sha256"] != request["manifest_sha256"]
        or payload["authorization_sha256"] != request["authorization_sha256"]
        or payload["intent_sha256"] != request["intent_sha256"]
        or payload["verification_nonce"] != attempt["verification_nonce"]
        or payload["verification_nonce"] != request["nonce"]
        or request["project_root"] != str(attempted_project)
        or payload["chain_depth"] != request["chain_depth"]
        or payload["run_state_sha256"] != attempt["run_state"]["sha256"]
        or diagnostic["status"] != "passed"
        or diagnostic["cleanup"]["complete"] is not True
        or payload["fresh_verifier_payload_sha256"] != diagnostic["payload_sha256"]
        or payload["fresh_verifier_validated_payload"] != validated_payload
        or payload["fresh_verifier_validated_payload_sha256"]
        != hashlib.sha256(_canonical_bytes(validated_payload)).hexdigest()
        or payload["fresh_verifier_diagnostic_sha256"] != _compact_sha256(diagnostic)
        or payload["controller_process_id"] != diagnostic["controller_process_id"]
        or payload["verifier_process_id"] != diagnostic["verifier_process_id"]
        or payload["verifier_parent_process_id"] != diagnostic["controller_process_id"]
        or payload["verifier_process_id"] == payload["controller_process_id"]
        or type(payload["chain_depth"]) is not int
        or payload["chain_depth"] != 4
        or any(
            payload[field] is not expected
            for field, expected in (
                ("outcome_value_interpretation_performed", False),
                ("scientific_execution_performed", False),
                ("publication_performed", True),
            )
        )
    ):
        raise ControlError("replacement-v2 success marker contract differs")
    for field in (
        "attempt_id",
        "attempt_marker_sha256",
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "authorization_sha256",
        "intent_sha256",
        "verification_nonce",
        "fresh_verifier_validated_payload_sha256",
        "fresh_verifier_payload_sha256",
        "fresh_verifier_diagnostic_sha256",
        "run_state_sha256",
    ):
        _sha(payload[field], f"replacement-v2 success {field}")
    _exact_path_text(
        payload["authority_directory"],
        attempt["intended_authority_directory"],
        "replacement-v2 committed authority",
    )
    _exact_path_text(
        payload["parent_authority_directory"],
        attempt["parent_authority_directory"],
        "replacement-v2 committed parent",
    )
    for field in (
        "controller_process_id",
        "verifier_process_id",
        "verifier_parent_process_id",
    ):
        _positive_int(payload[field], f"replacement-v2 success {field}")
    _canonical_bytes(payload)
    return payload


def _canonical_failure_v2(
    value: object,
    *,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
) -> dict[str, Any]:
    payload = _exact_dict(value, _FAILURE_V2_FIELDS, "replacement-v2 failure marker")
    diagnostic = _canonical_fresh_diagnostic(payload["fresh_verifier_diagnostic"])
    request = diagnostic["request"]
    attempted_parent = _absolute(attempt["parent_authority_directory"])
    try:
        attempted_project = attempted_parent.parents[2]
    except IndexError as exc:
        raise ControlError("replacement-v2 attempted parent cannot bind a project root") from exc
    validated_payload: dict[str, Any] | None = None
    if diagnostic["status"] == "passed":
        request_object = VerifyRequestV2(
            project_root=Path(request["project_root"]),
            successor_directory=Path(request["successor_directory"]),
            parent_directory=Path(request["parent_directory"]),
            artifact_root_sha256=request["artifact_root_sha256"],
            manifest_sha256=request["manifest_sha256"],
            authorization_sha256=request["authorization_sha256"],
            intent_sha256=request["intent_sha256"],
            nonce=request["nonce"],
            chain_depth=request["chain_depth"],
            python_executable=request["python_executable"],
        ).checked()
        validated_payload = _verified_fresh_payload(
            payload["fresh_verifier_validated_payload"],
            request=request_object,
            controller_pid=diagnostic["controller_process_id"],
            child_pid=diagnostic["verifier_process_id"],
        )
    claimed_at = datetime.strptime(
        attempt["claimed_at_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    failed_at = datetime.strptime(
        _canonical_timestamp(payload["failed_at_utc"], "replacement-v2 failure time"),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    rollback_at = datetime.strptime(
        _canonical_timestamp(
            payload["rollback_checked_at_utc"],
            "replacement-v2 rollback check time",
        ),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    expected_diagnostic_statuses = {
        "authority_creation_before_fresh_verifier": {"not_invoked"},
        "fresh_verifier": {"spawn_failed", "failed"},
        "authority_creation_after_fresh_verifier": {"passed"},
    }
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["policy"] != FAILURE_V2_POLICY
        or payload["status"] != "rolled_back_failure_no_retry"
        or payload["failure_phase"] not in _FAILURE_PHASES
        or payload["automatic_retry_allowed"] is not False
        or not claimed_at <= failed_at <= rollback_at <= datetime.now(UTC)
        or payload["attempt_id"] != attempt["attempt_id"]
        or payload["attempt_marker_sha256"] != attempt_sha256
        or payload["intended_authority_directory"] != attempt["intended_authority_directory"]
        or payload["parent_authority_directory"] != attempt["parent_authority_directory"]
        or request["project_root"] != str(attempted_project)
        or request["successor_directory"] != attempt["intended_authority_directory"]
        or request["parent_directory"] != attempt["parent_authority_directory"]
        or request["authorization_sha256"] != attempt["technical_authorization_sha256"]
        or request["intent_sha256"] != attempt["intent_sha256"]
        or request["nonce"] != attempt["verification_nonce"]
        or request["chain_depth"] != 4
        or payload["fresh_verifier_validated_payload"] != validated_payload
        or payload["fresh_verifier_validated_payload_sha256"]
        != (
            None
            if validated_payload is None
            else hashlib.sha256(_canonical_bytes(validated_payload)).hexdigest()
        )
        or (
            diagnostic["status"] == "not_invoked"
            and (
                request["artifact_root_sha256"] != "0" * 64
                or request["manifest_sha256"] != "0" * 64
            )
        )
        or diagnostic["status"] not in expected_diagnostic_statuses[payload["failure_phase"]]
        or diagnostic["cleanup"]["complete"] is not True
        or payload["fresh_verifier_diagnostic_sha256"] != _compact_sha256(diagnostic)
        or type(payload["rollback_scan_count"]) is not int
        or payload["rollback_scan_count"] != 2
        or type(payload["candidate_directories_after_rollback"]) is not list
        or payload["candidate_directories_after_rollback"] != []
        or payload["authority_absent_after_rollback"] is not True
        or payload["run_state_sha256"] != attempt["run_state"]["sha256"]
        or any(
            payload[field] is not False
            for field in (
                "outcome_value_interpretation_performed",
                "scientific_execution_performed",
                "publication_performed",
            )
        )
    ):
        raise ControlError("replacement-v2 failure marker contract differs")
    for field in (
        "attempt_id",
        "attempt_marker_sha256",
        "error_type_sha256",
        "error_sha256",
        "fresh_verifier_diagnostic_sha256",
        "run_state_sha256",
    ):
        _sha(payload[field], f"replacement-v2 failure {field}")
    _exact_path_text(
        payload["intended_authority_directory"],
        attempt["intended_authority_directory"],
        "replacement-v2 failed destination",
    )
    _exact_path_text(
        payload["parent_authority_directory"],
        attempt["parent_authority_directory"],
        "replacement-v2 failed parent",
    )
    _canonical_bytes(payload)
    return payload


def _build_success_v2(
    *,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    verifier: TransactionVerifierV2,
    committed_at: datetime,
) -> dict[str, Any]:
    if verifier.result is None:
        raise AmbiguousStateError("success marker requires a verified fresh result")
    result = verifier.result
    process = _exact_dict(
        result.payload["process_boundary"],
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
        "success fresh process boundary",
    )
    payload = {
        "schema_version": 2,
        "policy": SUCCESS_V2_POLICY,
        "status": "committed",
        "committed_at_utc": _timestamp(committed_at),
        "automatic_retry_allowed": False,
        "attempt_id": attempt["attempt_id"],
        "attempt_marker_sha256": attempt_sha256,
        "authority_directory": str(result.request.successor_directory),
        "parent_authority_directory": str(result.request.parent_directory),
        "artifact_root_sha256": result.request.artifact_root_sha256,
        "sha256_manifest_sha256": result.request.manifest_sha256,
        "authorization_sha256": result.request.authorization_sha256,
        "intent_sha256": result.request.intent_sha256,
        "verification_nonce": result.request.nonce,
        "fresh_verifier_validated_payload": result.payload,
        "fresh_verifier_validated_payload_sha256": hashlib.sha256(
            _canonical_bytes(result.payload)
        ).hexdigest(),
        "fresh_verifier_payload_sha256": result.payload_sha256,
        "fresh_verifier_diagnostic": result.diagnostic,
        "fresh_verifier_diagnostic_sha256": _compact_sha256(result.diagnostic),
        "controller_process_id": process["controller_process_id"],
        "verifier_process_id": process["verifier_process_id"],
        "verifier_parent_process_id": process["verifier_parent_process_id"],
        "chain_depth": result.request.chain_depth,
        "run_state_sha256": attempt["run_state"]["sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": True,
    }
    canonical = _canonical_success_v2(
        payload,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    if canonical != payload:
        raise ControlError("replacement-v2 success builder is noncanonical")
    return payload


def _build_failure_v2(
    *,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    verifier: TransactionVerifierV2,
    error: BaseException,
    failure_phase: str,
    failed_at: datetime,
    rollback_checked_at: datetime,
) -> dict[str, Any]:
    diagnostic = _canonical_fresh_diagnostic(verifier.diagnostic)
    if diagnostic["cleanup"]["complete"] is not True:
        raise AmbiguousStateError(
            "fresh verifier cleanup is incomplete; failure marker is forbidden"
        )
    error_type = type(error).__name__
    validated_payload = (
        verifier.result.payload
        if diagnostic["status"] == "passed" and verifier.result is not None
        else None
    )
    payload = {
        "schema_version": 2,
        "policy": FAILURE_V2_POLICY,
        "status": "rolled_back_failure_no_retry",
        "failed_at_utc": _timestamp(failed_at),
        "failure_phase": failure_phase,
        "automatic_retry_allowed": False,
        "attempt_id": attempt["attempt_id"],
        "attempt_marker_sha256": attempt_sha256,
        "intended_authority_directory": attempt["intended_authority_directory"],
        "parent_authority_directory": attempt["parent_authority_directory"],
        "error_type_sha256": hashlib.sha256(error_type.encode("utf-8")).hexdigest(),
        "error_sha256": hashlib.sha256(f"{error_type}: {error}".encode()).hexdigest(),
        "fresh_verifier_validated_payload": validated_payload,
        "fresh_verifier_validated_payload_sha256": (
            hashlib.sha256(_canonical_bytes(validated_payload)).hexdigest()
            if validated_payload is not None
            else None
        ),
        "fresh_verifier_diagnostic": diagnostic,
        "fresh_verifier_diagnostic_sha256": _compact_sha256(diagnostic),
        "rollback_checked_at_utc": _timestamp(rollback_checked_at),
        "rollback_scan_count": 2,
        "candidate_directories_after_rollback": [],
        "authority_absent_after_rollback": True,
        "run_state_sha256": attempt["run_state"]["sha256"],
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }
    canonical = _canonical_failure_v2(
        payload,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    if canonical != payload:
        raise ControlError("replacement-v2 failure builder is noncanonical")
    return payload


def _read_success_v2(
    namespace: Namespace,
    *,
    attempt: Mapping[str, Any] | None = None,
    attempt_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    if attempt is None or attempt_sha256 is None:
        attempt, attempt_sha256 = _read_attempt_v2(namespace, verify_live=False)
    encoded = _read_bytes(namespace.success_v2, "replacement-v2 success marker")
    payload = _canonical_success_v2(
        _strict_json_object(encoded, "replacement-v2 success marker"),
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    if encoded != _canonical_bytes(payload):
        raise ControlError("replacement-v2 success marker bytes are not canonical")
    return payload, hashlib.sha256(encoded).hexdigest()


def _read_failure_v2(
    namespace: Namespace,
    *,
    attempt: Mapping[str, Any] | None = None,
    attempt_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    if attempt is None or attempt_sha256 is None:
        attempt, attempt_sha256 = _read_attempt_v2(namespace, verify_live=False)
    encoded = _read_bytes(namespace.failure_v2, "replacement-v2 failure marker")
    payload = _canonical_failure_v2(
        _strict_json_object(encoded, "replacement-v2 failure marker"),
        attempt=attempt,
        attempt_sha256=attempt_sha256,
    )
    if encoded != _canonical_bytes(payload):
        raise ControlError("replacement-v2 failure marker bytes are not canonical")
    return payload, hashlib.sha256(encoded).hexdigest()


def _write_terminal_marker(
    path: Path,
    payload: Mapping[str, Any],
    *,
    role: str,
) -> str:
    encoded = _canonical_bytes(payload)
    published = publish_bytes_no_overwrite(encoded, path)
    readback = _read_bytes(path, role)
    digest = hashlib.sha256(readback).hexdigest()
    if readback != encoded or digest != published.sha256 or not published.still_owned():
        raise AmbiguousStateError(f"{role} write/readback is ambiguous")
    return digest


def _candidate_tuple(
    discoverer: Callable[[str | Path], Sequence[str | Path]],
    parent: Path,
) -> tuple[Path, ...]:
    raw = discoverer(parent)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ControlError("replacement-v2 candidate discoverer returned a non-sequence")
    candidates = tuple(_absolute(item) for item in raw)
    if (
        len(set(candidates)) != len(candidates)
        or any(candidate.parent != parent.parent for candidate in candidates)
        or tuple(sorted(candidates, key=lambda path: str(path).casefold())) != candidates
    ):
        raise ControlError("replacement-v2 candidate set is noncanonical")
    return candidates


def _trusted_protocol_time(
    clock: Callable[[], datetime],
    *,
    role: str,
    not_before: datetime | None = None,
) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ControlError(f"{role} clock is not timezone-aware")
    moment = value.astimezone(UTC)
    if moment > datetime.now(UTC) or (
        not_before is not None and moment < not_before.astimezone(UTC)
    ):
        raise ControlError(f"{role} clock is out of order")
    return moment


def _require_uninvoked_transaction_verifier(
    *,
    verifier: TransactionVerifierV2,
    api: SchemaV3API,
    project: Path,
    parent: Path,
    destination: Path,
    authorization_sha256: str,
    intent_sha256: str,
    verification_nonce: str,
) -> None:
    """Reject a stale or differently scoped callback before A2 can be consumed."""

    expected_diagnostic = _not_invoked_fresh_diagnostic(
        timeout_seconds=verifier.timeout_seconds,
        project_root=project,
        successor_directory=destination,
        parent_directory=parent,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        verification_nonce=verification_nonce,
    )
    if (
        verifier.api is not api
        or not _same(verifier.project_root, project)
        or not _same(verifier.parent, parent)
        or not _same(verifier.destination, destination)
        or verifier.authorization_sha256 != authorization_sha256
        or verifier.intent_sha256 != intent_sha256
        or verifier.verification_nonce != verification_nonce
        or verifier.containment_preflight_completed is not False
        or verifier.invoked is not False
        or verifier.result is not None
        or verifier.published_result is not None
        or _canonical_fresh_diagnostic(verifier.diagnostic) != expected_diagnostic
    ):
        raise ControlError("replacement-v2 transaction verifier is stale or differently scoped")


def _verify_live_governed_baseline_v2(
    namespace: Namespace,
    *,
    parent_authority_directory: Path,
    authorization: Mapping[str, Any],
    authorization_receipt_sha256: str,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    expected_presence: Mapping[str, bool],
    expected_candidates: Sequence[Path],
    verify_live_run_state: bool = True,
    candidate_discoverer: Callable[
        [str | Path],
        Sequence[str | Path],
    ] = discover_candidates,
) -> None:
    """Re-read every governed live referent without applying the historical no-D rule."""

    from histo_audit.config import config_sha256, load_config
    from histo_audit.experiment.study_contracts import (
        validate_resource_bounded_confirmatory_config,
    )

    project, parent = _require_parent(namespace, parent_authority_directory)
    expected_candidate_tuple = tuple(_absolute(path) for path in expected_candidates)
    canonical_presence = {
        "qualification": True,
        "inputs": True,
        "authorization": True,
        "attempt": True,
        "success": bool(expected_presence.get("success")),
        "failure": bool(expected_presence.get("failure")),
    }
    if (
        type(expected_presence) is not dict
        or set(expected_presence) != set(canonical_presence)
        or any(type(expected_presence[field]) is not bool for field in canonical_presence)
        or dict(expected_presence) != canonical_presence
        or len(set(expected_candidate_tuple)) != len(expected_candidate_tuple)
        or type(verify_live_run_state) is not bool
    ):
        raise ControlError("replacement-v2 live-baseline expectation is noncanonical")

    terminal, terminal_sha256 = _read_terminal_qualification(
        namespace,
        pins=DEFAULT_HISTORICAL_PINS,
        verify_live_history=False,
    )
    frozen_payloads, frozen_records, frozen_root = _read_input_v3(
        namespace,
        pins=DEFAULT_HISTORICAL_PINS,
        verify_live=False,
    )
    static_authorization, static_authorization_sha256 = _read_publication_authorization_v2(
        namespace, verify_live=False
    )
    static_attempt, static_attempt_sha256 = _read_attempt_v2(
        namespace,
        authorization=static_authorization,
        authorization_receipt_sha256=static_authorization_sha256,
        verify_live=False,
    )
    contract = static_authorization["preflight"]["contract"]
    terminal_envelope = contract["terminal_qualification"]
    frozen_bundle = contract["frozen_input_bundle"]
    frozen = frozen_payloads["frozen_source_receipt"]
    if (
        static_authorization != authorization
        or static_authorization_sha256 != authorization_receipt_sha256
        or static_attempt != attempt
        or static_attempt_sha256 != attempt_sha256
        or terminal_envelope["terminal_qualification_receipt"] != terminal
        or terminal_envelope["terminal_qualification_receipt_sha256"] != terminal_sha256
        or frozen_bundle["files"] != frozen_records
        or frozen_bundle["records_sha256"] != frozen_root
        or attempt["frozen_input_bundle"] != frozen_bundle
    ):
        raise ControlError("replacement-v2 sealed baseline records changed")

    for record in terminal["lock_quiescence"]["reads_between_scans"]:
        if not verify_live_run_state and str(record["role"]).startswith("run_state_"):
            continue
        observed = _file_record(
            Path(record["path"]),
            f"replacement-v2 governed baseline {record['role']}",
        )
        if any(observed[field] != record[field] for field in ("path", "size_bytes", "sha256")):
            raise ControlError(f"replacement-v2 governed referent changed: {record['role']}")
    controllers = terminal["controller_identities"]
    current_controller = _controller_identity()
    if (
        any(
            current_controller[field] != controllers["qualifying_live_controller"][field]
            for field in ("path", "size_bytes", "sha256")
        )
        or current_controller != contract["controller"]
        or current_controller != attempt["controller"]
        or frozen["controller_path"] != current_controller["path"]
        or frozen["controller_size_bytes"] != current_controller["size_bytes"]
        or frozen["controller_sha256"] != current_controller["sha256"]
    ):
        raise ControlError("replacement-v2 governed controller changed")
    diagnosed = _file_record(
        Path(controllers["diagnosed_fixed_legacy_controller"]["path"]),
        "replacement-v2 governed diagnosed legacy controller",
    )
    if any(
        diagnosed[field] != controllers["diagnosed_fixed_legacy_controller"][field]
        for field in ("path", "size_bytes", "sha256")
    ):
        raise ControlError("replacement-v2 diagnosed legacy controller changed")
    terminal_namespace = terminal["terminal_namespace"]
    if os.path.lexists(terminal_namespace["success_marker_absence"]["path"]) or os.path.lexists(
        terminal_namespace["intended_authority_absence"]["path"]
    ):
        raise ControlError("replacement-v2 historical terminal absence changed")

    historical = _historical_paths(namespace, parent)
    _historical_payloads, historical_records = _input_v2_records(
        historical,
        DEFAULT_HISTORICAL_PINS,
    )
    if (
        historical_records != terminal["frozen_v2_inputs"]["files"]
        or _authority_c_receipt(parent) != terminal["authority_c"]
        or _protected_receipt(project) != terminal["protected_bindings"]
        or (
            verify_live_run_state
            and _run_state_receipt(project, terminal["run_state"]["sha256"])
            != terminal["run_state"]
        )
    ):
        raise ControlError("replacement-v2 historical or protected baseline changed")

    live_run_state: dict[str, Any] | None = None
    if verify_live_run_state:
        live_run_state = _live_run_state_hashes(project)
        if (
            live_run_state != contract["run_state"]
            or live_run_state != attempt["run_state"]
            or frozen["run_state_root"] != live_run_state["root"]
            or frozen["run_state_files"] != live_run_state["files"]
            or frozen["run_state_sha256"] != live_run_state["sha256"]
        ):
            raise ControlError("replacement-v2 governed run-state changed")
    pinned_live_records = {
        "config_file_sha256": _file_record(
            Path(frozen["config_path"]),
            "replacement-v2 governed config",
        )["sha256"],
        "manifest_sha256": _file_record(
            Path(frozen["manifest_path"]),
            "replacement-v2 governed PanNuke manifest",
        )["sha256"],
        "failed_preflight_receipt_sha256": _file_record(
            Path(frozen["failed_preflight_receipt_path"]),
            "replacement-v2 governed failed preflight",
        )["sha256"],
        "prior_failure_receipt_sha256": _file_record(
            Path(frozen["prior_failure_receipt_path"]),
            "replacement-v2 governed prior failure",
        )["sha256"],
        "retired_input_invalidation_receipt_sha256": _file_record(
            Path(frozen["retired_input_invalidation_receipt_path"]),
            "replacement-v2 governed input invalidation",
        )["sha256"],
    }
    if any(frozen[field] != digest for field, digest in pinned_live_records.items()):
        raise ControlError("replacement-v2 governed pinned file changed")
    live_config = validate_resource_bounded_confirmatory_config(
        load_config(Path(frozen["config_path"]))
    )
    if config_sha256(live_config) != frozen["config_semantic_sha256"]:
        raise ControlError("replacement-v2 governed config semantics changed")
    live_source = _derive_live_source_v3(
        _input_v3_live_paths(namespace, project=project, parent=parent)
    )
    if (
        frozen["execution_source_root_sha256"] != live_source["current_source"]["root_sha256"]
        or frozen["execution_source_manifest_sha256"] != live_source["current_manifest_sha256"]
        or frozen["execution_source_delta_sha256"] != live_source["delta_sha256"]
        or frozen["execution_source_artifact_count"]
        != live_source["current_source"]["artifact_count"]
        or frozen_payloads["source_allowlist"] != live_source["allowlist"]
        or contract["source"]
        != {
            "root_sha256": frozen["execution_source_root_sha256"],
            "manifest_sha256": frozen["execution_source_manifest_sha256"],
            "delta_sha256": frozen["execution_source_delta_sha256"],
            "allowlisted_change_count": len(_EXPECTED_SOURCE_CHANGE_KINDS),
        }
        or attempt["source"] != contract["source"]
    ):
        raise ControlError("replacement-v2 governed execution source changed")

    allowed_extra = expected_candidate_tuple[0] if len(expected_candidate_tuple) == 1 else None
    if (
        len(expected_candidate_tuple) > 1
        or _reserved_family_presence(namespace) != canonical_presence
        or _candidate_tuple(candidate_discoverer, parent) != expected_candidate_tuple
    ):
        raise ControlError("replacement-v2 governed namespace changed")
    _stable_amendment_inventory(parent.parent, allowed_extra=allowed_extra)

    repeated_terminal, repeated_terminal_sha256 = _read_terminal_qualification(
        namespace,
        pins=DEFAULT_HISTORICAL_PINS,
        verify_live_history=False,
    )
    repeated_payloads, repeated_records, repeated_root = _read_input_v3(
        namespace,
        pins=DEFAULT_HISTORICAL_PINS,
        verify_live=False,
    )
    repeated_authorization, repeated_authorization_sha256 = _read_publication_authorization_v2(
        namespace, verify_live=False
    )
    repeated_attempt, repeated_attempt_sha256 = _read_attempt_v2(
        namespace,
        authorization=repeated_authorization,
        authorization_receipt_sha256=repeated_authorization_sha256,
        verify_live=False,
    )
    repeated_historical_payloads, repeated_historical_records = _input_v2_records(
        historical,
        DEFAULT_HISTORICAL_PINS,
    )
    del repeated_historical_payloads
    repeated_pinned_records = {
        field: _file_record(
            Path(frozen[path_field]),
            f"repeated replacement-v2 governed {field}",
        )["sha256"]
        for field, path_field in (
            ("config_file_sha256", "config_path"),
            ("manifest_sha256", "manifest_path"),
            (
                "failed_preflight_receipt_sha256",
                "failed_preflight_receipt_path",
            ),
            ("prior_failure_receipt_sha256", "prior_failure_receipt_path"),
            (
                "retired_input_invalidation_receipt_sha256",
                "retired_input_invalidation_receipt_path",
            ),
        )
    }
    repeated_config = validate_resource_bounded_confirmatory_config(
        load_config(Path(frozen["config_path"]))
    )
    repeated_source = _derive_live_source_v3(
        _input_v3_live_paths(namespace, project=project, parent=parent)
    )
    repeated_diagnosed = _file_record(
        Path(controllers["diagnosed_fixed_legacy_controller"]["path"]),
        "repeated replacement-v2 diagnosed legacy controller",
    )
    if (
        (repeated_terminal, repeated_terminal_sha256) != (terminal, terminal_sha256)
        or (repeated_payloads, repeated_records, repeated_root)
        != (frozen_payloads, frozen_records, frozen_root)
        or (repeated_authorization, repeated_authorization_sha256)
        != (static_authorization, static_authorization_sha256)
        or (repeated_attempt, repeated_attempt_sha256) != (static_attempt, static_attempt_sha256)
        or (verify_live_run_state and _live_run_state_hashes(project) != live_run_state)
        or _protected_receipt(project) != terminal["protected_bindings"]
        or _controller_identity() != current_controller
        or repeated_historical_records != historical_records
        or _authority_c_receipt(parent) != terminal["authority_c"]
        or repeated_pinned_records != pinned_live_records
        or config_sha256(repeated_config) != frozen["config_semantic_sha256"]
        or repeated_source != live_source
        or any(
            repeated_diagnosed[field] != controllers["diagnosed_fixed_legacy_controller"][field]
            for field in ("path", "size_bytes", "sha256")
        )
        or os.path.lexists(terminal_namespace["success_marker_absence"]["path"])
        or os.path.lexists(terminal_namespace["intended_authority_absence"]["path"])
        or _candidate_tuple(candidate_discoverer, parent) != expected_candidate_tuple
        or _reserved_family_presence(namespace) != canonical_presence
    ):
        raise AmbiguousStateError("replacement-v2 governed baseline changed during verification")


def _execute_publication_v2_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    authorization: Mapping[str, Any],
    authorization_receipt_sha256: str,
    api: SchemaV3API,
    verifier: TransactionVerifierV2,
    clock: Callable[[], datetime],
    candidate_discoverer: Callable[
        [str | Path],
        Sequence[str | Path],
    ] = discover_candidates,
) -> PublicationResultV2:
    project, parent = _require_parent(namespace, parent_authority_directory)
    destination = _absolute(authorization["publication"]["intended_authority_directory"])
    technical_successor = authorization["preflight"]["contract"]["technical_successor"]
    _require_uninvoked_transaction_verifier(
        verifier=verifier,
        api=api,
        project=project,
        parent=parent,
        destination=destination,
        authorization_sha256=technical_successor["authorization_sha256"],
        intent_sha256=technical_successor["intent_sha256"],
        verification_nonce=_verification_nonce_v2(
            authorization,
            authorization_receipt_sha256,
        ),
    )
    verifier.preflight_containment()
    terminal_lineage = authorization["preflight"]["contract"]["terminal_qualification"]
    legacy_lock_paths = _legacy_scoped_lock_paths(namespace, parent=parent)
    attempt_written = False
    attempt: dict[str, Any] | None = None
    attempt_sha256: str | None = None
    try:
        with (
            _protocol_lock(
                namespace,
                parent=parent,
                role="resource Authority-D replacement-v2 one-shot claim",
            ) as protocol_lock,
            ExclusiveBundlePublicationLock(
                (parent,),
                role="resource Authority-D replacement-v2 claim Authority-C parent guard",
            ) as parent_guard,
        ):
            owned_locks = (protocol_lock, parent_guard)
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            live_authorization, live_authorization_sha256 = _read_publication_authorization_v2(
                namespace, verify_live=True
            )
            if (
                live_authorization != authorization
                or live_authorization_sha256 != authorization_receipt_sha256
                or _reserved_family_presence(namespace)
                != {
                    "qualification": True,
                    "inputs": True,
                    "authorization": True,
                    "attempt": False,
                    "success": False,
                    "failure": False,
                }
            ):
                raise ControlError("replacement-v2 preclaim authorization changed")
            _stable_amendment_inventory(parent.parent)
            if _candidate_tuple(candidate_discoverer, parent):
                raise ControlError("replacement-v2 preclaim found Authority D")
            claimed_at = _trusted_protocol_time(
                clock,
                role="replacement-v2 claim",
            )
            attempt = _build_attempt_v2(
                namespace=namespace,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                claimed_at=claimed_at,
            )
            for lock in owned_locks:
                lock.assert_owned()
            published_attempt = publish_bytes_no_overwrite(
                _canonical_bytes(attempt),
                namespace.attempt_v2,
            )
            attempt_written = True
            for lock in owned_locks:
                lock.assert_owned()
            readback, attempt_sha256 = _read_attempt_v2(
                namespace,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                verify_live=True,
            )
            if (
                readback != attempt
                or attempt_sha256 != published_attempt.sha256
                or not published_attempt.still_owned()
                or _reserved_family_presence(namespace)
                != {
                    "qualification": True,
                    "inputs": True,
                    "authorization": True,
                    "attempt": True,
                    "success": False,
                    "failure": False,
                }
            ):
                raise AmbiguousStateError("replacement-v2 one-shot claim changed")
            _stable_amendment_inventory(parent.parent)
            if _candidate_tuple(candidate_discoverer, parent):
                raise AmbiguousStateError("Authority D appeared during one-shot claim")
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            _verify_live_governed_baseline_v2(
                namespace,
                parent_authority_directory=parent,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                expected_presence={
                    "qualification": True,
                    "inputs": True,
                    "authorization": True,
                    "attempt": True,
                    "success": False,
                    "failure": False,
                },
                expected_candidates=(),
                candidate_discoverer=candidate_discoverer,
            )
            for lock in owned_locks:
                lock.assert_owned()
    except BaseException as exc:
        if attempt_written:
            raise AmbiguousStateError(
                "replacement-v2 A2 was retained but its claim boundary did not close; stop"
            ) from exc
        raise
    if attempt is None or attempt_sha256 is None:  # pragma: no cover - defensive
        raise AmbiguousStateError("replacement-v2 claim returned without exact A2")

    def governed_post_publication_check(published: Any) -> None:
        expected_presence = {
            "qualification": True,
            "inputs": True,
            "authorization": True,
            "attempt": True,
            "success": False,
            "failure": False,
        }
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence=expected_presence,
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )
        verifier(published)
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence=expected_presence,
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )

    try:
        transaction_result = api.create_authority(
            authorization=authorization["preflight"]["contract"]["technical_successor"][
                "authorization"
            ],
            post_publication_check=governed_post_publication_check,
        )
    except BaseException as creator_error:
        diagnostic = _canonical_fresh_diagnostic(verifier.diagnostic)
        if diagnostic["cleanup"]["complete"] is not True:
            raise AmbiguousStateError(
                "creator failed with incomplete fresh-verifier cleanup; A2 retained"
            ) from creator_error
        if diagnostic["status"] == "not_invoked":
            failure_phase = "authority_creation_before_fresh_verifier"
        elif diagnostic["status"] in {"spawn_failed", "failed"}:
            failure_phase = "fresh_verifier"
        elif diagnostic["status"] == "passed":
            failure_phase = "authority_creation_after_fresh_verifier"
        else:  # pragma: no cover - canonical diagnostic is exhaustive
            raise AmbiguousStateError(
                "creator failure diagnostic is unsupported"
            ) from creator_error
        failed_at = _trusted_protocol_time(
            clock,
            role="replacement-v2 creator failure",
        )
        result: PublicationResultV2 | None = None
        with (
            _protocol_lock(
                namespace,
                parent=parent,
                role="resource Authority-D replacement-v2 rollback disposition",
            ) as protocol_lock,
            ExclusiveBundlePublicationLock(
                (parent,),
                role="resource Authority-D replacement-v2 rollback Authority-C parent guard",
            ) as parent_guard,
        ):
            owned_locks = (protocol_lock, parent_guard)
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            static_authorization, static_authorization_sha256 = _read_publication_authorization_v2(
                namespace, verify_live=False
            )
            static_attempt, static_attempt_sha256 = _read_attempt_v2(
                namespace,
                authorization=static_authorization,
                authorization_receipt_sha256=static_authorization_sha256,
                verify_live=False,
            )
            if (
                static_authorization != authorization
                or static_authorization_sha256 != authorization_receipt_sha256
                or static_attempt != attempt
                or static_attempt_sha256 != attempt_sha256
                or _reserved_family_presence(namespace)
                != {
                    "qualification": True,
                    "inputs": True,
                    "authorization": True,
                    "attempt": True,
                    "success": False,
                    "failure": False,
                }
            ):
                raise AmbiguousStateError(
                    "replacement-v2 rollback disposition inputs changed"
                ) from creator_error
            rollback_presence = {
                "qualification": True,
                "inputs": True,
                "authorization": True,
                "attempt": True,
                "success": False,
                "failure": False,
            }
            _verify_live_governed_baseline_v2(
                namespace,
                parent_authority_directory=parent,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                expected_presence=rollback_presence,
                expected_candidates=(),
                candidate_discoverer=candidate_discoverer,
            )
            first_candidates = _candidate_tuple(candidate_discoverer, parent)
            _stable_amendment_inventory(parent.parent)
            first_destination_absent = not os.path.lexists(destination)
            second_candidates = _candidate_tuple(candidate_discoverer, parent)
            _stable_amendment_inventory(parent.parent)
            second_destination_absent = not os.path.lexists(destination)
            if (
                first_candidates
                or second_candidates
                or not first_destination_absent
                or not second_destination_absent
            ):
                raise AmbiguousStateError(
                    "creator failed but Authority D may exist; F2 is forbidden"
                ) from creator_error
            rollback_at = _trusted_protocol_time(
                clock,
                role="replacement-v2 rollback disposition",
                not_before=failed_at,
            )
            failure = _build_failure_v2(
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                verifier=verifier,
                error=creator_error,
                failure_phase=failure_phase,
                failed_at=failed_at,
                rollback_checked_at=rollback_at,
            )
            _verify_live_governed_baseline_v2(
                namespace,
                parent_authority_directory=parent,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                expected_presence=rollback_presence,
                expected_candidates=(),
                candidate_discoverer=candidate_discoverer,
            )
            for lock in owned_locks:
                lock.assert_owned()
            failure_sha256 = _write_terminal_marker(
                namespace.failure_v2,
                failure,
                role="replacement-v2 failure marker",
            )
            repeated_failure, repeated_failure_sha256 = _read_failure_v2(
                namespace,
                attempt=attempt,
                attempt_sha256=attempt_sha256,
            )
            if repeated_failure != failure or repeated_failure_sha256 != failure_sha256:
                raise AmbiguousStateError(
                    "replacement-v2 F2 changed during terminal readback"
                ) from creator_error
            _verify_live_governed_baseline_v2(
                namespace,
                parent_authority_directory=parent,
                authorization=authorization,
                authorization_receipt_sha256=authorization_receipt_sha256,
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                expected_presence={
                    **rollback_presence,
                    "failure": True,
                },
                expected_candidates=(),
                candidate_discoverer=candidate_discoverer,
            )
            for lock in owned_locks:
                lock.assert_owned()
            result = PublicationResultV2(
                state=State.ROLLED_BACK_FAILURE,
                marker_path=namespace.failure_v2,
                marker_sha256=failure_sha256,
                authority_directory=None,
            )
        if result is None:  # pragma: no cover - defensive
            raise AmbiguousStateError(
                "replacement-v2 rollback returned no result"
            ) from creator_error
        return result
    if (
        not verifier.invoked
        or verifier.result is None
        or verifier.published_result is None
        or transaction_result is not verifier.published_result
    ):
        raise AmbiguousStateError("creator returned without exact transaction-scoped verification")
    returned_pins = api.authority_pins(transaction_result)
    verified_request = verifier.result.request
    expected_pins = AuthorityPins(
        directory=verified_request.successor_directory,
        parent_directory=verified_request.parent_directory,
        artifact_root_sha256=verified_request.artifact_root_sha256,
        sha256_manifest_sha256=verified_request.manifest_sha256,
        authorization_sha256=verified_request.authorization_sha256,
        intent_sha256=verified_request.intent_sha256,
        chain_depth=verified_request.chain_depth,
    )
    if returned_pins != expected_pins:
        raise AmbiguousStateError("creator result differs from fresh-verifier pins")
    committed_result: PublicationResultV2 | None = None
    with (
        _protocol_lock(
            namespace,
            parent=parent,
            role="resource Authority-D replacement-v2 commit disposition",
        ) as protocol_lock,
        ExclusiveBundlePublicationLock(
            (parent,),
            role="resource Authority-D replacement-v2 commit Authority-C parent guard",
        ) as parent_guard,
    ):
        owned_locks = (protocol_lock, parent_guard)
        _require_legacy_lock_state_under_protocol_lock(
            legacy_paths=legacy_lock_paths,
            owned_locks=owned_locks,
        )
        static_authorization, static_authorization_sha256 = _read_publication_authorization_v2(
            namespace, verify_live=False
        )
        static_attempt, static_attempt_sha256 = _read_attempt_v2(
            namespace,
            authorization=static_authorization,
            authorization_receipt_sha256=static_authorization_sha256,
            verify_live=False,
        )
        candidates = _candidate_tuple(candidate_discoverer, parent)
        if (
            static_authorization != authorization
            or static_authorization_sha256 != authorization_receipt_sha256
            or static_attempt != attempt
            or static_attempt_sha256 != attempt_sha256
            or candidates != (destination,)
            or _reserved_family_presence(namespace)
            != {
                "qualification": True,
                "inputs": True,
                "authorization": True,
                "attempt": True,
                "success": False,
                "failure": False,
            }
        ):
            raise AmbiguousStateError("replacement-v2 commit disposition inputs changed")
        commit_presence = {
            "qualification": True,
            "inputs": True,
            "authorization": True,
            "attempt": True,
            "success": False,
            "failure": False,
        }
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence=commit_presence,
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )
        _stable_amendment_inventory(parent.parent, allowed_extra=destination)
        api.verify_committed(
            destination,
            expected=expected_pins,
            replacement_publication_failure_lineage=terminal_lineage,
        )
        repeated_candidates = _candidate_tuple(candidate_discoverer, parent)
        _stable_amendment_inventory(parent.parent, allowed_extra=destination)
        repeated_attempt, repeated_attempt_sha256 = _read_attempt_v2(
            namespace,
            authorization=static_authorization,
            authorization_receipt_sha256=static_authorization_sha256,
            verify_live=False,
        )
        if (
            repeated_candidates != (destination,)
            or repeated_attempt != attempt
            or repeated_attempt_sha256 != attempt_sha256
            or _reserved_family_presence(namespace)
            != {
                "qualification": True,
                "inputs": True,
                "authorization": True,
                "attempt": True,
                "success": False,
                "failure": False,
            }
        ):
            raise AmbiguousStateError(
                "replacement-v2 commit inputs changed after static D verification"
            )
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence=commit_presence,
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )
        committed_at = _trusted_protocol_time(
            clock,
            role="replacement-v2 commit",
        )
        success = _build_success_v2(
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            verifier=verifier,
            committed_at=committed_at,
        )
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence=commit_presence,
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )
        for lock in owned_locks:
            lock.assert_owned()
        success_sha256 = _write_terminal_marker(
            namespace.success_v2,
            success,
            role="replacement-v2 success marker",
        )
        repeated_success, repeated_success_sha256 = _read_success_v2(
            namespace,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
        )
        if repeated_success != success or repeated_success_sha256 != success_sha256:
            raise AmbiguousStateError("replacement-v2 S2 changed during terminal readback")
        _verify_live_governed_baseline_v2(
            namespace,
            parent_authority_directory=parent,
            authorization=authorization,
            authorization_receipt_sha256=authorization_receipt_sha256,
            attempt=attempt,
            attempt_sha256=attempt_sha256,
            expected_presence={
                **commit_presence,
                "success": True,
            },
            expected_candidates=(destination,),
            candidate_discoverer=candidate_discoverer,
        )
        api.verify_committed(
            destination,
            expected=expected_pins,
            replacement_publication_failure_lineage=terminal_lineage,
        )
        for lock in owned_locks:
            lock.assert_owned()
        committed_result = PublicationResultV2(
            state=State.COMMITTED,
            marker_path=namespace.success_v2,
            marker_sha256=success_sha256,
            authority_directory=destination,
        )
    if committed_result is None:  # pragma: no cover - defensive
        raise AmbiguousStateError("replacement-v2 commit returned no result")
    del project
    return committed_result


def _publish_replacement_authority_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
    api: SchemaV3API | None,
    clock: Callable[[], datetime],
    popen_factory: Callable[..., Any],
) -> PublicationResultV2:
    project, parent = _require_parent(namespace, parent_authority_directory)
    authorization, authorization_sha256 = _read_publication_authorization_v2(
        namespace,
        verify_live=True,
    )
    timestamp = datetime.strptime(
        authorization["publication"]["amendment_timestamp_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    preflight = _build_live_preflight_v2(
        namespace=namespace,
        parent_authority_directory=parent,
        amendment_timestamp=timestamp,
        authorization_present=True,
    )
    if (
        preflight["contract"] != authorization["preflight"]["contract"]
        or preflight["preflight_fingerprint_sha256"]
        != authorization["preflight"]["preflight_fingerprint_sha256"]
    ):
        raise ControlError("replacement-v2 publication preflight changed")
    context = preflight["context"]
    terminal_lineage = authorization["preflight"]["contract"]["terminal_qualification"]
    technical = authorization["preflight"]["contract"]["technical_successor"]
    destination = _absolute(authorization["publication"]["intended_authority_directory"])
    selected_api: SchemaV3API = api or ProductionSchemaV3API(
        project_root=project,
        parent=parent,
        destination=destination,
        config_path=Path(authorization["preflight"]["contract"]["config"]["path"]),
        timestamp=timestamp,
        storage_policy=technical["storage_policy"],
        terminal_lineage=terminal_lineage,
        expected_authorization_sha256=technical["authorization_sha256"],
        expected_intent_sha256=technical["intent_sha256"],
    )
    canonical_authorization = selected_api.canonicalize_authorization(
        technical["authorization"],
        replacement_publication_failure_lineage=terminal_lineage,
    )
    if canonical_authorization != technical["authorization"]:
        raise ControlError("replacement-v2 schema-v3 API changed authorization")
    verifier = TransactionVerifierV2(
        api=selected_api,
        project_root=project,
        parent=parent,
        destination=destination,
        authorization_sha256=technical["authorization_sha256"],
        intent_sha256=technical["intent_sha256"],
        verification_nonce=_verification_nonce_v2(
            authorization,
            authorization_sha256,
        ),
        popen_factory=popen_factory,
    )
    del context
    return _execute_publication_v2_once(
        namespace=namespace,
        parent_authority_directory=parent,
        authorization=authorization,
        authorization_receipt_sha256=authorization_sha256,
        api=selected_api,
        verifier=verifier,
        clock=clock,
    )


def publish_replacement_authority_once(
    *,
    namespace: Namespace,
    parent_authority_directory: str | Path,
) -> PublicationResultV2:
    """Consume U2 in one permanent A2 -> exactly one F2 or D2+S2 transition."""

    return _publish_replacement_authority_once(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
        api=None,
        clock=lambda: datetime.now(UTC),
        popen_factory=subprocess.Popen,
    )


def _expected_state_from_presence(
    presence: Mapping[str, bool],
    candidate_count: int,
) -> State:
    expected_fields = (
        "qualification",
        "inputs",
        "authorization",
        "attempt",
        "success",
        "failure",
    )
    if (
        type(presence) is not dict
        or tuple(presence) != expected_fields
        or any(type(presence[field]) is not bool for field in expected_fields)
        or type(candidate_count) is not int
        or candidate_count < 0
    ):
        return State.STOP_AMBIGUOUS
    bits = tuple(presence[field] for field in expected_fields)
    table = {
        ((False, False, False, False, False, False), 0): (State.QUALIFICATION_REQUIRED),
        ((True, False, False, False, False, False), 0): (State.INPUT_FREEZE_REQUIRED),
        ((True, True, False, False, False, False), 0): (State.AUTHORIZATION_REQUIRED),
        ((True, True, True, False, False, False), 0): State.READY,
        ((True, True, True, True, False, True), 0): (State.ROLLED_BACK_FAILURE),
        ((True, True, True, True, True, False), 1): State.COMMITTED,
    }
    return table.get((bits, candidate_count), State.STOP_AMBIGUOUS)


def _verify_committed_candidate_v2(
    *,
    namespace: Namespace,
    candidate: Path,
    success: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    project, parent = _require_parent(
        namespace,
        success["parent_authority_directory"],
    )
    publication = authorization["publication"]
    technical = authorization["preflight"]["contract"]["technical_successor"]
    terminal_lineage = authorization["preflight"]["contract"]["terminal_qualification"]
    timestamp = datetime.strptime(
        publication["amendment_timestamp_utc"],
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=UTC)
    api = ProductionSchemaV3API(
        project_root=project,
        parent=parent,
        destination=candidate,
        config_path=Path(authorization["preflight"]["contract"]["config"]["path"]),
        timestamp=timestamp,
        storage_policy=technical["storage_policy"],
        terminal_lineage=terminal_lineage,
        expected_authorization_sha256=technical["authorization_sha256"],
        expected_intent_sha256=technical["intent_sha256"],
    )
    expected = AuthorityPins(
        directory=candidate,
        parent_directory=parent,
        artifact_root_sha256=success["artifact_root_sha256"],
        sha256_manifest_sha256=success["sha256_manifest_sha256"],
        authorization_sha256=success["authorization_sha256"],
        intent_sha256=success["intent_sha256"],
        chain_depth=success["chain_depth"],
    )
    api.verify_committed(
        candidate,
        expected=expected,
        replacement_publication_failure_lineage=terminal_lineage,
    )


def classify(
    namespace: Namespace,
    *,
    parent_authority_directory: str | Path,
    candidate_discoverer: Callable[
        [str | Path],
        Sequence[str | Path],
    ] = discover_candidates,
    committed_candidate_verifier: Callable[
        [Path, Mapping[str, Any]],
        None,
    ]
    | None = None,
) -> Classification:
    """Apply the closed Q/I/U/A/S/F + D truth table under shared guards."""

    candidates: tuple[Path, ...] = ()
    hashes: dict[str, str | None] = {
        "terminal_qualification_sha256": None,
        "input_v3_sha256": None,
        "authorization_v2_sha256": None,
        "attempt_v2_sha256": None,
        "success_v2_sha256": None,
        "failure_v2_sha256": None,
    }
    try:
        _project, parent = _require_parent(namespace, parent_authority_directory)
        legacy_lock_paths = _legacy_scoped_lock_paths(namespace, parent=parent)
        with (
            _protocol_lock(
                namespace,
                parent=parent,
                role="resource Authority-D replacement-v2 state classification",
            ) as protocol_lock,
            ExclusiveBundlePublicationLock(
                (parent,),
                role="resource Authority-D replacement-v2 classifier Authority-C parent guard",
            ) as parent_guard,
        ):
            owned_locks = (protocol_lock, parent_guard)
            _require_legacy_lock_state_under_protocol_lock(
                legacy_paths=legacy_lock_paths,
                owned_locks=owned_locks,
            )
            presence = _reserved_family_presence(namespace)
            candidates = _candidate_tuple(candidate_discoverer, parent)
            state = _expected_state_from_presence(presence, len(candidates))
            if state is State.STOP_AMBIGUOUS:
                return Classification(
                    state=state,
                    reason="state is outside the closed Q/I/U/A/S/F + D truth table",
                    candidates=candidates,
                    **hashes,
                )
            allowed_extra = candidates[0] if state is State.COMMITTED else None
            _stable_amendment_inventory(
                parent.parent,
                allowed_extra=allowed_extra,
            )
            terminal: dict[str, Any] | None = None
            input_root: str | None = None
            authorization: dict[str, Any] | None = None
            authorization_sha256: str | None = None
            attempt: dict[str, Any] | None = None
            attempt_sha256: str | None = None
            terminal_state = state in {
                State.ROLLED_BACK_FAILURE,
                State.COMMITTED,
            }
            if presence["qualification"]:
                terminal, terminal_sha256 = _read_terminal_qualification(
                    namespace,
                    pins=DEFAULT_HISTORICAL_PINS,
                    verify_live_history=not terminal_state,
                )
                hashes["terminal_qualification_sha256"] = terminal_sha256
            if presence["inputs"]:
                _input_payloads, _input_records, input_root = _read_input_v3(
                    namespace,
                    verify_live=not terminal_state,
                )
                hashes["input_v3_sha256"] = input_root
            if presence["authorization"]:
                authorization, authorization_sha256 = _read_publication_authorization_v2(
                    namespace,
                    verify_live=not terminal_state,
                )
                hashes["authorization_v2_sha256"] = authorization_sha256
            if presence["attempt"]:
                if authorization is None or authorization_sha256 is None:
                    raise ControlError("A2 exists without canonical U2")
                attempt, attempt_sha256 = _read_attempt_v2(
                    namespace,
                    authorization=authorization,
                    authorization_receipt_sha256=authorization_sha256,
                    verify_live=False,
                )
                hashes["attempt_v2_sha256"] = attempt_sha256
            if terminal_state:
                if (
                    authorization is None
                    or authorization_sha256 is None
                    or attempt is None
                    or attempt_sha256 is None
                ):
                    raise ControlError("terminal replacement-v2 state lacks exact U2/A2")
                _verify_live_governed_baseline_v2(
                    namespace,
                    parent_authority_directory=parent,
                    authorization=authorization,
                    authorization_receipt_sha256=authorization_sha256,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                    expected_presence=presence,
                    expected_candidates=candidates,
                    verify_live_run_state=state is not State.COMMITTED,
                    candidate_discoverer=candidate_discoverer,
                )
            if state is State.ROLLED_BACK_FAILURE:
                if attempt is None or attempt_sha256 is None:
                    raise ControlError("F2 exists without canonical A2")
                failure, failure_sha256 = _read_failure_v2(
                    namespace,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                )
                hashes["failure_v2_sha256"] = failure_sha256
                if candidates or os.path.lexists(Path(failure["intended_authority_directory"])):
                    raise ControlError("F2 exists while Authority D may exist")
            if state is State.COMMITTED:
                if attempt is None or attempt_sha256 is None or authorization is None:
                    raise ControlError("S2/D2 exist without canonical A2/U2")
                success, success_sha256 = _read_success_v2(
                    namespace,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                )
                hashes["success_v2_sha256"] = success_sha256
                if (
                    candidates != (_absolute(success["authority_directory"]),)
                    or success["authority_directory"] != attempt["intended_authority_directory"]
                ):
                    raise ControlError("S2 does not bind the exact singleton D2")
                if committed_candidate_verifier is None:
                    _verify_committed_candidate_v2(
                        namespace=namespace,
                        candidate=candidates[0],
                        success=success,
                        authorization=authorization,
                    )
                else:
                    committed_candidate_verifier(candidates[0], success)
            if state is State.ROLLED_BACK_FAILURE:
                assert (
                    authorization is not None
                    and authorization_sha256 is not None
                    and attempt is not None
                    and attempt_sha256 is not None
                )
                repeated_failure, repeated_failure_sha256 = _read_failure_v2(
                    namespace,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                )
                _verify_live_governed_baseline_v2(
                    namespace,
                    parent_authority_directory=parent,
                    authorization=authorization,
                    authorization_receipt_sha256=authorization_sha256,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                    expected_presence=presence,
                    expected_candidates=candidates,
                    verify_live_run_state=True,
                    candidate_discoverer=candidate_discoverer,
                )
                if repeated_failure != failure or repeated_failure_sha256 != failure_sha256:
                    raise AmbiguousStateError("replacement-v2 F2 changed during classification")
            if state is State.COMMITTED:
                assert (
                    authorization is not None
                    and authorization_sha256 is not None
                    and attempt is not None
                    and attempt_sha256 is not None
                )
                repeated_success, repeated_success_sha256 = _read_success_v2(
                    namespace,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                )
                _verify_live_governed_baseline_v2(
                    namespace,
                    parent_authority_directory=parent,
                    authorization=authorization,
                    authorization_receipt_sha256=authorization_sha256,
                    attempt=attempt,
                    attempt_sha256=attempt_sha256,
                    expected_presence=presence,
                    expected_candidates=candidates,
                    verify_live_run_state=False,
                    candidate_discoverer=candidate_discoverer,
                )
                if repeated_success != success or repeated_success_sha256 != success_sha256:
                    raise AmbiguousStateError("replacement-v2 S2 changed during classification")
                if committed_candidate_verifier is None:
                    _verify_committed_candidate_v2(
                        namespace=namespace,
                        candidate=candidates[0],
                        success=success,
                        authorization=authorization,
                    )
                else:
                    committed_candidate_verifier(candidates[0], success)
            repeated_presence = _reserved_family_presence(namespace)
            repeated_candidates = _candidate_tuple(candidate_discoverer, parent)
            if (
                repeated_presence != presence
                or repeated_candidates != candidates
                or _expected_state_from_presence(
                    repeated_presence,
                    len(repeated_candidates),
                )
                is not state
            ):
                raise AmbiguousStateError("replacement-v2 state changed during classification")
            for lock in owned_locks:
                lock.assert_owned()
            reasons = {
                State.QUALIFICATION_REQUIRED: ("terminal qualification Q has not been published"),
                State.INPUT_FREEZE_REQUIRED: ("exact live Q exists; input-v3 freeze I is required"),
                State.AUTHORIZATION_REQUIRED: (
                    "exact live Q+I exist; publication authorization U is required"
                ),
                State.READY: "exact live Q+I+U exist with no A/S/F/D",
                State.ROLLED_BACK_FAILURE: ("exact sealed Q+I+U+A+F exist and D is absent"),
                State.COMMITTED: ("exact sealed Q+I+U+A+S and one statically verified D exist"),
            }
            del terminal, input_root
            return Classification(
                state=state,
                reason=reasons[state],
                candidates=candidates,
                **hashes,
            )
    except BaseException as exc:
        return Classification(
            state=State.STOP_AMBIGUOUS,
            reason=f"{type(exc).__name__}: {exc}",
            candidates=candidates,
            **hashes,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--classify", action="store_true")
    mode.add_argument("--qualify-terminal", action="store_true")
    mode.add_argument("--freeze-inputs", action="store_true")
    mode.add_argument("--authorize-publication", action="store_true")
    mode.add_argument("--publish-once", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--parent-authority-dir", type=Path, required=True)
    return parser


def _cli_mutation_snapshot(
    *,
    namespace: Namespace,
    parent_authority_directory: Path,
) -> dict[str, Any] | None:
    """Capture names and existence only for fail-closed CLI disposition."""

    try:
        _project, parent = _require_parent(
            namespace,
            parent_authority_directory,
        )
        paths = {
            "terminal_qualification": namespace.terminal_qualification,
            "input_v3": namespace.input_v3,
            "authorization_v2": namespace.authorization_v2,
            "attempt_v2": namespace.attempt_v2,
            "success_v2": namespace.success_v2,
            "failure_v2": namespace.failure_v2,
        }
        return {
            "control_entries": {
                role: os.path.lexists(path) for role, path in sorted(paths.items())
            },
            "amendment_entries": tuple(sorted(entry.name for entry in os.scandir(parent.parent))),
        }
    except (OSError, TypeError, ValueError):
        return None


def _cli_exception_disposition(
    *,
    namespace: Namespace | None,
    parent_authority_directory: Path | None,
    baseline: Mapping[str, Any] | None,
) -> tuple[str, bool | None, str | None, int]:
    if namespace is None or parent_authority_directory is None:
        return "stopped_without_write", False, None, 1
    current = _cli_mutation_snapshot(
        namespace=namespace,
        parent_authority_directory=parent_authority_directory,
    )
    changed = (
        baseline is None
        or current is None
        or _canonical_bytes(baseline) != _canonical_bytes(current)
    )
    observed = classify(
        namespace,
        parent_authority_directory=parent_authority_directory,
    )
    if observed.state is State.COMMITTED:
        return "stopped_after_attempt", True, observed.state.value, 3
    if observed.state is State.ROLLED_BACK_FAILURE:
        return "stopped_after_control_write", False, observed.state.value, 3
    if observed.state is State.STOP_AMBIGUOUS:
        return "stopped_ambiguous", None, observed.state.value, 3
    if changed:
        return "stopped_after_control_write", False, observed.state.value, 3
    return "stopped_without_write", False, observed.state.value, 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    namespace: Namespace | None = None
    parent: Path | None = None
    mutation_baseline: dict[str, Any] | None = None
    try:
        project = _absolute(args.project_root)
        parent = args.parent_authority_dir
        if not parent.is_absolute():
            parent = project / parent
        parent = _absolute(parent)
        namespace = Namespace.for_project(project)
        mutation_baseline = _cli_mutation_snapshot(
            namespace=namespace,
            parent_authority_directory=parent,
        )
        if args.classify:
            result = classify(
                namespace,
                parent_authority_directory=parent,
            )
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            return 1 if result.state is State.STOP_AMBIGUOUS else 0
        if args.qualify_terminal:
            receipt, digest = qualify_historical_terminal_once(
                namespace=namespace,
                parent_authority_directory=parent,
            )
            output = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": receipt["status"],
                "terminal_qualification_receipt_path": str(namespace.terminal_qualification),
                "terminal_qualification_receipt_sha256": digest,
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            }
        elif args.freeze_inputs:
            _payloads, records, records_sha256 = freeze_input_v3_once(
                namespace=namespace,
                parent_authority_directory=parent,
            )
            output = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "input_v3_frozen",
                "input_v3_directory": str(namespace.input_v3),
                "input_v3_file_count": len(records),
                "input_v3_records_sha256": records_sha256,
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            }
        elif args.authorize_publication:
            authorization, digest = authorize_publication_v2_once(
                namespace=namespace,
                parent_authority_directory=parent,
            )
            output = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": authorization["status"],
                "publication_authorization_v2_path": str(namespace.authorization_v2),
                "publication_authorization_v2_sha256": digest,
                "authorized_attempt_id": authorization["authorized_attempt_id"],
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            }
        elif args.publish_once:
            publication = publish_replacement_authority_once(
                namespace=namespace,
                parent_authority_directory=parent,
            )
            output = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "state": publication.state.value,
                "terminal_marker_path": str(publication.marker_path),
                "terminal_marker_sha256": publication.marker_sha256,
                "authority_directory": (
                    None
                    if publication.authority_directory is None
                    else str(publication.authority_directory)
                ),
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": publication.state is State.COMMITTED,
            }
            print(json.dumps(output, indent=2, sort_keys=True))
            return 0 if publication.state is State.COMMITTED else 3
        else:  # pragma: no cover - argparse makes this unreachable
            raise ControlError("replacement-v2 CLI mode is unreachable")
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        status, publication_performed, state, exit_code = _cli_exception_disposition(
            namespace=namespace,
            parent_authority_directory=parent,
            baseline=mutation_baseline,
        )
        print(
            json.dumps(
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "status": status,
                    "replacement_state": state,
                    "automatic_retry_allowed": False,
                    "publication_performed": publication_performed,
                    "outcome_value_interpretation_performed": False,
                    "scientific_execution_performed": False,
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
