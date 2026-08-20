"""Immutable, hash-linked successor authorities for preregistration amendments.

The base preregistration freeze remains byte-for-byte immutable. Every amendment is
published as a complete, independently sealed sibling bundle that names and
cryptographically authenticates its parent authority. A successor contains full
preregistration, primary-config, confirmatory-config, and execution-source snapshots;
it is never a delta that must be applied to mutable live files.

Schema-v2 amendment evidence may additionally authorize one finalization-only
successor from an exact, terminally failed, sealed predecessor. The authorization
binds the predecessor seal, computation source, registration authority, and
all-or-nothing reuse counts while fixing both outcome inspection and retraining to
false/zero. Its live boundary requires complete technical integrity and the exact
append-only integrity-registry record, but never interprets outcome values or makes
the failed predecessor eligible by itself. The same schema may bind the single-copy
confirmatory checkpoint-storage policy as a closed, constant-valued block; this
storage-only authority is valid only alongside the finalization authorization.

Schema-v3 resource technical-successor authorization remains backward-compatible
with historical schema-v2 readers while additionally binding one canonical,
read-only terminal-qualification receipt for the consumed replacement-publication
failure. The receipt distinguishes the historical controller, its later diagnostic
fix, and the new qualifying controller; it authorizes neither retry of the consumed
attempt nor publication by itself.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from histo_audit.config import config_sha256, load_config
from histo_audit.pannuke.publication import (
    ExclusiveBundlePublicationLock,
    PublishedPath,
    publish_bytes_no_overwrite,
    publish_flat_directory_physical_copy_no_overwrite,
    read_file_anchored,
    rollback_owned_publications,
)
from histo_audit.utils.run_tracking import (
    ARTIFACT_MANIFEST_FILENAME as RUN_ARTIFACT_MANIFEST_FILENAME,
)
from histo_audit.utils.run_tracking import (
    IMMUTABLE_MARKER as RUN_IMMUTABLE_MARKER,
)
from histo_audit.utils.run_tracking import (
    SOURCE_TREE_MANIFEST_FILENAME as RUN_SOURCE_TREE_MANIFEST_FILENAME,
)
from histo_audit.utils.run_tracking import STATUS_FILENAME as RUN_STATUS_FILENAME
from histo_audit.utils.run_tracking import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    capture_source_tree,
    require_run_stage_eligibility_receipt,
    sha256_file,
    sha256_path,
    verify_run_integrity,
)
from histo_audit.workflows.preregistration import verify_preregistration_freeze

if TYPE_CHECKING:
    from histo_audit.experiment.confirmatory_memory_workspace import (
        ConfirmatoryWorkspaceArraySpec,
        ConfirmatoryWorkspaceIndexSpec,
    )
    from histo_audit.experiment.primary_statistics import (
        AuthorizedPriorNumericVerificationProof,
    )

AuthorityKind = Literal["base_freeze", "preregistration_amendment"]

_AUTHORITY_KIND = "preregistration_amendment"
_IMMUTABLE_MARKER = ".immutable.json"
_MANIFEST_FILENAME = "sha256_manifest.json"
_EVIDENCE_FILENAME = "amendment_evidence.json"
_PREREGISTRATION_SNAPSHOT = "PRE_REGISTRATION_FROZEN.md"
_PRIMARY_CONFIG_SNAPSHOT = "primary_frozen.yaml"
_CONFIRMATORY_CONFIG_SNAPSHOT = "confirmatory_frozen.yaml"
_SOURCE_TREE_SNAPSHOT = "source_tree_manifest.json"
_REPORT_FILENAME = "AMENDMENT.md"
_EXACT_AMENDMENT_BUNDLE_FILENAMES = frozenset(
    {
        _IMMUTABLE_MARKER,
        _MANIFEST_FILENAME,
        _EVIDENCE_FILENAME,
        _PREREGISTRATION_SNAPSHOT,
        _PRIMARY_CONFIG_SNAPSHOT,
        _CONFIRMATORY_CONFIG_SNAPSHOT,
        _SOURCE_TREE_SNAPSHOT,
        _REPORT_FILENAME,
    }
)
_POST_PUBLICATION_CHECK_ACTIVE: ContextVar[bool] = ContextVar(
    "histo_audit_post_publication_check_active",
    default=False,
)
_AMENDMENT_CREATION_PROCESS_GUARD = Lock()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_EXECUTION_SCOPE = ["src/**", "configs/**", "pyproject.toml", "uv.lock"]
_EXECUTION_EXCLUDED_ROOTS = [".git", ".venv", "artifacts", "data"]
_EXECUTION_EXCLUDED_PATHS = [
    "configs/confirmatory_frozen.yaml",
    "configs/primary_frozen.yaml",
]
_MAX_CHAIN_DEPTH = 64
_MAX_RESOURCE_CONTROL_RECEIPT_BYTES = 1024 * 1024
_FINALIZATION_SUCCESSOR_POLICY = "sealed_failed_primary_finalization_successor_v1"
_FINALIZATION_SUCCESSOR_POLICY_V2 = "sealed_failed_primary_finalization_successor_v2"
_FINALIZATION_SUCCESSOR_EVIDENCE_FILENAME = "primary_finalization_successor_evidence.json"
_FINALIZATION_SUCCESSOR_COMPLETION_FLAG = "finalization_only_successor"
_FINALIZATION_SELECTION_POLICY = "all_declared_cells_exact_set_no_outcome_selection"
_FINALIZATION_ACCESS_POLICY = "read_only_checksum_verified"
_FINALIZATION_OUTPUT_POLICY = "new_run_directory_retry_of_predecessor_no_mutation"
INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE = "inherited_prior_numeric_verification_v1"
_INHERITED_PRIOR_NUMERIC_POLICY = "trusted_double_verifier_terminal_failure_proof_v1"
_INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION = (
    "trusted_local_process_no_dependency_injection_import_hook_hotpatch_or_concurrent_writer"
)
_INHERITED_PRIOR_NUMERIC_LIMITATION = (
    "control_flow_and_content_addressed_inheritance_not_fresh_semantic_recomputation"
)
_TRUSTED_PROCESS_OBSERVATION_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "process_observations"
    / "20260726T185709.6378333Z_pid20792_cli_receipt.json"
)
_TRUSTED_PROCESS_OBSERVATION_SIZE_BYTES = 4138
_TRUSTED_PROCESS_OBSERVATION_SHA256 = (
    "28767a0b1a14d28aa5d055484c9de2ec0d8fad728abad4f797f67af99b24c151"
)
_CONFIRMATORY_STORAGE_POLICY = "single_canonical_checkpoint_copy_v1"
_CONFIRMATORY_STORAGE_SCOPE = "one_checkpoint_per_completed_cnn_cell_oof_fold"
_CONFIRMATORY_STORAGE_PATH_TEMPLATE = "cells/{cell_id}/checkpoints/fold_{fold_id:02d}.pt"
_CONFIRMATORY_STORAGE_LINK_POLICY = "regular_file_no_symlink_no_junction_no_hardlink"
_CONFIRMATORY_STORAGE_VERIFICATION_POLICY = (
    "size_sha256_fold_evidence_checkpoint_manifest_postseal_exact_set"
)
_CONFIRMATORY_STORAGE_SCIENTIFIC_EFFECT = (
    "storage_only_no_model_data_split_seed_prediction_metric_restoration_or_estimand_change"
)
_PRIMARY_RECOVERY_POLICY = "interrupted_unsealed_primary_recovery_v1"
_PRIMARY_RECOVERY_INTERRUPTION_KIND = "host_reboot"
_PRIMARY_RECOVERY_ANALYSIS_DISPOSITION = "amended_or_exploratory"
_PRIMARY_RECOVERY_EVENT_ID = 12
RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE = "resource_bounded_confirmatory_execution"
_RESOURCE_BOUNDED_CONFIRMATORY_POLICY = "post_outcome_resource_bounded_confirmatory_execution_v1"
_RESOURCE_BOUNDED_SOURCE_DELTA_POLICY = "exact_nonempty_resource_execution_source_delta_v1"
_RESOURCE_BOUNDED_OUTCOME_USE_POLICY = (
    "resource_constraints_only_no_outcome_value_selection_tuning_or_exclusion"
)
RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE = "resource_bounded_confirmatory_technical_successor"
_RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY = (
    "post_outcome_resource_bounded_confirmatory_technical_successor_v2"
)
_RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY_V3 = (
    "post_outcome_resource_bounded_confirmatory_technical_successor_v3"
)
_RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_POLICY = (
    "resource_authority_d_replacement_v1_terminal_qualification_v1"
)
_RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_FILENAME = (
    "resource_authority_d_replacement_v1_terminal_qualification_v1.json"
)
_RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_COMPONENT = "20260727T170413.080954Z"
_RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_ARTIFACT_ROOT_SHA256 = (
    "57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627"
)
_RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_MANIFEST_SHA256 = (
    "4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156"
)
_RESOURCE_BOUNDED_REPLACEMENT_AUTHORIZATION_SHA256 = (
    "4c892f7e518964a46569290e1a486d7f7e193121ed870522895946413dbee565"
)
_RESOURCE_BOUNDED_REPLACEMENT_ATTEMPT_SHA256 = (
    "e602993753949ecbd5bfe3dfd9ba77d1890d63ae6232db9db6d66caff48e3ace"
)
_RESOURCE_BOUNDED_REPLACEMENT_FAILURE_SHA256 = (
    "e66305dac9a2c1b59d5cb554081470c1947b939d8a07ade3cf77046f0e353b12"
)
_RESOURCE_BOUNDED_REPLACEMENT_TECHNICAL_AUTHORIZATION_SHA256 = (
    "886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8"
)
_RESOURCE_BOUNDED_REPLACEMENT_INTENT_SHA256 = (
    "9c5018d37a4a9f4d26dd40d1b4c3eb902c97601459720085ae12bc91d4e4e347"
)
_RESOURCE_BOUNDED_REPLACEMENT_PREFLIGHT_FINGERPRINT_SHA256 = (
    "9e828dd7652a2be3c3ecee798fae9f7b1b1167875129e7c3fed581443550270a"
)
_RESOURCE_BOUNDED_REPLACEMENT_ATTEMPT_ID = (
    "c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86"
)
_RESOURCE_BOUNDED_REPLACEMENT_TIMESTAMP = "2026-07-28T18:19:20.303224Z"
_RESOURCE_BOUNDED_REPLACEMENT_ERROR_TYPE_SHA256 = (
    "431fd4d500d504c9f02a7e5f505eb2065cf1612fc36b6474af72b89e5d3a8ffd"
)
_RESOURCE_BOUNDED_REPLACEMENT_ERROR_SHA256 = (
    "a6a9e199b4080a911e7f07f3243e3982049a85fc18ba712883de9f9cc099e1fb"
)
_RESOURCE_BOUNDED_REPLACEMENT_CONSUMED_CONTROLLER_SHA256 = (
    "cbea3c3536dbad729383c96e0ef602042c7e3c4e000f9b0cb79e50c13b2ced58"
)
_RESOURCE_BOUNDED_REPLACEMENT_DIAGNOSED_CONTROLLER_SHA256 = (
    "e20278105b6ea4e2786713c64d9e8cf7bb06d9e4c8155f35a46861e72cb67b5f"
)
_RESOURCE_BOUNDED_REPLACEMENT_RUN_STATE_SHA256 = (
    "5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4"
)
_RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_ROOT_SHA256 = (
    "c81e7a01bc6949d82d5cb76a206776dde4ceda47c1506a71bd8edf736649bd75"
)
_RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_MANIFEST_SHA256 = (
    "49857657c88249999278b4d9f51fa70cc622b9c23fb1e16a800c7e7f9e8a1a0f"
)
_RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_DELTA_SHA256 = (
    "82acb5a60100141a2c54f2094b8a438fd725bcfe4227c132e4dff185608d7217"
)
_RESOURCE_BOUNDED_REPLACEMENT_V2_RECORDS_SHA256 = (
    "8de462490b465d639badefe4cfd411773c4071cb028929e8b0faf73755333aa1"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_POLICY = (
    "resource_authority_d_prior_publication_failure_v1"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_ATTEMPT_SHA256 = (
    "8c93e65eca0bb4d64af4e94012004d74178448941cc746675d0e8e72ac5e90e2"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_SHA256 = (
    "de123683a56ab0349c44536e969f536843ed0c557bae8573187664cab7fc8615"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_STDOUT_SHA256 = (
    "c2ff6925d7d3339ebd2cd399074470f7308a804dc86099cd2d34e9f77e644c2d"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_STDERR_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_CONTROLLER_SHA256 = (
    "1b5337301bcf39307c1ec41e0fd0d2a26ffe2fbb18def824dc1868cc46234d37"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_INVENTORY_SHA256 = (
    "beba860f18ef656c5f9a3e874b41d1ae9a55919fcce78213317d256c6775e76d"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_RUN_STATE_SHA256 = (
    "5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_TYPE_SHA256 = (
    "16a21d770d950531feebcfba139f8da7c2bb758dfaccfc3b8b0cecf5aef41d7e"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_SHA256 = (
    "1ddb48209c3cca372cc2053492cd41d8fec3fe5e04840250634e3d5f48d49f33"
)
_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_FILES = {
    "authority_d_cnn_correction_receipt.json": (
        1814,
        "89c6d475d691b480478b76d25e5e96653a3d225d738ce23c6726a5bab409e6c3",
    ),
    "authority_d_frozen_source_receipt.json": (
        1006,
        "f5bd9384ac22be05b53e5b7fa987a059c84f74051067f47a7d30b14789e01c08",
    ),
    "authority_d_source_allowlist.json": (
        3444,
        "d6436c55e3134807ee0eb99d7e3b5c0a0416b06c1ced22372e31ba2ce268f176",
    ),
    "authority_d_workspace_plan.json": (
        12186,
        "d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b",
    ),
}
_RESOURCE_BOUNDED_TECHNICAL_SOURCE_DELTA_POLICY = (
    "explicit_exact_resource_technical_source_delta_v1"
)
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_POLICY = "resource_bounded_failed_preflight_no_tracker_v1"
_RESOURCE_BOUNDED_CNN_PROVENANCE_CORRECTION_POLICY = (
    "resource_bounded_cnn_logical_provenance_correction_v1"
)
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_OPERATION = "resource_bounded_sensitivity_preflight"
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_TYPE = "ValueError"
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_MESSAGE = (
    "CNN logical encoder/preprocessing provenance does not recompute from "
    "the verified crop view: cnn_context_rgb"
)
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_STATUS_ERROR = (
    f"{_RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_TYPE}: "
    f"{_RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_MESSAGE}"
)
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_INVOKED_AT = "2026-07-27T17:18:45.990Z"
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_OBSERVED_AT = "2026-07-27T17:30:54.689Z"
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_INVOCATION_CELL = 359
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_EXIT_CODE = 1
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_WALL_SECONDS = 728.7
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID = "cnn_context_rgb"
_RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID = "cnn_context_rgb_cache"
RESOURCE_BOUNDED_CAPACITY_POLICY_V2 = "resource_bounded_confirmatory_capacity_v2"
RESOURCE_BOUNDED_CAPACITY_POLICY_V3 = "resource_bounded_confirmatory_capacity_v3"
_RESOURCE_BOUNDED_CAPACITY_POLICY = RESOURCE_BOUNDED_CAPACITY_POLICY_V2
_RESOURCE_BOUNDED_CAPACITY: dict[str, Any] = {
    "schema_version": 2,
    "policy": _RESOURCE_BOUNDED_CAPACITY_POLICY,
    "planned_required_cells": 24,
    "planned_cnn_cells": 6,
    "planned_cnn_fold_checkpoints": 30,
    "max_epochs": 4,
    "projected_stable_run_bytes": 12_884_901_888,
    "fixed_safety_margin_bytes": 10_737_418_240,
    "minimum_free_bytes_before_tracker": 23_622_320_128,
    "max_active_atomic_temp_checkpoints": 1,
    "minimum_total_ram_bytes": 32_212_254_720,
    "minimum_available_ram_bytes_before_data": 17_179_869_184,
    "minimum_available_ram_bytes_before_tracker": 12_884_901_888,
    "cuda_required": True,
    "cuda_device_index": 0,
    "minimum_total_vram_bytes": 10_737_418_240,
    "minimum_free_vram_bytes": 8_589_934_592,
    "cudnn_required": True,
    "amp_required": True,
    "amp_dtype": "float16",
    "cuda_smoke_input_shape": [1, 3, 224, 224],
    "cuda_smoke_forward_backward_required": True,
    "cuda_smoke_finite_required": True,
    "cuda_smoke_max_peak_allocated_bytes": 536_870_912,
    "official_weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
    "official_weight_sha256": ("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "implicit_weight_download_allowed": False,
}
_RESOURCE_BOUNDED_CAPACITY_V3: dict[str, Any] = {
    **_RESOURCE_BOUNDED_CAPACITY,
    "schema_version": 3,
    "policy": RESOURCE_BOUNDED_CAPACITY_POLICY_V3,
    "workspace_layout_policy": (
        "checksum_bound_native_memmap_source_with_index_only_partitions_v1"
    ),
    "workspace_source_array_count": 12,
    "workspace_crop_npy_bytes": 3_128_777_281,
    "workspace_native_feature_npy_bytes": 1_165_404_988,
    "workspace_shared_backing_bytes": 4_294_182_269,
    "workspace_partition_count": 9,
    "workspace_partition_index_dtype": "int64",
    "workspace_partition_index_bytes": 4_521_144,
    "projected_workspace_bytes": 4_298_703_413,
    "workspace_nonpayload_headroom_bytes": 268_435_456,
    "maximum_workspace_bytes": 4_567_138_869,
    "maximum_active_workspace_builds": 1,
    "workspace_extra_disk_scratch_bytes": 0,
    "workspace_feature_backing_dtype": "float32",
    "workspace_feature_logical_exposure_dtype": "float64",
    "workspace_cleanup_policy": ("owned_ephemeral_staging_and_sealed_workspace_cleanup_v1"),
    "workspace_reuse_allowed": False,
    "minimum_free_bytes_before_workspace_build": 28_189_458_997,
}
_RESOURCE_BOUNDED_EXPERIMENT_NAME = "pannuke_resource_bounded_confirmatory"
_RESOURCE_BOUNDED_PRIMARY_EXPERIMENT_NAME = "pannuke_primary_orphan_recovery"
_RESOURCE_BOUNDED_PROFILE_ID = "resource_bounded_confirmatory_v1"
RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256 = (
    "1c9a41b92dabbeafbb92b1bc8aced158337046fc1d6e056b011f6a27b98e8298"
)
_RESOURCE_BOUNDED_CONFIG_SEMANTIC_SHA256 = RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256
_RESOURCE_BOUNDED_COMPLETION_STAGE: None = None
_RESOURCE_BOUNDED_PROFILE_SHAPE = {
    "planned_required_cells": 24,
    "planned_cnn_cells": 6,
    "planned_cnn_fold_checkpoints": 30,
}
_RESOURCE_BOUNDED_CNN_PROVENANCE_FIELDS = {
    "id",
    "representation_id",
    "status",
    "cache_file_sha256",
    "sidecar_semantic_sha256",
    "sample_order_sha256",
    "manifest_sha256",
    "encoder_identifier",
    "encoder_metadata_sha256",
    "weight_identifier",
    "weights_sha256",
    "preprocessing_identifier",
    "preprocessing_sha256",
    "input_variant",
}
_RESOURCE_BOUNDED_CNN_CONFIG_CORRECTION_FIELDS = {
    "encoder_metadata_sha256",
    "preprocessing_sha256",
}
_RESOURCE_BOUNDED_SOURCE_DELTA_KINDS = {
    "configs/confirmatory_resource_bounded_amended.yaml": "added",
    "src/histo_audit/cli.py": "modified",
    "src/histo_audit/experiment/__init__.py": "modified",
    "src/histo_audit/experiment/confirmatory_cli_inputs.py": "added",
    "src/histo_audit/experiment/confirmatory_completion.py": "modified",
    "src/histo_audit/experiment/confirmatory_core.py": "modified",
    "src/histo_audit/experiment/confirmatory_runner.py": "modified",
    "src/histo_audit/experiment/resource_bounded_resume.py": "added",
    "src/histo_audit/experiment/resource_bounded_runner.py": "added",
    "src/histo_audit/experiment/study_contracts.py": "modified",
    "src/histo_audit/models/cnn.py": "modified",
    "src/histo_audit/workflows/__init__.py": "modified",
    "src/histo_audit/workflows/lifecycle_qualification.py": "added",
    "src/histo_audit/workflows/preregistration_amendment.py": "modified",
    "src/histo_audit/workflows/study_gates.py": "modified",
}


@dataclass(frozen=True, slots=True)
class ConfirmatoryStoragePolicy:
    """Closed single-copy checkpoint-storage authority.

    The type accepts no caller-controlled fields. This makes the CLI's constant flag
    produce exactly one policy while the canonical verifier remains fail-closed for
    mappings loaded from sealed JSON evidence.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": _CONFIRMATORY_STORAGE_POLICY,
            "scope": _CONFIRMATORY_STORAGE_SCOPE,
            "canonical_relative_path_template": _CONFIRMATORY_STORAGE_PATH_TEMPLATE,
            "retained_copy_count": 1,
            "link_policy": _CONFIRMATORY_STORAGE_LINK_POLICY,
            "verification_policy": _CONFIRMATORY_STORAGE_VERIFICATION_POLICY,
            "scientific_effect": _CONFIRMATORY_STORAGE_SCIENTIFIC_EFFECT,
        }


def _trusted_prior_numeric_source_signature_base() -> dict[str, Any]:
    """Return the immutable code/control-flow signature eligible for B-fast."""

    return {
        "schema_version": 1,
        "signature_id": "pannuke_primary_20260719_missing_statistics_attestation_v1",
        "predecessor_run_id": (
            "20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
        ),
        "source_tree_root_sha256": (
            "c0850f54e88483c1df76a4c8836343f667a7a1adbf2d05d571990cd6119cf532"
        ),
        "source_tree_manifest_sha256": (
            "9a4afcad586ba445ec41a3a2413ae8f8efa6a422c22e6d148d7bb6a4a2014307"
        ),
        "files": [
            {
                "path": "src/histo_audit/experiment/primary_completion.py",
                "size_bytes": 84226,
                "sha256": ("81670eadbee7f8ea4902c8c4a3cfdc778fcd64a98b449c48a1f06ace9b2968c8"),
            },
            {
                "path": "src/histo_audit/experiment/primary_runner.py",
                "size_bytes": 38900,
                "sha256": ("5c0722d59714596d6a2d0dacc0522d6d708e44184c6a238378fcd65e9177efd6"),
            },
            {
                "path": "src/histo_audit/experiment/primary_statistics.py",
                "size_bytes": 54023,
                "sha256": ("3ac82d6d0f7b5aa9219b31dc506a6075400bd08c38c6da30bbaa1d4c469dc929"),
            },
        ],
        "control_flow": {
            "statistics_aggregator_call_line": 736,
            "aggregator_internal_verifier_call_line": 1216,
            "statistics_verifier_call_line": 740,
            "completion_builder_call_line": 767,
            "failure_raise_line": 1589,
            "expected_comparison_count": 36,
            "expected_error_type": "ValueError",
            "expected_error_message": (
                "eligible primary completion requires passed attested primary "
                "statistics verification"
            ),
            "ordered_traceback_frames": [
                {
                    "path_suffix": "src/histo_audit/experiment/primary_runner.py",
                    "line": 767,
                    "function": "execute_primary_study",
                },
                {
                    "path_suffix": "src/histo_audit/experiment/primary_completion.py",
                    "line": 1589,
                    "function": "build_primary_completion_evidence",
                },
            ],
            "semantic_proof": (
                "aggregate returned after its internal numeric verifier; the explicit "
                "numeric verifier returned; strict filesystem readback returned; only then "
                "the completion builder raised because the caller omitted the statistics "
                "attestation argument"
            ),
        },
    }


def _canonical_embedded_process_observation_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "encoding",
        "size_bytes",
        "sha256",
        "evidence_scope",
        "payload_base64",
    }:
        raise _AuthorityValidationError(
            "embedded process-observation receipt has an invalid field set"
        )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("encoding") != "base64"
        or type(value.get("size_bytes")) is not int
        or value.get("size_bytes") != _TRUSTED_PROCESS_OBSERVATION_SIZE_BYTES
        or value.get("sha256") != _TRUSTED_PROCESS_OBSERVATION_SHA256
        or value.get("evidence_scope")
        != "corroborating_trusted_process_observation_not_fresh_replay"
        or not isinstance(value.get("payload_base64"), str)
    ):
        raise _AuthorityValidationError("embedded process-observation receipt is invalid")
    encoded = str(value["payload_base64"])
    try:
        payload_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _AuthorityValidationError(
            "embedded process-observation receipt is not canonical base64"
        ) from exc
    if (
        base64.b64encode(payload_bytes).decode("ascii") != encoded
        or len(payload_bytes) != _TRUSTED_PROCESS_OBSERVATION_SIZE_BYTES
        or hashlib.sha256(payload_bytes).hexdigest() != _TRUSTED_PROCESS_OBSERVATION_SHA256
    ):
        raise _AuthorityValidationError(
            "embedded process-observation receipt differs from its fixed bytes"
        )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            "embedded process-observation receipt payload is invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "limitations",
        "observation_kind",
        "observation_timestamp_utc",
        "parent_process",
        "process",
        "run",
        "schema_version",
        "trust_model",
    }:
        raise _AuthorityValidationError(
            "embedded process-observation receipt payload has an invalid schema"
        )
    run = payload.get("run")
    process = payload.get("process")
    parent_process = payload.get("parent_process")
    limitations = payload.get("limitations")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("observation_kind") != "live_windows_process_cli_receipt_v1"
        or payload.get("trust_model") != _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
        or not isinstance(run, Mapping)
        or run.get("run_id")
        != "20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
        or run.get("source_tree_root_sha256")
        != "c0850f54e88483c1df76a4c8836343f667a7a1adbf2d05d571990cd6119cf532"
        or run.get("terminal_artifacts_present") is not False
        or not isinstance(process, Mapping)
        or not isinstance(parent_process, Mapping)
        or " -m histo_audit experiment primary " not in str(process.get("command_line"))
        or " -m histo_audit experiment primary " not in str(parent_process.get("command_line"))
        or not isinstance(limitations, list)
        or "receipt is corroborating operational evidence, not a fresh semantic statistics recomputation"
        not in limitations
    ):
        raise _AuthorityValidationError(
            "embedded process-observation receipt does not prove the trusted CLI observation"
        )
    return {
        "schema_version": 1,
        "encoding": "base64",
        "size_bytes": _TRUSTED_PROCESS_OBSERVATION_SIZE_BYTES,
        "sha256": _TRUSTED_PROCESS_OBSERVATION_SHA256,
        "evidence_scope": "corroborating_trusted_process_observation_not_fresh_replay",
        "payload_base64": encoded,
    }


def _trusted_prior_numeric_source_signature() -> dict[str, Any]:
    """Load the contemporaneous receipt once and embed its exact bytes in the authority."""

    receipt_bytes = _require_regular_file(
        _TRUSTED_PROCESS_OBSERVATION_SOURCE,
        role="trusted process observation receipt",
    )
    receipt = _canonical_embedded_process_observation_receipt(
        {
            "schema_version": 1,
            "encoding": "base64",
            "size_bytes": len(receipt_bytes),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "evidence_scope": "corroborating_trusted_process_observation_not_fresh_replay",
            "payload_base64": base64.b64encode(receipt_bytes).decode("ascii"),
        }
    )
    return {
        **_trusted_prior_numeric_source_signature_base(),
        "process_observation_receipt": receipt,
    }


def _canonical_trusted_prior_numeric_source_signature(value: Any) -> dict[str, Any]:
    expected = _trusted_prior_numeric_source_signature_base()
    if not isinstance(value, Mapping) or set(value) != {
        *expected,
        "process_observation_receipt",
    }:
        raise _AuthorityValidationError("trusted prior-numeric source signature is invalid")
    for key, expected_value in expected.items():
        observed = value.get(key)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise _AuthorityValidationError(
                f"trusted prior-numeric source signature field {key!r} differs"
            )
    return {
        **expected,
        "process_observation_receipt": _canonical_embedded_process_observation_receipt(
            value["process_observation_receipt"]
        ),
    }


@dataclass(frozen=True, slots=True)
class InheritedPriorNumericVerificationAuthorization:
    """Closed authorization block derived from one exact sealed terminal failure."""

    trusted_source_signature: Mapping[str, Any]
    terminal_evidence: Mapping[str, Any]
    prior_numeric_verification_proof_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
            "policy": _INHERITED_PRIOR_NUMERIC_POLICY,
            "trust_assumption": _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
            "limitation": _INHERITED_PRIOR_NUMERIC_LIMITATION,
            "trusted_source_signature": dict(self.trusted_source_signature),
            "terminal_evidence": dict(self.terminal_evidence),
            "prior_numeric_verification_proof_sha256": (
                self.prior_numeric_verification_proof_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class FinalizationSuccessorAuthorization:
    """Exact inputs for one outcome-blind finalization-only successor authority.

    Terminal status, outcome inspection, retraining, and reuse policies are constants
    rather than caller-controlled fields. This keeps CLI integration narrow and makes
    an accidental completed-run, post-outcome, or retraining authorization
    unrepresentable through this typed API.
    """

    predecessor_run_id: str
    predecessor_run_directory: Path
    predecessor_artifact_root_sha256: str
    predecessor_artifact_manifest_sha256: str
    predecessor_execution_source_root_sha256: str
    predecessor_execution_source_manifest_sha256: str
    predecessor_registration_authority_directory: Path
    predecessor_registration_authority_kind: AuthorityKind
    predecessor_registration_authority_artifact_root_sha256: str
    predecessor_registration_authority_manifest_sha256: str
    reused_required_cell_count: int
    reused_optional_cell_count: int
    numeric_verification: InheritedPriorNumericVerificationAuthorization | None = None

    def as_dict(self) -> dict[str, Any]:
        is_inherited = self.numeric_verification is not None
        return {
            "schema_version": 2 if is_inherited else 1,
            "policy": (
                _FINALIZATION_SUCCESSOR_POLICY_V2
                if is_inherited
                else _FINALIZATION_SUCCESSOR_POLICY
            ),
            "predecessor": {
                "run_id": self.predecessor_run_id,
                "run_directory": str(self.predecessor_run_directory),
                "terminal_status": "failed",
                "artifact_root_sha256": self.predecessor_artifact_root_sha256,
                "artifact_manifest_sha256": self.predecessor_artifact_manifest_sha256,
                "execution_source_root_sha256": (self.predecessor_execution_source_root_sha256),
                "execution_source_manifest_sha256": (
                    self.predecessor_execution_source_manifest_sha256
                ),
                "registration_authority": {
                    "directory": str(self.predecessor_registration_authority_directory),
                    "kind": self.predecessor_registration_authority_kind,
                    "artifact_root_sha256": (
                        self.predecessor_registration_authority_artifact_root_sha256
                    ),
                    "sha256_manifest_sha256": (
                        self.predecessor_registration_authority_manifest_sha256
                    ),
                },
            },
            "reuse": {
                "reused_required_cell_count": self.reused_required_cell_count,
                "reused_optional_cell_count": self.reused_optional_cell_count,
                "retrained_cell_count": 0,
                "selection_policy": _FINALIZATION_SELECTION_POLICY,
                "predecessor_access_policy": _FINALIZATION_ACCESS_POLICY,
            },
            "outcomes_inspected": False,
            "successor_evidence_filename": _FINALIZATION_SUCCESSOR_EVIDENCE_FILENAME,
            "successor_completion": {_FINALIZATION_SUCCESSOR_COMPLETION_FLAG: True},
            "successor_output_policy": _FINALIZATION_OUTPUT_POLICY,
            **(
                {"numeric_verification": self.numeric_verification.as_dict()}
                if self.numeric_verification is not None
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class ResourceBoundedConfirmatoryAuthorization:
    """Exact post-outcome authority for one resource-bounded confirmatory execution.

    The historical primary remains bound to its recovery amendment.  This object
    authorizes only the separately snapshotted confirmatory execution source and
    resource profile; it cannot rebind or mutate the primary and cannot claim the
    original confirmatory analysis or a completion stage.
    """

    primary_run_id: str
    primary_run_directory: Path
    primary_artifact_root_sha256: str
    primary_artifact_manifest_sha256: str
    primary_completion_evidence_sha256: str
    primary_execution_gate_sha256: str
    primary_stage_attestation_record_sha256: str
    primary_stage_attestation_verification_sha256: str
    primary_recovery_evidence_sha256: str
    primary_recovery_authorization_sha256: str
    recovery_authority_directory: Path
    recovery_authority_artifact_root_sha256: str
    recovery_authority_manifest_sha256: str
    recovery_authority_chain_depth: int
    resource_profile_id: str
    parent_confirmatory_config_file_sha256: str
    parent_confirmatory_config_semantic_sha256: str
    resource_confirmatory_config_file_sha256: str
    resource_confirmatory_config_semantic_sha256: str
    parent_execution_source_root_sha256: str
    parent_execution_source_manifest_sha256: str
    resource_execution_source_root_sha256: str
    resource_execution_source_manifest_sha256: str
    source_delta_records: tuple[Mapping[str, Any], ...]
    source_delta_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": _RESOURCE_BOUNDED_CONFIRMATORY_POLICY,
            "purpose": RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE,
            "historical_primary": {
                "experiment_name": _RESOURCE_BOUNDED_PRIMARY_EXPERIMENT_NAME,
                "run_id": self.primary_run_id,
                "run_directory": str(self.primary_run_directory),
                "terminal_status": "completed",
                "artifact_root_sha256": self.primary_artifact_root_sha256,
                "artifact_manifest_sha256": self.primary_artifact_manifest_sha256,
                "completion_evidence_sha256": self.primary_completion_evidence_sha256,
                "primary_execution_gate_sha256": self.primary_execution_gate_sha256,
                "stage_attestation_record_sha256": (self.primary_stage_attestation_record_sha256),
                "stage_attestation_verification_sha256": (
                    self.primary_stage_attestation_verification_sha256
                ),
                "recovery_evidence_sha256": self.primary_recovery_evidence_sha256,
                "recovery_authorization_sha256": (self.primary_recovery_authorization_sha256),
                "registration_authority": {
                    "directory": str(self.recovery_authority_directory),
                    "kind": _AUTHORITY_KIND,
                    "artifact_root_sha256": (self.recovery_authority_artifact_root_sha256),
                    "sha256_manifest_sha256": self.recovery_authority_manifest_sha256,
                    "chain_depth": self.recovery_authority_chain_depth,
                },
            },
            "resource_profile": {
                "profile_id": self.resource_profile_id,
                "experiment_name": _RESOURCE_BOUNDED_EXPERIMENT_NAME,
                "parent_confirmatory_config_file_sha256": (
                    self.parent_confirmatory_config_file_sha256
                ),
                "parent_confirmatory_config_semantic_sha256": (
                    self.parent_confirmatory_config_semantic_sha256
                ),
                "resource_confirmatory_config_file_sha256": (
                    self.resource_confirmatory_config_file_sha256
                ),
                "resource_confirmatory_config_semantic_sha256": (
                    self.resource_confirmatory_config_semantic_sha256
                ),
            },
            "execution_source_delta": {
                "policy": _RESOURCE_BOUNDED_SOURCE_DELTA_POLICY,
                "parent_root_sha256": self.parent_execution_source_root_sha256,
                "parent_manifest_sha256": (self.parent_execution_source_manifest_sha256),
                "resource_root_sha256": self.resource_execution_source_root_sha256,
                "resource_manifest_sha256": (self.resource_execution_source_manifest_sha256),
                "allowlisted_changes": [dict(record) for record in self.source_delta_records],
                "delta_sha256": self.source_delta_sha256,
            },
            "resource_capacity_policy": dict(_RESOURCE_BOUNDED_CAPACITY),
            "outcomes_inspected": True,
            "analysis_disposition": _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION,
            "outcome_use_policy": _RESOURCE_BOUNDED_OUTCOME_USE_POLICY,
            "original_confirmatory_claim_allowed": False,
            "study_outcome_eligible": False,
            "completion_stage": _RESOURCE_BOUNDED_COMPLETION_STAGE,
            "primary_rebinding_allowed": False,
            "primary_mutation_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class ResourceBoundedCnnProvenanceCorrection:
    """Exact before/after receipt for the failed CNN logical-provenance gate."""

    before_execution_cnn_source_sha256: str
    before_logical_provenance_source_sha256: str
    before_recomputed_record_sha256: str
    after_execution_cnn_source_sha256: str
    after_logical_provenance_source_sha256: str
    after_recomputed_record_sha256: str
    semantic_equivalence_evidence_sha256: str
    crop_cache_file_sha256: str
    crop_sidecar_file_sha256: str
    crop_sidecar_semantic_sha256: str

    def as_dict(
        self,
        *,
        before_config_record: Mapping[str, Any],
        after_config_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        before_record = dict(before_config_record)
        after_record = dict(after_config_record)
        return {
            "schema_version": 1,
            "policy": _RESOURCE_BOUNDED_CNN_PROVENANCE_CORRECTION_POLICY,
            "scenario_id": _RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID,
            "cache_provenance_id": _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID,
            "before_config_record": before_record,
            "before_config_record_sha256": _canonical_mapping_sha256(before_record),
            "after_config_record": after_record,
            "after_config_record_sha256": _canonical_mapping_sha256(after_record),
            "before": {
                "execution_cnn_source_sha256": (self.before_execution_cnn_source_sha256),
                "logical_provenance_source_sha256": (self.before_logical_provenance_source_sha256),
                "recomputed_record_sha256": self.before_recomputed_record_sha256,
                "matches_config_record": False,
            },
            "after": {
                "execution_cnn_source_sha256": (self.after_execution_cnn_source_sha256),
                "logical_provenance_source_sha256": (self.after_logical_provenance_source_sha256),
                "recomputed_record_sha256": self.after_recomputed_record_sha256,
                "matches_config_record": True,
                "semantic_equivalence_evidence_sha256": (self.semantic_equivalence_evidence_sha256),
            },
            "unchanged_cache_artifacts": {
                "cache_file_sha256": self.crop_cache_file_sha256,
                "sidecar_file_sha256": self.crop_sidecar_file_sha256,
                "sidecar_semantic_sha256": self.crop_sidecar_semantic_sha256,
                "cache_bytes_changed": False,
                "sidecar_bytes_changed": False,
            },
            "scientific_profile_changed": False,
        }


@dataclass(frozen=True, slots=True)
class ResourceBoundedTechnicalSuccessorAuthorization:
    """One direct child-C technical correction with no scientific broadening."""

    superseded_authority: Mapping[str, Any]
    prior_publication_failure: Mapping[str, Any]
    failed_preflight: Mapping[str, Any]
    historical_primary: Mapping[str, Any]
    resource_profile: Mapping[str, Any]
    execution_source_delta: Mapping[str, Any]
    cnn_provenance_correction: Mapping[str, Any]
    resource_capacity_policy: Mapping[str, Any]
    resource_input_workspace_plan: Mapping[str, Any]
    expected_successor_config_semantic_sha256: str
    replacement_publication_failure_lineage: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        schema_version = 3 if self.replacement_publication_failure_lineage is not None else 2
        payload = {
            "schema_version": schema_version,
            "policy": (
                _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY_V3
                if schema_version == 3
                else _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY
            ),
            "purpose": RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
            "supersedes": dict(self.superseded_authority),
            "prior_publication_failure": dict(self.prior_publication_failure),
            "failed_preflight": dict(self.failed_preflight),
            "historical_primary": dict(self.historical_primary),
            "resource_profile": dict(self.resource_profile),
            "execution_source_delta": dict(self.execution_source_delta),
            "cnn_provenance_correction": dict(self.cnn_provenance_correction),
            "resource_capacity_policy": dict(self.resource_capacity_policy),
            "resource_input_workspace_plan": dict(self.resource_input_workspace_plan),
            "expected_successor_config_semantic_sha256": (
                self.expected_successor_config_semantic_sha256
            ),
            "resource_profile_shape": dict(_RESOURCE_BOUNDED_PROFILE_SHAPE),
            "outcomes_inspected": True,
            "analysis_disposition": _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION,
            "outcome_use_policy": _RESOURCE_BOUNDED_OUTCOME_USE_POLICY,
            "original_confirmatory_claim_allowed": False,
            "study_outcome_eligible": False,
            "completion_stage": _RESOURCE_BOUNDED_COMPLETION_STAGE,
            "primary_rebinding_allowed": False,
            "primary_mutation_allowed": False,
            "automatic_retry_allowed": False,
            "scientific_profile_change_allowed": False,
        }
        if self.replacement_publication_failure_lineage is not None:
            payload["replacement_publication_failure_lineage"] = dict(
                self.replacement_publication_failure_lineage
            )
        return payload


@dataclass(frozen=True, slots=True)
class PreregistrationAmendmentResult:
    """Published paths and immutable identity for one successor authority."""

    amendment_directory: Path
    parent_authority_directory: Path
    amendment_timestamp_utc: str
    chain_depth: int
    amendment_evidence_path: Path
    amended_preregistration_path: Path
    amended_primary_config_path: Path
    amended_confirmatory_config_path: Path
    source_tree_manifest_path: Path
    sha256_manifest_path: Path
    immutable_marker_path: Path
    artifact_root_sha256: str
    sha256_manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class PreregistrationAmendmentVerification:
    """Independent verification result for a complete amendment chain."""

    valid: bool
    amendment_directory: Path
    chain_depth: int | None
    artifact_root_sha256: str | None
    sha256_manifest_sha256: str | None
    parent_authority_directory: Path | None
    errors: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True, slots=True)
class ResourceBoundedTechnicalSuccessorVerification:
    """Fresh-process-ready proof for one exact effective schema-v5 authority D."""

    successor_directory: Path
    parent_authority_directory: Path
    chain_depth: int
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    authorization_sha256: str
    intent_sha256: str
    flat_file_inventory_sha256: str
    confirmatory_storage_policy_sha256: str
    flat_file_count: int
    manifest_artifact_count: int
    controller_process_id: int
    verifier_process_id: int
    verifier_parent_process_id: int
    verification_nonce: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "verification_schema_version": 2,
            "verification_kind": ("resource_bounded_technical_successor_fresh_process"),
            "process_boundary": {
                "controller_process_id": self.controller_process_id,
                "verifier_process_id": self.verifier_process_id,
                "verifier_parent_process_id": self.verifier_parent_process_id,
                "distinct_processes": True,
                "direct_child_process": True,
                "verification_nonce": self.verification_nonce,
            },
            "successor_authority": {
                "directory": str(self.successor_directory),
                "schema_version": 5,
                "purpose": RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
                "chain_depth": self.chain_depth,
                "artifact_root_sha256": self.artifact_root_sha256,
                "sha256_manifest_sha256": self.sha256_manifest_sha256,
                "authorization_sha256": self.authorization_sha256,
                "intent_sha256": self.intent_sha256,
            },
            "superseded_authority": {
                "directory": str(self.parent_authority_directory),
                "schema_version": 4,
                "historically_verified": True,
                "effective_execution_leaf": False,
            },
            "bundle": {
                "flat_file_count": self.flat_file_count,
                "manifest_artifact_count": self.manifest_artifact_count,
                "flat_file_inventory_sha256": self.flat_file_inventory_sha256,
                "flat_file_hashes_verified": True,
            },
            "confirmatory_storage_policy_sha256": (self.confirmatory_storage_policy_sha256),
            "successor_candidate_count": 1,
            "checks": {
                "generic_chain_integrity": True,
                "typed_successor_authorization": True,
                "effective_execution_leaf": True,
                "historical_c_integrity": True,
                "historical_c_typed_authorization": True,
                "c_superseded_for_execution": True,
                "unique_direct_successor": True,
                "storage_policy_inherited_unchanged": True,
                "flat_exact_file_set": True,
                "external_intent_binding": True,
                "fresh_process_boundary": True,
                "live_prior_publication_failure": True,
                "live_failed_preflight_receipt": True,
                "live_historical_primary": True,
            },
            "outcome_value_interpretation_performed": False,
            "scientific_execution_performed": False,
            "publication_performed": False,
        }


@dataclass(frozen=True, slots=True)
class _BundleIntegrity:
    artifact_root_sha256: str
    sha256_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _AuthorityState:
    directory: Path
    kind: AuthorityKind
    timestamp_utc: str
    timestamp: datetime
    chain_depth: int
    artifact_root_sha256: str
    sha256_manifest_sha256: str
    snapshot_hashes: dict[str, Any]
    parent_directory: Path | None


class _AuthorityValidationError(ValueError):
    """Internal error converted to a structured public verification result."""


def _json_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            f"{role} is missing or invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise _AuthorityValidationError(f"{role} must be a JSON object: {path}")
    return payload


def _canonical_root(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    value = path.stat(follow_symlinks=False)
    attributes = int(getattr(value, "st_file_attributes", 0))
    return path.is_symlink() or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _require_real_directory(path: Path, *, role: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    chain = (*reversed(lexical.parents), lexical)
    try:
        for component in chain:
            if not os.path.lexists(component):
                raise _AuthorityValidationError(
                    f"{role} does not exist as a directory: {component}"
                )
            value = component.stat(follow_symlinks=False)
            if _is_link_or_reparse(component):
                raise _AuthorityValidationError(
                    f"{role} path must not traverse a symlink or reparse point: {component}"
                )
            if not stat.S_ISDIR(value.st_mode):
                raise _AuthorityValidationError(
                    f"{role} path component is not a directory: {component}"
                )
    except OSError as exc:
        raise _AuthorityValidationError(
            f"{role} could not be validated as a lexical directory: {lexical}: {exc}"
        ) from exc
    return lexical


def _require_regular_file(path: Path, *, role: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if _is_link_or_reparse(path):
        raise ValueError(f"{role} must not be a symlink or reparse point: {path}")
    content = path.read_bytes()
    if not content:
        raise ValueError(f"{role} must not be empty: {path}")
    return content


def _read_stable_single_link_file(
    path: Path,
    *,
    role: str,
    allow_empty: bool = False,
    max_bytes: int | None = None,
) -> bytes:
    """Read one exact file through a no-follow, parent-anchored descriptor."""

    try:
        return read_file_anchored(
            path,
            allow_empty=allow_empty,
            require_single_link=True,
            max_bytes=max_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise _AuthorityValidationError(
            f"{role} failed anchored single-link readback: {path}: {exc}"
        ) from exc


def _normalise_timestamp(value: datetime | None) -> tuple[datetime, str, str]:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("amendment timestamp must be timezone-aware")
    utc = moment.astimezone(UTC)
    rendered = utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc, rendered, utc.strftime("%Y%m%dT%H%M%S.%fZ")


def _parse_timestamp(value: Any, *, role: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise _AuthorityValidationError(f"{role} must be a UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _AuthorityValidationError(f"{role} is not a valid timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.now(UTC).utcoffset():
        raise _AuthorityValidationError(f"{role} must have UTC offset zero")
    return parsed.astimezone(UTC)


def _normalise_text(value: str, *, role: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{role} must not be blank")
    if any(character in normalised for character in ("\r", "\n", "\0")):
        raise ValueError(f"{role} must be a single safe text line")
    return normalised


def _normalise_items(values: Sequence[str], *, role: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{role} must be a sequence of explicit strings")
    items = tuple(_normalise_text(value, role=f"{role} item") for value in values)
    if not items:
        raise ValueError(f"{role} must contain at least one explicit item")
    if len(set(items)) != len(items):
        raise ValueError(f"{role} must not contain duplicate items")
    return items


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _AuthorityValidationError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _absolute_record_path(value: Any, *, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _AuthorityValidationError(f"{role} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise _AuthorityValidationError(f"{role} must be an absolute path")
    return path.resolve(strict=False)


def _absolute_lexical_record_path(value: Any, *, role: str) -> Path:
    """Canonicalise an evidence path without following any filesystem component."""

    if not isinstance(value, str) or not value:
        raise _AuthorityValidationError(f"{role} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise _AuthorityValidationError(f"{role} must be an absolute path")
    return Path(os.path.abspath(path))


def _exact_schema_v3_record_path(
    value: Any,
    *,
    expected: Path,
    role: str,
) -> Path:
    """Require the exact canonical lexical spelling stored by schema-v3 evidence."""

    if type(value) is not str or value != str(expected):
        raise _AuthorityValidationError(f"{role} must use its exact canonical absolute path")
    return expected


def _strict_cell_count(value: Any, *, role: str, positive: bool) -> int:
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise _AuthorityValidationError(f"{role} must be a {qualifier} exact integer")
    return value


def _require_registry_backed_failed_run(
    run_directory: Path,
    *,
    run_id: str,
    artifact_root_sha256: str,
    artifact_manifest_sha256: str,
) -> None:
    """Require one exact failed RunTracker seal and its append-only registry record."""

    integrity = verify_run_integrity(run_directory)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_id
        or integrity.expected_root_sha256 != artifact_root_sha256
        or integrity.actual_root_sha256 != artifact_root_sha256
        or sha256_file(run_directory / RUN_ARTIFACT_MANIFEST_FILENAME) != artifact_manifest_sha256
    ):
        details = "; ".join(integrity.errors) or "technical integrity mismatch"
        raise _AuthorityValidationError(
            "finalization predecessor must be one exact registry-backed, "
            f"integrity-valid failed seal: {details}"
        )

    status_path = run_directory / RUN_STATUS_FILENAME
    _require_regular_file(status_path, role="failed predecessor status")
    status = _json_mapping(status_path, role="failed predecessor status")
    if status.get("run_id") != run_id or status.get("status") != "failed":
        raise _AuthorityValidationError(
            "finalization predecessor status does not identify the exact terminally failed run"
        )


def _validate_live_failed_predecessor(record: Mapping[str, Any]) -> None:
    predecessor = record["predecessor"]
    if not isinstance(predecessor, Mapping):  # Defensive; canonical validation runs first.
        raise _AuthorityValidationError("finalization predecessor must be an object")
    run_directory = _require_real_directory(
        Path(str(predecessor["run_directory"])), role="finalization predecessor run"
    )
    run_id = str(predecessor["run_id"])
    if run_directory.name != run_id:
        raise _AuthorityValidationError(
            "finalization predecessor run path does not end in its exact run ID"
        )

    manifest_path = run_directory / RUN_ARTIFACT_MANIFEST_FILENAME
    marker_path = run_directory / RUN_IMMUTABLE_MARKER
    _require_regular_file(manifest_path, role="failed predecessor artifact manifest")
    _require_regular_file(marker_path, role="failed predecessor immutable marker")
    manifest = _json_mapping(manifest_path, role="failed predecessor artifact manifest")
    marker = _json_mapping(marker_path, role="failed predecessor immutable marker")
    manifest_sha256 = sha256_file(manifest_path)
    manifest_records = _normalise_manifest_records(manifest.get("artifacts"), flat=False)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "failed"
        or manifest.get("artifact_count") != len(manifest_records)
        or manifest.get("artifact_root_sha256") != _canonical_root(manifest_records)
    ):
        raise _AuthorityValidationError(
            "finalization predecessor must have an exact schema-v1 terminal failed manifest"
        )
    if (
        manifest.get("artifact_root_sha256") != predecessor["artifact_root_sha256"]
        or manifest_sha256 != predecessor["artifact_manifest_sha256"]
    ):
        raise _AuthorityValidationError(
            "finalization predecessor seal hashes differ from the authorization"
        )
    if (
        marker.get("artifact_count") != len(manifest_records)
        or marker.get("run_id") != run_id
        or marker.get("status") != "failed"
        or marker.get("artifact_root_sha256") != predecessor["artifact_root_sha256"]
        or marker.get("artifact_manifest_sha256") != manifest_sha256
        or not isinstance(marker.get("run_path"), str)
        or Path(str(marker["run_path"])).resolve(strict=False) != run_directory
    ):
        raise _AuthorityValidationError(
            "finalization predecessor immutable marker does not bind the exact failed seal"
        )

    source_manifest_path = run_directory / RUN_SOURCE_TREE_MANIFEST_FILENAME
    gate_path = run_directory / "primary_execution_gate.json"
    records_by_path = {record["path"]: record for record in manifest_records}
    for artifact_path in (source_manifest_path, gate_path):
        artifact_record = records_by_path.get(artifact_path.name)
        if (
            artifact_record is None
            or not artifact_path.is_file()
            or _is_link_or_reparse(artifact_path)
            or artifact_record["size_bytes"] != artifact_path.stat().st_size
            or artifact_record["sha256"] != sha256_file(artifact_path)
        ):
            raise _AuthorityValidationError(
                "failed predecessor structural authority files differ from the sealed manifest"
            )
    source_hashes = _execution_hashes(source_manifest_path)
    if (
        source_hashes["root_sha256"] != predecessor["execution_source_root_sha256"]
        or source_hashes["manifest_sha256"] != predecessor["execution_source_manifest_sha256"]
    ):
        raise _AuthorityValidationError(
            "finalization predecessor computation source differs from the authorization"
        )

    gate = _json_mapping(gate_path, role="failed predecessor primary execution gate")
    authority = predecessor["registration_authority"]
    if not isinstance(authority, Mapping):  # Defensive; canonical validation runs first.
        raise _AuthorityValidationError("predecessor registration authority must be an object")
    gate_directory = gate.get("freeze_directory")
    if (
        not isinstance(gate_directory, str)
        or Path(gate_directory).resolve(strict=False) != Path(str(authority["directory"]))
        or gate.get("registration_authority_kind") != authority["kind"]
        or gate.get("freeze_artifact_root_sha256") != authority["artifact_root_sha256"]
        or gate.get("freeze_manifest_sha256") != authority["sha256_manifest_sha256"]
        or gate.get("source_tree_root_sha256") != predecessor["execution_source_root_sha256"]
    ):
        raise _AuthorityValidationError(
            "failed predecessor execution gate differs from its computation authority binding"
        )
    _require_registry_backed_failed_run(
        run_directory,
        run_id=run_id,
        artifact_root_sha256=str(predecessor["artifact_root_sha256"]),
        artifact_manifest_sha256=str(predecessor["artifact_manifest_sha256"]),
    )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json_sha256(value: Any) -> str:
    return hashlib.sha256(_atomic_json_bytes(value)).hexdigest()


def _canonical_execution_source_records(
    value: Mapping[str, Any],
    *,
    role: str,
) -> list[dict[str, Any]]:
    expected_fields = {
        "schema_version",
        "scope_kind",
        "scope",
        "excluded_roots",
        "excluded_paths",
        "artifact_count",
        "root_sha256",
        "artifacts",
    }
    if (
        set(value) != expected_fields
        or value.get("schema_version") != 3
        or value.get("scope_kind") != "execution_source"
        or value.get("scope") != _EXECUTION_SCOPE
        or value.get("excluded_roots") != _EXECUTION_EXCLUDED_ROOTS
        or value.get("excluded_paths") != _EXECUTION_EXCLUDED_PATHS
    ):
        raise _AuthorityValidationError(f"{role} has an invalid execution-source schema")
    records = _normalise_manifest_records(value.get("artifacts"), flat=False)
    if value.get("artifact_count") != len(records) or value.get("root_sha256") != _canonical_root(
        records
    ):
        raise _AuthorityValidationError(f"{role} has an invalid artifact count or root")
    return records


def _canonical_source_delta_with_allowlist(
    parent_source: Mapping[str, Any],
    resource_source: Mapping[str, Any],
    *,
    allowlisted_change_kinds: Mapping[str, str],
    role: str,
) -> tuple[tuple[dict[str, Any], ...], str]:
    if not isinstance(allowlisted_change_kinds, Mapping) or not allowlisted_change_kinds:
        raise _AuthorityValidationError(f"{role} allowlist must be a non-empty mapping")
    canonical_allowlist: dict[str, str] = {}
    for raw_path, raw_kind in allowlisted_change_kinds.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise _AuthorityValidationError(f"{role} allowlist contains an invalid path")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or path.as_posix() != raw_path
            or ".." in path.parts
            or any(character in raw_path for character in ("*", "?", "[", "]", "{", "}", "\\"))
            or not (
                raw_path in {"pyproject.toml", "uv.lock"}
                or raw_path.startswith("src/")
                or raw_path.startswith("configs/")
            )
            or raw_path in _EXECUTION_EXCLUDED_PATHS
        ):
            raise _AuthorityValidationError(
                f"{role} allowlist path is not one exact execution-source path: {raw_path!r}"
            )
        if raw_kind not in {"added", "modified", "removed"}:
            raise _AuthorityValidationError(
                f"{role} allowlist has an invalid change kind for {raw_path!r}"
            )
        canonical_allowlist[raw_path] = raw_kind
    if list(canonical_allowlist) != sorted(canonical_allowlist):
        canonical_allowlist = {
            path: canonical_allowlist[path] for path in sorted(canonical_allowlist)
        }

    parent_records = {
        record["path"]: record
        for record in _canonical_execution_source_records(
            parent_source,
            role=f"{role} parent execution source",
        )
    }
    resource_records = {
        record["path"]: record
        for record in _canonical_execution_source_records(
            resource_source,
            role=f"{role} execution source",
        )
    }
    delta: list[dict[str, Any]] = []
    for path in sorted(set(parent_records).union(resource_records)):
        before = parent_records.get(path)
        after = resource_records.get(path)
        if before == after:
            continue
        if before is None:
            change_kind = "added"
        elif after is None:
            change_kind = "removed"
        else:
            change_kind = "modified"
        delta.append(
            {
                "path": path,
                "change_kind": change_kind,
                "before": (
                    None
                    if before is None
                    else {
                        "size_bytes": before["size_bytes"],
                        "sha256": before["sha256"],
                    }
                ),
                "after": (
                    None
                    if after is None
                    else {
                        "size_bytes": after["size_bytes"],
                        "sha256": after["sha256"],
                    }
                ),
            }
        )
    if not delta:
        raise _AuthorityValidationError(f"{role} execution source delta must be non-empty")
    observed_kinds = {str(record["path"]): str(record["change_kind"]) for record in delta}
    if observed_kinds != canonical_allowlist:
        unknown = sorted(set(observed_kinds).difference(canonical_allowlist))
        missing = sorted(set(canonical_allowlist).difference(observed_kinds))
        wrong_kind = sorted(
            path
            for path in set(observed_kinds).intersection(canonical_allowlist)
            if observed_kinds[path] != canonical_allowlist[path]
        )
        raise _AuthorityValidationError(
            f"{role} execution source delta differs from its closed allowlist: "
            f"unknown={unknown}, missing={missing}, wrong_kind={wrong_kind}"
        )
    canonical = tuple(delta)
    return canonical, _canonical_value_sha256(list(canonical))


def _canonical_resource_source_delta(
    parent_source: Mapping[str, Any],
    resource_source: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str]:
    return _canonical_source_delta_with_allowlist(
        parent_source,
        resource_source,
        allowlisted_change_kinds=_RESOURCE_BOUNDED_SOURCE_DELTA_KINDS,
        role="resource-bounded",
    )


def _resource_parent_recovery_authorization(
    parent_state: _AuthorityState,
) -> dict[str, Any]:
    if parent_state.kind != _AUTHORITY_KIND:
        raise _AuthorityValidationError(
            "resource-bounded execution requires a direct recovery-amendment parent"
        )
    evidence = _json_mapping(
        parent_state.directory / _EVIDENCE_FILENAME,
        role="resource-bounded parent amendment evidence",
    )
    if (
        evidence.get("schema_version") != 3
        or evidence.get("outcomes_inspected") is not True
        or "primary_recovery_authorization" not in evidence
        or "finalization_successor_authorization" in evidence
    ):
        raise _AuthorityValidationError(
            "resource-bounded execution parent must be the exact post-outcome "
            "primary-recovery authority"
        )
    return require_primary_recovery_authorization(parent_state.directory)


def _validate_live_resource_primary_binding(
    historical_primary: Mapping[str, Any],
    *,
    parent_state: _AuthorityState,
    recovery_authorization_sha256: str,
) -> None:
    run_directory = _absolute_record_path(
        historical_primary.get("run_directory"),
        role="resource-bounded historical primary run",
    )
    run_id = historical_primary.get("run_id")
    if (
        not isinstance(run_id, str)
        or run_directory.name != run_id
        or historical_primary.get("experiment_name") != _RESOURCE_BOUNDED_PRIMARY_EXPERIMENT_NAME
        or historical_primary.get("terminal_status") != "completed"
    ):
        raise _AuthorityValidationError("resource-bounded historical primary identity is invalid")
    integrity = verify_run_integrity(run_directory)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run_id
        or integrity.expected_root_sha256 != historical_primary.get("artifact_root_sha256")
        or integrity.actual_root_sha256 != historical_primary.get("artifact_root_sha256")
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary failed sealed integrity verification"
        )
    receipt = require_run_stage_eligibility_receipt(run_directory, integrity=integrity)
    if (
        receipt is None
        or not receipt.valid
        or receipt.run_directory.resolve() != run_directory
        or receipt.run_id != run_id
        or receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
        or receipt.record_sha256 != historical_primary.get("stage_attestation_record_sha256")
        or receipt.verification_sha256
        != historical_primary.get("stage_attestation_verification_sha256")
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary lacks its exact positive stage attestation"
        )
    required_paths = {
        "artifact_manifest_sha256": run_directory / RUN_ARTIFACT_MANIFEST_FILENAME,
        "completion_evidence_sha256": run_directory / "completion_evidence.json",
        "primary_execution_gate_sha256": run_directory / "primary_execution_gate.json",
        "recovery_evidence_sha256": run_directory / "primary_recovery_evidence.json",
    }
    for field, path in required_paths.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != historical_primary.get(field)
        ):
            raise _AuthorityValidationError(
                f"resource-bounded historical primary {field} differs from its live seal"
            )
    gate = _json_mapping(
        run_directory / "primary_execution_gate.json",
        role="resource-bounded historical primary execution gate",
    )
    if (
        gate.get("freeze_directory") != str(parent_state.directory)
        or gate.get("freeze_artifact_root_sha256") != parent_state.artifact_root_sha256
        or gate.get("freeze_manifest_sha256") != parent_state.sha256_manifest_sha256
        or historical_primary.get("recovery_authorization_sha256") != recovery_authorization_sha256
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary is not bound to the direct recovery parent"
        )


def _build_inherited_prior_numeric_verification(
    run_directory: Path,
    *,
    trusted_source_signature: Mapping[str, Any] | None = None,
) -> InheritedPriorNumericVerificationAuthorization:
    """Derive the closed B-fast proof from exact sealed terminal artifacts."""

    signature = (
        _trusted_prior_numeric_source_signature()
        if trusted_source_signature is None
        else _canonical_trusted_prior_numeric_source_signature(trusted_source_signature)
    )
    run = run_directory.resolve()
    if run.name != signature.get("predecessor_run_id"):
        raise _AuthorityValidationError(
            "inherited prior numeric verification is restricted to the trusted predecessor"
        )
    source_path = run / RUN_SOURCE_TREE_MANIFEST_FILENAME
    source = _json_mapping(source_path, role="trusted predecessor source manifest")
    source_records = source.get("artifacts")
    trusted_files = signature.get("files")
    if (
        source.get("root_sha256") != signature.get("source_tree_root_sha256")
        or sha256_file(source_path) != signature.get("source_tree_manifest_sha256")
        or not isinstance(source_records, list)
        or not isinstance(trusted_files, list)
    ):
        raise _AuthorityValidationError("trusted predecessor source signature differs")
    source_by_path = {
        record.get("path"): record for record in source_records if isinstance(record, Mapping)
    }
    if any(source_by_path.get(record.get("path")) != record for record in trusted_files):
        raise _AuthorityValidationError("trusted predecessor source-file signature differs")
    control_flow = signature.get("control_flow")
    if not isinstance(control_flow, Mapping):
        raise _AuthorityValidationError("trusted numeric control-flow signature is invalid")
    error_type = control_flow.get("expected_error_type")
    error_message = control_flow.get("expected_error_message")
    expected_frames = control_flow.get("ordered_traceback_frames")
    if (
        not isinstance(error_type, str)
        or not isinstance(error_message, str)
        or not isinstance(expected_frames, list)
        or not expected_frames
    ):
        raise _AuthorityValidationError("trusted numeric terminal signature is invalid")

    status_path = run / RUN_STATUS_FILENAME
    traceback_path = run / "traceback.txt"
    events_path = run / "events.jsonl"
    completion_path = run / "completion_evidence.json"
    manifest_path = run / RUN_ARTIFACT_MANIFEST_FILENAME
    for path, role in (
        (status_path, "trusted predecessor status"),
        (traceback_path, "trusted predecessor traceback"),
        (events_path, "trusted predecessor events"),
        (completion_path, "trusted predecessor demoted completion"),
        (manifest_path, "trusted predecessor artifact manifest"),
    ):
        _require_regular_file(path, role=role)

    status = _json_mapping(status_path, role="trusted predecessor status")
    completion = _json_mapping(completion_path, role="trusted predecessor demoted completion")
    expected_failure = f"{error_type}: {error_message}"
    if (
        status.get("run_id") != run.name
        or status.get("status") != "failed"
        or status.get("experiment_name") != "pannuke_primary_frozen_feature_benchmark"
        or status.get("traceback") != "traceback.txt"
        or completion.get("completion_stage") is not None
        or completion.get("study_outcome_eligible") is not False
        or completion.get("valid_completion_claim") is not False
        or completion.get("artifact_scope") != "real_pannuke_primary_study"
        or completion.get("runner_failure") != expected_failure
    ):
        raise _AuthorityValidationError(
            "trusted predecessor status or fail-closed demotion differs"
        )

    traceback_text = traceback_path.read_text(encoding="utf-8")
    forbidden_traceback_markers = (
        "During handling of the above exception",
        "The above exception was the direct cause",
        "ExceptionGroup",
        "+ Exception Group Traceback",
    )
    if (
        traceback_text.count("Traceback (most recent call last):") != 1
        or not traceback_text.startswith("Traceback (most recent call last):\n")
        or any(marker in traceback_text for marker in forbidden_traceback_markers)
    ):
        raise _AuthorityValidationError(
            "trusted predecessor traceback must contain one exact exception chain"
        )
    traceback_frames = re.findall(
        r'  File "([^"]+)", line ([0-9]+), in ([^\r\n]+)',
        traceback_text,
    )
    if len(traceback_frames) != len(expected_frames):
        raise _AuthorityValidationError(
            "trusted predecessor traceback has a missing or extra frame"
        )
    for observed, expected in zip(traceback_frames, expected_frames, strict=True):
        if (
            not isinstance(expected, Mapping)
            or not observed[0].replace("\\", "/").endswith(str(expected.get("path_suffix")))
            or int(observed[1]) != expected.get("line")
            or observed[2].strip() != expected.get("function")
        ):
            raise _AuthorityValidationError("trusted predecessor traceback frame signature differs")
    final_line = next(
        (line for line in reversed(traceback_text.splitlines()) if line.strip()),
        "",
    )
    if final_line != expected_failure:
        raise _AuthorityValidationError("trusted predecessor traceback failure differs")

    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
            events.append(event)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _AuthorityValidationError(f"trusted predecessor events are invalid: {exc}") from exc
    failed_events = [event for event in events if event.get("event") == "run_failed"]
    if (
        len(failed_events) != 1
        or failed_events[0].get("run_id") != run.name
        or failed_events[0].get("status") != "failed"
        or failed_events[0].get("error_type") != error_type
        or failed_events[0].get("error_message") != error_message
        or failed_events[0].get("traceback") != "traceback.txt"
    ):
        raise _AuthorityValidationError(
            "trusted predecessor lacks one exact sealed run_failed event"
        )

    manifest = _json_mapping(manifest_path, role="trusted predecessor artifact manifest")
    records = _normalise_manifest_records(manifest.get("artifacts"), flat=False)
    records_by_path = {record["path"]: record for record in records}
    terminal_names = (
        RUN_STATUS_FILENAME,
        "traceback.txt",
        "events.jsonl",
        "completion_evidence.json",
        RUN_SOURCE_TREE_MANIFEST_FILENAME,
    )
    quartet_names = (
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
    )
    for name in (*terminal_names, *quartet_names):
        record = records_by_path.get(name)
        path = run / name
        if (
            record is None
            or not path.is_file()
            or _is_link_or_reparse(path)
            or record["size_bytes"] != path.stat().st_size
            or record["sha256"] != sha256_file(path)
        ):
            raise _AuthorityValidationError(f"trusted predecessor sealed artifact differs: {name}")
    statistics_manifest = _json_mapping(
        run / "primary_statistics_manifest.json",
        role="trusted predecessor statistics manifest",
    )
    statistics_source_root = _require_sha256(
        statistics_manifest.get("source_filesystem_readback_root_sha256"),
        role="trusted predecessor statistics source readback root",
    )
    expected_comparison_count = control_flow.get("expected_comparison_count")
    if type(expected_comparison_count) is not int or expected_comparison_count != 36:
        raise _AuthorityValidationError("trusted numeric comparison count is invalid")
    terminal_evidence = {
        "schema_version": 1,
        "run_id": run.name,
        "artifact_root_sha256": manifest.get("artifact_root_sha256"),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "status_sha256": sha256_file(status_path),
        "traceback_sha256": sha256_file(traceback_path),
        "events_sha256": sha256_file(events_path),
        "completion_evidence_sha256": sha256_file(completion_path),
        "run_failed_event_sha256": _canonical_mapping_sha256(failed_events[0]),
        "statistics_quartet": [records_by_path[name] for name in quartet_names],
        "statistics_source_readback_root_sha256": statistics_source_root,
        "statistics_comparison_count": expected_comparison_count,
        "completed_required_cell_count": 185,
        "required_cell_count": 185,
    }
    proof_payload = {
        "trust_assumption": _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
        "limitation": _INHERITED_PRIOR_NUMERIC_LIMITATION,
        "trusted_source_signature": signature,
        "terminal_evidence": terminal_evidence,
    }
    return InheritedPriorNumericVerificationAuthorization(
        trusted_source_signature=signature,
        terminal_evidence=terminal_evidence,
        prior_numeric_verification_proof_sha256=_canonical_mapping_sha256(proof_payload),
    )


def _canonical_inherited_prior_numeric_verification(
    value: InheritedPriorNumericVerificationAuthorization | Mapping[str, Any],
    *,
    run_directory: Path,
    verify_live_predecessor: bool,
) -> dict[str, Any]:
    raw = (
        value.as_dict()
        if isinstance(value, InheritedPriorNumericVerificationAuthorization)
        else value
    )
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "mode",
        "policy",
        "trust_assumption",
        "limitation",
        "trusted_source_signature",
        "terminal_evidence",
        "prior_numeric_verification_proof_sha256",
    }:
        raise _AuthorityValidationError(
            "inherited numeric-verification authorization has an invalid field set"
        )
    signature = raw.get("trusted_source_signature")
    terminal = raw.get("terminal_evidence")
    canonical_signature = _canonical_trusted_prior_numeric_source_signature(signature)
    expected_terminal_fields = {
        "schema_version",
        "run_id",
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "status_sha256",
        "traceback_sha256",
        "events_sha256",
        "completion_evidence_sha256",
        "run_failed_event_sha256",
        "statistics_quartet",
        "statistics_source_readback_root_sha256",
        "statistics_comparison_count",
        "completed_required_cell_count",
        "required_cell_count",
    }
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("mode") != INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
        or raw.get("policy") != _INHERITED_PRIOR_NUMERIC_POLICY
        or raw.get("trust_assumption") != _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
        or raw.get("limitation") != _INHERITED_PRIOR_NUMERIC_LIMITATION
        or not isinstance(terminal, Mapping)
        or set(terminal) != expected_terminal_fields
        or type(terminal.get("schema_version")) is not int
        or terminal.get("schema_version") != 1
        or type(terminal.get("statistics_comparison_count")) is not int
        or terminal.get("statistics_comparison_count") != 36
        or type(terminal.get("completed_required_cell_count")) is not int
        or terminal.get("completed_required_cell_count") != 185
        or type(terminal.get("required_cell_count")) is not int
        or terminal.get("required_cell_count") != 185
        or _SHA256.fullmatch(str(terminal.get("statistics_source_readback_root_sha256"))) is None
    ):
        raise _AuthorityValidationError(
            "inherited numeric-verification policy/signature is invalid"
        )
    quartet_names = (
        "primary_statistics.json",
        "primary_bootstrap_evidence.npz",
        "primary_subgroups.csv",
        "primary_statistics_manifest.json",
    )
    raw_quartet = terminal.get("statistics_quartet")
    if not isinstance(raw_quartet, list) or len(raw_quartet) != len(quartet_names):
        raise _AuthorityValidationError("inherited statistics quartet is incomplete")
    canonical_quartet: list[dict[str, Any]] = []
    for expected_name, record in zip(quartet_names, raw_quartet, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise _AuthorityValidationError("inherited statistics quartet record is invalid")
        size_bytes = record.get("size_bytes")
        if record.get("path") != expected_name or type(size_bytes) is not int or size_bytes < 0:
            raise _AuthorityValidationError("inherited statistics quartet fields are invalid")
        canonical_quartet.append(
            {
                "path": expected_name,
                "size_bytes": size_bytes,
                "sha256": _require_sha256(
                    record.get("sha256"),
                    role=f"inherited statistics quartet {expected_name}",
                ),
            }
        )
    canonical_terminal = {
        "schema_version": 1,
        "run_id": canonical_signature["predecessor_run_id"],
        "artifact_root_sha256": _require_sha256(
            terminal.get("artifact_root_sha256"),
            role="prior-numeric terminal artifact root",
        ),
        "artifact_manifest_sha256": _require_sha256(
            terminal.get("artifact_manifest_sha256"),
            role="prior-numeric terminal artifact manifest",
        ),
        "status_sha256": _require_sha256(
            terminal.get("status_sha256"),
            role="prior-numeric terminal status",
        ),
        "traceback_sha256": _require_sha256(
            terminal.get("traceback_sha256"),
            role="prior-numeric terminal traceback",
        ),
        "events_sha256": _require_sha256(
            terminal.get("events_sha256"),
            role="prior-numeric terminal events",
        ),
        "completion_evidence_sha256": _require_sha256(
            terminal.get("completion_evidence_sha256"),
            role="prior-numeric terminal completion",
        ),
        "run_failed_event_sha256": _require_sha256(
            terminal.get("run_failed_event_sha256"),
            role="prior-numeric terminal failed event",
        ),
        "statistics_quartet": canonical_quartet,
        "statistics_source_readback_root_sha256": _require_sha256(
            terminal.get("statistics_source_readback_root_sha256"),
            role="prior-numeric statistics source readback",
        ),
        "statistics_comparison_count": 36,
        "completed_required_cell_count": 185,
        "required_cell_count": 185,
    }
    if dict(terminal) != canonical_terminal:
        raise _AuthorityValidationError(
            "inherited numeric-verification terminal evidence is not canonical"
        )
    proof_payload = {
        "trust_assumption": _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
        "limitation": _INHERITED_PRIOR_NUMERIC_LIMITATION,
        "trusted_source_signature": canonical_signature,
        "terminal_evidence": canonical_terminal,
    }
    proof_sha = _require_sha256(
        raw.get("prior_numeric_verification_proof_sha256"),
        role="prior numeric verification proof",
    )
    if proof_sha != _canonical_mapping_sha256(proof_payload):
        raise _AuthorityValidationError("prior numeric verification proof hash differs")
    if verify_live_predecessor:
        observed = _build_inherited_prior_numeric_verification(
            run_directory,
            trusted_source_signature=canonical_signature,
        ).as_dict()
        if dict(raw) != observed:
            raise _AuthorityValidationError("inherited numeric-verification terminal proof changed")
    return {
        "schema_version": 1,
        "mode": INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        "policy": _INHERITED_PRIOR_NUMERIC_POLICY,
        "trust_assumption": _INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION,
        "limitation": _INHERITED_PRIOR_NUMERIC_LIMITATION,
        "trusted_source_signature": canonical_signature,
        "terminal_evidence": canonical_terminal,
        "prior_numeric_verification_proof_sha256": proof_sha,
    }


def _canonical_finalization_successor_authorization(
    value: FinalizationSuccessorAuthorization | Mapping[str, Any],
    *,
    parent_state: _AuthorityState,
    verify_live_predecessor: bool,
) -> dict[str, Any]:
    raw: Mapping[str, Any]
    if isinstance(value, FinalizationSuccessorAuthorization):
        raw = value.as_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(
            "finalization_successor_authorization must be a typed authorization or mapping"
        )
    base_fields = {
        "schema_version",
        "policy",
        "predecessor",
        "reuse",
        "outcomes_inspected",
        "successor_evidence_filename",
        "successor_completion",
        "successor_output_policy",
    }
    authorization_schema = raw.get("schema_version")
    if type(authorization_schema) is not int:
        raise _AuthorityValidationError(
            "finalization-successor authorization schema must be an exact integer"
        )
    if authorization_schema == 1 and raw.get("policy") == _FINALIZATION_SUCCESSOR_POLICY:
        expected_fields = base_fields
    elif authorization_schema == 2 and raw.get("policy") == _FINALIZATION_SUCCESSOR_POLICY_V2:
        expected_fields = {*base_fields, "numeric_verification"}
    else:
        raise _AuthorityValidationError(
            "finalization-successor authorization schema/policy is invalid"
        )
    if set(raw) != expected_fields:
        raise _AuthorityValidationError(
            "finalization-successor authorization has an unexpected top-level field set"
        )
    if raw.get("outcomes_inspected") is not False:
        raise _AuthorityValidationError(
            "finalization-only successor requires outcomes_inspected=false"
        )
    if raw.get("successor_evidence_filename") != _FINALIZATION_SUCCESSOR_EVIDENCE_FILENAME:
        raise _AuthorityValidationError("finalization successor evidence filename is invalid")
    if raw.get("successor_completion") != {_FINALIZATION_SUCCESSOR_COMPLETION_FLAG: True}:
        raise _AuthorityValidationError("finalization successor completion flag is invalid")
    if raw.get("successor_output_policy") != _FINALIZATION_OUTPUT_POLICY:
        raise _AuthorityValidationError("finalization successor output policy is invalid")

    predecessor = raw.get("predecessor")
    if not isinstance(predecessor, Mapping) or set(predecessor) != {
        "run_id",
        "run_directory",
        "terminal_status",
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "registration_authority",
    }:
        raise _AuthorityValidationError("finalization predecessor binding has an invalid field set")
    run_id = predecessor.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id.strip() != run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        raise _AuthorityValidationError("predecessor run_id must be one safe exact run ID")
    run_directory = _absolute_record_path(
        predecessor.get("run_directory"), role="predecessor run directory"
    )
    if run_directory.name != run_id:
        raise _AuthorityValidationError("predecessor run directory must end in its exact run ID")
    if predecessor.get("terminal_status") != "failed":
        raise _AuthorityValidationError("finalization predecessor terminal status must be failed")

    authority = predecessor.get("registration_authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "directory",
        "kind",
        "artifact_root_sha256",
        "sha256_manifest_sha256",
    }:
        raise _AuthorityValidationError(
            "predecessor registration-authority binding has an invalid field set"
        )
    authority_directory = _absolute_record_path(
        authority.get("directory"), role="predecessor registration authority"
    )
    if authority_directory != parent_state.directory:
        raise _AuthorityValidationError(
            "finalization predecessor computation authority must be the immediate amendment parent"
        )
    if (
        authority.get("kind") != parent_state.kind
        or authority.get("artifact_root_sha256") != parent_state.artifact_root_sha256
        or authority.get("sha256_manifest_sha256") != parent_state.sha256_manifest_sha256
    ):
        raise _AuthorityValidationError(
            "predecessor registration-authority hashes differ from the immediate parent"
        )
    parent_source = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_source, Mapping):
        raise _AuthorityValidationError("amendment parent lacks execution-source evidence")
    execution_root = _require_sha256(
        predecessor.get("execution_source_root_sha256"),
        role="predecessor execution-source root",
    )
    execution_manifest = _require_sha256(
        predecessor.get("execution_source_manifest_sha256"),
        role="predecessor execution-source manifest",
    )
    if execution_root != parent_source.get("root_sha256"):
        raise _AuthorityValidationError(
            "predecessor computation source differs from the immediate parent authority"
        )

    reuse = raw.get("reuse")
    if not isinstance(reuse, Mapping) or set(reuse) != {
        "reused_required_cell_count",
        "reused_optional_cell_count",
        "retrained_cell_count",
        "selection_policy",
        "predecessor_access_policy",
    }:
        raise _AuthorityValidationError("finalization reuse binding has an invalid field set")
    required_count = _strict_cell_count(
        reuse.get("reused_required_cell_count"), role="reused required-cell count", positive=True
    )
    optional_count = _strict_cell_count(
        reuse.get("reused_optional_cell_count"), role="reused optional-cell count", positive=False
    )
    if type(reuse.get("retrained_cell_count")) is not int or reuse.get("retrained_cell_count") != 0:
        raise _AuthorityValidationError(
            "finalization-only successor requires retrained_cell_count=0"
        )
    if reuse.get("selection_policy") != _FINALIZATION_SELECTION_POLICY:
        raise _AuthorityValidationError("finalization successor selection policy is invalid")
    if reuse.get("predecessor_access_policy") != _FINALIZATION_ACCESS_POLICY:
        raise _AuthorityValidationError("finalization predecessor access policy is invalid")

    numeric_verification = (
        _canonical_inherited_prior_numeric_verification(
            raw["numeric_verification"],
            run_directory=run_directory,
            verify_live_predecessor=verify_live_predecessor,
        )
        if authorization_schema == 2
        else None
    )
    canonical = {
        "schema_version": authorization_schema,
        "policy": (
            _FINALIZATION_SUCCESSOR_POLICY_V2
            if authorization_schema == 2
            else _FINALIZATION_SUCCESSOR_POLICY
        ),
        "predecessor": {
            "run_id": run_id,
            "run_directory": str(run_directory),
            "terminal_status": "failed",
            "artifact_root_sha256": _require_sha256(
                predecessor.get("artifact_root_sha256"),
                role="predecessor artifact root",
            ),
            "artifact_manifest_sha256": _require_sha256(
                predecessor.get("artifact_manifest_sha256"),
                role="predecessor artifact manifest",
            ),
            "execution_source_root_sha256": execution_root,
            "execution_source_manifest_sha256": execution_manifest,
            "registration_authority": {
                "directory": str(authority_directory),
                "kind": parent_state.kind,
                "artifact_root_sha256": parent_state.artifact_root_sha256,
                "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
            },
        },
        "reuse": {
            "reused_required_cell_count": required_count,
            "reused_optional_cell_count": optional_count,
            "retrained_cell_count": 0,
            "selection_policy": _FINALIZATION_SELECTION_POLICY,
            "predecessor_access_policy": _FINALIZATION_ACCESS_POLICY,
        },
        "outcomes_inspected": False,
        "successor_evidence_filename": _FINALIZATION_SUCCESSOR_EVIDENCE_FILENAME,
        "successor_completion": {_FINALIZATION_SUCCESSOR_COMPLETION_FLAG: True},
        "successor_output_policy": _FINALIZATION_OUTPUT_POLICY,
        **(
            {"numeric_verification": numeric_verification}
            if numeric_verification is not None
            else {}
        ),
    }
    if verify_live_predecessor:
        _validate_live_failed_predecessor(canonical)
    return canonical


def _canonical_confirmatory_storage_policy(
    value: ConfirmatoryStoragePolicy | Mapping[str, Any],
) -> dict[str, Any]:
    raw: Mapping[str, Any]
    if isinstance(value, ConfirmatoryStoragePolicy):
        raw = value.as_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("confirmatory_storage_policy must be a typed policy or mapping")
    expected = ConfirmatoryStoragePolicy().as_dict()
    if set(raw) != set(expected):
        raise _AuthorityValidationError("confirmatory storage policy has an unexpected field set")
    for key, expected_value in expected.items():
        observed_value = raw.get(key)
        if type(observed_value) is not type(expected_value) or observed_value != expected_value:
            raise _AuthorityValidationError(f"confirmatory storage policy field {key!r} is invalid")
    return expected


def _canonical_resource_bounded_confirmatory_authorization(
    value: ResourceBoundedConfirmatoryAuthorization | Mapping[str, Any],
    *,
    parent_state: _AuthorityState,
    resource_source: Mapping[str, Any],
    resource_source_manifest_sha256: str,
    resource_config_file_sha256: str,
    resource_config_semantic_sha256: str,
    verify_live_primary: bool,
) -> dict[str, Any]:
    raw: Mapping[str, Any]
    if isinstance(value, ResourceBoundedConfirmatoryAuthorization):
        raw = value.as_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(
            "resource_bounded_confirmatory_authorization must be a typed authorization or mapping"
        )
    expected_fields = {
        "schema_version",
        "policy",
        "purpose",
        "historical_primary",
        "resource_profile",
        "execution_source_delta",
        "resource_capacity_policy",
        "outcomes_inspected",
        "analysis_disposition",
        "outcome_use_policy",
        "original_confirmatory_claim_allowed",
        "study_outcome_eligible",
        "completion_stage",
        "primary_rebinding_allowed",
        "primary_mutation_allowed",
    }
    if set(raw) != expected_fields:
        raise _AuthorityValidationError(
            "resource-bounded confirmatory authorization has an unexpected field set"
        )
    if (
        raw.get("schema_version") != 1
        or raw.get("policy") != _RESOURCE_BOUNDED_CONFIRMATORY_POLICY
        or raw.get("purpose") != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        or raw.get("outcomes_inspected") is not True
        or raw.get("analysis_disposition") != _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION
        or raw.get("outcome_use_policy") != _RESOURCE_BOUNDED_OUTCOME_USE_POLICY
        or raw.get("original_confirmatory_claim_allowed") is not False
        or raw.get("study_outcome_eligible") is not False
        or raw.get("completion_stage") is not None
        or raw.get("primary_rebinding_allowed") is not False
        or raw.get("primary_mutation_allowed") is not False
    ):
        raise _AuthorityValidationError(
            "resource-bounded confirmatory authorization violates its fixed "
            "post-outcome exploratory policy"
        )

    recovery_authorization = _resource_parent_recovery_authorization(parent_state)
    recovery_authorization_sha256 = _canonical_mapping_sha256(recovery_authorization)
    historical = raw.get("historical_primary")
    expected_historical_fields = {
        "experiment_name",
        "run_id",
        "run_directory",
        "terminal_status",
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "completion_evidence_sha256",
        "primary_execution_gate_sha256",
        "stage_attestation_record_sha256",
        "stage_attestation_verification_sha256",
        "recovery_evidence_sha256",
        "recovery_authorization_sha256",
        "registration_authority",
    }
    if not isinstance(historical, Mapping) or set(historical) != expected_historical_fields:
        raise _AuthorityValidationError(
            "resource-bounded historical primary has an invalid field set"
        )
    run_id = historical.get("run_id")
    run_directory = _absolute_record_path(
        historical.get("run_directory"),
        role="resource-bounded historical primary run",
    )
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id.strip() != run_id
        or run_directory.name != run_id
        or historical.get("experiment_name") != _RESOURCE_BOUNDED_PRIMARY_EXPERIMENT_NAME
        or historical.get("terminal_status") != "completed"
    ):
        raise _AuthorityValidationError("resource-bounded historical primary identity is invalid")
    historical_hash_fields = (
        "artifact_root_sha256",
        "artifact_manifest_sha256",
        "completion_evidence_sha256",
        "primary_execution_gate_sha256",
        "stage_attestation_record_sha256",
        "stage_attestation_verification_sha256",
        "recovery_evidence_sha256",
        "recovery_authorization_sha256",
    )
    historical_hashes = {
        field: _require_sha256(
            historical.get(field),
            role=f"resource-bounded historical primary {field}",
        )
        for field in historical_hash_fields
    }
    if historical_hashes["recovery_authorization_sha256"] != recovery_authorization_sha256:
        raise _AuthorityValidationError(
            "resource-bounded historical primary recovery authorization differs "
            "from the direct parent"
        )
    registration = historical.get("registration_authority")
    expected_registration = {
        "directory": str(parent_state.directory),
        "kind": _AUTHORITY_KIND,
        "artifact_root_sha256": parent_state.artifact_root_sha256,
        "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
        "chain_depth": parent_state.chain_depth,
    }
    if not isinstance(registration, Mapping) or dict(registration) != expected_registration:
        raise _AuthorityValidationError(
            "resource-bounded historical primary registration authority differs "
            "from the direct recovery parent"
        )

    profile = raw.get("resource_profile")
    expected_profile_fields = {
        "profile_id",
        "experiment_name",
        "parent_confirmatory_config_file_sha256",
        "parent_confirmatory_config_semantic_sha256",
        "resource_confirmatory_config_file_sha256",
        "resource_confirmatory_config_semantic_sha256",
    }
    if not isinstance(profile, Mapping) or set(profile) != expected_profile_fields:
        raise _AuthorityValidationError(
            "resource-bounded confirmatory profile has an invalid field set"
        )
    profile_id = profile.get("profile_id")
    if (
        not isinstance(profile_id, str)
        or _RESOURCE_PROFILE_ID.fullmatch(profile_id) is None
        or profile_id != _RESOURCE_BOUNDED_PROFILE_ID
        or profile.get("experiment_name") != _RESOURCE_BOUNDED_EXPERIMENT_NAME
    ):
        raise _AuthorityValidationError("resource-bounded confirmatory profile identity is invalid")
    parent_confirmatory = parent_state.snapshot_hashes.get("confirmatory_config")
    if not isinstance(parent_confirmatory, Mapping):
        raise _AuthorityValidationError(
            "resource-bounded parent lacks confirmatory-config evidence"
        )
    expected_parent_config_file = _require_sha256(
        parent_confirmatory.get("file_sha256"),
        role="resource-bounded parent confirmatory config file",
    )
    expected_parent_config_semantic = _require_sha256(
        parent_confirmatory.get("semantic_sha256"),
        role="resource-bounded parent confirmatory config semantics",
    )
    expected_profile = {
        "profile_id": profile_id,
        "experiment_name": _RESOURCE_BOUNDED_EXPERIMENT_NAME,
        "parent_confirmatory_config_file_sha256": expected_parent_config_file,
        "parent_confirmatory_config_semantic_sha256": expected_parent_config_semantic,
        "resource_confirmatory_config_file_sha256": _require_sha256(
            resource_config_file_sha256,
            role="resource-bounded confirmatory config file",
        ),
        "resource_confirmatory_config_semantic_sha256": _require_sha256(
            resource_config_semantic_sha256,
            role="resource-bounded confirmatory config semantics",
        ),
    }
    if dict(profile) != expected_profile:
        raise _AuthorityValidationError(
            "resource-bounded confirmatory profile differs from its exact snapshots"
        )
    if (
        expected_profile["resource_confirmatory_config_semantic_sha256"]
        != _RESOURCE_BOUNDED_CONFIG_SEMANTIC_SHA256
    ):
        raise _AuthorityValidationError(
            "resource-bounded confirmatory config is not the exact registered resource profile"
        )
    if (
        expected_profile["resource_confirmatory_config_file_sha256"] == expected_parent_config_file
        or expected_profile["resource_confirmatory_config_semantic_sha256"]
        == expected_parent_config_semantic
    ):
        raise _AuthorityValidationError(
            "resource-bounded confirmatory profile must be an actual file and semantic change"
        )

    parent_source_path = parent_state.directory / _SOURCE_TREE_SNAPSHOT
    parent_source = _json_mapping(
        parent_source_path,
        role="resource-bounded parent execution source",
    )
    canonical_delta, delta_sha256 = _canonical_resource_source_delta(
        parent_source,
        resource_source,
    )
    parent_source_hashes = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_source_hashes, Mapping):
        raise _AuthorityValidationError("resource-bounded parent lacks execution-source evidence")
    source_delta = raw.get("execution_source_delta")
    expected_source_fields = {
        "policy",
        "parent_root_sha256",
        "parent_manifest_sha256",
        "resource_root_sha256",
        "resource_manifest_sha256",
        "allowlisted_changes",
        "delta_sha256",
    }
    if not isinstance(source_delta, Mapping) or set(source_delta) != expected_source_fields:
        raise _AuthorityValidationError(
            "resource-bounded execution-source delta has an invalid field set"
        )
    expected_source_delta = {
        "policy": _RESOURCE_BOUNDED_SOURCE_DELTA_POLICY,
        "parent_root_sha256": _require_sha256(
            parent_source_hashes.get("root_sha256"),
            role="resource-bounded parent execution-source root",
        ),
        "parent_manifest_sha256": _require_sha256(
            parent_source_hashes.get("manifest_sha256"),
            role="resource-bounded parent execution-source manifest",
        ),
        "resource_root_sha256": _require_sha256(
            resource_source.get("root_sha256"),
            role="resource-bounded execution-source root",
        ),
        "resource_manifest_sha256": _require_sha256(
            resource_source_manifest_sha256,
            role="resource-bounded execution-source manifest",
        ),
        "allowlisted_changes": [dict(record) for record in canonical_delta],
        "delta_sha256": delta_sha256,
    }
    if dict(source_delta) != expected_source_delta:
        raise _AuthorityValidationError(
            "resource-bounded execution-source delta differs from the recomputed closed allowlist"
        )
    capacity = raw.get("resource_capacity_policy")
    if (
        not isinstance(capacity, Mapping)
        or set(capacity) != set(_RESOURCE_BOUNDED_CAPACITY)
        or any(
            type(capacity.get(field)) is not type(expected) or capacity.get(field) != expected
            for field, expected in _RESOURCE_BOUNDED_CAPACITY.items()
        )
    ):
        raise _AuthorityValidationError(
            "resource-bounded capacity policy differs from its exact audited bound"
        )

    canonical: dict[str, Any] = {
        "schema_version": 1,
        "policy": _RESOURCE_BOUNDED_CONFIRMATORY_POLICY,
        "purpose": RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE,
        "historical_primary": {
            "experiment_name": _RESOURCE_BOUNDED_PRIMARY_EXPERIMENT_NAME,
            "run_id": run_id,
            "run_directory": str(run_directory),
            "terminal_status": "completed",
            **historical_hashes,
            "registration_authority": expected_registration,
        },
        "resource_profile": expected_profile,
        "execution_source_delta": expected_source_delta,
        "resource_capacity_policy": dict(_RESOURCE_BOUNDED_CAPACITY),
        "outcomes_inspected": True,
        "analysis_disposition": _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION,
        "outcome_use_policy": _RESOURCE_BOUNDED_OUTCOME_USE_POLICY,
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "primary_rebinding_allowed": False,
        "primary_mutation_allowed": False,
    }
    if verify_live_primary:
        _validate_live_resource_primary_binding(
            canonical["historical_primary"],
            parent_state=parent_state,
            recovery_authorization_sha256=recovery_authorization_sha256,
        )
    return canonical


def _canonical_failed_resource_preflight(
    value: Mapping[str, Any],
    *,
    resource_parent: _AuthorityState,
    resource_authorization_sha256: str,
    verify_live_receipt: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "receipt_path",
        "receipt_sha256",
        "evidence",
    }:
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight receipt has an invalid field set"
        )
    receipt_path = _absolute_record_path(
        value.get("receipt_path"),
        role="resource technical successor failed-preflight receipt",
    )
    receipt_sha256 = _require_sha256(
        value.get("receipt_sha256"),
        role="resource technical successor failed-preflight receipt",
    )
    evidence = value.get("evidence")
    expected_evidence_fields = {
        "schema_version",
        "policy",
        "invoked_at_utc",
        "observed_at_utc",
        "invocation_cell",
        "exit_code",
        "wall_seconds",
        "operation",
        "status",
        "status_error",
        "preflight_only",
        "run_mode",
        "retry_of_run_id",
        "tracked_failure_run_id",
        "tracked_failure_run_directory",
        "authority",
        "scenario_id",
        "cache_provenance_id",
        "failure_type",
        "failure_message",
        "authority_execution_cnn_source_sha256",
        "logical_provenance_source_sha256",
        "run_tracker_created",
        "run_directory_created",
        "registry_row_created",
        "stage_record_created",
        "disposition_record_created",
        "training_invoked",
        "matrix_executor_invoked",
        "lock_retained",
        "input_cache_modified",
        "authority_modified",
        "outcomes_inspected",
        "outcomes_inspected_at_utc",
        "scientific_outcome_values_read",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_fields:
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight evidence has an invalid field set"
        )
    authority = evidence.get("authority")
    expected_authority = {
        "directory": str(resource_parent.directory),
        "artifact_root_sha256": resource_parent.artifact_root_sha256,
        "sha256_manifest_sha256": resource_parent.sha256_manifest_sha256,
        "chain_depth": resource_parent.chain_depth,
        "authorization_sha256": resource_authorization_sha256,
    }
    false_fields = (
        "run_tracker_created",
        "run_directory_created",
        "registry_row_created",
        "stage_record_created",
        "disposition_record_created",
        "training_invoked",
        "matrix_executor_invoked",
        "lock_retained",
        "input_cache_modified",
        "authority_modified",
        "outcomes_inspected",
        "scientific_outcome_values_read",
    )
    if (
        evidence.get("schema_version") != 1
        or evidence.get("policy") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_POLICY
        or evidence.get("invoked_at_utc") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_INVOKED_AT
        or evidence.get("observed_at_utc") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_OBSERVED_AT
        or evidence.get("invocation_cell") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_INVOCATION_CELL
        or evidence.get("exit_code") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_EXIT_CODE
        or type(evidence.get("wall_seconds")) is not float
        or evidence.get("wall_seconds") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_WALL_SECONDS
        or evidence.get("operation") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_OPERATION
        or evidence.get("status") != "preflight_failed"
        or evidence.get("status_error") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_STATUS_ERROR
        or evidence.get("preflight_only") is not True
        or evidence.get("run_mode") != "fresh"
        or evidence.get("retry_of_run_id") is not None
        or evidence.get("tracked_failure_run_id") is not None
        or evidence.get("tracked_failure_run_directory") is not None
        or evidence.get("outcomes_inspected_at_utc") is not None
        or evidence.get("scenario_id") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID
        or evidence.get("cache_provenance_id") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID
        or evidence.get("failure_type") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_TYPE
        or evidence.get("failure_message") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_ERROR_MESSAGE
        or not isinstance(authority, Mapping)
        or dict(authority) != expected_authority
        or any(evidence.get(field) is not False for field in false_fields)
    ):
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight evidence is not the exact "
            "no-tracker CNN-provenance failure"
        )
    invoked_at = _parse_timestamp(
        evidence.get("invoked_at_utc"),
        role="resource technical successor failed-preflight invocation timestamp",
    )
    observed_at = _parse_timestamp(
        evidence.get("observed_at_utc"),
        role="resource technical successor failed-preflight timestamp",
    )
    if invoked_at >= observed_at:
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight timestamps are not ordered"
        )
    for field in (
        "authority_execution_cnn_source_sha256",
        "logical_provenance_source_sha256",
    ):
        _require_sha256(
            evidence.get(field),
            role=f"resource technical successor failed-preflight {field}",
        )
    canonical = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "evidence": dict(evidence),
    }
    if verify_live_receipt:
        receipt_bytes = _require_regular_file(
            receipt_path,
            role="resource technical successor failed-preflight receipt",
        )
        try:
            live_receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise _AuthorityValidationError(
                "resource technical successor failed-preflight receipt is invalid JSON"
            ) from exc
        if (
            not isinstance(live_receipt, Mapping)
            or dict(live_receipt) != dict(evidence)
            or sha256_file(receipt_path) != receipt_sha256
        ):
            raise _AuthorityValidationError(
                "resource technical successor failed-preflight receipt differs from "
                "its sealed evidence"
            )
    return canonical


def _canonical_prior_resource_authority_d_publication_failure(
    value: Mapping[str, Any],
    *,
    resource_parent: _AuthorityState,
    verify_live_receipt: bool,
) -> dict[str, Any]:
    """Bind the consumed, rolled-back v1 D publication before replacement D.

    The receipt is historical evidence only. It never authorizes an automatic
    retry, never makes the failed destination an authority, and never compares the
    old v2 source bundle with the replacement source.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "receipt_path",
        "receipt_sha256",
        "evidence",
    }:
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure has an invalid field set"
        )
    artifacts_root = Path(os.path.abspath(resource_parent.directory.parent.parent))
    control_root = artifacts_root / "resource_control"
    run_state_root = artifacts_root / "runs"
    frozen_input_root = control_root / "authority_d_inputs_20260727Tfinal_source_v2"
    expected_receipt_path = (
        control_root / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    )
    receipt_path = _absolute_lexical_record_path(
        value.get("receipt_path"),
        role="resource technical successor prior-publication failure receipt",
    )
    receipt_sha256 = _require_sha256(
        value.get("receipt_sha256"),
        role="resource technical successor prior-publication failure receipt",
    )
    if receipt_path != expected_receipt_path:
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure receipt uses "
            "the wrong canonical path"
        )
    evidence = value.get("evidence")
    expected_evidence_fields = {
        "schema_version",
        "policy",
        "observed_at_utc",
        "attempt_marker",
        "failure_marker",
        "terminal_stdout",
        "terminal_stderr",
        "executed_controller",
        "frozen_v2_inputs",
        "error",
        "run_state",
        "absent_paths",
        "disposition",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_fields:
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure evidence has "
            "an invalid field set"
        )
    observed_at = _parse_timestamp(
        evidence.get("observed_at_utc"),
        role="resource technical successor prior-publication failure observation",
    )
    canonical_observed_at = observed_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or evidence.get("policy") != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_POLICY
        or evidence.get("observed_at_utc") != canonical_observed_at
        or receipt_sha256 != _atomic_json_sha256(evidence)
    ):
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure receipt identity is invalid"
        )

    def canonical_file_record(
        record_value: Any,
        *,
        expected_path: Path,
        expected_size: int,
        expected_sha256: str,
        role: str,
        embedded_json: bool,
        canonical_json_bytes: bool = True,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        expected_fields = {"path", "size_bytes", "sha256"}
        if embedded_json:
            expected_fields.add("payload")
        if not isinstance(record_value, Mapping) or set(record_value) != expected_fields:
            raise _AuthorityValidationError(f"{role} has an invalid field set")
        record_path = _absolute_lexical_record_path(record_value.get("path"), role=role)
        record_sha256 = _require_sha256(record_value.get("sha256"), role=role)
        if (
            record_path != expected_path
            or type(record_value.get("size_bytes")) is not int
            or record_value.get("size_bytes") != expected_size
            or record_sha256 != expected_sha256
        ):
            raise _AuthorityValidationError(f"{role} differs from its exact identity")
        payload_value = record_value.get("payload")
        if embedded_json and not isinstance(payload_value, Mapping):
            raise _AuthorityValidationError(f"{role} embedded payload must be a JSON object")
        if (
            embedded_json
            and canonical_json_bytes
            and _atomic_json_sha256(payload_value) != expected_sha256
        ):
            raise _AuthorityValidationError(f"{role} embedded payload differs from its exact bytes")
        canonical_payload = dict(payload_value) if isinstance(payload_value, Mapping) else None
        canonical_record: dict[str, Any] = {
            "path": str(record_path),
            "size_bytes": expected_size,
            "sha256": expected_sha256,
        }
        if embedded_json:
            assert canonical_payload is not None
            canonical_record["payload"] = canonical_payload
        if verify_live_receipt:
            live_bytes = _read_stable_single_link_file(
                record_path,
                role=role,
                allow_empty=allow_empty,
            )
            if (
                len(live_bytes) != expected_size
                or hashlib.sha256(live_bytes).hexdigest() != expected_sha256
            ):
                raise _AuthorityValidationError(f"{role} live bytes changed")
            if embedded_json:
                try:
                    live_payload = json.loads(live_bytes.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise _AuthorityValidationError(f"{role} live payload is invalid JSON") from exc
                if (
                    not isinstance(live_payload, Mapping)
                    or canonical_payload is None
                    or dict(live_payload) != canonical_payload
                ):
                    raise _AuthorityValidationError(
                        f"{role} live payload differs from embedded evidence"
                    )
        return canonical_record

    expected_attempt_path = control_root / "resource_authority_d_publication_attempt.json"
    expected_failure_path = control_root / "resource_authority_d_publication_failure.json"
    expected_stdout_path = control_root / "authority_d_v2_publish_20260727T212335.140Z.stdout.log"
    expected_stderr_path = control_root / "authority_d_v2_publish_20260727T212335.140Z.stderr.log"
    expected_controller_path = control_root / "prepare_resource_authority_d_once.py"
    attempt_record = canonical_file_record(
        evidence.get("attempt_marker"),
        expected_path=expected_attempt_path,
        expected_size=1793,
        expected_sha256=_RESOURCE_BOUNDED_PRIOR_PUBLICATION_ATTEMPT_SHA256,
        role="resource authority D prior publication attempt marker",
        embedded_json=True,
    )
    failure_record = canonical_file_record(
        evidence.get("failure_marker"),
        expected_path=expected_failure_path,
        expected_size=2026,
        expected_sha256=_RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_SHA256,
        role="resource authority D prior publication failure marker",
        embedded_json=True,
    )
    stdout_record = canonical_file_record(
        evidence.get("terminal_stdout"),
        expected_path=expected_stdout_path,
        expected_size=2292,
        expected_sha256=_RESOURCE_BOUNDED_PRIOR_PUBLICATION_STDOUT_SHA256,
        role="resource authority D prior publication stdout",
        embedded_json=True,
        canonical_json_bytes=False,
    )
    stderr_record = canonical_file_record(
        evidence.get("terminal_stderr"),
        expected_path=expected_stderr_path,
        expected_size=0,
        expected_sha256=_RESOURCE_BOUNDED_PRIOR_PUBLICATION_STDERR_SHA256,
        role="resource authority D prior publication stderr",
        embedded_json=False,
        allow_empty=True,
    )
    controller_record = canonical_file_record(
        evidence.get("executed_controller"),
        expected_path=expected_controller_path,
        expected_size=74144,
        expected_sha256=_RESOURCE_BOUNDED_PRIOR_PUBLICATION_CONTROLLER_SHA256,
        role="resource authority D prior publication controller",
        embedded_json=False,
    )

    attempt = attempt_record["payload"]
    failure = failure_record["payload"]
    stdout_payload = stdout_record["payload"]
    expected_attempt_fields = {
        "attempt_timestamp_utc",
        "automatic_retry_allowed",
        "cnn_correction_receipt_sha256",
        "execution_source_delta_sha256",
        "execution_source_root_sha256",
        "failed_preflight_receipt_sha256",
        "frozen_source_receipt_sha256",
        "intended_authority_directory",
        "parent_authority_directory",
        "policy",
        "run_state_before",
        "schema_version",
        "source_allowlist_sha256",
        "workspace_plan_sha256",
    }
    expected_failure_fields = {
        "attempt_marker_path",
        "attempt_marker_sha256",
        "automatic_retry_allowed",
        "build_call_count",
        "create_call_count",
        "error_sha256",
        "error_type_sha256",
        "intended_authority_directory",
        "phase",
        "run_state_after",
        "run_state_before",
        "run_state_unchanged",
        "schema_version",
        "status",
    }
    failed_destination = resource_parent.directory.parent / "20260727T212711.019137Z"
    run_state = attempt.get("run_state_before")
    if (
        set(attempt) != expected_attempt_fields
        or type(attempt.get("schema_version")) is not int
        or attempt.get("schema_version") != 1
        or attempt.get("policy") != "single_schema_v5_resource_authority_d_attempt_v1"
        or attempt.get("attempt_timestamp_utc") != "2026-07-27T21:27:11.019137Z"
        or attempt.get("automatic_retry_allowed") is not False
        or attempt.get("parent_authority_directory") != str(resource_parent.directory)
        or attempt.get("intended_authority_directory") != str(failed_destination)
        or not isinstance(run_state, Mapping)
        or _canonical_mapping_sha256(run_state)
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_RUN_STATE_SHA256
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication attempt is not the exact "
            "consumed one-shot attempt"
        )
    if (
        set(failure) != expected_failure_fields
        or type(failure.get("schema_version")) is not int
        or failure.get("schema_version") != 1
        or failure.get("status") != "failed_no_retry"
        or failure.get("phase") != "create_schema_v5_authority"
        or failure.get("automatic_retry_allowed") is not False
        or failure.get("build_call_count") != 1
        or failure.get("create_call_count") != 1
        or failure.get("run_state_unchanged") is not True
        or failure.get("attempt_marker_path") != str(expected_attempt_path)
        or failure.get("attempt_marker_sha256")
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ATTEMPT_SHA256
        or failure.get("intended_authority_directory") != str(failed_destination)
        or failure.get("run_state_before") != run_state
        or failure.get("run_state_after") != run_state
        or failure.get("error_type_sha256") != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_TYPE_SHA256
        or failure.get("error_sha256") != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_SHA256
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication failure is not the exact "
            "terminal failed-no-retry state"
        )
    expected_stdout_payload = {
        **dict(failure),
        "failure_marker_path": str(expected_failure_path),
        "failure_marker_sha256": _RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_SHA256,
    }
    if _canonical_value_sha256(stdout_payload) != _canonical_value_sha256(expected_stdout_payload):
        raise _AuthorityValidationError(
            "resource authority D prior publication stdout differs from its terminal failure marker"
        )

    frozen_v2 = evidence.get("frozen_v2_inputs")
    if not isinstance(frozen_v2, Mapping) or set(frozen_v2) != {
        "directory",
        "records",
        "records_sha256",
    }:
        raise _AuthorityValidationError(
            "resource authority D prior frozen-v2 input inventory has an invalid field set"
        )
    if (
        _absolute_lexical_record_path(
            frozen_v2.get("directory"),
            role="resource authority D prior frozen-v2 input directory",
        )
        != frozen_input_root
    ):
        raise _AuthorityValidationError(
            "resource authority D prior frozen-v2 input directory changed"
        )
    frozen_records_value = frozen_v2.get("records")
    if not isinstance(frozen_records_value, list) or len(frozen_records_value) != 4:
        raise _AuthorityValidationError(
            "resource authority D prior frozen-v2 inventory must contain four files"
        )
    frozen_records: list[dict[str, Any]] = []
    for record_value, (name, (size_bytes, sha256)) in zip(
        frozen_records_value,
        sorted(_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_FILES.items()),
        strict=True,
    ):
        frozen_records.append(
            canonical_file_record(
                record_value,
                expected_path=frozen_input_root / name,
                expected_size=size_bytes,
                expected_sha256=sha256,
                role=f"resource authority D prior frozen-v2 input {name}",
                embedded_json=False,
            )
        )
    if (
        frozen_v2.get("records_sha256")
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_INVENTORY_SHA256
        or _canonical_value_sha256(frozen_records)
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_INVENTORY_SHA256
    ):
        raise _AuthorityValidationError(
            "resource authority D prior frozen-v2 inventory hash changed"
        )

    expected_error_message = (
        "published amendment failed independent verification: "
        "_AuthorityValidationError: resource authority C is historically valid "
        "but no longer the effective execution leaf; use successor D "
        f"{failed_destination}"
    )
    error = evidence.get("error")
    if (
        not isinstance(error, Mapping)
        or set(error)
        != {
            "exception_type",
            "exception_message",
            "exception_type_sha256",
            "exception_sha256",
        }
        or error.get("exception_type") != "RuntimeError"
        or error.get("exception_message") != expected_error_message
        or error.get("exception_type_sha256")
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_TYPE_SHA256
        or error.get("exception_sha256") != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_SHA256
        or hashlib.sha256(b"RuntimeError").hexdigest()
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_TYPE_SHA256
        or hashlib.sha256(f"RuntimeError: {expected_error_message}".encode()).hexdigest()
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_ERROR_SHA256
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication error binding changed"
        )

    run_state_evidence = evidence.get("run_state")
    if (
        not isinstance(run_state_evidence, Mapping)
        or set(run_state_evidence) != {"root", "files", "canonical_sha256"}
        or _absolute_lexical_record_path(
            run_state_evidence.get("root"),
            role="resource authority D prior publication run-state root",
        )
        != run_state_root
        or run_state_evidence.get("files") != run_state
        or run_state_evidence.get("canonical_sha256")
        != _RESOURCE_BOUNDED_PRIOR_PUBLICATION_RUN_STATE_SHA256
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication run-state binding changed"
        )
    expected_success_marker = control_root / "resource_authority_d_publication_success.json"
    absent_paths = evidence.get("absent_paths")
    if (
        not isinstance(absent_paths, Mapping)
        or set(absent_paths) != {"prior_success_marker", "failed_intended_authority"}
        or _absolute_lexical_record_path(
            absent_paths.get("prior_success_marker"),
            role="resource authority D prior success marker",
        )
        != expected_success_marker
        or _absolute_lexical_record_path(
            absent_paths.get("failed_intended_authority"),
            role="resource authority D failed intended authority",
        )
        != failed_destination
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication absence binding changed"
        )
    expected_disposition = {
        "prior_attempt_consumed": True,
        "prior_authority_published": False,
        "failed_intended_authority_absent": True,
        "prior_success_marker_absent": True,
        "replacement_mode": "manual_new_one_shot_after_rolled_back_publication",
        "automatic_retry_allowed": False,
        "scientific_outcome_values_read": False,
        "scientific_profile_changed": False,
    }
    if _canonical_value_sha256(evidence.get("disposition")) != _canonical_value_sha256(
        expected_disposition
    ):
        raise _AuthorityValidationError(
            "resource authority D prior publication disposition is not fail-closed"
        )
    attempt_at = _parse_timestamp(
        attempt.get("attempt_timestamp_utc"),
        role="resource authority D prior publication attempt timestamp",
    )
    if observed_at <= attempt_at:
        raise _AuthorityValidationError(
            "resource authority D prior publication receipt predates its attempt"
        )

    if verify_live_receipt:
        _require_real_directory(
            control_root,
            role="resource authority D prior publication control root",
        )
        live_frozen_root = _require_real_directory(
            frozen_input_root,
            role="resource authority D prior frozen-v2 input root",
        )

        def frozen_input_snapshot() -> tuple[Any, ...]:
            root_status = live_frozen_root.stat(follow_symlinks=False)
            children = tuple(
                sorted(
                    live_frozen_root.iterdir(),
                    key=lambda path: path.name,
                )
            )
            if tuple(path.name for path in children) != tuple(
                sorted(_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_FILES)
            ):
                raise _AuthorityValidationError(
                    "resource authority D prior frozen-v2 input set changed"
                )
            child_identities: list[tuple[Any, ...]] = []
            for path in children:
                expected_size, expected_sha256 = _RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_FILES[
                    path.name
                ]
                payload = _read_stable_single_link_file(
                    path,
                    role=f"resource authority D prior frozen-v2 input {path.name}",
                )
                observed_sha256 = hashlib.sha256(payload).hexdigest()
                if len(payload) != expected_size or observed_sha256 != expected_sha256:
                    raise _AuthorityValidationError(
                        "resource authority D prior frozen-v2 input changed from its "
                        f"exact historical bytes: {path.name}"
                    )
                child_identities.append(
                    (
                        path.name,
                        len(payload),
                        observed_sha256,
                    )
                )
            return (
                root_status.st_dev,
                root_status.st_ino,
                root_status.st_mtime_ns,
                root_status.st_ctime_ns,
                tuple(child_identities),
            )

        frozen_inputs_before = frozen_input_snapshot()

        def current_run_state() -> dict[str, str]:
            observed: dict[str, str] = {}
            for name, expected_sha256 in sorted(run_state.items()):
                if not isinstance(name, str) or not isinstance(expected_sha256, str):
                    raise _AuthorityValidationError(
                        "resource authority D prior run-state record is invalid"
                    )
                payload = _read_stable_single_link_file(
                    run_state_root / name,
                    role=f"resource authority D prior run-state file {name}",
                    allow_empty=True,
                )
                observed[name] = hashlib.sha256(payload).hexdigest()
            return observed

        run_state_before = current_run_state()
        if run_state_before != dict(run_state):
            raise _AuthorityValidationError(
                "resource authority D live run state changed after failed publication"
            )
        if os.path.lexists(expected_success_marker) or os.path.lexists(failed_destination):
            raise _AuthorityValidationError(
                "resource authority D failed publication is no longer terminally absent"
            )
        receipt_bytes = _read_stable_single_link_file(
            receipt_path,
            role="resource technical successor prior-publication failure receipt",
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
        try:
            live_receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise _AuthorityValidationError(
                "resource technical successor prior-publication failure receipt is invalid JSON"
            ) from exc
        if (
            hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256
            or not isinstance(live_receipt, Mapping)
            or dict(live_receipt) != dict(evidence)
            or current_run_state() != run_state_before
            or frozen_input_snapshot() != frozen_inputs_before
            or os.path.lexists(expected_success_marker)
            or os.path.lexists(failed_destination)
        ):
            raise _AuthorityValidationError(
                "resource authority D prior-publication live evidence changed during readback"
            )

    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "evidence": {
            "schema_version": 1,
            "policy": _RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_POLICY,
            "observed_at_utc": canonical_observed_at,
            "attempt_marker": attempt_record,
            "failure_marker": failure_record,
            "terminal_stdout": stdout_record,
            "terminal_stderr": stderr_record,
            "executed_controller": controller_record,
            "frozen_v2_inputs": {
                "directory": str(frozen_input_root),
                "records": frozen_records,
                "records_sha256": (_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_INVENTORY_SHA256),
            },
            "error": dict(error),
            "run_state": {
                "root": str(run_state_root),
                "files": dict(run_state),
                "canonical_sha256": (_RESOURCE_BOUNDED_PRIOR_PUBLICATION_RUN_STATE_SHA256),
            },
            "absent_paths": {
                "prior_success_marker": str(expected_success_marker),
                "failed_intended_authority": str(failed_destination),
            },
            "disposition": expected_disposition,
        },
    }


def _canonical_cnn_provenance_record(
    value: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_BOUNDED_CNN_PROVENANCE_FIELDS:
        raise _AuthorityValidationError(f"{role} has an invalid field set")
    record = dict(value)
    if (
        record.get("id") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID
        or record.get("representation_id") != "cnn_context_rgb_pixels"
        or record.get("status") != "available"
        or record.get("cache_file_sha256") is not None
        or record.get("encoder_identifier") != "resnet18_imagenet1k_v1"
        or record.get("weight_identifier") != "ResNet18_Weights.IMAGENET1K_V1"
        or record.get("preprocessing_identifier") != "confirmatory_context_rgb_224_v1"
        or record.get("input_variant") != "context_rgb"
    ):
        raise _AuthorityValidationError(f"{role} is not the exact context-RGB CNN record")
    for field in (
        "sidecar_semantic_sha256",
        "sample_order_sha256",
        "manifest_sha256",
        "encoder_metadata_sha256",
        "weights_sha256",
        "preprocessing_sha256",
    ):
        _require_sha256(record.get(field), role=f"{role} {field}")
    return record


def _execution_source_record_sha256(
    source: Mapping[str, Any],
    path: str,
    *,
    role: str,
) -> str:
    records = _canonical_execution_source_records(source, role=role)
    matches = [record for record in records if record["path"] == path]
    if len(matches) != 1:
        raise _AuthorityValidationError(f"{role} lacks exactly one {path!r} record")
    return _require_sha256(matches[0].get("sha256"), role=f"{role} {path}")


def _canonical_cnn_provenance_correction(
    value: Mapping[str, Any],
    *,
    before_config_record: Mapping[str, Any],
    after_config_record: Mapping[str, Any],
    parent_source: Mapping[str, Any],
    successor_source: Mapping[str, Any],
    failed_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
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
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise _AuthorityValidationError(
            "resource technical successor CNN correction has an invalid field set"
        )
    canonical_before_config = _canonical_cnn_provenance_record(
        before_config_record,
        role="resource authority C CNN provenance",
    )
    canonical_after_config = _canonical_cnn_provenance_record(
        after_config_record,
        role="resource technical successor D CNN provenance",
    )
    embedded_before = value.get("before_config_record")
    embedded_after = value.get("after_config_record")
    if (
        not isinstance(embedded_before, Mapping)
        or dict(embedded_before) != canonical_before_config
        or value.get("before_config_record_sha256")
        != _canonical_mapping_sha256(canonical_before_config)
        or not isinstance(embedded_after, Mapping)
        or dict(embedded_after) != canonical_after_config
        or value.get("after_config_record_sha256")
        != _canonical_mapping_sha256(canonical_after_config)
    ):
        raise _AuthorityValidationError(
            "resource technical successor CNN correction differs from the exact C/D configs"
        )
    before = value.get("before")
    after = value.get("after")
    cache = value.get("unchanged_cache_artifacts")
    if (
        not isinstance(before, Mapping)
        or set(before)
        != {
            "execution_cnn_source_sha256",
            "logical_provenance_source_sha256",
            "recomputed_record_sha256",
            "matches_config_record",
        }
        or not isinstance(after, Mapping)
        or set(after)
        != {
            "execution_cnn_source_sha256",
            "logical_provenance_source_sha256",
            "recomputed_record_sha256",
            "matches_config_record",
            "semantic_equivalence_evidence_sha256",
        }
        or not isinstance(cache, Mapping)
        or set(cache)
        != {
            "cache_file_sha256",
            "sidecar_file_sha256",
            "sidecar_semantic_sha256",
            "cache_bytes_changed",
            "sidecar_bytes_changed",
        }
    ):
        raise _AuthorityValidationError(
            "resource technical successor CNN correction before/after evidence is invalid"
        )
    before_config_sha256 = _canonical_mapping_sha256(canonical_before_config)
    after_config_sha256 = _canonical_mapping_sha256(canonical_after_config)
    failed_evidence = failed_preflight.get("evidence")
    if not isinstance(failed_evidence, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor CNN correction lacks failed-preflight evidence"
        )
    expected_before_execution_sha = _execution_source_record_sha256(
        parent_source,
        "src/histo_audit/models/cnn.py",
        role="resource authority C execution source",
    )
    expected_after_execution_sha = _execution_source_record_sha256(
        successor_source,
        "src/histo_audit/models/cnn.py",
        role="resource technical successor execution source",
    )
    before_hashes = {
        field: _require_sha256(
            before.get(field),
            role=f"resource technical successor CNN correction before {field}",
        )
        for field in (
            "execution_cnn_source_sha256",
            "logical_provenance_source_sha256",
            "recomputed_record_sha256",
        )
    }
    after_hashes = {
        field: _require_sha256(
            after.get(field),
            role=f"resource technical successor CNN correction after {field}",
        )
        for field in (
            "execution_cnn_source_sha256",
            "logical_provenance_source_sha256",
            "recomputed_record_sha256",
            "semantic_equivalence_evidence_sha256",
        )
    }
    cache_hashes = {
        field: _require_sha256(
            cache.get(field),
            role=f"resource technical successor CNN correction cache {field}",
        )
        for field in (
            "cache_file_sha256",
            "sidecar_file_sha256",
            "sidecar_semantic_sha256",
        )
    }
    if (
        value.get("schema_version") != 1
        or value.get("policy") != _RESOURCE_BOUNDED_CNN_PROVENANCE_CORRECTION_POLICY
        or value.get("scenario_id") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID
        or value.get("cache_provenance_id") != _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID
        or value.get("scientific_profile_changed") is not False
        or before.get("matches_config_record") is not False
        or after.get("matches_config_record") is not True
        or before_hashes["recomputed_record_sha256"] == before_config_sha256
        or after_hashes["recomputed_record_sha256"] != after_config_sha256
        or before_hashes["execution_cnn_source_sha256"] != expected_before_execution_sha
        or after_hashes["execution_cnn_source_sha256"] != expected_after_execution_sha
        or before_hashes["execution_cnn_source_sha256"]
        != failed_evidence.get("authority_execution_cnn_source_sha256")
        or before_hashes["logical_provenance_source_sha256"]
        != failed_evidence.get("logical_provenance_source_sha256")
        or cache_hashes["sidecar_semantic_sha256"]
        != canonical_before_config["sidecar_semantic_sha256"]
        or canonical_before_config["sidecar_semantic_sha256"]
        != canonical_after_config["sidecar_semantic_sha256"]
        or canonical_before_config["cache_file_sha256"]
        != canonical_after_config["cache_file_sha256"]
        or cache.get("cache_bytes_changed") is not False
        or cache.get("sidecar_bytes_changed") is not False
    ):
        raise _AuthorityValidationError(
            "resource technical successor CNN correction does not prove one failed "
            "before state, one matching after state, and unchanged cache artifacts"
        )
    return {
        "schema_version": 1,
        "policy": _RESOURCE_BOUNDED_CNN_PROVENANCE_CORRECTION_POLICY,
        "scenario_id": _RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID,
        "cache_provenance_id": _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID,
        "before_config_record": canonical_before_config,
        "before_config_record_sha256": before_config_sha256,
        "after_config_record": canonical_after_config,
        "after_config_record_sha256": after_config_sha256,
        "before": {**before_hashes, "matches_config_record": False},
        "after": {**after_hashes, "matches_config_record": True},
        "unchanged_cache_artifacts": {
            **cache_hashes,
            "cache_bytes_changed": False,
            "sidecar_bytes_changed": False,
        },
        "scientific_profile_changed": False,
    }


def _resource_config_cnn_record(resource_config: Mapping[str, Any]) -> dict[str, Any]:
    raw_records = resource_config.get("cache_provenance")
    if not isinstance(raw_records, list):
        raise _AuthorityValidationError(
            "resource technical successor config lacks cache provenance"
        )
    matches = [
        record
        for record in raw_records
        if isinstance(record, Mapping)
        and record.get("id") == _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID
    ]
    if len(matches) != 1:
        raise _AuthorityValidationError(
            "resource technical successor config lacks exactly one context-RGB CNN record"
        )
    return _canonical_cnn_provenance_record(
        matches[0],
        role="resource technical successor config CNN provenance",
    )


def _require_resource_technical_config_correction(
    parent_config: Mapping[str, Any],
    successor_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove that D changes exactly the two corrected logical CNN digests."""

    before_record = _resource_config_cnn_record(parent_config)
    after_record = _resource_config_cnn_record(successor_config)
    changed_record_fields = {
        field
        for field in _RESOURCE_BOUNDED_CNN_PROVENANCE_FIELDS
        if before_record[field] != after_record[field]
    }
    if changed_record_fields != _RESOURCE_BOUNDED_CNN_CONFIG_CORRECTION_FIELDS:
        raise _AuthorityValidationError(
            "resource technical successor config must change exactly the logical "
            "CNN encoder-metadata and preprocessing provenance digests"
        )

    normalized_successor = copy.deepcopy(dict(successor_config))
    raw_records = normalized_successor.get("cache_provenance")
    if not isinstance(raw_records, list):
        raise _AuthorityValidationError(
            "resource technical successor config lacks cache provenance"
        )
    matches = [
        record
        for record in raw_records
        if isinstance(record, dict)
        and record.get("id") == _RESOURCE_BOUNDED_FAILED_PREFLIGHT_CACHE_ID
    ]
    if len(matches) != 1:
        raise _AuthorityValidationError(
            "resource technical successor config lacks exactly one context-RGB CNN record"
        )
    for field in _RESOURCE_BOUNDED_CNN_CONFIG_CORRECTION_FIELDS:
        matches[0][field] = before_record[field]
    if normalized_successor != dict(parent_config):
        raise _AuthorityValidationError(
            "resource technical successor config changed outside the exact logical "
            "CNN provenance correction"
        )
    return before_record, after_record


def _canonical_resource_input_workspace_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize the provider-built, outcome-blind workspace recipe bound by D."""

    expected_fields = {
        "schema_version",
        "recipe_id",
        "workspace_reuse_allowed",
        "workspace_key",
        "source_row_count",
        "arrays",
        "partition_index_specs",
        "expected_extracted_file_bytes",
        "expected_raw_array_nbytes",
        "expected_index_npy_file_bytes",
        "expected_index_raw_nbytes",
        "metadata_capacity_allowance_bytes",
        "planned_workspace_bytes",
        "minimum_free_bytes_after",
        "maximum_workspace_bytes",
        "required_free_bytes_before",
        "plan_without_self_hash_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise _AuthorityValidationError(
            "resource input workspace plan has an invalid provider field set"
        )
    raw_arrays = value.get("arrays")
    raw_indices = value.get("partition_index_specs")
    source_row_count = value.get("source_row_count")
    if (
        value.get("schema_version") != 1
        or value.get("recipe_id") != "pannuke_confirmatory_shared_memmap_workspace_v1"
        or value.get("workspace_reuse_allowed") is not False
        or not isinstance(value.get("workspace_key"), str)
        or _SHA256.fullmatch(str(value.get("workspace_key"))) is None
        or type(source_row_count) is not int
        or source_row_count <= 0
        or not isinstance(raw_arrays, list)
        or len(raw_arrays) != _RESOURCE_BOUNDED_CAPACITY_V3["workspace_source_array_count"]
        or not isinstance(raw_indices, list)
        or len(raw_indices) != _RESOURCE_BOUNDED_CAPACITY_V3["workspace_partition_count"]
    ):
        raise _AuthorityValidationError(
            "resource input workspace plan lacks the exact recipe, 12 arrays, or "
            "nine partition-index specifications"
        )

    array_fields = {
        "array_id",
        "source_npz_sha256",
        "source_sidecar_sha256",
        "source_member_name",
        "source_member_crc32",
        "source_member_compression",
        "source_member_compressed_bytes",
        "source_member_uncompressed_bytes",
        "dtype",
        "shape",
        "raw_array_nbytes",
        "expected_array_sha256",
    }
    arrays: list[dict[str, Any]] = []
    for raw in raw_arrays:
        if not isinstance(raw, Mapping) or set(raw) != array_fields:
            raise _AuthorityValidationError(
                "resource input workspace array specification has an invalid field set"
            )
        array_id = raw.get("array_id")
        member = raw.get("source_member_name")
        shape = raw.get("shape")
        if (
            not isinstance(array_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", array_id) is None
            or not isinstance(member, str)
            or not member.endswith(".npy")
            or "/" in member
            or "\\" in member
            or not isinstance(raw.get("source_member_crc32"), str)
            or re.fullmatch(r"[0-9a-f]{8}", str(raw.get("source_member_crc32"))) is None
            or raw.get("source_member_compression") not in {"deflated", "stored"}
            or not isinstance(raw.get("dtype"), str)
            or not raw.get("dtype")
            or not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape[0] != source_row_count
        ):
            raise _AuthorityValidationError(
                "resource input workspace array specification has invalid typed metadata"
            )
        for field in (
            "source_npz_sha256",
            "source_sidecar_sha256",
            "expected_array_sha256",
        ):
            _require_sha256(
                raw.get(field),
                role=f"resource input workspace array {array_id} {field}",
            )
        for field in (
            "source_member_compressed_bytes",
            "source_member_uncompressed_bytes",
            "raw_array_nbytes",
        ):
            if type(raw.get(field)) is not int or int(raw[field]) <= 0:
                raise _AuthorityValidationError(
                    f"resource input workspace array {array_id} {field} must be positive"
                )
        if int(raw["raw_array_nbytes"]) > int(raw["source_member_uncompressed_bytes"]):
            raise _AuthorityValidationError(
                "resource input workspace raw array exceeds its NPY member bytes"
            )
        arrays.append(dict(raw))
    if [record["array_id"] for record in arrays] != sorted(
        {str(record["array_id"]) for record in arrays}
    ):
        raise _AuthorityValidationError(
            "resource input workspace arrays must have unique sorted IDs"
        )

    index_fields = {
        "outer_fold",
        "role",
        "row_count",
        "source_indices_sha256",
        "ordered_unique",
        "relative_path",
        "raw_nbytes",
        "npy_file_bytes",
    }
    expected_index_keys = {
        (fold, role)
        for fold in (1, 2, 3)
        for role in ("audit", "reference_validation", "final_reference")
    }
    indices: list[dict[str, Any]] = []
    observed_index_keys: set[tuple[int, str]] = set()
    for raw in raw_indices:
        if not isinstance(raw, Mapping) or set(raw) != index_fields:
            raise _AuthorityValidationError(
                "resource input workspace partition-index specification has an invalid field set"
            )
        fold = raw.get("outer_fold")
        role = raw.get("role")
        row_count = raw.get("row_count")
        raw_nbytes = raw.get("raw_nbytes")
        npy_file_bytes = raw.get("npy_file_bytes")
        if (
            type(fold) is not int
            or fold not in {1, 2, 3}
            or role not in {"audit", "reference_validation", "final_reference"}
            or type(row_count) is not int
            or row_count <= 0
            or raw.get("ordered_unique") is not True
            or raw.get("relative_path") != f"indices/fold_{fold}__{role}.npy"
            or type(raw_nbytes) is not int
            or raw_nbytes != row_count * 8
            or type(npy_file_bytes) is not int
        ):
            raise _AuthorityValidationError(
                "resource input workspace partition-index specification is invalid"
            )
        if npy_file_bytes <= raw_nbytes:
            raise _AuthorityValidationError(
                "resource input workspace partition-index NPY bytes are invalid"
            )
        _require_sha256(
            raw.get("source_indices_sha256"),
            role="resource input workspace partition-index vector",
        )
        key = (fold, str(role))
        if key in observed_index_keys:
            raise _AuthorityValidationError(
                "resource input workspace partition-index specification is duplicated"
            )
        observed_index_keys.add(key)
        indices.append(dict(raw))
    if (
        observed_index_keys != expected_index_keys
        or [(int(record["outer_fold"]), str(record["role"])) for record in indices]
        != sorted(expected_index_keys)
        or any(
            sum(int(record["row_count"]) for record in indices if record["outer_fold"] == fold)
            != source_row_count
            for fold in (1, 2, 3)
        )
    ):
        raise _AuthorityValidationError(
            "resource input workspace indices do not form the exact sorted 3x3 fold/role carrier"
        )

    integer_fields: dict[str, int] = {
        "expected_extracted_file_bytes": sum(
            int(record["source_member_uncompressed_bytes"]) for record in arrays
        ),
        "expected_raw_array_nbytes": sum(int(record["raw_array_nbytes"]) for record in arrays),
        "expected_index_npy_file_bytes": sum(int(record["npy_file_bytes"]) for record in indices),
        "expected_index_raw_nbytes": sum(int(record["raw_nbytes"]) for record in indices),
        "metadata_capacity_allowance_bytes": 16_777_216,
        "minimum_free_bytes_after": _RESOURCE_BOUNDED_CAPACITY_V3[
            "minimum_free_bytes_before_tracker"
        ],
        "maximum_workspace_bytes": _RESOURCE_BOUNDED_CAPACITY_V3["maximum_workspace_bytes"],
    }
    if any(
        type(value.get(field)) is not int or value.get(field) != expected
        for field, expected in integer_fields.items()
    ):
        raise _AuthorityValidationError(
            "resource input workspace plan differs from its exact array/capacity totals"
        )
    expected_extracted = integer_fields["expected_extracted_file_bytes"]
    planned_workspace = (
        expected_extracted
        + integer_fields["expected_index_npy_file_bytes"]
        + integer_fields["metadata_capacity_allowance_bytes"]
    )
    minimum_after = integer_fields["minimum_free_bytes_after"]
    if (
        expected_extracted != _RESOURCE_BOUNDED_CAPACITY_V3["workspace_shared_backing_bytes"]
        or integer_fields["expected_index_npy_file_bytes"]
        != _RESOURCE_BOUNDED_CAPACITY_V3["workspace_partition_index_bytes"]
        or expected_extracted + integer_fields["expected_index_npy_file_bytes"]
        != _RESOURCE_BOUNDED_CAPACITY_V3["projected_workspace_bytes"]
        or type(value.get("planned_workspace_bytes")) is not int
        or value.get("planned_workspace_bytes") != planned_workspace
        or planned_workspace > integer_fields["maximum_workspace_bytes"]
        or type(value.get("required_free_bytes_before")) is not int
        or value.get("required_free_bytes_before") != minimum_after + planned_workspace
    ):
        raise _AuthorityValidationError(
            "resource input workspace plan is inconsistent with capacity-v3"
        )
    canonical = {
        "schema_version": 1,
        "recipe_id": "pannuke_confirmatory_shared_memmap_workspace_v1",
        "workspace_reuse_allowed": False,
        "workspace_key": str(value["workspace_key"]),
        "source_row_count": source_row_count,
        "arrays": arrays,
        "partition_index_specs": indices,
        **integer_fields,
        "planned_workspace_bytes": planned_workspace,
        "required_free_bytes_before": minimum_after + planned_workspace,
    }
    expected_self_hash = _canonical_mapping_sha256(canonical)
    if value.get("plan_without_self_hash_sha256") != expected_self_hash:
        raise _AuthorityValidationError(
            "resource input workspace plan self-excluding SHA-256 is invalid"
        )
    return {**canonical, "plan_without_self_hash_sha256": expected_self_hash}


def _canonical_replacement_publication_terminal_qualification(
    value: Mapping[str, Any],
    *,
    resource_parent: _AuthorityState,
    verify_live_records: bool,
    verify_live_run_state: bool | None = None,
) -> dict[str, Any]:
    """Canonicalize the closed A1+F1/no-S1/no-D qualification receipt."""

    live_run_state = verify_live_records if verify_live_run_state is None else verify_live_run_state
    if live_run_state and not verify_live_records:
        raise _AuthorityValidationError(
            "replacement publication run-state live verification requires live Q records"
        )
    top_fields = {
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
    if type(value) is not dict or set(value) != top_fields:
        raise _AuthorityValidationError(
            "replacement publication terminal qualification has an invalid field set"
        )
    artifacts_root = Path(os.path.abspath(resource_parent.directory.parent.parent))
    project_root = artifacts_root.parent
    control_root = artifacts_root / "resource_control"
    run_state_root = artifacts_root / "runs"
    expected_resource_parent = (
        artifacts_root
        / "preregistration_amendments"
        / _RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_COMPONENT
    )
    if (
        resource_parent.directory != expected_resource_parent
        or resource_parent.kind != "preregistration_amendment"
        or type(resource_parent.chain_depth) is not int
        or resource_parent.chain_depth != 3
        or resource_parent.artifact_root_sha256
        != _RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_ARTIFACT_ROOT_SHA256
        or resource_parent.sha256_manifest_sha256
        != _RESOURCE_BOUNDED_REPLACEMENT_AUTHORITY_C_MANIFEST_SHA256
    ):
        raise _AuthorityValidationError(
            "replacement publication terminal qualification has the wrong Authority C"
        )
    consumed_controller_path = (
        project_root
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_controller.py"
    )
    qualifying_controller_path = (
        project_root
        / "src"
        / "histo_audit"
        / "workflows"
        / "resource_authority_d_replacement_v2_controller.py"
    )
    qualified_at = _parse_timestamp(
        value.get("qualified_at_utc"),
        role="replacement publication terminal-qualification timestamp",
    )
    canonical_qualified_at = qualified_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
    consumed_attempt_at = _parse_timestamp(
        _RESOURCE_BOUNDED_REPLACEMENT_TIMESTAMP,
        role="consumed replacement publication timestamp",
    )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("policy") != _RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_POLICY
        or value.get("status") != "qualified_rolled_back_failure_no_retry"
        or value.get("qualified_at_utc") != canonical_qualified_at
        or qualified_at <= consumed_attempt_at
        or (verify_live_records and qualified_at > datetime.now(UTC))
        or _exact_schema_v3_record_path(
            value.get("project_root"),
            expected=project_root,
            role="replacement publication terminal-qualification project root",
        )
        != project_root
        or value.get("outcome_value_interpretation_performed") is not False
        or value.get("scientific_execution_performed") is not False
        or value.get("publication_performed") is not False
    ):
        raise _AuthorityValidationError(
            "replacement publication terminal qualification violates its fixed policy"
        )

    def canonical_record(
        raw: Any,
        *,
        path: Path,
        size_bytes: int,
        sha256: str,
        role: str,
        read_live: bool = True,
    ) -> dict[str, Any]:
        if (
            type(raw) is not dict
            or set(raw) != {"path", "size_bytes", "sha256"}
            or _exact_schema_v3_record_path(
                raw.get("path"),
                expected=path,
                role=role,
            )
            != path
            or type(raw.get("size_bytes")) is not int
            or raw.get("size_bytes") != size_bytes
            or _require_sha256(raw.get("sha256"), role=role) != sha256
        ):
            raise _AuthorityValidationError(f"{role} differs from its exact identity")
        if verify_live_records and read_live:
            payload = _read_stable_single_link_file(
                path,
                role=role,
                max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
            )
            if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256:
                raise _AuthorityValidationError(f"{role} live bytes changed")
        return {
            "path": str(path),
            "size_bytes": size_bytes,
            "sha256": sha256,
        }

    def current_record(path: Path, *, role: str) -> dict[str, Any]:
        payload = _read_stable_single_link_file(
            path,
            role=role,
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
        return {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    authority = value.get("authority_c")
    authority_fields = {
        "directory",
        "schema_version",
        "chain_depth",
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "flat_file_count",
        "files",
        "integrity_verified",
    }
    authority_paths = {
        "amendment_evidence": resource_parent.directory / _EVIDENCE_FILENAME,
        "amendment_report": resource_parent.directory / _REPORT_FILENAME,
        "confirmatory_config": resource_parent.directory / _CONFIRMATORY_CONFIG_SNAPSHOT,
        "immutable_marker": resource_parent.directory / _IMMUTABLE_MARKER,
        "preregistration": resource_parent.directory / _PREREGISTRATION_SNAPSHOT,
        "primary_config": resource_parent.directory / _PRIMARY_CONFIG_SNAPSHOT,
        "sha256_manifest": resource_parent.directory / _MANIFEST_FILENAME,
        "source_tree_manifest": resource_parent.directory / _SOURCE_TREE_SNAPSHOT,
    }
    authority_files = {
        role: current_record(path, role=f"replacement qualification Authority C {role}")
        for role, path in authority_paths.items()
    }
    if (
        type(authority) is not dict
        or set(authority) != authority_fields
        or _exact_schema_v3_record_path(
            authority.get("directory"),
            expected=resource_parent.directory,
            role="replacement qualification Authority C",
        )
        != resource_parent.directory
        or type(authority.get("schema_version")) is not int
        or authority.get("schema_version") != 4
        or type(authority.get("chain_depth")) is not int
        or authority.get("chain_depth") != resource_parent.chain_depth
        or authority.get("artifact_root_sha256") != resource_parent.artifact_root_sha256
        or authority.get("sha256_manifest_sha256") != resource_parent.sha256_manifest_sha256
        or type(authority.get("flat_file_count")) is not int
        or authority.get("flat_file_count") != 8
        or type(authority.get("files")) is not dict
        or authority.get("files") != authority_files
        or authority.get("integrity_verified") is not True
    ):
        raise _AuthorityValidationError(
            "replacement publication qualification Authority C is not exact"
        )
    canonical_authority = {
        "directory": str(resource_parent.directory),
        "schema_version": 4,
        "chain_depth": resource_parent.chain_depth,
        "artifact_root_sha256": resource_parent.artifact_root_sha256,
        "sha256_manifest_sha256": resource_parent.sha256_manifest_sha256,
        "flat_file_count": 8,
        "files": authority_files,
        "integrity_verified": True,
    }

    authorization_path = (
        control_root / "resource_authority_d_replacement_publication_authorization_v1.json"
    )
    attempt_path = control_root / "resource_authority_d_replacement_v1_publication_attempt.json"
    failure_path = control_root / "resource_authority_d_replacement_v1_publication_failure.json"
    success_path = control_root / "resource_authority_d_replacement_v1_publication_success.json"
    intended_authority = artifacts_root / "preregistration_amendments" / "20260728T181920.303224Z"
    terminal = value.get("terminal_namespace")
    terminal_fields = {
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
    if type(terminal) is not dict or set(terminal) != terminal_fields:
        raise _AuthorityValidationError(
            "replacement publication terminal namespace has an invalid field set"
        )
    authorization_record = canonical_record(
        terminal.get("publication_authorization_receipt"),
        path=authorization_path,
        size_bytes=9396,
        sha256=_RESOURCE_BOUNDED_REPLACEMENT_AUTHORIZATION_SHA256,
        role="replacement publication authorization receipt",
    )
    attempt_record = canonical_record(
        terminal.get("attempt_marker"),
        path=attempt_path,
        size_bytes=3420,
        sha256=_RESOURCE_BOUNDED_REPLACEMENT_ATTEMPT_SHA256,
        role="replacement publication attempt marker",
    )
    failure_record = canonical_record(
        terminal.get("failure_marker"),
        path=failure_path,
        size_bytes=924,
        sha256=_RESOURCE_BOUNDED_REPLACEMENT_FAILURE_SHA256,
        role="replacement publication failure marker",
    )

    def canonical_absence(raw: Any, *, path: Path, role: str) -> dict[str, Any]:
        if (
            type(raw) is not dict
            or set(raw) != {"path", "absent"}
            or _exact_schema_v3_record_path(
                raw.get("path"),
                expected=path,
                role=role,
            )
            != path
            or raw.get("absent") is not True
        ):
            raise _AuthorityValidationError(f"{role} is not the exact absence record")
        if verify_live_records and os.path.lexists(path):
            raise _AuthorityValidationError(f"{role} no longer holds")
        return {"path": str(path), "absent": True}

    success_absence = canonical_absence(
        terminal.get("success_marker_absence"),
        path=success_path,
        role="replacement publication success-marker absence",
    )
    authority_absence = canonical_absence(
        terminal.get("intended_authority_absence"),
        path=intended_authority,
        role="replacement publication intended-authority absence",
    )
    if (
        terminal.get("classification") != "rolled_back_failure"
        or terminal.get("classification_reason") != "exact A+F exist and S/D/candidates are absent"
        or type(terminal.get("candidate_count")) is not int
        or terminal.get("candidate_count") != 0
        or type(terminal.get("candidate_paths")) is not list
        or terminal.get("candidate_paths") != []
    ):
        raise _AuthorityValidationError(
            "replacement publication namespace is not exact A+F/no-S/no-D"
        )
    canonical_terminal = {
        "classification": "rolled_back_failure",
        "classification_reason": "exact A+F exist and S/D/candidates are absent",
        "publication_authorization_receipt": authorization_record,
        "attempt_marker": attempt_record,
        "failure_marker": failure_record,
        "success_marker_absence": success_absence,
        "intended_authority_absence": authority_absence,
        "candidate_count": 0,
        "candidate_paths": [],
    }

    terminal_links = value.get("terminal_links")
    canonical_terminal_links = {
        "attempt_id": _RESOURCE_BOUNDED_REPLACEMENT_ATTEMPT_ID,
        "max_attempt_count": 1,
        "automatic_retry_allowed": False,
        "authorization_receipt_sha256": _RESOURCE_BOUNDED_REPLACEMENT_AUTHORIZATION_SHA256,
        "attempt_marker_sha256": _RESOURCE_BOUNDED_REPLACEMENT_ATTEMPT_SHA256,
        "technical_successor_authorization_sha256": (
            _RESOURCE_BOUNDED_REPLACEMENT_TECHNICAL_AUTHORIZATION_SHA256
        ),
        "intent_sha256": _RESOURCE_BOUNDED_REPLACEMENT_INTENT_SHA256,
        "preflight_fingerprint_sha256": (
            _RESOURCE_BOUNDED_REPLACEMENT_PREFLIGHT_FINGERPRINT_SHA256
        ),
        "amendment_timestamp_utc": _RESOURCE_BOUNDED_REPLACEMENT_TIMESTAMP,
        "parent_authority_directory": str(resource_parent.directory),
        "run_state_sha256": _RESOURCE_BOUNDED_REPLACEMENT_RUN_STATE_SHA256,
    }
    if (
        type(terminal_links) is not dict
        or set(terminal_links) != set(canonical_terminal_links)
        or type(terminal_links.get("max_attempt_count")) is not int
        or terminal_links.get("automatic_retry_allowed") is not False
        or dict(terminal_links) != canonical_terminal_links
    ):
        raise _AuthorityValidationError("replacement publication terminal links are not exact")

    v2_root = control_root / "authority_d_replacement_inputs_v2"
    v2_specs = {
        "cnn_correction_receipt": (
            v2_root / "authority_d_replacement_cnn_correction_receipt.json",
            4452,
            "0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e",
        ),
        "frozen_source_receipt": (
            v2_root / "authority_d_replacement_frozen_source_receipt.json",
            3943,
            "1acbcfd44b3f95d6387d7da573786547a6c1ff5dcd0d05b4198d311fbe813605",
        ),
        "source_allowlist": (
            v2_root / "authority_d_replacement_source_allowlist.json",
            3903,
            "397fac0240f36fb598095e7605dae770b55faf114d4c692e555a7101fd47c369",
        ),
        "workspace_plan": (
            v2_root / "authority_d_replacement_workspace_plan.json",
            12186,
            "d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b",
        ),
    }
    frozen = value.get("frozen_v2_inputs")
    frozen_fields = {
        "directory",
        "files",
        "execution_source_root_sha256",
        "execution_source_manifest_sha256",
        "execution_source_delta_sha256",
        "records_sha256",
    }
    raw_v2_files = frozen.get("files") if type(frozen) is dict else None
    if (
        type(frozen) is not dict
        or set(frozen) != frozen_fields
        or type(raw_v2_files) is not dict
        or set(raw_v2_files) != set(v2_specs)
    ):
        raise _AuthorityValidationError(
            "replacement publication frozen-v2 binding has an invalid field set"
        )
    canonical_v2_files = {
        role: canonical_record(
            raw_v2_files[role],
            path=path,
            size_bytes=size,
            sha256=sha256,
            role=f"replacement publication frozen-v2 {role}",
        )
        for role, (path, size, sha256) in v2_specs.items()
    }
    if (
        _exact_schema_v3_record_path(
            frozen.get("directory"),
            expected=v2_root,
            role="replacement publication frozen-v2 directory",
        )
        != v2_root
        or frozen.get("execution_source_root_sha256")
        != _RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_ROOT_SHA256
        or frozen.get("execution_source_manifest_sha256")
        != _RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_MANIFEST_SHA256
        or frozen.get("execution_source_delta_sha256")
        != _RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_DELTA_SHA256
        or frozen.get("records_sha256") != _RESOURCE_BOUNDED_REPLACEMENT_V2_RECORDS_SHA256
        or _canonical_mapping_sha256(canonical_v2_files)
        != _RESOURCE_BOUNDED_REPLACEMENT_V2_RECORDS_SHA256
    ):
        raise _AuthorityValidationError("replacement publication frozen-v2 binding is not exact")
    canonical_frozen = {
        "directory": str(v2_root),
        "files": canonical_v2_files,
        "execution_source_root_sha256": (_RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_ROOT_SHA256),
        "execution_source_manifest_sha256": (
            _RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_MANIFEST_SHA256
        ),
        "execution_source_delta_sha256": (_RESOURCE_BOUNDED_REPLACEMENT_V2_SOURCE_DELTA_SHA256),
        "records_sha256": _RESOURCE_BOUNDED_REPLACEMENT_V2_RECORDS_SHA256,
    }

    controllers = value.get("controller_identities")
    if type(controllers) is not dict or set(controllers) != {
        "consumed_attempt_controller",
        "diagnosed_fixed_legacy_controller",
        "qualifying_live_controller",
    }:
        raise _AuthorityValidationError(
            "replacement publication controller identities have an invalid field set"
        )
    consumed_controller = controllers.get("consumed_attempt_controller")
    consumed_attestations = [
        "retired_v1_invalidation_receipt",
        "attempt_marker",
        "publication_authorization_receipt",
        "v2_frozen_source_receipt",
        "v2_source_allowlist",
    ]
    if (
        type(consumed_controller) is not dict
        or set(consumed_controller)
        != {"path", "size_bytes", "sha256", "live_file_match", "attested_by"}
        or _exact_schema_v3_record_path(
            consumed_controller.get("path"),
            expected=consumed_controller_path,
            role="consumed replacement publication controller",
        )
        != consumed_controller_path
        or type(consumed_controller.get("size_bytes")) is not int
        or consumed_controller.get("size_bytes") != 216288
        or consumed_controller.get("sha256")
        != _RESOURCE_BOUNDED_REPLACEMENT_CONSUMED_CONTROLLER_SHA256
        or consumed_controller.get("live_file_match") is not False
        or type(consumed_controller.get("attested_by")) is not list
        or consumed_controller.get("attested_by") != consumed_attestations
    ):
        raise _AuthorityValidationError(
            "consumed replacement controller is not historically pinned"
        )
    diagnosed_controller = controllers.get("diagnosed_fixed_legacy_controller")
    diagnosed_fields = {
        "path",
        "size_bytes",
        "sha256",
        "distinct_from_consumed_attempt_controller",
        "authorized_to_retry_v1",
        "diagnostic_scope",
    }
    if (
        type(diagnosed_controller) is not dict
        or set(diagnosed_controller) != diagnosed_fields
        or _exact_schema_v3_record_path(
            diagnosed_controller.get("path"),
            expected=consumed_controller_path,
            role="diagnosed fixed legacy replacement controller",
        )
        != consumed_controller_path
        or type(diagnosed_controller.get("size_bytes")) is not int
        or diagnosed_controller.get("size_bytes") != 218766
        or diagnosed_controller.get("sha256")
        != _RESOURCE_BOUNDED_REPLACEMENT_DIAGNOSED_CONTROLLER_SHA256
        or diagnosed_controller.get("distinct_from_consumed_attempt_controller") is not True
        or diagnosed_controller.get("authorized_to_retry_v1") is not False
        or diagnosed_controller.get("diagnostic_scope")
        != "future_process_boundary_regression_only_no_v1_retry"
    ):
        raise _AuthorityValidationError(
            "diagnosed fixed legacy replacement controller is not exactly scoped"
        )
    if verify_live_records:
        diagnosed_live_controller = _read_stable_single_link_file(
            consumed_controller_path,
            role="diagnosed fixed legacy replacement controller",
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
        if (
            len(diagnosed_live_controller) != 218766
            or hashlib.sha256(diagnosed_live_controller).hexdigest()
            != _RESOURCE_BOUNDED_REPLACEMENT_DIAGNOSED_CONTROLLER_SHA256
        ):
            raise _AuthorityValidationError(
                "diagnosed fixed legacy replacement controller differs from live source"
            )
    qualifying_controller = controllers.get("qualifying_live_controller")
    if type(qualifying_controller) is not dict or set(qualifying_controller) != {
        "path",
        "size_bytes",
        "sha256",
        "distinct_from_consumed_attempt_controller",
        "authorized_to_retry_v1",
    }:
        raise _AuthorityValidationError(
            "qualifying replacement controller has an invalid field set"
        )
    qualifying_path = _exact_schema_v3_record_path(
        qualifying_controller.get("path"),
        expected=qualifying_controller_path,
        role="qualifying replacement publication controller",
    )
    qualifying_sha256 = _require_sha256(
        qualifying_controller.get("sha256"),
        role="qualifying replacement publication controller",
    )
    qualifying_size = qualifying_controller.get("size_bytes")
    if (
        qualifying_path != qualifying_controller_path
        or type(qualifying_size) is not int
        or qualifying_size <= 0
        or qualifying_sha256 == _RESOURCE_BOUNDED_REPLACEMENT_CONSUMED_CONTROLLER_SHA256
        or qualifying_controller.get("distinct_from_consumed_attempt_controller") is not True
        or qualifying_controller.get("authorized_to_retry_v1") is not False
    ):
        raise _AuthorityValidationError(
            "qualifying replacement controller violates the no-retry lineage"
        )
    if verify_live_records:
        live_controller = _read_stable_single_link_file(
            qualifying_controller_path,
            role="qualifying replacement publication controller",
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
        if (
            len(live_controller) != qualifying_size
            or hashlib.sha256(live_controller).hexdigest() != qualifying_sha256
        ):
            raise _AuthorityValidationError(
                "qualifying replacement controller differs from live source"
            )
    canonical_controllers = {
        "consumed_attempt_controller": {
            "path": str(consumed_controller_path),
            "size_bytes": 216288,
            "sha256": _RESOURCE_BOUNDED_REPLACEMENT_CONSUMED_CONTROLLER_SHA256,
            "live_file_match": False,
            "attested_by": consumed_attestations,
        },
        "diagnosed_fixed_legacy_controller": {
            "path": str(consumed_controller_path),
            "size_bytes": 218766,
            "sha256": _RESOURCE_BOUNDED_REPLACEMENT_DIAGNOSED_CONTROLLER_SHA256,
            "distinct_from_consumed_attempt_controller": True,
            "authorized_to_retry_v1": False,
            "diagnostic_scope": ("future_process_boundary_regression_only_no_v1_retry"),
        },
        "qualifying_live_controller": {
            "path": str(qualifying_controller_path),
            "size_bytes": qualifying_size,
            "sha256": qualifying_sha256,
            "distinct_from_consumed_attempt_controller": True,
            "authorized_to_retry_v1": False,
        },
    }

    failure_cause = value.get("failure_cause")
    canonical_failure_cause = {
        "error_type": "FreshVerifierError",
        "error_type_sha256": _RESOURCE_BOUNDED_REPLACEMENT_ERROR_TYPE_SHA256,
        "error_text": "FreshVerifierError: fresh verifier process did not exit cleanly",
        "error_sha256": _RESOURCE_BOUNDED_REPLACEMENT_ERROR_SHA256,
        "reason_code": "windows_venv_launcher_breaks_direct_child_ppid_contract",
        "scientific_or_evidence_corruption": False,
    }
    if (
        type(failure_cause) is not dict
        or set(failure_cause) != set(canonical_failure_cause)
        or failure_cause.get("scientific_or_evidence_corruption") is not False
        or dict(failure_cause) != canonical_failure_cause
    ):
        raise _AuthorityValidationError(
            "replacement publication failure cause is not the qualified defect"
        )

    run_specs = {
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
    run_state = value.get("run_state")
    raw_run_files = run_state.get("files") if type(run_state) is dict else None
    if (
        type(run_state) is not dict
        or set(run_state) != {"root", "files", "sha256"}
        or type(raw_run_files) is not dict
        or set(raw_run_files) != set(run_specs)
    ):
        raise _AuthorityValidationError(
            "replacement publication run-state snapshot has an invalid field set"
        )
    canonical_run_files = {
        name: canonical_record(
            raw_run_files[name],
            path=run_state_root / name,
            size_bytes=size,
            sha256=sha256,
            role=f"replacement publication run-state {name}",
            read_live=live_run_state,
        )
        for name, (size, sha256) in run_specs.items()
    }
    recomputed_run_state_sha256 = _canonical_mapping_sha256(
        {name: canonical_run_files[name]["sha256"] for name in run_specs}
    )
    if (
        _exact_schema_v3_record_path(
            run_state.get("root"),
            expected=run_state_root,
            role="replacement publication run-state root",
        )
        != run_state_root
        or run_state.get("sha256") != _RESOURCE_BOUNDED_REPLACEMENT_RUN_STATE_SHA256
        or recomputed_run_state_sha256 != _RESOURCE_BOUNDED_REPLACEMENT_RUN_STATE_SHA256
    ):
        raise _AuthorityValidationError("replacement publication run-state snapshot is not exact")
    canonical_run_state = {
        "root": str(run_state_root),
        "files": canonical_run_files,
        "sha256": _RESOURCE_BOUNDED_REPLACEMENT_RUN_STATE_SHA256,
    }

    protected_specs = {
        "specification": (
            project_root / "SPEC.md",
            11275,
            "9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0",
        ),
        "pre_registration": (
            project_root / "PRE_REGISTRATION.md",
            32538,
            "7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b",
        ),
        "primary_config": (
            project_root / "configs" / "primary_frozen.yaml",
            28497,
            "0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9",
        ),
        "confirmatory_config": (
            project_root / "configs" / "confirmatory_frozen.yaml",
            16099,
            "4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009",
        ),
    }
    protected = value.get("protected_bindings")
    if type(protected) is not dict or set(protected) != set(protected_specs):
        raise _AuthorityValidationError(
            "replacement publication protected bindings have an invalid role set"
        )
    canonical_protected = {
        role: canonical_record(
            protected[role],
            path=path,
            size_bytes=size,
            sha256=sha256,
            role=f"replacement publication protected {role}",
        )
        for role, (path, size, sha256) in protected_specs.items()
    }

    process = value.get("process_quiescence")
    if type(process) is not dict or set(process) != {
        "query_method",
        "observer_pid",
        "observed_at_utc",
        "matches",
        "historical_pid_inference_performed",
    }:
        raise _AuthorityValidationError(
            "replacement publication process-quiescence evidence has an invalid field set"
        )
    process_observed_at = _parse_timestamp(
        process.get("observed_at_utc"),
        role="replacement publication process-quiescence timestamp",
    )
    canonical_process_observed_at = process_observed_at.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    if type(process.get("observer_pid")) is not int:
        raise _AuthorityValidationError(
            "replacement publication process observer PID must be an exact integer"
        )
    observer_pid = _strict_cell_count(
        process.get("observer_pid"),
        role="replacement publication process observer PID",
        positive=True,
    )
    if (
        process.get("query_method") != "windows_cim_process_command_line_query_v1"
        or process.get("observed_at_utc") != canonical_process_observed_at
        or process_observed_at <= consumed_attempt_at
        or process_observed_at > qualified_at
        or (qualified_at - process_observed_at).total_seconds() > 60
        or type(process.get("matches")) is not list
        or process.get("matches") != []
        or process.get("historical_pid_inference_performed") is not False
    ):
        raise _AuthorityValidationError(
            "replacement publication process-quiescence evidence is not canonical"
        )
    canonical_process = {
        "query_method": "windows_cim_process_command_line_query_v1",
        "observer_pid": observer_pid,
        "observed_at_utc": canonical_process_observed_at,
        "matches": [],
        "historical_pid_inference_performed": False,
    }

    fixed_auxiliary_specs = {
        "retired_v1_invalidation_receipt": (
            control_root / "authority_d_replacement_inputs_v1.invalidation.json",
            6550,
            "0b9af7cdb9ca3fcb60c8dd6c123eda22f13631c1188ff390cd9421998e28e997",
        ),
        "prior_publication_failure_receipt": (
            control_root / "resource_authority_d_prior_publication_failure_receipt_v1.json",
            11413,
            "2b46f11d1580a6469715a525c0738d39fb3ae0f74f542e142ecd293ae7beed00",
        ),
        "failed_preflight_receipt": (
            control_root / "failed_resource_preflight_20260727T173054.689Z.json",
            1994,
            "e308aa0089a84caaca3f0722711e623579372636d47e161d9f32ca5a71f8c6eb",
        ),
    }
    auxiliary_records = {
        role: canonical_record(
            {
                "path": str(path),
                "size_bytes": size,
                "sha256": sha256,
            },
            path=path,
            size_bytes=size,
            sha256=sha256,
            role=f"replacement publication {role}",
        )
        for role, (path, size, sha256) in fixed_auxiliary_specs.items()
    }
    lock_read_records = [
        {"role": "publication_authorization_receipt", **authorization_record},
        {"role": "publication_attempt_marker", **attempt_record},
        {"role": "publication_failure_marker", **failure_record},
        {
            "role": "retired_v1_invalidation_receipt",
            **auxiliary_records["retired_v1_invalidation_receipt"],
        },
        {
            "role": "prior_publication_failure_receipt",
            **auxiliary_records["prior_publication_failure_receipt"],
        },
        {
            "role": "failed_preflight_receipt",
            **auxiliary_records["failed_preflight_receipt"],
        },
        {
            "role": "v2_cnn_correction_receipt",
            **canonical_v2_files["cnn_correction_receipt"],
        },
        {
            "role": "v2_frozen_source_receipt",
            **canonical_v2_files["frozen_source_receipt"],
        },
        {"role": "v2_source_allowlist", **canonical_v2_files["source_allowlist"]},
        {"role": "v2_workspace_plan", **canonical_v2_files["workspace_plan"]},
        {"role": "protected_specification", **canonical_protected["specification"]},
        {"role": "protected_pre_registration", **canonical_protected["pre_registration"]},
        {"role": "protected_primary_config", **canonical_protected["primary_config"]},
        {
            "role": "protected_confirmatory_config",
            **canonical_protected["confirmatory_config"],
        },
        {
            "role": "authority_c_amendment_evidence",
            **authority_files["amendment_evidence"],
        },
        {
            "role": "authority_c_amendment_report",
            **authority_files["amendment_report"],
        },
        {
            "role": "authority_c_confirmatory_config",
            **authority_files["confirmatory_config"],
        },
        {
            "role": "authority_c_immutable_marker",
            **authority_files["immutable_marker"],
        },
        {
            "role": "authority_c_preregistration",
            **authority_files["preregistration"],
        },
        {
            "role": "authority_c_primary_config",
            **authority_files["primary_config"],
        },
        {
            "role": "authority_c_sha256_manifest",
            **authority_files["sha256_manifest"],
        },
        {
            "role": "authority_c_source_tree_manifest",
            **authority_files["source_tree_manifest"],
        },
        {
            "role": "run_state_integrity_registry.jsonl",
            **canonical_run_files["integrity_registry.jsonl"],
        },
        {"role": "run_state_registry.csv", **canonical_run_files["registry.csv"]},
        {
            "role": "run_state_run_dispositions.anchor.json",
            **canonical_run_files["run_dispositions.anchor.json"],
        },
        {
            "role": "run_state_run_dispositions.jsonl",
            **canonical_run_files["run_dispositions.jsonl"],
        },
        {
            "role": "run_state_run_stage_attestations.anchor.json",
            **canonical_run_files["run_stage_attestations.anchor.json"],
        },
        {
            "role": "run_state_run_stage_attestations.jsonl",
            **canonical_run_files["run_stage_attestations.jsonl"],
        },
    ]
    lock = value.get("lock_quiescence")
    lock_reads = lock.get("reads_between_scans") if type(lock) is dict else None
    if (
        type(lock) is not dict
        or set(lock)
        != {
            "scan_method",
            "first_scan_paths",
            "second_scan_paths",
            "reads_between_scans",
        }
        or lock.get("scan_method") != "two_pass_scoped_lock_path_scan_v1"
        or type(lock.get("first_scan_paths")) is not list
        or lock.get("first_scan_paths") != []
        or type(lock.get("second_scan_paths")) is not list
        or lock.get("second_scan_paths") != []
        or type(lock_reads) is not list
        or len(lock_reads) != 28
        or any(type(record) is not dict for record in lock_reads)
        or lock_reads != lock_read_records
    ):
        raise _AuthorityValidationError(
            "replacement publication lock-quiescence evidence is not canonical"
        )
    canonical_lock = {
        "scan_method": "two_pass_scoped_lock_path_scan_v1",
        "first_scan_paths": [],
        "second_scan_paths": [],
        "reads_between_scans": lock_read_records,
    }

    canonical_disposition = {
        "v1_attempt_consumed": True,
        "v1_retry_allowed": False,
        "v1_artifacts_may_be_modified_moved_or_deleted": False,
        "successor_requires_new_namespace": True,
        "successor_may_reuse_v2_inputs": False,
        "qualification_authorizes_publication": False,
        "outcome_values_read": False,
    }
    disposition = value.get("disposition")
    if (
        type(disposition) is not dict
        or set(disposition) != set(canonical_disposition)
        or any(
            disposition.get(field) is not expected
            for field, expected in canonical_disposition.items()
        )
        or dict(disposition) != canonical_disposition
    ):
        raise _AuthorityValidationError(
            "replacement publication qualification disposition violates no-retry policy"
        )
    return {
        "schema_version": 1,
        "policy": _RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_POLICY,
        "status": "qualified_rolled_back_failure_no_retry",
        "qualified_at_utc": canonical_qualified_at,
        "project_root": str(project_root),
        "authority_c": canonical_authority,
        "terminal_namespace": canonical_terminal,
        "terminal_links": canonical_terminal_links,
        "frozen_v2_inputs": canonical_frozen,
        "controller_identities": canonical_controllers,
        "failure_cause": canonical_failure_cause,
        "run_state": canonical_run_state,
        "protected_bindings": canonical_protected,
        "process_quiescence": canonical_process,
        "lock_quiescence": canonical_lock,
        "disposition": canonical_disposition,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
    }


def _canonical_replacement_publication_failure_lineage(
    value: Mapping[str, Any],
    *,
    resource_parent: _AuthorityState,
    verify_live_receipt: bool,
    verify_live_run_state: bool = False,
) -> dict[str, Any]:
    """Bind one strict terminal-qualification receipt into schema-v3 D."""

    if verify_live_run_state and not verify_live_receipt:
        raise _AuthorityValidationError(
            "replacement publication live run-state verification requires live Q receipt"
        )
    fields = {
        "terminal_qualification_receipt_path",
        "terminal_qualification_receipt_sha256",
        "terminal_qualification_receipt",
    }
    if type(value) is not dict or set(value) != fields:
        raise _AuthorityValidationError(
            "replacement publication failure lineage has an invalid envelope"
        )
    artifacts_root = Path(os.path.abspath(resource_parent.directory.parent.parent))
    expected_path = (
        artifacts_root
        / "resource_control"
        / _RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_FILENAME
    )
    receipt_path = _exact_schema_v3_record_path(
        value.get("terminal_qualification_receipt_path"),
        expected=expected_path,
        role="replacement publication terminal-qualification receipt",
    )
    receipt_sha256 = _require_sha256(
        value.get("terminal_qualification_receipt_sha256"),
        role="replacement publication terminal-qualification receipt",
    )
    receipt = value.get("terminal_qualification_receipt")
    if receipt_path != expected_path or type(receipt) is not dict:
        raise _AuthorityValidationError(
            "replacement publication failure lineage has the wrong receipt path or payload"
        )
    canonical_receipt = _canonical_replacement_publication_terminal_qualification(
        receipt,
        resource_parent=resource_parent,
        verify_live_records=verify_live_receipt,
        verify_live_run_state=verify_live_run_state,
    )
    expected_sha256 = _atomic_json_sha256(canonical_receipt)
    if dict(receipt) != canonical_receipt or receipt_sha256 != expected_sha256:
        raise _AuthorityValidationError(
            "replacement publication terminal-qualification receipt is not canonical"
        )
    if verify_live_receipt:
        live_bytes = _read_stable_single_link_file(
            receipt_path,
            role="replacement publication terminal-qualification receipt",
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
        if (
            live_bytes != _atomic_json_bytes(canonical_receipt)
            or hashlib.sha256(live_bytes).hexdigest() != receipt_sha256
        ):
            raise _AuthorityValidationError(
                "replacement publication terminal-qualification live receipt changed"
            )
    return {
        "terminal_qualification_receipt_path": str(receipt_path),
        "terminal_qualification_receipt_sha256": receipt_sha256,
        "terminal_qualification_receipt": canonical_receipt,
    }


def verify_resource_bounded_replacement_terminal_qualification_receipt(
    receipt_path: str | Path,
    *,
    project_root: str | Path,
    parent_authority_directory: str | Path,
) -> dict[str, Any]:
    """Return the canonical schema-v3 lineage envelope from one anchored receipt."""

    root = _require_real_directory(
        Path(project_root).expanduser(),
        role="replacement terminal-qualification project root",
    )
    parent = _require_real_directory(
        Path(parent_authority_directory).expanduser(),
        role="replacement terminal-qualification Authority C",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    expected_project_root = parent.parent.parent.parent
    expected_receipt = (
        root
        / "artifacts"
        / "resource_control"
        / _RESOURCE_BOUNDED_REPLACEMENT_TERMINAL_QUALIFICATION_FILENAME
    )
    supplied_receipt_lexical = Path(receipt_path).expanduser()
    if not supplied_receipt_lexical.is_absolute() or str(supplied_receipt_lexical) != str(
        expected_receipt
    ):
        raise _AuthorityValidationError(
            "replacement terminal-qualification receipt path is not the exact "
            "canonical lexical path"
        )
    supplied_receipt = expected_receipt
    if root != expected_project_root:
        raise _AuthorityValidationError(
            "replacement terminal-qualification project, parent, or receipt path is not canonical"
        )
    receipt_bytes = _read_stable_single_link_file(
        supplied_receipt,
        role="replacement publication terminal-qualification receipt",
        max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
    )
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            "replacement publication terminal-qualification receipt is invalid JSON"
        ) from exc
    if type(receipt) is not dict:
        raise _AuthorityValidationError(
            "replacement publication terminal-qualification receipt must be a JSON object"
        )
    return _canonical_replacement_publication_failure_lineage(
        {
            "terminal_qualification_receipt_path": str(supplied_receipt),
            "terminal_qualification_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "terminal_qualification_receipt": dict(receipt),
        },
        resource_parent=parent_state,
        verify_live_receipt=True,
        verify_live_run_state=True,
    )


def validate_resource_bounded_capacity_v3(
    capacity_policy: Mapping[str, Any],
    resource_input_workspace_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact public capacity-v3/typed-workspace contract or fail closed."""

    if (
        not isinstance(capacity_policy, Mapping)
        or set(capacity_policy) != set(_RESOURCE_BOUNDED_CAPACITY_V3)
        or any(
            type(capacity_policy.get(field)) is not type(expected)
            or capacity_policy.get(field) != expected
            for field, expected in _RESOURCE_BOUNDED_CAPACITY_V3.items()
        )
    ):
        raise _AuthorityValidationError(
            "resource-bounded capacity-v3 differs from its exact audited policy"
        )
    return (
        dict(_RESOURCE_BOUNDED_CAPACITY_V3),
        _canonical_resource_input_workspace_plan(resource_input_workspace_plan),
    )


def _canonical_resource_bounded_technical_successor_authorization(
    value: ResourceBoundedTechnicalSuccessorAuthorization | Mapping[str, Any],
    *,
    parent_state: _AuthorityState,
    parent_resource_authorization: Mapping[str, Any],
    successor_source: Mapping[str, Any],
    successor_source_manifest_sha256: str,
    successor_config: Mapping[str, Any],
    successor_config_file_sha256: str,
    successor_config_semantic_sha256: str,
    verify_live_receipt: bool,
    verify_live_replacement_run_state: bool = False,
) -> dict[str, Any]:
    if isinstance(value, ResourceBoundedTechnicalSuccessorAuthorization):
        raw: Mapping[str, Any] = value.as_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(
            "resource_bounded_technical_successor_authorization must be typed or a mapping"
        )
    authorization_schema_version = raw.get("schema_version")
    if type(authorization_schema_version) is not int or authorization_schema_version not in {
        2,
        3,
    }:
        raise _AuthorityValidationError(
            "resource technical successor violates the unchanged post-outcome policy: "
            "authorization schema must be exact integer 2 or 3"
        )
    expected_fields = {
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
    }
    if authorization_schema_version == 3:
        expected_fields.add("replacement_publication_failure_lineage")
    if set(raw) != expected_fields:
        raise _AuthorityValidationError(
            "resource technical successor authorization has an unexpected field set"
        )
    fixed = (
        raw.get("policy")
        == (
            _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY_V3
            if authorization_schema_version == 3
            else _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY
        )
        and raw.get("purpose") == RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE
        and raw.get("outcomes_inspected") is True
        and raw.get("analysis_disposition") == _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION
        and raw.get("outcome_use_policy") == _RESOURCE_BOUNDED_OUTCOME_USE_POLICY
        and raw.get("original_confirmatory_claim_allowed") is False
        and raw.get("study_outcome_eligible") is False
        and raw.get("completion_stage") is None
        and raw.get("primary_rebinding_allowed") is False
        and raw.get("primary_mutation_allowed") is False
        and raw.get("automatic_retry_allowed") is False
        and raw.get("scientific_profile_change_allowed") is False
    )
    if not fixed:
        raise _AuthorityValidationError(
            "resource technical successor violates the unchanged post-outcome policy"
        )
    parent_evidence_path = parent_state.directory / _EVIDENCE_FILENAME
    parent_evidence = _json_mapping(
        parent_evidence_path,
        role="resource authority C amendment evidence",
    )
    if (
        parent_evidence.get("schema_version") != 4
        or parent_evidence.get("amendment_purpose")
        != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
    ):
        raise _AuthorityValidationError(
            "resource technical successor must be a direct child of schema-v4 authority C"
        )
    parent_authorization_sha256 = _canonical_mapping_sha256(parent_resource_authorization)
    expected_supersedes = {
        "authority_directory": str(parent_state.directory),
        "authority_schema_version": 4,
        "artifact_root_sha256": parent_state.artifact_root_sha256,
        "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
        "chain_depth": parent_state.chain_depth,
        "amendment_evidence_sha256": sha256_file(parent_evidence_path),
        "authorization_sha256": parent_authorization_sha256,
        "effective_execution_leaf": False,
        "historical_verification_retained": True,
    }
    supersedes = raw.get("supersedes")
    if not isinstance(supersedes, Mapping) or dict(supersedes) != expected_supersedes:
        raise _AuthorityValidationError(
            "resource technical successor does not exactly supersede its direct authority C"
        )
    prior_publication_failure = raw.get("prior_publication_failure")
    if not isinstance(prior_publication_failure, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks its prior-publication failure receipt"
        )
    canonical_prior_publication_failure = _canonical_prior_resource_authority_d_publication_failure(
        prior_publication_failure,
        resource_parent=parent_state,
        verify_live_receipt=verify_live_receipt,
    )
    canonical_replacement_failure_lineage: dict[str, Any] | None = None
    if authorization_schema_version == 3:
        lineage = raw.get("replacement_publication_failure_lineage")
        if not isinstance(lineage, Mapping):
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor lacks its replacement "
                "publication failure lineage"
            )
        canonical_replacement_failure_lineage = _canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=verify_live_receipt,
            verify_live_run_state=verify_live_replacement_run_state,
        )
        terminal_receipt = canonical_replacement_failure_lineage["terminal_qualification_receipt"]
        qualifying_controller = terminal_receipt["controller_identities"][
            "qualifying_live_controller"
        ]
        source_records = successor_source.get("artifacts")
        expected_controller_relative_path = (
            "src/histo_audit/workflows/resource_authority_d_replacement_v2_controller.py"
        )
        if not isinstance(source_records, list):
            raise _AuthorityValidationError(
                "schema-v3 successor source lacks its artifact inventory"
            )
        matched_controller_records = [
            record
            for record in source_records
            if isinstance(record, Mapping)
            and record.get("path") == expected_controller_relative_path
        ]
        if (
            len(matched_controller_records) != 1
            or matched_controller_records[0].get("size_bytes")
            != qualifying_controller["size_bytes"]
            or matched_controller_records[0].get("sha256") != qualifying_controller["sha256"]
        ):
            raise _AuthorityValidationError(
                "schema-v3 successor source does not bind the qualifying v2 controller"
            )
    historical_primary = parent_resource_authorization.get("historical_primary")
    if (
        not isinstance(historical_primary, Mapping)
        or raw.get("historical_primary") != historical_primary
    ):
        raise _AuthorityValidationError(
            "resource technical successor changed the historical primary binding"
        )
    parent_profile = parent_resource_authorization.get("resource_profile")
    if not isinstance(parent_profile, Mapping):
        raise _AuthorityValidationError("resource authority C lacks its exact profile")
    parent_confirmatory_hashes = parent_state.snapshot_hashes.get("confirmatory_config")
    if not isinstance(parent_confirmatory_hashes, Mapping):
        raise _AuthorityValidationError(
            "resource authority C lacks confirmatory-config snapshot hashes"
        )
    parent_config_file_sha256 = _require_sha256(
        parent_profile.get("resource_confirmatory_config_file_sha256"),
        role="resource authority C confirmatory config file",
    )
    parent_config_semantic_sha256 = _require_sha256(
        parent_profile.get("resource_confirmatory_config_semantic_sha256"),
        role="resource authority C confirmatory config semantics",
    )
    if parent_config_file_sha256 != _require_sha256(
        parent_confirmatory_hashes.get("file_sha256"),
        role="resource authority C snapshot confirmatory config file",
    ) or parent_config_semantic_sha256 != _require_sha256(
        parent_confirmatory_hashes.get("semantic_sha256"),
        role="resource authority C snapshot confirmatory config semantics",
    ):
        raise _AuthorityValidationError(
            "resource authority C profile differs from its sealed config snapshot"
        )
    successor_file_sha256 = _require_sha256(
        successor_config_file_sha256,
        role="resource technical successor confirmatory config file",
    )
    successor_semantic_sha256 = _require_sha256(
        successor_config_semantic_sha256,
        role="resource technical successor confirmatory config semantics",
    )
    expected_profile = {
        "profile_id": _RESOURCE_BOUNDED_PROFILE_ID,
        "experiment_name": _RESOURCE_BOUNDED_EXPERIMENT_NAME,
        "parent_confirmatory_config_file_sha256": parent_config_file_sha256,
        "parent_confirmatory_config_semantic_sha256": parent_config_semantic_sha256,
        "resource_confirmatory_config_file_sha256": successor_file_sha256,
        "resource_confirmatory_config_semantic_sha256": successor_semantic_sha256,
    }
    if (
        raw.get("resource_profile") != expected_profile
        or raw.get("expected_successor_config_semantic_sha256") != successor_semantic_sha256
        or successor_file_sha256 == parent_config_file_sha256
        or successor_semantic_sha256 == parent_config_semantic_sha256
        or successor_config.get("execution_profile") != _RESOURCE_BOUNDED_PROFILE_ID
        or successor_config.get("analysis_disposition") != _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION
        or successor_config.get("original_confirmatory_claim_allowed") is not False
        or successor_config.get("completion_stage") is not None
    ):
        raise _AuthorityValidationError(
            "resource technical successor must bind one different, explicitly expected "
            "child config while retaining the post-outcome profile"
        )
    parent_config = load_config(parent_state.directory / _CONFIRMATORY_CONFIG_SNAPSHOT)
    before_config_record, after_config_record = _require_resource_technical_config_correction(
        parent_config, successor_config
    )
    parent_source = _json_mapping(
        parent_state.directory / _SOURCE_TREE_SNAPSHOT,
        role="resource authority C execution source",
    )
    source_delta = raw.get("execution_source_delta")
    if not isinstance(source_delta, Mapping) or set(source_delta) != {
        "policy",
        "parent_root_sha256",
        "parent_manifest_sha256",
        "resource_root_sha256",
        "resource_manifest_sha256",
        "allowlisted_change_kinds",
        "allowlisted_changes",
        "delta_sha256",
    }:
        raise _AuthorityValidationError(
            "resource technical successor source delta has an invalid field set"
        )
    allowlisted_kinds = source_delta.get("allowlisted_change_kinds")
    if not isinstance(allowlisted_kinds, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor source delta lacks its explicit allowlist"
        )
    canonical_delta, delta_sha256 = _canonical_source_delta_with_allowlist(
        parent_source,
        successor_source,
        allowlisted_change_kinds=allowlisted_kinds,
        role="resource technical successor",
    )
    parent_execution = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_execution, Mapping):
        raise _AuthorityValidationError(
            "resource authority C lacks execution-source snapshot hashes"
        )
    expected_source_delta = {
        "policy": _RESOURCE_BOUNDED_TECHNICAL_SOURCE_DELTA_POLICY,
        "parent_root_sha256": _require_sha256(
            parent_execution.get("root_sha256"),
            role="resource authority C execution-source root",
        ),
        "parent_manifest_sha256": _require_sha256(
            parent_execution.get("manifest_sha256"),
            role="resource authority C execution-source manifest",
        ),
        "resource_root_sha256": _require_sha256(
            successor_source.get("root_sha256"),
            role="resource technical successor execution-source root",
        ),
        "resource_manifest_sha256": _require_sha256(
            successor_source_manifest_sha256,
            role="resource technical successor execution-source manifest",
        ),
        "allowlisted_change_kinds": {
            str(path): str(kind)
            for path, kind in sorted(
                allowlisted_kinds.items(),
                key=lambda item: str(item[0]),
            )
        },
        "allowlisted_changes": [dict(record) for record in canonical_delta],
        "delta_sha256": delta_sha256,
    }
    if dict(source_delta) != expected_source_delta:
        raise _AuthorityValidationError(
            "resource technical successor source delta differs from its exact snapshots"
        )
    failed = raw.get("failed_preflight")
    if not isinstance(failed, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks failed-preflight evidence"
        )
    canonical_failed = _canonical_failed_resource_preflight(
        failed,
        resource_parent=parent_state,
        resource_authorization_sha256=parent_authorization_sha256,
        verify_live_receipt=verify_live_receipt,
    )
    correction = raw.get("cnn_provenance_correction")
    if not isinstance(correction, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks CNN-provenance correction evidence"
        )
    canonical_correction = _canonical_cnn_provenance_correction(
        correction,
        before_config_record=before_config_record,
        after_config_record=after_config_record,
        parent_source=parent_source,
        successor_source=successor_source,
        failed_preflight=canonical_failed,
    )
    workspace_plan_value = raw.get("resource_input_workspace_plan")
    if not isinstance(workspace_plan_value, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks its typed input-workspace plan"
        )
    capacity = raw.get("resource_capacity_policy")
    parent_capacity = parent_resource_authorization.get("resource_capacity_policy")
    if not isinstance(capacity, Mapping):
        raise _AuthorityValidationError("resource technical successor lacks capacity-v3")
    canonical_capacity, canonical_workspace_plan = validate_resource_bounded_capacity_v3(
        capacity, workspace_plan_value
    )
    if (
        not isinstance(parent_capacity, Mapping)
        or set(parent_capacity) != set(_RESOURCE_BOUNDED_CAPACITY)
        or any(
            type(parent_capacity.get(field)) is not type(expected)
            or parent_capacity.get(field) != expected
            for field, expected in _RESOURCE_BOUNDED_CAPACITY.items()
        )
        or raw.get("resource_profile_shape") != _RESOURCE_BOUNDED_PROFILE_SHAPE
    ):
        raise _AuthorityValidationError(
            "resource technical successor changed the 24/6/30 profile, parent "
            "capacity-v2, or exact workspace-aware capacity-v3 policy"
        )
    return ResourceBoundedTechnicalSuccessorAuthorization(
        superseded_authority=expected_supersedes,
        prior_publication_failure=canonical_prior_publication_failure,
        failed_preflight=canonical_failed,
        historical_primary=dict(historical_primary),
        resource_profile=expected_profile,
        execution_source_delta=expected_source_delta,
        cnn_provenance_correction=canonical_correction,
        resource_capacity_policy=canonical_capacity,
        resource_input_workspace_plan=canonical_workspace_plan,
        expected_successor_config_semantic_sha256=successor_semantic_sha256,
        replacement_publication_failure_lineage=canonical_replacement_failure_lineage,
    ).as_dict()


def _canonical_primary_recovery_authorization(
    value: Mapping[str, Any],
    *,
    parent_state: _AuthorityState,
    outcomes_inspected: bool,
    outcomes_inspected_at_utc: str | None,
    amendment_reason: str,
) -> dict[str, Any]:
    """Canonicalize one sealed, post-outcome interrupted-primary recovery authority.

    This boundary authenticates policy and immutable expected digests only. It never
    opens the unsealed source run or performs the expensive filesystem readback; the
    recovery runner must compare the live source with every expected digest exactly
    once before creating a new run.
    """

    if not isinstance(value, Mapping):
        raise TypeError("primary_recovery_authorization must be a mapping")
    expected_fields = {
        "schema_version",
        "policy",
        "source_run_id",
        "source_run_directory",
        "interruption_evidence",
        "outcomes_inspected",
        "outcome_inspection_at_utc",
        "analysis_disposition",
        "scientific_method_changes",
        "expected_status_sha256",
        "expected_primary_execution_gate_sha256",
        "expected_source_tree_manifest_sha256",
        "expected_source_tree_root_sha256",
        "expected_source_snapshot_root_sha256",
        "expected_source_filesystem_readback_root_sha256",
        "expected_restoration_readback_root_sha256",
        "expected_statistics_manifest_sha256",
        "trust_assumption",
        "limitation",
        "reason",
    }
    if set(value) != expected_fields:
        raise _AuthorityValidationError(
            "primary recovery authorization has an unexpected field set"
        )
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise _AuthorityValidationError("primary recovery authorization schema is invalid")
    if value.get("policy") != _PRIMARY_RECOVERY_POLICY:
        raise _AuthorityValidationError("primary recovery authorization policy is invalid")

    run_id = value.get("source_run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id != run_id.strip()
        or Path(run_id).name != run_id
    ):
        raise _AuthorityValidationError(
            "primary recovery source_run_id must be one safe non-empty run ID"
        )
    run_directory = _absolute_record_path(
        value.get("source_run_directory"),
        role="primary recovery source run directory",
    )
    if run_directory.name != run_id:
        raise _AuthorityValidationError(
            "primary recovery source run directory does not match source_run_id"
        )

    interruption = value.get("interruption_evidence")
    expected_interruption_fields = {
        "kind",
        "receipt_path",
        "receipt_sha256",
        "observed_at_utc",
        "last_boot_at_utc",
        "event_id",
        "source_process_id",
        "process_checked_at_utc",
        "process_active",
    }
    if not isinstance(interruption, Mapping) or set(interruption) != expected_interruption_fields:
        raise _AuthorityValidationError(
            "primary recovery interruption evidence has an unexpected field set"
        )
    if interruption.get("kind") != _PRIMARY_RECOVERY_INTERRUPTION_KIND:
        raise _AuthorityValidationError("primary recovery interruption kind is invalid")
    receipt_path = _absolute_record_path(
        interruption.get("receipt_path"),
        role="primary recovery interruption receipt",
    )
    receipt_sha256 = _require_sha256(
        interruption.get("receipt_sha256"),
        role="primary recovery interruption receipt",
    )
    if (
        type(interruption.get("event_id")) is not int
        or interruption.get("event_id") != _PRIMARY_RECOVERY_EVENT_ID
    ):
        raise _AuthorityValidationError("primary recovery interruption event_id is invalid")
    process_id = interruption.get("source_process_id")
    if type(process_id) is not int or process_id <= 0:
        raise _AuthorityValidationError(
            "primary recovery source_process_id must be a positive exact integer"
        )
    if interruption.get("process_active") is not False:
        raise _AuthorityValidationError("primary recovery requires process_active=false")
    boot_time = _parse_timestamp(
        interruption.get("last_boot_at_utc"),
        role="primary recovery last-boot timestamp",
    )
    checked_time = _parse_timestamp(
        interruption.get("process_checked_at_utc"),
        role="primary recovery process-check timestamp",
    )
    observed_time = _parse_timestamp(
        interruption.get("observed_at_utc"),
        role="primary recovery interruption-observation timestamp",
    )
    if not boot_time <= checked_time <= observed_time:
        raise _AuthorityValidationError(
            "primary recovery interruption timestamps are not chronologically ordered"
        )

    if outcomes_inspected is not True or value.get("outcomes_inspected") is not True:
        raise _AuthorityValidationError("primary recovery requires outcomes_inspected=true")
    if outcomes_inspected_at_utc is None:
        raise _AuthorityValidationError("primary recovery requires an outcome-inspection timestamp")
    inspection_value = value.get("outcome_inspection_at_utc")
    inspection_time = _parse_timestamp(
        inspection_value,
        role="primary recovery outcome-inspection timestamp",
    )
    if inspection_value != outcomes_inspected_at_utc:
        raise _AuthorityValidationError(
            "primary recovery outcome-inspection timestamp differs from the amendment"
        )
    if boot_time > inspection_time:
        raise _AuthorityValidationError(
            "primary recovery host reboot cannot be later than outcome inspection"
        )
    if value.get("analysis_disposition") != _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION:
        raise _AuthorityValidationError(
            "primary recovery analysis disposition must be amended_or_exploratory"
        )
    if type(value.get("scientific_method_changes")) is not list or value.get(
        "scientific_method_changes"
    ):
        raise _AuthorityValidationError(
            "primary recovery scientific_method_changes must be an exact empty list"
        )

    expected_source_root = _require_sha256(
        value.get("expected_source_tree_root_sha256"),
        role="primary recovery source-tree root",
    )
    parent_source = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_source, Mapping) or expected_source_root != parent_source.get(
        "root_sha256"
    ):
        raise _AuthorityValidationError(
            "primary recovery source-tree root differs from the immediate parent authority"
        )
    digest_fields = (
        "expected_status_sha256",
        "expected_primary_execution_gate_sha256",
        "expected_source_tree_manifest_sha256",
        "expected_source_snapshot_root_sha256",
        "expected_source_filesystem_readback_root_sha256",
        "expected_restoration_readback_root_sha256",
        "expected_statistics_manifest_sha256",
    )
    digests = {
        field: _require_sha256(
            value.get(field),
            role=f"primary recovery {field}",
        )
        for field in digest_fields
    }
    raw_trust_assumption = value.get("trust_assumption")
    raw_limitation = value.get("limitation")
    raw_reason = value.get("reason")
    if (
        not isinstance(raw_trust_assumption, str)
        or not isinstance(raw_limitation, str)
        or not isinstance(raw_reason, str)
    ):
        raise _AuthorityValidationError(
            "primary recovery trust assumption, limitation, and reason must be strings"
        )
    trust_assumption = _normalise_text(
        raw_trust_assumption,
        role="primary recovery trust assumption",
    )
    limitation = _normalise_text(
        raw_limitation,
        role="primary recovery limitation",
    )
    reason = _normalise_text(raw_reason, role="primary recovery reason")
    if reason != amendment_reason:
        raise _AuthorityValidationError("primary recovery reason differs from the amendment reason")
    for field, canonical_text in (
        ("trust_assumption", trust_assumption),
        ("limitation", limitation),
        ("reason", reason),
    ):
        if value.get(field) != canonical_text:
            raise _AuthorityValidationError(
                f"primary recovery field {field!r} is not in canonical form"
            )

    return {
        "schema_version": 1,
        "policy": _PRIMARY_RECOVERY_POLICY,
        "source_run_id": run_id,
        "source_run_directory": str(run_directory),
        "interruption_evidence": {
            "kind": _PRIMARY_RECOVERY_INTERRUPTION_KIND,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "observed_at_utc": interruption["observed_at_utc"],
            "last_boot_at_utc": interruption["last_boot_at_utc"],
            "event_id": _PRIMARY_RECOVERY_EVENT_ID,
            "source_process_id": process_id,
            "process_checked_at_utc": interruption["process_checked_at_utc"],
            "process_active": False,
        },
        "outcomes_inspected": True,
        "outcome_inspection_at_utc": inspection_value,
        "analysis_disposition": _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION,
        "scientific_method_changes": [],
        "expected_status_sha256": digests["expected_status_sha256"],
        "expected_primary_execution_gate_sha256": digests["expected_primary_execution_gate_sha256"],
        "expected_source_tree_manifest_sha256": digests["expected_source_tree_manifest_sha256"],
        "expected_source_tree_root_sha256": expected_source_root,
        "expected_source_snapshot_root_sha256": digests["expected_source_snapshot_root_sha256"],
        "expected_source_filesystem_readback_root_sha256": digests[
            "expected_source_filesystem_readback_root_sha256"
        ],
        "expected_restoration_readback_root_sha256": digests[
            "expected_restoration_readback_root_sha256"
        ],
        "expected_statistics_manifest_sha256": digests["expected_statistics_manifest_sha256"],
        "trust_assumption": trust_assumption,
        "limitation": limitation,
        "reason": reason,
    }


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    excluded = {_IMMUTABLE_MARKER, _MANIFEST_FILENAME}
    records: list[dict[str, Any]] = []
    entries = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    expected_names = tuple(path.name for path in entries)
    for path in entries:
        payload = _read_stable_single_link_file(
            path,
            role=f"amendment bundle artifact {path.name}",
        )
        if path.name in excluded:
            continue
        records.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if tuple(sorted(path.name for path in directory.iterdir())) != expected_names:
        raise _AuthorityValidationError(
            "amendment bundle file set changed during anchored readback"
        )
    return records


def _normalise_manifest_records(raw: Any, *, flat: bool) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise _AuthorityValidationError("checksum-manifest artifacts must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"path", "size_bytes", "sha256"}:
            raise _AuthorityValidationError(
                f"checksum record {index} must contain exactly path/size_bytes/sha256"
            )
        path = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        pure_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or pure_path is None
            or pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.as_posix() != path
            or (flat and len(pure_path.parts) != 1)
            or path in seen
        ):
            raise _AuthorityValidationError(f"invalid or duplicate artifact path: {path!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise _AuthorityValidationError(f"invalid size for artifact {path!r}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise _AuthorityValidationError(f"invalid SHA-256 for artifact {path!r}")
        seen.add(path)
        records.append({"path": path, "size_bytes": size, "sha256": digest})
    if [record["path"] for record in records] != sorted(seen):
        raise _AuthorityValidationError("checksum records must be sorted by path")
    return records


def _verify_amendment_bundle(directory: Path) -> _BundleIntegrity:
    # Validate every leaf, including both excluded seal files, before any JSON
    # parser is allowed to follow a pathname into authority-controlled content.
    actual_records = _artifact_records(directory)
    manifest = _json_mapping(directory / _MANIFEST_FILENAME, role="amendment checksum manifest")
    marker = _json_mapping(directory / _IMMUTABLE_MARKER, role="amendment immutable marker")
    expected_records = _normalise_manifest_records(manifest.get("artifacts"), flat=True)
    expected_by_path = {record["path"]: record for record in expected_records}
    actual_by_path = {record["path"]: record for record in actual_records}
    missing = sorted(set(expected_by_path).difference(actual_by_path))
    added = sorted(set(actual_by_path).difference(expected_by_path))
    changed = sorted(
        path
        for path in set(expected_by_path).intersection(actual_by_path)
        if expected_by_path[path] != actual_by_path[path]
    )
    if missing or added or changed:
        raise _AuthorityValidationError(
            "amendment artifact integrity mismatch: "
            f"missing={missing}, added={added}, changed={changed}"
        )
    actual_root = _canonical_root(actual_records)
    expected_root = manifest.get("artifact_root_sha256")
    manifest_sha = sha256_file(directory / _MANIFEST_FILENAME)
    if manifest.get("schema_version") != 1 or manifest.get("authority_kind") != _AUTHORITY_KIND:
        raise _AuthorityValidationError("amendment checksum manifest schema/kind is invalid")
    if manifest.get("artifact_count") != len(expected_records):
        raise _AuthorityValidationError("amendment checksum manifest artifact count is invalid")
    if expected_root != actual_root or not isinstance(expected_root, str):
        raise _AuthorityValidationError("amendment artifact root differs from actual artifacts")
    if marker.get("schema_version") != 1 or marker.get("status") != "amended":
        raise _AuthorityValidationError("amendment immutable marker schema/status is invalid")
    if marker.get("authority_kind") != _AUTHORITY_KIND:
        raise _AuthorityValidationError("amendment immutable marker kind is invalid")
    if marker.get("artifact_root_sha256") != actual_root:
        raise _AuthorityValidationError("amendment marker does not authenticate the artifact root")
    if marker.get("sha256_manifest_sha256") != manifest_sha:
        raise _AuthorityValidationError("amendment marker does not authenticate its manifest")
    return _BundleIntegrity(actual_root, manifest_sha)


def _execution_hashes(path: Path) -> dict[str, Any]:
    payload = _json_mapping(path, role="execution-source manifest")
    if set(payload) != {
        "schema_version",
        "scope_kind",
        "scope",
        "excluded_roots",
        "excluded_paths",
        "artifact_count",
        "root_sha256",
        "artifacts",
    }:
        raise _AuthorityValidationError(
            "execution-source manifest must contain exactly the schema-v3 fields"
        )
    if payload.get("schema_version") != 3 or payload.get("scope_kind") != "execution_source":
        raise _AuthorityValidationError(
            "execution-source manifest must use schema v3 execution scope"
        )
    if payload.get("scope") != _EXECUTION_SCOPE:
        raise _AuthorityValidationError("execution-source manifest has an unexpected scope")
    if payload.get("excluded_roots") != _EXECUTION_EXCLUDED_ROOTS:
        raise _AuthorityValidationError("execution-source manifest has unexpected excluded roots")
    if payload.get("excluded_paths") != _EXECUTION_EXCLUDED_PATHS:
        raise _AuthorityValidationError("execution-source manifest has unexpected excluded paths")
    records = _normalise_manifest_records(payload.get("artifacts"), flat=False)
    invalid_paths = [
        record["path"]
        for record in records
        if record["path"] in _EXECUTION_EXCLUDED_PATHS
        or not (
            record["path"] in {"pyproject.toml", "uv.lock"}
            or record["path"].startswith("src/")
            or record["path"].startswith("configs/")
        )
    ]
    if invalid_paths:
        raise _AuthorityValidationError(
            f"execution-source manifest contains out-of-scope paths: {invalid_paths}"
        )
    if payload.get("artifact_count") != len(records):
        raise _AuthorityValidationError("execution-source artifact count is invalid")
    root = _canonical_root(records)
    if payload.get("root_sha256") != root:
        raise _AuthorityValidationError("execution-source root does not match its records")
    return {"manifest_sha256": sha256_file(path), "root_sha256": root}


def _snapshot_hashes(directory: Path) -> dict[str, Any]:
    preregistration = directory / _PREREGISTRATION_SNAPSHOT
    primary = directory / _PRIMARY_CONFIG_SNAPSHOT
    confirmatory = directory / _CONFIRMATORY_CONFIG_SNAPSHOT
    source_tree = directory / _SOURCE_TREE_SNAPSHOT
    for path, role in (
        (preregistration, "preregistration snapshot"),
        (primary, "primary-config snapshot"),
        (confirmatory, "confirmatory-config snapshot"),
        (source_tree, "execution-source snapshot"),
    ):
        if not path.is_file() or path.is_symlink():
            raise _AuthorityValidationError(f"{role} is missing or not a regular file: {path}")
    primary_config = load_config(primary)
    confirmatory_config = load_config(confirmatory)
    return {
        "preregistration": {
            "snapshot": _PREREGISTRATION_SNAPSHOT,
            "file_sha256": sha256_file(preregistration),
        },
        "primary_config": {
            "snapshot": _PRIMARY_CONFIG_SNAPSHOT,
            "file_sha256": sha256_file(primary),
            "semantic_sha256": config_sha256(primary_config),
        },
        "confirmatory_config": {
            "snapshot": _CONFIRMATORY_CONFIG_SNAPSHOT,
            "file_sha256": sha256_file(confirmatory),
            "semantic_sha256": config_sha256(confirmatory_config),
        },
        "execution_source": {
            "snapshot": _SOURCE_TREE_SNAPSHOT,
            **_execution_hashes(source_tree),
        },
    }


def _frozen_scientific_identity(snapshot_hashes: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the bytes/semantics that constitute an actual frozen change.

    The execution manifest's own file digest is deliberately not part of this
    identity: two canonical serialisations can describe the same authenticated
    execution-source root.  The complete before/after structures still retain that
    file digest for exact provenance and tamper detection.
    """

    try:
        preregistration = snapshot_hashes["preregistration"]
        primary = snapshot_hashes["primary_config"]
        confirmatory = snapshot_hashes["confirmatory_config"]
        execution = snapshot_hashes["execution_source"]
        if not all(
            isinstance(item, Mapping)
            for item in (preregistration, primary, confirmatory, execution)
        ):
            raise KeyError("snapshot hash section is not a mapping")
        identity = (
            preregistration["file_sha256"],
            primary["file_sha256"],
            primary["semantic_sha256"],
            confirmatory["file_sha256"],
            confirmatory["semantic_sha256"],
            execution["root_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise _AuthorityValidationError(
            "snapshot hashes cannot establish a complete frozen scientific identity"
        ) from exc
    if not all(isinstance(digest, str) and _SHA256.fullmatch(digest) for digest in identity):
        raise _AuthorityValidationError(
            "snapshot hashes contain an invalid frozen scientific identity digest"
        )
    return identity


def _require_storage_only_scientific_identity(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    for section in ("preregistration", "primary_config", "confirmatory_config"):
        before_record = before.get(section)
        after_record = after.get(section)
        if (
            not isinstance(before_record, Mapping)
            or not isinstance(after_record, Mapping)
            or dict(before_record) != dict(after_record)
        ):
            raise _AuthorityValidationError(
                "confirmatory storage policy requires unchanged frozen preregistration "
                f"and configs; {section} differs from the immediate parent"
            )


def _require_resource_technical_parent_identity(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Require D to retain C's preregistration and primary config exactly."""

    for section in ("preregistration", "primary_config"):
        before_record = before.get(section)
        after_record = after.get(section)
        if (
            not isinstance(before_record, Mapping)
            or not isinstance(after_record, Mapping)
            or dict(before_record) != dict(after_record)
        ):
            raise _AuthorityValidationError(
                "resource technical successor requires unchanged preregistration and "
                f"primary config; {section} differs from authority C"
            )


def _authority_key(directory: Path) -> tuple[str, int, int]:
    value = directory.stat(follow_symlinks=False)
    return (str(directory).casefold(), value.st_dev, value.st_ino)


def _base_authority_state(directory: Path) -> _AuthorityState:
    verification = verify_preregistration_freeze(directory)
    if not verification.valid or verification.expected_artifact_root_sha256 is None:
        raise _AuthorityValidationError(
            "parent base freeze failed immutable verification: "
            f"errors={verification.errors}, missing={verification.missing_paths}, "
            f"added={verification.added_paths}, changed={verification.changed_paths}"
        )
    marker = _json_mapping(directory / _IMMUTABLE_MARKER, role="base freeze marker")
    timestamp_value = marker.get("freeze_timestamp_utc")
    timestamp = _parse_timestamp(timestamp_value, role="base freeze timestamp")
    return _AuthorityState(
        directory=directory,
        kind="base_freeze",
        timestamp_utc=str(timestamp_value),
        timestamp=timestamp,
        chain_depth=0,
        artifact_root_sha256=verification.expected_artifact_root_sha256,
        sha256_manifest_sha256=sha256_file(directory / _MANIFEST_FILENAME),
        snapshot_hashes=_snapshot_hashes(directory),
        parent_directory=None,
    )


def _analysis_dispositions(
    analyses: Sequence[str], *, outcomes_inspected: bool
) -> list[dict[str, Any]]:
    status = "amended_or_exploratory" if outcomes_inspected else "amended_before_outcome_inspection"
    return [
        {
            "analysis": analysis,
            "registration_status": status,
            "original_unamended_primary_claim_allowed": False,
            "amended_primary_claim_allowed": not outcomes_inspected,
        }
        for analysis in analyses
    ]


def _validate_amendment_evidence(
    directory: Path,
    *,
    integrity: _BundleIntegrity,
    visited: set[tuple[str, int, int]],
    depth: int,
    max_chain_depth: int,
) -> _AuthorityState:
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    marker = _json_mapping(directory / _IMMUTABLE_MARKER, role="amendment marker")
    base_required_keys = {
        "schema_version",
        "authority_kind",
        "amendment_timestamp_utc",
        "chain_depth",
        "parent",
        "reason",
        "affected_hypotheses",
        "affected_analyses",
        "outcomes_inspected",
        "outcomes_inspected_at_utc",
        "analysis_dispositions",
        "before",
        "after",
        "snapshots",
        "overwrite_policy",
    }
    evidence_schema = evidence.get("schema_version")
    if type(evidence_schema) is not int:
        raise _AuthorityValidationError("amendment evidence schema must be an exact integer")
    recovery_branch = False
    resource_branch = False
    resource_technical_branch = False
    if evidence_schema == 1:
        required_keys = base_required_keys
    elif evidence_schema == 2:
        required_keys = {*base_required_keys, "finalization_successor_authorization"}
        if "confirmatory_storage_policy" in evidence:
            required_keys.add("confirmatory_storage_policy")
    elif evidence_schema == 3:
        has_finalization = "finalization_successor_authorization" in evidence
        has_recovery = "primary_recovery_authorization" in evidence
        if has_finalization == has_recovery:
            raise _AuthorityValidationError(
                "schema-v3 amendment must contain exactly one finalization or primary "
                "recovery authorization"
            )
        recovery_branch = has_recovery
        authorization_field = (
            "primary_recovery_authorization"
            if recovery_branch
            else "finalization_successor_authorization"
        )
        required_keys = {*base_required_keys, authorization_field}
        if "confirmatory_storage_policy" in evidence:
            required_keys.add("confirmatory_storage_policy")
    elif evidence_schema == 4:
        resource_branch = True
        required_keys = {
            *base_required_keys,
            "amendment_purpose",
            "resource_bounded_confirmatory_authorization",
            "confirmatory_storage_policy",
        }
    elif evidence_schema == 5:
        resource_technical_branch = True
        required_keys = {
            *base_required_keys,
            "amendment_purpose",
            "resource_bounded_technical_successor_authorization",
            "confirmatory_storage_policy",
        }
    else:
        raise _AuthorityValidationError("amendment evidence schema is unsupported")
    if set(evidence) != required_keys:
        raise _AuthorityValidationError(
            f"amendment evidence must contain exactly the strict schema-v{evidence_schema} fields"
        )
    if evidence.get("authority_kind") != _AUTHORITY_KIND:
        raise _AuthorityValidationError("amendment evidence schema/kind is invalid")
    timestamp_value = evidence.get("amendment_timestamp_utc")
    timestamp = _parse_timestamp(timestamp_value, role="amendment timestamp")
    if marker.get("amendment_timestamp_utc") != timestamp_value:
        raise _AuthorityValidationError("amendment marker timestamp differs from its evidence")
    reason = _normalise_text(str(evidence.get("reason", "")), role="amendment reason")
    if reason != evidence.get("reason"):
        raise _AuthorityValidationError("amendment reason is not in canonical form")
    hypotheses = _normalise_items(
        evidence.get("affected_hypotheses", ()), role="affected hypotheses"
    )
    analyses = _normalise_items(evidence.get("affected_analyses", ()), role="affected analyses")
    if list(hypotheses) != evidence.get("affected_hypotheses") or list(analyses) != evidence.get(
        "affected_analyses"
    ):
        raise _AuthorityValidationError("amendment affected-item lists are not canonical")
    outcomes_inspected = evidence.get("outcomes_inspected")
    if not isinstance(outcomes_inspected, bool):
        raise _AuthorityValidationError("outcomes_inspected must be an exact boolean")
    outcomes_time = evidence.get("outcomes_inspected_at_utc")
    if outcomes_inspected:
        inspected_at = _parse_timestamp(outcomes_time, role="outcome-inspection timestamp")
        if inspected_at > timestamp:
            raise _AuthorityValidationError(
                "outcome-inspection timestamp cannot be later than the amendment"
            )
    elif outcomes_time is not None:
        raise _AuthorityValidationError(
            "outcomes_inspected_at_utc must be null when outcomes_inspected is false"
        )
    expected_dispositions = _analysis_dispositions(analyses, outcomes_inspected=outcomes_inspected)
    if evidence.get("analysis_dispositions") != expected_dispositions:
        raise _AuthorityValidationError(
            "affected-analysis dispositions do not enforce the amendment reporting policy"
        )
    parent = evidence.get("parent")
    if not isinstance(parent, Mapping) or set(parent) != {
        "authority_directory",
        "authority_kind",
        "artifact_root_sha256",
        "sha256_manifest_sha256",
        "chain_depth",
    }:
        raise _AuthorityValidationError("amendment parent record is invalid")
    parent_value = parent.get("authority_directory")
    if not isinstance(parent_value, str) or not Path(parent_value).is_absolute():
        raise _AuthorityValidationError("amendment parent path must be explicit and absolute")
    parent_directory = _require_real_directory(
        Path(parent_value), role="amendment parent authority"
    )
    parent_state = _authority_state(
        parent_directory,
        visited=visited,
        depth=depth + 1,
        max_chain_depth=max_chain_depth,
    )
    if recovery_branch:
        if outcomes_inspected is not True:
            raise _AuthorityValidationError(
                "primary recovery amendment requires outcomes_inspected=true"
            )
        authorization = evidence.get("primary_recovery_authorization")
        if not isinstance(authorization, Mapping):
            raise _AuthorityValidationError("primary recovery authorization must be a JSON object")
        canonical_recovery = _canonical_primary_recovery_authorization(
            authorization,
            parent_state=parent_state,
            outcomes_inspected=outcomes_inspected,
            outcomes_inspected_at_utc=(
                str(outcomes_time) if isinstance(outcomes_time, str) else None
            ),
            amendment_reason=reason,
        )
        if dict(authorization) != canonical_recovery:
            raise _AuthorityValidationError(
                "primary recovery authorization is not in canonical form"
            )
        observed_at = _parse_timestamp(
            canonical_recovery["interruption_evidence"]["observed_at_utc"],
            role="primary recovery interruption-observation timestamp",
        )
        if observed_at > timestamp:
            raise _AuthorityValidationError(
                "primary recovery interruption observation cannot be later than the amendment"
            )
    elif evidence_schema in {2, 3}:
        if outcomes_inspected is not False:
            raise _AuthorityValidationError(
                "finalization-only successor amendment requires outcomes_inspected=false"
            )
        authorization = evidence.get("finalization_successor_authorization")
        if not isinstance(authorization, Mapping):
            raise _AuthorityValidationError(
                "finalization-successor authorization must be a JSON object"
            )
        canonical_authorization = _canonical_finalization_successor_authorization(
            authorization,
            parent_state=parent_state,
            verify_live_predecessor=False,
        )
        if dict(authorization) != canonical_authorization:
            raise _AuthorityValidationError(
                "finalization-successor authorization is not in canonical form"
            )
        expected_authorization_schema = 1 if evidence_schema == 2 else 2
        if (
            type(authorization.get("schema_version")) is not int
            or authorization.get("schema_version") != expected_authorization_schema
        ):
            raise _AuthorityValidationError(
                "amendment/finalization authorization schema versions are inconsistent"
            )
    elif resource_branch:
        if (
            outcomes_inspected is not True
            or evidence.get("amendment_purpose") != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        ):
            raise _AuthorityValidationError(
                "schema-v4 resource-bounded amendment requires its exact purpose and "
                "outcomes_inspected=true"
            )
        if not isinstance(
            evidence.get("resource_bounded_confirmatory_authorization"),
            Mapping,
        ):
            raise _AuthorityValidationError(
                "resource-bounded confirmatory authorization must be a JSON object"
            )
        _resource_parent_recovery_authorization(parent_state)
    elif resource_technical_branch:
        if (
            outcomes_inspected is not True
            or evidence.get("amendment_purpose") != RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE
        ):
            raise _AuthorityValidationError(
                "schema-v5 resource technical successor requires its exact purpose "
                "and outcomes_inspected=true"
            )
        if not isinstance(
            evidence.get("resource_bounded_technical_successor_authorization"),
            Mapping,
        ):
            raise _AuthorityValidationError(
                "resource technical successor authorization must be a JSON object"
            )
        parent_evidence = _json_mapping(
            parent_state.directory / _EVIDENCE_FILENAME,
            role="resource technical successor parent evidence",
        )
        if (
            parent_evidence.get("schema_version") != 4
            or parent_evidence.get("amendment_purpose")
            != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        ):
            raise _AuthorityValidationError(
                "schema-v5 resource technical successor must be a direct child of C"
            )
    if "confirmatory_storage_policy" in evidence:
        storage_policy = evidence.get("confirmatory_storage_policy")
        if not isinstance(storage_policy, Mapping):
            raise _AuthorityValidationError("confirmatory storage policy must be a JSON object")
        canonical_storage_policy = _canonical_confirmatory_storage_policy(storage_policy)
        if dict(storage_policy) != canonical_storage_policy:
            raise _AuthorityValidationError("confirmatory storage policy is not in canonical form")
        if resource_branch:
            parent_storage_policy = require_confirmatory_storage_policy(parent_state.directory)
            if dict(storage_policy) != parent_storage_policy:
                raise _AuthorityValidationError(
                    "resource-bounded confirmatory storage policy must be inherited "
                    "unchanged from its recovery parent"
                )
        elif resource_technical_branch:
            parent_storage_policy = (
                _require_historical_resource_bounded_confirmatory_storage_policy(
                    parent_state.directory
                )
            )
            if dict(storage_policy) != parent_storage_policy:
                raise _AuthorityValidationError(
                    "resource technical successor must inherit C's storage policy unchanged"
                )
    expected_parent = {
        "authority_directory": str(parent_state.directory),
        "authority_kind": parent_state.kind,
        "artifact_root_sha256": parent_state.artifact_root_sha256,
        "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
        "chain_depth": parent_state.chain_depth,
    }
    if dict(parent) != expected_parent:
        raise _AuthorityValidationError("amendment parent hashes/identity differ from live parent")
    if timestamp <= parent_state.timestamp:
        raise _AuthorityValidationError("amendment timestamp must be later than its parent")
    if evidence.get("chain_depth") != parent_state.chain_depth + 1:
        raise _AuthorityValidationError("amendment chain depth is inconsistent with its parent")
    current_hashes = _snapshot_hashes(directory)
    if evidence.get("before") != parent_state.snapshot_hashes:
        raise _AuthorityValidationError("amendment before-hashes differ from its parent snapshots")
    if evidence.get("after") != current_hashes:
        raise _AuthorityValidationError("amendment after-hashes differ from its full snapshots")
    if resource_branch:
        before = parent_state.snapshot_hashes
        for section in ("preregistration", "primary_config"):
            before_section = before.get(section)
            after_section = current_hashes.get(section)
            if (
                not isinstance(before_section, Mapping)
                or not isinstance(after_section, Mapping)
                or dict(before_section) != dict(after_section)
            ):
                raise _AuthorityValidationError(
                    "resource-bounded amendment requires unchanged preregistration "
                    f"and primary config; {section} differs"
                )
        before_confirmatory = before.get("confirmatory_config")
        after_confirmatory = current_hashes.get("confirmatory_config")
        if (
            not isinstance(before_confirmatory, Mapping)
            or not isinstance(after_confirmatory, Mapping)
            or before_confirmatory.get("file_sha256") == after_confirmatory.get("file_sha256")
            or before_confirmatory.get("semantic_sha256")
            == after_confirmatory.get("semantic_sha256")
        ):
            raise _AuthorityValidationError(
                "resource-bounded amendment requires an exact changed confirmatory profile"
            )
        execution_hashes = current_hashes.get("execution_source")
        if not isinstance(execution_hashes, Mapping):
            raise _AuthorityValidationError(
                "resource-bounded amendment lacks execution-source snapshot hashes"
            )
        resource_source = _json_mapping(
            directory / _SOURCE_TREE_SNAPSHOT,
            role="resource-bounded execution source",
        )
        resource_authorization = evidence["resource_bounded_confirmatory_authorization"]
        if not isinstance(resource_authorization, Mapping):  # Defensive for type narrowing.
            raise _AuthorityValidationError(
                "resource-bounded confirmatory authorization must be a mapping"
            )
        canonical_resource = _canonical_resource_bounded_confirmatory_authorization(
            resource_authorization,
            parent_state=parent_state,
            resource_source=resource_source,
            resource_source_manifest_sha256=_require_sha256(
                execution_hashes.get("manifest_sha256"),
                role="resource-bounded execution-source manifest",
            ),
            resource_config_file_sha256=_require_sha256(
                after_confirmatory.get("file_sha256"),
                role="resource-bounded confirmatory config file",
            ),
            resource_config_semantic_sha256=_require_sha256(
                after_confirmatory.get("semantic_sha256"),
                role="resource-bounded confirmatory config semantics",
            ),
            verify_live_primary=False,
        )
        if dict(resource_authorization) != canonical_resource:
            raise _AuthorityValidationError(
                "resource-bounded confirmatory authorization is not in canonical form"
            )
    elif resource_technical_branch:
        _require_resource_technical_parent_identity(
            parent_state.snapshot_hashes,
            current_hashes,
        )
        before_execution = parent_state.snapshot_hashes.get("execution_source")
        after_execution = current_hashes.get("execution_source")
        if (
            not isinstance(before_execution, Mapping)
            or not isinstance(after_execution, Mapping)
            or before_execution.get("root_sha256") == after_execution.get("root_sha256")
        ):
            raise _AuthorityValidationError(
                "resource technical successor requires one actual execution-source change"
            )
        parent_resource_authorization = _require_resource_bounded_confirmatory_authorization(
            parent_state.directory,
            verify_live_primary=False,
        )
        successor_authorization = evidence["resource_bounded_technical_successor_authorization"]
        if not isinstance(successor_authorization, Mapping):
            raise _AuthorityValidationError(
                "resource technical successor authorization must be a mapping"
            )
        successor_source = _json_mapping(
            directory / _SOURCE_TREE_SNAPSHOT,
            role="resource technical successor execution source",
        )
        successor_config = load_config(directory / _CONFIRMATORY_CONFIG_SNAPSHOT)
        if not isinstance(after_execution, Mapping):
            raise _AuthorityValidationError(
                "resource technical successor lacks execution-source hashes"
            )
        after_confirmatory = current_hashes.get("confirmatory_config")
        if not isinstance(after_confirmatory, Mapping):
            raise _AuthorityValidationError(
                "resource technical successor lacks confirmatory-config hashes"
            )
        canonical_successor = _canonical_resource_bounded_technical_successor_authorization(
            successor_authorization,
            parent_state=parent_state,
            parent_resource_authorization=parent_resource_authorization,
            successor_source=successor_source,
            successor_source_manifest_sha256=_require_sha256(
                after_execution.get("manifest_sha256"),
                role="resource technical successor execution-source manifest",
            ),
            successor_config=successor_config,
            successor_config_file_sha256=_require_sha256(
                after_confirmatory.get("file_sha256"),
                role="resource technical successor confirmatory config file",
            ),
            successor_config_semantic_sha256=_require_sha256(
                after_confirmatory.get("semantic_sha256"),
                role="resource technical successor confirmatory config semantics",
            ),
            verify_live_receipt=False,
        )
        if dict(successor_authorization) != canonical_successor:
            raise _AuthorityValidationError(
                "resource technical successor authorization is not in canonical form"
            )
        failed_preflight = canonical_successor["failed_preflight"]["evidence"]
        failed_at = _parse_timestamp(
            failed_preflight["observed_at_utc"],
            role="resource technical successor failed-preflight timestamp",
        )
        prior_publication_failure = canonical_successor["prior_publication_failure"]["evidence"]
        prior_publication_failure_at = _parse_timestamp(
            prior_publication_failure["observed_at_utc"],
            role="resource technical successor prior-publication failure timestamp",
        )
        if (
            failed_at <= parent_state.timestamp
            or failed_at >= prior_publication_failure_at
            or prior_publication_failure_at > timestamp
        ):
            raise _AuthorityValidationError(
                "resource technical successor timestamps do not follow the exact "
                "C < failed-preflight < failed-publication receipt <= D order"
            )
        if canonical_successor.get("schema_version") == 3:
            lineage = canonical_successor["replacement_publication_failure_lineage"]
            qualification_at = _parse_timestamp(
                lineage["terminal_qualification_receipt"]["qualified_at_utc"],
                role="resource technical successor replacement qualification timestamp",
            )
            if qualification_at <= prior_publication_failure_at or qualification_at > timestamp:
                raise _AuthorityValidationError(
                    "schema-v3 resource technical successor timestamps do not follow "
                    "the failed-publication receipt < replacement qualification <= D order"
                )
    elif "confirmatory_storage_policy" in evidence or recovery_branch:
        _require_storage_only_scientific_identity(
            parent_state.snapshot_hashes,
            current_hashes,
        )
    if _frozen_scientific_identity(parent_state.snapshot_hashes) == (
        _frozen_scientific_identity(current_hashes)
    ):
        raise _AuthorityValidationError("amendment must contain at least one actual frozen change")
    expected_snapshots = {
        "preregistration": _PREREGISTRATION_SNAPSHOT,
        "primary_config": _PRIMARY_CONFIG_SNAPSHOT,
        "confirmatory_config": _CONFIRMATORY_CONFIG_SNAPSHOT,
        "execution_source": _SOURCE_TREE_SNAPSHOT,
        "report": _REPORT_FILENAME,
    }
    if evidence.get("snapshots") != expected_snapshots:
        raise _AuthorityValidationError("amendment snapshot-name contract is invalid")
    if evidence.get("overwrite_policy") != "immutable successor; never overwrite parent or peer":
        raise _AuthorityValidationError("amendment overwrite policy is invalid")
    return _AuthorityState(
        directory=directory,
        kind="preregistration_amendment",
        timestamp_utc=str(timestamp_value),
        timestamp=timestamp,
        chain_depth=parent_state.chain_depth + 1,
        artifact_root_sha256=integrity.artifact_root_sha256,
        sha256_manifest_sha256=integrity.sha256_manifest_sha256,
        snapshot_hashes=current_hashes,
        parent_directory=parent_state.directory,
    )


def _authority_state(
    authority_directory: Path,
    *,
    visited: set[tuple[str, int, int]],
    depth: int,
    max_chain_depth: int,
) -> _AuthorityState:
    if depth > max_chain_depth:
        raise _AuthorityValidationError(
            f"amendment chain exceeds the maximum depth of {max_chain_depth}"
        )
    directory = _require_real_directory(authority_directory, role="preregistration authority")
    key = _authority_key(directory)
    if key in visited:
        raise _AuthorityValidationError(f"amendment parent cycle detected at {directory}")
    next_visited = {*visited, key}
    marker = _json_mapping(directory / _IMMUTABLE_MARKER, role="authority immutable marker")
    status = marker.get("status")
    if status == "frozen":
        return _base_authority_state(directory)
    if status != "amended" or marker.get("authority_kind") != _AUTHORITY_KIND:
        raise _AuthorityValidationError(
            f"authority is neither a valid base freeze nor amendment: {directory}"
        )
    integrity = _verify_amendment_bundle(directory)
    return _validate_amendment_evidence(
        directory,
        integrity=integrity,
        visited=next_visited,
        depth=depth,
        max_chain_depth=max_chain_depth,
    )


def verify_preregistration_amendment(
    amendment_directory: str | Path,
    *,
    max_chain_depth: int = _MAX_CHAIN_DEPTH,
) -> PreregistrationAmendmentVerification:
    """Verify an amendment bundle and its complete immutable parent chain."""

    supplied = Path(amendment_directory).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    directory = Path(os.path.abspath(supplied))
    if (
        not isinstance(max_chain_depth, int)
        or isinstance(max_chain_depth, bool)
        or max_chain_depth < 1
    ):
        return PreregistrationAmendmentVerification(
            False,
            directory,
            None,
            None,
            None,
            None,
            ("max_chain_depth must be a positive exact integer",),
        )
    try:
        state = _authority_state(
            directory,
            visited=set(),
            depth=0,
            max_chain_depth=max_chain_depth,
        )
        if state.kind != "preregistration_amendment":
            raise _AuthorityValidationError(
                "requested authority is a base freeze, not an amendment"
            )
    except (OSError, ValueError, TypeError) as exc:
        return PreregistrationAmendmentVerification(
            False,
            directory,
            None,
            None,
            None,
            None,
            (f"{type(exc).__name__}: {exc}",),
        )
    return PreregistrationAmendmentVerification(
        True,
        directory,
        state.chain_depth,
        state.artifact_root_sha256,
        state.sha256_manifest_sha256,
        state.parent_directory,
        (),
    )


def _require_finalization_successor_authorization(
    amendment_directory: str | Path,
    *,
    verify_live_predecessor: bool,
) -> dict[str, Any]:
    verification = verify_preregistration_amendment(amendment_directory)
    if not verification.valid:
        raise _AuthorityValidationError(
            "finalization-successor amendment failed chain/integrity verification: "
            + "; ".join(verification.errors)
        )
    directory = _require_real_directory(
        Path(amendment_directory).expanduser().resolve(),
        role="finalization-successor amendment",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if type(evidence.get("schema_version")) is not int or evidence.get("schema_version") not in {
        2,
        3,
    }:
        raise _AuthorityValidationError(
            "amendment does not contain a finalization-successor authorization"
        )
    parent = evidence.get("parent")
    authorization = evidence.get("finalization_successor_authorization")
    if not isinstance(parent, Mapping) or not isinstance(authorization, Mapping):
        raise _AuthorityValidationError(
            "finalization-successor amendment lacks its strict parent/binding objects"
        )
    parent_directory = _absolute_record_path(
        parent.get("authority_directory"), role="finalization-successor parent authority"
    )
    parent_state = _authority_state(
        parent_directory,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    canonical = _canonical_finalization_successor_authorization(
        authorization,
        parent_state=parent_state,
        verify_live_predecessor=verify_live_predecessor,
    )
    if dict(authorization) != canonical:
        raise _AuthorityValidationError(
            "finalization-successor authorization is not in canonical form"
        )
    return canonical


def require_finalization_successor_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return the canonical binding after rechecking its live failed seal.

    This is the narrow execution API for a finalization-only CLI/runner. It verifies
    the complete amendment chain, the predecessor's complete technical integrity,
    exact append-only integrity-registry record, and sealed structural authority
    files. Post-seal consumers should use the internal structural readback only after
    one fresh live verification has already been bound into the successor seal.
    """

    return _require_finalization_successor_authorization(
        amendment_directory,
        verify_live_predecessor=True,
    )


def require_primary_recovery_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return one canonical recovery payload from a verified immutable authority.

    The returned mapping is the sealed policy and expected-digest contract. The
    recovery runner remains responsible for one fresh, typed comparison with the
    unsealed source run and interruption receipt before it creates any output.
    """

    verification = verify_preregistration_amendment(amendment_directory)
    if not verification.valid:
        raise _AuthorityValidationError(
            "primary recovery amendment failed chain/integrity verification: "
            + "; ".join(verification.errors)
        )
    directory = _require_real_directory(
        Path(amendment_directory).expanduser().resolve(),
        role="primary recovery amendment",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 3
        or "finalization_successor_authorization" in evidence
    ):
        raise _AuthorityValidationError(
            "amendment does not contain an exclusive primary recovery authorization"
        )
    authorization = evidence.get("primary_recovery_authorization")
    parent = evidence.get("parent")
    if not isinstance(authorization, Mapping) or not isinstance(parent, Mapping):
        raise _AuthorityValidationError(
            "primary recovery amendment lacks its strict parent/binding objects"
        )
    parent_directory = _absolute_record_path(
        parent.get("authority_directory"),
        role="primary recovery parent authority",
    )
    parent_state = _authority_state(
        parent_directory,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    outcomes_time = evidence.get("outcomes_inspected_at_utc")
    canonical = _canonical_primary_recovery_authorization(
        authorization,
        parent_state=parent_state,
        outcomes_inspected=evidence.get("outcomes_inspected") is True,
        outcomes_inspected_at_utc=(str(outcomes_time) if isinstance(outcomes_time, str) else None),
        amendment_reason=_normalise_text(
            str(evidence.get("reason", "")),
            role="amendment reason",
        ),
    )
    if dict(authorization) != canonical:
        raise _AuthorityValidationError("primary recovery authorization is not in canonical form")
    return canonical


def build_resource_bounded_confirmatory_authorization(
    *,
    project_root: str | Path,
    parent_recovery_authority_directory: str | Path,
    primary_run_directory: str | Path,
    resource_confirmatory_config_path: str | Path,
    resource_profile_id: str,
) -> ResourceBoundedConfirmatoryAuthorization:
    """Derive one exact schema-v4 authorization from live hash-only evidence.

    This builder reads no scientific outcome value. It verifies the sealed recovery
    run and its positive stage receipt, then binds only file/root hashes, the exact
    resource config, and the closed source-delta receipt.
    """

    root = _require_real_directory(
        Path(project_root).expanduser().resolve(),
        role="resource-bounded project root",
    )
    parent = _require_real_directory(
        Path(parent_recovery_authority_directory).expanduser().resolve(),
        role="resource-bounded recovery authority",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    recovery_authorization = _resource_parent_recovery_authorization(parent_state)
    recovery_authorization_sha256 = _canonical_mapping_sha256(recovery_authorization)
    run = _require_real_directory(
        Path(primary_run_directory).expanduser().resolve(),
        role="resource-bounded historical primary",
    )
    integrity = verify_run_integrity(run)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != run.name
        or not isinstance(integrity.expected_root_sha256, str)
        or _SHA256.fullmatch(integrity.expected_root_sha256) is None
        or integrity.actual_root_sha256 != integrity.expected_root_sha256
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary is not an exact registry-backed seal"
        )
    receipt = require_run_stage_eligibility_receipt(run, integrity=integrity)
    if (
        receipt is None
        or not receipt.valid
        or receipt.run_directory.resolve() != run
        or receipt.run_id != run.name
        or receipt.completion_stage != "PRIMARY_STUDY_COMPLETE"
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary lacks a positive stage receipt"
        )
    required_paths = {
        "artifact_manifest": run / RUN_ARTIFACT_MANIFEST_FILENAME,
        "completion": run / "completion_evidence.json",
        "execution_gate": run / "primary_execution_gate.json",
        "recovery_evidence": run / "primary_recovery_evidence.json",
    }
    for role, path in required_paths.items():
        _require_regular_file(path, role=f"resource-bounded primary {role}")
    gate = _json_mapping(
        required_paths["execution_gate"],
        role="resource-bounded historical primary execution gate",
    )
    if (
        gate.get("freeze_directory") != str(parent_state.directory)
        or gate.get("freeze_artifact_root_sha256") != parent_state.artifact_root_sha256
        or gate.get("freeze_manifest_sha256") != parent_state.sha256_manifest_sha256
    ):
        raise _AuthorityValidationError(
            "resource-bounded historical primary gate differs from the recovery parent"
        )

    config_path = Path(resource_confirmatory_config_path).expanduser().resolve()
    expected_config_path = (
        root / "configs" / "confirmatory_resource_bounded_amended.yaml"
    ).resolve()
    if config_path != expected_config_path:
        raise _AuthorityValidationError(
            "resource-bounded confirmatory config must use its canonical project path"
        )
    _require_regular_file(config_path, role="resource-bounded confirmatory config")
    resource_config = load_config(config_path)
    if (
        resource_profile_id != _RESOURCE_BOUNDED_PROFILE_ID
        or resource_config.get("execution_profile") != _RESOURCE_BOUNDED_PROFILE_ID
        or resource_config.get("analysis_disposition") != _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION
        or resource_config.get("original_confirmatory_claim_allowed") is not False
        or resource_config.get("completion_stage") is not None
    ):
        raise _AuthorityValidationError(
            "resource-bounded confirmatory config does not expose the exact "
            "post-outcome resource profile"
        )
    current_source = capture_source_tree(root)
    parent_source = _json_mapping(
        parent_state.directory / _SOURCE_TREE_SNAPSHOT,
        role="resource-bounded parent execution source",
    )
    source_delta, source_delta_sha256 = _canonical_resource_source_delta(
        parent_source,
        current_source,
    )
    parent_confirmatory = parent_state.snapshot_hashes.get("confirmatory_config")
    parent_execution = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_confirmatory, Mapping) or not isinstance(parent_execution, Mapping):
        raise _AuthorityValidationError(
            "resource-bounded recovery parent lacks config/source evidence"
        )
    authorization = ResourceBoundedConfirmatoryAuthorization(
        primary_run_id=run.name,
        primary_run_directory=run,
        primary_artifact_root_sha256=integrity.expected_root_sha256,
        primary_artifact_manifest_sha256=sha256_file(required_paths["artifact_manifest"]),
        primary_completion_evidence_sha256=sha256_file(required_paths["completion"]),
        primary_execution_gate_sha256=sha256_file(required_paths["execution_gate"]),
        primary_stage_attestation_record_sha256=receipt.record_sha256,
        primary_stage_attestation_verification_sha256=receipt.verification_sha256,
        primary_recovery_evidence_sha256=sha256_file(required_paths["recovery_evidence"]),
        primary_recovery_authorization_sha256=recovery_authorization_sha256,
        recovery_authority_directory=parent_state.directory,
        recovery_authority_artifact_root_sha256=parent_state.artifact_root_sha256,
        recovery_authority_manifest_sha256=parent_state.sha256_manifest_sha256,
        recovery_authority_chain_depth=parent_state.chain_depth,
        resource_profile_id=resource_profile_id,
        parent_confirmatory_config_file_sha256=_require_sha256(
            parent_confirmatory.get("file_sha256"),
            role="resource-bounded parent confirmatory config file",
        ),
        parent_confirmatory_config_semantic_sha256=_require_sha256(
            parent_confirmatory.get("semantic_sha256"),
            role="resource-bounded parent confirmatory config semantics",
        ),
        resource_confirmatory_config_file_sha256=sha256_file(config_path),
        resource_confirmatory_config_semantic_sha256=config_sha256(resource_config),
        parent_execution_source_root_sha256=_require_sha256(
            parent_execution.get("root_sha256"),
            role="resource-bounded parent execution-source root",
        ),
        parent_execution_source_manifest_sha256=_require_sha256(
            parent_execution.get("manifest_sha256"),
            role="resource-bounded parent execution-source manifest",
        ),
        resource_execution_source_root_sha256=_require_sha256(
            current_source.get("root_sha256"),
            role="resource-bounded execution-source root",
        ),
        resource_execution_source_manifest_sha256=_atomic_json_sha256(current_source),
        source_delta_records=source_delta,
        source_delta_sha256=source_delta_sha256,
    )
    canonical = _canonical_resource_bounded_confirmatory_authorization(
        authorization,
        parent_state=parent_state,
        resource_source=current_source,
        resource_source_manifest_sha256=_atomic_json_sha256(current_source),
        resource_config_file_sha256=sha256_file(config_path),
        resource_config_semantic_sha256=config_sha256(resource_config),
        verify_live_primary=True,
    )
    if canonical != authorization.as_dict():
        raise RuntimeError("resource-bounded authorization builder produced noncanonical evidence")
    return authorization


def build_resource_bounded_prior_publication_failure_evidence(
    *,
    superseded_resource_authority_directory: str | Path,
    observed_at: datetime,
) -> dict[str, Any]:
    """Build the exact read-only receipt payload for the consumed v1 D attempt."""

    parent = _require_real_directory(
        Path(superseded_resource_authority_directory).expanduser(),
        role="superseded resource authority C",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    _require_resource_bounded_confirmatory_authorization(
        parent,
        verify_live_primary=False,
    )
    if observed_at.tzinfo is None:
        raise ValueError("prior-publication failure observation must be timezone-aware")
    observed_timestamp = observed_at.astimezone(UTC)
    observed_text = observed_timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    artifacts_root = Path(os.path.abspath(parent.parent.parent))
    control_root = _require_real_directory(
        artifacts_root / "resource_control",
        role="resource authority D prior publication control root",
    )
    run_state_root = _require_real_directory(
        artifacts_root / "runs",
        role="resource authority D prior publication run-state root",
    )
    frozen_input_root = _require_real_directory(
        control_root / "authority_d_inputs_20260727Tfinal_source_v2",
        role="resource authority D prior frozen-v2 input root",
    )

    def file_record(
        path: Path,
        *,
        role: str,
        embedded_json: bool,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        payload = _read_stable_single_link_file(
            path,
            role=role,
            allow_empty=allow_empty,
        )
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if embedded_json:
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise _AuthorityValidationError(f"{role} is invalid JSON") from exc
            if not isinstance(decoded, Mapping):
                raise _AuthorityValidationError(f"{role} must contain a JSON object")
            record["payload"] = dict(decoded)
        return record

    attempt = file_record(
        control_root / "resource_authority_d_publication_attempt.json",
        role="resource authority D prior publication attempt marker",
        embedded_json=True,
    )
    failure = file_record(
        control_root / "resource_authority_d_publication_failure.json",
        role="resource authority D prior publication failure marker",
        embedded_json=True,
    )
    stdout = file_record(
        control_root / "authority_d_v2_publish_20260727T212335.140Z.stdout.log",
        role="resource authority D prior publication stdout",
        embedded_json=True,
    )
    stderr = file_record(
        control_root / "authority_d_v2_publish_20260727T212335.140Z.stderr.log",
        role="resource authority D prior publication stderr",
        embedded_json=False,
        allow_empty=True,
    )
    controller = file_record(
        control_root / "prepare_resource_authority_d_once.py",
        role="resource authority D prior publication controller",
        embedded_json=False,
    )
    expected_frozen_names = tuple(sorted(_RESOURCE_BOUNDED_PRIOR_PUBLICATION_INPUT_FILES))
    observed_frozen_children = tuple(sorted(path.name for path in frozen_input_root.iterdir()))
    if observed_frozen_children != expected_frozen_names:
        raise _AuthorityValidationError("resource authority D prior frozen-v2 input set changed")
    frozen_records = [
        file_record(
            frozen_input_root / name,
            role=f"resource authority D prior frozen-v2 input {name}",
            embedded_json=False,
        )
        for name in expected_frozen_names
    ]
    attempt_payload = attempt.get("payload")
    failure_payload = failure.get("payload")
    if not isinstance(attempt_payload, Mapping) or not isinstance(failure_payload, Mapping):
        raise RuntimeError("prior publication marker builder lost JSON payloads")
    run_state = attempt_payload.get("run_state_before")
    failed_destination = parent.parent / "20260727T212711.019137Z"
    if not isinstance(run_state, Mapping):
        raise _AuthorityValidationError(
            "resource authority D prior publication attempt lacks run state"
        )
    for name, expected_sha256 in sorted(run_state.items()):
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise _AuthorityValidationError(
                "resource authority D prior publication run state is invalid"
            )
        live_payload = _read_stable_single_link_file(
            run_state_root / name,
            role=f"resource authority D prior publication run-state file {name}",
            allow_empty=True,
        )
        if hashlib.sha256(live_payload).hexdigest() != expected_sha256:
            raise _AuthorityValidationError(
                "resource authority D prior publication run state changed"
            )
    success_marker = control_root / "resource_authority_d_publication_success.json"
    if os.path.lexists(success_marker) or os.path.lexists(failed_destination):
        raise _AuthorityValidationError(
            "resource authority D prior publication is not terminally absent"
        )
    expected_error_message = (
        "published amendment failed independent verification: "
        "_AuthorityValidationError: resource authority C is historically valid "
        "but no longer the effective execution leaf; use successor D "
        f"{failed_destination}"
    )
    evidence = {
        "schema_version": 1,
        "policy": _RESOURCE_BOUNDED_PRIOR_PUBLICATION_FAILURE_POLICY,
        "observed_at_utc": observed_text,
        "attempt_marker": attempt,
        "failure_marker": failure,
        "terminal_stdout": stdout,
        "terminal_stderr": stderr,
        "executed_controller": controller,
        "frozen_v2_inputs": {
            "directory": str(frozen_input_root),
            "records": frozen_records,
            "records_sha256": _canonical_value_sha256(frozen_records),
        },
        "error": {
            "exception_type": "RuntimeError",
            "exception_message": expected_error_message,
            "exception_type_sha256": hashlib.sha256(b"RuntimeError").hexdigest(),
            "exception_sha256": hashlib.sha256(
                f"RuntimeError: {expected_error_message}".encode()
            ).hexdigest(),
        },
        "run_state": {
            "root": str(run_state_root),
            "files": dict(run_state),
            "canonical_sha256": _canonical_mapping_sha256(run_state),
        },
        "absent_paths": {
            "prior_success_marker": str(success_marker),
            "failed_intended_authority": str(failed_destination),
        },
        "disposition": {
            "prior_attempt_consumed": True,
            "prior_authority_published": False,
            "failed_intended_authority_absent": True,
            "prior_success_marker_absent": True,
            "replacement_mode": ("manual_new_one_shot_after_rolled_back_publication"),
            "automatic_retry_allowed": False,
            "scientific_outcome_values_read": False,
            "scientific_profile_changed": False,
        },
    }
    receipt_path = control_root / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    canonical = _canonical_prior_resource_authority_d_publication_failure(
        {
            "receipt_path": str(receipt_path),
            "receipt_sha256": _atomic_json_sha256(evidence),
            "evidence": evidence,
        },
        resource_parent=parent_state,
        verify_live_receipt=False,
    )
    if canonical["evidence"] != evidence:
        raise RuntimeError(
            "prior-publication failure evidence builder produced noncanonical evidence"
        )
    return evidence


def verify_resource_bounded_prior_publication_failure_receipt(
    *,
    superseded_resource_authority_directory: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the canonical one-shot receipt against all live historical evidence."""

    parent = _require_real_directory(
        Path(superseded_resource_authority_directory).expanduser(),
        role="superseded resource authority C",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    expected_path = (
        Path(os.path.abspath(parent.parent.parent))
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    )
    supplied_path = (
        expected_path
        if receipt_path is None
        else Path(os.path.abspath(Path(receipt_path).expanduser()))
    )
    if supplied_path != expected_path:
        raise ValueError("prior-publication failure receipt must use its canonical project path")
    try:
        payload = read_file_anchored(
            supplied_path,
            require_single_link=True,
            max_bytes=_MAX_RESOURCE_CONTROL_RECEIPT_BYTES,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise _AuthorityValidationError(
            "prior-publication failure receipt failed bounded anchored readback"
        ) from exc
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            "prior-publication failure receipt is invalid JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise _AuthorityValidationError(
            "prior-publication failure receipt must contain a JSON object"
        )
    observed_at = _parse_timestamp(
        decoded.get("observed_at_utc"),
        role="prior-publication failure receipt observation",
    )
    rebuilt = build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=parent,
        observed_at=observed_at,
    )
    if dict(decoded) != rebuilt or payload != _atomic_json_bytes(rebuilt):
        raise _AuthorityValidationError(
            "prior-publication failure receipt differs from canonical live evidence"
        )
    canonical = _canonical_prior_resource_authority_d_publication_failure(
        {
            "receipt_path": str(supplied_path),
            "receipt_sha256": hashlib.sha256(payload).hexdigest(),
            "evidence": rebuilt,
        },
        resource_parent=parent_state,
        verify_live_receipt=True,
    )
    if canonical["evidence"] != rebuilt:
        raise RuntimeError(
            "prior-publication failure receipt verifier produced noncanonical evidence"
        )
    return canonical


def publish_resource_bounded_prior_publication_failure_receipt(
    *,
    superseded_resource_authority_directory: str | Path,
    observed_at: datetime,
) -> dict[str, Any]:
    """Publish and verify the canonical historical-failure receipt exactly once.

    The final path is created with no-overwrite semantics. Any failure after this
    call creates the file rolls back only the exact object owned by this call.
    """

    parent = Path(os.path.abspath(Path(superseded_resource_authority_directory).expanduser()))
    evidence = build_resource_bounded_prior_publication_failure_evidence(
        superseded_resource_authority_directory=parent,
        observed_at=observed_at,
    )
    payload = _atomic_json_bytes(evidence)
    if len(payload) > _MAX_RESOURCE_CONTROL_RECEIPT_BYTES:
        raise ValueError("prior-publication failure receipt exceeds its fixed size limit")
    destination = (
        parent.parent.parent
        / "resource_control"
        / "resource_authority_d_prior_publication_failure_receipt_v1.json"
    )
    published: PublishedPath | None = None
    verification: dict[str, Any] | None = None
    try:
        with ExclusiveBundlePublicationLock(
            [destination],
            role="resource Authority-D prior-publication failure receipt",
        ) as publication_lock:
            published = publish_bytes_no_overwrite(payload, destination)
            publication_lock.assert_owned()
            verification = verify_resource_bounded_prior_publication_failure_receipt(
                superseded_resource_authority_directory=parent,
                receipt_path=destination,
            )
            if (
                verification["receipt_sha256"] != hashlib.sha256(payload).hexdigest()
                or verification["evidence"] != evidence
            ):
                raise RuntimeError("prior-publication failure receipt changed during publication")
            publication_lock.assert_owned()
    except BaseException as publication_error:
        if published is not None:
            try:
                rollback_owned_publications([published])
            except (OSError, RuntimeError) as rollback_error:
                raise RuntimeError(
                    "prior-publication failure receipt publication failed and "
                    "ownership-safe rollback was incomplete"
                ) from rollback_error
        raise publication_error
    if verification is None:  # pragma: no cover - defensive postcondition
        raise RuntimeError("prior-publication failure receipt returned without verification")
    return verification


def build_resource_bounded_technical_successor_authorization(
    *,
    project_root: str | Path,
    superseded_resource_authority_directory: str | Path,
    resource_confirmatory_config_path: str | Path,
    failed_preflight_receipt_path: str | Path,
    prior_publication_failure_receipt_path: str | Path,
    cnn_provenance_correction: (ResourceBoundedCnnProvenanceCorrection | Mapping[str, Any]),
    source_delta_allowlist: Mapping[str, str],
    resource_input_workspace_plan: Mapping[str, Any],
    resource_input_workspace_array_specs: Sequence[ConfirmatoryWorkspaceArraySpec],
    resource_input_workspace_index_specs: Sequence[ConfirmatoryWorkspaceIndexSpec],
    expected_successor_config_semantic_sha256: str,
    replacement_publication_terminal_qualification_receipt_path: str | Path | None = None,
) -> ResourceBoundedTechnicalSuccessorAuthorization:
    """Build D from live hash-only evidence without publishing an amendment.

    The allowlist is caller-supplied, exact, and path-by-path. Wildcards, directory
    prefixes, missing changes, and undeclared changes all fail closed.
    """

    from histo_audit.experiment.confirmatory_memory_workspace import (
        validate_confirmatory_memory_workspace_plan,
    )
    from histo_audit.experiment.study_contracts import (
        build_confirmatory_matrix_plan,
        validate_resource_bounded_confirmatory_config,
    )

    root = _require_real_directory(
        Path(project_root).expanduser(),
        role="resource technical successor project root",
    )
    parent = _require_real_directory(
        Path(superseded_resource_authority_directory).expanduser(),
        role="superseded resource authority C",
    )
    if _resource_technical_successor_candidate_directories(parent):
        raise _AuthorityValidationError(
            "resource authority C already has a technical-successor candidate; "
            "a fork or second D is forbidden"
        )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    parent_authorization = _require_resource_bounded_confirmatory_authorization(
        parent,
        verify_live_primary=False,
    )
    config_path = Path(os.path.abspath(Path(resource_confirmatory_config_path).expanduser()))
    expected_config_path = (
        root / "configs" / "confirmatory_resource_bounded_amended.yaml"
    ).resolve()
    if config_path != expected_config_path:
        raise _AuthorityValidationError(
            "resource technical successor config must use the canonical project path"
        )
    _require_regular_file(
        config_path,
        role="resource technical successor confirmatory config",
    )
    resource_config = validate_resource_bounded_confirmatory_config(load_config(config_path))
    observed_successor_config_semantic_sha256 = config_sha256(resource_config)
    expected_successor_semantic = _require_sha256(
        expected_successor_config_semantic_sha256,
        role="expected resource technical successor config semantics",
    )
    if observed_successor_config_semantic_sha256 != expected_successor_semantic:
        raise _AuthorityValidationError(
            "resource technical successor config differs from the caller's exact "
            "post-integration semantic SHA-256"
        )
    plan = build_confirmatory_matrix_plan(resource_config)
    cnn_cells = sum(
        cell.scenario_id == _RESOURCE_BOUNDED_FAILED_PREFLIGHT_SCENARIO_ID for cell in plan.cells
    )
    observed_shape = {
        "planned_required_cells": plan.required_cell_count,
        "planned_cnn_cells": cnn_cells,
        "planned_cnn_fold_checkpoints": cnn_cells * 5,
    }
    if observed_shape != _RESOURCE_BOUNDED_PROFILE_SHAPE:
        raise _AuthorityValidationError(
            "resource technical successor config changed the exact 24/6/30 profile"
        )
    provider_workspace_plan = validate_confirmatory_memory_workspace_plan(
        resource_input_workspace_plan,
        resource_input_workspace_array_specs,
        resource_input_workspace_index_specs,
        minimum_free_bytes_after=_RESOURCE_BOUNDED_CAPACITY_V3["minimum_free_bytes_before_tracker"],
        maximum_workspace_bytes=_RESOURCE_BOUNDED_CAPACITY_V3["maximum_workspace_bytes"],
    )
    canonical_workspace_plan = _canonical_resource_input_workspace_plan(provider_workspace_plan)
    successor_source = capture_source_tree(root)
    parent_source = _json_mapping(
        parent / _SOURCE_TREE_SNAPSHOT,
        role="resource authority C execution source",
    )
    source_delta, source_delta_sha256 = _canonical_source_delta_with_allowlist(
        parent_source,
        successor_source,
        allowlisted_change_kinds=source_delta_allowlist,
        role="resource technical successor",
    )
    parent_execution = parent_state.snapshot_hashes.get("execution_source")
    parent_profile = parent_authorization.get("resource_profile")
    historical_primary = parent_authorization.get("historical_primary")
    if (
        not isinstance(parent_execution, Mapping)
        or not isinstance(parent_profile, Mapping)
        or not isinstance(historical_primary, Mapping)
    ):
        raise _AuthorityValidationError(
            "resource authority C lacks source, profile, or historical-primary evidence"
        )
    parent_evidence_path = parent / _EVIDENCE_FILENAME
    parent_authorization_sha256 = _canonical_mapping_sha256(parent_authorization)
    prior_failure_receipt_path = Path(
        os.path.abspath(Path(prior_publication_failure_receipt_path).expanduser())
    )
    prior_failure_receipt_bytes = _read_stable_single_link_file(
        prior_failure_receipt_path,
        role="resource technical successor prior-publication failure receipt",
    )
    try:
        prior_failure_receipt_evidence = json.loads(prior_failure_receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure receipt is invalid JSON"
        ) from exc
    if not isinstance(prior_failure_receipt_evidence, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor prior-publication failure receipt must be a JSON object"
        )
    prior_publication_failure = {
        "receipt_path": str(prior_failure_receipt_path),
        "receipt_sha256": hashlib.sha256(prior_failure_receipt_bytes).hexdigest(),
        "evidence": dict(prior_failure_receipt_evidence),
    }
    canonical_prior_publication_failure = _canonical_prior_resource_authority_d_publication_failure(
        prior_publication_failure,
        resource_parent=parent_state,
        verify_live_receipt=True,
    )
    replacement_failure_lineage: dict[str, Any] | None = None
    if replacement_publication_terminal_qualification_receipt_path is not None:
        replacement_failure_lineage = (
            verify_resource_bounded_replacement_terminal_qualification_receipt(
                replacement_publication_terminal_qualification_receipt_path,
                project_root=root,
                parent_authority_directory=parent,
            )
        )
    receipt_path = Path(os.path.abspath(Path(failed_preflight_receipt_path).expanduser()))
    receipt_bytes = _require_regular_file(
        receipt_path,
        role="resource technical successor failed-preflight receipt",
    )
    try:
        receipt_evidence = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight receipt is invalid JSON"
        ) from exc
    if not isinstance(receipt_evidence, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor failed-preflight receipt must be a JSON object"
        )
    failed_preflight = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "evidence": dict(receipt_evidence),
    }
    canonical_failed = _canonical_failed_resource_preflight(
        failed_preflight,
        resource_parent=parent_state,
        resource_authorization_sha256=parent_authorization_sha256,
        verify_live_receipt=True,
    )
    failed_at = _parse_timestamp(
        canonical_failed["evidence"]["observed_at_utc"],
        role="resource technical successor failed-preflight timestamp",
    )
    if failed_at <= parent_state.timestamp:
        raise _AuthorityValidationError(
            "resource technical successor failed preflight must occur after authority C"
        )
    parent_config = load_config(parent / _CONFIRMATORY_CONFIG_SNAPSHOT)
    before_config_record, after_config_record = _require_resource_technical_config_correction(
        parent_config, resource_config
    )
    correction_value = (
        cnn_provenance_correction.as_dict(
            before_config_record=before_config_record,
            after_config_record=after_config_record,
        )
        if isinstance(
            cnn_provenance_correction,
            ResourceBoundedCnnProvenanceCorrection,
        )
        else dict(cnn_provenance_correction)
    )
    canonical_correction = _canonical_cnn_provenance_correction(
        correction_value,
        before_config_record=before_config_record,
        after_config_record=after_config_record,
        parent_source=parent_source,
        successor_source=successor_source,
        failed_preflight=canonical_failed,
    )
    source_delta_payload = {
        "policy": _RESOURCE_BOUNDED_TECHNICAL_SOURCE_DELTA_POLICY,
        "parent_root_sha256": _require_sha256(
            parent_execution.get("root_sha256"),
            role="resource authority C execution-source root",
        ),
        "parent_manifest_sha256": _require_sha256(
            parent_execution.get("manifest_sha256"),
            role="resource authority C execution-source manifest",
        ),
        "resource_root_sha256": _require_sha256(
            successor_source.get("root_sha256"),
            role="resource technical successor execution-source root",
        ),
        "resource_manifest_sha256": _atomic_json_sha256(successor_source),
        "allowlisted_change_kinds": {
            str(path): str(kind)
            for path, kind in sorted(
                source_delta_allowlist.items(),
                key=lambda item: str(item[0]),
            )
        },
        "allowlisted_changes": [dict(record) for record in source_delta],
        "delta_sha256": source_delta_sha256,
    }
    profile_payload = {
        "profile_id": _RESOURCE_BOUNDED_PROFILE_ID,
        "experiment_name": _RESOURCE_BOUNDED_EXPERIMENT_NAME,
        "parent_confirmatory_config_file_sha256": _require_sha256(
            parent_profile.get("resource_confirmatory_config_file_sha256"),
            role="resource authority C confirmatory config file",
        ),
        "parent_confirmatory_config_semantic_sha256": _require_sha256(
            parent_profile.get("resource_confirmatory_config_semantic_sha256"),
            role="resource authority C confirmatory config semantics",
        ),
        "resource_confirmatory_config_file_sha256": sha256_file(config_path),
        "resource_confirmatory_config_semantic_sha256": config_sha256(resource_config),
    }
    supersedes = {
        "authority_directory": str(parent),
        "authority_schema_version": 4,
        "artifact_root_sha256": parent_state.artifact_root_sha256,
        "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
        "chain_depth": parent_state.chain_depth,
        "amendment_evidence_sha256": sha256_file(parent_evidence_path),
        "authorization_sha256": parent_authorization_sha256,
        "effective_execution_leaf": False,
        "historical_verification_retained": True,
    }
    authorization = ResourceBoundedTechnicalSuccessorAuthorization(
        superseded_authority=supersedes,
        prior_publication_failure=canonical_prior_publication_failure,
        failed_preflight=canonical_failed,
        historical_primary=dict(historical_primary),
        resource_profile=profile_payload,
        execution_source_delta=source_delta_payload,
        cnn_provenance_correction=canonical_correction,
        resource_capacity_policy=dict(_RESOURCE_BOUNDED_CAPACITY_V3),
        resource_input_workspace_plan=canonical_workspace_plan,
        expected_successor_config_semantic_sha256=expected_successor_semantic,
        replacement_publication_failure_lineage=replacement_failure_lineage,
    )
    canonical = _canonical_resource_bounded_technical_successor_authorization(
        authorization,
        parent_state=parent_state,
        parent_resource_authorization=parent_authorization,
        successor_source=successor_source,
        successor_source_manifest_sha256=_atomic_json_sha256(successor_source),
        successor_config=resource_config,
        successor_config_file_sha256=sha256_file(config_path),
        successor_config_semantic_sha256=config_sha256(resource_config),
        verify_live_receipt=True,
        verify_live_replacement_run_state=(replacement_failure_lineage is not None),
    )
    if canonical != authorization.as_dict():
        raise RuntimeError("resource technical successor builder produced noncanonical evidence")
    return authorization


def _require_resource_bounded_confirmatory_authorization(
    amendment_directory: str | Path,
    *,
    verify_live_primary: bool,
) -> dict[str, Any]:
    """Return canonical schema-v4 evidence with an explicit internal live boundary."""

    verification = verify_preregistration_amendment(amendment_directory)
    if not verification.valid:
        raise _AuthorityValidationError(
            "resource-bounded amendment failed chain/integrity verification: "
            + "; ".join(verification.errors)
        )
    directory = _require_real_directory(
        Path(amendment_directory).expanduser().resolve(),
        role="resource-bounded amendment",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if (
        evidence.get("schema_version") != 4
        or evidence.get("amendment_purpose") != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        or "primary_recovery_authorization" in evidence
        or "finalization_successor_authorization" in evidence
    ):
        raise _AuthorityValidationError(
            "amendment is not an exclusive schema-v4 resource-bounded authority"
        )
    parent = evidence.get("parent")
    authorization = evidence.get("resource_bounded_confirmatory_authorization")
    if not isinstance(parent, Mapping) or not isinstance(authorization, Mapping):
        raise _AuthorityValidationError(
            "resource-bounded amendment lacks its parent or authorization"
        )
    parent_directory = _absolute_record_path(
        parent.get("authority_directory"),
        role="resource-bounded parent authority",
    )
    parent_state = _authority_state(
        parent_directory,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    hashes = _snapshot_hashes(directory)
    confirmatory_hashes = hashes.get("confirmatory_config")
    execution_hashes = hashes.get("execution_source")
    if not isinstance(confirmatory_hashes, Mapping) or not isinstance(execution_hashes, Mapping):
        raise _AuthorityValidationError(
            "resource-bounded amendment lacks config/source snapshot hashes"
        )
    resource_source = _json_mapping(
        directory / _SOURCE_TREE_SNAPSHOT,
        role="resource-bounded execution source",
    )
    canonical = _canonical_resource_bounded_confirmatory_authorization(
        authorization,
        parent_state=parent_state,
        resource_source=resource_source,
        resource_source_manifest_sha256=_require_sha256(
            execution_hashes.get("manifest_sha256"),
            role="resource-bounded execution-source manifest",
        ),
        resource_config_file_sha256=_require_sha256(
            confirmatory_hashes.get("file_sha256"),
            role="resource-bounded confirmatory config file",
        ),
        resource_config_semantic_sha256=_require_sha256(
            confirmatory_hashes.get("semantic_sha256"),
            role="resource-bounded confirmatory config semantics",
        ),
        verify_live_primary=verify_live_primary,
    )
    if dict(authorization) != canonical:
        raise _AuthorityValidationError(
            "resource-bounded confirmatory authorization is not canonical"
        )
    return canonical


def require_resource_bounded_confirmatory_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return the canonical schema-v4 resource authority after live revalidation."""

    return _require_resource_bounded_confirmatory_authorization(
        amendment_directory,
        verify_live_primary=True,
    )


def _require_sealed_resource_bounded_confirmatory_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Authenticate sealed C; its dedicated gate independently verifies primary P."""

    return _require_resource_bounded_confirmatory_authorization(
        amendment_directory,
        verify_live_primary=False,
    )


def _require_historical_resource_bounded_confirmatory_storage_policy(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Read sealed schema-v4 C policy without treating C as the execution leaf.

    This boundary exists only for validating a direct technical successor D.  It
    authenticates C and its complete historical chain, but deliberately does not
    scan for a successor: D is already present while its own bundle is being
    verified.  Execution callers must continue to use
    ``require_confirmatory_storage_policy`` or the effective-authority gate.
    """

    directory = _require_real_directory(
        Path(amendment_directory).expanduser(),
        role="historical resource authority C",
    )
    evidence_before = _json_mapping(
        directory / _EVIDENCE_FILENAME,
        role="historical resource authority C evidence",
    )
    if (
        evidence_before.get("schema_version") != 4
        or evidence_before.get("amendment_purpose")
        != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        or "resource_bounded_confirmatory_authorization" not in evidence_before
    ):
        raise _AuthorityValidationError(
            "historical resource storage-policy readback requires schema-v4 authority C"
        )
    _require_sealed_resource_bounded_confirmatory_authorization(directory)
    evidence_after = _json_mapping(
        directory / _EVIDENCE_FILENAME,
        role="historical resource authority C evidence",
    )
    if evidence_after != evidence_before:
        raise _AuthorityValidationError(
            "historical resource authority C changed during storage-policy readback"
        )
    policy = evidence_after.get("confirmatory_storage_policy")
    if not isinstance(policy, Mapping):
        raise _AuthorityValidationError(
            "historical resource authority C lacks a confirmatory storage policy"
        )
    canonical = _canonical_confirmatory_storage_policy(policy)
    if dict(policy) != canonical:
        raise _AuthorityValidationError(
            "historical resource authority C storage policy is not canonical"
        )
    return canonical


def _resource_technical_successor_candidate_directories(
    resource_authority_directory: str | Path,
) -> tuple[Path, ...]:
    """List every schema-v5/successor-shaped peer under C's canonical root."""

    authority = _require_real_directory(
        Path(resource_authority_directory).expanduser(),
        role="resource authority C",
    )
    candidates: list[Path] = []
    try:
        peers = tuple(authority.parent.iterdir())
    except OSError as exc:
        raise _AuthorityValidationError(
            "resource amendment root is unavailable for successor uniqueness"
        ) from exc
    for peer in peers:
        if peer == authority:
            continue
        evidence_path = peer / _EVIDENCE_FILENAME
        marker_path = peer / _IMMUTABLE_MARKER
        if not os.path.lexists(evidence_path) and not os.path.lexists(marker_path):
            continue
        try:
            value = peer.stat(follow_symlinks=False)
        except OSError as exc:
            raise _AuthorityValidationError(
                "resource amendment peer cannot be inspected for successor uniqueness"
            ) from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or _is_link_or_reparse(peer)
            or not evidence_path.is_file()
            or _is_link_or_reparse(evidence_path)
        ):
            raise _AuthorityValidationError(
                "resource amendment root contains an ambiguous non-regular authority peer"
            )
        evidence = _json_mapping(
            evidence_path,
            role="resource amendment peer evidence",
        )
        if (
            evidence.get("schema_version") == 5
            or evidence.get("amendment_purpose") == RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE
            or "resource_bounded_technical_successor_authorization" in evidence
        ):
            candidates.append(peer.resolve())
    return tuple(sorted(candidates, key=lambda path: str(path).casefold()))


def _strict_schema_v3_resource_amendment_root_inventory(
    resource_parent: Path,
    successor: Path,
) -> tuple[Path, ...]:
    """Require exactly baseline amendment ancestors A/P/C plus one schema-v3 D."""

    amendment_root = _require_real_directory(
        resource_parent.parent,
        role="schema-v3 resource amendment root",
    )
    expected: set[Path] = {successor}
    state = _authority_state(
        resource_parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    while state.kind == "preregistration_amendment":
        if state.directory.parent != amendment_root:
            raise _AuthorityValidationError(
                "schema-v3 resource lineage leaves the canonical amendment root"
            )
        expected.add(state.directory)
        if state.parent_directory is None:
            raise _AuthorityValidationError(
                "schema-v3 resource amendment lineage ends without its base freeze"
            )
        state = _authority_state(
            state.parent_directory,
            visited=set(),
            depth=0,
            max_chain_depth=_MAX_CHAIN_DEPTH,
        )
    try:
        entries = tuple(amendment_root.iterdir())
    except OSError as exc:
        raise _AuthorityValidationError(
            "schema-v3 resource amendment root cannot be inventoried"
        ) from exc
    observed: list[Path] = []
    for entry in entries:
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise _AuthorityValidationError(
                "schema-v3 resource amendment peer cannot be inspected"
            ) from exc
        if not stat.S_ISDIR(entry_stat.st_mode) or _is_link_or_reparse(entry):
            raise _AuthorityValidationError(
                "schema-v3 resource amendment root contains a file, link, or reparse peer"
            )
        observed.append(entry.resolve())
    canonical_observed = tuple(sorted(observed, key=lambda path: str(path).casefold()))
    canonical_expected = tuple(sorted(expected, key=lambda path: str(path).casefold()))
    if canonical_observed != canonical_expected:
        raise _AuthorityValidationError(
            "schema-v3 resource amendment root must contain exactly baseline A/P/C "
            f"plus successor D; observed={[str(path) for path in canonical_observed]}"
        )
    return canonical_observed


def _require_resource_bounded_technical_successor_authorization(
    amendment_directory: str | Path,
    *,
    verify_live_primary: bool,
    verify_live_receipt: bool,
    enforce_unique_leaf: bool,
) -> dict[str, Any]:
    verification = verify_preregistration_amendment(amendment_directory)
    if not verification.valid:
        raise _AuthorityValidationError(
            "resource technical successor failed chain/integrity verification: "
            + "; ".join(verification.errors)
        )
    directory = _require_real_directory(
        Path(amendment_directory).expanduser(),
        role="resource technical successor D",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if (
        evidence.get("schema_version") != 5
        or evidence.get("amendment_purpose") != RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE
        or "resource_bounded_confirmatory_authorization" in evidence
        or "primary_recovery_authorization" in evidence
        or "finalization_successor_authorization" in evidence
    ):
        raise _AuthorityValidationError(
            "amendment is not an exclusive schema-v5 resource technical successor"
        )
    parent = evidence.get("parent")
    authorization = evidence.get("resource_bounded_technical_successor_authorization")
    if not isinstance(parent, Mapping) or not isinstance(authorization, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks its direct parent or authorization"
        )
    parent_directory = _absolute_record_path(
        parent.get("authority_directory"),
        role="resource technical successor parent authority C",
    )
    parent_state = _authority_state(
        parent_directory,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    parent_authorization = _require_resource_bounded_confirmatory_authorization(
        parent_directory,
        verify_live_primary=verify_live_primary,
    )
    hashes = _snapshot_hashes(directory)
    config_hashes = hashes.get("confirmatory_config")
    source_hashes = hashes.get("execution_source")
    if not isinstance(config_hashes, Mapping) or not isinstance(source_hashes, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks config/source snapshot hashes"
        )
    successor_source = _json_mapping(
        directory / _SOURCE_TREE_SNAPSHOT,
        role="resource technical successor execution source",
    )
    successor_config = load_config(directory / _CONFIRMATORY_CONFIG_SNAPSHOT)
    canonical = _canonical_resource_bounded_technical_successor_authorization(
        authorization,
        parent_state=parent_state,
        parent_resource_authorization=parent_authorization,
        successor_source=successor_source,
        successor_source_manifest_sha256=_require_sha256(
            source_hashes.get("manifest_sha256"),
            role="resource technical successor execution-source manifest",
        ),
        successor_config=successor_config,
        successor_config_file_sha256=_require_sha256(
            config_hashes.get("file_sha256"),
            role="resource technical successor confirmatory config file",
        ),
        successor_config_semantic_sha256=_require_sha256(
            config_hashes.get("semantic_sha256"),
            role="resource technical successor confirmatory config semantics",
        ),
        verify_live_receipt=verify_live_receipt,
    )
    if dict(authorization) != canonical:
        raise _AuthorityValidationError(
            "resource technical successor authorization is not canonical"
        )
    if enforce_unique_leaf:
        candidates = _resource_technical_successor_candidate_directories(parent_directory)
        if candidates != (directory,):
            raise _AuthorityValidationError(
                "resource authority C must have exactly one technical successor D; "
                f"candidates={[str(path) for path in candidates]}"
            )
    return canonical


def require_resource_bounded_technical_successor_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return the one canonical schema-v5 D after lineage/uniqueness verification."""

    return _require_resource_bounded_technical_successor_authorization(
        amendment_directory,
        verify_live_primary=True,
        verify_live_receipt=True,
        enforce_unique_leaf=True,
    )


def _require_sealed_effective_resource_bounded_confirmatory_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Authenticate the only effective resource leaf without rescanning primary P."""

    directory = _require_real_directory(
        Path(amendment_directory).expanduser(),
        role="effective resource execution authority",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if evidence.get("schema_version") == 4:
        canonical = _require_resource_bounded_confirmatory_authorization(
            directory,
            verify_live_primary=False,
        )
        candidates = _resource_technical_successor_candidate_directories(directory)
        if candidates:
            if len(candidates) == 1:
                raise _AuthorityValidationError(
                    "resource authority C is historically valid but no longer the "
                    f"effective execution leaf; use successor D {candidates[0]}"
                )
            raise _AuthorityValidationError(
                "resource authority C has a forbidden technical-successor fork: "
                f"{[str(path) for path in candidates]}"
            )
        return canonical
    if evidence.get("schema_version") == 5:
        return _require_resource_bounded_technical_successor_authorization(
            directory,
            verify_live_primary=False,
            verify_live_receipt=True,
            enforce_unique_leaf=True,
        )
    raise _AuthorityValidationError(
        "effective resource execution authority must be schema-v4 C or schema-v5 D"
    )


def require_effective_resource_bounded_confirmatory_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return only C before a successor exists, otherwise the unique direct child D."""

    directory = _require_real_directory(
        Path(amendment_directory).expanduser(),
        role="effective resource execution authority",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if evidence.get("schema_version") == 4:
        canonical = require_resource_bounded_confirmatory_authorization(directory)
        candidates = _resource_technical_successor_candidate_directories(directory)
        if candidates:
            if len(candidates) == 1:
                raise _AuthorityValidationError(
                    "resource authority C is historically valid but superseded by "
                    f"effective successor D {candidates[0]}"
                )
            raise _AuthorityValidationError(
                "resource authority C has a forbidden technical-successor fork"
            )
        return canonical
    if evidence.get("schema_version") == 5:
        return require_resource_bounded_technical_successor_authorization(directory)
    raise _AuthorityValidationError(
        "effective resource execution authority must be schema-v4 C or schema-v5 D"
    )


def _require_sealed_finalization_successor_authorization(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Recheck the immutable authority without rescanning predecessor payloads."""

    return _require_finalization_successor_authorization(
        amendment_directory,
        verify_live_predecessor=False,
    )


def require_authorized_prior_numeric_verification_proof(
    amendment_directory: str | Path,
    *,
    canonical_authorization: Mapping[str, Any] | None = None,
) -> AuthorizedPriorNumericVerificationProof:
    """Issue the private B-fast capability from one freshly verified schema-v3 authority."""

    from histo_audit.experiment.primary_statistics import (
        _issue_authorized_prior_numeric_verification_proof,
    )

    directory = _require_real_directory(
        Path(amendment_directory).expanduser().resolve(),
        role="B-fast amendment",
    )
    if canonical_authorization is None:
        canonical = require_finalization_successor_authorization(directory)
    else:
        sealed_canonical = _require_sealed_finalization_successor_authorization(directory)
        if dict(canonical_authorization) != sealed_canonical:
            raise _AuthorityValidationError(
                "provided finalization authorization differs from the sealed amendment"
            )
        canonical = sealed_canonical
    if type(canonical.get("schema_version")) is not int or canonical.get("schema_version") != 2:
        raise _AuthorityValidationError("B-fast capability requires schema-v2 authorization")
    predecessor = canonical.get("predecessor")
    numeric = canonical.get("numeric_verification")
    if not isinstance(predecessor, Mapping) or not isinstance(numeric, Mapping):
        raise _AuthorityValidationError("B-fast capability lacks canonical predecessor evidence")
    terminal = numeric.get("terminal_evidence")
    if not isinstance(terminal, Mapping):
        raise _AuthorityValidationError("B-fast capability lacks canonical terminal evidence")
    quartet = terminal.get("statistics_quartet")
    if not isinstance(quartet, list):
        raise _AuthorityValidationError("B-fast capability lacks its exact statistics quartet")
    return _issue_authorized_prior_numeric_verification_proof(
        amendment_directory=directory,
        authorization_sha256=_canonical_mapping_sha256(canonical),
        predecessor_run_id=str(predecessor["run_id"]),
        predecessor_artifact_root_sha256=str(predecessor["artifact_root_sha256"]),
        predecessor_artifact_manifest_sha256=str(predecessor["artifact_manifest_sha256"]),
        predecessor_source_tree_root_sha256=str(predecessor["execution_source_root_sha256"]),
        source_readback_root_sha256=str(terminal["statistics_source_readback_root_sha256"]),
        prior_numeric_verification_proof_sha256=str(
            numeric["prior_numeric_verification_proof_sha256"]
        ),
        trust_assumption=str(numeric["trust_assumption"]),
        limitation=str(numeric["limitation"]),
        statistics_quartet=quartet,
        comparison_count=int(terminal["statistics_comparison_count"]),
    )


def require_confirmatory_storage_policy(
    amendment_directory: str | Path,
) -> dict[str, Any]:
    """Return the canonical storage block after its lineage authorization verifies.

    Legacy finalization authorities retain their live failed-seal verification.
    Interrupted-primary recovery authorities instead authenticate the sealed recovery
    contract; their completed successor evidence is verified by the downstream gate.
    A confirmatory runner must still enforce checkpoint layout and post-seal readback.
    """

    verification = verify_preregistration_amendment(amendment_directory)
    if not verification.valid:
        raise _AuthorityValidationError(
            "confirmatory-storage amendment failed chain/integrity verification: "
            + "; ".join(verification.errors)
        )
    directory = _require_real_directory(
        Path(amendment_directory).expanduser().resolve(),
        role="confirmatory-storage amendment",
    )
    evidence = _json_mapping(directory / _EVIDENCE_FILENAME, role="amendment evidence")
    if "resource_bounded_technical_successor_authorization" in evidence:
        _require_resource_bounded_technical_successor_authorization(
            directory,
            verify_live_primary=False,
            verify_live_receipt=True,
            enforce_unique_leaf=True,
        )
    elif "resource_bounded_confirmatory_authorization" in evidence:
        _require_sealed_effective_resource_bounded_confirmatory_authorization(directory)
    elif "primary_recovery_authorization" in evidence:
        require_primary_recovery_authorization(directory)
    elif "finalization_successor_authorization" in evidence:
        require_finalization_successor_authorization(directory)
    else:
        raise _AuthorityValidationError(
            "amendment does not contain a confirmatory storage policy authorization"
        )
    policy = evidence.get("confirmatory_storage_policy")
    if not isinstance(policy, Mapping):
        raise _AuthorityValidationError("amendment does not contain a confirmatory storage policy")
    canonical = _canonical_confirmatory_storage_policy(policy)
    if dict(policy) != canonical:
        raise _AuthorityValidationError("confirmatory storage policy is not in canonical form")
    return canonical


def _strict_amendment_flat_inventory(
    directory: Path,
) -> tuple[list[dict[str, Any]], str]:
    entries = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    names = {entry.name for entry in entries}
    if names != _EXACT_AMENDMENT_BUNDLE_FILENAMES:
        raise _AuthorityValidationError(
            "resource technical successor must contain exactly the eight amendment "
            f"files; missing={sorted(_EXACT_AMENDMENT_BUNDLE_FILENAMES - names)}, "
            f"added={sorted(names - _EXACT_AMENDMENT_BUNDLE_FILENAMES)}"
        )
    records: list[dict[str, Any]] = []
    for path in entries:
        payload = _read_stable_single_link_file(
            path,
            role=f"resource technical successor inventory file {path.name}",
        )
        records.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if tuple(sorted(path.name for path in directory.iterdir())) != tuple(
        path.name for path in entries
    ):
        raise _AuthorityValidationError(
            "resource technical successor file set changed during anchored inventory"
        )
    inventory_sha256 = _canonical_mapping_sha256(
        {
            "schema_version": 1,
            "kind": "exact_flat_preregistration_amendment_inventory",
            "files": records,
        }
    )
    return records, inventory_sha256


def resource_bounded_technical_successor_intent_sha256(
    *,
    parent_authority_directory: str | Path,
    amendment_timestamp_utc: str,
    reason: str,
    affected_hypotheses: Sequence[str],
    affected_analyses: Sequence[str],
    outcomes_inspected_at_utc: str,
    authorization: Mapping[str, Any],
    confirmatory_storage_policy: Mapping[str, Any],
) -> str:
    """Hash the complete non-claiming schema-v5 publication intent.

    A replacement controller computes this before mutation. The fresh-process
    verifier recomputes it from the sealed successor, preventing a valid but
    differently described D from satisfying the publication attempt.
    """

    parent = _require_real_directory(
        Path(parent_authority_directory).expanduser(),
        role="resource technical successor intent parent C",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    parent_evidence = _json_mapping(
        parent / _EVIDENCE_FILENAME,
        role="resource technical successor intent parent evidence",
    )
    if (
        parent_evidence.get("schema_version") != 4
        or parent_evidence.get("amendment_purpose")
        != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
    ):
        raise _AuthorityValidationError(
            "resource technical successor intent parent must be schema-v4 authority C"
        )
    timestamp = _parse_timestamp(
        amendment_timestamp_utc,
        role="resource technical successor intent timestamp",
    )
    canonical_timestamp = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if amendment_timestamp_utc != canonical_timestamp or timestamp <= parent_state.timestamp:
        raise _AuthorityValidationError(
            "resource technical successor intent timestamp is not canonical or later than C"
        )
    inspected = _parse_timestamp(
        outcomes_inspected_at_utc,
        role="resource technical successor outcome-inspection timestamp",
    )
    canonical_inspected = inspected.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if outcomes_inspected_at_utc != canonical_inspected or inspected > timestamp:
        raise _AuthorityValidationError(
            "resource technical successor outcome-inspection timestamp is invalid"
        )
    canonical_reason = _normalise_text(
        reason,
        role="resource technical successor intent reason",
    )
    hypotheses = _normalise_items(
        affected_hypotheses,
        role="resource technical successor intent hypotheses",
    )
    analyses = _normalise_items(
        affected_analyses,
        role="resource technical successor intent analyses",
    )
    if not isinstance(authorization, Mapping):
        raise TypeError("resource technical successor intent authorization must be a mapping")
    canonical_authorization = dict(authorization)
    authorization_schema_version = canonical_authorization.get("schema_version")
    if (
        type(authorization_schema_version) is not int
        or authorization_schema_version not in {2, 3}
        or canonical_authorization.get("policy")
        != (
            _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY_V3
            if authorization_schema_version == 3
            else _RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_POLICY
        )
        or canonical_authorization.get("purpose") != RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE
        or not isinstance(
            canonical_authorization.get("prior_publication_failure"),
            Mapping,
        )
        or canonical_authorization.get("outcomes_inspected") is not True
        or canonical_authorization.get("analysis_disposition")
        != _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION
        or canonical_authorization.get("original_confirmatory_claim_allowed") is not False
        or canonical_authorization.get("study_outcome_eligible") is not False
        or canonical_authorization.get("completion_stage") is not None
        or canonical_authorization.get("automatic_retry_allowed") is not False
        or canonical_authorization.get("scientific_profile_change_allowed") is not False
    ):
        raise _AuthorityValidationError(
            "resource technical successor intent authorization is not non-claiming"
        )
    if authorization_schema_version == 3:
        lineage = canonical_authorization.get("replacement_publication_failure_lineage")
        if not isinstance(lineage, Mapping):
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor intent lacks its replacement "
                "publication failure lineage"
            )
        canonical_lineage = _canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=parent_state,
            verify_live_receipt=False,
        )
        if dict(lineage) != canonical_lineage:
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor intent lineage is not canonical"
            )
    elif "replacement_publication_failure_lineage" in canonical_authorization:
        raise _AuthorityValidationError(
            "schema-v2 resource technical successor intent must not carry replacement lineage"
        )
    prior_failure = canonical_authorization["prior_publication_failure"]
    failed_preflight = canonical_authorization.get("failed_preflight")
    prior_evidence = prior_failure.get("evidence")
    failed_evidence = (
        failed_preflight.get("evidence") if isinstance(failed_preflight, Mapping) else None
    )
    if not isinstance(prior_evidence, Mapping) or not isinstance(failed_evidence, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor intent lacks ordered failure evidence"
        )
    prior_failure_at = _parse_timestamp(
        prior_evidence.get("observed_at_utc"),
        role="resource technical successor intent prior-publication failure timestamp",
    )
    failed_preflight_at = _parse_timestamp(
        failed_evidence.get("observed_at_utc"),
        role="resource technical successor intent failed-preflight timestamp",
    )
    if (
        failed_preflight_at <= parent_state.timestamp
        or failed_preflight_at >= prior_failure_at
        or prior_failure_at > timestamp
    ):
        raise _AuthorityValidationError(
            "resource technical successor intent timestamps do not follow the exact "
            "C < failed-preflight < failed-publication receipt <= D order"
        )
    if authorization_schema_version == 3:
        lineage_value = canonical_authorization["replacement_publication_failure_lineage"]
        qualification_at = _parse_timestamp(
            lineage_value["terminal_qualification_receipt"]["qualified_at_utc"],
            role="resource technical successor replacement qualification timestamp",
        )
        if qualification_at <= prior_failure_at or qualification_at > timestamp:
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor timestamps do not follow the "
                "failed-publication receipt < replacement qualification <= D order"
            )
    policy = _canonical_confirmatory_storage_policy(confirmatory_storage_policy)
    parent_record = {
        "authority_directory": str(parent_state.directory),
        "authority_kind": parent_state.kind,
        "artifact_root_sha256": parent_state.artifact_root_sha256,
        "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
        "chain_depth": parent_state.chain_depth,
    }
    payload = {
        "schema_version": authorization_schema_version,
        "kind": "resource_bounded_technical_successor_publication_intent",
        "amendment_schema_version": 5,
        "amendment_purpose": RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE,
        "amendment_timestamp_utc": canonical_timestamp,
        "chain_depth": parent_state.chain_depth + 1,
        "parent": parent_record,
        "reason": canonical_reason,
        "affected_hypotheses": list(hypotheses),
        "affected_analyses": list(analyses),
        "outcomes_inspected": True,
        "outcomes_inspected_at_utc": canonical_inspected,
        "analysis_dispositions": _analysis_dispositions(
            analyses,
            outcomes_inspected=True,
        ),
        "authorization_sha256": _canonical_mapping_sha256(canonical_authorization),
        "confirmatory_storage_policy": policy,
    }
    return _canonical_mapping_sha256(payload)


def verify_resource_bounded_technical_successor(
    successor_directory: str | Path,
    *,
    expected_parent_authority_directory: str | Path,
    expected_artifact_root_sha256: str,
    expected_sha256_manifest_sha256: str,
    expected_authorization_sha256: str,
    expected_intent_sha256: str,
    expected_controller_process_id: int,
    verification_nonce: str,
) -> ResourceBoundedTechnicalSuccessorVerification:
    """Verify one exact schema-v5 D for a fresh-process pre-commit readback.

    This performs no publication or scientific execution. It executes the expensive
    live typed successor check exactly once, then uses sealed structural APIs for
    independent effective-leaf and historical-parent checks.
    """

    expected_root = _require_sha256(
        expected_artifact_root_sha256,
        role="expected resource technical successor artifact root",
    )
    expected_manifest = _require_sha256(
        expected_sha256_manifest_sha256,
        role="expected resource technical successor manifest",
    )
    expected_authorization = _require_sha256(
        expected_authorization_sha256,
        role="expected resource technical successor authorization",
    )
    expected_intent = _require_sha256(
        expected_intent_sha256,
        role="expected resource technical successor intent",
    )
    if type(expected_controller_process_id) is not int or expected_controller_process_id <= 0:
        raise _AuthorityValidationError(
            "expected resource technical successor controller PID must be a positive exact integer"
        )
    verifier_process_id = os.getpid()
    if expected_controller_process_id == verifier_process_id:
        raise _AuthorityValidationError(
            "resource technical successor verifier must run in a fresh process "
            "distinct from its publication controller"
        )
    verifier_parent_process_id = os.getppid()
    if expected_controller_process_id != verifier_parent_process_id:
        raise _AuthorityValidationError(
            "resource technical successor verifier must be a direct child of its "
            "publication controller"
        )
    canonical_nonce = _require_sha256(
        verification_nonce,
        role="resource technical successor verification nonce",
    )
    successor = _require_real_directory(
        Path(successor_directory).expanduser(),
        role="resource technical successor D",
    )
    expected_parent = _require_real_directory(
        Path(expected_parent_authority_directory).expanduser(),
        role="expected resource authority C",
    )
    parent_directory_sha256_before = sha256_path(expected_parent)
    inventory_before, inventory_sha256_before = _strict_amendment_flat_inventory(successor)
    candidates_before = _resource_technical_successor_candidate_directories(expected_parent)
    if candidates_before != (successor,):
        raise _AuthorityValidationError(
            "resource authority C does not have the exact singleton successor D"
        )

    generic = verify_preregistration_amendment(successor)
    if (
        not generic.valid
        or generic.parent_authority_directory != expected_parent
        or generic.chain_depth is None
        or generic.artifact_root_sha256 != expected_root
        or generic.sha256_manifest_sha256 != expected_manifest
    ):
        raise _AuthorityValidationError(
            "resource technical successor differs from its exact generic publication pins"
        )
    evidence = _json_mapping(
        successor / _EVIDENCE_FILENAME,
        role="resource technical successor evidence",
    )
    authorization = evidence.get("resource_bounded_technical_successor_authorization")
    if not isinstance(authorization, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks its authorization mapping"
        )
    schema_v3_root_inventory_before: tuple[Path, ...] | None = None
    if authorization.get("schema_version") == 3:
        schema_v3_root_inventory_before = _strict_schema_v3_resource_amendment_root_inventory(
            expected_parent,
            successor,
        )
    authorization_sha256 = _canonical_mapping_sha256(authorization)
    if authorization_sha256 != expected_authorization:
        raise _AuthorityValidationError(
            "resource technical successor authorization differs from its external pin"
        )
    intent_sha256 = resource_bounded_technical_successor_intent_sha256(
        parent_authority_directory=expected_parent,
        amendment_timestamp_utc=str(evidence.get("amendment_timestamp_utc", "")),
        reason=str(evidence.get("reason", "")),
        affected_hypotheses=evidence.get("affected_hypotheses", ()),
        affected_analyses=evidence.get("affected_analyses", ()),
        outcomes_inspected_at_utc=str(evidence.get("outcomes_inspected_at_utc", "")),
        authorization=authorization,
        confirmatory_storage_policy=evidence.get("confirmatory_storage_policy", {}),
    )
    if intent_sha256 != expected_intent:
        raise _AuthorityValidationError(
            "resource technical successor intent differs from its external pin"
        )

    typed = require_resource_bounded_technical_successor_authorization(successor)
    effective = _require_sealed_effective_resource_bounded_confirmatory_authorization(successor)
    if typed != effective or _canonical_mapping_sha256(typed) != authorization_sha256:
        raise _AuthorityValidationError(
            "resource technical successor typed and effective readbacks differ"
        )
    authorization_schema_version = typed.get("schema_version")
    if type(authorization_schema_version) is not int or authorization_schema_version not in {
        2,
        3,
    }:
        raise _AuthorityValidationError(
            "resource technical successor typed readback has an unsupported schema"
        )
    if authorization_schema_version == 3:
        lineage = typed.get("replacement_publication_failure_lineage")
        if not isinstance(lineage, Mapping):
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor lacks replacement failure lineage"
            )
        canonical_lineage = _canonical_replacement_publication_failure_lineage(
            lineage,
            resource_parent=_authority_state(
                expected_parent,
                visited=set(),
                depth=0,
                max_chain_depth=_MAX_CHAIN_DEPTH,
            ),
            verify_live_receipt=True,
            verify_live_run_state=True,
        )
        if dict(lineage) != canonical_lineage:
            raise _AuthorityValidationError(
                "schema-v3 resource technical successor replacement failure lineage "
                "changed during fresh verification"
            )
    fixed_intent = {
        "analysis_disposition": _PRIMARY_RECOVERY_ANALYSIS_DISPOSITION,
        "original_confirmatory_claim_allowed": False,
        "study_outcome_eligible": False,
        "completion_stage": None,
        "primary_rebinding_allowed": False,
        "primary_mutation_allowed": False,
        "automatic_retry_allowed": False,
        "scientific_profile_change_allowed": False,
    }
    if any(typed.get(key) != value for key, value in fixed_intent.items()):
        raise _AuthorityValidationError(
            "resource technical successor changed its non-claiming execution policy"
        )

    parent_generic = verify_preregistration_amendment(expected_parent)
    if (
        not parent_generic.valid
        or parent_generic.artifact_root_sha256 is None
        or parent_generic.sha256_manifest_sha256 is None
    ):
        raise _AuthorityValidationError(
            "historical resource authority C failed generic verification"
        )
    historical_parent = _require_resource_bounded_confirmatory_authorization(
        expected_parent,
        verify_live_primary=False,
    )
    supersedes = typed.get("supersedes")
    if not isinstance(supersedes, Mapping):
        raise _AuthorityValidationError(
            "resource technical successor lacks its superseded-C binding"
        )
    if (
        supersedes.get("authority_directory") != str(expected_parent)
        or supersedes.get("artifact_root_sha256") != parent_generic.artifact_root_sha256
        or supersedes.get("sha256_manifest_sha256") != parent_generic.sha256_manifest_sha256
        or supersedes.get("authorization_sha256") != _canonical_mapping_sha256(historical_parent)
        or supersedes.get("effective_execution_leaf") is not False
        or supersedes.get("historical_verification_retained") is not True
    ):
        raise _AuthorityValidationError(
            "resource technical successor does not retain exact historical C"
        )
    candidates = _resource_technical_successor_candidate_directories(expected_parent)
    if candidates != (successor,):
        raise _AuthorityValidationError(
            "resource technical successor uniqueness changed during verification"
        )
    historical_policy = _require_historical_resource_bounded_confirmatory_storage_policy(
        expected_parent
    )
    successor_policy = require_confirmatory_storage_policy(successor)
    if successor_policy != historical_policy:
        raise _AuthorityValidationError(
            "resource technical successor changed C's confirmatory storage policy"
        )
    expected_superseded_error = (
        "resource authority C is historically valid but no longer the effective "
        f"execution leaf; use successor D {successor}"
    )
    try:
        require_confirmatory_storage_policy(expected_parent)
    except _AuthorityValidationError as exc:
        if str(exc) != expected_superseded_error:
            raise _AuthorityValidationError(
                "resource authority C failed for a reason other than exact supersession"
            ) from exc
    else:
        raise _AuthorityValidationError(
            "resource authority C remained an effective execution leaf after D"
        )
    manifest = _json_mapping(
        successor / _MANIFEST_FILENAME,
        role="resource technical successor manifest",
    )
    if manifest.get("artifact_count") != 6:
        raise _AuthorityValidationError(
            "resource technical successor manifest must authenticate exactly six content artifacts"
        )

    inventory_after, inventory_sha256_after = _strict_amendment_flat_inventory(successor)
    parent_generic_after = verify_preregistration_amendment(expected_parent)
    successor_generic_after = verify_preregistration_amendment(successor)
    candidates_after = _resource_technical_successor_candidate_directories(expected_parent)
    schema_v3_root_inventory_after = (
        _strict_schema_v3_resource_amendment_root_inventory(
            expected_parent,
            successor,
        )
        if schema_v3_root_inventory_before is not None
        else None
    )
    if (
        inventory_after != inventory_before
        or inventory_sha256_after != inventory_sha256_before
        or parent_directory_sha256_before != sha256_path(expected_parent)
        or parent_generic_after != parent_generic
        or successor_generic_after != generic
        or candidates_after != (successor,)
        or schema_v3_root_inventory_after != schema_v3_root_inventory_before
    ):
        raise _AuthorityValidationError(
            "resource technical successor or historical C changed during verification"
        )
    return ResourceBoundedTechnicalSuccessorVerification(
        successor_directory=successor,
        parent_authority_directory=expected_parent,
        chain_depth=generic.chain_depth,
        artifact_root_sha256=expected_root,
        sha256_manifest_sha256=expected_manifest,
        authorization_sha256=authorization_sha256,
        intent_sha256=intent_sha256,
        flat_file_inventory_sha256=inventory_sha256_before,
        confirmatory_storage_policy_sha256=_canonical_mapping_sha256(successor_policy),
        flat_file_count=len(inventory_before),
        manifest_artifact_count=6,
        controller_process_id=expected_controller_process_id,
        verifier_process_id=verifier_process_id,
        verifier_parent_process_id=verifier_parent_process_id,
        verification_nonce=canonical_nonce,
    )


def build_finalization_successor_authorization(
    predecessor_run_directory: str | Path,
    parent_authority_directory: str | Path,
    *,
    numeric_verification_mode: str | None = None,
) -> FinalizationSuccessorAuthorization:
    """Derive an exact authorization from a registry-backed failed run seal.

    The builder verifies complete technical run integrity plus the exact append-only
    integrity-registry record, then reads the source/gate bindings and frozen matrix
    plan. It does not interpret outcome values. Typed artifact/cell verification
    remains mandatory in the finalization-only successor runner.
    """

    run = _require_real_directory(
        Path(predecessor_run_directory).expanduser().resolve(),
        role="finalization predecessor run",
    )
    parent = _require_real_directory(
        Path(parent_authority_directory).expanduser().resolve(),
        role="predecessor registration authority",
    )
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )

    manifest_path = run / RUN_ARTIFACT_MANIFEST_FILENAME
    marker_path = run / RUN_IMMUTABLE_MARKER
    _require_regular_file(manifest_path, role="failed predecessor artifact manifest")
    _require_regular_file(marker_path, role="failed predecessor immutable marker")
    manifest = _json_mapping(manifest_path, role="failed predecessor artifact manifest")
    marker = _json_mapping(marker_path, role="failed predecessor immutable marker")
    records = _normalise_manifest_records(manifest.get("artifacts"), flat=False)
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or run.name != run_id
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "failed"
        or manifest.get("artifact_count") != len(records)
        or manifest.get("artifact_root_sha256") != _canonical_root(records)
        or marker.get("run_id") != run_id
        or marker.get("status") != "failed"
        or marker.get("artifact_root_sha256") != manifest.get("artifact_root_sha256")
        or marker.get("artifact_manifest_sha256") != sha256_file(manifest_path)
        or not isinstance(marker.get("run_path"), str)
        or Path(str(marker["run_path"])).resolve(strict=False) != run
    ):
        raise _AuthorityValidationError(
            "authorization builder requires one exact terminally failed sealed primary run"
        )

    records_by_path = {record["path"]: record for record in records}
    structural_paths = {
        RUN_SOURCE_TREE_MANIFEST_FILENAME: run / RUN_SOURCE_TREE_MANIFEST_FILENAME,
        "primary_execution_gate.json": run / "primary_execution_gate.json",
        "matrix_plan.json": run / "matrix_plan.json",
    }
    for relative_path, path in structural_paths.items():
        record = records_by_path.get(relative_path)
        if (
            record is None
            or not path.is_file()
            or _is_link_or_reparse(path)
            or record["size_bytes"] != path.stat().st_size
            or record["sha256"] != sha256_file(path)
        ):
            raise _AuthorityValidationError(
                f"authorization builder structural file differs from the failed seal: "
                f"{relative_path}"
            )

    source_path = structural_paths[RUN_SOURCE_TREE_MANIFEST_FILENAME]
    source_hashes = _execution_hashes(source_path)
    parent_source = parent_state.snapshot_hashes.get("execution_source")
    if not isinstance(parent_source, Mapping) or source_hashes["root_sha256"] != parent_source.get(
        "root_sha256"
    ):
        raise _AuthorityValidationError(
            "failed predecessor computation source differs from the parent authority"
        )

    plan = _json_mapping(
        structural_paths["matrix_plan.json"], role="failed predecessor matrix plan"
    )
    expected_plan_fields = {
        "schema_version",
        "config_sha256",
        "scenario_count",
        "cell_count",
        "required_cell_count",
        "optional_cell_count",
        "scenarios",
        "cells",
    }
    cells = plan.get("cells")
    scenarios = plan.get("scenarios")
    if (
        set(plan) != expected_plan_fields
        or not isinstance(cells, list)
        or not isinstance(scenarios, list)
        or plan.get("cell_count") != len(cells)
        or plan.get("scenario_count") != len(scenarios)
    ):
        raise _AuthorityValidationError("failed predecessor matrix plan has an invalid schema")
    required_flags: list[bool] = []
    cell_ids: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or not isinstance(cell.get("required"), bool):
            raise _AuthorityValidationError(
                f"failed predecessor matrix-plan cell {index} has an invalid required flag"
            )
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or Path(cell_id).name != cell_id:
            raise _AuthorityValidationError(
                f"failed predecessor matrix-plan cell {index} has an unsafe cell ID"
            )
        required_flags.append(bool(cell["required"]))
        cell_ids.append(cell_id)
    if len(set(cell_ids)) != len(cell_ids):
        raise _AuthorityValidationError("failed predecessor matrix plan has duplicate cell IDs")
    required_count = sum(required_flags)
    optional_count = len(required_flags) - required_count
    if (
        plan.get("required_cell_count") != required_count
        or plan.get("optional_cell_count") != optional_count
    ):
        raise _AuthorityValidationError(
            "failed predecessor matrix-plan cell counts are internally inconsistent"
        )
    _strict_cell_count(required_count, role="reused required-cell count", positive=True)
    _strict_cell_count(optional_count, role="reused optional-cell count", positive=False)

    if numeric_verification_mode not in {
        None,
        INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
    }:
        raise _AuthorityValidationError("unsupported finalization numeric-verification mode")
    numeric_verification = (
        _build_inherited_prior_numeric_verification(run)
        if numeric_verification_mode == INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
        else None
    )
    if numeric_verification is not None and (required_count != 185 or optional_count != 37):
        raise _AuthorityValidationError(
            "inherited prior numeric verification requires the exact 185/185 required "
            "and 37 optional frozen cell plan"
        )
    authorization = FinalizationSuccessorAuthorization(
        predecessor_run_id=run_id,
        predecessor_run_directory=run,
        predecessor_artifact_root_sha256=_require_sha256(
            manifest.get("artifact_root_sha256"), role="predecessor artifact root"
        ),
        predecessor_artifact_manifest_sha256=sha256_file(manifest_path),
        predecessor_execution_source_root_sha256=source_hashes["root_sha256"],
        predecessor_execution_source_manifest_sha256=source_hashes["manifest_sha256"],
        predecessor_registration_authority_directory=parent_state.directory,
        predecessor_registration_authority_kind=parent_state.kind,
        predecessor_registration_authority_artifact_root_sha256=(parent_state.artifact_root_sha256),
        predecessor_registration_authority_manifest_sha256=(parent_state.sha256_manifest_sha256),
        reused_required_cell_count=required_count,
        reused_optional_cell_count=optional_count,
        numeric_verification=numeric_verification,
    )
    canonical = _canonical_finalization_successor_authorization(
        authorization,
        parent_state=parent_state,
        verify_live_predecessor=True,
    )
    if authorization.as_dict() != canonical:
        raise _AuthorityValidationError(
            "derived finalization-successor authorization is not canonical"
        )
    return authorization


def _source_file_hashes(paths: Sequence[Path]) -> dict[Path, str]:
    return {path.resolve(): sha256_file(path) for path in paths}


def _assert_source_files_unchanged(expected: Mapping[Path, str]) -> None:
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"amendment input changed during publication: {path}")


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _assert_safe_destination(project_root: Path, parent: Path, destination: Path) -> None:
    if _within(destination, parent):
        raise ValueError("amendment destination must not be inside its immutable parent")
    protected = (
        project_root / "src",
        project_root / "configs",
        project_root / "data" / "raw",
        project_root / "artifacts" / "runs",
        project_root / ".git",
    )
    for root in protected:
        resolved = root.resolve(strict=False)
        if _within(destination, resolved):
            raise ValueError(f"amendment destination is inside protected path: {resolved}")


def _amendment_markdown(
    *,
    timestamp: str,
    parent: _AuthorityState,
    reason: str,
    hypotheses: Sequence[str],
    analyses: Sequence[str],
    outcomes_inspected: bool,
    outcomes_inspected_at: str | None,
    finalization_successor_authorization: Mapping[str, Any] | None,
    primary_recovery_authorization: Mapping[str, Any] | None,
    resource_bounded_confirmatory_authorization: Mapping[str, Any] | None,
    resource_bounded_technical_successor_authorization: Mapping[str, Any] | None,
    confirmatory_storage_policy: Mapping[str, Any] | None,
) -> str:
    hypothesis_lines = "\n".join(f"- {item}" for item in hypotheses)
    analysis_lines = "\n".join(f"- {item}" for item in analyses)
    reporting = (
        "Every affected analysis is amended or exploratory and can never be reported as the "
        "original unamended primary analysis."
        if outcomes_inspected
        else "Affected analyses may be reported only as amended-before-outcome-inspection."
    )
    finalization_section = ""
    if finalization_successor_authorization is not None:
        predecessor = finalization_successor_authorization["predecessor"]
        reuse = finalization_successor_authorization["reuse"]
        authority = predecessor["registration_authority"]
        numeric = finalization_successor_authorization.get("numeric_verification")
        numeric_lines = (
            "\n- Numeric verification mode: "
            f"`{numeric['mode']}`"
            "\n- Prior numeric verification proof SHA-256: "
            f"`{numeric['prior_numeric_verification_proof_sha256']}`"
            "\n- Trust assumption: "
            f"`{numeric['trust_assumption']}`"
            "\n- Limitation: "
            f"`{numeric['limitation']}`"
            "\n- Process observation receipt: exact bytes embedded in the sealed "
            "numeric-verification authority"
            "\n- Numeric policy: inherited sealed prior verification; no automatic "
            "fallback or mode switch"
            if isinstance(numeric, Mapping)
            else "\n- Numeric verification mode: `legacy_full_recomputation_or_reverification`"
        )
        finalization_section = f"""

## Finalization-only successor authorization

- Policy: `{finalization_successor_authorization["policy"]}`
- Predecessor run ID: `{predecessor["run_id"]}`
- Predecessor run directory: `{predecessor["run_directory"]}`
- Required terminal status: `{predecessor["terminal_status"]}`
- Predecessor artifact root SHA-256: `{predecessor["artifact_root_sha256"]}`
- Predecessor artifact-manifest SHA-256: `{predecessor["artifact_manifest_sha256"]}`
- Predecessor execution-source root SHA-256: `{predecessor["execution_source_root_sha256"]}`
- Predecessor execution-source manifest SHA-256: `{predecessor["execution_source_manifest_sha256"]}`
- Predecessor registration authority: `{authority["directory"]}`
- Predecessor registration-authority kind: `{authority["kind"]}`
- Predecessor registration-authority root SHA-256: `{authority["artifact_root_sha256"]}`
- Predecessor registration-authority manifest SHA-256: `{authority["sha256_manifest_sha256"]}`
- Reused required cells: {reuse["reused_required_cell_count"]}
- Reused optional cells: {reuse["reused_optional_cell_count"]}
- Retrained cells: {reuse["retrained_cell_count"]}
- Outcomes inspected before amendment: {str(finalization_successor_authorization["outcomes_inspected"]).lower()}
- Required successor evidence file: `{finalization_successor_authorization["successor_evidence_filename"]}`
- Required completion flag: `{_FINALIZATION_SUCCESSOR_COMPLETION_FLAG}=true`
- Predecessor access: `{reuse["predecessor_access_policy"]}`
- Cell selection: `{reuse["selection_policy"]}`
- Successor output: `{finalization_successor_authorization["successor_output_policy"]}`
{numeric_lines}

This authorization does not make the failed predecessor eligible. The inherited
control-flow proof is not a fresh semantic recomputation and is valid only under the
explicit trust assumption printed above. The successor must
verify the complete predecessor seal and cell artifacts read-only, use a new run
directory with the exact predecessor as `retry_of_run_id`, retrain zero cells, and
become eligible only after its own independent post-seal verification. The inherited
mode is automatic, outcome-blind verification only and may not read results for
selection or tuning.
"""
    recovery_section = ""
    if primary_recovery_authorization is not None:
        interruption = primary_recovery_authorization["interruption_evidence"]
        recovery_section = f"""

## Interrupted-primary recovery authorization

- Policy: `{primary_recovery_authorization["policy"]}`
- Source run ID: `{primary_recovery_authorization["source_run_id"]}`
- Source run directory: `{primary_recovery_authorization["source_run_directory"]}`
- Interruption kind: `{interruption["kind"]}`
- Host boot timestamp (UTC): `{interruption["last_boot_at_utc"]}`
- Process absence checked at (UTC): `{interruption["process_checked_at_utc"]}`
- Source process ID: {interruption["source_process_id"]}
- Source process active: {str(interruption["process_active"]).lower()}
- Interruption receipt: `{interruption["receipt_path"]}`
- Interruption receipt SHA-256: `{interruption["receipt_sha256"]}`
- Outcomes inspected: {str(primary_recovery_authorization["outcomes_inspected"]).lower()}
- Outcome inspection timestamp (UTC): `{primary_recovery_authorization["outcome_inspection_at_utc"]}`
- Analysis disposition: `{primary_recovery_authorization["analysis_disposition"]}`
- Scientific method changes: none
- Expected source snapshot root SHA-256: `{primary_recovery_authorization["expected_source_snapshot_root_sha256"]}`
- Expected filesystem readback root SHA-256: `{primary_recovery_authorization["expected_source_filesystem_readback_root_sha256"]}`
- Expected restoration readback root SHA-256: `{primary_recovery_authorization["expected_restoration_readback_root_sha256"]}`
- Expected statistics manifest SHA-256: `{primary_recovery_authorization["expected_statistics_manifest_sha256"]}`
- Trust assumption: `{primary_recovery_authorization["trust_assumption"]}`
- Limitation: `{primary_recovery_authorization["limitation"]}`

The source remains an unsealed, ineligible, read-only orphan and may not be repaired,
overwritten, or retroactively sealed. Recovery must verify the interruption receipt and
every expected digest, physically copy only the exact authorized artifact set into one
new run with `retry_of_run_id`, retrain zero cells, and perform no training fallback,
automatic retry, selection, tuning, or scientific-method change. Any recovered analysis
is permanently `amended_or_exploratory`, never the original unamended primary analysis.
"""
    resource_section = ""
    if resource_bounded_confirmatory_authorization is not None:
        historical = resource_bounded_confirmatory_authorization["historical_primary"]
        authority = historical["registration_authority"]
        profile = resource_bounded_confirmatory_authorization["resource_profile"]
        source_delta = resource_bounded_confirmatory_authorization["execution_source_delta"]
        capacity = resource_bounded_confirmatory_authorization["resource_capacity_policy"]
        changed_paths = "\n".join(
            f"- `{record['path']}` ({record['change_kind']})"
            for record in source_delta["allowlisted_changes"]
        )
        resource_section = f"""

## Resource-bounded confirmatory execution authority

- Purpose: `{RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE}`
- Policy: `{resource_bounded_confirmatory_authorization["policy"]}`
- Historical primary run ID: `{historical["run_id"]}`
- Historical primary run directory: `{historical["run_directory"]}`
- Historical primary artifact root SHA-256: `{historical["artifact_root_sha256"]}`
- Historical primary artifact-manifest SHA-256: `{historical["artifact_manifest_sha256"]}`
- Historical primary completion evidence SHA-256: `{historical["completion_evidence_sha256"]}`
- Historical primary execution-gate SHA-256: `{historical["primary_execution_gate_sha256"]}`
- Historical primary stage-attestation record SHA-256: `{historical["stage_attestation_record_sha256"]}`
- Historical primary stage-attestation verification SHA-256: `{historical["stage_attestation_verification_sha256"]}`
- Historical primary recovery evidence SHA-256: `{historical["recovery_evidence_sha256"]}`
- Historical primary recovery authorization SHA-256: `{historical["recovery_authorization_sha256"]}`
- Historical registration authority: `{authority["directory"]}`
- Historical registration-authority root SHA-256: `{authority["artifact_root_sha256"]}`
- Resource profile ID: `{profile["profile_id"]}`
- Resource confirmatory-config file SHA-256: `{profile["resource_confirmatory_config_file_sha256"]}`
- Resource confirmatory-config semantic SHA-256: `{profile["resource_confirmatory_config_semantic_sha256"]}`
- Parent execution-source root SHA-256: `{source_delta["parent_root_sha256"]}`
- Resource execution-source root SHA-256: `{source_delta["resource_root_sha256"]}`
- Exact source-delta SHA-256: `{source_delta["delta_sha256"]}`
- Planned required cells: {capacity["planned_required_cells"]}
- Planned CNN cells: {capacity["planned_cnn_cells"]}
- Planned CNN fold checkpoints: {capacity["planned_cnn_fold_checkpoints"]}
- Maximum epochs: {capacity["max_epochs"]}
- Projected stable run bytes: {capacity["projected_stable_run_bytes"]}
- Fixed safety margin bytes: {capacity["fixed_safety_margin_bytes"]}
- Minimum free bytes before tracker creation: {capacity["minimum_free_bytes_before_tracker"]}
- Maximum active atomic temporary checkpoints: {capacity["max_active_atomic_temp_checkpoints"]}
- Minimum total host RAM bytes: {capacity["minimum_total_ram_bytes"]}
- Minimum available host RAM bytes before data loading: {capacity["minimum_available_ram_bytes_before_data"]}
- Minimum available host RAM bytes immediately before tracker creation: {capacity["minimum_available_ram_bytes_before_tracker"]}
- CUDA required: {str(capacity["cuda_required"]).lower()}
- Fixed CUDA device index: {capacity["cuda_device_index"]}
- Minimum total VRAM bytes: {capacity["minimum_total_vram_bytes"]}
- Minimum free VRAM bytes: {capacity["minimum_free_vram_bytes"]}
- cuDNN required: {str(capacity["cudnn_required"]).lower()}
- AMP required: {str(capacity["amp_required"]).lower()}
- AMP dtype: `{capacity["amp_dtype"]}`
- CUDA smoke input shape: `{capacity["cuda_smoke_input_shape"]}`
- CUDA smoke forward/backward required: {str(capacity["cuda_smoke_forward_backward_required"]).lower()}
- CUDA smoke finite forward/backward required: {str(capacity["cuda_smoke_finite_required"]).lower()}
- CUDA smoke maximum peak allocated bytes: {capacity["cuda_smoke_max_peak_allocated_bytes"]}
- Official weight identifier: `{capacity["official_weight_identifier"]}`
- Official weight SHA-256: `{capacity["official_weight_sha256"]}`
- Implicit weight download allowed: {str(capacity["implicit_weight_download_allowed"]).lower()}
- Outcomes inspected: true
- Analysis disposition: `amended_or_exploratory`
- Original confirmatory claim allowed: false
- Study-outcome eligible: false
- Completion stage: null
- Primary rebinding allowed: false
- Primary mutation allowed: false

Exact allowlisted execution-source changes:

{changed_paths}

This child authority does not alter or replace the historical primary authority. The
historical primary is verified under its direct recovery amendment, while the current
resource profile and live execution source are verified independently under this
child. The child is permanently post-outcome and `amended_or_exploratory`; it may not
support the original confirmatory claim or any completion-stage transition.
"""
    technical_successor_section = ""
    if resource_bounded_technical_successor_authorization is not None:
        successor = resource_bounded_technical_successor_authorization
        supersedes = successor["supersedes"]
        prior_publication_failure = successor["prior_publication_failure"]
        failed = successor["failed_preflight"]
        source_delta = successor["execution_source_delta"]
        correction = successor["cnn_provenance_correction"]
        replacement_failure_lineage = successor.get("replacement_publication_failure_lineage")
        replacement_failure_lines = ""
        if isinstance(replacement_failure_lineage, Mapping):
            replacement_receipt = replacement_failure_lineage["terminal_qualification_receipt"]
            replacement_failure_lines = (
                "\n- Replacement-v1 terminal qualification receipt: "
                f"`{replacement_failure_lineage['terminal_qualification_receipt_path']}`"
                "\n- Replacement-v1 terminal qualification receipt SHA-256: "
                f"`{replacement_failure_lineage['terminal_qualification_receipt_sha256']}`"
                "\n- Replacement-v1 terminal disposition: "
                f"`{replacement_receipt['status']}`"
                "\n- Replacement-v1 retry allowed: false"
            )
        changed_paths = "\n".join(
            f"- `{record['path']}` ({record['change_kind']})"
            for record in source_delta["allowlisted_changes"]
        )
        technical_successor_section = f"""

## Resource-bounded technical successor D

- Purpose: `{RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE}`
- Policy: `{successor["policy"]}`
- Superseded authority C: `{supersedes["authority_directory"]}`
- Superseded authority root SHA-256: `{supersedes["artifact_root_sha256"]}`
- Superseded authorization SHA-256: `{supersedes["authorization_sha256"]}`
- Prior failed publication receipt: `{prior_publication_failure["receipt_path"]}`
- Prior failed publication receipt SHA-256: `{prior_publication_failure["receipt_sha256"]}`
- Prior failed publication disposition: `manual_new_one_shot_after_rolled_back_publication`{replacement_failure_lines}
- Failed preflight receipt: `{failed["receipt_path"]}`
- Failed preflight receipt SHA-256: `{failed["receipt_sha256"]}`
- CNN scenario: `{correction["scenario_id"]}`
- CNN cache provenance ID: `{correction["cache_provenance_id"]}`
- Authority-C CNN provenance SHA-256: `{correction["before_config_record_sha256"]}`
- Successor-D CNN provenance SHA-256: `{correction["after_config_record_sha256"]}`
- Before recomputed provenance SHA-256: `{correction["before"]["recomputed_record_sha256"]}`
- After recomputed provenance SHA-256: `{correction["after"]["recomputed_record_sha256"]}`
- Semantic-equivalence evidence SHA-256: `{correction["after"]["semantic_equivalence_evidence_sha256"]}`
- Technical source-delta SHA-256: `{source_delta["delta_sha256"]}`
- Expected successor config semantic SHA-256: `{successor["expected_successor_config_semantic_sha256"]}`
- Resource input-workspace plan SHA-256: `{successor["resource_input_workspace_plan"]["plan_without_self_hash_sha256"]}`
- Planned required/CNN/checkpoint counts: 24 / 6 / 30
- Outcomes inspected: true
- Analysis disposition: `amended_or_exploratory`
- Original confirmatory claim allowed: false
- Study-outcome eligible: false
- Completion stage: null
- Automatic retry allowed: false

Exact allowlisted execution-source changes:

{changed_paths}

Authority C remains immutable and generically verifiable historical evidence, but is
not an effective execution leaf after this unique direct child D exists. D changes no
cache bytes, model, data, split, seed, metric, restoration rule, estimand, completion
stage, or scientific disposition. Its config change is limited to the two corrected
logical CNN-provenance digests; capacity-v3 adds only checksum-bound workspace
resource controls.
"""
    storage_section = ""
    if confirmatory_storage_policy is not None:
        storage_section = f"""

## Confirmatory checkpoint-storage policy

- Policy: `{confirmatory_storage_policy["policy"]}`
- Scope: `{confirmatory_storage_policy["scope"]}`
- Canonical relative path: `{confirmatory_storage_policy["canonical_relative_path_template"]}`
- Retained copy count: {confirmatory_storage_policy["retained_copy_count"]}
- Link policy: `{confirmatory_storage_policy["link_policy"]}`
- Verification policy: `{confirmatory_storage_policy["verification_policy"]}`
- Scientific effect: `{confirmatory_storage_policy["scientific_effect"]}`

This authority changes storage and reconciliation only. It does not change models,
data, splits, seeds, predictions, metrics, restoration, or estimands. A confirmatory
runner must enforce this policy independently before any eligibility claim.
"""
    return f"""# Preregistration amendment

- Timestamp (UTC): {timestamp}
- Parent authority: {parent.directory}
- Parent artifact root: {parent.artifact_root_sha256}
- Outcomes inspected: {str(outcomes_inspected).lower()}
- Outcomes inspected at (UTC): {outcomes_inspected_at or "not_applicable"}
- Reason: {reason}

## Affected hypotheses

{hypothesis_lines}

## Affected analyses

{analysis_lines}

## Reporting policy

{reporting}
{finalization_section}
{recovery_section}
{resource_section}
{technical_successor_section}
{storage_section}
"""


def _process_exclusive_amendment_creation[**P, R](
    function: Callable[P, R],
) -> Callable[P, R]:
    """Reject same-process concurrent or recursive authority creation.

    The filesystem publication lock below separately serialises every direct-child
    namespace across processes. This process guard closes callback bypasses through
    a fresh thread or ``contextvars.Context`` before any staging or mutation starts.
    """

    @wraps(function)
    def guarded(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if not _AMENDMENT_CREATION_PROCESS_GUARD.acquire(blocking=False):
            raise RuntimeError(
                "concurrent or recursive amendment creation is forbidden while "
                "another amendment transaction is active in this process"
            )
        try:
            return function(*args, **kwargs)
        finally:
            _AMENDMENT_CREATION_PROCESS_GUARD.release()

    return guarded


@_process_exclusive_amendment_creation
def create_preregistration_amendment(
    *,
    project_root: str | Path,
    parent_authority_directory: str | Path,
    amendment_root: str | Path,
    preregistration_path: str | Path,
    primary_config_path: str | Path,
    confirmatory_config_path: str | Path,
    reason: str,
    affected_hypotheses: Sequence[str],
    affected_analyses: Sequence[str],
    outcomes_inspected: bool,
    outcomes_inspected_at: datetime | None,
    finalization_successor_authorization: (
        FinalizationSuccessorAuthorization | Mapping[str, Any] | None
    ) = None,
    primary_recovery_authorization: Mapping[str, Any] | None = None,
    resource_bounded_confirmatory_authorization: (
        ResourceBoundedConfirmatoryAuthorization | Mapping[str, Any] | None
    ) = None,
    resource_bounded_technical_successor_authorization: (
        ResourceBoundedTechnicalSuccessorAuthorization | Mapping[str, Any] | None
    ) = None,
    confirmatory_storage_policy: (ConfirmatoryStoragePolicy | Mapping[str, Any] | None) = None,
    post_publication_check: (Callable[[PreregistrationAmendmentResult], None] | None) = None,
    timestamp: datetime | None = None,
) -> PreregistrationAmendmentResult:
    """Create one full immutable successor without mutating its parent authority.

    ``post_publication_check`` is an additive transaction-scoped verifier. It runs
    only after the published bundle has passed the ordinary generic and typed
    readbacks, while this function still owns every publication token and the bundle
    lock. Any exception therefore enters the same ownership-safe rollback path as an
    internal publication failure. The callback cannot replace or weaken the built-in
    verification.
    """

    if _POST_PUBLICATION_CHECK_ACTIVE.get():
        raise RuntimeError(
            "recursive amendment creation is forbidden during transaction-scoped "
            "post-publication verification"
        )
    if not isinstance(outcomes_inspected, bool):
        raise TypeError("outcomes_inspected must be an exact boolean")
    if post_publication_check is not None and not callable(post_publication_check):
        raise TypeError("post_publication_check must be callable or None")
    authority_count = sum(
        authorization is not None
        for authorization in (
            finalization_successor_authorization,
            primary_recovery_authorization,
            resource_bounded_confirmatory_authorization,
            resource_bounded_technical_successor_authorization,
        )
    )
    if authority_count > 1:
        raise ValueError(
            "finalization, primary recovery, resource-bounded, and resource technical "
            "successor authorizations "
            "are mutually exclusive"
        )
    if (
        confirmatory_storage_policy is not None
        and finalization_successor_authorization is None
        and primary_recovery_authorization is None
        and resource_bounded_confirmatory_authorization is None
        and resource_bounded_technical_successor_authorization is None
    ):
        raise ValueError(
            "confirmatory storage policy requires a finalization, primary recovery, "
            "resource-bounded, or resource technical successor authorization"
        )
    if (
        confirmatory_storage_policy is not None
        and finalization_successor_authorization is not None
        and outcomes_inspected is not False
    ):
        raise ValueError("confirmatory storage policy requires outcomes_inspected=false")
    if finalization_successor_authorization is not None and outcomes_inspected is not False:
        raise ValueError("finalization-only successor requires outcomes_inspected=false")
    if primary_recovery_authorization is not None and outcomes_inspected is not True:
        raise ValueError("primary recovery requires outcomes_inspected=true")
    if resource_bounded_confirmatory_authorization is not None and outcomes_inspected is not True:
        raise ValueError("resource-bounded confirmatory execution requires outcomes_inspected=true")
    if (
        resource_bounded_technical_successor_authorization is not None
        and outcomes_inspected is not True
    ):
        raise ValueError("resource-bounded technical successor requires outcomes_inspected=true")
    if (
        resource_bounded_confirmatory_authorization is not None
        and confirmatory_storage_policy is None
    ):
        raise ValueError(
            "resource-bounded confirmatory execution requires the inherited "
            "confirmatory storage policy"
        )
    if (
        resource_bounded_technical_successor_authorization is not None
        and confirmatory_storage_policy is None
    ):
        raise ValueError(
            "resource-bounded technical successor requires C's inherited "
            "confirmatory storage policy"
        )
    amendment_moment, amendment_timestamp, timestamp_component = _normalise_timestamp(timestamp)
    if outcomes_inspected:
        if outcomes_inspected_at is None:
            raise ValueError("outcomes_inspected_at is required when outcomes_inspected is true")
        inspected_moment, inspected_timestamp, _ = _normalise_timestamp(outcomes_inspected_at)
        if inspected_moment > amendment_moment:
            raise ValueError("outcomes_inspected_at cannot be later than the amendment timestamp")
    else:
        if outcomes_inspected_at is not None:
            raise ValueError(
                "outcomes_inspected_at must be omitted when outcomes_inspected is false"
            )
        inspected_timestamp = None
    canonical_reason = _normalise_text(reason, role="amendment reason")
    hypotheses = _normalise_items(affected_hypotheses, role="affected hypotheses")
    analyses = _normalise_items(affected_analyses, role="affected analyses")

    root = Path(project_root).expanduser().resolve()
    parent_supplied = Path(parent_authority_directory).expanduser()
    if not parent_supplied.is_absolute():
        parent_supplied = Path.cwd() / parent_supplied
    if parent_supplied.is_symlink():
        raise ValueError("parent authority path must not be a symlink")
    parent = _require_real_directory(parent_supplied.resolve(), role="parent authority")
    parent_state = _authority_state(
        parent,
        visited=set(),
        depth=0,
        max_chain_depth=_MAX_CHAIN_DEPTH,
    )
    if amendment_moment <= parent_state.timestamp:
        raise ValueError("amendment timestamp must be later than its parent authority")
    canonical_finalization_authorization = (
        _canonical_finalization_successor_authorization(
            finalization_successor_authorization,
            parent_state=parent_state,
            verify_live_predecessor=True,
        )
        if finalization_successor_authorization is not None
        else None
    )
    canonical_recovery_authorization = (
        _canonical_primary_recovery_authorization(
            primary_recovery_authorization,
            parent_state=parent_state,
            outcomes_inspected=outcomes_inspected,
            outcomes_inspected_at_utc=inspected_timestamp,
            amendment_reason=canonical_reason,
        )
        if primary_recovery_authorization is not None
        else None
    )
    if canonical_recovery_authorization is not None:
        observed_at = _parse_timestamp(
            canonical_recovery_authorization["interruption_evidence"]["observed_at_utc"],
            role="primary recovery interruption-observation timestamp",
        )
        if observed_at > amendment_moment:
            raise ValueError(
                "primary recovery interruption observation cannot be later than the amendment"
            )
    canonical_storage_policy = (
        _canonical_confirmatory_storage_policy(confirmatory_storage_policy)
        if confirmatory_storage_policy is not None
        else None
    )
    if resource_bounded_confirmatory_authorization is not None:
        _resource_parent_recovery_authorization(parent_state)
        inherited_storage_policy = require_confirmatory_storage_policy(parent_state.directory)
        if canonical_storage_policy != inherited_storage_policy:
            raise _AuthorityValidationError(
                "resource-bounded confirmatory execution must inherit the direct "
                "recovery parent's storage policy unchanged"
            )
    if resource_bounded_technical_successor_authorization is not None:
        parent_evidence = _json_mapping(
            parent_state.directory / _EVIDENCE_FILENAME,
            role="resource technical successor parent evidence",
        )
        if (
            parent_evidence.get("schema_version") != 4
            or parent_evidence.get("amendment_purpose")
            != RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE
        ):
            raise _AuthorityValidationError(
                "resource technical successor must be a direct child of schema-v4 C"
            )
        if _resource_technical_successor_candidate_directories(parent_state.directory):
            raise _AuthorityValidationError(
                "resource authority C already has a technical-successor candidate"
            )
        inherited_storage_policy = _require_historical_resource_bounded_confirmatory_storage_policy(
            parent_state.directory
        )
        if canonical_storage_policy != inherited_storage_policy:
            raise _AuthorityValidationError(
                "resource technical successor must inherit C's storage policy unchanged"
            )
    canonical_resource_authorization: dict[str, Any] | None = None
    canonical_resource_technical_successor: dict[str, Any] | None = None

    output_parent_supplied = Path(amendment_root).expanduser()
    if not output_parent_supplied.is_absolute():
        output_parent_supplied = root / output_parent_supplied
    unresolved_destination = output_parent_supplied.resolve(strict=False) / timestamp_component
    _assert_safe_destination(root, parent, unresolved_destination)
    output_parent_supplied.mkdir(parents=True, exist_ok=True)
    output_parent = _require_real_directory(output_parent_supplied, role="amendment output parent")
    destination = output_parent / timestamp_component
    _assert_safe_destination(root, parent, destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"amendment destination already exists: {destination}")

    preregistration = Path(preregistration_path).expanduser().resolve()
    primary_config = Path(primary_config_path).expanduser().resolve()
    confirmatory_config = Path(confirmatory_config_path).expanduser().resolve()
    preregistration_bytes = _require_regular_file(preregistration, role="amended preregistration")
    primary_bytes = _require_regular_file(primary_config, role="amended primary config")
    confirmatory_bytes = _require_regular_file(
        confirmatory_config, role="amended confirmatory config"
    )
    load_config(primary_config)
    load_config(confirmatory_config)
    source_paths = (preregistration, primary_config, confirmatory_config)
    source_hashes = _source_file_hashes(source_paths)
    execution_source = capture_source_tree(root)
    if (
        execution_source.get("schema_version") != 3
        or execution_source.get("scope_kind") != "execution_source"
    ):
        raise RuntimeError("live execution source capture is not schema-v3 execution-only")
    parent_directory_hash = sha256_path(parent)

    staging_parent = Path(tempfile.mkdtemp(prefix=f"histo-audit-amend-{timestamp_component}."))
    staging = staging_parent / "snapshot"
    staging.mkdir()
    publications: list[PublishedPath] = []
    published_result: PreregistrationAmendmentResult | None = None
    try:
        atomic_write_bytes(staging / _PREREGISTRATION_SNAPSHOT, preregistration_bytes)
        atomic_write_bytes(staging / _PRIMARY_CONFIG_SNAPSHOT, primary_bytes)
        atomic_write_bytes(staging / _CONFIRMATORY_CONFIG_SNAPSHOT, confirmatory_bytes)
        atomic_write_json(staging / _SOURCE_TREE_SNAPSHOT, execution_source)
        after_hashes = _snapshot_hashes(staging)
        before_hashes = parent_state.snapshot_hashes
        if resource_bounded_confirmatory_authorization is not None:
            for section in ("preregistration", "primary_config"):
                before_section = before_hashes.get(section)
                after_section = after_hashes.get(section)
                if (
                    not isinstance(before_section, Mapping)
                    or not isinstance(after_section, Mapping)
                    or dict(before_section) != dict(after_section)
                ):
                    raise _AuthorityValidationError(
                        f"resource-bounded amendment requires unchanged {section} identity"
                    )
            after_confirmatory = after_hashes.get("confirmatory_config")
            if not isinstance(after_confirmatory, Mapping):
                raise _AuthorityValidationError(
                    "resource-bounded amendment lacks confirmatory-config evidence"
                )
            canonical_resource_authorization = (
                _canonical_resource_bounded_confirmatory_authorization(
                    resource_bounded_confirmatory_authorization,
                    parent_state=parent_state,
                    resource_source=execution_source,
                    resource_source_manifest_sha256=_atomic_json_sha256(execution_source),
                    resource_config_file_sha256=_require_sha256(
                        after_confirmatory.get("file_sha256"),
                        role="resource-bounded confirmatory config file",
                    ),
                    resource_config_semantic_sha256=_require_sha256(
                        after_confirmatory.get("semantic_sha256"),
                        role="resource-bounded confirmatory config semantics",
                    ),
                    verify_live_primary=True,
                )
            )
        elif resource_bounded_technical_successor_authorization is not None:
            _require_resource_technical_parent_identity(before_hashes, after_hashes)
            parent_resource_authorization = _require_resource_bounded_confirmatory_authorization(
                parent_state.directory,
                verify_live_primary=True,
            )
            after_confirmatory = after_hashes.get("confirmatory_config")
            after_execution = after_hashes.get("execution_source")
            if not isinstance(after_confirmatory, Mapping) or not isinstance(
                after_execution, Mapping
            ):
                raise _AuthorityValidationError(
                    "resource technical successor lacks config/source snapshot hashes"
                )
            canonical_resource_technical_successor = (
                _canonical_resource_bounded_technical_successor_authorization(
                    resource_bounded_technical_successor_authorization,
                    parent_state=parent_state,
                    parent_resource_authorization=parent_resource_authorization,
                    successor_source=execution_source,
                    successor_source_manifest_sha256=_require_sha256(
                        after_execution.get("manifest_sha256"),
                        role="resource technical successor execution-source manifest",
                    ),
                    successor_config=load_config(confirmatory_config),
                    successor_config_file_sha256=_require_sha256(
                        after_confirmatory.get("file_sha256"),
                        role="resource technical successor confirmatory config file",
                    ),
                    successor_config_semantic_sha256=_require_sha256(
                        after_confirmatory.get("semantic_sha256"),
                        role="resource technical successor confirmatory config semantics",
                    ),
                    verify_live_receipt=True,
                    verify_live_replacement_run_state=True,
                )
            )
            if canonical_resource_technical_successor.get("schema_version") == 3:
                lineage = canonical_resource_technical_successor[
                    "replacement_publication_failure_lineage"
                ]
                qualification_at = _parse_timestamp(
                    lineage["terminal_qualification_receipt"]["qualified_at_utc"],
                    role="resource technical successor replacement qualification timestamp",
                )
                if qualification_at > amendment_moment:
                    raise _AuthorityValidationError(
                        "replacement terminal qualification cannot be later than "
                        "schema-v3 Authority D"
                    )
        elif canonical_storage_policy is not None or canonical_recovery_authorization is not None:
            _require_storage_only_scientific_identity(before_hashes, after_hashes)
        if _frozen_scientific_identity(before_hashes) == _frozen_scientific_identity(after_hashes):
            raise ValueError("amendment must contain at least one actual frozen change")
        chain_depth = parent_state.chain_depth + 1
        parent_record = {
            "authority_directory": str(parent_state.directory),
            "authority_kind": parent_state.kind,
            "artifact_root_sha256": parent_state.artifact_root_sha256,
            "sha256_manifest_sha256": parent_state.sha256_manifest_sha256,
            "chain_depth": parent_state.chain_depth,
        }
        atomic_write_text(
            staging / _REPORT_FILENAME,
            _amendment_markdown(
                timestamp=amendment_timestamp,
                parent=parent_state,
                reason=canonical_reason,
                hypotheses=hypotheses,
                analyses=analyses,
                outcomes_inspected=outcomes_inspected,
                outcomes_inspected_at=inspected_timestamp,
                finalization_successor_authorization=(canonical_finalization_authorization),
                primary_recovery_authorization=canonical_recovery_authorization,
                resource_bounded_confirmatory_authorization=(canonical_resource_authorization),
                resource_bounded_technical_successor_authorization=(
                    canonical_resource_technical_successor
                ),
                confirmatory_storage_policy=canonical_storage_policy,
            ),
        )
        evidence_schema_version = (
            4
            if canonical_resource_authorization is not None
            else (
                5
                if canonical_resource_technical_successor is not None
                else (
                    3
                    if canonical_recovery_authorization is not None
                    else (
                        int(canonical_finalization_authorization["schema_version"]) + 1
                        if canonical_finalization_authorization
                        else 1
                    )
                )
            )
        )
        atomic_write_json(
            staging / _EVIDENCE_FILENAME,
            {
                "schema_version": evidence_schema_version,
                "authority_kind": _AUTHORITY_KIND,
                "amendment_timestamp_utc": amendment_timestamp,
                "chain_depth": chain_depth,
                "parent": parent_record,
                "reason": canonical_reason,
                "affected_hypotheses": list(hypotheses),
                "affected_analyses": list(analyses),
                "outcomes_inspected": outcomes_inspected,
                "outcomes_inspected_at_utc": inspected_timestamp,
                "analysis_dispositions": _analysis_dispositions(
                    analyses, outcomes_inspected=outcomes_inspected
                ),
                "before": before_hashes,
                "after": after_hashes,
                "snapshots": {
                    "preregistration": _PREREGISTRATION_SNAPSHOT,
                    "primary_config": _PRIMARY_CONFIG_SNAPSHOT,
                    "confirmatory_config": _CONFIRMATORY_CONFIG_SNAPSHOT,
                    "execution_source": _SOURCE_TREE_SNAPSHOT,
                    "report": _REPORT_FILENAME,
                },
                "overwrite_policy": "immutable successor; never overwrite parent or peer",
                **(
                    {"finalization_successor_authorization": (canonical_finalization_authorization)}
                    if canonical_finalization_authorization is not None
                    else {}
                ),
                **(
                    {
                        "amendment_purpose": (RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE),
                        "resource_bounded_technical_successor_authorization": (
                            canonical_resource_technical_successor
                        ),
                    }
                    if canonical_resource_technical_successor is not None
                    else {}
                ),
                **(
                    {"primary_recovery_authorization": canonical_recovery_authorization}
                    if canonical_recovery_authorization is not None
                    else {}
                ),
                **(
                    {
                        "amendment_purpose": (RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE),
                        "resource_bounded_confirmatory_authorization": (
                            canonical_resource_authorization
                        ),
                    }
                    if canonical_resource_authorization is not None
                    else {}
                ),
                **(
                    {"confirmatory_storage_policy": canonical_storage_policy}
                    if canonical_storage_policy is not None
                    else {}
                ),
            },
        )
        records = _artifact_records(staging)
        artifact_root = _canonical_root(records)
        manifest_path = atomic_write_json(
            staging / _MANIFEST_FILENAME,
            {
                "schema_version": 1,
                "authority_kind": _AUTHORITY_KIND,
                "amendment_timestamp_utc": amendment_timestamp,
                "artifact_count": len(records),
                "artifact_root_sha256": artifact_root,
                "root_digest_algorithm": "sha256(canonical-json-artifact-records)",
                "excluded_paths": [_IMMUTABLE_MARKER, _MANIFEST_FILENAME],
                "artifacts": records,
            },
        )
        manifest_sha = sha256_file(manifest_path)
        atomic_write_json(
            staging / _IMMUTABLE_MARKER,
            {
                "schema_version": 1,
                "status": "amended",
                "authority_kind": _AUTHORITY_KIND,
                "amendment_timestamp_utc": amendment_timestamp,
                "artifact_root_sha256": artifact_root,
                "sha256_manifest_sha256": manifest_sha,
                "amendment_only": True,
            },
        )
        published_result = PreregistrationAmendmentResult(
            amendment_directory=destination,
            parent_authority_directory=parent,
            amendment_timestamp_utc=amendment_timestamp,
            chain_depth=chain_depth,
            amendment_evidence_path=destination / _EVIDENCE_FILENAME,
            amended_preregistration_path=destination / _PREREGISTRATION_SNAPSHOT,
            amended_primary_config_path=destination / _PRIMARY_CONFIG_SNAPSHOT,
            amended_confirmatory_config_path=destination / _CONFIRMATORY_CONFIG_SNAPSHOT,
            source_tree_manifest_path=destination / _SOURCE_TREE_SNAPSHOT,
            sha256_manifest_path=destination / _MANIFEST_FILENAME,
            immutable_marker_path=destination / _IMMUTABLE_MARKER,
            artifact_root_sha256=artifact_root,
            sha256_manifest_sha256=manifest_sha,
        )

        def verify_unchanged() -> None:
            _assert_source_files_unchanged(source_hashes)
            if capture_source_tree(root) != execution_source:
                raise RuntimeError("execution source changed during amendment publication")
            if sha256_path(parent) != parent_directory_hash:
                raise RuntimeError("parent authority bytes changed during amendment publication")
            observed_parent = _authority_state(
                parent,
                visited=set(),
                depth=0,
                max_chain_depth=_MAX_CHAIN_DEPTH,
            )
            if observed_parent != parent_state:
                raise RuntimeError("parent authority identity changed during amendment publication")
            if canonical_finalization_authorization is not None:
                observed_authorization = _canonical_finalization_successor_authorization(
                    canonical_finalization_authorization,
                    parent_state=observed_parent,
                    verify_live_predecessor=True,
                )
                if observed_authorization != canonical_finalization_authorization:
                    raise RuntimeError(
                        "failed predecessor authorization changed during amendment publication"
                    )
            if canonical_recovery_authorization is not None:
                observed_recovery = _canonical_primary_recovery_authorization(
                    canonical_recovery_authorization,
                    parent_state=observed_parent,
                    outcomes_inspected=outcomes_inspected,
                    outcomes_inspected_at_utc=inspected_timestamp,
                    amendment_reason=canonical_reason,
                )
                if observed_recovery != canonical_recovery_authorization:
                    raise RuntimeError(
                        "primary recovery authorization changed during amendment publication"
                    )
            if canonical_resource_authorization is not None:
                observed_resource = _canonical_resource_bounded_confirmatory_authorization(
                    canonical_resource_authorization,
                    parent_state=observed_parent,
                    resource_source=execution_source,
                    resource_source_manifest_sha256=_atomic_json_sha256(execution_source),
                    resource_config_file_sha256=sha256_file(confirmatory_config),
                    resource_config_semantic_sha256=config_sha256(load_config(confirmatory_config)),
                    verify_live_primary=True,
                )
                if observed_resource != canonical_resource_authorization:
                    raise RuntimeError(
                        "resource-bounded authorization changed during amendment publication"
                    )
            if canonical_resource_technical_successor is not None:
                observed_parent_resource = _require_resource_bounded_confirmatory_authorization(
                    observed_parent.directory,
                    verify_live_primary=True,
                )
                observed_successor = _canonical_resource_bounded_technical_successor_authorization(
                    canonical_resource_technical_successor,
                    parent_state=observed_parent,
                    parent_resource_authorization=observed_parent_resource,
                    successor_source=execution_source,
                    successor_source_manifest_sha256=_atomic_json_sha256(execution_source),
                    successor_config=load_config(confirmatory_config),
                    successor_config_file_sha256=sha256_file(confirmatory_config),
                    successor_config_semantic_sha256=config_sha256(
                        load_config(confirmatory_config)
                    ),
                    verify_live_receipt=True,
                    verify_live_replacement_run_state=True,
                )
                if observed_successor != canonical_resource_technical_successor:
                    raise RuntimeError(
                        "resource technical successor authorization changed during "
                        "amendment publication"
                    )
                candidates = _resource_technical_successor_candidate_directories(
                    observed_parent.directory
                )
                if candidates and candidates != (destination.resolve(),):
                    raise RuntimeError(
                        "resource authority C gained a competing technical successor"
                    )

        verify_unchanged()
        publication_body_completed = False
        try:
            with ExclusiveBundlePublicationLock(
                (parent, destination), role="preregistration amendment"
            ) as lock:
                try:
                    lock.assert_owned()
                    _assert_safe_destination(root, parent, destination)
                    if os.path.lexists(destination):
                        raise FileExistsError(
                            f"amendment destination already exists: {destination}"
                        )
                    verify_unchanged()
                    publications = publish_flat_directory_physical_copy_no_overwrite(
                        staging,
                        destination,
                        success_marker_name=_IMMUTABLE_MARKER,
                    )
                    lock.assert_owned()
                    verify_unchanged()
                    verification = verify_preregistration_amendment(destination)
                    if not verification.valid:
                        raise RuntimeError(
                            "published amendment failed independent verification: "
                            + "; ".join(verification.errors)
                        )
                    published_inventory, published_inventory_sha256 = (
                        _strict_amendment_flat_inventory(destination)
                    )
                    if canonical_resource_technical_successor is not None:
                        observed_successor = (
                            _require_resource_bounded_technical_successor_authorization(
                                destination,
                                verify_live_primary=False,
                                verify_live_receipt=True,
                                enforce_unique_leaf=True,
                            )
                        )
                        if observed_successor != canonical_resource_technical_successor:
                            raise RuntimeError(
                                "published resource technical successor failed exact readback"
                            )
                    if not all(publication.still_owned() for publication in publications):
                        raise RuntimeError(
                            "amendment publication ownership changed before "
                            "transaction-scoped verification"
                        )
                    if post_publication_check is not None:
                        callback_token = _POST_PUBLICATION_CHECK_ACTIVE.set(True)
                        try:
                            callback_result = post_publication_check(published_result)
                        finally:
                            _POST_PUBLICATION_CHECK_ACTIVE.reset(callback_token)
                        if callback_result is not None:
                            raise TypeError(
                                "post_publication_check must return None after "
                                "read-only verification"
                            )
                        lock.assert_owned()
                        verify_unchanged()
                        post_check_verification = verify_preregistration_amendment(destination)
                        if not post_check_verification.valid:
                            raise RuntimeError(
                                "amendment changed during transaction-scoped "
                                "post-publication verification: "
                                + "; ".join(post_check_verification.errors)
                            )
                        if canonical_resource_technical_successor is not None:
                            post_check_successor = (
                                _require_resource_bounded_technical_successor_authorization(
                                    destination,
                                    verify_live_primary=False,
                                    verify_live_receipt=True,
                                    enforce_unique_leaf=True,
                                )
                            )
                            if post_check_successor != canonical_resource_technical_successor:
                                raise RuntimeError(
                                    "resource technical successor changed during "
                                    "transaction-scoped post-publication verification"
                                )
                    lock.assert_owned()
                    final_inventory, final_inventory_sha256 = _strict_amendment_flat_inventory(
                        destination
                    )
                    if (
                        final_inventory != published_inventory
                        or final_inventory_sha256 != published_inventory_sha256
                    ):
                        raise RuntimeError(
                            "amendment flat-file inventory changed before transaction completion"
                        )
                    if canonical_resource_technical_successor is not None:
                        final_candidates = _resource_technical_successor_candidate_directories(
                            parent
                        )
                        if final_candidates != (destination.resolve(),):
                            raise RuntimeError(
                                "resource authority C successor namespace changed before "
                                "transaction completion"
                            )
                    if not all(publication.still_owned() for publication in publications):
                        raise RuntimeError(
                            "amendment publication ownership changed before completion"
                        )
                    publication_body_completed = True
                except BaseException as publication_error:
                    if publications:
                        try:
                            rollback_owned_publications(publications)
                        except RuntimeError as rollback_error:
                            raise RuntimeError(
                                "amendment publication failed and ownership-safe "
                                "rollback was incomplete"
                            ) from rollback_error
                    raise publication_error
        except BaseException as publication_error:
            if publication_body_completed and publications:
                try:
                    rollback_owned_publications(publications)
                except RuntimeError as rollback_error:
                    raise RuntimeError(
                        "amendment lock exit failed after publication and "
                        "ownership-safe rollback was incomplete"
                    ) from rollback_error
            raise publication_error
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    if published_result is None:  # pragma: no cover - defensive postcondition
        raise RuntimeError("amendment publication returned without a result")
    return published_result


__all__ = [
    "INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE",
    "RESOURCE_BOUNDED_AUTHORITY_C_CONFIG_SEMANTIC_SHA256",
    "RESOURCE_BOUNDED_CAPACITY_POLICY_V2",
    "RESOURCE_BOUNDED_CAPACITY_POLICY_V3",
    "RESOURCE_BOUNDED_CONFIRMATORY_EXECUTION_PURPOSE",
    "RESOURCE_BOUNDED_TECHNICAL_SUCCESSOR_PURPOSE",
    "ConfirmatoryStoragePolicy",
    "FinalizationSuccessorAuthorization",
    "InheritedPriorNumericVerificationAuthorization",
    "PreregistrationAmendmentResult",
    "PreregistrationAmendmentVerification",
    "ResourceBoundedCnnProvenanceCorrection",
    "ResourceBoundedConfirmatoryAuthorization",
    "ResourceBoundedTechnicalSuccessorAuthorization",
    "ResourceBoundedTechnicalSuccessorVerification",
    "build_finalization_successor_authorization",
    "build_resource_bounded_confirmatory_authorization",
    "build_resource_bounded_prior_publication_failure_evidence",
    "build_resource_bounded_technical_successor_authorization",
    "create_preregistration_amendment",
    "publish_resource_bounded_prior_publication_failure_receipt",
    "require_authorized_prior_numeric_verification_proof",
    "require_confirmatory_storage_policy",
    "require_effective_resource_bounded_confirmatory_authorization",
    "require_finalization_successor_authorization",
    "require_primary_recovery_authorization",
    "require_resource_bounded_confirmatory_authorization",
    "require_resource_bounded_technical_successor_authorization",
    "resource_bounded_technical_successor_intent_sha256",
    "validate_resource_bounded_capacity_v3",
    "verify_preregistration_amendment",
    "verify_resource_bounded_prior_publication_failure_receipt",
    "verify_resource_bounded_replacement_terminal_qualification_receipt",
    "verify_resource_bounded_technical_successor",
]
