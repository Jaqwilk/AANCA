"""Fail-closed PanNuke cache adapter for confirmatory fold rotation.

The adapter turns checksum-bound, sample-aligned representation caches into
immutable role-specific inputs.  It never accepts final-reference images as a
training/audit argument: every official fold is isolated in its own
``final_reference`` partition before any downstream model is invoked.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from histo_audit.cross_validation.oof import make_group_stratified_folds
from histo_audit.experiment.confirmatory_memory_workspace import (
    ConfirmatoryMemoryWorkspace,
    ConfirmatoryWorkspaceArraySpec,
    ConfirmatoryWorkspaceIndexSpec,
    ReadOnlyBackingArray,
    RowIndexedArray,
    workspace_index_id,
)
from histo_audit.experiment.reference_groups import (
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    build_confirmatory_matrix_plan,
)
from histo_audit.representations.cache_provenance import (
    array_artifact_sha256,
    ordered_sample_ids_sha256,
)
from histo_audit.representations.eligibility import validate_analysis_eligibility_provenance
from histo_audit.representations.imagenet import load_embedding_cache
from histo_audit.utils.run_tracking import sha256_file

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PartitionRole = Literal["audit", "reference_validation", "final_reference"]
type PartitionArray = NDArray[Any] | RowIndexedArray
type SharedArray = NDArray[Any] | ReadOnlyBackingArray

_CROP_WORKSPACE_MEMBERS = {
    "sample_ids": "sample_ids.npy",
    "group_ids": "group_ids.npy",
    "official_folds": "official_folds.npy",
    "pre_corruption_labels": "pre_corruption_labels.npy",
    "confirmatory_eligible": "confirmatory_eligible.npy",
    "context_rgb": "context_rgb.npy",
    "identity_verified": "identity_verified.npy",
    "primary_eligible": "primary_eligible.npy",
    "target_masks": "target_masks.npy",
}


@dataclass(frozen=True, slots=True)
class ConfirmatoryFrozenFeatureCacheSpec:
    """Exact cache binding for one predeclared frozen-feature scenario."""

    scenario_id: str
    cache_path: Path
    expected_cache_sha256: str
    expected_metadata_sha256: str
    expected_weight_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ConfirmatoryObservedLabelSet:
    """One rotation's explicit controlled-label state in crop-cache order."""

    outer_fold: int
    sample_ids: tuple[str, ...]
    pre_corruption_labels: NDArray[np.int64]
    observed_labels: NDArray[np.int64]
    is_injected_corruption: NDArray[np.bool_]
    corruption_types: tuple[str, ...]
    configuration_sha256: str


@dataclass(frozen=True, slots=True)
class ConfirmatoryFrozenFeatureAvailability:
    """Honest availability and provenance for one frozen-feature scenario."""

    scenario_id: str
    family: str
    required: bool
    available: bool
    blocker: str | None
    cache_path: str | None
    cache_sha256: str | None
    metadata_path: str | None
    metadata_sha256: str | None
    manifest_binding: str | None
    weight_sha256: str | None


@dataclass(frozen=True, slots=True)
class ConfirmatoryPartitionFeature:
    """One read-only feature matrix within a role-specific partition."""

    scenario_id: str
    values: PartitionArray


@dataclass(frozen=True, slots=True)
class ConfirmatoryPartitionInputs:
    """Immutable images, labels, identities, and optional features for one role."""

    role: PartitionRole
    source_indices: NDArray[np.int64]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    context_rgb: PartitionArray
    target_masks: PartitionArray
    pre_corruption_labels: NDArray[np.int64]
    observed_labels: NDArray[np.int64]
    is_injected_corruption: NDArray[np.bool_]
    corruption_types: tuple[str, ...]
    frozen_features: tuple[ConfirmatoryPartitionFeature, ...]

    def validate(self) -> None:
        """Reject mutable, malformed, or semantically inconsistent arrays."""

        n = len(self.sample_ids)
        if not n or len(set(self.sample_ids)) != n or any(not value for value in self.sample_ids):
            raise ValueError(f"{self.role} sample IDs must be non-empty and unique")
        if len(self.group_ids) != n or any(not value for value in self.group_ids):
            raise ValueError(f"{self.role} group IDs must be aligned and non-empty")
        arrays = (
            self.source_indices,
            self.context_rgb,
            self.target_masks,
            self.pre_corruption_labels,
            self.observed_labels,
            self.is_injected_corruption,
        )
        if any(value.flags.writeable for value in arrays):
            raise ValueError(f"{self.role} arrays must be read-only")
        if (
            self.source_indices.shape != (n,)
            or self.source_indices.dtype != np.dtype(np.int64)
            or int(self.source_indices.min()) < 0
            or len(np.unique(self.source_indices)) != n
        ):
            raise ValueError(f"{self.role} source indices are misaligned")
        for name, values in (
            ("context_rgb", self.context_rgb),
            ("target_masks", self.target_masks),
        ):
            if isinstance(values, RowIndexedArray) and not np.array_equal(
                values.source_indices,
                self.source_indices,
            ):
                raise ValueError(
                    f"{self.role} {name} rows differ from authoritative source indices"
                )
        if (
            self.context_rgb.ndim != 4
            or self.context_rgb.shape[0] != n
            or self.context_rgb.shape[-1] != 3
            or self.context_rgb.dtype != np.uint8
        ):
            raise ValueError(f"{self.role} context RGB must be aligned uint8 NHWC")
        if self.target_masks.shape != self.context_rgb.shape[:3] or self.target_masks.dtype != bool:
            raise ValueError(f"{self.role} target masks do not align with RGB")
        if not _all_masks_nonempty(self.target_masks):
            raise ValueError(f"{self.role} contains an empty target mask")
        for name, labels in (
            ("pre_corruption_labels", self.pre_corruption_labels),
            ("observed_labels", self.observed_labels),
        ):
            if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"{self.role} {name} must be an aligned integer vector")
            outside = sorted(set(int(value) for value in labels).difference(CLASS_ORDER))
            if outside:
                raise ValueError(f"{self.role} {name} lies outside fixed classes: {outside}")
        changed = self.observed_labels != self.pre_corruption_labels
        if self.is_injected_corruption.shape != (n,) or not np.array_equal(
            changed, self.is_injected_corruption
        ):
            raise ValueError(f"{self.role} corruption flag differs from label change")
        if len(self.corruption_types) != n:
            raise ValueError(f"{self.role} corruption metadata is misaligned")
        for changed_value, corruption_type in zip(changed, self.corruption_types, strict=True):
            normalised = corruption_type.strip().casefold()
            if bool(changed_value) == (normalised in {"", "none", "uncorrupted"}):
                raise ValueError(f"{self.role} corruption type differs from label state")
        feature_ids: set[str] = set()
        for feature in self.frozen_features:
            if feature.scenario_id in feature_ids:
                raise ValueError(f"{self.role} has duplicate frozen-feature scenarios")
            feature_ids.add(feature.scenario_id)
            if isinstance(feature.values, RowIndexedArray) and not np.array_equal(
                feature.values.source_indices,
                self.source_indices,
            ):
                raise ValueError(
                    f"{self.role}/{feature.scenario_id} rows differ from authoritative "
                    "source indices"
                )
            if (
                feature.values.ndim != 2
                or feature.values.shape[0] != n
                or not feature.values.shape[1]
                or not _all_finite(feature.values)
                or feature.values.flags.writeable
            ):
                raise ValueError(f"{self.role}/{feature.scenario_id} features are invalid")


@dataclass(frozen=True, slots=True)
class PanNukeConfirmatoryRotationInputs:
    """One official-fold rotation with strict train/validation/final roles."""

    outer_fold: int
    split_seed: int
    audit: ConfirmatoryPartitionInputs
    reference_validation: ConfirmatoryPartitionInputs
    final_reference: ConfirmatoryPartitionInputs

    def validate(self, *, oof_splits: int) -> None:
        """Validate role separation and five-class model feasibility."""

        if self.audit.role != "audit":
            raise ValueError("audit partition has an incorrect role")
        if self.reference_validation.role != "reference_validation":
            raise ValueError("reference-validation partition has an incorrect role")
        if self.final_reference.role != "final_reference":
            raise ValueError("final-reference partition has an incorrect role")
        partitions = (self.audit, self.reference_validation, self.final_reference)
        for partition in partitions:
            partition.validate()
            if set(int(value) for value in partition.pre_corruption_labels) != set(CLASS_ORDER):
                raise ValueError(
                    f"outer fold {self.outer_fold} {partition.role} lacks all five classes"
                )
        for first in range(len(partitions)):
            for second in range(first + 1, len(partitions)):
                if set(partitions[first].sample_ids).intersection(partitions[second].sample_ids):
                    raise ValueError(f"outer fold {self.outer_fold} has sample leakage")
                if set(partitions[first].group_ids).intersection(partitions[second].group_ids):
                    raise ValueError(f"outer fold {self.outer_fold} has source-group leakage")
        for clean_partition in (self.reference_validation, self.final_reference):
            if clean_partition.is_injected_corruption.any() or not np.array_equal(
                clean_partition.observed_labels, clean_partition.pre_corruption_labels
            ):
                raise ValueError(
                    f"outer fold {self.outer_fold} {clean_partition.role} must be uncorrupted"
                )
        if set(int(value) for value in self.audit.observed_labels) != set(CLASS_ORDER):
            raise ValueError(f"outer fold {self.outer_fold} audit observed labels lack classes")
        folds = make_group_stratified_folds(
            self.audit.pre_corruption_labels,
            self.audit.group_ids,
            n_splits=oof_splits,
            class_order=CLASS_ORDER,
            seed=self.split_seed,
        )
        required = set(CLASS_ORDER)
        for fold in folds:
            training_classes = set(
                int(value) for value in self.audit.observed_labels[fold.train_indices]
            )
            if missing := sorted(required.difference(training_classes)):
                raise ValueError(
                    f"outer fold {self.outer_fold} OOF fold {fold.fold_id} "
                    f"training data lacks classes: {missing}"
                )


@dataclass(frozen=True, slots=True)
class PanNukeConfirmatoryInputs:
    """Checksum-bound inputs for all three confirmatory fold rotations."""

    config_sha256: str
    manifest_sha256: str
    raw_inventory_sha256: str
    crop_cache_path: str
    crop_cache_sha256: str
    crop_metadata_path: str
    crop_metadata_sha256: str
    rotations: tuple[PanNukeConfirmatoryRotationInputs, ...]
    frozen_feature_availability: tuple[ConfirmatoryFrozenFeatureAvailability, ...]
    execution_mode: str
    study_outcome_eligible: bool
    ineligibility_reasons: tuple[str, ...]
    eligibility_provenance: Mapping[str, Any]
    memory_workspace_path: str | None = None
    memory_workspace_receipt_sha256: str | None = None
    memory_workspace_artifact_root_sha256: str | None = None
    memory_workspace_plan_sha256: str | None = None

    def validate(self, *, official_folds: tuple[int, ...], oof_splits: int) -> None:
        """Reconcile fold coverage, execution eligibility, and all rotations."""

        for digest in (
            self.config_sha256,
            self.manifest_sha256,
            self.raw_inventory_sha256,
            self.crop_cache_sha256,
            self.crop_metadata_sha256,
        ):
            _require_sha256(digest, "stored provenance SHA-256")
        if tuple(rotation.outer_fold for rotation in self.rotations) != official_folds:
            raise ValueError("confirmatory rotations do not follow frozen official-fold order")
        for rotation in self.rotations:
            rotation.validate(oof_splits=oof_splits)
        if self.execution_mode in {"cpu_test_only", "synthetic_fixture"}:
            if self.study_outcome_eligible or not self.ineligibility_reasons:
                raise ValueError("CPU/synthetic inputs must be explicitly ineligible")
        elif self.execution_mode == "real_study":
            if not self.study_outcome_eligible or self.ineligibility_reasons:
                raise ValueError("strict real-study inputs should be eligible")
        else:
            raise ValueError(f"unknown confirmatory execution mode: {self.execution_mode}")
        if not self.eligibility_provenance.get("semantic_sha256"):
            raise ValueError("confirmatory inputs lack analysis-eligibility provenance")
        workspace_values = (
            self.memory_workspace_path,
            self.memory_workspace_receipt_sha256,
            self.memory_workspace_artifact_root_sha256,
            self.memory_workspace_plan_sha256,
        )
        if any(value is None for value in workspace_values) != all(
            value is None for value in workspace_values
        ):
            raise ValueError("confirmatory memory-workspace provenance is only partially bound")
        if self.memory_workspace_path is not None:
            _require_sha256(
                str(self.memory_workspace_receipt_sha256),
                "memory-workspace receipt SHA-256",
            )
            _require_sha256(
                str(self.memory_workspace_artifact_root_sha256),
                "memory-workspace artifact root SHA-256",
            )
            _require_sha256(
                str(self.memory_workspace_plan_sha256),
                "memory-workspace plan SHA-256",
            )


@dataclass(frozen=True, slots=True)
class _CropCache:
    sample_ids: NDArray[np.str_]
    context_rgb: SharedArray
    target_masks: SharedArray
    official_folds: NDArray[np.integer[Any]]
    pre_corruption_labels: NDArray[np.int64]
    group_ids: NDArray[np.str_]
    identity_verified: NDArray[np.bool_]
    primary_eligible: NDArray[np.bool_]
    confirmatory_eligible: NDArray[np.bool_]
    metadata: dict[str, Any]
    eligibility_provenance: dict[str, Any]
    memory_workspace: ConfirmatoryMemoryWorkspace | None


def _require_sha256(value: str, name: str) -> str:
    normalised = str(value).casefold()
    if _SHA256.fullmatch(normalised) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalised


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict finite JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _readonly(values: NDArray[Any], dtype: Any) -> NDArray[Any]:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _shared_chunks(
    values: PartitionArray | SharedArray,
    *,
    max_chunk_bytes: int = 32 * 1024 * 1024,
) -> Iterator[NDArray[Any]]:
    if isinstance(values, RowIndexedArray):
        yield from values.iter_chunks(max_chunk_bytes)
        return
    if isinstance(values, ReadOnlyBackingArray):
        shape = values.shape
        dtype = values.dtype
    else:
        raw = np.asarray(values)
        shape = tuple(raw.shape)
        dtype = raw.dtype
    row_elements = int(np.prod(shape[1:], dtype=np.int64))
    row_bytes = max(1, row_elements * dtype.itemsize)
    if row_bytes > max_chunk_bytes:
        raise ValueError("one shared-array row exceeds the strict chunk byte limit")
    rows_per_chunk = max_chunk_bytes // row_bytes
    if not isinstance(values, ReadOnlyBackingArray):
        for start in range(0, len(values), rows_per_chunk):
            yield np.asarray(raw[start : start + rows_per_chunk])
        return
    for start in range(0, shape[0], rows_per_chunk):
        values.assert_verified_file_identity()
        mapped = np.load(values.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(mapped, np.memmap):
            raise ValueError("shared backing cannot be reopened as a read-only memmap")
        try:
            yield np.asarray(mapped[start : start + rows_per_chunk])
            values.assert_verified_file_identity()
        finally:
            mmap_handle = getattr(mapped, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()


def _all_masks_nonempty(values: PartitionArray | SharedArray) -> bool:
    if isinstance(values, RowIndexedArray) and values.all_rows_nonempty:
        return True
    if isinstance(values, ReadOnlyBackingArray) and values.all_rows_nonempty:
        return True
    for chunk in _shared_chunks(values):
        if not np.asarray(chunk).reshape(len(chunk), -1).any(axis=1).all():
            return False
    return True


def _all_finite(values: PartitionArray | SharedArray) -> bool:
    if isinstance(values, RowIndexedArray) and values.all_finite:
        return True
    if isinstance(values, ReadOnlyBackingArray) and values.all_finite:
        return True
    return all(np.isfinite(chunk).all() for chunk in _shared_chunks(values))


def _shared_shape(values: SharedArray) -> tuple[int, ...]:
    return values.shape if isinstance(values, ReadOnlyBackingArray) else tuple(values.shape)


def _shared_dtype(values: SharedArray) -> np.dtype[Any]:
    return values.dtype


def _shared_readonly(
    values: SharedArray,
    dtype: Any,
) -> SharedArray:
    if isinstance(values, ReadOnlyBackingArray):
        if values.dtype != np.dtype(dtype) or values.flags.writeable:
            raise ValueError("shared backing array differs from its required read-only dtype")
        return values
    return _readonly(values, dtype)


def _partition_array(
    values: SharedArray,
    indices: NDArray[np.int64],
    *,
    logical_dtype: Any,
) -> PartitionArray:
    if isinstance(values, ReadOnlyBackingArray):
        return RowIndexedArray(
            values,
            indices,
            logical_dtype=logical_dtype,
            allow_repeated_indices=False,
        )
    return _readonly(values[indices], logical_dtype)


def _exact_crop_workspace_backings(
    memory_workspace: ConfirmatoryMemoryWorkspace,
    *,
    expected_cache_sha256: str,
    expected_metadata_sha256: str,
) -> dict[str, ReadOnlyBackingArray]:
    """Require and return all nine exact crop-cache members from one workspace."""

    crop_array_ids = {
        array_id
        for array_id, backing in memory_workspace.arrays.items()
        if (
            backing.source_npz_sha256 == expected_cache_sha256
            or backing.source_sidecar_sha256 == expected_metadata_sha256
        )
    }
    if crop_array_ids != set(_CROP_WORKSPACE_MEMBERS):
        raise ValueError(
            "memory workspace must contain exactly the nine checksum-bound crop arrays"
        )
    output: dict[str, ReadOnlyBackingArray] = {}
    for array_id, member_name in _CROP_WORKSPACE_MEMBERS.items():
        backing = memory_workspace.arrays.get(array_id)
        if (
            backing is None
            or backing.array_id != array_id
            or backing.source_npz_sha256 != expected_cache_sha256
            or backing.source_sidecar_sha256 != expected_metadata_sha256
            or backing.source_member_name != member_name
            or not isinstance(backing.values, np.memmap)
            or backing.values.flags.writeable
        ):
            raise ValueError(f"memory workspace lacks the exact crop-cache backing {array_id!r}")
        backing.assert_verified_file_identity()
        output[array_id] = backing
    return output


def _workspace_vector(
    backing: ReadOnlyBackingArray,
    *,
    name: str,
) -> np.memmap:
    """Expose one verified lightweight workspace member without an array copy."""

    backing.assert_verified_file_identity()
    values = backing.values
    if (
        not isinstance(values, np.memmap)
        or values.ndim != 1
        or values.dtype.hasobject
        or values.flags.writeable
    ):
        raise ValueError(f"workspace crop {name} must be a read-only primitive vector")
    return values


def _workspace_identifier_vector(
    backing: ReadOnlyBackingArray,
    *,
    name: str,
) -> NDArray[np.str_]:
    """Return canonical text IDs, copying only legacy fixed-width byte strings."""

    values = _workspace_vector(backing, name=name)
    if (
        values.dtype.kind not in {"U", "S"}
        or values.dtype.fields is not None
        or values.dtype.subdtype is not None
    ):
        raise ValueError(
            f"workspace crop {name} must use a fixed-width Unicode or byte-string dtype"
        )
    if values.dtype.kind == "U":
        return cast(NDArray[np.str_], values)
    try:
        output = np.asarray(values, dtype=np.str_)
    except (UnicodeError, ValueError) as error:
        raise ValueError(
            f"workspace crop {name} contains an invalid fixed-width byte identifier"
        ) from error
    output.setflags(write=False)
    return output


def _load_crop_cache(
    path: Path,
    *,
    expected_cache_sha256: str,
    expected_metadata_sha256: str,
    expected_manifest_sha256: str,
    expected_raw_inventory_sha256: str,
    memory_workspace: ConfirmatoryMemoryWorkspace | None,
) -> _CropCache:
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    if path.suffix.casefold() != ".npz" or not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("crop NPZ and metadata sidecar must both exist")
    actual_cache_sha = sha256_file(path)
    actual_metadata_sha = sha256_file(metadata_path)
    if actual_cache_sha != _require_sha256(expected_cache_sha256, "crop cache SHA-256"):
        raise ValueError("crop cache SHA-256 differs from the frozen binding")
    if actual_metadata_sha != _require_sha256(expected_metadata_sha256, "crop metadata SHA-256"):
        raise ValueError("crop metadata SHA-256 differs from the frozen binding")
    metadata = _read_json(metadata_path, "crop metadata")
    if metadata.get("cache_npz_sha256") != actual_cache_sha:
        raise ValueError("crop cache checksum differs from its metadata sidecar")
    if metadata.get("manifest_sha256") != _require_sha256(
        expected_manifest_sha256, "manifest SHA-256"
    ):
        raise ValueError("crop metadata is bound to a different manifest")
    if metadata.get("raw_inventory_sha256") != _require_sha256(
        expected_raw_inventory_sha256, "raw inventory SHA-256"
    ):
        raise ValueError("crop metadata is bound to a different raw inventory")
    required = {
        *(_CROP_WORKSPACE_MEMBERS),
    }
    context_rgb: SharedArray
    target_masks: SharedArray
    if memory_workspace is None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if missing := sorted(required.difference(payload.files)):
                    raise ValueError(f"crop cache is missing required arrays: {missing}")
                sample_raw = payload["sample_ids"]
                group_raw = payload["group_ids"]
                if sample_raw.dtype.kind not in {"U", "S"} or group_raw.dtype.kind not in {
                    "U",
                    "S",
                }:
                    raise ValueError("crop identifiers must not require pickle/object loading")
                sample_ids = np.asarray(sample_raw, dtype=np.str_)
                context_rgb = np.asarray(payload["context_rgb"])
                target_masks = np.asarray(payload["target_masks"])
                official_folds = np.asarray(payload["official_folds"])
                pre_labels = np.asarray(payload["pre_corruption_labels"])
                group_ids = np.asarray(group_raw, dtype=np.str_)
                identity_verified = np.asarray(payload["identity_verified"])
                primary_eligible = np.asarray(payload["primary_eligible"])
                confirmatory_eligible = np.asarray(payload["confirmatory_eligible"])
        except (OSError, KeyError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith("crop"):
                raise
            raise ValueError("crop cache cannot be loaded without pickle") from error
    else:
        crop_backings = _exact_crop_workspace_backings(
            memory_workspace,
            expected_cache_sha256=actual_cache_sha,
            expected_metadata_sha256=actual_metadata_sha,
        )
        sample_ids = _workspace_identifier_vector(
            crop_backings["sample_ids"],
            name="sample_ids",
        )
        group_ids = _workspace_identifier_vector(
            crop_backings["group_ids"],
            name="group_ids",
        )
        official_folds = _workspace_vector(
            crop_backings["official_folds"],
            name="official_folds",
        )
        pre_labels = _workspace_vector(
            crop_backings["pre_corruption_labels"],
            name="pre_corruption_labels",
        )
        identity_verified = _workspace_vector(
            crop_backings["identity_verified"],
            name="identity_verified",
        )
        primary_eligible = _workspace_vector(
            crop_backings["primary_eligible"],
            name="primary_eligible",
        )
        confirmatory_eligible = _workspace_vector(
            crop_backings["confirmatory_eligible"],
            name="confirmatory_eligible",
        )
        context_rgb = crop_backings["context_rgb"]
        target_masks = crop_backings["target_masks"]
    if sample_ids.dtype.kind not in {"U", "S"} or group_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("crop identifiers must not require pickle/object loading")
    n = len(sample_ids)
    if not n or sample_ids.shape != (n,) or len(set(sample_ids.tolist())) != n:
        raise ValueError("crop sample IDs must be non-empty and unique")
    if (
        any(not value for value in sample_ids)
        or group_ids.shape != (n,)
        or any(not value for value in group_ids)
    ):
        raise ValueError("crop sample/group identities are invalid")
    context_shape = _shared_shape(context_rgb)
    mask_shape = _shared_shape(target_masks)
    if len(context_shape) != 4 or context_shape[0] != n or context_shape[-1] != 3:
        raise ValueError("crop context RGB must have shape (n, height, width, 3)")
    if _shared_dtype(context_rgb) != np.uint8:
        raise ValueError("crop context RGB must be uint8")
    if mask_shape != context_shape[:3] or _shared_dtype(target_masks) != np.dtype(bool):
        raise ValueError("crop target masks must be aligned boolean arrays")
    if not _all_masks_nonempty(target_masks):
        raise ValueError("every crop must retain a non-empty target mask")
    if isinstance(target_masks, ReadOnlyBackingArray):
        target_masks = replace(target_masks, all_rows_nonempty=True)
    if official_folds.shape != (n,) or not np.issubdtype(official_folds.dtype, np.integer):
        raise ValueError("crop official folds must be an aligned integer vector")
    if pre_labels.shape != (n,) or not np.issubdtype(pre_labels.dtype, np.integer):
        raise ValueError("crop pre-corruption labels must be an aligned integer vector")
    if sorted(set(int(value) for value in pre_labels)) != list(CLASS_ORDER):
        raise ValueError("crop cache must contain exactly the fixed five classes")
    if (
        identity_verified.shape != (n,)
        or identity_verified.dtype != bool
        or not identity_verified.all()
    ):
        raise ValueError("crop cache contains an unverified target identity")
    if metadata.get("sample_count") != n:
        raise ValueError("crop metadata sample count differs from arrays")
    eligibility_provenance = validate_analysis_eligibility_provenance(
        metadata,
        sample_ids,
        primary_eligible=primary_eligible,
        confirmatory_eligible=confirmatory_eligible,
    )
    for group in np.unique(group_ids):
        if len(np.unique(official_folds[group_ids == group])) != 1:
            raise ValueError(f"source group {group} spans official folds")
    if memory_workspace is None:
        output_sample_ids = _readonly(sample_ids, np.str_)
        output_official_folds = _readonly(official_folds, np.int64)
        output_pre_labels = _readonly(pre_labels, np.int64)
        output_group_ids = _readonly(group_ids, np.str_)
        output_identity_verified = _readonly(identity_verified, np.bool_)
        output_primary_eligible = _readonly(primary_eligible, np.bool_)
        output_confirmatory_eligible = _readonly(confirmatory_eligible, np.bool_)
    else:
        output_sample_ids = sample_ids
        output_official_folds = official_folds
        output_pre_labels = pre_labels
        output_group_ids = group_ids
        output_identity_verified = identity_verified
        output_primary_eligible = primary_eligible
        output_confirmatory_eligible = confirmatory_eligible
    return _CropCache(
        sample_ids=output_sample_ids,
        context_rgb=_shared_readonly(context_rgb, np.uint8),
        target_masks=_shared_readonly(target_masks, np.bool_),
        official_folds=output_official_folds,
        pre_corruption_labels=output_pre_labels,
        group_ids=output_group_ids,
        identity_verified=output_identity_verified,
        primary_eligible=output_primary_eligible,
        confirmatory_eligible=output_confirmatory_eligible,
        metadata=metadata,
        eligibility_provenance=eligibility_provenance,
        memory_workspace=memory_workspace,
    )


def _select_reference_groups(
    labels: NDArray[np.int64],
    groups: NDArray[np.str_],
    *,
    fraction: float,
    seed: int,
) -> set[str]:
    return set(
        deterministic_group_greedy_class_distribution_v1(
            labels,
            groups,
            class_order=CLASS_ORDER,
            fraction=fraction,
            seed=seed,
        )
    )


def _clean_observed_labels(
    outer_fold: int,
    cache: _CropCache,
    *,
    config_sha256: str,
) -> ConfirmatoryObservedLabelSet:
    n = len(cache.sample_ids)
    return ConfirmatoryObservedLabelSet(
        outer_fold=outer_fold,
        sample_ids=tuple(str(value) for value in cache.sample_ids),
        pre_corruption_labels=_readonly(cache.pre_corruption_labels, np.int64),
        observed_labels=_readonly(cache.pre_corruption_labels, np.int64),
        is_injected_corruption=_readonly(np.zeros(n, dtype=bool), np.bool_),
        corruption_types=tuple("none" for _ in range(n)),
        configuration_sha256=config_sha256,
    )


def _validate_observed_labels(
    values: ConfirmatoryObservedLabelSet,
    cache: _CropCache,
    *,
    outer_fold: int,
) -> ConfirmatoryObservedLabelSet:
    n = len(cache.sample_ids)
    if values.outer_fold != outer_fold:
        raise ValueError("observed-label rotation key differs from its outer_fold")
    if values.sample_ids != tuple(str(value) for value in cache.sample_ids):
        raise ValueError("observed-label sample IDs/order differ from crop cache")
    pre = np.asarray(values.pre_corruption_labels)
    observed = np.asarray(values.observed_labels)
    injected = np.asarray(values.is_injected_corruption)
    if pre.shape != (n,) or not np.issubdtype(pre.dtype, np.integer):
        raise ValueError("observed-label pre-corruption vector is invalid")
    if not np.array_equal(pre, cache.pre_corruption_labels):
        raise ValueError("observed-label reference vector differs from crop cache")
    if observed.shape != (n,) or not np.issubdtype(observed.dtype, np.integer):
        raise ValueError("observed labels must be an aligned integer vector")
    if not np.isin(observed, CLASS_ORDER).all():
        raise ValueError("observed labels lie outside the fixed class order")
    if injected.shape != (n,) or injected.dtype != bool:
        raise ValueError("injected-corruption flags must be aligned booleans")
    if not np.array_equal(injected, observed != pre):
        raise ValueError("injected-corruption flags differ from label changes")
    if len(values.corruption_types) != n:
        raise ValueError("corruption types do not align with observed labels")
    _require_sha256(values.configuration_sha256, "corruption configuration SHA-256")
    output = ConfirmatoryObservedLabelSet(
        outer_fold=outer_fold,
        sample_ids=values.sample_ids,
        pre_corruption_labels=_readonly(pre, np.int64),
        observed_labels=_readonly(observed, np.int64),
        is_injected_corruption=_readonly(injected, np.bool_),
        corruption_types=tuple(str(value) for value in values.corruption_types),
        configuration_sha256=values.configuration_sha256,
    )
    return output


def _load_feature_cache(
    spec: ConfirmatoryFrozenFeatureCacheSpec,
    *,
    sample_ids: NDArray[np.str_],
    manifest_sha256: str,
    expected_eligibility_sha256: str,
    memory_workspace: ConfirmatoryMemoryWorkspace | None,
) -> tuple[SharedArray, dict[str, Any], str]:
    path = Path(spec.cache_path).resolve()
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"frozen-feature cache/sidecar is absent: {path}")
    cache_sha = sha256_file(path)
    metadata_sha = sha256_file(metadata_path)
    if cache_sha != _require_sha256(spec.expected_cache_sha256, "feature cache SHA-256"):
        raise ValueError(f"{spec.scenario_id} feature cache differs from frozen SHA-256")
    if metadata_sha != _require_sha256(spec.expected_metadata_sha256, "feature metadata SHA-256"):
        raise ValueError(f"{spec.scenario_id} feature sidecar differs from frozen SHA-256")
    metadata = _read_json(metadata_path, f"{spec.scenario_id} feature metadata")
    if metadata.get("cache_npz_sha256") != cache_sha:
        raise ValueError(f"{spec.scenario_id} cache checksum differs from sidecar")
    try:
        with np.load(path, allow_pickle=False) as payload:
            files = set(payload.files)
    except (OSError, ValueError) as error:
        raise ValueError(f"{spec.scenario_id} cache cannot be opened without pickle") from error
    backing_id = f"feature__{spec.scenario_id}"
    feature_backing = (
        memory_workspace.arrays.get(backing_id) if memory_workspace is not None else None
    )
    if feature_backing is not None and (
        feature_backing.source_npz_sha256 != cache_sha
        or feature_backing.source_sidecar_sha256 != metadata_sha
    ):
        raise ValueError(f"{spec.scenario_id} memory backing differs from its cache binding")
    if {"embeddings", "sample_ids", "metadata_json"}.issubset(files):
        if feature_backing is None:
            loaded = load_embedding_cache(path)
            matrix: SharedArray = np.asarray(loaded.embeddings, dtype=np.float64)
            identifiers = np.asarray(loaded.sample_ids, dtype=np.str_)
        else:
            if feature_backing.source_member_name != "embeddings.npy":
                raise ValueError(f"{spec.scenario_id} memory backing has the wrong NPZ member")
            with np.load(path, allow_pickle=False) as payload:
                raw_ids = payload["sample_ids"]
            if raw_ids.dtype.kind not in {"U", "S"}:
                raise ValueError(f"{spec.scenario_id} sample IDs require object/pickle")
            identifiers = np.asarray(raw_ids, dtype=np.str_)
            matrix = feature_backing
    elif {"values", "sample_ids"}.issubset(files):
        with np.load(path, allow_pickle=False) as payload:
            raw_ids = payload["sample_ids"]
            if raw_ids.dtype.kind not in {"U", "S"}:
                raise ValueError(f"{spec.scenario_id} sample IDs require object/pickle")
            identifiers = np.asarray(raw_ids, dtype=np.str_)
            matrix = (
                feature_backing
                if feature_backing is not None
                else np.asarray(payload["values"], dtype=np.float64)
            )
        if feature_backing is not None and feature_backing.source_member_name != "values.npy":
            raise ValueError(f"{spec.scenario_id} memory backing has the wrong NPZ member")
    else:
        raise ValueError(f"{spec.scenario_id} has an unsupported feature-cache schema")
    if memory_workspace is not None and feature_backing is None:
        raise ValueError(f"{spec.scenario_id} is absent from the exact memory workspace")
    if not np.array_equal(identifiers, sample_ids):
        raise ValueError(f"{spec.scenario_id} sample IDs/order differ from crop cache")
    matrix_shape = _shared_shape(matrix)
    if len(matrix_shape) != 2 or matrix_shape[0] != len(sample_ids) or not matrix_shape[1]:
        raise ValueError(f"{spec.scenario_id} feature matrix is misaligned")
    if not _all_finite(matrix):
        raise ValueError(f"{spec.scenario_id} feature matrix contains non-finite values")
    if isinstance(matrix, ReadOnlyBackingArray):
        matrix = replace(matrix, all_finite=True)
    feature_eligibility = validate_analysis_eligibility_provenance(metadata, sample_ids)
    if feature_eligibility["semantic_sha256"] != expected_eligibility_sha256:
        raise ValueError(f"{spec.scenario_id} eligibility provenance differs from crop cache")
    direct_manifest = metadata.get("crop_manifest_sha256", metadata.get("manifest_sha256"))
    if direct_manifest is not None and direct_manifest != manifest_sha256:
        raise ValueError(f"{spec.scenario_id} feature sidecar references another manifest")
    manifest_binding = (
        "feature_sidecar_manifest_sha256"
        if direct_manifest is not None
        else "exact_sample_order_to_checksum_bound_crop_manifest"
    )
    weight_sha = metadata.get("weight_sha256")
    if spec.expected_weight_sha256 is not None:
        expected_weight = _require_sha256(spec.expected_weight_sha256, "encoder weight SHA-256")
        if weight_sha != expected_weight:
            raise ValueError(f"{spec.scenario_id} encoder weight SHA-256 differs")
    output: SharedArray = (
        matrix if isinstance(matrix, ReadOnlyBackingArray) else _readonly(matrix, np.float64)
    )
    return output, metadata, manifest_binding


def _partition(
    outer_fold: int,
    role: PartitionRole,
    indices: NDArray[np.int64],
    cache: _CropCache,
    labels: ConfirmatoryObservedLabelSet,
    features: Mapping[str, SharedArray],
) -> ConfirmatoryPartitionInputs:
    computed_indices = _readonly(indices, np.int64)
    if cache.memory_workspace is None:
        typed_indices = computed_indices
    else:
        index_id = workspace_index_id(outer_fold, role)
        physical = cache.memory_workspace.index_arrays.get(index_id)
        if (
            physical is None
            or physical.dtype != np.dtype(np.int64)
            or physical.shape != computed_indices.shape
            or physical.flags.writeable
            or not np.array_equal(physical, computed_indices)
            or array_artifact_sha256(physical) != array_artifact_sha256(computed_indices)
        ):
            raise ValueError(
                f"memory workspace index {index_id!r} differs from the frozen role split"
            )
        typed_indices = physical
    output = ConfirmatoryPartitionInputs(
        role=role,
        source_indices=typed_indices,
        sample_ids=tuple(str(cache.sample_ids[index]) for index in indices),
        group_ids=tuple(str(cache.group_ids[index]) for index in indices),
        context_rgb=_partition_array(cache.context_rgb, typed_indices, logical_dtype=np.uint8),
        target_masks=_partition_array(cache.target_masks, typed_indices, logical_dtype=np.bool_),
        pre_corruption_labels=_readonly(cache.pre_corruption_labels[indices], np.int64),
        observed_labels=_readonly(labels.observed_labels[indices], np.int64),
        is_injected_corruption=_readonly(labels.is_injected_corruption[indices], np.bool_),
        corruption_types=tuple(labels.corruption_types[index] for index in indices),
        frozen_features=tuple(
            ConfirmatoryPartitionFeature(
                scenario_id=scenario_id,
                values=_partition_array(values, typed_indices, logical_dtype=np.float64),
            )
            for scenario_id, values in sorted(features.items())
        ),
    )
    output.validate()
    return output


def _workspace_member_header(
    path: Path,
    member_name: str,
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Read only a checksum-bound NPY header from a compressed NPZ member."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            matches = [item for item in archive.infolist() if item.filename == member_name]
            if len(matches) != 1:
                raise ValueError(
                    f"workspace source must contain exactly one {member_name!r} member"
                )
            info = matches[0]
            if info.flag_bits & 0x1 or info.compress_type not in {
                zipfile.ZIP_DEFLATED,
                zipfile.ZIP_STORED,
            }:
                raise ValueError("workspace source member violates the ZIP security policy")
            with archive.open(info, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY member version {version!r}")
                header_bytes = stream.tell()
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(f"workspace source member {member_name!r} is invalid") from error
    exact_dtype = np.dtype(dtype)
    exact_shape = tuple(int(value) for value in shape)
    expected_file_bytes = header_bytes + (
        int(np.prod(exact_shape, dtype=np.int64)) * exact_dtype.itemsize
    )
    if (
        fortran_order
        or not exact_shape
        or any(value <= 0 for value in exact_shape)
        or exact_dtype.hasobject
        or exact_dtype.fields is not None
        or exact_dtype.subdtype is not None
        or exact_dtype.kind not in {"S", "U", "b", "i", "u", "f"}
        or info.file_size != expected_file_bytes
    ):
        raise ValueError(
            f"workspace source member {member_name!r} lacks an exact primitive C array"
        )
    return exact_dtype, exact_shape


def _workspace_array_specs_from_sidecar(
    *,
    source_path: Path,
    expected_source_sha256: str,
    expected_sidecar_sha256: str,
    member_keys_by_array_id: Mapping[str, str],
) -> tuple[ConfirmatoryWorkspaceArraySpec, ...]:
    source = source_path.resolve()
    sidecar = source.with_suffix(f"{source.suffix}.metadata.json")
    if not member_keys_by_array_id:
        raise ValueError("workspace source requires at least one exact array member")
    source_sha = _require_sha256(expected_source_sha256, "workspace source SHA-256")
    sidecar_sha = _require_sha256(
        expected_sidecar_sha256,
        "workspace source sidecar SHA-256",
    )
    if (
        source.suffix.casefold() != ".npz"
        or not source.is_file()
        or not sidecar.is_file()
        or sha256_file(source) != source_sha
        or sha256_file(sidecar) != sidecar_sha
    ):
        raise ValueError("workspace source differs from its frozen binding")
    metadata = _read_json(sidecar, "workspace source sidecar")
    hashes = metadata.get("cache_array_sha256_by_name")
    if metadata.get("cache_npz_sha256") != source_sha or not isinstance(hashes, Mapping):
        raise ValueError("workspace source sidecar lacks exact cache-array provenance")
    output: list[ConfirmatoryWorkspaceArraySpec] = []
    for array_id, member_key in member_keys_by_array_id.items():
        if member_key not in hashes:
            raise ValueError(f"{array_id} sidecar lacks exact cache-array provenance")
        expected_array_sha = _require_sha256(
            str(hashes[member_key]),
            f"{array_id} logical array SHA-256",
        )
        member_name = f"{member_key}.npy"
        dtype, shape = _workspace_member_header(source, member_name)
        output.append(
            ConfirmatoryWorkspaceArraySpec(
                array_id=array_id,
                source_npz_path=source,
                source_sidecar_path=sidecar,
                expected_source_sha256=source_sha,
                expected_source_sidecar_sha256=sidecar_sha,
                member_name=member_name,
                expected_dtype=dtype.str,
                expected_shape=shape,
                expected_array_sha256=expected_array_sha,
            )
        )
    return tuple(output)


def derive_pannuke_confirmatory_workspace_array_specs(
    crop_cache_path: str | Path,
    *,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec],
) -> tuple[ConfirmatoryWorkspaceArraySpec, ...]:
    """Derive all 12 extraction specs without materialising any heavy member."""

    crop_path = Path(crop_cache_path)
    output = list(
        _workspace_array_specs_from_sidecar(
            source_path=crop_path,
            expected_source_sha256=expected_crop_cache_sha256,
            expected_sidecar_sha256=expected_crop_metadata_sha256,
            member_keys_by_array_id={
                array_id: member_name.removesuffix(".npy")
                for array_id, member_name in _CROP_WORKSPACE_MEMBERS.items()
            },
        )
    )
    seen_scenarios: set[str] = set()
    for feature in frozen_feature_caches:
        if feature.scenario_id in seen_scenarios:
            raise ValueError("workspace feature specifications contain a duplicate scenario")
        seen_scenarios.add(feature.scenario_id)
        source = Path(feature.cache_path).resolve()
        try:
            with zipfile.ZipFile(source, "r") as archive:
                names = {item.filename for item in archive.infolist()}
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"{feature.scenario_id} workspace feature cache is invalid") from error
        if "embeddings.npy" in names:
            member_key = "embeddings"
        elif "values.npy" in names:
            member_key = "values"
        else:
            raise ValueError(
                f"{feature.scenario_id} workspace feature cache lacks a supported matrix"
            )
        output.extend(
            _workspace_array_specs_from_sidecar(
                source_path=source,
                expected_source_sha256=feature.expected_cache_sha256,
                expected_sidecar_sha256=feature.expected_metadata_sha256,
                member_keys_by_array_id={
                    f"feature__{feature.scenario_id}": member_key,
                },
            )
        )
    return tuple(sorted(output, key=lambda value: value.array_id))


def derive_pannuke_confirmatory_workspace_index_specs(
    crop_cache_path: str | Path,
    *,
    confirmatory_config: Mapping[str, Any],
    expected_config_sha256: str,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_manifest_sha256: str,
    expected_raw_inventory_sha256: str,
) -> tuple[ConfirmatoryWorkspaceIndexSpec, ...]:
    """Derive all nine role-index bindings without loading heavy NPZ members."""

    plan = build_confirmatory_matrix_plan(confirmatory_config)
    frozen_config_sha = _require_sha256(expected_config_sha256, "confirmatory config SHA-256")
    if plan.config_sha256 != frozen_config_sha:
        raise ValueError("confirmatory configuration differs from the frozen SHA-256")
    data = confirmatory_config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("confirmatory data controls are unavailable")
    official_folds = tuple(int(value) for value in data["official_folds"])
    if official_folds != (1, 2, 3):
        raise ValueError("PanNuke workspace indices require the exact three official folds")
    fraction = float(data["reference_validation_fraction_groups"])
    split_seed = int(data["split_seed"])
    authority = data.get("analysis_manifest_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("confirmatory analysis manifest authority is unavailable")
    authority_manifest = _require_sha256(
        str(authority["canonical_manifest_sha256"]),
        "confirmatory canonical manifest SHA-256",
    )
    if _require_sha256(expected_manifest_sha256, "manifest SHA-256") != authority_manifest:
        raise ValueError("workspace indices use a different canonical manifest")

    path = Path(crop_cache_path).resolve()
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    if (
        path.suffix.casefold() != ".npz"
        or not path.is_file()
        or not metadata_path.is_file()
        or sha256_file(path) != _require_sha256(expected_crop_cache_sha256, "crop cache SHA-256")
        or sha256_file(metadata_path)
        != _require_sha256(expected_crop_metadata_sha256, "crop metadata SHA-256")
    ):
        raise ValueError("workspace-index crop cache differs from its exact binding")
    metadata = _read_json(metadata_path, "crop metadata")
    if (
        metadata.get("cache_npz_sha256") != expected_crop_cache_sha256
        or metadata.get("manifest_sha256") != expected_manifest_sha256
        or metadata.get("raw_inventory_sha256")
        != _require_sha256(expected_raw_inventory_sha256, "raw inventory SHA-256")
    ):
        raise ValueError("workspace-index crop metadata differs from its exact authorities")
    try:
        with np.load(path, allow_pickle=False) as payload:
            sample_ids = np.asarray(payload["sample_ids"], dtype=np.str_)
            group_ids = np.asarray(payload["group_ids"], dtype=np.str_)
            official_fold_values = np.asarray(payload["official_folds"], dtype=np.int64)
            labels = np.asarray(payload["pre_corruption_labels"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("workspace index fields cannot be loaded safely") from error
    n = len(sample_ids)
    if (
        not n
        or group_ids.shape != (n,)
        or official_fold_values.shape != (n,)
        or labels.shape != (n,)
        or len(set(sample_ids.tolist())) != n
        or any(not value for value in sample_ids)
        or any(not value for value in group_ids)
        or set(int(value) for value in official_fold_values) != set(official_folds)
        or set(int(value) for value in labels) != set(CLASS_ORDER)
        or metadata.get("sample_count") != n
        or int(authority["analysis_eligible_sample_count"]) != n
        or ordered_sample_ids_sha256(sample_ids)
        != _require_sha256(
            str(authority["analysis_eligible_sample_order_sha256"]),
            "confirmatory analysis sample-order SHA-256",
        )
    ):
        raise ValueError("workspace index fields differ from the frozen sample universe")
    for group in np.unique(group_ids):
        if len(np.unique(official_fold_values[group_ids == group])) != 1:
            raise ValueError(f"source group {group} spans official folds")

    all_indices = np.arange(n, dtype=np.int64)
    output: list[ConfirmatoryWorkspaceIndexSpec] = []
    for outer_fold in official_folds:
        final_mask = official_fold_values == outer_fold
        development_mask = ~final_mask
        reference_groups = _select_reference_groups(
            labels[development_mask],
            group_ids[development_mask],
            fraction=fraction,
            seed=split_seed,
        )
        reference_mask = development_mask & np.isin(group_ids, tuple(reference_groups))
        role_indices = {
            "audit": all_indices[development_mask & ~reference_mask],
            "reference_validation": all_indices[reference_mask],
            "final_reference": all_indices[final_mask],
        }
        if sum(len(values) for values in role_indices.values()) != n:
            raise RuntimeError("workspace role indices do not partition the source universe")
        for role in ("audit", "reference_validation", "final_reference"):
            values = np.ascontiguousarray(role_indices[role], dtype=np.int64)
            if not len(values) or len(np.unique(values)) != len(values):
                raise ValueError("workspace role indices must be non-empty and unique")
            output.append(
                ConfirmatoryWorkspaceIndexSpec(
                    outer_fold=outer_fold,
                    role=cast(Any, role),
                    source_indices=values,
                )
            )
    return tuple(output)


def load_pannuke_confirmatory_inputs(
    crop_cache_path: str | Path,
    *,
    confirmatory_config: Mapping[str, Any],
    expected_config_sha256: str,
    expected_crop_cache_sha256: str,
    expected_crop_metadata_sha256: str,
    expected_manifest_sha256: str,
    expected_raw_inventory_sha256: str,
    frozen_feature_caches: Sequence[ConfirmatoryFrozenFeatureCacheSpec] = (),
    observed_label_sets: Sequence[ConfirmatoryObservedLabelSet] = (),
    cpu_test_only: bool = False,
    synthetic_fixture: bool = False,
    memory_workspace: ConfirmatoryMemoryWorkspace | None = None,
) -> PanNukeConfirmatoryInputs:
    """Load immutable, group-safe data for every frozen official-fold rotation.

    Missing required frozen-feature scenarios fail.  Missing optional pathology
    scenarios are retained as explicit unavailable records; no feature matrix or
    pathology outcome is fabricated.
    """

    if cpu_test_only and synthetic_fixture:
        raise ValueError("CPU test-only and synthetic-fixture modes are mutually exclusive")
    plan = build_confirmatory_matrix_plan(confirmatory_config)
    frozen_config_sha = _require_sha256(expected_config_sha256, "confirmatory config SHA-256")
    if plan.config_sha256 != frozen_config_sha:
        raise ValueError("confirmatory configuration differs from the frozen SHA-256")
    data = confirmatory_config.get("data")
    oof = confirmatory_config.get("oof")
    scenarios_raw = confirmatory_config.get("scenarios")
    if (
        not isinstance(data, Mapping)
        or not isinstance(oof, Mapping)
        or not isinstance(scenarios_raw, Sequence)
    ):
        raise ValueError("validated confirmatory configuration structure is unavailable")
    official_folds = tuple(int(value) for value in data["official_folds"])
    fraction = float(data["reference_validation_fraction_groups"])
    split_seed = int(data["split_seed"])
    oof_splits = int(oof["n_splits"])
    authority = data["analysis_manifest_authority"]
    if not isinstance(authority, Mapping):  # pragma: no cover - strict plan rejects this first
        raise ValueError("confirmatory analysis manifest authority is unavailable")
    authority_manifest_sha256 = _require_sha256(
        str(authority["canonical_manifest_sha256"]),
        "confirmatory canonical manifest SHA-256",
    )
    authority_sample_order_sha256 = _require_sha256(
        str(authority["analysis_eligible_sample_order_sha256"]),
        "confirmatory analysis sample-order SHA-256",
    )
    authority_sample_count = int(authority["analysis_eligible_sample_count"])
    if _require_sha256(expected_manifest_sha256, "manifest SHA-256") != authority_manifest_sha256:
        raise ValueError(
            "confirmatory inputs use a different manifest than data.analysis_manifest_authority"
        )

    crop_path = Path(crop_cache_path).resolve()
    cache = _load_crop_cache(
        crop_path,
        expected_cache_sha256=expected_crop_cache_sha256,
        expected_metadata_sha256=expected_crop_metadata_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_raw_inventory_sha256=expected_raw_inventory_sha256,
        memory_workspace=memory_workspace,
    )
    if len(cache.sample_ids) != authority_sample_count:
        raise ValueError("crop cache sample count differs from data.analysis_manifest_authority")
    if ordered_sample_ids_sha256(cache.sample_ids) != authority_sample_order_sha256:
        raise ValueError("crop cache sample order differs from data.analysis_manifest_authority")
    actual_folds = set(int(value) for value in cache.official_folds)
    if actual_folds != set(official_folds):
        raise ValueError(
            "crop cache official folds differ from frozen rotation; "
            f"expected={sorted(official_folds)}, actual={sorted(actual_folds)}"
        )

    scenarios: dict[str, Mapping[str, Any]] = {}
    for raw in scenarios_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("confirmatory scenario must be a mapping")
        scenarios[str(raw["id"])] = raw
    frozen_scenarios = {
        scenario_id: scenario
        for scenario_id, scenario in scenarios.items()
        if str(scenario["family"]) in {"imagenet_frozen", "pathology_frozen"}
    }
    specs = {spec.scenario_id: spec for spec in frozen_feature_caches}
    if len(specs) != len(tuple(frozen_feature_caches)):
        raise ValueError("frozen-feature cache specs contain duplicate scenario IDs")
    if unexpected := sorted(set(specs).difference(frozen_scenarios)):
        raise ValueError(f"feature caches target undeclared/non-frozen scenarios: {unexpected}")
    feature_matrices: dict[str, SharedArray] = {}
    availability: list[ConfirmatoryFrozenFeatureAvailability] = []
    for scenario_id, scenario in frozen_scenarios.items():
        required = bool(scenario["required"])
        family = str(scenario["family"])
        spec = specs.get(scenario_id)
        if spec is None:
            if required:
                raise FileNotFoundError(
                    f"required frozen-feature cache was not supplied: {scenario_id}"
                )
            availability.append(
                ConfirmatoryFrozenFeatureAvailability(
                    scenario_id=scenario_id,
                    family=family,
                    required=False,
                    available=False,
                    blocker="optional frozen-feature cache not supplied",
                    cache_path=None,
                    cache_sha256=None,
                    metadata_path=None,
                    metadata_sha256=None,
                    manifest_binding=None,
                    weight_sha256=None,
                )
            )
            continue
        try:
            matrix, metadata, manifest_binding = _load_feature_cache(
                spec,
                sample_ids=cache.sample_ids,
                manifest_sha256=expected_manifest_sha256,
                expected_eligibility_sha256=str(cache.eligibility_provenance["semantic_sha256"]),
                memory_workspace=memory_workspace,
            )
        except (FileNotFoundError, ValueError) as error:
            if required:
                raise
            availability.append(
                ConfirmatoryFrozenFeatureAvailability(
                    scenario_id=scenario_id,
                    family=family,
                    required=False,
                    available=False,
                    blocker=str(error),
                    cache_path=str(Path(spec.cache_path).resolve()),
                    cache_sha256=None,
                    metadata_path=str(
                        Path(spec.cache_path)
                        .resolve()
                        .with_suffix(f"{Path(spec.cache_path).suffix}.metadata.json")
                    ),
                    metadata_sha256=None,
                    manifest_binding=None,
                    weight_sha256=None,
                )
            )
            continue
        feature_matrices[scenario_id] = matrix
        path = Path(spec.cache_path).resolve()
        sidecar = path.with_suffix(f"{path.suffix}.metadata.json")
        availability.append(
            ConfirmatoryFrozenFeatureAvailability(
                scenario_id=scenario_id,
                family=family,
                required=required,
                available=True,
                blocker=None,
                cache_path=str(path),
                cache_sha256=sha256_file(path),
                metadata_path=str(sidecar),
                metadata_sha256=sha256_file(sidecar),
                manifest_binding=manifest_binding,
                weight_sha256=(
                    str(metadata["weight_sha256"])
                    if metadata.get("weight_sha256") is not None
                    else None
                ),
            )
        )

    observed_by_fold = {value.outer_fold: value for value in observed_label_sets}
    if len(observed_by_fold) != len(tuple(observed_label_sets)):
        raise ValueError("observed-label sets contain duplicate outer folds")
    if observed_by_fold and set(observed_by_fold) != set(official_folds):
        raise ValueError("observed-label sets must cover every frozen official fold")
    rotations: list[PanNukeConfirmatoryRotationInputs] = []
    all_indices = np.arange(len(cache.sample_ids), dtype=np.int64)
    for outer_fold in official_folds:
        raw_labels = observed_by_fold.get(outer_fold) or _clean_observed_labels(
            outer_fold,
            cache,
            config_sha256=frozen_config_sha,
        )
        labels = _validate_observed_labels(raw_labels, cache, outer_fold=outer_fold)
        final_mask = cache.official_folds == outer_fold
        development_mask = ~final_mask
        development_indices = all_indices[development_mask]
        final_indices = all_indices[final_mask]
        reference_groups = _select_reference_groups(
            cache.pre_corruption_labels[development_mask],
            cache.group_ids[development_mask],
            fraction=fraction,
            seed=split_seed,
        )
        reference_mask = development_mask & np.isin(cache.group_ids, tuple(reference_groups))
        audit_mask = development_mask & ~reference_mask
        audit_indices = all_indices[audit_mask]
        reference_indices = all_indices[reference_mask]
        if len(development_indices) != len(audit_indices) + len(reference_indices):
            raise RuntimeError("internal confirmatory development split lost samples")
        rotation = PanNukeConfirmatoryRotationInputs(
            outer_fold=outer_fold,
            split_seed=split_seed,
            audit=_partition(
                outer_fold,
                "audit",
                audit_indices,
                cache,
                labels,
                feature_matrices,
            ),
            reference_validation=_partition(
                outer_fold,
                "reference_validation",
                reference_indices,
                cache,
                labels,
                feature_matrices,
            ),
            final_reference=_partition(
                outer_fold,
                "final_reference",
                final_indices,
                cache,
                labels,
                feature_matrices,
            ),
        )
        rotation.validate(oof_splits=oof_splits)
        rotations.append(rotation)

    reasons: tuple[str, ...]
    if cpu_test_only:
        execution_mode = "cpu_test_only"
        reasons = ("CPU test-only inputs are permanently ineligible for study outcomes",)
    elif synthetic_fixture:
        execution_mode = "synthetic_fixture"
        reasons = ("synthetic fixture inputs are permanently ineligible for study outcomes",)
    else:
        execution_mode = "real_study"
        reasons = ()
    metadata_path = crop_path.with_suffix(f"{crop_path.suffix}.metadata.json")
    result = PanNukeConfirmatoryInputs(
        config_sha256=frozen_config_sha,
        manifest_sha256=expected_manifest_sha256,
        raw_inventory_sha256=expected_raw_inventory_sha256,
        crop_cache_path=str(crop_path),
        crop_cache_sha256=sha256_file(crop_path),
        crop_metadata_path=str(metadata_path),
        crop_metadata_sha256=sha256_file(metadata_path),
        rotations=tuple(rotations),
        frozen_feature_availability=tuple(availability),
        execution_mode=execution_mode,
        study_outcome_eligible=not reasons,
        ineligibility_reasons=reasons,
        eligibility_provenance=cache.eligibility_provenance,
        memory_workspace_path=(
            str(memory_workspace.root) if memory_workspace is not None else None
        ),
        memory_workspace_receipt_sha256=(
            memory_workspace.receipt_sha256 if memory_workspace is not None else None
        ),
        memory_workspace_artifact_root_sha256=(
            memory_workspace.artifact_root_sha256 if memory_workspace is not None else None
        ),
        memory_workspace_plan_sha256=(
            memory_workspace.resource_input_workspace_plan_sha256
            if memory_workspace is not None
            else None
        ),
    )
    result.validate(official_folds=official_folds, oof_splits=oof_splits)
    return result


__all__ = [
    "ConfirmatoryFrozenFeatureAvailability",
    "ConfirmatoryFrozenFeatureCacheSpec",
    "ConfirmatoryObservedLabelSet",
    "ConfirmatoryPartitionFeature",
    "ConfirmatoryPartitionInputs",
    "PanNukeConfirmatoryInputs",
    "PanNukeConfirmatoryRotationInputs",
    "derive_pannuke_confirmatory_workspace_array_specs",
    "derive_pannuke_confirmatory_workspace_index_specs",
    "load_pannuke_confirmatory_inputs",
]
