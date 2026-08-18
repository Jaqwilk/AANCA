"""Fail-closed reconciliation of tracked synthetic smoke artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.data.manifest import NucleusRecord
from histo_audit.data.targets import highlight_target, mask_bbox
from histo_audit.reporting.synthetic_duplicates import (
    SyntheticDuplicateAuditError,
    reconcile_synthetic_duplicate_audit,
)


class ArtifactReconciliationError(ValueError):
    """Raised when saved smoke artifacts disagree with one another."""


@dataclass(frozen=True, slots=True)
class _SyntheticDatasetEvidence:
    summary: Mapping[str, int]
    final_sample_ids: tuple[str, ...]
    final_group_ids: tuple[str, ...]
    final_reference_labels: NDArray[np.int64]
    final_is_injected_corruption: NDArray[np.bool_]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactReconciliationError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactReconciliationError(f"{path} must be an array")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactReconciliationError(f"required artifact is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactReconciliationError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactReconciliationError(f"{path.name} root must be an object")
    return value


def _strings(values: NDArray[np.generic], path: str) -> tuple[str, ...]:
    if values.ndim != 1:
        raise ArtifactReconciliationError(f"{path} must be a vector")
    if values.dtype.kind not in {"S", "U"}:
        raise ArtifactReconciliationError(f"{path} must contain saved string identifiers")
    rendered = tuple(str(value) for value in values.tolist())
    if any(not value for value in rendered):
        raise ArtifactReconciliationError(f"{path} contains an empty identifier")
    return rendered


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactReconciliationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactReconciliationError(f"{path} must be at least {minimum}")
    return value


def _integer_array(value: Any, path: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ArtifactReconciliationError(f"{path} must contain saved integer values")
    return np.asarray(array, dtype=np.int64)


def _boolean_array(value: Any, path: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.bool_):
        raise ArtifactReconciliationError(f"{path} must contain saved boolean values")
    return np.asarray(array, dtype=bool)


def _string_array(value: Any, path: str) -> NDArray[np.str_]:
    array = np.asarray(value)
    if array.dtype.kind not in {"S", "U"}:
        raise ArtifactReconciliationError(f"{path} must contain saved string values")
    return np.asarray(array, dtype=np.str_)


def _scalar_integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    array = np.asarray(value)
    if (
        array.shape != ()
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ArtifactReconciliationError(f"{path} must be a saved integer scalar")
    rendered = int(array.item())
    if minimum is not None and rendered < minimum:
        raise ArtifactReconciliationError(f"{path} must be at least {minimum}")
    return rendered


def _scalar_float(value: Any, path: str) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise ArtifactReconciliationError(f"{path} must be a saved numeric scalar")
    rendered = float(array.item())
    if not math.isfinite(rendered):
        raise ArtifactReconciliationError(f"{path} must be finite")
    return rendered


def _finite_number(value: Any, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ArtifactReconciliationError(f"{path} must be a finite number")
    return float(value)


def _scalar_string(value: Any, path: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"S", "U"}:
        raise ArtifactReconciliationError(f"{path} must be a saved string scalar")
    rendered = str(array.item())
    if not rendered:
        raise ArtifactReconciliationError(f"{path} must not be empty")
    return rendered


def _sha256_digest(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactReconciliationError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _configuration_hashes(value: Any, path: str = "corruption_manifest") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if key == "configuration_hash":
                found.append((nested_path, nested))
            else:
                found.extend(_configuration_hashes(nested, nested_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_configuration_hashes(nested, f"{path}[{index}]"))
    return found


def _parse_bool(value: str, path: str) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ArtifactReconciliationError(f"{path} is not a saved boolean: {value!r}")


def _assert_close(saved: Any, expected: float | None, path: str) -> None:
    if expected is None:
        if (
            not isinstance(saved, Mapping)
            or saved.get("status") != "not_applicable"
            or saved.get("value") is not None
            or not isinstance(saved.get("reason"), str)
            or not str(saved["reason"]).strip()
        ):
            raise ArtifactReconciliationError(f"{path} must be a documented not_applicable object")
        return
    if not isinstance(saved, (int, float)) or isinstance(saved, bool):
        raise ArtifactReconciliationError(f"{path} must be numeric")
    if not math.isclose(float(saved), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ArtifactReconciliationError(
            f"{path} disagrees with saved rankings: {saved!r} != {expected!r}"
        )


def _average_precision(flags: NDArray[np.bool_], scores: NDArray[np.float64]) -> float | None:
    positives = int(flags.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_flags = flags[order]
    cumulative = np.cumsum(sorted_flags)
    threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    previous_recall = 0.0
    area = 0.0
    for end in threshold_ends:
        true_positive = int(cumulative[end])
        recall = true_positive / positives
        precision = true_positive / (int(end) + 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def _binary_auroc(flags: NDArray[np.bool_], scores: NDArray[np.float64]) -> float | None:
    positives = int(flags.sum())
    negatives = len(flags) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[flags].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _budget_count(n_samples: int, fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ArtifactReconciliationError(f"review budget lies outside [0,1]: {fraction}")
    if not n_samples or not fraction:
        return 0
    return min(n_samples, math.ceil(n_samples * fraction))


def _reconcile_representation_example(run_path: Path, sample_ids: tuple[str, ...]) -> str:
    representation_path = run_path / "target_representation_example.npz"
    if not representation_path.is_file():
        raise ArtifactReconciliationError(
            "required artifact is missing: target_representation_example.npz"
        )
    required = {
        "sample_id",
        "target_instance_id",
        "full_patch",
        "full_target_mask",
        "source_bbox",
        "crop_source_box",
        "target_crop",
        "crop_target_mask",
        "highlighted_full_patch",
        "highlighted_crop",
    }
    with np.load(representation_path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ArtifactReconciliationError(
                f"target representation NPZ lacks arrays: {sorted(missing)}"
            )
        sample_array = np.asarray(payload["sample_id"])
        target_id_array = np.asarray(payload["target_instance_id"])
        full_patch = np.asarray(payload["full_patch"])
        full_mask = np.asarray(payload["full_target_mask"])
        source_bbox = _integer_array(payload["source_bbox"], "representation.source_bbox")
        crop_box = _integer_array(payload["crop_source_box"], "representation.crop_source_box")
        target_crop = np.asarray(payload["target_crop"])
        crop_mask = np.asarray(payload["crop_target_mask"])
        highlighted_full = np.asarray(payload["highlighted_full_patch"])
        highlighted_crop = np.asarray(payload["highlighted_crop"])
    if sample_array.shape != () or sample_array.dtype.kind not in {"S", "U"}:
        raise ArtifactReconciliationError("representation sample_id must be a saved string scalar")
    sample_id = str(sample_array.item())
    if sample_id not in set(sample_ids):
        raise ArtifactReconciliationError("representation sample_id is not in the audit NPZ")
    if (
        target_id_array.shape != ()
        or not np.issubdtype(target_id_array.dtype, np.integer)
        or np.issubdtype(target_id_array.dtype, np.bool_)
        or int(target_id_array.item()) <= 0
    ):
        raise ArtifactReconciliationError("representation target_instance_id is invalid")
    if full_patch.dtype != np.uint8 or full_patch.ndim != 3 or full_patch.shape[-1] != 3:
        raise ArtifactReconciliationError("representation full_patch must be uint8 RGB")
    if (
        full_mask.dtype != np.bool_
        or full_mask.shape != full_patch.shape[:2]
        or not full_mask.any()
    ):
        raise ArtifactReconciliationError("representation full target mask is invalid")
    if source_bbox.shape != (4,) or tuple(int(value) for value in source_bbox) != mask_bbox(
        full_mask
    ):
        raise ArtifactReconciliationError("representation source_bbox disagrees with target mask")
    if crop_box.shape != (4,):
        raise ArtifactReconciliationError("representation crop_source_box must have four values")
    crop_x0, crop_y0, crop_x1, crop_y1 = (int(value) for value in crop_box)
    height, width = full_mask.shape
    if not 0 <= crop_x0 < crop_x1 <= width or not 0 <= crop_y0 < crop_y1 <= height:
        raise ArtifactReconciliationError("representation crop_source_box lies outside full patch")
    x0, y0, x1, y1 = (int(value) for value in source_bbox)
    if not crop_x0 <= x0 < x1 <= crop_x1 or not crop_y0 <= y0 < y1 <= crop_y1:
        raise ArtifactReconciliationError("representation crop does not contain the target bbox")
    if (
        target_crop.dtype != np.uint8
        or target_crop.ndim != 3
        or target_crop.shape[-1] != 3
        or crop_mask.dtype != np.bool_
        or crop_mask.shape != target_crop.shape[:2]
        or not crop_mask.any()
    ):
        raise ArtifactReconciliationError("representation target crop/mask is invalid")
    if not np.array_equal(highlighted_full, highlight_target(full_patch, full_mask)):
        raise ArtifactReconciliationError("highlighted_full_patch is not reproducible")
    if not np.array_equal(highlighted_crop, highlight_target(target_crop, crop_mask)):
        raise ArtifactReconciliationError("highlighted_crop is not reproducible")
    return sample_id


def _validate_subgroups(
    entries_value: Any,
    *,
    path: str,
    flags: NDArray[np.bool_],
    scores: NDArray[np.float64],
    expected_labels: tuple[str, ...] | None,
) -> tuple[tuple[str, int, int, str], ...]:
    entries = _sequence(entries_value, path)
    if not entries:
        raise ArtifactReconciliationError(f"{path} must not be empty")
    seen: set[str] = set()
    signature: list[tuple[str, int, int, str]] = []
    for entry_value in entries:
        entry = _mapping(entry_value, f"{path} entry")
        subgroup = entry.get("subgroup")
        if not isinstance(subgroup, str) or not subgroup or subgroup in seen:
            raise ArtifactReconciliationError(f"{path} has empty/duplicate subgroup names")
        seen.add(subgroup)
        total = _integer(entry.get("total_examples"), f"{path}.{subgroup}.total", minimum=1)
        positives = _integer(
            entry.get("injected_corruptions"),
            f"{path}.{subgroup}.injected",
            minimum=0,
        )
        if positives > total:
            raise ArtifactReconciliationError(f"{path}.{subgroup} injected count exceeds total")
        status = entry.get("status")
        if status == "reported":
            saved_ap = entry.get("average_precision")
            if (
                not isinstance(saved_ap, (int, float))
                or isinstance(saved_ap, bool)
                or not 0.0 <= float(saved_ap) <= 1.0
                or entry.get("reason") is not None
            ):
                raise ArtifactReconciliationError(f"{path}.{subgroup} has invalid reported AP")
        elif status == "insufficient_support":
            if not isinstance(entry.get("reason"), str) or not str(entry["reason"]).strip():
                raise ArtifactReconciliationError(f"{path}.{subgroup} lacks support-gate reason")
            _assert_close(
                entry.get("average_precision"),
                None,
                f"{path}.{subgroup}.average_precision",
            )
        else:
            raise ArtifactReconciliationError(f"{path}.{subgroup} has invalid status")
        if expected_labels is not None:
            membership = np.asarray([label == subgroup for label in expected_labels], dtype=bool)
            if int(membership.sum()) != total or int(flags[membership].sum()) != positives:
                raise ArtifactReconciliationError(f"{path}.{subgroup} counts disagree with NPZ")
            if status == "reported":
                _assert_close(
                    entry.get("average_precision"),
                    _average_precision(flags[membership], scores[membership]),
                    f"{path}.{subgroup}.average_precision",
                )
        signature.append((subgroup, total, positives, str(status)))
    if sum(item[1] for item in signature) != len(flags) or sum(
        item[2] for item in signature
    ) != int(flags.sum()):
        raise ArtifactReconciliationError(f"{path} does not partition audit counts")
    if expected_labels is not None and seen != set(expected_labels):
        raise ArtifactReconciliationError(f"{path} subgroup names disagree with NPZ classes")
    return tuple(sorted(signature))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reconcile_synthetic_dataset_evidence(
    run_path: Path,
    *,
    metrics: Mapping[str, Any],
    class_names: tuple[str, ...],
    audit_sample_ids: tuple[str, ...],
    audit_group_ids: tuple[str, ...],
    audit_pre: NDArray[np.int64],
    audit_observed: NDArray[np.int64],
    audit_injected: NDArray[np.bool_],
    audit_fold_ids: NDArray[np.int64],
    audit_groups: set[str],
    reference_groups: set[str],
    final_groups: set[str],
) -> _SyntheticDatasetEvidence:
    dataset_path = run_path / "synthetic_dataset_evidence.npz"
    manifest_path = run_path / "synthetic_source_manifest.json"
    csv_path = run_path / "synthetic_source_manifest.csv"
    for path in (dataset_path, manifest_path, csv_path):
        if not path.is_file():
            raise ArtifactReconciliationError(f"required artifact is missing: {path.name}")
    required = {
        "schema_version",
        "record_index",
        "sample_ids",
        "group_ids",
        "patch_ids",
        "instance_id",
        "tissue_type",
        "official_fold",
        "split_partition",
        "oof_fold_id",
        "images",
        "target_masks",
        "audit_features",
        "corruption_features",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "original_class",
        "replacement_class",
        "corruption_type",
        "class_names",
        "dataset_configuration_hash",
        "corruption_configuration_hash",
    }
    arrays: dict[str, NDArray[np.generic]] = {}
    with np.load(dataset_path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ArtifactReconciliationError(
                f"synthetic_dataset_evidence.npz lacks arrays: {sorted(missing)}"
            )
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if _scalar_integer(arrays["schema_version"], "dataset.schema_version") != 1:
        raise ArtifactReconciliationError("unsupported synthetic dataset evidence schema")
    sample_ids = _strings(arrays["sample_ids"], "dataset.sample_ids")
    group_ids = _strings(arrays["group_ids"], "dataset.group_ids")
    patch_ids = _strings(arrays["patch_ids"], "dataset.patch_ids")
    tissue_types = _strings(arrays["tissue_type"], "dataset.tissue_type")
    partitions = _strings(arrays["split_partition"], "dataset.split_partition")
    record_index = _integer_array(arrays["record_index"], "dataset.record_index")
    instance_ids = _integer_array(arrays["instance_id"], "dataset.instance_id")
    official_fold = _integer_array(arrays["official_fold"], "dataset.official_fold")
    oof_fold = _integer_array(arrays["oof_fold_id"], "dataset.oof_fold_id")
    pre = _integer_array(arrays["pre_corruption_label"], "dataset.pre_corruption_label")
    observed = _integer_array(arrays["observed_label"], "dataset.observed_label")
    injected = _boolean_array(arrays["is_injected_corruption"], "dataset.is_injected_corruption")
    original = _integer_array(arrays["original_class"], "dataset.original_class")
    replacement = _integer_array(arrays["replacement_class"], "dataset.replacement_class")
    corruption_types = _strings(arrays["corruption_type"], "dataset.corruption_type")
    saved_class_names = _strings(arrays["class_names"], "dataset.class_names")
    dataset_configuration_hash = _sha256_digest(
        _scalar_string(arrays["dataset_configuration_hash"], "dataset.configuration_hash"),
        "dataset.configuration_hash",
    )
    corruption_configuration_hash = _sha256_digest(
        _scalar_string(
            arrays["corruption_configuration_hash"],
            "dataset.corruption_configuration_hash",
        ),
        "dataset.corruption_configuration_hash",
    )
    images = arrays["images"]
    masks = arrays["target_masks"]
    audit_features = np.asarray(arrays["audit_features"], dtype=np.float64)
    corruption_features = np.asarray(arrays["corruption_features"], dtype=np.float64)
    n_samples = len(sample_ids)
    sample_counts = _mapping(metrics.get("sample_counts"), "metrics.sample_counts")
    if (
        n_samples == 0
        or len(set(sample_ids)) != n_samples
        or _integer(sample_counts.get("total"), "metrics.sample_counts.total", minimum=1)
        != n_samples
    ):
        raise ArtifactReconciliationError("dataset evidence total/sample identities are invalid")
    for name, string_values in {
        "group_ids": group_ids,
        "patch_ids": patch_ids,
        "tissue_type": tissue_types,
        "split_partition": partitions,
    }.items():
        if len(string_values) != n_samples:
            raise ArtifactReconciliationError(f"dataset.{name} does not align with samples")
    for name, array_values in {
        "record_index": record_index,
        "instance_id": instance_ids,
        "official_fold": official_fold,
        "oof_fold_id": oof_fold,
        "pre_corruption_label": pre,
        "observed_label": observed,
        "is_injected_corruption": injected,
        "original_class": original,
        "replacement_class": replacement,
    }.items():
        if array_values.shape != (n_samples,):
            raise ArtifactReconciliationError(f"dataset.{name} does not align with samples")
    if not np.array_equal(record_index, np.arange(n_samples)) or np.any(instance_ids <= 0):
        raise ArtifactReconciliationError("dataset record/instance indices are invalid")
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ArtifactReconciliationError("dataset images must be full uint8 RGB arrays")
    if (
        masks.dtype != np.bool_
        or masks.shape != images.shape[:3]
        or masks.shape[0] != n_samples
        or not np.all(masks.reshape(n_samples, -1).any(axis=1))
    ):
        raise ArtifactReconciliationError("dataset target masks are invalid or incomplete")
    if (
        audit_features.ndim != 2
        or corruption_features.ndim != 2
        or audit_features.shape[0] != n_samples
        or corruption_features.shape[0] != n_samples
        or not np.isfinite(audit_features).all()
        or not np.isfinite(corruption_features).all()
    ):
        raise ArtifactReconciliationError("dataset feature matrices are invalid or incomplete")
    if saved_class_names != class_names:
        raise ArtifactReconciliationError("dataset class names disagree with report inputs")
    valid_classes = set(range(len(class_names)))
    if any(int(value) not in valid_classes for value in np.r_[pre, observed, original]):
        raise ArtifactReconciliationError("dataset labels lie outside the saved class order")
    if not np.array_equal(pre != observed, injected) or not np.array_equal(original, pre):
        raise ArtifactReconciliationError("dataset corruption labels/flags are inconsistent")
    if (
        np.any(replacement[injected] != observed[injected])
        or np.any(replacement[~injected] != -1)
        or any(
            value == "none" for value, flag in zip(corruption_types, injected, strict=True) if flag
        )
        or any(
            value != "none"
            for value, flag in zip(corruption_types, injected, strict=True)
            if not flag
        )
    ):
        raise ArtifactReconciliationError("dataset corruption metadata is inconsistent")
    corruption_metrics = _mapping(metrics.get("corruption"), "metrics.corruption")
    if corruption_configuration_hash != _sha256_digest(
        corruption_metrics.get("configuration_hash"),
        "metrics.corruption.configuration_hash",
    ):
        raise ArtifactReconciliationError("dataset corruption hash disagrees with metrics")
    partition_names = {"audit_pool", "reference_validation", "final_reference_test"}
    if set(partitions) != partition_names:
        raise ArtifactReconciliationError("dataset split partitions are incomplete or unknown")
    partition_indices = {
        name: np.flatnonzero(np.asarray(partitions) == name) for name in partition_names
    }
    expected_partition_counts = {
        "audit_pool": _integer(
            sample_counts.get("audit_pool"), "metrics.sample_counts.audit_pool", minimum=0
        ),
        "reference_validation": _integer(
            sample_counts.get("reference_validation"),
            "metrics.sample_counts.reference_validation",
            minimum=0,
        ),
        "final_reference_test": _integer(
            sample_counts.get("final_reference_test"),
            "metrics.sample_counts.final_reference_test",
            minimum=0,
        ),
    }
    if any(
        len(partition_indices[name]) != expected_partition_counts[name] for name in partition_names
    ):
        raise ArtifactReconciliationError("dataset partition counts disagree with metrics")
    expected_group_sets = {
        "audit_pool": audit_groups,
        "reference_validation": reference_groups,
        "final_reference_test": final_groups,
    }
    group_partitions: dict[str, set[str]] = {}
    for group, partition in zip(group_ids, partitions, strict=True):
        group_partitions.setdefault(group, set()).add(partition)
    if any(len(values) != 1 for values in group_partitions.values()):
        raise ArtifactReconciliationError("a dataset source group spans outer partitions")
    for name, indices in partition_indices.items():
        if {group_ids[int(index)] for index in indices} != expected_group_sets[name]:
            raise ArtifactReconciliationError(f"dataset {name} groups disagree with split evidence")
    audit_indices = partition_indices["audit_pool"]
    if (
        tuple(sample_ids[int(index)] for index in audit_indices) != audit_sample_ids
        or tuple(group_ids[int(index)] for index in audit_indices) != audit_group_ids
    ):
        raise ArtifactReconciliationError("dataset audit partition order disagrees with OOF")
    if not (
        np.array_equal(pre[audit_indices], audit_pre)
        and np.array_equal(observed[audit_indices], audit_observed)
        and np.array_equal(injected[audit_indices], audit_injected)
        and np.array_equal(oof_fold[audit_indices], audit_fold_ids)
    ):
        raise ArtifactReconciliationError("dataset audit arrays disagree with OOF evidence")
    non_audit = np.asarray(partitions) != "audit_pool"
    if (
        np.any(oof_fold[non_audit] != -1)
        or injected[non_audit].any()
        or not np.array_equal(observed[non_audit], pre[non_audit])
    ):
        raise ArtifactReconciliationError(
            "reference-validation/final-reference dataset evidence is not untouched"
        )
    final_indices = partition_indices["final_reference_test"]

    manifest = _load_object(manifest_path)
    if _integer(manifest.get("schema_version"), "source_manifest.schema_version") != 1:
        raise ArtifactReconciliationError("unsupported synthetic source manifest schema")
    if (
        _integer(manifest.get("record_count"), "source_manifest.record_count", minimum=1)
        != n_samples
    ):
        raise ArtifactReconciliationError("source manifest record count disagrees with dataset")
    if (
        tuple(str(value) for value in _sequence(manifest.get("class_names"), "manifest classes"))
        != class_names
    ):
        raise ArtifactReconciliationError("source manifest class names disagree with dataset")
    dataset_metadata = _mapping(
        manifest.get("dataset_evidence"), "source_manifest.dataset_evidence"
    )
    if dataset_metadata.get("file") != dataset_path.name or _sha256_digest(
        dataset_metadata.get("sha256"), "source_manifest.dataset_evidence.sha256"
    ) != _file_sha256(dataset_path):
        raise ArtifactReconciliationError("source manifest dataset checksum is invalid")
    array_metadata = _mapping(dataset_metadata.get("arrays"), "source_manifest.arrays")
    for name, array in arrays.items():
        metadata = _mapping(array_metadata.get(name), f"source_manifest.arrays.{name}")
        if metadata.get("dtype") != str(array.dtype) or list(
            _sequence(metadata.get("shape"), f"source_manifest.arrays.{name}.shape")
        ) != list(array.shape):
            raise ArtifactReconciliationError(
                f"source manifest array metadata disagrees for {name}"
            )
    csv_metadata = _mapping(manifest.get("csv_manifest"), "source_manifest.csv_manifest")
    if csv_metadata.get("file") != csv_path.name or _sha256_digest(
        csv_metadata.get("sha256"), "source_manifest.csv.sha256"
    ) != _file_sha256(csv_path):
        raise ArtifactReconciliationError("source manifest CSV checksum is invalid")
    records = _sequence(manifest.get("records"), "source_manifest.records")
    if len(records) != n_samples or _sha256_digest(
        manifest.get("records_sha256"), "source_manifest.records_sha256"
    ) != canonical_sha256(records):
        raise ArtifactReconciliationError("source manifest records hash/count is invalid")
    nucleus_fields = {field.name for field in fields(NucleusRecord)}
    augmentation_fields = {
        "record_index",
        "split_partition",
        "oof_fold_id",
        "dataset_configuration_hash",
        "corruption_configuration_hash",
        "corruption_upstream_manifest_hash",
        "corruption_independence_status",
    }
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"source_manifest.records[{index}]")
        if not nucleus_fields.union(augmentation_fields).issubset(record):
            raise ArtifactReconciliationError("source manifest omits complete record fields")
        if (
            _integer(record.get("record_index"), "source record index", minimum=0) != index
            or str(record.get("sample_id")) != sample_ids[index]
            or str(record.get("group_id")) != group_ids[index]
            or str(record.get("patch_id")) != patch_ids[index]
            or str(record.get("tissue_type")) != tissue_types[index]
            or str(record.get("split_partition")) != partitions[index]
            or _integer(record.get("pre_corruption_label"), "source record pre label")
            != int(pre[index])
            or _integer(record.get("observed_label"), "source record observed label")
            != int(observed[index])
            or record.get("is_injected_corruption") is not bool(injected[index])
            or record.get("dataset_configuration_hash") != dataset_configuration_hash
        ):
            raise ArtifactReconciliationError(f"source manifest row disagrees at index {index}")
        expected_oof = int(oof_fold[index]) if int(oof_fold[index]) >= 0 else None
        if record.get("oof_fold_id") != expected_oof:
            raise ArtifactReconciliationError("source manifest OOF assignment disagrees with NPZ")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    columns = tuple(
        str(value) for value in _sequence(csv_metadata.get("columns"), "manifest CSV columns")
    )
    if len(csv_rows) != n_samples or not csv_rows or tuple(csv_rows[0]) != columns:
        raise ArtifactReconciliationError("source manifest CSV columns/count are invalid")
    for index, row in enumerate(csv_rows):
        if (
            int(row["record_index"]) != index
            or row["sample_id"] != sample_ids[index]
            or row["group_id"] != group_ids[index]
            or row["split_partition"] != partitions[index]
            or int(row["pre_corruption_label"]) != int(pre[index])
            or int(row["observed_label"]) != int(observed[index])
            or _parse_bool(row["is_injected_corruption"], "source manifest CSV flag")
            != bool(injected[index])
        ):
            raise ArtifactReconciliationError(f"source manifest CSV row disagrees at index {index}")
    return _SyntheticDatasetEvidence(
        summary={
            "dataset_evidence_sample_count": n_samples,
            "dataset_evidence_final_reference_count": len(final_indices),
            "source_manifest_record_count": len(records),
        },
        final_sample_ids=tuple(sample_ids[int(index)] for index in final_indices),
        final_group_ids=tuple(group_ids[int(index)] for index in final_indices),
        final_reference_labels=pre[final_indices],
        final_is_injected_corruption=injected[final_indices],
    )


def _reconcile_neighbour_evidence(
    run_path: Path,
    *,
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    observed: NDArray[np.int64],
    fold_ids: NDArray[np.int64],
    class_order: NDArray[np.int64],
    training_groups_by_fold: Mapping[int, set[str]],
) -> NDArray[np.float64]:
    evidence_path = run_path / "neighbour_evidence.npz"
    if not evidence_path.is_file():
        raise ArtifactReconciliationError("required artifact is missing: neighbour_evidence.npz")
    required = {
        "sample_ids",
        "group_ids",
        "risk_scores",
        "alternative_class_support",
        "suggested_class",
        "neighbour_count",
        "neighbour_ids",
        "neighbour_groups",
        "neighbour_distances",
        "k",
        "metric",
    }
    with np.load(evidence_path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ArtifactReconciliationError(
                f"neighbour_evidence.npz lacks arrays: {sorted(missing)}"
            )
        saved_sample_ids = _strings(np.asarray(payload["sample_ids"]), "neighbour.sample_ids")
        saved_group_ids = _strings(np.asarray(payload["group_ids"]), "neighbour.group_ids")
        risk_scores = np.asarray(payload["risk_scores"], dtype=np.float64)
        alternative_support = np.asarray(payload["alternative_class_support"], dtype=np.float64)
        suggested_class = _integer_array(payload["suggested_class"], "neighbour.suggested_class")
        neighbour_count = _integer_array(payload["neighbour_count"], "neighbour.neighbour_count")
        neighbour_ids = _string_array(payload["neighbour_ids"], "neighbour.neighbour_ids")
        neighbour_groups = _string_array(payload["neighbour_groups"], "neighbour.neighbour_groups")
        neighbour_distances = np.asarray(payload["neighbour_distances"], dtype=np.float64)
        k = _scalar_integer(payload["k"], "neighbour.k", minimum=1)
        metric = _scalar_string(payload["metric"], "neighbour.metric")
    n_samples = len(sample_ids)
    if saved_sample_ids != sample_ids or saved_group_ids != group_ids:
        raise ArtifactReconciliationError(
            "neighbour evidence sample/group order disagrees with OOF predictions"
        )
    for name, array in {
        "risk_scores": risk_scores,
        "alternative_class_support": alternative_support,
        "suggested_class": suggested_class,
        "neighbour_count": neighbour_count,
    }.items():
        if array.shape != (n_samples,):
            raise ArtifactReconciliationError(f"neighbour.{name} does not align with samples")
    if (
        neighbour_ids.ndim != 2
        or neighbour_groups.shape != neighbour_ids.shape
        or neighbour_distances.shape != neighbour_ids.shape
    ):
        raise ArtifactReconciliationError("neighbour evidence matrices must share a 2-D shape")
    width = neighbour_ids.shape[1]
    if neighbour_ids.shape[0] != n_samples or width <= 0 or width > k:
        raise ArtifactReconciliationError("neighbour evidence width is invalid for saved k")
    if np.any(neighbour_count <= 0) or np.any(neighbour_count > k):
        raise ArtifactReconciliationError("neighbour_count lies outside [1, k]")
    if int(neighbour_count.max()) != width:
        raise ArtifactReconciliationError("neighbour evidence padding width is not canonical")
    if metric not in {"euclidean", "cosine"}:
        raise ArtifactReconciliationError("neighbour metric is unsupported")
    if (
        not np.isfinite(risk_scores).all()
        or not np.isfinite(alternative_support).all()
        or np.any(risk_scores < 0.0)
        or np.any(risk_scores > 1.0)
        or np.any(alternative_support < 0.0)
        or np.any(alternative_support > 1.0)
    ):
        raise ArtifactReconciliationError("neighbour scores/support are invalid")
    classes = tuple(int(value) for value in class_order)
    if len(classes) < 2:
        raise ArtifactReconciliationError("neighbour evidence requires at least two classes")
    class_lookup = {value: index for index, value in enumerate(classes)}
    valid_classes = set(classes)
    if any(int(value) not in valid_classes for value in suggested_class):
        raise ArtifactReconciliationError("neighbour suggested_class lies outside class_order")
    sample_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    recomputed_risk = np.empty(n_samples, dtype=np.float64)
    recomputed_alternative = np.empty(n_samples, dtype=np.float64)
    recomputed_suggested = np.empty(n_samples, dtype=np.int64)
    for query_index in range(n_samples):
        count = int(neighbour_count[query_index])
        active_ids = tuple(str(value) for value in neighbour_ids[query_index, :count])
        active_groups = tuple(str(value) for value in neighbour_groups[query_index, :count])
        active_distances = neighbour_distances[query_index, :count]
        if (
            any(not value for value in active_ids + active_groups)
            or len(set(active_ids)) != count
            or not np.isfinite(active_distances).all()
            or np.any(active_distances < 0.0)
        ):
            raise ArtifactReconciliationError(
                f"neighbour evidence contains invalid active rows for {sample_ids[query_index]}"
            )
        if any(value not in sample_to_index for value in active_ids):
            raise ArtifactReconciliationError("neighbour evidence references an unknown sample")
        if sample_ids[query_index] in active_ids or group_ids[query_index] in active_groups:
            raise ArtifactReconciliationError(
                f"neighbour evidence includes query/self group: {sample_ids[query_index]}"
            )
        fold = int(fold_ids[query_index])
        allowed_groups = training_groups_by_fold.get(fold)
        if allowed_groups is None or any(group not in allowed_groups for group in active_groups):
            raise ArtifactReconciliationError(
                f"neighbour evidence is not fold-safe for {sample_ids[query_index]}"
            )
        for neighbour_id, neighbour_group in zip(active_ids, active_groups, strict=True):
            if group_ids[sample_to_index[neighbour_id]] != neighbour_group:
                raise ArtifactReconciliationError("neighbour ID/group provenance mismatch")
        if (
            np.any(neighbour_ids[query_index, count:] != "")
            or np.any(neighbour_groups[query_index, count:] != "")
            or not np.isnan(neighbour_distances[query_index, count:]).all()
        ):
            raise ArtifactReconciliationError("neighbour evidence uses non-canonical padding")
        weights = 1.0 / np.maximum(active_distances, 1.0e-8)
        support = np.zeros(len(classes), dtype=np.float64)
        for neighbour_id, weight in zip(active_ids, weights, strict=True):
            label = int(observed[sample_to_index[neighbour_id]])
            support[class_lookup[label]] += float(weight)
        support /= support.sum()
        observed_column = class_lookup[int(observed[query_index])]
        recomputed_risk[query_index] = 1.0 - support[observed_column]
        alternatives = support.copy()
        alternatives[observed_column] = -np.inf
        recomputed_alternative[query_index] = float(alternatives.max())
        recomputed_suggested[query_index] = classes[int(np.argmax(support))]
    if not np.allclose(risk_scores, recomputed_risk, rtol=1e-10, atol=1e-12):
        raise ArtifactReconciliationError("saved neighbour risk scores are not reconstructible")
    if not np.allclose(alternative_support, recomputed_alternative, rtol=1e-10, atol=1e-12):
        raise ArtifactReconciliationError(
            "saved neighbour alternative support is not reconstructible"
        )
    if not np.array_equal(suggested_class, recomputed_suggested):
        raise ArtifactReconciliationError("saved neighbour suggested_class is not reconstructible")
    return risk_scores


def _validate_probabilities(
    probabilities: NDArray[np.float64],
    predicted: NDArray[np.int64],
    *,
    expected_shape: tuple[int, ...],
    class_order: NDArray[np.int64],
    path: str,
) -> None:
    if probabilities.shape != (*expected_shape, len(class_order)):
        raise ArtifactReconciliationError(f"{path} probability shape is invalid")
    if predicted.shape != expected_shape:
        raise ArtifactReconciliationError(f"{path} predicted-class shape is invalid")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-10, atol=1e-8)
    ):
        raise ArtifactReconciliationError(f"{path} probabilities are invalid")
    expected = class_order[np.argmax(probabilities, axis=-1)]
    if not np.array_equal(predicted, expected):
        raise ArtifactReconciliationError(f"{path} predicted classes disagree with argmax")


def _reconcile_restoration_evidence(
    run_path: Path,
    *,
    metrics: Mapping[str, Any],
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    pre: NDArray[np.int64],
    observed: NDArray[np.int64],
    injected: NDArray[np.bool_],
    class_order: NDArray[np.int64],
    final_groups: set[str],
    final_count: int,
    expected_final_sample_ids: tuple[str, ...],
    expected_final_group_ids: tuple[str, ...],
    expected_final_reference_labels: NDArray[np.int64],
    expected_final_is_injected_corruption: NDArray[np.bool_],
) -> dict[str, int]:
    path = run_path / "restoration_evidence.npz"
    if not path.is_file():
        raise ArtifactReconciliationError("required artifact is missing: restoration_evidence.npz")
    condition_names = (
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "audit_guided_restoration",
    )
    required = {
        "development_sample_ids",
        "development_group_ids",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "final_test_sample_ids",
        "final_test_group_ids",
        "final_test_reference_label",
        "final_test_is_injected_corruption",
        "class_order",
        "review_budget_fraction",
        "review_budget_count",
        "audit_guided_reviewed_indices",
        "audit_guided_reviewed_sample_ids",
        "audit_guided_restored_label",
        "audit_guided_restored_mask",
        "random_review_seeds",
        "random_reviewed_indices",
        "random_reviewed_sample_ids",
        "random_restored_label",
        "random_restored_mask",
        "random_review_restoration_final_test_probabilities",
        "random_review_restoration_final_test_predicted_class",
    }
    for name in condition_names:
        required.add(f"{name}_final_test_probabilities")
        required.add(f"{name}_final_test_predicted_class")
    with np.load(path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ArtifactReconciliationError(
                f"restoration_evidence.npz lacks arrays: {sorted(missing)}"
            )
        development_sample_ids = _strings(
            np.asarray(payload["development_sample_ids"]), "restoration.development_sample_ids"
        )
        development_group_ids = _strings(
            np.asarray(payload["development_group_ids"]), "restoration.development_group_ids"
        )
        saved_pre = _integer_array(payload["pre_corruption_label"], "restoration.pre")
        saved_observed = _integer_array(payload["observed_label"], "restoration.observed")
        saved_injected = _boolean_array(
            payload["is_injected_corruption"], "restoration.is_injected_corruption"
        )
        final_sample_ids = _strings(
            np.asarray(payload["final_test_sample_ids"]), "restoration.final_test_sample_ids"
        )
        final_group_ids = _strings(
            np.asarray(payload["final_test_group_ids"]), "restoration.final_test_group_ids"
        )
        final_reference = _integer_array(
            payload["final_test_reference_label"], "restoration.final_test_reference_label"
        )
        final_injected = _boolean_array(
            payload["final_test_is_injected_corruption"],
            "restoration.final_test_is_injected_corruption",
        )
        saved_class_order = _integer_array(payload["class_order"], "restoration.class_order")
        budget_fraction = _scalar_float(
            payload["review_budget_fraction"], "restoration.review_budget_fraction"
        )
        budget_count = _scalar_integer(
            payload["review_budget_count"], "restoration.review_budget_count", minimum=0
        )
        guided_indices = _integer_array(
            payload["audit_guided_reviewed_indices"], "restoration.guided_indices"
        )
        guided_ids = _strings(
            np.asarray(payload["audit_guided_reviewed_sample_ids"]),
            "restoration.guided_sample_ids",
        )
        guided_labels = _integer_array(
            payload["audit_guided_restored_label"], "restoration.guided_restored_label"
        )
        guided_mask = _boolean_array(
            payload["audit_guided_restored_mask"], "restoration.guided_restored_mask"
        )
        random_seeds = _integer_array(payload["random_review_seeds"], "restoration.random_seeds")
        random_indices = _integer_array(
            payload["random_reviewed_indices"], "restoration.random_reviewed_indices"
        )
        random_ids = _string_array(
            payload["random_reviewed_sample_ids"], "restoration.random_reviewed_sample_ids"
        )
        random_labels = _integer_array(
            payload["random_restored_label"], "restoration.random_restored_label"
        )
        random_masks = _boolean_array(
            payload["random_restored_mask"], "restoration.random_restored_mask"
        )
        condition_probabilities = {
            name: np.asarray(payload[f"{name}_final_test_probabilities"], dtype=np.float64)
            for name in condition_names
        }
        condition_predicted = {
            name: _integer_array(
                payload[f"{name}_final_test_predicted_class"],
                f"restoration.{name}.predicted_class",
            )
            for name in condition_names
        }
        random_probabilities = np.asarray(
            payload["random_review_restoration_final_test_probabilities"], dtype=np.float64
        )
        random_predicted = _integer_array(
            payload["random_review_restoration_final_test_predicted_class"],
            "restoration.random.predicted_class",
        )
    n_samples = len(sample_ids)
    if development_sample_ids != sample_ids or development_group_ids != group_ids:
        raise ArtifactReconciliationError("restoration development identities disagree with OOF")
    if not (
        np.array_equal(saved_pre, pre)
        and np.array_equal(saved_observed, observed)
        and np.array_equal(saved_injected, injected)
    ):
        raise ArtifactReconciliationError("restoration development labels disagree with OOF")
    if not np.array_equal(saved_class_order, class_order):
        raise ArtifactReconciliationError("restoration class_order disagrees with OOF")
    if (
        len(final_sample_ids) != final_count
        or len(set(final_sample_ids)) != final_count
        or set(final_sample_ids) & set(sample_ids)
        or len(final_group_ids) != final_count
        or set(final_group_ids) != final_groups
        or final_reference.shape != (final_count,)
        or final_injected.shape != (final_count,)
        or final_injected.any()
    ):
        raise ArtifactReconciliationError("restoration final-reference partition is invalid")
    if (
        final_sample_ids != expected_final_sample_ids
        or final_group_ids != expected_final_group_ids
        or not np.array_equal(final_reference, expected_final_reference_labels)
        or not np.array_equal(final_injected, expected_final_is_injected_corruption)
    ):
        raise ArtifactReconciliationError(
            "restoration final-reference arrays disagree with full dataset evidence"
        )
    valid_classes = set(int(value) for value in class_order)
    if any(int(value) not in valid_classes for value in final_reference):
        raise ArtifactReconciliationError(
            "restoration final-reference label is outside class_order"
        )
    downstream = _mapping(metrics.get("downstream_restoration"), "metrics.downstream_restoration")
    saved_fraction = downstream.get("review_budget_fraction")
    if (
        not isinstance(saved_fraction, (int, float))
        or isinstance(saved_fraction, bool)
        or not math.isclose(float(saved_fraction), budget_fraction, abs_tol=1e-12)
    ):
        raise ArtifactReconciliationError("restoration budget fraction disagrees with metrics")
    if _integer(
        downstream.get("review_budget_count"),
        "metrics.downstream_restoration.review_budget_count",
        minimum=0,
    ) != budget_count or budget_count != _budget_count(n_samples, budget_fraction):
        raise ArtifactReconciliationError("restoration review budget count is inconsistent")
    if (
        guided_indices.shape != (budget_count,)
        or len(set(int(value) for value in guided_indices)) != budget_count
        or np.any(guided_indices < 0)
        or np.any(guided_indices >= n_samples)
        or guided_ids != tuple(sample_ids[int(index)] for index in guided_indices)
        or guided_labels.shape != (n_samples,)
        or guided_mask.shape != (n_samples,)
    ):
        raise ArtifactReconciliationError("guided restoration selection is invalid")
    expected_guided_mask = np.zeros(n_samples, dtype=bool)
    expected_guided_mask[guided_indices] = injected[guided_indices]
    expected_guided_labels = observed.copy()
    expected_guided_labels[expected_guided_mask] = pre[expected_guided_mask]
    if not np.array_equal(guided_mask, expected_guided_mask) or not np.array_equal(
        guided_labels, expected_guided_labels
    ):
        raise ArtifactReconciliationError("guided restored labels/mask are not reconstructible")
    guided_metrics = _mapping(
        downstream.get("audit_guided_restoration"),
        "metrics.downstream_restoration.audit_guided_restoration",
    )
    if _integer(
        guided_metrics.get("reviewed_count"), "guided.reviewed_count", minimum=0
    ) != budget_count or _integer(
        guided_metrics.get("restored_count"), "guided.restored_count", minimum=0
    ) != int(guided_mask.sum()):
        raise ArtifactReconciliationError("guided restoration counts disagree with metrics")
    random_metrics = _mapping(
        downstream.get("random_review_restoration"),
        "metrics.downstream_restoration.random_review_restoration",
    )
    random_runs = _sequence(random_metrics.get("runs"), "random restoration runs")
    repeats = len(random_seeds)
    if not repeats or len(random_runs) != repeats:
        raise ArtifactReconciliationError("random restoration repeat count disagrees with metrics")
    if (
        random_indices.shape != (repeats, budget_count)
        or random_ids.shape != (repeats, budget_count)
        or random_labels.shape != (repeats, n_samples)
        or random_masks.shape != (repeats, n_samples)
    ):
        raise ArtifactReconciliationError("random restoration arrays have invalid shapes")
    for repeat in range(repeats):
        indices = random_indices[repeat]
        if (
            len(set(int(value) for value in indices)) != budget_count
            or np.any(indices < 0)
            or np.any(indices >= n_samples)
            or tuple(str(value) for value in random_ids[repeat])
            != tuple(sample_ids[int(index)] for index in indices)
        ):
            raise ArtifactReconciliationError(f"random restoration selection {repeat} is invalid")
        expected_mask = np.zeros(n_samples, dtype=bool)
        expected_mask[indices] = injected[indices]
        expected_labels = observed.copy()
        expected_labels[expected_mask] = pre[expected_mask]
        if not np.array_equal(random_masks[repeat], expected_mask) or not np.array_equal(
            random_labels[repeat], expected_labels
        ):
            raise ArtifactReconciliationError(
                f"random restored labels/mask {repeat} are not reconstructible"
            )
        run = _mapping(random_runs[repeat], f"random restoration run {repeat}")
        if (
            _integer(run.get("review_seed"), f"random run {repeat}.review_seed")
            != int(random_seeds[repeat])
            or _integer(run.get("reviewed_count"), f"random run {repeat}.reviewed_count", minimum=0)
            != budget_count
            or _integer(run.get("restored_count"), f"random run {repeat}.restored_count", minimum=0)
            != int(expected_mask.sum())
        ):
            raise ArtifactReconciliationError(
                f"random restoration counts/seeds disagree at repeat {repeat}"
            )
    for name in condition_names:
        _validate_probabilities(
            condition_probabilities[name],
            condition_predicted[name],
            expected_shape=(final_count,),
            class_order=class_order,
            path=f"restoration.{name}",
        )
    _validate_probabilities(
        random_probabilities,
        random_predicted,
        expected_shape=(repeats, final_count),
        class_order=class_order,
        path="restoration.random_review_restoration",
    )
    return {
        "restoration_random_repeat_count": repeats,
        "restoration_review_budget_count": budget_count,
        "restoration_final_reference_count": final_count,
    }


def _reconcile_bootstrap_evidence(
    run_path: Path,
    *,
    metrics: Mapping[str, Any],
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    injected: NDArray[np.bool_],
    scores_by_method: Mapping[str, NDArray[np.float64]],
) -> dict[str, Any]:
    """Verify saved shared group draws and every paired method comparison from source arrays."""

    path = run_path / "bootstrap_evidence.npz"
    if not path.is_file():
        raise ArtifactReconciliationError("required artifact is missing: bootstrap_evidence.npz")
    required = {
        "schema_version",
        "status",
        "reason",
        "metric_name",
        "comparator_method",
        "comparison_methods",
        "difference_direction",
        "pairing_unit",
        "requested_iterations",
        "valid_iterations",
        "bootstrap_seed",
        "sample_ids",
        "group_ids",
        "draw_indices",
        "draw_offsets",
        "valid_draw_indices",
        "metric_a",
        "metric_b",
        "differences",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ArtifactReconciliationError(
                f"bootstrap_evidence.npz lacks arrays: {sorted(missing)}"
            )
        schema_version = _scalar_integer(payload["schema_version"], "bootstrap.schema_version")
        status = _scalar_string(payload["status"], "bootstrap.status")
        reason_array = np.asarray(payload["reason"])
        if reason_array.shape != () or reason_array.dtype.kind not in {"S", "U"}:
            raise ArtifactReconciliationError("bootstrap.reason must be a saved string scalar")
        reason = str(reason_array.item())
        metric_name = _scalar_string(payload["metric_name"], "bootstrap.metric_name")
        comparator = _scalar_string(payload["comparator_method"], "bootstrap.comparator_method")
        methods = _strings(
            np.asarray(payload["comparison_methods"]), "bootstrap.comparison_methods"
        )
        direction = _scalar_string(
            payload["difference_direction"], "bootstrap.difference_direction"
        )
        pairing_unit = _scalar_string(payload["pairing_unit"], "bootstrap.pairing_unit")
        requested = _scalar_integer(
            payload["requested_iterations"], "bootstrap.requested_iterations", minimum=1
        )
        valid = _scalar_integer(
            payload["valid_iterations"], "bootstrap.valid_iterations", minimum=0
        )
        seed = _scalar_integer(payload["bootstrap_seed"], "bootstrap.bootstrap_seed", minimum=0)
        saved_sample_ids = _strings(np.asarray(payload["sample_ids"]), "bootstrap.sample_ids")
        saved_group_ids = _strings(np.asarray(payload["group_ids"]), "bootstrap.group_ids")
        draw_indices = _integer_array(payload["draw_indices"], "bootstrap.draw_indices")
        draw_offsets = _integer_array(payload["draw_offsets"], "bootstrap.draw_offsets")
        valid_draw_indices = _integer_array(
            payload["valid_draw_indices"], "bootstrap.valid_draw_indices"
        )
        metric_a = np.asarray(payload["metric_a"], dtype=np.float64)
        metric_b = np.asarray(payload["metric_b"], dtype=np.float64)
        differences = np.asarray(payload["differences"], dtype=np.float64)

    if schema_version != 1:
        raise ArtifactReconciliationError("unsupported bootstrap evidence schema version")
    if tuple(saved_sample_ids) != sample_ids or tuple(saved_group_ids) != group_ids:
        raise ArtifactReconciliationError(
            "bootstrap sample/group order disagrees with OOF evidence"
        )
    method_names = tuple(methods)
    if len(set(method_names)) != len(method_names):
        raise ArtifactReconciliationError("bootstrap comparison methods must be unique")
    if metric_name != "average_precision":
        raise ArtifactReconciliationError("bootstrap metric must be average_precision")
    if direction != "selected_method_minus_comparator" or pairing_unit != "source_group":
        raise ArtifactReconciliationError("bootstrap direction/pairing unit is not predeclared")
    if comparator not in scores_by_method or any(
        method not in scores_by_method for method in method_names
    ):
        raise ArtifactReconciliationError("bootstrap references an unknown ranking method")
    expected_shape = (len(method_names), valid)
    if metric_a.shape != expected_shape or metric_b.shape != expected_shape:
        raise ArtifactReconciliationError("bootstrap metric arrays have invalid shape")
    if differences.shape != expected_shape:
        raise ArtifactReconciliationError("bootstrap differences have invalid shape")
    if (
        not np.isfinite(metric_a).all()
        or not np.isfinite(metric_b).all()
        or not np.isfinite(differences).all()
        or not np.allclose(differences, metric_a - metric_b, rtol=1e-12, atol=1e-14)
    ):
        raise ArtifactReconciliationError("bootstrap numeric arrays are invalid or inconsistent")

    paired_metrics = _mapping(
        metrics.get("paired_method_differences"), "metrics.paired_method_differences"
    )
    if (
        paired_metrics.get("metric") != metric_name
        or paired_metrics.get("comparator") != comparator
        or paired_metrics.get("difference_direction") != direction
        or paired_metrics.get("pairing_unit") != pairing_unit
        or _integer(
            paired_metrics.get("iterations"), "paired_method_differences.iterations", minimum=1
        )
        != requested
        or _integer(
            paired_metrics.get("valid_iterations"),
            "paired_method_differences.valid_iterations",
            minimum=0,
        )
        != valid
        or _integer(
            paired_metrics.get("bootstrap_seed"),
            "paired_method_differences.bootstrap_seed",
            minimum=0,
        )
        != seed
    ):
        raise ArtifactReconciliationError("paired method metrics disagree with bootstrap metadata")
    comparison_order = tuple(
        str(value)
        for value in _sequence(
            paired_metrics.get("comparison_order"), "paired_method_differences.comparison_order"
        )
    )
    if comparison_order != method_names:
        raise ArtifactReconciliationError("paired method order disagrees with bootstrap evidence")
    comparisons = _mapping(
        paired_metrics.get("comparisons"), "paired_method_differences.comparisons"
    )
    if set(comparisons) != set(method_names):
        raise ArtifactReconciliationError("paired method summaries do not match saved methods")

    if status == "not_applicable":
        if (
            injected.any()
            or not reason
            or paired_metrics.get("status") != "not_applicable"
            or valid != 0
            or draw_indices.size
            or not np.array_equal(draw_offsets, np.asarray([0], dtype=np.int64))
            or valid_draw_indices.size
            or metric_a.size
            or metric_b.size
            or differences.size
        ):
            raise ArtifactReconciliationError(
                "not-applicable bootstrap evidence contains fabricated draw results"
            )
        for method in method_names:
            summary = _mapping(comparisons[method], f"paired comparison {method}")
            if (
                summary.get("status") != "not_applicable"
                or not isinstance(summary.get("reason"), str)
                or not str(summary["reason"]).strip()
            ):
                raise ArtifactReconciliationError(
                    f"0% paired comparison lacks documented N/A status: {method}"
                )
        return {
            "bootstrap_status": "not_applicable",
            "bootstrap_requested_iterations": requested,
            "bootstrap_valid_iterations": 0,
            "bootstrap_comparison_count": len(method_names),
        }
    if status != "reported" or reason or not injected.any():
        raise ArtifactReconciliationError("reported bootstrap status disagrees with event support")
    if paired_metrics.get("status") != "reported" or valid <= 0:
        raise ArtifactReconciliationError("reported paired metrics lack valid bootstrap draws")
    if (
        draw_indices.ndim != 1
        or draw_offsets.shape != (requested + 1,)
        or draw_offsets[0] != 0
        or draw_offsets[-1] != len(draw_indices)
        or np.any(np.diff(draw_offsets) <= 0)
        or np.any(draw_indices < 0)
        or np.any(draw_indices >= len(sample_ids))
        or valid_draw_indices.shape != (valid,)
        or np.any(valid_draw_indices < 0)
        or np.any(valid_draw_indices >= requested)
        or not np.array_equal(np.unique(valid_draw_indices), valid_draw_indices)
    ):
        raise ArtifactReconciliationError("reported bootstrap draw indexing is invalid")

    group_array = np.asarray(group_ids, dtype=np.str_)
    unique_groups = np.unique(group_array)
    group_members = {str(group): np.flatnonzero(group_array == group) for group in unique_groups}
    recomputed_valid: list[int] = []
    valid_column = 0
    for draw_number in range(requested):
        indices = draw_indices[draw_offsets[draw_number] : draw_offsets[draw_number + 1]]
        multiplicity_total = 0
        for group, members in group_members.items():
            counts = np.bincount(indices, minlength=len(sample_ids))[members]
            if len(set(int(value) for value in counts)) != 1:
                raise ArtifactReconciliationError(
                    f"bootstrap draw {draw_number} does not resample whole source group {group}"
                )
            multiplicity_total += int(counts[0])
        if multiplicity_total != len(unique_groups):
            raise ArtifactReconciliationError(
                f"bootstrap draw {draw_number} does not sample the source-group count"
            )
        if not injected[indices].any():
            continue
        recomputed_valid.append(draw_number)
        comparator_ap = _average_precision(injected[indices], scores_by_method[comparator][indices])
        if comparator_ap is None:
            raise ArtifactReconciliationError("valid bootstrap draw has undefined comparator AP")
        for row, method in enumerate(method_names):
            method_ap = _average_precision(injected[indices], scores_by_method[method][indices])
            if (
                method_ap is None
                or not math.isclose(
                    metric_a[row, valid_column], method_ap, rel_tol=1e-10, abs_tol=1e-12
                )
                or not math.isclose(
                    metric_b[row, valid_column], comparator_ap, rel_tol=1e-10, abs_tol=1e-12
                )
            ):
                raise ArtifactReconciliationError(
                    f"bootstrap metric draw disagrees with rankings for {method}/{draw_number}"
                )
        valid_column += 1
    if valid_column != valid or not np.array_equal(
        valid_draw_indices, np.asarray(recomputed_valid, dtype=np.int64)
    ):
        raise ArtifactReconciliationError(
            "bootstrap valid draw indices disagree with event support"
        )

    for row, method in enumerate(method_names):
        summary = _mapping(comparisons[method], f"paired comparison {method}")
        values = differences[row]
        recomputed_mean = float(values.mean())
        recomputed_interval = tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
        recomputed_probability = float(np.mean(values > 0.0) + 0.5 * np.mean(values == 0.0))
        reported_mean = _finite_number(
            summary.get("mean_difference"), f"paired comparison {method}.mean_difference"
        )
        reported_interval_raw = _sequence(
            summary.get("interval_95"), f"paired comparison {method}.interval_95"
        )
        if len(reported_interval_raw) != 2:
            raise ArtifactReconciliationError(
                f"paired comparison {method}.interval_95 must contain two values"
            )
        reported_interval = tuple(
            _finite_number(value, f"paired comparison {method}.interval_95[{index}]")
            for index, value in enumerate(reported_interval_raw)
        )
        reported_probability = _finite_number(
            summary.get("probability_positive"),
            f"paired comparison {method}.probability_positive",
        )
        if (
            summary.get("method") != method
            or summary.get("comparator") != comparator
            or summary.get("metric") != metric_name
            or summary.get("iterations") != requested
            or summary.get("valid_iterations") != valid
            or not math.isclose(
                reported_mean,
                recomputed_mean,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or not np.allclose(reported_interval, recomputed_interval, rtol=1e-10, atol=1e-12)
            or not math.isclose(
                reported_probability,
                recomputed_probability,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise ArtifactReconciliationError(
                f"paired method summary disagrees with saved draws for {method}"
            )
        if method == "fixed_hybrid":
            legacy = _mapping(
                metrics.get("paired_group_bootstrap_hybrid_minus_self_confidence"),
                "metrics.paired_group_bootstrap_hybrid_minus_self_confidence",
            )
            for key in (
                "iterations",
                "valid_iterations",
                "mean_difference",
                "interval_95",
                "probability_positive",
            ):
                if legacy.get(key) != summary.get(key):
                    raise ArtifactReconciliationError(
                        f"legacy fixed-hybrid bootstrap field disagrees with paired evidence: {key}"
                    )
    return {
        "bootstrap_status": "reported",
        "bootstrap_requested_iterations": requested,
        "bootstrap_valid_iterations": valid,
        "bootstrap_comparison_count": len(method_names),
    }


def reconcile_synthetic_smoke_artifacts(
    run_directory: str | Path,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check every primary saved artifact before a run can be sealed."""

    run_path = Path(run_directory)
    predictions_path = run_path / "oof_predictions.npz"
    if not predictions_path.is_file():
        raise ArtifactReconciliationError("required artifact is missing: oof_predictions.npz")
    required_arrays = {
        "sample_ids",
        "group_ids",
        "tissue_type",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "probabilities",
        "predicted_class",
        "fold_id",
        "class_order",
    }
    with np.load(predictions_path, allow_pickle=False) as payload:
        missing_arrays = required_arrays.difference(payload.files)
        if missing_arrays:
            raise ArtifactReconciliationError(
                f"oof_predictions.npz lacks arrays: {sorted(missing_arrays)}"
            )
        sample_ids = _strings(np.asarray(payload["sample_ids"]), "sample_ids")
        group_ids = _strings(np.asarray(payload["group_ids"]), "group_ids")
        tissue_types = _strings(np.asarray(payload["tissue_type"]), "tissue_type")
        pre = _integer_array(payload["pre_corruption_label"], "pre_corruption_label")
        observed = _integer_array(payload["observed_label"], "observed_label")
        injected_raw = np.asarray(payload["is_injected_corruption"])
        if not np.issubdtype(injected_raw.dtype, np.bool_):
            raise ArtifactReconciliationError(
                "is_injected_corruption must contain saved boolean values"
            )
        injected = np.asarray(injected_raw, dtype=bool)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        predicted = _integer_array(payload["predicted_class"], "predicted_class")
        fold_ids = _integer_array(payload["fold_id"], "fold_id")
        class_order = _integer_array(payload["class_order"], "class_order")
    n_samples = len(sample_ids)
    if n_samples == 0 or len(set(sample_ids)) != n_samples:
        raise ArtifactReconciliationError("sample_ids must be non-empty and unique")
    representation_sample_id = _reconcile_representation_example(run_path, sample_ids)
    if len(group_ids) != n_samples or len(tissue_types) != n_samples:
        raise ArtifactReconciliationError("group/tissue IDs do not align with sample_ids")
    for name, array in {
        "pre_corruption_label": pre,
        "observed_label": observed,
        "is_injected_corruption": injected,
        "predicted_class": predicted,
        "fold_id": fold_ids,
    }.items():
        if array.shape != (n_samples,):
            raise ArtifactReconciliationError(f"{name} does not align with sample_ids")
    if (
        class_order.ndim != 1
        or not len(class_order)
        or len(set(class_order.tolist())) != len(class_order)
    ):
        raise ArtifactReconciliationError("class_order must be a non-empty unique vector")
    if probabilities.shape != (n_samples, len(class_order)):
        raise ArtifactReconciliationError("probabilities shape disagrees with samples/classes")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ArtifactReconciliationError("probabilities contain invalid values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10, atol=1e-8):
        raise ArtifactReconciliationError("probability rows do not sum to one")
    expected_predicted = class_order[np.argmax(probabilities, axis=1)]
    if not np.array_equal(predicted, expected_predicted):
        raise ArtifactReconciliationError("predicted_class disagrees with probability argmax")
    valid_classes = set(int(value) for value in class_order)
    if any(int(value) not in valid_classes for value in np.r_[pre, observed]):
        raise ArtifactReconciliationError("saved label lies outside class_order")
    changed = pre != observed
    if not np.array_equal(changed, injected):
        raise ArtifactReconciliationError(
            "is_injected_corruption must equal (pre_corruption_label != observed_label)"
        )
    group_to_folds: dict[str, set[int]] = {}
    sample_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    for group_id, fold_id in zip(group_ids, fold_ids, strict=True):
        group_to_folds.setdefault(group_id, set()).add(int(fold_id))
    if any(len(values) != 1 for values in group_to_folds.values()):
        raise ArtifactReconciliationError("a source group is split across OOF folds")
    if np.any(fold_ids < 0):
        raise ArtifactReconciliationError("fold_id values must be non-negative")

    sample_counts = _mapping(metrics.get("sample_counts"), "metrics.sample_counts")
    if (
        _integer(sample_counts.get("audit_pool"), "metrics.sample_counts.audit_pool", minimum=0)
        != n_samples
    ):
        raise ArtifactReconciliationError("metrics sample_counts.audit_pool disagrees with NPZ")
    corruption_metrics = _mapping(metrics.get("corruption"), "metrics.corruption")
    if _integer(
        corruption_metrics.get("exact_count"), "metrics.corruption.exact_count", minimum=0
    ) != int(injected.sum()):
        raise ArtifactReconciliationError("metrics corruption.exact_count disagrees with NPZ")

    ranking_path = run_path / "ranking.csv"
    if not ranking_path.is_file():
        raise ArtifactReconciliationError("required artifact is missing: ranking.csv")
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))
    required_columns = {
        "rank",
        "sample_id",
        "group_id",
        "tissue_type",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "predicted_class",
    }
    if not ranking_rows or not required_columns.issubset(ranking_rows[0]):
        raise ArtifactReconciliationError("ranking.csv lacks required columns")
    if len(ranking_rows) != n_samples:
        raise ArtifactReconciliationError("ranking.csv row count disagrees with NPZ")
    ranked_ids = [row["sample_id"] for row in ranking_rows]
    if len(set(ranked_ids)) != n_samples or set(ranked_ids) != set(sample_ids):
        raise ArtifactReconciliationError("ranking.csv sample IDs do not match NPZ")
    if [int(row["rank"]) for row in ranking_rows] != list(range(1, n_samples + 1)):
        raise ArtifactReconciliationError("ranking.csv ranks must be contiguous from one")
    for row_number, row in enumerate(ranking_rows, start=2):
        index = sample_to_index[row["sample_id"]]
        if row["group_id"] != group_ids[index]:
            raise ArtifactReconciliationError(f"ranking.csv group mismatch at row {row_number}")
        if row["tissue_type"] != tissue_types[index]:
            raise ArtifactReconciliationError(f"ranking.csv tissue mismatch at row {row_number}")
        if int(row["pre_corruption_label"]) != int(pre[index]):
            raise ArtifactReconciliationError(f"ranking.csv pre-label mismatch at row {row_number}")
        if int(row["observed_label"]) != int(observed[index]):
            raise ArtifactReconciliationError(
                f"ranking.csv observed-label mismatch at row {row_number}"
            )
        if _parse_bool(row["is_injected_corruption"], f"ranking row {row_number}") != bool(
            injected[index]
        ):
            raise ArtifactReconciliationError(f"ranking.csv flag mismatch at row {row_number}")
        if int(row["predicted_class"]) != int(predicted[index]):
            raise ArtifactReconciliationError(
                f"ranking.csv predicted-class mismatch at row {row_number}"
            )

    corruption_manifest = _load_object(run_path / "corruption_manifest.json")
    manifest_rows = _sequence(corruption_manifest.get("rows"), "corruption_manifest.rows")
    if len(manifest_rows) != n_samples:
        raise ArtifactReconciliationError("corruption manifest row count disagrees with NPZ")
    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        record = _mapping(row, "corruption manifest row")
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in manifest_by_id:
            raise ArtifactReconciliationError("corruption manifest sample IDs are empty/duplicate")
        manifest_by_id[sample_id] = record
    if set(manifest_by_id) != set(sample_ids):
        raise ArtifactReconciliationError("corruption manifest sample IDs do not match NPZ")
    configuration_hash = _sha256_digest(
        corruption_metrics.get("configuration_hash"),
        "metrics.corruption.configuration_hash",
    )
    manifest_hashes = _configuration_hashes(corruption_manifest)
    if not manifest_hashes:
        raise ArtifactReconciliationError("corruption manifest lacks configuration hashes")
    for hash_path, hash_value in manifest_hashes:
        if _sha256_digest(hash_value, hash_path) != configuration_hash:
            raise ArtifactReconciliationError(
                f"{hash_path} disagrees with metrics.corruption.configuration_hash"
            )
    configuration_payload = _mapping(
        corruption_manifest.get("configuration_payload"),
        "corruption_manifest.configuration_payload",
    )
    payload_json = json.dumps(
        configuration_payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != configuration_hash:
        raise ArtifactReconciliationError(
            "corruption manifest configuration payload does not match its SHA-256"
        )
    for sample_id, index in sample_to_index.items():
        record = manifest_by_id[sample_id]
        if record.get("configuration_hash") != configuration_hash:
            raise ArtifactReconciliationError(f"corruption manifest row hash mismatch: {sample_id}")
        if str(record.get("group_id")) != group_ids[index]:
            raise ArtifactReconciliationError(f"corruption manifest group mismatch: {sample_id}")
        for field, expected in (
            ("pre_corruption_label", int(pre[index])),
            ("observed_label", int(observed[index])),
        ):
            if _integer(record.get(field), f"corruption manifest {field}") != expected:
                raise ArtifactReconciliationError(
                    f"corruption manifest {field} mismatch: {sample_id}"
                )
        manifest_flag = record.get("is_injected_corruption")
        if not isinstance(manifest_flag, bool) or manifest_flag != bool(injected[index]):
            raise ArtifactReconciliationError(f"corruption manifest flag mismatch: {sample_id}")

    report_inputs = _load_object(run_path / "report_inputs.json")
    tracked_artifact_names = {
        "metrics_path": "metrics.json",
        "predictions_path": "oof_predictions.npz",
        "rankings_path": "ranking.csv",
        "corruption_manifest_path": "corruption_manifest.json",
        "oof_provenance_path": "oof_provenance.json",
        "representation_example_path": "target_representation_example.npz",
        "neighbour_evidence_path": "neighbour_evidence.npz",
        "restoration_evidence_path": "restoration_evidence.npz",
        "bootstrap_evidence_path": "bootstrap_evidence.npz",
        "dataset_evidence_path": "synthetic_dataset_evidence.npz",
        "source_manifest_path": "synthetic_source_manifest.json",
        "source_manifest_csv_path": "synthetic_source_manifest.csv",
        "report_inputs_path": "report_inputs.json",
    }
    for key, filename in tracked_artifact_names.items():
        saved_path = report_inputs.get(key)
        expected_path = (run_path / filename).resolve()
        if not isinstance(saved_path, str) or Path(saved_path).resolve() != expected_path:
            raise ArtifactReconciliationError(
                f"report_inputs.{key} must point to the tracked {filename}"
            )
        if not expected_path.is_file():
            raise ArtifactReconciliationError(f"report_inputs references missing {filename}")
    duplicate_artifact_names = {
        "duplicate_audit_path": "duplicate_audit.json",
        "duplicate_candidates_csv_path": "duplicate_candidates.csv",
        "duplicate_candidates_figure_path": "figures/duplicate_candidates.png",
    }
    duplicate_files_present = {
        key: (run_path / filename).is_file() for key, filename in duplicate_artifact_names.items()
    }
    duplicate_inputs_present = {key: key in report_inputs for key in duplicate_artifact_names}
    if any(duplicate_files_present.values()) or any(duplicate_inputs_present.values()):
        if not all(duplicate_files_present.values()) or not all(duplicate_inputs_present.values()):
            raise ArtifactReconciliationError(
                "synthetic duplicate audit requires JSON, CSV, figure, and report-input paths"
            )
        for key, filename in duplicate_artifact_names.items():
            expected_path = (run_path / filename).resolve()
            saved_path = report_inputs.get(key)
            if not isinstance(saved_path, str) or Path(saved_path).resolve() != expected_path:
                raise ArtifactReconciliationError(
                    f"report_inputs.{key} must point to the tracked {filename}"
                )
    split = _mapping(report_inputs.get("split"), "report_inputs.split")
    audit_groups = {str(value) for value in _sequence(split.get("audit_groups"), "audit_groups")}
    reference_groups = {
        str(value)
        for value in _sequence(split.get("reference_validation_groups"), "reference groups")
    }
    final_groups = {
        str(value) for value in _sequence(split.get("final_test_groups"), "final groups")
    }
    if not reference_groups or not final_groups:
        raise ArtifactReconciliationError(
            "synthetic outer split requires reference-validation and final-reference groups"
        )
    if (
        audit_groups & reference_groups
        or audit_groups & final_groups
        or reference_groups & final_groups
    ):
        raise ArtifactReconciliationError("outer split group sets overlap")
    npz_groups = set(group_ids)
    if npz_groups != audit_groups:
        raise ArtifactReconciliationError("NPZ groups do not exactly match report audit groups")
    if npz_groups & final_groups:
        raise ArtifactReconciliationError("final-reference groups appear in audit predictions")
    reference_count = _integer(
        sample_counts.get("reference_validation"),
        "metrics.sample_counts.reference_validation",
        minimum=0,
    )
    final_count = _integer(
        sample_counts.get("final_reference_test"),
        "metrics.sample_counts.final_reference_test",
        minimum=0,
    )
    total_count = _integer(sample_counts.get("total"), "metrics.sample_counts.total", minimum=0)
    source_group_count = _integer(
        sample_counts.get("source_groups"),
        "metrics.sample_counts.source_groups",
        minimum=0,
    )
    if total_count != n_samples + reference_count + final_count:
        raise ArtifactReconciliationError("metrics total sample count is not partition-additive")
    split_group_count = len(audit_groups | reference_groups | final_groups)
    if source_group_count != split_group_count:
        raise ArtifactReconciliationError("metrics source-group count disagrees with outer split")
    if reference_count < len(reference_groups) or final_count < len(final_groups):
        raise ArtifactReconciliationError("outer split contains a group with no counted sample")
    class_names = tuple(
        str(value) for value in _sequence(report_inputs.get("class_names"), "class_names")
    )
    if len(class_names) != len(class_order) or any(not value for value in class_names):
        raise ArtifactReconciliationError("report class names do not align with class_order")
    class_name_by_id = {
        int(class_id): class_name
        for class_id, class_name in zip(class_order, class_names, strict=True)
    }
    pre_corruption_class_names = tuple(class_name_by_id[int(value)] for value in pre)
    dataset_evidence = _reconcile_synthetic_dataset_evidence(
        run_path,
        metrics=metrics,
        class_names=class_names,
        audit_sample_ids=sample_ids,
        audit_group_ids=group_ids,
        audit_pre=pre,
        audit_observed=observed,
        audit_injected=injected,
        audit_fold_ids=fold_ids,
        audit_groups=audit_groups,
        reference_groups=reference_groups,
        final_groups=final_groups,
    )
    duplicate_summary: dict[str, Any] = {}
    if all(duplicate_files_present.values()):
        try:
            duplicate_summary = reconcile_synthetic_duplicate_audit(
                run_path / "synthetic_dataset_evidence.npz",
                run_path / "duplicate_audit.json",
                run_path / "duplicate_candidates.csv",
                run_path / "figures" / "duplicate_candidates.png",
            )
        except SyntheticDuplicateAuditError as error:
            raise ArtifactReconciliationError(
                f"synthetic duplicate audit reconciliation failed: {error}"
            ) from error

    provenance = _load_object(run_path / "oof_provenance.json")
    if tuple(
        _integer(value, "oof_provenance.class_order")
        for value in _sequence(provenance.get("class_order"), "class_order")
    ) != tuple(int(value) for value in class_order):
        raise ArtifactReconciliationError("OOF provenance class_order disagrees with NPZ")
    folds = _sequence(provenance.get("folds"), "oof_provenance.folds")
    if not folds:
        raise ArtifactReconciliationError("OOF provenance must contain non-empty folds")
    held_sample_counts: Counter[str] = Counter()
    held_group_counts: Counter[str] = Counter()
    seen_fold_ids: set[int] = set()
    training_groups_by_fold: dict[int, set[str]] = {}
    for fold_value in folds:
        fold = _mapping(fold_value, "OOF fold")
        fold_id = _integer(fold.get("fold_id"), "OOF fold_id", minimum=0)
        if fold_id in seen_fold_ids:
            raise ArtifactReconciliationError(f"OOF fold_id is duplicated: {fold_id}")
        seen_fold_ids.add(fold_id)
        training = {str(value) for value in _sequence(fold.get("training_groups"), "training")}
        held_groups = {str(value) for value in _sequence(fold.get("held_out_groups"), "held")}
        overlap = _sequence(fold.get("group_overlap"), "group_overlap")
        if not training or not held_groups or overlap or training & held_groups:
            raise ArtifactReconciliationError(f"OOF fold {fold_id} has invalid group provenance")
        if training & final_groups or held_groups & final_groups:
            raise ArtifactReconciliationError(f"OOF fold {fold_id} includes final-reference groups")
        expected_held_groups = {
            group_id for group_id, group_folds in group_to_folds.items() if group_folds == {fold_id}
        }
        if held_groups != expected_held_groups:
            raise ArtifactReconciliationError(
                f"OOF fold {fold_id} held-out groups disagree with NPZ assignments"
            )
        if training != audit_groups.difference(held_groups):
            raise ArtifactReconciliationError(
                f"OOF fold {fold_id} training groups are not the audit-pool complement"
            )
        training_groups_by_fold[fold_id] = training
        held_group_counts.update(held_groups)
        held_samples = [
            str(value) for value in _sequence(fold.get("held_out_sample_ids"), "held samples")
        ]
        if not held_samples:
            raise ArtifactReconciliationError(f"OOF fold {fold_id} has no held-out samples")
        for sample_id in held_samples:
            if sample_id not in sample_to_index:
                raise ArtifactReconciliationError(
                    f"OOF fold references unknown sample: {sample_id}"
                )
            index = sample_to_index[sample_id]
            if group_ids[index] not in held_groups or int(fold_ids[index]) != fold_id:
                raise ArtifactReconciliationError(
                    f"OOF fold/sample/group assignment mismatch: {sample_id}"
                )
            held_sample_counts[sample_id] += 1
    if held_sample_counts != Counter({sample_id: 1 for sample_id in sample_ids}):
        raise ArtifactReconciliationError("OOF held-out samples lack exact once-only coverage")
    if held_group_counts != Counter({group_id: 1 for group_id in audit_groups}):
        raise ArtifactReconciliationError("OOF held-out groups lack exact once-only coverage")
    if seen_fold_ids != set(int(value) for value in np.unique(fold_ids)):
        raise ArtifactReconciliationError("OOF provenance fold IDs disagree with NPZ")
    oof_metrics = _mapping(metrics.get("oof"), "metrics.oof")
    if _integer(oof_metrics.get("folds"), "metrics.oof.folds", minimum=1) != len(folds):
        raise ArtifactReconciliationError("metrics OOF fold count disagrees with provenance")
    if oof_metrics.get("complete_once_coverage") is not True:
        raise ArtifactReconciliationError("metrics OOF coverage flag must record exact coverage")
    if (
        _integer(
            oof_metrics.get("group_overlap_count"),
            "metrics.oof.group_overlap_count",
            minimum=0,
        )
        != 0
    ):
        raise ArtifactReconciliationError("metrics OOF overlap count disagrees with provenance")
    maximum_probability_sum_error = float(
        np.max(np.abs(probabilities.sum(axis=1) - 1.0), initial=0.0)
    )
    saved_probability_error = oof_metrics.get("maximum_probability_sum_error")
    if not isinstance(saved_probability_error, (int, float)) or isinstance(
        saved_probability_error, bool
    ):
        raise ArtifactReconciliationError("metrics OOF probability-sum error must be numeric")
    if not math.isclose(
        float(saved_probability_error),
        maximum_probability_sum_error,
        rel_tol=1e-9,
        abs_tol=1e-15,
    ):
        raise ArtifactReconciliationError(
            "metrics OOF maximum probability-sum error disagrees with NPZ"
        )

    neighbour_risk_scores = _reconcile_neighbour_evidence(
        run_path,
        sample_ids=sample_ids,
        group_ids=group_ids,
        observed=observed,
        fold_ids=fold_ids,
        class_order=class_order,
        training_groups_by_fold=training_groups_by_fold,
    )
    restoration_summary = _reconcile_restoration_evidence(
        run_path,
        metrics=metrics,
        sample_ids=sample_ids,
        group_ids=group_ids,
        pre=pre,
        observed=observed,
        injected=injected,
        class_order=class_order,
        final_groups=final_groups,
        final_count=final_count,
        expected_final_sample_ids=dataset_evidence.final_sample_ids,
        expected_final_group_ids=dataset_evidence.final_group_ids,
        expected_final_reference_labels=dataset_evidence.final_reference_labels,
        expected_final_is_injected_corruption=(dataset_evidence.final_is_injected_corruption),
    )

    ranking_metrics = _mapping(metrics.get("ranking"), "metrics.ranking")
    method_count = 0
    scores_by_method: dict[str, NDArray[np.float64]] = {}
    subgroup_signatures: dict[str, tuple[tuple[str, int, int, str], ...]] = {}
    for method, method_value in ranking_metrics.items():
        method_payload = _mapping(method_value, f"metrics.ranking.{method}")
        budgets = _mapping(
            method_payload.get("review_budgets"), f"metrics.ranking.{method}.review_budgets"
        )
        if method not in ranking_rows[0]:
            raise ArtifactReconciliationError(f"ranking.csv lacks risk column for {method}")
        scores_by_sample: dict[str, float] = {}
        for row in ranking_rows:
            try:
                score = float(row[method])
            except (KeyError, ValueError) as exc:
                raise ArtifactReconciliationError(f"invalid ranking score for {method}") from exc
            if not math.isfinite(score):
                raise ArtifactReconciliationError(f"non-finite ranking score for {method}")
            scores_by_sample[row["sample_id"]] = score
        scores = np.asarray(
            [scores_by_sample[sample_id] for sample_id in sample_ids], dtype=np.float64
        )
        scores_by_method[str(method)] = scores
        if method == "neighbour_disagreement" and not np.allclose(
            scores, neighbour_risk_scores, rtol=1e-10, atol=1e-12
        ):
            raise ArtifactReconciliationError(
                "ranking.csv neighbour_disagreement disagrees with neighbour evidence"
            )
        recomputed_ap = _average_precision(injected, scores)
        _assert_close(
            method_payload.get("auroc"),
            _binary_auroc(injected, scores),
            f"metrics.ranking.{method}.auroc",
        )
        subgroup_payload = _mapping(
            method_payload.get("subgroups"), f"metrics.ranking.{method}.subgroups"
        )
        for dimension, expected_labels in (
            ("pre_corruption_class", pre_corruption_class_names),
            ("tissue_type", tissue_types),
        ):
            signature = _validate_subgroups(
                subgroup_payload.get(dimension),
                path=f"metrics.ranking.{method}.subgroups.{dimension}",
                flags=injected,
                scores=scores,
                expected_labels=expected_labels,
            )
            previous_signature = subgroup_signatures.setdefault(dimension, signature)
            if previous_signature != signature:
                raise ArtifactReconciliationError(
                    f"subgroup counts/status differ between methods for {dimension}"
                )
        tie_ids = np.asarray(sample_ids, dtype=np.str_)
        order = np.lexsort((tie_ids, -scores))
        if method == "fixed_hybrid" and ranked_ids != [sample_ids[index] for index in order]:
            raise ArtifactReconciliationError("ranking.csv rank order disagrees with fixed_hybrid")
        for budget_name, budget_value in budgets.items():
            saved = _mapping(budget_value, f"metrics.ranking.{method}.{budget_name}")
            fraction_value = saved.get("budget_fraction")
            if not isinstance(fraction_value, (int, float)) or isinstance(fraction_value, bool):
                raise ArtifactReconciliationError("budget_fraction must be numeric")
            fraction = float(fraction_value)
            reviewed_count = _budget_count(n_samples, fraction)
            found = int(injected[order[:reviewed_count]].sum())
            positives = int(injected.sum())
            precision = found / reviewed_count if reviewed_count else None
            recall = found / positives if positives else None
            expected_random_recall = reviewed_count / n_samples if positives else None
            lift = (
                recall / expected_random_recall
                if recall is not None and expected_random_recall
                else None
            )
            saved_reviewed = _integer(
                saved.get("reviewed_count"),
                f"{method}/{budget_name}/reviewed_count",
                minimum=0,
            )
            saved_injected = _integer(
                saved.get("injected_reviewed"),
                f"{method}/{budget_name}/injected_reviewed",
                minimum=0,
            )
            if saved_reviewed != reviewed_count or saved_injected != found:
                raise ArtifactReconciliationError(
                    f"saved budget counts disagree for {method}/{budget_name}"
                )
            if _integer(
                saved.get("total_examples"),
                f"{method}/{budget_name}/total_examples",
                minimum=0,
            ) != n_samples or _integer(
                saved.get("injected_total"),
                f"{method}/{budget_name}/injected_total",
                minimum=0,
            ) != int(injected.sum()):
                raise ArtifactReconciliationError(
                    f"saved budget totals disagree for {method}/{budget_name}"
                )
            if (
                _integer(
                    saved.get("false_alert_count"),
                    f"{method}/{budget_name}/false_alert_count",
                    minimum=0,
                )
                != reviewed_count - found
            ):
                raise ArtifactReconciliationError(
                    f"saved false-alert count disagrees for {method}/{budget_name}"
                )
            if not int(injected.sum()) and (
                saved.get("status") != "not_applicable"
                or not isinstance(saved.get("reason"), str)
                or not str(saved["reason"]).strip()
            ):
                raise ArtifactReconciliationError(
                    f"0% corruption budget lacks documented N/A status: {method}/{budget_name}"
                )
            _assert_close(saved.get("precision"), precision, f"{method}/{budget_name}/precision")
            _assert_close(saved.get("recall"), recall, f"{method}/{budget_name}/recall")
            _assert_close(
                saved.get("expected_random_recall"),
                expected_random_recall,
                f"{method}/{budget_name}/expected_random_recall",
            )
            _assert_close(saved.get("lift_over_random"), lift, f"{method}/{budget_name}/lift")
            _assert_close(
                saved.get("average_precision"),
                recomputed_ap,
                f"{method}/{budget_name}/average_precision",
            )
        method_count += 1
    if method_count == 0:
        raise ArtifactReconciliationError("metrics contain no ranking methods")
    bootstrap_summary = _reconcile_bootstrap_evidence(
        run_path,
        metrics=metrics,
        sample_ids=sample_ids,
        group_ids=group_ids,
        injected=injected,
        scores_by_method=scores_by_method,
    )
    return {
        "status": "passed",
        "sample_count": n_samples,
        "group_count": len(npz_groups),
        "class_count": len(class_order),
        "injected_corruption_count": int(injected.sum()),
        "oof_fold_count": len(folds),
        "ranking_method_count": method_count,
        "ranking_row_count": len(ranking_rows),
        "corruption_manifest_row_count": len(manifest_rows),
        "final_reference_group_overlap_count": len(npz_groups & final_groups),
        "representation_example_sample_id": representation_sample_id,
        "neighbour_evidence_sample_count": len(neighbour_risk_scores),
        **dataset_evidence.summary,
        **duplicate_summary,
        **restoration_summary,
        **bootstrap_summary,
    }


__all__ = ["ArtifactReconciliationError", "reconcile_synthetic_smoke_artifacts"]
