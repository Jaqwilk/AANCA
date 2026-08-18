"""Read-only duplicate evidence for tracked synthetic smoke runs.

This module deliberately audits source patches rather than nucleus rows.  Its exact
array and deterministic average-hash signals are both derived from the same synthetic
pixels, so the result is software-validation evidence only.  It is not the independent
two-signal PanNuke duplicate gate.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

from histo_audit.data.duplicates import (
    canonical_array_sha256,
    find_exact_duplicate_pairs,
    find_perceptual_duplicate_candidates,
    perceptual_hash,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: I001


_CSV_FIELDS = (
    "candidate_rank",
    "patch_id_a",
    "patch_id_b",
    "representative_sample_id_a",
    "representative_sample_id_b",
    "split_partition_a",
    "split_partition_b",
    "official_fold_a",
    "official_fold_b",
    "cross_partition",
    "cross_official_fold",
    "exact_match",
    "exact_sha256",
    "perceptual_hash_a",
    "perceptual_hash_b",
    "perceptual_hamming_distance",
    "signals",
    "recommended_action",
    "automatic_deletion",
)


class SyntheticDuplicateAuditError(ValueError):
    """Raised when synthetic duplicate evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SyntheticDuplicateAuditArtifacts:
    """Paths and payload produced by one tracked synthetic patch audit."""

    json_path: Path
    csv_path: Path
    figure_path: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _UniquePatchEvidence:
    patch_ids: tuple[str, ...]
    representative_sample_ids: tuple[str, ...]
    representative_indices: tuple[int, ...]
    nucleus_counts: tuple[int, ...]
    split_partitions: tuple[str, ...]
    official_folds: tuple[int, ...]
    images: NDArray[np.uint8]


def _strings(values: NDArray[np.generic], name: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise SyntheticDuplicateAuditError(f"{name} must be one-dimensional")
    result = tuple(str(value) for value in array.tolist())
    if any(not value for value in result):
        raise SyntheticDuplicateAuditError(f"{name} contains an empty identifier")
    return result


def _load_unique_patches(dataset_path: Path) -> tuple[_UniquePatchEvidence, int]:
    required = {
        "sample_ids",
        "patch_ids",
        "official_fold",
        "split_partition",
        "images",
    }
    try:
        with np.load(dataset_path, allow_pickle=False) as payload:
            missing = required.difference(payload.files)
            if missing:
                raise SyntheticDuplicateAuditError(
                    f"synthetic dataset evidence lacks arrays: {sorted(missing)}"
                )
            sample_ids = _strings(np.asarray(payload["sample_ids"]), "sample_ids")
            patch_ids = _strings(np.asarray(payload["patch_ids"]), "patch_ids")
            partitions = _strings(np.asarray(payload["split_partition"]), "split_partition")
            folds_raw = np.asarray(payload["official_fold"])
            images_raw = np.asarray(payload["images"])
    except OSError as error:
        raise SyntheticDuplicateAuditError(
            f"could not read synthetic dataset evidence: {dataset_path}"
        ) from error

    sample_count = len(sample_ids)
    if sample_count == 0 or len(set(sample_ids)) != sample_count:
        raise SyntheticDuplicateAuditError("sample_ids must be non-empty and unique")
    if len(patch_ids) != sample_count or len(partitions) != sample_count:
        raise SyntheticDuplicateAuditError("patch/partition arrays do not align with samples")
    if folds_raw.shape != (sample_count,) or not np.issubdtype(folds_raw.dtype, np.integer):
        raise SyntheticDuplicateAuditError("official_fold must be an aligned integer vector")
    if (
        images_raw.ndim != 4
        or images_raw.shape[0] != sample_count
        or images_raw.shape[-1] != 3
        or images_raw.dtype != np.uint8
    ):
        raise SyntheticDuplicateAuditError(
            "images must be aligned uint8 channel-last RGB source patches"
        )

    indices_by_patch: dict[str, list[int]] = {}
    for index, patch_id in enumerate(patch_ids):
        indices_by_patch.setdefault(patch_id, []).append(index)
    ordered_patch_ids = tuple(indices_by_patch)
    representative_indices: list[int] = []
    representative_sample_ids: list[str] = []
    nucleus_counts: list[int] = []
    unique_partitions: list[str] = []
    unique_folds: list[int] = []
    unique_images: list[NDArray[np.uint8]] = []
    for patch_id in ordered_patch_ids:
        indices = indices_by_patch[patch_id]
        first = indices[0]
        reference_image = np.asarray(images_raw[first], dtype=np.uint8)
        partition_values = {partitions[index] for index in indices}
        fold_values = {int(folds_raw[index]) for index in indices}
        if len(partition_values) != 1 or len(fold_values) != 1:
            raise SyntheticDuplicateAuditError(
                f"nucleus rows for patch {patch_id!r} disagree on partition or official fold"
            )
        if any(not np.array_equal(reference_image, images_raw[index]) for index in indices[1:]):
            raise SyntheticDuplicateAuditError(
                f"nucleus rows for patch {patch_id!r} do not retain one source image"
            )
        representative_indices.append(first)
        representative_sample_ids.append(sample_ids[first])
        nucleus_counts.append(len(indices))
        unique_partitions.append(next(iter(partition_values)))
        unique_folds.append(next(iter(fold_values)))
        unique_images.append(reference_image)
    if not unique_images:
        raise SyntheticDuplicateAuditError("no unique synthetic source patches were found")
    return (
        _UniquePatchEvidence(
            patch_ids=ordered_patch_ids,
            representative_sample_ids=tuple(representative_sample_ids),
            representative_indices=tuple(representative_indices),
            nucleus_counts=tuple(nucleus_counts),
            split_partitions=tuple(unique_partitions),
            official_folds=tuple(unique_folds),
            images=np.stack(unique_images).astype(np.uint8, copy=False),
        ),
        sample_count,
    )


def _analyse(
    dataset_path: Path,
    *,
    hash_size: int,
    max_hamming_distance: int,
) -> tuple[dict[str, Any], _UniquePatchEvidence]:
    if hash_size <= 0:
        raise SyntheticDuplicateAuditError("hash_size must be positive")
    if max_hamming_distance < 0 or max_hamming_distance > hash_size * hash_size:
        raise SyntheticDuplicateAuditError(
            "max_hamming_distance must lie between zero and the perceptual hash bit count"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"synthetic dataset evidence does not exist: {dataset_path}")
    unique, sample_count = _load_unique_patches(dataset_path)
    boundary_keys = tuple(
        f"partition={partition}|official_fold={fold}"
        for partition, fold in zip(unique.split_partitions, unique.official_folds, strict=True)
    )
    exact = find_exact_duplicate_pairs(
        unique.images,
        sample_ids=unique.patch_ids,
        folds=boundary_keys,
        cross_fold_only=True,
    )
    perceptual = find_perceptual_duplicate_candidates(
        unique.images,
        sample_ids=unique.patch_ids,
        folds=boundary_keys,
        max_hamming_distance=max_hamming_distance,
        hash_size=hash_size,
        cross_fold_only=True,
    )
    exact_pairs = {(row.sample_id_a, row.sample_id_b): row for row in exact}
    perceptual_pairs = {(row.sample_id_a, row.sample_id_b): row for row in perceptual}
    patch_index = {patch_id: index for index, patch_id in enumerate(unique.patch_ids)}
    patch_hashes = tuple(perceptual_hash(image, hash_size=hash_size) for image in unique.images)

    rows: list[dict[str, Any]] = []
    for patch_a, patch_b in sorted(set(exact_pairs) | set(perceptual_pairs)):
        first = patch_index[patch_a]
        second = patch_index[patch_b]
        exact_row = exact_pairs.get((patch_a, patch_b))
        perceptual_row = perceptual_pairs.get((patch_a, patch_b))
        exact_match = exact_row is not None
        distance = (
            int(perceptual_row.perceptual_hamming_distance)
            if perceptual_row is not None and perceptual_row.perceptual_hamming_distance is not None
            else 0
        )
        signals = ["exact_array_sha256"] if exact_match else []
        if perceptual_row is not None:
            signals.append("deterministic_average_hash")
        rows.append(
            {
                "patch_id_a": patch_a,
                "patch_id_b": patch_b,
                "representative_sample_id_a": unique.representative_sample_ids[first],
                "representative_sample_id_b": unique.representative_sample_ids[second],
                "split_partition_a": unique.split_partitions[first],
                "split_partition_b": unique.split_partitions[second],
                "official_fold_a": unique.official_folds[first],
                "official_fold_b": unique.official_folds[second],
                "cross_partition": (
                    unique.split_partitions[first] != unique.split_partitions[second]
                ),
                "cross_official_fold": (
                    unique.official_folds[first] != unique.official_folds[second]
                ),
                "exact_match": exact_match,
                "exact_sha256": (exact_row.exact_sha256 if exact_row is not None else None),
                "perceptual_hash_a": patch_hashes[first],
                "perceptual_hash_b": patch_hashes[second],
                "perceptual_hamming_distance": distance,
                "signals": signals,
                "recommended_action": "review_only",
                "automatic_deletion": False,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["exact_match"]),
            int(row["perceptual_hamming_distance"]),
            str(row["patch_id_a"]),
            str(row["patch_id_b"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["candidate_rank"] = rank

    unique_pair_count = len(unique.patch_ids) * (len(unique.patch_ids) - 1) // 2
    cross_boundary_pair_count = sum(
        1
        for first in range(len(unique.patch_ids))
        for second in range(first + 1, len(unique.patch_ids))
        if boundary_keys[first] != boundary_keys[second]
    )
    inventory = [
        {
            "patch_id": patch_id,
            "representative_sample_id": unique.representative_sample_ids[index],
            "representative_record_index": unique.representative_indices[index],
            "nucleus_row_count": unique.nucleus_counts[index],
            "split_partition": unique.split_partitions[index],
            "official_fold": unique.official_folds[index],
            "exact_array_sha256": canonical_array_sha256(unique.images[index]),
            "perceptual_hash": patch_hashes[index],
        }
        for index, patch_id in enumerate(unique.patch_ids)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "artifact_scope": "synthetic_software_validation",
        "audit_scope": "unique_synthetic_source_patches",
        "study_outcome_eligible": False,
        "real_data_duplicate_gate_eligible": False,
        "required_two_signal_near_duplicate_gate_complete": False,
        "scope_limitation": (
            "Synthetic exact-array and deterministic average-hash evidence only. Both "
            "signals derive from the same pixels; this is not independent PanNuke "
            "perceptual-plus-embedding evidence and cannot satisfy a real-data gate."
        ),
        "automatic_deletion": False,
        "candidate_disposition": "recommended_for_review_only",
        "dataset_evidence": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "nucleus_sample_count": sample_count,
            "unique_patch_count": len(unique.patch_ids),
            "deduplicated_nucleus_row_count": sample_count - len(unique.patch_ids),
        },
        "deduplication": {
            "unit": "patch_id",
            "representative_rule": "first record_index for each patch_id",
            "within_patch_image_equality_required": True,
            "within_patch_partition_and_fold_consistency_required": True,
        },
        "candidate_definition": {
            "boundary_rule": ("split_partition differs OR official_fold differs"),
            "exact_rule": "canonical array SHA-256 match confirmed by array equality",
            "perceptual_rule": (
                "deterministic average-hash Hamming distance at or below threshold"
            ),
            "near_duplicate_signals_independent": False,
        },
        "thresholds": {
            "perceptual_hash_algorithm": "deterministic_average_hash",
            "perceptual_hash_size": hash_size,
            "perceptual_hash_bits": hash_size * hash_size,
            "max_perceptual_hamming_distance": max_hamming_distance,
        },
        "pair_counts": {
            "all_unique_patch_pairs": unique_pair_count,
            "evaluated_cross_boundary_pairs": cross_boundary_pair_count,
        },
        "candidate_counts": {
            "exact": len(exact_pairs),
            "perceptual_including_exact": len(perceptual_pairs),
            "union": len(rows),
            "cross_partition_union": sum(bool(row["cross_partition"]) for row in rows),
            "cross_official_fold_union": sum(bool(row["cross_official_fold"]) for row in rows),
        },
        "unique_patch_inventory": inventory,
        "candidates": rows,
    }
    return payload, unique


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = dict(row)
        values["signals"] = json.dumps(values["signals"], separators=(",", ":"))
        values["cross_partition"] = str(bool(values["cross_partition"])).lower()
        values["cross_official_fold"] = str(bool(values["cross_official_fold"])).lower()
        values["exact_match"] = str(bool(values["exact_match"])).lower()
        values["automatic_deletion"] = str(bool(values["automatic_deletion"])).lower()
        values["exact_sha256"] = values["exact_sha256"] or ""
        writer.writerow({field: values[field] for field in _CSV_FIELDS})
    return stream.getvalue()


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format="png", dpi=140, bbox_inches="tight")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)


def _render_figure(
    payload: Mapping[str, Any],
    unique: _UniquePatchEvidence,
    destination: Path,
    *,
    max_pairs: int = 4,
) -> None:
    candidates_raw = payload.get("candidates")
    candidates = (
        [dict(row) for row in candidates_raw if isinstance(row, Mapping)]
        if isinstance(candidates_raw, Sequence)
        else []
    )
    counts = payload["candidate_counts"]
    thresholds = payload["thresholds"]
    if not candidates:
        figure, axis = plt.subplots(figsize=(9.2, 4.2))
        axis.axis("off")
        axis.text(
            0.5,
            0.62,
            "No cross-boundary synthetic duplicate candidates",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
            color="#243b53",
        )
        axis.text(
            0.5,
            0.38,
            (
                f"{payload['dataset_evidence']['unique_patch_count']} unique patch_id values; "
                f"exact equality + deterministic {thresholds['perceptual_hash_bits']}-bit "
                f"average hash (maximum distance "
                f"{thresholds['max_perceptual_hamming_distance']}).\n"
                "Synthetic software-validation evidence only; nothing was deleted."
            ),
            ha="center",
            va="center",
            fontsize=10,
            color="#52616b",
        )
        _save_figure(figure, destination)
        return

    shown = candidates[: max(1, min(max_pairs, len(candidates)))]
    patch_index = {patch_id: index for index, patch_id in enumerate(unique.patch_ids)}
    figure, axes = plt.subplots(len(shown), 2, figsize=(10.5, 3.4 * len(shown)), squeeze=False)
    for row_index, candidate in enumerate(shown):
        for column, suffix in enumerate(("a", "b")):
            patch_id = str(candidate[f"patch_id_{suffix}"])
            index = patch_index[patch_id]
            axis = axes[row_index, column]
            axis.imshow(unique.images[index])
            axis.set_title(
                f"{patch_id}\n{candidate[f'split_partition_{suffix}']} | "
                f"official fold {candidate[f'official_fold_{suffix}']}",
                fontsize=9,
            )
            axis.axis("off")
        signal_text = (
            "exact + aHash"
            if bool(candidate["exact_match"])
            else f"aHash distance={candidate['perceptual_hamming_distance']}"
        )
        axes[row_index, 0].text(
            0.0,
            -0.13,
            f"Candidate {candidate['candidate_rank']}: {signal_text}; review only",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#334e68",
        )
    figure.suptitle(
        "Synthetic cross-partition/fold duplicate candidates — no automatic deletion\n"
        f"union={counts['union']}, exact={counts['exact']}, "
        f"average-hash={counts['perceptual_including_exact']}; displaying {len(shown)}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(figure, destination)


def audit_synthetic_duplicate_patches(
    dataset_evidence_path: str | Path,
    output_directory: str | Path,
    *,
    hash_size: int = 8,
    max_hamming_distance: int = 4,
) -> SyntheticDuplicateAuditArtifacts:
    """Audit unique synthetic source patches and write immutable-ready evidence.

    Nucleus rows are collapsed by ``patch_id`` before any pair is evaluated.  Candidate
    patches are retained unchanged and are recommended for review only.
    """

    dataset_path = Path(dataset_evidence_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload, unique = _analyse(
        dataset_path,
        hash_size=hash_size,
        max_hamming_distance=max_hamming_distance,
    )
    csv_path = output / "duplicate_candidates.csv"
    figure_path = output / "figures" / "duplicate_candidates.png"
    json_path = output / "duplicate_audit.json"
    atomic_write_text(csv_path, _csv_text(payload["candidates"]))
    _render_figure(payload, unique, figure_path)
    payload["artifacts"] = {
        "candidate_csv": {
            "path": csv_path.name,
            "sha256": sha256_file(csv_path),
        },
        "candidate_figure": {
            "path": str(Path("figures") / figure_path.name).replace("\\", "/"),
            "sha256": sha256_file(figure_path),
        },
    }
    atomic_write_json(json_path, payload)
    return SyntheticDuplicateAuditArtifacts(
        json_path=json_path,
        csv_path=csv_path,
        figure_path=figure_path,
        payload=payload,
    )


def load_synthetic_duplicate_audit(path: str | Path) -> dict[str, Any]:
    """Load a synthetic duplicate-audit JSON object without accepting non-finite values."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SyntheticDuplicateAuditError(f"invalid duplicate-audit JSON: {source}") from error
    if not isinstance(payload, dict):
        raise SyntheticDuplicateAuditError("duplicate-audit JSON root must be an object")
    return payload


def reconcile_synthetic_duplicate_audit(
    dataset_evidence_path: str | Path,
    duplicate_audit_path: str | Path,
    candidate_csv_path: str | Path,
    candidate_figure_path: str | Path,
) -> dict[str, Any]:
    """Recompute and reconcile every machine-readable synthetic duplicate claim."""

    dataset = Path(dataset_evidence_path).resolve()
    audit_path = Path(duplicate_audit_path).resolve()
    csv_path = Path(candidate_csv_path).resolve()
    figure_path = Path(candidate_figure_path).resolve()
    payload = load_synthetic_duplicate_audit(audit_path)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise SyntheticDuplicateAuditError("duplicate audit lacks thresholds")
    hash_size = thresholds.get("perceptual_hash_size")
    maximum = thresholds.get("max_perceptual_hamming_distance")
    if not isinstance(hash_size, int) or isinstance(hash_size, bool):
        raise SyntheticDuplicateAuditError("duplicate audit hash size is invalid")
    if not isinstance(maximum, int) or isinstance(maximum, bool):
        raise SyntheticDuplicateAuditError("duplicate audit Hamming threshold is invalid")
    expected, _ = _analyse(dataset, hash_size=hash_size, max_hamming_distance=maximum)
    observed_core = {key: value for key, value in payload.items() if key != "artifacts"}
    if observed_core != expected:
        raise SyntheticDuplicateAuditError(
            "duplicate-audit claims disagree with recomputation from dataset evidence"
        )
    if not csv_path.is_file() or csv_path.read_text(encoding="utf-8") != _csv_text(
        expected["candidates"]
    ):
        raise SyntheticDuplicateAuditError("duplicate candidate CSV disagrees with recomputation")
    if not figure_path.is_file() or not figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise SyntheticDuplicateAuditError("duplicate candidate figure is missing or not PNG")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SyntheticDuplicateAuditError("duplicate audit lacks artifact hash bindings")
    expected_bindings = {
        "candidate_csv": (csv_path.name, sha256_file(csv_path)),
        "candidate_figure": (
            str(Path("figures") / figure_path.name).replace("\\", "/"),
            sha256_file(figure_path),
        ),
    }
    for key, (relative_path, digest) in expected_bindings.items():
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise SyntheticDuplicateAuditError(f"duplicate audit lacks {key} binding")
        if record.get("path") != relative_path or record.get("sha256") != digest:
            raise SyntheticDuplicateAuditError(f"duplicate audit {key} hash binding changed")
    if payload.get("real_data_duplicate_gate_eligible") is not False:
        raise SyntheticDuplicateAuditError(
            "synthetic duplicate audit must fail closed for real data"
        )
    dataset_record = payload.get("dataset_evidence")
    if not isinstance(dataset_record, Mapping) or dataset_record.get("path") != str(dataset):
        raise SyntheticDuplicateAuditError("duplicate audit dataset path binding changed")
    counts = payload["candidate_counts"]
    return {
        "synthetic_duplicate_audit_status": "passed",
        "synthetic_duplicate_audit_nucleus_sample_count": payload["dataset_evidence"][
            "nucleus_sample_count"
        ],
        "synthetic_duplicate_audit_unique_patch_count": payload["dataset_evidence"][
            "unique_patch_count"
        ],
        "synthetic_duplicate_audit_candidate_count": counts["union"],
        "synthetic_duplicate_audit_exact_candidate_count": counts["exact"],
        "synthetic_duplicate_audit_perceptual_candidate_count": counts[
            "perceptual_including_exact"
        ],
        "synthetic_duplicate_audit_real_data_gate_eligible": False,
    }


__all__ = [
    "SyntheticDuplicateAuditArtifacts",
    "SyntheticDuplicateAuditError",
    "audit_synthetic_duplicate_patches",
    "load_synthetic_duplicate_audit",
    "reconcile_synthetic_duplicate_audit",
]
