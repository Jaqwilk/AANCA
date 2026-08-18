from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import stat
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "data" / "raw" / "pannuke"
OUTPUT_ROOT = Path(__file__).resolve().parent
POSITIVE_CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective_soft_tissue",
    "dead",
    "non_neoplastic_epithelial",
)
BACKGROUND_CHANNEL_INDEX = 5
EXPECTED_ARCHIVES = {
    "fold_1.zip": {
        "size_bytes": 700_275_281,
        "sha256": "6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
        "source": "DECISIONS.md:60 (locally computed; not publisher-provided)",
    },
    "fold_2.zip": {
        "size_bytes": 658_842_552,
        "sha256": "5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
        "source": "DECISIONS.md:60 (locally computed; not publisher-provided)",
    },
    "fold_3.zip": {
        "size_bytes": 717_969_882,
        "sha256": "c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d",
        "source": "DECISIONS.md:60 (locally computed; not publisher-provided)",
    },
}
HASH_CHUNK_BYTES = 16 * 1024 * 1024
MASK_BATCH_PATCHES = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def file_hashes(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return digest.hexdigest(), f"{crc & 0xFFFFFFFF:08x}"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def shape_text(shape: tuple[int, ...] | None) -> str:
    return "" if shape is None else "x".join(str(value) for value in shape)


def inspect_npy(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    data_offset = path.stat().st_size - int(array.nbytes)
    return {
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(array.dtype),
        "dtype_str": array.dtype.str,
        "fortran_order": bool(np.isfortran(array)),
        "c_contiguous": bool(array.flags.c_contiguous),
        "nbytes": int(array.nbytes),
        "npy_header_bytes": int(data_offset),
    }


def safe_zip_member(name: str, raw_root: Path) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    if "\x00" in name:
        reasons.append("nul_byte")
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or normalized.startswith("/"):
        reasons.append("absolute_path")
    if windows.drive or normalized.startswith("//"):
        reasons.append("drive_or_unc_path")
    if any(part in {"..", "."} for part in posix.parts):
        reasons.append("dot_or_parent_component")
    try:
        destination = (raw_root / Path(*posix.parts)).resolve()
        destination.relative_to(raw_root.resolve())
    except (OSError, ValueError):
        reasons.append("resolved_escape")
    return not reasons, reasons, normalized


def audit_archives(
    inventory_by_relative: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for archive_name, expected in EXPECTED_ARCHIVES.items():
        path = RAW_ROOT / archive_name
        row: dict[str, Any] = {
            "archive": archive_name,
            "exists": path.is_file(),
            "expected_size_bytes": expected["size_bytes"],
            "expected_sha256": expected["sha256"],
            "expected_identity_source": expected["source"],
        }
        if not path.is_file():
            row.update(
                {
                    "identity_matches_expected": False,
                    "crc_test_passed": False,
                    "path_safety_passed": False,
                    "error": "archive_missing",
                }
            )
            archive_rows.append(row)
            continue

        inventory = inventory_by_relative[relative(path)]
        row.update(
            {
                "size_bytes": int(path.stat().st_size),
                "sha256": inventory["sha256"],
                "identity_matches_expected": (
                    int(path.stat().st_size) == expected["size_bytes"]
                    and inventory["sha256"] == expected["sha256"]
                ),
            }
        )
        exact_names: set[str] = set()
        normalized_names: set[str] = set()
        casefold_names: set[str] = set()
        unsafe_count = 0
        duplicate_count = 0
        symlink_count = 0
        encrypted_count = 0
        extracted_mismatch_count = 0
        started = time.perf_counter()
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            for info in infos:
                safe, reasons, normalized = safe_zip_member(info.filename, RAW_ROOT)
                mode = (info.external_attr >> 16) & 0xFFFF
                is_symlink = stat.S_ISLNK(mode)
                encrypted = bool(info.flag_bits & 0x1)
                duplicates: list[str] = []
                if info.filename in exact_names:
                    duplicates.append("exact")
                if normalized in normalized_names:
                    duplicates.append("normalized")
                if normalized.casefold() in casefold_names:
                    duplicates.append("casefold")
                exact_names.add(info.filename)
                normalized_names.add(normalized)
                casefold_names.add(normalized.casefold())
                unsafe_count += int(not safe)
                duplicate_count += int(bool(duplicates))
                symlink_count += int(is_symlink)
                encrypted_count += int(encrypted)

                extracted_relative = f"data/raw/pannuke/{normalized.rstrip('/')}"
                extracted = inventory_by_relative.get(extracted_relative)
                extracted_matches = None
                if not info.is_dir():
                    extracted_matches = bool(
                        extracted is not None
                        and extracted["size_bytes"] == int(info.file_size)
                        and extracted["crc32"] == f"{info.CRC:08x}"
                    )
                    extracted_mismatch_count += int(not extracted_matches)
                member_rows.append(
                    {
                        "archive": archive_name,
                        "member": info.filename,
                        "normalized_member": normalized,
                        "is_directory": info.is_dir(),
                        "compression_method": int(info.compress_type),
                        "compressed_size_bytes": int(info.compress_size),
                        "uncompressed_size_bytes": int(info.file_size),
                        "crc32": f"{info.CRC:08x}",
                        "encrypted": encrypted,
                        "is_symlink": is_symlink,
                        "path_safe": safe,
                        "path_safety_reasons": ";".join(reasons),
                        "duplicate_name_types": ";".join(duplicates),
                        "extracted_relative_path": (
                            "" if info.is_dir() else extracted_relative
                        ),
                        "extracted_size_crc_match": (
                            "" if extracted_matches is None else extracted_matches
                        ),
                    }
                )
            bad_member = archive.testzip()
        row.update(
            {
                "member_count": len(infos),
                "file_member_count": sum(not item.is_dir() for item in infos),
                "directory_member_count": sum(item.is_dir() for item in infos),
                "compressed_member_bytes": sum(item.compress_size for item in infos),
                "uncompressed_member_bytes": sum(item.file_size for item in infos),
                "crc_test_bad_member": bad_member,
                "crc_test_passed": bad_member is None,
                "crc_test_seconds": round(time.perf_counter() - started, 6),
                "unsafe_member_count": unsafe_count,
                "duplicate_member_count": duplicate_count,
                "symlink_member_count": symlink_count,
                "encrypted_member_count": encrypted_count,
                "path_safety_passed": (
                    unsafe_count == duplicate_count == symlink_count == encrypted_count == 0
                ),
                "extracted_member_mismatch_count": extracted_mismatch_count,
                "extracted_members_match_size_crc": extracted_mismatch_count == 0,
                "error": "",
            }
        )
        archive_rows.append(row)
    return archive_rows, member_rows


def _integer_id(value: float) -> int | str:
    return int(value) if np.isfinite(value) and value.is_integer() else repr(value)


def scan_mask_fold(fold_id: int, path: Path) -> dict[str, Any]:
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    if masks.ndim != 4 or masks.shape[-1] != 6:
        raise RuntimeError(f"fold {fold_id}: unsupported mask shape {masks.shape}")

    n_patches, height, width, channels = (int(value) for value in masks.shape)
    category_totals: Counter[str] = Counter()
    category_patch_counts: Counter[str] = Counter()
    channel_min = np.full(channels, np.inf, dtype=np.float64)
    channel_max = np.full(channels, -np.inf, dtype=np.float64)
    channel_nonfinite = np.zeros(channels, dtype=np.int64)
    channel_negative = np.zeros(channels, dtype=np.int64)
    channel_fractional = np.zeros(channels, dtype=np.int64)
    channel_positive_values = np.zeros(channels, dtype=np.int64)
    background_nonbinary = 0
    affected_instances_by_class: dict[int, set[tuple[int, float]]] = defaultdict(set)
    affected_numeric_ids_by_class: dict[int, set[float]] = defaultdict(set)
    all_instances_by_class: dict[int, set[tuple[int, float]]] = defaultdict(set)
    affected_overlap_classes: set[int] = set()
    pair_overlap_pixels: Counter[tuple[int, int]] = Counter()
    affected_instance_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    overlap_active_value_count = 0
    overlap_invalid_instance_value_count = 0
    overlap_touched_duplicate_keys: set[tuple[int, float]] = set()
    total_duplicate_keys: set[tuple[int, float]] = set()

    for start in range(0, n_patches, MASK_BATCH_PATCHES):
        stop = min(n_patches, start + MASK_BATCH_PATCHES)
        batch = np.asarray(masks[start:stop])
        for channel_index in range(channels):
            values = batch[..., channel_index]
            finite = np.isfinite(values)
            channel_nonfinite[channel_index] += int(np.count_nonzero(~finite))
            if np.any(finite):
                channel_min[channel_index] = min(
                    channel_min[channel_index], float(np.min(values[finite]))
                )
                channel_max[channel_index] = max(
                    channel_max[channel_index], float(np.max(values[finite]))
                )
                channel_negative[channel_index] += int(
                    np.count_nonzero(values[finite] < 0)
                )
                channel_fractional[channel_index] += int(
                    np.count_nonzero(values[finite] != np.floor(values[finite]))
                )
                channel_positive_values[channel_index] += int(
                    np.count_nonzero(values[finite] > 0)
                )

        positives = batch[..., :BACKGROUND_CHANNEL_INDEX] > 0
        background_values = batch[..., BACKGROUND_CHANNEL_INDEX]
        background = background_values > 0
        background_nonbinary += int(
            np.count_nonzero(~np.isin(background_values, (0.0, 1.0)))
        )
        positive_count = np.count_nonzero(positives, axis=-1)
        any_positive = positive_count > 0
        overlap = positive_count > 1
        positive_background = any_positive & background
        background_only = ~any_positive & background
        positive_only = any_positive & ~background
        all_zero = ~any_positive & ~background
        complement_mismatch = positive_background | all_zero

        for local_index in range(stop - start):
            patch_index = start + local_index
            patch_masks = batch[local_index]
            patch_positives = positives[local_index]
            patch_overlap = overlap[local_index]
            patch_category_values = {
                "cross_class_overlap_pixels": int(np.count_nonzero(patch_overlap)),
                "positive_background_pixels": int(
                    np.count_nonzero(positive_background[local_index])
                ),
                "background_only_pixels": int(
                    np.count_nonzero(background_only[local_index])
                ),
                "positive_only_pixels": int(np.count_nonzero(positive_only[local_index])),
                "all_zero_pixels": int(np.count_nonzero(all_zero[local_index])),
                "void_pixels": int(np.count_nonzero(all_zero[local_index])),
                "background_complement_mismatch_pixels": int(
                    np.count_nonzero(complement_mismatch[local_index])
                ),
            }
            for key, count in patch_category_values.items():
                category_totals[key] += count
                category_patch_counts[key.replace("_pixels", "_patches")] += int(count > 0)

            class_id_sets: dict[int, set[float]] = {}
            patch_affected_instances: set[tuple[int, float]] = set()
            patch_affected_classes: set[int] = set()
            for class_index in range(BACKGROUND_CHANNEL_INDEX):
                values = np.unique(patch_masks[..., class_index])
                ids = {float(value) for value in values if np.isfinite(value) and value > 0}
                class_id_sets[class_index] = ids
                all_instances_by_class[class_index].update(
                    (patch_index, value) for value in ids
                )
                if np.any(patch_overlap):
                    overlap_values = np.unique(
                        patch_masks[..., class_index][
                            patch_overlap & patch_positives[..., class_index]
                        ]
                    )
                    affected = {
                        float(value)
                        for value in overlap_values
                        if np.isfinite(value) and value > 0
                    }
                    if affected:
                        affected_overlap_classes.add(class_index)
                        patch_affected_classes.add(class_index)
                        affected_instances_by_class[class_index].update(
                            (patch_index, value) for value in affected
                        )
                        affected_numeric_ids_by_class[class_index].update(affected)
                        patch_affected_instances.update(
                            (class_index, value) for value in affected
                        )
                        for instance_id in sorted(affected):
                            instance_pixels = (
                                patch_masks[..., class_index] == instance_id
                            )
                            overlap_pixels = instance_pixels & patch_overlap
                            overlapping_keys: set[tuple[int, float]] = set()
                            for other_class_index in range(BACKGROUND_CHANNEL_INDEX):
                                if other_class_index == class_index:
                                    continue
                                other_values = np.unique(
                                    patch_masks[..., other_class_index][
                                        overlap_pixels
                                        & patch_positives[..., other_class_index]
                                    ]
                                )
                                overlapping_keys.update(
                                    (other_class_index, float(other_value))
                                    for other_value in other_values
                                    if np.isfinite(other_value) and other_value > 0
                                )
                            affected_instance_rows.append(
                                {
                                    "fold_id": fold_id,
                                    "patch_index": patch_index,
                                    "class_index": class_index,
                                    "class_name": POSITIVE_CLASS_NAMES[class_index],
                                    "instance_id": _integer_id(instance_id),
                                    "total_instance_pixels": int(
                                        np.count_nonzero(instance_pixels)
                                    ),
                                    "cross_class_overlap_pixels": int(
                                        np.count_nonzero(overlap_pixels)
                                    ),
                                    "overlapping_instance_keys": ";".join(
                                        f"{other_class_index}:{_integer_id(other_id)}"
                                        for other_class_index, other_id in sorted(
                                            overlapping_keys
                                        )
                                    ),
                                }
                            )

            numeric_id_classes: dict[float, list[int]] = defaultdict(list)
            for class_index, ids in class_id_sets.items():
                for instance_id in ids:
                    numeric_id_classes[instance_id].append(class_index)
            duplicate_ids = {
                value: tuple(classes)
                for value, classes in numeric_id_classes.items()
                if len(classes) > 1
            }
            for instance_id, classes_with_id in sorted(duplicate_ids.items()):
                total_duplicate_keys.add((patch_index, instance_id))
                touched_by_overlap = any(
                    (class_index, instance_id) in patch_affected_instances
                    for class_index in classes_with_id
                )
                if touched_by_overlap:
                    overlap_touched_duplicate_keys.add((patch_index, instance_id))
                same_id_spatial_overlap = np.ones((height, width), dtype=bool)
                for class_index in classes_with_id:
                    same_id_spatial_overlap &= (
                        patch_masks[..., class_index] == instance_id
                    )
                duplicate_rows.append(
                    {
                        "fold_id": fold_id,
                        "patch_index": patch_index,
                        "instance_id": _integer_id(instance_id),
                        "class_indices": ";".join(str(value) for value in classes_with_id),
                        "class_names": ";".join(
                            POSITIVE_CLASS_NAMES[value] for value in classes_with_id
                        ),
                        "class_count": len(classes_with_id),
                        "touches_cross_class_overlap": touched_by_overlap,
                        "same_id_spatial_overlap_pixels_all_listed_classes": int(
                            np.count_nonzero(same_id_spatial_overlap)
                        ),
                    }
                )

            for left, right in combinations(range(BACKGROUND_CHANNEL_INDEX), 2):
                pair_overlap_pixels[(left, right)] += int(
                    np.count_nonzero(
                        patch_positives[..., left] & patch_positives[..., right]
                    )
                )

            if np.any(patch_overlap):
                for class_index in range(BACKGROUND_CHANNEL_INDEX):
                    active_values = patch_masks[..., class_index][
                        patch_overlap & patch_positives[..., class_index]
                    ]
                    overlap_active_value_count += int(active_values.size)
                    overlap_invalid_instance_value_count += int(
                        np.count_nonzero(
                            (~np.isfinite(active_values))
                            | (active_values <= 0)
                            | (active_values != np.floor(active_values))
                        )
                    )

            patch_rows.append(
                {
                    "fold_id": fold_id,
                    "patch_index": patch_index,
                    **patch_category_values,
                    "affected_instance_keys": len(patch_affected_instances),
                    "affected_class_count": len(patch_affected_classes),
                    "affected_class_indices": ";".join(
                        str(value) for value in sorted(patch_affected_classes)
                    ),
                    "cross_channel_duplicate_instance_ids": len(duplicate_ids),
                }
            )

    pixel_count = n_patches * height * width
    partition_total = sum(
        category_totals[key]
        for key in (
            "positive_background_pixels",
            "background_only_pixels",
            "positive_only_pixels",
            "all_zero_pixels",
        )
    )
    affected_instance_count = sum(
        len(values) for values in affected_instances_by_class.values()
    )
    result = {
        "fold_id": fold_id,
        "path": relative(path),
        "shape": [n_patches, height, width, channels],
        "dtype": str(masks.dtype),
        "patch_count": n_patches,
        "pixels_per_patch": height * width,
        "pixel_count": pixel_count,
        **dict(category_totals),
        **dict(category_patch_counts),
        "four_way_partition_pixel_count": partition_total,
        "four_way_partition_matches_total": partition_total == pixel_count,
        "affected_positive_instance_keys": affected_instance_count,
        "affected_positive_classes": len(affected_overlap_classes),
        "affected_positive_class_indices": sorted(affected_overlap_classes),
        "overlap_active_instance_value_count": overlap_active_value_count,
        "overlap_invalid_instance_value_count": overlap_invalid_instance_value_count,
        "overlap_touches_only_valid_positive_integer_instance_ids": (
            overlap_invalid_instance_value_count == 0
        ),
        "cross_channel_duplicate_instance_keys": len(total_duplicate_keys),
        "cross_channel_duplicate_instance_patches": len(
            {patch_index for patch_index, _ in total_duplicate_keys}
        ),
        "overlap_touched_duplicate_instance_keys": len(overlap_touched_duplicate_keys),
        "background_nonbinary_value_count": background_nonbinary,
        "channel_min": [None if np.isinf(value) else float(value) for value in channel_min],
        "channel_max": [None if np.isinf(value) else float(value) for value in channel_max],
        "channel_nonfinite_value_count": channel_nonfinite.tolist(),
        "channel_negative_value_count": channel_negative.tolist(),
        "channel_fractional_value_count": channel_fractional.tolist(),
        "channel_positive_value_count": channel_positive_values.tolist(),
        "class_metrics": [
            {
                "fold_id": fold_id,
                "class_index": class_index,
                "class_name": POSITIVE_CLASS_NAMES[class_index],
                "total_instance_keys": len(all_instances_by_class[class_index]),
                "overlap_affected_instance_keys": len(
                    affected_instances_by_class[class_index]
                ),
                "overlap_affected_distinct_numeric_id_values": len(
                    affected_numeric_ids_by_class[class_index]
                ),
            }
            for class_index in range(BACKGROUND_CHANNEL_INDEX)
        ],
        "class_pair_metrics": [
            {
                "fold_id": fold_id,
                "left_class_index": left,
                "left_class_name": POSITIVE_CLASS_NAMES[left],
                "right_class_index": right,
                "right_class_name": POSITIVE_CLASS_NAMES[right],
                "overlap_pixels": pair_overlap_pixels[(left, right)],
            }
            for left, right in combinations(range(BACKGROUND_CHANNEL_INDEX), 2)
        ],
        "affected_instance_rows": affected_instance_rows,
        "patch_rows": patch_rows,
        "duplicate_rows": duplicate_rows,
    }
    return result


def main() -> None:
    started_at = utc_now()
    started = time.perf_counter()
    script_sha256, _ = file_hashes(Path(__file__).resolve())

    input_paths = sorted(item for item in RAW_ROOT.rglob("*") if item.is_file())
    input_rows: list[dict[str, Any]] = []
    input_by_relative: dict[str, dict[str, Any]] = {}
    for path in input_paths:
        sha256, crc32 = file_hashes(path)
        npy = inspect_npy(path) if path.suffix.lower() == ".npy" else None
        file_stat = path.stat()
        row = {
            "relative_path": relative(path),
            "size_bytes": int(file_stat.st_size),
            "mtime_ns_before": int(file_stat.st_mtime_ns),
            "sha256": sha256,
            "crc32": crc32,
            "suffix": path.suffix.lower(),
            "array_shape": shape_text(npy["shape"] if npy else None),
            "array_dtype": npy["dtype"] if npy else "",
            "array_dtype_str": npy["dtype_str"] if npy else "",
            "array_nbytes": npy["nbytes"] if npy else "",
            "npy_header_bytes": npy["npy_header_bytes"] if npy else "",
            "fortran_order": npy["fortran_order"] if npy else "",
            "c_contiguous": npy["c_contiguous"] if npy else "",
        }
        input_rows.append(row)
        input_by_relative[row["relative_path"]] = row

    archive_rows, member_rows = audit_archives(input_by_relative)
    fold_results: list[dict[str, Any]] = []
    for fold_id in (1, 2, 3):
        mask_path = RAW_ROOT / f"Fold {fold_id}" / "masks" / f"fold{fold_id}" / "masks.npy"
        fold_results.append(scan_mask_fold(fold_id, mask_path))

    patch_rows = [row for fold in fold_results for row in fold.pop("patch_rows")]
    affected_instance_rows = [
        row for fold in fold_results for row in fold.pop("affected_instance_rows")
    ]
    duplicate_rows = [row for fold in fold_results for row in fold.pop("duplicate_rows")]
    class_rows = [row for fold in fold_results for row in fold.pop("class_metrics")]
    pair_rows = [row for fold in fold_results for row in fold.pop("class_pair_metrics")]

    definitions = {
        "positive_channels": "channels 0-4; active iff value > 0",
        "background_channel": "supplied channel 5; active iff value > 0; never repaired or replaced",
        "cross_class_overlap": "more than one positive channel active at a pixel",
        "positive_background": "at least one positive channel and supplied background both active",
        "background_only": "no positive channel active and supplied background active",
        "positive_only": "at least one positive channel active and supplied background inactive",
        "all_zero": "no positive channel active and supplied background inactive",
        "void": "operationally identical to all_zero (non-background, non-positive pixel)",
        "background_complement_mismatch": "positive_background OR all_zero",
        "instance_key_scope": "(fold_id, patch_index, positive_class_index, numeric instance_id)",
        "cross_channel_duplicate_instance_key": (
            "same positive numeric instance_id appears in two or more positive class channels "
            "within one patch; reported key scope is (fold_id, patch_index, numeric instance_id)"
        ),
    }
    totals = {
        key: sum(int(fold[key]) for fold in fold_results)
        for key in (
            "patch_count",
            "pixel_count",
            "cross_class_overlap_pixels",
            "cross_class_overlap_patches",
            "affected_positive_instance_keys",
            "positive_background_pixels",
            "positive_background_patches",
            "background_only_pixels",
            "background_only_patches",
            "positive_only_pixels",
            "positive_only_patches",
            "all_zero_pixels",
            "all_zero_patches",
            "void_pixels",
            "void_patches",
            "background_complement_mismatch_pixels",
            "background_complement_mismatch_patches",
            "cross_channel_duplicate_instance_keys",
            "cross_channel_duplicate_instance_patches",
            "overlap_touched_duplicate_instance_keys",
        )
    }
    totals["affected_positive_classes_union"] = sorted(
        {
            value
            for fold in fold_results
            for value in fold["affected_positive_class_indices"]
        }
    )
    totals["affected_positive_class_count_union"] = len(
        totals["affected_positive_classes_union"]
    )
    totals["archive_identity_all_match"] = all(
        row["identity_matches_expected"] for row in archive_rows
    )
    totals["archive_crc_all_pass"] = all(row["crc_test_passed"] for row in archive_rows)
    totals["archive_path_safety_all_pass"] = all(
        row["path_safety_passed"] for row in archive_rows
    )
    totals["archive_extracted_members_all_match_size_crc"] = all(
        row["extracted_members_match_size_crc"] for row in archive_rows
    )
    totals["mask_four_way_partitions_all_complete"] = all(
        fold["four_way_partition_matches_total"] for fold in fold_results
    )
    totals["overlap_all_touches_valid_positive_integer_instance_ids"] = all(
        fold["overlap_touches_only_valid_positive_integer_instance_ids"]
        for fold in fold_results
    )
    totals["affected_instance_rows"] = len(affected_instance_rows)
    if totals["affected_instance_rows"] != totals["affected_positive_instance_keys"]:
        raise RuntimeError(
            "affected-instance row count does not reconcile with the fold aggregates"
        )

    raw_stat_changes: list[dict[str, Any]] = []
    for row in input_rows:
        path = REPOSITORY_ROOT / row["relative_path"]
        current = path.stat()
        if (
            int(current.st_size) != int(row["size_bytes"])
            or int(current.st_mtime_ns) != int(row["mtime_ns_before"])
        ):
            raw_stat_changes.append(
                {
                    "relative_path": row["relative_path"],
                    "size_bytes_before": row["size_bytes"],
                    "size_bytes_after": int(current.st_size),
                    "mtime_ns_before": row["mtime_ns_before"],
                    "mtime_ns_after": int(current.st_mtime_ns),
                }
            )

    summary = {
        "schema_version": 2,
        "audit_kind": "independent_read_only_local_pannuke_qc",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "repository_root": str(REPOSITORY_ROOT),
        "raw_root": str(RAW_ROOT),
        "raw_data_modified": bool(raw_stat_changes),
        "raw_file_stat_changes": raw_stat_changes,
        "raw_file_stats_unchanged_during_audit": not raw_stat_changes,
        "script": {
            "relative_path": relative(Path(__file__).resolve()),
            "sha256": script_sha256,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "mask_batch_patches": MASK_BATCH_PATCHES,
            "hash_chunk_bytes": HASH_CHUNK_BYTES,
        },
        "definitions": definitions,
        "input_file_count": len(input_rows),
        "npy_file_count": sum(row["suffix"] == ".npy" for row in input_rows),
        "archive_results": archive_rows,
        "fold_results": fold_results,
        "totals": totals,
        "known_documentation_gap": (
            "STATUS.md:87 records that a dedicated machine-readable acquisition manifest "
            "is pending; this working audit is not a canonical acquisition manifest."
        ),
    }

    write_csv(
        OUTPUT_ROOT / "pannuke_input_inventory.csv",
        input_rows,
        list(input_rows[0]),
    )
    write_csv(
        OUTPUT_ROOT / "pannuke_zip_members.csv",
        member_rows,
        list(member_rows[0]),
    )
    write_csv(
        OUTPUT_ROOT / "pannuke_fold_metrics.csv",
        fold_results,
        [
            "fold_id",
            "path",
            "shape",
            "dtype",
            "patch_count",
            "pixels_per_patch",
            "pixel_count",
            "cross_class_overlap_pixels",
            "cross_class_overlap_patches",
            "affected_positive_instance_keys",
            "affected_positive_classes",
            "affected_positive_class_indices",
            "overlap_active_instance_value_count",
            "overlap_invalid_instance_value_count",
            "overlap_touches_only_valid_positive_integer_instance_ids",
            "positive_background_pixels",
            "positive_background_patches",
            "background_only_pixels",
            "background_only_patches",
            "positive_only_pixels",
            "positive_only_patches",
            "all_zero_pixels",
            "all_zero_patches",
            "void_pixels",
            "void_patches",
            "background_complement_mismatch_pixels",
            "background_complement_mismatch_patches",
            "cross_channel_duplicate_instance_keys",
            "cross_channel_duplicate_instance_patches",
            "overlap_touched_duplicate_instance_keys",
            "background_nonbinary_value_count",
            "four_way_partition_pixel_count",
            "four_way_partition_matches_total",
            "channel_min",
            "channel_max",
            "channel_nonfinite_value_count",
            "channel_negative_value_count",
            "channel_fractional_value_count",
            "channel_positive_value_count",
        ],
    )
    write_csv(OUTPUT_ROOT / "pannuke_patch_qc.csv", patch_rows, list(patch_rows[0]))
    write_csv(
        OUTPUT_ROOT / "pannuke_overlap_touching_instances.csv",
        affected_instance_rows,
        list(affected_instance_rows[0]),
    )
    write_csv(
        OUTPUT_ROOT / "pannuke_overlap_class_metrics.csv",
        class_rows,
        list(class_rows[0]),
    )
    write_csv(
        OUTPUT_ROOT / "pannuke_overlap_class_pairs.csv",
        pair_rows,
        list(pair_rows[0]),
    )
    duplicate_fields = [
        "fold_id",
        "patch_index",
        "instance_id",
        "class_indices",
        "class_names",
        "class_count",
        "touches_cross_class_overlap",
        "same_id_spatial_overlap_pixels_all_listed_classes",
    ]
    write_csv(
        OUTPUT_ROOT / "pannuke_cross_channel_duplicate_ids.csv",
        duplicate_rows,
        duplicate_fields,
    )
    write_json(OUTPUT_ROOT / "pannuke_qc_summary.json", summary)

    print(json.dumps({"totals": totals, "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
