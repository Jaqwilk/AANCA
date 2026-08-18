"""Verified, deterministic PanNuke image assets for blinded expert review.

Every target is first resolved through the canonical manifest and revalidated
against its exact raw positive-class channel and instance ID by the shared crop
extractor.  The generated images are display aids only: a single fixed contour
colour is used for every class and source arrays are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image

from histo_audit.pannuke.exceptions import PanNukeSemanticsError
from histo_audit.pannuke.io import (
    ensure_derived_output_outside_raw,
    open_npy_mmap,
    sha256_file,
)
from histo_audit.pannuke.manifest import validate_manifest_invariants
from histo_audit.pannuke.models import PanNukeValidationResult
from histo_audit.pannuke.validation import verify_raw_inventory_unchanged
from histo_audit.representations.pannuke import (
    PanNukeCropConfig,
    extract_pannuke_crop_batch,
)
from histo_audit.utils.run_tracking import atomic_write_bytes, atomic_write_json, atomic_write_text

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTOUR_RGB = np.asarray((255, 0, 255), dtype=np.uint8)
_ASSET_ROLES = ("full_patch", "target_crop", "target_contour")


@dataclass(frozen=True, slots=True)
class PanNukeReviewerAssetsResult:
    """Published paths and provenance for one atomic reviewer-asset set."""

    output_directory: Path
    asset_manifest_csv: Path
    metadata_json: Path
    sample_count: int
    manifest_sha256: str
    raw_inventory_sha256: str
    crop_configuration_sha256: str
    asset_roles: tuple[str, ...] = _ASSET_ROLES

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result record."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


def _require_sha256(value: str, name: str) -> str:
    normalised = str(value).casefold()
    if _SHA256.fullmatch(normalised) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalised


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _channel_last_patch(
    array: NDArray[np.generic], index: int, channel_axis: int
) -> NDArray[np.generic]:
    patch = np.asarray(array[index])
    patch_axis = channel_axis - 1
    return np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch


def _uint8_rgb(image: NDArray[np.generic], *, sample_id: str) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{sample_id}: source patch is not channel-last RGB")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError(f"{sample_id}: source patch is not finite numeric RGB")
    converted = image.astype(np.float64)
    if float(converted.min()) < 0.0 or float(converted.max()) > 255.0:
        raise ValueError(f"{sample_id}: source RGB lies outside [0, 255]")
    return np.rint(converted).astype(np.uint8)


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    """Encode RGB pixels without time-dependent metadata."""

    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("review asset must be uint8 RGB")
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def _overlay_contour(
    rgb: NDArray[np.uint8],
    contour_xy: NDArray[np.int32],
) -> NDArray[np.uint8]:
    if contour_xy.ndim != 2 or contour_xy.shape[1] != 2 or not len(contour_xy):
        raise ValueError("exact target contour must be a non-empty (x, y) matrix")
    result = rgb.copy()
    x = contour_xy[:, 0].astype(np.int64, copy=False)
    y = contour_xy[:, 1].astype(np.int64, copy=False)
    if (
        np.any(x < 0)
        or np.any(y < 0)
        or np.any(x >= result.shape[1])
        or np.any(y >= result.shape[0])
    ):
        raise ValueError("exact target contour lies outside its image")
    result[y, x] = _CONTOUR_RGB
    return result


def _mask_contour_xy(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("target crop contour is empty")
    return np.column_stack((x, y)).astype(np.int32, copy=False)


def _inventory_sha256(validation: PanNukeValidationResult) -> str:
    return _canonical_sha256([record.as_dict() for record in validation.inventory])


def _inventory_by_path(validation: PanNukeValidationResult) -> dict[str, str]:
    return {
        record.relative_path.replace("\\", "/"): record.sha256 for record in validation.inventory
    }


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("asset manifest cannot be empty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise RuntimeError("asset manifest row schemas differ")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _safe_asset_stem(sample_id: str) -> str:
    return f"sample-{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:20]}"


def _validate_staged_assets(
    root: Path,
    rows: Sequence[dict[str, Any]],
    *,
    expected_sample_ids: tuple[str, ...],
) -> None:
    if tuple(str(row["sample_id"]) for row in rows) != expected_sample_ids:
        raise RuntimeError("staged asset manifest sample order changed")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise RuntimeError("staged asset manifest contains duplicate sample IDs")
    seen_paths: set[str] = set()
    for row in rows:
        if str(row["target_identity_verified"]).casefold() != "true":
            raise RuntimeError("staged asset row lacks target-identity verification")
        disconnected = str(row["disconnected_instance_id"]).casefold() == "true"
        component_count = int(row["raw_target_component_count"])
        projected_union_count = int(row["projected_union_component_count"])
        fallback_count = int(row["projection_fallback_component_count"])
        collision_pixel_count = int(row["projection_collision_pixel_count"])
        collision_excess_count = int(row["projection_collision_excess_count"])
        adjacency_pair_count = int(row["projection_adjacency_pair_count"])
        topology_changed = str(row["projection_topology_changed"]).casefold() == "true"
        mask_sha256 = str(row["disconnected_raw_target_mask_sha256"])
        if (
            component_count < 1
            or projected_union_count < 1
            or projected_union_count > component_count
            or not 0 <= fallback_count <= component_count
            or collision_pixel_count < 0
            or collision_excess_count < collision_pixel_count
            or ((collision_pixel_count == 0) != (collision_excess_count == 0))
            or not 0 <= adjacency_pair_count <= component_count * (component_count - 1) // 2
            or topology_changed != (projected_union_count != component_count)
            or disconnected != (component_count > 1)
            or (disconnected and _SHA256.fullmatch(mask_sha256) is None)
            or (not disconnected and mask_sha256)
        ):
            raise RuntimeError("staged asset row has invalid disconnected-instance evidence")
        for role in _ASSET_ROLES:
            raw_path = str(row[f"{role}_path"])
            if raw_path in seen_paths:
                raise RuntimeError("staged asset paths are not unique")
            seen_paths.add(raw_path)
            path = (root / raw_path).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise RuntimeError("staged asset is missing or escapes the output directory")
            if path.suffix.casefold() != ".png":
                raise RuntimeError("staged review asset is not PNG")
            if sha256_file(path) != row[f"{role}_sha256"]:
                raise RuntimeError("staged review asset checksum differs from manifest")
            with Image.open(path) as image:
                image.verify()


def build_pannuke_reviewer_assets(
    validation: PanNukeValidationResult,
    manifest_path: str | Path,
    sample_ids: Sequence[str],
    output_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    crop_config: PanNukeCropConfig | None = None,
) -> PanNukeReviewerAssetsResult:
    """Create three exact-target PNG assets per explicitly requested sample.

    Publication uses a new-directory transaction.  Any validation or write
    failure removes the staging tree and leaves ``output_directory`` absent.
    """

    destination = ensure_derived_output_outside_raw(
        output_directory,
        validation.root,
        purpose="PanNuke reviewer-asset output directory",
    )
    if os.path.lexists(destination):
        raise FileExistsError(f"reviewer asset directory already exists: {destination}")
    requested = tuple(str(value) for value in sample_ids)
    if not requested or any(not value for value in requested):
        raise ValueError("sample_ids must be explicit, non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("sample_ids must be unique")
    source_manifest = Path(manifest_path).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"canonical manifest does not exist: {source_manifest}")
    actual_manifest_sha = sha256_file(source_manifest)
    if actual_manifest_sha != _require_sha256(
        expected_manifest_sha256, "expected manifest SHA-256"
    ):
        raise ValueError("canonical manifest SHA-256 differs from the frozen binding")
    table = pq.read_table(source_manifest)
    if sha256_file(source_manifest) != actual_manifest_sha:
        raise PanNukeSemanticsError("canonical manifest changed during reviewer-asset parsing")
    validate_manifest_invariants(table)
    frame = table.to_pandas().set_index("sample_id", drop=False)
    missing = [sample_id for sample_id in requested if sample_id not in frame.index]
    if missing:
        raise KeyError(f"canonical manifest lacks requested sample IDs: {missing[:5]}")
    selected = frame.loc[list(requested)]
    if len(selected) != len(requested):
        raise ValueError("canonical manifest sample selection is not one-to-one")

    settings = crop_config or PanNukeCropConfig()
    settings.validate()
    crop_config_sha = _canonical_sha256(asdict(settings))
    crops = extract_pannuke_crop_batch(
        validation,
        source_manifest,
        sample_ids=requested,
        config=settings,
        eligibility_scope="review_only",
    )
    crops.validate()
    if tuple(str(value) for value in crops.sample_ids) != requested:
        raise RuntimeError("verified crop extractor changed requested sample order")
    if crops.metadata.get("manifest_sha256") != actual_manifest_sha:
        raise RuntimeError("crop extractor manifest binding differs")
    raw_inventory_sha = _inventory_sha256(validation)
    if crops.metadata.get("raw_inventory_sha256") != raw_inventory_sha:
        raise RuntimeError("crop extractor raw-inventory binding differs")
    projection_policy = crops.metadata.get("target_mask_projection")
    if not isinstance(projection_policy, dict):
        raise RuntimeError("crop extractor lacks target-mask projection provenance")
    disconnected_by_sample = {
        str(record["sample_id"]): dict(record)
        for record in projection_policy.get("disconnected_instances", [])
    }
    if len(disconnected_by_sample) != int(projection_policy.get("disconnected_instance_count", -1)):
        raise RuntimeError("crop extractor disconnected-instance provenance is inconsistent")

    def verify_fresh_inputs_and_destination() -> None:
        ensure_derived_output_outside_raw(
            destination,
            validation.root,
            purpose="PanNuke reviewer-asset output directory",
        )
        if not source_manifest.is_file() or sha256_file(source_manifest) != actual_manifest_sha:
            raise PanNukeSemanticsError(
                "canonical manifest changed during reviewer-asset generation"
            )
        verify_raw_inventory_unchanged(validation)

    verify_fresh_inputs_and_destination()

    folds = {fold.fold_id: fold for fold in validation.folds}
    raw_images = {fold_id: open_npy_mmap(fold.image_path) for fold_id, fold in folds.items()}
    inventory = _inventory_by_path(validation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staged = staging_root / "reviewer_assets"
    staged_assets = staged / "assets"
    staged_assets.mkdir(parents=True)
    published = False
    published_identity: tuple[int, int] | None = None
    try:
        rows: list[dict[str, Any]] = []
        used_stems: set[str] = set()
        for index, sample_id in enumerate(requested):
            row = selected.iloc[index]
            disconnected_record = disconnected_by_sample.get(sample_id)
            component_count = int(crops.raw_component_counts[index])
            if (component_count > 1) != (disconnected_record is not None):
                raise RuntimeError("crop component vector and disconnected-instance ledger differ")
            fold_id = int(row["official_fold"])
            fold = folds.get(fold_id)
            if fold is None:
                raise ValueError(f"{sample_id}: canonical manifest references unknown fold")
            patch_index = int(row["source_patch_index"])
            source_rgb = _uint8_rgb(
                _channel_last_patch(raw_images[fold_id], patch_index, fold.image_channel_axis),
                sample_id=sample_id,
            )
            source_contour = crops.source_contours_xy[index]
            full_patch = _overlay_contour(source_rgb, source_contour)
            target_crop = np.asarray(crops.context_rgb[index], dtype=np.uint8)
            crop_contour_xy = _mask_contour_xy(crops.target_contour_masks[index])
            target_contour = _overlay_contour(target_crop, crop_contour_xy)
            stem = _safe_asset_stem(sample_id)
            if stem in used_stems:
                raise RuntimeError("sample-ID asset filename hash collision")
            used_stems.add(stem)
            relative_paths: dict[str, str] = {}
            asset_hashes: dict[str, str] = {}
            for role, pixels in (
                ("full_patch", full_patch),
                ("target_crop", target_crop),
                ("target_contour", target_contour),
            ):
                relative = Path("assets") / f"{stem}_{role}.png"
                payload = _png_bytes(pixels)
                atomic_write_bytes(staged / relative, payload)
                relative_paths[role] = relative.as_posix()
                asset_hashes[role] = hashlib.sha256(payload).hexdigest()

            source_image_relative = str(row["source_image_path"]).replace("\\", "/")
            source_mask_relative = str(row["source_mask_path"]).replace("\\", "/")
            if source_image_relative not in inventory or source_mask_relative not in inventory:
                raise ValueError(f"{sample_id}: canonical raw paths are absent from inventory")
            rows.append(
                {
                    "sample_id": sample_id,
                    "observed_label": int(row["observed_label"]),
                    "official_fold": fold_id,
                    "group_id": str(row["group_id"]),
                    "instance_channel_index": int(row["instance_channel_index"]),
                    "instance_id": int(row["instance_id"]),
                    "target_identity_verified": "true",
                    "disconnected_instance_id": str(component_count > 1).lower(),
                    "raw_target_component_count": component_count,
                    "projected_union_component_count": int(
                        crops.projected_union_component_counts[index]
                    ),
                    "projection_fallback_component_count": int(
                        crops.projection_fallback_component_counts[index]
                    ),
                    "projection_collision_pixel_count": int(
                        crops.projection_collision_pixel_counts[index]
                    ),
                    "projection_collision_excess_count": int(
                        crops.projection_collision_excess_counts[index]
                    ),
                    "projection_adjacency_pair_count": int(
                        crops.projection_adjacency_pair_counts[index]
                    ),
                    "projection_topology_changed": str(
                        bool(crops.projection_topology_changed[index])
                    ).lower(),
                    "disconnected_raw_target_mask_sha256": (
                        str(disconnected_record["raw_target_mask_sha256"])
                        if disconnected_record is not None
                        else ""
                    ),
                    "full_patch_path": relative_paths["full_patch"],
                    "full_patch_sha256": asset_hashes["full_patch"],
                    "target_crop_path": relative_paths["target_crop"],
                    "target_crop_sha256": asset_hashes["target_crop"],
                    "target_contour_path": relative_paths["target_contour"],
                    "target_contour_sha256": asset_hashes["target_contour"],
                    "source_image_path": source_image_relative,
                    "source_image_sha256": inventory[source_image_relative],
                    "source_mask_path": source_mask_relative,
                    "source_mask_sha256": inventory[source_mask_relative],
                    "raw_inventory_sha256": raw_inventory_sha,
                    "manifest_sha256": actual_manifest_sha,
                    "crop_configuration_sha256": crop_config_sha,
                    "source_annotations_modified": "false",
                }
            )

        asset_manifest = staged / "asset_manifest.csv"
        metadata_path = staged / "asset_metadata.json"
        atomic_write_text(asset_manifest, _csv_text(rows))
        atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "dataset": "PanNuke",
                "purpose": "display assets for blinded expert annotation review",
                "sample_count": len(rows),
                "sample_ids_sha256": _canonical_sha256(list(requested)),
                "asset_roles": list(_ASSET_ROLES),
                "manifest_sha256": actual_manifest_sha,
                "raw_inventory_sha256": raw_inventory_sha,
                "crop_configuration": asdict(settings),
                "crop_configuration_sha256": crop_config_sha,
                "target_identity_policy": crops.metadata["target_identity_policy"],
                "target_mask_projection": projection_policy,
                "target_identity_verified_for_every_sample": True,
                "contour_policy": {
                    "colour_rgb": _CONTOUR_RGB.tolist(),
                    "colour_is_constant_across_classes": True,
                    "full_patch": "exact source-coordinate target boundary pixels",
                    "target_contour": (
                        "boundary pixels of the component-covering projected target mask"
                    ),
                },
                "class_encoded_by_contour_colour": False,
                "source_annotations_modified": False,
                "automatic_source_annotation_modification": False,
            },
        )
        _validate_staged_assets(
            staged,
            rows,
            expected_sample_ids=requested,
        )
        verify_fresh_inputs_and_destination()
        if os.path.lexists(destination):
            raise FileExistsError(f"reviewer asset directory already exists: {destination}")
        staged_stat = staged.stat()
        os.replace(staged, destination)
        published = True
        published_identity = (int(staged_stat.st_dev), int(staged_stat.st_ino))
        verify_fresh_inputs_and_destination()
    except BaseException as error:
        if published and os.path.lexists(destination):
            destination_stat = destination.stat(follow_symlinks=False)
            destination_identity = (
                int(destination_stat.st_dev),
                int(destination_stat.st_ino),
            )
            if destination_identity != published_identity:
                raise RuntimeError(
                    "reviewer-asset publication failed after destination ownership changed; "
                    "foreign output preserved"
                ) from error
            shutil.rmtree(destination)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    result = PanNukeReviewerAssetsResult(
        output_directory=destination,
        asset_manifest_csv=destination / "asset_manifest.csv",
        metadata_json=destination / "asset_metadata.json",
        sample_count=len(requested),
        manifest_sha256=actual_manifest_sha,
        raw_inventory_sha256=raw_inventory_sha,
        crop_configuration_sha256=crop_config_sha,
    )
    if not result.asset_manifest_csv.is_file() or not result.metadata_json.is_file():
        raise RuntimeError("published PanNuke reviewer assets are incomplete")
    return result


__all__ = ["PanNukeReviewerAssetsResult", "build_pannuke_reviewer_assets"]
