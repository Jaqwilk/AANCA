"""Deterministic target-nucleus highlighting, cropping, and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

UInt8Array = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TargetCrop:
    """A resized target crop with its aligned binary target mask."""

    image: UInt8Array
    target_mask: BoolArray
    source_box: tuple[int, int, int, int]


def _validate_image_mask(image: NDArray[np.generic], target_mask: NDArray[np.generic]) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if target_mask.shape != image.shape[:2]:
        raise ValueError("target mask must match image height and width")
    if not np.asarray(target_mask, dtype=bool).any():
        raise ValueError("target mask is empty")


def mask_bbox(target_mask: NDArray[np.generic]) -> tuple[int, int, int, int]:
    """Return the half-open bounding box of a non-empty mask."""

    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("target mask must be a non-empty 2-D array")
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_perimeter(target_mask: NDArray[np.generic]) -> float:
    """Calculate a deterministic four-connected pixel-edge perimeter."""

    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("target mask must be two-dimensional")
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return float(vertical + horizontal)


def highlight_target(
    image: NDArray[np.generic],
    target_mask: NDArray[np.generic],
    *,
    context_brightness: float = 0.45,
) -> UInt8Array:
    """Preserve target pixels and dim context without encoding a class label."""

    _validate_image_mask(image, target_mask)
    if not 0.0 <= context_brightness <= 1.0:
        raise ValueError("context_brightness must lie in [0, 1]")
    rgb = np.asarray(image, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    highlighted = rgb * context_brightness
    highlighted[mask] = rgb[mask]
    return np.clip(np.rint(highlighted), 0, 255).astype(np.uint8)


def extract_target_crop(
    image: NDArray[np.generic],
    target_mask: NDArray[np.generic],
    *,
    output_size: int = 48,
    padding: int = 6,
) -> TargetCrop:
    """Extract a deterministic square crop around exactly the supplied target mask."""

    _validate_image_mask(image, target_mask)
    if output_size <= 0 or padding < 0:
        raise ValueError("output_size must be positive and padding non-negative")
    rgb = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(target_mask, dtype=bool)
    height, width = mask.shape
    x0, y0, x1, y1 = mask_bbox(mask)
    side = max(x1 - x0, y1 - y0) + 2 * padding
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    crop_x0 = max(0, int(np.floor(cx - side / 2.0)))
    crop_y0 = max(0, int(np.floor(cy - side / 2.0)))
    crop_x1 = min(width, crop_x0 + side)
    crop_y1 = min(height, crop_y0 + side)
    crop_x0 = max(0, crop_x1 - side)
    crop_y0 = max(0, crop_y1 - side)

    image_crop = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
    mask_crop = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    resized_image = np.asarray(
        Image.fromarray(image_crop).resize(
            (output_size, output_size), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    resized_mask = (
        np.asarray(
            Image.fromarray(mask_crop.astype(np.uint8) * 255).resize(
                (output_size, output_size), resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        > 0
    )
    if not resized_mask.any():
        raise RuntimeError("target identity was lost during crop resizing")
    return TargetCrop(
        image=resized_image,
        target_mask=resized_mask,
        source_box=(crop_x0, crop_y0, crop_x1, crop_y1),
    )


def extract_target_features(
    images: NDArray[np.generic], target_masks: NDArray[np.generic]
) -> FloatArray:
    """Extract deterministic target-specific colour and geometry audit features."""

    rgb = np.asarray(images)
    masks = np.asarray(target_masks, dtype=bool)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3]:
        raise ValueError("expected images (n,h,w,3) and masks (n,h,w)")
    rows: list[list[float]] = []
    for image, mask in zip(rgb, masks, strict=True):
        _validate_image_mask(image, mask)
        pixels = image[mask].astype(np.float64) / 255.0
        background = image[~mask].astype(np.float64) / 255.0
        x0, y0, x1, y1 = mask_bbox(mask)
        area = float(mask.sum())
        perimeter = mask_perimeter(mask)
        aspect = (x1 - x0) / max(y1 - y0, 1)
        ys, xs = np.nonzero(mask)
        height, width = mask.shape
        row = [
            *pixels.mean(axis=0).tolist(),
            *pixels.std(axis=0).tolist(),
            *(pixels.mean(axis=0) - background.mean(axis=0)).tolist(),
            area / (height * width),
            perimeter / (2.0 * (height + width)),
            aspect,
            float(xs.mean() / width),
            float(ys.mean() / height),
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def extract_target_colour_features(
    images: NDArray[np.generic], target_masks: NDArray[np.generic]
) -> FloatArray:
    """Extract colour-only target statistics with no morphology feature reuse."""

    rgb = np.asarray(images)
    masks = np.asarray(target_masks, dtype=bool)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3]:
        raise ValueError("expected images (n,h,w,3) and masks (n,h,w)")
    rows: list[list[float]] = []
    for image, mask in zip(rgb, masks, strict=True):
        _validate_image_mask(image, mask)
        target_pixels = image[mask].astype(np.float64) / 255.0
        context_pixels = image[~mask].astype(np.float64) / 255.0
        rows.append(
            [
                *target_pixels.mean(axis=0).tolist(),
                *target_pixels.std(axis=0).tolist(),
                *(target_pixels.mean(axis=0) - context_pixels.mean(axis=0)).tolist(),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def extract_morphology_features(target_masks: NDArray[np.generic]) -> FloatArray:
    """Extract a geometry-only feature space suitable for corruption generation."""

    masks = np.asarray(target_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("target_masks must have shape (n, height, width)")
    rows: list[list[float]] = []
    for mask in masks:
        if not mask.any():
            raise ValueError("target mask is empty")
        height, width = mask.shape
        x0, y0, x1, y1 = mask_bbox(mask)
        ys, xs = np.nonzero(mask)
        x_centered = xs - xs.mean()
        y_centered = ys - ys.mean()
        covariance = np.cov(np.vstack([x_centered, y_centered]), bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        major = float(np.sqrt(max(eigenvalues[-1], 0.0)))
        minor = float(np.sqrt(max(eigenvalues[0], 0.0)))
        eccentricity = np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-12)))
        area = float(mask.sum())
        rows.append(
            [
                area / (height * width),
                mask_perimeter(mask) / (2.0 * (height + width)),
                (x1 - x0) / max(y1 - y0, 1),
                float(eccentricity),
                area / max((x1 - x0) * (y1 - y0), 1),
            ]
        )
    return np.asarray(rows, dtype=np.float64)
