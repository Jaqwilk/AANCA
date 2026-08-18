"""Deterministic target-specific engineered image features.

The extractor deliberately receives an RGB image and an exact binary target mask.
No class-label input is accepted.  Features therefore describe the indicated
nucleus and its local appearance without encoding an annotation suggestion.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]
from skimage.feature import hog
from skimage.measure import euler_number
from skimage.morphology import convex_hull_image
from skimage.transform import resize

from histo_audit.utils.run_tracking import sha256_file

from .cache_provenance import (
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    canonical_sha256,
    explicit_unlearned_weights_sha256,
)

FloatMatrix = NDArray[np.float64]

_CHANNEL_NAMES = ("red", "green", "blue")
_TEXTURE_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "angular_second_moment",
    "energy",
    "correlation",
    "entropy",
)
_TEXTURE_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True, slots=True)
class EngineeredFeatureSet:
    """A feature matrix paired with its stable, ordered feature names."""

    values: FloatMatrix
    names: tuple[str, ...]

    def validate(self, expected_samples: int | None = None) -> None:
        """Validate alignment, dimensionality, and finite feature values."""

        if self.values.ndim != 2:
            raise ValueError("engineered feature values must be two-dimensional")
        if expected_samples is not None and self.values.shape[0] != expected_samples:
            raise ValueError("engineered feature rows do not align with samples")
        if self.values.shape[1] != len(self.names):
            raise ValueError("engineered feature names do not align with columns")
        if len(set(self.names)) != len(self.names):
            raise ValueError("engineered feature names must be unique")
        if not np.isfinite(self.values).all():
            raise ValueError("engineered features contain non-finite values")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _normalise_rgb(images: NDArray[np.generic]) -> FloatMatrix:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] == 0:
        raise ValueError("images must have non-empty shape (n, height, width, 3)")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("images must contain numeric RGB values")
    if not np.isfinite(array).all():
        raise ValueError("images contain non-finite values")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.min < 0 and int(array.min()) < 0:
            raise ValueError("integer RGB values must be non-negative")
        scale = float(info.max)
        normalised = array.astype(np.float64) / scale
    else:
        normalised = array.astype(np.float64)
        minimum = float(normalised.min())
        maximum = float(normalised.max())
        if minimum < 0.0:
            raise ValueError("floating-point RGB values must be non-negative")
        if maximum <= 1.0:
            pass
        elif maximum <= 255.0:
            normalised /= 255.0
        else:
            raise ValueError("floating-point RGB values must lie in [0, 1] or [0, 255]")
    return np.clip(normalised, 0.0, 1.0)


def _validate_masks(
    target_masks: NDArray[np.generic], image_shape: tuple[int, int, int]
) -> NDArray[np.bool_]:
    masks = np.asarray(target_masks, dtype=bool)
    if masks.shape != image_shape:
        raise ValueError("target_masks must align with image sample, height, and width axes")
    if not np.all(masks.reshape(masks.shape[0], -1).any(axis=1)):
        raise ValueError("every target mask must contain at least one pixel")
    return masks


def _bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _perimeter(mask: NDArray[np.bool_]) -> float:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return float(vertical + horizontal)


def _morphology_and_mask_features(mask: NDArray[np.bool_]) -> list[float]:
    height, width = mask.shape
    y0, x0, y1, x1 = _bbox(mask)
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    perimeter = _perimeter(mask)
    bbox_area = float((y1 - y0) * (x1 - x0))

    coordinates = np.vstack((xs.astype(np.float64), ys.astype(np.float64)))
    covariance = np.cov(coordinates, bias=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    minor_variance, major_variance = (float(value) for value in eigenvalues)
    eccentricity = np.sqrt(
        max(0.0, 1.0 - minor_variance / max(major_variance, np.finfo(float).eps))
    )

    convex_hull_area = float(convex_hull_image(mask).sum())
    labels, component_count = ndimage.label(mask)
    del labels
    hole_count = max(0, int(component_count) - int(euler_number(mask, connectivity=2)))
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & ~eroded
    distances = ndimage.distance_transform_edt(mask)
    target_distances = distances[mask]
    image_border = np.zeros_like(mask)
    image_border[[0, -1], :] = True
    image_border[:, [0, -1]] = True

    return [
        area / float(height * width),
        float(np.sqrt(4.0 * area / np.pi)),
        float(np.sqrt(4.0 * area / np.pi) / max(height, width)),
        perimeter / float(2 * (height + width)),
        float((x1 - x0) / max(y1 - y0, 1)),
        area / max(bbox_area, 1.0),
        float(eccentricity),
        area / max(convex_hull_area, 1.0),
        float((perimeter * perimeter) / max(4.0 * np.pi * area, np.finfo(float).eps)),
        float(xs.mean() / max(width - 1, 1)),
        float(ys.mean() / max(height - 1, 1)),
        float(component_count),
        float(hole_count),
        float(np.count_nonzero(mask & image_border) / max(area, 1.0)),
        float(boundary.sum() / max(area, 1.0)),
        float(target_distances.mean() / max(height, width)),
        float(target_distances.max() / max(height, width)),
    ]


def _masked_glcm_properties(gray: FloatMatrix, mask: NDArray[np.bool_], levels: int) -> list[float]:
    quantised = np.minimum((gray * levels).astype(np.int64), levels - 1)
    direction_values: list[list[float]] = []
    for dy, dx in _TEXTURE_OFFSETS:
        source_y = slice(max(0, -dy), min(mask.shape[0], mask.shape[0] - dy))
        source_x = slice(max(0, -dx), min(mask.shape[1], mask.shape[1] - dx))
        target_y = slice(max(0, dy), min(mask.shape[0], mask.shape[0] + dy))
        target_x = slice(max(0, dx), min(mask.shape[1], mask.shape[1] + dx))
        valid = mask[source_y, source_x] & mask[target_y, target_x]
        matrix = np.zeros((levels, levels), dtype=np.float64)
        if valid.any():
            first = quantised[source_y, source_x][valid]
            second = quantised[target_y, target_x][valid]
            np.add.at(matrix, (first, second), 1.0)
            np.add.at(matrix, (second, first), 1.0)
        else:
            # A one-pixel target has no valid neighbour pair.  A unit diagonal
            # distribution yields finite neutral texture descriptors.
            value = int(quantised[mask][0])
            matrix[value, value] = 1.0
        probability = matrix / matrix.sum()
        i, j = np.indices(probability.shape, dtype=np.float64)
        delta = np.abs(i - j)
        contrast = float(np.sum(probability * delta**2))
        dissimilarity = float(np.sum(probability * delta))
        homogeneity = float(np.sum(probability / (1.0 + delta**2)))
        angular_second_moment = float(np.sum(probability**2))
        energy = float(np.sqrt(angular_second_moment))
        mean_i = float(np.sum(probability * i))
        mean_j = float(np.sum(probability * j))
        std_i = float(np.sqrt(np.sum(probability * (i - mean_i) ** 2)))
        std_j = float(np.sqrt(np.sum(probability * (j - mean_j) ** 2)))
        denominator = std_i * std_j
        correlation = (
            float(np.sum(probability * (i - mean_i) * (j - mean_j)) / denominator)
            if denominator > np.finfo(float).eps
            else 1.0
        )
        nonzero = probability[probability > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        direction_values.append(
            [
                contrast,
                dissimilarity,
                homogeneity,
                angular_second_moment,
                energy,
                correlation,
                entropy,
            ]
        )
    direction_matrix = np.asarray(direction_values, dtype=np.float64)
    return [*direction_matrix.mean(axis=0).tolist(), *direction_matrix.std(axis=0).tolist()]


def _hog_vector(
    gray: FloatMatrix,
    mask: NDArray[np.bool_],
    *,
    hog_size: int,
    orientations: int,
    pixels_per_cell: int,
) -> FloatMatrix:
    y0, x0, y1, x1 = _bbox(mask)
    target = np.zeros_like(gray)
    target[mask] = gray[mask]
    target_crop = target[y0:y1, x0:x1]
    resized = resize(
        target_crop,
        (hog_size, hog_size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    vector = hog(
        resized,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(vector, dtype=np.float64)


def engineered_feature_names(
    *,
    histogram_bins: int = 8,
    hog_size: int = 32,
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 8,
    texture_levels: int = 16,
) -> tuple[str, ...]:
    """Return the stable column order for :func:`extract_engineered_features`."""

    _validate_configuration(
        histogram_bins,
        hog_size,
        hog_orientations,
        hog_pixels_per_cell,
        texture_levels,
    )
    names = [
        "morphology.area_fraction",
        "morphology.equivalent_diameter_pixels",
        "morphology.normalised_equivalent_diameter",
        "morphology.normalised_perimeter",
        "morphology.bbox_aspect_ratio",
        "morphology.extent",
        "morphology.eccentricity",
        "morphology.solidity",
        "morphology.inverse_compactness",
        "morphology.centroid_x",
        "morphology.centroid_y",
        "mask.connected_component_count",
        "mask.hole_count",
        "mask.image_border_fraction",
        "mask.boundary_fraction",
        "mask.mean_internal_distance",
        "mask.max_internal_distance",
    ]
    names.extend(
        f"colour.target_{channel}_histogram_bin_{index:02d}"
        for channel in _CHANNEL_NAMES
        for index in range(histogram_bins)
    )
    names.extend(
        [
            "intensity.target_mean",
            "intensity.target_std",
            "intensity.target_min",
            "intensity.target_max",
            "intensity.target_quantile_10",
            "intensity.target_quantile_25",
            "intensity.target_median",
            "intensity.target_quantile_75",
            "intensity.target_quantile_90",
            "intensity.context_mean",
            "intensity.context_std",
            "intensity.target_minus_context_mean",
            "intensity.boundary_mean",
            "intensity.boundary_std",
            "intensity.boundary_min",
            "intensity.boundary_max",
            "intensity.boundary_minus_interior_mean",
            "intensity.boundary_minus_context_mean",
        ]
    )
    for aggregate in ("mean", "std"):
        names.extend(
            f"texture.glcm_{property_name}_{aggregate}" for property_name in _TEXTURE_PROPERTIES
        )
    dummy = np.zeros((hog_size, hog_size), dtype=np.float64)
    hog_length = len(
        hog(
            dummy,
            orientations=hog_orientations,
            pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
    )
    names.extend(f"hog.target_{index:04d}" for index in range(hog_length))
    return tuple(names)


def _validate_configuration(
    histogram_bins: int,
    hog_size: int,
    hog_orientations: int,
    hog_pixels_per_cell: int,
    texture_levels: int,
) -> None:
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")
    if hog_size < 2 * hog_pixels_per_cell or hog_pixels_per_cell <= 0:
        raise ValueError("hog_size must contain at least two positive HOG cells per axis")
    if hog_orientations <= 0:
        raise ValueError("hog_orientations must be positive")
    if not 2 <= texture_levels <= 256:
        raise ValueError("texture_levels must lie in [2, 256]")


def build_engineered_feature_set(
    images: NDArray[np.generic],
    target_masks: NDArray[np.generic],
    *,
    histogram_bins: int = 8,
    hog_size: int = 32,
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 8,
    texture_levels: int = 16,
) -> EngineeredFeatureSet:
    """Extract deterministic morphology, colour, HOG, texture, and mask features."""

    _validate_configuration(
        histogram_bins,
        hog_size,
        hog_orientations,
        hog_pixels_per_cell,
        texture_levels,
    )
    rgb = _normalise_rgb(images)
    masks = _validate_masks(target_masks, rgb.shape[:3])
    rows: list[list[float]] = []
    for image, mask in zip(rgb, masks, strict=True):
        target_pixels = image[mask]
        gray = np.tensordot(image, np.asarray([0.2989, 0.5870, 0.1140]), axes=([-1], [0]))
        target_intensity = gray[mask]
        context_intensity = gray[~mask]
        if context_intensity.size == 0:
            context_intensity = target_intensity
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        boundary_intensity = gray[mask & ~eroded]
        interior_intensity = gray[eroded]
        if interior_intensity.size == 0:
            interior_intensity = target_intensity

        row = _morphology_and_mask_features(mask)
        for channel in range(3):
            histogram, _ = np.histogram(
                target_pixels[:, channel], bins=histogram_bins, range=(0.0, 1.0)
            )
            row.extend((histogram.astype(np.float64) / target_pixels.shape[0]).tolist())
        quantiles = np.quantile(target_intensity, [0.10, 0.25, 0.50, 0.75, 0.90])
        row.extend(
            [
                float(target_intensity.mean()),
                float(target_intensity.std()),
                float(target_intensity.min()),
                float(target_intensity.max()),
                *quantiles.astype(np.float64).tolist(),
                float(context_intensity.mean()),
                float(context_intensity.std()),
                float(target_intensity.mean() - context_intensity.mean()),
                float(boundary_intensity.mean()),
                float(boundary_intensity.std()),
                float(boundary_intensity.min()),
                float(boundary_intensity.max()),
                float(boundary_intensity.mean() - interior_intensity.mean()),
                float(boundary_intensity.mean() - context_intensity.mean()),
            ]
        )
        row.extend(_masked_glcm_properties(gray, mask, texture_levels))
        row.extend(
            _hog_vector(
                gray,
                mask,
                hog_size=hog_size,
                orientations=hog_orientations,
                pixels_per_cell=hog_pixels_per_cell,
            ).tolist()
        )
        rows.append(row)

    feature_set = EngineeredFeatureSet(
        values=np.asarray(rows, dtype=np.float64),
        names=engineered_feature_names(
            histogram_bins=histogram_bins,
            hog_size=hog_size,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            texture_levels=texture_levels,
        ),
    )
    feature_set.validate(expected_samples=len(rgb))
    return feature_set


def extract_engineered_features(
    images: NDArray[np.generic],
    target_masks: NDArray[np.generic],
    **configuration: int,
) -> FloatMatrix:
    """Return only the engineered matrix for estimator-friendly call sites."""

    return build_engineered_feature_set(images, target_masks, **configuration).values


def select_target_morphometrics(features: EngineeredFeatureSet) -> EngineeredFeatureSet:
    """Select the frozen target-morphology columns in their existing order."""

    features.validate()
    indices = tuple(
        index for index, name in enumerate(features.names) if name.startswith("morphology.")
    )
    if not indices:
        raise ValueError("engineered feature set contains no target morphometrics")
    selected = EngineeredFeatureSet(
        values=np.asarray(features.values[:, indices], dtype=np.float64),
        names=tuple(features.names[index] for index in indices),
    )
    selected.validate(expected_samples=len(features.values))
    return selected


def save_engineered_feature_cache(
    features: EngineeredFeatureSet,
    sample_ids: NDArray[np.str_],
    destination: str | Path,
    *,
    manifest_sha256: str,
    raw_inventory_sha256: str,
    analysis_eligibility: Mapping[str, object] | None = None,
    target_mask_projection: Mapping[str, object] | None = None,
    source_crop_cache_binding: Mapping[str, object] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Persist a stage-eligible engineered cache with final frozen provenance."""

    identifiers = np.asarray(sample_ids, dtype=np.str_)
    features.validate(expected_samples=len(identifiers))
    names = np.asarray(features.names, dtype=np.str_)
    weight_identifier = "unlearned:no_learned_weights_engineered_target_features_v1"
    projection = dict(target_mask_projection) if target_mask_projection is not None else None
    projection_sha256 = canonical_sha256(projection) if projection is not None else None
    projection_semantic_sha256 = None
    if projection is not None:
        projection_semantic_sha256 = projection.get("semantic_sha256")
        semantic_payload = dict(projection)
        semantic_payload.pop("semantic_sha256", None)
        if projection_semantic_sha256 != canonical_sha256(semantic_payload):
            raise ValueError("target-mask projection semantic SHA-256 is invalid")
    crop_binding = (
        dict(source_crop_cache_binding) if source_crop_cache_binding is not None else None
    )
    crop_binding_sha256 = canonical_sha256(crop_binding) if crop_binding is not None else None
    preprocessing_identifier = (
        "pannuke_context_rgb_component_covering_projected_mask_engineered_v2"
        if projection is not None
        else "pannuke_context_rgb_exact_target_mask_engineered_v1"
    )
    preprocessing = {
        "identifier": preprocessing_identifier,
        "rgb_normalisation": "integer dtype maximum or floating [0,1]/[0,255] to float64 [0,1]",
        "target_mask": (
            "component-covering projected binary target union; morphology is projected, "
            "while exact raw geometry remains bound in the crop cache"
            if projection is not None
            else "exact aligned binary target instance mask"
        ),
        "feature_column_order": list(features.names),
        **({"target_mask_projection": projection} if projection is not None else {}),
    }
    encoder_metadata = {
        "feature_count": len(features.names),
        "feature_names_sha256": canonical_sha256(list(features.names)),
        "feature_families": [
            "morphology",
            "mask",
            "colour",
            "intensity",
            "texture",
            "hog",
        ],
        "learned_parameters": False,
        "output_dtype": str(features.values.dtype),
        **(
            {
                "source_crop_cache_binding": crop_binding,
                "source_crop_cache_binding_sha256": crop_binding_sha256,
            }
            if crop_binding is not None
            else {}
        ),
    }
    encoder_implementation = {
        "module": "histo_audit.representations.engineered",
        "entrypoint": "build_engineered_feature_set",
        "source_file_sha256": sha256_file(Path(__file__)),
    }
    cache_recipe = {
        "identifier": "engineered_feature_npz_v2",
        "array_keys": ["names", "sample_ids", "values"],
        "column_order_sha256": canonical_sha256(list(features.names)),
        "sample_alignment": "exact ordered sample_ids axis 0",
        "pickle_allowed": False,
        **(
            {"source_crop_cache_binding_sha256": crop_binding_sha256}
            if crop_binding is not None
            else {}
        ),
    }
    metadata = build_frozen_cache_metadata(
        base_metadata={
            "schema_version": 1,
            "feature_count": len(features.names),
            "feature_families": encoder_metadata["feature_families"],
            "crop_manifest_sha256": manifest_sha256,
            "source_annotations_modified": False,
            **(
                {
                    "source_crop_cache_binding": crop_binding,
                    "source_crop_cache_binding_sha256": crop_binding_sha256,
                }
                if crop_binding is not None
                else {}
            ),
            **(
                {
                    "target_mask_projection_sha256": projection_sha256,
                    "target_mask_projection_semantic_sha256": projection_semantic_sha256,
                }
                if projection_sha256 is not None
                else {}
            ),
            **(
                {"analysis_eligibility": dict(analysis_eligibility)}
                if analysis_eligibility is not None
                else {}
            ),
        },
        sample_ids=identifiers,
        manifest_sha256=manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        representation_id="engineered_target_features",
        input_variant="context_rgb_plus_binary_target_mask",
        encoder_identifier="engineered_target_features_v1",
        encoder_metadata=encoder_metadata,
        encoder_implementation=encoder_implementation,
        weight_identifier=weight_identifier,
        weights_sha256=explicit_unlearned_weights_sha256(weight_identifier),
        preprocessing_identifier=preprocessing_identifier,
        preprocessing=preprocessing,
        cache_recipe=cache_recipe,
        dtype=str(features.values.dtype),
        feature_dimension=len(features.names),
        package_versions={
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "scikit-image": _package_version("scikit-image"),
        },
        matrix_key="values",
        provenance_scope="stage_eligible",
    )
    cache, sidecar, complete = atomic_save_npz_with_sidecar(
        destination,
        arrays={
            "values": features.values,
            "names": names,
            "sample_ids": identifiers,
        },
        metadata=metadata,
    )
    return cache, sidecar, complete


__all__ = [
    "EngineeredFeatureSet",
    "build_engineered_feature_set",
    "engineered_feature_names",
    "extract_engineered_features",
    "save_engineered_feature_cache",
    "select_target_morphometrics",
]
