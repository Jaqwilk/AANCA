"""Frozen NuCLS leave-one-pathologist-out disagreement validation.

The input-only preparation and scoring functions in this module cannot see the
pathologist reference table.  Hidden individual votes are attached only after every
OOF probability and AANCA risk score has been computed.  Disagreement is an outcome
for expert-review prioritisation, not proof that any pathologist was wrong.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from PIL import Image
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from histo_audit.auditing.strategies import group_safe_audit_scores
from histo_audit.auditing.two_queue import draw_matched_random_comparator
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.statistics.review import average_precision, rank_indices
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    sha256_file,
)

CLASS_ORDER = ("tumor_any", "nonTIL_stromal", "sTIL")
CLASS_CODES = {name: index for index, name in enumerate(CLASS_ORDER)}
RAW_TO_SUPER = {
    "tumor": "tumor_any",
    "mitotic_figure": "tumor_any",
    "fibroblast": "nonTIL_stromal",
    "vascular_endothelium": "nonTIL_stromal",
    "macrophage": "nonTIL_stromal",
    "lymphocyte": "sTIL",
    "plasma_cell": "sTIL",
}
REFERENCE_STATUSES = (
    "consensus_agree",
    "consensus_disagree",
    "ambiguous",
    "insufficient_reference",
)
INDIVIDUAL_PATHOLOGIST_RE = re.compile(r"^(?:SP|JP)\.\d+$")
BOUNDS_RE = re.compile(
    r"(?P<slide>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{2}-[A-Z0-9]{3})"
    r".*?_left-(?P<left>-?\d+)_top-(?P<top>-?\d+)_bottom-(?P<bottom>-?\d+)"
    r"_right-(?P<right>-?\d+)",
    re.IGNORECASE,
)
ANCHOR_FOV_RE = re.compile(r"^(ANCHFOV-\d+)_", re.IGNORECASE)
EXPECTED_CONFIG_SHA256 = "3e2f3e269385a0225d6c58af263ea1706932fa98dbb19bd7f7599980a4f1d6cd"


@dataclass(frozen=True, slots=True)
class ImageBounds:
    """One label-independent H&E image and its slide-coordinate authority."""

    path: Path
    anchor_fov_id: str
    slide_id: str
    patient_id: str
    left: float
    top: float
    bottom: float
    right: float
    width: int
    height: int
    scale: float


@dataclass(frozen=True, slots=True)
class InputPreparedData:
    """Observed-label-only manifest and aligned multiscale RGB crops."""

    manifest: pd.DataFrame
    crops: Mapping[int, NDArray[np.uint8]]
    exclusions: Mapping[str, int]
    source_inventory: tuple[dict[str, Any], ...]
    source_inventory_sha256: str
    manifest_sha256: str
    crop_sha256: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class ScoredInputData:
    """OOF AANCA evidence created before the hidden reference is opened."""

    scored_manifest: pd.DataFrame
    probabilities: NDArray[np.float64]
    fold_ids: NDArray[np.int64]
    training_groups_by_fold: Mapping[int, tuple[str, ...]]
    fold_evidence: tuple[dict[str, Any], ...]
    risk_metadata: Mapping[str, Any]
    all_models_converged: bool


@dataclass(frozen=True, slots=True)
class ConsensusOutcome:
    """One leave-one-pathologist-out reference decision."""

    status: str
    consensus_label_name: str | None
    consensus_label: int | None
    independent_vote_count: int
    voter_ids: tuple[str, ...]
    vote_labels: tuple[str, ...]
    reason: str


def _normalise_label(value: Any) -> str | None:
    if value is None or bool(pd.isna(value)):
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    lowered = rendered.casefold()
    if lowered.startswith("correction_"):
        lowered = lowered.removeprefix("correction_")
    return lowered


def map_nucls_superclass(value: Any) -> tuple[int | None, str | None]:
    """Map an official raw NuCLS class to the frozen three-superclass scheme."""

    raw = _normalise_label(value)
    if raw is None:
        return None, None
    name = RAW_TO_SUPER.get(raw)
    if name is None:
        return None, None
    return CLASS_CODES[name], name


def _parse_bounds(value: str | Path) -> tuple[str, str, float, float, float, float]:
    match = BOUNDS_RE.search(Path(value).name if isinstance(value, Path) else str(value))
    if match is None:
        raise ValueError(f"NuCLS filename lacks frozen slide/FOV bounds: {value}")
    slide = match.group("slide").upper()
    patient = "-".join(slide.split("-")[:3])
    left, top, bottom, right = (
        float(match.group(name)) for name in ("left", "top", "bottom", "right")
    )
    if right <= left or bottom <= top:
        raise ValueError(f"NuCLS FOV has invalid bounds: {value}")
    return slide, patient, left, top, bottom, right


def _anchor_fov_id(path: Path) -> str:
    match = ANCHOR_FOV_RE.match(path.name)
    if match is None:
        raise ValueError(f"NuCLS RGB filename lacks ANCHFOV identity: {path.name}")
    return match.group(1).upper()


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    records = json.loads(frame.to_json(orient="records", double_precision=15))
    return _semantic_sha256(records)


def _source_inventory(repository_root: Path, paths: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    resolved = sorted({path.resolve() for path in paths}, key=lambda item: item.as_posix())
    records: list[dict[str, Any]] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"NuCLS source input is missing: {path}")
        try:
            relative = path.relative_to(repository_root).as_posix()
        except ValueError:
            relative = path.as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(records)


def load_frozen_independent_pathologist_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load and authenticate the prospectively frozen protocol config."""

    root = Path(repository_root).resolve()
    path = root / "configs" / "nucls_independent_pathologist_validation.yaml"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().casefold()
    if digest != expected or digest != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("independent-pathologist config differs from its frozen authority")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("independent-pathologist config must contain a mapping")
    protocol = root / str(payload.get("protocol", ""))
    candidate = root / str(cast(Mapping[str, Any], payload["candidate"])["record"])
    if not protocol.is_file() or not candidate.is_file():
        raise FileNotFoundError("frozen protocol or selected-candidate record is missing")
    candidate_payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if (
        not isinstance(candidate_payload, dict)
        or candidate_payload.get("candidate_sha256")
        != cast(Mapping[str, Any], payload["candidate"])["sha256"]
    ):
        raise RuntimeError("selected AANCA candidate no longer matches the frozen config")
    if cast(Mapping[str, Any], payload["data"])["input_annotators"] != ["JP.1"]:
        raise RuntimeError("frozen implementation supports only the qualified JP.1 rotation")
    return payload, digest


def _index_images(rgb_directory: Path) -> tuple[ImageBounds, ...]:
    candidates = sorted(
        (
            path
            for path in rgb_directory.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ),
        key=lambda item: item.name.casefold(),
    )
    if not candidates:
        raise FileNotFoundError(f"NuCLS RGB images are missing: {rgb_directory}")
    output: list[ImageBounds] = []
    for path in candidates:
        slide, patient, left, top, bottom, right = _parse_bounds(path)
        with Image.open(path) as opened:
            width, height = opened.size
        scales = np.asarray([width / (right - left), height / (bottom - top)], dtype=float)
        if not np.isfinite(scales).all() or np.any(scales <= 1.0) or np.any(scales >= 2.0):
            raise ValueError(f"NuCLS RGB scale is outside the frozen authority: {path.name}")
        if float(np.ptp(scales)) > 0.05:
            raise ValueError(f"NuCLS RGB axes have inconsistent coordinate scales: {path.name}")
        output.append(
            ImageBounds(
                path=path.resolve(),
                anchor_fov_id=_anchor_fov_id(path),
                slide_id=slide,
                patient_id=patient,
                left=left,
                top=top,
                bottom=bottom,
                right=right,
                width=width,
                height=height,
                scale=float(np.median(scales)),
            )
        )
    if len({item.anchor_fov_id for item in output}) != len(output):
        raise ValueError("NuCLS RGB ANCHFOV identities are not unique")
    return tuple(output)


def _pair_raw_files_to_images(
    raw_files: Sequence[Path], images: Sequence[ImageBounds]
) -> dict[Path, ImageBounds]:
    """Pair label-independent FOV authorities one-to-one without using outcomes."""

    output: dict[Path, ImageBounds] = {}
    slides = sorted({_parse_bounds(path)[0] for path in raw_files})
    for slide in slides:
        slide_raw = sorted(
            (path.resolve() for path in raw_files if _parse_bounds(path)[0] == slide),
            key=lambda item: item.name.casefold(),
        )
        slide_images = sorted(
            (item for item in images if item.slide_id == slide),
            key=lambda item: item.path.name.casefold(),
        )
        if len(slide_raw) != len(slide_images):
            raise RuntimeError(
                f"qualified annotator FOV/image authority is not one-to-one for {slide}: "
                f"{len(slide_raw)} != {len(slide_images)}"
            )
        cost = np.empty((len(slide_raw), len(slide_images)), dtype=np.float64)
        for row, raw_path in enumerate(slide_raw):
            _, _, left, top, bottom, right = _parse_bounds(raw_path)
            raw_bounds = np.asarray([left, top, bottom, right], dtype=np.float64)
            for column, image in enumerate(slide_images):
                image_bounds = np.asarray(
                    [image.left, image.top, image.bottom, image.right], dtype=np.float64
                )
                cost[row, column] = float(np.abs(raw_bounds - image_bounds).sum())
        rows, columns = linear_sum_assignment(cost)
        if len(rows) != len(slide_raw):
            raise RuntimeError("NuCLS FOV/image assignment is incomplete")
        for row, column in zip(rows, columns, strict=True):
            output[slide_raw[int(row)]] = slide_images[int(column)]
    if len(output) != len(raw_files) or len({item.path for item in output.values()}) != len(output):
        raise RuntimeError("NuCLS FOV/image assignment is not bijective")
    return output


def _fixed_crop(
    image: NDArray[np.uint8], *, centre_x: int, centre_y: int, size: int
) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("NuCLS source image must be uint8 RGB")
    if size <= 0 or size % 2:
        raise ValueError("frozen crop size must be a positive even integer")
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")
    shifted_x = centre_x + half
    shifted_y = centre_y + half
    crop = padded[shifted_y - half : shifted_y + half, shifted_x - half : shifted_x + half]
    if crop.shape != (size, size, 3):
        raise RuntimeError("NuCLS crop geometry did not produce the frozen output shape")
    return np.asarray(crop, dtype=np.uint8)


def prepare_pathologist_input(
    repository_root: str | Path,
    *,
    annotator: str,
    config: Mapping[str, Any],
) -> InputPreparedData:
    """Prepare observed input labels and crops without opening the reference table."""

    root = Path(repository_root).resolve()
    data = cast(Mapping[str, Any], config["data"])
    candidate = cast(Mapping[str, Any], config["candidate"])
    if annotator not in tuple(str(value) for value in data["input_annotators"]):
        raise ValueError(f"annotator {annotator!r} is not frozen as an eligible input")
    raw_glob = str(data["raw_annotation_glob"]).format(annotator=annotator)
    raw_files = sorted(root.glob(raw_glob), key=lambda item: item.name.casefold())
    if not raw_files:
        raise FileNotFoundError(f"no raw NuCLS files found for {annotator}")
    rgb_directory = root / str(data["rgb_directory"])
    images = _index_images(rgb_directory)
    paired = _pair_raw_files_to_images(raw_files, images)
    crop_sizes = tuple(int(value) for value in candidate["crop_sizes"])
    if crop_sizes != (64, 128):
        raise RuntimeError("frozen independent-pathologist representation must be 64+128 px")

    exclusions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    crops: dict[int, list[NDArray[np.uint8]]] = {size: [] for size in crop_sizes}
    image_cache: dict[Path, NDArray[np.uint8]] = {}
    used_images: set[Path] = set()
    for raw_path in raw_files:
        image_authority = paired[raw_path.resolve()]
        used_images.add(image_authority.path)
        frame = pd.read_csv(raw_path)
        frame = frame.loc[
            :, [column for column in frame.columns if not column.startswith("Unnamed:")]
        ]
        required = {
            "raw_classification",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"raw NuCLS file {raw_path.name} lacks {sorted(missing)}")
        slide, patient, raw_left, raw_top, _, _ = _parse_bounds(raw_path)
        if slide != image_authority.slide_id or patient != image_authority.patient_id:
            raise RuntimeError("raw FOV/image identity mismatch")
        if image_authority.path not in image_cache:
            with Image.open(image_authority.path) as opened:
                image_cache[image_authority.path] = np.asarray(
                    opened.convert("RGB"), dtype=np.uint8
                )
        image = image_cache[image_authority.path]
        for source_row_index, row in frame.iterrows():
            source_row_number = int(str(source_row_index))
            observed_label, observed_class = map_nucls_superclass(row["raw_classification"])
            if observed_label is None or observed_class is None:
                raw = _normalise_label(row["raw_classification"]) or "missing"
                exclusions[f"input_unmapped:{raw}"] += 1
                continue
            coordinates = np.asarray(
                [row["xmin"], row["ymin"], row["xmax"], row["ymax"]], dtype=np.float64
            )
            if (
                not np.isfinite(coordinates).all()
                or coordinates[2] <= coordinates[0]
                or coordinates[3] <= coordinates[1]
            ):
                exclusions["invalid_raw_bbox"] += 1
                continue
            absolute = np.asarray(
                [
                    raw_left + coordinates[0] / image_authority.scale,
                    raw_top + coordinates[1] / image_authority.scale,
                    raw_left + coordinates[2] / image_authority.scale,
                    raw_top + coordinates[3] / image_authority.scale,
                ],
                dtype=np.float64,
            )
            absolute_centre_x = float((absolute[0] + absolute[2]) / 2.0)
            absolute_centre_y = float((absolute[1] + absolute[3]) / 2.0)
            centre_x = math.floor(
                (absolute_centre_x - image_authority.left) * image_authority.scale
            )
            centre_y = math.floor((absolute_centre_y - image_authority.top) * image_authority.scale)
            if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
                exclusions["raw_bbox_centre_outside_rgb"] += 1
                continue
            relative_raw = raw_path.resolve().relative_to(root).as_posix()
            identity = {
                "study_id": config["study_id"],
                "annotator": annotator,
                "raw_file": relative_raw,
                "source_row_index": source_row_number,
                "raw_bbox": [float(value) for value in coordinates],
            }
            sample_id = (
                f"nucls-{annotator.lower().replace('.', '')}-{_semantic_sha256(identity)[:24]}"
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "input_annotator": annotator,
                    "patient_id": patient,
                    "slide_id": slide,
                    "group_id": patient,
                    "anchor_fov_id": image_authority.anchor_fov_id,
                    "raw_fov_file": relative_raw,
                    "rgb_file": image_authority.path.relative_to(root).as_posix(),
                    "source_row_index": source_row_number,
                    "raw_classification": str(row["raw_classification"]),
                    "observed_label": int(observed_label),
                    "observed_class": observed_class,
                    "raw_xmin": float(coordinates[0]),
                    "raw_ymin": float(coordinates[1]),
                    "raw_xmax": float(coordinates[2]),
                    "raw_ymax": float(coordinates[3]),
                    "absolute_xmin": float(absolute[0]),
                    "absolute_ymin": float(absolute[1]),
                    "absolute_xmax": float(absolute[2]),
                    "absolute_ymax": float(absolute[3]),
                    "rgb_centre_x": int(centre_x),
                    "rgb_centre_y": int(centre_y),
                    "coordinate_scale": float(image_authority.scale),
                }
            )
            for size in crop_sizes:
                crops[size].append(
                    _fixed_crop(image, centre_x=centre_x, centre_y=centre_y, size=size)
                )

    if not records:
        raise RuntimeError("NuCLS observed input preparation produced no scorable rows")
    manifest = pd.DataFrame.from_records(records)
    order = np.argsort(manifest["sample_id"].to_numpy(dtype=np.str_), kind="stable")
    manifest = manifest.iloc[order].reset_index(drop=True)
    crop_arrays = {
        size: np.stack(values, axis=0)[order].astype(np.uint8, copy=False)
        for size, values in crops.items()
    }
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("NuCLS input sample identities are not unique")
    minimum_groups = int(data["minimum_patient_groups"])
    if manifest["group_id"].nunique() < minimum_groups:
        raise RuntimeError("qualified NuCLS input lost required patient-group support")
    if set(manifest["observed_class"].unique()) != set(CLASS_ORDER):
        raise RuntimeError("qualified NuCLS input no longer contains every frozen superclass")
    inventory = _source_inventory(root, (*raw_files, *sorted(used_images)))
    inventory_sha = _semantic_sha256(inventory)
    manifest_sha = _canonical_frame_sha256(manifest)
    crop_sha = {size: _array_sha256(array) for size, array in crop_arrays.items()}
    return InputPreparedData(
        manifest=manifest,
        crops=crop_arrays,
        exclusions=dict(sorted(exclusions.items())),
        source_inventory=inventory,
        source_inventory_sha256=inventory_sha,
        manifest_sha256=manifest_sha,
        crop_sha256=crop_sha,
    )


def extract_multiscale_input_embeddings(
    prepared: InputPreparedData,
    *,
    cache_directory: str | Path,
    study_id: str,
    device: str = "auto",
) -> tuple[NDArray[np.float32], dict[int, dict[str, Any]]]:
    """Extract the frozen 64+128 ResNet-18 view with crop-bound provenance."""

    destination = Path(cache_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
    matrices: list[NDArray[np.float32]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for size in (64, 128):
        cache_path = destination / f"jp1_resnet18_context_{size}.npz"
        representation_id = f"nucls_jp1_resnet18_imagenet1k_v1_context_{size}px"
        eligibility = {
            "study_id": study_id,
            "eligible": True,
            "sample_count": len(sample_ids),
            "source_context_px": size,
            "source_crops_sha256": prepared.crop_sha256[size],
            "hidden_reference_loaded": False,
        }
        if cache_path.is_file():
            cached = load_embedding_cache(cache_path)
            cached.validate()
            recorded = cached.metadata.get("analysis_eligibility")
            if (
                cached.sample_ids.tolist() != sample_ids
                or cached.metadata.get("manifest_sha256") != prepared.manifest_sha256
                or cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256
                or cached.metadata.get("representation_id") != representation_id
                or not isinstance(recorded, Mapping)
                or recorded.get("source_crops_sha256") != prepared.crop_sha256[size]
                or recorded.get("hidden_reference_loaded") is not False
            ):
                raise RuntimeError("cached pathologist-input embeddings fail provenance binding")
            matrix = np.asarray(cached.embeddings, dtype=np.float32)
            embedded_metadata = dict(cached.metadata)
        else:
            result = extract_resnet18_embeddings(
                prepared.crops[size],
                sample_ids,
                config=ResNet18EmbeddingConfig(
                    weight_identifier="IMAGENET1K_V1",
                    input_variant="rgb",
                    device=device,
                    batch_size=64,
                    minimum_batch_size=1,
                    use_amp=True,
                    output_dtype="float32",
                    allow_weight_download=False,
                ),
                cache_path=cache_path,
                manifest_sha256=prepared.manifest_sha256,
                raw_inventory_sha256=prepared.source_inventory_sha256,
                representation_id=representation_id,
                analysis_eligibility=eligibility,
            )
            result.validate()
            matrix = np.asarray(result.embeddings, dtype=np.float32)
            embedded_metadata = dict(result.metadata)
        if matrix.shape != (len(sample_ids), 512) or not np.isfinite(matrix).all():
            raise RuntimeError("frozen ResNet-18 produced invalid pathologist-input embeddings")
        matrices.append(matrix)
        metadata[size] = embedded_metadata
    multiscale = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    return multiscale, metadata


def score_pathologist_input(
    prepared: InputPreparedData,
    embeddings: NDArray[np.generic],
    *,
    config: Mapping[str, Any],
) -> ScoredInputData:
    """Create frozen patient-group-safe AANCA scores without hidden reference data."""

    candidate = cast(Mapping[str, Any], config["candidate"])
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.shape != (len(prepared.manifest), 1024) or not np.isfinite(matrix).all():
        raise ValueError("multiscale embeddings must align with the observed input manifest")
    labels = prepared.manifest["observed_label"].to_numpy(dtype=np.int64)
    groups = prepared.manifest["group_id"].astype(str).tolist()
    sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
    classes = tuple(range(len(CLASS_ORDER)))
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=int(candidate["oof_folds"]),
        class_order=classes,
        seed=int(candidate["split_seed"]),
    )
    probabilities = np.full((len(labels), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    training_groups_by_fold: dict[int, tuple[str, ...]] = {}
    fold_evidence: list[dict[str, Any]] = []
    converged: list[bool] = []
    for fold in plan.folds:
        if set(fold.training_groups).intersection(fold.held_out_groups):
            raise RuntimeError("patient group crossed an AANCA OOF boundary")
        model = MultinomialLogisticRegression(
            class_order=classes,
            l2=float(candidate["audit_l2"]),
            max_iter=int(candidate["audit_max_iter"]),
            class_weight_balanced=bool(candidate["audit_class_weight_balanced"]),
        ).fit(matrix[fold.train_indices], labels[fold.train_indices])
        probabilities[fold.holdout_indices] = model.predict_proba(matrix[fold.holdout_indices])
        fold_ids[fold.holdout_indices] = fold.fold_id
        training_groups_by_fold[fold.fold_id] = fold.training_groups
        converged.append(bool(model.converged_))
        fold_evidence.append(
            {
                "fold_id": fold.fold_id,
                "training_groups": list(fold.training_groups),
                "held_out_groups": list(fold.held_out_groups),
                "held_out_sample_ids": [sample_ids[index] for index in fold.holdout_indices],
                "training_count": len(fold.train_indices),
                "holdout_count": len(fold.holdout_indices),
                "model_converged": bool(model.converged_),
            }
        )
    if np.any(fold_ids < 0) or not np.isfinite(probabilities).all():
        raise RuntimeError("AANCA OOF scoring is incomplete")
    if not all(converged):
        raise RuntimeError("a required AANCA OOF model did not converge")
    scores = group_safe_audit_scores(
        matrix,
        labels,
        probabilities,
        groups,
        fold_ids,
        training_groups_by_fold,
        sample_ids=sample_ids,
        method=str(candidate["risk_method"]),
        class_order=classes,
        neighbour_k=int(candidate["neighbour_k"]),
        neighbour_metric=str(candidate["neighbour_metric"]),
        hybrid_weights=(
            float(candidate["hybrid_self_confidence_weight"]),
            float(candidate["hybrid_neighbour_weight"]),
        ),
    )
    neighbour = scores.neighbour_evidence
    if neighbour is None:
        raise RuntimeError("frozen AANCA hybrid lacks neighbour evidence")
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    scored = prepared.manifest.copy()
    scored["oof_fold_id"] = fold_ids
    for index, class_name in enumerate(CLASS_ORDER):
        scored[f"oof_probability_{class_name}"] = probabilities[:, index]
    scored["proposed_label"] = proposed
    scored["proposed_class"] = [CLASS_ORDER[index] for index in proposed]
    scored["proposed_transition"] = [
        f"{source}->{target}"
        for source, target in zip(scored["observed_class"], scored["proposed_class"], strict=True)
    ]
    scored["risk_self_confidence"] = scores.component_scores["self_confidence"]
    scored["risk_neighbour_disagreement"] = scores.component_scores[
        "nearest_neighbour_disagreement"
    ]
    scored["risk_aanca"] = scores.risk_scores
    scored["neighbour_ids_json"] = [json.dumps(value) for value in neighbour.neighbour_ids]
    scored["neighbour_groups_json"] = [json.dumps(value) for value in neighbour.neighbour_groups]
    scored["neighbour_distances_json"] = [
        json.dumps(value, separators=(",", ":")) for value in neighbour.neighbour_distances
    ]
    return ScoredInputData(
        scored_manifest=scored,
        probabilities=probabilities,
        fold_ids=fold_ids,
        training_groups_by_fold=training_groups_by_fold,
        fold_evidence=tuple(fold_evidence),
        risk_metadata={
            **scores.as_dict(),
            "score_population_count": len(scored),
            "score_population_manifest_sha256": prepared.manifest_sha256,
            "reference_loaded_during_scoring": False,
            "splitter_class_name": plan.splitter_class_name,
            "splitter_fallback_status": plan.splitter_fallback_status,
            "splitter_fallback_reason": plan.splitter_fallback_reason,
        },
        all_models_converged=all(converged),
    )


def construct_leave_one_out_consensus(
    row: Mapping[str, Any],
    *,
    input_annotator: str,
    pathologist_columns: Sequence[str],
    minimum_votes: int,
) -> ConsensusOutcome:
    """Construct a strict majority after removing the input pathologist."""

    if input_annotator not in pathologist_columns:
        raise ValueError("input annotator is absent from the individual pathologist fields")
    if minimum_votes < 2:
        raise ValueError("independent pathologist consensus requires at least two votes")
    voters: list[str] = []
    labels: list[str] = []
    for pathologist in pathologist_columns:
        if pathologist == input_annotator:
            continue
        _, mapped = map_nucls_superclass(row.get(pathologist))
        if mapped is None:
            continue
        voters.append(pathologist)
        labels.append(mapped)
    if input_annotator in voters:
        raise RuntimeError("input annotator leaked into the hidden consensus")
    if len(labels) < minimum_votes:
        return ConsensusOutcome(
            status="insufficient_reference",
            consensus_label_name=None,
            consensus_label=None,
            independent_vote_count=len(labels),
            voter_ids=tuple(voters),
            vote_labels=tuple(labels),
            reason="fewer_than_minimum_independent_mappable_votes",
        )
    counts = Counter(labels)
    consensus_name, count = counts.most_common(1)[0]
    if count <= len(labels) / 2.0:
        return ConsensusOutcome(
            status="ambiguous",
            consensus_label_name=None,
            consensus_label=None,
            independent_vote_count=len(labels),
            voter_ids=tuple(voters),
            vote_labels=tuple(labels),
            reason="no_strict_majority",
        )
    consensus_label = CLASS_CODES[consensus_name]
    observed_label, _ = map_nucls_superclass(row.get(input_annotator))
    if observed_label is None:
        return ConsensusOutcome(
            status="insufficient_reference",
            consensus_label_name=None,
            consensus_label=None,
            independent_vote_count=len(labels),
            voter_ids=tuple(voters),
            vote_labels=tuple(labels),
            reason="master_input_vote_is_unmappable",
        )
    status = "consensus_agree" if observed_label == consensus_label else "consensus_disagree"
    return ConsensusOutcome(
        status=status,
        consensus_label_name=consensus_name,
        consensus_label=consensus_label,
        independent_vote_count=len(labels),
        voter_ids=tuple(voters),
        vote_labels=tuple(labels),
        reason="strict_majority",
    )


def _bbox_iou_matrix(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    intersection_xmin = np.maximum(first[:, None, 0], second[None, :, 0])
    intersection_ymin = np.maximum(first[:, None, 1], second[None, :, 1])
    intersection_xmax = np.minimum(first[:, None, 2], second[None, :, 2])
    intersection_ymax = np.minimum(first[:, None, 3], second[None, :, 3])
    intersection = np.maximum(0.0, intersection_xmax - intersection_xmin) * np.maximum(
        0.0, intersection_ymax - intersection_ymin
    )
    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    return intersection / (first_area[:, None] + second_area[None, :] - intersection + 1e-12)


def attach_hidden_reference(
    scored: ScoredInputData,
    repository_root: str | Path,
    *,
    annotator: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Attach other-pathologist votes after scoring; never recompute AANCA risk."""

    root = Path(repository_root).resolve()
    data = cast(Mapping[str, Any], config["data"])
    labels_config = cast(Mapping[str, Any], config["labels"])
    master_path = (root / str(data["hidden_reference_table"])).resolve()
    master = pd.read_csv(master_path, low_memory=False)
    pathologists = tuple(
        column for column in master.columns if INDIVIDUAL_PATHOLOGIST_RE.fullmatch(column)
    )
    if annotator not in pathologists:
        raise ValueError("input annotator is absent from the hidden-reference table")
    required = {"anchor_id", "xmin", "ymin", "xmax", "ymax", *pathologists}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"NuCLS reference table lacks {sorted(missing)}")
    # Aggregate truth/probability fields are deliberately not copied into this view.
    master = master.loc[:, sorted(required)].copy()
    master["anchor_id"] = master["anchor_id"].astype(str)
    if master["anchor_id"].duplicated().any():
        raise ValueError("NuCLS hidden-reference anchor IDs are not unique")
    master_by_anchor = master.set_index("anchor_id", drop=False)
    contour_root = master_path.parent / "contours"
    output = scored.scored_manifest.copy()
    output["official_anchor_id"] = ""
    output["anchor_iou"] = np.nan
    output["reference_status"] = "insufficient_reference"
    output["reference_reason"] = "no_official_anchor_assignment"
    output["reference_consensus_label"] = pd.Series([pd.NA] * len(output), dtype="Int64")
    output["reference_consensus_class"] = ""
    output["independent_vote_count"] = 0
    output["reference_voter_ids_json"] = "[]"
    output["reference_vote_labels_json"] = "[]"

    iou_threshold = float(data["anchor_iou_threshold"])
    minimum_votes = int(labels_config["minimum_independent_mappable_votes"])
    fov_evidence: list[dict[str, Any]] = []
    used_contours: list[Path] = []
    for anchor_fov_id, local in output.groupby("anchor_fov_id", sort=True):
        contour_files = sorted(contour_root.glob(f"{anchor_fov_id}_*.csv"))
        if len(contour_files) != 1:
            output.loc[local.index, "reference_reason"] = "missing_unique_p_truth_fov_export"
            fov_evidence.append(
                {
                    "anchor_fov_id": anchor_fov_id,
                    "input_count": len(local),
                    "anchor_count": 0,
                    "assigned_count": 0,
                    "above_iou_threshold_count": 0,
                    "identity_match_count": 0,
                    "status": "missing_unique_p_truth_fov_export",
                }
            )
            continue
        contour_path = contour_files[0].resolve()
        used_contours.append(contour_path)
        contours = pd.read_csv(contour_path)
        if "anchor_id" not in contours:
            raise ValueError(f"P-truth contour file lacks anchor_id: {contour_path.name}")
        anchor_ids = contours["anchor_id"].astype(str).tolist()
        available_ids = [
            anchor_id for anchor_id in anchor_ids if anchor_id in master_by_anchor.index
        ]
        anchors = master_by_anchor.loc[available_ids]
        if isinstance(anchors, pd.Series):
            anchors = anchors.to_frame().T
        if anchors.empty:
            output.loc[local.index, "reference_reason"] = "p_truth_fov_has_no_master_anchors"
            continue
        input_boxes = local[
            ["absolute_xmin", "absolute_ymin", "absolute_xmax", "absolute_ymax"]
        ].to_numpy(dtype=np.float64)
        anchor_boxes = anchors[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
        ious = _bbox_iou_matrix(input_boxes, anchor_boxes)
        rows, columns = linear_sum_assignment(-ious)
        above = 0
        identity_matches = 0
        for raw_row, anchor_column in zip(rows, columns, strict=True):
            output_index = int(local.index[int(raw_row)])
            anchor = anchors.iloc[int(anchor_column)]
            iou = float(ious[int(raw_row), int(anchor_column)])
            output.at[output_index, "anchor_iou"] = iou
            if iou < iou_threshold:
                output.at[output_index, "reference_reason"] = "anchor_iou_below_threshold"
                continue
            above += 1
            anchor_id = str(anchor["anchor_id"])
            output.at[output_index, "official_anchor_id"] = anchor_id
            master_input, _ = map_nucls_superclass(anchor[annotator])
            if master_input != int(cast(Any, output.at[output_index, "observed_label"])):
                output.at[output_index, "reference_reason"] = "master_input_identity_conflict"
                continue
            identity_matches += 1
            outcome = construct_leave_one_out_consensus(
                cast(dict[str, Any], anchor.to_dict()),
                input_annotator=annotator,
                pathologist_columns=pathologists,
                minimum_votes=minimum_votes,
            )
            output.at[output_index, "reference_status"] = outcome.status
            output.at[output_index, "reference_reason"] = outcome.reason
            if outcome.consensus_label is not None:
                output.at[output_index, "reference_consensus_label"] = outcome.consensus_label
            output.at[output_index, "reference_consensus_class"] = (
                outcome.consensus_label_name or ""
            )
            output.at[output_index, "independent_vote_count"] = outcome.independent_vote_count
            output.at[output_index, "reference_voter_ids_json"] = json.dumps(outcome.voter_ids)
            output.at[output_index, "reference_vote_labels_json"] = json.dumps(outcome.vote_labels)
        fov_evidence.append(
            {
                "anchor_fov_id": anchor_fov_id,
                "input_count": len(local),
                "anchor_count": len(anchors),
                "assigned_count": len(rows),
                "above_iou_threshold_count": above,
                "identity_match_count": identity_matches,
                "status": "processed",
            }
        )
    output["binary_reference_eligible"] = output["reference_status"].isin(
        ("consensus_agree", "consensus_disagree")
    )
    output["independent_consensus_disagreement"] = (
        output["reference_status"] == "consensus_disagree"
    )
    for value in output["reference_voter_ids_json"]:
        if annotator in tuple(json.loads(str(value))):
            raise RuntimeError("input annotator leaked into a persisted reference vote list")
    if not set(output["reference_status"]).issubset(REFERENCE_STATUSES):
        raise RuntimeError("unknown hidden-reference status was produced")
    inventory = _source_inventory(root, (master_path, *used_contours))
    voter_usage: Counter[str] = Counter()
    for value in output["reference_voter_ids_json"]:
        voter_usage.update(str(item) for item in json.loads(str(value)))
    outcome_counts = {
        key: int((output["reference_status"] == key).sum()) for key in REFERENCE_STATUSES
    }
    strict_consensus_count = (
        outcome_counts["consensus_agree"] + outcome_counts["consensus_disagree"]
    )
    minimum_vote_count = strict_consensus_count + outcome_counts["ambiguous"]
    evidence = {
        "input_annotator": annotator,
        "individual_pathologist_fields": list(pathologists),
        "individual_reference_fields_after_leave_one_out": [
            value for value in pathologists if value != annotator
        ],
        "input_annotator_removed_from_reference": True,
        "aggregate_p_truth_fields_read": False,
        "minimum_independent_mappable_votes": minimum_votes,
        "consensus_rule": "strict_majority",
        "anchor_iou_threshold": iou_threshold,
        "outcome_counts": outcome_counts,
        "reference_reason_counts": output["reference_reason"].value_counts().to_dict(),
        "reference_voter_usage_counts": dict(sorted(voter_usage.items())),
        "independent_vote_count_distribution": {
            str(key): int(value)
            for key, value in sorted(
                output["independent_vote_count"].value_counts().to_dict().items()
            )
        },
        "strict_consensus_rate_when_minimum_votes_met": (
            strict_consensus_count / minimum_vote_count if minimum_vote_count else None
        ),
        "observed_vs_strict_consensus_agreement_rate": (
            outcome_counts["consensus_agree"] / strict_consensus_count
            if strict_consensus_count
            else None
        ),
        "binary_eligible_count": int(output["binary_reference_eligible"].sum()),
        "reference_positive_count": int(output["independent_consensus_disagreement"].sum()),
        "source_inventory_sha256": _semantic_sha256(inventory),
        "fov_alignment": fov_evidence,
    }
    return output, evidence, inventory


def select_exact_comparator_capable_queue(
    frame: pd.DataFrame,
    *,
    budget: float,
    match_fields: Sequence[str],
) -> NDArray[np.int64]:
    """Select highest risks while reserving a disjoint exact matched pool."""

    if not 0.0 < budget <= 1.0 or frame.empty:
        raise ValueError("review budget and binary-reference frame must be non-empty")
    fields = tuple(str(value) for value in match_fields)
    missing = set(fields).union({"risk_aanca", "sample_id"}).difference(frame.columns)
    if missing:
        raise ValueError(f"exact-capacity queue lacks fields: {sorted(missing)}")
    requested = max(1, math.ceil(len(frame) * budget))
    strata = [
        tuple(str(frame.iloc[index][field]) for field in fields) for index in range(len(frame))
    ]
    totals = Counter(strata)
    capacity = {stratum: count // 2 for stratum, count in totals.items()}
    order = np.lexsort(
        (
            frame["sample_id"].astype(str).to_numpy(dtype=np.str_),
            -frame["risk_aanca"].to_numpy(dtype=np.float64),
        )
    )
    selected: list[int] = []
    selected_counts: Counter[tuple[str, ...]] = Counter()
    for value in order:
        index = int(value)
        stratum = strata[index]
        if selected_counts[stratum] >= capacity[stratum]:
            continue
        selected.append(index)
        selected_counts[stratum] += 1
        if len(selected) == requested:
            break
    if len(selected) != requested:
        raise RuntimeError("exact-comparator-capable AANCA queue cannot fill the review budget")
    return np.asarray(selected, dtype=np.int64)


def _precision(events: NDArray[np.bool_], indices: NDArray[np.int64]) -> float:
    if not len(indices):
        raise ValueError("precision requires a non-empty review queue")
    return float(events[indices].mean())


def _bootstrap_primary(
    events: NDArray[np.bool_],
    groups: NDArray[np.str_],
    selected: NDArray[np.int64],
    random_queues: Sequence[NDArray[np.int64]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or iterations <= 0:
        raise ValueError("patient bootstrap requires multiple groups and positive iterations")
    group_to_column = {group: index for index, group in enumerate(unique_groups)}
    selected_columns = np.asarray([group_to_column[group] for group in groups[selected]])
    random_columns = tuple(
        np.asarray([group_to_column[group] for group in groups[indices]])
        for indices in random_queues
    )
    selected_events = events[selected].astype(np.float64)
    random_events = tuple(events[indices].astype(np.float64) for indices in random_queues)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    enrichments: list[float] = []
    undefined_difference = 0
    undefined_enrichment = 0
    for _ in range(iterations):
        draw = rng.integers(0, len(unique_groups), size=len(unique_groups))
        multiplicity = np.bincount(draw, minlength=len(unique_groups)).astype(np.float64)
        selected_weights = multiplicity[selected_columns]
        selected_denominator = float(selected_weights.sum())
        if selected_denominator <= 0.0:
            undefined_difference += 1
            undefined_enrichment += 1
            continue
        selected_precision = (
            float(np.sum(selected_events * selected_weights)) / selected_denominator
        )
        comparator_precisions: list[float] = []
        for values, columns in zip(random_events, random_columns, strict=True):
            weights = multiplicity[columns]
            denominator = float(weights.sum())
            if denominator > 0.0:
                comparator_precisions.append(float(np.sum(values * weights)) / denominator)
        if not comparator_precisions:
            undefined_difference += 1
            undefined_enrichment += 1
            continue
        comparator_precision = float(np.mean(comparator_precisions))
        differences.append(selected_precision - comparator_precision)
        if comparator_precision > 0.0:
            enrichments.append(selected_precision / comparator_precision)
        else:
            undefined_enrichment += 1
    if not differences:
        raise RuntimeError("patient bootstrap produced no finite precision differences")

    def interval(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        return [float(value) for value in np.quantile(values, (0.025, 0.975))]

    return {
        "unit": "patient_id",
        "unique_group_count": len(unique_groups),
        "requested_iterations": iterations,
        "valid_difference_iterations": len(differences),
        "undefined_difference_iterations": undefined_difference,
        "precision_difference_interval_95": interval(differences),
        "valid_enrichment_iterations": len(enrichments),
        "undefined_enrichment_iterations": undefined_enrichment,
        "enrichment_ratio_interval_95": interval(enrichments),
        "seed": seed,
    }


def evaluate_scored_reference(
    evidence: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Evaluate frozen endpoints from already scored, reference-attached evidence."""

    evaluation = cast(Mapping[str, Any], config["evaluation"])
    eligible = evidence.loc[evidence["binary_reference_eligible"]].copy()
    eligible = eligible.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("no binary-reference-eligible nuclei are available")
    events = eligible["independent_consensus_disagreement"].to_numpy(dtype=bool)
    groups = eligible["patient_id"].astype(str).to_numpy(dtype=np.str_)
    sample_ids = eligible["sample_id"].astype(str).tolist()
    fields = tuple(str(value) for value in evaluation["matched_random_fields"])
    match_values = {field: eligible[field].astype(str).tolist() for field in fields}
    repeats = int(evaluation["matched_random_repetitions"])
    seed_start = int(evaluation["matched_random_seed"])
    budgets = tuple(float(value) for value in evaluation["secondary_budgets"])
    budget_results: dict[str, Any] = {}
    queue_payload: dict[str, Any] = {
        "schema_version": 1,
        "match_fields": list(fields),
        "matched_random_repetitions": repeats,
        "budgets": {},
    }
    primary_selected: NDArray[np.int64] | None = None
    primary_random: tuple[NDArray[np.int64], ...] | None = None
    for budget in budgets:
        selected = select_exact_comparator_capable_queue(
            eligible, budget=budget, match_fields=fields
        )
        random_queues: list[NDArray[np.int64]] = []
        comparator_records: list[dict[str, Any]] = []
        budget_seed = seed_start + round(10000 * budget) * 1000
        for repeat in range(repeats):
            comparator = draw_matched_random_comparator(
                selected,
                np.ones(len(eligible), dtype=bool),
                sample_ids,
                match_values,
                seed=budget_seed + repeat,
            )
            if not comparator.available:
                raise RuntimeError(
                    f"exact matched-random comparator unavailable: {comparator.unavailable_reason}"
                )
            indices = np.asarray(comparator.comparator_indices, dtype=np.int64)
            if len(indices) != len(selected) or set(indices).intersection(selected.tolist()):
                raise RuntimeError("matched-random comparator is unequal or overlaps AANCA")
            if Counter(comparator.top_match_strata) != Counter(comparator.comparator_match_strata):
                raise RuntimeError("matched-random comparator does not preserve exact strata")
            random_queues.append(indices)
            comparator_records.append(
                {
                    "repeat": repeat,
                    "seed": comparator.seed,
                    "sample_ids": [sample_ids[index] for index in indices],
                    "match_strata": list(comparator.comparator_match_strata),
                }
            )
        selected_precision = _precision(events, selected)
        random_precisions = [_precision(events, indices) for indices in random_queues]
        random_mean = float(np.mean(random_precisions))
        reviewed_positives = int(events[selected].sum())
        result = {
            "budget_fraction": budget,
            "eligible_count": len(eligible),
            "reviewed_count": len(selected),
            "reference_positive_count": int(events.sum()),
            "aanca_reference_positive_reviewed": reviewed_positives,
            "aanca_precision": selected_precision,
            "aanca_recall": reviewed_positives / int(events.sum()) if events.any() else None,
            "matched_random_mean_precision": random_mean,
            "matched_random_precision_min": float(np.min(random_precisions)),
            "matched_random_precision_max": float(np.max(random_precisions)),
            "matched_random_mean_recall": (
                float(
                    np.mean(
                        [int(events[index].sum()) / int(events.sum()) for index in random_queues]
                    )
                )
                if events.any()
                else None
            ),
            "precision_difference": selected_precision - random_mean,
            "enrichment_ratio": selected_precision / random_mean if random_mean > 0.0 else None,
            "exact_equal_budget": True,
            "exact_strata_preserved": True,
            "aanca_random_disjoint": True,
        }
        key = f"{budget:.2f}"
        budget_results[key] = result
        queue_payload["budgets"][key] = {
            "budget_fraction": budget,
            "aanca_indices": selected.tolist(),
            "aanca_sample_ids": [sample_ids[index] for index in selected],
            "aanca_match_strata": [
                json.dumps(
                    tuple(str(eligible.iloc[index][field]) for field in sorted(fields)),
                    separators=(",", ":"),
                )
                for index in selected
            ],
            "matched_random": comparator_records,
        }
        if math.isclose(budget, float(evaluation["primary_budget"]), abs_tol=1e-12):
            primary_selected = selected
            primary_random = tuple(random_queues)
    if primary_selected is None or primary_random is None:
        raise RuntimeError("primary budget is absent from frozen secondary budgets")
    primary_key = f"{float(evaluation['primary_budget']):.2f}"
    bootstrap = _bootstrap_primary(
        events,
        groups,
        primary_selected,
        primary_random,
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    primary = {**budget_results[primary_key], "bootstrap": bootstrap}
    interval = bootstrap["precision_difference_interval_95"]
    primary_gate_pass = bool(
        interval is not None
        and float(interval[0]) > 0.0
        and primary["exact_equal_budget"]
        and primary["exact_strata_preserved"]
        and primary["aanca_random_disjoint"]
    )

    class_results: dict[str, Any] = {}
    for class_name in CLASS_ORDER:
        mask = eligible["observed_class"].to_numpy(dtype=str) == class_name
        selected_class = primary_selected[mask[primary_selected]]
        random_class = tuple(indices[mask[indices]] for indices in primary_random)
        aanca_precision = _precision(events, selected_class) if len(selected_class) else None
        random_values = [_precision(events, indices) for indices in random_class if len(indices)]
        class_random_mean: float | None = float(np.mean(random_values)) if random_values else None
        difference = (
            aanca_precision - class_random_mean
            if aanca_precision is not None and class_random_mean is not None
            else None
        )
        class_results[class_name] = {
            "eligible_count": int(mask.sum()),
            "reference_positive_count": int(events[mask].sum()),
            "aanca_reviewed_count": len(selected_class),
            "aanca_precision": aanca_precision,
            "matched_random_mean_precision": class_random_mean,
            "precision_difference": difference,
            "enrichment_ratio": (
                aanca_precision / class_random_mean
                if aanca_precision is not None
                and class_random_mean is not None
                and class_random_mean > 0.0
                else None
            ),
            "class_failure_flag": bool(difference is not None and difference < 0.0),
        }

    risk = eligible["risk_aanca"].to_numpy(dtype=np.float64)
    order = rank_indices(risk, tie_break_ids=sample_ids)
    cumulative_positive = np.cumsum(events[order]).astype(np.int64)
    prevalence = float(events.mean())
    ranks = np.arange(1, len(order) + 1, dtype=np.int64)
    curve = pd.DataFrame(
        {
            "rank": ranks,
            "sample_id": eligible.iloc[order]["sample_id"].astype(str).tolist(),
            "risk_aanca": risk[order],
            "cumulative_reference_positive": cumulative_positive,
            "cumulative_precision": cumulative_positive / ranks,
            "cumulative_recall": (
                cumulative_positive / int(events.sum())
                if events.any()
                else np.full(len(order), np.nan)
            ),
            "cumulative_enrichment": (
                (cumulative_positive / ranks) / prevalence
                if prevalence > 0.0
                else np.full(len(order), np.nan)
            ),
        }
    )
    metrics = {
        "eligible_count": len(eligible),
        "reference_positive_count": int(events.sum()),
        "reference_positive_prevalence": prevalence,
        "average_precision": average_precision(events, risk),
        "primary": primary,
        "secondary_budgets": budget_results,
        "by_observed_class": class_results,
        "primary_gate_pass": primary_gate_pass,
        "primary_gate_rule": evaluation["primary_success_rule"],
    }
    return metrics, queue_payload, curve


def _render_report(results: Mapping[str, Any]) -> str:
    metrics = cast(Mapping[str, Any], results["metrics"])
    primary = cast(Mapping[str, Any], metrics["primary"])
    bootstrap = cast(Mapping[str, Any], primary["bootstrap"])
    reference = cast(Mapping[str, Any], results["reference"])
    lines = [
        "# NuCLS independent-pathologist validation results",
        "",
        f"Status: `{results['execution_status']}`",
        "",
        "This frozen evaluation treats disagreements as review outcomes, not as proof that a "
        "pathologist was wrong. Source annotations were not modified.",
        "",
        "## Design",
        "",
        "- Dataset: NuCLS `U-control`.",
        "- Observed input: raw `JP.1` `raw_classification` and individual bbox.",
        "- Hidden reference: strict majority of at least two mappable votes from other "
        "individual pathologists; `JP.1` excluded.",
        "- AANCA: frozen 64+128 px ResNet-18 hybrid candidate, patient-group-safe OOF.",
        "- Comparator: 100 disjoint, exact equal-budget matched-random queues, matched on "
        "patient, observed class, and OOF proposed transition.",
        "",
        "## Primary endpoint (top 5%)",
        "",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Binary-reference-eligible nuclei | {primary['eligible_count']:,} |",
        f"| Reference-positive nuclei | {primary['reference_positive_count']:,} |",
        f"| Reviewed nuclei | {primary['reviewed_count']:,} |",
        f"| AANCA precision | {primary['aanca_precision']:.6f} |",
        f"| Matched-random mean precision | {primary['matched_random_mean_precision']:.6f} |",
        f"| Absolute precision difference | {primary['precision_difference']:+.6f} |",
        f"| Enrichment ratio | {_format_optional(primary['enrichment_ratio'])} |",
        f"| Difference 95% patient-bootstrap CI | {_format_interval(bootstrap['precision_difference_interval_95'])} |",
        f"| Enrichment 95% patient-bootstrap CI | {_format_interval(bootstrap['enrichment_ratio_interval_95'])} |",
        f"| Primary gate | {'PASS' if metrics['primary_gate_pass'] else 'FAIL'} |",
        "",
        "## Per-pathologist result",
        "",
        "Only one input pathologist met the prospectively frozen geometry and patient-group "
        "criteria, so no multi-pathologist pooled estimate is presented.",
        "",
        "| Input pathologist | Status | Eligible | Reviewed | AANCA precision | Random precision | Difference | Enrichment | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        f"| `JP.1` | reported | {primary['eligible_count']:,} | {primary['reviewed_count']:,} | "
        f"{primary['aanca_precision']:.6f} | {primary['matched_random_mean_precision']:.6f} | "
        f"{primary['precision_difference']:+.6f} | {_format_optional(primary['enrichment_ratio'])} | "
        f"{'PASS' if metrics['primary_gate_pass'] else 'FAIL'} |",
        "",
        "## Outcome accounting",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ]
    for status in REFERENCE_STATUSES:
        lines.append(f"| `{status}` | {reference['outcome_counts'][status]:,} |")
    lines.extend(
        [
            "",
            f"Strict consensus among nuclei with the minimum vote count was available for "
            f"`{100 * reference['strict_consensus_rate_when_minimum_votes_met']:.2f}%`; "
            f"the observed `JP.1` label agreed with that strict consensus in "
            f"`{100 * reference['observed_vs_strict_consensus_agreement_rate']:.2f}%` of "
            "binary-reference-eligible nuclei.",
            "",
            "## Secondary budgets",
            "",
            "| Budget | Reviewed | AANCA precision | Random precision | Difference | Enrichment | AANCA recall |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cast(Mapping[str, Mapping[str, Any]], metrics["secondary_budgets"]).values():
        lines.append(
            f"| {100 * item['budget_fraction']:.0f}% | {item['reviewed_count']:,} | "
            f"{item['aanca_precision']:.6f} | {item['matched_random_mean_precision']:.6f} | "
            f"{item['precision_difference']:+.6f} | {_format_optional(item['enrichment_ratio'])} | "
            f"{_format_optional(item['aanca_recall'])} |"
        )
    lines.extend(
        [
            "",
            f"AUPRC on the binary-reference-eligible cohort: `{_format_optional(metrics['average_precision'])}`.",
            "",
            "## Observed-class breakdown at 5%",
            "",
            "| Observed class | Eligible | Positives | AANCA reviewed | AANCA precision | Random precision | Difference | Failure flag |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for class_name, item in cast(
        Mapping[str, Mapping[str, Any]], metrics["by_observed_class"]
    ).items():
        lines.append(
            f"| `{class_name}` | {item['eligible_count']:,} | {item['reference_positive_count']:,} | "
            f"{item['aanca_reviewed_count']:,} | {_format_optional(item['aanca_precision'])} | "
            f"{_format_optional(item['matched_random_mean_precision'])} | "
            f"{_format_signed_optional(item['precision_difference'])} | "
            f"{'YES' if item['class_failure_flag'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Validity and limitations",
            "",
            f"- The input annotator was removed from reference: `{reference['input_annotator_removed_from_reference']}`.",
            f"- Aggregate P-truth fields were read: `{reference['aggregate_p_truth_fields_read']}`.",
            f"- All OOF models converged: `{results['scoring']['all_models_converged']}`.",
            "- Only `JP.1` qualified. `JP.2` had only two patient groups; other pathologists "
            "lacked public individual raw geometry. The per-pathologist and aggregate result "
            "are therefore the same single rotation.",
            "- The feasibility audit saw reference prevalence/category counts before freeze, "
            "but no AANCA score--reference association. Protocol and execution share one "
            "repository change, without an independent timestamp.",
            "- There are only five patient clusters, limiting bootstrap resolution and "
            "generalisation.",
            "- This validates the global frozen risk ranking, not the five-patient-infeasible "
            "`balanced_relaxed` deployment queue.",
            "",
            "## Claim boundary",
            "",
            "A positive result supports enrichment of independent-pathologist disagreement "
            "for this eligible NuCLS `JP.1` cohort relative to the frozen matched-random "
            "control. It does not establish pathologist error, clinical error, automatic "
            "correction safety, senior-pathologist generalisation, multi-site generalisation, "
            "or downstream/clinical utility. Every flagged nucleus remains recommended for "
            "expert review only.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _format_signed_optional(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def _format_interval(value: Any) -> str:
    if value is None:
        return "NA"
    lower, upper = value
    return f"[{float(lower):+.6f}, {float(upper):+.6f}]"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    records = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
        "artifact_root_sha256": _semantic_sha256(records),
    }


def run_independent_pathologist_validation(
    repository_root: str | Path,
    *,
    output_directory: str | Path,
    report_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Execute the frozen two-phase JP.1 validation and persist portable evidence."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_independent_pathologist_config(root)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.iterdir() if path.name != "embeddings"]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing validation artifacts: {output}")
    annotator = str(
        cast(Sequence[Any], cast(Mapping[str, Any], config["data"])["input_annotators"])[0]
    )

    # Phase 1: observed labels, pixels, embeddings, and scores only.
    prepared = prepare_pathologist_input(root, annotator=annotator, config=config)
    embeddings, embedding_metadata = extract_multiscale_input_embeddings(
        prepared,
        cache_directory=output / "embeddings",
        study_id=str(config["study_id"]),
        device=device,
    )
    scored = score_pathologist_input(prepared, embeddings, config=config)

    # Phase 2: the hidden reference is opened only after risk is complete.
    attached, reference_evidence, reference_inventory = attach_hidden_reference(
        scored,
        root,
        annotator=annotator,
        config=config,
    )
    metrics, queues, curve = evaluate_scored_reference(attached, config=config)
    primary_budget_key = (
        f"{float(cast(Mapping[str, Any], config['evaluation'])['primary_budget']):.2f}"
    )
    eligible = attached.loc[attached["binary_reference_eligible"]].sort_values(
        "sample_id", kind="stable"
    )
    primary_ids = set(
        cast(Mapping[str, Any], queues["budgets"])[primary_budget_key]["aanca_sample_ids"]
    )
    attached["selected_primary_aanca"] = attached["sample_id"].isin(primary_ids)
    random_ids = {
        sample_id
        for record in cast(Mapping[str, Any], queues["budgets"])[primary_budget_key][
            "matched_random"
        ]
        for sample_id in record["sample_ids"]
    }
    attached["selected_in_any_primary_random_repeat"] = attached["sample_id"].isin(random_ids)

    paths = cast(Mapping[str, Any], config["outputs"])
    scored_path = output / str(paths["scored_samples_csv"])
    queues_path = output / str(paths["matched_queues_json"])
    curve_path = output / str(paths["enrichment_curve_csv"])
    source_path = output / str(paths["source_inventory_json"])
    numeric_path = output / str(paths["numeric_evidence_npz"])
    results_path = output / str(paths["results_json"])
    _write_csv(scored_path, attached)
    atomic_write_json(queues_path, queues)
    _write_csv(curve_path, curve)
    source_payload = {
        "schema_version": 1,
        "input_only_inventory": list(prepared.source_inventory),
        "input_only_inventory_sha256": prepared.source_inventory_sha256,
        "hidden_reference_inventory": list(reference_inventory),
        "hidden_reference_inventory_sha256": reference_evidence["source_inventory_sha256"],
        "hidden_reference_loaded_after_scoring": True,
    }
    atomic_write_json(source_path, source_payload)
    probabilities = scored.probabilities
    atomic_write_npz(
        numeric_path,
        {
            "sample_ids": attached["sample_id"].astype(str).to_numpy(dtype=np.str_),
            "patient_ids": attached["patient_id"].astype(str).to_numpy(dtype=np.str_),
            "observed_labels": attached["observed_label"].to_numpy(dtype=np.int64),
            "oof_probabilities": probabilities,
            "oof_fold_ids": scored.fold_ids,
            "risk_self_confidence": attached["risk_self_confidence"].to_numpy(dtype=np.float64),
            "risk_neighbour_disagreement": attached["risk_neighbour_disagreement"].to_numpy(
                dtype=np.float64
            ),
            "risk_aanca": attached["risk_aanca"].to_numpy(dtype=np.float64),
            "reference_status": attached["reference_status"].astype(str).to_numpy(dtype=np.str_),
            "binary_reference_eligible": attached["binary_reference_eligible"].to_numpy(dtype=bool),
            "independent_consensus_disagreement": attached[
                "independent_consensus_disagreement"
            ].to_numpy(dtype=bool),
            "selected_primary_aanca": attached["selected_primary_aanca"].to_numpy(dtype=bool),
            "eligible_sample_ids": eligible["sample_id"].astype(str).to_numpy(dtype=np.str_),
        },
    )
    protocol_path = root / str(config["protocol"])
    results: dict[str, Any] = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "execution_status": "EXTERNAL_VALIDATION_COMPLETE",
        "config_sha256": config_sha,
        "protocol_sha256": sha256_file(protocol_path),
        "candidate_sha256": cast(Mapping[str, Any], config["candidate"])["sha256"],
        "input_annotator": annotator,
        "qualified_input_annotators": [annotator],
        "excluded_annotators": cast(Mapping[str, Any], config["data"])["excluded_annotators"],
        "prepared_input": {
            "scorable_count": len(prepared.manifest),
            "patient_groups": sorted(prepared.manifest["group_id"].astype(str).unique()),
            "patient_group_count": prepared.manifest["group_id"].nunique(),
            "fov_count": prepared.manifest["anchor_fov_id"].nunique(),
            "observed_class_counts": prepared.manifest["observed_class"].value_counts().to_dict(),
            "exclusions": dict(prepared.exclusions),
            "manifest_sha256": prepared.manifest_sha256,
            "crop_sha256": {str(key): value for key, value in prepared.crop_sha256.items()},
            "input_source_inventory_sha256": prepared.source_inventory_sha256,
        },
        "scoring": {
            "risk": dict(scored.risk_metadata),
            "folds": list(scored.fold_evidence),
            "all_models_converged": scored.all_models_converged,
            "hidden_reference_loaded_during_scoring": False,
            "embedding_metadata": {str(key): value for key, value in embedding_metadata.items()},
            "embedding_multiscale_sha256": _array_sha256(embeddings),
        },
        "reference": reference_evidence,
        "metrics": metrics,
        "pathologist_results": {
            annotator: {
                "status": "reported",
                "role": "only_qualifying_rotation_no_multi_pathologist_pooling",
                "eligible_count": metrics["eligible_count"],
                "reference_positive_count": metrics["reference_positive_count"],
                "primary": metrics["primary"],
                "by_observed_class": metrics["by_observed_class"],
                "primary_gate_pass": metrics["primary_gate_pass"],
            }
        },
        "validity": {
            "source_annotations_modified": False,
            "input_annotator_absent_from_reference_votes": True,
            "aggregate_p_truth_fields_used": False,
            "patient_group_safe_oof": True,
            "fold_safe_neighbours": True,
            "exact_equal_budget_comparators": True,
            "protocol_and_execution_same_change": True,
            "reference_feasibility_counts_inspected_at_freeze": True,
            "aanca_reference_association_inspected_at_freeze": False,
        },
        "claim_boundary": config["claim_boundary"],
        "portable_evidence": {
            "scored_samples_csv": scored_path.name,
            "matched_queues_json": queues_path.name,
            "enrichment_curve_csv": curve_path.name,
            "source_inventory_json": source_path.name,
            "numeric_evidence_npz": numeric_path.name,
            "metrics_recomputable_without_retraining": True,
        },
    }
    atomic_write_json(results_path, results)
    report = _render_report(results)
    internal_report = output / "report.md"
    atomic_write_text(internal_report, report)
    external_report = Path(report_path).resolve()
    atomic_write_text(external_report, report)
    manifest = _artifact_manifest(output)
    atomic_write_json(output / str(paths["artifact_manifest_json"]), manifest)
    return results


def _assert_close(actual: Any, expected: Any, field: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f"verification differs for {field}: {actual} != {expected}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"verification differs for {field}: {actual} != {expected}")


def verify_independent_pathologist_artifacts(
    repository_root: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Read back artifacts and independently recalculate all endpoint metrics."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_independent_pathologist_config(root)
    output = Path(output_directory).resolve()
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = cast(Sequence[Mapping[str, Any]], manifest["artifacts"])
    for record in records:
        path = output / str(record["path"])
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != str(
            record["sha256"]
        ):
            raise RuntimeError(f"artifact integrity failed: {record['path']}")
    if _semantic_sha256(records) != manifest["artifact_root_sha256"]:
        raise RuntimeError("artifact root hash differs")
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if results["config_sha256"] != config_sha:
        raise RuntimeError("results do not match the frozen config")
    frame = pd.read_csv(output / "scored_samples.csv", float_precision="round_trip")
    bool_columns = (
        "binary_reference_eligible",
        "independent_consensus_disagreement",
        "selected_primary_aanca",
        "selected_in_any_primary_random_repeat",
    )
    for column in bool_columns:
        if frame[column].dtype != bool:
            frame[column] = (
                frame[column].astype(str).str.casefold().map({"true": True, "false": False})
            )
            if frame[column].isna().any():
                raise RuntimeError(f"portable evidence has invalid boolean field {column}")
    recalculated, queues, curve = evaluate_scored_reference(frame, config=config)
    saved_metrics = cast(Mapping[str, Any], results["metrics"])
    for budget, actual in cast(
        Mapping[str, Mapping[str, Any]], recalculated["secondary_budgets"]
    ).items():
        expected = cast(Mapping[str, Any], saved_metrics["secondary_budgets"])[budget]
        for field in (
            "aanca_precision",
            "matched_random_mean_precision",
            "precision_difference",
            "enrichment_ratio",
            "aanca_recall",
            "matched_random_mean_recall",
        ):
            _assert_close(actual[field], expected[field], f"secondary.{budget}.{field}")
        if actual["reviewed_count"] != expected["reviewed_count"]:
            raise RuntimeError(f"review budget differs at {budget}")
    actual_primary = cast(Mapping[str, Any], recalculated["primary"])
    expected_primary = cast(Mapping[str, Any], saved_metrics["primary"])
    for field in (
        "aanca_precision",
        "matched_random_mean_precision",
        "precision_difference",
        "enrichment_ratio",
    ):
        _assert_close(actual_primary[field], expected_primary[field], f"primary.{field}")
    for field in ("precision_difference_interval_95", "enrichment_ratio_interval_95"):
        actual_interval = actual_primary["bootstrap"][field]
        expected_interval = expected_primary["bootstrap"][field]
        if actual_interval is None or expected_interval is None:
            if actual_interval is not expected_interval:
                raise RuntimeError(f"bootstrap interval differs: {field}")
        else:
            for index, value in enumerate(actual_interval):
                _assert_close(value, expected_interval[index], f"bootstrap.{field}[{index}]")
    saved_queues = json.loads((output / "matched_random_queues.json").read_text(encoding="utf-8"))
    if queues != saved_queues:
        raise RuntimeError("matched-random queue readback differs")
    saved_curve = pd.read_csv(output / "enrichment_curve.csv")
    if saved_curve["sample_id"].astype(str).tolist() != curve["sample_id"].astype(str).tolist():
        raise RuntimeError("enrichment-curve ranking differs")
    if not np.allclose(
        saved_curve["cumulative_precision"].to_numpy(dtype=float),
        curve["cumulative_precision"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    ):
        raise RuntimeError("enrichment-curve metrics differ")
    return {
        "verified": True,
        "study_id": results["study_id"],
        "primary_gate_pass": recalculated["primary_gate_pass"],
        "artifact_count": manifest["artifact_count"],
        "metrics_recomputed_without_retraining": True,
        "source_annotations_modified": False,
    }


__all__ = [
    "CLASS_ORDER",
    "REFERENCE_STATUSES",
    "ConsensusOutcome",
    "InputPreparedData",
    "ScoredInputData",
    "attach_hidden_reference",
    "construct_leave_one_out_consensus",
    "evaluate_scored_reference",
    "extract_multiscale_input_embeddings",
    "load_frozen_independent_pathologist_config",
    "map_nucls_superclass",
    "prepare_pathologist_input",
    "run_independent_pathologist_validation",
    "score_pathologist_input",
    "select_exact_comparator_capable_queue",
    "verify_independent_pathologist_artifacts",
]
