"""Prospectively frozen PUMA new-data evaluation for the existing AANCA method."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from histo_audit.auditing.scores import fixed_hybrid_score, score_annotations
from histo_audit.auditing.two_queue import (
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
    draw_matched_random_comparator,
)
from histo_audit.config import load_pinned_config
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.evaluation.restoration import (
    macro_f1_from_confusion,
    per_class_recall_from_confusion,
)
from histo_audit.evaluation.review_training import SoftTargetMultinomialLogisticRegression
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.research.frozen_candidate import load_frozen_development_candidate
from histo_audit.statistics.review import average_precision, budget_count

CLASS_ORDER = ("tumor", "lymphocyte", "other")
CLASS_CODES = {name: index for index, name in enumerate(CLASS_ORDER)}
RAW_TO_PRIMARY = {
    "nuclei_tumor": "tumor",
    "nuclei_lymphocyte": "lymphocyte",
    "nuclei_plasma_cell": "lymphocyte",
    "nuclei_stroma": "other",
    "nuclei_endothelium": "other",
    "nuclei_histiocyte": "other",
    "nuclei_melanophage": "other",
    "nuclei_neutrophil": "other",
    "nuclei_apoptosis": "other",
    "nuclei_epithelium": "other",
}
EXPECTED_NATIVE_COUNTS = {
    "nuclei_apoptosis": 1850,
    "nuclei_endothelium": 1701,
    "nuclei_epithelium": 2211,
    "nuclei_histiocyte": 7168,
    "nuclei_lymphocyte": 21643,
    "nuclei_melanophage": 695,
    "nuclei_neutrophil": 366,
    "nuclei_plasma_cell": 520,
    "nuclei_stroma": 3856,
    "nuclei_tumor": 57419,
}


@dataclass(frozen=True, slots=True)
class PUMAManifest:
    """Canonical PUMA nuclei manifest with a frozen case-level partition."""

    frame: pd.DataFrame
    manifest_sha256: str
    source_inventory_sha256: str
    final_case_ids: tuple[str, ...]
    development_case_ids: tuple[str, ...]
    native_class_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PUMAPreparedData:
    """One PUMA partition and label-independent fixed RGB crops."""

    split: str
    manifest: pd.DataFrame
    crops: NDArray[np.uint8]
    exclusions: tuple[dict[str, Any], ...]
    manifest_sha256: str
    source_inventory_sha256: str
    crops_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    records = [
        {str(key): None if bool(pd.isna(value)) else value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    return _semantic_sha256(records)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_puma_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Load and verify both prospectively frozen PUMA authority files."""

    root = Path(repository_root).resolve()
    config_path = root / "configs" / "puma_new_data_confirmation.yaml"
    amendment_path = root / "configs" / "puma_new_data_runtime_amendment.yaml"
    config, config_sha256 = load_pinned_config(
        config_path,
        "434452cf94dce9cf9ce88edd79761686821043c53d64ccb00c02a3a547bdef30",
        role="PUMA prospective configuration",
    )
    amendment, amendment_sha256 = load_pinned_config(
        amendment_path,
        "d93bb2a353e0cdc931edf432fc8b83a82a67857aa90fe95e473ccedd732d0d07",
        role="PUMA runtime amendment",
    )
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("PUMA prospective configuration is malformed")
    if not isinstance(amendment, dict) or amendment.get("schema_version") != 1:
        raise ValueError("PUMA runtime amendment is malformed")
    if (
        config.get("outcomes_opened_at_freeze") is not False
        or config.get("replacement_project_or_v2") is not False
        or amendment.get("candidate_changed") is not False
        or amendment.get("success_gates_changed") is not False
        or amendment.get("metric_outcomes_available_at_freeze") is not False
        or amendment.get("parent_config_sha256") != config_sha256
    ):
        raise RuntimeError("PUMA freeze or runtime-amendment boundary is invalid")
    frozen = load_frozen_development_candidate(root / str(config["candidate"]["record"]))
    if frozen.candidate_sha256 != str(config["candidate"]["sha256"]):
        raise RuntimeError("PUMA config does not point to the frozen AANCA candidate")
    return config, config_sha256, amendment, amendment_sha256


def _case_stratum(case_id: str) -> str:
    if case_id.startswith("training_set_primary_roi_"):
        return "primary"
    if case_id.startswith("training_set_metastatic_roi_"):
        return "metastatic"
    raise ValueError(f"unsupported PUMA case identifier: {case_id!r}")


def frozen_puma_case_split(
    case_ids: Sequence[str], config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply the prospectively frozen, source-stratified case split."""

    unique = tuple(sorted(set(str(value) for value in case_ids)))
    if len(unique) != len(case_ids):
        raise ValueError("PUMA case identifiers must be unique")
    split = config["split"]
    salt = str(split["salt"])
    final_per_stratum = int(split["final_cases_per_stratum"])
    final: list[str] = []
    development: list[str] = []
    for stratum in ("primary", "metastatic"):
        members = [value for value in unique if _case_stratum(value) == stratum]
        if len(members) != 103:
            raise ValueError(f"PUMA {stratum} case count differs from the official 103")
        ordered = sorted(
            members,
            key=lambda value: (
                hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(),
                value,
            ),
        )
        final.extend(ordered[:final_per_stratum])
        development.extend(ordered[final_per_stratum:])
    final_tuple = tuple(sorted(final))
    development_tuple = tuple(sorted(development))
    if set(final_tuple).intersection(development_tuple):
        raise RuntimeError("PUMA frozen case split overlaps")
    if len(final_tuple) != 62 or len(development_tuple) != 144:
        raise RuntimeError("PUMA frozen case split has unexpected group counts")
    return development_tuple, final_tuple


def _points(value: Any) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        output.append((float(value[0]), float(value[1])))
    elif isinstance(value, list):
        for child in value:
            output.extend(_points(child))
    return output


def build_puma_manifest(data_root: str | Path, config: Mapping[str, Any]) -> PUMAManifest:
    """Parse the official PUMA release without deriving any candidate outcome."""

    root = Path(data_root).resolve()
    image_archive = root / str(config["data"]["image_archive"]["name"])
    nuclei_archive = root / str(config["data"]["nuclei_archive"]["name"])
    for path, expected in (
        (image_archive, str(config["data"]["image_archive"]["md5"])),
        (nuclei_archive, str(config["data"]["nuclei_archive"]["md5"])),
    ):
        if not path.is_file() or _md5_file(path) != expected:
            raise RuntimeError(f"official PUMA archive failed its frozen MD5 guard: {path.name}")
    image_root = root / "01_training_dataset_tif_ROIs"
    nuclei_root = root / "01_training_dataset_geojson_nuclei"
    images = {
        path.stem: path
        for path in sorted(image_root.iterdir())
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }
    nuclei = {
        path.stem.removesuffix("_nuclei"): path
        for path in sorted(nuclei_root.glob("*_nuclei.geojson"))
    }
    if set(images) != set(nuclei) or len(images) != 206:
        raise RuntimeError("PUMA extracted image/annotation case pairing is incomplete")
    development_cases, final_cases = frozen_puma_case_split(tuple(images), config)
    final_set = set(final_cases)
    rows: list[dict[str, Any]] = []
    native_counts: Counter[str] = Counter()
    for case_id in sorted(images):
        payload = json.loads(nuclei[case_id].read_text(encoding="utf-8"))
        features = payload.get("features")
        if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
            raise ValueError(f"PUMA GeoJSON is malformed: {nuclei[case_id].name}")
        seen_ids: set[str] = set()
        for position, feature in enumerate(features):
            if not isinstance(feature, Mapping):
                raise ValueError("PUMA GeoJSON feature is not an object")
            annotation_id = str(feature.get("id") or f"row-{position:08d}")
            if annotation_id in seen_ids:
                raise ValueError(f"duplicate PUMA annotation ID in {case_id}: {annotation_id}")
            seen_ids.add(annotation_id)
            properties = feature.get("properties")
            geometry = feature.get("geometry")
            if not isinstance(properties, Mapping) or not isinstance(geometry, Mapping):
                raise ValueError("PUMA GeoJSON feature lacks properties or geometry")
            classification = properties.get("classification")
            if not isinstance(classification, Mapping):
                raise ValueError("PUMA GeoJSON feature lacks classification")
            raw_label = str(classification.get("name"))
            if raw_label not in RAW_TO_PRIMARY:
                raise ValueError(f"unknown PUMA nucleus class: {raw_label!r}")
            coordinates = _points(geometry.get("coordinates"))
            if len(coordinates) < 3:
                raise ValueError(f"PUMA nucleus has invalid polygon geometry: {annotation_id}")
            x_values = np.asarray([point[0] for point in coordinates], dtype=np.float64)
            y_values = np.asarray([point[1] for point in coordinates], dtype=np.float64)
            centre_x = float((x_values.min() + x_values.max()) / 2.0)
            centre_y = float((y_values.min() + y_values.max()) / 2.0)
            primary = RAW_TO_PRIMARY[raw_label]
            native_counts[raw_label] += 1
            rows.append(
                {
                    "sample_id": f"puma-{case_id}-{annotation_id}",
                    "annotation_id": annotation_id,
                    "case_id": case_id,
                    "group_id": case_id,
                    "source_stratum": _case_stratum(case_id),
                    "partition": "final" if case_id in final_set else "development",
                    "image_name": images[case_id].name,
                    "geojson_name": nuclei[case_id].name,
                    "centre_x": centre_x,
                    "centre_y": centre_y,
                    "bbox_width": float(x_values.max() - x_values.min()),
                    "bbox_height": float(y_values.max() - y_values.min()),
                    "native_label": raw_label,
                    "reference_label_name": primary,
                    "reference_label": CLASS_CODES[primary],
                }
            )
    frame = pd.DataFrame(rows).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(frame) != 97429 or dict(sorted(native_counts.items())) != EXPECTED_NATIVE_COUNTS:
        raise RuntimeError("PUMA manifest counts differ from the official release")
    if set(frame["reference_label"].unique()) != set(range(len(CLASS_ORDER))):
        raise RuntimeError("PUMA primary mapping does not contain every frozen class")
    inventory = {
        "image_archive": {
            "name": image_archive.name,
            "bytes": image_archive.stat().st_size,
            "md5": _md5_file(image_archive),
        },
        "nuclei_archive": {
            "name": nuclei_archive.name,
            "bytes": nuclei_archive.stat().st_size,
            "md5": _md5_file(nuclei_archive),
        },
        "images": [(path.name, path.stat().st_size) for path in images.values()],
        "geojson": [(path.name, path.stat().st_size) for path in nuclei.values()],
    }
    return PUMAManifest(
        frame=frame,
        manifest_sha256=_canonical_frame_sha256(frame),
        source_inventory_sha256=_semantic_sha256(inventory),
        final_case_ids=final_cases,
        development_case_ids=development_cases,
        native_class_counts=dict(sorted(native_counts.items())),
    )


def _fixed_crop(
    image: NDArray[np.uint8], *, centre_x: float, centre_y: float, size: int
) -> NDArray[np.uint8]:
    if size <= 0 or size % 2:
        raise ValueError("PUMA crop size must be a positive even integer")
    height, width = image.shape[:2]
    x = round(centre_x)
    y = round(centre_y)
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError("PUMA nucleus centre lies outside its ROI")
    half = size // 2
    x_indices = np.clip(np.arange(x - half, x + half), 0, width - 1)
    y_indices = np.clip(np.arange(y - half, y + half), 0, height - 1)
    crop = image[y_indices[:, None], x_indices[None, :]]
    if crop.shape != (size, size, 3):
        raise RuntimeError("PUMA fixed crop has an unexpected shape")
    return np.asarray(crop, dtype=np.uint8)


def prepare_puma_split(
    data_root: str | Path,
    *,
    split: str,
    crop_size: int,
    config: Mapping[str, Any],
) -> PUMAPreparedData:
    """Build one frozen PUMA split with label-independent crops."""

    if split not in {"development", "final"}:
        raise ValueError("PUMA split must be 'development' or 'final'")
    root = Path(data_root).resolve()
    authority = build_puma_manifest(root, config)
    manifest = (
        authority.frame.loc[authority.frame["partition"] == split].copy().reset_index(drop=True)
    )
    crops = np.empty((len(manifest), crop_size, crop_size, 3), dtype=np.uint8)
    image_root = root / "01_training_dataset_tif_ROIs"
    for image_name, indices in manifest.groupby("image_name", sort=True).groups.items():
        with Image.open(image_root / str(image_name)) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if image.shape != (1024, 1024, 3):
            raise RuntimeError(f"PUMA ROI does not have the official 1024x1024 shape: {image_name}")
        for index in indices:
            row = manifest.iloc[int(index)]
            crops[int(index)] = _fixed_crop(
                image,
                centre_x=float(row["centre_x"]),
                centre_y=float(row["centre_y"]),
                size=crop_size,
            )
    expected_groups = 144 if split == "development" else 62
    if int(manifest["group_id"].nunique()) != expected_groups:
        raise RuntimeError("PUMA prepared split differs from the frozen case count")
    return PUMAPreparedData(
        split=split,
        manifest=manifest,
        crops=crops,
        exclusions=(),
        manifest_sha256=_canonical_frame_sha256(manifest),
        source_inventory_sha256=authority.source_inventory_sha256,
        crops_sha256=_array_sha256(crops),
    )


def extract_puma_embeddings(
    prepared: PUMAPreparedData,
    *,
    cache_path: str | Path,
    scale_px: int,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Extract or verify one frozen ResNet-18 PUMA crop scale."""

    if prepared.crops.shape[1:] != (scale_px, scale_px, 3):
        raise ValueError("PUMA crops differ from the requested embedding scale")
    destination = Path(cache_path).resolve()
    expected_ids = prepared.manifest["sample_id"].astype(str).tolist()
    representation_id = (
        f"puma_{prepared.split}_resnet18_imagenet1k_v1_context_{scale_px}px_frozen_aanca_candidate"
    )
    if destination.is_file():
        cached = load_embedding_cache(destination)
        cached.validate()
        eligibility = cached.metadata.get("analysis_eligibility")
        if (
            cached.sample_ids.tolist() != expected_ids
            or cached.metadata.get("manifest_sha256") != prepared.manifest_sha256
            or cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256
            or cached.metadata.get("representation_id") != representation_id
            or not isinstance(eligibility, Mapping)
            or eligibility.get("source_crops_sha256") != prepared.crops_sha256
        ):
            raise RuntimeError("cached PUMA embeddings fail provenance binding")
        return np.asarray(cached.embeddings, dtype=np.float32)
    result = extract_resnet18_embeddings(
        prepared.crops,
        expected_ids,
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
        cache_path=destination,
        manifest_sha256=prepared.manifest_sha256,
        raw_inventory_sha256=prepared.source_inventory_sha256,
        representation_id=representation_id,
        analysis_eligibility={
            "study_id": "puma_new_data_confirmation_v1",
            "eligible": True,
            "split": prepared.split,
            "sample_count": len(prepared.manifest),
            "source_context_px": scale_px,
            "source_crops_sha256": prepared.crops_sha256,
            "candidate_selection_permitted": False,
        },
    )
    result.validate()
    return np.asarray(result.embeddings, dtype=np.float32)


def validate_puma_embedding_alignment(
    prepared: PUMAPreparedData,
    embeddings_64: NDArray[np.generic],
    embeddings_128: NDArray[np.generic],
) -> None:
    """Prove that the two scales align to one immutable manifest."""

    if (
        embeddings_64.shape != (len(prepared.manifest), 512)
        or embeddings_128.shape != (len(prepared.manifest), 512)
        or not np.isfinite(embeddings_64).all()
        or not np.isfinite(embeddings_128).all()
    ):
        raise RuntimeError("PUMA multiscale embeddings are incomplete or misaligned")


def _one_hot(labels: NDArray[np.int64]) -> NDArray[np.float64]:
    output = np.zeros((len(labels), len(CLASS_ORDER)), dtype=np.float64)
    output[np.arange(len(labels)), labels] = 1.0
    return output


def _fit_predict(
    train_features: NDArray[np.generic],
    labels: NDArray[np.int64],
    test_features: NDArray[np.generic],
    *,
    l2: float,
    class_weight_balanced: bool,
    max_iter: int,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], bool]:
    classes = tuple(range(len(CLASS_ORDER)))
    if sample_weight is None or np.all(sample_weight == 1.0):
        model: Any = MultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=class_weight_balanced,
        ).fit(train_features, labels)
    else:
        model = SoftTargetMultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=class_weight_balanced,
        ).fit_soft_targets(
            train_features,
            _one_hot(labels),
            sample_weight=sample_weight,
        )
    return np.asarray(model.predict_proba(test_features), dtype=np.float64), bool(model.converged_)


def _oof_probabilities(
    features: NDArray[np.generic],
    reference: NDArray[np.int64],
    observed: NDArray[np.int64],
    groups: Sequence[str],
    *,
    folds: int,
    split_seed: int,
    l2: float,
    class_weight_balanced: bool,
    max_iter: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    dict[int, tuple[str, ...]],
    tuple[dict[str, Any], ...],
]:
    classes = tuple(range(len(CLASS_ORDER)))
    plan = make_group_stratified_fold_plan(
        reference,
        groups,
        n_splits=folds,
        class_order=classes,
        seed=split_seed,
    )
    probabilities = np.full((len(observed), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(observed), -1, dtype=np.int64)
    training_groups: dict[int, tuple[str, ...]] = {}
    evidence: list[dict[str, Any]] = []
    for fold in plan.folds:
        model = MultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=class_weight_balanced,
        ).fit(features[fold.train_indices], observed[fold.train_indices])
        probabilities[fold.holdout_indices] = model.predict_proba(features[fold.holdout_indices])
        fold_ids[fold.holdout_indices] = fold.fold_id
        training_groups[fold.fold_id] = fold.training_groups
        evidence.append(
            {
                "fold_id": fold.fold_id,
                "training_groups": list(fold.training_groups),
                "held_out_groups": list(fold.held_out_groups),
                "training_count": len(fold.train_indices),
                "holdout_count": len(fold.holdout_indices),
                "converged": bool(model.converged_),
            }
        )
    if np.any(fold_ids < 0) or not np.isfinite(probabilities).all():
        raise RuntimeError("PUMA group-safe OOF probabilities are incomplete")
    return probabilities, fold_ids, training_groups, tuple(evidence)


def _exact_fold_neighbours(
    features: NDArray[np.generic],
    fold_ids: NDArray[np.int64],
    training_groups_by_fold: Mapping[int, Sequence[str]],
    groups: Sequence[str],
    sample_ids: Sequence[str],
    *,
    k: int,
    device: str,
    query_chunk_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32], dict[str, Any]]:
    """Compute reusable exact fold-safe cosine neighbours at PUMA scale."""

    import torch
    import torch.nn.functional as functional

    matrix = np.asarray(features, dtype=np.float32)
    group_values = np.asarray(tuple(str(value) for value in groups), dtype=np.str_)
    identifiers = np.asarray(tuple(str(value) for value in sample_ids), dtype=np.str_)
    if matrix.ndim != 2 or fold_ids.shape != (len(matrix),):
        raise ValueError("PUMA neighbour inputs do not align")
    chosen_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    neighbour_indices = np.full((len(matrix), k), -1, dtype=np.int64)
    neighbour_distances = np.full((len(matrix), k), np.nan, dtype=np.float32)
    fold_evidence: list[dict[str, Any]] = []
    full_tie_rows = 0
    for fold_id in sorted(int(value) for value in np.unique(fold_ids)):
        allowed = set(str(value) for value in training_groups_by_fold[fold_id])
        reference_indices = np.flatnonzero(np.isin(group_values, tuple(allowed))).astype(np.int64)
        query_indices = np.flatnonzero(fold_ids == fold_id).astype(np.int64)
        if len(reference_indices) <= k or not len(query_indices):
            raise RuntimeError("PUMA neighbour fold lacks sufficient references or queries")
        if set(group_values[reference_indices]).intersection(group_values[query_indices]):
            raise RuntimeError("PUMA neighbour reference contains a held-out query group")
        reference_order = np.argsort(identifiers[reference_indices], kind="stable")
        reference_indices = reference_indices[reference_order]
        reference = matrix[reference_indices]
        query = matrix[query_indices]
        mean = reference.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = reference.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1.0e-12] = 1.0
        reference_tensor = functional.normalize(
            torch.from_numpy((reference - mean) / scale).to(chosen_device), dim=1
        )
        for start in range(0, len(query_indices), query_chunk_size):
            stop = min(start + query_chunk_size, len(query_indices))
            query_tensor = functional.normalize(
                torch.from_numpy((query[start:stop] - mean) / scale).to(chosen_device), dim=1
            )
            similarities = query_tensor @ reference_tensor.T
            values_tensor, positions_tensor = torch.topk(
                similarities,
                k=k + 1,
                dim=1,
                largest=True,
                sorted=False,
            )
            values = values_tensor.detach().cpu().numpy()
            positions = positions_tensor.detach().cpu().numpy()
            similarities_cpu: NDArray[np.float32] | None = None
            for local_row in range(stop - start):
                candidate_positions = positions[local_row]
                candidate_indices = reference_indices[candidate_positions]
                distances = np.clip(1.0 - values[local_row], 0.0, 2.0)
                order = np.lexsort((identifiers[candidate_indices], distances))
                # The backend contract is deterministic float32 cosine search.
                # A boundary tie therefore means equal representable distances,
                # not merely two close values.  A tolerance here turns ordinary
                # dense-neighbour gaps into false ties and forces needless full
                # reference sorts without changing the actual top-k membership.
                boundary_tied = bool(distances[order[k - 1]] == distances[order[k]])
                if boundary_tied:
                    if similarities_cpu is None:
                        similarities_cpu = similarities.detach().cpu().numpy()
                    full_tie_rows += 1
                    full_distances = np.clip(1.0 - similarities_cpu[local_row], 0.0, 2.0)
                    full_order = np.lexsort((identifiers[reference_indices], full_distances))[:k]
                    chosen_indices = reference_indices[full_order]
                    chosen_distances = full_distances[full_order]
                else:
                    chosen = order[:k]
                    chosen_indices = candidate_indices[chosen]
                    chosen_distances = distances[chosen]
                output_row = int(query_indices[start + local_row])
                neighbour_indices[output_row] = chosen_indices
                neighbour_distances[output_row] = chosen_distances.astype(np.float32)
            del similarities, values_tensor, positions_tensor, query_tensor
        fold_evidence.append(
            {
                "fold_id": fold_id,
                "reference_count": len(reference_indices),
                "query_count": len(query_indices),
            }
        )
        del reference_tensor, reference, query
        if chosen_device.startswith("cuda"):
            torch.cuda.empty_cache()
    if np.any(neighbour_indices < 0) or not np.isfinite(neighbour_distances).all():
        raise RuntimeError("PUMA neighbour search did not cover every training nucleus")
    for row, chosen in enumerate(neighbour_indices):
        if str(group_values[row]) in set(str(value) for value in group_values[chosen]):
            raise RuntimeError("PUMA neighbour search leaked a query case")
    evidence = {
        "backend": "exact_fold_safe_torch_float32_cosine",
        "device": chosen_device,
        "k": k,
        "query_chunk_size": query_chunk_size,
        "boundary_tie_full_scan_rows": full_tie_rows,
        "folds": fold_evidence,
        "indices_sha256": _array_sha256(neighbour_indices),
        "distances_sha256": _array_sha256(neighbour_distances),
    }
    return neighbour_indices, neighbour_distances, evidence


def _neighbour_risk(
    observed: NDArray[np.int64],
    neighbour_indices: NDArray[np.int64],
    neighbour_distances: NDArray[np.float32],
) -> NDArray[np.float64]:
    weights = 1.0 / np.maximum(neighbour_distances.astype(np.float64), 1.0e-8)
    support = np.zeros((len(observed), len(CLASS_ORDER)), dtype=np.float64)
    neighbour_labels = observed[neighbour_indices]
    for class_index in range(len(CLASS_ORDER)):
        support[:, class_index] = np.sum(weights * (neighbour_labels == class_index), axis=1)
    support /= support.sum(axis=1, keepdims=True)
    risk = 1.0 - support[np.arange(len(observed)), observed]
    if not np.isfinite(risk).all():
        raise RuntimeError("PUMA neighbour risk contains non-finite values")
    return risk


def _queue_constraints(review_count: int) -> QueueConstraints:
    def cap(fraction: float) -> int:
        return max(1, math.ceil(review_count * fraction))

    return QueueConstraints(
        requested_count=review_count,
        max_per_group=cap(0.10),
        max_per_class=cap(0.50),
        max_per_tissue=cap(0.50),
        max_per_transition=cap(0.30),
        minimum_cosine_distance=0.0,
    )


def _select_queue(
    manifest: pd.DataFrame,
    features: NDArray[np.generic],
    observed: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    risk: NDArray[np.float64],
    *,
    review_budget: float,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    count = budget_count(len(manifest), review_budget)
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    queues = build_two_review_queues(
        risk,
        manifest["group_id"].astype(str).tolist(),
        observed.tolist(),
        manifest["sample_id"].astype(str).tolist(),
        quality_constraints=_queue_constraints(count),
        model_constraints=QueueConstraints(requested_count=count),
        annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
        proposed_labels=proposed.tolist(),
        tissue_types=manifest["source_stratum"].astype(str).tolist(),
        embeddings=features,
    )
    queue = queues.quality_control
    if queue.underfilled or queue.selected_count != count:
        raise RuntimeError("frozen PUMA queue underfilled")
    return np.asarray(queue.selected_indices, dtype=np.int64), {
        "requested_count": count,
        "selected_count": queue.selected_count,
        "underfilled": queue.underfilled,
        "rejection_counts": queue.rejection_counts,
        "group_counts": queue.group_counts,
        "class_counts": queue.class_counts,
        "tissue_counts": queue.tissue_counts,
        "transition_counts": queue.transition_counts,
    }


def _matched_random(
    selected: NDArray[np.int64],
    manifest: pd.DataFrame,
    observed: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    repetitions: int,
    seed_start: int,
    corruption_seed: int,
) -> tuple[NDArray[np.int64], ...]:
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    values = {
        "observed_class": observed.tolist(),
        "organ": manifest["source_stratum"].astype(str).tolist(),
        "proposed_transition": [
            f"{source}->{target}" for source, target in zip(observed, proposed, strict=True)
        ],
    }
    output: list[NDArray[np.int64]] = []
    for repeat in range(repetitions):
        comparator = draw_matched_random_comparator(
            selected,
            np.ones(len(manifest), dtype=bool),
            manifest["sample_id"].astype(str).tolist(),
            values,
            seed=seed_start + corruption_seed % 10000 + repeat,
        )
        if not comparator.available:
            raise RuntimeError(
                "PUMA exact matched-random comparator is unavailable: "
                f"{comparator.unavailable_reason}"
            )
        output.append(np.asarray(comparator.comparator_indices, dtype=np.int64))
    return tuple(output)


def _retrieval_group_counts(
    indices: NDArray[np.int64],
    injected: NDArray[np.bool_],
    groups: NDArray[np.str_],
    unique_groups: tuple[str, ...],
) -> NDArray[np.int64]:
    output = np.zeros((len(unique_groups), 2), dtype=np.int64)
    lookup = {group: position for position, group in enumerate(unique_groups)}
    for index in indices:
        row = lookup[str(groups[int(index)])]
        output[row, 0] += 1
        output[row, 1] += int(injected[int(index)])
    return output


def _interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("PUMA bootstrap produced no finite values")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _precision(counts: NDArray[np.int64]) -> float:
    reviewed = int(counts[..., 0].sum())
    return float(counts[..., 1].sum() / reviewed) if reviewed else float("nan")


def _retrieval_summary(
    candidate_counts: NDArray[np.int64],
    random_counts: NDArray[np.int64],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    candidate_by_seed = np.asarray([_precision(value) for value in candidate_counts])
    random_by_seed = np.asarray(
        [np.mean([_precision(value) for value in seed_values]) for seed_values in random_counts]
    )
    differences_by_seed = candidate_by_seed - random_by_seed
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    group_count = candidate_counts.shape[1]
    for _ in range(iterations):
        sampled = rng.integers(0, group_count, size=group_count)
        values: list[float] = []
        for seed_index in range(candidate_counts.shape[0]):
            candidate_value = _precision(candidate_counts[seed_index, sampled])
            random_values = [
                _precision(random_counts[seed_index, repeat, sampled])
                for repeat in range(random_counts.shape[1])
            ]
            values.append(candidate_value - float(np.mean(random_values)))
        differences.append(float(np.mean(values)))
    return {
        "candidate_precision": float(candidate_by_seed.mean()),
        "mean_matched_random_precision": float(random_by_seed.mean()),
        "candidate_minus_matched_random_precision": float(differences_by_seed.mean()),
        "interval_95": list(_interval(differences)),
        "differences_by_corruption_seed": differences_by_seed.tolist(),
        "reviewed_decisions": int(candidate_counts[..., 0].sum()),
        "injected_changes_found": int(candidate_counts[..., 1].sum()),
        "matched_random_repetitions": random_counts.shape[1],
        "bootstrap_iterations": iterations,
    }


def _group_confusions(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    unique_groups: tuple[str, ...],
) -> NDArray[np.int64]:
    output = np.zeros((len(unique_groups), len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    lookup = {group: position for position, group in enumerate(unique_groups)}
    for truth, prediction, group in zip(
        reference, np.argmax(probabilities, axis=1), groups, strict=True
    ):
        output[lookup[str(group)], int(truth), int(prediction)] += 1
    return output


def _downstream_summary(
    reference: NDArray[np.int64],
    groups: NDArray[np.str_],
    candidate_probabilities: NDArray[np.float64],
    baseline_probabilities: NDArray[np.float64],
    random_probabilities: NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = tuple(sorted(set(str(value) for value in groups)))
    candidate_confusions = np.stack(
        [
            _group_confusions(reference, value, groups, unique_groups)
            for value in candidate_probabilities
        ]
    )
    baseline_confusions = np.stack(
        [
            _group_confusions(reference, value, groups, unique_groups)
            for value in baseline_probabilities
        ]
    )
    random_confusions = np.stack(
        [
            np.stack(
                [
                    _group_confusions(reference, value, groups, unique_groups)
                    for value in seed_values
                ]
            )
            for seed_values in random_probabilities
        ]
    )

    def complete_macro(values: NDArray[np.int64]) -> float:
        return macro_f1_from_confusion(values.sum(axis=0))

    candidate_by_seed = np.asarray([complete_macro(value) for value in candidate_confusions])
    baseline_by_seed = np.asarray([complete_macro(value) for value in baseline_confusions])
    random_by_seed = np.asarray(
        [
            np.mean([complete_macro(value) for value in seed_values])
            for seed_values in random_confusions
        ]
    )
    candidate_minus_baseline = candidate_by_seed - baseline_by_seed
    candidate_minus_random = candidate_by_seed - random_by_seed
    rng = np.random.default_rng(seed)
    baseline_samples: list[float] = []
    random_samples: list[float] = []
    class_samples: list[list[float]] = [[] for _ in CLASS_ORDER]
    for _ in range(iterations):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        seed_baseline: list[float] = []
        seed_random: list[float] = []
        seed_classes: list[list[float]] = [[] for _ in CLASS_ORDER]
        for seed_index in range(candidate_confusions.shape[0]):
            candidate_confusion = candidate_confusions[seed_index, sampled].sum(axis=0)
            baseline_confusion = baseline_confusions[seed_index, sampled].sum(axis=0)
            random_seed_confusions = [
                value[sampled].sum(axis=0) for value in random_confusions[seed_index]
            ]
            seed_baseline.append(
                macro_f1_from_confusion(candidate_confusion)
                - macro_f1_from_confusion(baseline_confusion)
            )
            seed_random.append(
                macro_f1_from_confusion(candidate_confusion)
                - float(
                    np.mean([macro_f1_from_confusion(value) for value in random_seed_confusions])
                )
            )
            candidate_recall = per_class_recall_from_confusion(candidate_confusion)
            baseline_recall = per_class_recall_from_confusion(baseline_confusion)
            for class_index in range(len(CLASS_ORDER)):
                difference = candidate_recall[class_index] - baseline_recall[class_index]
                if np.isfinite(difference):
                    seed_classes[class_index].append(float(difference))
        baseline_samples.append(float(np.mean(seed_baseline)))
        random_samples.append(float(np.mean(seed_random)))
        for class_index, values in enumerate(seed_classes):
            if values:
                class_samples[class_index].append(float(np.mean(values)))
    complete_candidate = candidate_confusions.sum(axis=(0, 1))
    complete_baseline = baseline_confusions.sum(axis=(0, 1))
    recall_point = per_class_recall_from_confusion(
        complete_candidate
    ) - per_class_recall_from_confusion(complete_baseline)
    return {
        "candidate_macro_f1": float(candidate_by_seed.mean()),
        "uncorrected_macro_f1": float(baseline_by_seed.mean()),
        "mean_matched_random_macro_f1": float(random_by_seed.mean()),
        "candidate_minus_uncorrected_macro_f1": float(candidate_minus_baseline.mean()),
        "candidate_minus_uncorrected_interval_95": list(_interval(baseline_samples)),
        "candidate_minus_matched_random_macro_f1": float(candidate_minus_random.mean()),
        "candidate_minus_matched_random_interval_95": list(_interval(random_samples)),
        "candidate_minus_uncorrected_by_corruption_seed": candidate_minus_baseline.tolist(),
        "candidate_minus_matched_random_by_corruption_seed": candidate_minus_random.tolist(),
        "candidate_minus_uncorrected_recall": {
            name: float(recall_point[index]) for index, name in enumerate(CLASS_ORDER)
        },
        "candidate_minus_uncorrected_recall_intervals_95": {
            name: list(_interval(class_samples[index])) for index, name in enumerate(CLASS_ORDER)
        },
        "bootstrap_iterations": iterations,
        "evaluated_case_groups": len(unique_groups),
    }


def run_puma_new_data_confirmation(
    development: PUMAPreparedData,
    final: PUMAPreparedData,
    development_features: NDArray[np.generic],
    final_features: NDArray[np.generic],
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
    *,
    config_sha256: str,
    amendment_sha256: str,
    neighbour_device: str = "auto",
) -> tuple[dict[str, Any], dict[str, NDArray[np.generic]]]:
    """Execute the one-shot frozen PUMA comparison without candidate tuning."""

    if development.split != "development" or final.split != "final":
        raise ValueError("PUMA evaluator received the wrong prepared split roles")
    train_x = np.asarray(development_features, dtype=np.float32)
    test_x = np.asarray(final_features, dtype=np.float32)
    if train_x.shape != (len(development.manifest), 1024) or test_x.shape != (
        len(final.manifest),
        1024,
    ):
        raise ValueError("PUMA evaluator requires the frozen 64+128 embedding view")
    train_groups = development.manifest["group_id"].astype(str).to_numpy(dtype=np.str_)
    test_groups = final.manifest["group_id"].astype(str).to_numpy(dtype=np.str_)
    if set(train_groups).intersection(test_groups):
        raise RuntimeError("PUMA development and final case groups overlap")
    reference_train = development.manifest["reference_label"].to_numpy(dtype=np.int64)
    reference_test = final.manifest["reference_label"].to_numpy(dtype=np.int64)
    runtime = amendment["runtime"]
    candidate = config["candidate"]
    fold_seed = int(runtime["audit_group_fold_seed"])
    fold_plan = make_group_stratified_fold_plan(
        reference_train,
        train_groups.tolist(),
        n_splits=int(config["corruption"]["audit_folds"]),
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=fold_seed,
    )
    shared_fold_ids = np.full(len(reference_train), -1, dtype=np.int64)
    shared_training_groups: dict[int, tuple[str, ...]] = {}
    for fold in fold_plan.folds:
        shared_fold_ids[fold.holdout_indices] = fold.fold_id
        shared_training_groups[fold.fold_id] = fold.training_groups
    neighbour_indices, neighbour_distances, neighbour_evidence = _exact_fold_neighbours(
        train_x,
        shared_fold_ids,
        shared_training_groups,
        train_groups.tolist(),
        development.manifest["sample_id"].astype(str).tolist(),
        k=int(candidate["neighbour_k"]),
        device=neighbour_device,
        query_chunk_size=int(runtime["neighbour_query_chunk_size"]),
    )
    seeds = tuple(int(value) for value in config["corruption"]["seeds"])
    repetitions = int(config["evaluation"]["matched_random_repetitions"])
    unique_train_groups = tuple(sorted(set(str(value) for value in train_groups)))
    candidate_counts = np.zeros((len(seeds), len(unique_train_groups), 2), dtype=np.int64)
    random_counts = np.zeros((len(seeds), repetitions, len(unique_train_groups), 2), dtype=np.int64)
    candidate_probabilities: list[NDArray[np.float64]] = []
    baseline_probabilities: list[NDArray[np.float64]] = []
    random_probabilities: list[NDArray[np.float64]] = []
    observed_by_seed: list[NDArray[np.int64]] = []
    injected_by_seed: list[NDArray[np.bool_]] = []
    risk_by_seed: list[NDArray[np.float64]] = []
    selected_by_seed: list[NDArray[np.int64]] = []
    random_indices_by_seed: list[NDArray[np.int64]] = []
    oof_probabilities_by_seed: list[NDArray[np.float64]] = []
    seed_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for seed_index, corruption_seed in enumerate(seeds):
        corruption = apply_controlled_corruption(
            reference_train,
            sample_ids=development.manifest["sample_id"].astype(str).tolist(),
            group_ids=train_groups.tolist(),
            rate=float(config["corruption"]["fraction"]),
            mechanism="symmetric",
            seed=corruption_seed,
            n_classes=len(CLASS_ORDER),
            generator_representation=None,
            auditor_representation="resnet18_multiscale_64_128",
            upstream_manifest_hash=development.manifest_sha256,
        )
        observed = np.asarray(corruption.observed_labels, dtype=np.int64)
        injected = np.asarray(corruption.is_injected_corruption, dtype=bool)
        probabilities, fold_ids, training_groups, oof_evidence = _oof_probabilities(
            train_x,
            reference_train,
            observed,
            train_groups.tolist(),
            folds=int(config["corruption"]["audit_folds"]),
            split_seed=fold_seed,
            l2=float(candidate["audit_l2"]),
            class_weight_balanced=bool(candidate["audit_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
        )
        if (
            not np.array_equal(fold_ids, shared_fold_ids)
            or training_groups != shared_training_groups
        ):
            raise RuntimeError("PUMA audit folds changed across corruption seeds")
        neighbour_risk = _neighbour_risk(observed, neighbour_indices, neighbour_distances)
        self_confidence = score_annotations(
            observed,
            probabilities,
            method="self_confidence",
            class_order=tuple(range(len(CLASS_ORDER))),
        )
        risk = fixed_hybrid_score(
            {
                "self_confidence": self_confidence,
                "nearest_neighbour_disagreement": neighbour_risk,
            },
            components=("self_confidence", "nearest_neighbour_disagreement"),
            weights=(
                float(candidate["hybrid_self_confidence_weight"]),
                1.0 - float(candidate["hybrid_self_confidence_weight"]),
            ),
        )
        selected, queue_evidence = _select_queue(
            development.manifest,
            train_x,
            observed,
            probabilities,
            risk,
            review_budget=float(candidate["review_budget"]),
        )
        comparators = _matched_random(
            selected,
            development.manifest,
            observed,
            probabilities,
            repetitions=repetitions,
            seed_start=int(runtime["matched_random_seed_start"]),
            corruption_seed=corruption_seed,
        )
        candidate_counts[seed_index] = _retrieval_group_counts(
            selected, injected, train_groups, unique_train_groups
        )
        for repeat, indices in enumerate(comparators):
            random_counts[seed_index, repeat] = _retrieval_group_counts(
                indices, injected, train_groups, unique_train_groups
            )
        baseline_values, baseline_converged = _fit_predict(
            train_x,
            observed,
            test_x,
            l2=float(candidate["downstream_l2"]),
            class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
        )
        candidate_weight = np.ones(len(observed), dtype=np.float64)
        candidate_weight[selected] = 0.0
        candidate_values, candidate_converged = _fit_predict(
            train_x,
            observed,
            test_x,
            l2=float(candidate["downstream_l2"]),
            class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
            max_iter=int(runtime["max_iter"]),
            sample_weight=candidate_weight,
        )
        seed_random_probabilities: list[NDArray[np.float64]] = []
        random_convergence: list[bool] = []
        for indices in comparators:
            weights = np.ones(len(observed), dtype=np.float64)
            weights[indices] = 0.0
            values, converged = _fit_predict(
                train_x,
                observed,
                test_x,
                l2=float(candidate["downstream_l2"]),
                class_weight_balanced=bool(candidate["downstream_class_weight_balanced"]),
                max_iter=int(runtime["max_iter"]),
                sample_weight=weights,
            )
            seed_random_probabilities.append(values)
            random_convergence.append(converged)
        candidate_probabilities.append(candidate_values)
        baseline_probabilities.append(baseline_values)
        random_probabilities.append(np.stack(seed_random_probabilities))
        observed_by_seed.append(observed)
        injected_by_seed.append(injected)
        risk_by_seed.append(np.asarray(risk, dtype=np.float64))
        selected_by_seed.append(selected)
        random_indices_by_seed.append(np.stack(comparators))
        oof_probabilities_by_seed.append(probabilities)
        audit_convergence = [bool(value["converged"]) for value in oof_evidence]
        fit_records.extend(
            [
                {
                    "corruption_seed": corruption_seed,
                    "role": "audit_oof",
                    "fold_id": value["fold_id"],
                    "converged": value["converged"],
                }
                for value in oof_evidence
            ]
        )
        fit_records.append(
            {
                "corruption_seed": corruption_seed,
                "role": "downstream_uncorrected",
                "converged": baseline_converged,
            }
        )
        fit_records.append(
            {
                "corruption_seed": corruption_seed,
                "role": "downstream_aanca_flag_exclude",
                "converged": candidate_converged,
            }
        )
        fit_records.extend(
            {
                "corruption_seed": corruption_seed,
                "role": "downstream_matched_random_flag_exclude",
                "repeat": repeat,
                "converged": converged,
            }
            for repeat, converged in enumerate(random_convergence)
        )
        seed_records.append(
            {
                "corruption_seed": corruption_seed,
                "configuration_hash": corruption.configuration_hash,
                "injected_count": int(injected.sum()),
                "reviewed_count": len(selected),
                "injected_found": int(injected[selected].sum()),
                "precision": float(injected[selected].mean()),
                "average_precision": average_precision(injected, risk),
                "queue": queue_evidence,
                "all_audit_fits_converged": all(audit_convergence),
                "candidate_downstream_converged": candidate_converged,
                "baseline_downstream_converged": baseline_converged,
                "all_random_downstream_fits_converged": all(random_convergence),
            }
        )
    candidate_probability_array = np.stack(candidate_probabilities)
    baseline_probability_array = np.stack(baseline_probabilities)
    random_probability_array = np.stack(random_probabilities)
    retrieval = _retrieval_summary(
        candidate_counts,
        random_counts,
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(runtime["bootstrap_seed"]),
    )
    downstream = _downstream_summary(
        reference_test,
        test_groups,
        candidate_probability_array,
        baseline_probability_array,
        random_probability_array,
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(runtime["bootstrap_seed"]) + 1,
    )
    recall_intervals = downstream["candidate_minus_uncorrected_recall_intervals_95"]
    all_fits_converged = bool(fit_records) and all(
        bool(value["converged"]) for value in fit_records
    )
    success_conditions = {
        "retrieval_precision_lower_bound_gt_matched_random": retrieval["interval_95"][0] > 0.0,
        "downstream_macro_f1_lower_bound_gt_unchanged": downstream[
            "candidate_minus_uncorrected_interval_95"
        ][0]
        > 0.0,
        "downstream_macro_f1_lower_bound_gt_matched_random": downstream[
            "candidate_minus_matched_random_interval_95"
        ][0]
        > 0.0,
        "all_four_seed_directions_positive_against_both_controls": all(
            value > 0.0
            for value in downstream["candidate_minus_uncorrected_by_corruption_seed"]
            + downstream["candidate_minus_matched_random_by_corruption_seed"]
        ),
        "every_primary_class_recall_lower_bound_gte_minus_0_01": all(
            float(recall_intervals[name][0]) >= -0.01 for name in CLASS_ORDER
        ),
        "all_required_models_converged": all_fits_converged,
        "all_hash_group_split_and_final_fold_guards_passed": True,
    }
    all_success = all(success_conditions.values())
    result = {
        "schema_version": 1,
        "study_id": str(config["study_id"]),
        "project": "AANCA",
        "replacement_project_or_v2": False,
        "analysis_disposition": "prospectively_frozen_new_source_controlled_confirmation",
        "config_sha256": config_sha256,
        "runtime_amendment_sha256": amendment_sha256,
        "candidate_sha256": str(candidate["sha256"]),
        "dataset": {
            "name": "PUMA",
            "licence": str(config["data"]["licence"]),
            "development_nuclei": len(development.manifest),
            "final_nuclei": len(final.manifest),
            "development_case_groups": int(development.manifest["group_id"].nunique()),
            "final_case_groups": int(final.manifest["group_id"].nunique()),
            "development_manifest_sha256": development.manifest_sha256,
            "final_manifest_sha256": final.manifest_sha256,
            "source_inventory_sha256": development.source_inventory_sha256,
            "primary_class_order": list(CLASS_ORDER),
            "one_roi_per_case": True,
            "source_annotations_modified": False,
        },
        "candidate": {
            "feature_view": "resnet18_multiscale_64_128",
            "risk_method": "fixed_hybrid",
            "hybrid_self_confidence_weight": float(candidate["hybrid_self_confidence_weight"]),
            "neighbour_k": int(candidate["neighbour_k"]),
            "queue_preset": str(candidate["queue_preset"]),
            "review_budget": float(candidate["review_budget"]),
            "intervention": str(candidate["intervention"]),
            "downstream_l2": float(candidate["downstream_l2"]),
        },
        "neighbour_evidence": neighbour_evidence,
        "corruption_seeds": seed_records,
        "retrieval": retrieval,
        "downstream": downstream,
        "fit_count": len(fit_records),
        "all_models_converged": all_fits_converged,
        "fits": fit_records,
        "success_conditions": success_conditions,
        "all_success_conditions_met": all_success,
        "decision": (
            "controlled-noise transfer supported on frozen PUMA final cases"
            if all_success
            else "frozen PUMA success criteria not met; retain the unchanged-label action"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "natural_error_detection_evaluated": False,
        "pathologist_error_detection_proven": False,
        "clinical_utility_proven": False,
    }
    arrays: dict[str, NDArray[np.generic]] = {
        "development_sample_ids": development.manifest["sample_id"]
        .astype(str)
        .to_numpy(dtype=np.str_),
        "final_sample_ids": final.manifest["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "development_group_ids": train_groups,
        "final_group_ids": test_groups,
        "development_reference_labels": reference_train,
        "final_reference_labels": reference_test,
        "oof_fold_ids": shared_fold_ids,
        "neighbour_indices": neighbour_indices,
        "neighbour_distances": neighbour_distances,
        "observed_labels_by_seed": np.stack(observed_by_seed),
        "is_injected_corruption_by_seed": np.stack(injected_by_seed),
        "risk_by_seed": np.stack(risk_by_seed),
        "selected_indices_by_seed": np.stack(selected_by_seed),
        "matched_random_indices_by_seed": np.stack(random_indices_by_seed),
        "oof_probabilities_by_seed": np.stack(oof_probabilities_by_seed),
        "final_candidate_probabilities_by_seed": candidate_probability_array,
        "final_uncorrected_probabilities_by_seed": baseline_probability_array,
        "final_matched_random_probabilities_by_seed": random_probability_array,
    }
    return result, arrays


def render_puma_report(result: Mapping[str, Any]) -> str:
    """Render the frozen result without enlarging its claim."""

    dataset = result["dataset"]
    retrieval = result["retrieval"]
    downstream = result["downstream"]
    lines = [
        "# Frozen PUMA new-data confirmation",
        "",
        f"**Decision:** {result['decision']}.",
        "",
        "## Design",
        "",
        (
            f"The frozen AANCA candidate was applied without PUMA tuning to "
            f"{dataset['development_case_groups']} development cases and evaluated on "
            f"{dataset['final_case_groups']} untouched final cases. The primary mapping was "
            "the official PUMA tumor / lymphocyte / other benchmark."
        ),
        "",
        "## Controlled corruption retrieval",
        "",
        (
            f"AANCA precision was `{retrieval['candidate_precision']:.6f}` versus "
            f"`{retrieval['mean_matched_random_precision']:.6f}` for exact matched random. "
            f"The difference was `{retrieval['candidate_minus_matched_random_precision']:+.6f}` "
            f"with whole-case 95% interval "
            f"`[{retrieval['interval_95'][0]:+.6f}, {retrieval['interval_95'][1]:+.6f}]`."
        ),
        "",
        "## Downstream final-case result",
        "",
        (
            f"Flag-exclude macro-F1 was `{downstream['candidate_macro_f1']:.6f}`, unchanged "
            f"corrupted-label training was `{downstream['uncorrected_macro_f1']:.6f}`, and mean "
            f"matched-random exclusion was `{downstream['mean_matched_random_macro_f1']:.6f}`."
        ),
        (
            f"AANCA minus unchanged was "
            f"`{downstream['candidate_minus_uncorrected_macro_f1']:+.6f}` with 95% interval "
            f"`[{downstream['candidate_minus_uncorrected_interval_95'][0]:+.6f}, "
            f"{downstream['candidate_minus_uncorrected_interval_95'][1]:+.6f}]`."
        ),
        (
            f"AANCA minus matched random was "
            f"`{downstream['candidate_minus_matched_random_macro_f1']:+.6f}` with 95% interval "
            f"`[{downstream['candidate_minus_matched_random_interval_95'][0]:+.6f}, "
            f"{downstream['candidate_minus_matched_random_interval_95'][1]:+.6f}]`."
        ),
        "",
        "## Frozen gates",
        "",
    ]
    for name, passed in result["success_conditions"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a genuinely new-source controlled annotation-noise result. PUMA does "
                "not release paired natural pre/post review labels, so this experiment does not "
                "show that AANCA detects pathologist errors, biological truth or clinical benefit. "
                "Source annotations were not modified automatically."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CLASS_ORDER",
    "PUMAManifest",
    "PUMAPreparedData",
    "build_puma_manifest",
    "extract_puma_embeddings",
    "frozen_puma_case_split",
    "load_frozen_puma_config",
    "prepare_puma_split",
    "render_puma_report",
    "run_puma_new_data_confirmation",
    "validate_puma_embedding_alignment",
]
