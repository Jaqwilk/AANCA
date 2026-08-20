"""Frozen controlled external benchmark on the official MoNuSAC release."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml
from numpy.typing import NDArray

from histo_audit.auditing.strategies import GroupSafeAuditScoreResult, group_safe_audit_scores
from histo_audit.auditing.two_queue import (
    GROUP_SAFE_OOF_EVIDENCE,
    MatchedRandomComparator,
    QueueConstraints,
    build_two_review_queues,
    draw_matched_random_comparator,
)
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.evaluation.restoration import ClassificationMetrics, classification_metrics
from histo_audit.evaluation.retraining_guard import (
    INDEPENDENT_GROUP_VALIDATION,
    evaluate_multicriteria_retraining_guard,
)
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.statistics.review import average_precision, budget_count, rank_indices
from histo_audit.utils.run_tracking import sha256_file

CLASS_ORDER = ("Epithelial", "Lymphocyte", "Macrophage", "Neutrophil")
CLASS_CODES = {name: index for index, name in enumerate(CLASS_ORDER)}
AMBIGUOUS_LABEL = "Ambiguous"

_ORGAN_PATIENTS = {
    "Lung": (
        "TCGA-55-1594",
        "TCGA-69-7760",
        "TCGA-69-A59K",
        "TCGA-73-4668",
        "TCGA-78-7220",
        "TCGA-86-7713",
        "TCGA-86-8672",
        "TCGA-L4-A4E5",
        "TCGA-MP-A4SY",
        "TCGA-MP-A4T7",
        "TCGA-49-6743",
        "TCGA-50-6591",
        "TCGA-55-7570",
        "TCGA-55-7573",
        "TCGA-73-4662",
        "TCGA-78-7152",
    ),
    "Kidney": (
        "TCGA-5P-A9K0",
        "TCGA-B9-A44B",
        "TCGA-B9-A8YI",
        "TCGA-DW-7841",
        "TCGA-EV-5903",
        "TCGA-F9-A97G",
        "TCGA-G7-A8LD",
        "TCGA-MH-A560",
        "TCGA-P4-AAVK",
        "TCGA-SX-A7SR",
        "TCGA-UZ-A9PO",
        "TCGA-UZ-A9PU",
        "TCGA-2Z-A9JG",
        "TCGA-2Z-A9JN",
        "TCGA-DW-7838",
        "TCGA-DW-7963",
        "TCGA-F9-A8NY",
        "TCGA-IZ-A6M9",
        "TCGA-MH-A55W",
    ),
    "Breast": (
        "TCGA-A2-A0CV",
        "TCGA-A2-A0ES",
        "TCGA-B6-A0WZ",
        "TCGA-BH-A18T",
        "TCGA-D8-A1X5",
        "TCGA-E2-A154",
        "TCGA-E9-A22B",
        "TCGA-E9-A22G",
        "TCGA-EW-A6SD",
        "TCGA-S3-AA11",
        "TCGA-A2-A04X",
        "TCGA-D8-A3Z6",
        "TCGA-E2-A108",
        "TCGA-EW-A6SB",
    ),
    "Prostate": (
        "TCGA-EJ-5495",
        "TCGA-EJ-5505",
        "TCGA-EJ-5517",
        "TCGA-G9-6342",
        "TCGA-G9-6499",
        "TCGA-J4-A67Q",
        "TCGA-J4-A67T",
        "TCGA-KK-A59X",
        "TCGA-KK-A6E0",
        "TCGA-KK-A7AW",
        "TCGA-V1-A8WL",
        "TCGA-V1-A9O9",
        "TCGA-X4-A8KQ",
        "TCGA-YL-A9WY",
        "TCGA-G9-6356",
        "TCGA-G9-6367",
        "TCGA-VP-A87E",
        "TCGA-VP-A87H",
        "TCGA-X4-A8KS",
        "TCGA-YL-A9WL",
    ),
}
PATIENT_TO_ORGAN = {
    patient: organ for organ, patients in _ORGAN_PATIENTS.items() for patient in patients
}


@dataclass(frozen=True, slots=True)
class MoNuSACPreparedData:
    """Canonical manifest and label-independent crops for one official split."""

    split: str
    manifest: pd.DataFrame
    crops: NDArray[np.uint8]
    exclusions: dict[str, int]
    source_inventory: tuple[dict[str, Any], ...]
    source_inventory_sha256: str
    manifest_sha256: str
    crops_sha256: str


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _patient_id(path: Path) -> str:
    value = path.parent.name[:12].upper()
    if len(value) != 12 or not value.startswith("TCGA-"):
        raise ValueError(f"cannot derive MoNuSAC patient ID from {path}")
    return value


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    with tifffile.TiffFile(path) as archive:
        image = np.asarray(archive.series[0].asarray())
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {3, 4} and image.shape[-1] not in {3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError(f"MoNuSAC image is not RGB/RGBA: {path} has shape {image.shape}")
    image = image[..., :3]
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.integer):
            raise ValueError(f"MoNuSAC image dtype is unsupported: {path} has {image.dtype}")
        maximum = int(np.iinfo(image.dtype).max)
        image = np.rint(image.astype(np.float64) * (255.0 / maximum)).astype(np.uint8)
    return np.asarray(image, dtype=np.uint8)


def _fixed_crop(
    image: NDArray[np.uint8], *, centre_x: int, centre_y: int, size: int
) -> NDArray[np.uint8]:
    if size <= 0 or size % 2:
        raise ValueError("MoNuSAC crop size must be a positive even integer")
    if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
        raise ValueError("MoNuSAC crop centre lies outside the image")
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")
    shifted_x = centre_x + half
    shifted_y = centre_y + half
    crop = padded[shifted_y - half : shifted_y + half, shifted_x - half : shifted_x + half]
    if crop.shape != (size, size, 3):
        raise RuntimeError("MoNuSAC fixed crop has an unexpected shape")
    return np.asarray(crop, dtype=np.uint8)


def _annotation_name(annotation: ET.Element) -> str:
    attributes = annotation.findall("./Attributes/Attribute")
    names = [str(item.attrib.get("Name", "")).strip() for item in attributes]
    names = [name for name in names if name]
    if len(names) == 1:
        return names[0]
    fallback = str(annotation.attrib.get("Name", "")).strip()
    if fallback:
        return fallback
    raise ValueError("MoNuSAC XML annotation lacks one class name")


def _xml_regions(xml_path: Path) -> tuple[tuple[str, str, NDArray[np.float64]], ...]:
    root = ET.parse(xml_path).getroot()
    output: list[tuple[str, str, NDArray[np.float64]]] = []
    for annotation in root.findall(".//Annotation"):
        class_name = _annotation_name(annotation)
        annotation_id = str(annotation.attrib.get("Id", ""))
        for region in annotation.findall("./Regions/Region"):
            region_id = str(region.attrib.get("Id", ""))
            vertices = [
                (float(vertex.attrib["X"]), float(vertex.attrib["Y"]))
                for vertex in region.findall("./Vertices/Vertex")
            ]
            if len(vertices) < 3:
                continue
            output.append(
                (class_name, f"{annotation_id}:{region_id}", np.asarray(vertices, dtype=np.float64))
            )
    return tuple(output)


def _source_inventory(root: Path, paths: Sequence[Path]) -> tuple[tuple[dict[str, Any], ...], str]:
    records = tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths), key=lambda value: value.as_posix().casefold())
    )
    payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return records, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_monusac_split(
    split_root: str | Path,
    *,
    split: str,
    crop_size: int = 64,
    excluded_patients: Sequence[str] = (),
) -> MoNuSACPreparedData:
    """Parse official polygons and construct immutable per-nucleus records."""

    root = Path(split_root).resolve()
    if not root.is_dir() or split not in {"train", "test"}:
        raise ValueError("MoNuSAC split root and split name must be valid")
    excluded = {str(value).upper() for value in excluded_patients}
    xml_paths = sorted(root.rglob("*.xml"), key=lambda value: value.as_posix().casefold())
    if not xml_paths:
        raise FileNotFoundError(f"MoNuSAC split contains no XML files: {root}")
    records: list[dict[str, Any]] = []
    crops: list[NDArray[np.uint8]] = []
    source_paths: list[Path] = []
    exclusion_counts: dict[str, int] = {}

    def exclude(reason: str, count: int = 1) -> None:
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + count

    for xml_path in xml_paths:
        patient = _patient_id(xml_path)
        regions = _xml_regions(xml_path)
        if patient in excluded:
            exclude("overlap_patient_excluded_from_development", len(regions))
            continue
        if patient not in PATIENT_TO_ORGAN:
            raise ValueError(f"MoNuSAC patient lacks official organ metadata: {patient}")
        tif_path = xml_path.with_suffix(".tif")
        image_path = tif_path if tif_path.is_file() else xml_path.with_suffix(".svs")
        if not image_path.is_file():
            raise FileNotFoundError(f"MoNuSAC image is absent for {xml_path}")
        image = _read_rgb(image_path)
        source_paths.extend((xml_path, image_path))
        for class_name, region_authority, vertices in regions:
            if class_name == AMBIGUOUS_LABEL:
                exclude("ambiguous_region_excluded")
                continue
            if class_name not in CLASS_CODES:
                raise ValueError(f"unknown MoNuSAC class {class_name!r} in {xml_path}")
            if not np.isfinite(vertices).all():
                exclude("nonfinite_polygon")
                continue
            xmin, ymin = vertices.min(axis=0)
            xmax, ymax = vertices.max(axis=0)
            centre_x = math.floor((float(xmin) + float(xmax)) / 2.0)
            centre_y = math.floor((float(ymin) + float(ymax)) / 2.0)
            if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
                exclude("polygon_centre_outside_image")
                continue
            authority = f"{split}\0{patient}\0{xml_path.stem}\0{region_authority}"
            sample_digest = hashlib.sha256(authority.encode("utf-8")).hexdigest()
            records.append(
                {
                    "sample_id": f"monusac-{split}-{sample_digest[:24]}",
                    "split": split,
                    "patient_id": patient,
                    "group_id": patient,
                    "organ": PATIENT_TO_ORGAN[patient],
                    "image_id": xml_path.stem,
                    "region_authority": region_authority,
                    "reference_label": CLASS_CODES[class_name],
                    "reference_label_name": class_name,
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax),
                    "xml_file": xml_path.relative_to(root).as_posix(),
                    "image_file": image_path.relative_to(root).as_posix(),
                }
            )
            crops.append(_fixed_crop(image, centre_x=centre_x, centre_y=centre_y, size=crop_size))
    if not records:
        raise RuntimeError("MoNuSAC preparation produced no eligible nuclei")
    manifest = pd.DataFrame.from_records(records).sort_values("sample_id", kind="mergesort")
    order = manifest.index.to_numpy(dtype=np.int64)
    manifest = manifest.reset_index(drop=True)
    crop_array = np.stack(crops, axis=0)[order].astype(np.uint8, copy=False)
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("MoNuSAC canonical manifest contains duplicate sample IDs")
    if set(manifest["reference_label"].unique()) != set(range(len(CLASS_ORDER))):
        raise RuntimeError("MoNuSAC eligible split does not contain all four classes")
    if manifest["group_id"].nunique() < 5:
        raise RuntimeError("MoNuSAC split has fewer than five patient groups")
    inventory, inventory_sha = _source_inventory(root, source_paths)
    return MoNuSACPreparedData(
        split=split,
        manifest=manifest,
        crops=crop_array,
        exclusions=dict(sorted(exclusion_counts.items())),
        source_inventory=inventory,
        source_inventory_sha256=inventory_sha,
        manifest_sha256=_canonical_frame_sha256(manifest),
        crops_sha256=_array_sha256(crop_array),
    )


def extract_monusac_embeddings(
    prepared: MoNuSACPreparedData,
    *,
    cache_path: str | Path,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Extract or verify the frozen ResNet-18 context representation."""

    destination = Path(cache_path).resolve()
    expected_ids = prepared.manifest["sample_id"].astype(str).tolist()
    if destination.is_file():
        cached = load_embedding_cache(destination)
        cached.validate()
        if cached.sample_ids.tolist() != expected_ids:
            raise RuntimeError("cached MoNuSAC embeddings differ from canonical sample order")
        if cached.metadata.get("manifest_sha256") != prepared.manifest_sha256:
            raise RuntimeError("cached MoNuSAC embeddings differ from the canonical manifest")
        if cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256:
            raise RuntimeError("cached MoNuSAC embeddings differ from the source inventory")
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
        representation_id=f"monusac_{prepared.split}_resnet18_imagenet1k_v1_context_64px",
        analysis_eligibility={
            "study_id": "monusac_current_aanca_controlled_external_v1",
            "eligible": True,
            "sample_count": len(prepared.manifest),
        },
    )
    result.validate()
    return np.asarray(result.embeddings, dtype=np.float32)


def _oof_probabilities(
    features: NDArray[np.generic],
    labels: NDArray[np.int64],
    groups: Sequence[str],
    *,
    n_splits: int,
    split_seed: int,
    l2: float,
    max_iter: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    dict[int, tuple[str, ...]],
    dict[str, Any],
]:
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=n_splits,
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=split_seed,
    )
    matrix = np.asarray(features, dtype=np.float64)
    probabilities = np.full((len(labels), len(CLASS_ORDER)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    training_groups_by_fold: dict[int, tuple[str, ...]] = {}
    folds: list[dict[str, Any]] = []
    for fold in plan.folds:
        model = MultinomialLogisticRegression(
            class_order=tuple(range(len(CLASS_ORDER))),
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=True,
        ).fit(matrix[fold.train_indices], labels[fold.train_indices])
        probabilities[fold.holdout_indices] = model.predict_proba(matrix[fold.holdout_indices])
        fold_ids[fold.holdout_indices] = fold.fold_id
        training_groups_by_fold[fold.fold_id] = fold.training_groups
        folds.append(
            {
                "fold_id": fold.fold_id,
                "training_groups": list(fold.training_groups),
                "held_out_groups": list(fold.held_out_groups),
                "training_count": len(fold.train_indices),
                "holdout_count": len(fold.holdout_indices),
                "converged": model.converged_,
            }
        )
    if np.any(fold_ids < 0) or not np.isfinite(probabilities).all():
        raise RuntimeError("MoNuSAC group-safe OOF probabilities are incomplete")
    return (
        probabilities,
        fold_ids,
        training_groups_by_fold,
        {
            "splitter": plan.splitter_class_name,
            "fallback_status": plan.splitter_fallback_status,
            "fallback_reason": plan.splitter_fallback_reason,
            "folds": folds,
        },
    )


def _fit_predict(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    test_features: NDArray[np.float64],
    *,
    l2: float,
    max_iter: int,
) -> NDArray[np.float64]:
    model = MultinomialLogisticRegression(
        class_order=tuple(range(len(CLASS_ORDER))),
        l2=l2,
        max_iter=max_iter,
        class_weight_balanced=True,
    ).fit(train_features, train_labels)
    return np.asarray(model.predict_proba(test_features), dtype=np.float64)


def _restored_labels(
    observed: NDArray[np.int64],
    reference: NDArray[np.int64],
    injected: NDArray[np.bool_],
    selected: NDArray[np.int64],
) -> tuple[NDArray[np.int64], int]:
    output = observed.copy()
    restored = selected[injected[selected]]
    output[restored] = reference[restored]
    return output, len(restored)


def _ceil_fraction(count: int, fraction: float) -> int:
    return max(1, math.ceil(count * fraction))


def _queue_constraints(config: Mapping[str, Any], review_count: int) -> QueueConstraints:
    return QueueConstraints(
        requested_count=review_count,
        max_per_group=_ceil_fraction(
            review_count, float(config["max_per_group_fraction_of_review_count"])
        ),
        max_per_class=_ceil_fraction(
            review_count, float(config["max_per_class_fraction_of_review_count"])
        ),
        max_per_tissue=_ceil_fraction(
            review_count, float(config["max_per_tissue_fraction_of_review_count"])
        ),
        max_per_transition=_ceil_fraction(
            review_count, float(config["max_per_transition_fraction_of_review_count"])
        ),
        minimum_cosine_distance=float(config["minimum_cosine_distance"]),
    )


def _strategy_queues(
    manifest: pd.DataFrame,
    features: NDArray[np.float64],
    observed: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    fold_ids: NDArray[np.int64],
    training_groups_by_fold: Mapping[int, Sequence[str]],
    config: Mapping[str, Any],
) -> tuple[
    dict[str, GroupSafeAuditScoreResult],
    dict[str, NDArray[np.int64]],
    dict[str, Any],
]:
    ranking = config["ranking"]
    groups = manifest["group_id"].astype(str).tolist()
    sample_ids = manifest["sample_id"].astype(str).tolist()
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    methods = {
        "self_confidence": "self_confidence",
        "nearest_neighbour_disagreement": "nearest_neighbour_disagreement",
        "fixed_hybrid": "fixed_hybrid",
    }
    scores = {
        name: group_safe_audit_scores(
            features,
            observed,
            probabilities,
            groups,
            fold_ids,
            training_groups_by_fold,
            sample_ids=sample_ids,
            method=method,
            class_order=tuple(range(len(CLASS_ORDER))),
            neighbour_k=int(ranking["neighbour_k"]),
            neighbour_metric=str(ranking["neighbour_metric"]),
            hybrid_weights=tuple(float(value) for value in ranking["hybrid_weights"]),
        )
        for name, method in methods.items()
    }
    review_count = budget_count(len(manifest), float(ranking["review_budget"]))
    global_self = rank_indices(
        scores["self_confidence"].risk_scores,
        tie_break_ids=sample_ids,
    )[:review_count].astype(np.int64)
    constraints = _queue_constraints(config["balanced_quality_queue"], review_count)
    selected = {"self_confidence_global": global_self}
    evidence: dict[str, Any] = {
        "review_count": review_count,
        "constraints": {
            "requested_count": constraints.requested_count,
            "max_per_group": constraints.max_per_group,
            "max_per_class": constraints.max_per_class,
            "max_per_tissue": constraints.max_per_tissue,
            "max_per_transition": constraints.max_per_transition,
            "minimum_cosine_distance": constraints.minimum_cosine_distance,
        },
        "model_improvement_queue_available": False,
        "model_improvement_queue_reason": (
            "no nested cross-fitted measured development intervention utility is available"
        ),
    }
    for score_name, output_name in (
        ("self_confidence", "self_confidence_balanced"),
        ("nearest_neighbour_disagreement", "nearest_neighbour_disagreement_balanced"),
        ("fixed_hybrid", "fixed_hybrid_balanced"),
    ):
        queues = build_two_review_queues(
            scores[score_name].risk_scores,
            groups,
            observed,
            sample_ids,
            quality_constraints=constraints,
            model_constraints=QueueConstraints(requested_count=review_count),
            annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
            proposed_labels=proposed,
            tissue_types=manifest["organ"].astype(str).tolist(),
            embeddings=features,
        )
        selected[output_name] = queues.quality_control.selected_indices
        evidence[output_name] = {
            "selected_count": queues.quality_control.selected_count,
            "underfilled": queues.quality_control.underfilled,
            "rejection_counts": queues.quality_control.rejection_counts,
            "group_counts": queues.quality_control.group_counts,
            "class_counts": queues.quality_control.class_counts,
            "tissue_counts": queues.quality_control.tissue_counts,
            "transition_counts": queues.quality_control.transition_counts,
        }
    return scores, selected, evidence


def _group_confusions(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    unique_groups: tuple[str, ...],
) -> NDArray[np.int64]:
    predictions = np.argmax(probabilities, axis=1)
    output = np.zeros((len(unique_groups), len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    lookup = {group: index for index, group in enumerate(unique_groups)}
    for truth, prediction, group in zip(reference, predictions, groups, strict=True):
        output[lookup[str(group)], int(truth), int(prediction)] += 1
    return output


def _macro_f1(confusion: NDArray[np.integer]) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def _interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("MoNuSAC bootstrap produced no finite values")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _candidate_minus_random_bootstrap(
    reference: NDArray[np.int64],
    candidate_probabilities: NDArray[np.float64],
    random_probabilities: NDArray[np.float64],
    groups: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    group_values = np.asarray(tuple(str(value) for value in groups), dtype=np.str_)
    unique = tuple(sorted(set(str(value) for value in group_values)))
    candidate_confusions = _group_confusions(
        reference, candidate_probabilities, group_values, unique
    )
    random_confusions = np.stack(
        [
            _group_confusions(reference, probabilities, group_values, unique)
            for probabilities in random_probabilities
        ],
        axis=0,
    )
    point_candidate = _macro_f1(candidate_confusions.sum(axis=0))
    point_random = np.asarray(
        [_macro_f1(confusions.sum(axis=0)) for confusions in random_confusions],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        candidate_value = _macro_f1(candidate_confusions[sampled].sum(axis=0))
        random_values = [
            _macro_f1(confusions[sampled].sum(axis=0)) for confusions in random_confusions
        ]
        differences[iteration] = candidate_value - float(np.mean(random_values))
    return {
        "candidate_macro_f1": point_candidate,
        "mean_matched_random_macro_f1": float(point_random.mean()),
        "candidate_minus_mean_matched_random_macro_f1": float(
            point_candidate - point_random.mean()
        ),
        "interval_95": list(_interval(differences)),
        "iterations": iterations,
        "random_repetitions": len(random_probabilities),
    }


def _retrieval_bootstrap(
    injected: NDArray[np.bool_],
    groups: Sequence[str],
    top_indices: NDArray[np.int64],
    random_indices: Sequence[NDArray[np.int64]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    group_values = np.asarray(tuple(str(value) for value in groups), dtype=np.str_)
    unique = tuple(sorted(set(str(value) for value in group_values)))

    def counts(indices: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        selected = np.zeros(len(unique), dtype=np.int64)
        found = np.zeros(len(unique), dtype=np.int64)
        lookup = {group: index for index, group in enumerate(unique)}
        for index in indices:
            column = lookup[str(group_values[int(index)])]
            selected[column] += 1
            found[column] += int(injected[int(index)])
        return selected, found

    top_selected, top_found = counts(top_indices)
    random_counts = [counts(indices) for indices in random_indices]
    top_precision = float(top_found.sum() / top_selected.sum())
    random_precisions = np.asarray(
        [float(found.sum() / selected.sum()) for selected, found in random_counts],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        top_denominator = int(top_selected[sampled].sum())
        if not top_denominator:
            continue
        top_value = float(top_found[sampled].sum() / top_denominator)
        sampled_random: list[float] = []
        for selected, found in random_counts:
            denominator = int(selected[sampled].sum())
            if denominator:
                sampled_random.append(float(found[sampled].sum() / denominator))
        if sampled_random:
            differences.append(top_value - float(np.mean(sampled_random)))
    return {
        "top_found": int(injected[top_indices].sum()),
        "top_reviewed": len(top_indices),
        "top_precision": top_precision,
        "mean_matched_random_found": float(
            np.mean([injected[indices].sum() for indices in random_indices])
        ),
        "mean_matched_random_precision": float(random_precisions.mean()),
        "top_minus_mean_matched_random_precision": float(top_precision - random_precisions.mean()),
        "interval_95": list(_interval(differences)),
        "valid_iterations": len(differences),
        "requested_iterations": iterations,
    }


def _matched_comparators(
    selected: NDArray[np.int64],
    manifest: pd.DataFrame,
    observed: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    config: Mapping[str, Any],
) -> tuple[MatchedRandomComparator, ...]:
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    transitions = [f"{source}->{target}" for source, target in zip(observed, proposed, strict=True)]
    values = {
        "observed_class": observed.tolist(),
        "organ": manifest["organ"].astype(str).tolist(),
        "proposed_transition": transitions,
    }
    match_config = config["matched_random_control"]
    permitted = tuple(sorted(str(value) for value in match_config["fields"]))
    supplied = {name: values[name] for name in permitted}
    output = tuple(
        draw_matched_random_comparator(
            selected,
            np.ones(len(manifest), dtype=bool),
            manifest["sample_id"].astype(str).tolist(),
            supplied,
            seed=int(match_config["seed_start"]) + repeat,
        )
        for repeat in range(int(match_config["repetitions"]))
    )
    unavailable = [item.unavailable_reason for item in output if not item.available]
    if unavailable:
        raise RuntimeError(
            "frozen exact MoNuSAC matched comparator is unavailable: "
            + "; ".join(str(value) for value in unavailable)
        )
    return output


def load_frozen_monusac_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load the prospectively frozen configuration and return its byte identity."""

    path = Path(repository_root).resolve() / "configs/monusac_current_aanca_external.yaml"
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("MoNuSAC frozen configuration is malformed")
    freeze = config.get("freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("MoNuSAC configuration lacks freeze evidence")
    if (
        freeze.get("state") != "frozen_before_metric_execution"
        or freeze.get("outcome_metrics_inspected_before_freeze") is not False
        or freeze.get("parameters_may_change_after_outcome_execution") is not False
    ):
        raise ValueError("MoNuSAC configuration is not prospectively frozen")
    return config, hashlib.sha256(raw).hexdigest()


def run_monusac_controlled_external(
    train: MoNuSACPreparedData,
    test: MoNuSACPreparedData,
    train_embeddings: NDArray[np.generic],
    test_embeddings: NDArray[np.generic],
    config: Mapping[str, Any],
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, NDArray[np.generic]]]:
    """Execute the frozen new-data comparison without tuning from its outcome."""

    train_x = np.asarray(train_embeddings, dtype=np.float64)
    test_x = np.asarray(test_embeddings, dtype=np.float64)
    if train_x.shape != (len(train.manifest), 512) or test_x.shape != (
        len(test.manifest),
        512,
    ):
        raise ValueError("MoNuSAC embeddings do not follow the frozen 512D manifests")
    train_groups = train.manifest["group_id"].astype(str).tolist()
    test_groups = test.manifest["group_id"].astype(str).tolist()
    overlap = sorted(set(train_groups).intersection(test_groups))
    if overlap:
        raise RuntimeError(f"MoNuSAC development/final patient overlap remains: {overlap}")
    reference_train = train.manifest["reference_label"].to_numpy(dtype=np.int64)
    reference_test = test.manifest["reference_label"].to_numpy(dtype=np.int64)
    corruption_config = config["controlled_corruption"]
    corruption = apply_controlled_corruption(
        reference_train,
        sample_ids=train.manifest["sample_id"].astype(str).tolist(),
        group_ids=train_groups,
        rate=float(corruption_config["rate"]),
        mechanism=str(corruption_config["mechanism"]),
        seed=int(corruption_config["seed"]),
        n_classes=len(CLASS_ORDER),
        generator_representation=None,
        auditor_representation="resnet18_imagenet1k_v1_context_rgb",
        upstream_manifest_hash=train.manifest_sha256,
    )
    observed = np.asarray(corruption.observed_labels, dtype=np.int64)
    injected = np.asarray(corruption.is_injected_corruption, dtype=bool)
    model_config = config["model"]
    ranking_config = config["ranking"]
    oof_probabilities, fold_ids, training_groups_by_fold, oof_evidence = _oof_probabilities(
        train_x,
        observed,
        train_groups,
        n_splits=int(ranking_config["folds"]),
        split_seed=int(ranking_config["split_seed"]),
        l2=float(model_config["l2"]),
        max_iter=int(model_config["max_iter"]),
    )
    scores, selections, queue_evidence = _strategy_queues(
        train.manifest,
        train_x,
        observed,
        oof_probabilities,
        fold_ids,
        training_groups_by_fold,
        config,
    )
    primary_name = str(ranking_config["primary_candidate"])
    primary_selected = selections[primary_name]
    comparators = _matched_comparators(
        primary_selected,
        train.manifest,
        observed,
        oof_probabilities,
        config,
    )
    random_indices = tuple(item.comparator_indices for item in comparators)
    ranking_summary: dict[str, Any] = {}
    for name, indices in selections.items():
        score_key = name.replace("_global", "").replace("_balanced", "")
        score = scores[score_key]
        ap = average_precision(injected, score.risk_scores)
        ranking_summary[name] = {
            "strategy": score.as_dict(),
            "average_precision": ap,
            "injected_prevalence": float(injected.mean()),
            "reviewed_count": len(indices),
            "injected_found": int(injected[indices].sum()),
            "precision": float(injected[indices].mean()) if len(indices) else None,
            "underfilled": len(indices) < int(queue_evidence["review_count"]),
        }
    retrieval = _retrieval_bootstrap(
        injected,
        train_groups,
        primary_selected,
        random_indices,
        iterations=int(ranking_config["bootstrap_iterations"]),
        seed=int(ranking_config["bootstrap_seed"]),
    )

    l2 = float(model_config["l2"])
    max_iter = int(model_config["max_iter"])
    label_conditions: dict[str, NDArray[np.int64]] = {"corrupted_uncorrected": observed}
    restored_counts: dict[str, int] = {}
    for name, indices in selections.items():
        restored, count = _restored_labels(observed, reference_train, injected, indices)
        condition = f"{name}_review"
        label_conditions[condition] = restored
        restored_counts[condition] = count
    label_conditions["uncorrupted_reference_ceiling"] = reference_train
    probabilities: dict[str, NDArray[np.float64]] = {
        name: _fit_predict(train_x, labels, test_x, l2=l2, max_iter=max_iter)
        for name, labels in label_conditions.items()
    }
    random_probability_list: list[NDArray[np.float64]] = []
    random_restored_counts: list[int] = []
    for indices in random_indices:
        labels, count = _restored_labels(observed, reference_train, injected, indices)
        random_restored_counts.append(count)
        random_probability_list.append(
            _fit_predict(train_x, labels, test_x, l2=l2, max_iter=max_iter)
        )
    random_probabilities = np.stack(random_probability_list, axis=0)
    metrics: dict[str, ClassificationMetrics] = {
        name: classification_metrics(
            reference_test,
            values,
            class_order=tuple(range(len(CLASS_ORDER))),
        )
        for name, values in probabilities.items()
    }
    baseline = probabilities["corrupted_uncorrected"]
    downstream_config = config["downstream"]
    guards = {
        name: evaluate_multicriteria_retraining_guard(
            reference_test,
            baseline,
            values,
            test_groups,
            class_order=tuple(range(len(CLASS_ORDER))),
            evidence_role=INDEPENDENT_GROUP_VALIDATION,
            important_classes=tuple(range(len(CLASS_ORDER))),
            n_iterations=int(downstream_config["bootstrap_iterations"]),
            seed=int(downstream_config["bootstrap_seed"]),
            minimum_macro_f1_effect=float(downstream_config["minimum_macro_f1_effect"]),
            minimum_per_class_recall_effect=float(
                downstream_config["minimum_per_class_recall_effect"]
            ),
        )
        for name, values in probabilities.items()
        if name not in {"corrupted_uncorrected", "uncorrupted_reference_ceiling"}
    }
    primary_condition = f"{primary_name}_review"
    versus_random = _candidate_minus_random_bootstrap(
        reference_test,
        probabilities[primary_condition],
        random_probabilities,
        test_groups,
        iterations=int(downstream_config["bootstrap_iterations"]),
        seed=int(downstream_config["bootstrap_seed"]) + 1,
    )
    primary_guard = guards[primary_condition]
    success_conditions = {
        "primary_top_k_beats_exact_matched_random_control": retrieval["interval_95"][0] > 0.0,
        "primary_intervention_macro_f1_ci95_lower_gt_corrupted_uncorrected": (
            primary_guard.macro_f1.interval_95 is not None
            and primary_guard.macro_f1.interval_95[0] > 0.0
        ),
        "primary_intervention_macro_f1_ci95_lower_gt_mean_matched_random": (
            versus_random["interval_95"][0] > 0.0
        ),
        "no_important_class_recall_ci95_lower_below_minus_0_01": (
            primary_guard.important_classes_pass
        ),
    }
    all_success = all(success_conditions.values())
    result = {
        "schema_version": 1,
        "study_id": str(config["study_id"]),
        "project": "AANCA",
        "replacement_project_or_v2": False,
        "analysis_disposition": "prospectively_frozen_controlled_external_benchmark",
        "config_sha256": config_sha256,
        "dataset": {
            "name": "MoNuSAC2020",
            "license": str(config["authority"]["license"]),
            "train_eligible_nuclei": len(train.manifest),
            "test_eligible_nuclei": len(test.manifest),
            "train_patient_groups": int(train.manifest["group_id"].nunique()),
            "test_patient_groups": int(test.manifest["group_id"].nunique()),
            "excluded_overlap_patients": list(config["dataset"]["overlap_policy"]["identities"]),
            "train_exclusions": train.exclusions,
            "test_exclusions": test.exclusions,
            "train_manifest_sha256": train.manifest_sha256,
            "test_manifest_sha256": test.manifest_sha256,
            "train_source_inventory_sha256": train.source_inventory_sha256,
            "test_source_inventory_sha256": test.source_inventory_sha256,
            "cross_dataset_pannuke_patient_overlap_excluded": False,
            "cross_dataset_pannuke_patient_overlap_limitation": (
                "PanNuke release metadata does not expose sufficient patient identities to prove "
                "non-overlap"
            ),
        },
        "controlled_corruption": {
            "mechanism": corruption.mechanism,
            "rate": corruption.requested_rate,
            "exact_count": corruption.exact_count,
            "seed": corruption.corruption_seed,
            "configuration_hash": corruption.configuration_hash,
            "source_annotations_modified": False,
        },
        "oof_evidence": oof_evidence,
        "queue_evidence": queue_evidence,
        "ranking": ranking_summary,
        "primary_matched_random_retrieval": retrieval,
        "downstream": {
            "metrics": {name: value.as_dict() for name, value in metrics.items()},
            "restored_injected_counts": restored_counts,
            "matched_random_restored_count_mean": float(np.mean(random_restored_counts)),
            "matched_random_restored_count_range": [
                int(min(random_restored_counts)),
                int(max(random_restored_counts)),
            ],
            "adoption_guards": {name: value.as_dict() for name, value in guards.items()},
            "primary_minus_mean_matched_random": versus_random,
        },
        "success_conditions": success_conditions,
        "all_success_conditions_met": all_success,
        "decision": (
            "controlled external success; result remains final-test evidence and cannot tune a "
            "new candidate"
            if all_success
            else "not supported; retain corrupted-baseline comparison and do not claim a better "
            "real-world model"
        ),
        "unavailable_empirical_components": {
            "multi_reviewer_soft_labels": (
                "MoNuSAC does not provide raw independent reviewer vote distributions"
            ),
            "model_improvement_queue": (
                "no prior measured per-case intervention utility is available"
            ),
            "pathology_encoder_comparison": (
                "no pathology encoder passed the frozen source, licence, weight and smoke gates"
            ),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    arrays: dict[str, NDArray[np.generic]] = {
        "train_sample_ids": train.manifest["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "test_sample_ids": test.manifest["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "train_group_ids": train.manifest["group_id"].astype(str).to_numpy(dtype=np.str_),
        "test_group_ids": test.manifest["group_id"].astype(str).to_numpy(dtype=np.str_),
        "train_reference_labels": reference_train,
        "train_observed_labels": observed,
        "is_injected_corruption": injected,
        "oof_probabilities": oof_probabilities,
        "test_reference_labels": reference_test,
        "matched_random_probabilities": random_probabilities.astype(np.float32),
    }
    arrays.update({f"risk_{name}": score.risk_scores for name, score in scores.items()})
    arrays.update({f"selected_{name}": values for name, values in selections.items()})
    arrays.update({f"test_probabilities_{name}": values for name, values in probabilities.items()})
    return result, arrays


def render_monusac_report(result: Mapping[str, Any]) -> str:
    """Render the frozen outcome without turning controlled changes into natural errors."""

    dataset = result["dataset"]
    corruption = result["controlled_corruption"]
    lines = [
        "# Current AANCA on new MoNuSAC data",
        "",
        f"**Disposition:** {result['analysis_disposition']}  ",
        f"**Decision:** {result['decision']}  ",
        f"**All frozen success conditions met:** {result['all_success_conditions_met']}",
        "",
        "This is a controlled external benchmark on new images, not evidence of natural",
        "pathologist-error detection or clinical utility.",
        "",
        "## Dataset and corruption",
        "",
        f"- Development: {dataset['train_eligible_nuclei']} nuclei in "
        f"{dataset['train_patient_groups']} patient groups.",
        f"- Untouched final test: {dataset['test_eligible_nuclei']} nuclei in "
        f"{dataset['test_patient_groups']} patient groups.",
        f"- Controlled corruption: {corruption['exact_count']} labels "
        f"({corruption['rate']:.1%}), seed `{corruption['seed']}`.",
        "- Two patient identities present in both official archives were excluded from",
        "  development only; the test split remained intact.",
        "",
        "## Ranking at the frozen review budget",
        "",
        "| Queue | AP | Found / reviewed | Precision | Underfilled |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, evidence in result["ranking"].items():
        ap = evidence["average_precision"]
        lines.append(
            f"| {name} | {ap:.6f} | {evidence['injected_found']} / "
            f"{evidence['reviewed_count']} | {evidence['precision']:.6f} | "
            f"{evidence['underfilled']} |"
        )
    retrieval = result["primary_matched_random_retrieval"]
    lines.extend(
        [
            "",
            "Primary top-minus-matched-random precision: "
            f"`{retrieval['top_minus_mean_matched_random_precision']:+.6f}`; 95% "
            f"whole-patient interval `[{retrieval['interval_95'][0]:+.6f}, "
            f"{retrieval['interval_95'][1]:+.6f}]`.",
            "",
            "## Downstream final-test results",
            "",
            "| Condition | Macro F1 | Accuracy | Balanced accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in result["downstream"]["metrics"].items():
        lines.append(
            f"| {name} | {metrics['macro_f1']:.6f} | {metrics['accuracy']:.6f} | "
            f"{metrics['balanced_accuracy']:.6f} |"
        )
    primary_name = "nearest_neighbour_disagreement_balanced_review"
    guard = result["downstream"]["adoption_guards"][primary_name]
    macro = guard["macro_f1"]
    versus_random = result["downstream"]["primary_minus_mean_matched_random"]
    lines.extend(
        [
            "",
            "Primary minus corrupted/no-review macro F1: "
            f"`{macro['candidate_minus_uncorrected_macro_f1']:+.6f}`; 95% interval "
            f"`[{macro['interval_95'][0]:+.6f}, {macro['interval_95'][1]:+.6f}]`.",
            "",
            "Primary minus mean matched-random macro F1: "
            f"`{versus_random['candidate_minus_mean_matched_random_macro_f1']:+.6f}`; "
            f"95% interval `[{versus_random['interval_95'][0]:+.6f}, "
            f"{versus_random['interval_95'][1]:+.6f}]`.",
            "",
            "## Frozen success gates",
            "",
        ]
    )
    for name, passed in result["success_conditions"].items():
        lines.append(f"- `{name}`: **{'passed' if passed else 'failed'}**")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The source labels were not modified. Soft-label and multi-reviewer behaviour",
            "cannot be evaluated because raw independent vote distributions are unavailable.",
            "The per-case model-improvement queue remains unavailable because this release",
            "does not contain prior measured intervention utility. PanNuke patient metadata",
            "is insufficient to rule out every cross-dataset patient overlap.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CLASS_CODES",
    "CLASS_ORDER",
    "MoNuSACPreparedData",
    "extract_monusac_embeddings",
    "load_frozen_monusac_config",
    "prepare_monusac_split",
    "render_monusac_report",
    "run_monusac_controlled_external",
]
