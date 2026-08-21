"""Frozen NuCLS multi-rater validation for natural annotation disagreement.

The observed label is the official inferred non-pathologist label and the hidden
reference is the official inferred pathologist consensus.  Neither is biological
truth.  This module never changes downloaded source files and never describes a
disagreement as a pathologist error.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from PIL import Image

from histo_audit.auditing.strategies import (
    GroupSafeAuditScoreResult,
    group_safe_audit_scores,
)
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.evaluation.restoration import classification_metrics
from histo_audit.evaluation.retraining_guard import evaluate_retraining_guard
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.statistics.review import (
    average_precision,
    binary_auroc,
    budget_count,
    draw_group_bootstrap_indices,
    rank_indices,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_npz, sha256_file

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
EXCLUDED_LABELS = {
    "undetected",
    "ambiguous",
    "other_nucleus",
    "apoptotic_body",
    "unlabeled",
    "neutrophil",
    "eosinophil",
    "myoepithelium",
    "ductal_epithelium",
    "non-existent",
}
TCGA_PATIENT_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
TCGA_SLIDE_RE = re.compile(
    r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{2}-[A-Z0-9]{3})",
    re.IGNORECASE,
)
FOV_TAIL_RE = re.compile(r"(TCGA-.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NuCLSPreparedData:
    """Canonical paired natural-label dataset and fixed RGB crops."""

    manifest: pd.DataFrame
    crops: NDArray[np.uint8]
    exclusions: dict[str, int]
    source_inventory: tuple[dict[str, Any], ...]
    source_inventory_sha256: str
    manifest_sha256: str
    crops_sha256: str


@dataclass(frozen=True, slots=True)
class IntervalSummary:
    """Finite bootstrap distribution summary."""

    estimate: float
    mean: float
    interval_95: tuple[float, float]
    valid_iterations: int
    requested_iterations: int


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_records_sha256(frame: pd.DataFrame) -> str:
    records = frame.to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_inventory(
    roles: Mapping[str, tuple[Path, Sequence[Path]]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    records: list[dict[str, Any]] = []
    for role, (root, paths) in sorted(roles.items()):
        for path in sorted(set(paths), key=lambda item: item.as_posix().casefold()):
            if not path.is_file() or not path.is_relative_to(root):
                raise FileNotFoundError(
                    f"NuCLS authority file is missing or outside its root: {path}"
                )
            records.append(
                {
                    "role": role,
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise ValueError("NuCLS selected source inventory is empty")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return tuple(records), hashlib.sha256(payload.encode()).hexdigest()


def load_frozen_nucls_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the public freeze and reject any changed protocol/config bytes."""

    root = Path(repository_root).resolve()
    config_path = root / "configs" / "nucls_external_validation.yaml"
    protocol_path = root / "NUCLS_EXTERNAL_VALIDATION_PREREGISTRATION.md"
    freeze_path = root / "reports" / "nucls_external_validation_freeze.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = freeze.get("frozen_files", {})
    actual = {
        "NUCLS_EXTERNAL_VALIDATION_PREREGISTRATION.md": sha256_file(protocol_path),
        "configs/nucls_external_validation.yaml": sha256_file(config_path),
    }
    if expected != actual:
        raise RuntimeError(
            "NuCLS external-validation freeze does not match protocol/config bytes; "
            f"expected={expected!r}, actual={actual!r}"
        )
    if config.get("freeze_state") != "frozen_before_outcome_inspection":
        raise RuntimeError("NuCLS configuration is not frozen before outcome inspection")
    if config.get("outcome_tables_inspected_before_freeze") is not False:
        raise RuntimeError("NuCLS freeze does not certify outcome blinding")
    return config, freeze


def _normalise_raw_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    lower = rendered.casefold()
    if lower.startswith("correction_"):
        lower = lower.removeprefix("correction_")
    return lower


def _map_label(value: Any) -> tuple[int | None, str]:
    raw = _normalise_raw_label(value)
    if raw is None:
        return None, "missing_label"
    if raw in EXCLUDED_LABELS:
        return None, f"excluded_label:{raw}"
    try:
        superclass = RAW_TO_SUPER[raw]
    except KeyError as error:
        raise ValueError(f"unknown NuCLS raw label encountered: {value!r}") from error
    return CLASS_CODES[superclass], superclass


def _read_contours(directory: Path, *, role: str) -> pd.DataFrame:
    contour_root = directory / "contours"
    files = sorted(contour_root.glob("*.csv"), key=lambda path: path.name.casefold())
    if not files:
        raise FileNotFoundError(f"{role} contour CSV files are missing: {contour_root}")
    frames: list[pd.DataFrame] = []
    required = {"anchor_id", "group", "xmin", "ymin", "xmax", "ymax"}
    for path in files:
        frame = pd.read_csv(path)
        frame = frame.loc[
            :, [column for column in frame.columns if not column.startswith("Unnamed:")]
        ]
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{role} contour file {path.name} lacks columns: {sorted(missing)}")
        frame = frame.copy()
        frame["contour_file"] = path.resolve().as_posix()
        frame["contour_stem"] = path.stem
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["anchor_id"] = combined["anchor_id"].astype(str)
    duplicates = combined[combined["anchor_id"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(f"{role} contains duplicate anchor_id values")
    return combined


def _index_rgb_images(directory: Path) -> dict[str, Path]:
    candidates: list[Path] = []
    for folder_name in ("rgb", "rgbs", "images", "image", "fov_images", "png"):
        folder = directory / folder_name
        if folder.is_dir():
            candidates.extend(
                path
                for path in folder.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
    if not candidates:
        raise FileNotFoundError(f"no NuCLS RGB image directory found below {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(candidates, key=lambda item: item.as_posix().casefold()):
        if path.stem in indexed:
            raise ValueError(f"duplicate NuCLS RGB image stem: {path.stem}")
        indexed[path.stem] = path.resolve()
    return indexed


def _fov_key(anchor_id: str) -> str:
    match = FOV_TAIL_RE.search(anchor_id)
    if match is None:
        raise ValueError(f"anchor_id lacks a TCGA FOV authority: {anchor_id!r}")
    return match.group(1)


def _identities(authority: str) -> tuple[str, str]:
    patient = TCGA_PATIENT_RE.search(authority)
    slide = TCGA_SLIDE_RE.search(authority)
    if patient is None or slide is None:
        raise ValueError(f"cannot derive TCGA patient/slide identity from {authority!r}")
    return patient.group(1).upper(), slide.group(1).upper()


def _fixed_crop(
    image: NDArray[np.uint8], *, centre_x: int, centre_y: int, size: int
) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("NuCLS source image must be uint8 RGB")
    if size <= 0 or size % 2:
        raise ValueError("frozen NuCLS crop size must be a positive even integer")
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")
    shifted_x = centre_x + half
    shifted_y = centre_y + half
    crop = padded[shifted_y - half : shifted_y + half, shifted_x - half : shifted_x + half]
    if crop.shape != (size, size, 3):
        raise ValueError("NuCLS crop geometry does not produce the frozen output shape")
    return np.asarray(crop, dtype=np.uint8)


def prepare_nucls_subset(
    np_truth_directory: str | Path,
    p_truth_directory: str | Path,
    *,
    subset_name: str,
    crop_size: int = 64,
) -> NuCLSPreparedData:
    """Pair exact NP/P anchors and construct label-independent NP-centred crops."""

    np_root = Path(np_truth_directory).resolve()
    p_root = Path(p_truth_directory).resolve()
    observed = _read_contours(np_root, role="NP-label")
    master_files = sorted(p_root.glob("v3.1_final_anchors_*_Ps_AreTruth.csv"))
    if len(master_files) != 1:
        raise FileNotFoundError(
            "NuCLS P-truth directory must contain exactly one official "
            "v3.1_final_anchors_*_Ps_AreTruth.csv table"
        )
    master_path = master_files[0].resolve()
    master = pd.read_csv(master_path)
    master_required = {"anchor_id", "EM_inferred_label_NPs", "EM_inferred_label_Ps"}
    master_missing = master_required.difference(master.columns)
    if master_missing:
        raise ValueError(f"NuCLS P-truth table lacks columns: {sorted(master_missing)}")
    master = master.loc[:, sorted(master_required)].copy()
    master["anchor_id"] = master["anchor_id"].astype(str)
    if master["anchor_id"].duplicated().any():
        raise ValueError("NuCLS P-truth table contains duplicate anchor_id values")
    images = _index_rgb_images(np_root)
    source_inventory, source_inventory_sha = _selected_inventory(
        {
            "np_contours": (
                np_root,
                tuple(Path(value) for value in observed["contour_file"].unique()),
            ),
            "np_rgb": (np_root, tuple(images.values())),
            "p_truth_master": (p_root, (master_path,)),
        }
    )
    master_by_anchor = master.set_index("anchor_id").to_dict(orient="index")

    exclusions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    crops: list[NDArray[np.uint8]] = []
    image_cache: dict[Path, NDArray[np.uint8]] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for row in observed.sort_values("anchor_id", kind="mergesort").itertuples(index=False):
        anchor_id = str(row.anchor_id)
        if anchor_id not in master_by_anchor:
            exclude("missing_exact_p_truth_anchor")
            continue
        authorities = master_by_anchor[anchor_id]
        observed_code, observed_name = _map_label(authorities["EM_inferred_label_NPs"])
        reference_code, reference_name = _map_label(authorities["EM_inferred_label_Ps"])
        if observed_code is None:
            exclude(observed_name)
            continue
        if reference_code is None:
            exclude(f"p_truth_{reference_name}")
            continue
        exported_code, _ = _map_label(row.group)
        if exported_code != observed_code:
            raise RuntimeError(
                "NuCLS NP contour export conflicts with EM_inferred_label_NPs for "
                f"anchor {anchor_id}"
            )
        image_path = images.get(str(row.contour_stem))
        if image_path is None:
            exclude("missing_exact_np_rgb_image")
            continue
        xmin = float(cast(Any, row.xmin))
        ymin = float(cast(Any, row.ymin))
        xmax = float(cast(Any, row.xmax))
        ymax = float(cast(Any, row.ymax))
        coordinates = np.asarray([xmin, ymin, xmax, ymax], dtype=np.float64)
        if not np.isfinite(coordinates).all() or xmax <= xmin or ymax <= ymin:
            exclude("invalid_np_anchor_box")
            continue
        if image_path not in image_cache:
            with Image.open(image_path) as opened:
                image_cache[image_path] = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        image = image_cache[image_path]
        centre_x = math.floor((xmin + xmax) / 2.0)
        centre_y = math.floor((ymin + ymax) / 2.0)
        if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
            exclude("np_anchor_centre_outside_rgb")
            continue
        fov = _fov_key(anchor_id)
        patient_id, slide_id = _identities(fov)
        sample_digest = hashlib.sha256(f"{subset_name}\0{anchor_id}".encode()).hexdigest()
        records.append(
            {
                "sample_id": f"nucls-{subset_name}-{sample_digest[:24]}",
                "subset": subset_name,
                "anchor_id": anchor_id,
                "fov_id": fov,
                "patient_id": patient_id,
                "slide_id": slide_id,
                "group_id": patient_id,
                "observed_label": int(observed_code),
                "observed_label_name": observed_name,
                "reference_label": int(reference_code),
                "reference_label_name": reference_name,
                "natural_disagreement": bool(observed_code != reference_code),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "np_contour_file": Path(str(cast(Any, row.contour_file)))
                .relative_to(np_root)
                .as_posix(),
                "np_rgb_file": image_path.relative_to(np_root).as_posix(),
                "p_truth_master_file": master_path.relative_to(p_root).as_posix(),
            }
        )
        crops.append(_fixed_crop(image, centre_x=centre_x, centre_y=centre_y, size=crop_size))

    if not records:
        raise ValueError("NuCLS exact-anchor pairing produced no eligible samples")
    manifest = pd.DataFrame.from_records(records).sort_values("sample_id", kind="mergesort")
    order = manifest.index.to_numpy(dtype=np.int64)
    manifest = manifest.reset_index(drop=True)
    crop_array = np.stack(crops, axis=0)[order].astype(np.uint8, copy=False)
    if manifest["sample_id"].duplicated().any() or manifest["anchor_id"].duplicated().any():
        raise RuntimeError("canonical NuCLS manifest contains duplicate identities")
    missing_group_fraction = float(manifest["group_id"].isna().mean())
    if missing_group_fraction > 0.05:
        raise RuntimeError("more than 5% of NuCLS anchors lack patient/slide grouping")
    if manifest["group_id"].nunique() < 5:
        raise RuntimeError("fewer than five patient groups are available for frozen folds")
    if set(manifest["observed_label"].unique()) != set(range(len(CLASS_ORDER))):
        raise RuntimeError("eligible NuCLS observed labels do not contain every frozen class")
    manifest_sha = _canonical_records_sha256(manifest)
    crops_sha = _array_sha256(crop_array)
    return NuCLSPreparedData(
        manifest=manifest,
        crops=crop_array,
        exclusions=dict(sorted(exclusions.items())),
        source_inventory=source_inventory,
        source_inventory_sha256=source_inventory_sha,
        manifest_sha256=manifest_sha,
        crops_sha256=crops_sha,
    )


def extract_nucls_embeddings(
    prepared: NuCLSPreparedData,
    *,
    cache_path: str | Path,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Extract the frozen official ResNet-18 representation."""

    destination = Path(cache_path).resolve()
    if destination.is_file():
        cached = load_embedding_cache(destination)
        cached.validate()
        expected_ids = prepared.manifest["sample_id"].astype(str).tolist()
        if cached.sample_ids.tolist() != expected_ids:
            raise RuntimeError("cached NuCLS embeddings do not match canonical sample order")
        if cached.metadata.get("manifest_sha256") != prepared.manifest_sha256:
            raise RuntimeError("cached NuCLS embeddings do not match canonical manifest")
        if cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256:
            raise RuntimeError("cached NuCLS embeddings do not match source inventory")
        return np.asarray(cached.embeddings, dtype=np.float32)
    result = extract_resnet18_embeddings(
        prepared.crops,
        prepared.manifest["sample_id"].astype(str).tolist(),
        config=ResNet18EmbeddingConfig(
            weight_identifier="IMAGENET1K_V1",
            input_variant="rgb",
            device=device,
            batch_size=32,
            minimum_batch_size=1,
            use_amp=True,
            output_dtype="float32",
            allow_weight_download=False,
        ),
        cache_path=destination,
        manifest_sha256=prepared.manifest_sha256,
        raw_inventory_sha256=prepared.source_inventory_sha256,
        representation_id="nucls_resnet18_imagenet1k_v1_context_64px",
        analysis_eligibility={
            "study_id": "nucls_natural_label_external_validation_v1",
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
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=n_splits,
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=split_seed,
    )
    matrix = np.asarray(features, dtype=np.float64)
    probabilities = np.full((len(labels), len(CLASS_ORDER)), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for fold in plan.folds:
        model = MultinomialLogisticRegression(
            class_order=tuple(range(len(CLASS_ORDER))),
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=True,
        ).fit(matrix[fold.train_indices], labels[fold.train_indices])
        probabilities[fold.holdout_indices] = model.predict_proba(matrix[fold.holdout_indices])
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
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise RuntimeError("NuCLS OOF probabilities are incomplete or invalid")
    return probabilities, {
        "splitter": plan.splitter_class_name,
        "fallback_status": plan.splitter_fallback_status,
        "fallback_reason": plan.splitter_fallback_reason,
        "folds": folds,
    }


def _nucls_audit_risk(
    features: NDArray[np.generic],
    observed_labels: NDArray[np.int64],
    probabilities: NDArray[np.generic],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    method: str,
    n_splits: int,
    split_seed: int,
    neighbour_k: int = 7,
    neighbour_metric: str = "cosine",
    hybrid_weights: Sequence[float] = (0.5, 0.5),
) -> GroupSafeAuditScoreResult:
    """Recreate exact fold provenance and build one group-safe NuCLS risk vector."""

    plan = make_group_stratified_fold_plan(
        observed_labels,
        group_ids,
        n_splits=n_splits,
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=split_seed,
    )
    fold_ids = np.full(len(observed_labels), -1, dtype=np.int64)
    training_groups_by_fold: dict[int, tuple[str, ...]] = {}
    for fold in plan.folds:
        fold_ids[fold.holdout_indices] = fold.fold_id
        training_groups_by_fold[fold.fold_id] = fold.training_groups
    if np.any(fold_ids < 0):
        raise RuntimeError("NuCLS score construction did not cover every sample")
    return group_safe_audit_scores(
        features,
        observed_labels,
        probabilities,
        group_ids,
        fold_ids,
        training_groups_by_fold,
        sample_ids=sample_ids,
        method=method,
        class_order=tuple(range(len(CLASS_ORDER))),
        neighbour_k=neighbour_k,
        neighbour_metric=neighbour_metric,
        hybrid_weights=hybrid_weights,
    )


def _interval(values: Sequence[float], *, estimate: float, requested: int) -> IntervalSummary:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("no valid bootstrap values")
    return IntervalSummary(
        estimate=float(estimate),
        mean=float(array.mean()),
        interval_95=(float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))),
        valid_iterations=len(array),
        requested_iterations=requested,
    )


def _ranking_validation(
    manifest: pd.DataFrame,
    features: NDArray[np.generic],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, NDArray[np.generic]]]:
    ranking = config["ranking"]
    model_config = config["model"]
    observed = manifest["observed_label"].to_numpy(dtype=np.int64)
    reference = manifest["reference_label"].to_numpy(dtype=np.int64)
    events = observed != reference
    groups = manifest["group_id"].astype(str).tolist()
    probabilities, fold_evidence = _oof_probabilities(
        features,
        observed,
        groups,
        n_splits=int(ranking["folds"]),
        split_seed=int(ranking["split_seed"]),
        l2=float(model_config["l2"]),
        max_iter=int(model_config["max_iter"]),
    )
    score_result = _nucls_audit_risk(
        features,
        observed,
        probabilities,
        groups,
        manifest["sample_id"].astype(str).tolist(),
        method=str(ranking["score"]),
        n_splits=int(ranking["folds"]),
        split_seed=int(ranking["split_seed"]),
        neighbour_k=int(ranking.get("neighbour_k", 7)),
        neighbour_metric=str(ranking.get("neighbour_metric", "cosine")),
        hybrid_weights=tuple(float(value) for value in ranking.get("hybrid_weights", (0.5, 0.5))),
    )
    risk = score_result.risk_scores
    prevalence = float(events.mean())
    ap = average_precision(events, risk)
    if ap is None:
        raise RuntimeError("NuCLS primary subset contains no natural disagreements")
    auroc = binary_auroc(events, risk)
    budgets: dict[str, Any] = {}
    for raw_budget in ranking["budgets"]:
        budget = float(raw_budget)
        count = budget_count(len(events), budget)
        selected = rank_indices(risk, tie_break_ids=manifest["sample_id"].tolist())[:count]
        found = int(events[selected].sum())
        precision = found / count if count else None
        recall = found / int(events.sum()) if events.sum() else None
        budgets[str(budget)] = {
            "fraction": budget,
            "reviewed_count": count,
            "disagreements_found": found,
            "precision": precision,
            "recall": recall,
            "lift_over_prevalence": precision / prevalence
            if precision is not None and prevalence
            else None,
        }

    primary_budget = float(ranking["primary_budget"])
    primary_count = budget_count(len(events), primary_budget)
    primary_selected = rank_indices(risk, tie_break_ids=manifest["sample_id"].tolist())[
        :primary_count
    ]
    primary_precision = float(events[primary_selected].mean())
    draws = draw_group_bootstrap_indices(
        groups,
        n_iterations=int(ranking["bootstrap_iterations"]),
        seed=int(ranking["bootstrap_seed"]),
    )
    ap_differences: list[float] = []
    precision_differences: list[float] = []
    for indices in draws:
        sampled_events = events[indices]
        sampled_prevalence = float(sampled_events.mean())
        sampled_ap = average_precision(sampled_events, risk[indices])
        if sampled_ap is not None:
            ap_differences.append(float(sampled_ap - sampled_prevalence))
        sampled_count = budget_count(len(indices), primary_budget)
        sampled_selected = rank_indices(risk[indices])[:sampled_count]
        sampled_precision = float(sampled_events[sampled_selected].mean())
        precision_differences.append(sampled_precision - sampled_prevalence)
    ap_interval = _interval(
        ap_differences,
        estimate=float(ap - prevalence),
        requested=len(draws),
    )
    precision_interval = _interval(
        precision_differences,
        estimate=primary_precision - prevalence,
        requested=len(draws),
    )

    random_aps: list[float] = []
    random_precisions: list[float] = []
    random_seed = int(ranking["random_seed"])
    for repeat in range(int(ranking["random_repetitions"])):
        rng = np.random.default_rng(random_seed + repeat)
        random_scores = rng.random(len(events))
        random_ap = average_precision(events, random_scores)
        if random_ap is not None:
            random_aps.append(float(random_ap))
        selected = rank_indices(random_scores)[:primary_count]
        random_precisions.append(float(events[selected].mean()))

    supported = ap_interval.interval_95[0] > 0.0 and precision_interval.interval_95[0] > 0.0
    result = {
        "sample_count": len(manifest),
        "group_count": manifest["group_id"].nunique(),
        "natural_disagreement_count": int(events.sum()),
        "natural_disagreement_prevalence": prevalence,
        "average_precision": float(ap),
        "auroc": float(auroc) if auroc is not None else None,
        "budgets": budgets,
        "ap_minus_prevalence": asdict(ap_interval),
        "precision_at_5_percent_minus_prevalence": asdict(precision_interval),
        "random_rankings": {
            "repetitions": len(random_aps),
            "ap_mean": float(np.mean(random_aps)),
            "ap_interval_95": [float(value) for value in np.quantile(random_aps, [0.025, 0.975])],
            "precision_mean": float(np.mean(random_precisions)),
            "precision_interval_95": [
                float(value) for value in np.quantile(random_precisions, [0.025, 0.975])
            ],
        },
        "fold_evidence": fold_evidence,
        "success_conditions_met": supported,
        "claim": (
            "supports prioritisation of natural NP/P disagreements in this NuCLS subset"
            if supported
            else "does not establish prioritisation of natural NP/P disagreements"
        ),
    }
    if str(ranking["score"]) != "one_minus_probability_of_observed_label":
        result["risk_strategy"] = score_result.as_dict()
    arrays: dict[str, NDArray[np.generic]] = {
        "observed_labels": observed,
        "reference_labels": reference,
        "natural_disagreement": events,
        "oof_probabilities": probabilities,
        "risk_scores": risk,
        "ranking_ap_minus_prevalence_bootstrap": np.asarray(ap_differences, dtype=np.float64),
        "ranking_precision_minus_prevalence_bootstrap": np.asarray(
            precision_differences, dtype=np.float64
        ),
        "random_ranking_ap": np.asarray(random_aps, dtype=np.float64),
        "random_ranking_precision": np.asarray(random_precisions, dtype=np.float64),
    }
    return result, arrays


def _fit_probabilities(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
    *,
    l2: float,
    max_iter: int,
) -> NDArray[np.float64]:
    model = MultinomialLogisticRegression(
        class_order=tuple(range(len(CLASS_ORDER))),
        l2=l2,
        max_iter=max_iter,
        class_weight_balanced=True,
    ).fit(features[train_indices], labels[train_indices])
    return model.predict_proba(features[test_indices])


def _group_confusion_tensor(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.generic],
    groups: Sequence[str],
    unique_groups: Sequence[str],
) -> NDArray[np.int64]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    if matrix.ndim != 3 or matrix.shape[1:] != (len(reference), len(CLASS_ORDER)):
        raise ValueError("probabilities do not align for grouped confusion calculation")
    group_values = np.asarray(groups, dtype=np.str_)
    predictions = np.argmax(matrix, axis=2)
    tensor = np.zeros(
        (matrix.shape[0], len(unique_groups), len(CLASS_ORDER), len(CLASS_ORDER)),
        dtype=np.int64,
    )
    for group_index, group in enumerate(unique_groups):
        members = np.flatnonzero(group_values == group)
        for repeat in range(matrix.shape[0]):
            np.add.at(
                tensor[repeat, group_index], (reference[members], predictions[repeat, members]), 1
            )
    return tensor


def _macro_f1_from_confusions(confusions: NDArray[np.integer]) -> NDArray[np.float64]:
    matrix = np.asarray(confusions, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    predicted_count = matrix.sum(axis=-2)
    actual_count = matrix.sum(axis=-1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return f1.mean(axis=-1)


def _downstream_validation(
    manifest: pd.DataFrame,
    features: NDArray[np.generic],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, NDArray[np.generic]]]:
    downstream = config["downstream"]
    model_config = config["model"]
    matrix = np.asarray(features, dtype=np.float64)
    observed = manifest["observed_label"].to_numpy(dtype=np.int64)
    reference = manifest["reference_label"].to_numpy(dtype=np.int64)
    groups = manifest["group_id"].astype(str).tolist()
    outer = make_group_stratified_fold_plan(
        observed,
        groups,
        n_splits=int(downstream["outer_folds"]),
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=int(downstream["outer_split_seed"]),
    )
    shape = (len(manifest), len(CLASS_ORDER))
    uncorrected_probabilities = np.full(shape, np.nan, dtype=np.float64)
    guided_probabilities = np.full(shape, np.nan, dtype=np.float64)
    reference_probabilities = np.full(shape, np.nan, dtype=np.float64)
    repeats = int(downstream["random_repetitions"])
    random_probabilities = np.full((repeats, *shape), np.nan, dtype=np.float32)
    review_counts: list[dict[str, Any]] = []
    l2 = float(model_config["l2"])
    max_iter = int(model_config["max_iter"])
    ranking_config = config["ranking"]
    audit_method = str(downstream.get("audit_score", ranking_config["score"]))
    audit_neighbour_k = int(downstream.get("neighbour_k", ranking_config.get("neighbour_k", 7)))
    audit_neighbour_metric = str(
        downstream.get("neighbour_metric", ranking_config.get("neighbour_metric", "cosine"))
    )
    audit_hybrid_weights = tuple(
        float(value)
        for value in downstream.get(
            "hybrid_weights", ranking_config.get("hybrid_weights", (0.5, 0.5))
        )
    )

    for fold in outer.folds:
        train = fold.train_indices.astype(np.int64)
        test = fold.holdout_indices.astype(np.int64)
        train_groups = [groups[index] for index in train]
        inner_probabilities, inner_evidence = _oof_probabilities(
            matrix[train],
            observed[train],
            train_groups,
            n_splits=int(downstream["inner_audit_folds"]),
            split_seed=int(downstream["inner_split_seed"]) + fold.fold_id,
            l2=l2,
            max_iter=max_iter,
        )
        inner_score = _nucls_audit_risk(
            matrix[train],
            observed[train],
            inner_probabilities,
            train_groups,
            manifest.iloc[train]["sample_id"].astype(str).tolist(),
            method=audit_method,
            n_splits=int(downstream["inner_audit_folds"]),
            split_seed=int(downstream["inner_split_seed"]) + fold.fold_id,
            neighbour_k=audit_neighbour_k,
            neighbour_metric=audit_neighbour_metric,
            hybrid_weights=audit_hybrid_weights,
        )
        inner_risk = inner_score.risk_scores
        review_count = budget_count(len(train), float(downstream["review_budget"]))
        guided_local = rank_indices(
            inner_risk,
            tie_break_ids=manifest.iloc[train]["sample_id"].astype(str).tolist(),
        )[:review_count]
        guided_labels = observed.copy()
        guided_global = train[guided_local]
        guided_changed = guided_global[observed[guided_global] != reference[guided_global]]
        guided_labels[guided_changed] = reference[guided_changed]

        uncorrected_probabilities[test] = _fit_probabilities(
            matrix, observed, train, test, l2=l2, max_iter=max_iter
        )
        guided_probabilities[test] = _fit_probabilities(
            matrix, guided_labels, train, test, l2=l2, max_iter=max_iter
        )
        reference_probabilities[test] = _fit_probabilities(
            matrix, reference, train, test, l2=l2, max_iter=max_iter
        )

        random_changed_counts: list[int] = []
        for repeat in range(repeats):
            repeat_seed = (
                int(downstream["random_seed_start"])
                + repeat * int(downstream["outer_folds"])
                + fold.fold_id
            )
            rng = np.random.default_rng(repeat_seed)
            random_local = rng.choice(len(train), size=review_count, replace=False)
            random_global = train[random_local]
            random_labels = observed.copy()
            changed = random_global[observed[random_global] != reference[random_global]]
            random_labels[changed] = reference[changed]
            random_changed_counts.append(len(changed))
            random_probabilities[repeat, test] = _fit_probabilities(
                matrix, random_labels, train, test, l2=l2, max_iter=max_iter
            ).astype(np.float32)
        review_record = {
            "fold_id": fold.fold_id,
            "training_groups": list(fold.training_groups),
            "held_out_groups": list(fold.held_out_groups),
            "training_count": len(train),
            "test_count": len(test),
            "review_count": review_count,
            "guided_disagreements_corrected": len(guided_changed),
            "random_disagreements_corrected_mean": float(np.mean(random_changed_counts)),
            "inner_fold_evidence": inner_evidence,
        }
        if audit_method != "one_minus_probability_of_observed_label":
            review_record["risk_strategy"] = inner_score.as_dict()
        review_counts.append(review_record)

    arrays_to_check: list[NDArray[np.generic]] = [
        uncorrected_probabilities,
        guided_probabilities,
        reference_probabilities,
        random_probabilities,
    ]
    if any(not np.isfinite(array).all() for array in arrays_to_check):
        raise RuntimeError("NuCLS downstream probabilities are incomplete")
    uncorrected_metrics = classification_metrics(
        reference, uncorrected_probabilities, class_order=tuple(range(len(CLASS_ORDER)))
    )
    guided_metrics = classification_metrics(
        reference, guided_probabilities, class_order=tuple(range(len(CLASS_ORDER)))
    )
    ceiling_metrics = classification_metrics(
        reference, reference_probabilities, class_order=tuple(range(len(CLASS_ORDER)))
    )
    random_metrics = [
        classification_metrics(
            reference,
            random_probabilities[repeat],
            class_order=tuple(range(len(CLASS_ORDER))),
        )
        for repeat in range(repeats)
    ]
    random_macro_f1 = np.asarray([metric.macro_f1 for metric in random_metrics], dtype=np.float64)
    estimate_guided_random = guided_metrics.macro_f1 - float(random_macro_f1.mean())
    estimate_guided_uncorrected = guided_metrics.macro_f1 - uncorrected_metrics.macro_f1

    unique_groups = tuple(sorted(set(groups)))
    guided_group_confusions = _group_confusion_tensor(
        reference, guided_probabilities, groups, unique_groups
    )[0]
    uncorrected_group_confusions = _group_confusion_tensor(
        reference, uncorrected_probabilities, groups, unique_groups
    )[0]
    random_group_confusions = _group_confusion_tensor(
        reference, random_probabilities, groups, unique_groups
    )
    requested_bootstrap = int(downstream["bootstrap_iterations"])
    bootstrap_rng = np.random.default_rng(int(downstream["bootstrap_seed"]))
    guided_random_differences: list[float] = []
    guided_uncorrected_differences: list[float] = []
    for _ in range(requested_bootstrap):
        sampled_groups = bootstrap_rng.integers(0, len(unique_groups), size=len(unique_groups))
        guided_confusion = guided_group_confusions[sampled_groups].sum(axis=0)
        uncorrected_confusion = uncorrected_group_confusions[sampled_groups].sum(axis=0)
        random_confusions = random_group_confusions[:, sampled_groups].sum(axis=1)
        guided_boot = float(_macro_f1_from_confusions(guided_confusion)[0])
        uncorrected_boot = float(_macro_f1_from_confusions(uncorrected_confusion)[0])
        random_boot = float(_macro_f1_from_confusions(random_confusions).mean())
        guided_random_differences.append(float(guided_boot - random_boot))
        guided_uncorrected_differences.append(float(guided_boot - uncorrected_boot))

    guided_random_interval = _interval(
        guided_random_differences,
        estimate=estimate_guided_random,
        requested=requested_bootstrap,
    )
    guided_uncorrected_interval = _interval(
        guided_uncorrected_differences,
        estimate=estimate_guided_uncorrected,
        requested=requested_bootstrap,
    )
    supported = (
        guided_random_interval.interval_95[0] > 0.0
        and guided_uncorrected_interval.interval_95[0] > 0.0
    )
    result: dict[str, Any] = {
        "primary_metric": "macro_f1",
        "uncorrected_observed": asdict(uncorrected_metrics),
        "audit_guided_review": asdict(guided_metrics),
        "random_review": {
            "repetitions": repeats,
            "macro_f1_mean": float(random_macro_f1.mean()),
            "macro_f1_interval_95_across_repetitions": [
                float(value) for value in np.quantile(random_macro_f1, [0.025, 0.975])
            ],
        },
        "pathologist_reference_ceiling": asdict(ceiling_metrics),
        "guided_minus_mean_random_macro_f1": asdict(guided_random_interval),
        "guided_minus_uncorrected_macro_f1": asdict(guided_uncorrected_interval),
        "outer_splitter": outer.splitter_class_name,
        "outer_fallback_status": outer.splitter_fallback_status,
        "outer_fallback_reason": outer.splitter_fallback_reason,
        "folds": review_counts,
        "success_conditions_met": supported,
        "claim": (
            "supports retrospective downstream utility in this NuCLS subset"
            if supported
            else "does not establish retrospective downstream utility"
        ),
    }
    arrays: dict[str, NDArray[np.generic]] = {
        "downstream_uncorrected_probabilities": uncorrected_probabilities,
        "downstream_guided_probabilities": guided_probabilities,
        "downstream_reference_ceiling_probabilities": reference_probabilities,
        "downstream_random_probabilities": random_probabilities,
        "downstream_random_macro_f1": random_macro_f1,
        "downstream_guided_minus_random_bootstrap": np.asarray(
            guided_random_differences, dtype=np.float64
        ),
        "downstream_guided_minus_uncorrected_bootstrap": np.asarray(
            guided_uncorrected_differences, dtype=np.float64
        ),
    }
    guard_config = downstream.get("application_guard")
    if isinstance(guard_config, Mapping) and guard_config.get("enabled") is True:
        guard = evaluate_retraining_guard(
            reference,
            uncorrected_probabilities,
            guided_probabilities,
            groups,
            class_order=tuple(range(len(CLASS_ORDER))),
            evidence_role=str(guard_config.get("evidence_role", "")),
            n_iterations=int(guard_config.get("bootstrap_iterations", 2000)),
            seed=int(guard_config.get("bootstrap_seed", 26082071)),
            minimum_effect=float(guard_config.get("minimum_macro_f1_effect", 0.0)),
        )
        guarded_probabilities = (
            guided_probabilities if guard.apply_candidate else uncorrected_probabilities
        )
        result["application_guard"] = guard.as_dict()
        result["guarded_application"] = {
            "selected_condition": (
                "audit_guided_review" if guard.apply_candidate else "uncorrected_observed"
            ),
            "metrics": asdict(
                classification_metrics(
                    reference,
                    guarded_probabilities,
                    class_order=tuple(range(len(CLASS_ORDER))),
                )
            ),
            "automatic_source_annotation_modification": False,
        }
        arrays["downstream_guarded_probabilities"] = guarded_probabilities
    return result, arrays


def run_nucls_external_validation(
    prepared: NuCLSPreparedData,
    embeddings: NDArray[np.generic],
    *,
    config: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Execute the frozen ranking and downstream analyses and seal numeric evidence."""

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"NuCLS validation output already exists: {destination}")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (len(prepared.manifest), 512) or not np.isfinite(matrix).all():
        raise ValueError("NuCLS embeddings must be finite and aligned 512-vectors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        prepared.manifest.to_csv(
            staging / "canonical_manifest.csv", index=False, lineterminator="\n"
        )
        dataset_config = next(
            dataset
            for dataset in config["datasets"].values()
            if str(dataset["name"]) == str(prepared.manifest["subset"].iloc[0])
        )
        atomic_write_json(
            staging / "source_inventory.json",
            {
                "schema_version": 1,
                "study_id": config["study_id"],
                "subset": str(prepared.manifest["subset"].iloc[0]),
                "official_repository": config["authorities"]["official_repository"],
                "official_repository_commit": config["authorities"]["official_repository_commit"],
                "official_folder_ids": {
                    key: dataset_config[key]
                    for key in ("raw_folder_id", "np_truth_folder_id", "p_truth_folder_id")
                },
                "license": config["authorities"]["license"],
                "files": list(prepared.source_inventory),
                "files_canonical_sha256": prepared.source_inventory_sha256,
            },
        )
        ranking, ranking_arrays = _ranking_validation(prepared.manifest, matrix, config)
        downstream, downstream_arrays = _downstream_validation(prepared.manifest, matrix, config)
        evidence_path = staging / "numeric_evidence.npz"
        atomic_write_npz(
            evidence_path,
            {
                "sample_ids": prepared.manifest["sample_id"].to_numpy(dtype=np.str_),
                "group_ids": prepared.manifest["group_id"].to_numpy(dtype=np.str_),
                "embeddings": matrix,
                **ranking_arrays,
                **downstream_arrays,
            },
        )
        result = {
            "schema_version": 1,
            "study_id": config["study_id"],
            "subset": str(prepared.manifest["subset"].iloc[0]),
            "status": "completed",
            "reference_interpretation": (
                "natural NP-label disagreement relative to inferred pathologist consensus; "
                "not guaranteed biological truth"
            ),
            "sample_count": len(prepared.manifest),
            "patient_group_count": prepared.manifest["group_id"].nunique(),
            "class_order": list(CLASS_ORDER),
            "exclusions": prepared.exclusions,
            "source_inventory_sha256": prepared.source_inventory_sha256,
            "manifest_sha256": prepared.manifest_sha256,
            "crops_sha256": prepared.crops_sha256,
            "embeddings_sha256": _array_sha256(matrix),
            "ranking": ranking,
            "downstream": downstream,
            "claim_boundary": {
                "pathologist_error_proven": False,
                "biological_truth_proven": False,
                "clinical_utility_proven": False,
                "automatic_source_changes_permitted": False,
            },
        }
        atomic_write_json(staging / "results.json", result)
        atomic_write_json(
            staging / "artifact_manifest.json",
            {
                "schema_version": 1,
                "files": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in sorted(staging.iterdir())
                    if path.is_file()
                },
            },
        )
        if destination.exists():
            raise FileExistsError(
                f"NuCLS validation output appeared during execution: {destination}"
            )
        os.replace(staging, destination)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "CLASS_ORDER",
    "NuCLSPreparedData",
    "extract_nucls_embeddings",
    "load_frozen_nucls_config",
    "prepare_nucls_subset",
    "run_nucls_external_validation",
]
