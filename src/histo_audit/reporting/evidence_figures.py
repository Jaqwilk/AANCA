"""Evidence-rich synthetic audit figures reconstructed from sealed run artifacts.

The figures in this module never infer annotation truth from model disagreement.  They
visualise the controlled injected-corruption target and saved group-safe OOF evidence.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

from histo_audit.data.targets import extract_target_crop
from histo_audit.reporting.figures import save_figure

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class EvidenceFigureArtifact:
    """One PNG and exact artifact fields/selection rule used to build it."""

    key: str
    title: str
    alt_text: str
    path: Path
    sources: tuple[str, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Evidence:
    sample_ids: NDArray[np.str_]
    group_ids: NDArray[np.str_]
    tissue_types: NDArray[np.str_]
    pre: NDArray[np.int64]
    observed: NDArray[np.int64]
    injected: NDArray[np.bool_]
    probabilities: NDArray[np.float64]
    predicted: NDArray[np.int64]
    fold_ids: NDArray[np.int64]
    class_order: NDArray[np.int64]
    class_names: tuple[str, ...]
    scores: Mapping[str, NDArray[np.float64]]
    dataset_sample_ids: NDArray[np.str_]
    images: NDArray[np.uint8]
    masks: NDArray[np.bool_]
    instance_ids: NDArray[np.int64]
    corruption_rows: Mapping[str, Mapping[str, Any]]
    neighbour_ids: NDArray[np.str_]
    neighbour_groups: NDArray[np.str_]
    neighbour_distances: NDArray[np.float64]
    neighbour_count: NDArray[np.int64]
    neighbour_risk: NDArray[np.float64]
    neighbour_support: NDArray[np.float64]
    neighbour_suggested: NDArray[np.int64]

    @property
    def primary_method(self) -> str:
        if "fixed_hybrid" in self.scores:
            return "fixed_hybrid"
        if "self_confidence" in self.scores:
            return "self_confidence"
        return sorted(self.scores)[0]


@dataclass(frozen=True, slots=True)
class _BootstrapEvidence:
    status: str
    metric_name: str
    comparator: str
    methods: tuple[str, ...]
    requested_iterations: int
    valid_iterations: int
    metric_a: NDArray[np.float64]
    metric_b: NDArray[np.float64]
    differences: NDArray[np.float64]


def _npz_scalar_string(payload: Any, name: str) -> str:
    value = np.asarray(payload[name])
    if value.shape != () or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"bootstrap evidence {name} must be a saved string scalar")
    return str(value.item())


def _npz_scalar_integer(payload: Any, name: str) -> int:
    value = np.asarray(payload[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"bootstrap evidence {name} must be a saved integer scalar")
    return int(value.item())


def _finite_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _load_bootstrap_evidence(path: Path) -> _BootstrapEvidence:
    required = {
        "status",
        "reason",
        "metric_name",
        "comparator_method",
        "comparison_methods",
        "requested_iterations",
        "valid_iterations",
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
            raise ValueError(f"bootstrap evidence lacks arrays: {sorted(missing)}")
        status = _npz_scalar_string(payload, "status")
        reason = _npz_scalar_string(payload, "reason")
        metric_name = _npz_scalar_string(payload, "metric_name")
        comparator = _npz_scalar_string(payload, "comparator_method")
        methods_array = np.asarray(payload["comparison_methods"], dtype=np.str_)
        requested = _npz_scalar_integer(payload, "requested_iterations")
        valid = _npz_scalar_integer(payload, "valid_iterations")
        draw_indices = np.asarray(payload["draw_indices"], dtype=np.int64)
        draw_offsets = np.asarray(payload["draw_offsets"], dtype=np.int64)
        valid_draw_indices = np.asarray(payload["valid_draw_indices"], dtype=np.int64)
        metric_a = np.asarray(payload["metric_a"], dtype=np.float64)
        metric_b = np.asarray(payload["metric_b"], dtype=np.float64)
        differences = np.asarray(payload["differences"], dtype=np.float64)
    if (
        methods_array.ndim != 1
        or not len(methods_array)
        or len(set(methods_array.tolist())) != len(methods_array)
        or any(not str(value) for value in methods_array)
    ):
        raise ValueError("bootstrap comparison methods must be non-empty and unique")
    methods = tuple(str(value) for value in methods_array)
    if not metric_name or not comparator or requested <= 0 or valid < 0:
        raise ValueError("bootstrap evidence has invalid metric/comparator/iteration metadata")
    expected_shape = (len(methods), valid)
    if metric_a.shape != expected_shape or metric_b.shape != expected_shape:
        raise ValueError("bootstrap metric arrays do not align with methods/valid draws")
    if differences.shape != expected_shape:
        raise ValueError("bootstrap difference array does not align with methods/valid draws")
    if not np.isfinite(metric_a).all() or not np.isfinite(metric_b).all():
        raise ValueError("bootstrap metric arrays contain non-finite values")
    if not np.allclose(differences, metric_a - metric_b, rtol=1e-12, atol=1e-14):
        raise ValueError("bootstrap differences are not metric A minus metric B")
    if status == "reported":
        if reason or valid <= 0:
            raise ValueError("reported bootstrap evidence has invalid status metadata")
        if (
            draw_indices.ndim != 1
            or draw_offsets.shape != (requested + 1,)
            or draw_offsets[0] != 0
            or draw_offsets[-1] != len(draw_indices)
            or np.any(np.diff(draw_offsets) <= 0)
            or valid_draw_indices.shape != (valid,)
        ):
            raise ValueError("reported bootstrap draw indices/offsets are invalid")
    elif status == "not_applicable":
        if (
            not reason
            or valid != 0
            or draw_indices.size
            or not np.array_equal(draw_offsets, np.asarray([0], dtype=np.int64))
            or valid_draw_indices.size
            or metric_a.size
            or metric_b.size
            or differences.size
        ):
            raise ValueError("not-applicable bootstrap evidence must not contain draw results")
    else:
        raise ValueError(f"unsupported bootstrap evidence status: {status!r}")
    return _BootstrapEvidence(
        status=status,
        metric_name=metric_name,
        comparator=comparator,
        methods=methods,
        requested_iterations=requested,
        valid_iterations=valid,
        metric_a=metric_a,
        metric_b=metric_b,
        differences=differences,
    )


def _strings(value: NDArray[np.generic], name: str) -> NDArray[np.str_]:
    array = np.asarray(value, dtype=np.str_)
    if array.ndim != 1 or not len(array) or np.any(np.char.str_len(array) == 0):
        raise ValueError(f"{name} must be a non-empty vector of non-empty strings")
    return array


def _load_evidence(
    *,
    metrics: Mapping[str, Any],
    predictions_path: Path,
    rankings_path: Path,
    neighbour_evidence_path: Path,
    dataset_evidence_path: Path,
    corruption_manifest_path: Path,
    class_names: Sequence[str] | None,
) -> _Evidence:
    with np.load(predictions_path, allow_pickle=False) as payload:
        required = {
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
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"OOF predictions lack evidence arrays: {sorted(missing)}")
        sample_ids = _strings(payload["sample_ids"], "prediction sample_ids")
        group_ids = _strings(payload["group_ids"], "prediction group_ids")
        tissue_types = _strings(payload["tissue_type"], "prediction tissue_type")
        pre = np.asarray(payload["pre_corruption_label"], dtype=np.int64)
        observed = np.asarray(payload["observed_label"], dtype=np.int64)
        injected = np.asarray(payload["is_injected_corruption"], dtype=bool)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        predicted = np.asarray(payload["predicted_class"], dtype=np.int64)
        fold_ids = np.asarray(payload["fold_id"], dtype=np.int64)
        class_order = np.asarray(payload["class_order"], dtype=np.int64)
    n_samples = len(sample_ids)
    aligned_vectors = (group_ids, tissue_types, pre, observed, injected, predicted, fold_ids)
    if any(value.shape != (n_samples,) for value in aligned_vectors):
        raise ValueError("OOF example evidence is not sample-aligned")
    if len(set(sample_ids.tolist())) != n_samples:
        raise ValueError("OOF sample IDs must be unique")
    if (
        class_order.ndim != 1
        or not len(class_order)
        or len(set(class_order.tolist())) != len(class_order)
    ):
        raise ValueError("OOF class_order must be a unique vector")
    if probabilities.shape != (n_samples, len(class_order)):
        raise ValueError("OOF probability matrix shape disagrees with samples/class_order")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("OOF probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("OOF probability rows must sum to one")
    allowed = set(int(value) for value in class_order)
    if any(int(value) not in allowed for value in np.concatenate((pre, observed, predicted))):
        raise ValueError("OOF labels lie outside class_order")
    resolved_names = (
        tuple(str(value) for value in class_names)
        if class_names is not None
        else tuple(f"Class {int(value)}" for value in class_order)
    )
    if len(resolved_names) != len(class_order) or any(not value for value in resolved_names):
        raise ValueError("class_names must align with class_order")

    ranking_metrics = metrics.get("ranking")
    if not isinstance(ranking_metrics, Mapping) or not ranking_metrics:
        raise ValueError("metrics lack ranking methods for evidence figures")
    methods = tuple(sorted(str(value) for value in ranking_metrics))
    with rankings_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("ranking CSV lacks a header")
        missing_columns = {"sample_id", *methods}.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"ranking CSV lacks score columns: {sorted(missing_columns)}")
        ranking_rows = list(reader)
    if len(ranking_rows) != n_samples:
        raise ValueError("ranking CSV row count disagrees with OOF predictions")
    row_by_id = {str(row["sample_id"]): row for row in ranking_rows}
    if len(row_by_id) != n_samples or set(row_by_id) != set(sample_ids.tolist()):
        raise ValueError("ranking CSV sample IDs disagree with OOF predictions")
    scores: dict[str, NDArray[np.float64]] = {}
    for method in methods:
        try:
            values = np.asarray(
                [float(row_by_id[str(sample_id)][method]) for sample_id in sample_ids],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ranking CSV has invalid scores for {method}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"ranking CSV has non-finite scores for {method}")
        scores[method] = values

    with np.load(dataset_evidence_path, allow_pickle=False) as payload:
        required_dataset = {"sample_ids", "images", "target_masks", "instance_id"}
        missing_dataset = required_dataset.difference(payload.files)
        if missing_dataset:
            raise ValueError(f"synthetic dataset evidence lacks arrays: {sorted(missing_dataset)}")
        dataset_sample_ids = _strings(payload["sample_ids"], "dataset sample_ids")
        images = np.asarray(payload["images"], dtype=np.uint8)
        masks = np.asarray(payload["target_masks"], dtype=bool)
        instance_ids = np.asarray(payload["instance_id"], dtype=np.int64)
    n_dataset = len(dataset_sample_ids)
    if len(set(dataset_sample_ids.tolist())) != n_dataset:
        raise ValueError("dataset evidence sample IDs must be unique")
    if (
        images.ndim != 4
        or images.shape[0] != n_dataset
        or images.shape[-1] != 3
        or masks.shape != images.shape[:3]
        or instance_ids.shape != (n_dataset,)
        or np.any(~masks.reshape(n_dataset, -1).any(axis=1))
    ):
        raise ValueError("dataset image/mask/instance evidence is invalid")
    if not set(sample_ids.tolist()).issubset(dataset_sample_ids.tolist()):
        raise ValueError("dataset evidence does not contain every OOF sample")

    try:
        corruption_payload = json.loads(corruption_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("corruption manifest is not valid JSON") from exc
    if not isinstance(corruption_payload, Mapping) or not isinstance(
        corruption_payload.get("rows"), list
    ):
        raise ValueError("corruption manifest must contain a rows list")
    corruption_rows: dict[str, Mapping[str, Any]] = {}
    for row in corruption_payload["rows"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("sample_id"), str):
            raise ValueError("corruption manifest contains an invalid row")
        corruption_rows[str(row["sample_id"])] = row
    if set(corruption_rows) != set(sample_ids.tolist()):
        raise ValueError("corruption manifest sample IDs disagree with OOF predictions")

    with np.load(neighbour_evidence_path, allow_pickle=False) as payload:
        required_neighbours = {
            "sample_ids",
            "group_ids",
            "risk_scores",
            "alternative_class_support",
            "suggested_class",
            "neighbour_count",
            "neighbour_ids",
            "neighbour_groups",
            "neighbour_distances",
        }
        missing_neighbours = required_neighbours.difference(payload.files)
        if missing_neighbours:
            raise ValueError(f"neighbour evidence lacks arrays: {sorted(missing_neighbours)}")
        neighbour_sample_ids = _strings(payload["sample_ids"], "neighbour sample_ids")
        neighbour_query_groups = _strings(payload["group_ids"], "neighbour group_ids")
        neighbour_risk = np.asarray(payload["risk_scores"], dtype=np.float64)
        neighbour_support = np.asarray(payload["alternative_class_support"], dtype=np.float64)
        neighbour_suggested = np.asarray(payload["suggested_class"], dtype=np.int64)
        neighbour_count = np.asarray(payload["neighbour_count"], dtype=np.int64)
        neighbour_ids = np.asarray(payload["neighbour_ids"], dtype=np.str_)
        neighbour_groups = np.asarray(payload["neighbour_groups"], dtype=np.str_)
        neighbour_distances = np.asarray(payload["neighbour_distances"], dtype=np.float64)
    if not np.array_equal(neighbour_sample_ids, sample_ids) or not np.array_equal(
        neighbour_query_groups, group_ids
    ):
        raise ValueError("neighbour query order disagrees with OOF predictions")
    if any(
        value.shape != (n_samples,)
        for value in (neighbour_risk, neighbour_support, neighbour_suggested, neighbour_count)
    ):
        raise ValueError("neighbour scalar evidence is not sample-aligned")
    if (
        neighbour_ids.ndim != 2
        or neighbour_groups.shape != neighbour_ids.shape
        or neighbour_distances.shape != neighbour_ids.shape
        or neighbour_ids.shape[0] != n_samples
        or np.any(neighbour_count <= 0)
        or np.any(neighbour_count > neighbour_ids.shape[1])
        or not np.isfinite(neighbour_risk).all()
        or not np.isfinite(neighbour_support).all()
    ):
        raise ValueError("neighbour matrix evidence is invalid")
    sample_to_group = dict(zip(sample_ids.tolist(), group_ids.tolist(), strict=True))
    for index, count in enumerate(neighbour_count):
        active_ids = neighbour_ids[index, : int(count)]
        active_groups = neighbour_groups[index, : int(count)]
        active_distances = neighbour_distances[index, : int(count)]
        if (
            np.any(np.char.str_len(active_ids) == 0)
            or np.any(np.char.str_len(active_groups) == 0)
            or not np.isfinite(active_distances).all()
            or np.any(active_distances < 0.0)
            or str(group_ids[index]) in active_groups.tolist()
        ):
            raise ValueError("active nearest-neighbour evidence is not fold-safe/finite")
        for neighbour_id, neighbour_group in zip(active_ids, active_groups, strict=True):
            if sample_to_group.get(str(neighbour_id)) != str(neighbour_group):
                raise ValueError("nearest-neighbour ID/group provenance disagrees")

    return _Evidence(
        sample_ids=sample_ids,
        group_ids=group_ids,
        tissue_types=tissue_types,
        pre=pre,
        observed=observed,
        injected=injected,
        probabilities=probabilities,
        predicted=predicted,
        fold_ids=fold_ids,
        class_order=class_order,
        class_names=resolved_names,
        scores=scores,
        dataset_sample_ids=dataset_sample_ids,
        images=images,
        masks=masks,
        instance_ids=instance_ids,
        corruption_rows=corruption_rows,
        neighbour_ids=neighbour_ids,
        neighbour_groups=neighbour_groups,
        neighbour_distances=neighbour_distances,
        neighbour_count=neighbour_count,
        neighbour_risk=neighbour_risk,
        neighbour_support=neighbour_support,
        neighbour_suggested=neighbour_suggested,
    )


def _label(evidence: _Evidence, class_id: int) -> str:
    lookup = {int(value): index for index, value in enumerate(evidence.class_order)}
    index = lookup.get(int(class_id))
    return evidence.class_names[index] if index is not None else f"Class {class_id}"


def _contour(image: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    interior = mask.copy()
    interior[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    result = image.copy()
    result[mask & ~interior] = np.asarray([238, 32, 46], dtype=np.uint8)
    return result


def _ranked_indices(
    scores: NDArray[np.float64], sample_ids: NDArray[np.str_], *, descending: bool
) -> NDArray[np.int64]:
    ordered_scores = -scores if descending else scores
    return np.lexsort((sample_ids, ordered_scores)).astype(np.int64)


def _dataset_index(evidence: _Evidence, sample_id: str) -> int:
    matches = np.flatnonzero(evidence.dataset_sample_ids == sample_id)
    if len(matches) != 1:
        raise ValueError(f"dataset evidence does not resolve exactly one row for {sample_id}")
    return int(matches[0])


def _score_text(evidence: _Evidence, index: int) -> str:
    ordered = ["fixed_hybrid", "self_confidence", "neighbour_disagreement", "cleanlab"]
    selected = [method for method in ordered if method in evidence.scores]
    if not selected:
        selected = list(sorted(evidence.scores)[:3])
    return "; ".join(
        f"{method.replace('_', ' ')}={float(evidence.scores[method][index]):.3f}"
        for method in selected
    )


def _example_text(evidence: _Evidence, index: int, reason: str) -> str:
    sample_id = str(evidence.sample_ids[index])
    class_lookup = {int(value): position for position, value in enumerate(evidence.class_order)}
    observed_column = class_lookup[int(evidence.observed[index])]
    observed_probability = float(evidence.probabilities[index, observed_column])
    row = evidence.corruption_rows[sample_id]
    count = int(evidence.neighbour_count[index])
    neighbour_lines: list[str] = []
    id_to_index = {str(value): position for position, value in enumerate(evidence.sample_ids)}
    for neighbour_id, neighbour_group, distance in zip(
        evidence.neighbour_ids[index, : min(count, 3)],
        evidence.neighbour_groups[index, : min(count, 3)],
        evidence.neighbour_distances[index, : min(count, 3)],
        strict=True,
    ):
        neighbour_index = id_to_index[str(neighbour_id)]
        neighbour_lines.append(
            f"  {neighbour_id} | group={neighbour_group} | "
            f"obs={_label(evidence, int(evidence.observed[neighbour_index]))} | d={distance:.3f}"
        )
    replacement = row.get("replacement_class")
    replacement_text = "N/A" if replacement is None else _label(evidence, int(replacement))
    return "\n".join(
        (
            f"{sample_id} | instance={int(evidence.instance_ids[_dataset_index(evidence, sample_id)])}",
            f"group={evidence.group_ids[index]} | OOF fold={int(evidence.fold_ids[index])}",
            f"tissue={evidence.tissue_types[index]}",
            f"observed_label={_label(evidence, int(evidence.observed[index]))}",
            f"pre_corruption_label={_label(evidence, int(evidence.pre[index]))}",
            f"predicted class={_label(evidence, int(evidence.predicted[index]))}",
            f"P(observed_label)={observed_probability:.3f}",
            f"risk scores: {_score_text(evidence, index)}",
            f"review reason: {reason}",
            (
                "controlled corruption: "
                f"injected={bool(evidence.injected[index])}; type={row.get('corruption_type')}; "
                f"seed={row.get('corruption_seed')}; replacement={replacement_text}"
            ),
            (
                "fold-safe neighbours: "
                f"risk={evidence.neighbour_risk[index]:.3f}; "
                f"alternative support={evidence.neighbour_support[index]:.3f}; "
                f"suggested={_label(evidence, int(evidence.neighbour_suggested[index]))}"
            ),
            *neighbour_lines,
        )
    )


def _example_grid(
    evidence: _Evidence,
    indices: Sequence[int],
    *,
    output_directory: Path,
    filename: str,
    key: str,
    title: str,
    reason: str,
    selection_rule: str,
) -> EvidenceFigureArtifact:
    if not indices:
        raise ValueError("example grid requires at least one selected sample")
    rows = len(indices)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(14.2, max(3.4, 3.0 * rows)),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.0, 0.85, 0.85, 2.9)},
    )
    selected_ids: list[str] = []
    for row_number, query_index in enumerate(indices):
        sample_id = str(evidence.sample_ids[query_index])
        selected_ids.append(sample_id)
        dataset_index = _dataset_index(evidence, sample_id)
        image = evidence.images[dataset_index]
        mask = evidence.masks[dataset_index]
        crop = extract_target_crop(image, mask, output_size=48, padding=6)
        count = int(evidence.neighbour_count[query_index])
        nearest_id = str(evidence.neighbour_ids[query_index, 0])
        neighbour_dataset_index = _dataset_index(evidence, nearest_id)
        neighbour_crop = extract_target_crop(
            evidence.images[neighbour_dataset_index],
            evidence.masks[neighbour_dataset_index],
            output_size=48,
            padding=6,
        )
        panels = (
            (_contour(image, mask), "Full patch + exact target contour"),
            (_contour(crop.image, crop.target_mask), "Target crop + contour"),
            (
                _contour(neighbour_crop.image, neighbour_crop.target_mask),
                f"Nearest fold-safe neighbour\n{nearest_id}\n({count} retained)",
            ),
        )
        for column, (panel, panel_title) in enumerate(panels):
            axes[row_number, column].imshow(panel)
            axes[row_number, column].set_title(panel_title, fontsize=7.5)
            axes[row_number, column].axis("off")
        axes[row_number, 3].axis("off")
        axes[row_number, 3].text(
            0.0,
            1.0,
            _example_text(evidence, query_index, reason),
            transform=axes[row_number, 3].transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            family="monospace",
            linespacing=1.22,
        )
    figure.suptitle(title, fontsize=11)
    figure.text(
        0.5,
        0.006,
        "Controlled synthetic software validation; model evidence recommends expert review only.",
        ha="center",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    path = output_directory / filename
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key=key,
        title=title,
        alt_text=(
            f"{title}: full patches, target crops and contours, OOF label/probability/risk "
            "evidence, controlled corruption metadata, tissue, and fold-safe neighbours"
        ),
        path=path,
        sources=(
            "synthetic_dataset_evidence.npz:sample_ids/images/target_masks/instance_id",
            "oof_predictions.npz:sample_ids/group_ids/tissue_type/labels/probabilities/fold_id",
            "ranking.csv:sample_id and saved method risk scores",
            "corruption_manifest.json:rows",
            "neighbour_evidence.npz:all saved neighbour evidence arrays",
        ),
        provenance={
            "selection_rule": selection_rule,
            "selected_sample_ids": selected_ids,
            "ranking_method": evidence.primary_method,
            "tie_break": "sample_id ascending",
            "crop_transform": "extract_target_crop(output_size=48,padding=6)",
            "contour_transform": "four-neighbour binary-mask interior boundary",
        },
    )


def _precision_recall_figure(evidence: _Evidence, output_directory: Path) -> EvidenceFigureArtifact:
    positives = int(evidence.injected.sum())
    if positives <= 0:
        raise ValueError("precision-recall curves require injected-corruption positives")
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    colours = plt.get_cmap("tab10")
    for colour_index, method in enumerate(sorted(evidence.scores)):
        scores = evidence.scores[method]
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_injected = evidence.injected[order]
        cumulative_true = np.cumsum(sorted_injected)
        threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
        recall = cumulative_true[threshold_ends] / positives
        precision = cumulative_true[threshold_ends] / (threshold_ends + 1)
        x = np.r_[0.0, recall]
        y = np.r_[1.0, precision]
        ap = float(np.sum(np.diff(x) * y[1:]))
        axis.step(
            x,
            y,
            where="post",
            linewidth=1.55,
            color=colours(colour_index % 10),
            label=f"{method.replace('_', ' ')} (AP={ap:.3f})",
        )
    prevalence = positives / len(evidence.injected)
    axis.axhline(
        prevalence,
        color="#6b7280",
        linestyle="--",
        linewidth=1.1,
        label=f"Random prevalence ({prevalence:.3f})",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.03)
    axis.set_xlabel("Recall of injected corruptions")
    axis.set_ylabel("Precision")
    axis.set_title("Injected-corruption precision-recall curves from saved OOF rankings")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7.2, ncols=2)
    figure.tight_layout()
    path = output_directory / "precision_recall_curves.png"
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key="precision_recall_curves",
        title="Precision-recall curves",
        alt_text="Precision-recall curves for injected corruptions from each saved audit score",
        path=path,
        sources=(
            "oof_predictions.npz:is_injected_corruption",
            "ranking.csv:sample_id and saved method risk scores",
        ),
        provenance={
            "positive_event": "is_injected_corruption == true",
            "curve_rule": "descending unique score thresholds; tied scores share a threshold",
            "ap_rule": "non-interpolated area over threshold recall increments",
            "method_names": sorted(evidence.scores),
        },
    )


def _bootstrap_figure(metrics: Mapping[str, Any], output_directory: Path) -> EvidenceFigureArtifact:
    payload = metrics.get("paired_group_bootstrap_hybrid_minus_self_confidence")
    if not isinstance(payload, Mapping):
        raise ValueError("metrics lack paired hybrid/self-confidence bootstrap evidence")
    mean = payload.get("mean_difference")
    interval = payload.get("interval_95")
    probability = payload.get("probability_positive")
    valid_iterations = payload.get("valid_iterations")
    requested_iterations = payload.get("iterations")
    if (
        not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes, bytearray))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in interval
        )
        or not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not isinstance(valid_iterations, int)
        or not isinstance(requested_iterations, int)
    ):
        raise ValueError("paired bootstrap evidence is not numeric/complete")
    lower, upper = float(interval[0]), float(interval[1])
    mean_value = float(mean)
    if not np.isfinite([lower, upper, mean_value, float(probability)]).all() or lower > upper:
        raise ValueError("paired bootstrap interval evidence is invalid")
    figure, axis = plt.subplots(figsize=(8.0, 2.8))
    axis.hlines(0, lower, upper, color="#3b82a0", linewidth=5)
    axis.scatter(mean_value, 0, color="#b03a2e", s=55, zorder=3, label="Mean difference")
    axis.scatter([lower, upper], [0, 0], color="#243b53", marker="|", s=120, zorder=4)
    axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.1)
    padding = max(0.02, (upper - lower) * 0.3)
    axis.set_xlim(min(lower, mean_value, 0.0) - padding, max(upper, mean_value, 0.0) + padding)
    axis.set_yticks([0], labels=["fixed hybrid - self confidence"])
    axis.set_xlabel("Paired source-group bootstrap difference in average precision")
    axis.set_title(
        f"Mean={mean_value:.3f}; 95% interval=[{lower:.3f}, {upper:.3f}]; "
        f"P(difference>0)={float(probability):.3f}"
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "paired_bootstrap_interval.png"
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key="paired_bootstrap_interval",
        title="Paired hybrid-versus-self-confidence bootstrap interval",
        alt_text=(
            "Paired source-group bootstrap mean and 95 percent interval for the fixed hybrid "
            "minus self-confidence average-precision difference"
        ),
        path=path,
        sources=("metrics.json:paired_group_bootstrap_hybrid_minus_self_confidence",),
        provenance={
            "comparison": "fixed_hybrid minus self_confidence",
            "metric": "average_precision",
            "requested_iterations": requested_iterations,
            "valid_iterations": valid_iterations,
            "pairing_unit": "saved source groups",
        },
    )


def _paired_method_summaries(
    metrics: Mapping[str, Any],
    evidence: _BootstrapEvidence,
) -> tuple[tuple[str, float, float, float, float], ...]:
    """Reconcile saved paired draws with every reported method-level summary."""

    payload = metrics.get("paired_method_differences")
    if not isinstance(payload, Mapping) or payload.get("status") != "reported":
        raise ValueError("metrics lack reported paired method-difference evidence")
    if payload.get("metric") != evidence.metric_name:
        raise ValueError("paired method metric disagrees with bootstrap evidence")
    if payload.get("comparator") != evidence.comparator:
        raise ValueError("paired method comparator disagrees with bootstrap evidence")
    if payload.get("comparison_order") != list(evidence.methods):
        comparison_order = payload.get("comparison_order")
        if (
            not isinstance(comparison_order, Sequence)
            or isinstance(comparison_order, (str, bytes, bytearray))
            or tuple(str(value) for value in comparison_order) != evidence.methods
        ):
            raise ValueError("paired method comparison order disagrees with bootstrap evidence")
    if payload.get("iterations") != evidence.requested_iterations:
        raise ValueError("paired method requested iterations disagree with bootstrap evidence")
    if payload.get("valid_iterations") != evidence.valid_iterations:
        raise ValueError("paired method valid iterations disagree with bootstrap evidence")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(evidence.methods):
        raise ValueError("paired method summaries do not exactly match saved methods")

    summaries: list[tuple[str, float, float, float, float]] = []
    for row, method in enumerate(evidence.methods):
        saved = comparisons.get(method)
        if not isinstance(saved, Mapping):
            raise ValueError(f"paired method summary is invalid for {method}")
        interval = saved.get("interval_95")
        if (
            saved.get("method") != method
            or saved.get("comparator") != evidence.comparator
            or saved.get("metric") != evidence.metric_name
            or saved.get("iterations") != evidence.requested_iterations
            or saved.get("valid_iterations") != evidence.valid_iterations
            or not isinstance(interval, Sequence)
            or isinstance(interval, (str, bytes, bytearray))
            or len(interval) != 2
        ):
            raise ValueError(f"paired method metadata is inconsistent for {method}")
        reported = (
            _finite_number(saved.get("mean_difference"), f"{method}.mean_difference"),
            _finite_number(interval[0], f"{method}.interval_95[0]"),
            _finite_number(interval[1], f"{method}.interval_95[1]"),
            _finite_number(saved.get("probability_positive"), f"{method}.probability_positive"),
        )
        differences = evidence.differences[row]
        recomputed = (
            float(differences.mean()),
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
            float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0)),
        )
        if not np.allclose(reported, recomputed, rtol=1e-10, atol=1e-12):
            raise ValueError(f"paired method summary disagrees with saved draws for {method}")
        mean, lower, upper, positive = recomputed
        if lower > upper or not 0.0 <= positive <= 1.0:
            raise ValueError(f"paired method interval/probability is invalid for {method}")
        summaries.append((method, mean, lower, upper, positive))
    return tuple(summaries)


def _paired_method_difference_figure(
    metrics: Mapping[str, Any],
    evidence: _BootstrapEvidence,
    output_directory: Path,
) -> EvidenceFigureArtifact:
    summaries = _paired_method_summaries(metrics, evidence)
    positions = np.arange(len(summaries), dtype=np.float64)
    means = np.asarray([summary[1] for summary in summaries], dtype=np.float64)
    lower = np.asarray([summary[2] for summary in summaries], dtype=np.float64)
    upper = np.asarray([summary[3] for summary in summaries], dtype=np.float64)
    labels = [
        f"{method.replace('_', ' ')} - {evidence.comparator.replace('_', ' ')}"
        for method, *_ in summaries
    ]

    figure, axis = plt.subplots(figsize=(9.2, max(4.5, 0.62 * len(summaries) + 1.8)))
    axis.hlines(positions, lower, upper, color="#3b82a0", linewidth=4)
    axis.scatter(means, positions, color="#b03a2e", s=42, zorder=3)
    axis.scatter(lower, positions, color="#243b53", marker="|", s=90, zorder=4)
    axis.scatter(upper, positions, color="#243b53", marker="|", s=90, zorder=4)
    axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.1)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired source-group bootstrap difference in average precision")
    axis.set_title("All predeclared audit methods versus self-confidence")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "paired_method_differences.png"
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key="paired_method_differences",
        title="Paired method differences",
        alt_text=(
            "Mean and 95 percent paired source-group bootstrap intervals for every saved "
            "audit method versus the predeclared self-confidence comparator"
        ),
        path=path,
        sources=(
            "bootstrap_evidence.npz:comparison_methods/differences",
            "metrics.json:paired_method_differences.comparisons",
        ),
        provenance={
            "metric": evidence.metric_name,
            "comparator": evidence.comparator,
            "methods": list(evidence.methods),
            "requested_iterations": evidence.requested_iterations,
            "valid_iterations": evidence.valid_iterations,
            "interval_rule": "numpy quantiles at 0.025 and 0.975 over shared valid draws",
        },
    )


def _paired_bootstrap_distribution_figure(
    evidence: _BootstrapEvidence,
    output_directory: Path,
) -> EvidenceFigureArtifact:
    try:
        row = evidence.methods.index("fixed_hybrid")
    except ValueError as exc:
        raise ValueError(
            "bootstrap evidence lacks the predeclared fixed-hybrid comparison"
        ) from exc
    differences = evidence.differences[row]
    if not len(differences):
        raise ValueError("reported fixed-hybrid bootstrap distribution is empty")
    mean = float(differences.mean())
    lower, upper = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
    bins = min(40, max(8, int(np.sqrt(len(differences)))))
    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    axis.hist(differences, bins=bins, color="#3b82a0", alpha=0.82, edgecolor="white")
    axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.1, label="No difference")
    axis.axvline(mean, color="#b03a2e", linewidth=1.7, label=f"Mean={mean:.3f}")
    axis.axvspan(lower, upper, color="#d9e2ec", alpha=0.45, label="Central 95% interval")
    axis.set_xlabel("Fixed hybrid - self-confidence average precision")
    axis.set_ylabel("Shared valid bootstrap draws")
    axis.set_title("Paired source-group bootstrap distribution")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "paired_bootstrap_distribution.png"
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key="paired_bootstrap_distribution",
        title="Paired bootstrap distribution",
        alt_text=(
            "Distribution of saved paired source-group bootstrap differences in average "
            "precision for fixed hybrid minus self-confidence"
        ),
        path=path,
        sources=("bootstrap_evidence.npz:differences[fixed_hybrid]",),
        provenance={
            "metric": evidence.metric_name,
            "comparison": f"fixed_hybrid minus {evidence.comparator}",
            "requested_iterations": evidence.requested_iterations,
            "valid_iterations": evidence.valid_iterations,
            "histogram_bins": bins,
        },
    )


def _subgroup_figure(
    metrics: Mapping[str, Any],
    *,
    dimension: str,
    output_directory: Path,
) -> EvidenceFigureArtifact:
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping) or not ranking:
        raise ValueError("metrics lack ranking evidence for subgroup figures")
    method_names = sorted(str(value) for value in ranking)
    entries_by_method: dict[str, dict[str, Mapping[str, Any]]] = {}
    subgroup_names: set[str] = set()
    for method in method_names:
        method_payload = ranking[method]
        if not isinstance(method_payload, Mapping):
            raise ValueError(f"ranking payload for {method} is invalid")
        subgroups = method_payload.get("subgroups")
        if not isinstance(subgroups, Mapping) or not isinstance(subgroups.get(dimension), list):
            raise ValueError(f"ranking payload for {method} lacks {dimension} subgroups")
        method_entries: dict[str, Mapping[str, Any]] = {}
        for entry in subgroups[dimension]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("subgroup"), str):
                raise ValueError(f"{dimension} subgroup entry for {method} is invalid")
            method_entries[str(entry["subgroup"])] = entry
        entries_by_method[method] = method_entries
        subgroup_names.update(method_entries)
    ordered_subgroups = sorted(subgroup_names)
    if not ordered_subgroups:
        raise ValueError(f"no {dimension} subgroup entries are available")
    values = np.full((len(method_names), len(ordered_subgroups)), np.nan, dtype=np.float64)
    total_support: list[int] = []
    injected_support: list[int] = []
    for column, subgroup in enumerate(ordered_subgroups):
        support_pairs: set[tuple[int, int]] = set()
        for row, method in enumerate(method_names):
            entry = entries_by_method[method].get(subgroup)
            if entry is None:
                raise ValueError(f"{dimension} subgroup {subgroup} is absent for {method}")
            total = entry.get("total_examples")
            injected = entry.get("injected_corruptions")
            if not isinstance(total, int) or not isinstance(injected, int):
                raise ValueError(f"{dimension} subgroup support is invalid")
            support_pairs.add((total, injected))
            ap = entry.get("average_precision")
            if isinstance(ap, (int, float)) and not isinstance(ap, bool):
                values[row, column] = float(ap)
            elif not isinstance(ap, Mapping) or ap.get("status") not in {
                "insufficient_support",
                "not_applicable",
            }:
                raise ValueError(f"{dimension} subgroup AP lacks structured N/A evidence")
        if len(support_pairs) != 1:
            raise ValueError(f"{dimension} subgroup support varies across audit methods")
        total, injected = next(iter(support_pairs))
        total_support.append(total)
        injected_support.append(injected)

    figure = plt.figure(figsize=(max(9.2, 1.25 * len(ordered_subgroups)), 8.2))
    grid = figure.add_gridspec(2, 1, height_ratios=(2.15, 1.0), hspace=0.78)
    result_axis = figure.add_subplot(grid[0])
    masked = np.ma.masked_invalid(values)
    colour_map = plt.get_cmap("viridis").with_extremes(bad="#eef2f6")
    image = result_axis.imshow(masked, cmap=colour_map, vmin=0.0, vmax=1.0, aspect="auto")
    result_axis.set_xticks(
        range(len(ordered_subgroups)),
        labels=[value.replace("_", " ") for value in ordered_subgroups],
        rotation=25,
        ha="right",
    )
    result_axis.set_yticks(
        range(len(method_names)), labels=[value.replace("_", " ") for value in method_names]
    )
    result_axis.set_title(
        f"{dimension.replace('_', ' ').title()} average precision (N/A = insufficient support)"
    )
    for row in range(len(method_names)):
        for column in range(len(ordered_subgroups)):
            value = values[row, column]
            result_axis.text(
                column,
                row,
                "N/A" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#243b53" if np.isnan(value) else ("white" if value < 0.6 else "black"),
            )
    figure.colorbar(image, ax=result_axis, label="Average precision", fraction=0.025, pad=0.02)

    support_axis = figure.add_subplot(grid[1])
    positions = np.arange(len(ordered_subgroups))
    support_axis.bar(positions, total_support, color="#9fb3c8", label="All examples")
    support_axis.bar(positions, injected_support, color="#b03a2e", label="Injected corruptions")
    support_axis.set_xticks(
        positions,
        labels=[value.replace("_", " ") for value in ordered_subgroups],
        rotation=25,
        ha="right",
    )
    support_axis.set_ylabel("Saved support count")
    support_axis.set_title("Support used by the preregistered subgroup reporting gate")
    support_axis.set_ylim(0.0, max(total_support) * 1.35)
    support_axis.legend(fontsize=8)
    for position, total, injected in zip(positions, total_support, injected_support, strict=True):
        support_axis.text(
            float(position),
            total,
            f"N={total}\ninj={injected}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    filename = (
        "per_class_results_support.png"
        if dimension == "pre_corruption_class"
        else "per_tissue_results_support.png"
    )
    key = (
        "per_class_results_support"
        if dimension == "pre_corruption_class"
        else "per_tissue_results_support"
    )
    figure.suptitle(
        "Controlled synthetic subgroup results; missing AP is never replaced by zero",
        fontsize=10,
        y=0.995,
    )
    path = output_directory / filename
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key=key,
        title=f"{dimension.replace('_', ' ').title()} results and support",
        alt_text=(
            f"Average-precision matrix and sample/injected-corruption support for {dimension}; "
            "insufficient-support results are explicitly marked N/A"
        ),
        path=path,
        sources=(f"metrics.json:ranking.*.subgroups.{dimension}",),
        provenance={
            "dimension": dimension,
            "missing_display": "N/A (never numeric zero)",
            "subgroups": ordered_subgroups,
            "methods": method_names,
        },
    )


def _neighbour_grid(evidence: _Evidence, output_directory: Path) -> EvidenceFigureArtifact:
    method = evidence.primary_method
    query_index = int(
        _ranked_indices(evidence.scores[method], evidence.sample_ids, descending=True)[0]
    )
    query_id = str(evidence.sample_ids[query_index])
    query_dataset_index = _dataset_index(evidence, query_id)
    query_crop = extract_target_crop(
        evidence.images[query_dataset_index],
        evidence.masks[query_dataset_index],
        output_size=48,
        padding=6,
    )
    count = int(evidence.neighbour_count[query_index])
    displayed = min(count, 4)
    figure = plt.figure(figsize=(13.6, 6.4))
    grid = figure.add_gridspec(2, 4, height_ratios=(1.2, 1.0), hspace=0.35)
    full_axis = figure.add_subplot(grid[0, 0])
    full_axis.imshow(
        _contour(evidence.images[query_dataset_index], evidence.masks[query_dataset_index])
    )
    full_axis.set_title("Query full patch + exact contour", fontsize=8)
    full_axis.axis("off")
    crop_axis = figure.add_subplot(grid[0, 1])
    crop_axis.imshow(_contour(query_crop.image, query_crop.target_mask))
    crop_axis.set_title("Query target crop + contour", fontsize=8)
    crop_axis.axis("off")
    text_axis = figure.add_subplot(grid[0, 2:])
    text_axis.axis("off")
    text_axis.text(
        0.0,
        1.0,
        _example_text(
            evidence,
            query_index,
            f"highest saved {method.replace('_', ' ')} score",
        ),
        ha="left",
        va="top",
        fontsize=6.8,
        family="monospace",
        linespacing=1.2,
    )
    id_to_index = {str(value): position for position, value in enumerate(evidence.sample_ids)}
    displayed_ids: list[str] = []
    for column in range(4):
        axis = figure.add_subplot(grid[1, column])
        if column >= displayed:
            axis.axis("off")
            continue
        neighbour_id = str(evidence.neighbour_ids[query_index, column])
        displayed_ids.append(neighbour_id)
        neighbour_index = id_to_index[neighbour_id]
        dataset_index = _dataset_index(evidence, neighbour_id)
        crop = extract_target_crop(
            evidence.images[dataset_index], evidence.masks[dataset_index], output_size=48, padding=6
        )
        axis.imshow(_contour(crop.image, crop.target_mask))
        axis.set_title(
            f"NN {column + 1}: {neighbour_id}\n"
            f"obs={_label(evidence, int(evidence.observed[neighbour_index]))}; "
            f"d={evidence.neighbour_distances[query_index, column]:.3f}\n"
            f"group={evidence.neighbour_groups[query_index, column]}",
            fontsize=7,
        )
        axis.axis("off")
    figure.suptitle(
        "Fold-safe nearest-neighbour explanation (query group excluded from every neighbour)",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.91, hspace=0.35, wspace=0.18)
    path = output_directory / "fold_safe_neighbour_explanation_grid.png"
    save_figure(figure, path)
    return EvidenceFigureArtifact(
        key="fold_safe_neighbour_explanation_grid",
        title="Fold-safe nearest-neighbour explanation grid",
        alt_text=(
            "Highest-risk query with full patch, exact target crop/contour, saved OOF and "
            "controlled-corruption metadata, and target crops of fold-safe neighbours"
        ),
        path=path,
        sources=(
            "synthetic_dataset_evidence.npz:sample_ids/images/target_masks",
            "oof_predictions.npz:all saved query/label/probability/fold arrays",
            "ranking.csv:saved risk scores",
            "corruption_manifest.json:rows",
            "neighbour_evidence.npz:all saved neighbour arrays",
        ),
        provenance={
            "query_selection": f"highest {method}; sample_id ascending tie break",
            "query_sample_id": query_id,
            "displayed_neighbour_ids": displayed_ids,
            "fold_safe_rule": "query/self group absent from saved neighbour groups",
        },
    )


def build_evidence_figures(
    metrics: Mapping[str, Any],
    *,
    output_directory: str | Path,
    predictions_path: str | Path,
    rankings_path: str | Path,
    neighbour_evidence_path: str | Path,
    dataset_evidence_path: str | Path,
    corruption_manifest_path: str | Path,
    representation_example_path: str | Path | None = None,
    bootstrap_evidence_path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
) -> tuple[EvidenceFigureArtifact, ...]:
    """Build every evidence-rich figure applicable to one controlled synthetic run."""

    destination = Path(output_directory)
    evidence = _load_evidence(
        metrics=metrics,
        predictions_path=Path(predictions_path),
        rankings_path=Path(rankings_path),
        neighbour_evidence_path=Path(neighbour_evidence_path),
        dataset_evidence_path=Path(dataset_evidence_path),
        corruption_manifest_path=Path(corruption_manifest_path),
        class_names=class_names,
    )
    artifacts: list[EvidenceFigureArtifact] = []
    has_positives = bool(evidence.injected.any())
    if has_positives:
        artifacts.append(_precision_recall_figure(evidence, destination))
        artifacts.append(_bootstrap_figure(metrics, destination))
    bootstrap_evidence = (
        _load_bootstrap_evidence(Path(bootstrap_evidence_path))
        if bootstrap_evidence_path is not None
        else None
    )
    if bootstrap_evidence is not None:
        if has_positives and bootstrap_evidence.status != "reported":
            raise ValueError("positive-event run has non-reported bootstrap evidence")
        if not has_positives and bootstrap_evidence.status != "not_applicable":
            raise ValueError("0% corruption run has fabricated reported bootstrap evidence")
        if has_positives:
            artifacts.append(
                _paired_method_difference_figure(metrics, bootstrap_evidence, destination)
            )
            artifacts.append(_paired_bootstrap_distribution_figure(bootstrap_evidence, destination))
    artifacts.append(
        _subgroup_figure(metrics, dimension="pre_corruption_class", output_directory=destination)
    )
    artifacts.append(
        _subgroup_figure(metrics, dimension="tissue_type", output_directory=destination)
    )

    method = evidence.primary_method
    order = _ranked_indices(evidence.scores[method], evidence.sample_ids, descending=True)
    top_indices = [int(value) for value in order[: min(4, len(order))]]
    artifacts.append(
        _example_grid(
            evidence,
            top_indices,
            output_directory=destination,
            filename="top_suspicious_controlled_examples.png",
            key="top_suspicious_controlled_examples",
            title="Top potentially inconsistent controlled annotations",
            reason=f"highest saved {method.replace('_', ' ')} scores",
            selection_rule=f"top 4 {method}; sample_id ascending tie break",
        )
    )
    artifacts.append(_neighbour_grid(evidence, destination))

    representation_sample_id: str | None = None
    if representation_example_path is not None:
        with np.load(representation_example_path, allow_pickle=False) as payload:
            if "sample_id" not in payload.files:
                raise ValueError("target representation example lacks sample_id")
            representation_sample_id = str(np.asarray(payload["sample_id"]).item())
        matches = np.flatnonzero(evidence.sample_ids == representation_sample_id)
        if len(matches) != 1:
            raise ValueError("target representation sample is absent from OOF evidence")
        artifacts.append(
            _example_grid(
                evidence,
                [int(matches[0])],
                output_directory=destination,
                filename="target_example_audit_evidence.png",
                key="target_example_audit_evidence",
                title="Saved target example with complete audit evidence",
                reason="saved deterministic target-representation example",
                selection_rule="sample_id saved in target_representation_example.npz",
            )
        )

    noncorrupt = np.flatnonzero(~evidence.injected)
    false_high_order = _ranked_indices(
        evidence.scores[method][noncorrupt], evidence.sample_ids[noncorrupt], descending=True
    )
    false_high = [
        int(noncorrupt[value])
        for value in false_high_order[: min(2 if has_positives else 4, len(false_high_order))]
    ]
    if has_positives:
        corrupt = np.flatnonzero(evidence.injected)
        false_low_order = _ranked_indices(
            evidence.scores[method][corrupt], evidence.sample_ids[corrupt], descending=False
        )
        false_low = [
            int(corrupt[value]) for value in false_low_order[: min(2, len(false_low_order))]
        ]
        artifacts.append(
            _example_grid(
                evidence,
                [*false_high, *false_low],
                output_directory=destination,
                filename="false_high_and_low_risk_examples.png",
                key="false_high_and_low_risk_examples",
                title="Controlled false-high-risk and false-low-risk examples",
                reason=(
                    "first rows: highest-risk non-corruptions; final rows: lowest-risk "
                    "injected corruptions"
                ),
                selection_rule=(
                    f"top 2 non-corruptions and bottom 2 injected corruptions by {method}; "
                    "sample_id ascending tie break"
                ),
            )
        )
    else:
        artifacts.append(
            _example_grid(
                evidence,
                false_high,
                output_directory=destination,
                filename="false_high_risk_examples.png",
                key="false_high_risk_examples",
                title="Highest-risk false alerts at 0% injected corruption",
                reason=(
                    "highest saved risk scores; false alert is defined only relative to the "
                    "controlled injected-corruption target"
                ),
                selection_rule=(
                    f"top 4 examples by {method}; sample_id ascending tie break; no "
                    "false-low-risk selection because no positive event exists"
                ),
            )
        )
    return tuple(artifacts)


__all__ = ["EvidenceFigureArtifact", "build_evidence_figures"]
