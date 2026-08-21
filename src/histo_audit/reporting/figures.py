"""Small, fully sourced figure set for the synthetic software-validation report."""

from __future__ import annotations

import hashlib
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

from histo_audit.utils.run_tracking import atomic_write_bytes, atomic_write_json, sha256_file

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """One generated PNG and the machine-readable fields that source it."""

    key: str
    title: str
    alt_text: str
    path: Path
    sources: tuple[str, ...]
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SyntheticFigureSet:
    """Generated figure artifacts plus their source manifest."""

    figures: tuple[FigureArtifact, ...]
    manifest_path: Path | None


def save_figure(figure: Any, destination: Path) -> None:
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


def _labels(class_order: NDArray[np.int64], class_names: Sequence[str] | None) -> list[str]:
    if class_names is not None and len(class_names) == len(class_order):
        return [str(value).replace("_", " ") for value in class_names]
    return [f"Class {int(value)}" for value in class_order]


def _bar_annotations(axis: Any, values: Sequence[float], *, decimals: int = 0) -> None:
    for patch, value in zip(axis.patches, values, strict=True):
        label = f"{value:.{decimals}f}"
        axis.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _contour_overlay(image: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    interior = mask.copy()
    interior[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    contour = mask & ~interior
    overlay = image.copy()
    overlay[contour] = np.asarray([238, 32, 46], dtype=np.uint8)
    return overlay


def _representation_figure(
    representation_path: Path,
    output_directory: Path,
) -> FigureArtifact:
    with np.load(representation_path, allow_pickle=False) as payload:
        required = {
            "sample_id",
            "target_instance_id",
            "full_patch",
            "full_target_mask",
            "source_bbox",
            "crop_source_box",
            "target_crop",
            "crop_target_mask",
            "highlighted_full_patch",
            "highlighted_crop",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"target representation NPZ lacks arrays: {sorted(missing)}")
        sample_id = str(np.asarray(payload["sample_id"]).item())
        target_instance_id = int(np.asarray(payload["target_instance_id"]).item())
        full_patch = np.asarray(payload["full_patch"], dtype=np.uint8)
        full_mask = np.asarray(payload["full_target_mask"], dtype=bool)
        source_bbox = np.asarray(payload["source_bbox"], dtype=np.int64)
        crop = np.asarray(payload["target_crop"], dtype=np.uint8)
        crop_mask = np.asarray(payload["crop_target_mask"], dtype=bool)
        highlighted_full = np.asarray(payload["highlighted_full_patch"], dtype=np.uint8)
        highlighted_crop = np.asarray(payload["highlighted_crop"], dtype=np.uint8)
    if not sample_id or target_instance_id <= 0:
        raise ValueError("target representation has invalid target identity")
    if full_patch.ndim != 3 or full_patch.shape[-1] != 3 or full_mask.shape != full_patch.shape[:2]:
        raise ValueError("target representation full patch/mask shapes disagree")
    if crop.ndim != 3 or crop.shape[-1] != 3 or crop_mask.shape != crop.shape[:2]:
        raise ValueError("target representation crop/mask shapes disagree")
    if source_bbox.shape != (4,) or not full_mask.any() or not crop_mask.any():
        raise ValueError("target representation contains invalid target geometry")
    if highlighted_full.shape != full_patch.shape or highlighted_crop.shape != crop.shape:
        raise ValueError("target-highlight representations have invalid shapes")

    full_contour = _contour_overlay(full_patch, full_mask)
    crop_contour = _contour_overlay(crop, crop_mask)
    figure, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))
    panels = (
        (full_contour, "Full source patch\n+ target contour"),
        (crop, "Deterministic\ntarget crop"),
        (crop_contour, "Target crop\n+ contour"),
        (highlighted_crop, "Target-highlighted\nrepresentation"),
    )
    for axis, (panel, title) in zip(axes, panels, strict=True):
        axis.imshow(panel)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    x0, y0, x1, y1 = (int(value) for value in source_bbox)
    axes[0].add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#ffd166",
            linewidth=1.2,
        )
    )
    figure.suptitle(
        f"Saved target representation: {sample_id} (instance {target_instance_id})",
        fontsize=10,
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    path = output_directory / "target_representation_example.png"
    save_figure(figure, path)
    return FigureArtifact(
        key="target_representation_example",
        title="Target-nucleus representation example",
        alt_text=(
            "Saved synthetic full source patch, deterministic target crop, target contour, "
            f"and target-highlighted crop for {sample_id}"
        ),
        path=path,
        sources=(
            "target_representation_example.npz:sample_id",
            "target_representation_example.npz:target_instance_id",
            "target_representation_example.npz:full_patch",
            "target_representation_example.npz:full_target_mask",
            "target_representation_example.npz:source_bbox",
            "target_representation_example.npz:target_crop",
            "target_representation_example.npz:crop_target_mask",
            "target_representation_example.npz:highlighted_crop",
        ),
    )


def _prediction_figures(
    predictions_path: Path,
    output_directory: Path,
    *,
    class_names: Sequence[str] | None,
) -> list[FigureArtifact]:
    with np.load(predictions_path, allow_pickle=False) as payload:
        required = {
            "pre_corruption_label",
            "observed_label",
            "is_injected_corruption",
            "fold_id",
            "class_order",
            "tissue_type",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"prediction NPZ lacks figure source arrays: {sorted(missing)}")
        pre = np.asarray(payload["pre_corruption_label"], dtype=np.int64)
        observed = np.asarray(payload["observed_label"], dtype=np.int64)
        injected = np.asarray(payload["is_injected_corruption"], dtype=bool)
        folds = np.asarray(payload["fold_id"], dtype=np.int64)
        class_order = np.asarray(payload["class_order"], dtype=np.int64)
        tissue_types = np.asarray(payload["tissue_type"], dtype=np.str_)
    if pre.ndim != 1 or observed.shape != pre.shape or injected.shape != pre.shape:
        raise ValueError("prediction label/corruption arrays must be aligned vectors")
    if folds.shape != pre.shape or tissue_types.shape != pre.shape or class_order.ndim != 1:
        raise ValueError("prediction fold/class-order arrays have invalid shape")
    labels = _labels(class_order, class_names)
    colour = "#3b82a0"
    artifacts: list[FigureArtifact] = []

    class_counts = [int(np.sum(pre == class_id)) for class_id in class_order]
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    axis.bar(labels, class_counts, color=colour)
    axis.set_ylabel("Audit-pool nuclei")
    axis.set_title("Synthetic audit-pool pre-corruption label distribution")
    axis.tick_params(axis="x", rotation=25)
    _bar_annotations(axis, [float(value) for value in class_counts])
    figure.tight_layout()
    path = output_directory / "class_distribution.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="class_distribution",
            title="Class distribution",
            alt_text="Bar chart of synthetic audit-pool pre-corruption labels by class",
            path=path,
            sources=("oof_predictions.npz:pre_corruption_label", "oof_predictions.npz:class_order"),
        )
    )

    tissue_names, tissue_counts_array = np.unique(tissue_types, return_counts=True)
    tissue_counts = [int(value) for value in tissue_counts_array]
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    axis.bar(
        [str(value).replace("_", " ") for value in tissue_names], tissue_counts, color="#587f5f"
    )
    axis.set_ylabel("Audit-pool nuclei")
    axis.set_title("Synthetic audit-pool tissue distribution")
    axis.tick_params(axis="x", rotation=25)
    _bar_annotations(axis, [float(value) for value in tissue_counts])
    figure.tight_layout()
    path = output_directory / "tissue_distribution.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="tissue_distribution",
            title="Tissue distribution",
            alt_text="Bar chart of synthetic audit-pool nuclei by synthetic tissue group",
            path=path,
            sources=("oof_predictions.npz:tissue_type",),
        )
    )

    unique_folds, fold_counts = np.unique(folds, return_counts=True)
    fold_labels = [f"Fold {int(value)}" for value in unique_folds]
    figure, axis = plt.subplots(figsize=(6.6, 4.1))
    axis.bar(fold_labels, fold_counts.tolist(), color="#6b8e6b")
    axis.set_ylabel("OOF holdout samples")
    axis.set_title("Group-safe OOF fold distribution")
    _bar_annotations(axis, [float(value) for value in fold_counts])
    figure.tight_layout()
    path = output_directory / "oof_fold_distribution.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="oof_fold_distribution",
            title="OOF fold distribution",
            alt_text="Bar chart of audit samples assigned to each group-safe OOF fold",
            path=path,
            sources=("oof_predictions.npz:fold_id",),
        )
    )

    class_to_index = {int(value): index for index, value in enumerate(class_order)}
    transition = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    for original, replacement in zip(pre[injected], observed[injected], strict=True):
        if int(original) not in class_to_index or int(replacement) not in class_to_index:
            raise ValueError("injected label lies outside saved class_order")
        transition[class_to_index[int(original)], class_to_index[int(replacement)]] += 1
    figure, axis = plt.subplots(figsize=(7.0, 5.8))
    image = axis.imshow(transition, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Observed replacement label")
    axis.set_ylabel("Pre-corruption label")
    axis.set_title("Injected corruption transition counts")
    maximum = int(transition.max(initial=0))
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(transition[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if maximum and value > maximum / 2 else "#17202a",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Injected nuclei")
    figure.tight_layout()
    path = output_directory / "corruption_transition_matrix.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="corruption_transition_matrix",
            title="Injected corruption transition matrix",
            alt_text="Matrix of injected transitions from pre-corruption to observed labels",
            path=path,
            sources=(
                "oof_predictions.npz:pre_corruption_label",
                "oof_predictions.npz:observed_label",
                "oof_predictions.npz:is_injected_corruption",
                "oof_predictions.npz:class_order",
            ),
        )
    )
    return artifacts


def _ranking_series(
    metrics: Mapping[str, Any],
) -> dict[str, list[tuple[float, Mapping[str, Any]]]]:
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping):
        return {}
    series: dict[str, list[tuple[float, Mapping[str, Any]]]] = {}
    for method, method_payload in sorted(ranking.items()):
        if not isinstance(method_payload, Mapping):
            continue
        budgets = method_payload.get("review_budgets")
        if not isinstance(budgets, Mapping):
            continue
        points: list[tuple[float, Mapping[str, Any]]] = []
        for budget_payload in budgets.values():
            if not isinstance(budget_payload, Mapping):
                continue
            budget = budget_payload.get("budget_fraction")
            if isinstance(budget, (int, float)) and not isinstance(budget, bool):
                points.append((float(budget), budget_payload))
        if points:
            series[str(method)] = sorted(points, key=lambda point: point[0])
    return series


def _zero_corruption_figures(
    metrics: Mapping[str, Any],
    series: Mapping[str, list[tuple[float, Mapping[str, Any]]]],
    output_directory: Path,
) -> list[FigureArtifact]:
    """Plot valid score/false-alert evidence when detection metrics are undefined."""

    corruption = metrics.get("corruption")
    if not isinstance(corruption, Mapping) or corruption.get("exact_count") != 0:
        return []
    ranking = metrics.get("ranking")
    if not isinstance(ranking, Mapping) or not ranking:
        raise ValueError("0% corruption figures require ranking method evidence")
    artifacts: list[FigureArtifact] = []
    labels: list[str] = []
    distributions: list[tuple[float, ...]] = []
    fields = (
        "minimum",
        "p05",
        "p25",
        "p50",
        "mean",
        "p75",
        "p95",
        "maximum",
    )
    for method, method_payload in sorted(ranking.items()):
        if not isinstance(method_payload, Mapping):
            raise ValueError(f"ranking method {method} is not an object")
        distribution = method_payload.get("score_distribution")
        if not isinstance(distribution, Mapping):
            raise ValueError(f"0% corruption lacks score distribution for {method}")
        values = tuple(distribution.get(field) for field in fields)
        numeric_values: list[float] = []
        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"0% corruption has invalid score distribution for {method}")
            numeric_values.append(float(value))
        count = distribution.get("count")
        standard_deviation = distribution.get("standard_deviation")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(standard_deviation, (int, float))
            or isinstance(standard_deviation, bool)
            or not np.isfinite(float(standard_deviation))
            or float(standard_deviation) < 0
        ):
            raise ValueError(f"0% corruption has invalid score summary for {method}")
        minimum, p05, p25, p50, mean, p75, p95, maximum = numeric_values
        if not minimum <= p05 <= p25 <= p50 <= p75 <= p95 <= maximum:
            raise ValueError(f"0% corruption quantiles are not ordered for {method}")
        if not minimum <= mean <= maximum:
            raise ValueError(f"0% corruption mean is outside score range for {method}")
        labels.append(str(method).replace("_", " "))
        distributions.append((minimum, p05, p25, p50, mean, p75, p95, maximum))

    figure, axis = plt.subplots(figsize=(8.8, max(4.8, 0.55 * len(labels) + 1.8)))
    positions = np.arange(len(labels))
    for position, values in zip(positions, distributions, strict=True):
        minimum, p05, p25, median, mean, p75, p95, maximum = values
        axis.hlines(position, minimum, maximum, color="#bcccdc", linewidth=2)
        axis.hlines(position, p05, p95, color="#6b8e6b", linewidth=4)
        axis.hlines(position, p25, p75, color="#3b82a0", linewidth=9)
        axis.scatter(median, position, marker="o", color="white", edgecolor="#17202a", zorder=3)
        axis.scatter(mean, position, marker="x", color="#b03a2e", zorder=4)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Saved annotation-risk score")
    axis.set_title(
        "Risk-score distributions with 0% injected corruption\n"
        "Method-specific raw scales; compare positions within a method only"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.plot([], [], marker="o", color="white", markeredgecolor="#17202a", label="Median")
    axis.plot([], [], marker="x", color="#b03a2e", linestyle="none", label="Mean")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output_directory / "score_distribution_by_method.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="score_distribution_by_method",
            title="Risk-score distributions at 0% corruption",
            alt_text=(
                "Interval plot of sourced risk-score quantiles, medians, and means by method "
                "when no corruption was injected; raw score scales differ across methods and "
                "are not directly comparable"
            ),
            path=path,
            sources=("metrics.json:ranking.*.score_distribution",),
        )
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    plotted = False
    colours = plt.get_cmap("tab10")
    for index, (method, points) in enumerate(series.items()):
        budgets: list[float] = []
        false_alerts: list[float] = []
        for budget, payload in points:
            false_alert = payload.get("false_alert_count")
            reviewed = payload.get("reviewed_count")
            if (
                not isinstance(false_alert, int)
                or isinstance(false_alert, bool)
                or not isinstance(reviewed, int)
                or isinstance(reviewed, bool)
                or false_alert != reviewed
            ):
                raise ValueError(f"0% corruption has invalid false-alert count for {method}")
            budgets.append(100.0 * budget)
            false_alerts.append(float(false_alert))
        if budgets:
            plotted = True
            axis.plot(
                budgets,
                false_alerts,
                marker="o",
                linewidth=1.7,
                label=method.replace("_", " "),
                color=colours(index % 10),
            )
    if not plotted:
        plt.close(figure)
        raise ValueError("0% corruption lacks review-budget false-alert evidence")
    axis.set_xlabel("Review budget (%)")
    axis.set_ylabel("False-alert count")
    axis.set_title("False alerts versus expert-review budget at 0% corruption")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5, ncols=2)
    figure.tight_layout()
    path = output_directory / "false_alerts_vs_review_budget.png"
    save_figure(figure, path)
    artifacts.append(
        FigureArtifact(
            key="false_alerts_vs_review_budget",
            title="False alerts versus review budget at 0% corruption",
            alt_text=(
                "Line chart of sourced false-alert counts against review budget when no "
                "corruption was injected"
            ),
            path=path,
            sources=(
                "metrics.json:ranking.*.review_budgets.*.budget_fraction",
                "metrics.json:ranking.*.review_budgets.*.reviewed_count",
                "metrics.json:ranking.*.review_budgets.*.false_alert_count",
            ),
        )
    )
    return artifacts


def _metrics_figures(
    metrics: Mapping[str, Any],
    output_directory: Path,
) -> list[FigureArtifact]:
    series = _ranking_series(metrics)
    if not series:
        return []
    colours = plt.get_cmap("tab10")
    artifacts: list[FigureArtifact] = []
    artifacts.extend(_zero_corruption_figures(metrics, series, output_directory))
    method_names: list[str] = []
    average_precision: list[float] = []
    for method, points in series.items():
        ap_values = [point_payload.get("average_precision") for _, point_payload in points]
        numeric = [
            float(ap_value)
            for ap_value in ap_values
            if isinstance(ap_value, (int, float)) and not isinstance(ap_value, bool)
        ]
        if not numeric:
            continue
        if max(numeric) - min(numeric) > 1e-10:
            raise ValueError(f"average precision varies by review budget for method {method}")
        method_names.append(method.replace("_", " "))
        average_precision.append(numeric[0])
    if method_names:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        positions = np.arange(len(method_names))
        axis.barh(positions, average_precision, color="#3b82a0")
        axis.set_yticks(positions, labels=method_names)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Average precision")
        axis.set_title("Injected-corruption average precision by audit method")
        for position, value in zip(positions, average_precision, strict=True):
            axis.text(
                min(value + 0.015, 0.97),
                float(position),
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
        figure.tight_layout()
        path = output_directory / "average_precision_by_method.png"
        save_figure(figure, path)
        artifacts.append(
            FigureArtifact(
                key="average_precision_by_method",
                title="Average precision by risk method",
                alt_text="Horizontal bars showing injected-corruption average precision by method",
                path=path,
                sources=("metrics.json:ranking.*.review_budgets.*.average_precision",),
            )
        )

    for metric_name, label, filename, key in (
        (
            "recall",
            "Recall of injected corruptions",
            "recall_vs_review_budget.png",
            "recall_vs_budget",
        ),
        (
            "lift_over_random",
            "Lift over random review",
            "lift_vs_review_budget.png",
            "lift_vs_budget",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8.2, 5.0))
        plotted = False
        for index, (method, points) in enumerate(series.items()):
            x_values: list[float] = []
            y_values: list[float] = []
            for budget, point_payload in points:
                metric_value = point_payload.get(metric_name)
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    x_values.append(100.0 * budget)
                    y_values.append(float(metric_value))
            if x_values:
                plotted = True
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.7,
                    label=method.replace("_", " "),
                    color=colours(index % 10),
                )
        if not plotted:
            plt.close(figure)
            continue
        axis.set_xlabel("Review budget (%)")
        axis.set_ylabel(label)
        axis.set_title(f"{label} versus expert-review budget")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, ncols=2)
        figure.tight_layout()
        path = output_directory / filename
        save_figure(figure, path)
        artifacts.append(
            FigureArtifact(
                key=key,
                title=label,
                alt_text=f"Line chart of {label.lower()} against review budget by audit method",
                path=path,
                sources=(f"metrics.json:ranking.*.review_budgets.*.{metric_name}",),
            )
        )

    downstream = metrics.get("downstream_restoration")
    if isinstance(downstream, Mapping):
        names = (
            "uncorrupted_reference_baseline",
            "corrupted_observed_baseline",
            "random_review_restoration",
            "audit_guided_restoration",
        )
        labels = (
            "Uncorrupted\nreference",
            "Corrupted\nobserved",
            "Random review\nmean",
            "Audit guided",
        )
        downstream_values: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        valid = True
        for name in names:
            downstream_payload = downstream.get(name)
            if not isinstance(downstream_payload, Mapping):
                valid = False
                break
            if name == "random_review_restoration":
                random_value = downstream_payload.get("macro_f1_mean")
                interval = downstream_payload.get("macro_f1_interval_95")
                if (
                    not isinstance(random_value, (int, float))
                    or isinstance(random_value, bool)
                    or not isinstance(interval, Sequence)
                    or len(interval) != 2
                    or not all(isinstance(item, (int, float)) for item in interval)
                ):
                    valid = False
                    break
                numeric_value = float(random_value)
                low, high = float(interval[0]), float(interval[1])
                downstream_values.append(numeric_value)
                lower_errors.append(max(0.0, numeric_value - low))
                upper_errors.append(max(0.0, high - numeric_value))
            else:
                result_metrics = downstream_payload.get("metrics")
                outcome_value = (
                    result_metrics.get("macro_f1") if isinstance(result_metrics, Mapping) else None
                )
                if not isinstance(outcome_value, (int, float)) or isinstance(outcome_value, bool):
                    valid = False
                    break
                downstream_values.append(float(outcome_value))
                lower_errors.append(0.0)
                upper_errors.append(0.0)
        if valid:
            figure, axis = plt.subplots(figsize=(8.0, 4.8))
            positions = np.arange(len(names))
            axis.bar(
                positions,
                downstream_values,
                color=("#829ab1", "#d48a76", "#b8a35a", "#3b82a0"),
            )
            axis.errorbar(
                positions,
                downstream_values,
                yerr=np.asarray([lower_errors, upper_errors]),
                fmt="none",
                ecolor="#17202a",
                capsize=4,
                linewidth=1.2,
            )
            axis.set_xticks(positions, labels=labels)
            axis.set_ylabel("Macro F1 on untouched final reference set")
            axis.set_ylim(0.0, 1.0)
            axis.set_title("Downstream synthetic restoration comparison")
            _bar_annotations(axis, downstream_values, decimals=3)
            figure.tight_layout()
            path = output_directory / "downstream_macro_f1.png"
            save_figure(figure, path)
            artifacts.append(
                FigureArtifact(
                    key="downstream_macro_f1",
                    title="Downstream macro F1",
                    alt_text="Bar chart comparing downstream macro F1, with random-review interval",
                    path=path,
                    sources=(
                        "metrics.json:downstream_restoration.*.metrics.macro_f1",
                        "metrics.json:downstream_restoration.random_review_restoration.macro_f1_mean",
                        "metrics.json:downstream_restoration.random_review_restoration.macro_f1_interval_95",
                    ),
                )
            )
    return artifacts


def build_synthetic_figures(
    metrics: Mapping[str, Any],
    *,
    metrics_path: str | Path,
    output_directory: str | Path,
    predictions_path: str | Path | None = None,
    representation_example_path: str | Path | None = None,
    bootstrap_evidence_path: str | Path | None = None,
    rankings_path: str | Path | None = None,
    neighbour_evidence_path: str | Path | None = None,
    dataset_evidence_path: str | Path | None = None,
    corruption_manifest_path: str | Path | None = None,
    duplicate_audit_path: str | Path | None = None,
    duplicate_candidates_csv_path: str | Path | None = None,
    duplicate_candidates_figure_path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
) -> SyntheticFigureSet:
    """Generate every available synthetic figure and a checksum/source manifest."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figures: list[FigureArtifact] = []
    if predictions_path is not None:
        prediction_source = Path(predictions_path)
        if not prediction_source.is_file():
            raise FileNotFoundError(
                f"prediction NPZ for figures does not exist: {prediction_source}"
            )
        figures.extend(
            _prediction_figures(
                prediction_source,
                destination,
                class_names=class_names,
            )
        )
    if representation_example_path is not None:
        representation_source = Path(representation_example_path)
        if not representation_source.is_file():
            raise FileNotFoundError(
                f"target representation NPZ for figures does not exist: {representation_source}"
            )
        figures.append(_representation_figure(representation_source, destination))
    bootstrap_source = (
        Path(bootstrap_evidence_path).resolve() if bootstrap_evidence_path is not None else None
    )
    if bootstrap_source is not None and not bootstrap_source.is_file():
        raise FileNotFoundError(
            f"bootstrap evidence NPZ for figures does not exist: {bootstrap_source}"
        )
    figures.extend(_metrics_figures(metrics, destination))
    evidence_inputs = {
        "rankings_path": rankings_path,
        "neighbour_evidence_path": neighbour_evidence_path,
        "dataset_evidence_path": dataset_evidence_path,
        "corruption_manifest_path": corruption_manifest_path,
    }
    supplied_evidence_inputs = {
        name: Path(value).resolve() for name, value in evidence_inputs.items() if value is not None
    }
    if supplied_evidence_inputs and len(supplied_evidence_inputs) != len(evidence_inputs):
        missing = sorted(set(evidence_inputs).difference(supplied_evidence_inputs))
        raise ValueError(
            f"evidence-rich synthetic figures require all tracked inputs; missing {missing}"
        )
    if supplied_evidence_inputs:
        if predictions_path is None:
            raise ValueError("evidence-rich synthetic figures require OOF predictions")
        for name, source in supplied_evidence_inputs.items():
            if not source.is_file():
                raise FileNotFoundError(f"{name} for evidence figures does not exist: {source}")
        from histo_audit.reporting.evidence_figures import build_evidence_figures

        evidence_figures = build_evidence_figures(
            metrics,
            output_directory=destination,
            predictions_path=Path(predictions_path).resolve(),
            rankings_path=supplied_evidence_inputs["rankings_path"],
            neighbour_evidence_path=supplied_evidence_inputs["neighbour_evidence_path"],
            dataset_evidence_path=supplied_evidence_inputs["dataset_evidence_path"],
            corruption_manifest_path=supplied_evidence_inputs["corruption_manifest_path"],
            representation_example_path=representation_example_path,
            bootstrap_evidence_path=bootstrap_source,
            class_names=class_names,
        )
        figures.extend(
            FigureArtifact(
                key=figure.key,
                title=figure.title,
                alt_text=figure.alt_text,
                path=figure.path,
                sources=figure.sources,
                provenance=figure.provenance,
            )
            for figure in evidence_figures
        )
    duplicate_inputs = {
        "duplicate_audit_path": duplicate_audit_path,
        "duplicate_candidates_csv_path": duplicate_candidates_csv_path,
        "duplicate_candidates_figure_path": duplicate_candidates_figure_path,
    }
    supplied_duplicate_inputs = {
        name: Path(value).resolve() for name, value in duplicate_inputs.items() if value is not None
    }
    if supplied_duplicate_inputs and len(supplied_duplicate_inputs) != len(duplicate_inputs):
        missing = sorted(set(duplicate_inputs).difference(supplied_duplicate_inputs))
        raise ValueError(f"synthetic duplicate figure requires all audit inputs; missing {missing}")
    if supplied_duplicate_inputs:
        for name, source in supplied_duplicate_inputs.items():
            if not source.is_file():
                raise FileNotFoundError(f"{name} does not exist: {source}")
        from histo_audit.reporting.synthetic_duplicates import load_synthetic_duplicate_audit

        duplicate_payload = load_synthetic_duplicate_audit(
            supplied_duplicate_inputs["duplicate_audit_path"]
        )
        if duplicate_payload.get("real_data_duplicate_gate_eligible") is not False:
            raise ValueError("synthetic duplicate evidence must not satisfy a real-data gate")
        artifacts = duplicate_payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("synthetic duplicate audit lacks artifact hash bindings")
        for key, input_name in (
            ("candidate_csv", "duplicate_candidates_csv_path"),
            ("candidate_figure", "duplicate_candidates_figure_path"),
        ):
            record = artifacts.get(key)
            source = supplied_duplicate_inputs[input_name]
            if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(source):
                raise ValueError(f"synthetic duplicate {key} hash binding changed")
        source_figure = supplied_duplicate_inputs["duplicate_candidates_figure_path"]
        destination_figure = (destination / "duplicate_candidates.png").resolve()
        if source_figure != destination_figure:
            atomic_write_bytes(destination_figure, source_figure.read_bytes())
        counts = duplicate_payload.get("candidate_counts")
        thresholds = duplicate_payload.get("thresholds")
        figures.append(
            FigureArtifact(
                key="duplicate_candidates",
                title="Synthetic cross-boundary duplicate candidates",
                alt_text=(
                    "Read-only exact-array and deterministic average-hash candidates among "
                    "unique synthetic source patches"
                ),
                path=destination_figure,
                sources=(
                    "duplicate_audit.json:dataset binding, thresholds, counts, and candidates",
                    "duplicate_candidates.csv:complete candidate rows",
                    "synthetic_dataset_evidence.npz:unique source-patch pixels",
                ),
                provenance={
                    "candidate_counts": dict(counts) if isinstance(counts, Mapping) else None,
                    "thresholds": (dict(thresholds) if isinstance(thresholds, Mapping) else None),
                    "deduplication_unit": "patch_id",
                    "automatic_deletion": False,
                    "real_data_duplicate_gate_eligible": False,
                },
            )
        )
    if not figures:
        return SyntheticFigureSet(figures=(), manifest_path=None)
    manifest = destination / "figure_sources.json"
    class_name_payload = list(class_names) if class_names is not None else None
    class_name_json = json.dumps(
        class_name_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    atomic_write_json(
        manifest,
        {
            "metrics_path": str(Path(metrics_path).resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_path": (
                str(Path(predictions_path).resolve()) if predictions_path is not None else None
            ),
            "predictions_sha256": (
                sha256_file(predictions_path) if predictions_path is not None else None
            ),
            "representation_example_path": (
                str(Path(representation_example_path).resolve())
                if representation_example_path is not None
                else None
            ),
            "representation_example_sha256": (
                sha256_file(representation_example_path)
                if representation_example_path is not None
                else None
            ),
            "bootstrap_evidence_input": (
                {
                    "path": str(bootstrap_source),
                    "sha256": sha256_file(bootstrap_source),
                }
                if bootstrap_source is not None
                else None
            ),
            "evidence_inputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(supplied_evidence_inputs.items())
            },
            "duplicate_audit_inputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(supplied_duplicate_inputs.items())
            },
            "class_names": class_name_payload,
            "class_names_sha256": hashlib.sha256(class_name_json.encode("utf-8")).hexdigest(),
            "figures": [
                {
                    "key": figure.key,
                    "title": figure.title,
                    "path": str(figure.path.resolve()),
                    "sha256": sha256_file(figure.path),
                    "sources": list(figure.sources),
                    "provenance": dict(figure.provenance) if figure.provenance else None,
                }
                for figure in figures
            ],
        },
    )
    return SyntheticFigureSet(figures=tuple(figures), manifest_path=manifest)


__all__ = [
    "FigureArtifact",
    "SyntheticFigureSet",
    "build_synthetic_figures",
]
