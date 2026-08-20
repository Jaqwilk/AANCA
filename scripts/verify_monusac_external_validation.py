"""Independently verify the frozen MoNuSAC controlled-external evidence.

The verifier intentionally imports only the Python standard library and NumPy. It
does not import AANCA analysis code or scikit-learn. It pins the published files and
recalculates the ranking, exact-matched controls, downstream metrics, and all
whole-patient bootstrap gates from the saved numeric evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ABSOLUTE_TOLERANCE = 1.0e-12
CLASS_COUNT = 4
PRIMARY = "nearest_neighbour_disagreement_balanced"
EXPECTED_ARTIFACT_MANIFEST = (
    657,
    "e4b1c0c327bba39f98677fc5e6f742f4158c77d0b0ba660ee29f5378b7510e7b",
)
EXPECTED_FILES: dict[str, tuple[int, str]] = {
    "numeric_evidence.npz": (
        9_889_779,
        "bda87a00b79db4962c71177a2dd3dea0c4c65b8b2d7299c577fd2ce4fdc1e8ec",
    ),
    "report.md": (
        2_711,
        "e6911fd73f2103a3ffbb650da180816f344a527691326ec62d641ef55663be42",
    ),
    "results.json": (
        33_249,
        "b2724e3e0baedcd0f1eb0fc7dfae127bf3789b03ac626d2701477ff4bae8e7d4",
    ),
    "source_inventory.json": (
        114_728,
        "2b84809ea064552c8d011e17c32b86c6a47e0870f7d3801ef9c15ba1eeb87b0d",
    ),
}
EXPECTED_ARRAYS = {
    "is_injected_corruption",
    "matched_random_indices",
    "matched_random_probabilities",
    "oof_fold_ids",
    "oof_probabilities",
    "risk_fixed_hybrid",
    "risk_nearest_neighbour_disagreement",
    "risk_self_confidence",
    "selected_fixed_hybrid_balanced",
    "selected_nearest_neighbour_disagreement_balanced",
    "selected_self_confidence_balanced",
    "selected_self_confidence_global",
    "test_group_ids",
    "test_probabilities_corrupted_uncorrected",
    "test_probabilities_fixed_hybrid_balanced_review",
    "test_probabilities_nearest_neighbour_disagreement_balanced_review",
    "test_probabilities_self_confidence_balanced_review",
    "test_probabilities_self_confidence_global_review",
    "test_probabilities_uncorrupted_reference_ceiling",
    "test_reference_labels",
    "test_sample_ids",
    "train_group_ids",
    "train_observed_labels",
    "train_organs",
    "train_reference_labels",
    "train_sample_ids",
}


class VerificationError(RuntimeError):
    """Raised when evidence identity or an independent recalculation differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{path.name} must contain a JSON object")
    return value


def _same_number(actual: float, expected: object, field: str) -> None:
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise VerificationError(f"{field} is absent or non-numeric")
    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
        raise VerificationError(f"{field} differs: recalculated {actual}, saved {expected}")


def _same_vector(actual: NDArray[np.float64], expected: object, field: str) -> None:
    if not isinstance(expected, list) or len(expected) != len(actual):
        raise VerificationError(f"{field} has a different length")
    for index, value in enumerate(actual):
        _same_number(float(value), expected[index], f"{field}[{index}]")


def _average_precision(events: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    positives = int(events.sum())
    if positives <= 0:
        raise VerificationError("ranking has no positive controlled events")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.cumsum(events[order])
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    previous_recall = 0.0
    area = 0.0
    for end in ends:
        recall = int(cumulative[end]) / positives
        precision = int(cumulative[end]) / (int(end) + 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def _confusion(
    reference: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> NDArray[np.int64]:
    if probabilities.shape != (len(reference), CLASS_COUNT):
        raise VerificationError("classification probability shape differs")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1.0e-7
    ):
        raise VerificationError("classification probabilities are invalid")
    predicted = np.argmax(probabilities, axis=1)
    output = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(output, (reference, predicted), 1)
    return output


def _macro_f1(confusion: NDArray[np.integer]) -> float:
    matrix = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def _recall(confusion: NDArray[np.integer]) -> NDArray[np.float64]:
    matrix = np.asarray(confusion, dtype=np.float64)
    actual = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        actual,
        out=np.full(CLASS_COUNT, np.nan, dtype=np.float64),
        where=actual > 0,
    )


def _classification_metrics(
    reference: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> dict[str, Any]:
    confusion = _confusion(reference, probabilities)
    predictions = np.argmax(probabilities, axis=1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[reference]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predictions == reference
    edges = np.linspace(0.0, 1.0, 11)
    calibration_error = 0.0
    for index in range(10):
        if index == 9:
            members = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            members = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if members.any():
            calibration_error += float(members.mean()) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall[actual > 0].mean()),
        "confusion_matrix": confusion.tolist(),
        "expected_calibration_error": calibration_error,
        "macro_f1": _macro_f1(confusion),
        "multiclass_brier_score": brier,
        "per_class_precision": precision,
        "per_class_recall": recall,
    }


def _verify_metrics(actual: dict[str, Any], expected: object, field: str) -> None:
    if not isinstance(expected, dict):
        raise VerificationError(f"{field} is absent")
    for name in (
        "accuracy",
        "balanced_accuracy",
        "expected_calibration_error",
        "macro_f1",
        "multiclass_brier_score",
    ):
        _same_number(float(actual[name]), expected.get(name), f"{field}.{name}")
    for name in ("per_class_precision", "per_class_recall"):
        _same_vector(np.asarray(actual[name]), expected.get(name), f"{field}.{name}")
    if actual["confusion_matrix"] != expected.get("confusion_matrix"):
        raise VerificationError(f"{field}.confusion_matrix differs")


def _group_confusions(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    unique: tuple[str, ...],
) -> NDArray[np.int64]:
    predictions = np.argmax(probabilities, axis=1)
    lookup = {group: index for index, group in enumerate(unique)}
    output = np.zeros((len(unique), CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for truth, prediction, group in zip(reference, predictions, groups, strict=True):
        output[lookup[str(group)], int(truth), int(prediction)] += 1
    return output


def _interval(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.quantile(values[np.isfinite(values)], (0.025, 0.975)))


def _guard_bootstrap(
    reference: NDArray[np.int64],
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    groups: NDArray[np.str_],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], dict[int, NDArray[np.float64]]]:
    unique = tuple(sorted(set(groups.tolist())))
    baseline_confusions = _group_confusions(reference, baseline, groups, unique)
    candidate_confusions = _group_confusions(reference, candidate, groups, unique)
    point_recall = _recall(candidate_confusions.sum(axis=0)) - _recall(
        baseline_confusions.sum(axis=0)
    )
    point_macro = _macro_f1(candidate_confusions.sum(axis=0)) - _macro_f1(
        baseline_confusions.sum(axis=0)
    )
    macro_values = np.empty(iterations, dtype=np.float64)
    recall_values: dict[int, list[float]] = {label: [] for label in range(CLASS_COUNT)}
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        baseline_confusion = baseline_confusions[sampled].sum(axis=0)
        candidate_confusion = candidate_confusions[sampled].sum(axis=0)
        macro_values[iteration] = _macro_f1(candidate_confusion) - _macro_f1(baseline_confusion)
        difference = _recall(candidate_confusion) - _recall(baseline_confusion)
        for label, value in enumerate(difference):
            if np.isfinite(value):
                recall_values[label].append(float(value))
    return (
        point_macro,
        _interval(macro_values),
        point_recall,
        {
            label: _interval(np.asarray(values, dtype=np.float64))
            for label, values in recall_values.items()
        },
    )


def _candidate_minus_random(
    reference: NDArray[np.int64],
    candidate: NDArray[np.float64],
    random_probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, NDArray[np.float64]]:
    unique = tuple(sorted(set(groups.tolist())))
    candidate_confusions = _group_confusions(reference, candidate, groups, unique)
    random_confusions = np.stack(
        [
            _group_confusions(reference, probability, groups, unique)
            for probability in random_probabilities
        ]
    )
    candidate_value = _macro_f1(candidate_confusions.sum(axis=0))
    random_values = np.asarray(
        [_macro_f1(value.sum(axis=0)) for value in random_confusions], dtype=np.float64
    )
    differences = np.empty(iterations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        sampled_candidate = _macro_f1(candidate_confusions[sampled].sum(axis=0))
        sampled_random = np.asarray(
            [_macro_f1(value[sampled].sum(axis=0)) for value in random_confusions]
        )
        differences[iteration] = sampled_candidate - float(sampled_random.mean())
    random_mean = float(random_values.mean())
    return candidate_value, random_mean, candidate_value - random_mean, _interval(differences)


def _selection_counts(
    indices: NDArray[np.int64],
    events: NDArray[np.bool_],
    groups: NDArray[np.str_],
    unique: tuple[str, ...],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    selected = np.zeros(len(unique), dtype=np.int64)
    found = np.zeros(len(unique), dtype=np.int64)
    lookup = {group: index for index, group in enumerate(unique)}
    for index in indices:
        column = lookup[str(groups[int(index)])]
        selected[column] += 1
        found[column] += int(events[int(index)])
    return selected, found


def _retrieval_bootstrap(
    events: NDArray[np.bool_],
    groups: NDArray[np.str_],
    top_indices: NDArray[np.int64],
    random_indices: NDArray[np.int64],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique = tuple(sorted(set(groups.tolist())))
    top_selected, top_found = _selection_counts(top_indices, events, groups, unique)
    random_counts = [
        _selection_counts(indices, events, groups, unique) for indices in random_indices
    ]
    top_precision = float(top_found.sum() / top_selected.sum())
    random_precision = np.asarray(
        [found.sum() / selected.sum() for selected, found in random_counts], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        top_denominator = int(top_selected[sampled].sum())
        if not top_denominator:
            continue
        top_value = float(top_found[sampled].sum() / top_denominator)
        random_values = []
        for selected, found in random_counts:
            denominator = int(selected[sampled].sum())
            if denominator:
                random_values.append(float(found[sampled].sum() / denominator))
        if random_values:
            differences.append(top_value - float(np.mean(random_values)))
    return {
        "top_found": int(events[top_indices].sum()),
        "top_reviewed": len(top_indices),
        "top_precision": top_precision,
        "mean_matched_random_found": float(
            np.mean([events[indices].sum() for indices in random_indices])
        ),
        "mean_matched_random_precision": float(random_precision.mean()),
        "top_minus_mean_matched_random_precision": float(top_precision - random_precision.mean()),
        "interval_95": _interval(np.asarray(differences)),
        "valid_iterations": len(differences),
    }


def _verify_file_identity(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if (manifest_path.stat().st_size, _sha256(manifest_path)) != EXPECTED_ARTIFACT_MANIFEST:
        raise VerificationError("pinned artifact-manifest identity differs")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise VerificationError("artifact manifest file set differs")
    for name, (expected_bytes, expected_hash) in EXPECTED_FILES.items():
        path = root / name
        entry = files.get(name)
        if not path.is_file() or not isinstance(entry, dict):
            raise VerificationError(f"published file is absent: {name}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if (actual_bytes, actual_hash) != (expected_bytes, expected_hash):
            raise VerificationError(f"pinned identity differs for {name}")
        if entry.get("bytes") != actual_bytes or entry.get("sha256") != actual_hash:
            raise VerificationError(f"artifact manifest differs for {name}")
    return manifest


def verify(root: Path, repository_root: Path) -> dict[str, Any]:
    """Verify all saved MoNuSAC evidence and return a compact readback."""

    artifact_manifest = _verify_file_identity(root)
    result = _load_json(root / "results.json")
    config_path = repository_root / "configs" / "monusac_current_aanca_external.yaml"
    if _sha256(config_path) != result.get("config_sha256"):
        raise VerificationError("frozen configuration identity differs")
    if result.get("study_id") != artifact_manifest.get("study_id"):
        raise VerificationError("study identity differs between result and manifest")

    try:
        with np.load(root / "numeric_evidence.npz", allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_ARRAYS:
                raise VerificationError("numeric evidence has missing or extra arrays")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, VerificationError):
            raise
        raise VerificationError(f"numeric evidence is invalid: {error}") from error

    train_ids = np.asarray(arrays["train_sample_ids"], dtype=np.str_)
    test_ids = np.asarray(arrays["test_sample_ids"], dtype=np.str_)
    train_groups = np.asarray(arrays["train_group_ids"], dtype=np.str_)
    test_groups = np.asarray(arrays["test_group_ids"], dtype=np.str_)
    organs = np.asarray(arrays["train_organs"], dtype=np.str_)
    reference = np.asarray(arrays["train_reference_labels"], dtype=np.int64)
    observed = np.asarray(arrays["train_observed_labels"], dtype=np.int64)
    events = np.asarray(arrays["is_injected_corruption"], dtype=bool)
    test_reference = np.asarray(arrays["test_reference_labels"], dtype=np.int64)
    oof = np.asarray(arrays["oof_probabilities"], dtype=np.float64)
    folds = np.asarray(arrays["oof_fold_ids"], dtype=np.int64)
    random_indices = np.asarray(arrays["matched_random_indices"], dtype=np.int64)
    random_probabilities = np.asarray(arrays["matched_random_probabilities"], dtype=np.float64)
    dataset = result.get("dataset", {})
    if not isinstance(dataset, dict):
        raise VerificationError("dataset result is absent")
    if len(set(train_ids.tolist())) != len(train_ids) or len(set(test_ids.tolist())) != len(
        test_ids
    ):
        raise VerificationError("sample identifiers are not unique")
    if set(train_groups).intersection(test_groups):
        raise VerificationError("development and final patient groups overlap")
    if any(
        len(value) != len(reference)
        for value in (train_ids, train_groups, organs, observed, events, folds)
    ):
        raise VerificationError("development arrays do not align")
    if len(test_ids) != len(test_reference) or len(test_groups) != len(test_reference):
        raise VerificationError("final-test arrays do not align")
    if not np.array_equal(reference != observed, events):
        raise VerificationError("injected mask does not exactly identify label changes")
    if int(events.sum()) != result["controlled_corruption"].get("exact_count"):
        raise VerificationError("controlled-corruption count differs")
    if len(set(train_groups)) != dataset.get("train_patient_groups") or len(
        set(test_groups)
    ) != dataset.get("test_patient_groups"):
        raise VerificationError("patient-group count differs")
    if len(reference) != dataset.get("train_eligible_nuclei") or len(test_reference) != dataset.get(
        "test_eligible_nuclei"
    ):
        raise VerificationError("eligible-nucleus count differs")
    if set(dataset.get("excluded_overlap_patients", ())).intersection(train_groups):
        raise VerificationError("declared overlap patient remains in development")
    if oof.shape != (len(reference), CLASS_COUNT) or not np.isfinite(oof).all():
        raise VerificationError("OOF probabilities are invalid")
    if not np.allclose(oof.sum(axis=1), 1.0, atol=1.0e-7):
        raise VerificationError("OOF rows do not sum to one")
    if set(folds.tolist()) != set(range(5)):
        raise VerificationError("OOF fold identifiers differ")
    for group in set(train_groups):
        if len(set(folds[train_groups == group].tolist())) != 1:
            raise VerificationError(f"patient group spans OOF folds: {group}")
    for fold in result["oof_evidence"]["folds"]:
        fold_id = int(fold["fold_id"])
        held_out = set(train_groups[folds == fold_id].tolist())
        if held_out != set(fold["held_out_groups"]):
            raise VerificationError(f"saved held-out groups differ in fold {fold_id}")
        if held_out.intersection(fold["training_groups"]):
            raise VerificationError(f"OOF training leakage in fold {fold_id}")

    ranking = result.get("ranking")
    if not isinstance(ranking, dict):
        raise VerificationError("ranking results are absent")
    score_names = {
        "self_confidence_global": "risk_self_confidence",
        "self_confidence_balanced": "risk_self_confidence",
        PRIMARY: "risk_nearest_neighbour_disagreement",
        "fixed_hybrid_balanced": "risk_fixed_hybrid",
    }
    selected_by_name: dict[str, NDArray[np.int64]] = {}
    for name, score_name in score_names.items():
        selected = np.asarray(arrays[f"selected_{name}"], dtype=np.int64)
        selected_by_name[name] = selected
        if (
            len(selected) != len(set(selected.tolist()))
            or (selected < 0).any()
            or (selected >= len(reference)).any()
        ):
            raise VerificationError(f"selection is invalid: {name}")
        saved = ranking[name]
        _same_number(
            _average_precision(events, np.asarray(arrays[score_name], dtype=np.float64)),
            saved.get("average_precision"),
            f"ranking.{name}.average_precision",
        )
        if len(selected) != saved.get("reviewed_count") or int(events[selected].sum()) != saved.get(
            "injected_found"
        ):
            raise VerificationError(f"ranking counts differ: {name}")
        _same_number(
            float(events[selected].mean()), saved.get("precision"), f"ranking.{name}.precision"
        )

    primary_selected = selected_by_name[PRIMARY]
    if random_indices.ndim != 2 or random_indices.shape != (20, len(primary_selected)):
        raise VerificationError("matched-random index matrix differs")
    proposed = np.argmax(oof, axis=1)
    primary_strata = Counter(
        (int(observed[index]), str(organs[index]), f"{observed[index]}->{proposed[index]}")
        for index in primary_selected
    )
    for repeat, indices in enumerate(random_indices):
        if (
            len(set(indices.tolist())) != len(indices)
            or (indices < 0).any()
            or (indices >= len(reference)).any()
        ):
            raise VerificationError(f"matched-random selection {repeat} is invalid")
        strata = Counter(
            (int(observed[index]), str(organs[index]), f"{observed[index]}->{proposed[index]}")
            for index in indices
        )
        if strata != primary_strata:
            raise VerificationError(f"matched-random strata differ in repetition {repeat}")

    retrieval = _retrieval_bootstrap(
        events,
        train_groups,
        primary_selected,
        random_indices,
        iterations=2000,
        seed=26082082,
    )
    saved_retrieval = result["primary_matched_random_retrieval"]
    for name in (
        "top_precision",
        "mean_matched_random_found",
        "mean_matched_random_precision",
        "top_minus_mean_matched_random_precision",
    ):
        _same_number(retrieval[name], saved_retrieval.get(name), f"retrieval.{name}")
    if retrieval["top_found"] != saved_retrieval.get("top_found") or retrieval[
        "top_reviewed"
    ] != saved_retrieval.get("top_reviewed"):
        raise VerificationError("retrieval counts differ")
    if retrieval["valid_iterations"] != saved_retrieval.get("valid_iterations"):
        raise VerificationError("retrieval bootstrap count differs")
    _same_vector(retrieval["interval_95"], saved_retrieval.get("interval_95"), "retrieval")

    downstream = result.get("downstream")
    if not isinstance(downstream, dict):
        raise VerificationError("downstream result is absent")
    probability_names = {
        name.removeprefix("test_probabilities_"): name
        for name in arrays
        if name.startswith("test_probabilities_")
    }
    for condition, array_name in probability_names.items():
        probabilities = np.asarray(arrays[array_name], dtype=np.float64)
        actual = _classification_metrics(test_reference, probabilities)
        _verify_metrics(actual, downstream["metrics"].get(condition), f"downstream.{condition}")

    random_restored = events[random_indices].sum(axis=1)
    _same_number(
        float(random_restored.mean()),
        downstream.get("matched_random_restored_count_mean"),
        "downstream.matched_random_restored_count_mean",
    )
    if [int(random_restored.min()), int(random_restored.max())] != downstream.get(
        "matched_random_restored_count_range"
    ):
        raise VerificationError("matched-random restored-count range differs")

    baseline = np.asarray(arrays["test_probabilities_corrupted_uncorrected"], dtype=np.float64)
    guard_pass: dict[str, bool] = {}
    for condition, saved_guard in downstream["adoption_guards"].items():
        candidate = np.asarray(arrays[f"test_probabilities_{condition}"], dtype=np.float64)
        point_macro, macro_interval, point_recall, recall_intervals = _guard_bootstrap(
            test_reference,
            baseline,
            candidate,
            test_groups,
            iterations=2000,
            seed=26082083,
        )
        saved_macro = saved_guard["macro_f1"]
        _same_number(
            point_macro,
            saved_macro.get("candidate_minus_uncorrected_macro_f1"),
            f"guards.{condition}.macro_f1_difference",
        )
        _same_vector(
            macro_interval, saved_macro.get("interval_95"), f"guards.{condition}.macro_f1_interval"
        )
        for label in range(CLASS_COUNT):
            _same_number(
                float(point_recall[label]),
                saved_guard["candidate_minus_uncorrected_recall"].get(str(label)),
                f"guards.{condition}.recall_{label}",
            )
            _same_vector(
                recall_intervals[label],
                saved_guard["per_class_recall_intervals_95"].get(str(label)),
                f"guards.{condition}.recall_interval_{label}",
            )
        class_pass = all(recall_intervals[label][0] >= -0.01 for label in range(CLASS_COUNT))
        macro_pass = macro_interval[0] > 0.0
        guard_pass[condition] = macro_pass and class_pass
        if class_pass != saved_guard.get("important_classes_pass"):
            raise VerificationError(f"important-class guard differs: {condition}")
        if guard_pass[condition] != saved_guard.get("apply_candidate"):
            raise VerificationError(f"adoption decision differs: {condition}")

    primary_condition = f"{PRIMARY}_review"
    primary_probabilities = np.asarray(
        arrays[f"test_probabilities_{primary_condition}"], dtype=np.float64
    )
    candidate_value, random_mean, difference, random_interval = _candidate_minus_random(
        test_reference,
        primary_probabilities,
        random_probabilities,
        test_groups,
        iterations=2000,
        seed=26082084,
    )
    saved_random = downstream["primary_minus_mean_matched_random"]
    for actual, name in (
        (candidate_value, "candidate_macro_f1"),
        (random_mean, "mean_matched_random_macro_f1"),
        (difference, "candidate_minus_mean_matched_random_macro_f1"),
    ):
        _same_number(actual, saved_random.get(name), f"downstream.random.{name}")
    _same_vector(random_interval, saved_random.get("interval_95"), "downstream.random.interval")

    success = {
        "primary_top_k_beats_exact_matched_random_control": retrieval["interval_95"][0] > 0.0,
        "primary_intervention_macro_f1_ci95_lower_gt_corrupted_uncorrected": (
            downstream["adoption_guards"][primary_condition]["macro_f1"]["interval_95"][0] > 0.0
        ),
        "primary_intervention_macro_f1_ci95_lower_gt_mean_matched_random": (
            random_interval[0] > 0.0
        ),
        "no_important_class_recall_ci95_lower_below_minus_0_01": downstream["adoption_guards"][
            primary_condition
        ]["important_classes_pass"],
    }
    if success != result.get("success_conditions"):
        raise VerificationError("frozen success conditions differ")
    all_success = all(success.values())
    if all_success != result.get("all_success_conditions_met"):
        raise VerificationError("overall frozen decision differs")
    return {
        "all_success_conditions_met": all_success,
        "decision": result.get("decision"),
        "files_verified": len(EXPECTED_FILES),
        "ranking_candidates_verified": len(score_names),
        "status": "verified",
        "study_id": result.get("study_id"),
        "whole_patient_bootstrap_iterations": 2000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/monusac_external_validation"),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = (repository_root / args.artifact_root).resolve()
    print(json.dumps(verify(artifact_root, repository_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
