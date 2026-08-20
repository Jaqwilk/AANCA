"""Independently verify the published NuCLS external-validation evidence.

This verifier intentionally imports only the Python standard library and NumPy. It
does not import the AANCA package or scikit-learn. It pins every published file,
checks the portable sample manifest against the numeric arrays, and independently
recalculates the ranking, downstream, random-baseline, and group-bootstrap results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ABSOLUTE_TOLERANCE = 1.0e-12
CLASS_COUNT = 3
EXPECTED_RELEASES = {
    "unbiased-v1": {
        "subset": "unbiased_control",
        "files": {
            "artifact_manifest.json": (
                609,
                "f87b10860040855d0982aaf196389e321b2e32bf3e6598bcf053b33e159ae789",
            ),
            "canonical_manifest.csv": (
                454_586,
                "145a1e3d8bafd6121ae099dfa1d3201b2338475a6848d001874a945328034513",
            ),
            "numeric_evidence.npz": (
                1_712_144,
                "b03e578b5b4939dbb2554e26fedbd28515fb5c52e3353027f853cad03d3b75b9",
            ),
            "results.json": (
                19_653,
                "a931100a57e8d4b2a34a0216f047de8aa9d21c7275bc1beafd3b361fd955e9f6",
            ),
            "source_inventory.json": (
                29_132,
                "c89fcb83d52ee449fa3ff45638ad477d8880714b1c5bfec9efaac2d0be992243",
            ),
        },
    },
    "evaluation-v1": {
        "subset": "evaluation",
        "files": {
            "artifact_manifest.json": (
                609,
                "c6c6df030ffa1c840790e19392679ff5602cb23974f30c0b26116631fc8d0d55",
            ),
            "canonical_manifest.csv": (
                489_171,
                "90b8b67509b4d3fc65b9bb94f790900137f69a3eadd9b82df5514c93c1c26858",
            ),
            "numeric_evidence.npz": (
                2_144_648,
                "940bcc23d6b8c82f2d4a13587c0a3e3c11208b6363d31277d642f303906392d0",
            ),
            "results.json": (
                19_558,
                "c07131dced4113d89617029f88d544da5e33a0b88f4fc67c3a5ac634909fd28b",
            ),
            "source_inventory.json": (
                29_069,
                "fe7a46f1681827877bd0ec0fc6e2374fd63899abc53759b6ccf3e2f7d6cab96c",
            ),
        },
    },
}
EXPECTED_ARRAYS = {
    "downstream_guided_minus_random_bootstrap",
    "downstream_guided_minus_uncorrected_bootstrap",
    "downstream_guided_probabilities",
    "downstream_random_macro_f1",
    "downstream_random_probabilities",
    "downstream_reference_ceiling_probabilities",
    "downstream_uncorrected_probabilities",
    "embeddings",
    "group_ids",
    "natural_disagreement",
    "observed_labels",
    "oof_probabilities",
    "random_ranking_ap",
    "random_ranking_precision",
    "ranking_ap_minus_prevalence_bootstrap",
    "ranking_precision_minus_prevalence_bootstrap",
    "reference_labels",
    "risk_scores",
    "sample_ids",
}


class VerificationError(RuntimeError):
    """Raised when evidence identity or a recalculation differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise VerificationError(f"{path.name} must contain a JSON object")
    return payload


def _same_number(actual: float, expected: object, field: str) -> None:
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise VerificationError(f"{field} is missing or not numeric")
    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
        raise VerificationError(f"{field} differs: recalculated {actual}, saved {expected}")


def _same_vector(actual: NDArray[np.float64], expected: object, field: str) -> None:
    if not isinstance(expected, list) or len(expected) != len(actual):
        raise VerificationError(f"{field} has a different length")
    for index, value in enumerate(actual):
        _same_number(float(value), expected[index], f"{field}[{index}]")


def _same_array(actual: NDArray[np.generic], saved: NDArray[np.generic], field: str) -> None:
    if not np.array_equal(actual, saved):
        difference = (
            float(np.max(np.abs(actual.astype(np.float64) - saved.astype(np.float64))))
            if actual.shape == saved.shape and actual.size
            else None
        )
        raise VerificationError(f"{field} differs from regenerated values; max_diff={difference}")


def _average_precision(events: NDArray[np.bool_], scores: NDArray[np.float64]) -> float | None:
    positives = int(events.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.cumsum(events[order])
    threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    previous_recall = 0.0
    area = 0.0
    for end in threshold_ends:
        recall = int(cumulative[end]) / positives
        precision = int(cumulative[end]) / (int(end) + 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def _auroc(events: NDArray[np.bool_], scores: NDArray[np.float64]) -> float | None:
    positives = int(events.sum())
    negatives = len(events) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    statistic = float(ranks[events].sum()) - positives * (positives + 1) / 2.0
    return statistic / (positives * negatives)


def _rank(
    scores: NDArray[np.float64], tie_ids: NDArray[np.str_] | None = None
) -> NDArray[np.int64]:
    ties: NDArray[np.generic] = (
        np.arange(len(scores), dtype=np.int64) if tie_ids is None else tie_ids
    )
    return np.lexsort((ties, -scores)).astype(np.int64)


def _confusion(
    reference: NDArray[np.int64], probabilities: NDArray[np.generic]
) -> NDArray[np.int64]:
    predicted = np.argmax(np.asarray(probabilities, dtype=np.float64), axis=1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (reference, predicted), 1)
    return confusion


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


def _classification_metrics(
    reference: NDArray[np.int64], probabilities: NDArray[np.generic]
) -> dict[str, Any]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (len(reference), CLASS_COUNT):
        raise VerificationError("classification probability shape differs")
    if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7):
        raise VerificationError("classification probabilities are invalid")
    predictions = np.argmax(matrix, axis=1)
    confusion = _confusion(reference, matrix)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[reference]
    confidence = matrix.max(axis=1)
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
        "multiclass_brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }


def _verify_metrics(actual: dict[str, Any], saved: object, field: str) -> None:
    if not isinstance(saved, dict):
        raise VerificationError(f"{field} is not an object")
    if actual["confusion_matrix"] != saved.get("confusion_matrix"):
        raise VerificationError(f"{field}.confusion_matrix differs")
    for name in (
        "accuracy",
        "balanced_accuracy",
        "expected_calibration_error",
        "macro_f1",
        "multiclass_brier_score",
    ):
        _same_number(float(actual[name]), saved.get(name), f"{field}.{name}")
    for name in ("per_class_precision", "per_class_recall"):
        _same_vector(np.asarray(actual[name], dtype=np.float64), saved.get(name), f"{field}.{name}")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            record: dict[str, Any] = dict(raw)
            for field in ("observed_label", "reference_label"):
                record[field] = int(record[field])
            for field in ("xmin", "ymin", "xmax", "ymax"):
                record[field] = float(record[field])
            if record["natural_disagreement"] not in {"True", "False"}:
                raise VerificationError("manifest natural_disagreement is not boolean")
            record["natural_disagreement"] = record["natural_disagreement"] == "True"
            for field in ("np_contour_file", "np_rgb_file", "p_truth_master_file"):
                source = Path(str(record[field]))
                if source.is_absolute() or ":" in str(record[field]):
                    raise VerificationError(f"manifest contains a local absolute path in {field}")
            records.append(record)
    if not records:
        raise VerificationError("canonical manifest is empty")
    return records


def _verify_file_identities(root: Path, expected: dict[str, tuple[int, str]]) -> None:
    if {path.name for path in root.iterdir() if path.is_file()} != set(expected):
        raise VerificationError(f"{root.name} file set differs")
    for name, (expected_size, expected_hash) in expected.items():
        path = root / name
        if path.stat().st_size != expected_size:
            raise VerificationError(f"{root.name}/{name} byte size differs")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise VerificationError(f"{root.name}/{name} SHA-256 differs")
    manifest = _load_json(root / "artifact_manifest.json")
    listed = manifest.get("files")
    expected_listed = set(expected) - {"artifact_manifest.json"}
    if not isinstance(listed, dict) or set(listed) != expected_listed:
        raise VerificationError(f"{root.name} artifact manifest file set differs")
    for name in expected_listed:
        size, digest = expected[name]
        if listed[name] != {"bytes": size, "sha256": digest}:
            raise VerificationError(f"{root.name} artifact manifest differs for {name}")


def _verify_interval(
    values: NDArray[np.float64], saved: object, field: str, estimate: float
) -> None:
    if not isinstance(saved, dict):
        raise VerificationError(f"{field} is not an object")
    _same_number(estimate, saved.get("estimate"), f"{field}.estimate")
    _same_number(float(values.mean()), saved.get("mean"), f"{field}.mean")
    _same_vector(np.quantile(values, (0.025, 0.975)), saved.get("interval_95"), field)
    if saved.get("valid_iterations") != len(values):
        raise VerificationError(f"{field}.valid_iterations differs")
    if saved.get("requested_iterations") != 2000:
        raise VerificationError(f"{field}.requested_iterations differs")


def _ranking_bootstrap(
    events: NDArray[np.bool_],
    risk: NDArray[np.float64],
    groups: NDArray[np.str_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    unique_groups = np.unique(groups)
    members = {str(group): np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(26_082_031)
    ap_differences: list[float] = []
    precision_differences: list[float] = []
    for _ in range(2000):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[str(group)] for group in selected_groups])
        sampled_events = events[indices]
        prevalence = float(sampled_events.mean())
        ap = _average_precision(sampled_events, risk[indices])
        if ap is not None:
            ap_differences.append(ap - prevalence)
        count = int(np.ceil(len(indices) * 0.05))
        selected = _rank(risk[indices])[:count]
        precision_differences.append(float(sampled_events[selected].mean()) - prevalence)
    return (
        np.asarray(ap_differences, dtype=np.float64),
        np.asarray(precision_differences, dtype=np.float64),
    )


def _group_confusions(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.generic],
    groups: NDArray[np.str_],
) -> NDArray[np.int64]:
    matrix = np.asarray(probabilities)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    unique_groups = np.unique(groups)
    output = np.zeros(
        (matrix.shape[0], len(unique_groups), CLASS_COUNT, CLASS_COUNT), dtype=np.int64
    )
    predictions = np.argmax(matrix, axis=2)
    for repeat in range(matrix.shape[0]):
        for group_index, group in enumerate(unique_groups):
            members = groups == group
            np.add.at(
                output[repeat, group_index],
                (reference[members], predictions[repeat, members]),
                1,
            )
    return output


def _macro_f1_many(confusions: NDArray[np.integer]) -> NDArray[np.float64]:
    matrix = np.asarray(confusions, dtype=np.float64)
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    predicted = matrix.sum(axis=-2)
    actual = matrix.sum(axis=-1)
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
    return f1.mean(axis=-1)


def _downstream_bootstrap(
    reference: NDArray[np.int64],
    guided: NDArray[np.float64],
    uncorrected: NDArray[np.float64],
    random_probabilities: NDArray[np.float32],
    groups: NDArray[np.str_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    guided_groups = _group_confusions(reference, guided, groups)[0]
    uncorrected_groups = _group_confusions(reference, uncorrected, groups)[0]
    random_groups = _group_confusions(reference, random_probabilities, groups)
    rng = np.random.default_rng(26_082_051)
    guided_random = np.empty(2000, dtype=np.float64)
    guided_uncorrected = np.empty(2000, dtype=np.float64)
    group_count = guided_groups.shape[0]
    for iteration in range(2000):
        sampled = rng.integers(0, group_count, size=group_count)
        guided_f1 = _macro_f1_many(guided_groups[sampled].sum(axis=0)[None, ...])[0]
        uncorrected_f1 = _macro_f1_many(uncorrected_groups[sampled].sum(axis=0)[None, ...])[0]
        random_f1 = _macro_f1_many(random_groups[:, sampled].sum(axis=1)).mean()
        guided_random[iteration] = guided_f1 - random_f1
        guided_uncorrected[iteration] = guided_f1 - uncorrected_f1
    return guided_random, guided_uncorrected


def _verify_subset(root: Path, release_name: str, expected: dict[str, Any]) -> dict[str, Any]:
    _verify_file_identities(root, expected["files"])
    result = _load_json(root / "results.json")
    if result.get("study_id") != "nucls_natural_label_external_validation_v1":
        raise VerificationError(f"{release_name} study ID differs")
    if result.get("subset") != expected["subset"] or result.get("status") != "completed":
        raise VerificationError(f"{release_name} subset/status differs")
    boundary = result.get("claim_boundary")
    if not isinstance(boundary, dict) or any(boundary.values()):
        raise VerificationError(f"{release_name} claim boundary was weakened")

    source_inventory = _load_json(root / "source_inventory.json")
    source_files = source_inventory.get("files")
    if not isinstance(source_files, list) or len(source_files) != 109:
        raise VerificationError(f"{release_name} source inventory count differs")
    for record in source_files:
        if not isinstance(record, dict) or set(record) != {
            "role",
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise VerificationError(f"{release_name} source inventory schema differs")
        source_path = str(record["relative_path"])
        if Path(source_path).is_absolute() or ":" in source_path:
            raise VerificationError(f"{release_name} source inventory leaks a local path")
    source_hash = hashlib.sha256(
        json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if source_hash != result.get("source_inventory_sha256") or source_hash != source_inventory.get(
        "files_canonical_sha256"
    ):
        raise VerificationError(f"{release_name} source inventory payload hash differs")

    records = _load_manifest(root / "canonical_manifest.csv")
    canonical_hash = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    if canonical_hash != result.get("manifest_sha256"):
        raise VerificationError(f"{release_name} canonical manifest payload hash differs")

    with np.load(root / "numeric_evidence.npz", allow_pickle=False) as evidence:
        if set(evidence.files) != EXPECTED_ARRAYS:
            raise VerificationError(f"{release_name} numeric evidence array set differs")
        sample_ids = np.asarray(evidence["sample_ids"], dtype=np.str_)
        groups = np.asarray(evidence["group_ids"], dtype=np.str_)
        observed = np.asarray(evidence["observed_labels"], dtype=np.int64)
        reference = np.asarray(evidence["reference_labels"], dtype=np.int64)
        events = np.asarray(evidence["natural_disagreement"], dtype=np.bool_)
        oof = np.asarray(evidence["oof_probabilities"], dtype=np.float64)
        risk = np.asarray(evidence["risk_scores"], dtype=np.float64)
        count = len(records)
        if result.get("sample_count") != count or len(sample_ids) != count:
            raise VerificationError(f"{release_name} sample count differs")
        if len(np.unique(sample_ids)) != count or len(np.unique(groups)) != 5:
            raise VerificationError(f"{release_name} sample/group identities differ")
        _same_array(
            np.asarray([record["sample_id"] for record in records], dtype=np.str_),
            sample_ids,
            f"{release_name}.sample_ids",
        )
        _same_array(
            np.asarray([record["group_id"] for record in records], dtype=np.str_),
            groups,
            f"{release_name}.group_ids",
        )
        _same_array(
            np.asarray([record["observed_label"] for record in records], dtype=np.int64),
            observed,
            f"{release_name}.observed_labels",
        )
        _same_array(
            np.asarray([record["reference_label"] for record in records], dtype=np.int64),
            reference,
            f"{release_name}.reference_labels",
        )
        _same_array(observed != reference, events, f"{release_name}.natural_disagreement")
        if oof.shape != (count, CLASS_COUNT) or not np.allclose(oof.sum(axis=1), 1.0):
            raise VerificationError(f"{release_name} OOF probabilities are invalid")
        _same_array(1.0 - oof[np.arange(count), observed], risk, f"{release_name}.risk")
        if _array_sha256(np.asarray(evidence["embeddings"])) != result.get("embeddings_sha256"):
            raise VerificationError(f"{release_name} embedding array hash differs")

        ranking = result.get("ranking")
        if not isinstance(ranking, dict):
            raise VerificationError(f"{release_name} ranking result is missing")
        prevalence = float(events.mean())
        ap = _average_precision(events, risk)
        auroc = _auroc(events, risk)
        if ap is None or auroc is None:
            raise VerificationError(f"{release_name} ranking endpoints are undefined")
        if ranking.get("natural_disagreement_count") != int(events.sum()):
            raise VerificationError(f"{release_name} disagreement count differs")
        _same_number(prevalence, ranking.get("natural_disagreement_prevalence"), "prevalence")
        _same_number(ap, ranking.get("average_precision"), "average_precision")
        _same_number(auroc, ranking.get("auroc"), "auroc")
        for budget in (0.05, 0.10, 0.20):
            budget_result = ranking.get("budgets", {}).get(str(budget))
            if not isinstance(budget_result, dict):
                raise VerificationError(f"{release_name} budget {budget} is missing")
            reviewed = int(np.ceil(count * budget))
            selected = _rank(risk, sample_ids)[:reviewed]
            found = int(events[selected].sum())
            precision = found / reviewed
            recall = found / int(events.sum())
            if (
                budget_result.get("reviewed_count") != reviewed
                or budget_result.get("disagreements_found") != found
            ):
                raise VerificationError(f"{release_name} budget {budget} counts differ")
            _same_number(precision, budget_result.get("precision"), f"budget.{budget}.precision")
            _same_number(recall, budget_result.get("recall"), f"budget.{budget}.recall")
            _same_number(
                precision / prevalence,
                budget_result.get("lift_over_prevalence"),
                f"budget.{budget}.lift",
            )

        ap_boot, precision_boot = _ranking_bootstrap(events, risk, groups)
        _same_array(
            ap_boot,
            np.asarray(evidence["ranking_ap_minus_prevalence_bootstrap"], dtype=np.float64),
            f"{release_name}.ranking_ap_bootstrap",
        )
        _same_array(
            precision_boot,
            np.asarray(evidence["ranking_precision_minus_prevalence_bootstrap"], dtype=np.float64),
            f"{release_name}.ranking_precision_bootstrap",
        )
        top_count = int(np.ceil(count * 0.05))
        top_precision = float(events[_rank(risk, sample_ids)[:top_count]].mean())
        _verify_interval(
            ap_boot,
            ranking.get("ap_minus_prevalence"),
            f"{release_name}.ap_minus_prevalence",
            ap - prevalence,
        )
        _verify_interval(
            precision_boot,
            ranking.get("precision_at_5_percent_minus_prevalence"),
            f"{release_name}.precision_minus_prevalence",
            top_precision - prevalence,
        )

        random_aps = np.empty(1000, dtype=np.float64)
        random_precision = np.empty(1000, dtype=np.float64)
        for repeat in range(1000):
            random_scores = np.random.default_rng(26_082_027 + repeat).random(count)
            random_ap = _average_precision(events, random_scores)
            if random_ap is None:
                raise VerificationError("random ranking AP is undefined")
            random_aps[repeat] = random_ap
            random_precision[repeat] = float(events[_rank(random_scores)[:top_count]].mean())
        _same_array(
            random_aps,
            np.asarray(evidence["random_ranking_ap"], dtype=np.float64),
            f"{release_name}.random_ranking_ap",
        )
        _same_array(
            random_precision,
            np.asarray(evidence["random_ranking_precision"], dtype=np.float64),
            f"{release_name}.random_ranking_precision",
        )
        random_saved = ranking.get("random_rankings", {})
        _same_number(float(random_aps.mean()), random_saved.get("ap_mean"), "random.ap_mean")
        _same_vector(
            np.quantile(random_aps, (0.025, 0.975)),
            random_saved.get("ap_interval_95"),
            "random.ap_interval",
        )
        _same_number(
            float(random_precision.mean()),
            random_saved.get("precision_mean"),
            "random.precision_mean",
        )
        _same_vector(
            np.quantile(random_precision, (0.025, 0.975)),
            random_saved.get("precision_interval_95"),
            "random.precision_interval",
        )
        ranking_supported = bool(ap_boot.min() > 0.0 and precision_boot.min() > 0.0)
        if ranking.get("success_conditions_met") is not ranking_supported:
            raise VerificationError(f"{release_name} ranking success decision differs")

        downstream = result.get("downstream")
        if not isinstance(downstream, dict):
            raise VerificationError(f"{release_name} downstream result is missing")
        uncorrected = np.asarray(evidence["downstream_uncorrected_probabilities"], dtype=np.float64)
        guided = np.asarray(evidence["downstream_guided_probabilities"], dtype=np.float64)
        ceiling = np.asarray(
            evidence["downstream_reference_ceiling_probabilities"], dtype=np.float64
        )
        random_probabilities = np.asarray(
            evidence["downstream_random_probabilities"], dtype=np.float32
        )
        _verify_metrics(
            _classification_metrics(reference, uncorrected),
            downstream.get("uncorrected_observed"),
            f"{release_name}.uncorrected",
        )
        _verify_metrics(
            _classification_metrics(reference, guided),
            downstream.get("audit_guided_review"),
            f"{release_name}.guided",
        )
        _verify_metrics(
            _classification_metrics(reference, ceiling),
            downstream.get("pathologist_reference_ceiling"),
            f"{release_name}.ceiling",
        )
        random_f1 = np.asarray(
            [
                _classification_metrics(reference, probabilities)["macro_f1"]
                for probabilities in random_probabilities
            ],
            dtype=np.float64,
        )
        _same_array(
            random_f1,
            np.asarray(evidence["downstream_random_macro_f1"], dtype=np.float64),
            f"{release_name}.random_macro_f1",
        )
        random_downstream = downstream.get("random_review", {})
        _same_number(
            float(random_f1.mean()),
            random_downstream.get("macro_f1_mean"),
            "downstream.random.mean",
        )
        _same_vector(
            np.quantile(random_f1, (0.025, 0.975)),
            random_downstream.get("macro_f1_interval_95_across_repetitions"),
            "downstream.random.interval",
        )

        guided_random, guided_uncorrected = _downstream_bootstrap(
            reference, guided, uncorrected, random_probabilities, groups
        )
        _same_array(
            guided_random,
            np.asarray(evidence["downstream_guided_minus_random_bootstrap"], dtype=np.float64),
            f"{release_name}.guided_random_bootstrap",
        )
        _same_array(
            guided_uncorrected,
            np.asarray(evidence["downstream_guided_minus_uncorrected_bootstrap"], dtype=np.float64),
            f"{release_name}.guided_uncorrected_bootstrap",
        )
        guided_f1 = _classification_metrics(reference, guided)["macro_f1"]
        uncorrected_f1 = _classification_metrics(reference, uncorrected)["macro_f1"]
        _verify_interval(
            guided_random,
            downstream.get("guided_minus_mean_random_macro_f1"),
            f"{release_name}.guided_minus_random",
            float(guided_f1) - float(random_f1.mean()),
        )
        _verify_interval(
            guided_uncorrected,
            downstream.get("guided_minus_uncorrected_macro_f1"),
            f"{release_name}.guided_minus_uncorrected",
            float(guided_f1) - float(uncorrected_f1),
        )
        downstream_supported = bool(
            np.quantile(guided_random, 0.025) > 0.0 and np.quantile(guided_uncorrected, 0.025) > 0.0
        )
        if downstream.get("success_conditions_met") is not downstream_supported:
            raise VerificationError(f"{release_name} downstream success decision differs")

    return {
        "subset": expected["subset"],
        "sample_count": count,
        "patient_group_count": len(np.unique(groups)),
        "natural_disagreement_count": int(events.sum()),
        "average_precision": ap,
        "precision_at_5_percent": float(result["ranking"]["budgets"]["0.05"]["precision"]),
        "ranking_success_conditions_met": ranking_supported,
        "guided_minus_uncorrected_macro_f1": float(
            downstream["guided_minus_uncorrected_macro_f1"]["estimate"]
        ),
        "downstream_success_conditions_met": downstream_supported,
    }


def verify_release(root: Path) -> dict[str, Any]:
    """Verify both frozen NuCLS subsets from immutable files through conclusions."""

    root = root.resolve()
    if not root.is_dir():
        raise VerificationError(f"evidence root does not exist: {root}")
    subsets = []
    for release_name, expected in EXPECTED_RELEASES.items():
        subset_root = root / release_name
        if not subset_root.is_dir():
            raise VerificationError(f"missing evidence directory: {subset_root}")
        subsets.append(_verify_subset(subset_root, release_name, expected))
    primary = subsets[0]
    conclusion = (
        "supported"
        if primary["ranking_success_conditions_met"]
        and primary["downstream_success_conditions_met"]
        else "not_supported"
    )
    return {
        "study_id": "nucls_natural_label_external_validation_v1",
        "file_identity_status": "passed",
        "portable_manifest_status": "passed",
        "independent_recalculation_status": "passed",
        "subsets": subsets,
        "primary_claim_conclusion": conclusion,
        "pathologist_error_proven": False,
        "clinical_utility_proven": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the frozen NuCLS external-validation evidence."
    )
    parser.add_argument(
        "evidence_root",
        type=Path,
        nargs="?",
        default=Path("artifacts/nucls_external_validation"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_release(args.evidence_root)
    except VerificationError as error:
        payload = {"status": "failed", "error": str(error)}
        print(json.dumps(payload, indent=2) if args.json else f"FAILED: {error}")
        return 1
    if args.json:
        print(json.dumps({"status": "passed", **result}, indent=2))
    else:
        print("PASSED: all NuCLS evidence file identities and portable manifests")
        print("PASSED: ranking, random baselines, downstream metrics, and bootstraps recalculated")
        print(f"PRIMARY CLAIM: {result['primary_claim_conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
