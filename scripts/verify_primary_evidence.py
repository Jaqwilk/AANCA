"""Independently verify the published AANCA primary-study evidence release.

This script intentionally uses only Python's standard library and NumPy.  It does
not import the AANCA analysis package, so it can detect a disagreement between the
saved evidence and the implementation that originally produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ACCEPTED_RUN_ID = "20260727T133947.089370Z_pannuke_primary_orphan_recovery"
EXPECTED_FILES = {
    "primary_statistics.json": (
        22_498_321,
        "c3685fe9863fd73b1298f0558212cb5267b07c3ce6e4e4f37018dec55c115ac0",
    ),
    "primary_subgroups.csv": (
        4_208_358,
        "36be649fef067de82cd11b77f508f0a6fe62f649d393a1c9975a4523c24d166e",
    ),
    "primary_bootstrap_evidence.npz": (
        372_330_793,
        "35f8017cfcc887a1e94498a72e6868481088ce0fac4d8d2369d32504780bafa2",
    ),
    "primary_statistics_manifest.json": (
        19_892,
        "2d3c8115d371d7bbe55df3a0af83f28a875ec463823c0b07e0086efe1623318f",
    ),
}
EXPECTED_RESTORATION_FILES = {
    "restoration.json": (
        461_031,
        "1dd6f9e105066d0ce0314839783d15d3d0a4f96ece7430216bc3c6733961a27e",
    ),
    "restoration_evidence.npz": (
        267_636_826,
        "192c76b46a024d1124562ec461207ff24c20a5aaaf35d4b4b1998b53c7a8b956",
    ),
    "restoration_manifest.json": (
        464,
        "f93321dbf11af543567f218d160ef2584f552f4d165b598a14c9f33fddc019ce",
    ),
}
ABSOLUTE_TOLERANCE = 5e-15


class VerificationError(RuntimeError):
    """Raised when published evidence does not match the accepted result."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON evidence {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise VerificationError(f"{path.name} must contain a JSON object")
    return payload


def _verify_file_identities(
    directory: Path,
    expected: dict[str, tuple[int, str]],
) -> None:
    for name, (size, digest) in expected.items():
        path = directory / name
        if not path.is_file():
            raise VerificationError(f"missing evidence file: {path}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise VerificationError(f"{name} size differs: expected {size}, found {actual_size}")
        actual_digest = _sha256(path)
        if actual_digest != digest:
            raise VerificationError(
                f"{name} SHA-256 differs: expected {digest}, found {actual_digest}"
            )


def _same_number(actual: float, expected: object, field: str) -> None:
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise VerificationError(f"{field} is missing or is not numeric")
    if not math.isclose(
        actual,
        float(expected),
        rel_tol=0.0,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        raise VerificationError(f"{field} differs: recalculated {actual}, saved {expected}")


def _same_interval(actual: NDArray[np.float64], expected: object, field: str) -> None:
    if not isinstance(expected, list) or len(expected) != 2:
        raise VerificationError(f"{field} is not a two-value interval")
    _same_number(float(actual[0]), expected[0], f"{field}[0]")
    _same_number(float(actual[1]), expected[1], f"{field}[1]")


def _holm_adjust(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise VerificationError("Holm inputs must be a finite non-empty vector")
    order = np.argsort(values, kind="stable")
    scaled = (len(values) - np.arange(len(values))) * values[order]
    monotone = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(monotone)
    adjusted[order] = monotone
    return adjusted


def _verify_statistics_manifest(root: Path, statistics: dict[str, Any]) -> None:
    manifest = _load_json(root / "primary_statistics_manifest.json")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise VerificationError("primary statistics manifest has no artifact list")
    by_name = {str(record.get("path")): record for record in records if isinstance(record, dict)}
    expected_names = set(EXPECTED_FILES) - {"primary_statistics_manifest.json"}
    if set(by_name) != expected_names:
        raise VerificationError("primary statistics manifest artifact set differs")
    for name in sorted(expected_names):
        size, digest = EXPECTED_FILES[name]
        if by_name[name].get("size_bytes") != size or by_name[name].get("sha256") != digest:
            raise VerificationError(f"primary statistics manifest record differs for {name}")
    payload_hash = manifest.get("statistics_payload_sha256")
    recalculated_hash = _canonical_sha256(statistics)
    if payload_hash != recalculated_hash:
        raise VerificationError(
            "primary statistics canonical payload hash differs from its manifest"
        )
    for field in (
        "execution_controls_binding_sha256",
        "source_filesystem_readback_root_sha256",
        "primary_input_bindings_sha256",
        "crop_cache_sha256",
    ):
        if manifest.get(field) != statistics.get(field):
            raise VerificationError(f"statistics and manifest disagree on {field}")


def _verify_primary_comparisons(
    root: Path,
    statistics: dict[str, Any],
) -> tuple[int, int]:
    comparisons = statistics.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 36:
        raise VerificationError("expected exactly 36 frozen primary comparisons")
    with np.load(root / "primary_bootstrap_evidence.npz", allow_pickle=False) as evidence:
        ids = evidence["comparison_ids"].astype(str).tolist()
        kinds = evidence["comparison_kinds"].astype(str).tolist()
        saved_ids = [item.get("comparison_id") for item in comparisons]
        saved_kinds = [item.get("kind") for item in comparisons]
        if ids != saved_ids or kinds != saved_kinds:
            raise VerificationError("comparison order differs between JSON and bootstrap evidence")
        draw_count = len(evidence["draw_offsets"]) - 1
        if draw_count != statistics.get("bootstrap", {}).get("saved_draw_count"):
            raise VerificationError("saved bootstrap draw count differs")

        reported = 0
        unavailable = 0
        p_values_by_family: dict[str, list[tuple[int, float]]] = {}
        for index, item in enumerate(comparisons):
            prefix = f"comparison_{index:03d}"
            valid = np.asarray(evidence[f"{prefix}_valid_draw_indices"], dtype=np.int64)
            metric_a = np.asarray(evidence[f"{prefix}_metric_a"], dtype=np.float64)
            metric_b = np.asarray(evidence[f"{prefix}_metric_b"], dtype=np.float64)
            differences = np.asarray(evidence[f"{prefix}_differences"], dtype=np.float64)
            if not (len(valid) == len(metric_a) == len(metric_b) == len(differences)):
                raise VerificationError(f"{ids[index]} bootstrap array lengths differ")
            if not np.array_equal(differences, metric_a - metric_b):
                raise VerificationError(f"{ids[index]} saved differences are not metric_a-metric_b")
            if len(valid) and (valid.min() < 0 or valid.max() >= draw_count):
                raise VerificationError(f"{ids[index]} has an out-of-range bootstrap draw index")
            if item.get("status") != "reported":
                unavailable += 1
                if len(valid) or item.get("valid_bootstrap_iterations") != 0:
                    raise VerificationError(
                        f"{ids[index]} unavailable result contains bootstrap data"
                    )
                continue

            reported += 1
            if not len(differences) or not np.isfinite(differences).all():
                raise VerificationError(f"{ids[index]} has no finite bootstrap evidence")
            if item.get("valid_bootstrap_iterations") != len(differences):
                raise VerificationError(f"{ids[index]} valid iteration count differs")
            probability = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
            p_value = float((1 + np.count_nonzero(differences <= 0.0)) / (len(differences) + 1))
            _same_number(
                float(differences.mean()),
                item.get("bootstrap_mean_difference"),
                f"{ids[index]}.bootstrap_mean_difference",
            )
            _same_interval(
                np.quantile(differences, (0.025, 0.975)),
                item.get("interval_95"),
                f"{ids[index]}.interval_95",
            )
            _same_number(
                probability,
                item.get("probability_positive"),
                f"{ids[index]}.probability_positive",
            )
            _same_number(
                p_value,
                item.get("p_value_unadjusted"),
                f"{ids[index]}.p_value_unadjusted",
            )
            family = item.get("holm_family")
            if not isinstance(family, str):
                raise VerificationError(f"{ids[index]} has no Holm family")
            p_values_by_family.setdefault(family, []).append((index, p_value))

        for entries in p_values_by_family.values():
            adjusted = _holm_adjust(np.asarray([value for _, value in entries], dtype=np.float64))
            for (index, _), value in zip(entries, adjusted, strict=True):
                _same_number(
                    float(value),
                    comparisons[index].get("p_value_holm"),
                    f"{ids[index]}.p_value_holm",
                )
    return reported, unavailable


def _verify_restoration(restoration_root: Path) -> float:
    restoration = _load_json(restoration_root / "restoration.json")
    manifest = _load_json(restoration_root / "restoration_manifest.json")
    records = manifest.get("artifacts")
    expected_names = {"restoration.json", "restoration_evidence.npz"}
    if not isinstance(records, list):
        raise VerificationError("restoration manifest has no artifact list")
    by_name = {str(record.get("path")): record for record in records if isinstance(record, dict)}
    if set(by_name) != expected_names:
        raise VerificationError("restoration manifest artifact set differs")
    for name in expected_names:
        if by_name[name].get("sha256") != EXPECTED_RESTORATION_FILES[name][1]:
            raise VerificationError(f"restoration manifest record differs for {name}")
    comparisons = restoration.get("downstream_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise VerificationError("expected one frozen H4 downstream comparison")
    comparison = comparisons[0]
    if comparison.get("comparison_id") != "audit_guided_minus_random_macro_f1":
        raise VerificationError("unexpected H4 comparison identity")
    with np.load(restoration_root / "restoration_evidence.npz", allow_pickle=False) as evidence:
        ids = evidence["downstream_comparison_ids"].astype(str).tolist()
        if ids != [comparison["comparison_id"]]:
            raise VerificationError("H4 comparison identity differs in restoration evidence")
        metric_a = np.asarray(evidence["downstream_comparison_000_metric_a"], dtype=np.float64)
        metric_b = np.asarray(evidence["downstream_comparison_000_metric_b"], dtype=np.float64)
        differences = np.asarray(
            evidence["downstream_comparison_000_differences"], dtype=np.float64
        )
        if not np.array_equal(differences, metric_a - metric_b):
            raise VerificationError("H4 differences are not metric_a-metric_b")
        if len(differences) != comparison.get("random_repetitions"):
            raise VerificationError("H4 repetition count differs")
        _same_number(
            float(metric_a.mean()),
            comparison.get("point_metric_a"),
            "H4.point_metric_a",
        )
        _same_number(
            float(metric_b.mean()),
            comparison.get("point_metric_b"),
            "H4.point_metric_b",
        )
        _same_number(
            float(differences.mean()),
            comparison.get("mean_difference"),
            "H4.mean_difference",
        )
        _same_number(
            float(metric_a.mean() - metric_b.mean()),
            comparison.get("point_difference"),
            "H4.point_difference",
        )
        _same_interval(
            np.quantile(differences, (0.025, 0.975)),
            comparison.get("interval_95"),
            "H4.interval_95",
        )
        probability = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
        _same_number(
            probability,
            comparison.get("probability_positive"),
            "H4.probability_positive",
        )
    return float(comparison["mean_difference"])


def verify_release(root: Path, restoration_root: Path) -> dict[str, Any]:
    """Verify immutable identities and independently recalculate H1-H7 statistics."""

    root = root.resolve()
    restoration_root = restoration_root.resolve()
    if not root.is_dir():
        raise VerificationError(f"primary evidence directory does not exist: {root}")
    if not restoration_root.is_dir():
        raise VerificationError(
            f"restoration evidence directory does not exist: {restoration_root}"
        )
    _verify_file_identities(root, EXPECTED_FILES)
    _verify_file_identities(restoration_root, EXPECTED_RESTORATION_FILES)
    statistics = _load_json(root / "primary_statistics.json")
    if statistics.get("analysis_scope") != "real_pannuke_primary_statistics":
        raise VerificationError("primary analysis scope differs")
    if statistics.get("schema_version") != 1:
        raise VerificationError("primary statistics schema version differs")
    _verify_statistics_manifest(root, statistics)
    reported, unavailable = _verify_primary_comparisons(root, statistics)
    h4_difference = _verify_restoration(restoration_root)
    return {
        "accepted_run_id": ACCEPTED_RUN_ID,
        "file_identity_status": "passed",
        "primary_comparison_recalculation_status": "passed",
        "reported_primary_comparisons": reported,
        "explicitly_unavailable_primary_comparisons": unavailable,
        "h4_recalculation_status": "passed",
        "h4_mean_macro_f1_difference": h4_difference,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify checksums and independently recalculate the saved AANCA H1-H7 "
            "comparison statistics."
        )
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="directory containing the four primary statistics artifacts",
    )
    parser.add_argument(
        "--restoration-directory",
        type=Path,
        help=(
            "directory containing restoration.json, restoration_evidence.npz and "
            "restoration_manifest.json; defaults to evidence_directory"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print a JSON result")
    return parser


def main() -> int:
    args = _parser().parse_args()
    restoration_root = args.restoration_directory or args.evidence_directory
    try:
        result = verify_release(args.evidence_directory, restoration_root)
    except VerificationError as error:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        else:
            print(f"FAILED: {error}")
        return 1
    if args.json:
        print(json.dumps({"status": "passed", **result}, indent=2))
    else:
        print(f"PASSED: immutable evidence for {result['accepted_run_id']}")
        print(
            "PASSED: "
            f"{result['reported_primary_comparisons']} reported comparisons recalculated; "
            f"{result['explicitly_unavailable_primary_comparisons']} unavailable H6 entries "
            "preserved"
        )
        print(
            "PASSED: H4 mean macro-F1 difference recalculated as "
            f"{result['h4_mean_macro_f1_difference']:.12f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
