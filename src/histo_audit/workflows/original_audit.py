"""Exploratory, non-diagnostic ranking of unmodified original annotations."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from histo_audit.auditing.scores import score_annotations
from histo_audit.cross_validation.oof import OOFResult, grouped_oof_logistic
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file

ManifestInput = str | Path | pd.DataFrame
REVIEW_RECOMMENDATION = "recommended for expert review as a potentially inconsistent annotation"


@dataclass(frozen=True, slots=True)
class OriginalLabelAuditResult:
    """Immutable output paths and counts from an exploratory original-label audit."""

    output_directory: Path
    ranking_all_path: Path
    top_overall_path: Path
    top_per_class_path: Path
    top_per_tissue_path: Path
    oof_predictions_path: Path
    oof_provenance_path: Path
    metadata_path: Path
    report_path: Path
    sample_count: int
    group_count: int
    top_overall_count: int
    top_per_class_count: int
    top_per_tissue_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible audit output evidence."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


def _read_manifest(source: ManifestInput) -> tuple[pd.DataFrame, Path | None, str | None]:
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True), None, None
    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"original-label manifest does not exist: {path}")
    before = sha256_file(path)
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.casefold() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.casefold() == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}")
    return frame, path, before


def _validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "group_id",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"original-label manifest lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("original-label manifest cannot be empty")
    result = frame.copy(deep=True)
    has_canonical_tissue = "tissue_type" in result.columns
    has_legacy_tissue = "tissue" in result.columns
    if not has_canonical_tissue and not has_legacy_tissue:
        raise ValueError(
            "original-label manifest lacks required tissue_type column "
            "(legacy tissue is accepted only for non-stage compatibility)"
        )
    if has_canonical_tissue and has_legacy_tissue:
        canonical = result["tissue_type"]
        legacy = result["tissue"]
        if (
            canonical.isna().any()
            or legacy.isna().any()
            or not bool(canonical.astype(str).eq(legacy.astype(str)).all())
        ):
            raise ValueError("manifest tissue_type and legacy tissue columns conflict")
    elif has_legacy_tissue:
        result["tissue_type"] = result["tissue"]
    for column in ("sample_id", "group_id", "tissue_type"):
        if result[column].isna().any() or (result[column].astype(str).str.strip() == "").any():
            raise ValueError(f"manifest column {column} contains empty values")
        result[column] = result[column].astype(str)
    if result["sample_id"].duplicated().any():
        raise ValueError("manifest sample_id values must be unique")
    corruption_values = result["is_injected_corruption"]
    allowed_boolean_values = corruption_values.map(
        lambda value: (
            isinstance(value, (bool, np.bool_))
            or (isinstance(value, (int, np.integer)) and int(value) in (0, 1))
        )
    )
    if not bool(allowed_boolean_values.all()):
        raise ValueError("is_injected_corruption must contain only boolean values")
    if bool(corruption_values.astype(bool).any()):
        raise ValueError("original-label audit refuses manifests containing injected corruption")
    pre = pd.to_numeric(result["pre_corruption_label"], errors="coerce")
    observed = pd.to_numeric(result["observed_label"], errors="coerce")
    if pre.isna().any() or observed.isna().any():
        raise ValueError("pre_corruption_label and observed_label must be numeric")
    if not bool((pre == observed).all()):
        raise ValueError("original-label audit requires observed_label == pre_corruption_label")
    result["pre_corruption_label"] = pre.astype(np.int64)
    result["observed_label"] = observed.astype(np.int64)
    return result


def _aligned_features(
    features: NDArray[np.generic],
    feature_sample_ids: Sequence[str],
    manifest_sample_ids: Sequence[str],
    *,
    allow_extra_features: bool = False,
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    identifiers = tuple(str(value) for value in feature_sample_ids)
    expected = tuple(str(value) for value in manifest_sample_ids)
    if matrix.ndim != 2 or matrix.shape[0] != len(identifiers) or not len(identifiers):
        raise ValueError("features must be a non-empty 2-D matrix aligned with feature_sample_ids")
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        raise ValueError("feature_sample_ids must be non-empty and unique")
    identifiers_set = set(identifiers)
    expected_set = set(expected)
    if (allow_extra_features and not expected_set.issubset(identifiers_set)) or (
        not allow_extra_features and identifiers_set != expected_set
    ):
        missing = sorted(expected_set.difference(identifiers_set))
        extra = sorted(identifiers_set.difference(expected_set))
        raise ValueError(f"feature/manifest sample ID mismatch; missing={missing}, extra={extra}")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    lookup = {identifier: index for index, identifier in enumerate(identifiers)}
    return matrix[np.asarray([lookup[identifier] for identifier in expected], dtype=np.int64)]


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _csv_text(frame: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue()


def _oof_provenance(result: OOFResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "representation": result.representation,
        "model_seed": result.model_seed,
        "split_seed": result.split_seed,
        "splitter_class_name": result.splitter_class_name,
        "splitter_fallback_status": result.splitter_fallback_status,
        "splitter_fallback_reason": result.splitter_fallback_reason,
        "class_order": list(result.class_order),
        "sample_count": len(result.sample_ids),
        "group_count": len(set(result.group_ids)),
        "coverage_exactly_once": bool(np.all(result.coverage_count == 1)),
        "final_reference_groups": list(result.final_reference_groups),
        "final_reference_groups_absent_from_audit": not bool(
            set(result.final_reference_groups).intersection(result.group_ids)
        ),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "training_groups": list(fold.training_groups),
                "held_out_groups": list(fold.held_out_groups),
                "held_out_sample_ids": list(fold.held_out_sample_ids),
            }
            for fold in result.folds
        ],
    }


def _report(
    *,
    sample_count: int,
    group_count: int,
    method: str,
    top_overall_count: int,
    top_per_class_count: int,
    top_per_tissue_count: int,
) -> str:
    return f"""# Exploratory original-label audit

This research-only output ranks potentially inconsistent annotations and recommends them
for expert review. It is not a diagnostic system, model disagreement does not establish an
annotation error, and no source annotation was modified.

## Scope and evidence

- Audited nuclei: {sample_count}
- Source groups: {group_count}
- Risk method: `{method}` from group-safe out-of-fold probabilities
- Top overall rows: {top_overall_count}
- Top per observed class rows: {top_per_class_count}
- Top per tissue rows: {top_per_tissue_count}
- Injected corruption: none; `observed_label` equals `pre_corruption_label` for every row
- Final-reference groups: excluded from this audit and unavailable to OOF fitting

No average precision, precision, recall, or verified-reference comparison is reported because
original annotations contain no injected-corruption reference events and no genuine expert
responses were provided. The rankings are prioritisation evidence only.
"""


def audit_original_labels(
    manifest: ManifestInput,
    features: NDArray[np.generic],
    feature_sample_ids: Sequence[str],
    output_directory: str | Path,
    *,
    final_reference_group_ids: Collection[str],
    audit_sample_ids: Sequence[str] | None = None,
    class_order: Sequence[int] = (0, 1, 2, 3, 4),
    n_splits: int = 5,
    split_seed: int = 223,
    model_seed: int = 227,
    representation: str = "frozen_features",
    method: str = "self_confidence",
    top_count_overall: int = 100,
    top_count_per_class: int = 20,
    top_count_per_tissue: int = 20,
    l2: float = 1.0e-2,
    max_iter: int = 400,
) -> OriginalLabelAuditResult:
    """Rank original labels with group-safe OOF evidence without changing source data."""

    if min(top_count_overall, top_count_per_class, top_count_per_tissue) <= 0:
        raise ValueError("all top-count limits must be positive")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"original-label audit output already exists: {destination}")
    frame, manifest_path, manifest_before = _read_manifest(manifest)
    frame = _validate_manifest(frame)
    selected_sample_ids: tuple[str, ...] | None = None
    if audit_sample_ids is not None:
        selected_sample_ids = tuple(str(value) for value in audit_sample_ids)
        if (
            not selected_sample_ids
            or len(set(selected_sample_ids)) != len(selected_sample_ids)
            or any(not value.strip() for value in selected_sample_ids)
        ):
            raise ValueError("audit_sample_ids must be non-empty and unique")
        manifest_ids = set(frame["sample_id"].astype(str))
        missing_selected = sorted(set(selected_sample_ids).difference(manifest_ids))
        if missing_selected:
            raise ValueError(
                "frozen audit sample order contains IDs absent from the manifest: "
                f"{missing_selected}"
            )
        frame = (
            frame.set_index("sample_id", drop=False)
            .loc[list(selected_sample_ids)]
            .reset_index(drop=True)
        )
    matrix = _aligned_features(
        features,
        feature_sample_ids,
        frame["sample_id"].tolist(),
        allow_extra_features=selected_sample_ids is not None,
    )
    classes = tuple(int(value) for value in class_order)
    observed = frame["observed_label"].to_numpy(dtype=np.int64)
    if any(int(value) not in classes for value in observed):
        raise ValueError("observed label is outside the declared fixed class_order")
    final_groups = {str(value) for value in final_reference_group_ids}
    if not final_groups:
        raise ValueError("non-empty final-reference group evidence is mandatory")
    overlap = set(frame["group_id"]).intersection(final_groups)
    if overlap:
        raise ValueError(
            f"final-reference groups are present in original audit rows: {sorted(overlap)}"
        )

    oof = grouped_oof_logistic(
        matrix,
        observed,
        frame["group_id"].tolist(),
        final_reference_group_ids=final_groups,
        sample_ids=frame["sample_id"].tolist(),
        n_splits=n_splits,
        class_order=classes,
        split_seed=split_seed,
        model_seed=model_seed,
        representation=representation,
        l2=l2,
        max_iter=max_iter,
    )
    risks = score_annotations(observed, oof.probabilities, method=method, class_order=classes)
    ranking = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "group_id": frame["group_id"],
            "tissue_type": frame["tissue_type"],
            "observed_label": observed,
            "risk_score": risks,
            "oof_predicted_class": oof.predicted_class,
            "oof_fold_id": oof.fold_id,
            "review_recommendation": REVIEW_RECOMMENDATION,
        }
    ).sort_values(["risk_score", "sample_id"], ascending=[False, True], kind="mergesort")
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, "overall_rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
    by_class = ranking.sort_values(
        ["observed_label", "risk_score", "sample_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    by_class.insert(
        1,
        "within_observed_class_rank",
        by_class.groupby("observed_label", sort=True).cumcount() + 1,
    )
    top_class = by_class[by_class["within_observed_class_rank"] <= top_count_per_class]
    by_tissue = ranking.sort_values(
        ["tissue_type", "risk_score", "sample_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    by_tissue.insert(
        1,
        "within_tissue_rank",
        by_tissue.groupby("tissue_type", sort=True).cumcount() + 1,
    )
    top_tissue = by_tissue[by_tissue["within_tissue_rank"] <= top_count_per_tissue]
    top_overall = ranking.head(top_count_overall)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    created_destination = False
    try:
        atomic_write_text(staging / "ranking_all.csv", _csv_text(ranking))
        atomic_write_text(staging / "top_overall.csv", _csv_text(top_overall))
        atomic_write_text(staging / "top_per_class.csv", _csv_text(top_class))
        atomic_write_text(staging / "top_per_tissue.csv", _csv_text(top_tissue))
        np.savez_compressed(
            staging / "oof_predictions.npz",
            sample_ids=np.asarray(oof.sample_ids, dtype=np.str_),
            observed_label=observed,
            probabilities=oof.probabilities,
            predicted_class=oof.predicted_class,
            fold_id=oof.fold_id,
            coverage_count=oof.coverage_count,
            class_order=np.asarray(oof.class_order, dtype=np.int64),
        )
        provenance = _oof_provenance(oof)
        atomic_write_json(staging / "oof_provenance.json", provenance)
        manifest_after = sha256_file(manifest_path) if manifest_path is not None else None
        if manifest_before is not None and manifest_after != manifest_before:
            raise RuntimeError("source manifest changed while the read-only audit was running")
        atomic_write_json(
            staging / "audit_metadata.json",
            {
                "schema_version": 1,
                "purpose": "exploratory ranking of potentially inconsistent annotations",
                "non_diagnostic": True,
                "automatic_source_annotation_modification": False,
                "source_manifest": {
                    "path": str(manifest_path) if manifest_path is not None else None,
                    "sha256_before": manifest_before,
                    "sha256_after": manifest_after,
                    "input_mode": "read_only_file"
                    if manifest_path is not None
                    else "in_memory_copy",
                },
                "feature_matrix_sha256": _array_sha256(matrix),
                "frozen_audit_sample_selection": (
                    {
                        "enabled": True,
                        "sample_count": len(selected_sample_ids),
                        "sample_order_sha256": hashlib.sha256(
                            json.dumps(
                                list(selected_sample_ids),
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                    if selected_sample_ids is not None
                    else {"enabled": False}
                ),
                "sample_count": len(frame),
                "group_count": int(frame["group_id"].nunique()),
                "observed_equals_pre_corruption": True,
                "injected_corruption_count": 0,
                "risk_method": method,
                "risk_direction": "larger means more suspicious, not confirmed incorrect",
                "review_recommendation_language": REVIEW_RECOMMENDATION,
                "group_safe_oof": provenance,
                "selection_counts": {
                    "top_overall": len(top_overall),
                    "top_per_class": len(top_class),
                    "top_per_tissue": len(top_tissue),
                },
                "outcome_metrics": "not_applicable_without_injected_or_genuine_expert_reference",
            },
        )
        atomic_write_text(
            staging / "report.md",
            _report(
                sample_count=len(frame),
                group_count=int(frame["group_id"].nunique()),
                method=method,
                top_overall_count=len(top_overall),
                top_per_class_count=len(top_class),
                top_per_tissue_count=len(top_tissue),
            ),
        )
        destination.mkdir(exist_ok=False)
        created_destination = True
        for artifact in staging.iterdir():
            artifact.rename(destination / artifact.name)
        staging.rmdir()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if created_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise

    return OriginalLabelAuditResult(
        output_directory=destination,
        ranking_all_path=destination / "ranking_all.csv",
        top_overall_path=destination / "top_overall.csv",
        top_per_class_path=destination / "top_per_class.csv",
        top_per_tissue_path=destination / "top_per_tissue.csv",
        oof_predictions_path=destination / "oof_predictions.npz",
        oof_provenance_path=destination / "oof_provenance.json",
        metadata_path=destination / "audit_metadata.json",
        report_path=destination / "report.md",
        sample_count=len(frame),
        group_count=int(frame["group_id"].nunique()),
        top_overall_count=len(top_overall),
        top_per_class_count=len(top_class),
        top_per_tissue_count=len(top_tissue),
    )


__all__ = ["REVIEW_RECOMMENDATION", "OriginalLabelAuditResult", "audit_original_labels"]
