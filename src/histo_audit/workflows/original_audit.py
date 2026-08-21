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

from histo_audit.auditing.strategies import group_safe_audit_scores
from histo_audit.auditing.two_queue import (
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
)
from histo_audit.cross_validation.oof import OOFResult, grouped_oof_logistic
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    sha256_file,
)

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
    balanced_quality_queue_path: Path | None = None
    balanced_queue_evidence_path: Path | None = None
    balanced_quality_count: int = 0
    balanced_quality_underfilled: bool = False

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
    balanced_quality_count: int,
    balanced_quality_underfilled: bool,
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
- Balanced quality-control rows: {balanced_quality_count}
- Balanced queue underfilled by registered quotas: {balanced_quality_underfilled}
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
    neighbour_k: int = 7,
    neighbour_metric: str = "cosine",
    top_count_overall: int = 100,
    top_count_per_class: int = 20,
    top_count_per_tissue: int = 20,
    balanced_top_count: int | None = None,
    balanced_max_per_group: int | None = None,
    balanced_max_per_class: int | None = None,
    balanced_max_per_tissue: int | None = None,
    balanced_max_per_transition: int | None = None,
    balanced_minimum_cosine_distance: float | None = None,
    l2: float = 1.0e-2,
    max_iter: int = 400,
) -> OriginalLabelAuditResult:
    """Rank original labels with group-safe OOF evidence without changing source data."""

    if min(top_count_overall, top_count_per_class, top_count_per_tissue) <= 0:
        raise ValueError("all top-count limits must be positive")
    if balanced_top_count is not None and balanced_top_count <= 0:
        raise ValueError("balanced_top_count must be positive when supplied")
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
    score_result = group_safe_audit_scores(
        matrix,
        observed,
        oof.probabilities,
        frame["group_id"].astype(str).tolist(),
        oof.fold_id,
        oof.training_groups_by_fold,
        sample_ids=frame["sample_id"].astype(str).tolist(),
        method=method,
        class_order=classes,
        neighbour_k=neighbour_k,
        neighbour_metric=neighbour_metric,
    )
    risks = score_result.risk_scores
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
    balanced_queue = None
    balanced_rows = pd.DataFrame()
    balanced_constraints = None
    if balanced_top_count is not None:
        balanced_constraints = QueueConstraints(
            requested_count=balanced_top_count,
            max_per_group=balanced_max_per_group,
            max_per_class=balanced_max_per_class,
            max_per_tissue=balanced_max_per_tissue,
            max_per_transition=balanced_max_per_transition,
            minimum_cosine_distance=balanced_minimum_cosine_distance,
        )
        queues = build_two_review_queues(
            risks,
            frame["group_id"].astype(str).tolist(),
            observed.tolist(),
            frame["sample_id"].astype(str).tolist(),
            quality_constraints=balanced_constraints,
            model_constraints=QueueConstraints(requested_count=balanced_top_count),
            annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
            proposed_labels=oof.predicted_class.tolist(),
            tissue_types=frame["tissue_type"].astype(str).tolist(),
            embeddings=matrix,
            minimum_annotation_score=float(np.min(risks)),
        )
        balanced_queue = queues.quality_control
        ranking_by_id = ranking.set_index("sample_id", drop=False)
        balanced_rows = ranking_by_id.loc[list(balanced_queue.selected_sample_ids)].copy()
        balanced_rows.insert(
            0,
            "balanced_queue_rank",
            np.arange(1, len(balanced_rows) + 1, dtype=np.int64),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    created_destination = False
    try:
        atomic_write_text(staging / "ranking_all.csv", _csv_text(ranking))
        atomic_write_text(staging / "top_overall.csv", _csv_text(top_overall))
        atomic_write_text(staging / "top_per_class.csv", _csv_text(top_class))
        atomic_write_text(staging / "top_per_tissue.csv", _csv_text(top_tissue))
        if balanced_queue is not None and balanced_constraints is not None:
            atomic_write_text(staging / "balanced_quality_queue.csv", _csv_text(balanced_rows))
            atomic_write_json(
                staging / "balanced_queue_evidence.json",
                {
                    "schema_version": 1,
                    "purpose": "balanced annotation-quality review queue",
                    "annotation_evidence_role": GROUP_SAFE_OOF_EVIDENCE,
                    "constraints": asdict(balanced_constraints),
                    "requested_count": balanced_queue.requested_count,
                    "eligible_count": balanced_queue.eligible_count,
                    "selected_count": balanced_queue.selected_count,
                    "selected_sample_ids": list(balanced_queue.selected_sample_ids),
                    "underfilled": balanced_queue.underfilled,
                    "rejection_counts": balanced_queue.rejection_counts,
                    "group_counts": balanced_queue.group_counts,
                    "class_counts": balanced_queue.class_counts,
                    "tissue_counts": balanced_queue.tissue_counts,
                    "transition_counts": balanced_queue.transition_counts,
                    "model_improvement_queue": {
                        "available": False,
                        "reason": (
                            "no independently measured cross-fitted downstream utility was supplied"
                        ),
                    },
                    "automatic_source_annotation_modification": False,
                },
            )
        score_arrays = {
            f"risk_component_{name}": values
            for name, values in score_result.component_scores.items()
        }
        atomic_write_npz(
            staging / "oof_predictions.npz",
            {
                "sample_ids": np.asarray(oof.sample_ids, dtype=np.str_),
                "observed_label": observed,
                "probabilities": oof.probabilities,
                "predicted_class": oof.predicted_class,
                "fold_id": oof.fold_id,
                "coverage_count": oof.coverage_count,
                "class_order": np.asarray(oof.class_order, dtype=np.int64),
                "risk_score": risks,
                **score_arrays,
            },
        )
        if score_result.neighbour_evidence is not None:
            neighbour = score_result.neighbour_evidence
            atomic_write_json(
                staging / "neighbour_evidence.json",
                {
                    "schema_version": 1,
                    "strategy": score_result.as_dict(),
                    "records": [
                        {
                            "sample_id": sample_id,
                            "neighbour_ids": list(neighbour_ids),
                            "neighbour_groups": list(neighbour_groups),
                            "neighbour_distances": list(neighbour_distances),
                            "suggested_class": int(suggested_class),
                            "risk_score": float(risk_score),
                        }
                        for sample_id, neighbour_ids, neighbour_groups, neighbour_distances, suggested_class, risk_score in zip(
                            oof.sample_ids,
                            neighbour.neighbour_ids,
                            neighbour.neighbour_groups,
                            neighbour.neighbour_distances,
                            neighbour.suggested_class,
                            neighbour.risk_scores,
                            strict=True,
                        )
                    ],
                },
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
                "risk_method": score_result.method,
                "risk_strategy": score_result.as_dict(),
                "risk_direction": "larger means more suspicious, not confirmed incorrect",
                "review_recommendation_language": REVIEW_RECOMMENDATION,
                "group_safe_oof": provenance,
                "selection_counts": {
                    "top_overall": len(top_overall),
                    "top_per_class": len(top_class),
                    "top_per_tissue": len(top_tissue),
                    "balanced_quality": len(balanced_rows),
                },
                "outcome_metrics": "not_applicable_without_injected_or_genuine_expert_reference",
            },
        )
        atomic_write_text(
            staging / "report.md",
            _report(
                sample_count=len(frame),
                group_count=int(frame["group_id"].nunique()),
                method=score_result.method,
                top_overall_count=len(top_overall),
                top_per_class_count=len(top_class),
                top_per_tissue_count=len(top_tissue),
                balanced_quality_count=len(balanced_rows),
                balanced_quality_underfilled=(
                    balanced_queue.underfilled if balanced_queue is not None else False
                ),
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
        balanced_quality_queue_path=(
            destination / "balanced_quality_queue.csv" if balanced_queue is not None else None
        ),
        balanced_queue_evidence_path=(
            destination / "balanced_queue_evidence.json" if balanced_queue is not None else None
        ),
        balanced_quality_count=len(balanced_rows),
        balanced_quality_underfilled=(
            balanced_queue.underfilled if balanced_queue is not None else False
        ),
    )


__all__ = ["REVIEW_RECOMMENDATION", "OriginalLabelAuditResult", "audit_original_labels"]
