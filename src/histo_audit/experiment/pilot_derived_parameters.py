"""Produce deterministic primary-study parameters from the sealed real-data pilot.

The producer is intentionally development-only.  It accepts a sealed pilot whose
model-facing crop and embedding caches contain official folds 1/2 only, rebuilds a
clean group-safe OOF classifier on the fixed audit pool, and derives:

* the confusion-targeted transition matrix, and
* tissue weights for group-conditional corruption.

No final-reference sample identifier, label, representation, or outcome is read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import array_artifact_sha256, canonical_sha256
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    sha256_file,
    verify_run_integrity,
)

CLASS_ORDER = (0, 1, 2, 3, 4)
PRODUCER_ID = "pilot_derived_primary_parameters_v1"


class PilotParameterDerivationError(ValueError):
    """Raised when pilot evidence is unsafe, incomplete, or misaligned."""


def _sequence_sha256(values: Sequence[object]) -> str:
    return canonical_sha256(list(values))


def _require_string_vector(values: NDArray[np.generic], name: str) -> NDArray[np.str_]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"S", "U"}:
        raise PilotParameterDerivationError(f"{name} must be a one-dimensional string array")
    result = np.asarray(array, dtype=np.str_)
    if not len(result) or any(not str(value) for value in result):
        raise PilotParameterDerivationError(f"{name} must contain non-empty values")
    if len(np.unique(result)) != len(result):
        raise PilotParameterDerivationError(f"{name} must be unique")
    return result


def derive_clean_oof_primary_parameters(
    features: NDArray[np.generic],
    pre_corruption_labels: Sequence[int] | NDArray[np.integer[Any]],
    group_ids: Sequence[str] | NDArray[np.str_],
    tissue_types: Sequence[str] | NDArray[np.str_],
    sample_ids: Sequence[str] | NDArray[np.str_],
    *,
    n_splits: int = 5,
    split_seed: int = 223,
    model_seed: int = 227,
    l2: float = 0.01,
    max_iter: int = 400,
    pseudocount: float = 1.0,
) -> dict[str, Any]:
    """Derive deterministic transition/tissue parameters from clean development OOF.

    ``model_seed`` is frozen and recorded even though the convex, zero-initialised
    logistic implementation does not consume it.
    """

    matrix = np.asarray(features)
    labels = np.asarray(pre_corruption_labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.str_)
    tissues = np.asarray(tissue_types, dtype=np.str_)
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    n = len(labels)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != n
        or groups.shape != (n,)
        or tissues.shape != (n,)
        or identifiers.shape != (n,)
        or not n
    ):
        raise PilotParameterDerivationError("features and audit metadata must be aligned")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise PilotParameterDerivationError("features must be a finite floating matrix")
    if set(labels.tolist()) != set(CLASS_ORDER):
        raise PilotParameterDerivationError("audit labels must contain exactly all fixed classes")
    if any(not str(value) for value in groups) or any(not str(value) for value in tissues):
        raise PilotParameterDerivationError("group and tissue identifiers must be non-empty")
    if any(not str(value) for value in identifiers) or len(np.unique(identifiers)) != n:
        raise PilotParameterDerivationError("sample identifiers must be non-empty and unique")
    if pseudocount <= 0.0 or not np.isfinite(pseudocount):
        raise PilotParameterDerivationError("pseudocount must be finite and positive")

    fold_plan = make_group_stratified_fold_plan(
        labels,
        tuple(str(value) for value in groups),
        n_splits=n_splits,
        class_order=CLASS_ORDER,
        seed=split_seed,
    )
    probabilities = np.full((n, len(CLASS_ORDER)), np.nan, dtype=np.float64)
    fold_ids = np.full(n, -1, dtype=np.int64)
    for fold in fold_plan.folds:
        estimator = MultinomialLogisticRegression(
            class_order=CLASS_ORDER,
            l2=l2,
            max_iter=max_iter,
            class_weight_balanced=True,
        )
        estimator.fit(matrix[fold.train_indices], labels[fold.train_indices])
        probabilities[fold.holdout_indices] = estimator.predict_proba(matrix[fold.holdout_indices])
        fold_ids[fold.holdout_indices] = fold.fold_id
    if (
        not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
        or np.any(fold_ids < 0)
    ):
        raise RuntimeError(
            "clean OOF derivation did not assign one valid probability row per sample"
        )

    predictions = np.asarray(CLASS_ORDER, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    errors = predictions != labels
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    transition = np.zeros_like(confusion, dtype=np.float64)
    for row in range(len(CLASS_ORDER)):
        off_diagonal = np.asarray(
            [confusion[row, column] + pseudocount for column in CLASS_ORDER if column != row],
            dtype=np.float64,
        )
        off_diagonal /= off_diagonal.sum()
        transition[row, np.asarray(CLASS_ORDER) != row] = off_diagonal
    if not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("derived transition matrix rows do not sum to one")

    tissue_rows: dict[str, dict[str, Any]] = {}
    for tissue in sorted(str(value) for value in np.unique(tissues)):
        members = tissues == tissue
        sample_count = int(members.sum())
        error_count = int(errors[members].sum())
        smoothed_error_rate = float((error_count + 1.0) / (sample_count + 2.0))
        tissue_rows[tissue] = {
            "sample_count": sample_count,
            "clean_oof_error_count": error_count,
            "beta_1_1_smoothed_error_rate": smoothed_error_rate,
        }
    maximum_rate = max(float(row["beta_1_1_smoothed_error_rate"]) for row in tissue_rows.values())
    weights = {
        tissue: float(row["beta_1_1_smoothed_error_rate"]) / maximum_rate
        for tissue, row in tissue_rows.items()
    }
    pooled_rate = float((int(errors.sum()) + 1.0) / (n + 2.0))
    default_weight = pooled_rate / maximum_rate

    return {
        "schema_version": 1,
        "producer_id": PRODUCER_ID,
        "class_order": list(CLASS_ORDER),
        "audit_pool": {
            "sample_count": n,
            "group_count": len(np.unique(groups)),
            "tissue_count": len(tissue_rows),
            "sample_sequence_sha256": _sequence_sha256([str(value) for value in identifiers]),
            "group_sequence_sha256": _sequence_sha256([str(value) for value in groups]),
            "pre_corruption_label_sequence_sha256": _sequence_sha256(
                [int(value) for value in labels]
            ),
            "tissue_sequence_sha256": _sequence_sha256([str(value) for value in tissues]),
            "feature_array_sha256": array_artifact_sha256(matrix),
        },
        "clean_group_oof": {
            "label_source": "pre_corruption_label",
            "fit_scope": "development_audit_pool_only",
            "splitter_class_name": fold_plan.splitter_class_name,
            "splitter_fallback_status": fold_plan.splitter_fallback_status,
            "splitter_fallback_reason": fold_plan.splitter_fallback_reason,
            "n_splits": n_splits,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "classifier": "multinomial_logistic_regression",
            "l2": l2,
            "max_iter": max_iter,
            "class_weight": "balanced_from_clean_development_audit_labels",
            "probabilities_sha256": array_artifact_sha256(probabilities),
            "predicted_labels_sha256": array_artifact_sha256(predictions),
            "fold_ids_sha256": array_artifact_sha256(fold_ids),
            "clean_oof_error_count": int(errors.sum()),
            "clean_oof_error_rate": float(errors.mean()),
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        },
        "confusion_targeted_corruption": {
            "derivation": (
                "zero diagonal; add one pseudo-count to every off-diagonal clean-OOF "
                "confusion count; normalise each row"
            ),
            "pseudocount": pseudocount,
            "transition_matrix": transition.tolist(),
            "clinical_realism_claim": False,
        },
        "group_conditional_corruption": {
            "grouping_field": "tissue_type",
            "derivation": (
                "Beta(1,1)-smoothed clean-OOF error rate per tissue, divided by the "
                "maximum tissue rate; default is the pooled smoothed rate on the same scale"
            ),
            "tissue_evidence": tissue_rows,
            "weights_by_value": weights,
            "default_weight": default_weight,
        },
    }


def _load_json_mapping(path: Path, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotParameterDerivationError(f"cannot load {role}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PilotParameterDerivationError(f"{role} must be a JSON object")
    return value


def produce_pilot_derived_primary_parameters(
    run_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate a sealed pilot and write one deterministic derivation record."""

    run = Path(run_directory).resolve()
    integrity = verify_run_integrity(run)
    if not integrity.valid:
        raise PilotParameterDerivationError(
            "pilot run integrity failed: " + "; ".join(integrity.errors)
        )
    marker_path = run / ".immutable.json"
    metrics_path = run / "metrics.json"
    selection_path = run / "selected_groups_and_samples.json"
    crop_path = run / "representations" / "pannuke_crops.npz"
    crop_sidecar_path = crop_path.with_suffix(f"{crop_path.suffix}.metadata.json")
    embedding_path = run / "representations" / "pannuke_resnet18_target_highlighted_embeddings.npz"
    embedding_sidecar_path = embedding_path.with_suffix(f"{embedding_path.suffix}.metadata.json")
    required_paths = (
        marker_path,
        metrics_path,
        selection_path,
        crop_path,
        crop_sidecar_path,
        embedding_path,
        embedding_sidecar_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise PilotParameterDerivationError(f"pilot derivation inputs are missing: {missing}")

    marker = _load_json_mapping(marker_path, "immutable marker")
    selection = _load_json_mapping(selection_path, "pilot selection record")
    crop_sidecar = _load_json_mapping(crop_sidecar_path, "crop-cache sidecar")
    embedding_sidecar = _load_json_mapping(embedding_sidecar_path, "embedding-cache sidecar")
    if marker.get("status") != "completed" or marker.get("run_id") != run.name:
        raise PilotParameterDerivationError(
            "pilot immutable marker is not a completed matching run"
        )
    if selection.get("development_official_folds") != [1, 2]:
        raise PilotParameterDerivationError("pilot selection is not restricted to folds 1/2")
    for forbidden_flag in (
        "final_reference_class_labels_read",
        "final_reference_sample_ids_read",
        "final_reference_representations_extracted",
        "final_reference_outcomes_used",
    ):
        if selection.get(forbidden_flag) is not False:
            raise PilotParameterDerivationError(
                f"pilot selection does not prove {forbidden_flag}=false"
            )

    try:
        with np.load(crop_path, allow_pickle=False) as crop_payload:
            crop_sample_ids = _require_string_vector(crop_payload["sample_ids"], "crop sample IDs")
            crop_groups = np.asarray(crop_payload["group_ids"], dtype=np.str_)
            crop_labels = np.asarray(crop_payload["pre_corruption_labels"], dtype=np.int64)
            crop_tissues = np.asarray(crop_payload["tissue_types"], dtype=np.str_)
            crop_folds = np.asarray(crop_payload["official_folds"], dtype=np.int64)
            primary_eligible = np.asarray(crop_payload["primary_eligible"])
        with np.load(embedding_path, allow_pickle=False) as embedding_payload:
            embedding_sample_ids = _require_string_vector(
                embedding_payload["sample_ids"], "embedding sample IDs"
            )
            embeddings = np.asarray(embedding_payload["embeddings"])
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, PilotParameterDerivationError):
            raise
        raise PilotParameterDerivationError(f"cannot safely load pilot caches: {exc}") from exc

    n = len(crop_sample_ids)
    if (
        crop_groups.shape != (n,)
        or crop_labels.shape != (n,)
        or crop_tissues.shape != (n,)
        or crop_folds.shape != (n,)
        or primary_eligible.shape != (n,)
        or primary_eligible.dtype != np.bool_
        or not primary_eligible.all()
    ):
        raise PilotParameterDerivationError("development crop-cache metadata are malformed")
    if set(crop_folds.tolist()) != {1, 2}:
        raise PilotParameterDerivationError(
            "model-facing crop cache must contain development folds 1/2 only"
        )
    if not np.array_equal(embedding_sample_ids, crop_sample_ids):
        raise PilotParameterDerivationError("embedding/crop sample order differs")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != n
        or not np.issubdtype(embeddings.dtype, np.floating)
        or not np.isfinite(embeddings).all()
    ):
        raise PilotParameterDerivationError("embedding matrix is malformed")
    if crop_sidecar.get("cache_file_sha256") != sha256_file(crop_path):
        raise PilotParameterDerivationError("crop cache differs from its sidecar")
    if embedding_sidecar.get("cache_file_sha256") != sha256_file(embedding_path):
        raise PilotParameterDerivationError("embedding cache differs from its sidecar")

    raw_audit_ids = selection.get("audit_sample_ids")
    raw_audit_groups = selection.get("audit_groups")
    if not isinstance(raw_audit_ids, list) or not isinstance(raw_audit_groups, list):
        raise PilotParameterDerivationError("pilot selection lacks audit sample/group lists")
    audit_ids = tuple(str(value) for value in raw_audit_ids)
    audit_groups = {str(value) for value in raw_audit_groups}
    if not audit_ids or len(set(audit_ids)) != len(audit_ids) or not audit_groups:
        raise PilotParameterDerivationError("pilot audit sample/group selection is invalid")
    index_by_sample = {str(value): index for index, value in enumerate(crop_sample_ids)}
    if any(value not in index_by_sample for value in audit_ids):
        raise PilotParameterDerivationError("pilot audit selection is absent from crop cache")
    audit_indices = np.asarray([index_by_sample[value] for value in audit_ids], dtype=np.int64)
    selected_groups = crop_groups[audit_indices]
    if set(str(value) for value in selected_groups) != audit_groups:
        raise PilotParameterDerivationError("pilot audit groups differ from selected sample groups")

    derived = derive_clean_oof_primary_parameters(
        embeddings[audit_indices],
        crop_labels[audit_indices],
        selected_groups,
        crop_tissues[audit_indices],
        np.asarray(audit_ids, dtype=np.str_),
    )
    record = {
        **derived,
        "producer_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_pilot": {
            "run_id": run.name,
            "artifact_root_sha256": integrity.actual_root_sha256,
            "immutable_marker_sha256": sha256_file(marker_path),
            "metrics_sha256": sha256_file(metrics_path),
            "selection_record_sha256": sha256_file(selection_path),
            "crop_cache_sha256": sha256_file(crop_path),
            "crop_sidecar_sha256": sha256_file(crop_sidecar_path),
            "highlighted_embedding_cache_sha256": sha256_file(embedding_path),
            "highlighted_embedding_sidecar_sha256": sha256_file(embedding_sidecar_path),
            "development_official_folds": [1, 2],
            "final_reference_policy": (
                "no final-fold sample identifier, label, representation, or outcome read"
            ),
        },
    }
    destination = atomic_write_json(output_path, record)
    reloaded = _load_json_mapping(destination, "written pilot-derived parameter record")
    if reloaded != record:
        raise RuntimeError("written pilot-derived parameter record failed semantic readback")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Sealed eligible PanNuke pilot run")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON record")
    args = parser.parse_args(argv)
    record = produce_pilot_derived_primary_parameters(args.run, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "producer_id": record["producer_id"],
                "sample_count": record["audit_pool"]["sample_count"],
                "source_run_id": record["source_pilot"]["run_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the functional command
    raise SystemExit(main())


__all__ = [
    "PilotParameterDerivationError",
    "derive_clean_oof_primary_parameters",
    "produce_pilot_derived_primary_parameters",
]
