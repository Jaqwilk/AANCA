"""Autoresearch-inspired development search with patient-group safety.

This module deliberately accepts only a prepared MoNuSAC *training* split.  It has
no path, argument or loader for the published external tests.  Candidate selection
is nested inside patient groups and source labels remain immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from histo_audit.auditing.scores import fixed_hybrid_score, score_annotations
from histo_audit.auditing.strategies import group_safe_audit_scores
from histo_audit.auditing.two_queue import (
    GROUP_SAFE_OOF_EVIDENCE,
    QueueConstraints,
    build_two_review_queues,
    draw_matched_random_comparator,
)
from histo_audit.config import load_config_with_file_sha256
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.evaluation.restoration import (
    classification_metrics,
    macro_f1_from_confusion,
    per_class_recall_from_confusion,
)
from histo_audit.evaluation.review_training import SoftTargetMultinomialLogisticRegression
from histo_audit.external_validation.monusac import CLASS_ORDER, MoNuSACPreparedData
from histo_audit.statistics.review import average_precision, budget_count, rank_indices

TrialStage = Literal["ranking_screen", "downstream_screen", "full_nested", "lockbox"]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def load_autoresearch_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load the development search config and enforce its claim/test boundaries."""

    path = Path(repository_root).resolve() / "configs" / "aanca_autoresearch_development.yaml"
    config, digest = load_config_with_file_sha256(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("AANCA autoresearch configuration is malformed")
    if config.get("disposition") != "post_external_development_only_method_search":
        raise ValueError("AANCA autoresearch must remain development-only")
    data = config.get("data")
    boundary = config.get("claim_boundary")
    if not isinstance(data, Mapping) or data.get("permitted_split") != "official_train_only":
        raise ValueError("AANCA autoresearch may consume only official MoNuSAC training data")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(name) is not False
        for name in (
            "natural_error_detection_claim_permitted",
            "pathologist_error_claim_permitted",
            "clinical_or_operational_utility_claim_permitted",
            "automatic_annotation_change_permitted",
            "internal_lockbox_is_external_confirmation",
        )
    ):
        raise ValueError("AANCA autoresearch claim boundary is not fail-closed")
    return config, digest


@dataclass(frozen=True, slots=True)
class AutoresearchPartition:
    """One deterministic discovery/lockbox split and nested discovery folds."""

    discovery_indices: NDArray[np.int64]
    lockbox_indices: NDArray[np.int64]
    discovery_outer_folds: tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]
    discovery_groups: tuple[str, ...]
    lockbox_groups: tuple[str, ...]
    partition_sha256: str
    split_seed: int
    source_annotations_modified: bool = False

    def as_dict(self, sample_ids: Sequence[str]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "split_seed": self.split_seed,
            "partition_sha256": self.partition_sha256,
            "discovery_groups": list(self.discovery_groups),
            "lockbox_groups": list(self.lockbox_groups),
            "discovery_sample_ids": [str(sample_ids[index]) for index in self.discovery_indices],
            "lockbox_sample_ids": [str(sample_ids[index]) for index in self.lockbox_indices],
            "outer_folds": [
                {
                    "training_sample_ids": [str(sample_ids[index]) for index in train],
                    "validation_sample_ids": [str(sample_ids[index]) for index in validation],
                }
                for train, validation in self.discovery_outer_folds
            ],
            "source_annotations_modified": False,
        }


def _validate_training_only(prepared: MoNuSACPreparedData, config: Mapping[str, Any]) -> None:
    expected_prefix = str(config["data"]["expected_sample_id_prefix"])
    sample_ids = prepared.manifest["sample_id"].astype(str)
    if prepared.split != "train" or not sample_ids.str.startswith(expected_prefix).all():
        raise ValueError("autoresearch received a non-training MoNuSAC sample")
    if int(prepared.manifest["group_id"].nunique()) != int(
        config["data"]["expected_patient_groups_after_exclusion"]
    ):
        raise ValueError("autoresearch training patient count differs from the frozen config")
    if (
        prepared.manifest["group_id"].astype(str).str[:12].tolist()
        != prepared.manifest["patient_id"].astype(str).str[:12].tolist()
    ):
        raise ValueError("autoresearch group_id is not the TCGA patient identity")


def build_autoresearch_partition(
    prepared: MoNuSACPreparedData, config: Mapping[str, Any]
) -> AutoresearchPartition:
    """Create the lockbox before any candidate outcome is calculated."""

    _validate_training_only(prepared, config)
    labels = prepared.manifest["reference_label"].to_numpy(dtype=np.int64)
    groups = prepared.manifest["group_id"].astype(str).tolist()
    partition_config = config["partition"]
    seed = int(partition_config["seed"])
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=int(partition_config["folds"]),
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=seed,
    )
    lockbox_fold = int(partition_config["internal_lockbox_fold"])
    if lockbox_fold < 0 or lockbox_fold >= len(plan.folds):
        raise ValueError("internal lockbox fold lies outside the group partition")
    selected_fold = plan.folds[lockbox_fold]
    discovery = np.asarray(selected_fold.train_indices, dtype=np.int64)
    lockbox = np.asarray(selected_fold.holdout_indices, dtype=np.int64)
    discovery_groups = tuple(sorted(set(str(groups[index]) for index in discovery)))
    lockbox_groups = tuple(sorted(set(str(groups[index]) for index in lockbox)))
    if set(discovery_groups).intersection(lockbox_groups):
        raise RuntimeError("autoresearch discovery and internal lockbox groups overlap")

    discovery_plan = make_group_stratified_fold_plan(
        labels[discovery],
        [groups[index] for index in discovery],
        n_splits=int(partition_config["discovery_outer_folds"]),
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=seed + 1,
    )
    outer_folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    validation_coverage = np.zeros(len(prepared.manifest), dtype=np.int64)
    for fold in discovery_plan.folds:
        train = discovery[fold.train_indices]
        validation = discovery[fold.holdout_indices]
        if set(groups[index] for index in train).intersection(
            groups[index] for index in validation
        ):
            raise RuntimeError("autoresearch outer patient-group leakage")
        validation_coverage[validation] += 1
        outer_folds.append((train, validation))
    if not np.array_equal(validation_coverage[discovery], np.ones(len(discovery), dtype=np.int64)):
        raise RuntimeError("autoresearch outer folds do not cover discovery exactly once")
    if np.any(validation_coverage[lockbox]):
        raise RuntimeError("internal lockbox entered discovery outer folds")

    authority = {
        "split_seed": seed,
        "discovery_sample_ids": prepared.manifest.iloc[discovery]["sample_id"].astype(str).tolist(),
        "lockbox_sample_ids": prepared.manifest.iloc[lockbox]["sample_id"].astype(str).tolist(),
        "outer_folds": [
            {
                "train": prepared.manifest.iloc[train]["sample_id"].astype(str).tolist(),
                "validation": prepared.manifest.iloc[validation]["sample_id"].astype(str).tolist(),
            }
            for train, validation in outer_folds
        ],
    }
    return AutoresearchPartition(
        discovery_indices=discovery,
        lockbox_indices=lockbox,
        discovery_outer_folds=tuple(outer_folds),
        discovery_groups=discovery_groups,
        lockbox_groups=lockbox_groups,
        partition_sha256=_semantic_sha256(authority),
        split_seed=seed,
    )


def _crop_statistics(prepared: MoNuSACPreparedData) -> NDArray[np.float32]:
    """Build compact label-independent colour, context and box features."""

    crops = prepared.crops
    manifest = prepared.manifest
    if crops.ndim != 4 or crops.shape[0] != len(manifest) or crops.shape[-1] != 3:
        raise ValueError("MoNuSAC crops do not align with the canonical manifest")
    output = np.empty((len(crops), 20), dtype=np.float32)
    half = crops.shape[1] // 2
    quarter = crops.shape[1] // 4
    centre_slice = (slice(half - quarter, half + quarter),) * 2
    chunk_size = 512
    for start in range(0, len(crops), chunk_size):
        stop = min(start + chunk_size, len(crops))
        chunk = crops[start:stop].astype(np.float32) / 255.0
        flattened = chunk.reshape(len(chunk), -1, 3)
        centre = chunk[:, centre_slice[0], centre_slice[1], :].reshape(len(chunk), -1, 3)
        global_mean = flattened.mean(axis=1)
        global_std = flattened.std(axis=1)
        centre_mean = centre.mean(axis=1)
        centre_std = centre.std(axis=1)
        output[start:stop, :3] = global_mean
        output[start:stop, 3:6] = global_std
        output[start:stop, 6:9] = centre_mean
        output[start:stop, 9:12] = centre_std
        output[start:stop, 12:15] = centre_mean - global_mean
    width = (manifest["xmax"] - manifest["xmin"]).to_numpy(dtype=np.float64)
    height = (manifest["ymax"] - manifest["ymin"]).to_numpy(dtype=np.float64)
    safe_height = np.maximum(height, 1.0e-6)
    output[:, 15] = np.log1p(np.maximum(width, 0.0)).astype(np.float32)
    output[:, 16] = np.log1p(np.maximum(height, 0.0)).astype(np.float32)
    output[:, 17] = np.log1p(np.maximum(width * height, 0.0)).astype(np.float32)
    output[:, 18] = (width / safe_height).astype(np.float32)
    output[:, 19] = np.log(np.maximum(width / safe_height, 1.0e-6)).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("label-independent MoNuSAC crop statistics are non-finite")
    return output


def build_autoresearch_feature_views(
    prepared: MoNuSACPreparedData,
    resnet18_context_64: NDArray[np.generic],
) -> dict[str, NDArray[np.float32]]:
    """Build only provenance-compatible feature views declared as required."""

    embeddings = np.asarray(resnet18_context_64, dtype=np.float32)
    if embeddings.shape != (len(prepared.manifest), 512) or not np.isfinite(embeddings).all():
        raise ValueError("ResNet-18 embeddings differ from the canonical MoNuSAC training split")
    statistics = _crop_statistics(prepared)
    combined = np.concatenate([embeddings, statistics], axis=1).astype(np.float32, copy=False)
    return {
        "resnet18_context_64": embeddings,
        "resnet18_context_64_plus_stats": combined,
    }


@dataclass(frozen=True, slots=True)
class AutoresearchCandidate:
    """One declared review-and-training policy candidate."""

    feature_view: str = "resnet18_context_64"
    audit_l2: float = 0.01
    audit_class_weight_balanced: bool = True
    risk_method: str = "nearest_neighbour_disagreement"
    neighbour_k: int = 7
    hybrid_self_confidence_weight: float = 0.5
    queue_preset: str = "balanced_current"
    review_budget: float = 0.05
    intervention: str = "controlled_restore"
    downstream_l2: float = 0.01
    downstream_class_weight_balanced: bool = True

    @property
    def candidate_sha256(self) -> str:
        return _semantic_sha256(self.as_dict())

    @property
    def short_id(self) -> str:
        return self.candidate_sha256[:12]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AutoresearchCandidate:
        return cls(**{field_name: value[field_name] for field_name in cls.__dataclass_fields__})

    def validate(self, config: Mapping[str, Any], feature_views: Mapping[str, Any]) -> None:
        space = config["candidate_space"]
        if self.feature_view not in feature_views:
            raise ValueError(f"candidate feature view is unavailable: {self.feature_view}")
        if self.audit_l2 not in tuple(float(value) for value in space["audit_l2"]):
            raise ValueError("candidate audit L2 is outside the declared search space")
        if self.audit_class_weight_balanced not in tuple(
            bool(value) for value in space["audit_class_weight_balanced"]
        ):
            raise ValueError("candidate audit class weighting is undeclared")
        probability = tuple(str(value) for value in space["risk_methods"]["probability"])
        if self.risk_method not in {
            *probability,
            "nearest_neighbour_disagreement",
            "fixed_hybrid",
        }:
            raise ValueError("candidate risk method is outside the declared search space")
        if self.neighbour_k not in tuple(
            int(value) for value in space["risk_methods"]["neighbour"]["k"]
        ):
            raise ValueError("candidate neighbour k is outside the declared search space")
        if self.risk_method == "fixed_hybrid" and self.hybrid_self_confidence_weight not in tuple(
            float(value) for value in space["risk_methods"]["hybrid"]["self_confidence_weights"]
        ):
            raise ValueError("candidate hybrid weight is outside the declared search space")
        if self.queue_preset not in space["queue_presets"]:
            raise ValueError("candidate queue preset is outside the declared search space")
        if self.review_budget not in tuple(float(value) for value in space["review_budgets"]):
            raise ValueError("candidate review budget is outside the declared search space")
        if self.intervention not in tuple(str(value) for value in space["interventions"]):
            raise ValueError("candidate intervention is outside the declared search space")
        if self.downstream_l2 not in tuple(float(value) for value in space["downstream_l2"]):
            raise ValueError("candidate downstream L2 is outside the declared search space")
        if self.downstream_class_weight_balanced not in tuple(
            bool(value) for value in space["downstream_class_weight_balanced"]
        ):
            raise ValueError("candidate downstream class weighting is undeclared")


@dataclass(slots=True)
class _AuditEvidence:
    full_train_indices: NDArray[np.int64]
    reference: NDArray[np.int64]
    observed: NDArray[np.int64]
    injected: NDArray[np.bool_]
    probabilities: NDArray[np.float64]
    fold_ids: NDArray[np.int64]
    training_groups_by_fold: dict[int, tuple[str, ...]]
    risk_cache: dict[tuple[str, int, float], NDArray[np.float64]] = field(default_factory=dict)


def _one_hot(labels: NDArray[np.int64], class_count: int) -> NDArray[np.float64]:
    output = np.zeros((len(labels), class_count), dtype=np.float64)
    output[np.arange(len(labels)), labels] = 1.0
    return output


def _fit_predict_task(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    validation_features: NDArray[np.float64],
    *,
    l2: float,
    class_weight_balanced: bool,
    sample_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    classes = tuple(range(len(CLASS_ORDER)))
    if sample_weights is None or np.all(sample_weights == 1.0):
        model = MultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=400,
            class_weight_balanced=class_weight_balanced,
        ).fit(train_features, train_labels)
    else:
        model = SoftTargetMultinomialLogisticRegression(
            class_order=classes,
            l2=l2,
            max_iter=400,
            class_weight_balanced=class_weight_balanced,
        ).fit_soft_targets(
            train_features,
            _one_hot(train_labels, len(classes)),
            sample_weight=sample_weights,
        )
    return np.asarray(model.predict_proba(validation_features), dtype=np.float64)


def _confusions_by_group(
    reference: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    unique_groups: tuple[str, ...],
) -> NDArray[np.int64]:
    predictions = np.argmax(probabilities, axis=1)
    lookup = {group: index for index, group in enumerate(unique_groups)}
    output = np.zeros((len(unique_groups), len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    for truth, prediction, group in zip(reference, predictions, groups, strict=True):
        output[lookup[str(group)], int(truth), int(prediction)] += 1
    return output


def _interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("autoresearch bootstrap has no finite values")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


class AutoresearchEvaluator:
    """Fixed evaluator with caches shared across development candidates."""

    def __init__(
        self,
        prepared: MoNuSACPreparedData,
        feature_views: Mapping[str, NDArray[np.generic]],
        partition: AutoresearchPartition,
        config: Mapping[str, Any],
        *,
        config_sha256: str,
    ) -> None:
        _validate_training_only(prepared, config)
        self.prepared = prepared
        self.config = config
        self.config_sha256 = str(config_sha256)
        self.partition = partition
        self.feature_views = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in feature_views.items()
        }
        for name, values in self.feature_views.items():
            if values.ndim != 2 or values.shape[0] != len(prepared.manifest):
                raise ValueError(f"feature view {name!r} differs from the training manifest")
            if not np.isfinite(values).all():
                raise ValueError(f"feature view {name!r} contains non-finite values")
        self._audit_cache: dict[tuple[object, ...], _AuditEvidence] = {}
        self._baseline_cache: dict[tuple[object, ...], NDArray[np.float64]] = {}
        self._full_discovery_audit_cache: dict[tuple[object, ...], _AuditEvidence] = {}
        self._neighbour_risk_cache: dict[tuple[object, ...], NDArray[np.float64]] = {}
        self._sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
        self._groups = prepared.manifest["group_id"].astype(str).tolist()
        self._organs = prepared.manifest["organ"].astype(str).tolist()
        self._reference = prepared.manifest["reference_label"].to_numpy(dtype=np.int64)
        self._classes = tuple(range(len(CLASS_ORDER)))
        if set(self._groups[index] for index in partition.discovery_indices).intersection(
            self._groups[index] for index in partition.lockbox_indices
        ):
            raise RuntimeError("autoresearch evaluator received an overlapping lockbox")

    def _corrupt(self, indices: NDArray[np.int64], seed: int, feature_view: str) -> Any:
        corruption_config = self.config["controlled_corruption"]
        authority = _semantic_sha256(
            {
                "manifest_sha256": self.prepared.manifest_sha256,
                "sample_ids": [self._sample_ids[index] for index in indices],
                "partition_sha256": self.partition.partition_sha256,
            }
        )
        return apply_controlled_corruption(
            self._reference[indices],
            sample_ids=[self._sample_ids[index] for index in indices],
            group_ids=[self._groups[index] for index in indices],
            rate=float(corruption_config["rate"]),
            mechanism=str(corruption_config["mechanism"]),
            seed=int(seed),
            n_classes=len(CLASS_ORDER),
            generator_representation=None,
            auditor_representation=feature_view,
            upstream_manifest_hash=authority,
        )

    def _oof_audit_evidence(
        self,
        full_train_indices: NDArray[np.int64],
        *,
        seed: int,
        feature_view: str,
        audit_l2: float,
        audit_class_weight_balanced: bool,
        inner_folds: int,
    ) -> _AuditEvidence:
        cache_key = (
            tuple(int(value) for value in full_train_indices),
            int(seed),
            feature_view,
            float(audit_l2),
            bool(audit_class_weight_balanced),
            int(inner_folds),
        )
        cached = self._audit_cache.get(cache_key)
        if cached is not None:
            return cached
        reference = self._reference[full_train_indices]
        corruption = self._corrupt(full_train_indices, seed, feature_view)
        observed = np.asarray(corruption.observed_labels, dtype=np.int64)
        injected = np.asarray(corruption.is_injected_corruption, dtype=bool)
        groups = [self._groups[index] for index in full_train_indices]
        features = self.feature_views[feature_view][full_train_indices]
        plan = make_group_stratified_fold_plan(
            reference,
            groups,
            n_splits=inner_folds,
            class_order=self._classes,
            seed=int(self.config["partition"]["seed"]) + 10,
        )
        probabilities = np.full(
            (len(full_train_indices), len(CLASS_ORDER)), np.nan, dtype=np.float64
        )
        fold_ids = np.full(len(full_train_indices), -1, dtype=np.int64)
        training_groups_by_fold: dict[int, tuple[str, ...]] = {}
        for fold in plan.folds:
            model = MultinomialLogisticRegression(
                class_order=self._classes,
                l2=audit_l2,
                max_iter=400,
                class_weight_balanced=audit_class_weight_balanced,
            ).fit(features[fold.train_indices], observed[fold.train_indices])
            probabilities[fold.holdout_indices] = model.predict_proba(
                features[fold.holdout_indices]
            )
            fold_ids[fold.holdout_indices] = fold.fold_id
            training_groups_by_fold[fold.fold_id] = fold.training_groups
        if np.any(fold_ids < 0) or not np.isfinite(probabilities).all():
            raise RuntimeError("autoresearch inner OOF audit evidence is incomplete")
        evidence = _AuditEvidence(
            full_train_indices=np.asarray(full_train_indices, dtype=np.int64),
            reference=np.asarray(reference, dtype=np.int64),
            observed=observed,
            injected=injected,
            probabilities=probabilities,
            fold_ids=fold_ids,
            training_groups_by_fold=training_groups_by_fold,
        )
        self._audit_cache[cache_key] = evidence
        return evidence

    def _risk(
        self, evidence: _AuditEvidence, candidate: AutoresearchCandidate
    ) -> NDArray[np.float64]:
        key = (
            candidate.risk_method,
            int(candidate.neighbour_k),
            float(candidate.hybrid_self_confidence_weight),
        )
        cached = evidence.risk_cache.get(key)
        if cached is not None:
            return cached
        if candidate.risk_method in {
            "self_confidence",
            "prediction_margin",
            "predictive_entropy",
        }:
            risk = score_annotations(
                evidence.observed,
                evidence.probabilities,
                method=candidate.risk_method,
                class_order=self._classes,
            )
        else:
            neighbour_key = (
                tuple(int(value) for value in evidence.full_train_indices),
                _semantic_sha256(evidence.observed.tolist()),
                candidate.feature_view,
                int(candidate.neighbour_k),
            )
            neighbour_risk = self._neighbour_risk_cache.get(neighbour_key)
            if neighbour_risk is None:
                score = group_safe_audit_scores(
                    self.feature_views[candidate.feature_view][evidence.full_train_indices],
                    evidence.observed,
                    evidence.probabilities,
                    [self._groups[index] for index in evidence.full_train_indices],
                    evidence.fold_ids,
                    evidence.training_groups_by_fold,
                    sample_ids=[self._sample_ids[index] for index in evidence.full_train_indices],
                    method="nearest_neighbour_disagreement",
                    class_order=self._classes,
                    neighbour_k=candidate.neighbour_k,
                    neighbour_metric="cosine",
                )
                neighbour_risk = np.asarray(score.risk_scores, dtype=np.float64)
                self._neighbour_risk_cache[neighbour_key] = neighbour_risk
            if candidate.risk_method == "nearest_neighbour_disagreement":
                risk = neighbour_risk
            else:
                self_confidence = score_annotations(
                    evidence.observed,
                    evidence.probabilities,
                    method="self_confidence",
                    class_order=self._classes,
                )
                risk = fixed_hybrid_score(
                    {
                        "self_confidence": self_confidence,
                        "nearest_neighbour_disagreement": neighbour_risk,
                    },
                    components=("self_confidence", "nearest_neighbour_disagreement"),
                    weights=(
                        candidate.hybrid_self_confidence_weight,
                        1.0 - candidate.hybrid_self_confidence_weight,
                    ),
                )
        risk = np.asarray(risk, dtype=np.float64)
        evidence.risk_cache[key] = risk
        return risk

    def _queue_constraints(
        self, candidate: AutoresearchCandidate, requested_count: int
    ) -> QueueConstraints | None:
        preset = self.config["candidate_space"]["queue_presets"][candidate.queue_preset]
        if preset is None:
            return None

        def cap(name: str) -> int:
            return max(1, math.ceil(requested_count * float(preset[name])))

        return QueueConstraints(
            requested_count=requested_count,
            max_per_group=cap("max_per_group_fraction"),
            max_per_class=cap("max_per_class_fraction"),
            max_per_tissue=cap("max_per_tissue_fraction"),
            max_per_transition=cap("max_per_transition_fraction"),
            minimum_cosine_distance=float(preset["minimum_cosine_distance"]),
        )

    def _select(
        self,
        evidence: _AuditEvidence,
        risk: NDArray[np.float64],
        candidate: AutoresearchCandidate,
    ) -> tuple[NDArray[np.int64], dict[str, object]]:
        count = budget_count(len(evidence.observed), candidate.review_budget)
        local_sample_ids = [self._sample_ids[index] for index in evidence.full_train_indices]
        constraints = self._queue_constraints(candidate, count)
        if constraints is None:
            selected = rank_indices(risk, tie_break_ids=local_sample_ids)[:count].astype(np.int64)
            return selected, {
                "requested_count": count,
                "selected_count": len(selected),
                "underfilled": False,
                "queue_preset": candidate.queue_preset,
            }
        proposed = np.argmax(evidence.probabilities, axis=1).astype(np.int64)
        queues = build_two_review_queues(
            risk,
            [self._groups[index] for index in evidence.full_train_indices],
            evidence.observed.tolist(),
            local_sample_ids,
            quality_constraints=constraints,
            model_constraints=QueueConstraints(requested_count=count),
            annotation_evidence_role=GROUP_SAFE_OOF_EVIDENCE,
            proposed_labels=proposed.tolist(),
            tissue_types=[self._organs[index] for index in evidence.full_train_indices],
            embeddings=self.feature_views[candidate.feature_view][evidence.full_train_indices],
        )
        queue = queues.quality_control
        return queue.selected_indices, {
            "requested_count": count,
            "selected_count": queue.selected_count,
            "underfilled": queue.underfilled,
            "queue_preset": candidate.queue_preset,
            "rejection_counts": queue.rejection_counts,
        }

    def _matched_random_indices(
        self,
        evidence: _AuditEvidence,
        selected: NDArray[np.int64],
        *,
        repetitions: int,
        seed: int,
        fold_id: int,
    ) -> tuple[NDArray[np.int64], ...]:
        proposed = np.argmax(evidence.probabilities, axis=1).astype(np.int64)
        transitions = [
            f"{source}->{target}"
            for source, target in zip(evidence.observed, proposed, strict=True)
        ]
        match_values = {
            "observed_class": evidence.observed.tolist(),
            "organ": [self._organs[index] for index in evidence.full_train_indices],
            "proposed_transition": transitions,
        }
        fields = tuple(str(value) for value in self.config["controls"]["matched_random_fields"])
        selected_values = {name: match_values[name] for name in fields}
        output: list[NDArray[np.int64]] = []
        base_seed = int(self.config["controls"]["bootstrap_seed"])
        for repeat in range(repetitions):
            comparator = draw_matched_random_comparator(
                selected,
                np.ones(len(evidence.observed), dtype=bool),
                [self._sample_ids[index] for index in evidence.full_train_indices],
                selected_values,
                seed=base_seed + int(seed) % 10000 + fold_id * 101 + repeat,
            )
            if not comparator.available:
                raise RuntimeError(
                    "exact matched-random comparator is unavailable: "
                    f"{comparator.unavailable_reason}"
                )
            output.append(np.asarray(comparator.comparator_indices, dtype=np.int64))
        return tuple(output)

    @staticmethod
    def _derive_intervention(
        observed: NDArray[np.int64],
        reference: NDArray[np.int64],
        injected: NDArray[np.bool_],
        selected: NDArray[np.int64],
        policy: str,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
        source_observed = observed.copy()
        labels = observed.copy()
        weights = np.ones(len(observed), dtype=np.float64)
        restored_count = 0
        if policy.startswith("controlled_restore"):
            restored = selected[injected[selected]]
            labels[restored] = reference[restored]
            restored_count = len(restored)
            if policy == "controlled_restore_selected_weight_0_5":
                weights[selected] = 0.5
            elif policy != "controlled_restore":
                raise ValueError(f"unsupported controlled restoration policy: {policy}")
        elif policy == "flag_downweight_0_5":
            weights[selected] = 0.5
        elif policy == "flag_exclude":
            weights[selected] = 0.0
        else:
            raise ValueError(f"unsupported autoresearch intervention policy: {policy}")
        if not np.array_equal(observed, source_observed):
            raise RuntimeError("source observed labels changed during intervention derivation")
        return labels, weights, restored_count

    def _baseline_probabilities(
        self,
        evidence: _AuditEvidence,
        validation_indices: NDArray[np.int64],
        candidate: AutoresearchCandidate,
        *,
        seed: int,
        fold_id: int,
    ) -> NDArray[np.float64]:
        key = (
            int(seed),
            int(fold_id),
            tuple(int(value) for value in evidence.full_train_indices),
            tuple(int(value) for value in validation_indices),
            candidate.feature_view,
            candidate.downstream_l2,
            candidate.downstream_class_weight_balanced,
        )
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached
        features = self.feature_views[candidate.feature_view]
        probabilities = _fit_predict_task(
            features[evidence.full_train_indices],
            evidence.observed,
            features[validation_indices],
            l2=candidate.downstream_l2,
            class_weight_balanced=candidate.downstream_class_weight_balanced,
        )
        self._baseline_cache[key] = probabilities
        return probabilities

    @staticmethod
    def _retrieval_summary(
        candidate_counts: NDArray[np.int64],
        random_counts: NDArray[np.int64],
        *,
        iterations: int,
        seed: int,
    ) -> dict[str, object]:
        """Compare repeated nested review decisions by resampling whole patients."""

        if candidate_counts.ndim != 3 or candidate_counts.shape[-1] != 2:
            raise ValueError("candidate retrieval counts have an unexpected shape")
        if (
            random_counts.ndim != 4
            or random_counts.shape[0] != candidate_counts.shape[0]
            or random_counts.shape[2:] != candidate_counts.shape[1:]
        ):
            raise ValueError("matched-random retrieval counts do not align")

        def precision(counts: NDArray[np.int64]) -> float:
            reviewed = int(counts[..., 0].sum())
            return float(counts[..., 1].sum() / reviewed) if reviewed else float("nan")

        candidate_by_seed = np.asarray(
            [precision(counts) for counts in candidate_counts], dtype=np.float64
        )
        random_by_seed = np.asarray(
            [
                np.mean([precision(counts) for counts in seed_counts])
                for seed_counts in random_counts
            ],
            dtype=np.float64,
        )
        differences_by_seed = candidate_by_seed - random_by_seed
        rng = np.random.default_rng(seed)
        differences: list[float] = []
        group_count = candidate_counts.shape[1]
        for _ in range(iterations):
            sampled = rng.integers(0, group_count, size=group_count)
            seed_differences: list[float] = []
            for seed_index in range(candidate_counts.shape[0]):
                candidate_value = precision(candidate_counts[seed_index, sampled])
                random_values = [
                    precision(random_counts[seed_index, repeat, sampled])
                    for repeat in range(random_counts.shape[1])
                ]
                random_values = [value for value in random_values if np.isfinite(value)]
                if np.isfinite(candidate_value) and random_values:
                    seed_differences.append(candidate_value - float(np.mean(random_values)))
            if seed_differences:
                differences.append(float(np.mean(seed_differences)))
        return {
            "candidate_precision": float(np.mean(candidate_by_seed)),
            "mean_matched_random_precision": float(np.mean(random_by_seed)),
            "candidate_minus_matched_random_precision": float(np.mean(differences_by_seed)),
            "interval_95": list(_interval(differences)),
            "differences_by_corruption_seed": differences_by_seed.tolist(),
            "reviewed_decisions": int(candidate_counts[..., 0].sum()),
            "injected_changes_found": int(candidate_counts[..., 1].sum()),
            "matched_random_repetitions": random_counts.shape[1],
            "bootstrap_iterations": iterations,
            "valid_bootstrap_iterations": len(differences),
        }

    @staticmethod
    def _downstream_summary(
        reference: NDArray[np.int64],
        groups: NDArray[np.str_],
        candidate_probabilities: NDArray[np.float64],
        baseline_probabilities: NDArray[np.float64],
        random_probabilities: NDArray[np.float64],
        *,
        iterations: int,
        seed: int,
    ) -> dict[str, object]:
        if candidate_probabilities.ndim != 3:
            raise ValueError("candidate downstream probabilities must be seed x sample x class")
        expected = candidate_probabilities.shape
        if baseline_probabilities.shape != expected:
            raise ValueError("baseline downstream probabilities do not align")
        if (
            random_probabilities.ndim != 4
            or random_probabilities.shape[0] != expected[0]
            or random_probabilities.shape[2:] != expected[1:]
        ):
            raise ValueError("matched-random downstream probabilities do not align")
        if reference.shape != (expected[1],) or groups.shape != reference.shape:
            raise ValueError("downstream references and groups do not align")
        for values in (candidate_probabilities, baseline_probabilities, random_probabilities):
            if not np.isfinite(values).all():
                raise RuntimeError("downstream probabilities contain non-finite values")
        unique_groups = tuple(sorted(set(str(value) for value in groups)))
        candidate_confusions = np.stack(
            [
                _confusions_by_group(reference, probabilities, groups, unique_groups)
                for probabilities in candidate_probabilities
            ],
            axis=0,
        )
        baseline_confusions = np.stack(
            [
                _confusions_by_group(reference, probabilities, groups, unique_groups)
                for probabilities in baseline_probabilities
            ],
            axis=0,
        )
        random_confusions = np.stack(
            [
                np.stack(
                    [
                        _confusions_by_group(reference, probabilities, groups, unique_groups)
                        for probabilities in seed_probabilities
                    ],
                    axis=0,
                )
                for seed_probabilities in random_probabilities
            ],
            axis=0,
        )

        def complete_macro(values: NDArray[np.int64]) -> float:
            return macro_f1_from_confusion(values.sum(axis=0))

        candidate_by_seed = np.asarray(
            [complete_macro(values) for values in candidate_confusions], dtype=np.float64
        )
        baseline_by_seed = np.asarray(
            [complete_macro(values) for values in baseline_confusions], dtype=np.float64
        )
        random_by_seed = np.asarray(
            [
                np.mean([complete_macro(values) for values in seed_values])
                for seed_values in random_confusions
            ],
            dtype=np.float64,
        )
        candidate_minus_baseline_by_seed = candidate_by_seed - baseline_by_seed
        candidate_minus_random_by_seed = candidate_by_seed - random_by_seed

        rng = np.random.default_rng(seed)
        macro_baseline_samples: list[float] = []
        macro_random_samples: list[float] = []
        class_samples: list[list[float]] = [[] for _ in CLASS_ORDER]
        group_count = len(unique_groups)
        for _ in range(iterations):
            sampled = rng.integers(0, group_count, size=group_count)
            baseline_seed_differences: list[float] = []
            random_seed_differences: list[float] = []
            class_seed_differences: list[list[float]] = [[] for _ in CLASS_ORDER]
            for seed_index in range(candidate_confusions.shape[0]):
                candidate_matrix = candidate_confusions[seed_index, sampled].sum(axis=0)
                baseline_matrix = baseline_confusions[seed_index, sampled].sum(axis=0)
                random_matrices = [
                    values[sampled].sum(axis=0) for values in random_confusions[seed_index]
                ]
                candidate_value = macro_f1_from_confusion(candidate_matrix)
                baseline_seed_differences.append(
                    candidate_value - macro_f1_from_confusion(baseline_matrix)
                )
                random_seed_differences.append(
                    candidate_value
                    - float(
                        np.mean([macro_f1_from_confusion(matrix) for matrix in random_matrices])
                    )
                )
                candidate_recall = per_class_recall_from_confusion(candidate_matrix)
                baseline_recall = per_class_recall_from_confusion(baseline_matrix)
                for class_index, difference in enumerate(candidate_recall - baseline_recall):
                    if np.isfinite(difference):
                        class_seed_differences[class_index].append(float(difference))
            macro_baseline_samples.append(float(np.mean(baseline_seed_differences)))
            macro_random_samples.append(float(np.mean(random_seed_differences)))
            for class_index, class_values in enumerate(class_seed_differences):
                if class_values:
                    class_samples[class_index].append(float(np.mean(class_values)))

        candidate_metrics = [
            classification_metrics(
                reference, probabilities, class_order=tuple(range(len(CLASS_ORDER)))
            )
            for probabilities in candidate_probabilities
        ]
        baseline_metrics = [
            classification_metrics(
                reference, probabilities, class_order=tuple(range(len(CLASS_ORDER)))
            )
            for probabilities in baseline_probabilities
        ]
        point_recall_differences = np.mean(
            np.stack(
                [
                    np.asarray(candidate.per_class_recall, dtype=np.float64)
                    - np.asarray(baseline.per_class_recall, dtype=np.float64)
                    for candidate, baseline in zip(candidate_metrics, baseline_metrics, strict=True)
                ],
                axis=0,
            ),
            axis=0,
        )
        return {
            "candidate_macro_f1": float(candidate_by_seed.mean()),
            "uncorrected_macro_f1": float(baseline_by_seed.mean()),
            "mean_matched_random_macro_f1": float(random_by_seed.mean()),
            "candidate_minus_uncorrected_macro_f1": float(candidate_minus_baseline_by_seed.mean()),
            "candidate_minus_uncorrected_interval_95": list(_interval(macro_baseline_samples)),
            "candidate_minus_matched_random_macro_f1": float(candidate_minus_random_by_seed.mean()),
            "candidate_minus_matched_random_interval_95": list(_interval(macro_random_samples)),
            "candidate_minus_uncorrected_by_corruption_seed": (
                candidate_minus_baseline_by_seed.tolist()
            ),
            "candidate_minus_matched_random_by_corruption_seed": (
                candidate_minus_random_by_seed.tolist()
            ),
            "candidate_minus_uncorrected_recall": point_recall_differences.tolist(),
            "candidate_minus_uncorrected_recall_intervals_95": [
                list(_interval(values)) if values else None for values in class_samples
            ],
            "candidate_accuracy": float(
                np.mean([metrics.accuracy for metrics in candidate_metrics])
            ),
            "uncorrected_accuracy": float(
                np.mean([metrics.accuracy for metrics in baseline_metrics])
            ),
            "matched_random_repetitions": random_probabilities.shape[1],
            "bootstrap_iterations": iterations,
        }

    def _result_with_gates(
        self,
        candidate: AutoresearchCandidate,
        *,
        stage: TrialStage,
        retrieval: Mapping[str, Any],
        downstream: Mapping[str, Any] | None,
        elapsed_seconds: float,
        queue_evidence: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        margin = float(self.config["adoption_guard"]["minimum_per_class_recall_effect"])
        retrieval_point = float(retrieval["candidate_minus_matched_random_precision"])
        retrieval_interval = tuple(float(value) for value in retrieval["interval_95"])
        if downstream is None:
            gates = {
                "retrieval_lower_bound_gt_zero_vs_exact_matched_random": (
                    retrieval_interval[0] > 0.0
                )
            }
            objective = retrieval_interval[0]
            keep = (
                gates["retrieval_lower_bound_gt_zero_vs_exact_matched_random"]
                or retrieval_point > 0.0
            )
        else:
            baseline_point = float(downstream["candidate_minus_uncorrected_macro_f1"])
            random_point = float(downstream["candidate_minus_matched_random_macro_f1"])
            baseline_interval = tuple(
                float(value) for value in downstream["candidate_minus_uncorrected_interval_95"]
            )
            random_interval = tuple(
                float(value) for value in downstream["candidate_minus_matched_random_interval_95"]
            )
            recall_points = tuple(
                float(value) for value in downstream["candidate_minus_uncorrected_recall"]
            )
            recall_intervals = tuple(downstream["candidate_minus_uncorrected_recall_intervals_95"])
            seed_baseline = tuple(
                float(value)
                for value in downstream["candidate_minus_uncorrected_by_corruption_seed"]
            )
            seed_random = tuple(
                float(value)
                for value in downstream["candidate_minus_matched_random_by_corruption_seed"]
            )
            full = stage in {"full_nested", "lockbox"}
            gates = {
                "retrieval_lower_bound_gt_zero_vs_exact_matched_random": (
                    retrieval_interval[0] > 0.0 if full else retrieval_point > 0.0
                ),
                "downstream_lower_bound_gt_zero_vs_uncorrected": (
                    baseline_interval[0] > 0.0 if full else baseline_point > 0.0
                ),
                "downstream_lower_bound_gt_zero_vs_exact_matched_random": (
                    random_interval[0] > 0.0 if full else random_point > 0.0
                ),
                "every_important_class_recall_lower_bound_gte_minus_0_01": (
                    all(
                        interval is not None and float(interval[0]) >= margin
                        for interval in recall_intervals
                    )
                    if full
                    else all(value >= margin for value in recall_points)
                ),
                "direction_consistent_across_corruption_seeds": all(
                    value > 0.0 for value in (*seed_baseline, *seed_random)
                ),
            }
            objective = (
                min(baseline_interval[0], random_interval[0])
                if full
                else min(baseline_point, random_point)
            )
            keep = all(gates.values())
        return {
            "schema_version": 1,
            "study_id": str(self.config["study_id"]),
            "stage": stage,
            "candidate": candidate.as_dict(),
            "candidate_sha256": candidate.candidate_sha256,
            "config_sha256": self.config_sha256,
            "partition_sha256": self.partition.partition_sha256,
            "status": "keep" if keep else "discard",
            "objective": float(objective),
            "success_gates": gates,
            "all_success_gates_pass": all(gates.values()),
            "retrieval": dict(retrieval),
            "downstream": dict(downstream) if downstream is not None else None,
            "queue_evidence": list(queue_evidence),
            "elapsed_seconds": float(elapsed_seconds),
            "source_annotations_modified": False,
            "final_external_test_used": False,
            "natural_error_detection_evaluated": False,
        }

    def evaluate_ranking(self, candidate: AutoresearchCandidate) -> dict[str, object]:
        """Cheap first-stage ranking screen on discovery patients only."""

        started = time.perf_counter()
        candidate.validate(self.config, self.feature_views)
        seed = int(self.config["controlled_corruption"]["discovery_seeds"][0])
        evidence = self._oof_audit_evidence(
            self.partition.discovery_indices,
            seed=seed,
            feature_view=candidate.feature_view,
            audit_l2=candidate.audit_l2,
            audit_class_weight_balanced=candidate.audit_class_weight_balanced,
            inner_folds=int(self.config["partition"]["audit_inner_folds"]),
        )
        risk = self._risk(evidence, candidate)
        selected, queue_evidence = self._select(evidence, risk, candidate)
        repetitions = int(self.config["controls"]["matched_random_repetitions_screen"])
        random_indices = self._matched_random_indices(
            evidence,
            selected,
            repetitions=repetitions,
            seed=seed,
            fold_id=0,
        )
        unique_groups = tuple(
            sorted(set(self._groups[index] for index in self.partition.discovery_indices))
        )
        group_lookup = {group: index for index, group in enumerate(unique_groups)}
        candidate_counts = np.zeros((1, len(unique_groups), 2), dtype=np.int64)
        random_counts = np.zeros((1, repetitions, len(unique_groups), 2), dtype=np.int64)
        for local_index in selected:
            group_column = group_lookup[
                self._groups[int(evidence.full_train_indices[int(local_index)])]
            ]
            candidate_counts[0, group_column, 0] += 1
            candidate_counts[0, group_column, 1] += int(evidence.injected[int(local_index)])
        for repeat, indices in enumerate(random_indices):
            for local_index in indices:
                group_column = group_lookup[
                    self._groups[int(evidence.full_train_indices[int(local_index)])]
                ]
                random_counts[0, repeat, group_column, 0] += 1
                random_counts[0, repeat, group_column, 1] += int(
                    evidence.injected[int(local_index)]
                )
        retrieval = self._retrieval_summary(
            candidate_counts,
            random_counts,
            iterations=int(self.config["controls"]["bootstrap_iterations_screen"]),
            seed=int(self.config["controls"]["bootstrap_seed"]),
        )
        retrieval["average_precision"] = average_precision(evidence.injected, risk)
        retrieval["injected_prevalence"] = float(evidence.injected.mean())
        return self._result_with_gates(
            candidate,
            stage="ranking_screen",
            retrieval=retrieval,
            downstream=None,
            elapsed_seconds=time.perf_counter() - started,
            queue_evidence=(queue_evidence,),
        )

    def _evaluate_nested(
        self,
        candidate: AutoresearchCandidate,
        *,
        stage: Literal["downstream_screen", "full_nested"],
    ) -> dict[str, object]:
        started = time.perf_counter()
        candidate.validate(self.config, self.feature_views)
        halving = self.config["successive_halving"]
        all_seeds = tuple(
            int(value) for value in self.config["controlled_corruption"]["discovery_seeds"]
        )
        seed_count = (
            int(halving["screen_corruption_seeds"])
            if stage == "downstream_screen"
            else int(halving["full_corruption_seeds"])
        )
        seeds = all_seeds[:seed_count]
        outer_folds = self.partition.discovery_outer_folds
        fold_count = (
            int(halving["screen_outer_folds"]) if stage == "downstream_screen" else len(outer_folds)
        )
        selected_folds = outer_folds[:fold_count]
        evaluated_indices = np.unique(
            np.concatenate([validation for _, validation in selected_folds])
        ).astype(np.int64)
        evaluated_lookup = {
            int(index): position for position, index in enumerate(evaluated_indices)
        }
        evaluated_groups = np.asarray(
            [self._groups[index] for index in evaluated_indices], dtype=np.str_
        )
        unique_retrieval_groups = tuple(
            sorted(set(self._groups[index] for index in self.partition.discovery_indices))
        )
        retrieval_lookup = {group: index for index, group in enumerate(unique_retrieval_groups)}
        repetitions = int(
            self.config["controls"][
                "matched_random_repetitions_screen"
                if stage == "downstream_screen"
                else "matched_random_repetitions_full"
            ]
        )
        bootstrap_iterations = int(
            self.config["controls"][
                "bootstrap_iterations_screen"
                if stage == "downstream_screen"
                else "bootstrap_iterations_full"
            ]
        )
        candidate_probabilities = np.full(
            (len(seeds), len(evaluated_indices), len(CLASS_ORDER)), np.nan, dtype=np.float64
        )
        baseline_probabilities = np.full_like(candidate_probabilities, np.nan)
        random_probabilities = np.full(
            (
                len(seeds),
                repetitions,
                len(evaluated_indices),
                len(CLASS_ORDER),
            ),
            np.nan,
            dtype=np.float64,
        )
        candidate_counts = np.zeros((len(seeds), len(unique_retrieval_groups), 2), dtype=np.int64)
        random_counts = np.zeros(
            (len(seeds), repetitions, len(unique_retrieval_groups), 2), dtype=np.int64
        )
        queue_records: list[Mapping[str, object]] = []
        restored_counts: list[int] = []
        random_restored_counts: list[int] = []
        features = self.feature_views[candidate.feature_view]
        for seed_index, corruption_seed in enumerate(seeds):
            for fold_id, (train_indices, validation_indices) in enumerate(selected_folds):
                evidence = self._oof_audit_evidence(
                    train_indices,
                    seed=corruption_seed,
                    feature_view=candidate.feature_view,
                    audit_l2=candidate.audit_l2,
                    audit_class_weight_balanced=candidate.audit_class_weight_balanced,
                    inner_folds=int(self.config["partition"]["audit_inner_folds"]),
                )
                risk = self._risk(evidence, candidate)
                selected, queue_evidence = self._select(evidence, risk, candidate)
                queue_records.append(
                    {
                        **queue_evidence,
                        "corruption_seed": corruption_seed,
                        "outer_fold_id": fold_id,
                    }
                )
                matched = self._matched_random_indices(
                    evidence,
                    selected,
                    repetitions=repetitions,
                    seed=corruption_seed,
                    fold_id=fold_id,
                )
                positions = np.asarray(
                    [evaluated_lookup[int(index)] for index in validation_indices],
                    dtype=np.int64,
                )
                baseline_probabilities[seed_index, positions] = self._baseline_probabilities(
                    evidence,
                    validation_indices,
                    candidate,
                    seed=corruption_seed,
                    fold_id=fold_id,
                )
                labels, weights, restored_count = self._derive_intervention(
                    evidence.observed,
                    evidence.reference,
                    evidence.injected,
                    selected,
                    candidate.intervention,
                )
                restored_counts.append(restored_count)
                candidate_probabilities[seed_index, positions] = _fit_predict_task(
                    features[train_indices],
                    labels,
                    features[validation_indices],
                    l2=candidate.downstream_l2,
                    class_weight_balanced=candidate.downstream_class_weight_balanced,
                    sample_weights=weights,
                )
                for repeat, random_selected in enumerate(matched):
                    random_labels, random_weights, random_restored = self._derive_intervention(
                        evidence.observed,
                        evidence.reference,
                        evidence.injected,
                        random_selected,
                        candidate.intervention,
                    )
                    random_restored_counts.append(random_restored)
                    random_probabilities[seed_index, repeat, positions] = _fit_predict_task(
                        features[train_indices],
                        random_labels,
                        features[validation_indices],
                        l2=candidate.downstream_l2,
                        class_weight_balanced=candidate.downstream_class_weight_balanced,
                        sample_weights=random_weights,
                    )

                for local_index in selected:
                    group_column = retrieval_lookup[
                        self._groups[int(train_indices[int(local_index)])]
                    ]
                    candidate_counts[seed_index, group_column, 0] += 1
                    candidate_counts[seed_index, group_column, 1] += int(
                        evidence.injected[int(local_index)]
                    )
                for repeat, random_selected in enumerate(matched):
                    for local_index in random_selected:
                        group_column = retrieval_lookup[
                            self._groups[int(train_indices[int(local_index)])]
                        ]
                        random_counts[seed_index, repeat, group_column, 0] += 1
                        random_counts[seed_index, repeat, group_column, 1] += int(
                            evidence.injected[int(local_index)]
                        )
        if not all(
            np.isfinite(values).all()
            for values in (
                candidate_probabilities,
                baseline_probabilities,
                random_probabilities,
            )
        ):
            raise RuntimeError("nested autoresearch prediction coverage is incomplete")
        retrieval = self._retrieval_summary(
            candidate_counts,
            random_counts,
            iterations=bootstrap_iterations,
            seed=int(self.config["controls"]["bootstrap_seed"]),
        )
        downstream = self._downstream_summary(
            self._reference[evaluated_indices],
            evaluated_groups,
            candidate_probabilities,
            baseline_probabilities,
            random_probabilities,
            iterations=bootstrap_iterations,
            seed=int(self.config["controls"]["bootstrap_seed"]) + 1,
        )
        downstream["controlled_restored_count_mean"] = float(np.mean(restored_counts))
        downstream["matched_random_restored_count_mean"] = float(np.mean(random_restored_counts))
        downstream["evaluated_patient_groups"] = len(set(str(value) for value in evaluated_groups))
        downstream["corruption_seeds"] = list(seeds)
        return self._result_with_gates(
            candidate,
            stage=stage,
            retrieval=retrieval,
            downstream=downstream,
            elapsed_seconds=time.perf_counter() - started,
            queue_evidence=queue_records,
        )

    def evaluate_downstream_screen(self, candidate: AutoresearchCandidate) -> dict[str, object]:
        return self._evaluate_nested(candidate, stage="downstream_screen")

    def evaluate_full_nested(self, candidate: AutoresearchCandidate) -> dict[str, object]:
        return self._evaluate_nested(candidate, stage="full_nested")

    def evaluate_lockbox(
        self,
        candidate: AutoresearchCandidate,
        *,
        frozen_candidate_sha256: str,
        freeze_file_sha256: str,
    ) -> dict[str, object]:
        """Evaluate one already-serialized candidate on the internal lockbox once."""

        if frozen_candidate_sha256 != candidate.candidate_sha256:
            raise ValueError("serialized candidate identity differs before lockbox evaluation")
        if len(freeze_file_sha256) != 64:
            raise ValueError("lockbox evaluation requires a SHA-256-pinned freeze file")
        started = time.perf_counter()
        candidate.validate(self.config, self.feature_views)
        seeds = tuple(
            int(value) for value in self.config["controlled_corruption"]["discovery_seeds"]
        )
        repetitions = int(self.config["controls"]["matched_random_repetitions_lockbox"])
        iterations = int(self.config["controls"]["bootstrap_iterations_full"])
        train_indices = self.partition.discovery_indices
        validation_indices = self.partition.lockbox_indices
        lockbox_groups = np.asarray(
            [self._groups[index] for index in validation_indices], dtype=np.str_
        )
        retrieval_groups = tuple(sorted(set(self._groups[index] for index in train_indices)))
        retrieval_lookup = {group: index for index, group in enumerate(retrieval_groups)}
        candidate_probabilities = np.full(
            (len(seeds), len(validation_indices), len(CLASS_ORDER)), np.nan, dtype=np.float64
        )
        baseline_probabilities = np.full_like(candidate_probabilities, np.nan)
        random_probabilities = np.full(
            (len(seeds), repetitions, len(validation_indices), len(CLASS_ORDER)),
            np.nan,
            dtype=np.float64,
        )
        candidate_counts = np.zeros((len(seeds), len(retrieval_groups), 2), dtype=np.int64)
        random_counts = np.zeros(
            (len(seeds), repetitions, len(retrieval_groups), 2), dtype=np.int64
        )
        queue_records: list[Mapping[str, object]] = []
        restored_counts: list[int] = []
        random_restored_counts: list[int] = []
        features = self.feature_views[candidate.feature_view]
        for seed_index, corruption_seed in enumerate(seeds):
            evidence = self._oof_audit_evidence(
                train_indices,
                seed=corruption_seed,
                feature_view=candidate.feature_view,
                audit_l2=candidate.audit_l2,
                audit_class_weight_balanced=candidate.audit_class_weight_balanced,
                inner_folds=int(self.config["partition"]["audit_inner_folds"]),
            )
            risk = self._risk(evidence, candidate)
            selected, queue_evidence = self._select(evidence, risk, candidate)
            queue_records.append(
                {
                    **queue_evidence,
                    "corruption_seed": corruption_seed,
                    "evaluation_role": "internal_lockbox",
                }
            )
            matched = self._matched_random_indices(
                evidence,
                selected,
                repetitions=repetitions,
                seed=corruption_seed,
                fold_id=999,
            )
            baseline_probabilities[seed_index] = self._baseline_probabilities(
                evidence,
                validation_indices,
                candidate,
                seed=corruption_seed,
                fold_id=999,
            )
            labels, weights, restored_count = self._derive_intervention(
                evidence.observed,
                evidence.reference,
                evidence.injected,
                selected,
                candidate.intervention,
            )
            restored_counts.append(restored_count)
            candidate_probabilities[seed_index] = _fit_predict_task(
                features[train_indices],
                labels,
                features[validation_indices],
                l2=candidate.downstream_l2,
                class_weight_balanced=candidate.downstream_class_weight_balanced,
                sample_weights=weights,
            )
            for repeat, random_selected in enumerate(matched):
                random_labels, random_weights, random_restored = self._derive_intervention(
                    evidence.observed,
                    evidence.reference,
                    evidence.injected,
                    random_selected,
                    candidate.intervention,
                )
                random_restored_counts.append(random_restored)
                random_probabilities[seed_index, repeat] = _fit_predict_task(
                    features[train_indices],
                    random_labels,
                    features[validation_indices],
                    l2=candidate.downstream_l2,
                    class_weight_balanced=candidate.downstream_class_weight_balanced,
                    sample_weights=random_weights,
                )
            for local_index in selected:
                group_column = retrieval_lookup[self._groups[int(train_indices[int(local_index)])]]
                candidate_counts[seed_index, group_column, 0] += 1
                candidate_counts[seed_index, group_column, 1] += int(
                    evidence.injected[int(local_index)]
                )
            for repeat, random_selected in enumerate(matched):
                for local_index in random_selected:
                    group_column = retrieval_lookup[
                        self._groups[int(train_indices[int(local_index)])]
                    ]
                    random_counts[seed_index, repeat, group_column, 0] += 1
                    random_counts[seed_index, repeat, group_column, 1] += int(
                        evidence.injected[int(local_index)]
                    )
        retrieval = self._retrieval_summary(
            candidate_counts,
            random_counts,
            iterations=iterations,
            seed=int(self.config["controls"]["bootstrap_seed"]),
        )
        downstream = self._downstream_summary(
            self._reference[validation_indices],
            lockbox_groups,
            candidate_probabilities,
            baseline_probabilities,
            random_probabilities,
            iterations=iterations,
            seed=int(self.config["controls"]["bootstrap_seed"]) + 1,
        )
        downstream["controlled_restored_count_mean"] = float(np.mean(restored_counts))
        downstream["matched_random_restored_count_mean"] = float(np.mean(random_restored_counts))
        downstream["evaluated_patient_groups"] = len(self.partition.lockbox_groups)
        downstream["corruption_seeds"] = list(seeds)
        result = self._result_with_gates(
            candidate,
            stage="lockbox",
            retrieval=retrieval,
            downstream=downstream,
            elapsed_seconds=time.perf_counter() - started,
            queue_evidence=queue_records,
        )
        result["frozen_candidate_sha256"] = frozen_candidate_sha256
        result["freeze_file_sha256"] = freeze_file_sha256
        result["internal_lockbox_is_external_confirmation"] = False
        return result


def _declared_risk_specs(config: Mapping[str, Any]) -> tuple[tuple[str, int, float], ...]:
    risk_config = config["candidate_space"]["risk_methods"]
    output: list[tuple[str, int, float]] = []
    for method in risk_config["probability"]:
        output.append((str(method), 7, 0.5))
    for neighbour_k in risk_config["neighbour"]["k"]:
        output.append(("nearest_neighbour_disagreement", int(neighbour_k), 0.5))
    for neighbour_k in risk_config["hybrid"]["neighbour_k"]:
        for weight in risk_config["hybrid"]["self_confidence_weights"]:
            output.append(("fixed_hybrid", int(neighbour_k), float(weight)))
    return tuple(output)


def _unique_candidates(
    values: Sequence[AutoresearchCandidate], *, maximum: int
) -> tuple[AutoresearchCandidate, ...]:
    output: list[AutoresearchCandidate] = []
    seen: set[str] = set()
    for candidate in values:
        if candidate.candidate_sha256 in seen:
            continue
        seen.add(candidate.candidate_sha256)
        output.append(candidate)
        if len(output) >= maximum:
            break
    return tuple(output)


def generate_ranking_candidates(
    config: Mapping[str, Any], feature_view_names: Sequence[str]
) -> tuple[AutoresearchCandidate, ...]:
    """Generate a deterministic broad screen plus one-factor baseline ablations."""

    maximum = int(config["successive_halving"]["ranking_screen_max_trials"])
    space = config["candidate_space"]
    baseline = AutoresearchCandidate()
    candidates: list[AutoresearchCandidate] = [baseline]

    def replace(**changes: object) -> AutoresearchCandidate:
        value = baseline.as_dict()
        value.update(changes)
        return AutoresearchCandidate.from_mapping(value)

    for feature_view in feature_view_names:
        candidates.append(replace(feature_view=str(feature_view)))
    for l2 in space["audit_l2"]:
        candidates.append(replace(audit_l2=float(l2)))
    for balanced in space["audit_class_weight_balanced"]:
        candidates.append(replace(audit_class_weight_balanced=bool(balanced)))
    for method, neighbour_k, weight in _declared_risk_specs(config):
        candidates.append(
            replace(
                risk_method=method,
                neighbour_k=neighbour_k,
                hybrid_self_confidence_weight=weight,
            )
        )
    for queue_preset in space["queue_presets"]:
        candidates.append(replace(queue_preset=str(queue_preset)))
    for review_budget in space["review_budgets"]:
        candidates.append(replace(review_budget=float(review_budget)))

    rng = np.random.default_rng(int(config["partition"]["seed"]) + 200)
    features = tuple(str(value) for value in feature_view_names)
    audit_l2 = tuple(float(value) for value in space["audit_l2"])
    audit_balanced = tuple(bool(value) for value in space["audit_class_weight_balanced"])
    risk_specs = _declared_risk_specs(config)
    queues = tuple(str(value) for value in space["queue_presets"])
    budgets = tuple(float(value) for value in space["review_budgets"])
    attempts = 0
    while len({candidate.candidate_sha256 for candidate in candidates}) < maximum:
        attempts += 1
        if attempts > maximum * 100:
            break
        method, neighbour_k, weight = risk_specs[int(rng.integers(0, len(risk_specs)))]
        candidates.append(
            AutoresearchCandidate(
                feature_view=features[int(rng.integers(0, len(features)))],
                audit_l2=audit_l2[int(rng.integers(0, len(audit_l2)))],
                audit_class_weight_balanced=audit_balanced[
                    int(rng.integers(0, len(audit_balanced)))
                ],
                risk_method=method,
                neighbour_k=neighbour_k,
                hybrid_self_confidence_weight=weight,
                queue_preset=queues[int(rng.integers(0, len(queues)))],
                review_budget=budgets[int(rng.integers(0, len(budgets)))],
            )
        )
    return _unique_candidates(candidates, maximum=maximum)


def select_ranking_finalists(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[AutoresearchCandidate, ...]:
    maximum = int(config["successive_halving"]["ranking_screen_max_finalists"])
    successful = [
        record
        for record in records
        if record.get("stage") == "ranking_screen" and record.get("status") != "crash"
    ]
    successful.sort(
        key=lambda record: (
            -float(record.get("objective", float("-inf"))),
            -float((record.get("retrieval") or {}).get("average_precision", float("-inf"))),
            str(record.get("candidate_sha256", "")),
        )
    )
    selected: list[AutoresearchCandidate] = [AutoresearchCandidate()]
    seen_families: set[tuple[str, float]] = set()
    for record in successful:
        candidate = AutoresearchCandidate.from_mapping(record["candidate"])
        family = (candidate.risk_method, candidate.review_budget)
        if family in seen_families:
            continue
        selected.append(candidate)
        seen_families.add(family)
        if len(_unique_candidates(selected, maximum=maximum)) >= maximum:
            return _unique_candidates(selected, maximum=maximum)
    selected.extend(
        AutoresearchCandidate.from_mapping(record["candidate"]) for record in successful
    )
    return _unique_candidates(selected, maximum=maximum)


def generate_downstream_candidates(
    ranking_finalists: Sequence[AutoresearchCandidate], config: Mapping[str, Any]
) -> tuple[AutoresearchCandidate, ...]:
    maximum = int(config["successive_halving"]["downstream_screen_max_trials"])
    space = config["candidate_space"]
    candidates: list[AutoresearchCandidate] = [AutoresearchCandidate()]
    candidates.extend(ranking_finalists)
    for finalist in ranking_finalists:
        base = finalist.as_dict()
        for intervention in space["interventions"]:
            value = dict(base)
            value["intervention"] = str(intervention)
            candidates.append(AutoresearchCandidate.from_mapping(value))
        for l2 in space["downstream_l2"]:
            value = dict(base)
            value["downstream_l2"] = float(l2)
            candidates.append(AutoresearchCandidate.from_mapping(value))
        for balanced in space["downstream_class_weight_balanced"]:
            value = dict(base)
            value["downstream_class_weight_balanced"] = bool(balanced)
            candidates.append(AutoresearchCandidate.from_mapping(value))
    rng = np.random.default_rng(int(config["partition"]["seed"]) + 300)
    interventions = tuple(str(value) for value in space["interventions"])
    downstream_l2 = tuple(float(value) for value in space["downstream_l2"])
    downstream_balanced = tuple(bool(value) for value in space["downstream_class_weight_balanced"])
    attempts = 0
    while len({candidate.candidate_sha256 for candidate in candidates}) < maximum:
        attempts += 1
        if attempts > maximum * 100:
            break
        finalist = ranking_finalists[int(rng.integers(0, len(ranking_finalists)))]
        value = finalist.as_dict()
        value["intervention"] = interventions[int(rng.integers(0, len(interventions)))]
        value["downstream_l2"] = downstream_l2[int(rng.integers(0, len(downstream_l2)))]
        value["downstream_class_weight_balanced"] = downstream_balanced[
            int(rng.integers(0, len(downstream_balanced)))
        ]
        candidates.append(AutoresearchCandidate.from_mapping(value))
    return _unique_candidates(candidates, maximum=maximum)


def select_full_nested_finalists(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[AutoresearchCandidate, ...]:
    maximum = int(config["successive_halving"]["full_nested_max_finalists"])
    eligible = [
        record
        for record in records
        if record.get("stage") == "downstream_screen"
        and record.get("status") not in {"crash", "timeout"}
    ]
    eligible.sort(
        key=lambda record: (
            0 if record.get("all_success_gates_pass") else 1,
            -float(record.get("objective", float("-inf"))),
            -float(
                (record.get("downstream") or {}).get(
                    "candidate_minus_uncorrected_macro_f1", float("-inf")
                )
            ),
            str(record.get("candidate_sha256", "")),
        )
    )
    candidates = [AutoresearchCandidate()]
    candidates.extend(
        AutoresearchCandidate.from_mapping(record["candidate"]) for record in eligible
    )
    return _unique_candidates(candidates, maximum=maximum)


def _candidate_complexity(candidate: AutoresearchCandidate) -> int:
    baseline = AutoresearchCandidate()
    return sum(
        candidate.as_dict()[name] != baseline.as_dict()[name] for name in candidate.as_dict()
    )


def select_passing_winner(
    records: Sequence[Mapping[str, Any]],
) -> AutoresearchCandidate | None:
    passing = [
        record
        for record in records
        if record.get("stage") == "full_nested"
        and record.get("all_success_gates_pass") is True
        and record.get("status") == "keep"
    ]
    passing.sort(
        key=lambda record: (
            -float(record["objective"]),
            -float(record["downstream"]["candidate_minus_uncorrected_macro_f1"]),
            _candidate_complexity(AutoresearchCandidate.from_mapping(record["candidate"])),
            str(record["candidate_sha256"]),
        )
    )
    return AutoresearchCandidate.from_mapping(passing[0]["candidate"]) if passing else None


__all__ = [
    "AutoresearchCandidate",
    "AutoresearchEvaluator",
    "AutoresearchPartition",
    "build_autoresearch_feature_views",
    "build_autoresearch_partition",
    "generate_downstream_candidates",
    "generate_ranking_candidates",
    "load_autoresearch_config",
    "select_full_nested_finalists",
    "select_passing_winner",
    "select_ranking_finalists",
]
