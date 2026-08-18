"""Exact, deterministic controlled class-label corruption."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from histo_audit.data.manifest import NucleusRecord, validate_manifest

CANONICAL_MECHANISMS: tuple[str, ...] = (
    "symmetric_random_corruption",
    "confusion_targeted_corruption",
    "group_conditional_corruption",
    "instance_dependent_corruption",
)

_ALIASES = {
    "symmetric": "symmetric_random_corruption",
    "symmetric_random": "symmetric_random_corruption",
    "targeted": "confusion_targeted_corruption",
    "confusion_targeted": "confusion_targeted_corruption",
    "group": "group_conditional_corruption",
    "group_conditional": "group_conditional_corruption",
    "instance": "instance_dependent_corruption",
    "instance_dependent": "instance_dependent_corruption",
}

ROUNDING_POLICY = "round_half_up_floor_n_times_rate_plus_0_5"
CORRUPTION_CONFIG_SCHEMA_VERSION = 2


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible semantic payload canonically."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_sha256(description: str) -> str:
    """Hash a non-empty implementation/preprocessing provenance declaration."""

    if not description.strip():
        raise ValueError("semantic hash description must be non-empty")
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def array_artifact_sha256(array: NDArray[np.generic]) -> str:
    """Hash array dtype, shape, and C-order content without numeric coercion."""

    value = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class FeatureSpaceEvidence:
    """Cryptographic provenance for one concrete feature space."""

    representation_name: str
    feature_artifact_hash: str
    family: str
    implementation_hash: str
    weights_hash: str
    preprocessing_hash: str
    fitted_data_hash: str

    @classmethod
    def from_array(
        cls,
        features: NDArray[np.generic],
        *,
        representation_name: str,
        family: str,
        implementation_hash: str,
        weights_hash: str,
        preprocessing_hash: str,
        fitted_data_hash: str,
    ) -> FeatureSpaceEvidence:
        """Create evidence whose artifact hash is bound to the supplied array."""

        evidence = cls(
            representation_name=representation_name,
            feature_artifact_hash=array_artifact_sha256(features),
            family=family,
            implementation_hash=implementation_hash,
            weights_hash=weights_hash,
            preprocessing_hash=preprocessing_hash,
            fitted_data_hash=fitted_data_hash,
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        for field_name in ("representation_name", "family"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "feature_artifact_hash",
            "implementation_hash",
            "weights_hash",
            "preprocessing_hash",
            "fitted_data_hash",
        ):
            _validate_sha256(str(getattr(self, field_name)), field_name)

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def space_signature(self) -> tuple[str, ...]:
        """Identify the transformation itself, separate from display names."""

        return (
            self.family,
            self.implementation_hash,
            self.weights_hash,
            self.preprocessing_hash,
            self.fitted_data_hash,
        )


@dataclass(frozen=True, slots=True)
class FeatureIndependenceEvidence:
    """A frozen matrix decision bound to full generator/auditor provenance."""

    matrix_version: str
    matrix_decision: str
    matrix_reason: str
    generator: FeatureSpaceEvidence
    auditor: FeatureSpaceEvidence
    independence_matrix_hash: str

    @classmethod
    def create(
        cls,
        *,
        matrix_version: str,
        matrix_decision: str,
        matrix_reason: str,
        generator: FeatureSpaceEvidence,
        auditor: FeatureSpaceEvidence,
    ) -> FeatureIndependenceEvidence:
        """Freeze a matrix entry; only ``verified_independent`` can certify it."""

        provisional = cls(
            matrix_version=matrix_version,
            matrix_decision=matrix_decision,
            matrix_reason=matrix_reason,
            generator=generator,
            auditor=auditor,
            independence_matrix_hash="0" * 64,
        )
        digest = canonical_sha256(provisional.matrix_payload())
        evidence = replace(provisional, independence_matrix_hash=digest)
        evidence.validate()
        return evidence

    def matrix_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "matrix_version": self.matrix_version,
            "matrix_decision": self.matrix_decision,
            "matrix_reason": self.matrix_reason,
            "generator": self.generator.as_dict(),
            "auditor": self.auditor.as_dict(),
        }

    def validate(self) -> None:
        if not self.matrix_version.strip() or not self.matrix_reason.strip():
            raise ValueError("independence matrix version and reason must be non-empty")
        if self.matrix_decision not in {"verified_independent", "not_independent"}:
            raise ValueError("matrix_decision must be verified_independent or not_independent")
        self.generator.validate()
        self.auditor.validate()
        _validate_sha256(self.independence_matrix_hash, "independence_matrix_hash")
        expected = canonical_sha256(self.matrix_payload())
        if self.independence_matrix_hash != expected:
            raise ValueError("independence matrix hash does not match its canonical payload")

    def as_dict(self) -> dict[str, object]:
        return {
            **self.matrix_payload(),
            "independence_matrix_hash": self.independence_matrix_hash,
        }


@dataclass(frozen=True, slots=True)
class CorruptionResult:
    """Immutable-by-convention controlled-corruption outcome and provenance."""

    pre_corruption_labels: NDArray[np.int64]
    observed_labels: NDArray[np.int64]
    is_injected_corruption: NDArray[np.bool_]
    selected_indices: NDArray[np.int64]
    original_class: NDArray[np.int64]
    replacement_class: NDArray[np.int64]
    mechanism: str
    requested_rate: float
    exact_count: int
    corruption_seed: int
    dataset_seed: int | None
    generator_representation: str | None
    auditor_representation: str | None
    feature_space_independent: bool | None
    independence_status: str
    independence_reason: str
    independence_evidence: FeatureIndependenceEvidence | None
    circularity_risk: bool
    configuration_hash: str
    configuration_payload_json: str
    upstream_manifest_hash: str | None
    timestamp_utc: str

    def validate(self, *, n_classes: int) -> None:
        """Validate exact count, label preservation, and no-self-replacement rules."""

        n = self.pre_corruption_labels.shape[0]
        for array in (
            self.observed_labels,
            self.is_injected_corruption,
            self.original_class,
            self.replacement_class,
        ):
            if array.shape != (n,):
                raise ValueError("corruption arrays must be one-dimensional and aligned")
        if self.exact_count != int(self.is_injected_corruption.sum()):
            raise ValueError("injected-corruption mask does not match exact count")
        if self.selected_indices.shape != (self.exact_count,):
            raise ValueError("selected index count mismatch")
        if np.any(self.pre_corruption_labels < 0) or np.any(
            self.pre_corruption_labels >= n_classes
        ):
            raise ValueError("pre-corruption labels outside class range")
        changed = self.observed_labels != self.pre_corruption_labels
        if not np.array_equal(changed, self.is_injected_corruption):
            raise ValueError("only flagged rows may change observed labels")
        if np.any(self.observed_labels < 0) or np.any(self.observed_labels >= n_classes):
            raise ValueError("observed labels outside class range")
        if np.any(
            self.replacement_class[self.is_injected_corruption]
            == self.original_class[self.is_injected_corruption]
        ):
            raise ValueError("a corruption replaced a label with itself")
        if (
            hashlib.sha256(self.configuration_payload_json.encode("utf-8")).hexdigest()
            != self.configuration_hash
        ):
            raise ValueError("configuration hash does not match canonical payload")
        if self.feature_space_independent is True:
            if self.independence_evidence is None or self.circularity_risk:
                raise ValueError("independence certification lacks valid evidence")
            self.independence_evidence.validate()
        if self.mechanism == "instance_dependent_corruption" and (
            self.feature_space_independent is not True and not self.circularity_risk
        ):
            raise ValueError("unverified instance-dependent corruption must carry circularity risk")

    def manifest_rows(
        self, sample_ids: Sequence[str], group_ids: Sequence[str]
    ) -> tuple[dict[str, object], ...]:
        """Return one immutable-manifest-style metadata row per sample."""

        if len(sample_ids) != len(self.pre_corruption_labels) or len(group_ids) != len(sample_ids):
            raise ValueError("sample and group identifiers must align with labels")
        rows: list[dict[str, object]] = []
        for index, (sample_id, group_id) in enumerate(zip(sample_ids, group_ids, strict=True)):
            injected = bool(self.is_injected_corruption[index])
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "group_id": str(group_id),
                    "pre_corruption_label": int(self.pre_corruption_labels[index]),
                    "observed_label": int(self.observed_labels[index]),
                    "is_injected_corruption": injected,
                    "corruption_type": self.mechanism if injected else "none",
                    "original_class": int(self.original_class[index]),
                    "replacement_class": (int(self.replacement_class[index]) if injected else None),
                    "corruption_rate": self.requested_rate,
                    "corruption_seed": self.corruption_seed,
                    "corruption_representation": self.generator_representation,
                    "auditor_representation": self.auditor_representation,
                    "feature_space_independent": self.feature_space_independent,
                    "independence_status": self.independence_status,
                    "independence_reason": self.independence_reason,
                    "independence_evidence": (
                        self.independence_evidence.as_dict()
                        if self.independence_evidence is not None
                        else None
                    ),
                    "circularity_risk": self.circularity_risk,
                    "dataset_seed": self.dataset_seed,
                    "upstream_manifest_hash": self.upstream_manifest_hash,
                    "configuration_hash": self.configuration_hash,
                    "timestamp_utc": self.timestamp_utc,
                }
            )
        return tuple(rows)


def normalise_rate(rate: float | int) -> float:
    """Accept a fraction or percentage and return a fraction in ``[0, 1]``."""

    value = float(rate)
    if value > 1.0:
        value /= 100.0
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("corruption rate must be a fraction or percentage in [0, 100]")
    return value


def exact_corruption_count(n_samples: int, rate: float | int) -> int:
    """Use documented round-half-up counting: ``floor(n * rate + 0.5)``."""

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    fraction = normalise_rate(rate)
    return min(n_samples, int(np.floor(n_samples * fraction + 0.5)))


def _canonical_mechanism(mechanism: str) -> str:
    canonical = _ALIASES.get(mechanism, mechanism)
    if canonical not in CANONICAL_MECHANISMS:
        raise ValueError(
            f"unknown corruption mechanism {mechanism!r}; expected {CANONICAL_MECHANISMS}"
        )
    return canonical


def _different_uniform_replacements(
    labels: NDArray[np.int64], indices: NDArray[np.int64], n_classes: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    draws = rng.integers(0, n_classes - 1, size=len(indices), dtype=np.int64)
    originals = labels[indices]
    return draws + (draws >= originals)


def _transition_replacements(
    labels: NDArray[np.int64],
    indices: NDArray[np.int64],
    transition_matrix: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    n_classes = matrix.shape[0]
    replacements = np.empty(len(indices), dtype=np.int64)
    for output_index, source_index in enumerate(indices):
        original = int(labels[source_index])
        replacements[output_index] = int(rng.choice(n_classes, p=matrix[original]))
    return replacements


def _effective_transition_matrix(
    n_classes: int, transition_matrix: NDArray[np.generic] | None
) -> tuple[NDArray[np.float64], str]:
    if transition_matrix is None:
        matrix = np.ones((n_classes, n_classes), dtype=np.float64)
        for class_index in range(n_classes):
            matrix[class_index, (class_index + 1) % n_classes] = 4.0
        source = "effective_default"
    else:
        matrix = np.asarray(transition_matrix, dtype=np.float64).copy()
        source = "provided"
    if matrix.shape != (n_classes, n_classes):
        raise ValueError("transition matrix must have shape (n_classes, n_classes)")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("transition matrix must contain finite non-negative weights")
    np.fill_diagonal(matrix, 0.0)
    for class_index in range(n_classes):
        row = matrix[class_index]
        if row.sum() <= 0.0:
            row = np.ones(n_classes, dtype=np.float64)
            row[class_index] = 0.0
        matrix[class_index] = row / row.sum()
    return matrix, source


def _effective_group_weights(
    groups: NDArray[np.str_], group_weights: Mapping[str, float] | None
) -> tuple[dict[str, float], str]:
    unique_groups = np.unique(groups)
    if group_weights is None:
        weights_by_group = {
            str(group): (2.0 if position % 3 == 0 else 0.5)
            for position, group in enumerate(unique_groups)
        }
        source = "effective_default"
    else:
        supplied = {str(key): float(value) for key, value in group_weights.items()}
        weights_by_group = {str(group): supplied.get(str(group), 1.0) for group in unique_groups}
        source = "provided"
    values = np.asarray(tuple(weights_by_group.values()), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("group weights must be finite and non-negative")
    return dict(sorted(weights_by_group.items())), source


def _weighted_group_selection(
    groups: NDArray[np.str_],
    count: int,
    effective_group_weights: Mapping[str, float],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    weights = np.asarray([effective_group_weights[str(group)] for group in groups])
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("group weights must be finite and non-negative")
    if np.count_nonzero(weights > 0) < count:
        raise ValueError("too few positive-weight samples for requested exact count")
    return np.sort(rng.choice(len(groups), size=count, replace=False, p=weights / weights.sum()))


def _instance_selection_and_replacements(
    labels: NDArray[np.int64],
    features: NDArray[np.generic],
    count: int,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("generator_features must have shape (n_samples, n_features)")
    if not np.isfinite(matrix).all():
        raise ValueError("generator features must be finite")
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardised = (matrix - mean) / scale
    centroids = np.empty((n_classes, matrix.shape[1]), dtype=np.float64)
    for class_index in range(n_classes):
        members = standardised[labels == class_index]
        if not len(members):
            raise ValueError(f"instance-dependent corruption needs class {class_index}")
        centroids[class_index] = members.mean(axis=0)
    distances = np.linalg.norm(standardised[:, None, :] - centroids[None, :, :], axis=2)
    own_distance = distances[np.arange(len(labels)), labels]
    distances[np.arange(len(labels)), labels] = np.inf
    alternative = np.argmin(distances, axis=1).astype(np.int64)
    ambiguity_margin = distances[np.arange(len(labels)), alternative] - own_distance
    jitter = rng.uniform(0.0, 1e-12, size=len(labels))
    selected = np.argsort(ambiguity_margin + jitter, kind="stable")[:count].astype(np.int64)
    return np.sort(selected), alternative[np.sort(selected)]


def _evaluate_independence(
    evidence: FeatureIndependenceEvidence | None,
    *,
    generator_features: NDArray[np.generic],
    generator_representation: str | None,
    auditor_representation: str | None,
) -> tuple[bool | None, str, str, bool]:
    """Require hash-bound evidence; display-name inequality is never sufficient."""

    if evidence is None:
        return (
            None,
            "unverified",
            "No frozen feature-space independence evidence was supplied.",
            True,
        )
    evidence.validate()
    if (
        evidence.generator.representation_name != generator_representation
        or evidence.auditor.representation_name != auditor_representation
    ):
        return (
            None,
            "unverified",
            "Evidence representation identities do not match the configured generator/auditor.",
            True,
        )
    if evidence.generator.feature_artifact_hash != array_artifact_sha256(generator_features):
        return (
            None,
            "unverified",
            "Generator feature artifact content does not match its evidence hash.",
            True,
        )
    identical_space = (
        evidence.generator.feature_artifact_hash == evidence.auditor.feature_artifact_hash
        or evidence.generator.space_signature() == evidence.auditor.space_signature()
    )
    if identical_space:
        return (
            False,
            "circularity_risk",
            "Generator and auditor evidence identifies the same feature space.",
            True,
        )
    if evidence.matrix_decision != "verified_independent":
        return False, "circularity_risk", evidence.matrix_reason, True
    return True, "verified_independent", evidence.matrix_reason, False


def _sequence_digest(values: Sequence[object]) -> str:
    return canonical_sha256(list(values))


def _feature_input_metadata(
    generator_features: NDArray[np.generic] | None,
) -> dict[str, object]:
    if generator_features is None:
        return {"status": "not_supplied"}
    features = np.asarray(generator_features)
    return {
        "status": "supplied",
        "dtype": features.dtype.str,
        "shape": list(features.shape),
        "content_sha256": array_artifact_sha256(features),
    }


def apply_controlled_corruption(
    pre_corruption_labels: Sequence[int] | NDArray[np.integer],
    *,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    rate: float | int,
    mechanism: str = "symmetric_random_corruption",
    seed: int = 17,
    n_classes: int = 5,
    transition_matrix: NDArray[np.generic] | None = None,
    group_weights: Mapping[str, float] | None = None,
    generator_features: NDArray[np.generic] | None = None,
    generator_representation: str | None = None,
    auditor_representation: str | None = None,
    independence_evidence: FeatureIndependenceEvidence | None = None,
    upstream_manifest_hash: str | None = None,
    dataset_seed: int | None = None,
    timestamp_utc: str | None = None,
) -> CorruptionResult:
    """Apply one exact controlled corruption without mutating source labels."""

    labels = np.asarray(pre_corruption_labels, dtype=np.int64).copy()
    if labels.ndim != 1 or not len(labels):
        raise ValueError("pre_corruption_labels must be a non-empty vector")
    if len(sample_ids) != len(labels) or len(group_ids) != len(labels):
        raise ValueError("sample_ids and group_ids must align with labels")
    if len(set(str(value) for value in sample_ids)) != len(labels):
        raise ValueError("sample_ids must be unique")
    if n_classes < 2 or np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels must lie in the configured class range")
    canonical = _canonical_mechanism(mechanism)
    fraction = normalise_rate(rate)
    count = exact_corruption_count(len(labels), fraction)
    rng = np.random.default_rng(seed)
    groups = np.asarray(group_ids, dtype=np.str_)
    sample_id_values = tuple(str(value) for value in sample_ids)
    group_id_values = tuple(str(value) for value in group_ids)
    if upstream_manifest_hash is not None:
        _validate_sha256(upstream_manifest_hash, "upstream_manifest_hash")

    if canonical == "confusion_targeted_corruption":
        effective_transition, transition_source = _effective_transition_matrix(
            n_classes, transition_matrix
        )
        transition_payload: dict[str, object] = {
            "status": "used",
            "source": transition_source,
            "effective_row_probabilities": effective_transition.tolist(),
        }
    else:
        effective_transition = np.empty((0, 0), dtype=np.float64)
        transition_payload = {
            "status": "not_applicable",
            "reason": "The selected corruption mechanism does not use a transition matrix.",
        }
    if canonical == "group_conditional_corruption":
        effective_group_weights, group_weight_source = _effective_group_weights(
            groups, group_weights
        )
        group_weight_payload: dict[str, object] = {
            "status": "used",
            "source": group_weight_source,
            "effective_weights": effective_group_weights,
        }
    else:
        effective_group_weights = {}
        group_weight_payload = {
            "status": "not_applicable",
            "reason": "The selected corruption mechanism does not use group weights.",
        }

    if canonical == "instance_dependent_corruption":
        if generator_features is None:
            raise ValueError("instance-dependent corruption requires generator_features")
        selected, replacements = _instance_selection_and_replacements(
            labels, generator_features, count, n_classes, rng
        )
    elif canonical == "group_conditional_corruption":
        selected = _weighted_group_selection(groups, count, effective_group_weights, rng)
        replacements = _different_uniform_replacements(labels, selected, n_classes, rng)
    else:
        selected = np.sort(rng.choice(len(labels), size=count, replace=False)).astype(np.int64)
        if canonical == "confusion_targeted_corruption":
            replacements = _transition_replacements(labels, selected, effective_transition, rng)
        else:
            replacements = _different_uniform_replacements(labels, selected, n_classes, rng)

    observed = labels.copy()
    observed[selected] = replacements
    injected = np.zeros(len(labels), dtype=bool)
    injected[selected] = True
    replacement_by_row = np.full(len(labels), -1, dtype=np.int64)
    replacement_by_row[selected] = replacements

    if canonical == "instance_dependent_corruption":
        if generator_features is None:  # guarded above; narrows the type for static checking.
            raise RuntimeError("generator features unexpectedly missing")
        independent, independence_status, independence_reason, circularity_risk = (
            _evaluate_independence(
                independence_evidence,
                generator_features=generator_features,
                generator_representation=generator_representation,
                auditor_representation=auditor_representation,
            )
        )
    else:
        independent = None
        independence_status = "not_applicable"
        independence_reason = (
            "Feature-space independence applies only to instance-dependent corruption."
        )
        circularity_risk = False
    config_payload = {
        "schema_version": CORRUPTION_CONFIG_SCHEMA_VERSION,
        "rounding_policy": ROUNDING_POLICY,
        "mechanism": canonical,
        "rate": fraction,
        "exact_count": count,
        "corruption_seed": seed,
        "dataset_seed": dataset_seed,
        "n_classes": n_classes,
        "inputs": {
            "n_samples": len(labels),
            "pre_corruption_labels_sha256": _sequence_digest(tuple(int(value) for value in labels)),
            "sample_ids_sha256": _sequence_digest(sample_id_values),
            "group_ids_sha256": _sequence_digest(group_id_values),
            "upstream_manifest": (
                {"status": "supplied", "sha256": upstream_manifest_hash}
                if upstream_manifest_hash is not None
                else {"status": "not_supplied"}
            ),
        },
        "transition_matrix": transition_payload,
        "group_weights": group_weight_payload,
        "generator_features": _feature_input_metadata(generator_features),
        "generator_representation": generator_representation,
        "auditor_representation": auditor_representation,
        "feature_space_independence": (
            independence_evidence.as_dict()
            if independence_evidence is not None
            else {"status": "not_supplied"}
        ),
    }
    configuration_payload_json = json.dumps(
        config_payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    config_hash = hashlib.sha256(configuration_payload_json.encode("utf-8")).hexdigest()
    result = CorruptionResult(
        pre_corruption_labels=labels,
        observed_labels=observed,
        is_injected_corruption=injected,
        selected_indices=selected,
        original_class=labels.copy(),
        replacement_class=replacement_by_row,
        mechanism=canonical,
        requested_rate=fraction,
        exact_count=count,
        corruption_seed=seed,
        dataset_seed=dataset_seed,
        generator_representation=generator_representation,
        auditor_representation=auditor_representation,
        feature_space_independent=independent,
        independence_status=independence_status,
        independence_reason=independence_reason,
        independence_evidence=independence_evidence,
        circularity_risk=circularity_risk,
        configuration_hash=config_hash,
        configuration_payload_json=configuration_payload_json,
        upstream_manifest_hash=upstream_manifest_hash,
        timestamp_utc=timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
    result.validate(n_classes=n_classes)
    return result


def apply_corruption_to_records(
    records: Sequence[NucleusRecord], result: CorruptionResult
) -> tuple[NucleusRecord, ...]:
    """Return new manifest records; source records are never mutated."""

    if len(records) != len(result.observed_labels):
        raise ValueError("records and corruption result have different lengths")
    updated: list[NucleusRecord] = []
    for index, record in enumerate(records):
        if record.pre_corruption_label != int(result.pre_corruption_labels[index]):
            raise ValueError(f"reference-label mismatch for {record.sample_id}")
        injected = bool(result.is_injected_corruption[index])
        updated.append(
            replace(
                record,
                observed_label=int(result.observed_labels[index]),
                is_injected_corruption=injected,
                corruption_type=result.mechanism if injected else "none",
                original_class=int(result.original_class[index]),
                replacement_class=(int(result.replacement_class[index]) if injected else None),
                corruption_seed=result.corruption_seed,
                corruption_rate=result.requested_rate,
                corruption_representation=result.generator_representation,
                auditor_representation=result.auditor_representation,
                feature_space_independent=result.feature_space_independent,
                circularity_risk=result.circularity_risk,
                configuration_hash=result.configuration_hash,
                corruption_timestamp_utc=result.timestamp_utc,
            )
        )
    n_classes = int(max(result.pre_corruption_labels.max(), result.observed_labels.max()) + 1)
    validate_manifest(updated, n_classes=n_classes)
    return tuple(updated)
