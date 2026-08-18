"""Strict, outcome-independent contracts for the primary and confirmatory studies.

The executable studies are deliberately downstream of these validators.  A plan that
contains a pilot-dependent placeholder or an outcome-dependent selection instruction
must never be frozen or expanded into an execution matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from histo_audit.config import config_sha256, resolve_config

CLASS_ORDER = (0, 1, 2, 3, 4)
PRIMARY_CORRUPTION_MECHANISMS = (
    "symmetric_random_corruption",
    "confusion_targeted_corruption",
    "group_conditional_corruption",
    "instance_dependent_corruption",
)
PRIMARY_AUDIT_METHODS = (
    "self_confidence",
    "negative_log_likelihood",
    "prediction_margin",
    "predictive_entropy",
    "cleanlab",
    "nearest_neighbour_disagreement",
    "fixed_hybrid",
)
CONFIRMATORY_ENSEMBLE_RISKS = (
    "predictive_entropy_of_mean",
    "mean_pairwise_js_divergence",
    "variation_ratio",
    "observed_label_probability_variance",
    "predicted_class_disagreement",
)
CONFIRMATORY_COMPARISON_RISK_IDS = (
    "self_confidence",
    "ensemble_disagreement",
    "fixed_hybrid",
    "hybrid_drop_self_confidence",
    "hybrid_drop_ensemble_disagreement",
)
RESOURCE_BOUNDED_CONFIRMATORY_PROFILE = "resource_bounded_confirmatory_v1"
RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256 = (
    "af99f0acfe3a075715a2e90d28ae6b896197fc30cf3819303d7b82837a0f6f88"
)
ORIGINAL_AUDIT_RISK_METHODS = (
    "self_confidence",
    "negative_log_likelihood",
    "prediction_margin",
    "predictive_entropy",
)
PRIMARY_GENERATOR_INDEPENDENCE_STATUSES = (
    "verified_independent",
    "circularity_risk",
    "unavailable_optional",
)
CONFIRMATORY_REQUIRED_FROZEN_ABLATIONS: Mapping[str, Mapping[str, str]] = {
    "imagenet_frozen_logistic": {
        "representation_id": "imagenet_resnet18_context_embeddings",
        "cache_provenance_id": "imagenet_context_embedding_cache",
        "family": "imagenet_frozen",
        "input_variant": "context_rgb",
        "encoder": "resnet18_imagenet1k_v1",
        "classifier": "multinomial_logistic_regression",
    },
    "imagenet_frozen_target_highlighted_logistic": {
        "representation_id": "imagenet_target_highlighted_embeddings",
        "cache_provenance_id": "imagenet_target_highlighted_embedding_cache",
        "family": "imagenet_frozen",
        "input_variant": "target_highlighted_rgb",
        "encoder": "resnet18_imagenet1k_v1",
        "classifier": "multinomial_logistic_regression",
    },
    "imagenet_frozen_context_morphometrics_logistic": {
        "representation_id": "imagenet_context_embeddings_plus_target_morphometrics",
        "cache_provenance_id": "imagenet_context_morphometrics_cache",
        "family": "imagenet_frozen",
        "input_variant": "context_rgb_plus_target_morphometrics",
        "encoder": "resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
        "classifier": "multinomial_logistic_regression",
    },
}
_OUTCOME_DEPENDENT = re.compile(
    r"(?:best[_ -]?primary|selected[_ -]?after[_ -]?primary|"
    r"choose[_ -]?(?:best|after)|outcome[_ -]?dependent|favourable[_ -]?result|"
    r"unresolved|unset|tbd|todo|placeholder|not[_ -]?frozen)",
    flags=re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,159}$")

_PRIMARY_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "status",
        "data",
        "pilot_derived_parameters",
        "corruption",
        "representations",
        "classifiers",
        "calibration",
        "oof",
        "audit",
        "evaluation",
        "statistics",
        "restoration",
    }
)
_CONFIRMATORY_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "status",
        "data",
        "corruption",
        "scenarios",
        "cache_provenance",
        "model_seeds",
        "training",
        "oof",
        "original_audit_selection",
        "ensemble",
        "fixed_hybrid",
        "restoration",
        "statistics",
        "fold_rotation",
    }
)
_RESOURCE_BOUNDED_CONFIRMATORY_TOP_LEVEL_FIELDS = _CONFIRMATORY_TOP_LEVEL_FIELDS | {
    "execution_profile",
    "analysis_disposition",
    "original_confirmatory_claim_allowed",
    "completion_stage",
}
_PRIMARY_DATA_FIELDS = frozenset(
    {
        "source",
        "class_order",
        "analysis_manifest_authority",
        "development_official_folds",
        "final_test_fold",
        "group_unit",
        "reference_validation_fraction_groups",
        "reference_group_selection_algorithm",
        "split_seed",
        "fold_assignment_labels",
        "exclusions",
    }
)
_CONFIRMATORY_DATA_FIELDS = frozenset(
    {
        "source",
        "analysis_manifest_authority",
        "official_folds",
        "group_unit",
        "reference_validation_fraction_groups",
        "reference_group_selection_algorithm",
        "split_seed",
        "fold_assignment_labels",
    }
)
_ANALYSIS_MANIFEST_AUTHORITY_FIELDS = frozenset(
    {
        "canonical_manifest_sha256",
        "analysis_eligible_sample_order_sha256",
        "analysis_eligible_sample_count",
    }
)
_PANNUKE_EXCLUSION_POLICY: Mapping[str, str] = {
    "cross_class_overlap_touching": "exclude_with_reason_touches_cross_class_overlap",
    "positive_background_conflict_touching": "retain_with_qc_flag_no_class_arbitration",
    "disconnected_instance_id": "retain_with_flag",
    "border_instance": "retain_with_flag",
    "malformed_or_structurally_invalid_mask": "fail_closed_at_dataset_gate",
    "duplicates": "flag_without_automatic_deletion",
    "void_pixels": "retain_as_unlabeled_void",
    "missing_required_data": "fail_closed_at_dataset_gate",
}
_CLEAN_REFERENCE_CELL: Mapping[str, Any] = {
    "id": "clean_reference_cell",
    "mechanism": "symmetric_random_corruption",
    "rate": 0.0,
    "seed": 404,
    "parameters": {},
}
_PRIMARY_CORRUPTION_SEEDS = (404, 405, 406)


class StudyContractError(ValueError):
    """A frozen-study candidate is incomplete, ambiguous, or outcome-dependent."""


@dataclass(frozen=True, slots=True)
class PrimaryScenario:
    """One immutable controlled-corruption scenario shared by all model cells."""

    scenario_id: str
    mechanism: str
    rate: float
    corruption_seed: int


@dataclass(frozen=True, slots=True)
class PrimaryCell:
    """One frozen-feature model cell in the primary execution matrix."""

    cell_id: str
    scenario_id: str
    mechanism: str
    rate: float
    corruption_seed: int
    representation_id: str
    classifier_id: str
    required: bool


@dataclass(frozen=True, slots=True)
class PrimaryMatrixPlan:
    """Deterministic expansion of a valid frozen primary configuration."""

    schema_version: int
    config_sha256: str
    scenarios: tuple[PrimaryScenario, ...]
    cells: tuple[PrimaryCell, ...]

    @property
    def required_cell_count(self) -> int:
        return sum(cell.required for cell in self.cells)

    @property
    def optional_cell_count(self) -> int:
        return len(self.cells) - self.required_cell_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "scenario_count": len(self.scenarios),
            "cell_count": len(self.cells),
            "required_cell_count": self.required_cell_count,
            "optional_cell_count": self.optional_cell_count,
            "scenarios": [asdict(item) for item in self.scenarios],
            "cells": [asdict(item) for item in self.cells],
        }


@dataclass(frozen=True, slots=True)
class ConfirmatoryCell:
    """One predeclared model cell in one official-fold rotation."""

    cell_id: str
    outer_fold: int
    corruption_cell_id: str
    scenario_id: str
    model_seed: int
    required: bool


@dataclass(frozen=True, slots=True)
class ConfirmatoryMatrixPlan:
    """Deterministic expansion of a valid frozen confirmatory configuration."""

    schema_version: int
    config_sha256: str
    cells: tuple[ConfirmatoryCell, ...]

    @property
    def required_cell_count(self) -> int:
        return sum(cell.required for cell in self.cells)

    @property
    def optional_cell_count(self) -> int:
        return len(self.cells) - self.required_cell_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "cell_count": len(self.cells),
            "required_cell_count": self.required_cell_count,
            "optional_cell_count": self.optional_cell_count,
            "cells": [asdict(item) for item in self.cells],
        }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyContractError(f"{location} must be a mapping")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StudyContractError(f"{location} must be a sequence")
    return value


def _require_fields(mapping: Mapping[str, Any], fields: Sequence[str], location: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise StudyContractError(f"{location} lacks required fields: {missing}")


def _require_exact_fields(
    mapping: Mapping[str, Any], fields: Sequence[str] | frozenset[str], location: str
) -> None:
    """Reject both missing and silently ignored fields in a frozen contract."""

    expected = set(fields)
    actual = set(mapping)
    if actual == expected:
        return
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    raise StudyContractError(
        f"{location} must contain exactly its frozen fields; "
        f"missing={missing}, unexpected={unexpected}"
    )


def _exact_identifier(value: Any, location: str) -> str:
    identifier = str(value)
    if _IDENTIFIER.fullmatch(identifier) is None:
        raise StudyContractError(
            f"{location} must match {_IDENTIFIER.pattern!r}; received {identifier!r}"
        )
    return identifier


def _positive_int(value: Any, location: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise StudyContractError(f"{location} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise StudyContractError(f"{location} must be an integer") from exc
    if result != value or result < minimum:
        raise StudyContractError(f"{location} must be an integer >= {minimum}")
    return result


def _probability(value: Any, location: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise StudyContractError(f"{location} must be a probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyContractError(f"{location} must be a probability") from exc
    lower_valid = result >= 0.0 if allow_zero else result > 0.0
    if not lower_valid or result > 1.0:
        boundary = "[0, 1]" if allow_zero else "(0, 1]"
        raise StudyContractError(f"{location} must be in {boundary}")
    return result


def _finite_float(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise StudyContractError(f"{location} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyContractError(f"{location} must be numeric") from exc
    if not math.isfinite(result):
        raise StudyContractError(f"{location} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise StudyContractError(f"{location} must be {relation} {minimum}")
    return result


def _sha256(value: Any, location: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise StudyContractError(f"{location} must be a SHA-256")
    return digest


def _analysis_manifest_authority(
    data: Mapping[str, Any],
    *,
    location: str,
) -> tuple[str, str, int]:
    """Return the exact manifest, row-order, and eligible-count authority."""

    authority = _mapping(data.get("analysis_manifest_authority"), location)
    _require_exact_fields(authority, _ANALYSIS_MANIFEST_AUTHORITY_FIELDS, location)
    return (
        _sha256(
            authority["canonical_manifest_sha256"],
            f"{location}.canonical_manifest_sha256",
        ),
        _sha256(
            authority["analysis_eligible_sample_order_sha256"],
            f"{location}.analysis_eligible_sample_order_sha256",
        ),
        _positive_int(
            authority["analysis_eligible_sample_count"],
            f"{location}.analysis_eligible_sample_count",
        ),
    )


def _assert_no_outcome_dependent_values(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_no_outcome_dependent_values(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_no_outcome_dependent_values(nested, f"{location}[{index}]")
    elif isinstance(value, str) and _OUTCOME_DEPENDENT.search(value):
        raise StudyContractError(
            f"{location} contains an incomplete or outcome-dependent value: {value!r}"
        )


def _unique_strings(values: Any, location: str, *, minimum: int = 1) -> tuple[str, ...]:
    result = tuple(str(value) for value in _sequence(values, location))
    if len(result) < minimum or any(not value for value in result):
        raise StudyContractError(f"{location} must contain at least {minimum} non-empty values")
    if len(set(result)) != len(result):
        raise StudyContractError(f"{location} contains duplicate values")
    return result


def _unique_ints(values: Any, location: str, *, minimum: int = 1) -> tuple[int, ...]:
    result = tuple(
        _positive_int(value, f"{location}[{index}]", minimum=0)
        for index, value in enumerate(_sequence(values, location))
    )
    if len(result) < minimum:
        raise StudyContractError(f"{location} must contain at least {minimum} values")
    if len(set(result)) != len(result):
        raise StudyContractError(f"{location} contains duplicate values")
    return result


def _validate_transition_matrix(
    value: Any,
    class_count: int,
    *,
    location: str = "corruption.mechanisms.confusion_targeted_corruption.transition_matrix",
) -> None:
    rows = _sequence(value, location)
    if len(rows) != class_count:
        raise StudyContractError("confusion transition_matrix must be square over class_order")
    for row_index, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"{location}[{row_index}]")
        if len(row) != class_count:
            raise StudyContractError("confusion transition_matrix must be square over class_order")
        probabilities = [
            _probability(
                item,
                f"{location}[{row_index}][{column_index}]",
                allow_zero=True,
            )
            for column_index, item in enumerate(row)
        ]
        if probabilities[row_index] != 0.0:
            raise StudyContractError("confusion transition_matrix diagonal must be zero")
        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise StudyContractError("each confusion transition_matrix row must sum to one")


def _validate_primary_cache_provenance(
    representation: Mapping[str, Any],
    *,
    family: str,
    required: bool,
    location: str,
) -> str:
    provenance = _mapping(representation["cache_provenance"], f"{location}.cache_provenance")
    expected_fields = {
        "status",
        "encoder_id",
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "sample_order_sha256",
        "dataset_manifest_sha256",
        "cache_recipe_sha256",
        "cache_file_sha256",
    }
    if set(provenance) != expected_fields:
        raise StudyContractError(
            f"{location}.cache_provenance must contain exactly the encoder, weights, "
            "preprocessing, sample-order, manifest, cache-recipe, and cache-file bindings"
        )
    if not str(provenance["encoder_id"]).strip():
        raise StudyContractError(f"{location}.cache_provenance.encoder_id must be explicit")
    for field in ("sample_order_sha256", "dataset_manifest_sha256", "cache_recipe_sha256"):
        _sha256(provenance[field], f"{location}.cache_provenance.{field}")
    status = str(provenance["status"])
    artifact_fields = (
        "encoder_implementation_sha256",
        "weights_sha256",
        "preprocessing_sha256",
        "cache_file_sha256",
    )
    if status == "available":
        for field in artifact_fields:
            _sha256(provenance[field], f"{location}.cache_provenance.{field}")
    elif status == "unavailable_optional":
        if required:
            raise StudyContractError(
                f"{location} is required and cannot use unavailable_optional cache provenance"
            )
        if family != "pathology":
            raise StudyContractError(
                f"{location} unavailable_optional cache provenance is reserved for pathology"
            )
        if any(provenance[field] is not None for field in artifact_fields):
            raise StudyContractError(
                f"{location} unavailable_optional provenance cannot claim unavailable artifact "
                "hashes"
            )
    else:
        raise StudyContractError(
            f"{location}.cache_provenance.status must be available or unavailable_optional"
        )
    return status


def _validate_primary_cell_selector(
    value: Any,
    *,
    location: str,
    representation_classifier_pairs: set[tuple[str, str]],
) -> tuple[str, ...]:
    selector = _mapping(value, location)
    if set(selector) == {"cell_id"}:
        return ("cell_id", _exact_identifier(selector["cell_id"], f"{location}.cell_id"))
    if set(selector) != {"representation_id", "classifier_id"}:
        raise StudyContractError(
            f"{location} must contain exactly cell_id or representation_id and classifier_id"
        )
    representation_id = _exact_identifier(
        selector["representation_id"], f"{location}.representation_id"
    )
    classifier_id = _exact_identifier(selector["classifier_id"], f"{location}.classifier_id")
    if (representation_id, classifier_id) not in representation_classifier_pairs:
        raise StudyContractError(
            f"{location} must name a frozen representation/classifier matrix cell"
        )
    return ("representation_classifier", representation_id, classifier_id)


def _validate_primary_corruption_filter(
    value: Any,
    *,
    location: str,
    rates: tuple[float, ...],
    seeds: tuple[int, ...],
) -> tuple[str, float, int]:
    corruption_filter = _mapping(value, location)
    if set(corruption_filter) != {"mechanism", "rate", "seed"}:
        raise StudyContractError(f"{location} must contain exactly mechanism, rate, and seed")
    mechanism = str(corruption_filter["mechanism"])
    if mechanism not in PRIMARY_CORRUPTION_MECHANISMS:
        raise StudyContractError(f"{location}.mechanism is not frozen in the primary matrix")
    rate = _probability(corruption_filter["rate"], f"{location}.rate")
    if rate not in rates:
        raise StudyContractError(f"{location}.rate is not frozen in the primary matrix")
    seed = _positive_int(corruption_filter["seed"], f"{location}.seed", minimum=0)
    if seed not in seeds:
        raise StudyContractError(f"{location}.seed is not frozen in the primary matrix")
    return mechanism, rate, seed


def _validate_primary_comparison_selector(
    value: Any,
    *,
    location: str,
    rates: tuple[float, ...],
    seeds: tuple[int, ...],
    representation_classifier_pairs: set[tuple[str, str]],
) -> tuple[str, ...]:
    selector = _mapping(value, location)
    if set(selector) == {"cell_id"}:
        return _validate_primary_cell_selector(
            selector,
            location=location,
            representation_classifier_pairs=representation_classifier_pairs,
        )
    expected_fields = {
        "mechanism",
        "rate",
        "seed",
        "representation_id",
        "classifier_id",
    }
    if set(selector) != expected_fields:
        raise StudyContractError(
            f"{location} must contain exactly cell_id or mechanism, rate, seed, "
            "representation_id, and classifier_id"
        )
    mechanism, rate, seed = _validate_primary_corruption_filter(
        {
            "mechanism": selector["mechanism"],
            "rate": selector["rate"],
            "seed": selector["seed"],
        },
        location=location,
        rates=rates,
        seeds=seeds,
    )
    representation_selector = _validate_primary_cell_selector(
        {
            "representation_id": selector["representation_id"],
            "classifier_id": selector["classifier_id"],
        },
        location=location,
        representation_classifier_pairs=representation_classifier_pairs,
    )
    return ("expanded", mechanism, str(rate), str(seed), *representation_selector[1:])


def validate_frozen_primary_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a portable schema-v2 primary configuration.

    This validates the scientific decisions needed to enumerate the complete matrix.  It
    intentionally rejects descriptive drafts and pilot-dependent placeholders.
    """

    resolved = resolve_config(config)
    _assert_no_outcome_dependent_values(resolved)
    _require_exact_fields(resolved, _PRIMARY_TOP_LEVEL_FIELDS, "primary config")
    if resolved.get("schema_version") != 2:
        raise StudyContractError("primary schema_version must be exactly 2")
    if resolved.get("experiment_name") != "primary_frozen_feature_benchmark":
        raise StudyContractError("primary experiment_name is invalid")
    if str(resolved.get("status", "")).casefold() not in {
        "ready_for_freeze",
        "complete_for_freeze",
    }:
        raise StudyContractError("primary status must be READY_FOR_FREEZE or COMPLETE_FOR_FREEZE")

    data = _mapping(resolved.get("data"), "data")
    _require_exact_fields(data, _PRIMARY_DATA_FIELDS, "data")
    if str(data["source"]).casefold() != "pannuke":
        raise StudyContractError("primary data.source must be pannuke")
    class_order = tuple(int(value) for value in _sequence(data["class_order"], "data.class_order"))
    if class_order != CLASS_ORDER:
        raise StudyContractError(f"data.class_order must be exactly {list(CLASS_ORDER)}")
    authority_manifest_sha256, authority_sample_order_sha256, _ = _analysis_manifest_authority(
        data,
        location="data.analysis_manifest_authority",
    )
    development_folds = _unique_ints(
        data["development_official_folds"], "data.development_official_folds", minimum=2
    )
    final_fold = _positive_int(data["final_test_fold"], "data.final_test_fold")
    if development_folds != (1, 2):
        raise StudyContractError(
            "PanNuke primary development_official_folds must be exactly [1, 2]"
        )
    if final_fold != 3:
        raise StudyContractError("PanNuke primary final_test_fold must be exactly 3")
    if data["group_unit"] != "source_patch_id":
        raise StudyContractError("data.group_unit must be exactly source_patch_id")
    reference_fraction = _probability(
        data["reference_validation_fraction_groups"],
        "data.reference_validation_fraction_groups",
    )
    if reference_fraction != 0.10:
        raise StudyContractError("data.reference_validation_fraction_groups must be exactly 0.10")
    if (
        data["reference_group_selection_algorithm"]
        != "deterministic_group_greedy_class_distribution_v1"
    ):
        raise StudyContractError(
            "data.reference_group_selection_algorithm must be exactly "
            "deterministic_group_greedy_class_distribution_v1"
        )
    if _positive_int(data["split_seed"], "data.split_seed", minimum=0) != 223:
        raise StudyContractError("data.split_seed must be exactly 223")
    if data["fold_assignment_labels"] != "pre_corruption_label":
        raise StudyContractError("OOF fold assignments must be fixed from pre_corruption_label")
    exclusions = _mapping(data["exclusions"], "data.exclusions")
    if exclusions != _PANNUKE_EXCLUSION_POLICY:
        raise StudyContractError(
            "data.exclusions must exactly freeze the PanNuke overlap, background-conflict, "
            "disconnected-instance, border, structural-mask, duplicate, void, and missing-data "
            "policies"
        )

    pilot_derived = _mapping(resolved.get("pilot_derived_parameters"), "pilot_derived_parameters")
    _require_exact_fields(
        pilot_derived,
        (
            "schema_version",
            "producer_id",
            "path",
            "sha256",
            "source_pilot_run_id",
            "source_pilot_artifact_root_sha256",
        ),
        "pilot_derived_parameters",
    )
    if pilot_derived["schema_version"] != 1:
        raise StudyContractError("pilot_derived_parameters.schema_version must be exactly 1")
    if pilot_derived["producer_id"] != "pilot_derived_primary_parameters_v1":
        raise StudyContractError(
            "pilot_derived_parameters.producer_id must be pilot_derived_primary_parameters_v1"
        )
    if pilot_derived["path"] != "reports/pilot_derived_primary_parameters.json":
        raise StudyContractError(
            "pilot_derived_parameters.path must be exactly "
            "reports/pilot_derived_primary_parameters.json"
        )
    _sha256(pilot_derived["sha256"], "pilot_derived_parameters.sha256")
    source_pilot_run_id = pilot_derived["source_pilot_run_id"]
    if not isinstance(source_pilot_run_id, str) or _RUN_ID.fullmatch(source_pilot_run_id) is None:
        raise StudyContractError(
            "pilot_derived_parameters.source_pilot_run_id must be an exact safe run identifier"
        )
    _sha256(
        pilot_derived["source_pilot_artifact_root_sha256"],
        "pilot_derived_parameters.source_pilot_artifact_root_sha256",
    )

    corruption = _mapping(resolved.get("corruption"), "corruption")
    _require_exact_fields(
        corruption,
        ("rounding_policy", "rates", "seeds", "mechanisms", "clean_reference"),
        "corruption",
    )
    if corruption["rounding_policy"] != "round_half_up":
        raise StudyContractError("corruption.rounding_policy must be round_half_up")
    rates = tuple(
        _probability(value, f"corruption.rates[{index}]")
        for index, value in enumerate(_sequence(corruption["rates"], "corruption.rates"))
    )
    if rates != (0.05, 0.10, 0.20):
        raise StudyContractError("primary corruption.rates must be exactly [0.05, 0.10, 0.20]")
    seeds = _unique_ints(corruption["seeds"], "corruption.seeds", minimum=3)
    if seeds != _PRIMARY_CORRUPTION_SEEDS:
        raise StudyContractError("primary corruption.seeds must be exactly [404, 405, 406]")
    clean_reference = _mapping(corruption["clean_reference"], "corruption.clean_reference")
    _require_exact_fields(
        clean_reference,
        ("id", "mechanism", "rate", "seed", "parameters"),
        "corruption.clean_reference",
    )
    if _exact_identifier(clean_reference["id"], "corruption.clean_reference.id") != str(
        _CLEAN_REFERENCE_CELL["id"]
    ):
        raise StudyContractError("corruption.clean_reference.id must be clean_reference_cell")
    if clean_reference["mechanism"] != _CLEAN_REFERENCE_CELL["mechanism"]:
        raise StudyContractError(
            "corruption.clean_reference.mechanism must be symmetric_random_corruption"
        )
    if (
        _probability(clean_reference["rate"], "corruption.clean_reference.rate", allow_zero=True)
        != 0.0
    ):
        raise StudyContractError("corruption.clean_reference.rate must be exactly 0.0")
    if _positive_int(clean_reference["seed"], "corruption.clean_reference.seed", minimum=0) != 404:
        raise StudyContractError("corruption.clean_reference.seed must be exactly 404")
    if _mapping(clean_reference["parameters"], "corruption.clean_reference.parameters"):
        raise StudyContractError("corruption.clean_reference.parameters must be exactly empty")
    mechanisms = _mapping(corruption["mechanisms"], "corruption.mechanisms")
    if set(mechanisms) != set(PRIMARY_CORRUPTION_MECHANISMS):
        raise StudyContractError(
            "primary corruption.mechanisms must contain exactly the four required mechanisms"
        )
    symmetric = _mapping(mechanisms["symmetric_random_corruption"], "symmetric mechanism")
    _require_exact_fields(symmetric, (), "symmetric mechanism")
    confusion = _mapping(mechanisms["confusion_targeted_corruption"], "confusion mechanism")
    _require_exact_fields(confusion, ("transition_matrix",), "confusion mechanism")
    _validate_transition_matrix(confusion["transition_matrix"], len(class_order))
    group_conditional = _mapping(
        mechanisms["group_conditional_corruption"], "group-conditional mechanism"
    )
    _require_exact_fields(
        group_conditional,
        ("grouping_field", "weights_by_value", "default_weight"),
        "group mechanism",
    )
    if not str(group_conditional["grouping_field"]):
        raise StudyContractError("group-conditional grouping_field must be explicit")
    if not _mapping(group_conditional["weights_by_value"], "group weights"):
        raise StudyContractError("group-conditional weights_by_value must be non-empty")
    _probability(group_conditional["default_weight"], "group default_weight", allow_zero=True)
    instance = _mapping(mechanisms["instance_dependent_corruption"], "instance mechanism")
    _require_exact_fields(
        instance,
        (
            "generator_representation",
            "auditor_representation_families",
            "independence_status",
            "independence_matrix_path",
            "independence_matrix_sha256",
        ),
        "instance mechanism",
    )
    if instance["independence_status"] != "per_representation_matrix":
        raise StudyContractError(
            "instance-dependent primary cells require independence_status=per_representation_matrix"
        )
    auditor_representation_families = set(
        _unique_strings(instance["auditor_representation_families"], "auditor families")
    )
    if not str(instance["generator_representation"]):
        raise StudyContractError("instance generator representation must be explicit")
    independence_matrix_sha256 = _sha256(
        instance["independence_matrix_sha256"],
        "instance independence_matrix_sha256",
    )
    if not str(instance["independence_matrix_path"]):
        raise StudyContractError("instance independence_matrix_path must be explicit")

    representations = _sequence(resolved.get("representations"), "representations")
    if not representations:
        raise StudyContractError("representations must be non-empty")
    representation_ids: list[str] = []
    required_families: set[str] = set()
    input_variants: set[str] = set()
    for index, raw_representation in enumerate(representations):
        location = f"representations[{index}]"
        representation = _mapping(raw_representation, location)
        family = _exact_identifier(representation.get("family"), f"{location}.family")
        expected_representation_fields = {
            "id",
            "family",
            "input_variant",
            "required",
            "classifiers",
            "cache_provenance",
            "generator_independence",
        }
        if family == "pathology":
            expected_representation_fields.add("availability_audit_sha256")
        if set(representation) != expected_representation_fields:
            raise StudyContractError(
                f"{location} must contain exactly its identity, classifiers, cache provenance, "
                "and generator-independence binding"
            )
        identifier = _exact_identifier(representation["id"], f"{location}.id")
        representation_ids.append(identifier)
        if not isinstance(representation["required"], bool):
            raise StudyContractError(f"{location}.required must be boolean")
        required = bool(representation["required"])
        if required:
            required_families.add(family)
        input_variants.add(
            _exact_identifier(representation["input_variant"], f"{location}.input_variant")
        )
        _unique_strings(representation["classifiers"], f"{location}.classifiers")
        if family == "pathology" and "availability_audit_sha256" not in representation:
            raise StudyContractError("pathology representation requires availability_audit_sha256")
        if family == "pathology":
            _sha256(
                representation["availability_audit_sha256"],
                f"{location}.availability_audit_sha256",
            )
        provenance_status = _validate_primary_cache_provenance(
            representation,
            family=family,
            required=required,
            location=location,
        )
        provenance = _mapping(
            representation["cache_provenance"],
            f"{location}.cache_provenance",
        )
        if (
            provenance["dataset_manifest_sha256"] != authority_manifest_sha256
            or provenance["sample_order_sha256"] != authority_sample_order_sha256
        ):
            raise StudyContractError(
                f"{location}.cache_provenance differs from data.analysis_manifest_authority"
            )
        generator_independence = _mapping(
            representation["generator_independence"],
            f"{location}.generator_independence",
        )
        if set(generator_independence) != {"status", "independence_matrix_sha256"}:
            raise StudyContractError(
                f"{location}.generator_independence must contain exactly status and "
                "independence_matrix_sha256"
            )
        generator_status = str(generator_independence["status"])
        if generator_status not in PRIMARY_GENERATOR_INDEPENDENCE_STATUSES:
            raise StudyContractError(
                f"{location}.generator_independence.status must be one of "
                f"{list(PRIMARY_GENERATOR_INDEPENDENCE_STATUSES)}"
            )
        representation_matrix_sha256 = _sha256(
            generator_independence["independence_matrix_sha256"],
            f"{location}.generator_independence.independence_matrix_sha256",
        )
        if representation_matrix_sha256 != independence_matrix_sha256:
            raise StudyContractError(
                f"{location} generator-independence matrix differs from the frozen corruption "
                "matrix"
            )
        if provenance_status == "available":
            if generator_status == "unavailable_optional":
                raise StudyContractError(
                    f"{location} has an available cache but unavailable generator-independence "
                    "evidence"
                )
            if family not in auditor_representation_families:
                raise StudyContractError(
                    f"{location}.family is absent from instance-dependent auditor families"
                )
        elif generator_status != "unavailable_optional":
            raise StudyContractError(
                f"{location} unavailable cache must use unavailable_optional independence status"
            )
        if (
            family == "engineered"
            and instance["generator_representation"] == "morphology_only_v1"
            and generator_status != "circularity_risk"
        ):
            raise StudyContractError(
                "engineered target features overlap morphology_only_v1 and must be frozen as "
                "circularity_risk"
            )
    if len(set(representation_ids)) != len(representation_ids):
        raise StudyContractError("representation IDs must be unique")
    if not {"engineered", "imagenet"}.issubset(required_families):
        raise StudyContractError("engineered and imagenet must be required representation families")
    if not {"context_rgb", "target_highlighted_rgb"}.issubset(input_variants):
        raise StudyContractError(
            "primary must freeze context_rgb and target_highlighted_rgb inputs"
        )

    classifiers = _mapping(resolved.get("classifiers"), "classifiers")
    if set(classifiers) != {"multinomial_logistic_regression", "small_mlp"}:
        raise StudyContractError(
            "primary classifiers must contain exactly multinomial_logistic_regression and small_mlp"
        )
    for classifier_id in ("multinomial_logistic_regression", "small_mlp"):
        classifier = _mapping(classifiers.get(classifier_id), f"classifiers.{classifier_id}")
        classifier_fields = (
            (
                "l2",
                "max_iter",
                "class_weight",
                "class_weight_label_source",
                "fit_label_source",
                "model_seed",
            )
            if classifier_id == "multinomial_logistic_regression"
            else (
                "hidden_dimensions",
                "dropout",
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "model_seed",
                "amp",
                "amp_dtype",
                "gradient_accumulation_steps",
                "minimum_batch_size",
                "early_stopping_patience",
                "early_stopping_min_delta",
                "fit_label_source",
            )
        )
        if set(classifier) != set(classifier_fields):
            raise StudyContractError(
                f"classifiers.{classifier_id} must contain exactly its frozen training controls"
            )
        if classifier["fit_label_source"] != "observed_development_labels_only":
            raise StudyContractError(
                f"classifiers.{classifier_id}.fit_label_source must be "
                "observed_development_labels_only"
            )
        if classifier_id == "multinomial_logistic_regression":
            _finite_float(classifier["l2"], "classifiers.logistic.l2", minimum=0.0)
            _positive_int(classifier["max_iter"], "classifiers.logistic.max_iter")
            if classifier["class_weight"] != "balanced":
                raise StudyContractError("primary logistic class_weight must be balanced")
            if classifier["class_weight_label_source"] != "observed_development_labels_only":
                raise StudyContractError(
                    "primary logistic class weights must use observed development labels only"
                )
            _positive_int(classifier["model_seed"], "classifiers.logistic.model_seed", minimum=0)
        else:
            widths = _sequence(
                classifier["hidden_dimensions"], "classifiers.small_mlp.hidden_dimensions"
            )
            if not widths:
                raise StudyContractError("small_mlp.hidden_dimensions must be non-empty")
            for index, width in enumerate(widths):
                _positive_int(width, f"small_mlp.hidden_dimensions[{index}]")
            dropout = _finite_float(classifier["dropout"], "small_mlp.dropout", minimum=0.0)
            if dropout >= 1.0:
                raise StudyContractError("small_mlp.dropout must be less than one")
            _positive_int(classifier["epochs"], "small_mlp.epochs")
            _positive_int(classifier["batch_size"], "small_mlp.batch_size")
            _finite_float(
                classifier["learning_rate"],
                "small_mlp.learning_rate",
                minimum=0.0,
                strict_minimum=True,
            )
            _finite_float(classifier["weight_decay"], "small_mlp.weight_decay", minimum=0.0)
            _positive_int(classifier["model_seed"], "small_mlp.model_seed", minimum=0)
            if not isinstance(classifier["amp"], bool):
                raise StudyContractError("small_mlp.amp must be boolean")
            if str(classifier["amp_dtype"]) not in {"float16", "bfloat16"}:
                raise StudyContractError("small_mlp.amp_dtype must be float16 or bfloat16")
            _positive_int(
                classifier["gradient_accumulation_steps"],
                "small_mlp.gradient_accumulation_steps",
            )
            minimum_batch = _positive_int(
                classifier["minimum_batch_size"], "small_mlp.minimum_batch_size"
            )
            if minimum_batch > int(classifier["batch_size"]):
                raise StudyContractError("small_mlp.minimum_batch_size exceeds batch_size")
            patience = classifier["early_stopping_patience"]
            if patience is not None:
                _positive_int(patience, "small_mlp.early_stopping_patience")
            _finite_float(
                classifier["early_stopping_min_delta"],
                "small_mlp.early_stopping_min_delta",
                minimum=0.0,
            )
    configured_classifier_ids = set(classifiers)
    representation_classifier_pairs: set[tuple[str, str]] = set()
    for index, raw_representation in enumerate(representations):
        representation = _mapping(raw_representation, f"representations[{index}]")
        representation_classifiers = set(str(value) for value in representation["classifiers"])
        unknown = representation_classifiers.difference(configured_classifier_ids)
        if unknown:
            raise StudyContractError(f"representation references unknown classifiers: {unknown}")
        representation_classifier_pairs.update(
            (str(representation["id"]), classifier_id)
            for classifier_id in representation_classifiers
        )

    calibration = _mapping(resolved.get("calibration"), "calibration")
    expected_calibration_fields = {
        "enabled",
        "method",
        "source",
        "reporting",
        "fit_labels_policy",
        "seed",
        "parameters",
    }
    if set(calibration) != expected_calibration_fields:
        raise StudyContractError(
            "calibration must contain exactly enabled, method, source, reporting, "
            "fit_labels_policy, seed, and parameters"
        )
    if not isinstance(calibration["enabled"], bool):
        raise StudyContractError("calibration.enabled must be boolean")
    calibration_method = _exact_identifier(calibration["method"], "calibration.method")
    calibration_parameters = _mapping(calibration["parameters"], "calibration.parameters")
    if calibration["enabled"]:
        if calibration_method == "none":
            raise StudyContractError("enabled calibration cannot use method=none")
    elif calibration_method != "none" or calibration_parameters:
        raise StudyContractError(
            "disabled calibration requires method=none and exactly empty parameters"
        )
    if calibration["source"] != "reference_validation_only":
        raise StudyContractError("calibration.source must be reference_validation_only")
    if calibration["reporting"] != "calibrated_and_uncalibrated":
        raise StudyContractError("calibration.reporting must be calibrated_and_uncalibrated")
    if calibration["fit_labels_policy"] != "observed_reference_validation_labels_only":
        raise StudyContractError(
            "calibration.fit_labels_policy must use observed reference-validation labels only"
        )
    _positive_int(calibration["seed"], "calibration.seed", minimum=0)

    oof = _mapping(resolved.get("oof"), "oof")
    _require_fields(oof, ("n_splits", "split_kind", "no_nucleus_level_fallback"), "oof")
    if _positive_int(oof["n_splits"], "oof.n_splits", minimum=2) < 2:
        raise AssertionError("unreachable")
    if oof["split_kind"] != "stratified_group":
        raise StudyContractError("oof.split_kind must be stratified_group")
    if oof["no_nucleus_level_fallback"] is not True:
        raise StudyContractError("nucleus-level OOF fallback must be disabled")

    audit = _mapping(resolved.get("audit"), "audit")
    _require_fields(
        audit,
        (
            "methods",
            "primary_method",
            "nearest_neighbour",
            "fixed_hybrid",
            "cleanlab_failure_policy",
        ),
        "audit",
    )
    methods = tuple(str(item) for item in _sequence(audit["methods"], "audit.methods"))
    if methods != PRIMARY_AUDIT_METHODS:
        raise StudyContractError("audit.methods must be the seven required methods in fixed order")
    if str(audit["primary_method"]) not in methods:
        raise StudyContractError("audit.primary_method must name a configured method")
    neighbour = _mapping(audit["nearest_neighbour"], "audit.nearest_neighbour")
    _require_fields(neighbour, ("k", "metric", "exclude_same_group"), "nearest neighbour")
    _positive_int(neighbour["k"], "audit.nearest_neighbour.k")
    if neighbour["exclude_same_group"] is not True:
        raise StudyContractError("nearest neighbours must exclude every same-group sample")
    hybrid = _mapping(audit["fixed_hybrid"], "audit.fixed_hybrid")
    _require_fields(hybrid, ("components", "weights"), "audit.fixed_hybrid")
    hybrid_components = _unique_strings(hybrid["components"], "fixed hybrid components", minimum=2)
    hybrid_weights = tuple(
        _probability(value, f"fixed hybrid weights[{index}]")
        for index, value in enumerate(_sequence(hybrid["weights"], "fixed hybrid weights"))
    )
    if len(hybrid_components) != len(hybrid_weights) or abs(sum(hybrid_weights) - 1.0) > 1e-9:
        raise StudyContractError("fixed hybrid components and weights must align and sum to one")
    equal_weight = 1.0 / len(hybrid_components)
    if any(abs(weight - equal_weight) > 1e-12 for weight in hybrid_weights):
        raise StudyContractError("fixed hybrid must use equal weights across its frozen components")
    if not set(hybrid_components).issubset(methods):
        raise StudyContractError("fixed hybrid components must name configured base methods")
    if "fixed_hybrid" in hybrid_components:
        raise StudyContractError("fixed_hybrid cannot recursively include itself")
    if audit["cleanlab_failure_policy"] != "missing_with_recorded_blocker":
        raise StudyContractError("Cleanlab failures must remain missing with a recorded blocker")

    evaluation = _mapping(resolved.get("evaluation"), "evaluation")
    _require_fields(
        evaluation,
        (
            "primary_metric",
            "primary_review_budget",
            "secondary_review_budgets",
            "random_review_repeats",
            "random_review_seed",
            "subgroup_min_samples",
            "subgroup_min_corruptions",
        ),
        "evaluation",
    )
    if evaluation["primary_metric"] != "average_precision":
        raise StudyContractError("primary ranking metric must be average_precision")
    if _probability(evaluation["primary_review_budget"], "primary review budget") != 0.05:
        raise StudyContractError("primary review budget must be exactly 0.05")
    secondary = tuple(
        _probability(value, f"secondary budgets[{index}]")
        for index, value in enumerate(
            _sequence(evaluation["secondary_review_budgets"], "secondary budgets")
        )
    )
    if secondary != (0.01, 0.10, 0.20):
        raise StudyContractError("secondary review budgets must be [0.01, 0.10, 0.20]")
    if _positive_int(evaluation["random_review_repeats"], "random repeats") < 100:
        raise StudyContractError("primary random review requires at least 100 repetitions")
    _positive_int(evaluation["random_review_seed"], "random review seed", minimum=0)
    if _positive_int(evaluation["subgroup_min_samples"], "subgroup minimum samples") < 100:
        raise StudyContractError("subgroup_min_samples must be at least 100")
    if _positive_int(evaluation["subgroup_min_corruptions"], "subgroup minimum corruptions") < 10:
        raise StudyContractError("subgroup_min_corruptions must be at least 10")

    statistics = _mapping(resolved.get("statistics"), "statistics")
    expected_statistics_fields = {
        "paired_group_bootstrap_iterations",
        "bootstrap_seed",
        "holm_families",
        "within_cell_comparisons",
        "cross_cell_comparisons",
        "method_vs_random_comparisons",
        "exploratory_multiple_comparison_correction",
    }
    if set(statistics) != expected_statistics_fields:
        raise StudyContractError(
            "primary statistics must contain exactly bootstrap controls, Holm families, "
            "within-cell, cross-cell, and method-vs-random comparisons, and exploratory "
            "correction"
        )
    if (
        _positive_int(statistics["paired_group_bootstrap_iterations"], "bootstrap iterations")
        < 2000
    ):
        raise StudyContractError("primary paired group bootstrap requires at least 2000 iterations")
    _positive_int(statistics["bootstrap_seed"], "bootstrap seed", minimum=0)
    holm_families = {
        _exact_identifier(value, f"statistics.holm_families[{index}]")
        for index, value in enumerate(
            _unique_strings(statistics["holm_families"], "statistics.holm_families")
        )
    }
    if len(holm_families) != len(statistics["holm_families"]):
        raise StudyContractError("statistics.holm_families contains duplicate values")
    non_budget_comparison_metrics = {
        "average_precision",
        "auroc",
    }
    supported_comparison_metrics = {
        *non_budget_comparison_metrics,
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    }
    comparison_ids: list[str] = []
    selector_references: list[tuple[str, tuple[str, ...]]] = []
    cross_cell_contracts: list[tuple[str, tuple[str, ...], tuple[str, ...], str, str]] = []

    within_cell_comparisons = _sequence(
        statistics["within_cell_comparisons"],
        "statistics.within_cell_comparisons",
    )
    if not within_cell_comparisons:
        raise StudyContractError("statistics.within_cell_comparisons must be non-empty")
    for index, raw_comparison in enumerate(within_cell_comparisons):
        location = f"statistics.within_cell_comparisons[{index}]"
        comparison = _mapping(raw_comparison, location)
        expected_comparison_fields = {
            "comparison_id",
            "selector",
            "method_a",
            "method_b",
            "metric",
            "direction",
            "holm_family",
        }
        if set(comparison) != expected_comparison_fields:
            raise StudyContractError(
                f"{location} must contain exactly its ID, selector, methods, metric, "
                "direction, and Holm family"
            )
        comparison_ids.append(
            _exact_identifier(comparison["comparison_id"], f"{location}.comparison_id")
        )
        selector = _validate_primary_comparison_selector(
            comparison["selector"],
            location=f"{location}.selector",
            rates=rates,
            seeds=seeds,
            representation_classifier_pairs=representation_classifier_pairs,
        )
        selector_references.append((f"{location}.selector", selector))
        if comparison["method_a"] not in methods or comparison["method_b"] not in methods:
            raise StudyContractError(f"{location} methods must name configured audit methods")
        if comparison["method_a"] == comparison["method_b"]:
            raise StudyContractError(f"{location} method_a and method_b must differ")
        if comparison["metric"] not in non_budget_comparison_metrics:
            raise StudyContractError(f"{location}.metric is unsupported")
        if comparison["direction"] != "method_a_minus_method_b":
            raise StudyContractError(f"{location}.direction must be method_a_minus_method_b")
        if comparison["holm_family"] not in holm_families:
            raise StudyContractError(f"{location}.holm_family is not declared")

    cross_cell_comparisons = _sequence(
        statistics["cross_cell_comparisons"],
        "statistics.cross_cell_comparisons",
    )
    if not cross_cell_comparisons:
        raise StudyContractError("statistics.cross_cell_comparisons must be non-empty")
    for index, raw_comparison in enumerate(cross_cell_comparisons):
        location = f"statistics.cross_cell_comparisons[{index}]"
        comparison = _mapping(raw_comparison, location)
        expected_comparison_fields = {
            "comparison_id",
            "selector_a",
            "selector_b",
            "method_a",
            "method_b",
            "metric",
            "direction",
            "holm_family",
        }
        if set(comparison) != expected_comparison_fields:
            raise StudyContractError(
                f"{location} must contain exactly its ID, explicit A/B selectors, methods, "
                "metric, direction, and Holm family"
            )
        comparison_ids.append(
            _exact_identifier(comparison["comparison_id"], f"{location}.comparison_id")
        )
        selector_a = _validate_primary_comparison_selector(
            comparison["selector_a"],
            location=f"{location}.selector_a",
            rates=rates,
            seeds=seeds,
            representation_classifier_pairs=representation_classifier_pairs,
        )
        selector_b = _validate_primary_comparison_selector(
            comparison["selector_b"],
            location=f"{location}.selector_b",
            rates=rates,
            seeds=seeds,
            representation_classifier_pairs=representation_classifier_pairs,
        )
        if selector_a == selector_b:
            raise StudyContractError(f"{location} selector_a and selector_b must differ")
        selector_references.extend(
            (
                (f"{location}.selector_a", selector_a),
                (f"{location}.selector_b", selector_b),
            )
        )
        if comparison["method_a"] not in methods or comparison["method_b"] not in methods:
            raise StudyContractError(f"{location} methods must name configured audit methods")
        if comparison["method_a"] != comparison["method_b"]:
            raise StudyContractError(
                f"{location} must hold the audit method fixed across its cell contrast"
            )
        cross_cell_contracts.append(
            (
                location,
                selector_a,
                selector_b,
                str(comparison["method_a"]),
                str(comparison["method_b"]),
            )
        )
        if comparison["metric"] not in non_budget_comparison_metrics:
            raise StudyContractError(f"{location}.metric is unsupported")
        if comparison["direction"] != "method_a_minus_method_b":
            raise StudyContractError(f"{location}.direction must be method_a_minus_method_b")
        if comparison["holm_family"] not in holm_families:
            raise StudyContractError(f"{location}.holm_family is not declared")

    method_vs_random = _sequence(
        statistics["method_vs_random_comparisons"],
        "statistics.method_vs_random_comparisons",
    )
    if not method_vs_random:
        raise StudyContractError("statistics.method_vs_random_comparisons must be non-empty")
    allowed_review_budgets = {
        float(evaluation["primary_review_budget"]),
        *(float(value) for value in evaluation["secondary_review_budgets"]),
    }
    for index, raw_comparison in enumerate(method_vs_random):
        location = f"statistics.method_vs_random_comparisons[{index}]"
        comparison = _mapping(raw_comparison, location)
        expected_comparison_fields = {
            "comparison_id",
            "selector",
            "method_a",
            "method_b",
            "metric",
            "review_budget",
            "direction",
            "holm_family",
        }
        if set(comparison) != expected_comparison_fields:
            raise StudyContractError(
                f"{location} must contain exactly its ID, selector, methods, "
                "metric, budget, direction, and Holm family"
            )
        comparison_ids.append(
            _exact_identifier(comparison["comparison_id"], f"{location}.comparison_id")
        )
        selector = _validate_primary_comparison_selector(
            comparison["selector"],
            location=f"{location}.selector",
            rates=rates,
            seeds=seeds,
            representation_classifier_pairs=representation_classifier_pairs,
        )
        selector_references.append((f"{location}.selector", selector))
        if comparison["method_a"] not in methods:
            raise StudyContractError(f"{location}.method_a must name a configured audit method")
        if comparison["method_b"] != "random_review":
            raise StudyContractError(f"{location}.method_b must be random_review")
        if comparison["metric"] not in supported_comparison_metrics:
            raise StudyContractError(f"{location}.metric is unsupported")
        review_budget = _probability(comparison["review_budget"], f"{location}.review_budget")
        if review_budget not in allowed_review_budgets:
            raise StudyContractError(f"{location}.review_budget is not a frozen evaluation budget")
        if comparison["direction"] != "method_a_minus_method_b":
            raise StudyContractError(f"{location}.direction must be method_a_minus_method_b")
        if comparison["holm_family"] not in holm_families:
            raise StudyContractError(f"{location}.holm_family is not declared")
    if len(set(comparison_ids)) != len(comparison_ids):
        raise StudyContractError("primary statistical comparison IDs must be globally unique")
    if statistics["exploratory_multiple_comparison_correction"] != "holm":
        raise StudyContractError("exploratory comparisons must use Holm correction")

    restoration = _mapping(resolved.get("restoration"), "restoration")
    _require_fields(
        restoration,
        (
            "enabled_cells",
            "ranking_method",
            "review_budget",
            "random_repeats",
            "random_seed",
            "include_reference_validation_in_training",
            "required_experiments",
            "downstream_comparisons",
        ),
        "restoration",
    )
    restoration_enabled_cells = _unique_strings(
        restoration["enabled_cells"], "restoration.enabled_cells"
    )
    if restoration["ranking_method"] not in methods:
        raise StudyContractError("restoration ranking_method must name a configured audit method")
    if _probability(restoration["review_budget"], "restoration.review_budget") != 0.05:
        raise StudyContractError("restoration review budget must be exactly 0.05")
    if _positive_int(restoration["random_repeats"], "restoration random repeats") < 100:
        raise StudyContractError("restoration random_repeats must be at least 100")
    _positive_int(restoration["random_seed"], "restoration random seed", minimum=0)
    if restoration["include_reference_validation_in_training"] is not True:
        raise StudyContractError("restoration training must include reference validation data")
    required_experiments = tuple(
        str(value)
        for value in _sequence(restoration["required_experiments"], "required experiments")
    )
    expected_experiments = (
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    )
    if required_experiments != expected_experiments:
        raise StudyContractError("restoration must freeze the four required experiment names")
    downstream_comparisons = _sequence(
        restoration["downstream_comparisons"],
        "restoration.downstream_comparisons",
    )
    if len(downstream_comparisons) != 1:
        raise StudyContractError(
            "restoration must freeze exactly audit-guided versus random downstream comparison"
        )
    downstream = _mapping(
        downstream_comparisons[0],
        "restoration.downstream_comparisons[0]",
    )
    if set(downstream) != {
        "comparison_id",
        "method_a",
        "method_b",
        "metric",
        "direction",
    }:
        raise StudyContractError(
            "downstream restoration comparison must contain exactly comparison_id, method_a, "
            "method_b, metric, and direction"
        )
    if downstream != {
        "comparison_id": "audit_guided_minus_random_macro_f1",
        "method_a": "audit_guided_restoration",
        "method_b": "random_review_restoration",
        "metric": "macro_f1",
        "direction": "method_a_minus_method_b",
    }:
        raise StudyContractError(
            "downstream restoration comparison must freeze audit-guided minus random macro_f1"
        )

    _, expanded_cells = _expand_primary_matrix_components(resolved)
    cell_by_id = {cell.cell_id: cell for cell in expanded_cells}
    cell_by_coordinates = {
        (
            cell.mechanism,
            cell.rate,
            cell.corruption_seed,
            cell.representation_id,
            cell.classifier_id,
        ): cell
        for cell in expanded_cells
    }

    def resolve_selector_cell(selector: tuple[str, ...], *, location: str) -> PrimaryCell | None:
        if selector[0] == "representation_classifier":
            return None
        if selector[0] == "cell_id":
            cell = cell_by_id.get(selector[1])
            if cell is None:
                raise StudyContractError(f"{location} names a cell_id absent from the matrix")
            return cell
        coordinates = (
            selector[1],
            float(selector[2]),
            int(selector[3]),
            selector[4],
            selector[5],
        )
        cell = cell_by_coordinates.get(coordinates)
        if cell is None:  # defensive: every expanded field was already validated above
            raise StudyContractError(f"{location} does not resolve to a frozen matrix cell")
        return cell

    for selector_location, selector in selector_references:
        selected_cell = resolve_selector_cell(selector, location=selector_location)
        if selected_cell is not None and selected_cell.rate == 0.0:
            raise StudyContractError(
                f"{selector_location} selects the 0% clean reference, where corruption-ranking "
                "inference is undefined"
            )

    for location, selector_a, selector_b, method_a, method_b in cross_cell_contracts:
        cell_a = resolve_selector_cell(selector_a, location=f"{location}.selector_a")
        cell_b = resolve_selector_cell(selector_b, location=f"{location}.selector_b")
        if cell_a is None or cell_b is None:
            if cell_a is not None or cell_b is not None:
                raise StudyContractError(
                    f"{location} cannot mix matrix-wide and explicitly expanded selectors"
                )
            changed_model_fields = sum(
                left != right for left, right in zip(selector_a[1:], selector_b[1:], strict=True)
            )
            if changed_model_fields != 1:
                raise StudyContractError(
                    f"{location} must vary exactly one representation/classifier factor"
                )
            continue
        if method_a != method_b:  # retained defensively beside the immediate schema check above
            raise StudyContractError(f"{location} must hold the audit method fixed")
        same_scenario = cell_a.scenario_id == cell_b.scenario_id
        if same_scenario:
            changed_model_fields = sum(
                (
                    cell_a.representation_id != cell_b.representation_id,
                    cell_a.classifier_id != cell_b.classifier_id,
                )
            )
            if changed_model_fields != 1:
                raise StudyContractError(
                    f"{location} must vary exactly one representation/classifier factor"
                )
            continue
        if (
            cell_a.corruption_seed != cell_b.corruption_seed
            or cell_a.representation_id != cell_b.representation_id
            or cell_a.classifier_id != cell_b.classifier_id
        ):
            raise StudyContractError(
                f"{location} controlled cross-scenario contrast must hold seed, "
                "representation, and classifier fixed"
            )
        changed_corruption_fields = sum(
            (
                cell_a.mechanism != cell_b.mechanism,
                cell_a.rate != cell_b.rate,
            )
        )
        if changed_corruption_fields != 1:
            raise StudyContractError(
                f"{location} controlled cross-scenario contrast must vary exactly one of "
                "mechanism or rate"
            )

    for index, cell_id in enumerate(restoration_enabled_cells):
        selected_cell = cell_by_id.get(cell_id)
        if selected_cell is None:
            raise StudyContractError(
                f"restoration.enabled_cells[{index}] names a cell absent from the matrix"
            )
        if selected_cell.rate == 0.0:
            raise StudyContractError(
                f"restoration.enabled_cells[{index}] cannot select the 0% clean reference"
            )
    return resolved


def _stable_id(prefix: str, payload: Mapping[str, Any], index: int) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{index:04d}_{hashlib.sha256(canonical).hexdigest()[:12]}"


def _expand_primary_matrix_components(
    resolved: Mapping[str, Any],
) -> tuple[tuple[PrimaryScenario, ...], tuple[PrimaryCell, ...]]:
    """Expand an already-resolved candidate without recursively validating it."""

    corruption = _mapping(resolved["corruption"], "corruption")
    rates = tuple(float(value) for value in corruption["rates"])
    seeds = tuple(int(value) for value in corruption["seeds"])
    scenarios: list[PrimaryScenario] = []
    cells: list[PrimaryCell] = []

    def append_scenario(*, scenario_id: str, mechanism: str, rate: float, seed: int) -> None:
        scenario_payload = {"mechanism": mechanism, "rate": rate, "seed": seed}
        scenarios.append(PrimaryScenario(scenario_id, mechanism, rate, seed))
        for raw_representation in resolved["representations"]:
            representation = _mapping(raw_representation, "representation")
            for classifier_id in representation["classifiers"]:
                cell_payload = {
                    **scenario_payload,
                    "representation_id": representation["id"],
                    "classifier_id": classifier_id,
                }
                cells.append(
                    PrimaryCell(
                        cell_id=_stable_id("primary", cell_payload, len(cells)),
                        scenario_id=scenario_id,
                        mechanism=mechanism,
                        rate=rate,
                        corruption_seed=seed,
                        representation_id=str(representation["id"]),
                        classifier_id=str(classifier_id),
                        required=bool(representation["required"]),
                    )
                )

    clean_reference = _mapping(corruption["clean_reference"], "corruption.clean_reference")
    append_scenario(
        scenario_id=str(clean_reference["id"]),
        mechanism=str(clean_reference["mechanism"]),
        rate=float(clean_reference["rate"]),
        seed=int(clean_reference["seed"]),
    )
    for mechanism in PRIMARY_CORRUPTION_MECHANISMS:
        for rate in rates:
            for seed in seeds:
                scenario_payload = {"mechanism": mechanism, "rate": rate, "seed": seed}
                scenario_id = _stable_id("scenario", scenario_payload, len(scenarios))
                append_scenario(
                    scenario_id=scenario_id,
                    mechanism=mechanism,
                    rate=rate,
                    seed=seed,
                )
    identifiers = [cell.cell_id for cell in cells]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("primary matrix expansion produced duplicate cell IDs")
    return tuple(scenarios), tuple(cells)


def build_primary_matrix_plan(config: Mapping[str, Any]) -> PrimaryMatrixPlan:
    """Expand a validated primary config into deterministic scenarios and cells."""

    resolved = validate_frozen_primary_config(config)
    scenarios, cells = _expand_primary_matrix_components(resolved)
    return PrimaryMatrixPlan(1, config_sha256(resolved), tuple(scenarios), tuple(cells))


def validate_frozen_confirmatory_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact confirmatory plan that must be frozen before primary outcomes."""

    resolved = resolve_config(config)
    _assert_no_outcome_dependent_values(resolved)
    _require_exact_fields(resolved, _CONFIRMATORY_TOP_LEVEL_FIELDS, "confirmatory config")
    if resolved.get("schema_version") != 2:
        raise StudyContractError("confirmatory schema_version must be exactly 2")
    if resolved.get("experiment_name") != "confirmatory_study":
        raise StudyContractError("confirmatory experiment_name is invalid")
    if str(resolved.get("status", "")).casefold() not in {
        "ready_for_freeze",
        "complete_for_freeze",
    }:
        raise StudyContractError(
            "confirmatory status must be READY_FOR_FREEZE or COMPLETE_FOR_FREEZE"
        )
    data = _mapping(resolved.get("data"), "data")
    _require_exact_fields(data, _CONFIRMATORY_DATA_FIELDS, "confirmatory.data")
    if data["source"] != "pannuke":
        raise StudyContractError("confirmatory data.source must be pannuke")
    authority_manifest_sha256, authority_sample_order_sha256, _ = _analysis_manifest_authority(
        data,
        location="confirmatory.data.analysis_manifest_authority",
    )
    folds = _unique_ints(data["official_folds"], "confirmatory official_folds", minimum=3)
    if folds != (1, 2, 3):
        raise StudyContractError("confirmatory official_folds must be exactly [1, 2, 3]")
    if data["group_unit"] != "source_patch_id":
        raise StudyContractError("confirmatory group_unit must be exactly source_patch_id")
    reference_fraction = _probability(
        data["reference_validation_fraction_groups"], "confirmatory reference fraction"
    )
    if reference_fraction != 0.10:
        raise StudyContractError(
            "confirmatory reference_validation_fraction_groups must be exactly 0.10"
        )
    if (
        data["reference_group_selection_algorithm"]
        != "deterministic_group_greedy_class_distribution_v1"
    ):
        raise StudyContractError(
            "confirmatory reference_group_selection_algorithm must be exactly "
            "deterministic_group_greedy_class_distribution_v1"
        )
    if _positive_int(data["split_seed"], "confirmatory split_seed", minimum=0) != 223:
        raise StudyContractError("confirmatory split_seed must be exactly 223")
    if data["fold_assignment_labels"] != "pre_corruption_label":
        raise StudyContractError(
            "confirmatory fold assignments must be fixed from pre_corruption_label"
        )

    corruption = _mapping(resolved.get("corruption"), "confirmatory corruption")
    _require_exact_fields(corruption, ("cells",), "confirmatory corruption")
    corruption_cells = _sequence(corruption.get("cells"), "confirmatory corruption.cells")
    if len(corruption_cells) != 2:
        raise StudyContractError(
            "confirmatory corruption.cells must contain exactly one clean and one nonzero cell"
        )
    corruption_ids: list[str] = []
    clean_cell_count = 0
    nonzero_cell_count = 0
    instance_auditor_families: set[str] = set()
    for index, raw_cell in enumerate(corruption_cells):
        cell = _mapping(raw_cell, f"corruption.cells[{index}]")
        cell_location = f"corruption.cells[{index}]"
        _require_exact_fields(
            cell,
            ("id", "mechanism", "rate", "seed", "parameters"),
            cell_location,
        )
        corruption_ids.append(_exact_identifier(cell["id"], f"corruption.cells[{index}].id"))
        mechanism = str(cell["mechanism"])
        if mechanism not in PRIMARY_CORRUPTION_MECHANISMS:
            raise StudyContractError("confirmatory corruption cell has an unsupported mechanism")
        rate = _probability(cell["rate"], f"corruption.cells[{index}].rate", allow_zero=True)
        seed = _positive_int(cell["seed"], f"corruption.cells[{index}].seed", minimum=0)
        if rate == 0.0:
            clean_cell_count += 1
            if cell != _CLEAN_REFERENCE_CELL:
                raise StudyContractError(
                    "confirmatory 0% cell must exactly match the frozen clean-reference cell"
                )
        else:
            nonzero_cell_count += 1
            if rate not in (0.05, 0.10, 0.20):
                raise StudyContractError(
                    "confirmatory nonzero corruption rate must be one of 0.05, 0.10, or 0.20"
                )
            if seed not in _PRIMARY_CORRUPTION_SEEDS:
                raise StudyContractError(
                    "confirmatory nonzero corruption seed must come from the primary seed set"
                )
        parameters = _mapping(cell["parameters"], f"{cell_location}.parameters")
        parameter_keys = set(parameters)
        if mechanism == "symmetric_random_corruption":
            if parameter_keys:
                raise StudyContractError(
                    "symmetric confirmatory corruption parameters must be exactly empty"
                )
        elif mechanism == "confusion_targeted_corruption":
            if parameter_keys != {"transition_matrix"}:
                raise StudyContractError(
                    "confusion-targeted confirmatory parameters must contain exactly "
                    "transition_matrix"
                )
            _validate_transition_matrix(
                parameters["transition_matrix"],
                len(CLASS_ORDER),
                location=f"{cell_location}.parameters.transition_matrix",
            )
        elif mechanism == "group_conditional_corruption":
            expected_group_parameters = {
                "grouping_field",
                "weights_by_value",
                "default_weight",
            }
            if parameter_keys != expected_group_parameters:
                raise StudyContractError(
                    "group-conditional confirmatory parameters must contain exactly "
                    "grouping_field, weights_by_value, and default_weight"
                )
            grouping_field = str(parameters["grouping_field"])
            if not grouping_field:
                raise StudyContractError(
                    "group-conditional confirmatory grouping_field must be explicit"
                )
            weights_by_value = _mapping(
                parameters["weights_by_value"],
                f"{cell_location}.parameters.weights_by_value",
            )
            if not weights_by_value or any(not str(key) for key in weights_by_value):
                raise StudyContractError(
                    "group-conditional confirmatory weights_by_value must be non-empty "
                    "with non-empty keys"
                )
            numeric_weights = tuple(
                _finite_float(
                    value,
                    f"{cell_location}.parameters.weights_by_value[{key!r}]",
                    minimum=0.0,
                )
                for key, value in weights_by_value.items()
            )
            default_weight = _finite_float(
                parameters["default_weight"],
                f"{cell_location}.parameters.default_weight",
                minimum=0.0,
            )
            if not any(value > 0.0 for value in (*numeric_weights, default_weight)):
                raise StudyContractError(
                    "group-conditional confirmatory weights cannot all be zero"
                )
        else:
            expected_instance_parameters = {
                "generator_representation",
                "auditor_representation_families",
                "independence_status",
                "independence_matrix_path",
                "independence_matrix_sha256",
            }
            if parameter_keys != expected_instance_parameters:
                raise StudyContractError(
                    "instance-dependent confirmatory parameters must contain exactly the "
                    "generator, auditor-family, and independence-evidence fields"
                )
            if not str(parameters["generator_representation"]):
                raise StudyContractError(
                    "instance-dependent confirmatory generator_representation must be explicit"
                )
            instance_auditor_families.update(
                _unique_strings(
                    parameters["auditor_representation_families"],
                    f"{cell_location}.parameters.auditor_representation_families",
                )
            )
            if parameters["independence_status"] != "verified_independent":
                raise StudyContractError(
                    "instance-dependent confirmatory corruption requires "
                    "independence_status=verified_independent"
                )
            if not str(parameters["independence_matrix_path"]):
                raise StudyContractError(
                    "instance-dependent confirmatory independence_matrix_path must be explicit"
                )
            if _SHA256.fullmatch(str(parameters["independence_matrix_sha256"])) is None:
                raise StudyContractError(
                    "instance-dependent confirmatory independence_matrix_sha256 must be a SHA-256"
                )
    if len(set(corruption_ids)) != len(corruption_ids):
        raise StudyContractError("confirmatory corruption cell IDs must be unique")
    if clean_cell_count != 1 or nonzero_cell_count != 1:
        raise StudyContractError(
            "confirmatory corruption must contain exactly one clean 0% cell and one nonzero cell"
        )

    scenarios = _sequence(resolved.get("scenarios"), "confirmatory scenarios")
    if not scenarios:
        raise StudyContractError("confirmatory scenarios must be non-empty")
    scenario_ids: list[str] = []
    representation_ids: list[str] = []
    scenario_by_id: dict[str, Mapping[str, Any]] = {}
    families: set[str] = set()
    inputs: set[str] = set()
    for index, raw_scenario in enumerate(scenarios):
        scenario = _mapping(raw_scenario, f"scenarios[{index}]")
        _require_fields(
            scenario,
            (
                "id",
                "representation_id",
                "cache_provenance_id",
                "family",
                "input_variant",
                "encoder",
                "classifier",
                "required",
            ),
            f"scenarios[{index}]",
        )
        scenario_id = _exact_identifier(scenario["id"], f"scenarios[{index}].id")
        representation_id = _exact_identifier(
            scenario["representation_id"],
            f"scenarios[{index}].representation_id",
        )
        _exact_identifier(
            scenario["cache_provenance_id"],
            f"scenarios[{index}].cache_provenance_id",
        )
        scenario_ids.append(scenario_id)
        representation_ids.append(representation_id)
        scenario_by_id[scenario_id] = scenario
        families.add(str(scenario["family"]))
        inputs.add(str(scenario["input_variant"]))
        if not isinstance(scenario["required"], bool):
            raise StudyContractError(f"scenarios[{index}].required must be boolean")
        if scenario["family"] == "pathology_frozen" and "availability_audit_sha256" not in scenario:
            raise StudyContractError("pathology scenario requires availability_audit_sha256")
        if (
            scenario["family"] == "pathology_frozen"
            and _SHA256.fullmatch(str(scenario["availability_audit_sha256"])) is None
        ):
            raise StudyContractError("pathology availability hash must be a SHA-256")
        if scenario["family"] == "pathology_frozen" and scenario["required"] is not False:
            raise StudyContractError("pathology scenario must be optional under availability gates")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise StudyContractError("confirmatory scenario IDs must be unique")
    if len(set(representation_ids)) != len(representation_ids):
        raise StudyContractError("confirmatory representation IDs must be unique")
    unknown_instance_families = instance_auditor_families.difference(families)
    if unknown_instance_families:
        raise StudyContractError(
            "instance-dependent auditor representation families are absent from scenarios: "
            f"{sorted(unknown_instance_families)}"
        )
    if "cnn" not in families:
        raise StudyContractError("confirmatory scenarios must include the CNN family")
    if not {"context_rgb", "context_rgb_plus_binary_target_mask"}.issubset(inputs):
        raise StudyContractError(
            "confirmatory scenarios must freeze RGB and RGB+target-mask inputs"
        )
    cnn_scenarios = [
        _mapping(value, "CNN scenario")
        for value in scenarios
        if isinstance(value, Mapping) and value.get("family") == "cnn"
    ]
    cnn_by_input = {str(value["input_variant"]): value for value in cnn_scenarios}
    if not {"context_rgb", "context_rgb_plus_binary_target_mask"}.issubset(cnn_by_input):
        raise StudyContractError("both target-indication CNN scenarios must be explicit")
    if len({str(value["classifier"]) for value in cnn_by_input.values()}) != 1:
        raise StudyContractError("target-indication CNN scenarios must use the same classifier")
    frozen_scenarios = [
        _mapping(value, "frozen scenario")
        for value in scenarios
        if isinstance(value, Mapping)
        and value.get("family") in {"imagenet_frozen", "pathology_frozen"}
    ]
    frozen_families = {str(value["family"]) for value in frozen_scenarios}
    if frozen_families != {"imagenet_frozen", "pathology_frozen"}:
        raise StudyContractError(
            "confirmatory scenarios must predeclare ImageNet and optional pathology frozen encoders"
        )
    if len({str(value["classifier"]) for value in frozen_scenarios}) != 1:
        raise StudyContractError("ImageNet/pathology comparison must use the same classifier")
    for scenario_id, expected in CONFIRMATORY_REQUIRED_FROZEN_ABLATIONS.items():
        ablation_scenario = scenario_by_id.get(scenario_id)
        if ablation_scenario is None:
            raise StudyContractError(
                f"confirmatory target-representation ablation requires scenario {scenario_id!r}"
            )
        if ablation_scenario["required"] is not True:
            raise StudyContractError(
                f"confirmatory target-representation ablation {scenario_id!r} must be required"
            )
        for field, expected_value in expected.items():
            if ablation_scenario[field] != expected_value:
                raise StudyContractError(
                    f"confirmatory ablation {scenario_id!r} must freeze {field}={expected_value!r}"
                )
    target_mask_cnn = scenario_by_id.get("cnn_context_target_mask")
    if target_mask_cnn is None or target_mask_cnn["required"] is not True:
        raise StudyContractError(
            "confirmatory target-representation ablation requires the RGB+mask CNN"
        )

    raw_cache_provenance = _sequence(
        resolved.get("cache_provenance"),
        "cache_provenance",
    )
    if not raw_cache_provenance:
        raise StudyContractError("cache_provenance must contain one record per scenario")
    cache_provenance_by_id: dict[str, Mapping[str, Any]] = {}
    available_cache_ids: set[str] = set()
    for index, raw_record in enumerate(raw_cache_provenance):
        location = f"cache_provenance[{index}]"
        record = _mapping(raw_record, location)
        _require_fields(
            record,
            ("id", "representation_id", "status", "input_variant"),
            location,
        )
        record_id = _exact_identifier(record["id"], f"{location}.id")
        if record_id in cache_provenance_by_id:
            raise StudyContractError("cache provenance IDs must be unique")
        representation_id = _exact_identifier(
            record["representation_id"],
            f"{location}.representation_id",
        )
        status = str(record["status"])
        if status == "available":
            expected_available_fields = {
                "id",
                "representation_id",
                "status",
                "cache_file_sha256",
                "sidecar_semantic_sha256",
                "sample_order_sha256",
                "manifest_sha256",
                "encoder_identifier",
                "encoder_metadata_sha256",
                "weight_identifier",
                "weights_sha256",
                "preprocessing_identifier",
                "preprocessing_sha256",
                "input_variant",
            }
            if set(record) != expected_available_fields:
                raise StudyContractError(
                    "available cache provenance must contain exactly the frozen cache/sidecar, "
                    "sample-order, manifest, encoder, weights, preprocessing, and input fields"
                )
            cache_file = record["cache_file_sha256"]
            sidecar = record["sidecar_semantic_sha256"]
            if cache_file is not None:
                _sha256(cache_file, f"{location}.cache_file_sha256")
            if sidecar is not None:
                _sha256(sidecar, f"{location}.sidecar_semantic_sha256")
            if (cache_file is None) == (sidecar is None):
                raise StudyContractError(
                    "available cache provenance requires exactly one valid cache_file_sha256 "
                    "or sidecar_semantic_sha256"
                )
            for field in (
                "sample_order_sha256",
                "manifest_sha256",
                "encoder_metadata_sha256",
                "weights_sha256",
                "preprocessing_sha256",
            ):
                _sha256(record[field], f"{location}.{field}")
            for field in (
                "encoder_identifier",
                "weight_identifier",
                "preprocessing_identifier",
                "input_variant",
            ):
                if not str(record[field]).strip():
                    raise StudyContractError(f"{location}.{field} must be explicit")
            available_cache_ids.add(record_id)
        elif status == "unavailable_with_frozen_blocker":
            expected_unavailable_fields = {
                "id",
                "representation_id",
                "status",
                "sample_order_sha256",
                "manifest_sha256",
                "encoder_identifier",
                "input_variant",
                "blocker_evidence_sha256",
            }
            if set(record) != expected_unavailable_fields:
                raise StudyContractError(
                    "unavailable cache provenance must contain exactly its representation, "
                    "sample-order, manifest, encoder, input, and frozen blocker evidence"
                )
            for field in (
                "sample_order_sha256",
                "manifest_sha256",
                "blocker_evidence_sha256",
            ):
                _sha256(record[field], f"{location}.{field}")
            if (
                not str(record["encoder_identifier"]).strip()
                or not str(record["input_variant"]).strip()
            ):
                raise StudyContractError(
                    "unavailable cache provenance encoder and input_variant must be explicit"
                )
        else:
            raise StudyContractError(
                "cache provenance status must be available or unavailable_with_frozen_blocker"
            )
        if (
            record["manifest_sha256"] != authority_manifest_sha256
            or record["sample_order_sha256"] != authority_sample_order_sha256
        ):
            raise StudyContractError(
                f"{location} differs from confirmatory.data.analysis_manifest_authority"
            )
        cache_provenance_by_id[record_id] = {
            **record,
            "representation_id": representation_id,
        }

    scenario_cache_ids: list[str] = []
    for scenario_id, scenario in scenario_by_id.items():
        provenance_id = str(scenario["cache_provenance_id"])
        scenario_cache_ids.append(provenance_id)
        if provenance_id not in cache_provenance_by_id:
            raise StudyContractError(
                f"scenario {scenario_id!r} references missing cache provenance"
            )
        provenance = cache_provenance_by_id[provenance_id]
        if provenance["representation_id"] != scenario["representation_id"]:
            raise StudyContractError(
                f"scenario {scenario_id!r} representation differs from cache provenance"
            )
        if provenance["input_variant"] != scenario["input_variant"]:
            raise StudyContractError(
                f"scenario {scenario_id!r} input_variant differs from cache provenance"
            )
        if provenance["encoder_identifier"] != scenario["encoder"]:
            raise StudyContractError(
                f"scenario {scenario_id!r} encoder differs from cache provenance"
            )
        if scenario["required"] is True and provenance_id not in available_cache_ids:
            raise StudyContractError(
                f"required scenario {scenario_id!r} lacks available frozen cache provenance"
            )
        if provenance_id not in available_cache_ids:
            blocker = provenance.get("blocker_evidence_sha256")
            if blocker != scenario.get("availability_audit_sha256"):
                raise StudyContractError(
                    f"optional scenario {scenario_id!r} cache blocker differs from its frozen "
                    "availability audit"
                )
    if len(set(scenario_cache_ids)) != len(scenario_cache_ids):
        raise StudyContractError("each confirmatory scenario must use a distinct cache record")
    if set(scenario_cache_ids) != set(cache_provenance_by_id):
        raise StudyContractError(
            "cache_provenance must contain exactly one referenced record per scenario"
        )

    model_seeds = _unique_ints(resolved.get("model_seeds"), "model_seeds", minimum=3)
    if model_seeds != (303, 304, 305):
        raise StudyContractError("confirmatory model_seeds must be exactly [303, 304, 305]")
    training = _mapping(resolved.get("training"), "training")
    _require_fields(
        training,
        (
            "optimizer",
            "input_size",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "early_stopping_source",
            "initial_batch_size",
            "minimum_batch_size",
            "gradient_accumulation_steps",
            "class_weight",
            "oom_policy",
            "amp",
            "amp_dtype",
            "checkpoint_resume",
            "cuda_required",
        ),
        "training",
    )
    if training["early_stopping_source"] != "reference_validation_only":
        raise StudyContractError("confirmatory early stopping must use reference validation only")
    if training["oom_policy"] != "halve_batch_and_retry_same_samples":
        raise StudyContractError("confirmatory OOM policy must halve batch and retry same samples")
    for required_true in ("amp", "checkpoint_resume", "cuda_required"):
        if training[required_true] is not True:
            raise StudyContractError(f"training.{required_true} must be true")
    initial_batch = _positive_int(training["initial_batch_size"], "initial_batch_size")
    minimum_batch = _positive_int(training["minimum_batch_size"], "minimum_batch_size")
    if minimum_batch > initial_batch:
        raise StudyContractError("minimum_batch_size cannot exceed initial_batch_size")
    _positive_int(training["input_size"], "training.input_size", minimum=32)
    _positive_int(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
    )
    if training["class_weight"] != "balanced":
        raise StudyContractError("confirmatory training.class_weight must be balanced")
    if str(training["optimizer"]).casefold() != "adamw":
        raise StudyContractError("confirmatory optimizer must be exactly adamw")
    _finite_float(
        training["learning_rate"],
        "training.learning_rate",
        minimum=0.0,
        strict_minimum=True,
    )
    _finite_float(training["weight_decay"], "training.weight_decay", minimum=0.0)
    _positive_int(training["max_epochs"], "training.max_epochs")
    _positive_int(training["early_stopping_patience"], "training.early_stopping_patience")
    _finite_float(
        training["early_stopping_min_delta"],
        "training.early_stopping_min_delta",
        minimum=0.0,
    )
    if str(training["amp_dtype"]) not in {"float16", "bfloat16"}:
        raise StudyContractError("training.amp_dtype must be float16 or bfloat16")

    oof = _mapping(resolved.get("oof"), "confirmatory oof")
    _require_fields(oof, ("n_splits", "split_kind", "no_nucleus_level_fallback"), "oof")
    _positive_int(oof["n_splits"], "confirmatory oof.n_splits", minimum=2)
    if oof["split_kind"] != "stratified_group" or oof["no_nucleus_level_fallback"] is not True:
        raise StudyContractError(
            "confirmatory OOF must be stratified-group with no nucleus fallback"
        )

    original_audit = _mapping(
        resolved.get("original_audit_selection"),
        "original_audit_selection",
    )
    expected_original_audit_fields = {
        "scenario_id",
        "representation_id",
        "model_seed",
        "risk_method",
        "n_splits",
        "classifier",
        "cache_provenance_id",
    }
    if set(original_audit) != expected_original_audit_fields:
        raise StudyContractError(
            "original_audit_selection must contain exactly scenario_id, representation_id, "
            "model_seed, risk_method, n_splits, classifier, and cache_provenance_id"
        )
    original_scenario_id = _exact_identifier(
        original_audit["scenario_id"],
        "original_audit_selection.scenario_id",
    )
    if original_scenario_id not in scenario_by_id:
        raise StudyContractError("original_audit_selection.scenario_id must name a frozen scenario")
    original_scenario = scenario_by_id[original_scenario_id]
    if original_scenario["required"] is not True:
        raise StudyContractError("original_audit_selection cannot depend on an optional scenario")
    if original_scenario["family"] not in {"imagenet_frozen", "pathology_frozen"}:
        raise StudyContractError("original_audit_selection must use a frozen-feature scenario")
    original_representation_id = _exact_identifier(
        original_audit["representation_id"],
        "original_audit_selection.representation_id",
    )
    if original_representation_id != str(original_scenario["representation_id"]):
        raise StudyContractError(
            "original_audit_selection representation_id must match its frozen scenario"
        )
    original_model_seed = _positive_int(
        original_audit["model_seed"],
        "original_audit_selection.model_seed",
        minimum=0,
    )
    if original_model_seed not in model_seeds:
        raise StudyContractError(
            "original_audit_selection model_seed must name a frozen model seed"
        )
    original_risk_method = _exact_identifier(
        original_audit["risk_method"],
        "original_audit_selection.risk_method",
    )
    if original_risk_method not in ORIGINAL_AUDIT_RISK_METHODS:
        raise StudyContractError(
            "original_audit_selection risk_method is unsupported by the frozen workflow"
        )
    _positive_int(
        original_audit["n_splits"],
        "original_audit_selection.n_splits",
        minimum=2,
    )

    original_classifier = _mapping(
        original_audit["classifier"],
        "original_audit_selection.classifier",
    )
    if set(original_classifier) != {"id", "parameters"}:
        raise StudyContractError(
            "original_audit_selection.classifier must contain exactly id and parameters"
        )
    original_classifier_id = _exact_identifier(
        original_classifier["id"],
        "original_audit_selection.classifier.id",
    )
    if original_classifier_id != str(original_scenario["classifier"]):
        raise StudyContractError(
            "original_audit_selection classifier must match its frozen scenario"
        )
    if original_classifier_id != "multinomial_logistic_regression":
        raise StudyContractError(
            "original_audit_selection currently requires a frozen-feature multinomial "
            "logistic-regression scenario"
        )
    classifier_parameters = _mapping(
        original_classifier["parameters"],
        "original_audit_selection.classifier.parameters",
    )
    if set(classifier_parameters) != {"l2", "max_iter", "class_weight"}:
        raise StudyContractError(
            "original-audit logistic parameters must contain exactly l2, max_iter, and class_weight"
        )
    _finite_float(
        classifier_parameters["l2"],
        "original_audit_selection.classifier.parameters.l2",
        minimum=0.0,
    )
    _positive_int(
        classifier_parameters["max_iter"],
        "original_audit_selection.classifier.parameters.max_iter",
    )
    if classifier_parameters["class_weight"] != "balanced":
        raise StudyContractError("original-audit logistic class_weight must be balanced")
    original_cache_provenance_id = _exact_identifier(
        original_audit["cache_provenance_id"],
        "original_audit_selection.cache_provenance_id",
    )
    if original_cache_provenance_id != str(original_scenario["cache_provenance_id"]):
        raise StudyContractError(
            "original_audit_selection cache_provenance_id must match its frozen scenario"
        )
    if original_cache_provenance_id not in available_cache_ids:
        raise StudyContractError(
            "original_audit_selection requires an available frozen cache provenance record"
        )

    ensemble = _mapping(resolved.get("ensemble"), "ensemble")
    _require_fields(ensemble, ("members", "primary_risk", "secondary_risks"), "ensemble")
    members = _sequence(ensemble["members"], "ensemble.members")
    if len(members) < 2:
        raise StudyContractError("confirmatory ensemble must freeze at least two members")
    member_keys: list[tuple[str, int]] = []
    for index, raw_member in enumerate(members):
        member = _mapping(raw_member, f"ensemble.members[{index}]")
        if set(member) != {"scenario_id", "model_seed"}:
            raise StudyContractError(
                "each confirmatory ensemble member must contain exactly scenario_id and model_seed"
            )
        scenario_id = _exact_identifier(
            member["scenario_id"], f"ensemble.members[{index}].scenario_id"
        )
        model_seed = _positive_int(
            member["model_seed"],
            f"ensemble.members[{index}].model_seed",
            minimum=0,
        )
        if scenario_id not in scenario_by_id:
            raise StudyContractError(f"ensemble member scenario is not frozen: {scenario_id!r}")
        if scenario_by_id[scenario_id]["required"] is not True:
            raise StudyContractError(
                "ensemble members must use required scenarios, not availability-gated optional "
                "scenarios"
            )
        if model_seed not in model_seeds:
            raise StudyContractError(f"ensemble member model seed is not frozen: {model_seed}")
        member_keys.append((scenario_id, model_seed))
    if len(set(member_keys)) != len(member_keys):
        raise StudyContractError("confirmatory ensemble members must be unique")
    primary_ensemble_risk = str(ensemble["primary_risk"])
    if primary_ensemble_risk not in CONFIRMATORY_ENSEMBLE_RISKS:
        raise StudyContractError("ensemble.primary_risk is unsupported")
    secondary_ensemble_risks = _unique_strings(
        ensemble["secondary_risks"], "ensemble.secondary_risks"
    )
    unknown_secondary_risks = set(secondary_ensemble_risks).difference(CONFIRMATORY_ENSEMBLE_RISKS)
    if unknown_secondary_risks:
        raise StudyContractError(
            f"ensemble.secondary_risks contain unsupported risks: {sorted(unknown_secondary_risks)}"
        )
    if primary_ensemble_risk in secondary_ensemble_risks:
        raise StudyContractError("ensemble primary_risk must not be repeated in secondary_risks")

    hybrid = _mapping(resolved.get("fixed_hybrid"), "fixed_hybrid")
    _require_fields(hybrid, ("components", "weights", "drop_one_ablations"), "fixed_hybrid")
    components = _unique_strings(hybrid["components"], "confirmatory hybrid components", minimum=2)
    weights = tuple(
        _probability(item, f"confirmatory hybrid weights[{index}]")
        for index, item in enumerate(_sequence(hybrid["weights"], "hybrid weights"))
    )
    if len(components) != len(weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise StudyContractError("confirmatory hybrid weights must align and sum to one")
    equal_weight = 1.0 / len(components)
    if any(abs(weight - equal_weight) > 1e-12 for weight in weights):
        raise StudyContractError(
            "confirmatory fixed hybrid must use equal weights across its frozen components"
        )
    if set(str(value) for value in hybrid["drop_one_ablations"]) != set(components):
        raise StudyContractError("drop_one_ablations must cover every fixed hybrid component")

    restoration = _mapping(resolved.get("restoration"), "confirmatory restoration")
    _require_fields(
        restoration,
        (
            "scenario_id",
            "model_seed",
            "representation_id",
            "ranking_method",
            "review_budget",
            "random_repeats",
            "random_seed",
            "conditions",
        ),
        "restoration",
    )
    restoration_scenario_id = _exact_identifier(
        restoration["scenario_id"], "restoration.scenario_id"
    )
    if restoration_scenario_id not in scenario_by_id:
        raise StudyContractError("restoration scenario_id must name a frozen scenario")
    restoration_scenario = scenario_by_id[restoration_scenario_id]
    if restoration_scenario["required"] is not True:
        raise StudyContractError("restoration cannot depend on an optional scenario")
    restoration_model_seed = _positive_int(
        restoration["model_seed"], "restoration.model_seed", minimum=0
    )
    if restoration_model_seed not in model_seeds:
        raise StudyContractError("restoration model_seed must name a frozen model seed")
    restoration_representation_id = _exact_identifier(
        restoration["representation_id"], "restoration.representation_id"
    )
    if restoration_representation_id != str(restoration_scenario["representation_id"]):
        raise StudyContractError(
            "restoration representation_id must exactly match its frozen scenario"
        )
    restoration_ranking_method = _exact_identifier(
        restoration["ranking_method"], "restoration.ranking_method"
    )
    available_restoration_risks = {
        "fixed_hybrid",
        *components,
        primary_ensemble_risk,
        *secondary_ensemble_risks,
    }
    if restoration_ranking_method not in available_restoration_risks:
        raise StudyContractError(
            "restoration ranking_method is not produced by the frozen confirmatory plan"
        )
    _probability(restoration["review_budget"], "confirmatory restoration budget")
    if _positive_int(restoration["random_repeats"], "confirmatory random repeats") < 100:
        raise StudyContractError("confirmatory random restoration needs at least 100 repeats")
    _positive_int(restoration["random_seed"], "confirmatory restoration.random_seed", minimum=0)
    expected_conditions = {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    }
    if set(str(value) for value in restoration["conditions"]) != expected_conditions:
        raise StudyContractError("confirmatory restoration must include exactly four conditions")

    statistics = _mapping(resolved.get("statistics"), "confirmatory statistics")
    _require_fields(
        statistics,
        (
            "paired_group_bootstrap_iterations",
            "bootstrap_seed",
            "preregistered_paired_comparisons",
            "holm_families",
        ),
        "statistics",
    )
    if (
        _positive_int(statistics["paired_group_bootstrap_iterations"], "bootstrap iterations")
        < 2000
    ):
        raise StudyContractError("confirmatory paired bootstrap requires at least 2000 iterations")
    _positive_int(statistics["bootstrap_seed"], "bootstrap seed", minimum=0)
    holm_families = tuple(
        _exact_identifier(value, f"statistics.holm_families[{index}]")
        for index, value in enumerate(
            _sequence(statistics["holm_families"], "statistics.holm_families")
        )
    )
    if not holm_families or len(set(holm_families)) != len(holm_families):
        raise StudyContractError("statistics.holm_families must be non-empty and unique")
    raw_comparisons = _sequence(
        statistics["preregistered_paired_comparisons"],
        "statistics.preregistered_paired_comparisons",
    )
    if not raw_comparisons:
        raise StudyContractError("statistics must freeze at least one paired comparison")
    comparison_ids: list[str] = []
    referenced_holm_families: set[str] = set()

    def validate_operand(value: Any, location: str) -> tuple[str, str, str, str, str, str, str]:
        operand = _mapping(value, location)
        expected_operand_fields = {
            "scenario_id",
            "representation_id",
            "classifier_id",
            "risk_id",
            "model_seed",
            "outer_fold",
            "corruption_cell",
        }
        if set(operand) != expected_operand_fields:
            raise StudyContractError(
                f"{location} must contain exactly scenario_id, representation_id, "
                "classifier_id, risk_id, model_seed, outer_fold, and corruption_cell"
            )
        scenario_id = _exact_identifier(operand["scenario_id"], f"{location}.scenario_id")
        if scenario_id not in scenario_by_id:
            raise StudyContractError(f"{location}.scenario_id must name a frozen scenario")
        scenario = scenario_by_id[scenario_id]
        representation_id = _exact_identifier(
            operand["representation_id"], f"{location}.representation_id"
        )
        if representation_id != str(scenario["representation_id"]):
            raise StudyContractError(f"{location}.representation_id must match its frozen scenario")
        classifier_id = _exact_identifier(operand["classifier_id"], f"{location}.classifier_id")
        if classifier_id != str(scenario["classifier"]):
            raise StudyContractError(f"{location}.classifier_id must match its frozen scenario")
        risk_id = _exact_identifier(operand["risk_id"], f"{location}.risk_id")
        if risk_id not in CONFIRMATORY_COMPARISON_RISK_IDS:
            raise StudyContractError(f"{location}.risk_id is not a frozen confirmatory risk")
        if operand["model_seed"] != "matched":
            raise StudyContractError(f"{location}.model_seed must be matched")
        raw_outer_fold = operand["outer_fold"]
        if raw_outer_fold == "all_matched":
            outer_fold_selector = "all_matched"
        else:
            exact_outer_fold = _positive_int(raw_outer_fold, f"{location}.outer_fold")
            if exact_outer_fold not in folds:
                raise StudyContractError(f"{location}.outer_fold is not frozen")
            outer_fold_selector = str(exact_outer_fold)
        raw_corruption_cell = operand["corruption_cell"]
        if raw_corruption_cell == "all_matched":
            corruption_selector = "all_matched"
        else:
            corruption_selector = _exact_identifier(
                raw_corruption_cell, f"{location}.corruption_cell"
            )
            if corruption_selector not in corruption_ids:
                raise StudyContractError(f"{location}.corruption_cell is not frozen")
        return (
            scenario_id,
            representation_id,
            classifier_id,
            risk_id,
            "matched",
            outer_fold_selector,
            corruption_selector,
        )

    for index, raw_comparison in enumerate(raw_comparisons):
        location = f"statistics.preregistered_paired_comparisons[{index}]"
        comparison = _mapping(raw_comparison, location)
        expected_comparison_fields = {
            "comparison_id",
            "metric",
            "operand_a",
            "operand_b",
            "direction",
            "holm_family",
        }
        if set(comparison) != expected_comparison_fields:
            raise StudyContractError(
                "each preregistered paired comparison must contain exactly comparison_id, "
                "metric, operand_a, operand_b, direction, and holm_family"
            )
        comparison_ids.append(
            _exact_identifier(
                comparison["comparison_id"],
                f"{location}.comparison_id",
            )
        )
        metric = str(comparison["metric"])
        if metric not in {"average_precision", "macro_f1"}:
            raise StudyContractError(
                "paired comparison metric must be average_precision or macro_f1"
            )
        operand_a = validate_operand(comparison["operand_a"], f"{location}.operand_a")
        operand_b = validate_operand(comparison["operand_b"], f"{location}.operand_b")
        if operand_a == operand_b:
            raise StudyContractError("paired comparison operands must differ")
        if operand_a[5:] != operand_b[5:]:
            raise StudyContractError(
                "paired comparison operands must use matched outer-fold and corruption selectors"
            )
        if comparison["direction"] != "method_a_minus_method_b":
            raise StudyContractError(
                "paired comparison direction must be exactly method_a_minus_method_b"
            )
        holm_family = _exact_identifier(
            comparison["holm_family"],
            f"{location}.holm_family",
        )
        if holm_family not in holm_families:
            raise StudyContractError(
                "paired comparison holm_family is absent from statistics.holm_families"
            )
        referenced_holm_families.add(holm_family)
    if len(set(comparison_ids)) != len(comparison_ids):
        raise StudyContractError("paired comparison IDs must be unique")
    if referenced_holm_families != set(holm_families):
        raise StudyContractError(
            "every declared Holm family must be referenced by a paired comparison"
        )

    rotation = _mapping(resolved.get("fold_rotation"), "fold_rotation")
    _require_fields(rotation, ("enabled", "feasibility_rule", "aggregate_policy"), "fold_rotation")
    if rotation["enabled"] is not True:
        raise StudyContractError("all feasible official-fold rotations must be enabled")
    if not str(rotation["feasibility_rule"]):
        raise StudyContractError("fold_rotation.feasibility_rule must be exact")
    if rotation["aggregate_policy"] != "report_each_rotation_and_descriptive_fold_mean":
        raise StudyContractError("confirmatory rotations require separate reporting")
    return resolved


def validate_resource_bounded_confirmatory_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact amended sensitivity profile without weakening schema-v2.

    This is deliberately a separate schema/profile contract.  Its semantic digest
    freezes every nested choice, while the explicit checks below make the
    non-confirmatory disposition and the principal resource bounds auditable.
    """

    resolved = resolve_config(config)
    _assert_no_outcome_dependent_values(resolved)
    _require_exact_fields(
        resolved,
        _RESOURCE_BOUNDED_CONFIRMATORY_TOP_LEVEL_FIELDS,
        "resource-bounded confirmatory config",
    )
    if resolved.get("schema_version") != 3:
        raise StudyContractError("resource-bounded confirmatory schema_version must be exactly 3")
    if resolved.get("experiment_name") != "confirmatory_resource_bounded_study":
        raise StudyContractError("resource-bounded confirmatory experiment_name is invalid")
    if resolved.get("status") != "READY_FOR_AMENDMENT":
        raise StudyContractError(
            "resource-bounded confirmatory status must be exactly READY_FOR_AMENDMENT"
        )
    if resolved.get("execution_profile") != RESOURCE_BOUNDED_CONFIRMATORY_PROFILE:
        raise StudyContractError("resource-bounded confirmatory execution_profile is invalid")
    if resolved.get("analysis_disposition") != "amended_or_exploratory":
        raise StudyContractError(
            "resource-bounded confirmatory analysis_disposition must be exactly "
            "amended_or_exploratory"
        )
    if resolved.get("original_confirmatory_claim_allowed") is not False:
        raise StudyContractError(
            "resource-bounded confirmatory original_confirmatory_claim_allowed must be false"
        )
    if resolved.get("completion_stage") is not None:
        raise StudyContractError("resource-bounded confirmatory completion_stage must remain null")

    data = _mapping(resolved.get("data"), "resource-bounded confirmatory.data")
    _require_exact_fields(
        data,
        _CONFIRMATORY_DATA_FIELDS,
        "resource-bounded confirmatory.data",
    )
    if data.get("source") != "pannuke":
        raise StudyContractError("resource-bounded confirmatory data.source must be pannuke")
    _analysis_manifest_authority(
        data,
        location="resource-bounded confirmatory.data.analysis_manifest_authority",
    )
    if tuple(data.get("official_folds", ())) != (1, 2, 3):
        raise StudyContractError(
            "resource-bounded confirmatory official_folds must be exactly [1, 2, 3]"
        )
    if data.get("group_unit") != "source_patch_id":
        raise StudyContractError("resource-bounded confirmatory group_unit must be source_patch_id")
    if data.get("fold_assignment_labels") != "pre_corruption_label":
        raise StudyContractError(
            "resource-bounded confirmatory fold assignments must use pre_corruption_label"
        )

    corruption = _mapping(
        resolved.get("corruption"),
        "resource-bounded confirmatory.corruption",
    )
    cells = _sequence(
        corruption.get("cells"),
        "resource-bounded confirmatory.corruption.cells",
    )
    if tuple(str(_mapping(cell, "corruption cell").get("id")) for cell in cells) != (
        "clean_reference_cell",
        "confusion_targeted_ten_percent",
    ):
        raise StudyContractError(
            "resource-bounded confirmatory requires the exact two frozen corruption cells"
        )

    scenarios = _sequence(
        resolved.get("scenarios"),
        "resource-bounded confirmatory.scenarios",
    )
    expected_scenarios = (
        "cnn_context_rgb",
        "imagenet_frozen_logistic",
        "imagenet_frozen_target_highlighted_logistic",
        "imagenet_frozen_context_morphometrics_logistic",
    )
    if tuple(str(_mapping(item, "scenario").get("id")) for item in scenarios) != (
        expected_scenarios
    ):
        raise StudyContractError(
            "resource-bounded confirmatory scenarios differ from the exact "
            "one-CNN/three-frozen profile"
        )
    if any(_mapping(item, "scenario").get("required") is not True for item in scenarios):
        raise StudyContractError("every resource-bounded confirmatory scenario must be required")
    if any(
        _mapping(item, "scenario").get("family") == "pathology_frozen"
        or _mapping(item, "scenario").get("input_variant") == "context_rgb_plus_binary_target_mask"
        for item in scenarios
    ):
        raise StudyContractError(
            "resource-bounded confirmatory excludes pathology and target-mask scenarios"
        )
    if tuple(resolved.get("model_seeds", ())) != (303,):
        raise StudyContractError("resource-bounded confirmatory model_seeds must be exactly [303]")

    training = _mapping(
        resolved.get("training"),
        "resource-bounded confirmatory.training",
    )
    if training.get("max_epochs") != 4 or training.get("early_stopping_patience") != 2:
        raise StudyContractError(
            "resource-bounded confirmatory training must use max_epochs=4 and "
            "early_stopping_patience=2"
        )
    oof = _mapping(resolved.get("oof"), "resource-bounded confirmatory.oof")
    if (
        oof.get("n_splits") != 5
        or oof.get("split_kind") != "stratified_group"
        or oof.get("no_nucleus_level_fallback") is not True
    ):
        raise StudyContractError(
            "resource-bounded confirmatory OOF must remain five-fold, "
            "stratified-group, and fail closed on nucleus-level fallback"
        )

    ensemble = _mapping(
        resolved.get("ensemble"),
        "resource-bounded confirmatory.ensemble",
    )
    ensemble_members = tuple(
        (
            str(_mapping(item, "ensemble member").get("scenario_id")),
            _mapping(item, "ensemble member").get("model_seed"),
        )
        for item in _sequence(ensemble.get("members"), "ensemble.members")
    )
    if ensemble_members != tuple((scenario_id, 303) for scenario_id in expected_scenarios[1:]):
        raise StudyContractError(
            "resource-bounded confirmatory ensemble must contain exactly the three "
            "frozen scenarios at seed 303"
        )

    restoration = _mapping(
        resolved.get("restoration"),
        "resource-bounded confirmatory.restoration",
    )
    if (
        restoration.get("scenario_id") != "imagenet_frozen_target_highlighted_logistic"
        or restoration.get("model_seed") != 303
        or restoration.get("ranking_method") != "fixed_hybrid"
        or restoration.get("random_repeats") != 100
    ):
        raise StudyContractError(
            "resource-bounded confirmatory restoration differs from the exact "
            "highlighted seed-303 profile"
        )

    statistics = _mapping(
        resolved.get("statistics"),
        "resource-bounded confirmatory.statistics",
    )
    if statistics.get("paired_group_bootstrap_iterations") != 2000:
        raise StudyContractError(
            "resource-bounded confirmatory requires exactly 2000 group bootstraps"
        )
    comparison_ids = tuple(
        str(_mapping(item, "paired comparison").get("comparison_id"))
        for item in _sequence(
            statistics.get("preregistered_paired_comparisons"),
            "statistics.preregistered_paired_comparisons",
        )
    )
    if comparison_ids != (
        "highlighted_frozen_minus_context_frozen",
        "context_morphometrics_minus_context_frozen",
        "ensemble_disagreement_minus_self_confidence",
        "fixed_hybrid_minus_drop_self_confidence",
        "fixed_hybrid_minus_drop_ensemble_disagreement",
        "cnn_context_minus_imagenet_context",
    ):
        raise StudyContractError(
            "resource-bounded confirmatory paired comparisons differ from the exact profile"
        )
    if tuple(statistics.get("holm_families", ())) != (
        "target_representation_family",
        "ranking_method_family",
        "model_family",
    ):
        raise StudyContractError(
            "resource-bounded confirmatory Holm families differ from the exact profile"
        )

    rotation = _mapping(
        resolved.get("fold_rotation"),
        "resource-bounded confirmatory.fold_rotation",
    )
    if (
        rotation.get("enabled") is not True
        or rotation.get("aggregate_policy") != "report_each_rotation_and_descriptive_fold_mean"
    ):
        raise StudyContractError(
            "resource-bounded confirmatory requires all three separately reported rotations"
        )

    semantic_sha256 = config_sha256(resolved)
    if semantic_sha256 != RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256:
        raise StudyContractError(
            "resource-bounded confirmatory config differs from the exact registered "
            f"{RESOURCE_BOUNDED_CONFIRMATORY_PROFILE} contract"
        )
    return resolved


def validate_confirmatory_execution_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch only to one of the two non-overlapping confirmatory contracts."""

    resolved = resolve_config(config)
    if resolved.get("schema_version") == 2:
        return validate_frozen_confirmatory_config(resolved)
    if (
        resolved.get("schema_version") == 3
        and resolved.get("execution_profile") == RESOURCE_BOUNDED_CONFIRMATORY_PROFILE
    ):
        return validate_resource_bounded_confirmatory_config(resolved)
    raise StudyContractError(
        "confirmatory execution config must be frozen schema-v2 or the exact "
        f"{RESOURCE_BOUNDED_CONFIRMATORY_PROFILE} schema-v3 profile"
    )


def validate_primary_confirmatory_cross_config(
    primary_config: Mapping[str, Any],
    confirmatory_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate strict configs and their shared immutable study bindings."""

    primary = validate_frozen_primary_config(primary_config)
    confirmatory = validate_frozen_confirmatory_config(confirmatory_config)
    primary_data = primary["data"]
    confirmatory_data = confirmatory["data"]
    shared_fields = (
        "analysis_manifest_authority",
        "group_unit",
        "reference_validation_fraction_groups",
        "reference_group_selection_algorithm",
        "split_seed",
        "fold_assignment_labels",
    )
    for field_name in shared_fields:
        if primary_data[field_name] != confirmatory_data[field_name]:
            raise StudyContractError(f"primary/confirmatory data.{field_name} bindings differ")

    nonzero_cells = [
        cell for cell in confirmatory["corruption"]["cells"] if float(cell["rate"]) > 0.0
    ]
    if len(nonzero_cells) != 1:
        raise StudyContractError(
            "confirmatory config must select exactly one nonzero corruption cell"
        )
    selected = nonzero_cells[0]
    primary_confusion = primary["corruption"]["mechanisms"]["confusion_targeted_corruption"]
    if (
        selected["id"] != "confusion_targeted_ten_percent"
        or selected["mechanism"] != "confusion_targeted_corruption"
        or float(selected["rate"]) != 0.10
        or int(selected["seed"]) != 404
        or selected["parameters"] != primary_confusion
    ):
        raise StudyContractError(
            "confirmatory nonzero corruption cell is not exactly bound to the primary plan"
        )

    primary_plan = build_primary_matrix_plan(primary)
    if not any(
        cell.mechanism == selected["mechanism"]
        and cell.rate == selected["rate"]
        and cell.corruption_seed == selected["seed"]
        for cell in primary_plan.cells
    ):
        raise StudyContractError(
            "confirmatory nonzero corruption cell is absent from the primary matrix"
        )
    return primary, confirmatory


def build_confirmatory_matrix_plan(config: Mapping[str, Any]) -> ConfirmatoryMatrixPlan:
    """Expand an exact confirmatory plan before any model outcome is computed."""

    resolved = validate_confirmatory_execution_config(config)
    cells: list[ConfirmatoryCell] = []
    for outer_fold in resolved["data"]["official_folds"]:
        for corruption_cell in resolved["corruption"]["cells"]:
            for scenario in resolved["scenarios"]:
                for model_seed in resolved["model_seeds"]:
                    payload = {
                        "outer_fold": int(outer_fold),
                        "corruption_cell_id": str(corruption_cell["id"]),
                        "scenario_id": str(scenario["id"]),
                        "model_seed": int(model_seed),
                    }
                    cells.append(
                        ConfirmatoryCell(
                            cell_id=_stable_id("confirmatory", payload, len(cells)),
                            outer_fold=int(outer_fold),
                            corruption_cell_id=str(corruption_cell["id"]),
                            scenario_id=str(scenario["id"]),
                            model_seed=int(model_seed),
                            required=bool(scenario["required"]),
                        )
                    )
    identifiers = [cell.cell_id for cell in cells]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("confirmatory matrix expansion produced duplicate cell IDs")
    return ConfirmatoryMatrixPlan(
        int(resolved["schema_version"]), config_sha256(resolved), tuple(cells)
    )


__all__ = [
    "CLASS_ORDER",
    "CONFIRMATORY_COMPARISON_RISK_IDS",
    "CONFIRMATORY_ENSEMBLE_RISKS",
    "CONFIRMATORY_REQUIRED_FROZEN_ABLATIONS",
    "ORIGINAL_AUDIT_RISK_METHODS",
    "PRIMARY_AUDIT_METHODS",
    "PRIMARY_CORRUPTION_MECHANISMS",
    "PRIMARY_GENERATOR_INDEPENDENCE_STATUSES",
    "RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256",
    "RESOURCE_BOUNDED_CONFIRMATORY_PROFILE",
    "ConfirmatoryCell",
    "ConfirmatoryMatrixPlan",
    "PrimaryCell",
    "PrimaryMatrixPlan",
    "PrimaryScenario",
    "StudyContractError",
    "build_confirmatory_matrix_plan",
    "build_primary_matrix_plan",
    "validate_confirmatory_execution_config",
    "validate_frozen_confirmatory_config",
    "validate_frozen_primary_config",
    "validate_primary_confirmatory_cross_config",
    "validate_resource_bounded_confirmatory_config",
]
