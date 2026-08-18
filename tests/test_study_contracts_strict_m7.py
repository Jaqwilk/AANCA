"""Regression tests for the strict M7 primary/confirmatory freeze contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from histo_audit.experiment.study_contracts import (
    StudyContractError,
    _expand_primary_matrix_components,
    build_primary_matrix_plan,
    validate_frozen_confirmatory_config,
    validate_frozen_primary_config,
)
from tests.test_study_contracts import complete_confirmatory_config, complete_primary_config

EXCLUSION_POLICY = {
    "cross_class_overlap_touching": "exclude_with_reason_touches_cross_class_overlap",
    "positive_background_conflict_touching": "retain_with_qc_flag_no_class_arbitration",
    "disconnected_instance_id": "retain_with_flag",
    "border_instance": "retain_with_flag",
    "malformed_or_structurally_invalid_mask": "fail_closed_at_dataset_gate",
    "duplicates": "flag_without_automatic_deletion",
    "void_pixels": "retain_as_unlabeled_void",
    "missing_required_data": "fail_closed_at_dataset_gate",
}
CLEAN_REFERENCE = {
    "id": "clean_reference_cell",
    "mechanism": "symmetric_random_corruption",
    "rate": 0.0,
    "seed": 404,
    "parameters": {},
}
PILOT_DERIVED_PARAMETERS = {
    "schema_version": 1,
    "producer_id": "pilot_derived_primary_parameters_v1",
    "path": "reports/pilot_derived_primary_parameters.json",
    "sha256": "8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b",
    "source_pilot_run_id": "20260718T143216.354310Z_pannuke_pilot_c7797330e0",
    "source_pilot_artifact_root_sha256": (
        "37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666"
    ),
}


def strict_primary_config() -> dict[str, Any]:
    config = complete_primary_config()
    config["data"]["reference_group_selection_algorithm"] = (
        "deterministic_group_greedy_class_distribution_v1"
    )
    config["data"]["exclusions"] = dict(EXCLUSION_POLICY)
    config["pilot_derived_parameters"] = dict(PILOT_DERIVED_PARAMETERS)
    config["corruption"]["clean_reference"] = dict(CLEAN_REFERENCE)
    _, cells = _expand_primary_matrix_components(config)
    restoration_cell = next(
        cell
        for cell in cells
        if cell.mechanism == "symmetric_random_corruption"
        and cell.rate == 0.10
        and cell.corruption_seed == 404
        and cell.representation_id == "imagenet_resnet18_highlighted"
        and cell.classifier_id == "multinomial_logistic_regression"
    )
    config["restoration"]["enabled_cells"] = [restoration_cell.cell_id]
    return config


def strict_confirmatory_config() -> dict[str, Any]:
    config = complete_confirmatory_config()
    config["data"]["reference_group_selection_algorithm"] = (
        "deterministic_group_greedy_class_distribution_v1"
    )
    config["data"]["fold_assignment_labels"] = "pre_corruption_label"
    config["corruption"]["cells"][0] = dict(CLEAN_REFERENCE)
    return config


def _selector(
    *,
    mechanism: str = "symmetric_random_corruption",
    rate: float = 0.10,
    seed: int = 404,
    representation_id: str = "imagenet_resnet18_highlighted",
    classifier_id: str = "multinomial_logistic_regression",
) -> dict[str, Any]:
    return {
        "mechanism": mechanism,
        "rate": rate,
        "seed": seed,
        "representation_id": representation_id,
        "classifier_id": classifier_id,
    }


def _cross_comparison(
    comparison_id: str,
    selector_a: dict[str, Any],
    selector_b: dict[str, Any],
    *,
    method_a: str = "self_confidence",
    method_b: str = "self_confidence",
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "selector_a": selector_a,
        "selector_b": selector_b,
        "method_a": method_a,
        "method_b": method_b,
        "metric": "average_precision",
        "direction": "method_a_minus_method_b",
        "holm_family": "primary_cross_cell",
    }


def test_strict_primary_contract_expands_clean_reference_once() -> None:
    config = strict_primary_config()

    assert validate_frozen_primary_config(config) == config
    plan = build_primary_matrix_plan(config)

    assert len(plan.scenarios) == 37
    assert len(plan.cells) == 222
    assert plan.required_cell_count == 185
    assert plan.optional_cell_count == 37
    clean_scenarios = [scenario for scenario in plan.scenarios if scenario.rate == 0.0]
    assert len(clean_scenarios) == 1
    assert clean_scenarios[0].scenario_id == "clean_reference_cell"


def test_primary_and_confirmatory_reference_selector_handshake_is_identical() -> None:
    primary = validate_frozen_primary_config(strict_primary_config())
    confirmatory = validate_frozen_confirmatory_config(strict_confirmatory_config())

    assert (
        primary["data"]["reference_group_selection_algorithm"]
        == confirmatory["data"]["reference_group_selection_algorithm"]
        == "deterministic_group_greedy_class_distribution_v1"
    )


@pytest.mark.parametrize(
    ("factory", "validator"),
    [
        (strict_primary_config, validate_frozen_primary_config),
        (strict_confirmatory_config, validate_frozen_confirmatory_config),
    ],
)
def test_frozen_contract_rejects_unknown_top_level_key(
    factory: Callable[[], dict[str, Any]],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    config = factory()
    config["silently_ignored_field"] = True

    with pytest.raises(StudyContractError, match="unexpected"):
        validator(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("development_official_folds", [2, 1]),
        ("final_test_fold", 2),
        ("group_unit", "nucleus_id"),
        ("reference_validation_fraction_groups", 0.99),
        ("reference_group_selection_algorithm", "random_group_split"),
        ("split_seed", 999),
        ("fold_assignment_labels", "observed_label"),
    ],
)
def test_primary_data_constants_are_exact(field: str, value: Any) -> None:
    config = strict_primary_config()
    config["data"][field] = value

    with pytest.raises(StudyContractError):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize("field", tuple(EXCLUSION_POLICY))
def test_primary_exclusion_policy_rejects_every_changed_value(field: str) -> None:
    config = strict_primary_config()
    config["data"]["exclusions"][field] = "silent_arbitration_or_deletion"

    with pytest.raises(StudyContractError, match=r"data\.exclusions"):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_primary_exclusion_policy_rejects_key_drift(mutation: str) -> None:
    config = strict_primary_config()
    if mutation == "missing":
        del config["data"]["exclusions"]["void_pixels"]
    else:
        config["data"]["exclusions"]["unknown_qc_case"] = "retain"

    with pytest.raises(StudyContractError, match=r"data\.exclusions"):
        validate_frozen_primary_config(config)


def test_primary_clean_reference_is_exact() -> None:
    config = strict_primary_config()
    config["corruption"]["clean_reference"]["seed"] = 405

    with pytest.raises(StudyContractError, match=r"clean_reference\.seed"):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("producer_id", "ad_hoc_report"),
        ("path", "reports/unbound_parameters.json"),
        ("sha256", "A" * 64),
        ("source_pilot_run_id", "unsafe/run/id"),
        ("source_pilot_artifact_root_sha256", "0" * 63),
    ],
)
def test_pilot_derived_parameters_are_exact_and_hash_bound(field: str, value: Any) -> None:
    config = strict_primary_config()
    config["pilot_derived_parameters"][field] = value

    with pytest.raises(StudyContractError, match="pilot_derived_parameters"):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_pilot_derived_parameters_reject_field_drift(mutation: str) -> None:
    config = strict_primary_config()
    if mutation == "missing":
        del config["pilot_derived_parameters"]["source_pilot_run_id"]
    else:
        config["pilot_derived_parameters"]["untracked_note"] = "ignored"

    with pytest.raises(StudyContractError, match="pilot_derived_parameters"):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize(
    ("factory", "validator", "hybrid_key"),
    [
        (strict_primary_config, validate_frozen_primary_config, "audit"),
        (strict_confirmatory_config, validate_frozen_confirmatory_config, "fixed_hybrid"),
    ],
)
def test_fixed_hybrid_requires_equal_weights(
    factory: Callable[[], dict[str, Any]],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    hybrid_key: str,
) -> None:
    config = factory()
    hybrid = config["audit"]["fixed_hybrid"] if hybrid_key == "audit" else config[hybrid_key]
    hybrid["weights"] = [0.9, 0.1]

    with pytest.raises(StudyContractError, match="equal weights"):
        validator(config)


def test_statistical_cell_id_must_exist_and_clean_cell_is_not_inferential() -> None:
    config = strict_primary_config()
    config["statistics"]["within_cell_comparisons"][0]["selector"] = {
        "cell_id": "primary_9999_deadbeefdead"
    }
    with pytest.raises(StudyContractError, match="absent from the matrix"):
        validate_frozen_primary_config(config)

    config = strict_primary_config()
    _, cells = _expand_primary_matrix_components(config)
    clean_cell = next(cell for cell in cells if cell.rate == 0.0)
    config["statistics"]["within_cell_comparisons"][0]["selector"] = {"cell_id": clean_cell.cell_id}
    with pytest.raises(StudyContractError, match="0% clean reference"):
        validate_frozen_primary_config(config)


def test_existing_positive_cell_id_is_accepted() -> None:
    config = strict_primary_config()
    selected_id = config["restoration"]["enabled_cells"][0]
    config["statistics"]["within_cell_comparisons"][0]["selector"] = {"cell_id": selected_id}

    assert validate_frozen_primary_config(config) == config


def test_restoration_enabled_cell_must_exist_and_be_nonzero() -> None:
    config = strict_primary_config()
    config["restoration"]["enabled_cells"] = ["primary_9999_deadbeefdead"]
    with pytest.raises(StudyContractError, match="absent from the matrix"):
        validate_frozen_primary_config(config)

    config = strict_primary_config()
    _, cells = _expand_primary_matrix_components(config)
    config["restoration"]["enabled_cells"] = [
        next(cell.cell_id for cell in cells if cell.rate == 0)
    ]
    with pytest.raises(StudyContractError, match="0% clean reference"):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize(
    ("comparison_id", "selector_a", "selector_b"),
    [
        (
            "rate_sensitivity_effect",
            _selector(rate=0.10),
            _selector(rate=0.20),
        ),
        (
            "h3_corruption_mechanism_effect",
            _selector(mechanism="symmetric_random_corruption"),
            _selector(mechanism="confusion_targeted_corruption"),
        ),
    ],
)
def test_controlled_cross_scenario_contrasts_are_valid(
    comparison_id: str,
    selector_a: dict[str, Any],
    selector_b: dict[str, Any],
) -> None:
    config = strict_primary_config()
    config["statistics"]["cross_cell_comparisons"].append(
        _cross_comparison(comparison_id, selector_a, selector_b)
    )

    assert validate_frozen_primary_config(config) == config


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        (
            _cross_comparison(
                "h3_mismatched_seed",
                _selector(mechanism="symmetric_random_corruption", seed=404),
                _selector(mechanism="confusion_targeted_corruption", seed=405),
            ),
            "hold seed, representation, and classifier fixed",
        ),
        (
            _cross_comparison(
                "two_corruption_factors_changed",
                _selector(mechanism="symmetric_random_corruption", rate=0.10),
                _selector(mechanism="confusion_targeted_corruption", rate=0.20),
            ),
            "vary exactly one of mechanism or rate",
        ),
        (
            _cross_comparison(
                "confounded_method_and_cell_effect",
                _selector(rate=0.10),
                _selector(rate=0.20),
                method_b="predictive_entropy",
            ),
            "hold the audit method fixed",
        ),
    ],
)
def test_cross_scenario_contrasts_reject_confounded_factors(
    comparison: dict[str, Any], message: str
) -> None:
    config = strict_primary_config()
    config["statistics"]["cross_cell_comparisons"].append(comparison)

    with pytest.raises(StudyContractError, match=message):
        validate_frozen_primary_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["corruption"]["cells"].pop(0),
        lambda config: config["corruption"]["cells"][0].update(seed=405),
        lambda config: config["corruption"]["cells"].append(dict(CLEAN_REFERENCE)),
    ],
)
def test_confirmatory_requires_one_exact_clean_reference(
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    config = strict_confirmatory_config()
    mutation(config)

    with pytest.raises(StudyContractError, match=r"clean|one clean"):
        validate_frozen_confirmatory_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_folds", [4, 5, 6]),
        ("group_unit", "nucleus_id"),
        ("reference_validation_fraction_groups", 0.50),
        ("reference_group_selection_algorithm", "random_group_split"),
        ("split_seed", 224),
        ("fold_assignment_labels", "observed_label"),
    ],
)
def test_confirmatory_data_constants_are_exact(field: str, value: Any) -> None:
    config = strict_confirmatory_config()
    config["data"][field] = value

    with pytest.raises(StudyContractError):
        validate_frozen_confirmatory_config(config)
