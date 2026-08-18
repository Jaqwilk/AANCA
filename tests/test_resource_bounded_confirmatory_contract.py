"""Fail-closed tests for the amended resource-bounded confirmatory sensitivity."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
    build_confirmatory_completion_evidence,
    reconcile_confirmatory_cell_outcomes,
)
from histo_audit.experiment.confirmatory_core import (
    _synthetic_frozen_runner,
    _synthetic_image_runner,
    _synthetic_rotation,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
)
from histo_audit.experiment.m7_config_finalization import (
    derive_confirmatory_cnn_logical_provenance,
)
from histo_audit.experiment.study_contracts import (
    RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256,
    StudyContractError,
    build_confirmatory_matrix_plan,
    validate_confirmatory_execution_config,
    validate_frozen_confirmatory_config,
    validate_resource_bounded_confirmatory_config,
)
from histo_audit.representations.cache_provenance import FrozenCacheVerification
from histo_audit.workflows import preregistration_amendment as amendment
from histo_audit.workflows.study_gates import (
    HistoricalPrimaryDependencyEvidence,
    PrimaryExecutionGateEvidence,
    ResourceBoundedExecutionAuthorityEvidence,
    ResourceBoundedExecutionGateEvidence,
)

_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_CONFIG = _ROOT / "configs" / "confirmatory_resource_bounded_amended.yaml"
_LEGACY_CONFIG = _ROOT / "configs" / "confirmatory_frozen.yaml"


def _resource_config() -> dict[str, Any]:
    return load_config(_RESOURCE_CONFIG)


def _resource_workspace_plan() -> dict[str, Any]:
    source_row_count = 188_333
    extracted_bytes = 4_294_182_269
    array_count = 12
    minimum_member_bytes = source_row_count + 128
    member_bytes = [minimum_member_bytes] * (array_count - 1)
    member_bytes.append(extracted_bytes - sum(member_bytes))
    arrays = [
        {
            "array_id": f"array_{index:02d}",
            "source_npz_sha256": f"{index + 1:064x}",
            "source_sidecar_sha256": f"{index + 17:064x}",
            "source_member_name": f"array_{index:02d}.npy",
            "source_member_crc32": f"{index:08x}",
            "source_member_compression": "deflated",
            "source_member_compressed_bytes": 1,
            "source_member_uncompressed_bytes": member_bytes[index],
            "dtype": "|u1",
            "shape": [source_row_count, 1],
            "raw_array_nbytes": source_row_count,
            "expected_array_sha256": f"{index + 33:064x}",
        }
        for index in range(array_count)
    ]
    role_counts = {
        "audit": 120_000,
        "reference_validation": 60_000,
        "final_reference": 8_333,
    }
    partition_specs = [
        {
            "outer_fold": fold,
            "role": role,
            "row_count": row_count,
            "source_indices_sha256": f"{fold * 3 + role_index + 48:064x}",
            "ordered_unique": True,
            "relative_path": f"indices/fold_{fold}__{role}.npy",
            "raw_nbytes": row_count * 8,
            "npy_file_bytes": row_count * 8 + 128,
        }
        for fold in (1, 2, 3)
        for role_index, (role, row_count) in enumerate(sorted(role_counts.items()))
    ]
    index_file_bytes = sum(record["npy_file_bytes"] for record in partition_specs)
    metadata_bytes = 16_777_216
    planned_bytes = extracted_bytes + index_file_bytes + metadata_bytes
    minimum_after = 23_622_320_128
    plan = {
        "schema_version": 1,
        "recipe_id": "pannuke_confirmatory_shared_memmap_workspace_v1",
        "workspace_reuse_allowed": False,
        "workspace_key": "a" * 64,
        "source_row_count": source_row_count,
        "arrays": arrays,
        "partition_index_specs": partition_specs,
        "expected_extracted_file_bytes": extracted_bytes,
        "expected_raw_array_nbytes": source_row_count * array_count,
        "expected_index_npy_file_bytes": index_file_bytes,
        "expected_index_raw_nbytes": sum(record["raw_nbytes"] for record in partition_specs),
        "metadata_capacity_allowance_bytes": metadata_bytes,
        "planned_workspace_bytes": planned_bytes,
        "minimum_free_bytes_after": minimum_after,
        "maximum_workspace_bytes": 4_567_138_869,
        "required_free_bytes_before": minimum_after + planned_bytes,
    }
    return {**plan, "plan_without_self_hash_sha256": canonical_sha256(plan)}


def test_capacity_v3_uses_physical_npy_index_bytes_in_projected_workspace() -> None:
    plan = _resource_workspace_plan()

    assert plan["expected_index_raw_nbytes"] == 4_519_992
    assert plan["expected_index_npy_file_bytes"] == 4_521_144
    assert plan["expected_index_npy_file_bytes"] - plan["expected_index_raw_nbytes"] == 9 * 128
    capacity, canonical = amendment.validate_resource_bounded_capacity_v3(
        amendment._RESOURCE_BOUNDED_CAPACITY_V3,
        plan,
    )

    assert capacity == amendment._RESOURCE_BOUNDED_CAPACITY_V3
    assert canonical == plan
    assert (
        canonical["expected_extracted_file_bytes"] + canonical["expected_index_npy_file_bytes"]
        == capacity["projected_workspace_bytes"]
    )


def _resource_gate() -> ResourceBoundedExecutionGateEvidence:
    primary_gate = PrimaryExecutionGateEvidence(
        freeze_directory=Path("P"),
        base_freeze_directory=Path("base"),
        freeze_artifact_root_sha256="1" * 64,
        freeze_manifest_sha256="2" * 64,
        preregistration_sha256="3" * 64,
        frozen_primary_config_sha256="4" * 64,
        frozen_confirmatory_config_sha256="5" * 64,
        primary_config_semantic_sha256="6" * 64,
        confirmatory_config_semantic_sha256="7" * 64,
        primary_matrix_cell_count=222,
        primary_required_cell_count=185,
        confirmatory_matrix_cell_count=108,
        pilot_run_id="pilot",
        pilot_artifact_root_sha256="8" * 64,
        dataset_sha256="9" * 64,
        manifest_sha256="a" * 64,
        duplicate_audit_sha256="b" * 64,
        pathology_encoder_audit_sha256="c" * 64,
        source_tree_root_sha256="d" * 64,
        registration_authority_kind="preregistration_amendment",
        registration_status="amended_or_exploratory",
        registration_authority_chain_depth=1,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=False,
    )
    historical = HistoricalPrimaryDependencyEvidence(
        primary_gate=primary_gate,
        primary_run_directory=Path("runs") / "primary",
        primary_run_id="primary",
        primary_artifact_root_sha256="0" * 64,
        primary_artifact_manifest_sha256="1" * 64,
        primary_completion_evidence_sha256="2" * 64,
        primary_execution_gate_sha256="3" * 64,
        primary_reconciliation_sha256="4" * 64,
        completed_required_cell_count=185,
        primary_statistics_manifest_sha256="5" * 64,
        primary_statistics_sha256="6" * 64,
        primary_bootstrap_evidence_sha256="7" * 64,
        primary_subgroups_sha256="8" * 64,
        primary_statistics_source_readback_root_sha256="9" * 64,
        primary_statistics_comparison_count=36,
        primary_stage_attestation_record_sha256="a" * 64,
        primary_stage_attestation_verification_sha256="b" * 64,
        primary_restoration_readback_root_sha256="c" * 64,
        primary_recovery_evidence_sha256="d" * 64,
        primary_recovery_authorization_sha256="e" * 64,
        primary_recovery_source_run_id="source",
        primary_recovery_source_snapshot_root_sha256="f" * 64,
        primary_recovery_analysis_disposition="amended_or_exploratory",
    )
    workspace_plan = _resource_workspace_plan()
    authority = ResourceBoundedExecutionAuthorityEvidence(
        authority_directory=Path("C"),
        authority_artifact_root_sha256="1" * 64,
        authority_manifest_sha256="2" * 64,
        authority_chain_depth=2,
        authorization_sha256="3" * 64,
        resource_profile_id="resource_bounded_confirmatory_v1",
        resource_confirmatory_config_file_sha256="4" * 64,
        resource_confirmatory_config_semantic_sha256=(RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256),
        resource_execution_source_root_sha256="5" * 64,
        resource_execution_source_manifest_sha256="6" * 64,
        resource_source_delta_sha256="7" * 64,
        confirmatory_storage_policy_sha256="8" * 64,
        resource_capacity_policy=dict(amendment._RESOURCE_BOUNDED_CAPACITY_V3),
        resource_input_workspace_plan=workspace_plan,
        resource_input_workspace_plan_sha256=workspace_plan["plan_without_self_hash_sha256"],
    )
    return ResourceBoundedExecutionGateEvidence(
        historical_primary=historical,
        execution_authority=authority,
    )


def _completed_outcomes(config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plan = build_confirmatory_matrix_plan(config)
    outcomes = [
        {
            "cell_id": cell.cell_id,
            "required": True,
            "status": "completed",
            "outer_fold": cell.outer_fold,
            "model_seed": cell.model_seed,
            "scenario_id": cell.scenario_id,
            "corruption_cell_id": cell.corruption_cell_id,
            "artifact_manifest_sha256": "a" * 64,
            "metrics_sha256": "b" * 64,
        }
        for cell in plan.cells
    ]
    return plan, outcomes


def test_exact_resource_bounded_profile_builds_only_24_required_cells() -> None:
    config = _resource_config()
    validated = validate_resource_bounded_confirmatory_config(config)
    plan = build_confirmatory_matrix_plan(config)

    assert config_sha256(validated) == RESOURCE_BOUNDED_CONFIRMATORY_CONFIG_SHA256
    assert plan.schema_version == 3
    assert len(plan.cells) == 24
    assert plan.required_cell_count == 24
    assert plan.optional_cell_count == 0
    assert {cell.outer_fold for cell in plan.cells} == {1, 2, 3}
    assert {cell.model_seed for cell in plan.cells} == {303}
    assert {cell.corruption_cell_id for cell in plan.cells} == {
        "clean_reference_cell",
        "confusion_targeted_ten_percent",
    }
    assert {cell.scenario_id for cell in plan.cells} == {
        "cnn_context_rgb",
        "imagenet_frozen_logistic",
        "imagenet_frozen_target_highlighted_logistic",
        "imagenet_frozen_context_morphometrics_logistic",
    }
    assert sum(cell.scenario_id == "cnn_context_rgb" for cell in plan.cells) == 6


def test_resource_cnn_provenance_exactly_matches_live_runtime_derivation() -> None:
    config = _resource_config()
    crop = FrozenCacheVerification(
        cache_path=Path("pannuke_crops.npz"),
        sidecar_path=Path("pannuke_crops.npz.metadata.json"),
        cache_file_sha256=("07d484be3e9f7826030f5d54d17e9878f61b68c282c4a91305a30ecfa86f4a01"),
        sidecar_file_sha256=("738d3f4b3146ff6d62555d283dac84a05a063b98199a6d13175100feb5d5dd42"),
        sidecar_semantic_sha256=(
            "cf74379bab82d41be0df6cf047f8d365c5beb8854d34e7e69e22b2be403756b9"
        ),
        metadata={
            "cache_array_sha256_by_name": {
                "context_rgb": ("aa699b0853ea0308a72f42cd6b4cb32a232c0093cd6f3e5b3d7baf1f00a44335"),
                "target_masks": (
                    "d0c545e3307bd822847f20dc3a06712a430c9c8c9c2e444c0089bdd9bab7df27"
                ),
            },
            "cache_content_sha256": (
                "40c4834dea663cd6510f4657da113b0c4f0c76ea1fb6ea0db3c22dbd9d1e2097"
            ),
            "sample_order_sha256": (
                "2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26"
            ),
            "manifest_sha256": ("7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e"),
        },
    )
    derived = derive_confirmatory_cnn_logical_provenance(
        crop,
        weight_identifier="ResNet18_Weights.IMAGENET1K_V1",
        weights_sha256=("f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
        input_size=224,
    )
    records = {record["id"]: record for record in config["cache_provenance"]}

    assert records["cnn_context_rgb_cache"] == derived["cnn_context_rgb_cache"]


def test_resource_profile_is_separate_and_schema_v2_is_unchanged() -> None:
    resource = _resource_config()
    with pytest.raises(StudyContractError, match=r"frozen fields|schema_version"):
        validate_frozen_confirmatory_config(resource)

    legacy = load_config(_LEGACY_CONFIG)
    validated = validate_frozen_confirmatory_config(legacy)
    dispatched = validate_confirmatory_execution_config(legacy)
    plan = build_confirmatory_matrix_plan(legacy)

    assert dispatched == validated
    assert config_sha256(validated) == (
        "ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b"
    )
    assert plan.schema_version == 2
    assert len(plan.cells) == 108
    assert plan.required_cell_count == 90
    assert plan.optional_cell_count == 18


@pytest.mark.parametrize(
    "mutator",
    [
        lambda config: config.update(original_confirmatory_claim_allowed=True),
        lambda config: config.update(completion_stage="CONFIRMATORY_COMPLETE"),
        lambda config: config.update(execution_profile="resource_bounded_confirmatory_v2"),
        lambda config: config["model_seeds"].append(304),
        lambda config: config["training"].update(max_epochs=5),
        lambda config: config["training"].update(early_stopping_patience=3),
        lambda config: config["oof"].update(split_kind="stratified"),
        lambda config: config["scenarios"][0].update(
            input_variant="context_rgb_plus_binary_target_mask"
        ),
        lambda config: config["ensemble"]["members"][0].update(model_seed=304),
        lambda config: config["restoration"].update(random_repeats=99),
        lambda config: config["statistics"].update(paired_group_bootstrap_iterations=1999),
        lambda config: config["statistics"]["preregistered_paired_comparisons"][-1].update(
            holm_family="ranking_method_family"
        ),
        lambda config: config["cache_provenance"][0].update(sidecar_semantic_sha256="0" * 64),
    ],
)
def test_resource_profile_rejects_every_scientific_or_disposition_mutation(
    mutator: Any,
) -> None:
    config = deepcopy(_resource_config())
    mutator(config)

    with pytest.raises(StudyContractError):
        validate_resource_bounded_confirmatory_config(config)


def test_resource_controls_bind_profile_disposition_and_exact_analyses() -> None:
    controls = confirmatory_execution_controls_from_frozen_config(_resource_config())
    serialised = controls.as_dict()

    assert controls.plan.schema_version == 3
    assert controls.model_seeds == (303,)
    assert controls.max_epochs == 4
    assert controls.early_stopping_patience == 2
    assert [(member.scenario_id, member.model_seed) for member in controls.ensemble_members] == [
        ("imagenet_frozen_logistic", 303),
        ("imagenet_frozen_target_highlighted_logistic", 303),
        ("imagenet_frozen_context_morphometrics_logistic", 303),
    ]
    assert controls.restoration_scenario_id == ("imagenet_frozen_target_highlighted_logistic")
    assert controls.restoration_model_seed == 303
    assert controls.restoration_random_repeats == 100
    assert controls.paired_group_bootstrap_iterations == 2000
    assert len(controls.paired_comparisons) == 6
    assert controls.holm_families == (
        "target_representation_family",
        "ranking_method_family",
        "model_family",
    )
    assert serialised["source"] == ("validated_resource_bounded_confirmatory_v1_schema_v3")
    assert serialised["execution_profile"] == "resource_bounded_confirmatory_v1"
    assert serialised["analysis_disposition"] == "amended_or_exploratory"
    assert serialised["original_confirmatory_claim_allowed"] is False
    assert serialised["completion_stage"] is None
    controls.validate_for_plan(controls.plan)

    legacy_serialised = confirmatory_execution_controls_from_frozen_config(
        load_config(_LEGACY_CONFIG)
    ).as_dict()
    assert legacy_serialised["source"] == ("validated_frozen_confirmatory_config_schema_v2")
    assert "execution_profile" not in legacy_serialised
    assert "analysis_disposition" not in legacy_serialised
    assert "original_confirmatory_claim_allowed" not in legacy_serialised
    assert "completion_stage" not in legacy_serialised


def test_resource_scope_can_never_claim_confirmatory_complete() -> None:
    plan, outcomes = _completed_outcomes(_resource_config())
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    assert reconciliation.passed
    assert reconciliation.fold_rotation_complete

    evidence = build_confirmatory_completion_evidence(
        plan=plan,
        reconciliation=reconciliation,
        artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        study_outcome_eligible=False,
        gate_evidence=_resource_gate(),
    )

    assert evidence["artifact_scope"] == ("resource_bounded_confirmatory_sensitivity")
    assert evidence["study_outcome_eligible"] is False
    assert evidence["completion_stage"] is None
    assert evidence["valid_completion_claim"] is False
    with pytest.raises(ValueError, match="can never be study-outcome eligible"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=_resource_gate(),
        )


def test_schema_v3_plan_cannot_enter_real_completion_scope() -> None:
    plan, outcomes = _completed_outcomes(_resource_config())
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)

    with pytest.raises(ValueError, match="original schema-v2 plan"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
        )


def test_schema_v3_plan_cannot_enter_real_matrix_scope(tmp_path: Path) -> None:
    controls = confirmatory_execution_controls_from_frozen_config(_resource_config())
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)

    with pytest.raises(ValueError, match="original schema-v2 plan"):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "wrong-real-scope",
            image_oof_runner=_synthetic_image_runner,
            frozen_oof_runner=_synthetic_frozen_runner,
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=True,
        )


def test_resource_completion_rejects_serialised_or_false_gate_evidence() -> None:
    plan, outcomes = _completed_outcomes(_resource_config())
    reconciliation = reconcile_confirmatory_cell_outcomes(plan, outcomes)
    gate = _resource_gate()

    with pytest.raises(ValueError, match="typed"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
            gate_evidence=gate.as_dict(),
        )
    with pytest.raises(ValueError, match="non-claiming P\\+"):
        build_confirmatory_completion_evidence(
            plan=plan,
            reconciliation=reconciliation,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            study_outcome_eligible=False,
            gate_evidence=replace(gate, outcomes_inspected=False),
        )


def test_resource_scope_executes_three_rotation_structure_but_stays_ineligible(
    tmp_path: Path,
) -> None:
    controls = confirmatory_execution_controls_from_frozen_config(_resource_config())
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)
    progress_events: list[dict[str, Any]] = []

    artifacts = execute_confirmatory_matrix(
        rotations,
        controls.plan,
        controls,
        output_directory=tmp_path / "resource-bounded",
        image_oof_runner=_synthetic_image_runner,
        frozen_oof_runner=_synthetic_frozen_runner,
        artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
        cpu_test_only=True,
        progress_callback=lambda event: progress_events.append(dict(event)),
        gate_evidence=_resource_gate(),
    )

    assert artifacts.reconciliation.passed
    assert artifacts.reconciliation.fold_rotation_complete
    assert len(artifacts.outcomes) == 24
    assert artifacts.study_outcome_eligible is False
    completion = json.loads(artifacts.completion_evidence_path.read_text(encoding="utf-8"))
    assert completion["artifact_scope"] == ("resource_bounded_confirmatory_sensitivity")
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["valid_completion_claim"] is False
    assert len(progress_events) == 48
    assert {event["status"] for event in progress_events} == {
        "started",
        "model_completed",
    }
    assert all(
        set(event)
        == {
            "cell_id",
            "scenario",
            "fold",
            "corruption",
            "seed",
            "status",
            "timestamp",
            "telemetry_contract",
            "prohibited_for_selection_tuning",
            "adaptive_execution_changes_allowed",
        }
        for event in progress_events
    )
    forbidden_outcome_keys = {
        "metric",
        "metrics",
        "ranking",
        "rankings",
        "probability",
        "probabilities",
        "observed_label",
        "pre_corruption_label",
        "is_injected_corruption",
    }
    assert all(not forbidden_outcome_keys.intersection(event) for event in progress_events)
    assert {event["telemetry_contract"] for event in progress_events} == {
        "outcome_value_free_operational_telemetry"
    }
    assert all(event["prohibited_for_selection_tuning"] is True for event in progress_events)
    assert all(event["adaptive_execution_changes_allowed"] is False for event in progress_events)


def test_resource_executor_still_requires_exactly_three_rotations(
    tmp_path: Path,
) -> None:
    controls = confirmatory_execution_controls_from_frozen_config(_resource_config())
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in (1, 2))

    with pytest.raises(ValueError, match="exactly all three"):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "two-rotations",
            image_oof_runner=_synthetic_image_runner,
            frozen_oof_runner=_synthetic_frozen_runner,
            artifact_scope=RESOURCE_BOUNDED_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=True,
            gate_evidence=_resource_gate(),
        )
