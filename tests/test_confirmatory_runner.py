from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from test_study_contracts import complete_confirmatory_config

import histo_audit.experiment.confirmatory_runner as confirmatory_runner_module
from histo_audit.config import load_config
from histo_audit.corruption.controlled import apply_controlled_corruption, canonical_sha256
from histo_audit.cross_validation.oof import make_group_stratified_folds
from histo_audit.experiment.confirmatory_completion import (
    SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryCellRequest,
    ConfirmatoryRunnerInputs,
    _synthetic_frozen_runner,
    _synthetic_image_runner,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
)
from histo_audit.experiment.confirmatory_runner import (
    ConfirmatoryRunnerDependencies,
    ConfirmatoryStudyRunnerError,
    _confirmatory_cnn_preflight_fingerprints,
    _require_exact_published_t0_lifecycle_pins,
    _validate_restoration_source_binding,
    bridge_pannuke_confirmatory_inputs,
    execute_confirmatory_study,
    finalize_confirmatory_stage,
    run_confirmatory_frozen_feature_oof,
)
from histo_audit.experiment.confirmatory_statistics import (
    aggregate_confirmatory_statistics,
)
from histo_audit.experiment.m7_config_finalization import (
    derive_confirmatory_cnn_logical_provenance,
)
from histo_audit.experiment.pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureAvailability,
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryPartitionFeature,
    ConfirmatoryPartitionInputs,
    PanNukeConfirmatoryInputs,
    PanNukeConfirmatoryRotationInputs,
)
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    build_confirmatory_matrix_plan,
)
from histo_audit.models.cnn import confirmatory_cnn_data_and_split_sha256
from histo_audit.representations.cache_provenance import (
    FrozenCacheVerification,
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    confirmatory_cache_provenance_record,
    verify_frozen_cache_sidecar,
)
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    sha256_file,
)
from histo_audit.workflows.study_gates import (
    ConfirmatoryExecutionGateEvidence,
    PrimaryExecutionGateEvidence,
)


@dataclass(frozen=True)
class _Bundle:
    root: Path
    config: dict[str, Any]
    prepared: PanNukeConfirmatoryInputs
    feature_specs: tuple[ConfirmatoryFrozenFeatureCacheSpec, ...]
    crop_path: Path
    crop_metadata_sha256: str
    manifest: Path
    pathology_audit: Path
    gate: ConfirmatoryExecutionGateEvidence


def _readonly(values: np.ndarray[Any, Any], dtype: Any) -> np.ndarray[Any, Any]:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _partition(
    *,
    role: str,
    indices: np.ndarray[Any, Any],
    sample_ids: np.ndarray[Any, Any],
    group_ids: np.ndarray[Any, Any],
    rgb: np.ndarray[Any, Any],
    masks: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    observed: np.ndarray[Any, Any],
    scenario_features: dict[str, np.ndarray[Any, Any]],
) -> ConfirmatoryPartitionInputs:
    changed = observed[indices] != labels[indices]
    output = ConfirmatoryPartitionInputs(
        role=role,  # type: ignore[arg-type]
        source_indices=_readonly(indices, np.int64),
        sample_ids=tuple(str(value) for value in sample_ids[indices]),
        group_ids=tuple(str(value) for value in group_ids[indices]),
        context_rgb=_readonly(rgb[indices], np.uint8),
        target_masks=_readonly(masks[indices], np.bool_),
        pre_corruption_labels=_readonly(labels[indices], np.int64),
        observed_labels=_readonly(observed[indices], np.int64),
        is_injected_corruption=_readonly(changed, np.bool_),
        corruption_types=tuple(
            "symmetric_random_corruption" if value else "none" for value in changed
        ),
        frozen_features=tuple(
            ConfirmatoryPartitionFeature(
                scenario_id=scenario_id,
                values=_readonly(values[indices], np.float64),
            )
            for scenario_id, values in sorted(scenario_features.items())
        ),
    )
    output.validate()
    return output


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _publish_test_cache(
    path: Path,
    *,
    arrays: dict[str, np.ndarray[Any, Any]],
    manifest_sha256: str,
    raw_inventory_sha256: str,
    representation_id: str,
    input_variant: str,
    encoder_identifier: str,
    weight_identifier: str,
    weights_sha256: str,
    preprocessing_identifier: str,
    matrix_key: str,
) -> FrozenCacheVerification:
    matrix = arrays[matrix_key]
    feature_dimension: int | list[int] = (
        int(matrix.shape[1]) if matrix.ndim == 2 else [int(value) for value in matrix.shape[1:]]
    )
    metadata = build_frozen_cache_metadata(
        base_metadata={"schema_version": 1, "test_fixture": True},
        sample_ids=np.asarray(arrays["sample_ids"], dtype=np.str_),
        manifest_sha256=manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        representation_id=representation_id,
        input_variant=input_variant,
        encoder_identifier=encoder_identifier,
        encoder_metadata={
            "schema_version": 1,
            "fixture_encoder": encoder_identifier,
            "representation_id": representation_id,
        },
        encoder_implementation={"module": "tests.test_confirmatory_runner"},
        weight_identifier=weight_identifier,
        weights_sha256=weights_sha256,
        preprocessing_identifier=preprocessing_identifier,
        preprocessing={
            "schema_version": 1,
            "fixture_preprocessing": preprocessing_identifier,
            "input_variant": input_variant,
        },
        cache_recipe={"schema_version": 1, "identifier": f"{representation_id}_fixture_v1"},
        dtype=str(matrix.dtype),
        feature_dimension=feature_dimension,
        package_versions={"fixture": "1"},
        matrix_key=matrix_key,
        provenance_scope="stage_eligible",
    )
    atomic_save_npz_with_sidecar(path, arrays=arrays, metadata=metadata)
    return verify_frozen_cache_sidecar(path)


def _bundle(tmp_path: Path) -> _Bundle:
    root = tmp_path / "project"
    root.mkdir()
    cache_root = root / "cache"
    cache_root.mkdir()
    manifest = _write(root / "data" / "manifest.parquet", "manifest")
    manifest_sha = sha256_file(manifest)
    pathology_audit = _write(root / "evidence" / "pathology.json", "{}")

    sample_ids: list[str] = []
    group_ids: list[str] = []
    official_folds: list[int] = []
    labels: list[int] = []
    for fold in (1, 2, 3):
        for group in range(3):
            for label in range(5):
                sample_ids.append(f"f{fold}-g{group}-c{label}")
                group_ids.append(f"f{fold}-g{group}")
                official_folds.append(fold)
                labels.append(label)
    sample_array = np.asarray(sample_ids, dtype=np.str_)
    group_array = np.asarray(group_ids, dtype=np.str_)
    fold_array = np.asarray(official_folds, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    n = len(sample_array)
    rgb = np.arange(n * 6 * 6 * 3, dtype=np.uint8).reshape(n, 6, 6, 3)
    masks = np.zeros((n, 6, 6), dtype=bool)
    masks[:, 2:4, 2:4] = True
    raw_inventory_sha256 = "a" * 64
    crop_path = cache_root / "crops.npz"
    crop_verification = _publish_test_cache(
        crop_path,
        arrays={
            "sample_ids": sample_array,
            "context_rgb": rgb,
            "target_masks": masks,
            "target_highlighted_rgb": rgb.copy(),
        },
        manifest_sha256=manifest_sha,
        raw_inventory_sha256=raw_inventory_sha256,
        representation_id="pannuke_component_covering_target_crops",
        input_variant=(
            "context_rgb_plus_component_covering_projected_binary_target_mask_and_"
            "raw_instance_identity"
        ),
        encoder_identifier="pannuke_component_covering_target_crop_v2",
        weight_identifier="unlearned:test_crop_fixture",
        weights_sha256="d" * 64,
        preprocessing_identifier="pannuke_component_covering_dynamic_square_crop_v2",
        matrix_key="context_rgb",
    )
    crop_metadata = crop_verification.sidecar_path

    config = complete_confirmatory_config()
    config["data"]["fold_assignment_labels"] = "pre_corruption_label"
    config["data"]["reference_group_selection_algorithm"] = (
        "deterministic_group_greedy_class_distribution_v1"
    )
    sample_order_sha = canonical_sha256(sample_ids)
    config["data"]["analysis_manifest_authority"] = {
        "canonical_manifest_sha256": manifest_sha,
        "analysis_eligible_sample_order_sha256": sample_order_sha,
        "analysis_eligible_sample_count": n,
    }
    required_frozen = [
        value
        for value in config["scenarios"]
        if value["family"] != "cnn" and value["required"] is True
    ]
    scenario_features: dict[str, np.ndarray[Any, Any]] = {}
    feature_specs: list[ConfirmatoryFrozenFeatureCacheSpec] = []
    availability: list[ConfirmatoryFrozenFeatureAvailability] = []
    frozen_records: dict[str, dict[str, Any]] = {}
    frozen_verifications: dict[str, FrozenCacheVerification] = {}
    draft_records = {str(value["id"]): value for value in config["cache_provenance"]}
    for index, scenario in enumerate(required_frozen):
        scenario_id = str(scenario["id"])
        provenance_id = str(scenario["cache_provenance_id"])
        draft_record = draft_records[provenance_id]
        rng = np.random.default_rng(100 + index)
        values = rng.normal(size=(n, 8)).astype(np.float64)
        scenario_features[scenario_id] = values
        path = cache_root / f"{scenario_id}.npz"
        verification = _publish_test_cache(
            path,
            arrays={"sample_ids": sample_array, "values": values},
            manifest_sha256=manifest_sha,
            raw_inventory_sha256=raw_inventory_sha256,
            representation_id=str(scenario["representation_id"]),
            input_variant=str(scenario["input_variant"]),
            encoder_identifier=str(scenario["encoder"]),
            weight_identifier=str(draft_record["weight_identifier"]),
            weights_sha256=str(draft_record["weights_sha256"]),
            preprocessing_identifier=str(draft_record["preprocessing_identifier"]),
            matrix_key="values",
        )
        frozen_verifications[scenario_id] = verification
        frozen_records[provenance_id] = confirmatory_cache_provenance_record(
            verification.metadata,
            record_id=provenance_id,
            bind_sidecar_semantics=True,
        )
        feature_specs.append(
            ConfirmatoryFrozenFeatureCacheSpec(
                scenario_id=scenario_id,
                cache_path=path,
                expected_cache_sha256=verification.cache_file_sha256,
                expected_metadata_sha256=verification.sidecar_file_sha256,
                expected_weight_sha256=str(verification.metadata["weights_sha256"]),
            )
        )
        availability.append(
            ConfirmatoryFrozenFeatureAvailability(
                scenario_id=scenario_id,
                family=str(scenario["family"]),
                required=True,
                available=True,
                blocker=None,
                cache_path=str(path),
                cache_sha256=verification.cache_file_sha256,
                metadata_path=str(verification.sidecar_path),
                metadata_sha256=verification.sidecar_file_sha256,
                manifest_binding="feature_sidecar_manifest_sha256",
                weight_sha256=str(verification.metadata["weights_sha256"]),
            )
        )

    pathology = next(
        value for value in config["scenarios"] if value["id"] == "pathology_frozen_logistic"
    )
    availability.append(
        ConfirmatoryFrozenFeatureAvailability(
            scenario_id="pathology_frozen_logistic",
            family="pathology_frozen",
            required=False,
            available=False,
            blocker="frozen pathology audit found no accessible encoder",
            cache_path=None,
            cache_sha256=None,
            metadata_path=None,
            metadata_sha256=None,
            manifest_binding=None,
            weight_sha256=None,
        )
    )
    pathology["availability_audit_sha256"] = "b" * 64
    context = frozen_verifications["imagenet_frozen_logistic"]
    cnn_records = derive_confirmatory_cnn_logical_provenance(
        crop_verification,
        weight_identifier=str(context.metadata["weight_identifier"]),
        weights_sha256=str(context.metadata["weights_sha256"]),
        input_size=int(config["training"]["input_size"]),
    )
    replacements = {**frozen_records, **cnn_records}
    for record in config["cache_provenance"]:
        if record["id"] == "pathology_context_embedding_cache":
            record["sample_order_sha256"] = sample_order_sha
            record["manifest_sha256"] = manifest_sha
    config["cache_provenance"] = [
        replacements.get(str(record["id"]), record) for record in config["cache_provenance"]
    ]

    controls = confirmatory_execution_controls_from_frozen_config(config)
    rotations: list[PanNukeConfirmatoryRotationInputs] = []
    all_indices = np.arange(n, dtype=np.int64)
    for outer_fold in controls.official_folds:
        final_indices = all_indices[fold_array == outer_fold]
        development = all_indices[fold_array != outer_fold]
        reference_group = str(group_array[development][0])
        reference_indices = development[group_array[development] == reference_group]
        audit_indices = development[group_array[development] != reference_group]
        observed = label_array.copy()
        corruption = apply_controlled_corruption(
            label_array[audit_indices],
            sample_ids=tuple(str(value) for value in sample_array[audit_indices]),
            group_ids=tuple(str(value) for value in group_array[audit_indices]),
            rate=0.10,
            mechanism="symmetric_random_corruption",
            seed=404,
            n_classes=5,
        )
        observed[audit_indices] = corruption.observed_labels
        rotations.append(
            PanNukeConfirmatoryRotationInputs(
                outer_fold=outer_fold,
                split_seed=controls.split_seed,
                audit=_partition(
                    role="audit",
                    indices=audit_indices,
                    sample_ids=sample_array,
                    group_ids=group_array,
                    rgb=rgb,
                    masks=masks,
                    labels=label_array,
                    observed=observed,
                    scenario_features=scenario_features,
                ),
                reference_validation=_partition(
                    role="reference_validation",
                    indices=reference_indices,
                    sample_ids=sample_array,
                    group_ids=group_array,
                    rgb=rgb,
                    masks=masks,
                    labels=label_array,
                    observed=label_array,
                    scenario_features=scenario_features,
                ),
                final_reference=_partition(
                    role="final_reference",
                    indices=final_indices,
                    sample_ids=sample_array,
                    group_ids=group_array,
                    rgb=rgb,
                    masks=masks,
                    labels=label_array,
                    observed=label_array,
                    scenario_features=scenario_features,
                ),
            )
        )
    prepared = PanNukeConfirmatoryInputs(
        config_sha256=controls.config_semantic_sha256,
        manifest_sha256=manifest_sha,
        raw_inventory_sha256="a" * 64,
        crop_cache_path=str(crop_path),
        crop_cache_sha256=sha256_file(crop_path),
        crop_metadata_path=str(crop_metadata),
        crop_metadata_sha256=sha256_file(crop_metadata),
        rotations=tuple(rotations),
        frozen_feature_availability=tuple(availability),
        execution_mode="real_study",
        study_outcome_eligible=True,
        ineligibility_reasons=(),
        eligibility_provenance={"semantic_sha256": "c" * 64},
    )
    prepared.validate(official_folds=controls.official_folds, oof_splits=controls.n_splits)

    freeze = root / "freeze"
    freeze.mkdir()
    primary_run = root / "primary-run"
    primary_run.mkdir()
    primary_gate = PrimaryExecutionGateEvidence(
        freeze_directory=freeze,
        base_freeze_directory=freeze,
        freeze_artifact_root_sha256="0" * 64,
        freeze_manifest_sha256="1" * 64,
        preregistration_sha256="2" * 64,
        frozen_primary_config_sha256="3" * 64,
        frozen_confirmatory_config_sha256="4" * 64,
        primary_config_semantic_sha256="5" * 64,
        confirmatory_config_semantic_sha256=controls.config_semantic_sha256,
        primary_matrix_cell_count=1,
        primary_required_cell_count=1,
        confirmatory_matrix_cell_count=len(controls.plan.cells),
        pilot_run_id="pilot",
        pilot_artifact_root_sha256="6" * 64,
        dataset_sha256="7" * 64,
        manifest_sha256=manifest_sha,
        duplicate_audit_sha256="8" * 64,
        pathology_encoder_audit_sha256="b" * 64,
        source_tree_root_sha256="9" * 64,
    )
    gate = ConfirmatoryExecutionGateEvidence(
        primary_gate=primary_gate,
        primary_run_directory=primary_run,
        primary_run_id="primary",
        primary_artifact_root_sha256="a" * 64,
        primary_completion_evidence_sha256="d" * 64,
        primary_reconciliation_sha256="e" * 64,
        completed_required_cell_count=1,
    )
    return _Bundle(
        root=root,
        config=config,
        prepared=prepared,
        feature_specs=tuple(feature_specs),
        crop_path=crop_path,
        crop_metadata_sha256=sha256_file(crop_metadata),
        manifest=manifest,
        pathology_audit=pathology_audit,
        gate=gate,
    )


def _naive_cnn_preflight_fingerprints(
    prepared: PanNukeConfirmatoryInputs,
    bridge: Any,
    controls: Any,
) -> dict[str, dict[str, dict[str, str]]]:
    """Reproduce the former per-cell implementation as a regression oracle."""

    source_by_fold = {value.outer_fold: value for value in prepared.rotations}
    core_by_fold = {value.outer_fold: value for value in bridge.rotations}
    fingerprints: dict[str, dict[str, dict[str, str]]] = {}
    for cell in controls.plan.cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        if scenario.family != "cnn":
            continue
        source = source_by_fold[cell.outer_fold]
        corruption = core_by_fold[cell.outer_fold].corruptions[cell.corruption_cell_id]
        folds = make_group_stratified_folds(
            source.audit.pre_corruption_labels,
            source.audit.group_ids,
            n_splits=controls.n_splits,
            class_order=CLASS_ORDER,
            seed=controls.split_seed,
        )
        uses_target_mask = scenario.input_variant == "context_rgb_plus_binary_target_mask"
        fingerprints[cell.cell_id] = {
            str(fold.fold_id): confirmatory_cnn_data_and_split_sha256(
                source.audit.context_rgb[fold.train_indices],
                corruption.observed_labels[fold.train_indices],
                training_sample_ids=np.asarray(source.audit.sample_ids, dtype=np.str_)[
                    fold.train_indices
                ],
                training_group_ids=np.asarray(source.audit.group_ids, dtype=np.str_)[
                    fold.train_indices
                ],
                reference_validation_images=source.reference_validation.context_rgb,
                reference_validation_labels=(source.reference_validation.pre_corruption_labels),
                reference_validation_sample_ids=(source.reference_validation.sample_ids),
                reference_validation_group_ids=(source.reference_validation.group_ids),
                input_variant=cast(Any, scenario.input_variant),
                training_target_masks=(
                    source.audit.target_masks[fold.train_indices] if uses_target_mask else None
                ),
                reference_validation_target_masks=(
                    source.reference_validation.target_masks if uses_target_mask else None
                ),
            )
            for fold in folds
        }
    return fingerprints


def _synthetic_published_t0_readiness(project_root: Path) -> tuple[Any, Any]:
    namespace = project_root / "artifacts" / "original_confirmatory_technical_authorities"
    authority_directory = namespace / "synthetic-authority"
    binding_unsigned = {
        "schema_version": 1,
        "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
        "namespace_directory": str(namespace),
        "namespace_claim_sha256": "1" * 64,
        "review_attempt_claim_sha256": "2" * 64,
        "technical_authority": {"binding_sha256": "3" * 64},
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
    }
    binding = {
        **binding_unsigned,
        "binding_sha256": canonical_sha256(binding_unsigned),
    }
    request = SimpleNamespace(
        technical_authority_namespace_directory=namespace,
        technical_authority_namespace_claim_sha256="1" * 64,
        technical_authority_directory=authority_directory,
        technical_authority_artifact_root_sha256="4" * 64,
        technical_authorization_sha256="5" * 64,
        published_technical_authority_lifecycle_binding_sha256=(binding["binding_sha256"]),
    )
    readiness = SimpleNamespace(
        published_t0_pins=SimpleNamespace(
            namespace_directory=namespace,
            namespace_claim_sha256="1" * 64,
            technical_authority_directory=authority_directory,
            technical_authority_artifact_root_sha256="4" * 64,
            technical_authorization_sha256="5" * 64,
            published_technical_authority_lifecycle_binding_sha256=(binding["binding_sha256"]),
        ),
        verified_published_technical_authority=SimpleNamespace(lifecycle_binding=lambda: binding),
    )
    return request, readiness


def test_capsule_runtime_cross_checks_all_static_v3_published_t0_pins(
    tmp_path: Path,
) -> None:
    request, readiness = _synthetic_published_t0_readiness(tmp_path.resolve())
    _require_exact_published_t0_lifecycle_pins(request, readiness)


@pytest.mark.parametrize(
    "field",
    [
        "namespace_directory",
        "namespace_claim_sha256",
        "technical_authority_directory",
        "technical_authority_artifact_root_sha256",
        "technical_authorization_sha256",
        "published_technical_authority_lifecycle_binding_sha256",
    ],
)
def test_capsule_runtime_rejects_each_live_published_t0_pin_substitution(
    tmp_path: Path,
    field: str,
) -> None:
    request, readiness = _synthetic_published_t0_readiness(tmp_path.resolve())
    pins = readiness.published_t0_pins
    original = getattr(pins, field)
    substitute = tmp_path.resolve() / "substituted" if isinstance(original, Path) else "f" * 64
    setattr(pins, field, substitute)

    with pytest.raises(
        ConfirmatoryStudyRunnerError,
        match="STATIC-v3 capsule request",
    ):
        _require_exact_published_t0_lifecycle_pins(request, readiness)


def test_capsule_runtime_rejects_review_attempt_mutation_hidden_behind_old_composite_hash(
    tmp_path: Path,
) -> None:
    request, readiness = _synthetic_published_t0_readiness(tmp_path.resolve())
    binding = readiness.verified_published_technical_authority.lifecycle_binding()
    binding["review_attempt_claim_sha256"] = "f" * 64

    with pytest.raises(
        ConfirmatoryStudyRunnerError,
        match="STATIC-v3 capsule request",
    ):
        _require_exact_published_t0_lifecycle_pins(request, readiness)


def test_bridge_builds_all_rotations_and_binds_corruption_and_provenance(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)

    result = bridge_pannuke_confirmatory_inputs(
        bundle.prepared,
        controls,
        pathology_encoder_audit_sha256="b" * 64,
    )

    assert tuple(value.outer_fold for value in result.rotations) == (1, 2, 3)
    assert set(result.frozen_blockers) == {"pathology_frozen_logistic"}
    assert all(len(value) == 64 for value in result.as_dict().values() if isinstance(value, str))
    for rotation in result.rotations:
        rotation.validate(controls)
        assert set(rotation.corruptions) == {
            "clean_reference_cell",
            "symmetric_ten_percent",
        }
        assert not rotation.corruptions["clean_reference_cell"].is_injected_corruption.any()
        assert int(rotation.corruptions["symmetric_ten_percent"].is_injected_corruption.sum()) == 3
        assert set(rotation.audit_official_folds).isdisjoint({rotation.outer_fold})
        assert set(rotation.final_official_folds) == {rotation.outer_fold}


def test_cnn_preflight_deduplicates_seed_invariant_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)
    bridge = bridge_pannuke_confirmatory_inputs(
        bundle.prepared,
        controls,
        pathology_encoder_audit_sha256="b" * 64,
    )
    expected = _naive_cnn_preflight_fingerprints(bundle.prepared, bridge, controls)
    original = confirmatory_runner_module.confirmatory_cnn_data_and_split_sha256
    call_count = 0

    def counted_fingerprint(*args: Any, **kwargs: Any) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        confirmatory_runner_module,
        "confirmatory_cnn_data_and_split_sha256",
        counted_fingerprint,
    )
    actual = _confirmatory_cnn_preflight_fingerprints(
        bundle.prepared,
        bridge,
        controls,
    )

    cnn_cells = [
        cell
        for cell in controls.plan.cells
        if controls.scenarios_by_id[cell.scenario_id].family == "cnn"
    ]
    cells_by_inputs: dict[tuple[int, str, str], list[str]] = {}
    for cell in cnn_cells:
        scenario = controls.scenarios_by_id[cell.scenario_id]
        key = (cell.outer_fold, cell.corruption_cell_id, scenario.input_variant)
        cells_by_inputs.setdefault(key, []).append(cell.cell_id)

    assert call_count == len(cells_by_inputs) * controls.n_splits
    assert actual == expected
    repeated_cell_ids = next(cell_ids for cell_ids in cells_by_inputs.values() if len(cell_ids) > 1)
    first_cell, second_cell = repeated_cell_ids[:2]
    assert actual[first_cell] is not actual[second_cell]
    for fold_id in actual[first_cell]:
        assert actual[first_cell][fold_id] is not actual[second_cell][fold_id]


def test_bridge_rejects_same_count_corruption_from_a_different_assignment(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)
    source = bundle.prepared.rotations[0]
    audit = source.audit
    observed = audit.pre_corruption_labels.copy()
    injected = np.zeros(len(observed), dtype=bool)
    injected[-3:] = True
    observed[-3:] = (observed[-3:] + 1) % 5
    altered_audit = replace(
        audit,
        observed_labels=_readonly(observed, np.int64),
        is_injected_corruption=_readonly(injected, np.bool_),
        corruption_types=tuple(
            "symmetric_random_corruption" if value else "none" for value in injected
        ),
    )
    altered_audit.validate()
    altered_rotation = replace(source, audit=altered_audit)
    altered = replace(
        bundle.prepared,
        rotations=(altered_rotation, *bundle.prepared.rotations[1:]),
    )

    with pytest.raises(ValueError, match="do not replay from frozen mechanism/rate/seed"):
        bridge_pannuke_confirmatory_inputs(
            altered,
            controls,
            pathology_encoder_audit_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("record_id", "field"),
    [
        ("cnn_context_rgb_cache", "encoder_metadata_sha256"),
        ("cnn_context_target_mask_cache", "preprocessing_sha256"),
        ("imagenet_context_embedding_cache", "encoder_metadata_sha256"),
    ],
)
def test_bridge_recomputes_logical_and_frozen_cache_provenance(
    tmp_path: Path,
    record_id: str,
    field: str,
) -> None:
    bundle = _bundle(tmp_path)
    config = deepcopy(bundle.config)
    record = next(value for value in config["cache_provenance"] if value["id"] == record_id)
    record[field] = "0" * 64
    controls = confirmatory_execution_controls_from_frozen_config(config)
    prepared = replace(bundle.prepared, config_sha256=controls.config_semantic_sha256)

    with pytest.raises(ValueError, match="provenance does not recompute"):
        bridge_pannuke_confirmatory_inputs(
            prepared,
            controls,
            pathology_encoder_audit_sha256=(
                bundle.gate.primary_gate.pathology_encoder_audit_sha256
            ),
        )


def test_real_frozen_runner_produces_group_safe_pre_label_fixed_oof(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)
    bridge = bridge_pannuke_confirmatory_inputs(
        bundle.prepared,
        controls,
        pathology_encoder_audit_sha256="b" * 64,
    )
    rotation = bridge.rotations[0]
    cell = next(
        value
        for value in controls.plan.cells
        if value.outer_fold == rotation.outer_fold
        and value.corruption_cell_id == "symmetric_ten_percent"
        and value.scenario_id == "imagenet_frozen_logistic"
        and value.model_seed == 303
    )
    scenario = controls.scenarios_by_id[cell.scenario_id]
    request = ConfirmatoryCellRequest(
        cell=cell,
        scenario=scenario,
        corruption=rotation.corruptions[cell.corruption_cell_id],
        inputs=ConfirmatoryRunnerInputs.from_rotation(rotation),
        controls=controls,
        checkpoint_directory=tmp_path / "checkpoints",
        checkpoint_execution_contract=None,
        checkpoint_directives=(),
        cpu_test_only=True,
    )

    execution = run_confirmatory_frozen_feature_oof(request)

    execution.oof_result.validate()
    assert execution.execution_mode == "cpu_test_only"
    assert execution.study_outcome_eligible is False
    assert execution.oof_result.fold_assignment_label_source == "pre_corruption_label"
    np.testing.assert_array_equal(
        execution.oof_result.fold_assignment_labels,
        request.corruption.pre_corruption_labels,
    )
    assert execution.oof_result.sample_ids == request.inputs.audit_sample_ids
    assert execution.oof_result.group_ids == request.inputs.audit_group_ids
    assert (
        execution.evidence["frozen_feature_provenance_sha256"]
        == request.inputs.frozen_feature_provenance[scenario.representation_id].semantic_sha256
    )

    eligible = run_confirmatory_frozen_feature_oof(replace(request, cpu_test_only=False))
    eligible.oof_result.validate()
    assert eligible.execution_mode == "real_study_cpu"
    assert eligible.study_outcome_eligible is True
    assert eligible.evidence["estimator_device"] == "cpu"
    assert eligible.evidence["cuda_execution_gate_required"] is False


def test_default_dependencies_install_production_finalizer() -> None:
    dependencies = ConfirmatoryRunnerDependencies()

    assert callable(dependencies.stage_finalizer)
    assert dependencies.stage_finalizer.__name__ == "_default_stage_finalizer"


def test_production_finalizer_replays_checksum_bound_restoration_arrays(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    controls = confirmatory_execution_controls_from_frozen_config(bundle.config)
    bridge = bridge_pannuke_confirmatory_inputs(
        bundle.prepared,
        controls,
        pathology_encoder_audit_sha256=(bundle.gate.primary_gate.pathology_encoder_audit_sha256),
    )

    run_directory = tmp_path / "scientific-tree"
    artifacts = execute_confirmatory_matrix(
        bridge.rotations,
        controls.plan,
        controls,
        output_directory=run_directory,
        image_oof_runner=_synthetic_image_runner,
        frozen_oof_runner=_synthetic_frozen_runner,
        frozen_blockers=bridge.frozen_blockers,
        artifact_scope=SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
        cpu_test_only=True,
    )
    statistics = aggregate_confirmatory_statistics(run_directory, controls)
    atomic_write_json(
        run_directory / "confirmatory_input_bindings.json",
        {
            "schema_version": 1,
            "config_semantic_sha256": controls.config_semantic_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "bridge": bridge.as_dict(),
        },
    )
    finalize_confirmatory_stage(
        run_directory=run_directory,
        matrix_artifacts=artifacts,
        statistics_artifacts=statistics,
        prepared_inputs=bundle.prepared,
        bridge=bridge,
        controls=controls,
        gate_evidence=bundle.gate,
    )
    _validate_restoration_source_binding(
        run_directory,
        prepared_inputs=bundle.prepared,
        bridge=bridge,
        controls=controls,
    )
    first_rotation = bundle.prepared.rotations[0]
    for partition in (
        first_rotation.audit,
        first_rotation.reference_validation,
        first_rotation.final_reference,
    ):
        feature_values = partition.frozen_features[0].values
        original_value = float(feature_values[0, 0])
        feature_values.setflags(write=True)
        feature_values[0, 0] = original_value + 1.0
        feature_values.setflags(write=False)
        with pytest.raises(
            ConfirmatoryStudyRunnerError,
            match="inputs changed after checksum-bound bridge preflight",
        ):
            _validate_restoration_source_binding(
                run_directory,
                prepared_inputs=bundle.prepared,
                bridge=bridge,
                controls=controls,
            )
        feature_values.setflags(write=True)
        feature_values[0, 0] = original_value
        feature_values.setflags(write=False)
    assert artifacts.artifact_manifest_path.name == "matrix_core_artifact_manifest.json"
    assert (run_directory / "restoration_metrics.json").is_file()
    assert (run_directory / "restoration_evidence.npz").is_file()
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "potentially inconsistent annotation" in report
    assert "recommended for expert review" in report

    evidence_path = run_directory / "restoration_evidence.npz"
    with np.load(evidence_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    probability_key = next(key for key in arrays if "__probabilities__" in key)
    arrays[probability_key] = arrays[probability_key].copy()
    arrays[probability_key].flat[0] += 1e-6
    evidence_path.chmod(0o666)
    np.savez_compressed(evidence_path, **arrays)
    with pytest.raises(ConfirmatoryStudyRunnerError, match="deterministic replay"):
        _validate_restoration_source_binding(
            run_directory,
            prepared_inputs=bundle.prepared,
            bridge=bridge,
            controls=controls,
        )


def test_outcome_entrypoint_rejects_injected_dependencies_before_execution(
    tmp_path: Path,
) -> None:
    dependencies = replace(
        ConfirmatoryRunnerDependencies(),
        restoration_verifier=lambda **_: None,
    )

    with pytest.raises(
        ConfirmatoryStudyRunnerError,
        match="forbids injected dependencies",
    ):
        execute_confirmatory_study(
            gate_evidence=None,  # type: ignore[arg-type]
            primary_run_directory=tmp_path / "primary",
            project_root=tmp_path,
            freeze_directory=tmp_path / "freeze",
            dataset_path=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.parquet",
            duplicate_audit_path=tmp_path / "duplicates.json",
            pathology_encoder_audit_path=tmp_path / "pathology.json",
            frozen_primary_config_path=tmp_path / "primary.yaml",
            frozen_confirmatory_config_path=tmp_path / "confirmatory.yaml",
            crop_cache_path=tmp_path / "crops.npz",
            expected_crop_cache_sha256="0" * 64,
            expected_crop_metadata_sha256="0" * 64,
            expected_raw_inventory_sha256="0" * 64,
            frozen_feature_caches=(),
            runs_root=tmp_path / "runs",
            dependencies=dependencies,
        )

    assert not (tmp_path / "runs").exists()


def test_fresh_outcome_entrypoint_rejects_retry_lineage_before_preflight(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ConfirmatoryStudyRunnerError,
        match="fresh original-confirmatory runner requires retry_of_run_id=null",
    ):
        execute_confirmatory_study(
            gate_evidence=None,  # type: ignore[arg-type]
            primary_run_directory=tmp_path / "primary",
            project_root=tmp_path,
            freeze_directory=tmp_path / "freeze",
            dataset_path=tmp_path / "dataset",
            manifest_path=tmp_path / "manifest.parquet",
            duplicate_audit_path=tmp_path / "duplicates.json",
            pathology_encoder_audit_path=tmp_path / "pathology.json",
            frozen_primary_config_path=tmp_path / "primary.yaml",
            frozen_confirmatory_config_path=tmp_path / "confirmatory.yaml",
            crop_cache_path=tmp_path / "crops.npz",
            expected_crop_cache_sha256="0" * 64,
            expected_crop_metadata_sha256="0" * 64,
            expected_raw_inventory_sha256="0" * 64,
            frozen_feature_caches=(),
            runs_root=tmp_path / "runs",
            retry_of_run_id="predecessor",
            lifecycle_readiness_run_directory=tmp_path / "readiness",
        )

    assert not (tmp_path / "runs").exists()


def test_legacy_runner_profile_guard_rejects_resource_schema() -> None:
    from histo_audit.experiment.confirmatory_runner import (
        _require_legacy_confirmatory_execution_profile,
    )

    resource_config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "confirmatory_resource_bounded_amended.yaml"
    )
    resource_plan = build_confirmatory_matrix_plan(resource_config)

    with pytest.raises(ConfirmatoryStudyRunnerError, match="schema-v2 frozen"):
        _require_legacy_confirmatory_execution_profile(resource_config, resource_plan)
