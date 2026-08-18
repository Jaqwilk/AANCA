"""Fail-closed tests for deterministic M7 config finalisation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from histo_audit.config import load_config
from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
)
from histo_audit.experiment import m7_config_finalization as finalizer_module
from histo_audit.experiment import pilot_derived_parameters as pilot_parameters_module
from histo_audit.experiment.m7_config_finalization import (
    M7CachePaths,
    M7ConfigFinalizationError,
    M7ConfigFinalizationResult,
    finalise_m7_configs,
)
from histo_audit.experiment.reference_groups import (
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    validate_frozen_confirmatory_config,
    validate_frozen_primary_config,
)
from histo_audit.representations.cache_provenance import (
    FrozenCacheVerification,
    atomic_save_npz_with_sidecar,
    build_frozen_cache_metadata,
    canonical_sha256,
    verify_frozen_cache_sidecar,
)
from histo_audit.representations.pathology import (
    unavailable_optional_pathology_cache_provenance,
)
from histo_audit.utils.run_tracking import RunTracker, sha256_file, verify_run_integrity

_MANIFEST_SHA = "a" * 64
_RAW_SHA = "b" * 64
_WEIGHT_SHA = "c" * 64
_WEIGHT_ID = "ResNet18_Weights.IMAGENET1K_V1"


@dataclass(frozen=True, slots=True)
class _Bundle:
    root: Path
    caches: M7CachePaths
    pathology: Path
    independence: Path
    pilot: Path
    primary_draft: Path
    confirmatory_draft: Path
    primary_output: Path
    confirmatory_output: Path


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _cache_metadata(
    *,
    sample_ids: np.ndarray[Any, Any],
    representation_id: str,
    input_variant: str,
    encoder_identifier: str,
    matrix_key: str,
    feature_dimension: int | list[int],
    base_metadata: dict[str, Any] | None = None,
    encoder_metadata: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    cache_recipe: dict[str, Any] | None = None,
    weight_identifier: str = _WEIGHT_ID,
    weights_sha256: str = _WEIGHT_SHA,
    dtype: str = "float32",
) -> dict[str, Any]:
    return build_frozen_cache_metadata(
        base_metadata={"schema_version": 1, **(base_metadata or {})},
        sample_ids=sample_ids,
        manifest_sha256=_MANIFEST_SHA,
        raw_inventory_sha256=_RAW_SHA,
        representation_id=representation_id,
        input_variant=input_variant,
        encoder_identifier=encoder_identifier,
        encoder_metadata=encoder_metadata or {"identifier": f"{representation_id}_encoder"},
        encoder_implementation={
            "module": "tests.test_m7_config_finalization",
            "entrypoint": representation_id,
            "source_file_sha256": "d" * 64,
        },
        weight_identifier=weight_identifier,
        weights_sha256=weights_sha256,
        preprocessing_identifier=f"{representation_id}_preprocessing_v1",
        preprocessing=preprocessing or {"identifier": f"{representation_id}_preprocessing_v1"},
        cache_recipe=cache_recipe or {"identifier": f"{representation_id}_cache_v1"},
        dtype=dtype,
        feature_dimension=feature_dimension,
        package_versions={"python": "test", "numpy": np.__version__},
        matrix_key=matrix_key,
        provenance_scope="stage_eligible",
    )


def _save_cache(
    path: Path,
    *,
    arrays: dict[str, np.ndarray[Any, Any]],
    metadata: dict[str, Any],
) -> FrozenCacheVerification:
    atomic_save_npz_with_sidecar(path, arrays=arrays, metadata=metadata)
    return verify_frozen_cache_sidecar(path)


def _crop_binding(crop: FrozenCacheVerification) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "binding_type": "fixture_crop_binding_v1",
        "crop_cache_file_sha256": crop.cache_file_sha256,
        "crop_cache_sidecar_file_sha256": crop.sidecar_file_sha256,
        "crop_cache_content_sha256": crop.metadata["cache_content_sha256"],
        "crop_manifest_sha256": crop.metadata["manifest_sha256"],
        "raw_inventory_sha256": crop.metadata["raw_inventory_sha256"],
        "sample_order_sha256": crop.metadata["sample_order_sha256"],
    }


def _engineered_binding(
    engineered: FrozenCacheVerification,
    crop_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "binding_type": "pannuke_engineered_feature_cache_v1",
        "engineered_cache_file_sha256": engineered.cache_file_sha256,
        "engineered_cache_sidecar_file_sha256": engineered.sidecar_file_sha256,
        "engineered_cache_content_sha256": engineered.metadata["cache_content_sha256"],
        "manifest_sha256": engineered.metadata["manifest_sha256"],
        "raw_inventory_sha256": engineered.metadata["raw_inventory_sha256"],
        "sample_order_sha256": engineered.metadata["sample_order_sha256"],
        "cache_array_sha256_by_name": engineered.metadata["cache_array_sha256_by_name"],
        "source_crop_cache_binding": crop_binding,
        "source_crop_cache_binding_sha256": canonical_sha256(crop_binding),
    }


def _arrays() -> dict[str, np.ndarray[Any, Any]]:
    groups = 15
    per_group = 5
    n = groups * per_group
    sample_ids = np.asarray([f"sample-{index:03d}" for index in range(n)], dtype=np.str_)
    group_ids = np.repeat(
        np.asarray([f"patch-{index:02d}" for index in range(groups)], dtype=np.str_),
        per_group,
    )
    labels = np.tile(np.arange(5, dtype=np.int64), groups)
    folds = np.repeat(np.asarray([1] * 6 + [2] * 6 + [3] * 3), per_group).astype(np.int64)
    rng = np.random.default_rng(144)
    context_rgb = rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
    masks = np.zeros((n, 8, 8), dtype=bool)
    masks[:, 2:6, 2:6] = True
    return {
        "sample_ids": sample_ids,
        "group_ids": group_ids,
        "pre_corruption_labels": labels,
        "official_folds": folds,
        "context_rgb": context_rgb,
        "target_masks": masks,
        "target_highlighted_rgb": context_rgb.copy(),
        "engineered": rng.normal(size=(n, 5)).astype(np.float64),
        "context": rng.normal(size=(n, 8)).astype(np.float32),
        "highlighted": rng.normal(size=(n, 8)).astype(np.float32),
    }


def _independence_payload(
    arrays: dict[str, np.ndarray[Any, Any]],
    engineered: FrozenCacheVerification,
    context: FrozenCacheVerification,
    highlighted: FrozenCacheVerification,
) -> dict[str, Any]:
    development = arrays["official_folds"] != 3
    reference_groups = deterministic_group_greedy_class_distribution_v1(
        arrays["pre_corruption_labels"][development],
        arrays["group_ids"][development],
        class_order=(0, 1, 2, 3, 4),
        fraction=0.10,
        seed=223,
    )
    audit = np.flatnonzero(development & ~np.isin(arrays["group_ids"], reference_groups))
    fitted_hash = canonical_sha256(
        {
            "sample_ids": arrays["sample_ids"][audit].tolist(),
            "group_ids": arrays["group_ids"][audit].tolist(),
        }
    )
    generator = FeatureSpaceEvidence.from_array(
        np.asarray(arrays["engineered"][:, :3], dtype=np.float64)[audit],
        representation_name="morphology_only_v1",
        family="morphology",
        implementation_hash=str(engineered.metadata["encoder_implementation_sha256"]),
        weights_hash=str(engineered.metadata["weights_sha256"]),
        preprocessing_hash=str(engineered.metadata["preprocessing_sha256"]),
        fitted_data_hash=fitted_hash,
    )
    rows = {
        "engineered_target_features": (
            arrays["engineered"][audit],
            engineered,
            "engineered",
            "not_independent",
        ),
        "imagenet_resnet18_context": (
            arrays["context"][audit],
            context,
            "imagenet",
            "verified_independent",
        ),
        "imagenet_resnet18_highlighted": (
            arrays["highlighted"][audit],
            highlighted,
            "imagenet",
            "verified_independent",
        ),
    }
    entries: dict[str, Any] = {}
    for identifier, (values, verification, family, decision) in rows.items():
        auditor = FeatureSpaceEvidence.from_array(
            values,
            representation_name=identifier,
            family=family,
            implementation_hash=str(verification.metadata["encoder_implementation_sha256"]),
            weights_hash=str(verification.metadata["weights_sha256"]),
            preprocessing_hash=str(verification.metadata["preprocessing_sha256"]),
            fitted_data_hash=fitted_hash,
        )
        evidence = FeatureIndependenceEvidence.create(
            matrix_version="m7_finalizer_fixture_v1",
            matrix_decision=decision,
            matrix_reason=f"fixture decision for {identifier}",
            generator=generator,
            auditor=auditor,
        )
        entries[identifier] = evidence.as_dict()
    return {"schema_version": 2, "entries": entries}


def _bundle(
    root: Path,
    *,
    mismatch_order: bool = False,
    wrong_morph_context: bool = False,
    alter_final_context: bool = False,
) -> _Bundle:
    root.mkdir(parents=True, exist_ok=True)
    arrays = _arrays()
    if alter_final_context:
        arrays["context"] = arrays["context"].copy()
        arrays["context"][arrays["official_folds"] == 3] += np.float32(7.0)
    cache_root = root / "artifacts" / "embeddings" / "pannuke"
    sample_ids = arrays["sample_ids"]
    crop = _save_cache(
        cache_root / "pannuke_crops.npz",
        arrays={
            key: arrays[key]
            for key in (
                "sample_ids",
                "group_ids",
                "pre_corruption_labels",
                "official_folds",
                "context_rgb",
                "target_masks",
                "target_highlighted_rgb",
            )
        },
        metadata=_cache_metadata(
            sample_ids=sample_ids,
            representation_id="pannuke_component_covering_target_crops",
            input_variant="context_rgb_plus_binary_target_mask_and_identity",
            encoder_identifier="pannuke_component_covering_target_crop_v2",
            matrix_key="context_rgb",
            feature_dimension=[8, 8, 3],
            weight_identifier="unlearned:crop",
            weights_sha256="4" * 64,
            dtype="uint8",
        ),
    )
    crop_binding = _crop_binding(crop)
    engineered = _save_cache(
        cache_root / "pannuke_engineered_features.npz",
        arrays={
            "sample_ids": sample_ids,
            "names": np.asarray(
                [
                    "morphology.area",
                    "morphology.eccentricity",
                    "morphology.solidity",
                    "colour.mean",
                    "texture.contrast",
                ],
                dtype=np.str_,
            ),
            "values": arrays["engineered"],
        },
        metadata=_cache_metadata(
            sample_ids=sample_ids,
            representation_id="engineered_target_features",
            input_variant="context_rgb_plus_binary_target_mask",
            encoder_identifier="engineered_target_features_v1",
            matrix_key="values",
            feature_dimension=5,
            base_metadata={"source_crop_cache_binding": crop_binding},
            weight_identifier="unlearned:engineered",
            weights_sha256="5" * 64,
            dtype="float64",
        ),
    )
    highlighted_ids = sample_ids[::-1] if mismatch_order else sample_ids
    highlighted_values = arrays["highlighted"][::-1] if mismatch_order else arrays["highlighted"]
    highlighted = _save_cache(
        cache_root / "pannuke_resnet18_target_highlighted_embeddings.npz",
        arrays={"sample_ids": highlighted_ids, "embeddings": highlighted_values},
        metadata=_cache_metadata(
            sample_ids=highlighted_ids,
            representation_id="imagenet_target_highlighted_embeddings",
            input_variant="target_highlighted_rgb",
            encoder_identifier="resnet18_imagenet1k_v1",
            matrix_key="embeddings",
            feature_dimension=8,
            base_metadata={"source_crop_cache_binding": crop_binding},
        ),
    )
    context = _save_cache(
        cache_root / "pannuke_resnet18_context_rgb_embeddings.npz",
        arrays={"sample_ids": sample_ids, "embeddings": arrays["context"]},
        metadata=_cache_metadata(
            sample_ids=sample_ids,
            representation_id="imagenet_resnet18_context_embeddings",
            input_variant="context_rgb",
            encoder_identifier="resnet18_imagenet1k_v1",
            matrix_key="embeddings",
            feature_dimension=8,
            base_metadata={"source_crop_cache_binding": crop_binding},
        ),
    )
    engineered_binding = _engineered_binding(engineered, crop_binding)
    wrong_hash = "f" * 64 if wrong_morph_context else context.cache_file_sha256
    morph_values = np.concatenate(
        (arrays["context"], arrays["engineered"][:, :3].astype(np.float32)), axis=1
    )
    morph = _save_cache(
        cache_root / "pannuke_resnet18_context_plus_target_morphometrics.npz",
        arrays={
            "sample_ids": sample_ids,
            "names": np.asarray([f"feature-{index}" for index in range(11)], dtype=np.str_),
            "values": morph_values,
        },
        metadata=_cache_metadata(
            sample_ids=sample_ids,
            representation_id="imagenet_context_embeddings_plus_target_morphometrics",
            input_variant="context_rgb_plus_target_morphometrics",
            encoder_identifier="resnet18_imagenet1k_v1_plus_target_morphometrics_v1",
            matrix_key="values",
            feature_dimension=11,
            base_metadata={"component_engineered_cache_binding": engineered_binding},
            encoder_metadata={
                "component_context_cache_file_sha256": wrong_hash,
                "component_context_cache_content_sha256": context.metadata["cache_content_sha256"],
                "component_context_encoder_metadata_sha256": context.metadata[
                    "encoder_metadata_sha256"
                ],
            },
            preprocessing={
                "context_embedding_preprocessing_identifier": context.metadata[
                    "preprocessing_identifier"
                ],
                "context_embedding_preprocessing_sha256": context.metadata["preprocessing_sha256"],
            },
            cache_recipe={
                "identifier": "context_plus_morphometrics_v1",
                "context_cache_recipe_sha256": context.metadata["cache_recipe_sha256"],
                "context_sample_order_sha256": context.metadata["sample_order_sha256"],
            },
        ),
    )

    pathology = root / "reports" / "pathology_encoder_availability.json"
    pathology_record = unavailable_optional_pathology_cache_provenance(
        sample_order_sha256=str(crop.metadata["sample_order_sha256"]),
        dataset_manifest_sha256=str(crop.metadata["manifest_sha256"]),
    )
    _write_json(
        pathology,
        {
            "status": "blocked",
            "selected_encoder": None,
            "blocker": "no encoder passed every frozen availability gate",
            "selection_rule": "select the first candidate passing every frozen gate",
            "records": [{"name": "fixture-pathology-encoder", "status": "blocked"}],
            "primary_cache_provenance": pathology_record,
        },
    )
    independence = root / "reports" / "representation_independence.json"
    _write_json(
        independence,
        _independence_payload(arrays, engineered, context, highlighted),
    )
    tracker = RunTracker.start(
        experiment_name="pannuke_pilot",
        config={
            "experiment_name": "pannuke_pilot",
            "seed": {"split": 223, "model": 227, "corruption": 404},
        },
        project_root=root,
        runs_root=root / "artifacts" / "runs",
        run_id="m7-finalizer-test-pilot",
        environment={"fixture": True},
    )
    tracker.write_metrics({"fixture": True})
    tracker.write_json(
        "selected_groups_and_samples.json",
        {
            "development_official_folds": [1, 2],
            "final_reference_class_labels_read": False,
            "final_reference_sample_ids_read": False,
            "final_reference_representations_extracted": False,
            "final_reference_outcomes_used": False,
        },
    )
    tracker.write_json(
        "final_reference_privacy_reconciliation.json",
        {
            "status": "passed",
            "policy": "final_reference_identity_and_outcome_nonpublication_v1",
            "final_reference_official_fold": 3,
        },
    )
    tracker.write_text("representations/pannuke_crops.npz", "fixture crop cache\n")
    tracker.write_text(
        "representations/pannuke_crops.npz.metadata.json",
        '{"fixture": true}\n',
    )
    tracker.write_text(
        "representations/pannuke_resnet18_target_highlighted_embeddings.npz",
        "fixture highlighted embeddings\n",
    )
    tracker.write_text(
        "representations/pannuke_resnet18_target_highlighted_embeddings.npz.metadata.json",
        '{"fixture": true}\n',
    )
    tracker.complete()
    integrity = verify_run_integrity(tracker.run_directory)
    assert integrity.valid
    assert integrity.actual_root_sha256 is not None

    draft = yaml.safe_load(Path("configs/primary.yaml").read_text(encoding="utf-8"))
    draft["data"]["analysis_manifest_authority"] = {
        "canonical_manifest_sha256": crop.metadata["manifest_sha256"],
        "analysis_eligible_sample_order_sha256": crop.metadata["sample_order_sha256"],
        "analysis_eligible_sample_count": crop.metadata["sample_count"],
    }
    pilot = root / "reports" / "pilot_derived_primary_parameters.json"
    pilot_sources = {
        field_name: sha256_file(tracker.run_directory / relative_path)
        for field_name, relative_path in finalizer_module._PILOT_SOURCE_ARTIFACTS.items()
    }
    _write_json(
        pilot,
        {
            "schema_version": 1,
            "producer_id": "pilot_derived_primary_parameters_v1",
            "producer_source_sha256": sha256_file(Path(pilot_parameters_module.__file__).resolve()),
            "class_order": [0, 1, 2, 3, 4],
            "source_pilot": {
                "run_id": tracker.run_id,
                "artifact_root_sha256": integrity.actual_root_sha256,
                "development_official_folds": [1, 2],
                "final_reference_policy": (
                    "no final-fold sample identifier, label, representation, or outcome read"
                ),
                **pilot_sources,
            },
            "confusion_targeted_corruption": draft["corruption"]["mechanisms"][
                "confusion_targeted_corruption"
            ],
            "group_conditional_corruption": draft["corruption"]["mechanisms"][
                "group_conditional_corruption"
            ],
        },
    )
    draft["pilot_derived_parameters"]["source_pilot_run_id"] = tracker.run_id
    draft["pilot_derived_parameters"]["source_pilot_artifact_root_sha256"] = (
        integrity.actual_root_sha256
    )
    draft["pilot_derived_parameters"]["sha256"] = sha256_file(pilot)
    draft_root = root / "drafts"
    primary_draft = draft_root / "primary.yaml"
    confirmatory_draft = draft_root / "confirmatory.yaml"
    primary_draft.parent.mkdir(parents=True, exist_ok=True)
    primary_draft.write_text(
        yaml.safe_dump(draft, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    confirmatory = yaml.safe_load(Path("configs/confirmatory.yaml").read_text(encoding="utf-8"))
    confirmatory["data"]["analysis_manifest_authority"] = dict(
        draft["data"]["analysis_manifest_authority"]
    )
    confirmatory_draft.write_text(
        yaml.safe_dump(confirmatory, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return _Bundle(
        root=root,
        caches=M7CachePaths(
            crop=crop.cache_path,
            engineered=engineered.cache_path,
            highlighted=highlighted.cache_path,
            context=context.cache_path,
            context_morphometrics=morph.cache_path,
        ),
        pathology=pathology,
        independence=independence,
        pilot=pilot,
        primary_draft=primary_draft,
        confirmatory_draft=confirmatory_draft,
        primary_output=root / "final" / "primary.yaml",
        confirmatory_output=root / "final" / "confirmatory.yaml",
    )


def _finalise(bundle: _Bundle, **kwargs: Any) -> M7ConfigFinalizationResult:
    return finalise_m7_configs(
        project_root=bundle.root,
        primary_draft_path=bundle.primary_draft,
        confirmatory_draft_path=bundle.confirmatory_draft,
        cache_paths=bundle.caches,
        pathology_audit_path=bundle.pathology,
        independence_matrix_path=bundle.independence,
        pilot_report_path=bundle.pilot,
        primary_output_path=bundle.primary_output,
        confirmatory_output_path=bundle.confirmatory_output,
        **kwargs,
    )


def test_finalises_both_configs_and_refuses_unapproved_overwrite(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = _finalise(bundle)

    primary = validate_frozen_primary_config(load_config(result.primary_output_path))
    confirmatory = validate_frozen_confirmatory_config(load_config(result.confirmatory_output_path))
    assert primary["status"] == confirmatory["status"] == "READY_FOR_FREEZE"
    assert (
        primary["data"]["reference_group_selection_algorithm"]
        == (confirmatory["data"]["reference_group_selection_algorithm"])
    )
    assert (
        primary["representations"][-1]["cache_provenance"]
        == json.loads(bundle.pathology.read_text(encoding="utf-8"))["primary_cache_provenance"]
    )
    assert len(result.primary_plan.cells) == 222
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _finalise(bundle)

    replaced = _finalise(
        bundle,
        replace_draft=True,
        expected_primary_output_sha256=sha256_file(bundle.primary_output),
        expected_confirmatory_output_sha256=sha256_file(bundle.confirmatory_output),
    )
    assert replaced.primary_config_sha256 == result.primary_config_sha256


def test_rejects_tampered_or_missing_sidecar_and_mismatched_order(tmp_path: Path) -> None:
    tampered = _bundle(tmp_path / "tampered")
    context_sidecar = tampered.caches.context.with_suffix(".npz.metadata.json")
    payload = json.loads(context_sidecar.read_text(encoding="utf-8"))
    payload["encoder_identifier"] = "tampered_encoder"
    _write_json(context_sidecar, payload)
    with pytest.raises(M7ConfigFinalizationError, match="frozen verification"):
        _finalise(tampered)

    missing = _bundle(tmp_path / "missing")
    missing.caches.highlighted.with_suffix(".npz.metadata.json").unlink()
    with pytest.raises(M7ConfigFinalizationError, match="frozen verification"):
        _finalise(missing)

    mismatched = _bundle(tmp_path / "mismatched", mismatch_order=True)
    with pytest.raises(M7ConfigFinalizationError, match="sample_order_sha256"):
        _finalise(mismatched)


def test_rejects_cache_set_outside_analysis_manifest_authority(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    primary = yaml.safe_load(bundle.primary_draft.read_text(encoding="utf-8"))
    primary["data"]["analysis_manifest_authority"]["analysis_eligible_sample_count"] += 1
    bundle.primary_draft.write_text(
        yaml.safe_dump(primary, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(M7ConfigFinalizationError, match="analysis_manifest_authority"):
        _finalise(bundle)
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()


def test_rejects_divergent_confirmatory_analysis_manifest_authority(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    confirmatory = yaml.safe_load(bundle.confirmatory_draft.read_text(encoding="utf-8"))
    confirmatory["data"]["analysis_manifest_authority"]["analysis_eligible_sample_count"] += 1
    bundle.confirmatory_draft.write_text(
        yaml.safe_dump(confirmatory, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        M7ConfigFinalizationError,
        match=r"primary/confirmatory data\.analysis_manifest_authority bindings differ",
    ):
        _finalise(bundle)
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()


def test_rejects_pathology_record_and_context_morph_lineage_mismatch(tmp_path: Path) -> None:
    pathology = _bundle(tmp_path / "pathology")
    report = json.loads(pathology.pathology.read_text(encoding="utf-8"))
    report["primary_cache_provenance"]["cache_recipe_sha256"] = "0" * 64
    _write_json(pathology.pathology, report)
    with pytest.raises(M7ConfigFinalizationError, match="exact producer policy"):
        _finalise(pathology)

    lineage = _bundle(tmp_path / "lineage", wrong_morph_context=True)
    with pytest.raises(M7ConfigFinalizationError, match="exact context embedding cache"):
        _finalise(lineage)


def test_rejects_pathology_reports_that_fail_the_freeze_semantic_gate(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = json.loads(bundle.pathology.read_text(encoding="utf-8"))

    for case in ("missing_selection_rule", "empty_records"):
        payload = json.loads(json.dumps(original))
        if case == "missing_selection_rule":
            payload.pop("selection_rule")
        else:
            payload["records"] = []
        _write_json(bundle.pathology, payload)
        with pytest.raises(M7ConfigFinalizationError, match="freeze semantic gate"):
            _finalise(bundle)
        assert not bundle.primary_output.exists(), case
        assert not bundle.confirmatory_output.exists(), case


def test_rejects_pilot_reports_that_fail_canonical_freeze_bindings(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = json.loads(bundle.pilot.read_text(encoding="utf-8"))

    for case in (
        "missing_producer_source",
        "wrong_class_order",
        "wrong_development_folds",
        "wrong_final_reference_policy",
        "wrong_source_artifact",
        "wrong_run_id",
        "wrong_artifact_root",
    ):
        payload = json.loads(json.dumps(original))
        if case == "missing_producer_source":
            payload.pop("producer_source_sha256")
        elif case == "wrong_class_order":
            payload["class_order"] = [4, 3, 2, 1, 0]
        elif case == "wrong_development_folds":
            payload["source_pilot"]["development_official_folds"] = [1]
        elif case == "wrong_final_reference_policy":
            payload["source_pilot"]["final_reference_policy"] = "final fold inspected"
        elif case == "wrong_source_artifact":
            payload["source_pilot"]["metrics_sha256"] = "0" * 64
        elif case == "wrong_run_id":
            payload["source_pilot"]["run_id"] = "self-asserted-other-run"
        else:
            payload["source_pilot"]["artifact_root_sha256"] = "0" * 64
        _write_json(bundle.pilot, payload)
        with pytest.raises(M7ConfigFinalizationError, match="pilot-derived"):
            _finalise(bundle)
        assert not bundle.primary_output.exists(), case
        assert not bundle.confirmatory_output.exists(), case


def test_independence_uses_audit_slice_and_ignores_final_fold_feature_values(
    tmp_path: Path,
) -> None:
    baseline = _bundle(tmp_path / "baseline")
    changed_final = _bundle(tmp_path / "changed", alter_final_context=True)

    assert sha256_file(baseline.independence) == sha256_file(changed_final.independence)
    assert sha256_file(baseline.caches.context) != sha256_file(changed_final.caches.context)
    _finalise(baseline)
    _finalise(changed_final)


def test_independence_rejects_audit_slice_tamper(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    report = json.loads(bundle.independence.read_text(encoding="utf-8"))
    entry = report["entries"]["imagenet_resnet18_context"]
    entry["auditor"]["feature_artifact_hash"] = "0" * 64
    _write_json(bundle.independence, report)

    with pytest.raises(M7ConfigFinalizationError, match="strict feature-independence matrix"):
        _finalise(bundle)


def test_independence_rejects_resigned_generator_semantic_tamper(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    report = json.loads(bundle.independence.read_text(encoding="utf-8"))
    for identifier, raw in report["entries"].items():
        generator = replace(
            FeatureSpaceEvidence(**raw["generator"]),
            implementation_hash="0" * 64,
        )
        evidence = FeatureIndependenceEvidence.create(
            matrix_version=raw["matrix_version"],
            matrix_decision=raw["matrix_decision"],
            matrix_reason=raw["matrix_reason"],
            generator=generator,
            auditor=FeatureSpaceEvidence(**raw["auditor"]),
        )
        report["entries"][identifier] = evidence.as_dict()
    _write_json(bundle.independence, report)

    with pytest.raises(M7ConfigFinalizationError, match="generator semantics"):
        _finalise(bundle)


def test_pair_publication_rolls_back_when_second_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original_commit = finalizer_module._commit_staged_output
    call_count = 0

    def fail_second_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected second commit failure")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(finalizer_module, "_commit_staged_output", fail_second_commit)
    with pytest.raises(OSError, match="injected second commit failure"):
        _finalise(bundle)

    assert call_count == 2
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()


def test_pair_publication_rolls_back_failure_after_second_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original_link = finalizer_module.os.link
    call_count = 0

    def link_then_fail(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        original_link(source, destination, *args, **kwargs)
        if call_count == 2:
            raise OSError("injected failure after second hard-link commit")

    monkeypatch.setattr(finalizer_module.os, "link", link_then_fail)
    with pytest.raises(OSError, match="after second hard-link commit"):
        _finalise(bundle)

    assert call_count == 2
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()


def test_partial_staging_is_removed_when_second_stage_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original_stage = finalizer_module._stage_output
    staged_paths: list[Path] = []
    call_count = 0

    def fail_second_stage(path: Path, content: str) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected second staging failure")
        staged = original_stage(path, content)
        staged_paths.append(staged)
        return staged

    monkeypatch.setattr(finalizer_module, "_stage_output", fail_second_stage)
    with pytest.raises(OSError, match="injected second staging failure"):
        _finalise(bundle)

    assert call_count == 2
    assert staged_paths and all(not path.exists() for path in staged_paths)
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()


def test_pair_publication_preserves_racing_unowned_second_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    original_commit = finalizer_module._commit_staged_output
    external_content = b"external concurrent writer\n"
    call_count = 0

    def race_second_output(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            bundle.confirmatory_output.write_bytes(external_content)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(finalizer_module, "_commit_staged_output", race_second_output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _finalise(bundle)

    assert call_count == 2
    assert not bundle.primary_output.exists()
    assert bundle.confirmatory_output.read_bytes() == external_content


def test_replace_pair_rollback_restores_exact_original_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    _finalise(bundle)
    original_primary = bundle.primary_output.read_bytes()
    original_confirmatory = bundle.confirmatory_output.read_bytes()
    original_primary_sha256 = sha256_file(bundle.primary_output)
    original_confirmatory_sha256 = sha256_file(bundle.confirmatory_output)

    pathology = json.loads(bundle.pathology.read_text(encoding="utf-8"))
    pathology["blocker"] = "changed blocker that produces different staged configs"
    _write_json(bundle.pathology, pathology)

    original_commit = finalizer_module._commit_staged_output
    call_count = 0

    def fail_second_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected second replacement failure")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(finalizer_module, "_commit_staged_output", fail_second_commit)
    with pytest.raises(OSError, match="injected second replacement failure"):
        _finalise(
            bundle,
            replace_draft=True,
            expected_primary_output_sha256=original_primary_sha256,
            expected_confirmatory_output_sha256=original_confirmatory_sha256,
        )

    assert call_count == 2
    assert bundle.primary_output.read_bytes() == original_primary
    assert bundle.confirmatory_output.read_bytes() == original_confirmatory


def test_cnn_source_drift_after_pair_commit_rolls_back_both_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    fake_cnn_source = tmp_path / "runtime_cnn_source.py"
    fake_cnn_source.write_text("SOURCE_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(finalizer_module.cnn_module, "__file__", str(fake_cnn_source))
    original_check = finalizer_module._assert_sources_fresh
    call_count = 0

    def mutate_before_second_check(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            fake_cnn_source.write_text("SOURCE_VERSION = 2\n", encoding="utf-8")
        original_check(*args, **kwargs)

    monkeypatch.setattr(
        finalizer_module,
        "_assert_sources_fresh",
        mutate_before_second_check,
    )
    with pytest.raises(M7ConfigFinalizationError, match="source changed"):
        _finalise(bundle)

    assert call_count == 2
    assert not bundle.primary_output.exists()
    assert not bundle.confirmatory_output.exists()
