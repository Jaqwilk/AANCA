"""Focused production-boundary tests for confirmatory CLI input derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from histo_audit.config import config_sha256, load_config
from histo_audit.corruption.controlled import apply_controlled_corruption
from histo_audit.experiment.confirmatory_cli_inputs import (
    ConfirmatoryCLIInputError,
    _candidate_cache_paths,
    _enforce_pathology_authority,
    _frozen_feature_specs,
    _materialize_observed_label_sets,
)
from histo_audit.experiment.reference_groups import (
    deterministic_group_greedy_class_distribution_v1,
)
from histo_audit.experiment.study_contracts import (
    CLASS_ORDER,
    validate_frozen_confirmatory_config,
)
from histo_audit.representations.cache_provenance import FrozenCacheVerification


def _frozen_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "confirmatory_frozen.yaml"
    return validate_frozen_confirmatory_config(load_config(path))


def _identity_crop(path: Path) -> tuple[np.ndarray[Any, Any], ...]:
    sample_ids: list[str] = []
    group_ids: list[str] = []
    folds: list[int] = []
    labels: list[int] = []
    for fold in (1, 2, 3):
        for group_index in range(20):
            group = f"fold-{fold}-group-{group_index:02d}"
            for label in CLASS_ORDER:
                sample_ids.append(f"{group}-class-{label}")
                group_ids.append(group)
                folds.append(fold)
                labels.append(label)
    arrays = (
        np.asarray(sample_ids, dtype=np.str_),
        np.asarray(group_ids, dtype=np.str_),
        np.asarray(folds, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )
    np.savez(
        path,
        sample_ids=arrays[0],
        group_ids=arrays[1],
        official_folds=arrays[2],
        pre_corruption_labels=arrays[3],
    )
    return arrays


def test_materializer_replays_only_audit_partition_and_keeps_protected_labels_clean(
    tmp_path: Path,
) -> None:
    crop = tmp_path / "crop.npz"
    sample_ids, group_ids, folds, labels = _identity_crop(crop)
    config = _frozen_config()
    semantic_sha = config_sha256(config)

    materialized = _materialize_observed_label_sets(
        crop,
        config=config,
        config_semantic_sha256=semantic_sha,
    )

    assert tuple(value.outer_fold for value in materialized) == (1, 2, 3)
    corruption = next(
        value for value in config["corruption"]["cells"] if float(value["rate"]) > 0.0
    )
    data = config["data"]
    for value in materialized:
        development_mask = folds != value.outer_fold
        final_mask = ~development_mask
        reference_groups = deterministic_group_greedy_class_distribution_v1(
            labels[development_mask],
            group_ids[development_mask],
            class_order=CLASS_ORDER,
            fraction=float(data["reference_validation_fraction_groups"]),
            seed=int(data["split_seed"]),
        )
        reference_mask = development_mask & np.isin(group_ids, reference_groups)
        audit_mask = development_mask & ~reference_mask
        replay = apply_controlled_corruption(
            labels[audit_mask],
            sample_ids=tuple(str(item) for item in sample_ids[audit_mask]),
            group_ids=tuple(str(item) for item in group_ids[audit_mask]),
            rate=float(corruption["rate"]),
            mechanism=str(corruption["mechanism"]),
            seed=int(corruption["seed"]),
            n_classes=len(CLASS_ORDER),
            transition_matrix=np.asarray(
                corruption["parameters"]["transition_matrix"], dtype=np.float64
            ),
        )

        assert value.configuration_sha256 == semantic_sha
        assert value.sample_ids == tuple(str(item) for item in sample_ids)
        assert np.array_equal(value.observed_labels[audit_mask], replay.observed_labels)
        assert np.array_equal(
            value.is_injected_corruption[audit_mask], replay.is_injected_corruption
        )
        assert np.array_equal(value.observed_labels[reference_mask], labels[reference_mask])
        assert np.array_equal(value.observed_labels[final_mask], labels[final_mask])
        assert not value.is_injected_corruption[reference_mask].any()
        assert not value.is_injected_corruption[final_mask].any()
        assert not value.pre_corruption_labels.flags.writeable
        assert not value.observed_labels.flags.writeable
        assert not value.is_injected_corruption.flags.writeable
        assert all(
            item == (corruption["mechanism"] if injected else "none")
            for item, injected in zip(
                value.corruption_types,
                value.is_injected_corruption.tolist(),
                strict=True,
            )
        )


def test_candidate_paths_use_exact_primary_bindings_and_canonical_morphometrics_name(
    tmp_path: Path,
) -> None:
    context = tmp_path / "pannuke_resnet18_context_rgb_embeddings.npz"
    highlighted = tmp_path / "pannuke_resnet18_target_highlighted_embeddings.npz"
    morphometrics = tmp_path / "pannuke_resnet18_context_plus_target_morphometrics.npz"
    unbound_alternative = tmp_path / "another_semantically_plausible_cache.npz"
    for path in (context, highlighted, morphometrics, unbound_alternative):
        path.write_bytes(b"fixture")
    bound = {
        "context_embedding_cache_path": context.resolve(),
        "highlighted_embedding_cache_path": highlighted.resolve(),
        "pathology_embedding_cache_path": None,
    }

    assert set(_candidate_cache_paths(bound)) == {
        context.resolve(),
        highlighted.resolve(),
        morphometrics.resolve(),
    }
    assert unbound_alternative.resolve() not in _candidate_cache_paths(bound)

    morphometrics.unlink()
    with pytest.raises(ConfirmatoryCLIInputError, match="canonical context-plus-morphometrics"):
        _candidate_cache_paths(bound)


def test_frozen_pathology_blocker_rejects_any_primary_bound_cache(tmp_path: Path) -> None:
    config = _frozen_config()
    _enforce_pathology_authority(
        config,
        bound_paths={"pathology_embedding_cache_path": None},
        expected_hashes={"pathology_embedding_cache_sha256": None},
        verified_hashes={"pathology_embedding_cache_sha256": None},
    )

    with pytest.raises(ConfirmatoryCLIInputError, match="frozen unavailable blocker"):
        _enforce_pathology_authority(
            config,
            bound_paths={"pathology_embedding_cache_path": tmp_path / "pathology.npz"},
            expected_hashes={"pathology_embedding_cache_sha256": "a" * 64},
            verified_hashes={"pathology_embedding_cache_sha256": "a" * 64},
        )


def test_frozen_feature_specs_require_exact_semantic_sidecar_match(
    tmp_path: Path,
) -> None:
    config = _frozen_config()
    records = {str(value["id"]): value for value in config["cache_provenance"]}
    frozen_scenarios = [
        value for value in config["scenarios"] if value["family"] == "imagenet_frozen"
    ]
    candidates: list[FrozenCacheVerification] = []
    for index, scenario in enumerate(frozen_scenarios, start=1):
        record = records[str(scenario["cache_provenance_id"])]
        cache_sha = f"{index:064x}"
        metadata = {
            "provenance_scope": "stage_eligible",
            "cache_file_sha256": cache_sha,
            "sidecar_semantic_sha256": record["sidecar_semantic_sha256"],
            "representation_id": record["representation_id"],
            "sample_order_sha256": record["sample_order_sha256"],
            "manifest_sha256": record["manifest_sha256"],
            "encoder_identifier": record["encoder_identifier"],
            "encoder_metadata_sha256": record["encoder_metadata_sha256"],
            "weight_identifier": record["weight_identifier"],
            "weights_sha256": record["weights_sha256"],
            "preprocessing_identifier": record["preprocessing_identifier"],
            "preprocessing_sha256": record["preprocessing_sha256"],
            "input_variant": record["input_variant"],
            "contract_input_variant": record["input_variant"],
        }
        cache_path = tmp_path / f"cache-{index}.npz"
        candidates.append(
            FrozenCacheVerification(
                cache_path=cache_path,
                sidecar_path=cache_path.with_suffix(".npz.metadata.json"),
                cache_file_sha256=cache_sha,
                sidecar_file_sha256=f"{index + 10:064x}",
                sidecar_semantic_sha256=str(record["sidecar_semantic_sha256"]),
                metadata=metadata,
            )
        )

    specs = _frozen_feature_specs(config, candidates)
    assert tuple(value.scenario_id for value in specs) == tuple(
        str(value["id"]) for value in frozen_scenarios
    )
    assert tuple(value.expected_cache_sha256 for value in specs) == tuple(
        value.cache_file_sha256 for value in candidates
    )

    duplicate = FrozenCacheVerification(
        cache_path=tmp_path / "duplicate.npz",
        sidecar_path=tmp_path / "duplicate.npz.metadata.json",
        cache_file_sha256="f" * 64,
        sidecar_file_sha256="e" * 64,
        sidecar_semantic_sha256=candidates[0].sidecar_semantic_sha256,
        metadata=dict(candidates[0].metadata),
    )
    with pytest.raises(ConfirmatoryCLIInputError, match="matches=2"):
        _frozen_feature_specs(config, [*candidates, duplicate])

    altered = dict(candidates[0].metadata)
    altered["preprocessing_identifier"] = "different_preprocessing"
    candidates[0] = FrozenCacheVerification(
        cache_path=candidates[0].cache_path,
        sidecar_path=candidates[0].sidecar_path,
        cache_file_sha256=candidates[0].cache_file_sha256,
        sidecar_file_sha256=candidates[0].sidecar_file_sha256,
        sidecar_semantic_sha256=candidates[0].sidecar_semantic_sha256,
        metadata=altered,
    )
    with pytest.raises(ConfirmatoryCLIInputError, match="matches=0"):
        _frozen_feature_specs(config, candidates)
