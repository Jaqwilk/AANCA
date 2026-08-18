from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from histo_audit.corruption.controlled import (
    FeatureIndependenceEvidence,
    FeatureSpaceEvidence,
    apply_controlled_corruption,
    apply_corruption_to_records,
    canonical_sha256,
    exact_corruption_count,
    semantic_sha256,
)


def _independence_evidence(
    synthetic_dataset,
    generator_features: np.ndarray | None = None,
    *,
    same_space: bool = False,
    reason: str = "Frozen test matrix verifies distinct feature implementations.",
) -> FeatureIndependenceEvidence:
    generator_features = (
        synthetic_dataset.corruption_features if generator_features is None else generator_features
    )
    fitted_hash = canonical_sha256({"sample_ids": synthetic_dataset.sample_ids.tolist()})
    generator = FeatureSpaceEvidence.from_array(
        generator_features,
        representation_name="morphology" if not same_space else "same_space",
        family="morphology" if not same_space else "shared",
        implementation_hash=semantic_sha256("generator_impl" if not same_space else "shared_impl"),
        weights_hash=semantic_sha256("generator_weights" if not same_space else "shared_weights"),
        preprocessing_hash=semantic_sha256(
            "generator_preprocessing" if not same_space else "shared_preprocessing"
        ),
        fitted_data_hash=fitted_hash,
    )
    auditor = (
        generator
        if same_space
        else FeatureSpaceEvidence.from_array(
            synthetic_dataset.audit_features,
            representation_name="colour",
            family="colour_only",
            implementation_hash=semantic_sha256("auditor_impl"),
            weights_hash=semantic_sha256("auditor_weights"),
            preprocessing_hash=semantic_sha256("rgb_statistics_without_morphology"),
            fitted_data_hash=fitted_hash,
        )
    )
    return FeatureIndependenceEvidence.create(
        matrix_version="test_matrix_v1",
        matrix_decision="verified_independent",
        matrix_reason=reason,
        generator=generator,
        auditor=auditor,
    )


@pytest.mark.parametrize("rate", [0, 5, 10, 20, 0.05, 0.10, 0.20])
def test_symmetric_corruption_is_exact_deterministic_and_never_self_replaces(
    synthetic_dataset, rate
) -> None:
    labels = synthetic_dataset.pre_corruption_labels
    kwargs = {
        "sample_ids": synthetic_dataset.sample_ids,
        "group_ids": synthetic_dataset.group_ids,
        "rate": rate,
        "mechanism": "symmetric_random_corruption",
        "seed": 71,
        "n_classes": 5,
    }
    first = apply_controlled_corruption(labels, **kwargs)
    second = apply_controlled_corruption(labels, **kwargs)
    assert first.exact_count == exact_corruption_count(len(labels), rate)
    assert int(first.is_injected_corruption.sum()) == first.exact_count
    np.testing.assert_array_equal(first.observed_labels, second.observed_labels)
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_array_equal(first.pre_corruption_labels, labels)
    changed = first.observed_labels != labels
    np.testing.assert_array_equal(changed, first.is_injected_corruption)
    assert np.all(first.observed_labels[changed] != labels[changed])


def test_zero_percent_is_a_true_no_op(synthetic_dataset) -> None:
    result = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=0,
        seed=1,
    )
    assert result.exact_count == 0
    assert result.selected_indices.size == 0
    assert not result.is_injected_corruption.any()
    np.testing.assert_array_equal(result.observed_labels, result.pre_corruption_labels)


def test_confusion_target_matrix_controls_replacements(synthetic_dataset) -> None:
    transition = np.zeros((5, 5), dtype=float)
    for source in range(5):
        transition[source, (source + 1) % 5] = 1.0
    result = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=20,
        mechanism="confusion_targeted_corruption",
        transition_matrix=transition,
        seed=3,
    )
    selected = result.selected_indices
    np.testing.assert_array_equal(
        result.observed_labels[selected],
        (result.pre_corruption_labels[selected] + 1) % 5,
    )


def test_group_conditional_still_has_exact_global_count(synthetic_dataset) -> None:
    unique_groups = sorted(set(synthetic_dataset.group_ids.tolist()))
    weights = {group: (8.0 if index < 3 else 0.2) for index, group in enumerate(unique_groups)}
    result = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=10,
        mechanism="group_conditional_corruption",
        group_weights=weights,
        seed=12,
    )
    assert result.exact_count == exact_corruption_count(len(synthetic_dataset.records), 10)
    assert np.all(
        result.pre_corruption_labels[result.selected_indices]
        != result.observed_labels[result.selected_indices]
    )


def test_instance_corruption_records_feature_independence_and_circularity(
    synthetic_dataset,
) -> None:
    unverified = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=10,
        mechanism="instance_dependent_corruption",
        generator_features=synthetic_dataset.corruption_features,
        generator_representation="morphology",
        auditor_representation="colour",
        seed=9,
    )
    assert unverified.feature_space_independent is None
    assert unverified.independence_status == "unverified"
    assert unverified.circularity_risk is True

    independent = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=10,
        mechanism="instance_dependent_corruption",
        generator_features=synthetic_dataset.corruption_features,
        generator_representation="morphology",
        auditor_representation="colour",
        independence_evidence=_independence_evidence(synthetic_dataset),
        seed=9,
    )
    assert independent.feature_space_independent is True
    assert independent.circularity_risk is False
    circular = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=10,
        mechanism="instance_dependent_corruption",
        generator_features=synthetic_dataset.corruption_features,
        generator_representation="same_space",
        auditor_representation="same_space",
        independence_evidence=_independence_evidence(
            synthetic_dataset,
            synthetic_dataset.corruption_features,
            same_space=True,
        ),
        seed=9,
    )
    assert circular.feature_space_independent is False
    assert circular.circularity_risk is True


def test_tampered_independence_matrix_hash_is_rejected(synthetic_dataset) -> None:
    evidence = _independence_evidence(synthetic_dataset)
    tampered = replace(evidence, independence_matrix_hash="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        apply_controlled_corruption(
            synthetic_dataset.pre_corruption_labels,
            sample_ids=synthetic_dataset.sample_ids,
            group_ids=synthetic_dataset.group_ids,
            rate=10,
            mechanism="instance_dependent_corruption",
            generator_features=synthetic_dataset.corruption_features,
            generator_representation="morphology",
            auditor_representation="colour",
            independence_evidence=tampered,
            seed=9,
        )


def test_corrupted_manifest_preserves_source_records(synthetic_dataset) -> None:
    source_records = synthetic_dataset.records
    result = apply_controlled_corruption(
        synthetic_dataset.pre_corruption_labels,
        sample_ids=synthetic_dataset.sample_ids,
        group_ids=synthetic_dataset.group_ids,
        rate=10,
        seed=5,
    )
    updated = apply_corruption_to_records(source_records, result)
    assert all(not row.is_injected_corruption for row in source_records)
    for old, new, injected in zip(
        source_records, updated, result.is_injected_corruption, strict=True
    ):
        assert old.pre_corruption_label == new.pre_corruption_label
        assert new.is_injected_corruption is bool(injected)
        if injected:
            assert new.replacement_class == new.observed_label != new.pre_corruption_label
        else:
            assert new.observed_label == new.pre_corruption_label


def test_configuration_hash_binds_all_identity_and_seed_inputs(synthetic_dataset) -> None:
    labels = synthetic_dataset.pre_corruption_labels
    sample_ids = synthetic_dataset.sample_ids.tolist()
    group_ids = synthetic_dataset.group_ids.tolist()

    def run(**overrides):
        arguments = {
            "pre_corruption_labels": labels,
            "sample_ids": sample_ids,
            "group_ids": group_ids,
            "rate": 10,
            "seed": 71,
            "dataset_seed": 72,
            "upstream_manifest_hash": semantic_sha256("manifest_a"),
        }
        arguments.update(overrides)
        return apply_controlled_corruption(**arguments)

    baseline = run()
    changed_labels = labels.copy()
    changed_labels[0] = (changed_labels[0] + 1) % 5
    changed_samples = sample_ids.copy()
    changed_samples[0] += "_changed"
    changed_groups = group_ids.copy()
    changed_groups[0] += "_changed"
    variants = (
        run(pre_corruption_labels=changed_labels),
        run(sample_ids=changed_samples),
        run(group_ids=changed_groups),
        run(upstream_manifest_hash=semantic_sha256("manifest_b")),
        run(rate=20),
        run(seed=73),
        run(dataset_seed=74),
    )
    hashes = {baseline.configuration_hash, *(variant.configuration_hash for variant in variants)}
    assert len(hashes) == 1 + len(variants)
    payload = json.loads(baseline.configuration_payload_json)
    assert payload["rounding_policy"] == "round_half_up_floor_n_times_rate_plus_0_5"
    assert payload["inputs"]["sample_ids_sha256"]
    assert payload["inputs"]["group_ids_sha256"]


def test_configuration_hash_binds_transition_and_group_weight_semantics(
    synthetic_dataset,
) -> None:
    common = {
        "pre_corruption_labels": synthetic_dataset.pre_corruption_labels,
        "sample_ids": synthetic_dataset.sample_ids.tolist(),
        "group_ids": synthetic_dataset.group_ids.tolist(),
        "rate": 10,
        "seed": 81,
    }
    forward = np.zeros((5, 5), dtype=float)
    reverse = np.zeros((5, 5), dtype=float)
    for source in range(5):
        forward[source, (source + 1) % 5] = 1.0
        reverse[source, (source - 1) % 5] = 1.0
    targeted_a = apply_controlled_corruption(
        **common,
        mechanism="confusion_targeted_corruption",
        transition_matrix=forward,
    )
    targeted_b = apply_controlled_corruption(
        **common,
        mechanism="confusion_targeted_corruption",
        transition_matrix=reverse,
    )
    groups = sorted(set(synthetic_dataset.group_ids.tolist()))
    weights_a = {group: 1.0 for group in groups}
    weights_b = {group: (2.0 if index == 0 else 1.0) for index, group in enumerate(groups)}
    grouped_a = apply_controlled_corruption(
        **common,
        mechanism="group_conditional_corruption",
        group_weights=weights_a,
    )
    grouped_b = apply_controlled_corruption(
        **common,
        mechanism="group_conditional_corruption",
        group_weights=weights_b,
    )
    assert targeted_a.configuration_hash != targeted_b.configuration_hash
    assert grouped_a.configuration_hash != grouped_b.configuration_hash


def test_configuration_hash_binds_feature_dtype_shape_content_and_evidence(
    synthetic_dataset,
) -> None:
    base = synthetic_dataset.corruption_features
    variants = [
        base,
        base.astype(np.float32),
        np.column_stack([base, np.zeros(len(base))]),
        base.copy(),
    ]
    variants[-1][0, 0] += 0.001
    results = []
    for index, features in enumerate(variants):
        results.append(
            apply_controlled_corruption(
                synthetic_dataset.pre_corruption_labels,
                sample_ids=synthetic_dataset.sample_ids.tolist(),
                group_ids=synthetic_dataset.group_ids.tolist(),
                rate=10,
                mechanism="instance_dependent_corruption",
                generator_features=features,
                generator_representation="morphology",
                auditor_representation="colour",
                independence_evidence=_independence_evidence(
                    synthetic_dataset,
                    features,
                    reason=f"Frozen evidence variant {index}.",
                ),
                seed=91,
            )
        )
    assert len({result.configuration_hash for result in results}) == len(results)
