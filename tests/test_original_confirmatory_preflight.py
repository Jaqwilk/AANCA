from __future__ import annotations

import inspect
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import histo_audit.experiment.original_confirmatory_preflight as preflight_module
from histo_audit.config import load_config
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryExecutionControls,
    confirmatory_execution_controls_from_frozen_config,
)
from histo_audit.experiment.original_confirmatory_preflight import (
    ORIGINAL_CONFIRMATORY_CAPACITY_POLICY,
    ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS,
    OriginalConfirmatoryCacheBinding,
    OriginalConfirmatoryPreflightError,
    recheck_original_confirmatory_capacity,
    require_original_confirmatory_preflight,
    validate_original_confirmatory_preflight_pair,
    validate_original_confirmatory_preflight_receipt,
)
from histo_audit.experiment.original_confirmatory_resume import (
    ORIGINAL_CONFIRMATORY_WEIGHT_SHA256,
)
from histo_audit.experiment.resource_bounded_runner import (
    RESOURCE_BOUNDED_CAPACITY_POLICY,
    ResourceCapacityEvidence,
)
from histo_audit.utils.run_tracking import sha256_file


class _Usage:
    def __init__(self, free: int) -> None:
        self.free = free


@pytest.fixture
def original_controls() -> ConfirmatoryExecutionControls:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "confirmatory_frozen.yaml")
    return confirmatory_execution_controls_from_frozen_config(config)


@pytest.fixture
def synthetic_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path]:
    cache = tmp_path / "crops.npz"
    metadata = tmp_path / "crops.npz.metadata.json"
    weight = tmp_path / "resnet18-f37072fd.pth"
    cache.write_bytes(b"synthetic crop cache")
    metadata.write_bytes(b'{"synthetic":true}')
    weight.write_bytes(b"synthetic test-only weight carrier")
    bindings: tuple[OriginalConfirmatoryCacheBinding, ...] = (
        OriginalConfirmatoryCacheBinding(
            role="crop_cache",
            path=cache,
            expected_sha256=sha256_file(cache),
        ),
        OriginalConfirmatoryCacheBinding(
            role="crop_cache_metadata",
            path=metadata,
            expected_sha256=sha256_file(metadata),
        ),
    )
    for scenario_id in ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS:
        feature = tmp_path / f"{scenario_id}.npz"
        feature_metadata = tmp_path / f"{scenario_id}.npz.metadata.json"
        feature.write_bytes(f"synthetic feature cache: {scenario_id}".encode())
        feature_metadata.write_bytes(f'{{"synthetic_scenario":"{scenario_id}"}}'.encode())
        bindings = (
            *bindings,
            OriginalConfirmatoryCacheBinding(
                role=f"frozen_feature_cache:{scenario_id}",
                path=feature,
                expected_sha256=sha256_file(feature),
            ),
            OriginalConfirmatoryCacheBinding(
                role=f"frozen_feature_metadata:{scenario_id}",
                path=feature_metadata,
                expected_sha256=sha256_file(feature_metadata),
            ),
        )
    real_hasher = preflight_module._hash_runtime_file

    def test_hasher(path: Path, *, role: str) -> str:
        if Path(path) == weight:
            return ORIGINAL_CONFIRMATORY_WEIGHT_SHA256
        return real_hasher(path, role=role)

    monkeypatch.setattr(preflight_module, "_hash_runtime_file", test_hasher)
    return bindings, weight


def _successful_observation(weight: Path) -> dict[str, Any]:
    policy = ORIGINAL_CONFIRMATORY_CAPACITY_POLICY
    return {
        "total_host_ram_bytes": int(policy["minimum_total_ram_bytes"]),
        "available_host_ram_bytes": int(policy["minimum_available_ram_bytes_before_data"]),
        "cuda_available": True,
        "cuda_device_count": 1,
        "selected_cuda_device_index": 0,
        "selected_cuda_device_name": "Synthetic CUDA",
        "total_vram_bytes": int(policy["minimum_total_vram_bytes"]),
        "free_vram_bytes": int(policy["minimum_free_vram_bytes"]),
        "cudnn_available": True,
        "amp_available": True,
        "amp_dtype": "float16",
        "grad_scaler_available": True,
        "grad_scaler_enabled": True,
        "weight_identifier": policy["official_weight_identifier"],
        "weight_path": str(weight),
        "weight_present": True,
        "weight_sha256": policy["official_weight_sha256"],
        "smoke_attempted": True,
        "smoke_completed": True,
        "smoke_input_shapes": [list(value) for value in policy["cuda_smoke_input_shapes"]],
        "smoke_forward_finite": True,
        "smoke_backward_finite": True,
        "smoke_peak_allocated_bytes": 512,
        "smoke_error": None,
    }


def _weight_provider(weight: Path) -> tuple[Path, str]:
    return weight, ORIGINAL_CONFIRMATORY_WEIGHT_SHA256


def _rehash_receipt(receipt: Any) -> Any:
    without_hash = replace(receipt, receipt_without_self_hash_sha256="")
    payload = without_hash.as_dict()
    payload.pop("receipt_without_self_hash_sha256")
    return replace(
        without_hash,
        receipt_without_self_hash_sha256=canonical_sha256(payload),
    )


def test_initial_and_final_preflight_bind_exact_original_plan(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    phases: list[str] = []

    def probe(_: Path) -> dict[str, Any]:
        observation = _successful_observation(weight)
        if phases:
            observation["available_host_ram_bytes"] = int(
                ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_available_ram_bytes_before_tracker"]
            )
        phases.append("probed")
        return observation

    initial = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path / "runs",
        cache_bindings=bindings,
        compute_probe=probe,
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        clock=lambda: "2026-07-30T10:00:00+00:00",
        official_weight_binding=lambda: _weight_provider(weight),
    )
    final = recheck_original_confirmatory_capacity(
        initial,
        original_controls,
        target=tmp_path / "runs",
        cache_bindings=bindings,
        compute_probe=probe,
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        clock=lambda: "2026-07-30T10:01:00+00:00",
        official_weight_binding=lambda: _weight_provider(weight),
    )

    assert initial.phase == "before_authority_or_data_loading"
    assert final.phase == "guarded_immediately_before_tracker"
    assert final.parent_receipt_sha256 == initial.receipt_without_self_hash_sha256
    assert initial.planned_cell_count == 108
    assert initial.planned_required_cell_count == 90
    assert initial.planned_optional_cell_count == 18
    assert initial.planned_cnn_cell_count == 36
    assert initial.planned_cnn_fold_checkpoint_count == 180
    assert (
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["policy"]
        == "original_confirmatory_sealed_plan_capacity_v2"
    )
    assert ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["schema_version"] == 2
    assert (
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["checkpoint_publication_policy"]
        == "canonical_plus_distinct_versioned_o_excl_physical_copy_v1"
    )
    assert ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["checkpoint_physical_copy_count"] == 2
    assert (
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["projected_checkpoint_bytes_per_physical_copy"]
        == 30 * 1_073_741_824
    )
    assert initial.disk_evidence.projected_active_bytes == 60 * 1_073_741_824
    assert initial.disk_evidence.safety_margin_bytes == 10 * 1_073_741_824
    assert initial.disk_evidence.minimum_free_bytes == 70 * 1_073_741_824
    assert len(phases) == 2
    validate_original_confirmatory_preflight_receipt(initial, original_controls)
    validate_original_confirmatory_preflight_receipt(final, original_controls)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cuda_available", False, "CUDA/AMP/cache/RAM"),
        ("amp_available", False, "CUDA/AMP/cache/RAM"),
        ("grad_scaler_enabled", False, "CUDA/AMP/cache/RAM"),
        ("smoke_backward_finite", False, "CUDA/AMP/cache/RAM"),
        ("available_host_ram_bytes", 1, "CUDA/AMP/cache/RAM"),
        ("free_vram_bytes", 1, "CUDA/AMP/cache/RAM"),
    ],
)
def test_compute_contract_fails_closed(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
    field: str,
    value: object,
    message: str,
) -> None:
    bindings, weight = synthetic_files
    observation = _successful_observation(weight)
    observation[field] = value

    with pytest.raises(OriginalConfirmatoryPreflightError, match=message):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=bindings,
            compute_probe=lambda _: observation,
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_disk_capacity_requires_both_physical_checkpoint_copies_plus_ten_gib(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files

    with pytest.raises(
        OriginalConfirmatoryPreflightError,
        match="60-GiB canonical-plus-versioned checkpoint plan plus 10-GiB margin",
    ):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=bindings,
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"]) - 1
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_physical_copy_count", 1),
        ("projected_checkpoint_bytes_per_physical_copy", 29 * 1_073_741_824),
        ("projected_all_checkpoint_copies_bytes", 30 * 1_073_741_824),
        ("fixed_safety_margin_bytes", 9 * 1_073_741_824),
        ("minimum_free_bytes", 40 * 1_073_741_824),
    ],
)
def test_sealed_dual_copy_capacity_arithmetic_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
    field: str,
    value: int,
) -> None:
    bindings, weight = synthetic_files
    tampered = dict(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY)
    tampered[field] = value
    monkeypatch.setattr(
        preflight_module,
        "ORIGINAL_CONFIRMATORY_CAPACITY_POLICY",
        MappingProxyType(tampered),
    )

    with pytest.raises(
        OriginalConfirmatoryPreflightError,
        match="sealed capacity arithmetic is invalid",
    ):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=bindings,
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(70 * 1_073_741_824),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_cache_hash_mismatch_fails_before_compute(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    changed = replace(bindings[0], expected_sha256="0" * 64)
    compute_called = False

    def probe(_: Path) -> dict[str, Any]:
        nonlocal compute_called
        compute_called = True
        return _successful_observation(weight)

    with pytest.raises(OriginalConfirmatoryPreflightError, match="differs from its hash"):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=(changed, *bindings[1:]),
            compute_probe=probe,
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )
    assert compute_called is False


@pytest.mark.parametrize(
    "missing_role",
    [
        "crop_cache",
        "crop_cache_metadata",
        *[
            role
            for scenario in ORIGINAL_CONFIRMATORY_REQUIRED_FEATURE_CACHE_SCENARIOS
            for role in (
                f"frozen_feature_cache:{scenario}",
                f"frozen_feature_metadata:{scenario}",
            )
        ],
    ],
)
def test_every_required_cache_binding_is_mandatory(
    missing_role: str,
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    reduced = tuple(value for value in bindings if value.role != missing_role)

    with pytest.raises(
        OriginalConfirmatoryPreflightError,
        match="exact required",
    ):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=reduced,
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_extra_cache_role_is_rejected(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    extra = tmp_path / "extra.npz"
    extra.write_bytes(b"extra")

    with pytest.raises(OriginalConfirmatoryPreflightError, match="exact required"):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=(
                *bindings,
                OriginalConfirmatoryCacheBinding(
                    role="frozen_feature_cache:undeclared",
                    path=extra,
                    expected_sha256=sha256_file(extra),
                ),
            ),
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_wrong_sidecar_and_physical_alias_are_rejected(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    sidecar_index = next(
        index
        for index, value in enumerate(bindings)
        if value.role.startswith("frozen_feature_metadata:")
    )
    wrong_sidecar = tmp_path / "wrong-sidecar.json"
    wrong_sidecar.write_bytes(b"{}")
    changed = list(bindings)
    changed[sidecar_index] = replace(
        changed[sidecar_index],
        path=wrong_sidecar,
        expected_sha256=sha256_file(wrong_sidecar),
    )
    with pytest.raises(OriginalConfirmatoryPreflightError, match="noncanonical"):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=tuple(changed),
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )

    feature_index = next(
        index
        for index, value in enumerate(bindings)
        if value.role.startswith("frozen_feature_cache:")
    )
    bindings[feature_index].path.unlink()
    os.link(bindings[0].path, bindings[feature_index].path)
    aliased = list(bindings)
    aliased[feature_index] = replace(
        aliased[feature_index],
        expected_sha256=sha256_file(bindings[0].path),
    )
    with pytest.raises(
        OriginalConfirmatoryPreflightError,
        match=r"private physical|alias one physical",
    ):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=tuple(aliased),
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_official_weight_provider_cannot_change_frozen_hash(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files

    with pytest.raises(OriginalConfirmatoryPreflightError, match="frozen original authority"):
        require_original_confirmatory_preflight(
            original_controls,
            target=tmp_path,
            cache_bindings=bindings,
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: (weight, "0" * 64),
        )


def test_receipt_tamper_is_rejected(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    receipt = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path,
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        official_weight_binding=lambda: _weight_provider(weight),
    )

    with pytest.raises(OriginalConfirmatoryPreflightError, match="incomplete or altered"):
        validate_original_confirmatory_preflight_receipt(
            replace(receipt, planned_cnn_fold_checkpoint_count=179),
            original_controls,
        )
    with pytest.raises(OriginalConfirmatoryPreflightError, match="self-hash differs"):
        validate_original_confirmatory_preflight_receipt(
            replace(receipt, checked_at_utc="2026-07-30T10:02:00+00:00"),
            original_controls,
        )


@pytest.mark.parametrize(
    ("compute_field", "value"),
    [
        ("total_host_ram_bytes", 1),
        ("available_host_ram_bytes", 1),
        ("total_vram_bytes", 1),
        ("free_vram_bytes", 1),
        ("cuda_available", False),
        ("grad_scaler_enabled", False),
    ],
)
def test_self_consistent_forged_compute_pass_is_independently_rejected(
    compute_field: str,
    value: object,
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    receipt = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path,
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        official_weight_binding=lambda: _weight_provider(weight),
    )
    observation = dict(receipt.compute_evidence.observation)
    observation[compute_field] = value
    forged_compute = replace(
        receipt.compute_evidence,
        observation=observation,
        observation_sha256=canonical_sha256(observation),
        passed=True,
    )
    forged = _rehash_receipt(replace(receipt, compute_evidence=forged_compute))

    with pytest.raises(OriginalConfirmatoryPreflightError):
        validate_original_confirmatory_preflight_receipt(
            forged,
            original_controls,
        )


def test_self_consistent_forged_disk_pass_is_independently_rejected(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    receipt = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path,
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        official_weight_binding=lambda: _weight_provider(weight),
    )
    forged = _rehash_receipt(
        replace(
            receipt,
            disk_evidence=replace(
                receipt.disk_evidence,
                free_bytes=1,
                passed=True,
            ),
        )
    )

    with pytest.raises(OriginalConfirmatoryPreflightError):
        validate_original_confirmatory_preflight_receipt(
            forged,
            original_controls,
        )


def test_final_pair_rejects_changed_cache_volume_and_nonlater_timestamp(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    initial = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path / "runs",
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        clock=lambda: "2026-07-30T10:00:00+00:00",
        official_weight_binding=lambda: _weight_provider(weight),
    )
    final = recheck_original_confirmatory_capacity(
        initial,
        original_controls,
        target=tmp_path / "runs",
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        clock=lambda: "2026-07-30T10:01:00+00:00",
        official_weight_binding=lambda: _weight_provider(weight),
    )

    changed_cache = list(final.cache_evidence)
    changed_cache[0] = replace(
        changed_cache[0],
        size_bytes=changed_cache[0].size_bytes + 1,
    )
    changed_cache_tuple = tuple(changed_cache)
    forged_cache = _rehash_receipt(
        replace(
            final,
            cache_evidence=changed_cache_tuple,
            cache_manifest_sha256=canonical_sha256(
                [value.as_dict() for value in changed_cache_tuple]
            ),
        )
    )
    with pytest.raises(OriginalConfirmatoryPreflightError, match="does not exactly descend"):
        validate_original_confirmatory_preflight_pair(
            initial,
            forged_cache,
            original_controls,
        )

    other_volume = (tmp_path / "other-volume").resolve()
    other_volume.mkdir()
    forged_volume = _rehash_receipt(
        replace(
            final,
            disk_evidence=replace(
                final.disk_evidence,
                probe_path=other_volume,
            ),
        )
    )
    with pytest.raises(OriginalConfirmatoryPreflightError, match="does not exactly descend"):
        validate_original_confirmatory_preflight_pair(
            initial,
            forged_volume,
            original_controls,
        )

    forged_time = _rehash_receipt(replace(final, checked_at_utc=initial.checked_at_utc))
    with pytest.raises(OriginalConfirmatoryPreflightError, match="does not exactly descend"):
        validate_original_confirmatory_preflight_pair(
            initial,
            forged_time,
            original_controls,
        )


def test_final_recheck_rehashes_cache_and_fails_without_successor_fallback(
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
    synthetic_files: tuple[tuple[OriginalConfirmatoryCacheBinding, ...], Path],
) -> None:
    bindings, weight = synthetic_files
    receipt = require_original_confirmatory_preflight(
        original_controls,
        target=tmp_path,
        cache_bindings=bindings,
        compute_probe=lambda _: _successful_observation(weight),
        disk_usage=lambda _: _Usage(
            int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
        ),
        official_weight_binding=lambda: _weight_provider(weight),
    )
    bindings[0].path.write_bytes(b"changed after initial preflight")

    with pytest.raises(OriginalConfirmatoryPreflightError, match="differs from its hash"):
        recheck_original_confirmatory_capacity(
            receipt,
            original_controls,
            target=tmp_path,
            cache_bindings=bindings,
            compute_probe=lambda _: _successful_observation(weight),
            disk_usage=lambda _: _Usage(
                int(ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["minimum_free_bytes"])
            ),
            official_weight_binding=lambda: _weight_provider(weight),
        )


def test_production_runner_orders_preflight_before_authority_data_and_tracker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    original_controls: ConfirmatoryExecutionControls,
) -> None:
    from types import SimpleNamespace

    from histo_audit.experiment import confirmatory_runner as runner_module
    from histo_audit.experiment import original_confirmatory_runner_core as core_module

    events: list[str] = []

    def observed(name: str, value: Any) -> Any:
        events.append(name)
        return value

    def forbidden(name: str) -> Any:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            events.append(name)
            raise AssertionError(f"{name} ran after a failed initial preflight")

        return fail

    dependencies = SimpleNamespace(
        config_loader=lambda _path: observed("config", {}),
        plan_builder=lambda _config: observed("plan", original_controls.plan),
        controls_builder=lambda _config: observed("controls", original_controls),
        gate_validator=forbidden("gate"),
        input_builder=forbidden("data"),
        tracker_starter=forbidden("tracker"),
    )
    monkeypatch.setattr(
        runner_module,
        "ConfirmatoryRunnerDependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        runner_module,
        "_require_legacy_confirmatory_execution_profile",
        lambda _config, _plan: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_verify_cache_files",
        lambda *_args, **_kwargs: events.append("cache"),
    )
    monkeypatch.setattr(
        runner_module,
        "_original_confirmatory_cache_bindings",
        lambda *_args, **_kwargs: observed("cache_bindings", ()),
    )

    def stop_at_initial_preflight(*_args: Any, **_kwargs: Any) -> Any:
        events.append("initial_preflight")
        raise OriginalConfirmatoryPreflightError("synthetic preflight stop")

    monkeypatch.setattr(
        runner_module,
        "require_original_confirmatory_preflight",
        stop_at_initial_preflight,
    )
    with pytest.raises(
        runner_module.ConfirmatoryStudyRunnerError,
        match="preflight failed before lifecycle/authority consumption, data loading, "
        "or RunTracker creation",
    ):
        runner_module.execute_confirmatory_study(
            gate_evidence=None,  # type: ignore[arg-type]
            primary_run_directory=tmp_path / "primary",
            project_root=tmp_path,
            freeze_directory=tmp_path / "freeze",
            dataset_path=tmp_path / "dataset.parquet",
            manifest_path=tmp_path / "manifest.parquet",
            duplicate_audit_path=tmp_path / "duplicate-audit.npz",
            pathology_encoder_audit_path=tmp_path / "pathology-audit.json",
            frozen_primary_config_path=tmp_path / "primary.yaml",
            frozen_confirmatory_config_path=tmp_path / "confirmatory.yaml",
            crop_cache_path=tmp_path / "crop-cache",
            expected_crop_cache_sha256="a" * 64,
            expected_crop_metadata_sha256="b" * 64,
            expected_raw_inventory_sha256="c" * 64,
            frozen_feature_caches=(),
            runs_root=tmp_path / "runs",
            lifecycle_readiness_run_directory=tmp_path / "readiness",
        )

    assert events == [
        "config",
        "plan",
        "controls",
        "cache",
        "cache_bindings",
        "initial_preflight",
    ]

    public_source = inspect.getsource(runner_module.execute_confirmatory_study)
    capsule_source = inspect.getsource(
        runner_module._execute_original_confirmatory_capsule_lifecycle
    )
    sealed_entry_source = inspect.getsource(core_module._run_original_confirmatory_capsule_request)
    source = inspect.getsource(runner_module._execute_confirmatory_study_lifecycle)

    initial = source.index("require_original_confirmatory_preflight(")
    lifecycle = source.index("require_current_lifecycle_readiness(")
    strict_lifecycle = source.index("require_current_original_confirmatory_lifecycle_readiness(")
    strict_pin_check = source.index("_require_exact_published_t0_lifecycle_pins(")
    public_gate = source.index("live_gate = deps.gate_validator(")
    data_loading = source.index("prepared = deps.input_builder(")
    final_recheck = source.index("recheck_original_confirmatory_capacity(")
    guarded_start = source.index("final_gate, tracker = _guarded_final_gate_and_start(")
    matrix_execution = source.index("artifacts = _execute_original_confirmatory_prepared_matrix(")

    assert initial < lifecycle < public_gate < data_loading
    assert initial < strict_lifecycle < strict_pin_check < public_gate
    assert "if capsule_request is None:" in source
    assert "is not OriginalConfirmatoryPublishedT0LifecycleReadinessVerification" in source
    assert "technical_authority_directory=(" in source
    assert "expected_technical_authority_artifact_root_sha256=" not in source
    assert "expected_technical_authorization_sha256=" not in source
    assert data_loading < final_recheck < guarded_start < matrix_execution
    assert "_execute_confirmatory_study_lifecycle(" in public_source
    assert "capsule_request=None" in public_source
    assert "_execute_confirmatory_study_lifecycle(" in capsule_source
    assert "capsule_request=request" in capsule_source
    assert "_execute_original_confirmatory_capsule_lifecycle(request)" in sealed_entry_source
    assert "cache_recheck=final_preflight_recheck" in source
    assert "ORIGINAL_CONFIRMATORY_INITIAL_PREFLIGHT_FILENAME" in source
    assert "ORIGINAL_CONFIRMATORY_FINAL_PREFLIGHT_FILENAME" in source


def test_original_preflight_is_distinct_and_has_no_outcome_or_predecessor_surface() -> None:
    signature = inspect.signature(require_original_confirmatory_preflight)

    assert (
        ORIGINAL_CONFIRMATORY_CAPACITY_POLICY["policy"]
        != RESOURCE_BOUNDED_CAPACITY_POLICY["policy"]
    )
    assert "ResourceCapacityEvidence" not in str(signature.return_annotation)
    assert ResourceCapacityEvidence.__name__ not in str(signature.return_annotation)
    assert not {
        "outcomes",
        "metrics",
        "rankings",
        "predecessor",
        "retry_of_run_id",
        "authority",
    }.intersection(signature.parameters)
