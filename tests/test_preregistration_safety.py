"""Adversarial safety coverage for the one-shot M7 freeze transaction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_workflow_stage_apis import _completed_pilot, _project_inputs

import histo_audit.workflows.preregistration as preregistration_module
from histo_audit.utils.run_tracking import (
    sha256_file,
    sha256_path,
    windows_compatible_relative_path_sort_key,
)
from histo_audit.workflows.preregistration import (
    freeze_preregistration,
    verify_preregistration_freeze,
)

_TIMESTAMP = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)


def _semantic_gate_stubs(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    independence = root / "reports" / "representation_independence.json"
    pilot_derived = root / "reports" / "pilot_derived_primary_parameters.json"
    pilot_derived.write_text("{}\n", encoding="utf-8")

    def validate_pilot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        attestation = {
            "schema_version": 1,
            "status": "passed",
            "scientific_stage_eligible": True,
        }
        return {
            "run_id": "semantic-pilot-fixture",
            "status": "completed",
            "artifact_root_sha256": "a" * 64,
            "post_seal_verification": attestation,
            "post_seal_verification_semantic_sha256": "b" * 64,
        }

    def validate_bound(*args: Any, **kwargs: Any) -> Path:
        del args, kwargs
        return independence

    def validate_pilot_derived(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        del args, kwargs
        return pilot_derived, {
            "schema_version": 1,
            "producer_id": "pilot_derived_primary_parameters_v1",
            "producer_source_sha256": "c" * 64,
            "canonical_path": "reports/pilot_derived_primary_parameters.json",
            "sha256": sha256_file(pilot_derived),
            "source_pilot_run_id": "semantic-pilot-fixture",
            "source_pilot_artifact_root_sha256": "a" * 64,
            "strict_semantic_validation_passed": True,
        }

    monkeypatch.setattr(preregistration_module, "_validate_completed_pilot", validate_pilot)
    monkeypatch.setattr(
        preregistration_module,
        "_validate_bound_evidence_hashes",
        validate_bound,
    )
    monkeypatch.setattr(
        preregistration_module,
        "_validate_pilot_derived_parameters",
        validate_pilot_derived,
    )
    monkeypatch.setattr(
        preregistration_module,
        "_validate_primary_config",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        preregistration_module,
        "_validate_confirmatory_config",
        lambda value: dict(value),
    )


def _freeze_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    root, dataset, manifest, raw_checksums, duplicate_audit, pathology = _project_inputs(tmp_path)
    raw_checksums.write_text(
        json.dumps(
            {
                "fold.npy": sha256_file(dataset / "fold.npy"),
                "nuclei.csv": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    pilot = _completed_pilot(root, dataset, manifest, duplicate_audit)
    (root / "reports" / "pilot-development.parquet").write_bytes(b"external-development-view")
    (root / "reports" / "pilot-gate.json").write_text("{}\n", encoding="utf-8")
    _semantic_gate_stubs(monkeypatch, root)
    return root, dataset, manifest, raw_checksums, duplicate_audit, pathology, pilot


def _freeze(
    root: Path,
    dataset: Path,
    manifest: Path,
    raw_checksums: Path,
    duplicate_audit: Path,
    pathology: Path,
    pilot: Path,
    **kwargs: Any,
) -> Any:
    return freeze_preregistration(
        project_root=root,
        pilot_run_directory=pilot,
        dataset_path=dataset,
        manifest_path=manifest,
        raw_checksum_manifest_path=raw_checksums,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology,
        pilot_development_manifest_path=root / "reports" / "pilot-development.parquet",
        pilot_gate_certificate_path=root / "reports" / "pilot-gate.json",
        timestamp=_TIMESTAMP,
        **kwargs,
    )


def _pilot_derived_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = root / "reports" / "pilot_derived_primary_parameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = "canonical-m6-pilot"
    artifact_root = "d" * 64
    transition_matrix = [[0.0, 1.0], [1.0, 0.0]]
    group_parameters = {
        "grouping_field": "tissue_type",
        "weights_by_value": {"tissue_a": 1.0, "tissue_b": 0.5},
        "default_weight": 0.75,
    }
    producer_source = (
        Path(preregistration_module.__file__).resolve().parents[1]
        / "experiment"
        / "pilot_derived_parameters.py"
    )
    payload = {
        "schema_version": 1,
        "producer_id": "pilot_derived_primary_parameters_v1",
        "producer_source_sha256": sha256_file(producer_source),
        "class_order": [0, 1],
        "confusion_targeted_corruption": {"transition_matrix": transition_matrix},
        "group_conditional_corruption": group_parameters,
        "source_pilot": {
            "run_id": run_id,
            "artifact_root_sha256": artifact_root,
            "development_official_folds": [1, 2],
            "final_reference_policy": (
                "no final-fold sample identifier, label, representation, or outcome read"
            ),
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    primary = {
        "pilot_derived_parameters": {
            "schema_version": 1,
            "producer_id": "pilot_derived_primary_parameters_v1",
            "path": "reports/pilot_derived_primary_parameters.json",
            "sha256": sha256_file(path),
            "source_pilot_run_id": run_id,
            "source_pilot_artifact_root_sha256": artifact_root,
        },
        "data": {
            "class_order": [0, 1],
            "development_official_folds": [1, 2],
        },
        "corruption": {
            "mechanisms": {
                "confusion_targeted_corruption": {
                    "transition_matrix": transition_matrix,
                },
                "group_conditional_corruption": group_parameters,
            }
        },
    }
    pilot_evidence = {"run_id": run_id, "artifact_root_sha256": artifact_root}
    return primary, pilot_evidence, path


def test_pilot_derived_parameters_bind_the_canonical_m6_run(tmp_path: Path) -> None:
    root = tmp_path / "project"
    primary, pilot_evidence, path = _pilot_derived_inputs(root)

    resolved, evidence = preregistration_module._validate_pilot_derived_parameters(
        primary,
        project_root=root,
        pilot_evidence=pilot_evidence,
    )

    assert resolved == path.resolve()
    assert evidence["strict_semantic_validation_passed"] is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_pilot"]["run_id"] = "different-sealed-pilot"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    primary["pilot_derived_parameters"]["sha256"] = sha256_file(path)
    primary["pilot_derived_parameters"]["source_pilot_run_id"] = "different-sealed-pilot"

    with pytest.raises(ValueError, match="canonical M6 pilot run_id"):
        preregistration_module._validate_pilot_derived_parameters(
            primary,
            project_root=root,
            pilot_evidence=pilot_evidence,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_manifest_sha256", "0" * 64),
        ("analysis_eligible_sample_order_sha256", "0" * 64),
        ("analysis_eligible_sample_count", 101),
    ],
)
def test_freeze_rejects_confirmatory_analysis_authority_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str | int,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    confirmatory_path = root / "configs" / "confirmatory.yaml"
    confirmatory = yaml.safe_load(confirmatory_path.read_text(encoding="utf-8"))
    confirmatory["data"]["analysis_manifest_authority"][field] = value
    confirmatory_path.write_text(
        yaml.safe_dump(confirmatory, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="analysis_manifest_authority"):
        _freeze(
            root,
            dataset,
            manifest,
            checksums,
            duplicate,
            pathology,
            pilot,
        )

    assert not (root / "artifacts" / "preregistration").exists()
    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()


def test_freeze_rejects_destination_under_raw_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="inside raw data"):
        _freeze(
            root,
            dataset,
            manifest,
            checksums,
            duplicate,
            pathology,
            pilot,
            freeze_root=dataset / "forbidden-freeze",
        )


def test_freeze_publishes_verified_marker_last_bundle_with_distinct_config_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )

    result = _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    verification = verify_preregistration_freeze(
        result.freeze_directory,
        frozen_primary_config_path=result.frozen_primary_config_path,
        frozen_confirmatory_config_path=result.frozen_confirmatory_config_path,
    )
    assert verification.valid
    assert result.pilot_post_seal_verification_path.is_file()
    assert result.representation_independence_snapshot_path.is_file()
    assert result.pilot_derived_parameters_snapshot_path.is_file()
    assert (result.freeze_directory / "governance_tree_manifest.json").is_file()
    evidence = json.loads((result.freeze_directory / "freeze_evidence.json").read_text("utf-8"))
    assert evidence["raw_dataset_checksum_manifest"]["exact_dataset_reconciliation"] is True
    assert evidence["source_tree_root_sha256"]
    assert evidence["governance_tree_root_sha256"]
    assert evidence["execution_source_tree"]["scope_kind"] == "execution_source"
    assert evidence["governance_tree"]["scope_kind"] == "governance_snapshot"
    assert evidence["representation_independence"]["strict_semantic_validation_passed"] is True
    assert evidence["pilot_derived_parameters"]["strict_semantic_validation_passed"] is True
    assert not result.frozen_primary_config_path.samefile(result.timestamped_primary_config_path)
    assert not result.frozen_confirmatory_config_path.samefile(
        result.timestamped_confirmatory_config_path
    )


def test_freeze_rejects_destination_under_sealed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="inside a sealed experiment run"):
        _freeze(
            root,
            dataset,
            manifest,
            checksums,
            duplicate,
            pathology,
            pilot,
            freeze_root=pilot / "forbidden-freeze",
        )


def test_freeze_requires_exact_raw_inventory_without_additions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    (dataset / "unexpected.bin").write_bytes(b"not declared by the raw checksum manifest")

    with pytest.raises(ValueError, match=r"exactly reconcile.*added=.*unexpected\.bin"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert not (root / "configs" / "primary_frozen.yaml").exists()


def test_freeze_dataset_digest_matches_run_tracker_without_mutating_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    dataset = root / "data" / "raw" / "mixed-case-tree"
    payloads = {
        "Fold 1/README.md": b"fold readme\n",
        "Fold 1/images/fold1/images.npy": b"image payload\n",
        "Fold 1/masks/README.md": b"mask readme\n",
        "a-.txt": b"flat payload\n",
        "a/file.txt": b"nested payload\n",
        "fold_1.zip": b"archive payload\n",
    }
    for relative, payload in payloads.items():
        destination = dataset / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    raw_checksums = root / "data" / "manifests" / "raw_files_sha256.json"
    raw_checksums.parent.mkdir(parents=True, exist_ok=True)
    raw_checksums.write_text(
        json.dumps(
            {relative: sha256_file(dataset / relative) for relative in reversed(tuple(payloads))}
        ),
        encoding="utf-8",
    )
    before = {
        relative: (
            (dataset / relative).read_bytes(),
            (dataset / relative).stat().st_mtime_ns,
        )
        for relative in payloads
    }

    expected = preregistration_module._parse_raw_checksum_manifest(
        raw_checksums,
        dataset=dataset,
        project_root=root,
    )
    snapshot = preregistration_module._capture_and_reconcile_dataset(
        dataset,
        expected=expected,
        evidence_paths=(raw_checksums,),
    )

    inventory_order = tuple(
        str(record["relative_path"]) for record in snapshot.raw_inventory_records
    )
    digest_order = tuple(sorted(payloads, key=windows_compatible_relative_path_sort_key))
    expected_inventory = [
        {
            "relative_path": relative,
            "size_bytes": len(payloads[relative]),
            "sha256": hashlib.sha256(payloads[relative]).hexdigest(),
        }
        for relative in sorted(payloads)
    ]
    assert inventory_order == tuple(sorted(payloads))
    assert digest_order != inventory_order
    assert list(snapshot.raw_inventory_records) == expected_inventory
    assert (
        snapshot.raw_inventory_sha256
        == hashlib.sha256(
            json.dumps(
                expected_inventory,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        snapshot.dataset_sha256
        == "bef34430bc7dfc7b5847001e535097cbed967934faff6b12059cc56cd85e65fc"
    )
    assert snapshot.dataset_sha256 == sha256_path(dataset)
    assert {
        relative: (
            (dataset / relative).read_bytes(),
            (dataset / relative).stat().st_mtime_ns,
        )
        for relative in payloads
    } == before


def test_freeze_rejects_unsafe_raw_inventory_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    checksums.write_text(
        json.dumps({"../fold.npy": sha256_file(dataset / "fold.npy")}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsafe"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)


def test_freeze_rolls_back_all_outputs_after_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    original = preregistration_module.publish_file_no_overwrite
    calls = 0

    def fail_third_publish(staged: str | Path, destination: str | Path) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected partial-publication failure")
        return original(staged, destination)

    monkeypatch.setattr(
        preregistration_module,
        "publish_file_no_overwrite",
        fail_third_publish,
    )

    with pytest.raises(OSError, match="injected partial-publication failure"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()
    assert not (root / "artifacts" / "preregistrations" / "20260718T150000.000000Z").exists()


def test_freeze_detects_source_toctou_before_success_marker_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    original = preregistration_module.publish_file_no_overwrite
    changed = False

    def mutate_raw_during_publish(staged: str | Path, destination: str | Path) -> Any:
        nonlocal changed
        published = original(staged, destination)
        if not changed:
            changed = True
            (dataset / "fold.npy").write_bytes(b"changed after the final pre-publish rehash")
        return published

    monkeypatch.setattr(
        preregistration_module,
        "publish_file_no_overwrite",
        mutate_raw_during_publish,
    )

    with pytest.raises((RuntimeError, ValueError), match=r"changed|reconcile"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()
    assert not (root / "artifacts" / "preregistrations" / "20260718T150000.000000Z").exists()


def test_freeze_git_state_comparison_ignores_only_observation_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    calls = 0
    substantive_state = {
        "available": True,
        "commit": "a" * 40,
        "branch": "codex/freeze",
        "dirty": True,
        "status_porcelain": "?? evidence.txt",
    }

    def capture_git_state(project_root: str | Path | None = None) -> dict[str, Any]:
        nonlocal calls
        assert Path(project_root or root).resolve() == root.resolve()
        calls += 1
        return {
            **substantive_state,
            "captured_at_utc": f"2026-07-18T15:00:0{calls}.000Z",
        }

    monkeypatch.setattr(preregistration_module, "capture_git_state", capture_git_state)

    result = _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert calls == 2
    assert verify_preregistration_freeze(
        result.freeze_directory,
        frozen_primary_config_path=result.frozen_primary_config_path,
        frozen_confirmatory_config_path=result.frozen_confirmatory_config_path,
    ).valid
    saved = json.loads((result.freeze_directory / "git_state.json").read_text(encoding="utf-8"))
    assert saved == {
        **substantive_state,
        "captured_at_utc": "2026-07-18T15:00:01.000Z",
    }


@pytest.mark.parametrize(
    ("initial_state", "changed_state"),
    [
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            {"available": False, "reason": "project root is not a Git work tree"},
            id="availability",
        ),
        pytest.param(
            {"available": False, "reason": "project root is not a Git work tree"},
            {"available": False, "reason": "git unavailable: injected"},
            id="reason",
        ),
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            {
                "available": True,
                "commit": "b" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            id="commit",
        ),
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "other",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            id="branch",
        ),
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": False,
                "status_porcelain": "",
            },
            id="dirty",
        ),
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? evidence.txt",
            },
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": True,
                "status_porcelain": "?? changed.txt",
            },
            id="status-porcelain",
        ),
    ],
)
def test_freeze_rejects_substantive_git_state_change_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: dict[str, Any],
    changed_state: dict[str, Any],
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    calls = 0

    def capture_git_state(project_root: str | Path | None = None) -> dict[str, Any]:
        nonlocal calls
        assert Path(project_root or root).resolve() == root.resolve()
        calls += 1
        state = initial_state if calls == 1 else changed_state
        return {
            **state,
            "captured_at_utc": f"2026-07-18T15:00:0{calls}.000Z",
        }

    monkeypatch.setattr(preregistration_module, "capture_git_state", capture_git_state)

    with pytest.raises(RuntimeError, match="Git state changed before preregistration publication"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert calls == 2
    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()
    assert not (root / "artifacts" / "preregistrations" / "20260718T150000.000000Z").exists()


@pytest.mark.parametrize(
    ("malformed_state", "message"),
    [
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": None,
                "status_porcelain": None,
            },
            "status_porcelain must be a string",
            id="status-and-dirty-none",
        ),
        pytest.param(
            {
                "available": True,
                "commit": "a" * 40,
                "branch": "codex/freeze",
                "dirty": False,
                "status_porcelain": "?? evidence.txt",
            },
            "dirty differs from status_porcelain",
            id="dirty-status-inconsistent",
        ),
    ],
)
def test_freeze_rejects_malformed_git_capture_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed_state: dict[str, Any],
    message: str,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    calls = 0

    def capture_git_state(project_root: str | Path | None = None) -> dict[str, Any]:
        nonlocal calls
        assert Path(project_root or root).resolve() == root.resolve()
        calls += 1
        return {
            **malformed_state,
            "captured_at_utc": "2026-07-18T15:00:01.000Z",
        }

    monkeypatch.setattr(preregistration_module, "capture_git_state", capture_git_state)

    with pytest.raises(RuntimeError, match=message):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert calls == 1
    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()
    assert not (root / "artifacts" / "preregistrations" / "20260718T150000.000000Z").exists()


def test_freeze_rechecks_pilot_stage_eligibility_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset, manifest, checksums, duplicate, pathology, pilot = _freeze_fixture(
        tmp_path, monkeypatch
    )
    original_publish = preregistration_module.publish_file_no_overwrite
    withdrawn = False

    def validate_pilot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        if withdrawn:
            raise ValueError("scientific stage eligibility changed during publication")
        return {
            "run_id": "semantic-pilot-fixture",
            "status": "completed",
            "artifact_root_sha256": "a" * 64,
            "post_seal_verification": {
                "schema_version": 1,
                "status": "passed",
                "scientific_stage_eligible": True,
            },
            "post_seal_verification_semantic_sha256": "b" * 64,
        }

    def withdraw_after_first_publish(staged: str | Path, destination: str | Path) -> Any:
        nonlocal withdrawn
        result = original_publish(staged, destination)
        withdrawn = True
        return result

    monkeypatch.setattr(preregistration_module, "_validate_completed_pilot", validate_pilot)
    monkeypatch.setattr(
        preregistration_module,
        "publish_file_no_overwrite",
        withdraw_after_first_publish,
    )

    with pytest.raises(ValueError, match="stage eligibility changed"):
        _freeze(root, dataset, manifest, checksums, duplicate, pathology, pilot)

    assert not (root / "configs" / "primary_frozen.yaml").exists()
    assert not (root / "configs" / "confirmatory_frozen.yaml").exists()
    assert not (root / "artifacts" / "preregistrations" / "20260718T150000.000000Z").exists()


def test_minimal_self_declared_sealed_pilot_is_not_freeze_eligible(tmp_path: Path) -> None:
    root, dataset, manifest, _, duplicate, _ = _project_inputs(tmp_path)
    pilot = _completed_pilot(root, dataset, manifest, duplicate)
    development = root / "reports" / "external-development.parquet"
    certificate = root / "reports" / "external-gate.json"
    development.write_bytes(b"external")
    certificate.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"development manifest|artifact"):
        preregistration_module._validate_completed_pilot(
            pilot,
            dataset_sha256=sha256_path(dataset),
            manifest_sha256=sha256_file(manifest),
            duplicate_audit_sha256=sha256_file(duplicate),
            development_manifest_source=development,
            gate_certificate_source=certificate,
        )
