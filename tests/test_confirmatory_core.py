"""Strict orchestration tests for the confirmatory matrix core."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_study_contracts import complete_confirmatory_config

from histo_audit.cross_validation.image_oof import (
    ConfirmatoryCheckpointContractError,
    ConfirmatoryCheckpointDirective,
    ConfirmatoryCheckpointPhysicalIdentity,
    ConfirmatoryImageOOFFoldEvidence,
)
from histo_audit.experiment.confirmatory_completion import (
    REAL_CONFIRMATORY_ARTIFACT_SCOPE,
    SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
)
from histo_audit.experiment.confirmatory_core import (
    ConfirmatoryCellRequest,
    ConfirmatoryFrozenBlocker,
    ConfirmatoryRunnerInputs,
    _atomic_npz,
    _checkpoint_tree_records,
    _normalise_image_execution,
    _prepare_cell_checkpoint_directory,
    _synthetic_frozen_runner,
    _synthetic_image_runner,
    _synthetic_rotation,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
    run_synthetic_confirmatory_contract_fixture,
)
from histo_audit.experiment.study_contracts import build_confirmatory_matrix_plan
from histo_audit.models.cnn import (
    OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
    ConfirmatoryCNNConfig,
)


def _blockers(config: dict[str, object]) -> dict[str, ConfirmatoryFrozenBlocker]:
    controls = confirmatory_execution_controls_from_frozen_config(config)
    return {
        scenario.scenario_id: ConfirmatoryFrozenBlocker(
            scenario_id=scenario.scenario_id,
            config_semantic_sha256=controls.config_semantic_sha256,
            availability_audit_sha256=str(scenario.availability_audit_sha256),
            blocker="frozen optional encoder unavailable",
        )
        for scenario in controls.scenario_specs
        if not scenario.required
    }


def _inject_same_byte_lexical_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    target_role: str,
) -> list[Path]:
    """Spoof a same-byte path replacement exactly while the held FD is live."""

    from histo_audit.experiment import confirmatory_core as module

    original_holder = module._hold_private_checkpoint_snapshot
    real_lstat = Path.lstat
    swapped: list[Path] = []

    @contextmanager
    def racing_holder(path: Path, *, role: str) -> Iterator[Any]:
        try:
            with original_holder(path, role=role) as snapshot:
                yield snapshot
                if role == target_role and not swapped:
                    foreign = tmp_path / f"same-byte-foreign-{len(swapped)}.bin"
                    foreign.write_bytes(snapshot.payload)
                    target = Path(path)

                    def spoofed_lstat(value: Path) -> os.stat_result:
                        if os.path.normcase(str(value)) == os.path.normcase(str(target)):
                            return real_lstat(foreign)
                        return real_lstat(value)

                    monkeypatch.setattr(Path, "lstat", spoofed_lstat)
                    swapped.append(target)
        finally:
            monkeypatch.setattr(Path, "lstat", real_lstat)

    monkeypatch.setattr(module, "_hold_private_checkpoint_snapshot", racing_holder)
    return swapped


def _real_normalise_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, Any]:
    from histo_audit.experiment import confirmatory_core as module

    checkpoint_directory = tmp_path / "cells" / "cell" / "checkpoints"
    output_directory = tmp_path / "cells" / "cell" / "checkpoint_versions" / "fold_00"
    execution_directory = tmp_path / "cells" / "cell" / "checkpoint_execution"
    checkpoint_directory.mkdir(parents=True)
    output_directory.mkdir(parents=True)
    execution_directory.mkdir(parents=True)
    checkpoint = output_directory / "epoch_0001.pt"
    checkpoint.write_bytes(b"same-byte-normalise-canary")
    canonical_checkpoint = checkpoint_directory / "fold_00.pt"
    canonical_checkpoint.write_bytes(checkpoint.read_bytes())
    commit_sidecar = output_directory / "epoch_0001.commit.json"
    commit_sidecar.write_bytes(b"same-byte-commit-sidecar")
    checkpoint.chmod(0o444)
    canonical_checkpoint.chmod(0o444)
    commit_sidecar.chmod(0o444)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(checkpoint.lstat())
    canonical_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(
        canonical_checkpoint.lstat()
    )
    assert canonical_identity.file_id_128 != checkpoint_identity.file_id_128
    commit_sha256 = hashlib.sha256(commit_sidecar.read_bytes()).hexdigest()
    commit_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(commit_sidecar.lstat())
    config = ConfirmatoryCNNConfig(
        input_variant="context_rgb",
        weight_identifier=OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        epochs=1,
        seed=303,
    )

    def canonical(value: Any) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(value: Any) -> str:
        return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

    configuration = asdict(config)
    model_metadata_binding = {"fixture": "normalise"}
    data_and_split_binding = {
        "training_data_sha256": "1" * 64,
        "reference_validation_data_sha256": "2" * 64,
        "training_split_sha256": "3" * 64,
        "reference_validation_split_sha256": "4" * 64,
    }
    config_sha256 = digest(configuration)
    directive = ConfirmatoryCheckpointDirective(
        execution_mode="fresh",
        cell_id="cell",
        fold_id=0,
        action="fresh_fit",
        source_predecessor_checkpoint=None,
        destination_imported_checkpoint=None,
        versioned_checkpoint_output_directory_relative_path=(
            "cells/cell/checkpoint_versions/fold_00"
        ),
        checkpoint_execution_manifest_relative_path=(
            "cells/cell/checkpoint_execution/fold_00.json"
        ),
        checkpoint_sha256=None,
        checkpoint_size_bytes=None,
        completed_epochs_before_fit=0,
        stopped_early_before_fit=None,
        next_epoch_index=0,
        maximum_epochs=1,
        expected_configuration_json=canonical(configuration),
        expected_configuration_sha256=config_sha256,
        expected_model_metadata_json=canonical(model_metadata_binding),
        expected_model_metadata_sha256=digest(model_metadata_binding),
        expected_data_and_split_json=canonical(data_and_split_binding),
        expected_data_and_split_sha256=digest(data_and_split_binding),
    )
    checkpoint_file_identity = {
        "path": str(checkpoint.resolve()),
        "file_id_128": checkpoint_identity.file_id_128,
        **checkpoint_identity.as_dict(),
        "sha256": checkpoint_sha256,
    }
    canonical_file_identity = {
        "path": str(canonical_checkpoint.resolve()),
        "file_id_128": canonical_identity.file_id_128,
        **canonical_identity.as_dict(),
        "sha256": checkpoint_sha256,
    }
    manifest_payload = {
        "schema_version": 3,
        "policy": "aanca_fold_boundary_checkpoint_execution_v3",
        "fit_id": "cell::fold_00",
        "fit_attempt": 1,
        "action": "fresh_fit",
        "directive_sha256": directive.directive_sha256,
        "source_predecessor_checkpoint": None,
        "destination_imported_checkpoint": None,
        "imported_checkpoint_observed": None,
        "canonical_working_checkpoint": canonical_file_identity,
        "canonical_working_checkpoint_read_only": True,
        "versioned_checkpoint_output_directory_relative_path": (
            "cells/cell/checkpoint_versions/fold_00"
        ),
        "checkpoint_execution_manifest_relative_path": (
            "cells/cell/checkpoint_execution/fold_00.json"
        ),
        "completed_epochs_before_fit": 0,
        "completed_epochs_after_fit": 1,
        "trained_epochs": 1,
        "publication_boundary": "successful_fold_completion",
        "versioned_outputs": [
            {
                "publication_index": 1,
                "completed_epochs": 1,
                "checkpoint_relative_path": (
                    "cells/cell/checkpoint_versions/fold_00/epoch_0001.pt"
                ),
                "checkpoint": checkpoint_file_identity,
                "commit_manifest_relative_path": (
                    "cells/cell/checkpoint_versions/fold_00/epoch_0001.commit.json"
                ),
                "commit_manifest_sha256": commit_sha256,
                "commit_manifest_size_bytes": commit_identity.size_bytes,
                "commit_manifest_physical_identity": commit_identity.as_dict(),
            }
        ],
        "final_checkpoint": checkpoint_file_identity,
        "automatic_retry_allowed": False,
        "imported_checkpoint_modified": False,
        "hardlink_or_replace_used_for_immutable_publication": False,
        "mutable_latest_path_created": False,
    }
    execution_manifest = execution_directory / "fold_00.json"
    execution_manifest.write_bytes(canonical(manifest_payload).encode("ascii") + b"\n")
    execution_manifest.chmod(0o444)
    manifest_identity = ConfirmatoryCheckpointPhysicalIdentity.from_stat(execution_manifest.lstat())
    provenance = SimpleNamespace(
        weight_identifier=OFFICIAL_IMAGENET_WEIGHT_IDENTIFIER,
        weights_sha256="f" * 64,
        semantic_sha256="e" * 64,
    )
    preprocessing = {
        "rgb_resize": "bilinear_antialias",
        "rgb_range_before_normalisation": [0.0, 1.0],
        "rgb_mean": [0.485, 0.456, 0.406],
        "rgb_standard_deviation": [0.229, 0.224, 0.225],
        "target_mask_resize": None,
    }
    evidence = ConfirmatoryImageOOFFoldEvidence(
        fold_id=0,
        model_seed=303,
        training_sample_ids=("training",),
        held_out_sample_ids=("held-out",),
        training_groups=("training-group",),
        held_out_groups=("held-out-group",),
        reference_validation_sample_ids=("validation",),
        reference_validation_groups=("validation-group",),
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_physical_identity=checkpoint_identity,
        checkpoint_execution_manifest_path=str(execution_manifest.resolve()),
        checkpoint_execution_manifest_sha256=hashlib.sha256(
            execution_manifest.read_bytes()
        ).hexdigest(),
        checkpoint_execution_manifest_physical_identity=manifest_identity,
        configuration_sha256=config_sha256,
        resumed_from_checkpoint=False,
        checkpoint_execution_mode="fresh",
        checkpoint_action="fresh_fit",
        checkpoint_sha256_before_fit=None,
        completed_epochs_before_fit=0,
        trained_epochs_this_invocation=1,
        successful_optimiser_steps_before_fit=0,
        successful_optimiser_steps_after_fit=1,
        successful_optimiser_steps_this_invocation=1,
        execution_mode="real_study_cuda",
        study_outcome_eligible=True,
        completed_epochs=1,
        best_epoch=1,
        best_reference_validation_loss=1.0,
        telemetry={
            "execution_mode": "real_study_cuda",
            "study_outcome_eligible": True,
        },
        model_metadata={
            "weight_identifier": provenance.weight_identifier,
            "weight_sha256": provenance.weights_sha256,
            "architecture": "torchvision.resnet18",
            "class_order": [0, 1, 2, 3, 4],
            "input_channels": 3,
            "preprocessing": preprocessing,
            "fourth_channel_initialisation": None,
        },
        data_and_split_sha256={
            "training_data_sha256": "1" * 64,
            "reference_validation_data_sha256": "2" * 64,
            "training_split_sha256": "3" * 64,
            "reference_validation_split_sha256": "4" * 64,
        },
    )
    request = SimpleNamespace(
        cpu_test_only=False,
        scenario=SimpleNamespace(
            representation_id="context_rgb",
            input_variant="context_rgb",
        ),
        cell=SimpleNamespace(cell_id="cell", model_seed=303),
        controls=SimpleNamespace(n_splits=1),
        inputs=SimpleNamespace(
            frozen_feature_provenance={"context_rgb": provenance},
        ),
        checkpoint_directory=checkpoint_directory,
        checkpoint_directives=(directive,),
    )
    result = SimpleNamespace(
        validate=lambda: None,
        oof_result=object(),
        fold_evidence=(evidence,),
        execution_mode="real_study_cuda",
        study_outcome_eligible=True,
    )
    monkeypatch.setattr(module, "_validate_oof", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_confirmatory_cnn_config",
        lambda _request, *, seed: replace(config, seed=seed),
    )
    return request, result


def test_atomic_npz_publishes_directly_without_link_or_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.npz"
    arrays = {"values": np.asarray([1, 2, 3], dtype=np.int64)}
    expected = io.BytesIO()
    np.savez_compressed(expected, **arrays)

    def forbidden_link(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("scientific publication attempted a hardlink")

    monkeypatch.setattr(os, "link", forbidden_link)
    assert _atomic_npz(destination, arrays) == destination

    assert destination.read_bytes() == expected.getvalue()


def test_atomic_npz_short_temp_supports_legal_near_max_path(tmp_path: Path) -> None:
    filename = "artifact.npz"
    base = str(tmp_path.resolve())
    padding = 249 - len(base) - len(filename) - 2
    if not 1 <= padding <= 240:
        pytest.skip("temporary root cannot construct the near-MAX_PATH fixture")
    destination = tmp_path / ("p" * padding) / filename

    published = _atomic_npz(
        destination,
        {"values": np.asarray([1, 2, 3], dtype=np.int64)},
    )

    assert len(str(destination)) == 249
    assert published == destination.resolve()
    assert destination.is_file()
    assert not list(destination.parent.glob(".s*"))


def test_normalise_image_execution_rejects_held_same_byte_lexical_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, result = _real_normalise_fixture(monkeypatch, tmp_path)
    _normalise_image_execution(request, result)
    swapped = _inject_same_byte_lexical_swap(
        monkeypatch,
        tmp_path,
        target_role="normalised image execution checkpoint",
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="changed before consumer adoption completed",
    ):
        _normalise_image_execution(request, result)

    assert len(swapped) == 1


@pytest.mark.parametrize(
    "target_role",
    [
        "normalised image execution versioned checkpoint 1",
        "normalised image execution commit sidecar 1",
    ],
)
def test_normalise_image_execution_holds_every_versioned_source(
    target_role: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    swapped = _inject_same_byte_lexical_swap(
        monkeypatch,
        tmp_path,
        target_role=target_role,
    )
    request, result = _real_normalise_fixture(monkeypatch, tmp_path)

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="changed before consumer adoption completed",
    ):
        _normalise_image_execution(request, result)

    assert len(swapped) == 1


def test_normalise_image_execution_rejects_missing_versioned_commit_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, result = _real_normalise_fixture(monkeypatch, tmp_path)
    commit_sidecar = (
        tmp_path / "cells" / "cell" / "checkpoint_versions" / "fold_00" / "epoch_0001.commit.json"
    )
    commit_sidecar.chmod(0o666)
    commit_sidecar.unlink()

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="commit sidecar 1 is unavailable",
    ):
        _normalise_image_execution(request, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "automatic_retry_allowed",
        "imported_checkpoint_modified",
        "mutable_latest_path_created",
        "extra_top_level",
        "missing_fit_id",
        "boolean_fit_attempt",
        "wrong_directive_sha256",
        "wrong_publication_index",
    ],
)
def test_normalise_image_execution_rejects_nonexact_schema_v3_manifest(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, result = _real_normalise_fixture(monkeypatch, tmp_path)
    row = result.fold_evidence[0]
    manifest = Path(row.checkpoint_execution_manifest_path)
    payload = json.loads(manifest.read_text(encoding="ascii"))
    if mutation in {
        "automatic_retry_allowed",
        "imported_checkpoint_modified",
        "mutable_latest_path_created",
    }:
        payload[mutation] = True
    elif mutation == "extra_top_level":
        payload["unapproved_field"] = "forbidden"
    elif mutation == "missing_fit_id":
        del payload["fit_id"]
    elif mutation == "boolean_fit_attempt":
        payload["fit_attempt"] = True
    elif mutation == "wrong_directive_sha256":
        payload["directive_sha256"] = "f" * 64
    elif mutation == "wrong_publication_index":
        payload["versioned_outputs"][0]["publication_index"] = 2
    else:  # pragma: no cover - closed parametrisation
        raise AssertionError(mutation)
    manifest.chmod(0o666)
    manifest.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    manifest.chmod(0o444)
    result.fold_evidence = (
        replace(
            row,
            checkpoint_execution_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            checkpoint_execution_manifest_physical_identity=(
                ConfirmatoryCheckpointPhysicalIdentity.from_stat(manifest.lstat())
            ),
        ),
    )

    with pytest.raises(ConfirmatoryCheckpointContractError):
        _normalise_image_execution(request, result)


def test_scientific_json_publication_never_overwrites_foreign_destination(
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    destination = tmp_path / "evidence.json"
    destination.write_bytes(b"foreign-evidence")

    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        module.atomic_write_json(destination, {"scientific": True})

    assert destination.read_bytes() == b"foreign-evidence"


def test_scientific_publication_rejects_same_byte_path_swap_during_creation_hold(
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    destination = tmp_path / "same-byte-swap.json"
    payload = b'{"scientific":true}\n'

    def swap_while_creation_handle_is_live(handle: Any) -> None:
        handle.write(payload)
        handle.flush()
        destination.chmod(0o666)
        destination.unlink()
        destination.write_bytes(payload)

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match=r"immutable no-overwrite publication failed|O_EXCL destination",
    ):
        module._publish_scientific_file(
            destination,
            swap_while_creation_handle_is_live,
            role="same-byte swap negative",
        )

    assert destination.exists()
    with pytest.raises(ConfirmatoryCheckpointContractError, match="not published"):
        module._scientific_artifact_record(destination)
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        module.atomic_write_text(destination, "retry is forbidden")


def test_scientific_publication_exception_after_claim_is_permanent_no_retry(
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    destination = tmp_path / "partial.json"

    def fail_after_claim(handle: Any) -> None:
        handle.write(b"partial")
        handle.flush()
        raise RuntimeError("synthetic writer failure after O_EXCL claim")

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="immutable no-overwrite publication failed",
    ):
        module._publish_scientific_file(
            destination,
            fail_after_claim,
            role="exception-after-claim negative",
        )

    assert destination.exists()
    with pytest.raises(ConfirmatoryCheckpointContractError, match="not published"):
        module._scientific_artifact_record(destination)
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        module.atomic_write_text(destination, "retry is forbidden")
    assert not any(path.name.startswith(".s") for path in tmp_path.iterdir())


def test_scientific_manifest_rejects_source_replaced_before_held_open(
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    source = module.atomic_write_json(tmp_path / "source.json", {"value": 1})
    original = source.read_bytes()
    source.chmod(0o666)
    source.unlink()
    source.write_bytes(original)

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="create-if-absent publication",
    ):
        module._publish_scientific_hash_manifest(
            tmp_path / "artifact_manifest.json",
            (source,),
        )

    assert not (tmp_path / "artifact_manifest.json").exists()


def test_scientific_manifest_holds_source_through_manifest_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    source = module.atomic_write_json(tmp_path / "source.json", {"value": 1})
    swapped = _inject_same_byte_lexical_swap(
        monkeypatch,
        tmp_path,
        target_role="scientific artifact-manifest source",
    )
    manifest = tmp_path / "artifact_manifest.json"

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="changed before consumer adoption completed",
    ):
        module._publish_scientific_hash_manifest(manifest, (source,))

    assert len(swapped) == 1
    assert manifest.is_file()


@pytest.mark.parametrize(
    "target_role",
    [
        "persisted image-OOF checkpoint",
        "persisted image-OOF checkpoint execution manifest",
    ],
)
def test_persist_completed_cell_rejects_held_same_byte_lexical_swap(
    target_role: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    swapped = _inject_same_byte_lexical_swap(
        monkeypatch,
        tmp_path,
        target_role=target_role,
    )

    output = tmp_path / "persist-race"
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="changed before consumer adoption completed",
    ):
        run_synthetic_confirmatory_contract_fixture(
            complete_confirmatory_config(),
            output_directory=output,
        )

    assert len(swapped) == 1
    cell_directory = swapped[0].parent.parent
    assert (cell_directory / "checkpoint_manifest.json").is_file()
    assert not (cell_directory / "telemetry.json").exists()
    assert not (cell_directory / "artifact_manifest.json").exists()
    assert not (output / "reconciliation.json").exists()
    assert not (output / "confirmatory_artifact_manifest.json").exists()


@pytest.mark.parametrize("streamed_role", ["root", "cell", "checkpoints"])
def test_checkpoint_tree_rejects_named_stream_on_every_directory_level(
    streamed_role: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    cells_root = tmp_path / "cells"
    cell = cells_root / "cnn_cell"
    checkpoints = cell / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "fold_00.pt").write_bytes(b"checkpoint")
    streamed = {
        "root": cells_root,
        "cell": cell,
        "checkpoints": checkpoints,
    }[streamed_role]

    monkeypatch.setattr(
        module,
        "_checkpoint_named_streams",
        lambda path: (":foreign:$DATA",) if path == streamed else (),
    )
    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="stream",
    ):
        _checkpoint_tree_records(cells_root)


def test_prepare_cell_checkpoint_directory_rejects_named_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from histo_audit.experiment import confirmatory_core as module

    checkpoint_directory = tmp_path / "cells" / "cnn_cell" / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "_checkpoint_named_streams",
        lambda path: (":foreign:$DATA",) if path == checkpoint_directory else (),
    )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="named stream",
    ):
        _prepare_cell_checkpoint_directory(checkpoint_directory, ())


@pytest.mark.skipif(os.name != "nt", reason="NTFS ADS is Windows-specific")
def test_checkpoint_tree_rejects_real_directory_ads_when_supported(
    tmp_path: Path,
) -> None:
    cells_root = tmp_path / "cells"
    checkpoint_directory = cells_root / "cnn_cell" / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    stream = Path(f"{checkpoint_directory}:aanca-test-stream")
    try:
        stream.write_bytes(b"foreign")
    except OSError:
        pytest.skip("temporary filesystem does not support directory ADS")
    try:
        with pytest.raises(
            ConfirmatoryCheckpointContractError,
            match="stream",
        ):
            _checkpoint_tree_records(cells_root)
    finally:
        stream.unlink(missing_ok=True)


def test_synthetic_fixture_executes_exact_matrix_and_stays_ineligible(tmp_path: Path) -> None:
    config = complete_confirmatory_config()
    plan = build_confirmatory_matrix_plan(config)
    result = run_synthetic_confirmatory_contract_fixture(
        config,
        output_directory=tmp_path / "confirmatory",
    )
    artifacts = result.artifacts

    assert artifacts.reconciliation.passed
    assert artifacts.reconciliation.fold_rotation_complete
    assert artifacts.reconciliation.planned_outer_folds == (1, 2, 3)
    assert result.completed_cell_count == plan.required_cell_count
    assert result.skipped_optional_cell_count == plan.optional_cell_count
    assert len(artifacts.outcomes) == len(plan.cells)
    assert artifacts.study_outcome_eligible is False
    assert artifacts.matrix_plan_path.is_file()
    assert artifacts.execution_controls_path.is_file()
    assert artifacts.frozen_feature_provenance_path.is_file()
    assert artifacts.original_audit_selection_path.is_file()
    assert artifacts.cell_index_path.is_file()
    assert artifacts.ensemble_evidence_path.is_file()
    assert artifacts.hybrid_ablations_path.is_file()
    assert artifacts.fold_aggregate_path.is_file()
    assert artifacts.reconciliation_path.is_file()
    assert artifacts.completion_evidence_path.is_file()
    assert artifacts.analysis_gaps_path.is_file()
    assert artifacts.figure_manifest_path.is_file()
    assert artifacts.report_path.is_file()
    assert artifacts.artifact_manifest_path.is_file()
    assert artifacts.artifact_manifest_path.name == "matrix_core_artifact_manifest.json"

    completed = [row for row in artifacts.outcomes if row["status"] == "completed"]
    assert {row["outer_fold"] for row in completed} == {1, 2, 3}
    assert {row["model_seed"] for row in completed} == {303, 304, 305}
    assert {row["corruption_cell_id"] for row in completed} == {
        "clean_reference_cell",
        "symmetric_ten_percent",
    }
    assert all(len(str(row["artifact_manifest_sha256"])) == 64 for row in completed)
    assert all(len(str(row["metrics_sha256"])) == 64 for row in completed)
    skipped = [row for row in artifacts.outcomes if row["status"] != "completed"]
    assert {row["status"] for row in skipped} == {"skipped_with_frozen_blocker"}
    assert all(row["frozen_unavailability"] is True for row in skipped)
    assert artifacts.cell_index_path.name == "cell_index.csv"
    completed_directory = tmp_path / "confirmatory" / "cells" / str(completed[0]["cell_id"])
    assert {
        "cell_identity.json",
        "oof_evidence.npz",
        "checkpoint_manifest.json",
        "telemetry.json",
        "risk_scores.npz",
        "ranking.csv",
        "metrics.json",
        "artifact_manifest.json",
    }.issubset({path.name for path in completed_directory.iterdir()})
    skipped_directory = tmp_path / "confirmatory" / "cells" / str(skipped[0]["cell_id"])
    assert {path.name for path in skipped_directory.iterdir()} == {
        "cell_identity.json",
        "blocker.json",
        "artifact_manifest.json",
    }
    original_binding = json.loads(
        artifacts.original_audit_selection_path.read_text(encoding="utf-8")
    )
    assert original_binding["selection"] == config["original_audit_selection"]
    assert set(original_binding["sealed_feature_cache_provenance_by_rotation"]) == {
        "1",
        "2",
        "3",
    }
    assert all(
        len(row["final_reference_group_ids_sha256"]) == 64
        for row in original_binding["sealed_feature_cache_provenance_by_rotation"].values()
    )
    frozen_provenance = json.loads(
        artifacts.frozen_feature_provenance_path.read_text(encoding="utf-8")
    )
    available_representations = {
        str(record["representation_id"])
        for record in config["cache_provenance"]
        if record["status"] == "available"
    }
    assert set(frozen_provenance["representations"]) == available_representations
    assert all(
        set(record["rotations"]) == {"1", "2", "3"}
        for record in frozen_provenance["representations"].values()
    )
    root_manifest = json.loads(artifacts.artifact_manifest_path.read_text(encoding="utf-8"))
    assert "frozen_feature_provenance.json" in root_manifest
    assert len(root_manifest["frozen_feature_provenance.json"]) == 64

    by_identity = {
        (
            int(row["outer_fold"]),
            str(row["corruption_cell_id"]),
            str(row["scenario_id"]),
            int(row["model_seed"]),
        ): str(row["cell_id"])
        for row in completed
    }
    clean_id = by_identity[(1, "clean_reference_cell", "cnn_context_rgb", 303)]
    corrupt_id = by_identity[(1, "symmetric_ten_percent", "cnn_context_rgb", 303)]
    with (
        np.load(
            tmp_path / "confirmatory" / "cells" / clean_id / "oof_evidence.npz",
            allow_pickle=False,
        ) as clean,
        np.load(
            tmp_path / "confirmatory" / "cells" / corrupt_id / "oof_evidence.npz",
            allow_pickle=False,
        ) as corrupted,
    ):
        np.testing.assert_array_equal(clean["fold_id"], corrupted["fold_id"])
        np.testing.assert_array_equal(
            clean["fold_assignment_labels"], corrupted["fold_assignment_labels"]
        )
    with np.load(
        tmp_path / "confirmatory" / "cells" / clean_id / "risk_scores.npz",
        allow_pickle=False,
    ) as risks:
        assert "fixed_hybrid" in risks.files
        assert "hybrid_drop_self_confidence" in risks.files
        assert "hybrid_drop_ensemble_disagreement" in risks.files

    input_names = (
        "sample_ids",
        "group_ids",
        "pre_corruption_label",
        "observed_label",
        "is_injected_corruption",
        "fold_id",
        "fold_assignment_labels",
    )
    for outer_fold in (1, 2, 3):
        for corruption_cell_id in {str(row["corruption_cell_id"]) for row in completed}:
            cell_ids = [
                str(row["cell_id"])
                for row in completed
                if row["outer_fold"] == outer_fold
                and row["corruption_cell_id"] == corruption_cell_id
            ]
            assert cell_ids
            with np.load(
                tmp_path / "confirmatory" / "cells" / cell_ids[0] / "oof_evidence.npz",
                allow_pickle=False,
            ) as first:
                expected = {name: first[name].copy() for name in input_names}
            for cell_id in cell_ids[1:]:
                with np.load(
                    tmp_path / "confirmatory" / "cells" / cell_id / "oof_evidence.npz",
                    allow_pickle=False,
                ) as candidate:
                    for name in input_names:
                        np.testing.assert_array_equal(candidate[name], expected[name])


def test_controls_bind_plan_and_every_new_executable_choice() -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)

    assert controls.plan == build_confirmatory_matrix_plan(config)
    assert controls.config_semantic_sha256 == controls.plan.config_sha256
    assert controls.statistical_group_unit == "source_patch_id"
    assert controls.input_size == 224
    assert controls.gradient_accumulation_steps == 2
    assert controls.class_weight == "balanced"
    assert [(item.scenario_id, item.model_seed) for item in controls.ensemble_members] == [
        ("cnn_context_rgb", 303),
        ("cnn_context_rgb", 304),
        ("cnn_context_rgb", 305),
    ]
    assert controls.restoration_scenario_id == "imagenet_frozen_logistic"
    assert controls.restoration_model_seed == 303
    assert controls.restoration_review_budget == 0.05
    assert controls.restoration_random_repeats == 100
    assert controls.restoration_random_seed == 443
    assert set(controls.restoration_conditions) == {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    }
    assert controls.paired_group_bootstrap_iterations == 2000
    assert controls.bootstrap_seed == 439
    assert len(controls.paired_comparisons) == 4
    target_comparison = controls.paired_comparisons[0]
    assert target_comparison.operand_a.scenario_id == "cnn_context_target_mask"
    assert target_comparison.operand_b.scenario_id == "cnn_context_rgb"
    assert target_comparison.operand_a.model_seed == "matched"
    assert target_comparison.operand_a.outer_fold == "all_matched"
    assert controls.holm_families == ("confirmatory_ranking",)
    controls.validate_for_plan(controls.plan)


@pytest.mark.parametrize("field", ["input_size", "gradient_accumulation_steps", "class_weight"])
def test_controls_have_no_hidden_training_defaults(field: str) -> None:
    config = complete_confirmatory_config()
    del config["training"][field]
    with pytest.raises(ValueError, match=field):
        confirmatory_execution_controls_from_frozen_config(config)


def test_rotation_rejects_final_mutation_and_cross_partition_group_leakage() -> None:
    controls = confirmatory_execution_controls_from_frozen_config(complete_confirmatory_config())
    rotation = _synthetic_rotation(1, controls)

    changed = rotation.final_observed_labels.copy()
    changed[0] = 1
    with pytest.raises(ValueError, match="final observed labels differ"):
        replace(rotation, final_observed_labels=changed).validate(controls)

    leaking_groups = list(rotation.reference_validation_group_ids)
    leaking_groups[0] = rotation.audit_group_ids[0]
    with pytest.raises(ValueError, match="source-group overlap"):
        replace(rotation, reference_validation_group_ids=tuple(leaking_groups)).validate(controls)


def test_runner_view_withholds_final_reference_outcomes() -> None:
    controls = confirmatory_execution_controls_from_frozen_config(complete_confirmatory_config())
    rotation = _synthetic_rotation(1, controls)

    runner_inputs = ConfirmatoryRunnerInputs.from_rotation(rotation)

    assert runner_inputs.final_reference_group_ids == rotation.final_group_ids
    assert not hasattr(runner_inputs, "final_sample_ids")
    assert not hasattr(runner_inputs, "final_pre_corruption_labels")
    assert not hasattr(runner_inputs, "final_observed_labels")
    assert not hasattr(runner_inputs, "final_is_injected_corruption")
    assert runner_inputs.audit_rgb.flags.writeable is False
    assert runner_inputs.reference_validation_labels.flags.writeable is False


def test_executor_rejects_plan_from_a_different_semantic_config_before_running(
    tmp_path: Path,
) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    changed = deepcopy(config)
    changed["ensemble"]["primary_risk"] = "variation_ratio"
    changed["ensemble"]["secondary_risks"] = [
        "predictive_entropy_of_mean",
        "mean_pairwise_js_divergence",
    ]
    changed_plan = build_confirmatory_matrix_plan(changed)
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)

    with pytest.raises(ValueError, match="different matrix plan"):
        execute_confirmatory_matrix(
            rotations,
            changed_plan,
            controls,
            output_directory=tmp_path / "wrong-plan",
            frozen_oof_runner=lambda _request: pytest.fail("runner must not be called"),
        )


def test_optional_skip_requires_exact_frozen_availability_binding(tmp_path: Path) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)
    blockers = _blockers(config)
    scenario_id = next(iter(blockers))
    blockers[scenario_id] = replace(blockers[scenario_id], availability_audit_sha256="0" * 64)

    with pytest.raises(ValueError, match="availability hash differs"):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "wrong-blocker",
            frozen_oof_runner=lambda _request: pytest.fail("runner must not be called"),
            frozen_blockers=blockers,
            artifact_scope=SYNTHETIC_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=True,
        )


def test_executor_requires_all_three_frozen_rotations(tmp_path: Path) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in (1, 2))

    with pytest.raises(ValueError, match="exactly all three"):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "two-rotations",
            frozen_oof_runner=lambda _request: pytest.fail("runner must not be called"),
            frozen_blockers=_blockers(config),
        )


def test_real_scope_rejects_execution_without_checkpoint_contract(
    tmp_path: Path,
) -> None:
    config = complete_confirmatory_config()
    controls = confirmatory_execution_controls_from_frozen_config(config)
    rotations = tuple(_synthetic_rotation(fold, controls) for fold in controls.official_folds)

    def relabelled_frozen(request: ConfirmatoryCellRequest) -> Any:
        result = _synthetic_frozen_runner(request)
        return replace(
            result,
            execution_mode="real_study_cpu",
            study_outcome_eligible=True,
        )

    def relabelled_image(request: ConfirmatoryCellRequest) -> Any:
        result = _synthetic_image_runner(request)
        folds = tuple(
            replace(
                row,
                execution_mode="real_study_cuda",
                study_outcome_eligible=True,
                telemetry={
                    "execution_mode": "real_study_cuda",
                    "study_outcome_eligible": True,
                },
            )
            for row in result.fold_evidence
        )
        return replace(
            result,
            fold_evidence=folds,
            execution_mode="real_study_cuda",
            study_outcome_eligible=True,
        )

    with pytest.raises(
        ConfirmatoryCheckpointContractError,
        match="explicit checkpoint contract",
    ):
        execute_confirmatory_matrix(
            rotations,
            controls.plan,
            controls,
            output_directory=tmp_path / "relabelled-fake-real",
            image_oof_runner=relabelled_image,
            frozen_oof_runner=relabelled_frozen,
            frozen_blockers=_blockers(config),
            artifact_scope=REAL_CONFIRMATORY_ARTIFACT_SCOPE,
            cpu_test_only=False,
        )
