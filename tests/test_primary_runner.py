from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import histo_audit.experiment.primary_runner as primary_runner_module
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.pannuke_primary_inputs import (
    PanNukePrimaryCachePaths,
    PanNukePrimaryHashExpectations,
)
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    SYNTHETIC_PRIMARY_ARTIFACT_SCOPE,
    PrimaryMatrixReconciliation,
)
from histo_audit.experiment.primary_core import PrimaryMatrixArtifacts
from histo_audit.experiment.primary_runner import (
    PrimaryRunnerDependencies,
    PrimaryStudyIntegrityError,
    PrimaryStudyRunnerError,
    default_primary_cache_paths,
    execute_primary_study,
)
from histo_audit.experiment.study_contracts import (
    ConfirmatoryCell,
    ConfirmatoryMatrixPlan,
    PrimaryCell,
    PrimaryMatrixPlan,
    PrimaryScenario,
)
from histo_audit.utils.run_tracking import (
    RunTracker,
    capture_source_tree,
    is_run_immutable,
    sha256_file,
    verify_run_integrity,
)
from histo_audit.workflows.study_gates import PrimaryExecutionGateEvidence


@dataclass(frozen=True)
class _FakeControls:
    plan: PrimaryMatrixPlan
    plan_sha256: str
    binding_sha256: str

    def validate_for_plan(self, plan: PrimaryMatrixPlan) -> None:
        if plan != self.plan:
            raise ValueError("wrong primary plan")


@dataclass(frozen=True)
class _FakeReadback:
    run_directory: Path
    reconciliation: PrimaryMatrixReconciliation
    completed_required_cell_count: int
    skipped_optional_cell_count: int
    status: str = "passed"
    readback_root_sha256: str = "d" * 64
    matrix_plan_sha256: str = "1" * 64
    execution_controls_sha256: str = "2" * 64
    execution_controls_binding_sha256: str = "b" * 64
    cell_index_sha256: str = "3" * 64

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self.reconciliation.passed


@dataclass(frozen=True)
class _Harness:
    root: Path
    freeze: Path
    dataset: Path
    manifest: Path
    duplicate_audit: Path
    pathology_audit: Path
    primary_config_path: Path
    confirmatory_config_path: Path
    runs_root: Path
    gate: PrimaryExecutionGateEvidence
    primary_plan: PrimaryMatrixPlan
    confirmatory_plan: ConfirmatoryMatrixPlan
    cache_paths: PanNukePrimaryCachePaths
    cache_hashes: PanNukePrimaryHashExpectations
    dependencies: PrimaryRunnerDependencies

    def execute(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "gate_evidence": self.gate,
            "project_root": self.root,
            "freeze_directory": self.freeze,
            "dataset_path": self.dataset,
            "manifest_path": self.manifest,
            "duplicate_audit_path": self.duplicate_audit,
            "pathology_encoder_audit_path": self.pathology_audit,
            "frozen_primary_config_path": self.primary_config_path,
            "frozen_confirmatory_config_path": self.confirmatory_config_path,
            "cache_paths": self.cache_paths,
            "cache_hashes": self.cache_hashes,
            "runs_root": self.runs_root,
            "dependencies": self.dependencies,
        }
        arguments.update(overrides)
        return execute_primary_study(**arguments)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _make_harness(
    tmp_path: Path,
    *,
    core_scope: str = REAL_PRIMARY_ARTIFACT_SCOPE,
    core_eligible: bool = False,
) -> _Harness:
    root = tmp_path / "project"
    root.mkdir()
    freeze = root / "freeze"
    freeze.mkdir()
    dataset = root / "data" / "raw" / "pannuke"
    dataset.mkdir(parents=True)
    _write(dataset / "fold_1.npy", "frozen dataset")
    manifest = _write(root / "data" / "manifest.parquet", "frozen manifest")
    duplicate_audit = _write(root / "evidence" / "duplicates.json", "{}")
    pathology_audit = _write(root / "evidence" / "pathology.json", "{}")
    primary_config_path = _write(root / "freeze" / "primary.yaml", "schema_version: 2\n")
    confirmatory_config_path = _write(root / "freeze" / "confirmatory.yaml", "schema_version: 2\n")
    freeze_record = _write(freeze / "freeze_evidence.json", "{}")

    primary_scenario = PrimaryScenario(
        scenario_id="symmetric_010_seed_11",
        mechanism="symmetric_random_corruption",
        rate=0.1,
        corruption_seed=11,
    )
    primary_cell = PrimaryCell(
        cell_id="symmetric_010_seed_11__engineered__logistic",
        scenario_id=primary_scenario.scenario_id,
        mechanism=primary_scenario.mechanism,
        rate=primary_scenario.rate,
        corruption_seed=primary_scenario.corruption_seed,
        representation_id="engineered_target_features",
        classifier_id="multinomial_logistic_regression",
        required=True,
    )
    primary_plan = PrimaryMatrixPlan(
        schema_version=1,
        config_sha256="a" * 64,
        scenarios=(primary_scenario,),
        cells=(primary_cell,),
    )
    confirmatory_plan = ConfirmatoryMatrixPlan(
        schema_version=1,
        config_sha256="c" * 64,
        cells=(
            ConfirmatoryCell(
                cell_id="fold_1__confirmatory",
                outer_fold=1,
                corruption_cell_id=primary_cell.cell_id,
                scenario_id=primary_scenario.scenario_id,
                model_seed=19,
                required=True,
            ),
        ),
    )
    controls = _FakeControls(
        plan=primary_plan,
        plan_sha256=canonical_sha256(primary_plan.as_dict()),
        binding_sha256="b" * 64,
    )

    cache_root = root / "cache"
    crop = _write(cache_root / "crops.npz", "crop")
    engineered = _write(cache_root / "engineered.npz", "engineered")
    context = _write(cache_root / "context.npz", "context")
    highlighted = _write(cache_root / "highlighted.npz", "highlighted")
    cache_paths = PanNukePrimaryCachePaths(
        crop_cache_path=crop,
        engineered_cache_path=engineered,
        context_embedding_cache_path=context,
        highlighted_embedding_cache_path=highlighted,
        pathology_availability_audit_path=pathology_audit,
        dataset_manifest_path=manifest,
        freeze_record_path=freeze_record,
    )
    cache_hashes = PanNukePrimaryHashExpectations(
        dataset_manifest_sha256=sha256_file(manifest),
        raw_inventory_sha256="f" * 64,
        crop_cache_sha256=sha256_file(crop),
        engineered_cache_sha256=sha256_file(engineered),
        context_embedding_cache_sha256=sha256_file(context),
        highlighted_embedding_cache_sha256=sha256_file(highlighted),
        freeze_record_sha256=sha256_file(freeze_record),
    )
    gate = PrimaryExecutionGateEvidence(
        freeze_directory=freeze,
        base_freeze_directory=freeze,
        freeze_artifact_root_sha256="0" * 64,
        freeze_manifest_sha256="1" * 64,
        preregistration_sha256="2" * 64,
        frozen_primary_config_sha256="3" * 64,
        frozen_confirmatory_config_sha256="4" * 64,
        primary_config_semantic_sha256=primary_plan.config_sha256,
        confirmatory_config_semantic_sha256=confirmatory_plan.config_sha256,
        primary_matrix_cell_count=1,
        primary_required_cell_count=1,
        confirmatory_matrix_cell_count=1,
        pilot_run_id="pilot-001",
        pilot_artifact_root_sha256="5" * 64,
        dataset_sha256="6" * 64,
        manifest_sha256=sha256_file(manifest),
        duplicate_audit_sha256="7" * 64,
        pathology_encoder_audit_sha256="8" * 64,
        source_tree_root_sha256=str(capture_source_tree(root)["root_sha256"]),
    )
    reconciliation = PrimaryMatrixReconciliation(
        status="passed",
        planned_cell_count=1,
        planned_required_cell_count=1,
        completed_cell_count=1,
        completed_required_cell_count=1,
        skipped_optional_cell_count=0,
        failed_cell_count=0,
        missing_cell_ids=(),
        extra_cell_ids=(),
        duplicate_cell_ids=(),
        invalid_cell_ids=(),
        errors=(),
    )

    def config_loader(path: str | Path) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "study": "primary" if Path(path) == primary_config_path else "confirmatory",
        }

    prepared = SimpleNamespace(
        inputs=object(),
        plan=primary_plan,
        config_sha256=primary_plan.config_sha256,
        plan_semantic_sha256=controls.plan_sha256,
        verified_hashes={"crop_cache_sha256": sha256_file(crop)},
        sample_order_sha256="a" * 64,
        partition_assignment_sha256="b" * 64,
        cache_provenance_by_representation={},
        representation_availability=(),
    )

    def matrix_executor(
        inputs: object,
        plan: PrimaryMatrixPlan,
        *,
        output_directory: str | Path,
        execution_controls: _FakeControls,
    ) -> PrimaryMatrixArtifacts:
        del inputs, execution_controls
        output = Path(output_directory)
        matrix_plan_path = _write(
            output / "matrix_plan.json", json.dumps(plan.as_dict(), sort_keys=True)
        )
        execution_controls_path = _write(output / "execution_controls.json", "{}")
        cell_index_path = _write(output / "cell_index.csv", "cell_id,status\n")
        reconciliation_path = _write(
            output / "reconciliation.json",
            json.dumps(reconciliation.as_dict(), sort_keys=True),
        )
        completion_path = _write(
            output / "completion_evidence.json",
            json.dumps(
                {
                    "schema_version": 2,
                    "completion_stage": None,
                    "study_outcome_eligible": core_eligible,
                    "artifact_scope": core_scope,
                    "matrix_config_sha256": plan.config_sha256,
                    "planned_cell_count": 1,
                    "required_cell_count": 1,
                    "completed_required_cell_count": 1,
                    "skipped_optional_cell_count": 0,
                    "failed_required_cell_count": 0,
                    "reconciliation_status": "passed",
                    "execution_controls_binding_sha256": controls.binding_sha256,
                },
                sort_keys=True,
            ),
        )
        restoration_path = _write(output / "restoration_index.json", "{}")
        report_path = _write(output / "report.md", "# Core draft\n")
        return PrimaryMatrixArtifacts(
            output_directory=output,
            matrix_plan_path=matrix_plan_path,
            execution_controls_path=execution_controls_path,
            cell_index_path=cell_index_path,
            reconciliation_path=reconciliation_path,
            completion_evidence_path=completion_path,
            restoration_path=restoration_path,
            report_path=report_path,
            outcomes=(),
            reconciliation=reconciliation,
        )

    def filesystem_reader(plan: PrimaryMatrixPlan, run_directory: str | Path) -> _FakeReadback:
        assert plan == primary_plan
        return _FakeReadback(
            run_directory=Path(run_directory).resolve(),
            reconciliation=reconciliation,
            completed_required_cell_count=1,
            skipped_optional_cell_count=0,
        )

    def completion_builder(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["gate_evidence"] == gate
        assert kwargs["filesystem_readback"].passed
        return {
            "schema_version": 2,
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "study_outcome_eligible": True,
            "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
            "matrix_config_sha256": primary_plan.config_sha256,
            "planned_cell_count": 1,
            "required_cell_count": 1,
            "completed_required_cell_count": 1,
            "skipped_optional_cell_count": 0,
            "failed_required_cell_count": 0,
            "reconciliation_status": "passed",
            "completion_stage_enabled_only_after_run_seal_and_integrity_verification": True,
            "filesystem_matrix_plan_sha256": kwargs["filesystem_readback"].matrix_plan_sha256,
            "filesystem_execution_controls_sha256": kwargs[
                "filesystem_readback"
            ].execution_controls_sha256,
            "filesystem_execution_controls_binding_sha256": kwargs[
                "filesystem_readback"
            ].execution_controls_binding_sha256,
            "filesystem_cell_index_sha256": kwargs["filesystem_readback"].cell_index_sha256,
            "filesystem_readback_root_sha256": kwargs["filesystem_readback"].readback_root_sha256,
            **{
                field: getattr(gate, field)
                for field in (
                    "freeze_artifact_root_sha256",
                    "frozen_primary_config_sha256",
                    "frozen_confirmatory_config_sha256",
                    "dataset_sha256",
                    "manifest_sha256",
                    "duplicate_audit_sha256",
                    "pathology_encoder_audit_sha256",
                    "source_tree_root_sha256",
                )
            },
        }

    def statistics_aggregator(
        run_directory: str | Path, execution_controls: _FakeControls
    ) -> SimpleNamespace:
        assert execution_controls == controls
        output = Path(run_directory)
        _write(output / "primary_statistics.json", "{}")
        _write(output / "primary_bootstrap_evidence.npz", "evidence")
        _write(output / "primary_subgroups.csv", "cell_id,status\n")
        manifest_path = _write(output / "primary_statistics_manifest.json", "{}")
        return SimpleNamespace(
            output_directory=output,
            manifest_path=manifest_path,
            manifest_sha256=sha256_file(manifest_path),
        )

    def statistics_verifier(
        run_directory: str | Path, execution_controls: _FakeControls
    ) -> SimpleNamespace:
        assert execution_controls == controls
        output = Path(run_directory)
        return SimpleNamespace(
            valid=True,
            output_directory=output,
            manifest_sha256=sha256_file(output / "primary_statistics_manifest.json"),
            source_readback_root_sha256="4" * 64,
            comparison_count=3,
        )

    dependencies = PrimaryRunnerDependencies(
        gate_validator=lambda **_: gate,
        config_loader=config_loader,
        primary_plan_builder=lambda _: primary_plan,
        confirmatory_plan_builder=lambda _: confirmatory_plan,
        controls_builder=lambda _: controls,  # type: ignore[arg-type]
        input_builder=lambda *_args, **_kwargs: prepared,  # type: ignore[arg-type]
        matrix_executor=matrix_executor,  # type: ignore[arg-type]
        filesystem_reader=filesystem_reader,  # type: ignore[arg-type]
        completion_builder=completion_builder,
        statistics_aggregator=statistics_aggregator,  # type: ignore[arg-type]
        statistics_verifier=statistics_verifier,  # type: ignore[arg-type]
        tracker_starter=RunTracker.start,
        integrity_verifier=verify_run_integrity,
    )
    return _Harness(
        root=root,
        freeze=freeze,
        dataset=dataset,
        manifest=manifest,
        duplicate_audit=duplicate_audit,
        pathology_audit=pathology_audit,
        primary_config_path=primary_config_path,
        confirmatory_config_path=confirmatory_config_path,
        runs_root=root / "runs",
        gate=gate,
        primary_plan=primary_plan,
        confirmatory_plan=confirmatory_plan,
        cache_paths=cache_paths,
        cache_hashes=cache_hashes,
        dependencies=dependencies,
    )


def _run_directories(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def test_default_primary_cache_paths_bind_the_explicit_base_freeze(tmp_path: Path) -> None:
    root = tmp_path / "project"
    base_freeze = root / "artifacts" / "preregistrations" / "base"
    manifest = root / "data" / "manifest.parquet"
    pathology_audit = root / "reports" / "pathology.json"

    paths = default_primary_cache_paths(
        project_root=root,
        freeze_directory=base_freeze,
        manifest_path=manifest,
        pathology_encoder_audit_path=pathology_audit,
    )

    cache_root = root / "artifacts" / "embeddings" / "pannuke"
    assert paths.crop_cache_path == cache_root / "pannuke_crops.npz"
    assert paths.engineered_cache_path == cache_root / "pannuke_engineered_features.npz"
    assert (
        paths.context_embedding_cache_path
        == cache_root / "pannuke_resnet18_context_rgb_embeddings.npz"
    )
    assert (
        paths.highlighted_embedding_cache_path
        == cache_root / "pannuke_resnet18_target_highlighted_embeddings.npz"
    )
    assert paths.dataset_manifest_path == manifest
    assert paths.pathology_availability_audit_path == pathology_audit
    assert paths.freeze_record_path == base_freeze / "freeze_evidence.json"


def test_live_gate_runs_before_tracker_and_gate_failure_creates_no_artifacts(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    calls: list[str] = []

    def fail_gate(**_: Any) -> PrimaryExecutionGateEvidence:
        calls.append("gate")
        raise ValueError("live gate rejected changed source tree")

    def forbidden_tracker(**_: Any) -> RunTracker:
        calls.append("tracker")
        raise AssertionError("tracker must not run after a failed gate")

    dependencies = replace(
        harness.dependencies,
        gate_validator=fail_gate,
        tracker_starter=forbidden_tracker,
    )
    with pytest.raises(ValueError, match="live gate rejected"):
        harness.execute(dependencies=dependencies)

    assert calls == ["gate"]
    assert not harness.runs_root.exists()


def test_final_pretracker_boundary_rejects_cache_drift_without_creating_run(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    original_builder = harness.dependencies.input_builder

    def mutate_after_preflight(*args: Any, **kwargs: Any):
        prepared = original_builder(*args, **kwargs)
        _write(harness.cache_paths.crop_cache_path, "changed after adapter preflight")
        return prepared

    dependencies = replace(
        harness.dependencies,
        input_builder=mutate_after_preflight,
    )
    with pytest.raises(PrimaryStudyRunnerError, match="crop cache differs"):
        harness.execute(dependencies=dependencies)

    assert not harness.runs_root.exists()


def test_source_only_amendment_resolves_cache_binding_from_base_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path)
    amendment = harness.root / "amendment"
    amendment.mkdir()
    amended_gate = replace(
        harness.gate,
        freeze_directory=amendment,
        registration_authority_kind="preregistration_amendment",
        registration_authority_chain_depth=1,
        original_unamended_primary_claim_allowed=False,
        amended_primary_claim_allowed=True,
    )
    resolved_freeze_directories: list[Path] = []

    def default_paths(**kwargs: Any) -> PanNukePrimaryCachePaths:
        resolved_freeze_directories.append(Path(kwargs["freeze_directory"]).resolve())
        return harness.cache_paths

    monkeypatch.setattr(primary_runner_module, "default_primary_cache_paths", default_paths)

    class ExpectedPretrackerStop(RuntimeError):
        pass

    def stop_after_cache_path_resolution(*_args: Any, **_kwargs: Any) -> Any:
        raise ExpectedPretrackerStop("cache paths resolved from base freeze")

    dependencies = replace(
        harness.dependencies,
        gate_validator=lambda **_: amended_gate,
        input_builder=stop_after_cache_path_resolution,
    )
    with pytest.raises(ExpectedPretrackerStop, match="resolved from base freeze"):
        harness.execute(
            gate_evidence=amended_gate,
            freeze_directory=amendment,
            cache_paths=None,
            dependencies=dependencies,
        )

    assert resolved_freeze_directories == [harness.freeze.resolve()]
    assert not harness.runs_root.exists()


@pytest.mark.parametrize(
    ("scope", "eligible"),
    [
        (SYNTHETIC_PRIMARY_ARTIFACT_SCOPE, False),
        (REAL_PRIMARY_ARTIFACT_SCOPE, True),
    ],
)
def test_rejects_synthetic_or_prematurely_eligible_core_draft_and_seals_failure(
    tmp_path: Path,
    scope: str,
    eligible: bool,
) -> None:
    harness = _make_harness(tmp_path, core_scope=scope, core_eligible=eligible)

    with pytest.raises(PrimaryStudyRunnerError, match="ineligible real-scope"):
        harness.execute(run_id="rejected-core")

    run_directory = harness.runs_root / "rejected-core"
    completion = json.loads(
        (run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
    assert completion["completion_stage"] is None
    assert completion["study_outcome_eligible"] is False
    assert completion["valid_completion_claim"] is False
    assert status["status"] == "failed"
    assert is_run_immutable(run_directory)
    assert verify_run_integrity(run_directory).valid


def test_post_seal_integrity_failure_never_returns_completion_claim(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)

    def corrupt_then_verify(run_directory: str | Path):
        _write(Path(run_directory) / "completion_evidence.json", "{}")
        return verify_run_integrity(run_directory)

    dependencies = replace(
        harness.dependencies,
        integrity_verifier=corrupt_then_verify,
    )
    with pytest.raises(PrimaryStudyIntegrityError, match="no valid PRIMARY_STUDY_COMPLETE"):
        harness.execute(run_id="integrity-failure", dependencies=dependencies)

    run_directory = harness.runs_root / "integrity-failure"
    assert is_run_immutable(run_directory)
    assert not verify_run_integrity(run_directory).valid
    assert (
        json.loads((run_directory / "completion_evidence.json").read_text(encoding="utf-8")) == {}
    )


def test_happy_mocked_full_path_seals_registers_verifies_then_returns_stage(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    call_order: list[str] = []

    def gate(**_: Any) -> PrimaryExecutionGateEvidence:
        call_order.append("gate")
        return harness.gate

    def tracker(**kwargs: Any) -> RunTracker:
        call_order.append("tracker")
        return RunTracker.start(**kwargs)

    dependencies = replace(
        harness.dependencies,
        gate_validator=gate,
        tracker_starter=tracker,
    )
    result = harness.execute(run_id="happy-primary", dependencies=dependencies)

    run_directory = Path(result["run_directory"])
    verification = verify_run_integrity(run_directory)
    completion = json.loads(
        (run_directory / "completion_evidence.json").read_text(encoding="utf-8")
    )
    assert call_order == ["gate", "gate", "tracker"]
    assert result["completion_stage"] == "PRIMARY_STUDY_COMPLETE"
    assert result["study_outcome_eligible"] is True
    assert result["artifact_root_sha256"] == verification.expected_root_sha256
    assert verification.valid and verification.registry_record_present
    assert completion["completion_stage"] == "PRIMARY_STUDY_COMPLETE"
    assert completion["study_outcome_eligible"] is True
    assert completion["run_id"] == "happy-primary"
    assert is_run_immutable(run_directory)


def test_resume_is_explicitly_rejected_without_starting_or_overwriting(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    with pytest.raises(NotImplementedError, match="resume is not implemented"):
        harness.execute(resume_run_directory=harness.root / "old-run")
    assert _run_directories(harness.runs_root) == []

    first = harness.execute(run_id="fixed-id")
    first_directory = Path(first["run_directory"])
    first_root = verify_run_integrity(first_directory).expected_root_sha256
    with pytest.raises(FileExistsError):
        harness.execute(run_id="fixed-id")
    assert verify_run_integrity(first_directory).expected_root_sha256 == first_root
    assert verify_run_integrity(first_directory).valid
    assert _run_directories(harness.runs_root) == [first_directory]
