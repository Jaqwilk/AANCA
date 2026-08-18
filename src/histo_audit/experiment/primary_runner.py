"""Tracked, fail-closed runner for the real frozen PanNuke primary study.

The CLI gate is intentionally not trusted as a capability token.  This module repeats
the complete read-only gate before it creates a :class:`RunTracker`, preflights every
frozen config/cache binding, executes only the public real-scope matrix API, reconciles
the concrete filesystem, seals the run, and returns a completion stage only after an
independent post-seal integrity verification succeeds.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from histo_audit.config import load_config
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.pannuke_primary_inputs import (
    PanNukePrimaryCachePaths,
    PanNukePrimaryHashExpectations,
    PanNukePrimaryInputsResult,
    build_pannuke_primary_inputs,
)
from histo_audit.experiment.primary_completion import (
    REAL_PRIMARY_ARTIFACT_SCOPE,
    PrimaryFilesystemReadbackEvidence,
    build_primary_completion_evidence,
    read_primary_filesystem_evidence,
)
from histo_audit.experiment.primary_core import (
    PrimaryExecutionControls,
    PrimaryMatrixArtifacts,
    execute_primary_matrix,
    primary_execution_controls_from_frozen_config,
)
from histo_audit.experiment.primary_statistics import (
    PrimaryStatisticsArtifacts,
    PrimaryStatisticsVerification,
    aggregate_primary_statistics,
    verify_primary_statistics_artifacts,
)
from histo_audit.experiment.study_contracts import (
    ConfirmatoryMatrixPlan,
    PrimaryMatrixPlan,
    build_confirmatory_matrix_plan,
    build_primary_matrix_plan,
)
from histo_audit.utils.run_tracking import (
    IntegrityVerification,
    RunTracker,
    sha256_file,
    verify_run_integrity,
)
from histo_audit.workflows.study_gates import (
    PrimaryExecutionGateEvidence,
    validate_primary_execution_gate,
)

_COMPLETION_STAGE = "PRIMARY_STUDY_COMPLETE"
_EXPERIMENT_NAME = "pannuke_primary_frozen_feature_benchmark"
_GATE_COMPLETION_BINDINGS = (
    "freeze_artifact_root_sha256",
    "frozen_primary_config_sha256",
    "frozen_confirmatory_config_sha256",
    "dataset_sha256",
    "manifest_sha256",
    "duplicate_audit_sha256",
    "pathology_encoder_audit_sha256",
    "source_tree_root_sha256",
)


class PrimaryStudyRunnerError(RuntimeError):
    """The real primary workflow failed without producing a valid stage claim."""


class PrimaryStudyIntegrityError(PrimaryStudyRunnerError):
    """A sealed candidate did not pass the required independent integrity check."""


@dataclass(frozen=True, slots=True)
class PrimaryRunnerDependencies:
    """Injectable workflow boundaries used by focused orchestration tests."""

    gate_validator: Callable[..., PrimaryExecutionGateEvidence] = validate_primary_execution_gate
    config_loader: Callable[[str | Path], dict[str, Any]] = load_config
    primary_plan_builder: Callable[[Mapping[str, Any]], PrimaryMatrixPlan] = (
        build_primary_matrix_plan
    )
    confirmatory_plan_builder: Callable[[Mapping[str, Any]], ConfirmatoryMatrixPlan] = (
        build_confirmatory_matrix_plan
    )
    controls_builder: Callable[[Mapping[str, Any]], PrimaryExecutionControls] = (
        primary_execution_controls_from_frozen_config
    )
    input_builder: Callable[..., PanNukePrimaryInputsResult] = build_pannuke_primary_inputs
    matrix_executor: Callable[..., PrimaryMatrixArtifacts] = execute_primary_matrix
    filesystem_reader: Callable[
        [PrimaryMatrixPlan, str | Path], PrimaryFilesystemReadbackEvidence
    ] = read_primary_filesystem_evidence
    completion_builder: Callable[..., dict[str, Any]] = build_primary_completion_evidence
    statistics_aggregator: Callable[
        [str | Path, PrimaryExecutionControls], PrimaryStatisticsArtifacts
    ] = aggregate_primary_statistics
    statistics_verifier: Callable[
        [str | Path, PrimaryExecutionControls], PrimaryStatisticsVerification
    ] = verify_primary_statistics_artifacts
    tracker_starter: Callable[..., RunTracker] = RunTracker.start
    integrity_verifier: Callable[[str | Path], IntegrityVerification] = verify_run_integrity


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _normalise_cache_paths(
    root: Path,
    paths: PanNukePrimaryCachePaths,
) -> PanNukePrimaryCachePaths:
    def optional(value: Path | None) -> Path | None:
        return _resolve(root, value) if value is not None else None

    return PanNukePrimaryCachePaths(
        crop_cache_path=_resolve(root, paths.crop_cache_path),
        engineered_cache_path=_resolve(root, paths.engineered_cache_path),
        context_embedding_cache_path=optional(paths.context_embedding_cache_path),
        highlighted_embedding_cache_path=optional(paths.highlighted_embedding_cache_path),
        pathology_embedding_cache_path=optional(paths.pathology_embedding_cache_path),
        pathology_availability_audit_path=optional(paths.pathology_availability_audit_path),
        dataset_evidence_path=optional(paths.dataset_evidence_path),
        dataset_manifest_path=optional(paths.dataset_manifest_path),
        freeze_record_path=optional(paths.freeze_record_path),
    )


def default_primary_cache_paths(
    *,
    project_root: str | Path,
    freeze_directory: str | Path,
    manifest_path: str | Path,
    pathology_encoder_audit_path: str | Path,
) -> PanNukePrimaryCachePaths:
    """Return the explicit conventional cache layout used by the extraction CLI."""

    root = Path(project_root).resolve()
    cache_root = root / "artifacts" / "embeddings" / "pannuke"
    pathology_cache = cache_root / "pannuke_pathology_embeddings.npz"
    return PanNukePrimaryCachePaths(
        crop_cache_path=cache_root / "pannuke_crops.npz",
        engineered_cache_path=cache_root / "pannuke_engineered_features.npz",
        context_embedding_cache_path=(cache_root / "pannuke_resnet18_context_rgb_embeddings.npz"),
        highlighted_embedding_cache_path=(
            cache_root / "pannuke_resnet18_target_highlighted_embeddings.npz"
        ),
        pathology_embedding_cache_path=(pathology_cache if pathology_cache.is_file() else None),
        pathology_availability_audit_path=_resolve(root, pathology_encoder_audit_path),
        dataset_manifest_path=_resolve(root, manifest_path),
        freeze_record_path=_resolve(root, freeze_directory) / "freeze_evidence.json",
    )


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_file(path: Path | None, role: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{role} path is not configured")
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    return path


def _crop_raw_inventory_sha(crop_cache: Path) -> str:
    metadata_path = crop_cache.with_suffix(f"{crop_cache.suffix}.metadata.json")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryStudyRunnerError(
            f"crop metadata sidecar is unavailable or invalid: {metadata_path}: {exc}"
        ) from exc
    value = payload.get("raw_inventory_sha256") if isinstance(payload, Mapping) else None
    if not _valid_sha(value):
        raise PrimaryStudyRunnerError("crop metadata lacks a valid raw_inventory_sha256")
    return str(value)


def derive_primary_cache_hashes(
    paths: PanNukePrimaryCachePaths,
    *,
    manifest_sha256: str,
) -> PanNukePrimaryHashExpectations:
    """Hash every configured cache before a tracked run is created."""

    crop = _required_file(paths.crop_cache_path, "crop cache")
    engineered = _required_file(paths.engineered_cache_path, "engineered cache")
    context = _required_file(paths.context_embedding_cache_path, "context embedding cache")
    highlighted = _required_file(
        paths.highlighted_embedding_cache_path, "highlighted embedding cache"
    )
    freeze = _required_file(paths.freeze_record_path, "freeze evidence")
    manifest = _required_file(paths.dataset_manifest_path, "dataset manifest")
    pathology = paths.pathology_embedding_cache_path
    if pathology is not None:
        pathology = _required_file(pathology, "pathology embedding cache")
    dataset_evidence = paths.dataset_evidence_path
    if dataset_evidence is not None:
        dataset_evidence = _required_file(dataset_evidence, "dataset evidence")
    if sha256_file(manifest) != manifest_sha256:
        raise PrimaryStudyRunnerError("cache manifest path differs from gate evidence")
    return PanNukePrimaryHashExpectations(
        dataset_evidence_sha256=(
            sha256_file(dataset_evidence) if dataset_evidence is not None else None
        ),
        dataset_manifest_sha256=manifest_sha256,
        raw_inventory_sha256=_crop_raw_inventory_sha(crop),
        crop_cache_sha256=sha256_file(crop),
        engineered_cache_sha256=sha256_file(engineered),
        context_embedding_cache_sha256=sha256_file(context),
        highlighted_embedding_cache_sha256=sha256_file(highlighted),
        pathology_embedding_cache_sha256=(
            sha256_file(pathology) if pathology is not None else None
        ),
        freeze_record_sha256=sha256_file(freeze),
    )


def _verify_explicit_cache_hashes(
    paths: PanNukePrimaryCachePaths,
    hashes: PanNukePrimaryHashExpectations,
    *,
    manifest_sha256: str,
) -> None:
    hashes.validate()
    checks = (
        (paths.crop_cache_path, hashes.crop_cache_sha256, "crop cache"),
        (paths.engineered_cache_path, hashes.engineered_cache_sha256, "engineered cache"),
        (
            paths.context_embedding_cache_path,
            hashes.context_embedding_cache_sha256,
            "context embedding cache",
        ),
        (
            paths.highlighted_embedding_cache_path,
            hashes.highlighted_embedding_cache_sha256,
            "highlighted embedding cache",
        ),
        (paths.freeze_record_path, hashes.freeze_record_sha256, "freeze evidence"),
        (paths.dataset_manifest_path, hashes.dataset_manifest_sha256, "dataset manifest"),
    )
    for raw_path, expected, role in checks:
        path = _required_file(raw_path, role)
        if expected is None or sha256_file(path) != expected:
            raise PrimaryStudyRunnerError(f"{role} differs from its explicit SHA-256")
    if hashes.dataset_manifest_sha256 != manifest_sha256:
        raise PrimaryStudyRunnerError("explicit manifest SHA differs from gate evidence")
    if not _valid_sha(hashes.raw_inventory_sha256):
        raise PrimaryStudyRunnerError("raw inventory SHA must be explicit")
    if (paths.pathology_embedding_cache_path is None) != (
        hashes.pathology_embedding_cache_sha256 is None
    ):
        raise PrimaryStudyRunnerError("pathology cache path/SHA must be supplied together")
    if (
        paths.pathology_embedding_cache_path is not None
        and sha256_file(
            _required_file(paths.pathology_embedding_cache_path, "pathology embedding cache")
        )
        != hashes.pathology_embedding_cache_sha256
    ):
        raise PrimaryStudyRunnerError("pathology embedding cache differs from its SHA-256")
    if (paths.dataset_evidence_path is None) != (hashes.dataset_evidence_sha256 is None):
        raise PrimaryStudyRunnerError("dataset evidence path/SHA must be supplied together")
    if (
        paths.dataset_evidence_path is not None
        and sha256_file(_required_file(paths.dataset_evidence_path, "dataset evidence"))
        != hashes.dataset_evidence_sha256
    ):
        raise PrimaryStudyRunnerError("dataset evidence differs from its SHA-256")


def _cache_paths_payload(paths: PanNukePrimaryCachePaths) -> dict[str, str | None]:
    return {key: str(value) if value is not None else None for key, value in asdict(paths).items()}


def _validate_gate_equality(
    supplied: PrimaryExecutionGateEvidence,
    live: PrimaryExecutionGateEvidence,
) -> None:
    if not isinstance(supplied, PrimaryExecutionGateEvidence):
        raise TypeError("gate_evidence must be a real PrimaryExecutionGateEvidence instance")
    if supplied != live:
        raise PrimaryStudyRunnerError(
            "supplied primary gate evidence differs from the mandatory live revalidation"
        )


def _validate_frozen_plans(
    gate: PrimaryExecutionGateEvidence,
    primary_plan: PrimaryMatrixPlan,
    confirmatory_plan: ConfirmatoryMatrixPlan,
    controls: PrimaryExecutionControls,
) -> None:
    controls.validate_for_plan(primary_plan)
    if primary_plan.config_sha256 != gate.primary_config_semantic_sha256:
        raise PrimaryStudyRunnerError("primary plan differs from gate semantic evidence")
    if confirmatory_plan.config_sha256 != gate.confirmatory_config_semantic_sha256:
        raise PrimaryStudyRunnerError("confirmatory plan differs from gate semantic evidence")
    if len(primary_plan.cells) != gate.primary_matrix_cell_count:
        raise PrimaryStudyRunnerError("primary matrix cell count differs from gate evidence")
    if primary_plan.required_cell_count != gate.primary_required_cell_count:
        raise PrimaryStudyRunnerError("required primary cell count differs from gate evidence")
    if len(confirmatory_plan.cells) != gate.confirmatory_matrix_cell_count:
        raise PrimaryStudyRunnerError("confirmatory matrix cell count differs from gate evidence")
    if controls.plan != primary_plan or controls.plan_sha256 != canonical_sha256(
        primary_plan.as_dict()
    ):
        raise PrimaryStudyRunnerError("primary execution controls differ from the frozen plan")


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryStudyRunnerError(f"{role} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrimaryStudyRunnerError(f"{role} must be a JSON object")
    return cast(dict[str, Any], payload)


def _validate_core_artifacts(
    artifacts: PrimaryMatrixArtifacts,
    *,
    run_directory: Path,
    plan: PrimaryMatrixPlan,
    controls: PrimaryExecutionControls,
) -> dict[str, Any]:
    if Path(artifacts.output_directory).resolve() != run_directory:
        raise PrimaryStudyRunnerError("matrix executor returned a different output directory")
    required_paths = (
        (artifacts.matrix_plan_path, run_directory / "matrix_plan.json"),
        (artifacts.execution_controls_path, run_directory / "execution_controls.json"),
        (artifacts.cell_index_path, run_directory / "cell_index.csv"),
        (artifacts.reconciliation_path, run_directory / "reconciliation.json"),
        (artifacts.completion_evidence_path, run_directory / "completion_evidence.json"),
        (artifacts.restoration_path, run_directory / "restoration_index.json"),
        (artifacts.report_path, run_directory / "report.md"),
    )
    if any(
        Path(actual).resolve() != expected.resolve() or not expected.is_file()
        for actual, expected in required_paths
    ):
        raise PrimaryStudyRunnerError(
            "matrix executor omitted or redirected a required real-scope root artifact"
        )
    core_completion = _read_json_object(
        Path(artifacts.completion_evidence_path), "core completion evidence"
    )
    if (
        core_completion.get("schema_version") != 2
        or core_completion.get("completion_stage") is not None
        or core_completion.get("study_outcome_eligible") is not False
        or core_completion.get("artifact_scope") != REAL_PRIMARY_ARTIFACT_SCOPE
        or core_completion.get("matrix_config_sha256") != plan.config_sha256
        or core_completion.get("planned_cell_count") != len(plan.cells)
        or core_completion.get("required_cell_count") != plan.required_cell_count
        or core_completion.get("completed_required_cell_count")
        != artifacts.reconciliation.completed_required_cell_count
        or core_completion.get("failed_required_cell_count")
        != plan.required_cell_count - artifacts.reconciliation.completed_required_cell_count
        or core_completion.get("reconciliation_status") != artifacts.reconciliation.status
        or core_completion.get("execution_controls_binding_sha256") != controls.binding_sha256
    ):
        raise PrimaryStudyRunnerError(
            "matrix executor must return an ineligible real-scope core completion draft"
        )
    return core_completion


def _validate_completion_candidate(
    candidate: Mapping[str, Any],
    *,
    plan: PrimaryMatrixPlan,
    gate: PrimaryExecutionGateEvidence,
    readback: PrimaryFilesystemReadbackEvidence,
) -> dict[str, Any]:
    if (
        candidate.get("schema_version") != 2
        or candidate.get("completion_stage") != _COMPLETION_STAGE
        or candidate.get("study_outcome_eligible") is not True
        or candidate.get("artifact_scope") != REAL_PRIMARY_ARTIFACT_SCOPE
        or candidate.get("matrix_config_sha256") != plan.config_sha256
        or candidate.get("planned_cell_count") != len(plan.cells)
        or candidate.get("required_cell_count") != plan.required_cell_count
        or candidate.get("completed_required_cell_count") != plan.required_cell_count
        or candidate.get("skipped_optional_cell_count") != readback.skipped_optional_cell_count
        or candidate.get("failed_required_cell_count") != 0
        or candidate.get("reconciliation_status") != "passed"
        or candidate.get("completion_stage_enabled_only_after_run_seal_and_integrity_verification")
        is not True
        or candidate.get("filesystem_matrix_plan_sha256") != readback.matrix_plan_sha256
        or candidate.get("filesystem_execution_controls_sha256")
        != readback.execution_controls_sha256
        or candidate.get("filesystem_execution_controls_binding_sha256")
        != readback.execution_controls_binding_sha256
        or candidate.get("filesystem_cell_index_sha256") != readback.cell_index_sha256
        or candidate.get("filesystem_readback_root_sha256") != readback.readback_root_sha256
    ):
        raise PrimaryStudyRunnerError(
            "completion builder did not produce an exact eligible real primary claim"
        )
    for field in _GATE_COMPLETION_BINDINGS:
        if candidate.get(field) != getattr(gate, field):
            raise PrimaryStudyRunnerError(
                f"completion builder did not preserve gate binding {field}"
            )
    return dict(candidate)


def _report_text(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# Frozen real PanNuke primary study",
            "",
            "This controlled benchmark ranks potentially inconsistent annotations for "
            "recommended expert review. It does not modify source annotations.",
            "",
            f"- Matrix cells: {metrics['planned_cell_count']}",
            f"- Required cells completed: {metrics['completed_required_cell_count']}",
            f"- Optional cells skipped under frozen blockers: "
            f"{metrics['skipped_optional_cell_count']}",
            f"- Filesystem readback: `{metrics['filesystem_readback_status']}`",
            f"- Frozen primary statistics manifest: "
            f"`{metrics['primary_statistics_manifest_sha256']}`",
            f"- Completion candidate: `{metrics['completion_stage']}`",
            "- The completion candidate is valid only after the immutable run seal, "
            "append-only registry record, and post-seal integrity verification all pass.",
            "- Final reference groups remained uncorrupted and unavailable for selection.",
            "",
        )
    )


def _demote_mutable_claim(
    tracker: RunTracker,
    *,
    core_completion: Mapping[str, Any] | None,
    error: BaseException,
) -> None:
    """Best-effort removal of an eligible candidate before a failed run is sealed."""

    if tracker.finalized:
        return
    fallback = dict(core_completion or {})
    fallback.update(
        {
            "schema_version": fallback.get("schema_version", 2),
            "completion_stage": None,
            "study_outcome_eligible": False,
            "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
            "runner_failure": f"{type(error).__name__}: {error}",
            "valid_completion_claim": False,
        }
    )
    tracker.write_json("completion_evidence.json", fallback)
    metrics_path = tracker.run_directory / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = _read_json_object(metrics_path, "mutable primary metrics")
        except BaseException as metrics_error:
            error.add_note(
                "invalid mutable metrics were replaced during claim demotion: "
                f"{type(metrics_error).__name__}: {metrics_error}"
            )
            metrics = {"schema_version": 1}
        metrics.update(
            completion_stage=None,
            study_outcome_eligible=False,
            valid_completion_claim=False,
        )
        try:
            tracker.write_json("metrics.json", metrics)
        except BaseException as metrics_error:
            error.add_note(
                "failed to replace ancillary metrics after completion claim demotion: "
                f"{type(metrics_error).__name__}: {metrics_error}"
            )
    try:
        tracker.write_text(
            "report.md",
            "# Failed real primary run\n\nNo completion stage is claimed. "
            f"Failure: `{type(error).__name__}: {error}`\n",
        )
    except BaseException as report_error:
        error.add_note(
            "failed to replace ancillary report after completion claim demotion: "
            f"{type(report_error).__name__}: {report_error}"
        )


def _finish_failed_run(
    tracker: RunTracker,
    error: BaseException,
    *,
    core_completion: Mapping[str, Any] | None,
) -> None:
    if tracker.finalized:
        return
    try:
        _demote_mutable_claim(tracker, core_completion=core_completion, error=error)
    except BaseException as demotion_error:
        error.add_note(
            "eligible completion claim could not be demoted, so the run was deliberately "
            "left unsealed and ineligible for downstream gates: "
            f"{type(demotion_error).__name__}: {demotion_error}"
        )
        return
    try:
        tracker.fail(error)
    except BaseException as finalize_error:
        error.add_note(
            "failed-run sealing also failed after completion-claim demotion: "
            f"{type(finalize_error).__name__}: {finalize_error}"
        )


def execute_primary_study(
    *,
    gate_evidence: PrimaryExecutionGateEvidence,
    project_root: str | Path,
    freeze_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    duplicate_audit_path: str | Path,
    pathology_encoder_audit_path: str | Path,
    frozen_primary_config_path: str | Path,
    frozen_confirmatory_config_path: str | Path,
    cache_paths: PanNukePrimaryCachePaths | None = None,
    cache_hashes: PanNukePrimaryHashExpectations | None = None,
    runs_root: str | Path | None = None,
    run_id: str | None = None,
    retry_of_run_id: str | None = None,
    resume_run_directory: str | Path | None = None,
    dependencies: PrimaryRunnerDependencies | None = None,
) -> dict[str, Any]:
    """Execute, seal, and post-verify the real primary study.

    Existing runs are never overwritten. Resume is deliberately unsupported until a
    frozen per-cell checkpoint protocol exists; a retry must name its predecessor and
    always receives a new run directory.
    """

    deps = dependencies or PrimaryRunnerDependencies()
    root = Path(project_root).resolve()
    freeze = _resolve(root, freeze_directory)
    dataset = _resolve(root, dataset_path)
    manifest = _resolve(root, manifest_path)
    duplicate_audit = _resolve(root, duplicate_audit_path)
    pathology_audit = _resolve(root, pathology_encoder_audit_path)
    primary_config_path = _resolve(root, frozen_primary_config_path)
    confirmatory_config_path = _resolve(root, frozen_confirmatory_config_path)

    # Mandatory live revalidation is the first operation capable of consulting study
    # inputs. In particular, no RunTracker or run directory exists before this returns.
    live_gate = deps.gate_validator(
        project_root=root,
        freeze_directory=freeze,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology_audit,
        frozen_primary_config_path=primary_config_path,
        frozen_confirmatory_config_path=confirmatory_config_path,
    )
    _validate_gate_equality(gate_evidence, live_gate)
    if resume_run_directory is not None:
        raise NotImplementedError(
            "primary resume is not implemented; use retry_of_run_id to create a new run"
        )

    primary_config = deps.config_loader(primary_config_path)
    confirmatory_config = deps.config_loader(confirmatory_config_path)
    primary_plan = deps.primary_plan_builder(primary_config)
    confirmatory_plan = deps.confirmatory_plan_builder(confirmatory_config)
    controls = deps.controls_builder(primary_config)
    _validate_frozen_plans(live_gate, primary_plan, confirmatory_plan, controls)

    resolved_cache_paths = _normalise_cache_paths(
        root,
        cache_paths
        or default_primary_cache_paths(
            project_root=root,
            freeze_directory=live_gate.base_freeze_directory,
            manifest_path=manifest,
            pathology_encoder_audit_path=pathology_audit,
        ),
    )
    if resolved_cache_paths.dataset_manifest_path != manifest:
        raise PrimaryStudyRunnerError("primary cache manifest path differs from gated manifest")
    if resolved_cache_paths.pathology_availability_audit_path != pathology_audit:
        raise PrimaryStudyRunnerError("primary cache pathology audit differs from gated audit")
    if (
        resolved_cache_paths.freeze_record_path
        != live_gate.base_freeze_directory / "freeze_evidence.json"
    ):
        raise PrimaryStudyRunnerError(
            "primary cache freeze record differs from the gated base freeze"
        )
    resolved_hashes = cache_hashes or derive_primary_cache_hashes(
        resolved_cache_paths,
        manifest_sha256=live_gate.manifest_sha256,
    )
    _verify_explicit_cache_hashes(
        resolved_cache_paths,
        resolved_hashes,
        manifest_sha256=live_gate.manifest_sha256,
    )
    prepared = deps.input_builder(
        primary_config,
        resolved_cache_paths,
        expected_config_sha256=live_gate.primary_config_semantic_sha256,
        expected_plan_semantic_sha256=controls.plan_sha256,
        project_root=root,
        expected_hashes=resolved_hashes,
    )
    if (
        prepared.plan != primary_plan
        or prepared.config_sha256 != primary_plan.config_sha256
        or prepared.plan_semantic_sha256 != controls.plan_sha256
    ):
        raise PrimaryStudyRunnerError("prepared PanNuke inputs differ from the frozen plan")

    run_root = _resolve(root, runs_root) if runs_root is not None else root / "artifacts" / "runs"
    if retry_of_run_id is not None:
        if Path(retry_of_run_id).name != retry_of_run_id or not retry_of_run_id.strip():
            raise ValueError("retry_of_run_id must be one safe non-empty run ID")
        previous = run_root / retry_of_run_id
        if not previous.is_dir():
            raise FileNotFoundError(f"retry predecessor does not exist: {previous}")

    # Adapter preflight may read large immutable caches. Re-run the complete gate and
    # the explicit cache checks at the final pre-tracker boundary so that a long load
    # cannot create a gate-to-execution TOCTOU window.
    final_gate = deps.gate_validator(
        project_root=root,
        freeze_directory=freeze,
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_path=duplicate_audit,
        pathology_encoder_audit_path=pathology_audit,
        frozen_primary_config_path=primary_config_path,
        frozen_confirmatory_config_path=confirmatory_config_path,
    )
    _validate_gate_equality(live_gate, final_gate)
    _verify_explicit_cache_hashes(
        resolved_cache_paths,
        resolved_hashes,
        manifest_sha256=final_gate.manifest_sha256,
    )

    tracker = deps.tracker_starter(
        experiment_name=_EXPERIMENT_NAME,
        config=primary_config,
        project_root=root,
        runs_root=run_root,
        run_id=run_id,
        environment={
            "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
            "study_outcome_eligible_only_after_post_seal_verification": True,
            "primary_gate": live_gate.as_dict(),
        },
        dataset_path=dataset,
        manifest_path=manifest,
        duplicate_audit_status=f"complete_sha256:{live_gate.duplicate_audit_sha256}",
    )
    core_completion: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    try:
        if tracker.source_tree.get("root_sha256") != final_gate.source_tree_root_sha256:
            raise PrimaryStudyRunnerError(
                "source tree changed between the final live gate and RunTracker capture"
            )
        tracker.write_json("primary_execution_gate.json", live_gate.as_dict())
        input_bindings_path = tracker.write_json(
            "primary_input_bindings.json",
            {
                "schema_version": 1,
                "cache_paths": _cache_paths_payload(resolved_cache_paths),
                "expected_hashes": asdict(resolved_hashes),
                "verified_hashes": dict(prepared.verified_hashes),
                "config_semantic_sha256": prepared.config_sha256,
                "plan_semantic_sha256": prepared.plan_semantic_sha256,
                "execution_controls_binding_sha256": controls.binding_sha256,
                "sample_order_sha256": prepared.sample_order_sha256,
                "partition_assignment_sha256": prepared.partition_assignment_sha256,
                "cache_provenance_by_representation": {
                    identifier: dict(provenance)
                    for identifier, provenance in sorted(
                        prepared.cache_provenance_by_representation.items()
                    )
                },
                "representation_availability": [
                    asdict(item) for item in prepared.representation_availability
                ],
                "retry_of_run_id": retry_of_run_id,
                "resume_policy": "unsupported; retry creates a new immutable run",
            },
        )
        tracker.write_provenance(
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            primary_gate=live_gate.as_dict(),
            primary_input_bindings_sha256=sha256_file(input_bindings_path),
            matrix_plan_sha256=controls.plan_sha256,
            execution_controls_binding_sha256=controls.binding_sha256,
            retry_of_run_id=retry_of_run_id,
            completion_stage_valid_only_after_post_seal_integrity=True,
        )
        artifacts = deps.matrix_executor(
            prepared.inputs,
            primary_plan,
            output_directory=tracker.run_directory,
            execution_controls=controls,
        )
        core_completion = _validate_core_artifacts(
            artifacts,
            run_directory=tracker.run_directory,
            plan=primary_plan,
            controls=controls,
        )
        statistics_artifacts = deps.statistics_aggregator(
            tracker.run_directory,
            controls,
        )
        statistics_verification = deps.statistics_verifier(
            tracker.run_directory,
            controls,
        )
        if (
            not statistics_verification.valid
            or Path(statistics_verification.output_directory).resolve() != tracker.run_directory
            or Path(statistics_artifacts.output_directory).resolve() != tracker.run_directory
            or Path(statistics_artifacts.manifest_path).resolve()
            != tracker.run_directory / "primary_statistics_manifest.json"
            or not Path(statistics_artifacts.manifest_path).is_file()
            or statistics_artifacts.manifest_sha256 != statistics_verification.manifest_sha256
            or statistics_verification.manifest_sha256
            != sha256_file(statistics_artifacts.manifest_path)
        ):
            raise PrimaryStudyRunnerError(
                "primary statistics aggregation failed strict filesystem verification"
            )
        readback = deps.filesystem_reader(primary_plan, tracker.run_directory)
        if (
            not readback.passed
            or Path(readback.run_directory).resolve() != tracker.run_directory
            or readback.reconciliation.as_dict() != artifacts.reconciliation.as_dict()
        ):
            raise PrimaryStudyRunnerError(
                "filesystem readback did not attest the exact matrix reconciliation"
            )
        raw_candidate = deps.completion_builder(
            plan=primary_plan,
            reconciliation=readback.reconciliation,
            artifact_scope=REAL_PRIMARY_ARTIFACT_SCOPE,
            study_outcome_eligible=True,
            gate_evidence=live_gate,
            filesystem_readback=readback,
        )
        candidate = _validate_completion_candidate(
            raw_candidate,
            plan=primary_plan,
            gate=live_gate,
            readback=readback,
        )
        for key in (
            "execution_controls_binding_sha256",
            "circularity_excluded_cell_count",
            "circularity_excluded_cell_ids",
            "primary_confirmatory_claims_require_exclusion_of_these_cells",
        ):
            if key in core_completion:
                candidate[key] = core_completion[key]
        candidate.update(
            {
                "run_id": tracker.run_id,
                "retry_of_run_id": retry_of_run_id,
                "primary_statistics_manifest_sha256": (statistics_verification.manifest_sha256),
                "primary_statistics_source_readback_root_sha256": (
                    statistics_verification.source_readback_root_sha256
                ),
                "post_seal_integrity_verification_required": True,
            }
        )
        tracker.write_json("core_completion_evidence.json", core_completion)
        completion_path = tracker.write_json("completion_evidence.json", candidate)
        metrics = {
            "schema_version": 1,
            "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
            "study_outcome_eligible": True,
            "completion_stage": _COMPLETION_STAGE,
            "run_id": tracker.run_id,
            "matrix_config_sha256": primary_plan.config_sha256,
            "matrix_plan_sha256": controls.plan_sha256,
            "execution_controls_binding_sha256": controls.binding_sha256,
            "planned_cell_count": len(primary_plan.cells),
            "required_cell_count": primary_plan.required_cell_count,
            "completed_required_cell_count": readback.completed_required_cell_count,
            "skipped_optional_cell_count": readback.skipped_optional_cell_count,
            "filesystem_readback_status": readback.status,
            "filesystem_readback_root_sha256": readback.readback_root_sha256,
            "completion_evidence_sha256": sha256_file(completion_path),
            "primary_statistics_manifest_sha256": (statistics_verification.manifest_sha256),
            "primary_statistics_comparison_count": statistics_verification.comparison_count,
            "post_seal_integrity_verification_required": True,
            "valid_completion_claim": "pending_post_seal_verification",
        }
        tracker.write_metrics(metrics)
        tracker.write_text("report.md", _report_text(metrics))
        tracker.log_event(
            "primary_completion_candidate_written",
            completion_stage=_COMPLETION_STAGE,
            completion_evidence_sha256=sha256_file(completion_path),
            filesystem_readback_root_sha256=readback.readback_root_sha256,
        )
        tracker.complete()
    except BaseException as exc:
        _finish_failed_run(
            tracker,
            exc,
            core_completion=core_completion,
        )
        raise

    integrity = deps.integrity_verifier(tracker.run_directory)
    if (
        not integrity.valid
        or not integrity.registry_record_present
        or integrity.run_id != tracker.run_id
        or not _valid_sha(integrity.expected_root_sha256)
    ):
        raise PrimaryStudyIntegrityError(
            "sealed primary candidate failed post-seal integrity verification; "
            f"no valid {_COMPLETION_STAGE} claim is returned: {integrity.errors}"
        )
    sealed_completion = _read_json_object(
        tracker.run_directory / "completion_evidence.json",
        "sealed primary completion evidence",
    )
    if candidate is None or sealed_completion != candidate:
        raise PrimaryStudyIntegrityError(
            "sealed completion evidence differs from the verified completion candidate"
        )
    return {
        "status": "completed",
        "completion_stage": _COMPLETION_STAGE,
        "study_outcome_eligible": True,
        "artifact_scope": REAL_PRIMARY_ARTIFACT_SCOPE,
        "run_id": tracker.run_id,
        "run_directory": str(tracker.run_directory),
        "artifact_root_sha256": integrity.expected_root_sha256,
        "registry_record_present": integrity.registry_record_present,
        "completion_evidence_path": str(tracker.run_directory / "completion_evidence.json"),
        "completion_evidence_sha256": sha256_file(
            tracker.run_directory / "completion_evidence.json"
        ),
        "reconciliation_path": str(tracker.run_directory / "reconciliation.json"),
        "metrics_path": str(tracker.run_directory / "metrics.json"),
        "report_path": str(tracker.run_directory / "report.md"),
        "planned_cell_count": len(primary_plan.cells),
        "completed_required_cell_count": primary_plan.required_cell_count,
        "retry_of_run_id": retry_of_run_id,
    }


__all__ = [
    "PrimaryRunnerDependencies",
    "PrimaryStudyIntegrityError",
    "PrimaryStudyRunnerError",
    "default_primary_cache_paths",
    "derive_primary_cache_hashes",
    "execute_primary_study",
]
