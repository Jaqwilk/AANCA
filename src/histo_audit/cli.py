"""Typer command line interface for the annotation-auditing workflow."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import typer

from histo_audit.workflows.original_confirmatory_technical_authority_publication_v1 import (
    original_confirmatory_technical_authority_app,
)

app = typer.Typer(
    name="histo-audit",
    help="Leakage-safe nucleus annotation-auditing research workflow.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
data_app = typer.Typer(help="Synthetic generation and gated real-data preparation.")
representations_app = typer.Typer(help="Representation extraction commands.")
experiment_app = typer.Typer(help="Tracked synthetic, pilot, and study experiments.")
preregistration_app = typer.Typer(help="Preregistration freeze and provenance commands.")
audit_app = typer.Typer(help="Exploratory original-label auditing commands.")
external_app = typer.Typer(help="External-validation package commands.")
report_app = typer.Typer(help="Machine-readable-artifact-backed reporting.")
demo_app = typer.Typer(help="Build, verify, and locally serve the static presentation MVP.")

# Both public PanNuke CLI paths must reproduce the immutable canonical validation
# evidence before doing any downstream work.  Keep these limits at the CLI boundary:
# lower-level validation APIs retain their intentionally bounded fixture-friendly
# defaults, while real CLI commands share one explicit scientific authority.
CANONICAL_PANNUKE_VALIDATION_MAX_SAMPLES_PER_FOLD = 100_000
CANONICAL_PANNUKE_VALIDATION_MAX_OVERLAY_PATCHES = 24

app.add_typer(data_app, name="data")
app.add_typer(representations_app, name="representations")
app.add_typer(experiment_app, name="experiment")
app.add_typer(preregistration_app, name="preregistration")
app.add_typer(audit_app, name="audit")
app.add_typer(external_app, name="external")
app.add_typer(report_app, name="report")
app.add_typer(demo_app, name="demo")
app.add_typer(
    original_confirmatory_technical_authority_app,
    name="original-confirmatory-technical-authority",
)


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _parse_explicit_utc_timestamp(value: str, *, role: str) -> datetime:
    """Parse an explicit reproducible UTC timestamp without accepting local time."""

    if value != value.strip() or not value.endswith("Z"):
        raise ValueError(f"{role} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{role} is not a valid ISO-8601 UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{role} must have UTC offset zero")
    return parsed.astimezone(UTC)


def _path_exists(path: Path) -> bool:
    """Return true for regular paths and broken symbolic links."""

    return os.path.lexists(path)


def _require_derived_destination_outside_raw(
    raw_root: Path,
    destination: Path,
    role: str,
    *,
    directory: bool = False,
) -> Path:
    """Resolve one derived destination and reject any overlap with immutable raw data."""

    raw = raw_root.resolve()
    fully_resolved = destination.resolve()
    resolved = (
        fully_resolved
        if directory
        else destination.expanduser().absolute().parent.resolve() / destination.name
    )
    inside_raw = fully_resolved == raw or raw in fully_resolved.parents
    contains_raw = directory and (fully_resolved == raw or fully_resolved in raw.parents)
    if inside_raw or contains_raw:
        relationship = "overlaps" if contains_raw else "is inside"
        raise ValueError(
            f"{role} {relationship} the immutable raw release: {resolved} (raw: {raw})"
        )
    return resolved


def _require_distinct_destinations(destinations: Sequence[tuple[str, Path]]) -> None:
    """Reject resolved aliases among derived files/directories before any write."""

    by_path: dict[Path, str] = {}
    for role, destination in destinations:
        resolved = destination.resolve()
        previous = by_path.get(resolved)
        if previous is not None:
            raise ValueError(
                f"derived destinations alias after resolution: {previous} and {role}: {resolved}"
            )
        by_path[resolved] = role


def _require_disjoint_file_destinations(destinations: Sequence[tuple[str, Path]]) -> None:
    """Reject file targets that collide through equality or path ancestry."""

    resolved = tuple((role, destination.resolve()) for role, destination in destinations)
    for index, (left_role, left) in enumerate(resolved):
        for right_role, right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "derived file destinations collide after resolution: "
                    f"{left_role} and {right_role}: {left}, {right}"
                )


def _require_disjoint_directories(left: tuple[str, Path], right: tuple[str, Path]) -> None:
    left_role, left_path = left
    right_role, right_path = right
    resolved_left = left_path.resolve()
    resolved_right = right_path.resolve()
    if (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    ):
        raise ValueError(
            "derived directory destinations overlap: "
            f"{left_role} and {right_role}: {resolved_left}, {resolved_right}"
        )


def _require_destination_suffix(destination: Path, suffix: str, role: str) -> None:
    if destination.suffix.lower() != suffix:
        raise ValueError(f"{role} must use the {suffix or '<no suffix>'} suffix: {destination}")


def _require_directory_disjoint_from_files(
    directory: tuple[str, Path], files: Sequence[tuple[str, Path]]
) -> None:
    directory_role, directory_path = directory
    resolved_directory = directory_path.resolve()
    for file_role, file_path in files:
        resolved_file = file_path.resolve()
        if (
            resolved_directory == resolved_file
            or resolved_directory in resolved_file.parents
            or resolved_file in resolved_directory.parents
        ):
            raise ValueError(
                f"derived directory/file destinations overlap: {directory_role} and "
                f"{file_role}: {resolved_directory}, {resolved_file}"
            )


def _promote_staged_artifact(source: Path, destination: Path) -> list[Any]:
    """Promote one staged file/directory without permitting an overwrite."""

    from histo_audit.pannuke.publication import (
        publish_file_no_overwrite,
        publish_flat_directory_no_overwrite,
    )

    if source.is_dir():
        return publish_flat_directory_no_overwrite(
            source,
            destination,
            success_marker_name="artifact_manifest.json",
        )
    return [publish_file_no_overwrite(source, destination)]


def _create_missing_parents(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not _path_exists(current):
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(f"validation artifact parent is not a directory: {current}")
    created: list[Path] = []
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)
    return created


def _publish_immutable_validation_artifacts(
    *,
    ancillary_files: Sequence[tuple[Path, Path]],
    staged_qc_bundle: Path,
    final_qc_bundle: Path,
    success_marker: tuple[Path, Path],
    raw_inventory_verifier: Callable[[], object],
) -> str:
    """Publish the complete base/QC set or leave every canonical byte unchanged."""

    staged_qc = staged_qc_bundle.resolve()
    final_qc = final_qc_bundle.parent.resolve() / final_qc_bundle.name
    staged_marker, final_marker = success_marker
    staged_pairs = tuple(
        (source.resolve(), target.parent.resolve() / target.name)
        for source, target in ancillary_files
    )
    staged_marker = staged_marker.resolve()
    final_marker = final_marker.parent.resolve() / final_marker.name
    if not staged_qc.is_dir():
        raise FileNotFoundError(f"staged mask-QC bundle is missing: {staged_qc}")
    qc_sources = tuple(sorted(staged_qc.iterdir(), key=lambda value: value.name))
    if not qc_sources or any(not value.is_file() for value in qc_sources):
        raise ValueError("staged mask-QC bundle must be a non-empty flat file set")
    for source, _destination in (*staged_pairs, (staged_marker, final_marker)):
        if not source.is_file():
            raise FileNotFoundError(f"staged validation artifact is missing: {source}")

    base_destinations = (*(target for _source, target in staged_pairs), final_marker)

    def require_exact_final_set() -> None:
        from histo_audit.pannuke import validate_mask_qc_report_bundle

        base_present = tuple(_path_exists(path) for path in base_destinations)
        if not all(base_present) or any(
            path.is_symlink() or not path.is_file() for path in base_destinations
        ):
            raise FileExistsError(
                "existing PanNuke validation artifact set is partial; refusing mutation"
            )
        if not _path_exists(final_qc) or not final_qc.is_dir() or final_qc.is_symlink():
            raise FileExistsError(
                "existing PanNuke validation artifact set is partial; refusing mutation"
            )
        final_qc_names = {value.name for value in final_qc.iterdir()}
        staged_qc_names = {value.name for value in qc_sources}
        if final_qc_names != staged_qc_names or any(
            (final_qc / name).is_symlink() or not (final_qc / name).is_file()
            for name in final_qc_names
        ):
            raise FileExistsError(
                "existing PanNuke mask-QC bundle is partial or has unexpected files"
            )
        comparisons = [(source, destination) for source, destination in staged_pairs] + [
            (staged_marker, final_marker)
        ]
        comparisons.extend((source, final_qc / source.name) for source in qc_sources)
        differing = [
            str(destination)
            for source, destination in comparisons
            if source.read_bytes() != destination.read_bytes()
        ]
        if differing:
            raise FileExistsError(
                "existing immutable PanNuke validation artifacts differ: " + ", ".join(differing)
            )
        validate_mask_qc_report_bundle(final_qc)

    raw_inventory_verifier()
    base_present = tuple(_path_exists(path) for path in base_destinations)
    qc_present = _path_exists(final_qc)
    any_present = any(base_present) or qc_present
    if any_present:
        require_exact_final_set()
        raw_inventory_verifier()
        require_exact_final_set()
        return "idempotent"

    for parent in {
        *(destination.parent for _source, destination in staged_pairs),
        final_qc.parent,
        final_marker.parent,
    }:
        _create_missing_parents(parent)
    from histo_audit.pannuke.publication import rollback_owned_publications

    published: list[Any] = []
    try:
        for source, destination in staged_pairs:
            published.extend(_promote_staged_artifact(source, destination))
        published.extend(_promote_staged_artifact(staged_qc, final_qc))
        # This JSON is the canonical success marker and is always published last.
        published.extend(_promote_staged_artifact(staged_marker, final_marker))
        raw_inventory_verifier()
        require_exact_final_set()
    except BaseException as publication_error:
        try:
            rollback_owned_publications(published)
        except RuntimeError:
            raise RuntimeError(
                "validation publication failed and ownership-safe rollback was incomplete"
            ) from publication_error
        raise
    return "published"


def _failure(message: str, *, exit_code: int = 1) -> NoReturn:
    typer.echo(f"ERROR: {message}", err=True)
    raise typer.Exit(code=exit_code)


def _gate(stage: str, reason: str, *, next_command: str | None = None) -> NoReturn:
    typer.echo(f"GATED [{stage}]: {reason}", err=True)
    if next_command:
        typer.echo(f"Next: {next_command}", err=True)
    raise typer.Exit(code=2)


def _tracked_failure_identity(exc: BaseException) -> tuple[str | None, str | None]:
    """Read an explicitly attached tracked-run identity without guessing from disk."""

    run_id: str | None = None
    run_directory: str | None = None
    for name in ("tracked_failure_run_id", "failure_run_id", "run_id"):
        value = getattr(exc, name, None)
        if isinstance(value, str) and value.strip():
            run_id = value
            break
    for name in (
        "tracked_failure_run_directory",
        "failure_run_directory",
        "run_directory",
    ):
        value = getattr(exc, name, None)
        if isinstance(value, (str, Path)) and str(value).strip():
            run_directory = str(value)
            break
    return run_id, run_directory


def _emit_primary_recovery_error(
    exc: BaseException,
    *,
    status: str,
    authority_directory: Path,
    source_run_id: str | None,
    exit_code: int,
) -> NoReturn:
    """Emit a stable one-shot recovery failure without an eligibility claim."""

    tracked_run_id, tracked_run_directory = _tracked_failure_identity(exc)
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "primary_orphan_recovery",
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "authority_directory": str(authority_directory),
                "source_run_id": source_run_id,
                "tracked_failure_run_id": tracked_run_id,
                "tracked_failure_run_directory": tracked_run_directory,
                "completion_stage": None,
                "training_invoked": False,
                "fallback_invoked": False,
                "automatic_retry_allowed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=exit_code)


def _resource_bounded_error_text(exc: BaseException) -> str:
    """Return an error description that cannot echo a forbidden stage claim."""

    message = f"{type(exc).__name__}: {exc}"
    return cast(str, _resource_bounded_safe_json_value(message))


def _resource_bounded_safe_json_value(value: Any) -> Any:
    """Recursively redact claim text that this CLI path must never emit."""

    if isinstance(value, str):
        replacements = (
            ("CONFIRMATORY_COMPLETE", "[forbidden-stage-claim]"),
            ("study_outcome_eligible=true", "study_outcome_eligible=[forbidden]"),
            ('"study_outcome_eligible": true', '"study_outcome_eligible": "[forbidden]"'),
            ("'study_outcome_eligible': True", "'study_outcome_eligible': '[forbidden]'"),
        )
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, Path):
        return _resource_bounded_safe_json_value(str(value))
    if isinstance(value, dict):
        return {key: _resource_bounded_safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resource_bounded_safe_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _resource_bounded_safe_json_value(str(value))


def _emit_resource_bounded_error(
    exc: BaseException,
    *,
    status: str,
    preflight_only: bool,
    run_mode: str,
    retry_of_run_id: str | None,
    exit_code: int,
) -> NoReturn:
    """Emit stable, permanently non-claiming resource-sensitivity failure JSON."""

    tracked_run_id, tracked_run_directory = _tracked_failure_identity(exc)
    typer.echo(
        json.dumps(
            _resource_bounded_safe_json_value(
                {
                    "schema_version": 1,
                    "workflow": "resource_bounded_sensitivity",
                    "status": status,
                    "error": _resource_bounded_error_text(exc),
                    "preflight_only": preflight_only,
                    "run_mode": run_mode,
                    "retry_of_run_id": retry_of_run_id,
                    "tracked_failure_run_id": tracked_run_id,
                    "tracked_failure_run_directory": tracked_run_directory,
                    "completion_stage": None,
                    "study_outcome_eligible": False,
                    "analysis_disposition": "amended_or_exploratory",
                    "original_confirmatory_claim_allowed": False,
                    "m9_unlock_allowed": False,
                    "automatic_retry_allowed": False,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=exit_code)


def _resource_bounded_success_payload(
    result: object,
    *,
    preflight_only: bool,
    run_mode: str,
    retry_of_run_id: str | None,
) -> dict[str, Any]:
    """Expose only an allowlist of non-claiming runner/preflight evidence."""

    if not isinstance(result, dict):
        raise TypeError("resource-bounded executor must return a JSON mapping")
    if (
        result.get("completion_stage") is not None
        or result.get("study_outcome_eligible") is not False
        or result.get("analysis_disposition") != "amended_or_exploratory"
    ):
        raise ValueError("resource-bounded executor returned a forbidden scientific-stage claim")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "resource_bounded_sensitivity",
        "status": "preflight_passed" if preflight_only else "executor_returned",
        "preflight_only": preflight_only,
        "run_mode": run_mode,
        "retry_of_run_id": retry_of_run_id,
        "completion_stage": None,
        "study_outcome_eligible": False,
        "analysis_disposition": "amended_or_exploratory",
        "original_confirmatory_claim_allowed": False,
        "m9_unlock_allowed": False,
        "automatic_retry_allowed": False,
    }
    allowed_result_fields = (
        "run_id",
        "run_directory",
        "artifact_root_sha256",
        "registry_record_present",
        "post_seal_filesystem_readback_status",
        "resource_capacity_policy_sha256",
        "resource_compute_evidence_sha256",
        "resource_resume_evidence_sha256",
        "resource_predecessor_qualification_sha256",
        "completion_evidence_path",
        "completion_evidence_sha256",
        "planned_cell_count",
        "completed_required_cell_count",
        "tracker_created",
        "scientific_run_created",
        "checkpoint_copy_performed",
        "predecessor_read_performed",
        "predecessor_qualified",
        "predecessor_qualification_sha256",
        "predecessor_directory",
        "capacity_checks",
        "compute_checks",
        "checkpoint_allowlist_count",
        "reusable_checkpoint_count",
        "missing_checkpoint_count",
        "resource_gate_sha256",
        "historical_primary_run_id",
        "resource_authorization_sha256",
        "data_and_cache_inputs_verified",
        "lifecycle_readiness_verified",
        "dual_authority_gate_verified",
        "full_pc_gate_validation_count",
        "stage_attestation_record_count",
        "stage_disposition_record_count",
    )
    evidence = {field: result[field] for field in allowed_result_fields if field in result}
    if evidence:
        payload["evidence"] = evidence
    return payload


def _load_optional_study_executor(
    module_name: str, function_name: str
) -> Callable[..., Any] | None:
    """Load a real-study executor without importing it before its gate passes."""

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    executor = getattr(module, function_name, None)
    if not callable(executor):
        return None
    return cast(Callable[..., Any], executor)


def _require_pannuke(project_root: Path, explicit: Path | None = None) -> Path:
    from histo_audit.pannuke import (
        PanNukeDiscoveryError,
        PanNukeNotFoundError,
        locate_pannuke_root,
    )

    try:
        located = locate_pannuke_root(explicit_path=explicit, project_root=project_root)
        has_release_payload = any(
            candidate.is_file()
            and (
                candidate.suffix.lower() == ".npy"
                or candidate.name.lower().endswith(
                    (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar")
                )
            )
            for candidate in located.rglob("*")
        )
        if not has_release_payload:
            _gate(
                "REAL_DATA_UNAVAILABLE",
                f"{located} contains no PanNuke .npy arrays or release archives. "
                "Follow DATASET_SETUP.md; do not treat the placeholder directory as data.",
                next_command="python -m histo_audit data validate-pannuke --root <verified-path>",
            )
        return located
    except PanNukeNotFoundError:
        _gate(
            "REAL_DATA_UNAVAILABLE",
            "verified PanNuke files were not found; no real-data command was executed. "
            "Follow DATASET_SETUP.md and set PANNUKE_ROOT after satisfying licence terms.",
            next_command="python -m histo_audit data validate-pannuke --root <verified-path>",
        )
    except PanNukeDiscoveryError as exc:
        _failure(f"PanNuke discovery failed: {exc}")


@app.command("doctor")
def doctor_command(
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Project root used for disk, dataset, Git, and write-access evidence.",
            file_okay=False,
            dir_okay=True,
            exists=True,
        ),
    ] = Path("."),
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="JSON output path; relative paths use project root."),
    ] = None,
) -> None:
    """Print and atomically save reproducible environment/hardware evidence."""

    from histo_audit.doctor import format_doctor_report, run_doctor

    root = project_root.resolve()
    try:
        report, destination = run_doctor(project_root=root, output_path=output)
    except Exception as exc:
        _failure(f"doctor failed: {type(exc).__name__}: {exc}")
    typer.echo(format_doctor_report(report))
    typer.echo(f"Saved doctor evidence: {destination}")


@demo_app.command("build")
def build_mvp_demo_command(
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False, dir_okay=True, exists=True),
    ] = Path("."),
    run_directory: Annotated[
        Path,
        typer.Option("--run-dir", file_okay=False, help="Accepted sealed primary run."),
    ] = Path("artifacts/runs/20260727T133947.089370Z_pannuke_primary_orphan_recovery"),
    qc_bundle_directory: Annotated[
        Path,
        typer.Option("--qc-bundle", file_okay=False, help="Checksum-bound PanNuke QC bundle."),
    ] = Path("reports/pannuke_qc"),
    output_directory: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False, help="New static-demo directory."),
    ] = Path("artifacts/mvp_demo"),
) -> None:
    """Build a read-only static demo from the accepted primary and PanNuke QC."""

    from histo_audit.mvp_demo import build_mvp_presentation

    try:
        artifacts = build_mvp_presentation(
            project_root=project_root,
            run_directory=run_directory,
            qc_bundle_directory=qc_bundle_directory,
            output_directory=output_directory,
        )
    except Exception as exc:
        _failure(f"MVP demo build failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "built_and_verified",
                "presentation_status": "DEMO_COMPLETE",
                "scientific_status": "PRIMARY_STUDY_COMPLETE",
                "output_directory": str(artifacts.output_directory),
                "html": str(artifacts.html_path),
                "evidence": str(artifacts.evidence_path),
                "manifest": str(artifacts.manifest_path),
                "manifest_root_sha256": artifacts.manifest_root_sha256,
            },
            indent=2,
        )
    )


@demo_app.command("verify")
def verify_mvp_demo_command(
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            exists=True,
            help="Existing static-demo directory.",
        ),
    ] = Path("artifacts/mvp_demo"),
) -> None:
    """Verify the closed allowlist and checksums of an existing MVP package."""

    from histo_audit.mvp_demo import verify_mvp_presentation

    try:
        result = verify_mvp_presentation(output_directory)
    except Exception as exc:
        _failure(f"MVP demo verification failed: {type(exc).__name__}: {exc}")
    typer.echo(json.dumps(result, indent=2))


@demo_app.command("serve")
def serve_mvp_demo_command(
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            exists=True,
            help="Existing static-demo directory; checksums are verified before serving.",
        ),
    ] = Path("artifacts/mvp_demo"),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Bind address. The loopback default keeps the presentation local.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            min=0,
            max=65_535,
            help="TCP port; use 0 to select an available port automatically.",
        ),
    ] = 8000,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the verified presentation in a browser."),
    ] = True,
) -> None:
    """Verify, serve, and optionally open the checked-in presentation."""

    from histo_audit.mvp_demo import create_mvp_http_server

    try:
        server, verification = create_mvp_http_server(
            output_directory,
            host=host,
            port=port,
        )
    except Exception as exc:
        _failure(f"MVP demo server failed: {type(exc).__name__}: {exc}")

    bound_port = int(server.server_address[1])
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{bound_port}/"
    typer.echo(
        json.dumps(
            {
                "status": "verified_and_serving",
                "presentation_status": verification["presentation_status"],
                "scientific_status": verification["scientific_status"],
                "manifest_root_sha256": verification["manifest_root_sha256"],
                "url": url,
            },
            indent=2,
        )
    )
    if open_browser:
        import webbrowser

        try:
            if not webbrowser.open(url, new=2):
                typer.echo(f"Browser launch was not confirmed; open {url} manually.", err=True)
        except Exception as exc:
            typer.echo(
                f"Browser launch failed ({type(exc).__name__}); open {url} manually.",
                err=True,
            )
    typer.echo("Press Ctrl+C to stop the local presentation server.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        typer.echo("\nPresentation server stopped.")
    finally:
        server.server_close()


@data_app.command("generate-synthetic")
def generate_synthetic_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/smoke.yaml"),
    output_directory: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Destination; defaults to a config-hash directory."),
    ] = None,
) -> None:
    """Generate deterministic five-class grouped software-validation data."""

    import numpy as np

    from histo_audit.config import config_sha256, load_config
    from histo_audit.data import generate_synthetic_dataset
    from histo_audit.data.synthetic import (
        SYNTHETIC_GENERATOR_SCHEMA_VERSION,
        synthetic_generator_code_sha256,
    )
    from histo_audit.utils.run_tracking import atomic_write_json

    root = project_root.resolve()
    try:
        config_file = _resolve_from_root(root, config_path)
        config = load_config(config_file)
        digest = config_sha256(config)
        generator_code_sha256 = synthetic_generator_code_sha256()
        definition_digest = hashlib.sha256(
            json.dumps(
                {
                    "configuration_sha256": digest,
                    "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
                    "generator_code_sha256": generator_code_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        data_config = config.get("data", {})
        seed_config = config.get("seed", {})
        if not isinstance(data_config, dict) or not isinstance(seed_config, dict):
            raise ValueError("smoke configuration data and seed entries must be mappings")
        destination = (
            _resolve_from_root(root, output_directory)
            if output_directory is not None
            else root / "data" / "synthetic" / definition_digest[:12]
        )
        arrays_path = destination / "dataset.npz"
        manifest_path = destination / "manifest.json"
        if arrays_path.exists() or manifest_path.exists():
            raise FileExistsError(
                f"synthetic output already exists and will not be overwritten: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        dataset = generate_synthetic_dataset(
            n_groups=int(data_config.get("groups", 60)),
            instances_per_group=int(data_config.get("samples_per_group", 5)),
            patch_size=int(data_config.get("image_size", 64)),
            seed=int(seed_config.get("dataset", 101)),
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".dataset.", suffix=".tmp", dir=destination
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    images=dataset.images,
                    target_masks=dataset.target_masks,
                    audit_features=dataset.audit_features,
                    corruption_features=dataset.corruption_features,
                    pre_corruption_labels=dataset.pre_corruption_labels,
                    observed_labels=dataset.observed_labels,
                    group_ids=dataset.group_ids,
                    sample_ids=dataset.sample_ids,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, arrays_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        atomic_write_json(manifest_path, [record.as_dict() for record in dataset.records])
        atomic_write_json(
            destination / "generation.json",
            {
                "software_validation_only": True,
                "configuration_path": str(config_file),
                "configuration_sha256": digest,
                "dataset_definition_sha256": definition_digest,
                "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
                "generator_code_sha256": generator_code_sha256,
                "n_samples": len(dataset.records),
                "n_groups": len(set(dataset.group_ids.tolist())),
                "class_names": list(dataset.class_names),
            },
        )
    except Exception as exc:
        _failure(f"synthetic generation failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "generated",
                "software_validation_only": True,
                "output_directory": str(destination.resolve()),
                "configuration_sha256": digest,
                "dataset_definition_sha256": definition_digest,
            },
            indent=2,
        )
    )


@data_app.command("verify-pannuke-acquisition")
def verify_pannuke_acquisition_command(
    verification_timestamp_utc: Annotated[
        str,
        typer.Option(
            "--verification-timestamp-utc",
            help="Explicit UTC evidence timestamp (ISO-8601 ending in Z).",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--root", "--data-root")] = None,
    manifest_output: Annotated[
        Path,
        typer.Option(
            "--manifest-output",
            help="Acquisition manifest path; relative paths use project root.",
        ),
    ] = Path("data/manifests/pannuke_acquisition.json"),
    report_output: Annotated[
        Path,
        typer.Option(
            "--report-output",
            help="Bound verification report path; relative paths use project root.",
        ),
    ] = Path("reports/pannuke_acquisition_verification.json"),
    expected_previous_manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-previous-manifest-sha256",
            help="Required compare-and-swap hash when intentionally replacing the manifest.",
        ),
    ] = None,
    expected_previous_report_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-previous-report-sha256",
            help="Required compare-and-swap hash when intentionally replacing the report.",
        ),
    ] = None,
) -> None:
    """Verify the already-local immutable PanNuke release; never download or extract it."""

    from histo_audit.pannuke import (
        PanNukeAcquisitionError,
        build_pannuke_acquisition_manifest,
        sha256_file,
        write_acquisition_artifact_bundle,
    )

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    manifest_path = _resolve_from_root(root, manifest_output)
    report_path = _resolve_from_root(root, report_output)
    try:
        for role, destination in (
            ("acquisition manifest", manifest_path),
            ("acquisition report", report_path),
        ):
            destination.relative_to(root)
            _require_derived_destination_outside_raw(located, destination, role)
            _require_destination_suffix(destination, ".json", role)
        _require_distinct_destinations(
            (
                ("acquisition manifest", manifest_path),
                ("acquisition report", report_path),
            )
        )
        manifest = build_pannuke_acquisition_manifest(
            root,
            located,
            verification_timestamp_utc=verification_timestamp_utc,
        )
        persisted_manifest, persisted_report = write_acquisition_artifact_bundle(
            manifest_path,
            report_path,
            manifest,
            project_root=root,
            expected_previous_manifest_sha256=expected_previous_manifest_sha256,
            expected_previous_report_sha256=expected_previous_report_sha256,
        )
    except (PanNukeAcquisitionError, FileExistsError, OSError, ValueError) as exc:
        _failure(f"PanNuke acquisition verification failed: {type(exc).__name__}: {exc}")
    archives = manifest["archives"]
    if not isinstance(archives, list):
        _failure("PanNuke acquisition verification returned an invalid archive inventory")
    typer.echo(
        json.dumps(
            {
                "status": "passed",
                "scope": "acquisition_provenance_only",
                "source_root": str(located),
                "archive_count": len(archives),
                "archives": [
                    {
                        "fold": value["fold"],
                        "size_bytes": value["size_bytes"],
                        "sha256": value["sha256"],
                        "zip_crc_status": value["zip_crc_status"],
                        "path_safety_status": value["path_safety_status"],
                    }
                    for value in archives
                ],
                "extracted_npy_count": len(manifest["extracted_npy_inventory"]),
                "extracted_document_count": len(manifest["extracted_document_inventory"]),
                "raw_release_read_only_verification": manifest[
                    "raw_release_read_only_verification"
                ],
                "manifest": str(persisted_manifest.resolve()),
                "manifest_sha256": sha256_file(persisted_manifest),
                "report": str(persisted_report.resolve()),
                "report_sha256": sha256_file(persisted_report),
                "download_performed": False,
                "extraction_performed": False,
                "scientific_stage_advanced": False,
            },
            indent=2,
        )
    )


def _validation_command_destinations(
    project_root: Path, output_directory: Path | None
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return the output directory, four base files, and immutable QC directory."""

    output = (
        _resolve_from_root(project_root, output_directory)
        if output_directory is not None
        else project_root / "reports"
    )
    json_path = output / "pannuke_validation.json"
    markdown_path = output / "pannuke_validation.md"
    overlay_path = (
        output / "pannuke_overlay_grid.png"
        if output_directory is not None
        else project_root / "artifacts" / "figures" / "pannuke_overlay_grid.png"
    )
    inventory_path = (
        output / "raw_files_sha256.csv"
        if output_directory is not None
        else project_root / "data" / "manifests" / "raw_files_sha256.csv"
    )
    return (
        output,
        json_path,
        markdown_path,
        overlay_path,
        inventory_path,
        output / "pannuke_qc",
    )


def _with_validation_command_lock(function: Callable[..., None]) -> Callable[..., None]:
    """Serialize the complete CLI base/QC build across every final output target."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        project_root = cast(Path, bound.arguments["project_root"])
        data_root = cast(Path | None, bound.arguments["data_root"])
        output_directory = cast(Path | None, bound.arguments["output_directory"])
        root = project_root.resolve()
        _require_pannuke(root, data_root)
        destinations = _validation_command_destinations(root, output_directory)
        from histo_audit.pannuke.publication import ExclusiveBundlePublicationLock

        with ExclusiveBundlePublicationLock(destinations, role="PanNuke CLI validation"):
            function(*args, **kwargs)

    return wrapped


@data_app.command("validate-pannuke")
@_with_validation_command_lock
def validate_pannuke_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--root", "--data-root")] = None,
    output_directory: Annotated[Path | None, typer.Option("--output-dir")] = None,
    max_samples_per_fold: Annotated[int, typer.Option(min=1)] = (
        CANONICAL_PANNUKE_VALIDATION_MAX_SAMPLES_PER_FOLD
    ),
    max_overlay_patches: Annotated[int, typer.Option(min=1)] = (
        CANONICAL_PANNUKE_VALIDATION_MAX_OVERLAY_PATCHES
    ),
) -> None:
    """Discover, hash, semantically validate, and report local PanNuke files."""

    from histo_audit.pannuke import (
        PanNukeError,
        validate_mask_qc_report_bundle,
        validate_pannuke,
        verify_raw_inventory_unchanged,
        write_mask_qc_report_bundle,
    )

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    (
        output,
        final_json_path,
        final_markdown_path,
        final_overlay_path,
        final_inventory_path,
        final_qc_bundle,
    ) = _validation_command_destinations(root, output_directory)
    staging_parent = root / "artifacts" / ".pannuke_validation_staging"
    staging_root: Path | None = None
    staging_parent_created = False
    final_files = (
        ("validation JSON", final_json_path),
        ("validation Markdown", final_markdown_path),
        ("validation overlay", final_overlay_path),
        ("raw inventory CSV", final_inventory_path),
    )
    try:
        output = _require_derived_destination_outside_raw(
            located, output, "validation output directory", directory=True
        )
        final_json_path = _require_derived_destination_outside_raw(
            located, final_json_path, "validation JSON"
        )
        final_markdown_path = _require_derived_destination_outside_raw(
            located, final_markdown_path, "validation Markdown"
        )
        final_overlay_path = _require_derived_destination_outside_raw(
            located, final_overlay_path, "validation overlay"
        )
        final_inventory_path = _require_derived_destination_outside_raw(
            located, final_inventory_path, "raw inventory CSV"
        )
        final_qc_bundle = _require_derived_destination_outside_raw(
            located, final_qc_bundle, "mask-QC bundle", directory=True
        )
        staging_parent = _require_derived_destination_outside_raw(
            located, staging_parent, "validation staging directory", directory=True
        )
        final_files = (
            ("validation JSON", final_json_path),
            ("validation Markdown", final_markdown_path),
            ("validation overlay", final_overlay_path),
            ("raw inventory CSV", final_inventory_path),
        )
        _require_distinct_destinations((*final_files, ("mask-QC bundle", final_qc_bundle)))
        _require_disjoint_file_destinations(final_files)
        _require_directory_disjoint_from_files(("mask-QC bundle", final_qc_bundle), final_files)
        _require_directory_disjoint_from_files(
            ("validation staging directory", staging_parent), final_files
        )
        _require_disjoint_directories(
            ("mask-QC bundle", final_qc_bundle),
            ("validation staging directory", staging_parent),
        )
        _require_destination_suffix(final_json_path, ".json", "validation JSON")
        _require_destination_suffix(final_markdown_path, ".md", "validation Markdown")
        _require_destination_suffix(final_overlay_path, ".png", "validation overlay")
        _require_destination_suffix(final_inventory_path, ".csv", "raw inventory CSV")
        _require_destination_suffix(final_qc_bundle, "", "mask-QC bundle")

        if not staging_parent.exists():
            staging_parent.mkdir(parents=True)
            staging_parent_created = True
        staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
        staged_output = staging_root / "base"
        staged_overlay_path = staging_root / "pannuke_overlay_grid.png"
        staged_inventory_path = staging_root / "raw_files_sha256.csv"
        artifacts = validate_pannuke(
            located,
            staged_output,
            max_samples_per_fold=max_samples_per_fold,
            max_overlay_patches=max_overlay_patches,
            raw_inventory_csv_path=staged_inventory_path,
            overlay_path=staged_overlay_path,
        )
        qc_artifacts = write_mask_qc_report_bundle(
            artifacts.result,
            staging_root / "pannuke_qc",
            max_overlay_patches=max_overlay_patches,
        )
        qc_artifacts = validate_mask_qc_report_bundle(qc_artifacts.bundle_dir)
        global_qc = artifacts.result.global_mask_qc
        qc_policy = artifacts.result.qc_policy
        if not artifacts.result.release_complete:
            raise ValueError("validation did not cover the complete expected PanNuke release")
        if any(
            fold.validation_scope != "full_semantic_scan"
            or fold.full_scan_patch_count != fold.n_patches
            for fold in artifacts.result.fold_validation
        ):
            raise ValueError("validation result does not prove a full semantic scan of every fold")
        if qc_artifacts.patch_row_count != global_qc.patch_count:
            raise ValueError("mask-QC patch CSV does not cover every validated source patch")
        if qc_artifacts.instance_row_count != global_qc.affected_instance_count:
            raise ValueError("mask-QC instance CSV does not cover every affected instance")
        if (
            qc_policy.source_masks_modified
            or not qc_policy.no_class_arbitration
            or qc_policy.supplied_background_is_exact_complement_required
            or qc_policy.release_annotation_anomalies_are_fatal
            or not qc_policy.structural_invalidity_is_fatal
            or not qc_policy.applies_identically_to_primary_and_confirmatory
        ):
            raise ValueError("mask-QC policy differs from the fixed anomaly-safe M5 policy")
        # Re-hash every raw file after all report rendering and immediately before publish.
        verify_raw_inventory_unchanged(artifacts.result)
        publication_status = _publish_immutable_validation_artifacts(
            ancillary_files=(
                (artifacts.markdown_path, final_markdown_path),
                (artifacts.overlay_path, final_overlay_path),
                (artifacts.raw_inventory_csv_path, final_inventory_path),
            ),
            staged_qc_bundle=qc_artifacts.bundle_dir,
            final_qc_bundle=final_qc_bundle,
            success_marker=(artifacts.json_path, final_json_path),
            raw_inventory_verifier=lambda: verify_raw_inventory_unchanged(artifacts.result),
        )
    except (PanNukeError, OSError, ValueError, RuntimeError, AttributeError, TypeError) as exc:
        _failure(f"PanNuke validation failed: {type(exc).__name__}: {exc}")
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        if staging_parent_created:
            with suppress(OSError):
                staging_parent.rmdir()
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "validation_scope": "full_semantic_scan",
                "source_root": str(artifacts.result.root),
                "fold_count": len(artifacts.result.folds),
                "patch_count": global_qc.patch_count,
                "raw_file_count": len(artifacts.result.inventory),
                "cross_class_overlap_pixel_count": global_qc.cross_class_overlap_pixel_count,
                "cross_class_overlap_patch_count": global_qc.cross_class_overlap_patch_count,
                "void_pixel_count": global_qc.void_pixel_count,
                "void_patch_count": global_qc.void_patch_count,
                "positive_and_background_pixel_count": (
                    global_qc.positive_and_background_pixel_count
                ),
                "positive_and_background_patch_count": (
                    global_qc.positive_and_background_patch_count
                ),
                "affected_patch_count": global_qc.anomaly_union_patch_count,
                "normal_patch_count": global_qc.normal_patch_count,
                "affected_instance_count": global_qc.affected_instance_count,
                "overlap_touching_instance_count": global_qc.overlap_touching_instance_count,
                "analysis_excluded_instance_count": global_qc.overlap_touching_instance_count,
                "primary_excluded_instance_count": global_qc.overlap_touching_instance_count,
                "confirmatory_excluded_instance_count": (global_qc.overlap_touching_instance_count),
                "analysis_exclusion_reason": qc_policy.analysis_instance_exclusion_reason,
                "applies_identically_to_primary_and_confirmatory": (
                    qc_policy.applies_identically_to_primary_and_confirmatory
                ),
                "no_class_arbitration": qc_policy.no_class_arbitration,
                "source_masks_modified": qc_policy.source_masks_modified,
                "publication": publication_status,
                "json": str(final_json_path),
                "markdown": str(final_markdown_path),
                "overlay": str(final_overlay_path),
                "raw_inventory_csv": str(final_inventory_path),
                "qc_bundle": str(final_qc_bundle),
                "qc_json": str(final_qc_bundle / "pannuke_mask_qc.json"),
                "qc_patch_csv": str(final_qc_bundle / "pannuke_mask_qc_patches.csv"),
                "qc_instance_csv": str(final_qc_bundle / "pannuke_mask_qc_instances.csv"),
                "qc_markdown": str(final_qc_bundle / "pannuke_mask_qc.md"),
                "qc_overlay": str(final_qc_bundle / "pannuke_mask_qc_overlays.png"),
                "qc_overlay_selection": str(
                    final_qc_bundle / "pannuke_mask_qc_overlay_selection.json"
                ),
                "qc_artifact_manifest": str(final_qc_bundle / "artifact_manifest.json"),
                "qc_selection_sha256": qc_artifacts.selection_sha256,
                "qc_overlay_sha256": qc_artifacts.overlay_sha256,
            },
            indent=2,
        )
    )


@data_app.command("audit-duplicates")
def audit_duplicates_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--root", "--data-root")] = None,
    output_directory: Annotated[Path | None, typer.Option("--output-dir")] = None,
    rankings_csv: Annotated[Path | None, typer.Option("--rankings-csv")] = None,
    embedding_cache: Annotated[Path | None, typer.Option("--embedding-cache")] = None,
    max_perceptual_patches: Annotated[int | None, typer.Option(min=1)] = None,
    max_hamming_distance: Annotated[int, typer.Option(min=0)] = 4,
    max_embedding_patches: Annotated[int | None, typer.Option(min=2)] = None,
    min_embedding_cosine: Annotated[float, typer.Option(min=-1.0, max=1.0)] = 0.995,
    embedding_device: Annotated[str, typer.Option()] = "auto",
    allow_weight_download: Annotated[bool, typer.Option("--allow-weight-download")] = False,
    skip_embedding_signal: Annotated[bool, typer.Option("--skip-embedding-signal")] = False,
) -> None:
    """Rank cross-fold exact and two-signal near duplicates for review only."""

    from histo_audit.pannuke import PanNukeError, audit_pannuke_duplicates

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    if output_directory is not None:
        custom_output = True
        output = _resolve_from_root(root, output_directory)
    else:
        custom_output = False
        output = root / "artifacts" / "duplicate_audit"
    report_path = (
        output / "cross_fold_duplicates.md"
        if custom_output
        else root / "reports" / "cross_fold_duplicates.md"
    )
    default_rankings = (
        output / "cross_fold_duplicate_candidates.csv"
        if custom_output
        else root / "artifacts" / "rankings" / "cross_fold_duplicate_candidates.csv"
    )
    rankings_path = (
        _resolve_from_root(root, rankings_csv) if rankings_csv is not None else default_rankings
    )
    provenance_path = (
        output / "pannuke_patch_hash_provenance.csv"
        if custom_output
        else root / "artifacts" / "provenance" / "pannuke_patch_hashes.csv"
    )
    visual_grid_path = (
        output / "cross_fold_duplicate_candidate_grid.png"
        if custom_output
        else root / "reports" / "figures" / "cross_fold_duplicate_candidates.png"
    )
    embedding_cache_path = (
        _resolve_from_root(root, embedding_cache) if embedding_cache is not None else None
    )
    try:
        output = _require_derived_destination_outside_raw(
            located, output, "duplicate-audit output directory", directory=True
        )
        report_path = _require_derived_destination_outside_raw(
            located, report_path, "duplicate-audit Markdown"
        )
        rankings_path = _require_derived_destination_outside_raw(
            located, rankings_path, "duplicate rankings CSV"
        )
        provenance_path = _require_derived_destination_outside_raw(
            located, provenance_path, "duplicate provenance CSV"
        )
        visual_grid_path = _require_derived_destination_outside_raw(
            located, visual_grid_path, "duplicate visual grid"
        )
        duplicate_destinations: list[tuple[str, Path]] = [
            ("duplicate-audit output directory", output),
            ("duplicate-audit Markdown", report_path),
            ("duplicate rankings CSV", rankings_path),
            ("duplicate provenance CSV", provenance_path),
            ("duplicate visual grid", visual_grid_path),
        ]
        if embedding_cache_path is not None:
            embedding_cache_path = _require_derived_destination_outside_raw(
                located, embedding_cache_path, "duplicate embedding cache"
            )
            duplicate_destinations.append(("duplicate embedding cache", embedding_cache_path))
            _require_destination_suffix(embedding_cache_path, ".npz", "duplicate embedding cache")
        _require_distinct_destinations(duplicate_destinations)
        _require_destination_suffix(report_path, ".md", "duplicate-audit Markdown")
        _require_destination_suffix(rankings_path, ".csv", "duplicate rankings CSV")
        _require_destination_suffix(provenance_path, ".csv", "duplicate provenance CSV")
        _require_destination_suffix(visual_grid_path, ".png", "duplicate visual grid")
        artifacts = audit_pannuke_duplicates(
            located,
            output,
            max_perceptual_patches=max_perceptual_patches,
            max_hamming_distance=max_hamming_distance,
            max_embedding_patches=max_embedding_patches,
            min_embedding_cosine_similarity=min_embedding_cosine,
            embedding_device=embedding_device,
            allow_weight_download=allow_weight_download,
            run_embedding_signal=not skip_embedding_signal,
            embedding_cache_path=embedding_cache_path,
            report_path=report_path,
            rankings_csv_path=rankings_path,
            hash_provenance_csv_path=provenance_path,
            visual_grid_path=visual_grid_path,
        )
    except (PanNukeError, OSError, ValueError) as exc:
        _failure(f"PanNuke duplicate audit failed: {type(exc).__name__}: {exc}")
    gate_complete = getattr(artifacts, "required_two_signal_gate_complete", None)
    typer.echo(
        json.dumps(
            {
                "status": (
                    "completed"
                    if gate_complete is True
                    else "completed_with_incomplete_required_duplicate_gate"
                ),
                "automatic_deletion": False,
                "exact_pair_count": artifacts.exact_pair_count,
                "perceptual_pair_count": artifacts.perceptual_pair_count,
                "embedding_pair_count": getattr(artifacts, "embedding_pair_count", None),
                "embedding_status": getattr(artifacts, "embedding_status", None),
                "required_two_signal_near_duplicate_gate_complete": getattr(
                    artifacts, "required_two_signal_gate_complete", None
                ),
                "sampled_patch_count": artifacts.sampled_patch_count,
                "embedding_sampled_patch_count": getattr(
                    artifacts, "embedding_sampled_patch_count", None
                ),
                "json": str(artifacts.json_path.resolve()),
                "rankings_csv": str(artifacts.csv_path.resolve()),
                "markdown": (
                    str(artifacts.markdown_path.resolve())
                    if hasattr(artifacts, "markdown_path")
                    else None
                ),
                "visual_grid": (
                    str(artifacts.visual_grid_path.resolve())
                    if hasattr(artifacts, "visual_grid_path")
                    else None
                ),
                "patch_hash_provenance_csv": (
                    str(artifacts.hash_provenance_csv_path.resolve())
                    if hasattr(artifacts, "hash_provenance_csv_path")
                    else None
                ),
            },
            indent=2,
        )
    )
    if gate_complete is False:
        typer.echo(
            "ERROR: required full-release pHash and frozen-ResNet duplicate coverage "
            "is incomplete; M5 remains open.",
            err=True,
        )
        raise typer.Exit(code=1)


@data_app.command("build-manifest")
def build_manifest_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--root", "--data-root")] = None,
    output_directory: Annotated[Path | None, typer.Option("--output-dir")] = None,
    batch_rows: Annotated[int, typer.Option(min=1)] = 4096,
) -> None:
    """Build a validated immutable-label nucleus manifest from local PanNuke."""

    from histo_audit.pannuke import PanNukeError, build_nucleus_manifest

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    output = (
        _resolve_from_root(root, output_directory)
        if output_directory is not None
        else root / "data" / "manifests" / "pannuke"
    )
    try:
        output = _require_derived_destination_outside_raw(
            located, output, "nucleus-manifest output directory", directory=True
        )
        artifacts = build_nucleus_manifest(located, output, batch_rows=batch_rows)
    except (PanNukeError, OSError, ValueError) as exc:
        _failure(f"PanNuke manifest build failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "completed",
                "row_count": artifacts.row_count,
                "patch_count": artifacts.patch_count,
                "manifest_sha256": artifacts.sha256,
                "parquet": str(artifacts.parquet_path.resolve()),
                "summary_csv": str(artifacts.summary_csv_path.resolve()),
            },
            indent=2,
        )
    )


@data_app.command("build-pilot-development-view")
def build_pilot_development_view_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    validation_json: Annotated[Path, typer.Option("--validation-json")] = Path(
        "reports/pannuke_validation.json"
    ),
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "data/manifests/pannuke/pannuke_nucleus_manifest.parquet"
    ),
    duplicate_audit_json: Annotated[Path, typer.Option("--duplicate-audit-json")] = Path(
        "artifacts/duplicate_audit/pannuke_duplicate_audit.json"
    ),
    output_path: Annotated[Path, typer.Option("--output")] = Path(
        "data/manifests/pannuke/pannuke_pilot_development_manifest.parquet"
    ),
) -> None:
    """Build the immutable folds-1/2 privacy boundary before starting a pilot run."""

    from histo_audit.experiment import build_pannuke_pilot_development_manifest_view

    root = project_root.resolve()
    validation = _resolve_from_root(root, validation_json)
    manifest = _resolve_from_root(root, manifest_path)
    duplicate = _resolve_from_root(root, duplicate_audit_json)
    output = _resolve_from_root(root, output_path)
    try:
        result = build_pannuke_pilot_development_manifest_view(
            validation,
            manifest,
            duplicate,
            output,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _failure(f"PanNuke pilot development-view construction failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "completed",
                "policy": "pre_pilot_privacy_gate_v1",
                "development_manifest": str(result.parquet_path.resolve()),
                "gate_certificate": str(result.metadata_path.resolve()),
                "canonical_manifest_sha256": result.canonical_manifest_sha256,
                "development_manifest_sha256": result.development_manifest_sha256,
                "development_instance_count": result.development_instance_count,
            },
            indent=2,
        )
    )


@representations_app.command("extract")
def extract_representations_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--data-root")] = None,
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "data/manifests/pannuke/pannuke_nucleus_manifest.parquet"
    ),
    output_directory: Annotated[Path, typer.Option("--output-dir")] = Path(
        "artifacts/embeddings/pannuke"
    ),
    independence_output_path: Annotated[
        Path | None,
        typer.Option(
            "--independence-output",
            help=(
                "Build the strict real audit-slice independence matrix in the same process; "
                "on failure the public cache bundle is retracted."
            ),
        ),
    ] = None,
    primary_config_path: Annotated[Path, typer.Option("--primary-config")] = Path(
        "configs/primary.yaml"
    ),
    sample_ids_file: Annotated[Path | None, typer.Option("--sample-ids-file")] = None,
    device: Annotated[str, typer.Option()] = "auto",
    allow_weight_download: Annotated[bool, typer.Option("--allow-weight-download")] = False,
    crop_output_size: Annotated[int, typer.Option(min=1)] = 64,
    crop_padding: Annotated[int, typer.Option(min=0)] = 8,
    context_brightness: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.45,
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    include_context_embeddings: Annotated[
        bool,
        typer.Option(
            "--include-context-embeddings",
            help=(
                "Also extract the context-RGB cache and its separate target-morphometrics "
                "concatenation ablation."
            ),
        ),
    ] = False,
) -> None:
    """Extract exact-target crops, engineered features, and frozen ResNet embeddings."""

    from histo_audit.experiment.representation_independence import (
        build_pannuke_representation_cache_with_independence,
    )
    from histo_audit.pannuke import PanNukeError, validate_pannuke
    from histo_audit.representations import (
        PanNukeCropConfig,
        ResNet18EmbeddingConfig,
        build_pannuke_representation_cache,
        require_full_manifest_cache_disk_space,
    )

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    manifest = _resolve_from_root(root, manifest_path)
    output = _resolve_from_root(root, output_directory)
    independence_output = (
        _resolve_from_root(root, independence_output_path)
        if independence_output_path is not None
        else None
    )
    requested_ids: tuple[str, ...] | None = None
    if sample_ids_file is not None:
        ids_path = _resolve_from_root(root, sample_ids_file)
        try:
            requested_ids = tuple(
                line.strip()
                for line in ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError as exc:
            _failure(f"cannot read sample ID file: {exc}")
        if not requested_ids or len(set(requested_ids)) != len(requested_ids):
            _failure("sample ID file must contain unique non-empty IDs")
    try:
        require_full_manifest_cache_disk_space(
            manifest,
            output,
            sample_ids=requested_ids,
        )
        validation = validate_pannuke(
            located,
            root / "reports",
            max_samples_per_fold=CANONICAL_PANNUKE_VALIDATION_MAX_SAMPLES_PER_FOLD,
            max_overlay_patches=CANONICAL_PANNUKE_VALIDATION_MAX_OVERLAY_PATCHES,
            raw_inventory_csv_path=root / "data" / "manifests" / "raw_files_sha256.csv",
            overlay_path=root / "artifacts" / "figures" / "pannuke_overlay_grid.png",
        )
        crop_config = PanNukeCropConfig(
            output_size=crop_output_size,
            padding=crop_padding,
            context_brightness=context_brightness,
        )
        resnet_config = ResNet18EmbeddingConfig(
            input_variant="target_highlighted_rgb",
            context_brightness=context_brightness,
            device=device,
            batch_size=batch_size,
            allow_weight_download=allow_weight_download,
        )
        independence_artifact = None
        if independence_output is None:
            artifacts = build_pannuke_representation_cache(
                validation.result,
                manifest,
                output,
                sample_ids=requested_ids,
                crop_config=crop_config,
                resnet_config=resnet_config,
                include_context_embeddings=include_context_embeddings,
            )
        else:
            if requested_ids is not None:
                raise ValueError(
                    "same-process primary independence publication requires the full "
                    "analysis-eligible manifest, not --sample-ids-file"
                )
            if not include_context_embeddings:
                raise ValueError("--independence-output requires --include-context-embeddings")
            from histo_audit.config import load_config

            primary_config = load_config(_resolve_from_root(root, primary_config_path))
            data_policy = primary_config.get("data")
            if not isinstance(data_policy, dict):
                raise ValueError("primary config data policy must be a mapping")
            manifest_authority = data_policy.get("analysis_manifest_authority")
            expected_authority_fields = {
                "canonical_manifest_sha256",
                "analysis_eligible_sample_order_sha256",
                "analysis_eligible_sample_count",
            }
            if (
                not isinstance(manifest_authority, dict)
                or set(manifest_authority) != expected_authority_fields
            ):
                raise ValueError(
                    "primary config data.analysis_manifest_authority must contain exactly "
                    "canonical_manifest_sha256, analysis_eligible_sample_order_sha256, and "
                    "analysis_eligible_sample_count"
                )
            bundle = build_pannuke_representation_cache_with_independence(
                validation.result,
                manifest,
                output,
                independence_output,
                class_order=tuple(int(value) for value in data_policy["class_order"]),
                development_official_folds=tuple(
                    int(value) for value in data_policy["development_official_folds"]
                ),
                final_test_fold=int(data_policy["final_test_fold"]),
                reference_validation_fraction_groups=float(
                    data_policy["reference_validation_fraction_groups"]
                ),
                split_seed=int(data_policy["split_seed"]),
                expected_canonical_manifest_sha256=str(
                    manifest_authority["canonical_manifest_sha256"]
                ),
                expected_analysis_eligible_sample_order_sha256=str(
                    manifest_authority["analysis_eligible_sample_order_sha256"]
                ),
                expected_analysis_eligible_sample_count=manifest_authority[
                    "analysis_eligible_sample_count"
                ],
                crop_config=crop_config,
                resnet_config=resnet_config,
            )
            artifacts = bundle.representations
            independence_artifact = bundle.independence
    except (PanNukeError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _failure(f"PanNuke representation extraction failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "completed",
                "sample_count": len(artifacts.crops.sample_ids),
                "identity_verified": bool(artifacts.crops.identity_verified.all()),
                "crop_cache": str(artifacts.crop_cache_path.resolve()),
                "engineered_cache": str(artifacts.engineered_cache_path.resolve()),
                "embedding_cache": str(artifacts.embeddings.cache_path.resolve())
                if artifacts.embeddings.cache_path is not None
                else None,
                "context_embedding_cache": (
                    str(artifacts.context_embeddings.cache_path.resolve())
                    if artifacts.context_embeddings is not None
                    and artifacts.context_embeddings.cache_path is not None
                    else None
                ),
                "context_morphometrics_cache": (
                    str(artifacts.context_morphometrics.cache_path.resolve())
                    if artifacts.context_morphometrics is not None
                    else None
                ),
                "embedding_device": artifacts.embeddings.metadata.get("device"),
                "weight_sha256": artifacts.embeddings.metadata.get("weight_sha256"),
                "representation_independence": (
                    str(independence_artifact.path.resolve())
                    if independence_artifact is not None
                    else None
                ),
                "representation_independence_sha256": (
                    independence_artifact.file_sha256 if independence_artifact is not None else None
                ),
            },
            indent=2,
        )
    )


@experiment_app.command("smoke")
def smoke_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/smoke.yaml"),
    runs_root: Annotated[
        Path | None,
        typer.Option("--runs-root", help="Optional run root; defaults to artifacts/runs."),
    ] = None,
) -> None:
    """Execute the deterministic core pipeline as an immutable tracked run."""

    from histo_audit.reporting.smoke_runner import execute_synthetic_smoke

    root = project_root.resolve()
    config_file = _resolve_from_root(root, config_path)
    resolved_runs_root = _resolve_from_root(root, runs_root) if runs_root else None
    try:
        result = execute_synthetic_smoke(
            project_root=root,
            config_path=config_file,
            runs_root=resolved_runs_root,
        )
    except Exception as exc:
        _failure(f"synthetic smoke failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "success": result.success,
                "status": result.status,
                "run_id": result.run_id,
                "run_directory": str(result.run_directory.resolve()),
                "metrics": str(result.metrics_path.resolve()),
                "report_markdown": str(result.report.markdown_path.resolve()),
                "report_html": str(result.report.html_path.resolve()),
                "figures": [str(path.resolve()) for path in result.report.figure_paths],
            },
            indent=2,
        )
    )


@experiment_app.command("withdraw-run-eligibility")
def withdraw_run_eligibility_command(
    run_directory: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help="Completed sealed run whose scientific stage eligibility is withdrawn.",
        ),
    ],
    reason_code: Annotated[
        str,
        typer.Option(
            "--reason-code",
            help="Stable lowercase machine-readable reason code.",
        ),
    ],
    reason: Annotated[
        str,
        typer.Option(
            "--reason",
            help="Auditable explanation; this does not alter the run's completed status.",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
) -> None:
    """Irreversibly withdraw one sealed run from scientific stage eligibility."""

    from histo_audit.utils.run_tracking import (
        RUN_DISPOSITION_REGISTRY_FILENAME,
        withdraw_run_eligibility,
    )

    root = project_root.resolve()
    run_path = _resolve_from_root(root, run_directory)
    try:
        record = withdraw_run_eligibility(
            run_path,
            reason_code=reason_code,
            reason=reason,
        )
    except (OSError, ValueError) as exc:
        _failure(f"run eligibility withdrawal failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "event_type": record["event_type"],
                "run_id": record["run_id"],
                "run_terminal_status": record["terminal_status"],
                "scientific_stage_eligible": record["scientific_stage_eligible"],
                "reason_code": record["reason_code"],
                "record_sha256": record["record_sha256"],
                "disposition_registry": str(
                    (run_path.parent / RUN_DISPOSITION_REGISTRY_FILENAME).resolve()
                ),
            },
            indent=2,
        )
    )


@experiment_app.command("pilot")
def pilot_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    data_root: Annotated[Path | None, typer.Option("--data-root")] = None,
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "data/manifests/pannuke/pannuke_nucleus_manifest.parquet"
    ),
    duplicate_audit_json: Annotated[Path, typer.Option("--duplicate-audit-json")] = Path(
        "artifacts/duplicate_audit/pannuke_duplicate_audit.json"
    ),
    development_manifest_path: Annotated[Path, typer.Option("--development-manifest")] = Path(
        "data/manifests/pannuke/pannuke_pilot_development_manifest.parquet"
    ),
    gate_certificate_path: Annotated[Path, typer.Option("--gate-certificate")] = Path(
        "data/manifests/pannuke/pannuke_pilot_development_manifest.parquet.metadata.json"
    ),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/pilot.yaml"),
    device: Annotated[str, typer.Option()] = "auto",
    allow_weight_download: Annotated[bool, typer.Option("--allow-weight-download")] = False,
) -> None:
    """Run the fixed pilot from a prebuilt privacy-safe folds-1/2 manifest view."""

    from histo_audit.experiment import run_pannuke_pilot
    from histo_audit.pannuke import sha256_file

    root = project_root.resolve()
    located = _require_pannuke(root, data_root)
    manifest = _resolve_from_root(root, manifest_path)
    duplicate_path = _resolve_from_root(root, duplicate_audit_json)
    development_manifest = _resolve_from_root(root, development_manifest_path)
    gate_certificate = _resolve_from_root(root, gate_certificate_path)
    config_file = _resolve_from_root(root, config_path)
    try:
        result = run_pannuke_pilot(
            gate_certificate,
            manifest,
            development_manifest_source=development_manifest,
            project_root=root,
            expected_data_root=located,
            config=config_file,
            allow_weight_download=allow_weight_download,
            device=device,
            duplicate_audit_status=f"complete_sha256:{sha256_file(duplicate_path)}",
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        _failure(f"PanNuke pilot failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "completed",
                "completion_stage_candidate": "PILOT_COMPLETE",
                "independent_post_seal_inspection_required": True,
                "run_id": result.run_id,
                "run_directory": str(result.run_directory.resolve()),
                "metrics": str(result.metrics_path.resolve()),
                "report": str(result.report_path.resolve()),
                "selected_ids": str(result.selected_ids_path.resolve()),
                "audit_sample_count": result.audit_sample_count,
                "final_reference_sample_count": result.final_reference_sample_count,
                "exact_corruption_count": result.exact_corruption_count,
            },
            indent=2,
        )
    )


@experiment_app.command("verify-pilot-post-seal")
def verify_pilot_post_seal_command(
    run_directory: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help="Explicit completed sealed PanNuke pilot run directory.",
        ),
    ],
    development_manifest_path: Annotated[
        Path,
        typer.Option(
            "--development-manifest",
            help="Explicit external folds-1/2 manifest used by the pilot.",
        ),
    ],
    gate_certificate_path: Annotated[
        Path,
        typer.Option(
            "--gate-certificate",
            help="Explicit external pre-pilot privacy-gate certificate.",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
) -> None:
    """Read-only, fail-closed verification of a sealed PanNuke pilot."""

    from histo_audit.experiment.pilot_postseal import verify_pilot_post_seal

    root = project_root.resolve()
    run_path = _resolve_from_root(root, run_directory)
    development_manifest = _resolve_from_root(root, development_manifest_path)
    gate_certificate = _resolve_from_root(root, gate_certificate_path)
    try:
        result = verify_pilot_post_seal(
            run_path,
            development_manifest_source=development_manifest,
            gate_certificate_source=gate_certificate,
        )
    except Exception as exc:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "policy": "read_only_pilot_post_seal_verification_v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "automatic_withdrawal_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@experiment_app.command("lifecycle-rehearsal")
def lifecycle_rehearsal_command(
    authority_directory: Annotated[
        Path,
        typer.Option(
            "--authority-dir",
            help="Verified immutable freeze/amendment authority for the current source tree.",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    runs_root: Annotated[
        Path,
        typer.Option("--runs-root", help="Run registry root for the operational rehearsal."),
    ] = Path("artifacts/runs"),
    retry_of_run_id: Annotated[
        str | None,
        typer.Option(
            "--retry-of-run-id",
            help="Optional exact sealed failed rehearsal predecessor run ID.",
        ),
    ] = None,
) -> None:
    """Exercise the real persistence/publication/seal lifecycle on synthetic evidence."""

    from histo_audit.workflows import execute_lifecycle_rehearsal

    root = project_root.resolve()
    authority = _resolve_from_root(root, authority_directory)
    run_root = _resolve_from_root(root, runs_root)
    try:
        result = execute_lifecycle_rehearsal(
            project_root=root,
            authority_directory=authority,
            runs_root=run_root,
            retry_of_run_id=retry_of_run_id,
        )
    except Exception as exc:
        _failure(f"lifecycle rehearsal failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "operational_lifecycle_qualification_rehearsal",
                "scientific_outcome": False,
                "project_completion_status_changed": False,
                "result": result.as_dict(),
                "next_command": (
                    ".venv\\Scripts\\python.exe -m histo_audit experiment "
                    "verify-lifecycle-rehearsal --project-root . --authority-dir "
                    f'"{authority}" --rehearsal-run-dir "{result.run_directory}" '
                    f'--runs-root "{run_root}"'
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@experiment_app.command("verify-lifecycle-rehearsal")
def verify_lifecycle_rehearsal_command(
    authority_directory: Annotated[
        Path,
        typer.Option("--authority-dir", help="Same verified authority used by the rehearsal."),
    ],
    rehearsal_run_directory: Annotated[
        Path,
        typer.Option("--rehearsal-run-dir", help="Sealed rehearsal run to reopen."),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    runs_root: Annotated[
        Path,
        typer.Option("--runs-root", help="Run registry root for the sealed readiness record."),
    ] = Path("artifacts/runs"),
) -> None:
    """Fresh-process verification that publishes one immutable readiness run."""

    from histo_audit.workflows import verify_lifecycle_rehearsal_fresh_process

    root = project_root.resolve()
    try:
        result = verify_lifecycle_rehearsal_fresh_process(
            project_root=root,
            authority_directory=_resolve_from_root(root, authority_directory),
            rehearsal_run_directory=_resolve_from_root(root, rehearsal_run_directory),
            runs_root=_resolve_from_root(root, runs_root),
        )
    except Exception as exc:
        _failure(f"lifecycle rehearsal verification failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "operational_lifecycle_qualification_fresh_verification",
                "decision": "passed",
                "scientific_outcome": False,
                "project_completion_status_changed": False,
                "result": result.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@experiment_app.command("primary")
def primary_command(
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    freeze_directory: Annotated[
        Path,
        typer.Option(
            "--freeze-dir",
            "--freeze-directory",
            help="Immutable timestamped post-pilot preregistration freeze directory.",
        ),
    ] = Path("artifacts/preregistrations/latest"),
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", help="Exact frozen PanNuke dataset file or directory."),
    ] = Path("data/raw/pannuke"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Exact frozen nucleus manifest."),
    ] = Path("data/manifests/pannuke/pannuke_nucleus_manifest.parquet"),
    duplicate_audit_path: Annotated[
        Path,
        typer.Option("--duplicate-audit", help="Exact frozen duplicate-audit JSON."),
    ] = Path("artifacts/duplicate_audit/pannuke_duplicate_audit.json"),
    pathology_encoder_audit_path: Annotated[
        Path,
        typer.Option(
            "--pathology-encoder-audit",
            "--pathology-audit",
            help="Exact frozen pathology-encoder availability-audit JSON.",
        ),
    ] = Path("reports/pathology_encoder_availability.json"),
    frozen_primary_config_path: Annotated[
        Path,
        typer.Option(
            "--primary-config",
            "--frozen-primary-config",
            help="Canonical primary config authenticated by the freeze.",
        ),
    ] = Path("configs/primary_frozen.yaml"),
    frozen_confirmatory_config_path: Annotated[
        Path,
        typer.Option(
            "--confirmatory-config",
            "--frozen-confirmatory-config",
            help="Canonical confirmatory config frozen before primary outcomes.",
        ),
    ] = Path("configs/confirmatory_frozen.yaml"),
) -> None:
    """Validate every frozen dependency before loading the real primary executor."""

    from histo_audit.workflows import validate_primary_execution_gate

    root = project_root.resolve()
    freeze = _resolve_from_root(root, freeze_directory)
    dataset = _resolve_from_root(root, dataset_path)
    manifest = _resolve_from_root(root, manifest_path)
    duplicate_audit = _resolve_from_root(root, duplicate_audit_path)
    pathology_audit = _resolve_from_root(root, pathology_encoder_audit_path)
    primary_config = _resolve_from_root(root, frozen_primary_config_path)
    confirmatory_config = _resolve_from_root(root, frozen_confirmatory_config_path)
    try:
        gate_evidence = validate_primary_execution_gate(
            project_root=root,
            freeze_directory=freeze,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology_audit,
            frozen_primary_config_path=primary_config,
            frozen_confirmatory_config_path=confirmatory_config,
        )
    except Exception as exc:
        _gate(
            "PRIMARY_STUDY_LOCKED",
            f"primary execution gate failed closed: {type(exc).__name__}: {exc}",
            next_command="python -m histo_audit preregistration freeze --help",
        )

    try:
        executor = _load_optional_study_executor(
            "histo_audit.experiment.primary_runner", "execute_primary_study"
        )
    except Exception as exc:
        _failure(f"primary executor import failed: {type(exc).__name__}: {exc}")
    if executor is None:
        _gate(
            "EXECUTOR_UNAVAILABLE",
            "the primary execution gate passed, but no real PanNuke primary-study executor "
            "is installed; no run directory was created and no completion stage was claimed.",
        )
    try:
        result = executor(
            gate_evidence=gate_evidence,
            project_root=root,
            freeze_directory=freeze,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology_audit,
            frozen_primary_config_path=primary_config,
            frozen_confirmatory_config_path=confirmatory_config,
        )
    except Exception as exc:
        _failure(f"real primary study failed: {type(exc).__name__}: {exc}")
    typer.echo(json.dumps({"status": "executor_returned", "result": result}, indent=2, default=str))


@experiment_app.command("primary-orphan-recovery")
def primary_orphan_recovery_command(
    authority_directory: Annotated[
        Path,
        typer.Option(
            "--authority-dir",
            "--freeze-dir",
            file_okay=False,
            help=(
                "Verified immutable post-outcome amendment containing the exclusive "
                "primary-recovery authorization."
            ),
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", help="Exact frozen PanNuke dataset file or directory."),
    ] = Path("data/raw/pannuke"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Exact frozen nucleus manifest."),
    ] = Path("data/manifests/pannuke/pannuke_nucleus_manifest.parquet"),
    duplicate_audit_path: Annotated[
        Path,
        typer.Option("--duplicate-audit", help="Exact frozen duplicate-audit JSON."),
    ] = Path("artifacts/duplicate_audit/pannuke_duplicate_audit.json"),
    pathology_encoder_audit_path: Annotated[
        Path,
        typer.Option(
            "--pathology-encoder-audit",
            "--pathology-audit",
            help="Exact frozen pathology-encoder availability-audit JSON.",
        ),
    ] = Path("reports/pathology_encoder_availability.json"),
    frozen_primary_config_path: Annotated[
        Path | None,
        typer.Option(
            "--primary-config",
            "--frozen-primary-config",
            help=(
                "Primary config snapshot inside --authority-dir; defaults to "
                "primary_frozen.yaml in that authority."
            ),
        ),
    ] = None,
    frozen_confirmatory_config_path: Annotated[
        Path | None,
        typer.Option(
            "--confirmatory-config",
            "--frozen-confirmatory-config",
            help=(
                "Confirmatory config snapshot inside --authority-dir; defaults to "
                "confirmatory_frozen.yaml in that authority."
            ),
        ),
    ] = None,
    runs_root: Annotated[
        Path,
        typer.Option(
            "--runs-root",
            file_okay=False,
            help="Run registry root containing the orphan and the new recovery run.",
        ),
    ] = Path("artifacts/runs"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Optional new unique recovery run ID; collisions fail without retry.",
        ),
    ] = None,
    preflight_only: Annotated[
        bool,
        typer.Option(
            "--preflight-only",
            help=(
                "Run only the public read-only recovery qualification; never create a "
                "RunTracker or copy artifacts."
            ),
        ),
    ] = False,
) -> None:
    """Execute one authorized zero-training orphan recovery with no automatic retry."""

    from histo_audit.config import load_config
    from histo_audit.experiment import (
        RecoveryAuthorization,
        build_primary_matrix_plan,
        primary_execution_controls_from_frozen_config,
    )
    from histo_audit.workflows import (
        PRIMARY_RECOVERY_EXPERIMENT_NAME,
        require_primary_recovery_authorization,
        validate_primary_execution_gate,
    )

    root = project_root.resolve()
    authority = _resolve_from_root(root, authority_directory)
    dataset = _resolve_from_root(root, dataset_path)
    manifest = _resolve_from_root(root, manifest_path)
    duplicate_audit = _resolve_from_root(root, duplicate_audit_path)
    pathology_audit = _resolve_from_root(root, pathology_encoder_audit_path)
    primary_config = (
        (authority / "primary_frozen.yaml").resolve()
        if frozen_primary_config_path is None
        else _resolve_from_root(root, frozen_primary_config_path)
    )
    confirmatory_config = (
        (authority / "confirmatory_frozen.yaml").resolve()
        if frozen_confirmatory_config_path is None
        else _resolve_from_root(root, frozen_confirmatory_config_path)
    )
    run_root = _resolve_from_root(root, runs_root)
    source_run_id: str | None = None
    try:
        gate_evidence = validate_primary_execution_gate(
            project_root=root,
            freeze_directory=authority,
            dataset_path=dataset,
            manifest_path=manifest,
            duplicate_audit_path=duplicate_audit,
            pathology_encoder_audit_path=pathology_audit,
            frozen_primary_config_path=primary_config,
            frozen_confirmatory_config_path=confirmatory_config,
            experiment_name=PRIMARY_RECOVERY_EXPERIMENT_NAME,
        )
        authorization_mapping = require_primary_recovery_authorization(authority)
        authorization = RecoveryAuthorization.from_mapping(
            authorization_mapping,
            authority_directory=authority,
            authority_artifact_root_sha256=gate_evidence.freeze_artifact_root_sha256,
            authority_manifest_sha256=gate_evidence.freeze_manifest_sha256,
        )
        source_run_id = authorization.source_run_id
        primary_config_payload = load_config(primary_config)
        plan = build_primary_matrix_plan(primary_config_payload)
        controls = primary_execution_controls_from_frozen_config(primary_config_payload)
    except Exception as exc:
        _emit_primary_recovery_error(
            exc,
            status="gated",
            authority_directory=authority,
            source_run_id=source_run_id,
            exit_code=2,
        )

    if preflight_only:
        try:
            preflight = _load_optional_study_executor(
                "histo_audit.experiment.primary_recovery_runner",
                "preflight_primary_orphan_recovery",
            )
        except Exception as exc:
            _emit_primary_recovery_error(
                exc,
                status="preflight_import_failed",
                authority_directory=authority,
                source_run_id=source_run_id,
                exit_code=1,
            )
        if preflight is None:
            _emit_primary_recovery_error(
                RuntimeError("primary orphan recovery preflight is unavailable"),
                status="preflight_unavailable",
                authority_directory=authority,
                source_run_id=source_run_id,
                exit_code=2,
            )
        try:
            result = preflight(
                gate_evidence=gate_evidence,
                plan=plan,
                controls=controls,
                authorization=authorization,
                runs_root=run_root,
                run_id=run_id,
            )
        except Exception as exc:
            _emit_primary_recovery_error(
                exc,
                status="preflight_failed",
                authority_directory=authority,
                source_run_id=source_run_id,
                exit_code=1,
            )
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "workflow": "primary_orphan_recovery",
                    "status": "preflight_passed",
                    "preflight_only": True,
                    "authority_directory": str(authority),
                    "source_run_id": source_run_id,
                    "completion_stage": None,
                    "study_outcome_eligible": False,
                    "run_tracker_created": False,
                    "copy_invoked": False,
                    "training_invoked": False,
                    "fallback_invoked": False,
                    "automatic_retry_allowed": False,
                    "result": result,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return

    try:
        executor = _load_optional_study_executor(
            "histo_audit.experiment.primary_recovery_runner",
            "execute_primary_orphan_recovery",
        )
    except Exception as exc:
        _emit_primary_recovery_error(
            exc,
            status="executor_import_failed",
            authority_directory=authority,
            source_run_id=source_run_id,
            exit_code=1,
        )
    if executor is None:
        _emit_primary_recovery_error(
            RuntimeError("primary orphan recovery executor is unavailable"),
            status="executor_unavailable",
            authority_directory=authority,
            source_run_id=source_run_id,
            exit_code=2,
        )
    try:
        result = executor(
            gate_evidence=gate_evidence,
            plan=plan,
            controls=controls,
            authorization=authorization,
            project_root=root,
            runs_root=run_root,
            run_id=run_id,
        )
    except Exception as exc:
        _emit_primary_recovery_error(
            exc,
            status="failed",
            authority_directory=authority,
            source_run_id=source_run_id,
            exit_code=1,
        )
    typer.echo(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "primary_orphan_recovery",
                "status": "executor_returned",
                "preflight_only": False,
                "authority_directory": str(authority),
                "source_run_id": source_run_id,
                "training_invoked": False,
                "fallback_invoked": False,
                "automatic_retry_allowed": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@experiment_app.command("confirmatory")
def confirmatory_command(
    primary_run_directory: Annotated[
        Path,
        typer.Option(
            "--primary-run-dir",
            help="Sealed, registry-backed PRIMARY_STUDY_COMPLETE run directory.",
        ),
    ] = Path("artifacts/runs/PRIMARY_RUN_REQUIRED"),
    lifecycle_readiness_run_directory: Annotated[
        Path,
        typer.Option(
            "--lifecycle-readiness-run-dir",
            help=(
                "Sealed fresh-process lifecycle-readiness verification run matching the "
                "current execution source and authority."
            ),
        ),
    ] = Path("artifacts/runs/LIFECYCLE_READINESS_REQUIRED"),
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    freeze_directory: Annotated[
        Path,
        typer.Option(
            "--freeze-dir",
            "--freeze-directory",
            help="Immutable timestamped post-pilot preregistration freeze directory.",
        ),
    ] = Path("artifacts/preregistrations/latest"),
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", help="Exact frozen PanNuke dataset file or directory."),
    ] = Path("data/raw/pannuke"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Exact frozen nucleus manifest."),
    ] = Path("data/manifests/pannuke/pannuke_nucleus_manifest.parquet"),
    duplicate_audit_path: Annotated[
        Path,
        typer.Option("--duplicate-audit", help="Exact frozen duplicate-audit JSON."),
    ] = Path("artifacts/duplicate_audit/pannuke_duplicate_audit.json"),
    pathology_encoder_audit_path: Annotated[
        Path,
        typer.Option(
            "--pathology-encoder-audit",
            "--pathology-audit",
            help="Exact frozen pathology-encoder availability-audit JSON.",
        ),
    ] = Path("reports/pathology_encoder_availability.json"),
    frozen_primary_config_path: Annotated[
        Path,
        typer.Option(
            "--primary-config",
            "--frozen-primary-config",
            help="Canonical primary config authenticated by the freeze.",
        ),
    ] = Path("configs/primary_frozen.yaml"),
    frozen_confirmatory_config_path: Annotated[
        Path,
        typer.Option(
            "--confirmatory-config",
            "--frozen-confirmatory-config",
            help="Canonical confirmatory config frozen before primary outcomes.",
        ),
    ] = Path("configs/confirmatory_frozen.yaml"),
) -> None:
    """Reject the superseded direct entry; protected execution requires Q/E."""

    _gate(
        "CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED",
        "direct confirmatory execution is disabled before lifecycle, dataset, cache, "
        "executor, or run-directory access. The unchanged original confirmatory may run "
        "only through the sealed execution capsule with the exact published Q, one-use E, "
        "strict published-T0 lifecycle result, and qualified event-driven supervisor.",
    )


@experiment_app.command("resource-bounded-sensitivity")
def resource_bounded_sensitivity_command(
    primary_run_directory: Annotated[
        Path,
        typer.Option(
            "--primary-run-dir",
            help="Exact sealed historical primary run authorized by authority C.",
        ),
    ] = Path("artifacts/runs/PRIMARY_RUN_REQUIRED"),
    resource_authority_directory: Annotated[
        Path,
        typer.Option(
            "--resource-authority-dir",
            help=(
                "Effective immutable resource-bounded post-outcome execution authority "
                "C or its unique technical successor D."
            ),
        ),
    ] = Path("artifacts/preregistration_amendments/RESOURCE_AUTHORITY_REQUIRED"),
    lifecycle_readiness_run_directory: Annotated[
        Path,
        typer.Option(
            "--lifecycle-readiness-run-dir",
            help=(
                "Sealed fresh-process lifecycle-readiness run for the current source "
                "and effective resource authority."
            ),
        ),
    ] = Path("artifacts/runs/LIFECYCLE_READINESS_REQUIRED"),
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", help="Exact frozen PanNuke dataset file or directory."),
    ] = Path("data/raw/pannuke"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Exact frozen nucleus manifest."),
    ] = Path("data/manifests/pannuke/pannuke_nucleus_manifest.parquet"),
    duplicate_audit_path: Annotated[
        Path,
        typer.Option("--duplicate-audit", help="Exact frozen duplicate-audit JSON."),
    ] = Path("artifacts/duplicate_audit/pannuke_duplicate_audit.json"),
    pathology_encoder_audit_path: Annotated[
        Path,
        typer.Option(
            "--pathology-encoder-audit",
            "--pathology-audit",
            help="Exact frozen pathology-encoder availability-audit JSON.",
        ),
    ] = Path("reports/pathology_encoder_availability.json"),
    runs_root: Annotated[
        Path,
        typer.Option(
            "--runs-root",
            file_okay=False,
            help="Explicit parent directory for tracked study runs and capacity checks.",
        ),
    ] = Path("artifacts/runs"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Optional new run ID; it must differ from any explicit predecessor.",
        ),
    ] = None,
    checkpoint_predecessor_run_directory: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-predecessor-run-dir",
            help=(
                "Explicit predecessor for one successor resume. Its exact basename is "
                "used as retry_of_run_id; no run is discovered automatically."
            ),
        ),
    ] = None,
    preflight_only: Annotated[
        bool,
        typer.Option(
            "--preflight-only",
            help=(
                "Validate lifecycle, dual authority, inputs, capacity, and resume "
                "eligibility without creating a tracked or scientific run."
            ),
        ),
    ] = False,
) -> None:
    """Run the permanently non-claiming resource-bounded sensitivity path."""

    root = project_root.resolve()
    primary_run = _resolve_from_root(root, primary_run_directory)
    authority = _resolve_from_root(root, resource_authority_directory)
    lifecycle_readiness = _resolve_from_root(
        root,
        lifecycle_readiness_run_directory,
    )
    dataset = _resolve_from_root(root, dataset_path)
    manifest = _resolve_from_root(root, manifest_path)
    duplicate_audit = _resolve_from_root(root, duplicate_audit_path)
    pathology_audit = _resolve_from_root(root, pathology_encoder_audit_path)
    run_root = _resolve_from_root(root, runs_root)
    predecessor = (
        _resolve_from_root(root, checkpoint_predecessor_run_directory)
        if checkpoint_predecessor_run_directory is not None
        else None
    )
    run_mode = "successor_resume" if predecessor is not None else "fresh"
    retry_of_run_id = predecessor.name if predecessor is not None else None
    if predecessor is not None and run_id == retry_of_run_id:
        _emit_resource_bounded_error(
            ValueError("a successor --run-id must differ from its explicit predecessor"),
            status="invalid_run_identity",
            preflight_only=preflight_only,
            run_mode=run_mode,
            retry_of_run_id=retry_of_run_id,
            exit_code=2,
        )

    function_name = (
        "preflight_resource_bounded_sensitivity"
        if preflight_only
        else "execute_resource_bounded_sensitivity"
    )
    try:
        executor = _load_optional_study_executor(
            "histo_audit.experiment.resource_bounded_runner",
            function_name,
        )
    except Exception as exc:
        _emit_resource_bounded_error(
            exc,
            status="executor_import_failed",
            preflight_only=preflight_only,
            run_mode=run_mode,
            retry_of_run_id=retry_of_run_id,
            exit_code=1,
        )
    if executor is None:
        _emit_resource_bounded_error(
            RuntimeError(f"resource-bounded {function_name} is unavailable"),
            status="executor_unavailable",
            preflight_only=preflight_only,
            run_mode=run_mode,
            retry_of_run_id=retry_of_run_id,
            exit_code=2,
        )
    executor_kwargs = {
        "run_mode": run_mode,
        "primary_run_directory": primary_run,
        "project_root": root,
        "resource_authority_directory": authority,
        "dataset_path": dataset,
        "manifest_path": manifest,
        "duplicate_audit_path": duplicate_audit,
        "pathology_encoder_audit_path": pathology_audit,
        "lifecycle_readiness_run_directory": lifecycle_readiness,
        "checkpoint_predecessor_run_directory": predecessor,
        "retry_of_run_id": retry_of_run_id,
        "runs_root": run_root,
        "run_id": run_id,
    }
    try:
        result = executor(**executor_kwargs)
        payload = _resource_bounded_success_payload(
            result,
            preflight_only=preflight_only,
            run_mode=run_mode,
            retry_of_run_id=retry_of_run_id,
        )
    except Exception as exc:
        _emit_resource_bounded_error(
            exc,
            status="preflight_failed" if preflight_only else "failed",
            preflight_only=preflight_only,
            run_mode=run_mode,
            retry_of_run_id=retry_of_run_id,
            exit_code=1,
        )
    typer.echo(
        json.dumps(
            _resource_bounded_safe_json_value(payload),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@preregistration_app.command("freeze")
def freeze_preregistration_command(
    pilot_run_directory: Annotated[
        Path,
        typer.Option(
            "--pilot-run-dir",
            help="Sealed completed PanNuke pilot run directory.",
        ),
    ],
    dataset_path: Annotated[
        Path,
        typer.Option(
            "--dataset",
            help="Exact dataset file or directory recorded by the sealed pilot.",
        ),
    ],
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Exact nucleus manifest file recorded by the sealed pilot.",
        ),
    ],
    raw_checksum_manifest_path: Annotated[
        Path,
        typer.Option(
            "--raw-checksum-manifest",
            help="Non-empty checksum inventory for the verified raw dataset.",
        ),
    ],
    duplicate_audit_path: Annotated[
        Path,
        typer.Option(
            "--duplicate-audit",
            help="Completed full-coverage two-signal PanNuke duplicate-audit JSON.",
        ),
    ],
    pathology_encoder_audit_path: Annotated[
        Path,
        typer.Option(
            "--pathology-encoder-audit",
            help="Frozen-priority pathology-encoder availability audit JSON.",
        ),
    ],
    pilot_development_manifest_path: Annotated[
        Path,
        typer.Option(
            "--pilot-dev-manifest",
            "--pilot-development-manifest",
            help="External folds-1/2 manifest used as byte-identity authority for the pilot.",
        ),
    ],
    pilot_gate_certificate_path: Annotated[
        Path,
        typer.Option(
            "--pilot-gate-certificate",
            help="External pre-pilot gate certificate used as byte-identity authority.",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    preregistration_path: Annotated[
        Path,
        typer.Option("--preregistration", help="Completed READY_FOR_FREEZE Markdown plan."),
    ] = Path("PRE_REGISTRATION.md"),
    primary_config_path: Annotated[
        Path,
        typer.Option("--primary-config", help="Completed primary-study YAML configuration."),
    ] = Path("configs/primary.yaml"),
    confirmatory_config_path: Annotated[
        Path,
        typer.Option(
            "--confirmatory-config",
            help="Exact confirmatory matrix YAML to freeze before primary outcomes.",
        ),
    ] = Path("configs/confirmatory.yaml"),
    freeze_root: Annotated[
        Path,
        typer.Option(
            "--freeze-root",
            help="Parent for the timestamped immutable freeze evidence directory.",
        ),
    ] = Path("artifacts/preregistrations"),
) -> None:
    """Freeze a completed primary definition after a verified, sealed real-data pilot."""

    from histo_audit.workflows import freeze_preregistration, verify_preregistration_freeze

    root = project_root.resolve()
    try:
        result = freeze_preregistration(
            project_root=root,
            pilot_run_directory=_resolve_from_root(root, pilot_run_directory),
            dataset_path=_resolve_from_root(root, dataset_path),
            manifest_path=_resolve_from_root(root, manifest_path),
            raw_checksum_manifest_path=_resolve_from_root(root, raw_checksum_manifest_path),
            duplicate_audit_path=_resolve_from_root(root, duplicate_audit_path),
            pathology_encoder_audit_path=_resolve_from_root(root, pathology_encoder_audit_path),
            pilot_development_manifest_path=_resolve_from_root(
                root, pilot_development_manifest_path
            ),
            pilot_gate_certificate_path=_resolve_from_root(root, pilot_gate_certificate_path),
            preregistration_path=_resolve_from_root(root, preregistration_path),
            primary_config_path=_resolve_from_root(root, primary_config_path),
            confirmatory_config_path=_resolve_from_root(root, confirmatory_config_path),
            freeze_root=_resolve_from_root(root, freeze_root),
        )
        verification = verify_preregistration_freeze(
            result.freeze_directory,
            frozen_primary_config_path=result.frozen_primary_config_path,
            frozen_confirmatory_config_path=result.frozen_confirmatory_config_path,
        )
        if not verification.valid:
            raise RuntimeError(
                "new preregistration freeze failed independent integrity verification: "
                f"errors={verification.errors}, missing={verification.missing_paths}, "
                f"added={verification.added_paths}, changed={verification.changed_paths}"
            )
    except Exception as exc:
        _failure(f"preregistration freeze failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "status": "PRE_REGISTRATION_FROZEN",
                **result.as_dict(),
                "integrity_verified": True,
            },
            indent=2,
        )
    )


@preregistration_app.command("amend")
def amend_preregistration_command(
    parent_authority_directory: Annotated[
        Path,
        typer.Option(
            "--parent-authority-dir",
            "--parent-authority",
            help="Explicit immutable base-freeze or predecessor-amendment authority.",
        ),
    ],
    preregistration_path: Annotated[
        Path,
        typer.Option(
            "--amended-preregistration",
            help="Complete successor preregistration Markdown; never a delta.",
        ),
    ],
    primary_config_path: Annotated[
        Path,
        typer.Option(
            "--amended-primary-config",
            help="Complete successor primary-study YAML configuration.",
        ),
    ],
    confirmatory_config_path: Annotated[
        Path,
        typer.Option(
            "--amended-confirmatory-config",
            help="Complete successor confirmatory-study YAML configuration.",
        ),
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Single-line scientific reason for the amendment."),
    ],
    affected_hypotheses: Annotated[
        list[str],
        typer.Option(
            "--affected-hypothesis",
            help="Affected hypothesis identifier; repeat for every affected hypothesis.",
        ),
    ],
    affected_analyses: Annotated[
        list[str],
        typer.Option(
            "--affected-analysis",
            help="Affected analysis identifier; repeat for every affected analysis.",
        ),
    ],
    amendment_timestamp_utc: Annotated[
        str,
        typer.Option(
            "--amendment-timestamp-utc",
            help="Explicit amendment timestamp as ISO-8601 UTC ending in Z.",
        ),
    ],
    amendment_root: Annotated[
        Path,
        typer.Option(
            "--amendment-root",
            help="Parent directory for the new timestamped immutable successor authority.",
        ),
    ],
    finalization_predecessor_run_directory: Annotated[
        Path | None,
        typer.Option(
            "--finalization-predecessor-run-dir",
            "--finalization-predecessor",
            file_okay=False,
            help=(
                "Reserved only to reject mixing the retired finalization-successor path "
                "with bounded orphan recovery."
            ),
        ),
    ] = None,
    primary_recovery_authorization_json: Annotated[
        Path | None,
        typer.Option(
            "--primary-recovery-authorization-json",
            dir_okay=False,
            help=(
                "Exact JSON object authorizing one interrupted, unsealed primary recovery. "
                "Mutually exclusive with --finalization-predecessor-run-dir."
            ),
        ),
    ] = None,
    confirmatory_single_copy_checkpoint_storage: Annotated[
        bool,
        typer.Option(
            "--confirmatory-single-copy-checkpoint-storage",
            help=(
                "Bind the closed single-canonical-copy checkpoint policy; requires the "
                "primary-recovery authorization."
            ),
        ),
    ] = False,
    outcomes_inspected: Annotated[
        bool,
        typer.Option(
            "--outcomes-inspected",
            help="Declare explicitly that study outcomes were inspected before amendment.",
        ),
    ] = False,
    outcomes_not_inspected: Annotated[
        bool,
        typer.Option(
            "--outcomes-not-inspected",
            help="Declare explicitly that study outcomes were not inspected before amendment.",
        ),
    ] = False,
    outcomes_inspected_at_utc: Annotated[
        str | None,
        typer.Option(
            "--outcomes-inspected-at-utc",
            help="Required UTC timestamp when --outcomes-inspected is declared.",
        ),
    ] = None,
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
) -> None:
    """Publish a full immutable, parent-hash-linked preregistration successor."""

    if outcomes_inspected == outcomes_not_inspected:
        _failure("declare exactly one of --outcomes-inspected or --outcomes-not-inspected")
    if (
        primary_recovery_authorization_json is not None
        and finalization_predecessor_run_directory is not None
    ):
        _failure(
            "--primary-recovery-authorization-json and "
            "--finalization-predecessor-run-dir are mutually exclusive"
        )
    if finalization_predecessor_run_directory is not None:
        _failure(
            "--finalization-predecessor-run-dir is retired; use the bounded "
            "--primary-recovery-authorization-json path"
        )
    if confirmatory_single_copy_checkpoint_storage and primary_recovery_authorization_json is None:
        _failure(
            "--confirmatory-single-copy-checkpoint-storage requires "
            "--primary-recovery-authorization-json"
        )
    if primary_recovery_authorization_json is not None and not outcomes_inspected:
        _failure("--primary-recovery-authorization-json requires --outcomes-inspected")
    if primary_recovery_authorization_json is not None and outcomes_inspected_at_utc is None:
        _failure("--primary-recovery-authorization-json requires --outcomes-inspected-at-utc")
    root = project_root.resolve()
    try:
        timestamp = _parse_explicit_utc_timestamp(
            amendment_timestamp_utc, role="amendment timestamp"
        )
        inspected_at = (
            _parse_explicit_utc_timestamp(
                outcomes_inspected_at_utc,
                role="outcomes-inspected timestamp",
            )
            if outcomes_inspected_at_utc is not None
            else None
        )
        from histo_audit.workflows import (
            ConfirmatoryStoragePolicy,
            create_preregistration_amendment,
            verify_preregistration_amendment,
        )

        parent_authority = _resolve_from_root(root, parent_authority_directory)
        recovery_authorization: dict[str, Any] | None = None
        if primary_recovery_authorization_json is not None:
            authorization_path = _resolve_from_root(
                root,
                primary_recovery_authorization_json,
            )
            raw_authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            if not isinstance(raw_authorization, dict):
                raise ValueError(
                    "--primary-recovery-authorization-json must contain one JSON object"
                )
            recovery_authorization = cast(dict[str, Any], raw_authorization)
        amendment_kwargs: dict[str, Any] = {
            "project_root": root,
            "parent_authority_directory": parent_authority,
            "amendment_root": _resolve_from_root(root, amendment_root),
            "preregistration_path": _resolve_from_root(root, preregistration_path),
            "primary_config_path": _resolve_from_root(root, primary_config_path),
            "confirmatory_config_path": _resolve_from_root(root, confirmatory_config_path),
            "reason": reason,
            "affected_hypotheses": affected_hypotheses,
            "affected_analyses": affected_analyses,
            "outcomes_inspected": outcomes_inspected,
            "outcomes_inspected_at": inspected_at,
            "timestamp": timestamp,
        }
        if recovery_authorization is not None:
            amendment_kwargs.update(
                {
                    "finalization_successor_authorization": None,
                    "primary_recovery_authorization": recovery_authorization,
                    "confirmatory_storage_policy": (
                        ConfirmatoryStoragePolicy()
                        if confirmatory_single_copy_checkpoint_storage
                        else None
                    ),
                }
            )
        result = create_preregistration_amendment(**amendment_kwargs)
        verification = verify_preregistration_amendment(result.amendment_directory)
        if not verification.valid:
            raise RuntimeError(
                "new preregistration amendment failed independent integrity verification: "
                + "; ".join(verification.errors)
            )
    except Exception as exc:
        _failure(f"preregistration amendment failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "authority_status": "amended",
                **result.as_dict(),
                "integrity_verified": True,
            },
            indent=2,
        )
    )


@preregistration_app.command("verify-amendment")
def verify_preregistration_amendment_command(
    amendment_directory: Annotated[
        Path,
        typer.Option(
            "--amendment-dir",
            help="Immutable preregistration-amendment authority to verify recursively.",
        ),
    ],
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    max_chain_depth: Annotated[
        int,
        typer.Option(
            "--max-chain-depth",
            min=1,
            help="Maximum predecessor depth accepted during recursive verification.",
        ),
    ] = 64,
) -> None:
    """Verify an amendment bundle and every immutable parent authority."""

    from histo_audit.workflows import verify_preregistration_amendment

    root = project_root.resolve()
    directory = _resolve_from_root(root, amendment_directory)
    verification = verify_preregistration_amendment(
        directory,
        max_chain_depth=max_chain_depth,
    )
    if not verification.valid:
        _failure("preregistration amendment verification failed: " + "; ".join(verification.errors))
    typer.echo(
        json.dumps(
            {
                "authority_status": "verified_amendment",
                "amendment_directory": str(verification.amendment_directory),
                "chain_depth": verification.chain_depth,
                "artifact_root_sha256": verification.artifact_root_sha256,
                "sha256_manifest_sha256": verification.sha256_manifest_sha256,
                "parent_authority_directory": (
                    str(verification.parent_authority_directory)
                    if verification.parent_authority_directory is not None
                    else None
                ),
                "integrity_verified": True,
            },
            indent=2,
        )
    )


@preregistration_app.command("verify-resource-technical-successor")
def verify_resource_technical_successor_command(
    successor_directory: Annotated[
        Path,
        typer.Option(
            "--successor-dir",
            help="Published schema-v5 resource technical successor D.",
        ),
    ],
    expected_parent_authority_directory: Annotated[
        Path,
        typer.Option(
            "--expected-parent-authority-dir",
            help="Exact superseded schema-v4 resource authority C.",
        ),
    ],
    expected_artifact_root_sha256: Annotated[
        str,
        typer.Option(
            "--expected-artifact-root-sha256",
            help="Externally pinned artifact-root SHA-256 for D.",
        ),
    ],
    expected_sha256_manifest_sha256: Annotated[
        str,
        typer.Option(
            "--expected-sha256-manifest-sha256",
            help="Externally pinned checksum-manifest SHA-256 for D.",
        ),
    ],
    expected_authorization_sha256: Annotated[
        str,
        typer.Option(
            "--expected-authorization-sha256",
            help="Externally pinned canonical schema-v5 authorization SHA-256.",
        ),
    ],
    expected_intent_sha256: Annotated[
        str,
        typer.Option(
            "--expected-intent-sha256",
            help="Pre-mutation canonical publication-intent SHA-256.",
        ),
    ],
    expected_controller_process_id: Annotated[
        int,
        typer.Option(
            "--expected-controller-pid",
            min=1,
            help="PID of the publication controller that spawned this verifier.",
        ),
    ],
    verification_nonce: Annotated[
        str,
        typer.Option(
            "--verification-nonce",
            help="One-use 64-hex challenge generated by the publication controller.",
        ),
    ],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", file_okay=False),
    ] = Path("."),
) -> None:
    """Verify the exact effective D in a read-only fresh-process boundary."""

    from histo_audit.workflows import (
        verify_resource_bounded_technical_successor,
    )

    root = project_root.resolve()
    try:
        verification = verify_resource_bounded_technical_successor(
            _resolve_from_root(root, successor_directory),
            expected_parent_authority_directory=_resolve_from_root(
                root,
                expected_parent_authority_directory,
            ),
            expected_artifact_root_sha256=expected_artifact_root_sha256,
            expected_sha256_manifest_sha256=(expected_sha256_manifest_sha256),
            expected_authorization_sha256=expected_authorization_sha256,
            expected_intent_sha256=expected_intent_sha256,
            expected_controller_process_id=expected_controller_process_id,
            verification_nonce=verification_nonce,
        )
    except Exception as exc:
        _failure(f"resource technical successor verification failed: {type(exc).__name__}: {exc}")
    typer.echo(json.dumps(verification.as_dict(), indent=2))


@audit_app.command("original")
def audit_original_command(
    ctx: typer.Context,
    manifest_path: Annotated[
        Path,
        typer.Option(
            "--manifest", help="Uncorrupted original-label CSV, JSON, or Parquet manifest."
        ),
    ],
    feature_cache_path: Annotated[
        Path,
        typer.Option(
            "--feature-cache",
            help="NPZ containing exactly aligned `features` and `sample_ids` arrays.",
        ),
    ],
    final_reference_groups_path: Annotated[
        Path,
        typer.Option(
            "--final-reference-groups",
            help="Newline-delimited final-reference group IDs that must be absent from audit rows.",
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-dir", help="New immutable exploratory-audit output directory."),
    ],
    dataset_path: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            help="Verified PanNuke release root; requires both real-evidence options.",
        ),
    ] = None,
    dataset_validation_json: Annotated[
        Path | None,
        typer.Option(
            "--dataset-validation-json",
            help="Full semantic-validation JSON for stage-eligible tracked execution.",
        ),
    ] = None,
    duplicate_audit_json: Annotated[
        Path | None,
        typer.Option(
            "--duplicate-audit-json",
            help="Complete two-signal duplicate-audit JSON for tracked execution.",
        ),
    ] = None,
    confirmatory_run_directory: Annotated[
        Path | None,
        typer.Option(
            "--confirmatory-run-dir",
            help="Sealed registry-backed CONFIRMATORY_COMPLETE run for stage execution.",
        ),
    ] = None,
    feature_cache_provenance_json: Annotated[
        Path | None,
        typer.Option(
            "--feature-cache-provenance-json",
            help="Exact cache/order/encoder/freeze sidecar bound to the confirmatory run.",
        ),
    ] = None,
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    class_order: Annotated[
        str,
        typer.Option("--class-order", help="Comma-separated fixed numeric class order."),
    ] = "0,1,2,3,4",
    n_splits: Annotated[int, typer.Option("--n-splits", min=2)] = 5,
    split_seed: Annotated[int, typer.Option("--split-seed")] = 223,
    model_seed: Annotated[int, typer.Option("--model-seed")] = 227,
    representation: Annotated[str, typer.Option("--representation")] = "frozen_features",
    method: Annotated[str, typer.Option("--method")] = "self_confidence",
    top_count_overall: Annotated[int, typer.Option("--top-count", min=1)] = 100,
    top_count_per_class: Annotated[int, typer.Option("--top-per-class", min=1)] = 20,
    top_count_per_tissue: Annotated[int, typer.Option("--top-per-tissue", min=1)] = 20,
    l2: Annotated[float, typer.Option("--l2", min=0.0)] = 1.0e-2,
    max_iter: Annotated[int, typer.Option("--max-iter", min=1)] = 400,
) -> None:
    """Rank unmodified original labels using strictly group-safe OOF evidence."""

    from histo_audit.external_validation import load_original_audit_feature_cache
    from histo_audit.workflows import audit_original_labels
    from histo_audit.workflows.tracked_original_audit import run_tracked_original_label_audit

    root = project_root.resolve()
    manifest = _resolve_from_root(root, manifest_path)
    cache_path = _resolve_from_root(root, feature_cache_path)
    final_groups_file = _resolve_from_root(root, final_reference_groups_path)
    destination = _resolve_from_root(root, output_directory)
    supplied_real_evidence = (
        dataset_path,
        dataset_validation_json,
        duplicate_audit_json,
        confirmatory_run_directory,
        feature_cache_provenance_json,
    )
    tracked = all(value is not None for value in supplied_real_evidence)
    try:
        if any(value is not None for value in supplied_real_evidence) and not tracked:
            raise ValueError(
                "--dataset, --dataset-validation-json, --duplicate-audit-json, "
                "--confirmatory-run-dir, and --feature-cache-provenance-json must be "
                "supplied together; omit all five for a non-stage fixture workflow"
            )
        features, loaded_sample_ids = load_original_audit_feature_cache(cache_path)
        sample_ids = list(loaded_sample_ids)
        group_lines = final_groups_file.read_text(encoding="utf-8").splitlines()
        final_groups = tuple(line.strip() for line in group_lines if line.strip())
        if not final_groups:
            raise ValueError("final-reference group file contains no group IDs")
        if len(set(final_groups)) != len(final_groups):
            raise ValueError("final-reference group file contains duplicate group IDs")
        if tracked:
            assert dataset_path is not None
            assert dataset_validation_json is not None
            assert duplicate_audit_json is not None
            assert confirmatory_run_directory is not None
            assert feature_cache_provenance_json is not None
            fixture_only_parameters = (
                "class_order",
                "n_splits",
                "split_seed",
                "model_seed",
                "representation",
                "method",
                "top_count_overall",
                "top_count_per_class",
                "top_count_per_tissue",
                "l2",
                "max_iter",
            )
            explicit_overrides = [
                name
                for name in fixture_only_parameters
                if getattr(ctx.get_parameter_source(name), "name", None) != "DEFAULT"
            ]
            if explicit_overrides:
                raise ValueError(
                    "stage-ready original audit rejects CLI model/selection overrides; "
                    f"frozen controls are mandatory: {explicit_overrides}"
                )
            result = run_tracked_original_label_audit(
                manifest,
                features,
                sample_ids,
                destination,
                dataset_path=_resolve_from_root(root, dataset_path),
                dataset_validation_json=_resolve_from_root(root, dataset_validation_json),
                duplicate_audit_json=_resolve_from_root(root, duplicate_audit_json),
                confirmatory_run_directory=_resolve_from_root(root, confirmatory_run_directory),
                feature_cache_path=cache_path,
                feature_cache_provenance_json=_resolve_from_root(
                    root, feature_cache_provenance_json
                ),
                final_reference_groups_path=final_groups_file,
                project_root=root,
                final_reference_group_ids=final_groups,
            )
        else:
            parsed_class_order = tuple(
                int(value.strip()) for value in class_order.split(",") if value.strip()
            )
            if len(parsed_class_order) < 2 or len(set(parsed_class_order)) != len(
                parsed_class_order
            ):
                raise ValueError("class order must contain at least two unique numeric labels")
            result = audit_original_labels(
                manifest,
                features,
                sample_ids,
                destination,
                final_reference_group_ids=final_groups,
                class_order=parsed_class_order,
                n_splits=n_splits,
                split_seed=split_seed,
                model_seed=model_seed,
                representation=representation,
                method=method,
                top_count_overall=top_count_overall,
                top_count_per_class=top_count_per_class,
                top_count_per_tissue=top_count_per_tissue,
                l2=l2,
                max_iter=max_iter,
            )
    except Exception as exc:
        _failure(f"original-label audit failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "workflow": "exploratory_original_label_audit",
                "result": "completed",
                "automatic_source_annotation_modification": False,
                "study_outcome_eligible": tracked,
                "sealed_registry_backed": tracked,
                **result.as_dict(),
            },
            indent=2,
        )
    )


@external_app.command("build-review-package")
def build_review_package_command(
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Eligible real-data manifest with reviewer assets."),
    ],
    ranking_path: Annotated[
        Path,
        typer.Option("--ranking", help="Eligible group-safe original-label ranking table."),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-dir", help="New blinded reviewer-facing package directory."),
    ],
    private_unblinding_key_path: Annotated[
        Path,
        typer.Option(
            "--private-key",
            help="New private linkage CSV path outside the reviewer-facing package.",
        ),
    ],
    audit_run_directory: Annotated[
        Path | None,
        typer.Option(
            "--audit-run-dir",
            help="Sealed registry-backed original-audit run; omit for non-stage fixtures.",
        ),
    ] = None,
    dataset_path: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            help="Exact verified PanNuke release bound to the original-audit run.",
        ),
    ] = None,
    dataset_validation_json: Annotated[
        Path | None,
        typer.Option(
            "--dataset-validation-json",
            help="Exact full semantic-validation JSON bound to the original-audit run.",
        ),
    ] = None,
    duplicate_audit_json: Annotated[
        Path | None,
        typer.Option(
            "--duplicate-audit-json",
            help="Exact complete duplicate-audit JSON bound to the original-audit run.",
        ),
    ] = None,
    confirmatory_run_directory: Annotated[
        Path | None,
        typer.Option(
            "--confirmatory-run-dir",
            help="Exact sealed CONFIRMATORY_COMPLETE run bound to the original audit.",
        ),
    ] = None,
    feature_cache_provenance_json: Annotated[
        Path | None,
        typer.Option(
            "--feature-cache-provenance-json",
            help="Exact cache provenance sidecar bound to the original audit.",
        ),
    ] = None,
    project_root: Annotated[Path, typer.Option("--project-root", file_okay=False)] = Path("."),
    top_count: Annotated[int, typer.Option("--top", min=1)] = 100,
    random_count: Annotated[int, typer.Option("--random", min=1)] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 509,
) -> None:
    """Build a package; emit readiness only for a verified sealed real-data audit."""

    from histo_audit.external_validation import (
        build_blinded_review_package,
        validate_blinded_review_package,
        verify_external_validation_eligibility,
    )

    root = project_root.resolve()
    result = None
    validation = None
    eligibility = None
    try:
        eligibility_options = (
            audit_run_directory,
            dataset_path,
            dataset_validation_json,
            duplicate_audit_json,
            confirmatory_run_directory,
            feature_cache_provenance_json,
        )
        eligibility_requested = any(value is not None for value in eligibility_options)
        if eligibility_requested and not all(value is not None for value in eligibility_options):
            raise ValueError(
                "--audit-run-dir, --dataset, --dataset-validation-json, and "
                "--duplicate-audit-json, --confirmatory-run-dir, and "
                "--feature-cache-provenance-json must be supplied together; omit all six "
                "for a non-stage fixture/package workflow"
            )
        resolved_manifest = _resolve_from_root(root, manifest_path)
        resolved_ranking = _resolve_from_root(root, ranking_path)
        if eligibility_requested:
            assert audit_run_directory is not None
            assert dataset_path is not None
            assert dataset_validation_json is not None
            assert duplicate_audit_json is not None
            assert confirmatory_run_directory is not None
            assert feature_cache_provenance_json is not None
            eligibility = verify_external_validation_eligibility(
                audit_run_directory=_resolve_from_root(root, audit_run_directory),
                dataset_path=_resolve_from_root(root, dataset_path),
                manifest_path=resolved_manifest,
                dataset_validation_json=_resolve_from_root(root, dataset_validation_json),
                duplicate_audit_json=_resolve_from_root(root, duplicate_audit_json),
                ranking_path=resolved_ranking,
                confirmatory_run_directory=_resolve_from_root(root, confirmatory_run_directory),
                feature_cache_provenance_json=_resolve_from_root(
                    root, feature_cache_provenance_json
                ),
            )
            if not eligibility.eligible:
                raise ValueError(
                    f"external-validation eligibility verification failed: {eligibility.errors}"
                )
        result = build_blinded_review_package(
            resolved_manifest,
            resolved_ranking,
            _resolve_from_root(root, output_directory),
            top_count=top_count,
            random_count=random_count,
            seed=seed,
            private_unblinding_key_path=_resolve_from_root(root, private_unblinding_key_path),
            study_outcome_eligible=eligibility is not None,
            eligibility_evidence=(eligibility.package_evidence() if eligibility else None),
        )
        validation = validate_blinded_review_package(
            result.package_directory,
            private_unblinding_key_path=result.private_unblinding_key_csv,
        )
        if not validation.valid or not validation.private_linkage_validated:
            raise RuntimeError(
                f"generated review package failed validation: {validation.as_dict()}"
            )
        if eligibility is not None:
            assert audit_run_directory is not None
            assert dataset_path is not None
            assert dataset_validation_json is not None
            assert duplicate_audit_json is not None
            assert confirmatory_run_directory is not None
            assert feature_cache_provenance_json is not None
            reverified = verify_external_validation_eligibility(
                audit_run_directory=_resolve_from_root(root, audit_run_directory),
                dataset_path=_resolve_from_root(root, dataset_path),
                manifest_path=resolved_manifest,
                dataset_validation_json=_resolve_from_root(root, dataset_validation_json),
                duplicate_audit_json=_resolve_from_root(root, duplicate_audit_json),
                ranking_path=resolved_ranking,
                confirmatory_run_directory=_resolve_from_root(root, confirmatory_run_directory),
                feature_cache_provenance_json=_resolve_from_root(
                    root, feature_cache_provenance_json
                ),
            )
            if not reverified.eligible:
                raise RuntimeError(
                    "external-validation evidence changed during package construction: "
                    f"{reverified.errors}"
                )
            if reverified.package_evidence() != eligibility.package_evidence():
                raise RuntimeError(
                    "external-validation evidence digests changed during package construction"
                )
            eligibility = reverified
    except Exception as exc:
        if result is not None:
            shutil.rmtree(result.package_directory, ignore_errors=True)
            result.private_unblinding_key_csv.unlink(missing_ok=True)
        _failure(f"review-package build failed: {type(exc).__name__}: {exc}")
    assert result is not None
    assert validation is not None
    payload = {
        **result.as_dict(),
        "validation": validation.as_dict(),
        "study_outcome_eligible": eligibility is not None,
    }
    if eligibility is not None:
        payload = {
            "status": "EXTERNAL_VALIDATION_READY",
            **payload,
            "eligibility": eligibility.as_dict(),
        }
    else:
        payload = {
            "workflow": "blinded_review_package_fixture_or_non_stage_build",
            "result": "package_built_and_structurally_validated",
            **payload,
        }
    typer.echo(json.dumps(payload, indent=2))


@report_app.command("build")
def build_report_command(
    metrics: Annotated[
        Path | None,
        typer.Option("--metrics", "-m", exists=True, dir_okay=False),
    ] = None,
    run_directory: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            exists=True,
            file_okay=False,
            help="Sealed tracked run to verify and rebuild from.",
        ),
    ] = None,
    predictions: Annotated[
        Path | None,
        typer.Option("--predictions", exists=True, dir_okay=False),
    ] = None,
    representation_example: Annotated[
        Path | None,
        typer.Option("--representation-example", exists=True, dir_okay=False),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Defaults to metrics directory; sealed sources require an external directory.",
        ),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    """Build sourced Markdown and static HTML from strict metrics JSON."""

    from histo_audit.reporting import build_synthetic_report
    from histo_audit.utils.run_tracking import sealed_run_ancestor, verify_run_integrity

    selected_metrics = metrics.resolve() if metrics is not None else None
    selected_predictions = predictions.resolve() if predictions is not None else None
    selected_representation = (
        representation_example.resolve() if representation_example is not None else None
    )
    verified_run: Path | None = None
    verification = None
    if run_directory is not None:
        verified_run = run_directory.resolve()
        verification = verify_run_integrity(verified_run)
        if not verification.valid:
            _failure(
                "source run integrity verification failed: "
                f"missing={list(verification.missing_paths)}, "
                f"added={list(verification.added_paths)}, "
                f"changed={list(verification.changed_paths)}, "
                f"errors={list(verification.errors)}"
            )
        run_metrics = (verified_run / "metrics.json").resolve()
        run_predictions = (verified_run / "oof_predictions.npz").resolve()
        run_representation = (verified_run / "target_representation_example.npz").resolve()
        if not run_metrics.is_file():
            _failure(f"verified source run lacks metrics.json: {verified_run}")
        if selected_metrics is not None and selected_metrics != run_metrics:
            _failure("--metrics conflicts with the verified --run-dir metrics artifact")
        if selected_predictions is not None and selected_predictions != run_predictions:
            _failure("--predictions conflicts with the verified --run-dir artifact")
        if selected_representation is not None and selected_representation != run_representation:
            _failure("--representation-example conflicts with the verified --run-dir artifact")
        selected_metrics = run_metrics
        selected_predictions = run_predictions if run_predictions.is_file() else None
        selected_representation = run_representation if run_representation.is_file() else None
    if selected_metrics is None:
        _failure("provide --metrics or --run-dir")
    sealed_source = sealed_run_ancestor(selected_metrics)
    if sealed_source is not None and sealed_source != verified_run:
        verification = verify_run_integrity(sealed_source)
        if not verification.valid:
            _failure("metrics belong to a sealed run that failed integrity verification")
        verified_run = sealed_source
        bound_predictions = (sealed_source / "oof_predictions.npz").resolve()
        bound_representation = (sealed_source / "target_representation_example.npz").resolve()
        if selected_predictions is not None and selected_predictions != bound_predictions:
            _failure("external --predictions are not checksum-bound to the sealed metrics run")
        if selected_representation is not None and selected_representation != bound_representation:
            _failure(
                "external --representation-example is not checksum-bound to the sealed metrics run"
            )
        selected_predictions = bound_predictions if bound_predictions.is_file() else None
        selected_representation = bound_representation if bound_representation.is_file() else None
    selected_run_id = run_id
    if verified_run is not None and verification is not None:
        if run_id is not None and run_id != verification.run_id:
            _failure("--run-id conflicts with the verified sealed-run identity")
        selected_run_id = verification.run_id

    try:
        artifacts = build_synthetic_report(
            selected_metrics,
            output_directory=output_directory,
            run_id=selected_run_id,
            predictions_path=selected_predictions,
            representation_example_path=selected_representation,
        )
    except Exception as exc:
        _failure(f"report build failed: {type(exc).__name__}: {exc}")
    typer.echo(
        json.dumps(
            {
                "markdown": str(artifacts.markdown_path.resolve()),
                "html": str(artifacts.html_path.resolve()),
                "metrics": str(artifacts.metrics_path.resolve()),
                "metrics_sha256": artifacts.metrics_sha256,
                "figures": [str(path.resolve()) for path in artifacts.figure_paths],
                "figure_sources": (
                    str(artifacts.figure_manifest_path.resolve())
                    if artifacts.figure_manifest_path is not None
                    else None
                ),
                "source_run_integrity": (
                    {
                        "run_directory": str(verified_run),
                        "valid": verification.valid,
                        "artifact_root_sha256": verification.actual_root_sha256,
                    }
                    if verified_run is not None and verification is not None
                    else None
                ),
            },
            indent=2,
        )
    )


__all__ = ["app"]
