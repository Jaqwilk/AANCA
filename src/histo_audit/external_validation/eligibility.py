"""Fail-closed eligibility checks for the external-validation readiness stage.

Structural package validation is intentionally separate from study-stage
eligibility.  Only a completed, registry-backed and cryptographically sealed
original-label audit over a fully validated PanNuke release can make a package
eligible for ``EXTERNAL_VALIDATION_READY``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from histo_audit.config import config_sha256
from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.confirmatory_completion import REAL_CONFIRMATORY_ARTIFACT_SCOPE
from histo_audit.experiment.study_contracts import validate_frozen_confirmatory_config
from histo_audit.pannuke.models import (
    OFFICIAL_METRICS_CLASS_MAPPING,
    OFFICIAL_RELEASE_FOLD_IDS,
)
from histo_audit.utils.run_tracking import (
    IMMUTABLE_MARKER,
    STATUS_FILENAME,
    sha256_file,
    sha256_path,
    verify_run_integrity,
)

ELIGIBILITY_FILENAME = "external_validation_eligibility.json"
ORIGINAL_AUDIT_EXPERIMENT_NAME = "original_label_audit"
CONFIRMATORY_EXPERIMENT_NAME = "confirmatory_study"
FEATURE_PROVENANCE_FILENAME = "frozen_feature_provenance.json"
ORIGINAL_AUDIT_SELECTION_FILENAME = "original_audit_selection.json"
ORIGINAL_AUDIT_FEATURE_SCOPE = "real_pannuke_original_audit_feature_cache"
ORIGINAL_AUDIT_CLASS_ORDER = (0, 1, 2, 3, 4)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESNET18_IMAGENET1K_V1_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


@dataclass(frozen=True, slots=True)
class ExternalValidationEligibilityResult:
    """Cryptographic audit-chain result used by the CLI stage gate."""

    eligible: bool
    audit_run_directory: Path
    run_id: str | None
    artifact_root_sha256: str | None
    dataset_sha256: str | None
    manifest_sha256: str | None
    dataset_validation_sha256: str | None
    duplicate_audit_sha256: str | None
    ranking_sha256: str | None
    confirmatory_run_id: str | None
    confirmatory_artifact_root_sha256: str | None
    confirmatory_completion_sha256: str | None
    feature_cache_provenance_sha256: str | None
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence without inventing a completion stage."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }

    def package_evidence(self) -> dict[str, Any]:
        """Return the non-identifying digest summary safe for package metadata."""

        if not self.eligible or self.run_id is None or self.artifact_root_sha256 is None:
            raise ValueError("ineligible evidence cannot be attached to a stage-eligible package")
        return {
            "schema_version": 1,
            "audit_run_id": self.run_id,
            "audit_artifact_root_sha256": self.artifact_root_sha256,
            "dataset_sha256": self.dataset_sha256,
            "manifest_sha256": self.manifest_sha256,
            "dataset_validation_sha256": self.dataset_validation_sha256,
            "duplicate_audit_sha256": self.duplicate_audit_sha256,
            "ranking_sha256": self.ranking_sha256,
            "confirmatory_run_id": self.confirmatory_run_id,
            "confirmatory_artifact_root_sha256": self.confirmatory_artifact_root_sha256,
            "confirmatory_completion_sha256": self.confirmatory_completion_sha256,
            "feature_cache_provenance_sha256": self.feature_cache_provenance_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenOriginalAuditSelection:
    """Only model controls accepted by the stage-eligible original-label audit."""

    scenario_id: str
    representation_id: str
    model_seed: int
    risk_method: str
    n_splits: int
    split_seed: int
    cache_provenance_id: str
    classifier_id: str
    l2: float
    max_iter: int
    class_weight: str
    selection_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OriginalAuditUpstreamVerification:
    """Verification of the sealed confirmatory and exact feature-cache chain."""

    eligible: bool
    confirmatory_run_directory: Path
    confirmatory_run_id: str | None
    confirmatory_artifact_root_sha256: str | None
    completion_evidence_sha256: str | None
    confirmatory_plan_sha256: str | None
    resolved_config_sha256: str | None
    feature_provenance_artifact_sha256: str | None
    selection_artifact_sha256: str | None
    selected_outer_fold: int | None
    feature_cache_provenance_path: Path
    feature_cache_provenance_sha256: str | None
    feature_cache_sha256: str | None
    manifest_sha256: str | None
    sample_order_sha256: str | None
    audit_sample_order_sha256: str | None
    audit_sample_ids: tuple[str, ...]
    selection: FrozenOriginalAuditSelection | None
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


def _mapping(path: Path, *, role: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{role} is missing or invalid JSON: {exc}")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{role} must be a JSON object")
        return {}
    return value


def _normal_sha(value: Any, *, role: str, errors: list[str]) -> str | None:
    digest = str(value).lower() if value is not None else ""
    if not _SHA256.fullmatch(digest):
        errors.append(f"{role} is not a SHA-256 digest")
        return None
    return digest


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _record_path(
    evidence: Mapping[str, Any],
    role: str,
    supplied: Path,
    *,
    directory: bool,
    errors: list[str],
) -> str | None:
    raw = evidence.get(role)
    if not isinstance(raw, Mapping):
        errors.append(f"eligibility evidence lacks the {role!r} record")
        return None
    expected_path = Path(str(raw.get("path", ""))).resolve()
    if expected_path != supplied:
        errors.append(f"supplied {role} path differs from sealed eligibility evidence")
        return None
    expected_digest = _normal_sha(raw.get("sha256"), role=f"sealed {role} hash", errors=errors)
    if expected_digest is None:
        return None
    try:
        actual_digest = sha256_path(supplied) if directory else sha256_file(supplied)
    except OSError as exc:
        errors.append(f"cannot hash supplied {role}: {exc}")
        return None
    if actual_digest != expected_digest:
        errors.append(f"supplied {role} hash differs from sealed eligibility evidence")
    return actual_digest


def _validate_run_registry(
    run: Path,
    *,
    run_id: str | None,
    dataset_sha256: str | None,
    manifest_sha256: str | None,
    errors: list[str],
) -> None:
    registry = run.parent / "registry.csv"
    try:
        with registry.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    except OSError as exc:
        errors.append(f"original-audit run registry is missing or unreadable: {exc}")
        return
    if len(rows) != 1:
        errors.append("original-audit run does not have exactly one standard registry record")
        return
    row = rows[0]
    if row.get("status") != "completed":
        errors.append("original-audit standard registry record is not completed")
    if row.get("experiment_name") != ORIGINAL_AUDIT_EXPERIMENT_NAME:
        errors.append("original-audit standard registry record has an unexpected experiment")
    if Path(str(row.get("run_path", ""))).resolve() != run:
        errors.append("original-audit standard registry path differs from the sealed run")
    if row.get("dataset_sha256") != dataset_sha256:
        errors.append("original-audit standard registry does not bind the supplied dataset")
    if row.get("manifest_sha256") != manifest_sha256:
        errors.append("original-audit standard registry does not bind the supplied manifest")


def _validate_confirmatory_registry(
    run: Path,
    *,
    run_id: str | None,
    config_sha256_value: str | None,
    dataset_sha256: str | None,
    manifest_sha256: str | None,
    errors: list[str],
) -> None:
    registry = run.parent / "registry.csv"
    try:
        with registry.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    except OSError as exc:
        errors.append(f"confirmatory run registry is missing or unreadable: {exc}")
        return
    if len(rows) != 1:
        errors.append("confirmatory run does not have exactly one standard registry record")
        return
    row = rows[0]
    expected = {
        "status": "completed",
        "experiment_name": CONFIRMATORY_EXPERIMENT_NAME,
        "config_sha256": config_sha256_value,
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(f"confirmatory standard registry {field} differs from sealed evidence")
    if Path(str(row.get("run_path", ""))).resolve() != run:
        errors.append("confirmatory standard registry path differs from the sealed run")


def ordered_sample_ids_sha256(sample_ids: list[str] | tuple[str, ...]) -> str:
    """Hash an exact ordered identifier sequence with the project's canonical JSON rule."""

    identifiers = [str(value) for value in sample_ids]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be non-empty and unique")
    if any(not value.strip() for value in identifiers):
        raise ValueError("sample IDs must not contain empty values")
    return canonical_sha256(identifiers)


def load_original_audit_feature_cache(
    feature_cache_path: str | Path,
) -> tuple[NDArray[np.generic], tuple[str, ...]]:
    """Safely load one immutable frozen-feature cache used by original-label auditing."""

    source = Path(feature_cache_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"original-audit feature cache does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as payload:
            value_keys = [key for key in ("features", "embeddings", "values") if key in payload]
            if len(value_keys) != 1 or "sample_ids" not in payload:
                raise ValueError(
                    "feature cache must contain sample_ids and exactly one of features, "
                    "embeddings, or values"
                )
            values = np.asarray(payload[value_keys[0]])
            raw_ids = np.asarray(payload["sample_ids"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot safely load original-audit feature cache: {exc}") from exc
    if values.ndim != 2 or not values.shape[0] or not np.issubdtype(values.dtype, np.number):
        raise ValueError("feature cache values must be a non-empty numeric 2-D matrix")
    if not np.isfinite(values).all():
        raise ValueError("feature cache contains non-finite values")
    if raw_ids.ndim != 1 or raw_ids.shape[0] != values.shape[0]:
        raise ValueError("feature cache sample_ids must align exactly with feature rows")
    if raw_ids.dtype.kind not in {"S", "U"}:
        raise ValueError("feature cache sample_ids must use a string dtype")
    if raw_ids.dtype.kind == "S":
        try:
            identifiers = tuple(bytes(value).decode("utf-8") for value in raw_ids)
        except UnicodeDecodeError as exc:
            raise ValueError("feature cache byte sample_ids must be valid UTF-8") from exc
    else:
        identifiers = tuple(str(value) for value in raw_ids)
    ordered_sample_ids_sha256(identifiers)
    return values, identifiers


def _read_yaml_mapping(path: Path, *, role: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"{role} is missing or invalid YAML: {exc}")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{role} must be a YAML mapping")
        return {}
    return value


def _parse_frozen_original_selection(
    config: Mapping[str, Any], *, errors: list[str]
) -> tuple[FrozenOriginalAuditSelection | None, Mapping[str, Any]]:
    raw = config.get("original_audit_selection")
    data = config.get("data")
    if not isinstance(raw, Mapping) or not isinstance(data, Mapping):
        errors.append("frozen confirmatory config lacks original_audit_selection/data")
        return None, {}
    classifier = raw.get("classifier")
    parameters = classifier.get("parameters") if isinstance(classifier, Mapping) else None
    if not isinstance(classifier, Mapping) or not isinstance(parameters, Mapping):
        errors.append("frozen original-audit selection lacks classifier parameters")
        return None, {}
    raw_cache_records = config.get("cache_provenance")
    if not isinstance(raw_cache_records, list):
        errors.append("frozen confirmatory config lacks cache_provenance records")
        return None, {}
    cache_identifier = str(raw.get("cache_provenance_id", ""))
    cache_matches = [
        item
        for item in raw_cache_records
        if isinstance(item, Mapping) and str(item.get("id", "")) == cache_identifier
    ]
    if len(cache_matches) != 1:
        errors.append("frozen original-audit cache_provenance_id is not uniquely defined")
        return None, {}
    cache = cache_matches[0]
    try:
        selection = FrozenOriginalAuditSelection(
            scenario_id=str(raw["scenario_id"]),
            representation_id=str(raw["representation_id"]),
            model_seed=int(raw["model_seed"]),
            risk_method=str(raw["risk_method"]),
            n_splits=int(raw["n_splits"]),
            split_seed=int(data["split_seed"]),
            cache_provenance_id=cache_identifier,
            classifier_id=str(classifier["id"]),
            l2=float(parameters["l2"]),
            max_iter=int(parameters["max_iter"]),
            class_weight=str(parameters["class_weight"]),
            selection_sha256=canonical_sha256(raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"frozen original-audit selection is incomplete: {exc}")
        return None, cache
    if (
        not selection.scenario_id.strip()
        or not selection.representation_id.strip()
        or selection.model_seed < 0
        or selection.n_splits < 2
        or not selection.cache_provenance_id.strip()
        or selection.classifier_id != "multinomial_logistic_regression"
        or not math.isfinite(selection.l2)
        or selection.l2 < 0.0
        or selection.max_iter < 1
        or selection.class_weight != "balanced"
    ):
        errors.append("frozen original-audit selection contains unsupported model controls")
    return selection, cache


def _valid_completed_confirmatory(
    completion: Mapping[str, Any], reconciliation: Mapping[str, Any], *, errors: list[str]
) -> None:
    if completion.get("schema_version") != 1:
        errors.append("confirmatory completion evidence schema is unsupported")
    expected = {
        "completion_stage": "CONFIRMATORY_COMPLETE",
        "study_outcome_eligible": True,
        "artifact_scope": REAL_CONFIRMATORY_ARTIFACT_SCOPE,
        "reconciliation_status": "passed",
        "fold_rotation_complete": True,
        "primary_run_sealed": True,
        "primary_run_registry_backed": True,
        "primary_completion_stage": "PRIMARY_STUDY_COMPLETE",
        "confirmatory_execution_gate_status": "passed",
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            errors.append(f"confirmatory completion {field} does not prove eligible completion")
    required = completion.get("required_cell_count")
    completed = completion.get("completed_required_cell_count")
    if (
        not isinstance(required, int)
        or isinstance(required, bool)
        or required < 1
        or completed != required
        or completion.get("failed_required_cell_count") != 0
    ):
        errors.append("confirmatory completion does not resolve every required cell")
    if (
        reconciliation.get("status") != "passed"
        or reconciliation.get("fold_rotation_complete") is not True
    ):
        errors.append("confirmatory reconciliation did not pass all fold rotations")
    for field in (
        "freeze_artifact_root_sha256",
        "freeze_manifest_sha256",
        "preregistration_sha256",
        "frozen_primary_config_sha256",
        "frozen_confirmatory_config_sha256",
        "primary_config_semantic_sha256",
        "confirmatory_config_semantic_sha256",
        "pilot_artifact_root_sha256",
        "dataset_sha256",
        "manifest_sha256",
        "duplicate_audit_sha256",
        "pathology_encoder_audit_sha256",
        "source_tree_root_sha256",
        "primary_artifact_root_sha256",
        "primary_completion_evidence_sha256",
        "primary_reconciliation_sha256",
    ):
        _normal_sha(completion.get(field), role=f"confirmatory binding {field}", errors=errors)
    if not str(completion.get("primary_run_id", "")).strip():
        errors.append("confirmatory completion lacks its sealed primary-run identity")
    if (
        tuple(completion.get("planned_outer_folds", ())) != OFFICIAL_RELEASE_FOLD_IDS
        or tuple(completion.get("completed_outer_folds", ())) != OFFICIAL_RELEASE_FOLD_IDS
    ):
        errors.append("confirmatory completion does not cover all three official folds")


def verify_original_audit_upstream(
    *,
    confirmatory_run_directory: str | Path,
    feature_cache_path: str | Path,
    feature_cache_provenance_json: str | Path,
    manifest_path: str | Path,
    final_reference_groups_path: str | Path,
) -> OriginalAuditUpstreamVerification:
    """Fail closed unless the exact cache is selected by sealed confirmatory evidence."""

    run = Path(confirmatory_run_directory).resolve()
    cache_path = Path(feature_cache_path).resolve()
    sidecar_path = Path(feature_cache_provenance_json).resolve()
    manifest = Path(manifest_path).resolve()
    final_groups = Path(final_reference_groups_path).resolve()
    errors: list[str] = []
    integrity = verify_run_integrity(run)
    if not integrity.valid or not integrity.registry_record_present:
        errors.append(f"confirmatory run integrity verification failed: {integrity.errors}")
    if integrity.run_id != run.name:
        errors.append("confirmatory run identity differs from its sealed directory name")
    status = _mapping(run / STATUS_FILENAME, role="confirmatory run status", errors=errors)
    marker = _mapping(run / IMMUTABLE_MARKER, role="confirmatory immutable marker", errors=errors)
    provenance = _mapping(
        run / "run_provenance.json", role="confirmatory provenance", errors=errors
    )
    if status.get("status") != "completed" or marker.get("status") != "completed":
        errors.append("confirmatory run was not sealed as completed")
    if provenance.get("experiment_name") != CONFIRMATORY_EXPERIMENT_NAME:
        errors.append("sealed upstream run is not a confirmatory study")

    completion_path = run / "completion_evidence.json"
    reconciliation_path = run / "reconciliation.json"
    plan_path = run / "matrix_plan.json"
    resolved_config_path = run / "resolved_config.yaml"
    execution_controls_path = run / "execution_controls.json"
    sealed_feature_path = run / FEATURE_PROVENANCE_FILENAME
    selection_artifact_path = run / ORIGINAL_AUDIT_SELECTION_FILENAME
    completion = _mapping(completion_path, role="confirmatory completion", errors=errors)
    reconciliation = _mapping(
        reconciliation_path, role="confirmatory reconciliation", errors=errors
    )
    plan = _mapping(plan_path, role="confirmatory matrix plan", errors=errors)
    sealed_features = _mapping(
        sealed_feature_path, role="confirmatory frozen-feature provenance", errors=errors
    )
    selection_artifact = _mapping(
        selection_artifact_path,
        role="confirmatory original-audit selection",
        errors=errors,
    )
    execution_controls = _mapping(
        execution_controls_path,
        role="confirmatory execution controls",
        errors=errors,
    )
    _valid_completed_confirmatory(completion, reconciliation, errors=errors)

    raw_config = _read_yaml_mapping(
        resolved_config_path, role="confirmatory resolved config", errors=errors
    )
    validated_config: Mapping[str, Any] = {}
    if raw_config:
        try:
            validated_config = validate_frozen_confirmatory_config(raw_config)
        except ValueError as exc:
            errors.append(f"confirmatory resolved config is not frozen-valid: {exc}")
    semantic_config_sha = config_sha256(validated_config) if validated_config else None
    if semantic_config_sha is not None:
        if completion.get("matrix_config_sha256") != semantic_config_sha:
            errors.append("confirmatory completion does not bind the resolved frozen config")
        if completion.get("confirmatory_config_semantic_sha256") != semantic_config_sha:
            errors.append("confirmatory completion names a different frozen config semantic hash")
        if plan.get("config_sha256") != semantic_config_sha:
            errors.append("confirmatory matrix plan does not bind the resolved frozen config")
        if provenance.get("config_sha256") != semantic_config_sha:
            errors.append("confirmatory run provenance does not bind the resolved frozen config")
    selection, frozen_cache_requirements = _parse_frozen_original_selection(
        validated_config, errors=errors
    )
    raw_frozen_selection = (
        validated_config.get("original_audit_selection") if validated_config else None
    )
    if (
        selection_artifact.get("schema_version") != 1
        or selection_artifact.get("status") != "completed"
    ):
        errors.append("confirmatory original-audit selection artifact is incomplete")
    if semantic_config_sha is not None:
        for field in (
            "confirmatory_config_semantic_sha256",
            "matrix_plan_config_sha256",
        ):
            if selection_artifact.get(field) != semantic_config_sha:
                errors.append(f"confirmatory original-audit selection {field} is invalid")
        if sealed_features.get("confirmatory_config_semantic_sha256") != semantic_config_sha:
            errors.append("frozen-feature provenance differs from the confirmatory config")
        if sealed_features.get("matrix_plan_config_sha256") != semantic_config_sha:
            errors.append("frozen-feature provenance differs from the confirmatory plan")
    if not isinstance(raw_frozen_selection, Mapping):
        errors.append("resolved confirmatory config lacks its original-audit selection")
    else:
        if selection_artifact.get("selection") != raw_frozen_selection:
            errors.append("sealed original-audit selection differs from the frozen config")
        if selection_artifact.get("selection_semantic_sha256") != canonical_sha256(
            raw_frozen_selection
        ):
            errors.append("sealed original-audit selection semantic hash is invalid")
        if execution_controls.get("original_audit_selection") != raw_frozen_selection:
            errors.append("confirmatory execution controls changed the frozen audit selection")
    if selection_artifact.get("execution_controls_binding_sha256") != execution_controls.get(
        "binding_sha256"
    ):
        errors.append("sealed original-audit selection has an invalid execution-control binding")
    if selection_artifact.get("frozen_cache_provenance_record") != frozen_cache_requirements:
        errors.append("sealed selected cache record differs from the frozen config")

    manifest_sha = None
    cache_sha = None
    sample_order_sha = None
    try:
        manifest_sha = sha256_file(manifest)
    except OSError as exc:
        errors.append(f"cannot hash original-audit manifest: {exc}")
    try:
        cache_sha = sha256_file(cache_path)
        _, cache_sample_ids = load_original_audit_feature_cache(cache_path)
        sample_order_sha = ordered_sample_ids_sha256(cache_sample_ids)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        cache_sample_ids = ()
    try:
        final_groups_sha = sha256_file(final_groups)
        final_group_ids = tuple(
            line.strip()
            for line in final_groups.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not final_group_ids or len(set(final_group_ids)) != len(final_group_ids):
            raise ValueError("final-reference group evidence must be non-empty and unique")
        final_group_ids_sha = canonical_sha256(sorted(final_group_ids))
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"cannot hash final-reference group evidence: {exc}")
        final_groups_sha = None
        final_group_ids_sha = None

    completion_manifest_sha = _normal_sha(
        completion.get("manifest_sha256"), role="confirmatory manifest hash", errors=errors
    )
    completion_dataset_sha = _normal_sha(
        completion.get("dataset_sha256"), role="confirmatory dataset hash", errors=errors
    )
    if manifest_sha is not None and completion_manifest_sha != manifest_sha:
        errors.append("original-audit manifest differs from sealed confirmatory evidence")

    sidecar = _mapping(sidecar_path, role="original-audit feature-cache sidecar", errors=errors)
    raw_audit_sample_ids = sidecar.get("audit_sample_ids")
    if not isinstance(raw_audit_sample_ids, list) or not all(
        isinstance(value, str) for value in raw_audit_sample_ids
    ):
        errors.append("feature-cache sidecar lacks the exact audit sample order")
        audit_sample_ids: tuple[str, ...] = ()
        audit_sample_order_sha = None
    else:
        audit_sample_ids = tuple(raw_audit_sample_ids)
        try:
            audit_sample_order_sha = ordered_sample_ids_sha256(audit_sample_ids)
        except ValueError as exc:
            errors.append(f"feature-cache sidecar audit sample order is invalid: {exc}")
            audit_sample_order_sha = None
        if not set(audit_sample_ids).issubset(cache_sample_ids):
            errors.append("feature-cache sidecar audit IDs are absent from the frozen cache")

    entries = sealed_features.get("representations")
    if (
        sealed_features.get("schema_version") != 1
        or sealed_features.get("status") != "completed"
        or not isinstance(entries, Mapping)
    ):
        errors.append("confirmatory frozen-feature provenance schema is invalid")
        entries = {}
    selected_entry: Mapping[str, Any] = {}
    selected_outer_fold: int | None = None
    aggregate_rotations: Mapping[str, Any] = {}
    selection_rotations = selection_artifact.get("sealed_feature_cache_provenance_by_rotation")
    if selection is not None:
        raw_representation = entries.get(selection.representation_id)
        raw_aggregate_rotations = (
            raw_representation.get("rotations") if isinstance(raw_representation, Mapping) else None
        )
        if not isinstance(raw_aggregate_rotations, Mapping):
            errors.append("sealed confirmatory provenance lacks the selected representation")
        else:
            aggregate_rotations = raw_aggregate_rotations
    if not isinstance(selection_rotations, Mapping):
        errors.append("sealed original-audit selection lacks per-rotation cache provenance")
        selection_rotations = {}
    matching_rotations: list[tuple[int, Mapping[str, Any]]] = []
    for raw_fold, raw_entry in selection_rotations.items():
        if not isinstance(raw_entry, Mapping):
            errors.append(f"selected cache rotation {raw_fold!r} is malformed")
            continue
        aggregate_entry = aggregate_rotations.get(str(raw_fold))
        without_final_groups = {
            key: value
            for key, value in raw_entry.items()
            if key != "final_reference_group_ids_sha256"
        }
        if aggregate_entry != without_final_groups:
            errors.append(f"selected cache rotation {raw_fold!r} differs from aggregate provenance")
        direct_cache_bound = (
            raw_entry.get("cache_file_sha256") == cache_sha
            and raw_entry.get("sidecar_semantic_sha256") is None
        )
        exact_cache_match = (
            direct_cache_bound
            and raw_entry.get("sample_order_sha256") == sample_order_sha
            and raw_entry.get("audit_sample_order_sha256") == audit_sample_order_sha
            and raw_entry.get("manifest_sha256") == manifest_sha
            and raw_entry.get("final_reference_group_ids_sha256") == final_group_ids_sha
        )
        if exact_cache_match:
            try:
                matching_rotations.append((int(str(raw_fold)), raw_entry))
            except ValueError:
                errors.append(f"selected cache rotation ID is not an integer: {raw_fold!r}")
    if len(matching_rotations) != 1:
        errors.append(
            "exactly one sealed confirmatory rotation must bind the supplied audit cache/order, "
            "manifest, and final-reference groups"
        )
    else:
        selected_outer_fold, selected_entry = matching_rotations[0]

    if frozen_cache_requirements.get("status") != "available":
        errors.append("frozen original-audit cache is not marked available")
    if frozen_cache_requirements.get("cache_file_sha256") != cache_sha:
        errors.append("frozen original-audit selection does not bind the exact cache file")
    if frozen_cache_requirements.get("sidecar_semantic_sha256") is not None:
        errors.append("stage-ready original audit requires a direct cache-file SHA binding")
    provenance_to_config_fields = {
        "cache_provenance_id": "id",
        "representation_id": "representation_id",
        "cache_file_sha256": "cache_file_sha256",
        "sidecar_semantic_sha256": "sidecar_semantic_sha256",
        "sample_order_sha256": "sample_order_sha256",
        "manifest_sha256": "manifest_sha256",
        "encoder_identifier": "encoder_identifier",
        "encoder_metadata_sha256": "encoder_metadata_sha256",
        "weight_identifier": "weight_identifier",
        "weights_sha256": "weights_sha256",
        "preprocessing_identifier": "preprocessing_identifier",
        "preprocessing_sha256": "preprocessing_sha256",
        "input_variant": "input_variant",
    }
    for provenance_field, config_field in provenance_to_config_fields.items():
        if selected_entry.get(provenance_field) != frozen_cache_requirements.get(config_field):
            errors.append(f"selected cache {provenance_field} differs from frozen cache provenance")

    if sidecar.get("schema_version") != 1:
        errors.append("original-audit feature-cache sidecar schema is unsupported")
    if sidecar.get("artifact_scope") != ORIGINAL_AUDIT_FEATURE_SCOPE:
        errors.append("feature-cache sidecar is not real PanNuke original-audit evidence")
    if sidecar.get("study_outcome_eligible") is not True:
        errors.append("feature-cache sidecar is not marked study-outcome eligible")
    cache_record = sidecar.get("feature_cache")
    if not isinstance(cache_record, Mapping):
        errors.append("feature-cache sidecar lacks its cache record")
    else:
        if Path(str(cache_record.get("path", ""))).resolve() != cache_path:
            errors.append("feature-cache sidecar path differs from the supplied cache")
        if cache_record.get("sha256") != cache_sha:
            errors.append("feature-cache sidecar hash differs from the supplied cache")
    sidecar_expected = {
        "representation_id": selection.representation_id if selection else None,
        "sample_count": len(cache_sample_ids),
        "sample_order_sha256": sample_order_sha,
        "audit_sample_count": len(audit_sample_ids),
        "audit_sample_order_sha256": audit_sample_order_sha,
        "manifest_sha256": manifest_sha,
        "final_reference_groups_sha256": final_groups_sha,
        "final_reference_group_ids_sha256": final_group_ids_sha,
        "confirmatory_outer_fold": selected_outer_fold,
        "cache_provenance_id": (selection.cache_provenance_id if selection is not None else None),
    }
    for field, expected_value in sidecar_expected.items():
        if sidecar.get(field) != expected_value:
            errors.append(f"feature-cache sidecar {field} differs from exact frozen evidence")
    sidecar_encoder = sidecar.get("encoder_metadata")
    if not isinstance(sidecar_encoder, Mapping):
        errors.append("feature-cache sidecar lacks encoder metadata")
    else:
        for field in (
            "encoder_identifier",
            "encoder_metadata_sha256",
            "weight_identifier",
            "weights_sha256",
            "preprocessing_identifier",
            "preprocessing_sha256",
            "input_variant",
        ):
            if sidecar_encoder.get(field) != selected_entry.get(field):
                errors.append(f"feature-cache sidecar encoder {field} differs from sealed evidence")

    completion_sha = sha256_file(completion_path) if completion_path.is_file() else None
    plan_sha = sha256_file(plan_path) if plan_path.is_file() else None
    resolved_config_sha = (
        sha256_file(resolved_config_path) if resolved_config_path.is_file() else None
    )
    sealed_feature_sha = sha256_file(sealed_feature_path) if sealed_feature_path.is_file() else None
    selection_artifact_sha = (
        sha256_file(selection_artifact_path) if selection_artifact_path.is_file() else None
    )
    execution_controls_sha = (
        sha256_file(execution_controls_path) if execution_controls_path.is_file() else None
    )
    frozen_bindings = sidecar.get("frozen_bindings")
    expected_bindings = {
        "confirmatory_run_id": integrity.run_id,
        "confirmatory_artifact_root_sha256": integrity.actual_root_sha256,
        "confirmatory_completion_sha256": completion_sha,
        "confirmatory_plan_sha256": plan_sha,
        "confirmatory_resolved_config_sha256": resolved_config_sha,
        "frozen_primary_config_sha256": completion.get("frozen_primary_config_sha256"),
        "frozen_confirmatory_config_sha256": completion.get("frozen_confirmatory_config_sha256"),
        "confirmatory_config_semantic_sha256": semantic_config_sha,
        "original_audit_selection_sha256": (
            selection.selection_sha256 if selection is not None else None
        ),
        "frozen_feature_provenance_sha256": sealed_feature_sha,
        "original_audit_selection_artifact_sha256": selection_artifact_sha,
        "confirmatory_execution_controls_sha256": execution_controls_sha,
    }
    if not isinstance(frozen_bindings, Mapping):
        errors.append("feature-cache sidecar lacks frozen config/plan bindings")
    else:
        for field, expected_value in expected_bindings.items():
            if frozen_bindings.get(field) != expected_value:
                errors.append(f"feature-cache sidecar binding {field} differs from sealed evidence")

    _validate_confirmatory_registry(
        run,
        run_id=integrity.run_id,
        config_sha256_value=semantic_config_sha,
        dataset_sha256=completion_dataset_sha,
        manifest_sha256=completion_manifest_sha,
        errors=errors,
    )
    sidecar_sha = sha256_file(sidecar_path) if sidecar_path.is_file() else None
    return OriginalAuditUpstreamVerification(
        eligible=not errors,
        confirmatory_run_directory=run,
        confirmatory_run_id=integrity.run_id,
        confirmatory_artifact_root_sha256=integrity.actual_root_sha256,
        completion_evidence_sha256=completion_sha,
        confirmatory_plan_sha256=plan_sha,
        resolved_config_sha256=resolved_config_sha,
        feature_provenance_artifact_sha256=sealed_feature_sha,
        selection_artifact_sha256=selection_artifact_sha,
        selected_outer_fold=selected_outer_fold,
        feature_cache_provenance_path=sidecar_path,
        feature_cache_provenance_sha256=sidecar_sha,
        feature_cache_sha256=cache_sha,
        manifest_sha256=manifest_sha,
        sample_order_sha256=sample_order_sha,
        audit_sample_order_sha256=audit_sample_order_sha,
        audit_sample_ids=audit_sample_ids,
        selection=selection,
        errors=tuple(errors),
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_anomaly_safe_mask_qc(
    validation: Mapping[str, Any],
    folds: list[Mapping[str, Any]],
    *,
    errors: list[str],
) -> None:
    """Reconcile complete mask-QC evidence without treating known anomalies as corruption."""

    if validation.get("automatic_source_annotation_modification") is not False:
        errors.append("PanNuke validation does not forbid automatic source-mask modification")
    policy = validation.get("qc_policy")
    expected_policy: dict[str, object] = {
        "policy_version": "pannuke-mask-qc-v2",
        "positive_channel_indices": list(ORIGINAL_AUDIT_CLASS_ORDER),
        "supplied_background_is_exact_complement_required": False,
        "analysis_instance_exclusion_reason": "touches_cross_class_overlap",
        "applies_identically_to_primary_and_confirmatory": True,
        "no_class_arbitration": True,
        "source_masks_modified": False,
        "release_annotation_anomalies_are_fatal": False,
        "structural_invalidity_is_fatal": True,
    }
    if not isinstance(policy, Mapping):
        errors.append("PanNuke validation lacks the anomaly-safe mask-QC policy")
        return
    for field, expected in expected_policy.items():
        if policy.get(field) != expected:
            errors.append(f"PanNuke mask-QC policy {field} differs from the frozen contract")

    background_by_fold = policy.get("background_channel_index_by_fold")
    if not isinstance(background_by_fold, Mapping):
        errors.append("PanNuke mask-QC policy lacks per-fold supplied-background channels")

    pixel_fields = (
        "total_pixel_count",
        "positive_any_pixel_count",
        "background_pixel_count",
        "void_pixel_count",
        "cross_class_overlap_pixel_count",
        "positive_and_background_pixel_count",
        "anomaly_union_pixel_count",
        "affected_instance_count",
    )
    global_sum_fields = (
        "patch_count",
        *pixel_fields,
        "void_patch_count",
        "cross_class_overlap_patch_count",
        "positive_and_background_patch_count",
        "anomaly_union_patch_count",
        "normal_patch_count",
        "overlap_touching_instance_count",
        "positive_background_touching_instance_count",
    )
    categories = (
        ("void", "void_pixel_count", "has_void"),
        (
            "cross_class_overlap",
            "cross_class_overlap_pixel_count",
            "has_cross_class_overlap",
        ),
        (
            "positive_and_background",
            "positive_and_background_pixel_count",
            "has_positive_and_background",
        ),
        ("anomaly_union", "anomaly_union_pixel_count", None),
    )
    required_mask_hashes = {
        "positive_any",
        "supplied_background",
        "void_unlabelled",
        "cross_class_overlap",
        "positive_and_background",
        "anomaly_union",
    }
    fold_summaries: list[Mapping[str, Any]] = []
    for fold in folds:
        fold_id = fold.get("fold_id")
        patch_count = fold.get("n_patches")
        height = fold.get("height")
        width = fold.get("width")
        qc = fold.get("mask_qc")
        if not all(_is_non_negative_int(value) for value in (fold_id, patch_count, height, width)):
            errors.append("PanNuke fold dimensions/counts are invalid for complete mask QC")
            continue
        fold_id = cast(int, fold_id)
        patch_count = cast(int, patch_count)
        height = cast(int, height)
        width = cast(int, width)
        if not isinstance(qc, Mapping):
            errors.append(f"PanNuke fold {fold_id} lacks complete mask-QC evidence")
            continue
        if qc.get("fold_id") != fold_id or qc.get("patch_count") != patch_count:
            errors.append(f"PanNuke fold {fold_id} mask-QC identity/coverage is inconsistent")
        if fold.get("positive_channel_indices") != list(ORIGINAL_AUDIT_CLASS_ORDER):
            errors.append(f"PanNuke fold {fold_id} positive-channel order differs from QC policy")
        if isinstance(background_by_fold, Mapping) and background_by_fold.get(
            str(fold_id)
        ) != fold.get("background_channel_index"):
            errors.append(
                f"PanNuke fold {fold_id} supplied-background channel differs from QC policy"
            )
        sampled_overlap = fold.get("overlap_pixel_count_sampled")
        full_overlap = qc.get("cross_class_overlap_pixel_count")
        sampled_overlap_invalid = not _is_non_negative_int(
            sampled_overlap
        ) or not _is_non_negative_int(full_overlap)
        if not sampled_overlap_invalid:
            sampled_overlap_invalid = cast(int, sampled_overlap) > cast(int, full_overlap)
        if sampled_overlap_invalid:
            errors.append(f"PanNuke fold {fold_id} sampled overlap is inconsistent with full QC")

        patches = qc.get("patches")
        if not isinstance(patches, list) or len(patches) != patch_count:
            errors.append(f"PanNuke fold {fold_id} mask QC does not contain every source patch")
            continue
        sums = {field: 0 for field in pixel_fields}
        category_indices: dict[str, list[int]] = {name: [] for name, _, _ in categories}
        overlap_touching_instances = 0
        positive_background_touching_instances = 0
        observed_indices: list[int] = []
        patch_records_valid = True
        for patch in patches:
            if not isinstance(patch, Mapping):
                patch_records_valid = False
                continue
            patch_index = patch.get("patch_index")
            if patch.get("fold_id") != fold_id or not _is_non_negative_int(patch_index):
                patch_records_valid = False
                continue
            patch_index = cast(int, patch_index)
            observed_indices.append(patch_index)
            raw_values = {field: patch.get(field) for field in pixel_fields}
            if not all(_is_non_negative_int(value) for value in raw_values.values()):
                patch_records_valid = False
                continue
            values = {field: cast(int, value) for field, value in raw_values.items()}
            expected_pixels = height * width
            if values["total_pixel_count"] != expected_pixels:
                patch_records_valid = False
            if (
                values["positive_any_pixel_count"]
                + values["background_pixel_count"]
                + values["void_pixel_count"]
                - values["positive_and_background_pixel_count"]
                != values["total_pixel_count"]
            ):
                patch_records_valid = False
            anomaly_components = (
                values["void_pixel_count"],
                values["cross_class_overlap_pixel_count"],
                values["positive_and_background_pixel_count"],
            )
            if (
                not max(anomaly_components)
                <= values["anomaly_union_pixel_count"]
                <= min(values["total_pixel_count"], sum(anomaly_components))
            ):
                patch_records_valid = False
            affected = patch.get("affected_instances")
            if not isinstance(affected, list) or len(affected) != values["affected_instance_count"]:
                patch_records_valid = False
                affected = []
            for instance in affected:
                if not isinstance(instance, Mapping):
                    patch_records_valid = False
                    continue
                overlap_count = instance.get("overlap_pixel_count")
                positive_background_count = instance.get("positive_background_pixel_count")
                if not _is_non_negative_int(overlap_count) or not _is_non_negative_int(
                    positive_background_count
                ):
                    patch_records_valid = False
                    continue
                overlap_count = cast(int, overlap_count)
                positive_background_count = cast(int, positive_background_count)
                overlap_touching_instances += int(overlap_count > 0)
                positive_background_touching_instances += int(positive_background_count > 0)
            mask_hashes = patch.get("mask_sha256_by_kind")
            if (
                not isinstance(mask_hashes, Mapping)
                or set(mask_hashes) != required_mask_hashes
                or any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in mask_hashes.values()
                )
            ):
                patch_records_valid = False
            for field in pixel_fields:
                sums[field] += values[field]
            for name, pixel_field, boolean_field in categories:
                present = values[pixel_field] > 0
                if boolean_field is not None and patch.get(boolean_field) is not present:
                    patch_records_valid = False
                if present:
                    category_indices[name].append(patch_index)
        if sorted(observed_indices) != list(range(patch_count)):
            patch_records_valid = False
        if not patch_records_valid:
            errors.append(f"PanNuke fold {fold_id} contains inconsistent patch-level mask QC")
        for field, observed in sums.items():
            if qc.get(field) != observed:
                errors.append(f"PanNuke fold {fold_id} mask-QC aggregate {field} is inconsistent")
        for name, _, _ in categories:
            indices = sorted(category_indices[name])
            if (
                qc.get(f"{name}_patch_count") != len(indices)
                or qc.get(f"{name}_patch_indices") != indices
            ):
                errors.append(f"PanNuke fold {fold_id} {name} patch QC is inconsistent")
        if qc.get("normal_patch_count") != patch_count - len(category_indices["anomaly_union"]):
            errors.append(f"PanNuke fold {fold_id} normal-patch QC is inconsistent")
        if qc.get("overlap_touching_instance_count") != overlap_touching_instances:
            errors.append(f"PanNuke fold {fold_id} overlap-touching instance QC is inconsistent")
        if (
            qc.get("positive_background_touching_instance_count")
            != positive_background_touching_instances
        ):
            errors.append(f"PanNuke fold {fold_id} positive-background instance QC is inconsistent")
        fold_summaries.append(qc)

    global_qc = validation.get("global_mask_qc")
    if not isinstance(global_qc, Mapping):
        errors.append("PanNuke validation lacks release-wide mask-QC evidence")
        return
    fold_ids = [fold.get("fold_id") for fold in folds]
    if global_qc.get("fold_ids") != fold_ids or global_qc.get("fold_count") != len(folds):
        errors.append("PanNuke release-wide mask-QC fold coverage is inconsistent")
    if len(fold_summaries) != len(folds):
        return
    for field in global_sum_fields:
        fold_values = [fold.get(field) for fold in fold_summaries]
        if not all(_is_non_negative_int(value) for value in fold_values):
            errors.append(f"PanNuke release-wide mask-QC aggregate {field} is inconsistent")
            continue
        expected = sum(cast(int, value) for value in fold_values)
        if global_qc.get(field) != expected:
            errors.append(f"PanNuke release-wide mask-QC aggregate {field} is inconsistent")


def validate_real_dataset_evidence(
    dataset_path: str | Path,
    dataset_validation_json: str | Path,
    duplicate_audit_json: str | Path,
) -> tuple[str, ...]:
    """Validate full-release semantic, inventory, and two-signal duplicate evidence."""

    dataset = Path(dataset_path).resolve()
    validation_path = Path(dataset_validation_json).resolve()
    duplicate_path = Path(duplicate_audit_json).resolve()
    errors: list[str] = []
    if not dataset.is_dir():
        errors.append(f"eligible real dataset must be a directory: {dataset}")
        return tuple(errors)

    validation = _mapping(validation_path, role="PanNuke validation evidence", errors=errors)
    if validation.get("status") != "valid":
        errors.append("PanNuke validation status is not valid")
    if validation.get("validation_scope") != "full_semantic_scan":
        errors.append("PanNuke validation did not perform a full semantic scan")
    if validation.get("release_complete") is not True:
        errors.append("PanNuke validation does not certify a complete release")
    if tuple(validation.get("expected_fold_ids", ())) != OFFICIAL_RELEASE_FOLD_IDS:
        errors.append("PanNuke validation does not cover all three official folds")
    if validation.get("grouping_unit") != "source_patch":
        errors.append("PanNuke validation lacks the required source-patch grouping unit")
    raw_root = validation.get("root")
    if not isinstance(raw_root, str) or Path(raw_root).resolve() != dataset:
        errors.append("PanNuke validation root differs from the supplied dataset")
    class_mapping = validation.get("class_mapping")
    expected_mapping = OFFICIAL_METRICS_CLASS_MAPPING.as_dict()
    if not isinstance(class_mapping, Mapping):
        errors.append("PanNuke validation lacks verified class-mapping evidence")
    else:
        for field in ("class_names", "source_revision", "verified"):
            observed = class_mapping.get(field)
            expected = expected_mapping[field]
            if field == "class_names":
                observed = tuple(observed) if isinstance(observed, list) else observed
                expected = tuple(expected)
            if observed != expected:
                errors.append(f"PanNuke class-mapping {field} differs from pinned evidence")

    fold_validation = validation.get("fold_validation")
    typed_folds: list[Mapping[str, Any]] = []
    if (
        not isinstance(fold_validation, list)
        or len(fold_validation) != len(OFFICIAL_RELEASE_FOLD_IDS)
        or not all(isinstance(item, Mapping) for item in fold_validation)
        or {item.get("fold_id") for item in fold_validation if isinstance(item, Mapping)}
        != set(OFFICIAL_RELEASE_FOLD_IDS)
    ):
        errors.append("PanNuke fold-validation evidence is incomplete")
    else:
        typed_folds = [item for item in fold_validation if isinstance(item, Mapping)]
        if any(
            item.get("validation_scope") != "full_semantic_scan"
            or item.get("full_scan_patch_count") != item.get("n_patches")
            or item.get("malformed_instance_count_sampled") != 0
            for item in typed_folds
        ):
            errors.append("one or more PanNuke folds lacks complete semantic coverage")
        _validate_anomaly_safe_mask_qc(validation, typed_folds, errors=errors)

    inventory = validation.get("raw_file_inventory")
    inventory_records: list[Mapping[str, Any]] = []
    if not isinstance(inventory, list) or not inventory:
        errors.append("PanNuke validation lacks a non-empty raw-file inventory")
    else:
        seen: set[str] = set()
        for index, item in enumerate(inventory):
            if not isinstance(item, Mapping):
                errors.append(f"raw-file inventory record {index} is invalid")
                continue
            inventory_records.append(item)
            relative = str(item.get("relative_path", ""))
            if not relative or relative in seen:
                errors.append("raw-file inventory paths are empty or duplicated")
                continue
            seen.add(relative)
            source = (dataset / relative).resolve()
            if not _inside(source, dataset) or not source.is_file():
                errors.append(f"raw-file inventory path is missing or escapes dataset: {relative}")
                continue
            expected_digest = _normal_sha(
                item.get("sha256"), role=f"raw-file inventory hash for {relative}", errors=errors
            )
            if expected_digest is not None and sha256_file(source) != expected_digest:
                errors.append(f"raw dataset file changed after validation: {relative}")
            expected_size = item.get("size_bytes")
            if not isinstance(expected_size, int) or source.stat().st_size != expected_size:
                errors.append(f"raw dataset file size differs from validation: {relative}")

    duplicate = _mapping(duplicate_path, role="duplicate-audit evidence", errors=errors)
    if duplicate.get("schema_version") != 2 or duplicate.get("status") != "completed":
        errors.append("duplicate audit schema/status is not the completed canonical contract")
    if duplicate.get("required_two_signal_near_duplicate_gate_complete") is not True:
        errors.append("duplicate audit lacks complete two-signal near-duplicate coverage")
    duplicate_policy = duplicate.get("policy")
    expected_duplicate_policy: dict[str, object] = {
        "automatic_deletion": False,
        "candidate_action": "review_only",
        "cross_fold_only": True,
        "split_or_exclusion_change_applied": False,
        "final_reference_outcomes_used": False,
        "grouping_unit": "source_patch",
    }
    if not isinstance(duplicate_policy, Mapping):
        errors.append("duplicate audit lacks its review-only split-preservation policy")
    else:
        for field, expected in expected_duplicate_policy.items():
            if duplicate_policy.get(field) != expected:
                errors.append(f"duplicate audit policy {field} differs from the safe contract")

    fold_counts: dict[str, int] = {}
    for fold in typed_folds:
        fold_id = fold.get("fold_id")
        fold_patch_count = fold.get("n_patches")
        if _is_non_negative_int(fold_id) and _is_non_negative_int(fold_patch_count):
            fold_counts[str(cast(int, fold_id))] = cast(int, fold_patch_count)
    expected_pair_counts: dict[str, int] = {}
    ordered_fold_ids = sorted(int(value) for value in fold_counts)
    for offset, first in enumerate(ordered_fold_ids):
        for second in ordered_fold_ids[offset + 1 :]:
            expected_pair_counts[f"{first}-{second}"] = (
                fold_counts[str(first)] * fold_counts[str(second)]
            )
    expected_patch_count = sum(fold_counts.values())
    expected_pair_count = sum(expected_pair_counts.values())
    coverage = duplicate.get("coverage")
    if not isinstance(coverage, Mapping):
        errors.append("duplicate audit lacks coverage evidence")
    else:
        total = coverage.get("total_source_patches")
        if not _is_non_negative_int(total):
            errors.append("duplicate audit total-source-patch count is invalid")
            total_patch_count = -1
        else:
            total_patch_count = cast(int, total)
            if total_patch_count <= 0:
                errors.append("duplicate audit total-source-patch count is invalid")
        if total_patch_count > 0 and any(
            coverage.get(name) != total_patch_count
            for name in (
                "patches_with_full_hash_provenance",
                "perceptual_comparison_patch_count",
                "embedding_patch_count",
            )
        ):
            errors.append("duplicate audit does not cover every source patch with both signals")
        if expected_patch_count <= 0 or total_patch_count != expected_patch_count:
            errors.append("duplicate audit patch coverage differs from validated official folds")
        if coverage.get("fold_patch_counts") != fold_counts:
            errors.append("duplicate audit per-fold patch coverage is inconsistent")
        for signal in ("full_release", "perceptual", "embedding"):
            if (
                coverage.get(f"{signal}_cross_fold_pair_counts_by_fold_pair")
                != expected_pair_counts
            ):
                errors.append(f"duplicate audit {signal} fold-pair coverage is inconsistent")
            if coverage.get(f"{signal}_cross_fold_pair_count") != expected_pair_count:
                errors.append(f"duplicate audit {signal} total pair coverage is inconsistent")

    expected_raw_inventory_binding: str | None = None
    if inventory_records and len(inventory_records) == len(inventory or []):
        expected_raw_inventory_binding = canonical_sha256(
            {
                "schema_version": 1,
                "kind": "pannuke_validated_raw_inventory",
                "files": [
                    dict(item)
                    for item in sorted(
                        inventory_records, key=lambda value: str(value.get("relative_path", ""))
                    )
                ],
            }
        )
    provenance_bindings = duplicate.get("provenance_bindings")
    patch_manifest_binding: str | None = None
    raw_inventory_binding: str | None = None
    rgb_input_binding: str | None = None
    if not isinstance(provenance_bindings, Mapping):
        errors.append("duplicate audit lacks full provenance bindings")
    else:
        patch_manifest_binding = _normal_sha(
            provenance_bindings.get("patch_manifest_sha256"),
            role="duplicate patch-manifest binding",
            errors=errors,
        )
        raw_inventory_binding = _normal_sha(
            provenance_bindings.get("raw_inventory_sha256"),
            role="duplicate raw-inventory binding",
            errors=errors,
        )
        rgb_input_binding = _normal_sha(
            provenance_bindings.get("canonical_rgb_embedding_input_sha256"),
            role="duplicate canonical-RGB input binding",
            errors=errors,
        )
        if (
            expected_raw_inventory_binding is not None
            and raw_inventory_binding != expected_raw_inventory_binding
        ):
            errors.append("duplicate audit raw-inventory binding differs from validation")
    embedding = duplicate.get("embedding_signal")
    if (
        not isinstance(embedding, Mapping)
        or embedding.get("status") != "passed"
        or embedding.get("full_patch_coverage") is not True
    ):
        errors.append("duplicate audit lacks passed full-coverage embedding evidence")
    else:
        metadata = embedding.get("metadata")
        if not isinstance(metadata, Mapping):
            errors.append("duplicate audit lacks frozen-encoder metadata")
        else:
            if metadata.get("encoder_name") != "torchvision.resnet18":
                errors.append("duplicate audit used an unexpected embedding encoder")
            if metadata.get("encoder_frozen") is not True:
                errors.append("duplicate audit encoder was not frozen")
            if metadata.get("weight_identifier") != "ResNet18_Weights.IMAGENET1K_V1":
                errors.append("duplicate audit did not use the pinned official ImageNet weights")
            if metadata.get("weight_sha256") != _RESNET18_IMAGENET1K_V1_SHA256:
                errors.append("duplicate audit ImageNet weight hash differs from the pinned weight")
            if metadata.get("input_variant") != "context_rgb":
                errors.append("duplicate audit cache does not use canonical context-RGB provenance")
            if metadata.get("sample_count") != expected_patch_count:
                errors.append("duplicate audit cache sample count differs from complete coverage")
            if metadata.get("manifest_sha256") != patch_manifest_binding:
                errors.append("duplicate audit cache patch-manifest binding differs from report")
            if metadata.get("raw_inventory_sha256") != raw_inventory_binding:
                errors.append("duplicate audit cache raw-inventory binding differs from report")
            if metadata.get("input_sha256") != rgb_input_binding:
                errors.append("duplicate audit cache canonical-RGB binding differs from report")
            if isinstance(coverage, Mapping):
                sample_order_binding = _normal_sha(
                    coverage.get("sample_order_sha256"),
                    role="duplicate sample-order binding",
                    errors=errors,
                )
                if metadata.get("sample_order_sha256") != sample_order_binding:
                    errors.append("duplicate audit cache sample order differs from report")
            preprocessing = metadata.get("preprocessing")
            if (
                not isinstance(preprocessing, Mapping)
                or preprocessing.get("api") != "torchvision weight_enum.transforms(antialias=True)"
            ):
                errors.append("duplicate audit lacks the official weight-enum preprocessing")
            if "test_substitution" in metadata:
                errors.append("duplicate audit contains an explicit test substitution")
            cache_path = Path(str(embedding.get("cache_path", ""))).resolve()
            cache_digest = _normal_sha(
                metadata.get("cache_npz_sha256"),
                role="duplicate embedding-cache hash",
                errors=errors,
            )
            if not cache_path.is_file():
                errors.append("duplicate audit frozen-embedding cache is missing")
            elif cache_digest is not None and sha256_file(cache_path) != cache_digest:
                errors.append("duplicate audit frozen-embedding cache hash changed")
            sidecar = cache_path.with_suffix(f"{cache_path.suffix}.metadata.json")
            sidecar_metadata = _mapping(
                sidecar, role="duplicate embedding-cache metadata", errors=errors
            )
            if sidecar_metadata and dict(sidecar_metadata) != dict(metadata):
                errors.append("duplicate audit embedding metadata differs from its cache sidecar")
    artifacts = duplicate.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        errors.append("duplicate audit lacks artifact hash records")
    else:
        for key, raw_path in artifacts.items():
            if key.endswith("_sha256"):
                continue
            digest_key = f"{key}_sha256"
            expected_digest = _normal_sha(
                artifacts.get(digest_key), role=f"duplicate artifact {digest_key}", errors=errors
            )
            source = Path(str(raw_path)).resolve()
            if not source.is_file():
                errors.append(f"duplicate-audit artifact is missing: {source}")
            elif expected_digest is not None and sha256_file(source) != expected_digest:
                errors.append(f"duplicate-audit artifact hash changed: {source}")
    return tuple(errors)


def verify_external_validation_eligibility(
    *,
    audit_run_directory: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    dataset_validation_json: str | Path,
    duplicate_audit_json: str | Path,
    ranking_path: str | Path,
    confirmatory_run_directory: str | Path | None = None,
    feature_cache_provenance_json: str | Path | None = None,
) -> ExternalValidationEligibilityResult:
    """Verify the complete sealed evidence chain required for the readiness stage."""

    run = Path(audit_run_directory).resolve()
    dataset = Path(dataset_path).resolve()
    manifest = Path(manifest_path).resolve()
    validation_path = Path(dataset_validation_json).resolve()
    duplicate_path = Path(duplicate_audit_json).resolve()
    ranking = Path(ranking_path).resolve()
    errors: list[str] = []
    integrity = verify_run_integrity(run)
    if not integrity.valid or not integrity.registry_record_present:
        errors.append(f"original-audit run integrity verification failed: {integrity.errors}")
    if integrity.run_id != run.name:
        errors.append("original-audit run identity differs from its sealed directory name")
    status = _mapping(run / STATUS_FILENAME, role="original-audit run status", errors=errors)
    marker = _mapping(run / IMMUTABLE_MARKER, role="original-audit immutable marker", errors=errors)
    provenance = _mapping(
        run / "run_provenance.json", role="original-audit provenance", errors=errors
    )
    if status.get("status") != "completed" or marker.get("status") != "completed":
        errors.append("original-audit run was not sealed as completed")
    if provenance.get("experiment_name") != ORIGINAL_AUDIT_EXPERIMENT_NAME:
        errors.append("sealed run is not an original-label audit")

    evidence = _mapping(run / ELIGIBILITY_FILENAME, role="eligibility evidence", errors=errors)
    if evidence.get("schema_version") != 2:
        errors.append("eligibility evidence schema version is unsupported")
    if evidence.get("study_outcome_eligible") is not True:
        errors.append("sealed audit is explicitly not eligible for a study completion stage")
    if evidence.get("workflow") != "exploratory_original_label_audit":
        errors.append("eligibility evidence names an unexpected workflow")
    if evidence.get("data_source") != "verified_pannuke_release":
        errors.append("eligibility evidence does not identify a verified PanNuke release")

    dataset_sha = _record_path(evidence, "dataset", dataset, directory=True, errors=errors)
    manifest_sha = _record_path(evidence, "manifest", manifest, directory=False, errors=errors)
    validation_sha = _record_path(
        evidence, "dataset_validation", validation_path, directory=False, errors=errors
    )
    duplicate_sha = _record_path(
        evidence, "duplicate_audit", duplicate_path, directory=False, errors=errors
    )
    ranking_sha = _record_path(evidence, "ranking", ranking, directory=False, errors=errors)
    if not _inside(ranking, run):
        errors.append("eligible ranking is not contained in the sealed original-audit run")

    bound_sources: dict[str, Path] = {}
    for role in ("feature_cache", "feature_cache_provenance", "final_reference_groups"):
        raw = evidence.get(role)
        if not isinstance(raw, Mapping):
            errors.append(f"eligibility evidence lacks the {role!r} record")
            continue
        source = Path(str(raw.get("path", ""))).resolve()
        bound_sources[role] = source
        expected = _normal_sha(raw.get("sha256"), role=f"sealed {role} hash", errors=errors)
        if not source.is_file():
            errors.append(f"sealed {role} source no longer exists: {source}")
        elif expected is not None and sha256_file(source) != expected:
            errors.append(f"sealed {role} source changed after the audit")

    supplied_confirmatory = (
        Path(confirmatory_run_directory).resolve()
        if confirmatory_run_directory is not None
        else None
    )
    supplied_feature_provenance = (
        Path(feature_cache_provenance_json).resolve()
        if feature_cache_provenance_json is not None
        else None
    )
    if supplied_confirmatory is None or supplied_feature_provenance is None:
        errors.append(
            "external readiness requires confirmatory_run_directory and "
            "feature_cache_provenance_json"
        )
    recorded_confirmatory = evidence.get("confirmatory_run")
    if not isinstance(recorded_confirmatory, Mapping):
        errors.append("eligibility evidence lacks the sealed confirmatory-run record")
    elif (
        supplied_confirmatory is not None
        and Path(str(recorded_confirmatory.get("path", ""))).resolve() != supplied_confirmatory
    ):
        errors.append("supplied confirmatory run differs from sealed eligibility evidence")
    recorded_sidecar = bound_sources.get("feature_cache_provenance")
    if (
        supplied_feature_provenance is not None
        and recorded_sidecar is not None
        and supplied_feature_provenance != recorded_sidecar
    ):
        errors.append("supplied feature-cache provenance differs from sealed evidence")

    upstream: OriginalAuditUpstreamVerification | None = None
    feature_cache_source = bound_sources.get("feature_cache")
    final_groups_source = bound_sources.get("final_reference_groups")
    if (
        supplied_confirmatory is not None
        and supplied_feature_provenance is not None
        and feature_cache_source is not None
        and final_groups_source is not None
    ):
        upstream = verify_original_audit_upstream(
            confirmatory_run_directory=supplied_confirmatory,
            feature_cache_path=feature_cache_source,
            feature_cache_provenance_json=supplied_feature_provenance,
            manifest_path=manifest,
            final_reference_groups_path=final_groups_source,
        )
        errors.extend(upstream.errors)
        if isinstance(recorded_confirmatory, Mapping):
            expected_confirmatory = {
                "run_id": upstream.confirmatory_run_id,
                "artifact_root_sha256": upstream.confirmatory_artifact_root_sha256,
                "completion_evidence_sha256": upstream.completion_evidence_sha256,
                "matrix_plan_sha256": upstream.confirmatory_plan_sha256,
                "resolved_config_sha256": upstream.resolved_config_sha256,
                "frozen_feature_provenance_sha256": (upstream.feature_provenance_artifact_sha256),
                "original_audit_selection_sha256": (
                    upstream.selection.selection_sha256 if upstream.selection else None
                ),
                "original_audit_selection_artifact_sha256": (upstream.selection_artifact_sha256),
                "selected_outer_fold": upstream.selected_outer_fold,
            }
            for field, expected_value in expected_confirmatory.items():
                if recorded_confirmatory.get(field) != expected_value:
                    errors.append(f"sealed original audit does not bind confirmatory {field}")

    checksums = _mapping(run / "checksums.json", role="original-audit checksums", errors=errors)
    dataset_record = checksums.get("dataset")
    manifest_record = checksums.get("manifest")
    if not isinstance(dataset_record, Mapping) or dataset_record.get("sha256") != dataset_sha:
        errors.append("run checksum record does not bind the supplied dataset")
    if not isinstance(manifest_record, Mapping) or manifest_record.get("sha256") != manifest_sha:
        errors.append("run checksum record does not bind the supplied manifest")
    if checksums.get("duplicate_audit_status") != f"complete_sha256:{duplicate_sha}":
        errors.append("run checksum record does not bind the complete duplicate audit")
    _validate_run_registry(
        run,
        run_id=integrity.run_id,
        dataset_sha256=dataset_sha,
        manifest_sha256=manifest_sha,
        errors=errors,
    )

    metadata = _mapping(run / "audit_metadata.json", role="original-audit metadata", errors=errors)
    source_manifest = metadata.get("source_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("input_mode") != "read_only_file"
        or source_manifest.get("sha256_before") != manifest_sha
        or source_manifest.get("sha256_after") != manifest_sha
    ):
        errors.append("original-audit metadata does not bind an unchanged file-backed manifest")
    if metadata.get("observed_equals_pre_corruption") is not True:
        errors.append("original-audit metadata does not certify unchanged original labels")
    if metadata.get("injected_corruption_count") != 0:
        errors.append("original-audit metadata contains injected corruptions")
    oof = metadata.get("group_safe_oof")
    if (
        not isinstance(oof, Mapping)
        or oof.get("coverage_exactly_once") is not True
        or oof.get("final_reference_groups_absent_from_audit") is not True
    ):
        errors.append("original-audit metadata lacks group-safe OOF/final-fold safeguards")
    if upstream is not None and upstream.selection is not None:
        selected = upstream.selection
        frozen_sample_selection = metadata.get("frozen_audit_sample_selection")
        if (
            not isinstance(frozen_sample_selection, Mapping)
            or frozen_sample_selection.get("enabled") is not True
            or frozen_sample_selection.get("sample_count") != len(upstream.audit_sample_ids)
            or frozen_sample_selection.get("sample_order_sha256")
            != upstream.audit_sample_order_sha256
        ):
            errors.append("original-audit metadata differs from the frozen audit sample order")
        if metadata.get("risk_method") != selected.risk_method:
            errors.append("original-audit risk method differs from the frozen selection")
        if isinstance(oof, Mapping):
            expected_oof = {
                "representation": selected.representation_id,
                "model_seed": selected.model_seed,
                "split_seed": selected.split_seed,
                "class_order": list(ORIGINAL_AUDIT_CLASS_ORDER),
            }
            for field, expected_oof_value in expected_oof.items():
                if oof.get(field) != expected_oof_value:
                    errors.append(f"original-audit OOF {field} differs from frozen selection")

    errors.extend(validate_real_dataset_evidence(dataset, validation_path, duplicate_path))
    return ExternalValidationEligibilityResult(
        eligible=not errors,
        audit_run_directory=run,
        run_id=integrity.run_id,
        artifact_root_sha256=integrity.actual_root_sha256,
        dataset_sha256=dataset_sha,
        manifest_sha256=manifest_sha,
        dataset_validation_sha256=validation_sha,
        duplicate_audit_sha256=duplicate_sha,
        ranking_sha256=ranking_sha,
        confirmatory_run_id=upstream.confirmatory_run_id if upstream else None,
        confirmatory_artifact_root_sha256=(
            upstream.confirmatory_artifact_root_sha256 if upstream else None
        ),
        confirmatory_completion_sha256=(upstream.completion_evidence_sha256 if upstream else None),
        feature_cache_provenance_sha256=(
            upstream.feature_cache_provenance_sha256 if upstream else None
        ),
        errors=tuple(errors),
    )


__all__ = [
    "ELIGIBILITY_FILENAME",
    "FEATURE_PROVENANCE_FILENAME",
    "ORIGINAL_AUDIT_CLASS_ORDER",
    "ORIGINAL_AUDIT_EXPERIMENT_NAME",
    "ORIGINAL_AUDIT_FEATURE_SCOPE",
    "ORIGINAL_AUDIT_SELECTION_FILENAME",
    "ExternalValidationEligibilityResult",
    "FrozenOriginalAuditSelection",
    "OriginalAuditUpstreamVerification",
    "load_original_audit_feature_cache",
    "ordered_sample_ids_sha256",
    "validate_real_dataset_evidence",
    "verify_external_validation_eligibility",
    "verify_original_audit_upstream",
]
