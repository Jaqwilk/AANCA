"""Read-only, fail-closed verification of a sealed real PanNuke pilot.

The verifier deliberately produces no artifact and never changes run disposition.
It recomputes the existing pilot evidence checks and returns a JSON-serialisable
attestation only after the sealed run and its external privacy-boundary inputs have
remained byte-stable for the complete verification window.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import yaml

from histo_audit.experiment.pilot import (
    _reconcile_pilot_final_reference_privacy as _format_aware_privacy_scan,
)
from histo_audit.experiment.pilot import reconcile_pilot_audit_evidence
from histo_audit.utils.run_tracking import (
    RUN_DISPOSITION_ANCHOR_FILENAME,
    RUN_DISPOSITION_REGISTRY_FILENAME,
    IntegrityVerification,
    read_run_dispositions,
    require_run_stage_eligible,
    verify_run_integrity,
)

_REQUIRED_TERMS = (
    "potentially inconsistent annotation",
    "recommended for expert review",
)
_PRIVACY_CORE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "status",
    "policy",
    "final_reference_official_fold",
    "final_fold_identity_pattern_absent",
    "final_sensitive_fields_unpopulated",
    "final_fold_representation_rows_absent",
)
_NO_OUTCOME_FINAL_ACCESS_FIELDS: tuple[str, ...] = (
    "class_labels_read",
    "outcomes_used",
    "representations_extracted",
    "sample_ids_read",
)
_SOURCE_PATCH_GROUP = re.compile(r"^pannuke-fold-[123]-patch-\d{6}$")


class PilotPostSealVerificationError(ValueError):
    """Raised when any mandatory post-seal pilot check fails closed."""


def _fail(message: str) -> NoReturn:
    raise PilotPostSealVerificationError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stable_file_bytes(path: Path, role: str) -> bytes:
    """Read one regular non-symlink file and reject a concurrent replacement."""

    if path.is_symlink() or not path.is_file():
        _fail(f"{role} must be a regular non-symlink file: {path}")
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise PilotPostSealVerificationError(f"{role} is unreadable: {path}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(content) != after.st_size:
        _fail(f"{role} changed while it was being read: {path}")
    return content


def _optional_stable_file_bytes(path: Path, role: str) -> bytes | None:
    if not os.path.lexists(path):
        return None
    return _stable_file_bytes(path, role)


def _json_mapping(content: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotPostSealVerificationError(f"{role} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        _fail(f"{role} must be a JSON object")
    return dict(value)


def _exact_int(value: object, role: str) -> int:
    if type(value) is not int:
        _fail(f"{role} must be an exact integer")
    return value


def _string_list(value: object, role: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{role} must be a sequence of strings")
    items = tuple(cast(Sequence[object], value))
    if (not allow_empty and not items) or any(
        not isinstance(item, str) or not item for item in items
    ):
        _fail(f"{role} must contain non-empty strings")
    return tuple(str(item) for item in items)


def _require_initial_integrity(run: Path) -> IntegrityVerification:
    integrity = verify_run_integrity(run)
    if not integrity.valid or not integrity.registry_record_present:
        _fail(
            "sealed pilot integrity or append-only registry verification failed: "
            f"{integrity.errors}"
        )
    if integrity.run_id != run.name:
        _fail("verified run ID does not exactly match the run directory name")
    if (
        not isinstance(integrity.expected_root_sha256, str)
        or integrity.expected_root_sha256 != integrity.actual_root_sha256
    ):
        _fail("sealed pilot artifact-root SHA-256 did not reconcile")
    return integrity


def _require_run_kind_and_terminology(
    *,
    run: Path,
    metrics: Mapping[str, Any],
    metrics_text: str,
    report_text: str,
) -> dict[str, Any]:
    status = _json_mapping(_stable_file_bytes(run / "status.json", "run status"), "run status")
    if (
        status.get("run_id") != run.name
        or status.get("experiment_name") != "pannuke_pilot"
        or status.get("status") != "completed"
        or status.get("traceback") is not None
    ):
        _fail("sealed run status is not a completed PanNuke pilot")
    if (
        metrics.get("artifact_scope") != "real_pannuke_controlled_corruption_pilot"
        or metrics.get("completion_stage_if_sealed") != "PILOT_COMPLETE"
        or metrics.get("diagnostic_claim") is not False
        or metrics.get("source_annotations_modified") is not False
    ):
        _fail("pilot metrics do not declare the required non-diagnostic PILOT_COMPLETE scope")

    folded_metrics = metrics_text.casefold()
    folded_report = report_text.casefold()
    for term in _REQUIRED_TERMS:
        if term not in folded_metrics or term not in folded_report:
            _fail(f"metrics.json and report.md must both contain the exact term: {term}")
    if "not a diagnostic" not in folded_metrics or "not a diagnostic" not in folded_report:
        _fail("metrics.json and report.md must both retain the non-diagnostic limitation")
    return {
        "required_terms": list(_REQUIRED_TERMS),
        "present_in_metrics": True,
        "present_in_report": True,
        "non_diagnostic_limitation_present": True,
    }


def _required_mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{role} must be a JSON/YAML object")
    return value


def _require_fixed_group_selection(
    selected: Mapping[str, Any], metrics: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Reconcile the fixed M6 group ledger with its aggregate metrics."""

    audit_groups = _string_list(selected.get("audit_groups"), "selected audit groups")
    reference_groups = _string_list(
        selected.get("reference_validation_groups"), "selected reference groups"
    )
    final_groups = _string_list(selected.get("final_reference_groups"), "selected final groups")
    selected_development_groups = _string_list(
        selected.get("selected_development_groups"), "selected development groups"
    )
    sample_counts = _required_mapping(metrics.get("sample_counts"), "metrics sample counts")
    development_union = set((*audit_groups, *reference_groups))
    if (
        len(audit_groups) != 225
        or len(set(audit_groups)) != 225
        or len(reference_groups) != 25
        or len(set(reference_groups)) != 25
        or len(development_union) != 250
        or len(selected_development_groups) != 250
        or len(set(selected_development_groups)) != 250
        or set(selected_development_groups) != development_union
        or selected.get("selected_development_group_limit") != 250
        or _exact_int(
            sample_counts.get("selected_development_groups"),
            "metrics selected development groups",
        )
        != 250
        or selected.get("development_official_folds") != [1, 2]
        or selected.get("selection_policy") != "fixed_deterministic_group_sample_documented_in_run"
        or selected.get("final_fold_complete") is not True
        or selected.get("final_group_limit") is not None
    ):
        _fail("sealed pilot selection does not implement the exact 90%/10% M6 group split")
    if not math.isclose(len(reference_groups) / 250.0, 0.1, rel_tol=0.0, abs_tol=0.0):
        _fail("reference-validation groups are not exactly 10% of selected development groups")
    return audit_groups, reference_groups, final_groups


def _require_fixed_m6_protocol(
    *,
    run: Path,
    metrics: Mapping[str, Any],
    final_fold: int,
) -> dict[str, Any]:
    """Require the exact fixed protocol that defined the real M6 pilot gate."""

    config_bytes = _stable_file_bytes(run / "resolved_config.yaml", "resolved pilot config")
    try:
        raw_config = yaml.safe_load(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PilotPostSealVerificationError(f"resolved pilot config is invalid: {exc}") from exc
    config = _required_mapping(raw_config, "resolved pilot config")
    data = _required_mapping(config.get("data"), "resolved pilot data config")
    corruption = _required_mapping(config.get("corruption"), "resolved pilot corruption config")
    model = _required_mapping(config.get("model"), "resolved pilot model config")
    representation = _required_mapping(
        config.get("representation"), "resolved pilot representation config"
    )
    audit = _required_mapping(config.get("audit"), "resolved pilot audit config")
    evaluation = _required_mapping(config.get("evaluation"), "resolved pilot evaluation config")
    seeds = _required_mapping(config.get("seed"), "resolved pilot seed config")
    gate = _required_mapping(config.get("gate"), "resolved pilot gate config")
    expected_scalar_fields = (
        (config.get("schema_version"), 1, "schema_version"),
        (config.get("experiment_name"), "pannuke_pilot", "experiment_name"),
        (
            config.get("configuration_role"),
            "fixed_pilot_protocol",
            "configuration_role",
        ),
        (data.get("source"), "pannuke", "data.source"),
        (data.get("development_group_limit"), 250, "data.development_group_limit"),
        (data.get("final_group_limit"), None, "data.final_group_limit"),
        (data.get("final_test_fold"), 3, "data.final_test_fold"),
        (
            data.get("group_unit"),
            "strongest_verified_or_source_patch",
            "data.group_unit",
        ),
        (data.get("reference_validation_fraction_groups"), 0.1, "reference fraction"),
        (
            data.get("final_reference_access_policy"),
            "metadata_only_until_preregistration_freeze",
            "final-reference access policy",
        ),
        (
            corruption.get("mechanism"),
            "symmetric_random_corruption",
            "corruption.mechanism",
        ),
        (corruption.get("rate"), 0.1, "corruption.rate"),
        (model.get("classifier"), "multinomial_logistic_regression", "model.classifier"),
        (model.get("oof_splits"), 5, "model.oof_splits"),
        (model.get("split_seed"), 223, "model.split_seed"),
        (model.get("seed"), 227, "model.seed"),
        (
            representation.get("encoder"),
            "official_imagenet_resnet18",
            "representation.encoder",
        ),
        (
            representation.get("weight_identifier"),
            "IMAGENET1K_V1",
            "representation.weight_identifier",
        ),
        (
            representation.get("input"),
            "target_highlighted_rgb",
            "representation.input",
        ),
        (
            evaluation.get("downstream_evaluation_partition"),
            "reference_validation",
            "evaluation.downstream_evaluation_partition",
        ),
        (evaluation.get("restoration_budget"), 0.05, "evaluation.restoration_budget"),
        (
            evaluation.get("restoration_ranking"),
            "self_confidence",
            "evaluation.restoration_ranking",
        ),
        (seeds.get("split"), 223, "seed.split"),
        (seeds.get("model"), 227, "seed.model"),
        (seeds.get("corruption"), 404, "seed.corruption"),
    )
    mismatches = [
        role for observed, expected, role in expected_scalar_fields if observed != expected
    ]
    if mismatches:
        _fail(f"sealed run does not use the exact fixed M6 pilot protocol: {mismatches}")
    if (
        data.get("development_official_folds") != [1, 2]
        or data.get("one_outer_split") is not True
        or data.get("require_all_five_positive_classes") is not True
        or corruption.get("seeds") != [404]
        or audit.get("methods") != ["self_confidence", "cleanlab", "nearest_neighbour_disagreement"]
        or evaluation.get("review_budgets") != [0.01, 0.05, 0.1, 0.2]
        or gate.get("must_complete_before_preregistration_freeze") is not True
        or gate.get("requires_verified_dataset") is not True
        or final_fold != 3
    ):
        _fail("sealed run's fixed folds/classes/methods/budgets/gate protocol is invalid")

    selected = _json_mapping(
        _stable_file_bytes(
            run / "selected_groups_and_samples.json", "pilot sample selection evidence"
        ),
        "pilot sample selection evidence",
    )
    audit_groups, reference_groups, final_groups = _require_fixed_group_selection(selected, metrics)
    if any(_SOURCE_PATCH_GROUP.fullmatch(group) is None for group in audit_groups):
        _fail("audit grouping is not at source-patch level")
    if any(_SOURCE_PATCH_GROUP.fullmatch(group) is None for group in reference_groups):
        _fail("reference-validation grouping is not at source-patch level")
    if any(_SOURCE_PATCH_GROUP.fullmatch(group) is None for group in final_groups):
        _fail("final-reference grouping is not at source-patch level")

    provenance = _json_mapping(
        _stable_file_bytes(run / "oof_provenance.json", "pilot OOF provenance"),
        "pilot OOF provenance",
    )
    if (
        provenance.get("class_order") != [0, 1, 2, 3, 4]
        or provenance.get("model_name") != "multinomial_logistic_regression"
        or provenance.get("representation") != "official_imagenet_resnet18_target_highlighted_rgb"
        or provenance.get("split_seed") != 223
        or provenance.get("model_seed") != 227
        or not isinstance(provenance.get("folds"), list)
        or len(provenance["folds"]) != 5
    ):
        _fail("sealed OOF provenance differs from the exact five-class M6 model protocol")

    corruption_manifest = _json_mapping(
        _stable_file_bytes(run / "corruption_manifest.json", "pilot corruption manifest"),
        "pilot corruption manifest",
    )
    raw_configuration_payload = corruption_manifest.get("configuration_payload")
    if not isinstance(raw_configuration_payload, str):
        _fail("pilot corruption manifest lacks its canonical configuration payload")
    try:
        corruption_payload = json.loads(raw_configuration_payload)
    except json.JSONDecodeError as exc:
        raise PilotPostSealVerificationError(
            f"pilot corruption configuration payload is invalid: {exc}"
        ) from exc
    if not isinstance(corruption_payload, Mapping) or (
        corruption_payload.get("mechanism") != "symmetric_random_corruption"
        or corruption_payload.get("rate") != 0.1
        or corruption_payload.get("corruption_seed") != 404
        or corruption_payload.get("n_classes") != 5
    ):
        _fail("pilot corruption manifest differs from the fixed 10%/seed-404 protocol")

    metric_corruption = _required_mapping(metrics.get("corruption"), "metrics corruption")
    metric_oof = _required_mapping(metrics.get("oof"), "metrics OOF")
    metric_ranking = _required_mapping(metrics.get("ranking"), "metrics ranking")
    metric_representation = _required_mapping(
        metrics.get("representation"), "metrics representation"
    )
    downstream = _required_mapping(
        metrics.get("downstream_restoration"), "metrics downstream restoration"
    )
    required_restoration = {
        "uncorrupted_reference_baseline",
        "corrupted_observed_baseline",
        "random_review_restoration",
        "audit_guided_restoration",
    }
    if (
        metric_corruption.get("mechanism") != "symmetric_random_corruption"
        or metric_corruption.get("requested_rate") != 0.1
        or metric_oof.get("fold_count") != 5
        or not {"self_confidence", "cleanlab", "nearest_neighbour_disagreement"}.issubset(
            metric_ranking
        )
        or metric_representation.get("encoder_name") != "torchvision.resnet18"
        or metric_representation.get("input_variant") != "target_highlighted_rgb"
        or metric_representation.get("weight_identifier") != "ResNet18_Weights.IMAGENET1K_V1"
        or metric_representation.get("encoder_frozen") is not True
        or not required_restoration.issubset(downstream)
        or downstream.get("review_budget_fraction") != 0.05
    ):
        _fail("pilot metrics do not evidence the exact M6 audit/representation/restoration path")
    return {
        "status": "passed",
        "protocol": "fixed_real_pannuke_m6_pilot_v1",
        "development_official_folds": [1, 2],
        "final_reference_official_fold": 3,
        "reference_validation_group_fraction": 0.1,
        "group_unit": "source_patch",
        "class_order": [0, 1, 2, 3, 4],
        "oof_fold_count": 5,
        "corruption": {
            "mechanism": "symmetric_random_corruption",
            "rate": 0.1,
            "seed": 404,
        },
        "representation": "official_imagenet_resnet18_target_highlighted_rgb",
        "classifier": "multinomial_logistic_regression",
        "required_audit_methods": [
            "self_confidence",
            "cleanlab",
            "nearest_neighbour_disagreement",
        ],
        "required_restoration_conditions_present": True,
    }


def _require_privacy_reconciliation(
    *,
    run: Path,
    final_fold: int,
) -> dict[str, Any]:
    saved = _json_mapping(
        _stable_file_bytes(
            run / "final_reference_privacy_reconciliation.json",
            "saved final-reference privacy reconciliation",
        ),
        "saved final-reference privacy reconciliation",
    )
    fresh = _format_aware_privacy_scan(run, final_fold=final_fold)
    if not isinstance(fresh, Mapping):
        _fail("format-aware privacy scan did not return an object")
    fresh = dict(fresh)
    for field in _PRIVACY_CORE_FIELDS:
        if saved.get(field) != fresh.get(field):
            _fail(f"saved and recomputed final-reference privacy evidence differ at {field}")
    if (
        any(
            fresh.get(field) is not True
            for field in (
                "final_fold_identity_pattern_absent",
                "final_sensitive_fields_unpopulated",
                "final_fold_representation_rows_absent",
            )
        )
        or fresh.get("status") != "passed"
    ):
        _fail("recomputed final-reference privacy evidence did not pass")

    saved_paths = _string_list(saved.get("scanned_paths"), "saved privacy scanned_paths")
    fresh_paths = _string_list(fresh.get("scanned_paths"), "fresh privacy scanned_paths")
    if len(saved_paths) != len(set(saved_paths)) or len(fresh_paths) != len(set(fresh_paths)):
        _fail("privacy reconciliation contains duplicate artifact paths")
    if _exact_int(saved.get("scanned_file_count"), "saved privacy scanned_file_count") != len(
        saved_paths
    ):
        _fail("saved privacy file count does not match its path ledger")
    if _exact_int(fresh.get("scanned_file_count"), "fresh privacy scanned_file_count") != len(
        fresh_paths
    ):
        _fail("fresh privacy file count does not match its path ledger")
    if not set(saved_paths).issubset(fresh_paths):
        _fail("post-seal privacy scan does not cover every artifact scanned before sealing")

    run_entries = tuple(run.rglob("*"))
    if any(entry.is_symlink() for entry in run_entries):
        _fail("sealed pilot contains a symbolic-link entry")
    actual_files = {entry.relative_to(run).as_posix() for entry in run_entries if entry.is_file()}
    if set(fresh_paths) != actual_files:
        missing = sorted(actual_files.difference(fresh_paths))
        extra = sorted(set(fresh_paths).difference(actual_files))
        _fail(
            "format-aware privacy scan did not cover the exact sealed file set: "
            f"missing={missing}, extra={extra}"
        )
    for field in (
        "npz_file_count",
        "npz_array_count",
        "parquet_file_count",
        "parquet_row_count",
    ):
        if saved.get(field) != fresh.get(field):
            _fail(f"saved and post-seal format-aware privacy counts differ at {field}")
    return {
        "status": "passed",
        "policy": fresh.get("policy"),
        "final_reference_official_fold": final_fold,
        "scanned_file_count": len(fresh_paths),
        "text_file_count": _exact_int(fresh.get("text_file_count"), "text_file_count"),
        "npz_file_count": _exact_int(fresh.get("npz_file_count"), "npz_file_count"),
        "npz_array_count": _exact_int(fresh.get("npz_array_count"), "npz_array_count"),
        "parquet_file_count": _exact_int(fresh.get("parquet_file_count"), "parquet_file_count"),
        "parquet_row_count": _exact_int(fresh.get("parquet_row_count"), "parquet_row_count"),
        "exact_sealed_file_set_scanned": True,
    }


def _require_oof_and_corruption_invariants(
    *,
    run: Path,
    metrics: Mapping[str, Any],
    final_fold: int,
) -> dict[str, Any]:
    selected = _json_mapping(
        _stable_file_bytes(
            run / "selected_groups_and_samples.json", "pilot sample selection evidence"
        ),
        "pilot sample selection evidence",
    )
    provenance = _json_mapping(
        _stable_file_bytes(run / "oof_provenance.json", "pilot OOF provenance"),
        "pilot OOF provenance",
    )
    raw_class_order = provenance.get("class_order")
    if not isinstance(raw_class_order, list) or not raw_class_order:
        _fail("pilot OOF provenance lacks a non-empty class order")
    class_order = tuple(_exact_int(value, "OOF class label") for value in raw_class_order)
    if len(class_order) != len(set(class_order)):
        _fail("pilot OOF class order contains duplicates")

    prediction_path = run / "oof_predictions.npz"
    try:
        with np.load(prediction_path, allow_pickle=False) as payload:
            required = {
                "sample_ids",
                "group_ids",
                "pre_corruption_label",
                "observed_label",
                "is_injected_corruption",
                "probabilities",
                "predicted_class",
                "fold_id",
                "self_confidence",
            }
            missing = required.difference(payload.files)
            if missing:
                _fail(f"pilot OOF evidence lacks arrays: {sorted(missing)}")
            arrays = {name: np.asarray(payload[name]) for name in required}
    except (OSError, ValueError) as exc:
        raise PilotPostSealVerificationError(
            f"pilot OOF evidence is missing, unreadable, or unsafe: {prediction_path}: {exc}"
        ) from exc

    sample_ids_raw = arrays["sample_ids"]
    group_ids_raw = arrays["group_ids"]
    pre = arrays["pre_corruption_label"]
    observed = arrays["observed_label"]
    injected = arrays["is_injected_corruption"]
    probabilities = arrays["probabilities"]
    predicted = arrays["predicted_class"]
    fold_ids = arrays["fold_id"]
    self_confidence = arrays["self_confidence"]
    if sample_ids_raw.dtype.kind not in {"U", "S"} or group_ids_raw.dtype.kind not in {"U", "S"}:
        _fail("pilot OOF sample_ids and group_ids must be string arrays")
    if any(array.dtype.kind not in {"i", "u"} for array in (pre, observed, predicted, fold_ids)):
        _fail("pilot OOF labels, predictions, and fold IDs must be exact integer arrays")
    if injected.dtype.kind != "b":
        _fail("pilot OOF is_injected_corruption must be a boolean array")
    if probabilities.dtype.kind != "f":
        _fail("pilot OOF probabilities must be a floating-point array")

    sample_ids = tuple(str(value) for value in sample_ids_raw.tolist())
    group_ids = tuple(str(value) for value in group_ids_raw.tolist())
    n_samples = len(sample_ids)
    vector_shapes = (
        sample_ids_raw.shape,
        group_ids_raw.shape,
        pre.shape,
        observed.shape,
        injected.shape,
        predicted.shape,
        fold_ids.shape,
        self_confidence.shape,
    )
    if (
        n_samples <= 0
        or len(set(sample_ids)) != n_samples
        or any(not value for value in (*sample_ids, *group_ids))
        or any(shape != (n_samples,) for shape in vector_shapes)
        or probabilities.shape != (n_samples, len(class_order))
    ):
        _fail("pilot OOF arrays do not share one exact, unique sample order")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        _fail("pilot OOF probabilities must be finite and lie in [0, 1]")
    probability_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if probability_error > 1e-12:
        _fail("pilot OOF probability rows do not sum to one within 1e-12")
    expected_predictions = np.asarray(class_order, dtype=np.int64)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(predicted, expected_predictions):
        _fail("pilot OOF predicted classes do not equal probability argmax decisions")

    valid_labels = set(class_order)
    if not set(int(value) for value in pre.tolist()).issubset(valid_labels) or not set(
        int(value) for value in observed.tolist()
    ).issubset(valid_labels):
        _fail("pilot OOF labels fall outside the fixed class order")
    class_to_column = {label: index for index, label in enumerate(class_order)}
    expected_self_confidence = np.asarray(
        [
            1.0 - probabilities[index, class_to_column[int(label)]]
            for index, label in enumerate(observed.tolist())
        ],
        dtype=np.float64,
    )
    if (
        self_confidence.dtype.kind != "f"
        or not np.isfinite(self_confidence).all()
        or not np.allclose(self_confidence, expected_self_confidence, rtol=0.0, atol=1e-12)
    ):
        _fail("pilot self-confidence risk does not reconstruct from OOF probabilities")
    changed = pre != observed
    if not np.array_equal(injected, changed):
        _fail("is_injected_corruption must exactly equal (pre_corruption_label != observed_label)")
    exact_corruption_count = int(changed.sum())

    group_to_fold: dict[str, int] = {}
    for group_id, fold_id in zip(group_ids, fold_ids.tolist(), strict=True):
        clean_fold = int(fold_id)
        prior = group_to_fold.setdefault(group_id, clean_fold)
        if prior != clean_fold:
            _fail("one source group appears in more than one OOF holdout fold")
    expected_fold_ids = {
        _exact_int(fold.get("fold_id"), "OOF provenance fold_id")
        for fold in provenance.get("folds", [])
        if isinstance(fold, Mapping)
    }
    if not expected_fold_ids or set(int(value) for value in fold_ids.tolist()) != expected_fold_ids:
        _fail("pilot OOF fold IDs do not exactly match OOF provenance")
    if provenance.get("fold_assignment_label_source") != "pre_corruption_label":
        _fail("controlled pilot OOF allocation must use pre_corruption_label only for splitting")

    selected_ids = _string_list(selected.get("audit_sample_ids"), "selected audit_sample_ids")
    audit_groups = _string_list(selected.get("audit_groups"), "selected audit_groups")
    reference_groups = _string_list(
        selected.get("reference_validation_groups"), "reference-validation groups"
    )
    final_groups = _string_list(selected.get("final_reference_groups"), "final-reference groups")
    if selected_ids != sample_ids or set(audit_groups) != set(group_ids):
        _fail("selected audit sample/group identities differ from OOF evidence")
    if (
        set(audit_groups).intersection(reference_groups)
        or set(audit_groups).intersection(final_groups)
        or set(reference_groups).intersection(final_groups)
    ):
        _fail("audit, reference-validation, and final-reference groups are not disjoint")
    if (
        selected.get("no_group_overlap_verified") is not True
        or _exact_int(selected.get("final_test_fold"), "selected final_test_fold") != final_fold
        or selected.get("final_reference_class_labels_read") is not False
        or selected.get("final_reference_outcomes_used") is not False
        or selected.get("final_reference_representations_extracted") is not False
        or selected.get("final_reference_sample_ids_read") is not False
    ):
        _fail("pilot selection evidence violates the untouched final-reference policy")
    pairwise = selected.get("pairwise_group_overlap_counts")
    if (
        not isinstance(pairwise, Mapping)
        or not pairwise
        or any(
            _exact_int(value, "pairwise group overlap count") != 0 for value in pairwise.values()
        )
    ):
        _fail("pilot selection does not record zero pairwise group overlap")

    corruption_metrics = metrics.get("corruption")
    oof_metrics = metrics.get("oof")
    sample_counts = metrics.get("sample_counts")
    final_access = metrics.get("final_reference_access")
    if not all(
        isinstance(value, Mapping)
        for value in (corruption_metrics, oof_metrics, sample_counts, final_access)
    ):
        _fail("pilot metrics lack corruption, OOF, sample-count, or final-access evidence")
    assert isinstance(corruption_metrics, Mapping)
    assert isinstance(oof_metrics, Mapping)
    assert isinstance(sample_counts, Mapping)
    assert isinstance(final_access, Mapping)
    requested_rate = corruption_metrics.get("requested_rate")
    if type(requested_rate) not in {int, float} or isinstance(requested_rate, bool):
        _fail("pilot corruption rate is not numeric")
    numeric_rate = cast(int | float, requested_rate)
    expected_corruption_count = math.floor(n_samples * float(numeric_rate) + 0.5)
    if (
        _exact_int(corruption_metrics.get("exact_count"), "metrics corruption exact_count")
        != exact_corruption_count
        or expected_corruption_count != exact_corruption_count
        or corruption_metrics.get("only_audit_pool_corrupted") is not True
        or corruption_metrics.get("final_reference_fold_uncorrupted") is not True
    ):
        _fail("pilot corruption metrics do not reconcile with exact changed labels")
    if (
        oof_metrics.get("complete_once_coverage") is not True
        or oof_metrics.get("final_reference_groups_excluded") is not True
        or _exact_int(oof_metrics.get("group_overlap_count"), "OOF group_overlap_count") != 0
        or _exact_int(oof_metrics.get("fold_count"), "OOF fold_count") != len(expected_fold_ids)
        or not math.isclose(
            float(oof_metrics.get("probability_sum_maximum_error", math.inf)),
            probability_error,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        _fail("pilot OOF metrics do not reconcile with saved prediction evidence")
    if (
        _exact_int(sample_counts.get("audit_pool"), "metrics audit_pool") != n_samples
        or _exact_int(sample_counts.get("audit_groups"), "metrics audit_groups")
        != len(set(group_ids))
        or _exact_int(sample_counts.get("reference_validation"), "metrics reference_validation")
        != len(_string_list(selected.get("reference_validation_sample_ids"), "reference IDs"))
        or _exact_int(sample_counts.get("final_reference_test"), "metrics final_reference_test")
        != _exact_int(selected.get("final_reference_sample_count"), "selected final count")
    ):
        _fail("pilot sample counts do not reconcile with selection and OOF evidence")
    if _exact_int(
        final_access.get("official_fold"), "final_reference_access.official_fold"
    ) != final_fold or any(
        final_access.get(field) is not False for field in _NO_OUTCOME_FINAL_ACCESS_FIELDS
    ):
        _fail("pilot metrics violate final-reference outcome blindness")
    return {
        "status": "passed",
        "sample_count": n_samples,
        "group_count": len(set(group_ids)),
        "fold_count": len(expected_fold_ids),
        "group_overlap_count": 0,
        "complete_once_coverage": True,
        "probability_sum_maximum_error": probability_error,
        "exact_corruption_count": exact_corruption_count,
        "corruption_label_separation_exact": True,
        "final_reference_outcomes_unavailable": True,
    }


def verify_pilot_post_seal(
    run_directory: str | Path,
    *,
    development_manifest_source: str | Path,
    gate_certificate_source: str | Path,
) -> dict[str, Any]:
    """Verify a sealed PanNuke pilot without writing or withdrawing anything.

    The two external source paths are mandatory.  A run-local copy cannot be used as
    its own byte-identity authority.
    """

    run = Path(run_directory).resolve()
    development_source = Path(development_manifest_source).resolve()
    certificate_source = Path(gate_certificate_source).resolve()
    if not run.is_dir() or run.is_symlink():
        _fail(f"run directory must be a regular non-symlink directory: {run}")
    run_local_development = run / "development_manifest_view.parquet"
    run_local_certificate = run / "pre_pilot_gate_certificate.json"
    if (
        development_source == run_local_development.resolve()
        or certificate_source == run_local_certificate.resolve()
        or development_source == certificate_source
        or run in development_source.parents
        or run in certificate_source.parents
    ):
        _fail("external byte-identity authorities must be distinct files outside the sealed run")

    development_source_bytes = _stable_file_bytes(
        development_source, "external development manifest"
    )
    certificate_source_bytes = _stable_file_bytes(
        certificate_source, "external pre-pilot gate certificate"
    )
    integrity = _require_initial_integrity(run)
    try:
        require_run_stage_eligible(run, integrity=integrity)
    except (OSError, ValueError, RuntimeError) as exc:
        raise PilotPostSealVerificationError(
            f"sealed pilot is not scientifically stage-eligible: {exc}"
        ) from exc

    disposition_registry = run.parent / RUN_DISPOSITION_REGISTRY_FILENAME
    disposition_anchor = run.parent / RUN_DISPOSITION_ANCHOR_FILENAME
    disposition_records = read_run_dispositions(disposition_registry)
    disposition_ledger_bytes = _optional_stable_file_bytes(
        disposition_registry, "run disposition ledger"
    )
    disposition_anchor_bytes = _stable_file_bytes(disposition_anchor, "run disposition anchor")
    if any(record.get("run_id") == run.name for record in disposition_records):
        _fail("stage-eligible run unexpectedly has a binding disposition record")

    local_development_bytes = _stable_file_bytes(
        run_local_development, "run-local development manifest"
    )
    local_certificate_bytes = _stable_file_bytes(
        run_local_certificate, "run-local pre-pilot gate certificate"
    )
    if local_development_bytes != development_source_bytes:
        _fail("run-local development manifest is not byte-identical to the explicit source")
    if local_certificate_bytes != certificate_source_bytes:
        _fail("run-local gate certificate is not byte-identical to the explicit source")

    metrics_bytes = _stable_file_bytes(run / "metrics.json", "pilot metrics")
    report_bytes = _stable_file_bytes(run / "report.md", "pilot report")
    metrics = _json_mapping(metrics_bytes, "pilot metrics")
    try:
        metrics_text = metrics_bytes.decode("utf-8")
        report_text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotPostSealVerificationError(
            "pilot metrics and report must be valid UTF-8"
        ) from exc
    terminology = _require_run_kind_and_terminology(
        run=run,
        metrics=metrics,
        metrics_text=metrics_text,
        report_text=report_text,
    )
    final_access = metrics.get("final_reference_access")
    if not isinstance(final_access, Mapping):
        _fail("pilot metrics lack final-reference access evidence")
    final_fold = _exact_int(final_access.get("official_fold"), "final reference official fold")
    fixed_protocol = _require_fixed_m6_protocol(
        run=run,
        metrics=metrics,
        final_fold=final_fold,
    )

    saved_audit = _json_mapping(
        _stable_file_bytes(
            run / "audit_evidence_reconciliation.json", "saved pilot audit reconciliation"
        ),
        "saved pilot audit reconciliation",
    )
    fresh_audit = reconcile_pilot_audit_evidence(run, require_sealed_integrity=True)
    if fresh_audit != saved_audit or fresh_audit.get("status") != "passed":
        _fail("saved and independently recomputed pilot audit reconciliation differ")
    privacy = _require_privacy_reconciliation(run=run, final_fold=final_fold)
    oof_corruption = _require_oof_and_corruption_invariants(
        run=run,
        metrics=metrics,
        final_fold=final_fold,
    )

    final_integrity = _require_initial_integrity(run)
    if final_integrity != integrity:
        _fail("sealed pilot integrity evidence changed during post-seal verification")
    try:
        require_run_stage_eligible(run, integrity=final_integrity)
    except (OSError, ValueError, RuntimeError) as exc:
        raise PilotPostSealVerificationError(
            f"sealed pilot eligibility changed during post-seal verification: {exc}"
        ) from exc
    if read_run_dispositions(disposition_registry) != disposition_records:
        _fail("run disposition records changed during post-seal verification")
    if (
        _optional_stable_file_bytes(disposition_registry, "run disposition ledger")
        != disposition_ledger_bytes
        or _stable_file_bytes(disposition_anchor, "run disposition anchor")
        != disposition_anchor_bytes
    ):
        _fail("run disposition ledger or anchor changed during post-seal verification")
    if (
        _stable_file_bytes(development_source, "external development manifest")
        != development_source_bytes
        or _stable_file_bytes(certificate_source, "external pre-pilot gate certificate")
        != certificate_source_bytes
    ):
        _fail("an external byte-identity authority changed during verification")

    return {
        "schema_version": 1,
        "status": "passed",
        "policy": "read_only_pilot_post_seal_verification_v1",
        "run_id": run.name,
        "run_directory": str(run),
        "scientific_stage_eligible": True,
        "sealed_run_unchanged": True,
        "integrity": {
            "valid": True,
            "registry_record_present": True,
            "artifact_root_sha256": integrity.actual_root_sha256,
            "artifact_manifest_root_sha256": integrity.expected_root_sha256,
        },
        "disposition": {
            "eligible": True,
            "matching_withdrawal_record_count": 0,
            "verified_record_count": len(disposition_records),
            "ledger_sha256": _sha256_bytes(disposition_ledger_bytes or b""),
            "anchor_sha256": _sha256_bytes(disposition_anchor_bytes),
        },
        "byte_identity": {
            "development_manifest": {
                "source_path": str(development_source),
                "run_path": str(run_local_development),
                "sha256": _sha256_bytes(development_source_bytes),
                "byte_identical": True,
            },
            "gate_certificate": {
                "source_path": str(certificate_source),
                "run_path": str(run_local_certificate),
                "sha256": _sha256_bytes(certificate_source_bytes),
                "byte_identical": True,
            },
        },
        "terminology": terminology,
        "fixed_m6_protocol": fixed_protocol,
        "format_aware_privacy": privacy,
        "audit_reconciliation": fresh_audit,
        "oof_and_corruption": oof_corruption,
        "automatic_withdrawal_performed": False,
    }


__all__ = [
    "PilotPostSealVerificationError",
    "verify_pilot_post_seal",
]
