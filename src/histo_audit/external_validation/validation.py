"""Structural and blinding validation for generated expert-review packages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_PACKAGE_FILES = (
    "review_items.csv",
    "response_template.csv",
    "response_schema.json",
    "review.html",
    "package_metadata.json",
)
_FORBIDDEN_REVIEW_COLUMNS = (
    "sample_id",
    "source",
    "group_id",
    "patch_id",
    "instance_id",
    "pre_corruption",
    "corruption",
    "risk",
    "score",
    "selection",
    "prediction",
    "suggest",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReviewPackageValidationResult:
    """Validation evidence without claiming any expert response exists."""

    valid: bool
    package_directory: Path
    item_count: int
    asset_count: int
    private_linkage_validated: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible validation evidence."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


def _json_mapping(path: Path, *, role: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{role} is invalid: {exc}")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{role} must be a JSON object")
        return {}
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_blinded_review_package(
    package_directory: str | Path,
    *,
    private_unblinding_key_path: str | Path | None = None,
) -> ReviewPackageValidationResult:
    """Validate reviewer-facing blinding and, when supplied, private linkage."""

    package = Path(package_directory).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not package.is_dir():
        return ReviewPackageValidationResult(
            False, package, 0, 0, False, (f"review package does not exist: {package}",), ()
        )
    missing = [name for name in _REQUIRED_PACKAGE_FILES if not (package / name).is_file()]
    if missing:
        return ReviewPackageValidationResult(
            False, package, 0, 0, False, (f"review package lacks required files: {missing}",), ()
        )
    try:
        items = pd.read_csv(package / "review_items.csv", keep_default_na=False)
        responses = pd.read_csv(package / "response_template.csv", keep_default_na=False)
    except Exception as exc:
        return ReviewPackageValidationResult(
            False, package, 0, 0, False, (f"review CSV parsing failed: {exc}",), ()
        )
    metadata = _json_mapping(
        package / "package_metadata.json", role="package metadata", errors=errors
    )
    schema = _json_mapping(package / "response_schema.json", role="response schema", errors=errors)
    required_item_columns = {"review_id", "display_order"}
    if not required_item_columns.issubset(items.columns):
        errors.append(
            f"review items lack columns: {sorted(required_item_columns.difference(items.columns))}"
        )
    if items.empty:
        errors.append("review items are empty")
    elif items["review_id"].astype(str).duplicated().any():
        errors.append("review IDs are not unique")
    unsafe_columns = [
        column
        for column in items.columns
        if any(fragment in str(column).casefold() for fragment in _FORBIDDEN_REVIEW_COLUMNS)
    ]
    if unsafe_columns:
        errors.append(f"review-facing columns break blinding: {unsafe_columns}")
    required_response_columns = {"review_id", "reviewer_id", "response", "notes"}
    if not required_response_columns.issubset(responses.columns):
        errors.append(
            f"response template lacks columns: {sorted(required_response_columns.difference(responses.columns))}"
        )
    else:
        if bool(
            responses[["reviewer_id", "response", "notes"]]
            .astype(str)
            .apply(lambda column: column.str.strip().ne(""))
            .to_numpy()
            .any()
        ):
            errors.append(
                "response template is not blank; generated expert responses are forbidden"
            )
        if set(responses["review_id"].astype(str)) != set(
            items.get("review_id", pd.Series(dtype=str)).astype(str)
        ):
            errors.append("response-template review IDs differ from review items")
    expected_count = metadata.get("total_count")
    if not isinstance(expected_count, int) or expected_count != len(items):
        errors.append("package metadata total_count differs from review items")
    if metadata.get("non_diagnostic") is not True:
        errors.append("package metadata must explicitly mark the package non-diagnostic")
    if not str(metadata.get("response_state", "")).startswith("blank_template_only"):
        errors.append("package metadata does not certify a blank response template")
    study_outcome_eligible = metadata.get("study_outcome_eligible")
    eligibility_evidence = metadata.get("eligibility_evidence")
    if not isinstance(study_outcome_eligible, bool):
        errors.append("package metadata lacks an explicit boolean study_outcome_eligible flag")
    elif study_outcome_eligible:
        required_evidence = {
            "audit_run_id",
            "audit_artifact_root_sha256",
            "dataset_sha256",
            "manifest_sha256",
            "dataset_validation_sha256",
            "duplicate_audit_sha256",
            "ranking_sha256",
            "confirmatory_run_id",
            "confirmatory_artifact_root_sha256",
            "confirmatory_completion_sha256",
            "feature_cache_provenance_sha256",
        }
        if not isinstance(eligibility_evidence, Mapping):
            errors.append("stage-eligible package metadata lacks eligibility evidence")
        else:
            missing_evidence = required_evidence.difference(eligibility_evidence)
            if missing_evidence:
                errors.append(
                    "stage-eligible package metadata lacks evidence fields: "
                    f"{sorted(missing_evidence)}"
                )
            if not str(eligibility_evidence.get("audit_run_id", "")).strip():
                errors.append("stage-eligible package metadata lacks an audit-run identity")
            if not str(eligibility_evidence.get("confirmatory_run_id", "")).strip():
                errors.append("stage-eligible package metadata lacks a confirmatory-run identity")
            for field in required_evidence.difference({"audit_run_id", "confirmatory_run_id"}):
                if not _SHA256.fullmatch(str(eligibility_evidence.get(field, ""))):
                    errors.append(f"package eligibility evidence {field} is not a SHA-256 digest")
    elif eligibility_evidence is not None:
        errors.append("non-stage package metadata must not carry eligibility evidence")
    blinding = metadata.get("blinding")
    if (
        not isinstance(blinding, Mapping)
        or not blinding
        or not all(value is True for value in blinding.values())
    ):
        errors.append("package metadata does not certify every declared blinding safeguard")

    asset_columns = [column for column in items.columns if str(column).startswith("asset_")]
    expected_roles = metadata.get("asset_roles")
    if not isinstance(expected_roles, list) or {
        str(column).removeprefix("asset_") for column in asset_columns
    } != {str(role) for role in expected_roles}:
        errors.append("review asset columns differ from package metadata roles")
    asset_count = 0
    for column in asset_columns:
        for raw in items[column].astype(str):
            relative = Path(raw)
            resolved = (package / relative).resolve()
            if (
                not raw
                or relative.is_absolute()
                or not _inside(resolved, package)
                or not resolved.is_file()
            ):
                errors.append(f"review asset is missing or escapes package: {raw!r}")
            else:
                asset_count += 1

    try:
        enum_ids = (
            schema.get("items", {}).get("properties", {}).get("review_id", {}).get("enum", [])
        )
    except AttributeError:
        enum_ids = []
    if set(str(value) for value in enum_ids) != set(
        items.get("review_id", pd.Series(dtype=str)).astype(str)
    ):
        errors.append("response schema review-ID enum differs from review items")

    private_validated = False
    if private_unblinding_key_path is None:
        warnings.append("private unblinding key was not supplied and its linkage was not validated")
    else:
        key_path = Path(private_unblinding_key_path).resolve()
        if _inside(key_path, package):
            errors.append("private unblinding key is inside the reviewer-facing package")
        elif not key_path.is_file():
            errors.append(f"private unblinding key does not exist: {key_path}")
        else:
            try:
                key = pd.read_csv(key_path)
            except Exception as exc:
                errors.append(f"private unblinding key parsing failed: {exc}")
            else:
                required_key_columns = {
                    "review_id",
                    "sample_id",
                    "selection_source",
                    "rank_position_among_joined",
                    "ranking_score",
                }
                if not required_key_columns.issubset(key.columns):
                    errors.append(
                        f"private key lacks columns: {sorted(required_key_columns.difference(key.columns))}"
                    )
                else:
                    key_ids = key["review_id"].astype(str)
                    if key_ids.duplicated().any() or set(key_ids) != set(
                        items["review_id"].astype(str)
                    ):
                        errors.append(
                            "private key review IDs are duplicate or differ from review items"
                        )
                    if key["sample_id"].astype(str).duplicated().any():
                        errors.append("private key reuses a source sample")
                    cohorts = set(key["selection_source"].astype(str))
                    if cohorts != {"top_ranked", "random"}:
                        errors.append(
                            "private key does not contain exactly top-ranked and random cohorts"
                        )
                    numeric_scores = pd.to_numeric(key["ranking_score"], errors="coerce")
                    if (
                        numeric_scores.isna().any()
                        or not np.isfinite(numeric_scores.to_numpy()).all()
                    ):
                        errors.append("private key ranking scores are not all finite")
                    top_count = int((key["selection_source"] == "top_ranked").sum())
                    random_count = int((key["selection_source"] == "random").sum())
                    if top_count != metadata.get("top_count") or random_count != metadata.get(
                        "random_count"
                    ):
                        errors.append("private cohort counts differ from package metadata")
                    private_validated = not errors

    return ReviewPackageValidationResult(
        valid=not errors,
        package_directory=package,
        item_count=len(items),
        asset_count=asset_count,
        private_linkage_validated=private_validated,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = ["ReviewPackageValidationResult", "validate_blinded_review_package"]
