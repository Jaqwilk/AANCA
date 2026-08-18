"""Shared fail-closed PanNuke analysis-eligibility contract.

The canonical nucleus manifest retains every raw instance, including instances
touching a cross-class overlap.  Analysis caches are a derived view: they may
contain only rows selected by the one shared primary/confirmatory eligibility
mask.  Explicit review-only extraction is kept separate and can retain excluded
rows without making those rows analysis eligible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

ELIGIBILITY_POLICY = "one_identical_primary_and_confirmatory_instance_mask"
OVERLAP_EXCLUSION_REASON = "touches_cross_class_overlap"
EligibilityScope = Literal["analysis", "review_only"]
MANIFEST_VIEW_METADATA_KEY = b"histo_audit_analysis_manifest_view"
DEVELOPMENT_MANIFEST_VIEW_SCOPE = "development_official_folds_only"


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validated_manifest_view_record(
    value: Mapping[str, Any],
    *,
    frame: Any | None = None,
    eligible_ids: Sequence[str] | None = None,
    excluded_ids: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the privacy boundary of a derived development-only manifest view."""

    record = dict(value)
    semantic = record.pop("semantic_sha256", None)
    if semantic != _semantic_sha256(record):
        raise ValueError("development manifest-view metadata hash is invalid")
    expected_fields = {
        "schema_version",
        "scope",
        "derivation_policy",
        "included_official_folds",
        "excluded_official_folds",
        "contains_final_reference_sample_ids",
        "contains_final_reference_class_labels",
        "canonical_manifest_sha256",
        "canonical_manifest_class_free_eligibility_sha256",
        "development_class_free_eligibility_sha256",
        "manifest_instance_count",
        "manifest_eligible_instance_count",
        "manifest_excluded_instance_count",
        "manifest_eligible_sample_ids_sha256",
        "manifest_excluded_sample_ids_sha256",
        "source_annotations_modified",
    }
    if set(record) != expected_fields:
        raise ValueError("development manifest-view metadata contains unexpected fields")
    if (
        record.get("schema_version") != 1
        or record.get("scope") != DEVELOPMENT_MANIFEST_VIEW_SCOPE
        or record.get("derivation_policy")
        != "complete_canonical_development_rows_selected_by_official_fold"
        or record.get("contains_final_reference_sample_ids") is not False
        or record.get("contains_final_reference_class_labels") is not False
        or record.get("source_annotations_modified") is not False
    ):
        raise ValueError("development manifest-view privacy policy is invalid")
    included = record.get("included_official_folds")
    excluded = record.get("excluded_official_folds")
    if (
        not isinstance(included, list)
        or not included
        or any(type(item) is not int or item <= 0 for item in included)
        or len(set(included)) != len(included)
        or not isinstance(excluded, list)
        or not excluded
        or any(type(item) is not int or item <= 0 for item in excluded)
        or len(set(excluded)) != len(excluded)
        or set(included).intersection(excluded)
    ):
        raise ValueError("development manifest-view fold scope is invalid")
    for field in (
        "canonical_manifest_sha256",
        "canonical_manifest_class_free_eligibility_sha256",
        "development_class_free_eligibility_sha256",
        "manifest_eligible_sample_ids_sha256",
        "manifest_excluded_sample_ids_sha256",
    ):
        if not _is_sha256(record.get(field)):
            raise ValueError(f"development manifest-view field {field} is invalid")
    counts = (
        record.get("manifest_instance_count"),
        record.get("manifest_eligible_instance_count"),
        record.get("manifest_excluded_instance_count"),
    )
    if any(type(item) is not int or item < 0 for item in counts):
        raise ValueError("development manifest-view support counts are invalid")
    total, eligible, excluded_count = cast(tuple[int, int, int], counts)
    if total != eligible + excluded_count or total <= 0:
        raise ValueError("development manifest-view support counts do not reconcile")
    if frame is not None:
        if "official_fold" not in frame.columns:
            raise ValueError("development manifest view lacks official_fold")
        actual_folds = set(int(item) for item in frame["official_fold"].unique())
        if actual_folds != set(included):
            raise ValueError("development manifest-view rows differ from declared folds")
        if len(frame) != total:
            raise ValueError("development manifest-view row count is inconsistent")
    if eligible_ids is not None and excluded_ids is not None:
        if len(eligible_ids) != eligible or len(excluded_ids) != excluded_count:
            raise ValueError("development manifest-view identity counts are inconsistent")
        if record["manifest_eligible_sample_ids_sha256"] != _ordered_ids_sha256(eligible_ids):
            raise ValueError("development manifest-view eligible identity hash is invalid")
        if record["manifest_excluded_sample_ids_sha256"] != _ordered_ids_sha256(excluded_ids):
            raise ValueError("development manifest-view excluded identity hash is invalid")
    if provenance is not None:
        for field in (
            "manifest_instance_count",
            "manifest_eligible_instance_count",
            "manifest_excluded_instance_count",
            "manifest_eligible_sample_ids_sha256",
            "manifest_excluded_sample_ids_sha256",
        ):
            if provenance.get(field) != record.get(field):
                raise ValueError(
                    f"development manifest-view field {field} differs from eligibility provenance"
                )
    record["semantic_sha256"] = semantic
    return record


def _manifest_view_record(
    table: Any,
    frame: Any,
    *,
    eligible_ids: Sequence[str],
    excluded_ids: Sequence[str],
) -> dict[str, Any] | None:
    metadata = table.schema.metadata or {}
    encoded = metadata.get(MANIFEST_VIEW_METADATA_KEY)
    if encoded is None:
        return None
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("development manifest-view metadata is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("development manifest-view metadata must be a mapping")
    return _validated_manifest_view_record(
        value,
        frame=frame,
        eligible_ids=eligible_ids,
        excluded_ids=excluded_ids,
    )


@dataclass(frozen=True, slots=True)
class ManifestEligibilitySelection:
    """Selected manifest rows plus complete exclusion/support provenance."""

    frame: Any
    provenance: dict[str, Any]


def select_manifest_rows(
    table: Any,
    *,
    sample_ids: tuple[str, ...] | None,
    scope: EligibilityScope,
) -> ManifestEligibilitySelection:
    """Select manifest rows under the fixed analysis or explicit review scope.

    Analysis selection filters the complete manifest before any feature
    extraction.  If callers provide an explicit analysis list, requesting even
    one excluded ID is an error rather than a silent order/count change.
    Review-only selection requires explicit IDs and never changes eligibility.
    """

    if scope not in {"analysis", "review_only"}:
        raise ValueError(f"unknown PanNuke eligibility scope: {scope}")
    frame = table.to_pandas()
    required = {
        "sample_id",
        "primary_eligible",
        "confirmatory_eligible",
        "cross_class_overlap_touching",
        "qc_exclusion_reason",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"manifest lacks analysis-eligibility fields: {missing}")
    identifiers = tuple(frame["sample_id"].astype(str).tolist())
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("manifest sample IDs must be non-empty and unique")
    primary = np.asarray(frame["primary_eligible"], dtype=np.bool_)
    confirmatory = np.asarray(frame["confirmatory_eligible"], dtype=np.bool_)
    touching = np.asarray(frame["cross_class_overlap_touching"], dtype=np.bool_)
    if not np.array_equal(primary, confirmatory):
        raise ValueError("primary and confirmatory eligibility masks differ")
    if not np.array_equal(~primary, touching):
        raise ValueError("analysis exclusion mask is not exactly overlap-touching instances")
    reasons = frame["qc_exclusion_reason"]
    for index, is_eligible in enumerate(primary.tolist()):
        reason = reasons.iloc[index]
        missing_reason = reason is None or (isinstance(reason, float) and np.isnan(reason))
        if is_eligible:
            if not missing_reason:
                raise ValueError("eligible manifest row carries a QC exclusion reason")
        elif reason != OVERLAP_EXCLUSION_REASON:
            raise ValueError("excluded manifest row lacks the fixed overlap exclusion reason")

    eligible_ids = tuple(value for value, keep in zip(identifiers, primary, strict=True) if keep)
    excluded_ids = tuple(
        value for value, keep in zip(identifiers, primary, strict=True) if not keep
    )
    if sample_ids is not None:
        if (
            not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or any(not value for value in sample_ids)
        ):
            raise ValueError("requested sample IDs must be non-empty and unique")
        by_id = frame.set_index("sample_id", drop=False)
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id.index]
        if missing:
            raise KeyError(f"manifest does not contain requested sample IDs: {missing[:5]}")
        selected = by_id.loc[list(sample_ids)].reset_index(drop=True)
    elif scope == "review_only":
        raise ValueError("review-only extraction requires explicit sample_ids")
    else:
        selected = frame.loc[primary].reset_index(drop=True)

    selected_primary = np.asarray(selected["primary_eligible"], dtype=np.bool_)
    selected_confirmatory = np.asarray(selected["confirmatory_eligible"], dtype=np.bool_)
    selected_ids = tuple(selected["sample_id"].astype(str).tolist())
    if scope == "analysis" and (not selected_primary.all() or not selected_confirmatory.all()):
        rejected = tuple(
            sample_id
            for sample_id, keep in zip(selected_ids, selected_primary, strict=True)
            if not keep
        )
        raise ValueError(
            "analysis sample IDs include instances excluded by "
            f"{OVERLAP_EXCLUSION_REASON}: {list(rejected[:5])}"
        )
    if selected.empty:
        raise ValueError(f"PanNuke {scope} eligibility selection is empty")

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "cross_class_overlap_exclusion_reason": OVERLAP_EXCLUSION_REASON,
        "selection_scope": scope,
        "manifest_instance_count": len(identifiers),
        "manifest_eligible_instance_count": len(eligible_ids),
        "manifest_excluded_instance_count": len(excluded_ids),
        "manifest_eligible_sample_ids_sha256": _ordered_ids_sha256(eligible_ids),
        "manifest_excluded_sample_ids": list(excluded_ids),
        "manifest_excluded_sample_ids_sha256": _ordered_ids_sha256(excluded_ids),
        "output_sample_count": len(selected_ids),
        "output_sample_ids_sha256": _ordered_ids_sha256(selected_ids),
        "all_output_primary_eligible": bool(selected_primary.all()),
        "all_output_confirmatory_eligible": bool(selected_confirmatory.all()),
        "source_annotations_modified": False,
    }
    manifest_view = _manifest_view_record(
        table,
        frame,
        eligible_ids=eligible_ids,
        excluded_ids=excluded_ids,
    )
    if manifest_view is not None:
        provenance["manifest_view"] = manifest_view
    provenance["semantic_sha256"] = _semantic_sha256(provenance)
    return ManifestEligibilitySelection(frame=selected, provenance=provenance)


def validate_analysis_eligibility_provenance(
    metadata: Mapping[str, Any],
    sample_ids: Sequence[str] | NDArray[np.str_],
    *,
    primary_eligible: NDArray[np.generic] | None = None,
    confirmatory_eligible: NDArray[np.generic] | None = None,
) -> dict[str, Any]:
    """Validate that a cache contains only the shared analysis-eligible view."""

    raw = metadata.get("analysis_eligibility")
    if not isinstance(raw, Mapping):
        raise ValueError("cache lacks analysis-eligibility provenance")
    provenance = dict(raw)
    semantic = provenance.pop("semantic_sha256", None)
    if semantic != _semantic_sha256(provenance):
        raise ValueError("cache analysis-eligibility provenance hash is invalid")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("eligibility_policy") != ELIGIBILITY_POLICY
        or provenance.get("cross_class_overlap_exclusion_reason") != OVERLAP_EXCLUSION_REASON
        or provenance.get("selection_scope") != "analysis"
        or provenance.get("source_annotations_modified") is not False
    ):
        raise ValueError("cache analysis-eligibility policy is invalid")
    identifiers = tuple(str(value) for value in np.asarray(sample_ids, dtype=np.str_).tolist())
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("cache analysis sample IDs must be non-empty and unique")
    counts = (
        provenance.get("manifest_instance_count"),
        provenance.get("manifest_eligible_instance_count"),
        provenance.get("manifest_excluded_instance_count"),
        provenance.get("output_sample_count"),
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("cache analysis-eligibility support counts are invalid")
    total, eligible, excluded, output = cast(tuple[int, int, int, int], counts)
    if total != eligible + excluded or output != len(identifiers) or output > eligible:
        raise ValueError("cache analysis-eligibility support counts do not reconcile")
    excluded_ids_raw = provenance.get("manifest_excluded_sample_ids")
    if not isinstance(excluded_ids_raw, list) or any(
        not isinstance(value, str) or not value for value in excluded_ids_raw
    ):
        raise ValueError("cache excluded sample-ID ledger is invalid")
    excluded_ids = tuple(excluded_ids_raw)
    if len(excluded_ids) != excluded or len(set(excluded_ids)) != excluded:
        raise ValueError("cache excluded sample-ID ledger count is invalid")
    if set(identifiers).intersection(excluded_ids):
        raise ValueError("cache contains a manifest-excluded sample ID")
    for field in (
        "manifest_eligible_sample_ids_sha256",
        "manifest_excluded_sample_ids_sha256",
        "output_sample_ids_sha256",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"cache analysis-eligibility field {field} is invalid")
    if provenance["manifest_excluded_sample_ids_sha256"] != _ordered_ids_sha256(excluded_ids):
        raise ValueError("cache excluded sample-ID ledger hash is invalid")
    if provenance["output_sample_ids_sha256"] != _ordered_ids_sha256(identifiers):
        raise ValueError("cache output sample-ID eligibility hash is invalid")
    if (
        provenance.get("all_output_primary_eligible") is not True
        or provenance.get("all_output_confirmatory_eligible") is not True
    ):
        raise ValueError("cache does not attest one all-true shared analysis mask")
    manifest_view = provenance.get("manifest_view")
    if manifest_view is not None:
        if not isinstance(manifest_view, Mapping):
            raise ValueError("cache development manifest-view provenance is invalid")
        _validated_manifest_view_record(manifest_view, provenance=provenance)
    if (primary_eligible is None) != (confirmatory_eligible is None):
        raise ValueError("both cache eligibility arrays must be supplied together")
    if primary_eligible is not None and confirmatory_eligible is not None:
        primary = np.asarray(primary_eligible)
        confirmatory = np.asarray(confirmatory_eligible)
        if (
            primary.shape != (len(identifiers),)
            or confirmatory.shape != primary.shape
            or primary.dtype != np.bool_
            or confirmatory.dtype != np.bool_
            or not np.array_equal(primary, confirmatory)
            or not bool(primary.all())
        ):
            raise ValueError("cache eligibility arrays are not one identical all-true mask")
    provenance["semantic_sha256"] = semantic
    return provenance


__all__ = [
    "DEVELOPMENT_MANIFEST_VIEW_SCOPE",
    "ELIGIBILITY_POLICY",
    "MANIFEST_VIEW_METADATA_KEY",
    "OVERLAP_EXCLUSION_REASON",
    "ManifestEligibilitySelection",
    "select_manifest_rows",
    "validate_analysis_eligibility_provenance",
]
