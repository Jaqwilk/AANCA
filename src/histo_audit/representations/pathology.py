"""Evidence-first pathology-encoder availability and priority audit.

This module never downloads or loads pathology weights.  It records supplied
evidence and selects only the first priority candidate for which every access,
licence, preprocessing, hardware, intended-use, and smoke-test gate is verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from histo_audit.representations.cache_provenance import canonical_sha256
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file, utc_now

AuthenticationStatus = Literal[
    "not_required", "credentials_available", "credentials_unavailable", "unknown"
]
HardwareFitStatus = Literal["verified_fit", "verified_not_fit", "not_assessed"]
CandidateStatus = Literal["selected", "eligible_not_selected", "blocked"]


@dataclass(frozen=True, slots=True)
class PathologyEncoderCandidate:
    """One priority candidate and the locally verified evidence available for it."""

    name: str
    priority: int
    original_source: str | None = None
    source_verified: bool = False
    licence: str | None = None
    licence_verified: bool = False
    weight_identifier: str | None = None
    weights_path: Path | None = None
    authentication_status: AuthenticationStatus = "unknown"
    preprocessing: str | None = None
    preprocessing_verified: bool = False
    hardware_fit_status: HardwareFitStatus = "not_assessed"
    intended_use: str | None = None
    intended_use_verified: bool = False
    embedding_smoke_passed: bool = False


@dataclass(frozen=True, slots=True)
class PathologyEncoderAuditRecord:
    """Audited availability state for one candidate."""

    name: str
    priority: int
    status: CandidateStatus
    blockers: tuple[str, ...]
    original_source: str | None
    source_verified: bool
    licence: str | None
    licence_verified: bool
    weight_identifier: str | None
    weights_path: str | None
    weights_sha256: str | None
    authentication_status: AuthenticationStatus
    preprocessing: str | None
    preprocessing_verified: bool
    hardware_fit_status: HardwareFitStatus
    intended_use: str | None
    intended_use_verified: bool
    embedding_smoke_passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PathologyEncoderAvailabilityAudit:
    """Priority selection result with an explicit overall blocker when absent."""

    status: Literal["available", "blocked"]
    selected_encoder: str | None
    blocker: str | None
    selection_rule: str
    audited_at_utc: str
    records: tuple[PathologyEncoderAuditRecord, ...]
    primary_cache_provenance: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return complete machine-readable evidence."""

        payload: dict[str, Any] = {
            "status": self.status,
            "selected_encoder": self.selected_encoder,
            "blocker": self.blocker,
            "selection_rule": self.selection_rule,
            "audited_at_utc": self.audited_at_utc,
            "records": [record.as_dict() for record in self.records],
        }
        if self.primary_cache_provenance is not None:
            payload["primary_cache_provenance"] = dict(self.primary_cache_provenance)
        return payload


def _require_sha256(value: str, *, field_name: str) -> str:
    normalised = str(value).casefold()
    if len(normalised) != 64 or any(
        character not in "0123456789abcdef" for character in normalised
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return normalised


def unavailable_optional_pathology_cache_provenance(
    *,
    sample_order_sha256: str,
    dataset_manifest_sha256: str,
    encoder_id: str = "availability_selected_pathology_encoder",
) -> dict[str, Any]:
    """Return the reproducible no-cache record for a blocked optional encoder.

    The recipe hash binds the predefined absence policy, not unavailable weights,
    preprocessing, implementation, or cache bytes.  Those artifact fields remain
    explicitly null and therefore cannot be mistaken for execution evidence.
    """

    if not encoder_id.strip():
        raise ValueError("pathology unavailable encoder_id must be explicit")
    sample_sha = _require_sha256(sample_order_sha256, field_name="sample_order_sha256")
    manifest_sha = _require_sha256(
        dataset_manifest_sha256,
        field_name="dataset_manifest_sha256",
    )
    recipe = {
        "schema_version": 1,
        "identifier": "pathology_optional_unavailable_cache_recipe_v1",
        "availability_policy": (
            "no cache is produced unless one candidate passes every frozen source, licence, "
            "weights, authentication, preprocessing, hardware, intended-use, and smoke gate"
        ),
        "cache_created": False,
        "unavailable_artifact_hashes_claimed": False,
    }
    return {
        "status": "unavailable_optional",
        "encoder_id": encoder_id,
        "encoder_implementation_sha256": None,
        "weights_sha256": None,
        "preprocessing_sha256": None,
        "sample_order_sha256": sample_sha,
        "dataset_manifest_sha256": manifest_sha,
        "cache_recipe_sha256": canonical_sha256(recipe),
        "cache_file_sha256": None,
    }


def default_pathology_encoder_candidates() -> tuple[PathologyEncoderCandidate, ...]:
    """Return a named priority rule with all external facts deliberately unverified.

    These records are not availability claims.  A project-specific caller must
    replace the ``False``/``None`` evidence fields after checking the original
    sources, terms, files, preprocessing, credentials, and local hardware smoke.
    """

    return (
        PathologyEncoderCandidate(name="UNI", priority=1),
        PathologyEncoderCandidate(name="CTransPath", priority=2),
        PathologyEncoderCandidate(name="other_verified_pathology_encoder", priority=3),
    )


def _candidate_blockers(candidate: PathologyEncoderCandidate) -> tuple[tuple[str, ...], str | None]:
    blockers: list[str] = []
    if not candidate.name.strip():
        blockers.append("encoder name is empty")
    if not candidate.original_source or not candidate.source_verified:
        blockers.append("original source is not verified")
    if not candidate.licence or not candidate.licence_verified:
        blockers.append("licence is not verified")
    if not candidate.weight_identifier:
        blockers.append("weight identifier is not recorded")
    weight_checksum: str | None = None
    if candidate.weights_path is None:
        blockers.append("weights path is not configured")
    else:
        path = Path(candidate.weights_path)
        if not path.is_file():
            blockers.append(f"weights file is unavailable: {path}")
        else:
            weight_checksum = sha256_file(path)
    if candidate.authentication_status not in ("not_required", "credentials_available"):
        blockers.append(
            f"authentication is not available (status={candidate.authentication_status})"
        )
    if not candidate.preprocessing or not candidate.preprocessing_verified:
        blockers.append("preprocessing is not reproducibly verified")
    if candidate.hardware_fit_status != "verified_fit":
        blockers.append(f"hardware fit is not verified (status={candidate.hardware_fit_status})")
    if not candidate.intended_use or not candidate.intended_use_verified:
        blockers.append("intended use is not verified")
    if not candidate.embedding_smoke_passed:
        blockers.append("embedding smoke test has not passed")
    return tuple(blockers), weight_checksum


def audit_pathology_encoder_availability(
    candidates: tuple[PathologyEncoderCandidate, ...]
    | list[PathologyEncoderCandidate]
    | None = None,
    *,
    output_path: str | Path | None = None,
    sample_order_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
) -> PathologyEncoderAvailabilityAudit:
    """Apply the frozen first-available priority rule and optionally save JSON."""

    supplied = (
        tuple(candidates) if candidates is not None else default_pathology_encoder_candidates()
    )
    if not supplied:
        raise ValueError("at least one pathology encoder candidate is required")
    priorities = [candidate.priority for candidate in supplied]
    if any(priority <= 0 for priority in priorities) or len(set(priorities)) != len(priorities):
        raise ValueError("candidate priorities must be unique positive integers")
    names = [candidate.name for candidate in supplied]
    if len(set(names)) != len(names):
        raise ValueError("pathology encoder candidate names must be unique")

    ordered = sorted(supplied, key=lambda candidate: candidate.priority)
    raw_records: list[tuple[PathologyEncoderCandidate, tuple[str, ...], str | None]] = []
    selected_name: str | None = None
    for candidate in ordered:
        blockers, checksum = _candidate_blockers(candidate)
        if not blockers and selected_name is None:
            selected_name = candidate.name
        raw_records.append((candidate, blockers, checksum))

    records: list[PathologyEncoderAuditRecord] = []
    for candidate, blockers, checksum in raw_records:
        if blockers:
            status: CandidateStatus = "blocked"
        elif candidate.name == selected_name:
            status = "selected"
        else:
            status = "eligible_not_selected"
        records.append(
            PathologyEncoderAuditRecord(
                name=candidate.name,
                priority=candidate.priority,
                status=status,
                blockers=blockers,
                original_source=candidate.original_source,
                source_verified=candidate.source_verified,
                licence=candidate.licence,
                licence_verified=candidate.licence_verified,
                weight_identifier=candidate.weight_identifier,
                weights_path=(
                    str(Path(candidate.weights_path).resolve())
                    if candidate.weights_path is not None
                    else None
                ),
                weights_sha256=checksum,
                authentication_status=candidate.authentication_status,
                preprocessing=candidate.preprocessing,
                preprocessing_verified=candidate.preprocessing_verified,
                hardware_fit_status=candidate.hardware_fit_status,
                intended_use=candidate.intended_use,
                intended_use_verified=candidate.intended_use_verified,
                embedding_smoke_passed=candidate.embedding_smoke_passed,
            )
        )

    selection_rule = (
        "Select the lowest numeric priority whose original source, licence, weights, "
        "authentication, preprocessing, hardware fit, intended use, and embedding smoke "
        "are all verified; do not select by primary-study performance."
    )
    if selected_name is None:
        details = "; ".join(f"{record.name}: {', '.join(record.blockers)}" for record in records)
        overall_blocker = (
            "no pathology encoder satisfies every predefined availability gate; "
            f"ImageNet work may continue independently. {details}"
        )
        status_value: Literal["available", "blocked"] = "blocked"
    else:
        overall_blocker = None
        status_value = "available"
    if (sample_order_sha256 is None) != (dataset_manifest_sha256 is None):
        raise ValueError(
            "sample_order_sha256 and dataset_manifest_sha256 must be supplied together"
        )
    primary_cache_provenance = None
    if sample_order_sha256 is not None and dataset_manifest_sha256 is not None:
        if status_value != "blocked":
            raise ValueError(
                "unavailable_optional primary cache provenance cannot describe an available "
                "pathology encoder"
            )
        primary_cache_provenance = unavailable_optional_pathology_cache_provenance(
            sample_order_sha256=sample_order_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    audit = PathologyEncoderAvailabilityAudit(
        status=status_value,
        selected_encoder=selected_name,
        blocker=overall_blocker,
        selection_rule=selection_rule,
        audited_at_utc=utc_now(),
        records=tuple(records),
        primary_cache_provenance=primary_cache_provenance,
    )
    if output_path is not None:
        atomic_write_json(output_path, audit.as_dict())
    return audit


__all__ = [
    "PathologyEncoderAuditRecord",
    "PathologyEncoderAvailabilityAudit",
    "PathologyEncoderCandidate",
    "audit_pathology_encoder_availability",
    "default_pathology_encoder_candidates",
    "unavailable_optional_pathology_cache_provenance",
]
