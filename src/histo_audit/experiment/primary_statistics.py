"""Filesystem-backed primary-study statistics and strict artifact verification.

This module is deliberately separate from the cell executor.  It consumes only a
completed, reconciled primary matrix and immutable schema-v2 execution controls.  It
never invents comparisons from the observed matrix: every inferential comparison must
be present in one of the frozen comparison families carried by the controls.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.corruption.controlled import canonical_sha256
from histo_audit.experiment.primary_completion import read_primary_filesystem_evidence
from histo_audit.experiment.primary_core import (
    PrimaryCrossCellComparison,
    PrimaryExecutionControls,
    PrimaryMethodVsRandomComparison,
    PrimaryWithinCellComparison,
)
from histo_audit.experiment.study_contracts import PrimaryCell
from histo_audit.statistics.review import (
    average_precision,
    binary_auroc,
    draw_group_bootstrap_indices,
    evaluate_review_budget,
    holm_adjust,
    random_review_baseline,
    rank_indices,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file

_STATISTICS_FILE = "primary_statistics.json"
_BOOTSTRAP_FILE = "primary_bootstrap_evidence.npz"
_SUBGROUP_FILE = "primary_subgroups.csv"
_MANIFEST_FILE = "primary_statistics_manifest.json"
_OUTPUT_FILES = (_STATISTICS_FILE, _BOOTSTRAP_FILE, _SUBGROUP_FILE)
_METRICS = {
    "average_precision",
    "auroc",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
}
_DIRECTION = "method_a_minus_method_b"
_BASE_RANKING_COLUMNS = (
    "rank",
    "sample_id",
    "group_id",
    "pre_corruption_label",
    "observed_label",
    "is_injected_corruption",
)
_OOF_KEYS = {
    "sample_ids",
    "group_ids",
    "pre_corruption_label",
    "observed_label",
    "is_injected_corruption",
    "probabilities",
    "predicted_class",
    "fold_id",
    "coverage_count",
    "fold_assignment_labels",
    "fold_assignment_label_source",
    "fold_assignment_labels_sha256",
}
_SHARED_CORRUPTION_FIELDS = (
    "sample_id",
    "group_id",
    "pre_corruption_label",
    "observed_label",
    "is_injected_corruption",
    "corruption_type",
    "original_class",
    "replacement_class",
    "corruption_rate",
    "corruption_seed",
    "corruption_representation",
)
_SUBGROUP_COLUMNS = (
    "cell_id",
    "scenario_id",
    "method",
    "dimension",
    "value",
    "sample_count",
    "injected_corruption_count",
    "average_precision_status",
    "average_precision",
    "suppression_reason",
)
_STATISTICS_VERIFICATION_ATTESTATION = object()
_INHERITED_STATISTICS_VERIFICATION_ATTESTATION = object()
_AUTHORIZED_PRIOR_NUMERIC_PROOF_ATTESTATION = object()
_AUTHORIZED_ORPHAN_NUMERIC_PROOF_ATTESTATION = object()
_FINALIZATION_SUCCESSOR_AUTHORIZATION_KIND = "finalization_successor"
_ORPHAN_RECOVERY_AUTHORIZATION_KIND = "orphan_recovery"
_RANDOM_AP_DRAW_CHUNK_SIZE = 4
INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE = "inherited_prior_numeric_verification_v1"
INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION = (
    "trusted_local_process_no_dependency_injection_import_hook_hotpatch_or_concurrent_writer"
)
INHERITED_PRIOR_NUMERIC_LIMITATION = (
    "control_flow_and_content_addressed_inheritance_not_fresh_semantic_recomputation"
)


@dataclass(frozen=True, slots=True)
class PrimaryStatisticsArtifacts:
    """The four atomically persisted primary statistics artifacts."""

    output_directory: Path
    statistics_path: Path
    bootstrap_evidence_path: Path
    subgroups_path: Path
    manifest_path: Path
    comparison_count: int
    reportable_comparison_count: int
    manifest_sha256: str
    verification: PrimaryStatisticsVerification


@dataclass(frozen=True, slots=True)
class PrimaryStatisticsVerification:
    """Successful strict reread of primary statistics and all of their inputs."""

    status: str
    output_directory: Path
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    subgroups_sha256: str
    manifest_sha256: str
    source_readback_root_sha256: str
    comparison_count: int
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return self.status == "passed" and self._attestation is _STATISTICS_VERIFICATION_ATTESTATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_directory": str(self.output_directory),
            "statistics_sha256": self.statistics_sha256,
            "bootstrap_evidence_sha256": self.bootstrap_evidence_sha256,
            "subgroups_sha256": self.subgroups_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_readback_root_sha256": self.source_readback_root_sha256,
            "comparison_count": self.comparison_count,
        }


@dataclass(frozen=True, slots=True)
class AuthorizedPriorNumericVerificationProof:
    """Unforgeable-in-process capability issued from one canonical sealed amendment."""

    amendment_directory: Path
    authorization_sha256: str
    predecessor_run_id: str
    predecessor_artifact_root_sha256: str
    predecessor_artifact_manifest_sha256: str
    predecessor_source_tree_root_sha256: str
    source_readback_root_sha256: str
    verification_mode: str
    prior_numeric_verification_proof_sha256: str
    trust_assumption: str
    limitation: str
    statistics_quartet: tuple[tuple[str, int, str], ...]
    comparison_count: int
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return (
            self._attestation is _AUTHORIZED_PRIOR_NUMERIC_PROOF_ATTESTATION
            and self.verification_mode == INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
            and len(self.authorization_sha256) == 64
            and len(self.prior_numeric_verification_proof_sha256) == 64
            and len(self.source_readback_root_sha256) == 64
            and self.trust_assumption == INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
            and self.limitation == INHERITED_PRIOR_NUMERIC_LIMITATION
            and len(self.statistics_quartet) == 4
            and self.comparison_count >= 0
        )

    @property
    def authorization_kind(self) -> str:
        return _FINALIZATION_SUCCESSOR_AUTHORIZATION_KIND


def _issue_authorized_prior_numeric_verification_proof(
    *,
    amendment_directory: Path,
    authorization_sha256: str,
    predecessor_run_id: str,
    predecessor_artifact_root_sha256: str,
    predecessor_artifact_manifest_sha256: str,
    predecessor_source_tree_root_sha256: str,
    source_readback_root_sha256: str,
    prior_numeric_verification_proof_sha256: str,
    trust_assumption: str,
    limitation: str,
    statistics_quartet: Sequence[Mapping[str, Any]],
    comparison_count: int,
) -> AuthorizedPriorNumericVerificationProof:
    """Mint the private B-fast capability after canonical amendment verification."""

    def require_sha(value: str, *, role: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{role} is not a lowercase SHA-256")
        return value

    expected_names = {
        _STATISTICS_FILE,
        _BOOTSTRAP_FILE,
        _SUBGROUP_FILE,
        _MANIFEST_FILE,
    }
    records: list[tuple[str, int, str]] = []
    for raw in statistics_quartet:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise ValueError("authorized statistics quartet has an invalid record")
        name = raw.get("path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or name not in expected_names
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
        ):
            raise ValueError("authorized statistics quartet has invalid typed fields")
        records.append((name, size, require_sha(digest, role=f"{name} SHA-256")))
    if len(records) != 4 or {name for name, _, _ in records} != expected_names:
        raise ValueError("authorized statistics quartet is incomplete or duplicated")
    if type(comparison_count) is not int or comparison_count < 0:
        raise ValueError("authorized comparison count must be a non-negative exact integer")
    proof = AuthorizedPriorNumericVerificationProof(
        amendment_directory=amendment_directory.resolve(),
        authorization_sha256=require_sha(
            authorization_sha256, role="finalization authorization SHA-256"
        ),
        predecessor_run_id=predecessor_run_id,
        predecessor_artifact_root_sha256=require_sha(
            predecessor_artifact_root_sha256, role="predecessor artifact root"
        ),
        predecessor_artifact_manifest_sha256=require_sha(
            predecessor_artifact_manifest_sha256,
            role="predecessor artifact manifest",
        ),
        predecessor_source_tree_root_sha256=require_sha(
            predecessor_source_tree_root_sha256,
            role="predecessor source-tree root",
        ),
        source_readback_root_sha256=require_sha(
            source_readback_root_sha256,
            role="predecessor statistics source readback root",
        ),
        verification_mode=INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        prior_numeric_verification_proof_sha256=require_sha(
            prior_numeric_verification_proof_sha256,
            role="prior numeric verification proof",
        ),
        trust_assumption=trust_assumption,
        limitation=limitation,
        statistics_quartet=tuple(sorted(records)),
        comparison_count=comparison_count,
    )
    if (
        proof.trust_assumption != INHERITED_PRIOR_NUMERIC_TRUST_ASSUMPTION
        or proof.limitation != INHERITED_PRIOR_NUMERIC_LIMITATION
    ):
        raise ValueError("authorized prior-numeric proof has an unsupported trust contract")
    object.__setattr__(
        proof,
        "_attestation",
        _AUTHORIZED_PRIOR_NUMERIC_PROOF_ATTESTATION,
    )
    return proof


@dataclass(frozen=True, slots=True)
class AuthorizedOrphanNumericVerificationProof:
    """Capability for one authorized unsealed-orphan numeric readback.

    Unlike :class:`AuthorizedPriorNumericVerificationProof`, this capability makes
    no claim that the source was a sealed predecessor run.  Its bindings are the
    read-only orphan inspection and the technical recovery authorization.
    """

    amendment_directory: Path
    authorization_sha256: str
    source_run_id: str
    source_snapshot_root_sha256: str
    source_status_sha256: str
    source_tree_root_sha256: str
    source_tree_manifest_sha256: str
    source_readback_root_sha256: str
    verification_mode: str
    prior_numeric_verification_proof_sha256: str
    trust_assumption: str
    limitation: str
    statistics_quartet: tuple[tuple[str, int, str], ...]
    comparison_count: int
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        sha_values = (
            self.authorization_sha256,
            self.source_snapshot_root_sha256,
            self.source_status_sha256,
            self.source_tree_root_sha256,
            self.source_tree_manifest_sha256,
            self.source_readback_root_sha256,
            self.prior_numeric_verification_proof_sha256,
        )
        return (
            self._attestation is _AUTHORIZED_ORPHAN_NUMERIC_PROOF_ATTESTATION
            and self.verification_mode == INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
            and bool(self.source_run_id)
            and Path(self.source_run_id).name == self.source_run_id
            and all(
                len(value) == 64 and all(character in "0123456789abcdef" for character in value)
                for value in sha_values
            )
            and bool(self.trust_assumption.strip())
            and bool(self.limitation.strip())
            and len(self.statistics_quartet) == 4
            and self.comparison_count >= 0
        )

    @property
    def authorization_kind(self) -> str:
        return _ORPHAN_RECOVERY_AUTHORIZATION_KIND


def _issue_authorized_orphan_numeric_verification_proof(
    *,
    amendment_directory: Path,
    authorization_sha256: str,
    source_run_id: str,
    source_snapshot_root_sha256: str,
    source_status_sha256: str,
    source_tree_root_sha256: str,
    source_tree_manifest_sha256: str,
    source_readback_root_sha256: str,
    prior_numeric_verification_proof_sha256: str,
    trust_assumption: str,
    limitation: str,
    statistics_quartet: Sequence[Mapping[str, Any]],
    comparison_count: int,
) -> AuthorizedOrphanNumericVerificationProof:
    """Mint a recovery-only capability after typed orphan qualification."""

    def require_sha(value: str, *, role: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{role} is not a lowercase SHA-256")
        return value

    if not source_run_id or Path(source_run_id).name != source_run_id:
        raise ValueError("authorized orphan source run id is unsafe")
    if not trust_assumption.strip() or not limitation.strip():
        raise ValueError("authorized orphan proof requires explicit trust and limitation")

    expected_names = {
        _STATISTICS_FILE,
        _BOOTSTRAP_FILE,
        _SUBGROUP_FILE,
        _MANIFEST_FILE,
    }
    records: list[tuple[str, int, str]] = []
    for raw in statistics_quartet:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise ValueError("authorized statistics quartet has an invalid record")
        name = raw.get("path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or name not in expected_names
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
        ):
            raise ValueError("authorized statistics quartet has invalid typed fields")
        records.append((name, size, require_sha(digest, role=f"{name} SHA-256")))
    if len(records) != 4 or {name for name, _, _ in records} != expected_names:
        raise ValueError("authorized statistics quartet is incomplete or duplicated")
    if type(comparison_count) is not int or comparison_count < 0:
        raise ValueError("authorized comparison count must be a non-negative exact integer")

    proof = AuthorizedOrphanNumericVerificationProof(
        amendment_directory=amendment_directory.resolve(),
        authorization_sha256=require_sha(
            authorization_sha256,
            role="orphan recovery authorization SHA-256",
        ),
        source_run_id=source_run_id,
        source_snapshot_root_sha256=require_sha(
            source_snapshot_root_sha256,
            role="orphan source snapshot root",
        ),
        source_status_sha256=require_sha(
            source_status_sha256,
            role="orphan source status",
        ),
        source_tree_root_sha256=require_sha(
            source_tree_root_sha256,
            role="orphan source-tree root",
        ),
        source_tree_manifest_sha256=require_sha(
            source_tree_manifest_sha256,
            role="orphan source-tree manifest",
        ),
        source_readback_root_sha256=require_sha(
            source_readback_root_sha256,
            role="orphan statistics source readback root",
        ),
        verification_mode=INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        prior_numeric_verification_proof_sha256=require_sha(
            prior_numeric_verification_proof_sha256,
            role="orphan prior numeric verification proof",
        ),
        trust_assumption=trust_assumption,
        limitation=limitation,
        statistics_quartet=tuple(sorted(records)),
        comparison_count=comparison_count,
    )
    object.__setattr__(
        proof,
        "_attestation",
        _AUTHORIZED_ORPHAN_NUMERIC_PROOF_ATTESTATION,
    )
    return proof


@dataclass(frozen=True, slots=True)
class InheritedPrimaryStatisticsVerificationProvenance:
    """Typed provenance for a narrowly authorized non-recomputation readback."""

    schema_version: int
    verification_mode: str
    prior_numeric_verification_proof_sha256: str
    finalization_successor_authorization_sha256: str
    predecessor_run_id: str
    predecessor_artifact_root_sha256: str
    verification_scope: str
    trust_assumption: str
    limitation: str
    source_readback_root_sha256: str
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    subgroups_sha256: str
    manifest_sha256: str
    comparison_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrphanRecoveryNumericVerificationProvenance:
    """Recovery provenance that never represents an orphan as a sealed predecessor."""

    schema_version: int
    verification_mode: str
    prior_numeric_verification_proof_sha256: str
    authorization_sha256: str
    source_run_id: str
    source_snapshot_root_sha256: str
    source_status_sha256: str
    source_tree_root_sha256: str
    source_tree_manifest_sha256: str
    verification_scope: str
    trust_assumption: str
    limitation: str
    source_readback_root_sha256: str
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    subgroups_sha256: str
    manifest_sha256: str
    comparison_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InheritedPrimaryStatisticsVerification:
    """Structural quartet verification backed by a sealed prior-numeric proof."""

    status: str
    output_directory: Path
    statistics_sha256: str
    bootstrap_evidence_sha256: str
    subgroups_sha256: str
    manifest_sha256: str
    source_readback_root_sha256: str
    comparison_count: int
    verification_mode: str
    prior_numeric_verification_proof_sha256: str
    _attestation: object | None = field(default=None, init=False, repr=False, compare=False)
    _authorization_kind: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def valid(self) -> bool:
        return (
            self.status == "passed"
            and self.verification_mode == INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE
            and self._attestation is _INHERITED_STATISTICS_VERIFICATION_ATTESTATION
            and self._authorization_kind
            in {
                _FINALIZATION_SUCCESSOR_AUTHORIZATION_KIND,
                _ORPHAN_RECOVERY_AUTHORIZATION_KIND,
            }
        )

    @property
    def authorization_kind(self) -> str | None:
        return self._authorization_kind

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_directory": str(self.output_directory),
            "statistics_sha256": self.statistics_sha256,
            "bootstrap_evidence_sha256": self.bootstrap_evidence_sha256,
            "subgroups_sha256": self.subgroups_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_readback_root_sha256": self.source_readback_root_sha256,
            "comparison_count": self.comparison_count,
            "verification_mode": self.verification_mode,
            "prior_numeric_verification_proof_sha256": (
                self.prior_numeric_verification_proof_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class _CellData:
    cell: PrimaryCell
    sample_ids: NDArray[np.str_]
    group_ids: NDArray[np.str_]
    pre_corruption_label: NDArray[np.int64]
    observed_label: NDArray[np.int64]
    injected: NDArray[np.bool_]
    risks: Mapping[str, NDArray[np.float64]]
    shared_corruption_rows: tuple[Mapping[str, Any], ...]
    shared_corruption_sha256: str
    circularity_risk: bool
    primary_confirmatory_eligible: bool


@dataclass(frozen=True, slots=True)
class _ComputedStatistics:
    payload: dict[str, Any]
    bootstrap_arrays: dict[str, NDArray[np.generic]]
    subgroup_rows: tuple[dict[str, Any], ...]
    source_readback_root_sha256: str
    source_cell_manifest_sha256: Mapping[str, str]
    primary_input_bindings_sha256: str
    crop_cache_sha256: str


def _read_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_SUBGROUP_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _budget_key(value: float) -> str:
    return format(float(value), ".12g")


def _metric_field(value: float | None, *, reason: str) -> dict[str, Any]:
    if value is None:
        return {"status": "not_applicable", "value": None, "reason": reason}
    return {"status": "reported", "value": float(value), "reason": None}


def _bool_csv(value: str, role: str) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"{role} must be true or false")


def _load_tissue_mapping(
    run_directory: Path,
    sample_ids: NDArray[np.str_],
) -> tuple[NDArray[np.str_], str, str]:
    """Load tissue labels from the hash-bound PanNuke crop cache used by the run."""

    bindings_path = run_directory / "primary_input_bindings.json"
    bindings = _read_json(bindings_path, "primary input bindings")
    cache_paths = bindings.get("cache_paths")
    expected_hashes = bindings.get("expected_hashes")
    verified_hashes = bindings.get("verified_hashes")
    if not isinstance(cache_paths, Mapping):
        raise ValueError("primary input bindings lack cache paths or hash records")
    if not isinstance(expected_hashes, Mapping) or not isinstance(verified_hashes, Mapping):
        raise ValueError("primary input bindings lack cache paths or hash records")
    raw_crop_path = cache_paths.get("crop_cache_path")
    if not isinstance(raw_crop_path, str) or not raw_crop_path.strip():
        raise ValueError("primary input bindings lack crop_cache_path")
    crop_path = Path(raw_crop_path).expanduser().resolve()
    if not crop_path.is_file():
        raise FileNotFoundError(f"hash-bound PanNuke crop cache is unavailable: {crop_path}")
    actual_sha = sha256_file(crop_path)
    expected_sha = expected_hashes.get("crop_cache_sha256")
    verified_sha = verified_hashes.get("crop_cache_sha256")
    if expected_sha != actual_sha or verified_sha != actual_sha:
        raise ValueError("crop cache hash differs from primary input bindings")
    try:
        with np.load(crop_path, allow_pickle=False) as archive:
            if not {"sample_ids", "tissue_types"}.issubset(archive.files):
                raise ValueError("crop cache lacks sample_ids or tissue_types")
            source_ids = np.asarray(archive["sample_ids"], dtype=np.str_)
            source_tissues = np.asarray(archive["tissue_types"], dtype=np.str_)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "crop cache lacks" in str(exc):
            raise
        raise ValueError("PanNuke crop cache is invalid or pickle-dependent") from exc
    if (
        source_ids.ndim != 1
        or source_tissues.shape != source_ids.shape
        or len(set(source_ids.tolist())) != len(source_ids)
        or np.any(source_tissues == "")
    ):
        raise ValueError("crop sample/tissue vectors are invalid")
    frozen_sample_order_sha = bindings.get("sample_order_sha256")
    if canonical_sha256(source_ids.tolist()) != frozen_sample_order_sha:
        raise ValueError("crop sample order differs from primary input bindings")
    tissue_by_id = dict(zip(source_ids.tolist(), source_tissues.tolist(), strict=True))
    missing = sorted(set(sample_ids.tolist()).difference(tissue_by_id))
    if missing:
        raise ValueError(f"crop cache lacks {len(missing)} audit sample IDs")
    tissues = np.asarray([tissue_by_id[value] for value in sample_ids.tolist()], dtype=np.str_)
    return tissues, actual_sha, sha256_file(bindings_path)


def _load_npz(path: Path, role: str) -> dict[str, NDArray[np.generic]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"{role} is invalid or pickle-dependent: {path}") from exc


def _load_cell(
    run_directory: Path,
    cell: PrimaryCell,
    controls: PrimaryExecutionControls,
) -> _CellData:
    directory = run_directory / "cells" / cell.cell_id
    oof = _load_npz(directory / "oof_predictions.npz", f"{cell.cell_id} OOF evidence")
    if set(oof) != _OOF_KEYS:
        raise ValueError(
            f"{cell.cell_id} OOF array set mismatch: "
            f"missing={sorted(_OOF_KEYS.difference(oof))}, "
            f"extra={sorted(set(oof).difference(_OOF_KEYS))}"
        )
    sample_ids = np.asarray(oof["sample_ids"], dtype=np.str_)
    group_ids = np.asarray(oof["group_ids"], dtype=np.str_)
    pre = np.asarray(oof["pre_corruption_label"], dtype=np.int64)
    observed = np.asarray(oof["observed_label"], dtype=np.int64)
    injected = np.asarray(oof["is_injected_corruption"], dtype=np.bool_)
    n = len(sample_ids)
    if (
        sample_ids.ndim != 1
        or n == 0
        or len(set(sample_ids.tolist())) != n
        or group_ids.shape != (n,)
        or np.any(group_ids == "")
        or pre.shape != (n,)
        or observed.shape != (n,)
        or injected.shape != (n,)
    ):
        raise ValueError(f"{cell.cell_id} OOF identity/corruption vectors are invalid")
    probabilities = np.asarray(oof["probabilities"], dtype=np.float64)
    if (
        probabilities.shape != (n, len(controls.class_order))
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
        or not np.all(np.asarray(oof["coverage_count"], dtype=np.int64) == 1)
    ):
        raise ValueError(f"{cell.cell_id} OOF probabilities or coverage are invalid")
    risks_raw = _load_npz(directory / "risk_scores.npz", f"{cell.cell_id} risk evidence")
    unknown = set(risks_raw).difference(controls.audit_methods)
    missing = set(controls.audit_methods).difference(risks_raw)
    cleanlab = _read_json(directory / "cleanlab_evidence.json", f"{cell.cell_id} Cleanlab")
    allowed_missing = {"cleanlab"} if cleanlab.get("available") is False else set()
    if unknown or not missing.issubset(allowed_missing):
        raise ValueError(
            f"{cell.cell_id} risk method set differs from frozen controls: "
            f"missing={sorted(missing)}, extra={sorted(unknown)}"
        )
    risks: dict[str, NDArray[np.float64]] = {}
    for method, raw_values in risks_raw.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"{cell.cell_id}/{method} risk vector is invalid")
        risks[method] = values

    ranking_path = directory / "ranking.csv"
    with ranking_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        ranking_rows = [dict(row) for row in reader]
    expected_columns = (*_BASE_RANKING_COLUMNS, *risks.keys())
    if fieldnames != expected_columns or len(ranking_rows) != n:
        raise ValueError(f"{cell.cell_id} ranking CSV schema/row count is invalid")
    expected_order = rank_indices(
        risks[controls.primary_ranking_method], tie_break_ids=sample_ids.tolist()
    )
    for rank, (row, source_index) in enumerate(
        zip(ranking_rows, expected_order.tolist(), strict=True), start=1
    ):
        if (
            int(row["rank"]) != rank
            or row["sample_id"] != sample_ids[source_index]
            or row["group_id"] != group_ids[source_index]
            or int(row["pre_corruption_label"]) != int(pre[source_index])
            or int(row["observed_label"]) != int(observed[source_index])
            or _bool_csv(row["is_injected_corruption"], f"{cell.cell_id} ranking flag")
            is not bool(injected[source_index])
        ):
            raise ValueError(f"{cell.cell_id} ranking identity/order differs from NPZ evidence")
        for method, values in risks.items():
            if float(row[method]) != float(values[source_index]):
                raise ValueError(f"{cell.cell_id}/{method} ranking score differs from NPZ")

    corruption = _read_json(
        directory / "corruption_manifest.json", f"{cell.cell_id} corruption manifest"
    )
    raw_rows = corruption.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != n:
        raise ValueError(f"{cell.cell_id} corruption rows do not align with OOF evidence")
    shared_rows: list[Mapping[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{cell.cell_id} corruption row {index} is not an object")
        shared = {field: raw_row.get(field) for field in _SHARED_CORRUPTION_FIELDS}
        if (
            shared["sample_id"] != str(sample_ids[index])
            or shared["group_id"] != str(group_ids[index])
            or shared["pre_corruption_label"] != int(pre[index])
            or shared["observed_label"] != int(observed[index])
            or shared["is_injected_corruption"] is not bool(injected[index])
        ):
            raise ValueError(f"{cell.cell_id} corruption row {index} differs from OOF evidence")
        shared_rows.append(shared)
    shared_sha = str(corruption.get("shared_scenario_corruption_hash", ""))
    if len(shared_sha) != 64:
        raise ValueError(f"{cell.cell_id} lacks a shared scenario corruption SHA")
    independence = _read_json(
        directory / "independence_evidence.json", f"{cell.cell_id} independence evidence"
    )
    circularity = independence.get("circularity_risk")
    eligible = independence.get("primary_confirmatory_eligible")
    if (
        not isinstance(circularity, bool)
        or not isinstance(eligible, bool)
        or circularity is eligible
    ):
        raise ValueError(f"{cell.cell_id} has inconsistent circularity eligibility")
    return _CellData(
        cell=cell,
        sample_ids=sample_ids,
        group_ids=group_ids,
        pre_corruption_label=pre,
        observed_label=observed,
        injected=injected,
        risks=risks,
        shared_corruption_rows=tuple(shared_rows),
        shared_corruption_sha256=shared_sha,
        circularity_risk=circularity,
        primary_confirmatory_eligible=eligible,
    )


def _selector_matches(selector: object, cell: PrimaryCell) -> bool:
    cell_id = getattr(selector, "cell_id", None)
    representation = getattr(selector, "representation_id", None)
    classifier = getattr(selector, "classifier_id", None)
    mechanism = getattr(selector, "mechanism", None)
    rate = getattr(selector, "rate", None)
    seed = getattr(selector, "seed", getattr(selector, "corruption_seed", None))
    if cell_id is not None:
        return (
            cell.cell_id == cell_id
            and representation is None
            and classifier is None
            and mechanism is None
            and rate is None
            and seed is None
        )
    if (
        representation is None
        or classifier is None
        or mechanism is None
        or rate is None
        or seed is None
    ):
        return False
    return (
        cell.representation_id == representation
        and cell.classifier_id == classifier
        and cell.mechanism == mechanism
        and cell.rate == float(rate)
        and cell.corruption_seed == int(seed)
    )


def _resolve_selector(
    controls: PrimaryExecutionControls, selector: object, role: str
) -> PrimaryCell:
    matches = [cell for cell in controls.plan.cells if _selector_matches(selector, cell)]
    if len(matches) != 1:
        raise ValueError(f"{role} must resolve exactly one frozen primary cell; got {len(matches)}")
    return matches[0]


def _metric_value(
    injected: NDArray[np.bool_],
    scores: NDArray[np.float64],
    metric: str,
    *,
    review_budget: float | None,
) -> float | None:
    if metric == "average_precision":
        return average_precision(injected, scores)
    if metric == "auroc":
        return binary_auroc(injected, scores)
    if review_budget is None:
        raise ValueError(f"{metric} requires a frozen review_budget")
    review = evaluate_review_budget(injected, scores, budget=review_budget)
    if metric == "precision_at_budget":
        return review.precision
    if metric == "recall_at_budget":
        return review.recall
    if metric == "lift_at_budget":
        return review.lift_over_random
    raise ValueError(f"unsupported frozen comparison metric: {metric!r}")


def _random_rankings(n_samples: int, repeats: int, seed: int) -> NDArray[np.float64]:
    scores = np.empty((repeats, n_samples), dtype=np.float64)
    for repeat in range(repeats):
        permutation = np.random.default_rng(seed + repeat).permutation(n_samples)
        scores[repeat, permutation] = np.arange(n_samples, 0, -1, dtype=np.float64)
    return scores


def _random_permutation_orders(
    random_scores: NDArray[np.float64],
) -> NDArray[np.int64] | None:
    """Return descending orders only for the exact frozen permutation-score form.

    The accelerated AP path relies on every original sample having one unique score.
    Requiring the exact ``n, ..., 1`` score set makes that precondition fail closed;
    callers retain the legacy expanded-draw implementation for every other input.
    """

    scores = np.asarray(random_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return None
    if not np.isfinite(scores).all():
        return None
    expected = np.arange(scores.shape[1], 0, -1, dtype=np.float64)
    orders = np.empty(scores.shape, dtype=np.int64)
    for repeat, row in enumerate(scores):
        order = np.argsort(-row, kind="stable").astype(np.int64, copy=False)
        if not np.array_equal(row[order], expected):
            return None
        orders[repeat] = order
    return orders


def _method_vs_random_average_precision_by_draw(
    injected: NDArray[np.bool_],
    draws: Sequence[NDArray[np.int64]],
    random_scores: NDArray[np.float64],
    *,
    chunk_size: int = _RANDOM_AP_DRAW_CHUNK_SIZE,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]] | None:
    """Compute frozen random-ranking APs without repeatedly sorting expanded draws.

    For a unique random ranking, whole-group resampling only changes each original
    sample's integer multiplicity.  Sorting the expanded draw is therefore exactly
    equivalent to scanning the one frozen descending order with those multiplicities.
    Integer cumulative counts, recall differencing, precision division, and the final
    sequential ``cumsum`` retain the legacy operation order.  ``None`` is a hard
    request to use the legacy implementation whenever the narrow equivalence proof's
    preconditions do not hold.
    """

    labels = np.asarray(injected, dtype=np.bool_)
    if labels.ndim != 1 or labels.size == 0 or not isinstance(chunk_size, int) or chunk_size <= 0:
        return None
    orders = _random_permutation_orders(random_scores)
    if orders is None or orders.shape[1] != labels.size:
        return None
    normalised_draws: list[NDArray[np.int64]] = []
    for raw_draw in draws:
        draw = np.asarray(raw_draw, dtype=np.int64)
        if draw.ndim != 1 or np.any(draw < 0) or np.any(draw >= labels.size):
            return None
        normalised_draws.append(draw)
    if not normalised_draws:
        return None

    valid = np.zeros(len(normalised_draws), dtype=np.bool_)
    means = np.full(len(normalised_draws), np.nan, dtype=np.float64)
    positive_columns = np.flatnonzero(labels)
    for start in range(0, len(normalised_draws), chunk_size):
        stop = min(start + chunk_size, len(normalised_draws))
        multiplicities = np.empty((stop - start, labels.size), dtype=np.int64)
        for local_index, draw in enumerate(normalised_draws[start:stop]):
            multiplicities[local_index] = np.bincount(draw, minlength=labels.size)
        positive_counts = multiplicities[:, positive_columns].sum(axis=1, dtype=np.int64)
        local_valid = positive_counts > 0
        valid[start:stop] = local_valid
        if not np.any(local_valid):
            continue

        valid_multiplicities = multiplicities[local_valid]
        valid_positive_counts = positive_counts[local_valid]
        per_repeat = np.empty((orders.shape[0], len(valid_positive_counts)), dtype=np.float64)
        for repeat, order in enumerate(orders):
            ordered_multiplicities = valid_multiplicities[:, order]
            ordered_positive = ordered_multiplicities * labels[order]
            cumulative_true = np.cumsum(ordered_positive, axis=1, dtype=np.int64)
            cumulative_total = np.cumsum(ordered_multiplicities, axis=1, dtype=np.int64)
            recall = cumulative_true / valid_positive_counts[:, np.newaxis]
            recall_increments = np.empty_like(recall)
            recall_increments[:, 0] = recall[:, 0]
            np.subtract(
                recall[:, 1:],
                recall[:, :-1],
                out=recall_increments[:, 1:],
            )
            precision = np.zeros_like(recall)
            np.divide(
                cumulative_true,
                cumulative_total,
                out=precision,
                where=cumulative_total != 0,
            )
            recall_increments *= precision
            per_repeat[repeat] = np.cumsum(
                recall_increments,
                axis=1,
                dtype=np.float64,
            )[:, -1]

        valid_positions = np.flatnonzero(local_valid)
        for column, local_position in enumerate(valid_positions):
            # The legacy path calls np.mean on one contiguous list per draw.  Preserve
            # that reduction shape/order instead of averaging across rankings algebraically.
            values = np.ascontiguousarray(per_repeat[:, column])
            means[start + int(local_position)] = float(np.mean(values))
    return valid, means


def _flatten_draws(
    draws: Sequence[NDArray[np.integer]],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    offsets = np.zeros(len(draws) + 1, dtype=np.int64)
    for index, draw in enumerate(draws):
        offsets[index + 1] = offsets[index] + len(draw)
    flattened = (
        np.concatenate(tuple(np.asarray(draw, dtype=np.int64) for draw in draws))
        if draws
        else np.empty(0, dtype=np.int64)
    )
    return flattened, offsets


def _validate_cross_cell_pair(
    *,
    comparison_id: str,
    cell_a: PrimaryCell,
    cell_b: PrimaryCell,
    data_a: _CellData | None,
    data_b: _CellData | None,
) -> None:
    """Validate one controlled, sample-aligned cross-cell contrast.

    Within-scenario comparisons retain the original strict requirement that the
    injected-event and observed-label arrays are identical.  Controlled cross-scenario
    comparisons (including the H2/H3 mechanism contrasts and rate sensitivity) may
    span scenarios only when they isolate one corruption factor and keep a common
    sample/group/reference-label order.
    """

    same_scenario = cell_a.scenario_id == cell_b.scenario_id
    if not same_scenario:
        if (
            cell_a.corruption_seed != cell_b.corruption_seed
            or cell_a.representation_id != cell_b.representation_id
            or cell_a.classifier_id != cell_b.classifier_id
        ):
            raise ValueError(
                f"{comparison_id} controlled cross-scenario comparison must hold seed, "
                "representation, and classifier fixed"
            )
        changed_corruption_factors = sum(
            (
                cell_a.mechanism != cell_b.mechanism,
                cell_a.rate != cell_b.rate,
            )
        )
        if changed_corruption_factors != 1:
            raise ValueError(
                f"{comparison_id} controlled cross-scenario comparison must vary exactly one "
                "of mechanism or rate"
            )
    if data_a is None or data_b is None:
        return
    if (
        not np.array_equal(data_a.sample_ids, data_b.sample_ids)
        or not np.array_equal(data_a.group_ids, data_b.group_ids)
        or not np.array_equal(data_a.pre_corruption_label, data_b.pre_corruption_label)
    ):
        raise ValueError(
            f"{comparison_id} cross-cell data lack exact sample/group/reference-label alignment"
        )
    if same_scenario and (
        not np.array_equal(data_a.injected, data_b.injected)
        or not np.array_equal(data_a.observed_label, data_b.observed_label)
    ):
        raise ValueError(
            f"{comparison_id} within-scenario cross-cell corruption data are not identical"
        )


def _comparison_result(
    *,
    comparison_id: str,
    kind: str,
    cell_a: _CellData | None,
    cell_b: _CellData | None,
    method_a: str,
    method_b: str,
    metric: str,
    direction: str,
    holm_family: str,
    review_budget: float | None,
    draws: Sequence[NDArray[np.int64]],
    random_scores: NDArray[np.float64],
) -> tuple[dict[str, Any], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    base = {
        "comparison_id": comparison_id,
        "kind": kind,
        "cell_id_a": None if cell_a is None else cell_a.cell.cell_id,
        "cell_id_b": None if cell_b is None else cell_b.cell.cell_id,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "direction": direction,
        "holm_family": holm_family,
        "review_budget": review_budget,
    }
    if cell_a is None or (kind == "cross_cell" and cell_b is None):
        return (
            {
                **base,
                "status": "not_available_frozen_optional_cell",
                "reason": "a frozen selector resolved to an unavailable optional cell",
                "point_metric_a": None,
                "point_metric_b": None,
                "point_difference": None,
                "bootstrap_mean_difference": None,
                "interval_95": None,
                "probability_positive": None,
                "p_value_unadjusted": None,
                "p_value_holm": None,
                "valid_bootstrap_iterations": 0,
            },
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    assert cell_a is not None
    comparison_cells = (cell_a,) if cell_b is None else (cell_a, cell_b)
    if any(cell.circularity_risk for cell in comparison_cells):
        return (
            {
                **base,
                "status": "excluded_circularity_risk",
                "reason": "circularity-risk cells are excluded from confirmatory comparisons",
                "point_metric_a": None,
                "point_metric_b": None,
                "point_difference": None,
                "bootstrap_mean_difference": None,
                "interval_95": None,
                "probability_positive": None,
                "p_value_unadjusted": None,
                "p_value_holm": None,
                "valid_bootstrap_iterations": 0,
            },
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    injected_a = cell_a.injected
    injected_b = cell_b.injected if kind == "cross_cell" and cell_b is not None else injected_a
    if int(injected_a.sum()) == 0 or int(injected_b.sum()) == 0:
        zero_reason = (
            "one or both cross-scenario cells have no injected-corruption events; "
            "no paired inference performed"
            if (
                kind == "cross_cell"
                and cell_b is not None
                and cell_a.cell.scenario_id != cell_b.cell.scenario_id
            )
            else "no injected-corruption events; no random or paired inference performed"
        )
        return (
            {
                **base,
                "status": "not_applicable_zero_corruption",
                "reason": zero_reason,
                "point_metric_a": None,
                "point_metric_b": None,
                "point_difference": None,
                "bootstrap_mean_difference": None,
                "interval_95": None,
                "probability_positive": None,
                "p_value_unadjusted": None,
                "p_value_holm": None,
                "valid_bootstrap_iterations": 0,
            },
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if method_a not in cell_a.risks:
        raise ValueError(f"{comparison_id} method_a is unavailable in its frozen cell")
    first_scores = cell_a.risks[method_a]
    point_b: float | None
    if kind == "method_vs_random":
        second_score_rows = random_scores
        point_b_values = [
            _metric_value(injected_b, row, metric, review_budget=review_budget)
            for row in second_score_rows
        ]
        finite_point_b = [value for value in point_b_values if value is not None]
        if len(finite_point_b) != len(point_b_values):
            raise RuntimeError("random point metrics unexpectedly became not applicable")
        point_b = float(np.mean(finite_point_b))
    else:
        target = cell_a if cell_b is None else cell_b
        if method_b not in target.risks:
            raise ValueError(f"{comparison_id} method_b is unavailable in its frozen cell")
        second_scores = target.risks[method_b]
        second_score_rows = second_scores[np.newaxis, :]
        point_b = _metric_value(injected_b, second_scores, metric, review_budget=review_budget)
    point_a = _metric_value(injected_a, first_scores, metric, review_budget=review_budget)
    if point_a is None or point_b is None:
        raise RuntimeError("nonzero comparison unexpectedly produced an inapplicable point metric")
    accelerated_random_ap = (
        _method_vs_random_average_precision_by_draw(
            injected_b,
            draws,
            random_scores,
        )
        if kind == "method_vs_random" and metric == "average_precision"
        else None
    )
    valid_draw_indices: list[int] = []
    metric_a_values: list[float] = []
    metric_b_values: list[float] = []
    for draw_index, raw_indices in enumerate(draws):
        indices = np.asarray(raw_indices, dtype=np.int64)
        draw_injected_a = injected_a[indices]
        draw_injected_b = injected_b[indices]
        if not int(draw_injected_a.sum()) or not int(draw_injected_b.sum()):
            continue
        first = _metric_value(
            draw_injected_a,
            first_scores[indices],
            metric,
            review_budget=review_budget,
        )
        if first is None:
            continue
        if accelerated_random_ap is None:
            seconds = [
                _metric_value(draw_injected_b, row[indices], metric, review_budget=review_budget)
                for row in second_score_rows
            ]
            if any(value is None for value in seconds):
                continue
            second_mean = float(np.mean([float(value) for value in seconds if value is not None]))
        else:
            accelerated_valid, accelerated_means = accelerated_random_ap
            if not accelerated_valid[draw_index] or not np.isfinite(accelerated_means[draw_index]):
                raise RuntimeError(
                    f"{comparison_id} accelerated random AP validity differs from the "
                    "legacy nonzero-event gate"
                )
            second_mean = float(accelerated_means[draw_index])
        valid_draw_indices.append(draw_index)
        metric_a_values.append(first)
        metric_b_values.append(second_mean)
    metric_a_array = np.asarray(metric_a_values, dtype=np.float64)
    metric_b_array = np.asarray(metric_b_values, dtype=np.float64)
    differences = metric_a_array - metric_b_array
    if not len(differences):
        raise RuntimeError(f"{comparison_id} has no valid whole-group bootstrap iterations")
    probability = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
    p_value = float((1 + np.count_nonzero(differences <= 0.0)) / (len(differences) + 1))
    return (
        {
            **base,
            "status": "reported",
            "reason": None,
            "point_metric_a": float(point_a),
            "point_metric_b": float(point_b),
            "point_difference": float(point_a - point_b),
            "bootstrap_mean_difference": float(differences.mean()),
            "interval_95": [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ],
            "probability_positive": probability,
            "p_value_unadjusted": p_value,
            "p_value_holm": None,
            "valid_bootstrap_iterations": len(differences),
        },
        np.asarray(valid_draw_indices, dtype=np.int64),
        metric_a_array,
        metric_b_array,
    )


def _cell_metrics(
    data: _CellData,
    controls: PrimaryExecutionControls,
) -> dict[str, Any]:
    budgets = (controls.primary_review_budget, *controls.secondary_review_budgets)
    zero_reason = "no injected-corruption events"
    methods: dict[str, Any] = {}
    random_by_budget: dict[str, Any] = {}
    for budget in budgets:
        random = random_review_baseline(
            data.injected,
            budget=budget,
            repeats=controls.random_review_repeats,
            seed=controls.random_review_seed,
        )
        random_by_budget[_budget_key(budget)] = {
            "budget_fraction": budget,
            "reviewed_count": random.reviewed_count,
            "repeats": len(random.seeds),
            "seeds": list(random.seeds),
            "mean_precision": _metric_field(
                random.mean_precision,
                reason="review budget selected no samples",
            ),
            "mean_recall": _metric_field(random.mean_recall, reason=zero_reason),
            "recall_interval_95": (
                None if random.recall_interval_95 is None else list(random.recall_interval_95)
            ),
            "inference_status": (
                "not_applicable_zero_corruption"
                if not int(data.injected.sum())
                else "descriptive_random_repetitions"
            ),
        }
    for method in controls.audit_methods:
        if method not in data.risks:
            methods[method] = {
                "status": "missing_with_recorded_blocker",
                "blocker": "Cleanlab unavailable under frozen failure policy",
            }
            continue
        scores = data.risks[method]
        review_budgets: dict[str, Any] = {}
        for budget in budgets:
            review = evaluate_review_budget(
                data.injected,
                scores,
                budget=budget,
                tie_break_ids=data.sample_ids.tolist(),
            )
            random = random_by_budget[_budget_key(budget)]
            review_budgets[_budget_key(budget)] = {
                "budget_fraction": budget,
                "reviewed_count": review.reviewed_count,
                "injected_reviewed": review.injected_reviewed,
                "precision": _metric_field(
                    review.precision, reason="review budget selected no samples"
                ),
                "recall": _metric_field(review.recall, reason=zero_reason),
                "lift_over_random": _metric_field(review.lift_over_random, reason=zero_reason),
                "expected_random_recall": _metric_field(
                    review.expected_random_recall, reason=zero_reason
                ),
                "tied_random_review": random,
            }
        methods[method] = {
            "status": "available",
            "average_precision": _metric_field(
                average_precision(data.injected, scores), reason=zero_reason
            ),
            "auroc": _metric_field(
                binary_auroc(data.injected, scores),
                reason=(
                    zero_reason
                    if not int(data.injected.sum())
                    else "AUROC requires both binary classes"
                ),
            ),
            "review_budgets": review_budgets,
        }
    return {
        "cell": asdict(data.cell),
        "circularity_risk": data.circularity_risk,
        "primary_confirmatory_eligible": data.primary_confirmatory_eligible,
        "sample_count": len(data.sample_ids),
        "group_count": len(set(data.group_ids.tolist())),
        "injected_corruption_count": int(data.injected.sum()),
        "zero_corruption_status": (
            "not_applicable_for_event_based_inference"
            if not int(data.injected.sum())
            else "has_injected_corruption_events"
        ),
        "methods": methods,
        "random_review_by_budget": random_by_budget,
    }


def _subgroup_rows(
    data: _CellData,
    tissues: NDArray[np.str_],
    controls: PrimaryExecutionControls,
) -> tuple[dict[str, Any], ...]:
    dimensions: tuple[tuple[str, NDArray[np.str_]], ...] = (
        ("class", data.pre_corruption_label.astype(np.str_)),
        ("tissue", tissues),
        ("mechanism", np.full(len(data.sample_ids), data.cell.mechanism, dtype=np.str_)),
        ("rate", np.full(len(data.sample_ids), _budget_key(data.cell.rate), dtype=np.str_)),
    )
    rows: list[dict[str, Any]] = []
    for method in controls.audit_methods:
        scores = data.risks.get(method)
        if scores is None:
            continue
        for dimension, values in dimensions:
            for value in sorted(np.unique(values).tolist()):
                members = values == value
                sample_count = int(members.sum())
                corruption_count = int(data.injected[members].sum())
                if corruption_count == 0:
                    status = "not_applicable_zero_corruption"
                    ap = None
                    reason = "no injected-corruption events in subgroup"
                elif (
                    sample_count < controls.subgroup_min_samples
                    or corruption_count < controls.subgroup_min_corruptions
                ):
                    status = "suppressed_insufficient_support"
                    ap = None
                    reason = (
                        f"requires >= {controls.subgroup_min_samples} samples and >= "
                        f"{controls.subgroup_min_corruptions} injected corruptions"
                    )
                else:
                    status = "reported"
                    ap = average_precision(data.injected[members], scores[members])
                    reason = None
                rows.append(
                    {
                        "cell_id": data.cell.cell_id,
                        "scenario_id": data.cell.scenario_id,
                        "method": method,
                        "dimension": dimension,
                        "value": str(value),
                        "sample_count": sample_count,
                        "injected_corruption_count": corruption_count,
                        "average_precision_status": status,
                        "average_precision": ap,
                        "suppression_reason": reason,
                    }
                )
    return tuple(rows)


def _compute_statistics(
    run_directory: Path,
    controls: PrimaryExecutionControls,
) -> _ComputedStatistics:
    if not isinstance(controls, PrimaryExecutionControls):
        raise TypeError("primary statistics require a real PrimaryExecutionControls instance")
    if controls.frozen_config_schema_version != 2:
        raise ValueError("primary statistics are restricted to frozen schema-v2 real controls")
    controls.validate_for_plan(controls.plan)
    execution_controls = _read_json(
        run_directory / "execution_controls.json", "primary execution controls"
    )
    if canonical_sha256(execution_controls) != canonical_sha256(controls.as_dict()):
        raise ValueError("filesystem execution controls differ from the typed frozen controls")
    readback = read_primary_filesystem_evidence(controls.plan, run_directory)
    if not readback.passed:
        raise ValueError("primary statistics require passed matrix filesystem readback")
    with (run_directory / "cell_index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    completed_ids = {row["cell_id"] for row in rows if row.get("status") == "completed"}
    completed: dict[str, _CellData] = {
        cell.cell_id: _load_cell(run_directory, cell, controls)
        for cell in controls.plan.cells
        if cell.cell_id in completed_ids
    }
    if not completed:
        raise ValueError("primary statistics have no completed frozen cells")
    first = next(iter(completed.values()))
    for data in completed.values():
        if (
            not np.array_equal(data.sample_ids, first.sample_ids)
            or not np.array_equal(data.group_ids, first.group_ids)
            or not np.array_equal(data.pre_corruption_label, first.pre_corruption_label)
        ):
            raise ValueError("primary cells do not share an exact sample/group/reference order")
    scenario_signature: dict[str, tuple[str, str]] = {}
    for data in completed.values():
        signature = (
            data.shared_corruption_sha256,
            canonical_sha256(list(data.shared_corruption_rows)),
        )
        previous = scenario_signature.setdefault(data.cell.scenario_id, signature)
        if previous != signature:
            raise ValueError(
                f"scenario {data.cell.scenario_id} does not share one corruption manifest"
            )
    tissues, crop_sha, input_bindings_sha = _load_tissue_mapping(run_directory, first.sample_ids)
    draws = draw_group_bootstrap_indices(
        first.group_ids.tolist(),
        n_iterations=controls.bootstrap_iterations,
        seed=controls.bootstrap_seed,
    )
    if len(draws) < controls.bootstrap_iterations or controls.bootstrap_iterations < 2_000:
        raise ValueError("primary statistics require at least the frozen 2,000 group bootstraps")
    draw_indices, draw_offsets = _flatten_draws(draws)
    random_scores = _random_rankings(
        len(first.sample_ids), controls.random_review_repeats, controls.random_review_seed
    )
    comparison_definitions: list[
        tuple[
            str,
            PrimaryWithinCellComparison
            | PrimaryMethodVsRandomComparison
            | PrimaryCrossCellComparison,
        ]
    ] = []
    families = (
        ("within_cell", getattr(controls, "within_cell_comparisons", None)),
        ("method_vs_random", getattr(controls, "method_vs_random_comparisons", None)),
        ("cross_cell", getattr(controls, "cross_cell_comparisons", None)),
    )
    if any(values is None for _, values in families):
        raise ValueError("frozen controls lack one or more exact comparison families")
    for kind, values in families:
        assert values is not None
        comparison_definitions.extend((kind, value) for value in values)
    comparison_ids = [value.comparison_id for _, value in comparison_definitions]
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("frozen comparison IDs must be globally unique")
    frozen_holm_families = tuple(str(value) for value in getattr(controls, "holm_families", ()))
    comparisons: list[dict[str, Any]] = []
    comparison_arrays: list[tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]] = []
    allowed_budgets = {controls.primary_review_budget, *controls.secondary_review_budgets}
    for kind, definition in comparison_definitions:
        comparison_id = definition.comparison_id
        metric = definition.metric
        direction = definition.direction
        holm_family = definition.holm_family
        method_a = definition.method_a
        method_b = definition.method_b
        if metric not in _METRICS or direction != _DIRECTION:
            raise ValueError(f"{comparison_id} has an unsupported metric or direction")
        if holm_family not in frozen_holm_families:
            raise ValueError(f"{comparison_id} names an unknown frozen Holm family")
        if method_a not in controls.audit_methods:
            raise ValueError(f"{comparison_id} method_a is absent from frozen audit methods")
        review_budget: float | None = None
        if kind == "cross_cell":
            if not isinstance(definition, PrimaryCrossCellComparison):
                raise TypeError("cross-cell family contains the wrong comparison type")
            planned_a = _resolve_selector(
                controls, definition.selector_a, f"{comparison_id}.selector_a"
            )
            planned_b = _resolve_selector(
                controls, definition.selector_b, f"{comparison_id}.selector_b"
            )
            if planned_a.scenario_id != planned_b.scenario_id and method_a != method_b:
                raise ValueError(
                    f"{comparison_id} controlled cross-scenario comparison must hold the audit "
                    "method fixed"
                )
            data_a = completed.get(planned_a.cell_id)
            data_b = completed.get(planned_b.cell_id)
            _validate_cross_cell_pair(
                comparison_id=comparison_id,
                cell_a=planned_a,
                cell_b=planned_b,
                data_a=data_a,
                data_b=data_b,
            )
        else:
            if not isinstance(
                definition, (PrimaryWithinCellComparison, PrimaryMethodVsRandomComparison)
            ):
                raise TypeError("within-cell family contains the wrong comparison type")
            planned_a = _resolve_selector(
                controls, definition.selector, f"{comparison_id}.selector"
            )
            data_a = completed.get(planned_a.cell_id)
            data_b = None
            if kind == "method_vs_random":
                if not isinstance(definition, PrimaryMethodVsRandomComparison):
                    raise TypeError("method-vs-random family contains the wrong type")
                if method_b != "random_review":
                    raise ValueError(f"{comparison_id} method_b must be random_review")
                review_budget = definition.review_budget
                if review_budget not in allowed_budgets:
                    raise ValueError(f"{comparison_id} review budget is not frozen")
            elif method_b not in controls.audit_methods:
                raise ValueError(f"{comparison_id} method_b is absent from frozen audit methods")
        if metric.endswith("_at_budget") and review_budget is None:
            raise ValueError(f"{comparison_id} budget metric lacks a frozen review budget")
        result, valid_indices, metric_a_values, metric_b_values = _comparison_result(
            comparison_id=comparison_id,
            kind=kind,
            cell_a=data_a,
            cell_b=data_b,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            direction=direction,
            holm_family=holm_family,
            review_budget=review_budget,
            draws=draws,
            random_scores=random_scores,
        )
        comparisons.append(result)
        comparison_arrays.append((valid_indices, metric_a_values, metric_b_values))
    for family in frozen_holm_families:
        indices = [
            index
            for index, item in enumerate(comparisons)
            if item["holm_family"] == family and item["status"] == "reported"
        ]
        if not indices:
            continue
        adjusted = holm_adjust(
            [float(comparisons[index]["p_value_unadjusted"]) for index in indices]
        )
        for index, value in zip(indices, adjusted.tolist(), strict=True):
            comparisons[index]["p_value_holm"] = float(value)

    subgroup_rows = tuple(
        row for data in completed.values() for row in _subgroup_rows(data, tissues, controls)
    )
    bootstrap_arrays: dict[str, NDArray[np.generic]] = {
        "draw_indices": draw_indices,
        "draw_offsets": draw_offsets,
        "sample_ids": first.sample_ids,
        "group_ids": first.group_ids,
        "tissue_types": tissues,
        "random_review_seeds": np.arange(
            controls.random_review_seed,
            controls.random_review_seed + controls.random_review_repeats,
            dtype=np.int64,
        ),
        "comparison_ids": np.asarray(comparison_ids, dtype=np.str_),
        "comparison_kinds": np.asarray([kind for kind, _ in comparison_definitions], dtype=np.str_),
    }
    for index, (valid_indices, metric_a_values, metric_b_values) in enumerate(comparison_arrays):
        prefix = f"comparison_{index:03d}"
        bootstrap_arrays[f"{prefix}_valid_draw_indices"] = valid_indices
        bootstrap_arrays[f"{prefix}_metric_a"] = metric_a_values
        bootstrap_arrays[f"{prefix}_metric_b"] = metric_b_values
        bootstrap_arrays[f"{prefix}_differences"] = metric_a_values - metric_b_values
    cell_payload = [
        _cell_metrics(completed[cell.cell_id], controls)
        for cell in controls.plan.cells
        if cell.cell_id in completed
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_scope": "real_pannuke_primary_statistics",
        "execution_controls_binding_sha256": controls.binding_sha256,
        "matrix_plan_sha256": controls.plan_sha256,
        "source_filesystem_readback_root_sha256": readback.readback_root_sha256,
        "primary_input_bindings_sha256": input_bindings_sha,
        "crop_cache_sha256": crop_sha,
        "primary_metric": controls.primary_metric,
        "review_budget_order": [
            controls.primary_review_budget,
            *controls.secondary_review_budgets,
        ],
        "random_review_algorithm": (
            "descriptive=np.random.Generator.choice via random_review_baseline; "
            "inferential=frozen-seed random permutation rankings"
        ),
        "bootstrap": {
            "unit": controls.statistical_group_unit,
            "resampling_scope": "whole_groups_only",
            "requested_iterations": controls.bootstrap_iterations,
            "saved_draw_count": len(draws),
            "seed": controls.bootstrap_seed,
            "shared_across_all_comparisons": True,
            "evidence_file": _BOOTSTRAP_FILE,
        },
        "multiple_comparison_correction": {
            "method": "holm",
            "families": list(frozen_holm_families),
            "one_sided_p_value_definition": (
                "(1 + count(bootstrap_difference <= 0)) / (1 + valid_iterations)"
            ),
        },
        "scenario_corruption_reconciliation": [
            {
                "scenario_id": scenario_id,
                "shared_scenario_corruption_sha256": signature[0],
                "shared_manifest_rows_sha256": signature[1],
            }
            for scenario_id, signature in sorted(scenario_signature.items())
        ],
        "cells": cell_payload,
        "comparisons": comparisons,
        "subgroups": {
            "file": _SUBGROUP_FILE,
            "row_count": len(subgroup_rows),
            "minimum_samples": controls.subgroup_min_samples,
            "minimum_injected_corruptions": controls.subgroup_min_corruptions,
            "dimensions": ["class", "tissue", "mechanism", "rate"],
        },
        "circularity_excluded_cell_ids": sorted(
            data.cell.cell_id for data in completed.values() if data.circularity_risk
        ),
    }
    return _ComputedStatistics(
        payload=payload,
        bootstrap_arrays=bootstrap_arrays,
        subgroup_rows=subgroup_rows,
        source_readback_root_sha256=readback.readback_root_sha256,
        source_cell_manifest_sha256=dict(readback.cell_artifact_manifest_sha256),
        primary_input_bindings_sha256=input_bindings_sha,
        crop_cache_sha256=crop_sha,
    )


def _manifest_payload(
    run_directory: Path,
    computed: _ComputedStatistics,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_controls_binding_sha256": computed.payload["execution_controls_binding_sha256"],
        "source_filesystem_readback_root_sha256": computed.source_readback_root_sha256,
        "source_cell_artifact_manifest_sha256": dict(
            sorted(computed.source_cell_manifest_sha256.items())
        ),
        "primary_input_bindings_sha256": computed.primary_input_bindings_sha256,
        "crop_cache_sha256": computed.crop_cache_sha256,
        "artifacts": [
            {
                "path": name,
                "size_bytes": (run_directory / name).stat().st_size,
                "sha256": sha256_file(run_directory / name),
            }
            for name in _OUTPUT_FILES
        ],
        "statistics_payload_sha256": canonical_sha256(computed.payload),
        "subgroup_rows_sha256": canonical_sha256(list(computed.subgroup_rows)),
    }


def aggregate_primary_statistics(
    run_directory: str | Path,
    controls: PrimaryExecutionControls,
) -> PrimaryStatisticsArtifacts:
    """Compute once, persist, and strictly attest all frozen primary comparisons."""

    run_path = Path(run_directory).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"primary run directory does not exist: {run_path}")
    for name in (*_OUTPUT_FILES, _MANIFEST_FILE):
        if (run_path / name).exists():
            raise FileExistsError(f"primary statistics never overwrite an artifact: {name}")
    computed = _compute_statistics(run_path, controls)
    bootstrap_path = _atomic_npz(run_path / _BOOTSTRAP_FILE, computed.bootstrap_arrays)
    subgroups_path = atomic_write_text(run_path / _SUBGROUP_FILE, _csv_text(computed.subgroup_rows))
    statistics_path = atomic_write_json(run_path / _STATISTICS_FILE, computed.payload)
    manifest_path = atomic_write_json(
        run_path / _MANIFEST_FILE, _manifest_payload(run_path, computed)
    )
    verification = _verify_persisted_statistics_against_computed(
        run_path,
        controls,
        computed,
    )
    reportable = sum(item["status"] == "reported" for item in computed.payload["comparisons"])
    return PrimaryStatisticsArtifacts(
        output_directory=run_path,
        statistics_path=statistics_path,
        bootstrap_evidence_path=bootstrap_path,
        subgroups_path=subgroups_path,
        manifest_path=manifest_path,
        comparison_count=len(computed.payload["comparisons"]),
        reportable_comparison_count=reportable,
        manifest_sha256=verification.manifest_sha256,
        verification=verification,
    )


def _arrays_equal(actual: NDArray[np.generic], expected: NDArray[np.generic]) -> bool:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    if np.issubdtype(actual.dtype, np.floating):
        return bool(np.array_equal(actual, expected, equal_nan=True))
    return bool(np.array_equal(actual, expected))


def _verify_persisted_statistics_against_computed(
    run_path: Path,
    controls: PrimaryExecutionControls,
    computed: _ComputedStatistics,
) -> PrimaryStatisticsVerification:
    """Attest persisted outputs against one already-computed in-memory result.

    This is deliberately separate from :func:`verify_primary_statistics_artifacts`.
    The aggregation path must not invoke ``_compute_statistics`` a second time, but
    it still fails closed on output serialization, hashes, and any source-matrix
    mutation that occurs between the computation and its attestation.
    """

    for name in (*_OUTPUT_FILES, _MANIFEST_FILE):
        if not (run_path / name).is_file():
            raise ValueError(f"primary statistics artifact is missing: {name}")
    saved_manifest_before = _read_json(run_path / _MANIFEST_FILE, "primary statistics manifest")
    expected_manifest_before = _manifest_payload(run_path, computed)
    if saved_manifest_before != expected_manifest_before:
        raise ValueError("primary statistics manifest is invalid or stale")
    saved_statistics = _read_json(run_path / _STATISTICS_FILE, "primary statistics")
    if saved_statistics != computed.payload:
        raise ValueError("primary_statistics.json differs from recomputed cell evidence")
    expected_subgroups = _csv_text(computed.subgroup_rows)
    if (run_path / _SUBGROUP_FILE).read_text(encoding="utf-8") != expected_subgroups:
        raise ValueError("primary_subgroups.csv differs from recomputed subgroup evidence")
    saved_arrays = _load_npz(run_path / _BOOTSTRAP_FILE, "primary bootstrap evidence")
    if set(saved_arrays) != set(computed.bootstrap_arrays):
        raise ValueError("primary bootstrap evidence has a missing or extra array")
    for name, expected in computed.bootstrap_arrays.items():
        if not _arrays_equal(saved_arrays[name], expected):
            raise ValueError(f"primary bootstrap array differs from recomputation: {name}")

    source_readback = read_primary_filesystem_evidence(controls.plan, run_path)
    if (
        not source_readback.passed
        or source_readback.run_directory.resolve() != run_path
        or source_readback.readback_root_sha256 != computed.source_readback_root_sha256
        or dict(source_readback.cell_artifact_manifest_sha256)
        != dict(computed.source_cell_manifest_sha256)
    ):
        raise ValueError(
            "primary matrix changed between statistics computation and persisted attestation"
        )

    expected_manifest = _manifest_payload(run_path, computed)
    saved_manifest_after = _read_json(run_path / _MANIFEST_FILE, "primary statistics manifest")
    if (
        expected_manifest != expected_manifest_before
        or saved_manifest_after != saved_manifest_before
        or saved_manifest_after != expected_manifest
    ):
        raise ValueError("primary statistics artifacts changed during persisted attestation")
    artifact_records = {str(record["path"]): record for record in expected_manifest["artifacts"]}
    verification = PrimaryStatisticsVerification(
        status="passed",
        output_directory=run_path,
        statistics_sha256=str(artifact_records[_STATISTICS_FILE]["sha256"]),
        bootstrap_evidence_sha256=str(artifact_records[_BOOTSTRAP_FILE]["sha256"]),
        subgroups_sha256=str(artifact_records[_SUBGROUP_FILE]["sha256"]),
        manifest_sha256=sha256_file(run_path / _MANIFEST_FILE),
        source_readback_root_sha256=source_readback.readback_root_sha256,
        comparison_count=len(computed.payload["comparisons"]),
    )
    object.__setattr__(verification, "_attestation", _STATISTICS_VERIFICATION_ATTESTATION)
    return verification


def attest_inherited_primary_statistics_artifacts(
    run_directory: str | Path,
    controls: PrimaryExecutionControls,
    *,
    authorization: (
        AuthorizedPriorNumericVerificationProof | AuthorizedOrphanNumericVerificationProof
    ),
) -> tuple[
    InheritedPrimaryStatisticsVerification,
    (
        InheritedPrimaryStatisticsVerificationProvenance
        | OrphanRecoveryNumericVerificationProvenance
    ),
]:
    """Attest only opaque bytes authorized by one typed prior-numeric proof.

    The function deliberately does not parse any statistics JSON, NPZ member, or
    subgroup CSV value. Numeric validity is inherited either from the exact terminal
    finalization proof or from a separately authorized orphan-recovery proof. This
    readback checks only the content-addressed quartet and the independently typed
    matrix readback.
    """

    if not isinstance(
        authorization,
        (AuthorizedPriorNumericVerificationProof, AuthorizedOrphanNumericVerificationProof),
    ) or not (authorization.valid):
        raise ValueError("inherited statistics require a genuine authorized proof capability")
    supplied_path = Path(run_directory).expanduser()
    if not supplied_path.is_absolute():
        supplied_path = Path.cwd() / supplied_path
    lexical_run_path = Path(os.path.abspath(supplied_path))
    for component in (*reversed(lexical_run_path.parents), lexical_run_path):
        if not component.exists():
            raise ValueError(f"inherited statistics path component is missing: {component}")
        value = component.lstat()
        if component.is_symlink() or bool(getattr(value, "st_file_attributes", 0) & 0x400):
            raise ValueError(
                f"inherited statistics path contains a link or reparse point: {component}"
            )
    run_path = lexical_run_path.resolve()
    if not run_path.is_dir():
        raise ValueError("inherited statistics run directory is not a regular directory")

    authorized_records = {
        name: (size_bytes, digest) for name, size_bytes, digest in authorization.statistics_quartet
    }
    expected_names = {*_OUTPUT_FILES, _MANIFEST_FILE}
    if set(authorized_records) != expected_names:
        raise ValueError("authorized inherited statistics quartet is not exact")
    observed_hashes: dict[str, str] = {}
    for name in sorted(expected_names):
        path = run_path / name
        value = path.lstat()
        if (
            not path.is_file()
            or path.is_symlink()
            or bool(getattr(value, "st_file_attributes", 0) & 0x400)
            or value.st_nlink != 1
        ):
            raise ValueError(
                f"inherited statistics artifact is not one unlinked regular file: {name}"
            )
        expected_size, expected_sha = authorized_records[name]
        observed_sha = sha256_file(path)
        if value.st_size != expected_size or observed_sha != expected_sha:
            raise ValueError(f"inherited statistics artifact differs from its sealed proof: {name}")
        observed_hashes[name] = observed_sha

    readback = read_primary_filesystem_evidence(controls.plan, run_path)
    if (
        not readback.passed
        or readback.run_directory.resolve() != run_path
        or readback.readback_root_sha256 != authorization.source_readback_root_sha256
    ):
        raise ValueError("inherited statistics source matrix differs from its authorized readback")
    verification = InheritedPrimaryStatisticsVerification(
        status="passed",
        output_directory=run_path,
        statistics_sha256=observed_hashes[_STATISTICS_FILE],
        bootstrap_evidence_sha256=observed_hashes[_BOOTSTRAP_FILE],
        subgroups_sha256=observed_hashes[_SUBGROUP_FILE],
        manifest_sha256=observed_hashes[_MANIFEST_FILE],
        source_readback_root_sha256=readback.readback_root_sha256,
        comparison_count=authorization.comparison_count,
        verification_mode=INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
        prior_numeric_verification_proof_sha256=(
            authorization.prior_numeric_verification_proof_sha256
        ),
    )
    object.__setattr__(
        verification,
        "_attestation",
        _INHERITED_STATISTICS_VERIFICATION_ATTESTATION,
    )
    object.__setattr__(
        verification,
        "_authorization_kind",
        authorization.authorization_kind,
    )
    if isinstance(authorization, AuthorizedPriorNumericVerificationProof):
        provenance: (
            InheritedPrimaryStatisticsVerificationProvenance
            | OrphanRecoveryNumericVerificationProvenance
        ) = InheritedPrimaryStatisticsVerificationProvenance(
            schema_version=1,
            verification_mode=INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
            prior_numeric_verification_proof_sha256=(
                authorization.prior_numeric_verification_proof_sha256
            ),
            finalization_successor_authorization_sha256=authorization.authorization_sha256,
            predecessor_run_id=authorization.predecessor_run_id,
            predecessor_artifact_root_sha256=authorization.predecessor_artifact_root_sha256,
            verification_scope="opaque_quartet_hash_and_typed_matrix_readback_only",
            trust_assumption=authorization.trust_assumption,
            limitation=authorization.limitation,
            source_readback_root_sha256=readback.readback_root_sha256,
            statistics_sha256=verification.statistics_sha256,
            bootstrap_evidence_sha256=verification.bootstrap_evidence_sha256,
            subgroups_sha256=verification.subgroups_sha256,
            manifest_sha256=verification.manifest_sha256,
            comparison_count=verification.comparison_count,
        )
    else:
        provenance = OrphanRecoveryNumericVerificationProvenance(
            schema_version=1,
            verification_mode=INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE,
            prior_numeric_verification_proof_sha256=(
                authorization.prior_numeric_verification_proof_sha256
            ),
            authorization_sha256=authorization.authorization_sha256,
            source_run_id=authorization.source_run_id,
            source_snapshot_root_sha256=authorization.source_snapshot_root_sha256,
            source_status_sha256=authorization.source_status_sha256,
            source_tree_root_sha256=authorization.source_tree_root_sha256,
            source_tree_manifest_sha256=authorization.source_tree_manifest_sha256,
            verification_scope="orphan_quartet_hash_and_typed_matrix_readback_only",
            trust_assumption=authorization.trust_assumption,
            limitation=authorization.limitation,
            source_readback_root_sha256=readback.readback_root_sha256,
            statistics_sha256=verification.statistics_sha256,
            bootstrap_evidence_sha256=verification.bootstrap_evidence_sha256,
            subgroups_sha256=verification.subgroups_sha256,
            manifest_sha256=verification.manifest_sha256,
            comparison_count=verification.comparison_count,
        )
    return verification, provenance


def verify_primary_statistics_artifacts(
    run_directory: str | Path,
    controls: PrimaryExecutionControls,
) -> PrimaryStatisticsVerification:
    """Strictly reread and recompute statistics; raise on any semantic tampering."""

    run_path = Path(run_directory).resolve()
    computed = _compute_statistics(run_path, controls)
    return _verify_persisted_statistics_against_computed(run_path, controls, computed)


__all__ = [
    "INHERITED_PRIOR_NUMERIC_VERIFICATION_MODE",
    "AuthorizedOrphanNumericVerificationProof",
    "InheritedPrimaryStatisticsVerification",
    "InheritedPrimaryStatisticsVerificationProvenance",
    "OrphanRecoveryNumericVerificationProvenance",
    "PrimaryStatisticsArtifacts",
    "PrimaryStatisticsVerification",
    "aggregate_primary_statistics",
    "attest_inherited_primary_statistics_artifacts",
    "verify_primary_statistics_artifacts",
]
