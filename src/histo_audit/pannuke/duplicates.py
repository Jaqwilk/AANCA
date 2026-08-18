"""Read-only, cross-fold PanNuke source-patch duplicate auditing.

Exact equality, perceptual hashes, and a frozen ImageNet representation are
kept as separate evidence signals.  Candidates are recommendations for manual
review only: this module never removes data or changes a split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from histo_audit.data.duplicates import perceptual_hash, perceptual_hash_distance
from histo_audit.representations.cache_provenance import verify_frozen_cache_sidecar
from histo_audit.representations.imagenet import (
    EmbeddingResult,
    PretrainedWeightsUnavailableError,
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
    save_embedding_cache,
)

from .discovery import discover_pannuke_release
from .exceptions import PanNukeSemanticsError
from .io import (
    atomic_replace_via_temp,
    atomic_write_json,
    atomic_write_text,
    deterministic_sample_indices,
    ensure_output_capacity,
    open_npy_mmap,
    sha256_file,
)
from .models import (
    DuplicateAuditArtifacts,
    PanNukeValidationResult,
    ValidationArtifacts,
    VerifiedClassMapping,
)
from .publication import (
    ExclusiveBundlePublicationLock,
    ExclusivePublicationLock,
    PublishedPath,
    create_directory_no_overwrite,
    publish_file_no_overwrite,
    rollback_owned_publications,
)
from .validation import (
    resolve_class_mapping,
    validate_discovered_release,
    verify_raw_inventory_unchanged,
)

EmbeddingStatus = Literal["passed", "blocked", "failed", "not_requested"]

_EMBEDDING_CACHE_NAME = "frozen_resnet18_duplicate_embeddings.npz"
_EMBEDDING_RESUME_DIRECTORY = ".frozen_resnet18_duplicate_embeddings.resume"


@dataclass(frozen=True, slots=True)
class PatchReference:
    """Stable identity and location for one released source patch."""

    sample_id: str
    fold_id: int
    patch_index: int
    image_path: Path
    image_channel_axis: int


@dataclass(frozen=True, slots=True)
class PatchHashRecord:
    """Complete per-patch cryptographic and perceptual provenance."""

    sample_id: str
    fold_id: int
    patch_index: int
    source_image_relative_path: str
    source_image_array_sha256: str
    patch_shape: str
    patch_dtype: str
    canonical_patch_sha256: str
    perceptual_average_hash: str
    perceptual_hash_size: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """One signal-specific candidate pair; never an exclusion decision."""

    method: str
    sample_id_a: str
    sample_id_b: str
    fold_a: int
    fold_b: int
    patch_index_a: int
    patch_index_b: int
    crosses_fold: bool
    exact_sha256: str | None
    perceptual_hash_a: str | None
    perceptual_hash_b: str | None
    perceptual_hamming_distance: int | None
    embedding_cosine_similarity: float | None
    recommended_action: str = "review_only"
    automatic_deletion: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankedDuplicateCandidate:
    """Pair-level consolidation of independent evidence for review ordering."""

    rank: int
    candidate_id: str
    sample_id_a: str
    sample_id_b: str
    fold_a: int
    fold_b: int
    patch_index_a: int
    patch_index_b: int
    exact_match: bool
    exact_sha256: str | None
    perceptual_hamming_distance: int | None
    embedding_cosine_similarity: float | None
    evidence_methods: str
    evidence_count: int
    crosses_fold: bool = True
    recommended_action: str = "review_only"
    automatic_deletion: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmbeddingSignalResult:
    """Honest availability and provenance record for the second near signal."""

    status: EmbeddingStatus
    blocker: str | None
    source: str | None
    sample_count: int
    pairs: tuple[DuplicatePair, ...]
    cache_path: Path | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ReportBundlePaths:
    """Resolved final destinations for the coordinated report transaction."""

    json: Path
    rankings_csv: Path
    markdown: Path
    hash_provenance_csv: Path
    visual_grid: Path

    def ordered_items(self) -> tuple[tuple[str, Path], ...]:
        # JSON is the completion marker and is deliberately published last.
        return (
            ("rankings_csv", self.rankings_csv),
            ("hash_provenance_csv", self.hash_provenance_csv),
            ("visual_grid", self.visual_grid),
            ("markdown", self.markdown),
            ("json", self.json),
        )


@dataclass(frozen=True, slots=True)
class _DuplicatePublicationPlan:
    """Canonical destinations fixed before either publication lock is acquired."""

    raw_root: Path
    output: Path
    report_bundle: _ReportBundlePaths
    selected_cache: Path | None
    selected_cache_sidecar: Path | None
    resume_directory: Path | None
    lock_paths: tuple[Path, ...]


@dataclass(slots=True)
class _DuplicatePublicationContext:
    plan: _DuplicatePublicationPlan
    bundle_lock: ExclusiveBundlePublicationLock
    embedding_publications: _EmbeddingPublicationTracker | None = None


_DUPLICATE_PUBLICATION_CONTEXT: ContextVar[_DuplicatePublicationContext | None] = ContextVar(
    "pannuke_duplicate_publication_context",
    default=None,
)


def _resolved(path: str | Path) -> Path:
    """Resolve traversal and existing symlink/junction components without writing."""

    return Path(path).expanduser().resolve(strict=False)


def _resolved_destination(path: str | Path) -> Path:
    """Resolve a destination parent while preserving its final path component."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve(strict=False) / candidate.name


def _is_within(candidate: Path, directory: Path) -> bool:
    try:
        candidate.relative_to(directory)
    except ValueError:
        return False
    return True


def _is_link_or_junction(path: Path) -> bool:
    """Reject both symbolic links and Windows directory junction reparse points."""

    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction())


def _raw_root_for_destination_guard(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    *,
    class_mapping: VerifiedClassMapping | None,
    use_documented_default_mapping: bool,
) -> Path:
    """Resolve the immutable release root using read-only discovery only."""

    if isinstance(source, ValidationArtifacts):
        return _resolved(source.result.root)
    if isinstance(source, PanNukeValidationResult):
        return _resolved(source.root)
    mapping = resolve_class_mapping(
        class_mapping, use_documented_default=use_documented_default_mapping
    )
    discovery = discover_pannuke_release(source, positive_class_count=len(mapping.class_names))
    return _resolved(discovery.root)


def _validate_suffix(path: Path, expected: str, label: str) -> None:
    if path.suffix.casefold() != expected:
        raise ValueError(f"{label} must use the {expected} suffix: {path}")


def _preflight_duplicate_destinations(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    output_dir: str | Path,
    *,
    class_mapping: VerifiedClassMapping | None,
    use_documented_default_mapping: bool,
    embedding_cache_path: str | Path | None,
    report_path: str | Path | None,
    rankings_csv_path: str | Path | None,
    hash_provenance_csv_path: str | Path | None,
    visual_grid_path: str | Path | None,
    run_embedding_signal: bool,
) -> _DuplicatePublicationPlan:
    """Reject raw writes, aliasing, and malformed destinations before any mkdir."""

    output = _resolved(output_dir)
    bundle = _ReportBundlePaths(
        json=_resolved_destination(output / "pannuke_duplicate_audit.json"),
        rankings_csv=_resolved_destination(
            rankings_csv_path
            if rankings_csv_path is not None
            else output / "cross_fold_duplicate_candidates.csv"
        ),
        markdown=_resolved_destination(
            report_path if report_path is not None else output / "cross_fold_duplicates.md"
        ),
        hash_provenance_csv=_resolved_destination(
            hash_provenance_csv_path
            if hash_provenance_csv_path is not None
            else output / "pannuke_patch_hash_provenance.csv"
        ),
        visual_grid=_resolved_destination(
            visual_grid_path
            if visual_grid_path is not None
            else output / "cross_fold_duplicate_candidate_grid.png"
        ),
    )
    _validate_suffix(bundle.json, ".json", "duplicate-audit JSON")
    _validate_suffix(bundle.rankings_csv, ".csv", "duplicate rankings")
    _validate_suffix(bundle.markdown, ".md", "duplicate Markdown report")
    _validate_suffix(bundle.hash_provenance_csv, ".csv", "patch-hash provenance")
    _validate_suffix(bundle.visual_grid, ".png", "duplicate visual grid")

    selected_cache: Path | None = None
    selected_cache_sidecar: Path | None = None
    resume_directory: Path | None = None
    if run_embedding_signal or embedding_cache_path is not None:
        selected_cache = _resolved_destination(
            embedding_cache_path
            if embedding_cache_path is not None
            else output / _EMBEDDING_CACHE_NAME
        )
        _validate_suffix(selected_cache, ".npz", "duplicate embedding cache")
        selected_cache_sidecar = _resolved_destination(
            selected_cache.with_suffix(f"{selected_cache.suffix}.metadata.json")
        )
        if run_embedding_signal and embedding_cache_path is None:
            resume_directory = _resolved_destination(output / _EMBEDDING_RESUME_DIRECTORY)

    named_paths: list[tuple[str, Path]] = [
        ("output_dir", output),
        *bundle.ordered_items(),
    ]
    file_like = list(bundle.ordered_items())
    if selected_cache is not None and selected_cache_sidecar is not None:
        named_paths.extend(
            (
                ("selected_embedding_cache", selected_cache),
                ("selected_embedding_cache_sidecar", selected_cache_sidecar),
            )
        )
        file_like.extend(
            (
                ("selected_embedding_cache", selected_cache),
                ("selected_embedding_cache_sidecar", selected_cache_sidecar),
            )
        )
    if resume_directory is not None:
        named_paths.append(("embedding_resume_directory", resume_directory))
    aliases: dict[Path, list[str]] = defaultdict(list)
    for label, path in file_like:
        aliases[path].append(label)
    collisions = {path: labels for path, labels in aliases.items() if len(set(labels)) > 1}
    if collisions:
        rendered = "; ".join(
            f"{path} ({', '.join(labels)})"
            for path, labels in sorted(collisions.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(f"duplicate-audit destinations must be pairwise distinct: {rendered}")
    for (first_label, first_path), (second_label, second_path) in combinations(file_like, 2):
        if first_path in second_path.parents or second_path in first_path.parents:
            raise ValueError(
                "duplicate-audit file destinations cannot contain one another: "
                f"{first_label}={first_path}; {second_label}={second_path}"
            )
        if os.path.lexists(first_path) and os.path.lexists(second_path):
            try:
                physically_same = os.path.samefile(first_path, second_path)
            except OSError:
                physically_same = False
            if physically_same:
                raise ValueError(
                    "duplicate-audit destinations physically alias the same file: "
                    f"{first_label}={first_path}; {second_label}={second_path}"
                )
    for label, path in bundle.ordered_items():
        if path == output:
            raise ValueError(f"{label} cannot alias the duplicate-audit output directory")
    if resume_directory is not None:
        for label, path in file_like:
            if path == resume_directory or _is_within(path, resume_directory):
                raise ValueError(
                    f"{label} cannot alias or be placed inside the embedding resume directory"
                )
        if os.path.lexists(resume_directory) and (
            _is_link_or_junction(resume_directory) or not resume_directory.is_dir()
        ):
            raise FileExistsError(
                f"embedding resume destination must be a real directory: {resume_directory}"
            )

    for label, path in file_like:
        if os.path.lexists(path) and _is_link_or_junction(path):
            raise FileExistsError(
                f"duplicate-audit destination cannot be a symbolic link: {label}={path}"
            )

    raw_root = _raw_root_for_destination_guard(
        source,
        class_mapping=class_mapping,
        use_documented_default_mapping=use_documented_default_mapping,
    )
    for label, path in named_paths:
        if path == raw_root or _is_within(path, raw_root):
            raise ValueError(
                "refusing derived duplicate-audit path inside immutable raw PanNuke "
                f"root: {label}={path}; raw_root={raw_root}"
            )
    raw_file_identities: set[tuple[int, int]] = set()
    for raw_path in raw_root.rglob("*"):
        if raw_path.is_file() and not _is_link_or_junction(raw_path):
            value = raw_path.stat(follow_symlinks=False)
            raw_file_identities.add((value.st_dev, value.st_ino))
    for label, path in file_like:
        if not os.path.lexists(path) or not path.is_file():
            continue
        value = path.stat(follow_symlinks=False)
        if (value.st_dev, value.st_ino) in raw_file_identities:
            raise ValueError(
                "refusing duplicate-audit destination hard-linked to immutable raw PanNuke "
                f"data: {label}={path}"
            )

    active_cache_paths = (
        (selected_cache, selected_cache_sidecar)
        if selected_cache is not None and selected_cache_sidecar is not None
        else ()
    )
    if active_cache_paths:
        cache_states = tuple(os.path.lexists(path) for path in active_cache_paths)
        if any(cache_states) and not all(cache_states):
            raise FileExistsError(
                "incomplete frozen duplicate-embedding cache pair; both NPZ and sidecar "
                "are required"
            )
        if all(cache_states) and any(not path.is_file() for path in active_cache_paths):
            raise FileExistsError(
                "frozen duplicate-embedding cache destinations must be regular files"
            )

    lock_paths = tuple(path for _, path in bundle.ordered_items()) + active_cache_paths
    if resume_directory is not None:
        lock_paths = (*lock_paths, resume_directory)
    return _DuplicatePublicationPlan(
        raw_root=raw_root,
        output=output,
        report_bundle=bundle,
        selected_cache=selected_cache,
        selected_cache_sidecar=selected_cache_sidecar,
        resume_directory=resume_directory,
        lock_paths=lock_paths,
    )


def _with_duplicate_publication_lock(
    function: Callable[..., DuplicateAuditArtifacts],
) -> Callable[..., DuplicateAuditArtifacts]:
    """Hold release-wide and canonical per-bundle locks across the whole audit."""

    @wraps(function)
    def wrapped(
        source: PanNukeValidationResult | ValidationArtifacts | str | Path,
        output_dir: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> DuplicateAuditArtifacts:
        if args:
            # The public API makes all remaining arguments keyword-only. Let the
            # wrapped signature produce its normal TypeError without any write.
            return function(source, output_dir, *args, **kwargs)
        plan = _preflight_duplicate_destinations(
            source,
            output_dir,
            class_mapping=kwargs.get("class_mapping"),
            use_documented_default_mapping=kwargs.get("use_documented_default_mapping", True),
            embedding_cache_path=kwargs.get("embedding_cache_path"),
            report_path=kwargs.get("report_path"),
            rankings_csv_path=kwargs.get("rankings_csv_path"),
            hash_provenance_csv_path=kwargs.get("hash_provenance_csv_path"),
            visual_grid_path=kwargs.get("visual_grid_path"),
            run_embedding_signal=kwargs.get("run_embedding_signal", True),
        )
        # The physical lock lives in the OS temp lock registry. Keying the
        # logical identity by raw root serializes all reports and shared caches
        # for one immutable release, even when output directories differ.
        lock_path = plan.raw_root / ".histo-audit-duplicates.logical-lock"
        with (
            ExclusivePublicationLock(lock_path, role="PanNuke duplicate audit"),
            ExclusiveBundlePublicationLock(
                plan.lock_paths,
                role="PanNuke duplicate-audit bundle",
            ) as bundle_lock,
        ):
            bundle_lock.assert_owned()
            locked_plan = _preflight_duplicate_destinations(
                source,
                output_dir,
                class_mapping=kwargs.get("class_mapping"),
                use_documented_default_mapping=kwargs.get("use_documented_default_mapping", True),
                embedding_cache_path=kwargs.get("embedding_cache_path"),
                report_path=kwargs.get("report_path"),
                rankings_csv_path=kwargs.get("rankings_csv_path"),
                hash_provenance_csv_path=kwargs.get("hash_provenance_csv_path"),
                visual_grid_path=kwargs.get("visual_grid_path"),
                run_embedding_signal=kwargs.get("run_embedding_signal", True),
            )
            if locked_plan != plan:
                raise ValueError(
                    "duplicate-audit destination resolution changed while acquiring locks"
                )
            plan.output.mkdir(parents=True, exist_ok=True)
            token = _DUPLICATE_PUBLICATION_CONTEXT.set(
                _DuplicatePublicationContext(plan=plan, bundle_lock=bundle_lock)
            )
            try:
                return function(source, output_dir, **kwargs)
            except BaseException as audit_error:
                active_context = _DUPLICATE_PUBLICATION_CONTEXT.get()
                tracker = (
                    active_context.embedding_publications if active_context is not None else None
                )
                if tracker is not None and tracker.publications:
                    try:
                        _rollback_tracked_embedding_publications(tracker)
                    except RuntimeError:
                        raise RuntimeError(
                            "duplicate audit failed and ownership-safe embedding-cache "
                            "rollback was incomplete"
                        ) from audit_error
                raise
            finally:
                _DUPLICATE_PUBLICATION_CONTEXT.reset(token)

    return wrapped


def _preflight_report_bundle_state(bundle: _ReportBundlePaths) -> Literal["absent", "complete"]:
    states = [(label, path, os.path.lexists(path)) for label, path in bundle.ordered_items()]
    existing = [item for item in states if item[2]]
    if existing and len(existing) != len(states):
        rendered = ", ".join(label for label, _, _ in existing)
        raise FileExistsError(
            "partial duplicate-report bundle exists; refusing any change before repair: "
            f"present={rendered}"
        )
    if existing:
        invalid = [label for label, path, _ in states if not path.is_file()]
        if invalid:
            raise FileExistsError(
                "duplicate-report destinations must be regular files: " + ", ".join(invalid)
            )
        return "complete"
    return "absent"


def _channel_last_patch(reference: PatchReference) -> NDArray[np.generic]:
    array = open_npy_mmap(reference.image_path)
    patch = np.asarray(array[reference.patch_index])
    patch_axis = reference.image_channel_axis - 1
    value = np.moveaxis(patch, patch_axis, -1) if patch_axis != patch.ndim - 1 else patch
    return cast(NDArray[np.generic], value)


def patch_sha256(image: NDArray[np.generic]) -> str:
    """Hash shape, dtype, and canonical channel-last patch bytes with SHA-256."""

    value = np.ascontiguousarray(image)
    header = json.dumps(
        {"shape": value.shape, "dtype": value.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _uint8_rgb(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    value = np.asarray(image[..., :3], dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("duplicate auditing requires finite channel-last RGB patches")
    minimum = float(value.min(initial=0.0))
    maximum = float(value.max(initial=0.0))
    if minimum >= 0.0 and maximum <= 1.0:
        value = value * 255.0
    elif minimum < 0.0 or maximum > 255.0:
        scale = maximum - minimum
        value = (value - minimum) * (255.0 / scale) if scale else np.zeros_like(value)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _references(validation: PanNukeValidationResult) -> tuple[PatchReference, ...]:
    facts = {item.fold_id: item for item in validation.fold_validation}
    values: list[PatchReference] = []
    for fold in validation.folds:
        for patch_index in range(facts[fold.fold_id].n_patches):
            values.append(
                PatchReference(
                    sample_id=f"pannuke-fold-{fold.fold_id}-patch-{patch_index:06d}",
                    fold_id=fold.fold_id,
                    patch_index=patch_index,
                    image_path=fold.image_path,
                    image_channel_axis=fold.image_channel_axis,
                )
            )
    return tuple(values)


def _source_image_hashes(validation: PanNukeValidationResult) -> dict[str, str]:
    return {item.relative_path: item.sha256 for item in validation.inventory}


def _hash_all_patches(
    validation: PanNukeValidationResult,
    references: Sequence[PatchReference],
    *,
    perceptual_hash_size: int,
) -> tuple[tuple[PatchHashRecord, ...], str]:
    source_hashes = _source_image_hashes(validation)
    for image_path in sorted({reference.image_path for reference in references}):
        relative = image_path.relative_to(validation.root).as_posix()
        expected = source_hashes.get(relative)
        if expected is None:
            raise ValueError(f"validated inventory lacks source image hash for {relative}")
        current = sha256_file(image_path)
        if current != expected:
            raise ValueError(
                f"source image array changed after validation: {relative}; "
                "rerun the PanNuke validation gate"
            )
    if not references:
        raise ValueError("duplicate auditing requires at least one source patch")
    first_rgb = _uint8_rgb(_channel_last_patch(references[0]))
    combined_shape = (len(references), *first_rgb.shape)
    rgb_input_digest = hashlib.sha256()
    rgb_input_digest.update(str(combined_shape).encode("ascii"))
    rgb_input_digest.update(np.dtype(np.uint8).str.encode("ascii"))
    records: list[PatchHashRecord] = []
    for reference in references:
        patch = _channel_last_patch(reference)
        rgb = _uint8_rgb(patch)
        if rgb.shape != first_rgb.shape:
            raise ValueError("duplicate auditing requires one canonical RGB patch shape")
        rgb_input_digest.update(np.ascontiguousarray(rgb).tobytes())
        relative = reference.image_path.relative_to(validation.root).as_posix()
        source_sha256 = source_hashes[relative]
        records.append(
            PatchHashRecord(
                sample_id=reference.sample_id,
                fold_id=reference.fold_id,
                patch_index=reference.patch_index,
                source_image_relative_path=relative,
                source_image_array_sha256=source_sha256,
                patch_shape=json.dumps(list(patch.shape), separators=(",", ":")),
                patch_dtype=patch.dtype.str,
                canonical_patch_sha256=patch_sha256(patch),
                perceptual_average_hash=perceptual_hash(rgb, hash_size=perceptual_hash_size),
                perceptual_hash_size=perceptual_hash_size,
            )
        )
    return tuple(records), rgb_input_digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _patch_manifest_binding_sha256(records: Sequence[PatchHashRecord]) -> str:
    """Bind the embedding cache to every ordered validated source patch."""

    return _canonical_json_sha256(
        {
            "schema_version": 1,
            "kind": "pannuke_duplicate_audit_patch_manifest",
            "patches": [
                {
                    "sample_id": item.sample_id,
                    "fold_id": item.fold_id,
                    "patch_index": item.patch_index,
                    "source_image_relative_path": item.source_image_relative_path,
                    "source_image_array_sha256": item.source_image_array_sha256,
                    "canonical_patch_sha256": item.canonical_patch_sha256,
                }
                for item in records
            ],
        }
    )


def _raw_inventory_binding_sha256(validation: PanNukeValidationResult) -> str:
    """Bind derived duplicate evidence to the complete validated raw inventory."""

    return _canonical_json_sha256(
        {
            "schema_version": 1,
            "kind": "pannuke_validated_raw_inventory",
            "files": [
                item.as_dict()
                for item in sorted(validation.inventory, key=lambda value: value.relative_path)
            ],
        }
    )


def _exact_pairs(
    references: Sequence[PatchReference], records: Sequence[PatchHashRecord]
) -> tuple[DuplicatePair, ...]:
    reference_by_id = {item.sample_id: item for item in references}
    by_digest: dict[str, list[PatchHashRecord]] = defaultdict(list)
    for record in records:
        by_digest[record.canonical_patch_sha256].append(record)
    pairs: list[DuplicatePair] = []
    for digest, group in sorted(by_digest.items()):
        for first, second in combinations(group, 2):
            if first.fold_id == second.fold_id:
                continue
            first_reference = reference_by_id[first.sample_id]
            second_reference = reference_by_id[second.sample_id]
            # A hash collision cannot become an exact candidate without equality.
            if not np.array_equal(
                _channel_last_patch(first_reference), _channel_last_patch(second_reference)
            ):
                continue
            pairs.append(
                DuplicatePair(
                    method="exact_sha256",
                    sample_id_a=first.sample_id,
                    sample_id_b=second.sample_id,
                    fold_a=first.fold_id,
                    fold_b=second.fold_id,
                    patch_index_a=first.patch_index,
                    patch_index_b=second.patch_index,
                    crosses_fold=True,
                    exact_sha256=digest,
                    perceptual_hash_a=first.perceptual_average_hash,
                    perceptual_hash_b=second.perceptual_average_hash,
                    perceptual_hamming_distance=perceptual_hash_distance(
                        first.perceptual_average_hash, second.perceptual_average_hash
                    ),
                    embedding_cosine_similarity=None,
                )
            )
    return tuple(pairs)


def _balanced_subset(
    references: Sequence[PatchReference],
    *,
    maximum: int | None,
    bytes_per_sample: int,
    memory_budget_bytes: int,
) -> tuple[PatchReference, ...]:
    if maximum is None or maximum >= len(references):
        return tuple(references)
    by_fold: dict[int, list[PatchReference]] = defaultdict(list)
    for reference in references:
        by_fold[reference.fold_id].append(reference)
    selected: list[PatchReference] = []
    fold_ids = sorted(by_fold)
    base_quota, remainder = divmod(maximum, len(fold_ids))
    for offset, fold_id in enumerate(fold_ids):
        values = by_fold[fold_id]
        quota = base_quota + (1 if offset < remainder else 0)
        if quota == 0:
            continue
        indices = deterministic_sample_indices(
            len(values),
            max_samples=quota,
            bytes_per_sample=bytes_per_sample,
            memory_budget_bytes=max(1, memory_budget_bytes // len(fold_ids)),
        )
        selected.extend(values[index] for index in indices)
    selected_ids = {item.sample_id for item in selected}
    # Preserve the globally stable source order.
    return tuple(item for item in references if item.sample_id in selected_ids)


def _perceptual_pairs(
    records: Sequence[PatchHashRecord],
    selected_ids: set[str],
    *,
    max_hamming_distance: int,
) -> tuple[DuplicatePair, ...]:
    selected = [record for record in records if record.sample_id in selected_ids]
    by_fold: dict[int, list[PatchHashRecord]] = defaultdict(list)
    for record in selected:
        by_fold[record.fold_id].append(record)
    pairs: list[DuplicatePair] = []
    for first_fold, second_fold in combinations(sorted(by_fold), 2):
        for first in by_fold[first_fold]:
            for second in by_fold[second_fold]:
                distance = perceptual_hash_distance(
                    first.perceptual_average_hash, second.perceptual_average_hash
                )
                if distance > max_hamming_distance:
                    continue
                pairs.append(
                    DuplicatePair(
                        method="perceptual_average_hash",
                        sample_id_a=first.sample_id,
                        sample_id_b=second.sample_id,
                        fold_a=first.fold_id,
                        fold_b=second.fold_id,
                        patch_index_a=first.patch_index,
                        patch_index_b=second.patch_index,
                        crosses_fold=True,
                        exact_sha256=(
                            first.canonical_patch_sha256
                            if first.canonical_patch_sha256 == second.canonical_patch_sha256
                            else None
                        ),
                        perceptual_hash_a=first.perceptual_average_hash,
                        perceptual_hash_b=second.perceptual_average_hash,
                        perceptual_hamming_distance=distance,
                        embedding_cosine_similarity=None,
                    )
                )
    return tuple(pairs)


def _streaming_embedding_input_sha256(
    sample_ids: Sequence[str], reference_by_id: dict[str, PatchReference]
) -> str:
    shapes = [_uint8_rgb(_channel_last_patch(reference_by_id[item])).shape for item in sample_ids]
    if not shapes or len(set(shapes)) != 1:
        raise ValueError("embedding input patches must have one non-empty RGB shape")
    combined_shape = (len(shapes), *shapes[0])
    digest = hashlib.sha256()
    digest.update(str(combined_shape).encode("ascii"))
    digest.update(np.dtype(np.uint8).str.encode("ascii"))
    for sample_id in sample_ids:
        patch = _uint8_rgb(_channel_last_patch(reference_by_id[sample_id]))
        digest.update(np.ascontiguousarray(patch).tobytes())
    return digest.hexdigest()


def _validate_embedding_result(
    result: EmbeddingResult,
    reference_by_id: dict[str, PatchReference],
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_input_sha256: str | None = None,
    expected_patch_manifest_sha256: str | None = None,
    expected_raw_inventory_sha256: str | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...], dict[str, Any]]:
    result.validate()
    metadata = result.metadata
    required = {
        "encoder_name": "torchvision.resnet18",
        "encoder_frozen": True,
        "classification_head": "removed (fc=Identity)",
        "weight_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "input_variant": "rgb",
        "output_dimension": 512,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"embedding metadata {key!r} is not the required {expected!r}")
    weight_sha256 = metadata.get("weight_sha256")
    if not isinstance(weight_sha256, str) or len(weight_sha256) != 64:
        raise ValueError("embedding metadata lacks a complete frozen-weight SHA-256")
    if not isinstance(metadata.get("preprocessing"), dict):
        raise ValueError("embedding metadata lacks official preprocessing provenance")
    identifiers = tuple(str(value) for value in result.sample_ids.tolist())
    unknown = set(identifiers) - set(reference_by_id)
    if unknown:
        raise ValueError(
            f"embedding cache contains unknown PanNuke sample IDs: {sorted(unknown)[:3]}"
        )
    if expected_sample_ids is not None and identifiers != tuple(expected_sample_ids):
        raise ValueError(
            "embedding cache sample coverage/order does not match the deterministic "
            "PanNuke duplicate-audit patch order"
        )
    current_input_sha256 = expected_input_sha256 or _streaming_embedding_input_sha256(
        identifiers, reference_by_id
    )
    if metadata.get("input_sha256") != current_input_sha256:
        raise ValueError(
            "embedding input checksum does not match the current canonical PanNuke patches"
        )
    for key, expected in (
        ("manifest_sha256", expected_patch_manifest_sha256),
        ("raw_inventory_sha256", expected_raw_inventory_sha256),
    ):
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f"embedding cache {key} does not match validated PanNuke evidence")
    if (
        expected_patch_manifest_sha256 is not None
        and expected_raw_inventory_sha256 is not None
        and metadata.get("provenance_scope") != "stage_eligible"
    ):
        raise ValueError("duplicate-audit embedding cache is not stage-eligible provenance")
    matrix = np.asarray(result.embeddings, dtype=np.float64)
    return matrix, identifiers, metadata


def _embedding_pairs(
    matrix: NDArray[np.float64],
    sample_ids: Sequence[str],
    reference_by_id: dict[str, PatchReference],
    *,
    min_cosine_similarity: float,
) -> tuple[DuplicatePair, ...]:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("frozen embedding matrix contains a zero-norm vector")
    normalised = matrix / norms[:, None]
    by_fold: dict[int, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        by_fold[reference_by_id[sample_id].fold_id].append(index)
    pairs: list[DuplicatePair] = []
    for first_fold, second_fold in combinations(sorted(by_fold), 2):
        first_indices = by_fold[first_fold]
        second_indices = by_fold[second_fold]
        similarities = np.clip(normalised[first_indices] @ normalised[second_indices].T, -1.0, 1.0)
        rows, columns = np.nonzero(similarities >= min_cosine_similarity)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            first_index = first_indices[row]
            second_index = second_indices[column]
            first = reference_by_id[sample_ids[first_index]]
            second = reference_by_id[sample_ids[second_index]]
            pairs.append(
                DuplicatePair(
                    method="frozen_resnet18_embedding_cosine",
                    sample_id_a=first.sample_id,
                    sample_id_b=second.sample_id,
                    fold_a=first.fold_id,
                    fold_b=second.fold_id,
                    patch_index_a=first.patch_index,
                    patch_index_b=second.patch_index,
                    crosses_fold=True,
                    exact_sha256=None,
                    perceptual_hash_a=None,
                    perceptual_hash_b=None,
                    perceptual_hamming_distance=None,
                    embedding_cosine_similarity=float(similarities[row, column]),
                )
            )
    return tuple(pairs)


def _run_embedding_signal(
    references: Sequence[PatchReference],
    output_dir: Path,
    *,
    embedding_cache_path: str | Path | None,
    max_embedding_patches: int | None,
    min_cosine_similarity: float,
    memory_budget_bytes: int,
    device: str,
    batch_size: int,
    allow_weight_download: bool,
    requested: bool,
    patch_manifest_sha256: str,
    raw_inventory_sha256: str,
    canonical_rgb_input_sha256: str,
    publication_tracker: _EmbeddingPublicationTracker | None = None,
) -> EmbeddingSignalResult:
    if not requested:
        return EmbeddingSignalResult("not_requested", "disabled explicitly", None, 0, (), None, {})
    tracker = publication_tracker or _EmbeddingPublicationTracker(publications=[])
    reference_by_id = {item.sample_id: item for item in references}
    try:
        first = references[0]
        pixels = int(np.prod(_channel_last_patch(first).shape))
        selected = _balanced_subset(
            references,
            maximum=max_embedding_patches,
            bytes_per_sample=max(1, pixels * 10),
            memory_budget_bytes=memory_budget_bytes,
        )
        expected_identifiers = tuple(item.sample_id for item in selected)
        if len(selected) == len(references):
            expected_full_input_sha256 = canonical_rgb_input_sha256
        else:
            expected_full_input_sha256 = _streaming_embedding_input_sha256(
                expected_identifiers, reference_by_id
            )

        if embedding_cache_path is not None:
            result = load_embedding_cache(embedding_cache_path)
            source = "validated_precomputed_cache"
        else:
            destination = output_dir / _EMBEDDING_CACHE_NAME
            destination_sidecar = destination.with_suffix(f"{destination.suffix}.metadata.json")
            destination_states = (
                os.path.lexists(destination),
                os.path.lexists(destination_sidecar),
            )
            if any(destination_states) and not all(destination_states):
                raise FileNotFoundError(
                    "incomplete frozen duplicate-embedding cache pair; both NPZ and sidecar "
                    "are required for safe resume"
                )
            if all(destination_states) and (
                _is_link_or_junction(destination)
                or _is_link_or_junction(destination_sidecar)
                or not destination.is_file()
                or not destination_sidecar.is_file()
            ):
                raise FileExistsError(
                    "frozen duplicate-embedding cache pair must contain real regular files"
                )
            if all(destination_states):
                result = load_embedding_cache(destination)
                source = str(
                    result.metadata.get(
                        "duplicate_audit_embedding_source",
                        "validated_existing_atomic_cache",
                    )
                )
            else:
                # Chunk caches are immutable checkpoints. A rerun validates and reuses every
                # complete chunk, while a missing/corrupt pair fails closed instead of silently
                # mixing representations or recomputing over ambiguous state.
                bytes_per_sample = max(1, pixels * 10)
                chunk_size = max(1, memory_budget_bytes // bytes_per_sample)
                resume_dir = output_dir / _EMBEDDING_RESUME_DIRECTORY
                _ensure_owned_resume_directory(resume_dir, tracker)
                embedding_chunks: list[NDArray[np.float32]] = []
                chunk_metadata: list[dict[str, Any]] = []
                base_metadata: dict[str, Any] | None = None
                reused_chunk_count = 0
                for chunk_index, start in enumerate(range(0, len(selected), chunk_size)):
                    chunk = selected[start : start + chunk_size]
                    chunk_ids = tuple(item.sample_id for item in chunk)
                    chunk_end = start + len(chunk)
                    chunk_path = resume_dir / (
                        f"chunk_{chunk_index:05d}_{start:07d}_{chunk_end:07d}.npz"
                    )
                    chunk_sidecar = chunk_path.with_suffix(f"{chunk_path.suffix}.metadata.json")
                    chunk_states = (
                        os.path.lexists(chunk_path),
                        os.path.lexists(chunk_sidecar),
                    )
                    if any(chunk_states) and not all(chunk_states):
                        raise FileNotFoundError(
                            f"incomplete resumable embedding chunk pair: {chunk_path.name}"
                        )
                    if all(chunk_states) and (
                        _is_link_or_junction(chunk_path)
                        or _is_link_or_junction(chunk_sidecar)
                        or not chunk_path.is_file()
                        or not chunk_sidecar.is_file()
                    ):
                        raise FileExistsError(
                            f"resumable embedding chunk pair is not regular: {chunk_path.name}"
                        )
                    if all(chunk_states):
                        chunk_result = load_embedding_cache(chunk_path)
                        expected_chunk_input_sha256 = _streaming_embedding_input_sha256(
                            chunk_ids, reference_by_id
                        )
                        reused_chunk_count += 1
                        chunk_source = "validated_resume_cache"
                    else:
                        images = np.stack([_uint8_rgb(_channel_last_patch(item)) for item in chunk])
                        chunk_result = extract_resnet18_embeddings(
                            images,
                            chunk_ids,
                            config=ResNet18EmbeddingConfig(
                                input_variant="rgb",
                                weight_identifier="IMAGENET1K_V1",
                                device=device,
                                batch_size=batch_size,
                                allow_weight_download=allow_weight_download,
                            ),
                        )
                        chunk_result.validate()
                        expected_chunk_input_sha256 = str(
                            chunk_result.metadata.get("input_sha256", "")
                        )
                        if len(expected_chunk_input_sha256) != 64:
                            raise ValueError("fresh frozen embedding chunk lacks its input SHA-256")
                        staged_metadata = dict(chunk_result.metadata)
                        staged_metadata.update(
                            {
                                "manifest_sha256": patch_manifest_sha256,
                                "raw_inventory_sha256": raw_inventory_sha256,
                                "provenance_scope": "stage_eligible",
                                "representation_id": (
                                    "pannuke_duplicate_audit_frozen_resnet18_rgb"
                                ),
                                "duplicate_audit_resume_chunk": True,
                                "duplicate_audit_chunk_index": chunk_index,
                                "duplicate_audit_chunk_start": start,
                                "duplicate_audit_chunk_end_exclusive": chunk_end,
                                "duplicate_audit_expected_total_samples": len(selected),
                            }
                        )
                        cache_path, metadata_path, complete_metadata = (
                            _save_tracked_embedding_cache(
                                chunk_path,
                                np.asarray(chunk_result.embeddings, dtype=np.float32),
                                np.asarray(chunk_ids, dtype=np.str_),
                                staged_metadata,
                                tracker,
                            )
                        )
                        chunk_result = EmbeddingResult(
                            embeddings=np.asarray(chunk_result.embeddings, dtype=np.float32),
                            sample_ids=np.asarray(chunk_ids, dtype=np.str_),
                            metadata=complete_metadata,
                            cache_path=cache_path,
                            metadata_path=metadata_path,
                        )
                        chunk_source = "fresh_extraction"
                    chunk_matrix, validated_chunk_ids, validated_chunk_metadata = (
                        _validate_embedding_result(
                            chunk_result,
                            reference_by_id,
                            expected_sample_ids=chunk_ids,
                            expected_input_sha256=expected_chunk_input_sha256,
                            expected_patch_manifest_sha256=patch_manifest_sha256,
                            expected_raw_inventory_sha256=raw_inventory_sha256,
                        )
                    )
                    if base_metadata is None:
                        base_metadata = dict(validated_chunk_metadata)
                    elif validated_chunk_metadata.get("weight_sha256") != base_metadata.get(
                        "weight_sha256"
                    ):
                        raise ValueError("frozen ResNet-18 weight identity changed across chunks")
                    embedding_chunks.append(np.asarray(chunk_matrix, dtype=np.float32))
                    chunk_metadata.append(
                        {
                            "chunk_index": chunk_index,
                            "sample_offset": start,
                            "sample_count": len(validated_chunk_ids),
                            "source": chunk_source,
                            "cache_path": str(chunk_path.resolve()),
                            "cache_sha256": sha256_file(chunk_path),
                            "batch_oom_backoffs": validated_chunk_metadata.get(
                                "batch_oom_backoffs", []
                            ),
                            "extraction_seconds": validated_chunk_metadata.get(
                                "extraction_seconds"
                            ),
                        }
                    )
                if base_metadata is None or not embedding_chunks:
                    raise ValueError("no patches were available for frozen embedding extraction")
                combined_embeddings = np.concatenate(embedding_chunks, axis=0)
                combined_ids = np.asarray(expected_identifiers, dtype=np.str_)
                stable_source = (
                    "resumed_atomic_chunk_caches"
                    if reused_chunk_count
                    else "fresh_official_frozen_resnet18_extraction"
                )
                base_metadata.update(
                    {
                        "sample_count": len(expected_identifiers),
                        "input_sha256": expected_full_input_sha256,
                        "manifest_sha256": patch_manifest_sha256,
                        "raw_inventory_sha256": raw_inventory_sha256,
                        "provenance_scope": "stage_eligible",
                        "representation_id": "pannuke_duplicate_audit_frozen_resnet18_rgb",
                        "duplicate_audit_chunked_extraction": True,
                        "duplicate_audit_chunk_memory_budget_bytes": memory_budget_bytes,
                        "duplicate_audit_extraction_chunks": chunk_metadata,
                        "duplicate_audit_resume_directory": str(resume_dir.resolve()),
                        "duplicate_audit_resumed_chunk_count": reused_chunk_count,
                        "duplicate_audit_embedding_source": stable_source,
                        "coverage_mode": (
                            "full_release"
                            if max_embedding_patches is None
                            else "explicit_bounded_exploratory"
                        ),
                    }
                )
                cache_path, metadata_path, complete_metadata = _save_tracked_embedding_cache(
                    destination,
                    combined_embeddings,
                    combined_ids,
                    base_metadata,
                    tracker,
                )
                result = EmbeddingResult(
                    embeddings=combined_embeddings,
                    sample_ids=combined_ids,
                    metadata=complete_metadata,
                    cache_path=cache_path,
                    metadata_path=metadata_path,
                )
                source = stable_source
        matrix, identifiers_tuple, metadata = _validate_embedding_result(
            result,
            reference_by_id,
            expected_sample_ids=expected_identifiers,
            expected_input_sha256=expected_full_input_sha256,
            expected_patch_manifest_sha256=patch_manifest_sha256,
            expected_raw_inventory_sha256=raw_inventory_sha256,
        )
        if result.cache_path is None:
            raise ValueError("validated duplicate embeddings lack their immutable cache path")
        canonical_metadata = verify_frozen_cache_sidecar(result.cache_path).metadata
        runtime_as_sidecar = dict(metadata)
        contract_input_variant = runtime_as_sidecar.get("contract_input_variant")
        if isinstance(contract_input_variant, str):
            runtime_as_sidecar["input_variant"] = contract_input_variant
        if runtime_as_sidecar != canonical_metadata:
            raise ValueError(
                "duplicate embedding runtime metadata differs from its canonical cache sidecar"
            )
        pairs = _embedding_pairs(
            matrix,
            identifiers_tuple,
            reference_by_id,
            min_cosine_similarity=min_cosine_similarity,
        )
        return EmbeddingSignalResult(
            "passed",
            None,
            source,
            len(identifiers_tuple),
            pairs,
            result.cache_path,
            canonical_metadata,
        )
    except PretrainedWeightsUnavailableError as exc:
        return EmbeddingSignalResult("blocked", str(exc), None, 0, (), None, {})
    except (FileNotFoundError, MemoryError, OSError, RuntimeError, ValueError) as exc:
        return EmbeddingSignalResult(
            "failed", f"{type(exc).__name__}: {exc}", None, 0, (), None, {}
        )


def _candidate_id(first: str, second: str) -> str:
    return hashlib.sha256(f"{first}\0{second}".encode()).hexdigest()[:20]


def _rank_candidates(pairs: Sequence[DuplicatePair]) -> tuple[RankedDuplicateCandidate, ...]:
    grouped: dict[tuple[str, str], list[DuplicatePair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.sample_id_a, pair.sample_id_b)].append(pair)

    interim: list[dict[str, Any]] = []
    for (sample_id_a, sample_id_b), evidence in grouped.items():
        exemplar = evidence[0]
        methods = sorted({item.method for item in evidence})
        exact_hashes = [item.exact_sha256 for item in evidence if item.exact_sha256]
        distances = [
            item.perceptual_hamming_distance
            for item in evidence
            if item.perceptual_hamming_distance is not None
        ]
        similarities = [
            item.embedding_cosine_similarity
            for item in evidence
            if item.embedding_cosine_similarity is not None
        ]
        interim.append(
            {
                "sample_id_a": sample_id_a,
                "sample_id_b": sample_id_b,
                "fold_a": exemplar.fold_a,
                "fold_b": exemplar.fold_b,
                "patch_index_a": exemplar.patch_index_a,
                "patch_index_b": exemplar.patch_index_b,
                "exact_match": "exact_sha256" in methods,
                "exact_sha256": exact_hashes[0] if exact_hashes else None,
                "perceptual_hamming_distance": min(distances) if distances else None,
                "embedding_cosine_similarity": max(similarities) if similarities else None,
                "evidence_methods": "|".join(methods),
                "evidence_count": len(methods),
            }
        )

    def order(item: dict[str, Any]) -> tuple[Any, ...]:
        distance = item["perceptual_hamming_distance"]
        similarity = item["embedding_cosine_similarity"]
        return (
            -int(item["exact_match"]),
            -int(item["evidence_count"]),
            -(float(similarity) if similarity is not None else -1.0),
            int(distance) if distance is not None else 10**9,
            str(item["sample_id_a"]),
            str(item["sample_id_b"]),
        )

    ranked: list[RankedDuplicateCandidate] = []
    for rank, item in enumerate(sorted(interim, key=order), start=1):
        ranked.append(
            RankedDuplicateCandidate(
                rank=rank,
                candidate_id=_candidate_id(item["sample_id_a"], item["sample_id_b"]),
                **item,
            )
        )
    return tuple(ranked)


def _write_csv(destination: Path, rows: Iterable[Any], fieldnames: Sequence[str]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_dict())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _allocate_staged_report_paths(bundle: _ReportBundlePaths) -> dict[str, Path]:
    """Allocate unique, non-final sibling paths for every report artifact."""

    staged: dict[str, Path] = {}
    try:
        for label, destination in bundle.ordered_items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.bundle.",
                suffix=destination.suffix,
                dir=destination.parent,
            )
            os.close(descriptor)
            candidate = Path(temporary_name)
            candidate.unlink()
            staged[label] = candidate
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    return staged


def _replace_staged_report_file(staged: Path, destination: Path) -> PublishedPath:
    """Single replace seam used by the coordinated publish/rollback transaction."""

    return publish_file_no_overwrite(staged, destination)


def _publish_or_verify_report_bundle(
    bundle: _ReportBundlePaths,
    staged: dict[str, Path],
    *,
    initial_state: Literal["absent", "complete"],
    raw_inventory_verifier: Callable[[], object],
    existing_bindings: dict[str, PublishedPath] | None = None,
    staged_bindings: dict[str, PublishedPath] | None = None,
) -> None:
    """Publish all reports together or prove an existing bundle is byte-identical."""

    expected_labels = {label for label, _ in bundle.ordered_items()}
    frozen_staged = staged_bindings or {
        label: _capture_cache_file(staged[label]) for label, _ in bundle.ordered_items()
    }

    def verify_staged_ownership() -> None:
        if set(frozen_staged) != expected_labels:
            raise ValueError("staged duplicate-report binding coverage is incomplete")
        for label, _ in bundle.ordered_items():
            binding = frozen_staged[label]
            if binding.path != staged[label] or not binding.still_owned():
                raise ValueError(
                    f"staged duplicate-report artifact changed after rendering: {label}"
                )

    verify_staged_ownership()
    raw_inventory_verifier()
    verify_staged_ownership()
    current_state = _preflight_report_bundle_state(bundle)
    if current_state != initial_state:
        raise FileExistsError(
            "duplicate-report destination state changed during generation; refusing publish"
        )
    if initial_state == "complete":
        bindings = existing_bindings or {
            label: _capture_cache_file(destination) for label, destination in bundle.ordered_items()
        }
        if set(bindings) != expected_labels or any(
            not binding.still_owned() for binding in bindings.values()
        ):
            raise FileExistsError(
                "complete duplicate-report bundle changed before deterministic readback"
            )
        differing = [
            label
            for label, destination in bundle.ordered_items()
            if sha256_file(destination) != frozen_staged[label].sha256
        ]
        if differing:
            raise FileExistsError(
                "complete duplicate-report bundle differs from the requested deterministic "
                "rerun; refusing overwrite: " + ", ".join(differing)
            )
        raw_inventory_verifier()
        verify_staged_ownership()
        if any(not binding.still_owned() for binding in bindings.values()):
            raise FileExistsError(
                "complete duplicate-report bundle changed during evidence readback"
            )
        return

    expected_sha256 = {
        label: cast(str, frozen_staged[label].sha256) for label, _ in bundle.ordered_items()
    }
    published: list[PublishedPath] = []
    try:
        for label, destination in bundle.ordered_items():
            verify_staged_ownership()
            publication = _replace_staged_report_file(staged[label], destination)
            published.append(publication)
            if publication.sha256 != expected_sha256[label]:
                raise OSError(
                    f"published duplicate-report artifact differs from frozen staging: {label}"
                )
        mismatches = [
            label
            for label, destination in bundle.ordered_items()
            if not destination.is_file() or sha256_file(destination) != expected_sha256[label]
        ]
        if mismatches:
            raise OSError(
                "published duplicate-report artifact failed readback: " + ", ".join(mismatches)
            )
        # A raw mutation racing any of the five promotions invalidates the
        # report transaction; the exception path below removes every new file.
        raw_inventory_verifier()
        verify_staged_ownership()
        if any(not publication.still_owned() for publication in published):
            raise OSError("published duplicate-report bundle changed during final readback")
    except BaseException as publish_error:
        try:
            rollback_owned_publications(published)
        except RuntimeError:
            raise RuntimeError(
                "duplicate-report publish failed and ownership-safe rollback was incomplete"
            ) from publish_error
        raise


def _verify_validation_inventory_unchanged(validation: PanNukeValidationResult) -> None:
    """Re-hash the complete raw inventory immediately before report publication."""

    try:
        verify_raw_inventory_unchanged(validation)
    except PanNukeSemanticsError as error:
        raise ValueError(
            f"raw PanNuke inventory changed during duplicate audit: {error}"
        ) from error


def _verify_duplicate_inputs_unchanged(
    validation: PanNukeValidationResult,
    references: Sequence[PatchReference],
    expected_records: Sequence[PatchHashRecord],
    *,
    perceptual_hash_size: int,
    expected_patch_manifest_sha256: str,
    expected_raw_inventory_sha256: str,
    expected_canonical_rgb_input_sha256: str,
) -> None:
    """Recompute every consumed patch binding, then re-hash the full raw inventory."""

    try:
        current_records, current_rgb_input_sha256 = _hash_all_patches(
            validation,
            references,
            perceptual_hash_size=perceptual_hash_size,
        )
    except ValueError as error:
        raise ValueError(
            f"raw PanNuke inventory changed during duplicate audit: {error}"
        ) from error
    if current_records != tuple(expected_records):
        raise ValueError("canonical ordered patch provenance changed during duplicate audit")
    if _patch_manifest_binding_sha256(current_records) != expected_patch_manifest_sha256:
        raise ValueError("canonical ordered patch-manifest binding changed during duplicate audit")
    if current_rgb_input_sha256 != expected_canonical_rgb_input_sha256:
        raise ValueError("canonical RGB embedding-input binding changed during duplicate audit")
    if _raw_inventory_binding_sha256(validation) != expected_raw_inventory_sha256:
        raise ValueError("validated raw-inventory binding changed during duplicate audit")
    _verify_validation_inventory_unchanged(validation)


@dataclass(slots=True)
class _EmbeddingPublicationTracker:
    """Ownership proofs for cache artifacts created by this audit invocation."""

    publications: list[PublishedPath]


@dataclass(frozen=True, slots=True)
class _EmbeddingCacheBinding:
    """Identity and checksum binding retained through report publication."""

    cache: PublishedPath
    sidecar: PublishedPath
    metadata: dict[str, Any]


def _capture_cache_file(path: Path) -> PublishedPath:
    if _is_link_or_junction(path) or not path.is_file():
        raise FileNotFoundError(f"embedding cache artifact is not a regular file: {path}")
    value = path.stat(follow_symlinks=False)
    return PublishedPath(
        path=path,
        identity=(value.st_dev, value.st_ino, value.st_size, value.st_ctime_ns),
        kind="file",
        sha256=sha256_file(path),
    )


def _capture_embedding_cache_binding(
    embedding: EmbeddingSignalResult,
    *,
    expected_cache_path: Path,
    expected_sidecar_path: Path,
) -> _EmbeddingCacheBinding:
    if embedding.status != "passed" or embedding.cache_path is None:
        raise ValueError("a cache binding requires a passed embedding signal")
    verification = verify_frozen_cache_sidecar(embedding.cache_path)
    if verification.cache_path != expected_cache_path:
        raise ValueError("duplicate embedding cache resolved outside its preflight destination")
    if verification.sidecar_path != expected_sidecar_path:
        raise ValueError("duplicate embedding sidecar resolved outside its preflight destination")
    if verification.metadata != embedding.metadata:
        raise ValueError("duplicate embedding metadata changed before report staging")
    binding = _EmbeddingCacheBinding(
        cache=_capture_cache_file(verification.cache_path),
        sidecar=_capture_cache_file(verification.sidecar_path),
        metadata=dict(verification.metadata),
    )
    if not binding.cache.still_owned() or not binding.sidecar.still_owned():
        raise ValueError("duplicate embedding cache pair changed while it was being bound")
    return binding


def _verify_embedding_cache_binding(binding: _EmbeddingCacheBinding) -> None:
    if not binding.cache.still_owned() or not binding.sidecar.still_owned():
        raise ValueError("duplicate embedding cache pair changed during report publication")
    verification = verify_frozen_cache_sidecar(binding.cache.path)
    if (
        verification.sidecar_path != binding.sidecar.path
        or verification.cache_file_sha256 != binding.cache.sha256
        or verification.sidecar_file_sha256 != binding.sidecar.sha256
        or verification.metadata != binding.metadata
    ):
        raise ValueError("duplicate embedding cache provenance changed during report publication")
    if not binding.cache.still_owned() or not binding.sidecar.still_owned():
        raise ValueError("duplicate embedding cache pair changed during readback")


def _ensure_owned_resume_directory(
    resume_directory: Path,
    tracker: _EmbeddingPublicationTracker,
) -> None:
    if os.path.lexists(resume_directory):
        if _is_link_or_junction(resume_directory) or not resume_directory.is_dir():
            raise FileExistsError(
                f"embedding resume destination is not a real directory: {resume_directory}"
            )
        return
    tracker.publications.append(create_directory_no_overwrite(resume_directory))


def _save_tracked_embedding_cache(
    destination: Path,
    embeddings: NDArray[np.float32],
    sample_ids: NDArray[np.str_],
    metadata: dict[str, Any],
    tracker: _EmbeddingPublicationTracker,
) -> tuple[Path, Path, dict[str, Any]]:
    """Stage a cache pair privately, then publish both with ownership proofs."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.cache-stage-",
        dir=destination.parent,
    ) as temporary_directory:
        staged_cache = Path(temporary_directory) / destination.name
        staged_path, staged_sidecar, complete_metadata = save_embedding_cache(
            staged_cache,
            embeddings,
            sample_ids,
            metadata,
        )
        final_sidecar = destination.with_suffix(f"{destination.suffix}.metadata.json")
        publications: list[PublishedPath] = []
        try:
            publications.append(publish_file_no_overwrite(staged_path, destination))
            publications.append(publish_file_no_overwrite(staged_sidecar, final_sidecar))
            persisted = load_embedding_cache(destination)
            if persisted.metadata != complete_metadata:
                raise ValueError(
                    "newly published duplicate embedding cache metadata changed on readback"
                )
        except BaseException as publication_error:
            try:
                rollback_owned_publications(publications)
            except RuntimeError:
                raise RuntimeError(
                    "embedding-cache pair publication failed and ownership-safe rollback "
                    "was incomplete"
                ) from publication_error
            raise
        tracker.publications.extend(publications)
        return destination, final_sidecar, complete_metadata


def _rollback_tracked_embedding_publications(
    tracker: _EmbeddingPublicationTracker,
) -> None:
    rollback_owned_publications(tracker.publications)
    tracker.publications.clear()


def _candidate_grid_label(candidate: RankedDuplicateCandidate) -> str:
    """Return a review-only caption with stable patch identities and both scores."""

    perceptual = (
        "not_available"
        if candidate.perceptual_hamming_distance is None
        else str(candidate.perceptual_hamming_distance)
    )
    cosine = (
        "not_available"
        if candidate.embedding_cosine_similarity is None
        else f"{candidate.embedding_cosine_similarity:.6f}"
    )
    return (
        f"#{candidate.rank} fold/patch {candidate.fold_a}/{candidate.patch_index_a} vs "
        f"{candidate.fold_b}/{candidate.patch_index_b}; pHash={perceptual}; cosine={cosine}\n"
        f"evidence={candidate.evidence_methods}; review_only"
    )


def _write_candidate_grid(
    destination: Path,
    candidates: Sequence[RankedDuplicateCandidate],
    references: Sequence[PatchReference],
    *,
    max_pairs: int,
) -> Path:
    reference_by_id = {item.sample_id: item for item in references}
    selected = candidates[:max_pairs]
    width = 640
    header_height = 42
    pair_height = 230
    canvas = Image.new("RGB", (width, header_height + max(1, len(selected)) * pair_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), "Cross-fold duplicate candidates — review only", fill="black", font=font)
    if not selected:
        draw.text(
            (12, 60), "No candidate pairs met the frozen thresholds.", fill="black", font=font
        )
    for row, candidate in enumerate(selected):
        y = header_height + row * pair_height
        images: list[Image.Image] = []
        for sample_id in (candidate.sample_id_a, candidate.sample_id_b):
            image = Image.fromarray(_uint8_rgb(_channel_last_patch(reference_by_id[sample_id])))
            image.thumbnail((250, 180), Image.Resampling.LANCZOS)
            images.append(image)
        canvas.paste(images[0], (12, y + 36))
        canvas.paste(images[1], (326, y + 36))
        draw.multiline_text(
            (12, y + 5),
            _candidate_grid_label(candidate),
            fill="black",
            font=font,
            spacing=2,
        )
        draw.line((315, y + 30, 315, y + 216), fill=(160, 160, 160), width=2)

    def writer(path: Path) -> None:
        canvas.save(path, format="PNG", optimize=True)

    return atomic_replace_via_temp(destination, writer)


def _write_markdown(
    destination: Path,
    *,
    total_patches: int,
    perceptual_count: int,
    exact_count: int,
    embedding: EmbeddingSignalResult,
    candidates: Sequence[RankedDuplicateCandidate],
    rankings_path: Path,
    rankings_sha256: str,
    provenance_path: Path,
    provenance_sha256: str,
    grid_path: Path,
    grid_sha256: str,
    validation: PanNukeValidationResult,
    max_hamming_distance: int,
    min_cosine_similarity: float,
) -> Path:
    coverage_complete = (
        embedding.status == "passed"
        and embedding.sample_count == total_patches
        and perceptual_count == total_patches
    )
    status = "complete" if coverage_complete else "incomplete second signal or coverage"
    lines = [
        "# Cross-fold duplicate audit",
        "",
        f"**Duplicate-analysis status:** {status}.",
        "",
        "This is a patch-level data-integrity audit, not a medical assessment. Every row is a "
        "candidate for review only. No patch was deleted, relabelled, or reassigned automatically.",
        "",
        "## Coverage and signals",
        "",
        f"- Source patches with complete SHA-256/perceptual provenance: {total_patches}.",
        f"- Cross-fold exact pairs confirmed by SHA-256 and array equality: {exact_count}.",
        f"- Patches compared by perceptual hash: {perceptual_count}; threshold: Hamming <= {max_hamming_distance}.",
        f"- Frozen ImageNet ResNet-18 embedding status: `{embedding.status}`; patches embedded: {embedding.sample_count}; threshold: cosine >= {min_cosine_similarity:.6f}.",
        "- The perceptual and frozen-embedding signals are methodologically distinct, not claimed to be statistically independent; both consume the same canonical RGB patch.",
        f"- Consolidated cross-fold candidate pairs: {len(candidates)}.",
        "",
    ]
    if embedding.blocker:
        lines.extend(
            [
                "### Embedding blocker",
                "",
                embedding.blocker,
                "",
                "No cosine values were fabricated. Resolve the official-weight/cache blocker and rerun before treating the required two-signal near-duplicate gate as complete.",
                "",
            ]
        )
    elif not coverage_complete:
        lines.extend(
            [
                "### Near-duplicate coverage limitation",
                "",
                "The required two-signal gate is not complete because perceptual and frozen-embedding comparisons did not both cover every validated source patch. Use a full, checksum-validated embedding cache (or a resource-approved full extraction) and rerun before split freeze.",
                "",
            ]
        )
    lines.extend(
        [
            "## Review artifacts",
            "",
            f"- Canonical rankings CSV: `{rankings_path}` (SHA-256 `{rankings_sha256}`).",
            f"- Full per-patch hash provenance: `{provenance_path}` (SHA-256 `{provenance_sha256}`).",
            f"- Candidate-pair visual grid: `{grid_path}` (SHA-256 `{grid_sha256}`).",
            "",
            "Ranking is deterministic and lexicographic: confirmed exact equality, number of distinct evidence signals, embedding cosine, then perceptual distance. It is a review order, not a probability of duplication.",
            "",
            "## Highest-ranked candidates",
            "",
            "| rank | fold/patch A | fold/patch B | exact | pHash distance | cosine | signals | action |",
            "|---:|---|---|:---:|---:|---:|---|---|",
        ]
    )
    for item in candidates[:50]:
        distance = (
            ""
            if item.perceptual_hamming_distance is None
            else str(item.perceptual_hamming_distance)
        )
        cosine = (
            ""
            if item.embedding_cosine_similarity is None
            else f"{item.embedding_cosine_similarity:.6f}"
        )
        lines.append(
            f"| {item.rank} | {item.fold_a}/{item.patch_index_a} | {item.fold_b}/{item.patch_index_b} | "
            f"{str(item.exact_match).lower()} | {distance} | {cosine} | {item.evidence_methods} | review_only |"
        )
    if not candidates:
        lines.append("| — | — | — | — | — | — | none above thresholds | review_only |")
    lines.extend(
        [
            "",
            "## Frozen handling policy before the primary study",
            "",
            "Candidates remain in the source release and are never automatically deleted. Any conservative group exclusion rule must be fixed before primary outcomes and without consulting final labels. If likely cross-fold duplicates are confirmed by dataset review, report the main predefined analysis and a sensitivity analysis excluding all affected source-patch groups symmetrically.",
            "",
            f"Independence limitation: {validation.independence_statement}",
            "",
        ]
    )
    return atomic_write_text(destination, "\n".join(lines))


def _coerce_validation(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    *,
    output_dir: Path,
    class_mapping: VerifiedClassMapping | None,
    use_documented_default_mapping: bool,
) -> PanNukeValidationResult:
    if isinstance(source, ValidationArtifacts):
        return source.result
    if isinstance(source, PanNukeValidationResult):
        return source
    mapping = resolve_class_mapping(
        class_mapping, use_documented_default=use_documented_default_mapping
    )
    discovery = discover_pannuke_release(source, positive_class_count=len(mapping.class_names))
    return validate_discovered_release(
        discovery,
        class_mapping=mapping,
        inventory_exclude_paths=(output_dir,),
    )


@_with_duplicate_publication_lock
def audit_pannuke_duplicates(
    source: PanNukeValidationResult | ValidationArtifacts | str | Path,
    output_dir: str | Path,
    *,
    class_mapping: VerifiedClassMapping | None = None,
    use_documented_default_mapping: bool = True,
    max_perceptual_patches: int | None = None,
    max_hamming_distance: int = 4,
    perceptual_hash_size: int = 8,
    cross_fold_only: bool = True,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    embedding_cache_path: str | Path | None = None,
    max_embedding_patches: int | None = None,
    min_embedding_cosine_similarity: float = 0.995,
    embedding_device: str = "auto",
    embedding_batch_size: int = 32,
    allow_weight_download: bool = False,
    run_embedding_signal: bool = True,
    report_path: str | Path | None = None,
    rankings_csv_path: str | Path | None = None,
    hash_provenance_csv_path: str | Path | None = None,
    visual_grid_path: str | Path | None = None,
    max_visual_pairs: int = 12,
) -> DuplicateAuditArtifacts:
    """Audit cross-fold duplicates while preserving source data unchanged.

    All patches receive canonical SHA-256 and perceptual-hash provenance.  The
    official frozen ResNet-18 signal uses either a checksum-validated cache or
    guarded extraction; unavailable weights are recorded as a blocker and never
    replaced with synthetic scores.
    """

    if (
        (max_perceptual_patches is not None and max_perceptual_patches <= 0)
        or max_hamming_distance < 0
        or perceptual_hash_size <= 0
        or (max_embedding_patches is not None and max_embedding_patches <= 0)
        or not -1.0 <= min_embedding_cosine_similarity <= 1.0
        or embedding_batch_size <= 0
        or max_visual_pairs <= 0
        or memory_budget_bytes <= 0
    ):
        raise ValueError("duplicate-audit limits are invalid")
    if not cross_fold_only:
        raise ValueError("PanNuke duplicate candidates are restricted to cross-fold pairs")
    publication_context = _DUPLICATE_PUBLICATION_CONTEXT.get()
    if publication_context is None:
        raise RuntimeError("duplicate audit requires its complete publication-lock context")
    publication_context.bundle_lock.assert_owned()
    plan = publication_context.plan
    raw_root = plan.raw_root
    resolved_output = plan.output
    report_bundle = plan.report_bundle
    report_bundle_state = _preflight_report_bundle_state(report_bundle)
    existing_report_bindings = (
        {label: _capture_cache_file(path) for label, path in report_bundle.ordered_items()}
        if report_bundle_state == "complete"
        else None
    )
    if report_bundle_state == "complete" and run_embedding_signal and embedding_cache_path is None:
        default_cache = plan.selected_cache
        default_sidecar = plan.selected_cache_sidecar
        assert default_cache is not None and default_sidecar is not None
        if not default_cache.is_file() or not default_sidecar.is_file():
            raise FileExistsError(
                "complete duplicate-report bundle lacks its immutable embedding cache; "
                "refusing a rerun that would change cache state"
            )
    output = ensure_output_capacity(resolved_output, estimated_bytes=32 * 1024 * 1024)
    validation = _coerce_validation(
        source,
        output_dir=output,
        class_mapping=class_mapping,
        use_documented_default_mapping=use_documented_default_mapping,
    )
    if _resolved(validation.root) != raw_root:
        raise ValueError(
            "read-only destination guard and semantic validation resolved different raw roots"
        )
    references = _references(validation)
    if len({item.fold_id for item in references}) < 2:
        raise ValueError("cross-fold duplicate auditing requires at least two validated folds")
    # Final cache + resumable chunks are each approximately n*512*4 bytes. Leave
    # additional headroom for atomic temporary files, reports, and candidate tables.
    estimated_embedding_bytes = len(references) * 512 * np.dtype(np.float32).itemsize
    ensure_output_capacity(
        output,
        estimated_bytes=max(64 * 1024 * 1024, estimated_embedding_bytes * 3),
    )

    records, canonical_rgb_input_sha256 = _hash_all_patches(
        validation,
        references,
        perceptual_hash_size=perceptual_hash_size,
    )
    patch_manifest_sha256 = _patch_manifest_binding_sha256(records)
    raw_inventory_sha256 = _raw_inventory_binding_sha256(validation)
    exact = _exact_pairs(references, records)
    perceptual_references = _balanced_subset(
        references,
        maximum=max_perceptual_patches,
        bytes_per_sample=0,
        memory_budget_bytes=memory_budget_bytes,
    )
    perceptual = _perceptual_pairs(
        records,
        {item.sample_id for item in perceptual_references},
        max_hamming_distance=max_hamming_distance,
    )
    embedding_publications = _EmbeddingPublicationTracker(publications=[])
    publication_context.embedding_publications = embedding_publications
    embedding = _run_embedding_signal(
        references,
        output,
        embedding_cache_path=embedding_cache_path,
        max_embedding_patches=max_embedding_patches,
        min_cosine_similarity=min_embedding_cosine_similarity,
        memory_budget_bytes=memory_budget_bytes,
        device=embedding_device,
        batch_size=embedding_batch_size,
        allow_weight_download=allow_weight_download,
        requested=run_embedding_signal,
        patch_manifest_sha256=patch_manifest_sha256,
        raw_inventory_sha256=raw_inventory_sha256,
        canonical_rgb_input_sha256=canonical_rgb_input_sha256,
        publication_tracker=embedding_publications,
    )
    cache_binding: _EmbeddingCacheBinding | None = None
    if embedding.status == "passed":
        if plan.selected_cache is None or plan.selected_cache_sidecar is None:
            raise ValueError("passed embedding signal lacks its preflight cache destinations")
        cache_binding = _capture_embedding_cache_binding(
            embedding,
            expected_cache_path=plan.selected_cache,
            expected_sidecar_path=plan.selected_cache_sidecar,
        )
    signal_pairs = tuple(
        sorted(
            (*exact, *perceptual, *embedding.pairs),
            key=lambda item: (item.sample_id_a, item.sample_id_b, item.method),
        )
    )
    candidates = _rank_candidates(signal_pairs)

    gate_ready = (
        embedding.status == "passed"
        and embedding.sample_count == len(references)
        and len(perceptual_references) == len(references)
        and len(records) == len(references)
    )
    if gate_ready:
        status = "completed"
    elif embedding.status in {"blocked", "failed", "not_requested"}:
        status = f"completed_with_embedding_{embedding.status}"
    else:
        status = "completed_with_incomplete_near_duplicate_coverage"
    fold_ids = sorted({item.fold_id for item in references})

    def coverage_by_fold(values: Sequence[PatchReference]) -> dict[str, int]:
        return {
            str(fold_id): sum(item.fold_id == fold_id for item in values) for fold_id in fold_ids
        }

    def pair_coverage(values: Sequence[PatchReference]) -> dict[str, int]:
        counts = coverage_by_fold(values)
        return {
            f"{first_fold}-{second_fold}": counts[str(first_fold)] * counts[str(second_fold)]
            for first_fold, second_fold in combinations(fold_ids, 2)
        }

    fold_patch_counts = coverage_by_fold(references)
    full_cross_fold_pair_counts = pair_coverage(references)
    perceptual_cross_fold_pair_counts = pair_coverage(perceptual_references)
    expected_embedding_references = _balanced_subset(
        references,
        maximum=max_embedding_patches,
        bytes_per_sample=max(1, int(np.prod(_channel_last_patch(references[0]).shape)) * 10),
        memory_budget_bytes=memory_budget_bytes,
    )
    embedded_references: Sequence[PatchReference] = (
        expected_embedding_references if embedding.status == "passed" else ()
    )
    embedding_cross_fold_pair_counts = pair_coverage(embedded_references)

    def verify_final_evidence() -> None:
        publication_context.bundle_lock.assert_owned()
        _verify_duplicate_inputs_unchanged(
            validation,
            references,
            records,
            perceptual_hash_size=perceptual_hash_size,
            expected_patch_manifest_sha256=patch_manifest_sha256,
            expected_raw_inventory_sha256=raw_inventory_sha256,
            expected_canonical_rgb_input_sha256=canonical_rgb_input_sha256,
        )
        if cache_binding is not None:
            _verify_embedding_cache_binding(cache_binding)

    staged = _allocate_staged_report_paths(report_bundle)
    try:
        _write_csv(
            staged["rankings_csv"],
            candidates,
            tuple(RankedDuplicateCandidate.__dataclass_fields__),
        )
        _write_csv(
            staged["hash_provenance_csv"],
            records,
            tuple(PatchHashRecord.__dataclass_fields__),
        )
        _write_candidate_grid(
            staged["visual_grid"], candidates, references, max_pairs=max_visual_pairs
        )
        rankings_sha256 = sha256_file(staged["rankings_csv"])
        provenance_sha256 = sha256_file(staged["hash_provenance_csv"])
        grid_sha256 = sha256_file(staged["visual_grid"])
        _write_markdown(
            staged["markdown"],
            total_patches=len(references),
            perceptual_count=len(perceptual_references),
            exact_count=len(exact),
            embedding=embedding,
            candidates=candidates,
            rankings_path=report_bundle.rankings_csv,
            rankings_sha256=rankings_sha256,
            provenance_path=report_bundle.hash_provenance_csv,
            provenance_sha256=provenance_sha256,
            grid_path=report_bundle.visual_grid,
            grid_sha256=grid_sha256,
            validation=validation,
            max_hamming_distance=max_hamming_distance,
            min_cosine_similarity=min_embedding_cosine_similarity,
        )
        markdown_sha256 = sha256_file(staged["markdown"])
        artifact_bindings: dict[str, str] = {
            "rankings_csv": str(report_bundle.rankings_csv),
            "rankings_csv_sha256": rankings_sha256,
            "patch_hash_provenance_csv": str(report_bundle.hash_provenance_csv),
            "patch_hash_provenance_csv_sha256": provenance_sha256,
            "markdown_report": str(report_bundle.markdown),
            "markdown_report_sha256": markdown_sha256,
            "candidate_visual_grid": str(report_bundle.visual_grid),
            "candidate_visual_grid_sha256": grid_sha256,
        }
        if cache_binding is not None:
            assert cache_binding.cache.sha256 is not None
            assert cache_binding.sidecar.sha256 is not None
            artifact_bindings.update(
                {
                    "embedding_cache": str(cache_binding.cache.path),
                    "embedding_cache_sha256": cache_binding.cache.sha256,
                    "embedding_cache_sidecar": str(cache_binding.sidecar.path),
                    "embedding_cache_sidecar_sha256": cache_binding.sidecar.sha256,
                }
            )
        payload = {
            "schema_version": 2,
            "status": status,
            "required_two_signal_near_duplicate_gate_complete": gate_ready,
            "policy": {
                "automatic_deletion": False,
                "candidate_action": "review_only",
                "cross_fold_only": True,
                "split_or_exclusion_change_applied": False,
                "exact_method": (
                    "SHA-256 over shape, dtype, and canonical image bytes plus array equality"
                ),
                "perceptual_method": "deterministic average hash",
                "embedding_method": (
                    "cosine similarity of official frozen torchvision ResNet-18 embeddings"
                ),
                "ranking_rule": (
                    "exact equality, evidence count, embedding cosine, perceptual distance, "
                    "stable IDs"
                ),
                "grouping_unit": "source_patch",
                "final_reference_fold_selection": ("not_selected_or_changed_by_duplicate_audit"),
                "final_reference_outcomes_used": False,
            },
            "near_duplicate_signal_independence": {
                "statement": (
                    "Perceptual average-hash distance and frozen ResNet-18 embedding cosine "
                    "are methodologically distinct pixel-summary and learned-representation "
                    "signals; they are not claimed to be statistically independent."
                ),
                "shared_input": "the same canonical RGB source patch",
                "separate_thresholds": True,
            },
            "thresholds": {
                "perceptual_hash_size": perceptual_hash_size,
                "max_hamming_distance": max_hamming_distance,
                "min_embedding_cosine_similarity": min_embedding_cosine_similarity,
            },
            "coverage": {
                "total_source_patches": len(references),
                "patches_with_full_hash_provenance": len(records),
                "perceptual_comparison_patch_count": len(perceptual_references),
                "embedding_patch_count": embedding.sample_count,
                "fold_patch_counts": fold_patch_counts,
                "full_release_cross_fold_pair_counts_by_fold_pair": (full_cross_fold_pair_counts),
                "perceptual_cross_fold_pair_counts_by_fold_pair": (
                    perceptual_cross_fold_pair_counts
                ),
                "embedding_cross_fold_pair_counts_by_fold_pair": (embedding_cross_fold_pair_counts),
                "full_release_cross_fold_pair_count": sum(full_cross_fold_pair_counts.values()),
                "perceptual_cross_fold_pair_count": sum(perceptual_cross_fold_pair_counts.values()),
                "embedding_cross_fold_pair_count": sum(embedding_cross_fold_pair_counts.values()),
                "sample_order_sha256": _canonical_json_sha256(
                    [item.sample_id for item in references]
                ),
            },
            "provenance_bindings": {
                "patch_manifest_sha256": patch_manifest_sha256,
                "raw_inventory_sha256": raw_inventory_sha256,
                "canonical_rgb_embedding_input_sha256": canonical_rgb_input_sha256,
            },
            # Stable compatibility fields retained for existing downstream readers.
            "total_source_patches_exactly_hashed": len(references),
            "perceptual_sample_patch_count": len(perceptual_references),
            "exact_pair_count": len(exact),
            "perceptual_pair_count": len(perceptual),
            "embedding_pair_count": len(embedding.pairs),
            "cross_fold_exact_pair_count": len(exact),
            "cross_fold_perceptual_pair_count": len(perceptual),
            "counts": {
                "exact_pair_count": len(exact),
                "perceptual_pair_count": len(perceptual),
                "embedding_pair_count": len(embedding.pairs),
                "consolidated_candidate_pair_count": len(candidates),
            },
            "embedding_signal": {
                "status": embedding.status,
                "blocker": embedding.blocker,
                "source": embedding.source,
                "sample_count": embedding.sample_count,
                "full_patch_coverage": embedding.sample_count == len(references),
                "cache_path": str(embedding.cache_path) if embedding.cache_path else None,
                "metadata": embedding.metadata,
            },
            "independence_statement": validation.independence_statement,
            "artifacts": artifact_bindings,
            "signal_pairs": [pair.as_dict() for pair in signal_pairs],
            "pairs": [pair.as_dict() for pair in signal_pairs],
            "ranked_candidates": [candidate.as_dict() for candidate in candidates],
        }
        atomic_write_json(staged["json"], payload)
        frozen_staged_bindings = {
            label: _capture_cache_file(staged[label]) for label, _ in report_bundle.ordered_items()
        }
        _publish_or_verify_report_bundle(
            report_bundle,
            staged,
            initial_state=report_bundle_state,
            raw_inventory_verifier=verify_final_evidence,
            existing_bindings=existing_report_bindings,
            staged_bindings=frozen_staged_bindings,
        )
    finally:
        for staged_path in staged.values():
            staged_path.unlink(missing_ok=True)
    return DuplicateAuditArtifacts(
        json_path=report_bundle.json,
        csv_path=report_bundle.rankings_csv,
        markdown_path=report_bundle.markdown,
        visual_grid_path=report_bundle.visual_grid,
        hash_provenance_csv_path=report_bundle.hash_provenance_csv,
        embedding_cache_path=embedding.cache_path,
        exact_pair_count=len(exact),
        perceptual_pair_count=len(perceptual),
        embedding_pair_count=len(embedding.pairs),
        sampled_patch_count=len(perceptual_references),
        embedding_sampled_patch_count=embedding.sample_count,
        embedding_status=embedding.status,
        required_two_signal_gate_complete=gate_ready,
    )


audit_duplicates = audit_pannuke_duplicates


__all__ = [
    "DuplicatePair",
    "EmbeddingSignalResult",
    "PatchHashRecord",
    "PatchReference",
    "RankedDuplicateCandidate",
    "audit_duplicates",
    "audit_pannuke_duplicates",
    "patch_sha256",
]
