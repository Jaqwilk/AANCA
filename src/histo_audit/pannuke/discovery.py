"""Conservative discovery of locally extracted PanNuke fold arrays."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .exceptions import PanNukeDiscoveryError, PanNukeSemanticsError
from .io import deterministic_sample_indices, is_archive, open_npy_mmap
from .models import ArrayInspection, DiscoveredFold, ReleaseDiscovery

_FOLD_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])fold[\s_-]*([1-9]\d*)(?=[^0-9]|$)")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def infer_fold_id(path: str | Path, root: str | Path) -> int | None:
    """Infer a fold only from explicit ``fold N`` path markers.

    Bare digits are deliberately ignored. Multiple different fold markers are an
    ambiguous layout and fail rather than being guessed.
    """

    source = Path(path).resolve()
    base = Path(root).resolve()
    try:
        relative = source.relative_to(base).as_posix()
    except ValueError:
        relative = source.as_posix()
    matches = {int(value) for value in _FOLD_PATTERN.findall(relative)}
    if len(matches) > 1:
        raise PanNukeDiscoveryError(
            f"path contains conflicting fold markers {sorted(matches)}: {relative}"
        )
    return next(iter(matches), None)


def _tokens(path: Path) -> set[str]:
    return set(_TOKEN_PATTERN.findall(path.as_posix().lower()))


def _channel_axes(shape: tuple[int, ...], *, minimum_channels: int) -> tuple[int, ...]:
    if len(shape) != 4:
        return ()
    candidates = [axis for axis in (1, 3) if minimum_channels <= shape[axis] <= 64]
    if len(candidates) <= 1:
        return tuple(candidates)
    first, last = candidates
    if shape[first] < shape[last]:
        return (first,)
    if shape[last] < shape[first]:
        return (last,)
    # Equal plausible edge dimensions do not reveal channel orientation.
    return tuple(candidates)


def _numeric_sample(
    array: np.ndarray,
    *,
    max_samples: int,
    memory_budget_bytes: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if array.ndim == 0:
        return np.asarray(array), ()
    per_sample = int(np.prod(array.shape[1:], dtype=np.int64)) * array.dtype.itemsize
    indices = deterministic_sample_indices(
        len(array),
        max_samples=max_samples,
        bytes_per_sample=per_sample,
        memory_budget_bytes=memory_budget_bytes,
    )
    if not indices:
        return np.asarray([], dtype=array.dtype), ()
    # Advanced indexing of a memmap materialises only this bounded inspection sample.
    return np.asarray(array[list(indices)]), indices


def inspect_npy_candidate(
    path: str | Path,
    *,
    root: str | Path,
    positive_class_count: int,
    max_samples: int = 8,
    memory_budget_bytes: int = 128 * 1024 * 1024,
) -> ArrayInspection:
    """Inspect a candidate by header and bounded contents without loading it whole."""

    source = Path(path).resolve()
    base = Path(root).resolve()
    relative = source.relative_to(base).as_posix()
    try:
        array = open_npy_mmap(source)
    except PanNukeSemanticsError as error:
        return ArrayInspection(
            relative_path=relative,
            shape=(),
            dtype="unavailable",
            role_scores={"image": 0.0, "mask": 0.0, "tissue": 0.0},
            sample_min=None,
            sample_max=None,
            finite=None,
            integer_like=None,
            zero_fraction=None,
            sampled_patch_indices=(),
            channel_axis_candidates=(),
            load_error=str(error),
        )

    shape = tuple(int(value) for value in array.shape)
    tokens = _tokens(Path(relative))
    numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    string_like = np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_)
    sample, indices = _numeric_sample(
        array, max_samples=max_samples, memory_budget_bytes=memory_budget_bytes
    )
    sample_min: float | str | None = None
    sample_max: float | str | None = None
    finite: bool | None = None
    integer_like: bool | None = None
    zero_fraction: float | None = None
    if sample.size and numeric:
        finite = bool(np.isfinite(sample).all())
        sample_min = float(np.nanmin(sample))
        sample_max = float(np.nanmax(sample))
        if finite:
            integer_like = bool(np.equal(sample, np.rint(sample)).all())
        zero_fraction = float(np.count_nonzero(sample == 0) / sample.size)
    elif sample.size and string_like:
        values = np.asarray(sample).astype(str)
        string_values = values.ravel().tolist()
        sample_min = min(string_values)
        sample_max = max(string_values)

    image_axes = tuple(
        axis for axis in _channel_axes(shape, minimum_channels=3) if shape[axis] in (3, 4)
    )
    mask_axes = _channel_axes(shape, minimum_channels=positive_class_count)
    scores = {"image": 0.0, "mask": 0.0, "tissue": 0.0}
    if numeric and len(shape) == 4 and len(image_axes) == 1:
        scores["image"] += 6.0
        if (
            finite
            and sample_min is not None
            and sample_max is not None
            and float(sample_min) >= 0.0
            and float(sample_max) <= 65535.0
        ):
            scores["image"] += 1.0
        if zero_fraction is not None and zero_fraction < 0.95:
            scores["image"] += 1.0
    if {"image", "images", "img"}.intersection(tokens):
        scores["image"] += 5.0

    if numeric and len(shape) == 4 and len(mask_axes) == 1:
        scores["mask"] += 6.0
        if integer_like:
            scores["mask"] += 3.0
        if sample_min is not None and float(sample_min) >= 0.0:
            scores["mask"] += 1.0
        if zero_fraction is not None and zero_fraction >= 0.1:
            scores["mask"] += 1.0
    if {"mask", "masks", "instance", "instances"}.intersection(tokens):
        scores["mask"] += 5.0

    if len(shape) == 1 and (string_like or numeric):
        scores["tissue"] += 4.0
        if string_like:
            scores["tissue"] += 3.0
    if {"type", "types", "tissue", "tissues"}.intersection(tokens):
        scores["tissue"] += 5.0

    all_channel_axes = tuple(sorted(set(image_axes).union(mask_axes)))
    return ArrayInspection(
        relative_path=relative,
        shape=shape,
        dtype=array.dtype.str,
        role_scores=scores,
        sample_min=sample_min,
        sample_max=sample_max,
        finite=finite,
        integer_like=integer_like,
        zero_fraction=zero_fraction,
        sampled_patch_indices=indices,
        channel_axis_candidates=all_channel_axes,
    )


def _select_role(
    inspections: list[ArrayInspection],
    *,
    role: str,
    fold_id: int,
    minimum_score: float,
) -> ArrayInspection:
    ranked = sorted(
        inspections,
        key=lambda item: (-item.role_scores[role], item.relative_path),
    )
    eligible = [item for item in ranked if item.role_scores[role] >= minimum_score]
    if not eligible:
        errors = [item.load_error for item in inspections if item.load_error]
        detail = f" Load errors: {'; '.join(errors)}" if errors else ""
        raise PanNukeDiscoveryError(
            f"fold {fold_id}: no credible {role} .npy array was found.{detail}"
        )
    top_score = eligible[0].role_scores[role]
    tied = [item for item in eligible if item.role_scores[role] == top_score]
    if len(tied) != 1:
        paths = [item.relative_path for item in tied]
        raise PanNukeDiscoveryError(
            f"fold {fold_id}: ambiguous {role} arrays with equal evidence: {paths}"
        )
    return tied[0]


def _role_axis(inspection: ArrayInspection, *, role: str, class_count: int) -> int:
    shape = inspection.shape
    if role == "image":
        candidates = tuple(
            axis for axis in _channel_axes(shape, minimum_channels=3) if shape[axis] in (3, 4)
        )
    else:
        candidates = _channel_axes(shape, minimum_channels=class_count)
    if len(candidates) != 1:
        raise PanNukeDiscoveryError(
            f"cannot resolve {role} channel axis for {inspection.relative_path}: "
            f"shape={shape}, candidates={candidates}"
        )
    return candidates[0]


def discover_pannuke_release(
    root: str | Path,
    *,
    positive_class_count: int = 5,
    max_inspection_samples: int = 8,
    memory_budget_bytes: int = 128 * 1024 * 1024,
) -> ReleaseDiscovery:
    """Discover extracted folds and archives without extracting or downloading data."""

    base = Path(root).resolve()
    if not base.is_dir():
        raise PanNukeDiscoveryError(f"PanNuke root is not a directory: {base}")
    if positive_class_count <= 0:
        raise ValueError("positive_class_count must be positive")
    all_files = sorted((path for path in base.rglob("*") if path.is_file()), key=str)
    archives = tuple(path for path in all_files if is_archive(path))
    npy_files = tuple(path for path in all_files if path.suffix.lower() == ".npy")
    if not npy_files:
        if archives:
            relative = [path.relative_to(base).as_posix() for path in archives]
            raise PanNukeDiscoveryError(
                "only PanNuke archive candidates were found; extract the verified release "
                f"without overwriting source archives, then rerun validation: {relative}"
            )
        raise PanNukeDiscoveryError(f"no .npy arrays or release archives found under {base}")

    fold_paths: dict[int, list[Path]] = {}
    unassigned: list[str] = []
    for path in npy_files:
        fold_id = infer_fold_id(path, base)
        if fold_id is None:
            unassigned.append(path.relative_to(base).as_posix())
        else:
            fold_paths.setdefault(fold_id, []).append(path)
    if unassigned:
        raise PanNukeDiscoveryError(
            "conservative discovery requires every candidate .npy path to carry an explicit "
            f"fold marker; unassigned arrays: {unassigned}"
        )
    inspections = tuple(
        inspect_npy_candidate(
            path,
            root=base,
            positive_class_count=positive_class_count,
            max_samples=max_inspection_samples,
            memory_budget_bytes=memory_budget_bytes,
        )
        for path in npy_files
    )
    by_relative = {item.relative_path: item for item in inspections}
    resolved: list[DiscoveredFold] = []
    for fold_id, paths in sorted(fold_paths.items()):
        fold_inspections = [by_relative[path.relative_to(base).as_posix()] for path in paths]
        image = _select_role(fold_inspections, role="image", fold_id=fold_id, minimum_score=6.0)
        mask = _select_role(fold_inspections, role="mask", fold_id=fold_id, minimum_score=6.0)
        tissue = _select_role(fold_inspections, role="tissue", fold_id=fold_id, minimum_score=4.0)
        selected = {image.relative_path, mask.relative_path, tissue.relative_path}
        if len(selected) != 3:
            raise PanNukeDiscoveryError(
                f"fold {fold_id}: one array was assigned multiple roles: {sorted(selected)}"
            )
        resolved.append(
            DiscoveredFold(
                fold_id=fold_id,
                image_path=base / image.relative_path,
                mask_path=base / mask.relative_path,
                tissue_path=base / tissue.relative_path,
                image_channel_axis=_role_axis(
                    image, role="image", class_count=positive_class_count
                ),
                mask_channel_axis=_role_axis(mask, role="mask", class_count=positive_class_count),
            )
        )
    return ReleaseDiscovery(
        root=base,
        fold_ids=tuple(sorted(fold_paths)),
        npy_files=npy_files,
        archives=archives,
        inspections=inspections,
        folds=tuple(resolved),
    )


# Short alias for CLI and external callers.
discover_release = discover_pannuke_release


__all__ = [
    "discover_pannuke_release",
    "discover_release",
    "infer_fold_id",
    "inspect_npy_candidate",
]
