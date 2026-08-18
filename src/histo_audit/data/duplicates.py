"""Read-only exact and near-duplicate candidate discovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """A candidate pair for review; this module never deletes either sample."""

    sample_id_a: str
    sample_id_b: str
    fold_a: int | str | None
    fold_b: int | str | None
    exact_sha256: str | None
    perceptual_hamming_distance: int | None
    embedding_cosine_similarity: float | None
    crosses_fold: bool
    recommended_action: str = "review_only"


def canonical_array_sha256(array: NDArray[np.generic]) -> str:
    """Hash canonical shape, dtype, and contiguous pixel/feature bytes."""

    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    header = json.dumps(
        {"shape": contiguous.shape, "dtype": contiguous.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def perceptual_hash(image: NDArray[np.generic], *, hash_size: int = 8) -> str:
    """Return a deterministic average hash as a zero-padded hexadecimal string."""

    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be grayscale or channel-last RGB")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError("image channel count must be 1, 3, or 4")
    uint8 = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    pil = (
        Image.fromarray(uint8)
        .convert("L")
        .resize((hash_size, hash_size), resample=Image.Resampling.LANCZOS)
    )
    pixels = np.asarray(pil, dtype=np.float64)
    bits = pixels >= pixels.mean()
    bit_string = "".join("1" if bit else "0" for bit in bits.ravel())
    width = int(np.ceil(hash_size * hash_size / 4))
    return f"{int(bit_string, 2):0{width}x}"


def perceptual_hash_distance(first: str, second: str) -> int:
    """Calculate Hamming distance between equal-width hexadecimal hashes."""

    if len(first) != len(second):
        raise ValueError("perceptual hashes must have equal width")
    try:
        return int((int(first, 16) ^ int(second, 16)).bit_count())
    except ValueError as error:
        raise ValueError("perceptual hashes must be hexadecimal") from error


def _validate_identifiers(
    n_samples: int,
    sample_ids: Sequence[str],
    folds: Sequence[int | str] | None,
) -> tuple[tuple[str, ...], tuple[int | str | None, ...]]:
    identifiers = tuple(str(value) for value in sample_ids)
    if len(identifiers) != n_samples or len(set(identifiers)) != n_samples:
        raise ValueError("sample IDs must align and be unique")
    if folds is None:
        fold_values: tuple[int | str | None, ...] = (None,) * n_samples
    else:
        if len(folds) != n_samples:
            raise ValueError("folds must align with samples")
        fold_values = tuple(folds)
    return identifiers, fold_values


def find_exact_duplicate_pairs(
    arrays: Sequence[NDArray[np.generic]] | NDArray[np.generic],
    *,
    sample_ids: Sequence[str],
    folds: Sequence[int | str] | None = None,
    cross_fold_only: bool = False,
) -> tuple[DuplicateCandidate, ...]:
    """Report SHA-256 candidates confirmed with exact array equality."""

    values = [np.asarray(value) for value in arrays]
    identifiers, fold_values = _validate_identifiers(len(values), sample_ids, folds)
    hashes = [canonical_array_sha256(value) for value in values]
    by_hash: dict[str, list[int]] = {}
    for index, digest in enumerate(hashes):
        by_hash.setdefault(digest, []).append(index)
    candidates: list[DuplicateCandidate] = []
    for digest, indices in sorted(by_hash.items()):
        for first, second in combinations(indices, 2):
            crosses = fold_values[first] != fold_values[second]
            if cross_fold_only and not crosses:
                continue
            if not np.array_equal(values[first], values[second]):
                continue
            candidates.append(
                DuplicateCandidate(
                    sample_id_a=identifiers[first],
                    sample_id_b=identifiers[second],
                    fold_a=fold_values[first],
                    fold_b=fold_values[second],
                    exact_sha256=digest,
                    perceptual_hamming_distance=0,
                    embedding_cosine_similarity=None,
                    crosses_fold=crosses,
                )
            )
    return tuple(candidates)


def find_perceptual_duplicate_candidates(
    images: Sequence[NDArray[np.generic]] | NDArray[np.generic],
    *,
    sample_ids: Sequence[str],
    folds: Sequence[int | str] | None = None,
    max_hamming_distance: int = 4,
    hash_size: int = 8,
    cross_fold_only: bool = True,
) -> tuple[DuplicateCandidate, ...]:
    """Report perceptually similar pairs without applying an exclusion policy."""

    if max_hamming_distance < 0:
        raise ValueError("max_hamming_distance must be non-negative")
    values = [np.asarray(value) for value in images]
    identifiers, fold_values = _validate_identifiers(len(values), sample_ids, folds)
    hashes = [perceptual_hash(value, hash_size=hash_size) for value in values]
    candidates: list[DuplicateCandidate] = []
    for first, second in combinations(range(len(values)), 2):
        crosses = fold_values[first] != fold_values[second]
        if cross_fold_only and not crosses:
            continue
        distance = perceptual_hash_distance(hashes[first], hashes[second])
        if distance <= max_hamming_distance:
            candidates.append(
                DuplicateCandidate(
                    sample_id_a=identifiers[first],
                    sample_id_b=identifiers[second],
                    fold_a=fold_values[first],
                    fold_b=fold_values[second],
                    exact_sha256=(
                        canonical_array_sha256(values[first])
                        if np.array_equal(values[first], values[second])
                        else None
                    ),
                    perceptual_hamming_distance=distance,
                    embedding_cosine_similarity=None,
                    crosses_fold=crosses,
                )
            )
    return tuple(candidates)


def find_embedding_duplicate_candidates(
    embeddings: NDArray[np.generic],
    *,
    sample_ids: Sequence[str],
    folds: Sequence[int | str] | None = None,
    min_cosine_similarity: float = 0.995,
    cross_fold_only: bool = True,
) -> tuple[DuplicateCandidate, ...]:
    """Report high-cosine candidate pairs as a second independent near-duplicate signal."""

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite non-empty matrix")
    if not -1.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must lie in [-1, 1]")
    identifiers, fold_values = _validate_identifiers(len(matrix), sample_ids, folds)
    norms = np.linalg.norm(matrix, axis=1)
    normalised = matrix / np.maximum(norms[:, None], 1e-12)
    candidates: list[DuplicateCandidate] = []
    for first, second in combinations(range(len(matrix)), 2):
        crosses = fold_values[first] != fold_values[second]
        if cross_fold_only and not crosses:
            continue
        similarity = float(np.clip(normalised[first] @ normalised[second], -1.0, 1.0))
        if similarity >= min_cosine_similarity:
            candidates.append(
                DuplicateCandidate(
                    sample_id_a=identifiers[first],
                    sample_id_b=identifiers[second],
                    fold_a=fold_values[first],
                    fold_b=fold_values[second],
                    exact_sha256=None,
                    perceptual_hamming_distance=None,
                    embedding_cosine_similarity=similarity,
                    crosses_fold=crosses,
                )
            )
    return tuple(candidates)
