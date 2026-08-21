from __future__ import annotations

import hashlib

import numpy as np

from histo_audit.representations.imagenet import _array_sha256, _uint8_rgb


def test_uint8_rgb_uses_the_exact_contiguous_input_without_float64_copy() -> None:
    images = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    converted = _uint8_rgb(images)
    assert converted is images
    np.testing.assert_array_equal(converted, images)


def test_uint8_rgb_only_materialises_non_contiguous_strides() -> None:
    source = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    images = source[:, ::-1]
    assert not images.flags.c_contiguous
    converted = _uint8_rgb(images)
    assert converted.flags.c_contiguous
    np.testing.assert_array_equal(converted, images)


def test_streamed_array_hash_matches_the_previous_byte_contract() -> None:
    array = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
    expected = hashlib.sha256()
    expected.update(str(array.shape).encode("ascii"))
    expected.update(array.dtype.str.encode("ascii"))
    expected.update(array.tobytes())
    assert _array_sha256(array) == expected.hexdigest()
