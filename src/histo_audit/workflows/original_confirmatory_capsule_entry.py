"""Strict, mode-only entry dispatcher for the sealed original-confirmatory capsule.

The archive bootstrap calls only :func:`_dispatch_original_confirmatory_capsule`.
This module validates the complete mode tail against the frozen authority before
importing any scientific or terminal implementation.  There is deliberately no
generic CLI, dynamic module name, fallback import, or caller-selected production
callable.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Final

from .original_confirmatory_capsule_authority import (
    CAPSULE_ALLOWED_MODES,
    CAPSULE_PRETERMINAL_MODE,
    CAPSULE_SCIENTIFIC_MODE,
    CAPSULE_TERMINAL_MODE,
    canonical_original_confirmatory_capsule_mode_tail,
)

CAPSULE_DISPATCH_INVALID_ARGUMENTS_EXIT_CODE: Final = 64
CAPSULE_DISPATCH_HANDLER_UNAVAILABLE_EXIT_CODE: Final = 69
CAPSULE_DISPATCH_HANDLER_FAILURE_EXIT_CODE: Final = 70
_STOP_PREFIX: Final = "AANCA original-confirmatory capsule STOP: "

type _CanonicalTailHandler = Callable[[tuple[str, ...]], int]
type _ProductionTarget = tuple[str, str]

_PRODUCTION_TARGETS: Final[tuple[tuple[str, _ProductionTarget], ...]] = (
    (
        CAPSULE_SCIENTIFIC_MODE,
        (
            "histo_audit.workflows.original_confirmatory_capsule_authority",
            "_dispatch_original_confirmatory_run_from_canonical_tail",
        ),
    ),
    (
        CAPSULE_PRETERMINAL_MODE,
        (
            "histo_audit.workflows.original_confirmatory_capsule_terminal",
            "_verify_original_confirmatory_preterminal_from_canonical_tail",
        ),
    ),
    (
        CAPSULE_TERMINAL_MODE,
        (
            "histo_audit.workflows.original_confirmatory_capsule_terminal",
            "_verify_original_confirmatory_terminal_from_canonical_tail",
        ),
    ),
)


def _emit_stop(message: str) -> None:
    """Best-effort diagnostic only; inability to log never changes the STOP code."""

    try:
        sys.stderr.write(f"{_STOP_PREFIX}{message}\n")
        sys.stderr.flush()
    except BaseException:
        # The process still returns its already-selected nonzero fail-closed code.
        return


def _canonical_dispatch_request(argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Return one exact mode and reconstructed canonical tail."""

    if (
        type(argv) is not tuple
        or not argv
        or any(type(item) is not str for item in argv)
        or argv[0] not in CAPSULE_ALLOWED_MODES
    ):
        raise ValueError("dispatcher argv is not one exact mode-prefixed string tuple")
    mode = argv[0]
    tail = canonical_original_confirmatory_capsule_mode_tail(
        capsule_mode=mode,
        tail_argv=argv[1:],
    )
    if argv[1:] != tail:
        raise ValueError("dispatcher argv differs from its canonical reconstruction")
    return mode, tail


def _production_target(mode: str) -> _ProductionTarget:
    for candidate_mode, target in _PRODUCTION_TARGETS:
        if candidate_mode == mode:
            return target
    raise LookupError("validated capsule mode has no production target")


def _load_production_handler(mode: str) -> _CanonicalTailHandler:
    """Import exactly one fixed mode implementation, with no fallback."""

    module_name, attribute_name = _production_target(mode)
    module = importlib.import_module(module_name)
    handler = getattr(module, attribute_name, None)
    if not callable(handler):
        raise LookupError("exact capsule mode handler is unavailable")
    return handler


def _invoke_handler(
    handler: _CanonicalTailHandler,
    canonical_tail: tuple[str, ...],
) -> int:
    try:
        result = handler(canonical_tail)
    except BaseException:
        _emit_stop("validated mode handler raised")
        return CAPSULE_DISPATCH_HANDLER_FAILURE_EXIT_CODE
    if type(result) is not int:
        _emit_stop("validated mode handler returned a non-integer exit code")
        return CAPSULE_DISPATCH_HANDLER_FAILURE_EXIT_CODE
    return result


def _dispatch_original_confirmatory_capsule(argv: tuple[str, ...]) -> int:
    """Validate completely, lazily load one fixed handler, and invoke it once."""

    try:
        mode, canonical_tail = _canonical_dispatch_request(argv)
    except BaseException:
        _emit_stop("invalid or noncanonical mode arguments")
        return CAPSULE_DISPATCH_INVALID_ARGUMENTS_EXIT_CODE
    try:
        handler = _load_production_handler(mode)
    except BaseException:
        _emit_stop("exact validated mode handler could not be loaded")
        return CAPSULE_DISPATCH_HANDLER_UNAVAILABLE_EXIT_CODE
    return _invoke_handler(handler, canonical_tail)


def _dispatch_original_confirmatory_capsule_with_map_for_test(
    argv: tuple[str, ...],
    *,
    dispatch_map: Mapping[str, _CanonicalTailHandler],
) -> int:
    """Narrow test seam; production bootstrap never accepts a dispatch map."""

    try:
        mode, canonical_tail = _canonical_dispatch_request(argv)
    except BaseException:
        _emit_stop("invalid or noncanonical mode arguments")
        return CAPSULE_DISPATCH_INVALID_ARGUMENTS_EXIT_CODE
    try:
        if (
            type(dispatch_map) is not dict
            or set(dispatch_map) != set(CAPSULE_ALLOWED_MODES)
            or any(not callable(dispatch_map[item]) for item in CAPSULE_ALLOWED_MODES)
        ):
            raise ValueError("test dispatch map is not exact")
        handler = dispatch_map[mode]
    except BaseException:
        _emit_stop("exact test mode handler is unavailable")
        return CAPSULE_DISPATCH_HANDLER_UNAVAILABLE_EXIT_CODE
    return _invoke_handler(handler, canonical_tail)


__all__ = ["_dispatch_original_confirmatory_capsule"]
