"""Explicit failures raised by the PanNuke data gate."""

from __future__ import annotations


class PanNukeError(RuntimeError):
    """Base class for an actionable PanNuke gate failure."""


class PanNukeNotFoundError(PanNukeError, FileNotFoundError):
    """No usable local PanNuke release root was found."""


class PanNukeDiscoveryError(PanNukeError):
    """The local release layout cannot be identified conservatively."""


class PanNukeSemanticsError(PanNukeError):
    """Array alignment or channel semantics are ambiguous or invalid."""
