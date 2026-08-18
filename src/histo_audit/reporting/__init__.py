"""Machine-readable-artifact-backed research reporting."""

from .builder import (
    ReportArtifacts,
    build_report,
    build_synthetic_report,
    load_metrics,
    markdown_to_static_html,
    render_synthetic_markdown,
    validate_metrics_payload,
)
from .figures import FigureArtifact, SyntheticFigureSet, build_synthetic_figures
from .synthetic_duplicates import (
    SyntheticDuplicateAuditArtifacts,
    SyntheticDuplicateAuditError,
    audit_synthetic_duplicate_patches,
    reconcile_synthetic_duplicate_audit,
)

__all__ = [
    "FigureArtifact",
    "ReportArtifacts",
    "SyntheticDuplicateAuditArtifacts",
    "SyntheticDuplicateAuditError",
    "SyntheticFigureSet",
    "audit_synthetic_duplicate_patches",
    "build_report",
    "build_synthetic_figures",
    "build_synthetic_report",
    "load_metrics",
    "markdown_to_static_html",
    "reconcile_synthetic_duplicate_audit",
    "render_synthetic_markdown",
    "validate_metrics_payload",
]
