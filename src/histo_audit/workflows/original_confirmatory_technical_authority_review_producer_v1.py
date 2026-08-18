"""Fresh-process, outcome-blind producer for the original-confirmatory T0 review.

This module is deliberately separate from the intent builder and T0 publisher.
It reads one already-created canonical intent, independently rechecks the bound
live inputs without reading scientific outcome values, builds the exact review
through the live schema, and CREATE_NEW-publishes one immutable review receipt.

It never publishes T0/Q/E, executes science, retries, adopts, overwrites, removes,
or repairs any artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from . import original_confirmatory_technical_authority_publication_v1 as publication
from . import (
    original_confirmatory_technical_authority_v1 as authority_schema,
)


class OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(RuntimeError):
    """The fresh outcome-blind review producer failed closed."""


_UTC_CLOCK_FOR_TESTS_ONLY: Callable[[], datetime] | None = None


def _utc_now() -> str:
    clock = _UTC_CLOCK_FOR_TESTS_ONLY
    value = datetime.now(UTC) if clock is None else clock()
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "internal UTC clock returned a non-UTC timestamp"
        )
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _resolve_from_root(project_root: Path, supplied: Path) -> Path:
    value = supplied if supplied.is_absolute() else project_root / supplied
    return Path(os.path.abspath(value))


def _paired_request_paths(
    *,
    project_root: Path,
    intent_json: Path,
    output: Path,
) -> tuple[Path, Path]:
    intent_path = publication._require_authority_request_leaf(
        project_root=project_root,
        path=_resolve_from_root(project_root, intent_json),
        expected_filename=publication.INTENT_REQUEST_FILENAME,
        role="technical intent input",
    )
    output_path = publication._require_authority_request_leaf(
        project_root=project_root,
        path=_resolve_from_root(project_root, output),
        expected_filename=publication.REVIEW_REQUEST_FILENAME,
        role="independent-review output",
    )
    return intent_path, output_path


def _capture_parent_controller_process_v1() -> dict[str, Any]:
    """Authenticate the live parent that must retain this reviewer child."""

    parent_process_id = os.getppid()
    if parent_process_id <= 0 or parent_process_id == os.getpid():
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "fresh reviewer has no distinct live parent controller"
        )
    try:
        return publication._capture_process_identity_v1(
            parent_process_id,
            Path(publication.__file__).resolve(strict=True),
        )
    except Exception as exc:
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "fresh reviewer cannot authenticate its live parent controller"
        ) from exc


def produce_original_confirmatory_technical_authority_review_v1(
    *,
    intent_json: str | Path,
    output: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build and durably CREATE_NEW-publish one exact outcome-blind review."""

    root = Path(project_root).resolve(strict=True)
    intent_path, output_path = _paired_request_paths(
        project_root=root,
        intent_json=Path(intent_json),
        output=Path(output),
    )
    if os.path.lexists(output_path):
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "review output already exists; overwrite, adoption, cleanup, and retry are forbidden"
        )

    intent = authority_schema.canonical_original_confirmatory_technical_authority_intent_v1(
        publication._read_json_input(intent_path, role="technical intent")
    )
    source_inventory_path = Path(intent["execution_source"]["manifest_path"])
    source_inventory = publication._read_json_input(
        source_inventory_path,
        role="execution source inventory",
    )

    review_attempt = publication.verify_original_confirmatory_technical_review_attempt_claim_v1(
        root
        / "artifacts"
        / publication.AUTHORITY_REQUEST_DIRECTORY_NAME
        / publication.REVIEW_ATTEMPT_FILENAME,
        intent=intent,
        project_root=root,
    )
    reviewer_process = publication.capture_current_process_identity_v1(
        Path(__file__).resolve(strict=True)
    )
    parent_controller_process = _capture_parent_controller_process_v1()
    builder_process = intent["builder_process"]
    controller_process = review_attempt["controller_process"]
    if (
        parent_controller_process != controller_process
        or reviewer_process["process_id"] == parent_controller_process["process_id"]
        or reviewer_process["process_id"] == builder_process["process_id"]
        or (
            reviewer_process["process_id"],
            reviewer_process["process_created_at_utc"],
        )
        == (
            builder_process["process_id"],
            builder_process["process_created_at_utc"],
        )
        or reviewer_process["implementation_path"] == builder_process["implementation_path"]
        or reviewer_process["implementation_sha256"] == builder_process["implementation_sha256"]
        or (
            reviewer_process["process_id"],
            reviewer_process["process_created_at_utc"],
        )
        == (
            controller_process["process_id"],
            controller_process["process_created_at_utc"],
        )
    ):
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "fresh reviewer is not process/source independent or lacks exact parent custody"
        )

    review_started_at_utc = _utc_now()
    if authority_schema._utc(
        review_attempt["attempt_created_at_utc"],
        role="review-attempt creation",
    ) > authority_schema._utc(
        review_started_at_utc,
        role="review start",
    ):
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "fresh reviewer starts before its permanent attempt claim"
        )

    publication.verify_original_confirmatory_technical_intent_live_bindings_v1(
        intent=intent,
        source_inventory=source_inventory,
        project_root=root,
        reviewer_process=reviewer_process,
    )
    review_completed_at_utc = _utc_now()
    review = authority_schema.build_original_confirmatory_technical_authority_review_v1(
        intent=intent,
        review_started_at_utc=review_started_at_utc,
        review_completed_at_utc=review_completed_at_utc,
        reviewer_process=reviewer_process,
    )
    canonical_review = (
        authority_schema.canonical_original_confirmatory_technical_authority_review_v1(
            review,
            intent=intent,
        )
    )
    payload = authority_schema.canonical_json_line_bytes(canonical_review)
    output_sha256 = publication.publish_canonical_control_leaf_create_new_v1(
        output_path,
        payload,
    )
    if output_sha256 != authority_schema._sha256_bytes(payload):
        raise OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error(
            "immutable review receipt hash differs after retained-handle publication"
        )
    return {
        "schema_version": 1,
        "workflow": "original_confirmatory_technical_authority_review_producer_v1",
        "decision": "passed",
        "output": str(output_path),
        "output_sha256": output_sha256,
        "review_root_sha256": canonical_review["review_root_sha256"],
        "reviewer_process": canonical_review["reviewer_process"],
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "publication_performed": False,
        "selection_performed": False,
        "tuning_performed": False,
        "automatic_retry_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="original-confirmatory-technical-authority-review-producer-v1",
        description=(
            "Fresh-process outcome-blind review of one original-confirmatory "
            "technical-authority intent."
        ),
    )
    parser.add_argument("--intent-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used once by the controller's fresh child process."""

    arguments = _parser().parse_args(argv)
    try:
        result = produce_original_confirmatory_technical_authority_review_v1(
            intent_json=arguments.intent_json,
            output=arguments.output,
            project_root=arguments.project_root,
        )
    except Exception as exc:
        _fail(f"original-confirmatory review production failed: {type(exc).__name__}: {exc}")
    sys.stdout.buffer.write(authority_schema.canonical_json_line_bytes(result))
    sys.stdout.buffer.flush()
    return 0


__all__ = [
    "OriginalConfirmatoryTechnicalAuthorityReviewProducerV1Error",
    "main",
    "produce_original_confirmatory_technical_authority_review_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
