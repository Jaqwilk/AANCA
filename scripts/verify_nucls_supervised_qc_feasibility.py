"""Verify whether official NuCLS single-rater assets support paired pre/post QC."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")'))


def _report(result: dict[str, Any]) -> str:
    counts = result["database_counts"]
    return "\n".join(
        [
            "# NuCLS supervised-QC pairing feasibility",
            "",
            "**Decision:** paired natural pre/post class-label evaluation is unavailable.",
            "",
            "## What the official release contains",
            "",
            (
                f"The official raw SQLite database contains `{counts['fov_rows']}` FOV rows, "
                f"`{counts['annotation_element_rows']}` annotation-element rows and "
                f"`{counts['preapproved_fovs']}` preapproved/QC FOVs. The official public page "
                "describes 2,168 uncorrected FOVs and 1,744 corrected FOVs. These are quality "
                "tiers of different FOVs, not two releases of identical nucleus instances."
            ),
            "",
            "## Why the prospective pairing cannot run",
            "",
            (
                "The database has one class field (`group`) per stable annotation element and "
                "no previous-label, replacement-label, revision-history or paired-state table. "
                "Repeated element IDs only arise where the same geometry intersects multiple "
                "FOV records; no stable element ID carries two distinct class labels."
            ),
            (
                f"There are `{counts['correction_prefixed_elements']}` correction-prefixed final "
                "annotations, but their former labels are not retained. Treating the prefix as "
                "an error outcome would expose the final QC state to the auditor and would be "
                "circular."
            ),
            "",
            "## Consequence",
            "",
            (
                "AANCA cannot be honestly tested here on annotations later changed during QC. "
                "Comparing corrected and uncorrected cohorts would confound annotation quality "
                "with different images, patients and class distributions. The pre-registered "
                "natural-QC action therefore remains `retain_uncorrected`; no pathologist-error "
                "claim is available."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/raw/nucls_single_rater/SingleRater_2020-04-05.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/nucls_supervised_qc_feasibility/results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/nucls_supervised_qc_feasibility.md"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    database = (root / args.database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database)
    try:
        tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        )
        expected_tables = ("annotation_docs", "annotation_elements", "fov_meta")
        if tables != expected_tables:
            raise RuntimeError("official NuCLS SQLite schema differs from the inspected release")
        columns = {table: _table_columns(connection, table) for table in tables}
        fov_rows, unique_fov_names, preapproved, corrected = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT fovname), "
            "SUM(is_preapproved_fov), SUM(is_corrected_fov) FROM fov_meta"
        ).fetchone()
        element_rows, unique_elements = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT element_girder_id) FROM annotation_elements"
        ).fetchone()
        multi_group_element_ids = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT element_girder_id FROM annotation_elements "
                "GROUP BY element_girder_id HAVING COUNT(DISTINCT `group`) > 1)"
            ).fetchone()[0]
        )
        repeated_element_ids = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT element_girder_id FROM annotation_elements "
                "GROUP BY element_girder_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )
        repeated_across_fovs = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT element_girder_id FROM annotation_elements "
                "GROUP BY element_girder_id HAVING COUNT(DISTINCT fov_id) > 1)"
            ).fetchone()[0]
        )
        correction_prefixed = int(
            connection.execute(
                "SELECT COUNT(*) FROM annotation_elements WHERE `group` LIKE 'correction_%'"
            ).fetchone()[0]
        )
        qc_counts = {
            str(label): int(count)
            for label, count in connection.execute(
                "SELECT QC_label, COUNT(*) FROM fov_meta GROUP BY QC_label ORDER BY QC_label"
            )
        }
    finally:
        connection.close()
    forbidden_state_fields = {
        "old_label",
        "previous_label",
        "pre_qc_label",
        "post_qc_label",
        "replacement_label",
        "reference_label",
    }
    all_columns = {value.lower() for values in columns.values() for value in values}
    if forbidden_state_fields.intersection(all_columns):
        raise RuntimeError("NuCLS schema unexpectedly exposes a paired label-state field")
    if multi_group_element_ids != 0 or repeated_element_ids != repeated_across_fovs:
        raise RuntimeError("NuCLS repeated element identity cannot be explained by FOV overlap")
    result: dict[str, Any] = {
        "schema_version": 1,
        "study_id": "nucls_supervised_qc_prospective_v1",
        "protocol_sha256": sha256_file(root / "AANCA_NUCLS_SUPERVISED_QC_PROTOCOL.md"),
        "config_sha256": sha256_file(root / "configs/nucls_supervised_qc_prospective.yaml"),
        "official_database": {
            "name": database.name,
            "bytes": database.stat().st_size,
            "sha256": sha256_file(database),
            "source_file_id": "1bp6qyOYNYYHjeqLiNczjNZzUXFd9-qrb",
        },
        "official_public_release_counts": {
            "uncorrected_fovs": 2168,
            "uncorrected_nuclei": 65568,
            "corrected_fovs": 1744,
            "corrected_nuclei": 59485,
        },
        "database_counts": {
            "fov_rows": int(fov_rows),
            "unique_fov_names": int(unique_fov_names),
            "preapproved_fovs": int(preapproved),
            "corrected_flag_fovs": int(corrected),
            "annotation_element_rows": int(element_rows),
            "unique_annotation_element_ids": int(unique_elements),
            "repeated_element_ids": repeated_element_ids,
            "repeated_element_ids_across_fovs": repeated_across_fovs,
            "stable_element_ids_with_multiple_class_labels": multi_group_element_ids,
            "correction_prefixed_elements": correction_prefixed,
            "qc_label_counts": qc_counts,
        },
        "tables": {table: list(values) for table, values in columns.items()},
        "paired_fov_release_available": False,
        "paired_nucleus_pre_post_label_available": False,
        "former_label_for_correction_prefixed_element_available": False,
        "prospective_evaluation_status": "unavailable",
        "unavailable_reason": (
            "official corrected and uncorrected single-rater releases are distinct FOV quality "
            "tiers and the raw database preserves no paired pre/post class label per nucleus"
        ),
        "failure_action": "retain_uncorrected",
        "natural_error_detection_evaluated": False,
        "pathologist_error_detection_proven": False,
        "source_annotations_modified": False,
    }
    output = (root / args.output).resolve()
    report = (root / args.report).resolve()
    atomic_write_json(output, result)
    atomic_write_text(report, _report(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
