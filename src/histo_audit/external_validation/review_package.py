"""Build deterministic, blinded expert-review packages from ranked annotations."""

from __future__ import annotations

import csv
import html
import io
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from histo_audit.utils.run_tracking import atomic_write_bytes, atomic_write_json, atomic_write_text

DEFAULT_REVIEW_OPTIONS: tuple[str, ...] = (
    "annotation_supported",
    "probably_inconsistent",
    "ambiguous",
    "insufficient_context",
    "exclude_technical_reason",
)

_ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "full_patch": ("full_patch_path", "source_patch_path", "source_image_path", "patch_path"),
    "target_crop": ("target_crop_path", "crop_path", "nucleus_crop_path"),
    "target_contour": ("target_contour_path", "contour_path", "overlay_path"),
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class ReviewPackageResult:
    """Paths and exact cohort counts for a generated blinded package."""

    package_directory: Path
    review_items_csv: Path
    response_template_csv: Path
    review_html: Path
    response_schema_json: Path
    package_metadata_json: Path
    private_unblinding_key_csv: Path
    top_count: int
    random_count: int
    total_count: int
    excluded_without_assets: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible package evidence."""

        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


TableInput = str | Path | pd.DataFrame | Sequence[Mapping[str, Any]]


def _read_table(table: TableInput, *, role: str) -> tuple[pd.DataFrame, Path]:
    if isinstance(table, pd.DataFrame):
        return table.copy(), Path.cwd()
    if isinstance(table, (str, Path)):
        source = Path(table)
        if not source.is_file():
            raise FileNotFoundError(f"{role} table does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(source)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        elif suffix == ".json":
            frame = pd.read_json(source)
        else:
            raise ValueError(f"unsupported {role} table format: {suffix}")
        return frame, source.resolve().parent
    return pd.DataFrame([dict(row) for row in table]), Path.cwd()


def _validate_identifier_column(frame: pd.DataFrame, column: str, *, role: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{role} table is missing identifier column {column!r}")
    if frame.empty:
        raise ValueError(f"{role} table cannot be empty")
    values = frame[column]
    if values.isna().any() or (values.astype(str).str.strip() == "").any():
        raise ValueError(f"{role} table contains empty identifiers")
    if values.astype(str).duplicated().any():
        raise ValueError(f"{role} table contains duplicate identifiers")


def _resolve_asset_columns(
    manifest: pd.DataFrame,
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    if supplied is not None:
        columns = {str(role): str(column) for role, column in supplied.items()}
        if not columns:
            raise ValueError("asset_columns cannot be empty")
        missing = set(columns.values()).difference(manifest.columns)
        if missing:
            raise ValueError(f"manifest is missing configured asset columns: {sorted(missing)!r}")
    else:
        columns = {}
        for role, aliases in _ASSET_ALIASES.items():
            match = next((alias for alias in aliases if alias in manifest.columns), None)
            if match is not None:
                columns[role] = match
    missing_roles = set(_ASSET_ALIASES).difference(columns)
    if missing_roles:
        raise ValueError(
            "every review item requires full patch, target crop, and exact target contour; "
            f"missing asset roles: {sorted(missing_roles)!r}"
        )
    return columns


def _resolve_asset(value: Any, base: Path) -> Path | None:
    if value is None or (not isinstance(value, (str, Path)) and pd.isna(value)):
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    path = Path(rendered)
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    if not resolved.is_file() or resolved.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    return resolved


def _validate_display_columns(columns: Sequence[str], available: pd.Index[str]) -> tuple[str, ...]:
    forbidden_fragments = (
        "pre_corruption",
        "suggest",
        "prediction",
        "risk",
        "score",
        "source",
        "sample_id",
        "group_id",
        "patch_id",
        "instance_id",
        "corruption",
    )
    result = tuple(str(column) for column in columns)
    missing = set(result).difference(available)
    if missing:
        raise ValueError(f"manifest is missing display columns: {sorted(missing)!r}")
    unsafe = [
        column
        for column in result
        if any(fragment in column.casefold() for fragment in forbidden_fragments)
    ]
    if unsafe:
        raise ValueError(f"display columns would break blinding: {unsafe!r}")
    return result


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _safe_value(value: Any) -> str:
    if value is None or (not isinstance(value, (str, Path)) and pd.isna(value)):
        return ""
    return str(value)


def _response_schema(review_ids: Sequence[str], options: tuple[str, ...]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Blinded nucleus-annotation expert review response",
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["review_id", "reviewer_id", "response"],
            "properties": {
                "review_id": {"type": "string", "enum": list(review_ids)},
                "reviewer_id": {"type": "string", "minLength": 1},
                "response": {"type": "string", "enum": list(options)},
                "notes": {"type": "string"},
            },
        },
        "x-response-policy": (
            "The generated response template is blank. Only genuine reviewer responses may be added."
        ),
    }


def _review_html(
    rows: Sequence[Mapping[str, Any]],
    display_columns: tuple[str, ...],
    asset_roles: tuple[str, ...],
    options: tuple[str, ...],
) -> str:
    cards: list[str] = []
    for row in rows:
        review_id = html.escape(str(row["review_id"]))
        details = "".join(
            f"<dt>{html.escape(column.replace('_', ' ').title())}</dt>"
            f"<dd>{html.escape(str(row[column]))}</dd>"
            for column in display_columns
        )
        images = "".join(
            (
                f'<figure><img loading="lazy" src="{html.escape(str(row[f"asset_{role}"]))}" '
                f'alt="{html.escape(role.replace("_", " "))} for {review_id}">'
                f"<figcaption>{html.escape(role.replace('_', ' ').title())}</figcaption></figure>"
            )
            for role in asset_roles
            if row.get(f"asset_{role}")
        )
        option_controls = "".join(
            f'<label><input type="radio" name="response_{review_id}" '
            f'value="{html.escape(option)}"> {html.escape(option.replace("_", " "))}</label>'
            for option in options
        )
        cards.append(
            f'<article class="review-card" id="{review_id}"><h2>{review_id}</h2>'
            f'<dl>{details}</dl><div class="images">{images}</div>'
            f"<fieldset><legend>Expert response (not pre-filled)</legend>{option_controls}</fieldset>"
            "</article>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blinded nucleus annotation review</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:1rem;background:#f5f5f5;color:#202020}
.notice{border-left:5px solid #8b4513;background:#fff8e8;padding:1rem}.review-card{background:white;margin:1.5rem 0;padding:1rem;border-radius:.4rem}
.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}.images img{max-width:100%;height:auto;border:1px solid #bbb}
dt{font-weight:700}dd{margin:0 0 .5rem}fieldset label{display:block;margin:.45rem 0}
</style></head><body>
<h1>Blinded nucleus annotation review</h1>
<p class="notice">Research annotation-quality review only. This is not a diagnostic system. Items are deterministically mixed and anonymised; ranking source and model suggestions are hidden. Record only genuine expert responses in the separate response template.</p>
"""
        + "\n".join(cards)
        + "\n</body></html>\n"
    )


def build_blinded_review_package(
    manifest: TableInput,
    ranking: TableInput,
    output_directory: str | Path,
    *,
    top_count: int,
    random_count: int,
    seed: int,
    sample_id_column: str = "sample_id",
    score_column: str = "risk_score",
    observed_label_column: str = "observed_label",
    asset_columns: Mapping[str, str] | None = None,
    display_columns: Sequence[str] | None = None,
    review_options: tuple[str, ...] = DEFAULT_REVIEW_OPTIONS,
    private_unblinding_key_path: str | Path | None = None,
    study_outcome_eligible: bool = False,
    eligibility_evidence: Mapping[str, Any] | None = None,
) -> ReviewPackageResult:
    """Select exact disjoint top/random cohorts and build blinded static artifacts.

    The reviewer-facing directory never contains original sample IDs, ranking
    scores, selection cohorts, source paths, model suggestions, or
    ``pre_corruption_label``.  A private linkage key is written as a sibling by
    default and must be withheld until responses are locked.
    """

    if top_count <= 0 or random_count <= 0:
        raise ValueError("top_count and random_count must both be positive")
    if not review_options or len(set(review_options)) != len(review_options):
        raise ValueError("review_options must be a non-empty unique tuple")
    if study_outcome_eligible and eligibility_evidence is None:
        raise ValueError("stage-eligible packages require verified eligibility evidence")
    if not study_outcome_eligible and eligibility_evidence is not None:
        raise ValueError("eligibility evidence cannot be attached to a non-stage package")
    destination = Path(output_directory).resolve()
    private_key = (
        Path(private_unblinding_key_path).resolve()
        if private_unblinding_key_path is not None
        else destination.parent / f"{destination.name}.private_unblinding_key.csv"
    )
    if destination.exists():
        raise FileExistsError(
            f"review package already exists and will not be overwritten: {destination}"
        )
    if private_key.exists():
        raise FileExistsError(f"private unblinding key already exists: {private_key}")
    if private_key.is_relative_to(destination):
        raise ValueError("private unblinding key must remain outside the reviewer-facing package")

    manifest_frame, manifest_base = _read_table(manifest, role="manifest")
    ranking_frame, _ = _read_table(ranking, role="ranking")
    _validate_identifier_column(manifest_frame, sample_id_column, role="manifest")
    _validate_identifier_column(ranking_frame, sample_id_column, role="ranking")
    if score_column not in ranking_frame.columns:
        raise ValueError(f"ranking table is missing score column {score_column!r}")
    ranking_scores = pd.to_numeric(ranking_frame[score_column], errors="coerce")
    if ranking_scores.isna().any() or not np.isfinite(ranking_scores.to_numpy()).all():
        raise ValueError("ranking scores must all be finite numeric values")
    ranking_frame = ranking_frame.copy()
    ranking_frame[score_column] = ranking_scores.astype(float)
    if observed_label_column not in manifest_frame.columns:
        raise ValueError(
            f"manifest is missing the reviewer-visible observed label {observed_label_column!r}"
        )
    resolved_assets = _resolve_asset_columns(manifest_frame, asset_columns)
    shown_columns = _validate_display_columns(
        display_columns or (observed_label_column,), manifest_frame.columns
    )

    manifest_frame = manifest_frame.copy()
    manifest_frame[sample_id_column] = manifest_frame[sample_id_column].astype(str)
    ranking_frame[sample_id_column] = ranking_frame[sample_id_column].astype(str)
    joined = ranking_frame[[sample_id_column, score_column]].merge(
        manifest_frame,
        on=sample_id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if joined.empty:
        raise ValueError("ranking and manifest have no common sample IDs")
    joined = joined.sort_values(
        [score_column, sample_id_column], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    joined["__rank_position"] = np.arange(1, len(joined) + 1)

    asset_maps: list[dict[str, Path]] = []
    eligible: list[bool] = []
    for _, row in joined.iterrows():
        paths: dict[str, Path] = {}
        for role, column in resolved_assets.items():
            path = _resolve_asset(row[column], manifest_base)
            if path is not None:
                paths[role] = path
        asset_maps.append(paths)
        eligible.append(set(paths) == set(resolved_assets))
    joined["__assets"] = pd.Series(asset_maps, index=joined.index, dtype=object)
    eligible_frame = joined[np.asarray(eligible, dtype=bool)].reset_index(drop=True)
    excluded_without_assets = len(joined) - len(eligible_frame)
    required = top_count + random_count
    if len(eligible_frame) < required:
        raise ValueError(
            f"only {len(eligible_frame)} ranked manifest rows have all required displayable "
            "full-patch, target-crop, and exact-contour assets; "
            f"{required} are required"
        )

    top = eligible_frame.iloc[:top_count].copy()
    random_pool = eligible_frame.iloc[top_count:].copy()
    if len(random_pool) < random_count:
        raise ValueError("insufficient disjoint non-top rows for the requested random cohort")
    rng = np.random.default_rng(seed)
    random_positions = rng.choice(len(random_pool), size=random_count, replace=False)
    random_selection = random_pool.iloc[np.sort(random_positions)].copy()
    top["__selection_source"] = "top_ranked"
    random_selection["__selection_source"] = "random"
    selected = pd.concat((top, random_selection), ignore_index=True)
    mixed = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    mixed["__review_id"] = [f"review-{index + 1:04d}" for index in range(len(mixed))]

    destination.parent.mkdir(parents=True, exist_ok=True)
    private_key.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staged_package = staging_root / "reviewer_package"
    staged_assets = staged_package / "assets"
    staged_assets.mkdir(parents=True)
    staged_key = staging_root / "private_unblinding_key.csv"
    published_destination = False
    published_key = False
    try:
        item_rows: list[dict[str, Any]] = []
        key_rows: list[dict[str, Any]] = []
        asset_roles = tuple(resolved_assets)
        for display_index, (_, row) in enumerate(mixed.iterrows(), start=1):
            review_id = str(row["__review_id"])
            item: dict[str, Any] = {"review_id": review_id, "display_order": display_index}
            item.update({column: _safe_value(row[column]) for column in shown_columns})
            available_assets = cast_asset_map(row["__assets"])
            for role in asset_roles:
                source = available_assets.get(role)
                if source is None:
                    item[f"asset_{role}"] = ""
                    continue
                filename = f"{review_id}_{role}{source.suffix.lower()}"
                relative = Path("assets") / filename
                atomic_write_bytes(staged_package / relative, source.read_bytes())
                item[f"asset_{role}"] = relative.as_posix()
            item_rows.append(item)
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_id": str(row[sample_id_column]),
                    "selection_source": str(row["__selection_source"]),
                    "rank_position_among_joined": int(row["__rank_position"]),
                    "ranking_score": float(row[score_column]),
                }
            )

        item_fields = [
            "review_id",
            "display_order",
            *shown_columns,
            *(f"asset_{role}" for role in asset_roles),
        ]
        review_ids = [str(row["review_id"]) for row in item_rows]
        response_rows = [
            {"review_id": review_id, "reviewer_id": "", "response": "", "notes": ""}
            for review_id in review_ids
        ]
        atomic_write_text(staged_package / "review_items.csv", _csv_text(item_fields, item_rows))
        atomic_write_text(
            staged_package / "response_template.csv",
            _csv_text(("review_id", "reviewer_id", "response", "notes"), response_rows),
        )
        atomic_write_json(
            staged_package / "response_schema.json", _response_schema(review_ids, review_options)
        )
        atomic_write_text(
            staged_package / "review.html",
            _review_html(item_rows, shown_columns, asset_roles, review_options),
        )
        atomic_write_json(
            staged_package / "package_metadata.json",
            {
                "schema_version": 1,
                "purpose": "blinded expert review of potentially inconsistent annotations",
                "non_diagnostic": True,
                "seed": seed,
                "top_count": top_count,
                "random_count": random_count,
                "total_count": required,
                "excluded_without_displayable_assets": excluded_without_assets,
                "asset_roles": list(asset_roles),
                "review_options": list(review_options),
                "blinding": {
                    "sample_ids_anonymised": True,
                    "selection_source_hidden": True,
                    "ranking_scores_hidden": True,
                    "recommendation_output_hidden": True,
                    "hidden_reference_labels_omitted": True,
                    "source_paths_hidden": True,
                    "private_key_stored_outside_package": True,
                },
                "response_state": "blank_template_only; no expert responses generated",
                "study_outcome_eligible": study_outcome_eligible,
                "eligibility_evidence": (
                    dict(eligibility_evidence) if eligibility_evidence is not None else None
                ),
            },
        )
        atomic_write_text(
            staged_key,
            _csv_text(
                (
                    "review_id",
                    "sample_id",
                    "selection_source",
                    "rank_position_among_joined",
                    "ranking_score",
                ),
                key_rows,
            ),
        )

        # Validate the complete staged package and linkage before either public
        # destination is created.  The CLI performs a second independent check
        # and rolls both paths back if that check ever fails.
        from .validation import validate_blinded_review_package

        staged_validation = validate_blinded_review_package(
            staged_package,
            private_unblinding_key_path=staged_key,
        )
        if not staged_validation.valid or not staged_validation.private_linkage_validated:
            raise RuntimeError(
                f"staged review package failed validation: {staged_validation.as_dict()}"
            )
        os.replace(staged_package, destination)
        published_destination = True
        os.replace(staged_key, private_key)
        published_key = True
    except BaseException:
        if published_destination and destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        if published_key:
            private_key.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return ReviewPackageResult(
        package_directory=destination,
        review_items_csv=destination / "review_items.csv",
        response_template_csv=destination / "response_template.csv",
        review_html=destination / "review.html",
        response_schema_json=destination / "response_schema.json",
        package_metadata_json=destination / "package_metadata.json",
        private_unblinding_key_csv=private_key,
        top_count=top_count,
        random_count=random_count,
        total_count=required,
        excluded_without_assets=excluded_without_assets,
    )


def cast_asset_map(value: Any) -> dict[str, Path]:
    """Validate the internal resolved-asset value kept in a pandas object column."""

    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(path, Path) for key, path in value.items()
    ):
        raise RuntimeError("internal resolved asset map is malformed")
    return value


__all__ = [
    "DEFAULT_REVIEW_OPTIONS",
    "ReviewPackageResult",
    "build_blinded_review_package",
]
