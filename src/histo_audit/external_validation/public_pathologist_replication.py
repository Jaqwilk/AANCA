"""Frozen public replication on natural independent-pathologist disagreements.

The workflow deliberately terminates between input snapshot creation, reference-blind
scoring, and hidden-reference evaluation. A disagreement is a review outcome, not proof
that a pathologist was wrong and not a diagnostic endpoint.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from PIL import Image

from histo_audit.auditing.strategies import group_safe_audit_scores
from histo_audit.auditing.two_queue import draw_matched_random_comparator
from histo_audit.cross_validation.oof import (
    MultinomialLogisticRegression,
    make_group_stratified_fold_plan,
)
from histo_audit.external_validation.nucls_independent_pathologist import (
    InputPreparedData,
    ScoredInputData,
    _array_sha256,
    _canonical_frame_sha256,
    _fixed_crop,
    _semantic_sha256,
    _source_inventory,
    select_exact_comparator_capable_queue,
)
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.statistics.review import average_precision, rank_indices
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file

EXPECTED_CONFIG_SHA256 = "1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321"
CONFIG_RELATIVE_PATH = Path("configs/public_independent_pathologist_replication.yaml")
RIVA_CLASS_ORDER = (
    "non_lesion",
    "low_grade_or_equivocal",
    "high_grade_or_malignant",
)
MIDOGPP_CLASS_ORDER = ("mitotic_figure", "not_mitotic_figure")
REFERENCE_STATUSES = (
    "consensus_agree",
    "consensus_disagree",
    "ambiguous",
    "insufficient_reference",
)
SNAPSHOT_REQUIRED_COLUMNS = (
    "sample_id",
    "source_annotation_id",
    "input_annotator",
    "image_file",
    "image_id",
    "group_id",
    "observed_raw_label",
    "observed_label",
    "observed_class",
    "centre_x",
    "centre_y",
)
SNAPSHOT_FORBIDDEN_FRAGMENTS = (
    "reference",
    "consensus",
    "cluster",
    "category_id",
    "adjudicat",
    "other_expert",
    "labels_json",
)


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    """Authenticated immutable settings for one public dataset."""

    name: str
    class_order: tuple[str, ...]
    rotations: tuple[str, ...]
    settings: Mapping[str, Any]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _md5_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_public_replication_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load the config only when both its sidecar and compiled authority match."""

    root = Path(repository_root).resolve()
    path = root / CONFIG_RELATIVE_PATH
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().casefold()
    if digest != expected or digest != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("public pathologist replication config differs from frozen authority")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("public replication config must be a mapping")
    protocol = root / str(payload["protocol"])
    candidate = root / str(cast(Mapping[str, Any], payload["candidate"])["record"])
    if not protocol.is_file() or not candidate.is_file():
        raise FileNotFoundError("frozen protocol or selected candidate is missing")
    candidate_payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if (
        not isinstance(candidate_payload, dict)
        or candidate_payload.get("candidate_sha256")
        != cast(Mapping[str, Any], payload["candidate"])["sha256"]
    ):
        raise RuntimeError("selected AANCA candidate differs from frozen authority")
    return payload, digest


def dataset_definition(config: Mapping[str, Any], dataset: str) -> DatasetDefinition:
    datasets = cast(Mapping[str, Any], config["datasets"])
    if dataset not in datasets:
        raise ValueError(f"unknown frozen dataset: {dataset}")
    settings = cast(Mapping[str, Any], datasets[dataset])
    return DatasetDefinition(
        name=dataset,
        class_order=tuple(str(value) for value in settings["class_order"]),
        rotations=tuple(str(value) for value in settings["rotations"]),
        settings=settings,
    )


def map_riva_class(value: Any) -> tuple[int, str]:
    """Map one released RIVA label to the frozen three-class audit space."""

    label = str(value).strip().upper()
    if label in {"NILM", "INFL", "ENDO", "SIN LESION"}:
        name = "non_lesion"
    elif label in {"ASCUS", "LSIL"}:
        name = "low_grade_or_equivocal"
    elif label in {"ASCH", "HSIL", "SCC", "CA"}:
        name = "high_grade_or_malignant"
    else:
        raise ValueError(f"unmapped RIVA label: {value!r}")
    return RIVA_CLASS_ORDER.index(name), name


def map_midogpp_class(value: Any) -> tuple[int, str]:
    """Map one released MIDOG++ expert label to the frozen binary audit space."""

    label = int(value)
    if label == 1:
        return 0, "mitotic_figure"
    if label == 2:
        return 1, "not_mitotic_figure"
    raise ValueError(f"unmapped MIDOG++ label: {value!r}")


def riva_smear_group(image_stem: str) -> str:
    """Remove the final field number from an official RIVA image stem."""

    parts = str(image_stem).removesuffix(".png").split("_")
    if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2].isdigit():
        raise ValueError(f"RIVA image identity lacks smear/field suffix: {image_stem}")
    return "_".join(parts[:-1])


def _snapshot_columns_are_input_only(frame: pd.DataFrame) -> bool:
    columns = {str(column) for column in frame.columns}
    if not set(SNAPSHOT_REQUIRED_COLUMNS).issubset(columns):
        return False
    lowered = tuple(column.casefold() for column in columns)
    return not any(
        fragment in column for column in lowered for fragment in SNAPSHOT_FORBIDDEN_FRAGMENTS
    )


def select_midogpp_images(
    metadata: Sequence[Mapping[str, Any]],
    available_ids: set[int],
    *,
    salt: str,
    per_tumor_type: int,
) -> tuple[dict[str, Any], ...]:
    """Select the frozen label-independent hash sample, stratified by tumor type."""

    by_tumor: defaultdict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for raw in metadata:
        image_id = int(raw["id"])
        if image_id not in available_ids:
            continue
        record = dict(raw)
        digest = hashlib.sha256(f"{salt}|{record['file_name']}".encode()).hexdigest()
        by_tumor[str(record["tumor_type"])].append((digest, record))
    selected: list[dict[str, Any]] = []
    for tumor_type in sorted(by_tumor):
        candidates = sorted(by_tumor[tumor_type], key=lambda item: (item[0], item[1]["id"]))
        if len(candidates) < per_tumor_type:
            raise RuntimeError(f"MIDOG++ tumor type lacks {per_tumor_type} available images")
        for digest, record in candidates[:per_tumor_type]:
            selected.append({**record, "selection_sha256": digest})
    return tuple(sorted(selected, key=lambda item: int(item["id"])))


def _authenticate_source(path: Path, *, sha256: str | None = None, md5: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required public source is missing: {path}")
    if sha256 is not None and sha256_file(path) != sha256.casefold():
        raise RuntimeError(f"SHA-256 mismatch for public source: {path}")
    if md5 is not None and _md5_file(path) != md5.casefold():
        raise RuntimeError(f"MD5 mismatch for public source: {path}")


def _snapshot_manifest_path(definition: DatasetDefinition, root: Path) -> Path:
    return root / str(definition.settings["input_snapshot_directory"]) / "snapshot_manifest.json"


def _persist_snapshot_set(
    destination: Path,
    *,
    dataset: str,
    config_sha256: str,
    frames: Mapping[str, pd.DataFrame],
    source_authorities: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Any] = {}
    for rotation, frame in frames.items():
        ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        if ordered["sample_id"].duplicated().any() or not _snapshot_columns_are_input_only(ordered):
            raise RuntimeError(f"{dataset}/{rotation} snapshot is not input-only and unique")
        path = destination / f"{rotation}.csv"
        _write_csv(path, ordered)
        snapshots[rotation] = {
            "path": path.name,
            "row_count": len(ordered),
            "group_count": int(ordered["group_id"].nunique()),
            "class_counts": {
                str(key): int(value)
                for key, value in ordered["observed_class"].value_counts().sort_index().items()
            },
            "sha256": sha256_file(path),
            "semantic_sha256": _canonical_frame_sha256(ordered),
            "columns": list(ordered.columns),
            "input_only": True,
        }
    manifest = {
        "schema_version": 1,
        "study_id": "public_independent_pathologist_replication_v1",
        "dataset": dataset,
        "config_sha256": config_sha256,
        "process_boundary": "prepare_process_must_terminate_before_scoring",
        "hidden_reference_used": False,
        "source_authorities": list(source_authorities),
        "selection": list(selection or ()),
        "snapshots": snapshots,
    }
    atomic_write_json(destination / "snapshot_manifest.json", manifest)
    return manifest


def materialize_riva_input_snapshots(repository_root: str | Path) -> dict[str, Any]:
    """Emit four isolated observed-label snapshots, without cluster/reference fields."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, "riva")
    settings = definition.settings
    source = root / str(settings["annotations"])
    _authenticate_source(source, sha256=str(settings["annotations_sha256"]))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RIVA annotation source must be keyed by image stem")
    image_root = root / str(settings["images_directory"])
    frames: dict[str, pd.DataFrame] = {}
    for rotation in definition.rotations:
        records: list[dict[str, Any]] = []
        for image_stem, per_annotator in sorted(payload.items()):
            if not isinstance(per_annotator, dict):
                raise ValueError("RIVA per-image annotations must be a mapping")
            image_name = f"{image_stem}.png"
            image_path = (image_root / image_name).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"RIVA source image is missing: {image_name}")
            with Image.open(image_path) as opened:
                width, height = opened.size
            annotations = per_annotator.get(rotation, [])
            if not isinstance(annotations, list):
                raise ValueError("RIVA annotator payload must be a list")
            for position, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    raise ValueError("RIVA annotation must be a mapping")
                raw_label = annotation["keypointlabels"]
                if isinstance(raw_label, list):
                    if len(raw_label) != 1:
                        raise ValueError("RIVA annotation must contain exactly one label")
                    raw_label = raw_label[0]
                observed_label, observed_class = map_riva_class(raw_label)
                x_percent = float(annotation["x"])
                y_percent = float(annotation["y"])
                if not (math.isfinite(x_percent) and math.isfinite(y_percent)):
                    raise ValueError("RIVA point coordinate must be finite")
                centre_x = min(width - 1, max(0, math.floor(x_percent * width / 100.0)))
                centre_y = min(height - 1, max(0, math.floor(y_percent * height / 100.0)))
                source_id = f"{image_stem}:{rotation}:{position}"
                sample_id = f"riva-{rotation}-{_semantic_sha256(source_id)[:24]}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "source_annotation_id": source_id,
                        "input_annotator": rotation,
                        "image_file": image_path.relative_to(root).as_posix(),
                        "image_id": str(image_stem),
                        "group_id": riva_smear_group(str(image_stem)),
                        "observed_raw_label": str(raw_label),
                        "observed_label": observed_label,
                        "observed_class": observed_class,
                        "centre_x": centre_x,
                        "centre_y": centre_y,
                        "point_x_percent": x_percent,
                        "point_y_percent": y_percent,
                        "image_width": width,
                        "image_height": height,
                    }
                )
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            raise RuntimeError(f"RIVA snapshot is empty for {rotation}")
        if set(frame["observed_class"]) != set(definition.class_order):
            raise RuntimeError(f"RIVA snapshot lost a frozen class for {rotation}")
        if frame["group_id"].nunique() < int(settings["minimum_groups"]):
            raise RuntimeError(f"RIVA snapshot has too few smear groups for {rotation}")
        frames[rotation] = frame
    return _persist_snapshot_set(
        _snapshot_manifest_path(definition, root).parent,
        dataset="riva",
        config_sha256=config_sha,
        frames=frames,
        source_authorities=(
            {
                "path": source.relative_to(root).as_posix(),
                "sha256": sha256_file(source),
                "contains_all_annotators": True,
                "opened_only_in_prepare_process": True,
            },
        ),
    )


def _available_midogpp_ids(path: Path) -> set[int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        ids = {int(row["Slide"]) for row in reader}
    if len(ids) != 503:
        raise RuntimeError("MIDOG++ availability authority no longer contains 503 cases")
    return ids


def _load_midogpp_selection(
    root: Path, settings: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    annotations_path = root / str(settings["annotations"])
    availability_path = root / str(settings["availability_manifest"])
    _authenticate_source(
        annotations_path,
        sha256=str(settings["annotations_sha256"]),
        md5=str(settings["annotations_md5"]),
    )
    _authenticate_source(
        availability_path,
        sha256=str(settings["availability_manifest_sha256"]),
        md5=str(settings["availability_manifest_md5"]),
    )
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MIDOG++ JSON must contain a mapping")
    available = _available_midogpp_ids(availability_path)
    selected = select_midogpp_images(
        cast(Sequence[Mapping[str, Any]], payload["images"]),
        available,
        salt=str(settings["selection_salt"]),
        per_tumor_type=int(settings["images_per_tumor_type"]),
    )
    if len({str(item["tumor_type"]) for item in selected}) != int(
        settings["expected_tumor_type_count"]
    ) or len(selected) != int(settings["expected_selected_image_count"]):
        raise RuntimeError("MIDOG++ frozen selection size or stratum count changed")
    return payload, selected


def materialize_midogpp_input_snapshots(repository_root: str | Path) -> dict[str, Any]:
    """Emit two positional expert snapshots without final/adjudicator labels."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, "midogpp")
    settings = definition.settings
    payload, selected = _load_midogpp_selection(root, settings)
    selected_by_id = {int(item["id"]): item for item in selected}
    frames: dict[str, pd.DataFrame] = {}
    positions = cast(Mapping[str, Any], settings["label_positions"])
    for rotation in definition.rotations:
        position = int(positions[rotation])
        records: list[dict[str, Any]] = []
        for annotation in cast(Sequence[Mapping[str, Any]], payload["annotations"]):
            image_id = int(annotation["image_id"])
            if image_id not in selected_by_id:
                continue
            labels = list(cast(Sequence[Any], annotation["labels"]))
            if len(labels) < 2:
                raise RuntimeError("MIDOG++ primary candidate lacks two expert labels")
            observed_raw = int(labels[position])
            observed_label, observed_class = map_midogpp_class(observed_raw)
            bbox = [float(value) for value in cast(Sequence[Any], annotation["bbox"])]
            if len(bbox) != 4 or not np.isfinite(np.asarray(bbox)).all():
                raise ValueError("MIDOG++ bbox must contain four finite coordinates")
            centre_x = math.floor((bbox[0] + bbox[2]) / 2.0)
            centre_y = math.floor((bbox[1] + bbox[3]) / 2.0)
            image = selected_by_id[image_id]
            source_id = str(int(annotation["id"]))
            sample_id = f"midogpp-{rotation}-{_semantic_sha256(source_id)[:24]}"
            records.append(
                {
                    "sample_id": sample_id,
                    "source_annotation_id": source_id,
                    "input_annotator": rotation,
                    "image_file": (
                        Path(str(settings["images_directory"])) / str(image["file_name"])
                    ).as_posix(),
                    "image_id": str(image_id),
                    "group_id": str(image_id),
                    "observed_raw_label": str(observed_raw),
                    "observed_label": observed_label,
                    "observed_class": observed_class,
                    "centre_x": centre_x,
                    "centre_y": centre_y,
                    "bbox_xmin": bbox[0],
                    "bbox_ymin": bbox[1],
                    "bbox_xmax": bbox[2],
                    "bbox_ymax": bbox[3],
                    "tumor_type": str(image["tumor_type"]),
                }
            )
        frame = pd.DataFrame.from_records(records)
        if frame.empty or set(frame["observed_class"]) != set(definition.class_order):
            raise RuntimeError(f"MIDOG++ snapshot lacks rows/classes for {rotation}")
        if frame["group_id"].nunique() < int(settings["minimum_groups"]):
            raise RuntimeError(f"MIDOG++ snapshot has too few case groups for {rotation}")
        frames[rotation] = frame
    annotations_path = root / str(settings["annotations"])
    availability_path = root / str(settings["availability_manifest"])
    public_selection = [
        {
            "image_id": int(item["id"]),
            "file_name": str(item["file_name"]),
            "tumor_type": str(item["tumor_type"]),
            "selection_sha256": str(item["selection_sha256"]),
        }
        for item in selected
    ]
    return _persist_snapshot_set(
        _snapshot_manifest_path(definition, root).parent,
        dataset="midogpp",
        config_sha256=config_sha,
        frames=frames,
        source_authorities=(
            {
                "path": annotations_path.relative_to(root).as_posix(),
                "sha256": sha256_file(annotations_path),
                "contains_all_expert_labels": True,
                "opened_only_in_prepare_process": True,
            },
            {
                "path": availability_path.relative_to(root).as_posix(),
                "sha256": sha256_file(availability_path),
                "selection_uses_labels": False,
            },
        ),
        selection=public_selection,
    )


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_midogpp_selected_images(repository_root: str | Path) -> dict[str, Any]:
    """Download and checksum only the prospectively hash-selected 70 TIFF cases."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, "midogpp")
    settings = definition.settings
    _, selected = _load_midogpp_selection(root, settings)
    selected_names = {str(item["file_name"]): item for item in selected}
    collection = str(settings["figshare_collection"])
    articles: list[Mapping[str, Any]] = []
    for page in range(1, 20):
        query = urllib.parse.urlencode({"page_size": 100, "page": page})
        batch = cast(
            list[Mapping[str, Any]],
            _fetch_json(f"https://api.figshare.com/v2/collections/{collection}/articles?{query}"),
        )
        articles.extend(batch)
        if len(batch) < 100:
            break

    def load_article(article: Mapping[str, Any]) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], _fetch_json(str(article["url"])))

    authorities: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for article in executor.map(load_article, articles):
            for raw_file in cast(Sequence[Mapping[str, Any]], article.get("files", [])):
                name = str(raw_file["name"])
                if name in selected_names:
                    if name in authorities:
                        raise RuntimeError(f"duplicate MIDOG++ Figshare file authority: {name}")
                    authorities[name] = {
                        "article_id": int(article["id"]),
                        "file_id": int(raw_file["id"]),
                        "file_name": name,
                        "size_bytes": int(raw_file["size"]),
                        "download_url": str(raw_file["download_url"]),
                        "supplied_md5": str(raw_file["supplied_md5"]).casefold(),
                    }
    if set(authorities) != set(selected_names):
        missing = sorted(set(selected_names).difference(authorities))
        raise RuntimeError(f"Figshare did not expose every frozen MIDOG++ image: {missing}")
    required_bytes = sum(record["size_bytes"] for record in authorities.values())
    image_root = root / str(settings["images_directory"])
    image_root.mkdir(parents=True, exist_ok=True)
    missing_bytes = sum(
        record["size_bytes"]
        for name, record in authorities.items()
        if not (image_root / name).is_file()
    )
    if shutil.disk_usage(image_root).free < missing_bytes + 2 * 1024**3:
        raise RuntimeError("insufficient free disk for frozen MIDOG++ subset plus safety margin")
    downloaded: list[dict[str, Any]] = []
    for name in sorted(authorities):
        authority = authorities[name]
        destination = image_root / name
        if destination.is_file() and (
            destination.stat().st_size != authority["size_bytes"]
            or _md5_file(destination) != authority["supplied_md5"]
        ):
            raise RuntimeError(f"existing MIDOG++ image fails its Figshare authority: {name}")
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            try:
                with (
                    urllib.request.urlopen(authority["download_url"], timeout=120) as source,
                    temporary.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if (
                    temporary.stat().st_size != authority["size_bytes"]
                    or _md5_file(temporary) != authority["supplied_md5"]
                ):
                    raise RuntimeError(f"downloaded MIDOG++ image fails checksum: {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        downloaded.append(
            {
                **authority,
                "path": destination.relative_to(root).as_posix(),
                "local_md5": _md5_file(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "dataset": "midogpp",
        "config_sha256": config_sha,
        "selected_image_count": len(downloaded),
        "required_bytes": required_bytes,
        "all_figshare_md5_checks_passed": True,
        "files": downloaded,
    }
    atomic_write_json(image_root / "download_manifest.json", manifest)
    return manifest


def _load_authenticated_snapshot(
    root: Path,
    definition: DatasetDefinition,
    *,
    rotation: str,
    config_sha256: str,
) -> tuple[pd.DataFrame, Path, Mapping[str, Any]]:
    if rotation not in definition.rotations:
        raise ValueError(f"{rotation!r} is not a frozen {definition.name} rotation")
    manifest_path = _snapshot_manifest_path(definition, root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != definition.name
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("hidden_reference_used") is not False
    ):
        raise RuntimeError("input snapshot manifest fails its frozen information barrier")
    snapshot_record = cast(Mapping[str, Any], manifest["snapshots"][rotation])
    snapshot_path = manifest_path.parent / str(snapshot_record["path"])
    if sha256_file(snapshot_path) != snapshot_record["sha256"]:
        raise RuntimeError("input snapshot bytes differ from the prepare-stage seal")
    frame = pd.read_csv(snapshot_path)
    if not _snapshot_columns_are_input_only(frame):
        raise RuntimeError("score stage rejected a snapshot with reference-like fields")
    if (
        len(frame) != int(snapshot_record["row_count"])
        or frame["sample_id"].duplicated().any()
        or set(frame["input_annotator"].astype(str)) != {rotation}
    ):
        raise RuntimeError("input snapshot identity/count validation failed")
    return frame, snapshot_path, snapshot_record


def prepare_snapshot_pixels(
    repository_root: str | Path,
    *,
    dataset: str,
    rotation: str,
) -> InputPreparedData:
    """Read one isolated input snapshot and pixels without opening a reference authority."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, dataset)
    frame, snapshot_path, _ = _load_authenticated_snapshot(
        root, definition, rotation=rotation, config_sha256=config_sha
    )
    image_paths = sorted(
        {root / str(value) for value in frame["image_file"].astype(str)},
        key=lambda item: item.as_posix(),
    )
    if not image_paths or any(not path.is_file() for path in image_paths):
        raise FileNotFoundError(f"{dataset}/{rotation} source pixels are incomplete")
    crop_sizes = tuple(
        int(value) for value in cast(Mapping[str, Any], config["candidate"])["crop_sizes"]
    )
    if crop_sizes != (64, 128):
        raise RuntimeError("public replication requires the frozen 64+128 px representation")
    crops: dict[int, list[NDArray[np.uint8]]] = {size: [] for size in crop_sizes}
    records: list[dict[str, Any]] = []
    for image_file, local in frame.groupby("image_file", sort=True):
        image_path = root / str(image_file)
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        for _, row in local.iterrows():
            centre_x = int(row["centre_x"])
            centre_y = int(row["centre_y"])
            if not (0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]):
                raise RuntimeError(f"{dataset}/{rotation} crop centre is outside source pixels")
            records.append(cast(dict[str, Any], row.to_dict()))
            for size in crop_sizes:
                crops[size].append(
                    _fixed_crop(image, centre_x=centre_x, centre_y=centre_y, size=size)
                )
    manifest = pd.DataFrame.from_records(records)
    order = np.argsort(manifest["sample_id"].astype(str).to_numpy(), kind="stable")
    manifest = manifest.iloc[order].reset_index(drop=True)
    crop_arrays = {
        size: np.stack(values, axis=0)[order].astype(np.uint8, copy=False)
        for size, values in crops.items()
    }
    if set(manifest["observed_class"].astype(str)) != set(definition.class_order):
        raise RuntimeError(f"{dataset}/{rotation} pixel preparation lost a frozen class")
    inventory_paths: list[Path] = [snapshot_path, *image_paths]
    if dataset == "midogpp":
        download_manifest = (
            root / str(definition.settings["images_directory"]) / "download_manifest.json"
        )
        if not download_manifest.is_file():
            raise FileNotFoundError("MIDOG++ download manifest is missing")
        inventory_paths.append(download_manifest)
    inventory = _source_inventory(root, inventory_paths)
    return InputPreparedData(
        manifest=manifest,
        crops=crop_arrays,
        exclusions={},
        source_inventory=inventory,
        source_inventory_sha256=_semantic_sha256(inventory),
        manifest_sha256=_canonical_frame_sha256(manifest),
        crop_sha256={size: _array_sha256(array) for size, array in crop_arrays.items()},
    )


def extract_public_multiscale_embeddings(
    prepared: InputPreparedData,
    *,
    cache_directory: str | Path,
    study_id: str,
    dataset: str,
    rotation: str,
    device: str = "auto",
) -> tuple[NDArray[np.float32], dict[int, dict[str, Any]]]:
    """Extract/cache the frozen input-only ResNet-18 representation."""

    destination = Path(cache_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
    matrices: list[NDArray[np.float32]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for size in (64, 128):
        cache_path = destination / f"{rotation}_resnet18_context_{size}.npz"
        representation_id = f"{dataset}_{rotation}_resnet18_imagenet1k_v1_context_{size}px"
        eligibility = {
            "study_id": study_id,
            "dataset": dataset,
            "input_rotation": rotation,
            "eligible": True,
            "sample_count": len(sample_ids),
            "source_context_px": size,
            "source_crops_sha256": prepared.crop_sha256[size],
            "hidden_reference_loaded": False,
        }
        if cache_path.is_file():
            cached = load_embedding_cache(cache_path)
            cached.validate()
            recorded = cached.metadata.get("analysis_eligibility")
            if (
                cached.sample_ids.tolist() != sample_ids
                or cached.metadata.get("manifest_sha256") != prepared.manifest_sha256
                or cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256
                or cached.metadata.get("representation_id") != representation_id
                or not isinstance(recorded, Mapping)
                or recorded.get("source_crops_sha256") != prepared.crop_sha256[size]
                or recorded.get("hidden_reference_loaded") is not False
            ):
                raise RuntimeError("cached public replication embeddings fail provenance binding")
            matrix = np.asarray(cached.embeddings, dtype=np.float32)
            embedded_metadata = dict(cached.metadata)
        else:
            result = extract_resnet18_embeddings(
                prepared.crops[size],
                sample_ids,
                config=ResNet18EmbeddingConfig(
                    weight_identifier="IMAGENET1K_V1",
                    input_variant="rgb",
                    device=device,
                    batch_size=64,
                    minimum_batch_size=1,
                    use_amp=True,
                    output_dtype="float32",
                    allow_weight_download=False,
                ),
                cache_path=cache_path,
                manifest_sha256=prepared.manifest_sha256,
                raw_inventory_sha256=prepared.source_inventory_sha256,
                representation_id=representation_id,
                analysis_eligibility=eligibility,
            )
            result.validate()
            matrix = np.asarray(result.embeddings, dtype=np.float32)
            embedded_metadata = dict(result.metadata)
        if matrix.shape != (len(sample_ids), 512) or not np.isfinite(matrix).all():
            raise RuntimeError("public replication ResNet-18 embeddings are invalid")
        matrices.append(matrix)
        metadata[size] = embedded_metadata
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False), metadata


def score_public_input(
    prepared: InputPreparedData,
    embeddings: NDArray[np.generic],
    *,
    config: Mapping[str, Any],
    class_order: Sequence[str],
) -> ScoredInputData:
    """Compute fully group-safe OOF risks in a dynamic frozen class space."""

    candidate = cast(Mapping[str, Any], config["candidate"])
    classes = tuple(range(len(class_order)))
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.shape != (len(prepared.manifest), 1024) or not np.isfinite(matrix).all():
        raise ValueError("public replication embeddings do not align with input snapshot")
    labels = prepared.manifest["observed_label"].to_numpy(dtype=np.int64)
    groups = prepared.manifest["group_id"].astype(str).tolist()
    sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
    if set(labels) != set(classes):
        raise RuntimeError("public replication input lacks one or more frozen classes")
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=int(candidate["oof_folds"]),
        class_order=classes,
        seed=int(candidate["split_seed"]),
    )
    probabilities = np.full((len(labels), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    training_groups_by_fold: dict[int, tuple[str, ...]] = {}
    fold_evidence: list[dict[str, Any]] = []
    converged: list[bool] = []
    for fold in plan.folds:
        if set(fold.training_groups).intersection(fold.held_out_groups):
            raise RuntimeError("group crossed a public-replication OOF boundary")
        model = MultinomialLogisticRegression(
            class_order=classes,
            l2=float(candidate["audit_l2"]),
            max_iter=int(candidate["audit_max_iter"]),
            class_weight_balanced=bool(candidate["audit_class_weight_balanced"]),
        ).fit(matrix[fold.train_indices], labels[fold.train_indices])
        probabilities[fold.holdout_indices] = model.predict_proba(matrix[fold.holdout_indices])
        fold_ids[fold.holdout_indices] = fold.fold_id
        training_groups_by_fold[fold.fold_id] = fold.training_groups
        converged.append(bool(model.converged_))
        fold_evidence.append(
            {
                "fold_id": fold.fold_id,
                "training_groups": list(fold.training_groups),
                "held_out_groups": list(fold.held_out_groups),
                "held_out_sample_ids": [sample_ids[index] for index in fold.holdout_indices],
                "training_count": len(fold.train_indices),
                "holdout_count": len(fold.holdout_indices),
                "model_converged": bool(model.converged_),
            }
        )
    if np.any(fold_ids < 0) or not np.isfinite(probabilities).all() or not all(converged):
        raise RuntimeError("public replication OOF scoring is incomplete or non-converged")
    scores = group_safe_audit_scores(
        matrix,
        labels,
        probabilities,
        groups,
        fold_ids,
        training_groups_by_fold,
        sample_ids=sample_ids,
        method=str(candidate["risk_method"]),
        class_order=classes,
        neighbour_k=int(candidate["neighbour_k"]),
        neighbour_metric=str(candidate["neighbour_metric"]),
        hybrid_weights=(
            float(candidate["hybrid_self_confidence_weight"]),
            float(candidate["hybrid_neighbour_weight"]),
        ),
    )
    neighbour = scores.neighbour_evidence
    if neighbour is None:
        raise RuntimeError("frozen public replication hybrid lacks neighbour evidence")
    proposed = np.argmax(probabilities, axis=1).astype(np.int64)
    scored = prepared.manifest.copy()
    scored["oof_fold_id"] = fold_ids
    for index, class_name in enumerate(class_order):
        scored[f"oof_probability_{class_name}"] = probabilities[:, index]
    scored["proposed_label"] = proposed
    scored["proposed_class"] = [class_order[index] for index in proposed]
    scored["proposed_transition"] = [
        f"{source}->{target}"
        for source, target in zip(scored["observed_class"], scored["proposed_class"], strict=True)
    ]
    scored["risk_self_confidence"] = scores.component_scores["self_confidence"]
    scored["risk_neighbour_disagreement"] = scores.component_scores[
        "nearest_neighbour_disagreement"
    ]
    scored["risk_aanca"] = scores.risk_scores
    scored["neighbour_ids_json"] = [json.dumps(value) for value in neighbour.neighbour_ids]
    scored["neighbour_groups_json"] = [json.dumps(value) for value in neighbour.neighbour_groups]
    scored["neighbour_distances_json"] = [
        json.dumps(value, separators=(",", ":")) for value in neighbour.neighbour_distances
    ]
    for _, row in scored.iterrows():
        if str(row["group_id"]) in set(json.loads(str(row["neighbour_groups_json"]))):
            raise RuntimeError("query group leaked into public replication neighbours")
    return ScoredInputData(
        scored_manifest=scored,
        probabilities=probabilities,
        fold_ids=fold_ids,
        training_groups_by_fold=training_groups_by_fold,
        fold_evidence=tuple(fold_evidence),
        risk_metadata={
            **scores.as_dict(),
            "score_population_count": len(scored),
            "score_population_manifest_sha256": prepared.manifest_sha256,
            "reference_loaded_during_scoring": False,
            "splitter_class_name": plan.splitter_class_name,
            "splitter_fallback_status": plan.splitter_fallback_status,
            "splitter_fallback_reason": plan.splitter_fallback_reason,
        },
        all_models_converged=True,
    )


def _dataset_output_directory(root: Path, config: Mapping[str, Any], dataset: str) -> Path:
    outputs = cast(Mapping[str, Any], config["outputs"])
    return root / str(outputs["directory"]) / str(outputs[f"{dataset}_directory"])


def score_public_replication_rotation(
    repository_root: str | Path,
    *,
    dataset: str,
    rotation: str,
    device: str = "auto",
) -> dict[str, Any]:
    """Run one reference-blind rotation and persist an authenticated score seal."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, dataset)
    prepared = prepare_snapshot_pixels(root, dataset=dataset, rotation=rotation)
    destination = _dataset_output_directory(root, config, dataset)
    rotation_directory = destination / "pre_reference_scores" / rotation
    embeddings, embedding_metadata = extract_public_multiscale_embeddings(
        prepared,
        cache_directory=rotation_directory / "embeddings",
        study_id=str(config["study_id"]),
        dataset=dataset,
        rotation=rotation,
        device=device,
    )
    scored = score_public_input(
        prepared, embeddings, config=config, class_order=definition.class_order
    )
    scored_path = rotation_directory / "scored_input.csv"
    fold_path = rotation_directory / "fold_evidence.json"
    inventory_path = rotation_directory / "source_inventory.json"
    _write_csv(scored_path, scored.scored_manifest)
    atomic_write_json(fold_path, list(scored.fold_evidence))
    atomic_write_json(inventory_path, list(prepared.source_inventory))
    snapshot_manifest = json.loads(
        _snapshot_manifest_path(definition, root).read_text(encoding="utf-8")
    )
    snapshot_record = cast(Mapping[str, Any], snapshot_manifest["snapshots"][rotation])
    seal = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "dataset": dataset,
        "rotation": rotation,
        "config_sha256": config_sha,
        "class_order": list(definition.class_order),
        "snapshot_sha256": snapshot_record["sha256"],
        "input_manifest_sha256": prepared.manifest_sha256,
        "source_inventory_sha256": prepared.source_inventory_sha256,
        "crop_sha256": {str(key): value for key, value in prepared.crop_sha256.items()},
        "scored_input_sha256": sha256_file(scored_path),
        "fold_evidence_sha256": sha256_file(fold_path),
        "source_inventory_file_sha256": sha256_file(inventory_path),
        "score_population_count": len(scored.scored_manifest),
        "all_models_converged": scored.all_models_converged,
        "risk_metadata": scored.risk_metadata,
        "embedding_metadata": {str(key): value for key, value in embedding_metadata.items()},
        "hidden_reference_loaded": False,
        "full_annotation_source_opened": False,
        "reference_authority_opened": False,
        "process_boundary": "score_process_must_terminate_before_evaluation",
    }
    atomic_write_json(rotation_directory / "score_seal.json", seal)
    return seal


def _load_sealed_scores(
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    definition: DatasetDefinition,
    rotation: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    directory = (
        _dataset_output_directory(root, config, definition.name) / "pre_reference_scores" / rotation
    )
    seal_path = directory / "score_seal.json"
    seal = cast(Mapping[str, Any], json.loads(seal_path.read_text(encoding="utf-8")))
    if (
        seal.get("study_id") != config["study_id"]
        or seal.get("dataset") != definition.name
        or seal.get("rotation") != rotation
        or seal.get("config_sha256") != config_sha256
        or seal.get("class_order") != list(definition.class_order)
        or seal.get("hidden_reference_loaded") is not False
        or seal.get("full_annotation_source_opened") is not False
        or seal.get("reference_authority_opened") is not False
        or seal.get("all_models_converged") is not True
    ):
        raise RuntimeError("pre-reference score seal fails the frozen information barrier")
    snapshot_manifest = json.loads(
        _snapshot_manifest_path(definition, root).read_text(encoding="utf-8")
    )
    snapshot = cast(Mapping[str, Any], snapshot_manifest["snapshots"][rotation])
    if seal.get("snapshot_sha256") != snapshot["sha256"]:
        raise RuntimeError("score seal is not bound to the current input-only snapshot")
    paths = {
        "scored_input_sha256": directory / "scored_input.csv",
        "fold_evidence_sha256": directory / "fold_evidence.json",
        "source_inventory_file_sha256": directory / "source_inventory.json",
    }
    for field, path in paths.items():
        if sha256_file(path) != seal[field]:
            raise RuntimeError(f"pre-reference score artifact fails checksum: {path.name}")
    frame = pd.read_csv(paths["scored_input_sha256"])
    if (
        len(frame) != int(seal["score_population_count"])
        or frame["sample_id"].duplicated().any()
        or set(frame["input_annotator"].astype(str)) != {rotation}
        or not np.isfinite(frame["risk_aanca"].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError("sealed score population identity is invalid")
    return frame, seal


def _riva_bethesda_label(value: Any) -> str:
    label = str(value).strip().upper()
    if label in {"INFL", "ENDO"}:
        return "NILM"
    if label == "SCC":
        return "CA"
    return label


def attach_riva_hidden_reference(
    scored_by_rotation: Mapping[str, pd.DataFrame],
    repository_root: str | Path,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Attach strict leave-one-out votes from official raw cluster rows only."""

    root = Path(repository_root).resolve()
    definition = dataset_definition(config, "riva")
    settings = definition.settings
    cluster_path = root / str(settings["cluster_table"])
    _authenticate_source(cluster_path, sha256=str(settings["cluster_table_sha256"]))
    clusters = pd.read_csv(cluster_path)
    required = {
        "image_filename",
        "nucleus_x",
        "nucleus_y",
        "annotator_id",
        "class_bethesda",
        "cluster_idx",
    }
    if missing := required.difference(clusters.columns):
        raise ValueError(f"RIVA cluster authority lacks {sorted(missing)}")
    clusters = clusters.loc[:, sorted(required)].copy()
    clusters["annotator_id"] = clusters["annotator_id"].astype(int)
    clusters["x6"] = clusters["nucleus_x"].round(6)
    clusters["y6"] = clusters["nucleus_y"].round(6)
    clusters["bethesda"] = clusters["class_bethesda"].map(_riva_bethesda_label)
    key_columns = ("image_filename", "annotator_id", "x6", "y6", "bethesda")
    if clusters.duplicated(list(key_columns)).any():
        raise RuntimeError("RIVA official input-row identity key is not unique")
    row_lookup = {
        tuple(row[column] for column in key_columns): cast(int, index)
        for index, row in clusters.iterrows()
    }
    grouped: dict[tuple[str, float], pd.DataFrame] = {}
    for raw_group_key, local in clusters.groupby(["image_filename", "cluster_idx"], sort=False):
        image, cluster_id = cast(tuple[Any, Any], raw_group_key)
        grouped[(str(image), float(cluster_id))] = local.copy()
    annotator_ids = {
        str(key): int(value)
        for key, value in cast(Mapping[str, Any], settings["official_annotator_ids"]).items()
    }
    attached: dict[str, pd.DataFrame] = {}
    rotation_evidence: dict[str, Any] = {}
    for rotation in definition.rotations:
        output = scored_by_rotation[rotation].copy()
        output["reference_status"] = "insufficient_reference"
        output["reference_reason"] = "no_official_cluster_assignment"
        output["reference_consensus_label"] = pd.Series([pd.NA] * len(output), dtype="Int64")
        output["reference_consensus_class"] = ""
        output["independent_vote_count"] = 0
        output["reference_voter_ids_json"] = "[]"
        output["reference_vote_labels_json"] = "[]"
        output["official_cluster_id"] = ""
        input_id = annotator_ids[rotation]
        for output_index, row in output.iterrows():
            key = (
                f"{row['image_id']}.png",
                input_id,
                round(float(row["point_x_percent"]), 6),
                round(float(row["point_y_percent"]), 6),
                _riva_bethesda_label(row["observed_raw_label"]),
            )
            cluster_row_index = row_lookup.get(key)
            if cluster_row_index is None:
                continue
            authority_row = clusters.loc[cluster_row_index]
            cluster_id = float(authority_row["cluster_idx"])
            cluster = grouped[(str(authority_row["image_filename"]), cluster_id)]
            output.at[output_index, "official_cluster_id"] = (
                f"{authority_row['image_filename']}:{cluster_id:g}"
            )
            _, authority_input_class = map_riva_class(authority_row["class_bethesda"])
            if authority_input_class != str(row["observed_class"]):
                output.at[output_index, "reference_reason"] = "input_label_identity_conflict"
                continue
            if cluster["annotator_id"].duplicated().any():
                output.at[output_index, "reference_reason"] = "duplicate_annotator_vote_in_cluster"
                continue
            others = cluster.loc[cluster["annotator_id"] != input_id]
            voters = tuple(f"official_{int(value)}" for value in others["annotator_id"])
            vote_labels = tuple(map_riva_class(value)[1] for value in others["class_bethesda"])
            output.at[output_index, "independent_vote_count"] = len(vote_labels)
            output.at[output_index, "reference_voter_ids_json"] = json.dumps(voters)
            output.at[output_index, "reference_vote_labels_json"] = json.dumps(vote_labels)
            if len(vote_labels) < int(settings["minimum_independent_votes"]):
                output.at[output_index, "reference_reason"] = "fewer_than_two_independent_votes"
                continue
            counts = Counter(vote_labels)
            consensus_class, count = counts.most_common(1)[0]
            if count <= len(vote_labels) / 2.0:
                output.at[output_index, "reference_status"] = "ambiguous"
                output.at[output_index, "reference_reason"] = "no_strict_majority"
                continue
            consensus_label = definition.class_order.index(consensus_class)
            output.at[output_index, "reference_consensus_label"] = consensus_label
            output.at[output_index, "reference_consensus_class"] = consensus_class
            agrees = str(row["observed_class"]) == consensus_class
            output.at[output_index, "reference_status"] = (
                "consensus_agree" if agrees else "consensus_disagree"
            )
            output.at[output_index, "reference_reason"] = "strict_leave_one_out_majority"
        output["binary_reference_eligible"] = output["reference_status"].isin(
            ("consensus_agree", "consensus_disagree")
        )
        output["independent_consensus_disagreement"] = (
            output["reference_status"] == "consensus_disagree"
        )
        for value in output["reference_voter_ids_json"]:
            if f"official_{input_id}" in set(json.loads(str(value))):
                raise RuntimeError("RIVA input annotator leaked into leave-one-out reference")
        if not set(output["reference_status"]).issubset(REFERENCE_STATUSES):
            raise RuntimeError("RIVA reference attachment produced an unknown status")
        attached[rotation] = output
        rotation_evidence[rotation] = {
            "score_population_count": len(output),
            "outcome_counts": {
                key: int((output["reference_status"] == key).sum()) for key in REFERENCE_STATUSES
            },
            "reference_reason_counts": {
                str(key): int(value)
                for key, value in output["reference_reason"].value_counts().items()
            },
            "input_annotator_removed_from_reference": True,
            "official_released_majority_label_read": False,
        }
    inventory = _source_inventory(root, (cluster_path,))
    evidence = {
        "dataset": "riva",
        "reference_type": "strict_leave_one_out_majority",
        "minimum_independent_votes": int(settings["minimum_independent_votes"]),
        "official_released_majority_label_read": False,
        "rotations": rotation_evidence,
        "source_inventory_sha256": _semantic_sha256(inventory),
    }
    return attached, evidence, inventory


def attach_midogpp_hidden_reference(
    scored_by_rotation: Mapping[str, pd.DataFrame],
    repository_root: str | Path,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Attach only the other primary expert's positional label; ignore final/adjudicator data."""

    root = Path(repository_root).resolve()
    definition = dataset_definition(config, "midogpp")
    settings = definition.settings
    source = root / str(settings["annotations"])
    _authenticate_source(
        source,
        sha256=str(settings["annotations_sha256"]),
        md5=str(settings["annotations_md5"]),
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    annotations = {
        str(int(item["id"])): item
        for item in cast(Sequence[Mapping[str, Any]], payload["annotations"])
    }
    positions = {
        str(key): int(value)
        for key, value in cast(Mapping[str, Any], settings["label_positions"]).items()
    }
    attached: dict[str, pd.DataFrame] = {}
    rotation_evidence: dict[str, Any] = {}
    for rotation in definition.rotations:
        output = scored_by_rotation[rotation].copy()
        statuses: list[str] = []
        reference_labels: list[int] = []
        reference_classes: list[str] = []
        other_rotation = next(value for value in definition.rotations if value != rotation)
        input_position = positions[rotation]
        reference_position = positions[other_rotation]
        for _, row in output.iterrows():
            annotation = annotations[str(row["source_annotation_id"])]
            labels = list(cast(Sequence[Any], annotation["labels"]))
            input_label, input_class = map_midogpp_class(labels[input_position])
            if input_label != int(row["observed_label"]) or input_class != row["observed_class"]:
                raise RuntimeError("MIDOG++ input label identity conflicts with hidden authority")
            reference_label, reference_class = map_midogpp_class(labels[reference_position])
            reference_labels.append(reference_label)
            reference_classes.append(reference_class)
            statuses.append(
                "consensus_agree" if input_label == reference_label else "consensus_disagree"
            )
        output["reference_status"] = statuses
        output["reference_reason"] = "independent_pairwise_expert_label"
        output["reference_consensus_label"] = reference_labels
        output["reference_consensus_class"] = reference_classes
        output["independent_vote_count"] = 1
        output["reference_voter_ids_json"] = json.dumps((other_rotation,))
        output["reference_vote_labels_json"] = [json.dumps((value,)) for value in reference_classes]
        output["binary_reference_eligible"] = True
        output["independent_consensus_disagreement"] = (
            output["reference_status"] == "consensus_disagree"
        )
        attached[rotation] = output
        rotation_evidence[rotation] = {
            "score_population_count": len(output),
            "outcome_counts": {
                key: int((output["reference_status"] == key).sum())
                for key in ("consensus_agree", "consensus_disagree")
            },
            "input_expert_removed_from_reference": True,
            "adjudicator_label_read": False,
            "final_category_read": False,
        }
    inventory = _source_inventory(root, (source,))
    evidence = {
        "dataset": "midogpp",
        "reference_type": "independent_pairwise_expert_label",
        "adjudicator_label_used": False,
        "final_category_used": False,
        "rotations": rotation_evidence,
        "source_inventory_sha256": _semantic_sha256(inventory),
    }
    return attached, evidence, inventory


def _precision(events: NDArray[np.bool_], indices: NDArray[np.int64]) -> float:
    if not len(indices):
        raise ValueError("precision requires a non-empty queue")
    return float(events[indices].mean())


def _build_rotation_evaluation(
    frame: pd.DataFrame,
    *,
    rotation: str,
    config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    evaluation = cast(Mapping[str, Any], config["evaluation"])
    eligible = frame.loc[frame["binary_reference_eligible"]].copy()
    eligible = eligible.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError(f"no reference-eligible rows for {rotation}")
    events = eligible["independent_consensus_disagreement"].to_numpy(dtype=bool)
    groups = eligible["group_id"].astype(str).to_numpy(dtype=np.str_)
    sample_ids = eligible["sample_id"].astype(str).tolist()
    fields = tuple(str(value) for value in evaluation["matched_random_fields"])
    match_values = {field: eligible[field].astype(str).tolist() for field in fields}
    repeats = int(evaluation["matched_random_repetitions"])
    seed_start = int(evaluation["matched_random_seed"]) + seed_offset * 100000
    budgets: dict[str, Any] = {}
    queues: dict[str, Any] = {}
    internal: dict[str, Any] = {}
    for budget in (float(value) for value in evaluation["secondary_budgets"]):
        selected = select_exact_comparator_capable_queue(
            eligible, budget=budget, match_fields=fields
        )
        random_queues: list[NDArray[np.int64]] = []
        random_records: list[dict[str, Any]] = []
        budget_seed = seed_start + round(10000 * budget) * 1000
        for repeat in range(repeats):
            comparator = draw_matched_random_comparator(
                selected,
                np.ones(len(eligible), dtype=bool),
                sample_ids,
                match_values,
                seed=budget_seed + repeat,
            )
            if not comparator.available:
                raise RuntimeError(
                    f"exact comparator unavailable for {rotation}: {comparator.unavailable_reason}"
                )
            indices = np.asarray(comparator.comparator_indices, dtype=np.int64)
            if len(indices) != len(selected) or set(indices).intersection(selected.tolist()):
                raise RuntimeError("matched comparator is unequal or overlaps the AANCA queue")
            if Counter(comparator.top_match_strata) != Counter(comparator.comparator_match_strata):
                raise RuntimeError("matched comparator does not preserve exact frozen strata")
            random_queues.append(indices)
            random_records.append(
                {
                    "repeat": repeat,
                    "seed": comparator.seed,
                    "sample_ids": [sample_ids[index] for index in indices],
                    "match_strata": list(comparator.comparator_match_strata),
                }
            )
        selected_precision = _precision(events, selected)
        random_precisions = [_precision(events, indices) for indices in random_queues]
        random_mean = float(np.mean(random_precisions))
        key = f"{budget:.2f}"
        budgets[key] = {
            "budget_fraction": budget,
            "eligible_count": len(eligible),
            "reviewed_count": len(selected),
            "reference_positive_count": int(events.sum()),
            "aanca_reference_positive_reviewed": int(events[selected].sum()),
            "aanca_precision": selected_precision,
            "matched_random_mean_precision": random_mean,
            "matched_random_precision_min": float(np.min(random_precisions)),
            "matched_random_precision_max": float(np.max(random_precisions)),
            "precision_difference": selected_precision - random_mean,
            "enrichment_ratio": selected_precision / random_mean if random_mean > 0 else None,
            "exact_equal_budget": True,
            "exact_strata_preserved": True,
            "aanca_random_disjoint": True,
        }
        queues[key] = {
            "budget_fraction": budget,
            "aanca_sample_ids": [sample_ids[index] for index in selected],
            "aanca_match_strata": [
                json.dumps(tuple(str(eligible.iloc[index][field]) for field in fields))
                for index in selected
            ],
            "matched_random": random_records,
        }
        internal[key] = {
            "selected": selected,
            "random": tuple(random_queues),
        }
    risk = eligible["risk_aanca"].to_numpy(dtype=np.float64)
    order = rank_indices(risk, tie_break_ids=sample_ids)
    cumulative_positive = np.cumsum(events[order]).astype(np.int64)
    ranks = np.arange(1, len(order) + 1, dtype=np.int64)
    prevalence = float(events.mean())
    curve = pd.DataFrame(
        {
            "rotation": rotation,
            "rank": ranks,
            "sample_id": eligible.iloc[order]["sample_id"].astype(str).tolist(),
            "risk_aanca": risk[order],
            "cumulative_reference_positive": cumulative_positive,
            "cumulative_precision": cumulative_positive / ranks,
            "cumulative_recall": cumulative_positive / int(events.sum()),
            "cumulative_enrichment": (cumulative_positive / ranks) / prevalence,
        }
    )
    return {
        "eligible": eligible,
        "events": events,
        "groups": groups,
        "budgets": budgets,
        "queues": queues,
        "internal": internal,
        "curve": curve,
        "average_precision": average_precision(events, risk),
        "reference_positive_prevalence": prevalence,
    }


def _aggregate_bootstrap(
    works: Mapping[str, Mapping[str, Any]],
    *,
    budget_key: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(
        np.concatenate([cast(NDArray[np.str_], work["groups"]) for work in works.values()])
    )
    if len(unique_groups) < 2:
        raise RuntimeError("aggregate bootstrap needs at least two independent groups")
    group_to_column = {str(group): index for index, group in enumerate(unique_groups)}
    repeats = len(
        cast(Mapping[str, Any], next(iter(works.values()))["internal"])[budget_key]["random"]
    )
    prepared: dict[str, Any] = {}
    for rotation, work in works.items():
        events = cast(NDArray[np.bool_], work["events"])
        groups = cast(NDArray[np.str_], work["groups"])
        budget = cast(Mapping[str, Any], work["internal"])[budget_key]
        selected = cast(NDArray[np.int64], budget["selected"])
        random_queues = cast(tuple[NDArray[np.int64], ...], budget["random"])
        if len(random_queues) != repeats:
            raise RuntimeError("rotation comparator repeat counts differ")
        prepared[rotation] = {
            "events": events,
            "selected": selected,
            "selected_columns": np.asarray(
                [group_to_column[str(value)] for value in groups[selected]], dtype=np.int64
            ),
            "random": random_queues,
            "random_columns": tuple(
                np.asarray(
                    [group_to_column[str(value)] for value in groups[indices]], dtype=np.int64
                )
                for indices in random_queues
            ),
        }
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    enrichments: list[float] = []
    undefined_difference = 0
    undefined_enrichment = 0
    for _ in range(iterations):
        draw = rng.integers(0, len(unique_groups), size=len(unique_groups))
        multiplicity = np.bincount(draw, minlength=len(unique_groups)).astype(np.float64)
        selected_numerator = 0.0
        selected_denominator = 0.0
        for item in prepared.values():
            indices = cast(NDArray[np.int64], item["selected"])
            weights = multiplicity[cast(NDArray[np.int64], item["selected_columns"])]
            selected_numerator += float(
                np.sum(cast(NDArray[np.bool_], item["events"])[indices] * weights)
            )
            selected_denominator += float(weights.sum())
        if selected_denominator <= 0:
            undefined_difference += 1
            undefined_enrichment += 1
            continue
        selected_precision = selected_numerator / selected_denominator
        comparator_precisions: list[float] = []
        for repeat in range(repeats):
            numerator = 0.0
            denominator = 0.0
            for item in prepared.values():
                indices = cast(tuple[NDArray[np.int64], ...], item["random"])[repeat]
                columns = cast(tuple[NDArray[np.int64], ...], item["random_columns"])[repeat]
                weights = multiplicity[columns]
                numerator += float(
                    np.sum(cast(NDArray[np.bool_], item["events"])[indices] * weights)
                )
                denominator += float(weights.sum())
            if denominator > 0:
                comparator_precisions.append(numerator / denominator)
        if not comparator_precisions:
            undefined_difference += 1
            undefined_enrichment += 1
            continue
        comparator_precision = float(np.mean(comparator_precisions))
        differences.append(selected_precision - comparator_precision)
        if comparator_precision > 0:
            enrichments.append(selected_precision / comparator_precision)
        else:
            undefined_enrichment += 1
    if not differences:
        raise RuntimeError("aggregate group bootstrap produced no finite differences")

    def interval(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        return [float(value) for value in np.quantile(values, (0.025, 0.975))]

    return {
        "unit": "dataset_group_id",
        "unique_group_count": len(unique_groups),
        "requested_iterations": iterations,
        "valid_difference_iterations": len(differences),
        "undefined_difference_iterations": undefined_difference,
        "precision_difference_interval_95": interval(differences),
        "valid_enrichment_iterations": len(enrichments),
        "undefined_enrichment_iterations": undefined_enrichment,
        "enrichment_ratio_interval_95": interval(enrichments),
        "seed": seed,
    }


def evaluate_attached_dataset(
    attached: Mapping[str, pd.DataFrame],
    *,
    dataset: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Evaluate the prospectively fixed queues and group bootstrap for one dataset."""

    definition = dataset_definition(config, dataset)
    evaluation = cast(Mapping[str, Any], config["evaluation"])
    works = {
        rotation: _build_rotation_evaluation(
            attached[rotation], rotation=rotation, config=config, seed_offset=index
        )
        for index, rotation in enumerate(definition.rotations)
    }
    rotation_metrics: dict[str, Any] = {}
    queue_payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "match_fields": list(evaluation["matched_random_fields"]),
        "rotations": {},
    }
    primary_key = f"{float(evaluation['primary_budget']):.2f}"
    for rotation, work in works.items():
        rotation_metrics[rotation] = {
            "eligible_count": len(cast(pd.DataFrame, work["eligible"])),
            "reference_positive_count": int(cast(NDArray[np.bool_], work["events"]).sum()),
            "reference_positive_prevalence": work["reference_positive_prevalence"],
            "average_precision": work["average_precision"],
            "primary": cast(Mapping[str, Any], work["budgets"])[primary_key],
            "secondary_budgets": work["budgets"],
        }
        cast(dict[str, Any], queue_payload["rotations"])[rotation] = work["queues"]
    aggregate_budgets: dict[str, Any] = {}
    for budget in (float(value) for value in evaluation["secondary_budgets"]):
        key = f"{budget:.2f}"
        selected_positive = 0
        selected_count = 0
        comparator_positive = np.zeros(int(evaluation["matched_random_repetitions"]))
        comparator_count = np.zeros(int(evaluation["matched_random_repetitions"]))
        eligible_count = 0
        reference_positive = 0
        for work in works.values():
            events = cast(NDArray[np.bool_], work["events"])
            internal = cast(Mapping[str, Any], work["internal"])[key]
            selected = cast(NDArray[np.int64], internal["selected"])
            random_queues = cast(tuple[NDArray[np.int64], ...], internal["random"])
            selected_positive += int(events[selected].sum())
            selected_count += len(selected)
            eligible_count += len(events)
            reference_positive += int(events.sum())
            for repeat, indices in enumerate(random_queues):
                comparator_positive[repeat] += int(events[indices].sum())
                comparator_count[repeat] += len(indices)
        selected_precision = selected_positive / selected_count
        random_precisions = comparator_positive / comparator_count
        random_mean = float(np.mean(random_precisions))
        aggregate_budgets[key] = {
            "budget_fraction": budget,
            "eligible_rotation_rows": eligible_count,
            "reviewed_rotation_rows": selected_count,
            "reference_positive_rotation_rows": reference_positive,
            "aanca_reference_positive_reviewed": selected_positive,
            "aanca_precision": selected_precision,
            "matched_random_mean_precision": random_mean,
            "matched_random_precision_min": float(np.min(random_precisions)),
            "matched_random_precision_max": float(np.max(random_precisions)),
            "precision_difference": selected_precision - random_mean,
            "enrichment_ratio": selected_precision / random_mean if random_mean > 0 else None,
            "exact_equal_budget": True,
            "exact_strata_preserved": True,
            "aanca_random_disjoint": True,
        }
    bootstrap = _aggregate_bootstrap(
        works,
        budget_key=primary_key,
        iterations=int(evaluation["bootstrap_iterations"]),
        seed=int(evaluation["bootstrap_seed"]) + (0 if dataset == "riva" else 100000),
    )
    primary = {**aggregate_budgets[primary_key], "bootstrap": bootstrap}
    interval = bootstrap["precision_difference_interval_95"]
    all_rotation_points_nonnegative = all(
        float(rotation_metrics[rotation]["primary"]["precision_difference"]) >= 0.0
        for rotation in definition.rotations
    )
    primary_gate_pass = bool(
        interval is not None
        and float(interval[0]) > 0.0
        and all_rotation_points_nonnegative
        and primary["exact_equal_budget"]
        and primary["exact_strata_preserved"]
        and primary["aanca_random_disjoint"]
    )
    by_class: dict[str, Any] = {}
    for class_name in definition.class_order:
        selected_positive = 0
        selected_count = 0
        random_positive = np.zeros(int(evaluation["matched_random_repetitions"]))
        random_count = np.zeros(int(evaluation["matched_random_repetitions"]))
        eligible_count = 0
        reference_positive = 0
        for work in works.values():
            eligible = cast(pd.DataFrame, work["eligible"])
            events = cast(NDArray[np.bool_], work["events"])
            mask = eligible["observed_class"].astype(str).to_numpy() == class_name
            internal = cast(Mapping[str, Any], work["internal"])[primary_key]
            selected = cast(NDArray[np.int64], internal["selected"])
            selected_class = selected[mask[selected]]
            selected_positive += int(events[selected_class].sum())
            selected_count += len(selected_class)
            eligible_count += int(mask.sum())
            reference_positive += int(events[mask].sum())
            for repeat, indices in enumerate(
                cast(tuple[NDArray[np.int64], ...], internal["random"])
            ):
                indices_class = indices[mask[indices]]
                random_positive[repeat] += int(events[indices_class].sum())
                random_count[repeat] += len(indices_class)
        random_values = random_positive[random_count > 0] / random_count[random_count > 0]
        aanca_precision = selected_positive / selected_count if selected_count else None
        class_random_mean = float(np.mean(random_values)) if len(random_values) else None
        by_class[class_name] = {
            "eligible_rotation_rows": eligible_count,
            "reference_positive_rotation_rows": reference_positive,
            "aanca_reviewed_rotation_rows": selected_count,
            "aanca_precision": aanca_precision,
            "matched_random_mean_precision": class_random_mean,
            "precision_difference": (
                aanca_precision - class_random_mean
                if aanca_precision is not None and class_random_mean is not None
                else None
            ),
        }
    curves = pd.concat(
        [cast(pd.DataFrame, work["curve"]) for work in works.values()], ignore_index=True
    )
    metrics = {
        "dataset": dataset,
        "primary": primary,
        "rotations": rotation_metrics,
        "secondary_budgets": aggregate_budgets,
        "by_observed_class": by_class,
        "all_rotation_point_differences_nonnegative": all_rotation_points_nonnegative,
        "primary_gate_pass": primary_gate_pass,
        "primary_gate_rule": evaluation["dataset_success_rule"],
    }
    return metrics, queue_payload, curves


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    records = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
        "artifact_root_sha256": _semantic_sha256(records),
    }


def evaluate_public_replication_dataset(
    repository_root: str | Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Authenticate all score seals, open one hidden authority, and evaluate endpoints."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, dataset)
    scored: dict[str, pd.DataFrame] = {}
    seals: dict[str, Mapping[str, Any]] = {}
    for rotation in definition.rotations:
        scored[rotation], seals[rotation] = _load_sealed_scores(
            root, config, config_sha, definition, rotation
        )
    if dataset == "riva":
        attached, reference_evidence, reference_inventory = attach_riva_hidden_reference(
            scored, root, config=config
        )
    elif dataset == "midogpp":
        attached, reference_evidence, reference_inventory = attach_midogpp_hidden_reference(
            scored, root, config=config
        )
    else:
        raise ValueError(f"unknown frozen dataset: {dataset}")
    metrics, queues, curves = evaluate_attached_dataset(attached, dataset=dataset, config=config)
    destination = _dataset_output_directory(root, config, dataset)
    combined = pd.concat(
        [attached[rotation] for rotation in definition.rotations], ignore_index=True
    )
    attached_path = destination / "reference_attached.csv"
    queue_path = destination / "matched_random_queues.json"
    curve_path = destination / "enrichment_curves.csv"
    inventory_path = destination / "reference_source_inventory.json"
    _write_csv(attached_path, combined)
    atomic_write_json(queue_path, queues)
    _write_csv(curve_path, curves)
    atomic_write_json(inventory_path, list(reference_inventory))
    results = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "dataset": dataset,
        "execution_complete": True,
        "config_sha256": config_sha,
        "protocol": config["protocol"],
        "pre_reference_score_seals": {
            rotation: {
                "scored_input_sha256": seal["scored_input_sha256"],
                "source_inventory_sha256": seal["source_inventory_sha256"],
                "hidden_reference_loaded": seal["hidden_reference_loaded"],
            }
            for rotation, seal in seals.items()
        },
        "information_barrier": {
            "input_only_snapshots": True,
            "separate_prepare_score_evaluate_processes_required": True,
            "risks_recomputed_after_reference": False,
            "all_score_seals_authenticated_before_reference": True,
        },
        "reference": reference_evidence,
        "metrics": metrics,
        "artifact_hashes": {
            "reference_attached_csv": sha256_file(attached_path),
            "matched_random_queues_json": sha256_file(queue_path),
            "enrichment_curves_csv": sha256_file(curve_path),
            "reference_source_inventory_json": sha256_file(inventory_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    results_path = destination / "results.json"
    atomic_write_json(results_path, results)
    manifest = _artifact_manifest(destination)
    atomic_write_json(destination / "artifact_manifest.json", manifest)
    return results


def _format_float(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _format_signed(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def _format_interval(value: Any) -> str:
    if value is None:
        return "NA"
    return f"[{float(value[0]):+.6f}, {float(value[1]):+.6f}]"


def _dataset_report_lines(result: Mapping[str, Any]) -> list[str]:
    dataset = str(result["dataset"])
    metrics = cast(Mapping[str, Any], result["metrics"])
    primary = cast(Mapping[str, Any], metrics["primary"])
    bootstrap = cast(Mapping[str, Any], primary["bootstrap"])
    title = "RIVA" if dataset == "riva" else "MIDOG++"
    lines = [
        f"## {title}",
        "",
        f"Frozen dataset gate: `{'PASS' if metrics['primary_gate_pass'] else 'FAIL'}`.",
        "",
        f"- Eligible rotation-rows: {primary['eligible_rotation_rows']}.",
        f"- Reviewed rotation-rows at 5%: {primary['reviewed_rotation_rows']}.",
        f"- AANCA disagreement precision: {_format_float(primary['aanca_precision'])}.",
        "- Exact matched-random mean precision: "
        f"{_format_float(primary['matched_random_mean_precision'])}.",
        f"- Precision difference: {_format_signed(primary['precision_difference'])}.",
        f"- Enrichment ratio: {_format_float(primary['enrichment_ratio'])}.",
        "- Group-bootstrap 95% interval for the precision difference: "
        f"{_format_interval(bootstrap['precision_difference_interval_95'])}.",
        f"- Independent groups in bootstrap: {bootstrap['unique_group_count']}.",
        "- Every rotation point estimate non-negative: "
        f"`{str(metrics['all_rotation_point_differences_nonnegative']).lower()}`.",
        "",
        "| Input rotation | Eligible | Disagreements | AANCA precision | Random precision | Difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rotation, raw in cast(Mapping[str, Any], metrics["rotations"]).items():
        rotation_metrics = cast(Mapping[str, Any], raw)
        rotation_primary = cast(Mapping[str, Any], rotation_metrics["primary"])
        lines.append(
            f"| `{rotation}` | {rotation_metrics['eligible_count']} | "
            f"{rotation_metrics['reference_positive_count']} | "
            f"{_format_float(rotation_primary['aanca_precision'])} | "
            f"{_format_float(rotation_primary['matched_random_mean_precision'])} | "
            f"{_format_signed(rotation_primary['precision_difference'])} |"
        )
    lines.extend(
        [
            "",
            "All queues use exact equal budgets, exact frozen strata, and disjoint random "
            "comparators. Source annotations were not modified.",
            "",
        ]
    )
    return lines


def render_public_replication_report(repository_root: str | Path) -> dict[str, Any]:
    """Combine both frozen dataset outcomes without changing either result."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    results: dict[str, Mapping[str, Any]] = {}
    for dataset in ("riva", "midogpp"):
        path = _dataset_output_directory(root, config, dataset) / "results.json"
        result = cast(Mapping[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if result.get("config_sha256") != config_sha or result.get("dataset") != dataset:
            raise RuntimeError(f"{dataset} result is not bound to the frozen public protocol")
        results[dataset] = result
    riva_pass = bool(cast(Mapping[str, Any], results["riva"]["metrics"])["primary_gate_pass"])
    midog_pass = bool(cast(Mapping[str, Any], results["midogpp"]["metrics"])["primary_gate_pass"])
    cross_dataset = riva_pass and midog_pass
    if cross_dataset:
        conclusion = (
            "Both frozen gates passed: the candidate replicated enrichment for natural "
            "independent-expert disagreements in these two public releases."
        )
    elif riva_pass:
        conclusion = (
            "RIVA passed but MIDOG++ did not; multi-rater cytology replication is supported, "
            "but cross-dataset replication is not."
        )
    elif midog_pass:
        conclusion = (
            "MIDOG++ passed but RIVA did not; pairwise mitosis replication is supported, "
            "but the primary multi-rater replication is not."
        )
    else:
        conclusion = (
            "Neither frozen gate passed; replicated disagreement enrichment is not supported."
        )
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "completion_stage": "EXTERNAL_VALIDATION_COMPLETE",
        "config_sha256": config_sha,
        "riva_gate_pass": riva_pass,
        "midogpp_gate_pass": midog_pass,
        "cross_dataset_replication_supported": cross_dataset,
        "conclusion": conclusion,
        "claim_boundary": config["claim_boundary"],
    }
    lines = [
        "# Public independent-pathologist replication results",
        "",
        "Completion stage: `EXTERNAL_VALIDATION_COMPLETE`.",
        "",
        conclusion,
        "",
        "These outcomes concern enrichment for potentially inconsistent annotations "
        "recommended for expert review. They do not prove pathologist error, diagnostic "
        "accuracy, clinical safety, or downstream utility.",
        "",
    ]
    lines.extend(_dataset_report_lines(results["riva"]))
    lines.extend(_dataset_report_lines(results["midogpp"]))
    lines.extend(
        [
            "## Frozen interpretation",
            "",
            f"- RIVA primary gate: `{'PASS' if riva_pass else 'FAIL'}`.",
            f"- MIDOG++ replication gate: `{'PASS' if midog_pass else 'FAIL'}`.",
            f"- Cross-dataset replication: `{'SUPPORTED' if cross_dataset else 'NOT_SUPPORTED'}`.",
            "- Earlier NuCLS and downstream results remain additive and unchanged.",
            "- A prospective blinded equal-budget workflow trial and untouched downstream "
            "test remain required.",
            "",
        ]
    )
    outputs = cast(Mapping[str, Any], config["outputs"])
    report_path = root / str(outputs["report"])
    atomic_write_text(report_path, "\n".join(lines))
    destination = root / str(outputs["directory"])
    atomic_write_json(destination / "summary.json", summary)
    atomic_write_text(destination / "report.md", "\n".join(lines))
    atomic_write_json(destination / "artifact_manifest.json", _artifact_manifest(destination))
    return summary


def verify_public_replication_dataset(
    repository_root: str | Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Independently recompute metrics/queues from sealed reference-attached rows."""

    root = Path(repository_root).resolve()
    config, config_sha = load_frozen_public_replication_config(root)
    definition = dataset_definition(config, dataset)
    destination = _dataset_output_directory(root, config, dataset)
    results = cast(
        Mapping[str, Any], json.loads((destination / "results.json").read_text(encoding="utf-8"))
    )
    if results.get("config_sha256") != config_sha or results.get("dataset") != dataset:
        raise RuntimeError("stored result is not bound to the frozen public config")
    hashes = cast(Mapping[str, Any], results["artifact_hashes"])
    checked = {
        "reference_attached_csv": destination / "reference_attached.csv",
        "matched_random_queues_json": destination / "matched_random_queues.json",
        "enrichment_curves_csv": destination / "enrichment_curves.csv",
        "reference_source_inventory_json": destination / "reference_source_inventory.json",
    }
    for field, path in checked.items():
        if sha256_file(path) != hashes[field]:
            raise RuntimeError(f"stored public replication artifact changed: {path.name}")
    combined = pd.read_csv(checked["reference_attached_csv"])
    attached = {
        rotation: combined.loc[combined["input_annotator"].astype(str) == rotation]
        .copy()
        .reset_index(drop=True)
        for rotation in definition.rotations
    }
    metrics, queues, curves = evaluate_attached_dataset(attached, dataset=dataset, config=config)
    stored_queues = json.loads(checked["matched_random_queues_json"].read_text(encoding="utf-8"))
    stored_curves = pd.read_csv(checked["enrichment_curves_csv"])
    if _semantic_sha256(metrics) != _semantic_sha256(results["metrics"]):
        raise RuntimeError("independently recomputed metrics differ from stored results")
    if _semantic_sha256(queues) != _semantic_sha256(stored_queues):
        raise RuntimeError("independently recomputed queues differ from stored queues")
    pd.testing.assert_frame_equal(
        curves.reset_index(drop=True),
        stored_curves.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    return {
        "study_id": config["study_id"],
        "dataset": dataset,
        "verification_passed": True,
        "primary_gate_pass": metrics["primary_gate_pass"],
        "config_sha256": config_sha,
    }


__all__ = [
    "MIDOGPP_CLASS_ORDER",
    "RIVA_CLASS_ORDER",
    "attach_midogpp_hidden_reference",
    "attach_riva_hidden_reference",
    "dataset_definition",
    "download_midogpp_selected_images",
    "evaluate_attached_dataset",
    "evaluate_public_replication_dataset",
    "load_frozen_public_replication_config",
    "map_midogpp_class",
    "map_riva_class",
    "materialize_midogpp_input_snapshots",
    "materialize_riva_input_snapshots",
    "prepare_snapshot_pixels",
    "render_public_replication_report",
    "riva_smear_group",
    "score_public_input",
    "score_public_replication_rotation",
    "select_midogpp_images",
    "verify_public_replication_dataset",
]
