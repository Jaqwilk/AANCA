"""Run the prospectively frozen PUMA confirmation without tuning on its outcome."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from histo_audit.external_validation.puma import (
    PUMAPreparedData,
    build_puma_manifest,
    extract_puma_embeddings,
    load_frozen_puma_config,
    prepare_puma_split,
    render_puma_report,
    run_puma_new_data_confirmation,
    validate_puma_embedding_alignment,
)
from histo_audit.utils.run_tracking import (
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    sha256_file,
)


def _without_crops(prepared: PUMAPreparedData) -> PUMAPreparedData:
    return PUMAPreparedData(
        split=prepared.split,
        manifest=prepared.manifest,
        crops=np.empty((0, 0, 0, 3), dtype=np.uint8),
        exclusions=prepared.exclusions,
        manifest_sha256=prepared.manifest_sha256,
        source_inventory_sha256=prepared.source_inventory_sha256,
        crops_sha256=prepared.crops_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/puma"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/puma_new_data_confirmation"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/puma_new_data_confirmation_results.md"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = (root / args.data_root).resolve()
    output_root = (root / args.output).resolve()
    report_path = (root / args.report).resolve()
    config, config_sha256, amendment, amendment_sha256 = load_frozen_puma_config(root)
    manifest_authority = build_puma_manifest(data_root, config)
    output_root.mkdir(parents=True, exist_ok=True)
    authority: dict[str, Any] = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "runtime_amendment_sha256": amendment_sha256,
        "candidate_sha256": config["candidate"]["sha256"],
        "candidate_record_sha256": sha256_file(root / config["candidate"]["record"]),
        "full_manifest_sha256": manifest_authority.manifest_sha256,
        "source_inventory_sha256": manifest_authority.source_inventory_sha256,
        "development_case_ids": list(manifest_authority.development_case_ids),
        "final_case_ids": list(manifest_authority.final_case_ids),
        "native_class_counts": dict(manifest_authority.native_class_counts),
        "candidate_selection_from_puma_permitted": False,
        "source_annotations_modified": False,
    }
    atomic_write_json(output_root / "run_authority.json", authority)
    print(json.dumps({"stage": "authority_verified", **authority}, sort_keys=True))

    embedding_root = root / "artifacts" / "embeddings" / "puma_new_data_confirmation"
    prepared_64: dict[str, PUMAPreparedData] = {}
    embeddings_64: dict[str, np.ndarray] = {}
    for split in ("development", "final"):
        print(json.dumps({"stage": "prepare_crops", "split": split, "scale": 64}))
        prepared = prepare_puma_split(data_root, split=split, crop_size=64, config=config)
        embeddings = extract_puma_embeddings(
            prepared,
            cache_path=embedding_root / f"puma_{split}_resnet18_context_64.npz",
            scale_px=64,
            device=args.device,
        )
        prepared_64[split] = _without_crops(prepared)
        embeddings_64[split] = embeddings
        del prepared
        gc.collect()
        print(json.dumps({"stage": "embedded", "split": split, "scale": 64}))

    embeddings_128: dict[str, np.ndarray] = {}
    for split in ("development", "final"):
        print(json.dumps({"stage": "prepare_crops", "split": split, "scale": 128}))
        prepared = prepare_puma_split(data_root, split=split, crop_size=128, config=config)
        if prepared.manifest_sha256 != prepared_64[
            split
        ].manifest_sha256 or not prepared.manifest.equals(prepared_64[split].manifest):
            raise RuntimeError("PUMA 64px and 128px manifests differ")
        embeddings = extract_puma_embeddings(
            prepared,
            cache_path=embedding_root / f"puma_{split}_resnet18_context_128.npz",
            scale_px=128,
            device=args.device,
        )
        validate_puma_embedding_alignment(prepared, embeddings_64[split], embeddings)
        embeddings_128[split] = embeddings
        del prepared
        gc.collect()
        print(json.dumps({"stage": "embedded", "split": split, "scale": 128}))

    development_features = np.concatenate(
        (embeddings_64["development"], embeddings_128["development"]), axis=1
    ).astype(np.float32, copy=False)
    final_features = np.concatenate(
        (embeddings_64["final"], embeddings_128["final"]), axis=1
    ).astype(np.float32, copy=False)
    del embeddings_64, embeddings_128
    gc.collect()
    print(
        json.dumps(
            {
                "stage": "execute_frozen_metrics",
                "development_shape": list(development_features.shape),
                "final_shape": list(final_features.shape),
            },
            sort_keys=True,
        )
    )
    result, arrays = run_puma_new_data_confirmation(
        prepared_64["development"],
        prepared_64["final"],
        development_features,
        final_features,
        config,
        amendment,
        config_sha256=config_sha256,
        amendment_sha256=amendment_sha256,
        neighbour_device=args.device,
    )
    atomic_write_json(output_root / "results.json", result)
    atomic_write_npz(output_root / "evidence_arrays.npz", arrays)
    atomic_write_text(report_path, render_puma_report(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["all_success_conditions_met"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
