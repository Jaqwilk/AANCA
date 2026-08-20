"""Execute the frozen current-AANCA controlled external benchmark on MoNuSAC."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from histo_audit.external_validation.monusac import (
    extract_monusac_embeddings,
    load_frozen_monusac_config,
    prepare_monusac_split,
    render_monusac_report,
    run_monusac_controlled_external,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_archive(path: Path, expected_sha256: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"official MoNuSAC {role} archive is absent: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"official MoNuSAC {role} archive SHA-256 differs: {actual} != {expected_sha256}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/monusac"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/monusac_external_validation")
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    data_root = (repository_root / args.data_root).resolve()
    output_root = (repository_root / args.output_root).resolve()
    config, config_sha256 = load_frozen_monusac_config(repository_root)
    authority = config["authority"]
    _verify_archive(
        data_root / "monusac_train.zip",
        str(authority["train_archive_sha256"]),
        "train",
    )
    _verify_archive(
        data_root / "monusac_test.zip",
        str(authority["test_archive_sha256"]),
        "test",
    )
    train_root = data_root / "train" / "MoNuSAC_images_and_annotations"
    test_root = data_root / "test" / "MoNuSAC Testing Data and Annotations"
    overlap = tuple(str(value) for value in config["dataset"]["overlap_policy"]["identities"])
    train = prepare_monusac_split(
        train_root,
        split="train",
        crop_size=int(config["representation"]["crop_size"]),
        excluded_patients=overlap,
    )
    test = prepare_monusac_split(
        test_root,
        split="test",
        crop_size=int(config["representation"]["crop_size"]),
    )
    nucls_patients: set[str] = set()
    for subset in ("unbiased-v1", "evaluation-v1"):
        manifest_path = (
            repository_root
            / "artifacts"
            / "nucls_external_validation"
            / subset
            / "canonical_manifest.csv"
        )
        if manifest_path.is_file():
            import pandas as pd

            nucls_patients.update(
                pd.read_csv(manifest_path, usecols=["patient_id"])["patient_id"]
                .astype(str)
                .tolist()
            )
    monusac_patients = set(train.manifest["patient_id"].astype(str)) | set(
        test.manifest["patient_id"].astype(str)
    )
    nucls_overlap = sorted(nucls_patients.intersection(monusac_patients))
    if nucls_overlap:
        raise RuntimeError(f"MoNuSAC patient identities overlap saved NuCLS: {nucls_overlap}")

    embedding_root = repository_root / "artifacts" / "embeddings" / "monusac_current_aanca"
    train_embeddings = extract_monusac_embeddings(
        train,
        cache_path=embedding_root / "train_resnet18.npz",
        device=args.device,
    )
    test_embeddings = extract_monusac_embeddings(
        test,
        cache_path=embedding_root / "test_resnet18.npz",
        device=args.device,
    )
    result, arrays = run_monusac_controlled_external(
        train,
        test,
        train_embeddings,
        test_embeddings,
        config,
        config_sha256=config_sha256,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.json"
    report_path = output_root / "report.md"
    evidence_path = output_root / "numeric_evidence.npz"
    inventory_path = output_root / "source_inventory.json"
    atomic_write_json(results_path, result)
    atomic_write_text(report_path, render_monusac_report(result))
    _atomic_npz(evidence_path, arrays)
    atomic_write_json(
        inventory_path,
        {
            "schema_version": 1,
            "train": list(train.source_inventory),
            "test": list(test.source_inventory),
        },
    )
    artifact_manifest = {
        "schema_version": 1,
        "study_id": result["study_id"],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (results_path, report_path, evidence_path, inventory_path)
        },
    }
    atomic_write_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(
        json.dumps(
            {
                "study_id": result["study_id"],
                "all_success_conditions_met": result["all_success_conditions_met"],
                "decision": result["decision"],
                "output_root": output_root.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
