"""Prepare and execute the frozen NuCLS multi-rater validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histo_audit.external_validation.nucls import (
    extract_nucls_embeddings,
    load_frozen_nucls_config,
    prepare_nucls_subset,
    run_nucls_external_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("unbiased", "evaluation"), default="unbiased")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/nucls"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    config, freeze = load_frozen_nucls_config(repository_root)
    subset_key = "primary" if args.subset == "unbiased" else "secondary"
    dataset_config = config["datasets"][subset_key]
    subset_root = args.data_root.resolve() / args.subset
    prepared = prepare_nucls_subset(
        subset_root / "np_truth",
        subset_root / "p_truth",
        subset_name=str(dataset_config["name"]),
        crop_size=int(config["representation"]["crop_size"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embedding_cache = args.output.parent / f"{args.output.name}.embeddings.npz"
    embeddings = extract_nucls_embeddings(
        prepared,
        cache_path=embedding_cache,
        device=args.device,
    )
    result = run_nucls_external_validation(
        prepared,
        embeddings,
        config=config,
        output_directory=args.output,
    )
    print(
        json.dumps(
            {
                "freeze": freeze,
                "result": result,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
