"""Execute one process-bounded stage of the frozen public pathologist replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histo_audit.external_validation.public_pathologist_replication import (
    download_midogpp_selected_images,
    evaluate_public_replication_dataset,
    materialize_midogpp_input_snapshots,
    materialize_riva_input_snapshots,
    render_public_replication_report,
    score_public_replication_rotation,
    verify_public_replication_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "download", "score", "evaluate", "report", "verify"),
    )
    parser.add_argument("--dataset", choices=("riva", "midogpp"))
    parser.add_argument("--rotation")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage == "prepare":
        if args.dataset == "riva":
            result = materialize_riva_input_snapshots(root)
        elif args.dataset == "midogpp":
            result = materialize_midogpp_input_snapshots(root)
        else:
            parser.error("prepare requires --dataset")
    elif args.stage == "download":
        if args.dataset != "midogpp":
            parser.error("download requires --dataset midogpp")
        result = download_midogpp_selected_images(root)
    elif args.stage == "score":
        if args.dataset is None or args.rotation is None:
            parser.error("score requires --dataset and --rotation")
        result = score_public_replication_rotation(
            root,
            dataset=args.dataset,
            rotation=args.rotation,
            device=args.device,
        )
    elif args.stage == "evaluate":
        if args.dataset is None:
            parser.error("evaluate requires --dataset")
        result = evaluate_public_replication_dataset(root, dataset=args.dataset)
    elif args.stage == "verify":
        if args.dataset is None:
            parser.error("verify requires --dataset")
        result = verify_public_replication_dataset(root, dataset=args.dataset)
    else:
        if args.dataset is not None or args.rotation is not None:
            parser.error("report does not accept --dataset or --rotation")
        result = render_public_replication_report(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
