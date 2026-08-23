"""Verify and recalculate the NuCLS independent-pathologist artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histo_audit.external_validation.nucls_independent_pathologist import (
    verify_independent_pathologist_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/nucls_independent_pathologist_validation"),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    verification = verify_independent_pathologist_artifacts(
        repository_root,
        output_directory=args.output,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
