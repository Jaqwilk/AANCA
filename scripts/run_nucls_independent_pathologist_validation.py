"""Run the frozen NuCLS leave-one-pathologist-out validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histo_audit.external_validation.nucls_independent_pathologist import (
    run_independent_pathologist_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/nucls_independent_pathologist_validation"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/nucls_independent_pathologist_validation_results.md"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    result = run_independent_pathologist_validation(
        repository_root,
        output_directory=args.output,
        report_path=args.report,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "study_id": result["study_id"],
                "execution_status": result["execution_status"],
                "primary_gate_pass": result["metrics"]["primary_gate_pass"],
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
