"""Recalculate the post-outcome improvement analysis for the current AANCA model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from histo_audit.external_validation.nucls import load_frozen_nucls_config
from histo_audit.external_validation.nucls_development import (
    analyze_current_aanca,
    render_current_aanca_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    frozen_config, _ = load_frozen_nucls_config(repository_root)
    development_config = yaml.safe_load(
        (repository_root / "configs/nucls_current_aanca_improvement.yaml").read_text(
            encoding="utf-8"
        )
    )
    result = analyze_current_aanca(repository_root, frozen_config, development_config)
    if args.format == "markdown":
        print(render_current_aanca_report(result), end="")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
