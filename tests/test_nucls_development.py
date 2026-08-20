from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from histo_audit.external_validation.nucls import load_frozen_nucls_config
from histo_audit.external_validation.nucls_development import (
    analyze_current_aanca,
    render_current_aanca_report,
)


def test_saved_evidence_supports_only_exploratory_primary_neighbour_result() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen, _ = load_frozen_nucls_config(root)
    development = yaml.safe_load(
        (root / "configs/nucls_current_aanca_improvement.yaml").read_text(encoding="utf-8")
    )

    result = analyze_current_aanca(root, frozen, development)

    primary = result["subsets"]["unbiased"]
    secondary = result["subsets"]["evaluation"]
    assert primary["ranking_candidates"]["nearest_neighbour_disagreement"][
        "strict_success_conditions_met"
    ]
    assert not secondary["ranking_candidates"]["nearest_neighbour_disagreement"][
        "strict_success_conditions_met"
    ]
    assert primary["ranking_candidates"]["nearest_neighbour_disagreement"][
        "average_precision"
    ] == pytest.approx(0.06891028769860284)
    assert result["candidate_promotion"]["promoted_to_new_default"] is False
    assert result["frozen_external_result_changed"] is False
    for subset in result["subsets"].values():
        assert not subset["retraining_guard"]["frozen_audit_guided_candidate"]["apply_candidate"]
        assert not subset["retraining_guard"]["full_consensus_label_candidate"]["apply_candidate"]
    report = render_current_aanca_report(result)
    assert "post-outcome exploratory" in report
    assert "Replacement project or v2:** no" in report
    assert "retain_uncorrected" in report
