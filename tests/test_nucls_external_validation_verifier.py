from __future__ import annotations

import importlib.util
from pathlib import Path


def test_published_nucls_evidence_recalculates_independently() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "scripts" / "verify_nucls_external_validation.py"
    spec = importlib.util.spec_from_file_location("verify_nucls_external_validation", script_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    result = verifier.verify_release(repository_root / "artifacts" / "nucls_external_validation")

    assert result["file_identity_status"] == "passed"
    assert result["independent_recalculation_status"] == "passed"
    assert result["primary_claim_conclusion"] == "not_supported"
    assert result["pathologist_error_proven"] is False
    assert result["clinical_utility_proven"] is False
