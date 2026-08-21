from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from histo_audit.research.autoresearch import AutoresearchCandidate
from histo_audit.research.frozen_candidate import load_frozen_development_candidate
from histo_audit.utils.run_tracking import sha256_file


def _record() -> dict[str, object]:
    candidate = AutoresearchCandidate(intervention="flag_exclude")
    return {
        "schema_version": 1,
        "record_id": "test",
        "project": "AANCA",
        "replacement_project_or_v2": False,
        "role": "development_candidate_only",
        "candidate": candidate.as_dict(),
        "candidate_sha256": candidate.candidate_sha256,
        "selection_authority": {
            "parent_study_id": "development",
            "parent_config_sha256": "1" * 64,
            "partition_sha256": "2" * 64,
            "runtime_amendment_id": "amendment",
            "runtime_amendment_sha256": "3" * 64,
            "parent_authority_sha256": "5" * 64,
            "parent_ledger_sha256": "4" * 64,
        },
        "development_evidence": {
            "final_external_test_used": False,
            "natural_error_detection_evaluated": False,
        },
        "claim_boundary": {
            "natural_error_detection_proven": False,
            "real_use_superiority_proven": False,
            "automatic_annotation_change_permitted": False,
            "source_annotations_modified": False,
        },
        "activation": {
            "external_confirmation_complete": False,
            "executable_action_until_new_confirmation": "retain_uncorrected",
        },
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_frozen_candidate_loads_but_cannot_activate_on_natural_data(tmp_path: Path) -> None:
    path = tmp_path / "candidate.yaml"
    _write(path, _record())
    result = load_frozen_development_candidate(path)

    assert result.candidate.intervention == "flag_exclude"
    assert result.natural_data_activation_permitted is False
    assert result.executable_action_until_new_confirmation == "retain_uncorrected"


def test_checked_in_selected_candidate_is_checksum_frozen_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "aanca_selected_development_candidate.yaml"
    result = load_frozen_development_candidate(path)

    assert result.candidate_sha256 == (
        "78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe"
    )
    assert result.selection_authority["parent_authority_sha256"] == (
        "3ef82963925cea7d20332f13488578ded5eba1df750c52cb55cef69521580042"
    )
    assert result.natural_data_activation_permitted is False
    assert result.executable_action_until_new_confirmation == "retain_uncorrected"


def test_frozen_candidate_rejects_identity_or_claim_inflation(tmp_path: Path) -> None:
    value = _record()
    value["candidate_sha256"] = "0" * 64
    path = tmp_path / "wrong-hash.yaml"
    _write(path, value)
    with pytest.raises(ValueError, match="identity"):
        load_frozen_development_candidate(path)

    value = _record()
    value["claim_boundary"]["real_use_superiority_proven"] = True  # type: ignore[index]
    path = tmp_path / "inflated-claim.yaml"
    _write(path, value)
    with pytest.raises(ValueError, match="overstates"):
        load_frozen_development_candidate(path)


def test_frozen_candidate_rejects_a_changed_checksummed_record(tmp_path: Path) -> None:
    path = tmp_path / "candidate.yaml"
    _write(path, _record())
    checksum = path.with_suffix(path.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")

    loaded = load_frozen_development_candidate(path)
    assert loaded.candidate_sha256

    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_development_candidate(path)
