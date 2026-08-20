from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import capsule_builder
from histo_audit.workflows import original_confirmatory_technical_authority_v1 as t0

PROJECT_ROOT = Path(
    os.environ.get("AANCA_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
ZERO = "0" * 64
SOURCE_ROOT = "1" * 64
CAPSULE_SHA = "2" * 64


def _process(tmp_path: Path, *, pid: int, created: str) -> dict[str, Any]:
    return {
        "process_id": pid,
        "process_created_at_utc": created,
        "executable_path": str((tmp_path / f"python-{pid}.exe").resolve()),
        "executable_size_bytes": 123,
        "executable_sha256": f"{pid % 10}" * 64,
        "implementation_path": str((tmp_path / f"builder-{pid}.py").resolve()),
        "implementation_sha256": f"{(pid + 1) % 10}" * 64,
    }


def _source_inventory() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "root_sha256": SOURCE_ROOT,
        "artifacts": [
            {
                "path": "src/example.py",
                "size_bytes": 7,
                "sha256": "3" * 64,
            }
        ],
    }


def _intent_inputs(tmp_path: Path) -> dict[str, Any]:
    inventory = _source_inventory()
    # The source file is a canonical JSON line, so bind the newline as production does.
    source_manifest_sha = hashlib.sha256(t0.canonical_json_line_bytes(inventory)).hexdigest()
    parent = (
        tmp_path / "artifacts" / "preregistration_amendments" / "20260727T133947.089370Z"
    ).resolve()
    primary = (tmp_path / t0.HISTORICAL_PRIMARY_RUN_ID).resolve()
    capsule = (
        tmp_path / "artifacts" / "execution_capsules" / CAPSULE_SHA / "original_confirmatory.pyz"
    ).resolve()
    return {
        "created_at_utc": "2026-07-31T00:00:00.000000Z",
        "builder_process": _process(tmp_path, pid=101, created="2026-07-30T23:59:59.000000Z"),
        "parent": {
            "schema_version": 1,
            "authority_kind": "preregistration_amendment",
            "authority_directory": str(parent),
            "chain_depth": t0.PARENT_CHAIN_DEPTH,
            "artifact_root_sha256": t0.PARENT_ARTIFACT_ROOT_SHA256,
            "sha256_manifest_sha256": t0.PARENT_MANIFEST_SHA256,
            "execution_source_root_sha256": t0.PARENT_SOURCE_ROOT_SHA256,
            "execution_source_manifest_sha256": t0.PARENT_SOURCE_MANIFEST_SHA256,
        },
        "frozen_science": {
            "schema_version": 1,
            "preregistration_path": str((PROJECT_ROOT / "PRE_REGISTRATION.md").resolve()),
            "preregistration_sha256": t0.PREREGISTRATION_SHA256,
            "primary_config_path": str((PROJECT_ROOT / "configs/primary.yaml").resolve()),
            "primary_config_sha256": t0.PRIMARY_CONFIG_SHA256,
            "primary_config_semantic_sha256": t0.PRIMARY_CONFIG_SEMANTIC_SHA256,
            "confirmatory_config_path": str((PROJECT_ROOT / "configs/confirmatory.yaml").resolve()),
            "confirmatory_config_sha256": t0.CONFIRMATORY_CONFIG_SHA256,
            "confirmatory_config_semantic_sha256": (t0.CONFIRMATORY_CONFIG_SEMANTIC_SHA256),
            "scientific_definition_changed": False,
        },
        "historical_primary": {
            "schema_version": 1,
            "run_directory": str(primary),
            "run_id": t0.HISTORICAL_PRIMARY_RUN_ID,
            "terminal_status": "completed",
            "completion_stage": "PRIMARY_STUDY_COMPLETE",
            "artifact_root_sha256": t0.HISTORICAL_PRIMARY_ARTIFACT_ROOT_SHA256,
            "artifact_manifest_sha256": t0.HISTORICAL_PRIMARY_MANIFEST_SHA256,
            "required_cell_count": 185,
            "completed_required_cell_count": 185,
            "skipped_optional_cell_count": 37,
            "failed_required_cell_count": 0,
            "retrained_cell_count": 0,
            "verification_scope": (
                "integrity_and_control_metadata_only_no_scientific_outcome_values"
            ),
            "outcome_values_read": False,
        },
        "execution_source": {
            "schema_version": 1,
            "policy": "runtracker_capture_source_tree_exact_object_v1",
            "manifest_path": str((tmp_path / "source_inventory.json").resolve()),
            "manifest_sha256": source_manifest_sha,
            "root_sha256": SOURCE_ROOT,
            "record_count": 1,
        },
        "execution_capsule": {
            "schema_version": 1,
            "policy": "content_addressed_original_confirmatory_execution_capsule_v1",
            "path": str(capsule),
            "size_bytes": 1234,
            "sha256": CAPSULE_SHA,
            "internal_manifest_sha256": "4" * 64,
            "source_records_root_sha256": "5" * 64,
            "publication_receipt_path": str((tmp_path / "capsule-publication.json").resolve()),
            "publication_receipt_sha256": "6" * 64,
            "independent_readback_path": str((tmp_path / "capsule-readback.json").resolve()),
            "independent_readback_sha256": "7" * 64,
            "content_addressed_create_new_verified": True,
            "scientific_execution_performed": False,
        },
        "capacity_v2": {
            "schema_version": t0.CAPACITY_SCHEMA_VERSION,
            "policy": t0.CAPACITY_POLICY_NAME,
            "policy_sha256": t0.CAPACITY_POLICY_SHA256,
            "receipt_path": str((tmp_path / "capacity.json").resolve()),
            "receipt_sha256": "8" * 64,
            "required_free_bytes": t0.CAPACITY_REQUIRED_FREE_BYTES,
            "observed_free_bytes": t0.CAPACITY_REQUIRED_FREE_BYTES + 1,
            "passed": True,
            "capsule_sha256": CAPSULE_SHA,
            "execution_source_root_sha256": SOURCE_ROOT,
            "outcome_values_read": False,
            "scientific_execution_performed": False,
        },
        "outcome_scope": {
            "schema_version": 1,
            "primary_outcomes_inspected": True,
            "primary_outcomes_inspected_at_utc": (t0.PRIMARY_OUTCOME_INSPECTION_AT_UTC),
            "primary_analysis_disposition": "amended_or_exploratory",
            "confirmatory_outcomes_inspected": False,
            "confirmatory_outcome_values_read": False,
            "confirmatory_registration_status": ("original_frozen_confirmatory_unchanged"),
            "selection_performed": False,
            "tuning_performed": False,
            "scientific_execution_performed": False,
            "automatic_retry_allowed": False,
        },
    }


def _build(
    tmp_path: Path,
) -> tuple[t0.OriginalConfirmatoryTechnicalAuthorityBundle, dict[str, Any], dict[str, Any]]:
    inputs = _intent_inputs(tmp_path)
    intent = t0.build_original_confirmatory_technical_authority_intent_v1(**inputs)
    review = t0.build_original_confirmatory_technical_authority_review_v1(
        intent=intent,
        review_started_at_utc="2026-07-31T00:00:01.000000Z",
        review_completed_at_utc="2026-07-31T00:00:02.000000Z",
        reviewer_process=_process(tmp_path, pid=202, created="2026-07-31T00:00:00.500000Z"),
    )
    bundle = t0.build_original_confirmatory_technical_authority_bundle_v1(
        authority_directory=(tmp_path / "authority").resolve(),
        intent=intent,
        independent_review=review,
        publication_timestamp_utc="2026-07-31T00:00:03.000000Z",
        preregistration_bytes=(PROJECT_ROOT / "PRE_REGISTRATION.md").read_bytes(),
        primary_config_bytes=(PROJECT_ROOT / "configs/primary.yaml").read_bytes(),
        confirmatory_config_bytes=(PROJECT_ROOT / "configs/confirmatory.yaml").read_bytes(),
        source_inventory=_source_inventory(),
    )
    return bundle, intent, review


def _materialize(bundle: t0.OriginalConfirmatoryTechnicalAuthorityBundle) -> Path:
    directory = bundle.authority_directory
    directory.mkdir(parents=True)
    for name, payload in bundle.artifacts.items():
        (directory / name).write_bytes(payload)
    (directory / t0.MANIFEST_FILENAME).write_bytes(bundle.sha256_manifest_bytes)
    (directory / t0.IMMUTABLE_MARKER_FILENAME).write_bytes(bundle.immutable_marker_bytes)
    (directory / t0.ATTEMPT_FILENAME).write_bytes(bundle.publication_attempt_bytes)
    (directory / t0.SUCCESS_FILENAME).write_bytes(bundle.publication_success_bytes)
    return directory


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(t0.canonical_json_line_bytes(value))


def _reroot(value: dict[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: t0.canonical_json_sha256(unsigned)}


def test_positive_bundle_and_verify_without_live_reads(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)

    verified = t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)

    assert set(path.name for path in directory.iterdir()) == t0.QUALIFYING_FILENAMES
    assert t0.STOP_FILENAME not in t0.QUALIFYING_FILENAMES
    assert len(bundle.artifacts) == 9
    assert set(bundle.artifacts) == t0.CORE_FILENAMES
    assert verified.artifact_root_sha256 == bundle.artifact_root_sha256
    assert verified.sha256_manifest_sha256 == bundle.sha256_manifest_sha256
    assert verified.technical_authorization_sha256 == bundle.technical_authorization_sha256
    binding = verified.lifecycle_binding()
    unsigned = dict(binding)
    binding_sha = unsigned.pop("binding_sha256")
    assert binding_sha == t0.canonical_json_sha256(unsigned)
    assert binding["primary_outcomes_inspected"] is True
    assert binding["confirmatory_outcomes_inspected"] is False
    assert binding["confirmatory_outcome_values_read"] is False
    assert binding["scientific_definition_changed"] is False
    assert binding["automatic_retry_allowed"] is False


@pytest.mark.parametrize("name", ["unexpected.json", t0.STOP_FILENAME])
def test_extra_or_stop_file_fails_exact_inventory(tmp_path: Path, name: str) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    (directory / name).write_bytes(
        bundle.publication_stop_bytes if name == t0.STOP_FILENAME else b"{}\n"
    )

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="inventory is not exact terminal success",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


@pytest.mark.parametrize("missing", [t0.SUCCESS_FILENAME, t0.ATTEMPT_FILENAME])
def test_incomplete_terminal_inventory_is_ambiguous(tmp_path: Path, missing: str) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    (directory / missing).unlink()

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="inventory is not exact terminal success",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


@pytest.mark.parametrize(
    "name",
    [
        t0.INTENT_FILENAME,
        t0.REVIEW_FILENAME,
        t0.EVIDENCE_FILENAME,
        t0.PREREGISTRATION_FILENAME,
        t0.PRIMARY_CONFIG_FILENAME,
        t0.CONFIRMATORY_CONFIG_FILENAME,
        t0.SOURCE_INVENTORY_FILENAME,
        t0.CAPSULE_BINDING_FILENAME,
        t0.CAPACITY_BINDING_FILENAME,
    ],
)
def test_tampered_core_file_fails_closed(tmp_path: Path, name: str) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / name
    path.write_bytes(path.read_bytes()[:-2] + b" \n")

    with pytest.raises(t0.OriginalConfirmatoryTechnicalAuthorityError):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


def test_tampered_manifest_fails_closed_even_when_canonical(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / t0.MANIFEST_FILENAME
    value = _json(path)
    value["artifact_root_sha256"] = ZERO
    _write_json(path, value)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="manifest is stale or invalid",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


def test_tampered_marker_fails_closed_even_when_canonical(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / t0.IMMUTABLE_MARKER_FILENAME
    value = _json(path)
    value["chain_depth"] += 1
    _write_json(path, value)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="immutable marker is stale or invalid",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


def test_tampered_success_fails_with_recomputed_self_root(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / t0.SUCCESS_FILENAME
    value = _json(path)
    value["attempt_count"] = 2
    _write_json(path, _reroot(value, "success_root_sha256"))

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="publication success is stale or ambiguous",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


def test_tampered_attempt_fails_with_recomputed_self_root(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / t0.ATTEMPT_FILENAME
    value = _json(path)
    value["automatic_retry_allowed"] = True
    _write_json(path, _reroot(value, "attempt_root_sha256"))

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="publication attempt is not the exact one-use claim",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    path = directory / t0.SUCCESS_FILENAME
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'{"adoption_allowed":', b'{"x":0,"x":1,"adoption_allowed":'))

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="differs from canonical JSON",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primary_outcomes_inspected", False),
        ("primary_analysis_disposition", "confirmatory"),
        ("confirmatory_outcomes_inspected", True),
        ("confirmatory_outcome_values_read", True),
        ("selection_performed", True),
        ("tuning_performed", True),
        ("scientific_execution_performed", True),
        ("automatic_retry_allowed", True),
    ],
)
def test_invalid_outcome_split_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    inputs = _intent_inputs(tmp_path)
    inputs["outcome_scope"][field] = value

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="outcome scope is not exact",
    ):
        t0.build_original_confirmatory_technical_authority_intent_v1(**inputs)


def test_downstream_field_cannot_enter_closed_intent(tmp_path: Path) -> None:
    _, intent, _ = _build(tmp_path)
    tampered = dict(intent)
    tampered["launcher"] = {"sha256": ZERO}
    unsigned = dict(tampered)
    unsigned.pop("intent_root_sha256")
    tampered["intent_root_sha256"] = t0.canonical_json_sha256(unsigned)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="unexpected field set",
    ):
        t0.canonical_original_confirmatory_technical_authority_intent_v1(tampered)


def test_same_process_custody_is_not_independent(tmp_path: Path) -> None:
    inputs = _intent_inputs(tmp_path)
    intent = t0.build_original_confirmatory_technical_authority_intent_v1(**inputs)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="non-independent",
    ):
        t0.build_original_confirmatory_technical_authority_review_v1(
            intent=intent,
            review_started_at_utc="2026-07-31T00:00:01.000000Z",
            review_completed_at_utc="2026-07-31T00:00:02.000000Z",
            reviewer_process=inputs["builder_process"],
        )


def test_distinct_process_with_same_implementation_is_not_independent(
    tmp_path: Path,
) -> None:
    inputs = _intent_inputs(tmp_path)
    intent = t0.build_original_confirmatory_technical_authority_intent_v1(**inputs)
    reviewer = _process(
        tmp_path,
        pid=202,
        created="2026-07-31T00:00:00.500000Z",
    )
    reviewer["implementation_path"] = inputs["builder_process"]["implementation_path"]
    reviewer["implementation_sha256"] = inputs["builder_process"]["implementation_sha256"]

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="non-independent",
    ):
        t0.build_original_confirmatory_technical_authority_review_v1(
            intent=intent,
            review_started_at_utc="2026-07-31T00:00:01.000000Z",
            review_completed_at_utc="2026-07-31T00:00:02.000000Z",
            reviewer_process=reviewer,
        )


@pytest.mark.parametrize(
    ("started", "completed"),
    [
        ("2026-07-30T23:59:59.000000Z", "2026-07-31T00:00:02.000000Z"),
        ("2026-07-31T00:00:02.000000Z", "2026-07-31T00:00:01.000000Z"),
    ],
)
def test_review_timestamp_order_fails_closed(tmp_path: Path, started: str, completed: str) -> None:
    intent = t0.build_original_confirmatory_technical_authority_intent_v1(
        **_intent_inputs(tmp_path)
    )

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="stale, non-independent, or not outcome-blind",
    ):
        t0.build_original_confirmatory_technical_authority_review_v1(
            intent=intent,
            review_started_at_utc=started,
            review_completed_at_utc=completed,
            reviewer_process=_process(tmp_path, pid=202, created="2026-07-31T00:00:00.500000Z"),
        )


def test_publication_must_be_strictly_after_review(tmp_path: Path) -> None:
    _, intent, review = _build(tmp_path)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="publication timestamp must be chosen after review completion",
    ):
        t0.build_original_confirmatory_technical_authority_bundle_v1(
            authority_directory=(tmp_path / "second-authority").resolve(),
            intent=intent,
            independent_review=review,
            publication_timestamp_utc="2026-07-31T00:00:02.000000Z",
            preregistration_bytes=(PROJECT_ROOT / "PRE_REGISTRATION.md").read_bytes(),
            primary_config_bytes=(PROJECT_ROOT / "configs/primary.yaml").read_bytes(),
            confirmatory_config_bytes=(PROJECT_ROOT / "configs/confirmatory.yaml").read_bytes(),
            source_inventory=_source_inventory(),
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("capacity_v2", "capsule_sha256", "9" * 64, "acyclic policy"),
        ("capacity_v2", "execution_source_root_sha256", "9" * 64, "acyclic policy"),
        (
            "capacity_v2",
            "observed_free_bytes",
            t0.CAPACITY_REQUIRED_FREE_BYTES - 1,
            "70-GiB gate",
        ),
        (
            "execution_capsule",
            "path",
            str((PROJECT_ROOT / "wrong" / "original_confirmatory.pyz").resolve()),
            "not content-addressed",
        ),
    ],
)
def test_capsule_capacity_source_cross_binding_fails_closed(
    tmp_path: Path, target: str, field: str, value: object, match: str
) -> None:
    inputs = _intent_inputs(tmp_path)
    inputs[target][field] = value

    with pytest.raises(t0.OriginalConfirmatoryTechnicalAuthorityError, match=match):
        t0.build_original_confirmatory_technical_authority_intent_v1(**inputs)


@pytest.mark.parametrize("mutation", ["root", "count", "bytes"])
def test_source_snapshot_cross_binding_fails_closed(tmp_path: Path, mutation: str) -> None:
    _, intent, review = _build(tmp_path)
    inventory = _source_inventory()
    if mutation == "root":
        inventory["root_sha256"] = "9" * 64
    elif mutation == "count":
        inventory["artifacts"].append(copy.deepcopy(inventory["artifacts"][0]))
    else:
        inventory["extra"] = "changes-canonical-bytes"

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="source inventory differs",
    ):
        t0.build_original_confirmatory_technical_authority_bundle_v1(
            authority_directory=(tmp_path / f"bad-source-{mutation}").resolve(),
            intent=intent,
            independent_review=review,
            publication_timestamp_utc="2026-07-31T00:00:03.000000Z",
            preregistration_bytes=(PROJECT_ROOT / "PRE_REGISTRATION.md").read_bytes(),
            primary_config_bytes=(PROJECT_ROOT / "configs/primary.yaml").read_bytes(),
            confirmatory_config_bytes=(PROJECT_ROOT / "configs/confirmatory.yaml").read_bytes(),
            source_inventory=inventory,
        )


def test_live_verification_requires_explicit_project_root(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="requires project_root",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(directory)


def test_live_verification_rejects_non_live_parent_before_other_reads(
    tmp_path: Path,
) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="parent path is not exact live P",
    ):
        t0.verify_original_confirmatory_technical_authority_v1(
            directory,
            project_root=PROJECT_ROOT,
            verify_live=True,
        )


def test_verify_is_read_only(tmp_path: Path) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    before = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()
    }

    t0.verify_original_confirmatory_technical_authority_v1(directory, verify_live=False)

    after = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()
    }
    assert after == before


def _semantic_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _semantic_sha256_file(path: Path) -> str:
    return _semantic_sha256_bytes(path.read_bytes())


def _semantic_payload_member(
    path: Path,
    *,
    relative_path: str,
    role: str,
) -> capsule_builder.PayloadMember:
    payload = path.read_bytes()
    value = path.stat(follow_symlinks=False)
    identity = capsule_builder.FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size_bytes=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
        link_count=int(value.st_nlink),
        file_attributes=int(getattr(value, "st_file_attributes", 0)),
    )
    return capsule_builder.PayloadMember(
        source_path=path,
        relative_path=relative_path,
        role=role,
        size_bytes=len(payload),
        sha256=_semantic_sha256_bytes(payload),
        payload=payload,
        identity=identity,
    )


def _semantic_process_record(
    *,
    implementation_path: Path,
    process_id: int,
    process_created_at_utc: str,
) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "process_id": process_id,
        "process_created_at_utc": process_created_at_utc,
        "executable_path": str(executable),
        "executable_size_bytes": executable.stat(follow_symlinks=False).st_size,
        "executable_sha256": _semantic_sha256_file(executable),
        "implementation_path": str(implementation_path.resolve()),
        "implementation_sha256": _semantic_sha256_file(implementation_path),
    }


def _semantic_fixture(tmp_path: Path) -> dict[str, Any]:
    package = tmp_path / "src" / "histo_audit"
    package.mkdir(parents=True)
    project_source = package / "example.py"
    bootstrap = tmp_path / "capsule_bootstrap.py"
    policy = tmp_path / "capsule_policy.json"
    contract = tmp_path / "entry_contract.json"
    project_source.write_bytes(b"x = 1\n")
    bootstrap.write_bytes(b"raise SystemExit(0)\n")
    policy.write_bytes(b"{}\n")
    contract.write_bytes(b"{}\n")

    members = (
        _semantic_payload_member(
            project_source,
            relative_path="histo_audit/example.py",
            role="project_source",
        ),
        _semantic_payload_member(
            bootstrap,
            relative_path="__main__.py",
            role="capsule_bootstrap",
        ),
        _semantic_payload_member(
            policy,
            relative_path="aanca_capsule/capsule_policy.json",
            role="capsule_policy",
        ),
        _semantic_payload_member(
            contract,
            relative_path="aanca_capsule/entry_contract.json",
            role="capsule_contract",
        ),
    )
    inventory = capsule_builder.source_inventory(members)
    build = capsule_builder.build_capsule_bytes(
        members=members,
        expected_inventory=inventory,
    )
    capsule_path = tmp_path / "original_confirmatory.pyz"
    capsule_path.write_bytes(build.archive_bytes)
    capsule: dict[str, Any] = {
        "path": str(capsule_path.resolve()),
        "size_bytes": build.size_bytes,
        "sha256": build.sha256,
        "internal_manifest_sha256": build.internal_manifest_sha256,
        "source_records_root_sha256": build.records_root_sha256,
    }
    source_inventory = {
        "artifacts": [
            {
                "path": "src/histo_audit/example.py",
                "size_bytes": project_source.stat(follow_symlinks=False).st_size,
                "sha256": _semantic_sha256_file(project_source),
            }
        ]
    }

    publisher_source = tmp_path / "capsule_publisher.py"
    reviewer_source = tmp_path / "capsule_reviewer.py"
    publisher_source.write_bytes(b"publisher = True\n")
    reviewer_source.write_bytes(b"reviewer = True\n")
    publisher_process = _semantic_process_record(
        implementation_path=publisher_source,
        process_id=101,
        process_created_at_utc="2026-07-31T00:00:00.000000Z",
    )
    reviewer_process = _semantic_process_record(
        implementation_path=reviewer_source,
        process_id=202,
        process_created_at_utc="2026-07-31T00:00:02.000000Z",
    )
    publication_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_execution_capsule_publication_receipt_v1",
        "published_at_utc": "2026-07-31T00:00:01.000000Z",
        "publisher_process": publisher_process,
        "capsule_path": capsule["path"],
        "capsule_size_bytes": capsule["size_bytes"],
        "capsule_sha256": capsule["sha256"],
        "internal_manifest_sha256": capsule["internal_manifest_sha256"],
        "source_records_root_sha256": capsule["source_records_root_sha256"],
        "creation_disposition": "CREATE_NEW",
        "same_handle_readback_verified": True,
        "archive_integrity_verified": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    publication = _reroot(publication_unsigned, "receipt_root_sha256")
    publication_path = tmp_path / "capsule_publication.json"
    _write_json(publication_path, publication)
    publication_sha256 = _semantic_sha256_file(publication_path)
    readback_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_execution_capsule_independent_readback_v1",
        "verified_at_utc": "2026-07-31T00:00:03.000000Z",
        "reviewer_process": reviewer_process,
        "publication_receipt_sha256": publication_sha256,
        "capsule_path": capsule["path"],
        "capsule_size_bytes": capsule["size_bytes"],
        "capsule_sha256": capsule["sha256"],
        "internal_manifest_sha256": capsule["internal_manifest_sha256"],
        "source_records_root_sha256": capsule["source_records_root_sha256"],
        "byte_readback_verified": True,
        "archive_integrity_verified": True,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "automatic_retry_allowed": False,
    }
    readback = _reroot(readback_unsigned, "readback_root_sha256")
    readback_path = tmp_path / "capsule_readback.json"
    _write_json(readback_path, readback)
    capsule.update(
        {
            "publication_receipt_path": str(publication_path.resolve()),
            "publication_receipt_sha256": publication_sha256,
            "independent_readback_path": str(readback_path.resolve()),
            "independent_readback_sha256": _semantic_sha256_file(readback_path),
        }
    )

    capacity_unsigned = {
        "schema_version": 2,
        "policy": t0.CAPACITY_POLICY_NAME,
        "policy_sha256": t0.CAPACITY_POLICY_SHA256,
        "checked_at_utc": "2026-07-31T00:00:04.000000Z",
        "phase": "before_technical_authority",
        "planned_cell_count": 108,
        "planned_required_cell_count": 90,
        "planned_optional_cell_count": 18,
        "planned_cnn_cell_count": 36,
        "planned_cnn_fold_checkpoint_count": 180,
        "checkpoint_physical_copy_count": 2,
        "projected_checkpoint_bytes_per_physical_copy": 30 * 1024**3,
        "projected_all_checkpoint_copies_bytes": 60 * 1024**3,
        "safety_margin_bytes": 10 * 1024**3,
        "required_free_bytes": t0.CAPACITY_REQUIRED_FREE_BYTES,
        "observed_free_bytes": t0.CAPACITY_REQUIRED_FREE_BYTES + 1,
        "passed": True,
        "capsule_sha256": capsule["sha256"],
        "execution_source_root_sha256": "1" * 64,
        "outcome_values_read": False,
        "scientific_execution_performed": False,
        "adaptive_execution_changes_allowed": False,
    }
    capacity = _reroot(capacity_unsigned, "capacity_receipt_root_sha256")
    capacity_path = tmp_path / "capacity.json"
    _write_json(capacity_path, capacity)
    capacity_binding = {
        "receipt_path": str(capacity_path.resolve()),
        "observed_free_bytes": capacity_unsigned["observed_free_bytes"],
        "capsule_sha256": capsule["sha256"],
        "execution_source_root_sha256": capacity_unsigned["execution_source_root_sha256"],
    }
    return {
        "capsule": capsule,
        "capsule_path": capsule_path,
        "source_inventory": source_inventory,
        "publication": publication,
        "publication_path": publication_path,
        "readback": readback,
        "readback_path": readback_path,
        "capacity": capacity,
        "capacity_path": capacity_path,
        "capacity_binding": capacity_binding,
    }


def test_semantic_fixed_zip_and_source_alignment_positive(tmp_path: Path) -> None:
    fixture = _semantic_fixture(tmp_path)
    records = t0._verify_capsule_archive(
        fixture["capsule_path"],
        capsule=fixture["capsule"],
    )

    t0._verify_capsule_source_alignment(
        records=records,
        source_inventory=fixture["source_inventory"],
        project_root=tmp_path,
    )


def test_semantic_fixed_zip_bad_records_root_fails_closed(tmp_path: Path) -> None:
    fixture = _semantic_fixture(tmp_path)
    capsule = {**fixture["capsule"], "source_records_root_sha256": ZERO}

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="source root",
    ):
        t0._verify_capsule_archive(
            fixture["capsule_path"],
            capsule=capsule,
        )


def test_semantic_capsule_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _semantic_fixture(tmp_path)
    records = t0._verify_capsule_archive(
        fixture["capsule_path"],
        capsule=fixture["capsule"],
    )
    source_inventory = copy.deepcopy(fixture["source_inventory"])
    source_inventory["artifacts"][0]["sha256"] = ZERO

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="authority-bound source tree",
    ):
        t0._verify_capsule_source_alignment(
            records=records,
            source_inventory=source_inventory,
            project_root=tmp_path,
        )


def test_semantic_capsule_receipts_and_capacity_positive(tmp_path: Path) -> None:
    fixture = _semantic_fixture(tmp_path)

    t0._verify_capsule_receipts(fixture["capsule"])
    t0._verify_capacity_receipt(fixture["capacity_binding"])


@pytest.mark.parametrize(
    "mutation",
    [
        "publication_root",
        "readback_schema_type",
        "reviewer_implementation_hash",
        "same_implementation",
    ],
)
def test_semantic_capsule_receipt_negatives_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _semantic_fixture(tmp_path)
    if mutation == "publication_root":
        publication = {
            **fixture["publication"],
            "receipt_root_sha256": ZERO,
        }
        _write_json(fixture["publication_path"], publication)
        match = "self-root"
    else:
        readback = dict(fixture["readback"])
        if mutation == "readback_schema_type":
            readback["schema_version"] = 1.0
            match = "not exact and independent"
        elif mutation == "reviewer_implementation_hash":
            process = dict(readback["reviewer_process"])
            process["implementation_sha256"] = ZERO
            readback["reviewer_process"] = process
            match = "implementation changed"
        else:
            process = dict(readback["reviewer_process"])
            publisher = fixture["publication"]["publisher_process"]
            process["implementation_path"] = publisher["implementation_path"]
            process["implementation_sha256"] = publisher["implementation_sha256"]
            readback["reviewer_process"] = process
            match = "not exact and independent"
        _write_json(
            fixture["readback_path"],
            _reroot(readback, "readback_root_sha256"),
        )

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match=match,
    ):
        t0._verify_capsule_receipts(fixture["capsule"])


@pytest.mark.parametrize("mutation", ["root", "numeric_type"])
def test_semantic_capacity_receipt_negatives_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _semantic_fixture(tmp_path)
    capacity = dict(fixture["capacity"])
    if mutation == "root":
        capacity["capacity_receipt_root_sha256"] = ZERO
        match = "self-root"
    else:
        capacity["planned_cell_count"] = 108.0
        capacity = _reroot(capacity, "capacity_receipt_root_sha256")
        match = "exact sealed arithmetic"
    _write_json(fixture["capacity_path"], capacity)

    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match=match,
    ):
        t0._verify_capacity_receipt(fixture["capacity_binding"])


@pytest.mark.skipif(os.name != "nt", reason="Windows ADS syntax")
def test_semantic_absolute_path_rejects_windows_ads() -> None:
    with pytest.raises(
        t0.OriginalConfirmatoryTechnicalAuthorityError,
        match="absolute canonical path",
    ):
        t0._absolute_path(
            r"C:\Users\NATAN\Documents\AANCA\receipt.json:alternate",
            role="synthetic receipt",
        )


def test_return_identity_is_frozen_before_final_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, _ = _build(tmp_path)
    directory = _materialize(bundle)
    original_snapshot = t0._directory_snapshot
    call_count = 0

    def mutate_after_snapshot(path: Path) -> tuple[tuple[str, int, str], ...]:
        nonlocal call_count
        call_count += 1
        snapshot = original_snapshot(path)
        if call_count == 2:
            marker = path / t0.IMMUTABLE_MARKER_FILENAME
            marker.write_bytes(marker.read_bytes() + b" ")
        return snapshot

    monkeypatch.setattr(t0, "_directory_snapshot", mutate_after_snapshot)

    verified = t0.verify_original_confirmatory_technical_authority_v1(
        directory,
        verify_live=False,
    )

    expected_marker_sha256 = hashlib.sha256(bundle.immutable_marker_bytes).hexdigest()
    assert call_count == 2
    assert verified.immutable_marker_sha256 == expected_marker_sha256
    assert (
        hashlib.sha256((directory / t0.IMMUTABLE_MARKER_FILENAME).read_bytes()).hexdigest()
        != expected_marker_sha256
    )
