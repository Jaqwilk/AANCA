from __future__ import annotations

import ast
import hashlib
import os
import stat
import sys
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest
from carrier_import_guard import PACKAGE_IMPORT_ROOT, TESTS_ROOT, import_exact

authority = import_exact(
    "histo_audit.workflows.original_confirmatory_capsule_authority",
    PACKAGE_IMPORT_ROOT
    / "histo_audit"
    / "workflows"
    / "original_confirmatory_capsule_authority.py",
)
_handoff_test_module = import_exact(
    "test_codex_handoff_authority",
    TESTS_ROOT / "test_codex_handoff_authority.py",
)
_codex_handoff_base = _handoff_test_module._base
_codex_handoff_creation = _handoff_test_module._creation

_MODULE_PATH = Path(authority.__file__).resolve()


class _EqualDict(dict[str, Any]):
    """A dict subclass whose ordinary equality hides an exact JSON type mismatch."""


class _AmbientEnvironmentSpy(Mapping[str, str]):
    """Fail on, and count, every attempted ambient-environment read."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.read_count = 0

    def _reject_read(self) -> NoReturn:
        self.read_count += 1
        raise AssertionError(f"pure authority read ambient environment containing {self._values!r}")

    def __getitem__(self, key: str) -> str:
        del key
        self._reject_read()

    def __iter__(self) -> Iterator[str]:
        self._reject_read()

    def __len__(self) -> int:
        self._reject_read()


@dataclass(frozen=True)
class CapsuleFixture:
    anchor: Path
    capsule_path: Path
    capsule_bytes: bytes
    capsule: authority.OriginalConfirmatoryExecutionCapsule
    leaf_lease: authority.OriginalConfirmatoryCapsuleLeaseIdentity
    ancestor_lease: authority.OriginalConfirmatoryCapsuleAncestorLease


@pytest.fixture
def capsule_fixture(tmp_path: Path) -> CapsuleFixture:
    capsule_bytes = b"deterministic synthetic capsule v1\n"
    capsule_sha256 = hashlib.sha256(capsule_bytes).hexdigest()
    capsule_path = (
        tmp_path / "artifacts" / "execution_capsules" / capsule_sha256 / "original_confirmatory.pyz"
    )
    capsule_path.parent.mkdir(parents=True)
    capsule_path.write_bytes(capsule_bytes)
    capsule_path.chmod(stat.S_IREAD)
    physical, _ = authority._read_stable_file(
        capsule_path,
        maximum_bytes=1024,
    )
    leaf_lease = authority.build_original_confirmatory_capsule_lease_identity(
        path=capsule_path,
        volume_serial_number=physical.volume_serial_number,
        file_id_128=physical.file_id_128,
        size_bytes=physical.size_bytes,
        sha256=physical.sha256,
        file_attributes=physical.file_attributes,
    )
    ancestor_paths = (
        tmp_path,
        tmp_path / "artifacts",
        tmp_path / "artifacts" / "execution_capsules",
        capsule_path.parent,
    )
    ancestor_records = []
    for path in ancestor_paths:
        handle, native, facts = authority._open_e_job_directory_handle(path)
        authority._close_e_job_directory_handle(handle, windows_native=native)
        ancestor_records.append(
            {
                "path": str(path),
                "volume_serial_number": facts[0],
                "file_id_128": facts[1],
                "file_attributes": facts[2],
                "reparse_point": False,
            }
        )
    ancestor_lease = authority.build_original_confirmatory_capsule_ancestor_lease(
        anchor_path=tmp_path,
        records=ancestor_records,
    )
    source_python_path = Path(sys.executable)
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(source_python_path.read_bytes())
    python_physical, _ = authority._read_stable_file(
        python_path,
        maximum_bytes=python_path.stat().st_size,
    )
    python_lease = authority.build_original_confirmatory_interpreter_lease_identity(
        path=python_path,
        volume_serial_number=python_physical.volume_serial_number,
        file_id_128=python_physical.file_id_128,
        size_bytes=python_physical.size_bytes,
        sha256=python_physical.sha256,
        file_attributes=python_physical.file_attributes,
    )
    python_anchor = tmp_path
    python_ancestor_paths = (
        python_anchor,
        python_path.parent.parent,
        python_path.parent,
    )
    python_ancestor_records = []
    for path in python_ancestor_paths:
        handle, native, facts = authority._open_e_job_directory_handle(path)
        authority._close_e_job_directory_handle(handle, windows_native=native)
        python_ancestor_records.append(
            {
                "path": str(path),
                "volume_serial_number": facts[0],
                "file_id_128": facts[1],
                "file_attributes": facts[2],
                "reparse_point": False,
            }
        )
    python_ancestor = authority.build_original_confirmatory_interpreter_ancestor_lease(
        anchor_path=python_anchor,
        records=python_ancestor_records,
    )
    runtime_python_path = Path(sys._base_executable)
    runtime_python_physical, _ = authority._read_stable_file(
        runtime_python_path,
        maximum_bytes=runtime_python_path.stat().st_size,
    )
    runtime_python_lease = authority.build_original_confirmatory_runtime_interpreter_lease_identity(
        path=runtime_python_path,
        volume_serial_number=runtime_python_physical.volume_serial_number,
        file_id_128=runtime_python_physical.file_id_128,
        size_bytes=runtime_python_physical.size_bytes,
        sha256=runtime_python_physical.sha256,
        file_attributes=runtime_python_physical.file_attributes,
    )
    runtime_user_root = Path(os.environ.get("USERPROFILE", str(Path.home())))
    runtime_relative_parent = runtime_python_path.parent.relative_to(runtime_user_root)
    runtime_ancestor_paths = [runtime_user_root]
    for part in runtime_relative_parent.parts:
        runtime_ancestor_paths.append(runtime_ancestor_paths[-1] / part)
    runtime_ancestor_records = []
    for path in runtime_ancestor_paths:
        handle, native, facts = authority._open_e_job_directory_handle(path)
        authority._close_e_job_directory_handle(handle, windows_native=native)
        runtime_ancestor_records.append(
            {
                "path": str(path),
                "volume_serial_number": facts[0],
                "file_id_128": facts[1],
                "file_attributes": facts[2],
                "reparse_point": False,
            }
        )
    runtime_python_ancestor = (
        authority.build_original_confirmatory_runtime_interpreter_ancestor_lease(
            anchor_path=runtime_user_root,
            records=runtime_ancestor_records,
        )
    )
    capsule = authority.build_original_confirmatory_execution_capsule(
        path=capsule_path,
        size_bytes=len(capsule_bytes),
        sha256=capsule_sha256,
        internal_manifest_sha256="1" * 64,
        capsule_policy_sha256="2" * 64,
        entry_contract_sha256="c" * 64,
        plan_sha256="3" * 64,
        runtime_release_root_sha256="4" * 64,
        terminal_release_root_sha256="5" * 64,
        python_path=python_path,
        python_sha256=python_physical.sha256,
        python_lease_identity=python_lease,
        python_ancestor_lease=python_ancestor,
        runtime_python_path=runtime_python_path,
        runtime_python_sha256=runtime_python_physical.sha256,
        runtime_python_lease_identity=runtime_python_lease,
        runtime_python_ancestor_lease=runtime_python_ancestor,
        capsule_lease_identity=leaf_lease,
        capsule_ancestor_lease=ancestor_lease,
    )
    return CapsuleFixture(
        anchor=tmp_path,
        capsule_path=capsule_path,
        capsule_bytes=capsule_bytes,
        capsule=capsule,
        leaf_lease=leaf_lease,
        ancestor_lease=ancestor_lease,
    )


def _environment(
    nonce: str = "a" * 64,
    *,
    user_profile: str = "C:\\Users\\Researcher",
) -> authority.ExpectedLaunchEnvironmentEnvelopeV1:
    local_app_data = user_profile + "\\AppData\\Local"
    supervisor_environment = {
        "LOCALAPPDATA": local_app_data,
        "SYSTEMROOT": "C:\\Windows",
        "TEMP": local_app_data + "\\Temp",
        "TMP": local_app_data + "\\Temp",
        "USERPROFILE": user_profile,
    }
    return authority.build_expected_launch_environment_envelope_v1(
        attempt_nonce=nonce,
        supervisor_environment=supervisor_environment,
        child_environment={
            **supervisor_environment,
            "AANCA_SUPERVISOR_ATTEMPT_NONCE": nonce,
        },
    )


def _tail(
    fixture: CapsuleFixture,
    mode: str,
    *,
    execution_mode: str = "fresh",
) -> tuple[str, ...]:
    job = fixture.anchor / "supervisor-root" / "jobs" / "job-1"
    staged_e = (
        fixture.anchor
        / "supervisor-root"
        / authority.CONTROL_STAGING_DIRECTORY_NAME
        / "job-1"
        / authority.E_INTENT_FILENAME
    )
    common: dict[str, Any] = {
        "capsule_mode": mode,
        "e_intent_path": staged_e,
        "e_intent_sha256": "6" * 64,
        "e_intent_core_sha256": "7" * 64,
        "q_authority_root_sha256": "8" * 64,
        "launch_nonce": "a" * 64,
        "supervisor_job_id": "job-1",
        "supervisor_job_directory": job,
        "attempt_id": "attempt-1",
        "run_id": "run-1",
        "execution_mode": execution_mode,
        "retry_of_run_id": "source-run-1" if execution_mode == "successor_resume" else None,
    }
    if mode == authority.CAPSULE_PRETERMINAL_MODE:
        common.update(
            {
                "run_spec_path": fixture.anchor / "run" / "run_spec.json",
                "launch_intent_path": job / "launch_intent.json",
                "process_started_path": job / "process_started.json",
                "preterminal_pin_path": job / "preterminal_pin.json",
            }
        )
    elif mode == authority.CAPSULE_TERMINAL_MODE:
        common.update(
            {
                "supervisor_terminal_path": job / "terminal_receipt.json",
                "verifier_stdout_path": job / "verifier.stdout.log",
                "preterminal_pin_path": job / "preterminal_pin.json",
                "composed_terminal_path": job / "composed_terminal.json",
            }
        )
    return authority.original_confirmatory_capsule_mode_tail(**common)


def _rehash_without_self(value: dict[str, Any], self_field: str) -> dict[str, Any]:
    result = dict(value)
    result[self_field] = authority.canonical_json_sha256(
        {key: item for key, item in result.items() if key != self_field}
    )
    return result


def _synthetic_physical_identity(
    tmp_path: Path,
    role: str,
) -> authority.OriginalConfirmatoryPhysicalFileIdentity:
    content = b"" if role == "preterminal-stderr" else f"{role}\n".encode()
    return authority.build_original_confirmatory_physical_file_identity(
        role=role,
        path=tmp_path / f"{role}.json",
        volume_serial_number=1,
        file_id_128="1" * 32,
        device=1,
        inode=2,
        size_bytes=len(content),
        mode=stat.S_IREAD,
        file_attributes=0x1,
        modified_time_ns=3,
        changed_time_ns=4,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _terminal_dependencies(
    fixture: CapsuleFixture,
    command: authority.OriginalConfirmatoryCapsuleCommand,
    environment_binding: authority.OriginalConfirmatoryProcessEnvironmentBinding,
    *,
    job_dir: Path | None = None,
) -> tuple[
    authority.OriginalConfirmatoryPreterminalOverlapHandshakeContract,
    authority.OriginalConfirmatoryPostwakeInputLeaseContract,
    authority.OriginalConfirmatoryPostwakeCustodySeed,
    authority.OriginalConfirmatoryPostwakeCustodyHandshakeContract,
]:
    job_dir = fixture.anchor / "supervisor-root" / "jobs" / "job-1" if job_dir is None else job_dir
    argv = list(command.argv)

    def argument(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    job_id = argument("--supervisor-job-id")
    attempt_id = argument("--attempt-id")
    run_id = argument("--run-id")
    launch_nonce = argument("--launch-nonce")
    execution_mode = argument("--execution-mode")
    retry_of_run_id = argument("--retry-of-run-id") if "--retry-of-run-id" in argv else None
    ancestor = authority.build_original_confirmatory_supervisor_job_ancestor_lease_contract(
        supervisor_root=job_dir.parent.parent,
        job_id=job_id,
    )
    overlap = authority.build_original_confirmatory_preterminal_overlap_handshake_contract(
        handshake_receipt_path=job_dir / "preterminal_overlap_handshake.json",
        ready_line_max_bytes=64 * 1024,
        ack_line_max_bytes=64 * 1024,
        handshake_timeout_ms=30_000,
    )
    input_lease = authority.build_original_confirmatory_postwake_input_lease_contract(
        supervisor_job_dir=job_dir,
        preterminal_pin_path=job_dir / "preterminal_pin.json",
        verifier_stdout_path=job_dir / "verifier.stdout.log",
        verifier_log_max_bytes=64 * 1024,
        terminal_receipt_path=job_dir / "terminal_receipt.json",
        lease_receipt_path=job_dir / "postwake_input_lease_receipt.json",
        supervisor_job_ancestor_lease_contract=ancestor,
    )
    seed = authority.build_original_confirmatory_postwake_custody_seed(
        q_authority_root_sha256="8" * 64,
        e_intent_path=(
            job_dir.parent.parent
            / authority.CONTROL_STAGING_DIRECTORY_NAME
            / job_id
            / authority.E_INTENT_FILENAME
        ),
        e_intent_file_sha256="6" * 64,
        e_intent_core_sha256="7" * 64,
        supervisor_job_id=job_id,
        supervisor_job_dir=job_dir,
        supervisor_spec_path=job_dir / "run_spec.json",
        launch_nonce=launch_nonce,
        attempt_id=attempt_id,
        run_id=run_id,
        execution_mode=execution_mode,
        retry_of_run_id=retry_of_run_id,
        execution_capsule_contract_sha256=fixture.capsule.contract_sha256,
        capsule_sha256=fixture.capsule.sha256,
        supervisor_release_root_sha256="d" * 64,
        terminal_release_root_sha256=fixture.capsule.terminal_release_root_sha256,
        supervisor_terminal_receipt_path=job_dir / "terminal_receipt.json",
        preterminal_pin_receipt_path=job_dir / "preterminal_pin.json",
        postwake_input_lease_receipt_path=job_dir / "postwake_input_lease_receipt.json",
        composed_terminal_receipt_path=job_dir / "composed_terminal.json",
        postwake_composed_readback_receipt_path=job_dir / "postwake_composed_readback_receipt.json",
    )
    custody = authority.build_original_confirmatory_postwake_custody_handshake_contract(
        custody_seed=seed,
        pipe_owner_sid="S-1-5-21-1000",
        readback_receipt_path=job_dir / "postwake_composed_readback_receipt.json",
        expected_composed_command_sha256=command.command_sha256,
        expected_composed_cwd=command.cwd,
        expected_composed_environment_sha256=(
            environment_binding.exact_integrity_verifier_environment_sha256
        ),
        ready_max_bytes=64 * 1024,
        ack_max_bytes=64 * 1024,
        terminal_client_arrival_timeout_ms=1_800_000,
        custody_exchange_timeout_ms=60_000,
    )
    return overlap, input_lease, seed, custody


def _terminal_custody_projection_for_command(
    fixture: CapsuleFixture,
    *,
    q: Mapping[str, Any],
    command: authority.OriginalConfirmatoryCapsuleCommand,
    tail: tuple[str, ...],
    environment_binding: authority.OriginalConfirmatoryProcessEnvironmentBinding,
    expected_run_directory: Path,
) -> dict[str, Any]:
    command_projection = (
        authority._derive_original_confirmatory_terminal_command_projection_from_concrete(
            command=command,
            capsule=fixture.capsule,
            canonical_tail=tail,
        )
    )
    job_id = tail[tail.index("--supervisor-job-id") + 1]
    job_directory = Path(tail[tail.index("--supervisor-job-dir") + 1])
    return authority.build_original_confirmatory_terminal_custody_authority_projection(
        run_id=tail[tail.index("--run-id") + 1],
        expected_run_directory=expected_run_directory,
        launcher_release=q["supervisor_release"]["terminal_client_launcher_release"],
        capsule=fixture.capsule,
        supervisor_job_id=job_id,
        supervisor_job_directory=job_directory,
        verify_terminal_command_projection_sha256=(command_projection.projection_sha256),
        verify_terminal_environment_sha256=(
            environment_binding.exact_integrity_verifier_environment_sha256
        ),
        verify_terminal_cwd=command.cwd,
    )


def _published_technical_authority_lifecycle_binding(
    project_root: Path,
    *,
    artifact_root_sha256: str,
    technical_authorization_sha256: str,
) -> dict[str, Any]:
    namespace = project_root / "artifacts" / "original_confirmatory_technical_authorities"
    technical_unsigned = {
        "schema_version": 1,
        "policy": "original_confirmatory_technical_authority_lifecycle_binding_v1",
        "authority_directory": str(namespace / "synthetic-authority"),
        "chain_depth": 3,
        "artifact_root_sha256": artifact_root_sha256,
        "sha256_manifest_sha256": "a" * 64,
        "execution_source_manifest_sha256": "b" * 64,
        "execution_source_root_sha256": "c" * 64,
        "parent_authority_directory": str(project_root / "artifacts" / "freeze"),
        "parent_artifact_root_sha256": "d" * 64,
        "parent_sha256_manifest_sha256": "e" * 64,
        "technical_authorization_sha256": technical_authorization_sha256,
        "independent_review_receipt_sha256": "f" * 64,
        "immutable_marker_sha256": "1" * 64,
        "publication_attempt_sha256": "2" * 64,
        "publication_success_sha256": "4" * 64,
        "primary_outcomes_inspected": True,
        "confirmatory_outcomes_inspected": False,
        "confirmatory_outcome_values_read": False,
        "scientific_definition_changed": False,
        "automatic_retry_allowed": False,
    }
    technical = {
        **technical_unsigned,
        "binding_sha256": authority.canonical_json_sha256(technical_unsigned),
    }
    published_unsigned = {
        "schema_version": 1,
        "policy": ("published_original_confirmatory_technical_authority_lifecycle_binding_v1"),
        "namespace_directory": str(namespace),
        "namespace_claim_sha256": "5" * 64,
        "review_attempt_claim_sha256": "6" * 64,
        "technical_authority": technical,
        "automatic_retry_allowed": False,
        "adoption_allowed": False,
        "cleanup_allowed": False,
    }
    return {
        **published_unsigned,
        "binding_sha256": authority.canonical_json_sha256(published_unsigned),
    }


def _scientific_authority(fixture: CapsuleFixture) -> dict[str, Any]:
    cli_binding = authority.build_original_confirmatory_cli_input_binding(
        project_root=fixture.anchor,
        crop_cache_path=fixture.anchor / "data" / "crop-cache.npz",
        expected_crop_cache_sha256="a" * 64,
        expected_crop_metadata_sha256="b" * 64,
        expected_raw_inventory_sha256="c" * 64,
        frozen_feature_caches=(),
        observed_label_sets=(),
        draft_checkpoint_contract={
            "schema_version": 1,
            "policy": "synthetic_exact_180",
            "directive_count": 180,
        },
        bridge_binding_sha256="d" * 64,
    )
    published_binding = _published_technical_authority_lifecycle_binding(
        fixture.anchor,
        artifact_root_sha256="9" * 64,
        technical_authorization_sha256="3" * 64,
    )
    technical_authority = published_binding["technical_authority"]
    static = authority.build_original_confirmatory_static_runner_binding(
        project_root=fixture.anchor,
        primary_run_directory=fixture.anchor / "artifacts" / "runs" / "primary",
        freeze_directory=fixture.anchor / "artifacts" / "freeze",
        technical_authority_directory=technical_authority["authority_directory"],
        technical_authority_artifact_root_sha256="9" * 64,
        technical_authorization_sha256="3" * 64,
        published_technical_authority_lifecycle_binding=published_binding,
        lifecycle_readiness_run_directory=fixture.anchor / "artifacts" / "lifecycle",
        dataset_path=fixture.anchor / "data" / "pannuke.npy",
        manifest_path=fixture.anchor / "data" / "manifest.json",
        duplicate_audit_path=fixture.anchor / "artifacts" / "duplicate.json",
        pathology_encoder_audit_path=fixture.anchor / "artifacts" / "pathology.json",
        frozen_primary_config_path=fixture.anchor / "configs" / "primary.json",
        frozen_confirmatory_config_path=fixture.anchor / "configs" / "confirmatory.json",
        runs_root=fixture.anchor / "artifacts" / "runs",
        expected_confirmatory_gate={"status": "passed"},
        expected_cli_input_binding=cli_binding,
    )
    return authority.build_original_confirmatory_scientific_authority_projection(
        static_runner_binding=static,
        historical_primary_authority_artifact_root_sha256="d" * 64,
        historical_primary_evidence_sha256="2" * 64,
        technical_authorization_sha256="3" * 64,
        technical_execution_source_root_sha256="c" * 64,
        technical_execution_source_manifest_sha256="b" * 64,
        source_delta_sha256="6" * 64,
        confirmatory_storage_policy_sha256="7" * 64,
        independent_review_receipt_sha256="f" * 64,
    )


def test_scientific_authority_rejects_rehashed_static_t0_authorization_mismatch(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    tampered = deepcopy(scientific)
    static = dict(tampered["static_runner_binding"])
    static["technical_authorization_sha256"] = "a" * 64
    published = dict(static["published_technical_authority_lifecycle_binding"])
    technical = dict(published["technical_authority"])
    technical["technical_authorization_sha256"] = "a" * 64
    technical = _rehash_without_self(technical, "binding_sha256")
    published["technical_authority"] = technical
    published = _rehash_without_self(published, "binding_sha256")
    static["published_technical_authority_lifecycle_binding"] = published
    static = _rehash_without_self(static, "binding_sha256")
    tampered["static_runner_binding"] = static
    tampered["static_runner_binding_sha256"] = static["binding_sha256"]
    tampered = _rehash_without_self(
        tampered,
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="nested roots differ",
    ):
        authority.canonical_original_confirmatory_scientific_authority_projection(tampered)


@pytest.mark.parametrize(
    ("flat_field", "nested_field"),
    [
        (
            "historical_primary_authority_artifact_root_sha256",
            "parent_artifact_root_sha256",
        ),
        ("technical_execution_source_root_sha256", "execution_source_root_sha256"),
        (
            "technical_execution_source_manifest_sha256",
            "execution_source_manifest_sha256",
        ),
        ("independent_review_receipt_sha256", "independent_review_receipt_sha256"),
    ],
)
def test_scientific_authority_rejects_rehashed_flat_published_t0_mismatch(
    capsule_fixture: CapsuleFixture,
    flat_field: str,
    nested_field: str,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    nested = scientific["static_runner_binding"]["published_technical_authority_lifecycle_binding"][
        "technical_authority"
    ][nested_field]
    assert scientific[flat_field] == nested
    scientific[flat_field] = "0" * 64
    scientific = _rehash_without_self(
        scientific,
        "scientific_authority_root_sha256",
    )

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="nested roots differ",
    ):
        authority.canonical_original_confirmatory_scientific_authority_projection(scientific)


def test_static_v3_rejects_rehashed_flat_t0_root_mismatch(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    static["technical_authority_artifact_root_sha256"] = "0" * 64
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="exact outcome-blind policy",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


def test_static_v3_rejects_rehashed_freeze_t0_parent_mismatch(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    static["freeze_directory"] = str(capsule_fixture.anchor / "artifacts" / "other-freeze")
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="exact outcome-blind policy",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


@pytest.mark.parametrize(
    ("flat_field", "replacement"),
    [
        ("technical_authority_directory", "other-technical-authority"),
        ("technical_authorization_sha256", "0" * 64),
    ],
)
def test_static_v3_rejects_other_rehashed_flat_t0_mismatches(
    capsule_fixture: CapsuleFixture,
    flat_field: str,
    replacement: str,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    static[flat_field] = (
        str(
            capsule_fixture.anchor
            / "artifacts"
            / "original_confirmatory_technical_authorities"
            / replacement
        )
        if flat_field == "technical_authority_directory"
        else replacement
    )
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="exact outcome-blind policy",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


def test_static_v3_rejects_rehashed_permissive_composite_binding(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    published = dict(static["published_technical_authority_lifecycle_binding"])
    published["automatic_retry_allowed"] = True
    published = _rehash_without_self(published, "binding_sha256")
    static["published_technical_authority_lifecycle_binding"] = published
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="exact one-use policy",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


def test_static_v3_rejects_composite_without_review_attempt_claim(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    published = dict(static["published_technical_authority_lifecycle_binding"])
    published.pop("review_attempt_claim_sha256")
    published = _rehash_without_self(published, "binding_sha256")
    static["published_technical_authority_lifecycle_binding"] = published
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="unexpected field set",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


def test_static_v3_rejects_self_hashed_v2_without_fallback(
    capsule_fixture: CapsuleFixture,
) -> None:
    scientific = _scientific_authority(capsule_fixture)
    static = dict(scientific["static_runner_binding"])
    static["schema_version"] = 2
    static["policy"] = "original_confirmatory_static_runner_binding_v2"
    static = _rehash_without_self(static, "binding_sha256")

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="exact outcome-blind policy",
    ):
        authority.canonical_original_confirmatory_static_runner_binding(static)


def _q_replacement_v2(
    fixture: CapsuleFixture,
) -> dict[str, Any]:
    (fixture.anchor / "artifacts" / "resource_control").mkdir(
        parents=True,
        exist_ok=True,
    )
    publication_ancestors = (
        authority.observe_original_confirmatory_control_publication_ancestor_lease(fixture.anchor)
    )
    external_release_root = "9" * 64
    external_publication_id = f"cpr-{'1' * 32}"
    external_attestation_path = (
        fixture.anchor.parent
        / authority.EXTERNAL_CONTROL_PLANE_RELEASE_DIRECTORY_NAME
        / authority.EXTERNAL_CONTROL_PLANE_VERIFICATIONS_DIRECTORY_NAME
        / external_publication_id
        / authority.EXTERNAL_CONTROL_PLANE_QUALIFICATION_ATTESTATION_FILENAME
    )
    supervisor_code_root = (
        fixture.anchor.parent
        / "AANCA-control-plane"
        / "releases"
        / external_release_root
        / "supervisor"
    )
    supervisor_state_root = fixture.anchor.parent / f"{fixture.anchor.name}-supervisor-state"
    supervisor_code_root.mkdir(parents=True, exist_ok=True)
    supervisor_state_root.mkdir(parents=True, exist_ok=True)
    terminal_client_source = (
        supervisor_code_root / authority.TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME
    )
    if not terminal_client_source.exists():
        terminal_client_source.write_bytes(b"synthetic terminal client launcher v1\n")
        terminal_client_source.chmod(stat.S_IREAD)
    terminal_client_physical, _ = authority._read_stable_file(
        terminal_client_source,
        maximum_bytes=1024,
    )
    terminal_client_identity = authority.build_original_confirmatory_physical_file_identity(
        role="terminal-client-launcher",
        path=terminal_client_source,
        volume_serial_number=terminal_client_physical.volume_serial_number,
        file_id_128=terminal_client_physical.file_id_128,
        device=terminal_client_physical.device,
        inode=terminal_client_physical.inode,
        size_bytes=terminal_client_physical.size_bytes,
        mode=stat.S_IMODE(terminal_client_physical.mode),
        file_attributes=terminal_client_physical.file_attributes,
        modified_time_ns=terminal_client_physical.modified_time_ns,
        changed_time_ns=terminal_client_physical.changed_time_ns,
        sha256=terminal_client_physical.sha256,
    )
    root_handle, root_native, root_facts = authority._open_e_job_directory_handle(
        supervisor_code_root
    )
    authority._close_e_job_directory_handle(root_handle, windows_native=root_native)
    terminal_client_ancestor = (
        authority.build_original_confirmatory_terminal_client_launcher_ancestor_lease(
            supervisor_root=supervisor_code_root,
            records=[
                {
                    "path": str(supervisor_code_root),
                    "volume_serial_number": root_facts[0],
                    "file_id_128": root_facts[1],
                    "file_attributes": root_facts[2],
                    "reparse_point": False,
                }
            ],
        )
    )
    codex_base = _codex_handoff_base(operational=True)
    scientific_authority = _scientific_authority(fixture)
    release = authority.build_original_confirmatory_supervisor_release_binding(
        fixture.capsule,
        external_control_plane_release_root_sha256=external_release_root,
        external_control_plane_publication_id=external_publication_id,
        external_control_plane_release_qualification_attestation_path=(external_attestation_path),
        external_control_plane_release_qualification_attestation_file_sha256="2" * 64,
        external_control_plane_release_qualification_attestation_root_sha256="3" * 64,
        supervisor_code_root=supervisor_code_root,
        supervisor_state_root=supervisor_state_root,
        supervisor_source_path=supervisor_code_root / "aanca_supervisor.py",
        supervisor_source_sha256="d" * 64,
        supervisor_launcher_path=supervisor_code_root / "launch_hidden.ps1",
        supervisor_launcher_sha256="e" * 64,
        external_codex_handoff_policy=authority.EXTERNAL_CODEX_HANDOFF_POLICY,
        external_codex_handoff_authority_spec_file_sha256="4" * 64,
        external_codex_handoff_authority_spec_canonical_root_sha256="5" * 64,
        internal_codex_wake_disposition=(authority.INTERNAL_CODEX_WAKE_DISPOSITION),
        terminal_client_launcher_source_physical_identity=terminal_client_identity,
        terminal_client_launcher_source_ancestor_lease=terminal_client_ancestor,
    )
    base_projection = {
        "schema_version": 2,
        "policy": authority.Q_REPLACEMENT_V2_POLICY,
        "authority_disposition": authority.Q_REPLACEMENT_V2_DISPOSITION,
        "q_path": str(authority.original_confirmatory_q_replacement_v2_path(fixture.anchor)),
        "project_root": str(fixture.anchor),
        "scientific_authority": scientific_authority,
        "execution_capsule": fixture.capsule.as_dict(),
        "publication_ancestor_lease": publication_ancestors,
        "publication_ancestor_lease_root_sha256": authority.canonical_json_sha256(
            publication_ancestors
        ),
        "command_derivation_contract": (
            authority.build_original_confirmatory_command_derivation_contract()
        ),
        "supervisor_release": release,
        "codex_handoff_base_authority": codex_base,
    }
    attempt_identity = authority.build_original_confirmatory_q_attempt_identity_projection(
        attempt_id=f"ocq-{'a' * 32}",
        q_base_authority_root_sha256=authority.canonical_json_sha256(base_projection),
    )
    creation = _codex_handoff_creation(
        codex_base,
        str(
            supervisor_state_root
            / "jobs"
            / attempt_identity["job_id"]
            / "codex_handoff_attempt_authority.json"
        ),
    )
    return authority.build_original_confirmatory_q_replacement_v2(
        project_root=fixture.anchor,
        attempt_id=f"ocq-{'a' * 32}",
        scientific_authority=scientific_authority,
        execution_capsule=fixture.capsule,
        publication_ancestor_lease=publication_ancestors,
        external_control_plane_release_root_sha256=external_release_root,
        external_control_plane_publication_id=external_publication_id,
        external_control_plane_release_qualification_attestation_path=(external_attestation_path),
        external_control_plane_release_qualification_attestation_file_sha256="2" * 64,
        external_control_plane_release_qualification_attestation_root_sha256="3" * 64,
        supervisor_code_root=supervisor_code_root,
        supervisor_state_root=supervisor_state_root,
        supervisor_source_sha256="d" * 64,
        supervisor_launcher_sha256="e" * 64,
        external_codex_handoff_policy=authority.EXTERNAL_CODEX_HANDOFF_POLICY,
        external_codex_handoff_authority_spec_file_sha256="4" * 64,
        external_codex_handoff_authority_spec_canonical_root_sha256="5" * 64,
        internal_codex_wake_disposition=(authority.INTERNAL_CODEX_WAKE_DISPOSITION),
        terminal_client_launcher_source_physical_identity=(terminal_client_identity),
        terminal_client_launcher_source_ancestor_lease=(terminal_client_ancestor),
        codex_handoff_base_authority=codex_base,
        codex_handoff_attempt_creation_authority=creation,
        expected_launch_environment=_environment(
            attempt_identity["launch_nonce"],
            user_profile=str(fixture.capsule.runtime_python_ancestor_lease.anchor_path),
        ),
    )


def _e_intent(
    fixture: CapsuleFixture,
    q: dict[str, Any],
) -> dict[str, Any]:
    attempt = q["attempt_identity_projection"]
    job_dir = Path(q["control_staging_projection"]["final_job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    scientific = authority.build_original_confirmatory_scientific_request_projection(
        scientific_authority=q["scientific_authority"],
        job_id=attempt["job_id"],
        attempt_id=attempt["attempt_id"],
        run_id=attempt["run_id"],
        execution_mode=attempt["execution_mode"],
        retry_of_run_id=attempt["retry_of_run_id"],
        plan_sha256="3" * 64,
        controls_binding_sha256="4" * 64,
        bridge_binding_sha256="d" * 64,
        checkpoint_authority_projection={
            "schema_version": 1,
            "policy": "synthetic_exact_180",
            "directive_count": 180,
        },
    )
    return authority.build_original_confirmatory_e_intent(
        q_authority=q,
        q_file_sha256=authority.canonical_json_line_sha256(q),
        supervisor_job_id=attempt["job_id"],
        supervisor_job_directory=job_dir,
        attempt_id=attempt["attempt_id"],
        run_id=attempt["run_id"],
        launch_nonce=attempt["launch_nonce"],
        execution_mode=attempt["execution_mode"],
        retry_of_run_id=attempt["retry_of_run_id"],
        scientific_request_projection=scientific,
        codex_handoff_attempt_creation_authority=_codex_handoff_creation(
            q["codex_handoff_base_authority"],
            str(job_dir / "codex_handoff_attempt_authority.json"),
        ),
    )


def _q_e_handshake_bundle(
    fixture: CapsuleFixture,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    q = _q_replacement_v2(fixture)
    e = _e_intent(fixture, q)
    job_dir = Path(e["job"]["supervisor_job_dir"])
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    contract = authority.build_original_confirmatory_q_e_custody_contract(
        supervisor_job_directory=job_dir,
    )

    def leaf(path: Path, sha256: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": authority._CONTROL_PUBLICATION_IDENTITY_POLICY,
            "path": str(path),
            "volume_serial_number": 1,
            "file_id_128": "1" * 32,
            "size_bytes": 1,
            "sha256": sha256,
            "file_attributes": 1,
            "regular_file": True,
            "read_only": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "share_access": ["FILE_SHARE_READ"],
            "write_handle_retained": False,
            "delete_access": False,
        }

    e_records = [
        {
            "path": str(path),
            "volume_serial_number": 1,
            "file_id_128": f"{index:032x}",
            "file_attributes": 0x10,
            "reparse_point": False,
        }
        for index, path in enumerate(
            (staged_e.parent.parent.parent, staged_e.parent.parent, staged_e.parent),
            start=1,
        )
    ]
    e_ancestor_lease = {
        "schema_version": 1,
        "policy": authority._Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(job_dir.parent.parent),
        "records": e_records,
        "record_count": 3,
        "records_root_sha256": authority.canonical_json_sha256(e_records),
        "directory_access_mask": authority.EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": authority._Q_E_CUSTODY_E_ANCESTOR_DISPOSITION,
    }
    creation = 116444736000000000
    process_identity = {
        "pid": 123,
        "creation_time_100ns": creation,
        "creation_time_utc": authority._windows_filetime_100ns_to_utc(creation),
        "program_path": str(fixture.anchor / "supervisor.exe"),
        "program_sha256": "a" * 64,
        "command_sha256": "b" * 64,
    }
    ready = authority.build_original_confirmatory_q_e_custody_ready(
        contract=contract,
        supervisor_job_id=e["job"]["job_id"],
        supervisor_process_identity=process_identity,
        controller_process_identity={
            **process_identity,
            "pid": 124,
            "program_path": str(fixture.anchor / "controller.exe"),
            "program_sha256": "c" * 64,
            "command_sha256": "d" * 64,
        },
        windows_boot_time_utc="2026-07-30T00:00:00Z",
        q_authority_root_sha256=q["q_authority_root_sha256"],
        q_file_sha256=authority.canonical_json_line_sha256(q),
        e_file_sha256=authority.canonical_json_line_sha256(e),
        q_leaf_physical_identity=leaf(
            Path(q["q_path"]),
            authority.canonical_json_line_sha256(q),
        ),
        q_ancestor_lease=q["publication_ancestor_lease"],
        e_leaf_physical_identity=leaf(
            staged_e,
            authority.canonical_json_line_sha256(e),
        ),
        e_ancestor_lease=e_ancestor_lease,
        q_leaf_handle=101,
        q_ancestor_handles=(102, 103, 104),
        e_leaf_handle=105,
        e_ancestor_handles=(106, 107, 108),
        independent_verifier_receipt_sha256=q["scientific_authority"][
            "independent_review_receipt_sha256"
        ],
        supervisor_job_directory=job_dir,
        staged_e_intent_path=staged_e,
    )
    receipt = authority.build_original_confirmatory_q_e_custody_receipt(
        contract=contract,
        ready=ready,
        supervisor_job_directory=job_dir,
        staged_e_intent_path=staged_e,
    )
    return q, e, contract, ready, receipt, process_identity


def _q_e_handshake_values(
    fixture: CapsuleFixture,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _q, _e, contract, ready, receipt, process_identity = _q_e_handshake_bundle(fixture)
    return contract, ready, receipt, process_identity


def test_canonical_json_hash_uses_no_line_but_line_helper_adds_one() -> None:
    value = {"b": 2, "a": 1}
    assert authority.canonical_json_bytes(value) == b'{"a":1,"b":2}'
    assert authority.canonical_json_line_bytes(value) == b'{"a":1,"b":2}\n'
    assert authority.canonical_json_sha256(value) != authority.canonical_json_line_sha256(value)
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_json_bytes({"bad": float("nan")})


def test_e_intent_round_trip_is_acyclic_and_derives_exact_three_commands(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    e = _e_intent(capsule_fixture, q)
    e_file_sha256 = authority.canonical_json_line_sha256(e)
    assert (
        authority.canonical_original_confirmatory_e_intent(
            e,
            q_authority=q,
            expected_q_file_sha256=authority.canonical_json_line_sha256(q),
        )
        == e
    )
    assert not authority._contains_mapping_key(e, "supervisor_spec_sha256")
    assert "e_intent_file_sha256" not in e
    assert len(e) == 23
    assert len(e["job"]) == 13
    terminal_projection = e["job"]["terminal_custody_authority_projection"]
    assert (
        e["job"]["terminal_custody_authority_projection_root_sha256"]
        == terminal_projection["projection_root_sha256"]
    )
    assert (
        terminal_projection["terminal_custody_authority_template_root_sha256"]
        == q["supervisor_release"]["terminal_custody_authority_template_root_sha256"]
    )
    assert (
        terminal_projection["outcome_blind_expected_artifact_instance"]["expected_run_directory"]
        == e["scientific_request_projection"]["expected_run_directory"]
    )
    static_template = (
        authority.build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    assert (
        static_template["template_root_sha256"]
        == (terminal_projection["terminal_custody_authority_template_root_sha256"])
    )
    transport = static_template["transport_contract"]
    assert "pipe_name" not in transport
    assert "postwake_custody_seed_sha256" not in transport
    assert transport["concrete_pipe_name_or_seed_in_e_allowed"] is False
    for mode in authority.CAPSULE_ALLOWED_MODES:
        command = authority.derive_original_confirmatory_capsule_command_from_e(
            e_intent=e,
            e_file_sha256=e_file_sha256,
            q_authority=q,
            capsule_mode=mode,
        )
        assert command.argv[4] == mode
        assert command.argv[command.argv.index("--e-intent-sha256") + 1] == (e_file_sha256)
        assert (
            command.argv[command.argv.index("--e-intent-core-sha256") + 1]
            == e["intent_core_sha256"]
        )


def test_external_supervisor_spec_payload_is_exact_q20_e23_spec52_input(
    capsule_fixture: CapsuleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, e, contract, ready, receipt, _identity = _q_e_handshake_bundle(capsule_fixture)
    job_dir = Path(e["job"]["supervisor_job_dir"])
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    custody_fields = authority.build_original_confirmatory_q_e_custody_spec_fields(
        contract=contract,
        ready=ready,
        receipt=receipt,
        supervisor_job_directory=job_dir,
        staged_e_intent_path=staged_e,
    )
    attempt_creation = e["codex_handoff_attempt_creation_authority"]
    external_handoff = {
        "policy": authority.EXTERNAL_CODEX_HANDOFF_POLICY,
        "staged_e_intent_path": str(staged_e),
        "staged_e_intent_file_sha256": authority.canonical_json_line_sha256(e),
        "staged_e_intent_core_root_sha256": e["intent_core_sha256"],
        "attempt_creation_authority_payload_sha256": attempt_creation["payload_sha256"],
        "attempt_authority_output_path": attempt_creation["payload"][
            "attempt_authority_output_path"
        ],
        "terminal_handoff_receipt_output_path": str(
            job_dir / "external_codex_terminal_handoff.json"
        ),
        "internal_codex_wake_allowed": False,
        "legacy_handoff_session_allowed": False,
        "single_wake_owner": authority.EXTERNAL_CODEX_HANDOFF_SINGLE_WAKE_OWNER,
    }

    def build_raw() -> dict[str, Any]:
        return authority.build_original_confirmatory_external_supervisor_spec_payload_v3(
            q_authority=q,
            e_intent=e,
            e_file_sha256=authority.canonical_json_line_sha256(e),
            q_e_custody_spec_fields=custody_fields,
            external_codex_handoff=external_handoff,
            pipe_owner_sid="S-1-5-21-1000",
        )

    raw = build_raw()
    raw_bytes = authority.canonical_json_bytes(raw)
    ambient_cases = (
        {},
        {
            "USERPROFILE": "Z:\\adversarial-profile",
            "HOME": "Z:\\adversarial-home",
            "PATH": "Z:\\adversarial-bin",
        },
    )
    for ambient in ambient_cases:
        environment_spy = _AmbientEnvironmentSpy(ambient)
        home_read_count = 0

        def reject_path_home(_path_type: type[Path]) -> NoReturn:
            nonlocal home_read_count
            home_read_count += 1
            raise AssertionError("pure authority called Path.home()")

        with monkeypatch.context() as ambient_patch:
            ambient_patch.setattr(authority.os, "environ", environment_spy)
            ambient_patch.setattr(authority.Path, "home", classmethod(reject_path_home))
            ambient_raw = build_raw()
        assert authority.canonical_json_bytes(ambient_raw) == raw_bytes
        assert environment_spy.read_count == 0
        assert home_read_count == 0

    mismatched_q = deepcopy(q)
    mismatched_q["expected_launch_environment"] = _environment(
        q["attempt_identity_projection"]["launch_nonce"],
        user_profile="C:\\Users\\SealedButDifferent",
    ).as_dict()
    mismatched_q = _rehash_without_self(
        mismatched_q,
        "q_authority_root_sha256",
    )
    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="Q replacement-v2 violates its exact closed policy",
    ):
        authority.canonical_original_confirmatory_q_replacement_v2(mismatched_q)

    assert len(raw) == 51
    assert set(raw) == authority._EXTERNAL_SUPERVISOR_SPEC_INPUT_FIELDS
    assert raw["schema_version"] == 3
    assert raw["codex"] is None
    assert raw["process_kind"] == "confirmatory"
    assert raw["external_codex_handoff"] == external_handoff
    assert raw["expected_environment"] == q["expected_launch_environment"]
    assert raw["argv"][raw["argv"].index("--e-intent") + 1] == str(staged_e)
    assert raw["integrity_verifier"]["argv"][
        raw["integrity_verifier"]["argv"].index("--e-intent") + 1
    ] == str(staged_e)
    terminal_argv = raw["terminal_composition_contract"]["verifier_command"]["argv"]
    assert terminal_argv[terminal_argv.index("--e-intent") + 1] == str(staged_e)

    q19 = deepcopy(q)
    del q19["expected_launch_environment"]
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_original_confirmatory_external_supervisor_spec_payload_v3(
            q_authority=q19,
            e_intent=e,
            e_file_sha256=authority.canonical_json_line_sha256(e),
            q_e_custody_spec_fields=custody_fields,
            external_codex_handoff=external_handoff,
            pipe_owner_sid="S-1-5-21-1000",
        )

    hybrid = deepcopy(q)
    hybrid["legacy_expected_environment"] = hybrid["expected_launch_environment"]
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_replacement_v2(hybrid)

    bad_external = deepcopy(external_handoff)
    bad_external["legacy_handoff_session_allowed"] = True
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_original_confirmatory_external_supervisor_spec_payload_v3(
            q_authority=q,
            e_intent=e,
            e_file_sha256=authority.canonical_json_line_sha256(e),
            q_e_custody_spec_fields=custody_fields,
            external_codex_handoff=bad_external,
            pipe_owner_sid="S-1-5-21-1000",
        )

    alternate_environment = authority.build_expected_launch_environment_envelope_v1(
        attempt_nonce=e["job"]["launch_nonce"],
        supervisor_environment={
            "LOCALAPPDATA": "C:\\Users\\Different\\AppData\\Local",
            "SYSTEMROOT": "C:\\Windows",
            "TEMP": "C:\\Users\\Different\\AppData\\Local\\Temp",
            "TMP": "C:\\Users\\Different\\AppData\\Local\\Temp",
            "USERPROFILE": "C:\\Users\\Different",
        },
        child_environment={
            "AANCA_SUPERVISOR_ATTEMPT_NONCE": e["job"]["launch_nonce"],
            "LOCALAPPDATA": "C:\\Users\\Different\\AppData\\Local",
            "SYSTEMROOT": "C:\\Windows",
            "TEMP": "C:\\Users\\Different\\AppData\\Local\\Temp",
            "TMP": "C:\\Users\\Different\\AppData\\Local\\Temp",
            "USERPROFILE": "C:\\Users\\Different",
        },
    )
    bad_e = deepcopy(e)
    bad_e["expected_launch_environment"] = alternate_environment.as_dict()
    bad_e["process_environment_binding"] = (
        authority.build_original_confirmatory_process_environment_binding(
            alternate_environment
        ).as_dict()
    )
    bad_e = _rehash_without_self(bad_e, "intent_core_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_e_intent(
            bad_e,
            q_authority=q,
        )


def test_outcome_blind_artifact_template_is_closed_and_q_bound(
    capsule_fixture: CapsuleFixture,
) -> None:
    template = authority.build_original_confirmatory_outcome_blind_expected_artifact_projection()
    q = _q_replacement_v2(capsule_fixture)
    assert (
        authority.canonical_original_confirmatory_outcome_blind_expected_artifact_projection(
            template
        )
        == template
    )
    assert [item["role"] for item in template["ordered_role_templates"]] == [
        "terminal_seal",
        "integrity_receipt",
        "completion_evidence",
        "integrity_registry",
        "stage_attestation_registry",
        "stage_attestation_anchor",
        "disposition_anchor",
    ]
    assert all(item["expected_sha256"] is None for item in template["ordered_role_templates"])
    assert template["dotted_paths_allowed"] is False
    assert template["numeric_or_list_indirection_allowed"] is False
    assert template["outcome_values_read"] is False
    terminal_template = (
        authority.build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    assert terminal_template["outcome_blind_expected_artifact_projection"] == template
    assert (
        q["supervisor_release"]["terminal_custody_authority_template_root_sha256"]
        == terminal_template["template_root_sha256"]
    )


@pytest.mark.parametrize(
    ("selector", "replacement"),
    [
        ("allowed_flat_json_selectors", ["run_id", "metrics.auroc"]),
        ("outcome_values_read", True),
        ("strict_expected_type_equality_required", 1),
    ],
)
def test_outcome_blind_artifact_template_rejects_rehashed_policy_mutation(
    selector: str,
    replacement: Any,
) -> None:
    template = authority.build_original_confirmatory_outcome_blind_expected_artifact_projection()
    tampered = deepcopy(template)
    tampered[selector] = replacement
    tampered = _rehash_without_self(tampered, "projection_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_outcome_blind_expected_artifact_projection(
            tampered
        )


def test_codex_terminal_wake_prompt_template_and_render_are_exact(
    tmp_path: Path,
) -> None:
    template = (
        authority.build_original_confirmatory_codex_terminal_wake_prompt_template_projection()
    )
    assert (
        authority.canonical_original_confirmatory_codex_terminal_wake_prompt_template_projection(
            template
        )
        == template
    )
    assert template["policy"] == authority.CODEX_TERMINAL_WAKE_PROMPT_RENDER_POLICY
    assert template["template_utf8_lf"].endswith("\n")
    assert "\r" not in template["template_utf8_lf"]
    assert (
        template["template_sha256"]
        == hashlib.sha256(template["template_utf8_lf"].encode("ascii")).hexdigest()
    )
    job_directory = tmp_path / "supervisor" / "jobs" / "job-1"
    rendered, rendered_sha256 = authority.render_original_confirmatory_codex_terminal_wake_prompt(
        job_id="job-1",
        supervisor_job_directory=job_directory,
        supervisor_spec_path=job_directory / "run_spec.json",
        supervisor_spec_sha256="1" * 64,
        terminal_receipt_sha256="2" * 64,
        terminal_client_launcher_argv=[
            "python.exe",
            "-I",
            "-S",
            "-B",
            "terminal_client_launcher_v1.py",
        ],
        terminal_client_launcher_command_sha256="3" * 64,
        verify_terminal_command_sha256="4" * 64,
    )
    assert rendered.endswith("\n")
    assert "\r" not in rendered
    assert "{" not in rendered and "}" not in rendered
    assert '["python.exe","-I","-S","-B","terminal_client_launcher_v1.py"]' in rendered
    assert rendered_sha256 == hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    weakened = deepcopy(template)
    weakened["automatic_retry_allowed"] = True
    weakened = _rehash_without_self(weakened, "projection_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_codex_terminal_wake_prompt_template_projection(
            weakened
        )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.render_original_confirmatory_codex_terminal_wake_prompt(
            job_id="job-1",
            supervisor_job_directory=job_directory,
            supervisor_spec_path=job_directory / "run_spec.json",
            supervisor_spec_sha256="1" * 64,
            terminal_receipt_sha256="2" * 64,
            terminal_client_launcher_argv=[
                "python.exe",
                "terminal_client_launcher_v1.py\n--forbidden",
            ],
            terminal_client_launcher_command_sha256="3" * 64,
            verify_terminal_command_sha256="4" * 64,
        )


@pytest.mark.parametrize(
    ("role_index", "selector", "replacement"),
    [
        (0, "metrics.auroc", 0.9),
        (1, "prediction", "positive"),
        (2, "study_outcome_eligible", 1),
    ],
)
def test_artifact_instance_rejects_arbitrary_or_type_weakened_json_selectors(
    tmp_path: Path,
    role_index: int,
    selector: str,
    replacement: Any,
) -> None:
    run_directory = tmp_path / "runs" / "run-1"
    instance = authority.build_original_confirmatory_outcome_blind_expected_artifact_instance(
        run_id="run-1",
        expected_run_directory=run_directory,
    )
    assert (
        authority.canonical_original_confirmatory_outcome_blind_expected_artifact_instance(
            instance,
            run_id="run-1",
            expected_run_directory=run_directory,
        )
        == instance
    )
    assert all(Path(item["path"]).is_absolute() for item in instance["expected_artifacts"])
    tampered = deepcopy(instance)
    tampered["expected_artifacts"][role_index]["json_equals"][selector] = replacement
    tampered["expected_artifacts_root_sha256"] = authority.canonical_json_sha256(
        tampered["expected_artifacts"]
    )
    tampered = _rehash_without_self(tampered, "projection_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_outcome_blind_expected_artifact_instance(
            tampered,
            run_id="run-1",
            expected_run_directory=run_directory,
        )


def test_terminal_custody_projection_exactly_binds_duplex_schemas_and_bounds(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    e = _e_intent(capsule_fixture, q)
    envelope = authority.canonical_expected_launch_environment_envelope_v1(
        q["expected_launch_environment"]
    )
    binding = authority.build_original_confirmatory_process_environment_binding(envelope)
    command = authority.derive_original_confirmatory_capsule_command_from_e(
        e_intent=e,
        e_file_sha256=authority.canonical_json_line_sha256(e),
        q_authority=q,
        capsule_mode=authority.CAPSULE_TERMINAL_MODE,
    )
    job_directory = Path(e["job"]["supervisor_job_dir"])
    _overlap, _input_lease, _seed, custody = _terminal_dependencies(
        capsule_fixture,
        command,
        binding,
        job_dir=job_directory,
    )
    run_directory = Path(e["scientific_request_projection"]["expected_run_directory"])
    projection = e["job"]["terminal_custody_authority_projection"]
    launcher_projection = projection["terminal_client_launcher_projection"]
    launcher_release = q["supervisor_release"]["terminal_client_launcher_release"]
    template = (
        authority.build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    assert (
        authority.canonical_original_confirmatory_terminal_custody_authority_template_projection(
            template
        )
        == template
    )
    assert (
        authority.canonical_original_confirmatory_terminal_custody_authority_projection(
            projection,
            run_id=e["job"]["run_id"],
            expected_run_directory=run_directory,
            custody_contract=custody,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            supervisor_job_id=e["job"]["job_id"],
            supervisor_job_directory=job_directory,
            verify_terminal_command_projection_sha256=launcher_projection[
                "verify_terminal_command_projection_sha256"
            ],
            verify_terminal_environment_sha256=(
                binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=capsule_fixture.anchor,
        )
        == projection
    )
    assert (
        projection["terminal_custody_authority_template_root_sha256"]
        == template["template_root_sha256"]
    )
    assert template["message_sequence"] == [
        "CLAIM_READY",
        "CUSTODY_GRANT",
        "COMPOSED_READY",
        "FINAL_ACK",
    ]
    assert len(template["message_contracts"]["CLAIM_READY"]["field_names"]) == 40
    assert len(template["message_contracts"]["CUSTODY_GRANT"]["field_names"]) == 57
    claim_fields = template["message_contracts"]["CLAIM_READY"]["field_names"]
    grant_fields = template["message_contracts"]["CUSTODY_GRANT"]["field_names"]
    assert "immediate_venv_redirector_pid" in claim_fields
    assert "immediate_venv_redirector_process_identity" in claim_fields
    assert "terminal_client_launcher_process_identity" in claim_fields
    assert "parent_launcher_pid" not in claim_fields
    assert "terminal_client_launch_intent_path" in claim_fields
    assert "terminal_client_launch_intent_policy" in claim_fields
    assert "terminal_client_launch_intent_read" in claim_fields
    assert "terminal_client_launch_intent_file_sha256" not in claim_fields
    assert "terminal_client_launch_intent_physical_identity" not in claim_fields
    assert "launcher_redirector_child_grandparent_chain_verified" in grant_fields
    assert "launcher_redirector_child_same_supervisor_job_verified" in grant_fields
    assert "terminal_client_launch_intent_file_sha256" in grant_fields
    assert "terminal_client_launch_intent_root_sha256" in grant_fields
    assert "terminal_client_launch_intent_physical_identity" in grant_fields
    assert "terminal_client_launch_intent_supervisor_handle_slot" in grant_fields
    assert "terminal_client_launch_intent_child_custody_active" not in grant_fields
    assert "terminal_client_launch_intent_child_open_after_grant_required" in grant_fields
    ready_fields = template["message_contracts"]["COMPOSED_READY"]["field_names"]
    final_ack_fields = template["message_contracts"]["FINAL_ACK"]["field_names"]
    readback_fields = template["readback_contract"]["field_names"]
    assert len(ready_fields) == 38
    assert len(final_ack_fields) == 43
    assert len(readback_fields) == 37
    assert "terminal_client_launch_intent_child_handle_slot" in ready_fields
    assert "terminal_client_launch_intent_child_custody_active" in ready_fields
    assert "terminal_client_launch_intent_physical_identity_exact_match" in ready_fields
    assert (
        "terminal_client_launch_intent_supervisor_custody_retained_through_ack" in final_ack_fields
    )
    assert (
        "terminal_client_launch_intent_supervisor_custody_retained_through_readback"
        in readback_fields
    )
    assert template["transport_contract"]["outbound_max_bytes"] == custody.ready_max_bytes
    assert template["transport_contract"]["inbound_max_bytes"] == custody.ack_max_bytes
    assert template["transport_contract"]["terminal_client_arrival_timeout_ms"] == (
        custody.terminal_client_arrival_timeout_ms
    )
    assert template["transport_contract"]["custody_exchange_timeout_ms"] == (
        custody.custody_exchange_timeout_ms
    )
    assert template["claim_identity_contract"]["delete_access_allowed"] is False
    assert template["claim_identity_contract"]["delete_share_allowed"] is False
    assert template["execution_control_contract"]["automatic_retry_allowed"] is False

    weakened = deepcopy(template)
    weakened["message_contracts"]["CUSTODY_GRANT"]["field_names"].remove(
        "target_granted_access_mask"
    )
    weakened = _rehash_without_self(weakened, "template_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_custody_authority_template_projection(
            weakened
        )

    nonfrozen_bounds = authority.build_original_confirmatory_postwake_custody_handshake_contract(
        custody_seed=_seed,
        pipe_owner_sid="S-1-5-21-1000",
        readback_receipt_path=custody.readback_receipt_path,
        expected_composed_command_sha256=command.command_sha256,
        expected_composed_cwd=command.cwd,
        expected_composed_environment_sha256=(binding.exact_integrity_verifier_environment_sha256),
        ready_max_bytes=128 * 1024,
        ack_max_bytes=64 * 1024,
        terminal_client_arrival_timeout_ms=1_800_000,
        custody_exchange_timeout_ms=60_000,
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_custody_authority_projection(
            projection,
            run_id="run-1",
            expected_run_directory=run_directory,
            custody_contract=nonfrozen_bounds,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            supervisor_job_id="job-1",
            supervisor_job_directory=job_directory,
            verify_terminal_command_projection_sha256=launcher_projection[
                "verify_terminal_command_projection_sha256"
            ],
            verify_terminal_environment_sha256=(
                binding.exact_integrity_verifier_environment_sha256
            ),
            verify_terminal_cwd=capsule_fixture.anchor,
        )


def test_e_rejects_recursive_downstream_spec_hash_even_when_rehashed(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    e = _e_intent(capsule_fixture, q)
    tampered = dict(e)
    scientific = dict(tampered["scientific_request_projection"])
    checkpoint = dict(scientific["checkpoint_authority_projection"])
    checkpoint["supervisor_spec_sha256"] = "f" * 64
    scientific["checkpoint_authority_projection"] = checkpoint
    scientific["checkpoint_authority_projection_sha256"] = authority.canonical_json_sha256(
        checkpoint
    )
    scientific = _rehash_without_self(scientific, "projection_sha256")
    tampered["scientific_request_projection"] = scientific
    tampered = _rehash_without_self(tampered, "intent_core_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_e_intent(
            tampered,
            q_authority=q,
        )


def test_e_rejects_rehashed_non_allowlisted_artifact_selector(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    tampered = deepcopy(_e_intent(capsule_fixture, q))
    job = deepcopy(tampered["job"])
    projection = deepcopy(job["terminal_custody_authority_projection"])
    instance = deepcopy(projection["outcome_blind_expected_artifact_instance"])
    instance["expected_artifacts"][0]["json_equals"]["rankings"] = [1, 2, 3]
    instance["expected_artifacts_root_sha256"] = authority.canonical_json_sha256(
        instance["expected_artifacts"]
    )
    instance = _rehash_without_self(instance, "projection_root_sha256")
    projection["outcome_blind_expected_artifact_instance"] = instance
    projection = _rehash_without_self(projection, "projection_root_sha256")
    job["terminal_custody_authority_projection"] = projection
    job["terminal_custody_authority_projection_root_sha256"] = projection["projection_root_sha256"]
    tampered["job"] = job
    tampered = _rehash_without_self(tampered, "intent_core_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_e_intent(
            tampered,
            q_authority=q,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows CREATE_NEW custody semantics")
def test_e_publication_is_create_new_and_independently_custodied(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    q_path, q_sha, q_identity, q_author = (
        authority.publish_original_confirmatory_q_replacement_v2_once(q)
    )
    _, _, q_verifier = authority.require_original_confirmatory_q_replacement_v2(
        q_path,
        expected_file_sha256=q_sha,
        expected_publication_identity=q_identity,
        author_custody=q_author,
    )
    e = _e_intent(capsule_fixture, q)
    e_path, e_sha, e_identity, e_author = authority.publish_original_confirmatory_e_intent_once(
        e,
        q_authority=q,
        expected_q_file_sha256=q_sha,
        q_verifier_custody=q_verifier,
    )
    canonical, verified_sha, e_verifier = authority.require_original_confirmatory_e_intent(
        e_path,
        expected_file_sha256=e_sha,
        expected_publication_identity=e_identity,
        q_authority=q,
        expected_q_file_sha256=q_sha,
        author_custody=e_author,
    )
    try:
        assert canonical == e
        assert verified_sha == e_sha
        e_verifier.require_active()
        with pytest.raises(OSError):
            authority.publish_original_confirmatory_e_intent_once(
                e,
                q_authority=q,
                expected_q_file_sha256=q_sha,
                q_verifier_custody=q_verifier,
            )
    finally:
        e_verifier.close()
        q_verifier.close()


def _staged_q_e_handshake_values(
    fixture: CapsuleFixture,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    q = _q_replacement_v2(fixture)
    e = _e_intent(fixture, q)
    final_job = Path(e["job"]["supervisor_job_dir"])
    final_job.rmdir()
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    staged_e.parent.mkdir(parents=True)
    contract = authority.build_original_confirmatory_q_e_custody_contract(
        supervisor_job_directory=final_job,
    )

    def leaf(path: Path, sha256: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": authority._CONTROL_PUBLICATION_IDENTITY_POLICY,
            "path": str(path),
            "volume_serial_number": 1,
            "file_id_128": "1" * 32,
            "size_bytes": 1,
            "sha256": sha256,
            "file_attributes": 1,
            "regular_file": True,
            "read_only": True,
            "link_count": 1,
            "named_alternate_data_streams": [],
            "opened_without_reparse_follow": True,
            "share_access": ["FILE_SHARE_READ"],
            "write_handle_retained": False,
            "delete_access": False,
        }

    stage_paths = (
        staged_e.parent.parent.parent,
        staged_e.parent.parent,
        staged_e.parent,
    )
    records = [
        {
            "path": str(path),
            "volume_serial_number": 1,
            "file_id_128": f"{index:032x}",
            "file_attributes": 0x10,
            "reparse_point": False,
        }
        for index, path in enumerate(stage_paths, start=1)
    ]
    e_ancestor_lease = {
        "schema_version": 1,
        "policy": authority._Q_E_CUSTODY_E_ANCESTOR_LEASE_POLICY,
        "supervisor_root": str(stage_paths[0]),
        "records": records,
        "record_count": 3,
        "records_root_sha256": authority.canonical_json_sha256(records),
        "directory_access_mask": authority.EXECUTABLE_ANCESTOR_ACCESS_MASK,
        "share_access": ["FILE_SHARE_READ", "FILE_SHARE_WRITE"],
        "delete_share": False,
        "write_access": False,
        "owner_process_identity_required": True,
        "handle_slot_per_record_required": True,
        "continuous_overlap_through_independent_verification_required": True,
        "continuous_overlap_into_supervisor_required": True,
        "acquisition_disposition": authority._Q_E_CUSTODY_E_ANCESTOR_DISPOSITION,
    }
    creation = 116444736000000000
    supervisor_identity = {
        "pid": 123,
        "creation_time_100ns": creation,
        "creation_time_utc": authority._windows_filetime_100ns_to_utc(creation),
        "program_path": str(fixture.anchor / "supervisor.exe"),
        "program_sha256": "a" * 64,
        "command_sha256": "b" * 64,
    }
    ready = authority.build_original_confirmatory_q_e_custody_ready(
        contract=contract,
        supervisor_job_id=final_job.name,
        supervisor_process_identity=supervisor_identity,
        controller_process_identity={
            **supervisor_identity,
            "pid": 124,
            "program_path": str(fixture.anchor / "controller.exe"),
            "program_sha256": "c" * 64,
            "command_sha256": "d" * 64,
        },
        windows_boot_time_utc="2026-07-30T00:00:00Z",
        q_authority_root_sha256=q["q_authority_root_sha256"],
        q_file_sha256=authority.canonical_json_line_sha256(q),
        e_file_sha256=authority.canonical_json_line_sha256(e),
        q_leaf_physical_identity=leaf(
            Path(q["q_path"]),
            authority.canonical_json_line_sha256(q),
        ),
        q_ancestor_lease=q["publication_ancestor_lease"],
        e_leaf_physical_identity=leaf(
            staged_e,
            authority.canonical_json_line_sha256(e),
        ),
        e_ancestor_lease=e_ancestor_lease,
        q_leaf_handle=101,
        q_ancestor_handles=(102, 103, 104),
        e_leaf_handle=105,
        e_ancestor_handles=(106, 107, 108),
        independent_verifier_receipt_sha256=q["scientific_authority"][
            "independent_review_receipt_sha256"
        ],
        supervisor_job_directory=final_job,
        staged_e_intent_path=staged_e,
    )
    receipt = authority.build_original_confirmatory_q_e_custody_receipt(
        contract=contract,
        ready=ready,
        supervisor_job_directory=final_job,
        staged_e_intent_path=staged_e,
    )
    return q, e, contract, ready, receipt, staged_e, final_job


def test_staged_q_e_wire_v1_keeps_exact_eight_roles_and_separate_path_domains(
    capsule_fixture: CapsuleFixture,
) -> None:
    _q, _e, contract, ready, receipt, staged_e, final_job = _staged_q_e_handshake_values(
        capsule_fixture
    )
    ack = authority.build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
        supervisor_job_directory=final_job,
        staged_e_intent_path=staged_e,
    )
    spec = authority.build_original_confirmatory_q_e_custody_spec_fields(
        contract=contract,
        ready=ready,
        receipt=receipt,
        supervisor_job_directory=final_job,
        staged_e_intent_path=staged_e,
    )

    assert contract["receipt_path"] == str(final_job / authority.Q_E_CUSTODY_RECEIPT_FILENAME)
    assert ready["e_leaf_physical_identity"]["path"] == str(staged_e)
    assert [item["path"] for item in ready["e_ancestor_lease"]["records"]] == [
        str(staged_e.parent.parent.parent),
        str(staged_e.parent.parent),
        str(staged_e.parent),
    ]
    assert (
        len(
            {
                ready["q_leaf_handle"],
                *ready["q_ancestor_handles"],
                ready["e_leaf_handle"],
                *ready["e_ancestor_handles"],
            }
        )
        == 8
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_ack(
            ack,
            contract=contract,
            ready=ready,
            receipt=receipt,
            supervisor_job_directory=final_job,
            staged_e_intent_path=staged_e,
        )
        == ack
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_spec_fields(
            spec,
            supervisor_job_directory=final_job,
            staged_e_intent_path=staged_e,
        )
        == spec
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_ready(
            ready,
            contract=contract,
        )
        == ready
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_e_custody_ready(
            ready,
            contract=contract,
            supervisor_job_directory=final_job,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows CREATE_NEW custody semantics")
def test_staged_e_publication_is_create_new_and_keeps_final_job_absent(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    q_path, q_sha, q_identity, q_author = (
        authority.publish_original_confirmatory_q_replacement_v2_once(q)
    )
    _, _, q_verifier = authority.require_original_confirmatory_q_replacement_v2(
        q_path,
        expected_file_sha256=q_sha,
        expected_publication_identity=q_identity,
        author_custody=q_author,
    )
    e = _e_intent(capsule_fixture, q)
    final_job = Path(e["job"]["supervisor_job_dir"])
    final_job.rmdir()
    staged_e = Path(q["control_staging_projection"]["e_intent_path"])
    staged_e.parent.mkdir(parents=True)
    e_path, e_sha, e_identity, e_author = (
        authority.publish_original_confirmatory_staged_e_intent_once(
            e,
            publication_path=staged_e,
            supervisor_job_directory=final_job,
            q_authority=q,
            expected_q_file_sha256=q_sha,
            q_verifier_custody=q_verifier,
        )
    )
    canonical, verified_sha, e_verifier = authority.require_original_confirmatory_staged_e_intent(
        e_path,
        supervisor_job_directory=final_job,
        expected_file_sha256=e_sha,
        expected_publication_identity=e_identity,
        q_authority=q,
        expected_q_file_sha256=q_sha,
        author_custody=e_author,
    )
    try:
        assert canonical == e
        assert verified_sha == e_sha
        assert not final_job.exists()
        e_verifier.require_active(final_job_must_be_absent=True)
        with pytest.raises(OSError):
            authority.publish_original_confirmatory_staged_e_intent_once(
                e,
                publication_path=staged_e,
                supervisor_job_directory=final_job,
                q_authority=q,
                expected_q_file_sha256=q_sha,
                q_verifier_custody=q_verifier,
            )
    finally:
        e_verifier.close()
        q_verifier.close()


def test_q_e_custody_ready_receipt_and_ack_are_closed_and_deterministic(
    capsule_fixture: CapsuleFixture,
) -> None:
    contract, ready, receipt, _identity = _q_e_handshake_values(capsule_fixture)
    ack = authority.build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_ready(
            ready,
            contract=contract,
        )
        == ready
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_receipt(
            receipt,
            contract=contract,
            ready=ready,
        )
        == receipt
    )
    assert (
        authority.canonical_original_confirmatory_q_e_custody_ack(
            ack,
            contract=contract,
            ready=ready,
            receipt=receipt,
        )
        == ack
    )
    assert "created_at_utc" not in receipt
    assert (
        authority.build_original_confirmatory_q_e_custody_receipt(
            contract=contract,
            ready=ready,
        )
        == receipt
    )
    assert len(ready["q_ancestor_handles"]) == 3
    assert len(ready["e_ancestor_handles"]) == 3
    assert ready["leaf_target_access_mask"] == 0x80000000
    assert receipt["q_leaf_retained_binding"]["target_granted_access_mask"] == 0x00120089


@pytest.mark.parametrize("coherently_reseal", [False, True])
@pytest.mark.parametrize(
    ("record", "path", "replacement"),
    [
        ("contract", ("schema_version",), True),
        ("contract", ("ready_max_bytes",), float(authority.Q_E_CUSTODY_LINE_MAX_BYTES)),
        ("contract", ("duplicate_options",), False),
        ("ready", ("schema_version",), True),
        ("receipt", ("schema_version",), True),
        (
            "receipt",
            ("q_leaf_retained_binding", "target_granted_access_mask"),
            float(0x00120089),
        ),
        ("ack", ("schema_version",), True),
        ("ack", ("all_target_handles_retained",), 1),
        ("spec", ("q_e_custody_contract", "schema_version"), True),
        ("spec", ("q_e_custody_handoff", "schema_version"), True),
    ],
)
def test_q_e_deterministic_records_reject_recursive_json_type_aliases(
    capsule_fixture: CapsuleFixture,
    coherently_reseal: bool,
    record: str,
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    contract, ready, receipt, _identity = _q_e_handshake_values(capsule_fixture)
    ack = authority.build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    spec = authority.build_original_confirmatory_q_e_custody_spec_fields(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    records = {
        "contract": contract,
        "ready": ready,
        "receipt": receipt,
        "ack": ack,
        "spec": spec,
    }
    tampered = deepcopy(records[record])
    target = tampered
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = replacement

    if coherently_reseal:
        if record == "contract":
            tampered = _rehash_without_self(tampered, "contract_sha256")
        elif record == "ready":
            tampered = _rehash_without_self(tampered, "handoff_root_sha256")
        elif record == "receipt":
            tampered = _rehash_without_self(tampered, "receipt_root_sha256")
        elif record == "ack":
            tampered = _rehash_without_self(tampered, "ack_sha256")
        elif path[0] == "q_e_custody_contract":
            tampered["q_e_custody_contract"] = _rehash_without_self(
                tampered["q_e_custody_contract"],
                "contract_sha256",
            )
        else:
            tampered["q_e_custody_handoff"] = _rehash_without_self(
                tampered["q_e_custody_handoff"],
                "handoff_root_sha256",
            )

    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        if record == "contract":
            authority.canonical_original_confirmatory_q_e_custody_contract(
                tampered,
                supervisor_job_directory=Path(contract["receipt_path"]).parent,
            )
        elif record == "ready":
            authority.canonical_original_confirmatory_q_e_custody_ready(
                tampered,
                contract=contract,
            )
        elif record == "receipt":
            authority.canonical_original_confirmatory_q_e_custody_receipt(
                tampered,
                contract=contract,
                ready=ready,
            )
        elif record == "ack":
            authority.canonical_original_confirmatory_q_e_custody_ack(
                tampered,
                contract=contract,
                ready=ready,
                receipt=receipt,
            )
        else:
            authority.canonical_original_confirmatory_q_e_custody_spec_fields(tampered)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("q_leaf_physical_identity", "schema_version"), True),
        (("q_leaf_physical_identity", "schema_version"), 1.0),
        (("e_leaf_physical_identity", "schema_version"), True),
        (("e_leaf_physical_identity", "size_bytes"), 1.0),
        (("e_ancestor_lease", "schema_version"), True),
        (("e_ancestor_lease", "schema_version"), 1.0),
        (("e_ancestor_lease", "record_count"), 3.0),
        (("supervisor_process_identity", "pid"), 123.0),
    ],
)
def test_q_e_coherently_resealed_nested_alias_cannot_enter_any_downstream_chain(
    capsule_fixture: CapsuleFixture,
    path: tuple[str, str],
    replacement: Any,
) -> None:
    contract, ready, receipt, _identity = _q_e_handshake_values(capsule_fixture)
    tampered_ready = deepcopy(ready)
    nested = tampered_ready[path[0]]
    assert isinstance(nested, dict)
    nested[path[1]] = replacement
    tampered_ready = _rehash_without_self(
        tampered_ready,
        "handoff_root_sha256",
    )
    ack = authority.build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    spec = authority.build_original_confirmatory_q_e_custody_spec_fields(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    tampered_spec = deepcopy(spec)
    tampered_spec["q_e_custody_handoff"] = tampered_ready

    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_e_custody_ready(
            tampered_ready,
            contract=contract,
        )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_original_confirmatory_q_e_custody_receipt(
            contract=contract,
            ready=tampered_ready,
        )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_original_confirmatory_q_e_custody_ack(
            contract=contract,
            ready=tampered_ready,
            receipt=receipt,
        )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_e_custody_spec_fields(tampered_spec)
    assert (
        authority.canonical_original_confirmatory_q_e_custody_ack(
            ack,
            contract=contract,
            ready=ready,
            receipt=receipt,
        )
        == ack
    )


def test_deterministic_json_containers_never_use_python_container_equality() -> None:
    container_fields = {
        "argv",
        "capsule_ancestor_lease",
        "capsule_lease_identity",
        "child_pin_share_access",
        "child_process_identity",
        "claim_physical_identity",
        "command_preimage_field_names",
        "downstream_hash_insertions",
        "e_consumption_contract",
        "expected_environment",
        "final_argument_order",
        "named_alternate_data_streams",
        "pipe_allowed_sids",
        "pipe_mode_flags",
        "pipe_open_mode_flags",
        "process_argv_prefix",
        "process_environment_binding",
        "python_ancestor_lease",
        "python_interpreter_flags",
        "python_isolated_flags",
        "python_lease_identity",
        "python_sys_argv_prefix",
        "required_success_roles",
        "runtime_python_ancestor_lease",
        "runtime_python_lease_identity",
        "sealed_input_allowlist",
        "share_access",
        "supervisor_pin_share_access",
        "supervisor_process_identity",
        "terminal_custody_authority_projection",
        "verifier_command",
    }
    deterministic_mapping_names = {
        "canonical_projections",
        "comparable_expected",
        "downstream_spec_fields",
        "e_from_author",
        "expected_projections",
        "expected_terminal_custody_projection",
        "guard_identity",
        "observed_environment",
        "q_from_author",
        "ready",
        "receipt_expected",
        "terminal_custody_projection",
    }

    def subscript_key(node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return node.slice.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    offenders: list[str] = []
    for path in (
        _MODULE_PATH,
        Path("capsule_bootstrap.py").resolve(),
    ):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare) or not any(
                isinstance(operator, (ast.Eq, ast.NotEq)) for operator in node.ops
            ):
                continue
            operands = [node.left, *node.comparators]
            operand_names = {operand.id for operand in operands if isinstance(operand, ast.Name)}
            has_raw_expected_pair = {"raw", "expected"}.issubset(operand_names)
            has_deterministic_mapping_name = bool(operand_names & deterministic_mapping_names)
            has_container_field = any(
                subscript_key(operand) in container_fields for operand in operands
            )
            if has_raw_expected_pair or has_deterministic_mapping_name or has_container_field:
                rendered = ast.get_source_segment(source, node) or "<unknown compare>"
                offenders.append(f"{path.name}:{node.lineno}:{rendered}")
    assert offenders == []

    exact_integer_fields = {
        "access_mask",
        "ack_max_bytes",
        "ancestor_target_access_mask",
        "creation_time_100ns",
        "directory_access_mask",
        "duplicate_options",
        "e_leaf_handle",
        "file_attributes",
        "index",
        "leaf_target_access_mask",
        "link_count",
        "max_attempt_count",
        "pid",
        "pipe_inbound_buffer_bytes",
        "pipe_outbound_buffer_bytes",
        "q_leaf_handle",
        "ready_max_bytes",
        "record_count",
        "schema_version",
        "size_bytes",
        "source_size_bytes",
        "target_granted_access_mask",
        "target_handle_value",
        "volume_serial_number",
    }
    untrusted_mapping_names = {
        "canonical_spec",
        "handshake",
        "identity",
        "lineage",
        "raw",
        "record",
        "value",
    }

    def integer_signature(node: ast.expr) -> tuple[str, str] | None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in untrusted_mapping_names
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in exact_integer_fields
        ):
            return node.value.id, node.slice.value
        return None

    missing_integer_guards: list[str] = []
    for path in (_MODULE_PATH, Path("capsule_bootstrap.py").resolve()):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and ("canonical" in node.name or "_require_" in node.name)
            and not node.name.startswith("_require_held_")
        ):
            compared: set[tuple[str, str]] = set()
            guarded: set[tuple[str, str]] = set()
            dynamic_integer_fields: set[str] = set()
            for assignment in ast.walk(function):
                if isinstance(assignment, ast.Assign) and isinstance(
                    assignment.value, (ast.Tuple, ast.List, ast.Set)
                ):
                    dynamic_integer_fields.update(
                        item.value
                        for item in assignment.value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                        and item.value in exact_integer_fields
                    )
            for node in ast.walk(function):
                if isinstance(node, ast.Compare):
                    operands = [node.left, *node.comparators]
                    compared.update(
                        signature
                        for operand in operands
                        if (signature := integer_signature(operand)) is not None
                    )
                    for operand in operands:
                        if (
                            isinstance(operand, ast.Call)
                            and isinstance(operand.func, ast.Name)
                            and operand.func.id == "type"
                            and len(operand.args) == 1
                            and any(
                                isinstance(other, ast.Name) and other.id == "int"
                                for other in operands
                            )
                        ):
                            signature = integer_signature(operand.args[0])
                            if signature is not None:
                                guarded.add(signature)
                            dynamic_subscript = operand.args[0]
                            if (
                                isinstance(dynamic_subscript, ast.Subscript)
                                and isinstance(dynamic_subscript.value, ast.Name)
                                and dynamic_subscript.value.id in untrusted_mapping_names
                                and isinstance(dynamic_subscript.slice, ast.Name)
                            ):
                                guarded.update(
                                    (
                                        dynamic_subscript.value.id,
                                        field,
                                    )
                                    for field in dynamic_integer_fields
                                )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"_exact_int", "_nonnegative_int", "_positive_int"}
                    and node.args
                ):
                    signature = integer_signature(node.args[0])
                    if signature is not None:
                        guarded.add(signature)
            for mapping_name, field in sorted(compared - guarded):
                missing_integer_guards.append(
                    f'{path.name}:{function.lineno}:{function.name}:{mapping_name}["{field}"]'
                )
    assert missing_integer_guards == []


def test_q_e_process_identity_is_exact_six_field_filetime_binding(
    capsule_fixture: CapsuleFixture,
) -> None:
    _contract, _ready, _receipt, identity = _q_e_handshake_values(capsule_fixture)
    assert set(identity) == {
        "pid",
        "creation_time_100ns",
        "creation_time_utc",
        "program_path",
        "program_sha256",
        "command_sha256",
    }
    assert (
        authority._canonical_e_process_identity(
            identity,
            role="synthetic Q/E process",
        )
        == identity
    )
    for mutate in (
        lambda value: value.pop("creation_time_utc"),
        lambda value: value.__setitem__(
            "creation_time_utc",
            "2026-07-30T00:00:00Z",
        ),
    ):
        tampered = dict(identity)
        mutate(tampered)
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority._canonical_e_process_identity(
                tampered,
                role="tampered Q/E process",
            )


def test_q_e_pipe_decoder_rejects_overflow_and_multiple_lines() -> None:
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority._decode_q_e_bounded_canonical_line(
            b'{"ok":true}\n',
            maximum_bytes=4,
            role="overflow",
        )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority._decode_q_e_bounded_canonical_line(
            b"{}\n{}\n",
            maximum_bytes=64,
            role="multiple lines",
        )


def test_q_e_handoff_receipt_must_be_the_one_frozen_in_q(
    capsule_fixture: CapsuleFixture,
    tmp_path: Path,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    held_path = tmp_path / "held-q.json"
    held_path.write_bytes(authority.canonical_json_line_bytes(q))
    descriptor = os.open(held_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        held = type("HeldQ", (), {"descriptor": descriptor})()
        observed, payload = authority._require_q_e_independent_review_receipt(
            held,
            expected_receipt_sha256=q["scientific_authority"]["independent_review_receipt_sha256"],
        )
        assert observed == q
        assert payload == authority.canonical_json_line_bytes(q)
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority._require_q_e_independent_review_receipt(
                held,
                expected_receipt_sha256="0" * 64,
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("record", "field", "replacement"),
    [
        ("ready", "scientific_inputs_before_ack_allowed", True),
        ("ready", "q_ancestor_handles", [102, 103]),
        ("receipt", "scientific_inputs_read", True),
        ("receipt", "created_at_utc", "2026-07-30T00:00:00Z"),
        ("ack", "all_target_handles_retained", False),
    ],
)
def test_q_e_custody_contract_rejects_open_or_weakened_messages(
    capsule_fixture: CapsuleFixture,
    record: str,
    field: str,
    replacement: Any,
) -> None:
    contract, ready, receipt, _identity = _q_e_handshake_values(capsule_fixture)
    ack = authority.build_original_confirmatory_q_e_custody_ack(
        contract=contract,
        ready=ready,
        receipt=receipt,
    )
    tampered = dict({"ready": ready, "receipt": receipt, "ack": ack}[record])
    tampered[field] = replacement
    if record == "ready" and field != "q_ancestor_handles":
        tampered = _rehash_without_self(tampered, "handoff_root_sha256")
    elif record == "receipt" and field != "created_at_utc":
        tampered = _rehash_without_self(tampered, "receipt_root_sha256")
    elif record == "ack":
        tampered = _rehash_without_self(tampered, "ack_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        if record == "ready":
            authority.canonical_original_confirmatory_q_e_custody_ready(
                tampered,
                contract=contract,
            )
        elif record == "receipt":
            authority.canonical_original_confirmatory_q_e_custody_receipt(
                tampered,
                contract=contract,
                ready=ready,
            )
        else:
            authority.canonical_original_confirmatory_q_e_custody_ack(
                tampered,
                contract=contract,
                ready=ready,
                receipt=receipt,
            )


def test_q_e_one_shot_orchestration_closes_sources_only_after_valid_ack(
    capsule_fixture: CapsuleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, ready, receipt, process_identity = _q_e_handshake_values(capsule_fixture)
    events: list[str] = []

    class Source:
        role = "independent-verifier"

        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def require_active(self) -> None:
            assert not self.closed

        def close(self) -> None:
            events.append(f"close-{self.name}")
            self.closed = True

    q_source = Source("q")
    e_source = Source("e")
    e_source.path = Path(ready["e_leaf_physical_identity"]["path"])
    e_source.q_custody = q_source
    state: dict[str, Any] = {}

    monkeypatch.setattr(
        authority,
        "_require_current_q_e_controller_identity",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_exact_q_e_process_handle_identity",
        lambda _handle, value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_q_e_independent_review_receipt",
        lambda *_args, **_kwargs: ({}, b""),
    )

    def duplicate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("duplicate")
        return ready

    monkeypatch.setattr(
        authority,
        "duplicate_original_confirmatory_q_e_custody_to_supervisor",
        duplicate,
    )

    def finalize(spec_fields: Mapping[str, Any]) -> None:
        events.append("spec")
        assert (
            authority.canonical_original_confirmatory_q_e_custody_spec_fields(spec_fields)
            == spec_fields
        )
        assert spec_fields["q_e_custody_contract"] == contract
        assert spec_fields["q_e_custody_handoff"] == ready
        assert spec_fields["q_e_custody_receipt"]["file_sha256"] == (
            authority.canonical_json_line_sha256(receipt)
        )
        state["receipt"] = receipt

    def receive(_maximum_bytes: int) -> bytes:
        events.append("ack")
        return authority.canonical_json_line_bytes(
            authority.build_original_confirmatory_q_e_custody_ack(
                contract=contract,
                ready=ready,
                receipt=state["receipt"],
            )
        )

    target = authority.OriginalConfirmatoryQESuspendedSupervisorTarget(
        process_handle=999,
        process_identity=process_identity,
        windows_boot_time_utc="2026-07-30T00:00:00Z",
        suspended=True,
        transport=authority.Q_E_CUSTODY_TRANSPORT,
        exact_job_object_membership_verified=True,
        bounded_anonymous_pipes_created_before_process=True,
        automatic_retry_allowed=False,
        finalize_downstream_spec_create_new=finalize,
        resume_supervisor_once=lambda: events.append("resume"),
        send_ready_line_once=lambda _payload: events.append("ready"),
        receive_ack_line_once=receive,
        read_receipt_once=lambda _path, _maximum: (
            events.append("receipt") or authority.canonical_json_line_bytes(state["receipt"])
        ),
        finalize_success_once=lambda: events.append("finalize-success"),
        abort_supervisor_on_failure=lambda: events.append("abort"),
    )
    result = authority.orchestrate_original_confirmatory_q_e_custody_once(
        e_source,
        contract=contract,
        controller_process_identity=ready["controller_process_identity"],
        supervisor_job_id=ready["supervisor_job_id"],
        independent_verifier_receipt_sha256=ready["independent_verifier_receipt_sha256"],
        start_supervisor_suspended_once=lambda: events.append("start") or target,
    )
    assert result == (
        ready,
        receipt,
        authority.build_original_confirmatory_q_e_custody_ack(
            contract=contract,
            ready=ready,
            receipt=receipt,
        ),
    )
    assert events == [
        "start",
        "duplicate",
        "spec",
        "resume",
        "ready",
        "ack",
        "receipt",
        "close-e",
        "close-q",
        "finalize-success",
    ]


def test_q_e_one_shot_invalid_ack_aborts_without_closing_sources(
    capsule_fixture: CapsuleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, ready, receipt, process_identity = _q_e_handshake_values(capsule_fixture)
    events: list[str] = []

    class Source:
        role = "independent-verifier"
        closed = False

        def require_active(self) -> None:
            assert not self.closed

        def close(self) -> None:
            self.closed = True
            events.append("closed")

    q_source = Source()
    e_source = Source()
    e_source.path = Path(ready["e_leaf_physical_identity"]["path"])
    e_source.q_custody = q_source
    monkeypatch.setattr(
        authority,
        "_require_current_q_e_controller_identity",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_exact_q_e_process_handle_identity",
        lambda _handle, value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_q_e_independent_review_receipt",
        lambda *_args, **_kwargs: ({}, b""),
    )
    monkeypatch.setattr(
        authority,
        "duplicate_original_confirmatory_q_e_custody_to_supervisor",
        lambda *_args, **_kwargs: ready,
    )
    target = authority.OriginalConfirmatoryQESuspendedSupervisorTarget(
        process_handle=999,
        process_identity=process_identity,
        windows_boot_time_utc="2026-07-30T00:00:00Z",
        suspended=True,
        transport=authority.Q_E_CUSTODY_TRANSPORT,
        exact_job_object_membership_verified=True,
        bounded_anonymous_pipes_created_before_process=True,
        automatic_retry_allowed=False,
        finalize_downstream_spec_create_new=lambda *_args: None,
        resume_supervisor_once=lambda: None,
        send_ready_line_once=lambda _payload: None,
        receive_ack_line_once=lambda _maximum: b'{"bad":true}\n',
        read_receipt_once=lambda _path, _maximum: authority.canonical_json_line_bytes(receipt),
        finalize_success_once=lambda: events.append("finalize-success"),
        abort_supervisor_on_failure=lambda: events.append("abort"),
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.orchestrate_original_confirmatory_q_e_custody_once(
            e_source,
            contract=contract,
            controller_process_identity=ready["controller_process_identity"],
            supervisor_job_id=ready["supervisor_job_id"],
            independent_verifier_receipt_sha256=ready["independent_verifier_receipt_sha256"],
            start_supervisor_suspended_once=lambda: target,
        )
    assert events == ["abort"]
    assert not e_source.closed
    assert not q_source.closed


@pytest.mark.parametrize(
    ("failure_point", "expected_events"),
    [
        ("source-close", ["close-e"]),
        ("finalize-success", ["close-e", "close-q", "finalize-success"]),
    ],
)
def test_q_e_post_ack_cleanup_failure_is_permanent_ambiguous_stop_without_abort(
    capsule_fixture: CapsuleFixture,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_events: list[str],
) -> None:
    contract, ready, receipt, process_identity = _q_e_handshake_values(capsule_fixture)
    events: list[str] = []

    class Source:
        role = "independent-verifier"

        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def require_active(self) -> None:
            assert not self.closed

        def close(self) -> None:
            events.append(f"close-{self.name}")
            if self.name == "e" and failure_point == "source-close":
                raise OSError("synthetic source-close failure")
            self.closed = True

    q_source = Source("q")
    e_source = Source("e")
    e_source.path = Path(ready["e_leaf_physical_identity"]["path"])
    e_source.q_custody = q_source
    monkeypatch.setattr(
        authority,
        "_require_current_q_e_controller_identity",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_exact_q_e_process_handle_identity",
        lambda _handle, value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        authority,
        "_require_q_e_independent_review_receipt",
        lambda *_args, **_kwargs: ({}, b""),
    )
    monkeypatch.setattr(
        authority,
        "duplicate_original_confirmatory_q_e_custody_to_supervisor",
        lambda *_args, **_kwargs: ready,
    )

    def finalize_success() -> None:
        events.append("finalize-success")
        if failure_point == "finalize-success":
            raise OSError("synthetic success-finalization failure")

    target = authority.OriginalConfirmatoryQESuspendedSupervisorTarget(
        process_handle=999,
        process_identity=process_identity,
        windows_boot_time_utc="2026-07-30T00:00:00Z",
        suspended=True,
        transport=authority.Q_E_CUSTODY_TRANSPORT,
        exact_job_object_membership_verified=True,
        bounded_anonymous_pipes_created_before_process=True,
        automatic_retry_allowed=False,
        finalize_downstream_spec_create_new=lambda *_args: None,
        resume_supervisor_once=lambda: None,
        send_ready_line_once=lambda _payload: None,
        receive_ack_line_once=lambda _maximum: authority.canonical_json_line_bytes(
            authority.build_original_confirmatory_q_e_custody_ack(
                contract=contract,
                ready=ready,
                receipt=receipt,
            )
        ),
        read_receipt_once=lambda _path, _maximum: authority.canonical_json_line_bytes(receipt),
        finalize_success_once=finalize_success,
        abort_supervisor_on_failure=lambda: events.append("abort"),
    )

    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="permanent ambiguous STOP",
    ):
        authority.orchestrate_original_confirmatory_q_e_custody_once(
            e_source,
            contract=contract,
            controller_process_identity=ready["controller_process_identity"],
            supervisor_job_id=ready["supervisor_job_id"],
            independent_verifier_receipt_sha256=(ready["independent_verifier_receipt_sha256"]),
            start_supervisor_suspended_once=lambda: target,
        )

    assert events == expected_events
    assert "abort" not in events
    if failure_point == "source-close":
        assert not e_source.closed
        assert not q_source.closed
    else:
        assert e_source.closed
        assert q_source.closed


def test_q_payload_bound_is_checked_before_create_new(
    capsule_fixture: CapsuleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    q_path = Path(q["q_path"])
    monkeypatch.setattr(authority, "MAX_CONTROL_FILE_BYTES", 16)
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.publish_original_confirmatory_q_replacement_v2_once(q)
    assert not q_path.exists()


def test_environment_envelope_and_process_binding_are_closed() -> None:
    envelope = _environment()
    assert len(envelope.as_dict()) == 16
    assert (
        envelope.supervisor_environment_sha256
        == "4b67fbb59522c8590ef8b51b80b79e9dbe46d2f0389d5700a7ece7d482327493"
    )
    assert (
        envelope.exact_environment_sha256
        == "021fcc1c62f0445a7e930b40d39337dac72ebcd78694ead350e3c0aaf4fe4ad0"
    )
    assert (
        envelope.launch_environment_root_sha256
        == "44f93486f32d4293b856be0764c57cff978009cf8d730a1a84cf41af053cb622"
    )
    assert (
        envelope.envelope_sha256
        == "92542d153f53f62edfea69b5f6845c844c7cd73270a5de3fdfbbb3416c82f822"
    )
    assert (
        authority.canonical_expected_launch_environment_envelope_v1(envelope.as_dict()) == envelope
    )
    binding = authority.build_original_confirmatory_process_environment_binding(envelope)
    assert binding.exact_environment_sha256 == binding.exact_integrity_verifier_environment_sha256
    assert (
        authority.canonical_original_confirmatory_process_environment_binding(
            binding.as_dict(),
            expected_environment=envelope,
        )
        == binding
    )

    extra = envelope.as_dict()
    extra["ambient_allowed"] = True
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_expected_launch_environment_envelope_v1(extra)

    wrong_nonce = envelope.as_dict()
    wrong_nonce["attempt_nonce"] = "A" * 64
    wrong_nonce = _rehash_without_self(wrong_nonce, "envelope_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_expected_launch_environment_envelope_v1(wrong_nonce)

    wrong_binding = binding.as_dict()
    wrong_binding["exact_integrity_verifier_environment_sha256"] = "f" * 64
    wrong_binding = _rehash_without_self(wrong_binding, "binding_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_process_environment_binding(
            wrong_binding,
            expected_environment=envelope,
        )


@pytest.mark.parametrize(
    ("supervisor_environment", "child_environment"),
    [
        ({"Path": "x"}, {"AANCA_SUPERVISOR_ATTEMPT_NONCE": "a" * 64}),
        ({"BAD=NAME": "x"}, {"AANCA_SUPERVISOR_ATTEMPT_NONCE": "a" * 64}),
        ({"PATH": "x\x00"}, {"AANCA_SUPERVISOR_ATTEMPT_NONCE": "a" * 64}),
        ({"PATH": "x"}, {"PATH": "x"}),
    ],
)
def test_environment_rejects_invalid_names_values_and_missing_nonce(
    supervisor_environment: dict[str, str],
    child_environment: dict[str, str],
) -> None:
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_expected_launch_environment_envelope_v1(
            attempt_nonce="a" * 64,
            supervisor_environment=supervisor_environment,
            child_environment=child_environment,
        )


def test_leaf_and_ancestor_leases_are_exact(capsule_fixture: CapsuleFixture) -> None:
    leaf = capsule_fixture.leaf_lease
    ancestor = capsule_fixture.ancestor_lease
    assert authority.canonical_original_confirmatory_capsule_lease_identity(leaf.as_dict()) == leaf
    assert (
        authority.canonical_original_confirmatory_capsule_ancestor_lease(ancestor.as_dict())
        == ancestor
    )
    assert ancestor.record_count == 4
    assert Path(str(ancestor.records[-1]["path"])) == capsule_fixture.capsule_path.parent

    leaf_extra = leaf.as_dict()
    leaf_extra["path_components_non_reparse"] = True
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_capsule_lease_identity(leaf_extra)

    ancestor_reparse = ancestor.as_dict()
    ancestor_reparse["records"][2]["reparse_point"] = True
    ancestor_reparse["records_root_sha256"] = authority.canonical_json_sha256(
        ancestor_reparse["records"]
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_capsule_ancestor_lease(ancestor_reparse)


@pytest.mark.parametrize(
    ("role", "disposition"),
    sorted(authority._RETAINED_HANDLE_DISPOSITION_BY_ROLE.items()),
)
def test_retained_handle_roles_have_exact_access_and_disposition(
    tmp_path: Path,
    role: str,
    disposition: str,
) -> None:
    identity = _synthetic_physical_identity(tmp_path, role)
    raw = {
        "schema_version": 1,
        "policy": authority.RETAINED_NATIVE_HANDLE_BINDING_POLICY,
        "role": role,
        "physical_identity": identity.as_dict(),
        "owner_pid": 1,
        "owner_creation_time_100ns": 2,
        "owner_windows_boot_time_utc": "2026-07-30T00:00:00Z",
        "handle_slot": 3,
        "access_mask": 0x80000000,
        "share_mode": authority._RETAINED_HANDLE_SHARE_MODE_BY_ROLE[role],
        "retained_handle_active": True,
        "acquisition_disposition": disposition,
    }
    assert authority.canonical_original_confirmatory_retained_handle_binding(raw).as_dict() == raw
    for field, replacement in (
        ("access_mask", 0xC0000000),
        (
            "share_mode",
            1 if authority._RETAINED_HANDLE_SHARE_MODE_BY_ROLE[role] == 3 else 3,
        ),
        ("acquisition_disposition", "arbitrary_but_well_formed_v1"),
    ):
        tampered = dict(raw)
        tampered[field] = replacement
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.canonical_original_confirmatory_retained_handle_binding(tampered)


def test_execution_capsule_cross_binds_path_content_and_both_leases(
    capsule_fixture: CapsuleFixture,
) -> None:
    capsule = capsule_fixture.capsule
    assert authority.canonical_original_confirmatory_execution_capsule(capsule.as_dict()) == capsule
    assert capsule.capsule_lease_identity_root_sha256 == authority.canonical_json_sha256(
        capsule_fixture.leaf_lease.as_dict()
    )
    assert capsule.capsule_ancestor_lease_root_sha256 == authority.canonical_json_sha256(
        capsule_fixture.ancestor_lease.as_dict()
    )

    wrong_root = capsule.as_dict()
    wrong_root["capsule_ancestor_lease_root_sha256"] = "0" * 64
    wrong_root = _rehash_without_self(wrong_root, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_execution_capsule(wrong_root)

    wrong_flags = capsule.as_dict()
    wrong_flags["python_isolated_flags"] = ["-B", "-I"]
    wrong_flags = _rehash_without_self(wrong_flags, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_execution_capsule(wrong_flags)


def test_q_replacement_v2_round_trips_and_closes_external_release(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    assert authority.canonical_original_confirmatory_q_replacement_v2(q) == q
    assert q["q_authority_root_sha256"] == authority.canonical_json_sha256(
        {key: item for key, item in q.items() if key != "q_authority_root_sha256"}
    )
    assert q["supervisor_release"]["supervisor_source_sha256"] == "d" * 64
    assert q["supervisor_release"]["q_e_custody_contract_policy"] == (
        authority.Q_E_CUSTODY_CONTRACT_POLICY
    )
    assert q["supervisor_release"]["q_e_custody_ack_policy"] == (authority.Q_E_CUSTODY_ACK_POLICY)
    assert q["supervisor_release"]["q_e_custody_receipt_filename"] == (
        authority.Q_E_CUSTODY_RECEIPT_FILENAME
    )
    assert len(q["supervisor_release"]) == 48
    assert q["supervisor_release"]["external_codex_handoff_policy"] == (
        authority.EXTERNAL_CODEX_HANDOFF_POLICY
    )
    assert q["supervisor_release"]["internal_codex_wake_disposition"] == (
        authority.INTERNAL_CODEX_WAKE_DISPOSITION
    )
    assert {
        "codex_terminal_wake_prompt_render_policy",
        "codex_terminal_wake_prompt_template_sha256",
        "codex_terminal_wake_prompt_template_root_sha256",
        "codex_terminal_wake_prompt_template_projection",
    }.isdisjoint(q["supervisor_release"])
    assert (
        q["execution_capsule"]["python_lease_identity_root_sha256"]
        == capsule_fixture.capsule.python_lease_identity_root_sha256
    )

    tampered = dict(q)
    tampered["supervisor_release"] = dict(q["supervisor_release"])
    tampered["supervisor_release"]["supervisor_source_sha256"] = "f" * 64
    tampered["q_authority_root_sha256"] = authority.canonical_json_sha256(
        {key: item for key, item in tampered.items() if key != "q_authority_root_sha256"}
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_replacement_v2(tampered)


def test_supervisor_command_is_base_runtime_direct_and_exactly_projected(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    release = q["supervisor_release"]
    derivation = release["supervisor_process_command_derivation_contract"]
    spec_path = Path(q["control_staging_projection"]["supervisor_launch_spec_path"])
    staged_e_intent_path = Path(q["control_staging_projection"]["e_intent_path"])
    projection = authority.build_original_confirmatory_supervisor_process_command_projection(
        derivation,
        capsule=capsule_fixture.capsule,
        supervisor_launch_spec_path=spec_path,
        staged_e_intent_path=staged_e_intent_path,
    )
    runtime_path = str(capsule_fixture.capsule.runtime_python_path)
    assert derivation["program_path"] == runtime_path
    assert derivation["createprocess_application_path"] == runtime_path
    assert derivation["logical_venv_python_path"] == str(capsule_fixture.capsule.python_path)
    assert projection["python_interpreter_flags"] == ["-I", "-S", "-B"]
    assert projection["os_launch_vector"] == [
        runtime_path,
        "-I",
        "-S",
        "-B",
        release["supervisor_source_path"],
        "--root",
        release["supervisor_state_root"],
        "run",
        "--staged-launch-spec",
        str(spec_path),
        "--staged-e-intent",
        str(staged_e_intent_path),
    ]
    assert "--spec" not in projection["os_launch_vector"]
    assert projection["expected_live_peb_argv"] == projection["os_launch_vector"]
    assert derivation["command_preimage_policy"] == (
        "original_confirmatory_supervisor_process_command_v2"
    )
    assert projection["command_preimage"]["policy"] == (
        "original_confirmatory_supervisor_process_command_v2"
    )
    assert derivation["supervisor_launcher_used_for_authorized_process_launch"] is False
    assert (
        authority.canonical_original_confirmatory_supervisor_process_command_projection(
            projection,
            derivation_contract=derivation,
            capsule=capsule_fixture.capsule,
            supervisor_launch_spec_path=spec_path,
            staged_e_intent_path=staged_e_intent_path,
        )
        == projection
    )

    tampered = deepcopy(projection)
    tampered["os_launch_vector"][1] = "-E"
    tampered["command_preimage"]["os_launch_vector"][1] = "-E"
    tampered["command_sha256"] = authority.canonical_json_sha256(tampered["command_preimage"])
    tampered = _rehash_without_self(tampered, "projection_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_process_command_projection(
            tampered,
            derivation_contract=derivation,
            capsule=capsule_fixture.capsule,
            supervisor_launch_spec_path=spec_path,
            staged_e_intent_path=staged_e_intent_path,
        )

    compact_factory_command_sha256 = authority.canonical_json_sha256(
        {
            "program_path": projection["program_path"],
            "program_sha256": projection["program_sha256"],
            "argv": projection["os_launch_vector"][1:],
            "cwd": projection["cwd"],
        }
    )
    assert compact_factory_command_sha256 != projection["command_sha256"]
    cross_domain_swap = deepcopy(projection)
    cross_domain_swap["command_sha256"] = compact_factory_command_sha256
    cross_domain_swap = _rehash_without_self(
        cross_domain_swap,
        "projection_root_sha256",
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_process_command_projection(
            cross_domain_swap,
            derivation_contract=derivation,
            capsule=capsule_fixture.capsule,
            supervisor_launch_spec_path=spec_path,
            staged_e_intent_path=staged_e_intent_path,
        )


def test_q20_release48_recursive_aliases_reject_after_coherent_reseal(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    release = q["supervisor_release"]
    derivation = release["supervisor_process_command_derivation_contract"]
    attempt = q["attempt_identity_projection"]
    control_staging = q["control_staging_projection"]

    assert len(q) == 20
    assert len(release) == 48
    assert len(derivation) == 39
    assert len(attempt) == 10
    assert len(control_staging) == 11
    assert len(authority._STATIC_RUNNER_BINDING_FIELDS) == 24
    assert len(authority._PUBLISHED_TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS) == 10
    assert len(authority._TECHNICAL_AUTHORITY_LIFECYCLE_BINDING_FIELDS) == 22

    process_alias = deepcopy(derivation)
    process_alias["schema_version"] = 2.0
    process_alias = _rehash_without_self(process_alias, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_process_command_derivation_contract(
            process_alias,
            capsule=capsule_fixture.capsule,
        )

    release_alias = deepcopy(release)
    release_alias["q_e_independent_verifier_receipt_required"] = 1
    release_alias = _rehash_without_self(release_alias, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_release_binding(
            release_alias,
            capsule=capsule_fixture.capsule,
        )

    control_alias = deepcopy(control_staging)
    control_alias["schema_version"] = 2.0
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_control_staging_projection(
            control_alias,
            job_id=attempt["job_id"],
            expected_sha256=authority.canonical_json_sha256(control_alias),
        )

    q_alias = deepcopy(q)
    q_alias["schema_version"] = 2.0
    base_projection = {field: q_alias[field] for field in authority._Q_BASE_AUTHORITY_FIELDS}
    base_root = authority.canonical_json_sha256(base_projection)
    alias_attempt = authority.build_original_confirmatory_q_attempt_identity_projection(
        attempt_id=attempt["attempt_id"],
        q_base_authority_root_sha256=base_root,
    )
    alias_control = authority.build_original_confirmatory_control_staging_projection(
        supervisor_state_root=release["supervisor_state_root"],
        job_id=alias_attempt["job_id"],
    )
    q_alias["q_base_authority_root_sha256"] = base_root
    q_alias["attempt_identity_projection"] = alias_attempt
    q_alias["attempt_identity_root_sha256"] = alias_attempt["attempt_identity_root_sha256"]
    q_alias["control_staging_projection"] = alias_control
    q_alias["control_staging_projection_sha256"] = authority.canonical_json_sha256(alias_control)
    q_alias = _rehash_without_self(q_alias, "q_authority_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_q_replacement_v2(q_alias)


def test_release48_rejects_stale_alias_and_overlapping_split_roots(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    release = q["supervisor_release"]

    stale_alias = deepcopy(release)
    stale_alias["supervisor_root"] = release["supervisor_state_root"]
    stale_alias = _rehash_without_self(stale_alias, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_release_binding(
            stale_alias,
            capsule=capsule_fixture.capsule,
        )

    overlapping = deepcopy(release)
    overlapping_derivation = overlapping["supervisor_process_command_derivation_contract"]
    overlapping["supervisor_state_root"] = overlapping["supervisor_code_root"]
    overlapping_derivation["supervisor_state_root"] = overlapping["supervisor_code_root"]
    overlapping_derivation["python_sys_argv_prefix"][2] = overlapping["supervisor_code_root"]
    overlapping_derivation = _rehash_without_self(
        overlapping_derivation,
        "contract_sha256",
    )
    overlapping["supervisor_process_command_derivation_contract"] = overlapping_derivation
    overlapping["supervisor_process_command_derivation_contract_sha256"] = overlapping_derivation[
        "contract_sha256"
    ]
    overlapping = _rehash_without_self(overlapping, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_release_binding(
            overlapping,
            capsule=capsule_fixture.capsule,
        )


@pytest.mark.parametrize("shape", ["stale_exact48", "hybrid_exact49", "hybrid_exact48"])
def test_release48_rejects_stale_or_hybrid_internal_codex_wake_fields(
    capsule_fixture: CapsuleFixture,
    shape: str,
) -> None:
    release = deepcopy(_q_replacement_v2(capsule_fixture)["supervisor_release"])
    old = {
        "codex_terminal_wake_prompt_render_policy": "legacy-render",
        "codex_terminal_wake_prompt_template_sha256": "6" * 64,
        "codex_terminal_wake_prompt_template_root_sha256": "7" * 64,
        "codex_terminal_wake_prompt_template_projection": {},
    }
    if shape == "stale_exact48":
        for field in (
            "external_codex_handoff_policy",
            "external_codex_handoff_authority_spec_file_sha256",
            "external_codex_handoff_authority_spec_canonical_root_sha256",
            "internal_codex_wake_disposition",
        ):
            release.pop(field)
        release.update(old)
    elif shape == "hybrid_exact49":
        release["codex_terminal_wake_prompt_render_policy"] = "legacy-render"
    else:
        release.pop("internal_codex_wake_disposition")
        release["codex_terminal_wake_prompt_render_policy"] = "legacy-render"
    release = _rehash_without_self(release, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_supervisor_release_binding(
            release,
            capsule=capsule_fixture.capsule,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "external_control_plane_publication_id",
            f"cpr-{'2' * 32}",
            "attestation path",
        ),
        (
            "external_control_plane_release_qualification_attestation_path",
            f"C:\\AANCA-control-plane-release-v2\\verifications\\cpr-{'1' * 32}\\other.json",
            "attestation path",
        ),
        (
            "external_control_plane_release_qualification_attestation_file_sha256",
            "g" * 64,
            "attestation file must be",
        ),
        (
            "external_control_plane_release_qualification_attestation_root_sha256",
            True,
            "attestation root must be",
        ),
    ],
)
def test_release48_rejects_invalid_attestation_transport_after_outer_reseal(
    capsule_fixture: CapsuleFixture,
    field: str,
    replacement: object,
    error: str,
) -> None:
    release = deepcopy(_q_replacement_v2(capsule_fixture)["supervisor_release"])
    release[field] = replacement
    release = _rehash_without_self(release, "contract_sha256")

    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError, match=error):
        authority.canonical_original_confirmatory_supervisor_release_binding(
            release,
            capsule=capsule_fixture.capsule,
        )


def test_terminal_client_release_binds_source_base_runtime_and_child_topology(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    release = q["supervisor_release"]["terminal_client_launcher_release"]
    assert (
        authority.canonical_original_confirmatory_terminal_client_launcher_release(
            release,
            capsule=capsule_fixture.capsule,
        )
        == release
    )
    assert release["source_path"] == str(
        Path(release["supervisor_root"]) / authority.TERMINAL_CLIENT_LAUNCHER_SOURCE_FILENAME
    )
    assert release["program_path"] == str(capsule_fixture.capsule.runtime_python_path)
    assert release["logical_venv_python_path"] == str(capsule_fixture.capsule.python_path)
    assert release["process_argv_prefix"][:4] == [
        str(capsule_fixture.capsule.runtime_python_path),
        "-I",
        "-S",
        "-B",
    ]
    assert release["verify_terminal_child_launch_topology"] == (
        "launcher_base_direct_to_venv_redirector_to_runtime_child_v1"
    )
    assert release["verify_terminal_immediate_redirector_program_path"] == str(
        capsule_fixture.capsule.python_path
    )
    assert release["verify_terminal_runtime_child_program_path"] == str(
        capsule_fixture.capsule.runtime_python_path
    )
    assert release["launch_intent_schema_version"] == 1
    assert release["launch_intent_status"] == ("reserved_before_verify_terminal_createprocess")
    assert release["launch_intent_create_disposition"] == "CREATE_NEW"
    assert "intent_root_sha256" in release["launch_intent_field_names"]
    assert release["launch_intent_physical_identity_role"] == ("terminal-client-launch-intent")
    assert release["launch_intent_physical_identity_policy"] == (
        authority.NO_FOLLOW_PHYSICAL_FILE_IDENTITY_POLICY
    )
    assert "sha256" in release["launch_intent_physical_identity_field_names"]
    assert release["launch_intent_created_before_child_process_required"] is True
    assert release["existing_or_partial_launch_intent_is_stop"] is True
    assert (
        release[
            "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required"
        ]
        is True
    )
    assert release["immediate_venv_redirector_live_through_runtime_child_exit_required"] is True
    assert release["terminal_client_launcher_live_through_redirector_waitforexit_required"] is True
    assert release["process_liveness_reverified_at_final_ack_required"] is True
    assert release["child_stdio_policy"] == authority.TERMINAL_CLIENT_CHILD_STDIO_POLICY
    assert release["child_inherited_handle_count"] == 3
    assert release["createprocess_inherit_handles"] is True
    assert release["startupinfoex_use_std_handles_required"] is True
    assert release["proc_thread_attribute_handle_list_required"] is True
    assert release["non_stdio_inherited_handles_allowed"] is False
    assert release["preterminal_or_terminal_input_file_handles_inherited_allowed"] is False
    assert release["child_stdout_single_canonical_json_line_required"] is True
    assert release["child_stderr_empty_required"] is True
    assert release["stdio_pipe_drains_event_driven_concurrent_required"] is True
    assert (
        release["launch_intent_supervisor_granted_access_mask"]
        == authority.FILE_GENERIC_READ_ACCESS_MASK
    )
    assert (
        release["launch_intent_child_duplicate_target_access_mask"]
        == authority.GENERIC_READ_ACCESS_REQUEST
    )
    assert release["launch_intent_child_duplicate_options"] == 0
    assert release["launch_intent_child_duplicate_close_source"] is False
    terminal_template = (
        authority.build_original_confirmatory_terminal_custody_authority_template_projection()
    )
    launcher_contract = terminal_template["terminal_client_launcher_contract"]
    assert launcher_contract["launch_intent_field_names"] == release["launch_intent_field_names"]
    assert launcher_contract["launch_intent_status"] == release["launch_intent_status"]
    final_ack_fields = terminal_template["message_contracts"]["FINAL_ACK"]["field_names"]
    assert "launcher_redirector_child_process_handles_retained_through_ack" in (final_ack_fields)
    assert "immediate_venv_redirector_process_identity_reverified" in final_ack_fields
    assert "terminal_client_launcher_process_identity_reverified" in final_ack_fields
    assert "launcher_redirector_child_grandparent_chain_reverified" in final_ack_fields
    assert "launcher_redirector_child_same_supervisor_job_reverified" in final_ack_fields
    assert "immediate_venv_redirector_live_at_final_ack" in final_ack_fields
    assert "terminal_client_launcher_live_at_final_ack" in final_ack_fields

    tampered = deepcopy(release)
    tampered["source_size_bytes"] += 1
    tampered = _rehash_without_self(tampered, "release_root_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_client_launcher_release(
            tampered,
            capsule=capsule_fixture.capsule,
        )


def test_e_launcher_projection_is_acyclic_and_final_command_is_deterministic(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    e = _e_intent(capsule_fixture, q)
    e_file_sha256 = authority.canonical_json_line_sha256(e)
    custody_projection = e["job"]["terminal_custody_authority_projection"]
    launcher_projection = custody_projection["terminal_client_launcher_projection"]
    launcher_release = q["supervisor_release"]["terminal_client_launcher_release"]
    assert "$SUPERVISOR_SPEC_SHA256" in launcher_projection["process_argv_template"]
    assert "$E_INTENT_FILE_SHA256" in launcher_projection["process_argv_template"]
    assert "$TERMINAL_RECEIPT_SHA256" in launcher_projection["process_argv_template"]
    assert "$VERIFY_TERMINAL_COMMAND_SHA256" in launcher_projection["process_argv_template"]
    for forbidden in (
        "supervisor_spec_sha256",
        "terminal_receipt_sha256",
        "terminal_client_launcher_command_sha256",
        "wake_intent_sha256",
    ):
        assert not authority._contains_mapping_key(launcher_projection, forbidden)
    assert launcher_projection["wake_intent_hash_in_launcher_argv_allowed"] is False
    assert launcher_projection["preterminal_pin_terminal_or_lease_input_read_allowed"] is False
    assert (
        launcher_projection[
            "supervisor_retains_launcher_redirector_child_process_handles_through_final_ack_required"
        ]
        is True
    )
    assert (
        launcher_projection["immediate_venv_redirector_live_through_runtime_child_exit_required"]
        is True
    )
    assert (
        launcher_projection["terminal_client_launcher_live_through_redirector_waitforexit_required"]
        is True
    )
    assert launcher_projection["process_liveness_reverified_at_final_ack_required"] is True

    child = authority.derive_original_confirmatory_capsule_command_from_e(
        e_intent=e,
        e_file_sha256=e_file_sha256,
        q_authority=q,
        capsule_mode=authority.CAPSULE_TERMINAL_MODE,
    )
    launcher_command = authority.build_original_confirmatory_terminal_client_launcher_command(
        launcher_projection=launcher_projection,
        launcher_release=launcher_release,
        capsule=capsule_fixture.capsule,
        supervisor_spec_sha256="1" * 64,
        e_intent_file_sha256=e_file_sha256,
        terminal_receipt_sha256="2" * 64,
        verify_terminal_command=child,
    )
    assert launcher_command["process_argv"][:4] == [
        str(capsule_fixture.capsule.runtime_python_path),
        "-I",
        "-S",
        "-B",
    ]
    assert launcher_command["python_sys_argv"][0] == launcher_release["source_path"]
    assert child.command_sha256 in launcher_command["process_argv"]
    assert not authority._contains_mapping_key(launcher_command, "wake_intent_sha256")
    assert (
        authority.canonical_original_confirmatory_terminal_client_launcher_command(
            launcher_command,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
        )
        == launcher_command
    )
    launcher_creation = 116444736000000000
    launcher_process_identity = {
        "pid": 321,
        "creation_time_100ns": launcher_creation,
        "creation_time_utc": authority._windows_filetime_100ns_to_utc(launcher_creation),
        "program_path": launcher_command["program_path"],
        "program_sha256": launcher_command["program_sha256"],
        "command_sha256": launcher_command["command_sha256"],
    }
    launch_intent = authority.build_original_confirmatory_terminal_client_launch_intent(
        launcher_command=launcher_command,
        launcher_projection=launcher_projection,
        launcher_release=launcher_release,
        capsule=capsule_fixture.capsule,
        verify_terminal_command=child,
        launcher_process_identity=launcher_process_identity,
        created_at_utc="2026-07-30T12:00:00Z",
    )
    assert launch_intent["status"] == "reserved_before_verify_terminal_createprocess"
    assert launch_intent["create_disposition"] == "CREATE_NEW"
    assert launch_intent["launch_attempt_count"] == 1
    assert launch_intent["child_process_created_before_intent"] is False
    assert launch_intent["existing_or_partial_intent_is_stop"] is True
    assert launch_intent["automatic_retry_allowed"] is False
    assert (
        launch_intent["verify_terminal_command_projection_sha256"]
        == launcher_projection["verify_terminal_command_projection_sha256"]
    )
    assert (
        authority.canonical_original_confirmatory_terminal_client_launch_intent(
            launch_intent,
            launcher_command=launcher_command,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
        )
        == launch_intent
    )
    tampered_intent = deepcopy(launch_intent)
    tampered_intent["launch_attempt_count"] = 2
    tampered_intent = _rehash_without_self(
        tampered_intent,
        "intent_root_sha256",
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_client_launch_intent(
            tampered_intent,
            launcher_command=launcher_command,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
        )
    replaced_path_intent = deepcopy(launch_intent)
    replaced_path_intent["terminal_receipt_path"] = str(
        capsule_fixture.anchor / "replacement-terminal.json"
    )
    replaced_path_intent = _rehash_without_self(
        replaced_path_intent,
        "intent_root_sha256",
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_client_launch_intent(
            replaced_path_intent,
            launcher_command=launcher_command,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
        )
    wrong_launcher_identity = dict(launcher_process_identity)
    wrong_launcher_identity["command_sha256"] = "f" * 64
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.build_original_confirmatory_terminal_client_launch_intent(
            launcher_command=launcher_command,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
            launcher_process_identity=wrong_launcher_identity,
            created_at_utc="2026-07-30T12:00:00Z",
        )

    tampered = deepcopy(launcher_command)
    tampered["process_argv"][-1] = str(capsule_fixture.anchor / "wrong-cwd")
    tampered["command_sha256"] = authority.canonical_json_sha256(
        {key: item for key, item in tampered.items() if key != "command_sha256"}
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_client_launcher_command(
            tampered,
            launcher_projection=launcher_projection,
            launcher_release=launcher_release,
            capsule=capsule_fixture.capsule,
            verify_terminal_command=child,
        )


def test_q_publication_uses_overlapping_author_and_independent_verifier_custody(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    path, file_sha256, identity, author_custody = (
        authority.publish_original_confirmatory_q_replacement_v2_once(q)
    )
    verifier_custody = None
    try:
        canonical, observed_sha256, verifier_custody = (
            authority.require_original_confirmatory_q_replacement_v2(
                path,
                expected_file_sha256=file_sha256,
                expected_publication_identity=identity,
                author_custody=author_custody,
            )
        )
        assert canonical == q
        assert observed_sha256 == file_sha256
        author_custody.close()
        verifier_custody.require_active()
        with pytest.raises(FileExistsError):
            authority.publish_original_confirmatory_q_replacement_v2_once(q)
    finally:
        author_custody.close()
        if verifier_custody is not None:
            verifier_custody.close()


@pytest.mark.parametrize(
    ("mode", "execution_mode"),
    [
        (authority.CAPSULE_SCIENTIFIC_MODE, "fresh"),
        (authority.CAPSULE_PRETERMINAL_MODE, "fresh"),
        (authority.CAPSULE_TERMINAL_MODE, "successor_resume"),
    ],
)
def test_mode_tails_are_exact_and_round_trip(
    capsule_fixture: CapsuleFixture,
    mode: str,
    execution_mode: str,
) -> None:
    tail = _tail(capsule_fixture, mode, execution_mode=execution_mode)
    assert (
        authority.canonical_original_confirmatory_capsule_mode_tail(
            capsule_mode=mode,
            tail_argv=tail,
        )
        == tail
    )
    if execution_mode == "successor_resume":
        assert "--retry-of-run-id" in tail
    else:
        assert "--retry-of-run-id" not in tail

    reordered = list(tail)
    reordered[0], reordered[2] = reordered[2], reordered[0]
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_capsule_mode_tail(
            capsule_mode=mode,
            tail_argv=reordered,
        )

    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_capsule_mode_tail(
            capsule_mode=mode,
            tail_argv=[*tail, "--extra", "forbidden"],
        )


def test_fresh_tail_forbids_retry_lineage(capsule_fixture: CapsuleFixture) -> None:
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.original_confirmatory_capsule_mode_tail(
            capsule_mode=authority.CAPSULE_SCIENTIFIC_MODE,
            e_intent_path=capsule_fixture.anchor / "e.json",
            e_intent_sha256="1" * 64,
            e_intent_core_sha256="2" * 64,
            q_authority_root_sha256="3" * 64,
            launch_nonce="4" * 64,
            supervisor_job_id="job",
            supervisor_job_directory=capsule_fixture.anchor / "job",
            attempt_id="attempt",
            run_id="run",
            execution_mode="fresh",
            retry_of_run_id="forbidden",
        )


def test_command_is_exact_q_bound_argv(capsule_fixture: CapsuleFixture) -> None:
    tail = _tail(capsule_fixture, authority.CAPSULE_SCIENTIFIC_MODE)
    command = authority.build_original_confirmatory_capsule_command(
        capsule=capsule_fixture.capsule,
        mode=authority.CAPSULE_SCIENTIFIC_MODE,
        tail_argv=tail,
        cwd=capsule_fixture.anchor,
    )
    assert command.argv[:5] == (
        str(capsule_fixture.capsule.python_path),
        "-I",
        "-B",
        str(capsule_fixture.capsule_path),
        "run-confirmatory",
    )
    assert (
        authority.canonical_original_confirmatory_capsule_command(
            command.as_dict(),
            capsule=capsule_fixture.capsule,
            expected_mode=authority.CAPSULE_SCIENTIFIC_MODE,
            expected_tail_argv=tail,
        )
        == command
    )

    extra = command.as_dict()
    extra["argv"].extend(["--retry", "forbidden"])
    extra = _rehash_without_self(extra, "command_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_capsule_command(
            extra,
            capsule=capsule_fixture.capsule,
            expected_mode=authority.CAPSULE_SCIENTIFIC_MODE,
            expected_tail_argv=tail,
        )


def test_preterminal_contract_binds_verifier_and_receipt(
    capsule_fixture: CapsuleFixture,
) -> None:
    tail = _tail(capsule_fixture, authority.CAPSULE_PRETERMINAL_MODE)
    command = authority.build_original_confirmatory_capsule_command(
        capsule=capsule_fixture.capsule,
        mode=authority.CAPSULE_PRETERMINAL_MODE,
        tail_argv=tail,
        cwd=capsule_fixture.anchor,
    )
    pin_path = Path(tail[tail.index("--preterminal-pin") + 1])
    contract = authority.build_original_confirmatory_preterminal_pin_contract(
        capsule=capsule_fixture.capsule,
        verifier_command=command,
        verifier_command_tail_argv=tail,
        preterminal_pin_receipt_path=pin_path,
        preterminal_pin_receipt_max_bytes=64 * 1024,
    )
    assert contract.preterminal_pin_receipt_path == pin_path
    assert (
        authority.canonical_original_confirmatory_preterminal_pin_contract(
            contract.as_dict(),
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
        )
        == contract
    )

    wrong_pin = contract.as_dict()
    wrong_pin["preterminal_pin_receipt_path"] = str(capsule_fixture.anchor / "wrong-pin.json")
    wrong_pin = _rehash_without_self(wrong_pin, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_preterminal_pin_contract(
            wrong_pin,
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
        )

    equal_but_wrong_container_type = contract.as_dict()
    equal_but_wrong_container_type["verifier_command"] = _EqualDict(
        equal_but_wrong_container_type["verifier_command"]
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_preterminal_pin_contract(
            equal_but_wrong_container_type,
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
        )


def test_terminal_composition_contract_binds_all_future_paths_without_hash_cycle(
    capsule_fixture: CapsuleFixture,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    envelope = _environment()
    environment_binding = authority.build_original_confirmatory_process_environment_binding(
        envelope
    )
    tail = _tail(capsule_fixture, authority.CAPSULE_TERMINAL_MODE)
    command = authority.build_original_confirmatory_capsule_command(
        capsule=capsule_fixture.capsule,
        mode=authority.CAPSULE_TERMINAL_MODE,
        tail_argv=tail,
        cwd=capsule_fixture.anchor,
    )
    overlap, input_lease, seed, custody = _terminal_dependencies(
        capsule_fixture,
        command,
        environment_binding,
    )
    values = {
        flag: Path(tail[tail.index(flag) + 1])
        for flag in (
            "--supervisor-terminal",
            "--verifier-stdout",
            "--preterminal-pin",
            "--composed-terminal",
        )
    }
    expected_run_directory = capsule_fixture.anchor / "artifacts" / "runs" / "run-1"
    terminal_custody_projection = _terminal_custody_projection_for_command(
        capsule_fixture,
        q=q,
        command=command,
        tail=tail,
        environment_binding=environment_binding,
        expected_run_directory=expected_run_directory,
    )
    launcher_release = q["supervisor_release"]["terminal_client_launcher_release"]
    contract = authority.build_original_confirmatory_terminal_composition_contract(
        capsule=capsule_fixture.capsule,
        verifier_command=command,
        verifier_command_tail_argv=tail,
        preterminal_pin_contract_sha256="9" * 64,
        preterminal_overlap_handshake_contract=overlap,
        postwake_input_lease_contract=input_lease,
        postwake_custody_seed=seed,
        postwake_custody_handshake_contract=custody,
        expected_run_directory=expected_run_directory,
        expected_terminal_custody_authority_projection=terminal_custody_projection,
        terminal_client_launcher_release=launcher_release,
        expected_environment=envelope,
        process_environment_binding=environment_binding,
        supervisor_terminal_receipt_path=values["--supervisor-terminal"],
        supervisor_terminal_receipt_max_bytes=1024 * 1024,
        verifier_stdout_path=values["--verifier-stdout"],
        verifier_stdout_max_bytes=64 * 1024,
        preterminal_pin_receipt_path=values["--preterminal-pin"],
        preterminal_pin_receipt_max_bytes=1024 * 1024,
        postwake_input_lease_receipt_max_bytes=1024 * 1024,
        postwake_composed_readback_receipt_max_bytes=1024 * 1024,
        composed_terminal_receipt_path=values["--composed-terminal"],
        composed_terminal_receipt_max_bytes=1024 * 1024,
    )
    assert contract.composed_terminal_receipt_path == values["--composed-terminal"]
    assert (
        authority.canonical_original_confirmatory_terminal_composition_contract(
            contract.as_dict(),
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
            preterminal_pin_contract_sha256="9" * 64,
            preterminal_overlap_handshake_contract=overlap,
            postwake_input_lease_contract=input_lease,
            postwake_custody_seed=seed,
            postwake_custody_handshake_contract=custody,
            expected_run_directory=(capsule_fixture.anchor / "artifacts" / "runs" / "run-1"),
            expected_terminal_custody_authority_projection=(
                contract.terminal_custody_authority_projection
            ),
            terminal_client_launcher_release=launcher_release,
            expected_environment=envelope,
            process_environment_binding=environment_binding,
        )
        == contract
    )
    assert contract.contract_sha256 == authority.canonical_json_sha256(
        contract.payload_without_self_hash()
    )
    assert "supervisor_terminal_receipt_sha256" not in contract.as_dict()
    assert "composed_terminal_receipt_sha256" not in contract.as_dict()
    assert "wake_intent_path" not in contract.as_dict()
    assert "preterminal_overlap_handshake_receipt_path" not in contract.as_dict()
    assert custody.supervisor_job_id == "job-1"
    assert contract.terminal_custody_authority_projection[
        "outcome_blind_expected_artifact_instance"
    ]["expected_run_directory"] == str(capsule_fixture.anchor / "artifacts" / "runs" / "run-1")

    equal_but_wrong_container_type = contract.as_dict()
    equal_but_wrong_container_type["terminal_custody_authority_projection"] = _EqualDict(
        equal_but_wrong_container_type["terminal_custody_authority_projection"]
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_composition_contract(
            equal_but_wrong_container_type,
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
            preterminal_pin_contract_sha256="9" * 64,
            preterminal_overlap_handshake_contract=overlap,
            postwake_input_lease_contract=input_lease,
            postwake_custody_seed=seed,
            postwake_custody_handshake_contract=custody,
            expected_run_directory=expected_run_directory,
            expected_terminal_custody_authority_projection=terminal_custody_projection,
            terminal_client_launcher_release=launcher_release,
            expected_environment=envelope,
            process_environment_binding=environment_binding,
        )

    mismatched_e_projection = deepcopy(contract.terminal_custody_authority_projection)
    mismatched_e_projection["terminal_custody_authority_template_root_sha256"] = "0" * 64
    mismatched_e_projection = _rehash_without_self(
        mismatched_e_projection,
        "projection_root_sha256",
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_composition_contract(
            contract.as_dict(),
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
            preterminal_pin_contract_sha256="9" * 64,
            preterminal_overlap_handshake_contract=overlap,
            postwake_input_lease_contract=input_lease,
            postwake_custody_seed=seed,
            postwake_custody_handshake_contract=custody,
            expected_run_directory=(capsule_fixture.anchor / "artifacts" / "runs" / "run-1"),
            expected_terminal_custody_authority_projection=mismatched_e_projection,
            terminal_client_launcher_release=launcher_release,
            expected_environment=envelope,
            process_environment_binding=environment_binding,
        )

    mismatched_child_projection = deepcopy(contract.terminal_custody_authority_projection)
    mismatched_launcher = mismatched_child_projection["terminal_client_launcher_projection"]
    replacement_projection_sha256 = "f" * 64
    old_projection_sha256 = mismatched_launcher["verify_terminal_command_projection_sha256"]
    mismatched_launcher["verify_terminal_command_projection_sha256"] = replacement_projection_sha256
    mismatched_launcher["python_sys_argv_template"] = [
        replacement_projection_sha256 if item == old_projection_sha256 else item
        for item in mismatched_launcher["python_sys_argv_template"]
    ]
    mismatched_launcher["process_argv_template"] = [
        replacement_projection_sha256 if item == old_projection_sha256 else item
        for item in mismatched_launcher["process_argv_template"]
    ]
    mismatched_launcher["verify_terminal_launch_root_sha256"] = authority.canonical_json_sha256(
        {
            "verify_terminal_command_projection_sha256": (replacement_projection_sha256),
            "verify_terminal_environment_sha256": mismatched_launcher[
                "verify_terminal_environment_sha256"
            ],
            "verify_terminal_cwd": mismatched_launcher["verify_terminal_cwd"],
            "verify_terminal_cwd_root_sha256": mismatched_launcher[
                "verify_terminal_cwd_root_sha256"
            ],
        }
    )
    mismatched_launcher = _rehash_without_self(
        mismatched_launcher,
        "projection_root_sha256",
    )
    mismatched_child_projection["terminal_client_launcher_projection"] = mismatched_launcher
    mismatched_child_projection["terminal_client_launcher_projection_root_sha256"] = (
        mismatched_launcher["projection_root_sha256"]
    )
    mismatched_child_projection = _rehash_without_self(
        mismatched_child_projection,
        "projection_root_sha256",
    )
    mismatched_contract = contract.as_dict()
    mismatched_contract["terminal_custody_authority_projection"] = mismatched_child_projection
    mismatched_contract = _rehash_without_self(
        mismatched_contract,
        "contract_sha256",
    )
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_composition_contract(
            mismatched_contract,
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
            preterminal_pin_contract_sha256="9" * 64,
            preterminal_overlap_handshake_contract=overlap,
            postwake_input_lease_contract=input_lease,
            postwake_custody_seed=seed,
            postwake_custody_handshake_contract=custody,
            expected_run_directory=expected_run_directory,
            expected_terminal_custody_authority_projection=(mismatched_child_projection),
            terminal_client_launcher_release=launcher_release,
            expected_environment=envelope,
            process_environment_binding=environment_binding,
        )

    weaker_custody = custody.as_dict()
    weaker_custody["custody_client_process_job_membership_required"] = False
    weaker_custody = _rehash_without_self(weaker_custody, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_postwake_custody_handshake_contract(
            weaker_custody,
            custody_seed=seed,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "supervisor_terminal_success_reason",
            "exit_zero_but_weaker_than_frozen_success",
        ),
        ("exact_integrity_verifier_environment_sha256", "0" * 64),
        ("composed_output_claim_before_input_read_required", False),
        ("automatic_retry_allowed", True),
    ],
)
def test_terminal_composition_contract_rejects_rehashed_semantic_tampering(
    capsule_fixture: CapsuleFixture,
    field: str,
    replacement: Any,
) -> None:
    q = _q_replacement_v2(capsule_fixture)
    envelope = _environment()
    binding = authority.build_original_confirmatory_process_environment_binding(envelope)
    tail = _tail(capsule_fixture, authority.CAPSULE_TERMINAL_MODE)
    command = authority.build_original_confirmatory_capsule_command(
        capsule=capsule_fixture.capsule,
        mode=authority.CAPSULE_TERMINAL_MODE,
        tail_argv=tail,
        cwd=capsule_fixture.anchor,
    )
    overlap, input_lease, seed, custody = _terminal_dependencies(
        capsule_fixture,
        command,
        binding,
    )
    values = {
        flag: Path(tail[tail.index(flag) + 1])
        for flag in (
            "--supervisor-terminal",
            "--verifier-stdout",
            "--preterminal-pin",
            "--composed-terminal",
        )
    }
    expected_run_directory = capsule_fixture.anchor / "artifacts" / "runs" / "run-1"
    terminal_custody_projection = _terminal_custody_projection_for_command(
        capsule_fixture,
        q=q,
        command=command,
        tail=tail,
        environment_binding=binding,
        expected_run_directory=expected_run_directory,
    )
    launcher_release = q["supervisor_release"]["terminal_client_launcher_release"]
    contract = authority.build_original_confirmatory_terminal_composition_contract(
        capsule=capsule_fixture.capsule,
        verifier_command=command,
        verifier_command_tail_argv=tail,
        preterminal_pin_contract_sha256="9" * 64,
        preterminal_overlap_handshake_contract=overlap,
        postwake_input_lease_contract=input_lease,
        postwake_custody_seed=seed,
        postwake_custody_handshake_contract=custody,
        expected_run_directory=expected_run_directory,
        expected_terminal_custody_authority_projection=terminal_custody_projection,
        terminal_client_launcher_release=launcher_release,
        expected_environment=envelope,
        process_environment_binding=binding,
        supervisor_terminal_receipt_path=values["--supervisor-terminal"],
        supervisor_terminal_receipt_max_bytes=1024 * 1024,
        verifier_stdout_path=values["--verifier-stdout"],
        verifier_stdout_max_bytes=64 * 1024,
        preterminal_pin_receipt_path=values["--preterminal-pin"],
        preterminal_pin_receipt_max_bytes=1024 * 1024,
        postwake_input_lease_receipt_max_bytes=1024 * 1024,
        postwake_composed_readback_receipt_max_bytes=1024 * 1024,
        composed_terminal_receipt_path=values["--composed-terminal"],
        composed_terminal_receipt_max_bytes=1024 * 1024,
    )
    tampered = contract.as_dict()
    tampered[field] = replacement
    tampered = _rehash_without_self(tampered, "contract_sha256")
    with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
        authority.canonical_original_confirmatory_terminal_composition_contract(
            tampered,
            capsule=capsule_fixture.capsule,
            verifier_command=command,
            verifier_command_tail_argv=tail,
            preterminal_pin_contract_sha256="9" * 64,
            preterminal_overlap_handshake_contract=overlap,
            postwake_input_lease_contract=input_lease,
            postwake_custody_seed=seed,
            postwake_custody_handshake_contract=custody,
            expected_run_directory=(capsule_fixture.anchor / "artifacts" / "runs" / "run-1"),
            expected_terminal_custody_authority_projection=(
                contract.terminal_custody_authority_projection
            ),
            terminal_client_launcher_release=launcher_release,
            expected_environment=envelope,
            process_environment_binding=binding,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-file semantics")
def test_live_capsule_readback_matches_native_q_identity(
    capsule_fixture: CapsuleFixture,
) -> None:
    capsule_identity, python_identity, runtime_python_identity = (
        authority.require_live_original_confirmatory_execution_capsule(capsule_fixture.capsule)
    )
    assert capsule_identity.file_id_128 == capsule_fixture.leaf_lease.file_id_128
    assert capsule_identity.volume_serial_number == (
        capsule_fixture.leaf_lease.volume_serial_number
    )
    assert python_identity.sha256 == capsule_fixture.capsule.python_sha256
    assert runtime_python_identity.sha256 == capsule_fixture.capsule.runtime_python_sha256


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-file semantics")
def test_live_capsule_rejects_same_byte_path_replacement(
    capsule_fixture: CapsuleFixture,
) -> None:
    original = capsule_fixture.capsule_path.with_name("original-retired.pyz")
    capsule_fixture.capsule_path.chmod(stat.S_IWRITE)
    capsule_fixture.capsule_path.replace(original)
    capsule_fixture.capsule_path.write_bytes(capsule_fixture.capsule_bytes)
    capsule_fixture.capsule_path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.require_live_original_confirmatory_execution_capsule(capsule_fixture.capsule)
    finally:
        capsule_fixture.capsule_path.chmod(stat.S_IWRITE)
        original.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows retained-file semantics")
def test_live_capsule_rejects_named_ads(capsule_fixture: CapsuleFixture) -> None:
    ads_path = Path(f"{capsule_fixture.capsule_path}:forbidden")
    capsule_fixture.capsule_path.chmod(stat.S_IWRITE)
    ads_path.write_bytes(b"hidden")
    capsule_fixture.capsule_path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(authority.OriginalConfirmatoryCapsuleAuthorityError):
            authority.require_live_original_confirmatory_execution_capsule(capsule_fixture.capsule)
    finally:
        capsule_fixture.capsule_path.chmod(stat.S_IWRITE)
        ads_path.unlink(missing_ok=True)


def test_scientific_tail_rejects_float_downstream_spec_schema_before_started_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    e_path = tmp_path / "e_intent.json"
    q_path = tmp_path / "q.json"
    spec_path = tmp_path / "run_spec.json"
    runtime_path = tmp_path / "python.exe"
    e_file_sha256 = "a" * 64
    e_core_sha256 = "b" * 64
    q_root_sha256 = "c" * 64
    q_file_sha256 = "d" * 64
    live_argv = ["original_confirmatory.pyz", "run-confirmatory"]
    command = authority.OriginalConfirmatoryCapsuleCommand(
        program_path=runtime_path,
        program_sha256="e" * 64,
        argv=(str(runtime_path), "-I", "-B", *live_argv),
        cwd=tmp_path,
        command_sha256="f" * 64,
    )
    e = {
        "q_authority": {"path": str(q_path)},
        "intent_core_sha256": e_core_sha256,
        "job": {"supervisor_spec_path": str(spec_path), "job_id": "oc-test"},
        "project_root": str(tmp_path),
        "expected_launch_environment": {"policy": "synthetic-environment"},
        "process_environment_binding": {"policy": "synthetic-binding"},
    }
    q = {
        "q_authority_root_sha256": q_root_sha256,
        "execution_capsule": {"runtime_python_path": str(runtime_path)},
        "supervisor_release": {"supervisor_launcher_sha256": "1" * 64},
    }
    spec = {
        "schema_version": 2.0,
        "job_id": "oc-test",
        "project_root": str(tmp_path),
        "program_path": str(runtime_path),
        "program_sha256": command.program_sha256,
        "argv": list(command.argv),
        "expected_environment": e["expected_launch_environment"],
        "process_environment_binding": e["process_environment_binding"],
        "supervisor_launcher_sha256": q["supervisor_release"]["supervisor_launcher_sha256"],
    }
    read_roles: list[str] = []

    def read_control(_path: Path, *, role: str) -> tuple[dict[str, Any], str, bytes]:
        read_roles.append(role)
        if role == "live E intent":
            return {"q_authority": {"path": str(q_path)}}, e_file_sha256, b"{}\n"
        if role == "live Q replacement-v2":
            return {}, q_file_sha256, b"{}\n"
        if role == "downstream supervisor spec":
            return spec, "2" * 64, b"{}\n"
        raise AssertionError(f"unexpected post-spec read: {role}")

    monkeypatch.setattr(
        authority,
        "canonical_original_confirmatory_capsule_mode_tail",
        lambda *, capsule_mode, tail_argv: tuple(tail_argv),
    )
    monkeypatch.setattr(authority, "_read_canonical_control_object", read_control)
    monkeypatch.setattr(
        authority, "canonical_original_confirmatory_q_replacement_v2", lambda raw: q
    )
    monkeypatch.setattr(
        authority,
        "canonical_original_confirmatory_e_intent",
        lambda raw, **_kwargs: e,
    )
    monkeypatch.setattr(
        authority,
        "derive_original_confirmatory_capsule_command_from_e",
        lambda **_kwargs: command,
    )
    monkeypatch.setattr(
        authority,
        "require_live_original_confirmatory_execution_capsule",
        lambda _capsule: None,
    )
    monkeypatch.setattr(authority.sys, "argv", live_argv)
    monkeypatch.setattr(authority.sys, "orig_argv", [str(runtime_path), "-I", "-B"])
    monkeypatch.setattr(authority.sys, "_base_executable", str(runtime_path), raising=False)
    monkeypatch.setattr(authority.os, "getcwd", lambda: str(tmp_path))

    tail = (
        "--e-intent",
        str(e_path),
        "--e-intent-sha256",
        e_file_sha256,
        "--e-intent-core-sha256",
        e_core_sha256,
        "--q-authority-root-sha256",
        q_root_sha256,
    )
    with pytest.raises(
        authority.OriginalConfirmatoryCapsuleAuthorityError,
        match="downstream supervisor spec does not rederive exact Q/E launch",
    ):
        authority._prevalidate_original_confirmatory_scientific_tail(tail)

    assert read_roles == [
        "live E intent",
        "live Q replacement-v2",
        "downstream supervisor spec",
    ]
