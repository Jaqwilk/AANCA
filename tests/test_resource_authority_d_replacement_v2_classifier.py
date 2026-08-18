"""Exhaustive state-machine regressions for Authority-D replacement-v2."""

from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from histo_audit import config as config_module
from histo_audit.experiment import study_contracts
from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)

_PROGRESS_STATES_BY_BITS = {
    (False, False, False, False, False, False): (controller.State.QUALIFICATION_REQUIRED),
    (True, False, False, False, False, False): controller.State.INPUT_FREEZE_REQUIRED,
    (True, True, False, False, False, False): (controller.State.AUTHORIZATION_REQUIRED),
    (True, True, True, False, False, False): controller.State.READY,
    (True, True, True, True, False, True): controller.State.ROLLED_BACK_FAILURE,
}
_COMMITTED_BITS = (True, True, True, True, True, False)
_PRESENCE_FIELDS = (
    "qualification",
    "inputs",
    "authorization",
    "attempt",
    "success",
    "failure",
)
_VERIFY_LIVE_GOVERNED_BASELINE_V2 = controller._verify_live_governed_baseline_v2


def _expected_state(
    bits: tuple[bool, bool, bool, bool, bool, bool],
    *,
    exact_d: bool,
) -> controller.State:
    if exact_d:
        return (
            controller.State.COMMITTED
            if bits == _COMMITTED_BITS
            else controller.State.STOP_AMBIGUOUS
        )
    return _PROGRESS_STATES_BY_BITS.get(bits, controller.State.STOP_AMBIGUOUS)


def _presence(
    bits: tuple[bool, bool, bool, bool, bool, bool],
) -> dict[str, bool]:
    return dict(zip(_PRESENCE_FIELDS, bits, strict=True))


class _OwnedGuard:
    def __enter__(self) -> _OwnedGuard:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def assert_owned(self) -> None:
        return None


@pytest.fixture
def classifier_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    project = tmp_path / "project"
    control_root = project / "artifacts" / "resource_control"
    amendment_root = project / "artifacts" / "preregistration_amendments"
    control_root.mkdir(parents=True)
    amendment_root.mkdir()
    for component in controller._AMENDMENT_BASELINE:
        (amendment_root / component).mkdir()
    parent = amendment_root / controller._AUTHORITY_C_COMPONENT
    destination = amendment_root / "20260728T235959.000000Z"
    namespace = controller.Namespace.for_project(project)

    monkeypatch.setattr(controller, "_legacy_scoped_lock_paths", lambda *_a, **_k: ())
    monkeypatch.setattr(
        controller,
        "_require_legacy_lock_state_under_protocol_lock",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        controller,
        "_protocol_lock",
        lambda *_args, **_kwargs: _OwnedGuard(),
    )
    monkeypatch.setattr(
        controller,
        "ExclusiveBundlePublicationLock",
        lambda *_args, **_kwargs: _OwnedGuard(),
    )
    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        lambda *_args, **_kwargs: None,
    )

    return SimpleNamespace(
        project=project,
        control_root=control_root,
        amendment_root=amendment_root,
        parent=parent,
        destination=destination,
        namespace=namespace,
    )


def _install_presence(
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
) -> None:
    expected = _presence(bits)
    monkeypatch.setattr(
        controller,
        "_reserved_family_presence",
        lambda _namespace: dict(expected),
    )


def _install_valid_readers(
    monkeypatch: pytest.MonkeyPatch,
    tree: SimpleNamespace,
    calls: list[tuple[str, bool]],
) -> None:
    authorization = {
        "publication": {
            "parent_authority_directory": str(tree.parent),
            "intended_authority_directory": str(tree.destination),
        },
        "preflight": {"contract": {}},
    }
    attempt = {
        "intended_authority_directory": str(tree.destination),
        "parent_authority_directory": str(tree.parent),
        "run_state": {"sha256": "a" * 64},
    }

    def read_q(
        _namespace: controller.Namespace,
        *,
        pins: Any,
        verify_live_history: bool,
    ) -> tuple[dict[str, Any], str]:
        del pins
        calls.append(("q", verify_live_history))
        return {"status": "qualified"}, "1" * 64

    def read_i(
        _namespace: controller.Namespace,
        *,
        verify_live: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        calls.append(("i", verify_live))
        return {}, {}, "2" * 64

    def read_u(
        _namespace: controller.Namespace,
        *,
        verify_live: bool,
    ) -> tuple[dict[str, Any], str]:
        calls.append(("u", verify_live))
        return authorization, "3" * 64

    def read_a(
        _namespace: controller.Namespace,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        calls.append(("a", kwargs["verify_live"]))
        return attempt, "4" * 64

    monkeypatch.setattr(controller, "_read_terminal_qualification", read_q)
    monkeypatch.setattr(controller, "_read_input_v3", read_i)
    monkeypatch.setattr(controller, "_read_publication_authorization_v2", read_u)
    monkeypatch.setattr(controller, "_read_attempt_v2", read_a)
    monkeypatch.setattr(
        controller,
        "_read_success_v2",
        lambda *_args, **_kwargs: (
            {"authority_directory": str(tree.destination)},
            "5" * 64,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_read_failure_v2",
        lambda *_args, **_kwargs: (
            {"intended_authority_directory": str(tree.destination)},
            "6" * 64,
        ),
    )


def _install_terminal_marker_reader_spy(
    monkeypatch: pytest.MonkeyPatch,
    tree: SimpleNamespace,
    *,
    with_d: bool,
    marker_reads: list[str],
) -> None:
    if with_d:

        def read_success(
            *_args: object,
            **_kwargs: object,
        ) -> tuple[dict[str, Any], str]:
            marker_reads.append("success")
            return {
                "authority_directory": str(tree.destination),
            }, "5" * 64

        monkeypatch.setattr(controller, "_read_success_v2", read_success)
        return

    def read_failure(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, Any], str]:
        marker_reads.append("failure")
        return {
            "intended_authority_directory": str(tree.destination),
        }, "6" * 64

    monkeypatch.setattr(controller, "_read_failure_v2", read_failure)


def _classify(
    tree: SimpleNamespace,
    *,
    candidates: tuple[Path, ...] = (),
    committed_verifier: Any | None = None,
) -> controller.Classification:
    return controller.classify(
        tree.namespace,
        parent_authority_directory=tree.parent,
        candidate_discoverer=lambda _parent: candidates,
        committed_candidate_verifier=committed_verifier,
    )


def _materialize_presence(
    tree: SimpleNamespace,
    bits: tuple[bool, bool, bool, bool, bool, bool],
) -> None:
    paths = (
        tree.namespace.terminal_qualification,
        tree.namespace.input_v3,
        tree.namespace.authorization_v2,
        tree.namespace.attempt_v2,
        tree.namespace.success_v2,
        tree.namespace.failure_v2,
    )
    for present, path in zip(bits, paths, strict=True):
        if not present:
            continue
        if path == tree.namespace.input_v3:
            path.mkdir()
        else:
            path.write_bytes(b"{}\n")


def _install_governed_baseline_model(
    monkeypatch: pytest.MonkeyPatch,
    tree: SimpleNamespace,
    *,
    drift_role: str | None,
    committed: bool,
) -> SimpleNamespace:
    def record(path: Path, digest: str) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "size_bytes": 11,
            "sha256": digest,
        }

    run_root = tree.project / "artifacts" / "runs"
    run_hashes = {
        filename: f"{index:x}" * 64
        for index, filename in enumerate(controller._RUN_STATE_FILENAMES, start=1)
    }
    live_run_state = {
        "root": str(run_root.resolve()),
        "files": run_hashes,
        "sha256": "a" * 64,
    }
    terminal_run_state = {
        "root": live_run_state["root"],
        "files": {
            filename: record(run_root / filename, digest) for filename, digest in run_hashes.items()
        },
        "sha256": live_run_state["sha256"],
    }
    controller_record = record(Path(controller.__file__), "b" * 64)
    diagnosed_record = record(tree.project / "legacy_controller.py", "c" * 64)
    protected = {
        role: record(tree.project / relative_path, digest)
        for role, (relative_path, _size, digest) in controller._PROTECTED_BINDINGS.items()
    }
    historical_records = {
        "source_allowlist": record(tree.project / "historical_allowlist.json", "d" * 64)
    }
    authority_c = {"artifact_root_sha256": "e" * 64}
    input_records = {
        role: record(tree.namespace.input_v3 / filename, f"{index:x}" * 64)
        for index, (role, filename) in enumerate(
            controller.INPUT_V3_FILENAMES.items(),
            start=7,
        )
    }
    input_root = "f" * 64
    source_allowlist = {"schema_version": 1, "records": []}
    live_source = {
        "current_source": {
            "root_sha256": "1" * 64,
            "artifact_count": 18,
        },
        "current_manifest_sha256": "2" * 64,
        "delta_sha256": "3" * 64,
        "allowlist": source_allowlist,
    }
    source_contract = {
        "root_sha256": live_source["current_source"]["root_sha256"],
        "manifest_sha256": live_source["current_manifest_sha256"],
        "delta_sha256": live_source["delta_sha256"],
        "allowlisted_change_count": len(controller._EXPECTED_SOURCE_CHANGE_KINDS),
    }
    resource_config = tree.project / "configs" / "resource.yaml"
    manifest = tree.project / "data" / "pannuke_manifest.parquet"
    failed = tree.control_root / "failed.json"
    prior = tree.control_root / "prior.json"
    invalidation = tree.control_root / "invalidation.json"
    frozen = {
        "controller_path": controller_record["path"],
        "controller_size_bytes": controller_record["size_bytes"],
        "controller_sha256": controller_record["sha256"],
        "run_state_root": live_run_state["root"],
        "run_state_files": live_run_state["files"],
        "run_state_sha256": live_run_state["sha256"],
        "config_path": str(resource_config.resolve()),
        "config_file_sha256": "4" * 64,
        "config_semantic_sha256": "5" * 64,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": "6" * 64,
        "failed_preflight_receipt_path": str(failed.resolve()),
        "failed_preflight_receipt_sha256": "7" * 64,
        "prior_failure_receipt_path": str(prior.resolve()),
        "prior_failure_receipt_sha256": "8" * 64,
        "retired_input_invalidation_receipt_path": str(invalidation.resolve()),
        "retired_input_invalidation_receipt_sha256": "9" * 64,
        "execution_source_root_sha256": source_contract["root_sha256"],
        "execution_source_manifest_sha256": source_contract["manifest_sha256"],
        "execution_source_delta_sha256": source_contract["delta_sha256"],
        "execution_source_artifact_count": live_source["current_source"]["artifact_count"],
    }
    input_payloads = {
        "frozen_source_receipt": frozen,
        "source_allowlist": source_allowlist,
    }
    terminal_sha256 = "0" * 64
    terminal = {
        "lock_quiescence": {
            "reads_between_scans": [
                {"role": f"run_state_{filename}", **file_record}
                for filename, file_record in terminal_run_state["files"].items()
            ]
        },
        "controller_identities": {
            "qualifying_live_controller": controller_record,
            "diagnosed_fixed_legacy_controller": diagnosed_record,
        },
        "terminal_namespace": {
            "success_marker_absence": {
                "path": str((tree.control_root / "legacy_success.json").resolve())
            },
            "intended_authority_absence": {
                "path": str((tree.amendment_root / "legacy_destination").resolve())
            },
        },
        "frozen_v2_inputs": {"files": historical_records},
        "authority_c": authority_c,
        "protected_bindings": protected,
        "run_state": terminal_run_state,
    }
    frozen_bundle = {
        "files": input_records,
        "records_sha256": input_root,
    }
    contract = {
        "terminal_qualification": {
            "terminal_qualification_receipt": terminal,
            "terminal_qualification_receipt_sha256": terminal_sha256,
        },
        "frozen_input_bundle": frozen_bundle,
        "controller": controller_record,
        "run_state": live_run_state,
        "source": source_contract,
    }
    authorization = {"preflight": {"contract": contract}}
    authorization_sha256 = "a1" * 32
    attempt = {
        "controller": controller_record,
        "frozen_input_bundle": frozen_bundle,
        "run_state": live_run_state,
        "source": source_contract,
    }
    attempt_sha256 = "a2" * 32
    presence = _presence(_COMMITTED_BITS if committed else (True, True, True, True, False, True))
    candidates = (tree.destination,) if committed else ()

    file_records = {
        diagnosed_record["path"]: diagnosed_record,
        str(resource_config.resolve()): record(resource_config, frozen["config_file_sha256"]),
        str(manifest.resolve()): record(manifest, frozen["manifest_sha256"]),
        str(failed.resolve()): record(
            failed,
            frozen["failed_preflight_receipt_sha256"],
        ),
        str(prior.resolve()): record(
            prior,
            frozen["prior_failure_receipt_sha256"],
        ),
        str(invalidation.resolve()): record(
            invalidation,
            frozen["retired_input_invalidation_receipt_sha256"],
        ),
        **{item["path"]: item for item in terminal_run_state["files"].values()},
    }

    def file_record(path: Path, _role: str) -> dict[str, Any]:
        observed = dict(file_records[str(path.resolve())])
        drift_paths = {
            "run_state_file": next(iter(terminal_run_state["files"].values()))["path"],
            "resource_config": str(resource_config.resolve()),
            "pannuke_manifest": str(manifest.resolve()),
            "historical_reference": str(failed.resolve()),
        }
        if drift_role in drift_paths and observed["path"] == drift_paths[drift_role]:
            observed["sha256"] = "f0" * 32
        return observed

    def current_controller() -> dict[str, Any]:
        observed = dict(controller_record)
        if drift_role == "current_controller":
            observed["sha256"] = "f1" * 32
        return observed

    def live_run() -> dict[str, Any]:
        observed = {
            "root": live_run_state["root"],
            "files": dict(live_run_state["files"]),
            "sha256": live_run_state["sha256"],
        }
        if drift_role == "run_state_file":
            first = next(iter(observed["files"]))
            observed["files"][first] = "f2" * 32
        if drift_role == "run_state_root":
            observed["root"] = str((run_root / "foreign").resolve())
        return observed

    def live_protected(_project: Path) -> dict[str, dict[str, Any]]:
        observed = {role: dict(value) for role, value in protected.items()}
        protected_drifts = {
            "protected_spec": "specification",
            "protected_pre_registration": "pre_registration",
            "protected_primary_config": "primary_config",
            "protected_confirmatory_config": "confirmatory_config",
        }
        if drift_role in protected_drifts:
            observed[protected_drifts[drift_role]]["sha256"] = "f3" * 32
        return observed

    def current_source(_paths: dict[str, Path]) -> dict[str, Any]:
        observed = {
            **live_source,
            "current_source": dict(live_source["current_source"]),
        }
        if drift_role == "source_tree":
            observed["current_source"]["root_sha256"] = "f4" * 32
        return observed

    monkeypatch.setattr(
        controller,
        "_read_terminal_qualification",
        lambda *_args, **_kwargs: (terminal, terminal_sha256),
    )
    monkeypatch.setattr(
        controller,
        "_read_input_v3",
        lambda *_args, **_kwargs: (input_payloads, input_records, input_root),
    )
    monkeypatch.setattr(
        controller,
        "_read_publication_authorization_v2",
        lambda *_args, **_kwargs: (authorization, authorization_sha256),
    )
    monkeypatch.setattr(
        controller,
        "_read_attempt_v2",
        lambda *_args, **_kwargs: (attempt, attempt_sha256),
    )
    monkeypatch.setattr(controller, "_file_record", file_record)
    monkeypatch.setattr(controller, "_controller_identity", current_controller)
    monkeypatch.setattr(
        controller,
        "_input_v2_records",
        lambda *_args, **_kwargs: ({}, historical_records),
    )
    monkeypatch.setattr(controller, "_authority_c_receipt", lambda _parent: authority_c)
    monkeypatch.setattr(controller, "_protected_receipt", live_protected)
    monkeypatch.setattr(
        controller,
        "_run_state_receipt",
        lambda *_args, **_kwargs: terminal_run_state,
    )
    monkeypatch.setattr(controller, "_live_run_state_hashes", lambda _project: live_run())
    monkeypatch.setattr(controller, "_derive_live_source_v3", current_source)
    monkeypatch.setattr(controller, "_reserved_family_presence", lambda _namespace: presence)
    monkeypatch.setattr(
        controller,
        "_stable_amendment_inventory",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(config_module, "load_config", lambda _path: {"fixture": True})
    monkeypatch.setattr(
        study_contracts,
        "validate_resource_bounded_confirmatory_config",
        lambda value: value,
    )
    monkeypatch.setattr(
        config_module,
        "config_sha256",
        lambda _value: frozen["config_semantic_sha256"],
    )

    return SimpleNamespace(
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        attempt=attempt,
        attempt_sha256=attempt_sha256,
        presence=presence,
        candidates=candidates,
        candidate_discoverer=lambda _parent: candidates,
    )


def _verify_governed_baseline_model(
    tree: SimpleNamespace,
    model: SimpleNamespace,
    *,
    verify_live_run_state: bool,
) -> None:
    _VERIFY_LIVE_GOVERNED_BASELINE_V2(
        tree.namespace,
        parent_authority_directory=tree.parent,
        authorization=model.authorization,
        authorization_receipt_sha256=model.authorization_sha256,
        attempt=model.attempt,
        attempt_sha256=model.attempt_sha256,
        expected_presence=model.presence,
        expected_candidates=model.candidates,
        verify_live_run_state=verify_live_run_state,
        candidate_discoverer=model.candidate_discoverer,
    )


@pytest.mark.parametrize(
    "drift_role",
    (
        "protected_spec",
        "protected_pre_registration",
        "protected_primary_config",
        "protected_confirmatory_config",
        "current_controller",
        "source_tree",
        "resource_config",
        "pannuke_manifest",
        "historical_reference",
    ),
)
@pytest.mark.parametrize("committed", (False, True))
def test_live_governed_baseline_detects_non_run_state_referent_drift(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    drift_role: str,
    committed: bool,
) -> None:
    model = _install_governed_baseline_model(
        monkeypatch,
        classifier_tree,
        drift_role=drift_role,
        committed=committed,
    )

    with pytest.raises(controller.ControlError):
        _verify_governed_baseline_model(
            classifier_tree,
            model,
            verify_live_run_state=not committed,
        )


@pytest.mark.parametrize("drift_role", ("run_state_file", "run_state_root"))
def test_rolled_back_live_governed_baseline_rejects_run_state_drift(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    drift_role: str,
) -> None:
    model = _install_governed_baseline_model(
        monkeypatch,
        classifier_tree,
        drift_role=drift_role,
        committed=False,
    )

    with pytest.raises(controller.ControlError):
        _verify_governed_baseline_model(
            classifier_tree,
            model,
            verify_live_run_state=True,
        )


@pytest.mark.parametrize("drift_role", ("run_state_file", "run_state_root"))
def test_committed_live_governed_baseline_permits_later_run_state_advance(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    drift_role: str,
) -> None:
    model = _install_governed_baseline_model(
        monkeypatch,
        classifier_tree,
        drift_role=drift_role,
        committed=True,
    )

    _verify_governed_baseline_model(
        classifier_tree,
        model,
        verify_live_run_state=False,
    )


def test_terminal_marker_write_is_canonical_and_strictly_no_overwrite(
    tmp_path: Path,
) -> None:
    marker = tmp_path / controller.FAILURE_V2_FILENAME
    payload = {
        "schema_version": 2,
        "policy": "synthetic_terminal_marker",
        "status": "synthetic",
    }

    digest = controller._write_terminal_marker(
        marker,
        payload,
        role="synthetic replacement-v2 terminal marker",
    )
    encoded = controller._canonical_bytes(payload)

    assert marker.read_bytes() == encoded
    assert encoded.endswith(b"\n")
    assert digest == hashlib.sha256(encoded).hexdigest()

    with pytest.raises(FileExistsError):
        controller._write_terminal_marker(
            marker,
            {**payload, "status": "replacement-must-not-win"},
            role="synthetic replacement-v2 terminal marker",
        )
    assert marker.read_bytes() == encoded


def test_classifier_helper_exhaustively_matches_all_64_states_for_each_d_count() -> None:
    observed: dict[tuple[tuple[bool, ...], int], controller.State] = {}
    for raw_bits in product((False, True), repeat=6):
        bits = (
            raw_bits[0],
            raw_bits[1],
            raw_bits[2],
            raw_bits[3],
            raw_bits[4],
            raw_bits[5],
        )
        for candidate_count in (0, 1, 2):
            expected = (
                _expected_state(bits, exact_d=False)
                if candidate_count == 0
                else (
                    _expected_state(bits, exact_d=True)
                    if candidate_count == 1
                    else controller.State.STOP_AMBIGUOUS
                )
            )
            actual = controller._expected_state_from_presence(
                _presence(bits),
                candidate_count,
            )
            observed[(bits, candidate_count)] = actual
            assert actual is expected

    assert len(observed) == 192
    assert sum(state is not controller.State.STOP_AMBIGUOUS for state in observed.values()) == 6
    assert (
        observed[((False, False, False, False, False, False), 0)]
        is controller.State.QUALIFICATION_REQUIRED
    )
    assert (
        observed[((True, False, False, False, False, False), 0)]
        is controller.State.INPUT_FREEZE_REQUIRED
    )
    assert (
        observed[((True, True, False, False, False, False), 0)]
        is controller.State.AUTHORIZATION_REQUIRED
    )
    assert observed[((True, True, True, False, False, False), 0)] is controller.State.READY
    assert (
        observed[((True, True, True, True, False, True), 0)] is controller.State.ROLLED_BACK_FAILURE
    )
    assert observed[((True, True, True, True, True, False), 1)] is controller.State.COMMITTED


@pytest.mark.parametrize(
    ("presence", "candidate_count"),
    (
        ({}, 0),
        (
            {
                "inputs": False,
                "qualification": False,
                "authorization": False,
                "attempt": False,
                "success": False,
                "failure": False,
            },
            0,
        ),
        ({field: False for field in _PRESENCE_FIELDS} | {"foreign": False}, 0),
        ({field: 0 for field in _PRESENCE_FIELDS}, 0),
        ({field: False for field in _PRESENCE_FIELDS}, True),
        ({field: False for field in _PRESENCE_FIELDS}, -1),
    ),
)
def test_classifier_helper_rejects_non_exact_presence_and_d_count_types(
    presence: dict[str, Any],
    candidate_count: Any,
) -> None:
    assert (
        controller._expected_state_from_presence(presence, candidate_count)
        is controller.State.STOP_AMBIGUOUS
    )


def test_public_classifier_exhaustively_matches_all_64_states_with_and_without_exact_d(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    for raw_bits in product((False, True), repeat=6):
        bits = (
            raw_bits[0],
            raw_bits[1],
            raw_bits[2],
            raw_bits[3],
            raw_bits[4],
            raw_bits[5],
        )
        _install_presence(monkeypatch, bits)
        result = _classify(classifier_tree)
        assert result.state is _expected_state(bits, exact_d=False), bits

    classifier_tree.destination.mkdir()
    for raw_bits in product((False, True), repeat=6):
        bits = (
            raw_bits[0],
            raw_bits[1],
            raw_bits[2],
            raw_bits[3],
            raw_bits[4],
            raw_bits[5],
        )
        _install_presence(monkeypatch, bits)
        result = _classify(
            classifier_tree,
            candidates=(classifier_tree.destination,),
            committed_verifier=lambda _candidate, _success: None,
        )
        assert result.state is _expected_state(bits, exact_d=True), bits


@pytest.mark.parametrize(
    ("bits", "expected_state", "expected_calls"),
    (
        (
            (True, False, False, False, False, False),
            controller.State.INPUT_FREEZE_REQUIRED,
            [("q", True)],
        ),
        (
            (True, True, False, False, False, False),
            controller.State.AUTHORIZATION_REQUIRED,
            [("q", True), ("i", True)],
        ),
        (
            (True, True, True, False, False, False),
            controller.State.READY,
            [("q", True), ("i", True), ("u", True)],
        ),
    ),
)
def test_pre_d_progress_states_use_full_live_q_i_and_u_readers(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    expected_state: controller.State,
    expected_calls: list[tuple[str, bool]],
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    result = _classify(classifier_tree)

    assert result.state is expected_state
    assert calls == expected_calls


@pytest.mark.parametrize(
    ("role", "bits"),
    (
        ("q", (True, False, False, False, False, False)),
        ("i", (True, True, False, False, False, False)),
        ("u", (True, True, True, False, False, False)),
    ),
)
def test_pre_d_tampered_q_i_or_u_fails_closed(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    bits: tuple[bool, bool, bool, bool, bool, bool],
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise controller.ControlError(f"synthetic {role} tamper")

    reader_names = {
        "q": "_read_terminal_qualification",
        "i": "_read_input_v3",
        "u": "_read_publication_authorization_v2",
    }
    monkeypatch.setattr(controller, reader_names[role], reject)

    result = _classify(classifier_tree)

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert f"synthetic {role} tamper" in result.reason


@pytest.mark.parametrize(
    ("bits", "expected_state", "with_d"),
    (
        (
            (True, True, True, True, False, True),
            controller.State.ROLLED_BACK_FAILURE,
            False,
        ),
        (
            (True, True, True, True, True, False),
            controller.State.COMMITTED,
            True,
        ),
    ),
)
def test_terminal_states_use_sealed_q_i_u_readbacks(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    expected_state: controller.State,
    with_d: bool,
) -> None:
    calls: list[tuple[str, bool]] = []
    baseline_calls: list[dict[str, Any]] = []
    marker_reads: list[str] = []
    verifier_calls: list[tuple[Path, dict[str, Any]]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)
    _install_terminal_marker_reader_spy(
        monkeypatch,
        classifier_tree,
        with_d=with_d,
        marker_reads=marker_reads,
    )
    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        lambda _namespace, **kwargs: baseline_calls.append(kwargs),
    )
    candidates: tuple[Path, ...] = ()

    def verifier(candidate: Path, success: dict[str, Any]) -> None:
        verifier_calls.append((candidate, success))

    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    first = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=verifier if with_d else None,
    )
    assert first.state is expected_state
    assert calls == [("q", False), ("i", False), ("u", False), ("a", False)]
    assert len(baseline_calls) == 2
    assert all(call["expected_presence"] == _presence(bits) for call in baseline_calls)
    assert all(tuple(call["expected_candidates"]) == candidates for call in baseline_calls)
    expected_marker = "success" if with_d else "failure"
    assert marker_reads == [expected_marker, expected_marker]
    assert len(verifier_calls) == (2 if with_d else 0)


@pytest.mark.parametrize("role", ("q", "i", "u"))
@pytest.mark.parametrize(
    ("bits", "with_d"),
    (
        ((True, True, True, True, False, True), False),
        ((True, True, True, True, True, False), True),
    ),
)
def test_terminal_q_i_or_u_byte_tamper_fails_closed(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise controller.ControlError(f"synthetic sealed-{role} byte tamper")

    monkeypatch.setattr(
        controller,
        {
            "q": "_read_terminal_qualification",
            "i": "_read_input_v3",
            "u": "_read_publication_authorization_v2",
        }[role],
        reject,
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert f"synthetic sealed-{role} byte tamper" in result.reason


@pytest.mark.parametrize(
    "drift_role",
    (
        "run_state_file",
        "run_state_root",
        "protected_spec",
        "protected_pre_registration",
        "protected_primary_config",
        "protected_confirmatory_config",
        "current_controller",
        "source_tree",
        "resource_config",
        "pannuke_manifest",
        "historical_reference",
    ),
)
@pytest.mark.parametrize(
    ("bits", "with_d"),
    (
        ((True, True, True, True, False, True), False),
        ((True, True, True, True, True, False), True),
    ),
)
def test_terminal_live_referent_drift_fails_closed_with_sealed_q_i_u_bytes(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    drift_role: str,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    def reject_baseline(*_args: object, **_kwargs: object) -> None:
        raise controller.ControlError(f"synthetic live drift: {drift_role}")

    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        reject_baseline,
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert f"synthetic live drift: {drift_role}" in result.reason
    assert calls == [("q", False), ("i", False), ("u", False), ("a", False)]


@pytest.mark.parametrize(
    ("bits", "with_d", "expected_verify_live_run_state"),
    (
        ((True, True, True, True, False, True), False, True),
        ((True, True, True, True, True, False), True, False),
    ),
)
def test_terminal_baseline_uses_phase_specific_run_state_policy(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
    expected_verify_live_run_state: bool,
) -> None:
    calls: list[tuple[str, bool]] = []
    baseline_modes: list[bool] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    def baseline(_namespace: controller.Namespace, **kwargs: Any) -> None:
        baseline_modes.append(kwargs["verify_live_run_state"])

    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        baseline,
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is (
        controller.State.COMMITTED if with_d else controller.State.ROLLED_BACK_FAILURE
    )
    assert baseline_modes == [
        expected_verify_live_run_state,
        expected_verify_live_run_state,
    ]


@pytest.mark.parametrize(
    ("bits", "with_d", "expected_mode"),
    (
        ((True, True, True, True, False, True), False, True),
        ((True, True, True, True, True, False), True, False),
    ),
)
def test_terminal_second_baseline_drift_fails_closed_after_marker_readback(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
    expected_mode: bool,
) -> None:
    calls: list[tuple[str, bool]] = []
    marker_reads: list[str] = []
    baseline_modes: list[bool] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)
    _install_terminal_marker_reader_spy(
        monkeypatch,
        classifier_tree,
        with_d=with_d,
        marker_reads=marker_reads,
    )

    def baseline(_namespace: controller.Namespace, **kwargs: Any) -> None:
        baseline_modes.append(kwargs["verify_live_run_state"])
        if len(baseline_modes) == 2:
            raise controller.ControlError("synthetic final-baseline drift")

    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        baseline,
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert "synthetic final-baseline drift" in result.reason
    assert baseline_modes == [expected_mode, expected_mode]
    expected_marker = "success" if with_d else "failure"
    assert marker_reads == [expected_marker, expected_marker]


@pytest.mark.parametrize(
    ("bits", "with_d", "expected_state"),
    (
        (
            (True, True, True, True, False, True),
            False,
            controller.State.STOP_AMBIGUOUS,
        ),
        (
            (True, True, True, True, True, False),
            True,
            controller.State.COMMITTED,
        ),
    ),
)
def test_only_committed_classifier_permits_governed_post_d_run_state_advance(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
    expected_state: controller.State,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, bits)
    _install_valid_readers(monkeypatch, classifier_tree, calls)

    def baseline(_namespace: controller.Namespace, **kwargs: Any) -> None:
        if kwargs["verify_live_run_state"]:
            raise controller.ControlError("synthetic governed run-state advanced")

    monkeypatch.setattr(
        controller,
        "_verify_live_governed_baseline_v2",
        baseline,
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is expected_state
    if expected_state is controller.State.STOP_AMBIGUOUS:
        assert "synthetic governed run-state advanced" in result.reason


@pytest.mark.parametrize(
    "d_drift",
    (
        "artifact_root",
        "sha256_manifest",
        "typed_effective_authorization",
    ),
)
def test_committed_d_root_manifest_or_typed_drift_fails_closed(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    d_drift: str,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, _COMMITTED_BITS)
    _install_valid_readers(monkeypatch, classifier_tree, calls)
    classifier_tree.destination.mkdir()

    def reject_d(_candidate: Path, _success: dict[str, Any]) -> None:
        raise controller.ControlError(f"synthetic committed D drift: {d_drift}")

    result = _classify(
        classifier_tree,
        candidates=(classifier_tree.destination,),
        committed_verifier=reject_d,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert f"synthetic committed D drift: {d_drift}" in result.reason


def test_foreign_and_multiple_d_candidates_fail_closed(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    _install_presence(monkeypatch, _COMMITTED_BITS)
    _install_valid_readers(monkeypatch, classifier_tree, calls)
    foreign = classifier_tree.amendment_root / "20260728T235958.000000Z"
    foreign.mkdir()

    foreign_result = _classify(
        classifier_tree,
        candidates=(foreign,),
        committed_verifier=lambda _candidate, _success: None,
    )

    assert foreign_result.state is controller.State.STOP_AMBIGUOUS
    assert "singleton D2" in foreign_result.reason

    second = classifier_tree.amendment_root / "20260728T235959.000001Z"
    second.mkdir()
    multiple_result = _classify(
        classifier_tree,
        candidates=tuple(sorted((foreign, second), key=lambda path: str(path).casefold())),
        committed_verifier=lambda _candidate, _success: None,
    )

    assert multiple_result.state is controller.State.STOP_AMBIGUOUS


@pytest.mark.parametrize(
    ("bits", "candidate_count"),
    (
        ((False, False, False, True, False, False), 0),  # A only
        ((True, True, True, True, True, True), 0),  # S and F together
        ((True, True, True, True, False, False), 1),  # D without S
    ),
)
def test_partial_marker_shapes_are_stop_ambiguous(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    candidate_count: int,
) -> None:
    _install_presence(monkeypatch, bits)
    candidate = classifier_tree.amendment_root / "unexpected_d"
    candidates = (candidate,) if candidate_count else ()

    result = _classify(classifier_tree, candidates=candidates)

    assert result.state is controller.State.STOP_AMBIGUOUS


def test_case_alias_in_reserved_namespace_is_stop_ambiguous(
    classifier_tree: SimpleNamespace,
) -> None:
    alias = (
        classifier_tree.control_root
        / "RESOURCE_AUTHORITY_D_REPLACEMENT_V2_PUBLICATION_attempt.json"
    )
    alias.write_bytes(b"{}\n")

    result = _classify(classifier_tree)

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert "reserved-family" in result.reason or "case aliases" in result.reason


@pytest.mark.parametrize(
    ("partial_role", "bits", "with_d"),
    (
        ("attempt", (True, True, True, True, False, True), False),
        ("failure", (True, True, True, True, False, True), False),
        ("success", (True, True, True, True, True, False), True),
    ),
)
def test_noncanonical_partial_terminal_marker_is_stop_ambiguous(
    classifier_tree: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    partial_role: str,
    bits: tuple[bool, bool, bool, bool, bool, bool],
    with_d: bool,
) -> None:
    real_readers = {
        "attempt": controller._read_attempt_v2,
        "success": controller._read_success_v2,
        "failure": controller._read_failure_v2,
    }
    _materialize_presence(classifier_tree, bits)
    partial_path = {
        "attempt": classifier_tree.namespace.attempt_v2,
        "success": classifier_tree.namespace.success_v2,
        "failure": classifier_tree.namespace.failure_v2,
    }[partial_role]
    partial_path.write_bytes(b"{")
    calls: list[tuple[str, bool]] = []
    _install_valid_readers(monkeypatch, classifier_tree, calls)
    monkeypatch.setattr(
        controller,
        f"_read_{partial_role}_v2",
        real_readers[partial_role],
    )
    candidates: tuple[Path, ...] = ()
    if with_d:
        classifier_tree.destination.mkdir()
        candidates = (classifier_tree.destination,)

    result = _classify(
        classifier_tree,
        candidates=candidates,
        committed_verifier=lambda _candidate, _success: None,
    )

    assert result.state is controller.State.STOP_AMBIGUOUS
    assert "canonical" in result.reason or "JSON" in result.reason
