"""Hermetic CLI contracts for Authority-D replacement-v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from histo_audit.workflows import (
    resource_authority_d_replacement_v2_controller as controller,
)

_PUBLIC_ENTRYPOINTS = (
    "classify",
    "qualify_historical_terminal_once",
    "freeze_input_v3_once",
    "authorize_publication_v2_once",
    "publish_replacement_authority_once",
)
_ACTION_SAFETY_FIELDS = {
    "automatic_retry_allowed": False,
    "outcome_value_interpretation_performed": False,
    "scientific_execution_performed": False,
}


@dataclass(frozen=True, slots=True)
class _CliLayout:
    project: Path
    namespace: controller.Namespace
    parent: Path
    destination: Path


def _layout(tmp_path: Path) -> _CliLayout:
    project = tmp_path.resolve()
    amendment_root = project / "artifacts" / "preregistration_amendments"
    for component in controller._AMENDMENT_BASELINE:
        (amendment_root / component).mkdir(parents=True)
    namespace = controller.Namespace.for_project(project)
    namespace.control_root.mkdir(parents=True)
    return _CliLayout(
        project=project,
        namespace=namespace,
        parent=amendment_root / controller._AUTHORITY_C_COMPONENT,
        destination=amendment_root / "20990101T000000.000000Z",
    )


def _argv(layout: _CliLayout, mode: str) -> list[str]:
    return [
        mode,
        "--project-root",
        str(layout.project),
        "--parent-authority-dir",
        str(layout.parent),
    ]


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    records: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            records.append((relative, "directory", 0, ""))
        else:
            payload = path.read_bytes()
            records.append(
                (
                    relative,
                    "file",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
    return tuple(records)


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    assert type(value) is dict
    return value


def _assert_action_safety(payload: dict[str, Any]) -> None:
    for field, expected in _ACTION_SAFETY_FIELDS.items():
        assert payload[field] is expected


def _install_dispatch_spies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected: str,
    selected_result: object,
    layout: _CliLayout,
) -> dict[str, int]:
    calls = dict.fromkeys(_PUBLIC_ENTRYPOINTS, 0)

    def build(name: str) -> Any:
        def dispatch(*args: Any, **kwargs: Any) -> object:
            calls[name] += 1
            assert name == selected, f"unexpected CLI dispatch to {name}"
            if name == "classify":
                assert args == (layout.namespace,)
            else:
                assert args == ()
                assert kwargs["namespace"] == layout.namespace
            assert kwargs["parent_authority_directory"] == layout.parent
            return selected_result

        return dispatch

    for name in _PUBLIC_ENTRYPOINTS:
        monkeypatch.setattr(controller, name, build(name))
    return calls


def test_parser_requires_one_mode_and_parent_and_help_lists_all_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _layout(tmp_path)
    parser = controller._parser()

    with pytest.raises(SystemExit) as no_arguments:
        parser.parse_args([])
    assert no_arguments.value.code == 2
    with pytest.raises(SystemExit) as no_parent:
        parser.parse_args(["--classify"])
    assert no_parent.value.code == 2
    with pytest.raises(SystemExit) as no_mode:
        parser.parse_args(["--parent-authority-dir", str(layout.parent)])
    assert no_mode.value.code == 2
    with pytest.raises(SystemExit) as conflicting:
        parser.parse_args(
            [
                "--classify",
                "--publish-once",
                "--parent-authority-dir",
                str(layout.parent),
            ]
        )
    assert conflicting.value.code == 2
    capsys.readouterr()

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])

    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--classify",
        "--qualify-terminal",
        "--freeze-inputs",
        "--authorize-publication",
        "--publish-once",
        "--project-root",
        "--parent-authority-dir",
    ):
        assert option in help_text


@pytest.mark.parametrize(
    ("state", "exit_code"),
    (
        (controller.State.READY, 0),
        (controller.State.STOP_AMBIGUOUS, 1),
    ),
)
def test_classify_cli_is_no_write_and_has_exact_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: controller.State,
    exit_code: int,
) -> None:
    layout = _layout(tmp_path)
    classification = controller.Classification(
        state=state,
        reason=f"synthetic {state.value}",
    )
    calls = _install_dispatch_spies(
        monkeypatch,
        selected="classify",
        selected_result=classification,
        layout=layout,
    )
    before = _tree_snapshot(layout.project)

    observed_exit = controller.main(_argv(layout, "--classify"))

    assert observed_exit == exit_code
    assert _output(capsys) == classification.as_dict()
    assert _tree_snapshot(layout.project) == before
    assert calls["classify"] == 1
    assert sum(calls.values()) == 1
    assert classification.as_dict()["automatic_retry_allowed"] is False


@pytest.mark.parametrize(
    (
        "mode",
        "entrypoint",
        "result",
        "expected",
    ),
    (
        (
            "--qualify-terminal",
            "qualify_historical_terminal_once",
            (
                {"status": "historical_terminal_qualified"},
                "a" * 64,
            ),
            {
                "schema_version": controller.PROTOCOL_SCHEMA_VERSION,
                "status": "historical_terminal_qualified",
                "terminal_qualification_receipt_sha256": "a" * 64,
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            },
        ),
        (
            "--freeze-inputs",
            "freeze_input_v3_once",
            (
                {"synthetic": {}},
                {"first": {}, "second": {}},
                "b" * 64,
            ),
            {
                "schema_version": controller.PROTOCOL_SCHEMA_VERSION,
                "status": "input_v3_frozen",
                "input_v3_file_count": 2,
                "input_v3_records_sha256": "b" * 64,
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            },
        ),
        (
            "--authorize-publication",
            "authorize_publication_v2_once",
            (
                {
                    "status": "authorized_for_one_attempt",
                    "authorized_attempt_id": "c" * 64,
                },
                "d" * 64,
            ),
            {
                "schema_version": controller.PROTOCOL_SCHEMA_VERSION,
                "status": "authorized_for_one_attempt",
                "publication_authorization_v2_sha256": "d" * 64,
                "authorized_attempt_id": "c" * 64,
                "automatic_retry_allowed": False,
                "outcome_value_interpretation_performed": False,
                "scientific_execution_performed": False,
                "publication_performed": False,
            },
        ),
    ),
)
def test_nonpublication_cli_mode_dispatches_once_with_closed_safe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    entrypoint: str,
    result: object,
    expected: dict[str, Any],
) -> None:
    layout = _layout(tmp_path)
    calls = _install_dispatch_spies(
        monkeypatch,
        selected=entrypoint,
        selected_result=result,
        layout=layout,
    )
    before = _tree_snapshot(layout.project)

    exit_code = controller.main(_argv(layout, mode))

    payload = _output(capsys)
    assert exit_code == 0
    if mode == "--qualify-terminal":
        expected["terminal_qualification_receipt_path"] = str(
            layout.namespace.terminal_qualification
        )
    elif mode == "--freeze-inputs":
        expected["input_v3_directory"] = str(layout.namespace.input_v3)
    else:
        expected["publication_authorization_v2_path"] = str(layout.namespace.authorization_v2)
    assert payload == expected
    _assert_action_safety(payload)
    assert payload["publication_performed"] is False
    assert _tree_snapshot(layout.project) == before
    assert calls[entrypoint] == 1
    assert sum(calls.values()) == 1


@pytest.mark.parametrize(
    (
        "state",
        "exit_code",
        "marker_name",
        "authority_present",
    ),
    (
        (
            controller.State.ROLLED_BACK_FAILURE,
            3,
            "failure_v2",
            False,
        ),
        (
            controller.State.COMMITTED,
            0,
            "success_v2",
            True,
        ),
    ),
)
def test_publish_cli_f2_and_committed_exit_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: controller.State,
    exit_code: int,
    marker_name: str,
    authority_present: bool,
) -> None:
    layout = _layout(tmp_path)
    marker_path = getattr(layout.namespace, marker_name)
    publication = controller.PublicationResultV2(
        state=state,
        marker_path=marker_path,
        marker_sha256="e" * 64,
        authority_directory=(layout.destination if authority_present else None),
    )
    calls = _install_dispatch_spies(
        monkeypatch,
        selected="publish_replacement_authority_once",
        selected_result=publication,
        layout=layout,
    )
    before = _tree_snapshot(layout.project)

    observed_exit = controller.main(_argv(layout, "--publish-once"))

    payload = _output(capsys)
    assert observed_exit == exit_code
    assert payload == {
        "schema_version": controller.PROTOCOL_SCHEMA_VERSION,
        "state": state.value,
        "terminal_marker_path": str(marker_path),
        "terminal_marker_sha256": "e" * 64,
        "authority_directory": (str(layout.destination) if authority_present else None),
        "automatic_retry_allowed": False,
        "outcome_value_interpretation_performed": False,
        "scientific_execution_performed": False,
        "publication_performed": authority_present,
    }
    _assert_action_safety(payload)
    assert _tree_snapshot(layout.project) == before
    assert calls["publish_replacement_authority_once"] == 1
    assert sum(calls.values()) == 1


@pytest.mark.parametrize(
    (
        "write_before_error",
        "observed_state",
        "expected_status",
        "expected_exit",
    ),
    (
        (
            False,
            controller.State.QUALIFICATION_REQUIRED,
            "stopped_without_write",
            1,
        ),
        (
            True,
            controller.State.INPUT_FREEZE_REQUIRED,
            "stopped_after_control_write",
            3,
        ),
    ),
)
def test_cli_exception_distinguishes_before_write_from_control_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_before_error: bool,
    observed_state: controller.State,
    expected_status: str,
    expected_exit: int,
) -> None:
    layout = _layout(tmp_path)
    calls = dict.fromkeys(_PUBLIC_ENTRYPOINTS, 0)

    def fail_qualification(**kwargs: Any) -> object:
        calls["qualify_historical_terminal_once"] += 1
        assert kwargs["namespace"] == layout.namespace
        assert kwargs["parent_authority_directory"] == layout.parent
        if write_before_error:
            layout.namespace.terminal_qualification.write_bytes(
                b'{"synthetic":"retained-control-write"}\n'
            )
        raise RuntimeError("synthetic CLI dispatch failure")

    def classify_after_error(
        namespace: controller.Namespace,
        *,
        parent_authority_directory: Path,
    ) -> controller.Classification:
        calls["classify"] += 1
        assert namespace == layout.namespace
        assert parent_authority_directory == layout.parent
        return controller.Classification(
            state=observed_state,
            reason="synthetic exception disposition",
        )

    monkeypatch.setattr(
        controller,
        "qualify_historical_terminal_once",
        fail_qualification,
    )
    monkeypatch.setattr(controller, "classify", classify_after_error)
    for name in (
        "freeze_input_v3_once",
        "authorize_publication_v2_once",
        "publish_replacement_authority_once",
    ):
        monkeypatch.setattr(
            controller,
            name,
            lambda **_kwargs: pytest.fail("unexpected CLI dispatch"),
        )

    before = _tree_snapshot(layout.project)
    exit_code = controller.main(_argv(layout, "--qualify-terminal"))

    payload = _output(capsys)
    assert exit_code == expected_exit
    assert set(payload) == {
        "schema_version",
        "status",
        "replacement_state",
        "automatic_retry_allowed",
        "publication_performed",
        "outcome_value_interpretation_performed",
        "scientific_execution_performed",
        "error_sha256",
    }
    assert payload["status"] == expected_status
    assert payload["replacement_state"] == observed_state.value
    assert payload["publication_performed"] is False
    assert (
        payload["error_sha256"]
        == hashlib.sha256(b"RuntimeError: synthetic CLI dispatch failure").hexdigest()
    )
    _assert_action_safety(payload)
    assert calls["qualify_historical_terminal_once"] == 1
    assert calls["classify"] == 1
    assert sum(calls.values()) == 2
    if write_before_error:
        assert layout.namespace.terminal_qualification.is_file()
        assert _tree_snapshot(layout.project) != before
    else:
        assert _tree_snapshot(layout.project) == before
