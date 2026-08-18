from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_ROOT = PROJECT_ROOT / "src" / "histo_audit" / "workflows"
TEST_PACKAGE = "_aanca_original_confirmatory_capsule_entry_test_package"


def _load_test_module(module_stem: str) -> Any:
    package = sys.modules.get(TEST_PACKAGE)
    if package is None:
        package = types.ModuleType(TEST_PACKAGE)
        package.__path__ = [str(WORKFLOWS_ROOT)]
        package.__package__ = TEST_PACKAGE
        sys.modules[TEST_PACKAGE] = package
    module_name = f"{TEST_PACKAGE}.{module_stem}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = WORKFLOWS_ROOT / f"{module_stem}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


authority = _load_test_module("original_confirmatory_capsule_authority")
entry = _load_test_module("original_confirmatory_capsule_entry")


def _tail(
    tmp_path: Path,
    mode: str,
    *,
    execution_mode: str = "fresh",
) -> tuple[str, ...]:
    common: dict[str, Any] = {
        "capsule_mode": mode,
        "e_intent_path": tmp_path / "e-intent.json",
        "e_intent_sha256": "1" * 64,
        "e_intent_core_sha256": "2" * 64,
        "q_authority_root_sha256": "3" * 64,
        "launch_nonce": "4" * 64,
        "supervisor_job_id": "job-1",
        "supervisor_job_directory": tmp_path / "supervisor" / "jobs" / "job-1",
        "attempt_id": "attempt-1",
        "run_id": "run-1",
        "execution_mode": execution_mode,
        "retry_of_run_id": "source-run-1" if execution_mode == "successor_resume" else None,
    }
    if mode == authority.CAPSULE_PRETERMINAL_MODE:
        common.update(
            {
                "run_spec_path": tmp_path / "run-spec.json",
                "launch_intent_path": tmp_path / "launch-intent.json",
                "process_started_path": tmp_path / "process-started.json",
                "preterminal_pin_path": tmp_path / "preterminal-pin.json",
            }
        )
    elif mode == authority.CAPSULE_TERMINAL_MODE:
        common.update(
            {
                "supervisor_terminal_path": tmp_path / "supervisor-terminal.json",
                "verifier_stdout_path": tmp_path / "verifier.stdout.log",
                "preterminal_pin_path": tmp_path / "preterminal-pin.json",
                "composed_terminal_path": tmp_path / "composed-terminal.json",
            }
        )
    return authority.original_confirmatory_capsule_mode_tail(**common)


def _dispatch_map(
    calls: list[tuple[str, tuple[str, ...]]],
    *,
    results: dict[str, Any] | None = None,
) -> dict[str, Callable[[tuple[str, ...]], int]]:
    configured = results or {}

    def handler_for(mode: str) -> Callable[[tuple[str, ...]], int]:
        def handler(tail: tuple[str, ...]) -> int:
            calls.append((mode, tail))
            value = configured.get(mode, 0)
            if isinstance(value, BaseException):
                raise value
            return value

        return handler

    return {mode: handler_for(mode) for mode in authority.CAPSULE_ALLOWED_MODES}


@pytest.mark.parametrize(
    ("mode", "execution_mode", "expected_code"),
    [
        (authority.CAPSULE_SCIENTIFIC_MODE, "fresh", 0),
        (authority.CAPSULE_PRETERMINAL_MODE, "fresh", 3),
        (authority.CAPSULE_TERMINAL_MODE, "successor_resume", 7),
    ],
)
def test_exact_mode_tail_dispatches_once(
    tmp_path: Path,
    mode: str,
    execution_mode: str,
    expected_code: int,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    dispatch_map = _dispatch_map(calls, results={mode: expected_code})
    tail = _tail(tmp_path, mode, execution_mode=execution_mode)

    result = entry._dispatch_original_confirmatory_capsule_with_map_for_test(
        (mode, *tail),
        dispatch_map=dispatch_map,
    )

    assert result == expected_code
    assert calls == [(mode, tail)]


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "wrong_mode", "list"])
def test_nonexact_argv_fails_before_dispatch_map_access(
    tmp_path: Path,
    mutation: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mode = authority.CAPSULE_TERMINAL_MODE
    valid: tuple[str, ...] | list[str] = (mode, *_tail(tmp_path, mode))
    if mutation == "missing":
        candidate: tuple[str, ...] | list[str] = valid[:-1]
    elif mutation == "extra":
        candidate = (*valid, "--unexpected")
    elif mutation == "reordered":
        changed = list(valid)
        changed[1], changed[3] = changed[3], changed[1]
        candidate = tuple(changed)
    elif mutation == "wrong_mode":
        candidate = ("other-mode", *valid[1:])
    else:
        candidate = list(valid)

    class ExplodingMapping(dict[str, Callable[[tuple[str, ...]], int]]):
        def __iter__(self) -> Any:
            raise AssertionError("invalid argv touched the dispatch map")

        def __getitem__(self, key: str) -> Callable[[tuple[str, ...]], int]:
            raise AssertionError(f"invalid argv touched handler {key}")

    result = entry._dispatch_original_confirmatory_capsule_with_map_for_test(
        candidate,  # type: ignore[arg-type]
        dispatch_map=ExplodingMapping(),
    )

    assert result == entry.CAPSULE_DISPATCH_INVALID_ARGUMENTS_EXIT_CODE
    assert "invalid or noncanonical mode arguments" in capsys.readouterr().err


def test_invalid_argv_does_not_import_any_mode_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def forbidden_import(name: str) -> Any:
        imports.append(name)
        raise AssertionError("invalid argv imported a mode module")

    monkeypatch.setattr(entry.importlib, "import_module", forbidden_import)
    valid = (
        authority.CAPSULE_PRETERMINAL_MODE,
        *_tail(tmp_path, authority.CAPSULE_PRETERMINAL_MODE),
    )

    assert (
        entry._dispatch_original_confirmatory_capsule(valid[:-1])
        == entry.CAPSULE_DISPATCH_INVALID_ARGUMENTS_EXIT_CODE
    )
    assert imports == []


@pytest.mark.parametrize(
    ("mode", "expected_module", "expected_attribute"),
    [
        (
            authority.CAPSULE_SCIENTIFIC_MODE,
            "histo_audit.workflows.original_confirmatory_capsule_authority",
            "_dispatch_original_confirmatory_run_from_canonical_tail",
        ),
        (
            authority.CAPSULE_PRETERMINAL_MODE,
            "histo_audit.workflows.original_confirmatory_capsule_terminal",
            "_verify_original_confirmatory_preterminal_from_canonical_tail",
        ),
        (
            authority.CAPSULE_TERMINAL_MODE,
            "histo_audit.workflows.original_confirmatory_capsule_terminal",
            "_verify_original_confirmatory_terminal_from_canonical_tail",
        ),
    ],
)
def test_production_dispatch_lazily_imports_only_the_exact_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_module: str,
    expected_attribute: str,
) -> None:
    imports: list[str] = []
    observed: list[tuple[str, ...]] = []
    tail = _tail(tmp_path, mode)

    def exact_handler(value: tuple[str, ...]) -> int:
        observed.append(value)
        return 0

    def fake_import(name: str) -> Any:
        imports.append(name)
        return types.SimpleNamespace(**{expected_attribute: exact_handler})

    monkeypatch.setattr(entry.importlib, "import_module", fake_import)

    assert entry._dispatch_original_confirmatory_capsule((mode, *tail)) == 0
    assert imports == [expected_module]
    assert observed == [tail]


def test_missing_exact_handler_has_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    mode = authority.CAPSULE_SCIENTIFIC_MODE

    def fake_import(name: str) -> Any:
        imports.append(name)
        return types.SimpleNamespace(fallback=lambda _tail: 0)

    monkeypatch.setattr(entry.importlib, "import_module", fake_import)

    assert (
        entry._dispatch_original_confirmatory_capsule((mode, *_tail(tmp_path, mode)))
        == entry.CAPSULE_DISPATCH_HANDLER_UNAVAILABLE_EXIT_CODE
    )
    assert imports == ["histo_audit.workflows.original_confirmatory_capsule_authority"]


@pytest.mark.parametrize(
    "bad_result",
    [None, True, "0", RuntimeError("synthetic handler failure"), SystemExit(0)],
)
def test_noninteger_or_exception_from_handler_fails_closed(
    tmp_path: Path,
    bad_result: Any,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    mode = authority.CAPSULE_SCIENTIFIC_MODE
    dispatch_map = _dispatch_map(calls, results={mode: bad_result})

    result = entry._dispatch_original_confirmatory_capsule_with_map_for_test(
        (mode, *_tail(tmp_path, mode)),
        dispatch_map=dispatch_map,
    )

    assert result == entry.CAPSULE_DISPATCH_HANDLER_FAILURE_EXIT_CODE
    assert len(calls) == 1


@pytest.mark.parametrize(
    "bad_map",
    [
        {},
        {authority.CAPSULE_SCIENTIFIC_MODE: lambda _tail: 0},
        {mode: None for mode in authority.CAPSULE_ALLOWED_MODES},
        {
            **{mode: (lambda _tail: 0) for mode in authority.CAPSULE_ALLOWED_MODES},
            "extra": lambda _tail: 0,
        },
    ],
)
def test_test_seam_requires_one_exact_closed_mode_map(
    tmp_path: Path,
    bad_map: dict[str, Any],
) -> None:
    mode = authority.CAPSULE_SCIENTIFIC_MODE

    result = entry._dispatch_original_confirmatory_capsule_with_map_for_test(
        (mode, *_tail(tmp_path, mode)),
        dispatch_map=bad_map,
    )

    assert result == entry.CAPSULE_DISPATCH_HANDLER_UNAVAILABLE_EXIT_CODE
