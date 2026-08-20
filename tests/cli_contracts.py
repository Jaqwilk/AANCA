"""Renderer-independent assertions for the public Typer command tree."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from typer.main import get_command


def resolve_cli_command(app: Any, path: Sequence[str]) -> Any:
    command = get_command(app)
    for component in path:
        commands = getattr(command, "commands", None)
        if not isinstance(commands, dict) or component not in commands:
            raise AssertionError(f"public CLI command is missing: {' '.join(path)}")
        command = commands[component]
    return command


def cli_options(app: Any, path: Sequence[str]) -> dict[str, Any]:
    command = resolve_cli_command(app, path)
    result: dict[str, Any] = {}
    for parameter in command.params:
        spellings = (
            *getattr(parameter, "opts", ()),
            *getattr(parameter, "secondary_opts", ()),
        )
        for spelling in spellings:
            result[spelling] = parameter
    return result
