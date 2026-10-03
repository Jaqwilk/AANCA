"""Shared deterministic fixtures for the synthetic scientific core."""

from __future__ import annotations

import json
import os
import platform
import runpy
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_SHARD_TOOLS = runpy.run_path(str(SRC.parent / "scripts" / "verify_test_shards.py"))
_SHARD_STATE: dict | None = None


def pytest_addoption(parser):
    group = parser.getgroup("aanca", "Optional complete-suite CI partitioning")
    group.addoption("--aanca-shard-count", type=int)
    group.addoption("--aanca-shard-index", type=int)
    group.addoption("--aanca-shard-manifest", type=Path)


def pytest_configure(config):
    global _SHARD_STATE
    _SHARD_STATE = None
    count = config.getoption("--aanca-shard-count")
    index = config.getoption("--aanca-shard-index")
    manifest = config.getoption("--aanca-shard-manifest")
    if count is None and index is None and manifest is None:
        return
    if count is None or index is None or manifest is None:
        raise pytest.UsageError("all three --aanca-shard-* options must be provided together")
    try:
        _SHARD_TOOLS["partition"]([], count, index)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    _SHARD_STATE = {
        "schema_version": 1,
        "selection_algorithm": _SHARD_TOOLS["ALGORITHM"],
        "platform": platform.system(),
        "source_revision": os.environ.get("GITHUB_SHA", "local-working-tree"),
        "shard_count": count,
        "shard_index": index,
        "all_node_ids": [],
        "selected_node_ids": [],
        "outcomes": {},
        "_manifest": manifest,
        "_started": time.monotonic(),
    }


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if _SHARD_STATE is None:
        return
    node_ids = sorted(item.nodeid.replace("\\", "/") for item in items)
    selected = _SHARD_TOOLS["partition"](
        node_ids, _SHARD_STATE["shard_count"], _SHARD_STATE["shard_index"]
    )
    _SHARD_STATE["all_node_ids"] = node_ids
    _SHARD_STATE["collection_sha256"] = _SHARD_TOOLS["collection_digest"](node_ids)
    _SHARD_STATE["selected_node_ids"] = selected
    selected_set = set(selected)
    deselected = [item for item in items if item.nodeid.replace("\\", "/") not in selected_set]
    items[:] = [item for item in items if item.nodeid.replace("\\", "/") in selected_set]
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_logreport(report):
    if _SHARD_STATE is None:
        return
    node = report.nodeid.replace("\\", "/")
    outcomes = _SHARD_STATE["outcomes"]
    if report.failed:
        outcomes[node] = "failed"
    elif outcomes.get(node) != "failed":
        if report.skipped:
            outcomes[node] = "skipped"
        elif report.when == "call" and report.passed:
            outcomes[node] = "passed"


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if _SHARD_STATE is None:
        return
    if exitstatus == 0 and set(_SHARD_STATE["outcomes"]) != set(_SHARD_STATE["selected_node_ids"]):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    receipt = {key: value for key, value in _SHARD_STATE.items() if not key.startswith("_")}
    receipt["pytest_exit_status"] = int(session.exitstatus)
    receipt["duration_seconds"] = round(time.monotonic() - _SHARD_STATE["_started"], 3)
    path = _SHARD_STATE["_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture(scope="session")
def synthetic_dataset():
    from histo_audit.data.synthetic import generate_synthetic_dataset

    return generate_synthetic_dataset(n_groups=18, instances_per_group=7, patch_size=48, seed=2027)
