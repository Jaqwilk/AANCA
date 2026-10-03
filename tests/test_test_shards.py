"""Coverage and failure gates for parallel test execution."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

TOOLS = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "verify_test_shards.py"))
partition = TOOLS["partition"]
verify = TOOLS["verify_receipts"]


def _receipts(tmp_path, *, count=2):
    nodes = [f"tests/test_example.py::test_case[{index}]" for index in range(40)]
    nodes.sort()
    paths = []
    for index in range(count):
        selected = partition(nodes, count, index)
        receipt = {
            "schema_version": 1,
            "selection_algorithm": TOOLS["ALGORITHM"],
            "platform": "Windows",
            "source_revision": "revision-one",
            "shard_count": count,
            "shard_index": index,
            "all_node_ids": nodes,
            "collection_sha256": TOOLS["collection_digest"](nodes),
            "selected_node_ids": selected,
            "outcomes": dict.fromkeys(selected, "passed"),
            "pytest_exit_status": 0,
        }
        path = tmp_path / f"pytest-shard-{index}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        paths.append(path)
    return paths


def test_partition_is_disjoint_complete_and_order_independent(tmp_path):
    paths = _receipts(tmp_path)
    result = verify(paths, platforms=["Windows"], shard_count=2, revision="revision-one")
    assert result["platforms"]["Windows"]["collected"] == 40
    nodes = ["z", "a", "b", "c"]
    selected = [partition(nodes, 2, index) for index in range(2)]
    assert sorted(selected[0] + selected[1]) == sorted(nodes)
    assert not set(selected[0]) & set(selected[1])
    assert partition(nodes[::-1], 2, 0) == selected[0]


@pytest.mark.parametrize(
    "defect",
    ["failed", "missing", "collection", "exit", "revision", "selection", "duplicate", "platform"],
)
def test_incomplete_or_inconsistent_execution_is_rejected(tmp_path, defect):
    paths = _receipts(tmp_path)
    receipt = json.loads(paths[1].read_text(encoding="utf-8"))
    node = next(iter(receipt["outcomes"]))
    if defect == "failed":
        receipt["outcomes"][node] = "failed"
    elif defect == "missing":
        del receipt["outcomes"][node]
    elif defect == "collection":
        receipt["all_node_ids"].pop()
        receipt["collection_sha256"] = TOOLS["collection_digest"](receipt["all_node_ids"])
    elif defect == "exit":
        receipt["pytest_exit_status"] = 1
    elif defect == "revision":
        receipt["source_revision"] = "different-revision"
    elif defect == "selection":
        receipt["selected_node_ids"].pop()
    elif defect == "duplicate":
        receipt = copy.deepcopy(json.loads(paths[0].read_text(encoding="utf-8")))
    elif defect == "platform":
        receipt["platform"] = "Linux"
    paths[1].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        verify(paths, platforms=["Windows"], shard_count=2, revision="revision-one")


def test_missing_receipt_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing or additional"):
        verify(_receipts(tmp_path)[:1], platforms=["Windows"], shard_count=2)


@pytest.mark.parametrize("count,index", [(0, 0), (2, 2), (2, -1), (True, 0), (2, True)])
def test_invalid_shard_configuration_is_rejected(count, index):
    with pytest.raises(ValueError):
        partition([], count, index)
