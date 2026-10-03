"""Partition pytest cases and require complete execution across isolated CI jobs.

This is test infrastructure only. It never changes scientific execution or skips
tests in a normal ``pytest`` invocation. Receipts are execution records, not proof
of scientific validity or an independently authenticated collection of tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ALGORITHM = "sha256-nodeid-modulo-v1"


def partition(node_ids: list[str], count: int, index: int) -> list[str]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("shard count must be a positive integer")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError("shard index must be between zero and count minus one")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate collected test case")
    return sorted(
        node
        for node in node_ids
        if int(hashlib.sha256(node.encode()).hexdigest(), 16) % count == index
    )


def collection_digest(node_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(node_ids), ensure_ascii=True).encode()).hexdigest()


def _nodes(value: Any, role: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(node, str) and node for node in value):
        raise ValueError(f"{role} must be a list of test identifiers")
    if value != sorted(set(value)):
        raise ValueError(f"{role} must be sorted and unique")
    return value


def verify_receipts(
    paths: list[Path], *, platforms: list[str], shard_count: int, revision: str | None = None
) -> dict[str, Any]:
    if not platforms or len(set(platforms)) != len(platforms):
        raise ValueError("expected platforms must be nonempty and unique")
    partition([], shard_count, 0)
    if len(paths) != len(platforms) * shard_count:
        raise ValueError("missing or additional shard receipts")
    groups: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in platforms}
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
        ):
            raise ValueError(f"invalid receipt schema: {path}")
        if receipt.get("selection_algorithm") != ALGORITHM:
            raise ValueError("unknown test selection algorithm")
        platform = receipt.get("platform")
        if not isinstance(platform, str) or platform not in groups:
            raise ValueError("unexpected platform")
        index = receipt.get("shard_index")
        if type(receipt.get("shard_count")) is not int or receipt["shard_count"] != shard_count:
            raise ValueError("shard count differs")
        partition([], shard_count, index)
        if index in groups[platform]:
            raise ValueError("duplicate shard receipt")
        if revision is not None and receipt.get("source_revision") != revision:
            raise ValueError("source revision differs")
        if type(receipt.get("pytest_exit_status")) is not int or receipt["pytest_exit_status"] != 0:
            raise ValueError("pytest did not finish successfully")
        all_nodes = _nodes(receipt.get("all_node_ids"), "collection")
        selected = _nodes(receipt.get("selected_node_ids"), "selection")
        if not all_nodes or receipt.get("collection_sha256") != collection_digest(all_nodes):
            raise ValueError("empty or inconsistent collection")
        if selected != partition(all_nodes, shard_count, index):
            raise ValueError("selection does not match the deterministic partition")
        outcomes = receipt.get("outcomes")
        if not isinstance(outcomes, dict) or set(outcomes) != set(selected):
            raise ValueError("selected tests lack complete execution outcomes")
        if any(outcome not in {"passed", "skipped"} for outcome in outcomes.values()):
            raise ValueError("a test failed or has an unsupported outcome")
        groups[platform][index] = receipt
    results = {}
    for platform, shards in groups.items():
        if sorted(shards) != list(range(shard_count)):
            raise ValueError("missing shard index")
        authority = shards[0]["all_node_ids"]
        if any(receipt["all_node_ids"] != authority for receipt in shards.values()):
            raise ValueError("shards collected different test suites")
        revisions = {receipt.get("source_revision") for receipt in shards.values()}
        if len(revisions) != 1:
            raise ValueError("shards ran different revisions")
        executed = [node for receipt in shards.values() for node in receipt["outcomes"]]
        if sorted(executed) != authority:
            raise ValueError("test coverage is incomplete or overlapping")
        outcomes = [
            outcome for receipt in shards.values() for outcome in receipt["outcomes"].values()
        ]
        results[platform] = {
            "collected": len(authority),
            "passed": outcomes.count("passed"),
            "skipped": outcomes.count("skipped"),
            "collection_sha256": collection_digest(authority),
            "source_revision": shards[0].get("source_revision"),
        }
    return {"status": "passed", "shard_count": shard_count, "platforms": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts_root", type=Path)
    parser.add_argument("--platform", action="append", required=True, dest="platforms")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    try:
        result = verify_receipts(
            sorted(args.receipts_root.rglob("pytest-shard-*.json")),
            platforms=args.platforms,
            shard_count=args.shard_count,
            revision=args.revision,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Test coverage verification failed: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
