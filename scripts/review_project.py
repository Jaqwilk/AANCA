"""Check the AANCA review materials without installing the research environment.

Default: standard-library integrity and narrative checks, no network or training.
--numeric: independently recalculate saved NuCLS/MoNuSAC arrays using NumPy.
--online: also compare the live article's served bytes with this local release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIT_MANIFEST = "reviewer-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def verify_kit(root: Path) -> dict[str, Any]:
    """Check a closed, portable reviewer-kit inventory; not an authenticity signature."""
    root = root.resolve(strict=True)
    manifest = json.loads((root / KIT_MANIFEST).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported reviewer manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_root_sha256"}
    if manifest.get("manifest_root_sha256") != canonical_digest(body):
        raise ValueError("reviewer manifest root differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_revision", ""))):
        raise ValueError("reviewer manifest lacks a full Git revision")
    if type(manifest.get("source_tree_dirty")) is not bool:
        raise ValueError("reviewer manifest lacks source-tree state")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("empty reviewer file inventory")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed reviewer file record")
        name = record.get("path")
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            raise ValueError("nonportable reviewer path")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {".", ".."} for part in relative.parts)
        ):
            raise ValueError("reviewer path escapes or is not canonical")
        if name in seen or name == KIT_MANIFEST:
            raise ValueError("duplicate or self-referential reviewer record")
        seen.add(name)
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"reviewer file absent or outside kit: {name}")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            raise ValueError("malformed reviewer file identity")
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f"reviewer file differs: {name}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != seen | {KIT_MANIFEST}:
        raise ValueError("reviewer kit contains missing or unexpected files")
    return {
        "file_count": len(records) + 1,
        "source_revision": manifest["source_revision"],
        "source_tree_dirty": manifest["source_tree_dirty"],
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "authenticity_scope": "internal integrity; compare ZIP SHA-256 with the published release",
    }


def review_project(
    root: Path, *, numeric: bool = False, online: str | None = None
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    checks: list[dict[str, Any]] = []

    def check(name, scope, operation):
        start = time.monotonic()
        try:
            result = operation()
            if isinstance(result, dict) and result.get("status") == "failed":
                raise ValueError(json.dumps(result))
            checks.append({"name": name, "status": "passed", "scope": scope, "result": result})
        except Exception as error:
            hint = (
                "; install: python -m pip install -r requirements-reviewer.txt"
                if isinstance(error, ModuleNotFoundError) and error.name == "numpy"
                else ""
            )
            checks.append(
                {
                    "name": name,
                    "status": "failed",
                    "scope": scope,
                    "error": f"{type(error).__name__}: {error}{hint}",
                }
            )
        checks[-1]["duration_seconds"] = round(time.monotonic() - start, 3)

    def tools(name):
        return runpy.run_path(str(root / "scripts" / name))

    if (root / KIT_MANIFEST).exists():
        check("offline_kit", "closed kit file identities", lambda: verify_kit(root))
        # Do not execute kit scripts whose recorded identity already failed.
        if checks[-1]["status"] == "failed":
            return {"status": "failed", "checks": checks, "models_trained": False}
    check(
        "article_and_sources",
        "13 presentation files, upstream source identities and selected narrative contracts",
        lambda: tools("verify_professor_release.py")["verify_professor_release"](),
    )
    if numeric:
        check(
            "nucls_saved_arrays",
            "NuCLS file identities, manifest, rankings, downstream metrics and group bootstraps",
            lambda: tools("verify_nucls_external_validation.py")["verify_release"](
                root / "artifacts" / "nucls_external_validation"
            ),
        )
        check(
            "monusac_saved_arrays",
            "MoNuSAC file identities, matched controls, metrics and whole-patient bootstrap gates",
            lambda: tools("verify_monusac_external_validation.py")["verify"](
                root / "artifacts" / "monusac_external_validation", root
            ),
        )
    if online:
        check(
            "served_article",
            "all served presentation bytes equal this local release",
            lambda: tools("verify_deployed_presentation.py")["verify_remote_package"](
                online,
                expected_manifest=json.loads(
                    (root / "artifacts/mvp_demo/manifest.json").read_text()
                ),
                timeout=30.0,
            ),
        )
    return {
        "schema_version": 1,
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "checks": checks,
        "numeric_recalculation_requested": numeric,
        "online_comparison_requested": bool(online),
        "models_trained": False,
        "source_annotations_modified": False,
        "not_verified": [
            "source-image/model retraining",
            "independent third-party replication",
            "pathologist error, clinical utility or prospective workflow benefit",
            "primary PanNuke arrays (separate primary-evidence-v1 release)",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--online", nargs="?", const="https://aancastudy.org/")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--report", type=Path, help="Save JSON outside an extracted kit, e.g. ../review.json"
    )
    args = parser.parse_args()
    result = review_project(PROJECT_ROOT, numeric=args.numeric, online=args.online)
    encoded = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.report:
        output = args.report.resolve()
        if (PROJECT_ROOT / KIT_MANIFEST).exists() and output.is_relative_to(PROJECT_ROOT):
            parser.error("save the report outside the immutable reviewer kit")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8", newline="\n")
    if args.json:
        print(encoded, end="")
    else:
        print(f"AANCA review checks: {result['status'].upper()}")
        for item in result["checks"]:
            print(f"  {item['status'].upper()}: {item['name']} ({item['duration_seconds']:.3f} s)")
            if "error" in item:
                print(f"    {item['error']}", file=sys.stderr)
            conclusion = item.get("result", {}).get("primary_claim_conclusion")
            if conclusion:
                print(f"    Retained scientific conclusion: {conclusion}")
            if "decision" in item.get("result", {}):
                print(f"    Retained scientific decision: {item['result']['decision']}")
        print(
            "No model training or source annotation changes. Passing checks do not establish clinical utility."
        )
        if not args.numeric:
            print(
                "Saved-array recalculation not requested; use --numeric after installing requirements-reviewer.txt."
            )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
