"""Build a deterministic, bounded AANCA reviewer ZIP and matching static guide.

No dataset download, model execution or mutation of scientific authorities. The
archive preserves the repository layout required by the existing scoped verifiers.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import runpy
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "aanca-reviewer-kit-v1.zip"
RELEASE = "https://github.com/Jaqwilk/AANCA/releases"
DOCS = (
    "README.md",
    "REVIEWER_GUIDE.md",
    "PROFESSOR_BRIEF.md",
    "PUBLIC_EVIDENCE.md",
    "ETHICS_AND_LIMITATIONS.md",
    "CITATION.cff",
    "REPRODUCIBILITY.md",
    "MVP_SCOPE.md",
    "STATUS.md",
    "FINAL_READINESS_REPORT.md",
    "SPEC.md",
    "PLAN.md",
    "PRE_REGISTRATION.md",
    "DECISIONS.md",
    "CONTRIBUTIONS.md",
    "LICENSE",
    "DATASET_SETUP.md",
    "evidence-release-manifest.json",
    "references/references.bib",
    "requirements-reviewer.txt",
    "configs/monusac_current_aanca_external.yaml",
)
SCRIPTS = (
    "review_project.py",
    "present_demo.py",
    "verify_professor_release.py",
    "verify_deployed_presentation.py",
    "verify_primary_evidence.py",
    "verify_nucls_external_validation.py",
    "verify_monusac_external_validation.py",
)


def render_guide(root: Path, *, offline: bool) -> str:
    evidence = json.loads((root / "artifacts/mvp_demo/evidence.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "artifacts/mvp_demo/manifest.json").read_text(encoding="utf-8"))
    docs = "" if offline else "https://github.com/Jaqwilk/AANCA/blob/main/"
    rows = []

    def row(name, text, source):
        link = docs + source
        rows.append(
            f'<div class="evidence-row"><dt>{html.escape(name)}</dt><dd>{text}'
            f'<a class="evidence-source" href="{html.escape(link)}" '
            f'aria-label="View source for {html.escape(name)}">View source'
            '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 12 12 4M4 4h8v8"/></svg>'
            "</a></dd></div>"
        )

    row(
        "PanNuke primary",
        "Positive controlled ranking; adverse H4 downstream restoration. The accepted primary analysis remains amended or exploratory.",
        "PUBLIC_EVIDENCE.md",
    )
    for name, key in (("RIVA", "riva"), ("MIDOG++", "midogpp")):
        value = evidence["public_independent_pathologist_replication"]["datasets"][key]
        delta = 100 * value["precision_difference"]
        row(
            name,
            f"Independent-expert disagreement precision increased by {delta:+.2f} percentage points versus matched random review at the 5% budget. Disagreement enrichment, not adjudicated error or clinical benefit.",
            "PUBLIC_EVIDENCE.md",
        )
    value = evidence["independent_pathologist_validation"]["primary"]
    row(
        "NuCLS independent pathologist",
        f"At 5% review, precision {value['aanca_precision']:.6f} versus {value['mean_matched_random_precision']:.6f} matched random. Limited to the qualifying JP.1 cohort across five patients.",
        "PROFESSOR_BRIEF.md",
    )
    row(
        "NuCLS multi-rater",
        "Frozen ranking rule failed; the guided downstream intervention was adverse. The retained primary claim is not supported.",
        "artifacts/nucls_external_validation/unbiased-v1/results.json",
    )
    row(
        "MoNuSAC controlled",
        "Retrieval improved, but downstream and class-safety gates failed. Action remains retain_uncorrected.",
        "artifacts/monusac_external_validation/results.json",
    )
    row(
        "PUMA controlled transfer",
        "All seven internally pre-specified gates passed. Controlled injected changes and flag_exclude training only; not natural-error detection. Public history does not independently establish pre-outcome freeze timing.",
        "artifacts/puma_new_data_confirmation/results.json",
    )
    template = (root / "docs/reviewer-guide.html").read_text(encoding="utf-8")
    font_css = []
    for filename, family, weights in (
        ("inter-latin", "Inter", "400 600"),
        ("jetbrains-mono-latin", "JetBrains Mono", "400 500"),
    ):
        directory = root / "docs/assets/reviewer-fonts"
        licence = (directory / f"{filename}-OFL.txt").read_text(encoding="utf-8")
        data = base64.b64encode((directory / f"{filename}.woff2").read_bytes()).decode("ascii")
        # Embed both face and licence so the generated guide has no asset/network
        # dependency, including when opened directly from a freshly extracted kit.
        font_css.append(
            f"/* {licence} */\n@font-face{{font-family:'{family}';font-style:normal;"
            f"font-weight:{weights};font-display:swap;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}"
        )
    substitutions = {
        "__FONT_CSS__": "\n".join(font_css),
        "__ARTICLE__": "artifacts/mvp_demo/index.html" if offline else "https://aancastudy.org/",
        "__REPO__": "https://github.com/Jaqwilk/AANCA",
        "__DOWNLOAD__": RELEASE + "/download/reviewer-kit-v1/" + ARCHIVE_NAME,
        "__CHECKSUM__": RELEASE + "/download/reviewer-kit-v1/" + ARCHIVE_NAME + ".sha256",
        "__DOCS__": docs,
        "__EVIDENCE_ROWS__": "\n".join(rows),
        "__ARTICLE_ROOT__": manifest["manifest_root_sha256"],
    }
    for token, replacement in substitutions.items():
        template = template.replace(token, replacement)
    return template


def build_kit(root: Path, output: Path, *, site_output: Path | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    review = runpy.run_path(str(root / "scripts/review_project.py"))
    professor = runpy.run_path(str(root / "scripts/verify_professor_release.py"))
    verified = professor["verify_professor_release"]()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    evidence = json.loads((root / "artifacts/mvp_demo/evidence.json").read_text(encoding="utf-8"))
    paths = set(DOCS) | {f"scripts/{name}" for name in SCRIPTS}
    presentation = runpy.run_path(str(root / "scripts/present_demo.py"))
    paths.update(
        f"artifacts/mvp_demo/{name}" for name in (*presentation["OUTPUT_FILES"], "manifest.json")
    )
    for record in professor["_source_records"](evidence):
        name = record["path"]
        paths.add(f"artifacts/mvp_demo/{name}" if name.startswith("assets/") else name)
    # Read the bounded, frozen file lists without importing the optional NumPy verifiers.
    # Five portable files per NuCLS subset and five MoNuSAC authority files.
    for subset in ("unbiased-v1", "evaluation-v1"):
        paths.update(
            f"artifacts/nucls_external_validation/{subset}/{name}"
            for name in (
                "artifact_manifest.json",
                "canonical_manifest.csv",
                "numeric_evidence.npz",
                "results.json",
                "source_inventory.json",
            )
        )
    paths.update(
        f"artifacts/monusac_external_validation/{name}"
        for name in (
            "artifact_manifest.json",
            "numeric_evidence.npz",
            "report.md",
            "results.json",
            "source_inventory.json",
        )
    )
    files = {}
    for name in sorted(paths):
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or "\\" in name
            or ":" in name
            or any(part in {".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"nonportable reviewer source path: {name}")
        source = root / name
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError(f"reviewer source absent or not regular: {name}")
        files[name] = source.read_bytes()
    files["index.html"] = render_guide(root, offline=True).encode()
    records = [
        {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    ]
    manifest = {
        "schema_version": 1,
        "source_revision": revision,
        "source_tree_dirty": dirty,
        "article_manifest_root_sha256": verified["package_manifest_root_sha256"],
        "files": records,
    }
    manifest["manifest_root_sha256"] = review["canonical_digest"](manifest)
    files["reviewer-manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_NAME
    temporary = archive.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(
                    "aanca-reviewer-kit-v1/" + name, date_time=(2026, 10, 3, 0, 0, 0)
                )
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                # The arrays and images are already compressed. Stored members also
                # avoid platform/zlib-version differences in the archive identity.
                target.writestr(info, data, compress_type=zipfile.ZIP_STORED)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    digest = review["sha256"](archive)
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii", newline="\n"
    )
    result = {
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": digest,
        "source_revision": revision,
        "source_tree_dirty": dirty,
        "file_count": len(files),
        "article_manifest_root_sha256": verified["package_manifest_root_sha256"],
        "kit_manifest_root_sha256": manifest["manifest_root_sha256"],
    }
    if site_output is not None:
        site_output.mkdir(parents=True, exist_ok=True)
        (site_output / "index.html").write_text(
            render_guide(root, offline=False), encoding="utf-8", newline="\n"
        )
        public_snapshot = {**result, "archive": ARCHIVE_NAME}
        (site_output / "snapshot.json").write_text(
            json.dumps(public_snapshot, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/qa/reviewer-kit"
    )
    parser.add_argument("--site-dir", type=Path)
    parser.add_argument(
        "--verify-extracted",
        action="store_true",
        help="Run the fresh kit outside the checkout with isolated Python",
    )
    args = parser.parse_args()
    try:
        result = build_kit(PROJECT_ROOT, args.output_dir, site_output=args.site_dir)
        if args.verify_extracted:
            with tempfile.TemporaryDirectory(prefix="aanca-review-") as directory:
                with zipfile.ZipFile(result["archive"]) as archive:
                    archive.extractall(directory)
                script = Path(directory) / "aanca-reviewer-kit-v1/scripts/review_project.py"
                subprocess.run([sys.executable, "-I", str(script)], cwd=directory, check=True)
            result["isolated_extraction_check"] = "passed"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Reviewer kit build failed: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
