<h1 align="center">AANCA</h1>

<p align="center"><strong>Automated Auditing of Nucleus Class Annotations</strong></p>

<p align="center">
A reproducible, group-safe research workflow for ranking nucleus class annotations
that may warrant expert review—without changing source labels.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Presentation status: DEMO_COMPLETE" src="https://img.shields.io/badge/presentation-DEMO__COMPLETE-6D67E4">
  <img alt="Scientific status: PRIMARY_STUDY_COMPLETE" src="https://img.shields.io/badge/science-PRIMARY__STUDY__COMPLETE-238636">
  <img alt="Research only, non-diagnostic" src="https://img.shields.io/badge/use-research_only_%7C_non--diagnostic-C2410C">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#evidence-at-a-glance">Evidence</a> ·
  <a href="#how-it-works">Method</a> ·
  <a href="#command-line-interface">CLI</a> ·
  <a href="#documentation-map">Documentation</a> ·
  <a href="#scope-ethics-and-limitations">Limitations</a>
</p>

![AANCA presentation showing immutable source annotations and a ranked expert-review queue](output/playwright/aanca-professor-release-hero.png)

<p align="center"><em>Conceptual workflow illustration from the static presentation; not benchmark data.</em></p>

> [!IMPORTANT]
> AANCA is a university research prototype, not a diagnostic system or medical
> device. A high score means only `potentially inconsistent annotation` and
> `recommended for expert review`. Model disagreement does not prove that an
> annotation or a pathologist is wrong, and the software never modifies source
> annotations automatically.

## Overview

AANCA asks a focused data-quality question:

> Can an automated, source-group-safe audit rank intentionally corrupted nucleus
> class labels more efficiently than random review, and does restoring the
> highest-ranked injected corruptions improve downstream classification?

The repository implements the complete controlled workflow around that question:
verified PanNuke ingestion, anomaly-safe mask quality control, immutable label
provenance, controlled corruption, group-safe out-of-fold (OOF) predictions,
multiple annotation-risk scores, fixed-budget review simulation, paired
group-bootstrap statistics, restoration experiments, sealed evidence, and a
checksum-verifiable static presentation.

The project is deliberately an **auditing and prioritisation system**. It is not a
nucleus detector, segmentation repair tool, clinical classifier, or automatic
relabelling system.

## Current status

| Area | Current state |
| --- | --- |
| Presentation | `DEMO_COMPLETE` — static, responsive, checksum-verifiable MVP |
| Scientific boundary | `PRIMARY_STUDY_COMPLETE` |
| Primary matrix | 185/185 required cells completed; 0 required failures |
| Optional primary cells | 37 pathology-encoder cells skipped under the frozen availability rule |
| Statistical output | 36 preregistered H1/H3/H5/H6/H7 entries: 33 numeric results and 3 explicitly unavailable H6 entries; 2,000 group-bootstrap iterations where applicable |
| Analysis disposition | `amended_or_exploratory` because outcomes had been inspected before recovery finalisation |
| Not claimed | `CONFIRMATORY_COMPLETE`, `EXTERNAL_VALIDATION_READY`, or `EXTERNAL_VALIDATION_COMPLETE` |

The accepted primary run is
`20260727T133947.089370Z_pannuke_primary_orphan_recovery`. Its full run directory
is intentionally local and ignored by Git; the compact checked-in presentation
contains a machine-readable, checksum-bound evidence snapshot.

See [`MVP_SCOPE.md`](MVP_SCOPE.md), [`STATUS.md`](STATUS.md), and
[`artifacts/mvp_demo/evidence.json`](artifacts/mvp_demo/evidence.json) for the
exact boundary and saved evidence.

### Presentation experience

The checked-in article is designed for both a research presentation and a close
technical reading. Its English narrative uses a centered editorial column, a
scroll-driven five-stage serpentine explanation of the benchmark, and visible
evidence panels that never require disclosure clicks. The H1–H7 ledger preserves
supportive, qualified, adverse, neutral, and unavailable outcomes together; the
dense comparison table becomes stacked records on narrow screens and can be
filtered by hypothesis, status, or free text.

GSAP coordinates the directional header, logo reveal, section transitions, chart
marks, and cumulative method story. Three.js renders the conceptual source-patch
to review-queue scene in the hero. Responsive and reduced-motion layouts retain
the complete scientific content without depending on those enhancements.

Runtime work is visibility-aware: the WebGL render loop pauses outside the hero
and while the browser tab is hidden, high-density rendering is capped, and the
large QC image is lazily decoded with explicit dimensions. These optimisations
reduce background CPU/GPU work and layout shift without changing any evidence.

## Evidence at a glance

The values below are read from the checked-in MVP evidence package. They measure
recovery of **injected label changes**, not validation of naturally occurring
annotation inconsistencies.

| Registered result | Saved evidence |
| --- | --- |
| H1 — self-confidence vs random review | Average-precision difference ranged from **+0.036206 to +0.301884** across 12 mechanism/seed comparisons |
| H2 — subgroup heterogeneity | 32,760 reported class/tissue/mechanism/rate rows; descriptive variation was substantial and is not an omnibus causal test |
| H3 — corruption difficulty | Instance-dependent corruption was markedly harder than symmetric corruption in the registered comparisons |
| H4 — downstream restoration | **Adverse to the hypothesis:** audit-guided minus random-review macro F1 = **−0.002156**, 95% group-bootstrap interval **[−0.002859, −0.001393]** |
| H5 — fixed hybrid | Average-precision gain over self-confidence ranged from **+0.020983 to +0.065502** across 12 comparisons |
| H6 — pathology representation | No result: the optional encoder did not satisfy every frozen access, licence, reproducibility, hardware, and smoke-test gate |
| H7 — explicit target indication | Differences ranged from **−0.005371 to +0.002997**; the saved comparisons do not show a consistent benefit |

Three registered instance-dependent seeds produced byte-identical rankings and OOF
predictions. They are retained for auditability but represent **one deterministic
realisation, not three independent replications**.

The primary study produced 2,288 sealed artifacts and retained neutral, adverse,
missing, and unavailable outcomes. Detailed comparison rows, intervals, adjusted
p-values, subgroup ranges, hashes, and the H4 restoration record are available in
[the evidence package](artifacts/mvp_demo/evidence.json).

## How it works

~~~mermaid
flowchart LR
    A["Verified local PanNuke release"] --> B["Immutable QC and nucleus manifest"]
    B --> C["Official-fold and group-safe partitioning"]
    C -->|"Development groups"| D["Controlled corruption of audit pool only"]
    D --> E["Independent representations and probabilistic models"]
    E --> F["Group-safe OOF probabilities"]
    F --> G["Risk scores and fixed review budgets"]
    G --> H["Injected-corruption ranking and restoration evaluation"]
    C -->|"Untouched official fold"| T["Uncorrupted final reference test<br/>unavailable for tuning"]
    T --> H
    H --> I["Sealed machine-readable artifacts"]
    I --> J["Static evidence-bound presentation"]
~~~

### Core safeguards

- **Group-safe splitting:** `group_id` is at least the source patch; individual
  nuclei are never randomly split across train and holdout.
- **Untouched final reference:** the primary final fold remains uncorrupted and
  unavailable for tuning, method selection, calibration, or review-budget choice.
- **OOF-only primary scoring:** every audited sample is scored by a model that did
  not train on that sample or any member of its source group.
- **Immutable label lineage:** `pre_corruption_label`, `observed_label`,
  `is_injected_corruption`, and corruption metadata remain separate.
- **Representation independence:** instance-dependent corruption and its auditor
  use independently assessed feature spaces; circular cases are labelled
  `circularity_risk` and excluded from confirmatory claims.
- **Fixed review budgets:** guided and random review use identical integer budgets.
- **Fail-closed evidence:** missing Cleanlab output, unavailable encoders, failed
  cells, or insufficient subgroup support remain explicit—nothing is fabricated.
- **No source mutation:** raw masks, source manifests, and source annotations are
  never rewritten.

### Audit signals

| Signal | Role |
| --- | --- |
| Self-confidence | `1 - P(observed_label)`; primary annotation-risk score |
| Negative log-likelihood | Penalises low probability assigned to the observed class |
| Prediction margin | Compares the strongest alternative with the observed class |
| Predictive entropy | Ambiguity baseline |
| Cleanlab | Label-quality signal computed only from group-safe OOF probabilities |
| Fold-safe neighbours | Disagreement with valid reference groups while excluding the sample and its source group |
| Fixed hybrid | Equal-weight percentile combination of complementary registered signals |
| Ensemble disagreement | Reserved for the deferred confirmatory design |

## Quick start

### 1. Verify and open the checked-in presentation

The recommended launcher uses only Python's standard library: it verifies every
package checksum, starts a loopback-only local server, and opens the article. It
does not install the ML environment, run a model, or need a dataset or GPU:

~~~powershell
git clone https://github.com/Jaqwilk/AANCA.git
Set-Location AANCA
python .\scripts\present_demo.py
~~~

Use `--no-open` for a headless session or `--port 0` to choose an available port.
The server binds to `127.0.0.1` by default. Directly opening
`artifacts\mvp_demo\index.html` remains a zero-server fallback; pinned Three.js and
GSAP enhancements load from jsDelivr when network access is available.

### 2. Verify the presentation package without opening it

The same dependency-free launcher can verify the closed package for CI or review:

~~~powershell
python .\scripts\present_demo.py --verify-only
~~~

A valid package reports a closed five-file allowlist with matching checksums,
`DEMO_COMPLETE` presentation status, and `PRIMARY_STUDY_COMPLETE` scientific
status. After `uv sync --dev`, the equivalent full-CLI commands are
`uv run histo-audit demo serve` and `uv run histo-audit demo verify`.

### 3. Run the deterministic software smoke path

The full research workflow uses Python 3.12 and
[`uv`](https://docs.astral.sh/uv/):

~~~powershell
uv sync --dev
uv run histo-audit doctor
uv run histo-audit data generate-synthetic --config configs\smoke.yaml
uv run histo-audit experiment smoke
uv run histo-audit experiment smoke --config configs\smoke_zero.yaml
~~~

Synthetic success validates the software pipeline only. It must not be described
as real-data, primary, confirmatory, expert-reviewed, or clinical validation.

### Platform note

The committed lock selects PyTorch 2.12.1 and torchvision 0.27.1 from the CUDA 12.6
index for the reference Windows/NVIDIA workstation. CPU-only, macOS, Linux, or
different accelerator environments should resolve an appropriate platform-specific
PyTorch source using the official selector; installing a system CUDA Toolkit is not
a substitute for resolving compatible wheels.

## PanNuke data

PanNuke binaries are **not distributed through this repository**. Obtain the release
legitimately from the
[official University of Warwick source](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/),
review its terms, and keep it immutable. The project never downloads from an
unverified mirror or bypasses access controls.

After placing a lawful local copy, set `PANNUKE_ROOT` and run the gated inspectors:

~~~powershell
$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path

.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root $env:PANNUKE_ROOT --max-samples-per-fold 100000 --max-overlay-patches 24
.venv\Scripts\python.exe -m histo_audit data build-manifest --project-root . --root $env:PANNUKE_ROOT --batch-rows 4096
.venv\Scripts\python.exe -m histo_audit data audit-duplicates --project-root . --root $env:PANNUKE_ROOT --embedding-device cuda
~~~

The validator inspects actual shapes, dtypes, class channels, supplied background,
voids, cross-class overlaps, instance identities, and raw-file hashes. Similarity
candidates remain review-only and never trigger automatic deletion or reassignment.
See [`DATASET_SETUP.md`](DATASET_SETUP.md) for the complete acquisition and licence
gate.

## Reproducibility and evidence

AANCA treats reproducibility as part of the scientific method rather than as a
post-processing step:

- `uv.lock` fixes the resolved Python dependency graph.
- YAML configs separate synthetic, pilot, frozen primary, confirmatory, and external
  validation definitions.
- Every tracked run receives a unique ID, registry record, configuration identity,
  provenance metadata, and immutable artifact manifest.
- Completed artifacts are sealed and verified by SHA-256; overwrite is rejected.
- Statistical comparisons resample complete source groups identically across
  operands.
- Reports and the static demo read machine-readable artifacts instead of embedding
  hand-entered benchmark values.
- Raw data, full run directories, embeddings, checkpoints, and predictions remain
  outside Git; small reviewable evidence and manifests are committed deliberately.
- Negative, neutral, missing, optional, and adverse outcomes are preserved.

To verify the checked-in MVP:

~~~powershell
python .\scripts\present_demo.py --verify-only
~~~

To rebuild it on the original research workspace, use a new output directory; the
builder refuses to overwrite an existing package:

~~~powershell
uv run histo-audit demo build --project-root . --run-dir artifacts\runs\20260727T133947.089370Z_pannuke_primary_orphan_recovery --qc-bundle reports\pannuke_qc --output-dir artifacts\mvp_demo_rebuild
~~~

This rebuild requires the full locally retained sealed run and QC bundle, which are
not fully distributed in Git.

## Command-line interface

Run `python -m histo_audit --help` for the authoritative command tree.

| Command group | Purpose |
| --- | --- |
| `doctor` | Record environment, hardware, CUDA, package, disk, and data availability |
| `data` | Generate synthetic data; verify, validate, inspect duplicates, and manifest PanNuke |
| `representations` | Extract target crops, engineered features, and frozen ResNet embeddings |
| `experiment` | Run immutable smoke, pilot, primary-recovery, and governed study workflows |
| `preregistration` | Freeze, amend, and verify immutable analysis authorities |
| `audit` | Rank unmodified original labels using group-safe OOF evidence |
| `external` | Build blinded external-review packages when eligibility gates are satisfied |
| `report` | Build sourced Markdown and static HTML from strict metrics JSON |
| `demo` | Build, checksum-verify, or locally serve the static presentation MVP |

Real-data study commands are intentionally hard-gated. The original confirmatory
study, original-label audit execution, genuine expert review, and external validation
remain deferred; an implemented command is not evidence that a stage has run.

## Repository layout

~~~text
AANCA/
├── src/histo_audit/
│   ├── data/                 # manifests, grouping, duplicates, synthetic data
│   ├── pannuke/              # acquisition, semantic validation, QC, publication
│   ├── representations/      # crops, morphometrics, ImageNet features
│   ├── corruption/           # controlled immutable label corruption
│   ├── cross_validation/     # group-safe OOF prediction
│   ├── auditing/             # annotation-risk scores and neighbours
│   ├── statistics/           # review metrics and group bootstrap
│   ├── evaluation/           # controlled restoration experiments
│   ├── experiment/           # tracked study runners and gates
│   ├── reporting/            # evidence-backed reports and figures
│   └── workflows/            # frozen/governed execution authorities
├── configs/                  # smoke, pilot, primary, confirmatory, external
├── scripts/present_demo.py   # dependency-free verified local presentation launcher
├── tests/                    # unit, integration, security, lifecycle, and CLI tests
├── reports/                  # compact QC, provenance, literature, and validation evidence
├── artifacts/mvp_demo/       # checked-in five-file static presentation package
├── references/               # bibliographic records
├── data/                     # ignored raw/interim/processed data roots
└── *.md                      # scientific governance and handoff documents
~~~

## Documentation map

| Document | Purpose |
| --- | --- |
| [`SPEC.md`](SPEC.md) | Frozen scientific terminology, hypotheses, leakage rules, and completion vocabulary |
| [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) | Exact primary and confirmatory analysis definition |
| [`PLAN.md`](PLAN.md) | Milestones, gates, acceptance criteria, and deferred work |
| [`STATUS.md`](STATUS.md) | Executed commands, evidence, blockers, and current handoff |
| [`DECISIONS.md`](DECISIONS.md) | Append-only rationale and operational decision history |
| [`MVP_SCOPE.md`](MVP_SCOPE.md) | Reduced presentation boundary and acceptance checks |
| [`DATASET_SETUP.md`](DATASET_SETUP.md) | PanNuke acquisition, integrity, QC, and licence gate |
| [`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md) | Non-diagnostic scope, scientific limits, and responsible language |
| [`CAPSULE_DESIGN.md`](CAPSULE_DESIGN.md) | Advanced governed execution-capsule architecture |
| [`references/references.bib`](references/references.bib) | Project bibliography, including both required PanNuke citations |

Read the first four documents before changing experiment code or scientific claims.

## Scope, ethics, and limitations

AANCA currently evaluates class-label consistency for **already segmented nucleus
instances**. It does not evaluate missing nuclei, merged or split instances, contour
quality, diagnosis, prognosis, treatment, patient outcomes, or deployment safety.

Key limitations include:

- Controlled corruption provides an objective software benchmark but may differ
  materially from natural ambiguity, annotator variation, and biological boundaries.
- Source-patch separation prevents local group leakage but does not establish
  patient- or whole-slide independence unless reliable identifiers exist.
- Model probabilities, class imbalance, tissue shift, representation choice, crop
  failures, and calibration can change audit rankings.
- PanNuke labels are quality-controlled reference annotations, not guaranteed
  biological truth.
- External multi-rater or blinded expert validation is still required before making
  claims about naturally occurring annotation inconsistency.
- Human disagreement and insufficient context must be preserved rather than forced
  into a single truth.
- Nothing in this repository supports clinical use or patient-level decisions.

The complete policy is in
[`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md).

## Roadmap

Completed evidence stages:

- `PIPELINE_COMPLETE`
- `PILOT_COMPLETE`
- `PRE_REGISTRATION_FROZEN`
- `PRIMARY_STUDY_COMPLETE`
- `DEMO_COMPLETE`

Explicitly deferred:

- the unchanged original confirmatory matrix;
- real original-label audit execution;
- a blinded expert-review package backed by eligible real audit evidence;
- genuine expert or multi-rater evaluation;
- external domain validation.

Accordingly, `CONFIRMATORY_COMPLETE`, `EXTERNAL_VALIDATION_READY`, and
`EXTERNAL_VALIDATION_COMPLETE` are not claimed.

## Development and validation

Before submitting a change:

~~~powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
.venv\Scripts\python.exe -m histo_audit demo verify --output-dir artifacts\mvp_demo
~~~

Mandatory scientific invariants live in [`AGENTS.md`](AGENTS.md) and
[`SPEC.md`](SPEC.md). Material changes must update the corresponding evidence in
`STATUS.md` and rationale in `DECISIONS.md`. A failed mandatory gate stops
advancement.

## Data terms, software licence, and citation

Dataset files and pretrained weights retain their own licences and access terms. The
local PanNuke release evidence explicitly applies CC BY-NC-SA 4.0 to the `masks/`
directory; it does not establish identical licence scope for every release file.
This project additionally restricts PanNuke use to non-commercial research and
requires both PanNuke works identified as `gamper2019pannuke` and
`gamper2020pannukeextension` in
[`references/references.bib`](references/references.bib).

This repository does not currently include a standalone general-purpose open-source
`LICENSE` file. Do not assume rights beyond the research-prototype statement in
[`pyproject.toml`](pyproject.toml) and the applicable licences of datasets,
dependencies, and pretrained weights.

## Author

University research prototype. Presentation release by **Natan Smogór**, dated
18 August 2026.
