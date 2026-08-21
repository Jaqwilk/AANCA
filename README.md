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
  <img alt="Controlled new-source PUMA result supported; natural validation incomplete" src="https://img.shields.io/badge/external-PUMA_controlled_supported-8B5CF6">
  <img alt="Research only, non-diagnostic" src="https://img.shields.io/badge/use-research_only_%7C_non--diagnostic-C2410C">
  <a href="https://github.com/Jaqwilk/AANCA/actions/workflows/scientific-software.yml"><img alt="Presentation integrity and cross-platform scientific software workflow" src="https://github.com/Jaqwilk/AANCA/actions/workflows/scientific-software.yml/badge.svg?branch=main"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#evidence-at-a-glance">Evidence</a> ·
  <a href="#how-it-works">Method</a> ·
  <a href="#command-line-interface">CLI</a> ·
  <a href="REPRODUCIBILITY.md">Reproducibility</a> ·
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
| External multi-rater evaluation | `EXTERNAL_VALIDATION_COMPLETE` — frozen NuCLS claims not supported |
| New-data controlled external benchmark | Frozen PUMA confirmation passed all seven gates; controlled-noise transfer supported |
| Bounded autoresearch development | 400 screens and 12 full nested patient-group evaluations completed; one development-only candidate passed every frozen gate |
| Primary matrix | 185/185 required cells completed; 0 required failures |
| Optional primary cells | 37 pathology-encoder cells skipped under the frozen availability rule |
| Statistical output | 36 preregistered H1/H3/H5/H6/H7 entries: 33 numeric results and 3 explicitly unavailable H6 entries; 2,000 group-bootstrap iterations where applicable |
| Analysis disposition | `amended_or_exploratory` because outcomes had been inspected before recovery finalisation |
| Not claimed | `CONFIRMATORY_COMPLETE`, natural/pathologist-error detection, or clinical utility |

The accepted primary run is
`20260727T133947.089370Z_pannuke_primary_orphan_recovery`. Its full run directory
is local and ignored by Git. The public
[`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)
release now provides all completed-cell OOF predictions and rankings, the full group
bootstrap, subgroup table and H4 restoration arrays. The independent verifier
recalculates the saved H1-H7 comparison statistics without importing AANCA.

The later NuCLS study was publicly frozen before outcome-table download and is now
complete. In the primary Unbiased Control subset, neither the combined ranking rule
nor the downstream rule passed. Guided correction was adverse versus leaving labels
unchanged. The checked-in derived evidence and independent verifier are documented
in [`reports/nucls_external_validation_results.md`](reports/nucls_external_validation_results.md).
This completes a genuine external multi-rater evaluation; it does not prove that a
pathologist was wrong or that consensus is biological truth.
The sealed derived evidence is also available as GitHub release
[`nucls-external-validation-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/nucls-external-validation-v1).

The prospectively frozen MoNuSAC controlled-external test is also complete. On
29,610 development nuclei from 44 patients, the primary fold-safe neighbour queue
found 1,035 of 2,961 injected changes while reviewing 1,481 nuclei. Its precision
gain over exact matched random was `+0.142843`, with 95% whole-patient interval
`[+0.099181, +0.188491]`. On 15,494 untouched test nuclei from 25 patients, the
primary macro-F1 point estimate improved by `+0.005526` over corrupted/no review,
but its interval `[-0.001506, +0.012833]` crossed zero, it was indistinguishable
from matched-random restoration and the registered class-recall safeguard failed.
The overall frozen decision is therefore **not supported**. Exact results and the
independent readback are in
[`reports/monusac_current_aanca_external_results.md`](reports/monusac_current_aanca_external_results.md).

After that test was sealed, a separate bounded development search used only the 44
eligible official MoNuSAC training patients. It screened 400 registered combinations
and evaluated 12 finalists with nested patient-group separation. The selected 5%
`flag_exclude` policy increased controlled-development macro-F1 from `0.504104` to
`0.547194`: `+0.043090`, 95% whole-patient interval `[+0.032553, +0.055065]`.
It also exceeded its exact matched-random comparator by `+0.054399`, interval
`[+0.034938, +0.075507]`, with positive effects in all four registered seeds. Both
registered development intervals excluded zero, but because this was an adaptive
development search they are not independent confirmatory intervals. The exact
search and selection record is in
[`reports/aanca_autoresearch_expanded_development_results.md`](reports/aanca_autoresearch_expanded_development_results.md).
An exact rerun reproduced the saved metrics and all `220/220` optimiser fits
converged.

That candidate was then frozen and evaluated without PUMA tuning on the official
PUMA melanoma source. The hash split used 144 development ROI/case groups (67,032
nuclei) and 62 untouched final groups (30,397 nuclei). AANCA review precision was
`0.537739`, versus `0.214379` for exact matched random. On the final groups,
`flag_exclude` improved macro F1 by `+0.006426` over unchanged corrupted training,
95% whole-group interval `[+0.003657, +0.009365]`, and by `+0.008067` over exact
matched-random exclusion, interval `[+0.004093, +0.011947]`. Every frozen retrieval,
downstream, seed-direction, class-safety, convergence and source/split gate passed,
and the independent verifier accepted the package. The complete result is in
[`reports/puma_new_data_confirmation_results.md`](reports/puma_new_data_confirmation_results.md).

This is positive new-source evidence for controlled annotation noise, not a claim
that natural pathologist errors were detected. PUMA publishes final expert-checked
labels without paired natural pre/post reviews. A later exploratory stress found
positive aggregate macro-F1 lower bounds in all nine clean and controlled-corruption
scenarios, but only one passed every class-recall safeguard. In particular, clean
5% exclusion reduced recall of the heterogeneous `other` class despite improving
macro F1. Exact stress results are in
[`reports/puma_realism_stress_results.md`](reports/puma_realism_stress_results.md),
and the natural-data action remains `retain_uncorrected`.

The primary controlled fold allocation used pre-corruption labels, which are not
available in a real audit. A separately frozen post-confirmation sensitivity rebuilt
folds and exact neighbours from `observed_label` only. All seven sensitivity gates
still passed: macro F1 improved by `+0.006679` over unchanged and `+0.009069` over
matched random, with positive intervals and all class safeguards intact. See
[`reports/puma_audit_time_label_sensitivity_results.md`](reports/puma_audit_time_label_sensitivity_results.md).

The full expert assessment—including the evidence ladder, implementation review,
claim boundary and in-place path to a blinded multi-site study—is in
[`reports/aanca_expert_system_assessment_2026-08-21.md`](reports/aanca_expert_system_assessment_2026-08-21.md).

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`MVP_SCOPE.md`](MVP_SCOPE.md), [`STATUS.md`](STATUS.md), and
[`artifacts/mvp_demo/evidence.json`](artifacts/mvp_demo/evidence.json) for the
exact boundary and saved evidence.

### Presentation experience

The checked-in article is designed for both a research presentation and a close
technical reading. Its English narrative uses one centered 640 px editorial
column and a compact five-stage diagram of the benchmark. The findings section
contains only its title, seven registered questions, and their complete answers.
On motion-capable desktops, each answer resolves word by word and becomes a firm
scroll stop before the next question can enter from below. A single gesture cannot
skip multiple questions. On narrow screens, with reduced motion, or without GSAP,
all seven answers become an immediately visible static article. The comparison
atlas remains visible; the duplicate 36-entry numeric table sits in one optional
audit disclosure and remains keyboard-scrollable and filterable when opened. Its
rows become labelled two-column records on narrow screens.
The atlas uses ordinary-flow headings, so internal scrolling cannot place an axis
or group label over a data row. Repeated row rules are removed; whitespace and one
subtle boundary between hypothesis groups provide the remaining structure.
Results, method notes, tables, and study limits use open editorial surfaces with
thin rules instead of nested dashboard cards.

Each findings answer now states the practical conclusion first, reports the exact
registered evidence needed to support it, and closes with the relevant scope
boundary. In particular, retrieval performance is kept separate from downstream
classification, unavailable H6 comparisons are not treated as zero, and no ranking
result is presented as proof that a natural annotation is wrong.

The final article section identifies Natan Smogór as the author and developer,
summarises the work demonstrated in AANCA, and records the owner's educational
background without overstating it: Management and Artificial Intelligence at
Kozminski University is marked as current study, while the 80-hour Artificial
Intelligence programme at Uniwersytet Młodzieżowy is marked as completed in the
2024/2025 academic year. The section explicitly avoids implying institutional
endorsement of the project.

GSAP coordinates the directional header, logo reveal, section transitions, chart
marks, and cumulative method story. Three.js renders the conceptual source-patch
to review-queue scene in the hero. Responsive and reduced-motion layouts retain
the complete scientific content without depending on those enhancements.

Runtime work is visibility-aware: the WebGL render loop pauses outside the hero
and while the browser tab is hidden, high-density rendering is capped, and the
large QC image is lazily decoded with explicit dimensions. These optimisations
reduce background CPU/GPU work and layout shift without changing any evidence.

## Evidence at a glance

The values below are read from the checked-in evidence packages. PanNuke and
MoNuSAC controlled tests measure recovery of **injected label changes**. The NuCLS
table reports the external natural-disagreement evaluation.

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

### External NuCLS result

| Frozen outcome | Primary Unbiased Control result |
| --- | --- |
| Eligible evidence | 811 nuclei, 5 TCGA patient groups, 27 NP/P disagreements |
| Ranking | AP 0.073489 vs prevalence 0.033292; 4/41 disagreements at the 5% budget |
| Ranking decision | **Not supported**: 5%-precision-minus-prevalence CI [−0.030075, 0.154075] crossed zero |
| Downstream decision | **Not supported/adverse**: guided minus uncorrected macro F1 −0.014633, CI [−0.026683, −0.002415] |
| Meaning | The frozen method did not establish natural NP/P disagreement prioritisation or retrospective model improvement |

NuCLS P-truth is inferred pathologist consensus, not guaranteed biological truth.
The Evaluation sensitivity subset also failed its frozen ranking and downstream
rules and could not rescue the primary outcome.

### New-data MoNuSAC result

| Frozen outcome | Controlled external result |
| --- | --- |
| Eligible evidence | 29,610 development nuclei / 44 patients; 15,494 untouched test nuclei / 25 patients |
| Primary ranking | AP 0.658142; 1,035/1,481 injected changes found at the 5% budget |
| Ranking decision | **Passed**: precision minus exact matched random +0.142843, CI [+0.099181, +0.188491] |
| Downstream vs no review | Macro-F1 +0.005526, CI [-0.001506, +0.012833] — failed |
| Downstream vs matched random | Macro-F1 +0.000031, CI [-0.008692, +0.008486] — failed |
| Overall meaning | Better controlled-change triage on new images was established; safe downstream improvement was not |

This is a controlled-corruption test, not a natural-error study. It does not show
that a pathologist was wrong or that AANCA improves a model in real use.

### Bounded development search after the frozen test

| Development-only outcome | Selected result |
| --- | --- |
| Search extent | 240 ranking screens, 160 downstream screens and 12 full nested patient-group finalists |
| Selected policy | Multiscale 64/128 px ResNet-18 features; fixed hybrid ranking; 5% review; `flag_exclude` |
| Controlled-change retrieval | Precision 0.947966; exact-matched-random difference +0.403950, CI [+0.360560, +0.448903] |
| Downstream vs no review | Macro-F1 +0.043090, CI [+0.032553, +0.055065] — passed |
| Downstream vs matched random | Macro-F1 +0.054399, CI [+0.034938, +0.075507] — passed |
| Important-class safety | All registered recall lower bounds remained above the frozen -0.01 margin |
| Permitted claim | Significant improvement in nested controlled MoNuSAC development only |

Candidate selection never read the official MoNuSAC test outcome. The test had
already been disclosed and remains permanently unavailable as confirmation for this
candidate. Source labels are unchanged, and the frozen record cannot authorise
automatic exclusion on natural data.

### Incremental improvement of the current AANCA model

This repository improves the same AANCA implementation; it does not replace it with
a “v2”. The original-label audit can now use the existing fold-safe neighbour score
or the registered fixed hybrid, while preserving exact OOF and source-group
exclusion evidence. A new fail-closed retraining guard retains the uncorrected model
unless the lower 95% whole-group bootstrap bound for macro-F1 improvement is above
the registered minimum effect.

The post-outcome NuCLS analysis found that the neighbour score passed both strict
ranking conditions in the primary Unbiased Control subset, but failed in the
Evaluation sensitivity subset. It was therefore **not promoted to the default**.
The guard rejected both the saved audit-guided correction and the full-consensus
training candidate on the preserved evidence. This prevents demonstrated or
uncertain degradation; it does not prove real-world improvement.

The current implementation now also separates annotation-quality ranking from
expected downstream benefit. The quality queue can be quota-balanced by source
group, class, tissue, proposed transition and embedding diversity. The
model-improvement queue is unavailable until genuinely measured development
interventions produce nested group-cross-fitted utility estimates with a positive
lower bound. Once those inputs exist, its priority is the percentile annotation-
inconsistency score multiplied by the positive conservative utility lower bound;
without either input it fails closed. Multi-rater review can derive keep, soft-label, downweight, exclude and
strictly gated hard-change views without mutating source annotations. Candidate
training policies are compared only on independent development groups, and adoption
requires both a positive macro-F1 lower bound and registered recall non-degradation
for every important class.

Calibration and stability are similarly fail-closed: temperature scaling accepts
only newly collected expert development labels and is evaluated group-cross-fitted;
the persistence signal requires 3-5 group-safe models over multiple checkpoints.
An exact matched-random selection plan is validated stratum by stratum before a
blinded review package is built. These are software safeguards for the same AANCA,
not positive empirical results.

Recalculate the result with:

```text
uv run python scripts/analyze_nucls_current_model.py --format markdown
```

Exact values and claim boundaries are in
[`reports/nucls_current_aanca_improvement.md`](reports/nucls_current_aanca_improvement.md).
The complete prospective policy and implementation boundary are in
[`CURRENT_AANCA_SAFE_INTERVENTION.md`](CURRENT_AANCA_SAFE_INTERVENTION.md) and
[`configs/current_aanca_intervention_policy.yaml`](configs/current_aanca_intervention_policy.yaml).

### Completed frozen new-data test

A separate controlled external benchmark was frozen on the official MoNuSAC 2020
train/test release before computing its metrics. It uses patient-group-safe OOF
ranking on deterministically corrupted training labels and evaluates downstream
models once on the untouched official test patients. Two patient IDs found in both
official archives are excluded from development only. The primary comparison is the
balanced fold-safe neighbour queue against corrupted/no-review and exact
matched-random baselines, with simultaneous per-class recall protection.

The primary retrieval gate passed, but all three downstream or class-safety gates
failed; the registered overall decision is `not supported`. It cannot show that a
natural annotation or pathologist is wrong. The exact freeze is in
[`MONUSAC_TEST_PROTOCOL.md`](MONUSAC_TEST_PROTOCOL.md), the outcome is in
[`reports/monusac_current_aanca_external_results.md`](reports/monusac_current_aanca_external_results.md), and the machine configuration is
[`configs/monusac_current_aanca_external.yaml`](configs/monusac_current_aanca_external.yaml).
Run it only with the checksum-matching official archives:

```text
uv run python scripts/run_monusac_current_aanca.py --device auto
uv run python scripts/verify_monusac_external_validation.py
```

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
- **Fail-closed retraining:** reviewed-label candidates are applied only after a
  positive lower whole-group confidence bound on independent validation and no
  registered important-class recall breach; otherwise the uncorrected model remains
  selected.
- **Separate review objectives:** annotation inconsistency never substitutes for
  expected downstream benefit; the second queue is unavailable without measured,
  cross-fitted development utility.
- **Multi-rater uncertainty:** raw votes, ambiguity, soft labels, downweighting and
  abstention remain explicit; hard changes are opt-in and require at least two votes.
- **Balanced and matched review:** queue caps limit concentration, and an optional
  blinded comparator plan must match top and random cohorts 1:1 within every frozen
  stratum.
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

### 3. Hand the presentation to a reviewer

If the repository remains private, either grant the reviewer GitHub access or send
the complete `artifacts\mvp_demo` directory as one archive. The reviewer should
extract every file and open `index.html`; sending only the HTML file would omit the
QC image, machine-readable evidence and checksum manifest. Repository links inside
the article follow the repository visibility configured on GitHub.

~~~powershell
Compress-Archive -Path artifacts\mvp_demo -DestinationPath AANCA-presentation.zip
~~~

### 4. Run the deterministic software smoke path

The full research workflow uses Python 3.12 and
[`uv`](https://docs.astral.sh/uv/):

~~~powershell
uv sync --dev
uv run histo-audit doctor
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke
uv run histo-audit experiment smoke --config configs/smoke_zero.yaml
~~~

The smoke commands use `artifacts/smoke_runs` by default. This isolated registry
is intentionally separate from the checked-in historical primary ledger, whose
absolute Windows paths describe the original workstation and are not portable.
Use `--runs-root <clean-directory>` to select another caller-owned location.

`generate-synthetic` is safe to repeat: it regenerates the expected data in memory,
verifies every saved array and manifest field, and reports `verified_existing` when
the on-disk package is identical. It never overwrites a changed or partial package.

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
- Raw data, full run directories, embeddings and checkpoints remain outside Git.
  Licence-compatible OOF predictions, rankings and statistical arrays are published
  as checksum-bound release assets rather than committed to the source tree.
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
| `experiment` | Run maintained smoke, pilot, and primary workflows |
| `preregistration` | Freeze the analysis definition and verify its evidence |
| `audit` | Rank unmodified original labels using group-safe OOF evidence |
| `external` | Build blinded external-review packages when eligibility gates are satisfied |
| `report` | Build sourced Markdown and static HTML from strict metrics JSON |
| `demo` | Build, checksum-verify, or locally serve the static presentation MVP |

Real-data study commands are intentionally hard-gated. The NuCLS external study and
MoNuSAC controlled-external benchmark have run and are backed by checked-in
immutable evidence; a new confirmatory study, a PanNuke original-label audit and
newly recruited blinded expert review remain deferred. An implemented component is
not evidence that a stage has run.

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
│   ├── external_validation/  # frozen NuCLS and MoNuSAC external analyses
│   ├── experiment/           # scientific contracts, statistics, and maintained runners
│   ├── reporting/            # evidence-backed reports and figures
│   └── workflows/            # preregistration, study gates, and review workflows
├── configs/                  # smoke, pilot, primary, confirmatory, external
├── scripts/present_demo.py   # dependency-free verified local presentation launcher
├── scripts/verify_nucls_external_validation.py # independent NuCLS recalculation
├── scripts/verify_monusac_external_validation.py # independent MoNuSAC recalculation
├── scripts/run_monusac_current_aanca.py # frozen current-AANCA MoNuSAC execution
├── scripts/analyze_nucls_current_model.py # post-outcome current-model analysis
├── tests/                    # unit, integration, portability, and CLI tests
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
| [`DECISIONS.md`](DECISIONS.md) | Current binding rationale; full history remains in Git |
| [`reports/repository_maintenance_2026-08-21.md`](reports/repository_maintenance_2026-08-21.md) | Cleanup inventory, retention boundary, code consolidation and recoverable quarantine |
| [`PUBLIC_EVIDENCE.md`](PUBLIC_EVIDENCE.md) | Download and independently recalculate the released primary evidence |
| [`reports/nucls_external_validation_results.md`](reports/nucls_external_validation_results.md) | Frozen NuCLS design, exact external result and independent verification |
| [`reports/nucls_current_aanca_improvement.md`](reports/nucls_current_aanca_improvement.md) | Exploratory ranking candidates and fail-closed retraining decisions for the same AANCA model |
| [`MONUSAC_TEST_PROTOCOL.md`](MONUSAC_TEST_PROTOCOL.md) | Prospectively frozen controlled-external new-data protocol |
| [`reports/monusac_current_aanca_external_results.md`](reports/monusac_current_aanca_external_results.md) | Exact MoNuSAC result, claim boundary and independent evidence identities |
| [`AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md`](AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md) | Frozen bounded-search spaces, screens, nested evaluation and selection rule |
| [`reports/aanca_autoresearch_expanded_development_results.md`](reports/aanca_autoresearch_expanded_development_results.md) | Exact 400-screen development search and selected-candidate result |
| [`configs/aanca_selected_development_candidate.yaml`](configs/aanca_selected_development_candidate.yaml) | Checksum-frozen identity and policy of the selected development candidate |
| [`AANCA_MEASURED_UTILITY_PROTOCOL.md`](AANCA_MEASURED_UTILITY_PROTOCOL.md) | Cross-fitted measured-utility requirements for the model-improvement queue |
| [`AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md`](AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md) | One-shot new-cohort confirmation and real-use claim ladder |
| [`CURRENT_AANCA_SAFE_INTERVENTION.md`](CURRENT_AANCA_SAFE_INTERVENTION.md) | Implemented two-queue, multi-rater, calibration, stability and adoption safeguards for the same AANCA system |
| [`configs/current_aanca_intervention_policy.yaml`](configs/current_aanca_intervention_policy.yaml) | Machine-readable prospective policy; NuCLS is excluded from selection and fresh evidence is still required |
| [`NUCLS_EXTERNAL_VALIDATION_PREREGISTRATION.md`](NUCLS_EXTERNAL_VALIDATION_PREREGISTRATION.md) | Publicly frozen NuCLS multi-rater protocol |
| [`EXPERT_REVIEW_PROTOCOL.md`](EXPERT_REVIEW_PROTOCOL.md) | Prospective requirements for a future blinded natural-case expert review |
| [`PROSPECTIVE_WORKFLOW_PROTOCOL.md`](PROSPECTIVE_WORKFLOW_PROTOCOL.md) | Multi-site with/without-AANCA workflow and downstream evaluation required for real-use claims |
| [`MVP_SCOPE.md`](MVP_SCOPE.md) | Reduced presentation boundary and acceptance checks |
| [`DATASET_SETUP.md`](DATASET_SETUP.md) | PanNuke acquisition, integrity, QC, and licence gate |
| [`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md) | Non-diagnostic scope, scientific limits, and responsible language |
| [`references/references.bib`](references/references.bib) | Project bibliography, including PanNuke, NuCLS and MoNuSAC sources |

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
- Controlled OOF fold allocation used the pre-corruption reference labels. Those
  labels were not model inputs, but they are benchmark-only information and can
  make fold balance more favourable than in a real audit.
- The timestamped July freeze artifacts were created before the repository's first
  public commit on 19 August 2026. Their internal hashes preserve file identity,
  but public Git history is not independent proof of prospective preregistration.
- The public evidence release supports independent recalculation of saved H1-H7
  statistics and inspection of per-cell OOF/ranking data. It does not constitute a
  second independent model-training run or external replication.
- The completed NuCLS multi-rater evaluation contains only five patient groups and
  did not meet its frozen success conditions. It cannot support natural-error or
  downstream-improvement claims.
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
- `EXTERNAL_VALIDATION_COMPLETE` — completed NuCLS evaluation; primary claims not supported
- `DEMO_COMPLETE`

Explicitly deferred:

- a new prospectively registered confirmatory study;
- real original-label audit execution;
- execution of the published blinded natural-case protocol in
  [`EXPERT_REVIEW_PROTOCOL.md`](EXPERT_REVIEW_PROTOCOL.md);
- fresh development execution of the safe-intervention policy in
  [`CURRENT_AANCA_SAFE_INTERVENTION.md`](CURRENT_AANCA_SAFE_INTERVENTION.md);
- newly recruited blinded expert evaluation;
- broader external replication, prospective utility and clinical outcomes.

Accordingly, `CONFIRMATORY_COMPLETE`, natural/pathologist-error detection and
clinical utility are not claimed. `EXTERNAL_VALIDATION_COMPLETE` records a completed
null/adverse multi-rater evaluation, not efficacy.

## Development and validation

Before submitting a change:

~~~powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
.venv\Scripts\python.exe -m histo_audit demo verify --output-dir artifacts\mvp_demo
~~~

One `Scientific software` workflow verifies the dependency-free five-file package,
then installs the locked environment on Ubuntu and Windows and runs linting,
formatting, the complete test suite and the deterministic synthetic workflow in a
clean registry. It does not retrain the PanNuke primary study. The release described in
`REPRODUCIBILITY.md` supports independent result recalculation; a full image-to-result
replication still requires the licensed dataset.

Mandatory scientific invariants live in [`AGENTS.md`](AGENTS.md) and
[`SPEC.md`](SPEC.md). Material changes must update the corresponding evidence in
`STATUS.md` and rationale in `DECISIONS.md`. A failed mandatory gate stops
advancement.

### Workspace hygiene

Raw PanNuke files, representation caches and sealed run directories are local and
ignored by Git, but they are not disposable build cache. In particular, do **not**
run `git clean -fdX`: it would also target licensed raw data and the immutable run
lineage required to reproduce the accepted evidence.

Routine cleanup covers tool caches (`.mypy_cache`, `.playwright-cli`,
`.pytest_cache`, `.ruff_cache`, `__pycache__`), `dist`, non-canonical
`artifacts/mvp_demo_*` previews and browser-QA output other than the README hero.
The canonical five-file package remains in `artifacts/mvp_demo`. Raw data, reusable
embeddings, the accepted run, frozen authorities and evidence reports are preserved;
an older run may be removed only after proving that the accepted run contains every
relative payload path and after recording the decision in `DECISIONS.md`.

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

University research prototype. Presentation release by **Natan Smogór**, updated
20 August 2026.
