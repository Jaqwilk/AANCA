<p align="center">
  <a href="https://aancastudy.org/"><img src="docs/assets/aanca-mark.svg" width="64" height="64" alt="AANCA four-tile mark"></a>
</p>

<h1 align="center">AANCA</h1>

<p align="center">
  <strong>Automated Auditing of Nucleus Class Annotations</strong><br>
  A reproducible framework for prioritising potentially inconsistent annotations
  for qualified expert review.
</p>

<p align="center">
  <a href="https://aancastudy.org/"><strong>Website</strong></a>
  · <a href="https://aancastudy.org/review/"><strong>Reviewer guide</strong></a>
  · <a href="#quick-start">Quick start</a>
  · <a href="#how-aanca-works">Method</a>
  · <a href="#current-conclusion">Evidence</a>
  · <a href="#reproducibility-levels">Reproduce</a>
  · <a href="#key-documents">Documentation</a>
</p>

<p align="center">
  <a href="https://github.com/Jaqwilk/AANCA/actions/workflows/scientific-software.yml"><img alt="Scientific software CI on Ubuntu and Windows" src="https://github.com/Jaqwilk/AANCA/actions/workflows/scientific-software.yml/badge.svg?branch=main"></a>
  <a href="pyproject.toml"><img alt="Reference Python version: 3.12" src="https://img.shields.io/badge/Python-3.12-5E6AD2"></a>
  <a href="ETHICS_AND_LIMITATIONS.md"><img alt="Use: non-clinical research" src="https://img.shields.io/badge/Use-non--clinical%20research-626874"></a>
  <a href="LICENSE"><img alt="Licence: limited evaluation permission" src="https://img.shields.io/badge/Licence-limited%20evaluation-626874"></a>
</p>

AANCA is a non-diagnostic research prototype. It ranks annotations for review,
never treats model disagreement as biological truth and never modifies source
annotations automatically. Outputs are described only as potentially inconsistent
annotations recommended for expert review.

## Start reviewing

The **[reviewer guide](https://aancastudy.org/review/)** connects methods, retained
negative results, provenance and verification scope. Download the small
[offline reviewer kit](https://github.com/Jaqwilk/AANCA/releases/tag/reviewer-kit-v1)
and, with Python 3.12, run `python -I scripts/review_project.py`. The default check
needs no third-party libraries, network, GPU or research environment. Optional
`--numeric` recalculates saved NuCLS/MoNuSAC evidence using NumPy only; a passing
check is not clinical validation. See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) for
download checksums, exact commands and the larger primary-evidence route.

- **Read the study:** [public article](https://aancastudy.org/) · [one-page brief](PROFESSOR_BRIEF.md). A browser is enough.
- **Inspect and verify evidence:** [reviewer guide](https://aancastudy.org/review/) · [offline kit](https://github.com/Jaqwilk/AANCA/releases/tag/reviewer-kit-v1). Local checks need Python 3.12; numeric checks additionally need NumPy.
- **Run the software:** [quick start](#quick-start) · [reproducibility instructions](REPRODUCIBILITY.md). Use Python 3.12, uv and the locked environment.

From the extracted reviewer kit:

~~~console
python -I scripts/review_project.py
~~~

<a id="install-and-run-the-portable-workflow"></a>

## Quick start

Requirements:

- Python `3.12`;
- [uv](https://docs.astral.sh/uv/);
- Git LFS for the released PUMA numeric evidence;
- a lawful local copy of any dataset used for a real-data re-execution.

~~~powershell
git clone https://github.com/Jaqwilk/AANCA.git
cd AANCA
git lfs pull
uv sync --frozen --dev

uv run histo-audit doctor
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke --runs-root artifacts/smoke_runs
~~~

The synthetic path validates software behaviour only. It is not medical or natural
annotation evidence.

The lock file resolves the reference Windows/NVIDIA environment with PyTorch CUDA
12.6 wheels. A different accelerator or CPU-only environment should use the official
PyTorch selector while preserving the project versions and scientific configs.

## How AANCA works

1. Preserve `pre_corruption_label`, `observed_label`, corruption metadata and the
   immutable source annotation as separate fields.
2. Split only by `group_id`, at least the complete source patch and stronger patient,
   WSI or case identifiers where the source provides them.
3. Produce model-based audit scores out of fold so a nucleus and its complete source
   group are absent from the model that scores it.
4. Combine OOF support for the observed label and fold-safe neighbourhood evidence into a
   fixed expert-review queue.
5. Compare the queue with exact equal-budget matched-random review.
6. Evaluate retrieval and downstream utility separately; a favourable ranking never
   substitutes for a favourable downstream result.

The primary scientific invariants are frozen in [`SPEC.md`](SPEC.md) and enforced in
code and tests.

<details>
<summary><strong>Released rankings, calibration and the selected candidate</strong></summary>

The maintained rankings use uncalibrated OOF probabilities. Cross-fitted temperature
calibration is an optional development capability requiring independent expert-review
evidence; it was not applied to the released rankings. A risk percentile is a ranking
factor, not a calibrated probability of annotation error.

The original-label CLI defaults to the reference self-confidence workflow, balanced
logistic fitting with `l2=0.01` and `k=7` neighbours. The selected later research
candidate is separately frozen in
[`configs/aanca_selected_development_candidate.yaml`](configs/aanca_selected_development_candidate.yaml):
multiscale 64/128 ResNet-18 features, a 60/40 confidence/neighbourhood hybrid,
`k=31`, unbalanced fitting and `l2=0.1`. Running the original-label CLI does not
implicitly load that selected candidate.

</details>

## Current conclusion

Under a score-before-reference public replication, the frozen AANCA ranking enriched
independent-expert disagreement relative to exact equal-budget matched-random review
on both RIVA and MIDOG++. A separate frozen NuCLS leave-one-pathologist-out
evaluation found enrichment for the only qualifying input annotator, `JP.1`. The
frozen current system also transferred to a new histopathology source under
**controlled label corruption** in PUMA. These are review-prioritisation and
controlled-transfer results, not adjudication of pathologist error or evidence of
prospective workflow benefit.

### Retained limits

- **PanNuke:** ranking evidence was positive, H4 downstream restoration was adverse.
  The accepted analysis is permanently `amended_or_exploratory` because outcomes
  were exposed during recovery.
- **NuCLS multi-rater:** the frozen ranking gate failed and guided correction was
  adverse. Natural-error and downstream-improvement claims were not supported.
- **MoNuSAC:** retrieval passed, but downstream and class-safety gates failed;
  action remained `retain_uncorrected`.
- **PUMA:** controlled-noise transfer does not establish natural/pathologist-error
  detection. Public Git history does not independently timestamp the freeze before
  results; every-class safeguards passed in only 1/9 post-confirmation stress scenarios.

The binding action for unreviewed natural data is `retain_uncorrected`.

<details>
<summary><strong>All evaluated studies: exact results and interpretation</strong></summary>

| Evaluation | Result | Responsible interpretation |
| --- | --- | --- |
| PanNuke primary controlled benchmark | `PRIMARY_STUDY_COMPLETE`; ranking evidence was positive, H4 downstream restoration was adverse | The accepted analysis is permanently `amended_or_exploratory` because outcomes were exposed during recovery |
| NuCLS genuine multi-rater disagreement | `EXTERNAL_VALIDATION_COMPLETE`; frozen ranking gate failed and guided correction changed macro-F1 by `-0.014633`, 95% CI `[-0.026683, -0.002415]` | Natural-error and downstream-improvement claims were not supported |
| NuCLS independent-pathologist LOO ranking | At 5%, `JP.1` precision was `0.333333` versus `0.215111` exact matched random; difference `+0.118222`, patient-bootstrap 95% CI `[+0.040000, +0.257143]` | Supports disagreement enrichment only for one junior pathologist across five patients; no multi-pathologist pooling, adjudicated-error or downstream claim |
| RIVA × MIDOG++ public independent-pathologist replication | RIVA precision `0.476357` versus `0.426900`, difference `+0.049457`, 95% CI `[+0.014037, +0.091632]`; MIDOG++ precision `0.325967` versus `0.249392`, difference `+0.076575`, 95% CI `[+0.021352, +0.138159]` | Both frozen dataset gates and the cross-dataset replication rule passed; this supports disagreement enrichment for review, not adjudicated error, downstream benefit or clinical utility |
| MoNuSAC controlled external benchmark | Retrieval precision `0.698852` versus `0.556009` matched random; downstream difference `+0.005526`, 95% CI `[-0.001506, +0.012833]` | Retrieval passed, but downstream and class-safety gates failed; action remained `retain_uncorrected` |
| PUMA internally frozen new-source controlled confirmation | All seven internally pre-specified gates passed on 62 held-out case/ROI groups | Supports controlled-noise transfer, not natural/pathologist-error detection; public Git history does not independently timestamp the freeze before results |
| PUMA post-confirmation realism stress | Positive aggregate downstream lower bounds in 9/9 scenarios; every class safeguard passed in only 1/9 | Useful robustness evidence and a binding class-safety warning; exploratory only |
| PUMA observed-label fold sensitivity | All seven sensitivity gates passed with audit-time labels; candidate unchanged | Shows the controlled PUMA result did not depend on clean labels for fold allocation; not independent confirmation |
| Prospective natural-case workflow | Not executed | `CONFIRMATORY_COMPLETE`, clinical utility and automatic natural-data intervention are not claimed |

</details>

<details>
<summary><strong>PUMA: exact endpoint, intervention and freeze chronology</strong></summary>

The exact frozen PUMA endpoint was:

- review precision `0.537739` versus `0.214379` matched random;
- precision difference `+0.323359`, 95% CI `[+0.259251, +0.384944]`;
- candidate macro-F1 `0.646310` versus `0.639884` unchanged and `0.638243`
  matched random;
- candidate minus unchanged `+0.006426`, 95% CI `[+0.003657, +0.009365]`;
- candidate minus matched random `+0.008067`, 95% CI
  `[+0.004093, +0.011947]`.

The 144/62 development/final partition is an AANCA-defined split of the 206 public
PUMA ROIs. It is not the official hidden PUMA challenge test set.

The downstream intervention was `flag_exclude`: the highest-ranked 5% of training
instances were omitted from downstream training. They were not reviewed, corrected
or automatically relabelled by an expert. Source annotations remained unchanged.

These values are read from
[`artifacts/puma_new_data_confirmation/results.json`](artifacts/puma_new_data_confirmation/results.json)
and checked by the PUMA evidence-readback script
[`scripts/verify_puma_new_data_confirmation.py`](scripts/verify_puma_new_data_confirmation.py).
Source PUMA annotations remained unchanged.

The PUMA protocol, configuration and result first entered public Git history together
in commit `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. Internal authorities record
the intended pre-outcome ordering, but GitHub is not independent proof of that timing.
The PUMA verifier is a project-coupled evidence-readback script that recomputes
metrics from saved predictions but does not retrain all 44 models. It is not
third-party validation. These are explicit reproducibility limits, not missing
positive results.

</details>

See [PUBLIC_EVIDENCE.md](PUBLIC_EVIDENCE.md) for release identities and [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md) for the complete claim boundary.

## What the project can claim

Current evidence supports the following statements:

- group-safe AANCA queues retrieve injected class-label changes more efficiently
  than equal-budget matched random review;
- the frozen selected candidate transferred to previously unused PUMA images under
  the registered controlled-noise experiment;
- the score-before-reference RIVA and MIDOG++ evaluations each enriched
  independent-expert disagreement relative to exact equal-budget matched-random
  review, satisfying the frozen cross-dataset replication rule;
- on the eligible NuCLS `JP.1` cohort, the frozen global risk ranking enriched
  leave-one-pathologist-out consensus disagreement relative to exact matched random
  review;
- in that PUMA experiment the intervention improved downstream macro-F1 over both
  unchanged labels and matched-random intervention with positive whole-group 95%
  intervals;
- the complete software, evidence and claim boundary are inspectable and
  checksum-verifiable.

Current evidence does **not** support these statements:

- that AANCA proves a naturally occurring annotation is wrong;
- that it proves a pathologist made an error or identifies biological truth;
- that the intervention is uniformly safe for every class and realistic error
  mechanism;
- that it improves review time, expert agreement, patient outcomes or clinical
  operations;
- that natural labels may be automatically excluded, relabelled or overwritten.

The binding action for unreviewed natural data is `retain_uncorrected`.

## Reproducibility levels

The repository deliberately separates three different tasks.

<a id="verify-the-published-presentation"></a>

<details>
<summary><strong>Verify the published presentation</strong></summary>

~~~powershell
python scripts/present_demo.py --verify-only
python scripts/verify_deployed_presentation.py --url https://aancastudy.org/
~~~

This verifies the closed article package and its machine-readable current-evidence
summary. The second command additionally checks all thirteen served files and
requires the deployed manifest to match the local release. Both are read-only.
Neither command recalculates a scientific result.

</details>

<a id="recalculate-released-evidence"></a>

<details>
<summary><strong>Recalculate released evidence</strong></summary>

~~~powershell
uv run python scripts/verify_primary_evidence.py PATH/TO/aanca-primary-evidence-v1
uv run python scripts/verify_nucls_external_validation.py
uv run python scripts/verify_monusac_external_validation.py
uv run python scripts/verify_nucls_independent_pathologist_validation.py
uv run python scripts/run_public_pathologist_replication.py verify --dataset riva
uv run python scripts/run_public_pathologist_replication.py verify --dataset midogpp
~~~

The primary verifier uses the checksum-bound `primary-evidence-v1` release and does
not import the analysis package. NuCLS and MoNuSAC verifiers independently recalculate
their saved numeric evidence. The PUMA script is deliberately described more narrowly
as an evidence readback: it imports maintained AANCA helpers, consumes saved
predictions and convergence flags, and does not independently retrain the 44 models.

The requirements and write behaviour of every verifier are listed in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md#verification-command-requirements).
The raw-dependent checks are separate from the released-array commands above:

~~~powershell
uv run python scripts/verify_puma_new_data_confirmation.py --output artifacts/qa/puma-verification.json
uv run python scripts/verify_nucls_supervised_qc_feasibility.py --output artifacts/qa/nucls-qc-verification.json --report artifacts/qa/nucls-qc-verification.md
~~~

These require the lawfully obtained PUMA source archives and NuCLS single-rater
SQLite database respectively. The PUMA check rebuilds the official source manifest
before reading saved predictions; the NuCLS check reassesses reference feasibility.

</details>

<a id="re-execute-from-images"></a>

<details>
<summary><strong>Re-execute from images</strong></summary>

Full image-to-result re-execution additionally requires the official dataset files,
their licences, sufficient compute and the governed acquisition checks in
[`DATASET_SETUP.md`](DATASET_SETUP.md). Raw images are not redistributed by this
repository. Some historical fold checkpoints were not retained, so reproduction is
a governed re-execution, not reuse of every original byte.

The selected-candidate convergence command performs the full nested development
evaluation, including model fitting. It additionally requires the ignored local
MoNuSAC selection authority and ledger; those inputs are not supplied by Git LFS:

~~~powershell
uv run python scripts/verify_aanca_selected_candidate.py --output artifacts/qa/selected-candidate-reexecution.json
~~~

It verifies an already frozen candidate and must not be used to tune on an opened
final-reference dataset. This command is not a released-array-only check.

</details>

## Repository layout

<details>
<summary><strong>Package layout and public repository boundary</strong></summary>

~~~text
AANCA/
├── src/histo_audit/          # maintained package
│   ├── auditing/             # review scores and two-queue policy
│   ├── cross_validation/     # group-safe OOF prediction
│   ├── evaluation/           # restoration and downstream utility
│   ├── external_validation/  # NuCLS and MoNuSAC analyses
│   ├── research/             # bounded candidate search and PUMA confirmation
│   ├── representations/      # image, morphology and embedding features
│   └── workflows/            # gates, preregistration and review workflow
├── configs/                  # frozen and portable study definitions
├── scripts/                  # launchers, runners and scoped verification scripts
├── tests/                    # unit, integration, CLI and portability tests
├── artifacts/                # compact published evidence and static article
├── reports/                  # human-readable results and provenance
├── references/               # verified bibliography
├── data/README.md            # ignored local data layout
└── *.md                      # scientific governance and handoff documents
~~~

### Public repository boundary

The current Git tree retains only material with an active scientific, engineering or
presentation role:

- maintained Python source, tests, dependency lock and CI;
- frozen protocols, configs, decisions and status records;
- compact reports, manifests and result authorities;
- Git LFS numeric arrays required by the PUMA evidence readback;
- the checksum-verifiable static presentation and its checksum-bound assets.

It excludes raw/licensed datasets, local virtual environments, reusable embeddings,
full run workspaces, model caches, superseded previews, browser-test output and
temporary cleanup files. Empty artifact placeholders were replaced by
[`data/README.md`](data/README.md); maintained commands create output directories as
needed.

Do not run `git clean -fdX` in a research workspace: ignored raw data and accepted
local run lineage are not disposable caches.

</details>

## Current stage and next phase

<details>
<summary><strong>Execution stages and the separate AANCA V2 programme</strong></summary>

Completed vocabulary stages:

- `PIPELINE_COMPLETE`;
- `PILOT_COMPLETE`;
- `PRE_REGISTRATION_FROZEN`;
- `PRIMARY_STUDY_COMPLETE`;
- `EXTERNAL_VALIDATION_COMPLETE`;
- `DEMO_COMPLETE`.

`CONFIRMATORY_COMPLETE` has not been reached. Completion records that a governed
evaluation ran and its evidence was preserved; it does not mean every result was
favourable.

The standalone **AANCA V2 research phase** is `INITIALISED`. It is developed as a
separate project without rewriting or upgrading any V1 result. Its access-controlled
preparation repository is
[`Jaqwilk/AANCA-V2`](https://github.com/Jaqwilk/AANCA-V2). Its promotion path is:

1. recruit new, independent blinded pathologists and preserve consensus,
   disagreement, ambiguity, abstention and insufficient-context outcomes;
2. develop a measured-utility queue only inside nested patient/WSI-group
   cross-fitting;
3. freeze one representation, queue, intervention, review budget and class-safety
   policy before inspecting new confirmation outcomes;
4. run one-shot untouched patient/WSI external confirmation;
5. compare multi-site review with and without AANCA prospectively.

Ranking, downstream confidence intervals, every-class safety, convergence and
workflow utility must pass together before any realistic natural-case improvement
claim. The detailed promotion contract is in [`NEXT_PHASE.md`](NEXT_PHASE.md).

</details>

## Key documents

| Document | Purpose |
| --- | --- |
| [`SPEC.md`](SPEC.md) | Frozen terminology, hypotheses, leakage rules and completion vocabulary |
| [`PLAN.md`](PLAN.md) | Milestones, gates and executed/deferred work |
| [`STATUS.md`](STATUS.md) | Current evidence, commands and handoff |
| [`DECISIONS.md`](DECISIONS.md) | Binding scientific and engineering rationale |
| [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) | Frozen primary and confirmatory analysis definitions |

<details>
<summary><strong>Evidence, methods and the next research phase</strong></summary>

| Document | Purpose |
| --- | --- |
| [`PUBLIC_EVIDENCE.md`](PUBLIC_EVIDENCE.md) | Primary evidence release and independent recalculation |
| [`PROFESSOR_BRIEF.md`](PROFESSOR_BRIEF.md) | One-page problem, method, strongest result, negative evidence and next experiment |
| [`FINAL_READINESS_REPORT.md`](FINAL_READINESS_REPORT.md) | Final professor-facing evidence and release audit |
| [`reports/v1_remediation_2026-10-02.md`](reports/v1_remediation_2026-10-02.md) | Completed V1 fixes, repeat audit and live-publication verification |
| [`reports/project_audit_2026-10-02.md`](reports/project_audit_2026-10-02.md) | Historical findings before the V1 fixes |
| [`AANCA_BRAND_SYSTEM.md`](AANCA_BRAND_SYSTEM.md) | Existing visual identity and scientific-language rules |
| [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Author contribution and AI-assistance disclosure |
| [`reports/aanca_internal_technical_assessment_2026-08-21.md`](reports/aanca_internal_technical_assessment_2026-08-21.md) | Internal evidence assessment; explicitly not external peer review |
| [`reports/nucls_external_validation_results.md`](reports/nucls_external_validation_results.md) | Genuine multi-rater result and boundary |
| [`reports/monusac_current_aanca_external_results.md`](reports/monusac_current_aanca_external_results.md) | Controlled MoNuSAC result |
| [`reports/puma_new_data_confirmation_results.md`](reports/puma_new_data_confirmation_results.md) | Frozen PUMA confirmation |
| [`reports/puma_realism_stress_results.md`](reports/puma_realism_stress_results.md) | Post-confirmation realism and class-safety stress |
| [`reports/puma_audit_time_label_sensitivity_results.md`](reports/puma_audit_time_label_sensitivity_results.md) | Audit-time-label sensitivity |
| [`PUBLIC_INDEPENDENT_PATHOLOGIST_REPLICATION_PROTOCOL.md`](PUBLIC_INDEPENDENT_PATHOLOGIST_REPLICATION_PROTOCOL.md) | Frozen RIVA and MIDOG++ score-before-reference protocol |
| [`reports/public_independent_pathologist_replication_results.md`](reports/public_independent_pathologist_replication_results.md) | Cross-dataset independent-expert disagreement result |
| [`CURRENT_AANCA_SAFE_INTERVENTION.md`](CURRENT_AANCA_SAFE_INTERVENTION.md) | Current intervention safeguards and fail-closed action |
| [`EXPERT_REVIEW_PROTOCOL.md`](EXPERT_REVIEW_PROTOCOL.md) | Required natural-case blinded review |
| [`PROSPECTIVE_WORKFLOW_PROTOCOL.md`](PROSPECTIVE_WORKFLOW_PROTOCOL.md) | Required with/without-AANCA multi-site workflow |
| [`NEXT_PHASE.md`](NEXT_PHASE.md) | Presentation-ready AANCA v2 evidence programme |
| [`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md) | Responsible-use and claim boundary |

</details>

Read the first five documents before changing scientific code or claims.

## Validation gates

<details>
<summary><strong>Run the complete software and evidence gates</strong></summary>

Every material change must pass:

~~~powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
python scripts/present_demo.py --verify-only
uv run python -I scripts/verify_professor_release.py
~~~

The maintained GitHub workflow runs the locked checks on Ubuntu and Windows plus the
deterministic synthetic workflow. A failed mandatory gate stops advancement.

</details>

## Data terms, licence and citation

Dataset files and pretrained weights retain their own licences and access terms.
PanNuke binaries are not distributed here. The local evidence applies
CC BY-NC-SA 4.0 specifically to the recorded PanNuke `masks/` directory; it does not
establish identical terms for every source file. PUMA is recorded under its official
CC0 authority. Review each source before use.

The project code and original documentation are covered by the explicit
all-rights-reserved [`LICENSE`](LICENSE), with a limited grant for local copying and
execution solely for non-clinical scientific evaluation, peer review and verification.
Other rights are reserved; this is not an open-source licence. It does not apply to
datasets, pretrained weights, dependencies, logos or other third-party materials.
Their separate terms remain controlling.

The verified project bibliography is
[`references/references.bib`](references/references.bib).
Machine-readable citation metadata are available in [`CITATION.cff`](CITATION.cff).
No DOI has been assigned yet; DOI publication requires an owner-controlled archival
release, for example through Zenodo.

## Author

Research direction, review and final scientific responsibility: **Natan Smogór**.
AI-assisted tools supported implementation, testing, orchestration, documentation
and presentation; they supplied no expert labels and are not independent validators.
See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). Updated 3 October 2026.
