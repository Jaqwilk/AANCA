# Milestone Plan

The stop-and-fix rule applies: do not advance past a mandatory milestone while its acceptance checks fail. Each milestone ends with focused tests, lint, a functional command, artifact inspection, documentation updates, and Git diff review.

## Checkpoint on 2026-07-27

- M0 — **complete**.
- M1 — **complete**.
- M2 — **complete** for deterministic synthetic data and implemented/tested for real-data representations.
- M3 — **complete** for the controlled pipeline primitives.
- M4 — **complete**. Canonical sealed runs: `20260717T162925.902444Z_synthetic_smoke_5573505315` and `20260717T162948.870526Z_synthetic_smoke_zero_corruption_a4d5f87ca0`.
- M5 — **complete**. Acquisition, anomaly-safe full-release validation, the saved QC
  bundle and overlays, the complete nucleus manifest, exact/pHash/frozen-ResNet
  duplicate evidence, publication/representation hardening, and the then-current M5
  CLI/QA sequence all passed.
- M6 — **complete**. The eligible sealed run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0` passed independent integrity,
  terminology, OOF/restoration, final-fold privacy, raw-inventory, and report
  inspection gates. The two earlier sealed runs remain preserved and permanently
  ineligible for the recorded reasons.
- M7 — **complete**. The full five-cache PanNuke bundle, strict
  representation-independence record, exact study configs, full QA sequence, and both
  real functional readbacks passed. The immutable authority
  `artifacts/preregistrations/20260719T002902.432341Z` and both canonical frozen configs
  passed fresh-process verification before any primary or final-reference outcome.
- M8 — **in progress; primary complete, original confirmatory deferred**. The bounded
  recovery `20260727T133947.089370Z_pannuke_primary_orphan_recovery` reused and
  checksum-verified all 185 required cells, retained 37 preregistered optional skips,
  retrained zero cells, sealed without overwrite, and passed independent
  `PRIMARY_STUDY_COMPLETE` attestation. The interrupted source run remains unchanged,
  unsealed, read-only, and ineligible. The original 108-cell confirmatory study has
  not run.
- M9 — **APIs implemented / execution and human evidence pending**. Original-label
  ranking and blinded-package builders are tested; no real ranking, package, expert
  response, or external-validation result exists.

Formally, M0–M7 are complete: **8/10 milestones = 80%**, with **20%** of the
milestone plan remaining. The completion status is exactly
`PRIMARY_STUDY_COMPLETE`; `CONFIRMATORY_COMPLETE` is not claimed. The one permitted
Option-B recovery is complete and consumed. Because real primary values were
accidentally exposed by a read-only search at `2026-07-27T10:57:07Z`, every later
affected analysis must declare `outcomes_inspected=true` and be reported as
`amended_or_exploratory`. No exposed value may inform implementation, tuning,
selection, thresholds, exclusions, hypotheses, or resource-profile choices.

### Bounded execution rule for the remaining project

- The former failed-seal wait and finalization-successor implementation are historical
  evidence only and must not be re-entered.
- Development uses fast focused tests. Once the recovery receipt is fixed, run the
  complete mandatory gate sequence once against that exact receipt; do not repeat an
  unchanged passing command merely to create another log.
- Recovery preflight is read-only and creates no run. The real path contains exactly
  one source qualification, one allowlisted physical copy, one destination
  verification, one completion/seal, one integrity verification, and one stage
  attestation. It has no recursive entry point or automatic retry.
- Existing 185 checksum-manifested required cells are reused. Starting their training
  from zero is reserved only for a separately authorized future decision if the
  content-addressed recovery evidence fails; it is not a fallback inside this path.
- One failed real recovery stops execution and preserves its evidence. It cannot
  silently launch another recovery, primary, or reduced study.
- The original frozen confirmatory definition remains immutable and deferred. A
  measured execution audit found that its 180 CNN fold fits require at least about
  5.5 days, approximately 30 GiB of full checkpoints, and a resume capability that
  the current production runner does not legally expose.
- The bounded replacement path is a distinct
  `resource_bounded_confirmatory_v1` sensitivity/feasibility run: one fixed context
  CNN, three existing frozen ImageNet representations, seed 303, both frozen
  corruption cells, all three official rotations, five-fold group-safe OOF, at most
  four CNN epochs, 2,000 paired group bootstraps, and the unchanged restoration
  budget. It is selected from operational evidence only and is always
  `amended_or_exploratory`.
- This reduced profile is not represented as the originally preregistered
  confirmatory matrix. It omits the target-mask CNN, model seeds 304/305, and the
  unavailable pathology scenario; it uses a fixed cross-representation ensemble and
  adds the explicitly amended `cnn_context_minus_imagenet_context` comparison in the
  `model_family` family. None of those differences may be used to make an original
  confirmatory claim.
- This resource run must use a separate post-outcome authority, a new run directory,
  fail-closed capacity/RAM/cache checks, explicit checkpoint-successor semantics,
  and no automatic retry. It may finish technically with `completion_stage=null`;
  it does not satisfy or attest `CONFIRMATORY_COMPLETE`, does not unlock M9, and does
  not alter the original frozen config or `PRE_REGISTRATION.md`.

## M0 — Environment and safeguards

- **Objective:** establish reproducible hardware/software evidence and scientific guardrails.
- **Files:** root governance documents, `reports/environment_initial.txt`, `reports/hardware.json`, `.gitignore`, environment metadata.
- **Acceptance:** OS/Python/Git/GPU/RAM/disk/write access recorded; CUDA tensor/backward test attempted; no global settings changed.
- **Validation:** `python --version`, `nvidia-smi`, project write probe, environment report inspection.
- **Outputs:** initial status and hardware snapshot.
- **Risks:** unavailable CUDA-compatible Python/PyTorch, missing Git identity, low disk.

## M1 — Project environment and CLI foundation

- **Objective:** create `.venv`, compatible locked dependencies, configuration parsing, run registry, atomic writes, and `doctor`.
- **Files:** `pyproject.toml`, lock/export files, `src/histo_audit/{cli,config,doctor,utils}`, `configs/`, tests.
- **Acceptance:** editable install succeeds; `python -m histo_audit doctor` prints/saves required fields; CLI help works.
- **Validation:** `.venv\\Scripts\\python -m histo_audit doctor`, focused pytest, Ruff, type check.
- **Outputs:** doctor report and reproducible dependency record.
- **Risks:** Python/package incompatibility, CUDA wheel mismatch, Windows multiprocessing.

## M2 — Deterministic synthetic data and representations

- **Objective:** generate five-class grouped RGB patches, exact target masks, dynamic crops, highlighting, morphometrics, manifests, and duplicate helpers.
- **Files:** `src/histo_audit/data`, `representations`, configs, tests.
- **Acceptance:** deterministic IDs/arrays; valid unique manifest; exact target identity; group-safe split; duplicate results reproducible.
- **Validation:** dataset/duplicate tests and `python -m histo_audit data generate-synthetic`.
- **Outputs:** standalone `dataset.npz`, `manifest.json`, and `generation.json`; tracked runs additionally retain complete JSON/CSV source manifests and figures.
- **Risks:** absent Parquet engine, unstable image feature routines.

## M3 — Corruption, OOF, auditing, metrics, restoration

- **Objective:** implement immutable controlled corruption, group-safe OOF probabilities, all core risk methods, review metrics/statistics, and restoration evaluation.
- **Files:** `corruption`, `cross_validation`, `auditing`, `statistics`, `evaluation`, tests.
- **Acceptance:** exact deterministic corruption; no group leakage; one OOF vector/sample; same-group neighbours excluded; equal review budgets; untouched final reference labels.
- **Validation:** corruption/OOF/auditing/statistics/restoration tests, Ruff, type check.
- **Outputs:** corruption manifests, predictions, rankings, metric JSON.
- **Risks:** class scarcity within folds, optional Cleanlab API drift, degenerate 0% metrics.

## M4 — Synthetic end-to-end gate

- **Objective:** execute an atomic tracked smoke run and generate sourced Markdown/HTML reporting and figures.
- **Files:** experiment runner, reporting, `configs/smoke.yaml`, `artifacts/runs/registry.csv`, tests.
- **Acceptance:** CPU run succeeds from empty generated data; machine-readable metrics back every report value; required smoke figures exist; run is immutable.
- **Validation:** `python -m histo_audit experiment smoke`, full pytest/lint/type check, manual artifact inspection.
- **Outputs:** unique smoke run directory and report.
- **Risks:** runtime, nondeterminism, missing figure/report dependency.

## M5 — Provenance, acquisition, and real-data validation

- **Objective:** verify literature/provenance/licence, independently reconcile the
  reported PanNuke archive identities and ZIP safety, inspect the immutable release
  under the fixed anomaly-safe policy, build the eligibility-bearing nucleus
  manifest, and audit cross-fold duplicates.
- **Files:** `references/`, literature reports, `DATASET_SETUP.md`, data inspectors, duplicate reports.
- **Acceptance:** all of the following must hold: the official Warwick URL, locally
  evidenced CC BY-NC-SA 4.0 scope for `masks/`, the project's broader non-commercial
  research restriction, and required citations for both PanNuke works are recorded
  with their evidence status; a machine-readable
  manifest recomputes archive sizes/SHA-256 and records ZIP CRC/path-safety results,
  extracted fold layout, and citation metadata; raw archives/arrays remain
  unchanged; every patch is included in overlap/void QC; supplied background is not
  treated as a complement; void pixels remain unlabeled; cross-class overlap is
  counted without class arbitration; touching instances are retained with the fixed
  exclusion reason and removed from every primary/confirmatory eligible population;
  affected patches are flagged; complete per-fold/patch/instance counts and
  representative overlays are saved and reviewed; the nucleus manifest is complete;
  and exact, pHash, and frozen-ResNet cross-fold duplicate evidence covers the full
  release. No anomaly threshold may be chosen from release frequency or outcomes.
- **Validation:** execute the exact command sequence below, inspect each referenced
  artifact, and stop on the first failed gate.
- **Outputs:** acquisition/checksum and QC evidence, validation JSON/Markdown,
  overlays, an eligibility-bearing nucleus manifest, and complete duplicate-audit
  evidence.
- **Risks:** manual licence/access, unofficial mirror, missing WSI/patient metadata, disk.

### M5 execution addendum — official release acquired on 2026-07-17

- [x] Confirm that all three fold archives and extracted fold directories are present
  under `data/raw/pannuke`. At the initial local-presence checkpoint, the URL,
  licence, byte sizes, SHA-256 values, and reported CRC/path-safety execution were
  user-supplied evidence; the separate acquisition item below subsequently completed
  independent repository reconciliation.
- [x] Recompute and save the acquisition/checksum inventory, including exact archive
  byte sizes, SHA-256 values, ZIP member CRC results, rejected unsafe-path count, and
  raw-file/fold inventory. Reconcile it against the reported values in
  `DECISIONS.md` and preserve the source URL, exact locally evidenced CC BY-NC-SA 4.0
  scope, the project's non-commercial research restriction, and both required
  PanNuke citation records.
- [x] Make validation distinguish archive/shape corruption from release-level
  annotation anomalies. Inspect all patches, count cross-class overlap and
  supplied-background voids, and never relabel or repair the immutable raw masks.
  The canonical full-release command passed on all 7,901 patches without raw changes.
- [x] Persist fold-, patch-, pixel-, affected-instance, and affected-patch QC
  evidence. Record channel 5 only as the supplied background channel; retain pixels
  assigned to neither positive class nor supplied background as unlabeled voids. The
  saved bundle and representative overlays passed independent machine and visual QA.
- [x] Apply the fixed pre-freeze rule from `PRE_REGISTRATION.md`: retain every
  overlap-touching instance with `touches_cross_class_overlap`, mark it ineligible
  before primary/confirmatory analysis, and apply one identical eligibility mask to
  every primary cell and required confirmatory scenario. Retain and flag its source
  patch. Do not arbitrate a class at overlap pixels. The full manifest retains and
  identically excludes all 1,411 affected identities; independent reconciliation
  found zero eligibility divergence.
- [x] Add regression tests for ordinary masks, cross-class overlap,
  supplied-background voids, touching/non-touching instances, affected-patch flags,
  malformed shapes, and structurally invalid masks. Use no release- or
  outcome-tuned anomaly threshold; report complete counts. A historical pre-closure
  checkpoint passed 540 tests; later integrity changes require a new full QA result.
- [x] Inspect representative normal, overlap, void, and exclusion overlays; build
  the full nucleus manifest; and complete exact/pHash/frozen-ResNet cross-fold
  duplicate auditing over every validated patch. Similarity candidates remain
  review-only. The audit covered 7,901 patches and all 20,798,326 cross-fold pairs:
  zero exact pairs, 121 pHash candidates, and zero frozen-ResNet candidates at cosine
  similarity at least 0.995. No candidate changed data, eligibility, or a split.
- [x] Run and pass the exact M5 commands and QA sequence below, inspect the saved
  artifacts, update documentation from actual command output, and review the Git
  state. The current exact reruns passed; full pytest reported 604 passed, Ruff and
  mypy passed, the public evidence validator returned zero errors, and the raw
  release remained unchanged. The repository is entirely untracked, so
  `git diff --check` can check whitespace but cannot provide a meaningful baseline
  diff until an initial commit exists.

```powershell
$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path
.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root $env:PANNUKE_ROOT --max-samples-per-fold 100000 --max-overlay-patches 24
.venv\Scripts\python.exe -m histo_audit data build-manifest --project-root . --root $env:PANNUKE_ROOT --batch-rows 4096
.venv\Scripts\python.exe -m histo_audit data audit-duplicates --project-root . --root $env:PANNUKE_ROOT --embedding-device cuda
.venv\Scripts\python.exe -m pytest -q tests\test_pannuke_gate.py tests\test_duplicates.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy src
git diff --check
git status --short
```

M6 must not start until every acceptance item above succeeds, the machine-readable
evidence is saved and reconciled, representative overlays are reviewed, and the
full-release duplicate gate is complete. The raw PanNuke release remains immutable
throughout. If CUDA or the frozen weights are unavailable, the duplicate command
must record the blocker and M5 remains open; no signal or score is fabricated.

## M6 — Real PanNuke pilot

- **Objective:** run the declared all-class pilot through restoration/report without inspecting final-test outcomes for selection.
- **Files:** `configs/pilot.yaml`, representation extraction, pilot runner/report.
- **Acceptance:** documented subset; one outer split; 10% corruption; ImageNet/logistic OOF; self-confidence/Cleanlab/neighbours; restoration; inspected report.
- **Validation:** `python -m histo_audit experiment pilot`, tests/lint, artifact inspection.
- **Outputs:** immutable pilot run.
- **Risks:** dataset absence, weight access, CUDA/runtime, class/group scarcity.

### M6 completion evidence — 2026-07-18

- [x] Full current-worktree QA passed: 641 tests, Ruff lint, Ruff format over 117
  files, mypy over 72 source files, and `git diff --check`.
- [x] The real GPU pilot completed as immutable run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0` with 5,481 audit samples in 225
  source-patch groups, 814 development reference-validation samples in 25 groups,
  and exactly 548 controlled label changes.
- [x] Group-safe OOF covered every audit sample exactly once with zero group overlap;
  fold allocation used only the declared controlled-benchmark
  `pre_corruption_label`, while fitting, Cleanlab, risk scores, and neighbours used
  `observed_label`. The untouched final-reference fold was absent from tuning,
  representations, OOF, corruption, and pilot outcome evaluation.
- [x] Independent post-seal checks passed for all 36 files. Artifact-root SHA-256 is
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`;
  the disposition ledger contains only the two historical withdrawals and the new
  run is stage-eligible.
- [x] The report and machine-readable limitations use both `potentially inconsistent
  annotation` and `recommended for expert review`, and state that the output is not
  diagnostic. The raw 22-file inventory was rehashed after execution with unchanged
  SHA-256 `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`.

## M7 — Preregistration freeze

- **Objective:** resolve primary choices after pilot and before primary outcomes, then freeze/hash without overwrite.
- **Files:** `PRE_REGISTRATION.md`, `configs/primary_frozen.yaml`,
  `configs/confirmatory_frozen.yaml`, the shared freeze/hash bundle, and `STATUS.md`.
- **Acceptance:** complete method/seed/statistics/exclusion definition; timestamp, dataset/manifest/config/Git hashes; status updated.
- **Validation:** `python -m histo_audit preregistration freeze` and independent SHA-256 verification.
- **Outputs:** immutable frozen registration and amendment mechanism.
- **Risks:** unresolved pilot decisions or dirty unreproducible state.

## M8 — Primary and confirmatory studies

- **Objective:** execute frozen corruption/representation/classifier matrix, then predefined CNN/encoder/target/hybrid ablations and paired statistics.
- **Files:** primary/confirmatory configs and runners, cached embeddings, result/report artifacts.
- **Acceptance:** all preregistered cells succeed or are explicitly failed; ≥2,000 group bootstraps unless documented; no test tuning; fold rotation status explicit.
- **Validation:** `experiment primary`, `experiment confirmatory`, full QA and artifact reconciliation.
- **Outputs:** primary and confirmatory reports/rankings/figures.
- **Risks:** compute budget, pathology weight credentials, OOM, long runtime.

### Mandatory bounded interrupted-orphan recovery before M8 can advance

This operational gate supersedes the failed-seal-only Option-B wait and its
finalization-successor critical path. It changes no frozen scientific hypothesis,
estimand, configuration, group-safe OOF rule, final-reference membership, corruption
policy, exclusion, decision threshold, terminology rule, or completion-stage
definition.

- [x] Implement and pass focused synthetic recovery and fault-injection tests before
  touching the real orphan. Tests must prove read-only source handling, exact source
  identity and hash binding, 185/185 required-cell completeness, the 37 declared
  optional skips, explicit-copy allowlisting, restoration/statistics readback,
  interruption handling, and no call to training. Missing, changed, extra, ambiguous,
  or unverifiable input must fail closed.
- [x] Pass the full mandatory code and data gates on the exact source receipt:
  `pytest`, `ruff check .`, `ruff format --check .`, `mypy src`, the full semantic
  PanNuke validator, the recovery CLI functional preflight, and fresh-process
  integrity/readback checks. Record exact commands and terminal results in
  `STATUS.md` and `DECISIONS.md`.
- [x] Publish and independently verify exactly one dated technical amendment before
  the real invocation. It must bind the current authority chain, exact execution-source
  and semantic-config hashes, the orphan run ID and inventory, the host-interruption
  evidence, the read-only import boundary, and `outcomes_inspected=true` with
  `inspection_timestamp=2026-07-27T10:57:07Z`. It must classify the recovered analysis
  as `amended_or_exploratory` and state that all frozen scientific definitions remain
  unchanged.
- [x] Run exactly one real `experiment primary-orphan-recovery` invocation producing
  experiment `pannuke_primary_orphan_recovery`. It must preserve the orphan unchanged,
  verify every imported artifact before and after physical copy, reuse only the 185
  completed required cells and declared optional skips, calculate each genuinely
  missing finalization operation at most once, and write a new run directory with exact
  `retry_of_run_id` lineage.
- [x] Because an uncompressed second tree no longer fits on C:, the physical copy must
  use the tested streaming Windows WOF/LZX policy. Each independently created file is
  compressed exactly once before the next file, then rehashed through the retained
  object handle. A per-file free-space guard must retain the fixed safety margin.
  Hardlinks, source mutation, manifest-only references and compression retry are
  forbidden.
- [x] The recovery must contain no training path, training fallback, verifier bypass,
  hot patch, source overwrite, retroactive seal, recursive self-invocation, or automatic
  retry. A failed real invocation stops M8 and requires a separate recorded decision;
  it must not silently start another primary or recovery.
- [x] Advance only after the new run passes ordinary completion, no-overwrite seal,
  artifact-manifest reconciliation, integrity verification, registry readback,
  statistics/restoration attestations, independent stage eligibility, and explicit
  no-training evidence in a fresh process.

A successful recovery may satisfy `PRIMARY_STUDY_COMPLETE` only when all gates above
pass, but its affected analysis remains permanently labelled
`amended_or_exploratory`; it is never presented as the original unamended primary
analysis. The historical instruction to execute the unchanged frozen confirmatory
study next is superseded operationally by the bounded-execution rule at the top of
this document: the original study remains immutable and deferred, while the reduced
resource profile is non-claiming and cannot unlock M9. If recovery or confirmatory
qualification fails, M8 remains open and no later completion stage is claimed. Any
claim requiring genuinely outcome-independent confirmation must be deferred to
independent external validation; `EXTERNAL_VALIDATION_READY` requires an actually
generated blinded package, and `EXTERNAL_VALIDATION_COMPLETE` requires genuine expert
or multi-rater evidence.

This recovery is an operational repair, not a new scientific analysis or completion
stage. It does not by itself advance M9 or authorize inspection-driven scientific
changes.

## M9 — Exploratory and external validation readiness

- **Objective:** rank original labels without automatic changes and build a blinded, suggestion-free review package or responsible multi-rater package.
- **Files:** original audit/external modules, configs, review artifacts, reports.
- **Acceptance:** rankings use group-safe OOF; language is non-diagnostic; top/random samples are anonymised and blinded; no fake responses.
- **Validation:** `audit original`, `external build-review-package`, visual/manual package inspection.
- **Outputs:** rankings and `artifacts/review_packages`.
- **Risks:** expert availability, category mismatch, insufficient context.

## Operational amendment on 2026-07-30 - complete the unchanged original confirmatory

This append-only operational amendment supersedes the remaining-project
execution order in the 2026-07-27 bounded-execution section. It does not change
`SPEC.md`, frozen `PRE_REGISTRATION.md`, `configs/confirmatory_frozen.yaml`, any
scientific cell, model, representation, corruption condition, fold, seed,
estimand, threshold, exclusion, restoration budget, or terminology rule.

- Do not execute `resource_bounded_confirmatory_v1` as a prerequisite for M8.
  Its expected 11-16 hour cost would still produce
  `completion_stage=null`, cannot claim `CONFIRMATORY_COMPLETE`, and cannot
  unlock M9. Preserve its code and governance evidence as an optional
  `amended_or_exploratory` sensitivity path; no result is invented.
- After the current technical publication/authority/runtime chain is
  independently qualified, use it to authorize the unchanged original
  108-cell confirmatory study. The exact frozen config and semantic hashes must
  be reverified before execution.
- Add only operational checkpoint-successor support to the original runner:
  `fresh` or explicitly selected `successor_resume`, never autodiscovery. A
  successor uses a new run directory with exact `retry_of_run_id`, physically
  copies only allowlisted canonical per-cell/per-fold checkpoints, validates
  config/model/data/split/optimizer/AMP/RNG/early-stopping state, and continues
  only incomplete fits. It must not read predecessor OOF predictions, metrics,
  rankings, reports, or outcome values for selection or tuning.
- The predecessor remains immutable. No hardlink, manifest-only reference,
  overwrite, cleanup, adoption, fallback-to-training, or automatic retry is
  permitted. A stopped attempt requires a new exact successor plan and a new
  one-use external authority.
- Before any real invocation, pass focused interruption/resume tests at epoch,
  fold, and cell boundaries; deterministic uninterrupted-versus-resumed tests;
  malformed/missing/extra/wrong-cell/wrong-fold/wrong-config/wrong-split
  checkpoint tests; link/reparse/ADS/PID-reuse/singleton/no-overwrite tests;
  full pytest/Ruff/format/mypy; the full PanNuke validator; exact
  108/90-required/18-optional and 180-CNN-fit plan readback; CUDA/AMP/cache/RAM
  and sealed-disk-budget-plus-10-GiB preflight; lifecycle rehearsal; immutable
  execution-source verification; and exact argv/cwd/interpreter/environment
  readback.
- Run each long confirmatory attempt only through the qualified event-driven
  Windows supervisor. The supervisor waits through the process handle without
  Codex polling, prevents sleep only while the child runs, verifies the exact
  terminal seal/integrity artifacts, and wakes the exact saved Codex session
  once. A restart or ambiguous terminal state writes STOP and never relaunches
  science.
- Emit `CONFIRMATORY_COMPLETE` only after 90/90 required cells, all three
  rotations, at least 2,000 declared group bootstraps, restoration/statistics,
  no-overwrite seal, registry/integrity reconciliation, and positive post-seal
  stage attestation all pass. Until then formal status remains
  `PRIMARY_STUDY_COMPLETE`.
- Only a qualifying `CONFIRMATORY_COMPLETE` unlocks M9. Then close the recorded
  run-name, final-reference-group provenance, cache-authority,
  eligibility-versus-reviewer-manifest, stage-attestation, cohort-freeze, and
  100-top/100-random seed-707 contracts before executing `audit original` and
  building the real blinded review package.

The existing estimate remains at least about 5.5 days and historically 10-15
days for 180 CNN fits, with about 30 GiB of active single-copy checkpoints.
Capacity must be recomputed from the final sealed plan immediately before each
attempt. The user's broad permission authorizes implementation and QA, but a
real attempt still requires a new external one-use authority bound to the exact
then-existing source, plan, command, runtime, and supervisor hashes.

## Operational amendment on 2026-07-30 - use one sealed fresh-process execution capsule

This append-only amendment replaces only the implementation architecture of the
remaining original-confirmatory launch chain. It does not change `SPEC.md`,
frozen `PRE_REGISTRATION.md`, either frozen scientific config, PanNuke data,
the 108/90/18-cell plan, 180 CNN fits, group-safe OOF, final-reference
isolation, corruption semantics, metrics, restoration, statistics, thresholds,
or terminology.

- Stop developing the nested same-process semantic-guard/capability/lease
  chain. Preserve all rejected candidates as evidence. Do not create another
  semantic-guard revision merely to close a Python-level handle-receipt window.
  Windows Job ownership, singleton enforcement, sleep prevention, and
  `WaitForExit` belong to the external supervisor.
- Build a deterministic, content-addressed execution capsule, preferably a
  stored ZIP application with sorted safe relative paths, fixed metadata, no
  compression, and a canonical internal source manifest. Two independent
  builds from the same qualified allowlist must be byte-identical.
- Q binds the capsule SHA-256, internal manifest root, exact interpreter,
  frozen configs/data/split/plan, supervisor-v2 source and launcher, terminal
  contracts, and static environment. Per-attempt E binds the exact fresh or
  successor lineage, one launch nonce, complete command/environment, job,
  checkpoints, preflight, and no-retry disposition.
- Launch the capsule only as a fresh isolated process with an exact absolute
  interpreter and `-I -B`. Before any dataset, cache, model, checkpoint, or
  scientific artifact action, it reads and physically verifies Q/E, its own
  capsule identity, frozen inputs, plan, command, and complete observed
  environment. It then directly executes preflight and the canonical runner;
  there is no caller-supplied callback, plugin, raw contract, capability, or
  lease handoff.
- All project-owned runtime modules must originate from the capsule. Reject a
  project root, current working directory, user-site directory, mutable source
  checkout, or arbitrary import hook as a source of `histo_audit` code.
  Third-party and standard-library imports remain pinned by the existing
  environment/package evidence.
- The same exact capsule is launched later by the supervisor as a distinct
  `verify-preterminal` process. It validates Q/E, process-started evidence, and
  scientific terminal artifacts, then writes one O_EXCL preterminal pin and a
  canonical stdout summary. It cannot bind the supervisor terminal receipt,
  which does not yet exist.
- The supervisor then authenticates the verifier process and writes its own
  terminal receipt. After the one event-driven Codex wake, the same capsule's
  `verify-terminal` mode composes and revalidates the preterminal pin and
  supervisor terminal receipt. This removes future-hash cycles and never
  requires the scientific producer PID to remain alive.
- Create a new external supervisor-v2 release while preserving the current
  qualified release read-only. The new release and launcher pass explicit,
  complete, collision-safe environments: the existing Q/E supervisor mapping
  to the supervisor and the existing child mapping to both science and
  preterminal verifier. No ambient variable or second nonce is inherited.
  Authorization and launch/process/terminal receipts bind all environment
  hashes and observed readbacks.
- The capsule threat boundary includes filesystem/path/link/reparse/ADS races,
  source/config/data/command/environment/process substitution, stale or
  duplicate authority, callback/plugin/import injection, artifact overwrite,
  and restart/PID reuse. Arbitrary malicious self-modification by already
  trusted, sealed code inside the fresh isolated process is out of scope; such
  a threat cannot be soundly solved by comparing mutable Python objects to
  themselves. No untrusted code receives control before the direct runner
  entry.
- Before Q, require independent zero-P0/zero-P1 audits of the deterministic
  capsule builder, capsule entry modes, runner/checkpoint completion, Q/E
  readers, supervisor-v2, and two-phase terminal verifier; full live pytest,
  Ruff, format, mypy, compile; two-build reproducibility; ZIP/path/import
  provenance tests; lifecycle rehearsal; exact PanNuke validator with retained
  exit code; and sealed capacity plus 10 GiB.

The existing one-Q and per-attempt one-E/no-retry rules remain in force. This
amendment removes a redundant in-process control plane; it does not skip a
scientific or validation gate.
