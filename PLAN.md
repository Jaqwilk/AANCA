# AANCA plan

This is the active milestone plan. Historical capsule, authority and recovery plans
are preserved in Git tag `pre-audit-simplification-2026-08-20`; they are not active
public workflows.

## Scientific objective

Evaluate whether group-safe OOF audit scores can prioritise controlled, injected
nucleus-label changes for expert review. Keep source annotations immutable and keep
all conclusions non-diagnostic.

## Binding rules

Every milestone must preserve the following:

1. split by `group_id`, at least source patch;
2. keep the final reference fold untouched and unavailable for selection;
3. compute primary model-based scores out of fold;
4. keep pre-corruption, observed and injected-event states separate;
5. exclude circularity-risk instance-dependent comparisons;
6. report unavailable, adverse and failed outcomes without substitution;
7. advance only through status names defined in `SPEC.md`.

## Completed milestones

### M1 — Repository and deterministic synthetic core

Status: complete (`PIPELINE_COMPLETE`)

- deterministic five-class synthetic dataset;
- immutable controlled corruption;
- group-safe OOF predictions and audit scores;
- fixed-budget review metrics and controlled restoration;
- tracked artifacts and machine-readable reports.

Acceptance evidence: the public synthetic smoke command runs from a clean registry
without PanNuke, GPU or workstation-specific paths. Repeating deterministic data
generation verifies and reuses identical outputs without overwriting changed files.

### M2 — PanNuke acquisition and semantic validation

Status: complete

- official-release acquisition remains explicit and offline;
- raw hashes, folds, shapes, channels and class mapping are validated;
- cross-class overlaps, void pixels and affected instances are reported;
- no automatic source-mask or source-label correction is permitted.

### M3 — Representation and duplicate controls

Status: complete

- deterministic crops and target identity checks;
- engineered and ImageNet-frozen features with provenance;
- duplicate candidates remain review-only;
- instance-dependent corruption/auditor feature independence is hash-bound.

### M4 — Pilot

Status: complete (`PILOT_COMPLETE`)

- development-only pilot executed;
- pilot-derived parameters were frozen before primary outcome interpretation;
- final reference data remained unavailable.

### M5 — Preregistration freeze

Status: complete (`PRE_REGISTRATION_FROZEN`)

- hypotheses H1-H7, comparison direction, metrics, budgets, seeds and bootstrap
  procedure were frozen;
- public-history limitation is disclosed: the July freeze was not independently
  timestamped and the accepted analysis is `amended_or_exploratory`.

### M6 — Primary frozen-feature study

Status: complete (`PRIMARY_STUDY_COMPLETE`)

- 185 required cells completed;
- 37 optional pathology-encoder cells were unavailable under the frozen gate;
- 33 registered numeric comparisons and three unavailable H6 entries retained;
- H4 was adverse and remains prominent;
- no claim of natural-error detection or downstream improvement is made.

### M7 — Public article and evidence

Status: complete (`DEMO_COMPLETE`)

- English long-form article with checksum-verifiable five-file package;
- all H1-H7 findings, H4, QC, limitations and author section retained;
- full saved statistics, H4, OOF and rankings published in
  `primary-evidence-v1`;
- a standalone evidence recalculator that does not import the primary analysis
  package recalculates all available comparisons and H4; this is not third-party
  validation.

## Completed engineering milestone

### M8 — Portable, maintainable public repository

Status: complete

Required gates:

- remove unexecuted capsule/authority/resource-controller orchestration;
- keep the scientific core and evidence readback intact;
- remove local PanNuke and Windows-path assumptions from CI;
- run the maintained full test suite on Ubuntu and Windows;
- run lint, formatting and synthetic smoke on both systems;
- keep the public article package identical to the deployed Hostinger site;
- document every public evidence asset and checksum.

Completion requires a green `Scientific software` workflow on both operating
systems after the simplified source is merged to `main`.

Acceptance evidence: workflow run `32389776361` passed the full maintained suite,
lint, formatting and synthetic smoke on Ubuntu and Windows; the verified static
package was deployed byte-identically.

### M11 — External multi-rater validation

Status: complete (`EXTERNAL_VALIDATION_COMPLETE`)

- the NuCLS protocol and configuration were publicly frozen before outcome-table
  download;
- exact official NP/P anchor pairing and TCGA-patient grouping were used;
- the Unbiased Control subset was primary and Evaluation subset secondary;
- both ranking and downstream success rules were evaluated without post-outcome
  tuning;
- portable source inventories, numeric evidence and a standalone checked-in
  recalculator are
  published;
- the primary result is retained as null/adverse: it does not establish natural
  disagreement prioritisation or downstream improvement.

Completion means the genuine multi-rater evaluation ran and was preserved. It does
not mean that AANCA detected biological truth or that a pathologist was wrong.

### M12 — Incremental safety improvement of the current AANCA model

Status: complete as engineering and post-outcome exploratory analysis; no new
scientific completion stage claimed

- keep the frozen NuCLS result and existing AANCA project unchanged as evidence;
- expose the already implemented fold-safe neighbour and fixed-hybrid scores in the
  current original-label audit workflow;
- persist exact neighbour identities, distances and excluded-group provenance;
- require a positive lower whole-group confidence bound before a reviewed-label
  retraining candidate may replace the uncorrected model;
- re-evaluate all declared ranking candidates on both saved NuCLS subsets;
- do not promote a candidate selected after outcome inspection unless it passes the
  declared rule on every subset and is subsequently frozen on fresh data.

The neighbour score passed the primary Unbiased Control ranking gates but failed the
Evaluation sensitivity gates, so it remains an available exploratory strategy rather
than a new default. The fail-closed guard retained the uncorrected model for every
saved correction candidate. This improves runtime safety, not the historical result.

### M14 — Safe review-to-training policy for the current AANCA implementation

Status: complete as engineering; prospective scientific execution remains open and
no new completion stage is claimed

- keep NuCLS permanently unavailable for method, threshold, encoder, calibration or
  review-budget selection;
- maintain separate annotation-quality and model-improvement queues;
- leave the model-improvement queue unavailable without measured, nested
  group-cross-fitted development utility and a positive lower bound;
- balance review selection by strongest source group, class, tissue, proposed
  transition and optional embedding diversity;
- enforce an exact 1:1 matched-random control plan before blinded package creation;
- preserve raw reviewer votes, ambiguity and abstention; keep hard changes disabled
  by default and require at least two independent votes when prospectively enabled;
- compare unchanged, hard, soft, downweighted and abstention-aware training policies
  only on disjoint development groups;
- require a positive macro-F1 group-bootstrap lower bound and registered
  non-degradation for every important class before adoption;
- evaluate calibration only with new expert development labels and stability only
  from 3-5 group-safe model histories;
- retain pathology-encoder availability and provenance gates; do not select an
  encoder on a final test.

Implementation and exact unresolved evidence requirements are recorded in
[`CURRENT_AANCA_SAFE_INTERVENTION.md`](CURRENT_AANCA_SAFE_INTERVENTION.md) and
[`configs/current_aanca_intervention_policy.yaml`](configs/current_aanca_intervention_policy.yaml).

### M15 — New-data controlled external benchmark on MoNuSAC

Status: completed; frozen overall claim not supported

- pin the official MoNuSAC train and test archives by SHA-256;
- exclude the two patient identifiers appearing in both archives from development
  only and leave the official test intact;
- create deterministic 10% symmetric corruption only in development;
- compute all audit scores with five-fold patient-group-safe OOF predictions;
- compare the earlier global queue with balanced self-confidence, fold-safe neighbour
  and fixed-hybrid queues at the same 5% review budget;
- compare the primary balanced neighbour queue against exact matched-random review;
- train every downstream condition on development and evaluate once on untouched test
  patients;
- require positive whole-patient confidence bounds against both uncorrected and
  matched-random baselines plus registered per-class recall non-degradation;
- retain the result without post-test tuning or natural-error claims.

Executed result:

- the primary 5% neighbour queue passed the exact-matched-random retrieval gate;
- its downstream macro-F1 point estimate was positive versus corrupted/no review,
  but the whole-patient interval crossed zero;
- it did not exceed mean exact-matched-random restoration and the registered
  important-class recall safeguard failed;
- all four conditions were required, so the final action is `retain_uncorrected`;
- the checked-in standalone evidence recalculator recomputes the metrics and all frozen
  bootstrap decisions without importing AANCA analysis code.

The frozen protocol is
[`MONUSAC_TEST_PROTOCOL.md`](MONUSAC_TEST_PROTOCOL.md); the machine configuration is
[`configs/monusac_current_aanca_external.yaml`](configs/monusac_current_aanca_external.yaml),
and the result is
[`reports/monusac_current_aanca_external_results.md`](reports/monusac_current_aanca_external_results.md).

### M16 — Bounded autoresearch development search

Status: completed as controlled development; no external or natural-error status claimed

- keep the official MoNuSAC test permanently unavailable to candidate generation,
  filtering, selection and promotion;
- use all 44 eligible official training patients through nested patient-group-safe
  development evaluation;
- screen the prospectively declared ranking, representation, budget, intervention,
  regularisation and class-weight combinations under fixed evaluator code;
- retain an append-only keep/discard/fail-closed ledger and exact authority hashes;
- advance only 12 declared finalists to full four-seed downstream evaluation;
- require positive lower confidence bounds against both unchanged and exact
  matched-random baselines, positive seed directions and simultaneous important-class
  recall safety;
- freeze one simplest passing candidate without authorising natural-data mutation.

Executed result:

- 240 ranking and 160 downstream screens completed; 12 finalists received the full
  nested evaluation;
- two finalists passed every frozen gate;
- the selected 5% `flag_exclude` candidate improved macro-F1 by `0.043090` over
  unchanged and `0.054399` over exact matched random, with wholly positive 95%
  whole-patient intervals and four positive seed directions;
- the selected candidate is checksum-frozen in
  [`configs/aanca_selected_development_candidate.yaml`](configs/aanca_selected_development_candidate.yaml);
- an exact verification rerun reproduced every frozen summary metric and all
  `220/220` audit, baseline and weighted downstream fits converged;
- this establishes controlled-development efficacy only. Confirmation still requires
  untouched new patients and natural-error claims still require blinded multi-rater
  expert evidence.

The frozen protocol is
[`AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md`](AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md),
and the complete result is
[`reports/aanca_autoresearch_expanded_development_results.md`](reports/aanca_autoresearch_expanded_development_results.md).

### M18 — Frozen PUMA new-source controlled confirmation

Status: completed (`EXTERNAL_VALIDATION_COMPLETE`); controlled-noise transfer
supported, no natural-error or clinical status claimed

- freeze the selected candidate, official PUMA archive identities, native-to-primary
  class map, source-group split, corruption seeds, review budget, exact matched
  controls, whole-group bootstrap and simultaneous success rule before metrics;
- use 144 development ROI/case groups and keep 62 final groups unavailable to every
  candidate, score, queue and model decision;
- retain source labels and use controlled corruption only in development;
- separately verify official manifests, zero split overlap, OOF folds, neighbour
  exclusion, exact controls, stored predictions, metrics and bootstrap gates.

Executed result:

- AANCA review precision was `0.537739` versus `0.214379` for exact matched random;
  the advantage was `+0.323359`, interval `[+0.259251, +0.384944]`;
- final macro F1 was `0.646310`, compared with `0.639884` for unchanged corrupted
  training and `0.638243` for matched-random exclusion;
- candidate minus unchanged was `+0.006426`, interval
  `[+0.003657, +0.009365]`; candidate minus matched random was `+0.008067`,
  interval `[+0.004093, +0.011947]`;
- all seed directions, every primary class-recall safeguard, convergence and all
  source/split/hash gates passed;
- the scoped PUMA evidence-readback verifier accepted the saved package; it does not
  independently retrain the 44 models from source images;
- because PUMA does not publish paired natural pre/post expert reviews, the result
  supports controlled transfer only and natural-data action remains
  `retain_uncorrected`.

Public chronology limitation: the PUMA protocol, configuration and result first
appeared together in commit `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`.
Local authorities record the intended freeze-before-metrics order, but public Git
history does not independently timestamp it.

The frozen protocol and result are
[`AANCA_PUMA_NEW_DATA_PROTOCOL.md`](AANCA_PUMA_NEW_DATA_PROTOCOL.md) and
[`reports/puma_new_data_confirmation_results.md`](reports/puma_new_data_confirmation_results.md).

### M19 — Post-confirmation PUMA realism and clean-label stress

Status: completed as exploratory robustness analysis; no independent confirmation
or completion stage claimed

- reuse the unchanged PUMA candidate and opened final groups without permitting
  selection or tuning;
- test clean labels, symmetric 1%, 2.5% and 5%, directional 5% and 10%,
  group-conditional 5% and 10%, and independent geometry-dependent 5% corruption;
- require the same aggregate, seed-direction, exact-control, convergence and
  per-class safety gates for corrupted scenarios and a frozen clean-label safety
  margin.

All nine scenarios had positive whole-group macro-F1 lower bounds against unchanged
and exact matched-random training. Only `group_conditional_10pct` passed every gate;
the other eight failed only the per-class recall safeguard. Clean-label exclusion
reduced `other` recall by `-0.013733`, interval `[-0.025390, -0.002789]`, despite
positive aggregate macro F1. This keeps unreviewed `flag_exclude` prohibited on
natural data and makes class-specific safeguards binding.

The protocol and complete table are
[`AANCA_PUMA_REALISM_STRESS_PROTOCOL.md`](AANCA_PUMA_REALISM_STRESS_PROTOCOL.md) and
[`reports/puma_realism_stress_results.md`](reports/puma_realism_stress_results.md).

### M20 — PUMA audit-time-label allocation sensitivity

Status: completed as post-confirmation exploratory sensitivity; no new completion
stage claimed

- identify that the primary controlled OOF plan used the pre-corruption reference
  label for group stratification;
- freeze an unchanged-candidate sensitivity after openly recording that the PUMA
  result is already known;
- rebuild every seed's group folds and exact neighbours from `observed_label` only;
- retain the primary representation, queue, matched controls, intervention, final
  groups, metrics and simultaneous gates.

Only 22.21%-33.08% of row-level fold assignments matched the primary plan. Despite
that material change, all seven sensitivity gates passed. Retrieval advantage was
`+0.323031`, interval `[+0.259734, +0.381312]`; macro F1 improved by `+0.006679`
over unchanged, interval `[+0.004141, +0.009506]`, and by `+0.009069` over exact
matched random, interval `[+0.005855, +0.012461]`. This supports audit-time-label
robustness but is not independent confirmation and does not evaluate natural error.

The frozen protocol and result are
[`AANCA_PUMA_AUDIT_TIME_LABEL_SENSITIVITY.md`](AANCA_PUMA_AUDIT_TIME_LABEL_SENSITIVITY.md)
and
[`reports/puma_audit_time_label_sensitivity_results.md`](reports/puma_audit_time_label_sensitivity_results.md).

### M21 — Repository maintenance and evidence retention

Status: completed engineering maintenance; no new scientific completion stage claimed

- inventory local size, tracked duplication, run disposition and live consumers;
- preserve raw data, frozen authorities, accepted evidence and verifier inputs;
- retain only the accepted primary and pilot run directories in the active run tree
  after full checksum verification;
- resolve superseded products into a dated quarantine, verify the retained
  authorities, then permanently remove only classified non-authoritative bulk while
  recording the deletion boundary;
- remove public mirrors, stale machine reports, orphaned scaffolding and non-release
  visual captures;
- consolidate behaviorally equivalent archive, metric, figure and pinned-config
  helpers while preserving deliberately independent verification code;
- keep large PUMA evidence under Git LFS and publish the complete retention audit.

The executed inventory and final deletion boundary are in
[`reports/repository_maintenance_2026-08-21.md`](reports/repository_maintenance_2026-08-21.md).

## Open scientific milestones

These require new evidence and cannot be completed by software refactoring.

### M9 — Blinded natural-case expert review

Target stage: no completion status claimed yet

Execution requirements are defined in
[`EXPERT_REVIEW_PROTOCOL.md`](EXPERT_REVIEW_PROTOCOL.md).

- preregister top-K and equal-sized random sampling from non-injected cases;
- obtain independent ratings from multiple qualified pathologists;
- permit ambiguity and abstention;
- report agreement and confidence without forcing a single truth;
- never expose auditor rank to raters.

### M10 — Audit-time-label sensitivity analysis

Status: completed for the frozen PUMA controlled confirmation; still open for the
accepted PanNuke primary benchmark

The post-confirmation PUMA sensitivity rebuilt folds and neighbour reference sets
from `observed_label` only. All seven sensitivity gates passed, but PUMA outcomes
were already open, so this remains exploratory and does not become a second
confirmation. A corresponding accepted-primary PanNuke analysis has not been run.

### M13 — Prospective current-system workflow comparison

Target stage: no completion status claimed yet

Execute [`PROSPECTIVE_WORKFLOW_PROTOCOL.md`](PROSPECTIVE_WORKFLOW_PROTOCOL.md) with
newly recruited reviewers, patient/WSI-separated arms, multiple sites and a fully
untouched external-site downstream test. The system under study remains the current
AANCA implementation. No real-use or model-improvement claim is allowed before that
evidence exists.

### M17 — Measured-utility queue and untouched confirmation

Target stage: no completion status claimed yet

- collect blinded, independent expert development interventions on eligible natural
  cases and retain ambiguity, abstention and every raw vote;
- cross-fit intervention utility by complete patient/source groups and build the
  model-improvement priority only as annotation-inconsistency percentile multiplied
  by a positive conservative utility lower bound;
- keep the selected candidate's natural-data action at `retain_uncorrected` until the
  measured inputs and adoption gates exist;
- retain the completed PUMA controlled confirmation as immutable evidence and never
  reuse its opened final groups to select a revised candidate;
- acquire a genuinely new natural-review cohort with stable paired outcomes rather
  than inferring revisions across unmatched public FOVs;
- freeze one natural-case mapping and one reviewer-gated candidate before outcomes;
- execute the selected policy once on untouched patient/WSI groups and, separately,
  complete a prospective blinded multi-site with/without-AANCA workflow before any
  real-use claim.

The machine and procedural gates are defined in
[`AANCA_MEASURED_UTILITY_PROTOCOL.md`](AANCA_MEASURED_UTILITY_PROTOCOL.md) and
[`AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md`](AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md).

### Consolidated next phase — AANCA v2 research programme

Status: `INITIALISED`; no new efficacy stage claimed

The remaining M9, PanNuke portion of M10, M13 and M17 work is presented as one
promotion sequence in [`NEXT_PHASE.md`](NEXT_PHASE.md): new blinded natural-case
multi-rater evidence, nested measured-utility development, one prospectively frozen
candidate, untouched patient/WSI confirmation and a multi-site with/without-AANCA
workflow study. The current natural-data action remains `retain_uncorrected` until
every required aggregate, class-safety and workflow gate passes.

## Standard validation order

```text
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke
python scripts/present_demo.py --verify-only
```

A failed mandatory gate stops the milestone. Update `STATUS.md` with the executed
command and result, and update `DECISIONS.md` only when a binding decision changes.
