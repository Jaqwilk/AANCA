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
- independent verifier recalculates all available comparisons and H4.

## Current engineering milestone

### M8 — Portable, maintainable public repository

Status: in progress

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

## Next scientific milestones

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

Target stage: no completion status claimed yet

- declare the analysis before execution;
- allocate folds without `pre_corruption_label`;
- preserve group and final-fold boundaries;
- compare against the accepted benchmark as a sensitivity analysis, not a rewrite.

### M11 — External validation

Target stages: `EXTERNAL_VALIDATION_READY`, then
`EXTERNAL_VALIDATION_COMPLETE`

- use a genuinely external dataset;
- verify patient/WSI identifiers and split at the strongest available level;
- freeze the intervention and downstream outcome before inspection;
- keep dataset-specific failures and unavailable cells explicit;
- require qualified expert review before any natural-error claim.

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
