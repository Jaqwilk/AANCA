# AANCA development autoresearch protocol

**Study ID:** `monusac_aanca_development_autoresearch_v1`  
**Disposition:** post-external, development-only method search  
**Project:** the existing AANCA implementation; no replacement project or “v2”

## Purpose

This protocol adapts the narrow experiment loop from Karpathy's `autoresearch` to
annotation auditing. The transferable ideas are a fixed evaluator, a bounded trial,
an append-only experiment ledger and automatic keep/discard decisions. The original
single validation-loss objective is not scientifically sufficient here. AANCA uses
patient-group separation, a matched-random review control, downstream utility and
important-class safety gates.

The search may improve a controlled benchmark candidate. It cannot manufacture
expert judgements, natural pathologist errors, prospective workflow evidence or
clinical utility. Those claims remain governed by `EXPERT_REVIEW_PROTOCOL.md` and
`PROSPECTIVE_WORKFLOW_PROTOCOL.md`.

## Immutable evaluator and forbidden inputs

The evaluator uses only the official MoNuSAC **training** archive after the two
previously identified overlapping patients are excluded. The official MoNuSAC test,
both frozen NuCLS subsets, PanNuke final-fold outcomes and every saved external-test
probability are forbidden inputs to candidate generation, ranking, stopping and
selection.

The runner must fail closed unless every sample ID begins with `monusac-train-`, the
grouping unit is the TCGA patient ID, all patient groups are disjoint across every
fit/evaluation boundary and the source annotations remain unchanged. Controlled
corruption retains separate reference, observed and injected-event arrays.

## Internal patient lockbox

Before any search metric is produced, one deterministic group-stratified partition
is created from official-train patients. Fold zero is the internal lockbox. Its
sample outcomes are not passed to the search evaluator. Remaining patients form the
discovery pool.

Candidate generation and successive halving use nested patient-group cross-fitting
inside the discovery pool. The runner must serialize and hash exactly one selected
candidate before it can evaluate that candidate on the internal lockbox. Because
these official-train images participated in earlier model fitting, this lockbox is
an internal overfitting check, not a new external confirmation.

## Search questions

The search separates three questions:

1. Which group-safe OOF score creates the most useful review queue?
2. Which review budget and balancing/diversity policy avoid a harmful selection
   shift?
3. Given simulated controlled review, which training intervention improves a task
   model on excluded patients?

Candidate families include probability risk, fold-safe neighbours and fixed
hybrids; review budgets from 1% through 20%; global and constrained queues; hard
controlled restoration, downweighting and exclusion; classifier regularisation;
and available, provenance-bound feature views. An optional pathology representation
may enter only after its source, licence, exact weights, preprocessing, hardware fit
and embedding smoke test are recorded.

## Nested evaluation

For each controlled-corruption seed and outer discovery fold:

1. keep the outer validation patients uncorrupted and unavailable to every fit;
2. corrupt only the outer-training labels;
3. generate audit probabilities in inner patient-group OOF folds;
4. select the fixed review budget without hidden reference or injected-event fields;
5. create an exact matched-random review comparator;
6. derive candidate training labels or weights without modifying source labels;
7. train the candidate and unchanged task models on outer-training patients;
8. predict only the outer validation patients.

Predictions from all outer folds are assembled once per corruption seed. Search
metrics are therefore out of patient group for both auditing and downstream task
evaluation.

## Selection objective

Ranking AP or top-K precision alone cannot win. The final development candidate is
chosen lexicographically from candidates that pass every available gate:

- candidate top-K precision exceeds its exact matched-random comparator;
- candidate downstream macro-F1 exceeds the corrupted/unchanged model;
- candidate downstream macro-F1 exceeds equal-budget exact matched-random review;
- no important-class recall breaches the registered `-0.01` margin;
- the result is directionally consistent across the registered corruption seeds.

At the final development stage, the lower 95% whole-patient bootstrap bounds must be
positive for both downstream comparisons and non-negative for the retrieval
comparison. If no candidate passes, the selected executable action remains
`retain_uncorrected`; the runner must not select the least adverse candidate and
call it an improvement.

Among passing candidates, maximise the smaller downstream lower bound, then the
point macro-F1 gain, then the retrieval lower bound. Prefer the simpler candidate
when objective values are numerically tied.

## Trial ledger and reproducibility

Every attempted trial is appended to JSONL and TSV ledgers with its candidate hash,
code/config hashes, stage, elapsed time, status (`keep`, `discard`, `crash` or
`timeout`), metrics and reason. Candidate outcomes never rewrite earlier rows. Raw
search artifacts remain local by default; the selected candidate, protocol,
aggregate report and exact evidence identities may be checked in after validation.

The evaluator and partition are fixed during a run. Changing either starts a new
explicit development-search study and does not rewrite this one.

## Meaning of a favourable result

A passing internal lockbox result would support only this statement:

> Under controlled injected corruption on held-out MoNuSAC development patients,
> this frozen AANCA review-and-training policy improved the registered task model
> relative to unchanged and matched-random review conditions.

It would not establish detection of natural errors, that a pathologist was wrong,
real laboratory benefit or superiority in clinical use. Those require newly
recruited blinded pathologists, natural cases, multiple sites and a genuinely new
external test unavailable to this search.

