# AANCA decisions

This file records current binding decisions. Detailed historical deliberations remain
available in Git history and in tag `pre-audit-simplification-2026-08-20`.

## D001 — Non-diagnostic scope

Status: accepted

AANCA ranks potentially inconsistent annotations for expert review. It never changes
source annotations automatically and never treats model disagreement as proof that a
pathologist or biological reference is wrong.

## D002 — Group-safe partitions

Status: accepted

Every scientific split uses `group_id` at least at source-patch level. Nuclei from one
source group may not cross fitting, validation or held-out partitions. The final
reference fold remains untouched and unavailable for model or method selection.

## D003 — Out-of-fold primary scores

Status: accepted

Primary model-based audit scores use group-safe OOF probabilities. A score produced by
a model fitted on the scored nucleus or its source group is ineligible.

## D004 — Immutable label states

Status: accepted

`pre_corruption_label`, `observed_label`, `is_injected_corruption` and corruption
metadata remain separate in storage and APIs. Restoration creates derived arrays and
does not mutate either source label state.

## D005 — Instance-dependent independence

Status: accepted

Instance-dependent corruption and its evaluated auditor must use independently bound
feature spaces. Any overlap is labelled `circularity_risk` and excluded from
confirmatory interpretation.

## D006 — Accepted analysis disposition

Status: accepted

The July freeze lacks an independent public timestamp and outcomes were encountered
during technical recovery. The accepted primary analysis is therefore permanently
described as `amended_or_exploratory`, not untouched confirmatory evidence.

## D007 — Adverse H4 remains prominent

Status: accepted

The saved H4 result is negative. Audit-guided restoration did not outperform random
review on final-fold macro-F1. It must be shown before favourable ranking results and
must not be reframed as downstream improvement.

## D008 — Public evidence release

Status: implemented

Publish the retained statistical, restoration, ranking and OOF evidence as immutable
GitHub Release assets, anchored by repository and per-cell manifests. Do not publish
raw PanNuke images or masks. State that checkpoints were not retained rather than
inventing or regenerating them after outcome inspection.

Release: `primary-evidence-v1`.

## D009 — Independent statistics verifier

Status: implemented

Maintain a small verifier that does not import the AANCA package. It must check fixed
file identities and independently recalculate all available primary comparisons,
Holm corrections and H4 from the released NumPy arrays.

## D010 — Remove unexecuted governance ceremony

Status: implemented

The original capsule, technical authority, resource-controller and resource-bounded
runner stack was not part of the completed primary evidence and caused most Linux CI
failures. Remove it from active source, CLI and tests. Preserve recoverability in Git
tag `pre-audit-simplification-2026-08-20`.

Retain the scientific contracts: group-safe OOF, frozen inputs, immutable labels,
filesystem readback, primary statistics, restoration, QC and external review-package
validation.

## D011 — Cross-platform CI

Status: accepted

The complete maintained test suite, lint, formatting and synthetic smoke run on both
Ubuntu and Windows. Windows-native handle/WOF checks must be explicitly scoped or
removed with their retired feature; local workstation paths and PanNuke files may not
be required during CI collection.

## D012 — Public CLI scope

Status: accepted

Expose commands that are maintained and can provide a truthful outcome: doctor,
synthetic/data preparation, representations, smoke, pilot, primary, preregistration
freeze, original-label audit, external review packaging, reporting and demo serving.

Do not expose the retired direct confirmatory capsule, lifecycle rehearsal,
resource-bounded sensitivity, historical amendment publication or orphan-recovery
orchestration as if they were supported public workflows.

## D013 — Article structure

Status: accepted

Use a continuous research-article flow: scope and thesis, method, reading guidance,
H4, H1-H7, detailed evidence, QC and limitations, reproducibility, author. Keep the
“What the study actually learned” animation while respecting reduced motion and
maintaining one consistent article typography system.

## D014 — Natural-error and clinical claims remain open

Status: accepted

Completion of a responsible external multi-rater study does not itself prove a
natural pathology error, pathologist error, clinical utility or unrestricted
patient/WSI generalisation. Those claims require evidence designed for each claim,
including newly recruited qualified reviewers or prospective clinical evaluation
where applicable.

## D015 — Repeatable synthetic quick-start

Status: implemented

The deterministic data command may reuse an existing output only after independently
regenerating the expected dataset and checking the complete file set, every array,
the manifest and generation evidence. It must fail without writing when any saved
content differs. This makes the documented quick-start repeatable without weakening
the no-overwrite rule.

## D016 — Preserve the frozen NuCLS external result

Status: implemented

Accept the NuCLS Unbiased Control analysis as the primary genuine external
multi-rater evaluation and the NuCLS Evaluation analysis as a secondary sensitivity
analysis. The protocol/configuration public freeze predates outcome download. Exact
official NP/P anchors, TCGA-patient groups and independently inferred pathologist
consensus define the endpoint; consensus is not guaranteed biological truth.

Both frozen primary success decisions are negative. The ranking rule fails because
the 5% precision-minus-prevalence interval crosses zero, and guided correction is
adverse versus leaving labels unchanged. Do not tune the method after outcome
inspection to turn this result positive. `EXTERNAL_VALIDATION_COMPLETE` records
execution and publication, not efficacy.

Publish portable source inventories, canonical paired manifests, numeric evidence,
all random and bootstrap arrays, and an independent verifier that imports neither
the AANCA package nor scikit-learn.

## D017 — Canonical external-evidence files use LF bytes

Status: implemented

Write the published NuCLS `canonical_manifest.csv` files with an explicit LF line
terminator. The repository already declares `eol=lf`; generation must therefore
produce the same bytes that Git checks out on every platform. Pin the normalized
files and their enclosing artifact manifests by byte count and SHA-256 in the
independent verifier. This is a serialization correction only: sample identities,
arrays, metrics, intervals and the negative primary conclusion remain unchanged.

## D018 — Improve the current model without rewriting the frozen result

Status: implemented

Keep one AANCA project and one immutable frozen NuCLS result. Add the existing
fold-safe neighbour signal and fixed hybrid to the current exploratory original-label
audit instead of creating a replacement “v2”. Because NuCLS outcomes were already
known, all candidate comparisons are permanently labelled `post_outcome_exploratory`.

The neighbour candidate passed both primary ranking gates but failed the Evaluation
sensitivity analysis. It is not promoted to the default. Future promotion requires a
fresh prospectively frozen dataset, not retrospective parameter selection.

Reviewed-label retraining is fail-closed: apply a candidate only when independent
group-held-out validation has a lower 95% bootstrap bound above the registered
minimum macro-F1 effect. Otherwise retain the uncorrected model. This rule prevents
uncertain or demonstrated degradation but is not proof of natural-error detection,
clinical utility or prospective workflow benefit.

## D019 — Separate annotation quality from downstream utility

Status: implemented as a prospective current-system policy

An annotation-inconsistency score answers which case merits expert review; it is not
an estimate of training benefit and is not named `P(error)` without new expert
calibration. Maintain two queues. The model-improvement queue fails closed unless
genuinely measured development interventions support nested group-cross-fitted
expected-gain estimates and their lower bounds exceed the frozen minimum. NuCLS may
not supply or tune these estimates after its outcome was inspected.

## D020 — Preserve multi-rater uncertainty and make hard changes exceptional

Status: implemented

Never collapse raw independent votes into a source rewrite. Derived training views
may keep, use a soft distribution, downweight, exclude or make a hard change. Hard
changes are disabled by default and require explicit prospective opt-in, at least two
independent label votes and the registered majority fraction. Ambiguity and
insufficient context remain outcomes rather than hidden missingness.

## D021 — Balance review and enforce the matched comparator in code

Status: implemented

The quality queue supports predeclared caps for source group, observed class, tissue
and proposed transition plus optional embedding-distance diversity. An exact matched
random comparator must contain one control for every top case within each declared
stratum. Under-populated strata fail closed; partial matching is not silently used.
The blinded package records a canonical selection-plan hash and verifies stratum
counts before publication.

## D022 — Candidate adoption requires global benefit and class safety

Status: implemented as software; fresh scientific evidence is not yet available

Compare unchanged, gated-hard, soft, downweighted and abstention-aware training only
on independent development groups. A candidate replaces the unchanged model only if
the macro-F1 whole-group lower bound exceeds the frozen benefit threshold and every
important-class recall lower bound remains above its registered non-degradation
limit. Report Brier score and expected calibration error. The final external test is
unavailable to this choice; adverse, uncertain, non-independent or unavailable
evidence always selects `retain_uncorrected`.

## D023 — Freeze a new controlled MoNuSAC external benchmark before metrics

Status: frozen before outcome execution

Use the official MoNuSAC train split as the controlled-corruption development source
and its official test split as an untouched final evaluation. Split every OOF model
by TCGA patient, exclude ambiguous test regions, retain source annotations unchanged
and pin both official archives by SHA-256. The benchmark evaluates injected-label
recovery and downstream classification only; it cannot establish natural pathology
errors, pathologist errors or clinical utility.

Identifier-only inspection found `TCGA-A2-A0ES` and `TCGA-MP-A4T7` in both official
archives. Exclude these identities from development only and leave the official test
intact. PanNuke lacks sufficient patient metadata to prove complete cross-dataset
non-overlap, so this limitation remains in every report.

The primary candidate is the balanced fold-safe neighbour queue. Its exact budget,
seeds, caps, matched control and simultaneous retrieval/downstream/class-safety rule
are frozen in `configs/monusac_current_aanca_external.yaml`. Do not tune or promote a
candidate from this final result. Preserve null and adverse outcomes.
