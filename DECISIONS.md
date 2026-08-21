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

The first execution attempt stopped before manifest preparation or metric execution
because the official LZW-compressed TIFF files require `imagecodecs`. Adding that
decoder is an input-compatibility correction only; the frozen scientific
configuration, labels, seeds, budgets, models and success rules remain unchanged.

The next attempt also stopped before metric execution because the first crop
implementation reflect-padded an entire tile for every nucleus and exhausted local
memory. Crop construction now slices the local window first and reflect-pads only a
missing border. The selected centre, 64-by-64 geometry and resulting pixel values are
unchanged; this is a resource correction, not an analytical amendment.

The first complete metric calculation then reached artifact publication but failed
on Windows because the temporary NPZ was opened read-only before `fsync`. After this
point no scientific parameter may change. Opening the same temporary file as `r+b`
is an artifact-durability correction only; the deterministic calculation is rerun
unchanged and its final evidence identities are recorded.

## D024 — Retain the MoNuSAC result without promotion

Status: binding after frozen outcome execution

The balanced fold-safe neighbour queue passed the registered controlled-change
retrieval comparison against exact matched random. It did not pass either downstream
benefit comparison or the simultaneous important-class recall safeguard. Because
the prospectively frozen rule required all four conditions, retain
`corrupted_uncorrected` as the comparison action and do not promote or tune any
candidate from the final MoNuSAC test.

The fixed hybrid had the largest ranking point estimate, but it was not the frozen
primary candidate and its downstream result was adverse. It is not promoted
post-outcome. The positive neighbour macro-F1 point estimate is reported alongside
its interval crossing zero and its practically null comparison with matched-random
restoration.

After metric execution, add only audit fields needed for independent verification:
OOF fold identifiers, organ strata and the exact matched-random indices. This does
not change an input, score, selection, model, seed, budget, metric or decision. A
second complete execution produced byte-identical results, report and source
inventory. Pin the final package and require the independent standard-library/NumPy
verifier to recalculate every published gate.

## D025 — Use a bounded autoresearch loop only inside controlled development

Status: binding after expanded development execution

Adopt the useful mechanics of `karpathy/autoresearch`—a fixed evaluator, bounded
candidate space, append-only keep/discard ledger and simple passing-winner rule—while
preserving AANCA's stronger patient-group, OOF, untouched-test and claim-boundary
requirements. The official MoNuSAC test is permanently unavailable to this search.
Representations, rankings, budgets, interventions and downstream hyperparameters may
compete only in nested development on the official training patients.

The initial full-candidate time allowance proved too short for the declared
multiscale candidates. Before any full-candidate outcome was available, freeze a
runtime-only amendment that raises the allowance to 10,800 seconds without changing
the candidate set, metric, seed, data partition, selection rule or scientific gate.
Record timed-out and exact-comparator-capacity candidates as fail-closed rather than
silently replacing them.

## D026 — Freeze the passing 5% exclusion policy as a development candidate

Status: selected for untouched confirmation; natural-data activation prohibited

Select candidate
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`:
multiscale 64/128 px ResNet-18 features, unbalanced L2=0.1 audit model, fixed hybrid
ranking with 31 neighbours and 0.6 self-confidence weight, relaxed balanced 5% queue,
`flag_exclude`, and balanced L2=0.01 downstream model. It passed the frozen
whole-patient comparisons against unchanged and exact matched random, four-seed
direction rule and important-class recall safeguard.

`flag_exclude` is an experimental controlled-data training view, not permission to
delete or rewrite a source annotation. The checksum-frozen candidate loader must
reject altered authority fields, an altered candidate identity or a missing/mismatched
sibling checksum. Loading this development record always returns
`natural_data_activation_permitted = false`.

Require a clean rerun that records every optimiser flag before accepting the frozen
record. The rerun reproduced the stored metrics and all 220 fits converged. Preserve
the detailed convergence artifact and its SHA-256 in the result report; any later
non-convergence fails the candidate closed.

## D027 — Rank model-improvement review by inconsistency times measured utility

Status: software implemented; empirical inputs and new confirmation remain open

Do not treat annotation inconsistency as downstream utility. Once nested
group-cross-fitted expert intervention outcomes exist, define model-improvement
priority as the percentile-normalised annotation-inconsistency score multiplied by
the positive conservative utility lower bound. A missing OOF audit score, missing
cross-fitted utility, non-independent group identity or non-positive lower bound
fails closed. This queue may never manufacture targets from pre-corruption labels or
from a disclosed final test.

Freeze the selected development candidate before acquiring the next authorised
external archive. The next untouched cohort can test controlled downstream
generalisation, but only a separate blinded multi-rater natural-case study and a
prospective multi-site workflow comparison can support natural-error or real-use
claims.

## D028 — Accept PUMA as positive controlled new-source confirmation

Status: binding after workflow-frozen execution and scoped evidence readback

Use the official PUMA public release under its recorded CC0 authority. Group by the
complete source ROI/case identifier, stratify primary and metastatic melanoma, and
freeze 144 development and 62 final groups by deterministic hash before metrics.
Map the official native labels to the challenge's tumor, lymphocyte/plasma-cell and
other primary classes. Do not use PUMA to modify the candidate selected on MoNuSAC
development.

The frozen candidate passed every registered PUMA gate. Retrieval precision exceeded
exact matched random by `+0.323359`, interval `[+0.259251, +0.384944]`.
Downstream macro F1 exceeded unchanged corrupted training by `+0.006426`, interval
`[+0.003657, +0.009365]`, and exact matched-random exclusion by `+0.008067`,
interval `[+0.004093, +0.011947]`. All four seed directions, all primary class
safeguards, all 44 fits and the source/split guards passed. The PUMA readback script
recomputed the saved-evidence result.

The protocol, configuration and result first appeared together in public commit
`c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. Local authorities record the
intended freeze-before-metrics order, but the public Git history is not an independent
pre-outcome timestamp. The PUMA readback imports maintained project helpers and does
not independently retrain all 44 models from source images. These limits must travel
with every public description of the PUMA result.

Accept the claim `controlled_noise_transfer_supported` for this candidate and
setting. Do not infer natural annotation errors, pathologist errors, clinical
utility or permission to alter source labels. PUMA contains final expert-checked
annotations without paired natural pre/post review states.

## D029 — Record NuCLS supervised-QC pairing as unavailable

Status: binding after official raw-asset feasibility inspection

The official uncorrected and corrected single-rater releases are different FOV
quality tiers, not two label states for the same set of nuclei. The raw SQLite
database contains one class field per stable annotation element and no previous
label, replacement label or revision-history table. Repeated element identifiers
come from geometries crossing FOV records and never expose two distinct class states.

Do not compare unmatched corrected and uncorrected cohorts, infer former labels from
`correction_*` names or treat final QC metadata as an auditor input. Such analyses
would confound source composition or leak the outcome. Preserve the prospective
protocol and publish the endpoint as explicitly unavailable. Natural-data action
remains `retain_uncorrected`.

## D030 — Make the PUMA class-safety stress binding on natural intervention

Status: binding after post-confirmation exploratory execution

The unchanged candidate had a positive whole-group macro-F1 lower bound versus both
unchanged and exact matched-random training in all nine clean and corrupted PUMA
stress scenarios. Only the 10% group-conditional scenario passed every gate. The
other eight failed exclusively because at least one class-recall lower bound breached
`-0.01`. On clean labels, `other` recall fell by `-0.013733`, interval
`[-0.025390, -0.002789]`, despite positive aggregate macro F1.

Do not tune or replace the candidate using the opened PUMA final groups. Treat the
stress as robustness and hazard identification only. Keep `flag_exclude` as a
controlled experimental arm and prohibit unreviewed exclusion on natural data.
Future reviewer-gated development must predeclare per-class and transition caps,
minimum retained counts and a class-specific no-action rule. A positive global
metric may never override a failed class-safety bound.

## D031 — Retain observed-label fold allocation as the realistic sensitivity

Status: binding after post-confirmation exploratory execution

The frozen PUMA benchmark used pre-corruption reference labels only to
stratify development OOF groups. Although every fold remained group-safe and final
groups were untouched, a natural audit does not possess that label. Freeze one
post-confirmation sensitivity that rebuilds both OOF models and exact neighbour
reference sets per seed from `observed_label` only. Do not change any candidate,
queue, intervention, final group, metric or gate.

All seven sensitivity gates passed despite only 22.21%-33.08% row-level fold
agreement with the primary plan. Retrieval advantage over exact matched random was
`+0.323031`, interval `[+0.259734, +0.381312]`; downstream improvement over unchanged
was `+0.006679`, interval `[+0.004141, +0.009506]`; and improvement over matched
random was `+0.009069`, interval `[+0.005855, +0.012461]`. Every class safeguard and
fit converged.

Use `observed_label` as the required fold-assignment authority for future natural
studies. Preserve this result as post-confirmation sensitivity only; it cannot become
a second independent PUMA confirmation or natural-error claim.

## D032 — Keep one authoritative copy and quarantine superseded local products

Status: binding engineering-retention decision; no scientific stage change

Retain raw inputs, frozen specifications, accepted run evidence, reusable embeddings,
independent-verification arrays and the release demo. Remove tracked mirrors,
machine-local reports, orphaned scaffolding and superseded browser captures from the
maintained repository. Preserve historical path references inside sealed provenance;
never rewrite immutable records to conceal that an older local run was retired.

The accepted recovered primary run and accepted pilot may be the only full PanNuke
runs kept in the active workspace after both pass complete integrity verification.
Interrupted, ineligible, smoke and rehearsal runs were first moved to a dated,
resolved quarantine. After integrity checks, the classified large superseded runs,
caches and test products were permanently removed to reclaim disk space. The last
small `mvp_demo_before_author_section/` rollback was removed after final package and
browser verification; no cleanup quarantine remains and the deleted material is not
recoverable.

Large PUMA numeric evidence remains public, checksum-verifiable and available for the
scoped evidence readback, but is stored with Git LFS. Consolidate only behaviorally equivalent helpers. Keep create-only
frozen-cache publication, confirmatory evidence publication and standalone verifier
metrics independent where that separation is itself an integrity control. The full
inventory and rollback location are recorded in
[`reports/repository_maintenance_2026-08-21.md`](reports/repository_maintenance_2026-08-21.md).

## D033 — Presentation article layout over sticky theatre

Status: accepted presentation decision; no scientific stage change

The professor-facing demo is a long-form article, not a scroll-hijacked product page.
Findings remain in normal document flow and receive only a subtle per-answer entrance
animation; mobile, reduced-motion and script-free readers receive the complete static
content. The hero is a typography-only masthead; the optional WebGL review-queue
illustration sits in Method as a captioned figure. Navigation chrome remains. No
multi-screen empty scroll theatre is permitted. The pre-polish rollback was removed
after the current generated package passed final verification.

## D034 — Generate the public article from every current evidence authority

Status: binding presentation and reproducibility decision; no new scientific stage

The checked-in article must be reproducible from the accepted PanNuke run, PanNuke QC
and the tracked NuCLS, MoNuSAC, PUMA confirmation, PUMA stress, PUMA audit-time-label
sensitivity and NuCLS paired-QC-feasibility authorities. Manual edits to generated
metrics or stage text are not authoritative.

Publish `EXTERNAL_VALIDATION_COMPLETE` as the current highest completed scientific
stage, retain `PRIMARY_STUDY_COMPLETE` as the primary-study stage and explicitly show
that `CONFIRMATORY_COMPLETE` is not reached. The natural-data action remains
`retain_uncorrected`. Present the next work as the `INITIALISED` AANCA v2 research
phase in `NEXT_PHASE.md`; this is a prospective evidence programme, not a retroactive
upgrade of the current model or its claims.

## D035 — Reserve independence claims for the operation actually performed

Status: accepted

Use **independent recalculation** only when a verifier does not import the analysis
package and recomputes its stated numeric evidence, as in the primary, NuCLS and
MoNuSAC scripts. Describe the PUMA script as a **scoped evidence readback**: it
rebuilds the official manifest, checks group and neighbour exclusions, and
recalculates decisions from saved predictions, but imports maintained PUMA helpers
and does not rerun the 44 trainings from source images.

Describe PUMA as a frozen controlled new-source confirmation. Do not call its public
history independently time-stamped or publicly preregistered, because protocol and
result entered GitHub together. This terminology correction does not alter any
metric, artifact identity or controlled conclusion.

## D036 — Fail visibly and accessibly at publication boundaries

Status: implemented

GitHub Actions must materialise Git LFS objects during checkout before any verifier
or test reads the PUMA NPZ archives. A repository test guards the checkout contract.
The optional WebGL Method figure must retain the same layout and caption when WebGL,
Three.js or the rendering context is unavailable; an inline static schematic is the
required fallback. The scientific article, evidence tables and “What the study
actually learned” section remain ordinary document content and never depend on
WebGL.
