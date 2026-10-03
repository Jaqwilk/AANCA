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
all random and bootstrap arrays, and a standalone evidence recalculator that imports
neither the AANCA package nor scikit-learn. This is software separation, not
third-party validation.

## D017 — Canonical external-evidence files use LF bytes

Status: implemented

Write the published NuCLS `canonical_manifest.csv` files with an explicit LF line
terminator. The repository already declares `eol=lf`; generation must therefore
produce the same bytes that Git checks out on every platform. Pin the normalized
files and their enclosing artifact manifests by byte count and SHA-256 in the
standalone recalculator. This is a serialization correction only: sample identities,
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
content. Vertical rhythm uses one token scale for paragraph, text-to-media and section
gaps; section separators are single hairlines, not stacked decorative rules. The hero
is a full-viewport typography masthead over an original Canvas 2D Second-Look
Review Field (`src/histo_audit/assets/hero-review-field.js`, inlined into the
eleven-file package). Six local transparent PNG nucleus sprites are copied into the
closed package and bound by its manifest; the animation never depends on an external
image host. The animation is decorative and non-interactive: source annotations
remain in place while visual copies enter a short expert-review queue, then the field
settles into a nucleus → patch → four-patch study composition and finally resolves
into the AANCA mark. It must never imply automatic diagnosis, label
correction or that a nucleus is a confirmed error. Reduced-motion users receive a
static final frame. The Method uses four concise article paragraphs covering the
controlled intervention, source-group-safe out-of-fold scoring, equal review budgets
and the separate downstream test. A checksum-bound minimalist workflow graphic mirrors
those states and separates retrieval from downstream evaluation. Output remains only a
ranking for expert review. The former workflow graphic, duplicate review-queue panel,
WebGL animation and square-grid
Priority Review Scaffold stay removed. Navigation chrome remains.
No multi-screen empty scroll theatre is permitted. The pre-polish rollback was removed
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

In public-facing prose, explicitly state that this independence describes a software
boundary and is not third-party validation, external peer review or a second
laboratory replication.

Describe PUMA as an **internally frozen controlled new-source confirmation** whose
success gates were **internally pre-specified**. Do not call its public history
independently time-stamped or publicly preregistered, because protocol and result
entered GitHub together. This terminology correction does not alter any metric,
artifact identity or controlled conclusion.

## D036 — Fail visibly and accessibly at publication boundaries

Status: implemented

GitHub Actions must materialise Git LFS objects during checkout before any verifier
or test reads the PUMA NPZ archives. A repository test guards the checkout contract.
The Method figure is a static checksum-bound PNG with an accessible caption, so it has
no WebGL or Three.js failure mode. The decorative masthead Canvas 2D field must stop under reduced motion and
render a static final frame instead. The scientific article, evidence tables and
“What the study actually learned” section remain ordinary document content and never
depend on animation.

## D037 — Freeze AANCA v1 scientifically; improve only public clarity and reproducibility

Status: implemented

Do not alter the frozen AANCA v1 candidate, PUMA split, parameters, metrics or gates
after outcome inspection. Publicly state that the 144/62 partition is an
AANCA-defined split of the 206 public PUMA ROIs rather than the hidden challenge test,
and that `flag_exclude` omitted the highest-ranked 5% of controlled training rows
without expert review or relabelling.

Use “internally pre-specified” and “internally frozen” for PUMA, together with the
public-history limitation. Add a concise professor-facing brief, a 90-second article
summary, author/AI contribution disclosure, type checking in CI, SRI for external JS,
machine-readable citation metadata and an explicit code-rights boundary. A custom
domain and archival DOI remain owner-controlled publication actions; do not claim
either before it exists.

The article-level disclosure must name AI assistance in project planning, code
drafting and iterative implementation while preserving the human accountability
boundary: Natan Smogór directed and reviewed the work, made the final scientific and
engineering decisions, and remains responsible for the code, analysis and claims.

## D038 — Keep dense evidence controls visually legible

Status: implemented presentation decision; no new scientific stage

Present compact numeric summaries on a shared centred axis so values and labels remain
balanced at desktop, tablet and mobile widths. Treat the seed-identity disclosure as
an explicit evidence control with a clear title, scope note and open-state affordance,
while retaining the exact seed rows and SHA-256 values. Keep a full editorial gap
between reproduction prose and the repository card.

These are presentation-only rules. They do not change evidence, metrics, identifiers,
claims, completion stages or the immutable source-annotation policy.

## D039 — Resolve the decorative review field into the AANCA identity

Status: implemented presentation decision; no new scientific stage

Build the hero field from a new Canvas 2D implementation and six local PNG nucleus
sprites. Brightness adjustment is applied once while preparing each runtime sprite
cache so the supplied dark-cell artwork remains legible without changing the sealed
source PNG files. The review story must keep marked source instances in place and move
only identical visual copies into the expert-review queue. The audit frame visits only
the four desktop or two mobile instances that are actually copied; decorative
non-selection scans are excluded from the story.

After the patch zoom, reveal four separate square patches in a 2 x 2 grid while the
camera is still moving. The sibling split, camera pull-out, group rotation and colour
transition share one overlapping 2.45-second timeline; do not stop at the intermediate
grid before starting the rotation. Preserve the AANCA mark's 8:3 tile-to-gap
proportion, rotate the complete group exactly 45 degrees and fade the tissue detail
into `#5e6ad2`. Do not add a Canvas wordmark. Hold the completed mark for 1.8
seconds. Then drive it through more than two additional rotations with continuous
in/out easing while scaling it into the camera: reach 120 times scale during the dive
and continue to 150 times during the black fade so no logo fragment remains on screen.
Keep the full-black frame for 0.32 seconds, reset the camera while it is hidden, then
release the black layer smoothly and reveal the unchanged field in eight short staggered
groups. Continue directly into the next audit cycle without clearing the canvas. Under
reduced motion, render the completed static mark and do not run the loop.

On mobile, keep the two inspected source nuclei, their two queued visual copies, the
patch zoom, study arrangement and completed mark entirely below the hero copy. Start
the first audit frame close to its target so both inspections read as deliberate scans.
Reach fully opaque black before the dive ends and retain it long enough that no edge of
the enlarged mark can survive into the field restart.

Keep the scan phase brisk enough to match the later transformations while retaining
continuous easing for frame travel, inspection, cloning and queue flight. Decouple
simulation time from display refresh rate and cap expensive Canvas rendering at 60
frames per second. Select bounded Canvas-pixel and patch-cache budgets from available
hardware, memory and data-saver signals; high-DPI or high-refresh screens must not
multiply work without a visible benefit. Pre-decode the six local sprites, use
`createImageBitmap` when available, avoid per-frame DOM updates and suspend the loop
when the hero is outside the viewport or the page is hidden.

Keep the hero copy left aligned in a wider 1320 px rail, remove its redundant visible
eyebrow, and use a shorter non-diagnostic explanation so the animation has a clear
right-side field. These changes affect presentation only and do not alter annotations,
evidence, metrics, scientific claims or completion stages.

## D040 — Use one minimal progress spine for the seven findings

Status: implemented presentation decision; no new scientific stage

Present “What the study actually learned.” as a restrained, centred desktop
scrollytelling sequence with one thin white progress spine, one active registered
question and answer, and one reversible GSAP timeline controlled by one ScrollTrigger.
Preserve the exact seven findings and their limitations. Previous and next stages may
remain faintly visible, but the interface must not behave like a carousel or create an
internal scroll container. The spine is navigation only: do not place mini forest
plots, quantitative glyphs, coloured evidence marks or an atlas preview beside it.
Keep the complete 36-entry forest plot as the separate quantitative view below.

Keep every node centred on the same one-pixel spine. Indicate the active stage with a
solid white fill rather than scaling the SVG circle, because scaling can shift its
rendered centre off the line. Separate consecutive desktop answers by a 320 px
transition offset so neighbouring H1–H7 stages remain visibly distinct.

At widths below 901 px, remove the tall sticky treatment and retain all seven answers
in normal article flow with small static glyphs and no horizontal overflow. Under
`prefers-reduced-motion`, expose a static white spine followed by all answers. These
changes affect only presentation; the hero, source annotations,
frozen evidence, metrics and completion stages remain unchanged.

## D041 — Keep the footer as one editorial surface

Status: implemented presentation decision; no new scientific stage

Use one responsive footer grid for project identity, study status, inspection links
and terms. Keep the copyright, repository licence, third-party asset boundary and
non-clinical statement inside that grid. Do not separate them into bordered strips
below the apparent footer.

Do not display the PUMA commit identifier or `evidence.json` SHA-256 as footer chrome.
Those technical provenance records remain available in the machine-readable evidence,
primary release and checksum-bound package manifest. This simplification changes no
evidence, provenance record, legal term or scientific claim.

## D042 — Use `aancastudy.org` as the official project website

Status: implemented publication decision; no new scientific stage

Use [`https://aancastudy.org`](https://aancastudy.org) as the canonical public
website in current repository presentation and citation metadata. Keep the earlier
Hostinger subdomain only in dated deployment records where it identifies the host
that was actually verified at that time.

The custom domain changes project discoverability and presentation only. It does not
alter a frozen candidate, dataset split, annotation, metric, evidence authority,
claim boundary or completion stage, and it does not constitute an archival DOI.

## D043 — Validate one eligible NuCLS pathologist without consensus leakage

Status: implemented; scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`

Use the frozen current AANCA 64+128 px hybrid candidate to score raw individual
`JP.1` nuclei from NuCLS `U-control`. Build all model inputs before opening the
multi-rater reference: raw `JP.1` bbox, raw observed class and label-independent H&E
pixels only. Split and bootstrap by TCGA patient. Keep score generation OOF and keep
fold-neighbour calculations fold-safe.

After scoring, remove `JP.1` from the reference and require a strict majority of at
least two mappable votes from other individual pathologists. Treat absent annotation,
unmappable vote, ambiguity and insufficient vote count as distinct outcomes. Never
read aggregate P-truth as the reference and never replace the input bbox with a
consensus anchor before scoring.

Report only annotators with public raw individual geometry across at least five
patient groups. Consequently, report `JP.1`; exclude `JP.2` for only two patient
groups and `SP.1`--`SP.3`/`JP.3`--`JP.6` for unavailable individual raw geometry. Do
not pool a one-rotation result.

At the primary 5% budget, compare the global risk ranking with 100 disjoint
equal-budget random queues exactly matched on patient, observed class and proposed
transition. The selected queue must be exact-comparator-capable within every stratum.
Because only five patient groups are available, the 10%-per-patient
`balanced_relaxed` deployment cap cannot fill any non-empty queue; this experiment
therefore validates the frozen global ranking, not that deployment queue.

Freeze the primary gate as a strictly positive lower 95% patient-cluster-bootstrap
bound for the precision difference. Retain the observed PASS (`+0.118222`, 95% CI
`[+0.040000, +0.257143]`) without post-result tuning. Preserve the earlier adverse
NuCLS aggregate/downstream authority unchanged. The allowed claim is enrichment of
independent-pathologist disagreement for the eligible `JP.1` cohort; pathologist
error, biological truth, multi-pathologist generalisation, clinical utility,
automatic correction and downstream improvement remain unestablished.

## D044 — Bind the public presentation to the frozen JP.1 authority

Status: implemented; scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`

Expose the independent-pathologist result on the public evidence page only through
the checksum-bound
`artifacts/nucls_independent_pathologist_validation/results.json` authority. The
presentation builder and standalone verifier must fail closed if the input
pathologist enters the reference, aggregate P-truth is used, patient-group OOF or
fold-safe neighbour constraints are absent, matched queues are not exact and
disjoint, the frozen primary gate is not PASS, or any prohibited claim is enabled.

Present the positive `JP.1` ranking result beside, not instead of, the earlier
adverse NuCLS aggregate/downstream result. State the one-rotation, five-patient,
non-adjudicated and non-prospective boundary directly on the page. Bump the public
evidence and package-manifest schemas so an older verifier cannot silently accept
the expanded evidence contract. Publishing this result changes neither source
annotations nor any completion stage.

## D045 — Replicate natural disagreement enrichment on RIVA and MIDOG++

Status: executed; both frozen gates passed and evidence is published

Retain the frozen 64+128 px AANCA candidate and add two public, additive validations.
Use RIVA as the primary four-rotation multi-rater study in three prospectively mapped
cytology audit classes. Remove the input pathologist and require a strict majority of
at least two other official raw cluster votes. Never use RIVA's released majority label
as the reference.

Use MIDOG++ as a secondary two-rotation pairwise study. Select exactly ten available
cases from each of seven tumor types by a label-independent SHA-256 filename rule.
Alternate the first and second recorded expert labels as input and reference; do not
use the adjudicator label or final category in the primary endpoint. This is pairwise
replication rather than consensus validation.

Enforce a process boundary between input-only snapshot creation, group-safe OOF
scoring, and hidden-reference evaluation. At 5%, compare each rotation with 100
disjoint equal-budget random queues exactly matched on group, observed class and OOF
proposed transition. Bootstrap complete smear/case groups across all rotations. A
dataset passes only if its aggregate precision-difference lower 95% bound is above
zero and every rotation point estimate is non-negative. Cross-dataset replication
requires both dataset gates.

The protocol is additive to the immutable NuCLS study and the frozen base
preregistration; it does not amend either. A positive outcome can support only
enrichment for potentially inconsistent annotations recommended for expert review.
It cannot establish pathologist error, clinical validity, downstream utility,
prospective workflow superiority or permission to change annotations automatically.

Outcome: the score-only artifacts were committed and pushed as
`09aede5000c43406759a432b674f6db37db98b26` before reference opening. RIVA passed
with precision difference `+0.049457` and 95% CI `[+0.014037, +0.091632]`;
MIDOG++ passed with `+0.076575` and `[+0.021352, +0.138159]`. Every one of the six
rotation point differences was non-negative, so the frozen cross-dataset rule is
satisfied. The result changes neither the candidate nor the natural-data action
`retain_uncorrected` and does not permit automatic source-annotation changes.

## D046 — Freeze an evidence-first, consistency-checked final release

Status: implemented presentation and engineering decision; no new scientific stage

Keep the complete long-form article because it exposes methods, negative results,
provenance and limitations, but place one compact “Evidence at a glance” section
before the detailed narrative. Populate it only from the checksum-verified evidence
object. Report NuCLS `JP.1`, RIVA and MIDOG++ natural-disagreement results separately
from PUMA controlled transfer; do not pool them or turn disagreement into an
adjudicated-error claim.

Add a standard-library release verifier that first invokes the closed-package
verifier, then authenticates every embedded upstream source record and checks exact
key metrics, dates and claim-boundary text across the website, professor brief,
ethics, reproducibility and status documents. Run it in CI before installing project
dependencies. Keep the root README as the single official website link.

Reduce the six hero source sprites to a maximum 512 px because the runtime already
uses a maximum 384 px sprite cache. Bind their new sizes and hashes into evidence and
add a regression test for both dimensions and combined transfer. This performance
change does not modify any annotation, model, metric, scientific authority or
completion stage.

## D047 — Preserve V1 and build V2 as a smaller isolated core

Status: accepted engineering-planning decision; no scientific stage change

Treat AANCA v1 as the frozen, auditable reference implementation. Do not refactor its
historical preregistration, recovery, orchestration or verifier paths merely to reduce
line count. A v1 change requires a demonstrated correctness, security,
dependency-compatibility or evidence-readback need and must preserve all existing
scientific and release gates.

Implement AANCA v2 separately with a read-only adapter for frozen v1 evidence. Use an
engineering planning target of approximately 15,000--30,000 physical lines of
production Python, excluding tests and generated evidence. The target is not a
scientific endpoint and cannot justify deleting validation, adverse results, claim
boundaries or group/leakage safeguards. Exceeding it triggers architecture review.

Prefer small single-purpose modules, a standard resumable workflow, independently
timestamped public protocols and content-addressed external artifact storage. Do not
copy the historical v1 capsule/authority/recovery machinery or extend the largest v1
modules with new v2 functionality. V2 may read v1 evidence but may never overwrite,
relabel or upgrade a v1 artifact, result, analysis disposition or completion stage.

## D048 — Require one fail-closed V2 pre-execution authority package

Status: implemented planning decision; no scientific stage change

Use the standalone `Jaqwilk/AANCA-V2` repository as the governing pre-execution
package for V2. Keep a strict authority hierarchy: repository safety and frozen V1
rules, V2 `SPEC.md`, a future checksum-bound preregistration/config/manifest,
procedural protocols, then plan, decision/status logs and reports. Conflicts stop
execution; after freeze they require an amendment, and after reference access
affected alternatives are exploratory.

Make dataset eligibility the first research gate. The immediately executable track
requires a previously unopened public multi-rater source with raw input geometry,
raw individual votes, at least two non-input qualified votes, patient/WSI/case-safe
groups, lawful access and a public score-before-reference seal. Previously opened
PanNuke, NuCLS, MoNuSAC, PUMA, RIVA and MIDOG++ evidence is development-only for V2.
If no candidate passes, record `NO_GO`; do not weaken the gate or manufacture a new
external claim.

Freeze one primary natural endpoint: independent-review-signal precision in the
top-5% queue versus exact equal-budget matched random. Preserve majority non-support,
no-majority ambiguity and raw votes separately; do not call them error or biological
truth. Natural source annotations remain unchanged and the action remains
`retain_uncorrected`. Planning templates retain explicit `UNRESOLVED_BLOCKING`
values until an outcome-blind authority resolves and freezes them.

## D049 — Separate V2 physically and preserve an immutable V1 evidence boundary

Status: implemented repository-boundary decision; no scientific stage change

Move the complete V2 pre-execution package out of the V1 working tree into a sibling
`AANCA-V2/` project directory and give it independent Git history in the private
preparation repository
[`Jaqwilk/AANCA-V2`](https://github.com/Jaqwilk/AANCA-V2). Do not copy V1 source,
artifacts or generated evidence into V2, and do not retain parent-relative links.

Bind V2's read-only V1 authority to public V1 commit
`79d806582c0b618a8c9e3ec1d70313c40be1278e` with explicit Git blob OIDs, SHA-256
digests and commit-pinned URLs. A future V1 baseline update requires an explicit V2
decision and reviewed authority diff; it is never inherited silently.

Repository separation, private visibility and remote backup are engineering and
access-control actions only. They do not approve a dataset, freeze preregistration,
open a new reference, train a model, create a metric or alter any V1/V2 completion
stage or claim.

## D050 — Restore a complete evidence-aware repository README

Status: implemented documentation and release-verification decision; no scientific
stage change

Supersede only the D046 requirement that the root README contain a single website
link. Restore the last complete README structure from the parent of commit `756cbee`
and bring its evidence summary forward using only already sealed RIVA, MIDOG++,
NuCLS and PUMA authorities. Preserve the official website as the primary navigation
target and keep the separate AANCA V2 repository access-controlled.

The professor-release verifier must fail closed on semantic README requirements:
official site, terminology, immutable-source boundary, current evidence values,
`retain_uncorrected`, completion-stage limits, reproducibility and validation. It
must no longer require byte equality with a one-line README. This change restores
repository usability without changing a model, annotation, metric, claim boundary,
scientific authority or completion stage.

## D051 — Consolidate the AANCA brand and design system

Status: accepted design and documentation decision; no scientific stage change

Use [`AANCA_BRAND_SYSTEM.md`](AANCA_BRAND_SYSTEM.md) as the design authority for
future AANCA websites, presentations, reports, posters, figures and interfaces. It
consolidates the existing dark editorial palette, Inter and JetBrains Mono typography,
editorial/figure/wide rails, restrained component language, evidence-visualisation
rules, Second-Look Review Field metaphor, accessibility behaviour and mandatory
scientific claim boundaries.

The canonical mark contains exactly four rounded square tiles in a 2 x 2 grid with an
8:3 tile-to-gap ratio, rotated as one group by 45 degrees. New assets must not add a
centre or fifth tile. Violet `#5e6ad2` is the primary non-text brand accent and
`#828fff` is the accessible normal-text and focus accent on dark surfaces.

The design system does not redefine methods, evidence or status. `SPEC.md`, frozen
protocols and accepted evidence remain authoritative whenever a visual or verbal
choice could affect a scientific claim. This decision changes no source annotation,
dataset, model, split, score, metric, evidence artifact, completion stage or
natural-data action.

## D052 — Record review findings without rewriting frozen evidence

Status: accepted audit and remediation-priority decision; no scientific stage change

Use the 2 October 2026
[`project audit`](reports/project_audit_2026-10-02.md) and its
[`observations`](reports/project_audit_2026-10-02_evidence.json) as an actionable
review record, not as replacement scientific evidence. Keep reproduced interface
failures distinct from failures actually demonstrated in released study artifacts.

Prioritise an accurate description of the primary H4 interval: its existing values
are quantiles over fixed-test random-review repetitions, not whole-group bootstrap
confidence bounds. Preserve those frozen values and their adverse point result.
Any new statistical analysis must have a separate, explicit post-outcome disposition.

Repair demonstrated validation gaps in probability ranges, class-label integer
semantics, adoption thresholds, optimiser-status reporting and neighbour provenance
with focused regression tests. Clarify which verification commands read released
arrays, require raw data, retrain models or write outputs. These changes are justified
correctness and evidence-readback work under D047; they do not justify a broad V1
rewrite merely to reduce line count.

Treat the local presentation and the deployed site as separate verification targets.
The observed official-domain 403, older technical-domain evidence and served-image
hash mismatches require deployment diagnosis and verification of the final served
package. No deployment or production-code remediation was performed by this audit.

Keep the existing release-status date required by the current verifier, and date the
audit as a separate addendum. Record the hard-coded date dependency as a maintenance
finding rather than silently weakening a release check during the review.

Preserve V1's frozen authorities and the separate V2 boundary. This review changes
no source annotation, model selection, scientific result, completion stage or
`retain_uncorrected` policy.

## D053 — Configure the official domain deployment without bypassing authentication

On 2026-10-02 the owner requested deployment of the existing website to
`aancastudy.org`. A local manifest now proposes that domain's `public_html` root,
using merge mode to preserve unrelated hosting files. The existing configured
credential failed SSH/SFTP authentication, so remote-path validation and deployment
remain pending. Do not redirect the upload to the historical temporary-domain root
without verifying its relationship to the requested domain. No scientific artifact,
claim or completion stage changed.

### D053 deployment outcome — 2026-10-02

Authentication was restored with the owner's supplied credential. The requested
apex-domain root was independently verified and the unchanged 13-file package was
deployed there after all mandatory gates passed. Backup `20261002-170743` supports
rollback. Origin files match local SHA-256 values; public CDN image transformations
are documented in STATUS.md. No scientific-stage transition is implied.

## D054 — Repair demonstrated V1 contracts and presentation readback

Status: accepted engineering and publication correction; no scientific stage change

Apply the bounded fixes recorded in
[`reports/v1_remediation_2026-10-02.md`](reports/v1_remediation_2026-10-02.md).
Shared numerical validation must reject invalid distributions, fractional/overflowing
class identifiers and negative global adoption thresholds without silently changing
the supplied data. Maintained OOF/downstream paths must reject a reported failed
fit and retain real diagnostics; absent convergence information stays `unknown`.
Neighbour provenance must exclude the entire holdout group set before any index fit.

Describe H4's existing interval as quantiles across random-review repetitions on
the same fixed final reference set. Preserve all frozen values, file identities and
adverse findings. Presentation schemas may advance to encode that explanation;
they do not amend the statistical analysis or introduce a new confidence interval.
Reject resealed metadata that changes this interpretation.

Living status dates may advance after material work, independently of immutable
scientific release dates. This supersedes D052's temporary requirement to retain the
old status header. Keep date validity, source integrity and claim-scope checks.
Document the actual inputs, model execution and output writes of each verifier.

The canonical website is `https://aancastudy.org/`. Check both origin and public
HTTP delivery. Keep server configuration separate from the thirteen-file package;
request untransformed delivery and require served bytes to match the local release.
Preserve recoverable hosting backups. An origin-only match cannot close a public
integrity failure, and transformed asset hashes must not replace source identities.

These are correctness fixes under D047. Broad V1 redesign, outcome-based tuning and
new confirmatory claims are outside this remediation. The scientific stage stays
`EXTERNAL_VALIDATION_COMPLETE`, presentation stays `DEMO_COMPLETE`, and natural-data
action stays `retain_uncorrected`.

## D055 — Publish the audited V1 changes to the existing GitHub repository

Status: accepted repository synchronisation; no scientific stage change

On 3 October 2026 the owner explicitly authorised resolving the gap between the
updated workspace and `https://github.com/Jaqwilk/AANCA`. Publish the reviewed V1
code, regression tests, release verifiers, current documentation, audit reports,
existing brand specification and corrected thirteen-file presentation on `main`.
Keep deployment configuration limited to the public, credential-free `.htaccess`.

Use a scoped commit and an ordinary fast-forward push. Preserve existing release
tags, frozen evidence identities, source annotations and scientific claim limits.
Check the remote commit, committed presentation bytes and the `Scientific software`
workflow for the published revision. A previous revision's successful CI run does
not establish that the new revision passed. This publication implements D054; it
does not create a new study or alter the V1/V2 boundary established by D047.

### D055 publication outcome — 3 October 2026

The scoped upload published all 35 reviewed files in
`1b64cfd1b0edf87e22d4b04244dfed0bc94fb846` on `main`. Direct remote readback matched
the local commit and the working tree was clean. The staged thirteen-file package
retained the verified manifest-bound bytes. No release tag, frozen scientific
authority or source annotation was changed. This follow-up records publication;
the automatic Ubuntu/Windows workflow provides revision-specific CI evidence.
