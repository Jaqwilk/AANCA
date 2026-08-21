# Internal technical assessment of the current AANCA system

**Assessment date:** 2026-08-21  
**System:** the existing AANCA implementation, not a replacement or V2  
**Review status:** project-maintainer evidence review; not external peer review,
independent expert endorsement or clinical certification  
**Decision:** scientifically credible research prototype; controlled new-source
model benefit supported; natural-error and real-workflow benefit not yet established

## Executive verdict

AANCA has a coherent scientific purpose and a substantially better implementation
than a typical demonstration-only annotation-quality project. It permanently
separates source, observed and controlled-corruption labels; scores training cases
with group-safe out-of-fold predictions; withholds final groups from selection;
uses exact matched controls; bootstraps complete source groups; preserves adverse
results; and fails closed when a global improvement would damage an important
class. A standalone evidence recalculator that does not import the primary analysis
package, together with immutable hashes and convergence records, makes the primary
numerical claims auditable; this is not third-party validation. PUMA instead uses a
project-coupled saved-evidence readback described below.

The strongest justified statement is now:

> The frozen AANCA candidate prioritised controlled label changes and produced a
> small but statistically supported downstream macro-F1 improvement on a new
> histopathology source, PUMA, while passing all internally pre-specified aggregate and
> class-safety gates of that study.

The requested stronger statement is not yet justified:

> AANCA detects true natural errors made by pathologists and improves a model in a
> prospective real-world pathology workflow.

No software refactor can manufacture that evidence. It requires the same nuclei to
receive blinded independent expert review, preserved raw votes, adjudication and an
untouched external-site downstream evaluation. PUMA publishes final expert-checked
labels, not paired labels before and after natural quality control. The official
NuCLS single-rater assets were inspected specifically for such pairing, but they do
not retain two class states for the same nucleus.

## Evidence-status scorecard

| Area | Evidence status | Reason |
| --- | --- | --- |
| Scientific design for a controlled annotation audit | **Supported** | Correct group separation, OOF scoring, explicit controls, immutable labels and conservative gates |
| Reproducibility and engineering | **Partially supported** | Configurations, hashes, saved-evidence recalculation, tests and fail-closed behaviour are public; a second image-to-result training execution is not published |
| Controlled new-source evidence | **Supported with limitations** | All seven internally pre-specified PUMA gates passed on 62 held-out ROI/case groups, but the effect is modest, corruption remains synthetic and the public commit history is not a pre-outcome timestamp |
| Natural annotation-error evidence | **Not supported** | NuCLS contains genuine multi-rater disagreement, but the combined frozen success rule failed and downstream performance was adverse |
| Prospective operational readiness | **Not evaluated** | No blinded with/without-AANCA workflow, independent sites, qualified new reviewers or external-site operational test has run |

This is an internal evidence classification, not a grade. It separates properties
that are directly inspectable from claims that still require independent people,
new cases or a second execution environment.

## Evidence ladder

| Evidence source | What it tested | Outcome | What it permits |
| --- | --- | --- | --- |
| PanNuke primary | Controlled injected changes and registered H1-H7 benchmark | Ranking evidence was positive in several comparisons; H4 was adverse (`-0.002156`, interval fully below zero) | Controlled benchmark claims only |
| NuCLS multi-rater | Natural disagreement with inferred multi-rater reference | Frozen ranking rule not fully supported; guided-minus-unchanged macro-F1 `-0.014633`, 95% interval `[-0.026683, -0.002415]` | Honest negative natural-disagreement result; no error claim |
| MoNuSAC frozen external | New images with controlled 10% corruption | Retrieval passed; downstream `+0.005526` interval crossed zero and class safety failed | Controlled retrieval transfer, not model improvement |
| MoNuSAC bounded development | Adaptive search within 44 training patients | Selected candidate `+0.043090` vs unchanged and `+0.054399` vs matched random, both intervals positive | Candidate selection only; not independent confirmation |
| PUMA internally frozen controlled confirmation | Frozen candidate on a previously unused official melanoma source | All seven internally pre-specified gates passed; details below | Controlled new-source downstream improvement, with the public-timestamp limitation below |
| PUMA audit-time-label sensitivity | Repeat with OOF allocation using only labels available at audit time | All seven sensitivity gates passed despite materially different folds | Original controlled result does not depend on clean labels for fold allocation |
| PUMA realism stress | Clean, rare, directional, group-clustered and geometry-dependent controlled scenarios | Aggregate downstream intervals were positive in all nine; only one of nine passed every class-safety gate | Exploratory robustness plus a concrete class-safety warning |
| NuCLS supervised-QC feasibility | Whether public data retain paired pre/post QC labels for identical nuclei | Unavailable: one class state per stable element; corrected and uncorrected releases are different FOV tiers | No natural pre/post conclusion |

## New-data confirmation on PUMA

The official PUMA release contains 103 primary and 103 metastatic melanoma ROIs in
the public set, 97,429 nuclei and annotations created by a medical expert and checked
and corrected by a dermatopathologist. The frozen adapter used the challenge's
three-class mapping: tumor, lymphocyte/plasma-cell and other. The official dataset
description and paper are available from the
[PUMA challenge](https://puma.grand-challenge.org/dataset/) and the
[peer-reviewed dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/).

The hash-frozen split contained 144 development ROI/case groups (67,032 nuclei) and
62 final groups (30,397 nuclei). The selected AANCA candidate was not tuned on PUMA.
Development labels received 10% symmetric controlled corruption under four frozen
seeds; final labels remained untouched.

The 144/62 development/final partition is an AANCA-defined split of the 206 public
PUMA ROIs. It is not the official hidden PUMA challenge test set.

The downstream intervention was `flag_exclude`: the highest-ranked 5% of training
instances were omitted from downstream training. They were not reviewed, corrected
or automatically relabelled by an expert. Source annotations remained unchanged.

- review precision: `0.537739` versus `0.214379` for exact matched random;
- precision advantage: `+0.323359`, 95% whole-group interval
  `[+0.259251, +0.384944]`;
- AANCA `flag_exclude` macro-F1: `0.646310`;
- unchanged corrupted-training macro-F1: `0.639884`;
- exact matched-random exclusion macro-F1: `0.638243`;
- AANCA minus unchanged: `+0.006426`, interval
  `[+0.003657, +0.009365]`;
- AANCA minus matched random: `+0.008067`, interval
  `[+0.004093, +0.011947]`;
- all four seed directions were positive against both controls;
- every class-recall lower bound remained above the frozen `-0.01` margin;
- all 44 required fits converged;
- the PUMA evidence-readback verifier passed every source, split, OOF, neighbour,
  comparator, metric and bootstrap guard available in the saved package.

This is the first previously unused-source result in the repository that supports both
retrieval and downstream controlled-noise transfer. The effect is real under the
internally pre-specified analysis but small in absolute macro-F1 terms. It is not a natural-label
intervention result.

### Public timing and verification scope

The PUMA protocol, configuration and result first appeared together in public Git
history in commit `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. The retained local
authorities record that the candidate and gates were frozen before PUMA metrics were
calculated, but GitHub does not provide an independent pre-outcome timestamp for that
ordering. The responsible description is therefore **frozen controlled
confirmation**, not publicly time-stamped prospective preregistration.

The PUMA verifier is a project-coupled evidence-readback script that recomputes
metrics from saved predictions but does not retrain all 44 models. It rebuilds the
released manifest, checks group separation and exact neighbour exclusions, and
recomputes metrics, controls and bootstrap decisions from saved arrays while
importing maintained AANCA helpers. This is not third-party validation or a second
image-to-result replication.

The primary PUMA OOF fold plan was stratified using pre-corruption labels. That is
valid inside a controlled benchmark but unavailable in a real audit. A separately
frozen post-confirmation sensitivity therefore rebuilt every seed's folds and exact
neighbours using only `observed_label`. Only `22.21%` to `33.08%` of row-level fold
assignments matched the original plan. Nevertheless, all seven sensitivity gates
passed: retrieval precision was `0.538186`, with advantage over exact matched random
`+0.323031`, interval `[+0.259734, +0.381312]`; downstream improvement over unchanged
was `+0.006679`, interval `[+0.004141, +0.009506]`; and improvement over matched
random was `+0.009069`, interval `[+0.005855, +0.012461]`. Every class remained above
the `-0.01` safety bound. This removes an important deployment-realism concern, but
is exploratory because PUMA outcomes were already open.

## What the broader stress test revealed

The unchanged candidate was then tested exploratorily on clean labels, symmetric
corruption at 1%, 2.5% and 5%, directional corruption at 5% and 10%, group-clustered
corruption at 5% and 10%, and independent geometry-dependent corruption at 5%.
PUMA outcomes had already been opened, so these scenarios cannot select or rescue
the candidate.

All nine scenarios had a positive 95% lower macro-F1 bound versus both unchanged
training and matched-random exclusion. This consistency is encouraging. Only
`group_conditional_10pct` passed every registered gate, however. The other eight
failed only the simultaneous per-class recall rule.

The clean-label result is the clearest safety warning. Excluding the 5% highest-risk
cases increased macro-F1 by `+0.004540`, interval `[+0.000322, +0.009156]`, while
reducing recall for the heterogeneous `other` class by `-0.013733`, interval
`[-0.025390, -0.002789]`. Under geometry-dependent corruption, `other` recall fell
by `-0.016943`, interval `[-0.028028, -0.005871]`. Directional corruption instead
exposed a lymphocyte-recall weakness.

Therefore the ranking is useful, but automatic unreviewed exclusion is not a safe
natural-data policy. A positive average can conceal redistribution of errors across
classes. The existing fail-closed multicriteria guard is scientifically necessary,
not merely conservative presentation language.

## Does the implementation make sense?

Yes, for its stated research purpose. The strongest implementation choices are:

1. Complete source groups remain together in every audit, downstream and bootstrap
   operation; individual nuclei are never treated as independent patients.
2. Primary audit probabilities are out of fold, and exact neighbours exclude the
   complete query group.
3. The final partition cannot tune the representation, score, queue, budget,
   intervention or downstream model.
4. Pre-corruption, observed and injected-corruption states are distinct immutable
   fields; source annotations are never overwritten.
5. Exact matched-random queues isolate the value of ranking from the effect of simply
   removing the same number and type of examples.
6. Candidate adoption requires aggregate benefit, every-seed direction, per-class
   safety and optimiser convergence.
7. The code records null, adverse and unavailable outcomes instead of replacing them
   with convenient estimates.

Remaining technical limitations are also clear:

- frozen ImageNet ResNet-18 features and a linear classifier may not represent all
  pathology-specific ambiguity;
- the three-class `other` superclass is biologically heterogeneous and is the main
  observed safety weakness;
- the 5% `flag_exclude` intervention discards information and does not use an
  expert-provided replacement, soft distribution or uncertainty state;
- PUMA is one source family and the public split is not a multi-site prospective
  workflow;
- repeatedly analysing the already opened PUMA final set can describe robustness
  but cannot generate a second independent confirmation;
- no public artifact inspected here contains the required paired natural label
  history for identical nuclei.

## Best in-place path forward

These changes extend the current AANCA system rather than discarding it. For public
planning they are now consolidated under the working name “AANCA v2 research phase”
in [`NEXT_PHASE.md`](../NEXT_PHASE.md); that name denotes the missing evidence
programme, not a retroactive upgrade of the current model or results.

### 1. Keep natural deployment reviewer-gated

Use AANCA to order a queue, never to declare an error. `flag_exclude` may remain an
experimental controlled-data arm, but a natural case may receive a changed label,
soft target or zero weight only after a preserved reviewer outcome. Until then the
runtime action stays `retain_uncorrected`.

### 2. Make class safety part of the intervention, not only final reporting

Freeze class and proposed-transition caps, minimum retained counts and a
lower-bound guard before the next study. If one class breaches its recall margin,
retain its unreviewed cases even when the global score is positive. The present
three-class collapse should also be compared with a native-class-aware queue in
development, because `other` combines eight distinct PUMA cell types.

### 3. Train the existing measured-utility queue from real interventions

For every newly reviewed development case, measure the downstream effect on a
different patient group. Cross-fit those effects and rank model-improvement review
by:

`inconsistency percentile × max(conservative utility lower bound, 0)`.

The implementation already fails closed without these inputs. The missing part is
prospective expert evidence, not another scoring formula.

### 4. Compare richer representations only inside fresh development

Compare the current 64+128 px view with larger context, morphometrics and a licensed
pathology encoder. Selection must use new development groups and cannot use the
opened PUMA final result. A representation wins only if aggregate and every-class
gates pass; ranking precision alone is insufficient.

### 5. Run the decisive natural-case study

Collect new patient/WSI groups at multiple sites. Have at least two qualified,
blinded reviewers assess equal-budget AANCA and matched-control queues, preserve
supported/inconsistent/ambiguous/insufficient-context outcomes, and use a separate
blinded adjudication panel. Lock all responses before training. Evaluate the
reviewed-data model once on a completely untouched external site.

This study can support the realistic claim only if AANCA finds more adjudicated
inconsistencies at the same budget, the reviewed-data model's whole-group lower
macro-F1 bound exceeds the unchanged model, every important class passes its safety
margin, and the direction is consistent across registered sites.

## Final claim matrix

| Claim | Current verdict |
| --- | --- |
| The software is a coherent, reproducible annotation-audit prototype | **Supported** |
| It ranks injected label changes better than matched random on new PUMA images | **Supported** |
| Its frozen intervention improves a downstream model under the internally pre-specified PUMA controlled-noise setting | **Supported** |
| It is robust in average macro-F1 across the nine post-confirmation stresses | **Supported, exploratory** |
| Automatic exclusion is safe for every class on clean or broadly realistic data | **Not supported** |
| It identifies natural annotations later judged inconsistent by independent experts | **Not yet supported** |
| It proves that a pathologist made an error | **Not a valid current claim** |
| It improves a real prospective pathology workflow | **Not yet evaluated** |
| It is diagnostic or clinically validated | **Not supported and outside the current evidence** |

The correct expert conclusion is therefore neither “the model does not work” nor
“the real-world objective has been achieved.” AANCA now has credible controlled
generalisation evidence and a useful expert-review ranking, while its automatic
training intervention remains intentionally blocked for natural data until the
missing human and multi-site evidence is collected.
