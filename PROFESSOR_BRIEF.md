# AANCA in one page

**Project website:** [`aancastudy.org`](https://aancastudy.org)  
**Public repository:** [`github.com/Jaqwilk/AANCA`](https://github.com/Jaqwilk/AANCA)

## Problem

Large histopathology datasets contain many segmented nuclei with class annotations.
Exhaustively reviewing every annotation is costly. AANCA asks whether a group-safe
model can rank *potentially inconsistent annotations* so a qualified reviewer sees
more useful cases within the same review budget. It never declares that a
pathologist was wrong and never automatically overwrites source labels.

## Method

AANCA preserves source, observed and controlled-corruption labels as separate
immutable fields. Audit scores are generated out of fold: the model scoring a
nucleus is trained without that nucleus and its complete source group. The ranked
queue is compared with an exact equal-budget matched-random queue. Retrieval and
downstream model utility are evaluated separately with whole-group bootstrap
intervals and class-safety gates.

## Strongest current result

The strongest result is the controlled PUMA confirmation. The 144/62
development/final partition is an AANCA-defined split of the 206 public PUMA ROIs;
it is not the official hidden PUMA challenge test set. After 10% controlled
development-label corruption, the 5% AANCA queue achieved precision `0.537739`
versus `0.214379` for matched random review. Its `flag_exclude` training arm reached
macro-F1 `0.646310`, compared with `0.639884` for unchanged corrupted training.
The difference was `+0.006426`, with whole-group 95% interval
`[+0.003657, +0.009365]`. All internally pre-specified aggregate, direction,
convergence and primary class-safety gates passed.

`flag_exclude` means that the highest-ranked 5% of controlled training instances
received zero weight in downstream fitting. They were not reviewed, corrected or
automatically relabelled by an expert, and the source annotations remained unchanged.

## Limited natural multi-rater ranking evidence

In a separate frozen NuCLS `U-control` leave-one-pathologist-out analysis, the only
qualifying input annotator was junior pathologist `JP.1`. At the 5% review budget,
AANCA selected 45 of 898 eligible nuclei and achieved disagreement precision
`0.333333`, compared with `0.215111` for 100 exact matched-random queues. The
difference was `+0.118222`; its five-patient cluster-bootstrap 95% interval was
`[+0.040000, +0.257143]`. The input pathologist was excluded from the strict-majority
reference and aggregate P-truth fields were not used.

This is evidence that the frozen global ranking enriched independent-pathologist
disagreement in one junior-pathologist cohort. It is not a pooled pathologist result,
an adjudicated error study or validation of the deployment queue.

## Public cross-dataset replication

The same frozen candidate was then scored on RIVA and MIDOG++ before their reference
labels were opened; the sealed score artifacts were published first. At 5% review,
RIVA precision was `0.476357` versus `0.426900` exact matched random, difference
`+0.049457`, 95% CI `[+0.014037, +0.091632]`. MIDOG++ precision was `0.325967`
versus `0.249392`, difference `+0.076575`, CI `[+0.021352, +0.138159]`.

All four RIVA leave-one-annotator-out rotations and both MIDOG++ pairwise-expert
rotations were non-negative, so both frozen gates passed. This is stronger evidence
that AANCA enriches review queues for natural independent-expert disagreement across
different public domains. It still does not identify which expert is correct or
establish downstream or clinical utility.

## Negative evidence retained

On the original PanNuke benchmark, guided restoration was worse than matched random
restoration by `-0.002156` macro-F1, with its interval fully below zero. In the frozen
NuCLS multi-rater evaluation, the ranking success rule failed and guided intervention
was `-0.014633` below unchanged training, interval `[-0.026683, -0.002415]`.
MoNuSAC retrieval was positive, but downstream and class-safety gates failed. These
outcomes are retained rather than explained away or used for post-result tuning.

## Interpretation and limitations

PUMA provides controlled-noise transfer evidence. NuCLS, RIVA and MIDOG++ add natural
independent-expert disagreement-enrichment evidence with different reference designs.
The RIVA x MIDOG++ scoring record was public before reference evaluation, but none of
these studies proves pathologist error or clinical benefit, and none contains paired
natural labels before and after blinded review of the same nuclei. The scoped
readbacks are project-maintained verification, not third-party clinical validation.

## Next decisive experiment

Recruit multiple qualified, blinded pathologists on new patient or WSI groups;
preserve raw votes, ambiguity and abstention; freeze one reviewer-gated intervention
before outcomes; evaluate it once on untouched external groups; and compare review
time and quality with and without AANCA across sites. Until those aggregate,
every-class and workflow gates pass, the natural-data action remains
`retain_uncorrected`.

## Final audit handoff

The repository-wide scientific, engineering, security, reproducibility and public-
claim review is summarised in [`FINAL_READINESS_REPORT.md`](FINAL_READINESS_REPORT.md).
It separates completed evidence, retained negative results, independently
recalculable evidence, project-maintained verification and the experiments that still
require new pathologists or a prospective workflow.

## Author contribution

Natan Smogór defined and directed the project, reviewed the retained experiments and
is responsible for the public claims. AI-assisted tools supported implementation,
testing, orchestration, documentation and presentation; they supplied no expert
labels and are not independent validators. Full disclosure is in
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
