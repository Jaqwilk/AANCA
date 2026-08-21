# AANCA in one page

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

## Negative evidence retained

On the original PanNuke benchmark, guided restoration was worse than matched random
restoration by `-0.002156` macro-F1, with its interval fully below zero. In the frozen
NuCLS multi-rater evaluation, the ranking success rule failed and guided intervention
was `-0.014633` below unchanged training, interval `[-0.026683, -0.002415]`.
MoNuSAC retrieval was positive, but downstream and class-safety gates failed. These
outcomes are retained rather than explained away or used for post-result tuning.

## Interpretation and limitations

PUMA provides controlled-noise transfer evidence, not proof of natural
pathologist-error detection or clinical benefit. It does not contain paired natural
labels before and after blinded review of the same nuclei. The PUMA protocol,
configuration and result first appeared together in public Git history, so GitHub
does not independently verify the intended pre-outcome timing. The scoped PUMA
readback recomputes saved metrics but does not retrain all 44 models.

## Next decisive experiment

Recruit multiple qualified, blinded pathologists on new patient or WSI groups;
preserve raw votes, ambiguity and abstention; freeze one reviewer-gated intervention
before outcomes; evaluate it once on untouched external groups; and compare review
time and quality with and without AANCA across sites. Until those aggregate,
every-class and workflow gates pass, the natural-data action remains
`retain_uncorrected`.

## Author contribution

Natan Smogór defined and directed the project, reviewed the retained experiments and
is responsible for the public claims. AI-assisted tools supported implementation,
testing, orchestration, documentation and presentation; they supplied no expert
labels and are not independent validators. Full disclosure is in
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
