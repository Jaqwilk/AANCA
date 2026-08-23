# AANCA natural independent-pathologist validation protocol

## Freeze statement

**Protocol must be frozen before final outcome inspection.**

This protocol and `configs/nucls_independent_pathologist_validation.yaml` freeze the
analysis before any AANCA score is compared with the independent-pathologist outcome.
The feasibility audit necessarily inspected source-schema fields, annotator coverage,
reference-vote counts, and aggregate outcome-category counts. It did **not** compute,
inspect, or use any association between AANCA rankings and the hidden reference.
Therefore this is a prospectively frozen score--outcome evaluation, but not a fully
outcome-naive preregistration. The protocol and results are implemented in the same
repository change, so there is no independent public timestamp separating freeze from
execution. Both limitations must remain visible in the final report.

The protocol is additive. It does not replace or modify the immutable earlier NuCLS
external-validation results.

## Research question

When AANCA receives only the natural nucleus-class annotations of one qualifying
pathologist, does its frozen risk score enrich annotations that disagree with a strict
consensus of other independent pathologists, compared with an exact equal-budget
matched-random review?

Disagreement is not proof of an annotation error. Every selected item is a
`potentially inconsistent annotation` and is `recommended for expert review` only.

## Dataset and authority

The primary source is the public NuCLS multi-rater **Unbiased control** (`U-control`).
NuCLS reports that the same FOVs were independently annotated by multiple participants,
that the multi-rater annotations were not pathologist-corrected, and that the Unbiased
control did not show algorithmic suggestions. The official paper defines `JP` as junior
pathologist, `SP` as senior pathologist, and `Ps` as junior or senior pathologists:

- <https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817>
- <https://github.com/PathologyDataScience/NuCLS>

Frozen local inputs are:

- `data/raw/nucls/unbiased/raw/csv/JP.1_*.csv` for the observed annotations and their
  individual bounding boxes;
- `data/raw/nucls/unbiased/np_truth/rgbs/*.png` for label-independent H&E pixels;
- `data/raw/nucls/unbiased/p_truth/v3.1_final_anchors_U-control_Ps_AreTruth.csv`
  solely for post-scoring nucleus alignment and hidden individual pathologist votes.

The aggregate fields `EM_inferred_label_Ps`, `MV_inferred_label_Ps`, and all
pathologist-consensus probability fields are forbidden as references because they can
include the input pathologist.

## Feasibility decision and annotator selection

An input annotator must have:

1. public individual raw class labels and individual raw geometry;
2. at least 5 distinct TCGA patient groups after label-independent input preparation;
3. all three frozen superclasses in the observed input;
4. enough samples for 5 patient-group OOF folds.

The public package exposes individual raw geometry for `JP.1` (54 FOVs, 5 patients)
and `JP.2` (20 FOVs, 2 patients). Only `JP.1` qualifies. `JP.2` is excluded before
score--outcome evaluation for insufficient patient-group support. `SP.1`, `SP.2`,
`SP.3`, `JP.3`, `JP.4`, `JP.5`, and `JP.6` lack public raw individual geometry in this
package and cannot be substituted with a consensus-derived anchor box. There is one
confirmatory rotation; a pooled multi-pathologist estimate is not claimed.

## Input construction

For each mapped `JP.1` row, `raw_classification` is the only observed class label.
The raw `xmin`, `ymin`, `xmax`, and `ymax` define the input nucleus geometry. Raw FOVs
are paired one-to-one with H&E RGB exports by TCGA slide and minimum frozen boundary
distance. Coordinate scale is derived only from RGB dimensions and filename bounds.
No pathologist reference label or reference-derived anchor geometry enters cropping,
embedding extraction, OOF fitting, probability prediction, neighbour search, or risk
calculation.

The official anchor table is opened only after AANCA scores are complete. Geometric
alignment uses the official NuCLS IoU threshold `0.25`, one-to-one assignment within an
FOV, and a duplicate check that the table's `JP.1` superclass agrees with the raw input
superclass. A conflict is `insufficient_reference`; it is never repaired using another
pathologist's label.

## Frozen class mapping

The three official clinically motivated NuCLS superclasses are used:

- `tumor_any`: `tumor`, `mitotic_figure`;
- `nonTIL_stromal`: `fibroblast`, `vascular_endothelium`, `macrophage`;
- `sTIL`: `lymphocyte`, `plasma_cell`.

Input rows outside these superclasses are excluded before scoring. For hidden reference
votes, `DidNotAnnotateFOV`, `undetected`, missing values, `unlabeled`, `ambiguous`,
`other_nucleus`, `apoptotic_body`, `neutrophil`, `eosinophil`, `myoepithelium`,
`ductal_epithelium`, and any unknown class are non-votes. `undetected` is not converted
to class disagreement because this protocol evaluates class annotations, not missed
nucleus detection.

## Independent reference construction

For a matched `JP.1` nucleus, the input column `JP.1` is removed before any vote is
counted. Only mappable votes from the remaining individual pathologist columns are
eligible. `EM_inferred_label_Ps`, majority-vote fields, non-pathologist votes, and
algorithmic suggestions are never consulted.

The minimum reference is 2 independent mappable pathologist votes. Outcomes are:

- `consensus_agree`: at least 2 other votes, a strict majority exists, and it equals
  the observed superclass;
- `consensus_disagree`: at least 2 other votes, a strict majority exists, and it differs
  from the observed superclass;
- `ambiguous`: at least 2 other votes but no class has more than half the votes;
- `insufficient_reference`: no valid anchor alignment, input-identity conflict, or fewer
  than 2 other mappable votes.

Only `consensus_agree` and `consensus_disagree` enter binary endpoints. Ambiguous and
insufficient-reference rows, with exact reasons, remain in the portable evidence.

## Frozen AANCA score

The score is the already selected AANCA development candidate
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`:

- fixed ImageNet-1K V1 ResNet-18 RGB embeddings from 64 px and 128 px crops,
  concatenated without outcome-dependent weighting;
- patient-group-safe 5-fold OOF multinomial logistic regression;
- audit L2 `0.1`, no class balancing, maximum 400 iterations;
- fold-safe cosine 31-neighbour disagreement;
- percentile-normalised fixed hybrid: `0.6` self-confidence plus `0.4` neighbour
  disagreement.

Every OOF holdout patient is absent from model training and neighbour references. All
scorable mapped input annotations are used to construct OOF scores before the hidden
reference is attached. Reference eligibility never changes model fitting or score
normalisation.

The study endpoint uses the global frozen risk ordering. The selected development
candidate's `balanced_relaxed` deployment queue cannot fill a 5-patient cohort because
its 10%-of-queue per-patient cap requires at least 10 patient groups. It is therefore not
silently relaxed. This validation asks the preregistered ranking question and uses a
study-specific exact-comparator-capable top-risk queue; it does not claim to validate
the full deployment queue policy.

## Primary endpoint and exact matched-random comparator

The primary budget is `ceil(0.05 * N)` over binary-reference-eligible `JP.1` nuclei.
The endpoint is precision of `consensus_disagree` in the AANCA queue versus the mean
precision across 100 deterministic exact matched-random queues of identical size.

Before selecting AANCA items, exact-comparator capacity is enforced within the frozen
strata:

- TCGA patient ID;
- observed superclass;
- OOF proposed transition (`observed -> argmax OOF probability`).

Within every stratum, AANCA may select at most `floor(stratum_size / 2)`, leaving a
disjoint comparator pool. It then selects the largest risks with deterministic sample-ID
tie breaking. Each random replicate samples, without replacement, exactly the AANCA
count in every stratum from non-AANCA rows. Partial, replacement, unmatched, or
unequal-budget comparators are forbidden.

Primary quantities are AANCA precision, mean matched-random precision, absolute
difference, enrichment ratio, reviewed count, and total reference-positive count.

## Confidence interval and aggregation

The 95% interval is the percentile interval from 5,000 deterministic patient-cluster
bootstrap draws. A draw resamples the 5 patient groups with replacement and applies the
same multiplicities to the fixed AANCA and all fixed matched-random selections. It
reports intervals for the precision difference and enrichment ratio. Draws with an
undefined denominator are excluded and their count is reported. Five patients provide
limited cluster-level resolution; this is a mandatory limitation, not a reason to
switch bootstrap units after seeing results.

There is one qualifying input pathologist, so the per-pathologist and aggregate result
are identical. No pseudo-pooled multi-pathologist precision is reported.

## Secondary endpoints

Frozen secondary endpoints are:

- precision, recall, and enrichment at 1%, 2%, 5%, and 10%;
- average precision (AUPRC) on the binary-reference-eligible cohort;
- the cumulative enrichment curve at every ranked eligible sample;
- counts of reference votes and outcome categories;
- descriptive results by observed superclass;
- point-estimate class failure flags where AANCA precision is below its exact matched
  control.

Secondary budgets use the same exact strata, comparator construction, seeds, and
capacity rule. They cannot replace the primary endpoint.

## Success and failure rules

The primary gate passes only if all validity checks pass and the lower endpoint of the
95% patient-cluster bootstrap interval for
`AANCA precision - mean exact-matched-random precision` is greater than zero.

The gate fails closed if the primary budget cannot be filled, any exact comparator is
unavailable or unequal, OOF patient isolation fails, the input pathologist appears in
the reference, source identity/provenance fails, a required model does not converge,
or the confidence interval is undefined. A null or adverse result is reported unchanged.

Class-level failure flags are secondary and do not redefine the primary gate. No
feature, weight, mapping, budget, annotator, threshold, or subgroup may be changed after
score--outcome inspection. Any later changes are exploratory and require a new protocol.

## Leakage prevention and readback

The implementation must prove:

- raw `JP.1` labels and raw boxes are prepared before the anchor table is read;
- hidden votes are absent from feature extraction, OOF fitting, neighbour search, score
  normalisation, and queue-risk ordering;
- patient groups never cross an OOF train/holdout boundary;
- neighbours never come from the query patient;
- `JP.1` is absent from every reference-vote list;
- exact matched queues are disjoint and stratum/count identical;
- source annotations are read-only and never modified;
- portable scored evidence can reproduce metrics without retraining.

The run writes source hashes, config/candidate/protocol hashes, annotator IDs, patient and
FOV IDs, OOF fold evidence, component scores, outcome categories/reasons, queue flags,
matched-random selections, JSON results, and a Markdown report. An independent readback
script recalculates the reported metrics from those artifacts.

## Responsible claim boundary

A positive result would support only this statement: on the eligible NuCLS U-control
`JP.1` class-annotation cohort, the frozen AANCA risk ranking enriched independent
pathologist disagreement relative to its preregistered exact matched-random review.

It would not establish that a pathologist was wrong, that disagreement is a medical
error, that automatic correction is safe, that the balanced deployment queue is
validated, that the effect generalises to senior pathologists, other institutions,
tissues, scanners, or class systems, or that clinical/downstream utility improves.
Source annotations must never be automatically changed.
