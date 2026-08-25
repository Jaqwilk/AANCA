# AANCA public independent-pathologist replication protocol

Status: prospectively frozen before any AANCA-to-reference association is inspected.

Freeze date: 2026-08-25.

Study ID: `public_independent_pathologist_replication_v1`.

## Purpose and claim boundary

This additive study asks whether the frozen AANCA development candidate enriches review
queues for natural, independently recorded pathologist disagreements in two public datasets.
It does not replace or reinterpret the earlier NuCLS result. A disagreement is a
`potentially inconsistent annotation` and a reason for expert review; it is not proof that
any pathologist was wrong, that a medical error occurred, or that AANCA is clinically safe.
Source annotations are immutable and AANCA never changes them automatically.

The study has two components:

1. RIVA is the primary multi-rater replication. Each of four pathologists is treated in
   turn as the input annotator, while a strict leave-one-out majority of at least two other
   pathologists is the hidden reference.
2. MIDOG++ is an independent pairwise replication. The first and second recorded expert
   labels are alternately the input and the hidden reference. The adjudicator label and the
   released final category are excluded from the primary endpoint.

Cross-dataset replication is supported only if both frozen dataset-level gates pass. A
positive result still supports disagreement enrichment only. It does not establish error
detection, diagnostic performance, downstream benefit, prospective workflow benefit, or
clinical utility.

## Frozen AANCA candidate

The candidate is inherited without tuning from
`configs/aanca_selected_development_candidate.yaml`, candidate SHA-256
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`:

- ImageNet `IMAGENET1K_V1` ResNet-18 embeddings from fixed 64 px and 128 px RGB crops,
  concatenated in that order.
- Five-fold group-safe out-of-fold multinomial logistic regression, L2 `0.1`, no balanced
  class weights, maximum 400 iterations.
- Cosine nearest-neighbour disagreement with `k=31`, restricted to training groups for the
  query fold.
- Fixed hybrid risk: `0.6 * self_confidence + 0.4 * neighbour_disagreement`.
- Global review queue selected under exact-comparator capacity constraints.

No dataset-specific model, representation, hyperparameter, budget, class weighting, queue
rule, or threshold may be selected using hidden-reference outcomes.

## Three-process information barrier

For both datasets the execution is split into processes that terminate between stages:

1. `prepare`: open the public source annotations and emit one input-only snapshot per
   rotation. Each snapshot contains only that rotation's observed label, geometry, image
   identity, and group identity. Snapshot hashes and forbidden-column checks are persisted.
2. `score`: read only one input-only snapshot and pixels. Create group-safe OOF predictions,
   AANCA risks, fold evidence, and a pre-reference score seal. This stage must not open the
   full annotation source, RIVA cluster assignments, other expert labels, MIDOG++ final
   categories, or adjudicator labels.
3. `evaluate`: authenticate the frozen config and score seals, then open the hidden
   reference and calculate the frozen endpoints. Risks are never recomputed after the
   reference is opened.

A source or score hash mismatch, a forbidden field in a snapshot, incomplete OOF coverage,
group overlap, non-convergence, query-group neighbour use, reference-voter leakage, or
unavailable exact comparator is a validity failure, not a result.

## RIVA primary multi-rater replication

### Frozen data authority and feasibility facts

The authority is RIVA v1.0 from Zenodo record `17288879`, archive MD5
`89329be851bac81c7b13bde413ae6a6f`, together with the official RIVA GitHub clustering file
at commit `711dfc2c1180d409346f5f94d0cd0ceb485c4ca6`.

The release contains 959 PNG fields, 26,158 annotations, 111 smear-level groups, and four
annotators. The official cluster table contains 17,716 annotations from 386 shared fields
and 7,507 spatial clusters. These feasibility counts and the label-only leave-one-out
outcome counts were inspected before freeze; no AANCA score or AANCA/reference association
was available. Under the frozen three-class mapping, eligible strict leave-one-out counts
for official annotator IDs 10, 11, 12, and 13 were respectively 2,605, 2,839, 2,941, and
2,990. Their disagreement counts were respectively 234, 299, 534, and 350.

The archive license file is treated as the controlling local license authority and raw RIVA
files remain untracked.

### Input, groups, and labels

Each rotation is scored on all mappable annotations made by that input pathologist,
including annotations from fields without a hidden reference. The score population is
therefore fixed before reference eligibility is known.

`group_id` is the smear identity obtained by removing the final field number from the image
stem. All annotations from the same smear stay in one OOF fold. The three frozen classes are:

- `non_lesion`: `NILM`, `INFL`, `ENDO`;
- `low_grade_or_equivocal`: `ASCUS`, `LSIL`;
- `high_grade_or_malignant`: `ASCH`, `HSIL`, `SCC`.

This coarse mapping is an audit label space, not a diagnostic output.

### Hidden reference

The official `curated_clusters.csv` is used only for its raw per-annotator rows and spatial
`cluster_idx`. Its released majority-vote label is forbidden. The input row must match its
own official row by image, official annotator ID, coordinates, and mapped label. The input
annotator is removed. A row is eligible only when at least two distinct other annotators
remain and one frozen three-class label has a strict majority. Duplicate votes by the same
annotator within one cluster, fewer than two independent votes, and ties are excluded with
explicit reasons.

### Primary endpoint

Each rotation contributes its own top-5% AANCA queue and 100 disjoint, equal-budget random
queues matched exactly on smear, observed class, and OOF proposed transition. The aggregate
precision difference is the micro-aggregated AANCA disagreement precision minus the mean
matched-random precision across all four rotations. The 95% percentile interval uses 5,000
bootstrap resamples of smear IDs; every occurrence of a smear across rotations is retained
together in a resample.

The RIVA primary gate passes only when:

- the aggregate precision-difference lower 95% bootstrap bound is greater than zero;
- every rotation's point-estimate precision difference is non-negative; and
- all validity, exact-match, equal-budget, and disjointness checks pass.

Budgets 1%, 2%, and 10%, per-rotation results, observed-class results, average precision,
and enrichment curves are secondary and cannot rescue a failed primary gate.

## MIDOG++ pairwise replication

### Frozen data authority and feasibility facts

The label authority is Figshare article `23531121`, `MIDOG++.json`, MD5
`686c91bcabcc079a000a5be13cc2f542`. The availability authority is
`datasets_xvalidation.csv`, MD5 `65a95814f30a9ebf8312906d2cd7f0ea`, from the official
MIDOG++ collection `6615571`. The JSON contains 553 metadata rows and 26,286 candidates;
503 image files are listed by the availability authority. These counts, class marginals,
and expert-pair disagreement totals were inspected before freeze without AANCA scores.

Downloading all images would require roughly 70--80 GB. The frozen feasibility subset uses
exactly 10 available images from each of the seven tumor types. Within each tumor type,
images are ordered by ascending SHA-256 of
`public_independent_pathologist_replication_v1|<file_name>` and the first ten are selected.
This rule uses no annotation label, agreement, final category, scanner result, or AANCA
quantity. It produces 70 cases and 3,612 candidate annotations in the frozen release.

### Input, groups, labels, and hidden reference

The two audit classes are `mitotic_figure` (public label 1) and
`not_mitotic_figure` (public label 2). `group_id` is the image/case ID. The two rotations
are:

- `expert_1`: `labels[0]` is observed and `labels[1]` is hidden reference;
- `expert_2`: `labels[1]` is observed and `labels[0]` is hidden reference.

The two recorded expert labels are used exactly as released. `category_id`, `labels[2]`,
and any adjudicated/final label are forbidden from the primary endpoint. This design tests
pairwise independent disagreement enrichment, not consensus correctness. It is secondary
because candidates were generated through the source study's expert-screening workflow and
because the third expert did not label every candidate.

Each rotation has its own top-5% queue and 100 disjoint, exact equal-budget random queues
matched on case, observed class, and OOF proposed transition. The aggregate interval uses
5,000 case-ID bootstrap resamples, retaining both rotations of each case together. The
MIDOG++ gate uses the same lower-bound rule and additionally requires non-negative point
estimates in both rotations. The 1%, 2%, and 10% budgets are secondary.

## Interpretation matrix

- Both gates pass: replicated public evidence that the frozen candidate enriches review
  queues for natural independent-expert disagreements across these two releases.
- Only RIVA passes: multi-rater cytology replication supported; cross-dataset replication
  not supported.
- Only MIDOG++ passes: pairwise mitosis replication supported; primary multi-rater
  replication not supported.
- Neither passes: this extension does not support replicated disagreement enrichment.

All outcomes are additive. Earlier positive and negative studies remain visible and are not
overwritten. A future prospective blinded equal-budget workflow trial and an untouched
external downstream test are still required before operational or clinical claims.
