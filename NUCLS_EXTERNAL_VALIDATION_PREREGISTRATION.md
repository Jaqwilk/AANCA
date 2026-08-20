# Preregistered NuCLS natural-label external validation

**State:** FROZEN BEFORE OUTCOME INSPECTION  
**Scientific stage at freeze:** `DEMO_COMPLETE`; no external-validation stage claimed  
**Frozen dataset outcome tables inspected before this document:** no

## Why this study exists

The completed PanNuke study establishes only that AANCA can prioritise labels that
the project changed intentionally. It does not establish that the same ranking
prioritises naturally occurring human-label disagreement or that review-guided label
correction improves a classifier on independently assessed natural labels.

This external study uses genuine NuCLS multi-rater annotations. NuCLS contains
independent annotations from non-pathologists and pathologists and publishes
separately inferred non-pathologist labels (`NP-label`) and pathologist consensus
(`P-truth`). The reference remains an inferred multi-rater judgement, not guaranteed
biological truth. A disagreement is never described as proof that a pathologist was
wrong.

## Frozen questions

1. Does a group-safe AANCA self-confidence score rank natural disagreements between
   the inferred NP-label and the independently inferred P-truth above random review?
2. At the same review budget, does replacing only reviewed NP/P disagreements with
   P-truth improve group-held-out classification macro F1 more than random review and
   more than leaving the NP-labels unchanged?

The first question concerns review prioritisation. The second is a retrospective
annotation-review intervention, not a clinical deployment experiment.

## Authorities and data

- Dataset: NuCLS, Amgad et al., GigaScience 2022, DOI `10.1093/gigascience/giac037`.
- Supporting-data DOI: `10.5524/102207`.
- Official project repository:
  `https://github.com/PathologyDataScience/NuCLS`, inspected source commit
  `a87ac50c05cbc8ea11a41516be819f6b31436be7`.
- Official dataset licence: CC0 1.0.
- Primary subset: `Unbiased Control`, because participants did not see algorithmic
  suggestions.
- Secondary sensitivity subset: `Evaluation`, in which participants saw refined
  algorithmic suggestions.
- Observed label: official `EM_inferred_label_NPs` after the official raw-to-super
  mapping.
- Hidden evaluation reference: official `EM_inferred_label_Ps` after the same
  mapping.

The exact Google Drive folder identifiers and checksums are recorded by the frozen
configuration and the acquisition manifest. Downloading is permitted only after the
protocol/configuration freeze is committed publicly.

## Eligible units and labels

The unit is one already localised NuCLS nucleus anchor. The primary endpoint concerns
classification labels only; missed nuclei, extra detections, contour quality and
segmentation quality are out of scope.

The primary class order is the three common NuCLS superclasses defined by the
official source code:

1. `tumor_any`
2. `nonTIL_stromal`
3. `sTIL`

An anchor is eligible only when:

- both NP-label and P-truth exist;
- both map to one of the three frozen superclasses;
- neither side is `undetected`, `AMBIGUOUS`, `other_nucleus`, null or unknown;
- an official FOV image and a valid NP-derived anchor box/centre are available;
- a verified group identifier can be derived without using P-truth;
- its crop and frozen embedding pass all finite-value and identity checks.

Every exclusion and its reason is retained. The analysis fails closed if an unknown
label is encountered, if identifiers are duplicated, or if the data authority does
not match the acquisition manifest.

## Grouping and leakage controls

The strongest available TCGA identity is used in this order: patient barcode,
whole-slide identifier, then FOV. Patient identity is extracted only from official
metadata or a valid TCGA barcode. If patient/slide identity is unavailable for more
than 5% of otherwise eligible anchors, the study fails rather than silently claiming
patient independence. FOV-only results may be generated later only as an explicitly
amended exploratory analysis.

All audit folds, nested review selection folds, downstream outer folds and bootstrap
resamples keep complete groups together. Fold allocation uses observed NP-labels and
never P-truth. Every training partition must contain all three classes; there is no
nucleus-level fallback.

P-truth is unavailable to feature extraction, fold allocation, audit fitting,
probability generation and score ranking. It is opened only for the frozen outcome
evaluation and for simulating the result of an expert reviewing a selected item.

## Frozen representation and auditor

- Input: a deterministic 64 x 64 RGB crop centred on the NP-derived anchor centre;
  out-of-image pixels use reflection padding.
- Representation: official torchvision ImageNet-1K V1 ResNet-18, classification head
  removed, frozen 512-dimensional context embedding.
- Auditor: deterministic balanced multinomial logistic regression, L2 `0.01`, maximum
  `400` iterations.
- Risk: `1 - P(observed NP-label)` from group-safe out-of-fold probabilities.
- Class order: `[tumor_any, nonTIL_stromal, sTIL]` encoded as `[0, 1, 2]`.
- Audit folds: five `StratifiedGroupKFold` folds, seed `26082026`; a group-only
  fallback is allowed only when the stratified splitter is infeasible and every
  fallback training partition still contains all classes. The fallback is recorded.

No hyperparameter, representation, crop size, class mapping or budget may be chosen
from the P-truth outcomes.

## Ranking outcomes

The positive event is `observed NP-label != P-truth`. The primary metric is average
precision (AP). The random AP reference is event prevalence. The operational primary
review budget is 5%; 10% and 20% are secondary.

Report AP, AUROC, precision/recall/lift at every budget, counts, prevalence and 1000
deterministic random rankings. Use 2000 group-bootstrap iterations with seed
`26082031`.

The ranking claim is supported only if both frozen conditions hold in the primary
Unbiased Control subset:

1. the 95% group-bootstrap interval for `AP - prevalence` is entirely above zero;
2. the 95% group-bootstrap interval for `precision_at_5_percent - prevalence` is
   entirely above zero.

Otherwise the ranking result is null or adverse and is reported unchanged.

## Retrospective downstream intervention

Use five outer `StratifiedGroupKFold` folds with seed `26082041`, allocated from
observed NP-labels only. Within each outer training partition, calculate fresh
four-fold group-safe audit scores with seed `26082043`; no outer-test sample may
contribute to the audit model used to select training corrections.

At a 5% training review budget:

- `uncorrected_observed`: train on all NP-labels unchanged;
- `random_review`: select the identical integer count uniformly at random and replace
  a label with P-truth only when the selected NP-label and P-truth disagree;
- `audit_guided_review`: select the highest frozen audit risks and apply the same
  P-truth replacement rule;
- `pathologist_reference_ceiling`: train on P-truth for descriptive context only.

The random condition uses 100 repetitions beginning at seed `26082100`. All
conditions use the same frozen embedding, classifier, outer folds and test samples.
Every outer test fold is evaluated only against P-truth. Concatenated group-held-out
predictions produce accuracy, balanced accuracy and macro F1; macro F1 is primary.

Use 2000 paired group-bootstrap iterations with seed `26082051`. For each bootstrap,
compare guided macro F1 with the mean macro F1 over the 100 matched random-review
predictions and with the uncorrected prediction.

Downstream improvement is supported only if both 95% intervals are entirely above
zero:

1. `audit_guided_review - mean(random_review)` macro F1;
2. `audit_guided_review - uncorrected_observed` macro F1.

The reference ceiling is not part of either success condition.

## Secondary sensitivity analysis

Repeat the frozen ranking and downstream analysis on the NuCLS Evaluation subset.
Because its annotators saw algorithmic suggestions, it cannot rescue a failed primary
Unbiased Control outcome. It is labelled secondary and reports domain/protocol shift.

## Integrity, reporting and claim boundary

Before outcome opening, save hashes for every downloaded input, canonical manifest,
crop array, embedding array, configuration, analysis source and environment. Output
publication is atomic and no-overwrite. All exclusions, failed folds, unavailable
metrics and adverse results remain visible.

Completing this genuine multi-rater analysis can establish
`EXTERNAL_VALIDATION_COMPLETE` even if its result is null or adverse. A positive
ranking result supports prioritisation of natural human-label disagreements relative
to NuCLS P-truth. A positive downstream result supports retrospective model utility
in this specific external dataset. Neither result proves biological truth, a
pathologist error, diagnosis, patient benefit, clinical safety or unrestricted
generalisation.

