# NuCLS external multi-rater validation result

**Status:** completed under the publicly frozen protocol  
**Completion stage:** `EXTERNAL_VALIDATION_COMPLETE`  
**Primary conclusion:** the frozen success conditions were not met  
**Claim boundary:** neither a pathologist error nor biological or clinical truth was
established

## What was tested

The external study asked whether a group-safe AANCA self-confidence score could
prioritise naturally occurring disagreement between the official NuCLS inferred
non-pathologist label (`NP-label`) and independently inferred pathologist consensus
(`P-truth`). It separately asked whether correcting the reviewed NP/P disagreements
improved a group-held-out classifier.

The protocol and configuration were committed before the outcome tables were
downloaded and are anchored by public Git tag
[`nucls-external-validation-preregistered-v1`](https://github.com/Jaqwilk/AANCA/tree/nucls-external-validation-preregistered-v1).
The primary subset was NuCLS **Unbiased Control**, in which annotators were not shown
algorithmic suggestions. The NuCLS Evaluation subset, where suggestions were shown,
was a frozen secondary sensitivity analysis and could not rescue a failed primary
result.

Both analyses used exact official `anchor_id` matching, the three official mapped
superclasses, frozen 64 × 64 context crops, ImageNet ResNet-18 embeddings, balanced
multinomial logistic regression, and complete TCGA-patient grouping. Five patients
were held out one at a time; no nucleus-level fallback was used.

## Primary result — Unbiased Control

The eligible set contained 811 nuclei from five TCGA patients. The NP-label and
P-truth differed for 27 nuclei, a prevalence of 0.033292.

| Frozen outcome | Result | 95% group-bootstrap interval | Decision |
| --- | ---: | ---: | --- |
| Average precision | 0.073489 | — | descriptive |
| AP minus prevalence | +0.040197 | [0.006105, 0.213624] | condition passed |
| Precision at the 5% budget | 4/41 = 0.097561 | — | descriptive |
| Precision at 5% minus prevalence | +0.064269 | [-0.030075, 0.154075] | condition failed |
| Guided minus mean-random macro F1 | -0.011750 | [-0.023866, 0.005800] | condition failed |
| Guided minus uncorrected macro F1 | -0.014633 | [-0.026683, -0.002415] | adverse |

Both ranking conditions were required. Although AP enrichment was positive, the
fixed 5% operational condition crossed zero, so the preregistered ranking claim was
not supported. Both downstream conditions were also required. Guided correction
produced macro F1 0.749335, compared with 0.761085 for mean random review and
0.763969 without correction. The frozen downstream claim was not supported and the
comparison with no correction was adverse.

## Secondary result — Evaluation subset

The eligible set contained 908 nuclei from the same five TCGA patient identities and
60 NP/P disagreements (prevalence 0.066079). Average precision was 0.083858, but the
AP-minus-prevalence interval `[-0.014534, 0.110686]` crossed zero. At the 5% budget,
3 of 46 reviewed nuclei disagreed, yielding precision 0.065217 and a
precision-minus-prevalence interval `[-0.083086, 0.125766]`.

Guided correction produced macro F1 0.689818, compared with 0.692547 under mean
random review and 0.698181 without correction. Both downstream intervals crossed
zero. The secondary analysis therefore agreed with the primary decision and, by
design, could not change it.

## What this proves—and what it does not

This execution proves that AANCA has now been evaluated on a genuine external
multi-rater dataset under a prospectively frozen analysis, and that its frozen
natural-disagreement and downstream-utility claims were **not established**. It
does not prove that AANCA detects true pathology errors. NuCLS P-truth is an inferred
multi-rater consensus signal, not guaranteed biological truth, and disagreement does
not mean that any pathologist was wrong.

The external stage is complete because the genuine multi-rater evaluation was
completed and preserved, not because its result was favourable. Blinded prospective
review by newly recruited pathologists, broader datasets, and clinical or patient
outcomes remain unperformed.

## Independent verification

The checked-in evidence contains portable source inventories, canonical matched
manifests, saved OOF probabilities, fixed-budget selections, downstream probability
arrays, random repetitions, and every bootstrap draw. Raw NuCLS images are not
published.

Public release:
[`nucls-external-validation-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/nucls-external-validation-v1)

- asset: `aanca-nucls-external-validation-v1.zip`;
- bytes: `4,001,323`;
- SHA-256: `e7384e2e8ff6eeab97485dfa3196ddbd261bbe335ebfa572d9f275de402a4d08`.

Run:

```text
uv run python scripts/verify_nucls_external_validation.py --json
```

The verifier imports neither `histo_audit` nor scikit-learn. It checks fixed file
sizes and SHA-256 identities, independently rebuilds every ranking and downstream
metric from the saved arrays, regenerates all random baselines and group-bootstrap
draws from the frozen seeds, and re-evaluates the success rules. Its accepted output
is `primary_claim_conclusion: not_supported`.

Primary evidence root: `artifacts/nucls_external_validation/unbiased-v1`  
Secondary evidence root: `artifacts/nucls_external_validation/evaluation-v1`
