# Current AANCA on new MoNuSAC data

**Disposition:** prospectively_frozen_controlled_external_benchmark  
**Decision:** not supported; retain corrupted-baseline comparison and do not claim a better real-world model  
**All frozen success conditions met:** False

This is a controlled external benchmark on new images, not evidence of natural
pathologist-error detection or clinical utility.

## Dataset and corruption

- Development: 29610 nuclei in 44 patient groups.
- Untouched final test: 15494 nuclei in 25 patient groups.
- Controlled corruption: 2961 labels (10.0%), seed `26082080`.
- Two patient identities present in both official archives were excluded from
  development only; the test split remained intact.

## Ranking at the frozen review budget

| Queue | AP | Found / reviewed | Precision | Underfilled |
| --- | ---: | ---: | ---: | --- |
| self_confidence_global | 0.547580 | 943 / 1481 | 0.636732 | False |
| self_confidence_balanced | 0.547580 | 949 / 1481 | 0.640783 | False |
| nearest_neighbour_disagreement_balanced | 0.658142 | 1035 / 1481 | 0.698852 | False |
| fixed_hybrid_balanced | 0.691649 | 1141 / 1481 | 0.770425 | False |

Primary top-minus-matched-random precision: `+0.142843`; 95% whole-patient interval `[+0.099181, +0.188491]`.

## Downstream final-test results

| Condition | Macro F1 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: |
| corrupted_uncorrected | 0.503835 | 0.750871 | 0.752082 |
| self_confidence_global_review | 0.492826 | 0.732993 | 0.744277 |
| self_confidence_balanced_review | 0.490778 | 0.728605 | 0.743788 |
| nearest_neighbour_disagreement_balanced_review | 0.509361 | 0.756357 | 0.759494 |
| fixed_hybrid_balanced_review | 0.493397 | 0.732735 | 0.748486 |
| uncorrupted_reference_ceiling | 0.614845 | 0.875565 | 0.807909 |

Primary minus corrupted/no-review macro F1: `+0.005526`; 95% interval `[-0.001506, +0.012833]`.

Primary minus mean matched-random macro F1: `+0.000031`; 95% interval `[-0.008692, +0.008486]`.

## Frozen success gates

- `primary_top_k_beats_exact_matched_random_control`: **passed**
- `primary_intervention_macro_f1_ci95_lower_gt_corrupted_uncorrected`: **failed**
- `primary_intervention_macro_f1_ci95_lower_gt_mean_matched_random`: **failed**
- `no_important_class_recall_ci95_lower_below_minus_0_01`: **failed**

## Boundary

The source labels were not modified. Soft-label and multi-reviewer behaviour
cannot be evaluated because raw independent vote distributions are unavailable.
The per-case model-improvement queue remains unavailable because this release
does not contain prior measured intervention utility. PanNuke patient metadata
is insufficient to rule out every cross-dataset patient overlap.
