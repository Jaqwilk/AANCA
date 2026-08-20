# Frozen MoNuSAC test of the current AANCA

Status before metric execution: **frozen**  
Project: **the existing AANCA, not a replacement or v2**

## Purpose

This test asks a narrow question on genuinely new images: after deterministic label
corruption in the official MoNuSAC training split, does the current AANCA review
policy recover useful training labels and improve a classifier on the untouched
official test split? It is a controlled external benchmark. It does not test whether
AANCA discovers natural pathology errors.

MoNuSAC supplies H&E image regions and nucleus polygons for epithelial cells,
lymphocytes, macrophages and neutrophils across breast, kidney, lung and prostate.
The release is CC BY-NC-SA 4.0. The official train and test archives are pinned by
SHA-256 in `configs/monusac_current_aanca_external.yaml`.

## Leakage controls fixed before outcomes

- The official test labels are not used for corruption, OOF scoring, queue building,
  threshold selection, model fitting or method selection.
- Every OOF fold is split by TCGA patient, never by nucleus or image tile.
- Identifier inspection found `TCGA-A2-A0ES` and `TCGA-MP-A4T7` in both official
  archives. Both patients are excluded from development only; the official test
  remains intact.
- The five NuCLS patients are checked against MoNuSAC identities before execution.
- PanNuke does not expose sufficient patient identifiers to prove cross-dataset
  non-overlap. The report must retain this limitation.
- Ambiguous MoNuSAC test regions are excluded as required by the dataset protocol.
- Source annotations remain read-only. Controlled observed labels and restoration
  labels are separate derived arrays.

## Frozen comparison

Ten percent of the eligible development labels are changed with exact symmetric
random corruption using seed `26082080`. A balanced multinomial logistic model uses
the frozen 64-pixel ImageNet ResNet-18 representation. The audit score is computed
from five-fold patient-group-safe OOF predictions.

At a five-percent review budget the test compares:

1. corrupted labels without review;
2. the earlier global self-confidence queue;
3. balanced self-confidence;
4. balanced fold-safe neighbour disagreement, the primary candidate;
5. the fixed balanced self-confidence/neighbour hybrid;
6. exact matched-random review for the primary candidate;
7. the original uncorrupted training labels as a ceiling, not an attainable policy.

In the controlled benchmark, a selected row is restored only when it is one of the
known injected changes. This simulates successful expert verification without
modifying the source dataset. The balanced queue has frozen patient, class, organ and
transition caps plus a minimum embedding distance. It may underfill; constraints are
not relaxed after seeing the result.

## Success rule

The primary candidate succeeds only if all registered conditions hold:

- its top-K retrieval beats its exact matched-random control;
- its downstream macro-F1 improvement over corrupted/no-review has a positive 95%
  whole-patient bootstrap lower bound;
- its downstream improvement over mean matched-random also has a positive lower
  bound;
- no class has a 95% recall-difference lower bound below `-0.01`.

No candidate is promoted or tuned from this final test. Null and adverse results are
retained.

## What this dataset cannot test

MoNuSAC does not provide the raw independent vote distributions required to evaluate
the new soft-label and multi-reviewer adjudication policies. It also does not provide
prior measured per-case intervention utility, so the model-improvement queue remains
unavailable. Those mechanisms are tested technically but require a new prospective
review study for empirical evaluation.
