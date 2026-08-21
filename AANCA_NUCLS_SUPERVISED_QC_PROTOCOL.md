# Prospective NuCLS supervised-QC evaluation protocol

**Freeze date:** 2026-08-21 (Europe/Warsaw)  
**State at freeze:** official folder metadata inspected; annotation rows, paired
outcomes and RGB pixels not downloaded or opened  
**Project:** the existing AANCA system; this is not a replacement project or V2

## Question and claim boundary

This study asks whether the frozen AANCA method prioritises single-rater NuCLS
annotations that were subsequently changed during the dataset's supervised quality
control, and whether spending the same review budget on the AANCA queue produces a
better downstream classifier than retaining the uncorrected labels or reviewing a
matched-random queue.

The official NuCLS description says that the uncorrected annotations were not
finally reviewed by pathologists, while the corrected annotations were created by
non-pathologists and corrected by study coordinators under pathologist supervision.
Consequently, a positive result may be described only as detection of annotations
later revised during supervised QC. It is not proof that a pathologist made an
error, that the corrected label is biological truth, or that AANCA has clinical
utility.

This is a fresh outcome subset, but not a fully independent source family: other
NuCLS multi-rater subsets were examined earlier in the project. Possible slide or
patient overlap with those subsets must be measured and reported. This study cannot
by itself receive the status `CONFIRMATORY_COMPLETE`.

## Frozen authorities

- official dataset page: <https://sites.google.com/view/nucls/single-rater>
- official code and dataset description:
  <https://github.com/PathologyDataScience/NuCLS>
- uncorrected folder ID: `1zEQCzaufsT14ZZAYVgg6gj4NjsIPrwcg`
- corrected folder ID: `1eGlF9Dgu3WMEik4fqj0wJ13LKVufsfZ0`
- licence: CC0 1.0, as stated by the official dataset page

Every downloaded asset must be hashed before parsing. Missing, inaccessible or
non-pairable data cause an explicit unavailable result; no mirror or inferred label
may silently replace an official asset.

## Frozen eligibility and pairing

1. Pair FOVs by the complete official filename stem. Derive `patient_id` from the
   TCGA barcode and keep every FOV from a patient in one fold.
2. Verify that the image pixels used for a pair correspond to the same official FOV.
   A mismatch excludes the whole FOV and is reported.
3. Pair nuclei without using either class label. Prefer an official stable nucleus
   identifier when it agrees with geometry. Otherwise perform deterministic,
   maximum-IoU one-to-one matching of the released geometries. A pair is eligible
   only at IoU >= 0.50; ties are resolved lexicographically by the two stable row
   identifiers. A sensitivity analysis at IoU >= 0.70 is descriptive only.
4. The primary endpoint includes only paired nuclei with valid image context and an
   eligible official superclass on both sides. Added, removed and geometry-only QC
   changes are retained as separately reported detection-QC outcomes, not forced
   into the class-label endpoint.
5. Freeze the existing NuCLS superclass map: `tumor_any` = tumor or mitotic figure;
   `nonTIL_stromal` = fibroblast, vascular endothelium or macrophage; `sTIL` =
   lymphocyte or plasma cell. Ambiguous, unlabeled and unsupported classes are
   excluded with counts and reasons.
6. Define `observed_label` only from the uncorrected release and `reference_label`
   only from the corrected release. Preserve both fields permanently. Define
   `qc_label_changed` only after all label-independent pairing and eligibility are
   complete.

## Frozen model and evaluation

No NuCLS supervised-QC outcome may alter the candidate. Transfer the already frozen
candidate hash `78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`:

- ImageNet ResNet-18 crops at 64 and 128 pixels, concatenated;
- audit logistic regression, L2 `0.1`, without balanced class weights;
- score = `0.6 * self_confidence_risk + 0.4 * group_safe_k31_neighbour_risk`;
- `balanced_relaxed` queue at a 5% review budget;
- downstream balanced logistic regression, L2 `0.01`.

Use five deterministic outer patient-group folds and four inner patient-group folds.
All audit probabilities for training nuclei must be group-safe out of fold. The
corrected label is unavailable to representation fitting, audit fitting, scoring,
queue construction, thresholds and candidate selection.

Within each outer training partition, simulate the intended expert-review workflow:
reveal the corrected label only for nuclei selected by the frozen AANCA queue and
relabel those reviewed training instances. Compare it with:

1. the unchanged uncorrected-label model;
2. five exact matched-random review queues of equal size, matched without outcome
   access on outer fold, uncorrected superclass and annotation type;
3. the fully corrected-label model as a labelled oracle reference, never as a fair
   budget comparator.

Evaluate all task models once on the outer patient fold using corrected QC labels.
Use identical features, fitting code and convergence requirements for every arm.
Source annotation files are immutable; simulated corrections exist only in derived
arrays.

## Frozen endpoints and success gates

Bootstrap whole patients (`3,000` deterministic replicates) and report fold-level
effects. Success requires all of the following:

- AANCA minus mean matched-random precision for `qc_label_changed` has a 95% lower
  bound above zero at the registered 5% budget;
- reviewed-AANCA minus unchanged macro-F1 has a 95% lower bound above zero;
- reviewed-AANCA minus mean matched-random-review macro-F1 has a 95% lower bound
  above zero;
- at least four of five outer-fold macro-F1 directions versus unchanged are
  positive;
- every eligible superclass recall lower bound versus unchanged is at least
  `-0.01`;
- every required fit converges and every group, pairing and exclusion audit passes.

If any gate fails, the natural-QC activation remains `retain_uncorrected`. A ranking
success without downstream success is reported as ranking-only evidence.

