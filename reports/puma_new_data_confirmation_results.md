# Internally frozen PUMA new-data confirmation

**Decision:** controlled-noise transfer supported on frozen PUMA final cases.

## Design

The frozen AANCA candidate was applied without PUMA tuning to 144 development cases and evaluated on 62 untouched final cases. The primary mapping was the official PUMA tumor / lymphocyte / other benchmark.

The 144/62 development/final partition is an AANCA-defined split of the 206 public PUMA ROIs. It is not the official hidden PUMA challenge test set.

The downstream intervention was `flag_exclude`: the highest-ranked 5% of training instances were omitted from downstream training. They were not reviewed, corrected or automatically relabelled by an expert. Source annotations remained unchanged.

## Controlled corruption retrieval

AANCA precision was `0.537739` versus `0.214379` for exact matched random. The difference was `+0.323359` with whole-case 95% interval `[+0.259251, +0.384944]`.

## Downstream final-case result

Flag-exclude macro-F1 was `0.646310`, unchanged corrupted-label training was `0.639884`, and mean matched-random exclusion was `0.638243`.
AANCA minus unchanged was `+0.006426` with 95% interval `[+0.003657, +0.009365]`.
AANCA minus matched random was `+0.008067` with 95% interval `[+0.004093, +0.011947]`.

## Internally pre-specified gates

- `retrieval_precision_lower_bound_gt_matched_random`: **PASS**
- `downstream_macro_f1_lower_bound_gt_unchanged`: **PASS**
- `downstream_macro_f1_lower_bound_gt_matched_random`: **PASS**
- `all_four_seed_directions_positive_against_both_controls`: **PASS**
- `every_primary_class_recall_lower_bound_gte_minus_0_01`: **PASS**
- `all_required_models_converged`: **PASS**
- `all_hash_group_split_and_final_fold_guards_passed`: **PASS**

## Interpretation boundary

This is a genuinely new-source controlled annotation-noise result. PUMA does not release paired natural pre/post review labels, so this experiment does not show that AANCA detects pathologist errors, biological truth or clinical benefit. Source annotations were not modified automatically.

The protocol was recorded internally as frozen before PUMA outcomes; public Git
history does not independently verify that timing. The protocol, configuration and
result first appeared together in public commit
`c5bd44193b2abd67bc7e7f1bd9384aa87435d500`.

The PUMA verifier is a project-coupled evidence-readback script that recomputes
metrics from saved predictions but does not retrain all 44 models. It is not
third-party validation or a second image-to-result replication.
