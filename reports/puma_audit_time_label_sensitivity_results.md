# PUMA audit-time-label sensitivity

This post-confirmation sensitivity allocated every audit fold from the observed label available at audit time. It did not change the candidate.

## Result

AANCA retrieval precision was `0.538186` versus `0.215155` for exact matched random. The difference was `+0.323031` with 95% interval `[+0.259734, +0.381312]`.

AANCA minus unchanged macro-F1 was `+0.006679` with interval `[+0.004141, +0.009506]`. AANCA minus exact matched random was `+0.009069` with interval `[+0.005855, +0.012461]`.

## Frozen sensitivity gates

- `retrieval_precision_lower_bound_gt_matched_random`: **PASS**
- `downstream_macro_f1_lower_bound_gt_unchanged`: **PASS**
- `downstream_macro_f1_lower_bound_gt_matched_random`: **PASS**
- `all_four_seed_directions_positive_against_both_controls`: **PASS**
- `every_primary_class_recall_lower_bound_gte_minus_0_01`: **PASS**
- `all_required_models_converged`: **PASS**
- `all_folds_use_observed_labels_and_exclude_query_groups`: **PASS**

## Boundary

PUMA outcomes were open before this sensitivity was frozen. This is not independent confirmation and does not evaluate natural errors, pathologist errors, clinical utility or automatic annotation changes.
