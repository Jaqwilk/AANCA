# Canonical synthetic smoke summary

This document summarises saved machine-readable artifacts from the final canonical runs. These are deterministic software-validation results for the injected process, not evidence about naturally occurring annotation inconsistencies or medical validity.

## 10% controlled-corruption run

Run: `artifacts/runs/20260717T162925.902444Z_synthetic_smoke_5573505315`

Artifact root SHA-256: `95fdefc840f725c5fadcb15804e1a7252aa907d21d6a4080d334144acff35876`; generating source-tree SHA-256: `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a`.

- 300 total samples in 60 source-patch groups.
- 215 audit-pool samples, 25 reference-validation samples, and 60 untouched final-reference samples.
- 22 exact symmetric injected corruptions (requested rate 10%, half-up integer rule).
- Five group-safe OOF folds, exactly-once sample coverage, zero group overlap, and maximum probability-sum error `2.220446049250313e-16`.
- Cleanlab 2.9.0 was available through `cleanlab.rank.get_label_quality_scores` and `cleanlab.filter.find_label_issues`; 41 issue flags were retained as method output, not interpreted as confirmed annotation errors.

### Ranking results

| Method | AP | AUROC | Precision at 5% | Recall at 5% | Lift at 5% |
|---|---:|---:|---:|---:|---:|
| Cleanlab | 0.908849 | 0.964673 | 1.000000 | 0.500000 | 9.772727 |
| Self-confidence | 0.908849 | 0.964673 | 1.000000 | 0.500000 | 9.772727 |
| Negative log likelihood | 0.908849 | 0.964673 | 1.000000 | 0.500000 | 9.772727 |
| Fixed hybrid | 0.901829 | 0.967381 | 1.000000 | 0.500000 | 9.772727 |
| Neighbour disagreement | 0.786484 | 0.968912 | 0.909091 | 0.454545 | 8.884298 |
| Prediction margin | 0.774386 | 0.939001 | 0.909091 | 0.454545 | 8.884298 |
| Predictive entropy | 0.115295 | 0.536740 | 0.000000 | 0.000000 | 0.000000 |

The 5% budget is 11 of 215 audit samples. Cleanlab, self-confidence, NLL, and the fixed hybrid retrieved 11 of 22 injected events with no false alert. Random review used the same 11-item budget over 100 retained deterministic seeds: mean precision 0.094545, mean recall 0.047273, and recall interval [0.000000, 0.160227].

The paired group bootstrap comparing hybrid AP with self-confidence AP used 200 iterations: mean difference -0.008035, interval [-0.069931, 0.032785], and probability of a positive difference 0.385. This smoke does not establish that the hybrid is better.

### Downstream restoration

| Required experiment name | Macro F1 | Reviewed | Restored injected labels |
|---|---:|---:|---:|
| `uncorrupted_reference_baseline` | 0.740171 | 0 | 0 |
| `corrupted_observed_baseline` | 0.686604 | 0 | 0 |
| `random_review_restoration` | 0.683187 mean | 11 per repeat | varies by retained repeat |
| `audit_guided_restoration` | 0.700000 | 11 | 11 |

Audit-guided restoration exceeded the corrupted baseline by 0.013396 macro F1 and the random-review mean by 0.016813 in this single synthetic configuration, but remained below the uncorrupted reference baseline by 0.040171. The result is not a primary-study test and should not be generalised beyond this generator/seed.

## 0% corruption run

Run: `artifacts/runs/20260717T162948.870526Z_synthetic_smoke_zero_corruption_a4d5f87ca0`

Artifact root SHA-256: `2ba716349b010c5bc71f8c7a3b509bfd0f7856c6b138ef8fa0d491050e9236bd`; generating source-tree SHA-256: `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a`.

- 120 total samples; 85 in the audit pool; zero injected corruptions.
- OOF coverage exactly once, zero group overlap, and maximum probability-sum error `4.440892098500626e-16`.
- AP, AUROC, expected random recall, recall, lift, and paired AP inference are saved as structured `not_applicable` values with reasons.
- At the 5% integer budget, five items are reviewed and all five are false alerts by definition because no injected event exists.
- All four downstream conditions have identical macro F1 0.821111 and restore zero labels; the random condition retained all 100 deterministic repeats.

## Persisted evidence and figures

Each canonical run contains the full synthetic arrays, complete JSON/CSV source manifest, controlled-corruption manifest, OOF predictions/provenance, seven-method ranking table, fold-safe neighbour identities/groups/distances, guided and all-random restoration decisions, and final-reference probabilities for every required downstream condition. The tracked reconciliation report validates these jointly before the run is sealed.

The positive report links the saved PR curves, paired-bootstrap interval, class/tissue support panels, top suspicious controlled examples, target audit evidence, fold-safe neighbour grid, and false-high/false-low controlled examples. The 0% report intentionally omits PR/bootstrap/false-low figures and instead links score distributions, false-alert budgets, and false-high-risk examples; its raw-score panel explicitly states that method-specific scales are not directly comparable.

## Interpretation boundary

The positive run demonstrates that the software can detect its deliberately injected symmetric labels, enforce grouping/OOF rules, compare equal budgets, and execute restoration. Predictive entropy was a weak detection score here, and the fixed hybrid did not improve over self-confidence. The 0% run demonstrates honest undefined-metric handling and shows that a forced review budget still creates review workload even when the controlled event count is zero.

No class- or tissue-specific AP is interpreted because every subgroup failed the predefined support threshold. No PanNuke, expert-reviewed, natural-error, diagnostic, patient-level, or clinical conclusion follows from these runs.
