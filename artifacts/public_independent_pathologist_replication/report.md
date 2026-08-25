# Public independent-pathologist replication results

Completion stage: `EXTERNAL_VALIDATION_COMPLETE`.

Both frozen gates passed: the candidate replicated enrichment for natural independent-expert disagreements in these two public releases.

These outcomes concern enrichment for potentially inconsistent annotations recommended for expert review. They do not prove pathologist error, diagnostic accuracy, clinical safety, or downstream utility.

## RIVA

Frozen dataset gate: `PASS`.

- Eligible rotation-rows: 11373.
- Reviewed rotation-rows at 5%: 571.
- AANCA disagreement precision: 0.476357.
- Exact matched-random mean precision: 0.426900.
- Precision difference: +0.049457.
- Enrichment ratio: 1.115852.
- Group-bootstrap 95% interval for the precision difference: [+0.014037, +0.091632].
- Independent groups in bootstrap: 33.
- Every rotation point estimate non-negative: `true`.

| Input rotation | Eligible | Disagreements | AANCA precision | Random precision | Difference |
|---|---:|---:|---:|---:|---:|
| `annotator_1` | 2988 | 348 | 0.706667 | 0.683200 | +0.023467 |
| `annotator_2` | 2941 | 534 | 0.655405 | 0.613108 | +0.042297 |
| `annotator_3` | 2839 | 299 | 0.246479 | 0.207113 | +0.039366 |
| `annotator_4` | 2605 | 234 | 0.259542 | 0.161298 | +0.098244 |

All queues use exact equal budgets, exact frozen strata, and disjoint random comparators. Source annotations were not modified.

## MIDOG++

Frozen dataset gate: `PASS`.

- Eligible rotation-rows: 7224.
- Reviewed rotation-rows at 5%: 362.
- AANCA disagreement precision: 0.325967.
- Exact matched-random mean precision: 0.249392.
- Precision difference: +0.076575.
- Enrichment ratio: 1.307045.
- Group-bootstrap 95% interval for the precision difference: [+0.021352, +0.138159].
- Independent groups in bootstrap: 70.
- Every rotation point estimate non-negative: `true`.

| Input rotation | Eligible | Disagreements | AANCA precision | Random precision | Difference |
|---|---:|---:|---:|---:|---:|
| `expert_1` | 3612 | 756 | 0.348066 | 0.246409 | +0.101657 |
| `expert_2` | 3612 | 756 | 0.303867 | 0.252376 | +0.051492 |

All queues use exact equal budgets, exact frozen strata, and disjoint random comparators. Source annotations were not modified.

## Frozen interpretation

- RIVA primary gate: `PASS`.
- MIDOG++ replication gate: `PASS`.
- Cross-dataset replication: `SUPPORTED`.
- Earlier NuCLS and downstream results remain additive and unchanged.
- A prospective blinded equal-budget workflow trial and untouched downstream test remain required.
