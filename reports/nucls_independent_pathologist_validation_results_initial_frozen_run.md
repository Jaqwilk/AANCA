# NuCLS independent-pathologist validation results

Status: `EXTERNAL_VALIDATION_COMPLETE`

This frozen evaluation treats disagreements as review outcomes, not as proof that a pathologist was wrong. Source annotations were not modified.

## Design

- Dataset: NuCLS `U-control`.
- Observed input: raw `JP.1` `raw_classification` and individual bbox.
- Hidden reference: strict majority of at least two mappable votes from other individual pathologists; `JP.1` excluded.
- AANCA: frozen 64+128 px ResNet-18 hybrid candidate, patient-group-safe OOF.
- Comparator: 100 disjoint, exact equal-budget matched-random queues, matched on patient, observed class, and OOF proposed transition.

## Primary endpoint (top 5%)

| Quantity | Result |
|---|---:|
| Binary-reference-eligible nuclei | 898 |
| Reference-positive nuclei | 105 |
| Reviewed nuclei | 45 |
| AANCA precision | 0.333333 |
| Matched-random mean precision | 0.215111 |
| Absolute precision difference | +0.118222 |
| Enrichment ratio | 1.549587 |
| Difference 95% patient-bootstrap CI | [+0.040000, +0.257143] |
| Enrichment 95% patient-bootstrap CI | [+1.073620, +10.000000] |
| Primary gate | PASS |

## Outcome accounting

| Outcome | Count |
|---|---:|
| `consensus_agree` | 793 |
| `consensus_disagree` | 105 |
| `ambiguous` | 53 |
| `insufficient_reference` | 368 |

## Secondary budgets

| Budget | Reviewed | AANCA precision | Random precision | Difference | Enrichment | AANCA recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 9 | 0.444444 | 0.231111 | +0.213333 | 1.923077 | 0.038095 |
| 2% | 18 | 0.388889 | 0.240000 | +0.148889 | 1.620370 | 0.066667 |
| 5% | 45 | 0.333333 | 0.215111 | +0.118222 | 1.549587 | 0.142857 |
| 10% | 90 | 0.244444 | 0.165889 | +0.078556 | 1.473543 | 0.209524 |

AUPRC on the binary-reference-eligible cohort: `0.237584`.

## Observed-class breakdown at 5%

| Observed class | Eligible | Positives | AANCA reviewed | AANCA precision | Random precision | Difference | Failure flag |
|---|---:|---:|---:|---:|---:|---:|---|
| `tumor_any` | 394 | 77 | 8 | 0.875000 | 0.815000 | +0.060000 | no |
| `nonTIL_stromal` | 201 | 13 | 20 | 0.150000 | 0.125000 | +0.025000 | no |
| `sTIL` | 303 | 15 | 17 | 0.294118 | 0.038824 | +0.255294 | no |

## Validity and limitations

- The input annotator was removed from reference: `True`.
- Aggregate P-truth fields were read: `False`.
- All OOF models converged: `True`.
- Only `JP.1` qualified. `JP.2` had only two patient groups; other pathologists lacked public individual raw geometry. The per-pathologist and aggregate result are therefore the same single rotation.
- The feasibility audit saw reference prevalence/category counts before freeze, but no AANCA score--reference association. Protocol and execution share one repository change, without an independent timestamp.
- There are only five patient clusters, limiting bootstrap resolution and generalisation.
- This validates the global frozen risk ranking, not the five-patient-infeasible `balanced_relaxed` deployment queue.

## Claim boundary

A positive result supports enrichment of independent-pathologist disagreement for this eligible NuCLS `JP.1` cohort relative to the frozen matched-random control. It does not establish pathologist error, clinical error, automatic correction safety, senior-pathologist generalisation, multi-site generalisation, or downstream/clinical utility. Every flagged nucleus remains recommended for expert review only.
