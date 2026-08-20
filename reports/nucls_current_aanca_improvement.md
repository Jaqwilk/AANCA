# Incremental improvement of the current AANCA model

**Disposition:** post-outcome exploratory method development  
**Frozen NuCLS decision:** unchanged  
**Replacement project or v2:** no

The existing AANCA pipeline now supports a fold-safe neighbour score and a fixed
self-confidence/neighbour hybrid. The analysis below reuses only the preserved
NuCLS manifests, embeddings and OOF probabilities. It does not turn consensus
disagreement into biological truth and cannot revise the frozen negative study.

## Ranking candidates

| Subset | Method | AP | Disagreements at 5% | AP-difference CI | Precision-difference CI | Strict result |
| --- | --- | ---: | ---: | --- | --- | --- |
| unbiased | self_confidence | 0.073489 | 4/41 | [0.006105, 0.213624] | [-0.030075, 0.154075] | failed |
| unbiased | nearest_neighbour_disagreement | 0.068910 | 4/41 | [0.008527, 0.079894] | [0.018415, 0.150843] | passed |
| unbiased | fixed_hybrid | 0.095456 | 5/41 | [0.007214, 0.233275] | [-0.030075, 0.218591] | failed |
| evaluation | self_confidence | 0.083858 | 3/46 | [-0.014534, 0.110686] | [-0.083086, 0.125766] | failed |
| evaluation | nearest_neighbour_disagreement | 0.072560 | 2/46 | [-0.011768, 0.052766] | [-0.099889, 0.075346] | failed |
| evaluation | fixed_hybrid | 0.080111 | 5/46 | [-0.012578, 0.118480] | [-0.099889, 0.178398] | failed |

The neighbour candidate passes both strict conditions in the primary Unbiased
Control subset but not in the declared Evaluation sensitivity subset. It is
therefore **not promoted; the existing default remains in place pending fresh
independent data**. This avoids post-hoc replacement of the frozen self-confidence
result with whichever candidate looks best on one table.

## Retraining safeguard

| Subset | Candidate | Macro-F1 difference | 95% group interval | Action |
| --- | --- | ---: | --- | --- |
| unbiased | frozen_audit_guided_candidate | -0.014633 | [-0.026683, -0.002544] | retain_uncorrected |
| unbiased | full_consensus_label_candidate | -0.013546 | [-0.031162, 0.006088] | retain_uncorrected |
| evaluation | frozen_audit_guided_candidate | -0.008364 | [-0.034640, 0.033899] | retain_uncorrected |
| evaluation | full_consensus_label_candidate | -0.060052 | [-0.081907, -0.036163] | retain_uncorrected |

Both saved correction candidates are rejected because the lower confidence bound
does not exceed zero. The runtime policy therefore retains the uncorrected model
instead of applying a correction that has not demonstrated non-degradation. This is
a safety improvement, not evidence that AANCA improves real deployment.

## Recalculation

The analysis verifies every input file against the immutable NuCLS artifact
manifests, checks exact sample, group and label alignment, reconstructs the original
group folds, and uses the same whole-patient bootstrap rules.

```text
uv run python scripts/analyze_nucls_current_model.py --format markdown
uv run python scripts/analyze_nucls_current_model.py --format json
```

## What remains genuinely open

Natural-error detection still requires blinded adjudication by newly recruited
qualified pathologists. Real-world benefit still requires a prospective, multi-site
comparison of work with and without AANCA. Those outcomes cannot be manufactured
from the existing retrospective evidence.
