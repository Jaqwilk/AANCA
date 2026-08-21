# PUMA post-confirmation realism and safety stress protocol

**Freeze date:** 2026-08-21 (Europe/Warsaw)  
**Disposition:** exploratory stress analysis after the frozen PUMA primary outcome
was opened; it cannot select, rescue or change the AANCA candidate  
**Candidate:** `78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`

## Purpose

The prospectively frozen PUMA study showed transfer under 10% symmetric label
corruption. Natural annotation problems can be rarer, directional, clustered by
case, or concentrated among morphologically ambiguous instances. This stress suite
tests whether the unchanged 5% `flag_exclude` policy remains useful or at least safe
under those deviations.

The same 144 development and 62 final cases, embeddings, exact fold-safe neighbours,
queue rules, task model and matched-random controls are reused. PUMA final outcomes
are already known, so every result here is exploratory robustness evidence only.

## Frozen scenarios

1. `clean_labels`: no injected corruption; one deterministic run. This is the main
   false-positive safety stress.
2. `symmetric_1pct`, `symmetric_2_5pct`, `symmetric_5pct`: four frozen seeds each.
3. `targeted_5pct`, `targeted_10pct`: four seeds each, with the fixed three-class
   off-diagonal transition probabilities in the accompanying YAML.
4. `group_conditional_5pct`, `group_conditional_10pct`: four seeds each. Cases whose
   SHA-256 of `PUMA-STRESS-HIGH-V1|<case_id>` starts below hexadecimal `40` receive
   weight `4.0`; all other cases receive `0.5`.
5. `instance_geometry_5pct`: four seeds. Selection is driven only by released
   bounding-box width, height, log-area, aspect ratio, normalised x and normalised y.
   These features are cryptographically recorded as independent from the ResNet-18
   auditor. This is a morphology-ambiguity stress, not natural truth.

## Evaluation

For every scenario, refit group-safe OOF audit models, build the unchanged 5% AANCA
queue, draw five exact matched-random queues, apply `flag_exclude`, and evaluate on
the same fixed final cases. Use 3,000 whole-case bootstrap replicates. Preserve all
source labels and convergence flags.

For corrupted scenarios, report the same retrieval and downstream gates as the PUMA
primary study. For the clean-label safety scenario, success means the lower macro-F1
bound is not below `-0.005`, no class-recall lower bound is below `-0.01`, and all
fits converge. A failed scenario narrows the operating claim; no scenario may alter
the frozen candidate after results are known.

