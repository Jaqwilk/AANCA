# AANCA expanded autoresearch development result

Study: `monusac_aanca_expanded_development_v1`  
Selected candidate:
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`  
Disposition: **frozen development candidate; new external confirmation pending**

## Verdict

The bounded search found a controlled-development policy whose registered intervals
were wholly positive against both unchanged corrupted training and an exact
matched-random intervention under the fixed evaluator. It passed all global,
seed-direction and important-class safety gates. Because the same development
resource supported adaptive candidate selection, these are selection-stage intervals,
not independent confirmatory intervals.

This fixes the earlier controlled downstream failure at the development-evidence
level. It does **not** show that AANCA detects true pathologist errors or is superior
in real use. No natural expert responses or prospective workflow outcomes were used,
and no final external test was opened for this selection.

## Search process

The implementation adapts the fixed-evaluator, bounded-trial, append-only-ledger and
keep/discard principles of Karpathy's `autoresearch` concept. It does not let an
agent edit the evaluator or success gates between trials.

The first search used a one-time 15-patient internal lockbox. Its selected candidate
improved macro F1 by `+0.004690`, but both registered confidence intervals crossed
zero; the lockbox was opened and permanently retired.

The expanded study then used all 44 eligible official MoNuSAC training patients as
development data only:

- 240 ranking-screen configurations;
- 160 downstream-screen configurations;
- 12 finalists fixed before full outcomes;
- 5 outer patient folds and 4 inner audit folds;
- 4 independent controlled-corruption seeds;
- 5 exact matched-random repetitions per seed/fold;
- 3,000 whole-patient bootstrap draws;
- 64 px, 128 px, multiscale, morphology/statistics and Phikon-v2 feature views;
- probability, neighbour and fixed-hybrid scores;
- budgets from 0.5% to 10%;
- restoration, weighting, downweighting and exclusion interventions.

Two ranking configurations failed closed during screening because a complete exact
matched comparator could not be formed. They were recorded as crashes rather than
silently evaluated with a partial control. No downstream or full finalist crashed.

The original 420-second full-trial budget proved too short for the already frozen
finalists. Before the second full outcome was available, a runtime-only amendment
raised the equal limit to 10,800 seconds without changing candidates, folds, seeds,
controls, metrics or gates. All 12 were included; the parent ledger was not rewritten.

## Selected policy

| Component | Frozen value |
| --- | --- |
| Representation | Frozen ImageNet ResNet-18 context, concatenated 64 + 128 px |
| Audit model | L2 `0.1`, no class balancing |
| Risk | Fixed hybrid: 0.6 self-confidence + 0.4 fold-safe 31-neighbour disagreement |
| Queue | `balanced_relaxed` |
| Review/training budget | 5% |
| Intervention | `flag_exclude` — selected training rows receive zero weight; source labels remain unchanged |
| Downstream model | Balanced multinomial logistic regression, L2 `0.01` |

The policy does not guess a replacement label. In the controlled experiment it
removes the high-risk 5% from fitting. On natural data this action remains disabled
until the frozen external and expert-review gates pass.

## Selected result

| Outcome | Selected | Comparator | Difference and 95% interval |
| --- | ---: | ---: | ---: |
| Downstream macro F1 vs unchanged | `0.547194` | `0.504104` | `+0.043090` `[+0.032553, +0.055065]` |
| Downstream macro F1 vs exact matched random | `0.547194` | `0.492795` | `+0.054399` `[+0.034938, +0.075507]` |
| Retrieval precision | `0.947966` | `0.544016` | `+0.403950` `[+0.360560, +0.448903]` |

Candidate-minus-unchanged macro-F1 differences by corruption seed were
`+0.036829`, `+0.047684`, `+0.044879` and `+0.042969`. Differences from the exact
matched-random intervention were `+0.043196`, `+0.059551`, `+0.057061` and
`+0.057791`. Every direction was positive.

Important-class recall differences and intervals:

| Class | Difference | 95% interval | Frozen `-0.01` gate |
| --- | ---: | ---: | --- |
| Epithelial | `+0.056453` | `[+0.044820, +0.067276]` | pass |
| Lymphocyte | `+0.059962` | `[+0.044921, +0.074735]` | pass |
| Macrophage | `+0.009325` | `[-0.006462, +0.024465]` | pass |
| Neutrophil | `+0.014943` | `[-0.001032, +0.029601]` | pass |

The nested folds and seeds contain 23,696 review decisions in total and 22,463
injected changes among them. These are repeated cross-validation decisions, not
23,696 unique clinical cases.

## All frozen finalists

| Candidate | Features | Risk | Budget | Intervention | Δ F1 vs unchanged (lower) | Δ F1 vs matched random (lower) | Minimum class-recall lower | Decision |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `ba572cd4aed1` | ResNet 64 | neighbour | 5.0% | restore | `+0.011479` (`+0.007599`) | `+0.001785` (`-0.002313`) | `-0.012959` | discard |
| `e12b1120b8f6` | Phikon-v2 + ResNet multiscale | neighbour | 10.0% | restore | `+0.087290` (`+0.075319`) | `+0.053248` (`+0.037416`) | `-0.027674` | discard |
| `654a46ba33f6` | ResNet multiscale | fixed hybrid | 7.5% | exclude | `+0.055089` (`+0.047750`) | `+0.078871` (`+0.063684`) | `-0.032773` | discard |
| `e1622805327a` | ResNet multiscale | fixed hybrid | 7.5% | restore, weight 0.5 | `+0.056618` (`+0.048787`) | `+0.055674` (`+0.045310`) | `-0.021944` | discard |
| `d5d3b5eab5d2` | ResNet multiscale | fixed hybrid | 7.5% | restore | `+0.064994` (`+0.056566`) | `+0.052500` (`+0.043499`) | `-0.024559` | discard |
| `08b58d6bbbd5` | ResNet multiscale | fixed hybrid | 7.5% | restore | `+0.057065` (`+0.048778`) | `+0.041537` (`+0.032588`) | `-0.017442` | discard |
| `0197e7546724` | ResNet multiscale | fixed hybrid | 7.5% | restore | `+0.053730` (`+0.045434`) | `+0.038877` (`+0.029173`) | `-0.019203` | discard |
| `78547a73ef23` | ResNet multiscale | fixed hybrid | 5.0% | exclude | `+0.043090` (`+0.032553`) | `+0.054399` (`+0.034938`) | `-0.006462` | **keep** |
| `21c09a7d090b` | ResNet multiscale | fixed hybrid | 7.5% | restore | `+0.045154` (`+0.037320`) | `+0.028042` (`+0.018879`) | `-0.019610` | discard |
| `a0ae130e7b16` | ResNet 64 | fixed hybrid | 10.0% | restore | `+0.030406` (`+0.017487`) | `+0.037687` (`+0.019864`) | `-0.006187` | **keep** |
| `5570417439b1` | ResNet 64 | neighbour | 5.0% | restore | `+0.012494` (`+0.008486`) | `+0.001193` (`-0.002961`) | `-0.001988` | discard |
| `b46926d2f21b` | ResNet multiscale | fixed hybrid | 7.5% | restore, weight 2 | `+0.056906` (`+0.048258`) | `+0.018966` (`+0.010772`) | `-0.014664` | discard |

The Phikon-v2 finalist had large aggregate improvements but failed macrophage and
neutrophil safety intervals. Its encoder was also pretrained on TCGA material, so it
could not provide overlap-free MoNuSAC confirmation. The selected ResNet candidate
has no known TCGA foundation-encoder overlap.

## Evidence identities

| Authority | SHA-256 |
| --- | --- |
| Expanded config | `370b7135858682d0dea52c035768b2fed72acc1fe74a1ddd67996780ad703692` |
| Patient partition | `93087764cf5ce3dd62474ac4da790ff6871d2deff6291d508e02c49ec75f2d2d` |
| Runtime amendment | `2e14a57c72dac193bac8c3179baa90e66b4271a4d3a4fef8f4f8cc0610324a98` |
| Parent run authority | `3ef82963925cea7d20332f13488578ded5eba1df750c52cb55cef69521580042` |
| Append-only parent ledger | `1e5378ebbb1a02cdd003fd6bed96d78a200b53be210b746c1477d46a2025e728` |
| Selected candidate record | `229bc293b3ba7c3909423178552f5f3789f00411223c2f87b5185eee1542487d` |
| Convergence evidence | `d10fbcb3179abe6058ae43231663f3aeefc7d754c40bac2bfa6fdea1a4abae38` |

## Numerical verification

The frozen winner was evaluated again from its pinned candidate, parent authority,
patient partition and append-only ledger. All `220/220` optimiser fits converged:
`100` hard-label audit/baseline fits and `120` weighted downstream fits. The rerun
reproduced candidate macro F1, both downstream differences and the retrieval
difference exactly to the stored floating-point values. The detailed local evidence
is `artifacts/autoresearch/monusac_aanca_expanded_convergence_v1.json` (50,604 bytes;
SHA-256 shown above).

## What remains unproved

- No natural case was labelled wrong by this experiment.
- No new blinded pathologist reviewed the selected queue.
- No prospective laboratory used AANCA versus standard practice.
- No new untouched external cohort evaluated the selected policy.
- The model-improvement utility queue still requires measured cross-fitted expert
  intervention outcomes; synthetic targets cannot unlock it.

The strongest permitted statement is: **under nested patient-group development on
controlled MoNuSAC label changes, the frozen AANCA exclusion policy had wholly
positive registered macro-F1 intervals over unchanged and exact matched-random
training while meeting every registered important-class recall safeguard.**

Natural inconsistency and real-use claims remain governed by
`AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md`, `EXPERT_REVIEW_PROTOCOL.md` and
`PROSPECTIVE_WORKFLOW_PROTOCOL.md`.
