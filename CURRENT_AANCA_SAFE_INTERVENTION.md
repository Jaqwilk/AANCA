# Safe intervention policy for the current AANCA system

Status: **implemented; controlled development candidate selected and confirmed on
new PUMA data; natural-case and real-workflow evidence not yet executed**
Project: **the existing AANCA repository, not a replacement or “v2”**

The completed NuCLS result remains unchanged: the frozen ranking did not satisfy
both success gates, and the reviewed-label candidate reduced macro F1. The changes
below prevent the current system from repeating that unsafe decision. They do not
rewrite the adverse result or manufacture evidence that does not exist.

## New controlled-development result

A bounded autoresearch-style search was run only on the official MoNuSAC training
patients after the earlier internal lockbox had been opened and retired. The fixed
evaluator screened 240 ranking configurations and 160 downstream configurations,
then recomputed 12 frozen finalists under five outer patient folds, four audit folds,
four corruption seeds, five exact matched-random repetitions and 3,000 whole-patient
bootstrap draws. Two finalists passed every registered retrieval, downstream,
direction and class-safety gate.

The selected candidate is frozen in
[`configs/aanca_selected_development_candidate.yaml`](configs/aanca_selected_development_candidate.yaml):

- multiscale ImageNet ResNet-18 context at 64 and 128 pixels;
- fixed hybrid risk: 60% self-confidence and 40% fold-safe 31-neighbour disagreement;
- relaxed balanced 5% queue;
- `flag_exclude`, which gives selected cases zero training weight while leaving the
  saved source labels untouched;
- balanced downstream logistic regression with L2 `0.01`.

Across 44 development patients, its macro F1 was `0.547194` versus `0.504104` for
unchanged corrupted training and `0.492795` for the exact matched-random exclusion.
The differences were `+0.043090` (95% interval `[+0.032553, +0.055065]`) and
`+0.054399` (`[+0.034938, +0.075507]`). All four seed-level differences were
positive. The lowest important-class recall lower bound was `-0.006462`, above the
frozen `-0.01` limit. Retrieval precision was `0.947966`, exceeding matched random
by `+0.403950` (`[+0.360560, +0.448903]`).

A clean verification rerun reproduced those stored metrics exactly and recorded
convergence for all `220/220` fitted models (`100` hard-label and `120` weighted
fits). Its evidence SHA-256 is
`d10fbcb3179abe6058ae43231663f3aeefc7d754c40bac2bfa6fdea1a4abae38`.

Both registered development intervals excluded zero. This remains an adaptive
**controlled development** result, not an independent confirmatory or new external
test. The official MoNuSAC test, NuCLS outcomes and PanNuke final outcomes were
forbidden from selection. The candidate is implemented behind the fail-closed policy
but remains inactive on natural data until the expert-review protocol succeeds.

## New-source controlled confirmation and safety stress

The candidate was frozen before the official PUMA archives were acquired or any
PUMA score was calculated. On 144 development and 62 final ROI/case groups, all
seven prospectively frozen controlled-noise gates passed. The 5% AANCA queue had
precision `0.537739` versus `0.214379` for exact matched random. On untouched final
groups, `flag_exclude` improved macro F1 by `+0.006426` over unchanged corrupted
training (95% whole-group interval `[+0.003657, +0.009365]`) and by `+0.008067`
over exact matched-random exclusion (`[+0.004093, +0.011947]`). Every seed direction
was positive, all primary class-recall lower bounds stayed above `-0.01`, all 44
fits converged and the independent verifier passed.

This supports controlled-noise transfer on a genuinely new histopathology source.
It does not activate natural-data exclusion because PUMA provides final
expert-checked labels rather than paired natural pre/post review outcomes.

The primary benchmark stratified OOF groups with pre-corruption labels. A frozen
post-confirmation sensitivity rebuilt each seed's folds and exact neighbours using
only `observed_label`, which is available during a real audit. Only 22.21%-33.08% of
row-level fold assignments matched the primary plan, yet all seven sensitivity gates
passed. AANCA improved macro F1 by `+0.006679` over unchanged, interval
`[+0.004141, +0.009506]`, and by `+0.009069` over exact matched random, interval
`[+0.005855, +0.012461]`; every class-safety bound passed. This is exploratory
robustness evidence, not an independent second confirmation.

A post-confirmation exploratory stress reused the unchanged candidate across clean,
rare, directional, group-clustered and independent geometry-dependent controlled
scenarios. All nine macro-F1 lower bounds were positive against both unchanged and
matched-random training, but only the 10% group-conditional scenario passed every
class-safety gate. On clean labels, the heterogeneous `other` class recall fell by
`-0.013733` with interval `[-0.025390, -0.002789]` despite positive aggregate macro
F1. Therefore `flag_exclude` remains a controlled experimental intervention, not an
automatic natural-data action. The machine policy continues to return
`retain_uncorrected` until reviewer-gated, class-safe evidence exists.

## What changed in the current system

| Problem | Current AANCA behaviour | Implementation |
| --- | --- | --- |
| Inconsistency was treated like useful correction | Two independent queues: annotation quality and model improvement | `histo_audit.auditing.two_queue` |
| No evidence of per-case downstream gain | Model-improvement queue fails closed until a measured, nested group-cross-fitted utility estimate and positive lower bound exist | `histo_audit.evaluation.downstream_utility` |
| Global top-K could concentrate on one patient, class or transition | Deterministic caps for group, class, tissue and transition plus optional embedding-distance diversity | `build_two_review_queues` and `audit original --balanced-top` |
| Random control was not code-enforced as matched | Exact 1:1 stratum matching; no partial comparator; the blinded package validates equal top/random counts in every stratum | `draw_matched_random_comparator` and `external build-review-package --selection-plan` |
| A disputed case could become one hard label | Votes remain separate; ambiguity, soft labels, downweighting, exclusion and keep are first-class actions | `derive_review_interventions` |
| Hard changes were too permissive | Disabled by default; explicit opt-in, at least two votes and at least two-thirds agreement are required | `derive_review_interventions` |
| Only hard-label retraining was available | Unchanged, gated-hard, soft, downweighted and soft-plus-abstention strategies can be compared on disjoint development groups | `compare_review_training_strategies` |
| Positive macro F1 could hide damage to one class | Adoption requires a positive macro-F1 lower bound and a registered non-degradation bound for every important class | `evaluate_multicriteria_retraining_guard` |
| One model or one training moment could dominate | A 3-5 model, multi-checkpoint persistence signal rejects transient spikes | `persistent_group_safe_risk` |
| Calibration could reuse the scored cases | Temperature scaling is group-cross-fitted and accepts only newly collected expert development labels | `cross_fitted_temperature_calibration` |

All APIs retain source annotations unchanged. Derived labels, soft targets and
weights are separate arrays. A score is still called an
`annotation_inconsistency_score`, not `P(error)`, until a new expert-labelled
development study calibrates that interpretation.

## Two queues, two different questions

The quality-control queue asks which annotation deserves expert attention. It may
use only group-safe out-of-fold evidence and balancing rules. It may exist without a
claim that a correction will improve a downstream model.

The model-improvement queue asks whether reviewing a case has a positive expected
training effect. It remains unavailable unless prior development interventions have
produced measured per-case utility targets. A nested group-cross-fitted regressor
then predicts utility without using the target group and supplies a conservative
lower bound. Eligibility requires that lower bound to exceed the registered minimum.
The inconsistency score alone can never unlock this queue.

When those measured targets become available, the maintained product queue orders
eligible cases by
`percentile(annotation_inconsistency_score) × max(utility_lower_bound, 0)`.
This prevents a high inconsistency score with no conservative training benefit from
entering the model-improvement queue. The score percentile is still not called a
probability of error.

## Training and adoption

Independent development groups compare:

1. unchanged observed labels;
2. explicitly gated hard changes;
3. multi-rater soft labels;
4. downweighted hard targets;
5. soft labels with ambiguity/abstention weights.

The final external test is not an argument to this selection API. If no candidate
has a positive whole-group macro-F1 lower bound, or if any important-class recall
interval breaches the frozen tolerance, the selected action is
`retain_uncorrected`. Brier score and expected calibration error are reported for
every available strategy.

## Representation and model stability

The existing repository already supports exact target masks, target morphology,
wider context, ImageNet embeddings, a three-member group-safe ensemble and strict
availability gates for UNI, CTransPath or another pathology encoder. A pathology
encoder is not declared available merely because its name exists in a config: source,
licence, exact weights, authentication, preprocessing, hardware fit, intended use
and an embedding smoke test must all pass.

Encoder, ensemble, persistence and calibration choices must be made on fresh
group-separated development data. NuCLS is permanently excluded from this choice.
The selected controlled candidate uses no pathology foundation encoder, so it has no
known TCGA pretraining-overlap limitation. The Phikon-v2 finalist is retained in the
ledger but failed the rare-class safety gate and was not selected.

## Evidence that still requires people and new natural data

Software cannot supply qualified pathologists, additional patients, independent
sites or a new untouched test set. These remain unexecuted:

- at least two independent reviewers with raw votes preserved;
- a prospectively powered number of independent patients or WSI;
- at least three sites before a multi-site generalisation claim;
- a new external-site final test unavailable to every development decision;
- prospective comparison of AANCA-assisted and standard review.

The exact policy is machine-readable in
[`configs/current_aanca_intervention_policy.yaml`](configs/current_aanca_intervention_policy.yaml).
The measured-utility and natural-workflow designs are frozen separately in
[`AANCA_MEASURED_UTILITY_PROTOCOL.md`](AANCA_MEASURED_UTILITY_PROTOCOL.md) and
[`AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md`](AANCA_NEW_DATA_CONFIRMATION_PROTOCOL.md).
Their presentation-ready execution order and promotion ladder are consolidated in
[`NEXT_PHASE.md`](NEXT_PHASE.md).
The PUMA controlled study satisfied its simultaneous retrieval, downstream and
class-safety rule. The remaining natural study must satisfy the same structure using
blinded reviewer outcomes and a new external site. Until that study is executed,
AANCA is safer, reproducible and externally tested under controlled noise, but
real-world benefit remains unproved.
