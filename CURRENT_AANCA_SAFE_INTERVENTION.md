# Safe intervention policy for the current AANCA system

Status: **implemented as software and protocol; not yet executed on fresh evidence**  
Project: **the existing AANCA repository, not a replacement or “v2”**

The completed NuCLS result remains unchanged: the frozen ranking did not satisfy
both success gates, and the reviewed-label candidate reduced macro F1. The changes
below prevent the current system from repeating that unsafe decision. They do not
rewrite the adverse result or manufacture evidence that does not exist.

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

## Evidence that still requires people and new data

Software cannot supply qualified pathologists, additional patients, independent
sites or a new untouched test set. These remain unexecuted:

- at least two independent reviewers with raw votes preserved;
- a prospectively powered number of independent patients or WSI;
- at least three sites before a multi-site generalisation claim;
- a new external-site final test unavailable to every development decision;
- prospective comparison of AANCA-assisted and standard review.

The exact policy is machine-readable in
[`configs/current_aanca_intervention_policy.yaml`](configs/current_aanca_intervention_policy.yaml).
Success requires all three conditions simultaneously: top-K beats its exact matched
control, the intervention model beats the unchanged model, and no important class
breaches its registered non-degradation limit. Until that study is executed, AANCA
is safer and more testable, but real-world benefit remains unproved.

