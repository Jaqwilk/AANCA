# Measured downstream-utility protocol for AANCA

**Execution status:** `INITIALISED` — protocol and software path only; no new
expert-reviewed utility outcomes have been collected  
**Project:** the existing AANCA implementation, not a replacement or “v2”

## Purpose

The annotation-quality queue and the model-improvement queue answer different
questions. A high annotation-inconsistency score can justify expert inspection, but
it does not show that changing, downweighting or excluding that case will improve a
later model. The model-improvement queue therefore remains unavailable until the
effect of review interventions has been measured on patient-disjoint development
outcomes.

The queue priority is frozen as:

`percentile(annotation_inconsistency_score) × max(utility_lower_bound, 0)`

The first factor is a ranking factor, not a probability of pathologist error. The
second factor is a conservative, nested group-cross-fitted estimate of downstream
benefit. A case is eligible only when its utility lower bound is strictly above the
registered minimum.

## Evidence source

Utility targets must come from newly collected, independently reviewed development
cases. Each case retains all raw reviewer votes and one of the registered actions:
keep, soft label, downweight, exclude, or an explicitly gated hard change. Synthetic
corruption may be used only as a mechanism test. It cannot unlock the natural-case
queue and cannot support a claim about pathologist error.

No outcome from NuCLS, the MoNuSAC official test, the opened MoNuSAC internal
lockbox, PanNuke final folds, or a future final external test may be used to train,
calibrate or choose this queue.

## Nested measurement design

1. Split by the strongest available group: patient, then whole slide, and only then
   source patch. No group crosses any training, utility-measurement or evaluation
   boundary.
2. Hold out one outer group fold for the downstream comparison. It is unavailable
   to audit fitting, utility target construction, queue fitting and threshold choice.
3. Within the outer-training groups, construct inner utility-measurement folds.
   Compare each registered intervention with unchanged training while evaluating
   both models on disjoint inner-validation groups.
4. Measure intervention units prospectively defined by source group, observed class,
   proposed transition and risk bin. Do not manufacture a per-nucleus target by
   copying one global model difference onto unrelated cases.
5. Fit the utility estimator only to intervention units whose outcome was measured
   outside their source group. Cross-fit its predictions again by source group and
   form one-sided conformal lower bounds from inner residuals.
6. Build the model-improvement queue using the frozen product priority, registered
   class/group/tissue/transition caps and an exact matched-random control.
7. Apply the queue intervention only to the outer-training data and compare with the
   unchanged and matched-random conditions on the untouched outer fold.

The primary utility target is the change in patient-balanced validation log loss per
review decision because it is continuous and does not force a noisy binary label.
Macro-F1 and every class recall remain adoption outcomes. The target definition,
unit size and minimum lower bound must be frozen before any intervention outcome is
opened.

## Candidate actions and budgets

Fresh development data compare review budgets of 2.5%, 5%, 7.5% and 10% and the
following actions under identical feature, model and split settings:

- retain the observed label;
- exclude the reviewed training case;
- downweight it to 0.5;
- use the independent reviewers' soft label distribution;
- use a hard reviewed label only when the prospective multi-rater gate permits it.

The controlled-development winner (`flag_exclude` at 5%) is the first candidate,
not a privileged conclusion. The utility-guided queue must outperform both the
unchanged model and an exact matched-random intervention under the same budget.

## Fail-closed adoption

All of the following are required in nested development and then once on genuinely
new external data:

- the whole-group 95% lower bound for macro-F1 improvement over unchanged is above
  the registered minimum;
- the lower bound over the exact matched-random intervention is above zero;
- every corruption/randomisation seed or prospectively registered repeat has the
  same positive direction;
- no important class recall lower bound is below `-0.01`;
- all fitted models pass the registered optimisation/convergence check;
- the source annotations remain immutable.

Otherwise the executable action is `retain_uncorrected`. Even a fully positive
retrospective result permits only a frozen candidate for prospective validation.
Claims about natural pathologist disagreement require the blinded expert protocol;
claims about real workflow superiority require the prospective workflow protocol.

