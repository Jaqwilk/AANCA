# Prospective workflow protocol for the current AANCA system

Status: **designed but not executed; controlled PUMA transfer is complete but does
not replace this human workflow study**
System under study: the existing AANCA project, not a replacement or “v2”

Frozen development candidate: `78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`
(multiscale 64+128 px, fixed hybrid, balanced 5% queue, `flag_exclude`). This
candidate passed all nested controlled-development gates and all seven frozen PUMA
controlled new-source gates. It remains inactive on natural data and may not be
changed after prospective outcomes are opened.

This protocol defines the new evidence needed to test whether AANCA helps in a real
annotation-review workflow and whether reviewed changes improve a downstream model.
It cannot be completed with the saved PanNuke or NuCLS outcomes alone.

## Design

Use a prospectively registered, multi-site, assessor-blinded comparison of standard
review against review supported by a frozen AANCA queue. Recruit qualified reviewers
who did not develop the system and did not participate in method selection. Allocate
cases by patient or whole slide; no patient, slide, or source group may cross arms.

A cluster-randomised crossover design is acceptable when each reviewer performs both
conditions with separately allocated cases, balanced order and a prespecified washout.
At least three independent sites are required for a multi-site generalisation claim.
The protocol, sample-size calculation, model hash, ranking rule, review budget,
endpoints, exclusions and analysis code must receive a public timestamp before any
outcome is inspected.

## Intervention and control

- **Control:** the site's usual annotation-review process at the same review budget.
- **Intervention:** the same process with AANCA ordering cases for review.
- AANCA may display only “recommended for expert review”; it may not declare an
  error, prescribe a replacement label, or alter a source annotation automatically.
- Reviewers may retain the label, change it, mark it ambiguous, request more context,
  or abstain. Every raw response remains preserved.
- A separate blinded adjudication panel establishes the study reference after both
  arms are locked. Disagreement and insufficient context remain reportable outcomes,
  not forced truth labels.
- The controlled-development `flag_exclude` result does not authorise automatic
  exclusion in this workflow. A reviewer decision or a separately frozen study arm
  is required before a case receives zero training weight.

## Frozen outcomes

Primary workflow outcome:

- adjudicated potentially inconsistent annotations resolved per fixed review budget,
  comparing AANCA-assisted review with standard review using group-aware inference.

Key secondary outcomes:

- time per reviewed and per resolved annotation;
- reviewer agreement, ambiguity, abstention and technical-exclusion rates;
- false-escalation burden under the adjudicated reference;
- reviewer- and site-specific effects, labelled exploratory unless registered;
- safety events, including any attempted automatic or unreviewed label change.

Downstream outcome:

- train the same prespecified classifier separately on the locked labels produced by
  each arm, then compare macro F1 on a completely untouched external-site test set.
  Hyperparameters, class order and stopping rules must be identical. The test set is
  unavailable for reviewer allocation, model selection and the retraining decision.

The retraining candidate may be adopted only if the lower 95% whole-group bootstrap
bound for candidate-minus-uncorrected macro F1 exceeds the registered minimum effect
and every important-class recall interval remains above its registered
non-degradation limit. Otherwise the action is `retain_uncorrected`. This
multicriteria fail-closed rule is now implemented in AANCA.

Before the final test, use fresh group-separated development data to compare the
unchanged model, gated hard changes, soft labels, downweighting and soft labels with
abstention weights. Report macro F1, every class, Brier score and expected calibration
error. Calibration may use only newly collected expert development labels and must
be group-cross-fitted. The external-site test is unavailable to encoder, queue,
threshold, calibration and training-policy selection.

The annotation-quality queue and model-improvement queue remain separate. The latter
is unavailable until measured development interventions support nested
group-cross-fitted per-case utility estimates whose lower bounds exceed zero. A high
annotation-inconsistency score alone cannot authorise retraining.
Its frozen priority is the product of the audit-risk percentile and the positive
cross-fitted utility lower bound, as defined in
`AANCA_MEASURED_UTILITY_PROTOCOL.md`.

The post-confirmation PUMA stress gives a specific safety requirement for this
study. Positive aggregate macro F1 coexisted with an adverse recall interval for a
single class in eight of nine scenarios, including clean labels. Review capacity,
intervention and analysis must therefore preserve predeclared per-class and
proposed-transition caps, minimum retained counts and class-specific stopping
guards. A global positive metric can never override a breached class-recall bound.

## Claim boundary

A positive workflow result may support review-prioritisation and operational-utility
claims within the studied sites and population. A positive downstream result may
support that specific frozen training comparison. Neither result alone establishes
biological truth, diagnosis, clinical safety, patient benefit or unrestricted
generalisation. Patient-outcome claims require a separately governed clinical study.

## Execution checklist

- [ ] Obtain ethics, governance, data-use and site approvals.
- [ ] Recruit independent qualified reviewers and adjudicators.
- [ ] Freeze and publicly timestamp the current AANCA artifact and full protocol.
- [ ] Complete the prospective sample-size calculation.
- [ ] Lock patient/WSI groups, arm allocation and the external test site.
- [ ] Run both review conditions without automatic source changes.
- [ ] Lock reviewer responses before adjudication and unblinding.
- [ ] Apply the registered group-aware workflow analysis.
- [ ] Apply the fail-closed downstream retraining guard.
- [ ] Confirm that no important class breaches its frozen non-degradation limit.
- [ ] Publish de-identified outcomes, exclusions, code, hashes and adverse results.
