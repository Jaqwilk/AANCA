# Blinded natural-case expert-review protocol

Status: **protocol only; not yet executed**  
Scientific stage: no new completion stage is claimed

This protocol defines the evidence required before AANCA may make any claim about
naturally occurring annotation inconsistency. It does not reinterpret the completed
controlled-corruption benchmark and it does not permit automatic label changes.

## Objective

Test whether a frozen AANCA ranking enriches for nucleus annotations that qualified
human reviewers judge to be `probably_inconsistent`, compared with a matched random
sample from the same eligible source population.

The unit remains an already segmented nucleus annotation. The protocol does not
evaluate diagnosis, prognosis, missing nuclei, contour quality, or clinical utility.

## Prerequisites and freeze

Before looking at reviewer responses, publish an immutable protocol record containing:

- the dataset identity, licence basis, inclusion and exclusion rules;
- the strongest available grouping identifiers (patient, WSI, then source patch);
- the exact eligible original-label audit run and ranking hash;
- top-sample size, matched-random sample size, strata, and random seed;
- the primary and secondary estimands, analysis code, and multiplicity rule;
- the reviewer qualification criteria and planned number of reviewers;
- the handling of missing context, abstention, disagreement, and technical exclusions.

The sample size must be justified by a prospective precision or power calculation. It
must not be chosen after examining responses. Reliable patient or WSI identifiers are
required for patient- or slide-level independence claims; source-patch grouping alone
must be named explicitly when stronger identifiers are unavailable.

## Sampling

1. Start only from non-injected, original-label cases with complete reviewer assets.
2. Select the frozen top-K queue without inspecting natural-case responses.
3. Draw an equal-sized random comparator from the remaining eligible population.
4. Match or stratify the comparator on prespecified tissue, observed class, and source
   group variables where available.
5. Record every inclusion, exclusion, duplicate, and asset failure before review.
6. Use group-aware inference at the strongest reliable patient/WSI/patch level.

Top-ranked and random cases may not be replaced after review starts except under the
prespecified technical-exclusion rule. Replacements retain the original frozen order
and are recorded in an append-only exclusion log.

## Reviewers and blinding

- Use at least two independent qualified pathology reviewers. Record qualification
  criteria without publishing personal identifiers.
- Reviewers receive the same cases in independently randomised order.
- They must not see audit score, rank, cohort assignment, model prediction,
  `pre_corruption_label`, corruption metadata, source identifiers, or another
  reviewer's response.
- Each item must include the full source patch, target crop, exact target contour, and
  the visible observed label. Insufficient context remains a valid outcome.
- A third qualified reviewer may adjudicate only under a prospectively specified
  policy. Raw independent responses remain primary evidence and are never overwritten.

The maintained package builder enforces reviewer-facing blinding and stores the
private unblinding key outside the review package.

## Allowed responses

The response vocabulary matches the maintained package schema:

- `annotation_supported`
- `probably_inconsistent`
- `ambiguous`
- `insufficient_context`
- `exclude_technical_reason`

Free-text notes are optional. Reviewers may abstain through `ambiguous` or
`insufficient_context`; no response may be coerced into a binary truth label. The term
`probably_inconsistent` means recommended for expert review, not proven pathologist
error.

## Outcomes and analysis

Primary outcome:

- difference in the proportion of `probably_inconsistent` responses between the
  frozen top-K cohort and matched random cohort, with a two-sided 95% confidence
  interval clustered at the strongest reliable source-group level.

Secondary outcomes:

- cohort risk ratio and enrichment at the registered review budget;
- reviewer-specific estimates and a prespecified multi-rater agreement statistic;
- rates of `ambiguous`, `insufficient_context`, and technical exclusion;
- tissue- and observed-class subgroup estimates labelled exploratory unless frozen as
  confirmatory;
- sensitivity analyses under unanimous, majority, and any-reviewer definitions,
  without replacing the raw categorical outcome.

Use the frozen estimator and resampling seed. Apply Holm correction to the registered
family of secondary hypothesis tests. Report missing or non-estimable results as
unavailable; do not impute reviewer agreement or silently discard disagreement.

## Evidence package

Retain and publish, subject to dataset licence and privacy restrictions:

- the immutable protocol and its public timestamp;
- hashes of the eligible audit run, ranking, manifest, and reviewer assets;
- blinded item IDs and cohort-independent display order;
- blank response schema and package-validation result;
- de-identified raw independent responses;
- private linkage under controlled access when public release is prohibited;
- exclusion log, analysis environment, executable analysis, and final statistics;
- a statement of every unavailable identifier or unexecuted analysis.

Image data must not be republished when the source licence forbids it. In that case,
publish hashes, generation instructions, and non-sensitive derived tables sufficient
for an authorised researcher to reconstruct the package.

## Stage and claim rules

- Building and validating a genuine blinded package may support
  `EXTERNAL_VALIDATION_READY` only when all repository eligibility gates pass.
- `EXTERNAL_VALIDATION_COMPLETE` requires genuine completed multi-rater responses,
  locked analysis, and published evidence. Synthetic or generated responses are
  forbidden.
- A positive enrichment result would support review prioritisation only. It would not
  establish diagnosis, clinical benefit, biological truth, or permission to modify
  source annotations automatically.
- A null or adverse result is retained and reported without changing the benchmark.

## Execution checklist

- [ ] Obtain lawful data access and reliable grouping identifiers.
- [ ] Freeze and publicly timestamp the natural-case protocol and analysis.
- [ ] Complete the prospective sample-size calculation.
- [ ] Produce an eligible original-label audit ranking.
- [ ] Build and checksum-validate the blinded review package.
- [ ] Recruit qualified independent reviewers and document consent/governance.
- [ ] Lock responses before unblinding cohort assignment.
- [ ] Run the frozen group-aware analysis and publish all eligible outcomes.
- [ ] Update `STATUS.md` only with an allowed completion stage supported by evidence.

