# AANCA next phase

**Working name:** AANCA v2 research phase  
**Stage:** `INITIALISED`  
**Current natural-data action:** `retain_uncorrected`  
**Promotion target:** credible natural-case and prospective workflow evidence for
the same core annotation-auditing system

## Purpose

The current AANCA system has passed all registered gates in a frozen new-source PUMA
test under controlled label corruption. The protocol and result entered public Git
history together, so the repository does not independently timestamp the pre-outcome
order. It has not shown that it detects true natural
annotation errors, improves qualified expert review or is safe across sites. The
next phase is designed to answer those missing questions without tuning on opened
PanNuke, NuCLS, MoNuSAC or PUMA final outcomes.

“AANCA v2” is a working name for this evidence programme. It does not replace,
relabel or upgrade the status of the current results.

## Starting evidence

- PanNuke demonstrates reproducible group-safe controlled ranking, but the accepted
  primary analysis is exploratory and H4 is adverse.
- NuCLS provides genuine multi-rater disagreement but only five patient groups; its
  frozen ranking claim failed and guided correction reduced downstream macro-F1.
- MoNuSAC supports controlled retrieval, while its downstream and class-safety gates
  failed.
- PUMA supports controlled-noise transfer on 62 held-out case/ROI groups; all seven
  frozen gates passed.
- PUMA stress remains cautionary: aggregate downstream intervals were positive in
  nine scenarios, but only one passed every class safeguard.
- The official NuCLS resources do not provide stable paired nucleus-level pre/post
  labels needed for the intended prospective natural correction endpoint.

None of these opened outcomes may serve as the untouched final confirmation for a
new natural-case claim.

## Required programme

### 1. Build a new natural-case reference

Recruit independent qualified pathologists on previously unused cases. Each item
must preserve:

- blinded initial decisions before any AANCA suggestion is shown;
- reviewer identity, case/WSI identity and acquisition site;
- support, probable inconsistency, ambiguity, abstention, insufficient context and
  technical-exclusion outcomes;
- confidence and adjudication records without forcing disagreement into one label;
- immutable source labels and a separate reviewed-label state.

The final reference may be consensus or probabilistic multi-rater evidence, but it
must never be described as guaranteed biological truth.

### 2. Learn measured utility without leakage

Development must occur only inside nested patient- or WSI-group cross-fitting. The
candidate queue may combine:

`estimated inconsistency probability × conservative downstream utility`

Compare fixed budgets, hard correction, down-weighting, exclusion, soft labels and
no change. All selection must use development groups only. Every-class recall,
calibration, convergence and queue stability are mandatory, not optional secondary
plots.

### 3. Freeze one candidate

Before any new confirmation outcome is inspected, publish and checksum-bind:

- representation and preprocessing;
- group definition and split authority;
- queue score and calibration rule;
- intervention and review budget;
- ambiguity, abstention and underfilled-queue behaviour;
- matched-random controls and bootstrap unit;
- minimum aggregate effect and every-class safety thresholds;
- failure action `retain_uncorrected`.

No post-outcome candidate substitution or threshold rescue is permitted.

### 4. Run untouched external confirmation

Use new patients or WSIs from a source not used in candidate selection. A single
promotion decision requires all of the following:

1. ranking lower confidence bound above the exact matched-random control;
2. downstream macro-F1 lower confidence bound above unchanged labels;
3. downstream macro-F1 lower confidence bound above matched-random intervention;
4. every prespecified class recall lower bound above its safety margin;
5. all required fits converged and all source/group/split checks passed.

A favourable subset is reported but does not promote the candidate.

### 5. Evaluate real workflow utility

Run a prospective, multi-site, blinded comparison of expert review with and without
AANCA. Measure at least:

- time per reviewed case and total review burden;
- accepted changes, rejected suggestions, ambiguity and abstention;
- inter-rater agreement before and after adjudication;
- downstream model performance on untouched external patients/WSIs;
- every-class failures, site-specific failures and technical exclusions;
- user trust, override behaviour and unsafe automation attempts.

Clinical or operational claims require this step; a retrospective benchmark cannot
substitute for it.

## Promotion ladder

| Claim | Minimum new evidence |
| --- | --- |
| Prioritises natural disagreement | Frozen multi-rater ranking gate on untouched patient/WSI groups |
| Finds likely annotation inconsistencies | Blinded adjudication with ambiguity and abstention preserved |
| Improves a downstream model | Positive whole-group intervals versus unchanged and matched-random controls plus every-class safety |
| Improves expert workflow | Prospective multi-site with/without-AANCA comparison |
| Supports clinical use | Separate clinical validation, governance and regulatory work outside the current project |

Passing a lower rung does not imply a higher one.

## Stop and fail-closed rules

- If paired natural reference evidence is unavailable, stop and retain labels.
- If any required aggregate or class-safety gate fails, do not apply the candidate.
- If grouping cannot prevent patient/WSI leakage, the study is ineligible for the
  corresponding claim.
- If the untouched confirmation outcome is opened, it cannot be reused to tune and
  reconfirm the same candidate.
- If reviewer disagreement cannot be represented faithfully, do not force an error
  label.
- Source annotations are never overwritten automatically.

## Definition of completion

The next phase is not complete when code exists. It is complete only when the new
reference, prospective freeze, untouched confirmation, independent verification and
workflow evidence are all preserved with their failures. Until then, the public
project remains `EXTERNAL_VALIDATION_COMPLETE`, not `CONFIRMATORY_COMPLETE`, and the
natural-data action remains `retain_uncorrected`.
