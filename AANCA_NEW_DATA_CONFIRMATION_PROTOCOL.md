# Genuinely new-data confirmation protocol for AANCA

**Controlled compatibility status:** `EXTERNAL_VALIDATION_COMPLETE` on PUMA; all
seven frozen controlled gates passed  
**Natural-case status:** `INITIALISED` — no paired blinded expert outcomes exist  
**Frozen candidate:**
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`;
the external outcome may not change it

## Why another dataset is necessary

The MoNuSAC official test and both NuCLS subsets have already been opened. They can
still be audited and independently recalculated, but they cannot become a new final
test for a model selected after their outcomes were known. Repeating evaluation on
them would underestimate selection bias rather than confirm improvement.

The next test needs new patients or whole slides that were unavailable throughout
representation, score, budget, intervention and threshold selection. It also needs
qualified blinded reviewers if the endpoint is natural annotation inconsistency.
Dataset labels by themselves cannot establish that a pathologist made an error.

## Public-data route

The preferred public compatibility study is now the official PUMA release because
it has direct, lawful CC0 archives, one ROI per case, expert-checked nucleus labels
and no prior use in this project. Its complete prospective rules were frozen before
archive download in `AANCA_PUMA_NEW_DATA_PROTOCOL.md` and
`configs/puma_new_data_confirmation.yaml`.

PUMA remains a controlled-noise transfer study, not a natural-error study. The
official release contains final expert-checked labels but not paired pre/post natural
reviews. The frozen execution used 144 development and 62 final ROI/case groups.
AANCA exceeded unchanged corrupted training by `+0.006426` macro F1 with interval
`[+0.003657, +0.009365]` and exceeded exact matched-random exclusion by `+0.008067`
with interval `[+0.004093, +0.011947]`. The project-coupled PUMA evidence readback
passed; this was not third-party validation. Exact
results are in `reports/puma_new_data_confirmation_results.md`.

CoNIC/Lizard remains a possible future compatibility source, but it is not needed to
reinterpret or replace the successful frozen PUMA endpoint. Because PUMA outcomes
are now open, neither repeated PUMA analysis nor a newly tuned candidate may call
the same final groups independent confirmation.

The official NuCLS single-rater uncorrected and supervised-QC releases appeared to
provide a closer natural workflow endpoint, so their pairing rules were frozen in
`AANCA_NUCLS_SUPERVISED_QC_PROTOCOL.md`. Direct inspection then showed that the two
releases are different FOV quality tiers and the raw database retains one class
state per stable annotation element, not paired pre/post labels. That endpoint is
therefore explicitly unavailable rather than inferred from unmatched cohorts. The
feasibility result is in `reports/nucls_supervised_qc_feasibility.md`.

## Stronger real-use route

The evidence needed for the requested real-use statement is a new prospective
multi-site cohort, not merely another public benchmark:

1. obtain governance and data-use approval at at least three independent sites;
2. lock new patient/WSI identifiers and prevent every cross-split patient, slide or
   source overlap;
3. freeze the exact AANCA candidate, review budget and queue before review;
4. compare an AANCA queue with an exact matched-random or standard-practice queue at
   the same review budget;
5. collect at least two independent qualified pathology reviews per case, followed
   by separately blinded adjudication under a frozen rule;
6. preserve supported, probably inconsistent, ambiguous, insufficient-context and
   technical-exclusion outcomes instead of forcing binary truth;
7. train unchanged and reviewed-data models identically and evaluate them once on a
   site that was unavailable to every earlier decision.

Only a positive natural-case enrichment result supports the statement that AANCA
prioritises expert-judged inconsistencies in the studied population. Only a positive
prospective downstream comparison supports superiority over the unchanged model in
that frozen setting. Neither permits a biological-truth, diagnostic, patient-benefit
or unrestricted pathologist-error claim.

## Required final gates

- AANCA top-K beats the exact matched control with a whole-group 95% lower bound
  above zero.
- The frozen intervention model beats the unchanged model with its whole-group 95%
  lower macro-F1 bound above the registered minimum.
- No important-class recall lower bound falls below `-0.01`.
- Results remain positive across the registered sites/repeats, or heterogeneity is
  explicitly reported and the claim narrowed.
- Optimisation convergence, missing assets, exclusions and all reviewer disagreement
  are reported; no unavailable value is replaced by an estimate.
- Source annotations are never changed automatically.

The controlled PUMA gates have been executed and supported. Until the separate
natural reviewer and prospective workflow gates are executed, the executable
natural-data action remains `retain_uncorrected`.
