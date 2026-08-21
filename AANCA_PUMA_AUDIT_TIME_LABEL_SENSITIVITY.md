# PUMA audit-time-label sensitivity protocol

**Freeze date:** 2026-08-21 (Europe/Warsaw)  
**Disposition:** exploratory sensitivity after the frozen PUMA primary result and
realism stress were opened; it cannot select, rescue or change the candidate  
**Candidate:** `78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`

## Reason

The internally frozen PUMA benchmark stratified development OOF groups using the
pre-corruption reference labels. This remained group-safe and never exposed final
groups, but natural use has no known pre-corruption truth. The relevant deployment
sensitivity is therefore to allocate every OOF fold using only the observed label
available at audit time.

## Frozen change

Repeat the primary PUMA 10% symmetric controlled experiment under the same four
corruption seeds. For each seed, build the four group-stratified OOF folds from that
seed's `observed_label` vector. Recompute exact 31-neighbour references from the
corresponding fold training groups. The complete query ROI/case remains excluded
from its OOF model and neighbour set.

Do not change the 64+128 px representation, audit regularisation, fixed hybrid
weights, balanced-relaxed 5% queue, exact matched-random strata, `flag_exclude`
intervention, downstream model, final groups, metrics or class-safety margin.

## Evaluation and boundary

Use 3,000 whole-ROI/case bootstrap replicates and the same simultaneous retrieval,
downstream, seed-direction, every-class recall and convergence gates as the PUMA
primary study. Store the observed-label fold identities and per-seed neighbour
identities.

Because all PUMA outcomes are already known, a positive result can show only that
the original controlled conclusion is insensitive to audit-time fold allocation.
It is not a second confirmation, does not establish a natural error, and cannot
authorise automatic source changes or unreviewed natural-data exclusion.
