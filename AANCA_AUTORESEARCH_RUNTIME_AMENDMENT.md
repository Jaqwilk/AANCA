# AANCA expanded autoresearch: frozen runtime amendment

Frozen: `2026-08-21T05:02:36.1601933+02:00`  
Amendment: `monusac_aanca_expanded_full_runtime_amendment_v1`

This is a computational-runtime amendment, not a scientific amendment. The original
expanded run remains immutable. It continues to record `timeout` under its original
420-second per-candidate rule.

## Trigger and timing

The first full nested candidate completed in 352.07 seconds. The second candidate,
which uses the 2,048-dimensional Phikon-v2 plus ResNet representation, crossed the
420-second limit before its outcome metrics were available. At the time this amendment
was frozen, only the first full candidate's result was known. The second and all later
full-candidate outcomes were unavailable.

The screening ledger had already deterministically selected all 12 full finalists.
Their ordered hashes are frozen in
`configs/aanca_autoresearch_full_runtime_amendment.yaml`; no candidate may be added,
removed, reordered or substituted.

### Clerical hash erratum

At `2026-08-21T05:40:05.5006626+02:00`, before the second full candidate's metrics
were available, a verification check found that the first candidate's manually copied
hash had the correct 12-character prefix but an incorrect suffix. It was replaced with
the exact hash already present in the immutable parent ledger:
`ba572cd4aed1367a723dd8f66b51961f48f66ff8791d021cbffe9ccb201b3ee6`.
The candidate mapping and order did not change. The amendment checksum was regenerated
after this documented clerical correction.

## Single permitted change

The equal per-candidate execution allowance is increased from 420 to 10,800 seconds.
The new limit was calculated without full-result metrics:

- slowest completed downstream screening trial: 273.68 seconds;
- maximum fit-count multiplier from the frozen screen design to the frozen full design:
  23.3333;
- runtime safety factor: 1.5;
- calculated allowance: 9,578.78 seconds, rounded upward to 10,800 seconds.

No representation, fold, patient assignment, corruption seed, matched-random draw,
bootstrap rule, metric, success gate, comparator, candidate or selection tie-break is
changed.

## Analysis rule

After all 12 parent-run calculations finish, an amended analysis may use their fully
computed records without recomputation only when every evaluator, dependency, config,
partition and candidate hash matches the parent authority and every elapsed time is at
most 10,800 seconds. All 12 frozen finalists must be included. Partial analysis and
cherry-picking are forbidden.

The parent ledger is never rewritten. For the amended analysis, the conjunction of the
already recorded individual `success_gates` is recomputed because the parent runner
sets the aggregate flag to false when it applies its original runtime timeout. No
scientific gate may be relaxed or replaced.

## Claim boundary

Any passing winner remains a development candidate. This experiment uses controlled
corruption on official MoNuSAC training data and does not evaluate natural pathologist
error, prospective workflow performance, diagnosis or clinical utility. The executable
action remains `retain_uncorrected` until a genuinely new, frozen external validation
and the separately registered blinded expert study succeed.
