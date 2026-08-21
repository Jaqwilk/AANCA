# AANCA expanded development autoresearch protocol

**Study ID:** `monusac_aanca_expanded_development_v1`  
**Disposition:** post-lockbox, development-only method search  
**Project:** the existing AANCA implementation; this is not a replacement project or product v2

## Why this is a separate study

The first autoresearch study opened its one-time 15-patient internal lockbox. Its
selected candidate had positive point estimates but did not pass the registered
downstream confidence-interval gates. That lockbox is now permanently ineligible
for selection, tuning or confirmation. This expanded study starts from a new
protocol and evaluator identity, uses all official MoNuSAC training patients only
as development data, and has no internal confirmation claim.

The earlier lockbox outcome may motivate the broad research questions (more image
context and trusted-review weighting), but no row, probability, class outcome or
patient-specific result from it may enter candidate generation or selection.

## Fixed evaluator

Every candidate is evaluated by nested patient-group cross-fitting. The outer fold
holds out complete TCGA patients for downstream evaluation. Inside each outer
training set, the audit probabilities are generated in group-safe folds and never
score a nucleus using a model fitted on that nucleus or its patient. Controlled
corruption is applied only to outer-training labels; outer-validation labels remain
unchanged. Source annotations are never modified.

Candidate, unchanged and exact-matched-random conditions share the same feature
view, downstream model settings, outer fold and corruption seed. The exact random
control is disjoint and matched on observed class, organ and proposed transition.
The selector prospectively reserves enough members of every exact stratum to form
that comparator; candidates that cannot provide the registered control fail closed.

## Expanded representations

The fixed search compares 64-pixel nucleus context, 128-pixel tissue context,
their concatenation, label-independent colour/box statistics and Phikon-v2 frozen
features. Phikon-v2 is pinned to one public Hugging Face revision, exact weights
hash and preprocessing. Its licence limits use to non-commercial research. Because
Phikon-v2 pretraining included TCGA and MoNuSAC uses TCGA material, its development
result is explicitly contamination-limited and cannot count as independent external
evidence. A future confirmation cohort must be new and non-overlapping.

The representation never consumes the observed or reference class label. The
controlled corruption generator also remains representation-independent.

## Search questions

The bounded search compares:

1. probability, neighbour and fixed-hybrid audit risk;
2. global, balanced, diverse and rare-class-protective review queues;
3. review budgets from 0.5% through 10%;
4. verified-label restoration, abstention/downweighting and increased weight for
   reviewed labels;
5. audit and downstream regularisation and class weighting;
6. single-scale, multiscale and pathology-specific frozen representations.

Increased reviewed-label weight is allowed only after simulated review has exposed
the controlled reference label for that selected row. It never grants access to
unreviewed reference labels and never changes the stored source annotation.

## Success and selection

Ranking precision alone cannot win. A full candidate passes only when all registered
conditions hold under whole-patient bootstrap evaluation:

- top-K precision is better than the exact-matched-random queue;
- downstream macro-F1 is better than the unchanged condition;
- downstream macro-F1 is better than exact-matched-random review;
- every important class respects the `-0.01` recall non-degradation margin;
- both downstream differences are positive for every corruption seed.

Among candidates passing all gates, maximise the smaller downstream lower bound,
then point macro-F1 improvement, then retrieval lower bound, with a simplicity
tiebreak. If no candidate passes, the executable action remains
`retain_uncorrected`.

The selected object is a **development candidate**, not a confirmed production
policy. It may be implemented only behind the existing fail-closed policy guard.
It cannot replace the frozen negative external results or be advertised as proven
on natural pathologist errors.

## What would permit the requested real-use claim

No retrospective parameter search can prove that a pathologist made a biological
error. That claim requires newly collected natural cases, independent blinded
ratings, adjudication rules defined before review, multiple sites and a frozen
comparison of work with and without AANCA. Real-use superiority additionally
requires a prospective workflow endpoint and a newly untouched external task test.

Until both studies succeed, the strongest permitted wording remains “ranks
potentially inconsistent annotations for expert review.”
