# Primary and confirmatory preregistration: execution-ready definition

**State:** READY_FOR_FREEZE

**Completion status:** `PILOT_COMPLETE`.

This document fixes the scientific design selected after the eligible PanNuke pilot
and before inspection of any primary-study or final-reference outcome. It is an
execution-ready pre-freeze authority candidate. The required representation-cache
provenance bundle and representation-independence record have been generated,
independently verified, and hash-bound below. No primary-study or final-reference
outcome was inspected while resolving these authorities.

## Primary design

The study asks whether a source-group-safe automated auditor ranks intentionally
injected nucleus-class corruptions more efficiently than random expert review and
whether restoring an equal review budget selected by the auditor improves downstream
nucleus classification. It is a controlled annotation-consistency benchmark, not a
diagnostic system and not evidence that a pathologist or source annotator was wrong.
Source annotations are never automatically changed.

The controlled positive event is `is_injected_corruption == true`. The model receives
only `observed_label`. The fields `pre_corruption_label`, `observed_label`,
`is_injected_corruption`, `restored_label`, `corruption_type`, `original_class`,
`replacement_class`, `corruption_seed`, generator representation, and configuration
hash remain permanently distinct. Hidden reference fields are available only to
controlled corruption evaluation, simulated restoration, split allocation as
specified below, and final downstream evaluation after all choices are fixed.

The hypotheses and their prespecified evaluations are:

- **H1:** group-safe OOF auditing outperforms random review. The formal primary
  comparisons are `self_confidence - random_review` in the 12 ImageNet-context,
  multinomial-logistic cells at 10% corruption: four mechanisms x seeds 404, 405,
  and 406. The metric is average precision and the operational budget is 5%; all
  comparisons belong to Holm family `h1_method_vs_random`.
- **H2:** performance varies by corruption mechanism, nucleus class, and tissue.
  Prespecified class-, tissue-, mechanism-, and rate-specific average precision is
  reported for every supported primary cell. A subgroup receives AP only with at
  least 100 samples and 10 injected corruptions; otherwise only support and event
  counts are reported. These are heterogeneity estimates, not claims of clinical
  realism.
- **H3:** confusion-targeted and instance-dependent corruption are harder than
  symmetric corruption. At 10%, matched ImageNet-context logistic cells compare
  symmetric minus confusion-targeted and symmetric minus instance-dependent AP for
  each seed 404, 405, and 406. The six comparisons use
  `h3_mechanism_hardness`.
- **H4:** at the same 5% review budget, `audit_guided_restoration` improves final-fold
  macro F1 more than `random_review_restoration`. The exact primary restoration cell
  is `primary_0027_8531672acd3c`: symmetric corruption, rate 10%, seed 404,
  `imagenet_resnet18_highlighted`, and multinomial logistic regression.
- **H5:** the fixed hybrid may improve ranking. In each of the 12 mechanism x seed
  ImageNet-context logistic cells at 10%, the comparison is
  `fixed_hybrid - self_confidence` AP, in family `h5_fixed_hybrid`. The confirmatory
  study additionally compares the fixed hybrid with each registered drop-one
  ablation.
- **H6:** a verified pathology-specific representation may outperform ImageNet.
  Primary comparisons are pathology minus ImageNet context at symmetric 10% for
  seeds 404, 405, and 406, in family `h6_encoder_family`; the confirmatory matched
  comparison is in `encoder_family`. The frozen availability audit selected no
  eligible pathology encoder, so these optional comparisons will be recorded as
  unavailable unless the predefined availability rule is satisfied before freeze.
  No substitute encoder is selected from study outcomes.
- **H7:** explicit target indication may reduce neighbouring-nucleus shortcut risk.
  Primary comparisons are highlighted minus context ImageNet logistic AP at
  symmetric 10% for seeds 404, 405, and 406, in family
  `h7_target_indication`. Confirmatory target-representation comparisons are fixed
  below.

## Dataset and split

- Dataset: the verified official PanNuke release, positive class order
  `[neoplastic, inflammatory, connective_soft_tissue, dead,
  non_neoplastic_epithelial]`, encoded as `[0, 1, 2, 3, 4]`.
- The only admissible full analysis manifest has SHA-256
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`,
  exactly 188,333 analysis-eligible rows, and canonical eligible sample-order
  SHA-256
  `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`.
  Both study configs bind the same three values; cache construction and the freeze
  cross-config gate fail closed if any authority differs.
- Primary development folds: official folds 1 and 2. Official fold 3 is the untouched,
  uncorrupted final reference fold for the primary study.
- Grouping unit: `source_patch_id`, exposed to algorithms as `group_id`. Every split,
  OOF fold, bootstrap, reference-validation selection, and train/evaluation boundary
  is source-patch-group-safe. Released metadata do not support patient- or WSI-level
  independence, so no stronger independence claim is made.
- Development reference validation: 10% of eligible development source-patch groups,
  selected once by
  `deterministic_group_greedy_class_distribution_v1` with split seed 223. It remains
  uncorrupted. All other eligible development groups form the audit pool; corruption
  occurs only there.
- Controlled-benchmark OOF: five folds, allocated before corruption using group-safe
  `StratifiedGroupKFold`, split seed 223, with `pre_corruption_label` used only for
  allocation. If any deterministic stratified training partition lacks a fixed
  class, seeded `GroupKFold` is the only permitted fallback and its reason is saved.
  There is no nucleus-level random fallback. The same group partition is reused for
  every rate, mechanism, corruption seed, representation, classifier, and audit
  method.
- Every audit nucleus receives exactly one probability vector from a model trained
  without that nucleus and its entire group. Training, class weights, Cleanlab,
  neighbours, and risk scores use `observed_label`, never
  `is_injected_corruption`. Probabilities retain class order `[0, 1, 2, 3, 4]`, sum
  to one, and are accompanied by train/holdout group identifiers.
- An exploratory audit of naturally occurring annotation inconsistency must allocate
  folds using `observed_label`, because a clean reference is not available there.
  The controlled use of `pre_corruption_label` for fold allocation must not be
  generalized as real-world error-detection performance.

The canonical M6 pilot used only a deterministic 250-group development subset
(selection seed 211) for feasibility. That pilot subset limit does not apply to the
primary matrix, which uses the complete eligible folds-1/2 population.

## Exclusions and missing data

The raw ZIP and NPY files are immutable. Validation, cache construction, corruption,
and evaluation do not repair, overwrite, relabel, split, or merge a source identity.
One eligibility mask is computed before splitting and is identical for every primary
and required confirmatory cell.

- Positive channels are 0-4. Channel 5 is only the supplied background channel and
  is not required or reconstructed to be the complement of the positive channels.
- Pixels in neither a positive channel nor supplied background remain unlabeled
  `void`. They are never promoted to background or a nucleus class. Canonical QC
  found 10,486,091 void pixels in 162 patches: 2,359,296 / 3,801,371 / 4,325,424
  pixels in folds 1/2/3.
- A pixel present in more than one positive class is `cross_class_overlap`. No class
  arbitration, precedence, majority vote, or outcome-informed resolution is
  permitted. Canonical QC found 4,318 such pixels in 575 patches: 1,216 / 1,572 /
  1,530 pixels and 194 / 190 / 191 affected patches in folds 1/2/3.
- Every source nucleus touching at least one cross-class-overlap pixel is retained in
  provenance, marked ineligible for primary and confirmatory outcome analysis, and
  assigned the exact reason `touches_cross_class_overlap`. The fixed exclusion covers
  1,411 instances: 471 / 465 / 475 in folds 1/2/3. An affected patch remains flagged;
  unaffected instances in that patch remain eligible unless another fixed rule
  applies.
- A positive-plus-supplied-background conflict is retained with a QC flag and no
  source-mask modification or class arbitration. It is not by itself an instance
  exclusion. Canonical full-release QC found zero such pixels, patches, and touching
  instances; the rule remains fixed if later integrity verification detects one.
- A `disconnected_instance_id` is retained as one raw
  `(fold, patch, class, instance_id)` identity and flagged. It is not automatically
  excluded, split, merged, or treated as download corruption. Border instances are
  likewise retained with a flag.
- Structurally invalid mask shapes/types, malformed required arrays, missing required
  fields, or incomplete release folds fail closed at the dataset gate. Required
  model inputs are not silently imputed. An unavailable optional pathology encoder
  produces a recorded unavailable cell, not fabricated values.
- Exact, pHash, and frozen-ResNet duplicate candidates are review-only flags and are
  not automatically deleted or reassigned. The full audit covered all 20,798,326
  cross-fold pairs per signal and found 0 exact, 121 pHash, and 0 frozen-ResNet
  candidates at the registered thresholds.

Derived target masks use
`nearest_per_component_with_forward_fallback_v1`. Each 4-connected component is
projected separately by nearest neighbour; the deterministic forward-footprint
fallback must preserve at least one unique 4-connected output pixel when nearest
projection would lose or disconnect a component. Raw component count, crop geometry,
per-component contribution, fallback use, collisions, new adjacency, and projected
component count are saved. Projection changes are reported and never written back.
Primary frozen-feature crops use output size 64, padding 8, and highlighted-context
brightness 0.45, matching the canonical M6 recipe; the exact recipe and implementation
hashes must be bound by the required cache sidecars. Confirmatory pixel inputs use the
registered 224-pixel preprocessing recipes.

## Corruption

The clean reference is one 0% cell represented as
`symmetric_random_corruption`, seed 404, with no replacements. Active primary rates
are 5%, 10%, and 20%, crossed with seeds 404, 405, and 406 and all four mechanisms
below. Counts use `round_half_up`, and replacement by the original class is forbidden.
Thus the primary plan contains 36 active corruption scenarios plus one clean
scenario. At 0%, AP, AUROC, recall, lift, and injected-event comparisons are
`not_applicable`; score distributions, review counts, and false alerts are reported.

1. `symmetric_random_corruption`: choose uniformly among the four alternative
   classes.
2. `confusion_targeted_corruption`: use the following row-normalized matrix in class
   order `[0, 1, 2, 3, 4]`; diagonal values are zero. It was derived from development
   clean-OOF confusion counts with one pseudocount per off-diagonal entry and carries
   no clinical-realism claim.

```text
[[0,                   0.13590033975084936, 0.3114382785956965,  0.1143827859569649,  0.43827859569648925],
 [0.11691542288557213, 0,                   0.23383084577114427, 0.2263681592039801,  0.4228855721393035 ],
 [0.344,               0.2784,              0,                   0.0736,              0.304              ],
 [0.2638888888888889,  0.4861111111111111,  0.2222222222222222,  0,                   0.027777777777777776],
 [0.4087403598971722,  0.26735218508997427, 0.3059125964010283,  0.017994858611825194, 0                  ]]
```

3. `group_conditional_corruption`: use `tissue_type` and the following fixed weights;
   default weight is `0.6795918743737025`.

```text
Adrenal_gland=1.0; Bile-duct=0.6221625801755902;
Bladder=0.4827071629213483; Breast=0.7740885942931048;
Cervix=0.5083850410867012; Colon=0.7693324520819563;
Esophagus=0.4320735444330949; HeadNeck=0.7671200473092844;
Kidney=0.6053713346122226; Liver=0.8550026752273943;
Lung=0.6762703337246353; Ovarian=0.5372740595994138;
Pancreatic=0.41691306918982846; Prostate=0.673852254817225;
Skin=0.6767244586650792; Stomach=0.3645222233270359;
Testis=0.5069662921348315; Thyroid=0.7704325073110666;
Uterus=0.6337078651685394
```

4. `instance_dependent_corruption`: hard cases and plausible neighbour alternatives
   are generated in `morphology_only_v1`. Auditor/generator independence is evaluated
   per representation. The exact matrix verifies the ImageNet context and highlighted
   auditor spaces as `verified_independent`. Engineered morphology is marked
   `not_independent` and `circularity_risk`; it is excluded from confirmatory claims
   about instance-dependent corruption. Any unverified or identical space is reported
   separately and excluded from confirmatory claims. The strict schema-v2 matrix is
   `reports/representation_independence.json`, SHA-256
   `846f421284de381401761a8dc4ceb108d3f3f2a0eece379706be7f7a512789c7`.

The exact pilot-derived parameter authority is
`reports/pilot_derived_primary_parameters.json`, SHA-256
`8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b`.
Its producer is `pilot_derived_primary_parameters_v1`; producer-source SHA-256 is
`6c1c2cf98d68e70509751d5fc6468823859ae8ff28c48157ee922bc850488a24`.
It is bound to eligible run
`20260718T143216.354310Z_pannuke_pilot_c7797330e0`, artifact root
`37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`.
Only folds-1/2 development evidence was used to derive the transition matrix and
tissue weights; no final-fold sample identity, label, representation, or outcome was
read.

## Representations and models

The primary expansion has 222 planned cells: 185 required and 37 optional pathology
cells. Five required model cells occur in each of the 37 corruption scenarios; the
pathology cell is optional.

- `engineered_target_features`: context RGB with target-specific morphology,
  colour, HOG, and texture features; required; multinomial logistic regression.
- `imagenet_resnet18_context`: context RGB, official torchvision
  `resnet18_imagenet1k_v1`; required; multinomial logistic regression and small MLP.
- `imagenet_resnet18_highlighted`: target-highlighted RGB with context retained and
  no class information encoded; same ImageNet encoder; required; multinomial
  logistic regression and small MLP.
- `pathology_encoder_optional`: context RGB and multinomial logistic regression;
  optional. The frozen-priority audit selected no candidate because no candidate
  satisfied every source, licence, weights, authentication, preprocessing, hardware,
  intended-use, and smoke-test gate. Audit SHA-256:
  `5e568cf29e489d8948bfcd33feae5b292cb48837eb4c93754202a565778a6e4a`.
  Its absence does not block the required ImageNet benchmark and is reported without
  an H6 result.

Official ImageNet ResNet-18 weight SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Each required cache must bind the exact encoder identifier and implementation,
weights, preprocessing, crop/projection recipe, dataset-manifest identity, sample
order, array bytes, and semantic sidecar. All samples in a shared cache have one
canonical order; reordered, missing, duplicated, or extra samples fail closed.

Primary multinomial logistic regression uses L2 `0.01`, `max_iter=400`, balanced
class weights calculated only from observed development labels, and model seed 227.
The small MLP uses one 64-unit hidden layer, dropout 0.10, 30 epochs, batch size 64,
learning rate 0.001, weight decay 0.0001, model seed 229, AMP float16, and gradient
accumulation 1; fitting labels are observed development labels only. Primary
probability calibration is disabled (`method: none`). Reference validation is the
only permitted source if a future amendment were to enable calibration; final-fold
calibration is prohibited.

## Audit methods

Every risk score increases with annotation suspicion. The primary method is
`self_confidence = 1 - P(observed_label)`. Registered methods are:

- self-confidence;
- clipped negative log-likelihood;
- alternative-minus-observed prediction margin;
- predictive entropy as an ambiguity baseline;
- stable Cleanlab using only group-safe OOF probabilities;
- fold-safe nearest-neighbour disagreement with `k=7`, cosine distance, exclusion of
  the sample and every member of its source group, and neighbours only from valid
  reference groups;
- a fixed hybrid of percentile-normalized self-confidence and nearest-neighbour
  disagreement with weights `[0.5, 0.5]`.

Cleanlab failure yields a missing value and recorded blocker; no Cleanlab value is
invented. The primary comparison does not select methods or hybrid weights from
outcomes.

Confirmatory ensemble disagreement uses the three matched `cnn_context_rgb` models
with seeds 303, 304, and 305. Its primary risk is mean pairwise Jensen-Shannon
divergence; predictive entropy of the mean and variation ratio are secondary. The
confirmatory fixed hybrid percentile-normalizes self-confidence and ensemble
disagreement with weights `[0.5, 0.5]`; both drop-one ablations are mandatory.

## Metrics and statistics

- Primary ranking metric: average precision/AUPRC.
- Primary review budget: 5%. Secondary budgets: 1%, 10%, and 20%.
- Additional ranking outputs: AUROC, precision, recall, lift, reviewed count,
  corruptions found, random expectation, score distributions, and class/tissue/
  mechanism/rate AP subject to the subgroup support rule.
- Random review uses exactly the same integer budget, deterministic tie handling,
  100 repetitions, and seed 419 for ranking comparisons.
- Primary paired group bootstrap: 2,000 iterations, seed 431, resampling complete
  source-patch groups identically across operands. Report the paired difference,
  two-sided 95% percentile interval, direction, and probability that the difference
  is positive.
- Confirmatory paired group bootstrap: 2,000 iterations and seed 439, with the same
  matched-group principle across registered folds/seeds/scenarios.
- Primary Holm families are `h1_method_vs_random`,
  `h3_mechanism_hardness`, `h5_fixed_hybrid`, `h6_encoder_family`, and
  `h7_target_indication`. Confirmatory families are
  `target_representation_family`, `ranking_method_family`, and `encoder_family`.
- Accuracy, macro F1, balanced accuracy, per-class precision/recall, confusion
  matrices, and applicable calibration metrics are saved for downstream models.
  Neutral, adverse, missing, and infeasible results are retained.

All values in reports must be read from sealed machine-readable artifacts. A failed,
skipped, optional, or circularity-excluded cell remains explicit and cannot be
reported as executed.

## Restoration

Only reviewed injected corruptions are restored from `observed_label` to
`pre_corruption_label`; all unreviewed labels remain unchanged. Restoration never
modifies the source manifest or raw masks. The required conditions are exactly:

1. `uncorrupted_reference_baseline`;
2. `corrupted_observed_baseline`;
3. `random_review_restoration`;
4. `audit_guided_restoration`.

Primary restoration uses cell `primary_0027_8531672acd3c`, self-confidence ranking,
5% review, 100 random repetitions, random seed 433, and includes the uncorrupted
reference-validation partition in downstream training. All four conditions use the
same representation, classifier, hyperparameters, seed, review count, and untouched
official-fold-3 test. The primary downstream comparison is
`audit_guided_restoration - random_review_restoration` macro F1.

Confirmatory restoration uses
`imagenet_frozen_target_highlighted_logistic`, representation
`imagenet_target_highlighted_embeddings`, model seed 303, fixed-hybrid ranking, 5%
review, 100 random repetitions, and random seed 443. It is repeated under the fixed
confusion-targeted 10% cell for each official-fold rotation with the same four
conditions and equal budgets.

## Confirmatory matrix and fold rotation

The confirmatory matrix is fixed before primary outcomes. It contains official folds
`[1, 2, 3]`, two corruption cells, six scenarios, and model seeds `[303, 304, 305]`,
for 108 planned cells. Ninety cells are required and 18 pathology cells are optional.

Corruption cells are:

- `clean_reference_cell`: symmetric mechanism, 0%, seed 404;
- `confusion_targeted_ten_percent`: the exact registered transition matrix above,
  10%, seed 404.

Scenarios are:

1. required `cnn_context_rgb`, ImageNet ResNet-18 CNN softmax head;
2. required `cnn_context_target_mask`, RGB plus a fourth binary target-mask channel,
   with zero-initialized fourth-channel weights;
3. required `imagenet_frozen_logistic`, context embeddings plus logistic regression;
4. required `imagenet_frozen_target_highlighted_logistic`;
5. required `imagenet_frozen_context_morphometrics_logistic`;
6. optional `pathology_frozen_logistic`, unavailable under the frozen audit unless
   its predefined availability gates are met before freeze.

CNN controls are AdamW, input size 224, learning rate 0.0001, weight decay 0.0001,
maximum 100 epochs, reference-validation-only early stopping with patience 10 and
minimum delta 0.0001, initial batch size 128, gradient accumulation 2, balanced class
weights, AMP float16, checkpoints/resume, and CUDA required. OOM handling halves the
batch and retries the same samples down to batch size 1; it does not change the data
or cell.

Registered confirmatory paired comparisons are:

- target-mask CNN minus context CNN;
- highlighted frozen ImageNet minus context frozen ImageNet;
- context ImageNet plus target morphometrics minus context frozen ImageNet;
- ensemble disagreement minus self-confidence;
- fixed hybrid minus the hybrid without self-confidence;
- fixed hybrid minus the hybrid without ensemble disagreement;
- optional pathology context minus ImageNet context.

Each comparison uses matched model seed, matched outer fold, the
`confusion_targeted_ten_percent` cell, average precision, and direction
`method_a_minus_method_b`.

Fold rotation is enabled for outer folds 1, 2, and 3. In each rotation one complete
official fold is the untouched final evaluation fold and the other two folds are
development; 10% of development groups are selected by the same registered selector.
The feasibility rule requires all five classes in every development training
partition. No result from one rotation may tune another. Every rotation is reported
separately, followed only by a descriptive fold mean; completion requires all
required feasible cells or an explicit recorded failure.

## Final-test policy

Official fold 3 remains embargoed for primary selection and tuning. It is never
corrupted and is excluded from primary OOF training, reference validation, method or
hybrid selection, calibration, early stopping, threshold/budget selection, cache
recipe selection, and favourable-result inspection. During M6, access was limited to
byte integrity and class-free aggregate/per-group eligibility metadata; the eligible
pilot contains no final-fold sample identity, label, geometry, representation, or
outcome.

Before freeze, deterministic required caches may be materialized under the already
registered recipes solely to establish byte identity and provenance. Their creation
must not inspect class performance, ranking metrics, restoration metrics, or any
final-fold outcome. Cache recipes, preprocessing, sample order, and bytes are then
immutable inputs. Primary final-fold outcomes may be opened only after the document,
configs, cache bundle, source tree, and governance evidence pass the freeze gate.

The confirmatory three-fold rotation is also fixed before any primary outcome. When
fold 3 later serves as development in rotations whose outer fold is 1 or 2, its prior
primary outcome cannot alter models, scenarios, hyperparameters, risks, budgets,
comparisons, or stopping rules. Rotation inputs and results remain segregated and
group-safe.

Exploratory original-label auditing occurs only after the controlled studies and
uses group-safe OOF evidence. Outputs must use the exact terms **potentially inconsistent annotation**
and **recommended for expert review**. They are non-diagnostic suggestions for
review, never confirmed medical errors, and no expert responses are simulated.

## Provenance identities and remaining pre-freeze gates

Known authorities at this pre-freeze checkpoint are:

- eligible M6 run:
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0`;
- pilot artifact root:
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`;
- pilot artifact-manifest SHA-256:
  `076f9cc45a42ac63f2dad9679b8979a50434b4c9a824d64fb94baf6b121b9074`;
- pilot immutable-marker SHA-256:
  `9f392bef4fe6b548b6e5c516356cef472d51702c091e4f10be5cdb4b81c0daa8`;
- full nucleus-manifest SHA-256:
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`;
- pilot folds-1/2 development-manifest SHA-256:
  `8107e1ddc033b08f03d3f351b5993ec1fd7a188677ee4c2afc0e4cbfe8432ef8`;
- pilot pre-execution gate-certificate SHA-256:
  `347f734c2355cd5009d631e959c077fe430adcad7e3997fe1306280004e90146`;
- validation JSON SHA-256:
  `094497f43e2ee0bd5dabddcd01f8c934657f130450a66f46600311451d36bc4a`;
- raw 22-file inventory identity:
  `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`;
- sealed dataset-tree SHA-256:
  `5647b4837fdaeb1281a5af0623f24aab1361263d3041549d012c8c5697fb31ed`,
  using the source-frozen `windows_compatible_relative_path_sort_key` component order;
- freeze reconciliation-record semantic SHA-256:
  `83e3eb7c4460c7c368a9bf70d49c3117f229f694daaa062456ba9f714c75651a`;
- duplicate-audit JSON SHA-256:
  `2a24d0f637bbf47e215f276e38d30b6bd65f1d312caa1b5b285ca5c00540612e`;
- pilot-derived parameter SHA-256:
  `8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b`;
- pathology availability-audit SHA-256:
  `5e568cf29e489d8948bfcd33feae5b292cb48837eb4c93754202a565778a6e4a`;
- canonical sample-order SHA-256 declared by both study configs:
  `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`;
- strict representation-independence JSON SHA-256:
  `846f421284de381401761a8dc4ceb108d3f3f2a0eece379706be7f7a512789c7`.

The finalized execution config files have the following identities:

- `configs/primary.yaml`: file SHA-256
  `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9`,
  semantic config SHA-256
  `c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15`,
  primary-plan semantic SHA-256
  `12a98f9dd40480927d94d8f25901392b0eb755194a0d44aebdbdb2ded26dee7f`;
- `configs/confirmatory.yaml`: file SHA-256
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`,
  semantic config SHA-256
  `ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b`,
  confirmatory-plan semantic SHA-256
  `c1993d4403982814a7259c524bbd21784537b7634e49a4f7150a9ca4de3c2c87`.

Both configs have status `READY_FOR_FREEZE`; their strict cross-config validation
produces 222 primary cells and 108 confirmatory cells. The independently verified
physical cache identities are:

- crops: NPZ `07d484be3e9f7826030f5d54d17e9878f61b68c282c4a91305a30ecfa86f4a01`,
  sidecar file `738d3f4b3146ff6d62555d283dac84a05a063b98199a6d13175100feb5d5dd42`,
  sidecar semantic `cf74379bab82d41be0df6cf047f8d365c5beb8854d34e7e69e22b2be403756b9`;
- engineered target features: NPZ
  `a4abe0cbb3d8ba4afc02f52e045af97a256c8e69ae70455294fc17aa5b752d32`,
  sidecar file `bd25dac4cbccb16e97066ebd5e9d6483c6522614f4ff276abf639ebc5e69301b`,
  sidecar semantic `8a3fdcae1b9dcc47fdfe08239db667bd35fbee7fc2d7e1d358bf6ca4529cda28`;
- target-highlighted ImageNet embeddings: NPZ
  `585b6ccace8130c911d7300e86ffe74e7b28f71fcd81fd2e3e62da5c1def9d79`,
  sidecar file `c8750217c5c48e41837d6da1688dd1cf70a32ce8f036dcf22ff96b0605229865`,
  sidecar semantic `efefa0cc0194f3b571cd82cce19fe3e61e963ded63ba172dcf2abce9c226f81e`;
- context-RGB ImageNet embeddings: NPZ
  `04cd2e0315f0a3c4270473af27aa1ca7d2a8df78268423186b3ad1bd5552e68b`,
  sidecar file `97fae8698de6949f30042d5ddf0b170510915cd7cde5e16ce5db78604f196410`,
  sidecar semantic `195cac1a2073d9d7974e71fbc485637f661c9db485ff53db78721021689ae618`;
- context plus target morphometrics: NPZ
  `547f26a947b527cb3ebe1dd5a670b86012ed4f960199cb3f204f5cfea2f5c0ef`,
  sidecar file `40a68b9dd1d9370bbd48066179e1513c45329d7febcf504667ff6a7f9aab9daa`,
  sidecar semantic `531a730ba54f6d1d6bd6447f3f8b336cd8b36184a73e32e2aaa7edec5d4421ee`.

Every physical cache binds 188,333 analysis-eligible samples, full manifest SHA-256
`7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`,
sample-order SHA-256
`2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`,
raw-inventory SHA-256
`51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`,
and `provenance_scope=stage_eligible`. Six required logical representations are backed
by the five physical pairs: the two confirmatory CNN pixel views intentionally share
the crop sidecar authority. Primary records bind physical cache-file identities;
every available confirmatory provenance record intentionally has
`cache_file_sha256: null` and binds `sidecar_semantic_sha256` as its record authority.
That sidecar semantic identity authenticates the corresponding physical NPZ hash and
content, including both logical CNN views backed by the shared crop cache.
The unavailable optional pathology cache retains its verified blocker evidence and is
not a required cache.

The 35-GiB operational start gate was `NO-GO` at the 2026-07-18 checkpoint and was
satisfied on 2026-07-19 without an override. The first production attempt stopped
before cache creation because extraction forwarded noncanonical base-validation
bounds. After that defect was fixed and tested, the corrected producer completed on
CUDA with exit 0 in 2,143.3 s, verified all 188,333 sample identities, and atomically
published the bundle and strict independence record. Independent cache, raw-data,
sealed-M6, configuration, and downstream-input readbacks all passed. The integrated
full test, lint, format, type, functional CLI, and independent freeze-verification
gates must still pass before this document becomes the immutable base preregistration
authority.

The pre-freeze dataset readback also exposed a host-order mismatch between the original
RunTracker directory digest and the reconciliation inventory order. No file identity
differed. Before freeze, both digest producers were bound to one source-frozen,
platform-independent ordering: lower-cased relative POSIX path components followed by
the exact original components as a tie-break; the digest still includes the original
UTF-8 path bytes. This preserves the sealed M6 tree SHA-256 shown above on both Windows
and POSIX. It does not reorder the case-sensitive reconciliation records or change
their semantic SHA-256.

## Amendments

There are no amendments to this pre-freeze definition. After the immutable base authority
is created, it is never overwritten or edited in place. A change uses
`preregistration amend` to publish a complete timestamped successor with:

- an explicit base-freeze or predecessor-amendment parent;
- timestamp, one-line reason, affected hypotheses and analyses;
- an explicit declaration of whether outcomes were inspected and, when true, the
  inspection timestamp;
- complete successor preregistration, primary config, confirmatory config, and
  execution-source snapshots;
- exact before/after file, semantic-config, execution-manifest, and execution-root
  hashes;
- a no-overwrite artifact seal and recursive parent-chain verification with cycle and
  tamper detection.

An amendment made after outcome inspection marks every affected analysis
`amended_or_exploratory`; it can never be reported as the original unamended primary
analysis. A pre-outcome amendment is explicitly labelled
`amended_before_outcome_inspection`. Verification uses
`preregistration verify-amendment`; a failed parent, hash, cycle, timestamp,
disposition, or artifact check fails closed.
