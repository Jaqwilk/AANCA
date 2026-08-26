# Public primary-study evidence

The GitHub release [`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)
publishes the retained, licence-compatible numeric evidence behind the AANCA PanNuke
results. It is rooted in the accepted run
`20260727T133947.089370Z_pannuke_primary_orphan_recovery` and is separate from the
small presentation extract committed under `artifacts/mvp_demo`.

Here, “standalone evidence recalculator” describes software separation: it does not
import the primary analysis package. It is not third-party validation, external peer
review or a second laboratory replication.

## Downloaded evidence

The release is split into three independently checksum-verifiable assets:

| Asset | Contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `aanca-primary-evidence-v1.zip` | Primary statistics, 2,000-draw group bootstrap, subgroup table, H4 restoration arrays, manifests and standalone evidence recalculator | 358,518,237 bytes | `7241104c749e5899b23aa89af1dcbff0effcefe61e044d0b233d320136d115fc` |
| `aanca-primary-rankings-v1.zip` | All 185 completed-cell ranking tables, per-cell artifact manifests and cell index | 1,512,550,075 bytes | `a5b4189583ea39a1aa82fd587f4adae2b8cc5d71e9aa45ed2c6b0337f7185319` |
| `aanca-primary-oof-v1.zip` | All 185 completed-cell OOF probability arrays, fold provenance, per-cell manifests and frozen matrix controls | 878,046,730 bytes | `79056c703401eaaf455212d86abe9e58eedd6376871ee78a8da95e33eed5a1a4` |

The machine-readable copy of this table is
[`evidence-release-manifest.json`](evidence-release-manifest.json). GitHub source
history and the release tag provide the trusted anchor for those asset identities;
the manifests inside each archive bind individual result files to the accepted run.

## Recalculate H1-H7

Download and extract `aanca-primary-evidence-v1.zip`, then run the verifier from a
checkout of this release tag:

```text
uv sync --dev
uv run python scripts/verify_primary_evidence.py PATH/TO/aanca-primary-evidence-v1
```

The verifier does not import `histo_audit`. It uses only the standard library and
NumPy to:

1. verify the fixed byte size and SHA-256 of every statistical and restoration file;
2. verify the saved manifests and canonical statistics-payload identity;
3. independently recalculate all 33 available preregistered comparison bootstrap
   means, 95% intervals, one-sided p-values and within-family Holm corrections;
4. confirm that the three H6 comparisons remain explicitly unavailable rather than
   being estimated;
5. independently recalculate the adverse H4 macro-F1 comparison from all 100 frozen
   random-review repetitions.

The ranking and OOF assets make the sample-level inputs inspectable. A reviewer can
check group IDs, fold coverage, pre-corruption and observed labels, injected-event
flags, model probabilities and every published audit score for each completed cell.

## What is and is not included

The release contains derived numeric evidence and textual provenance. It does not
contain PanNuke images, masks, raw dataset archives or patient identifiers. Reviewers
must obtain PanNuke lawfully from its official source to retrain models from images.

Fold models were fitted to produce OOF probabilities and were not retained as
checkpoints in the accepted run. The public OOF arrays are therefore the immutable
saved model outputs; model retraining remains a separate operation governed by
[`DATASET_SETUP.md`](DATASET_SETUP.md). This limitation is stated explicitly rather
than implying that unavailable checkpoints were published.

Publishing these artifacts makes the reported H1-H7 numbers independently
recalculable and the OOF/ranking evidence publicly inspectable. The PanNuke release
alone does not create expert or external validation and does not show that natural
annotation disagreement is a pathology error.

## External NuCLS evidence

The separate external study is checked in under
`artifacts/nucls_external_validation` and documented in
[`reports/nucls_external_validation_results.md`](reports/nucls_external_validation_results.md).
The immutable
[`nucls-external-validation-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/nucls-external-validation-v1)
release contains the same derived evidence and standalone recalculator in one archive
(4,001,323 bytes; SHA-256
`e7384e2e8ff6eeab97485dfa3196ddbd261bbe335ebfa572d9f275de402a4d08`).
It contains two frozen result bundles:

- `unbiased-v1`: primary Unbiased Control subset;
- `evaluation-v1`: secondary sensitivity subset.

Each bundle contains a portable per-file source inventory, exact paired-anchor
manifest, saved numeric evidence, result JSON and an artifact manifest. The source
inventories bind the official NuCLS files used in the analysis; raw NuCLS images and
outcome tables are not republished.

Run:

```text
uv sync --dev
uv run python scripts/verify_nucls_external_validation.py --json
```

The verifier imports neither `histo_audit` nor scikit-learn. It pins every evidence
file and independently recalculates ranking AP/AUROC and fixed-budget outcomes,
classification metrics, all deterministic random baselines and all group-bootstrap
draws from the frozen seeds. The accepted primary conclusion is `not_supported`.

This genuine external multi-rater execution establishes the completion stage
`EXTERNAL_VALIDATION_COMPLETE`, not a positive scientific claim. The primary ranking
rule failed because its 5% operational interval crossed zero, and guided correction
was adverse versus leaving labels unchanged. Inferred NuCLS pathologist consensus is
not guaranteed biological truth and disagreement is not proof that a pathologist
made an error.

### Independent-pathologist leave-one-out ranking

A distinct frozen NuCLS `U-control` analysis uses the raw `JP.1` bbox and
`raw_classification` as model input, removes `JP.1` from the reference, and derives a
strict-majority outcome from at least two mappable votes by other individual
pathologists. Aggregate P-truth fields are not used. Patient identity defines all five
OOF folds and the 5,000-draw bootstrap clusters.

At the primary 5% budget, 45 of 898 binary-reference-eligible nuclei were reviewed.
The AANCA queue precision was `0.333333` versus `0.215111` across 100 disjoint,
equal-budget random queues matched exactly on patient, observed class and proposed
transition. The difference was `+0.118222`, with patient-bootstrap 95% interval
`[+0.040000, +0.257143]`; the enrichment ratio was `1.549587`, interval
`[1.073620, 10.000000]`. The frozen primary ranking gate passed.

Only `JP.1` qualified for reporting. `JP.2` covered two patient groups and was
excluded by the frozen minimum of five; public individual raw geometry was unavailable
for `SP.1`--`SP.3` and `JP.3`--`JP.6`. The result is therefore one junior-pathologist
rotation, not a pooled multi-pathologist estimate. It validates the global risk
ranking rather than the five-patient-infeasible balanced deployment queue.

The protocol, report and machine-readable evidence are available at
[`NATURAL_PATHOLOGIST_VALIDATION_PROTOCOL.md`](NATURAL_PATHOLOGIST_VALIDATION_PROTOCOL.md),
[`reports/nucls_independent_pathologist_validation_results.md`](reports/nucls_independent_pathologist_validation_results.md)
and
[`artifacts/nucls_independent_pathologist_validation/results.json`](artifacts/nucls_independent_pathologist_validation/results.json).
Recalculate the saved metrics and verify source-annotation integrity with:

```text
uv run python scripts/verify_nucls_independent_pathologist_validation.py
```

This is project-coupled evidence readback, not third-party replication. Feasibility
counts were inspected before the freeze, protocol and execution share one repository
change without an independent timestamp, and only five patient clusters contribute.
No source annotation was changed. A positive disagreement outcome does not establish
which label is biologically correct, pathologist error, clinical utility or safe
automatic correction.

## Public RIVA and MIDOG++ independent-expert replication

The additive frozen study is rooted at
[`artifacts/public_independent_pathologist_replication`](artifacts/public_independent_pathologist_replication)
and documented in
[`reports/public_independent_pathologist_replication_results.md`](reports/public_independent_pathologist_replication_results.md).
The score-only artifacts were committed and pushed as
`09aede5000c43406759a432b674f6db37db98b26` before either reference was opened.

At the 5% budget, RIVA disagreement precision was `0.476357` versus `0.426900`
across 100 exact matched-random queues; the difference was `+0.049457`, with
whole-group 95% CI `[+0.014037, +0.091632]`. MIDOG++ precision was `0.325967`
versus `0.249392`; the difference was `+0.076575`, CI
`[+0.021352, +0.138159]`. All four RIVA leave-one-annotator-out rotations and both
MIDOG++ pairwise-expert rotations were non-negative, so both frozen dataset gates
and the cross-dataset replication rule passed.

Recalculate the saved reference association and metrics with:

```text
uv run python scripts/run_public_pathologist_replication.py verify --dataset riva
uv run python scripts/run_public_pathologist_replication.py verify --dataset midogpp
```

RIVA uses strict majorities from at least two other raw annotator votes and never the
released majority label. MIDOG++ alternates the two independent expert labels and
does not use the adjudicator or final category in the primary endpoint. This is
positive cross-dataset disagreement-enrichment evidence, not adjudication of which
pathologist is correct, a downstream-utility result or a prospective workflow trial.
Source annotations remained unchanged.

## MoNuSAC and PUMA evidence

The controlled MoNuSAC authority is
[`artifacts/monusac_external_validation/results.json`](artifacts/monusac_external_validation/results.json)
with its released numeric arrays and independent recalculation script. Run:

```text
uv run python scripts/verify_monusac_external_validation.py
```

Its retrieval gate passed, while downstream improvement and important-class safety
did not. The overall registered decision is `not_supported` and the action is
`retain_uncorrected`.

The frozen PUMA new-source confirmation is rooted at
[`artifacts/puma_new_data_confirmation/results.json`](artifacts/puma_new_data_confirmation/results.json).
The three large numeric archives are Git LFS objects because the evidence-readback
script uses their full arrays. After `git lfs pull`, run:

```text
uv run python scripts/verify_aanca_selected_candidate.py
uv run python scripts/verify_puma_new_data_confirmation.py
uv run python scripts/verify_nucls_supervised_qc_feasibility.py
```

The PUMA verifier rebuilds the official manifest and confirms the retrieval,
downstream, group-bootstrap, class-safety, source-integrity and 44 recorded model
convergence checks. It imports maintained PUMA helpers and reads saved predictions;
it does not independently retrain 44 models from source images. All seven frozen
PUMA gates passed. The related stress and observed-label sensitivity authorities are
tracked under `artifacts/`; they preserve their explicitly exploratory
post-confirmation status.

At the primary 5% budget, PUMA retrieval precision was `0.537739` versus
`0.214379` exact matched random. The downstream `flag_exclude` arm improved by
`+0.006426` macro-F1 over unchanged corrupted training, with whole-group 95% interval
`[+0.003657, +0.009365]`. These are controlled-corruption results on the AANCA-defined
held-out split, not natural-error or official hidden-challenge-test outcomes.

The PUMA protocol, configuration and result first entered public Git history
together in commit `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. Local authorities
record the intended freeze-before-metrics sequence, but that commit is not an
independent pre-outcome timestamp. This limits the chronology claim without changing
the saved controlled result.

PUMA supports controlled-noise transfer only. It does not contain the paired natural
pre/post expert outcomes required for a pathologist-error or real-workflow claim, and
source annotations were never modified automatically.
