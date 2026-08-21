# Public primary-study evidence

The GitHub release [`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)
publishes the retained, licence-compatible numeric evidence behind the AANCA PanNuke
results. It is rooted in the accepted run
`20260727T133947.089370Z_pannuke_primary_orphan_recovery` and is separate from the
small presentation extract committed under `artifacts/mvp_demo`.

## Downloaded evidence

The release is split into three independently checksum-verifiable assets:

| Asset | Contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `aanca-primary-evidence-v1.zip` | Primary statistics, 2,000-draw group bootstrap, subgroup table, H4 restoration arrays, manifests and independent verifier | 358,518,237 bytes | `7241104c749e5899b23aa89af1dcbff0effcefe61e044d0b233d320136d115fc` |
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
release contains the same derived evidence and independent verifier in one archive
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

The PUMA protocol, configuration and result first entered public Git history
together in commit `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. Local authorities
record the intended freeze-before-metrics sequence, but that commit is not an
independent pre-outcome timestamp. This limits the chronology claim without changing
the saved controlled result.

PUMA supports controlled-noise transfer only. It does not contain the paired natural
pre/post expert outcomes required for a pathologist-error or real-workflow claim, and
source annotations were never modified automatically.
