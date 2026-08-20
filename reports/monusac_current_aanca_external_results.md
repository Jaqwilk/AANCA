# Current AANCA on new MoNuSAC data

## Frozen disposition

Study: `monusac_current_aanca_controlled_external_v1`  
Disposition: `prospectively_frozen_controlled_external_benchmark`  
Overall frozen decision: **not supported**  
Executable action: retain the corrupted/no-review comparison; do not claim a better
real-world model.

The protocol, model, 5% review budget, seeds, queues, matched control and four-part
success rule were committed at `3036059` before any MoNuSAC outcome metric was
calculated. The test uses newly introduced controlled label changes on external
images. It is not a test of natural pathologist errors or clinical utility.

## Data and independence

- official MoNuSAC training archive: SHA-256
  `5b7cbeb34817a8f880d3fddc28391e48d3329a91bf3adcbd131ea149a725cd92`;
- official MoNuSAC test archive: SHA-256
  `bcbc38f6bf8b149230c90c29f3428cc7b2b76f8acd7766ce9fc908fc896c2674`;
- development: 29,610 eligible nuclei in 44 TCGA patient groups;
- untouched final test: 15,494 eligible nuclei in 25 TCGA patient groups;
- controlled development corruption: exactly 2,961 labels (10%), seed `26082080`;
- `TCGA-A2-A0ES` and `TCGA-MP-A4T7`, found in both official archives by
  identifier-only inspection, were excluded from development only;
- no MoNuSAC patient identity overlapped the saved NuCLS manifests;
- PanNuke does not expose enough patient identity metadata to prove complete
  cross-dataset non-overlap, so that limitation remains open.

All five audit folds kept complete patients together. The official final test was
not used for score selection, threshold selection, calibration or tuning. Source
annotations were never modified.

## Frozen 5% ranking result

| Queue | Average precision | Injected changes found / reviewed | Precision |
| --- | ---: | ---: | ---: |
| Self-confidence, global | 0.547580 | 943 / 1,481 | 0.636732 |
| Self-confidence, balanced | 0.547580 | 949 / 1,481 | 0.640783 |
| Neighbour disagreement, balanced — primary | 0.658142 | 1,035 / 1,481 | 0.698852 |
| Fixed hybrid, balanced | 0.691649 | 1,141 / 1,481 | 0.770425 |

The primary queue exceeded the exact class/organ/proposed-transition matched-random
control by `+0.142843` precision. Its 95% whole-patient interval was
`[+0.099181, +0.188491]`, so the registered retrieval gate passed.

The fixed hybrid had the best observed ranking point estimate, but the prospectively
declared primary candidate was the neighbour queue. The final test cannot be used to
replace or tune that primary candidate.

## Untouched final-test classification

| Development-label condition | Macro F1 | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Corrupted, no review | 0.503835 | 0.750871 | 0.752082 |
| Self-confidence global review | 0.492826 | 0.732993 | 0.744277 |
| Self-confidence balanced review | 0.490778 | 0.728605 | 0.743788 |
| Neighbour disagreement balanced review — primary | 0.509361 | 0.756357 | 0.759494 |
| Fixed hybrid balanced review | 0.493397 | 0.732735 | 0.748486 |
| Uncorrupted reference ceiling | 0.614845 | 0.875565 | 0.807909 |

The primary intervention improved the macro-F1 point estimate by `+0.005526` over
corrupted/no review, but the 95% whole-patient interval was
`[-0.001506, +0.012833]`. The lower bound was not above zero, so benefit was not
established.

Against the mean of 20 exact-matched random review conditions, the primary
difference was only `+0.000031`, with interval `[-0.008692, +0.008486]`. The observed
primary point estimate was therefore practically equal to the matched-random mean.

The important-class safeguard also failed. In particular, the neutrophil recall
difference had interval `[-0.036699, +0.034784]`, whose lower bound was below the
registered `-0.01` non-degradation margin. Every review condition was consequently
assigned `retain_uncorrected`.

## Frozen success rule

| Required condition | Result |
| --- | --- |
| Primary top-K beats exact matched random | Passed |
| Primary macro-F1 lower bound exceeds corrupted/no review | Failed |
| Primary macro-F1 lower bound exceeds matched random | Failed |
| Every important-class recall lower bound is at least -0.01 | Failed |

All four conditions were required. Passing one retrieval condition cannot override
the three failed downstream and class-safety conditions.

## Independent verification

`scripts/verify_monusac_external_validation.py` imports only the standard library
and NumPy. It pins every result file, rejects pickle-dependent arrays, checks patient
separation and OOF group allocation, verifies the exact matched strata, and
independently recalculates all four rankings, downstream metrics and 2,000-draw
whole-patient bootstrap decisions.

Final checked-in evidence identities:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Artifact manifest | 657 | `e4b1c0c327bba39f98677fc5e6f742f4158c77d0b0ba660ee29f5378b7510e7b` |
| Numeric evidence | 9,889,779 | `bda87a00b79db4962c71177a2dd3dea0c4c65b8b2d7299c577fd2ce4fdc1e8ec` |
| Results | 33,249 | `b2724e3e0baedcd0f1eb0fc7dfae127bf3789b03ac626d2701477ff4bae8e7d4` |
| Report | 2,711 | `e6911fd73f2103a3ffbb650da180816f344a527691326ec62d641ef55663be42` |
| Source inventory | 114,728 | `2b84809ea064552c8d011e17c32b86c6a47e0870f7d3801ef9c15ba1eeb87b0d` |

The complete experiment was executed twice. The results, report and source inventory
were byte-identical, and every scientific metric was unchanged. The second numeric
package intentionally adds fold IDs, organ strata and matched-random indices needed
for independent verification.

Run the readback with:

```text
uv run python scripts/verify_monusac_external_validation.py
```

## Interpretation boundary

This new dataset provides positive evidence that the current AANCA primary queue can
prioritise **artificially introduced label changes** on external MoNuSAC images more
efficiently than an exactly matched random queue. It does not provide sufficient
evidence that review improves the downstream classifier, and it does not show that
AANCA detects natural pathologist errors, improves laboratory work, benefits
patients or has clinical utility. Those claims still require a prospective blinded,
multi-rater, multi-site study on untouched natural cases.
