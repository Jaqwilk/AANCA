# AANCA status

Updated: 20 August 2026

## Current scientific stage

- `PRIMARY_STUDY_COMPLETE`: the frozen-feature PanNuke controlled-corruption
  benchmark and restoration experiment completed in the accepted recovery run.
- `EXTERNAL_VALIDATION_COMPLETE`: the prospectively frozen NuCLS genuine
  multi-rater evaluation completed; its primary ranking and downstream claims were
  **not supported**.
- An additional prospectively frozen controlled-external MoNuSAC benchmark
  completed; its retrieval gate passed, but the registered combined success rule
  was **not supported**.
- `DEMO_COMPLETE`: the checksum-verifiable static article package is built and
  deployed.
- `CONFIRMATORY_COMPLETE`: not reached.

Stage completion records that the prescribed evaluation ran and its evidence was
preserved. It does not mean the result was favourable. AANCA remains a
non-diagnostic research prototype, never modifies source annotations automatically,
and has not proved natural pathology errors, pathologist errors or clinical utility.

## Accepted PanNuke primary evidence

Accepted run:
`20260727T133947.089370Z_pannuke_primary_orphan_recovery`

- 185/185 required cells completed;
- 33 of 36 registered H1/H3/H5/H6/H7 comparisons are numeric;
- three H6 encoder comparisons are explicitly unavailable;
- H4 is adverse: guided minus random restoration macro F1 is
  `-0.0021560596665870235`, 95% interval
  `[-0.0028586383107464695, -0.0013925186647962915]`;
- the analysis remains permanently `amended_or_exploratory` because outcomes were
  exposed during technical recovery.

Public release:
[`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)

## NuCLS external multi-rater evidence

Study: `nucls_natural_label_external_validation_v1`

The protocol and configuration were committed at `b34cba5` and publicly anchored by
tag `nucls-external-validation-preregistered-v1` before outcome-table download. The
official NuCLS repository authority is commit
`a87ac50c05cbc8ea11a41516be819f6b31436be7`; every selected official source file is
listed with a portable path, byte size and SHA-256.

Primary Unbiased Control result:

- 811 eligible nuclei, five TCGA patient groups, 27 NP/P disagreements;
- AP `0.07348905384277785` versus prevalence `0.03329223181257707`;
- AP-minus-prevalence 95% interval
  `[0.006105441307506995, 0.21362402839313893]`;
- 4/41 disagreements at the 5% budget; precision-minus-prevalence 95% interval
  `[-0.03007518796992481, 0.15407470288624786]`;
- guided macro F1 `0.7493354052113146`, mean-random `0.7610853660302145`,
  uncorrected `0.7639687577478279`;
- guided-minus-random 95% interval
  `[-0.02386630572001025, 0.005799510666181229]`;
- guided-minus-uncorrected estimate `-0.014633352536513322`, 95% interval
  `[-0.026683314580239315, -0.002414596650361811]`.

The ranking claim required both ranking intervals to be above zero; the 5% condition
failed. The downstream claim required both downstream intervals to be above zero;
both failed, and the comparison with uncorrected labels was adverse.

Secondary Evaluation result:

- 908 eligible nuclei, five TCGA patient groups, 60 NP/P disagreements;
- AP `0.08385776787414528`, with AP-minus-prevalence interval
  `[-0.014534404982312607, 0.11068625558630536]`;
- 3/46 disagreements at the 5% budget;
- guided minus uncorrected macro F1 `-0.008363810600757304`;
- ranking and downstream success rules both failed.

The secondary subset cannot rescue the failed primary result. NuCLS P-truth is
inferred pathologist consensus, not guaranteed biological truth, and NP/P
disagreement is not proof that a pathologist was wrong.

## External evidence identities and readback

| Subset / artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Unbiased source inventory | 29,132 | `c89fcb83d52ee449fa3ff45638ad477d8880714b1c5bfec9efaac2d0be992243` |
| Unbiased numeric evidence | 1,712,144 | `b03e578b5b4939dbb2554e26fedbd28515fb5c52e3353027f853cad03d3b75b9` |
| Unbiased results | 19,653 | `a931100a57e8d4b2a34a0216f047de8aa9d21c7275bc1beafd3b361fd955e9f6` |
| Evaluation source inventory | 29,069 | `fe7a46f1681827877bd0ec0fc6e2374fd63899abc53759b6ccf3e2f7d6cab96c` |
| Evaluation numeric evidence | 2,144,648 | `940bcc23d6b8c82f2d4a13587c0a3e3c11208b6363d31277d642f303906392d0` |
| Evaluation results | 19,558 | `c07131dced4113d89617029f88d544da5e33a0b88f4fc67c3a5ac634909fd28b` |

`scripts/verify_nucls_external_validation.py` imports only the standard library and
NumPy. It independently verifies every published file, portable source and sample
manifest, OOF risk calculation, fixed-budget outcome, random baseline, downstream
metric and frozen group-bootstrap draw. Its accepted conclusion is
`primary_claim_conclusion: not_supported`.

Public release:
[`nucls-external-validation-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/nucls-external-validation-v1)

- target commit: `317022ccf268aa0352327064cf3f55453089e934`;
- asset: `aanca-nucls-external-validation-v1.zip`, 4,001,323 bytes;
- GitHub digest:
  `sha256:e7384e2e8ff6eeab97485dfa3196ddbd261bbe335ebfa572d9f275de402a4d08`;
- a fresh extraction passed the independent verifier before upload.

## Engineering and publication status

The repository has portable Ubuntu/Windows CI, deterministic synthetic reuse,
independent primary and external evidence verifiers, a fixed local article launcher,
and a minimal long-form public presentation. The “What the study actually learned”
animation remains in the article.

Final local validation for this external-evidence update:

- `ruff check .`: passed;
- `ruff format --check .`: passed for 164 files;
- complete suite: `1070 passed, 1 skipped` in 576.49 seconds;
- independent NuCLS verifier: both fixed file sets, portable source inventories,
  exact sample manifests, ranking, random baselines, downstream metrics and all
  frozen group bootstraps passed; primary conclusion `not_supported`;
- five-file presentation: valid, root
  `557cb17a81dcc64429060ec0a7c578c2bcd00bba477543a9b94aa6396031383d`,
  with `external_validation_completed: true` and the null/adverse claim boundary.

Evidence commit `317022ccf268aa0352327064cf3f55453089e934` is public on `main` and
is the target of release `nucls-external-validation-v1`. The live deployment
was published from documentation commit
`063658ade0010e8916e15dc9f134a3736b5b722c` to
[`mediumaquamarine-wombat-125861.hostingersite.com`](https://mediumaquamarine-wombat-125861.hostingersite.com/).

Hostinger deployment evidence:

- rollback backup: `20260820-195717`;
- health checks: page, local JSON asset and external animation assets returned HTTP
  200;
- all five files read directly from the Hostinger origin matched the local byte
  sizes and SHA-256 identities exactly;
- origin `index.html`: 219,789 bytes,
  `9464bcf51bc640109a3b6dbf4ca3ef7b34b8fc07175f7b27249627e93026b0d4`;
- origin QC PNG: 3,188,071 bytes,
  `a1bd87dd397417d711d1d4937429eae5f5d972d3fa6ffa27a45129339587f10a`;
- Hostinger CDN losslessly recompresses the public PNG response to 3,096,631 bytes.
  Decoded RGBA arrays were exactly equal, with shared pixel SHA-256
  `a834db2c180d6f4b961d92487d86117356f831137165a80282933febc1585b58`.
  The GitHub release and checked-in package remain the byte-verifiable evidence
  authorities; the CDN-delivered image is pixel-identical presentation media.

## Open research work

The following are still unperformed and cannot be closed by wording or code alone:

1. newly recruited, blinded review of natural cases by multiple qualified
   pathologists;
2. prospective clinical or workflow deployment and patient outcomes;
3. broader external replication with substantially more patients and sites;
4. a new untouched confirmatory PanNuke study;
5. the audit-time-label sensitivity analysis defined in `PLAN.md`.

Until evidence designed for those claims exists, do not claim natural-error
detection, pathologist-error detection, clinical utility, patient benefit or broad
external generalisation.

## Cross-platform evidence serialization correction

GitHub Actions run `32400233125` exposed a byte-portability defect that was not
visible in the Windows working tree: pandas had written the two canonical NuCLS CSV
files with CRLF, while `.gitattributes` correctly normalized tracked text to LF.
Consequently, an Ubuntu checkout contained the same records but did not match the
Windows-generated byte pins. The scientific calculations were not involved in the
failure.

The writer now sets `lineterminator="\n"` explicitly. Both checked-in CSV files and
their artifact manifests were normalized and re-pinned:

| Subset / artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Unbiased canonical manifest | 453,774 | `b5f4a41cec342c2de7427eee8b457ab50daf1da48f815ea026d617d47c38c657` |
| Unbiased artifact manifest | 609 | `f044f29688a7340d7b9d8c3c12134873bbfa5c2a8a29ddcb22a493a269581c61` |
| Evaluation canonical manifest | 488,262 | `7c707768398c2ef6e50b764a85a18510188ea5086f2ab4f63d01f77f8d7b5356` |
| Evaluation artifact manifest | 609 | `5dc9ba6187f6f6ae3e9307f172f9e4d75be91892ca8a72491e503ec921499e35` |

Independent recalculation after normalization passed for both subsets and returned
the unchanged conclusion `primary_claim_conclusion: not_supported`. The focused
external-validation tests passed (`5 passed`). The fresh complete local suite passed
with `1070 passed, 1 skipped` in 587.51 seconds; lint and formatting also passed for
all 164 maintained Python files. A fresh Ubuntu / Windows CI result must be recorded
before this portability correction is closed.

## Incremental improvement of the current AANCA model

No replacement repository or “v2” was created. The same original-label audit now
supports `nearest_neighbour_disagreement` and `fixed_hybrid` in addition to the
existing probability scores. Neighbour outputs preserve the exact reference sample,
source group and distance for every ranked nucleus and validate that the query source
group is absent.

The frozen NuCLS result remains byte-identical and scientifically unchanged. A
separate `post_outcome_exploratory` analysis of the preserved evidence produced:

- primary Unbiased Control neighbour score: AP `0.068910`, 4/41 at 5%, AP-difference
  interval `[0.008527, 0.079894]`, precision-difference interval
  `[0.018415, 0.150843]`; both exploratory gates passed;
- secondary Evaluation neighbour score: AP `0.072560`, 2/46 at 5%, intervals
  `[-0.011768, 0.052766]` and `[-0.099889, 0.075346]`; gates failed;
- the candidate therefore was not promoted to the default;
- the retraining guard rejected the frozen guided candidate in Unbiased Control
  (`-0.014633`, interval `[-0.026683, -0.002544]`) and Evaluation (`-0.008364`,
  interval `[-0.034640, 0.033899]`);
- even the full-consensus training candidate was rejected in both subsets.

The implemented runtime action is `retain_uncorrected` whenever independent
whole-group validation is adverse, neutral or uncertain. This fixes the unsafe
application policy; it does not turn the saved adverse outcome into an improvement.
Recalculation command:

```text
uv run python scripts/analyze_nucls_current_model.py --format markdown
```

Machine logic is in `src/histo_audit/auditing/strategies.py` and
`src/histo_audit/evaluation/retraining_guard.py`; exact results and the prospective
claim boundary are in `reports/nucls_current_aanca_improvement.md` and
`PROSPECTIVE_WORKFLOW_PROTOCOL.md`.

Validation after implementation:

- complete maintained suite: `1077 passed, 1 skipped` in 784.00 seconds;
- `ruff check .`: passed;
- `ruff format --check .`: 171 files already formatted;
- current-model recalculation: `promoted_to_new_default: false`,
  `frozen_external_result_changed: false`,
  `retraining_application_is_fail_closed: true`;
- independent frozen NuCLS verifier: both subsets and all file identities passed,
  with the unchanged conclusion `primary_claim_conclusion: not_supported`;
- `histo-audit audit original --help`: passed and exposes `--neighbour-k` and
  `--neighbour-metric` for non-stage exploratory execution.

## Fail-closed intervention layer for the current AANCA

The existing AANCA repository and model workflow were extended in place; no new
project or replacement “v2” was created. The saved NuCLS outcome remains immutable,
and neither its adverse downstream result nor its exposed external labels are used
to tune or select the new policy.

The current implementation now provides:

- two explicitly separate queues: an annotation-quality queue based only on exact
  group-safe OOF evidence, and a model-improvement queue that remains unavailable
  unless a cross-fitted development estimate supplies both measured expected utility
  and a conservative lower bound;
- deterministic review caps by source group, class, tissue and transition, with an
  optional feature-space diversity constraint, so a high-scoring cluster cannot
  consume the review budget silently;
- an exact matched-random review comparator and a blinded package selection plan;
  construction fails instead of returning a partial or unmatched control sample;
- preservation of all expert votes and derived interventions `keep`, `soft_label`,
  `downweight`, `exclude` and `hard_change`; hard changes are disabled by default and
  require at least two votes and two-thirds agreement when explicitly enabled;
- disjoint-development-group comparison of unchanged labels, gated hard correction,
  soft labels, downweighted hard labels and soft labels with abstention;
- a multicriteria retraining guard: a candidate must have a positive macro-F1 lower
  confidence bound and must not violate registered important-class recall margins;
  otherwise the executable action is `retain_uncorrected`;
- group-cross-fitted temperature calibration that accepts only newly collected
  expert development labels paired with group-safe OOF probabilities, plus a
  multi-model, multi-checkpoint stability signal that filters transient spikes;
- a nested group-cross-fitted development-utility estimator. It can learn only from
  genuinely measured intervention outcomes and cannot manufacture utility targets
  or consume the final external test;
- one frozen intervention policy in
  `configs/current_aanca_intervention_policy.yaml`, with the operational and claim
  boundaries documented in `CURRENT_AANCA_SAFE_INTERVENTION.md`.

The pathology-encoder route remains a gated candidate route, not an asserted
improvement. UNI/CTransPath-derived representations may enter development comparison
only after provenance, licensing, group independence and OOF requirements pass. No
encoder, score, calibration or intervention is promoted from the adverse NuCLS test.

Final validation of this in-place improvement:

- complete maintained suite: `1100 passed, 1 skipped` in 569.45 seconds; the skip is
  the documented Windows/POSIX file-deletion sharing test;
- `ruff check .`: passed;
- `ruff format --check .`: all 186 maintained Python files already formatted;
- independent frozen NuCLS verifier: file identities, manifests, ranking outcomes,
  random controls, downstream metrics and frozen bootstraps all passed, with the
  unchanged conclusion `primary_claim_conclusion: not_supported`;
- current-model analysis: no replacement project, no changed frozen outcome, no
  promoted neighbour candidate and `retain_uncorrected` for both saved retraining
  candidates;
- real synthetic `audit original` execution: 300 samples in 60 source groups,
  group-safe OOF provenance accepted, 16 of 20 requested balanced review items
  selected under the declared caps and diversity constraints, and the underfilled
  budget reported explicitly;
- real matched-package execution: two ranked and two random items, exact 1:1 matching
  in every recorded stratum, valid private linkage and 12 generated review assets;
- synthetic data reuse, smoke experiment and five-file presentation verification all
  completed successfully; the article package remains `DEMO_COMPLETE`.

This engineering closes the unsafe-correction and evaluation-policy defects. It does
not supply the still-missing empirical evidence: natural-error detection, operational
benefit and clinical utility still require a new prospective, blinded, multi-rater,
multi-site study on untouched cases. Until that study is executed, those claims remain
prohibited.

## MoNuSAC controlled external new-data evidence

Study: `monusac_current_aanca_controlled_external_v1`

The protocol and machine configuration were committed at `3036059` before outcome
metric execution. Official archive SHA-256 values matched the frozen authorities.
Two TCGA patient identities present in both official archives were excluded from
development only; the official test remained intact. There was no patient-ID
overlap with saved NuCLS manifests. PanNuke does not expose enough patient metadata
to exclude every cross-dataset overlap, so this remains an explicit limitation.

Data and intervention:

- 29,610 eligible development nuclei in 44 patient groups;
- 15,494 eligible untouched final-test nuclei in 25 patient groups;
- exactly 2,961 symmetric controlled label changes (10%), seed `26082080`;
- five OOF folds, each holding out complete TCGA patients;
- source annotations remained unchanged.

Frozen 5% primary neighbour queue:

- AP `0.6581415388588143`;
- 1,035 injected changes found among 1,481 reviewed nuclei;
- precision `0.6988521269412559`;
- precision minus exact matched random `0.1428426738690075`, 95% whole-patient
  interval `[0.0991810700012709, 0.18849133691115186]`: retrieval gate passed.

Frozen downstream result on the untouched official test:

- corrupted/no-review macro F1 `0.5038352361344909`;
- primary neighbour-review macro F1 `0.5093608506212538`;
- primary minus corrupted/no review `0.00552561448676292`, interval
  `[-0.001505873468356683, 0.012832750726675505]`: failed;
- primary minus mean exact-matched random `0.000030946417352573086`, interval
  `[-0.0086920112069291, 0.008485812521403327]`: failed;
- the important-class recall rule failed because at least one whole-patient lower
  bound was below the registered `-0.01` margin.

Only one of four simultaneously required conditions passed. The frozen decision is
`not supported`, and every candidate action is `retain_uncorrected`. This is positive
evidence for prioritising injected changes on new images, not evidence of natural
pathologist-error detection, real-use model improvement or clinical utility.

Evidence identities:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Artifact manifest | 657 | `e4b1c0c327bba39f98677fc5e6f742f4158c77d0b0ba660ee29f5378b7510e7b` |
| Numeric evidence | 9,889,779 | `bda87a00b79db4962c71177a2dd3dea0c4c65b8b2d7299c577fd2ce4fdc1e8ec` |
| Results | 33,249 | `b2724e3e0baedcd0f1eb0fc7dfae127bf3789b03ac626d2701477ff4bae8e7d4` |
| Report | 2,711 | `e6911fd73f2103a3ffbb650da180816f344a527691326ec62d641ef55663be42` |
| Source inventory | 114,728 | `2b84809ea064552c8d011e17c32b86c6a47e0870f7d3801ef9c15ba1eeb87b0d` |

The complete frozen run was repeated. Results, report, source inventory and all
scientific metrics were identical. The final numeric evidence adds only the fold,
organ and matched-index arrays required for independent recalculation.

`scripts/verify_monusac_external_validation.py` imports only the standard library
and NumPy. It verified the pinned package, patient separation, OOF group allocation,
all four rankings, exact matched strata, every downstream metric and all 2,000-draw
whole-patient bootstrap decisions. Its accepted readback is `status: verified` and
`all_success_conditions_met: false`.

Final local validation for this new-data result and publication update:

- complete maintained suite: `1103 passed, 1 skipped` in 610.40 seconds; the skip is
  the documented Windows/POSIX open-file rename difference;
- `ruff check .`: passed;
- `ruff format --check .`: all 190 maintained Python files formatted;
- focused post-format tests: `9 passed` for the presentation and MoNuSAC modules;
- independent MoNuSAC verifier: four pinned evidence files, four ranking candidates,
  exact controls, downstream metrics and 2,000 whole-patient bootstrap iterations
  passed; overall frozen success remained false;
- independent NuCLS verifier: all file identities, portable manifests, rankings,
  random baselines, downstream metrics and bootstraps passed; primary conclusion
  remained `not_supported`;
- current-model NuCLS recalculation: no replacement project, frozen result unchanged,
  neighbour candidate not promoted and retraining application fail-closed;
- deterministic synthetic data: existing checksum-matching dataset verified;
- synthetic smoke run `20260820T215257.976122Z_synthetic_smoke_7e2654bac3`:
  completed successfully;
- dependency-free and package-aware presentation verifiers: valid five-file package,
  root `113e3e8d20cf86dcde4afb09ffd9eb21f9aa78ab3733364e13c26d04945a8827`;
- real-browser readback: the article begins with the thesis, retains the animated
  “What the study actually learned” section, places NuCLS and MoNuSAC after detailed
  evidence, and renders the new centered article section without overflow.
