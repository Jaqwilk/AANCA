# AANCA status

Updated: 20 August 2026

## Current scientific stage

- `PRIMARY_STUDY_COMPLETE`: the frozen-feature PanNuke controlled-corruption
  benchmark and restoration experiment completed in the accepted recovery run.
- `EXTERNAL_VALIDATION_COMPLETE`: the prospectively frozen NuCLS genuine
  multi-rater evaluation completed; its primary ranking and downstream claims were
  **not supported**.
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
  `d0da474c0862233ddc9335110fac1ef2c3e877057a76c8a0486fafa051485f31`,
  with `external_validation_completed: true` and the null/adverse claim boundary.

GitHub identities and the live deployment identity for this update are recorded
after publication.

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
