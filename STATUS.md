# AANCA status

Updated: 20 August 2026

## Current scientific stage

- `PRIMARY_STUDY_COMPLETE`: the frozen-feature PanNuke benchmark and its restoration
  experiment completed in the accepted recovery run.
- `DEMO_COMPLETE`: the checksum-verifiable static article package is built and deployed.
- `CONFIRMATORY_COMPLETE`: not reached.
- `EXTERNAL_VALIDATION_COMPLETE`: not reached.

The project remains a non-diagnostic research prototype for ranking potentially
inconsistent annotations for expert review. It does not automatically modify source
annotations and has not been validated against natural pathology errors.

## Accepted primary evidence

Accepted run:
`20260727T133947.089370Z_pannuke_primary_orphan_recovery`

Principal immutable outputs:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `primary_statistics.json` | 22,498,321 | `c3685fe9863fd73b1298f0558212cb5267b07c3ce6e4e4f37018dec55c115ac0` |
| `primary_subgroups.csv` | 4,208,358 | `36be649fef067de82cd11b77f508f0a6fe62f649d393a1c9975a4523c24d166e` |
| `primary_bootstrap_evidence.npz` | 372,330,793 | `35f8017cfcc887a1e94498a72e6868481088ce0fac4d8d2369d32504780bafa2` |
| `restoration.json` | 461,031 | `1dd6f9e105066d0ce0314839783d15d3d0a4f96ece7430216bc3c6733961a27e` |
| `restoration_evidence.npz` | 267,636,826 | `192c76b46a024d1124562ec461207ff24c20a5aaaf35d4b4b1998b53c7a8b956` |

Saved statistical result:

- 33 of 36 registered H1/H3/H5/H6/H7 comparisons are numeric;
- the three H6 encoder comparisons are explicitly unavailable;
- H4 is adverse: guided minus random restoration macro-F1 is
  `-0.0021560596665870235` with saved 95% interval
  `[-0.0028586383107464695, -0.0013925186647962915]`.

## Public reproducibility

GitHub release:
[`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)

Published assets:

| Asset | Bytes | GitHub digest |
| --- | ---: | --- |
| `aanca-primary-evidence-v1.zip` | 358,518,237 | `sha256:7241104c749e5899b23aa89af1dcbff0effcefe61e044d0b233d320136d115fc` |
| `aanca-primary-rankings-v1.zip` | 1,512,550,075 | `sha256:a5b4189583ea39a1aa82fd587f4adae2b8cc5d71e9aa45ed2c6b0337f7185319` |
| `aanca-primary-oof-v1.zip` | 878,046,730 | `sha256:79056c703401eaaf455212d86abe9e58eedd6376871ee78a8da95e33eed5a1a4` |

The independent verifier is `scripts/verify_primary_evidence.py`. On the retained
run and on a freshly extracted release archive it passed file identities, all 33
numeric comparison recalculations, all three unavailable H6 entries and H4.

The release contains derived numeric evidence and provenance, not PanNuke images or
masks. Fold checkpoints were not retained in the accepted run; the public OOF arrays
are the saved model outputs. Retraining requires a lawful local PanNuke copy.

## Engineering status

Published code commit: `fc944c6f4508efccfe09c4952bd0cd101786a2c5` on `main`.

Completed in this audit pass:

- fixed portable smoke output and the local article launcher;
- added full Ubuntu and Windows GitHub Actions jobs;
- removed hard-coded local PanNuke dependencies from tests;
- corrected POSIX/Windows filesystem, symlink and option-introspection tests;
- published primary statistics, H4, rankings and OOF evidence;
- removed the unexecuted capsule/authority/resource-controller layer from active code;
- reduced Python source plus tests from approximately 257,834 to 145,569 lines;
- made deterministic synthetic generation verify and reuse identical outputs while
  rejecting changed or partial packages;
- published a prospective blinded natural-case protocol without claiming it was run;
- retained a recoverable pre-simplification snapshot at Git tag
  `pre-audit-simplification-2026-08-20`;
- reduced this status ledger to current evidence; prior detail remains in Git history.

Final local validation on 20 August 2026:

- `ruff check .`: passed;
- `ruff format --check .`: passed;
- complete suite: `1065 passed, 1 skipped` in 606.97 seconds;
- deterministic data command: `verified_existing` with all arrays and manifests equal;
- synthetic smoke: completed as
  `20260820T155522.499299Z_synthetic_smoke_8343c6c437`;
- released-evidence verifier: 33 comparisons passed, three H6 unavailable, H4 passed;
- five-file presentation: valid, root
  `e69b4a4bf5224ec841eaac845c67515ad2e5a4fb83149a398f0e361c6646a812`.

## Publication verification

- [`Scientific software` run 32389776361](https://github.com/Jaqwilk/AANCA/actions/runs/32389776361)
  passed on Ubuntu in 9 minutes 23 seconds and Windows in 27 minutes 44 seconds;
  both jobs completed lint, formatting, the full suite and synthetic smoke.
- `main` was fast-forwarded to the exact tested code commit `fc944c6`.
- Hostinger deployment completed with rollback backup `20260820-183718`.
- the live `index.html` returned HTTP 200 and exactly matched the checked-in file:
  216,201 bytes, SHA-256
  `73ca68df93b1935e103164aba99a0a6e20af70596a5bf3f730df365299e1b3d4`.

These engineering and publication results do not change a scientific completion
stage.

## Open research work

The following cannot be closed by code or wording alone:

1. blinded review of natural, non-injected cases by multiple qualified pathologists;
2. agreement, ambiguity and abstention reporting;
3. patient- or WSI-level grouping once verified identifiers are available;
4. a prospective fold-allocation sensitivity analysis that uses only audit-time labels;
5. an external dataset and preregistered downstream intervention.

The prospective execution requirements are in
[`EXPERT_REVIEW_PROTOCOL.md`](EXPERT_REVIEW_PROTOCOL.md).

Until those studies are executed, do not claim expert validation, clinical utility,
natural-error detection, external validation or downstream improvement.
