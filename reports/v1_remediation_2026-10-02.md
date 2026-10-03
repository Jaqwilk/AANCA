# AANCA V1 remediation and repeat audit

Started: 2 October 2026. Updated: 3 October 2026. Base commit:
`fa3311bf750c32b7f2bc0ac69097018620e77d71`.

This follows the [initial project audit](project_audit_2026-10-02.md) and the
owner's instruction to fix the demonstrated V1 errors and check everything again.
The official publication target is `https://aancastudy.org/`.

Closing machine-readable evidence is available in
[`v1_remediation_2026-10-02_evidence.json`](v1_remediation_2026-10-02_evidence.json).

## Changes and disposition

| Finding | Remediation | Disposition |
|---|---|---|
| F01: H4 interval mislabelled as a bootstrap CI | Kept every saved number; explicitly distinguish the central 95% random-review range on a fixed final reference set from the ranking bootstrap intervals. Added machine-readable interval provenance, updated HTML/ARIA/README, and tests rejecting resealed semantic drift. | Fixed and deployed |
| F02: official domain unavailable / technical domain old | The owner's separate deployment restored the official domain. This task deployed the corrected package and checked its equality to the current local release. | Fixed: apex and www return 200 and the current manifest |
| F03: eight HTTP PNGs differ from their own manifest | SFTP readback validates every origin file. Deployed `no-cache, no-transform` and added a complete HTTP verifier. Following refresh of the one stale apex CDN entry, both apex and www pass all thirteen files without cache-busting exemptions. | Fixed and verified |
| F04: invalid probabilities accepted by metrics | Shared validator checks shape, finite values, strict `[0,1]`, class alignment and absolute row-sum tolerance; it never clips or renormalises invalid input. | Fixed |
| F05: negative adoption thresholds allow degradation | Both retraining guards and the two-queue global gain reject negative thresholds. Existing negative per-class noninferiority margins remain separate and valid. | Fixed |
| F06: failed fits silently used / fallback reports convergence | Generic OOF and maintained downstream strategy/restoration paths reject a reported failed fit. OOF records actual optimiser diagnostics. Adam fallback checks its final gradient rather than inventing success. An estimator without a convergence flag remains explicitly `unknown`. | Fixed; no retrospective convergence claim |
| F07: fractional labels silently truncated | Maintained metrics, scoring, manifests, OOF, restoration, strategy and neighbour interfaces validate finite integer-valued labels/classes before conversion and reject int64 overflow. | Fixed |
| F08: neighbour provenance permits holdout leakage | Validate complete training/holdout disjointness, known groups and one holdout fold per source group before fitting a neighbour index. | Fixed |
| F09: verification/retraining/input requirements conflated | README, reproducibility and evidence documentation distinguish saved-array readback, raw-data inspection, model re-execution and output writes, with scratch-output examples. | Fixed documentation; unavailable public raw inputs remain explicit |
| F10: README implies calibration was executed | State that maintained ranking probabilities are uncalibrated; optional development calibration is a separate capability. Also distinguish original-audit defaults from the frozen selected candidate. | Fixed |
| F11: release verifier hardcodes the living status date | Require exactly one real `Updated` date while retaining checks on frozen release dates, claim boundaries and upstream source identities. | Fixed |
| Additional entropy validation error | An `epsilon` of `2` made entropy scores `[-0.0, -0.0]` for distinct distributions. NLL and entropy now reject nonfinite or out-of-range epsilon. | Fixed; regression coverage |
| Additional soft-target validation error | Training accepted target rows summing to `1.000001`. It now uses strict probability validation, rejecting these inputs before fitting. | Fixed; regression coverage |
| Duplicate decision identifier | Two independent records used D052. Retained the audit as D052 and assigned the hosting decision D053. | Fixed |

The shared numerical validator reduces duplicated contract logic without changing
valid-input scientific calculations or rewriting V1's evidence machinery. Broad
architectural reduction remains governed by D047 and the separate V2 boundary.

## Verification record

The final full test run passed: **1233 passed, 1 skipped in 631.11 seconds**.
The single skip is the documented POSIX open-file rename case on Windows.
Completed checks include:

- focused numerical/model/strategy regressions and presentation/publication tests;
- `ruff check .`, `ruff format --check .` (228 files), and `mypy` (111 source files);
- `uv lock --check`, `uv pip check`, dependency advisory audit (no known vulnerabilities),
  and `git diff --check`;
- successful synthetic CLI smoke runs, including the final-version run in
  `artifacts/qa/v1fix-smoke-accepted-20261002`;
- portable presentation verification and professor-release verification, including
  thirteen presentation files and nineteen upstream source authorities;
- AST parsing of all 112 source, 101 test and 21 script files, plus unique decision
  identifiers;
- exact SHA-256 readback of all nine frozen configuration sidecars;
- independent primary readback: all 33 numeric comparisons, three explicitly
  unavailable comparisons and the adverse H4 restoration result;
- NuCLS aggregate, MoNuSAC, NuCLS JP.1, RIVA, MIDOG++, PUMA and NuCLS paired-QC
  saved/raw evidence verification. PUMA and QC checks use scratch outputs;
- local Chromium checks at 1280 x 720 and 390 x 844: one H1, no heading-level skips,
  duplicate IDs, broken internal anchors, unnamed buttons, missing image alt text,
  failed completed image loads, horizontal overflow, console errors or warnings.
  The mobile menu opens/closes correctly; the comparison table opens, filters to
  three H6 rows, and returns to all 36 rows without horizontal page overflow.

Local presentation root:
`8719c185feb4ace95b242eedb60de063e3ccc4065df5590cfb331374814e93ca`.
The previous package is retained at
`artifacts/qa/v1fix-previous-presentation-20261002`.
Recursive before/after comparison confirmed all 878 pre-existing evidence values
are preserved, including every upstream identity. Only the evidence schema version
advances; interval provenance is additional metadata.

The preliminary full-suite attempt was interrupted before completion and supplies
no pass claim. During focused verification, the initial new tamper-test fixture
used the wrong evidence key; that test was corrected and passed. The newly added
OOF diagnostic initially failed static typing; this was repaired and `mypy` passed.
An initial primary verification command omitted `restorations/` from its input path;
the corrected command passed. These are recorded as verification repairs, not as
failures demonstrated in frozen scientific artifacts.

The first completed full-suite run (`1 failed, 1230 passed, 1 skipped`, 619.57 seconds)
found one integration regression: the initial
absolute `1e-7` row-sum tolerance rejected two valid float32 MLP probability outputs.
The revised absolute tolerance is `5e-7` with zero relative tolerance. Strict bounds,
finite-value checks and rejection of rows differing by `1e-6` remain enforced. A
real five-class PyTorch softmax has a `1.7508864402770996e-7` row-sum rounding error;
new tests cover both its original float32 values and unchanged float64 promotion.
The previously failing integration fixture and all 53 related focused cases passed.
No model output was clipped or renormalised, and no frozen fit controls changed.

## Publication

The Hostinger verification/build check passed before deployment. Reused that
successful verification with `deploy aancastudy.org --skip-build`; SSH/SFTP connected,
merge upload and HTTP health checks passed, and the separate server configuration
was uploaded. Recoverable backup: `20261002-235931`.

SFTP SHA-256 readback verifies all thirteen origin files and the exact `.htaccess`.
`verify_deployed_presentation.py --url https://www.aancastudy.org/` validates all
thirteen HTTP files and equality to the local package. Initially, only
`assets/hero/nuclei/nucleus-compact.png` remained stale on the apex: the CDN returned `HIT`,
291,149 bytes and SHA-256
`f1d1a24889b0fa560df840e6e13ed09056143e8e37069b95af49472ae6302c07`.
The expected 290,592-byte original is returned for the release-query URL with
`no-cache, no-transform`. Request `Cache-Control: no-cache` did not invalidate the
old entry. The initial strict apex publication check therefore failed.

The configured domain profile supplies working SSH/SFTP access but no API token.
The separate available API account returns no hosted websites and 404 for this
domain's read endpoint. No unauthorised cache mutation was attempted. The owner
was asked to flush this domain's CDN cache in hPanel. A subsequent ordinary strict
apex recheck passed all thirteen files with no query exemption. The cache refreshed
during the closing checks; no claim is made about which actor or mechanism refreshed
it. Both domains now match the same local root and every declared file identity.
No mandatory engineering or publication gate remains pending.

## Scientific and operational boundary

Every existing presentation scientific value and upstream source identity remains
unchanged. No source annotation, frozen configuration, final reference fold, model
selection or result file is changed. Saved-evidence recalculation supplies readback
checks, not a new study. The selected-candidate nested training rerun is not used for
this engineering remediation. The original analysis disposition, unavailable
Cleanlab values and negative findings remain visible.

Scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`, presentation stage remains
`DEMO_COMPLETE`, and natural-data action remains `retain_uncorrected`. The need for
new blinded, independently adjudicated, prospective evidence is unchanged. This
audit does not formally prove correctness of every possible input or environment.
