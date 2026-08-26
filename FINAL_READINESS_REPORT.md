# AANCA final readiness report

**Audit date:** 26 August 2026  
**Audit type:** repository-wide project-maintainer review, not external peer review  
**Scientific stage:** `EXTERNAL_VALIDATION_COMPLETE`  
**Presentation stage:** `DEMO_COMPLETE`

## Final verdict

AANCA is ready to be shown as a rigorous university research prototype for ranking
*potentially inconsistent annotations* for qualified expert review. Its strongest
scientifically justified story is now broader than a synthetic demonstration:

- one controlled new-source PUMA study supports retrieval and downstream transfer
  under injected label noise;
- three public datasets provide positive, separately reported evidence that the
  frozen ranking enriches queues for independent-expert disagreement;
- adverse PanNuke, aggregate/downstream NuCLS and MoNuSAC outcomes remain public and
  constrain the claim instead of being removed.

The project is not a diagnostic system, clinical decision aid or automatic correction
tool. It is also not a completed prospective workflow study. Those are evidence gaps,
not software defects that can be repaired by adding another retrospective metric.

## Evidence a professor can inspect first

All figures below are generated from checksum-bound authorities. Percentage-point
differences are shown separately because the reference designs and scientific
questions differ; no post-hoc pooled effect is claimed.

| Evidence source | Frozen comparison | Result | What it supports |
| --- | --- | --- | --- |
| NuCLS `JP.1` | 5% AANCA queue vs exact matched random | precision `0.333333` vs `0.215111`; difference `+0.118222`, 95% patient-group CI `[+0.040000, +0.257143]` | disagreement enrichment for one eligible junior-pathologist cohort across five patients |
| RIVA | four leave-one-annotator-out rotations | `0.476357` vs `0.426900`; difference `+0.049457`, 95% source-group CI `[+0.014037, +0.091632]` | natural independent-vote disagreement enrichment |
| MIDOG++ | two pairwise-expert rotations | `0.325967` vs `0.249392`; difference `+0.076575`, 95% image-group CI `[+0.021352, +0.138159]` | replication in a different public domain and reference design |
| PUMA | controlled label noise on a frozen 144/62 development/final split | retrieval `0.537739` vs `0.214379`; downstream macro-F1 `+0.006426` vs unchanged, 95% CI `[+0.003657, +0.009365]` | controlled-noise transfer; all seven internally frozen gates passed |

Every point difference in all four RIVA and both MIDOG++ rotations was non-negative.
This is a useful cross-dataset consistency result, but the aggregate intervals above
remain the inferential authorities and the datasets are not treated as independent
clinical trials.

## Negative evidence retained

- PanNuke H4 guided restoration was `-0.002156` macro-F1 relative to matched random,
  with its interval below zero.
- The earlier frozen NuCLS aggregate ranking rule failed, and its primary guided
  intervention was `-0.014633` below unchanged training with interval
  `[-0.026683, -0.002415]`.
- MoNuSAC passed retrieval but failed downstream and important-class safety gates.
- Only one of nine post-confirmation PUMA stress scenarios passed every class
  safeguard, even though all nine aggregate downstream lower bounds were positive.

These results prevent a broad automatic-correction claim and demonstrate that the
repository does not select only favourable experiments.

## No adjudicated natural-error claim

NuCLS, RIVA and MIDOG++ measure disagreement against independent expert votes; none
contains a blinded adjudication establishing which expert is biologically correct.
PUMA evaluates controlled corruption rather than paired natural pre/post review.
Consequently, the supported wording is “earlier enrichment of independent-expert
disagreements for review,” not “detection of pathologist mistakes.” Natural-data
intervention remains `retain_uncorrected`.

## Whole-project audit findings and changes

### Scientific and data integrity

- Group-safe OOF, final-fold isolation, immutable source labels and separate
  `pre_corruption_label`, `observed_label` and corruption metadata remain enforced.
- RIVA and MIDOG++ saved scores were committed publicly before either hidden reference
  was evaluated; verification confirmed no risk recomputation after reference access.
- The selected-candidate verifier refitted 220 models; all converged. The PUMA
  verifier then confirmed 44/44 convergence records, exact disjoint matched controls,
  whole-case neighbour exclusion, no development/final overlap, unchanged source
  labels and all seven gates.
- NuCLS and MoNuSAC readbacks again reproduced their negative complete-claim outcomes;
  the distinct NuCLS `JP.1`, RIVA and MIDOG++ ranking gates again passed.

### Code, dependencies and security

- The final suite passed `1182` tests with one documented platform-specific skip.
- Ruff lint and formatting passed; mypy found no issue in 105 source files.
- The locked environment is consistent, installed packages are compatible and
  `pip-audit` reported no known dependency vulnerability.
- Public Figshare downloads now require HTTPS, an approved canonical host, no URL
  credentials and either the default port or 443. Tests reject HTTP, file URLs,
  unexpected hosts, user information, non-443 ports and malformed URLs.
- Broad complexity lint identifies substantial historical technical debt in large
  frozen orchestration and preregistration modules. A release-time refactor was not
  attempted because those modules are active verifier authorities; changing them now
  would add more scientific-regression risk than presentation value. New work should
  be isolated rather than extending those modules further.

### Reproducibility and release consistency

- The standalone presentation verifier checks a closed 13-file allowlist, per-file
  hashes, the manifest root and fail-closed evidence boundaries without project
  dependencies.
- A new professor-release verifier additionally binds the sealed presentation to its
  upstream evidence authorities and checks exact key numbers, scope language, dates,
  the single-link root README and the concise handout. GitHub Actions runs it before
  installing the project environment.
- The synthetic end-to-end CLI completed successfully after the code changes.
- Primary PanNuke numeric evidence is independently recalculable from the public
  release. PUMA and later external checks are project-maintained saved-evidence
  readbacks unless their documentation explicitly states stronger separation.

### Presentation and performance

- A compact “Evidence at a glance” section now appears before the detailed method and
  reads all values from verified `evidence.json`. The full long-form evidence article
  remains below it for auditability.
- The professor brief remains a separate 750-word handout. The long page is not used
  as the only introduction.
- Six hero sprites were resized to the 512 px source ceiling while the runtime already
  renders them through a maximum 384 px cache. Their combined transfer fell from
  about 8.62 MiB to 1.24 MiB; the complete sealed package fell from 12.72 MiB to
  5.34 MiB.
- Browser checks at 1280 x 720 and 390 x 844 found no horizontal overflow, missing
  sprite, console error or warning. The page has one H1, no heading-level skips,
  duplicate IDs, broken internal anchors, unnamed buttons or images without `alt`.

## Final verification record

The following final-version gates passed locally on Windows:

```text
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv lock --check
uv pip check
uv run --with pip-audit pip-audit --local
uv run pytest
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke --runs-root artifacts/final-audit-smoke-runs
python -I scripts/present_demo.py --verify-only
python -I scripts/verify_professor_release.py
```

Final test outcome: `1182 passed, 1 skipped` in 604.46 seconds. The presentation has
13 files, manifest root
`395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`,
scientific stage `EXTERNAL_VALIDATION_COMPLETE` and presentation stage
`DEMO_COMPLETE`.

At audit closure, the configured Hostinger build check accepted this package, but
both remote verification and deployment stopped at SFTP authentication before a
backup or upload. The live domain therefore still serves the preceding release; no
remote file was changed. Restoring the configured credential is the only remaining
publication operation, not a scientific or build failure.

## What still requires genuinely new evidence

The next decisive step is a preregistered, blinded, multi-pathologist study on new
patient/WSI groups, with raw votes, ambiguity and abstention preserved, followed by
one untouched external-group evaluation and a multi-site with/without-AANCA workflow
comparison. Until those gates pass, the project should be presented as a strong,
auditable research prototype with encouraging disagreement-enrichment and controlled-
transfer evidence, not as clinically validated software.
