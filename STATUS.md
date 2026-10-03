# AANCA status

Updated: 3 October 2026

## Presentation UI note (22 August 2026)

The checked-in `artifacts/mvp_demo` package remains a single long-form article:
one CSS rhythm (`--para-gap` / `--block-gap` / `--section-space`), shared section
hairlines, and a full-viewport hero masthead. The hero uses an original Canvas 2D
Second-Look Review Field animation (canonical source
`src/histo_audit/assets/hero-review-field.js`, inlined into `index.html`) and six
checksum-bound local transparent nucleus sprites under
`src/histo_audit/assets/hero/nuclei/`. It shows many annotations, a paced audit pass,
a short review queue of copies while source marks stay in place, then a nucleus →
patch → AANCA-mark transformation. The split into four non-overlapping patches,
camera move, 45° rotation and transition to the mark colour overlap on one continuous
timeline rather than pausing at an intermediate grid. The completed mark holds for
1.8 seconds, then spins and dives fully through the camera before a 0.32-second fully
opaque black pause. Eight small stagger groups bring the cell field back for the next cycle. The
return is not a hard canvas cut. It does not claim
automatic diagnosis or label correction. Runtime rendering is capped at 60 frames per
second and uses hardware-aware pixel and cache budgets rather than scaling work with a
120–240 Hz display or an unnecessarily high device-pixel ratio.
On mobile, the audit now makes exactly two visible selections, keeps the initial field,
patch zoom, study view and mark below the hero copy, and reaches full black before the
field returns. A 390 x 844 Playwright capture sampled the complete loop every 275 ms;
the two selected nuclei and two queued copies remained visible and no mark fragment
survived the full-black frames.
The benchmark overview is implemented natively as a narrow, centred article-style
HTML/CSS section with five restrained text rows; it has no screenshot, decorative
panel or generated-image dependency. The Method workflow graphic has been removed;
four short article paragraphs now explain the controlled label intervention,
source-group-safe out-of-fold scoring, equal review budgets and the separate downstream
test alongside a checksum-bound minimalist workflow graphic. The graphic uses the
site's exact violet accent and distinguishes retrieval from downstream evaluation while
retaining recommendation-only output. A source-level audit against `SPEC.md`,
`PRE_REGISTRATION.md` and the grouped OOF implementation clarified that intervention
metadata records the injected change, risk is computed from OOF support for the
observed label, and restoration is limited to the PanNuke controlled experiment. The former bordered
`90-second summary` and redundant standalone research question have been removed, so
the benchmark setup flows directly into Method.

The author section now discloses that AI tools assisted throughout the project with
planning, code drafting and iterative implementation. It also states that Natan Smogór
directed the work, reviewed and revised AI-assisted outputs, made the final scientific
and engineering decisions, and remains responsible for the code, analysis and claims.

The footer is now one continuous four-column surface: AANCA, Study, Inspect and Terms.
The copyright, repository licence, third-party asset terms and non-clinical boundary
sit inside that grid without internal divider lines. The redundant visible PUMA commit
and `evidence.json` digest were removed; provenance remains available through the
machine-readable evidence, release and package manifest.

The Findings section uses one centred desktop sequence: seven exact registered
question-and-answer stages share one reversible GSAP timeline and one ScrollTrigger.
Its navigation is deliberately limited to a thin white line, seven white nodes and a
white progress stroke; it contains no mini-chart, quantitative glyph or atlas preview.
The active node now changes fill without SVG scaling, so all seven node centres remain
exactly on the shared one-pixel axis. Desktop finding rows use a 320 px transition
offset, leaving a clear editorial gap between adjacent H1–H7 answers.
The complete 36-entry forest plot remains the separate quantitative view below. H2
uses only the four saved subgroup ranges, H4 retains the adverse `-0.002156` macro-F1
result, and H6 remains explicitly unavailable rather than being plotted at zero.
Mobile uses the same seven answers in ordinary document flow with small static glyphs.
Under `prefers-reduced-motion`, the page shows a static white spine and all answers
without a tall sticky region. The hero and frozen scientific evidence were unchanged.

The package now contains thirteen checksum-bound files. Current validation:

- `uv run pytest tests/test_mvp_demo.py`: `9 passed`;
- `uv run pytest`: `1151 passed, 1 skipped` in 654.87 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy src`:
  passed; mypy reported no issues in 103 source files;
- the hero and Findings JavaScript syntax checks passed;
- standalone package verification: valid thirteen-file package, manifest root
  `6df01908b0a1bb45ab4fadf314f95fed32340454bec8cb035d22ecf2ec80772b`;
- synthetic CLI smoke run `20260822T134341.259054Z_synthetic_smoke_501e16e69c`
  completed successfully and its temporary output was removed after readback;
- NuCLS, MoNuSAC and PUMA evidence verification passed with their saved conclusions
  unchanged; the NuCLS paired pre/post feasibility endpoint remained unavailable and
  fail-closed;
- GitHub Actions run `32576578389` passed on Ubuntu and Windows for publication
  commit `6d6e90a5f0e6c7edfbb3a513c761cf95db1c23d9`;
- Playwright at 1440 x 900 confirmed the H1-H7 row/node sequence, final atlas state
  and reverse return to H1; 390 x 844 mobile had no sticky enhancement or horizontal
  overflow; reduced motion exposed all seven answers and both static evidence views;
  all three variants reported zero console errors and warnings.

The complete maintained suite and all release gates were repeated after the visual
corrections. Scientific evidence, accepted metrics and completion stages were
unchanged.

The release was deployed to
`mediumaquamarine-wombat-125861.hostingersite.com` after the deploy tool rebuilt and
verified the package, connected over SSH/SFTP and created rollback backup
`20260822-161538`. Only that domain's `public_html` was replaced. The page and every
followed asset returned HTTP 200. Independent cache-busted readback confirmed that
production `index.html`, `evidence.json`, `README.md` and `manifest.json` are byte-for-byte
identical to the checked-in package, and the GitHub `main` manifest is identical to the
local manifest with root
`6df01908b0a1bb45ab4fadf314f95fed32340454bec8cb035d22ecf2ec80772b`.
Production Playwright readback at 1440 x 900 and 390 x 844 reported no horizontal
overflow, console error or warning after lazy assets loaded. The mobile hero made
exactly two selections, queued two copies, loaded all six sprites without failure and
reached fully opaque black before beginning its next cycle.

## Current scientific stage

- `PRIMARY_STUDY_COMPLETE`: the frozen-feature PanNuke controlled-corruption
  benchmark and restoration experiment completed in the accepted recovery run.
- `EXTERNAL_VALIDATION_COMPLETE`: the prospectively frozen NuCLS genuine
  multi-rater evaluation completed; its primary ranking and downstream claims were
  **not supported**.
- An additional prospectively frozen controlled-external MoNuSAC benchmark
  completed; its retrieval gate passed, but the registered combined success rule
  was **not supported**.
- The frozen autoresearch-selected candidate completed a genuinely new-source PUMA
  controlled external confirmation. All seven internally pre-specified gates passed and the scoped
  PUMA evidence-readback verifier accepted the saved result. This supports
  controlled-noise transfer, not natural/pathologist-error detection or a second
  image-to-result replication.
- A post-confirmation PUMA realism stress found positive aggregate downstream lower
  bounds in all nine scenarios but full per-class safety in only one. Unreviewed
  natural-data `flag_exclude` remains prohibited.
- `DEMO_COMPLETE`: the checksum-verifiable static article package is built and
  deployed.
- `CONFIRMATORY_COMPLETE`: not reached.

Stage completion records that the prescribed evaluation ran and its evidence was
preserved. It does not mean the result was favourable. AANCA remains a
non-diagnostic research prototype, never modifies source annotations automatically,
and has not proved natural pathology errors, pathologist errors or clinical utility.

## PUMA new-source controlled confirmation

Study: `puma_new_data_confirmation_v1`

The candidate
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`
was frozen before the official PUMA archives were downloaded or any PUMA metric was
calculated. The official public release was hash-verified and mapped to its tumor,
lymphocyte/plasma-cell and other primary classes. A deterministic stratified split
placed 144 ROI/case groups (67,032 nuclei) in development and 62 groups (30,397
nuclei) in the final partition with no overlap.

The 144/62 development/final partition is an AANCA-defined split of the 206 public
PUMA ROIs. It is not the official hidden PUMA challenge test set. The downstream
intervention was `flag_exclude`: the highest-ranked 5% of training instances were
omitted from downstream training. They were not reviewed, corrected or automatically
relabelled by an expert, and source annotations remained unchanged.

Frozen controlled-corruption retrieval:

- 10% symmetric development corruption under four seeds;
- 5% queue, 3,352 reviews per seed and 13,408 review decisions total;
- AANCA precision `0.5377386634844868` versus exact matched-random
  `0.2143794749403341`;
- difference `+0.3233591885441528`, 95% whole-group interval
  `[0.2592505925988822, 0.3849444213529693]`;
- 7,210 injected changes found.

Frozen downstream result on untouched final groups:

- AANCA `flag_exclude` macro F1 `0.6463102793504962`;
- unchanged corrupted-training macro F1 `0.6398841192645623`;
- exact matched-random exclusion macro F1 `0.6382432313302706`;
- AANCA minus unchanged `+0.006426160085933802`, interval
  `[0.003657212432103269, 0.009365455935978527]`;
- AANCA minus matched random `+0.008067048020225648`, interval
  `[0.0040931053452727805, 0.011946725717818258]`;
- all seed directions were positive, every primary class-recall lower bound was at
  least `-0.01`, all 44 fits converged and all seven gates passed.

The project-coupled evidence readback rebuilt the official manifest and group split, confirmed
source labels unchanged, checked every corruption field, OOF group allocation,
complete-group neighbour exclusion and exact matched controls, and recomputed every
metric and 3,000-draw bootstrap decision. Evidence identities:

This readback recomputes metrics from saved predictions but does not retrain all 44
models. It is not third-party validation or a second image-to-result replication.

| Artifact | SHA-256 |
| --- | --- |
| Results | `e547d77a2ae5bffdaed9f64f83c369292fe11cd2e8a305d02fc3d2271131dd92` |
| Numeric evidence | `5f6d2d0fc65d95ee245d98c82b80547f0e0cf06c278acc40b48f44020e8c59d4` |
| Run authority | `c016dde668d6c7c2bdcbf6aad79d13892012cdd5b6ba87cd9d391d21ac4cf95b` |
| Verification record | `2f4b6d6c64d8dfa493a89be0111ca8752257dabce808d4d9fa35933cdd9c5d68` |

The accepted conclusion is `controlled_noise_transfer_supported`. PUMA publishes
final expert-checked labels but no paired natural pre/post review states, so natural
error detection, pathologist error and real workflow benefit remain untested.

Public-history and verification scope: the PUMA protocol, configuration and result
first appeared together in commit
`c5bd44193b2abd67bc7e7f1bd9384aa87435d500`. Local authorities record the
intended freeze-before-metrics order, but GitHub does not independently timestamp
it. `scripts/verify_puma_new_data_confirmation.py` rebuilds the manifest and
recalculates saved-evidence decisions while importing maintained PUMA helpers; it
does not retrain all 44 models from source images.

## PUMA post-confirmation realism and clean-label stress

Study: `puma_realism_stress_v1`

After the primary PUMA outcome was opened, the unchanged candidate was subjected to
nine explicitly exploratory scenarios: clean labels; symmetric 1%, 2.5% and 5%;
directional 5% and 10%; group-conditional 5% and 10%; and independent
geometry-dependent 5% corruption. Candidate selection and rescue were prohibited.

- every scenario had a positive macro-F1 lower bound versus unchanged and exact
  matched-random training;
- only `group_conditional_10pct` passed all registered gates;
- the other eight failed only the per-class recall non-degradation rule;
- on clean labels, macro F1 increased by `+0.004540`, interval
  `[+0.000322, +0.009156]`, but `other` recall fell by `-0.013733`, interval
  `[-0.025390, -0.002789]`;
- the geometry-dependent generator used released bounding-box geometry and was
  recorded as independently separated from the pixel ResNet auditor;
- all stress models converged and source annotations remained unchanged.

Stress evidence identities:

| Artifact | SHA-256 |
| --- | --- |
| Frozen stress config | `00214940ee2cc1faf51202e23fe800a7f42a4f9b2936dae26bbfe20e3ee2555a` |
| Results | `0091c14075e7304e0a6effef5a398c32c692f40aaefe7ee7d0b126a101cf7892` |
| Numeric evidence | `186861575266985b4e1071190a957fd45769752e69e138633e94fa3be6eb4d39` |

The policy consequence is binding: aggregate benefit cannot override a class-safety
failure, and natural-data action remains `retain_uncorrected` until reviewer-gated
interventions pass a fresh class-safe external study.

## PUMA audit-time-label sensitivity

Study: `puma_audit_time_label_sensitivity_v1`

The PUMA primary OOF plan used pre-corruption labels for group stratification. This
post-confirmation sensitivity kept the candidate, seeds, final groups, queue,
intervention and success rule fixed but rebuilt every seed's audit folds and exact
neighbours using only `observed_label`. PUMA outcomes were already known, so the
result is exploratory and cannot serve as independent confirmation.

- only 22.21%-33.08% of row-level fold assignments matched the original plan;
- AANCA precision was `0.5381861575178998` versus exact matched random
  `0.21515513126491645`;
- retrieval advantage was `+0.3230310262529833`, interval
  `[0.25973397585007, 0.38131225104714983]`;
- AANCA minus unchanged macro F1 was `+0.00667944296150233`, interval
  `[0.004141475700661824, 0.009505712132344529]`;
- AANCA minus matched random was `+0.009069355900155174`, interval
  `[0.005855100059258797, 0.01246062687290453]`;
- all seed directions, every class-recall guard, convergence and observed-label
  group/neighbour guard passed.

| Artifact | SHA-256 |
| --- | --- |
| Frozen config | `ed6fd1e85d15604efc331b634a0d7604ca2675ba58345aa31386c266781e661f` |
| Results | `8f524b236995a495048a0955ebf930e14e732a1214d3856d3711571af13fd5cd` |
| Numeric evidence | `aad24975c29f004e6f5b44575ea5b3d97daa1122457277cca902d69f36e64903` |

The accepted sensitivity conclusion is that the controlled PUMA result does not
depend on access to a clean fold-assignment label. Natural errors and real workflow
benefit remain unevaluated.

## NuCLS supervised-QC pairing feasibility

The official raw SQLite source was downloaded and inspected under the prospective
pairing protocol. It contains one class state per stable annotation element and no
previous-label, replacement-label or revision-history table. The official corrected
and uncorrected releases are different FOV quality tiers, and repeated element IDs
arise only where geometry intersects multiple FOV records. No stable ID carries two
class labels.

The paired natural pre/post endpoint is therefore `unavailable`; comparing unmatched
cohorts or inferring former labels would be confounded or circular. The report is
`reports/nucls_supervised_qc_feasibility.md`, and natural-data action remains
`retain_uncorrected`.

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
- a fresh extraction passed the standalone evidence recalculator before upload; this
  is a software-independence check, not third-party validation.

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

Publication evidence:

- scientific result and presentation commit `296f8afe1e7b2eb8d0d14f5ffd2c9599d0162ab0`
  was pushed to `origin/main`;
- Hostinger connection and pre-deploy build verification passed;
- rollback backup `20260820-235533` was created before replacing `public_html`;
- the page, `evidence.json` and all checked external animation assets returned HTTP
  200 after deployment;
- live `index.html`, `evidence.json`, `README.md` and `manifest.json` matched their
  local deployed files byte for byte;
- live `index.html`: 223,814 bytes, SHA-256
  `5871a7a8f7c620a7dcd8ed0b8f10c948bf7cdfdc125928f5577cd768076aaae8`;
- live machine evidence: 29,782 bytes, SHA-256
  `68de15f65a0e69c111f660354b05e2439666b613a3ce2e2482303b5baa681bbb`;
- deployed presentation manifest root:
  `113e3e8d20cf86dcde4afb09ffd9eb21f9aa78ab3733364e13c26d04945a8827`.

## Bounded autoresearch development search

Study: `monusac_aanca_expanded_development_v1`

Date completed: 2026-08-21

Role: controlled development only; no completion stage or real-use claim added

The implementation adapted the fixed-evaluator, bounded-trial, append-only-ledger
and keep/discard mechanics of `karpathy/autoresearch` while retaining AANCA's stricter
patient grouping, nested OOF audit evidence, exact matched control and untouched-test
rules. The official MoNuSAC test, both NuCLS subsets and the PanNuke final fold were
unavailable to candidate generation, screening and selection.

Executed search:

- all 44 eligible official MoNuSAC training patients were used only in nested
  development;
- 240 ranking configurations and 160 downstream configurations were screened;
- 12 finalists were frozen before their complete results;
- full evaluation used five outer patient folds, four inner audit folds, four
  corruption seeds, five exact matched-random repetitions and 3,000 whole-patient
  bootstrap draws;
- the search covered 64 px, 128 px, multiscale, morphology/statistics and Phikon-v2
  views; probability, neighbour and fixed-hybrid risks; 0.5%-10% budgets; restoration,
  weighting, downweighting and exclusion; and downstream regularisation/class
  weighting;
- two candidates passed every registered gate. The simpler best passing candidate
  without known TCGA encoder-pretraining overlap was selected.

Frozen candidate:

`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`

- multiscale 64+128 px ImageNet ResNet-18 context;
- unbalanced L2 `0.1` audit model;
- fixed hybrid risk with self-confidence weight `0.6`, fold-safe 31-neighbour
  disagreement weight `0.4`;
- relaxed balanced 5% queue;
- `flag_exclude`: selected controlled-training rows receive zero weight while source
  annotations remain unchanged;
- balanced L2 `0.01` downstream model.

Selected controlled-development result:

- candidate macro F1 `0.5471943284781748`;
- unchanged corrupted-training macro F1 `0.5041042261017459`;
- exact matched-random macro F1 `0.492794942022801`;
- candidate minus unchanged `+0.04309010237642889`, 95% whole-patient interval
  `[+0.03255265089898305, +0.05506540367582292]`;
- candidate minus exact matched random `+0.054399386455373844`, interval
  `[+0.034938293224264415, +0.07550715895022347]`;
- all four corruption-seed differences were positive against both comparators;
- retrieval precision `0.9479659014179609`, difference over exact matched random
  `+0.40395003376097227`, interval
  `[+0.36055964504995774, +0.4489032325505159]`;
- the lowest important-class recall lower bound was `-0.0064620394516020745`, above
  the frozen `-0.01` safety limit;
- all five frozen success gates passed.

The development intervals exclude zero but are not independent post-selection
confirmation intervals. They establish a strong selected development result, not
natural pathologist-error detection or real-use superiority.

Authority and verification:

| Evidence | SHA-256 |
| --- | --- |
| Expanded config | `370b7135858682d0dea52c035768b2fed72acc1fe74a1ddd67996780ad703692` |
| Patient partition | `93087764cf5ce3dd62474ac4da790ff6871d2deff6291d508e02c49ec75f2d2d` |
| Runtime-only amendment | `2e14a57c72dac193bac8c3179baa90e66b4271a4d3a4fef8f4f8cc0610324a98` |
| Parent authority | `3ef82963925cea7d20332f13488578ded5eba1df750c52cb55cef69521580042` |
| Append-only parent ledger | `1e5378ebbb1a02cdd003fd6bed96d78a200b53be210b746c1477d46a2025e728` |
| Selected candidate record | `229bc293b3ba7c3909423178552f5f3789f00411223c2f87b5185eee1542487d` |
| Convergence evidence | `d10fbcb3179abe6058ae43231663f3aeefc7d754c40bac2bfa6fdea1a4abae38` |

`scripts/verify_aanca_selected_candidate.py` rebuilt the frozen candidate from the
pinned archive and authority. It reproduced all stored summary metrics exactly and
reported `220/220` converged fits: 100 hard-label and 120 weighted fits. The detailed
local artifact is 50,604 bytes and reports no final-test use, natural-error evaluation
or source-label modification.

The selected record is checksum-validated and always loads as development-only with
`executable_action_until_new_confirmation: retain_uncorrected`. A measured-utility
queue is implemented as
`percentile(annotation_inconsistency_score) × max(utility_lower_bound, 0)` and fails
closed without group-safe OOF audit evidence and nested cross-fitted measured expert
utility. The PUMA controlled new-source confirmation is complete and positive. The
natural-case configuration remains `INITIALISED` because no paired blinded expert
outcomes or prospective multi-site workflow responses are present.

Final validation for this development-search work:

- focused autoresearch, runtime-amendment, frozen-candidate, measured-utility and
  current-policy tests: `23 passed` in 3.59 seconds;
- runtime-amendment analyser: all 12 frozen finalists complete, selected candidate
  unchanged, executable action `retain_uncorrected`;
- selected-candidate convergence verifier: exact metric reproduction and `220/220`
  converged fits;
- complete maintained suite: `1125 passed, 1 skipped` in 582.39 seconds; the skip is
  the documented Windows/POSIX open-file rename difference;
- `ruff check .`: passed;
- `ruff format --check .`: all 206 maintained Python files formatted;
- `git diff --check`: passed.

## Final validation for the PUMA and expert-assessment update

Executed on 2026-08-21 after all code, policy and evidence changes:

- focused PUMA confirmation, realism stress, audit-time-label sensitivity, memory,
  current-policy and future-protocol tests: `19 passed`;
- complete maintained suite: `1141 passed, 1 skipped` in 579.83 seconds; the single
  skip is the documented Windows/POSIX open-file rename difference;
- `ruff check .`: passed;
- `ruff format --check .`: all 216 maintained Python files formatted;
- project-coupled PUMA evidence readback: official manifest rebuilt, zero
  development/final overlap, source labels unchanged, corruption fields exact,
  every query group excluded from its neighbours, exact matched controls,
  retrieval/downstream metrics and 3,000-draw bootstrap recalculated; all seven
  internally pre-specified gates passed. The readback did not retrain all 44 models
  and is not third-party validation;
- NuCLS supervised-QC feasibility verifier: official 27,648,000-byte SQLite source
  re-read, 0 stable element IDs with multiple class labels, paired pre/post endpoint
  unavailable and fail-closed action `retain_uncorrected`;
- PUMA audit-time-label sensitivity: all seven gates passed with observed-label OOF
  allocation; no candidate or claim-boundary change;
- PUMA stress: all nine aggregate downstream intervals positive, one scenario passed
  all class-safety gates, and eight failures retained without post-outcome tuning;
- selected-candidate verifier: frozen metrics reproduced exactly, final external test
  unused and all `220/220` fits converged; convergence evidence remained
  `d10fbcb3179abe6058ae43231663f3aeefc7d754c40bac2bfa6fdea1a4abae38`;
- deterministic synthetic data returned `verified_existing`;
- synthetic smoke run `20260821T100430.181783Z_synthetic_smoke_6bc44efa2c` completed
  successfully with metrics, report and figures;
- `git diff --check`: passed before the status-only append and is rerun at handoff.

## Repository maintenance and retention audit — 2026-08-21

Engineering maintenance M21 is complete without changing the scientific completion
stage or expanding any claim. The active repository now keeps one authoritative copy
of released evidence and only the accepted full PanNuke pilot and primary run.

Executed cleanup:

- re-verified the accepted primary recovery and accepted pilot before retiring older
  run directories; both integrity checks were valid with no missing, added or changed
  paths and roots `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`
  and `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`;
- moved 18 interrupted, ineligible, smoke and rehearsal runs plus superseded caches,
  preview builds and test output to
  `C:\Users\NATAN\Documents\AANCA_cleanup_quarantine_20260821`;
- first quarantined 4,222 identified files totalling about 43.78 GiB, then
  permanently removed the verified superseded-run, cache, QA, smoke and non-release
  classes after the same-volume quarantine exhausted `C:`; approximately 43.79 GiB
  was physically reclaimed;
- temporarily retained the five-file, 3.245 MiB
  `mvp_demo_before_author_section/` rollback outside the repository, then removed it
  after final package and browser verification; the deleted material is not
  recoverable;
- reduced active `artifacts/` from about 92.1 GiB to 48.456 GiB and retained only the
  133,735-byte release hero under `output/`;
- removed the redundant tracked QA evidence mirror, nine superseded screenshots,
  orphaned capsule scaffolding, stale machine-local reports, the empty notebooks
  placeholder and the duplicate presentation CI workflow;
- moved static package verification into the Linux leg of the maintained
  cross-platform scientific workflow;
- configured the three 41–78 MB PUMA numeric-evidence archives for Git LFS while
  retaining their independently verifiable content;
- consolidated NumPy archive publication, confusion-derived macro-F1/recall, figure
  saving and checksum-pinned YAML loading; deliberately independent frozen
  publication and verifier implementations remain separate;
- reduced the measured maintained package categories from 104,231 to 104,116 Python
  lines while adding shared validation and tests;
- extracted the stable presentation stylesheet into one packaged asset, reducing
  the consistently measured nonblank count in `mvp_demo.py` to 4,060 lines and the
  complete nonblank Python workspace from 155,697 to 154,589 lines (net `-1,108`);
  the stylesheet has 810 nonblank lines.

Validation and corrections:

- an early focused run found two failures because PyYAML parsed unquoted frozen dates
  as `date` objects; ISO-8601 normalisation was added and all affected tests passed;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: 215 files already formatted;
- `uv run pytest`: `1145 passed, 1 skipped` in 837.69 seconds; the single skip is the
  documented Windows/POSIX open-file rename difference;
- after the presentation extraction, the focused suite passed `58` tests in 47.03
  seconds; an earlier attempt had produced no accepted result because a full disk
  interrupted pytest while writing its cache;
- `uv build --wheel`: passed and the built wheel contained the extracted presentation
  stylesheet;
- `uv run histo-audit data generate-synthetic --config configs/smoke.yaml`: generated
  deterministic definition
  `791fe34c3bb9042b73badd8209afa1b2b673922e20f2da2da28e9a70d67525b2`;
- `uv run histo-audit experiment smoke`: completed run
  `20260821T111042.663068Z_synthetic_smoke_4d457ebe70`;
- `python -I scripts/present_demo.py --verify-only`: valid `DEMO_COMPLETE` five-file
  package with root
  `1e4e403e08aefc8e9d2e4b18a1b44d24c30c4ab4df106fba45addfd598ca2b4b`;
- `uv run python scripts/verify_puma_new_data_confirmation.py`: verified; all seven
  frozen gates and all 44 model-convergence checks passed;
- `uv run python scripts/verify_nucls_supervised_qc_feasibility.py`: paired pre/post
  endpoint correctly unavailable, `retain_uncorrected` retained;
- `uv run python scripts/verify_aanca_selected_candidate.py`: exact stored metrics
  reproduced, `220/220` fits converged, final external test unused and source
  annotations unchanged;
- direct system-Python attempts to run those three package-dependent verifiers were
  rejected with `ModuleNotFoundError`; they produced no evidence and were rerun in
  the pinned `uv` environment as listed above;
- workflow YAML parsing and `git diff --check`: passed before this append and are
  rerun immediately before publication.

The complete retention inventory and final deletion boundary are in
[`reports/repository_maintenance_2026-08-21.md`](reports/repository_maintenance_2026-08-21.md).

## Presentation-ready release validation — 2026-08-21

The final repository and article were regenerated from the current evidence
authorities and validated after the retention audit. This pass changed no accepted
scientific result and did not expand the claim boundary.

Final gates:

- `uv run pytest`: `1147 passed, 1 skipped` in 805.06 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 215 maintained Python files formatted;
- `uv run mypy src`: no issues in 103 source files;
- focused final demo regression after whitespace correction: `8 passed`;
- `uv build --wheel`: passed; the isolated wheel installed successfully, exposed the
  complete CLI and contained all required external-validation and presentation
  modules (114 archive entries);
- deterministic synthetic generation: definition
  `791fe34c3bb9042b73badd8209afa1b2b673922e20f2da2da28e9a70d67525b2`;
- synthetic smoke: completed run
  `20260821T131337.066103Z_synthetic_smoke_9479d2b84a`; its generated dataset, run
  and build products were removed after validation;
- primary standalone evidence recalculator: 33 reported comparisons and 3 explicitly
  unavailable comparisons recalculated, including adverse H4
  `-0.0021560596665870235`;
- frozen NuCLS verifier: both subsets recalculated, primary natural and downstream
  claims not supported;
- MoNuSAC verifier: four ranking candidates and 2,000 whole-patient bootstrap draws
  verified, combined success rule not supported;
- PUMA saved-evidence readback: official manifest rebuilt, all 44 stored fits were
  marked converged and all seven internally pre-specified controlled-new-source gates
  passed; this is project-coupled readback, not third-party validation or retraining;
- selected-candidate verifier: exact development metrics reproduced with `220/220`
  converged fits and no final-test use;
- NuCLS supervised-QC feasibility: official database re-read, no paired natural
  pre/post nucleus label endpoint found, action `retain_uncorrected`;
- final five-file demo passed both verifiers with manifest root
  `edbba03401c50eb4a2e0fd2e5a43c744b0726af6c6a6aaed3e5ee3d5c0e29426`;
- Playwright desktop, iPhone 15 and reduced-motion QA: no horizontal overflow,
  broken images, missing anchors, duplicate IDs, console errors or warnings; all 13
  page resources returned HTTP 200, mobile navigation worked, every reveal became
  visible and the findings animation was observed while reduced motion remained
  fully static and readable;
- local Markdown links and `git diff --check`: passed.

The noncanonical 3,402,366-byte pre-polish rollback and its now-empty quarantine
root were deliberately removed after these checks. They are not recoverable; the
verified current package and all authoritative evidence remain intact.

## Publication record — 2026-08-21

- release-content commit
  `b3d6eba64dd3b9ae70fa01cde11777fdd844dedc` was pushed to GitHub `main` and
  confirmed by remote ref readback;
- the five-file package was deployed in replace mode to
  `mediumaquamarine-wombat-125861.hostingersite.com` after a remote backup was
  created with ID `20260821-153206`;
- the deployed page, `evidence.json` and all declared runtime assets returned HTTP
  200;
- direct SFTP readback matched the local SHA-256 for all five source files;
- Hostinger `hcdn` recompresses the served PNG representation, while the source PNG
  stored in `public_html` remains byte-identical to the manifest-bound local file;
- production Playwright readback on desktop and iPhone 15 found no horizontal
  overflow, hidden reveal, broken image, console error or warning. The current
  status, AANCA v2 plan and complete findings section were present.

## Public-audit remediation and republish — 2026-08-21

The repository and article were corrected against the later public audit without
changing any frozen metric, source label, candidate, gate or scientific completion
stage.

Implemented corrections:

- GitHub Actions checkout now sets `lfs: true`, so the three tracked PUMA NPZ files
  are materialised before package verification and tests; a repository regression
  test guards this requirement;
- the Method review-queue animation remains unchanged when WebGL works, while an
  inline static SVG replaces the canvas when WebGL, Three.js or the rendering
  context is unavailable;
- the “What the study actually learned.” section and its seven findings remain in
  normal article flow;
- `reports/aanca_expert_system_assessment_2026-08-21.md` was replaced by
  `reports/aanca_internal_technical_assessment_2026-08-21.md`; it now identifies
  itself as a project-maintainer review rather than external peer review and removes
  subjective numeric grades;
- public documentation now records that the PUMA protocol, configuration and result
  first appeared together in commit
  `c5bd44193b2abd67bc7e7f1bd9384aa87435d500`, so GitHub does not independently
  prove the intended pre-outcome timing;
- the PUMA verifier is consistently described as a scoped saved-evidence readback:
  it imports maintained PUMA helpers and consumes saved predictions and convergence
  flags, rather than independently retraining all 44 models from source images;
- `evidence.json` fail-closed records both limitations under `publication_limits`.

Final local gates:

- focused demo, CI and PUMA regression: `15 passed`;
- complete maintained suite: `1148 passed, 1 skipped` in 587.52 seconds; the single
  skip is the documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: 216 files formatted;
- `uv run mypy src`: no issues in 103 source files;
- `git diff --check`: passed;
- deterministic synthetic generation: definition
  `791fe34c3bb9042b73badd8209afa1b2b673922e20f2da2da28e9a70d67525b2`;
- isolated smoke run
  `20260821T144259.150479Z_synthetic_smoke_b7b0689fa9`: completed with
  `success: true` outside the repository;
- final five-file presentation: valid, manifest root
  `2ead3c5febb7fe904294b045563013778ac91190f981a5dbdf1b17cd172ddada`;
- Playwright desktop WebGL, forced-no-WebGL and 390 px mobile checks: no horizontal
  overflow, console errors or warnings; the normal renderer reported
  `threejs-review-queue`, and the forced-no-WebGL run reported
  `static-fallback` / `webgl-unavailable` with the canvas hidden and fallback shown.

Hostinger publication:

- previous live version backed up as `20260821-164856`;
- package deployed in replace mode to
  `mediumaquamarine-wombat-125861.hostingersite.com`;
- page, `evidence.json` and every declared external animation asset returned HTTP
  200;
- cache-busted production readback contained the static fallback, public-history
  disclosure and complete findings section;
- production `index.html`, `evidence.json`, `README.md` and `manifest.json` matched
  the local SHA-256 identities byte for byte.

## AANCA v1 public-claim and reproducibility clarification — 2026-08-21

This release pass changed no frozen AANCA v1 candidate, parameter, partition,
intervention, metric, confidence interval, success gate or source annotation. It
corrected how the existing evidence is described and made the professor-facing
release easier to inspect.

Public clarifications:

- PUMA is now consistently described as an internally frozen controlled new-source
  confirmation with internally pre-specified gates. The protocol, configuration and
  result first entered public Git history together, so GitHub does not independently
  verify the intended pre-outcome ordering;
- the 144/62 development/final partition is explicitly identified as an AANCA-defined
  split of the 206 public PUMA ROIs, not the official hidden challenge test set;
- `flag_exclude` is defined exactly: the highest-ranked 5% of controlled training
  instances were omitted from downstream fitting. They were not expert-reviewed,
  corrected or automatically relabelled;
- the primary verifier is a standalone software recalculator, not third-party
  validation. The PUMA verifier is a project-coupled saved-evidence readback that did
  not retrain all 44 models;
- the adverse downstream statement is scoped to the original PanNuke benchmark;
- the demo now includes a 90-second evidence summary, exact footer evidence identity,
  SRI-bound external animation scripts and an accessible static Method schematic;
- `PROFESSOR_BRIEF.md`, `CONTRIBUTIONS.md`, `CITATION.cff` and `LICENSE` record the
  one-page assessment, author/AI roles, citation metadata and code/data rights
  boundary. No custom domain or archival DOI is claimed before the owner creates it;
- GitHub Actions now runs `mypy` against the maintained Windows target before its
  full suite. Linux still runs the complete runtime test suite; intentional Win32 API
  calls are not misclassified by Linux typeshed. Missing-import suppression is
  limited to the two optional, runtime-guarded research dependencies
  `huggingface_hub` and `transformers`.

Final local evidence:

- complete suite: `1150 passed, 1 skipped` in 754.70 seconds; the single skip is the
  documented Windows/POSIX open-file rename difference;
- final demo/CI regression after source and stylesheet stabilisation: `11 passed` in
  0.92 seconds;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 216 maintained Python files formatted;
- `uv run mypy src`: no issues in 103 source files;
- `uv build --wheel`: passed, including the presentation stylesheet and five sealed
  benchmark-icon assets;
- standalone launcher verification: valid five-file package, scientific status
  `EXTERNAL_VALIDATION_COMPLETE`, manifest root
  `c2db7e0f5a7d9b8704e49725132e7a285ae14e7f7924a6015d9f9c4057e5759c`;
- final Playwright desktop and 390 px mobile readback: no horizontal overflow,
  console error or warning; all five benchmark icons and the static Method SVG
  rendered, lazy images loaded after traversal, every reveal became visible, and the
  mobile menu opened, closed on Escape and restored its ARIA state.

Hostinger publication:

- connectivity and the local five-file seal were verified before upload;
- the preceding live release was backed up under ID `20260821-223531`;
- the package was deployed in replace mode to
  `mediumaquamarine-wombat-125861.hostingersite.com`;
- the page, evidence and external animation assets returned HTTP 200;
- cache readback for `index.html`, `evidence.json`, `README.md` and `manifest.json`
  matched the local SHA-256 identities byte for byte.

## Local presentation alignment pass — 2026-08-21

Browser-comment follow-up changed only the generated presentation hierarchy:

- the PanNuke QC and primary-integrity statistic grids now centre each value and its
  label inside equal-height cells;
- the exact seed-identity disclosure is a bordered evidence card with a hash marker,
  scope note, visible action and rotating open-state chevron;
- the repository card now starts after a full editorial gap below the reproduction
  prose;
- the scientific evidence, identifiers, completion stages and claim boundaries are
  unchanged.

The five-file package verifies with manifest root
`c2db7e0f5a7d9b8704e49725132e7a285ae14e7f7924a6015d9f9c4057e5759c`.
Playwright at 985 x 698 confirmed the centred QC and integrity grids, both seed-card
states and the repository spacing.

Validation after the alignment pass:

- `uv run pytest`: `1150 passed, 1 skipped` in 724.65 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 216 maintained Python files formatted;
- `python -I scripts/present_demo.py --verify-only`: valid five-file package.

## Sprite-backed hero package refresh — 2026-08-21

The presentation-only Second-Look Review Field was refreshed without changing an
AANCA v1 candidate, split, label, metric, confidence interval, success gate or
scientific completion stage. The canonical Canvas 2D source is now
`src/histo_audit/assets/hero-review-field.js`. Six local transparent nucleus PNGs are
copied into the generated article and bound by the closed manifest, replacing the
procedural nucleus drawing while preserving the same non-diagnostic expert-review
metaphor and the static reduced-motion state.

Final validation:

- `uv run pytest`: `1151 passed, 1 skipped` in 642.85 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 216 maintained Python files formatted;
- `uv run mypy src`: no issues in 103 source files;
- `git diff --check`: passed;
- `node --check src/histo_audit/assets/hero-review-field.js`: passed;
- the final 60 Hz render-cadence and loop-timing follow-up passed `9` focused
  demonstrator tests in 1.32 seconds; the browser readback exposed bounded frame counts,
  all six sprites ready and no failed asset;
- `uv build --wheel`: passed; the wheel contains the hero script, stylesheet and all
  six sprite assets;
- standalone launcher verification: valid eleven-file package, scientific status
  `EXTERNAL_VALIDATION_COMPLETE`, manifest root
  `b58c2919742fbc43135d091dbbf8f856fba24116c99615fbd28c3cfa24000a1b`;
- local Playwright checks at 1440 x 900 and 393 x 727 found no horizontal overflow,
  console error or warning; the full-height canvas rendered and every sprite request
  returned HTTP 200.

Hostinger publication:

- the preceding live release was backed up under ID `20260822-001045`;
- the eleven-file package was deployed in replace mode to
  `mediumaquamarine-wombat-125861.hostingersite.com`;
- direct SFTP readback matched all eleven local files byte for byte;
- the Hostinger CDN served all PNGs successfully but recompressed their public HTTP
  representation, while `index.html`, `evidence.json`, `README.md` and
  `manifest.json` remained byte-identical to local sources;
- production Playwright readback found no horizontal overflow, console error or
  warning and observed the running `SELECT_AND_CLONE` hero state.

## Dynamic loop pacing and constrained-device optimisation — 2026-08-22

The presentation-only hero timing was balanced without changing any scientific
input, output, claim or completion stage. The four genuine desktop selections now
complete their queue at about 12.6 seconds. The following 2.45-second transformation
uses a single shared timeline: the first patch keeps moving while three siblings
separate, the camera pulls out, the group rotates and the tissue detail gives way to
colour. There is no held 2 x 2 intermediate state. The completed mark holds visibly
for 1.8 seconds. The final return performs more than two additional rotations
over 1.35 seconds, grows to 120 times its settled scale, continues to 150 times during
the fade, reaches an unbroken black frame and holds full black for 0.195 seconds.
During the following 0.46-second reveal, the black layer clears in the first 24% and
the cells enter in eight small deterministic groups rather than appearing together.

The renderer now:

- caps Canvas work at 60 rendered frames per second even on 120–240 Hz displays;
- retains a fixed 60 Hz simulation step so timing does not depend on refresh rate;
- selects constrained, balanced or high render profiles from available hardware,
  memory and data-saver signals;
- bounds Canvas pixels and pre-rendered patch-cache size per profile;
- continues to use decoded local sprite caches, `createImageBitmap` where available,
  zero per-frame DOM mutation, and visibility/intersection suspension;
- preserves the static completed mark under `prefers-reduced-motion`.

Local browser evidence for the final candidate:

- 1440 x 900 normal motion: during the same frame sequence the camera remained in
  motion while sibling reveal reached 0.856 and rotation had already begun; the dive
  crossed 100 times scale with no visible logo fragment and no horizontal overflow,
  console warning or page error. The completed logo remained settled for 1.809
  measured seconds, the shortened black hold measured 0.183 seconds, and screenshots
  at reveal progress 0.268 and 0.594 showed distinct cell groups;
- simulated constrained Pixel 5 viewport: 393 x 727 CSS pixels, DPR 2.75, two logical
  processors, 2 GB reported memory, data saver enabled and 4x CPU throttling; the
  constrained profile selected DPR 1.25, kept all three genuine mobile selections,
  completed the final synchronised repeat cycle in 18.033 seconds at 60.002 rendered
  frames per second and emitted no in-loop long tasks,
  errors or overflow;
- reduced-motion desktop readback reached the static completed mark with four genuine
  selections, all six sprites loaded and no overflow;
- focused demonstrator tests: `9 passed`;
- final canonical closed-package manifest root:
  `b58c2919742fbc43135d091dbbf8f856fba24116c99615fbd28c3cfa24000a1b`.

This local refresh has not been redeployed to Hostinger in this change set.

## Official domain and repository presentation refresh — 2026-08-22

The owner established [`aancastudy.org`](https://aancastudy.org) as the official
project website. DNS resolution and direct HTTPS access returned successfully before
the repository was updated. Historical Hostinger subdomain references remain in
dated publication records because they identify the host verified during those
specific releases.

Repository presentation was refreshed without changing scientific code or evidence:

- the README now opens with a centred project identity, concise navigation, a
  domain badge and a linked presentation hero;
- the official website replaces the temporary hosting address in current citation
  metadata and current repository guidance;
- the one-page professor brief exposes both the website and public repository;
- GitHub repository metadata now includes the official homepage, a concise project
  description and relevant discovery topics.

At the owner's request, the complete test suite was not repeated for this
documentation-only change. Validation was limited to Markdown link resolution,
`CITATION.cff` parsing, whitespace inspection and review of the exact Git diff.

## NuCLS independent-pathologist leave-one-out validation — 2026-08-23

The frozen current AANCA 64+128 px hybrid candidate was evaluated once against
natural independent-pathologist disagreement in NuCLS `U-control`. Raw `JP.1`
geometry, raw class and H&E pixels were assembled before the hidden reference was
opened. Patient identity defined five OOF folds and all bootstrap clusters. `JP.1`
was removed from the reference; aggregate P-truth fields were not used. A strict
majority of at least two mappable votes from other individual pathologists defined
binary agree/disagree outcomes.

The immutable authorities are candidate SHA-256
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`, config
SHA-256 `3e2f3e269385a0225d6c58af263ea1706932fa98dbb19bd7f7599980a4f1d6cd`
and protocol SHA-256
`fcccdfbdc9cfe00b96d173f9cba1c299f0a83db696eb382c9958201862c88d30`.

Only `JP.1` met the frozen public-geometry and five-patient requirements. `JP.2` was
excluded for only two patient groups; `SP.1`--`SP.3` and `JP.3`--`JP.6` lacked public
individual raw geometry. No pooled multi-pathologist estimate was produced.

Primary 5% outcome:

- 898 binary-reference-eligible nuclei, including 105 disagreements;
- 45 reviewed nuclei; AANCA found 15 disagreements;
- AANCA precision `0.333333`, recall `0.142857`;
- mean precision across 100 disjoint exact matched-random queues `0.215111`;
- precision difference `+0.118222`, patient-bootstrap 95% CI
  `[+0.040000, +0.257143]`;
- enrichment `1.549587`, 95% CI `[1.073620, 10.000000]`;
- 4,946 valid of 5,000 patient-bootstrap draws; 54 were undefined after cluster
  resampling;
- frozen primary gate: PASS.

Secondary AANCA versus matched-random precision differences were `+0.213333` at 1%,
`+0.148889` at 2%, `+0.118222` at 5% and `+0.078556` at 10%. AUPRC was `0.237584`.
At 5%, no observed-class point estimate was adverse: tumor `+0.060000`, stromal
`+0.025000`, sTIL `+0.255294`. These class results are descriptive and based on small
selected counts; they are not separate confirmatory gates.

Outcome accounting retained 793 consensus agreements, 105 disagreements, 53
ambiguous cases and 368 cases with insufficient reference. Strict consensus was
available in `94.43%` of cases meeting the minimum vote count; `JP.1` agreed with it
in `88.31%` of binary-reference-eligible cases.

Execution and verification:

- `uv run python scripts/run_nucls_independent_pathologist_validation.py`: completed
  as `EXTERNAL_VALIDATION_COMPLETE`, primary gate PASS;
- `uv run python scripts/verify_nucls_independent_pathologist_validation.py`: 11
  artifacts verified, metrics recalculated without retraining, source annotations
  unmodified;
- the reporting-complete rerun preserved exact `metrics` JSON and byte-identical
  `scored_samples.csv`, `matched_random_queues.json`, `numeric_evidence.npz` and
  `enrichment_curve.csv`; the initial frozen run remains under the explicit
  `*_initial_frozen_run` backup paths;
- `uv run pytest`: `1164 passed, 1 skipped` in 602.77 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 220 maintained Python files formatted.

The protocol and execution entered one repository change and feasibility
prevalence/category counts had been inspected before freeze, although no AANCA
score-reference association was inspected. There is no independent pre-outcome
timestamp. Five patients, one junior pathologist and unavailable raw geometry for
the remaining annotators sharply limit generalisation. The global ranking result
does not validate the five-patient-infeasible `balanced_relaxed` deployment queue.
It does not prove pathologist error, biological truth, clinical utility, downstream
benefit or safe automatic correction. The scientific stage remains
`EXTERNAL_VALIDATION_COMPLETE`; every flagged nucleus is recommended for expert
review only.

## Public presentation refresh — 2026-08-23

The generated public presentation now reads the frozen
`nucls_independent_pathologist_validation_v1` authority and displays its limited
`JP.1` leave-one-out disagreement-enrichment result beside the retained adverse
NuCLS aggregate/downstream evidence. The presentation evidence schema is `4`; the
package manifest schema is `5` with policy
`aanca_presentation_current_evidence_readback_v5`. The canonical package contains 13
files and has manifest root SHA-256
`3810912a69a96c759e2426b9556a7f86f04351eb392411b31d4b0add4b4b7597`.

Release validation:

- `uv run pytest`: `1165 passed, 1 skipped` in 590.52 seconds; the skip remains the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 220 maintained Python files formatted;
- `python -I scripts/present_demo.py --verify-only`: valid, 13 files, exact manifest
  root above;
- `uv run python scripts/verify_nucls_independent_pathologist_validation.py`: 11
  artifacts verified, metrics recalculated without retraining, source annotations
  unmodified, primary gate PASS;
- `git diff --check`: passed;
- local browser QA at 1440 x 900 and 393 x 727: no horizontal overflow, no console
  warnings or errors, and the checksum-bound `JP.1` figures and limitations were
  visible.

Hostinger deployment remains blocked before any remote write. Both `verify` attempts
completed the local package build but SFTP authentication failed for the configured
site profile because its required password environment variable is unavailable. The
in-app browser reached the Hostinger login page without a session, and a connected
Chrome session was unavailable. No remote file was changed and no backup ID was
created. The next deployment command, after restoring the configured credential, is:

```text
python C:\Users\NATAN\.codex\skills\hostinger-deploy\scripts\run_hostinger_deploy.py deploy mediumaquamarine-wombat-125861.hostingersite.com
```

This presentation refresh does not change source annotations, scientific evidence
authorities or the completion stage. Scientific status remains
`EXTERNAL_VALIDATION_COMPLETE`; presentation status remains `DEMO_COMPLETE`.

## CI type-check correction and README simplification — 2026-08-24

GitHub Actions run `32653195668` for commit `d1b65ce37caab3bc6ff139b03e9638ed66c34065`
failed at the Windows `uv run mypy src` step with five static type errors in the new
NuCLS independent-pathologist module. Runtime tests, scientific metrics and source
annotations were not the cause. The correction makes the existing integer conversion
and string-keyed DataFrame mapping types explicit and gives the optional class-level
matched-random mean its own nullable variable. It does not change the calculation or
any saved evidence authority.

The repository README now exposes the project website through one direct
`https://aancastudy.org` link only. The duplicate website badge, linked hero image,
presentation description and local presentation walkthrough were removed. The
package and its verification command remain documented under reproducibility.

Local reproduction of the maintained CI gates after the correction:

- `uv run mypy src`: no issues in 104 source files;
- `uv run pytest`: `1165 passed, 1 skipped` in 590.22 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 220 maintained Python files formatted;
- `python -I scripts/present_demo.py --verify-only`: valid 13-file package, manifest
  root `3810912a69a96c759e2426b9556a7f86f04351eb392411b31d4b0add4b4b7597`;
- independent JP1 evidence readback: 11 artifacts verified, primary gate PASS,
  source annotations unmodified;
- `uv run histo-audit experiment smoke --runs-root artifacts/ci-smoke-runs`:
  completed successfully as
  `20260824T192423.107036Z_synthetic_smoke_2b76b8eaa9`;
- `git diff --check`: passed.

Scientific and presentation completion stages remain `EXTERNAL_VALIDATION_COMPLETE`
and `DEMO_COMPLETE`.

## Public independent-pathologist replication pre-outcome freeze — 2026-08-25

An additive RIVA and MIDOG++ replication was defined without changing the frozen
AANCA candidate or any earlier evidence. The controlling config SHA-256 is
`1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321`.
No AANCA score/reference association has been inspected at this checkpoint.

Outcome-blind feasibility and source authentication completed:

- RIVA v1.0 archive MD5 matched `89329be851bac81c7b13bde413ae6a6f`;
  the release contains 959 images, 26,158 annotations and 111 derived smear groups;
- the official RIVA cluster authority is pinned to upstream commit
  `711dfc2c1180d409346f5f94d0cd0ceb485c4ca6` and contains 17,716 raw rows in
  7,507 clusters from 386 shared fields;
- MIDOG++ JSON MD5 matched `686c91bcabcc079a000a5be13cc2f542`, and the official
  availability CSV MD5 matched `65a95814f30a9ebf8312906d2cd7f0ea`;
- the frozen label-independent MIDOG++ rule selected 70 cases, ten from each of seven
  tumor types, containing 3,612 candidates and requiring about 9.0 GiB of TIFF data.

The process-isolated `prepare` commands completed and emitted only observed label,
geometry, image and group fields:

- RIVA rotations: 8,171 / 7,440 / 5,766 / 4,781 rows across 54 / 53 / 53 / 53
  groups;
- MIDOG++ rotations: 3,612 rows each across 70 case groups;
- every snapshot has `hidden_reference_used: false` and passes the forbidden-field
  control.

Pre-outcome implementation checks completed:

- focused public-replication tests: `6 passed`;
- full `pytest`: `1171 passed, 1 skipped` in 582.08 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- full Ruff lint and format check: passed, 223 maintained Python files formatted;
- full mypy: no issues in 105 source files;
- RIVA and MIDOG++ input-only functional CLI stages: passed;
- `git diff --check`: passed.

The next command after publishing this freeze is:

```text
uv run python scripts/run_public_pathologist_replication.py download --dataset midogpp
```

The current scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`; this checkpoint
does not add an efficacy result or change the natural-data action `retain_uncorrected`.

### Pre-outcome MIDOG++ downloader runtime amendment — 2026-08-25

After public freeze commit `3a805ee46f96309642de9aa3d3af25cd9c881aed`, the first
MIDOG++ download command stopped before downloading any TIFF because the unstable
Figshare collection pagination repeated the same `472.tiff` authority. No AANCA score
or AANCA/reference association existed. Direct readback authenticated the one genuine
file authority and showed no content conflict.

The additive runtime amendment
[`PUBLIC_INDEPENDENT_PATHOLOGIST_REPLICATION_RUNTIME_AMENDMENT.md`](PUBLIC_INDEPENDENT_PATHOLOGIST_REPLICATION_RUNTIME_AMENDMENT.md)
deduplicates article IDs, permits only byte-identical repeated file authorities and
fails closed on size, MD5 or URL conflict. It changes no scientific input, selection,
label, group, model, endpoint or gate; the frozen config SHA-256 remains unchanged.
Post-amendment validation passed: `1172 passed, 1 skipped` in 582.96 seconds, full
Ruff lint/format passed and mypy reported no issues in 105 source files.

### Pre-outcome MIDOG++ downloader runtime amendment v2 — 2026-08-26

The next outcome-blind download attempt also stopped before downloading any TIFF:
six separate Figshare pages omitted `372.tiff`, `472.tiff` and `476.tiff` under
unstable collection ordering. The official OpenAPI authority allows 1,000 records per
page, and one such request returned all 506 unique collection article IDs.

Runtime amendment v2 removes pagination, fails on truncation/duplicate IDs, lowers
article-detail concurrency from 20 to 8 and adds bounded retry handling for rate-limit
and transient HTTP responses. The frozen config, selection and all scientific rules
remain unchanged; no AANCA score or reference association had been computed.
Post-v2 validation passed: `1173 passed, 1 skipped` in 610.91 seconds, full Ruff
lint/format passed and mypy reported no issues in 105 source files.

### Pre-outcome MIDOG++ crop runtime amendment v3 — 2026-08-26

All four RIVA input-only rotations completed their score seals with converged models
and no reference access. The first MIDOG++ rotation then stopped before embeddings or
scores because the inherited NuCLS helper reflect-padded an entire large TIFF for one
small crop and could not allocate an extra 97.1 MiB array. No MIDOG++ score artifact
was created, and no RIVA or MIDOG++ reference had been opened.

Runtime amendment v3 replaces only that allocation strategy with an output-sized
reflect-index crop. A byte-equality test covers centres, edges and corners. Crop geometry,
pixels and every scientific rule are unchanged; the frozen config SHA-256 remains
unchanged.
Post-v3 validation passed: `1174 passed, 1 skipped` in 624.10 seconds, full Ruff
lint/format passed and mypy reported no issues in 105 source files.

### Pre-outcome MIDOG++ edge-geometry runtime amendment v4 — 2026-08-26

The next MIDOG++ attempt stopped before crop extraction because one released bbox in
`309.tiff` intersects the image while its centre is 5 px above the top edge. The
outcome-blind audit found no other outside centre among 3,612 candidates. No MIDOG++
embedding, score or reference artifact existed, and the RIVA reference remained
closed.

Runtime amendment v4 preserves the exact centre and permits periodic reflect pixels
only when every frozen crop window intersects the image; a completely outside window
still fails closed. It changes no data selection or scientific rule, and the frozen
config SHA-256 remains unchanged.
Post-v4 validation passed: `1175 passed, 1 skipped` in 603.52 seconds, full Ruff
lint/format passed and mypy reported no issues in 105 source files.

### Public pre-reference score freeze — 2026-08-26

All six process-isolated scoring rotations completed before either hidden reference
was opened. Every five-fold model converged, all splitters used
`StratifiedGroupKFold`, all neighbour lists excluded the query group and every score
seal reports `hidden_reference_loaded: false`, `full_annotation_source_opened: false`
and `reference_authority_opened: false`.

Score populations and sealed CSV SHA-256 values:

- RIVA `annotator_1`: 8,171,
  `d36445df00c4e96169feb14d25eb5e471d9bb57edb5076b4e4ce1ed46a8cf49b`;
- RIVA `annotator_2`: 7,440,
  `5cf701edadaa9b57a54140ec681bf8f26146eeb2cf7eabbb7382a6506d28a3a5`;
- RIVA `annotator_3`: 5,766,
  `cceaaa1e2e358e5bed7c9af98dad3ab7af8fb78d211b32652f3d5f4515374f5b`;
- RIVA `annotator_4`: 4,781,
  `109222af52f4a050d9612d6174dfdfa727ba74cb07926f473ae8bb890487e164`;
- MIDOG++ `expert_1`: 3,612,
  `900581373983a581212fbd052550dad0da58795c7c47657d4a61a6a425ca04bf`;
- MIDOG++ `expert_2`: 3,612,
  `8c9671dc33e650306043c9b41f7afebf047a5093e61500bc9f84d029d9dbb76d`.

The machine-readable authority is
`artifacts/public_independent_pathologist_replication/pre_reference_freeze.json`.
Evaluation has not run and no reference association is known at this checkpoint.

## Public independent-pathologist cross-dataset result — 2026-08-26

Commit `09aede5000c43406759a432b674f6db37db98b26` was confirmed identical on local
`HEAD` and `origin/main` before either hidden reference was opened. Both evaluation
commands authenticated every pre-reference score seal and did not recompute risks.

The frozen primary outcomes are:

- RIVA: 571 reviewed rotation-rows among 11,373 eligible; disagreement precision
  `0.476357` versus `0.426900` across 100 exact matched-random queues; difference
  `+0.049457`, enrichment `1.115852`, whole-group bootstrap 95% CI for the
  difference `[+0.014037, +0.091632]` across 33 source-image groups;
- MIDOG++: 362 reviewed rotation-rows among 7,224 eligible; disagreement precision
  `0.325967` versus `0.249392`; difference `+0.076575`, enrichment `1.307045`,
  whole-group bootstrap 95% CI `[+0.021352, +0.138159]` across 70 image groups;
- every point difference was non-negative in all four RIVA rotations and both
  MIDOG++ rotations;
- the RIVA and MIDOG++ dataset gates both passed, so the frozen cross-dataset
  replication rule passed;
- `verify --dataset riva` and `verify --dataset midogpp` both returned
  `verification_passed: true`.

The completion stage remains `EXTERNAL_VALIDATION_COMPLETE`. The positive result
supports enrichment for natural independent-expert disagreement in these two public
releases. It does not prove that a pathologist was wrong, establish clinical or
downstream utility, validate a prospective workflow or permit automatic annotation
changes. Source annotations were not modified and the binding natural-data action
remains `retain_uncorrected`.

The human-readable authority is
[`reports/public_independent_pathologist_replication_results.md`](reports/public_independent_pathologist_replication_results.md),
and the machine authorities are under
`artifacts/public_independent_pathologist_replication`. The presentation builder now
fail-closes on both dataset gates, the six-rotation boundary and the pre-reference
information barrier. Its updated 13-file package passed standalone checksum
verification with manifest root
`86bbcb32c515688c05e4f31567a73e0f234c157556daf26aa561223a899a2a08`.

Final local validation for the result and publication update:

- full `pytest`: `1175 passed, 1 skipped` in 593.53 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- `ruff check .`: passed;
- `ruff format --check .`: all 223 maintained Python files formatted;
- `mypy src`: no issues in 105 source files;
- RIVA and MIDOG++ frozen evidence recalculation: both primary gates and both
  `verification_passed` flags true;
- combined report regeneration: `cross_dataset_replication_supported: true`;
- standalone presentation verification: valid 13-file package with manifest root
  `86bbcb32c515688c05e4f31567a73e0f234c157556daf26aa561223a899a2a08`;
- Playwright local-browser readback: the RIVA x MIDOG++ card exposed both frozen
  comparisons and the claim boundary; console reported zero errors and warnings;
- `git diff --check`: passed.

Hostinger publication was attempted through the configured
`mediumaquamarine-wombat-125861.hostingersite.com` manifest. The prescribed local
build/verification passed with the current manifest root, but SFTP authentication
for the configured profile failed before backup or upload. No backup ID was created
and no remote file changed. The checked-in and GitHub-published static package is the
current deploy authority until that credential is restored.

## Final professor-readiness audit — 2026-08-26

The repository, retained evidence, documentation, professor brief and static website
were re-audited as one release surface. The scientific stage remains
`EXTERNAL_VALIDATION_COMPLETE`, the presentation stage remains `DEMO_COMPLETE`,
`CONFIRMATORY_COMPLETE` remains unreached and natural-data action remains
`retain_uncorrected`.

Material changes:

- added an evidence-first four-card website summary generated from sealed evidence,
  while retaining the complete long-form article and every adverse result;
- added `scripts/verify_professor_release.py` and a Linux CI step binding key public
  numbers, source identities, dates, package scope and claim-boundary wording;
- restricted public Figshare access to HTTPS canonical API/downloader hosts without
  credentials or non-443 ports, with positive and negative URL tests;
- reduced six hero sprites from 8.62 MiB to 1.24 MiB combined; the sealed 13-file
  package fell from 12.72 MiB to 5.34 MiB and now has manifest root
  `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`;
- updated stale release dates, the historical five-file package description and the
  evidence scope in citation, reproducibility and presentation-scope documentation;
- marked the 21 August internal assessment as superseded and published
  [`FINAL_READINESS_REPORT.md`](FINAL_READINESS_REPORT.md).

Executed final-version evidence checks:

- NuCLS aggregate/downstream readback: verified, complete claim `not_supported`;
- NuCLS `JP.1` independent-pathologist readback: verified, primary gate passed;
- RIVA and MIDOG++ reference-association readbacks: both
  `verification_passed: true`, both primary gates passed;
- MoNuSAC readback: verified, complete claim not supported;
- selected-candidate refit: 220/220 fits converged and final external test unused;
- PUMA readback: 44/44 convergence records, all seven frozen gates, exact controls,
  group exclusions, manifest and unchanged-source-label checks passed;
- NuCLS paired-QC feasibility: paired pre/post class labels unavailable, so no
  natural-error result was invented.

Executed final engineering gates:

- `pytest`: `1182 passed, 1 skipped` in 604.46 seconds;
- `ruff check .`: passed;
- `ruff format --check .`: all 224 maintained Python files formatted;
- `mypy src`: no issues in 105 source files;
- `uv lock --check`, `uv pip check`: passed;
- `pip-audit --local`: no known vulnerabilities;
- synthetic data generation and isolated `histo-audit experiment smoke`: completed;
- standalone 13-file presentation verification: valid;
- professor-facing release consistency: valid, with 19 unique upstream authorities
  authenticated;
- Playwright at 1280 x 720 and 390 x 844: zero console errors/warnings, no horizontal
  overflow or missing sprites, and no detected structural accessibility defect;
- temporary build and smoke-output directories created for this audit were removed.

Final Hostinger `verify` and `deploy` attempts both accepted the local package and
then failed SFTP authentication before backup or upload. No backup ID was created and
no remote file changed. Browser readback of `https://aancastudy.org/` confirmed that
the live domain still serves the preceding version without the RIVA/MIDOG++ block or
the new evidence snapshot. The Hostinger panel also required a fresh login and no
connected authenticated Chrome session was available. Deployment therefore remains
blocked only on restoring the configured hosting credential.

The supported final headline is cross-dataset enrichment of independent-expert
disagreement for review, together with controlled PUMA transfer. This does not prove
which pathologist is correct, adjudicated natural-error detection, automatic
correction safety, clinical utility or prospective workflow superiority.

## AANCA v2 codebase boundary — 2026-08-26

A read-only size audit of the current tracked repository counted 232 Python files and
172,959 physical Python lines: 121,908 under `src`, 43,857 in tests, 6,199 in scripts
and 995 in retained artifact helpers. The much larger 1,703,271-line tracked-text
total is not a code-size measure because approximately 89% consists of generated
artifacts and reports, principally JSON and CSV evidence.

Decision D047 records that this size is historical technical debt, not a reason to
rewrite the frozen v1 evidence paths before presentation. V1 remains the auditable
reference and receives only justified correctness, security, compatibility or
evidence-readback fixes. V2 will be isolated behind a read-only v1 compatibility
adapter, with an approximately 15,000--30,000-line production-Python planning target,
standard workflow tooling and external content-addressed artifact storage. The target
is non-scientific and cannot weaken validation, safety or claim-boundary controls.

This update changes planning documentation only. No source code, annotation, model,
metric, evidence artifact, completion stage or natural-data action changed. The v2
programme remains `INITIALISED`; v1 remains `EXTERNAL_VALIDATION_COMPLETE`,
`CONFIRMATORY_COMPLETE` remains unreached and natural-data action remains
`retain_uncorrected`.

Post-update validation passed: `git diff --check`, the standalone professor-release
verifier, `uv run ruff check .` and `uv run ruff format --check .`. The complete test
suite passed `1182` tests with the one documented Windows/POSIX open-file skip in
580.00 seconds. No scientific or functional command was rerun because the update
changes planning and decision documentation only; the relevant release-consistency
functional verifier accepted all 19 upstream authorities and the unchanged thirteen-
file presentation manifest.

## AANCA V2 pre-execution authority package and separation — 2026-08-28

Created the additive V2 planning package before any V2 research execution, then moved
it out of this working tree into the sibling `AANCA-V2` project with independent Git
history. Its access-controlled preparation remote is
[`Jaqwilk/AANCA-V2`](https://github.com/Jaqwilk/AANCA-V2). It contains a charter,
binding V2 specification, glossary, frozen V1 evidence boundary, targeted literature
rationale, gated plan, detailed preregistration draft, traceability matrix, change
control, risk register, execution checklist, decision/status logs, seven scientific
protocols, five engineering plans, machine-readable JSON authorities,
dataset/candidate/reference-access/claim/freeze/amendment/execution templates, an
opened-dataset registry and a standalone consistency validator.

The package narrows the immediately executable question to review-signal enrichment
on a genuinely new eligible public multi-rater source. It requires raw individual
votes, exclusion of the input rater, at least two non-input qualified votes, raw
input geometry, patient/WSI/case-safe grouping, one frozen 5% endpoint, exact
matched-random controls, group bootstrap and a public score-only seal before
reference attachment. All opened V1 sources remain development-only. A failed
dataset search produces `NO_GO`, not a weakened “confirmation.”

After separation, the package validator passed with 48 required paths and 48 local
links. It verified valid JSON/CSV authorities, the complete opened-source registry,
unique decision and traceability IDs, approved stage vocabulary, the immutable
read-only V1 authority at commit
`79d806582c0b618a8c9e3ec1d70313c40be1278e`, `DRAFT_NOT_FROZEN` preregistration and
no enabled V2 result or claim. The standalone environment also passed `uv run
pytest` (1 test), `uv run ruff check .` and `uv run ruff format --check .` (33
files).

Post-package gates passed: `git diff --check`, `uv run ruff check .`,
`uv run ruff format --check .` (225 files), and the standalone professor-release
verifier (19 authenticated upstream authorities and the unchanged 13-file
presentation manifest). The complete suite collected 1183 tests and finished with
1182 passed, the one documented Windows/POSIX open-file skip, and zero failures in
713.73 seconds.

This work created planning and validation files only. It did not approve a dataset,
train a model, generate a score, open a new reference, change a source annotation,
rerun a scientific analysis or alter any V1 artifact/result. V2 remains
`INITIALISED`, V1 remains `EXTERNAL_VALIDATION_COMPLETE`, and the natural-data action
remains `retain_uncorrected`.

The independent V2 repository was first published at commit
`548b9c22c43c1e0609afe14a39714f9f79a17ef4`; its publication record was then pushed
at `8de62fec9ea8956f98e26503cc2e9434e35703a9`. Local V2 `HEAD` and
`origin/main` matched after both pushes. Decision D049 records the physical and
provenance boundary. The move did not alter the pinned V1 commit or any scientific
evidence.

After the physical move and V1 link updates, the current V1 tree passed `git diff
--check`, `uv run ruff check .`, `uv run ruff format --check .` (224 files), and
`uv run python -I scripts/verify_professor_release.py` (19 authenticated upstream
authorities, unchanged 13-file presentation package, scientific status
`EXTERNAL_VALIDATION_COMPLETE`, presentation status `DEMO_COMPLETE`). The complete
current V1 suite collected 1183 tests and finished with 1182 passed, the same one
documented Windows/POSIX open-file skip, and zero failures in 750.17 seconds.

## Complete repository README restored — 2026-09-01

Restored the full root README structure from commit
`09aede5000c43406759a432b674f6db37db98b26`, the direct parent of the change that
collapsed it to one website link. The restored narrative was brought forward only
with already sealed RIVA, MIDOG++, NuCLS and PUMA evidence, the current
score-before-reference claim boundary, and the separate access-controlled AANCA V2
repository. All local README links resolve.

Decision D050 supersedes only the D046 single-link README requirement. The
professor-release verifier now checks the full README semantically against exact
sealed evidence values, responsible terminology, immutable-source policy,
`retain_uncorrected`, completion limits, reproducibility and validation sections.
It no longer requires byte equality with a one-line file.

Validation after the restoration passed:

- `git diff --check`;
- `uv run ruff check .`;
- `uv run ruff format --check .` — 224 files already formatted;
- `uv run mypy src` — no issues in 105 source files;
- `python -I scripts/present_demo.py --verify-only` — valid unchanged 13-file
  package, manifest root
  `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`;
- `uv run python -I scripts/verify_professor_release.py` — valid, 19 authenticated
  upstream authorities, scientific status `EXTERNAL_VALIDATION_COMPLETE` and
  presentation status `DEMO_COMPLETE`;
- `uv run pytest` — 1182 passed, the same one documented Windows/POSIX open-file
  skip, and zero failures in 771.50 seconds.

This documentation and fail-closed verification repair changes no source annotation,
dataset, model, split, score, metric, evidence artifact, scientific claim boundary,
completion stage or natural-data action.

## Consolidated AANCA brand and design system — 2026-09-01

Created [`AANCA_BRAND_SYSTEM.md`](AANCA_BRAND_SYSTEM.md) as the design authority for
AANCA websites, presentations, reports, posters, figures and future interfaces. It
consolidates the identity already present in the public article: the non-diagnostic
Second-Look metaphor, canonical four-tile 8:3 mark, dark editorial palette, accessible
violet roles, Inter/JetBrains Mono typography, layout rails, spacing, components,
scientific data-visualisation rules, imagery, motion, responsive behaviour, print,
reusable copy and a release checklist. It also includes canonical SVG and reusable CSS
implementations.

Decision D051 makes the four-tile mark and the separation between visual authority and
scientific authority explicit. `SPEC.md`, frozen protocols and accepted evidence still
override any brand treatment that could affect a method, result, completion stage or
claim.

Validation after the documentation change:

- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: 224 files already formatted;
- `uv run pytest`: 1182 passed, 1 skipped in 612.69 seconds; the skip is the same
  documented Windows/POSIX open-file rename difference;
- `python -I scripts/present_demo.py --verify-only`: valid unchanged 13-file package,
  manifest root
  `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`;
- brand reference/content checks and `git diff --check`: passed before this status
  append and are repeated at handoff.

This design-documentation work changes no source annotation, dataset, model, split,
score, metric, evidence artifact, scientific claim boundary, completion stage or
natural-data action.

## Repository, scientific-logic and deployed-site review — 2026-10-02

Recorded the requested review in
[`reports/project_audit_2026-10-02.md`](reports/project_audit_2026-10-02.md), with
machine-readable observations in
[`reports/project_audit_2026-10-02_evidence.json`](reports/project_audit_2026-10-02_evidence.json).
The reviewed source commit is `fa3311bf750c32b7f2bc0ac69097018620e77d71`.
Pre-existing brand-system work in this working tree was preserved.

The review found a presentation/statistical-method mismatch: primary H4 stores the
central quantile range of 100 guided-minus-random restoration differences on one
fixed final reference set, whereas the article calls its displayed intervals paired
whole-group bootstrap intervals and labels H4 as a 95% CI. The stored numbers
recalculate correctly; their statistical interpretation needs correction. No
favourable H4 result or replacement interval was generated.

Small synthetic probes also reproduced acceptance of out-of-range probabilities,
negative adoption/utility thresholds, fractional labels truncated to class integers,
unreported non-convergence in generic OOF, and invalid neighbour fold provenance.
These are interface-validation findings, not evidence that the released experiments
used those invalid inputs. The report separates them from actual scientific outcomes.

On 2 October the official domain and its www/HTTP variants returned HTTP 403 from
this environment. The technical Hostinger URL returned HTTP 200 but served the
22 August article and evidence schema 3, omitting the later NuCLS JP.1, RIVA and
MIDOG++ evidence present in the local 26 August article and schema 5. Eight served
PNG responses differed from the technical site's own manifest hashes; the cause
was not established. No hosting write or deployment was performed.

Executed validation:

- `uv run pytest --durations=20`: 1182 passed, 1 skipped, 1237.58 seconds; the skip
  is the documented Windows/POSIX open-file rename difference.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed, 224 files.
- `uv run mypy src`: passed, 105 source files.
- `uv run histo-audit doctor`: passed; CUDA and RTX 4070 available.
- `uv run histo-audit experiment smoke --runs-root artifacts/qa/review-20261002-smoke`:
  passed; run `20261002T141629.514774Z_synthetic_smoke_6d88919266`.
- `uv run python -I scripts/verify_professor_release.py`: passed, 19 upstream
  authorities and the unchanged 13-file local presentation package.
- `uv run python -I scripts/present_demo.py --verify-only`: passed; local manifest
  root `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`.
- Primary saved-evidence verification, including the recovered primary run and
  `primary_0027_8531672acd3c` restoration: passed. The exact command is in the report.
- NuCLS external, MoNuSAC, PUMA, NuCLS QC feasibility and NuCLS independent-pathologist
  verifiers: passed within their documented scopes. The QC endpoint remains unavailable.
- Public pathologist replication `verify --dataset riva` and
  `verify --dataset midogpp`: passed.
- Local Playwright checks: desktop/mobile layout, reduced-motion and no-JavaScript
  article readability, H6 unavailable filtering and empty search results passed.
  The technical Hostinger deployment was also inspected in the browser.

The documented `verify_aanca_selected_candidate.py` command was started and then
explicitly interrupted after confirming that it performs nested model re-execution
using raw MoNuSAC inputs and ignored local selection lineage. It is not recorded as
passed. Its canonical convergence output was not replaced. Reproduction instructions
need to distinguish this command from saved-array verification and disclose the raw
input requirements of the PUMA and NuCLS QC checks.

Post-documentation verification initially failed because the professor-release
verifier requires the exact text `Updated: 1 September 2026` in STATUS.md. The
existing release-status header is therefore retained, with this audit dated
separately above and in its own section. F11 records the brittle date dependency;
the verifier itself was not relaxed or changed.
After this repair both professor-release and presentation verification passed again:
19 upstream authorities, 13 presentation files and the unchanged manifest root.

Decision D052 records remediation priorities. No source code, frozen study authority,
source annotation or scientific result was changed. Scientific stage remains
`EXTERNAL_VALIDATION_COMPLETE`, presentation stage remains `DEMO_COMPLETE`, and the
natural-data action remains `retain_uncorrected`.

Next implementation action: correct the H4 interval description and carry its method
metadata into the presentation, with a targeted semantic regression test. Next command
after that change: `uv run pytest tests/test_mvp_demo.py`, followed by the required
full validation and presentation/release verification before publishing.

## Requested aancastudy.org deployment — 2026-10-02

Added a local Hostinger manifest for `aancastudy.org`, using the existing AANCA
credential profile and the unchanged `artifacts/mvp_demo` package. The proposed
mapping is `domains/aancastudy.org/public_html` in merge mode; remote path ownership
and domain routing remain unverified until authentication succeeds.

Executed `python C:\Users\NATAN\.codex\skills\hostinger-deploy\scripts\run_hostinger_deploy.py verify aancastudy.org`.
The configured build step (`scripts/present_demo.py --verify-only`) passed: 13 files,
manifest root `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`.
SSH/SFTP then failed with `Authentication failed`. No upload, remote backup or
post-deployment HTTP verification occurred. No source code or website file changed;
the full pytest/lint/format gates were not rerun for this blocked deployment attempt.

Next action: restore the `hostinger-bisque-jay` credential locally in the deploy
configuration, rerun the same verify command, verify the actual remote domain root,
and complete required release gates before deployment. Scientific stages and
natural-data action remain unchanged.

## Official-domain deployment completed — 2026-10-02

The owner supplied an updated SSH credential and explicitly authorized upload.
Updated only the local deployment credential store; no secret is recorded here.
The Hostinger skill wrapper `verify aancastudy.org` passed package verification and
SSH/SFTP access. SFTP readback confirmed the actual target directory
`/home/u786975226/domains/aancastudy.org/public_html`, initially containing only
`default.php`.

Pre-deployment gates passed:

- `uv run pytest`: 1182 passed, 1 documented Windows/POSIX skip in 721.24 seconds;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: 224 files already formatted;
- `uv run python -I scripts/verify_professor_release.py`: valid, 19 upstream authorities;
- configured build step `scripts/present_demo.py --verify-only`: valid 13-file package,
  root `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`.

Executed `python C:\Users\NATAN\.codex\skills\hostinger-deploy\scripts\run_hostinger_deploy.py deploy aancastudy.org --skip-build`,
reusing that successful unchanged package verification. Deployment completed with
backup ID `20261002-170743` under
`.codex-deploy-backups/aancastudy.org/20261002-170743`. Merge upload preserved the
hosting placeholder file. The wrapper's page and followed-asset HTTP checks passed.
Independent cache-busted HTTP checks confirmed both apex and www homepages match
local `index.html`; text/JSON/JavaScript package files also match. SFTP SHA-256
readback confirmed all 13 uploaded files match local originals.

HTTP byte equality does not hold for CDN-delivered PNGs. Seven PNGs retained identical
RGBA pixels; the workflow graphic is served at 1600 x 556 instead of its original
2128 x 739. The origin graphic matches exactly. This delivery transformation is
recorded rather than claimed as byte-identical public delivery. An initial optional
pixel-check script could not import requests in the project environment; it was
rerun using standard-library urllib without installing dependencies.

The user was advised to rotate the password exposed in chat after deployment.
No website source, scientific evidence, annotation, metric, claim or completion
stage changed. Next action: rotate the exposed SSH password and update the local
credential store for future deployments.

## V1 remediation and repeat audit — 2–3 October 2026

Implemented the bounded fixes in
[`reports/v1_remediation_2026-10-02.md`](reports/v1_remediation_2026-10-02.md)
under D054. Shared validation now rejects out-of-range/nonfinite probabilities,
fractional or overflowing labels/classes, and negative global adoption thresholds.
Generic OOF records real optimiser diagnostics and rejects reported failed fits;
maintained downstream paths also reject them. Adam fallback no longer invents
convergence. Neighbour provenance excludes the complete held-out group set before
index fitting. Additional repeat-audit probes found and fixed invalid entropy
epsilon and malformed soft-target acceptance.

Corrected H4's presentation description to a central 95% range over random-review
repetitions on the same fixed final reference set. Its point, interval and adverse
finding remain unchanged. Added explicit interval metadata and resealed-tamper
tests. The presentation is manifest schema 7 / evidence schema 6, with root
`8719c185feb4ace95b242eedb60de063e3ccc4065df5590cfb331374814e93ca`.
Recursive before/after comparison confirmed all 878 existing evidence values and
source identities are unchanged. The previous presentation is preserved in
`artifacts/qa/v1fix-previous-presentation-20261002`.

README/reproducibility/evidence documentation now distinguishes actual uncalibrated
ranking, optional calibration, original-audit defaults, the selected development
candidate, saved-array verification, raw-data checks, output writes and full model
re-execution. The professor-release verifier accepts one real living status date
while still checking frozen release dates and upstream identities. Duplicate D052
identifiers were reconciled: the hosting decision is D053.

Final completed gates:

- `uv run pytest -q`: **1233 passed, 1 documented Windows/POSIX skip in 631.11s**;
  log `artifacts/qa/v1fix-pytest-accepted.log`;
- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: 228 files already formatted;
- `uv run mypy`: no issues in 111 source files;
- `uv lock --check`, `uv pip check`, `git diff --check`: passed;
- `uv run --with pip-audit pip-audit --local`: no known vulnerabilities found;
- `uv run histo-audit experiment smoke --runs-root artifacts/qa/v1fix-smoke-accepted-20261002`:
  completed successfully, run `20261002T215315.795421Z_synthetic_smoke_b85a24a0de`;
- `scripts/present_demo.py --verify-only`: valid thirteen-file package;
- `scripts/verify_professor_release.py`: valid with nineteen upstream authorities;
- independent primary recalculation: all 33 numeric comparisons, three explicitly
  unavailable comparisons and H4 passed;
- NuCLS aggregate, MoNuSAC, NuCLS JP.1, RIVA, MIDOG++, PUMA and NuCLS paired-QC
  verification passed. PUMA/QC writes were routed into `artifacts/qa`;
- all nine frozen configuration files match their SHA-256 sidecars;
- AST parsing passed for 112 source, 101 test and 21 script files;
- Chromium desktop/mobile checks passed at 1280 x 720 and 390 x 844. Menu and
  evidence-table filtering work; no page overflow or console errors/warnings were
  observed. The live www page displays the corrected H4 caption.

The preliminary full-suite attempt was interrupted and supplies no pass claim.
The first completed run had `1 failed, 1230 passed, 1 skipped` in 619.57s: the new
absolute `1e-7` probability row-sum check rejected legitimate float32 MLP round-off.
Adjusted the absolute tolerance to `5e-7` with zero relative tolerance; real PyTorch
softmax regression cases cover both float32 and unchanged float64 promotion, while
rows differing by `1e-6` remain rejected. The previously failing integration fixture
and 53 related cases passed before the complete accepted rerun. No probability was
clipped or renormalised and no frozen fitting controls were changed. Initial new
fixture/type-check errors and a missing `restorations/` input-path component were
also repaired before final gates.

Hosting preparation and deployment used the Hostinger skill. Its package/build
verification and SSH/SFTP check passed. Reused the unchanged successful output with
`deploy aancastudy.org --skip-build`; merge upload and HTTP health checks passed.
Recoverable backup: `20261002-235931`. The separate tracked server configuration
`deploy/hostinger/aancastudy.htaccess` requests `no-cache, no-transform` and is
mapped to the domain's `.htaccess`, outside the thirteen-file presentation.

Post-deployment SFTP readback verifies all thirteen origin files and `.htaccess`.
The strict HTTP verifier passes all thirteen files on `https://www.aancastudy.org/`.
Both domains serve the new manifest and scientific summary. The initial apex check
failed solely for the cached `assets/hero/nuclei/nucleus-compact.png` response (CDN
`HIT`; 291,149 rather than 290,592 bytes). Its release-query response carries
`no-cache, no-transform` and matches the original exactly. An ordinary request
`no-cache` header did not invalidate the old CDN entry. Do not exempt that URL
from the strict check or overwrite the source hash with the transformed hash.

The owner was asked to flush this domain's CDN cache in hPanel. Working
SSH/SFTP credentials cannot control it; the available API account returns zero
hosted websites and 404 for this domain's read endpoint. No unauthorised API cache
mutation was attempted.

Closing check on 3 October: the same ordinary apex HTTP verifier now passes all
thirteen files and equality to the local release, with no query exemption. The
stale cache entry refreshed during the closing checks; the actor/mechanism is not
attributed. Both apex and www are now fully verified. No mandatory V1 remediation
or publication gate remains pending.

Next verification command after any future publication change:
`uv run python -I scripts/verify_deployed_presentation.py --url https://aancastudy.org/`.
Full source-model retraining and new study claims are not part of this remediation.
Scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`, presentation stage remains
`DEMO_COMPLETE`, and natural-data action remains `retain_uncorrected`. No source
annotation, frozen scientific authority, final reference split or published result
was modified.

## GitHub V1 publication preflight — 3 October 2026

The owner authorised publishing the audited workspace to the existing public
repository `https://github.com/Jaqwilk/AANCA`. Before publication, fetched
`origin/main` and verified that both it and local `main` were still
`fa3311bf750c32b7f2bc0ac69097018620e77d71`, with zero ahead/behind commits and no
pre-existing staged changes. The publication scope includes the D054 fixes,
regression tests, verifiers, corrected presentation, audit reports and existing
brand document. The README now links the current remediation report and distinguishes
it from the historical initial findings. Decision D055 records the publication rules.

Fresh pre-publication checks passed:

- `uv run ruff check .`;
- `uv run ruff format --check .`: 228 files already formatted;
- `uv run mypy`: no issues in 111 source files;
- `uv run python -I scripts/present_demo.py --verify-only`: valid thirteen-file package;
- `uv run python -I scripts/verify_professor_release.py`: nineteen upstream authorities;
- `uv run python -I scripts/verify_deployed_presentation.py --url https://aancastudy.org/`:
  all thirteen public files match the current local release;
- `git diff --check`.

The accepted full-suite and functional evidence remains the D054 run: 1,233 passed,
one documented platform skip, and successful deterministic synthetic smoke. No
runtime code has changed since those accepted checks. The publication performs no
source-model retraining or new outcome analysis. Check the new revision's Ubuntu
and Windows CI at
`https://github.com/Jaqwilk/AANCA/actions/workflows/scientific-software.yml`.
Scientific stage remains `EXTERNAL_VALIDATION_COMPLETE`, presentation stage remains
`DEMO_COMPLETE`, and natural-data action remains `retain_uncorrected`.
