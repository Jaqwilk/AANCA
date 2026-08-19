# Decision Log

Decisions are conservative, reversible where possible, and amended rather than silently overwritten.

## 2026-07-17 — Repository and evidence policy

- Use the existing local Git repository; create no remote and do not push.
- Do not modify global Git identity. With identity absent, record stable checkpoints in `STATUS.md` rather than fabricating commits.
- Every report metric must resolve to a saved machine-readable artifact and run ID.
- Every failed or reduced experiment must retain its intended configuration, failure evidence, and exact continuation command.

## 2026-07-17 — Project runtime and CUDA

- **Python version:** 3.12.3 in project-local `.venv`, managed with uv 0.11.15 and locked in `uv.lock`.
- **PyTorch selection:** current official PyTorch 2.12.1 with torchvision 0.27.1 from the CUDA 12.6 index. The installed NVIDIA 551.78 driver reports CUDA 12.4, but NVIDIA's current compatibility guide places drivers 525–579 in the CUDA 12.x minor-compatibility range. CUDA 13.x was rejected because it requires driver 580 or newer. No standalone CUDA Toolkit was installed.
- **Verification:** the earlier conservative PyTorch 2.6.0+cu124 fallback passed an RTX 4070 tensor multiplication/backward test. The final PyTorch 2.12.1+cu126 build also reported CUDA available, identified the RTX 4070/12,878,086,144 bytes VRAM, and passed the same 512×512 CUDA multiplication/backward test with finite gradients. `uv pip check` reported all installed packages compatible.

## Historical environment-backed decisions (initially pending)

- **Dataset source:** official PanNuke source required unless a documented, licensed, verified fallback is necessary.
- **Grouping unit:** source patch minimum; upgrade only if reliable WSI/patient identifiers exist.
- **Crop strategy:** dynamic nucleus-centred context crops with exact target mask/contour; dimensions/padding to be fixed after real-format inspection.
- **Input representations:** context RGB, target-highlighted RGB, and target morphometrics are mandatory.
- **Feature encoders:** engineered baseline and official ImageNet compact encoder mandatory; pathology encoder chosen by a frozen availability rule.
- **Classifier:** multinomial logistic regression primary frozen-feature baseline; small MLP secondary.
- **Calibration:** only from uncorrupted reference-validation groups; calibrated and raw results separate.
- **Duplicate handling:** the complete release audit found no exact or frozen-ResNet
  candidates and 121 pHash-only candidates. Every candidate remains `review_only`;
  no automatic deletion, eligibility change, or split change is permitted. Any later
  expert-adjudicated exclusion rule must be frozen before the primary study without
  consulting final-reference outcomes.
- **Statistical threshold:** primary group bootstrap target 2,000 iterations; subgroup AP requires ≥100 samples and ≥10 injected corruptions.
- **Preregistration deviations:** none; preregistration is not frozen.

## 2026-07-17 — Duplicate-audit implementation policy

- Restrict reported candidates to cross-fold source-patch pairs and retain every source item unchanged.
- Require SHA-256 plus exact array equality for an exact-match finding; use perceptual average hash and official frozen ResNet-18 cosine similarity as separate near-duplicate signals.
- Save full per-patch hash provenance and deterministic candidate rankings. Rankings are review order, not duplicate probabilities.
- Do not mark the required two-signal gate complete when official weights are unavailable or either near-duplicate signal has sampled rather than full patch coverage.
- Keep final exclusion/handling choices pending real-data review and freeze them before the primary study without consulting final labels.

## 2026-07-17 — Pipeline-complete checkpoint

- **Canonical synthetic evidence:** use only `20260717T162925.902444Z_synthetic_smoke_5573505315` (10%) and `20260717T162948.870526Z_synthetic_smoke_zero_corruption_a4d5f87ca0` (0%) for the current code. Older runs remain append-only historical evidence.
- **Generating source identity:** both canonical runs bind `src/**`, `configs/**`, `pyproject.toml`, and `uv.lock` to SHA-256 `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a`.
- **Synthetic split unit:** source patch is the mandatory group. The canonical OOF and restoration evidence has zero group overlap, exactly-once OOF coverage, and a separate uncorrupted final-reference partition.
- **Synthetic final-reference selector:** the canonical smoke selects final groups by a configured group fraction. Its resolved `final_test_fold` is therefore structured `not_applicable`; `-1` is retained only as the internal split sentinel and is not presented as an official fold.
- **Synthetic engineered spaces:** instance-dependent corruption tests use a morphology-only generator vector and a distinct colour-only auditor vector, with typed/hash-bound evidence. The canonical smoke uses symmetric corruption, so feature-space independence is `not_applicable` rather than falsely asserted.
- **Morphometric definition:** solidity means target area divided by the actual convex-hull area. Hole filling is not labelled solidity.
- **Smoke statistics:** 200 group-bootstrap iterations are software-validation only. The intended primary threshold remains at least 2,000 paired group resamples.
- **0% policy:** AP, AUROC, recall, lift, and comparative bootstrap inference are undefined when no injected event exists; save structured `not_applicable` values and show score/false-alert behaviour instead.
- **Evidence policy:** canonical runs retain the complete synthetic dataset and source manifest, corruption manifest, OOF probabilities, per-sample fold-safe neighbour evidence, every guided/random restoration decision, and all final-test probabilities. Reconciliation and sealing cover these artifacts and the report figures derived from them.
- **Figure policy:** PR curves use saved OOF rankings with shared-threshold tie handling; galleries record sample IDs, selection rule, tie break, transforms, and input hashes. Raw score scales in the 0% plot are explicitly method-specific and not compared across methods.
- **ImageNet representation:** official frozen torchvision ResNet-18 `IMAGENET1K_V1`, exact official preprocessing, checksum-bound weights, RGB and target-highlighted RGB. The successful CUDA cache smoke does not constitute PanNuke evidence.
- **Pathology encoder:** select the first predefined candidate only if source, licence, authentication, exact weights, preprocessing, intended use, local hardware fit, and embedding smoke are all verified. Current result is blocked/no selection; do not choose by outcome performance.
- **Real-data grouping:** source patch is the minimum. Upgrade to a reliable WSI/patient unit only if actual release metadata proves those identifiers; otherwise state that patient- and WSI-level independence cannot be guaranteed.
- **Real-data duplicate handling:** require full-release exact, pHash, and
  methodologically distinct frozen-ResNet signals before the pilot gate; similarity
  candidates remain review-only and are never auto-deleted.
- **Stage boundary:** primary and confirmatory execution remain pending. No claim may exceed `PIPELINE_COMPLETE` until each later gate has genuine artifacts.

## 2026-07-17 — Official PanNuke acquisition and anomaly-safe QC

- **Source, licence scope, and project restriction:** the acquisition source for the
  three folds is the [official University of Warwick PanNuke
  page](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/). Local release documents
  explicitly apply CC BY-NC-SA 4.0 to the `masks/` directory and its contents; the
  same local evidence does not establish that scope for every image/type file. The
  project separately restricts all use of the release to research/non-commercial
  work and requires both PanNuke citations: `gamper2019pannuke` and
  `gamper2020pannukeextension` in `references/references.bib`. The official archives
  and extracted arrays must remain unchanged under `data/raw/pannuke` and are
  excluded from Git.
- **Evidence status:** the archive byte sizes and SHA-256 values below were supplied
  by the user as local acquisition evidence. ZIP CRC and extraction path-safety
  checks were also reported as executed by the user. They are recorded for
  reconciliation. On 2026-07-18 the M5 schema-v2 acquisition manifest independently
  recomputed and exactly matched them; all ZIP CRC/path-safety and extracted-file
  checks passed without modifying raw data.
- **Reported archive identities:** `fold_1.zip` is 700,275,281 bytes with SHA-256
  `6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b`;
  `fold_2.zip` is 658,842,552 bytes with SHA-256
  `5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07`;
  `fold_3.zip` is 717,969,882 bytes with SHA-256
  `c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d`.
  These are reported local checksums, not publisher-provided checksums.
- **Reported release observations pending QC-artifact reconciliation:** read-only
  inspection reported 1,216 / 1,572 / 1,530 cross-class overlap pixels in folds
  1/2/3, affecting 194 / 190 / 191 patches. It also reported 2,359,296 /
  3,801,371 / 4,325,424 pixels assigned to neither a positive class nor the supplied
  background, and zero pixels assigned to both a positive class and supplied
  background. M5 requires the anomaly-safe validator to reproduce complete counts
  in saved machine-readable evidence before these observations close any gate.
- **Raw-data and label-semantics rule:** cross-class overlaps and supplied-background
  voids are release-level QC observations, not permission to rewrite a source mask.
  The supplied background is not treated as the complement of positive classes.
  Void pixels remain unlabeled. Cross-class overlap is counted and flagged without
  class arbitration, priority, majority vote, or silent repair.
- **Fixed pre-freeze eligibility rule:** a nucleus instance touching at least one
  cross-class-overlap pixel remains in the manifest with reason
  `touches_cross_class_overlap`, but is excluded from all primary- and
  confirmatory-outcome-eligible analysis populations before splitting or modelling.
  Its source patch remains retained and is flagged as affected. The same eligibility
  mask applies to every primary cell and every required confirmatory scenario.
- **No tuned threshold:** one touching overlap pixel is sufficient for exclusion;
  no overlap-frequency, area, model-performance, or final-reference threshold is
  selected from the observed release. Complete counts and downstream support impact
  must be reported.
- **Preregistration and gate effect:** this rule is being fixed while
  `PRE_REGISTRATION.md` is DRAFT, so it is not an amendment. It must enter the M7
  freeze with validation/manifest hashes. M5 remains open until machine provenance,
  checksum, CRC/path-safety, complete QC, representative overlays, the full nucleus
  manifest, and complete cross-fold duplicate evidence pass. M6 and all later stage
  labels remain unchanged.
- **Independent reconciliation on 2026-07-18:** a read-only mmap/batch audit executed
  over all 7,901 patches (517,799,936 pixels) reproduced the reported fold-level
  overlap and void counts. It identified 1,411 overlap-touching raw instances (471 /
  465 / 475 by fold), zero positive-plus-supplied-background pixels, and 162 patches
  containing void pixels. The working machine-readable evidence is stored under
  `artifacts/qc_independent/`; it does not replace the canonical M5 validator or
  acquisition manifest. The audit also found 478 numeric instance IDs reused between
  class channels, with no such reuse touching overlap and no spatial same-ID overlap.
  Consequently, the immutable instance identity is always `(fold, patch, class,
  instance_id)` and a numeric ID is never merged or adjudicated across classes.
- **Canonical acquisition evidence on 2026-07-18:** the public read-only acquisition
  CLI verified the three archive identities, 33 ZIP members, nine extracted NPY
  arrays, nine README/licence documents, local licence/citation evidence, Git-ignore,
  and an unchanged 22-file raw metadata snapshot. The final manifest is
  `data/manifests/pannuke_acquisition.json` with SHA-256
  `837fd4692ca94df4bc9dfa929bc84b1bb4dbdcbd1858c90c247bb76b0b197111`; its bound
  report is `reports/pannuke_acquisition_verification.json` with SHA-256
  `f058de85ffb023be6fc5b9aa674e4470696f8cd822d2368fb54ab9d737067fd8`.
  This closes acquisition provenance only and does not advance the scientific stage.

## 2026-07-18 — Disconnected raw instance IDs

- A full read-only connectivity preflight inspected every positive channel in all
  7,901 PanNuke patches. Under the existing 4-connected definition it found 67 / 63
  / 81 disconnected raw instance IDs in folds 1/2/3 (211 total; 201 affected
  patches). Under 8-connectivity, 38 / 38 / 43 remained disconnected (119 total;
  116 affected patches), so the release property is not only diagonal adjacency.
- Disconnectedness is treated as a reportable release-level semantic annotation
  property, not evidence of archive, shape, dtype, or download corruption. Every
  raw `(fold, patch, class, instance_id)` remains intact and is flagged
  `disconnected_instance_id`; the validator and manifest never split, merge, repair,
  or relabel it.
- Invalid shapes, non-finite values, negative instance IDs, and non-integer-like
  instance IDs remain fatal structural errors. No frequency or component-count
  threshold is tuned to this release.
- Disconnected rows remain eligible for the pilot unless an independently fixed
  rule already excludes them. One identical primary/confirmatory eligibility rule
  will be frozen after the pilot without consulting final-reference outcomes.
  Evidence and rationale are recorded in
  `reports/pannuke_disconnected_instance_preflight.md`.

## 2026-07-18 — Immutable M5 publication and concurrency policy

- Every derived M5 bundle must fail closed against raw inventory additions, removals,
  byte changes, and publish-time races. Full inventory equality is checked rather
  than rehashing only paths known at the start.
- Acquisition, validation, QC, nucleus-manifest, duplicate-report, and embedding-cache
  outputs must use OS-held cross-process locks, atomic create-if-absent publication, and
  explicit idempotent readback. An O_EXCL lock left by a crash is never reclaimed or
  deleted automatically; it remains a visible fail-closed condition for explicit
  review.
- Rollback may remove only a path whose recorded file identity and content hash still
  prove ownership by the current transaction. A foreign replacement is preserved and
  the operation fails loudly. Broken symlinks and hardlink aliases count as occupied
  destinations, never as permission to overwrite.
- The full duplicate-audit lock must be keyed by the immutable raw release and span the
  shared frozen-ResNet cache as well as all five reports. This prevents a losing
  concurrent process from publishing a false incomplete-gate result while another
  process builds the cache.
- Multi-file bundles publish their canonical success marker last. A process crash may
  leave a partial set, but it cannot look complete; the next invocation refuses to
  mutate the partial set and requires explicit repair. No automatic stale-output
  deletion is permitted.
- These rules are engineering integrity controls, not scientific outcomes. A
  historical checkpoint passed two 540-test full runs and a 114-test focused suite;
  a later independent adversarial audit nevertheless found publication-race,
  ownership-safe rollback, output-under-raw, and cross-cache binding gaps. Those
  findings supersede the earlier “no remaining P1/P2” claim and require current
  integration QA before M5 closes. The scientific stage remains `PIPELINE_COMPLETE`.

## 2026-07-18 — Canonical PanNuke semantic/QC validation

- The exact M5 validation command completed a full semantic scan of all 7,901 source
  patches (517,799,936 pixels) with exit code 0 in 309.877 seconds. Raw masks were
  never modified, supplied background was not treated as a complement, void remained
  unlabeled, and no overlap pixel received an arbitrated class.
- Canonical totals exactly reproduced the prior independent audit: 4,318 cross-class
  overlap pixels in 575 patches; 10,486,091 void pixels in 162 patches; zero
  positive-plus-supplied-background pixels; and 1,411 overlap-touching instances.
  All 1,411 retain their full `(fold, patch, class, instance_id)` identity and are
  marked primary- and confirmatory-ineligible with reason
  `touches_cross_class_overlap`.
- The validation JSON SHA-256 is
  `094497f43e2ee0bd5dabddcd01f8c934657f130450a66f46600311451d36bc4a`;
  the QC artifact-manifest SHA-256 is
  `0b188ecc586ed772b29845e15e169fb492ed8d2ad0f5b1a6643531ccee10857f`;
  the QC overlay SHA-256 is
  `a1bd87dd397417d711d1d4937429eae5f5d972d3fa6ffa27a45129339587f10a`.
- Independent post-publication QA found zero discrepancies across every patch and
  affected instance, validated all bundle hashes, passed 46 focused tests, and
  visually reviewed 24 deterministic QC panels. Full rehashes of all 22 raw files
  found no added, removed, or changed paths; Git tracks no raw ZIP/NPY or derived
  Parquet/duplicate-cache file.
- At this historical checkpoint only the canonical semantic/QC-validation portion
  of M5 was closed; the nucleus manifest and complete exact/pHash/frozen-ResNet
  duplicate audit were still mandatory. Later dated sections record their execution.
  The project remained `PIPELINE_COMPLETE`.

## 2026-07-18 — Full PanNuke nucleus manifest

- The exact M5 manifest command completed in 531.455 seconds and published 189,744
  unique nucleus identities in 7,558 nonempty source-patch groups. The 343 patches
  without a positive nucleus remain represented by patch-level QC and do not receive
  synthetic nucleus/void rows.
- Raw identity is the complete `(fold, patch, class, instance_id)` tuple and maps
  one-to-one to `sample_id`. Every `group_id` equals one source patch; no patient- or
  WSI-level identifier is claimed from unavailable metadata.
- All 1,411 overlap-touching identities are retained with
  `touches_cross_class_overlap` and both eligibility fields false. The remaining
  188,333 rows are eligible under the prospective overlap rule. Label provenance is
  clean: `pre_corruption_label == observed_label == original_class == class`, every
  `is_injected_corruption` is false, and no active corruption metadata or
  `circularity_risk` is present.
- The Parquet SHA-256 is
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`;
  the 250-row summary CSV SHA-256 is
  `3bd0f37f2bfe73180b10194db6d0dadcde45675ed685970c42149c5bf8841c91`;
  the configuration hash is
  `d91bcb5856828fb43e0fa9097a8925754b924556b5a9c1af4b70c1b7de72de2e`.
- Independent validation passed public invariants and 47 focused tests, reconstructed
  the summary exactly, and confirmed all raw/Git-ignore bindings. At that checkpoint
  this closed the nucleus-manifest portion of M5 while duplicate evidence was still
  pending; the next dated section records its later completion.

## 2026-07-18 — Canonical full-release PanNuke duplicate audit

- The canonical audit covered all 7,901 source patches and all 20,798,326 cross-fold
  pairs for the exact, perceptual-hash, and official frozen torchvision ResNet-18
  signals. Pair coverage was 6,701,088 for folds 1--2, 7,229,632 for folds 1--3,
  and 6,867,606 for folds 2--3.
- It found zero exact pairs, 121 perceptual-average-hash candidates at Hamming
  distance at most 4, and zero ResNet candidates at cosine similarity at least
  0.995. These are methodologically distinct evidence signals, not a claim of
  statistical independence and not proof that an annotation or patch is wrong.
- All 121 candidates are `review_only`. No patch was deleted, relabelled, declared a
  duplicate automatically, made ineligible, or moved between splits. Final-reference
  outcomes were not accessed and the duplicate audit did not select or change the
  final-reference fold.
- The authoritative contract-upgraded JSON SHA-256 is
  `2a24d0f637bbf47e215f276e38d30b6bd65f1d312caa1b5b285ca5c00540612e`;
  the ranking CSV is
  `f83bdd1a08d91bc19b50c7e4d12778e0f2a9a2c7e08106252a12baa12049e891`;
  the Markdown report is
  `ba5d30adf55898b3c0f4ab8330b6e2090faf7428c128d54b50bef750c5031b02`;
  the visual grid is
  `15434e11e8bd8801547a99ed63a1bb2593c138009144c50ee33cde1e8b9f067e`;
  and the full patch-provenance CSV is
  `d829ba2b6333021f11ca512557a72b0b439b57e4d21a9f2474d4cd56606ffbd1`.
  The active embedding NPZ and sidecar hashes are respectively
  `ef99e931adc160f9d7fb9ab86bf0287dd8427b9f1d3c42fb0496377a5e287618`
  and `9e2506bbf4592c9012ba492aff46ea181bcfec1f78088334d71e213f6c660392`.
- After publication hardening, the first exact rerun stopped fail-closed because the
  old complete JSON lacked the newly mandatory cache/sidecar path and SHA-256
  bindings. The other four report artifacts and every scientific count were
  byte-identical. The complete old report bundle was checksum-copied to the ignored
  `.superseded/20260718_cache_binding_contract_upgrade` directory before all five
  active report outputs were republished transactionally. No cache, resume artifact,
  raw file, eligibility value, or split was changed. The public evidence validator
  subsequently returned zero errors.
- Independent machine reconciliation and visual review passed. The top-12 grid is
  legible and dominated by low-information pHash collisions; none was promoted to a
  confirmed duplicate. The superseded first report bundle is quarantined and is not
  M5 authority because its published runtime alias did not exactly match the frozen
  cache sidecar.

## 2026-07-18 — Component-covering projection for disconnected raw identities

- A raw `(fold, patch, class, instance_id)` remains one immutable union even when it
  contains multiple 4-connected components. Fixed-size derived target masks use
  `nearest_per_component_with_forward_fallback_v1`: each raw component is projected
  separately, and a deterministic forward-footprint fallback is used whenever
  nearest-neighbour projection would lose or disconnect that component.
- Every raw component must contribute at least one unique 4-connected output pixel
  or preprocessing fails closed. Crop geometry, exact source identity, per-component
  contribution, fallback use, collisions, newly created adjacency, and projected
  union topology are recorded and bound to downstream caches.
- This policy does not claim that the projected union preserves source topology.
  It never splits, merges, repairs, relabels, or writes back a raw annotation.
  Projected morphometrics are explicitly derived quantities; raw-resolution geometry
  and identity remain authoritative.
- A read-only full disconnected-identity scan covered 211 identities and 480 raw
  components. It used fallback for 6 identities, lost zero components, created zero
  component collisions, and recorded 3 projected-union topology changes. The scan
  was a temporary preflight whose exact inline body was not retained; this is an
  audit limitation, not a substitute for saved pilot provenance. Eligibility is
  unchanged: 209 such rows remain pilot-eligible after the independent overlap rule.

## 2026-07-18 — M6 pre-execution leakage and seed hardening

- The pilot must not construct, read, or publish final-reference sample identifiers
  or class labels. Its public selection artifact retains only class-free final group
  IDs, sample counts, and a one-way SHA-256 binding over official fold, group ID, and
  per-group count. No final-reference representation or outcome is extracted before
  the frozen later-stage evaluation requires it.
- Split, model, corruption, OOF, ranking-repeat, and restoration-repeat seeds and
  counts require exact integer values; booleans and floats are rejected rather than
  silently coerced. OOF provenance records the split/model seeds, the group-safe fold
  assignment binding, and the actual seed used by every fold model.
- The fixed pilot is explicitly non-confirmatory. It uses 20 downstream random
  restoration refits while retaining 100 cheap random ranking repeats, and reports
  that reduction rather than presenting it as the future confirmatory protocol.
- The corrected eligible pilot fixes its group-safe OOF partition before corruption:
  seed 223 and `pre_corruption_label` are used only for controlled-benchmark fold
  allocation. `StratifiedGroupKFold` is used when every training partition retains
  the fixed class order; otherwise the recorded deterministic fallback is seeded
  `GroupKFold`, with no nucleus-level fallback. Model fitting, audit scores, Cleanlab,
  and neighbour labels continue to use only `observed_label`. This preserves paired
  partitions across corruption rates, mechanisms, and seeds and matches the intended
  primary contract already recorded in `configs/primary.yaml`; no metric from the
  withdrawn pilot was used to choose the policy. It is privileged controlled-study
  information and is unavailable for audits of naturally occurring unknown errors,
  which must allocate from `observed_label`; pilot performance is not evidence of
  real-world unknown-error detection performance.

## 2026-07-18 — M6 privacy-safe development boundary and publication

- The corrected pilot consumes a pre-run, checksum-bound Parquet view containing
  complete canonical rows from official folds 1/2 only. Runtime access to the full
  canonical manifest is restricted to class-free eligibility columns; fold 3 is
  accessed only for raw path/size/SHA integrity and permitted aggregate/per-group
  class-free counts. Final-reference sample IDs, labels, class-derived geometry,
  representations, and instance-level outcomes are forbidden from the view,
  certificate, and run artifacts.
- The development-view bundle is never overwritten. Its Parquet and certificate are
  published as one ownership-safe transaction. Windows publication uses native
  handle-relative create/link/readback/delete operations; POSIX uses directory-FD
  relative staging, link, readback, and rollback. This prevents a concurrent parent
  path change from redirecting even a transient write into immutable raw data.
- The producer rehashes the exact complete raw inventory and all source evidence
  before and after publication, rejects additions/removals as well as changed bytes,
  and performs stable output identity/hash readback. Idempotent calls preserve bytes,
  length, and modification time.
- Privacy reconciliation scans artifact paths, text structures, NPZ array names,
  exact fold values, structured/subarray dtype fields, aliases and titles, Parquet
  values, schema/field metadata, recursive Arrow type parameters and extensions, and
  the serialized Arrow schema. Object/opaque/unreadable binary structures fail
  closed. An independent bounded review found no remaining P1/P2/P3 findings.
- The real view contains 123,090 exact canonical rows from folds 1/2. Its SHA-256 is
  `8107e1ddc033b08f03d3f351b5993ec1fd7a188677ee4c2afc0e4cbfe8432ef8`;
  the gate-certificate SHA-256 is
  `347f734c2355cd5009d631e959c077fe430adcad7e3997fe1306280004e90146`.
  Independent inspection found no fold-3 sample identity or outcome in either file
  and rehashed all 22 raw files without discrepancy. This checkpoint authorizes a
  new pilot attempt but does not itself satisfy M6 or change the project stage.
- The first run produced through this corrected privacy boundary,
  `20260718T134701.590268Z_pannuke_pilot_4b62a55d63`, passed technical integrity,
  numerical reconciliation, and the independent final-fold privacy oracle. It was
  nevertheless withdrawn from scientific-stage eligibility because its immutable
  report did not contain the exact mandatory phrase `recommended for expert review`.
  The semantically close wording was not silently accepted and the sealed run was not
  edited. Disposition record SHA-256:
  `467b8e8a265a30fdc06951231f37f72678a49b7f08877bcb3dbc8291183cf12a`.
  The producer now states that each high-ranked `potentially inconsistent annotation`
  is `recommended for expert review` in both the report and machine-readable
  limitations, with a regression test for both phrases. This correction is linguistic
  only and was not selected from pilot metric values; a new sealed run is required.

## 2026-07-18 — M5 closure and stage boundary

- The exact current-worktree validation and manifest commands passed over the full
  release in 311.7 s and 533.2 s respectively. Validation was idempotent and reported
  `source_masks_modified=false`; the manifest retained its canonical Parquet hash.
- The first hardened duplicate rerun stopped fail-closed because the prior JSON lacked
  newly mandatory cache/sidecar bindings. After checksum-preserving quarantine of the
  complete prior report bundle, the official command transactionally republished all
  five reports in 334.7 s. Scientific counts and every non-JSON hash were unchanged;
  the public evidence validator returned zero errors.
- Final QA passed 38 required focused tests, 210 broad PanNuke integration tests, and
  604 full tests. Ruff, format, mypy for 72 source files, `git diff --check`, artifact
  hashes, and ignore rules passed. The raw release, eligibility mask, split policy,
  final-reference availability, and source labels were not changed.
- At this historical M5 closure checkpoint, the formal milestone plan was 6/10 =
  60% and the status remained exactly `PIPELINE_COMPLETE` because no eligible
  real-data M6 pilot existed yet. The later M6 transition is recorded separately
  below; no final-reference outcome was used for selection or tuning.

## 2026-07-18 — First M6 execution withdrawn after post-seal audit

- The first real pilot execution technically completed and sealed run
  `20260718T033036.351640Z_pannuke_pilot_a6a660d93e`. The seal and registries are
  internally valid and must remain untouched as historical evidence.
- Post-seal inspection found that global eligibility provenance published 475 unique
  official-fold-3 sample IDs in five JSON artifacts. PanNuke sample IDs encode the
  nucleus class, so the run violated its own metadata-only final-reference contract
  even though the dedicated final binding contained only class-free groups/counts.
- Model-facing arrays, representations, OOF predictions, corruption, and restoration
  used development folds 1/2 only; the defect is procedural leakage rather than a
  claim that the reported pilot metrics were computed on fold 3. It is still a P1 and
  invalidates the run for scientific/stage use.
- Do not edit the sealed run or append a fabricated terminal status. Its disposition
  is `execution completed and integrity-valid, but scientific/stage eligibility
  withdrawn`. It must not satisfy M6, `PILOT_COMPLETE`, preregistration freeze,
  primary selection, or canonical reporting.
- Before rerunning M6, the pilot must read only class-free final metadata needed for
  group/count/eligibility binding; use a checksum-bound development-only manifest
  view for feature extraction; omit final IDs and class-derived hashes from every
  public artifact; scan all text and binary sidecars for actual final IDs; and add an
  external append-only disposition ledger that makes the withdrawal machine-enforced
  without mutating sealed evidence.
- The machine-enforced withdrawal was appended on 2026-07-18 with reason code
  `post_seal_final_fold_identity_leak`. The disposition record SHA-256 is
  `22926ba5bf8b7a2139e1fc10676ce5e0ade616a7ddd18adbc0bc10e7b2b1755d`;
  the resulting ledger and anchor SHA-256 values are respectively
  `69976dde3bfd9b7bfabc2a16ae55f2ab201f7298382effd5e933566f8c67c0d7`
  and `743c8458ae5972c1ec7bb30ca0eb5f2e3a343b331e4ac17c43699ed01479ed3a`.
  The anchor binds record count 1, the chain head, and the complete ledger hash.
  The sealed run root remains
  `0f25bf15c3359213e5e0b77608da331200cdf99a7b4316fe07fb357b76121096`;
  integrity verification still passes, while scientific-stage eligibility now
  fails closed. No sealed-run file was edited, moved, or deleted.

## 2026-07-18 — Final eligible M6 pilot and stage transition

- The canonical M6 pilot is sealed run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0`, artifact-root SHA-256
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`.
  It is the only real PanNuke pilot currently eligible for scientific stage use.
  The two earlier completed runs remain immutable and permanently ineligible under
  their existing disposition records; no historical record is rewritten.
- M6 acceptance is based on the sealed artifacts and independent read-only replay,
  not on the CLI success message alone. Integrity, exact registry binding,
  disposition eligibility, byte-identical pre-pilot inputs, required terminology,
  group-safe OOF, corruption separation, Cleanlab/neighbour reconciliation,
  restoration, full post-seal privacy scanning, and the post-run 22-file raw
  inventory all passed.
- The pilot contains 5,481 audit samples, 814 development reference-validation
  samples, and exactly 548 controlled label changes. Fold 3 contributes only the
  permitted class-free eligibility/group counts and byte-level integrity evidence;
  its sample IDs, labels, geometry, representations, and outcomes were unavailable
  to pilot selection, fitting, scoring, and evaluation.
- The controlled-corruption pilot metrics may inform only the declared pre-primary
  decisions. They do not establish that any original annotation or pathologist was
  wrong, and they do not estimate performance on naturally occurring unknown label
  inconsistencies. Outputs remain rankings of a `potentially inconsistent annotation`
  that is `recommended for expert review`, never automatic source-label changes or
  diagnostic outputs.
- With current-worktree QA at 641 passed plus Ruff, format, mypy, and whitespace
  checks, the formal stage changes from `PIPELINE_COMPLETE` to `PILOT_COMPLETE`.
  This closes M6 only; the full programme is not complete.

## 2026-07-18 — M7 must freeze an execution-ready source tree

- The one-shot preregistration freeze binds the source-tree hash. Therefore every
  production executor needed by the frozen primary and confirmatory definitions,
  including the then-missing confirmatory PanNuke bridge, grouped runner, restoration,
  reporting, and sealing path, had to be implemented and tested before M7 freezes.
  Adding that code after freeze would require an explicit dated amendment/refreeze.
- Required full-analysis representation caches, their provenance sidecars, and the
  representation-independence record must likewise be built and audited before the
  configs are frozen. Their real hashes may not be invented or copied from test
  fixtures. Final-fold outcomes remain unavailable while deterministic caches are
  constructed.
- The freeze implementation must fail closed for destinations under immutable raw
  data or a sealed run, reverify pilot eligibility and every source hash immediately
  before publication, reconcile the raw checksum manifest to the exact dataset, and
  publish/rollback the bundle using the ownership-safe transaction guarantees already
  required for M5. Normal required updates to `STATUS.md` and `DECISIONS.md` must not
  silently invalidate the execution-scope hash; the execution/governance snapshot
  policy must be made explicit and tested before the real freeze.
- `PRE_REGISTRATION.md` remains DRAFT until these conditions and every exact config
  value are complete. No primary or final-reference outcome may be inspected before
  the verified immutable freeze exists.

## 2026-07-18 — M7 execution identity and amendment authority

- The frozen execution identity is the exact `src/**`, `configs/**`, `pyproject.toml`,
  and `uv.lock` tree. Governance files are snapshotted and authenticated separately so
  required post-freeze updates to `STATUS.md`, `PLAN.md`, and `DECISIONS.md` do not
  silently redefine executable analyses.
- `configs/primary_frozen.yaml` and `configs/confirmatory_frozen.yaml` are generated
  byte-identical publications authenticated independently by the freeze bundle. They
  are excluded by exact path from the execution-tree root; any other config/code
  change still changes that root and blocks study execution.
- A change after freeze requires a new immutable, parent-hash-linked amendment bundle.
  It records reason, affected hypotheses/analyses, outcome-inspection state, and exact
  before/after identities. An amendment after outcome inspection may authorize only
  explicitly amended or exploratory analyses, never retroactively redefine the
  original primary claim.

## 2026-07-18 — Pilot-derived choices and shared group allocation

- The canonical M6 pilot is the only authority used to derive the confusion-targeted
  transition matrix and tissue weights. The deterministic producer record is
  `reports/pilot_derived_primary_parameters.json`, SHA-256
  `8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b`;
  it is development-only and must be snapshotted by M7 freeze.
- Primary and confirmatory adapters use one named allocation algorithm,
  `deterministic_group_greedy_class_distribution_v1`, with `source_patch_id`, fraction
  0.10, seed 223, and `pre_corruption_label`. It fails closed unless every fixed class
  is present in both reference-validation and audit partitions. There is no
  nucleus-level fallback.
- The primary matrix contains one explicit clean 0% cell plus the four required
  positive corruption mechanisms at 5%, 10%, and 20% over seeds 404–406. Zero-event
  inferential metrics remain structured not-applicable. Controlled cross-scenario
  comparisons use each scenario's own injected-event mask and identical group draws.

## 2026-07-18 — Full-release cache construction policy

- Full PanNuke representation extraction must be bounded-memory, resumable, and
  atomic. Above 10,000 manifest rows the builder uses chunked memmaps and a private
  sibling lease; no partial cache is stage-eligible. A successful output contains
  exactly the five required NPZ caches and their five sidecars, while the temporary
  workspace is closed and removed on Windows.
- Required primary provenance is produced transactionally before sidecar publication.
  CNN context-RGB and RGB-plus-target-mask views require distinct deterministic logical
  provenance even though both are byte-bound to the same crop cache. No manually
  fabricated cache, encoder, preprocessing, or independence hash is permitted.
- The unavailable optional-pathology authority is the regenerated audit SHA-256
  `5e568cf29e489d8948bfcd33feae5b292cb48837eb4c93754202a565778a6e4a`.
  Its embedded producer record, including cache-recipe SHA-256
  `d84f444e34341d0ee739cb8504ba94612010f15b1836993d36249d490855060f`,
  must be copied exactly rather than reconstructed by a consumer.
- Full production extraction must use the same-process CLI transaction with
  `--include-context-embeddings --independence-output
  reports\representation_independence.json --primary-config configs\primary.yaml`.
  A cache bundle without its successfully read-back strict independence artifact is
  not eligible for M7 finalization.

## 2026-07-18 — Full-release disk-safety gate

- A real three-sample PanNuke CUDA smoke establishes that the chunked producer,
  sidecar provenance, five-cache publication shape, and normal resume-workspace cleanup
  work on this machine. It does not authorize the full release build and is not a
  scientific result.
- The full build may begin only when drive C reports at least 35 GiB free. Its measured
  fixed working state is about 7.1 GiB and the conservative simultaneous peak for the
  private sibling workspace plus published bundle is 15--16 GiB. Starting from roughly
  19.39 GiB free would leave too little recovery margin, so the decision at that checkpoint was
  fail-closed `NO-GO` until space is freed or an equivalently verified project-local
  storage target is provided.
- This threshold is a source-enforced start gate, not a manual reminder. It runs on
  the output volume before full-manifest validation/staging; there is no CLI override.
  Only an effective extraction of at most 10,000 rows may use the smoke/test bypass.
- Raw PanNuke files and sealed runs are never cleanup candidates. No automated broad
  deletion is authorized; any cleanup must use an explicitly resolved derived path and
  preserve caches/evidence required by the current M7 transaction.
- The point-in-time decision evidence is
  `reports/m7_full_cache_preflight.json`, SHA-256
  `6bf9325b0e41ec84621116b9f84e5c132c58752c2afa95dc4238048d0d95ad72`.

## 2026-07-18 — Operational terminology clarification

- The frozen `SPEC.md` contains one historical descriptive phrase, “low estimated
  label quality for potential expert review.” It is not an authorized report label
  and is not edited in place.
- Every generated ranking, report, package, UI, and machine-readable interpretation
  must instead use the exact operational terms `potentially inconsistent annotation`
  and `recommended for expert review`, as required by the current safeguards and
  preregistration. This clarification narrows language only; it does not redefine an
  endpoint, metric, hypothesis, or analysis and never asserts a confirmed medical
  error or an incorrect pathologist.

## 2026-07-18 — Anchored cache publication and manifest authority

- Public cache files and directories are committed relative to a locked parent handle
  with no-overwrite semantics. A cache transaction may roll back only publication
  records it owns; it must not use path-based recursive deletion of a public target.
- Every destination component is rejected before staging when an immutable-run marker
  is present. Cross-volume publication and any inability to preserve anchored atomic
  semantics fail closed.
- Full-release extraction accepts only the canonical analysis authority: manifest
  SHA-256 `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`,
  analysis-eligible order SHA-256
  `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`,
  and exactly 188,333 eligible samples. Direct sample subsets are not eligible for the
  production cache transaction.
- The five-cache directory and strict independence JSON form one logical evidence
  transaction. A post-publication reopen or independence failure retracts only owned
  public entries and preserves the private resumable workspace; it cannot leave a
  stage-eligible partial bundle.

## 2026-07-18 — Canonical no-flag PanNuke validation parameters

- The canonical full validation evidence was produced with 100,000 requested sampled
  patches per fold and 24 deterministic overlay patches. The documented command with
  explicit flags and the README command without flags must select the same parameters.
- The CLI defaults are therefore 100,000 and 24. Lower values remain available only as
  explicit user choices and must fail closed rather than overwrite the canonical
  immutable bundle.
- This is an execution-default correction, not a change to the QC policy or data. The
  full semantic results, overlap/void counts, raw inventory, and patch/instance CSVs
  were identical across the diagnosed mismatch; only sampled diagnostics, overlay
  selection, dependent PNGs, and their hashes differed.

## 2026-07-18 — Confirmatory completion is default-deny and independently bound

- A confirmatory run is not stage-eligible merely because its sealed artifacts exist.
  Consumers require a durable positive post-seal attestation whose complete
  verification payload is independently recomputed from the sealed filesystem state.
  Missing completion evidence, missing attestation, failed withdrawal, or a stale
  integrity object leaves the run ineligible.
- The outcome-eligible entry point accepts only the production dependency set. Every
  CNN checkpoint is a loadable Torch artifact with the exact model, optimizer,
  scaler, RNG, history, CUDA, config, split, and data schema; plaintext or
  self-consistent replacement of checkpoint plus telemetry is insufficient.
- Per-cell/fold data and split fingerprints are computed before training directly from
  the bridge-held audit and reference arrays. Completion independently reconciles
  those fingerprints with checkpoint and telemetry, while the restoration certificate
  binds every audit, reference-validation, and final-reference replay input.
- These are execution-integrity rules only. Their tests do not inspect or create a
  primary or final-reference outcome and do not advance the project beyond
  `PILOT_COMPLETE`.

## 2026-07-18 — M7 pre-cache implementation gates are closed

- Descriptor-anchored cache publication is required for the frozen producer. Windows
  creates every missing parent relative to a retained handle; POSIX uses private
  no-replace promotion. Rollback deletes only a retained owned object or uses a
  quarantine/verify/restore transaction, and post-evidence commit re-authenticates the
  canonical manifest plus all five NPZ/sidecar pairs.
- The integrated publication evidence is 68 owner tests, 144 independent publication/
  authority tests, 90 handle/cache/rollback tests, and 8 stage-CLI tests. The final
  repository gate is 854 passed, Ruff check passed, all 141 files formatted, and mypy
  passed for 80 source files. The real PanNuke validator and sealed-pilot verifier also
  pass on the same source tree.
- These gates close the current code-path blockers but do not create the required full
  representation bundle and do not advance the status. At the 2026-07-18 checkpoint,
  M7 remained blocked before
  freeze by the enforced 35-GiB start threshold: the last production-CLI check found
  15,848,607,744 bytes free, and the latest read-only check after Pytest retention
  cleanup found 15,778,172,928 bytes free. No threshold override, partial-cache
  publication, or cleanup of raw/sealed evidence is permitted.

## 2026-07-19 — Disk gate satisfied; canonical validation limits shared by extraction

- Drive C exceeded the unchanged 35-GiB start threshold without an override. The first
  full production command passed that gate with 93,371,658,240 bytes free, then stopped
  fail-closed before cache staging because extraction used library validation limits
  32/6 against immutable canonical evidence produced with 100,000/24.
- The canonical validation JSON and overlay were not replaced. Raw PanNuke, the sealed
  eligible M6 run, public cache path, resume workspace, and independence path remained
  unchanged or absent as appropriate.
- Both PanNuke CLI entry points now use one canonical validation-limit authority, and
  extraction explicitly forwards 100,000/24. Focused owner and independent regressions,
  Ruff, format, and mypy pass. This authorizes a corrected producer retry only; it does
  not create cache provenance, freeze the preregistration, inspect an outcome, or
  advance the project beyond `PILOT_COMPLETE`.

## 2026-07-19 — Full M7 cache authority accepted and configs finalized

- The corrected production extraction completed with exit 0 on CUDA, verified 188,333
  canonical sample identities, and atomically published exactly five NPZ/sidecar pairs
  plus the strict schema-v2 representation-independence record. Independent full
  reopen/hash/content/lineage checks passed, with no resume workspace, temporary file,
  or lock left behind. Raw PanNuke and the eligible sealed M6 run remained unchanged.
- The independence authority has SHA-256
  `846f421284de381401761a8dc4ceb108d3f3f2a0eece379706be7f7a512789c7`.
  Context and target-highlighted ImageNet representations are
  `verified_independent`; engineered target features are `not_independent` and remain
  `circularity_risk` for instance-dependent confirmatory claims.
- The M7 finalizer was run once against both exact candidate-file CAS hashes. It
  transactionally published `configs/primary.yaml` with file/semantic SHA-256
  `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9` /
  `c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15`
  and `configs/confirmatory.yaml` with file/semantic SHA-256
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009` /
  `ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b`.
  Independent readback, cross-config validation, and 222-primary/108-confirmatory
  matrix expansion passed.
- `PRE_REGISTRATION.md` may therefore enter `READY_FOR_FREEZE` with only real cache,
  independence, and config identities. This decision does not itself freeze the
  preregistration or advance the project beyond `PILOT_COMPLETE`. Full QA, the
  one-shot freeze, and independent freeze verification remain mandatory. No primary
  study, confirmatory study, training, or final-reference outcome access occurred.

## 2026-07-19 — Dataset tree ordering made portable before freeze

- The freeze preflight found that the sealed M6 dataset SHA and the exact
  reconciliation SHA differed solely because native Windows `Path` ordering is
  component-wise and case-normalized, whereas reconciliation records are deliberately
  case-sensitive POSIX strings. Per-file reconciliation remained exact for all 22 raw
  files; raw data were not modified.
- Retroactively replacing the immutable M6 authority or using a host-native fix was
  rejected. `sha256_path` and the freeze digest now share
  `windows_compatible_relative_path_sort_key`: relative POSIX components are ordered by
  `lower()` with the original components as a deterministic tie-break, while the hash
  still encodes the original path bytes. This reproduces the existing M6 tree SHA-256
  `5647b4837fdaeb1281a5af0623f24aab1361263d3041549d012c8c5697fb31ed`
  on both Windows and POSIX.
- The separate case-sensitive inventory order and reconciliation-record semantic
  SHA-256
  `83e3eb7c4460c7c368a9bf70d49c3117f229f694daaa062456ba9f714c75651a`
  remain unchanged. A golden digest, component-order counterexample, cross-platform
  path representations, inventory reconstruction, and raw byte/mtime nonmutation are
  regression-tested.
- This technical compatibility correction is part of the source tree that M7 will
  freeze. It changes no scientific choice, cache, config, sealed M6 artifact, source
  annotation, outcome, or completion stage. The project remains `PILOT_COMPLETE`
  pending full QA and independently verified one-shot freeze.

## 2026-07-19 — Finalized project configs replace pre-finalization test assumptions

- The first mandatory full Pytest gate passed 855 tests and failed two because two
  repository-level tests treated `configs/primary.yaml` and
  `configs/confirmatory.yaml` as permanent pre-finalization templates. M7 deliberately
  finalizes those canonical paths transactionally, so their current contract is
  `READY_FOR_FREEZE` with real provenance.
- Reverting the independently verified configs to
  `awaiting_required_cache_provenance`, fabricating parallel template authorities, or
  weakening the strict validators was rejected. The two tests now authenticate the
  finalized configs, exact semantic/config-plan hashes, cache provenance, shared
  manifest/order authority, cross-config contract, and frozen optional-pathology
  blocker. Existing synthetic negative fixtures retain fail-closed coverage for
  incomplete provenance.
- The exact replacement cases passed 2/2, the complete contract file passed 79/79, and
  the full-suite rerun reported **857 passed in 389.08 s**. The initial 855/2 failure
  remains recorded as fail-closed gate evidence. This test-only correction does not
  inspect an outcome, freeze the preregistration, or advance the project beyond
  `PILOT_COMPLETE`. It closes only the Pytest portion of final QA; the remaining
  functional and freeze gates still apply.

## 2026-07-19 — Pre-freeze QA closed; one-shot transaction authorized

- The synchronized code/config tree passed 857 Pytest tests, Ruff check, Ruff format
  for 141 files, and mypy for 80 source files after all fail-closed corrections. The
  earlier 855/2 run remains preserved as the trigger for correcting two obsolete
  pre-finalization test assumptions.
- The exact full PanNuke validator passed its 7,901-patch semantic scan idempotently and
  reproduced every overlap/void/exclusion count with no source modification or class
  arbitration. The explicit read-only M6 post-seal verifier also passed, preserving the
  sealed root, group-safe OOF evidence, final-reference embargo, and required
  terminology.
- These results authorize exactly one invocation of the documented preregistration
  freeze transaction. Success must be followed by a fresh-process verification of the
  exact JSON-reported timestamp directory and both canonical frozen configs; an
  interrupted or ambiguous invocation must not be retried automatically. This decision
  does not itself freeze anything, inspect an outcome, or advance status beyond
  `PILOT_COMPLETE`.

## 2026-07-19 — First freeze attempt failed closed on volatile Git capture time

- The first explicitly authorized freeze invocation exited 1 after 108 s at the final
  pre-publication freshness check with
  `RuntimeError: Git state changed before preregistration publication`. The repository
  identity had not semantically drifted; the complete captured mapping was unequal
  solely because every capture includes a new `captured_at_utc` evidence timestamp.
- Requiring equality of the volatile observation time, suppressing Git freshness
  checks, or accepting any change to commit, branch, dirty state, or exact porcelain
  output was rejected. The correction retains `captured_at_utc` in frozen provenance,
  validates the capture schema fail-closed, and removes only that timestamp from
  equality. Every other key/value remains bound. Positive timestamp-only and negative
  availability/reason/commit/branch/dirty/porcelain/malformed-state regressions pass.
- No timestamped authority or canonical frozen config was published, staging and locks
  were cleaned, and the freeze root is empty. This failed invocation authorizes no
  automatic retry, inspects no outcome, and does not advance the project beyond
  `PILOT_COMPLETE`. The corrected focused gate passed 33/33 in 25.47 s. A retry still
  requires the mandatory full QA and functional readbacks plus a new explicit
  authorization.

## 2026-07-19 — Corrected Git freshness passed all retry gates; one second invocation authorized

- The corrected source tree passed the mandatory full suite with **866 passed in
  393.06 s**, Ruff check, Ruff format check, and mypy. A separate independent selection
  of marker-last publication, rollback, TOCTOU, Git comparison, malformed capture, and
  pilot recheck cases passed 13/13. This validates the narrow `captured_at_utc`
  comparison correction without weakening substantive Git-drift checks or erasing the
  first failed invocation.
- The corrected-tree full PanNuke validator passed all 7,901 patches idempotently with
  the canonical overlap, void, and exclusion counts, no class arbitration, and no raw
  mask modification. The explicit sealed-M6 verifier also passed with unchanged
  artifact root, group-safe OOF evidence, and unavailable final-reference outcomes for
  untouched official fold 3.
- Independent final preflight reconfirmed an empty rollback state: no frozen configs,
  no timestamped freeze entry, no staging directory, no publication lock, and no
  competing QA, validation, study, or freeze process. Two live Git captures had equal
  stable projections and differed only by their retained observation timestamps.
- Therefore exactly one second invocation of the documented preregistration freeze
  transaction is authorized. Its exact JSON-reported directory must be independently
  verified in a fresh process together with both canonical frozen configs before M7 or
  formal status can advance. Any nonzero, interrupted, or ambiguous result stops
  progression and does not authorize a third invocation. No outcome has been inspected;
  status remains `PILOT_COMPLETE` until verification succeeds.

## 2026-07-19 — M7 immutable preregistration freeze verified and closed

- The single authorized second transaction exited 0 in 209.523 s and created exactly
  one timestamped authority, `artifacts/preregistrations/20260719T002902.432341Z`, and
  the two canonical frozen config copies. It reported
  `PRE_REGISTRATION_FROZEN`, `integrity_verified=true`, artifact root
  `d2f1f3dec19021e7216630b53297035e705b6c407c3b5d84118ce7637411dd65`, and
  manifest SHA-256 `f223b6edd8364c90c476e13c9e3e3c14718418dfdc9938f2752492e65ee83101`.
- A fresh process verified the exact CLI-emitted directory and both canonical configs:
  `valid=true`; actual, expected, and CLI artifact roots match; the independently
  hashed manifest matches the CLI value; and the missing, added, changed, and error
  collections are empty. The first auxiliary wrapper exited 1 only because it looked
  for a nonexistent result key; its embedded library result was already valid. The
  corrected read-only wrapper exited 0, and the freeze transaction was not repeated.
- M7 is therefore closed and formal status advances from `PILOT_COMPLETE` to
  `PRE_REGISTRATION_FROZEN` before any primary or final-reference outcome inspection.
  The original preregistration input and immutable authority are not edited; future
  scientific changes require the dated amendment mechanism. M8 begins with a read-only
  execution-authority gate, while official fold 3 remains unavailable for tuning,
  selection, calibration, early stopping, or favourable inspection.

## 2026-07-19 — M8 primary authority gate passed; one foreground execution authorized

- The direct read-only `validate_primary_execution_gate` passed against the exact base
  freeze and bound the live 94-file execution-source tree, immutable raw dataset,
  nucleus manifest, duplicate/pathology evidence, eligible sealed M6 run, both frozen
  configs, and the fixed 222/185-required primary and 108-cell confirmatory plans. The
  first presentation wrapper failed only while JSON-encoding a `WindowsPath` after the
  gate had returned; the corrected `default=str` wrapper exited 0. No run or outcome was
  created by either read-only invocation, and the focused frozen-identity gate passed
  8/8.
- There is no remaining scientific selection step before primary execution. Current
  resource evidence records 86.49 GB free disk, 23.32 GB free RAM, and 11.716 GB free
  RTX 4070 VRAM, with no competing study process. The workload is intentionally large
  and frozen; it must not be reduced after outcome access merely for convenience.
- Exactly one foreground `experiment primary` invocation with the explicit authority
  `artifacts/preregistrations/20260719T002902.432341Z` is authorized. The default
  nonexistent `latest` path is rejected. The runner has no resume protocol: an
  interruption or execution failure must be retained as an ineligible immutable run,
  and any retry requires a separate recorded decision and new directory. Status remains
  `PRE_REGISTRATION_FROZEN` until post-seal verification genuinely enables
  `PRIMARY_STUDY_COMPLETE`.

## 2026-07-19 — First primary start failed before RunTracker on stale freeze-schema contract

- The single authorized primary CLI exited 1 after 106.6 s while preparing immutable
  inputs, before `RunTracker`, matrix execution, training, registry publication, or any
  primary/final-reference outcome. Filesystem and registry inspection found no new run
  or completion claim; confirmatory remains fail-closed and formal status remains
  `PRE_REGISTRATION_FROZEN`.
- The emitted binding message was broader than the actual cause. Config, manifest,
  sample order, cache provenance, completion stage, and base-freeze hashes all match.
  The adapter alone still required freeze-evidence schema 1, while the sole producer and
  primary gate deliberately require schema 3. Its synthetic fixture also fabricated
  schema 1 and therefore masked the contract drift.
- Accepting both schemas, changing the immutable freeze, weakening config/manifest
  binding, or immediately retrying was rejected. The narrow correction must share the
  exact schema-3 authority, update the fixture, explicitly reject legacy schema 1, and
  pass focused plus full QA. Because this changes the frozen execution-source root, a
  dated pre-outcome preregistration amendment must authenticate the corrected source
  before any separately authorized primary retry. No scientific config or analysis
  choice is changed.

## 2026-07-19 — Schema-3/source-only-amendment compatibility correction accepted

- One public schema-3 constant now binds the freeze producer, execution gate, and
  primary adapter; legacy schema 1 remains fail-closed. The live gate also exposes its
  recursively verified base-freeze directory, allowing a source-only amendment to use
  the immutable base `freeze_evidence.json` rather than looking for a file the amendment
  format deliberately does not contain. Config, manifest, cache, dataset, split,
  corruption, method, seed, budget, and statistical definitions are unchanged.
- Regression coverage now spans schema-3 success, legacy rejection, the exact default
  cache layout, amendment-to-base resolution, and failure before RunTracker. After one
  recorded test-fixture-only failure, the corrected focused set passed 42/42 and the
  mandatory full suite passed 869/869 in 392.53 s; Ruff, format, and mypy pass globally.
- Because execution source changed after the base freeze, direct use of that base for
  primary execution is prohibited. Exactly one source-only amendment publication is
  authorized with the immutable base as parent, unchanged preregistration/config bytes,
  explicit H1--H7 and affected primary/downstream-gate analyses, and
  `outcomes_not_inspected`. It must verify recursively before a separate amended gate or
  primary start can be authorized. Status remains `PRE_REGISTRATION_FROZEN`.

## 2026-07-19 — Verified amendment and real-cache preflight authorize one primary start

- The single authorized source-only amendment was published at
  `artifacts/preregistration_amendments/20260719T011146.248393Z` and independently
  verified through its immutable base parent. Its root is
  `962ab8b5110d062a314591f6144e0f94bebf68239f9ae8b014e2635eaf42031f`, its manifest
  SHA-256 is `f82f4b86d7cdce416108d68d72b6b71b2c4a8f8f1e6744de55962493efce0d53`, and its
  current execution-source root is
  `c0850f54e88483c1df76a4c8836343f667a7a1adbf2d05d571990cd6119cf532`. The
  preregistration and both scientific configs are byte-identical to the base authority;
  `outcomes_inspected=false` and every affected analysis is explicitly
  `amended_before_outcome_inspection`. The base authority alone is no longer eligible
  for current-source execution.
- The amended authority gate and a separate full real-cache input-adapter preflight
  both passed. The latter exited 0 in 145.6 s and reconstructed the exact frozen 222-cell
  primary plan with 185 required cells and the 108-cell confirmatory plan. Required
  engineered/context/highlighted caches are present; optional pathology remains a
  documented frozen blocker. The run-tree file snapshot was exactly 564 before and
  after, proving that this preflight created no RunTracker, training output, or outcome.
- The resource snapshot of 82.38 GiB free disk, 24.26 GiB free RAM, and 11,733 MiB free
  RTX 4070 VRAM is sufficient for the frozen workload. Compute duration and the absence
  of resume are the remaining operational risks; scientific scope may not be reduced
  after opening outcomes for convenience.
- Exactly one foreground `experiment primary` invocation using the amendment directory
  and both config snapshots inside that directory is now authorized. The earlier base
  invocation failed before RunTracker and has no run ID, and the CLI exposes no retry
  parameter; this is therefore the first tracked amended run, with no fabricated
  `retry_of_run_id`. Any nonzero or ambiguous termination consumes this authorization,
  must be preserved and documented, and does not authorize an automatic rerun.
- Formal status remains `PRE_REGISTRATION_FROZEN` and PLAN progress remains 8/10 = 80%
  until an eligible sealed primary result passes independent post-seal verification.
  Confirmatory remains locked throughout the primary execution.

## 2026-07-21 — Outcome-blind finalization-only successor policy accepted conditionally

- The authorized primary run
  `20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`
  completed and checksum-manifested all 185 required matrix cells, explicitly skipped
  the 37 unavailable optional pathology cells, and entered its frozen statistics path.
  It remains live as PID 20792. It is not interrupted, patched, resumed, duplicated, or
  treated as an eligible result.
- Read-only code-path proof found an unavoidable fail-closed finalization error after
  the expensive statistics work: the active runner does not supply the mandatory
  statistics-verification and restoration-readback attestations to the completion
  builder. The old statistics path also performs three full computations. Advancing
  single-core CPU with no published statistics quartet is consistent with the first
  computation and is not evidence of a loop.
- A full retry of the 185 cells is not scientifically or operationally necessary if the
  predecessor first terminates naturally as an integrity-valid, registry-backed failed
  seal. A new finalization-only successor may then reuse the completed cell artifacts
  read-only, but only after verifying every manifest entry, size, SHA-256, source
  identity, matrix reconciliation, restoration source, authority binding, and immediate
  predecessor lineage. It physically copies allowlisted evidence into a new run, never
  hardlinks or overwrites the predecessor, retrains zero cells, and recomputes only the
  missing frozen statistics/restoration attestations.
- This is a technical execution amendment, not an outcome-driven change. No study
  outcome has been inspected for selection or tuning. H1--H7, group-safe OOF rules,
  frozen splits, final reference fold, corruption definitions, review budgets,
  statistical estimands, restoration analysis, confirmatory matrix, and terminology
  remain unchanged. The optimized AP-versus-random implementation is bit-equivalent to
  the legacy computation for its narrowly recognized input and falls back to the legacy
  path otherwise.
- The successor is default-deny. It requires an exact schema-v2 child amendment linked
  to the immediate parent authority and exact failed predecessor, a new
  `retry_of_run_id`, anchored physical-copy boundaries with ownership-safe rollback,
  complete statistics/restoration readback, an immutable completion seal, registry
  integrity, and exactly one durable positive post-seal primary-stage attestation.
  Missing, duplicate, stale, fabricated, withdrawn, or tampered evidence cannot satisfy
  `PRIMARY_STUDY_COMPLETE` or unlock confirmatory.
- The implementation and tests remain isolated in
  `C:\Users\NATAN\Documents\AANCA_successor_staging_20260720` until PID 20792 reaches a
  natural terminal state. The stable delta is 22 source/test files. Staging QA and
  independent adversarial review are recorded in `STATUS.md`; the original executable
  source, raw PanNuke data, current run, and immutable authorities remain unchanged.
- Integration before terminalization is rejected because it would change the exact live
  execution-source root. Manual sealing of an unsealed run, termination followed by
  synthetic recovery, outcome inspection, or verifier bypass is also rejected. If the
  predecessor is not a valid failed seal, Option B stops fail-closed and requires a new
  recorded decision.
- During amendment publication and successor execution, the operational assumption is
  exclusive cooperative control of `artifacts/runs`; a malicious local actor with
  independent rename rights is outside the repository's existing pathname-based
  `RunTracker` threat model. This limitation must remain explicit and does not permit
  weakening the anchored copy or post-seal checks.
- The measured disk budget authorizes the finalization-only successor but not the
  current confirmatory checkpoint lifecycle. The successor imports approximately
  42.741 GiB and enforces a 10 GiB margin. The current confirmatory path would retain
  two copies of each of 180 CNN fold checkpoints, with a checkpoint-only peak lower
  bound of 60.170 GiB after the successor, exceeding available capacity before other
  artifacts. Confirmatory must remain locked until either at least 80 GiB is free after
  the successor or an independently tested, fail-closed single-copy checkpoint
  lifecycle is frozen in the child authority. Such a lifecycle change may alter only
  storage/rollback/reconciliation mechanics, never models, folds, seeds, predictions,
  metrics, or registered analyses.
- Formal status remains `PRE_REGISTRATION_FROZEN`; M8 remains open and milestone
  progress remains 8/10 = 80%. Only a verified successor may advance the project to
  `PRIMARY_STUDY_COMPLETE`, after which the frozen confirmatory study is the next M8
  gate.

## 2026-07-21 — Schema-v2 single-copy storage and physical-publication policy accepted conditionally

- The prospective outcome-blind child amendment must carry one explicit typed policy,
  selected before outcome inspection, authorizing direct single-copy confirmatory CNN
  checkpoint storage. Its canonical policy hash is
  `d67fb56a3d51d9748998f75baa3f18ab9468a7c231f7b492a98d3bdea021e3ff`.
  Ordinary child authorities may not inherit this permission, and the base freeze may
  not authorize it. The exact amendment must bind the immediate failed finalization
  predecessor, record `outcomes_inspected=false`, and name the affected primary and
  confirmatory execution/storage analyses.
- For real confirmatory execution the only permitted checkpoint location is
  `<run>/cells/<cell_id>/checkpoints/fold_XX.pt`. Top-level duplicate checkpoint trees,
  external roots, hardlinks, symlinks/reparse points, copy fallback, missing or extra
  `.pt` files, and manifest/filesystem disagreement are rejected. This is a storage,
  rollback and reconciliation change only; the registered models, data, group-safe OOF
  folds, seeds, predictions, metrics, restoration analysis and scientific estimands do
  not change.
- The policy hash must agree across the verified amendment, typed study gate, input
  bindings, provenance, metrics, completion evidence, independent pre-seal readback
  and live post-seal authority readback. Confirmatory cannot rely on caller-supplied
  policy data or a stale verified object. Any mismatch makes the run ineligible rather
  than silently falling back to the former duplicated layout.
- Immutable amendment and successor evidence is published as private physical copies
  with exclusive no-overwrite creation and atomic no-replace commit. Hardlink-based
  authority publication is rejected because a retained writable alias could mutate
  trusted evidence after verification. Every authority leaf, including the manifest
  and immutable marker, must be independently verified as regular, non-link,
  non-reparse and `st_nlink == 1`.
- This decision remains conditional until the active predecessor naturally produces an
  integrity-valid, registry-backed failed seal and the stable 29-file candidate passes
  its final full gates in the live workspace. `PRE_REGISTRATION.md` is not edited in
  place, no amendment has yet been published, and no successor or confirmatory run has
  yet been started.

## 2026-07-21 — Canonical checkpoint namespace and exact report terminology are mandatory

- The schema-v2 single-copy policy is interpreted as an exact filesystem contract, not
  merely a manifest convention. Every completed CNN fold must exist only at
  `cells/<cell_id>/checkpoints/fold_<fold_id:02d>.pt`. The verifier derives that set from
  the registered completed-cell/fold product rather than trusting mutable manifest
  paths, and any additional regular file inside a checkpoint directory is incompatible
  with `retained_copy_count=1`, regardless of extension.
- A self-consistent rename, manifest rehash, alternate filename, legacy top-level copy,
  extra `.pt`, extra `.bak`, hardlink, symlink or reparse point therefore fails closed.
  This strengthens verification of the already selected storage policy and changes no
  scientific input, prediction, metric or estimand.
- Primary and finalization-successor reports must contain the exact terms
  `potentially inconsistent annotation` and `recommended for expert review`. Similar
  wording is not accepted as a substitute. The text continues to state that source
  annotations are not modified and makes no claim that model disagreement proves an
  expert or medical error.
- These requirements were fixed before integration, amendment publication or outcome
  inspection. Focused tests and independent re-review pass; final full-suite validation
  remains required before the conditional Option-B candidate can be integrated.

## 2026-07-21 - M8-to-M9 compatibility corrections are sequenced fail-closed

- The primary finalization successor is not coupled to M9 implementation and may run
  first once its predecessor has a valid failed seal and the Option-B amendment is
  verified. No M9 output may be produced merely because successor completion exists.
- Before the real confirmatory run, confirmatory selection evidence must hash final
  reference groups as the unique sorted group set. Sample-aligned group vectors remain
  unchanged and continue to have their own partition hashes. This corrects a technical
  incompatibility caused by multiple nuclei sharing one patch `group_id`; it changes no
  split, sample, model, score, metric or estimand. Because execution source changes, it
  must be authorized by an explicit technical child amendment before confirmatory M8.
- The remaining M9 repairs occur only after an integrity-valid, positively attested
  `CONFIRMATORY_COMPLETE` and before any ranking/assets/package execution. They must
  bind the real tracked experiment name, frozen semantic cache sidecar, deterministic
  final-test-fold producer, separate canonical and asset manifests, durable positive
  attestation, exact pre-asset top/random cohort, and the frozen external-validation
  config.
- Stage-mode M9 must enforce the frozen external contract: 100 top-ranked and 100 random
  records, seed 707, exact display roles and response options. It may not accept
  convenience overrides, silently replace a selected sample whose asset is missing, or
  expose private cohort IDs/scores in the blinded public package.
- Until focused/full QA and a positive real-structure chain prove those contracts,
  `EXTERNAL_VALIDATION_READY` is prohibited. Actual expert responses and responsible
  multi-rater analysis remain required for `EXTERNAL_VALIDATION_COMPLETE`.

## 2026-07-21 - Windows full-suite launcher is spawn-safe and path-bounded

- Mandatory Windows QA must invoke pytest through `python -m pytest`, not through
  programmatic `pytest.main()` in a `python -` stdin process. Cleanlab's optional API
  uses multiprocessing; the stdin launcher has no spawn-safe importable `__main__` and
  can recursively create workers without advancing the test. This is an execution
  harness defect, not evidence of a pipeline loop.
- Full test basetemp paths must be short enough for the deepest registered chunked-cache
  filenames under the Windows path limit. A failure caused solely by an overlong
  diagnostic basetemp is not accepted as a pass; the affected tests and then the full
  suite must pass using a short base.
- Staging-only provenance tests may be explicitly deselected from the staging-tree full
  command only if they are separately executed against the original repository path
  with the exact staging source on `PYTHONPATH`. Counts and both commands must be
  reported; no failure may be hidden or relabelled.

## 2026-07-21 - Final-reference group identity is set-level; partitions remain sample-level

- `final_reference_group_ids_sha256` denotes the identity of the final-reference group
  set and is therefore computed from the unique lexically sorted group IDs. Repeated
  nuclei from one source patch may not change that set identity merely by duplicating a
  patch ID in the sample-aligned vector.
- The sample-aligned `group_ids_sha256` remains order- and duplicate-preserving, and the
  complete vector remains in sealed NPZ evidence. This keeps partition/sample replay
  exact while making the explicit group-set authority compatible with M9's unique
  newline group file.
- Producer, runner-side evidence and independent completion readback each enforce the
  set-level hash. A multiset hash is rejected even if all other declared files are
  internally rehashed. The correction changes only derived provenance/manifest hashes,
  not data membership, labels, models, scores, metrics, restoration or estimands.
- The correction was implemented and fully tested before confirmatory execution and
  before outcome inspection. It must be named in the real technical amendment together
  with Option B and single-copy checkpoint storage; no additional pre-M8 source change
  is currently known to be required for M9 compatibility.

## 2026-07-27 — Replace failed-seal-only Option B with one bounded interrupted-orphan recovery

- A Windows restart is evidenced at **2026-07-27 12:37:04** Europe/Warsaw, with
  Kernel-General event ID 12 at **12:37:05**. PID **20792** is absent, while its run
  still says `running` and lacks `failure.json`, `artifact_manifest.json` and
  `.immutable.json`. The previous requirement for a natural, registry-backed failed
  seal can no longer be satisfied by this predecessor without fabricating or mutating
  evidence.
- The user authorized an operational redesign. This decision supersedes only the
  2026-07-21 requirement that recovery depend on a valid failed predecessor seal. It
  does not weaken no-overwrite, source-integrity, independent readback, new-run
  lineage, sealing, registry or fail-closed requirements.
- Exactly one real interrupted-orphan recovery/resume invocation may be authorized
  after focused synthetic recovery tests and all mandatory gates pass. It must keep the
  predecessor read-only, bind the exact source/config/freeze identities, verify all 185
  required completed cells and their checksums, copy only an explicit allowlist into a
  new directory carrying `retry_of_run_id`, record the truthful outcome-inspection
  declaration, perform
  no training, compute each missing finalization operation at most once, and pass the
  ordinary completion, seal, integrity, registry and independent stage checks.
- Any missing, changed, extra, ambiguous or unverifiable input causes that recovery run
  to fail closed. It must not fall back to training, patch the orphan, weaken a
  verifier, fabricate a seal or start another recovery automatically. A further attempt
  would require new evidence and a separate recorded decision.
- Before real execution, publish a dated technical amendment describing the host
  interruption, reason for the operational change, exact source binding, read-only
  import boundary and confirmatory impact. `PRE_REGISTRATION.md` and all frozen
  scientific definitions remain unchanged.
- Formal status remains `PRE_REGISTRATION_FROZEN`, progress remains **8/10 = 80%**, and
  M8 remains open until a genuinely sealed, integrity-verified and positively attested
  recovery run satisfies `PRIMARY_STUDY_COMPLETE`.

## 2026-07-27 — Accidental outcome exposure requires post-outcome recovery classification

- At **2026-07-27T10:57:07Z**, an over-broad read-only `rg` search for process-receipt
  evidence traversed real primary result artifacts and emitted fragments of subgroup,
  statistics and ranking values. The values are not repeated and must not be used for
  any decision.
- An audit found no earlier real primary values in project status/decision documents,
  stored test logs or root run logs; synthetic successor tests did not inspect real
  outcomes. That prior absence does not undo the new exposure.
- All recovery authorities and amendments must now state `outcomes_inspected=true`.
  Any recovered primary result affected by this technical amendment is
  `amended_or_exploratory`; it must not be represented as an original, outcome-blind,
  unamended confirmatory result.
- No scientific parameter may change in response: the primary/confirmatory configs,
  hypotheses, estimands, OOF/group rules, final-reference membership, corruption
  policy, exclusions and decision thresholds remain frozen. This prohibition is a
  fail-closed recovery condition and must be attested in the new run.

## 2026-07-27 — Recovery critical path has bounded I/O and single-pass computation

- Independent typed readbacks prove that the orphan already contains the complete
  185-required-cell matrix and valid restoration evidence. A zero-training recovery is
  therefore the selected implementation; a 185-cell retraining retry is not the
  default path.
- The new recovery command may perform one source qualification, one physical
  allowlisted copy, one destination matrix/restoration readback, one lightweight
  statistics-closure check, one seal, one integrity verification and one stage
  attestation. It contains no loop that can launch another recovery or primary.
- The ordinary primary aggregation path now computes heavy statistics once and verifies
  the persisted outputs against that in-memory computation and a fresh source readback.
  Re-running the public full semantic verifier is an explicit audit action, not an
  automatic second or third production computation.
- The 2026-07-27 host-interruption receipt is a corroborating operational artifact, not
  a replacement for a run seal. Only the new destination run may receive a new seal;
  the orphan remains permanently unsealed and ineligible.

## 2026-07-27 — Recovery numeric evidence must have its own truthful capability

- The existing inherited-statistics capability is specific to a sealed failed
  finalization predecessor and names its artifact root and artifact-manifest hash.
  The reboot orphan has neither object. Substituting the orphan snapshot root or
  statistics-manifest hash into those differently named fields is rejected as
  misleading provenance.
- Recovery therefore uses a separately named, in-process capability and saved
  provenance that bind the verified technical amendment, source run ID, exact orphan
  snapshot root, source status/source-tree identities, typed matrix readback, exact
  statistics quartet, comparison count, trust assumption and limitation.
- This capability permits only content-addressed inherited-statistics attestation. It
  does not claim fresh numerical recomputation, a predecessor seal, or an ordinary
  unamended primary analysis. The recovery remains `amended_or_exploratory`.
- Ordinary primary statistics verification and the historical sealed-predecessor
  finalization capability remain fail-closed and unchanged. Neither capability can be
  substituted for the other.

## 2026-07-27 — Use one streaming WOF/LZX physical recovery copy

- The orphan contains 46,291,408,111 logical bytes while the destination volume has
  only 31,623,032,832 free bytes. Starting the former uncompressed whole-tree copy
  would necessarily fail its preflight.
- A manifest-only successor and hardlinks are rejected because they would leave the
  new qualification dependent on mutable source pathnames or shared file objects.
  Deleting or modifying the orphan and raw PanNuke data is also rejected.
- The recovery copy will instead create independent destination files and apply
  Windows WOF/LZX exactly once immediately after each durable file copy. Logical
  bytes and SHA-256 identities remain unchanged. Temporary probes measured 23.1:1
  for a representative large JSON and 3.0:1 for representative NPZ and CSV files.
- Safety does not depend on those ratios. Before each file, the copier must require
  enough free space for that file's full logical size plus a fixed margin; after
  compression it must recheck the margin. Compression failure, identity change,
  hash mismatch, low space or boundary ambiguity stops the sole recovery invocation
  without retry.
- This changes storage mechanics only. Frozen hypotheses, models, seeds, folds,
  estimands, thresholds, exclusions, terminology and confirmatory configuration are
  unchanged.

## 2026-07-27 — Retire failed-seal Option B and bound amendment publication

- The host reboot made a natural failed seal impossible, and accidental result
  exposure made every new `outcomes_inspected=false` recovery declaration false.
  Therefore all prospective 2026-07-21 conditions requiring a sealed failed
  predecessor, outcome-blind schema-v2 finalization successor or PID polling are
  superseded. Historical pre-exposure authorities retain their truthful historical
  declarations, but they cannot authorize the current recovery.
- The recurring `aanca-primary-to-option-b` automation was deleted. It encoded the
  superseded failed-seal/outcome-blind branch and could otherwise repeatedly re-enter
  an impossible gate. There is no recurring process monitor, automatic recovery,
  automatic retry or second-primary launcher.
- The only permitted path is the bounded post-outcome orphan recovery:
  `outcomes_inspected=true`, canonical timestamp
  `2026-07-27T10:57:07.000000Z`, disposition `amended_or_exploratory`, zero scientific
  method changes, independent physical WOF/LZX copy, zero training and one real
  recovery invocation.
- The first amendment publication failed before any authority was created because the
  authorization and amendment represented the same timestamp with different text
  precision. One deliberate corrected publication is allowed with the canonical
  microsecond string. This is not a retry of recovery: no RunTracker, destination run,
  copy, seal or scientific computation existed. Any further publication failure
  requires a new recorded decision.
- The corrected amendment must bind `ConfirmatoryStoragePolicy()` as well as the
  recovery authorization. Outcome-eligible confirmatory execution still requires the
  exact padded production namespace
  `cells/<cell_id>/checkpoints/fold_<fold_id:02d>.pt`, one physical checkpoint per
  fold, no links/reparse points and exact policy hashes across gate, provenance,
  metrics, completion and post-seal readback. Compatibility accepted by synthetic
  legacy fixtures is not an authority for real execution.

## 2026-07-27 — One real orphan-recovery execution is authorized

- Immutable amendment `20260727T133947.089370Z` passed immediate and fresh-process
  verification. The subsequent public preflight independently matched the authorized
  source snapshot, all 185 required cells, all 37 optional skips, restoration,
  registered statistics and the WOF/LZX disk boundary.
- The real destination identity is fixed as
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery`. Exactly one executor
  invocation is permitted. It may copy 2,270 allowlisted artifacts and finalize the
  new run, but it may not call training, the matrix executor, statistics aggregation,
  fallback or automatic retry.
- A terminal failure or ambiguous publication consumes this authorization for
  automatic execution. The destination and evidence must be preserved, no second
  recovery or primary may start automatically, and a new decision is required.

## 2026-07-27 — Accept the single recovered primary and retire its execution authority

- The only authorized bounded recovery,
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery`, returned exit code 0,
  sealed without overwrite, entered the append-only integrity registry, and passed
  two independent fresh-process integrity, lineage, typed-readback, and stage audits.
  Its exact artifact root is
  `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`
  and its positive stage-attestation record SHA-256 is
  `5af827544502fbdf688a73916ec58b5dac0984c5a682a33ce6dfc97538228871`.
- This is accepted as the project primary authority and advances the formal status to
  `PRIMARY_STUDY_COMPLETE`. Its analysis disposition remains permanently
  `amended_or_exploratory` because `outcomes_inspected=true`; no original unamended
  primary claim is restored.
- The run reuses exactly 185 verified required cells, records 37 preregistered optional
  skips, and performs zero retraining, matrix execution, fallback, or automatic retry.
  The recovery authorization is now consumed and must never be invoked again.
- The interrupted predecessor
  `20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`
  remains unchanged, unsealed, read-only, and ineligible. It will not be retroactively
  repaired or treated as a second primary.
- This checkpoint originally named the unchanged frozen confirmatory study as the
  next execution. The later resource-bounded decision below supersedes that
  operational instruction after the runtime/storage/resume audit: the original
  confirmatory definition remains immutable and deferred, the reduced sensitivity
  cannot emit `CONFIRMATORY_COMPLETE`, and M9 remains locked.

## 2026-07-27 — Close confirmatory launch plumbing before execution

- A read-only integration audit found that the public `experiment confirmatory`
  command does not pass four mandatory runner inputs: crop-cache path, crop-cache
  SHA-256, crop-metadata SHA-256, and raw-inventory SHA-256. Starting it now would
  fail with `TypeError` before scientific execution.
- Real confirmatory execution is therefore not launched merely because primary
  eligibility passed. The next change is limited to outcome-independent,
  fail-closed CLI/input and capacity plumbing derived from already frozen cache
  sidecars and configuration. Frozen hypotheses, estimands, groups, labels, models,
  seeds, thresholds, exclusions, terminology, and final-reference isolation remain
  unchanged.
- Because this technical change alters the execution-source identity after outcomes
  were inspected, it requires full tests and a dated immutable technical amendment
  before the first real confirmatory RunTracker. A read-only preflight must reject
  missing/mismatched caches, insufficient space, or any authority mismatch without
  creating a run.

## 2026-07-27 — Replace the impractical live launch with a bounded sensitivity profile

- Do not launch the original 108-cell frozen confirmatory study on this workstation.
  A code-path, storage, RAM, and measured-runtime audit found three independent
  operational blockers: the frozen contract requires checkpoint/resume while the
  production runner rejects resume; 180 full CNN fold checkpoints have a roughly
  30.0 GiB lower bound; and the earliest plausible training stop is about 5.5 days,
  with a typical execution of 10–15 days. Starting that path would repeat the
  long-running, nonrecoverable failure mode the user explicitly asked to retire.
- Preserve `configs/confirmatory_frozen.yaml`, `SPEC.md`,
  `PRE_REGISTRATION.md`, the sealed recovered primary, and the interrupted orphan
  byte-for-byte. No original confirmatory result or completion claim is replaced.
- Add a separate strict `resource_bounded_confirmatory_v1` profile selected solely
  from operational constraints, not exposed outcome values. Its exact scope is:
  all three official rotations; both frozen corruption cells; five-fold group-safe
  OOF; seed 303; required `cnn_context_rgb` plus the three existing ImageNet frozen
  representations; four fixed maximum CNN epochs with patience two; an explicitly
  amended cross-representation ensemble; 2,000 paired group bootstraps; and the
  unchanged restoration budget. The expected matrix is 24 required cells, including
  six CNN cells and 30 CNN fold fits.
- This profile is a feasibility/sensitivity analysis only. Its authority must record
  `outcomes_inspected=true`, `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, and `completion_stage=null`.
  It cannot emit a positive `CONFIRMATORY_COMPLETE` attestation, cannot unlock M9,
  and leaves the project stage at `PRIMARY_STUDY_COMPLETE`.
- Historical primary authority and current execution authority are separate
  capabilities. The former continues to authenticate the sealed recovered primary;
  the latter must be a direct immutable child that binds the current source and
  resource config without rebinding or mutating primary history.
- The live resource run is permitted only after exact-profile tests, dual-authority
  tamper tests, RAM-safe preprocessing equivalence, explicit read-only checkpoint
  successor tests, the complete mandatory QA sequence, the real PanNuke validator,
  a sealed lifecycle rehearsal/readback, a capacity preflight with a 10 GiB margin,
  and one verified post-outcome amendment. There is no automatic retry or fallback.

## 2026-07-27 — Accept the final 15-path receipt and authorize one schema-v4 publication

- Accept the current resource execution source only under the exact pinned tuple:
  root `1179f91725a3027c0397e87691774377bbd4ba5469d588390c72b0b88515547b`,
  manifest
  `03bcc6020e3be5a22fe257c45820e4e8ebece3ce471c2b6cecff0e3e9419fc66`,
  and delta
  `7abd9e1627728c4ce89f59cc6162283ec8963468816db6c64849fff1a5ec290e`.
  The delta contains exactly 15 allowlisted paths: five additions, ten modifications,
  and no removal. Any later execution-source change invalidates this authorization.
- The full current-tree gate is accepted as passing: 1086 tests, Ruff, format, mypy,
  public resource CLI help, and the full semantic PanNuke validation all succeeded.
  The validator retained its complete `status=valid` JSON and empty stderr; its
  asynchronous launcher did not preserve the terminated process's exit-code object,
  so the unchanged five-minute command is not repeated merely to replace that
  operational logging limitation.
- Permit exactly one call to the ignored one-shot publisher. It must create a
  root-wide exclusive attempt marker before the expensive authority builder, take
  and compare all six registry/ledger identities before and after, reject any
  existing or staged resource authority, call the schema-v4 build/create path exactly
  once, perform public fresh live verification, and prove global uniqueness before
  declaring success. It has no loop, automatic retry, or alternate publisher.
- The publisher binds `outcomes_inspected=true`,
  `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, and `completion_stage=null`. It cannot
  change `SPEC.md`, `PRE_REGISTRATION.md`, either frozen config, the recovered
  primary, or the formal `PRIMARY_STUDY_COMPLETE` status.
- Two independent safety observations remain P2 hardening items rather than current
  launch blockers: component-wise reparse identity for the already verified ordinary
  canonical `artifacts/runs` directory, and an additional end-of-qualification
  integrity readback for a cooperative immutable predecessor. The fresh resource run
  has no predecessor; any future successor still requires an explicit canonical,
  registry-backed, sealed failed resource run and at least one valid checkpoint.
- If the single publication attempt fails or its terminal state is ambiguous, retain
  its attempt/failure evidence, do not invoke the publisher again, and record a new
  decision before any further authority or scientific execution.

## 2026-07-27 — Accept the unique verified resource authority C

- Accept immutable schema-v4 authority
  `20260727T170413.080954Z` as the sole resource execution authority. It is the direct
  child of P, has artifact root
  `57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627`,
  manifest
  `4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156`,
  and resource-authorization SHA-256
  `6e5c974be10d95e6f9f1dfbf1c09586473691bd4b6f8459a1d9c21e759bb12dc`.
- The publication authorization is consumed. The attempt and success markers are
  retained permanently; no second C, publisher call, or automatic retry is
  permitted.
- C authorizes only the fixed 24-cell non-claiming sensitivity under the pinned
  source/config receipt. It does not revise or replace the original 108-cell frozen
  confirmatory definition and cannot produce a positive scientific stage.
- Before any scientific RunTracker, require one fresh synthetic lifecycle rehearsal,
  a separate fresh-process rehearsal verification that publishes a readiness run,
  and one public `--preflight-only` invocation proving current disk, RAM, CUDA,
  cuDNN, AMP, official-weight, and finite forward/backward-smoke gates.

## 2026-07-27 — Retain C as immutable but technically inoperable

- The first public resource preflight stopped before RunTracker creation with
  `ValueError: CNN logical encoder/preprocessing provenance does not recompute from
  the verified crop view: cnn_context_rgb`. No resource run, failure run, marker,
  lock, cell, scientific result, or disposition record was created.
- The PanNuke crop NPZ and sidecar are correct and must not be regenerated. Their
  exact file, content, and sample-order bindings all match. The mismatch is limited
  to the logical CNN provenance record: C snapshots the newer allowlisted
  `models/cnn.py` but its resource config retained logical hashes derived from the
  historical version of that file. The fail-closed validator behaved correctly.
- C remains immutable, integral, and historically truthful as the sole authority
  produced by its consumed one-shot publisher. It must not be edited, republished,
  silently reinterpreted, or passed to a real execution after this incompatibility
  became known.
- A second independent pre-tracker blocker was observed: confirmatory input
  construction retained role-specific copies for all three rotations and peaked at
  approximately 21.2 GiB resident working memory. The implementation must use
  shared, read-only storage with bounded materialisation before another public
  preflight.
- The only permitted correction path is a new explicit technical successor that
  preserves C and the frozen scientific files, records the exact supersession reason,
  binds the corrected logical provenance and final execution-source receipt, retains
  `outcomes_inspected=true`, `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, `study_outcome_eligible=false`, and
  `completion_stage=null`, and is published once only after full QA. It may not read
  outcome values, change the 24-cell scientific profile, weaken provenance
  validation, or authorize more than one future real run.

## 2026-07-27 — Require the exact physical 12-array carrier before authority D

- Permanently invalidate the first authority-D input freeze as non-publishable
  evidence. Its five-array workspace plan omitted seven checksum-bound crop members.
  Keep the directory and its invalidation receipt unchanged; a replacement must use
  a new path and may not overwrite or reinterpret v1.
- Capacity-v3 represents physical workspace storage. Therefore
  `workspace_partition_index_bytes=4,521,144` is the serialized size of nine NPY
  files, including nine 128-byte headers; the raw int64 payload is 4,519,992 bytes.
  The public authority validator must compare the physical NPY total when deriving
  the 4,298,703,413-byte projected workspace. It may not weaken the fixed capacity
  constants to accommodate an incomplete provider plan.
- Require every source NPZ and sidecar to remain a plain, non-reparse file through
  the entire hash operation and final workspace verification. Bind the physical file
  identity and every lexical parent identity before and after hashing. A
  checksum-identical file symlink or parent junction swap must fail closed.
- Permit fixed-width `U` and `S` arrays but continue to reject object, structured,
  subdtype, or pickle-dependent arrays. Real PanNuke `U` identifiers remain zero-copy
  read-only memmaps. A legacy `S` identifier vector may receive exactly one
  lightweight, read-only normalization to canonical Unicode; images, masks, feature
  matrices, and role partitions may not be copied by that compatibility path.
- Accept the final execution-source receipt only under the exact tuple: 102
  artifacts; root
  `2a568873f317cd9d5ef87cd991dbc5488ceb00c4fbc924af3828a531ae372477`;
  atomic manifest
  `5868954f97131398f487534cb7cbe9acfab5b3cb511293836e57afb04feb62c4`;
  15-path delta
  `9e30dfeba955f6b8e96b1a914a2a79bbb610e8a80fea87a2c3c0e791f778cc3d`;
  and unchanged change-kind digest
  `e48f0b72011cc43a412ad014ad67b3b82088a7c3030336c1679d51f2bc950dcc`.
  Any later execution-source change invalidates this acceptance and requires new
  full gates and a new receipt.
- The passing evidence is 123 focused tests, 1,135 full tests, full Ruff/format/mypy,
  public CLI help, and a full real-PanNuke semantic validation. The deliberately
  aborted pre-fix full test process is non-qualifying and retained only as evidence
  that the P1 stopped advancement before publication.
- Authorize exactly one fresh input-freeze operation to the new v2 directory. The
  freeze may only derive and O_EXCL-write the four outcome-independent controller
  inputs. It creates no D marker or authority. After independent v2 hash and
  capacity readback, require one read-only controller preflight. Publication remains
  forbidden unless that preflight passes with zero source drift and zero run-state
  changes.
- Authority C, the recovered primary, raw PanNuke, `SPEC.md`,
  `PRE_REGISTRATION.md`, and both frozen configs remain immutable. Authority D and
  any later resource run must retain `outcomes_inspected=true`,
  `amended_or_exploratory`, `original_confirmatory_claim_allowed=false`,
  `study_outcome_eligible=false`, and `completion_stage=null`. Formal status remains
  `PRIMARY_STUDY_COMPLETE`; this path cannot unlock M9.

## 2026-07-27 — Accept v2 and authorize one schema-v5 authority-D publication

- Accept only
  `artifacts/resource_control/authority_d_inputs_20260727Tfinal_source_v2` as the
  publication input bundle. Its four exact SHA-256 values are
  `f5bd9384ac22be05b53e5b7fa987a059c84f74051067f47a7d30b14789e01c08`,
  `d6436c55e3134807ee0eb99d7e3b5c0a0416b06c1ced22372e31ba2ce268f176`,
  `d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b`,
  and
  `89c6d475d691b480478b76d25e5e96653a3d225d738ce23c6726a5bab409e6c3`.
  v1 remains permanently ineligible.
- The independently reconstructed v2 carrier and the sole read-only controller
  preflight both passed. They prove the final 15-path source delta, exact 12/9
  workspace, public capacity-v3 totals, exact CNN logical-provenance correction,
  unchanged run-state hashes, and absence of any D marker, authority, or run.
- Permit exactly one `--publish-once` call with those four paths and the explicit
  frozen-receipt SHA-256 pin. The controller must create its root-wide O_EXCL attempt
  marker before calling the schema-v5 authorization builder and amendment creator,
  must call each exactly once, and must verify the new direct child of C through the
  public fresh-process authority path before writing a success marker.
- A failed, partial, or ambiguous publication consumes automatic execution
  permission. Preserve every marker and artifact, do not invoke the publisher again,
  and record a new decision. A clean success must yield exactly one direct technical
  successor D, unchanged six-file run state, zero scientific cells, and no
  `RunTracker`.
- Publication does not authorize claims or change project completion. D must retain
  `outcomes_inspected=true`, `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, `study_outcome_eligible=false`, and
  `completion_stage=null`. Formal status stays `PRIMARY_STUDY_COMPLETE`; M9 remains
  locked.

## 2026-07-27 — Consume the failed Authority-D attempt and require new authority

- Record the sole authorized Authority-D publication as terminal
  `failed_no_retry`. Permanently preserve attempt marker
  `8c93e65eca0bb4d64af4e94012004d74178448941cc746675d0e8e72ac5e90e2`,
  failure marker
  `de123683a56ab0349c44536e969f536843ed0c557bae8573187664cab7fc8615`,
  and the v2 inputs. The absent success marker and ownership-safe removal of the
  attempted D directory are part of the evidence. The controller, marker namespace,
  frozen receipt, and v2 bundle may not be reused to publish another authority.
- Classify the cause as a deterministic P1 in authority validation. Schema-v5 D
  validation currently inherits C's storage policy through an API that also requires
  C to remain the effective execution leaf. Once D is physically present, that
  requirement rejects C and therefore makes D self-invalidating. The exact hashed
  exception is recorded in `STATUS.md`; this is not evidence of corrupt data or a
  failed immutable bundle.
- Any correction must introduce a historical, sealed readback for C's canonical
  storage policy and schema-v4 authorization during D-chain validation. It must not
  weaken the separate public execution gate: after a valid D exists, execution
  through superseded C must still fail closed.
- Before any separately authorized replacement publication, require regressions
  proving: real/minimal C-to-D creation and generic verification; historical C
  remains chain-valid; effective execution through C is rejected after D; exactly
  one D is accepted; competing/forked D candidates fail closed; publication rollback
  is ownership-safe. Then require full `pytest`, Ruff check, Ruff format check, mypy,
  relevant CLI, real PanNuke validation, a new execution-source receipt, a new
  immutable input freeze, and an independently verified read-only preflight.
- A future replacement is not an automatic retry and is not presently authorized.
  It requires an explicit new decision that binds both failure markers and fixes the
  circular validator before creating any new publication marker or amendment.
  Lifecycle rehearsal, resource execution, confirmatory execution, and M9 remain
  forbidden. Formal status remains `PRIMARY_STUDY_COMPLETE`.

## 2026-07-27 - Accept the historical/effective split but require an atomic replacement publisher

- Accept the private historical sealed-policy readback as the correction for the
  schema-v5 self-invalidating publication path. Its permitted scope is limited to
  inheriting and verifying Authority C's canonical storage policy while creating or
  validating its direct schema-v5 successor.
- Do not export or use the historical helper from study, lifecycle, attestation, or
  execution code. Those paths must continue to require the unique effective
  authority, reject superseded C after D exists, and reject competing D candidates.
- Accept the completed qualification evidence: 41 focused regressions, 1,142 full
  tests, full Ruff/format/mypy, the public Authority-C verifier, and full real
  PanNuke semantic validation. This qualifies the validator correction only; it does
  not authorize a replacement publication or scientific execution.
- Before recording a new publication authorization, require a new controller design
  in which all fallible post-publication validation runs inside the amendment
  creator's ownership-safe transaction. A failure before commit must roll back only
  paths owned by that transaction.
- Define the durable success marker as the publication commit point. After that
  marker exists, logging and stdout must be best-effort and no catch path may write
  a failure marker. Accept only `attempt + exact success + exact D` as committed and
  `attempt + exact failure + no D` as rolled-back failure. Treat every mixed,
  partial, attempt-only, or valid-D-without-success state as `STOP/ambiguous`; do not
  delete it and do not retry automatically.
- Require an independent fresh-Python verifier inside the transaction before commit
  to check the generic chain, typed schema-v5 authorization, unique effective D,
  rejected execution through C, intact historical C, canonical storage policy,
  exact file hashes, exact amendment intent, unchanged run state, and absence of
  scientific execution.
- Any future replacement must use a new source receipt, immutable input bundle,
  controller hash, authorization receipt, and attempt/success/failure namespace.
  It must bind the terminal v2 failed-attempt evidence. The old v2 assets remain
  preserved and permanently non-reusable.
- Formal status remains `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9 remains
  locked.

## 2026-07-28 - Accept the atomic substrate; authorize live wiring and a new outcome-blind freeze

- Accept the tracked replacement controller's atomic state-machine, bounded
  fresh-process verifier, anchored bounded read/write primitives, exact
  project/input/authorization bindings, and fault-injection regressions as the
  required substrate for a separately authorized Authority-D replacement.
- The qualification evidence is 73 focused tests, 173 broader integration tests
  with one platform skip, 1,238 full tests with one platform skip, full
  Ruff/format/mypy, a read-only real-C `READY` classification, and a full valid
  real-PanNuke semantic scan. This qualifies controller mechanics only; it does not
  create or authorize Authority D or scientific execution.
- Permit one final tracked source change that wires outcome-blind reconstruction,
  immutable input freezing, live preflight, and the exact amendment-creator call
  into this substrate. The old controller, v1/v2 inputs, and old marker namespace
  remain historical evidence and may not be imported, overwritten, or reused as
  replacement inputs.
- After that wiring passes the focused and mandatory gates, permit creation of the
  canonical prior-publication failure receipt and exactly one new immutable
  replacement input bundle. Both writes must be O_EXCL, independently read back,
  and must occur before any replacement attempt marker. Any source change after
  the freeze invalidates the bundle and requires a new decision.
- Publication remains blocked until the wired CLI passes an exact read-only live
  preflight against the new bundle, current six-file run state, Authority C, the
  terminal old failure evidence, capacity plus the fixed 10 GiB margin, and an
  empty replacement marker/D namespace. A later publication authorization must
  name the exact new hashes and permits at most one attempt.
- Do not change `SPEC.md`, `PRE_REGISTRATION.md`, either frozen config, Authority C,
  the recovered primary, raw PanNuke, or the old failed-attempt evidence. Formal
  status remains `PRIMARY_STUDY_COMPLETE`; M8 remains open at 8/10 and M9 remains
  locked.

## 2026-07-28 - Accept the live adapter; keep freeze conditional on capacity

- Accept the final live replacement adapter and its exact-source qualification:
  96 controller regressions, 1,274 full-suite passes with one expected Windows
  skip, full Ruff/format/mypy, the full real-PanNuke semantic scan, and exact
  read-only READY classification. The final controller regressions cover every
  blocking P1/P2 condition found during the pre-gate reviews.
- Require exactly one canonical frozen-input directory:
  `artifacts/resource_control/authority_d_replacement_inputs_v1`. Reject every
  suffix alias and case variant. Validate the destination before derivation or
  writing, use one foundation and two resource gates, and require full live-context
  equality both immediately before the first write and after bundle readback.
  Ownership-safe rollback is mandatory for any post-write drift.
- Require CLI and state evidence to use tri-state publication semantics:
  committed `A+S+D` is `true`; READY and exact rolled-back `A+F/no-D` are `false`;
  every ambiguous state, including A+D without a terminal marker, is `null` and
  stops with no retry.
- Treat `outcome_value_interpretation_performed=false` only as evidence that this
  implementation and resource decision did not inspect values for tuning. Any
  published Authority D must still retain `outcomes_inspected=true` from the
  recorded 2026-07-27 exposure, remain `amended_or_exploratory`, permit no original
  confirmatory claim, and carry `completion_stage=null`.
- Keep the previously granted singleton-freeze authority conditional on a fresh
  capacity-v3 pass. At the recorded observation, free space was
  25,711,448,064 bytes versus the exact 28,189,458,997-byte requirement. Do not
  create the prior-failure receipt or bundle while below that threshold.
- Permit manual cleanup of the exact pytest session created by the qualifying full
  suite,
  `C:\Users\NATAN\AppData\Local\Temp\pytest-of-NATAN\pytest-1667`, after confirming
  that no pytest process is active. Do not infer authority to delete `C:\pt3`,
  protected runs, raw PanNuke, or any project evidence.
- Once capacity passes, the next state-changing operation is the one canonical
  freeze only. Preflight, publication authorization, the single Authority-D attempt,
  lifecycle, resource execution, and M9 remain ordered later gates. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at 8/10 and M9 remains
  locked.

## 2026-07-28 - Preserve and invalidate replacement v1; require v2 after the precision fix

- Accept that the capacity condition cleared at **169,348,308,992 free bytes
  (~157.7 GiB)** and that the one authorized
  `authority_d_replacement_inputs_v1` freeze completed successfully.
- Record the first v1 preflight as a fail-closed control-plane compatibility
  failure. The controller required microsecond-canonical timestamps, while the
  immutable historical failed-preflight evidence records
  `2026-07-27T17:30:54.689Z` at millisecond precision. The timestamp order is
  semantically valid. This is not source drift, scientific execution, publication,
  or evidence corruption.
- Preserve the complete v1 bundle unchanged and explicitly invalidate it as
  non-publishable. It may remain only as historical evidence and must not be
  overwritten, reused, moved, deleted, or supplied to a publication attempt.
- Permit a bounded correction that accepts only the immutable historical timestamp
  precision without rewriting historical evidence or weakening microsecond
  canonicalization for new evidence. After focused and mandatory gates pass,
  require a new O_EXCL v2 bundle with fresh source bindings and two independently
  verified read-only preflights.
- The v1 invalidation receipt must be canonical, no-overwrite, independently read
  back, and bind the exact v1 files, prior-failure receipt, failed-preflight logs,
  error digest, corrected controller, and unchanged run-state root. The active
  classifier must accept v1 only as an exact invalidated historical pair; all
  preflight, authorization, attempt, and publication bindings must select v2 only.
- Do not create a publication authorization, replacement marker, Authority D,
  lifecycle run, resource-science run, or M9 artifact from v1. No automatic retry
  is authorized.
- Formal completion remains `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-28 - Accept the corrected controller and permit one v1 invalidation

- Accept the dedicated exact historical-timestamp parser and retain strict
  microsecond canonicalization for every newly created timestamp. Historical
  evidence must remain byte-identical.
- Accept the final atomicity hardening: invalidation, freeze, authorization, and
  attempt claiming overlap on the canonical v2 root; rollback is allowed only
  while exclusion ownership is positively proven; freeze checks ownership around
  every individual write; loss of ownership, incomplete rollback, or lock-exit
  uncertainty is terminal ambiguous state and never authorizes cleanup or retry.
- Accept the qualification evidence on the exact final controller snapshot:
  123 focused tests, 252 broader integration tests with one expected Windows skip,
  1,301 full-suite tests with one expected Windows skip, complete
  Ruff/format/mypy, read-only CLI checks, protected-integrity verification, and a
  full valid real-PanNuke semantic scan.
- Permit exactly one O_EXCL publication of the canonical v1 invalidation receipt.
  It must bind the exact immutable v1 files, prior-failure receipt,
  failed-preflight evidence, corrected controller, and unchanged run-state root.
  It may not rewrite, move, delete, or reuse v1.
- After independent invalidation readback, permit exactly one canonical v2 freeze
  and two stable outcome-value-blind read-only preflights. Publication
  authorization remains a later, separate one-attempt gate and is not implied by
  invalidation or freeze.
- Do not change `SPEC.md`, `PRE_REGISTRATION.md`, either frozen config, Authority
  C, the recovered primary, raw PanNuke, or historical failure evidence. Do not
  inspect scientific outcome values for tuning.
- Formal completion remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-28 - Require exact replacement-v2 resource and process evidence

- Require authorization-v2 to preserve two independent outcome-blind live
  preflights plus a third full-live readback under the protocol and Authority-C
  parent locks. Capacity and compute observations must use closed schemas, exact
  types, exact policies and phases, canonical stored UTC text, ordered chronology,
  recomputed nested hashes, and explicit no-outcome/no-tuning flags.
- Normalize the trusted resource runner's live `+00:00` timestamp to canonical
  six-microsecond `Z` only before evidence construction. Continue to reject stored
  timestamp aliases rather than normalizing untrusted receipt content.
- Require a final exact Q+I3+authorization-v2/no-A2/no-S2/no-F2 state scan,
  unchanged Authority-C inventory, no candidate D, unchanged legacy-lock state,
  and owned O_EXCL authorization bytes immediately before returning the one-attempt
  authorization.
- Require sealed authorization readback to bind the complete immutable Q and
  input-v3 carrier, source/config/manifest/history/run-state/workspace identities,
  schema-v3 technical authorization, inherited storage policy, and recomputed
  intent. A file-record root or authorization hash alone is insufficient.
- Require bounded fresh verification to attest the entire process tree, not only
  the direct `Popen` child. A surviving or unprovable descendant, incomplete pipe
  cleanup, inconsistent diagnostic state, PID/hash mismatch, raw stdout/stderr
  inclusion, or invalid successor purpose fails closed and cannot support S2 or F2.
- The passing schema-v3, authorization-v2, and fresh-security focused suites are
  implementation subgates only. They do not authorize Q, input-v3,
  authorization-v2, A2/S2/F2, Authority D, lifecycle, resource execution,
  confirmatory execution, or M9. A state-changing decision remains contingent on
  the complete controller, full mandatory gates, and a separate exact live
  qualification.
- Formal completion remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open and
  M9 remains locked.

## 2026-07-28 - Freeze the replacement-v2 schema/Q contract before any write

- Accept schema-v3 only as an additive dispatch: omission of replacement failure
  lineage must retain byte-identical schema-v2 authorization and intent behavior.
- Authenticate the consumed controller through exactly five agreeing historical
  sources, ordered as invalidation receipt, attempt marker, authorization receipt,
  frozen-source receipt, and source allowlist. Record the diagnosed fixed legacy
  controller and the qualifying replacement-v2 controller as separate identities;
  neither authorizes retry of replacement-v1.
- Require Q to contain exact integer/Boolean/path types, exact Authority C and
  protected pins, all six run-state records and their aggregate, and the exact
  ordered 28-record read set. Require a pre-lock process guard, an under-lock probe
  stored in Q, and a final under-lock probe after readback.
- Use a separate Authority-C parent guard while publishing Q, freezing input-v3, or
  authorizing publication. This overlaps every cooperating amendment creator at
  target C and closes the candidate-D race without adding C to the publication
  protocol lock used by the later creator transaction.
- Split mutable run-state verification by phase. Before D commits, Q/input/auth and
  the fresh verifier must rehash current run-state files. After D commits, ordinary
  typed/effective verification must authenticate the immutable Q bytes and embedded
  run-state snapshot but permit later governed append-only lifecycle changes.
- Treat the input-v3 singleton as four native reconstructed documents, never as a
  generic caller-supplied envelope and never as reused input-v2 bytes. A live pre-D
  read must independently rederive the CNN evidence, source delta, workspace,
  resource gates, and schema-v3 technical authorization.
- The passed schema/Q subgates authorize continued code, test, and documentation
  work only. They do not authorize writing Q, input-v3, authorization-v2, publication
  markers, Authority D, lifecycle state, or scientific results. Formal status stays
  `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9 remains locked.

## 2026-07-28 - Accept exact v1 invalidation and permit one canonical v2 freeze

- Accept the exact canonical invalidation receipt
  `0b9af7cdb9ca3fcb60c8dd6c123eda22f13631c1188ff390cd9421998e28e997`
  after independent readback proved that v1, its prior-failure lineage, the
  corrected controller, run state, protected files, and A/P/C authority inventory
  remain unchanged.
- Treat v1 as permanently preserved invalid, non-publishable evidence. It may not
  be overwritten, moved, deleted, repaired, supplied to preflight, or consumed by
  a publication attempt.
- Permit exactly one O_EXCL freeze of the canonical
  `authority_d_replacement_inputs_v2` singleton. The freeze must reconstruct all
  four payloads from live, outcome-value-blind evidence, pass the capacity and
  compute gates, bind the invalidation receipt, and independently read back every
  file.
- After a successful v2 readback, require two separate read-only preflights with
  stable immutable contract and fingerprint evidence. Dynamic observation times
  and free-space observations may differ only where the schema explicitly allows
  them.
- Publication authorization, Authority D, lifecycle execution, confirmatory
  execution, and M9 remain forbidden until their later ordered gates. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`.

## 2026-07-28 - Accept the active v2 freeze; require duplicate read-only preflight

- Accept the one canonical v2 four-file singleton after local exact inventory and
  hash readback. It is the only input bundle eligible for the replacement
  preflight and must remain byte-identical.
- The v2 bundle is an input freeze, not publication authorization. It created no
  A/S/F marker, Authority D, lifecycle state, scientific run, or outcome
  interpretation.
- Require an independent bundle audit plus two distinct fresh-process,
  outcome-value-blind `--preflight-only` executions. Both must bind the same
  immutable authorization, source, run-state, invalidation, prior-failure,
  Authority C, and frozen bundle identities. Their fresh timestamp, destination,
  intent, fingerprint, and resource observation are proposal-specific and must
  each validate independently rather than compare equal.
- Permit a one-attempt publication authorization only after those checks pass.
  Authorization must still be published separately with O_EXCL and independently
  read back before the single publication attempt.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9
  remains locked.

## 2026-07-28 - Accept duplicate v2 preflight; permit one authorization receipt

- Accept the independent v2 GO audit and the two separately executed passing
  preflights. Both bind immutable technical authorization
  `886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8`
  and the exact active v2 lineage.
- Treat each preflight's timestamp, proposed D directory, intent, fingerprint,
  compute observation, and free-space observation as a fresh internally bound
  proposal. Their inequality is expected and is not source or authority drift.
- Permit exactly one O_EXCL publication of the closed one-attempt authorization
  receipt. It must run its own repeated live preflight under the shared v2/A/S/F
  exclusion lock and bind one random 64-hex attempt ID, one exact D timestamp and
  destination, and `max_attempt_count=1`.
- The authorization receipt is not the attempt marker and does not publish
  Authority D. Independently read it back before permitting the separately guarded
  single publication call. Any ambiguity or retained partial state stops without
  automatic retry.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9
  remains locked.

## 2026-07-28 - Record the authorization receipt; retain the publication gate

- Record that the sole one-attempt authorization receipt was created with SHA-256
  `4c892f7e518964a46569290e1a486d7f7e193121ed870522895946413dbee565`
  and attempt ID
  `c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86`.
- The receipt consumes the authorization-creation step but not the publication
  attempt. It must not be overwritten or recreated.
- Retain the publication gate until an independent readback proves all receipt and
  live bindings, `READY`, absent A/S/F, absent D, and no conflicting lock/process.
  A failed or ambiguous readback stops; it does not authorize repair or a second
  receipt.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`.

## 2026-07-28 - Accept authorization audit GO; permit exactly one publication

- Accept the independent canonical receipt audit and frozen-timestamp live
  preflight reproduction. The receipt, active v2, historical lineage, source and
  run-state roots, protected files, Authority C, capacity, and empty A/S/F/D/lock
  namespace satisfy the publication gate.
- Permit exactly one invocation of the separately guarded `--publish-once`
  operation bound to attempt
  `c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86`
  and intended D `20260728T181920.303224Z`.
- The command may create the attempt marker and then either commit exact A+S+D or
  record exact A+F/no-D after creator rollback. It may never be repeated,
  automatically retried, or repaired in place. Any other state is terminal
  ambiguous and must be preserved.
- A committed D is a technical governance successor with
  `outcomes_inspected=true`, `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, and `completion_stage=null`; it
  does not itself execute science or change formal completion.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9
  remains locked.

## 2026-07-28 - Consume the sole attempt as terminal rolled-back failure

- Record exact A+F/no-D as the terminal disposition of attempt
  `c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86`.
  Preserve attempt SHA
  `e602993753949ecbd5bfe3dfd9ba77d1890d63ae6232db9db6d66caff48e3ace`
  and failure SHA
  `e66305dac9a2c1b59d5cb554081470c1947b939d8a07ade3cf77046f0e353b12`
  permanently.
- The published authorization is consumed. No automatic or manual rerun of this
  controller namespace, receipt, v2 bundle, timestamp, destination, or attempt ID
  is allowed.
- Classify the durable failure at the fresh-process verifier boundary:
  `FreshVerifierError: fresh verifier process did not exit cleanly`. The
  transaction rollback succeeded and D is absent, but the schema retained only a
  hash of the controller-level exception and not the bounded child stderr/return
  code. Do not infer the child-level verification cause.
- Permit only bounded read-only qualification and code-path analysis. Any future
  correction would require a new diagnostic evidence design, new namespace,
  newly frozen source/input lineage, full gates, and a separately recorded
  authorization; none is authorized by this decision.
- Lifecycle, resource execution, confirmatory execution, and M9 remain blocked.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open.

## 2026-07-28 - Accept the Windows process-boundary correction; require a new protocol

- Accept the deterministic portability diagnosis. On this Windows venv, launching
  `sys.executable` directly creates a redirector child and a Python grandchild,
  while both the verifier and controller require the executing verifier to be the
  controller's direct child. This necessarily reaches the recorded nonzero fresh
  verifier exit. It is a control-plane defect, not PanNuke corruption, capacity
  failure, scientific drift, or outcome inspection.
- Accept the minimal correction for future processes: retain the exact venv
  executable as `argv[0]`, but on Windows supply the validated
  `sys._base_executable` as `Popen(executable=...)`. Require a real regular
  non-reparse executable in the exact real `sys.base_prefix`, keep `shell=False`,
  isolated mode, bounded in-memory pipes, timeout/cleanup, nonce, and direct
  PID/PPID validation unchanged, and fail before spawning on any mismatch.
- Accept qualification evidence on corrected controller
  `e20278105b6ea4e2786713c64d9e8cf7bb06d9e4c8155f35a46861e72cb67b5f`:
  27 focused process tests, 131 controller tests, 260 broader integration tests,
  1,309 full-suite passes with one expected Windows skip, and complete
  Ruff/format/mypy gates.
- Preserve the exact terminal replacement-v1 lineage permanently. Its input-v2
  bundle was valid but is now consumed and non-publishable. Never reuse or alter
  authorization-v1, A1/F1, the absent S1 condition, attempt ID, timestamp,
  intended D path, intent, fingerprint, nonce, old capacity observation, or old
  lock token. The corrected live controller must not be compared to the
  historical controller pin `cbea3c3536dbad729383c96e0ef602042c7e3c4e000f9b0cb79e50c13b2ced58`;
  that old identity must instead be authenticated through exact historical
  receipts and hashes.
- Require a genuinely new protocol before another Authority-D publication can be
  considered:
  - terminal qualification receipt
    `resource_authority_d_replacement_v1_terminal_qualification_v1.json`;
  - newly reconstructed singleton `authority_d_replacement_inputs_v3`;
  - authorization
    `resource_authority_d_replacement_publication_authorization_v2.json`;
  - marker namespace
    `resource_authority_d_replacement_v2_publication_{attempt,success,failure}.json`;
  - classifier states `QUALIFICATION_REQUIRED`, `INPUT_FREEZE_REQUIRED`,
    `AUTHORIZATION_REQUIRED`, `READY`, `ROLLED_BACK_FAILURE`, `COMMITTED`, and
    fail-closed `STOP_AMBIGUOUS`.
- Require the future committed D to remain a direct child of C at chain depth 4
  and to embed the terminal replacement-failure lineage in a new typed
  technical-successor authorization schema/policy. It must retain
  `outcomes_inspected=true`, `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, `study_outcome_eligible=false`,
  and `completion_stage=null`. `PRE_REGISTRATION.md` remains frozen and unchanged;
  the future D is the dated technical amendment.
- Improve only future failure evidence to retain bounded verifier diagnostics
  such as requested/effective executable, process IDs, return code, and
  stdout/stderr sizes and hashes. Do not retrofit or rewrite F1. The current
  process-boundary defect is fully explained, but later verifier checks were not
  reached and may reveal additional defects.
- Authorize source/test/documentation implementation and read-only qualification
  of this replacement-v2 protocol. Do not yet authorize the terminal receipt,
  v3 freeze, authorization-v2, A2, Authority D, lifecycle, resource science,
  confirmatory science, or M9. A new state-changing permission follows only after
  the implementation passes all mandatory gates and independent historical/live
  readback.
- Formal completion remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-29 - Accept external unattended infrastructure, not a real-job launch

- Accept the independently tested, external LocalAppData supervisor release
  `09a0ffe11d52a997d9c0b02fe98d5f82433be13bd4d4150795379a0995b2a20a`
  and closed release manifest
  `010c38d37078ab162a65d0a085bc19bc23a0578196b8afc112b4a08435237719`
  as qualified infrastructure for a future separately authorized long process.
  It is not repository execution source, scientific authority, an amendment, or
  permission to launch a job.
- Require every protected job to use the read-only versioned source, pinned
  runtime, exact reviewed argv and working directory, one canonical
  authorization receipt, `max_attempt_count=1`, and
  `automatic_retry_allowed=false`. Required success evidence must include the
  exact terminal seal and integrity receipt plus an exit-0 read-only verifier.
- Accept the per-user Startup entry only for fail-closed `recover-all`. It may
  reconcile an existing consumed attempt after reboot, write STOP where proof is
  incomplete, and make the single permitted diagnosis wake. It may never invoke
  `run`, queue science, use Task Scheduler, or relaunch a disappeared process.
- Pin non-interactive handoff to session
  `019faaf3-c547-79e1-b0eb-26e35d214642` and the exact tested command
  `codex exec resume <SESSION_ID> <PROMPT>`. Forbid `--last`, alternate sessions,
  automatic scientific retry, and automatic reuse of a one-attempt authority.
- Define the delivery guarantee precisely: one local wake launch is attempted in
  the normal terminal path; after a crash in the external-acceptance/write-receipt
  gap, recovery records ambiguity and makes no second attempt. Strict remote
  exactly-once delivery is not claimed without a receiver idempotency/ACK
  mechanism.
- Require an exact supervisor spec and authorization review after all current
  repository/PanNuke gates pass and after a separate state-changing decision.
  The absence of a current `jobs/` directory and real spec is intentional. No
  primary, confirmatory, recovery, publication, lifecycle, or M9 action is
  authorized by this decision.
- Preserve `SPEC.md`, frozen `PRE_REGISTRATION.md`, both frozen configs, raw
  PanNuke, and existing runs unchanged. Formal completion remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-29 - Accept the PID-proof correction; retain every state-changing gate

- Treat the integrated **1,698 passed / 1 failed / 1 skip** result as useful
  fail-closed defect discovery, not qualification. Accept only the narrow
  correction that rejects equality of verifier and controller process IDs in
  canonical fresh diagnostics; no verifier, scientific, or governance condition
  is weakened.
- Accept the final same-snapshot evidence: **1,699 passed with one expected
  Windows skip**, complete Ruff and format gates, mypy over 89 source files, all
  365 replacement-v2 tests, 59 amendment/lifecycle tests, full real-PanNuke
  validation, protected-integrity verification, exact artifact/lock/process
  absence, public Authority-C verification, and stable controller SHA-256
  `db1e07cb4c8e5e4d1dbfef5ab3f2b5e0a815c09a4ddbcfcbd268fbcc9c76c679`.
- Accept `qualification_required` as the only current live state. It authorizes
  no write by itself. Terminal Q, input-v3, authorization-v2, A2/S2/F2,
  Authority D, lifecycle, a supervisor job, and science remain absent and
  unauthorized until a separate explicit state-changing decision.
- Define the residual post-D/pre-S2 crash state as terminal ambiguity:
  A2+D without S2/F2 must be preserved, never deleted or repaired, never receive
  a synthesized S2/F2, and never be automatically retried. Locks and repeated
  readbacks close cooperating interleavings but cannot make the cross-component
  callback and S2 publication one atomic write.
- Keep the qualified external supervisor outside project execution source and
  unarmed. Arming requires, in order: green repository and PanNuke evidence; a
  separately authorized legal long operation; exact readback of the pinned
  supervisor release/manifest/runtime/wrapper/session; one reviewed canonical
  job authorization with `max_attempt_count=1` and retry false; exact argv and
  working directory; expected terminal seal and integrity receipt; exit-0
  verifier; timeouts and log ceilings; and one reviewed job spec. Startup remains
  recovery-only and may never select `run` or relaunch science.
- Supervisor qualification and replacement-v2 read-only qualification change no
  completion stage: neither performed a protocol publication, lifecycle,
  scientific analysis, completion attestation, review package, or expert
  validation. Preserve all frozen, authority, raw-data, and run evidence. Formal
  completion remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-29 - Accept the sole durable Q and the verified Windows launcher boundary

- Accept the user's explicit authorization for exactly one durable terminal
  qualification Q write. Treat the two preceding exit-1 processes as
  pre-transaction launcher diagnostics, not Q writes: both failed at the first
  process-quiescence probe before locks, `O_EXCL`, or publication because the
  Windows venv redirector and real interpreter exposed two matching PIDs.
- Accept only the qualified base-interpreter launch with the venv executable
  retained as argv0. The canary proves one PID while retaining the exact venv
  prefix, executable identity, environment, arguments, and code. This removes a
  benign launcher duplicate and does not exclude a real competing process or
  weaken any fail-closed check.
- Accept Q SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`
  as the sole durable terminal qualification. Independent classifier and public
  lineage verification both pass, including empty process/lock evidence and
  exact state `input_freeze_required`.
- Continue only through the closed ordered protocol: one I3 freeze and
  independent readback; one U2 authorization and independent readback; then one
  A2 publication transaction. Preserve and stop on any nonzero, mismatch,
  `STOP_AMBIGUOUS`, or exact A2+F2/no-D rollback. Never retry A2 or synthesize a
  terminal marker.
- Q is governance evidence only. It performs no science, publishes no Authority
  D, and changes no formal completion stage. Status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-29 - Accept the independently reconstructed input-v3 carrier

- Accept the one-shot I3 result only after its fresh 230-second classifier
  reconstruction and separate file-level audit. The four-file carrier is
  canonical, regular, non-reparse, single-link, and has exact records root
  `70d74cfa98e22e97d52c3342a88f795796e9b16a5a08324a904f34b2dd970bbd`.
- Accept `authorization_required` as the only legal post-I3 state. Q and I3 are
  fixed; U2/A2/S2/F2 and Authority D remain absent. The I3 command is consumed
  and must never be repeated or repaired.
- Authorize the next ordered operation under the user's broad explicit
  permission: exactly one U2 creation. Require its two independent preflights,
  locked second preflight, O_EXCL receipt, third full live readback,
  `max_attempt_count=1`, and a fresh independent `ready` classifier before A2.
  A nonzero, mismatch, partial state, or ambiguity consumes the U2 operation and
  stops without retry.
- I3 freezes technical outcome-blind inputs only. It performs no training,
  scientific analysis, Authority-D publication, or lifecycle transition.
  Formal status remains `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-29 - Consume failed U2 and require a separately governed successor

- Classify the single U2 operation as consumed even though it failed before
  physical publication. This follows the already recorded decision that any
  nonzero U2 stops without retry. The broad user authorization does not weaken
  that one-shot scientific-governance boundary.
- Accept the exact deterministic cause and hash:
  the production builder omitted `parent_authority_directory` from the
  publication contract required by the real canonicalizer; the resulting
  `ControlError` hashes to
  `0e5cec346272d35f96b3a60cfdcc3194ac3ec5cbb525a2d1b609fc4b642862c1`.
  Independent in-memory reproduction and NTFS absence evidence prove the defect
  occurs before U2 `O_EXCL`.
- Forbid same-U2 retry, runtime monkeypatching, editing the current pinned
  controller, retroactive U2 creation, or a synthetic failure marker. Preserve
  Q and I3 byte-for-byte. Their current live classifier remains useful
  historical evidence but cannot be the authority for a changed execution
  source.
- Authorize outcome-blind implementation and testing only of a new,
  separately-versioned successor protocol in an isolated staging overlay. It
  must bind the consumed-U2 failure/absence, current Q/I3, old controller hash,
  protected files, Authority C and run state; use a new closed namespace; and
  execute the real production builder through the real canonicalizer in tests.
  No new terminal receipt, live source integration, source freeze,
  authorization, A/S/F, Authority D, lifecycle, supervisor job, or science is
  authorized until staging gates pass and a separate live-transition decision
  is recorded.
- This is a technical governance failure, not a scientific result. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-29 - Keep the sole Q consumed and close the external successor roots

- Treat the existing terminal qualification
  `resource_authority_d_replacement_v1_terminal_qualification_v1.json` as the
  sole durable Q authorized for the replacement-v2 protocol. A repeated user
  authorization does not reset the consumed one-shot boundary and must not
  create a second Q. Reverification is read-only and remains permitted.
- Require the successor authorization helper to accept only the exact fixed
  external roots under `%LOCALAPPDATA%\AANCA-control-plane`: `releases`,
  `governance`, and `external-one-use-authorization`. Reject a content-addressed
  but caller-selected decoy parent before any claim or write.
- Because the exact authorization-bound technical amendment proves that the
  required historical NTFS interval wrapped, make
  `complete_usn_corroboration` categorically ineligible for this chain. R3 may
  use only the exact `unavailable_due_journal_wrap` limitation branch with
  `qualification_relies_on_usn=false` and the complete compensating-evidence
  contract.
- Keep the normalized AST proof required by the staged R3 design. Represent its
  four fixed spans once through the canonical
  `deterministic_prewrite_proof` ordered-read-set role; retain exact three-field
  records for physical immutable files and separate closed roles for
  reproductions, fresh observations, locked-final evidence, NTFS limitation,
  and compensating evidence.
- Do not permit caller-supplied R3 cores, authorization hashes, release records,
  amendment selections, or relaxed verifier flags on any production entry
  point. The public qualifier derives evidence only from the fixed project and
  sealed external control plane. It must compute the final core while the same
  publication locks are held and publish at most one fail-closed terminal leaf.
- I4 and schema-v4 must consume the exact amendment embedded in the independently
  verified external authorization and require byte identity with the fixed
  amendment and R3 copy. Legacy/skinned R3 or authorization payloads are
  negative STOP cases, not production compatibility modes.
- These decisions change governance implementation only. They do not modify
  `SPEC.md`, `PRE_REGISTRATION.md`, frozen scientific configuration, raw data,
  the existing run, or any scientific outcome.

## 2026-07-29 - Freeze the corrected post-U2 controller before any successor write

- Accept the isolated R3/U3 controller only at 314,853 bytes and SHA-256
  `96f9b89d5df2f2b5431cdce890c38adf01eeedfccd797fdd764e5765f95b58a0`,
  and its R3 observation helper only at 139,948 bytes and SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`.
  Any byte change reopens the full controller audit and staging gates.
- Require the downstream verifier lifecycle to be controller-owned and closed:
  `R3_ONLY` before I4 and `I4_SEALED` after I4. A pre-seal observer, a
  caller-selected lifecycle, a relaxed present-role set, or a missing
  under-lock sealed-record readback is STOP.
- Accept the current **363 passed** stable aggregate and **220 passed**
  controller-focused result as a staging checkpoint only. They authorize no
  external release, amendment, one-use claim, successor receipt, Authority D,
  lifecycle, supervisor job, or science until the independent P0 audit, exact
  I4 delegation, schema-v4 mirror, complete static gates, and real read-only
  canaries all pass against these exact bytes.
- Keep the external supervisor unarmed. Its stale next-run plan and
  non-durable top-level recovery error path must be corrected and covered by
  synthetic restart/corruption/wrapper tests before it can guard any real long
  process. Recovery remains diagnosis-only: it may never launch or retry a
  scientific operation.
- Preserve the sole Q, four-file I3, failed U2 evidence, `SPEC.md`, frozen
  `PRE_REGISTRATION.md`, scientific configuration, raw PanNuke, and all existing
  run artifacts unchanged. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-29 - Revoke the first controller checkpoint and accept only the corrected unarmed supervisor release

- Revoke staging eligibility of controller SHA-256
  `96f9b89d5df2f2b5431cdce890c38adf01eeedfccd797fdd764e5765f95b58a0`
  and all later moving hashes examined by the audit. Their passing functional
  tests do not override the demonstrated injected-prelock, lazy/re-baselinable
  callable, incomplete immediate-rescan, or semantic-default/private-helper
  mutation paths. None may be released, frozen into I4, or used for a live
  successor write.
- Require one new byte-stable controller checkpoint with eager pre-yield sealing
  of the exact callable keyset, no post-yield baseline creation, pinning of code,
  defaults, keyword defaults, closures, and transitive release-owned private
  callables/class methods, plus active nested runtime guards and complete
  under-lock ten-role/process/lock/lifecycle rescans. Require a fresh independent
  full audit, not a diff-only review.
- Accept the external supervisor only at release SHA-256
  `75b91e95fe253b8e5fe42e8488d41fa8fd7677891a82de1aeaeaad928e9031d8`,
  wrapper SHA-256
  `d0c503cff0d43d6960dfa32dd2085f91423014503aedbab7b0e4efc9dcb5126a`,
  Startup SHA-256
  `23fcc0ab12a03ae00313871092231715de91bc179a74acc30597a31c8212c7b7`,
  and manifest SHA-256
  `016739b52c5aa916ba4ad9f171d7a5af45d1a73d75f2a870a89e06c78c19a192`.
  The exact corrected bytes passed 68 synthetic tests, repository-config Ruff
  check/format, strict mypy, component/runtime readback, direct recovery, and
  installed-Startup recovery.
- This supervisor acceptance is infrastructure qualification only. It grants no
  job authorization, does not arm a long process, and does not permit retry.
  Startup is recovery-only; root-level STOP may wake the exact saved session
  once for diagnosis but may never launch science. Arming still requires all
  successor, repository, PanNuke, lifecycle, and capacity gates plus an exact
  one-attempt run spec.
- Preserve Q, I3, failed U2, frozen project files, raw data, and run artifacts.
  No scientific outcome was read or changed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-29 - Accept the semantic core only and require one final integrated release

- Accept controller SHA-256
  `39816d2d4598afd7a2fdb66821ce827096027422435b2df2341dfc04ee352b4d`
  and reproducer SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`
  only as staging evidence that the exact-byte loader, transitive semantic
  surface, closure-held release/runtime vaults, internal pre-lock derivation,
  immediate successor rescans, and closed R3 read set pass. Do not publish or
  bind this controller as the final release because the real I4-to-controller
  lifecycle call graph lacks its required runtime-owned wrapper.
- Accept the corrected I4 integration-generator loader at SHA-256
  `8c957bb8b40a2a072f6448eb2840d4d4ea3e2eff2f0a8a2af65de4b1922d540c`
  as a staging checkpoint only. It executes the exact verified byte buffer and
  passed independent adversarial tests, but its four production source pins
  must remain `None` until the final integrated controller and reproducer are
  independently frozen.
- Require schema-v4 to protect code, defaults, keyword defaults, closures,
  transitive module-owned helpers, and authoritative baseline maps with the
  same fail-closed semantic policy as the controller. A frozen dataclass
  containing mutable dictionaries is not an immutable seal. Any adapter
  default/baseline mutation must stop before a publisher call or durable file.
- Replace the expiring calendar-date amendment path with exactly
  `%LOCALAPPDATA%\AANCA-control-plane\governance\r3_usn_wrap_technical_amendment_v1.json`.
  Keep the dated record in canonical `recorded_at_utc`, bind its exact
  path/size/hash/payload into the one-use authorization, require amendment time
  not later than authorization time, and never compare a sealed amendment with
  a later wall-clock date. Production CLI exposes no caller-controlled clock.
- Require a separately saved, content-addressed bootstrap publisher and an
  independently implemented read-only verifier before the first external
  control-plane mutation. The repeatable preflight is zero-write. The execution
  has one O_EXCL claim and no adoption, cleanup, repair, resume, or retry; any
  partial state is terminal STOP evidence. Root ownership/DACL, reparse/ADS,
  case aliases, exact process/lock absence, input hashes, and fresh-process
  readbacks are mandatory.
- Consolidate the lifecycle wrappers, schema factories, CLI adapter/committed
  verifier, amendment chronology, and bootstrap into one final integration
  checkpoint before repeating the complete independent audit. This closes the
  known integration gaps without authorizing open-ended threat-scope expansion.
  It changes only technical governance implementation, not `SPEC.md`, frozen
  `PRE_REGISTRATION.md`, scientific configurations, raw PanNuke, existing runs,
  hypotheses, estimands, thresholds, exclusions, or outcomes.

## 2026-07-29 - Require the complete 37-lock R3 union before any one-shot write

- Revoke release eligibility of controller SHA-256
  `8620c7f01f38b6848c684b10f9fe48de6ce6cf0e736c15be593d1279b4d40f17`
  and the staging aggregate that exercised it. Its mutation-capable R3 path
  acquired 27 unique protocol paths but its own fail-closed verifier required
  37, so a production R3 qualification could not legally reach publication.
  Passing tests that omitted the production lock topology do not override this
  deterministic defect.
- Define the closed production R3 lock inventory as the exact union of the
  verified I3 legacy-scoped component (16), verified v2 bundle component (12),
  successor bundle component (13), and Authority-C parent bundle component (2).
  The legacy and v2 components overlap in exactly 6 paths, producing
  **16 + 12 - 6 + 13 + 2 = 37 unique paths**. No caller-selected path, missing
  member, extra member, duplicate, or normalized/case alias is permitted.
- Require exact path text to be retained alongside normalized comparison keys.
  On Windows, `Path` equality is case-insensitive and therefore cannot by itself
  prove rejection of a case-only alias. Every topology derivation and every
  owned-lock scan must fail closed on differing exact text before any publisher
  call.
- Require a production-shaped positive canary that reaches the C3 write
  boundary with exactly 37 owned paths, plus independent 36-path, duplicate,
  extra, case-only-alias, and post-acquisition-topology-change canaries that
  prove zero publisher calls and zero durable leaves. Re-derive and check the
  topology under the same held lock at all required pre-C3/pre-R3/post-seal
  scans.
- Do not rerun the successful full PanNuke validator merely because its local
  convenience wrapper could not call the .NET 6-only static
  `SHA256.HashData` API. The exact command and live process command line were
  independently captured and hashed with the PowerShell-5.1-compatible SHA-256
  API, the process exited 0, stderr was empty, the scan covered all 7,901
  patches, and publication was idempotent. Repeating an unchanged passing
  mandatory gate only to obtain different logging metadata would violate the
  bounded-execution rule.
- Preserve the user's authorization as exactly one replacement-v2 operation.
  No external root or one-shot claim has been created, so it remains
  unconsumed. Broad permission to continue does not authorize retry after a
  claim, weaken STOP semantics, or permit publication from an unfrozen
  controller. These decisions affect technical governance only; formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`, M8 remains open at 8/10, and M9
  remains locked.

## 2026-07-29 - Accept controller a667 and its exact four-pin I4 binding

- Accept only controller SHA-256
  `a6677cc32fa23fcd09639cdc3dfd38a6ad98e647f6ce79d94eed25cbbe270919`
  at 379,960 bytes with reproducer SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`
  at 139,948 bytes as the qualified post-U2 controller checkpoint. Its
  independent 14-file snapshot and exact 37-path set were stable before/after,
  and all missing/duplicate/extra/alias/topology-change cases stopped before
  publication.
- Accept I4 SHA-256
  `25b721a87d364b5ab9a664cea2ae04799ab131be3ceb73baa07d93951bf4e3cd`
  at 188,584 bytes and its test SHA-256
  `899ab9d03a8671625b88ae136113802ae551c0480c9a5fde15f07882eacea11c`
  at 127,673 bytes. The rebind consists only of the controller size/hash pin
  and the two matching test expectations; the reproducer pins did not change.
  Exact reversal reconstructs the superseded I4 bytes, and the independent
  audit passed the complete I4, mismatch, loader, lifecycle, root-binding,
  Ruff/format/mypy, and compile gates.
- Define the canonical lint context for isolated staging as
  `cwd=<staging-root>`, relative staging package/test paths, the absolute live
  repository `pyproject.toml`, and `--no-cache`. This preserves the real
  first-party package topology. A Ruff invocation from an unrelated working
  directory with absolute staging paths is diagnostic only because it
  reclassifies the local `controller` import. Final repository lint remains a
  separate `cwd=<project-root>` gate and also passed with `--no-cache`.
- Revoke every earlier controller/I4 checkpoint as a release candidate. Do not
  alter the accepted a667 controller or 25b721 I4 while fixing bootstrap.
  Any future byte change reopens their independent audit. Bootstrap,
  combined-staging, immutable external preflight, and one-shot publication
  gates remain mandatory; this decision alone authorizes no live write.

## 2026-07-29 - Accept the isolated bootstrap and frozen 31-file release baseline

- Accept only the bootstrap publisher at 83,509 bytes/SHA-256
  `feab2a751a3118e5f5ec438648f160f64bddb60ef3bbd9a39be349b3fc9cd938`
  and independent verifier at 58,804 bytes/SHA-256
  `34380beeebec12e057704d222bc9250c6ded40bbc5da8345ca36df41190d83b2`.
  Their independently stable ten-file audit root is
  `e04d2e5a698048b560ea1e9f8edc34f176a475a347c7dde4314aebdf1997c184`.
  Any byte change revokes this qualification and requires a new independent
  audit before provisioning.
- Accept the 31-file isolated successor staging root
  `074b1ae5d6df74675e9cc0afe67657367ab7e11d56bbe0e57091a4524a31cdf4`
  as the sole release baseline. Its canonical live-config gate returned
  717 passed, Ruff/format PASS, strict mypy PASS, and in-memory compile PASS on
  unchanged bytes. Staging caches are not members of this baseline and must
  never be copied into an external release.
- Require production bootstrap self-execution under exact `-I -S -B` flags and
  require the independent verifier to attest its own path, requested/effective
  runtimes, flags, closed child environment, bounded I/O, fixed working
  directory, and immutable evidence inventory. Preserve the accepted P2 that
  Windows PowerShell may update OS-managed `StartupProfileData` outside
  governed roots; it does not relax any AANCA evidence, control, release,
  amendment, authorization, or scientific boundary.
- Separate provisioning from publication. Provisioning may create only the
  fixed protected content-addressed evidence directories and exact read-only
  publisher/verifier/plan bytes. The first public production action is one
  zero-governed-write `--preflight`. Only a fresh independent PASS on its exit,
  output, hashes, absences, locks, and process state permits exactly one
  `--execute-once`. A failure, ambiguity, missing receipt, drift, or partial
  state is STOP without retry, repair, adoption, cleanup, Q write, outcome
  reading, or scientific execution.
- These decisions change only technical governance. Formal project status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at 8/10, and M9
  remains locked.
- The fresh public Q verifier passed after the append with marker
  `Q_POST_BOOTSTRAP_DOC_APPEND_PASS`, exact size 21,274 bytes, exact SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and exact status `qualified_rolled_back_failure_no_retry`. This is read-only
  continuity evidence, not a second Q authorization or write.

## 2026-07-29 - Consume the bootstrap one-shot and permit only the closed R3 next step

- Accept the production provisioner only at 51,512 bytes/SHA-256
  `47220e778aaaa4828cd310ec12ebf2f4a87ddf805b7705740a321d79e1aebdf4`
  and the canonical plan only at 14,435 bytes/SHA-256
  `36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e`.
  Its sole invocation created only the exact protected publisher, verifier, and
  plan evidence. It did not invoke the publisher workflow or scientific code.
- Accept the public production preflight only as zero-write evidence. Its
  independently stable before/after snapshot was
  `e86d8ed9f5db29c58e61efa5f8cdc14777bb7fe3d016d662cf519406512b2923`.
  It did not consume the bootstrap claim and did not authorize R3 by itself.
- Record the successful `--execute-once` as the sole consumed bootstrap
  publication attempt. Its immutable claim and success exist, its STOP receipt
  is absent, and two independent read-only audits reconstructed evidence root
  `928a9f1bb5c9031da3241f81356ff8c556433b77261e486666cbe7832d75f86e`
  and stable terminal snapshot
  `62f7d424a4a8d5bce98cfab441d76cd1b0928a1ec7ed3e889d81a9562ed81ca6`.
  Never rerun `--execute-once`; any future ambiguity is STOP without retry,
  adoption, repair, cleanup, overwrite, or replacement.
- Accept only the released six-file control plane at content root
  `3f5f0f417012ab2b5c291dc1fd322ba492a187fe295bfc4dc14954e296f24501`,
  release-record SHA-256
  `15a2e3bdb59e9da8fc631b246b132413704f688680cee7a153d741093baf039c`,
  amendment SHA-256
  `ac793dac868c6677667aebbc1461d0600c9d05df7831d6bf3dcb73f32d75de4b`,
  and external-authorization SHA-256
  `1a9d229e483aef1f90912ea38de37baec190866df89a36fd54743272ad6c84fb`.
  Any byte, path, chronology, ownership, lock, process, or semantic-pin drift
  invalidates the next operation and requires STOP.
- Interpret the external authorization narrowly: it grants at most one closed
  attempt each for R3, I4, and U3, in that order and only after each preceding
  terminal readback qualifies. It does not authorize Authority-D publication,
  lifecycle execution, training, confirmatory analysis, automatic retry, or
  outcome-value interpretation.
- The next mutation may only be the exact released-controller
  `--qualify-failed-u2` R3 operation. It must independently reconstruct the
  deterministic prewrite U2 failure and the
  `unavailable_due_journal_wrap` compensating-evidence branch while the exact
  37 locks are held. Any mismatch, partial write, ambiguous receipt, or nonzero
  exit is STOP; do not continue to I4.
- Preserve the sole historical Q unchanged and consumed. Bootstrap performed
  no Q write and creates no authority for a second Q. Preserve `SPEC.md`,
  frozen `PRE_REGISTRATION.md`, scientific configurations, raw PanNuke, and
  all existing runs. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at 8/10, and M9 remains locked.

## 2026-07-29 - Treat the unsealed R3 stop as consumed ambiguous state and require a new versioned chain

- Record the sole invocation of the released `--qualify-failed-u2` operation
  as an ambiguous consumed fail-closed attempt. The externally captured
  process exited 1 with
  `status=stopped_without_write`, `automatic_retry_allowed=false`, stdout
  SHA-256
  `bfd49b3af042de7266a16042c50bf5c12edf572d26dd8c9e1e0bb6e46d7b82e9`,
  and error SHA-256
  `0392acec55bf2c078551f205c420a5586957014022354f75c007302841895a04`.
  No governed R3 receipt, attempt/failure seal, or sealed stdout/stderr exists.
  The process-level failure and pre-write ordering are proven, but the missing
  terminal seal requires the conservative ambiguous classification. Never
  rerun the same command or reuse its external authorization, even though the
  failure occurred before a durable R3 claim or output.
- Accept the independently reconstructed error preimage as exactly
  `ControlError: released authorization_helper semantic callable/default/closure surface changed`.
  The immutable source proves that this guard runs while opening the external
  control plane, before the R3 branch and its first write boundary. Therefore
  the stop is a control-plane implementation failure, not a scientific result,
  U2 reclassification, Authority-D publication, or partial R3 completion.
- Freeze the released content root
  `3f5f0f417012ab2b5c291dc1fd322ba492a187fe295bfc4dc14954e296f24501`
  as historical evidence. Do not hot-patch, overwrite, repair, adopt, or
  reinterpret it. I4 and U3 remain prohibited because their required
  qualifying R3 receipt does not exist.
- Permit only reversible development and testing of a separately versioned
  successor outside every live/frozen namespace. Its semantic sealing must be
  stable across legitimate execution of nested Python code while continuing
  to fail closed on actual code, binding, default, closure, runtime, or module
  drift. A real six-module authorization-path test and independent byte-level
  audit are mandatory.
- No existing one-shot authority permits publication or execution of that
  successor. A future live action requires a genuinely new versioned
  namespace, technical amendment, immutable authorization, release, and
  one-attempt limit. It must be explicitly authorized and must not be labelled
  a retry, resume, repair, or adoption of R3.
- This decision affects technical governance only. It changes no frozen
  scientific definition and no study result. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at 8/10, and M9 remains locked.

## 2026-07-30 - Reject semantic-guard v6 and retain protocol v4 only as an audit candidate

- Reject semantic-guard specimen v6 root
  `f38cfdfc11cd7c04a99f898743b829af3a7e3b0057d258b4299577522fb3223a`
  for release, live publication, or authority use. Preserve its exact
  read-only bytes as negative qualification evidence; do not patch,
  overwrite, adopt, or reinterpret it.
- Require a separately rooted v7 to eliminate the manifest re-pinning race and
  bind the real runtime boundary before target execution. It must also close
  the recorded process-cleanup, inherited-handle, exact-argv, canonical-JSON,
  repeated-read, complete-role, process-image/runtime-closure, optimization,
  and ACL-boundary gaps. Passing v6 tests cannot substitute for a fresh
  independent v7 audit.
- Retain protocol-v4 root
  `b95b2a6aa57bfa7e764e477a50e3b782579601f808090d628ff512d4259fb9d3`
  only as a frozen static-audit candidate. Its 78 working-tree tests, 78
  snapshot tests, Ruff, format, strict mypy, and compile results authorize no
  state-changing operation. Independent static audit remains mandatory.
- Preserve `EXTERNAL_AUTHORITY_REQUIRED` as an unconditional protocol state.
  No local helper, broad prior permission, consumed Q/R3 authority, or test
  fixture may mint or substitute the required future trusted-host event.
- Semantic-guard primitives and a pure protocol schema are insufficient for a
  live gate. A later distinct-root runtime must separately bind and qualify the
  exact wrapper, publisher, publication verifier, candidate verifier,
  trusted-event verifier, runtime closure, plan hash, argv, environment,
  locks, process identities, attempt-consumption boundary, atomic commit, and
  fresh terminal full-bundle readback.
- These decisions affect technical governance only. They do not alter
  `SPEC.md`, frozen `PRE_REGISTRATION.md`, scientific configurations, raw
  PanNuke, existing runs, hypotheses, estimands, or outcomes. Formal
  completion remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject protocol v4 for promotion and require protocol v5

- Preserve protocol-v4 root
  `b95b2a6aa57bfa7e764e477a50e3b782579601f808090d628ff512d4259fb9d3`
  as a safe-blocked, read-only negative audit artifact. Its hard-false
  readiness and unconditional runtime blockers prevent current publication,
  but that is not sufficient for future promotion.
- Require a separately rooted protocol v5 to carry complete reconstructible
  raw QA before/after/error material, require nonempty rooted and
  parent-complete tree inventories, enforce unique critical-program logic
  roots and module identities, bind raw monitor observations to the claimed
  monotonic interval, and reject Win32 superscript device aliases.
- Do not weaken or remove any of the nine unconditional runtime blockers while
  making those schema fixes. Protocol v5 must remain
  `EXTERNAL_AUTHORITY_REQUIRED`, pure, and incapable of live publication.
- The post-documentation public-Q verifier readback remains exact at 21,274
  bytes and SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  The first diagnostic wrapper's later `KeyError` concerned only an incorrect
  print expression after successful verification; the corrected fresh
  wrapper exited 0. No Q write, retry, replacement, or authority consumption
  occurred.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Resolve the protocol-v5 split audit conservatively

- Preserve protocol-v5 root
  `85bd6018030513daa83215fb44a827a60d1070b2894708c69bbd672e44cfbaab`
  unchanged as safe-blocked evidence. Its 104 tests and static gates do not
  authorize promotion or a live action.
- Where the two independent auditors disagree about whether structural-record
  completeness belongs to the pure schema or only to the future verifier,
  adopt the stricter interpretation. Integrity of an arbitrary submitted
  subset is not evidence that the complete
  `deterministic_structural_code_record_v2` surface was retained.
- Require a separately rooted protocol v6 to bind one exact full semantic
  surface, exact verifier stdout bytes, stable before/after key universes, and
  mutation-class-relevant changed paths. Preserve all hard-false readiness and
  external-authority blockers.
- This conservative choice changes no frozen scientific definition, data,
  run, authority, or outcome. Formal status remains
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject protocol v6 and require provenance-bound protocol v7

- Preserve protocol-v6 root
  `b4a0f48fff4867cb6e8d69f7fbf287659d9f863ec6fe69cee64846aed95f26a8`
  unchanged as read-only, safe-blocked negative evidence. Its exact three-file
  inventory is 275,561 bytes and passed 114 focused tests plus Ruff,
  format-check, strict mypy, and compilation before freezing.
- Adopt the common conservative verdict of both independent static auditors:
  protocol v6 is **NOT QUALIFIED for promotion**. Exact canonical encoding and
  stdout-byte equality do not prove that a semantic surface was derived from
  the pinned candidate bytes, and a section-wide `/functions` predicate does
  not prove that the leaf changed by a mutation matches its declared class.
- Require a separately rooted protocol v7 to bind a closed canonical QA stdout
  document containing the exact ordered six role entries, each role's
  path/size/SHA-256 readback, structural policy, full surface bytes, and
  derived surface root. Require capture and all eight mutation records to equal
  their corresponding stdout entries. Replace broad section predicates with
  exact leaf-token predicates for code/nested-code, defaults, keyword defaults,
  closure, source bytes, and identity/binding mutation classes.
- Require coherent adversarial tests for a substituted full surface under
  recomputed hashes and for a wrong field inside the otherwise permitted
  section. A fresh immutable snapshot and a new independent static audit are
  mandatory; v6 may not be patched or reinterpreted.
- Retain every hard-false runtime blocker and
  `EXTERNAL_AUTHORITY_REQUIRED`. This decision grants no live write,
  publication, scientific process, retry, or trusted authority. It changes no
  frozen scientific definition, data, run, or outcome. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-30 - Reject runtime v1 and adopt a bounded trusted-host runtime v2

- Reject post-fix runtime-v1 aggregate
  `a85ff808b36d1fa8c3ba97e119d7a009383e051d68f08588c7f1bed578ff7f08`.
  Preserve it as negative working evidence. Passing synthetic lifecycle tests
  do not override its reparse/source-execution P0 defects or its terminal,
  singleton, scan, output-bound, and process-evidence P1 defects.
- Continue only in a separately rooted runtime v2. Require exact fixed
  publisher/verifier argv, a compile-time synthetic jail, a project-wide
  singleton and durable global attempt namespace, held and verified program
  source bytes delivered through a site-free private bootstrap, bounded
  streaming stdout/stderr, process-tree termination and wait, exact terminal
  topology and schemas, and a fresh full readback after the final receipt.
- State the security boundary honestly. Runtime v2 protects against accidental
  drift, ordinary concurrent starts, stale/PID-reused processes, cache/path
  substitution, partial state, and unintended retry on the identified trusted
  Windows host. It must use read-only/owner-DACL/no-reparse/no-ADS evidence and
  held handles. It does not claim protection against an administrator, kernel
  compromise, pre-existing writable process mappings, or a malicious
  same-token owner that can change its own DACL. Residual risks outside that
  boundary must be documented, not represented as proven security.
- Reject the experimental `NULL RootDirectory` rename primitive after its
  synthetic path resolved relative to process CWD. Use no handle-relative
  mutation primitive until its semantics are independently proven. A hard-false
  commit is preferred to an ambiguous write.
- Treat the interrupted protocol-v7 memory-growth run only as resource-guard
  evidence. Require one canonical outer stdout document and exact size/SHA-256
  crosslinks; never duplicate the full raw document in each observation.
  Require a bounded size/memory canary before the next full suite.
- These decisions authorize no live mutation, authority, supervisor job,
  training, recovery, confirmatory run, or publication. They change no
  scientific configuration, data, outcome, `SPEC.md`, or frozen
  `PRE_REGISTRATION.md`. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject semantic guard v7 and require process-tree-safe v8

- Preserve exact semantic-guard-v7 root
  `abda274e236ee83093d7bdb4716a3aefe8bd4de72b00bac4d0286cdab1452326`
  unchanged as read-only negative qualification evidence. Its 226-test
  combined gate and semantic manifest/controller improvements do not override
  two independent **NOT QUALIFIED / STOP** verdicts.
- Retain the declared trusted-host boundary: the component does not claim
  protection from administrator/kernel compromise, runtime-install mutation,
  pre-existing writable mappings, or malicious same-token process-memory
  injection. Do not manufacture a P0 by silently expanding that model.
  Nevertheless, exact-closure, truthful-receipt, bounded-wait, and
  process-tree-terminal guarantees must hold within the trusted-host model.
- Require a separately rooted v8 to:
  - remove/reject the nonexistent `python312.zip` entry before mutable imports
    and bind both original and sanitized `sys.path`;
  - retain exact share-denying handles for the complete pre-verification
    source/native import closure and bind absence of alternate ZIP/PYD/PYC
    loaders;
  - bind manifest creation-time volume/file ID/topology rather than first
    observing identity only after a later open;
  - run the outer harness through a source-only, cache-safe bootstrap;
  - create each child suspended, assign it to a non-breakaway
    `KILL_ON_JOB_CLOSE` Job Object, and resume only after assignment;
  - kill and prove the entire job empty before releasing protected handles;
  - use bounded stdin, stdout, stderr, and drain deadlines;
  - validate exact typed finite timeout/no-retry, actual child
    argv/environment/cwd/image/parent, exact inherited-handle inventory, and
    job policy.
- Require adversarial grandchild-pipe/handle, output-flood, non-reading stdin,
  persistent handle-reset, concurrent native spawn, byte-identical manifest
  identity replacement, alternate-loader, and nonfinite-timeout tests.
- This is a technical-governance correction only. It changes no frozen
  scientific definition, data, run, outcome, `SPEC.md`, or
  `PRE_REGISTRATION.md`; it grants no authority, retry, publication, or
  scientific execution. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Prefer the unchanged original confirmatory over the non-completing sensitivity run

- Amend only the operational order in `PLAN.md`. Do not change `SPEC.md`,
  frozen `PRE_REGISTRATION.md`, the frozen confirmatory config, or any
  scientific definition.
- Do not spend the next 11-16 hours on
  `resource_bounded_confirmatory_v1` as a prerequisite. It is permanently
  `amended_or_exploratory`, has `completion_stage=null`, and cannot unlock M9.
  Retain it as an optional sensitivity path.
- Once the current technical chain qualifies, target the unchanged original
  108-cell confirmatory study. Permit development of an operational
  checkpoint-successor mode that preserves the exact cells, seeds, folds,
  models, estimands, exclusions, and statistics while adding only explicit,
  independently verified resume mechanics.
- Require a new run and exact `retry_of_run_id`, an explicitly named immutable
  predecessor, physical allowlisted checkpoint copies, full training-state
  validation, no outcome-bearing predecessor reads, no overwrite/link/adoption,
  and no automatic retry. A failure requires a new exact plan and authority.
- Use the event-driven supervisor for every long attempt. This decision does
  not arm it and does not authorize science. The future one-use authority must
  bind the exact final source, plan, command, runtime, expected terminal
  artifacts, verifier, and supervisor hashes.
- This course is selected to make real milestone progress: only a sealed,
  registry-backed, stage-eligible original confirmatory can emit
  `CONFIRMATORY_COMPLETE` and unlock M9. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject runtime-v2 aggregate 2cc1 and keep protocol-v7 resource bounded

- Reject runtime-v2 aggregate
  `2cc1a3767fc4524e0e83b2b0c9071dcc14ba7c131dcdaacdd2cc39efc71a4480`
  despite 75 passing component tests. A passing test set cannot override the
  independent proof of partial inheritable-handle rollback, nonterminal
  whole-Job cleanup, incomplete child-observed process/handle/job evidence,
  incomplete parent-to-child source provenance, or incomplete pending-tree
  terminal evidence.
- Require one consolidated repair and formatting cycle, cache-free gates, and a
  fresh independent audit on a new exact aggregate. Preserve the rejected
  aggregate only as working evidence; it authorizes no publication, authority,
  supervisor job, or scientific execution.
- Keep atomic commit fail-closed until a separate same-parent, source-handle
  bound, no-replace Windows rename primitive passes synthetic adversarial tests
  and independent review. A deliberate hard block is safe but cannot qualify a
  functional publisher.
- Treat protocol-v7's 41,593,333-byte terminal observation as a genuine failed
  size acceptance gate. Do not run the broad suite until the production graph
  actually resolves its content-addressed QA reference and every persisted
  governance/terminal document is below 16 MiB with canonicalization peak
  below 512 MiB.
- For the future original-confirmatory successor, require the full closed
  180-checkpoint expectation set rather than discovery of present files.
  Permit only two explicit predecessor classes: a valid
  `sealed_failed_demoted` run, or a separately diagnosed
  `unsealed_interrupted_orphan` with a gone exact process instance, zero locks,
  zero success/completion/stage artifacts, stable content-addressed checkpoint
  inventory, a technical amendment, and a new exact one-use authority.
  Supervisor restart handling may only write STOP and wake Codex; it may never
  select the orphan class or relaunch science automatically.
- Both predecessor classes remain read-only and require physical
  no-overwrite copies into a new `retry_of_run_id` successor. Parsing
  predecessor OOF predictions, metrics, rankings, reports, or outcome values
  for selection or tuning remains forbidden.
- These decisions change no frozen scientific definition and do not consume
  the future real-attempt authority. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject v8/runtime-v2 and reserve the one-use Q replacement for the final envelope

- Reject semantic-guard-v8 root
  `6fbd4e561ecdc279c10603e24cf70b5bab32c40b8c8ca958f652f41892ad31d1`
  and runtime-v2 aggregate
  `c6751f8a1c4f1d438f610ce53e460b9ae4d38a6b5cb643e4550e8cd34a7ee185`
  for promotion. Their passing focused tests do not override the independent
  P0/P1 findings concerning incomplete source closure, nonterminal Job
  cleanup, non-absolute deadlines, ambiguous handle-close readback,
  incomplete persisted STOP evidence, opaque access policy, partial
  pre-mutation revalidation, or the red mypy gate. Preserve both candidates
  unchanged as negative qualification evidence and repair only in separately
  rooted v9/runtime-v3 trees.
- Do not promote protocol-v7 merely from its **194/194** local test result or
  memory evidence. Bind promotion to the current exact candidate root
  `d078badcc1c9e78054a8f54489734ee745d42e748b9b99bded3e260534e9c768`
  and a fresh static audit proving exact six-role derivation, one-time QA
  evaluation, terminal chronology, leaf-specific mutation predicates, and
  closed size/memory limits.
- Treat frozen original-confirmatory resume root
  `0a0a40250143aee0e6fb4dc0ff20b76985d0fae5c2b3b6e59068518993f05979`
  as a candidate allowlist only. Its 180-checkpoint contract, physical
  no-overwrite copies, per-fold no-fallback directives, and one-use execution
  authority must pass independent source and live-compatibility review before
  integration. A checkpoint that was authorized for resume may never silently
  fall back to fresh training if it disappears or fails validation.
- Replace the impossible single-current-source confirmatory gate with a
  dual-evidence gate. Historical primary dependency evidence remains verified
  against the exact sealed preregistration and primary roots without comparing
  them to current execution source. A separate outcome-blind Q technical
  amendment must be a direct child of the historical preregistration
  authority, bind the new execution-only source delta, prove frozen science
  unchanged, list no affected hypotheses, and be independently reviewed.
  Resource-bounded authority C is not the parent for this unchanged original
  confirmatory path.
- Accept the user's authorization for exactly one future `Q replacement-v2`
  write and one independent verification, but do not consume it while any
  exact byte, hash, gate, or review is provisional. Never modify or retry the
  frozen historical public Q. The one-use replacement must bind the final
  source root, PLAN root, literal source-only loader, exact argv,
  `sys.orig_argv`, `GetCommandLineW`, interpreter, runtime bundle, static
  environment, separately attested dynamic supervisor nonce, supervisor
  release and exact saved Codex session, expected terminal receipt, exact
  108/90/18 cells, 36 CNN cells, 180 fits, three rotations, checkpoint
  allowlist, attempt count one, and no automatic retry.
- Keep the event-driven supervisor unarmed until that complete envelope and
  its external verifier are frozen. The supervisor may wait without Codex
  tokens and wake the exact saved session once after a terminal result, but it
  may never infer success, retry science, reuse authority, or start a second
  primary/confirmatory/recovery/publication process.
- These decisions change no `SPEC.md`, frozen `PRE_REGISTRATION.md`, scientific
  configuration, data, outcome, run, registry, or existing authority. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**,
  and M9 remains locked.

## 2026-07-30 - Require fresh first execution and durable one-shot Q publication

- The next original-confirmatory attempt must be `fresh`. `PLAN.md` records
  that the original 108-cell study has never run, and read-only disk inventory
  found no real confirmatory checkpoint. Require
  `execution_mode="fresh"`, `retry_of_run_id=null`, no predecessor read or
  copy, and exact predeclared absence for all 180 CNN-fold checkpoint paths.
  Preserve `successor_resume` as a disjoint future mode requiring a real
  failed/interrupted predecessor, at least one valid checkpoint, physical
  allowlisted copies, and a new one-use authority.
- Reject dual-authority root
  `87503f863626dfebdbb5f08e1fe9a6e1a9ec8d1b3437cff3e8df7848b6ae61ea`.
  Require a new root with a canonical pre-Q intent envelope and recomputed
  intent hash, a separately parsed closed review receipt bound to that exact
  intent and an independent reviewer process/source/runtime, mandatory typed E
  in the production gate and lifecycle readiness, exact Q and program/source
  bundle bindings in E, and final stable recapture of source/config/PLAN.
- A one-shot check over only valid surviving Q directories is insufficient.
  Before writing any Q byte, atomically persist one immutable attempt/intent
  receipt in a singleton namespace bound directly to P. Success must cross-link
  the exact Q; every failed, partial, corrupt, linked, missing, crash-left, or
  ambiguous attempt must persist STOP and permanently forbid a second Q under
  the authorization. Do not delete the evidence that proves consumption.
- Reject terminal-receipt-v1. Its declaration-only hashes cannot attest
  success, its non-null retry lineage cannot represent the required fresh
  attempt, and it still binds rejected runtime-v2. Require terminal-receipt-v2
  plus a distinct outcome-blind reader/verifier that opens and re-reads actual
  terminal files, identities, hashes, seal/integrity roots, registry and stage
  records, matrix reconciliation, statistics, and restoration. Externally pin
  every material section while holding pins fixed under adversarial receipt
  mutations.
- Reject protocol-v7 root
  `d078badcc1c9e78054a8f54489734ee745d42e748b9b99bded3e260534e9c768`.
  Individual single-link claims do not prove cross-record identity
  disjointness. Require a complete uniqueness set for all bundle files,
  terminal manifest/claim/root identities, and directory identities, with
  coherent alias tests, before another exact-root audit.
- Keep the frozen successor-resume module as a future-mode candidate while a
  separate isolated runner core implements both closed execution modes. The
  live image OOF path must consume immutable per-fold directives and may never
  infer resume from `checkpoint_path.is_file()`: disappearance or corruption
  of an authorized checkpoint is fatal, a terminal checkpoint performs zero
  training, and an incomplete checkpoint continues only at its exact next
  epoch.
- The user's exactly-one Q replacement-v2 authorization remains reserved and
  unconsumed until every final source/PLAN/command/runtime/environment/
  supervisor/terminal byte and independent review is frozen. These decisions
  authorize no live mutation, Q/E write, supervisor job, retry, or scientific
  process and change no frozen science, data, outcome, or existing authority.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject resume-v1 and require lineage-derived directives

- Reject frozen successor-resume-v1 despite its exact read-only inventory and
  earlier focused tests. A dead PID tuple is not predecessor lineage; bind the
  orphan PID/create-time to exact launch, status, supervisor, job, run, and
  process evidence, and reject unrelated or reused PIDs.
- A resume authority and supervisor nonce must be durably consumed before any
  successor RunTracker or checkpoint-copy write. Require exact current child
  PID/create-time/parent-supervisor identity and one immutable launch intent.
  Missing supervisor identity, nonce replay, concurrent entry, or crash-left
  consumption produces STOP and never retries.
- Do not trust a caller-recomputed directive-list hash. Derive each of the 180
  fold actions from the immutable predecessor snapshot and physical copy
  receipt, then compare the data-plane directive byte-for-byte. An imported
  checkpoint can never become `fresh`; disappearance, corruption, wrong
  action, or wrong copy evidence is fatal.
- Replace opaque Q adapter pins with typed evidence from the independently
  verified future Q replacement-v2 plus mandatory one-use E. Production code
  must not expose a raw-pins route around Q/P/source/PLAN/config/runtime/
  supervisor/mode/lineage validation.
- The original live resume boolean and existence test are rejected. Require
  one closed immutable fresh/successor contract covering exactly 36 CNN cells
  by five folds. A dedicated checkpoint-contract exception must fail the whole
  run rather than being recorded as a recoverable cell failure. Terminal
  restore performs zero optimizer steps and remains byte-identical; incomplete
  resume starts at the exact next epoch.
- Accept only provisional local progress from the new isolated roots:
  dual-authority **15/15**, terminal-receipt **32/32**, runner migration
  **27/27**, runtime loader **4/4** and contract **37/37**, and semantic guard
  real target 8.54 seconds. These are focused engineering gates, not promotion,
  Q/E authority, readiness, or scientific execution.
- Protocol root
  `9a8be13f8ae06c2eb54bfe54006cbb789ba7f4c73ed816433cdbb4fab444ce03`
  is held at **0 P0 / 0 P1** but remains unqualified until its test-only
  module-identity P2 proves that the repaired canonical/raw lane policy is
  actually reached. Passing tests that fail in an earlier validator do not
  provide regression evidence.
- These decisions change no frozen scientific definition, data, result, run,
  registry, existing authority, or supervisor state. The next real
  confirmatory mode remains `fresh`; successor resume is a future contingency
  requiring separate authority. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Accept protocol-v7 as frozen external evidence

- Accept protocol-v7 only at exact three-file root
  `e701b8c362a033887e85bd74db6b2ae9bead9b59091b6eb473f2b30e677a0a33`
  under
  `C:\Users\NATAN\Documents\AANCA_semantic_guard_successor_bootstrap_frozen_20260730_e701b8c3`.
  Its fresh independent verdict is **0 P0, 0 P1, 0 material P2** and the
  mechanical freeze preserved every audited byte.
- Treat any content, path, inventory, identity, stream, link-count, or
  read-only drift as a new unqualified candidate. This acceptance grants no
  execution authority, readiness, publication, Q/E write, or scientific
  result.
- Continue to require separate qualification of semantic-guard-v9,
  runtime-v3, dual-authority-v2, terminal-receipt-v2,
  successor-resume-v2, and runner-core before integration. The runner's
  current 48/48 focused gate is engineering evidence only.
- Preserve `fresh` with null retry lineage as the only mode for the next real
  original confirmatory attempt. Protocol qualification does not authorize
  fabricating a predecessor or weakening the closed future
  `successor_resume` mode.
- The user's exactly-one Q replacement-v2 authorization remains reserved and
  unconsumed. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject terminal-receipt-v2 and require v3

- Reject terminal-receipt-v2 root
  `fc139e8d08cf14b6a494bedcf98f96fcb9b1c63394c0a0d80afcca58892c0d61`.
  Passing 43 focused tests does not overcome the independently reproduced
  semantic-hook spoof, self-derived full-receipt pin, ignored active locks,
  pre-seal stage attestation, Windows alias collision, and stale-positive
  disposition defects.
- Terminal-receipt-v3 must invoke and authenticate exact production semantic
  readers without a caller-injected qualifying route. Every material section
  and the full canonical receipt must be pinned by a distinct external
  producer/reviewer before validation.
- Require actual lock enumeration, positive stage attestation strictly after
  immutable seal, current disposition/stage eligibility with no later
  withdrawal, canonical physical identity separation, ancestor ADS/reparse
  and retained-identity checks, read-only terminal evidence, strict schemas
  for predecessor/status/completion documents, and independent supervisor
  job/attempt/no-retry/exit/terminal cross-links.
- Accept runtime-v3 root
  `acf1d9799d74137db56675c6cf132f43b192761de43a7b2b8c53eba0873cfd92`
  only as a candidate entering independent audit. Its 101-test and static
  gates are not promotion or execution authority.
- Do not consume Q replacement-v2, create E, arm the supervisor, or start
  confirmatory science until terminal-v3 and every other final component have
  independently closed all P0/P1 findings. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Require one canonical Q with per-attempt E and preflight

- The one future Q replacement-v2 must be a stable policy/source/protocol/
  runtime/terminal authority for both closed modes. It must not encode a
  synthetic predecessor or require reconstruction after a future interrupted
  attempt. Every actual `fresh` or `successor_resume` launch instead requires
  a new one-use E bound to its exact mode, lineage, 180 directives, command,
  supervisor launch, preflight, and terminal contract.
- Q/E must bind and independently read back the complete Windows command
  envelope: full argv, `sys.orig_argv`, `GetCommandLineW`, exact cwd,
  requested/effective interpreter, loader, environment, supervisor command,
  exact saved session, launch intent/nonce, current child creation time,
  parent supervisor identity, and Job identity. Self-derived or merely
  syntactic hashes are rejected.
- Require full recomputation and cross-linking of every ordered source record,
  program root, private runtime bundle, semantic/runtime/protocol root, and
  expected terminal-receipt/seal/integrity verifier. A different supervisor
  job specification cannot satisfy the authority.
- Before E consumption, require a fresh typed original-confirmatory preflight
  receipt for the exact 108/90/18-cell, 36-CNN, 180-fit, three-rotation plan,
  five caches, CUDA/AMP, RAM, exact volume, and conservative sealed-plan peak
  plus exactly 10 GiB. The non-claiming resource-bounded receipt is not a
  substitute.
- Repair M9 before it can unlock: require the qualifying positive post-seal
  stage ledger and current disposition, reconcile the canonical real run
  name, and enforce exactly 100 top / 100 random / seed 707. No package may
  reach `EXTERNAL_VALIDATION_READY` through arbitrary CLI counts or an
  unqualified completion JSON.
- These rules consume no authority and authorize no execution. The next real
  attempt remains `fresh` with null retry lineage; future resume is a separate
  E under the same qualified Q and requires an actual predecessor. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**,
  and M9 remains locked.

## 2026-07-30 - Reject runtime-v3 and require a held runtime-v4

- Reject runtime-v3 root
  `acf1d9799d74137db56675c6cf132f43b192761de43a7b2b8c53eba0873cfd92`.
  Its 101 passing tests do not overcome independent P0 reproductions for
  import-before-runtime-verification, forged loader/child evidence, and
  self-attesting terminal success hashes.
- Runtime-v4 must authenticate and retain handles/file identities for the
  complete interpreter, stdlib, PYD/DLL, and native dependency closure before
  any application module imports or executes. Later path-based hashing cannot
  establish this precondition.
- Require exact supervisor, initial-loader, launcher, child, parent, creation
  time, executable, Job, argv/`sys.orig_argv`/`GetCommandLineW`, cwd,
  environment, stdout/stderr, and parsed-output evidence from held OS
  identities. Caller-supplied observed fields are not qualifying evidence.
- Terminal commitment must independently open and recompute every
  claim/payload/manifest/private/commit/final file and semantic cross-link.
  Syntax-valid hashes or a matching tree root alone are insufficient.
- Require an exact handle/object inventory with alias/duplicate and unexpected
  handle rejection, exact nonempty wait chronology under one absolute
  deadline, and one global singleton over every public operational entry.
- Runtime-v4 must also pass Ruff/format under the repository's actual
  configuration, configured and strict mypy, compile, and adversarial tests
  before another audit. This decision grants no execution or authority.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Require active M9 authority and forbid automatic readiness claims

- A detached positive confirmatory receipt is insufficient for tracked M9
  execution. The original-label audit must hold the active
  primary -> confirmatory -> original-audit authority chain for its entire
  invocation and seal. Missing, withdrawn, mismatched, or concurrently
  invalidated stage authority is fatal.
- Bind the exact current positive `CONFIRMATORY_COMPLETE` post-seal
  attestation record hash into the tracked original-audit resolved config,
  sealed eligibility evidence, external eligibility readback, and review
  candidate metadata. The production run identity is
  `pannuke_confirmatory_study`; the frozen scientific config identifier is
  unchanged.
- For a stage-eligible blinded cohort, freeze exactly 100 top-ranked items,
  100 disjoint random items, and seed 707. Direct builder and CLI paths must
  reject every other count/seed. Smaller fixtures remain explicitly
  non-stage and non-claiming.
- Building and structurally validating a package does not itself establish
  `EXTERNAL_VALIDATION_READY`. Until a tracked no-overwrite package run,
  independent technical inspection, integrity readback, and positive
  post-seal attestation all exist, the package candidate must retain
  `completion_stage=null`.
- Treat M9 guard root
  `073b00a0fcdafaf9317b6551427c0ccb2e78a572491dc8b5b43c2b6610e678ce`
  only as a candidate entering independent audit despite its 102-test/static
  gates. Treat semantic-v9 root `c37ad83b...` and terminal-v3 root
  `dd08e9d...` identically until their independent verdicts return.
- These decisions change no result, source annotation, frozen science,
  authority, or completion stage. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject terminal-v3 and bind the full launch command/environment

- Reject unchanged terminal-v3 root
  `dd08e9d67c4f6e8677584e814b4ceac755f4e9e6f2d7b1e6992705ddaaf9b80c`
  after an independent audit reproduced five P0 and five P1 defect classes.
  Passing focused tests cannot override runtime-v3 binding, injectable
  semantic functions, self-derived pins, incomplete terminal roles,
  declaration-only successor evidence, or unretained mutable filesystem
  identities.
- Terminal-v4 must depend on qualified runtime-v4 and semantic-v9 only. It
  must independently persist/read back pins, recompute every predecessor,
  copy, terminal, stage, integrity, and disposition relationship, keep
  retained Windows identities, require read-only closed schemas, and hold the
  final eligibility/lock guard through atomic publication.
- Define canonical JSON for launch authority as UTF-8 without BOM or newline,
  sorted keys, compact separators, `ensure_ascii=False`, and
  `allow_nan=False`. Hash only closed typed objects.
- Define `exact_command_sha256` as the hash of the complete expected launch
  envelope: exact supervisor/loader/child argv, expected `sys.orig_argv`,
  expected `sys.argv`, exact expected native `GetCommandLineW`, exact cwd, and
  requested/effective interpreter identities. A separate child-observed
  envelope must be independently collected and match it exactly.
- Pass two complete explicit sanitized environment mappings: one for the
  supervisor and one for the scientific child. Normalize names to uppercase,
  reject case collisions, NUL, equals-sign names, and all extra keys. Bind
  separate hashes for both mappings into the launch root; do not treat an
  inherited subset as exact environment evidence.
- This decision authorizes implementation and testing only. It consumes no Q
  or E, grants no retry, and does not authorize confirmatory execution.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject semantic-v9 and declaration-only preflight evidence

- Reject semantic-v9 root
  `c37ad83ba77c12331aa484f0211ffa3d426d1933ce835966450227580037840b`
  despite all 79 focused and 209 protocol tests passing. A Job handle may
  never be duplicated into the scientific child: the parent must retain the
  sole Job authority so parent death deterministically terminates the child
  before a singleton can be reacquired.
- On an ambiguous handle-close failure, retain and quarantine every affected
  authority object. Do not zero fields, drop the child object, or release the
  singleton until current OS state is independently requalified. Every
  constructor must roll back all already-created handles on every exception.
- Reject the reviewed preflight source
  `1915d8ddd7921b9e734da9432db73751421d9caf4f7084cf68648e01ca78aace`.
  Hashing an opaque reservation is insufficient: invoke its canonical parser
  and cross-link every field. A test seam cannot publish a qualifying
  acceptance or bypass the parent E reservation.
- At child acceptance, independently observe the full command/environment,
  rebuild the complete preflight snapshot, and compare it with the consumed
  receipt. Hold no-follow parent/leaf identities through publication. Derive
  the exact 108/90/18, 36, 180, and three-rotation plan from the real sealed
  execution plan and production builder/config/cache inputs; independently
  read back a tracked rehearsal seal, manifest, and integrity receipt.
- These rejections consume no Q or E and grant no execution or retry.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Separate structural review packages from M9 stage authority

- Reject M9-guard-v1 root
  `073b00a0fcdafaf9317b6551427c0ccb2e78a572491dc8b5b43c2b6610e678ce`.
  A public builder or structural validator may never turn a caller-supplied
  mapping into verified stage authority, even when every supplied hash has
  valid syntax.
- The public package builder creates only a non-stage candidate unless a
  separate verifier independently opens and reconstructs the tracked
  confirmatory receipt, active stage/disposition chain, exact positive
  post-seal attestation, and sealed M9 inputs. A Python type or mapping supplied
  by the caller is not non-forgeable authority.
- Structural package validation reports structure and private-linkage
  consistency only. It cannot establish `study_outcome_eligible`. The stage
  verifier separately enforces exact schema-v2, 100 top, 100 disjoint random,
  seed 707, and all current upstream authority.
- A forged-evidence canary with syntactically correct hashes is mandatory, and
  the final exact candidate root must be made read-only before its independent
  audit. This decision executes no M9 work and changes no completion stage.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Require independent consumed-E readback and immutable checkpoints

- A structural Python Protocol, dataclass instance, or caller-supplied
  expected hash is not execution authority. The production runner receives
  only the exact consumed-E receipt path/root bound by the command envelope,
  independently invokes the authenticated canonical dual-Q/E verifier, and
  reconstructs all 180 directives before building an executable contract.
- Test adapters and fixtures are explicitly non-qualifying and cannot reach
  the grouped-OOF production entrypoint. The real CLI must perform the
  consumed-E binding before the first matrix, cache, model, or checkpoint
  action.
- Do not use a path-separated verify-then-`os.replace` checkpoint flow. Publish
  unique immutable, versioned checkpoints no-overwrite from held temporary
  identities, verify the final file ID, and use an append-only/no-overwrite
  manifest instead of a mutable `latest` checkpoint path.
- Every checkpoint-originated exception is fatal in both fresh and successor
  modes. E binds exact source and destination identities; validation preserves
  full resume identity and rejects ADS on both files and directories.
- This repair changes no scientific plan, result, authority, or completion
  stage and grants no execution. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Reject reconstructible preflight authority

- A mapping, boolean `verified` field, dataclass instance, public typed binding,
  or caller-supplied hash is never a consumed-E or child-launch capability.
  Production runner entry must occur inside a retained no-follow capability
  context created by the exact canonical gate from the current process and
  exact O_EXCL artifacts.
- Consumption must create a persistent no-overwrite replay marker bound to the
  acceptance receipt, process identity, command/environment roots, Q/E,
  reservation, and preflight roots. An in-memory flag alone is insufficient.
- Every production claim requires the exact canonical parent reservation.
  There is no null or test-only route to a qualifying acceptance.
- Do not resolve critical authority, parser, or OS-observer behavior from
  replaceable `sys.modules` entries. Load and verify the exact held/sealed
  source and runtime identity required by Q, or fail closed.
- A manifest filename, immutable marker, or SHA-shaped value is not semantic
  evidence. Invoke authenticated canonical readers and independently
  reconstruct every artifact, seal, and integrity relationship.
- Reject preserved preflight-v1 root
  `2e368d1142460c11d9d4c76099e216af14f4a12f6f1b1af308318e44845c580b`.
  All repairs occur in a separately audited v2 root and grant no execution.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Close launch lineage, nonce, and post-run evidence unions

- Execution lineage is a closed discriminated union. `fresh` requires null
  retry lineage, predecessor-binding hash, and resume-adapter hash.
  `successor_resume` requires a real predecessor plus both exact 64-hex hashes.
  Mixed states are fatal.
- The next qualified supervisor release generates launch nonces with
  `secrets.token_hex(32)`: exactly 64 lowercase hexadecimal characters.
  Legacy 32-character UUID nonces, uppercase values, reuse, restart replay, or
  mismatched cross-links are rejected. Do not weaken the authority schema for
  compatibility with an old unqualified release.
- Child execution acceptance SHA/file identity is observed after its O_EXCL
  write and is never predeclared inside its own command/E preimage. The held
  child boundary retains the exact acceptance identity through runner entry
  and publishes a persistent O_EXCL runner-consumption marker.
- After `WaitForExit`, a separate exact authenticated validator independently
  opens and recomputes the persistent child acceptance, E acceptance,
  runner-consumption marker, Q/E/reservation/preflight, command/environment,
  process, exit, and terminal chain. It does not require a live PID/FD and a
  caller-supplied token is never sufficient authority.
- These contracts authorize implementation/testing only. No Q/E, supervisor
  launch, scientific execution, or completion-stage transition is granted.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Preserve semantic-sidecar cache authority in M9

- A frozen cache-provenance record is a closed exclusive union: either direct
  `cache_file_sha256` authority or `sidecar_semantic_sha256` authority, never
  both and never neither. Do not normalize one branch into the other.
- The real confirmatory records use the semantic-sidecar branch. M9 must invoke
  the canonical `verify_frozen_cache_sidecar` reader, independently verify the
  live cache and sidecar, and compare the resulting semantic checksum to the
  frozen record. A null direct cache SHA is required in that branch and is not
  missing evidence.
- Tests for M9 eligibility must exercise the actual frozen branch. Fixtures
  that replace it with a direct SHA and null semantic sidecar cannot establish
  scientific eligibility. Add inverse, both, neither, cache tamper, sidecar
  tamper, and stale-semantic-checksum regressions.
- Reject M9-guard-v2 root
  `6ae9e06a135c2887f414b11819a6aa9f91de758e34f6606be7b118636af6106d`;
  preserve it read-only and repair only a separately audited v3 root. This
  changes no frozen config, result, authority, or completion stage. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**,
  and M9 remains locked.

## 2026-07-30 - Require held identities through every runner handoff

- A pre-read `lstat` followed by path-based hashing is not a stable handoff,
  even if size and bytes match. Open once no-follow, derive full identity and
  hash from that descriptor, verify the same handle before/after reading, and
  confirm the path still names the held file before adoption.
- Consumers use retained bytes/handles rather than reopening a pathname.
  Same-byte file-ID replacement is fatal and receives explicit canaries at
  normalization, completed-cell persistence, and post-fit evidence boundaries.
- All immutable runner artifacts, including NPZ, use no-overwrite publication
  from held temporary identities. Remove remaining `os.replace` paths where
  the scientific contract requires that the destination did not already
  exist.
- These requirements change no model, matrix, frozen plan, authority, or
  result. They grant no execution. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Preserve an independently readable guard through ambiguous close

- A positive Windows handle value is not proof that the handle remains valid
  after a close wrapper raises: the native close may already have succeeded.
  Quarantine evidence must distinguish before-native-close from
  after-native-close ambiguity and must not infer validity from a nonzero
  integer.
- Before releasing the last Job or singleton authority guard, retain a
  separately readable authority handle and complete a staged independent
  readback. Every close stage, including both final guard closes, requires
  adversarial canaries for exceptions before and after the native close.
- Reject semantic-v10 root
  `a2eade224ff1ff15b2c76947c5020eafdccec1b23383ea9e04c8772bf6d5018c`.
  Preserve it byte-identically and repair only a separately rooted v11
  candidate. A self-reported pass is insufficient; v11 requires a fresh
  read-only audit after all full gates.

## 2026-07-30 - Treat the user's Q-v2 permission as one future unconsumed write

- The user authorizes exactly one future Q replacement-v2 publication and one
  independent verification. The authorization is not consumed by recording
  it, by isolated implementation, by testing, by a dry run, or by read-only
  verification of the historical Q.
- Do not perform the Q write until the complete exact input set is
  independently qualified and bound. A failed premature write would consume
  the only authorized attempt without advancing the study.
- The user's broad permission to keep implementation and QA moving does not
  create an unbounded retry policy and does not replace the later exact
  one-use E/run authority for a specific final source, command, environment,
  runtime, supervisor release, capacity receipt, and scientific plan.
- No authority, result, or completion stage changes here. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## 2026-07-30 - Bind private bootstrap execution without a self-hash

- Never place the expected whole-file SHA inside the file whose SHA it is
  expected to equal. The preflight source identity is external, mandatory, and
  Q-bound through a canonical nested contract.
- The qualifying bootstrap release is an ordered five-role closure: complete
  literal-loader bytes, preflight source, authority source, runtime-observer
  source, and runner-core source. The deterministic loader opens and retains
  exact no-follow physical identities, compiles the pinned bytes into private
  hash-derived module names, and does not qualify canonical package imports.
- The persistent preflight marker records all four executed source modules.
  Consumed E directly binds the Q pre-run-contract root. Top-level Q/E schemas
  need not change, but the nested contract, marker, and authenticated readers
  must agree exactly.
- A function-object identity is not a semantic seal. Qualifying capture must
  reject code/default/keyword-default/closure/referenced-global mutation,
  module namespace additions/deletions/rebinding, and mutable runner API
  substitution before and after execution. Explicitly stateful registries may
  change contents only while their cell and content-object identities remain
  fixed.
- Runner entry is one-use and token-only. No raw contract, lease, callback, or
  caller-supplied runner may escape or be rearmed after garbage collection.
  This approved design direction grants no Q/E or live execution.

## 2026-07-30 - Anchor terminal pins in a distinct supervised process

- A canonical JSON receipt that names a PID does not prove that the named
  process created it. The terminal reviewer/pinner must be a distinct process
  pre-bound by Q/E and launched through the qualified supervisor's
  integrity-verifier path after the scientific child exits.
- Q/E bind the exact verifier argv, cwd, interpreter, executable, source,
  environment, and output path. The supervisor terminal receipt binds the
  verifier PID and creation time, native executable/source/command/environment,
  exit code, and the no-overwrite pin artifact's physical identity and SHA.
- Public terminal verification independently recomputes that complete chain
  and invokes the full physical dual-Q/E post-run validator before reading any
  semantic or outcome artifact. A validator-root constant or in-memory token
  alone is never execution evidence.

## 2026-07-30 - Expand runner completion scope rather than emit an unsealable schema

- The checkpoint runner may not emit schema-v2 versioned checkpoints while
  downstream completion continues to validate only the legacy final path.
  Expand the isolated runner allowlist to include
  `confirmatory_completion.py` and its filesystem-completion tests.
- Final completion must physically verify the final checkpoint, execution
  manifest, every versioned checkpoint, and every commit sidecar, including
  identity, link/reparse/ADS, canonical schema, and hash relationships.
  Deleting or replacing any historical version is fatal.
- The expanded allowlist is implementation authority only. It requires a fresh
  zero-P0/zero-P1 exact-root audit and does not authorize live integration,
  Q/E, training, retry, or a completion-stage transition.

## 2026-07-30 - Make the process the authority boundary, not mutable Python objects

- Stop attempting to prove trusted execution by recursively comparing Python
  functions, globals, defaults, closures, private tokens, or numeric handle
  slots inside the same mutable interpreter. A process capable of maliciously
  changing those objects can also change the in-process checker; increasingly
  elaborate self-checks do not create an independent authority boundary.
- Use one deterministic content-addressed execution capsule in a fresh
  isolated process. Q binds the entire capsule and its internal source
  manifest; E binds one exact command/environment/nonce/mode/lineage.
  Project-owned imports originate only from the capsule.
- The fresh capsule validates Q/E and immediately invokes a closed direct
  preflight/runner entry. It accepts no untrusted callback, plugin, import hook,
  raw contract, capability, or lease. Standard-library and pinned third-party
  code remain governed by the existing environment evidence.
- Windows Job lifecycle, singleton, process-tree termination, `WaitForExit`,
  and sleep prevention are owned only by the external supervisor. Do not nest
  a competing Python Job/handle state machine inside the scientific child.
- Terminal authority is independent because the supervisor launches the same
  capsule again in verifier mode after the scientific process exits. Its
  preterminal pin is composed with the later supervisor terminal receipt by a
  separate post-wake read-only verification.
- Preserve semantic-guard-v10, v11, and v12 as rejected evidence. Do not create
  v13. This architectural change grants no Q/E or execution and does not alter
  any scientific invariant or frozen file.

## 2026-07-30 - Require exact environments in supervisor-v2

- Preserve the current qualified external supervisor release and wrapper
  read-only. Build a separate v2 release rather than weakening the Q/E
  environment contract to fit v1.
- Reuse exactly the two existing Q/E mappings: `supervisor_environment`
  launches the external supervisor and excludes the attempt nonce;
  `child_environment` includes the sole exact 64-lowercase-hex nonce and is
  passed unchanged to both the scientific process and integrity verifier.
- Clear inherited variables. Reject missing, extra, lowercase, `=`, NUL, or
  Windows-casefold-colliding names. Bind canonical mapping hashes through the
  one-use authorization, launch intent, process-started evidence, verifier
  record, and terminal receipt; require independent observed readbacks.
- The integrity verifier must not generate a second UUID nonce. It produces a
  preterminal pin before `terminal_receipt.json` exists, so neither artifact
  may predeclare the other's future SHA.

## 2026-07-30 - Accept M9-guard-v3 as a qualified locked future input

- Accept the independently reproduced M9-guard-v3 root
  `f6040f057e0704f2d3fdfd436aa4bff6b647fabedac71de8274173966cbb154c`
  as zero-P0/zero-P1 technical evidence. Preserve its nine-file read-only
  candidate unchanged until a later mechanical integration.
- Qualification does not unlock M9. Do not integrate, execute, inspect, or use
  the guard to produce rankings until an original confirmatory run has
  independently satisfied every M8 terminal and stage-attestation requirement
  and legitimately reached `CONFIRMATORY_COMPLETE`.
- Preserve the semantic-sidecar exclusive-union behavior proven against the
  real frozen config. Do not replace its null direct cache SHA with a fixture
  shortcut or treat the sidecar branch as missing evidence.
- Withdraw the obsolete fixed 57/63 capsule-member count. The deterministic
  capsule source set is every regular `.py` below one exact final frozen
  `src/histo_audit/**` tree, mapped by stripping `src/`, plus the exact
  `__main__.py`, capsule policy, entry contract, and final internal manifest.
  Counts are derived from the frozen inventory. Runtime import tracing remains
  an acceptance proof, not a source-selection mechanism.
- Do not qualify or publish the capsule until the finite closed-key
  post-wake `verify-terminal` command and composed-receipt schema is frozen,
  implemented, and independently shown to fail closed. This is the sole open
  design-review P1 at this checkpoint.
- This decision consumes neither the authorized one future Q replacement-v2
  write nor the later one-use E. It changes no science, frozen file, authority,
  result, or completion stage. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-30 - Retain terminal composition inputs through the Codex wake

- Do not rely on a path hash or read-only attribute after the producing
  verifier/supervisor closes a file. Supervisor-v2 retains no-follow native
  handles for the preterminal pin, verifier stdout/stderr, and supervisor
  terminal receipt until the exact resumed Codex process and post-wake capsule
  verification finish.
- Retained completed-file handles permit read sharing only and deny write and
  delete. Each identity binds canonical path, volume/file ID, size, SHA-256,
  attributes, read-only state, link count one, and no alternate data streams.
  Q/E bind exact expected paths and policy; observed future identities are
  sealed in the supervisor terminal rather than predicted circularly.
- Create verifier stdout/stderr with parent read/write handles that are retained
  continuously. Pass only inheritable duplicates to the verifier. Authenticate
  the separately created preterminal pin against the secured stdout summary's
  full physical identity before retaining it.
- Create the supervisor terminal with create-new semantics and retain its
  underlying identity through composition. Retain no-delete directory handles
  for the supervisor root, jobs directory, and exact job directory so neither
  inputs nor the composed output pathname can be ancestor-swapped.
- If process death or Windows restart loses these handles before the composed
  receipt and wake complete, write `STOP` and wake once for diagnosis. Do not
  reconstruct success from unguarded files and do not retry science,
  verification, composition, publication, Q, or E.
- Preserve the frozen `models/cnn.py` bytes and provenance. Operational
  checkpoint hardening belongs outside that scientific module; never resolve a
  provenance mismatch by changing the frozen resource config or normalizing
  its expected hashes.
- Keep the existing `original_confirmatory_resume.py` canonical per-fold
  checkpoint allowlist unchanged. After each successful fold, copy the
  held-verified canonical checkpoint to a distinct versioned O_EXCL artifact
  and commit, then seal the canonical file read-only. Do not hardlink, replace,
  delete, autodiscover, or rewrite the resume contract merely to save disk.
  Count both copies in the sealed capacity calculation plus its required
  10 GiB margin.

## 2026-07-30 - Require overlapping pin custody and an acyclic live-handle receipt

- A preterminal pin must never pass through a close-then-reopen custody gap.
  Use supervisor-owned bounded stdout and ACK pipes: the verifier holds the
  create-new pin open, emits one canonical READY line, waits; the supervisor
  opens and validates that same physical identity, then ACKs; the verifier
  exits only after validating ACK. This is local blocking IPC, not Codex
  polling.
- Keep the frozen command suffix and exact environment unchanged. Stdin/stdout
  pipe semantics, bounded transcript, inherited-handle allowlist, timeout, and
  no-retry disposition are closed Q/E/supervisor contracts. Derive the exact
  empty stderr log path from the bound job directory rather than adding an
  argv flag.
- Freeze terminal dependencies as `P -> T -> L -> C`. T cannot contain its own
  final bytes hash. L is a deterministic after-terminal continuity receipt
  that records final identities/hashes and supervisor handle slots for
  P/stdout/stderr/T, plus only pre-serialization identity/slot facts about L
  itself. C independently duplicates those still-live handles, hashes their
  final bytes, and binds L's final hash.
- A live PID alone is insufficient. C verifies supervisor PID, creation time,
  boot identity, handle access/share policy, pathname equality, and every
  duplicated physical identity. Failure or loss before C is a terminal STOP,
  never reconstruction, adoption, or automatic retry.

## 2026-07-30 - Anchor bootstrap execution in Q/E before project imports

- Reject the self-consistent-only bootstrap snapshot identified by the fresh
  audit. Content addressing and a valid internal manifest are necessary but
  do not independently prove that the user-authorized project Q/E selected the
  capsule.
- Before importing any `histo_audit` module, sealed stdlib-only bootstrap code
  parses the exact common argv, held-reads the named E and exact project Q,
  verifies their file/core/root hashes, and closes the minimum outer schema.
  It binds the exact project root, canonical Q path, content-addressed capsule
  path/size/SHA, internal manifest, policy and entry contract, interpreter,
  PLAN, supervisor, command, environment, job, nonce, and lineage roots. The
  full authority reader reruns all checks before any scientific input or
  output action.
- Reset import state to an exact sealed builtin/frozen/zip/file finder set,
  clear prior importer caches/hooks, and require every project module and
  package search path to map to one exact capsule-manifest member. Check before
  dispatch, after dispatch, and in `finally`. A mutable checkout, editable
  finder, site package, or missing-member fallback is fatal.
- A production publisher performs two independent byte-identical builds, then
  publishes only to
  `PROJECT_ROOT/artifacts/execution_capsules/<sha256>/original_confirmatory.pyz`
  with native no-overwrite retained-handle readback. Arbitrary output paths are
  private tests only. A preexisting or partial production path is STOP and is
  never adopted, cleaned, or retried.

## 2026-07-30 - Extend terminal composition to one supervisor-readback receipt

- Freeze the live sequence as `P -> T -> L -> C -> R`. C acquires its actual
  composed output path with CREATE_NEW/read-only-at-create before reading
  P/T/L. Any empty or partial crash claim remains permanent evidence and
  blocks a second invocation.
- C duplicates its retained handle into the exact live supervisor and sends
  one bounded custody READY. While event-waiting on the exact Codex resume
  process, the already Q/E-bound supervisor verifies C's wake-tree membership,
  interpreter/capsule/argv/cwd/environment, remote handle, bytes, identity, and
  P/T/L roots. It writes one O_EXCL retained readback R and only then ACKs C.
  C emits its sole canonical stdout line and exits after validating that ACK.
- Do not add a fourth capsule mode or call `verify-terminal` twice. Do not read
  the overlap-receipt or wake-intent pathname directly in C; T carries final
  overlap evidence and the live supervisor proves the wake actor. Extra direct
  path fields are rejected.

## 2026-07-30 - Treat current user authorization as forward permission, not retry

- The single Q replacement-v2 publication and its single independent
  verification remain the only authorized Q write. They occur only after the
  complete final input set is independently qualified.
- Once those gates pass, no additional conversational confirmation is required
  to create and consume one exact E, arm one qualified event-driven
  supervisor, and run one unchanged original-confirmatory attempt. The
  supervisor wakes the exact saved Codex session once for terminal
  verification and the legal next step.
- This forward permission never authorizes a second Q, automatic E/science/
  verification/publication retry, overwrite, cleanup of a one-use claim,
  result-guided tuning, or bypass of `SPEC.md`, `PLAN.md`, frozen
  `PRE_REGISTRATION.md`, final-reference isolation, or group-safe OOF.

## 2026-07-30 - Make E command binding a single-file acyclic hash DAG

- Do not place a final command hash or complete argv inside E when that argv
  contains `--e-intent-sha256` for E's own final bytes. That creates an
  unsatisfiable cryptographic self-reference.
- Keep one canonical E authority file. Its finite closed schema binds exact
  per-mode command projections and one exact derivation policy, including the
  executable, content-addressed capsule, cwd, ordered non-self common fields,
  fixed suffix paths, environment contract, and lineage. It contains neither
  a wildcard nor an unconstrained argv fragment.
- Seal E once with native CREATE_NEW semantics. Only then derive final argv by
  inserting E's final file SHA-256 and core SHA-256 at the two fixed flag
  positions. The supervisor job spec records the exact final argv and its
  SHA-256 together with the direct E path/file/core binding.
- Before launch and after `WaitForExit`, independently reconstruct every
  command from Q/E and require byte-for-byte argv and command-hash equality
  with the retained supervisor spec. Q binds the derivation contract and
  supervisor release. No second E file, command override, extra argv/env,
  adoption, or automatic retry is allowed.

## 2026-07-30 - Amend unpublished Q-v2 with explicit publication custody

- Reject the unpublished 10-key Q-v2 draft because its writer and reader had
  close/reopen and pathname-trust windows. No live Q-v2 exists, so this does
  not amend or replace an executed authority.
- The first and only live Q-v2 schema version 2 has exactly 12 top-level keys:
  the prior closed set plus `publication_ancestor_lease` and
  `publication_ancestor_lease_root_sha256`. All authority, bootstrap, and
  independent-verifier implementations must match that exact set.
- The publication lease is anchored at the exact Q project root and covers the
  existing `project_root/artifacts/resource_control` chain with retained
  native no-delete handles. Publication uses one native CREATE_NEW leaf held
  through canonical write, flush, read-only/ADS/link/identity/root checks and
  overlapping independent verification. Do not create a missing ancestor,
  change permissions by pathname, close and reopen, clean a partial claim, or
  retry.
- A Q-bound interpreter under this project's `.venv` uses the exact retained
  ancestor chain `project_root -> .venv -> Scripts`, followed by the exact
  `python.exe` leaf. A self-selected suffix-only ancestor list is invalid.
  Supervisor evidence must prove actual retained handles for the full chain
  across every phase launch.
- Keep the E/spec graph acyclic: upstream E contains no final supervisor-spec
  hash. It binds the deterministic destination/schema/release. The downstream
  CREATE_NEW spec alone records E path/file/core, exact rederived argv, and
  command hashes and is revalidated before launch and after process wait.

## 2026-07-30 - Transfer Q/E custody from a short authority controller

- Reject supervisor self-publication of Q/E. It would combine authority
  creation with the long-lived execution actor and require duplicating or
  importing the Q/E authority implementation into the supervisor release.
- A short sealed authority controller performs the only authorized Q-v2
  CREATE_NEW publication, exactly one independent transition/readback with
  continuous leaf-and-ancestor custody, then the one-use E and downstream spec
  publication.
- The controller starts the exact supervisor-v2 and transfers Q leaf/full
  ancestors and E leaf/ancestors using `DuplicateHandle` plus one bounded
  anonymous-pipe READY/ACK protocol. The downstream spec is the exact carrier
  of dynamic process, remote-handle, Q/E file/core/root, access/share, and
  receipt facts; the existing Q supervisor-release contract hash and E
  spec-policy/path bindings keep the upstream DAG acyclic.
- Supervisor-v2 must validate physical identities, bytes, read-only/ADS/link
  constraints, controller and supervisor PID/creation/boot identities, and
  reconstruct the exact Q/E/command/spec seed before writing a retained custody
  receipt and ACK. Science is forbidden before ACK. Source leases close only
  after ACK; every partial handoff is permanent STOP evidence with no retry.

## 2026-07-30 - Bind two correct Python runtime identities, not one conflated path

- External Windows process identity APIs (`QueryFullProcessImageNameW` and CIM
  `ExecutablePath`) observe the venv launcher
  `C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe`, SHA-256
  `864530d708039551a2c672ddd65e5900fbc08b0981479679723a5b468f8082bc`.
- Inside that same process, `GetModuleFileNameW(NULL)`,
  `sys._base_executable`, and `sys.orig_argv[0]` identify the base CPython
  runtime
  `C:\Users\NATAN\AppData\Local\Programs\Python\Python312\python.exe`,
  SHA-256
  `15b41a488c356c0e331facdea6c836a6cec021f12d5fde9844e7ca4a1aa0361a`.
- Preserve both as separate retained dependencies. Compare each observation
  only with the identity defined for that API; treating either path as proof
  that the other is wrong is a fail-closed implementation bug.

## 2026-07-30 - Authorize successor copies before launch but materialize them after RunTracker

- Reject both precreating/adopting a successor run directory and extending
  `RunTracker` with an attach path. The canonical new-run no-overwrite
  invariant remains unchanged.
- A successor E is a pre-copy authority, not a claim about future filesystem
  identities. It binds the qualified predecessor snapshot, exact source
  checkpoint identities/hashes and state, exact allowed actions, deterministic
  destination relative paths, copy policy, and `retry_of_run_id`. It cannot
  contain a future destination file ID or future receipt hash.
- After E ACK, canonical `RunTracker.start` creates the absent directory. The
  sealed lifecycle then performs only the authorized O_EXCL copies, verifies
  non-aliasing/bytes/state, records actual destination identities and the copy
  receipt, and derives the registered execution contract. Exact equality to
  the E pre-copy authority is required before a fit.
- A partial copy is a failed run and permanent STOP evidence. Never clean it,
  adopt it, autodiscover it, fall back to fresh training, or retry it
  automatically.

## 2026-07-30 - Integrate only the gated fresh/successor runner bytes

- Accept the exact 19-file runner allowlist with root
  `6567f5f746090d5b9bbb230c1485baa2147276c179c83735ad2e923f2e7e8a88`
  after source and destination size/SHA-256 equality checks. Exclude the WIP
  documentation, data, caches, and temporary test state.
- Keep the public compatibility wrapper and sealed capsule entry on one
  internal full lifecycle. The acceptance proof is functional ordering at that
  shared boundary, not a brittle requirement that every wrapper directly
  contain every lifecycle call.
- Preserve the exact two-branch checkpoint contract. Fresh execution remains
  unchanged. A successor is authorized before launch, but only a new canonical
  `RunTracker.start` directory may receive the later O_EXCL physical copies and
  receipt. Never add an attach/adopt path to make the authority graph easier.
- Split Windows path observers by role. Delete sharing is forbidden for
  ordinary and live-writer checkpoint observation and is permitted only for
  the explicitly identified `DELETE_ON_CLOSE` owner-lock path.
- This integration is an implementation gate, not a scientific execution or a
  completion-stage transition. It consumes no Q/E authority and leaves formal
  status exactly `PRIMARY_STUDY_COMPLETE`.

## 2026-07-30 - Make protected terminal inspection a Q-bound control-only template

- Reject arbitrary `json_equals` selectors for every protected
  original-confirmatory supervisor job. A verifier that can be directed to
  read a metric, ranking, prediction, effect, p-value, statistics value, or
  restoration value cannot truthfully assert `outcome_values_read=false`.
- Freeze one exact ordered seven-role template for `.immutable.json`,
  `artifact_manifest.json`, `completion_evidence.json`, and the four
  registry/anchor files. Permit only the flat control keys `run_id`, `status`,
  `completion_stage`, and exact `study_outcome_eligible`; require strict JSON
  type equality and zero JSON decoding for empty-check registry/anchor rules.
- Do not import or hash-trust the earlier unqualified
  `original_confirmatory_terminal_prelaunch.py` scratch. Canonical authority
  owns and self-hashes the template directly.
- Preserve the acyclic authority graph: Q's closed nested
  `supervisor_release` binds the template root; E's closed nested job object
  binds the exact run-specific instance and root; the downstream spec must be
  canonically equal to E. Q remains 12 top-level fields, E remains 20
  top-level fields, and the scientific request remains 25 fields.
- Bootstrap and every later reader reject a template, instance, path, role,
  order, selector, value, or type mismatch before import, artifact access,
  claim arm/take, ACK, or scientific release. This operational hardening does
  not inspect an outcome or change any frozen scientific rule.

## 2026-07-30 - Accept the runner allowlist after independent read-only QA

- Accept the already integrated 19-file runner allowlist only after an
  independent main-tree run reproduced its exact manifest root and passed
  228 focused tests, Ruff, format checking, mypy, and compileall with no
  failures.
- Keep this as a component qualification, not the combined project gate. The
  latter remains blocked on frozen authority, terminal, capsule, and
  supervisor integration and must still include full pytest, full
  Ruff/format, a retained-output PanNuke validator, and the relevant functional
  CLI.
- Reject a supervisor command identity that is dynamically generated and then
  compared only with its own retained value. Q must bind an acyclic static
  derivation separating the externally observed venv launcher, the internal
  base runtime, the supervisor source path/hash, Python `sys.argv`, working
  directory, and a versioned canonical command preimage. Supervisor and
  terminal must independently reconstruct the concrete identity from Q plus
  the sealed E/spec and mutation-test every field.
- No authority was consumed and no scientific execution occurred. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`.

## 2026-07-30 - Use a sealed terminal-client launcher for exact child custody

- Reject direct launch of the protected terminal verifier by resumed Codex.
  The Codex/tool environment is not the E-sealed child environment and can
  legitimately contain session or shell-policy additions, so direct
  inheritance is fail-closed but not reliably live.
- Bind one external supervisor-release launcher in Q by canonical path,
  physical identity, size, SHA-256, isolated interpreter vector, source,
  working-directory rule, and versioned command derivation. E binds the exact
  per-job launcher and existing child command/environment/cwd projection.
  The downstream spec and wake intent bind the final instances.
- The launcher uses native O_EXCL evidence before one direct CreateProcessW
  child, constructs the exact sealed environment instead of forwarding its
  own, remains in the fresh wake Job, and waits event-driven for the child.
  It is not a new verifier mode and may not discover, retry, fall back, or
  launch an alternate command.
- Keep the graph acyclic by placing the final launcher command only in the
  post-spec/post-T wake intent. Never put the wake-intent hash into the
  launcher command. Prove live parent-launcher and child identities,
  parentage, PEB argv/cwd/environment, and same-Job membership.
- Split the resume-to-child-arrival bound (1,800,000 ms) from the
  post-CLAIM custody exchange (60,000 ms). A restart after loss of protected
  live custody creates permanent STOP evidence and permits at most one
  diagnosis wake only if no prior normal wake intent exists.
- Require full semantic validation of P, T, and their dual-evidence parity
  before FINAL_ACK; a dynamic self-comparison or roles/root-only check is not
  an independent attestation.

## 2026-07-30 - Distinguish the Windows venv redirector from its runtime child

- Supersede any earlier operational statement that one observed path alone
  identifies a venv-launched Python process. A Windows venv invocation has two
  live processes: the CreateProcess root is the venv redirector, while a
  separate base-runtime child executes Python code. Their image paths, PEB
  argv[0], PIDs, creation times, and roles must remain distinct.
- Launch the stdlib-only supervisor and terminal-client launcher directly with
  the separately bound base runtime and `-I -S -B`. This avoids transferring
  Q/E handles into a redirector rather than the actual receiver and removes an
  impossible PEB-equality claim.
- Continue launching the capsule through the bound venv interpreter because
  it needs the pinned venv environment. Treat its venv redirector as the
  immediate parent and the sealed terminal-client launcher as the grandparent.
  Require exact live identities, lineage, and same-Job membership for all
  three processes. Never collapse the redirector and executing runtime into
  one self-attested identity.

## 2026-07-30 - Accept the frozen authority interface after two audits

- Accept authority source SHA-256
  `6ed3c651a972c45cc057138e30555f23614ae994a20b01b77049cc061c3c0d23`
  only as a qualified component after its complete owner QA and two
  independent unchanged-byte audits each reported zero P0/P1/P2.
- Freeze its exact acyclic order:
  static Q release and derivations, closed E projection, downstream spec and
  T, then one rendered launcher command and wake intent. The CREATE_NEW
  launcher intent is unread in CLAIM, validated and retained by the supervisor
  before GRANT, duplicated/read by C only after GRANT, and revalidated through
  READY, readback, and FINAL_ACK.
- Keep this acceptance distinct from production authority publication.
  Component qualification does not consume the user's one Q-v2 write or its
  independent verification and does not authorize science until every
  downstream implementation and combined gate passes.

## 2026-07-30 - Accept the frozen terminal capsule after independent audit

- Accept terminal source SHA-256
  `768aba256cd007ebc6e13851db44f9eeeddfa9f8930bc5f9f58eb678cf530631`
  as a qualified component after complete owner QA and an independent
  zero-P0/zero-P1/zero-P2 unchanged-byte audit.
- Preserve the causal custody order: CLAIM advertises but does not read the
  future launch intent; the supervisor retains and validates it before GRANT;
  C duplicates and reads the retained source only after GRANT; READY, R, and
  FINAL_ACK bind and revalidate the same physical source and three-process
  lineage.
- Do not integrate or publish these bytes until the standalone launcher and
  supervisor demonstrate the same contracts in real synthetic Windows
  subprocess/Job/PEB tests. Static terminal qualification is necessary but is
  not a substitute for that E2E gate.

## 2026-07-30 - Source-anchor the exact production capsule in the launcher

- Classify a coherent E + supervisor-spec + CLI capsule substitution as P0.
  Caller-recomputed hashes and internally consistent self-roots are never
  independent authority, even when an upstream component previously validated
  an earlier instance of those files.
- Preserve frozen release112 and CLI14 schemas. Do not add a Q read to the
  sealed launcher allowlist and do not weaken the launcher's post-wake
  fail-closed responsibility.
- Break the dependency without a hash fixed point by building the deterministic
  capsule first. Its path is content-addressed by its whole-file SHA-256 and
  the capsule does not embed the standalone launcher source. Materialize that
  final path and hash as trusted literals in the launcher source only after two
  independent builds yield identical bytes.
- Require the final Q/supervisor-held launcher source to compare its literal
  anchor against every E/spec/command/capsule view and the live file before
  creating launch intent or a process. An unresolved template and a coherently
  rehashed wrong capsule both produce permanent STOP evidence with no retry.
- Treat template QA as preparatory only. Final launcher qualification, final
  supervisor qualification, and the saved-session synthetic E2E must use the
  materialized source bytes and occur before the one authorized Q-v2 write.

## 2026-07-31 - Separate template qualification from production authority

- Accept the exact launcher/materializer/supervisor bytes recorded in
  `STATUS.md` only as `TEMPLATE_READY`. Their synthetic and static contracts
  are complete, but unresolved production literals deliberately prevent use.
- Never attempt a hash fixed point in which launcher source embeds Q file,
  Q-root, or supervisor-release hashes while the same Q supervisor release
  embeds the launcher source hash. Self-consistent caller hashes are not
  authority and repeated rendering cannot solve that cycle.
- Replace the three cyclic literals with one acyclic post-Q proof rooted in
  information fixed before Q and held by the trusted controller/supervisor.
  The proof must bind the exact Q, E/job, terminal receipt, launcher command,
  and no-retry disposition before launch-intent publication or child creation.
  It requires independent mutation/replay/restart review before production.
- Keep `entry_contract.json` fail-closed until that authority proof and the
  complete synthetic E-consumption/terminal E2E pass. Ready-contract promotion
  is one controlled delta that also updates its checked-in readback test and
  `CAPSULE_DESIGN.md`; all affected roots must then be recomputed.

## 2026-07-31 - Isolate the bootstrap success fixture without weakening policy

- Preserve the production rule that any pre-imported `histo_audit` module is
  a terminal bootstrap error. A test process that intentionally collected
  other project tests is not a fresh capsule interpreter.
- For the sanitizer success fixture only, remove ambient `histo_audit` and
  `histo_audit.*` entries through pytest's restoring monkeypatch. Maintain a
  separate parametrized negative proving root and submodule pre-imports fail
  before `sys.path`, finders, hooks, or importer cache can change.

## 2026-07-31 - Keep the live source-delta regression exact after capsule integration

- Extend only the independent test's expected live delta with the seven
  confirmatory/capsule source files actually integrated after its previous
  snapshot. Keep every change kind explicit as `added`; never derive the
  expected mapping from the production observer.
- Do not alter the immutable 2026-07-27 parent source manifest, historical
  authority, or scientific definition to make the regression pass. The test
  must continue to fail on an unknown, missing, removed, or wrong-kind path.
- Treat the production-entry/control-only-handler E test as a composition
  gate, not as scientific execution and not as a substitute for the real
  terminal/supervisor custody tests. Its purpose is to prove one unchanged E
  and CREATE_NEW claim cross all three capsule modes without any run tree or
  scientific input.

## 2026-07-31 - Accept physical custody and promote ready without granting authority

- Accept the final external supervisor and launcher templates only after their
  real Windows process/pipe/Job/handle rehearsals, byte-stable QA, and
  independent **P0=0/P1=0/P2=0** audits. Both protected ACK boundaries must
  continue to route exclusively through Q/E semantic and physical
  revalidation; no raw ACK-write alternative is permitted.
- Treat a wrong test interpreter as an invalid runner invocation, not as a
  product failure and not as a passed gate. Preserve its result explicitly and
  require the exact project venv redirector plus base-runtime child for the
  qualifying rerun. This rule applies only to test infrastructure and does not
  authorize retry of any scientific, publication, Q, or E operation.
- Promote `entry_contract.json` only to the exact 305-byte ready value with
  SHA-256
  `50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844`.
  Preserve the exact historical incomplete value as a synthetic negative
  rather than keeping the live checked-in contract permanently blocked.
- Interpret `ready` narrowly: it exposes the already-qualified sealed
  dispatcher. It is not Q, E, a one-use publication authority, a supervisor
  job, a scientific execution authorization, or evidence that a production
  capsule exists.
- Keep the supervisor outside project execution source. Do not install its
  Startup recovery hook, materialize unresolved production launcher anchors,
  publish a capsule, consume the user's Q-v2 authorization, create E, or start
  original confirmatory science until the promoted repository passes every
  final live QA, CLI, PanNuke, two-build reproducibility, capacity, and
  lifecycle-readiness gate.

## 2026-07-31 - Fail closed on QA receipts while retaining scientific no-retry rules

- A successful-looking terminal test summary is not a valid event-driven
  process receipt when the numeric exit code or terminal state is absent.
  Preserve the ambiguous wrapper artifacts, write STOP, diagnose the local
  wait/state bug, and require a short exit-code probe before one qualifying
  test-gate rerun.
- Keep this bounded test-infrastructure rerun distinct from scientific and
  authority operations. It does not permit automatic retry of training,
  original confirmatory execution, Q, E, CREATE_NEW publication, or any
  one-use authority.
- For Windows venv children started through `Start-Process`, acquire the
  `System.Diagnostics.Process.Handle` while the process is live before
  `WaitForExit`, then refresh and read `ExitCode`. Atomic terminal state
  replacement must explicitly permit replacing the existing running-state
  pathname. A short zero-exit probe is required before entrusting a long QA
  command to this local wait path.
- Treat raw-data integrity inventories as fail-closed artifacts too. A
  nonterminating aggregation error or invalid trailing serialization bytes
  disqualifies that JSON even if its per-file hashes are present. Preserve the
  bad artifact and STOP record; a correction may remove only exactly proven
  serialization bytes and must reparse, recompute count/size/root, and state
  whether raw data was reread.
- Accept the final PanNuke immutability proof only because a separate
  post-validator scan rehashed every raw byte and matched every pre-capture
  relative path, size, UTC mtime, SHA-256, and the aggregate records root.
- Passing final pytest, static, CLI, and data gates authorizes only two
  non-publishing deterministic capsule builds and capacity computation. It
  does not itself authorize CREATE_NEW publication, Q-v2, E, supervisor
  arming, Startup installation, or science.

## 2026-07-31 - Accept exact capsule reproducibility without granting execution authority

- Accept the two independent non-publication candidates as one exact
  reproducible capsule because their complete 7,050,492-byte streams are
  identical at SHA-256
  `3e38dde3aa8efb76a0021985e0bab4a7091765c6b11e102037ced32c8a294e6c`.
  The independent audit also reproduced every ZIP member, the 19,414-byte
  manifest SHA-256
  `fd41910c77e70002ef3d2a3e21346317e094e9b600fe0524592fdf73d8a4ddb3`,
  and the 105-source records root
  `66874297ebefe74e1760abd1519abae0b91121541c4bf8748bc8cf597558ccb8`.
- Treat differing build timestamps and output paths in the two external
  receipts as declared provenance, not capsule nondeterminism; every other
  receipt field is equal. Preserve both candidate locations read-only and do
  not adopt either path as production authority.
- Permit only the exact typed sealed-capacity gate and a reviewed,
  content-addressed `publish_capsule_create_new` call. Publication must use
  the already verified whole-file hash, an absent exact destination, retained
  ancestor/leaf identity, CREATE_NEW, and same-handle byte/hash/archive
  readback. Any partial publication is a permanent STOP and must not be
  cleaned up, adopted, or retried.
- This decision does not consume the user's one Q-v2 authorization, create E,
  arm or install the supervisor, import the capsule dispatcher, or authorize
  scientific execution. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## 2026-07-31 - Supersede the stale single-copy capacity bound before publication

- Reject the first pair of capsule candidates as production-ineligible even
  though they are deterministic. Their sealed source contains a capacity
  policy that counts only one 30-GiB checkpoint copy, contradicting the later
  binding requirement for canonical and distinct versioned O_EXCL physical
  copies. No CREATE_NEW publication was attempted, so this is a validation
  correction and new non-publication build, not a publication retry.
- Adopt `original_confirmatory_sealed_plan_capacity_v2`, schema version 2,
  with explicit count two, 30 GiB per physical copy, 60 GiB for both copies,
  and exactly 10 GiB additional safety margin. The minimum is therefore
  70 GiB, not the obsolete 40 GiB. Bind the unchanged 108/90/18-cell,
  36-CNN, 180-fit plan roots and reject any inconsistent component or
  arithmetic.
- Preserve the rejected candidates and their receipts unchanged alongside an
  external immutable rejection record. Never publish, adopt, delete, rename,
  or reinterpret them as a qualifying capsule.
- Add `artifacts/execution_capsules/**` to Git exclusions before any
  production publication. Reconfirm raw data and derived large-file ignore
  coverage with `git check-ignore`.
- Because one execution-source member changed, require the complete project
  gate and two entirely new independent deterministic builds before
  publication. This correction changes no outcome, scientific cell, model,
  data, split, metric, restoration, statistic, threshold, or frozen file and
  grants no Q/E/scientific authority.

## 2026-07-31 - Keep lifecycle authority upstream of launcher materialization

- Reject any lifecycle-consumed technical-authority schema that binds the
  final terminal-client launcher, supervisor release, saved session, or
  terminal-composition identity. The launcher protects the exact static runner
  binding, the static binding contains the fresh lifecycle-readiness run, and
  lifecycle readiness consumes the technical authority. Binding the launcher
  in that upstream authority would therefore create an unresolvable
  dependency cycle rather than independent evidence.
- Bind the pre-lifecycle authority only to the exact parent P, unchanged frozen
  scientific inputs, the historical sealed primary, the final qualified live
  execution source, the content-addressed published capsule, capacity-v2
  evidence, and an independent outcome-split review declaring primary outcomes
  inspected and confirmatory outcomes uninspected.
- After that authority is published and verified, create the fresh lifecycle
  readiness evidence. Only then derive the exact static runner binding,
  materialize the launcher and supervisor release, and bind those downstream
  identities in Q-v2. No precomputed lifecycle path or hash may substitute for
  the actual sealed RunTracker evidence.
- Require a production one-shot Q-v2 controller before consuming the user's
  authorization. It must durably record immutable intent before the first
  publication byte, retain author-to-verifier custody, produce one durable
  success receipt or permanent STOP evidence, and never retry, adopt, clean
  up, overwrite, or reinterpret an ambiguous publication.

## 2026-07-31 - Require a published-wrapper T0 and four explicit phases

- Keep the T0 core upstream and acyclic. Its sealed directory binds parent P,
  frozen science, historical primary, final source, capsule, capacity, and the
  outcome-split review; it never binds lifecycle, launcher, supervisor,
  session, Q, E, or terminal artifacts.
- Treat process independence as more than a caller-declared PID. Builder and
  reviewer, and capsule publisher and readback reviewer, must have distinct
  PID/creation identities, implementation paths, and implementation hashes.
  The production producer must capture real OS identity in a fresh child.
- Require four explicit operations: `build-intent`, fresh-child
  `review-intent`, one CREATE_NEW `publish`, and separate read-only `verify`.
  Publication uses one fixed project namespace with a permanent claim; any
  ambiguity writes STOP and permanently forbids retry, adoption, cleanup, or
  overwrite.
- Lifecycle may consume only the composite verified-published wrapper, never a
  bare self-consistent T0 directory. The wrapper must prove the exact singleton
  namespace claim before and after full live verification while retaining the
  claim and authority files.
- Do not consume the user's one Q-v2 publication authorization while the
  controller can accept self-supplied T0/lifecycle records, choose an arbitrary
  state root, lose intent custody, or fabricate downstream success through a
  callback. Repair those boundaries and rerun independent P0/P1 review first.
- Passing component tests does not authorize a capsule build or publication.
  A final complete live QA/CLI/PanNuke gate is required after all T0,
  lifecycle, Q-controller, and source changes are integrated.

## 2026-07-31 - Bind the singleton publication wrapper through static v3

- Exclude transient T0 `intent` and independent-review receipts under
  `artifacts/original_confirmatory_technical_authority_requests/**` from Git.
  They are execution provenance, not execution source, and must not
  accidentally perturb the source inventory they attest.
- Supersede static-runner binding v2 with v3 before Q. V3 embeds the complete
  composite published-T0 lifecycle binding and cross-checks its namespace
  directory, namespace-claim SHA-256, technical-authority directory, artifact
  root, and technical-authorization SHA-256 against the existing flat pins.
  No discovery, legacy fallback, or independently supplied duplicate identity
  is allowed.
- Generic lifecycle APIs and their CLI aliases must reject a T0 authority.
  The strict original-confirmatory lifecycle operation alone accepts it,
  internally performs exactly one combined published-T0
  `verify_live=True`, uses only private shallow rechecks afterwards, and
  returns a closed readiness/composite/six-pin result.
- Do not expose a public `preverified` argument or caller-controlled
  `verify_live=False` switch. Q calls the strict lifecycle operation once and
  derives its binding from the returned verified object; it may not separately
  declare that T0 or lifecycle verification occurred.
- Any ambiguous or malformed supervisor restart inventory is a durable STOP
  with zero wake. Any durable wake intent, result, failure, or ambiguity
  evidence consumes wake authority even if another member of that evidence
  set is missing; recovery must never reinterpret it as permission for a
  second saved-session wake.
- These are execution-governance corrections only. They do not alter frozen
  science, consume Q/E authority, publish a capsule/T0, or authorize a
  scientific process.

## 2026-07-31 - Treat static typing as a fail-closed release gate

- A green functional supervisor suite and zero-P0/zero-P1 recovery audit do
  not qualify a release while production mypy is red. Repair every production
  diagnostic with explicit validation or narrowing; do not add ignores,
  weaken configuration, or classify missing runtime guards as cosmetic typing.
- In the protected authorization block, add an explicit non-None guard for
  every required custody/terminal contract so both runtime and the type checker
  see the same fail-closed mode invariant. The legal nonprotected test path
  does not enter this block; preserve that behavior and cover both modes.
- The older installed supervisor cannot substitute for this repair because it
  lacks the current Q/E, preterminal, and postwake contracts. Keep it unarmed;
  do not switch Startup or materialize a launcher until the new release,
  launcher pins, and synthetic saved-session E2E all pass together.
- Launcher type-only repairs are acceptable before materialization when they
  preserve all unresolved trust tokens and pass the complete functional,
  formatting, strict-mypy, compilation, and independent-review gates. They do
  not themselves authorize launcher creation, Q/E, wake, or science.

## 2026-07-31 - Keep mutable reporting outside execution-source identity

- Treat the execution-source and governance snapshots as intentionally
  disjoint identities. Execution authority hashes only executable code,
  configuration, and dependency definitions; truthful `STATUS.md` and
  `DECISIONS.md` updates belong to the separate governance snapshot.
- Therefore, recording a material command or decision after T0 publication
  does not by itself invalidate the execution-source root. Any change under
  `src/**`, `configs/**`, `pyproject.toml`, or `uv.lock` still invalidates it
  and requires a fresh source inventory before publication or execution.
- Preserve this boundary fail-closed through its focused tests and through the
  exact source manifest readback in T0. Do not move mutable status documents
  into execution-source merely to make reporting appear frozen.
- Do not qualify the external supervisor on functional tests plus mypy alone.
  Its exact release bytes must also pass the live repository Ruff policy and
  formatting check after deterministic mechanical repair, followed by a full
  regression rerun and independent audit. This grants no publication,
  supervisor arming, wake, or scientific authority.

## 2026-07-31 - A declared freeze is void if any candidate byte moves

- Revoke the entire freeze/audit claim immediately when a candidate changes
  after its exact hash and size are announced. Passing tests on an earlier
  file cannot qualify a later file, even when the later edit closes a genuine
  custody gap.
- Preserve the parent-controller hardening objective: a fresh reviewer must
  prove that its live parent is the exact controller recorded by the permanent
  review-attempt claim. A manually started child, abandoned-attempt adoption,
  vanished parent, or PID-reused parent must fail closed.
- Require this order for a replacement freeze: finish edits; run compilation;
  run the complete tests and static/help gates; hash twice without intervening
  writes; then let an independent reader rehash before auditing. Any mismatch
  restarts qualification but does not consume scientific or one-use
  publication authority.
- Keep Q-v2 disabled until its request-only internal constructor builds the
  real, exact source-pinned E/native-handle/IPC/supervisor path. Synthetic
  operation injection in test support proves state-machine behavior only and
  is not release evidence.

## 2026-07-31 - Bind reviewer lifetime in the kernel, not by a late PID check

- A one-time parent PID/create-time comparison is necessary but insufficient
  for the long T0 review. A second comparison before publication still leaves
  a race between that comparison and CREATE_NEW.
- Start the sole reviewer suspended and place it in a Job Object whose only
  kill-on-close handle is retained by the controller. Resume only after Job
  assignment and exact process identity verification. Controller hard death
  must therefore terminate the reviewer before it can publish.
- Do not duplicate or inherit the Job handle into the reviewer, because that
  would defeat parent-death termination. Assignment, resume, identity capture,
  live-review, or wait failure is a permanent STOP with no second child,
  adoption, cleanup, or retry.
- Keep duplicate static provenance outside STATIC-v3 when the downstream
  runner never receives both representations. Cross-check the flat and nested
  values only at the Q/scientific-authority canonicalizer and capsule
  bootstrap boundaries where both are present; do not grow the 24-field
  static contract merely to repeat them.
- Do not release Q/E around a fabricated downstream success mapping. A
  qualifying path must use exact live canonicalizers, retain physical
  leaf/ancestor and transport handles through ACK validation, and expose an
  explicit success-side finalizer for controller-owned handles. The unresolved
  suspended-spec path contradiction remains a blocker, not permission to
  precreate/adopt the supervisor's final job directory.

## 2026-07-31 - Assign the reviewer Job atomically at CreateProcess

- Supersede the earlier create-suspended-then-assign sequence. It has a hard
  parent-death interval before `AssignProcessToJobObject`, and the stock
  `subprocess.Popen` wrapper does not retain the original primary-thread handle
  needed to close that interval cleanly.
- Use direct `CreateProcessW` with an initialized `STARTUPINFOEX` attribute list
  containing `PROC_THREAD_ATTRIBUTE_JOB_LIST`, plus an exact inherited-handle
  list only for required standard streams. Retain the returned process and
  primary-thread handles and resume only after exact identity and Job
  membership readback.
- Treat any platform/API absence, attribute-list failure, create failure,
  identity mismatch, resume failure, or wait failure as permanent STOP. There
  is no qualifying fallback to `Popen`, post-creation Job assignment,
  breakaway, retry, adoption, or cleanup.

## 2026-07-31 - Separate the scientific matrix plan from release-plan hashes

- Define E `scientific_request_projection.plan_sha256` exclusively as the
  canonical frozen confirmatory matrix plan produced from
  `configs/confirmatory_frozen.yaml`. It must equal
  `ORIGINAL_CONFIRMATORY_PLAN_SEMANTIC_SHA256` and be rederived by the
  source-pinned production builder.
- Do not equate that value with the project `PLAN.md` file hash or the Q
  execution-capsule field that is also named `plan_sha256`. The current code
  supplies no equality invariant between those identities.
- Reconstruct fresh E's complete 180-directive checkpoint authority from
  exact frozen controls and outcome-blind data/split fingerprints. Cross-check
  its summary and roots against Q/static authority; do not expand the
  24-field STATIC schema merely to duplicate the full projection and do not
  read scientific outcomes.

## 2026-07-31 - Use one fixed CREATE_NEW control-staging namespace

- Resolve the pre-resume spec/final-job collision with
  `<supervisor_root>/control_staging/<job_id>`. E and the source launch
  documents live in that external directory; the final
  `<supervisor_root>/jobs/<job_id>` must remain absent until the resumed
  supervisor creates it.
- The one-use sealed attempt request must bind `job_id`, `attempt_id`,
  `run_id`, `execution_mode`, `retry_of_run_id`, `launch_nonce`, the exact
  staging directory, final job directory and E path before Q publication.
  Those values are produced by one explicit attempt planner and are neither
  caller-injected into the E factory nor generated after Q.
- Use a closed staging inventory: a durable attempt marker first, then
  CREATE_NEW E, the independent supervisor-v2 launch authorization, the
  staged launch spec, and a final ready marker. Retain no-follow leaf and
  ancestor custody through the validated Q/E ACK. The final outer
  `run_spec.json` must cross-bind the source spec and staging identities.
- Treat `launch_authorization.json` as a separate dynamic supervisor
  authorization, not as T0 technical authorization. Build its exact closed
  field set after the suspended process identity and deterministic Q/E
  READY/expected receipt are known, but before writing the staged spec or
  resuming the supervisor.
- Add a mandatory one-shot success finalizer for remaining controller-owned
  process/thread/Job/pipe/staged-spec/receipt/source-pin handles. After a
  validated ACK, any Q/E custody-close or success-finalizer failure is a
  permanent ambiguous STOP with no abort, retry, adoption, cleanup, or second
  launch.
- Extend one-shot startup recovery to the fixed staging inventory. Any
  abandoned, partial, corrupt, reparse, extra, or unmatched staging state
  writes STOP and may attempt the pinned diagnosis wake at most once; it must
  never launch science. The former heartbeat automation is absent and must
  not be recreated.
- These decisions authorize implementation and synthetic testing only. No
  live Q/E write, launcher materialization, supervisor arming, Codex wake, or
  scientific execution is authorized before the combined release gates.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-31 - Retire the generic confirmatory CLI execution path

- `experiment confirmatory` is no longer an execution entrypoint. It must fail
  before importing the executor or reading lifecycle, gate, dataset, cache or
  run inputs, with the explicit
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED` disposition.
- The sole qualifying real entry is the isolated capsule
  `run-confirmatory` mode after exact published T0, strict lifecycle, Q,
  one-use E, launch authorization, staged supervisor spec and event-driven
  supervisor custody have all passed.
- Keep CLI help for discoverability, but do not retain an unreachable legacy
  executor body that could be re-enabled or monkeypatched into a generic
  lifecycle run. Focused tests must prove zero lifecycle, gate and executor
  calls.
- This change invalidates every older execution-source snapshot for future
  publication. It does not consume Q/E, launch a process or alter the frozen
  science. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-31 - Allow only metadata in the final job before Q/E ACK

- Preserve the existing Q/E custody receipt authority at
  `jobs/<job_id>/q_e_custody_receipt.json`. Moving it into staging or adding a
  sixth staging file would break the terminal custody chain and the exact
  staging allowlist.
- After validating the committed five-file staging inventory, the resumed
  supervisor may CREATE_NEW-create `jobs/<job_id>`, its outer
  `run_spec.json`, and the custody receipt before emitting ACK. This is the
  only permitted pre-ACK final-job state.
- Require zero process C/scientific launch before the exact receipt and ACK
  have been independently validated and Q/E custody has transferred. Any
  mismatch, crash cut, ambiguous close, or restart remains a permanent STOP
  with no scientific retry or adoption.
- Freeze the staging inventory to exactly
  `staging_attempt.json`, `e_intent.json`,
  `launch_authorization.json`, `supervisor_launch_spec.json`, and
  `staging_ready.json` in that order. The last file is the commit marker.
- Place the closed `control_staging_projection` and
  `control_staging_projection_sha256` in Q-v2 and repeat them in
  `staging_attempt.json`. Keep both 44-field supervisor payloads unchanged.
  Bind the projection and first four identities in `staging_ready.json`, and
  bind the projection plus all five identities in the final outer
  `run_spec.json`.
- This resolves a causal contradiction in the execution-control mechanism;
  it does not amend frozen science, consume one-use authority, or launch an
  experiment.

## 2026-07-31 - Derive attempt identity from an acyclic Q base root

- Reject the provisional attempt preimage containing final
  `q_authority_root_sha256`: final Q also contains the attempt-derived
  staging projection, so that graph has no computable fixed point.
- Derive `q_base_authority_root_sha256` from exactly the 11 original static
  unsigned Q-v2 fields. Use this base root in the attempt-identity preimage.
  Derive attempt identity, job/run/nonce and `control_staging_projection` in
  that order, then calculate final `q_authority_root_sha256` over every other
  serialized Q field.
- Keep the serialized canonical Q field set closed at 17 fields: 11 base
  fields, one base root, two attempt fields, two staging fields, and one final
  root. Reconstruct the base projection rather than serializing it a second
  time.
- Use the final Q root for E, custody, terminal and scientific bindings. Use
  the base root only as the explicit acyclic seed and preserve both values in
  attempt evidence.

## 2026-07-31 - Transfer the outer Windows Job without clearing kill-on-close

- Create the suspended supervisor atomically inside an outer Windows Job by
  `PROC_THREAD_ATTRIBUTE_JOB_LIST`; keep
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set for the Job's entire lifetime and
  forbid breakaway.
- Do not set `ACTIVE_PROCESS_LIMIT=1`: the supervisor must run its authorized
  science, integrity-verifier, and one-shot Codex wake children. Exclusivity
  is enforced by the existing singleton and project mutexes.
- Keep the controller as the only Job-handle owner through staging and Q/E
  ACK. Use two dedicated bounded anonymous pipes for Job release/acceptance;
  do not weaken the existing one-line-plus-EOF Q/E stdin/stdout contract.
  Bind the child-side pipe handles and transport roots in the final
  `staging_ready.json`.
- Then `DuplicateHandle` a non-inheritable Job handle into the live
  supervisor, send one rooted `JOB_CUSTODY_RELEASE`, require the supervisor
  to validate the handle, Job flags and membership, CREATE_NEW-write the
  canonical accepted payload to
  `jobs/<job_id>/outer_job_custody_accepted.json`, and require the identical
  rooted `JOB_CUSTODY_ACCEPTED` on the dedicated pipe before any scientific
  process starts. The controller must compare the pipe and retained file
  bytes before releasing its source handle.
- Close the controller Job handle only after accepted custody. Never clear the
  kill flag: clearing it creates an orphan window and would allow descendants
  to survive a later supervisor crash.
- Exclude the transferred handle from every child handle list. Prove
  pre-duplicate, post-duplicate/pre-release, and post-release crash semantics
  plus nested child launch on real Windows before release qualification.

## 2026-07-31 - Require preallocated ownership of CreateProcess handles

- A successful `CreateProcessW` may be followed by an asynchronous exception
  before Python stores a returned process-information value. Job
  kill-on-close protects execution safety but does not close the controller's
  leaked `hProcess` and `hThread`.
- Allocate the mutable process-information owner in the caller before the API
  invocation. Pass it into the atomic create helper and make that helper
  return no unowned handles. Cleanup must therefore see every nonzero handle
  even across a return-boundary exception.
- Transfer each handle to its final owner by explicitly zeroing the
  preallocated slot; never permit the preallocated and final owners to close
  the same numeric handle. Qualify this with a real create-then-interrupt
  injection and exact close counts before T0 integration.

## 2026-07-31 - Make raw Windows launch resources owned before Python stores them

- Extend the preallocated-ownership rule beyond `PROCESS_INFORMATION`.
  Job, file/NUL, pipe and similar HANDLE-producing calls must return or fill
  an idempotent owner object that already owns the native value before the
  next Python STORE/UNPACK opcode. Returning a bare integer or tuple and
  assigning it later is not sufficient.
- For handle-return APIs, use one tested RAII object as the native `ctypes`
  result and retain that same object through transfer; never detach a bare
  integer across bytecodes. For output-parameter APIs such as `CreatePipe`,
  fill preallocated owned endpoints directly. Explicit close and finalizer
  close must be idempotent and must not double-close a reused numeric slot.
- Keep `PROCESS_INFORMATION` and the startup attribute list under the same
  caller-owned launch scope. Attribute-list cleanup is armed only after a
  successful initialization, is executed exactly once on every later exit,
  and is never called for a failed/uninitialized list.
- Qualify the implementation with real Windows opcode cuts after each native
  CALL, before UNPACK and before every owner STORE, plus normal transfer,
  double-close, failed initialization, child-terminal and process-handle
  delta tests. This is a bounded infrastructure repair, not authority to
  publish T0 or launch science.

## 2026-07-31 - Replace remote Job-handle injection with receiver-initiated duplication

- Reject the controller-preduplicated remote numeric handle contract. A real
  test proved an ABA slot failure: controller-side round-trip succeeded, but
  the running supervisor later observed the slot as a non-Job handle or
  invalid handle. That design cannot qualify even though it failed closed.
- After the exact Q/E ACK, keep the source Job handle in the authenticated
  controller. RELEASE binds the retained controller and supervisor process
  identities, the controller-local source handle value, the exact Job and
  transport roots, the bounded controller process access mask,
  `DUPLICATE_SAME_ACCESS`, non-inheritance, and `close_source=false`.
- The supervisor opens the exact already-bound controller process with only
  duplication, limited-query and synchronize rights; validates the same
  retained process instance and liveness before and after duplication; and
  calls `DuplicateHandle` from that controller source slot into
  `GetCurrentProcess`. It immediately validates that its new local handle is
  non-inheritable, is the exact no-breakaway/KILL_ON_JOB_CLOSE Job, and
  contains the supervisor.
- Keep the controller process handle through ACCEPTED write/flush and close it
  in a failure-safe finalizer. Keep the new local Job handle until supervisor
  process teardown after ACCEPTED; exclude it from every descendant
  HANDLE_LIST. The controller retains its source Job until it validates the
  exact ACCEPTED pipe bytes and retained receipt, then may round-trip the
  stable supervisor-local handle before closing its source.
- Bind the transfer direction, access mask, duplicate options,
  non-inheritance, and close-source policy inside the nested
  `outer_job_custody_transport` committed by `staging_ready.json`, not only in
  the later RELEASE. The top-level ready inventory remains closed; stale
  `remote_handle_*` names must be replaced with controller-source and
  supervisor-local names.
- Controller death before the supervisor duplicates closes the last Job
  handle and kills the suspended supervisor. Death after a local duplicate
  but before a valid ACCEPTED makes the supervisor's pipe operation fail; it
  closes its unaccepted local handle and exits without science. After valid
  ACCEPTED, the supervisor is the Job owner. No retry, adoption, breakaway,
  Job-flag clearing, or scientific launch before Q/E ACK is introduced.

## 2026-07-31 - Require controller COMMIT before science

- Supersede the statement that ACCEPTED alone completes usable custody.
  ACCEPTED proves that the supervisor has a valid local Job handle, but the
  controller has not yet independently compared that object or proved exact
  relinquishment of its source handle. Science therefore remains forbidden.
- Use three distinct bounded anonymous pipes and exact one-line-plus-EOF
  messages: RELEASE, ACCEPTED and COMMIT. The supervisor HANDLE_LIST contains
  exactly six inherited endpoints including the new COMMIT read endpoint.
  Each inclusive byte bound is 65536. Do not multiplex RELEASE and COMMIT on
  one stream and do not add a fourth ACK.
- Freeze the exact v2 inventories at these sizes: outer contract 8,
  transport 32, source-to-supervisor duplicate contract 13, controller
  round-trip contract 13, RELEASE 17, transfer preimage 10, ACCEPTED 27,
  round-trip evidence 21, COMMIT 24 and COMMITTED receipt 31. Keep the
  staging-ready top-level set at 39 fields and reject every v1 alias.
- Bind the controller-local source Job value, exact Q/E-bound controller
  process-identity root, duplicate and round-trip subcontracts, three pipe
  endpoints, both final receipt paths, source-retention requirement and
  source-close-before-COMMIT requirement in the nested transport committed by
  staging READY.
- Use controller open mask `0x00101040`: duplicate-handle, limited-query and
  synchronize only. Authenticate PID, creation time, image path/hash and
  liveness on one retained process handle before and after duplication.
  Preserve the exact command SHA binding from Q/E READY, but do not claim a
  live controller PEB/argv readback.
- After ACCEPTED and its identical retained receipt are verified, the
  controller round-trips the supervisor-local Job to a temporary local
  handle, revalidates the source Job and supervisor membership, and requires
  `CompareObjectHandles(source, temporary)` to return true. It closes the
  temporary handle, then exact-closes the source Job, and only then emits
  COMMIT. A failed or ambiguous source close forbids COMMIT.
- The supervisor persists the canonical COMMIT as CREATE_NEW
  `outer_job_custody_committed.json`, seals and rereads it, revalidates its
  retained local Job, membership, exact flags and noninheritance, and records
  `scientific_process_launched=false`. Only after that retained evidence may
  process C start.
- Any death or malformed/partial/extra/EOF state before valid COMMIT makes
  the supervisor close its uncommitted local Job and exit without science.
  Startup recovery never reconstructs numeric handles: accepted-only and
  committed-without-terminal states are permanent STOP, zero adoption and
  zero automatic retry.

## 2026-07-31 - Bound asynchronous native-resource testing to one fault at a time

- Test one injected asynchronous cut at every acquisition, native return,
  UNPACK, STORE and ownership-transfer boundary, plus ordinary cleanup
  failures as separate cases. Require exact close/idempotence and real handle
  validity/count evidence for that single-fault model.
- Do not recursively require a second asynchronous injection while the
  idempotent cleanup for the first injected fault is executing. Hard process
  death remains covered independently by Windows process teardown and the
  continuously set Job `KILL_ON_JOB_CLOSE` contract.
- This bound prevents an unfinishable recursive fault model while retaining
  all scientifically material fail-closed, no-orphan and no-retry
  guarantees.

## 2026-07-31 - Admit qualified T0 source, reject the old Q-v2 snapshot

- Admit only the independently audited six-path T0 integration carrier:
  frozen publisher and reviewer sources, their two frozen test files, the
  minimal main CLI import/registration, and one main CLI routing test. Every
  other file in the external T0 WIP is excluded.
- Treat source integration and one-use publication as separate events. The
  live CLI may build, review, publish, and verify T0, but no authority is
  published until the final source/capsule/capacity inputs and downstream
  lifecycle ordering are ready.
- Keep the generic `experiment confirmatory` entry fail-closed even after T0
  registration. T0 availability is not authority to bypass the sealed
  capsule, published-Q, one-use-E, lifecycle, or supervisor chain.
- Supersede the earlier qualification of Q-v2 bytes
  `7B1D802B99E3BD8FEEBC22E192654F26504B63D36265E46FC44CB764CA45FE78`.
  A real asynchronous cut proved that raw ctypes `CreateFileW` returns can
  leak between CALL and Python STORE. Passing ordinary tests does not waive
  this native-ownership defect.
- Require the Q replacement to own every native HANDLE on the evaluation
  stack at return, use preowned/idempotent custody for HANDLE-to-CRT and
  CRT-to-file transfers, and test one asynchronous cut at every acquisition,
  return, unpack, store, and ownership-transfer boundary. The rejected Q
  snapshot remains immutable evidence and must not be patched in place.

## 2026-07-31 - Require strict JSON types and context-independent QA

- Closed-schema equality is not sufficient when Python considers
  `True == 1`, `False == 0`, or `1 == 1.0`. Every bool and integer field in Q,
  E, staging, supervisor spec, authorization, ATTEMPT, READY, Job transport,
  and terminal receipts must first pass exact type identity and only then
  value/root checks.
- Regression tests must replace the value with a wrong-type equal value,
  recompute every dependent root and file hash, and still observe fail-closed
  rejection. Tests that only leave a stale root do not prove the typed
  contract.
- Native handle ownership starts at the native return boundary, not at the
  later Python assignment. `OpenProcess` requires an owned ctypes restype and
  `DuplicateHandle` requires a preallocated owned output slot. Wrapping a raw
  integer after CALL or after native success remains disqualifying.
- A candidate's mandatory QA must be invariant to the invocation directory.
  Standalone success does not qualify a control-plane file if the same strict
  mypy command under the project configuration fails. No ignore, cast-only
  masking, or configuration weakening is permitted.
- Keep the fresh-180 scientific material builder separate from the already
  qualified E/factory transport. It must bind the frozen scientific plan,
  controls, source/config/split fingerprints, and exact 180-directive typed
  projection without reading outcomes. Transport parity alone is not a
  complete E authority.

## 2026-07-31 - Admit lifecycle/STATIC-v3 and reject the first fresh-180 builder

- Admit only the independently audited 12-path lifecycle/STATIC-v3 allowlist.
  Preserve its exact STATIC24, composite10, lifecycle22 schemas, nine
  equality checks, and public-then-private `[verify_live=True,
  verify_live=False]` verification order. Keep the external compatibility
  test outside live execution source.
- Treat the lifecycle carrier as necessary infrastructure, not authority to
  publish T0, write Q/E, arm the supervisor, or execute confirmatory science.
  The generic direct confirmatory CLI remains a mandatory exit-2 hard stop.
- Reject the first fresh-180 builder snapshot as a production pin. Hashing 14
  files does not authenticate a roughly 70-module transitive import closure,
  and closing those handles before import permits a source substitution
  window. A supplied execution-source root or manifest hash must be
  independently recomputed, not accepted because it is SHA-shaped.
- Require the successor to bind each of the 180 directives to the exact
  frozen 36-CNN-cell set and five folds. A validator may not derive its own
  expected set from the rows being validated. It must also close nested
  scientific schemas and pass coherent-tamper tests that recompute every
  dependent hash/root.
- Keep immutable content-addressed release code and mutable supervisor state
  in disjoint roots. Publication, verification, Q, E, supervisor arming, and
  science remain separate one-use transitions with no automatic scientific
  retry.

## 2026-07-31 - Bind release commands and filesystem evidence to live OS identity

- Define the publisher command record as exactly eight fields:
  `schema_version`, `policy`, `program_path`, `program_sha256`, `source_path`,
  `source_sha256`, `argv`, and `cwd`. Define the independent verifier record
  as the same shape without source path/hash, exactly six fields. Their
  command SHA values are hashes of the complete canonical records, not hashes
  of a reduced argv-only wrapper.
- Require full `sys.orig_argv` with exact isolated prefix
  `program_path -I -S -B source_path` and exact control-root working
  directory. A reduced three-field Q authority wrapper, if retained at all,
  is separately named and may not masquerade as the publisher or verifier
  command digest.
- For every release file and directory in both readback passes, retain a
  no-write/no-delete-share handle; observe volume, 128-bit file ID, attributes
  and link count before; read file bytes through that same handle; perform
  unavoidable path security and ADS checks while custody remains held; match
  an independent path handle to the retained identity; and require an exact
  retained after-observation. Any mismatch is STOP.
- Require a live protected-DACL bit, canonical owner SID, exact owner+DACL
  SDDL hash, zero disallowed alternate streams, no reparse point, and
  link-count one. Receipt booleans or self-reported hashes do not substitute
  for those OS observations.
- Pin the current handoff session exactly as
  `019f703b-661d-7c50-b423-9270657d8d6d`, never `--last`. Do not claim a
  successful exact-resume test by rewriting old evidence. The final
  short-process handoff must execute the one real resume first, then record
  and rehash that evidence, and rerun the focused protected gates.

## 2026-07-31 - Separate Q source qualification from the one authorized Q write

- Admit `FC83539A...58A96E` only as the independently qualified Q controller
  source candidate. Its passing tests and audit do not themselves consume the
  one-use Q publication authority.
- Require the final immutable control-plane release manifest and independent
  release receipt to bind that exact source before constructing the Q
  request. Then perform at most one CREATE_NEW Q attempt and one independent
  readback; never retry, overwrite, adopt a partial attempt, or substitute a
  later source hash.
- Keep the production Q gate disabled-before-read until every release,
  lifecycle, E/factory, fresh-180, supervisor and capacity prerequisite is
  closed. Source qualification is not permission to inspect outcomes or run
  confirmatory science.

## 2026-07-31 - Admit only the full-closure fresh-180 successor

- Supersede the rejected 14-file builder with the independently audited
  successor rooted at `93926caa...abc3e`. Require all 114 source records,
  retained-byte loading of the complete 105-module project closure, a
  default-deny import policy, and exact final recapture before material may
  enter E.
- Bind the 180 directives to independently frozen exact sets: 108 total
  cells, exactly 36 CNN cells, and exactly five folds per CNN cell. Never
  derive the expected set from the candidate rows being validated.
- Require coherent-tamper tests to recompute every dependent hash and root.
  Rejection based only on a stale parent hash is insufficient evidence for
  the cell-set or nested-schema contract.
- Keep builder qualification separate from E construction and E publication.
  The builder remains outcome-blind and non-writing; its passing audit does
  not authorize Q, E, supervisor arming, or science.

## 2026-07-31 - Exact JSON types precede mapping equality in E/factory

- Revoke E/factory source hashes `848DA97D...` and `4B4DD632...`. A bounded
  module audit that passes local tests does not survive a later coherent
  wrong-type-equal counterexample.
- Never use Python dictionary equality as proof of a JSON contract before
  recursively proving exact field types. Validate `type(value) is bool`,
  `type(value) is int`, and exact string/list/dict structure as applicable;
  only then compare values and recomputed roots.
- Require mutations such as `2 -> 2.0`, `1 -> True`, and `false -> 0` with
  every enclosing hash/root coherently recomputed. The validator must reject
  them for the intended type reason, not merely because a stale digest
  remains.
- Preserve the failed bytes as evidence and create a new explicit successor.
  Never silently upgrade an already cited candidate hash.

## 2026-07-31 - Default-deny every control-plane execution alias

- Reject release-tools bytes that search only familiar alias names such as
  `current`, `latest`, or `*.lnk`. A verifier may not assert alias absence
  from a partial name denylist.
- Enumerate the complete control root under retained custody and allow only
  the exact closed namespace required by the release contract. Reject every
  other file, directory, reparse point, junction, symlink, mount point,
  shortcut, or pointer-like entry regardless of its name.
- Include arbitrary `active` files and opaque directory symlinks in the
  coherent positive-publication-then-negative-verification matrix. The test
  must reach the alias predicate rather than fail earlier on an unrelated
  stale root.
- Treat a publisher SUCCESS as non-qualifying if the publisher can still fail
  afterward. Mutex release, source-lease close, retained-handle close, flush,
  ACL/read-only sealing, logging and interpreter teardown must not create an
  ambiguous `SUCCESS + nonzero exit` state that a later verifier accepts.
- If SUCCESS cannot causally follow every fallible publisher cleanup, require
  a distinct parent-owned exit attestation created only after WaitForExit
  observes code zero and binds the exact publisher process identity, command,
  SUCCESS bytes and release root. Do not let the child attest its own future
  exit code.
- Use one combined parent terminal attestation after both publisher and
  verifier children exit zero, rather than recursively attesting each waiter.
  The parent holds the singleton across both children, waits by process
  handle, revalidates both child receipts, and makes its one CREATE_NEW
  attestation commit the final authority. Q consumes this combined
  attestation, not raw child success alone.

## 2026-07-31 - Q17/release44 must exist in the live canonical authority

- Isolated Q validation is insufficient when Q delegates to a different live
  Q12/release42 canonicalizer. Require exact field-set and type parity across
  live authority, bootstrap, Q, E/factory, supervisor and release verifier
  before any one-use write.
- Replace, rather than alias, the combined `supervisor_root` contract.
  Immutable code must derive only from the content-addressed release root;
  mutable jobs and control staging must derive only from the supervisor state
  root. Stale field names are rejected even when their value happens to equal
  the state root.
- Implement this as a separately versioned, bounded live-carrier successor.
  Preserve STATIC24, published-T0 composite10, lifecycle22 and their frozen
  verification ordering exactly; do not smuggle Q/E/Job fields into those
  upstream schemas.
- Treat `FC83539A...` as isolated qualification evidence only. A corrected Q
  successor must independently enforce recursive exact types and constants,
  use Q17/release44 names, and pass parity against the eventual live
  canonicalizer bytes before publication.
- Bind the protected supervisor launch to the exact staged Option-A bootstrap
  argv:
  `python -I -S -B <source> --root <state> run
  --staged-launch-spec <state-path> --staged-e-intent <state-path>`.
  Do not authorize the older arbitrary `run --spec` vector. The staged
  bootstrap is the causal path that lets Q/E bind the live supervisor process
  identity and retained custody before any protected work.

## 2026-07-31 - Make the parent attestation rename the terminal operation

- Do not CREATE_NEW the authoritative parent-attestation path and then perform
  fallible writes, sealing, security changes, readback, handle cleanup, or
  collection growth. A terminal file that can outlive a later nonzero waiter
  exit is not a causal qualification anchor.
- Build the attestation at a closed, non-authoritative staging path inside the
  exact transaction. Write, flush, seal, apply and verify the protected DACL,
  verify readonly/single-link/no-ADS/non-reparse state, and retain all required
  handles before publication.
- Publish authority with one tested handle-based, no-replace rename to the
  exact `release_qualification_attestation.json` path. That rename is the last
  fallible action. Preallocate the postcommit state slot, retain rather than
  close handles and mutexes, and call `os._exit(0)` immediately after the
  successful slot store.
- Default-deny the staging leaf in every terminal consumer. Every injected
  failure before rename must leave the final authority path absent; collisions
  and ambiguous staging are permanent STOP states, never retry signals.
- Use a fresh, explicitly versioned empty control root for this protocol.
  Preserve historical control-plane directories as evidence and reject,
  rather than clean or adopt, a nonempty legacy namespace.

## 2026-07-31 - Exact-type validation also covers custody wire handles

- Do not treat prior recursive JSON hardening as sufficient when a custody
  canonicalizer later performs `actual != expected` on numeric handle values.
  Python equality aliases equal integers and floats and can also alias booleans
  and integers.
- Validate every wire handle with one closed exact-integer predicate before
  any value comparison, including accepted, evidence, commit and nested
  committed records on both controller-source and supervisor-local sides.
- Require coherent adversarial tests that recompute every dependent record
  root after changing an integer handle to an equal float or boolean. Reject
  the candidate inventory `75fa00e5...f14bb`; a passing local 84-test suite
  does not override the independent counterexample.
- Apply the rule to entire expected records, not only handle fields.
  `schema_version: true` must never compare equal to integer `1`, and a
  canonicalizer must never silently normalize such an input by returning its
  internally rebuilt expected mapping. Audit every whole mapping/list equality
  in the authority and factory and require recursive type identity before
  equality.

## 2026-07-31 - Apply recursive equality across the live Q/E custody carrier

- Do not freeze or integrate the Q17/release44 carrier merely because its new
  schemas pass focused tests. Its retained Q/E custody contract, receipt, ACK
  and downstream-spec canonicalizers are part of the same authority boundary
  and must reject type aliases as well.
- Replace ordinary deterministic mapping/list equality with the existing
  recursive strict-JSON comparator and keep explicit exact scalar predicates.
  Test `true` versus `1`, `false` versus `0`, and integral floats versus
  integers while preserving or coherently rebuilding every relevant enclosing
  root.
- Recursive equality at the outer record does not repair a nested
  canonicalizer that already accepted and returned a wrong scalar type.
  Require exact integer predicates inside every nested identity, ancestor
  lease, schema, count, size and PID canonicalizer, then test the complete
  READY-to-receipt-to-ACK-to-spec chain with all enclosing roots recomputed.
- Hold dependent Q and supervisor bytes until they are tested against the
  corrected live authority. A pre-freeze counterexample changes no scientific
  authority and consumes no one-use publication permission.

## 2026-07-31 - Admit the corrected E/factory exact10 candidate

- Admit external E/factory inventory
  `35d5dec04d20e33ea233c7eb0c218039ccf0639df81754995b8aee23d713a63f`
  as the only current qualified successor to the rejected `75fa...` snapshot.
  Its independent P0/P1/P2 result is 0/0/0 and its before/after digest is
  stable.
- Pin authority source
  `4ba53986041a01ef6dc0e4ae65a324411c4cad722e02291d553f42b4379f8a3e`
  and factory source
  `61b0683f73285425da8b2320d18256ffe17ba13f311c1be554f3f3f847d4f1d2`
  only when constructing the eventual immutable external release. Do not
  substitute the rejected predecessors or a later modified WIP.
- Qualification does not authorize E construction or publication. Those
  remain downstream one-use transitions after the live authority, release
  waiter, Q and supervisor contracts have independently converged.

## 2026-07-31 - Bind release tools to direct interpreter execution

- Never infer a release tool's executed source from `__file__` alone. An
  imported module can report the qualified file while an unqualified stdin,
  `-c`, module or wrapper program controls its state and entry call.
- Before any production authority read or control-root write, require exact
  native program identity, `-I -S -B`, direct source position, authorized argv
  tail, cwd, `sys.orig_argv`, `sys.argv` and PEB command-line parity.
- Apply the same rule to waiter, publisher and verifier. Child capability and
  parent-process validation supplement this direct-shape proof; they do not
  replace it. Wrapper rejection must leave no attempt marker, child or
  terminal artifact.
- Generic status/root validation is not sufficient for publisher SUCCESS or
  verifier VERIFIED receipts. Canonicalize the complete receipt-specific
  schema recursively, including every boolean, integer, nested identity and
  causal crosslink, before a child completion can enter the parent
  attestation.
- A final inventory scan is not a namespace lock. Before scanning, retain the
  complete directory identity set, deny new write/delete opens with protected
  DACLs and compatible share modes, and retain a pre-authorized target
  directory handle. Publish the terminal name with a handle-relative
  no-replace rename. Concurrent extra-entry injection in the scan-to-rename
  window must be impossible or must prevent authority publication.

## 2026-07-31 - Close every supervisor prior-exit proof variant

- Treat nested restart evidence as an authority object, not an arbitrary JSON
  payload under a closed outer record. Define the exact field set and exact
  integer-or-null types for every `_prior_supervisor_exit_proof` variant.
- Reject boolean/float PID, exit-code and Win32-error aliases and every extra
  key. Bind the independently generated recovery verifier with recursive
  exact-type equality rather than Python mapping equality.
- A restart remains STOP-only and never initiates or retries science. Passing
  this repair does not by itself authorize a real startup-recovery or Codex
  wake test.

## 2026-07-31 - A run-spec self-hash does not replace bootstrap semantics

- The fresh-process bootstrap independently enforces the Q/E custody contract,
  READY/handoff, receipt and downstream spec binding before importing project
  code. An exact outer field set and a self-consistent run-spec hash are
  necessary but not sufficient.
- Reconstruct trusted deterministic fields from the held job directory and
  pinned Q/E authority, validate every dynamic nested identity/handle record
  with closed exact types, and verify all roots and crosslinks. Coherent
  boolean/integer/float aliases in a rehashed envelope must fail before any
  scientific input is read.

## 2026-07-31 - Admit the six-file live carrier inventory

- Admit only live-carrier inventory root
  `89A21A001BB9307C7657807753E4FD6381CA0DFB931C7AB3E5FC5C31E1EB5CD7`.
  It includes the four independently qualified carrier files plus the exact
  dependent terminal caller and regression test required by configured mypy.
- Derive the supervisor launch command only from the canonical Q control
  staging projection. Never substitute final `run_spec.json` for the staged
  launch spec or staged E intent, and retain final run-spec validation as a
  distinct downstream step.
- Treat the repository-specific import reorder as a new audited snapshot, not
  as if the external WIP hash remained unchanged.

## 2026-07-31 - Make the release authority a sealed directory capsule

- Withdraw the unimplementable requirement to freeze the writable destination
  parent and then rename into it using the same Windows token. A pre-opened
  handle does not bypass the later destination access check, and no passing
  test may be used to imply otherwise.
- Freeze and read back the complete source qualification capsule instead.
  Commit the whole four-leaf directory by one no-replace rename into the
  versioned verification namespace. That directory rename is the last
  fallible operation; after it, only preallocated in-memory state and process
  exit are allowed.
- Scope the terminal claim precisely to exact immutable capsule contents and
  the pinned final path. Do not claim that unrelated control-root siblings are
  globally absent forever. A final-name collision or any capsule mutation is
  permanent STOP and never an automatic retry.

## 2026-07-31 - Do not admit old-session handoff evidence

- Component tests using `019faaf3-c547-79e1-b0eb-26e35d214642` do not satisfy
  the current exact-session requirement. They remain useful behavior tests but
  cannot pin the production handoff.
- Preserve the old receipt as superseded evidence. Only the one authentic,
  controlled short-process resume of
  `019f703b-661d-7c50-b423-9270657d8d6d`, after all non-wake gates, may create
  its replacement. Never rewrite old evidence, use `--last`, or retry a failed
  or ambiguous wake.

## 2026-07-31 - Keep synthetic transaction time causally ordered

- Synthetic publication tests must derive attempt/review/publication times
  from the live test-controller process creation timestamp. A calendar literal
  eventually predates the process and creates false failures unrelated to the
  contract under test.
- This is test-fixture maintenance only. It changes no frozen scientific time,
  run evidence, preregistration, Q/E authority, result or completion stage.

## 2026-07-31 - Admit only the final sealed-directory release inventory

- Admit release-tools inventory root
  `DA9B0EF9353760A1E8DC1D555B34B6A46D7AEF1F81640878A63F3ED17C4A8CC5`
  as the sole qualified candidate for the versioned production release. Its
  exact eight files total 679,648 bytes and its final independent audit found
  no P0, P1 or P2 issue with unchanged before/after hashes.
- Retain exact protocol shapes waiter 17, child 19, capsule manifest 18,
  capsule binding 13 and attestation 57. Publication commits one protected
  four-leaf qualification directory by a final no-replace rename; it does not
  claim global immutability for unrelated verification siblings.
- Treat every earlier release-tools root as obsolete HOLD evidence. Passing
  qualification authorizes later construction of the one-use transaction; it
  does not itself authorize publication, Q/E writes, a Codex wake or science.

## 2026-07-31 - Transport the publication-specific attestation in exact48

- Replace the live `supervisor_release` exact44 schema with exact48 by adding
  only publication ID, qualification-attestation path, attestation file
  SHA-256 and attestation semantic-root SHA-256. A release-root digest alone
  cannot derive a `cpr-...` directory and must never trigger a scan or
  latest/alias choice.
- Require the exact scoped `AANCA-control-plane-release-v2/verifications/`
  `<cpr-id>/release_qualification_attestation.json` shape and carry both hash
  domains through Q, E, READY/spec and bootstrap/supervisor crosslinks.
- Keep release-tools provenance root `DA9B0EF9...A8CC5` outside this runtime
  schema. It is the qualification root of an eight-file WIP containing tests
  and documentation, not an exact57 attestation root, waiter17 root or runtime
  release root. Do not compare different hash domains or add an ungrounded
  compiled constant merely to transport that provenance value.
- Admit live exact48 inventory root `D9963D39...73DA0` after focused QA and an
  unchanged-byte audit with no P0/P1. Preserve the disclosed optional
  test-module strict-mypy debt; mandatory configured typing remains the
  production `src` gate.

## 2026-07-31 - Retain FINAL2 evidence instead of duplicating its parser

- A successful pre-spawn exact57 read is not continuous custody. Retain
  no-follow reduced-read handles for the pinned qualification attestation,
  release manifest, four capsule leaves and all critical referenced
  runtime/source/copy/receipt leaves from validation through process creation,
  resume and terminal completion.
- Revalidate physical identity, bytes/hash, link/ADS and protected DACL at the
  pre-create and pre-resume boundaries. Any drift or unavailable retained
  handle is STOP with no automatic retry.
- Do not duplicate the complete exact57 parser in `capsule_bootstrap.py` merely
  to perform another path read. Without retained custody that adds a second
  TOCTOU; with the supervisor lease it is redundant. The bootstrap remains an
  independent exact48/Q/E structural and hash validator before project import.

## 2026-07-31 - Q validates producer schemas, not only their generic roots

- Treat FINAL2 release authorization and release attempt as closed producer
  objects: exact40 and exact24 with exact scalar types, policies, commands,
  process identities, one-use nonce and causal timestamps. A valid self/root
  hash alone is insufficient.
- Crosslink authorization -> attempt -> both exact19 child completions ->
  source/copy receipts -> exact57. Bind the complete production argv and
  command/process identities across both representations, not only an
  interpreter prefix or receipt path/hash.
- Keep Q on HOLD while any of those links is under-validated, even when the
  full local test suite passes. Correct producer-policy/path differences as
  literal parity fixes rather than weakening the FINAL2 producer.

## 2026-07-31 - Admit E/factory exact10 root e0e214d9

- Admit external E/factory inventory root
  `E0E214D99CBE8BE21EF0357EC9ADA2CE83C6D355D365896CBB049C4A6AE70E56`
  after full QA and an independent unchanged-byte P0/P1/P2 = 0/0/0 audit.
- Require the complete exact48 value and its existing contract root in the
  exact8 wrapper, then propagate the same four FINAL2 fields through SPEC51,
  AUTH51 and READY46. Preserve the distinct rich-Q versus compact-factory
  hash preimages and reject cross-domain substitution.
- A synthetic Windows path-length failure under an excessively long basetemp
  is not a protocol retry or hidden test pass. Record it, use one short fresh
  TEMP root, and require the same unchanged bytes to pass, as they did.
- Qualification does not authorize construction or publication of real E.

## 2026-07-31 - Admit Ruff-canonical fresh-180 root f46d9fca

- Supersede the earlier unformatted fresh-180 candidate with exact four-file
  root `F46D9FCA1D91A43E0F28E77A65EA5C0BED4A64D67A2706AEB6B084D17AAF8C25`
  after mechanical formatting, complete requalification and an independent
  P0/P1/P2 = 0/0/0 unchanged-byte audit.
- Keep fresh-180 outside the supervisor-release schema. It produces only the
  closed outcome-blind fresh scientific material; E/factory supplies the
  separate exact48 and staging authority. Do not add redundant control-plane
  fields or silently pin an earlier root.
- Qualification does not authorize a real material build or E publication.

## 2026-07-31 - Admit Q source root 084455b8 without production pins

- Admit Q inventory root
  `084455B8DB84CC6A8264E4AC8FD5318AE1F53B5193797DBB378B7348E3E341D9`
  after complete producer-parity repair, `239 passed` and an independent
  unchanged-byte P0/P1/P2 = 0/0/0 audit.
- Require closed exact40/exact24/exact19/exact57 semantics, complete command
  and process crosslinks, producer-literal policies/paths and sealed log
  leaves. Do not regress to generic-root-only validation.
- Keep production manifest and qualification-attestation pins out of compiled
  source before publication. Supply them through the sealed one-shot request
  after exact release publication and validate them before any Q write; this
  preserves disabled-before-read behavior and avoids a self-containing release
  hash cycle.
- Qualification authorizes later one-use invocation only after the supervisor
  and release gates pass; it does not consume the single Q permission.

## 2026-07-31 - Post-wake FINAL2 revalidation is unconditional

- Put the complete wake invocation and result persistence/readback inside one
  guarded `try/finally`. Once a real resume attempt may have occurred, FINAL2
  retained-custody revalidation must run on success and on every exception.
- Failure or ambiguity after resume writes durable STOP and consumes the
  one-wake opportunity. It never permits a second wake, retry, relabelled
  receipt or continuation to science.
- Reject supervisor source hash `FBE624E4...`; passing local tests before this
  counterexample do not qualify it.

## 2026-07-31 - Disabled Q qualification is not production admission

- A Q source can pass all parser, parity and disabled-before-read tests while
  remaining intentionally unable to execute. Do not label such a snapshot
  production-ready when compiled final pins are absent or its live-authority
  constructor ends in an unconditional STOP.
- Build one explicit final-Q successor only after all non-cyclic source hashes
  are final. It must return the complete internal authority tuple, including a
  real source-pinned E/supervisor publish-verify-handoff function and
  native-handle-backed downstream canonicalizer. No caller-supplied callback or
  symbolic placeholder is acceptable.
- The control-plane release contains eight runtime files under 13 exact tree
  entries. The execution-capsule bootstrap is not a ninth runtime role.
- Root `084455B8...341D9` remains admitted only as the qualified disabled
  checkpoint from which that successor may be derived; it does not authorize
  a release or consume Q.

## 2026-07-31 - Wake consumption follows durable attempt evidence

- Do not classify every exception inside the wake wrapper as post-wake
  consumption. A pre-attempt failure and an ambiguous/post-attempt failure have
  different legal outcomes.
- Before durable wake-attempt authority exists and before resume is invoked,
  revalidation may still permit the one legal first wake. After durable
  attempt evidence or possible resume, the opportunity is consumed and every
  failure is permanent STOP with no retry.
- Integrity failure before wake never wakes Codex; failure after a possible
  wake never wakes it again.

## 2026-07-31 - Never resume the same active Codex session concurrently

- Treat the exact current session being active as a STOP condition for its
  one-shot resume test. CLI syntax alone is not evidence that concurrent turns
  on one session are safe.
- Preserve the authentic old-session receipt as superseded evidence. Replace
  neither its session ID nor its provenance fields. The current Desktop
  session needs an honest origin/presence record tied to its first
  `session_meta`, not a fabricated non-interactive creation test.
- Qualify an event-driven delayed-success or equivalent idle-boundary gate
  before the one permitted current-session resume. Bind a unique nonce and
  marker; record exact CLI identity, command, prompt, process identity, logs
  and terminal state. Any ambiguity consumes the attempt and forbids retry.

## 2026-07-31 - Formal qualification records are distinct from inventory roots

- Build exactly five closed qualification records for the eight FINAL2 runtime
  components. Each record has its own policy, role, exact artifact list and
  canonical `qualification_root_sha256`; a convenient directory inventory
  root cannot be substituted for it.
- Add a closed `python_runtime_identity` external dependency bound to exact
  interpreter path, file hash and native identity. Keep it separate from the
  execution-source manifest file hash and records root.
- Assemble the exact 20-field release projection and publish input only through
  a qualified dry-run builder with production disabled until final Q,
  supervisor and reviewed qualification receipts exist.
- Compile only independent pre-existing source/evidence pins into Q. Keep the
  release root, manifest and publication-specific exact57 as sealed request
  pins to avoid self-hash cycles.

## 2026-07-31 - Staged E and final supervisor job are separate path authorities

- Reject any E adapter that writes `e_intent.json` into the future final job
  directory or derives final receipt paths from the physical E parent.
- The E leaf and its retained ancestor handles belong to the fixed
  `control_staging/<job_id>` domain. The final supervisor job/receipt authority
  is a distinct domain and must remain absent until the suspended child and
  exact handoff have passed their pre-resume checks.
- Implement this separation in a new, synthetic-only authority successor and
  cross-validate both domains in READY/ACK/readback. Do not weaken the existing
  no-follow, no-overwrite, native-identity or retained-handle checks.

## 2026-07-31 - A passing waiter must exclude pre-open write handles from commit

- Permanently disqualify release-tools candidate root
  `DA9B0EF9...A8CC5`. Ordinary unit tests and a protected DACL are insufficient
  when a write-capable handle was granted before protection and survives the
  final scan.
- New staging leaves may share only read access while the owner retains the
  one legitimate write handle through sealing. Write/delete sharing is not
  allowed for independently opened handles in the commit window.
- A deterministic pre-open-writer regression is mandatory: mutation after the
  final scan must be impossible, and no ambiguous or mutated state may produce
  exit 0. Repair only in a separate successor; retain the failed candidate as
  immutable diagnostic evidence.

## 2026-07-31 - Test-only supervisor changes require a new full inventory audit

- A production-source hash alone is not sufficient to carry a prior bundle
  qualification across a test change. Supersede the previous nine-file root,
  record the new complete inventory, and repeat the independent audit on
  unchanged bytes.
- Stabilizing the handle-soak baseline with an explicit collection is accepted
  only because the failure was a decreasing handle count and the repaired test
  still exercises repeated lifecycle and both wake-boundary regressions. It
  does not authorize current-session resume or production arming.
- Use historical qualification root
  `1FB073478EE24C3244503FE22AB4F7B898F3623ED0303E66C9E14D05E3073D5F`
  for this exact foundation inventory. Retain `00231CBF...` only as a declared
  diagnostic alias from a different canonical-record schema; never compare
  unlabeled roots from different preimage policies.

## 2026-07-31 - Wake only after the exact root task-complete event

- Arm the current-session handoff against the stable first `session_meta`
  record, the exact root JSONL physical identity, its append offset, exact
  current `turn_id` and a one-use marker included in the final answer.
- Hash the first record from shared raw file bytes, excluding its single LF
  delimiter. Do not use PowerShell text decoding/re-encoding as provenance;
  the authoritative first-record identity is 41,079 bytes and SHA-256
  `5BDD7B37A5466F264719C7968F054F4966CC911D493CA4AF16DF5971C5426E7D`.
- Wait event-driven with `ReadDirectoryChangesW`. Require, in order, an
  assistant `phase=final_answer` record and a subsequent
  `payload.type=task_complete` record with the same turn ID and marker. The
  second record, not elapsed time or an assumed delay, is the legal idle
  boundary for the single resume attempt.
- Any replacement, truncation, malformed line, wrong turn, missing marker or
  restart before the terminal event writes STOP. It cannot automatically wake
  or retry.

## 2026-07-31 - Preserve exact-three dependencies with a two-scope source composite

- Keep the reviewed technical `src/**` authority distinct from the wider
  fresh-180 capture. Neither root may be copied, renamed or substituted for
  the other.
- Preserve the FINAL2 dependency count of three by making the existing
  execution-source manifest/records pair a closed composite with two explicit
  subrecords. Python runtime identity remains the third role.
- Bind both inner manifests, roots, scopes, exclusions and counts plus the
  ordered composite root. Q must validate its old technical fields and new
  nested fresh-180 fields against the appropriate subrecords before imports or
  material construction. This decision does not authorize publication.

## 2026-07-31 - Separate synthetic and operational handoff capabilities

- A capability-disabled synthetic authority is immutable evidence only. Never
  edit or relabel its `production_arm_enabled=false`,
  `real_resume_enabled=false`, `synthetic_only=true` values.
- A real handoff may use only a separately frozen and independently audited
  operational authority with the inverse exact profile, exact current-session
  and CLI pins, and the same closed no-retry contract. A Q or E crosslink must
  reject substitution between the profiles.
- Bound the growing session log at 1 GiB rather than the superseded 256 MiB
  draft because the live file was already above 200 MB before arming. Exceeding
  the bound is durable STOP, not rotation, cleanup or retry.

## 2026-07-31 - Persist handoff base in Q and the one-use attempt in E

- The sealed request carries two distinct closed envelopes. Q records and
  crosslinks the complete base authority; E records the complete arm-specific
  attempt plus the exact Q base-payload hash.
- Keep physical staged E custody separate from the future final supervisor job
  and receipt custody. Neither authority may infer the final job directory
  from the staged E parent or pre-create it.
- Exact prompt text and substitutions, argv, session native identity,
  pre-arm prefix, watcher registration and same-turn terminal event are data
  authority, not hidden source constants. Unknown, missing or mismatched fields
  fail closed before a wake.

## 2026-07-31 - Hold release-tools v3 until its regression is end-to-end

- Passing ordinary tests and a plausible read-share implementation are not
  enough to close the previously demonstrated finalization race. The
  regression must run the complete waiter, use a deterministic barrier at the
  final-scan/rename boundary, and assert both immutable committed bytes and a
  non-success disposition for every ambiguous mutation case.
- Preserve v3 root `0DA52569...E1` as HOLD evidence. Make the enhanced proof in
  a separate v4 successor and require a new unchanged-byte audit before
  admission.

## 2026-07-31 - Admit the exact v4 release tools after end-to-end proof

- Admit exact8 release-tools root
  `BDB49545BAA6796C9FE9ADBF40AB56A0CF7CF87CC00EA1540F8F18D911FD1226`
  after an unchanged-byte P0/P1/P2 = 0/0/0 audit and the complete waiter
  barrier regression. Keep v3 and the original failed v2 immutable as
  superseded evidence.
- For an external WIP, run Ruff with the repository's explicit
  `pyproject.toml`; a default Ruff configuration discovered from a different
  working directory is not the AANCA formatting gate. This does not waive the
  format gate: v4 passed the canonical configured command without edits.
- Admission does not authorize a release publication or consume Q/E.

## 2026-07-31 - Admit the repaired production-disabled release-input builder

- Admit exact4 root
  `6DD777F533E8ED3952AF11E60D513123FF2C034FFA446B544E32D7DEA7323552`
  as the synthetic-only FINAL2 projection/qualification-input builder after
  independent 0/0/0 review.
- Require physical canonical readback of both inner source manifests and their
  composite, and treat FINAL2 paths only as bytes to hash. Never dynamically
  import or execute caller-selected release modules during input construction.
- Keep production hard-disabled until final Q, E/supervisor, five formal
  qualification receipts, Python runtime identity and release pins all exist.

## 2026-07-31 - Supersede concrete prelaunch attempt-in-E with creation authority

- This decision explicitly supersedes the earlier requirement that E contain a
  fully rendered arm-specific attempt, live offset, prefix, prompt and argv.
  Those values do not exist when immutable E is staged; retaining that rule
  would require fabricated future evidence.
- The sealed request contains exactly two handoff envelopes: full operational
  base and full attempt-creation authority. Q persists the full base plus the
  creation payload hash outside the Q-base preimage. E persists the full
  creation envelope plus exact Q base and creation-hash crosslinks. Do not add
  redundant E payload/path aliases.
- After unchanged JOB custody reaches COMMITTED, the replacement supervisor
  registers the watcher before snapshot, records the real offset/prefix/native
  identity and CREATE_NEW-publishes the concrete attempt at the fixed
  job-scoped output. Only postterminal `wake_intent` selects success versus
  diagnosis and binds the exact rendered prompt, substitutions and argv before
  the sole Codex spawn.
- This is an operational evidence-timing amendment only. It changes no frozen
  scientific definition and does not require editing `SPEC.md` or frozen
  `PRE_REGISTRATION.md`.

## 2026-07-31 - One operational supervisor role owns both wait and wake

- Replace the existing release role `option_a_supervisor` with the integrated
  operational successor. Preserve the existing six-pipe JOB custody handshake;
  do not add a ninth runtime role, a second wrapper wake path or a new handle
  transport protocol.
- Operational SPEC52 requires `codex=null` and an exact10 external-handoff
  reference to sealed E and the two output receipts. Internal legacy Codex wake
  and `handoff_session.json` are prohibited, not renamed.
- Interpret the one-hour bound only as the same-turn completion-boundary
  capture deadline. Once the concrete attempt/idle receipt is durably sealed,
  it remains evidence while the bounded long process runs. Apply separate
  one-hour postterminal intent and six-hour Codex-process bounds.
- Map only exit 0 plus every required seal, integrity check and expected
  artifact to `QUALIFIED_SUCCESS`. Map failure, loss, ambiguity or any mismatch
  to durable STOP and the diagnosis-only branch. No branch permits automatic
  scientific retry.

## 2026-07-31 - Runtime identity is a fresh external dependency

- Use the qualified builder's retained-handle native identity for the exact
  `.venv` interpreter. Keep its canonical root outside the release and require
  it before Q as the third FINAL2 external dependency.
- A diagnostic root is useful for integration, but it is not a formal receipt
  or permanent pin. Rebuild and compare it immediately before final release/Q;
  any byte, native-file-identity, version or architecture change is STOP and
  requires a new qualification input, never silent adoption.

## 2026-07-31 - Admit only the disabled qualification-receipt builder bytes

- Admit exact4 qualification-receipt-builder root
  `17451D819FC03C626D1B738279DAEB8942A7E081CF5B4CBF18650AA920A2743E`
  after an unchanged-byte P0/P1/P2 = 0/0/0 audit. This admission covers only
  the synthetic builder implementation and its closed schemas/tests.
- Do not treat builder admission as a qualification receipt. Production
  enablement, receipt issuance, persistence, release publication, process
  launch and automatic retry remain forbidden until the final exact component
  roots and evidence are supplied through a separately authorized invocation.
- The absent optional JSON-Schema meta-validator is a recorded tooling
  limitation, not permission to weaken the schema gate: the built-in tests,
  Draft 2020-12 parse and independent exact-field/closed-object inspection must
  remain green when the final receipt set is built.

## 2026-07-31 - Distinguish the session-origin CLI from the resume executable

- Preserve the CLI version in the session's first record as origin evidence,
  but separately bind the exact runnable native executable, size, SHA-256 and
  `--version` stdout that will perform the one resume. Do not infer that the
  two versions are equal and do not execute a PowerShell/CMD wrapper under the
  no-shell handoff policy.
- A readable file is not sufficient program authority. An app-packaged binary
  that cannot be directly executed by the supervisor context is ineligible.
  Final admission requires a direct no-shell launch and then the authorized
  one-shot synthetic resume test against the exact saved session ID.
- A compatibility failure is durable STOP and diagnosis; it cannot trigger a
  CLI substitution, `--last`, a second resume attempt or a scientific launch.

## 2026-08-04 - Preserve authoritative storage and defer capacity admission

- Treat the two approximately 43-GiB primary/recovery run trees and the raw
  PanNuke tree as authoritative immutable inputs. Do not reclaim capacity by
  deleting, recompressing, moving, hardlinking or otherwise rewriting them.
- Continue source integration and synthetic QA with the current 39.61-GiB free
  space. Before original confirmatory, recompute required active storage from
  the final sealed manifest and require that amount plus 10 GiB. A failure at
  that later gate is STOP and requires additional user-authorized storage or a
  separately qualified non-scientific storage policy; it is not permission to
  weaken the scientific plan or overwrite historical evidence.

## 2026-08-04 - Apply the frozen 70-GiB capacity threshold exactly

- Interpret the original-confirmatory storage gate as the existing exact
  75,161,927,680-byte (70-GiB) minimum in preflight and T0, which supersedes
  any informal estimate based only on one active tree plus a 10-GiB margin.
- Do not create capacity-v2, T0, final Q, E or a scientific attempt while the
  live capsule volume is below that threshold. Continue nonpublishing source
  integration and synthetic qualification work because it cannot consume the
  one-use scientific authorities.
- Capacity may be restored only outside immutable raw PanNuke and historical
  run/evidence trees unless a separately reviewed storage policy explicitly
  authorizes another action. No capacity observation may be reused after a
  material time or filesystem-state change; recheck immediately at each gate.

## 2026-08-04 - Admit the supervisor and launcher bytes without arming them

- Admit supervisor exact-eight root
  `0EC92D749EBCFC4010C11D6B3AD2C94AEB3373AADF588F467C6CEF68CE9C82A8`
  after owner QA and a separate unchanged-byte P0/P1/P2 = 0/0/0 integration audit.
  Admission covers the event-driven, no-retry implementation and authority schema;
  it does not install startup recovery, arm a job or authorize a Codex resume.
- Admit terminal-launcher exact-four root
  `B42FE549796A07D36FE3F32307703E3B281E1DD66AD6E355959F6A6FD432E2B8`
  as a production-disabled base after independent P0/P1/P2 = 0/0/0 review. Keep its
  sole materialization-authority path/hash unset until the final upstream capsule,
  T0, lifecycle, STATIC-v3, runner, terminal and release evidence exists.
- A source qualification is not an operational launch permission. Both components
  remain `HOLD`; no real process may use them before the later immutable release,
  exact-session compatibility gate and one-use Q/E chain pass.

## 2026-08-04 - Permanently reject issuer root 97916518

- Reject exact-five root
  `97916518222FB461D85A6C7E03209B96281DD08745A923305E0BA42ACE3BFB3B`.
  Ordinary passing tests do not override the independent P1 proofs: output custody
  ended before terminal success, Python equality admitted type aliases and a
  normalized report root, and three nested schema objects remained open.
- Preserve the rejected bytes as diagnostic evidence. Implement the repair only in
  a new successor: retain every output-leaf handle through the terminal receipt and
  final full-set revalidation; hash the exact raw unsigned report; require recursive
  exact JSON types; close every object schema; and add deterministic mutation/type/
  schema regressions. A new root requires fresh full QA and independent audit.
- Keep production issuance compile-time disabled throughout. No candidate receipt,
  even from a clean synthetic dry-run, is a formal production qualification record.

## 2026-08-04 - Do not reclaim unrelated storage automatically

- The read-only audit identified enough potential external capacity in Downloads or
  the unrelated Hostinger migration repository, but choosing or deleting those files
  is outside the scientific implementation step and may destroy private or
  non-reconstructible data. Do not infer deletion authority from the 70-GiB gate.
- Do not manually prune `.git\objects` or `.git\lfs`. If the Hostinger repository is
  selected later, first verify the exact remote and LFS objects and prove a clean
  restore/clone; prefer removing a complete explicitly selected redundant copy over
  partial object-store surgery.
- Continue nonpublishing integration and synthetic QA. Reobserve physical free space
  only after those suites end, and require at least 75,161,927,680 free bytes at each
  real T0/Q/E/scientific boundary.

## 2026-08-04 - Freeze Q20 and its staged-E authority parity without consuming Q

- Retain Q20 exact-four root
  `65F444510A345CE63927D53F0FB62291B49A6206D58DB892D0A1E4E33416E02C`
  and staged-E authority SHA-256
  `8C09D2ABD97EA7A8C250C1D038A6F9C08DE6779B6744943AADA307ADD703F3DA`
  as the unchanged source candidates that passed the one permitted focused parity.
- Do not repeat the parity merely because its original tool-output cell was lost;
  the durable session JSONL supplied the terminal exit and output, and before/after
  hashes prove the inputs were unchanged. A repeat would add no authority.
- This source/parity freeze does not consume the user's one Q replacement-v2
  authorization. Keep final production pins absent and both paths `HOLD` until the
  external release, formal qualifications, capacity/T0 and downstream custody gates
  are complete.

## 2026-08-04 - Reject ambient-dependent E authority and provisional adapter roots

- Reject authority SHA-256
  `8C09D2ABD97EA7A8C250C1D038A6F9C08DE6779B6744943AADA307ADD703F3DA`
  and adapter root
  `50E0D98BB85C764E9DA91F689DE3C3FDB17C8C806E796B5F08800B51182B7C30`
  for production integration. Green ordinary tests and parity cannot override the
  demonstrated dependence on ambient `USERPROFILE` inside the pure builder path.
- Preserve both candidates unchanged. In a successor, derive the permitted profile
  root exclusively from the sealed runtime ancestor lease and reject an inconsistent
  sealed relation. Tests must make HOME/USERPROFILE/PATH absent and adversarial while
  observing zero ambient reads and identical canonical output.
- The release inventory must be an actual exact allowlist. Exclude every cache/PYC
  file from the new successor before its root is declared; do not relabel a full
  directory containing cache as exact11.

## 2026-08-04 - Reject issuer root 5b3ddf3f and move final Q after T0

- Reject issuer successor root
  `5B3DDF3FBA7B06DBA525AAB10A996D29A089BA9FC748F96E0F5288F7EB431E27`.
  It repaired the earlier output/type/schema defects but still cannot produce a legal
  production preassembly because of three P0 integration contradictions.
- A later issuer must keep `qualification_output_root` distinct from the untouched,
  existing and exactly empty final release-control root. It must bind the latter in
  the waiter exact17 record while writing its ten leaves only beneath the former.
  Qualification input, output, final control and supervisor state roots must be
  pairwise disjoint with alias/reparse/ancestry checks.
- Digest strings are not dependency evidence. Retain-handle verify the technical and
  fresh execution-source manifests, their ordered composite/records receipt, and a
  fresh formal runtime-identity receipt plus exact interpreter. Bind their physical
  paths, bytes, native identities and canonical roots into the issuer terminal receipt.
- Do not qualify the current Q20 template as final Q. A final-Q source successor with
  all upstream `FINAL_*` pins fixed is created only after capsule/T0, lifecycle/static,
  execution-source, terminal-launcher, supervisor/handoff and session-compatibility
  evidence exists. Keep only final release root/manifest/exact57 as request-time
  values to avoid the self-hash cycle.
- Consequently, capacity/T0 precedes final Q, formal qualifications and production
  release. Continue only source-level repairs and the release-independent short
  current-session handoff gate while free space remains below 70 GiB.

## 2026-08-04 - Build a separate compatibility-only resume authority

- Do not use the synthetic `false/false/true` handoff authority or the Q/E-bound
  science wake path to test the current Desktop session. They have different trust
  domains and cannot be relabelled.
- Before the one real resume, qualify a separate external bundle with operational
  source inventory and audit evidence, an exact live-verified operational-base
  builder, and a compatibility-only watcher/launcher/verifier. Keep its runtime
  enablement false throughout source construction and audit.
- Bind the exact session ID, retained JSONL native identity and first raw record,
  direct Codex executable path/size/hash/live version, closed resume environment,
  stable supervisor/spec hashes and the `true/true/false` capability profile.
- The later one-use state must be CREATE_NEW. Arm `ReadDirectoryChangesW` before the
  final-answer offset/prefix snapshot; require matching same-turn `final_answer` then
  `task_complete`; create wake intent and attempt before one direct explicit-session
  spawn; persist logs/result/readback. Missing or ambiguous evidence is permanent STOP
  and never permits `--last`, an alternate CLI or a second attempt.

## 2026-08-04 - Admit the ambient-independent staged-E authority and adapter bytes

- Admit staged-E authority exact-four root
  `53B166F4C437802B8879ECAB39281FFB9DD0B7B7D4C4A0252C06975DD3760A39`
  and Q-to-E adapter exact-eleven root
  `422BEC399FA93252E61A5EBC16E45723386264100E3CF21FC3CC9299034FD0ED`
  after owner QA and a separate unchanged-byte P0/P1/P2 = 0/0/0 audit.
- These roots supersede only the rejected ambient-dependent authority/adapter
  candidates. Preserve the rejected bytes as diagnostic evidence; do not alias,
  overwrite or relabel them as admitted.
- Treat the sealed runtime-ancestor profile anchor as the sole permitted source of
  the profile relationship. Ambient `USERPROFILE`, `HOME` and `PATH` are never
  authority inputs; absence, adversarial values or a mismatching sealed anchor must
  remain fail-closed and covered by exact-output regressions.
- Admission is source-level only. It does not consume the one authorized Q write,
  stage E, arm the supervisor, resume Codex, publish a release or launch science.
  Production remains `HOLD` until the downstream capacity/T0, execution-source,
  current-session compatibility, final-Q and one-use release authorities all pass.

## 2026-08-04 - Reject issuer v3.1 and require four native-disjoint root domains

- Permanently reject issuer exact-five root
  `5D05513BFE3DDEAA3939CAF8CA311281BB11D046ECB03D41A421B84FFBF591C2`.
  Its exact10 output/readback and ordinary QA are retained positive implementation
  evidence, but they do not compensate for the independent P0/P1 findings.
- Qualification input, qualification output, final release control and supervisor
  state must be pairwise ancestry-disjoint in both directions. For every existing
  directory, retain a native handle and compare volume/file identities in addition to
  canonical path/commonpath checks. The absent output's retained parent must also be
  bound so alias, short-path, reparse or directory substitution cannot collapse trust
  domains. Tool/input containment is likewise symmetric.
- Snapshot the closed request into a deep canonical object before opening evidence.
  Never retain or reread a caller-owned dict/list. A deterministic barrier that mutates
  the caller after snapshot must either leave the canonical result unchanged or stop;
  it may never alter emitted paths, records or hashes.
- Q source qualification must reject every additional binding or deletion of a fixed
  `FINAL_*` name, including import aliases, exception names, parameters, definition
  names, comprehensions, context managers, pattern captures and dynamic namespace/code
  mutation paths. Exactly one ordered top-level literal assignment per required pin is
  the only admitted binding form.
- Exact allowlists count every directory entry, not only regular files. Any extra
  directory, cache, PYC or unknown entry invalidates the release-tools or issuer audit.
  Implement these repairs only in a new v3.2 successor and repeat owner plus independent
  unchanged-byte QA. Do not create the live final control root or issue production
  evidence while capacity/T0/final-Q/release prerequisites remain incomplete.

## 2026-08-04 - Bound the v3.3 AST repair to direct import injection

- Permanently reject issuer v3.2 exact-five root
  `DC6D95FA4A3050E79E3BAE0100023703356FC49F04E7ABCD3CD12D0D73F6D9A7`.
  Its root-separation, custody, deep-snapshot and exact-cardinality repairs remain
  useful evidence, but the accepted top-level wildcard import can directly replace
  the fixed global pins and violates the frozen import-injection boundary.
- Implement v3.3 as a new, production-disabled successor. Reject every
  `ast.ImportFrom` containing `alias.name == "*"`, retain all v3.2 regressions, and
  prove both fail-closed wildcard-import rejection and a passing canonical Q fixture.
  Require a new exact inventory, full owner QA and independent unchanged-byte audit.
- Do not turn the validator into a Python sandbox. Deliberate reflective mutation by
  already trusted, sealed code remains outside scope exactly as PLAN.md lines 464-467
  state. This boundary is the terminal rule for the audit: only a demonstrated
  in-scope substitution, import injection, ordinary binding or custody failure can
  block v3.3.

## 2026-08-04 - Admit issuer v3.3 bytes but keep production issuance on HOLD

- Admit exact-five root
  `FCD62EA48CDED74E62F6D45377DF265E419862839487152D2E71E95BF625037E`
  after complete owner QA and an independent unchanged-byte P0/P1/P2 = 0/0/0 audit.
  The decision is limited to the exact production-disabled source allowlist.
- Treat unconditional `ImportFrom("*")` rejection as the complete in-scope repair for
  the v3.2 wildcard-import finding. Preserve the rejected v3.2 bytes and its evidence;
  do not overwrite or relabel that root.
- Keep `PRODUCTION_ISSUANCE_COMPILED=False` until capacity/T0, compatibility-resume,
  final-Q and release prerequisites exist. Source admission must not be interpreted as
  permission to write a qualification receipt, consume Q/E authority or launch a
  scientific process.

## 2026-08-04 - Admit only the disabled compatibility-resume foundation

- Admit exact-seven root
  `6CE26CBE8E35E53B4E00A8A99F1567B20E61C354ECF59484437369BFE7992ADA`
  only as `QUALIFY_DISABLED_IMPLEMENTATION / HOLD_ENABLEMENT`, after owner QA and an
  independent unchanged-byte P0/P1/P2 = 0/0/0 audit.
- Require all state, STOP and log readbacks to parse bytes obtained from a retained
  no-follow handle; a prior path reopen after handle verification is ineligible even
  when its hash check would usually fail closed. Reject direct wildcard imports in
  the static enablement-flag verifier.
- The admitted root deliberately has
  `COMPATIBILITY_RESUME_ENABLED=False`, `SYNTHETIC_STUB_TESTING_ENABLED=True` and
  `EVENT_DRIVEN_ARM_ACK_IMPLEMENTED=False`. It cannot be enabled by editing in place.
  Create a new successor/root that physically implements an event-driven watcher-arm
  acknowledgement, obtains a fresh audit, and only then materializes one-use
  enablement authority for the single exact-session compatibility attempt.
- Do not run the current disabled bundle as evidence for resume compatibility and do
  not infer permission for science, Q/E, release or retry from this source admission.

## 2026-08-04 - Limit capacity cleanup to exact64 and treat the pass as transient

- Authorize and record removal only of ordered allowlist SHA-256
  `BEACB20E10FBC9B6BA31C629EAE1619961D895879C13676FC534F56442B7A26E`
  after duplicate read-only qualification, immediate root/process revalidation and a
  single native PowerShell run. Terminal evidence proves 64/64 unique removals.
- Preserve the repository, environment, raw PanNuke, historical runs, `C:\pt3`, all
  external authorities/WIP, the volatile `pytest-of-NATAN` root and unrelated user
  data. Do not infer a broader deletion policy from the exact64 operation.
- A free-space observation above 75,161,927,680 bytes is time-local. The current pass
  has less than 1 GiB headroom; repeat the exact live capacity check after any QA or
  filesystem change and at every T0/Q/E/science boundary. Never consume one-use
  scientific authority on the basis of this historical observation alone.

## 2026-08-04 - Build and qualify the carrier before live authority integration

- Do not integrate the admitted Q20/E23/SPEC52 authority into a live bootstrap and
  terminal that still implement Q/E v2 and `SUPERVISOR_V2_POLICY`. Partial copying is
  a contract contradiction, even when every copied source hash is independently
  admitted.
- First build a new external, production-disabled carrier successor containing the
  bootstrap/terminal changes and focused regressions for Q20, E23, SPEC52,
  `external_codex_handoff` and `SUPERVISOR_V3_POLICY`. Require exact inventory, full
  owner QA and an independent unchanged-byte audit before any repository write.
- Then use one integration owner to copy only the qualified mapping into live repo:
  authority source/test, new Codex-handoff test, carrier bootstrap/terminal and their
  tests. Full live QA/CLI/PanNuke must pass before two-build capsule reproducibility or
  T0 begins.
- Preserve all historical capsule/T0/lifecycle receipts as evidence but do not reuse
  them for the changed source root. Rebuild downstream inventory, capsule, T0,
  lifecycle/STATIC, launcher and final pins in dependency order; this does not permit
  Q/E/resume/science before their later one-use gates.

## 2026-08-04 - Reject in-memory-only serialization of watcher ACK versus STOP

- Do not freeze or enable transient arm-ACK exact-six root
  `AB36A0F2F64FE638C82D8705C8897456096B2E81F577AA5E01F67E02F70913AF`.
  Passing ordinary tests is insufficient when another durable STOP writer can win
  before final ACK publication without participating in the in-memory latch.
- Require one CREATE_NEW durable winner token before final publication. Every
  pre-ACK failure must first claim `ABORTED_PRE_ACK`; the launcher must first claim
  `ACK_CLAIMED`, then perform fail-if-exists receipt promotion and retained exact
  readback, and only then signal in-memory `ARMED`. `ACK_CLAIMED` without the exact
  final receipt is terminal STOP evidence and never permission to retry or resume.
- A pre-existing canonical STOP blocks ACK; a malformed STOP is ambiguous and also
  blocks ACK. A later post-ACK STOP is legal only when it binds the exact claim and
  final receipt and remains dominant during state verification.
- Accept an already-exited boundary worker only when synchronized state contains a
  complete boundary and no worker error. Bound cancellation independently of a
  potentially blocking backend callback. Require deterministic regressions for both
  STOP races, immediate completion and blocking cancellation before a new audit.

## 2026-08-04 - Admit the frozen coherent carrier and permit only its exact-nine map

- Admit the production-disabled carrier governed exact-18 root
  `E9A740AA99AD4818FBC40169316F84F79A7C06FF769DD9A2AC5349160154B462`
  and frozen exact-19 seal root
  `0A76A4C048BDA498F8A26847C44FB4A2660E2B3D96D2F66203D0766EC2067BE0`
  after two independent P0/P1/P2 = 0/0/0 audits and complete owner QA.
- Permit one integration owner to apply exactly the nine mapped leaves bound by
  mapping self-root
  `A3B4B13224B75B4EB6066EB7BC6D6E5F3545C39E65361AAF8AFB4C72A7630DD8`.
  Re-read all six exact-file and three required-absence preimages immediately
  before the write, require all nine postimages to equal the carrier pins and
  stop on any drift. Partial integration is forbidden.
- Never copy the seven reference-only leaves. They exist only to qualify imports,
  type checking and immutable upstream pins inside the external carrier.
- The bounded one-node replay after the independent combined suite encountered a
  Windows 262-character `MAX_PATH` is valid only because the same unchanged bytes
  passed the exact node under a verified-empty short basetemp. It does not waive any
  code, scientific, integrity or production gate.
- Keep production disabled. Admission and integration permission do not authorize
  T0, Q, E, resume, publication or science. After the exact-nine write, require
  focused live tests, full `pytest`, Ruff check/format, strict mypy, compilation,
  the expected direct-confirmatory fail-closed CLI and the real PanNuke validator
  before advancing.

## 2026-08-04 - Release only exact38 old physical pytest roots under C:\pt3

- Supersede the earlier blanket preservation of `C:\pt3` only for the 38 physical,
  non-reparse July-21 pytest roots in the ordered UTF-8/LF allowlist with SHA-256
  `72CDCCA58C073ED9CD721239452EF877AB7F0AB8C552086B055E737AC78D00CF`.
  All other `C:\pt3` entries remain preserved, including every top-level
  `*current` alias and the recent carrier-audit directory.
- Two independent read-only inventories reproduced exactly 38 direct physical
  candidates, 2,217,387,020 logical bytes and about 2.25 GB of unique NTFS
  allocation. They were created by the recorded short-basetemp run on 2026-07-21;
  their contents are synthetic confirmatory/freeze/checkpoint test outputs.
- The inventories found zero active Python/pytest process or command-line reference,
  zero external hardlink or reparse target and zero scan error. The one hardlink pair
  and one nested symbolic link are wholly contained in their respective candidate.
- Permit one native PowerShell deletion only after immediate reconstruction of the
  exact list hash, direct-child/non-reparse checks, the exact logical total and zero
  process references. Resolve every literal path and require its parent to be exactly
  `C:\pt3`; delete nothing computed outside the hard-coded allowlist. Any mismatch is
  `HOLD` and performs no deletion.
- This cleanup is capacity maintenance only. It does not alter repository sources,
  raw PanNuke, historical real runs, authorities, WIPs or scientific evidence, and
  it grants no standing permission to clean other paths.

## 2026-08-04 - Admit only the frozen event-driven arm-ACK source profile

- Admit exact-seven seal root
  `41E7DD4A70D49D3B4800F6F1F344D3F2F74333020FC76C1C2515AFEC156EF798`
  after owner QA, unchanged-byte audit and post-freeze P0/P1/P2 = 0/0/0 audit.
  The underlying exact-six material root is
  `E483C4F99511E5128B76334350CC77C9A424F9E7F789E03F83BAE21095630C0A`.
- Admission covers the production-disabled source profile only. It proves the
  production runner can reach `EventDrivenBoundaryWorker`, serializes durable
  ACK versus STOP, validates the five persisted envelopes and exact nested types,
  and enforces the complete chronology. It does not itself materialize authority
  or authorize a real resume.
- Keep `compatibility_resume_enabled=false` and
  `production_materialization_enabled=false` in the frozen bytes. A later one-use
  materialization must bind the exact current session/base/turn and may run only
  after the live carrier/capsule/T0/lifecycle gates. It remains one attempt, no
  `--last`, no retry, no science and permanent STOP on ambiguity.
- Preserve rejected roots `5918E7C8...` and `89A10CAC...` as diagnostic evidence;
  never relabel their chronology or type-coercion findings as admitted behavior.

## 2026-08-04 - Accept the live carrier integration and isolated Windows test oracle

- Accept the single exact-nine integration after immediate postimage readback and
  a separate P0/P1/P2 = 0/0/0 live-origin audit. The only post-carrier test delta
  is two `SOURCE_ROOT` to `PACKAGE_IMPORT_ROOT` reads required by the repository
  `src` layout; production bytes and contracts remain exactly admitted.
- Treat the first full-suite 5-failure result as a mandatory failed gate, not a
  waiver. The failures were resolved only after exact isolated replay and an
  independent controlled-GC experiment proved the process-global handle count was
  a non-isolated oracle. Require exact owned CloseHandle identities/cardinalities
  and a live foreign sentinel instead. This strengthens ownership verification and
  changes no production source.
- Preserve the second full result as the qualifying gate: 2,387 passed, one
  documented Windows skip, zero failed. Keep Ruff lint active on the immutable
  authority, but exclude that one pinned file only from formatter rewrites.
- The project-configured mypy result is authoritative for this gate and passes 100
  source files. Retain the additional `--strict` 27-finding output as transparent
  technical debt; do not broaden M8 into unrelated scientific refactors merely to
  silence a stricter optional profile.
- The real PanNuke validator and direct exit-2 CLI are qualifying live evidence.
  They authorize rebuilding fresh downstream execution-source/capsule/T0 evidence,
  not Q, E, resume, publication or science. Formal project status remains exactly
  `PRIMARY_STUDY_COMPLETE`, M8 remains 8/10 and M9 remains locked.

## 2026-08-04 - Admit only fresh candidate reproducibility, not publication

- Treat the 108-entry source inventory rooted at
  `BDDCB13B71573A2366C1D8050B5BFEF7A8D6B783098277F93F48056C77A83C35`
  and the two byte-identical candidate capsules at
  `E87EFDD814FB5916A76EADA23E478BD2B13A074FFC3516F882D72EDDCC271E90`
  as fresh non-publication evidence. They do not inherit eligibility from either
  rejected July candidate and do not alter those retained artifacts.
- Require an unchanged-byte independent audit before using this exact candidate
  identity downstream. After admission, proceed only in acyclic order through
  reviewed content-addressed CREATE_NEW capsule publication, distinct-process
  independent readback, a fresh post-readback capacity-v2 receipt, T0
  publication/verification, lifecycle readiness, STATIC-v3 and launcher
  qualification. Stop on any changed source, candidate or receipt.
- This gate consumes no Q or E authority and grants no permission to resume,
  publish results or execute science. Keep the formal status exactly
  `PRIMARY_STUDY_COMPLETE`, M8 at 8/10 and M9 locked.

## 2026-08-04 - Require external one-use capsule publication tools before T0

- Accept the independent A/B verdict at P0/P1/P2 = 0/0/0, but keep the real
  publication boundary on `HOLD`. The hardened builder API alone is not a
  qualifying workflow because T0 also requires a publisher receipt carrying
  real process identity and a later receipt from a distinct process and
  distinct implementation.
- Build those implementations outside the repository so their creation does not
  invalidate the admitted 108-entry capsule inventory or 114-entry execution
  source. They must be production-disabled during synthetic QA, CREATE_NEW-only,
  no-adoption/no-overwrite/no-cleanup, one attempt, no automatic retry, no
  outcome-value access and no scientific execution. Freeze their exact paths,
  sizes and SHA-256 values and require an independent unchanged-byte audit before
  the first real invocation.
- Create the final capacity-v2 receipt only after the independent capsule
  readback. The live verifier's chronology
  `published < verified <= capacity_checked <= T0 intent` is authoritative;
  a pre-publication disk observation is only a preflight and can never be
  relabelled as the receipt.

## 2026-08-18 - Reduce the deliverable to the presentation MVP

- Accept the owner's explicit instruction to switch to an MVP. Preserve the
  original `SPEC.md`, `PLAN.md` and frozen `PRE_REGISTRATION.md` byte-for-byte;
  record the reduced presentation boundary separately in `MVP_SCOPE.md`.
- Use only the already accepted `PRIMARY_STUDY_COMPLETE` run and validated
  PanNuke QC. Do not run another training, confirmatory attempt, Q/E process,
  publication, recovery or session-resume process for MVP completion.
- Defer original confirmatory, M9 expert-review/original-label work, external
  validation and their production control-plane infrastructure as future work.
  Do not claim those PLAN milestones or completion stages.
- Permit `DEMO_COMPLETE` only for the static non-diagnostic presentation after
  its tests, Ruff gates and real build/verify pass. Keep the scientific result
  boundary `PRIMARY_STUDY_COMPLETE` and the accepted primary disposition
  `amended_or_exploratory`.
- Require the presentation builder to verify selected source hashes, run seal,
  positive stage attestation and QC policy; include all 36 comparisons; refuse
  overwrite; emit a closed checksum manifest; and never modify source data,
  labels, frozen files or historical runs.

## 2026-08-18 - Require complete H1-H7 presentation evidence

- Preserve the sealed accepted primary and its `amended_or_exploratory`
  disposition. Do not rerun, replace, or tune a scientific result after outcome
  inspection merely to improve the presentation.
- Require presentation schema v2 to include the complete descriptive H2 subgroup
  summary and the registered H4 downstream comparison in addition to all 36
  H1/H3/H5/H6/H7 comparisons. H4 is adverse to the registered hypothesis and must
  be shown prominently rather than treated as an optional appendix.
- Verify the ranking and OOF artifacts for the registered 10% instance-dependent
  ImageNet-context logistic cells at seeds 404, 405, and 406. Because both artifact
  types are byte-identical across the three cells, retain their frozen rows but
  treat them as one deterministic realisation, never as three independent
  replications.
- Label Holm-adjusted p-values explicitly as one-sided and keep the saved 95%
  bootstrap interval visible. Do not conceal cases in which the interval and the
  directional p-value invite different shorthand interpretations.
- Keep the output allowlist at five files and preserve CREATE-NEW builds. Track only
  the canonical `artifacts/mvp_demo` package in Git as a portable handoff exception;
  continue ignoring licensed data, embeddings, full runs, and scratch demo builds.

## 2026-08-18 - Adopt an editorial scrollytelling presentation system

- Combine the supplied Linear-style token guidance with the centered editorial
  pacing of the owner's reference article, while retaining AANCA's own identity,
  wording and evidence hierarchy rather than copying another site's branding or
  content.
- Keep all scientific visuals code-native and evidence-bound: render H4 downstream
  values, the H1/H3/H5/H6/H7 forest plot, H2 subgroup ranges, seed-integrity
  evidence and the complete comparison table directly from verified package data.
  Use the deterministic canvas field and workflow SVG only as explanatory interface
  graphics; do not represent them as microscopy data.
- Keep the adverse H4 result and the non-independent seed limitation prominent.
  Visual polish must never select results, alter the frozen interpretation or hide
  unavailable H6 cells.
- Make the scroll-linked sequence progressive enhancement. At narrow widths it
  becomes a static readable sequence; `prefers-reduced-motion` removes transition
  dependence; print output exposes the full story without sticky positioning.
- This is a presentation-only change. It does not modify `SPEC.md`, `PLAN.md`,
  `PRE_REGISTRATION.md`, source annotations, the accepted run or its scientific
  status.

## 2026-08-18 - Publish an English expert-facing GSAP and Three.js presentation

- Make all reader-facing copy English and identify Natan Smogór as author with a
  release date of 18 August 2026. Explain the benchmark first in plain language,
  then expose the exact design, statistical evidence and limitations expected by
  an expert reviewer.
- Apply the supplied Linear-style design rules through a near-black canvas,
  layered neutral surfaces, one lavender accent, compact Inter typography,
  restrained radii and generous section spacing. Keep the layout and wording
  specific to AANCA rather than copying the reference article's brand or content.
- Pin GSAP and ScrollTrigger at 3.15.0 and Three.js at 0.185.1. Use them as
  progressive enhancement for the hero, section reveals, chart marks, reading
  progress and synchronized method story; retain readable static, reduced-motion,
  print and network-failure fallbacks.
- Replace machine-like comparison labels with concise human-readable names while
  retaining every raw comparison ID for traceability. Render numeric table and
  forest-plot values at fixed six-decimal precision so small values do not wrap in
  scientific notation. On narrow screens, convert each row into a labelled card
  and require filtered rows to remain hidden despite the card display rules.
- Remove internal completion and disposition codes from visible presentation copy
  when they add no explanatory value. Preserve them unchanged in machine-readable
  evidence, verification output and governance documents.
- This decision changes presentation only. It does not modify scientific values,
  selections, hypotheses, source annotations, frozen authorities or the accepted
  run.

## 2026-08-19 - Make the hero animation explain immutable-source review triage

- Replace the generic nucleus field with a deterministic conceptual workflow:
  irregular nucleus contours stay inside a labelled source patch, the four-tile
  AANCA reticle focuses one candidate, five class-signal pulses appear, and a copy
  follows a visible path into a numbered six-slot expert-review queue.
- Keep the selected source contour in place throughout the sequence. The animation
  communicates ranking and recommendation only; it must not depict label deletion,
  correction, diagnosis or automatic source-data modification.
- Label the scene as conceptual workflow rather than benchmark data. Treat Three.js
  as progressive enhancement, render a static completed queue when reduced motion
  is requested, and keep the mobile composition below the hero copy without
  horizontal overflow.
- In reduced-motion mode, show every method-story step and every workflow stage at
  full opacity. Motion preference must never make earlier explanatory content look
  unavailable or subordinate to the final step.
- This is a presentation-only change. It does not alter evidence, metrics,
  experiments, source annotations, frozen authorities or the accepted primary run.

## 2026-08-19 - Adopt a professor-facing editorial article and cumulative benchmark story

- Present AANCA as a long-form English research article with a centered reading
  column, restrained display type, generous chapter spacing and wide evidence
  figures. The interaction rhythm may be informed by Anthropic's recursive
  self-improvement article, but the visual language, diagrams and scientific copy
  remain specific to AANCA and the supplied Linear-style design rules.
- Explain the benchmark as one cumulative five-stage serpentine sequence rather
  than five independent stacked cards. On desktop, a pinned split view keeps the
  growing diagram beside the active explanation; on mobile, reduced motion and
  print, all stages remain statically visible and readable.
- Hide the header on downward scrolling and restore it after any deliberate upward
  movement. When it returns, reveal the AANCA wordmark from the persistent mark.
  Remove the reading-progress rule beneath navigation; retain only active-section
  colour as orientation.
- Keep the complete H1-H7 interpretation, seed identities, QC evidence and 36-row
  comparison table visible without disclosure controls. State explicitly that 33
  comparison entries are numeric and three H6 entries are unavailable, and refresh
  ScrollTrigger geometry whenever filters change the table height.
- Include a repository card, three progressively deeper usage paths and a
  line-separated footer covering author, release, scientific boundary, responsible
  use, evidence links, licence limits and dataset terms. The root and packaged
  READMEs must describe the same presentation and verification workflow.
- Use GSAP and Three.js only as progressive enhancement. Core interpretation,
  navigation, evidence, responsive layout, reduced-motion layout and print output
  must remain usable without animation or network-loaded modules.
- This is a presentation and documentation decision only. It does not change any
  hypothesis, metric, annotation, accepted-run artifact, frozen authority or
  scientific completion stage.

## 2026-08-19 - Adopt dependency-free and visibility-aware presentation delivery

- Keep the existing CLI architecture because measured warm help and verification
  startup are already small; avoid a broad import refactor without a demonstrated
  bottleneck.
- Add `scripts/present_demo.py` as the default presentation path. It must use only
  Python's standard library, verify the closed package before serving, bind to
  `127.0.0.1` by default, support browser-free and verification-only operation,
  and allow port `0` for automatic free-port selection. Retain `histo-audit demo
  serve` as the equivalent command for an installed research environment.
- Require both verifiers to match the manifest records against the exact unique
  four-file payload allowlist before trusting any record hash. A correct file
  count alone is insufficient because duplicate record paths could otherwise
  leave one allowed file unchecked.
- Pause the Three.js render loop whenever the hero is outside the viewport or the
  browser tab is hidden. Use a low-power WebGL preference, cap high-density pixel
  ratios, honour data-saving and reduced-motion preferences, lazily decode the
  large QC image with explicit dimensions, preconnect the pinned CDN, and coalesce
  table-triggered ScrollTrigger refreshes.
- Keep generated QA iterations out of new Git commits while retaining the selected
  professor-release captures referenced by the project. Optimisation must not
  alter evidence, metrics, labels, frozen authorities, the accepted run or the
  scientific completion boundary.

## 2026-08-19 - Add remote integrity checks and harden local presentation delivery

- Add one deliberately small GitHub Actions workflow that runs the standard-library
  presentation verifier on every `main` push and pull request. Keep permissions
  read-only, disable credential persistence, cancel superseded runs and pin every
  third-party action to a reviewed full commit SHA. Do not install the large ML
  environment merely to verify the static presentation package.
- Treat the workflow as a fast remote integrity guard, not evidence that the full
  scientific suite ran. Material changes still require pytest, Ruff check, Ruff
  format check, mypy, dependency/lock checks and the relevant functional CLI.
- Serve the locally verified package with no-cache and browser-hardening headers
  and without disclosing the Python runtime version. Preserve loopback as the
  default and retain the exact package-verification gate before binding a socket.
- Use Natan Smogór consistently as the distributable package and presentation
  author. This is metadata correction only; it does not change study attribution,
  evidence or scientific status.
- Do not change the frozen torch/torchvision stack solely to force a newer
  Setuptools release. The known Setuptools advisory affects macOS sdist file
  exclusion; this project uses Hatchling, the presentation path is dependency-free,
  and torch currently constrains Setuptools below the fixed release. Record the
  constraint and unreachable path explicitly and revisit it when the governed ML
  stack is intentionally upgraded and fully requalified.
- Do not change GitHub repository visibility or grant third-party access as an
  inferred side effect of a code audit. Report the anonymous-access limitation so
  the owner can choose whether to make the repository public or invite the intended
  reviewer before presentation.
- Preserve `SPEC.md`, `PLAN.md`, `PRE_REGISTRATION.md`, the accepted primary run,
  source annotations and all scientific values byte-for-byte. These changes improve
  delivery, metadata, remote verification and auditability only.
