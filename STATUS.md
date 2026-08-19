# Project Status

## Current state

- **Completion status:** `PRIMARY_STUDY_COMPLETE`.
- **Current milestone:** M8 — controlled primary and confirmatory studies.
- **Date:** 2026-07-31 (Europe/Warsaw).
- This status means that the verified real PanNuke pilot, immutable preregistration
  freeze, and independently sealed/attested primary recovery are complete. It does
  **not** mean that the confirmatory study or external validation is complete.
- **Formal progress:** M0--M7 are closed, so 8/10 milestones = **80%**; **20%** of
  the milestone plan remains. M8 is active. Exactly one bounded recovery reused and
  checksum-verified all 185 required cells from the interrupted read-only orphan,
  skipped the 37 preregistered optional cells, retrained zero cells, and obtained a
  positive `PRIMARY_STUDY_COMPLETE` post-seal attestation. A broad read-only search
  accidentally exposed fragments of primary values at 2026-07-27T10:57:07Z, so
  `outcomes_inspected=true` and the affected recovered analysis is permanently
  `amended_or_exploratory`.

## Completed work

- Created the durable specification, milestone plan, status/decision records, draft preregistration, ethics statement, dataset setup guide, and a local Git repository without changing Git identity or creating a remote.
- Saved OS, Python, package, disk, RAM, GPU, driver, CUDA, dataset-discovery, Git, and write-access evidence in `reports/environment_initial.txt`, `reports/hardware.json`, and `reports/doctor.json`.
- Created `.venv` with Python 3.12.3 and locked the environment in `uv.lock`.
- Installed and verified PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, Cleanlab 2.9.0, and the scientific/QA dependencies. A 512×512 CUDA matrix multiplication and backward pass produced finite loss and gradients.
- Built deterministic grouped synthetic generation, exact target masks/crops/highlighting, engineered features, immutable corruption manifests, group-safe OOF models, seven audit rankings, Cleanlab integration, fold-safe neighbours, tied-budget random review, paired group bootstrap, downstream restoration, strict cross-artifact reconciliation, figures, Markdown/HTML reports, run logs, checksums, source-tree capture, sealing, and independent integrity verification.
- Persisted the complete synthetic arrays and source manifests, per-sample neighbour identities/groups/distances, every guided and random restoration selection/label/mask, and final-test probabilities for all four required downstream conditions. Every canonical report input is copied into the sealed run and reconciled before completion.
- Added artifact-backed PR curves, bootstrap intervals, class/tissue support panels, complete target/contour galleries, fold-safe neighbour explanations, and controlled false-high/false-low examples. Figure provenance retains hashes, sample IDs, selection rules, tie handling, and transforms.
- Added the explicit 0% corruption path: AP/AUROC/recall/lift are structured `not_applicable`, while score distributions and false-alert counts are reported.
- Implemented PanNuke discovery, anomaly-safe semantic validation, raw-file hashing,
  complete-fold enforcement, nucleus-manifest creation, exact/perceptual/ResNet
  duplicate auditing, component-covering target representation extraction, and a
  fixed real-data pilot runner. Canonical validation now handles release-level
  overlap and supplied-background voids without changing or arbitrating raw masks.
- Publication hardening now covers acquisition, validation/QC, the nucleus manifest,
  duplicate-audit reports/cache, and representation/reviewer caches with
  per-bundle/per-target locks, raw-path and collision guards, exact cross-cache
  lineage, ownership-safe rollback, and final readback. The completed M5 scopes passed
  their focused review and QA. The later M7 descriptor-anchored cache publication,
  strict concurrency review, full global QA, and immutable freeze have also passed.
- All three reported official Warwick PanNuke folds are present under
  `data/raw/pannuke`. The user supplied the acquisition, server-size, initial local
  SHA-256, CRC, safe-path, and extraction evidence. The independent working audit has
  since reproduced the local archive identities and ZIP safety results. The dedicated
  schema-v2 acquisition manifest and its bound report are now the checked M5 authority
  for acquisition provenance; semantic QC and the remaining M5 gates are separate.
- Completed an independent read-only, full-release PanNuke mask audit over all 7,901
  patches and 517,799,936 pixels. It reproduced the overlap/void counts, identified
  1,411 overlap-touching nucleus instances, and verified that raw file sizes and
  modification times did not change. The canonical validator subsequently reconciled
  those counts exactly.
- Completed a read-only full-release connectivity preflight. It found 211 raw
  instance IDs in 201 patches with multiple 4-connected components; 119 instances
  in 116 patches remained disconnected under 8-connectivity. The validator now
  retains and reports these immutable raw identities instead of mistaking them for
  download corruption, while genuine array-structure failures remain fatal.
- Completed the canonical read-only acquisition verification and its public CLI rerun.
  All three archive sizes and SHA-256 values reconciled exactly; ZIP CRC and
  path-safety checks passed with zero failed or rejected members; nine extracted NPY
  files and nine release documents were checksum-bound to the archives; and the
  22-file raw metadata snapshot was identical before and after verification. The
  final acquisition manifest SHA-256 is
  `837fd4692ca94df4bc9dfa929bc84b1bb4dbdcbd1858c90c247bb76b0b197111` and its bound
  verification-report SHA-256 is
  `f058de85ffb023be6fc5b9aa674e4470696f8cd822d2368fb54ab9d737067fd8`.
- Executed an official torchvision ResNet-18 ImageNet weight smoke on CUDA for RGB and target-highlighted RGB; both caches contain finite 2×512 embeddings. Weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Audited pathology-encoder availability with a frozen priority rule. No candidate satisfied all source/licence/weights/preprocessing/hardware/intended-use/smoke gates, so no pathology encoder was selected.
- Implemented fail-closed APIs and CLI commands for post-pilot preregistration freezing, group-safe exploratory original-label auditing, and structurally validated blinded expert-review packages. No expert responses are generated.
- Created and verified a 21-record primary-source bibliography, literature review/matrix, and search log, plus external-validation and geometric-audit future-work protocols.

## Commands actually executed at the final checkpoint

- `uv sync --dev`
- `uv pip check`
- `.venv\Scripts\python.exe -m histo_audit doctor --project-root .`
- Explicit CUDA 512×512 matrix-multiplication/backward probe
- `.venv\Scripts\python.exe -m histo_audit --help`
- `.venv\Scripts\python.exe -m histo_audit data generate-synthetic --project-root . --config configs\smoke.yaml`
- `.venv\Scripts\python.exe -m histo_audit experiment smoke --project-root . --config configs\smoke.yaml`
- `.venv\Scripts\python.exe -m histo_audit experiment smoke --project-root . --config configs\smoke_zero.yaml`
- Independent `verify_run_integrity(...)` and current-source-tree comparison for both canonical runs
- Manual visual inspection of the PR curve, paired-bootstrap interval, complete target panel, top-ranked gallery, fold-safe neighbour grid, false-high/false-low gallery, and the 0% score/false-alert alternatives
- `.venv\Scripts\python.exe -m pytest -q`
- `.venv\Scripts\python.exe -m ruff check .`
- `.venv\Scripts\python.exe -m ruff format --check .`
- `.venv\Scripts\python.exe -m mypy src`
- `.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root data\raw\pannuke` — after acquisition reached semantic checks and stopped with exit code 1 because sampled official masks contain cross-class overlap
- `.venv\Scripts\python.exe artifacts\qc_independent\pannuke_independent_qc.py` — exit code 0 in 177.132228 s; mmap/batch scan covered all 7,901 patches and 517,799,936 mask pixels without modifying raw data
- First full acquisition-artifact attempt through an ad-hoc stdin Python driver —
  exit code 1 after 91.5 s, with no artifacts written, because the licence parser
  incorrectly required Markdown-linked text to be contiguous. The exact ephemeral
  stdin body was not retained, which is an audit limitation; the parser and a saved
  regression test were corrected against the exact local README.
- `.venv\Scripts\python.exe -m histo_audit data verify-pannuke-acquisition --project-root . --root data\raw\pannuke --verification-timestamp-utc 2026-07-17T23:45:00.282Z --expected-previous-manifest-sha256 02735a974e9491270a46c166d26569b8a1245134072cecb6437389490c8feba2 --expected-previous-report-sha256 d7dcf543683bdf5795608bc31e82d0b2cc43566e6e72bbe79dbd2b69a4d0ab47` — exit code 0 in 90.483065 s; canonical manifest/report replaced together by explicit CAS, raw unchanged, and the scientific stage did not advance
- `$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path; .venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root $env:PANNUKE_ROOT --max-samples-per-fold 100000 --max-overlay-patches 24` — exit code 0 in 309.877 s; full semantic scan covered all 7,901 patches and published the base validation plus immutable QC bundle. It recorded 4,318 overlap pixels / 575 patches, 10,486,091 void pixels / 162 patches, zero positive-plus-background pixels, and 1,411 identically primary/confirmatory-ineligible overlap-touching instances without modifying raw masks.
- `$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path; .venv\Scripts\python.exe -m histo_audit data build-manifest --project-root . --root $env:PANNUKE_ROOT --batch-rows 4096` — exit code 0 in 531.455 s; published 189,744 unique nucleus identities across 7,558 nonempty source-patch groups. Exactly 1,411 retained rows are identically primary/confirmatory-ineligible with `touches_cross_class_overlap`; 343 patches without positive instances produce no synthetic void rows.
- `$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path; .venv\Scripts\python.exe -m histo_audit data audit-duplicates --project-root . --root $env:PANNUKE_ROOT --embedding-device cuda` — exit code 0 in 254.702 s; the republished canonical bundle covered all 7,901 patches and all 20,798,326 cross-fold pairs for each required signal. It found zero exact pairs, 121 pHash candidates at Hamming distance at most 4, and zero frozen-ResNet candidates at cosine similarity at least 0.995. Every candidate is `review_only`; no data, eligibility, split, or final-reference outcome was changed or used.
- `.venv\Scripts\python.exe -m pytest -q tests\test_pannuke_gate.py tests\test_duplicates.py` — **28 passed in 7.95 s** after the canonical duplicate republication.
- Current-worktree exact validation rerun — exit code 0 in 311.7 s; all 7,901 patches
  and the same 4,318 overlap / 10,486,091 void / 1,411 exclusion counts reconciled,
  raw remained unchanged, and publication was explicitly `idempotent`.
- Current-worktree exact manifest rerun — exit code 0 in 533.2 s; 189,744 rows and
  7,558 groups reconciled, with unchanged Parquet SHA-256
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`.
- First current-worktree duplicate rerun — exit code 1 after 255.2 s, fail-closed and
  without changing any active file, because the old complete JSON lacked four newly
  mandatory embedding-cache/sidecar bindings while the other four reports were
  byte-identical. The full old bundle was checksum-copied to the ignored
  `.superseded/20260718_cache_binding_contract_upgrade` directory.
- Explicit full duplicate-bundle republication through the same official command —
  exit code 0 in 334.7 s; the scientific result remained 0 exact / 121 pHash / 0
  ResNet candidates with full coverage and no automatic action. New JSON SHA-256:
  `2a24d0f637bbf47e215f276e38d30b6bd65f1d312caa1b5b285ca5c00540612e`;
  all other report/cache hashes remained unchanged. The public evidence validator
  returned `ERROR_COUNT 0` in 29.6 s.
- Historical pre-acquisition invocation of `.venv\Scripts\python.exe -m histo_audit
  experiment pilot --project-root . --data-root data\raw\pannuke --config
  configs\pilot.yaml` correctly gated with exit code 2. It was later superseded by the
  eligible sealed M6 run documented below.
- `.venv\Scripts\python.exe -m histo_audit experiment primary` — correctly gated with exit code 2
- `.venv\Scripts\python.exe -m histo_audit experiment confirmatory` — correctly gated with exit code 2
- Read-only inline Python connectivity preflight over all three memory-mapped mask
  arrays — exit code 0 in 26.158 s; 211 four-connected and 119 eight-connected
  disconnected raw instance IDs; no source file written.
- `.venv\Scripts\python.exe -m pytest -q tests\test_pannuke_gate.py tests\test_pannuke_qc.py tests\test_pannuke_qc_reporting.py` — 40 passed.
- Focused M5/M6 regression suite — 149 passed; changed-file Ruff check, Ruff format
  check, and mypy all passed.
- Future-gate API/execution suites — 141 plus 55 tests passed, covering M7–M9
  contracts, primary/confirmatory statistics, checkpoint controls, filesystem
  reconciliation, and external-readiness APIs. These are implementation tests, not
  executed study outcomes or completion-stage evidence.
- First current-worktree full `.venv\Scripts\python.exe -m pytest -q` — 8 failed and
  496 passed in 159.98 s. Six failures came from a legacy schema-v1 primary-statistics
  fixture incorrectly labelled schema-v2; two exposed a missing
  `selected_injected_event_count` in produced confirmatory statistics. The production
  schema-v2 validator remained fail-closed; the fixture and confirmatory producer were
  corrected, and legal primary zero-event coverage was retained.
- Historical focused publication/concurrency suite — 114 passed. A later, broader
  adversarial audit found additional P1 publication-race, ownership-safe rollback,
  output-under-raw, and cross-cache-binding gaps, so the earlier “no remaining P1/P2”
  interpretation was withdrawn. Those findings were subsequently corrected and
  independently re-audited before the current closure gate.
- Historical pre-closure full `.venv\Scripts\python.exe -m pytest -q` — 540 passed in
  281.77 s. An independent simultaneous run passed 540 tests in 234.31 s. Later M5
  integrity changes require a new full result before closure.
- At that historical checkpoint Ruff check, Ruff format, mypy for 72 source files,
  and `git diff --check` passed. `git status --short` showed the initial repository
  contents as untracked; no commit or remote was created.

## Current M5 closure QA

The final stable M5 worktree passed the complete gate on 2026-07-18:

- Required focused M5 command: **38 passed in 8.89 s**.
- Broad PanNuke acquisition/QC/manifest/duplicate/representation/pilot-readiness
  integration suite: **210 passed in 45.44 s**.
- Full pytest: **604 passed in 212.38 s**.
- Ruff lint: **passed**.
- Ruff format check: **115 files already formatted**.
- mypy: **passed for 72 source files**.
- `git diff --check`: **passed**.
- Git-ignore checks passed for raw ZIP/NPY, the nested nucleus-manifest Parquet,
  duplicate NPZ/sidecar/resume, and `.superseded` evidence; `git ls-files` returned
  none of those data files.
- `git status --short` still reports the entire initial repository as untracked.
  Consequently, whitespace validation is valid but a baseline-aware Git diff is not
  yet available. No commit, staging operation, remote, or push was created.

## Post-seal M6 pilot audit — eligibility withdrawn

- The first real pilot command returned exit code 0 in 504.6 s and technically sealed
  run `20260718T033036.351640Z_pannuke_pilot_a6a660d93e` with terminal status
  `completed`. Its artifact-root SHA-256 is
  `0f25bf15c3359213e5e0b77608da331200cdf99a7b4316fe07fb357b76121096`;
  `verify_run_integrity` still correctly reports technical integrity and a matching
  append-only registry record.
- Mandatory post-seal inspection found a procedural P1. Although the dedicated final
  metadata object omitted final sample IDs and class labels, the global eligibility
  provenance embedded 475 unique fold-3 sample IDs whose canonical IDs encode class.
  They appeared in five JSON artifacts (2,375 total occurrences): the selection file,
  metrics, and three representation sidecars.
- Representation NPZ arrays contained development folds 1/2 only, and the audit/OOF/
  restoration computations did not consume final-fold features or labels. The defect
  nevertheless violates the declared metadata-only final-fold access contract and
  makes the run scientifically and stage ineligible.
- The sealed run, `registry.csv`, and `integrity_registry.jsonl` remain unchanged as
  append-only historical evidence. It must not satisfy M6, `PILOT_COMPLETE`,
  preregistration freeze, primary selection, or reported canonical pilot evidence.
  A privacy-safe manifest/provenance fix and a new clean pilot run are required.
- The external disposition mechanism is now machine-enforced. The append-only
  `run_dispositions.jsonl` contains one `eligibility_withdrawn` record with SHA-256
  `22926ba5bf8b7a2139e1fc10676ce5e0ade616a7ddd18adbc0bc10e7b2b1755d`.
  Its ledger SHA-256 is
  `69976dde3bfd9b7bfabc2a16ae55f2ab201f7298382effd5e933566f8c67c0d7`,
  and the head/count/full-ledger anchor SHA-256 is
  `743c8458ae5972c1ec7bb30ca0eb5f2e3a343b331e4ac17c43699ed01479ed3a`.
  `verify_run_integrity` still passes with the unchanged sealed artifact root,
  whereas `require_run_stage_eligible` now rejects the run. Focused ledger and
  run-tracking regression tests passed: **42 passed in 211.80 s**.

## Historical global QA checkpoint before final M5 hardening

The worktree before the later M5 integrity hardening passed this gate on 2026-07-18;
these counts are historical and do not close the current M5 worktree:

- Pytest: **540 passed in 281.77 s**; independent second run: **540 passed in
  234.31 s**.
- Ruff lint: **passed**.
- Ruff format check: **115 files already formatted**.
- mypy: **passed for 72 source files**.
- `git diff --check`: **passed**.
- Doctor/CUDA remain available from the environment gate; this QA result does not by
  itself claim that the real PanNuke validator, manifest, or duplicate audit ran.

## Canonical experiment runs

### 10% controlled corruption

- Run ID: `20260717T162925.902444Z_synthetic_smoke_5573505315`
- Status: completed, sealed, registry-backed, integrity-valid.
- Artifact root SHA-256: `95fdefc840f725c5fadcb15804e1a7252aa907d21d6a4080d334144acff35876`.
- Generating source-tree SHA-256: `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a` (matched the source tree at that sealed synthetic checkpoint; subsequent M5 implementation intentionally changes the current source tree without altering the sealed run).
- 300 total samples in 60 source groups; 215 audit-pool samples; 60 untouched final-reference samples; 22 exact injected corruptions.
- OOF coverage exactly once; OOF group overlap 0; final-reference group overlap 0; final reference verified uncorrupted.
- At the 5% review budget (11 samples), self-confidence, NLL, Cleanlab, and the fixed hybrid each retrieved 11/22 injected corruptions with precision 1.0, recall 0.5, and lift 9.7727 over random expectation.
- AP: self-confidence/NLL/Cleanlab 0.908849; fixed hybrid 0.901829; neighbour disagreement 0.786484; prediction margin 0.774386; predictive entropy 0.115295.
- Random review (100 deterministic repeats, identical 11-item budget): mean recall 0.047273 and mean precision 0.094545.
- Downstream macro F1: `uncorrupted_reference_baseline` 0.740171; `corrupted_observed_baseline` 0.686604; mean `random_review_restoration` 0.683187; `audit_guided_restoration` 0.700000.
- The 200-iteration paired group bootstrap hybrid-minus-self-confidence AP difference was -0.008035, 95% interval [-0.069931, 0.032785], probability positive 0.385; this smoke does not support a hybrid advantage.

### 0% corruption edge case

- Run ID: `20260717T162948.870526Z_synthetic_smoke_zero_corruption_a4d5f87ca0`
- Status: completed, sealed, registry-backed, integrity-valid.
- Artifact root SHA-256: `2ba716349b010c5bc71f8c7a3b509bfd0f7856c6b138ef8fa0d491050e9236bd`.
- Generating source-tree SHA-256: `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a`.
- 120 total samples, 85 audit-pool samples, and zero injected corruptions.
- AP, AUROC, recall, lift, and paired-difference inference are correctly `not_applicable`; a 5% review budget contains five false alerts because every reviewed item is necessarily a non-corruption.
- All four downstream conditions have identical macro F1 0.821111 and restore zero labels.

Earlier failed or pre-hardening runs remain in the append-only registry as historical evidence and are not used as canonical results.

## Dataset and stage gates

- The reported official PanNuke release is present under `data/raw/pannuke`: all three ZIP archives and extracted folds are local. Canonical schema-v2 acquisition evidence is saved at `data/manifests/pannuke_acquisition.json` with a bound report at `reports/pannuke_acquisition_verification.json`; archive identity, CRC, path safety, extracted inventory, local licence/citation evidence, Git-ignore, and raw read-only checks passed.
- An independently executed full read-only mask audit reproduced 1,216 / 1,572 / 1,530 cross-class-overlap pixels in folds 1/2/3, affecting 194 / 190 / 191 patches and 471 / 465 / 475 unique raw instances (`fold, patch, class, instance_id`). It found 2,359,296 / 3,801,371 / 4,325,424 void pixels in 36 / 59 / 67 patches and zero pixels that were both positive and supplied background. The totals are 4,318 overlap pixels, 575 affected patches, 1,411 affected instances, and 10,486,091 void pixels. The complete working evidence is under `artifacts/qc_independent/` and has now been exactly reconciled by the canonical validator.
- The canonical full-release validator subsequently reproduced those exact counts with zero patch- or instance-level discrepancies. Its JSON/Markdown/raw inventory and the seven-file QC bundle are saved under `reports/`, `data/manifests/`, and `artifacts/figures/`. Validation JSON SHA-256: `094497f43e2ee0bd5dabddcd01f8c934657f130450a66f46600311451d36bc4a`; QC artifact-manifest SHA-256: `0b188ecc586ed772b29845e15e169fb492ed8d2ad0f5b1a6643531ccee10857f`; QC overlay SHA-256: `a1bd87dd397417d711d1d4937429eae5f5d972d3fa6ffa27a45129339587f10a`.
- The same audit found 478 numeric instance IDs reused across positive-class channels in 159 patches, but none touched cross-class overlap and none represented spatial overlap of the same ID. Raw identity therefore remains the full `(fold, patch, class, instance_id)` tuple; numeric IDs are never merged across classes.
- The full nucleus manifest is complete and independently reconciled: 189,744 unique full identities, 7,558 `source_patch` groups, 188,333 analysis-eligible rows, 1,411 identically excluded rows, and no injected corruption. Parquet SHA-256: `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`; summary SHA-256: `3bd0f37f2bfe73180b10194db6d0dadcde45675ed685970c42149c5bf8841c91`.
- Acquisition provenance, canonical anomaly-safe validation, saved QC evidence, visual
  overlay review, the full nucleus manifest, and the complete duplicate audit are
  present. Duplicate JSON SHA-256:
  `2a24d0f637bbf47e215f276e38d30b6bd65f1d312caa1b5b285ca5c00540612e`;
  active embedding NPZ/sidecar SHA-256:
  `ef99e931adc160f9d7fb9ab86bf0287dd8427b9f1d3c42fb0496377a5e287618` /
  `9e2506bbf4592c9012ba492aff46ea181bcfec1f78088334d71e213f6c660392`.
- Component-covering projection no longer automatically rejects the 209 eligible raw
  identities flagged `disconnected_instance_id`. A read-only temporary full scan of
  all 211 disconnected identities and 480 raw components found 6 identities using
  fallback, zero lost components, zero component collisions, and 3 recorded
  projected-union topology changes. The exact temporary inline scan body was not
  retained, which is an audit limitation; saved pilot provenance and current
  self-verifying cache tests remain mandatory. Raw identities were not split, merged,
  repaired, relabelled, or overwritten.
- `PILOT_COMPLETE`: reached by the eligible sealed run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0` after independent post-seal
  inspection.
- `PRE_REGISTRATION_FROZEN`: reached by the independently verified authority
  `artifacts/preregistrations/20260719T002902.432341Z` before any primary or
  final-reference outcome inspection. The original `PRE_REGISTRATION.md` remains the
  recorded `READY_FOR_FREEZE` input; `PRE_REGISTRATION_FROZEN.md` inside that immutable
  authority is the frozen registration.
- `PRIMARY_STUDY_COMPLETE`: reached by the separately sealed bounded recovery
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery`, which checksum-verified
  and reused all 185 required cells, retained the 37 declared optional skips, and
  retrained zero cells. The interrupted source run remains unsealed, read-only, and
  ineligible; this later recovery evidence supersedes the historical pre-recovery
  statement formerly recorded here.
- `CONFIRMATORY_COMPLETE`: not reached. The unchanged original frozen 108-cell
  study is the active post-infrastructure direction and has not run. The
  separate resource-bounded sensitivity path is retained only as an optional
  non-claiming artifact; it is not the current prerequisite and, if ever run,
  must retain `completion_stage=null`.
- `EXTERNAL_VALIDATION_READY` / `EXTERNAL_VALIDATION_COMPLETE`: not reached; no eligible real ranking or genuine expert response exists.

## M6 corrected pre-execution checkpoint

- The withdrawn pilot remains preserved and ineligible. No metric from it was used
  to choose the corrected protocol.
- The pilot producer now consumes a separately materialized folds-1/2 development
  manifest and a checksum-bound gate certificate. Model-facing semantic reads are
  development-only; fold 3 is restricted to byte-level integrity plus class-free
  aggregate/per-group eligibility metadata.
- Group-safe OOF allocation is fixed before controlled corruption with split seed
  223 and `pre_corruption_label` used only for fold allocation. Fit, Cleanlab, risk
  scores, and neighbours use `observed_label`. The limitation for naturally occurring
  unknown inconsistencies is retained in `PRE_REGISTRATION.md`.
- Publication and privacy hardening passed independent review. Windows uses native
  handle-relative no-overwrite publication and POSIX uses directory-FD relative
  publication; a forced parent rename/symlink probe made 3,559 observations with zero
  transient target visibility outside the anchored directory. Raw-inventory
  additions, source/output changes, partial publication, collision, rollback,
  structured NPZ metadata, recursive Arrow metadata/types, and artifact-path leaks
  all fail closed.
- Final code gates after these corrections: **640 passed in 225.73 s**; global Ruff
  passed; Ruff format reported 116 formatted files; mypy passed for 72 source files;
  `git diff --check` passed. The focused integrated pilot/CLI/OOF/cache suite passed
  73 tests, and the final pilot privacy file passed 31 tests. Independent review
  reported no remaining P1/P2/P3 finding.
- Real command executed:
  `.venv\Scripts\python.exe -m histo_audit data build-pilot-development-view
  --project-root . --validation-json reports\pannuke_validation.json --manifest
  data\manifests\pannuke\pannuke_nucleus_manifest.parquet --duplicate-audit-json
  artifacts\duplicate_audit\pannuke_duplicate_audit.json --output
  data\manifests\pannuke\pannuke_pilot_development_manifest.parquet`.
  The first publication completed in 155.7 s. A later correctly instrumented
  idempotence run completed in 137.1 s and preserved SHA-256, byte length, and mtime
  for both files exactly. One intervening helper wrapper had an empty PowerShell path
  array; its successful CLI result was retained, but its empty comparison was not
  treated as idempotence evidence.
- Real development view: 123,090 rows, 79 columns, folds 1/2 only, exact row equality
  with the corresponding canonical manifest partition. Parquet SHA-256:
  `8107e1ddc033b08f03d3f351b5993ec1fd7a188677ee4c2afc0e4cbfe8432ef8`;
  certificate SHA-256:
  `347f734c2355cd5009d631e959c077fe430adcad7e3997fe1306280004e90146`.
  Independent inspection found zero final-reference IDs in decoded values, paths,
  field/schema metadata, serialized Arrow schema, certificate JSON, or raw Parquet
  bytes. All 22 raw files (39,359,162,655 bytes) rehashed exactly; inventory SHA-256:
  `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`.
- At this historical pre-execution checkpoint, M6 was not satisfied: completion was
  `PIPELINE_COMPLETE`, formal progress was 6/10 = 60%, and `PRE_REGISTRATION.md`
  remained DRAFT pending a new sealed pilot. The later final eligible execution and
  current stage are recorded below.
- Corrected-boundary run
  `20260718T134701.590268Z_pannuke_pilot_4b62a55d63` completed in 511.3 s and sealed
  34 artifacts with root SHA-256
  `50f300b1ab60282def3309ba5c3980bfb94ff97ebd69890d00bb3dc8154e1e68`.
  It contained 5,481 audit samples, 814 development reference-validation samples,
  548 exact corruptions, and only the permitted class-free count of 66,179 eligible
  final-reference rows. Its run-local dev Parquet was byte-identical to the pre-pilot
  view; independent scanning of all 36 files / 174,255,355 bytes found no final-fold
  identity, class, geometry, representation, or outcome leak. Integrity, OOF,
  Cleanlab/neighbour reconciliation, and the old run's withdrawal status all passed.
- The same inspection found one non-numerical P2: `report.md` lacked the exact
  mandatory phrase `recommended for expert review`. The sealed run was not edited.
  It was withdrawn with reason code `post_seal_required_terminology_mismatch`; record
  SHA-256:
  `467b8e8a265a30fdc06951231f37f72678a49b7f08877bcb3dbc8291183cf12a`.
  The two-record disposition ledger SHA-256 is
  `c7ee7d5a6a4f27c64b6c8b9f7d3af05d3405e014262f48463ce66b86a6121f07`;
  anchor SHA-256:
  `ea29ec96e3ab449bb44809114f9c1cc321291dd41c90f1442514f819b5282fed`.
  Generator and regression test now require both `potentially inconsistent
  annotation` and `recommended for expert review`; focused pilot tests passed 31/31,
  and Ruff, format, mypy for 72 source files, and `git diff --check` passed. The
  terminology-only correction requires one new run and does not use metric values.

## M6 final eligible execution and closure

- Before the final execution, two newly discovered workflow defects were fixed and
  regression-tested: confirmatory execution now rejects a withdrawn primary run
  before reading its outcome artifacts, and the preregistration freeze command no
  longer depends on a favourable import order. The fresh-process freeze regression
  and the focused workflow tests passed 11/11.
- Mandatory code gates then passed on the current worktree: **641 passed in 478.04 s**;
  Ruff lint passed; Ruff format reported **117 files already formatted**; mypy passed
  for **72 source files**; `git diff --check` passed.
- The exact real command in the previous `Next exact command` block completed in
  **770 s** and created sealed run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0`. It contains 5,481 audit samples
  in 225 source-patch groups, 814 development reference-validation samples in 25
  groups, exactly 548 controlled label changes, and only the permitted class-free
  final-reference count of 66,179 rows in 2,608 groups.
- The sealed run has 34 manifest artifacts and 36 total files. Its artifact-root
  SHA-256 is
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`;
  artifact-manifest SHA-256 is
  `076f9cc45a42ac63f2dad9679b8979a50434b4c9a824d64fb94baf6b121b9074`.
  The append-only disposition ledger still has exactly the two historical withdrawal
  records and no record for this run; `require_run_stage_eligible` passes.
- Multiple independent read-only post-seal checks passed. All 36 files were scanned;
  no fold-3 sample identity, class, geometry, representation, or outcome was found.
  The run-local development Parquet and gate certificate are byte-identical to the
  pre-pilot sources. OOF covers each audit sample exactly once across five group-safe
  folds with zero group overlap and maximum probability-sum error
  `4.440892098500626e-16`. Corruption separation is exact:
  `is_injected_corruption == (pre_corruption_label != observed_label)` for all rows.
- Full numerical replay found no P1/P2/P3 issue. Self-confidence/Cleanlab AP is
  `0.3954650789236523`; at the fixed 5% budget, 147/548 controlled changes are
  retrieved (precision `0.5345454545454545`, recall `0.26824817518248173`, lift
  `5.346429993364299`). Guided restoration changes macro F1 from
  `0.4434302869476296` to `0.4585613475637464`; these are pilot controlled-corruption
  results, not evidence that any original annotation or pathologist was wrong.
- `report.md` and `metrics.json` contain both exact mandatory phrases:
  `potentially inconsistent annotation` and `recommended for expert review`, plus
  the non-diagnostic limitation. The final post-run raw check rehashed all 22 files,
  39,359,162,655 bytes, to unchanged inventory SHA-256
  `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`;
  `source_annotations_modified=false`.
- M6 is therefore closed. The project stage is exactly `PILOT_COMPLETE`; formal
  progress is 7/10 = 70%, with 30% remaining. `PRE_REGISTRATION.md` is now
  `READY_FOR_FREEZE` but not frozen, and no primary/final-reference outcome has been
  inspected.

## M7 implementation checkpoint — contracts and freeze path

- A public, read-only `experiment verify-pilot-post-seal` command now independently
  requires the external folds-1/2 development manifest and pre-pilot gate certificate.
  On the canonical run it passed twice after a group-ledger regression was corrected:
  36/36 sealed files, six NPZ files / 77 arrays, one Parquet / 123,090 rows, exact
  group-safe OOF coverage, 548 separated injected corruptions, mandatory terminology,
  zero final-fold outcome publication, and no automatic withdrawal or write.
- The pilot-derived primary-parameter producer was executed twice byte-identically on
  the eligible folds-1/2 pilot. It wrote
  `reports/pilot_derived_primary_parameters.json`, SHA-256
  `8380b963a02b7ea4451039e9e5b37600809b22c689734202664522bdeda6113b`.
  The record binds the pilot/root/cache/order hashes and deterministically derives the
  confusion-targeted transition matrix and tissue weights without reading a fold-3
  sample identity, label, representation, or outcome.
- Primary and confirmatory contracts now require the exact PanNuke fold policy,
  `source_patch_id`, the shared
  `deterministic_group_greedy_class_distribution_v1` reference selector, split seed
  223, `pre_corruption_label` allocation, the complete eight-case QC policy, one clean
  0% cell, positive rates 5/10/20%, equal-weight hybrids, real expanded cell IDs, and
  controlled H2/H3 comparisons. Cross-scenario statistics evaluate each cell against
  its own `is_injected_corruption` mask with identical group-bootstrap draws.
- Freeze publication is now one-shot, ownership-safe, TOCTOU-rechecked, and
  fail-closed. It snapshots and authenticates execution source separately from mutable
  governance, the strict representation-independence evidence, pathology availability,
  and pilot-derived parameters. The two generated canonical frozen configs are
  separately authenticated and intentionally excluded from the execution-source root,
  preventing publication from invalidating its own source identity.
- A real immutable preregistration-amendment API and CLI now create and recursively
  verify parent-hash-linked successor authorities. Post-outcome amendments cannot be
  relabelled as the original primary analysis. Study execution gates accept either a
  verified base freeze or a verified amendment, reject code drift, and permit later
  `STATUS.md`/governance logging without changing the frozen scientific identity.
- The full-release representation builder now switches automatically to bounded
  chunk/resume extraction above 10,000 rows. Partial state remains in a private sibling
  lease, final publication contains exactly five NPZ files and five sidecars, and the
  Windows cleanup API/atexit path removes successful-workspace memmaps. Focused chunk
  and cache-provenance tests passed 6/6.
- Focused evidence completed at this checkpoint includes: 52 source/amendment tests,
  26 amendment/CLI integration tests, 35 freeze/study-gate tests, 63 strict-contract
  and cross-scenario tests, 20 primary-statistics tests, and the real post-seal pilot
  command. Ruff and focused mypy checks passed in each integrated owned scope. These
  are implementation/evidence checks, not primary or confirmatory study outcomes.
- At that checkpoint, the real freeze preflight correctly stopped at
  `status=awaiting_required_cache_provenance`. It had not published a freeze directory
  or canonical frozen config. The configs contained resolved scientific choices but
  deliberately retained fail-closed cache placeholders until the real full-release
  caches, independence matrix, and updated optional-pathology blocker were generated.
- A bounded real-data CUDA smoke of the chunked full-release builder passed on three
  PanNuke samples (one from each official fold), chunk size two, with exactly five
  NPZ caches and five sidecars. Every sidecar passed frozen-cache verification and
  carried the exact producer `primary_cache_provenance`; the private resume workspace
  existed only during extraction and was removed on normal cleanup. The ignored smoke
  bundle at `artifacts/embeddings/preflight/m7_chunked_real_20260718T2138Z` is about
  12.95 MB and is not stage evidence.
- The production CLI now builds the five-cache bundle and strict schema-v2 audit-slice
  `reports/representation_independence.json` in the same process while its memmaps are
  live. An injected independence failure retracts the public cache directory, retains
  the private checkpoint, and retries without repeating completed crops. The focused
  cache/independence/pathology/primary-input/preregistration/CLI selection passed
  **85/85**; its Ruff, format, and focused mypy gates passed. The regenerated blocked
  optional-pathology audit has SHA-256
  `5e568cf29e489d8948bfcd33feae5b292cb48837eb4c93754202a565778a6e4a`
  and exact unavailable-cache recipe SHA-256
  `d84f444e34341d0ee739cb8504ba94612010f15b1836993d36249d490855060f`.
- M7 config finalization now reuses the exact freeze validators for the canonical M6
  pilot-derived report and pathology audit, snapshots all source/evidence inputs used
  in derived provenance, authenticates the full manifest/count/order authority, and
  publishes the two configs as one rollback-safe CAS pair. Root's integrated
  finalizer/strict-contract/primary-input selection passed **160/160** in 19.74 s;
  the preregistration safety suite independently passed **10/10** in 13.68 s. After
  extending the same authority to confirmatory and the freeze cross-validator, the
  owner focus passed **171/171**, and root's independent
  finalizer/strict-contract/preregistration-safety selection passed **156/156** in
  30.93 s. A later combined authority/finalizer/cache selection passed **197/197**;
  the confirmatory core/statistics/completion/workflow selection passed **40/40**.
- A subsequent independent fault-injection review found two cache-publication gaps not
  exercised by that 85-test selection: a failure while reopening memmaps after the
  chunked directory rename could leave a public ten-file cache bundle without its
  independence JSON, and cache outputs did not yet reject every ancestor carrying a
  sealed-run `.immutable.json` marker. The repaired path now uses anchored,
  never-overwrite publication, ownership-record rollback without path-based recursive
  deletion, early immutable-ancestor rejection, and fail-closed same-volume checks. It
  authenticates the full manifest, the 188,333 analysis-eligible sample count, and the
  canonical sample order before and after extraction; direct subsets are forbidden.
  The owner selection passed **104/104**, global Ruff passed, and focused mypy passed;
  root independently repeated the integrated cache/independence/disk/publication/
  preregistration selection with **41/41** passing. A later read-only review found a
  retry-only closed-memmap alias defect; reloading the published
  context-plus-morphometrics artifact fixed it and its regression is green. The same
  review then exposed missing-parent create/open adoption races. These are fixed with
  descriptor-relative Windows creation and private no-replace POSIX promotion;
  rollback uses retained handles or quarantine/verify/restore rather than deleting an
  unverified path. Final owner publication tests passed **68/68**; independent
  publication/authority tests passed **144/144**, handle/cache/rollback tests passed
  **90/90**, and stage CLI tests passed **8/8**. Global Ruff/format and focused mypy
  were clean. No full bundle was started or published.
- An expanded preregistration safety run also exposed a marker-last publication
  regression: the generic mutable-destination guard treated the freeze transaction's
  own already-published manifest as a foreign sealed ancestor while publishing its
  final `.immutable.json`. The real freeze has not been attempted. This is now fixed
  by a separate transaction-owned success-marker primitive that retains the ordinary
  sealed-ancestor checks and rejects any second seal marker. The preregistration safety
  suite passed 10/10; amendment/CLI/QC reporting passed 34/34; Ruff and mypy passed.
- The production confirmatory runner initially passed a 132-test focused suite and an
  independent 93-test selection, but a second adversarial review correctly prevented
  closure. It showed that dependency-injected test doubles and non-Torch checkpoint
  bytes could still support an eligible completion claim, restoration replay used the
  current mutable feature arrays rather than every bridge-time hash, and eligibility
  became visible before a durable positive post-seal attestation. These are code-path
  blockers only; no confirmatory study or final-reference outcome was executed. The
  eligible entry point, checkpoint parser, restoration certificate, and default-deny
  post-seal eligibility protocol were repaired and passed their focused regressions.
  The repaired owner suite first passed **104/104**, including
  fail-closed completion/attestation, production-dependency, strict Torch checkpoint,
  all-partition restoration-certificate, complete report-contract, and real CUDA
  resume checks. Exact CNN data/split fingerprints are now computed independently
  before training from each cell/fold bridge input and reconciled with both checkpoint
  and telemetry at completion. The final owner suite passed **105/105**; independent
  QA over nine confirmatory/contract/tracking files passed **236/236** in 200.09 s,
  with Ruff, format, and mypy all clean. No confirmatory study was executed.
- The stable M7 contract/post-seal/amendment/source test selection passed **156/156**:
  `.venv\Scripts\python.exe -m pytest -q tests\test_pilot_postseal_cli.py
  tests\test_preregistration_amendment.py
  tests\test_preregistration_amendment_cli.py
  tests\test_execution_source_identity.py tests\test_study_contracts.py
  tests\test_study_contracts_strict_m7.py`.
- At the 2026-07-18 checkpoint, full-release disk preflight was **NO-GO**. The five final caches and their
  private sibling workspace have a conservative simultaneous peak of 15--16 GiB;
  drive C had about 19.39 GiB free at the recorded check. Extraction must not start below the
  fixed operational threshold of 35 GiB free, leaving at least 19--20 GiB of safety
  margin for publication, filesystem overhead, and failure recovery. GPU preflight is
  GO (RTX 4070 12 GB; locally cached ResNet-18 weights verified). The machine-readable
  point-in-time record is `reports/m7_full_cache_preflight.json`, SHA-256
  `6bf9325b0e41ec84621116b9f84e5c132c58752c2afa95dc4238048d0d95ad72`.
- The 35-GiB threshold is now enforced in code before full-manifest validation,
  staging, or resume-workspace creation, including direct chunked and same-process
  independence entry points. There is no threshold override; only effective builds of
  at most 10,000 requested rows bypass it. Seven dedicated and 23 integrated
  cache/independence/CLI tests passed; Ruff, format, and mypy over 80 source files
  passed. The real heavy command therefore failed closed at that checkpoint's free space.
- The exact production CLI was then invoked with the full manifest, CUDA, all five
  caches, and same-process independence output. It exited in 4.3 s with
  `InsufficientFullManifestCacheDiskSpaceError`: `free_bytes=19602132992`,
  `required_free_bytes=37580963840`, and 189,744 total manifest rows observed for the
  start gate. Readback confirmed that `artifacts/embeddings/pannuke`, its private
  `.pannuke.chunked-resume` sibling, and `reports/representation_independence.json`
  were all absent. This expected fail-closed result is functional-gate evidence, not a
  completed extraction.
- A read-only Git/data-safety audit found no coverage gap: all 21 raw PanNuke files,
  both nested Parquet manifests, all 18 current embedding/smoke artifacts, and all 120
  duplicate-audit NPZ/NPY/sidecar/resume artifacts are ignored. Sizes and SHA-256 for
  all 3 ZIP archives, 9 extracted NPY arrays, and 9 bundled documents still match the
  acquisition authority. Nothing is tracked or staged; `.gitignore` itself must be
  included when the repository receives its future initial commit.
- The exact real-data `data validate-pannuke` command was rerun read-only. Its full
  semantic scan completed in 290.97 s and reconfirmed 4,318 cross-class-overlap pixels
  across 575 patches / 1,411 instances and 10,486,091 void pixels across 162 patches,
  but publication failed closed because seven immutable selection/overlay-dependent
  outputs differed. Raw metadata was byte-for-byte unchanged. The cause is a CLI
  default regression: canonical evidence used 100,000 sampled patches per fold and 24
  overlay patches, while the current no-flag command requested 32 and 6. Apart from
  the sampled/selection-dependent fields, the recomputed JSON is identical. Both
  CLI defaults are restored to 100,000/24, and the no-flag forwarding/idempotency
  regressions passed **2/2** with focused Ruff and format checks. After the shared
  publication primitive was hardened, the M5 CLI/gate/QC selection passed **71/71**.
  The exact no-flag real command was then repeated: exit code 0 in 322.74 s,
  `status=valid`, `validation_scope=full_semantic_scan`, and
  `publication=idempotent`. It reproduced the canonical selection SHA-256
  `09886588591d9ebb9a725db1022bb0ab8fb94b4bcca419b486e2549b0cc5fd36`
  and overlay SHA-256
  `a1bd87dd397417d711d1d4937429eae5f5d972d3fa6ffa27a45129339587f10a`.
  All 11 canonical artifact hashes, sizes, and modification times and all 22 raw-file
  metadata records remained exactly unchanged; no staging directory remained.
- A read-only full-gate preflight initially collected **847** tests in 4.50 s with plugin
  autoload disabled. It confirmed that the suite uses temporary synthetic trees for
  primary/confirmatory completion tests, does not download weights/data, and does not
  execute a real study. After later regressions were added, the first completed global
  run produced **852 passed / 2 failed** in 385.12 s; both failures were outdated fast
  image-OOF test adapters missing the newly mandatory data/split fingerprint. The
  fixtures were repaired through the production fingerprint helper without weakening
  source validation, and the focused file passed **10/10** twice. The first corrected
  global run passed **854/854** in 380.81 s; after synchronising the governance and
  preregistration documents, the exact final-tree rerun again passed **854/854** in
  383.07 s. `ruff check .`, `ruff format --check .` (141 files), and `mypy src` (80
  source files) all passed. An earlier 124-s process-tool timeout produced no Pytest
  result and is not counted as a test failure. After recording those results, the
  final preregistration/governance/source-contract selection passed **158/158** in
  14.47 s, and Ruff, format, and mypy passed again.
- Current-tree functional readback also passed: `experiment verify-pilot-post-seal`
  returned `status=passed`, `scientific_stage_eligible=true`, and
  `sealed_run_unchanged=true`, with exact group-safe OOF/corruption/privacy and both
  required terminology terms. At the 2026-07-18 checkpoint, the production full-cache CLI failed before
  staging with `free_bytes=15848607744` and
  `required_free_bytes=37580963840`; all cache, resume, independence, frozen-config,
  and preregistration-freeze destinations remain absent.
- On 2026-07-19 the disk gate became GO: 93,371,658,240 bytes (86.96 GiB) were
  observed before launch, well above the unchanged 37,580,963,840-byte threshold.
  The exact production cache/independence command then exited 1 after 165.9 s during
  prerequisite immutable base-validation reconciliation. It attempted to recompute
  the canonical validation JSON/overlay with library defaults 32/6 instead of the
  canonical CLI limits 100,000/24, so the immutable publisher correctly rejected
  `reports/pannuke_validation.json` and
  `artifacts/figures/pannuke_overlay_grid.png`. Both canonical files, all raw files,
  and the eligible sealed M6 run remained unchanged; no cache, resume workspace,
  independence JSON, temp output, or new lock remained.
- The extraction call now shares the canonical CLI constants and explicitly forwards
  100,000/24. Its bounded owner suite passed **85/85**, independent focused regression
  passed **11/11**, Ruff/format passed, and `mypy src` passed for 80 source files.
- The corrected exact production command completed with exit 0 in **2,143.3 s** on
  CUDA. It verified **188,333** identities and atomically published exactly five NPZ
  caches plus five sidecars and strict schema-v2
  `reports/representation_independence.json` (SHA-256
  `846f421284de381401761a8dc4ceb108d3f3f2a0eece379706be7f7a512789c7`).
  The private resume workspace, temporary files, and locks are absent. All cache files
  are ignored by Git. The production command was:
  `.venv\Scripts\python.exe -m histo_audit representations extract --project-root .
  --data-root data\raw\pannuke --manifest
  data\manifests\pannuke\pannuke_nucleus_manifest.parquet --output-dir
  artifacts\embeddings\pannuke --include-context-embeddings --independence-output
  reports\representation_independence.json --primary-config configs\primary.yaml
  --device cuda --batch-size 16`.
- Root and two independent reviewers fully reopened and verified every NPZ array,
  sidecar hash/semantic binding, lineage, manifest, order, raw inventory, and
  independence decision. All five caches bind 188,333 samples, manifest
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`
  and order
  `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`.
  Raw PanNuke remains 22 files / 39,359,162,655 bytes with inventory identity
  `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`;
  the sealed M6 root remains
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`.
- The transactional M7 config finalizer then exited 0 in **204.8 s**, replacing the
  two known candidate bytes only after their CAS preconditions matched. Independent
  readback passed: primary file SHA-256
  `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9`
  and semantic SHA-256
  `c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15`;
  confirmatory file SHA-256
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`
  and semantic SHA-256
  `ff2ce8d5043813b08db23efe797abe444ba6b6bde292810a094d78757f74460b`.
  Both statuses are `READY_FOR_FREEZE`; strict cross-config validation yields 222
  primary and 108 confirmatory cells. Read-only downstream input validation loaded all
  188,333 samples without training, study execution, final-reference outcome access,
  config writes, or a new run directory.
- A read-only freeze semantic preflight then stopped before publication because two
  directory-hash producers ordered mixed-case PanNuke paths differently. The old
  reconciliation order produced
  `2bc894d52f5a30ac363642762af5ed4ee39671bdfce546df1dfb9058d7ccd19d`,
  while the immutable M6 authority was
  `5647b4837fdaeb1281a5af0623f24aab1361263d3041549d012c8c5697fb31ed`.
  All 22 file names, sizes, and hashes reconciled exactly; this was ordering logic, not
  data corruption, and no freeze output was created.
- RunTracker and freeze now share one host-independent, Windows-compatible component
  ordering with exact-case tie-break. It preserves the existing M6 hash on Windows and
  POSIX while leaving the case-sensitive reconciliation-record order and semantic hash
  `83e3eb7c4460c7c368a9bf70d49c3117f229f694daaa062456ba9f714c75651a`
  unchanged. Regression coverage includes a golden tree hash, mixed-case and nested
  paths, explicit PureWindows/PurePosix ordering, inventory-semantic reconstruction,
  and byte/mtime nonmutation. The integrated focused selection passed **17/17**;
  Ruff, format, and mypy passed for all four touched source/test files. A real PanNuke
  readback again reconciled 22/22 files and matched the sealed M6 tree hash and root.
- The first mandatory full-QA command, `.venv\Scripts\python.exe -m pytest -q`,
  stopped the M7 gate with **855 passed and 2 failed** in 392.94 s. The failures were
  `test_confirmatory_template_records_exact_cache_blocker_and_cannot_freeze` and
  `test_primary_template_records_resolved_pilot_choices_and_cache_blocker`. Both still
  asserted the pre-finalization `awaiting_required_cache_provenance` lifecycle against
  the canonical config paths that the verified M7 finalizer had legitimately converted
  to `READY_FOR_FREEZE`. Strict primary, confirmatory, and cross-config validation
  remained valid. This was stale test expectation, not a config, cache, manifest,
  raw-data, or sealed-M6 failure; no freeze was attempted and status remained
  `PILOT_COMPLETE`.
- Only those two project-config contract tests were corrected. Their replacements,
  `test_confirmatory_project_config_records_finalized_cache_provenance` and
  `test_primary_project_config_records_finalized_cache_provenance`, require the exact
  finalized semantic hashes, shared manifest/sample-order authority, available cache
  provenance, frozen optional-pathology blocker, strict validators, cross-config
  validation, and canonical 108/222-cell plan hashes. Existing synthetic negative
  tests still reject missing or incomplete provenance; no production validator or
  scientific/config/data authority changed. The exact two-test rerun passed **2/2** in
  3.27 s, and the complete contract file passed **79/79** in 3.70 s. The full-suite
  rerun then reported **857 passed in 389.08 s**. This closes the Pytest portion of
  final QA. Post-correction `ruff check .` passed, `ruff format --check .` reported all
  141 files formatted, and `mypy src` passed for 80 source files. The relevant
  functional CLI, one-shot freeze, and independent freeze verification remain
  mandatory.
- The exact real functional command
  `.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root .
  --root data\raw\pannuke` exited 0 in **312.3 s** with
  `status=valid`, `validation_scope=full_semantic_scan`, 7,901 patches, 22 raw files,
  4,318 cross-class-overlap pixels, 10,486,091 void pixels, 1,411 overlap-touching
  instances excluded identically from primary and confirmatory analysis,
  `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- The read-only functional command
  `.venv\Scripts\python.exe -m histo_audit experiment verify-pilot-post-seal` with the
  explicit eligible M6 run, development manifest, and gate certificate exited 0 in
  **31.9 s**. It returned `status=passed`, `scientific_stage_eligible=true`,
  `sealed_run_unchanged=true`, complete group-safe OOF coverage, zero group overlap,
  exact corruption-label separation, unavailable final-reference outcomes, and both
  required terminology terms. Validator publication locks were released normally.
- The authorized one-shot preregistration freeze command was invoked once and exited 1
  after **108 s**, before publication, with
  `RuntimeError: Git state changed before preregistration publication`. The final
  freshness check recaptured `capture_git_state()` and compared the complete mapping
  with the initial capture. Root-cause reproduction showed identical
  reproducibility-bearing fields (`available`, `commit`, `branch`, `dirty`, and
  `status_porcelain`) but a necessarily different volatile `captured_at_utc`; the
  observation timestamp was incorrectly treated as Git identity drift.
- The failed transaction remained fail-closed. No timestamped freeze authority,
  canonical frozen config, success marker, temporary staging entry, or lock was
  published or retained; `artifacts/preregistrations` is only an empty parent
  directory. No source annotation, raw file, cache, sealed-M6 artifact, config
  authority, or outcome was changed or inspected. This attempt does not satisfy M7 or
  change formal status from `PILOT_COMPLETE`.
- Git freshness now snapshots the complete original mapping, including
  `captured_at_utc`, but compares a validated stable projection that removes only that
  observation timestamp. Every other present or absent field remains bound. Available
  captures require a string porcelain status, a consistent boolean dirty flag, a
  string branch, and a null or non-empty commit; unavailable captures require an exact
  non-empty reason. Timestamp-only recapture passes, while availability, reason,
  commit, branch, dirty, porcelain, malformed-null, and dirty/status inconsistencies
  all fail before publication. The exact focused command in the next-command ledger
  passed **33/33** in 25.47 s; owner and independent focused runs also passed Ruff,
  format, and mypy.
- After that correction, the exact mandatory full-suite rerun
  `.venv\Scripts\python.exe -m pytest -q` reported **866 passed in 393.06 s**. The
  synchronized corrected tree also passed `.venv\Scripts\python.exe -m ruff check .`,
  `.venv\Scripts\python.exe -m ruff format --check .`, and
  `.venv\Scripts\python.exe -m mypy src`. An additional independent transaction-safety
  selection passed **13/13** in 20.37 s. These results close the corrected-tree code-QA
  gate without erasing the first failed invocation.
- The exact full PanNuke validator was then rerun on the corrected tree and exited 0 in
  **310.888 s**. It again returned `status=valid`,
  `validation_scope=full_semantic_scan`, 7,901 patches, 4,318 cross-class-overlap
  pixels, 10,486,091 void pixels, and 1,411 overlap-touching instances excluded
  identically from primary and confirmatory analyses, with
  `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- The exact explicit M6 post-seal verifier also exited 0 in **32.1 s** on the corrected
  tree. It returned `status=passed`, `scientific_stage_eligible=true`,
  `sealed_run_unchanged=true`, artifact root
  `37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`, complete
  group-safe OOF coverage with zero group overlap, and
  `final_reference_outcomes_unavailable=true` for untouched official fold 3.
- A final read-only retry preflight found no active QA, validation, study, or freeze
  process; no canonical frozen YAML; zero entries under `artifacts/preregistrations`;
  no `histo-audit-freeze-*` staging directory; and no recent publication lock. Two real
  Git captures differed only in `captured_at_utc`; their validated stable projections
  were equal. Primary and confirmatory config file hashes remain respectively
  `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9` and
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`.
- The single authorized second freeze invocation exited 0 in **209.523 s** and created
  exactly one new authority directory,
  `artifacts/preregistrations/20260719T002902.432341Z`, plus the two canonical frozen
  configs. The CLI returned `status=PRE_REGISTRATION_FROZEN`,
  `integrity_verified=true`, artifact root
  `d2f1f3dec19021e7216630b53297035e705b6c407c3b5d84118ce7637411dd65`, and manifest
  SHA-256 `f223b6edd8364c90c476e13c9e3e3c14718418dfdc9938f2752492e65ee83101`.
- The first fresh-process verifier call already returned `valid=true`, empty
  missing/added/changed/error collections, and a matching manifest hash, but an
  auxiliary wrapper assertion used the nonexistent key `actual_artifact_root_sha256`
  instead of the dataclass key `artifact_root_sha256`, so that wrapper exited 1. No
  artifact or config was changed and freeze was not rerun. The corrected read-only
  wrapper then exited 0 in **4.3 s**, with `valid=true`, exact expected/actual/CLI root
  equality, the exact CLI manifest hash, and all four discrepancy collections empty.
- The exact read-only M8 `validate_primary_execution_gate(...)` call completed every
  authority check and returned evidence, but its first display wrapper exited 1 because
  `json.dumps(asdict(result))` cannot serialize `WindowsPath`. No run directory or
  outcome was created. Repeating only the read-only check with `default=str` exited 0 in
  **29.2 s** and bound the exact freeze, source root
  `454c02f237a95532d959c6f06da6dd69fee0f23c9011ea361b6f4ac378155f43`, dataset root
  `5647b4837fdaeb1281a5af0623f24aab1361263d3041549d012c8c5697fb31ed`, manifest,
  duplicate audit, pathology audit, eligible M6 root, both frozen configs, 222 primary
  cells, 185 required primary cells, and 108 confirmatory cells. An independent focused
  frozen-identity/gate selection passed **8/8** in 16.97 s.
- Pre-execution resources are 86.49 GB free disk, 23.32 GB free / 31.75 GB total RAM,
  and 11.716 GB free on the RTX 4070. No competing QA or study process is active. The
  frozen primary workload is substantial: 185 required cells, 5-fold group-safe OOF,
  at least 925 OOF fits, 103 restoration fits, and 2,000 group bootstraps. Resume is
  deliberately unsupported; an interruption produces an immutable failed run and any
  retry must be separately justified with a new run directory.
- The single authorized foreground primary CLI then exited 1 after **106.6 s**, before
  `RunTracker` creation or matrix execution, with `PanNukePrimaryInputError: freeze
  record is not bound to this frozen primary config and manifest`. No new run directory,
  registry row, completion evidence, outcome, study process, or retained lock exists;
  the newest run remains the eligible M6 pilot. Formal status therefore remains
  `PRE_REGISTRATION_FROZEN`, and confirmatory execution remains locked.
- Read-only root-cause reconciliation found that every substantive binding matches:
  primary semantic SHA-256
  `c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15`, manifest
  SHA-256 `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e`,
  sample-order SHA-256
  `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`, completion
  stage, and cache provenance. The sole failed predicate is a stale adapter literal:
  the only freeze producer and execution gate require schema 3, but
  `pannuke_primary_inputs.py` and its tiny fixture still require/fabricate schema 1.
  No retry is authorized on the base authority. The compatibility correction must
  require exactly schema 3, pass full QA, and enter a dated pre-outcome source amendment
  before any new execution attempt.
- The compatibility correction now defines one public
  `BASE_FREEZE_EVIDENCE_SCHEMA_VERSION = 3` used by the producer, primary gate, and
  adapter; schema 1 is explicitly rejected. `PrimaryExecutionGateEvidence` now carries
  the already-verified `base_freeze_directory`, and the runner uses that base authority
  for cache/freeze binding when the active authority is a source-only amendment. It
  neither copies nor changes the immutable base freeze. Regression coverage includes
  the real default cache layout, schema-3 success, legacy-schema rejection, amendment
  base resolution, and pre-RunTracker containment.
- The first integrated focused run exposed only two new-test fixture issues and stopped
  with 21 failed / 20 passed; after correcting those tests, the focused adapter/runner/
  gate/import-order selection passed **42/42**. The final mandatory full suite passed
  **869/869 in 392.53 s**. `ruff check .`, `ruff format --check .` (141 files), and
  `mypy src` (80 source files) all pass. No primary retry, RunTracker, or outcome was
  created during this correction or QA.
- The single authorized source-only amendment command exited 0 and published exactly
  `artifacts/preregistration_amendments/20260719T011146.248393Z`, with the immutable
  base freeze as its parent, `outcomes_inspected=false`, unchanged preregistration and
  frozen-config bytes, and H1--H7 plus the four named primary/downstream analyses marked
  `amended_before_outcome_inspection`. Its artifact root is
  `962ab8b5110d062a314591f6144e0f94bebf68239f9ae8b014e2635eaf42031f`, its manifest
  SHA-256 is `f82f4b86d7cdce416108d68d72b6b71b2c4a8f8f1e6744de55962493efce0d53`, and its
  94-artifact execution-source root is
  `c0850f54e88483c1df76a4c8836343f667a7a1adbf2d05d571990cd6119cf532`.
- The exact fresh-process command `.venv\Scripts\python.exe -m histo_audit
  preregistration verify-amendment --project-root . --amendment-dir
  artifacts\preregistration_amendments\20260719T011146.248393Z` exited 0 in **5.3 s**.
  It returned `authority_status=verified_amendment`, `integrity_verified=true`, chain
  depth 1, the exact root/manifest above, and parent
  `artifacts/preregistrations/20260719T002902.432341Z`. A separate source audit found
  no execution-source drift after publication. The base and amended config snapshots
  remain byte-identical: primary file SHA-256
  `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9` and
  confirmatory file SHA-256
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`.
- The amended read-only execution gate passed and correctly resolved cache binding to
  the verified base `freeze_evidence.json`. A separate real-cache input-adapter
  preflight then executed, in order, `validate_primary_execution_gate`, frozen-config
  loading, `primary_execution_controls_from_frozen_config`,
  `default_primary_cache_paths` using `gate.base_freeze_directory`,
  `derive_primary_cache_hashes`, and `build_pannuke_primary_inputs`. It exited 0 in
  **145.6 s** (140.873 s measured inside Python), with 564 files under `artifacts/runs`
  before and after, no RunTracker, no executor, no study outcome, and no write.
- The preflight bound primary semantic SHA-256
  `c9949769ed8ab28514925ed2574958146b319d4ff848423559e0568c308cba15`, plan SHA-256
  `12a98f9dd40480927d94d8f25901392b0eb755194a0d44aebdbdb2ded26dee7f`, sample-order
  SHA-256 `2b95c283b0a76d6eada176a7cd72b7fd322f2663a1d87b929cb1559687da8d26`, and
  partition-assignment SHA-256
  `86b87d99aadec63d9aec614095c2963fdfc0ced89330e7dc98086722b483885f`. It reconstructed
  exactly 222 primary cells / 185 required, 108 confirmatory cells, 495 selected
  reference-validation groups, and 11 morphology features. Required engineered,
  context, and highlighted representations are available; pathology remains the
  preregistered optional unavailable representation with an explicit blocker and no
  cache. The manifest SHA-256 remains
  `7bf0ed664da19103c0f1119623789bc9be3f23189dabef3920bc8bd1f8c49d9e` and raw-inventory
  SHA-256 remains `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`.
- The final pre-start resource snapshot records **82.38 GiB** free disk, **24.26 / 31.75
  GiB** free/total RAM, and **11,733 / 12,282 MiB** free/total RTX 4070 VRAM. No primary
  run, registry outcome, or retained lock exists. Capacity is sufficient; the principal
  operational risk is the long, non-resumable 185-required-cell execution.

## 2026-07-21 M8 primary finalization checkpoint

- The one authorized primary execution remains active, with worker PID **20792** inside
  its original waiting launcher/wrapper process tree; no second primary invocation
  exists. Its run ID is
  `20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`.
  It must not be restarted, interrupted, hot-patched, or accompanied by a second
  primary. At the 2026-07-21 00:13 Europe/Warsaw readback the worker was responsive,
  CPU time advanced, status remained `running`, and neither `failure.json` nor
  `.immutable.json` existed.
- All **185/185 required cells** are complete and checksum-manifested; all **37 optional
  pathology cells** are explicitly skipped and no cell failure is present. The current
  process is spending deterministic CPU in the first statistics computation. The old
  implementation computes the same frozen statistics three times: aggregation,
  aggregation-internal semantic verification, and the runner's independent verifier.
  No statistics file is published until the first computation returns. The absent
  quartet proves that the first statistics computation has not returned; advancing CPU
  is consistent with the expected long pass and shows no process restart, but does not
  by itself prove algorithmic forward progress or exclude an internal stall.
- Static analysis proved that the active source cannot form a valid completion object:
  its completion-builder call omits the mandatory statistics and restoration
  attestations, while the builder rejects their absence fail-closed. The current run is
  therefore expected to finish naturally as an immutable, registry-backed failed run
  only after its old statistics work completes. It is not eligible for
  `PRIMARY_STUDY_COMPLETE`.
- A finalization-only successor implementation exists only in the isolated directory
  `C:\Users\NATAN\Documents\AANCA_successor_staging_20260720`. It verifies the failed
  predecessor and all imported hashes, physically copies only allowlisted completed
  evidence, retrains **zero** cells, recomputes only missing frozen statistics and
  restoration attestations, writes a new `retry_of_run_id` lineage, and requires an
  exact post-seal positive primary-stage attestation. It never modifies source
  annotations or the predecessor run.
- The stable integration delta is exactly **22 files**: 15 changed and 7 new, all under
  `src/` or `tests/`; there are no staging data/artifact files, junctions, symlinks, or
  reparse points. Generated Python/pytest/Ruff/mypy caches are excluded from integration.
- Full staging QA before the final two CLI-only hardening changes collected 1,069 tests:
  **1,052 passed, 15 POSIX-only skipped, 2 fixture-only failures** caused solely by the
  intentionally absent staging `.gitignore` and acquisition manifest. The same two
  assertions passed **2/2** against the real workspace with the staging source. After
  the CLI changes, the complete affected CLI suite passed **13/13**, full-package mypy
  passed for 82 source files, focused Ruff/format passed, and both public successor CLI
  help paths exited 0. Independent successor/attestation, copy-boundary, authority, and
  public-chain reviews found no remaining Option-B correctness blocker.
- A read-only resource audit measured **87.640 GiB** free on C:, a provisional sealed-
  predecessor successor allowlist of **42.741 GiB**, and the enforced successor
  preflight of imported bytes plus 10 GiB. The successor alone is therefore `GO` and
  would leave about **44.900 GiB** before its new statistics/metadata. The current
  confirmatory runner is `NO-GO` at that point: 36 required CNN cells x 5 OOF folds
  persist 180 checkpoints and then copy them again into cell evidence, giving a hard
  checkpoint-only peak lower bound of **60.170 GiB** before other outputs. Do not launch
  confirmatory under that storage contract. A fail-closed single-copy checkpoint design
  is being reviewed; otherwise at least 80 GiB must be free after the successor
  (recommended 125--130 GiB before it).
- No primary outcome was read for tuning, no raw PanNuke file or immutable authority was
  changed, and the live execution-source root remains exactly
  `c0850f54e88483c1df76a4c8836343f667a7a1adbf2d05d571990cd6119cf532`.
  Formal status therefore remains `PRE_REGISTRATION_FROZEN`, PLAN progress remains
  8/10 = 80%, and confirmatory remains locked.

## M9 forward-readiness audit (no M9 execution)

- M9 remains locked until a sealed, registry-backed, stage-eligible
  `CONFIRMATORY_COMPLETE` exists. No real original-label ranking, reviewer-asset cohort,
  blinded package, private review key, or expert response has been created.
- The existing APIs already implement group-safe OOF original-label ranking based on
  `observed_label`, without injected corruption or source-annotation modification. They
  also implement the required wording, deterministic disjoint top/random sampling,
  blinded packages, private keys outside the package, empty response templates, and
  full-patch/crop/contour asset APIs. These capabilities are unit-tested but do not yet
  constitute a production M9 chain.
- A read-only contract audit found blockers that must be corrected after M8 and before
  any M9 execution: the eligibility verifier expects experiment name
  `confirmatory_study` while the real runner writes `pannuke_confirmatory_study`; the
  verifier's direct cache-hash contract contradicts the frozen semantic-sidecar cache;
  no production producer exists for the M9 feature-cache sidecar or final-reference
  groups; one CLI `--manifest` currently has incompatible canonical-eligibility and
  private-asset roles; M9 verifiers do not yet require the run-stage eligibility ledger
  and positive confirmatory attestation; exact top/random cohort binding and a public
  reviewer-asset command are missing; and `external_validation.yaml` defaults are not
  enforced by the CLI.
- `EXTERNAL_VALIDATION_READY` may be set only after those contracts are repaired,
  covered by a positive real-structure end-to-end test, passed through full QA, and a
  real sealed ranking/assets/blinded-package chain is manually inspected. Actual expert
  responses or a responsible multi-rater analysis are still required for
  `EXTERNAL_VALIDATION_COMPLETE`; no such evidence is fabricated.

## Historical gate condition superseded on 2026-07-27

This section records the former failed-seal-only Option-B boundary. It is no longer
the current execution path; the bounded interrupted-orphan recovery section below
supersedes it.

The former gate was external execution state: PID 20792 had to terminate naturally
and the predecessor must then verify as an exact `failed`, sealed, registry-backed run.
An unsealed, completed, missing-registry, source-mismatched, or integrity-invalid
predecessor stops Option B fail-closed. Until that condition is met, integrating the
staged `src/` would change the live execution-source identity and is prohibited.

After a valid failed seal, the required order is: verify predecessor integrity; integrate
the exact 22-file receipt; run full `pytest`, `ruff check .`, `ruff format --check .`,
`mypy`, real PanNuke validation, and parent-amendment verification; update this status
and `DECISIONS.md`; publish and verify a schema-v2 outcome-blind child amendment; then
run exactly one `experiment primary-finalize-successor`. Confirmatory may start only
after the successor independently proves `PRIMARY_STUDY_COMPLETE` and its separate
storage preflight passes; the currently duplicated-checkpoint implementation is not
authorized with the measured free space.

## Historical next command (superseded)

Read-only monitoring only; do not launch another primary:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## 2026-07-27 — Windows restart left the primary as an unsealed read-only orphan

- Read-only host evidence records `LastBootUpTime` as **2026-07-27 12:37:04**
  Europe/Warsaw and the latest `Microsoft-Windows-Kernel-General` event ID 12 at
  **12:37:05**. At the subsequent inspection PID **20792** no longer existed. This is
  evidence of a host-level interruption and is not evidence that the primary reached a
  natural terminal state.
- Run
  `20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`
  still declares `status=running` and has no `failure.json`, `artifact_manifest.json`
  or `.immutable.json`. It is therefore neither an eligible failed seal nor a completed
  primary result. Its passed 185/185 required-cell reconciliation and existing
  statistics quartet are durable recovery inputs only; they do not establish
  `PRIMARY_STUDY_COMPLETE`.
- The source run is now an unsealed orphan recovery candidate that must be treated as
  immutable and read-only by operational policy despite lacking an immutable seal. No
  source artifact, status file, preregistration authority or raw PanNuke file may be
  repaired, overwritten or retroactively sealed.
- The user explicitly authorized changing the operational recovery plan. The
  failed-seal-only Option-B successor is removed from the critical path and replaced by
  one bounded recovery/resume attempt: verify source identity, complete inventory and
  hashes; import only an explicit allowlist into a new run with `retry_of_run_id`;
  calculate only missing finalization attestations once; then use the ordinary
  completion, seal, integrity and registry gates. There is no automatic retry, training
  fallback or second primary.
- Frozen scientific definitions remain unchanged: hypotheses, group-safe OOF rules,
  final-reference isolation, corruption/label separation, exclusions, estimands,
  restoration analysis, confirmatory definitions and mandatory terminology are not
  reopened. The operational correction requires a dated technical amendment with its
  reason, scope, source-run binding and impact on confirmatory eligibility.
- Formal status remains exactly `PRE_REGISTRATION_FROZEN`; M0–M7 remain closed,
  progress remains **8/10 = 80%**, M8 remains open, and neither
  `PRIMARY_STUDY_COMPLETE` nor `CONFIRMATORY_COMPLETE` is claimed.

## Next exact command (interrupted-orphan recovery implementation)

```powershell
.venv\Scripts\python.exe -m pytest tests\test_primary_recovery.py -q
```

## 2026-07-27 — Outcome-inspection declaration corrected after accidental broad search

- At **2026-07-27T10:57:07Z**, an over-broad read-only `rg` command intended to locate
  PID/process receipts also traversed the orphan run and emitted fragments of real
  primary subgroup/statistics/ranking values. No values are repeated here.
- Before that command, the audit found no real primary outcome values in
  `STATUS.md`, `DECISIONS.md`, stored test logs, `run.log`, `events.jsonl` or
  `report.md`; B-fast testing used synthetic fixtures. Nevertheless, the literal
  declaration `outcomes_inspected=false` ceased to be defensible at the timestamp
  above.
- Every recovery authorization and technical amendment must therefore record
  `outcomes_inspected=true`, the timestamp and this accidental-exposure reason. The
  affected recovered primary analysis is classified `amended_or_exploratory`, not an
  original unamended confirmatory result.
- The exposed values are prohibited from informing code, thresholds, model choice,
  exclusions, hypotheses, confirmatory configuration or any other scientific tuning.
  The recovery remains a byte-preserving operational repair and no outcome value is
  used or reported while implementing it.
- Formal status remains `PRE_REGISTRATION_FROZEN`; M8 remains open and
  `PRIMARY_STUDY_COMPLETE` is not claimed.

## 2026-07-27 — Recovery evidence and single-pass statistics gate

- Fresh read-only typed qualification of the orphan passed:
  `read_primary_filesystem_evidence(...)` found **185/185** required cells complete,
  **37** optional cells skipped, and zero failed, missing, extra or invalid cells; its
  readback root is
  `7ce192b40d241fd2c8b394c1e03905d51a5c654b1373646a5324768a08cc4039`.
  `read_primary_restoration_evidence(...)` also passed and bound the same source root.
  The two typed calls took 120.740 s and 144.160 s respectively.
- The four statistics artifacts are present and their manifest-declared payload
  SHA-256/size records passed read-only rehash. This records structure and integrity
  only; no result value is reported here.
- Host interruption evidence is saved at
  `artifacts/process_observations/20260727T110704.4225163Z_primary_orphan_boot_receipt.json`
  with SHA-256
  `2cb53b1efe1ab6441ff9fa6b93c929fb7a5395bd52a20556a3149366fc46d8cc`.
- The primary statistics producer was corrected so the heavy computation executes
  exactly once. The public independent full verifier remains available but is no
  longer invoked twice by the normal aggregation/runner path.
- Executed focused gates in the isolated staging source:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
.venv\Scripts\python.exe -m pytest -q tests\test_primary_statistics.py tests\test_primary_runner.py --basetemp C:\Users\NATAN\AppData\Local\Temp\aanca_primary_single_pass_final2_d3c0cf5b42674672bde89db7f5dbce25
.venv\Scripts\python.exe -m ruff check src\histo_audit\experiment\primary_statistics.py src\histo_audit\experiment\primary_runner.py tests\test_primary_statistics.py tests\test_primary_runner.py
.venv\Scripts\python.exe -m ruff format --check src\histo_audit\experiment\primary_statistics.py src\histo_audit\experiment\primary_runner.py tests\test_primary_statistics.py tests\test_primary_runner.py
.venv\Scripts\python.exe -m mypy src\histo_audit\experiment\primary_statistics.py src\histo_audit\experiment\primary_runner.py
```

Results: **34 passed in 61.08 s**; Ruff check passed; all four files were already
formatted; mypy reported no issues in the two source files. These are focused gates,
not the still-required final full-suite gates.

## 2026-07-21 current authoritative checkpoint (supersedes earlier same-day tails)

- Option B remains isolated and outcome-blind. Its post-fix executable/test receipt is
  exactly **29 Python files (22 modified, 7 added)**. Using canonical records
  `kind<TAB>POSIX-path<TAB>lowercase-file-SHA256<LF>`, sorted by path and encoded as
  UTF-8 without BOM, the current receipt SHA-256 is
  `9f897651a2911cf5ca924f4864aee3ed0d8076589eef149219eba7b6f8691d54`.
  This receipt excludes staging documentation and all Python/pytest/Ruff/mypy caches.
- Post-fix evidence is: **102/102 focused tests passed**, full Ruff passed, all 148 files
  are formatted, full configured mypy passed for 82 source files, all required CLI help
  paths exited 0, independent review found **0 P0/P1**, and the real full-semantic
  PanNuke validator exited 0 in 417.308 seconds. One final full `pytest` is currently
  running against this unchanged receipt; this section does not claim that gate before
  its terminal summary exists.
- Read-only M9 readiness review confirmed seven contract blockers but separated their
  timing. Before real confirmatory M8, the only required M9-compatibility correction is
  to bind `final_reference_group_ids_sha256` to the unique group set
  `sorted(set(final_group_ids))` in producer, runner verification and completion
  readback, while leaving sample-aligned partition hashes unchanged. Repeated nuclei
  from the same patch currently make the M8 multiset hash incompatible with M9's unique
  group-file contract. This technical source change requires a verified amendment
  before confirmatory execution; it does not block the primary finalization successor.
- After an attested `CONFIRMATORY_COMPLETE` and before any M9 execution, the remaining
  work is to: distinguish real run name `pannuke_confirmatory_study` from semantic config
  name `confirmatory_study`; support the frozen semantic-sidecar cache authority; create
  outcome-independent producers for the exact feature provenance and final-group file;
  split canonical eligibility manifest from reviewer-asset manifest; require
  `require_run_stage_eligible` and carry positive-attestation hashes; freeze the exact
  top/random cohort before assets so missing assets cannot substitute other samples;
  and enforce the already frozen `configs/external_validation.yaml` contract (100 top,
  100 random, seed 707, exact roles/options). M9 remains locked and no ranking, assets,
  private key, package or expert response has been produced.
- PID **20792** remains the sole active primary worker, responsive and consuming CPU;
  no statistics quartet, `failure.json` or `.immutable.json` exists. It remains
  prohibited to integrate source, publish the real amendment, or execute the successor
  until this predecessor naturally terminates as an integrity-valid, registry-backed
  `failed` seal.
- Formal status remains `PRE_REGISTRATION_FROZEN`; PLAN progress remains **8/10 = 80%**.

## Next exact command (authoritative)

Receive the already running final pytest and monitor only the existing primary:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## 2026-07-21 Option-B P1 closure and final QA rerun

- Independent final review found two P1 defects in the isolated candidate before any
  live integration or execution. The first allowed a self-consistent checkpoint rename
  and did not discover a non-`.pt` physical copy inside a `checkpoints/` directory. The
  second primary/successor report used similar wording but omitted the exact mandatory
  phrase `recommended for expert review`.
- Both defects are closed in staging. Confirmatory readback now requires the exact
  `checkpoints/fold_{fold_id:02d}.pt` path, derives the expected global set independently
  from completed CNN cells x registered fold IDs, and compares it against every regular
  file inside checkpoint directories as well as every `.pt` in the run. Regression
  tests cover a rename plus manifest rehash and an extra `backup.bak` copy. The shared
  primary/successor report now contains both `potentially inconsistent annotation` and
  `recommended for expert review`, with coverage through ordinary primary and the
  finalization-only successor path.
- Focused post-fix tests passed **102/102**: confirmatory filesystem completion **39**,
  primary runner **25**, and finalization successor **38**. Focused Ruff and format
  passed. The repeated complete static/CLI gate also passed: `ruff check .`,
  `ruff format --check .` (**148 files**), configured `mypy` (**82 source files**), and
  root/amendment/successor/confirmatory help all exited 0. An independent read-only
  re-review reports **0 P0/P1** and confirms that the integration receipt remains
  exactly 29 `src/tests` Python files.
- The mandatory real PanNuke validator passed with exit 0 in **417.308 s** over all
  3 folds, 7,901 patches and 22 raw files. It reported 4,318 cross-class-overlap pixels
  in 575 patches, 10,486,091 void pixels in 162 patches, 1,411 affected/excluded
  instances, `no_class_arbitration=true`, `source_masks_modified=false`, identical
  primary/confirmatory exclusion policy, idempotent publication, and no residual lock,
  staging directory or validator process.
- The first full-suite attempt was deliberately stopped at 24% only after the P1
  findings made that candidate obsolete; the stop was limited to its own test process
  tree and PID 20792 remained alive. One final full `pytest` is now running against the
  corrected stable candidate with a four-hour limit. No staging edit is permitted until
  it returns.
- The active primary remains untouched and formal status remains
  `PRE_REGISTRATION_FROZEN`, with PLAN progress **8/10 = 80%**. A real schema-v2
  amendment, successor and confirmatory execution remain prohibited until the existing
  primary naturally produces a verified registry-backed failed seal.

## Next exact command (QA and primary monitoring)

Receive the already running final pytest and continue read-only monitoring of the
already active primary; do not start or patch another primary:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## 2026-07-21 Option-B integration candidate and final staging gates

- The isolated Option-B candidate is now a stable receipt of exactly **29 Python
  files: 22 modified and 7 new** relative to the live workspace. The earlier 22-file
  count predated the complete confirmatory storage-policy and attestation hardening and
  is superseded by this receipt. No generated cache, dataset, run artifact, frozen
  authority, or documentation file is part of the executable integration delta.
- The candidate adds a typed schema-v2, pre-outcome-inspection confirmatory storage
  policy. Its canonical SHA-256 is
  `d67fb56a3d51d9748998f75baa3f18ab9468a7c231f7b492a98d3bdea021e3ff`.
  The policy is accepted only from an exact finalization-predecessor amendment with
  `outcomes_inspected=false`; it is bound independently into the confirmatory gate,
  input bindings, provenance, completion evidence, metrics and pre/post-seal
  readbacks. Missing, inherited, stale, duplicated or mismatched policy evidence is
  rejected fail-closed.
- Real confirmatory checkpoint persistence is single-copy and canonical at
  `<run>/cells/<cell_id>/checkpoints/fold_XX.pt`. A real run rejects an external
  checkpoint root before model execution, forbids copy fallback, requires one regular
  private-link-count-one file per completed CNN cell/fold, and reconciles the complete
  filesystem `.pt` set against all manifests. The legacy top-level
  `<run>/checkpoints` tree is forbidden. This removes approximately **30 GiB** of
  stable duplicate checkpoint storage without changing a model, fold, seed,
  prediction, metric, estimand, or restoration analysis.
- Amendment and successor evidence publication now uses no-overwrite physical copies,
  not hardlinks. Source objects and destination parents are anchored, copied bytes are
  size/hash verified, final publication is atomic and no-replace, rollback is
  ownership-safe, and every published authority leaf (including marker and manifest)
  must be regular, non-reparse and `st_nlink == 1` before semantic verification.
- Focused regression suites for successor authorization, publication faults,
  amendment verification, storage-policy binding, single-copy checkpoints and
  independent attestations are green. The final static/CLI staging gate is also green:
  `ruff check .`, `ruff format --check .` (**148 files**), full configured `mypy`
  (**82 source files**), CLI root help, amendment help, successor help and confirmatory
  help all exited 0. The final full `pytest` and the real PanNuke validator are running
  once against the stable candidate; no staging edit is permitted while they run.
- At the latest read-only check PID **20792** remained responsive with increasing CPU.
  Its status is still `running`; the statistics quartet, `failure.json` and
  `.immutable.json` are still absent. Therefore no integration, real amendment,
  successor or confirmatory execution is yet authorized. The live source tree, active
  run and raw PanNuke files remain unchanged.
- Formal status remains `PRE_REGISTRATION_FROZEN`; PLAN remains **8/10 = 80%**. The
  next transition is still conditional on a natural, integrity-valid,
  registry-backed `failed` seal for the predecessor, followed by the complete original
  workspace gates and a verified schema-v2 child amendment.

## Next exact command (unchanged execution boundary)

Continue read-only monitoring of the already active process/cell; do not start a second
primary and do not integrate staged executable source before terminalization:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## Historical 2026-07-21 status pointer (superseded)

At that date, the same-day section `current authoritative checkpoint (supersedes
earlier same-day tails)` was the controlling record. It is superseded by the
2026-07-27 interrupted-orphan recovery sections. Historically, the corrected 29-file
Option-B receipt was
unchanged; focused tests, full static/CLI checks, independent review and the real
PanNuke validator pass; one final full `pytest` is still running; PID 20792 remains the
sole live primary without a terminal seal. Formal status is
`PRE_REGISTRATION_FROZEN`, 8/10 = 80%, and no real amendment, successor, confirmatory
or M9 execution is yet authorized.

## 2026-07-21 final Option-B staging gate PASS

- The corrected stable 29-file receipt passed the final Windows test gate. One direct,
  spawn-safe `python -m pytest` command with a short `C:\pt3` base exited 0 with
  **1,098 passed, 15 POSIX-only skipped and 2 explicitly deselected** in 603.78 seconds.
  The two deselected provenance tests depend on the real repository `.gitignore` and
  acquisition manifest, so they were run separately from the original workspace with
  staging `PYTHONPATH` and passed **2/2** in 4.43 seconds. Thus all **1,100 executable
  tests** in the 1,115-collected suite passed; no test failure remains.
- A preceding diagnostic run exposed two launcher-only hazards rather than code defects.
  Programmatic `pytest.main()` launched from `python -` is unsafe for Cleanlab's Windows
  multiprocessing and spawned 16 workers at its optional-integration test; direct
  `python -m pytest` made the same test pass in 4.76 seconds. A deliberately long
  basetemp caused six Windows path-length failures; all six passed in 8.24 seconds with
  a short base before the clean full rerun. No scientific code was changed for either
  launcher issue.
- All staging gates are now terminal and green: focused tests **102/102**, final full
  test coverage as above, full Ruff, format, configured mypy, required CLI paths, real
  PanNuke validation, and independent review **0 P0/P1**. The Option-B staging tree is
  frozen pending integration.
- PID **20792** remains responsive and CPU continues to increase. Only `status.json`
  exists among the terminal/statistics artifacts checked; there is still no statistics
  quartet, `failure.json` or `.immutable.json`. Therefore integration remains correctly
  locked despite the green staging gate.
- C: currently has 81,530,183,680 free bytes after test temporaries. This exceeds the
  successor's hard imported-bytes-plus-10-GiB preflight but is below the preferred
  combined successor/confirmatory operating margin. Automated removal of old pytest
  directories was blocked before execution by tool policy, so no directory was
  deleted. Reclaim space before confirmatory if the live preflight requires it.
- Formal status remains `PRE_REGISTRATION_FROZEN`; PLAN remains **8/10 = 80%**. The
  next action is now purely conditional on the existing primary's natural terminal
  failed seal; no second primary is authorized.

## Next exact command (final staging complete)

Use the already active monitor only:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## 2026-07-21 pre-M8 group-set compatibility gate PASS

- The one correction identified as mandatory before real confirmatory M8 is now included
  in the isolated candidate. Producer, runner evidence and independent completion
  readback compute `final_reference_group_ids_sha256` from
  `sorted(set(final_group_ids))`. Sample-aligned `group_ids_sha256`, the full repeated
  group vector in NPZ evidence, sample order, labels, splits, corruption, restoration,
  statistics and outcomes are unchanged.
- Regression evidence includes multiple nuclei sharing one final-reference patch and
  proves that the unique-set hash is accepted while the previous duplicate-preserving
  multiset hash is rejected. The three affected modules passed **63/63** tests; an
  independent review and dynamic writer/readback checks report **0 P0/P1**.
- Final post-change gates pass: full Ruff, all **148 files** formatted, configured mypy
  over **82 source files**, required CLI help paths, direct spawn-safe full pytest
  **1,098 passed / 15 POSIX-only skipped / 2 provenance-only deselected** in 616.24 s,
  and the two provenance tests separately passed **2/2** in 4.69 s from the original
  workspace. All 1,100 executable tests therefore pass under the final source.
- The integration delta remains exactly **29 Python files (22 modified, 7 added)**.
  Under the documented canonical receipt encoding, its new final SHA-256 is
  `65cac6d97e3770cba89d3140ca53bc90b55f25c1c854519ff4c80cba858f68c4`,
  superseding the earlier pre-compatibility receipt.
- PID **20792** remains responsive with increasing CPU and no terminal/statistics
  artifacts beyond `status.json`. Current free C: space is 75,639,230,464 bytes after
  QA temporaries. This still exceeds the successor hard preflight, but space must be
  reclaimed before confirmatory if its live preflight lacks the required margin.
- Formal status remains `PRE_REGISTRATION_FROZEN`, 8/10 = 80%. The candidate is frozen
  again; only a natural, integrity-valid, registry-backed failed predecessor seal can
  authorize integration and the real outcome-blind amendment.

## Next exact command (all isolated implementation gates complete)

Continue the existing read-only monitor only:

```powershell
$run = "artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f"
Get-Process -Id 20792 -ErrorAction SilentlyContinue
Get-Content "$run\status.json"
Test-Path "$run\failure.json"
Test-Path "$run\.immutable.json"
```

## 2026-07-27 — Bounded recovery core focused gate PASS

- `PLAN.md` now has an explicit anti-loop execution budget: the obsolete
  failed-seal/finalization-successor path cannot be re-entered; preflight creates no
  run; the real path has one source qualification, one physical copy, one destination
  verification, one seal/integrity/attestation sequence, and no automatic retry.
- The isolated recovery core and its synthetic/fault tests remain outside the live
  executable source pending integration. Root independently executed:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q tests\test_primary_recovery.py --basetemp C:\Users\NATAN\AppData\Local\Temp\aanca_recovery_root_review
```

- Result: **17 passed in 5.23 s**. Coverage includes exact recovery authorization,
  truthful `outcomes_inspected=true`, `amended_or_exploratory`, inactive PID/status/
  seal checks, exact allowlisting, source nonmutation, one independent physical copy,
  lightweight statistics closure, tamper/missing failures, and runtime/static proof
  that training, aggregation, `_compute_statistics`, fallback and retry are absent.
- A focused integration review found one fail-closed blocker before the runner can be
  accepted: the existing inherited-numeric capability is named and bound only for a
  sealed failed finalization predecessor. The reboot orphan has no artifact seal or
  artifact-manifest root. Recovery will therefore use a separately named
  recovery-specific capability/provenance; snapshot hashes will not be mislabeled as
  sealed predecessor hashes.
- Formal status remains `PRE_REGISTRATION_FROZEN`, M8 remains open, and progress
  remains **80%**.

## Next exact command

After the recovery-specific statistics capability is present:

```powershell
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q tests\test_primary_recovery.py tests\test_primary_recovery_statistics.py
```

## 2026-07-27 — Bounded recovery orchestration and storage redesign

- The isolated recovery cluster now has a public preflight and a one-shot runner.
  Root executed the focused recovery, statistics, runner, stage, CLI, amendment and
  frozen-identity suites from the staging source; the result was **107 passed in
  90.29 s**. After three additional adversarial runner tests, the runner suite passed
  **10/10** in 13.32 s. Those tests cover pre-tracker trust rejection, exactly one
  completion-seal attempt and a terminal positive attestation with no later file
  reads.
- The runner has no import from the legacy primary retry/finalization runner and no
  training, matrix-executor, aggregation, fallback or automatic-retry path. Its
  recovery evidence has an exact 22-field schema. A source/destination boundary
  ambiguity remains unsealed and default-deny; a partial completion seal is never
  followed by a second failure-seal attempt.
- A fresh disk check found only **31,623,032,832 free bytes** on C:, below the
  46,291,408,111 logical bytes in the orphan run. Therefore the previous
  uncompressed-copy preflight cannot be used and no real recovery was started.
- Windows WOF/LZX was tested on temporary physical copies without reading result
  values and without changing the source. A 329,633,152-byte JSON occupied
  14,241,792 bytes after compression (**23.1:1**); a 372,330,793-byte NPZ occupied
  125,747,200 bytes (**3.0:1**); and a 24,112,644-byte CSV occupied 8,073,216 bytes
  (**3.0:1**). The temporary probes were deleted after measurement. Ordinary NTFS
  compression is disabled on this volume, so only the explicitly tested WOF/LZX
  mode is eligible.
- The selected storage redesign is still a byte-identical physical no-overwrite
  copy: each destination file will be durably copied, checked, compressed exactly
  once with WOF/LZX, and rehashed through a retained handle before the next file.
  Free space will be checked before every file and after every compression with a
  fixed safety margin. Hardlinks, source mutation and manifest-only references remain
  forbidden.
- Formal status remains `PRE_REGISTRATION_FROZEN`; M8 remains open and progress
  remains **8/10 = 80%**. The recovery code is still isolated pending WOF fault tests,
  downstream confirmatory identity tests, selective integration and the full live
  gates.

## Next exact command

After the WOF copy patch is complete:

```powershell
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q tests\test_primary_recovery_runner.py tests\test_primary_recovery.py tests\test_pannuke_acquisition.py
```

## 2026-07-27 — Authoritative bounded-recovery checkpoint after live integration

- This section supersedes every earlier `Next exact command`, PID-monitor,
  failed-seal Option-B, finalization-successor and outcome-blind recovery instruction
  in this file. Those entries remain historical evidence and must not be executed.
  The obsolete `aanca-primary-to-option-b` heartbeat automation was deleted so it
  cannot re-enter that retired path.
- Live recovery/WOF integration is complete. Focused recovery/amendment/attestation
  tests passed **95/95**, focused confirmatory completion/runner/downstream tests
  passed **62/62**, and the recovery CLI tests passed **7/7**.
- Mandatory live gates passed against the stabilized source:
  - `.venv\Scripts\python.exe -m pytest` — **949 passed in 453.93 s**;
  - `.venv\Scripts\python.exe -m ruff check .` — passed;
  - `.venv\Scripts\python.exe -m ruff format --check .` — **150 files already
    formatted**;
  - `.venv\Scripts\python.exe -m mypy src` — **no issues in 82 source files**.
- The mandatory real-data command
  `.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root .
  --root data\raw\pannuke` exited 0 in **306 s** with `status=valid`,
  `validation_scope=full_semantic_scan` and `publication=idempotent`. It verified
  3 folds, 7,901 patches and 22 raw files; 4,318 cross-class-overlap pixels in
  575 patches; 10,486,091 void pixels in 162 patches; 1,411 affected/excluded
  instances; `no_class_arbitration=true`; and `source_masks_modified=false`.
- Git ignore readback continues to classify the raw PanNuke fold directories/ZIPs,
  nested manifest Parquet files and duplicate-audit NPZ files as ignored. Raw data
  and the orphan run were not modified.
- The first amendment-publication invocation stopped before creating any authority
  with `_AuthorityValidationError: primary recovery outcome-inspection timestamp
  differs from the amendment`. The authorization used
  `2026-07-27T10:57:07Z`, while the amendment API canonically renders the identical
  instant as `2026-07-27T10:57:07.000000Z` and requires byte-identical text.
  Exactly one process ran, no amendment directory was created, and no recovery
  RunTracker/copy/training/fallback/retry was started.
- The control artifact was corrected only to the canonical microsecond rendering.
  This is a deliberate pre-publication governance correction, not an automatic
  recovery retry. A second publication attempt is authorized once; any further
  failure stops for new evidence and a recorded decision.
- Formal status remains exactly `PRE_REGISTRATION_FROZEN`; progress remains
  **8/10 = 80%** and M8 remains open.

## Next exact command (authoritative; execute once)

```powershell
.venv\Scripts\python.exe artifacts\recovery_control\prepare_recovery_amendment_once.py
```

On success, verify the new amendment in a fresh process and execute exactly one
`experiment primary-orphan-recovery --preflight-only`. On failure or ambiguity, do
not rerun automatically and do not launch recovery or a second primary.

## 2026-07-27 — Recovery authority and read-only preflight passed

- The corrected publication exited 0 and created immutable amendment
  `artifacts/preregistration_amendments/20260727T133947.089370Z`, chain depth 2.
  Its artifact root is
  `4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a`
  and manifest SHA-256 is
  `b5efc656f074b2933138b1a623de24e099bea9e7e2b75edc8e484d22ca176d10`.
  A separate fresh-process `preregistration verify-amendment` exited 0.
- The authority binds `outcomes_inspected=true`,
  `analysis_disposition=amended_or_exploratory`, the exact orphan source,
  185 required completed cells, 37 optional skipped cells, zero retraining and
  `ConfirmatoryStoragePolicy()`.
- Exactly one public `primary-orphan-recovery --preflight-only` exited 0 in
  **291.6 s**. It re-established source snapshot root
  `bc224f73960792a495e03e5039c075c0061ee3d86cce095afb21a91a061cc027`,
  2,270 allowlisted artifacts and 46,291,340,622 logical copy bytes.
- Preflight safety flags were all exact:
  `run_tracker_created=false`, `copy_invoked=false`,
  `matrix_executor_invoked=false`, `training_invoked=false`,
  `fallback_invoked=false`, `automatic_retry_allowed=false`,
  `study_outcome_eligible=false`, and `completion_stage=null`.
- The WOF disk gate passed with 26,847,830,016 observed free bytes,
  11,109,749,033 required bytes, a 372,330,793-byte largest next artifact and a
  10,737,418,240-byte fixed margin. No recovery destination exists yet.
- Formal status remains `PRE_REGISTRATION_FROZEN`; the real one-shot recovery is now
  authorized but has not yet run.

## Next exact command (authoritative one-shot execution; never auto-repeat)

```powershell
.venv\Scripts\python.exe -m histo_audit experiment primary-orphan-recovery `
  --project-root . `
  --authority-dir artifacts\preregistration_amendments\20260727T133947.089370Z `
  --runs-root artifacts\runs `
  --run-id 20260727T133947.089370Z_pannuke_primary_orphan_recovery
```

If the command fails or its terminal state is ambiguous, preserve the new run and
stop. Do not invoke the command a second time and do not automatically launch primary.

## 2026-07-27 — Bounded orphan recovery completed and independently qualified

- The authoritative one-shot command
  `.venv\Scripts\python.exe -m histo_audit experiment primary-orphan-recovery
  --project-root . --authority-dir
  artifacts\preregistration_amendments\20260727T133947.089370Z --runs-root
  artifacts\runs --run-id
  20260727T133947.089370Z_pannuke_primary_orphan_recovery` returned naturally exactly
  once with exit code 0 after **1658.5 s** and empty stderr. The tracked run work
  completed in **1216.982 s**.
- The new run is
  `artifacts/runs/20260727T133947.089370Z_pannuke_primary_orphan_recovery`, with exact
  `retry_of_run_id=20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`.
  The predecessor remains unchanged, unsealed, read-only, and ineligible.
- Recovery copied and rehashed the authorized **2,270 artifacts** and
  **46,291,340,622 logical bytes** with streaming WOF/LZX. It reused all **185/185**
  required cells, retained the **37** declared optional skips, retrained **0** cells,
  and recorded `training_invoked=false`, `matrix_executor_invoked=false`,
  `fallback_invoked=false`, and `automatic_retry_allowed=false`.
- The completed no-overwrite seal contains **2,288 artifacts**. Artifact root SHA-256
  is `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`;
  artifact-manifest SHA-256 is
  `9abff1b2f0e745a50b3aa1922d3d725bb629276f6090c32e9b423fea82d0e0ce`;
  completion-evidence SHA-256 is
  `77a02877f9882a608968303e077963abdfbcfabdc997424cfaceadf0df86349c`;
  recovery-evidence SHA-256 is
  `3f937dac3dc2788131dafc84ce35ae7fda4a24a35bcf437ee14827169b8ccaf1`.
- Two separate fresh-process audits passed without printing scientific outcome values.
  They reverified the complete amendment chain, exact artifact root, manifest,
  immutable registry record, recovery authorization/evidence, restoration/statistics
  bindings, and positive stage ledger. The stage-attestation record SHA-256 is
  `5af827544502fbdf688a73916ec58b5dac0984c5a682a33ce6dfc97538228871`;
  its verification SHA-256 is
  `e19f17b14a375e8e048e1d2fc3060c19831e4259c8311831d88f75753024c768`.
- The formal completion status is now `PRIMARY_STUDY_COMPLETE`. The recovered primary
  analysis remains permanently `amended_or_exploratory`; it is not represented as the
  original unamended primary result. M8 remains active until the unchanged frozen
  confirmatory study qualifies as `CONFIRMATORY_COMPLETE`, so milestone progress
  remains **8/10 = 80%**.
- The one-shot recovery authorization is consumed. Do not rerun recovery and do not
  start another primary. The old recovery command above is historical evidence, not a
  next command.
- Git still has no HEAD and zero tracked paths. Raw PanNuke ZIP/NPY, nested manifest
  Parquet, duplicate-audit NPZ, and run payloads are ignored. Before any future broad
  staging, large generated QC/provenance reports must be reviewed explicitly rather
  than added accidentally.

## Next exact action (bounded confirmatory readiness)

Do not start the real confirmatory run yet. First close the detected fail-closed
functional blocker: the public `experiment confirmatory` command currently omits the
required crop-cache and provenance arguments expected by its runner. Add only
outcome-independent CLI/input/capacity plumbing, test it, pass the mandatory gates,
record a dated technical amendment without changing frozen scientific definitions,
and run one read-only confirmatory preflight before any RunTracker is created.

## 2026-07-27 — Resource-bounded confirmatory redesign in progress

- This section supersedes the preceding bounded-confirmatory next action. The sealed
  Option-B primary remains the sole accepted primary and the formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`.
- Confirmatory cache/input and lifecycle-rehearsal plumbing is integrated. The
  combined focused command
  `.venv\Scripts\python.exe -m pytest tests\test_confirmatory_cli_inputs.py
  tests\test_lifecycle_qualification.py tests\test_confirmatory_runner.py
  tests\test_stage_cli.py tests\test_workflow_stage_apis.py -q` passed
  **61/61** in **93.12 s**. Focused Ruff check and format check passed, and a separate
  focused mypy invocation reported no issues in the four changed source modules.
- RAM-safe image preparation now retains a C-contiguous `uint8` RGB buffer and a
  C-contiguous boolean target-mask buffer without float64 expansion. Non-contiguous
  inputs receive one contiguous copy. CNN preflight fingerprints are computed once
  per `(outer_fold, corruption_cell_id, input_variant)` and copied into every
  affected cell without changing hashes. New focused tests passed **3/3**; the full
  CNN/runner pair passed **18/18**; Ruff and format checks passed.
- A read-only execution audit classified the original frozen 108-cell confirmatory
  study as operational `NO-GO` on this host: its runner contradicts the frozen
  checkpoint/resume requirement, 180 full CNN checkpoints have an approximately
  30.0 GiB lower bound, and measured throughput implies at least about 5.5 days
  (typically 10–15 days). No confirmatory run or second primary was started.
- The selected replacement is a separate strict 24-required-cell
  `resource_bounded_confirmatory_v1` sensitivity/feasibility profile with one context
  CNN, three frozen ImageNet representations, seed 303, both corruption cells, all
  three rotations, group-safe five-fold OOF, at most four CNN epochs, 2,000 group
  bootstraps, and the unchanged restoration budget. Estimated execution is
  11–16 hours and 8–12 GiB stable storage before the fixed 10 GiB safety margin.
- This is a post-outcome amendment and is permanently
  `amended_or_exploratory`. It cannot claim the original frozen confirmation:
  `original_confirmatory_claim_allowed=false`, `completion_stage=null`, no positive
  `CONFIRMATORY_COMPLETE` attestation, no M9 unlock, and project progress remains
  **8/10 = 80%**.
- Implementation is currently split among non-overlapping owners: strict profile and
  matrix contract; historical-primary/current-execution dual authority; and an
  explicit read-only checkpoint-successor module. Existing frozen authorities and
  configs are not being modified, and no amendment has yet been published.
- The strict profile handoff is now independently green: exact semantic SHA-256
  `1c9a41b92dabbeafbb92b1bc8aced158337046fc1d6e056b011f6a27b98e8298`,
  24 required / 0 optional cells, **19/19** focused tests in root's rerun, and
  focused mypy with no issues. Broader owner suites passed 44 and 173 tests.
- The schema-v4 resource authority and dual historical-primary/current-execution
  gate are complete. Combined new/legacy tests passed **56/56**; mypy, Ruff and
  format passed. A real read-only historical-primary validation passed in **381 s**,
  re-establishing the exact 185-cell recovered run, artifact root
  `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`,
  `amended_or_exploratory` disposition, and positive primary stage record
  `5af827544502fbdf688a73916ec58b5dac0984c5a682a33ce6dfc97538228871`.
- The explicit checkpoint-successor module passed **12/12** in both its owner and
  root reruns, plus mypy, Ruff and format. Adversarial review found and the owner
  fixed a fabricated typed snapshot, fabricated receipt, and parent
  symlink/junction escape before integration. Fresh evidence records exactly 30
  `missing_fresh` paths, zero imported bytes, no predecessor read and no retry ID.
- No resource authority has been published. Its publisher intentionally remains
  fail-closed until the new runner, CLI and every path in the closed execution-source
  delta are integrated and pass the complete QA sequence.

## Next exact command

After those three isolated patches are handed off and integrated, run their combined
focused suites before any publication or scientific execution:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_resource_bounded_confirmatory_contract.py tests\test_resource_bounded_authority_gate.py tests\test_resource_bounded_resume.py
```

## 2026-07-27 — Resource-bounded final QA and pinned source receipt passed

- The sealed Option-B recovery remains the sole accepted primary. Its artifact root is
  `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`;
  the interrupted source run remains unchanged, unsealed, read-only, and ineligible.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`, M8 remains open, and
  progress remains **8/10 = 80%**.
- The integrated focused resource suite passed **150/150** in **80.84 s**. It covers
  the fixed profile, schema-v4 dual authority, checkpoint resume, runner,
  full execute/seal fault paths, public CLI, lifecycle qualification, and PanNuke
  confirmatory inputs. Independent runner safety tests passed **99/99** and resume
  qualification tests passed **61/61**. No P0/P1 finding remains.
- The mandatory current-tree code gates passed:
  - `.venv\Scripts\python.exe -m pytest -q` — **1086 passed in 514.21 s**;
  - `.venv\Scripts\python.exe -m ruff check .` — passed;
  - `.venv\Scripts\python.exe -m ruff format --check .` — **162 files already
    formatted**;
  - `.venv\Scripts\python.exe -m mypy src` — **no issues in 86 source files**;
  - `.venv\Scripts\python.exe -m histo_audit experiment
    resource-bounded-sensitivity --help` — exited 0.
- The full `.venv\Scripts\python.exe -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke` process completed naturally with empty
  stderr and `status=valid`, `validation_scope=full_semantic_scan`, and
  `publication=idempotent`. It again covered all three folds and 7,901 patches,
  reported 4,318 cross-class-overlap pixels, 10,486,091 void pixels, and 1,411
  identically excluded overlap-touching instances, with
  `no_class_arbitration=true` and `source_masks_modified=false`. The asynchronous
  launcher did not retain the OS exit-code object after process exit; the complete
  success JSON and empty stderr are retained as the functional result, and the
  unchanged passing command is not repeated solely to manufacture another log.
- The final parent-P-to-live execution-source delta is exactly **15 paths**:
  **5 added, 10 modified, 0 removed**, matching the closed allowlist. Its pinned
  identities are:
  - execution-source root
    `1179f91725a3027c0397e87691774377bbd4ba5469d588390c72b0b88515547b`;
  - execution-source manifest
    `03bcc6020e3be5a22fe257c45820e4e8ebece3ce471c2b6cecff0e3e9419fc66`;
  - source-delta SHA-256
    `7abd9e1627728c4ce89f59cc6162283ec8963468816db6c64849fff1a5ec290e`.
- The ignored one-shot controller now pins those three identities. Its first pinned
  lightweight preflight correctly failed before any write because one manually
  transcribed historical-primary immutable-marker hash contained 62 rather than 64
  characters. The exact existing marker hash was restored, a canonical-hash-length
  regression was added, and the controller suite passed **9/9** with Ruff and format
  checks passing.
- The corrected `--preflight-only` then exited 0 with
  `status=preflight_passed`, `publication_performed=false`,
  `live_primary_builder_invoked=false`, and `automatic_retry_allowed=false`. It
  matched parent amendment root
  `4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a`,
  the recovered-primary root, resource-config file SHA-256
  `783968e8afc132cca0c877aadf953fc68d3c35f606021b3a97ed380478dbad4a`,
  and resource-config semantic SHA-256
  `1c9a41b92dabbeafbb92b1bc8aced158337046fc1d6e056b011f6a27b98e8298`.
  No attempt/success/failure marker exists, and the amendment inventory still
  contains only the two pre-existing authorities. No resource study has run.

## Next exact command (one-shot authority publication)

Execute once. A failure or ambiguity preserves its marker/evidence and stops the
critical path; do not invoke it again automatically:

```powershell
.venv\Scripts\python.exe artifacts\resource_control\prepare_resource_authority_once.py --publish-once
```

## 2026-07-27 — Resource authority C published once and verified

- The only `--publish-once` invocation completed naturally in **517.9 s** with
  `status=published_and_verified`. The durable attempt and success markers exist;
  no failure marker exists, `automatic_retry_allowed=false`, and the controller
  reports exactly one builder call and one amendment-creation call.
- Immutable schema-v4 authority C is
  `artifacts/preregistration_amendments/20260727T170413.080954Z`, a direct child of
  amendment P `20260727T133947.089370Z`. Exact identities are:
  - artifact root
    `57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627`;
  - SHA-256 manifest
    `4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156`;
  - amendment evidence
    `c2531787116e125bdb46e862f6803429c72e5d4766d5127f2cefa693e320912a`;
  - resource authorization
    `6e5c974be10d95e6f9f1dfbf1c09586473691bd4b6f8459a1d9c21e759bb12dc`;
  - immutable marker
    `49caed80e2e1c07b14a862767ffd5b674c941a7d5c42f7a950ceb902ecee2821`.
- All six registry, integrity, stage, and disposition files had byte-identical hashes
  before and after C publication. The final global inventory found exactly one
  resource authority C. No run, stage attestation, disposition, training, or
  scientific result was created by publication.
- A separate public `preregistration verify-amendment` exited 0 with
  `authority_status=verified_amendment`, `chain_depth=3`, matching root/manifest, and
  `integrity_verified=true`.
- A second fresh process successfully returned from
  `require_resource_bounded_confirmatory_authorization(C)` after live primary
  revalidation, then its display-only wrapper raised `KeyError:
  'authorization_sha256'` because the canonical authorization mapping intentionally
  has no self-hash field. This happened after validation and changed no artifact.
  The costly live read is not repeated merely to print a field already independently
  computed and verified by the one-shot controller.
- C permanently binds `outcomes_inspected=true`,
  `analysis_disposition=amended_or_exploratory`,
  `original_confirmatory_claim_allowed=false`, `completion_stage=null`, and
  `study_outcome_eligible=false`. Formal project status therefore remains
  `PRIMARY_STUDY_COMPLETE`; C does not satisfy `CONFIRMATORY_COMPLETE` and does not
  unlock M9.

## Next exact command (fresh lifecycle rehearsal)

```powershell
.venv\Scripts\python.exe -m histo_audit experiment lifecycle-rehearsal `
  --project-root . `
  --authority-dir artifacts\preregistration_amendments\20260727T170413.080954Z `
  --runs-root artifacts\runs
```

## 2026-07-27 — Lifecycle passed; public resource preflight failed cleanly

- Lifecycle rehearsal sealed
  `20260727T171808.578732Z_lifecycle_qualification_rehearsal_73c7ff3a7d`
  with artifact root
  `82b548133d29daba3122df008ac9b4920615fae69528af597864bc45e4a517d8`.
  The independent readiness run
  `lifecycle_ready_02fd28480073_185b5c8c0b6e27d6` also sealed and passed, with
  artifact root
  `5ba80c6c6166c9ce7895d698ca7d8e2e1bda3c8cd9c0bb6e808854061b380a67`.
- Exactly one public `resource-bounded-sensitivity --preflight-only` invocation ran.
  It stopped before RunTracker creation with
  `ValueError: CNN logical encoder/preprocessing provenance does not recompute from
  the verified crop view: cnn_context_rgb`. No resource/confirmatory registry row,
  run directory, failure run, marker, lock, cell, stage record, or disposition record
  was created.
- Read-only post-failure qualification found zero active resource Python processes
  and zero locks. Registry, integrity, and stage changes after authority publication
  are exactly the two sealed lifecycle runs and their one readiness record. Authority
  C, both lifecycle runs, and their append-only records remain valid and must be
  preserved.
- All 22 raw PanNuke files were streamed through SHA-256 again: 22/22 matched,
  39,359,162,655 bytes, with zero missing, added, or changed files. The three ZIP
  hashes remain `6e19ad...ed0b`, `5bc540...9b07`, and `c14d37...a39d`.
- The crop cache is not corrupt. Its NPZ SHA-256 is
  `07d484be3e9f7826030f5d54d17e9878f61b68c282c4a91305a30ecfa86f4a01`;
  sidecar SHA-256 is
  `738d3f4b3146ff6d62555d283dac84a05a063b98199a6d13175100feb5d5dd42`;
  content and sample-order bindings match. Only the logical CNN provenance differs:
  C combines current `cnn.py` source SHA-256
  `61a6c3c53965703c6dc4b4bf53b78a0ec773c28fbd4a0d713da79d69abe87766`
  with logical hashes produced from historical source SHA-256
  `a9a346f504e23b404e5a09ff9a9e2d99bed23a3f80f9329cbefef16f8fd47031`.
- Input construction also peaked at approximately 21.2 GiB because full images,
  masks, and features were copied into each role for each of three rotations.
  No real run will start until shared read-only/indexed storage passes a bounded-memory
  regression and an empirical preflight.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open and
  progress remains **8/10 = 80%**.

## Next exact command (after the isolated provenance and memory patches land)

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\test_pannuke_confirmatory_inputs.py `
  tests\test_resource_bounded_confirmatory_contract.py `
  tests\test_resource_bounded_authority_gate.py `
  tests\test_resource_bounded_runner.py `
  tests\test_resource_bounded_runner_execution.py
```

Do not rerun preflight or launch scientific execution before this focused gate, the
full mandatory QA sequence, and publication of an explicit immutable technical
successor to C.

## 2026-07-27 — Exact-12 workspace and final authority-D source gates passed

- The first frozen authority-D input directory,
  `artifacts/resource_control/authority_d_inputs_20260727Tfinal_source_v1`, is
  preserved unchanged as invalid, non-publishable evidence. Its workspace plan had
  only 5 source arrays instead of the required 12. The explicit invalidation receipt
  has SHA-256
  `29c854110d3e8191520e8800999823ad80f424b14ab516e4c86c12d2b00e6e2b`.
  No D marker, amendment, preflight, or resource run was created from v1.
- The shared workspace now binds exactly 9 crop-cache members and 3 native feature
  matrices, plus exactly 9 group-safe fold/role index vectors. The real PanNuke plan
  passed the public capacity-v3 validator with:
  - 188,333 source rows;
  - 12 arrays and 9 partition indices;
  - 4,294,182,269 extracted backing bytes;
  - 4,521,144 physical index-NPY bytes and 4,519,992 raw index bytes;
  - 4,298,703,413 projected payload bytes;
  - 4,315,480,629 planned bytes;
  - plan SHA-256
    `da4b0d6b4fd6f7e4b1f6504666df3170a0a78ff3b4b8746a53ab3c94fa7679dd`.
- Three fail-closed defects found before publication were fixed and regression
  tested: the capacity validator had compared raw index payload with physical NPY
  bytes (a 1,152-byte error); final source verification admitted a checksum-identical
  file/parent reparse swap; and fixed-width byte-string IDs were not normalized
  consistently with Unicode IDs. The final independent exact-12 audit reports zero
  remaining P0/P1/P2 findings.
- The combined final focused suite passed **123/123**. The first attempted full suite
  was deliberately stopped and preserved as non-qualifying when the reparse P1 was
  found; its explicit abort receipt is
  `artifacts/resource_control/full_pytest_exact12_20260727T201712.374Z.aborted.json`.
  After the fixes, the complete mandatory gates passed on the unchanged final
  execution source:
  - `.venv\Scripts\python.exe -m pytest -q` — **1135 passed in 532.24 s**;
    stdout SHA-256
    `53d04f94601212588d2a6ce730328b823dbf7a1241a73c260652c9023aa1981c`,
    empty stderr;
  - `.venv\Scripts\python.exe -m ruff check .` — passed;
  - `.venv\Scripts\python.exe -m ruff format --check .` — **167 files already
    formatted**;
  - `.venv\Scripts\python.exe -m mypy src` — **no issues in 87 source files**;
  - `.venv\Scripts\python.exe -m histo_audit experiment
    resource-bounded-sensitivity --help` — exited 0.
- The full real-PanNuke semantic validator completed naturally with empty stderr,
  `status=valid`, `validation_scope=full_semantic_scan`, and
  `publication=idempotent`. It covered 7,901 patches and again reported 4,318
  cross-class-overlap pixels, 10,486,091 void pixels, and 1,411
  overlap-touching instances excluded identically from primary and confirmatory
  analysis, with `no_class_arbitration=true` and `source_masks_modified=false`.
  Its stdout SHA-256 is
  `f7ff2cd5dafe2f6c33ccc0e5b557b89c43546bf6c91aba1c7413dcf5d814cd88`.
- Three independent captures agree on the final C-to-live execution source:
  102 artifacts, exactly 15 paths (**1 added, 14 modified, 0 removed**), root
  `2a568873f317cd9d5ef87cd991dbc5488ceb00c4fbc924af3828a531ae372477`,
  atomic-manifest SHA-256
  `5868954f97131398f487534cb7cbe9acfab5b3cb511293836e57afb04feb62c4`,
  delta SHA-256
  `9e30dfeba955f6b8e96b1a914a2a79bbb610e8a80fea87a2c3c0e791f778cc3d`,
  and change-kind SHA-256
  `e48f0b72011cc43a412ad014ad67b3b82088a7c3030336c1679d51f2bc950dcc`.
  `SPEC.md`, `PRE_REGISTRATION.md`, and both frozen configs remain byte-identical.
- The ignored one-shot D controller now pins that final tuple and passed **16/16**
  tests plus Ruff, format, and mypy. Its capacity-v3 validator runs before a freeze
  directory or publication marker can be created. All D attempt/success/failure
  markers remain absent, the amendment inventory still contains exactly three
  authorities, and no resource study has run.
- Free space after test-temp cleanup is 28,410,880,000 bytes, 221,421,003 bytes above
  the conservative 28,189,458,997-byte pre-workspace guard. The old unrelated
  `C:\pt3` pytest directory is approximately 2.33 GB and may be removed manually for
  safer headroom; it has not been modified automatically.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open,
  progress remains **8/10 = 80%**, and M9 remains locked.

## Next exact command (new input freeze; no marker and no publication)

```powershell
.venv\Scripts\python.exe artifacts\resource_control\prepare_resource_authority_d_once.py `
  --freeze-inputs artifacts\resource_control\authority_d_inputs_20260727Tfinal_source_v2
```

Preserve v1 unchanged. After v2 is independently verified as the exact 12-array /
9-index carrier, run the controller's read-only `--preflight-only` with the four v2
files and the exact frozen-source-receipt SHA-256. Do not invoke `--publish-once`
until that preflight passes.

## 2026-07-27 — Authority-D v2 input freeze and read-only preflight passed

- Exactly one new input-freeze invocation completed with
  `status=frozen_inputs_created_and_verified`,
  `publication_performed=false`, and `attempt_marker_created=false`. The preserved v2
  directory contains exactly four files:
  - frozen source receipt SHA-256
    `f5bd9384ac22be05b53e5b7fa987a059c84f74051067f47a7d30b14789e01c08`;
  - 15-path source allowlist SHA-256
    `d6436c55e3134807ee0eb99d7e3b5c0a0416b06c1ced22372e31ba2ce268f176`;
  - exact workspace-plan file SHA-256
    `d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b`;
  - CNN-correction receipt SHA-256
    `89c6d475d691b480478b76d25e5e96653a3d225d738ce23c6726a5bab409e6c3`.
  The freeze stdout SHA-256 is
  `0d7fd56295947915ed9b674103df232948e2eb5565f64d5eb8ac53227b9febdd`;
  stderr is empty.
- Independent v2 reconstruction passed with P0=0 and P1=0. The four files bind the
  final source tuple, exactly 12 arrays, exactly 9 index vectors, the public
  capacity-v3 canonical form, plan semantic SHA-256
  `da4b0d6b4fd6f7e4b1f6504666df3170a0a78ff3b4b8746a53ab3c94fa7679dd`,
  and an exact two-field logical-CNN correction. All six run-state hashes are
  unchanged, v1 remains invalid evidence, and no authority/run was created.
- Exactly one controller `--preflight-only` invocation using the four v2 paths and
  the explicit receipt pin completed with:
  `status=preflight_passed`, `publication_performed=false`,
  `attempt_marker_created=false`, `automatic_retry_allowed=false`,
  `source_allowlist_count=15`, `execution_source_delta_count=15`,
  `analysis_disposition=amended_or_exploratory`,
  `study_outcome_eligible=false`, and `completion_stage=null`.
  Its stdout SHA-256 is
  `30db4f0444e82212ac72c72b0f77e1b411c5313095e8a2f646e052d49d44775b`;
  stderr is empty.
- All three D publication markers remain absent and the amendment inventory still
  contains exactly the three pre-existing authorities ending at C. Free space at
  preflight completion was 28,704,415,744 bytes, 514,956,747 bytes above the
  conservative guard. The narrow margin remains a P2 operational concern for the
  later workspace build, not a publication mismatch.
- Formal status remains `PRIMARY_STUDY_COMPLETE`; no scientific execution occurred.

## Next exact command (single irreversible authority-D publication attempt)

Execute exactly once. Do not retry automatically if an attempt/failure marker or
ambiguous terminal state is observed:

```powershell
.venv\Scripts\python.exe artifacts\resource_control\prepare_resource_authority_d_once.py `
  --publish-once `
  --frozen-source-receipt artifacts\resource_control\authority_d_inputs_20260727Tfinal_source_v2\authority_d_frozen_source_receipt.json `
  --frozen-source-receipt-sha256 f5bd9384ac22be05b53e5b7fa987a059c84f74051067f47a7d30b14789e01c08 `
  --source-allowlist artifacts\resource_control\authority_d_inputs_20260727Tfinal_source_v2\authority_d_source_allowlist.json `
  --workspace-plan artifacts\resource_control\authority_d_inputs_20260727Tfinal_source_v2\authority_d_workspace_plan.json `
  --cnn-correction-receipt artifacts\resource_control\authority_d_inputs_20260727Tfinal_source_v2\authority_d_cnn_correction_receipt.json
```

## 2026-07-27 — Authority-D publication failed closed; automatic retry forbidden

- The authorized `--publish-once` command above was invoked exactly once. The
  controller completed naturally with `status=failed_no_retry`,
  `phase=create_schema_v5_authority`, `build_call_count=1`,
  `create_call_count=1`, `automatic_retry_allowed=false`, and
  `run_state_unchanged=true`.
- The durable attempt marker has SHA-256
  `8c93e65eca0bb4d64af4e94012004d74178448941cc746675d0e8e72ac5e90e2`;
  the failure marker has SHA-256
  `de123683a56ab0349c44536e969f536843ed0c557bae8573187664cab7fc8615`;
  no success marker exists. Publication stdout is
  `artifacts/resource_control/authority_d_v2_publish_20260727T212335.140Z.stdout.log`
  with SHA-256
  `c2ff6925d7d3339ebd2cd399074470f7308a804dc86099cd2d34e9f77e644c2d`;
  stderr is empty with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- The marker's error-type SHA-256 identifies `RuntimeError`. Independent source
  tracing reproduced its exact error SHA-256
  `1ddb48209c3cca372cc2053492cd41d8fec3fe5e04840250634e3d5f48d49f33`.
  The deterministic failure is a circular schema-v5 validation path: after the
  eight-file D bundle is physically published, D validation requests C's storage
  policy through an effective-leaf API; that API sees the just-published D and
  rejects C as superseded. The creator then reports failed independent verification
  and performs its ownership-safe rollback.
- This is a code-path P1, not a PanNuke, capacity, manifest, checksum, or transient
  filesystem failure. The rollback removed the intended directory
  `artifacts/preregistration_amendments/20260727T212711.019137Z`; no partial or
  temporary D remains. A public verifier invocation against that rolled-back path
  exited 1 because the directory no longer exists.
- The amendment inventory remains exactly the three pre-existing authorities ending
  at C. All six registry/ledger hashes equal both the before and after records in
  the failure marker; there are zero locks and no active publisher, resource,
  confirmatory, or primary process. No scientific cell or outcome was executed or
  read.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open, project
  progress remains **8/10 = 80%**, and M9 remains locked.

## Next exact action — STOP pending new explicit authority

Do not invoke `--publish-once` again, do not remove or replace either marker, and do
not start lifecycle, resource-bounded, confirmatory, or M9 execution. A separately
authorized correction must first split historical sealed-policy readback from the
effective execution-leaf gate, retain fail-closed rejection of superseded C for
execution, add end-to-end C-to-D regressions, pass all mandatory gates on a new
source receipt, and bind this failed-attempt evidence. No executable next command is
authorized until that decision is made explicitly.

## 2026-07-27 - Historical-C / effective-D correction qualified; publication remains blocked

- The schema-v5 circular validation defect is corrected in code without changing
  `SPEC.md`, `PRE_REGISTRATION.md`, either frozen config, Authority C, the terminal
  failed-attempt evidence, any scientific run, or raw PanNuke. Schema-v5 creation
  and chain verification now use a private sealed historical readback only for C's
  inherited storage policy. Public study, lifecycle, attestation, and execution
  gates still require the unique effective authority and continue to reject C after
  a valid D exists.
- The focused C-to-D regression set passed: **41 passed in 16.09 s**. It reproduces
  the original self-invalidating D path and ownership-safe rollback, verifies the
  corrected D path, proves that superseded C is rejected for execution, proves that
  a competing D fork fails closed, and directly exercises fail-closed detection
  when C's evidence changes during historical policy readback.
- Mandatory repository gates passed on the corrected source:
  - `.venv\Scripts\python.exe -m pytest -q`:
    **1,142 passed in 533.21 s**;
  - `.venv\Scripts\python.exe -m ruff check .`: passed;
  - `.venv\Scripts\python.exe -m ruff format --check .`:
    **167 files already formatted**;
  - `.venv\Scripts\python.exe -m mypy src`:
    **no issues in 87 source files**.
- The public read-only Authority-C chain verifier passed with `chain_depth=3`,
  `integrity_verified=true`, root
  `57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627`,
  and manifest
  `4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156`.
- The full real-PanNuke semantic validator completed in 308.3 s with
  `status=valid`, `validation_scope=full_semantic_scan`, 7,901 patches, 4,318
  cross-class-overlap pixels, 10,486,091 void pixels, and 1,411
  overlap-touching instances excluded identically from primary and confirmatory.
  It reported `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- Independent reviews found no P0 or P1 defect in the historical/effective split.
  A separate pre-publication review found three P1 conditions in the old
  controller design that must be fixed before any replacement publication:
  post-create checks currently occur outside the creator's ownership-safe rollback
  scope; stdout failure after a durable success marker could create contradictory
  success/failure evidence; and the final typed/effective verification is not yet
  repeated in a fresh Python process.
- The old v2 receipt, controller, marker namespace, and failed publication remain
  terminal and non-reusable. No replacement D, publication marker, lifecycle run,
  successor run, confirmatory result, or M9 result was created.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open, progress
  remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - make replacement publication atomic before new authorization

Add a transaction-scoped post-publication validator to the amendment creator so
result, inventory, run-state, source, intent, and fresh-process checks execute while
the creator still owns rollback tokens. Make the durable success marker the final
state-changing commit operation and classify every mixed/partial terminal state as
`STOP/ambiguous` with no automatic retry. Then add fault-injection regressions and
repeat the mandatory gates. Do not create a new receipt, controller marker,
Authority D, lifecycle run, or scientific run before that qualification is green
and a separate replacement-publication decision is recorded.

## 2026-07-28 - Atomic replacement-controller substrate qualified; live publication remains blocked

- The tracked replacement controller now has a new, non-reusable marker namespace
  and an exact fail-closed state machine. It binds one project, one direct C-to-D
  path, the executing controller, the terminal v2 failure evidence, a new immutable
  input bundle, the six pre-publication run-state files, authorization, intent,
  direct-child PID/PPID, verifier nonce, and generic/typed D identities.
- All fallible D verification remains inside the amendment creator's rollback
  scope. The outer protocol lock is cleanly released before the creator starts;
  the durable success/failure marker is therefore the final repository mutation.
  A creator must return the exact typed object passed to the one-use callback, and
  a final singleton-D readback occurs before success. Attempt-only, mixed, partial,
  cross-project, input-drift, and D-without-success states stop without retry.
- Fresh verification no longer uses unbounded `communicate()`. Two bounded reader
  threads retain at most 1 MiB stdout and 64 KiB stderr plus one sentinel byte,
  while every timeout, overflow, invalid PID, read error, wait error, or signal
  failure enters bounded terminate/kill/reap/close handling. Anchored file reads
  now accept an exact `max_bytes` bound before payload accumulation.
- Regression coverage includes the full 16-state truth table, exact JSON types,
  nonce and chain-depth tampering, case-variant namespaces, ancestor reparse and
  hard-link rejection, stdout/stderr overflow, pipe errors, timeout cleanup,
  cross-root configuration, every immutable input pin, authorization-to-input
  binding, post-callback creator rollback, deleted/mismatched transaction results,
  protocol-lock cleanup failure, recursive entry, and partial terminal markers.
  The controller/publication focused set passed **73/73**.
- The broader amendment/publication/authority/lifecycle integration set passed
  **173 tests with 1 expected Windows skip** in 159.60 s.
- Mandatory repository gates passed on the same source:
  - `.venv\Scripts\python.exe -B -m pytest -q`:
    **1,238 passed, 1 skipped in 609.26 s**;
  - `.venv\Scripts\python.exe -B -m ruff check .`: passed;
  - `.venv\Scripts\python.exe -B -m ruff format --check .`:
    **171 files already formatted**;
  - `.venv\Scripts\python.exe -B -m mypy src`:
    **no issues in 88 source files**.
- The read-only real-C command
  `.venv\Scripts\python.exe -B -m
  histo_audit.workflows.resource_authority_d_replacement_controller
  --preflight-only --project-root .
  --parent-authority-dir
  artifacts\preregistration_amendments\20260727T170413.080954Z`
  exited 0 with exact state `ready`, zero candidates, no marker hashes, and
  `publication_performed=false`.
- The full real-PanNuke command
  `.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke`
  exited 0 in 305.8 s with `status=valid`,
  `validation_scope=full_semantic_scan`, 7,901 patches, 4,318 cross-class-overlap
  pixels, 10,486,091 void pixels, and 1,411 overlap-touching instances excluded
  identically from primary and confirmatory. It reported
  `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- `SPEC.md`, `PRE_REGISTRATION.md`, both frozen configs, and the historical
  controller remain byte-identical at their prior exact hashes. The amendment
  inventory still contains only A/P/C; no prior-failure receipt, replacement
  attempt/success/failure marker, lock, Authority D, lifecycle run, scientific
  run, or M9 artifact was created.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open,
  progress remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - wire the qualified substrate before freezing new source

Add the outcome-blind live preflight/freeze/publication adapter to the tracked
replacement controller. It must reconstruct, not reuse, the old v2 inputs; create
no marker during preflight/freeze; bind the canonical terminal v2 failure receipt;
and invoke the amendment creator exactly once through the qualified callback.
Repeat focused and mandatory gates after this final source change. Only then may a
new prior-failure receipt and immutable replacement input bundle be O_EXCL-frozen
and independently read back. `--publish-once` remains deliberately unwired and
must not be invoked yet.

## 2026-07-28 - Live replacement adapter qualified; freeze blocked by capacity

- The tracked replacement controller now exposes the complete bounded live sequence:
  exact singleton freeze, read-only preflight, one-attempt authorization, and one
  atomic publication call. It reconstructs current source/workspace/CNN evidence and
  does not reuse the old v2 input bundle.
- Final fail-closed hardening requires the one canonical directory
  `artifacts/resource_control/authority_d_replacement_inputs_v1`; prefixed aliases
  and case variants stop as ambiguous. Freeze validates that path before expensive
  derivation or any write, derives one foundation, runs two public resource gates,
  and rechecks source, Authority C/config/manifest, run state, controller, and the
  prior-failure receipt before the gate, after the gate before the first bundle
  write, and after exact readback. Post-write drift rolls back only owned bundle
  paths.
- CLI exception reporting now distinguishes no-write, retained control-write,
  committed, and ambiguous states. `publication_performed` is `true` only for exact
  `A+S+D`, `false` for READY or exact `A+F/no-D`, and `null` for ambiguous state.
  Direct tests cover A-only, A+D, committed, retained-receipt, singleton/case
  aliases, resource-gate failure, post-write input drift, authorization rollback,
  exactly two authorization/publication preflights, and exactly one amendment
  creator call.
- Focused controller regression completed with **96 passed**. The broader
  amendment/authority/lifecycle integration selection completed with **190 passed**
  before the final two hardening edits; the exact final source is qualified by the
  complete suite below.
- The first direct full-pytest invocation was terminated after five seconds by the
  command harness timeout; a process audit proved that no pytest process survived.
  One monitored replacement invocation on the final source then completed:
  `.venv\Scripts\python.exe -B -m pytest -q` — **1,274 passed, 1 expected Windows
  skip in 623.40 s**. Its stdout is
  `artifacts/resource_control/option_b_full_pytest_20260728T022645.950Z.stdout.log`
  with SHA-256
  `af35f46761158fc8fd461968b1b134ee985acd82cf760bf8d12b56d913f65386`;
  stderr is empty with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- The remaining mandatory code gates passed on that source:
  `.venv\Scripts\python.exe -B -m ruff check .`;
  `.venv\Scripts\python.exe -B -m ruff format --check .` with **171 files already
  formatted**; and `.venv\Scripts\python.exe -B -m mypy src` with **no issues in
  88 source files**.
- The full real-PanNuke validator passed with `status=valid`,
  `validation_scope=full_semantic_scan`, 7,901 patches, 4,318 cross-class-overlap
  pixels, 10,486,091 void pixels, and 1,411 overlap-touching instances excluded
  identically from primary and confirmatory. It retained
  `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`. Validator stdout SHA-256 is
  `f7ff2cd5dafe2f6c33ccc0e5b557b89c43546bf6c91aba1c7413dcf5d814cd88`;
  stderr is empty.
- The functional controller `--classify` command returned exact `READY`, zero
  candidates, null marker hashes, `publication_performed=false`, and no automatic
  retry. `git diff --check` passed.
- `SPEC.md`, `PRE_REGISTRATION.md`, both frozen configs, and the historical
  controller retain their exact protected hashes. The amendment inventory remains
  exactly A/P/C. No prior-failure receipt, replacement bundle, publication
  authorization, A/S/F marker, lock, Authority D, lifecycle run, scientific run, or
  M9 artifact was created.
- The exact capacity-v3 threshold remains **28,189,458,997 bytes**. The recorded
  post-gate observation found **25,711,448,064 bytes** free, a deficit of
  **2,478,010,933 bytes**. The completed full suite created the owned temporary
  directory
  `C:\Users\NATAN\AppData\Local\Temp\pytest-of-NATAN\pytest-1667`
  (47,573 files; 2,474,254,135 logical bytes). A guarded cleanup attempt was blocked
  by the execution policy and deleted nothing. The protected run, `C:\pt3`, raw
  PanNuke, and all project evidence remain unchanged.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open, progress
  remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - external free-space blocker before the singleton freeze

Free at least 3 GiB, preferably 5 GiB, outside protected AANCA data and runs. The
owned `pytest-1667` directory above may be removed manually after confirming no
pytest process is active. Then remeasure free space and require at least
28,189,458,997 bytes before executing exactly one canonical freeze:

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --freeze-inputs artifacts\resource_control\authority_d_replacement_inputs_v1 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

Do not run preflight, authorization, publication, lifecycle, resource science, or
M9 before the singleton freeze completes and is independently read back.

## 2026-07-28 - Capacity cleared; v1 frozen; first preflight failed closed

- The external capacity blocker cleared. The canonical freeze measured
  **169,348,308,992 bytes free (~157.7 GiB)** against the exact
  **28,189,458,997-byte** requirement, and both resource gates passed.
- The singleton freeze to
  `artifacts/resource_control/authority_d_replacement_inputs_v1` completed with
  `status=replacement_inputs_frozen_and_verified`. It created the immutable
  prior-publication failure receipt and four-file v1 bundle; it performed no
  publication or scientific execution. The freeze stdout is
  `artifacts/resource_control/option_b_replacement_freeze_20260728T160420.324Z.stdout.log`
  with SHA-256
  `1cb46124ed48800ab5614886b5ecf786e31835ccedf32f30ec62d82b9a9c5a3c`;
  stderr is empty.
- The first read-only live preflight then failed closed with error SHA-256
  `c103cd60f4aba8d6673884180ceb1c168f1f8b1506ae75cabe3311a956fb02e3`:
  `failed-preflight observation is not canonical to microseconds`. The immutable
  historical observation is `2026-07-27T17:30:54.689Z`; its millisecond precision
  is valid preserved evidence but is rejected by the replacement controller's
  stricter microsecond-only parser. The failed preflight stdout is
  `artifacts/resource_control/option_b_replacement_preflight1_20260728T161040.612Z.stdout.log`
  with SHA-256
  `c51dd63b0534b1d2d245709752b64a035f51ab2418385ae4061928e01fac7b1d`.
- A bounded read-only traceback reproduced the exact cause. Its stdout and stderr
  SHA-256 values are respectively
  `1b8916f31e87165480342e8ba432cff57d95eed8e4f8d3f1c2bf41733bd6e4aa`
  and
  `163b208d12a2828af832f4e25ea629a6b8e87a8fa2c578437bd83c45ab6cbf6a`.
  The timestamp order itself remains valid:
  `Authority C < failed preflight < prior-failure receipt < proposed D`.
- The failure occurred before any publication authorization, replacement
  attempt/success/failure marker, Authority D, lifecycle run, `RunTracker`,
  scientific cell, or M9 artifact. The controller reported
  `status=stopped_without_write`, `publication_performed=false`, and
  `automatic_retry_allowed=false`.
- Preserve v1 byte-for-byte and record an explicit invalidation; do not overwrite,
  reuse, or publish from it. Correcting the tracked compatibility defect changes
  the frozen execution source, so a separately named immutable v2 input freeze and
  fresh read-only preflights are required before any publication authorization.
- `SPEC.md`, `PRE_REGISTRATION.md`, both frozen configs, Authority C, the recovered
  primary run, raw PanNuke, and the six run-state files remain unchanged. The
  amendment inventory remains exactly A/P/C.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - preserve v1 and qualify a corrected v2 lineage

Add a separate strict parser only for the SHA-pinned historical failed-preflight
timestamp; keep microsecond canonicalization unchanged for every newly produced
control or authority timestamp. Add an O_EXCL, exact-readback invalidation receipt
that preserves v1 as non-publishable evidence, and make
`authority_d_replacement_inputs_v2` the sole active singleton. Run focused and
mandatory gates before publishing that receipt or freezing v2. Do not authorize or
publish from v1.

## 2026-07-28 - Corrected v2 lineage qualified; v1 invalidation is next

- The replacement controller now accepts the exact SHA-bound historical
  `2026-07-27T17:30:54.689Z` observation through a dedicated parser while retaining
  microsecond-only canonicalization for all new evidence. The immutable v1 bundle
  remains unchanged and non-publishable.
- The v1 invalidation, v2 freeze, publication authorization, and attempt claim now
  share the canonical active-v2 exclusion target. Every rollback-capable publisher
  verifies lock ownership before deletion and again after rollback. The v2 freeze
  also verifies ownership immediately before and after creating its directory and
  each of its four files. Loss of ownership or lock-cleanup uncertainty preserves
  the remaining state and stops as ambiguous.
- Focused controller tests passed with **123 passed**. The broader
  publication/authority/lifecycle selection passed with **252 passed, 1 expected
  Windows skip**. The complete mandatory suite passed with **1,301 passed, 1
  expected Windows skip in 638.92 s**.
- Static gates passed on the same source snapshot:
  `.venv\Scripts\python.exe -B -m ruff check .`,
  `.venv\Scripts\python.exe -B -m ruff format --check .` with **171 files already
  formatted**, and `.venv\Scripts\python.exe -B -m mypy src` with **no issues in
  88 source files**. The corrected controller is 216,288 bytes with SHA-256
  `cbea3c3536dbad729383c96e0ef602042c7e3c4e000f9b0cb79e50c13b2ced58`;
  its focused test file is 122,087 bytes with SHA-256
  `479dd4f2354dcfce13b3ae732ef9ef077b6c106f580211e34543d3393897b4a2`.
- The full real-PanNuke validator passed with `status=valid`,
  `validation_scope=full_semantic_scan`, 7,901 patches, 4,318 cross-class-overlap
  pixels, 10,486,091 void pixels, and 1,411 overlap-touching instances excluded
  identically from primary and confirmatory. It reported
  `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- Read-only `--help` passed. The exact pre-invalidation `--classify` stopped
  fail-closed with `state=stop_ambiguous` because v1 and its invalidation receipt
  must exist as a pair. Recursive before/after snapshots proved that this check
  wrote nothing.
- All protected hashes remain exact. Authority C retains its exact 8/8 inventory
  and verified chain; amendment inventory remains exactly A/P/C. There is no
  invalidation receipt, v2 bundle, publication authorization, replacement A/S/F
  marker, Authority D, relevant process, or active replacement bundle lock.
- Free space is **169,420,222,464 bytes (157.785 GiB)** against the exact
  **28,189,458,997-byte** requirement, leaving a **131.531 GiB** margin.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - preserve v1 with one canonical invalidation receipt

Execute exactly one state-changing command:

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --invalidate-v1 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

Independently read back the receipt and confirm that v1, the prior-failure
receipt, protected files, run-state root, and A/P/C inventory remain unchanged and
that v2, authorization, A/S/F, and D remain absent. Only then may read-only
classification become `READY` and the canonical v2 freeze run once. Do not
authorize publication or execute science before v2 and two stable preflights.

## 2026-07-28 - Retired v1 invalidation published and independently read back

- Exactly one `--invalidate-v1` invocation completed with
  `status=retired_v1_preserved_invalid_nonpublishable`. It created only
  `artifacts/resource_control/authority_d_replacement_inputs_v1.invalidation.json`,
  6,550 bytes, SHA-256
  `0b9af7cdb9ca3fcb60c8dd6c123eda22f13631c1188ff390cd9421998e28e997`.
  It reported `publication_performed=false`,
  `scientific_execution_performed=false`, and
  `outcome_value_interpretation_performed=false`.
- Independent readback retained the exact v1 file identities:
  CNN correction
  `0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e`
  / 4,452 bytes, frozen source
  `18b573b1b18f2a0fdcefe5f06862a4c900efdf17fcf259567827a6846acf99ea`
  / 3,663 bytes, source allowlist
  `8b90ce20910afef617fc4029c72f2df6cca561e0487844b43d92f3aa94338a70`
  / 3,903 bytes, and workspace plan
  `d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b`
  / 12,186 bytes. The prior-failure receipt remains
  `2b46f11d1580a6469715a525c0738d39fb3ae0f74f542e142ecd293ae7beed00`
  / 11,413 bytes.
- The receipt binds corrected controller
  `cbea3c3536dbad729383c96e0ef602042c7e3c4e000f9b0cb79e50c13b2ced58`,
  retired bundle root
  `f8b6eeeaa6e4f2f4e70aef08a2969fdd2ca5ebdaa5ab1dc5a35cd22c1e6103fc`,
  and unchanged run-state root
  `5692af0ac890f2f138d5b531fd4acbeab6843905fb41154750dbac0167a714a4`.
- Post-write `--classify` returned exact `READY`, zero candidates, null A/S/F
  hashes, `publication_performed=false`, and no automatic retry. The v2 bundle,
  publication authorization, replacement A/S/F, and Authority D remain absent.
- All protected hashes remain exact and the amendment inventory remains A/P/C.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze the canonical active v2 bundle once

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --freeze-inputs artifacts\resource_control\authority_d_replacement_inputs_v2 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

After exact independent readback, execute two read-only `--preflight-only`
commands and require stable immutable input bindings. Each fresh proposal has its
own timestamp, destination, intent, fingerprint, and resource observation. Do not
create a publication authorization until both preflights pass.

## 2026-07-28 - Canonical active v2 bundle frozen and locally read back

- Exactly one canonical v2 freeze completed with
  `status=replacement_inputs_frozen_and_verified`. Capacity passed with
  169,144,799,232 free bytes against the 28,189,458,997-byte minimum; compute
  evidence SHA-256 is
  `e3ac3eba3ea3b3cabb40acce3b846959748869a726afc0f35a5cfb7dbec74715`.
- The active singleton contains exactly four regular files:
  CNN correction, 4,452 bytes,
  `0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e`;
  frozen source, 3,943 bytes,
  `1acbcfd44b3f95d6387d7da573786547a6c1ff5dcd0d05b4198d311fbe813605`;
  source allowlist, 3,903 bytes,
  `397fac0240f36fb598095e7605dae770b55faf114d4c692e555a7101fd47c369`;
  and workspace plan, 12,186 bytes,
  `d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b`.
- The bundle binds invalidation receipt
  `0b9af7cdb9ca3fcb60c8dd6c123eda22f13631c1188ff390cd9421998e28e997`,
  prior-failure receipt
  `2b46f11d1580a6469715a525c0738d39fb3ae0f74f542e142ecd293ae7beed00`,
  execution-source root
  `c81e7a01bc6949d82d5cb76a206776dde4ceda47c1506a71bd8edf736649bd75`,
  source-delta root
  `82acb5a60100141a2c54f2094b8a438fd725bcfe4227c132e4dff185608d7217`,
  and technical authorization SHA-256
  `886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8`.
- The freeze reported `publication_performed=false`,
  `scientific_execution_performed=false`, and
  `outcome_value_interpretation_performed=false`. Post-freeze classification
  remains exact `READY`; authorization, A/S/F, Authority D, and scientific runs
  remain absent.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - two independent read-only v2 preflights

Run the following command twice as separate processes and require both to pass:

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --preflight-only `
  --frozen-input-dir `
  artifacts\resource_control\authority_d_replacement_inputs_v2 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

Compare immutable authorization, source, bundle, run-state, invalidation, prior,
and Authority-C bindings. Validate each proposal-specific timestamp, destination,
intent, fingerprint, and capacity/compute observation independently; do not
require those dynamic values to be equal. Do not create the one-attempt
publication authorization unless the independent bundle audit and both
preflights are green.

## 2026-07-28 - Independent v2 audit and two live preflights passed

- The independent v2 audit returned GO: exact four-file canonical reconstruction,
  anchored readback, v1/invalidation/prior/controller/run-state/source bindings,
  protected hashes, and A/P/C were all exact. Authorization, A/S/F, D, scoped
  locks, and relevant processes were absent. Read-only classification was `READY`.
- Preflight 1 passed with immutable authorization SHA-256
  `886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8`,
  proposed timestamp `2026-07-28T18:10:25.426598Z`, intent
  `40bc47db60a3cbd3095e39398de509222d918c679f8a978c115c7ed810f8723c`,
  fingerprint
  `44897d43dee70fb0b8eb56ca6653f950466d596103320ceeea66da29f0af9c52`,
  compute observation
  `2c66f0405d155f106c5bd719aa23cb600ea97998f858d3e3c492832f09b8f3da`,
  and 169,116,037,120 free bytes.
- Preflight 2 passed in a separate process with the same immutable authorization,
  proposed timestamp `2026-07-28T18:14:32.860173Z`, intent
  `fdd1894b07b29c5af694e7767a6510b6841e032bff309f50ddda9756f3d3a1fc`,
  fingerprint
  `2814ef49a7b09511f2e1d1ecdbb04734f5dd6010edcd4045b0af4f02000942d7`,
  compute observation
  `cb8980aa9976d5d689db519279a08f5e74e4aacc8fb2870e0b32f9d0fc52b4e2`,
  and 169,113,915,392 free bytes.
- Both returned `status=passed`, `automatic_retry_allowed=false`,
  `publication_performed=false`, `scientific_execution_performed=false`, and
  `outcome_value_interpretation_performed=false`. Their proposal-specific
  timestamp, destination, intent, fingerprint, and resource observation differ as
  required; immutable v2 and authorization bindings agree.
- A post-preflight readback proved v2 unchanged, authorization/A/S/F absent,
  amendment inventory still A/P/C, and classifier still exact `READY`.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - create one one-attempt publication authorization

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --authorize-publication `
  --frozen-input-dir `
  artifacts\resource_control\authority_d_replacement_inputs_v2 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

Read back the canonical authorization receipt and require its exact one-attempt
bindings, unchanged v2/protected/run-state/A-P-C evidence, no A/S/F or D, and
`READY`. Only then may the separately guarded single `--publish-once` command run.

## 2026-07-28 - One-attempt authorization receipt created; publication pending readback

- Exactly one `--authorize-publication` invocation completed with
  `status=publication_authorized_for_one_attempt`. The canonical receipt is 9,396
  bytes with SHA-256
  `4c892f7e518964a46569290e1a486d7f7e193121ed870522895946413dbee565`.
- It binds authorized attempt ID
  `c2cfdbdf80d19804de4542e18313fb7eebf4b2afd81272b1042cbfb63c8eaa86`,
  `max_attempt_count=1`, `automatic_retry_allowed=false`, amendment timestamp
  `2026-07-28T18:19:20.303224Z`, intended Authority D
  `artifacts/preregistration_amendments/20260728T181920.303224Z`, intent
  `9c5018d37a4a9f4d26dd40d1b4c3eb902c97601459720085ae12bc91d4e4e347`,
  and preflight fingerprint
  `9e828dd7652a2be3c3ecee798fae9f7b1b1167875129e7c3fed581443550270a`.
- Local readback confirmed the closed fixed policy, exact receipt hash,
  `publication_performed=false`, `scientific_execution_performed=false`, absent
  A/S/F, and absent intended D. Classification remains exact `READY`.
- A separate independent receipt audit is required before consuming the one-shot
  authorization. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains open and M9 remains locked.

## Next exact action - independent authorization readback, then one publication

Do not invoke `--publish-once` until the independent audit confirms the receipt,
all live bindings, unchanged v2/protected/run-state/A-P-C state, absent A/S/F and
D, and no conflicting lock or process.

## 2026-07-28 - Independent authorization audit GO; one publication is eligible

- The independent audit verified the canonical closed 9,396-byte receipt at exact
  SHA-256
  `4c892f7e518964a46569290e1a486d7f7e193121ed870522895946413dbee565`,
  including exact attempt ID, timestamp, destination, schema-v5 purpose/depth,
  `max_attempt_count=1`, and all false execution/publication/outcome flags.
- A fresh outcome-value-blind preflight at the authorized timestamp reproduced the
  stored contract byte-for-byte, fingerprint
  `9e828dd7652a2be3c3ecee798fae9f7b1b1167875129e7c3fed581443550270a`,
  intent
  `9c5018d37a4a9f4d26dd40d1b4c3eb902c97601459720085ae12bc91d4e4e347`,
  technical authorization
  `886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8`,
  and intended destination. Fresh free space was 169,077,133,312 bytes.
- v2, invalidation, prior, failed evidence, corrected controller, run-state,
  source roots, protected files, and Authority C remained exact. Amendment
  inventory remained A/P/C. Replacement A/S/F, D, all 19 scoped locks, and
  relevant processes were absent; fresh classification was exact `READY`.
- The sole separately guarded `--publish-once` invocation is now eligible. Any
  failure, mixed state, or ambiguity consumes the attempt and stops without
  automatic retry.

## Next exact action - consume the authorization once

```powershell
.venv\Scripts\python.exe -B -m `
  histo_audit.workflows.resource_authority_d_replacement_controller `
  --publish-once `
  --frozen-input-dir `
  artifacts\resource_control\authority_d_replacement_inputs_v2 `
  --project-root . `
  --parent-authority-dir `
  artifacts\preregistration_amendments\20260727T170413.080954Z
```

Do not repeat this command under any disposition. After it exits, classify and
independently verify exact A+S+D commit or exact A+F/no-D rollback; every other
state is STOP/ambiguous.

## 2026-07-28 - Sole publication attempt rolled back; no retry is authorized

- The exactly once `--publish-once` invocation terminated with
  `state=rolled_back_failure`. The intended Authority D was removed by the
  amendment creator's ownership-safe rollback and no success marker was written.
- Exact attempt marker: 3,420 bytes, SHA-256
  `e602993753949ecbd5bfe3dfd9ba77d1890d63ae6232db9db6d66caff48e3ace`.
  It binds the authorized attempt ID, v2, invalidation, prior, controller,
  run-state, authorization receipt, timestamp, destination, intent, and
  fingerprint exactly.
- Exact terminal failure marker: 924 bytes, SHA-256
  `e66305dac9a2c1b59d5cb554081470c1947b939d8a07ade3cf77046f0e353b12`,
  `status=rolled_back_failure_no_retry`,
  `authority_absent_after_rollback=true`, and
  `automatic_retry_allowed=false`.
- The failure type SHA-256
  `431fd4d500d504c9f02a7e5f505eb2065cf1612fc36b6474af72b89e5d3a8ffd`
  identifies `FreshVerifierError`. The full error SHA-256
  `a6a9e199b4080a911e7f07f3243e3982049a85fc18ba712883de9f9cc099e1fb`
  exactly identifies
  `FreshVerifierError: fresh verifier process did not exit cleanly`.
  This proves the amendment reached its transaction-scoped fresh-verifier step,
  but the child returned nonzero or emitted stderr. The current failure schema
  does not preserve the bounded child stderr/return code, so the child-level cause
  is not yet available from durable evidence.
- Fresh classification is exact `ROLLED_BACK_FAILURE`: A+F exist, S and D are
  absent, candidate list is empty, `publication_performed=false`, and automatic
  retry is disabled.
- Do not rerun, delete, overwrite, or repair the attempt, failure, authorization,
  v2, invalidation, or historical evidence. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open and M9 remains locked.

## Next exact action - bounded read-only failure qualification

Independently verify exact A+F/no-D, absence of locks/processes, unchanged
v2/authorization/lineage/protected/run-state/A-P-C evidence, and the precise code
path for the hashed `FreshVerifierError`. No replacement publication, lifecycle,
resource run, confirmatory run, or M9 action is authorized until a new explicit
technical/governance decision is recorded from durable evidence.

## 2026-07-28 - Terminal rollback qualified; Windows direct-child defect corrected

- Independent and local read-only qualification confirmed the exact terminal
  truth table: replacement-v1 attempt A exists at SHA-256
  `e602993753949ecbd5bfe3dfd9ba77d1890d63ae6232db9db6d66caff48e3ace`,
  failure F exists at SHA-256
  `e66305dac9a2c1b59d5cb554081470c1947b939d8a07ade3cf77046f0e353b12`,
  success S is absent, and intended D
  `20260728T181920.303224Z` is absent. The consumed authorization-v1 and
  input-v2 remain present; there are zero matching lock files and zero other
  replacement/resource/primary/confirmatory processes.
- The exact fail-closed code path was reproduced. `VerifyRequest` correctly kept
  the venv `sys.executable` as `argv[0]`, but Windows started that redirector as
  the `Popen` child and the real Python verifier as its child. The mandatory
  verifier condition `os.getppid() == controller_pid` therefore failed before
  successor verification, the verifier exited nonzero, and the controller
  recorded the hashed generic `FreshVerifierError`. The creator then completed
  ownership-safe rollback before F was written.
- Two process probes distinguished the mechanisms. With the venv launcher,
  `Popen.pid` identified the shim and the executing Python reported that shim as
  parent. With `executable=sys._base_executable` while retaining the venv path as
  `argv[0]`, `Popen.pid` equalled the executing Python PID, its PPID equalled the
  controller PID, and `sys.prefix`, `sys.executable`, isolated mode, and installed
  venv packages remained correct.
- The controller now applies that override only on Windows and only after
  fail-closed validation of an absolute, real, regular, non-reparse base-prefix
  interpreter. The direct-parent PID/PPID and nonce checks are unchanged. Invalid,
  missing, relative, directory, wrong-parent, or noncanonical candidates are
  rejected before `Popen`.
- Executed gates on the final corrected snapshot:
  - focused spawn/process regressions: **27 passed, 104 deselected**;
  - full controller file: **131 passed**;
  - broader amendment/authority/lifecycle/runner selection: **260 passed**;
  - complete suite: **1,309 passed, 1 expected Windows skip in 634.05 s**;
  - `ruff check .`: passed;
  - `ruff format --check .`: **171 files already formatted**;
  - `mypy src`: no issues in **88 source files**.
- The corrected controller is 218,766 bytes, SHA-256
  `e20278105b6ea4e2786713c64d9e8cf7bb06d9e4c8155f35a46861e72cb67b5f`.
  Its test file is 126,652 bytes, SHA-256
  `3202c60bb75f1cf44e053d40a67bf57db2dc24669de82b10f3bfcad0c5ca3be6`.
  Free space after the gates was 168,912,392,192 bytes (157.312 GiB), a
  140,722,933,195-byte (131.058 GiB) margin over the capacity-v3 threshold.
- This correction does not authorize or replay replacement-v1. Authorization-v1,
  input-v2, A1/F1, its timestamp, destination, intent, and attempt ID remain
  permanently consumed historical evidence. Because the old verifier stopped at
  the process-boundary check, later latent verifier failures cannot yet be
  excluded.
- `SPEC.md`, `PRE_REGISTRATION.md`, both frozen configs, Authority C, the
  recovered primary, raw PanNuke, and run-state remain unchanged. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%** and
  M9 remains locked.

## Next exact action - implement a separate replacement-v2 protocol

Implement and test, without publishing artifacts, an independently versioned
protocol that authenticates replacement-v1 as historical A1+F1/no-S1/no-D,
creates a future O_EXCL terminal-qualification receipt, uses a newly reconstructed
`authority_d_replacement_inputs_v3`, authorization-v2, and publication-v2 A/S/F
namespace, and binds the terminal failure lineage into a typed schema-v3
technical-successor authorization. Preserve the old controller hash through
historical pins rather than comparing it with the corrected live source. Do not
write the qualification receipt, freeze v3, authorize, publish, run lifecycle, or
start science until this new implementation passes focused, broader, full, Ruff,
format, mypy, protected-integrity, and read-only CLI gates.

## 2026-07-28 - Replacement-v2 schema and terminal-Q test subgates passed

- Implemented the typed schema-v3 technical-successor authorization while preserving
  schema-v2 serialization and intent compatibility. An independent reconstruction of
  the consumed schema-v2 lineage reproduced technical authorization SHA-256
  `886b8d1264028c8863ab2698f0cf10a4f85e25704c88bac3a7d595607ced75b8`
  and intent SHA-256
  `9c5018d37a4a9f4d26dd40d1b4c3eb902c97601459720085ae12bc91d4e4e347`
  exactly.
- The schema-v3 layer now binds the closed terminal qualification, exact five-source
  historical controller attestation, Authority C, protected files, the six-file
  run-state snapshot, and the ordered 28-record read set. Pre-commit verification
  rechecks the mutable run state live; sealed post-D verification retains the exact Q
  snapshot without rejecting later permitted append-only lifecycle records.
- A public-Q project-root derivation defect found by the full lifecycle fixture was
  corrected from the `artifacts` directory to the actual project root. The focused
  schema-v3 suite passed **23 passed, 34 deselected** after that correction. Before
  the correction, the complete amendment test file had passed **57 passed**; the
  final complete repository gate remains pending until controller integration stops
  changing.
- Added an independent terminal-Q security suite. Final executed results:
  - `pytest -q tests\test_resource_authority_d_replacement_v2_terminal_security.py`
    — **22 passed in 4.02 s**;
  - `ruff check` on that file — passed;
  - `ruff format --check` on that file — one file already formatted;
  - the independent source-delta contract test — **1 passed in 3.52 s**.
- The Q implementation uses a sealed public API, exact reserved-family enumeration,
  three process probes, a shared Authority-C parent guard, the historical
  `7+6+6+3`/16-lock topology, O_EXCL publication, and ownership-safe rollback.
  Input-v3 and authorization-v2 implementation remains in progress and has not yet
  passed its focused or full gates.
- No Q, input-v3, authorization-v2, A2/S2/F2, Authority D, lifecycle, or scientific
  artifact was created. `SPEC.md`, `PRE_REGISTRATION.md`, and both frozen configs
  retain their exact pinned sizes and SHA-256 values. Free space was independently
  confirmed at **157.2 GiB**.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - complete and qualify the replacement-v2 controller

Finish the input-v3, authorization-v2, exhaustive classifier, publication-v2,
bounded fresh-verifier diagnostic, and production CLI paths. Run their focused
tests, then the broader and complete mandatory gates. Do not create Q or any later
protocol/scientific artifact until those read-only gates and an independent live
qualification pass.

## 2026-07-28 - Replacement-v2 authorization and fresh-process security subgates passed

- Re-ran the complete schema-v3 amendment and lifecycle selection after the public-Q
  project-root correction:
  - `pytest -q tests\test_preregistration_amendment.py
    tests\test_resource_authority_d_schema_v3_lifecycle.py` — **59 passed in
    129.33 s**;
  - targeted Ruff check and format check — passed.
- Expanded the publication-authorization-v2 regression suite from its initial
  22 cases to exact observation, chronology, late-live-readback, residual-state,
  rollback, and sealed-carrier coverage:
  - `pytest -q
    tests\test_resource_authority_d_replacement_v2_preflight_authorization.py`
    — **85 passed in 4.11 s**;
  - the independently run targeted result was also **85 passed**;
  - Ruff check and format check for the test file — passed.
- Added a separate bounded fresh-verifier security suite. It covers the Windows
  venv/base-executable direct-child path, a real verifier child that creates a
  sleeping grandchild, bounded whole-tree cleanup, exact diagnostic-state
  consistency, payload/PID/hash bindings, and rejection of an invalid successor
  purpose:
  - `pytest -q
    tests\test_resource_authority_d_replacement_v2_fresh_security.py` —
    **35 passed in 3.55 s**;
  - the real Windows descendant cleanup case passed three repeated runs;
  - Ruff check and format check for the test file — passed.
- Live disk capacity was 171,591,901,184 bytes (159.81 GiB) against the exact
  28,189,458,997-byte (26.25 GiB) successor threshold: 143,402,442,187 bytes
  (133.55 GiB) headroom, or 6.09 times the requirement.
- The protected files remain byte-identical at their pinned SHA-256 values.
  No Q, input-v3, authorization-v2, A2/S2/F2, Authority D, lifecycle, resource,
  confirmatory, or M9 artifact was created; no protocol lock or matching Python
  process remained after the tests.
- The replacement-v2 controller remains under active integration. The concrete
  schema-v3 adapter, A2/S2/F2 transaction, exhaustive classifier, execute-once
  operation, and CLI are not yet complete. Full sealed U-to-I3/Q/source/config/
  history/run-state/workspace cross-links are the remaining authorization review
  item. Focused successes above are subgates, not permission to publish.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish the replacement-v2 transaction tail

Complete the sealed authorization cross-links, concrete schema-v3 adapter,
fresh-verifier transaction binding, A2/S2/F2 markers, exhaustive 64-state
classifier, execute-once path, and read-only CLI. Then run focused, broader, full
pytest, Ruff check, Ruff format check, mypy, protected-integrity, artifact-absence,
PanNuke validator, and functional CLI gates. Do not publish Q or any later
protocol/scientific artifact before those gates pass.

## 2026-07-29 - External event-driven unattended supervisor qualified and installed

- Built the Windows supervisor entirely outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-supervisor`. It is not project execution
  source or scientific authority and did not modify `SPEC.md`,
  `PRE_REGISTRATION.md`, either frozen config, raw PanNuke, or any existing run.
- The frozen source/test/helper identities are:
  - supervisor: 196,450 bytes, SHA-256
    `09a0ffe11d52a997d9c0b02fe98d5f82433be13bd4d4150795379a0995b2a20a`;
  - tests: 87,540 bytes, SHA-256
    `c21f2386fd48b3dfd8d2561bea318dda52cfcc07a3ed3aec39d46cd32fb2cb90`;
  - synthetic helper:
    `cf96558b8706212e30d64f6e4e125ad4a0667da82b118711f99f14e8c6d6aefd`;
  - hard-crash driver:
    `be56890a70afda5707354137a247ebba682a10559fb52414f4ae6acccf6555fb`;
  - provisional-recovery driver:
    `c7187a8868018610dedd06e8439d837427ae0ac8df1f1786fe76cb83b23b01be`.
- Executed the complete synthetic suite on the byte-identical tested source:
  **63/63 passed in 61.298 s**. It covers success, nonzero exit, missing seal,
  restart, PID reuse, double launch and cross-root singleton, corrupted state,
  bounded-log overflow, failed/nonzero Codex wake, exact session and no `--last`,
  power assertion cleanup, hard supervisor crash and descendant cleanup, main,
  verifier and wake timeouts, suspended-process assignment gaps, `ResumeThread`
  ambiguity, reparse/durable-move faults, terminal/spec/wake/provisional
  post-wake tampering, timestamp/identity forgeries, and recovery no-retry
  behavior. Independent rerun also passed 63/63; targeted wake/verifier coverage
  passed 33 tests. Component Ruff check and format check passed.
- Installed a read-only, byte-identical operational release at
  `C:\Users\NATAN\AppData\Local\AANCA-supervisor\releases\09a0ffe11d52a997d9c0b02fe98d5f82433be13bd4d4150795379a0995b2a20a\aanca_supervisor.py`.
  The mutable development copy is never selected by the operational wrappers.
- Pinned Python 3.12.3 at SHA-256
  `15b41a488c356c0e331facdea6c836a6cec021f12d5fde9844e7ca4a1aa0361a`
  and private Codex CLI 0.145.0 at SHA-256
  `83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c`.
  The read-only launch wrapper has SHA-256
  `a15b3bddd2026671af7bab68729513d752fcb35e10faf6af718dfd38b668df8c`
  and verifies both program hashes before launch.
- Created and live-tested exact non-interactive session
  `019faaf3-c547-79e1-b0eb-26e35d214642`. Creation and exact
  `codex exec resume <SESSION_ID> <PROMPT>` both returned exit code 0; the pinned
  handoff receipt SHA-256 is
  `9c9b54f14191d75ea71d002e699fc4451905038eb4c3d3c11ec4246d662af23b`.
  `--last` and automatic retry are forbidden.
- Installed exactly one read-only per-user Startup entry,
  `AANCA-supervisor-recover.cmd`, SHA-256
  `3ee9c635a703ef4e3668ce2c77bdf655476bee698a374ffdb5c08c201a43bebe`.
  It invokes only the pinned global-root `recover-all` path. Direct and installed
  Startup canaries both returned exit code 0 and `no_jobs_to_recover`. Readback
  found zero AANCA common-Startup, Run/RunOnce, or Scheduled Task entries.
- The supervisor uses suspended process creation plus atomic Job assignment,
  PID/creation-time/image/command identity, a project-wide named mutex,
  `WaitForSingleObject` and Job completion events, scoped
  `SetThreadExecutionState`, bounded hashed logs, durable atomic receipts, exact
  seal/integrity roles, and an exit-0 verifier. It has no agent/model polling,
  heartbeat, Task Scheduler, scientific retry, or global power-plan change.
- A successful or STOP terminal state permits one local Codex wake attempt.
  Crash ambiguity after external acceptance cannot provide strict end-to-end
  exactly-once delivery without receiver idempotency/ACK; the fail-closed policy
  records ambiguity and never launches a second local wake.
- The closed external release manifest passed exact path/size/hash/attribute/
  reparse readback for 10 components and is read-only at SHA-256
  `010c38d37078ab162a65d0a085bc19bc23a0578196b8afc112b4a08435237719`.
  At the final installation inventory, `jobs/` was absent, no real supervisor
  spec was present, no matching long process was active, and no real AANCA
  process had been launched by this infrastructure.
- This is infrastructure qualification only. It does not authorize a primary,
  confirmatory, recovery, publication, or other scientific operation. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish replacement-v2 and rerun current-snapshot gates

Complete the replacement-v2 transaction tail already under integration, then run
its focused/broader/full pytest, Ruff check, Ruff format check, mypy,
protected-integrity and artifact-absence gates, the real PanNuke validator, and
the applicable read-only functional CLI on the same source snapshot. Do not arm
the supervisor with a real spec until that work passes, a separate legal
state-changing decision exists, and the exact command, authorization, expected
terminal seal, integrity receipt, verifier, timeouts, log ceilings, session, and
release hashes have been reviewed.

## 2026-07-29 - Replacement-v2 current-snapshot qualification gates passed

- The first complete repository test on the integrated snapshot was diagnostic,
  not a passed gate: **1,698 passed, 1 failed, 1 expected Windows skip**. The
  exact failure was
  `test_fresh_diagnostic_rejects_impossible_status_combinations` for a nominal
  `passed` diagnostic with
  `verifier_process_id == controller_process_id`. That state contradicted the
  required distinct-process/direct-child proof.
- The minimal fail-closed correction in `_canonical_fresh_diagnostic` rejects
  verifier/controller PID equality for both passed and analogous failed
  diagnostic states. No scientific definition, artifact, outcome value, frozen
  authority, or run was changed. The final controller is 413,798 bytes,
  SHA-256
  `db1e07cb4c8e5e4d1dbfef5ab3f2b5e0a815c09a4ddbcfcbd268fbcc9c76c679`.
- Post-correction focused gates passed:
  - replacement-v2 classifier/CLI/public/transaction E2E:
    **104 passed in 7.64 s**;
  - preflight authorization, fresh-process security, and terminal security:
    **261 passed in 20.10 s**;
  - amendment and schema-v3 lifecycle:
    **59 passed in 127.41 s**.
  The 104- and 261-test partitions form the complete **365-test**
  replacement-v2 set.
- The first final full-suite invocation was terminated by the shell harness after
  64.025 s with exit 124 before pytest emitted a result. Its process tree was
  confirmed absent. The integration owner then explicitly authorized one
  non-scientific QA replacement invocation with a sufficient tool timeout; it
  completed with exit 0:
  `.venv\Scripts\python.exe -B -m pytest -q` —
  **1,699 passed, 1 expected Windows skip in 687.98 s**. The skip is the declared
  Windows/POSIX open-file rename semantic case in
  `test_pannuke_publication_read_toctou.py`.
- Mandatory static gates on the same unchanged source passed:
  - `.venv\Scripts\python.exe -B -m ruff check .` — exit 0;
  - `.venv\Scripts\python.exe -B -m ruff format --check .` — **180 files already
    formatted**;
  - `.venv\Scripts\python.exe -B -m mypy src` — **no issues in 89 source
    files**.
- The full real-data command
  `.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke` completed with exit 0 in **306.3
  s**. It scanned all 3 folds, 7,901 patches, and 22 raw files; reported 4,318
  cross-class-overlap pixels in 575 patches, 10,486,091 void pixels in 162
  patches, zero positive-plus-background pixels, and exactly 1,411
  overlap-touching instances excluded identically from primary and confirmatory.
  It retained `no_class_arbitration=true`, `source_masks_modified=false`, and
  `publication=idempotent`.
- Independent post-validator readback matched **22/22** raw files by size and
  SHA-256 with zero mismatch and unchanged timestamps. The three archive hashes
  remain exactly the recorded `6E19...ED0B`, `5BC5...9B07`, and
  `C14D...A39D`. The principal validation JSON remains SHA-256
  `094497f43e2ee0bd5dabddcd01f8c934657f130450a66f46600311451d36bc4`;
  the immutable QC manifest remains
  `0b188ecc586ed772b29845e15e169fb492ed8d2ad0f5b1a6643531ccee10857f`.
- The public read-only Authority-C verifier passed with `chain_depth=3`,
  `integrity_verified=true`, artifact root
  `57f9345eb78e700267916a059f23c550aa60b606125a5420e2c51152268d8627`,
  and manifest
  `4f8db0571252a851645b13fa523c8d53914d7939c7178e43d8319f84fa560156`.
  Protected hashes remain exact:
  - `SPEC.md`:
    `9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`;
  - `PRE_REGISTRATION.md`:
    `7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`;
  - primary frozen config:
    `0b11c1cccb47e954274511577d5ca02fbf7f84b04d8d7cb0ed1880fc70cb1fd9`;
  - confirmatory frozen config:
    `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`.
- Independent and final local `--classify` executions both returned exit 0 with
  exact state `qualification_required`, reason `terminal qualification Q has not
  been published`, zero candidate directories, null Q/I3/U2/A2/S2/F2 hashes,
  `publication_performed=false`, and `automatic_retry_allowed=false`. Final
  inventory found exact amendment chain A/P/C, zero of 24 relevant locks, zero
  controller or pytest processes, and no Q, input-v3, authorization-v2,
  A2/S2/F2, Authority D, lifecycle, scientific, confirmatory, or M9 artifact.
- One unavoidable cross-component crash boundary remains explicit: after the
  amendment creator commits D but before durable S2, a process loss can leave
  A2+D without S2/F2. That state must be preserved and classified
  `STOP_AMBIGUOUS`; it is never synthesized into `COMMITTED` and never
  automatically retried.
- Free space after all gates was 166,676,078,592 bytes (155.229 GiB).
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - separate permission for the first immutable Q write

All implementation and read-only qualification gates are green. No state-changing
command is currently authorized. The next proposed gate, only after a separate
explicit governance decision, is exactly:

```powershell
.venv\Scripts\python.exe -B -m histo_audit.workflows.resource_authority_d_replacement_v2_controller --qualify-terminal --project-root . --parent-authority-dir artifacts\preregistration_amendments\20260727T170413.080954Z
```

That command may publish only the O_EXCL terminal qualification Q. It must be
followed by an independent exact readback before any input-v3 freeze, authorization,
publication, lifecycle, supervisor arming, or scientific execution. Without that
separate permission, stop here with the qualified absence state intact.

## 2026-07-29 - Replacement-v2 terminal qualification Q published and verified

- The user explicitly authorized exactly one durable replacement-v2 terminal
  qualification write and its independent verification. The pre-write live gate
  returned exact `qualification_required`, zero Authority-D candidates, null
  Q/I3/U2/A2/S2/F2 hashes, zero of 24 relevant locks, zero competing scientific
  or replacement processes, exact protected-file hashes, and a valid Authority-C
  chain at depth 3.
- Two direct `.venv\Scripts\python.exe -m ... --qualify-terminal` launchers
  stopped fail-closed with exit 1, `status=stopped_without_write`,
  `replacement_state=qualification_required`, and error SHA-256
  `0073b082e608bfa94d44d6c99ed1e5e0013a20e44bb876754ba4817c61ff5a48`.
  The hash maps exactly to
  `ControlError: terminal process quiescence is not exact`. Read-only process-tree
  forensics proved that the Windows venv launcher created a shim PID and a real
  interpreter PID with the same module command line. The first process-quiescence
  probe runs before any publication lock, `O_EXCL` open, or Q write, so neither
  launcher entered the write transaction or consumed the authorized durable Q.
- A base-interpreter/venv-argv0 canary produced exactly one controller PID while
  preserving the exact venv `sys.executable`, `sys.prefix`, base interpreter,
  installed environment, arguments, and controller bytes. This changes only the
  Windows launcher topology and does not bypass or weaken process, lock, source,
  history, Authority-C, or receipt verification.
- The sole durable Q write through that qualified launcher exited 0 with
  `status=qualified_rolled_back_failure_no_retry`. It created exactly
  `artifacts/resource_control/resource_authority_d_replacement_v1_terminal_qualification_v1.json`,
  21,274 bytes, SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  It reported `publication_performed=false`,
  `scientific_execution_performed=false`, and
  `outcome_value_interpretation_performed=false`.
- A fresh-process classifier independently returned exact
  `input_freeze_required`, the same Q SHA-256, zero candidates, null
  I3/U2/A2/S2/F2 hashes, `publication_performed=false`, and
  `automatic_retry_allowed=false`. A separate public schema-v3 lineage verifier
  passed exact canonical receipt readback, Authority-C lineage, protected/live
  bindings, empty process matches, and empty first/second lock scans.
- No input-v3, publication authorization-v2, A2/S2/F2, Authority D, lifecycle,
  supervisor job, scientific cell, confirmatory result, or M9 artifact was
  created. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze the singleton input-v3 carrier once

Execute the controller's one-shot `--freeze-inputs` operation through the same
qualified base-interpreter/venv-argv0 launcher. Then require a fresh independent
classifier state of exact `authorization_required`, exactly four regular
single-link input-v3 files, matching record-root SHA-256, unchanged Q and
protected/run-state/A-P-C evidence, and absent U2/A2/S2/F2/D. Any nonzero,
ambiguous, partial, or mismatched state is STOP with no automatic retry.

## 2026-07-29 - Exact direct-venv launcher contract and input-v3 gate PASS

- The exact successful Windows process contract used for Q and I3 was:
  - CreateProcess executable:
    `C:\Users\NATAN\AppData\Local\Programs\Python\Python312\python.exe`;
  - argv[0]:
    `C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe`;
  - argv prefix:
    `-B -m histo_audit.workflows.resource_authority_d_replacement_v2_controller`;
  - fixed trailing argv:
    `--project-root . --parent-authority-dir
    artifacts\preregistration_amendments\20260727T170413.080954Z`;
  - working directory:
    `C:\Users\NATAN\Documents\AANCA`;
  - stdin was `DEVNULL`; stdout/stderr were captured separately; no shell was
    used for the child CreateProcess.
  Python `subprocess.run(argv, executable=sys._base_executable, ...)` supplied
  the distinct executable/argv0 pair. The outer venv process used only
  `-I -B -c "import os; exec(os.environ['AANCA_WRAPPER_CODE'])"` and did not
  carry the replacement-controller token. Q used mode `--qualify-terminal`;
  I3 used mode `--freeze-inputs`; the independent classifiers used
  `--classify`.
- The exact independent public Q verification call was:

```python
verify_resource_bounded_replacement_terminal_qualification_receipt(
    Path.cwd().resolve()
    / "artifacts"
    / "resource_control"
    / "resource_authority_d_replacement_v1_terminal_qualification_v1.json",
    project_root=Path.cwd().resolve(),
    parent_authority_directory=(
        Path.cwd().resolve()
        / "artifacts"
        / "preregistration_amendments"
        / "20260727T170413.080954Z"
    ),
)
```

- The one-shot I3 process completed naturally with exit 0 in **907.8 s** and
  returned `status=input_v3_frozen`, `input_v3_file_count=4`, and records root
  `70d74cfa98e22e97d52c3342a88f795796e9b16a5a08324a904f34b2dd970bbd`.
  It reported `automatic_retry_allowed=false` and all three
  publication/scientific/outcome-interpretation flags false.
- The exact carrier
  `artifacts/resource_control/authority_d_replacement_inputs_v3` contains only:
  - `authority_d_replacement_cnn_correction_receipt.json`: 4,452 bytes,
    SHA-256
    `0cbe705d23a4c168af1abdfc39b4e4d3be63903c9a776e561c3c9d3959ea898e`;
  - `authority_d_replacement_frozen_source_receipt.json`: 4,233 bytes,
    SHA-256
    `a49973544f652ecac166b0c47a73faa45508c186e4a5cf116555e1308dd67e32`;
  - `authority_d_replacement_source_allowlist.json`: 4,150 bytes,
    SHA-256
    `05128735fe5b1a7f1552c3f6b542a14e6b2e8cf632c718735da3321d18519ec3`;
  - `authority_d_replacement_workspace_plan.json`: 12,186 bytes,
    SHA-256
    `d3c7c30f86a35d7f0fa242db892ea200c2d2e043522ab0f5e0ade0aa59c5f87b`.
  All four are canonical UTF-8 JSON regular files, non-reparse and
  single-link. Independent canonical records-root recomputation exactly matched
  the CLI root.
- A fresh classifier completed in **230.0 s** with exact
  `authorization_required`, unchanged Q SHA-256, matching I3 root, zero
  Authority-D candidates, null U2/A2/S2/F2 hashes,
  `publication_performed=false`, and `automatic_retry_allowed=false`.
- No U2, A2/S2/F2, Authority D, lifecycle, supervisor job, scientific cell, or
  M9 artifact was created. Formal status remains
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact command - create one publication authorization-v2 receipt

Use the direct-venv process contract above with this exact child argv:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m histo_audit.workflows.resource_authority_d_replacement_v2_controller --authorize-publication --project-root . --parent-authority-dir artifacts\preregistration_amendments\20260727T170413.080954Z
```

CreateProcess must use the exact base executable recorded above while retaining
the venv path as argv[0]. Execute once only. On exit 0 require
`status=authorized_for_one_attempt`, `max_attempt_count=1`,
`automatic_retry_allowed=false`, and a fresh independent classifier state of
exact `ready`. Any nonzero, ambiguity, partial receipt, hash mismatch, lock,
process, candidate, or protected/run-state/A-P-C drift is STOP without retry.

## 2026-07-29 - Authorization-v2 stopped before write; deterministic contract defect proved

- The one authorized U2 operation used the exact direct-venv process contract
  above and ran once. It terminated naturally after **1,141.1 s** with exit 1,
  `status=stopped_without_write`, exact
  `replacement_state=authorization_required`, error SHA-256
  `0e5cec346272d35f96b3a60cfdcc3194ac3ec5cbb525a2d1b609fc4b642862c1`,
  `publication_performed=false`, `automatic_retry_allowed=false`, and both
  scientific/outcome-interpretation flags false.
- Static source tracing and a separate **922.5-second**, strictly in-memory
  two-preflight reproduction identified the exact exception:

```text
ControlError: publication authorization-v2 publication must contain exactly ['amendment_purpose', 'amendment_schema_version', 'amendment_timestamp_utc', 'chain_depth', 'intended_authority_directory', 'parent_authority_directory']
```

  The dynamic reproduction failed at exact stage `canonicalize_receipt` with
  the same full SHA-256 and `publication_performed=false`.
- The production live-preflight builder creates the publication mapping without
  `parent_authority_directory`, while the real authorization canonicalizer
  requires that sixth field. Receipt canonicalization occurs before the
  `publish_bytes_no_overwrite(...)` call, so the failure is deterministic,
  pre-publication, and unrelated to PanNuke, capacity, clocks, source drift, or
  transient filesystem state.
- The missed regression is also exact: synthetic canonicalizer fixtures already
  included `parent_authority_directory`, and the flow harness replaced the real
  canonicalizer with an identity function. Therefore the final integrated suite
  did not execute the mismatched production builder through the real
  canonicalizer.
- Independent post-failure qualification found exact durable Q+I3 only. U2,
  A2/S2/F2, and every Authority-D candidate are absent; the amendment inventory
  remains A/P/C; all relevant bundle/target locks and replacement processes are
  absent. NTFS USN evidence contains the expected Q and I3 create/close records
  but zero records for the exact U2/A2/S2/F2 filenames, proving that U2 was not
  even transiently created and rolled back.
- The current U2 operation is nevertheless consumed by the recorded one-shot
  governance rule. Do not invoke `--authorize-publication` again. Do not
  hot-patch, monkeypatch, or edit the current replacement-v2 controller: Q and
  I3 both pin it at 413,798 bytes and SHA-256
  `db1e07cb4c8e5e4d1dbfef5ab3f2b5e0a815c09a4ddbcfcbd268fbcc9c76c679`.
  Retrying unchanged code would deterministically fail; changing it in place
  would invalidate the live Q/I3 controller binding.
- No Authority D, lifecycle, supervisor job, scientific cell, confirmatory
  result, or M9 artifact was created. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - isolate and qualify a new post-U2 successor protocol

Preserve Q, I3, the failed U2 invocation, protected files, Authority C, and all
run-state evidence read-only. Implement a separately versioned controller and
marker namespace in an isolated staging overlay, not in the live execution
source. It must authenticate the exact pre-write U2 failure and absence proof,
read Q/I3 as historical sealed evidence, reconstruct the corrected six-field
publication contract, use a new one-attempt authorization and A/S/F namespace,
and retain every fail-closed/no-retry/crash-boundary rule. Add a real
production-builder-to-real-canonicalizer regression. Run focused and broader
gates in staging before any live integration, terminal receipt, new source
freeze, authorization, publication, lifecycle, supervisor arming, or science.

## 2026-07-29 - Post-U2 successor hardening checkpoint; no live successor write

- Re-ran the independent public terminal-Q verifier in a fresh process against
  `artifacts/resource_control/resource_authority_d_replacement_v1_terminal_qualification_v1.json`.
  It passed exact canonical readback at 21,274 bytes and SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  with status `qualified_rolled_back_failure_no_retry`. The sole durable Q write
  remains consumed; no second Q was created or attempted. The same verifier was
  run again after the initial append-only STATUS/DECISIONS checkpoint and returned
  `Q_POST_DOC_APPEND_PASS`, confirming that the recorded governance append did
  not violate the sealed Q boundary.
- Kept all successor implementation work isolated under
  `C:\Users\NATAN\Documents\AANCA_u2_successor_staging_20260729`.
  The external production root
  `C:\Users\NATAN\AppData\Local\AANCA-control-plane` remains absent, as do R3,
  I4, U3, A3, S3/F3, and every successor Authority-D artifact.
- Closed the fresh-preflight style/type gate without changing its two-child,
  direct-process, no-retry semantics. The current staged helper is 70,406 bytes,
  SHA-256
  `3a9399a564b0489a62488fe05d535f7ebde30e31e2488998d7ce3767c32bd966`;
  its focused result was 7 passed plus repo-config Ruff, format, and strict mypy
  PASS.
- Hardened the external one-use authorization helper so a self-consistent decoy
  release or authorization directory cannot consume the attempt. It now
  requires the exact fixed roots
  `%LOCALAPPDATA%\AANCA-control-plane\releases` and
  `%LOCALAPPDATA%\AANCA-control-plane\external-one-use-authorization`.
  The staged helper is 109,845 bytes, SHA-256
  `ef9089bdec45b7ed0726fd129b2d6451c0c00444937c0abb998b37f421b6d2bb`;
  its independent result was 94 passed plus repo-config Ruff, format, and strict
  mypy PASS.
- Executed the stable-component aggregate from the isolated staging root:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q tests\test_external_release_builder.py tests\test_r3_reproducer_source_closure.py tests\test_u3_external_release_binding.py tests\test_u3_fresh_preflight.py authorization_tests\test_external_one_use_authorization.py
```

  Result: **151 passed in 24.22 s**.
- Executed the current schema-v4 focused gate:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q schema_v4_tests\test_post_u2_schema_v4.py
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\ruff.exe check --config C:\Users\NATAN\Documents\AANCA\pyproject.toml schema_v4 schema_v4_tests
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\ruff.exe format --check --config C:\Users\NATAN\Documents\AANCA\pyproject.toml schema_v4 schema_v4_tests
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m mypy --strict --config-file C:\Users\NATAN\Documents\AANCA\pyproject.toml schema_v4\resource_authority_d_post_u2_schema_v4.py schema_v4\generate_preregistration_schema_v4_dispatch.py
```

  Result: **40 passed** and all three static gates PASS. This is not yet a
  frozen architecture gate because the exact final R3 ordered-read-set contract
  is still moving.
- A separate read-only audit and executable adversarial canary proved that the
  previous I4 validator accepted arbitrary nested amendment claims, including
  the prohibited obsolete `qualified` scope. The same audit found that the
  previous I4 accepted a skeletal R3/C3, did not cross-link the exact external
  authorization, release, amendment, Q/I3/controller evidence, and did not
  repeat full verification after the I4 seal. I4 publication remains blocked
  until these paths delegate to the byte-verified released controller and pass
  adversarial tests.
- The current moving controller aggregate returned **189 passed, 8 failed**.
  The eight failures are stale fixtures/signatures for the newly required
  amendment, downstream source-release binding, and removed injected baseline
  seam. More importantly, the production `--qualify-failed-u2` collector and
  its under-lock final observation were not yet complete at this checkpoint.
  No live publication is authorized until the production-only reconstruction,
  exact closed read set, under-lock final evidence, downstream-safe verifier,
  and full CLI E2E all pass.
- No training, primary, confirmatory, recovery, publication, scientific-cell,
  or outcome-reading process was running during this checkpoint. Available
  capacity was 157.63 GiB, so disk capacity is not the blocker.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish and independently audit the production R3 transaction

Complete the staged `--qualify-failed-u2` path so it reconstructs all evidence
from the fixed project and external control plane, accepts no caller-supplied R3
core, collects the final absence/process/lock observation while the same
publication locks are held, and binds the exact authorization amendment. Then
run the focused controller tests, the independent second-pass audit, the
combined auth/schema/I4 gates, and the complete staging Ruff/format/mypy/pytest
gates. Do not create the external release, amendment, authorization, R3, I4,
U3, Authority D, supervisor job, or scientific result before all of those gates
pass.

## 2026-07-29 - R3/U3 controller checkpoint frozen; external publication still blocked

- Froze the isolated successor controller at 314,853 bytes and SHA-256
  `96f9b89d5df2f2b5431cdce890c38adf01eeedfccd797fdd764e5765f95b58a0`.
  Froze its released R3 observation helper at 139,948 bytes and SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`.
- The sealed verifier no longer calls the pre-seal observer. Its fixed internal
  lifecycle selects only `R3_ONLY` or `I4_SEALED`, invokes the corresponding
  fixed two-child run and validation APIs, requires exact sealed R3/I4
  before/after records, and repeats the sealed readback under the same 37
  protocol locks. Neither public verifier accepts a caller-selected lifecycle
  or relaxed present/absence set.
- The production R3 qualifier accepts only the project namespace and fixed
  parent. It reconstructs Q, I3, Authority C, protected/run-state evidence,
  governance excerpts, external release, exact authorization/amendment,
  deterministic failure proof, process/lock absence, and NTFS-wrap compensating
  evidence as a closed ordered 43-role read set. It accepts no caller-supplied
  receipt core, authorization hash, release record, publisher, clock, or
  relaxed-verification switch.
- Executed the independent stable staging aggregate:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q tests authorization_tests
```

  Result: **363 passed in 29.49 s**. The controller and reproducer SHA-256
  values were identical before and after the run. The integration owner's
  narrower final controller gate returned **220 passed**, with strict mypy,
  Ruff check, and Ruff format-check all passing.
- The public terminal-Q readback remains exact at 21,274 bytes and SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  Historical I3 readback remains the same four files with records root
  `70d74cfa98e22e97d52c3342a88f795796e9b16a5a08324a904f34b2dd970bbd`.
  The independent public verifier was rerun after this append-only documentation
  checkpoint and returned `Q_POST_CHECKPOINT_DOC_APPEND_PASS`. These were
  read-only checks; neither one-shot operation was repeated.
- No Python training, primary, confirmatory, recovery, successor-publication,
  or scientific process was running. The fixed live external control-plane root
  remains absent. R3, I4, U3, A3, S3/F3, Authority D, lifecycle, and scientific
  outputs therefore remain absent.
- A separate read-only supervisor audit re-ran all **63 synthetic tests** and
  found no active job/state/STOP. It also found two pre-arm readiness defects:
  the external next-run plan still describes the retired Q/I3/U2 flow, and a
  corrupt top-level recovery state can fail only to invisible `pythonw` stderr
  instead of a durable root STOP plus one diagnosis wake. Supervisor arming
  remains prohibited until those paths and exact-argv process absence are
  hardened and independently retested.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - complete independent P0, I4, schema-v4, and supervisor gates

Keep the frozen controller and reproducer byte-identical. Complete the
independent P0 call-graph/injection/read-set audit, bind I4 to the exact
byte-verified `I4_SEALED` verifier with fixed external roots, and mirror the
closed 43-role contract in schema-v4. In parallel, harden only the external
supervisor recovery/plan/process-absence paths and rerun its synthetic gates.
Then run the combined full staging pytest, Ruff check, Ruff format-check, strict
mypy, and read-only real CLI canaries. Do not create the external release,
technical amendment, one-use authorization, R3, I4, U3, Authority D,
supervisor job, or scientific result until every gate passes.

## 2026-07-29 - Independent audit rejected the first controller checkpoint; supervisor qualified and unarmed

- The independent P0 audit rejected staged controller SHA-256
  `96f9b89d5df2f2b5431cdce890c38adf01eeedfccd797fdd764e5765f95b58a0`
  before any live write. It found that an internal mutation-capable transaction
  still accepted caller-supplied prelock evidence, released callable bindings
  were initially code/identity-only and lazily re-baselinable, and the final
  locked evidence did not repeat the complete ten-role successor namespace
  immediately before C3. Later moving snapshots closed those defects but were
  also rejected because defaults, keyword defaults, closures, and transitive
  release-owned private helpers/class methods were not yet semantically pinned.
  No rejected hash is eligible for release or I4 binding.
- The corrected controller work remains staging-only. The required final
  checkpoint must remove all mutation-capable prelock input, eagerly seal the
  exact callable inventory before yielding control, pin code plus semantic
  defaults/closures and the transitive release-owned callable surface, retain
  active runtime guards around sensitive calls, and repeat complete under-lock
  namespace/process/lock/lifecycle scans before C3, before R3, and after seal.
- Independently hardened the external Windows supervisor. The parent rerun
  first reproduced a repo-config style-gate defect (four `UP012` findings) even
  though the dedicated audit used a different Ruff surface. Only those
  semantics-preserving encoding/style corrections and repo-config formatting
  were applied. The exact corrected bytes then passed:

```text
python.exe -B -m pytest -q test_aanca_supervisor.py
ruff check --config C:\Users\NATAN\Documents\AANCA\pyproject.toml <five supervisor Python files>
ruff format --check --config C:\Users\NATAN\Documents\AANCA\pyproject.toml <five supervisor Python files>
python.exe -m mypy --strict --config-file C:\Users\NATAN\Documents\AANCA\pyproject.toml aanca_supervisor.py
```

  Result: **68 passed in 109.72 s**, Ruff check PASS, Ruff format-check PASS,
  and strict mypy PASS. The bounded `launch_hidden.ps1` synthetic E2E also
  passed separately.
- Installed the corrected supervisor as a new preserved, read-only,
  content-addressed release:
  `C:\Users\NATAN\AppData\Local\AANCA-supervisor\releases\75b91e95fe253b8e5fe42e8488d41fa8fd7677891a82de1aeaeaad928e9031d8`.
  Its file is 218,146 bytes and its SHA-256 exactly equals the directory name.
  The two superseded releases remain read-only and are not selected.
- The fixed wrapper is 3,648 bytes, SHA-256
  `d0c503cff0d43d6960dfa32dd2085f91423014503aedbab7b0e4efc9dcb5126a`.
  The recovery-only Startup template and installed copy are byte-identical,
  read-only, 312 bytes, SHA-256
  `23fcc0ab12a03ae00313871092231715de91bc179a74acc30597a31c8212c7b7`.
  Both point only to the corrected release and can select only `recover-all`.
- The final external supervisor manifest is 10,081 bytes, SHA-256
  `016739b52c5aa916ba4ad9f171d7a5af45d1a73d75f2a870a89e06c78c19a192`.
  An independent component/runtime validator returned `errors=[]`; direct and
  installed-Startup recovery canaries both returned exit 0 and
  `no_jobs_to_recover`.
- The exact saved handoff session remains
  `019faaf3-c547-79e1-b0eb-26e35d214642`; its tested command forbids `--last`.
  Current readback found no `jobs/`, no `recovery_stops/`, no external
  AANCA control plane, zero matching supervisor/synthetic/handoff processes,
  exactly one per-user recovery-only Startup entry, and zero common-Startup,
  Run/RunOnce, or Scheduled Task entries. The supervisor is infrastructure-
  qualified but deliberately **not armed**.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze and independently pass the semantic controller checkpoint

Finish the transitive semantic callable pin and its adversarial mutation tests,
then freeze one exact controller hash. Run a fresh independent P0 audit against
that hash. Only after an explicit PASS may I4 replace its four fail-closed
`None` constants, schema-v4 mirror the final exact contract, and the combined
staging gates run. The qualified supervisor remains unarmed, and no external
release/amendment/authorization/R3/I4/U3/Authority-D/scientific write is allowed
before those gates.

## 2026-07-29 - Semantic controller core passed; final integration remains fail-closed

- Froze the isolated semantic controller checkpoint at 362,257 bytes and
  SHA-256
  `39816d2d4598afd7a2fdb66821ce827096027422435b2df2341dfc04ee352b4d`.
  Its unchanged R3 reproducer is 139,948 bytes with SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`.
  The exact governance and external-binding tests are respectively 30,568
  bytes/SHA-256
  `168f27e40c3273f8f9c4e50d48997c05ea940c8dbee41a68e9de658faf0255bf`
  and 24,733 bytes/SHA-256
  `fbd639481bf013976e639c3fbf37528f1cb7fdd060cf23d42b052d7bc48f1f8c`.
- The parent independently ran the seven-file focused suite against unchanged
  bytes: **319 passed in 20.32 s**, with zero hash changes. Repository-config
  Ruff check, Ruff format-check, strict mypy on the four controller sources,
  and `py_compile` all passed. A separate audit passed exact-byte external
  loading, semantic defaults/keyword-defaults/closures and transitive
  release-owned surfaces, the closure-held runtime vault, mirror-rebind
  rejection, internally derived pre-lock evidence, the immediate ten-role
  rescans, and the closed 43-role read set.
- That audit also reproduced one production lifecycle integration defect before
  any live write: the I4 loader retained the sealed external release but called
  the raw R3 verifier without entering the required `R3_ONLY` or `I4_SEALED`
  runtime context. The deterministic error was `verified I4 live dependencies
  are not active in this operation`. Therefore this hash is accepted only as a
  semantic-core checkpoint and is not eligible as the final released
  controller.
- Corrected the separate I4 integration-generator TOCTOU in isolated staging.
  The generator is now compiled and executed from the already verified byte
  buffer, followed by exact source readback and interpreter-state checks; it
  does not reopen the source through a path/spec/pyc loader. Adversarial source
  swap, malicious pyc, forbidden path loader, and interpreter-mutation canaries
  passed. The exact corrected I4 source is 186,848 bytes/SHA-256
  `8c957bb8b40a2a072f6448eb2840d4d4ea3e2eff2f0a8a2af65de4b1922d540c`;
  its tests are 117,512 bytes/SHA-256
  `689c04031b40d78c42c2316419e6936202a7fc3dd038f054654440294d2500ee`.
  The agent and parent independently obtained **93 passed** on stable bytes;
  Ruff check, format-check, strict mypy, and `py_compile` passed. All four
  production I4 controller/reproducer constants remain `None`.
- A fresh schema-v4 audit found that its controller adapter pinned only
  `__code__`, leaving keyword defaults, closures, transitive helpers, and
  mutable baseline maps outside the seal. A child probe changed verification
  keyword defaults without changing the old code hash. Schema-v4 publication
  remains STOP until an immutable closure-held semantic seal and adversarial
  zero-write tests pass.
- Read-only publication readiness found two further pre-publication
  requirements. The calendar-date filename and same-date timestamp coupling
  must be replaced by the fixed singleton
  `r3_usn_wrap_technical_amendment_v1.json`, with canonical UTC microseconds and
  exact amendment-to-authorization chronology rechecked from the bound
  envelope. A saved, content-addressed bootstrap publisher and independent
  verifier are also required; stdin snippets are prohibited. Their preflight
  must be zero-write, and their real path must claim once, never adopt, repair,
  clean up, resume, or retry a partial publication.
- No live external control-plane root, R3, I4, U3, A3, S3/F3, Authority D,
  supervisor job, training, primary, confirmatory, recovery, or publication
  process was created. Free space was 157.455 GiB. Formal project status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**,
  and M9 remains locked.

## Next exact action - one consolidated final integration pass

Keep the semantic checkpoint as read-only evidence while integrating, in one
bounded pass, the two lifecycle-owned verifier runners, schema-v4 semantic
factory seal, real `--publish-once` and committed-classifier wiring, the fixed
versioned amendment and chronology contract, and the saved bootstrap
publisher/verifier. Then freeze one new final controller/schema/I4/auth/
bootstrap snapshot, bind the four I4 constants only to those audited bytes, and
run the full combined staging pytest, Ruff check, Ruff format-check, strict
mypy, `py_compile`, and real read-only canaries. Do not create the external
release, amendment, authorization, R3, I4, U3, Authority D, supervisor job, or
scientific output before every gate passes.

## 2026-07-29 - Full repository and real PanNuke gates passed; final successor snapshot still blocked

- The single logged full-repository test process completed naturally. Its exact
  command was:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

  The command SHA-256 was
  `0006fd77fec6e692a9824593221088d63d7a8b0a884e261d20bc317260f2ace1`.
  The terminal result was **1699 passed, 1 skipped in 703.54 s**, exit code 0,
  with empty stderr. The skip is the declared Windows/POSIX open-file rename
  semantic case. The atomic result record is 538 bytes/SHA-256
  `4eeb092901791c057fc097a6ad3e7a4e0dd6c422bf6bc466ecb8907d05fca18b`;
  stdout is 4,412 bytes/SHA-256
  `68fde7ea493e30a4487d0f14600615142069da35bc1b4f5ac865609cbd801127`;
  stderr is empty with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  All wrapper and Python PIDs exited after the atomic result appeared. No
  duplicate test process was launched.
- The repository static gates on the same project source also passed:
  `ruff check .`, `ruff format --check .` (**180 files already formatted**),
  and `mypy src` (**89 source files, no issues**).
- Executed exactly one full semantic real-data command:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke --project-root . --root data\raw\pannuke
```

  It completed in 321.11 s with exit code 0, empty stderr, and status `valid`.
  It read all 3 folds/7,901 patches/22 raw files and reproduced 4,318
  cross-class-overlap pixels in 575 patches, 10,486,091 unlabeled/void pixels
  in 162 patches, 737 affected patches, and 1,411 overlap-touching instances.
  All 1,411 instances retain the fixed
  `touches_cross_class_overlap` exclusion identically for primary and
  confirmatory analysis. It reported `no_class_arbitration=true`,
  `source_masks_modified=false`, and `publication=idempotent`.
- The validator stdout is 4,682 bytes/SHA-256
  `b9112963422f5921e42709d370b3709514a221a3e83c34a91dc621479e484f50`;
  its atomic result record is 523 bytes/SHA-256
  `d481e20278423854f70b4a1ea42c152aa6f60e8c12dfe5eea7c637cfe039d5ea`;
  stderr is empty. Windows PowerShell 5.1 lacked the launcher's optional static
  `SHA256.HashData` method, so the wrapper's `command_sha256` field is blank.
  The exact live command line was independently read back immediately after
  launch and hashed with `SHA256.Create().ComputeHash` as
  `438ba8b8957202028d3bc57078dffde4e8f08658f6b5cbdc1dc95f5b139ebd95`.
  The successful validator was not repeated merely to replace that optional
  wrapper metadata.
- A fresh production-shaped review rejected the previously passing combined
  staging checkpoint before any live write. `_protocol_locks` supplied only 27
  unique paths while the fail-closed R3 verifier requires 37. The exact closed
  topology is the union of 16 legacy-scoped paths and 12 v2 bundle paths
  (6 overlap, hence 22 unique), plus 13 successor bundle paths and 2
  Authority-C parent bundle paths: **37 unique paths**. Intermediate fixes are
  ineligible until exact-text case-alias rejection and the 36/duplicate/extra/
  case-alias STOP-before-publisher canaries pass on one byte-stable controller.
- The prior 660-test staging aggregate is retained only as historical evidence;
  it cannot qualify the controller discovered to have the 27/37 lock defect.
  I4 remains bound to that superseded controller and must be rebound only after
  the corrected controller is independently frozen. No external control-plane
  root, replacement-v2 attempt/success record, release, amendment,
  authorization, R3, I4, U3, Authority D, supervisor job, or scientific output
  was created. The user's one authorized replacement-v2 operation therefore
  remains unconsumed.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze the exact 37-lock controller and rebind I4

Finish the bounded controller fix and independently prove the exact 37-path
inventory, exact-text alias rejection, three under-lock rescans, fixed
Authority-D destination guard, and STOP-before-publisher negative cases on one
stable hash. Rebind I4 only to that hash, complete the independent bootstrap
preflight audit, and rerun the full combined staging pytest/Ruff/format/mypy/
`py_compile`/real canary sequence on unchanged bytes. Do not provision or
execute the one-shot external publication before all of those gates pass.

## 2026-07-29 - Exact 37-lock controller and four-pin I4 rebind independently qualified

- Froze and independently qualified the corrected successor controller at
  379,960 bytes/SHA-256
  `a6677cc32fa23fcd09639cdc3dfd38a6ad98e647f6ce79d94eed25cbbe270919`.
  Its unchanged R3 reproducer remains 139,948 bytes/SHA-256
  `7c5c24a92414f46260f6783a59253331e1d3c2379681422484ad0e2eb61559ba`.
  The independent 14-file snapshot root was
  `50cafd7b7a9f8a6694025c1977af3ff73351a0db2d52376c23130598558c4d0c`
  before and after audit.
- Independently reconstructed the exact production lock set. The verified v2
  component contributes 12 lock paths, the successor component 13, and the
  Authority-C parent component 2. Their old union was 27. The 16-path legacy
  component overlaps that union in 6 paths and contributes 10 missing paths,
  giving exactly 37. The complete sorted exact-set SHA-256 is
  `1deadeb975a4ea873596961e88d6c7079ccdf1e24ace3edebd04fee2be1651e9`.
- The production controller now derives that set only from the verified I3
  topology and fixed successor/parent geometry, retains exact text alongside
  normalized comparison keys, and re-derives the topology on every owned-lock
  scan. The same owned lock is checked in the initial, repeated, and immediate
  pre-C3 scans, with further pre-R3 and terminal checks. The independently
  executed 37-positive, 36, duplicate, extra, case-only-alias,
  cross-component-alias, and post-acquisition-topology-change canaries all
  passed; every negative stopped before a publisher call. The focused R3 suite
  returned **64 passed**; the integration-owner aggregate returned
  **356 passed**. Strict mypy and the real six-file sealed-release canary also
  passed without a context leak.
- The canonical staging lint invocation is from the staging package root with
  relative staging paths, the live repository `pyproject.toml`, and
  `--no-cache`. It passed Ruff check and format-check on the unchanged bytes.
  Invoking Ruff from an unrelated working directory with absolute staging test
  paths changes Ruff's first-party classification of the local `controller`
  package; that noncanonical invocation was not used as a release gate. To
  ensure the repository gate was not cache-dependent, the parent separately
  reran `ruff check --no-cache .` and `ruff format --check --no-cache .` in the
  repository: all checks passed and all 180 files were already formatted.
- Rebound I4 exactly once to the final controller. Only the controller
  size/hash constants and their two test expectations changed; the reproducer
  pins remained unchanged. The frozen I4 source is 188,584 bytes/SHA-256
  `25b721a87d364b5ab9a664cea2ae04799ab131be3ceb73baa07d93951bf4e3cd`;
  its tests are 127,673 bytes/SHA-256
  `899ab9d03a8671625b88ae136113802ae551c0480c9a5fde15f07882eacea11c`.
  Reversing only those four occurrences reconstructs the prior source/test
  hashes exactly.
- The independent I4 audit returned **100 passed** plus **2 passed** exact/
  mismatch canaries, canonical no-cache Ruff/format PASS, strict mypy PASS on
  production I4, and `py_compile` PASS. It verified exact-byte
  stable-read/compile/exec, no path/spec/pyc loader, post-seal use of the
  production `I4_SEALED` verifier, fixed external/auth/amendment/control roots,
  and no caller-selected relaxed path. All four audited files were identical
  before and after.
- The read-only public Q verifier was rerun after the prior documentation
  append and returned `Q_POST_37_LOCK_DOC_APPEND_PASS`: 21,274 bytes, SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  status `qualified_rolled_back_failure_no_retry`. No second Q was attempted.
- The live external control plane, external authorization root, R3, I4, U3,
  A3/S3/F3, Authority D, supervisor job, and scientific outputs remain absent.
  Bootstrap hardening is the only current pre-staging blocker: its fresh child
  must be isolated from both environment and venv `sitecustomize`, use bounded
  subprocess I/O, and replace lazy `psutil` imports with the fixed
  standard-library/Windows inventory before its final independent audit.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze and independently qualify the isolated bootstrap

Finish the already bounded `-I -S -B` bootstrap hardening, fixed child
environment/working directory/stdin/handle/output bounds, standard-library
process inventory, exact self-attestation, and malicious `PYTHONPATH` plus venv
`sitecustomize` canaries. Freeze one publisher/verifier/test snapshot and pass
the independent audit. Then run the complete combined staging gates once on
the unchanged controller/schema/I4/auth/bootstrap bytes. Do not provision the
external evidence root or invoke `--execute-once` before those gates.

## 2026-07-29 - Isolated bootstrap and complete successor staging gate qualified

- Froze the external bootstrap publisher at 83,509 bytes/SHA-256
  `feab2a751a3118e5f5ec438648f160f64bddb60ef3bbd9a39be349b3fc9cd938`
  and its independent verifier at 58,804 bytes/SHA-256
  `34380beeebec12e057704d222bc9250c6ded40bbc5da8345ca36df41190d83b2`.
  The independently reconstructed ten-file bootstrap aggregate was identical
  before and after audit at
  `e04d2e5a698048b560ea1e9f8edc34f176a475a347c7dde4314aebdf1997c184`.
- The bootstrap suite returned **107 passed**. A separate real-process canary
  executed the content-addressed publisher as `python -I -S -B ... --preflight`
  and passed with exit 0, zero governed writes, and no `--execute-once`.
  Non-isolated publisher invocations stop before loading the plan. Independent
  Ruff/format, strict mypy, in-memory compile, hardlink/reparse/ADS/writable
  evidence, closed-environment, malicious `sitecustomize`, PowerShell-module,
  bounded-output/timeout, runtime/self/flags, O_EXCL, and no-retry canaries all
  passed. The audit found no P0/P1; the only accepted P2 is Windows-managed
  PowerShell `StartupProfileData` outside all governed AANCA roots.
- Froze the complete isolated staging inventory at **31 Python files**: 11
  production files plus 20 test/support files. Its canonical manifest root was
  identical before and after:
  `074b1ae5d6df74675e9cc0afe67657367ab7e11d56bbe0e57091a4524a31cdf4`.
  The inventory contained no missing or extra production source, content
  duplicate, case-insensitive alias, or reparse path.
- Executed the canonical complete staging test command from the staging root:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -c C:\Users\NATAN\Documents\AANCA\pyproject.toml tests authorization_tests schema_v4_tests i4_tests bootstrap_tests
```

  It exited 0 with **717 passed in 235.82 s**, empty stderr, command SHA-256
  `044f4e205b84a28ca0509ff39e3d7a0e7bb2236ab167580f07cf7ec44fc3066f`,
  and stdout SHA-256
  `8e2c7d4cbef288b49ac2a428d1748fb6cdcf3c073c56d6e5116afd9d71397890`.
  An immediately preceding diagnostic invocation without the explicit
  `-c` also returned 717 passed, but it is not the qualifying run.
- The same frozen staging bytes passed canonical no-cache `ruff check`,
  `ruff format --check` (**31 files**), strict non-incremental mypy
  (**11 source files**), and in-memory compilation (**11 source files**).
  The manifest root remained exact after the gates. Test-generated
  `__pycache__`/tool-cache files are staging-only, outside the manifest and
  every release role; no cache is eligible for external publication.
- The external control-plane and bootstrap evidence roots remain absent, as do
  R3, I4, U3, A3/S3/F3, Authority D, a supervisor job, and scientific output.
  No training or confirmatory process is running. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - immutable bootstrap evidence and zero-write production preflight

Construct the closed production bootstrap plan from the frozen 31-file
inventory and historical Q/I3 evidence. Independently verify every path,
size/hash, semantic pin, exact 37-lock inventory, runtime pin, safety flag, and
content root before any write. Then provision only the content-addressed
publisher, verifier, and canonical plan under the fixed protected external
bootstrap evidence root, make their evidence bytes read-only, and run exactly
one production `python -I -S -B ... --preflight`. It must exit 0 with the
zero-write receipt and leave the claim/success/failure, external control plane,
amendment, authorization, and Authority-D targets absent. Only after an
independent readback of that state may exactly one `--execute-once` be invoked;
there is no automatic retry.

The fresh public Q verifier was rerun after this documentation append and
returned `Q_POST_BOOTSTRAP_DOC_APPEND_PASS`: 21,274 bytes, SHA-256
`9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
status `qualified_rolled_back_failure_no_retry`. No second Q was attempted.

## 2026-07-29 - Immutable bootstrap evidence and one-shot external control plane completed

- Froze and independently qualified the external production provisioner at
  51,512 bytes/SHA-256
  `47220e778aaaa4828cd310ec12ebf2f4a87ddf805b7705740a321d79e1aebdf4`
  and its test source at 45,510 bytes/SHA-256
  `9492cf7a2d3ae5b3aca7b5b064328c0b9857be84b449f17cbc1a50b539039a2e`.
  The independent gate returned **78 passed**, Ruff check/format, strict mypy,
  in-memory compile, and an isolated zero-write dry run PASS. The canonical
  dry-run stdout was 16,155 bytes/SHA-256
  `c054fa1c50b1e8a301491051914d3979b365f2ca3d73e6ee04e1b4f20119aadf`.
  The audit found zero P0/P1.
- Executed exactly one provisioner process:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -I -S -B C:\Users\NATAN\Documents\AANCA_bootstrap_provisioner_staging_20260729\bootstrap_plan_provisioner.py --provision-once --expected-provisioner-size-bytes 51512 --expected-provisioner-sha256 47220e778aaaa4828cd310ec12ebf2f4a87ddf805b7705740a321d79e1aebdf4
```

  The exact command-text SHA-256 was
  `395d0ad1dc2d85b031c024957c0dd08eff16a505c7bcc0e7010ef3c70801ede9`.
  It exited 0 with status `provisioned_once_and_verified`, empty stderr, and
  canonical stdout of 1,965 bytes/SHA-256
  `5036cd8c2f938228a25b8885a4ef176c43c27c053103ff3ebca445a6c46e30f1`.
  A preceding capture-wrapper attempt stopped before `Start-Process` because
  Windows PowerShell 5.1 does not support `New-Item -LiteralPath`; it created no
  directory, process, or governed state. The corrected wrapper changed only
  that argument to `-Path`, and the provisioner itself was invoked once.
- Provisioning created exactly seven protected evidence directories and three
  read-only, single-link files: publisher 83,509 bytes/SHA-256
  `feab2a751a3118e5f5ec438648f160f64bddb60ef3bbd9a39be349b3fc9cd938`,
  verifier 58,804 bytes/SHA-256
  `34380beeebec12e057704d222bc9250c6ded40bbc5da8345ca36df41190d83b2`,
  and plan 14,435 bytes/SHA-256
  `36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e`.
  The two-pass independent post-provision snapshot was
  `e86d8ed9f5db29c58e61efa5f8cdc14777bb7fe3d016d662cf519406512b2923`;
  it verified the exact 37-lock root
  `1deadeb975a4ea873596961e88d6c7079ccdf1e24ace3edebd04fee2be1651e9`,
  secure DACLs, no ADS/reparse aliases, and absence of the claim, terminal
  records, external control root, locks, and protected processes.
- Executed exactly one public production zero-write preflight from the
  content-addressed publisher directory:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -I -S -B C:\Users\NATAN\AppData\Local\AANCA-control-plane-bootstrap\publishers\feab2a751a3118e5f5ec438648f160f64bddb60ef3bbd9a39be349b3fc9cd938\external_control_plane_bootstrap.py --preflight --plan C:\Users\NATAN\AppData\Local\AANCA-control-plane-bootstrap\plans\36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e\external_control_plane_bootstrap_plan_v1.json --expected-plan-sha256 36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e
```

  Its command-text SHA-256 was
  `1efb7a452e8587c938447a43e82dbb60c2cd0a7f4a4e1f75ff97eaa4d599fa99`.
  It exited 0 with `preflight_passed_zero_write`, `write_performed=false`,
  empty stderr, and canonical stdout of 317 bytes/SHA-256
  `3ccf5d40afa92f3f35c8861a2b46b20f623d92739124b16c27609bcb224116d9`.
  Two independent before/after readbacks retained snapshot
  `e86d8ed9f5db29c58e61efa5f8cdc14777bb7fe3d016d662cf519406512b2923`.
- After that gate, executed exactly one production `--execute-once` with the
  same runtime, publisher, plan, working directory, and isolation flags:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -I -S -B C:\Users\NATAN\AppData\Local\AANCA-control-plane-bootstrap\publishers\feab2a751a3118e5f5ec438648f160f64bddb60ef3bbd9a39be349b3fc9cd938\external_control_plane_bootstrap.py --execute-once --plan C:\Users\NATAN\AppData\Local\AANCA-control-plane-bootstrap\plans\36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e\external_control_plane_bootstrap_plan_v1.json --expected-plan-sha256 36eb8593522b01d4b24d8834a875774356b2d4f6dcf3c06610b6b68cb75d475e
```

  Its command-text SHA-256 was
  `c09c4c31d21a24491caa2d7847ce3b5264fa3ccca5a83eed69f28c2dd7e693cb`.
  It exited 0 after 269.1 seconds with empty stderr and canonical file-envelope
  stdout of 922 bytes/SHA-256
  `2efc6b8a3e1e1d16a673f06cfd10945cc7b2fc27b6f3eff03579595294468c51`.
  The immutable claim is 829 bytes/SHA-256
  `609486e4b022adddaf2428845a1b9608064e2251fc80dcd46626520d92b9d2c0`;
  the terminal success is 670 bytes/SHA-256
  `382ccd614a4dc48159ebdb801058f87c4d373c63f7054740e66e67c10a12d8f3`,
  status `bootstrap_completed_and_independently_verified`, with evidence root
  `928a9f1bb5c9031da3241f81356ff8c556433b77261e486666cbe7832d75f86e`.
  `bootstrap_stop_v1.json` is absent.
- The chain published and freshly verified the six-file immutable release at
  content root
  `3f5f0f417012ab2b5c291dc1fd322ba492a187fe295bfc4dc14954e296f24501`,
  records root
  `53af851fda4e519b353e7aa62ad5daa5053a183d03b0381a223b21bd554b3fae`,
  and release-record SHA-256
  `15a2e3bdb59e9da8fc631b246b132413704f688680cee7a153d741093baf039c`;
  the technical amendment SHA-256 is
  `ac793dac868c6677667aebbc1461d0600c9d05df7831d6bf3dcb73f32d75de4b`;
  and the closed one-use authorization SHA-256 is
  `1a9d229e483aef1f90912ea38de37baec190866df89a36fd54743272ad6c84fb`.
  The authorization permits only one closed attempt each for R3, I4, and U3.
- Two independent terminal audits reconstructed the three fresh-verifier
  records and the complete success evidence root without invoking a forbidden
  post-success verifier. The stable closed-tree snapshot was
  `62f7d424a4a8d5bce98cfab441d76cd1b0928a1ec7ed3e889d81a9562ed81ca6`:
  exactly 17 directories and 15 files, all hashes/ACLs/read-only and Q/I3
  cross-links valid, all 37 locks absent, zero protected-process matches, the
  claim PID gone, and R3/I4/U3/A3/S3/F3/Authority D absent.
- Bootstrap performed no Q write, Authority-D publication, outcome-value
  interpretation, training, confirmatory execution, or other scientific
  execution. The sole Q remains 21,274 bytes/SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`
  with status `qualified_rolled_back_failure_no_retry`; a second Q was not
  attempted. `SPEC.md`, frozen `PRE_REGISTRATION.md`, scientific configs, raw
  PanNuke, and existing runs were unchanged. Formal project status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9
  remains locked.

## Next exact action - qualify failed U2 as R3 once

After a fresh post-append public-Q verification and a final read-only
precondition snapshot, run exactly one released-controller R3 command from
working directory
`C:\Users\NATAN\AppData\Local\AANCA-control-plane\releases\3f5f0f417012ab2b5c291dc1fd322ba492a187fe295bfc4dc14954e296f24501`:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m controller.u3_successor_controller --qualify-failed-u2 --project-root C:\Users\NATAN\Documents\AANCA --parent-authority-dir C:\Users\NATAN\Documents\AANCA\artifacts\preregistration_amendments\20260727T170413.080954Z
```

Do not add `-I`, `-S`, or a direct script path. Any nonzero exit, drift,
partial state, R3 failure marker, or mismatched terminal readback is STOP
without retry. Only a qualified immutable R3 receipt may permit the separately
bounded I4 step.

## 2026-07-29 - The sole released R3 process stopped before its write boundary

- After the documented public-Q verification and two independent read-only
  precondition checks, executed the released R3 command exactly once from
  `C:\Users\NATAN\AppData\Local\AANCA-control-plane\releases\3f5f0f417012ab2b5c291dc1fd322ba492a187fe295bfc4dc14954e296f24501`:

```text
C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -B -m controller.u3_successor_controller --qualify-failed-u2 --project-root C:\Users\NATAN\Documents\AANCA --parent-authority-dir C:\Users\NATAN\Documents\AANCA\artifacts\preregistration_amendments\20260727T170413.080954Z
```

  The exact command-text SHA-256 was
  `cf7d183510210fc470e7e1d33f6235c58175b201a6ed867d257bad7a20348926`.
  The process exited 1 after 2.2 seconds. The external capture's 322-byte
  stdout has SHA-256
  `bfd49b3af042de7266a16042c50bf5c12edf572d26dd8c9e1e0bb6e46d7b82e9`;
  stderr was empty. The closed output reported
  `status=stopped_without_write`, `automatic_retry_allowed=false`, and all
  outcome-interpretation, scientific-execution, and publication flags false.
  Its `error_sha256` was
  `0392acec55bf2c078551f205c420a5586957014022354f75c007302841895a04`.
  This process capture is diagnostic evidence; it is not a governed R3
  terminal seal.
- Two independent static reconstructions map that hash uniquely to the
  93-byte message
  `ControlError: released authorization_helper semantic callable/default/closure surface changed`.
  In the immutable released controller, `main()` opens the verified external
  control plane before reaching the `--qualify-failed-u2` branch. The retained
  authorization helper is invoked through
  `_call_active_external_release_function`; its unconditional `finally`
  recaptures every released semantic surface, and
  `_require_released_module_semantic_surface_stable` raises at line 2692 when
  the recaptured semantic digest differs. The outer fail-closed handler at
  lines 10255-10273 emits the observed receipt and returns 1.
- This path precedes the first R3 claim/write boundary and the R3 branch
  itself. Current readback found no R3 receipt, no governed control-R3 attempt
  or failure seal, no I4/U3/A3/S3/F3 record, no Authority-D output, and no
  surviving relevant lock or process. The external stdout/stderr capture was
  not sealed into the governed control plane. Consequently the technical
  failure cause and pre-write ordering are proven, while the one-shot
  governance state is conservatively classified as an ambiguous consumed
  attempt. No training, confirmatory execution, outcome-value interpretation,
  or publication occurred.
- A production-shaped, read-only reproduction identified the exact
  implementation cause. Default marshal format v4 emits reference-sensitive
  `FLAG_REF` tags. The real authorization read retains an otherwise unchanged
  timestamp-format string in `_strptime._regex_cache` and temporarily creates
  a self-referential nested-closure cycle; those runtime references change the
  v4 marshal byte stream for `_timestamp_moment` and
  `_recursive_file_inventory` even though their public code fields and source
  are unchanged. `_timestamp_moment` has no nested `CodeType`, and changing
  only CPython adaptive bytecode did not reproduce the digest drift, so nested
  code quickening is specifically excluded as the root cause. The qualified
  candidate must structurally encode every public `CodeType` field and nested
  compiler constant without marshal reference tags, using exact IEEE-754 bits
  for float/complex values and fail-closing on unsupported constants.
- The sole command is consumed as an ambiguous fail-closed attempt. Do not
  rerun it,
  patch the immutable release, continue to I4/U3, repair/adopt state, or weaken
  the verifier. A separately versioned candidate may be built and tested only
  outside live/frozen state. Any future executable successor requires a new
  namespace/protocol, independent qualification, and new explicit external
  authorization; it cannot be represented as a retry, resume, or adoption of
  this chain.
- `SPEC.md`, frozen `PRE_REGISTRATION.md`, raw PanNuke, scientific
  configurations, existing runs, the immutable release, and the sole Q were
  unchanged. Formal project status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - qualify a separately versioned semantic-guard candidate

Complete the isolated implementation and require a production-shaped test that
loads all six real released modules, executes the authorization-helper path,
and proves that repeated before/after structural semantic captures remain
stable while real source, every public code-field family, binding, default,
kwdefault, closure, function/class surface, nested-code, exact numeric bits,
runtime, or module-identity changes still fail closed. Run its full isolated
pytest, Ruff check/format, strict mypy, and in-memory compile gates, freeze the
exact inventory, and have an independent verifier reproduce every gate. Do not
publish, authorize, invoke, or integrate the candidate into live state until a
new explicit versioned authority and one-shot release path have themselves
passed independent zero-write review.

## 2026-07-30 - Semantic-guard v6 rejected; protocol v4 frozen for static audit

- The fresh read-only semantic-guard specimen v6 at
  `C:\Users\NATAN\Documents\AANCA_r3_semantic_guard_successor_specimen_v6_20260730`
  contains exactly 13 files / 1,291,759 bytes. Its canonical 1,845-byte
  inventory root is
  `f38cfdfc11cd7c04a99f898743b829af3a7e3b0057d258b4299577522fb3223a`.
  Before and after the independent audit it retained all items read-only, with
  zero cache files/directories, reparse points, named ADS, `.pyc`, or `.pyo`.
- The preceding byte-equivalent working specimen passed 205 combined tests,
  including 15 focused startup/cache/handle tests. Ruff check, Ruff
  format-check, strict mypy, and in-memory compilation also passed. Those local
  gates do not override the independent static, no-import/no-execution verdict:
  **NOT QUALIFIED**.
- Two independent-audit P0 findings block v6:
  1. Python runtime/stdlib code, including `hashlib` and `pathlib`, executes
     before target verification while the interpreter, Python DLL, stdlib,
     `.pyd`, and dependent-DLL closure are not authenticated.
  2. The launch harness ignores the originally frozen manifest artifact and
     re-reads and re-pins the current manifest/controller paths, permitting a
     coherent pre-execution substitution without a SHA-256 collision.
- P1 findings cover incomplete timeout terminate/kill/WaitForExit handling,
  inheritable-handle mutation beginning outside protected cleanup, ordinary
  imports of the specimen controller/builder before the protected child,
  recorded launcher argv differing from the actual
  `-I -S -B -X ... -c` argv, noncanonical JSON and duplicate/extra fields,
  separate repeated manifest reads, only five of six roles statically pinned,
  and no exact process-image/file-ID or runtime-closure binding. P2 findings
  require explicit `optimize=0` and retain the Windows `ReadOnly` attribute
  only as auxiliary state evidence, not a security boundary.
- Preserve v6 unchanged as rejected read-only evidence. It authorizes no
  release, publication, amendment, Authority D, lifecycle, supervisor job, or
  scientific execution. A separately rooted v7 working copy is being built to
  close every P0/P1/P2; v5 and v6 remain unchanged.
- Protocol snapshot v4 is preserved at
  `C:\Users\NATAN\Documents\AANCA_semantic_guard_successor_bootstrap_staging_20260730_v4`.
  It contains exactly three files / 234,430 bytes. Its canonical 312-byte
  inventory root is
  `b95b2a6aa57bfa7e764e477a50e3b782579601f808090d628ff512d4259fb9d3`.
  Exact files are:
  - `DESIGN_V3.md`: 16,985 bytes,
    `984e771c02165da743d26d71f74712fbe09afd15f1235dc92c1ff865bb9c3cbd`;
  - `semantic_guard_successor_protocol_v3.py`: 147,383 bytes,
    `87a88b6814f73d0ba690acdd2a63ee1d122b2933bbef091f6b94af1c602f3812`;
  - `test_semantic_guard_successor_protocol_v3.py`: 70,062 bytes,
    `f3de4cde15ce4210d3624865beeddff617647811c36ae04129ee60dfc70ccf67`.
- The exact working bytes passed `78 passed in 43.47s`; the copied read-only
  snapshot passed the no-cache read-only verification with
  `78 passed in 44.02s`. Ruff check, Ruff format-check, strict mypy, and
  in-memory compilation passed. The snapshot has zero caches, reparse points,
  or non-default ADS. It remains unconditionally
  `EXTERNAL_AUTHORITY_REQUIRED`; its independent static audit is still
  pending, and these gates authorize no state-changing operation.
- No live/frozen project source, raw PanNuke file, existing run, scientific
  authority, or outcome was changed. `SPEC.md` and frozen
  `PRE_REGISTRATION.md` remain unchanged. Formal project status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - independently qualify v7 and protocol v4

Complete v7 in a separate working root with immutable original manifest bytes,
strict canonical parsing, six-of-six static role binding, actual
argv/environment/bootstrap/handle receipts, bounded
terminate/kill/WaitForExit cleanup, protected inheritable-handle lifecycle,
explicit `optimize=0`, and an authenticated or narrowly and honestly trusted
runtime boundary. Freeze a fresh read-only v7 snapshot and audit it
independently.

In parallel, finish the independent static audit of exact protocol-v4 root
`b95b2a6aa57bfa7e764e477a50e3b782579601f808090d628ff512d4259fb9d3`.
Do not execute or mutate that snapshot further. Only after both independent
verdicts pass may work proceed to a distinct live launcher/wrapper, publisher,
fresh verifier, OS-derived lock/process evidence, content-aware terminal
readback, exact plan/argv/runtime binding, and a new trusted-host external
authority.

## 2026-07-30 - Protocol v4 rejected for promotion; public Q remains exact

- The independent static-only audit reconfirmed protocol-v4 root
  `b95b2a6aa57bfa7e764e477a50e3b782579601f808090d628ff512d4259fb9d3`,
  exact size/inventory, read-only state, and absence of caches, reparse points,
  and non-default ADS. It found no P0 or P1 route by which the current snapshot
  could become ready: trusted-host binding is hard-coded `None`, all nine
  runtime blockers are unconditional, production readiness and publication
  are hard-false, and the module exposes no filesystem or subprocess action.
- The same audit nevertheless returned **NOT QUALIFIED for future promotion**.
  Four P1 schema gaps must be corrected in a separately rooted protocol v5:
  raw candidate-QA evidence must contain reconstructible structural
  before/after/error bytes rather than only claimed roots; tree inventories
  must require the `.` record and complete parent closure and reject empty
  inventories; critical programs must have unique logic roots and module
  names/paths; and raw monitor evidence must bind its claimed monotonic
  interval to the observations. One P2 path gap requires rejecting Win32
  superscript device aliases `COM¹`/`COM²`/`COM³` and
  `LPT¹`/`LPT²`/`LPT³`.
- Preserve protocol v4 unchanged as safe-blocked negative audit evidence. A new
  protocol-v5 working root was created from its exact bytes and is limited to
  those five audit findings; all nine unconditional runtime blockers and
  `EXTERNAL_AUTHORITY_REQUIRED` remain mandatory.
- After the preceding documentation append, the first read-only public-Q
  wrapper successfully returned from
  `verify_resource_bounded_replacement_terminal_qualification_receipt` and
  printed `Q_POST_V6_V4_DOC_APPEND_PASS`, but the diagnostic wrapper then
  exited 1 because it incorrectly attempted a nonexistent top-level
  `value["status"]`. A corrected fresh-process wrapper exited 0, printed the
  same marker, returned the expected verifier dictionary keys, and read
  exactly 21,274 bytes. Independent file readback remained SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  This was read-only verification after the verifier had already succeeded;
  no Q write or second replacement attempt occurred.

## Next exact action - qualify semantic guard v7 and protocol v5

Complete the two separately rooted working candidates, run their bounded
synthetic QA gates, freeze fresh read-only cache-free snapshots, and obtain
independent static verdicts. Do not treat the rejected v6/v4 snapshots or the
existing broad user permission as live authority. The distinct live
launcher/publisher/verifier layer remains prohibited until both components
qualify and an exact plan is independently frozen.

## 2026-07-30 - Protocol v5 frozen and conservatively rejected after split audit

- Protocol v5 is preserved read-only at
  `C:\Users\NATAN\Documents\AANCA_semantic_guard_successor_bootstrap_staging_20260730_v5`.
  It contains exactly three files / 254,238 bytes, with a 312-byte canonical
  inventory root
  `85bd6018030513daa83215fb44a827a60d1070b2894708c69bbd672e44cfbaab`.
  Root and files are read-only; caches, nested directories, reparse points,
  and non-default ADS are absent.
- Exact files are:
  - `DESIGN_V3.md`: 19,316 bytes,
    `af8b6fc97124ad47b209ed65bca900d341694554dafacd940defffb827c7fc70`;
  - `semantic_guard_successor_protocol_v3.py`: 155,392 bytes,
    `082ed8578bab0aa16a2a7beb7584d885ebf47c518dd60e9df25f3e053a30943b`;
  - `test_semantic_guard_successor_protocol_v3.py`: 79,530 bytes,
    `9d832aca4b9d14dd555c5ed18c2cb13d98b74cc10612676abb5edf54c12c798f`.
- Final exact working bytes passed `104 passed in 64.39s`, Ruff check, Ruff
  format-check, strict mypy for both Python files, and in-memory compilation.
  Source-to-snapshot size/hash equality was exact.
- Two independent static audits agreed that all five v4 findings were repaired,
  readiness remained hard-false, all nine runtime blockers remained
  unconditional, and no current live/publication path existed. They disagreed
  on one future-promotion P1:
  - one auditor accepted completeness generation as the responsibility of the
    exact future QA verifier and OS wrapper;
  - the stricter auditor proved that `_canonical_structural_records` itself
    accepted any nonempty sorted caller-supplied subset and therefore derived
    only the integrity of that subset, not completeness under
    `deterministic_structural_code_record_v2`. It also did not require a
    mutation-class-relevant changed path or bind the retained capture document
    to exact verifier stdout.
- Apply the conservative verdict: **NOT QUALIFIED for promotion**. Preserve v5
  unchanged. A separately rooted protocol v6 must require one complete
  canonical semantic-surface object with exact top-level maps `module`,
  `classes`, `class_attributes`, and `functions`; bind the exact bytes to the
  verifier process output; retain identical before/after key universes; and
  enforce a closed mutation-class-to-relevant-changed-path policy.
- Formal project status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  open at **8/10 = 80%**, and M9 remains locked. No project/live/raw/run,
  scientific, or outcome state changed.

## 2026-07-30 - Protocol v6 frozen and rejected by two independent audits

- Protocol v6 is preserved unchanged and read-only at
  `C:\Users\NATAN\Documents\AANCA_semantic_guard_successor_bootstrap_staging_20260730_v6`.
  Its canonical 312-byte ordinal-name inventory covers exactly three files /
  275,561 bytes and has root
  `b4a0f48fff4867cb6e8d69f7fbf287659d9f863ec6fe69cee64846aed95f26a8`.
  Exact files are:
  - `DESIGN_V3.md`: 21,778 bytes,
    `223745f8e37a0ecddf1fe4552778f1e1b2803c408845059184e208102c57936e`;
  - `semantic_guard_successor_protocol_v3.py`: 163,988 bytes,
    `5c5fc155a77332d7d611bc611b5abbe5811c89d41d65b2c0d657d6ad5b8012d1`;
  - `test_semantic_guard_successor_protocol_v3.py`: 89,795 bytes,
    `4b62edd65a9312f67c92e8f8713ff8ffe31e31d6bf5ef1b2a1598bc78f4e89c8`.
  The root and files are read-only, with zero nested files, cache objects,
  reparse points, or non-default ADS.
- The exact pre-freeze working bytes passed `114 passed` in the final focused
  run, Ruff check, Ruff format-check, strict mypy for both Python files, and
  in-memory compilation. Source-to-snapshot size/hash equality passed. These
  local gates do not override the independent audit verdict.
- Two independent static-only audits reconstructed the exact frozen root and
  agreed on **P0=0, P1=2** and **NOT QUALIFIED for promotion**:
  1. the canonical four-map semantic surface and its stdout bytes are not
     provenance-linked to the exact candidate-role file readbacks, so an
     unrelated but internally coherent surface can be attributed to the
     candidate;
  2. function-related mutation classes still share a section-wide
     `/functions` path rule, so a coherently rehashed wrong leaf within that
     section can be mislabeled as the declared mutation.
  The existing tests covered a missing top-level surface and a wrong top-level
  section, but not coherent whole-surface substitution or a wrong leaf inside
  the permitted section.
- Preserve v6 as safe-blocked negative evidence. A separately rooted protocol
  v7 working copy is limited to a closed canonical QA stdout document with
  exact ordered six-role file readbacks and semantic surfaces, exact ordered
  mutation observations, and mutation-class-specific leaf predicates. All
  nine runtime blockers, hard-false readiness, and
  `EXTERNAL_AUTHORITY_REQUIRED` remain mandatory.
- No scientific command, training, publication, authority write, project run,
  raw PanNuke change, or outcome read occurred. `SPEC.md` and frozen
  `PRE_REGISTRATION.md` remain unchanged. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - independently audit semantic guard v7 and test protocol v7

Complete the static-only independent audit of exact semantic-guard v7 root
`abda274e236ee83093d7bdb4716a3aefe8bd4de72b00bac4d0286cdab1452326`.
In parallel, implement the two bounded protocol-v7 repairs and run their focused
adversarial gates. Do not publish, integrate, create a trusted event, or execute
science. The distinct runtime remains independently blocked until its own
Windows path/process/lifecycle audit passes.

## 2026-07-30 - Reject offline runtime v1; bound protocol-v7 resource failure

- The independent static-only audit of the post-fix offline runtime-v1 working
  bytes reconstructed the same 11-file metadata inventory twice. Its 1,355-byte
  `name|size|mtime_utc|sha256` preimage has aggregate
  `a85ff808b36d1fa8c3ba97e119d7a009383e051d68f08588c7f1bed578ff7f08`.
  Core records were:
  - contract: 25,103 bytes,
    `a47ea3db4f1a8b0751a8f2cc194674b5cd9709a2ea0858b09e22b67d223ac44c`;
  - lifecycle: 42,339 bytes,
    `eddbf6006f44e1530447f24280e4100e6f9d311baa4a5577a256d37e59573857`;
  - Windows evidence: 36,964 bytes,
    `31b9838e0af52879fcbc5674773a0d1faedcb0b9d6e9d3c956afb37d060f4029`;
  - synthetic builder: 6,231 bytes,
    `c9bbe4e0b09b01dc42e96f5114436ab806b87171d0fb06d546816615acec9812`.
- The verdict is **NOT QUALIFIED**. Two P0 defects remained: a write-sharing
  window in the directory guard allowed an in-place reparse substitution before
  a later detection, and publisher/verifier source was hashed then closed before
  `Popen`, permitting source/bootstrap/cache TOCTOU before terminal STOP.
  P1 findings covered non-exact terminal topology and incomplete success
  cross-hashes; plan-local rather than global singleton/attempt consumption;
  non-handle-bound concurrent tree/ADS scans; post-hoc unbounded
  `communicate()` capture and timeout drain; and incomplete OS-derived
  process/runtime/source binding. P2 findings covered pre-move source topology,
  coercive bool/int/float equality in later records, nonempty-stderr acceptance,
  and weak timestamp/version validation.
- Preserve runtime v1 unchanged as rejected working evidence. Runtime v2 is a
  separately rooted synthetic-only copy. Its first 37 contract, 10 Windows, and
  20 lifecycle component tests passed individually, but nominal success was
  deliberately hard-blocked at the verified pending bundle. No combined or
  qualification claim is made.
- During one handle-relative rename experiment, a `NULL RootDirectory` test
  resolved a 19-byte synthetic `published_bundle\payload.json` into the
  runtime-v2 working directory rather than its intended temporary jail. The
  owner stopped immediately, verified the exact path and sole payload, removed
  only that file and its now-empty directory after an absolute-parent check,
  and rejected that primitive. A fresh recursive read-only search found no
  `published_bundle`, `published_bundle.pending`, or `payload.json` below either
  runtime working root or the AANCA project. No project, control-plane, raw
  PanNuke, run, authority, or scientific file was touched.
- A protocol-v7 positive synthetic pytest was interrupted by the integration
  owner, not allowed to reach a terminal test result. The fixture embedded the
  same full canonical stdout inside 18 raw observations, grew to about 37.4 GiB
  private memory, and left about 0.36 GiB free physical RAM. After exact
  command-line verification, only pytest PIDs 11384 and 25364 were terminated;
  free physical memory recovered to about 23.69 GiB. Record this run only as
  `interrupted_due_bounded_resource_guard`, never as PASS or FAIL.
- Protocol v7 now retains one canonical outer stdout document and uses exact
  size/SHA-256 crosslinks from captures and mutations instead of duplicating
  raw stdout. It must also use class-specific real mutation witnesses:
  source-byte envelopes, module-object identity, callable-binding inventory,
  and mirrored semantic paths for defaults/keyword defaults/closure/code/runtime.
  A bounded canary must pass before another full protocol suite.
- The post-protocol-v6 documentation public-Q verifier exited 0 with marker
  `Q_POST_PROTOCOL_V6_REJECT_DOC_APPEND_PASS`; it re-read exactly 21,274 bytes
  and exact SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  This was read-only continuity evidence, not a Q write or retry.
- A separate read-only supervisor check reconfirmed release SHA-256
  `75b91e95fe253b8e5fe42e8488d41fa8fd7677891a82de1aeaeaad928e9031d8`,
  manifest SHA-256
  `016739b52c5aa916ba4ad9f171d7a5af45d1a73d75f2a870a89e06c78c19a192`,
  handoff SHA-256
  `9c9b54f14191d75ea71d002e699fc4451905038eb4c3d3c11ec4246d662af23b`,
  and exact session `019faaf3-c547-79e1-b0eb-26e35d214642`. `jobs/` and
  `recovery_stops/` were absent and no Python supervisor process was active.
  The supervisor remains qualified but unarmed.
- No training, primary, confirmatory, recovery, publication, or scientific
  process was started. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish immutable v7 audits and bounded v2 repairs

Finish both independent static audits of the exact semantic-guard-v7 snapshot.
Run only bounded protocol-v7 canaries until its canonical evidence size and
memory use are proven, then execute the focused adversarial gates. In parallel,
close all runtime-v1 P0/P1/P2 items in runtime v2 under an explicit trusted-host
boundary, with held source bytes/private bootstrap, exact terminal topology and
readback, global singleton/attempt consumption, and bounded child output. Do not
freeze, publish, create authority, arm the supervisor, or execute science.

## 2026-07-30 - Semantic guard v7 frozen and independently rejected

- The byte-equivalent v7 working allowlist passed:
  - `53 passed` focused in 153.24 seconds;
  - `173 passed` inherited controller tests in 4.80 seconds;
  - one cache-free combined invocation with `226 passed` in 159.78 seconds;
  - Ruff check, target/test format-check, strict mypy, a read-only compile of
    12 Python sources, the exact bootstrap pin, and the exact 48,836-byte
    target-source pin.
  Whole-specimen format-check would reformat seven inherited frozen upstream
  files; they were not changed because doing so would invalidate the six exact
  role pins.
- The fresh, never-imported and never-executed snapshot is preserved read-only
  at
  `C:\Users\NATAN\Documents\AANCA_r3_semantic_guard_successor_specimen_v7_20260730`.
  Two independent auditors reconstructed the same compact 1,845-byte canonical
  JSON inventory: exactly 13 files / 1,352,842 bytes, root
  `abda274e236ee83093d7bdb4716a3aefe8bd4de72b00bac4d0286cdab1452326`.
  All eight directories and 13 files were read-only, every file had one link,
  and caches, `.pyc`/`.pyo`, reparse points, and non-default ADS were absent.
- Both static-only audits returned **NOT QUALIFIED / STOP**. Under the declared
  trusted-host boundary, one audit classified P0=0/P1=5/P2=3; the process-tree
  auditor classified the generic lifecycle escape as P0. The common blocking
  facts are:
  - protected handles are made inheritable, but timeout/reset/final cleanup
    controls only the direct child; no Job Object proves the process tree empty,
    and later `communicate()`/`wait()` paths are unbounded;
  - output is buffered without a hard cap, and a descendant retaining a pipe
    or protected handle can prevent terminal cleanup;
  - `timeout_policy` is present but its exact schema, finite timeout,
    `automatic_retry_allowed=false`, terminal action, effective handle list,
    and job/tree policy are not verified by the child;
  - the effective pre-verification import closure is incomplete. The target
    imports mutable stdlib modules before its runtime check, the expected
    `sys.path` begins with a nonexistent `python312.zip`, and the fingerprint
    does not cover every importable ZIP/PYD/PYC/config alternative;
  - the manifest retains its original content record but not creation-time
    file identity, so a byte-identical pre-open replacement is not
    distinguishable;
  - the outer harness itself uses ordinary imports before creating the
    protected child.
  The local `_PROCESS_LAUNCH_LOCK` also cannot serialize unrelated native
  process creation during the inheritable-handle window.
- The auditors confirmed that the semantic core repairs are real: the
  site-free bootstrap imports only builtin `sys` and compiles private target
  bytes with `optimize=0`; actual argv/environment/cwd/interpreter are checked;
  strict JSON rejects duplicate/nonfinite/noncanonical input; the original
  manifest content, retained controller handle, and hardcoded 6/6 role hashes
  block coherent manifest/controller content substitution.
- Preserve v7 unchanged as rejected evidence. A separately rooted semantic
  guard v8 working copy is limited to: sanitized and bound `sys.path`; a
  retained exact pre-verification import closure; creation-time manifest
  identity; source-only outer bootstrap; `CREATE_SUSPENDED` plus
  `KILL_ON_JOB_CLOSE` Job Object; bounded input/output/drain; exact typed
  timeout/no-retry/job/handle receipt; and grandchild, output-flood,
  persistent-reset, concurrent-spawn, identity-replacement, and nonfinite
  timeout tests.
- The post-runtime-v1 documentation public-Q verifier exited 0 with marker
  `Q_POST_RUNTIME_V1_REJECT_DOC_APPEND_PASS`, exact size 21,274 bytes and exact
  SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  No Q write or retry occurred.
- No live/project/raw/run/authority/scientific state changed. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.

## Next exact action - qualify v8, protocol v7, and runtime v2 independently

Run bounded focused tests for the three separately rooted working components,
then freeze fresh cache-free read-only snapshots only after their local gates
are green. Each snapshot requires a new static-only independent audit. No
component may be integrated or promoted while any P0/P1 remains, and none of
these test results authorize a live write, trusted event, supervisor job, or
scientific execution.

## 2026-07-30 - Remaining-project execution order amended toward completion

- A read-only PLAN/code reconciliation proved that executing
  `resource_bounded_confirmatory_v1` next would cost approximately 11-16 hours
  yet remain `completion_stage=null`; it cannot close M8 or unlock M9.
- `PLAN.md` now contains a dated append-only operational amendment: preserve
  the resource-bounded path as optional `amended_or_exploratory` evidence, but
  use the qualified technical chain for the unchanged original 108-cell
  confirmatory study with explicit checkpoint-successor resume and the
  event-driven supervisor.
- Frozen science remains unchanged. The current config still defines exactly
  108 cells, 90 required and 18 optional pathology cells, including 36 CNN
  cells and 180 five-fold CNN fits. The recorded lower bound is about 5.5 days,
  the historical estimate is 10-15 days, and active single-copy checkpoints
  are about 30 GiB. Fresh sealed capacity plus a 10-GiB margin remains mandatory.
- An isolated external implementation task was started for the resume layer;
  it may copy and modify only a new working root, never the live repository,
  configs, data, runs, authority, or supervisor. No confirmatory or other
  scientific command was started.
- The amended order does not grant a real attempt. Exact source/plan/argv/runtime
  and supervisor hashes do not yet exist, so a future one-use external
  authority remains mandatory. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish technical qualification and isolated resume QA

Continue the bounded v8/protocol-v7/runtime-v2 gates and the isolated original
confirmatory resume implementation in parallel. Do not integrate the resume
allowlist until its focused interruption/corruption/link tests and independent
review pass. Do not execute resource-bounded or original confirmatory science.

## 2026-07-30 - Runtime-v2 rejected before promotion; protocol-v7 size gate stopped broad QA

- The runtime-v2 pre-audit working aggregate was
  `2cc1a3767fc4524e0e83b2b0c9071dcc14ba7c131dcdaacdd2cc39efc71a4480`
  over exactly eight intended files. On those unchanged bytes, the bounded
  component suite passed **75/75** and `ruff check --no-cache .` passed.
  `ruff format --check --no-cache .` correctly reported that
  `runtime_lifecycle_v1.py` still required formatting, so this aggregate was
  never a promotion candidate.
- The independent static-only audit returned **NOT QUALIFIED**. Its material
  blockers were:
  - a partial failure while making the child handle set inheritable can leave
    earlier protected handles inheritable, and rollback/readback is not total;
  - `TerminateJobObject` failure can suppress later whole-job cleanup, while
    the fallback controls only the direct PID and can release protected source
    handles before `ActiveProcesses=0` is proved;
  - the child does not independently attest `sys.orig_argv`, `optimize=0`, the
    complete effective handle inventory, queried Job identity/limits, or typed
    end-to-end deadlines;
  - parent held-source records are not fully cross-linked to the exact program
    records executed from the private child bundle;
  - the intentional atomic-commit hard block proves only the pending directory
    identity, not the unchanged complete pending tree or a persisted,
    cross-checked `STOP_AMBIGUOUS` receipt.
- Further P2 hardening is required for finite/bounded job waits, bounded
  no-follow tree depth/object/byte counts, multilink rejection, and mutex
  owner/DACL evidence. The owner is applying one consolidated fix plus
  mechanical formatting; all no-cache gates and a new independent audit must
  bind the resulting new bytes. The current aggregate remains rejected.
- Protocol-v7 collection found **176 tests**. Its guarded smoke passed
  **3 tests** with peak process-tree private memory 48,902,144 bytes, and the
  guarded semantic subset passed **56 tests** with 120 deselected and peak
  36,741,120 bytes. The first bounded full-chain size test produced a real
  assertion failure, not a resource interruption: plan 28,326 bytes, QA
  802,410 bytes, manifest 1,337,620 bytes, each verification approximately
  5.87 MB, success 13,332,286 bytes, and terminal committed observation
  41,593,333 bytes. Because that terminal document exceeds the fixed 16-MiB
  limit and the production graph did not yet resolve the QA content reference,
  the full suite was deliberately not run. Refactoring to single-copy
  content-addressed records continues under the same bounded gates.
- The isolated original-confirmatory successor design now derives a closed
  180-checkpoint expectation set from the complete 36-CNN-cell by five-fold
  plan, including explicit `missing_fresh` decisions and per-fold
  `model_seed + fold_id` configuration. It remains external working evidence;
  predecessor terminal/orphan qualification and its first focused gate are
  still in progress, and nothing has been integrated.
- A read-only host check reported 156.09 GiB free on C:, 22.54 GiB free RAM,
  and an idle RTX 4070 with 11,642 MiB free GPU memory. No Python training,
  primary, confirmatory, recovery, publication, or supervisor process was
  active. No live project source, raw PanNuke data, run, authority, or frozen
  file changed.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - repair and re-audit the three isolated technical components

Complete the runtime-v2 handle/job/envelope repairs and protocol-v7
content-reference refactor, then run their bounded no-cache gates and obtain
fresh independent hash-bound audits. In parallel, finish the closed
original-confirmatory predecessor/resume contract and its focused synthetic
fault tests. Do not integrate, publish, arm the supervisor, create authority,
or execute scientific training until every P0/P1 and mandatory gate passes.

## 2026-07-30 - Second qualification checkpoint: v8 and runtime-v2 rejected; original-resume candidate frozen

- A fresh read-only process inventory found no Python training, primary,
  confirmatory, recovery, publication, or supervisor job. The only relevant
  long-lived application process was the Codex app and its command-safety
  helper. No scientific command was started or restarted.
- The installed event-driven supervisor remains qualified but deliberately
  unarmed. Its exact release is 218,146 bytes with SHA-256
  `75b91e95fe253b8e5fe42e8488d41fa8fd7677891a82de1aeaeaad928e9031d8`;
  its manifest is 10,081 bytes with SHA-256
  `016739b52c5aa916ba4ad9f171d7a5af45d1a73d75f2a870a89e06c78c19a192`;
  and its saved handoff is 2,402 bytes with SHA-256
  `9c9b54f14191d75ea71d002e699fc4451905038eb4c3d3c11ec4246d662af23b`.
  It remains bound to exact Codex session
  `019faaf3-c547-79e1-b0eb-26e35d214642`. Its `jobs`, `processes`, and
  recovery-STOP namespaces are empty.
- Preserve semantic-guard-v8 root
  `6fbd4e561ecdc279c10603e24cf70b5bab32c40b8c8ca958f652f41892ad31d1`
  unchanged as rejected qualification evidence. Its independent static review
  returned **NOT QUALIFIED**: source preverification did not hold the complete
  transitive import closure; failure cleanup could release protected sources
  without proving the Job empty; the policy was self-declared rather than an
  executable wrapper contract; and lifetime singleton, manifest creation
  identity, absolute-deadline, handle-table, Job-query, rollback, and partial
  write guarantees remained incomplete. A separately rooted v9 is in bounded
  implementation and has not yet passed tests or review.
- Preserve runtime-v2 aggregate
  `c6751f8a1c4f1d438f610ce53e460b9ae4d38a6b5cb643e4550e8cd34a7ee185`
  unchanged as rejected qualification evidence. Its component suite passed
  **87/87**, its focused subsets passed, and Ruff check/format passed, but
  configured mypy reported 67 errors and strict mypy reported 71. Its fresh
  static review returned **NOT QUALIFIED** because the initial launch was not
  yet a source-only literal loader, child I/O and cleanup lacked one absolute
  deadline, close success-then-raise lacked independent closed-handle proof,
  persisted STOP evidence was incomplete, DACL policy was opaque, the
  publisher revalidated only a partial attempt before mutation, and the outer
  environment hash was not cross-linked. A distinct runtime-v3 is now the only
  permitted repair target.
- Protocol-v7 root
  `9173beb1c0b355d398fcb09400593e5bf046bbd0ddb374b645fd916551f0ec97`
  was superseded before audit after a hidden second canonicalization was found.
  The current cache-free three-file candidate is
  `d078badcc1c9e78054a8f54489734ee745d42e748b9b99bded3e260534e9c768`.
  On those candidate bytes the focused **17/17** gate, production-shape memory
  gate, full **194/194** gate, Ruff check, and Ruff format check passed. The
  observed production process-tree private memory was 136,740,864 bytes and
  the full-suite peak was 148,287,488 bytes, below the 512-MiB ceiling. A fresh
  exact-root static audit is still running; these results do not yet qualify
  promotion.
- The isolated original-confirmatory resume candidate was frozen read-only at
  root
  `0a0a40250143aee0e6fb4dc0ff20b76985d0fae5c2b3b6e59068518993f05979`.
  It contains exactly two files and no cache: source SHA-256
  `11cb625fae879b81f3fd2b79fc01ab6cd74a5eceba22dcc8a6152a384e5c71e6`
  and test SHA-256
  `8d8b98935d6f0c5d1ded5785473230537551c4729c5e7c6249a9e2563b58df97`.
  Its **32/32** focused tests, Ruff checks, and strict source mypy passed. It
  defines the closed 180-checkpoint expectation set, explicit predecessor
  classes, physical no-overwrite copies, per-fold no-fallback directives,
  one-use execution authority, static-environment plus dynamic supervisor
  nonce binding, and no automatic retry. It remains external and requires a
  fresh independent compatibility review before any integration.
- Read-only live-code analysis proved that the historical single-authority
  confirmatory gate cannot legally bind a new technical source while also
  preserving the sealed primary source. The repair is a dual gate: immutable
  historical-primary dependency evidence remains bound to preregistration
  root
  `4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a`,
  while a new outcome-blind original-confirmatory technical authority binds
  only the current execution source and unchanged frozen science. This
  implementation is still isolated and unqualified.
- The user authorized exactly one future write of `Q replacement-v2` and one
  independent verification. This authorization is **not consumed**. The
  historical public Q is frozen evidence and will not be rewritten, retried,
  or reinterpreted. The authorized replacement may be created only after the
  exact source, plan, literal argv/orig-argv/command line, runtime bundle,
  static environment, dynamic supervisor nonce, supervisor release/session,
  terminal contract, checkpoint allowlist, and independent review are all
  frozen and hash-bound.
- The last read-only public-Q verification after the preceding documentation
  append exited 0 with marker
  `Q_POST_RUNTIME_V2_PROTOCOL_V7_AUDIT_DOC_APPEND_PASS`, exact size 21,274
  bytes, and SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  No Q write or retry occurred.
- The immediate read-only verification after this qualification checkpoint
  also exited 0 with marker
  `Q_POST_V8_RUNTIME_V2_RESUME_DOC_APPEND_PASS`, the same exact size 21,274
  bytes, the same exact SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the expected three-key canonical lineage envelope. This was verification
  only; no authority was written or consumed.
- No live source, raw PanNuke data, run, frozen scientific file, registry,
  authority, or supervisor state changed in this checkpoint. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.

## Next exact action - finish independent qualification before consuming Q

Obtain exact-root PASS reviews for protocol-v7 and the frozen resume layer;
finish and independently qualify semantic-guard-v9, runtime-v3, the dual
historical/current authority gate, and the outcome-blind terminal receipt.
Only then integrate their frozen allowlists, run the full repository,
lifecycle, CLI, and PanNuke gates, freeze the exact execution envelope, consume
the single authorized Q replacement-v2 plus its independent verification, and
arm exactly one no-retry original-confirmatory supervisor job.

## 2026-07-30 - Fresh-mode correction and three fail-closed Q blockers

- Read-only reconciliation with `PLAN.md` and the host filesystem proved that
  the original 108-cell confirmatory study has never run and no real
  confirmatory fold checkpoint exists. The only `.pt` file under the live
  project is a lifecycle-rehearsal fixture. Therefore the next legal real
  attempt is explicitly `fresh`, with `retry_of_run_id=null`, zero predecessor
  reads/copies, and 180 predeclared fresh fold directives. A
  `successor_resume` remains a separate future mode requiring an actual
  predecessor and new one-use authority; a fictitious predecessor is
  forbidden.
- Independent review rejected dual-authority frozen root
  `87503f863626dfebdbb5f08e1fe9a6e1a9ec8d1b3437cff3e8df7848b6ae61ea`
  with **2 P0, 3 P1, and 1 P2**. The seven-file, 773,501-byte read-only
  inventory remained exact on three reads, so the verdict concerns semantics,
  not mutation:
  - `reviewed_intent_sha256` is syntax-checked but never compared with a
    recomputed canonical pre-Q intent, and the receipt file is hashed but not
    parsed/cross-linked;
  - lifecycle and the execution gate can qualify Q without receiving or
    canonicalising one-use E;
  - even its unused E canonicaliser does not bind the exact Q
    directory/root/manifest or the complete program/source bundle;
  - malformed, partial, or linked Q namespace entries are ignored rather than
    consuming the one-shot attempt and producing STOP;
  - source, config, and PLAN are not recaptured in the final gate readback.
- A separate read-only preflight found a third Q-publication P0: the candidate
  publisher has no durable pre-publication one-use attempt/tombstone. A
  post-check failure deletes owned Q bytes, and a crash-left unparseable
  partial can be skipped by the uniqueness scan, permitting another call.
  The repaired publisher must serialize one immutable attempt against P before
  the first Q byte; failed or ambiguous publication permanently consumes that
  attempt and records STOP. No automatic or manual retry may occur under the
  same authorization.
- The same preflight independently confirmed the only valid lineage:
  base freeze -> A -> historical recovery authority P
  (`4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a`)
  -> future original-confirmatory Q. Resource C/D are sibling branches and
  cannot parent, replace, or be reinterpreted as this Q. Historical primary run
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery` remains the exact
  sealed dependency with 185/185 required cells and zero retraining.
- The frozen terminal-receipt-v1 candidate was independently rejected with
  **2 P0, 2 P1, and 1 P2**, while both exact file hashes and physical
  read-only/single-link state remained stable:
  - it requires a non-null string `retry_of_run_id` and cannot represent the
    required fresh attempt;
  - external pins omit matrix, statistics, restoration, registry, stage, and
    terminal time, so a self-consistent false receipt can change those
    sections under unchanged pins;
  - its claimed two readbacks and verifier separation are declarations because
    no independent reader opens the referenced terminal artifacts;
  - Windows ancestor-reparse, ADS/device-name, dot-segment, immutable-state,
    and no-follow protections are incomplete;
  - it still names rejected runtime-v2 rather than runtime-v3, and its reported
    aggregate-root serialization was not reproducible from the supplied
    contract.
- Terminal-receipt-v2 work is isolated and must add a closed
  `fresh | successor_resume` union plus an independently frozen,
  outcome-blind artifact reader that performs stable no-follow readbacks of
  the actual seal, manifest, status, completion, integrity, registry, stage,
  matrix, statistics, restoration, and terminal artifacts. Every material
  section must be externally pinned; pins may not be derived from the receipt
  under validation.
- Protocol-v7 root
  `d078badcc1c9e78054a8f54489734ee745d42e748b9b99bded3e260534e9c768`
  was also rejected before freeze despite **194/194** local tests. The success
  graph checked individual `link_count=1` but did not prove identity
  uniqueness/disjointness across bundle files, manifest, claim, root, and
  directories. Coherent aliases could therefore satisfy size/SHA checks.
  A separately rooted correction must add complete file/directory identity
  uniqueness and terminal-manifest identity before a new audit.
- Preserve all rejected frozen candidates unchanged. New work is limited to
  isolated dual-authority-v2, terminal-receipt-v2, protocol successor,
  semantic-guard-v9, runtime-v3, and a separate fresh/successor data-plane
  runner core. No live source, raw PanNuke data, run, registry, frozen
  scientific file, authority, or supervisor state changed. No real scientific
  process was started.
- The user's authorization for exactly one future Q replacement-v2 write and
  one independent verification remains **unconsumed**. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.
- The immediate read-only historical public-Q verification after this
  documentation append exited 0 with marker
  `Q_POST_FRESH_MODE_P0_DOC_APPEND_PASS`, exact size 21,274 bytes, expected
  three-key canonical lineage envelope, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  No authority was written, retried, or consumed.

## Next exact action - qualify repaired v2/v3 components and the fresh runner

Complete focused adversarial gates for dual-authority-v2, terminal-receipt-v2,
protocol successor, semantic-guard-v9, runtime-v3, the frozen
successor-resume module, and the isolated fresh/successor runner core. Freeze
and independently audit each exact allowlist. Do not integrate, publish Q,
create E, arm the supervisor, or start confirmatory science while any P0/P1
remains.

## 2026-07-30 - Resume-v1 rejected; repaired components reach first green gates

- Independent static/live-compatibility review rejected the frozen
  successor-resume-v1 candidate while reconfirming its exact two-file,
  199,094-byte read-only inventory. Four P0 defects require a separately
  rooted v2:
  - orphan process evidence is not lineage-bound: the authority omits the
    predecessor PID/create-time and tests can qualify with an unrelated process
    that was merely spawned and killed;
  - supervisor/child/one-use provenance is incomplete: supervisor PID may be
    absent, current child PID/create-time/parent are not checked, the nonce is
    not durably consumed, and an arbitrary authority PID can be excluded from
    the live-process scan;
  - the caller can replace an imported resume/terminal directive with `fresh`,
    delete the checkpoint, recompute the directive-list hash, and recreate the
    forbidden existence-based fallback;
  - the Q adapter rehashes caller-supplied pins without deriving the Q/P
    semantic binding from a separately verified replacement-v2 receipt.
- Resume-v2 work is now isolated in a new root. It must bind exact predecessor
  launch/status/supervisor/process identity, consume one nonce/authorization
  before any RunTracker write, derive every directive from immutable
  snapshot/copy evidence, and accept only typed verified Q/E evidence. The
  frozen v1 root remains unchanged. The next real attempt is still `fresh`;
  resume-v2 is for a separately authorized future successor only.
- A read-only integration trace separately confirmed the live fail-open path:
  `image_oof.py` derives `fold_resume` from current file existence,
  `confirmatory_core.py` exposes only one global resume boolean, and its
  per-cell exception handler can demote a structural checkpoint violation to
  an ordinary cell failure. The isolated runner core now replaces this with a
  closed immutable 180-directive contract and a fatal structural exception.
  Its first migrated image/core/indexed suite reached **27/27 passed**; dedicated
  fresh/incomplete/terminal and full-180 tests remain in progress.
- Dual-authority-v2 reached **15/15 focused tests** for the canonical pre-Q
  intent, parsed independent review receipt, closed fresh/resume E, 180
  directive binding, stable no-follow E readback, and durable Q-attempt
  consumption. Physical crash/concurrency tests and production
  study/lifecycle E wiring remain open; no completion stage may be claimed by
  pre-execution readiness.
- Terminal-receipt-v2 reached **32/32 focused tests**. It uses a closed
  `fresh | successor_resume` union, runtime-v3 binding, section-by-section plus
  full canonical receipt pins, and a distinct stable no-follow/two-readback
  artifact producer. Static/type review and additional hardening remain open.
- Runtime-v3 now exercises the literal source-only loader and held closure:
  **4/4** loader/drift/tamper cases, **37/37** contract cases, and a nominal full
  literal-loader lifecycle pass. Ruff is green. Its mypy cleanup is still
  being integrated and all affected suites must be rerun on final bytes.
- Semantic-guard-v9 now verifies the held 1,817-file, 54,978,151-byte runtime
  bundle before target imports. The real six-module focused target passes in
  8.54 seconds, and the manifest-creation race test also passes. Full rollback,
  quarantine, handle, deadline, and adversarial gates remain required before a
  candidate exists.
- Protocol candidate
  `9a8be13f8ae06c2eb54bfe54006cbb789ba7f4c73ed816433cdbb4fab444ce03`
  had **0 P0 and 0 P1** in static review, but was not frozen because eight
  advertised module-identity negative tests failed earlier during surface
  consistency and did not reach the repaired lane/leaf policy. Test-only
  correction is in progress: coherent mirrored wrong-section inputs,
  valid module-only wrong leaves, and a positive nonempty
  module-binding-identity path.
- A fresh read-only live-state check found exactly three historical amendment
  directories, zero supervisor jobs, zero Python AANCA processes, and the
  unchanged historical public-Q file at 21,274 bytes/SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  No training, primary, confirmatory, recovery, publication, or scientific
  verifier is active.
- The immediate read-only verification after this documentation append exited
  0 with marker `Q_POST_RESUME_V1_REJECT_DOC_APPEND_PASS`, the same exact
  21,274-byte size, expected three-key lineage envelope, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
  No Q attempt, write, retry, or consumption occurred.
- No live source, frozen science, raw PanNuke data, run, registry, authority,
  or supervisor state changed. The user's one future Q replacement-v2
  publication and independent-verification authorization remains unconsumed.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - close the remaining isolated P0/P1 and freeze candidates

Finish resume-v2, runner-core directive tests, dual Q/E lifecycle/crash tests,
terminal-v2 independent-reader gates, runtime-v3 full typing/lifecycle gates,
semantic-guard-v9 rollback/handle/deadline gates, and the protocol test-only
correction. Freeze only byte-stable cache-free roots and obtain fresh
independent audits. No live integration or authority write precedes those
verdicts.

## 2026-07-30 - Protocol-v7 independently qualified and mechanically frozen

- The repaired protocol-v7 candidate passed a fresh independent static audit
  with **0 P0, 0 P1, and 0 material P2**. The auditor reconfirmed all six
  roles, eight allowed mutations, exactly-once QA, chronology, complete
  cross-record identity disjointness, the 16 MiB/512 MiB bounds, all nine
  fail-closed blockers, false readiness, and absence of local publication.
- The exact audited three-file bundle was copied byte-for-byte, without
  executing or editing it, to the read-only external root
  `C:\Users\NATAN\Documents\AANCA_semantic_guard_successor_bootstrap_frozen_20260730_e701b8c3`.
  Its canonical inventory is 447 bytes and its root was recomputed twice as
  `e701b8c362a033887e85bd74db6b2ae9bead9b59091b6eb473f2b30e677a0a33`.
  The freeze proof found exactly three files, zero subdirectories/caches,
  zero reparse points, zero non-default ADS, one hardlink per file, distinct
  file identities, and read-only attributes on the root and all files.
- Exact frozen files:
  - `DESIGN_V3.md`: 48,961 bytes,
    SHA-256 `5772da03423470e877e289942b2c71741c9bba83bc91abf3fc7facbd11c8eebf`;
  - protocol source: 249,573 bytes,
    SHA-256 `d030b1b1a106761333e7f9295040e0b63378c21fabfac893a95e8ce4b3ba7c38`;
  - protocol tests: 173,112 bytes,
    SHA-256 `03211ba9685bf958494c7321e4026ffb3136d6c741c6b85b78ce15daf76286c1`.
- The isolated fresh/successor runner core now has **48 focused tests passed**,
  including exact 180-directive projection, mixed incomplete/terminal
  successor behavior, zero-step byte-identical terminal restoration,
  immediate fatal rejection of missing/changed/hardlinked checkpoints,
  rejection of pre-existing fresh paths, direct-caller rehash rejection, and
  escape of structural errors from per-cell failure demotion. This remains a
  working candidate, not a live integration.
- The mandatory read-only public-Q verifier after this documentation update
  exited 0 with marker `Q_POST_PROTOCOL_V7_FREEZE_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No scientific process, publication, Q/E authority, supervisor job, frozen
  science, raw data, existing run, or registry changed. The user's one
  future Q replacement-v2 authorization remains unconsumed. Formal status is
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - finish and independently audit the remaining candidates

Complete and stabilize semantic-guard-v9, runtime-v3, dual-authority-v2,
terminal-receipt-v2, successor-resume-v2, and runner-core. Independently audit
each exact allowlist before any live integration. Do not publish Q, create E,
arm the supervisor, or start confirmatory science while any P0/P1 remains.

## 2026-07-30 - Terminal-receipt-v2 rejected by independent audit

- The byte-stable three-file terminal-receipt-v2 candidate remained exactly
  170,932 bytes at root
  `fc139e8d08cf14b6a494bedcf98f96fcb9b1c63394c0a0d80afcca58892c0d61`,
  but a fresh independent audit returned **NOT QUALIFIED**.
- Reproduced P0 defects:
  - injectable semantic hooks were authenticated only through their reported
    source-file path; a `co_filename`-spoofed hook set could claim
    `CONFIRMATORY_COMPLETE` from invalid statistics/restoration bytes;
  - the builder derived the full-receipt pin from the same receipt that it
    then validated instead of consuming independently fixed external pins;
  - three real RunTracker lock files could remain active while the receipt
    hard-coded `active_lock_count=0`;
  - a positive stage attestation timestamped before the immutable seal was
    accepted;
  - trailing-dot Windows aliases allowed two lexical evidence roles to refer
    to the same physical file;
  - current stage eligibility/disposition was not rechecked, so a later
    withdrawal could coexist with an earlier positive row.
- P1 defects included missing ancestor ADS inspection, missing read-only and
  retained ancestor-identity checks, incomplete standalone-reader path
  hardening, declaration-only predecessor qualification/copy documents,
  permissive unknown status/completion fields, and no independent cross-link
  of supervisor job/attempt/no-retry/exit/terminal state. Strict mypy also
  found one source error. The focused suite still passed 43 tests, showing why
  independent adversarial qualification is mandatory.
- The rejected root remains unchanged and is not frozen, promoted, or
  integrated. A new isolated terminal-receipt-v3 repair has been assigned.
- Runtime-v3 separately reached a byte-stable candidate boundary with
  **101 passed**, Ruff/format/configured-and-strict-mypy/compile clean, eight
  cache-free files, and twice-reproduced root
  `acf1d9799d74137db56675c6cf132f43b192761de43a7b2b8c53eba0873cfd92`.
  It is now under an independent read-only audit and is not yet qualified.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_TERMINAL_V2_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No scientific process, Q/E write, supervisor job, source integration,
  frozen science, raw data, existing run, or registry changed. The one
  future Q replacement-v2 authorization remains unconsumed. Formal status is
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - repair terminal v3 and finish independent component gates

Implement terminal-receipt-v3 in a new isolated root while the independent
runtime-v3 audit, semantic-guard-v9 gates, dual Q/E repairs, resume-v2, and
runner-core continue. Freeze or integrate nothing until each exact candidate
has a fresh 0-P0/0-P1 verdict.

## 2026-07-30 - Cross-component pre-Q audit returns STOP / NOT READY

- A bounded read-only audit of the complete proposed Q -> E -> supervisor ->
  terminal -> M9 chain returned **STOP / NOT READY** before any authority was
  written. It reconfirmed the frozen protocol root
  `e701b8c362a033887e85bd74db6b2ae9bead9b59091b6eb473f2b30e677a0a33`
  byte-for-byte at both start and end.
- Confirmed pre-Q P0 defects in the current dual-gate-v2 working snapshot:
  - Q/E did not bind the full argv tail, `sys.orig_argv`,
    `GetCommandLineW`, exact cwd, requested/effective interpreter relation,
    or independently recomputed supervisor command;
  - launch nonce/session/current-child/parent-supervisor/job identity were
    partially compared with values supplied by E itself; the consumption
    receipt recorded only the current PID;
  - source records, programs root, and private bundle were not completely
    recomputed and cross-linked to Q/runtime;
  - Q/E did not bind the exact expected terminal-v3 receipt, seal, integrity,
    and supervisor terminal-verifier contract.
- Confirmed downstream P0 defects that must be repaired before M9 execution:
  - the original-audit eligibility path did not require the positive
    post-seal stage ledger and current disposition;
  - review-package CLI accepted arbitrary top/random counts and seed 509
    instead of the frozen **100 top / 100 random / seed 707** contract.
- Cross-component P1 blockers:
  - the production gate and runner did not yet expose one closed, canonical
    `fresh | successor_resume` chain;
  - resume-v2 used a second successor-specific Q model and synthetic
    supervisor schema/release identifiers incompatible with the installed
    qualified supervisor;
  - no typed original-confirmatory CUDA/AMP/cache/RAM/sealed-plan-capacity
    receipt existed before E consumption; CUDA failure occurred only inside
    training;
  - M9 eligibility expected run name `confirmatory_study` while the real
    runner seals `pannuke_confirmatory_study`.
- Dual-authority-v2 and resume-v2 owners are repairing the canonical one-Q,
  per-attempt-one-E architecture. A separate isolated preflight owner is now
  building the exact original-confirmatory receipt: 108/90/18 cells,
  36 CNN cells, 180 fits, three rotations, five-cache identity, CUDA/AMP,
  RAM, run-volume capacity from a sealed plan plus exactly 10 GiB, and
  launch-freshness/replay protection.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_CROSS_COMPONENT_STOP_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, authority, supervisor job, training, confirmatory
  output, or M9 output was created. The single Q replacement-v2
  authorization remains unconsumed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - close the cross-component authority and preflight gaps

Finish isolated Q/E, resume, runner, preflight, semantic/runtime, and
terminal-v3 candidates; then independently audit exact byte-stable roots.
Only a zero-P0/zero-P1 set may enter live integration and the final mandatory
QA/PanNuke/lifecycle gates.

## 2026-07-30 - Runtime-v3 rejected; runtime-v4 required

- Independent read-only audit returned **NOT QUALIFIED / STOP** for the
  unchanged eight-file runtime-v3 root
  `acf1d9799d74137db56675c6cf132f43b192761de43a7b2b8c53eba0873cfd92`
  with **3 P0, 3 P1, and 1 P2/QA**.
- Reproduced P0 defects:
  - application modules could execute before the interpreter/native runtime
    was fully authenticated; the contract lacked a complete held
    Python/stdlib/PYD/DLL closure and used later path-based reads;
  - the supervisor -> initial loader -> launcher/child chain omitted exact
    loader PID/creation/parent/Job proof and accepted caller-supplied process
    output fields; forged `process_identity`, stdout, stderr, and parsed
    stdout were accepted;
  - terminal `committed` accepted syntactically valid but arbitrary
    claim/payload/manifest/private/commit/final hashes instead of independently
    recomputing their files and semantic relationships.
- P1 defects covered incomplete handle-object/alias inventories, an empty
  wait/deadline attestation being accepted, and the global singleton not
  covering direct public publisher/verifier entry. The independent QA also
  found three Ruff violations and seven files not matching the repository
  formatter, despite 101 tests and both configured/strict mypy passing.
- Runtime-v3 remains unchanged and is not promoted. A new isolated runtime-v4
  owner must hold and verify the entire runtime closure before any target
  import, bind the real supervisor/process chain, independently recompute
  terminal claims, enforce exact handle/deadline evidence, and cover every
  public operational path with one singleton.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_RUNTIME_V3_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No source integration, Q/E write, supervisor job, scientific process,
  existing run, registry, frozen science, or raw data changed. The single
  Q replacement-v2 authorization remains unconsumed. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - produce and independently audit runtime-v4

Implement runtime-v4 in a new isolated root with the reproduced P0/P1 cases as
mandatory adversarial regressions. Continue the other isolated candidates in
parallel; do not freeze, integrate, publish Q, create E, arm the supervisor,
or start science.

## 2026-07-30 - Semantic-v9 and terminal-v3 enter independent audit; M9 guard isolated

- Semantic-guard-v9 reached a byte-stable 13-file, 1,516,336-byte candidate
  at root
  `c37ad83ba77c12331aa484f0211ffa3d426d1933ce835966450227580037840b`.
  Its exact-byte gates passed **79 tests in 632.19 seconds**, Ruff,
  format-check, configured and strict mypy, and in-memory compile. Python and
  PowerShell independently reproduced the same inventory/root. The author is
  stopped and a fresh read-only audit is in progress; the candidate is not
  frozen or qualified yet.
- Terminal-receipt-v3 reached a byte-stable five-file candidate at twice
  reproduced root
  `dd08e9d67c4f6e8677584e814b4ceac755f4e9e6f2d7b1e6992705ddaaf9b80c`.
  Its focused adversarial/production-validator suite passed **33 tests**;
  Ruff, format, configured/strict mypy, and compile passed. The rejected v2
  root remains unchanged. All final `EXPECTED_*` pins intentionally remain
  unset, so Q/E fail closed while a new independent audit is running.
- A separate isolated M9 guard candidate now closes the read-only
  cross-component findings without executing M9:
  - the tracked original-label audit holds active upstream confirmatory stage
    authority for its full invocation and binds the exact positive
    post-seal-attestation hash into config and sealed evidence;
  - the real confirmatory run name is reconciled as
    `pannuke_confirmatory_study`;
  - a stage-eligible review candidate requires exactly 100 top, 100 random,
    and seed 707;
  - the package CLI no longer self-claims `EXTERNAL_VALIDATION_READY`; it
    returns `completion_stage=null` pending a tracked seal, independent
    technical inspection, and positive post-seal attestation.
- The M9 guard selection passed **102 tests**, Ruff/format, configured and
  strict mypy, and in-memory compile. Its clean nine-file candidate has
  334,441 bytes, 1,011 canonical-record bytes, and independently reproduced
  root
  `073b00a0fcdafaf9317b6551427c0ccb2e78a572491dc8b5b43c2b6610e678ce`.
  It is engineering evidence awaiting independent audit, not M9 execution or
  a completion-stage claim.
- No live source integration, Q/E write, supervisor job, scientific process,
  run, registry, frozen science, raw data, original-label ranking, review
  package, or expert response was created. The single Q replacement-v2
  authorization remains unconsumed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_STABLE_CANDIDATES_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - independently qualify the three stable candidates

Complete semantic-v9 and terminal-v3 audits, then assign a fresh read-only
audit to exact M9 guard root `073b00a0...` when a bounded agent slot becomes
available. In parallel, continue runtime-v4, dual Q/E, preflight, resume-v2,
and the runner TOCTOU repair. No live integration precedes zero-P0/zero-P1
verdicts.

## 2026-07-30 - Terminal-receipt-v3 rejected; command envelope closed

- The independent read-only audit returned **NOT QUALIFIED / STOP** for the
  unchanged five-file, 250,336-byte terminal-v3 root
  `dd08e9d67c4f6e8677584e814b4ceac755f4e9e6f2d7b1e6992705ddaaf9b80c`.
  The exact root and every file hash were reproduced before and after review.
- Five P0 classes were reproduced: binding to rejected runtime-v3; an
  injectable semantic-function route; self-derived rather than independently
  persisted receipt pins; an incomplete seven-role E terminal contract; and
  declaration-only predecessor/copy validation in successor mode.
- Five P1 classes were also reproduced: writable terminal artifacts were not
  rejected; ancestor/leaf identities were not retained by Windows handles;
  status/completion and nested supervisor schemas were not closed; the final
  lock/disposition check was not held through publication; and supervisor
  terminal paths, hashes, time, and duplicate roles were not completely
  cross-linked.
- Terminal-v3 remains unchanged and is not integrated. Terminal-v4 must use
  only qualified runtime-v4, a non-injectable authenticated semantic loader,
  independently persisted pins, retained no-follow Windows identities,
  recomputed predecessor/copy evidence, closed schemas, the complete terminal
  artifact-role set, and a final guard held through atomic publication.
- To remove a cross-component ambiguity before further implementation,
  `exact_command_sha256` is defined over a closed canonical expected launch
  envelope containing the exact supervisor, loader, and child argv; expected
  `sys.orig_argv`, `sys.argv`, and native `GetCommandLineW`; cwd; and requested
  and effective interpreter identities. The child must independently observe
  and match that envelope.
- Execution environments are complete explicit sanitized mappings, not
  inherited subsets. Supervisor and scientific child mappings have separate
  hashes, reject case-colliding or extra keys, and are both bound by the launch
  root. The canonical serialization is compact sorted JSON, UTF-8 without BOM
  or trailing newline, `ensure_ascii=False`, and `allow_nan=False`.
- The historical public-Q readback immediately before this entry passed with
  marker `Q_POST_STABLE_CANDIDATES_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live source, Q/E, supervisor job, scientific process, run, registry,
  frozen science, raw data, or result changed. The one Q replacement-v2 write
  and independent-verification authorization remains unconsumed. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**,
  and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_TERMINAL_V3_REJECT_COMMAND_ENV_DOC_APPEND_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - implement terminal-v4 while remaining gates finish

Start terminal-v4 from the reproduced audit findings as mandatory regressions.
In parallel, finish semantic-v9, preflight, runtime-v4, dual Q/E, resume-v2,
and runner-core qualification. Do not integrate, write Q/E, arm the supervisor,
or start confirmatory science before zero-P0/zero-P1 exact-root verdicts.

## 2026-07-30 - Semantic-v9 and first preflight candidate rejected

- The independent read-only semantic-v9 audit returned
  **NOT QUALIFIED / STOP** for the unchanged 13-file, 1,516,336-byte root
  `c37ad83ba77c12331aa484f0211ffa3d426d1933ce835966450227580037840b`.
  The canonical preimage remained 1,391 bytes and the exact root was reproduced
  before and after review.
- Its 79 candidate tests, 209 frozen protocol-v7 tests, Ruff, format, configured
  and strict mypy, and in-memory compile all passed. The audit nevertheless
  reproduced one P0: the parent duplicated the Job handle into the child, so
  a parent crash did not trigger `KILL_ON_JOB_CLOSE`; the parent-owned singleton
  could then be reacquired while the old child survived.
- Two P1 classes were also reproduced: ambiguous `CloseHandle` failure lost
  authority and still released Job/singleton instead of quarantining state;
  constructor failure paths could leak Job, completion-port, or cache-barrier
  handles. Semantic-v10 must keep the Job handle parent-only, test parent death,
  preserve/quarantine ambiguous handles, and make every constructor rollback
  complete.
- A separate independent adversarial review returned **STOP / NOT QUALIFIED**
  for the then-current preflight source SHA-256
  `1915d8ddd7921b9e734da9432db73751421d9caf4f7084cf68648e01ca78aace`.
  Reproduced blockers included an unparsed attacker-controlled reservation,
  a public route that bypassed mandatory E reservation, caller-supplied rather
  than OS-observed command/environment evidence, no full child-boundary
  snapshot readback, parent-directory publication TOCTOU, and declaration-only
  plan/rehearsal evidence.
- The preflight owner is repairing those findings with canonical reservation
  parsing, a non-qualifying private test seam, runtime-v4 observed launch
  envelopes, a complete child-boundary snapshot, retained no-follow directory
  identities, a real sealed execution-plan reconstruction, and tracked
  rehearsal seal/manifest/integrity readback. Its release pins remain unset.
- A read-only Git protection check again confirmed that representative raw
  PanNuke ZIP/NPY, nested manifest Parquet, and duplicate-audit NPZ files are
  ignored by `.gitignore` rules `data/raw/**`,
  `data/manifests/**/*.parquet`, and `artifacts/duplicate_audit/*.npz`.
- The historical public-Q verifier immediately before this entry exited 0
  with marker
  `Q_POST_TERMINAL_V3_REJECT_COMMAND_ENV_MARKER_DOC_FINAL_PASS`, the exact
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, Q/E write, supervisor job, scientific process, run,
  registry, frozen science, raw data, or result changed. The exactly-one Q
  replacement-v2 write and independent-verification authorization remains
  unconsumed. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_SEMANTIC_V9_PREFLIGHT_REJECT_DOC_APPEND_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - implement semantic-v10 and re-audit preflight

Implement semantic-v10 from the reproduced handle-lifetime failures while
preflight repairs its independent findings. Continue terminal-v4, runtime-v4,
dual Q/E, runner-core, and the read-only M9 guard audit in parallel. No
candidate may be integrated or bound into Q before a fresh zero-P0/zero-P1
exact-root verdict.

## 2026-07-30 - M9-guard-v1 rejected before any original-label audit

- The independent read-only audit returned **NOT QUALIFIED / early STOP** for
  the unchanged nine-file, 334,441-byte M9-guard-v1 root
  `073b00a0fcdafaf9317b6551427c0ccb2e78a572491dc8b5b43c2b6610e678ce`.
  Its 1,011-byte canonical records and exact root were reproduced before and
  after review; there were no caches, reparse points, extra ADS, aliases, or
  shared hardlinks.
- One P1 public-authority bypass was reproduced. The review-package builder
  accepted an arbitrary nonempty mapping with forged SHA-shaped evidence as
  verified eligibility, and the structural validator accepted the resulting
  200-item package as `study_outcome_eligible=true` without reconstructing the
  real confirmatory authority. The CLI gate alone could not protect direct
  public API use.
- The validator also reconciled counts only against self-declared metadata
  instead of independently requiring exact schema-v2, 100 top-ranked items,
  100 disjoint random items, and seed 707. Candidate files were writable,
  recorded as a P2 freeze defect.
- M9-guard-v2 must make the public builder non-stage by default and must not
  accept an arbitrary mapping as authority. A distinct stage verifier must
  reconstruct active confirmatory authority, the exact post-seal attestation,
  current disposition, schema-v2 evidence, 100/100/707, and cohort
  disjointness. Forged but syntactically valid hashes are a mandatory
  adversarial regression. The final audited root must be cache-free and
  read-only.
- No original-label ranking, review package, expert response, live source,
  Q/E, supervisor job, scientific process, run, registry, frozen science, raw
  data, or result changed. The one Q replacement-v2 authorization remains
  unconsumed. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%**, and M9 remains locked.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_SEMANTIC_V9_PREFLIGHT_REJECT_MARKER_DOC_FINAL_PASS`, the exact
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_M9_GUARD_V1_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - implement and independently audit M9-guard-v2

Repair the public builder/verifier boundary in a new isolated candidate while
the M8 critical-path components continue. M9 remains locked until a fresh
zero-P0/zero-P1 read-only exact-root verdict and, later, a qualifying
`CONFIRMATORY_COMPLETE` post-seal authority.

## 2026-07-30 - Runner-core working candidate fails independent contract audit

- An independent read-only audit returned **NOT QUALIFIED** for the
  then-current isolated runner-core working candidate. It was not byte-stable,
  had no promoted root, and remains outside live source.
- Three P0 classes were reproduced: a caller-created structural Protocol and
  caller-supplied hashes could fabricate an E binding; the production runner
  built an unbound 180-fit draft and never invoked a real consumed-E binding;
  and checkpoint publication retained a verify-then-`os.replace` TOCTOU that
  accepted a same-byte identity swap.
- P1 findings showed that fresh checkpoint-originated exceptions could be
  demoted to an ordinary cell failure; E directives omitted destination
  identity; the full resume identity was reduced to a weaker surrogate; and
  directory ADS were not rejected. Missing race/error tests were recorded as
  P2.
- The repair now requires the production entrypoint to accept an exact
  consumed-E receipt path/root and independently invoke the authenticated
  canonical dual-Q/E verifier. Caller-created objects or hashes are never
  authority. Test fixtures are explicitly non-qualifying.
- Mutable checkpoint replacement is removed. Each fit/epoch/attempt publishes
  a unique immutable no-overwrite checkpoint from a held temporary identity,
  verifies the destination is that same file identity, and records it in an
  append-only/no-overwrite manifest. Fresh checkpoint errors are fatal; exact
  source and destination identities, full resume identity, and directory ADS
  checks are mandatory.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_M9_GUARD_V1_REJECT_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, Q/E write, supervisor job, scientific process,
  checkpoint, run, registry, frozen science, raw data, or result changed. The
  one Q replacement-v2 authorization remains unconsumed. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_RUNNER_CORE_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - repair runner authority and immutable publication

Complete the independent consumed-E readback and versioned checkpoint
publication regressions, then produce a new minimal byte-stable allowlist for
fresh audit. Continue the other isolated v4/v10/v2 repairs in parallel.

## 2026-07-30 - Real PanNuke baseline validator is semantically valid

- Executed the current live functional command against the complete local
  dataset without changing raw files:

  `.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke --project-root . --root data\raw\pannuke --output-dir artifacts\qa\pannuke_baseline_20260730T052711.9477865Z\report`

- The detached process pair PID 25256/22220 ended naturally. Structured stdout
  reported `status=valid`, `validation_scope=full_semantic_scan`, three folds,
  7,901 patches, and 22 raw files; stderr was empty. The launcher did not retain
  the OS exit code in a separate receipt, so this is explicit baseline evidence,
  not the final qualifying CLI gate. The post-integration run must retain and
  verify exit code 0.
- Full QC reproduced 4,318 cross-class-overlap pixels in 575 patches,
  10,486,091 void pixels in 162 patches, 737 union-affected patches, 7,164
  normal patches, and 1,411 overlap-touching instances. All 1,411 are excluded
  identically from primary and confirmatory analysis with reason
  `touches_cross_class_overlap`; no class arbitration occurred, no positive
  pixel overlapped supplied background, and `source_masks_modified=false`.
- Per-fold results exactly reproduced the known raw behavior:
  - fold 1: 1,216 overlap pixels / 194 patches; 2,359,296 void pixels;
  - fold 2: 1,572 overlap pixels / 190 patches; 3,801,371 void pixels;
  - fold 3: 1,530 overlap pixels / 191 patches; 4,325,424 void pixels.
- Shapes were exactly `(2656,256,256,6)`, `(2523,256,256,6)`, and
  `(2722,256,256,6)`. The generated raw inventory independently matched all
  user-supplied archive sizes and SHA-256 values:
  - fold_1.zip: 700,275,281 bytes,
    `6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b`;
  - fold_2.zip: 658,842,552 bytes,
    `5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07`;
  - fold_3.zip: 717,969,882 bytes,
    `c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d`.
- Baseline repository lint gates also passed:
  `.venv\Scripts\python.exe -m ruff check .` returned `All checks passed!`,
  and `.venv\Scripts\python.exe -m ruff format --check .` reported all 180
  files formatted. These must also be repeated after integration.
- Reports are isolated under
  `artifacts/qa/pannuke_baseline_20260730T052711.9477865Z`; raw ZIP/NPY files
  remain unchanged and ignored by Git. This successful data validation alone
  does not advance the scientific completion stage or close M8.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_RUNNER_CORE_REJECT_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No Q/E, supervisor job, training, confirmatory run, original-label ranking,
  frozen science, source annotation, raw data, or scientific result changed.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_REAL_PANNUKE_BASELINE_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - finish candidate audits, then repeat qualifying QA

Continue the isolated semantic-v10, runtime-v4, terminal-v4, dual-Q/E,
preflight-v2, runner, and M9-guard-v2 repairs. After only independently
qualified allowlists are integrated, rerun full pytest, Ruff, the functional
CLI with retained exit code, and this complete PanNuke validator.

## 2026-07-30 - Preflight-v1 rejected and preserved; v2 isolated

- The fresh read-only audit returned **NOT QUALIFIED / STOP** for preflight-v1.
  Its exact six-file root
  `2e368d1142460c11d9d4c76099e216af14f4a12f6f1b1af308318e44845c580b`
  and module SHA-256
  `3bcd510d3e209f93dd2e133f0de66948ed2561b2af0183320f7682b21bf00cbb`
  were independently reproduced.
- Reproduced P0/P1 classes were: runner entry trusted a caller-created
  consumed-E mapping with `verified=True`; a publicly reconstructible
  `ChildVerifiedLaunchBinding` could substitute for held authority and no
  persistent replay marker was required; dynamic `sys.modules` substitution
  could replace authority/parser/OS-observer modules; tracked manifest and
  immutable evidence were opaque-hashed instead of semantically verified; and
  call-4 claim did not require the exact parent reservation.
- The owner briefly began post-audit repairs in the v1 working directory,
  immediately stopped, moved all new work to
  `AANCA_original_confirmatory_preflight_v2_working_20260730`, restored v1,
  removed temporary cache drift, and reproduced the exact audited v1 module
  and six-file root above. V1 is now closed and will not be modified again.
- Preflight-v2 must independently acquire a retained, no-follow,
  non-reconstructible production capability; publish an O_EXCL persistent
  runner-consumption marker; reject every mapping/boolean/public-binding
  substitute; require and canonically parse the parent reservation; use only
  exact held/sealed dependency identities; and semantically reconstruct
  tracked manifests, immutable seals, and integrity roots.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_REAL_PANNUKE_BASELINE_MARKER_DOC_FINAL_PASS`, the exact
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live source, Q/E, supervisor job, training, run, registry, frozen science,
  raw data, or scientific result changed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_PREFLIGHT_V1_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - audit the v2 capability boundary before new QA claims

Finish preflight-v2's held capability and semantic reader repair, then create
a new stable root and run a different independent audit. Continue the other
critical-path candidates without binding preflight-v1 anywhere.

## 2026-07-30 - M9-guard-v2 enters fresh read-only audit

- M9-guard-v2 reached a read-only, cache-free candidate at
  `C:\Users\NATAN\Documents\AANCA_m9_guard_v2_candidate_20260730_6ae9e06a`.
  Python and PowerShell independently reproduced the exact nine-file,
  373,084-byte root
  `6ae9e06a135c2887f414b11819a6aa9f91de758e34f6606be7b118636af6106d`
  from a 1,012-byte canonical record stream.
- Candidate gates passed 35 focused tests and 76 impacted tests cache-off,
  Ruff check and format, configured mypy over 89 files, strict mypy over five
  files, and in-memory compile over eight files. All nine files and six
  directories are read-only; no cache, reparse, ADS, shared-hardlink,
  duplicate-file-ID, or alias defect was observed.
- The public builder is now non-stage and rejects caller mappings/typed
  authority. Structural validation cannot attest study eligibility. A separate
  path-only stage verifier independently reconstructs active upstream
  authority, schema-v2 evidence, the exact 100+100/seed-707 disjoint cohort,
  assets, labels, schema, and HTML under held stage/disposition locks.
- Forged SHA-shaped evidence, direct API bypass, wrong count/seed,
  non-disjoint cohort, static and mid-flight tamper, and withdrawal races are
  explicit regressions. The v1 root `073b00a0...` was rechecked unchanged.
- The v2 author stopped. A different agent is now auditing the exact read-only
  root; this is candidate evidence only, not integration, M9 execution, a
  review package, expert work, or an `EXTERNAL_VALIDATION_READY` claim.
- Cross-component implementation also fixed three launch-contract decisions:
  fresh execution requires null predecessor/resume hashes while successor
  requires both exact hashes; the new supervisor contract requires exactly 64
  lowercase-hex nonce characters with no legacy 32-hex compatibility; and
  post-`WaitForExit` E validation independently reconstructs persistent
  acceptance/runner-consumption evidence instead of requiring a live child.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_PREFLIGHT_V1_REJECT_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, Q/E, supervisor launch, science, M9, package, ranking,
  expert response, raw-data change, or result occurred. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_M9_GUARD_V2_CANDIDATE_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - obtain the independent M9-v2 verdict

Complete the exact-root M9-v2 audit while semantic-v10, runtime-v4,
terminal-v4, dual-held Q/E, preflight-v2, and runner repairs continue. Promote
nothing unless the fresh verdict is zero P0 and zero P1.

## 2026-07-30 - M9-guard-v2 rejected on the real frozen cache contract

- The independent read-only audit returned **NOT QUALIFIED / early STOP** for
  exact root
  `6ae9e06a135c2887f414b11819a6aa9f91de758e34f6606be7b118636af6106d`
  with zero P0 found before stop and one P1.
- The real frozen `imagenet_context_embedding_cache` intentionally has
  `cache_file_sha256: null` and a non-null `sidecar_semantic_sha256`, exactly
  as PRE_REGISTRATION.md defines for every available confirmatory cache
  record. The v2 verifier required the inverse: a direct cache SHA equal to the
  actual file and a null semantic-sidecar authority. Consequently every legal
  frozen config would always be declared ineligible.
- V2 tests concealed the defect by constructing a non-scientific fixture with
  a direct cache SHA and null semantic sidecar. The candidate remained
  unchanged after audit: nine files, 373,084 bytes, 1,012 canonical-record
  bytes, all 15 items read-only, and no ADS, reparse, alias, hardlink, or
  duplicate-file-ID issue.
- M9-guard-v3 must load the actual frozen record, require exactly one authority
  branch, and for the real semantic-sidecar branch invoke the canonical
  `verify_frozen_cache_sidecar` reader. It must independently recompute the
  current cache/sidecar relationship and compare the verifier's
  `sidecar_semantic_sha256` with the frozen value. It must not invent or write
  a direct cache SHA into the frozen config. Positive tests must use the real
  frozen contract; inverse, neither, both, tamper, and stale-sidecar cases must
  fail closed.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_M9_GUARD_V2_CANDIDATE_MARKER_DOC_FINAL_PASS`, the exact
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No M9 execution, package, ranking, expert response, live integration, Q/E,
  supervisor launch, science, frozen-file edit, raw-data change, or result
  occurred. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_M9_GUARD_V2_REJECT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - implement M9-guard-v3 without changing frozen inputs

Preserve v2 read-only and repair only a new isolated v3 candidate using the
canonical semantic-sidecar verifier and actual frozen fixtures. Repeat the
full v2 adversarial surface and a fresh independent exact-root audit.

## 2026-07-30 - Revised runner data-plane stopped on a remaining handoff TOCTOU

- A fresh scoped read-only audit returned **NOT QUALIFIED / P1** with no P0
  for the revised six-source/seven-test root
  `b178ac29eb65745ee666a341c939e61ea7d0d3cd332d9216a4b9d1ed7ced1ee2`.
  The 90 scoped tests passed in 41.33 seconds.
- The revision correctly removed mutable checkpoint `latest`, published held
  temporary files by no-overwrite hardlink, checked destination file ID and
  final link count, preserved full seven-field identity, enforced fresh versus
  successor lineage, exact 36x5 versus 6x5 profiles, directory ADS rejection,
  and fatal typed checkpoint errors.
- One handoff remained path-racy in `_normalise_image_execution` and
  `_persist_completed_cell`: code compared `lstat` identity, then called a
  separate path-based `sha256_file` without a post-read identity check.
  A same-byte file replacement in that window could be adopted. `_atomic_npz`
  also retained a literal `os.replace`.
- The next revision must read/hash through one retained no-follow descriptor,
  verify the same handle before and after reading, verify the current path still
  resolves to that held file ID, and retain the identity/bytes through
  adoption. Immutable NPZ publication must also be no-overwrite from a held
  temporary identity. Same-byte swap, pre-link source swap, restoration
  manifest identity, and concrete authority-context canaries are mandatory.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_M9_GUARD_V2_REJECT_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, Q/E, supervisor launch, science, checkpoint, run,
  registry, frozen-file edit, raw-data change, or result occurred. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**,
  and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_RUNNER_HANDOFF_P1_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - close held-descriptor handoffs and re-audit

Repair both remaining path-separated handoffs and immutable NPZ publication,
rerun scoped and impacted gates, and submit a new byte-stable minimal root to a
different read-only auditor. The dual-held production entry remains separately
fail-closed until qualified.

## 2026-07-30 - Semantic-guard-v10 enters fresh exact-root audit

- Semantic-v10 reached a byte-stable 13-file, 1,544,041-byte working root
  `a2eade224ff1ff15b2c76947c5020eafdccec1b23383ea9e04c8772bf6d5018c`
  with a 1,391-byte canonical inventory preimage independently reproduced by
  Python and PowerShell. V9 remained unchanged at `c37ad83...`.
- The first full attempt honestly produced 85 passes and one positive-test
  deadline failure. The failure was reproduced; only that QA timeout changed
  from 20 to 40 seconds while the one-absolute-deadline production semantics
  remained unchanged. The complete clean rerun passed **86 tests in 660.33
  seconds**, and frozen protocol-v7 passed **209 tests in 17.86 seconds**.
- Ruff without cache, format-check, configured and strict mypy over 12 files,
  in-memory compile over 12 files, and bootstrap/target self-consistency all
  passed. Two topology reads found 13 files/seven directories and no cache,
  reparse, extra ADS, non-singleton hardlink, or duplicate file-ID issue.
- The exact three-file delta is:
  - `SEMANTIC_GUARD_POLICY.md`, 17,585 bytes,
    `0dec58ece79d429d5a18e699b8ab9cca23af7c7dfee694e8a196be170577c1db`;
  - `target_launcher/semantic_guard_target.py`, 75,029 bytes,
    `f312ed29b3ea6c832e98503d5f6ae9a223754bf748660c83d0ce4052b2c391a2`;
  - `tests/test_r3_semantic_guard_successor.py`, 238,024 bytes,
    `514ae52d5b187dff2a22c3c42ecc7a83db1ec86d3354e97aa1cc379b542d4645`.
- The revision makes Job authority parent-only, proves parent-death
  `KILL_ON_JOB_CLOSE`, keeps child proof free of Job ownership, quarantines
  ambiguous close state, prevents caller drop/retry, and completes Job,
  completion-port, and cache-barrier rollback. The author stopped and a fresh
  independent auditor is now testing the exact root; it is not self-qualified,
  frozen, integrated, or pinned.
- The historical public-Q verifier immediately before this entry passed with
  marker `Q_POST_RUNNER_HANDOFF_P1_MARKER_DOC_FINAL_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No live integration, Q/E, supervisor launch, science, run, registry, frozen
  science, raw data, or result changed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_SEMANTIC_V10_CANDIDATE_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - obtain semantic-v10 independent verdict

Complete the full exact-root read-only audit. Only a zero-P0/zero-P1 verdict
may permit a mechanical read-only freeze and later Q binding; terminal-v4
remains intentionally unpinned meanwhile.

## 2026-07-30 - Semantic-guard-v10 rejected; one future Q-v2 write authorized but unconsumed

- A fresh independent, read-only audit returned **REJECTED / 0 P0 / 1 P1** for
  semantic-v10. Python and PowerShell independently reproduced the unchanged
  before/after root
  `a2eade224ff1ff15b2c76947c5020eafdccec1b23383ea9e04c8772bf6d5018c`:
  13 files, 1,544,041 bytes, and a 1,391-byte canonical inventory preimage.
  Two topology passes were identical: 13 files, seven descendant directories,
  unique physical file IDs, link count one, and no ADS, reparse point, cache,
  alias, or duplicate-ID issue.
- The reproducible P1 is an after-native-close ambiguity at the final Job or
  singleton authority guard. The v10 close sequence retained only one guard of
  each kind after closing the originals. If native `CloseHandle` succeeded and
  the wrapper then raised, quarantine recorded the positive handle although it
  was no longer readable. Job-guard requalification failed with Windows error
  6; singleton-guard requalification failed because no independently readable
  Job authority remained. The existing ambiguity test injected only at the
  child process handle and did not cover either final guard stage.
- Per the audit early-stop rule, the auditor did not rerun the full 86+209
  suites after reproducing the P1. The author-reported gates therefore do not
  qualify v10. V10 remains byte-identical and rejected; a separate v11 working
  root is being built with staged independent authority readback and
  before/after-native-close canaries for every close stage.
- The user authorized exactly **one future Q replacement-v2 write and one
  independent verification**. That authorization is recorded but remains
  **unconsumed**: no Q or E was written, no live source was integrated, and no
  supervisor or scientific process was launched. The write may occur only
  after all exact source, runtime, preflight, command/environment, terminal,
  resume, runner, lifecycle, and capacity inputs are independently qualified.
  General permission to continue implementation and QA does not substitute for
  the later exact, one-use E/run authority bound to those final hashes.
- The installed event-driven Windows supervisor remains qualified and unarmed.
  PID 20792 and launcher PID 9476 are absent, and the current process audit
  found no AANCA training, primary, confirmatory, recovery, publication, or
  supervisor process. Codex continues short implementation/QA work now; the
  supervisor will be armed only for one subsequently qualified long process.
- The historical public-Q verifier immediately before this entry exited 0 with
  the exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No completion-stage transition occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_SEMANTIC_V10_REJECT_AUTHORIZATION_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - qualify v11 and the remaining isolated launch chain

Repair semantic-v11 in a new isolated root while terminal-v4, runtime-v4,
preflight-v2, dual Q/E, runner-core, resume-v2, and M9-guard-v3 complete their
independent gates. Do not integrate, write Q/E, arm the supervisor, or execute
science until every P0/P1 is closed and the exact allowlists are frozen.

## 2026-07-30 - Interface freeze produced concrete rejections and three new audit candidates

- Semantic-v11 passed its author gates but a fresh read-only probe reproduced
  another P1: a successfully closed Job or semaphore handle could have its
  numeric value immediately reused before the Python receipt update, allowing
  an unrelated object with matching readback to be mistaken for retained
  authority. V11 root
  `b92de6142bfe99c8e2c407fadda4f2c8a764012e743336460d1f8ff3dd8df797`
  remains byte-identical and rejected.
- The separate semantic-v12 candidate removes handle-liveness inference,
  treats an unresolved native close as non-authoritative, clears the active
  slot before ABA reuse, and adds real Job and singleton ABA canaries. Its
  author gates passed: 87 tests in 697.79 seconds, frozen protocol-v7 209
  tests, Ruff check/format, configured and strict mypy, and compile. The exact
  13-file, 1,575,657-byte candidate root is
  `106ab9fcadae4005e6806455e0054869561f1c54fd4e9fc0e37ce18057e75863`.
  It is not qualified; a different auditor is being assigned.
- The isolated runtime-v4 candidate reached 119 tests in 253.56 seconds,
  Ruff, format, configured mypy, compile, and clean physical checks over eight
  files/720,185 bytes. Its exact working root is
  `0745cc323a6a283eae2d9b11ba43a801f21f0738e19b376773debdfec4c935e0`.
  It remains mutable, synthetic-only, and pending a fresh read-only audit.
- M9-guard-v3 reached an immutable read-only candidate at
  `C:\Users\NATAN\Documents\AANCA_m9_guard_v3_candidate_20260730_f6040f05`,
  root
  `f6040f057e0704f2d3fdfd436aa4bff6b647fabedac71de8274173966cbb154c`,
  nine files/386,096 bytes. Its owner gates passed 44 focused and 94 impacted
  tests plus Ruff, format, configured/scoped mypy, compile, and physical
  checks. It correctly preserves the real semantic-sidecar cache branch, but
  it is not integrated or qualified until a different auditor reproduces the
  exact root and P0/P1 result.
- Preflight-v2 root
  `e38e0fb2c0686b65b6d34b51f21a02583f18d98e9a97348c86b410c2b7016539`
  was rejected. Its source SHA constant was a whole-file self-hash fixed-point,
  unset pins were not fail-closed, executing code was not proven to come from
  the held bytes, the runner contract escaped its context, and the physical
  identity readback was incomplete. Several isolated preflight-v3 snapshots
  then passed ordinary QA but were also withdrawn after adversarial probes
  found mutable semantic captures and no-escape gaps. None was integrated.
- The approved repair direction changes only the nested preflight contract:
  a Q-bound deterministic private literal loader holds a five-role bootstrap
  closure (loader, preflight, authority, runtime observer, runner core);
  the persistent marker binds four executed source roles; consumed E binds the
  Q pre-run-contract root; critical functions/types are captured from the
  private modules; and runner execution is token-only with no caller callback,
  raw contract, or lease escape. Top-level Q/E fields remain unchanged. The
  approval is for isolated implementation and fresh audit only.
- Terminal-v4 passed 24 focused tests, Ruff, format, and strict mypy, but is
  **nonqualifying**. A forged persisted reviewer pin with an arbitrary PID was
  accepted, the final lock cleanup had a close-before-unlink race, and the
  terminal path did not call the full physical dual-Q/E post-run validator.
  The repair must use an exact pre-bound distinct verifier/pinner child through
  the supervisor integrity-verifier path and bind its process/source/command/
  environment/exit evidence plus the O_EXCL pin identity and SHA into the
  supervisor terminal receipt.
- Runner-core audit rejected its working roots. The latest complete audit
  reported 0 P0, 10 P1, and one P2, including MAX_PATH temporary names,
  replace-based scientific publication, descriptor handoff windows, incomplete
  schema-v2 completion, error demotion, directive-root mismatch, and lease
  introspection/rebind. A later root still failed because downstream completion
  did not validate every versioned checkpoint, commit sidecar, and execution
  manifest. Its isolated allowlist is therefore explicitly expanded by
  `confirmatory_completion.py` and its filesystem-completion test; a new
  exact-root audit is mandatory.
- Dual Q/E closed its scoped native Windows file-ID, no-follow, environment,
  ancestor-swap, hardlink, and no-leak defects in isolated tests, but remains
  nonqualifying until it consumes the final private preflight/runner APIs,
  validates the four-role marker and Q pre-run root, and the terminal performs
  the complete post-run validation. No shared root or live pin exists.
- The old heartbeat automation `aanca-primary-to-option-b` is absent from the
  Codex application, so no polling automation competes with the qualified,
  unarmed event-driven Windows supervisor. No `jobs/` directory exists and no
  real AANCA training, primary, confirmatory, recovery, publication, or
  supervisor process is running.
- The user-authorized single future Q replacement-v2 write and independent
  verification remain unconsumed. No live integration, Q/E, scientific run,
  registry, frozen science, raw data, result, M9 package, ranking, or expert
  review write occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public-Q verifier immediately before this entry exited 0 with
  marker
  `Q_POST_SEMANTIC_V10_REJECT_AUTHORIZATION_MARKER_DOC_FINAL_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_INTERFACE_REJECTIONS_V12_CANDIDATES_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - finish independent roots before any live integration

Obtain independent verdicts for semantic-v12, runtime-v4, and M9-guard-v3 while
the isolated preflight/dual/terminal/runner chain closes its listed P0/P1
findings under the approved interface. Integrate only zero-P0/zero-P1 exact
allowlists, then run full live pytest/Ruff/format/mypy, retained-exit-code
PanNuke validation, lifecycle rehearsal, and capacity before consuming Q.

## 2026-07-30 - Semantic-v12 rejected; remaining launch chain pivots to a sealed capsule

- A fresh read-only auditor reproduced the semantic-v12 root twice in Python
  and twice in PowerShell:
  `106ab9fcadae4005e6806455e0054869561f1c54fd4e9fc0e37ce18057e75863`,
  13 files, seven directories, 1,575,657 bytes, and a 1,391-byte preimage.
  Topology was clean: unique physical file IDs, link count one, and no extra
  ADS, reparse point, cache, or alias.
- The audit then early-stopped with **0 P0 / 1 P1 / 0 P2**. After native
  `CloseHandle` returned a receipt with `entered=true` and `succeeded=true`, an
  injected caller-side fault before outcome/field clearing left the closed Job
  numeric slot recorded as active with no quarantine. An unrelated Job reused
  the exact slot; retry reported closed while four handles remained and the
  machine-global singleton could not be reacquired. The result reproduced
  twice. Per policy, the auditor correctly withheld the full 87+209 rerun.
- V10, v11, and v12 remain byte-identical rejected QA evidence. No v13 will be
  created. The nested Python Job/handle semantic guard is redundant with and
  less reliable than the external supervisor's native Job ownership and
  process-handle wait.
- `PLAN.md` now contains an append-only operational amendment selecting one
  deterministic content-addressed fresh-process execution capsule. Q will bind
  the whole capsule and internal manifest; E will bind one exact launch. The
  capsule directly validates Q/E and runs preflight plus runner with no public
  capability, lease, callback, or mutable project import.
- The same capsule will provide separate `verify-preterminal` and
  `verify-terminal` modes. The preterminal verifier runs as the supervisor's
  distinct integrity verifier, writes an O_EXCL pin before the supervisor
  terminal exists, and emits a canonical stdout summary. The public terminal
  verifier later composes that pin with the authenticated supervisor terminal,
  avoiding a self/future-hash cycle and any live-producer dependency.
- A new external supervisor-v2 release is required because the current
  preserved release copies ambient environment and generates a separate
  32-hex verifier nonce. V2 will launch the supervisor with the existing exact
  supervisor environment and both science/verifier with the same existing
  exact child environment and sole 64-hex nonce. It will bind the three process
  roles' expected and observed hashes. The old release and launcher remain
  read-only and unarmed.
- Runner-core checkpoint/completion hardening remains applicable and continues
  in its expanded isolated scope. M9-guard-v3 remains a separate future audit
  candidate. The former in-process preflight/dual/terminal candidates are
  preserved as nonqualifying evidence; reusable native identity, no-follow,
  environment, and canonicalization fixes may move into the capsule readers.
- No live integration, Q/E, supervisor job, scientific process, run, registry,
  frozen science, raw-data mutation, result, M9 execution/package, ranking, or
  expert-review write occurred. The authorized one future Q write and
  independent verification remain unconsumed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public-Q verifier immediately before this entry exited 0 with
  marker
  `Q_POST_INTERFACE_REJECTIONS_V12_CANDIDATES_MARKER_DOC_FINAL_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_SEALED_CAPSULE_PIVOT_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.

## Next exact action - build the deterministic capsule and supervisor-v2 in isolation

Freeze the capsule contract and minimal allowlist, build it twice
byte-identically, and close its source/import/Q/E/preflight/runner/terminal
tests while runner completion and supervisor-v2 proceed in parallel. Do not
integrate or consume Q until both independent exact-root audits pass.

## 2026-07-30 - M9-guard-v3 independently qualified but remains locked

- A fresh independent read-only audit returned **QUALIFIED / 0 P0 / 0 P1**
  for the immutable M9-guard-v3 candidate at
  `C:\Users\NATAN\Documents\AANCA_m9_guard_v3_candidate_20260730_f6040f05`.
  Python and PowerShell independently reproduced exact root
  `f6040f057e0704f2d3fdfd436aa4bff6b647fabedac71de8274173966cbb154c`:
  nine files, 386,096 bytes, a 1,012-byte
  `path\0decimal_size\0lower_sha256\n` preimage, and five directories.
- All nine files remained read-only and single-link. Two physical reads found
  no reparse point, symlink, alternate data stream, duplicate physical file
  ID, hardlink, case collision, cache, or candidate mutation. The real config
  SHA-256 was
  `4bfe26c15e326387f301eff2d78bf10c20eab6f37458f44263211808af269009`;
  it selected exactly one semantic-sidecar authority record and retained
  `cache_file_sha256=null`.
- A disposable overlay outside the read-only candidate passed **44 focused**
  and **94 impacted** tests, `ruff check .`, `ruff format --check .` over 180
  files, configured mypy over 90 source files, and compileall. The first full
  rerun used an excessively long external `--basetemp` and encountered only
  Windows `MAX_PATH` infrastructure failures; the complete rerun under
  `C:\m9v3f604` passed. Both/neither cache-authority branches, cache/sidecar
  tampering, stale digest, wrong selection, and duplicate records fail closed.
- This qualification prepares only the future M9 guard. It does not integrate
  or execute M9: M9 remains locked until a qualifying original confirmatory
  run reaches `CONFIRMATORY_COMPLETE`. No ranking, review package, outcome
  inspection, Q/E write, live integration, or scientific process occurred.
- The sealed-capsule design review now has **0 open P0 and 1 open P1**. The old
  fixed 57/63 source list is withdrawn; the builder will enumerate every
  regular `.py` under the exact final frozen `src/histo_audit/**` tree, strip
  `src/` for archive names, and add only the exact bootstrap, policy, entry
  contract, and manifest members. The sole open P1 is the still-unfrozen
  finite closed-key post-wake `verify-terminal`/composed-receipt contract.
- Supervisor-v2 and runner work remain synthetic-only. The new supervisor
  passed two short exact-environment success/wake trials, and runner
  filesystem-completion tests passed 31 cases. One full `pytest -q` process is
  currently progressing with CPU activity; it is a test process, not training
  or an AANCA scientific run, and it has not been interrupted.
- The historical public-Q verifier immediately before this entry exited 0
  with marker `Q_PRE_M9_V3_INDEPENDENT_QUALIFICATION_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The authorized one future Q replacement-v2 write and its one independent
  verification remain unconsumed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_M9_V3_INDEPENDENT_QUALIFICATION_DOC_APPEND_PASS`, the exact expected
  three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The final read-only historical public-Q verifier after recording that marker
  exited 0 with marker
  `Q_POST_M9_V3_INDEPENDENT_QUALIFICATION_MARKER_DOC_FINAL_PASS`, the same
  exact envelope, size, and SHA-256.

## Next exact action - close the terminal contract and freeze the capsule tree

Freeze and test the exact post-wake composed-terminal schema while the
supervisor-v2 and direct runner finish their isolated gates. Then build the
whole-package capsule twice from the one frozen final source tree and obtain a
fresh zero-P0/zero-P1 audit before any live integration or Q write.

## 2026-07-30 - Post-wake terminal inputs require retained native leases

- Terminal-contract review found a new P1 before schema freeze: hashing the
  verifier log and terminal receipt without retaining their identities would
  permit a filesystem substitution between supervisor publication and the
  post-wake `verify-terminal` process. A descriptive read-only flag is not
  sufficient.
- Supervisor-v2 must retain native, no-follow handles for the preterminal pin,
  verifier stdout, verifier stderr, and supervisor terminal receipt through
  the blocking exact-session Codex wake and post-wake capsule verification.
  Completed files are regular, read-only, single-link, ADS-free, exactly
  hashed/identified, and shared for read only so write and delete remain
  denied.
- The supervisor retains its stdout/stderr parent handles continuously from
  create-new through verifier exit; only bounded inheritable duplicates reach
  the verifier. The secured canonical stdout summary binds the pin's full
  physical identity before the supervisor opens and retains that pin. The
  supervisor terminal receipt is also create-new, fsynced, read back, made
  read-only, and retained without an unguarded pathname window.
- No-delete directory handles cover the exact supervisor-root-to-job-directory
  chain, preventing ancestor rename or replacement while still permitting the
  one composed-terminal create-new publication. Q/E bind the paths and lease
  policy; the terminal binds the observed identities and roots.
- Loss of the supervisor or any retained handle before composition and wake
  completion is an unambiguous `STOP` and one diagnostic wake. It never
  triggers composition from unguarded paths, scientific retry, or publication
  retry.
- The runner full gate also identified one separate source-candidate P1:
  changing `models/cnn.py` altered frozen model provenance. The accepted repair
  is to restore that file byte-identically at 74,851 bytes and SHA-256
  `9fcec232df28345be7460bfdeba96844ab78d48bb0f2860949d74b088cecb4bf`,
  retain its existing checkpoint API, and move operational immutable
  checkpoint publication outside the scientific model. Frozen config hashes
  will not be changed.
- Preserve the existing resume scanner and its canonical per-fold checkpoint
  path. At successful fold completion the operational runner copies the held,
  verified canonical checkpoint with O_EXCL semantics to one versioned
  checkpoint plus commit, then makes the canonical file read-only. No hardlink,
  replacement, cleanup, or resume-scanner rewrite is permitted. Both physical
  copies and the mandatory additional 10 GiB are included in the later sealed
  capacity calculation; current free capacity is sufficient for this safer
  reduced-code path.
- The historical public-Q verifier immediately before this entry exited 0
  with marker `Q_PRE_POSTWAKE_INPUT_LEASE_DECISION_DOC_APPEND_PASS`, the exact
  expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No Q/E/live/scientific write or process occurred. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, M9 remains
  locked, and the one authorized future Q-v2 write remains unconsumed.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_POSTWAKE_INPUT_LEASE_DECISION_DOC_APPEND_PASS`, the exact expected
  envelope, size, and SHA-256. The final verifier after recording this marker
  exited 0 with marker
  `Q_POST_POSTWAKE_INPUT_LEASE_DECISION_MARKER_DOC_FINAL_PASS`.

## Next exact action - implement and test the retained terminal leases

Freeze the finite composed-terminal schema together with these retained input
and ancestor lease fields, pass the synthetic substitution/restart matrix, and
independently audit the repaired runner root before freezing the whole capsule
source tree.

## 2026-07-30 - Pin handoff is overlap-based and terminal composition is acyclic

- Independent review rejected a verifier-exit-to-supervisor-open gap for the
  preterminal pin. Matching bytes and file ID after a close are insufficient
  against delete/recreate and ID-reuse races.
- The approved preterminal protocol changes no scientific argv or environment:
  the verifier keeps its native create-new pin handle open, writes exactly one
  bounded canonical READY/summary line to a supervisor-owned stdout pipe, and
  blocks on a supervisor-owned stdin ACK pipe. The supervisor persists those
  exact stdout bytes through its retained log handle, opens and verifies the
  same pin identity while the child handle is still live, and sends ACK only
  after the overlap is proven. The child validates ACK and only then exits.
  Stderr remains an exact-empty continuously retained file. All inherited
  handles are restricted explicitly; no polling is used.
- The exact terminal dependency order is `P -> T -> L -> C`: preterminal pin
  `P`; supervisor terminal `T`; deterministic after-terminal live-handle
  receipt `L`; composed public receipt `C`. `T` may bind final P/stdout/stderr
  evidence but only pre-serialization identity/handle facts about itself.
  `L` binds final P/stdout/stderr/T identities and hashes plus every retained
  handle slot, but excludes its own future size/hash. No artifact contains or
  predicts its own whole-file SHA.
- During post-wake verification the capsule opens the deterministic L path,
  verifies the exact still-live supervisor PID/creation/boot identity, and
  duplicates L/P/stdout/stderr/T handles from that process. It proves the
  pathname L bytes and every listed file equal the still-retained objects,
  rehashes them, and only then creates C with O_EXCL semantics. C binds the
  final L hash and all preceding roots.
- Any missing READY/ACK, extra stdout byte, nonempty stderr, early exit,
  timeout, failed handle duplication, PID/boot mismatch, replacement, or lost
  supervisor becomes `STOP` with no retry. The composition verifier performs
  no training, tuning, outcome selection, or scientific publication.
- The capsule builder's synthetic deterministic gate currently passes
  **12 tests** including byte-identical independent builds, exact stored-ZIP
  metadata/manifest checks, delete-each-required-member failure, and unsafe
  path negatives. It has not built a production capsule; import-origin and
  final Q/E/terminal integration tests remain pending the frozen source tree.
- The historical public-Q verifier immediately before this entry exited 0
  with marker `Q_PRE_OVERLAP_TERMINAL_CHAIN_DECISION_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No Q/E/live/scientific write or process occurred. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, M9 remains
  locked, and the one authorized Q-v2 write remains unconsumed.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_OVERLAP_TERMINAL_CHAIN_DECISION_DOC_APPEND_PASS`. The final verifier
  after recording this marker exited 0 with marker
  `Q_POST_OVERLAP_TERMINAL_CHAIN_DECISION_MARKER_DOC_FINAL_PASS`; both returned
  the same exact envelope, size, and SHA-256.

## Next exact action - finish finite terminal fields and negative handoff tests

Implement the READY/ACK and retained-handle chain in supervisor-v2 and the
capsule authority, freeze their exact closed field sets, and require the
independent reviewer to report zero P0/P1 before the final whole-package build.

## 2026-07-30 - Bootstrap audit rejected; Q/E and import anchoring remain mandatory

- The isolated deterministic builder reached **30 passed** with Ruff check,
  Ruff format-check, strict mypy over builder/bootstrap, and py_compile all
  passing. Hardening now rejects manifest casefold aliases, unknown roles,
  relative or parent paths, duplicate physical identities, hardlinks,
  symlinks, Windows junctions, source-replacement races, and local ZIP-header
  disagreement. It holds source files through final lexical/ADS readback and
  validates exact local, central, and EOCD ZIP structure.
- A different fresh auditor nevertheless returned **REJECTED / 0 P0 / 2 P1**
  for the bootstrap design, plus one P2. The audit correctly did not qualify a
  root. It inspected the mutable WIP at
  `C:\Users\NATAN\Documents\AANCA_original_confirmatory_capsule_v1_working_20260730`;
  no frozen snapshot or retrospective exact-root claim exists.
- Exact rejected-snapshot hashes recorded before repair mutation were:
  - `capsule_builder.py`:
    `92fa9d8a2ee023b3be94a143c7151b474810618d26c7b9b1c256babda29635e4`;
  - `capsule_bootstrap.py`:
    `9577e834ea36ad8aeace377324a2f65c38144fd6c347639a36608793edf4840e`;
  - `capsule_policy.json`:
    `a64808c964ffce7eea8b3ee7b52244a5804d45b580a6a764fc78741581ace386`;
  - intentionally incomplete/STOP `entry_contract.json`:
    `8003dd488e708712972561d21f46220f2821535b0de33e28b8d7583946f4ba64`;
  - bootstrap test:
    `6bb6924f91e87b15bc487053cd5a681ac03600ff587d18f3ff9dceb716400b0`;
  - hardening test:
    `4e4aeba3fd479df4ecd59c655a6ffa1ec097a516420e8f1c262bb919ad92ae62`.
  The auditor's focused suite passed 20 tests; the P1 findings were
  architectural, not ordinary test failures.
- P1-1: a self-consistent content-addressed capsule could dispatch before
  proving the exact authorized Q/E, project root, capsule path, manifest,
  policy, entry contract, interpreter, plan, and supervisor roots. Repair
  requires a minimal stdlib-only held Q/E anchor before importing any
  `histo_audit` module, followed by the full canonical authority reader before
  any dataset/model/cache/checkpoint/output action.
- P1-2: project-origin checking was post-hoc while mutable import hooks,
  importer caches, or package paths could act first. Repair requires an exact
  sealed builtin/frozen/zip/file finder set, cleared importer cache, exact
  manifest-to-module mapping, and capsule-only package `__path__` and
  submodule search locations. The origin guard runs before dispatch, after
  dispatch, and in `finally` on exceptions or non-integer returns.
- Terminal design is now the acyclic `P -> T -> L -> C -> R` chain. Pin P uses
  a live READY/ACK handle overlap. T binds final overlap, pin, stdout/stderr,
  process, and supervisor evidence without hashing itself. L binds retained
  P/stdout/stderr/T handles without its own future hash. C claims its real
  O_EXCL read-only path before terminal input reads and duplicates the same
  handle into the live supervisor. The already Q/E-bound supervisor validates
  C event-driven during the exact Codex wake and emits retained readback R
  before ACK; no fourth capsule mode or second C invocation exists.
  Handshake-receipt and wake-intent paths are not direct C inputs; their
  required facts are bound through retained T and the live supervisor actor.
- Review also confirmed that the new authority WIP still lacked top-level Q/E
  persistence. Its owner is adding a separate new Q replacement-v2 path and
  closed CREATE_NEW writer/reader/independent verifier, plus an E intent whose
  run/preterminal/terminal/custody commands and roots are exact. Only
  `run-confirmatory` creates the consume-once tombstone, before scientific
  input access; later verifier phases read the same consumed E without
  consuming or rearming it. The historical Q remains untouched.
- Runner repair restored frozen `models/cnn.py` byte-identically and is moving
  all checkpoint hardening outside the model. Native no-follow leaf handles,
  no-gap read-only transition, retained no-delete ancestor-directory handles,
  O_EXCL versioned copies, and no cleanup of acquired scientific namespaces
  are being tested before a new immutable audit candidate is declared.
- The user reconfirmed exactly one future Q replacement-v2 write and one
  independent verification, and authorized the remaining qualified project
  processes to proceed without an additional conversational pause. This
  permits creation/consumption of one exact E and one supervisor-bound
  confirmatory attempt only after all prescribed gates pass. It does not
  authorize any automatic retry, overwrite, adoption, authority reuse, result
  inspection for tuning, or skipped validation.
- The historical public-Q verifier immediately before this entry exited 0
  with marker `Q_PRE_CAPSULE_BOOTSTRAP_REJECTION_HANDOFF_DOC_APPEND_PASS`, the
  exact expected three-key envelope, 21,274 bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- No production capsule, Q/E, supervisor job, training, scientific run,
  ranking, raw-data change, or status transition occurred. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.
- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker
  `Q_POST_CAPSULE_BOOTSTRAP_REJECTION_HANDOFF_DOC_APPEND_PASS`. The final
  verifier after recording this marker exited 0 with marker
  `Q_POST_CAPSULE_BOOTSTRAP_REJECTION_HANDOFF_MARKER_DOC_FINAL_PASS`; both
  returned the same exact envelope, size, and SHA-256.

## Next exact action - repair pre-import authority and freeze isolated contracts

Close the bootstrap Q/E anchor and project-import finder before obtaining a new
fresh audit. In parallel, finish/gate the top-level Q/E authority, terminal
entry/handlers, supervisor-v2 custody actor, and native checkpoint runner.
Only zero-P0/zero-P1 immutable candidates may enter the final integration tree.

## 2026-07-30 - Freeze the acyclic single-file E command derivation

- A read-only process inventory found no live AANCA training, primary,
  confirmatory, recovery, publication, or armed supervisor process. The only
  matching process was the inventory command itself. Isolated capsule,
  authority, runner, and supervisor-v2 implementation work continues in
  parallel; this is active integration work, not a wait loop.
- Supervisor-v2's isolated synthetic suite currently passes **77 tests plus 3
  subtests** for the preterminal READY/ACK overlap and retained read-only
  handle. Terminal C/R custody, same-Job membership, exact one-shot wake, and
  STOP negatives remain under implementation. No production install, arm, or
  scientific command occurred.
- Reject a self-referential E command hash. A canonical E file cannot contain
  an exact argv/hash that itself contains `--e-intent-sha256` equal to the
  final hash of those same E bytes.
- Freeze one E file, not a two-file authority. E contains finite exact
  per-mode command projections: mode, executable, capsule, cwd, ordered common
  fields except the self file hash, fixed suffix paths, and one closed
  derivation-policy identifier. After E is CREATE_NEW-sealed, the authority
  inserts E's final file SHA-256 and core SHA-256 at their fixed argv
  positions and derives the only permitted exact commands.
- The CREATE_NEW/read-only supervisor job spec is the sole persistent carrier
  of each final argv and command SHA-256. It directly binds E path, file hash,
  and core hash; it is independently re-derived and checked from Q/E before
  launch and after `WaitForExit`. Q binds the command-derivation contract and
  supervisor release, never a future self-dependent command hash. Any extra
  argv or environment entry fails closed.
- The user's authorization for exactly one future Q replacement-v2 write and
  one independent verification remains unconsumed. No production capsule,
  Q/E, supervisor job, scientific run, ranking, raw-data change, or status
  transition occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public-Q readback immediately before this entry was 21,274
  bytes with unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_E_COMMAND_DAG_DECISION_DOC_APPEND_PASS`.

## Next exact action - qualify the four isolated handoff inputs

Finish and independently gate the repaired bootstrap, top-level Q/E authority
and entry/terminal handlers, native checkpoint runner, and supervisor-v2 C/R
custody actor. Integrate only immutable zero-P0/zero-P1 candidates, then run
the complete live QA/CLI/PanNuke gates before consuming the single Q write.

- The mandatory read-only historical public-Q verifier after this append
  exited 0 with marker `Q_POST_E_COMMAND_DAG_DECISION_DOC_APPEND_PASS`, 21,274
  bytes, and unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The final read-only verifier after recording that marker exited 0 with
  marker `Q_POST_E_COMMAND_DAG_DECISION_MARKER_DOC_FINAL_PASS` and returned
  the same exact size and SHA-256.

## 2026-07-30 - Reject the draft Q writer and amend the first-live Q-v2 custody schema

- Independent read-only review found that the mutable draft Q publisher
  created/adopted parent directories, used generic `os.open`, closed the file,
  and only then changed it to read-only. Its reader also closed the no-follow
  handle before ADS, canonical-root, and self-path checks. These are concrete
  publication/readback TOCTOU failures. No production Q-v2 was written.
- Reject the unpublished 10-key Q-v2 draft. The first-live Q-v2 remains schema
  version 2 but has exactly two additional top-level custody fields:
  `publication_ancestor_lease` and
  `publication_ancestor_lease_root_sha256` (**12 exact top-level keys total**).
  The lease binds the already-existing
  `PROJECT_ROOT -> artifacts -> resource_control` chain. This is an explicit
  pre-publication technical amendment, not silent schema drift.
- The replacement publisher must not create or adopt an ancestor. It uses
  native CREATE_NEW/no-follow/read-only-at-create semantics, retains the exact
  leaf and no-delete ancestor handles through write, flush, hash, ADS,
  single-link, physical-identity, canonical-root, self-path, and independent
  readback checks, and leaves every partial claim as permanent STOP evidence.
  There is no path `chmod`, cleanup, overwrite, adoption, or retry.
- Remove `supervisor_spec_sha256` from every upstream E/Q object. The
  downstream supervisor spec necessarily contains final argv with E's final
  file SHA-256, so putting that spec hash into E would recreate an impossible
  cycle. E binds the deterministic spec path/schema/policy and supervisor
  release; after E is sealed, the CREATE_NEW spec binds E path/file/core and
  the exact derived commands and hashes.
- Interpreter custody is also exact for this project: the ancestor lease is
  anchored at Q's project root and must cover
  `PROJECT_ROOT -> PROJECT_ROOT/.venv -> PROJECT_ROOT/.venv/Scripts` before
  the final `python.exe` leaf. The supervisor retains every native handle
  continuously through each phase launch. A one-record self-selected
  `Scripts` lease is rejected because it leaves `.venv` replaceable.
- Runner gates after native checkpoint/ancestor repair currently include
  **119 passed** across the impacted seven-file suite, **33 passed** in the
  full filesystem-completion suite, **46 passed** for image OOF/native
  canaries, **36 passed** for confirmatory core, and the pinned provenance test
  1/1. Full-source mypy reports no issues in 93 source files. The frozen
  `models/cnn.py` remains 74,851 bytes with SHA-256
  `9fcec232df28345be7460bfdeba96844ab78d48bb0f2860949d74b088cecb4bf`.
  Known inherited Ruff/format failures in resume/preflight are being repaired
  mechanically before any candidate freeze; mandatory global gates are not
  waived.
- The historical public-Q readback immediately before this entry was 21,274
  bytes with unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_QV2_CUSTODY_DAG_AMENDMENT_DOC_APPEND_PASS`.
- The user's exactly-one future Q-v2 publication and one independent
  verification remain unconsumed. No production capsule, Q/E, supervisor job,
  science, training, ranking, raw-data change, or completion-stage transition
  occurred. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - close the amended schemas and obtain immutable candidates

Require exact 12-key Q agreement across authority/bootstrap/tests, finish the
single-file E builder/reader and downstream command-spec carrier, enforce the
full interpreter ancestor chain, and pass the complete runner and
supervisor-v2 gates. Only then freeze and independently audit candidates for
live integration.

- The mandatory read-only historical public-Q verifier after this amendment
  exited 0 with marker
  `Q_POST_QV2_CUSTODY_DAG_AMENDMENT_DOC_APPEND_PASS`, 21,274 bytes, and
  unchanged SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`.
- The final read-only verifier after recording that marker exited 0 with
  marker `Q_POST_QV2_CUSTODY_DAG_AMENDMENT_MARKER_DOC_FINAL_PASS` and returned
  the same exact size and SHA-256.

## 2026-07-30 - Freeze the capsule core and close the Q/E custody architecture

- A read-only process inventory found no live AANCA training, primary,
  confirmatory, recovery, publication, or armed supervisor process. No
  scientific process was interrupted or started.
- The isolated capsule/bootstrap core reached a byte-stable seven-file
  handoff at
  `C:\Users\NATAN\Documents\AANCA_original_confirmatory_capsule_v1_working_20260730`.
  Its exact allowlist is:
  - `CAPSULE_DESIGN.md`: 6,609 bytes, SHA-256
    `0c8ec247b9f6d88950b3800de7d69e1c8dbec65881bde9c90a883613afda1851`;
  - `capsule_bootstrap.py`: 120,820 bytes, SHA-256
    `0aa8e893ad3ad76154fb046a625c82d67772852bb5bc0749781a45059691cedb`;
  - `capsule_builder.py`: 46,472 bytes, SHA-256
    `dd436f2255eb735ffab2db26cf016149512651ae0fc17a8fa7108e0acffbf85e`;
  - `capsule_policy.json`: 306 bytes, SHA-256
    `a64808c964ffce7eea8b3ee7b52244a5804d45b580a6a764fc78741581ace386`;
  - `entry_contract.json`: 246 bytes, SHA-256
    `8003dd488e708712972561d21f46220f2821535b0de33e28b8d7583946f4ba64`;
  - `tests/test_capsule_builder.py`: 76,215 bytes, SHA-256
    `6e6a5e2b0dba122ef034c2191406617dba042e901a7757df73a5bc1685c7057a`;
  - `tests/test_capsule_builder_hardening.py`: 17,475 bytes, SHA-256
    `3afee36c251d687cf2388cbc2c83807ae41ef9a20d3fc22efea98db9f6c3581a`.
- The exact handoff root is
  `57daa7cab159ba9ff1868372d551fc45caba4cf0f7e06ec4ae608162690dd459`.
  It is SHA-256 over the 661-byte, listed-order preimage
  `path ASCII + NUL + decimal size + NUL + lowercase SHA-256 + LF`.
- On those exact bytes, from the isolated WIP root:
  - `C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .test-tmp-schema-final tests`
    passed **64/64 in 4.35 s**;
  - the explicit 11-test deterministic-build, Q/E-schema, invalid-anchor,
    invalid-capsule/Q/command, one-use claim, dispatcher-take, and verifier
    consumed-claim selection passed **11/11 in 1.74 s** using
    `--basetemp .test-tmp-handoff-final`;
  - exact four-file `ruff check` passed, `ruff format --check` reported four
    files already formatted, strict two-source mypy reported no issues, and
    four-file `py_compile` exited 0.
- Bootstrap and current authority have exact AST field parity: Q 12/12, E
  20/20, scientific authority 13/13, static runner binding 20/20, and
  per-attempt scientific request 25/25. The E consume-once claim is now
  unavailable until the held capsule and complete stdlib-only Q/E/command/spec
  prevalidation succeeds. Invalid capsule, Q, E, or command inputs leave no
  claim; a valid mode must take the claim exactly once.
- This is a **core handoff, not an independently qualified production
  capsule**. `entry_contract.json` deliberately remains
  `incomplete_fail_closed_pending_terminal_composed_receipt_v1`; it cannot be
  promoted until the scientific, preterminal, and terminal handlers are real,
  integrated, and a fresh immutable zero-P0/zero-P1 audit passes.
- Close the Q/E custody fork as follows. A short sealed authority controller,
  not supervisor-v2, performs the single authorized Q publication, exactly one
  independent transition/readback with continuous leaf-and-ancestor overlap,
  and the one-use E/spec publication. It starts the exact supervisor-v2 and
  transfers the retained Q leaf/full ancestor chain and E leaf/ancestor chain
  by `DuplicateHandle` over one bounded anonymous-pipe READY/ACK handshake.
  Supervisor-v2 validates the downstream spec, controller/supervisor process
  identities, access/share masks, physical identities, hashes, read-only
  state, ADS/link constraints, and exact seed reconstruction; it writes a
  retained custody receipt before ACK. Scientific launch is forbidden before
  ACK. The controller closes its leases only after ACK. There is no supervisor
  self-publication, adoption, overwrite, cleanup, or retry.
- The interpreter evidence has two deliberately different API views. External
  `QueryFullProcessImageNameW`/CIM process identity is the logical venv
  launcher
  `C:\Users\NATAN\Documents\AANCA\.venv\Scripts\python.exe`
  (274,712 bytes, SHA-256
  `864530d708039551a2c672ddd65e5900fbc08b0981479679723a5b468f8082bc`).
  Inside that process, `GetModuleFileNameW(NULL)`, `sys._base_executable`, and
  `sys.orig_argv[0]` identify the retained base runtime
  `C:\Users\NATAN\AppData\Local\Programs\Python\Python312\python.exe`
  (103,192 bytes, SHA-256
  `15b41a488c356c0e331facdea6c836a6cec021f12d5fde9844e7ca4a1aa0361a`).
  Q/E/supervisor evidence must preserve both roles and must not compare the
  result of one API with the expected result of the other.
- A provisional moving-root compatibility audit correctly reports open
  integration blockers rather than qualifications: the scientific and two
  terminal handlers are not yet all present; bootstrap claim arm/take has no
  integrated handler call site; supervisor-v2 has not yet consumed the new Q/E
  custody handoff; and its declared local custody pipe is not yet wired through
  the full `P -> T -> L -> C -> R` path. These are assigned implementation
  items and block production Q, E, capsule publication, supervisor arming, and
  science.
- The historical public-Q readback immediately before this entry remained
  21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_CAPSULE_CORE_AND_CUSTODY_DECISION_DOC_APPEND_PASS`.
- The user's exactly-one future Q-v2 publication and one independent
  verification remain unconsumed. No production capsule, Q/E, supervisor job,
  training, scientific result, ranking, raw-data change, or completion-stage
  transition occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish the three handlers and supervisor custody actor

Finish and gate the high-level full-lifecycle scientific handler, the exact
preterminal/terminal `P -> T -> L -> C -> R` module, and supervisor-v2's Q/E
custody plus terminal duplex. Then build a new integrated immutable capsule
candidate with a ready entry contract and obtain a fresh zero-P0/zero-P1 audit
before any live Q-v2 write.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_POST_CAPSULE_CORE_AND_CUSTODY_DECISION_DOC_APPEND_PASS`.

## 2026-07-30 - Make successor checkpoint import acyclic with RunTracker

- A compatibility review found that the unpublished successor E projection
  attempted to bind destination checkpoint physical identities and a copy
  receipt before launch. Those files require the successor run directory to
  exist, while the canonical `RunTracker.start` correctly requires the new run
  directory not to exist. Precreating and later attaching/adopting that
  directory is rejected.
- Freeze a two-phase successor import contract. Before launch, E binds the
  qualified predecessor snapshot/root, exact source checkpoint physical
  identities and hashes, the complete allowlist/actions/configuration/split/
  optimizer/AMP/RNG/early-stopping state, exact deterministic destination
  relative paths, copy policy, and `retry_of_run_id`. It records that the copy
  is not yet performed and contains no future destination file IDs or future
  copy-receipt hash.
- Only after E-consumption ACK does the normal full lifecycle call
  `RunTracker.start` to create the previously absent successor directory. It
  then copies only the E-authorized sources to the exact destinations with
  O_EXCL/no-hardlink semantics, validates non-aliasing and all bytes/state,
  writes the post-copy receipt, builds the registered execution contract with
  actual destination identities, and checks it against the E pre-copy
  authority before any matrix fit.
- A partial import remains in the failed run as STOP evidence. There is no
  cleanup, attach, adoption, autodiscovery, fallback to fresh, or automatic
  retry. The fresh request and its exact-180 contract remain unchanged.
- This is an amendment to an unpublished technical E projection only. No Q/E,
  run directory, checkpoint copy, scientific process, or result was created;
  the exactly-one Q-v2 authorization remains unconsumed. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - implement and gate both typed execution branches

Complete the fresh/successor typed request union, the post-RunTracker successor
import receipt, the authority result binding, and interruption/corruption
tests. In parallel, finish supervisor Q/E custody and terminal duplex before
constructing the integrated immutable candidate.

- The mandatory read-only historical public-Q readback after this amendment
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_POST_SUCCESSOR_ACYCLIC_DECISION_DOC_APPEND_PASS`.

## 2026-07-30 - Integrate the gated fresh/successor runner allowlist

- The integration owner accepted only the exact 19-file code/test allowlist
  from
  `C:\Users\NATAN\Documents\AANCA_original_confirmatory_runner_core_v1_working_20260730`.
  Source size and SHA-256 were checked before copying and destination size and
  SHA-256 were checked after copying. Seven files were new and twelve replaced
  their older main-tree versions. No WIP copy of `PLAN.md`, `STATUS.md`,
  `DECISIONS.md`, data, cache, or test temporaries was integrated.
- The listed-order integration root is
  `6567f5f746090d5b9bbb230c1485baa2147276c179c83735ad2e923f2e7e8a88`.
  It is SHA-256 over the 2,289-byte preimage
  `forward-slash relative path ASCII + NUL + decimal size + NUL + lowercase
  SHA-256 + LF`.
- The runner now has an exact tagged execution union: unchanged fresh
  execution or
  `original_confirmatory_successor_precopy_projection_v1`. The successor
  authority binds the qualified predecessor and all 180 source
  checkpoint directives before launch, but materialization occurs only after
  canonical `RunTracker.start` creates the previously absent successor run
  directory. Copying is O_EXCL, physically independent, allowlisted, and
  read back before any fit. A partial copy remains failed STOP evidence; there
  is no attach, adoption, cleanup, autodiscovery, fresh fallback, or automatic
  retry.
- Windows checkpoint observation is role-bound. An ordinary observer requests
  read access with read share, a live-writer observer uses read plus write
  sharing without delete sharing, and only the explicit `DELETE_ON_CLOSE`
  owner-lock observer may advertise delete sharing. The frozen
  `src/histo_audit/models/cnn.py` remained 74,851 bytes with SHA-256
  `9fcec232df28345be7460bfdeba96844ab78d48bb0f2860949d74b088cecb4bf`.
- The first full focused WIP run exposed one stale source-inspection test:
  **227 passed, 1 failed**. The test incorrectly required the compatibility
  wrapper itself to contain the preflight call after the lifecycle moved behind
  the sealed capsule boundary. It was replaced with a functional order test
  proving that both the public wrapper and capsule path enter the same
  lifecycle and that initial preflight precedes lifecycle readiness, the live
  gate, input loading, the guarded final recheck, `RunTracker`, and matrix
  execution. The bounded regression then passed **1/1**, and the complete
  focused command passed **228/228 in 417.69 s**.
- On the exact integrated bytes in the isolated WIP, the explicit 19-file
  `ruff check` and `ruff format --check` commands passed, mypy passed for all
  nine source files, and `compileall` passed for all 19 files. Main-tree
  integrated full pytest/Ruff/format/mypy/CLI validation remains mandatory
  after the authority, terminal, capsule, and supervisor allowlists are also
  frozen and integrated; this entry does not claim that later combined gate.
- A diagnostic current-tree
  `.venv\Scripts\python.exe -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke` invocation exceeded the shell
  tool's 64-second window. Its launcher/base-runtime process pair
  (PID 12524/19216) continued consuming CPU and later disappeared without
  intervention, but the detached invocation retained neither stdout nor an
  observable exit code and changed no file under `artifacts` or `data`.
  Therefore it is explicitly **not** counted as the mandatory PanNuke gate; the
  final integrated validator must be rerun with retained logs and exit code.
- Real existing samples confirm Git ignore coverage for
  `data/raw/pannuke/fold_1.zip`, extracted
  `Fold 1/images/fold1/images.npy`,
  `data/manifests/pannuke/pannuke_nucleus_manifest.parquet`, and
  `artifacts/duplicate_audit/frozen_resnet18_duplicate_embeddings.npz`.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_RUNNER_ALLOWLIST_INTEGRATION_DOC_APPEND_PASS`.
- No Q-v2 file, E, supervisor job, run directory, checkpoint copy, training,
  scientific result, ranking, or raw-data modification was created. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - close terminal custody and integrate the complete capsule

Freeze the Q-bound outcome-blind terminal authority template, its exact E/spec
instance, supervisor Q/E receiver, and the live
`CLAIM_READY -> CUSTODY_GRANT -> COMPOSED_READY -> FINAL_ACK` path producing R.
Then integrate the exact authority/terminal/capsule allowlists and rerun the
combined mandatory gates before constructing or publishing any production
capsule or Q-v2.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_POST_RUNNER_ALLOWLIST_INTEGRATION_DOC_APPEND_PASS`.

## 2026-07-30 - Bind outcome-blind terminal artifact inspection before arm

- A read-only cross-WIP audit found that protected `expected_artifacts`
  previously accepted an arbitrary nonempty `json_equals` selector. The
  preterminal verifier could therefore be instructed to dereference a metric
  or other scientific outcome while emitting
  `outcome_values_read=false`. That contract is rejected and blocks capsule/Q
  publication.
- The replacement is one finite ordered control-only template with exactly
  seven roles: terminal seal, integrity receipt, completion evidence,
  integrity registry, stage-attestation registry, stage-attestation anchor,
  and disposition anchor. Only `run_id`, `status`, `completion_stage`, and the
  exact eligibility-control predicate `study_outcome_eligible` may be read.
  Selectors with dots, list/numeric indirection, aliases, metrics, rankings,
  predictions, effects, p-values, statistics, restoration values, or any
  unlisted field are rejected before bootstrap claim arm/take. Type comparison
  is strict, so JSON integer `1` cannot satisfy expected Boolean `true`. The
  four registry/anchor rules with empty checks are not JSON-decoded.
- The old reference implementation at
  `C:\Users\NATAN\Documents\AANCA_original_confirmatory_terminal_receipt_v4_working_20260730\src\histo_audit\workflows\original_confirmatory_terminal_prelaunch.py`
  (32,167 bytes, SHA-256
  `5a7bc8aa70f1635a16b99f431c9b3cfa2641f7c3e8b5fce34814984381aede66`)
  is explicitly unqualified WIP: its independent pins are null and no current
  Q/E or frozen plan binds its identity. It is not imported or trusted.
- Canonical authority instead owns a self-contained, self-hashed template.
  Q remains exactly 12 top-level fields and binds the template root inside its
  closed `supervisor_release`. E remains exactly 20 top-level fields and its
  closed per-job object contains the exact instantiated projection and root
  for the bound `run_id` and expected run directory. The downstream supervisor
  spec must contain a canonically identical instance. Bootstrap, full
  authority, terminal, and supervisor must all recompute the same
  `Q template -> E instance -> spec equality` chain before project import,
  protected artifact access, or irreversible claim arm/take.
- The authority, bootstrap, terminal, and supervisor WIPs are implementing and
  mutation-testing this contract. This entry records the fail-closed decision,
  not a completed combined gate. No Q-v2, E, capsule, job, run, or scientific
  output was created; formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`, M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_OUTCOME_BLIND_AUTHORITY_DECISION_DOC_APPEND_PASS`.

## Next exact action - finish the live terminal duplex and prove full parity

Complete the canonical template/instance implementation and the supervisor's
event-driven `P -> T -> L -> C -> R` call site, then run real subprocess
mutation tests and integrate only the newly frozen allowlists.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_POST_OUTCOME_BLIND_AUTHORITY_DECISION_DOC_APPEND_PASS`.

## 2026-07-30 - Independently qualify the integrated runner allowlist

- A read-only main-tree qualification reran the exact focused ten-test-file
  runner suite with the pytest cache disabled and a basetemp outside the
  repository. It passed **228/228 in 415.90 s** with process exit code 0
  (416.828 s independent stopwatch).
- On the same bytes, the exact 19-path `ruff check --no-cache` command passed,
  `ruff format --check` reported all 19 files already formatted, mypy reported
  no issues in the nine source files, and isolated `compileall -q -f` passed.
- The final main/WIP allowlist readback contained 19 files and 1,620,924 bytes.
  Both sides produced the same 2,289-byte canonical preimage and SHA-256 root
  `6567f5f746090d5b9bbb230c1485baa2147276c179c83735ad2e923f2e7e8a88`;
  every `(relative path, size, SHA-256)` record matched.
- The frozen `src/histo_audit/models/cnn.py` remained 74,851 bytes with
  SHA-256
  `9fcec232df28345be7460bfdeba96844ab78d48bb0f2860949d74b088cecb4bf`.
  The run-directory count remained 20, no Q-v2 content/path was present, and
  no data, source, documentation, run, or scientific artifact was written by
  the independent qualification.
- A four-thread read-only PanNuke raw inventory check found all 22 expected
  files, no extras, no missing files, and no hash mismatches. Its observed
  inventory root exactly matched
  `51634109b6a0c2db34e54c437fa950dc624d0cae458a84e4cf4f92f1e15478d0`.
- This focused independent result accepts the runner allowlist only. It is not
  the later combined mandatory gate: full pytest, full Ruff/format, the
  retained-output PanNuke validator, and functional CLI remain required after
  authority, terminal, capsule, and supervisor integration. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_RUNNER_INDEPENDENT_QA_DOC_APPEND_PASS`.

## Next exact action - eliminate terminal command self-attestation

Freeze a Q-bound, acyclic supervisor program/source/argv/cwd derivation and
make the supervisor and terminal independently reconstruct it. Then finish
the live duplex mutation matrix and integrate only exact frozen allowlists.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count remained zero; marker:
  `Q_POST_RUNNER_INDEPENDENT_QA_DOC_APPEND_PASS`.

## 2026-07-30 - Replace direct Codex-to-terminal launch with a sealed launcher

- An independent liveness audit proved that a terminal verifier launched
  directly by resumed Codex cannot be assumed to receive E's exact
  environment. Codex shell policy and session/tool variables can add or
  transform inherited values. The existing fail-closed environment checks
  would correctly reject that drift, but the direct-launch path is not
  qualified for unattended use.
- The replacement is one small Q-bound terminal-client launcher belonging to
  the external supervisor release, not a fourth capsule/verifier mode. The
  rendered Codex prompt launches that exact launcher once. With isolated
  interpreter flags, the launcher validates the sealed job/spec/E command
  projection, creates an O_EXCL per-job launch intent, and directly invokes
  exactly one existing `verify-terminal` child using a freshly constructed
  UTF-16 environment block, exact cwd, and exact argv. It imports no project
  code, forwards no inherited environment, does not break away from the wake
  Job, waits for the child, and has no retry or fallback path.
- The authority graph remains acyclic:
  `Q static launcher/template -> E per-job launcher and child projection ->
  sealed run spec -> T -> rendered prompt and launcher command in
  wake_intent`. The launcher command must never include the wake-intent hash.
  The child claim identifies its live parent launcher; the supervisor verifies
  both PEB identities, parentage, argv/cwd/environment, and same fresh Job.
- Prompt schema B is ASCII UTF-8/LF and has eight ordered placeholders:
  `job_id`, `supervisor_job_directory`, `supervisor_spec_path`,
  `supervisor_spec_sha256`, `terminal_receipt_sha256`,
  `terminal_client_launcher_argv_json`,
  `terminal_client_launcher_command_sha256`, and
  `verify_terminal_command_sha256`. It explicitly forbids direct child launch,
  discovery, alternate commands, fallback, and retry. Rendering occurs only
  after the final spec and T exist; braces, control characters, reordered,
  duplicate, injected, or unknown placeholders are rejected.
- The former single 60-second resume-to-ACK window is rejected. The frozen
  protocol now separates an event-driven 1,800,000 ms resume-to-CLAIM arrival
  bound from a 60,000 ms post-CLAIM custody-exchange bound, both within the
  overall six-hour Codex wake bound.
- Restart handling must CREATE_NEW a durable protected restart STOP. It may
  wake Codex for diagnosis exactly once only when no original job wake intent
  ever existed; an existing wake intent suppresses a second wake. It never
  reconstructs handles, continues science, or retries.
- The authority, supervisor, terminal, capsule, and handoff WIPs are being
  synchronized and mutation-tested against this decision. It is not yet an
  integrated gate. No Q-v2, E, production supervisor job, run, training,
  scientific output, or raw-data modification was created. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count was zero; marker:
  `Q_PRE_TERMINAL_CLIENT_LAUNCHER_DECISION_DOC_APPEND_PASS`.

## Next exact action - freeze and mutation-test the synchronized protocol

Finish the Q/E/spec/terminal/launcher closed schemas, independently validate
full P and P-to-T semantics before FINAL_ACK, regenerate the exact handoff
template and pins, and run real short subprocess tests without AANCA science.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count remained zero; marker:
  `Q_POST_TERMINAL_CLIENT_LAUNCHER_DECISION_DOC_APPEND_PASS`.

## 2026-07-30 - Correct Windows venv redirector process identity

- Two short non-scientific probes resolved a contradictory process-identity
  assumption. When invoked through
  `.venv\Scripts\python.exe` with `-I -S -B`, Python code observes the base
  CPython path through `QueryFullProcessImageNameW`,
  `GetModuleFileNameW(NULL)`, PEB/`CommandLineToArgvW`, and
  `sys._base_executable`, while `sys.executable` retains the venv path.
- A separate direct process-tree probe proved why both observations are true.
  CreateProcess returned a venv-redirector process whose image and command
  line used `.venv\Scripts\python.exe`; it created and waited for a distinct
  child whose image and command line used the base CPython runtime. Both
  exited naturally with code 0. No AANCA module, data, run, authority, or
  scientific command was touched.
- The venv launcher remains 23,552 bytes with SHA-256
  `864530d708039551a2c672ddd65e5900fbc08b0981479679723a5b468f8082bc`.
  The base runtime remains 103,192 bytes with SHA-256
  `15b41a488c356c0e331facdea6c836a6cec021f12d5fde9844e7ca4a1aa0361a`.
- The short stdlib-only supervisor and terminal-client launcher must therefore
  use the exact base runtime directly with `-I -S -B`. This makes the
  CreateProcess handle/PID the same process that receives transferred handles
  and runs the pipe server. The venv launcher and base runtime remain separately
  bound dependencies.
- The capsule still needs the venv path. Its executing Python process is
  therefore the base-runtime child of a venv redirector. For terminal C, the
  venv redirector is the immediate parent and the sealed terminal-client
  launcher is the grandparent. The custody contract must bind and live-verify
  both process identities, the exact grandparent chain, creation times/PEB
  values, and same-Job membership; it must not assert a false direct
  C-to-terminal-client-launcher parent relation.
- This is a proven Windows launch-semantics correction inside the already
  frozen implementation scope. It does not change the scientific plan and
  does not authorize execution. The authority/supervisor/terminal/capsule
  WIPs are synchronizing it before hash freeze.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count was zero; marker:
  `Q_PRE_WINDOWS_VENV_REDIRECTOR_CORRECTION_DOC_APPEND_PASS`.

## Next exact action - freeze the corrected two-level launch lineage

Bind the base-direct supervisor/launcher commands and the
launcher-to-redirector-to-C lineage, rerun real synthetic process/PEB/Job
tests, then resume the synchronized zero-P0/zero-P1 audit.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count remained zero; marker:
  `Q_POST_WINDOWS_VENV_REDIRECTOR_CORRECTION_DOC_APPEND_PASS`.

## 2026-07-30 - Freeze and independently qualify the authority interface

- The authority implementation is frozen at 563,522 bytes with SHA-256
  `6ed3c651a972c45cc057138e30555f23614ae994a20b01b77049cc061c3c0d23`.
  Its test file is 103,606 bytes with SHA-256
  `6ce77cff2d655e1a942aabc6c370473eb537709e8cf208010605f03d80ceee3f`.
  The unchanged entry and entry-test files remain respectively
  `7c50b6cac8fe6772514a58a2afd6a9d8d09c4e5038b5e0002ac3d3482c23fd68`
  and
  `76a5f47a15675cf2f806cfc20bbaff0078c904bf7f2c5140805c780d30cd795a`.
- Owner QA passed **59 focused tests** and **81 full WIP tests**, plus
  `ruff check .`, `ruff format --check .`, and mypy for the authority source.
- The frozen static terminal template root is
  `43b23fc71c17a52630de6a3b4f4e876e9805ffc5c9805c6eb67ffcf026a39b21`.
  Exact closed counts are CLAIM 40, GRANT 57, COMPOSED_READY 38, FINAL_ACK
  43, readback 37, launcher release 112, E launcher projection 88,
  CREATE_NEW launch intent 24, and final launcher command 14.
- Two independent read-only audits both returned **P0=0, P1=0, P2=0**.
  The first reconstructed the terminal command projection, closed E DAG,
  base-direct supervisor/launcher identities, and causal launch-intent
  custody. The second passed 35/35 independent static/AST checks covering the
  same graph plus the exact three-process Job/PEB lineage and prompt renderer.
  Both verified the authority size/SHA before and after without changing it.
- The prompt-B template remains 1,620 ASCII/LF bytes with SHA-256
  `cad312b3150cc2f470675caef8c6b4d0871076198362d96d836bdcf0ed3bb214`
  and exactly eight ordered placeholders. Q/E contain no future concrete
  spec/T/launcher-command hashes; the final launcher command is materialized
  only after E, spec, and T.
- This qualifies the authority component only. Supervisor, launcher,
  terminal, bootstrap, combined lifecycle, saved-session E2E, main-tree full
  gates, two capsule builds, and production publication remain separate
  mandatory gates. No Q-v2 or E was published and no real job or scientific
  process was launched. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count was zero; marker:
  `Q_PRE_AUTHORITY_INTERFACE_FREEZE_DOC_APPEND_PASS`.

## Next exact action - qualify bootstrap, terminal, launcher, and supervisor

Complete independent bootstrap audit, freeze the terminal and standalone
launcher allowlists, finish supervisor runtime custody and synthetic
saved-session E2E, then integrate exact qualified bytes into the candidate.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count remained zero; marker:
  `Q_POST_AUTHORITY_INTERFACE_FREEZE_DOC_APPEND_PASS`.

## 2026-07-30 - Freeze and independently qualify the terminal capsule

- The terminal implementation is frozen at 273,307 bytes with SHA-256
  `768aba256cd007ebc6e13851db44f9eeeddfa9f8930bc5f9f58eb678cf530631`.
  Its test file is frozen at SHA-256
  `05befb95a7ddb4dcf51a6c418561e51a2bc1a17fedf54250976ec7e9e31ce7f4`.
  Its bundled authority is byte-identical to the qualified
  `6ed3c651a972c45cc057138e30555f23614ae994a20b01b77049cc061c3c0d23`
  source.
- Owner QA passed **35 focused tests** and **116 complete scratch-bundle
  tests**, plus Ruff check, Ruff format-check, and strict mypy for the terminal
  source.
- An independent unchanged-byte audit returned **P0=0, P1=0, P2=0**. It
  confirmed exact 40/57/38/43/37 message parity and semantic use of every
  receiver field; L contains no future launch-intent field.
- C proves the complete live launcher-to-venv-redirector-to-runtime lineage
  before taking the E claim. It reads the terminal-client launch intent only
  after GRANT by duplicating the supervisor's retained source handle, then
  revalidates the same handle, path, bytes, full physical identity, root,
  slot, and access through COMPOSED_READY and FINAL_ACK.
- Strict P/T and zero-discard log checks, actual R-file readback, and one
  event-driven shared post-CLAIM deadline all passed the static audit. No
  success summary is emitted before final ACK, R, C, source, intent, process,
  and deadline revalidation.
- This qualifies terminal bytes only. They are not yet integrated into main;
  the standalone launcher and supervisor runtime E2E still gate integration.
  No production capsule, Q-v2, E, job, or scientific process was created.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count was zero; marker:
  `Q_PRE_TERMINAL_CAPSULE_FREEZE_DOC_APPEND_PASS`.

## Next exact action - close launcher/supervisor E2E and bootstrap test coverage

Finish the standalone launcher's native smoke and independent audit, complete
supervisor runtime custody against these frozen message sets, and add the
small missing bootstrap mutation tests without changing qualified source.

- The mandatory read-only historical public-Q readback after this append
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count remained zero; marker:
  `Q_POST_TERMINAL_CAPSULE_FREEZE_DOC_APPEND_PASS`.

## 2026-07-30 - Reject coherently substitutable launcher capsule inputs

- An independent adversarial launcher audit reproduced a complete coherent
  capsule substitution. Replacing the `verify-terminal` capsule path, hash,
  and argv in E, rebuilding the matching supervisor-spec capsule lease and
  roots, and supplying the recomputed CLI hashes caused the current working
  launcher to hash the substituted capsule, publish its launch intent, and
  create one child. No SHA-256 collision was required.
- This is a blocking **P0**, not an out-of-scope caller compromise. Frozen
  release112 authenticates the launcher source, base runtime, venv redirector,
  and runtime child, but it contains no independent terminal-capsule path or
  hash. CLI14 binds spec/E/T/projection/command/environment hashes and cwd, but
  those values can be recomputed as one coherent caller-supplied surface. E's
  Q metadata and self-roots therefore do not constitute non-forgeable capsule
  authority for the post-wake launcher.
- The final launcher must contain literal trusted values for the exact
  content-addressed production path
  `artifacts/execution_capsules/<whole_sha256>/original_confirmatory.pyz` and
  its whole-file SHA-256. Its already Q-bound and supervisor-held source
  identity then supplies the independent anchor. Before publishing a
  CREATE_NEW launch intent or calling `CreateProcessW`, it must compare those
  literals with E's terminal projection, the supervisor spec contract and
  capsule lease, the downstream command tail, and the live capsule bytes and
  identity. An unresolved template or any coherent substitution must STOP
  with zero intent and zero child.
- This changes build order but not frozen scientific policy or authority
  schemas: first integrate the qualified handlers and builder, pass full main
  gates, build the capsule twice with identical bytes, and publish it once at
  its content-addressed path; only then materialize and independently audit
  the final launcher, integrate those exact bytes into the supervisor, and run
  the synthetic saved-session E2E. Q-v2 remains later and one-shot.
- A second launcher audit also found a Windows FILETIME formatting mismatch
  for 4/10 fractional residues because one implementation truncated while the
  authority/supervisor rounded. This remains a blocking P1 until one exact
  conversion and residue tests agree across all components.
- No launcher or supervisor bytes are qualified by the pre-anchor tests. No
  production capsule, Q-v2, E, supervisor job, Codex resume, or scientific
  process was created. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 JSON candidate count was zero; marker:
  `Q_PRE_LAUNCHER_CAPSULE_ANCHOR_DECISION_DOC_APPEND_PASS`.

## Next exact action - finish component gates and build the capsule first

Finish the bootstrap delta audit and supervisor topology repairs, integrate
only exact qualified runner/authority/entry/terminal/bootstrap allowlists, and
run full main QA, CLI, and PanNuke validation. Then perform two byte-identical
capsule builds before final launcher materialization.

- The mandatory read-only historical public-Q readback after this append must
  remain 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  with zero Q-v2 candidates; marker:
  `Q_POST_LAUNCHER_CAPSULE_ANCHOR_DECISION_DOC_APPEND_PASS`.

## 2026-07-31 - Integrate qualified capsule components and qualify templates

- Main received exactly six qualified authority/entry/terminal files and seven
  qualified bootstrap/builder files. Direct post-copy SHA-256 readback matched
  every source WIP record. The protected main
  `src/histo_audit/workflows/__init__.py` remained 4,741 bytes with SHA-256
  `73df298fa6cff115b7fd7876778983d3b165d20146fbed874620e8e6fc99136c`;
  the stale 66-byte WIP initializer was not copied. Frozen `models/cnn.py`
  remained 74,851 bytes with SHA-256
  `9fcec232df28345be7460bfdeba96844ab78d48bb0f2860949d74b088cecb4bf`.
- Independent read-only integration audit initially reproduced all exact
  roots: runner 19 files / 1,620,924 bytes / root
  `6567f5f746090d5b9bbb230c1485baa2147276c179c83735ad2e923f2e7e8a88`;
  authority/entry/terminal 6 files / 1,070,227 bytes / root
  `8b211dbf629cd611a2c04d637db8731c462fdaad10ed7c382c0acfe7796b4f41`;
  bootstrap 7 files / 467,939 bytes / root
  `978e7f6e35b8cadc64d5e70595a9ab6a97fc80ba7a089d62e19d1b86cadf0e82`.
  The combined 32-file readback was 3,159,090 bytes. Its recorded
  `a836df336925b57e01377afdfbde847ac9e11121c36856a46076e0d5e8f46af4`
  root uses PowerShell `Sort-Object` under `pl-PL`, not ordinal ASCII order.
- Main focused gates passed **116** authority/entry/terminal tests and **148**
  bootstrap tests. The 15-file combined gate then ran **492 tests** and found
  one order-dependent test isolation failure after **491 passed**: earlier
  tests had legitimately imported `histo_audit`, so the bootstrap import
  sanitizer correctly stopped the later success fixture.
- Production bootstrap code was not weakened. The success fixture now removes
  and pytest-restores ambient `histo_audit*` module entries. A separate
  parametrized negative proves both `histo_audit` and
  `histo_audit.foreign` pre-imports stop before any import-state mutation.
  Ruff also applied five mechanical UP012 fixes and formatted that test file.
  Current `tests/test_capsule_builder.py` is 170,841 bytes with SHA-256
  `5ef90e75b396c79b320a65937806104d932a9bcda85fd8dedbd3cda34035f390`.
  Exact order-sensitive verification passed **62/62**; the complete combined
  and full-project gates remain pending after the final entry-contract delta.
  The former seven- and 32-file roots are therefore historical pre-fix
  integration evidence, not current roots.
- `entry_contract.json` deliberately remains the 246-byte
  `incomplete_fail_closed_pending_terminal_composed_receipt_v1` contract with
  SHA-256
  `8003dd488e708712972561d21f46220f2821535b0de33e28b8d7583946f4ba64`.
  Promotion to the exact 305-byte ready contract is forbidden until the common
  synthetic E-consumption/terminal E2E passes; no production capsule was
  built or published.
- The standalone launcher and supervisor reached a separate hash-specific
  **TEMPLATE_READY** verdict with independent P0=0/P1=0/P2=0. Exact template
  hashes are launcher
  `c4922c5cbd43cb59c6f9c9d1522ee37184f3633888989abe7eb966c51079b3f4`,
  materializer
  `ada354fe384db7bf80f353f3f242e9b961bf87f0bc286a4a9e017c92769da0e7`,
  launcher tests
  `608e14c88b173a74d7fc4ac01032685769887b1565567cb22c7f51b7c1b40d90`,
  supervisor
  `dc5a3632c7f104b3abadf4939d95bd88d3ff9d714766c2bfaf5f8ed0df234d4e`,
  and supervisor tests
  `ef22f843c37789786081651545269859c0b6e43a84b6041c0a1ad899be9687e5`.
  Combined direct/pytest, authority-oracle, schema, deadline, mutation,
  compile, Ruff, format, and byte-stability gates passed.
- **FINAL_QUALIFIED remains withheld.** Three launcher pins cannot contain the
  future Q file SHA, Q authority root, or supervisor-release root because Q
  itself contains the final launcher source SHA. Literal substitution would
  be a cryptographic fixed-point cycle. An acyclic post-Q authority mechanism
  must replace those three sentinels before ready-contract promotion, capsule
  publication, Q-v2, E, supervisor arming, or science.
- Read-only process inventory found no live AANCA Python training, primary,
  confirmatory, recovery, publication, or armed supervisor process. No Q-v2
  candidate exists. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The historical public Q immediately before this documentation update was
  still 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  marker: `Q_PRE_COMPONENT_INTEGRATION_TEMPLATE_READY_DOC_APPEND_PASS`.

## Next exact action - replace the cyclic pins with acyclic post-Q authority

Freeze one non-self-referential authority DAG for the three future Q/release
values, implement it only in the isolated launcher/supervisor/controller WIPs,
and require a fresh zero-P0/zero-P1 audit. Then run the common synthetic
E-consumption/terminal E2E before promoting the checked-in entry contract.

- The mandatory read-only historical public-Q readback after this append must
  remain 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  with zero Q-v2 candidates; marker:
  `Q_POST_COMPONENT_INTEGRATION_TEMPLATE_READY_DOC_APPEND_PASS`.

## 2026-07-31 - Exercise the common E path and repair the live source-delta regression

- The one existing full baseline command,
  `.venv\Scripts\python.exe -m pytest`, collected **2,099 tests** and ended
  with exit code 1 after 945.5 seconds. It was not counted as a passed gate
  and was not duplicated while running. Its retained output was truncated by
  the command wrapper, so stale cache node IDs were not misreported as current
  failures.
- A focused `--last-failed` readback reproduced one current failure:
  `test_real_parent_live_source_delta_matches_independent_registered_contract`.
  The independent expected delta still described the pre-capsule live tree
  and omitted exactly seven now-integrated files:
  `original_confirmatory_preflight.py`,
  `original_confirmatory_resume.py`,
  `original_confirmatory_runner_core.py`,
  `resource_bounded_checkpoint_execution.py`, and the authority, entry, and
  terminal capsule workflow modules. The observed change kind for every file
  was `added`; no frozen parent manifest or production authority was changed.
- The explicit test allowlist now contains those seven exact paths. A focused
  rerun of that regression and the new common E test passed **2/2**. Ruff
  check passed; Ruff format initially identified only the newly appended test
  layout, the file was mechanically formatted, and the final Ruff check and
  format-check both passed.
- The new synthetic gate uses the production capsule entry dispatcher inside
  a content-addressed `.pyz` and exact ready-contract bytes (305 bytes,
  SHA-256
  `50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844`).
  Control-only synthetic handlers consume one CREATE_NEW E claim, then pass
  the same unchanged E and claim through `verify-preterminal` and
  `verify-terminal`. It verifies capsule import origins and that no
  `artifacts/runs` tree is created. This executes no training or scientific
  analysis.
- Current changed test identities are:
  - `tests/test_capsule_builder.py`: 178,817 bytes, SHA-256
    `8d0977681b302702b4dc71016525d3993d7ca364a374334e790a0eceb0ba3e37`;
  - `tests/test_resource_bounded_authority_gate.py`: 38,162 bytes, SHA-256
    `704ebfd8bba8cd1ac2d740a29ce4c55267e7b493aab577ac4fac3df5f6b3f34a`.
- `entry_contract.json` remains deliberately incomplete and fail-closed.
  Promotion still waits for the combined real supervisor custody proof; the
  synthetic common-E test is necessary but is not by itself a qualification
  of the full production custody path.
- The historical public Q immediately before this documentation update
  remained 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`,
  and the exact Q-v2 candidate remained absent; marker:
  `Q_PRE_COMMON_E_SOURCE_DELTA_DOC_APPEND_PASS`.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - close the real custody proof before ready promotion

Finish the external supervisor's production-root, restart diagnosis,
pre-GRANT Q/E semantic revalidation, startup-hook, and combined mismatch
tests. After an independent zero-P0/zero-P1 verdict, promote the exact ready
entry contract, retain an explicit synthetic incomplete-contract negative,
and run the five-file focused capsule gate.

## 2026-07-31 - Qualify unattended custody and promote the exact ready entry contract

- The event-driven Windows supervisor remains outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-supervisor-v2-working-20260730`;
  it is not project execution source, Q, E, a scientific run, or a publication
  authority. No Startup hook was installed and no Codex wake or scientific
  process was executed.
- The acyclic terminal-client launcher remains byte-stable:
  `terminal_client_launcher_v1.py` is 271,956 bytes with SHA-256
  `e03f486edc5f02f333793de5377a70f0e8ba2fbce1cb51072772d64de0357362`;
  its materializer is 27,687 bytes with SHA-256
  `fbea62723afae56850ac22b64b7d992c5cceb6b0f050bb170c202d75cc8ec4d4`;
  its test is 176,338 bytes with SHA-256
  `06dc15b72d738ceba2729cba64582d4f8d84429207bdb7922f25a223dc64aa73`.
  The launcher gate passed 45 tests plus 53 subtests, Ruff, format, and
  compilation, with an independent zero-P0/zero-P1/zero-P2 audit.
- The final supervisor source is 850,258 bytes with SHA-256
  `b719c2f4ab7c6a4092ac9a79231246d75b0a4850efca10e208b8e95d45559bc3`.
  Final test-only identities are:
  `test_aanca_supervisor.py`
  `fc290ad2292508c59e07852808ff3ff21ecaebc07bb1f094fb8a126b2e31f8a8`,
  `synthetic_helper.py`
  `c6b72c5675aec40bacf06a8c2ae4f792494ae6b40b43aace4efc499b875f913b`,
  and `hard_crash_driver.py`
  `0133642932b120033b2c8ba2dad1028b15b1581c2c48dfc7e17b7381730817fb`.
- A new no-science rehearsal uses a real protected local named pipe, a real
  suspended client process and Job Object, a normally constructed
  `_QECustodyLease` with eight retained Win32 handles, the physical
  leaf/ancestor validators, and the production ACK helper. A Q hard-link
  mutation after a real claim prevents GRANT; a clean GRANT is delivered on a
  separate run, then an E hard-link mutation prevents FINAL_ACK. The focused
  test passed independently on the final bytes.
- One direct supervisor test invocation used the system base interpreter by
  mistake and produced exactly two expected redirector/runtime-layout
  failures. It is recorded as `INVALID_RUNNER_INTERPRETER`, made no edits, and
  is not counted as a gate. The qualifying project-venv direct run passed
  **115/115** in 180.192 seconds, and the qualifying project-venv pytest passed
  **115 tests plus 22 subtests** in 179.68 seconds. Exact repository-config
  Ruff check, Ruff format check, and `py_compile` passed. Post-test hashes were
  unchanged. A fresh independent audit reported **P0=0, P1=0, P2=0**.
- `entry_contract.json` was then promoted from the historical 246-byte
  incomplete contract to the exact 305-byte ready contract with SHA-256
  `50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844`.
  The historical incomplete bytes remain an explicit fail-before-import and
  fail-before-claim negative. The common-E test now reads the checked-in exact
  ready bytes. `CAPSULE_DESIGN.md` states explicitly that ready status is not
  Q, E, publication authority, or permission to execute science.
- Post-promotion byte readback passed. The two exact contract tests passed
  **2/2**, and the complete authority/entry/terminal/builder/hardening gate
  passed **267/267 in 15.48 seconds**. Current identities are
  `tests/test_capsule_builder.py` 179,560 bytes with SHA-256
  `564dd2f6eae58182bc51fdf1670e02a738dcc592776a1ba14310059f9adbccd7`
  and `CAPSULE_DESIGN.md` 7,073 bytes with SHA-256
  `996ec6b791b39fdb35ffc4daf367c682e8b597e14131f377c797e87d418b30f7`.
- A pre-promotion full static baseline also passed: Ruff check, Ruff format
  check for 197 files, and mypy for 96 source files. It is not substituted for
  the final post-promotion mandatory gates.
- The historical public Q remains 21,274 bytes with SHA-256
  `9e62e55d96cd60286312e7c459f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  zero Q-v2 candidates exist. The user's one authorized Q-v2 write and its
  independent verification remain unconsumed. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.
- The mandatory post-documentation readback reproduced the same historical Q
  identity with zero Q-v2 candidates; marker:
  `Q_POST_UNATTENDED_CUSTODY_ENTRY_READY_DOC_APPEND_PASS`.

## Next exact action - run the final live project gates

Run one full no-cache pytest on the promoted snapshot, followed by exact
repository Ruff check, Ruff format check, configured mypy, compile, the
read-only functional CLI checks, and the full semantic PanNuke validator with
raw-data before/after integrity readback. A failed mandatory gate must be
fixed before any capsule build, CREATE_NEW publication, Q-v2, E, supervisor
arming, or science.

## 2026-07-31 - Pass final live QA and prove raw PanNuke byte identity

- The first event-driven full-pytest wrapper let the child finish naturally
  and retained a complete terminal stdout summary of **2,099 passed and 1
  skipped**, with empty stderr. Its wrapper receipt failed closed because the
  venv `Process` handle had not been retained before `WaitForExit`, so
  `exit_code` was null, and the final atomic state replacement lacked
  overwrite semantics. The immutable STOP record is 1,045 bytes with SHA-256
  `89318e7fcd5e7a795050ee567a9018ee143025cc065550c01c112ac685cb65e0`.
  This invocation is not counted as the mandatory gate.
- A short probe proved the corrected wait path retains the process handle,
  records a numeric exit code, and atomically replaces both state and receipt.
  One qualifying test-gate rerun then used exact command hash
  `81ef5ec5615ee6cfd92100c1a01624363ee44c7b958b6ec97d61e7ccb2c9c177`,
  PID 17868, `automatic_retry_allowed=false`, and no cache provider. Its state
  and receipt are byte-identical 925-byte JSON files with SHA-256
  `882cb6aeee7b20084453bc161577b0244814acbea269744e426f0d9f7ee348e3`.
  The retained exit code is **0**; pytest reported **2,099 passed, 1 skipped in
  876.46 seconds**, and stderr is empty. The stdout log is 2,691 bytes with
  SHA-256
  `0fea37af687a417e8a403342c972423792f2bef1825c6a0dfc8425a9fcb95395`.
- On the same promoted project snapshot, `ruff check .` passed,
  `ruff format --check .` reported **197 files already formatted**,
  configured mypy reported no issues in **96 source files**, and forced
  `compileall` over `src`, `tests`, `capsule_bootstrap.py`, and
  `capsule_builder.py` passed.
- Read-only functional CLI gates passed for the root CLI,
  `data validate-pannuke`, `experiment confirmatory`, and
  `experiment lifecycle-rehearsal`. The exact full semantic validator command
  was run through a retained process handle with command hash
  `bb9aa328f05aea26e536f0e3cd18acbb37f468ff357370d790981413a905d400`.
  Its 1,089-byte receipt has SHA-256
  `db6faf5bdbedddfefbf08943b0958fa1435efb80715c79c0272ee6940027ce04`,
  records exit code **0**, and has empty stderr.
- The validator returned `status=valid` over all **7,901 patches** and all 22
  raw files. It reproduced 4,318 cross-class-overlap pixels in 575 patches,
  10,486,091 void pixels in 162 patches, zero positive-plus-background
  pixels, and 1,411 identically primary/confirmatory-ineligible
  overlap-touching instances. It reported `no_class_arbitration=true` and
  `source_masks_modified=false`. Its 2,340-byte stdout JSON has SHA-256
  `f7ff2cd5dafe2f6c33ccc0e5b557b89c43546bf6c91aba1c7413dcf5d814cd88`.
- Two pre-inventory serialization attempts were preserved rather than
  silently repaired: the first had a nonterminating
  `OrderedDictionary` size-aggregation error, and the second appended literal
  backtick-`n` bytes instead of LF. Their STOP records have SHA-256
  `f9925a8499c78b893007c3c31ef04285c25e572d6135d4a1a63013ad1df04b67`
  and
  `c5d9a8a13701f249f5dcbd53b499f2903d4f7f6eff6f860d07065ed4133c41c9`.
  The corrected readback removed only those exact two trailing serialization
  bytes, reparsed the original pre-validator capture, and reread no raw file;
  its 8,765-byte artifact has SHA-256
  `ab17a7c318b58fa5629a19ea03c18fad16f78eac6d5cee35779efe33efc4c1f6`.
- The independent post-validator content scan rehashed all 22 files and all
  **39,359,162,655 bytes**. Its 8,536-byte artifact has SHA-256
  `dda4dd400bf9713554307eb7e9d10bf11de0100e72d2eadbfcd5c0a62cd9de51`.
  Every relative path, size, UTC mtime, and file SHA-256 matches the pre-scan;
  both record roots are
  `2e611ada1afb86dfb53375cdcc2719213fe9ea633472bea10ea79907b0e573fc`,
  with zero differences. Raw PanNuke is byte-identical before and after.
- These gates qualify the promoted source snapshot for deterministic capsule
  construction only. No capsule has been published; Q-v2, E, supervisor
  arming, Startup installation, and science remain absent. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.

## Next exact action - build twice and compare exact capsule bytes

Freeze the exact source inventory, run two independent deterministic capsule
builds into separate non-publication locations, and require whole-file,
internal-manifest, member-order, and source-inventory equality. Then recompute
sealed capacity plus 10 GiB. Do not invoke CREATE_NEW publication, write Q-v2,
create E, arm the supervisor, or execute science unless these gates pass.

## 2026-07-31 - Pass the independent two-build capsule reproducibility gate

- A no-publication runner outside the repository froze an exact 105-entry
  source inventory at
  `C:\Users\NATAN\AppData\Local\AANCA-capsule-build-final-20260731`.
  The runner is 12,710 bytes with SHA-256
  `5609ed27bc714d33301c0b3e2bd5c17090c837aa7f9c6096b85118cb7a5a459b`;
  the 19,785-byte inventory has SHA-256
  `28c790498d8a831578986c8a8cf132a66303d2564eb1c9520137c9247b1a414c`
  and source-records root
  `66874297ebefe74e1760abd1519abae0b91121541c4bf8748bc8cf597558ccb8`.
- Two independent workers built into pre-absent `build-a` and `build-b`
  locations. Both commands exited 0 with empty stderr; neither called
  `publish_capsule_create_new`, created Q/E, or executed science. Both
  candidate files are read-only, 7,050,492 bytes, and byte-identical with
  SHA-256
  `3e38dde3aa8efb76a0021985e0bab4a7091765c6b11e102037ced32c8a294e6c`.
  Receipt A is 1,328 bytes with SHA-256
  `e46af84d48eeaecd08cddf8b2e9465f64c8f7783ba2a240bb606a798b01c54e2`;
  receipt B is 1,328 bytes with SHA-256
  `928210ce6b468530eedb9a1dbbf8c6d032f66d4e4ecdcf4518f6245aa243f743`.
  They differ only in the declared creation timestamp and output path.
- An independent read-only audit passed. The archive contains 106 entries
  (105 payload entries plus the manifest), with identical order, stored bytes,
  CRCs, fixed metadata, local headers, and no extras, comments, gaps, or
  directory members. The canonical 19,414-byte internal manifest has SHA-256
  `fd41910c77e70002ef3d2a3e21346317e094e9b600fe0524592fdf73d8a4ddb3`,
  payload size 7,014,710 bytes, and independently reproduced the exact source
  records root. All 105 live source path/size/hash records still match.
  The ready entry-contract bytes match repository, inventory, and capsule at
  SHA-256
  `50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844`.
- No bootstrap mode was invoked: there is intentionally no identity-only
  mode, and all three allowed modes require Q/E. The content-addressed
  production destination remains absent, as do Q-v2 and E. The two-build
  gate authorizes only the exact sealed-capacity computation and publication
  preflight.
- The mandatory post-documentation historical-Q readback reproduced 21,274
  bytes and SHA-256
  `9e62e55d96cd60286312e7c4591f1d3ac8377ffe38eceacf0db8f97294330ee3`;
  zero Q-v2 candidates exist. Marker:
  `Q_POST_CAPSULE_REPRO_DOC_APPEND_PASS`.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - pass capacity, then publish the verified capsule once

Recompute the typed original-confirmatory 108/90/18-cell, 36-CNN,
180-fit sealed-plan preflight and require at least the 30-GiB projected active
single-copy checkpoint budget plus exactly 10 GiB. Independently verify the
CREATE_NEW destination and publication call. Only if both pass, publish the
already byte-verified capsule exactly once to its content-addressed path and
perform a same-byte, same-manifest readback. Do not write Q-v2, create E, arm
the supervisor, or execute science during this step.

## 2026-07-31 - Reject the first candidates and correct sealed capacity to dual-copy v2

- The capacity gate stopped publication before any CREATE_NEW call. The live
  preflight still implemented the older single-copy estimate of 30 GiB plus
  10 GiB, while the later binding operational decision requires every one of
  the 180 CNN-fold checkpoints to have both a canonical physical file and a
  distinct versioned O_EXCL physical copy. Both copies must be counted.
- The two otherwise reproducible candidates with SHA-256
  `3e38dde3aa8efb76a0021985e0bab4a7091765c6b11e102037ced32c8a294e6c`
  are therefore preserved but ineligible for adoption or publication. A
  canonical external rejection record is 795 bytes with SHA-256
  `6134bb337cb56bf3c80c68fa9e1d2f6dcc1fe8fadab64f1695fb3cd67a7433ea`.
  It records `publication_attempt_performed=false` and
  `scientific_execution_performed=false`.
- The integrated outcome-blind policy is now
  `original_confirmatory_sealed_plan_capacity_v2`, schema version 2, with
  policy SHA-256
  `b83d3e8e1693a640b8a306a1e5a4b7722fe323bedeefcff8aa1d29c7927bf284`.
  It binds the unchanged 108/90/18 plan, 36 CNN cells, 180 fold fits, and the
  `canonical_plus_distinct_versioned_o_excl_physical_copy_v1` policy. Its
  arithmetic is exactly 30 GiB canonical + 30 GiB versioned + 10 GiB safety
  margin = **70 GiB (75,161,927,680 bytes)**. Each component, copy count,
  planned checkpoint count, and sum is checked fail-closed.
- The source is 47,891 bytes with SHA-256
  `df0b46a941cd4a246dfb9e25c374600d6a6979c8955ae67d5beed26effa74aea`;
  its 31,953-byte test is SHA-256
  `58fa1b2f4e04eb40c9b7bdec6ff2fcff394b891f112df89bbff30904fc182235`.
  Five explicit arithmetic-regression cases reject a single copy, a reduced
  per-copy estimate, a reduced aggregate, a reduced margin, or the obsolete
  40-GiB minimum.
- The focused preflight suite passed **37/37**; a wider
  original-confirmatory/runner/checkpoint/completion/capsule suite passed
  **403/403**. Focused Ruff, format, and mypy passed after formatting the
  final source bytes. A full no-cache project pytest is now running exactly
  once under a retained process handle; it is not a scientific process.
- `.gitignore` now explicitly covers
  `artifacts/execution_capsules/**`; `git check-ignore` also reconfirmed raw
  ZIP/NPY, nested manifest Parquet, and duplicate-audit NPZ coverage. Its
  current 1,419-byte SHA-256 is
  `31480f63b2601b14329fcbfcb4829bf027f6ddc30a1f5878287ff8007077873a`.
- No Q-v2, E, supervisor job, Startup hook, capsule publication, training, or
  scientific result was created. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Corrected next exact action - qualify v2, then rebuild twice

Require the running full pytest to end with a retained numeric zero exit code,
then pass full Ruff, format, mypy, compile, functional CLI, and capacity-v2
readback. Freeze a new exact source inventory and create two new
non-publication candidates in fresh locations; the rejected candidates must
never be reused. Only byte-identical new candidates may proceed to the
reviewed content-addressed CREATE_NEW publication path. Q-v2, E, supervisor
arming, and science remain forbidden during these gates.

## 2026-07-31 - Qualify the capacity-v2 regression baseline and stop an authority cycle

- The one active no-cache full project pytest ended naturally. The retained
  process receipt records command SHA-256
  `064993e15567779397afc57b6bb1f8c6b5fe854469a261c166c2d76b5b2ce257`,
  numeric exit code **0**, **2,104 passed and 1 skipped in 920.46 seconds**,
  and empty stderr. The completed state and receipt are byte-identical
  1,001-byte files with SHA-256
  `14084ef4d23fc93ffc40b5b712807a1ec2387df773db9ec45192e71fcdc0e8bb`.
  Stdout is 2,691 bytes with SHA-256
  `cb19c30fde861fe510b60d17bc10e27f1316a11a50817d72489f69308c449e0f`;
  stderr is the empty SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  All three tracked wrapper/runtime PIDs were absent at terminal readback.
- This is a valid capacity-v2 regression baseline, not the final source gate:
  production still lacks the typed outcome-split original-confirmatory
  technical authority and a durable one-shot Q-v2 publication controller.
  Those missing execution-source components must be integrated and then the
  complete gates must run again before a new capsule build.
- Dependency review rejected a draft authority shape that required final
  launcher or supervisor identities. The launcher protects the static runner
  binding; that binding contains the newly sealed lifecycle-readiness run;
  and lifecycle readiness must consume the technical authority. Requiring the
  launcher in that authority would create the cycle
  `authority -> lifecycle -> static binding -> launcher -> authority`.
- The corrected acyclic boundary makes the lifecycle-consumed technical
  authority bind the exact parent P, unchanged frozen science, historical
  sealed primary, final qualified repository source, published capsule,
  capacity-v2 evidence, and an independent outcome-split review. Lifecycle
  readiness is created next. Its exact static binding then materializes the
  launcher and supervisor release, and Q-v2 binds those downstream identities.
- Q-v2 and E remain absent, the user's one Q-v2 publication authorization is
  unconsumed, and no capsule publication, supervisor arming, Startup hook,
  training, or scientific execution occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Corrected next exact action - integrate the missing one-use governance path

Implement and independently test the typed outcome-split technical-authority
builder/verifier/CLI, its lifecycle consumer, and the durable one-shot Q-v2
controller. Require exact parent P, unchanged frozen inputs, explicit
`primary_outcomes_inspected=true` and
`confirmatory_outcomes_inspected=false`, an outcome-blind independent review,
CREATE_NEW/O_EXCL publication, durable pre-write intent, and permanent STOP on
ambiguity. Then rerun the complete live QA/CLI/PanNuke gates before freezing a
new source inventory or building any capsule.

## 2026-07-31 - Integrate the typed T0 core and lifecycle/static consumers

- Added the closed-schema, outcome-split
  `original_confirmatory_technical_authority_v1` core. It binds exact parent P,
  unchanged frozen science, the sealed 185/37 historical primary, exact
  execution source, a content-addressed capsule plus its independent receipts,
  capacity-v2, and the explicit split
  `primary_outcomes_inspected=true` /
  `confirmatory_outcomes_inspected=false`. Launcher, supervisor, saved-session,
  Q, E, and terminal fields are forbidden upstream.
- Live verification is read-only and fail-closed. It validates the parent
  chain, frozen hashes, complete source inventory, capsule ZIP/member/source
  alignment, publication and independent-readback receipts, exact
  108/90/18-cell and dual-copy 70-GiB capacity receipt, sealed historical
  primary stage receipt, process-source identities, chronology, and current
  free capacity. It performs a second outcome-blind readback of every external
  binding and a final authority-directory snapshot before returning.
- A final-snapshot regression initially found that four return hashes were
  reread after the last snapshot. They are now cached before that snapshot;
  no filesystem read occurs afterwards. Independent mutation injection
  confirms that a post-snapshot replacement cannot alter the verified return
  identity.
- “Independent” now requires a different PID/creation tuple, implementation
  path, and implementation SHA-256 for both the T0 builder/reviewer and capsule
  publisher/readback pairs. Distinct caller-declared PIDs running the same
  implementation fail closed. A fresh-child producer must still capture the
  real OS process identity before publication.
- Exact current T0 identities are:
  `src/histo_audit/workflows/original_confirmatory_technical_authority_v1.py`
  89,221 bytes, SHA-256
  `ec45c2a274464873760ec4b2a7f01a19ecad9f51eb3c0b0e493127dddbf69252`;
  `tests/test_original_confirmatory_technical_authority_v1.py` 37,169 bytes,
  SHA-256
  `45257962e4b5dc068ce8b0ca09f610f6c904c11b94ea5755bb7e3e2e776ef07c`.
  Its exact focused gate passed **55/55**; Ruff check/format, strict mypy,
  `py_compile`, and an independent adversarial audit passed with P0=0/P1=0.
- Integrated the T0 consumer through lifecycle rehearsal, fresh-process
  verification, readiness, static-runner binding schema v2, capsule bootstrap,
  request parsing, and capsule-only runtime gates. The three static pins are
  the exact technical-authority directory, artifact root, and technical
  authorization SHA-256. Each public lifecycle entrypoint performs one full
  live T0 verification, internal/final comparisons are shallow, and there is
  no cross-process cache. The affected owner gate passed **286 tests**, the
  focused call-count gate passed **8 tests**, Ruff/format passed, and strict
  mypy passed for six production files.
- The combined current T0/lifecycle/preflight/authority/runner/builder suite
  passed **341/341 in 91.91 s**. An earlier invocation was terminated by an
  incorrectly short five-second command-wrapper timeout with exit 124; no
  child remained, and it is not counted as a gate.
- The external one-use publisher/composite verifier baseline at SHA-256
  `25886ea297701f80e2033ae10de78c5dcc246a9d2262c0f8e3e76ded614bbc59`
  independently passed with P0=0/P1=0, including the permanent global
  namespace claim, retained handles, directory flushes, STOP, success-last,
  and no retry/adoption/cleanup. It is not integrated yet: its missing
  `build-intent` and fresh-child `review-intent` phases are being added and
  require a new exact audit.
- `.gitignore` now excludes
  `artifacts/original_confirmatory_technical_authorities/**`; direct
  `git check-ignore -v` readback passed for both the permanent namespace claim
  and an authority child. The file is 1,476 bytes with SHA-256
  `6a827cf0406f3e5c2bb44781d5d5ef330359b75ba718633b8cba8de6359d6ecd`.
- A separate read-only audit rejected the current external Q-v2 controller as
  production-ready. It remains disabled and unpublished because it does not
  yet consume the combined live T0 verifier, correlate lifecycle/static
  bindings, enforce one fixed global namespace with retained intent custody,
  or use the typed E/supervisor path. No Q-v2 authorization was consumed.
- No capsule was published, no T0 directory was published, no lifecycle run
  was created, no Q-v2 or E was written, no supervisor was armed, and no
  scientific process ran. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains open at **8/10 = 80%**, and M9 remains
  locked.

## Next exact action - finish the four-phase one-use chain before final gates

Finish and independently audit `build-intent`, fresh-child `review-intent`,
global CREATE_NEW T0 publication, and composite read-only verification. Route
lifecycle exclusively through the verified-published wrapper and its namespace
claim. Repair the Q-v2 controller and the supervisor restart one-wake
ambiguity in their external WIPs, then integrate only byte-stable reviewed
files. After that exact source snapshot, run the complete mandatory
pytest/Ruff/format/mypy/compile/CLI/PanNuke gates once before freezing a new
source inventory or building a new capsule.

## 2026-07-31 - Close transient T0 provenance and composite-binding gaps

- Added `artifacts/original_confirmatory_technical_authority_requests/**` to
  `.gitignore`. Direct `git check-ignore -v` readback passed for a synthetic
  `probe.intent.json`; the existing ZIP, NPY, raw-data, nested manifest
  Parquet, execution-capsule, and published-T0 exclusions remain present.
  `.gitignore` is now 1,540 bytes with SHA-256
  `9d1d1a9b0c65a171b750aca0124d29f2411b45af1631a4212730b546c7c1615b`.
- Read-only integration review found that the current static binding's three
  flat T0 pins do not yet bind the permanent singleton namespace claim or the
  composite published-T0 lifecycle binding. A static-binding v3 external WIP
  is therefore required before Q, E, or scientific execution.
- The same review found that legacy generic lifecycle APIs can currently
  accept a T0 marker while requesting only shallow verification. They must
  reject T0; only the strict original-confirmatory public path may accept the
  published wrapper and it must own exactly one combined `verify_live=True`
  call. Private downstream rechecks remain shallow.
- The external Q-v2 controller is being refactored to call that single strict
  lifecycle operation and consume its closed six-pin result: namespace
  directory/claim, technical-authority directory/artifact root/authorization,
  and composite lifecycle-binding SHA-256. No caller-supplied preverified
  object or live-verification switch is permitted.
- An independent supervisor restart audit found two blocking P1 cases:
  malformed STOP evidence could escape global ambiguity detection and prior
  wake terminal evidence without its intent could permit a second recovery
  wake. The external WIP remains uninstalled and is being repaired with
  zero-wake adversarial tests.
- No source inventory was frozen, no capsule or T0 was published, no Q-v2 or E
  was written, no supervisor was armed, and no scientific process ran. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - integrate only independently qualified final bytes

Finish the external publisher/reviewer, strict composite lifecycle,
static-binding v3, Q-v2 controller, and supervisor restart fixes. Require
P0=0/P1=0 and exact focused QA for each, then integrate with `apply_patch` and
run the combined affected suite. Only after the complete repository gates pass
may a final source inventory and new deterministic capsule pair be built.

## 2026-07-31 - Qualify restart semantics and keep red static gates blocking

- The external supervisor-v2 restart repair passed **127 tests plus 22
  subtests** in 185.92 seconds. Twelve focused adversarial cases cover
  malformed STOP plus another candidate, scan/decision exceptions, missing
  wake intent with result/failure/ambiguity evidence, global ambiguity, and
  exact zero-wake behavior. Ruff check/format passed, and a fresh review found
  no remaining P0/P1 in that recovery change.
- Exact still-uninstalled supervisor WIP identities are
  `aanca_supervisor.py` 883,706 bytes, SHA-256
  `3105dd6c3b7f533a2b36bfa7d54cf931f9653465b2c5f461394d180b6876864b`,
  and its test 293,664 bytes, SHA-256
  `ec5a3d702425a182eae0534733d52cb50cb72e5ffedeb5c5c632821a60863cfe`.
  It remains **not qualified**: strict/configured production mypy reports 51
  errors. Read-only triage found no missing stubs and identified seven
  optional-contract reads that require an explicit protected-mode non-None
  guard, plus finite HANDLE/narrowing/name-collision issues. Deeper control-flow
  review confirmed that legal nonprotected test mode does not execute that
  block, so this cluster is a typing/narrowing gap rather than a reproduced
  runtime P1. A separate external clone is repairing it without ignores or
  configuration weakening.
- The launcher external WIP's twelve strict-mypy failures were repaired with
  explicit Optional/HANDLE checks and cached UTC-offset narrowing. Its full
  gate passed **45 tests plus 53 subtests**, Ruff check/format, strict
  no-incremental mypy for both production files, and `py_compile`; review
  returned P0=0/P1=0. Exact hashes are launcher
  `a2f3a263c6064698b0bfd0b424172cd683d142983462c113329450c1cd9c95d2`,
  materializer
  `2e4c71aef3be86caacbe26e9fd5fd77c93d1c98746e7cc952fed90d0ddc1a5a3`,
  unchanged test
  `06dc15b72d738ceba2729cba64582d4f8d84429207bdb7922f25a223dc64aa73`,
  and unchanged numeric-path inventory
  `155feb9c092b7ca9fff3b32936feed5dff097c471c8a726293e323006782b8d8`.
  All anchor, pre-Q, and materializer trust tokens remain unresolved; no
  launcher was materialized.
- The provisional static-binding v3 external gate passed **270 tests** with
  three path-relative WIP-only tests deferred to live integration; Ruff and
  strict mypy passed in its completed scopes. It is not frozen until the final
  publisher composite and lifecycle API settle.
- The external Q-v2 controller reached **51/51 tests**, Ruff, and compilation
  with its singleton intent before the sole strict live gate, retained
  namespace custody, exact `sys.orig_argv`, post-gate authority receipt,
  static-v3 composite equality, and physical E/receipt/ACK readback. Production
  remains deliberately disabled until final source pins and a non-injectable
  source-pinned E/supervisor factory exist.
- The installed supervisor remains the older unarmed qualified v1 only.
  Neither it nor any WIP was armed, installed, or used to wake Codex. No
  capsule/T0/Q/E was published and no scientific process ran. Formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9
  remains locked.

## Next exact action - finish T0 bytes and make supervisor source type-clean

Complete the permanent review-attempt-bound T0 publisher and its independent
audit. In parallel, bring the cloned supervisor production source to zero
strict-mypy errors and rerun all functional/adversarial gates. Then freeze the
publisher field set so strict lifecycle, static v3, and Q can converge on one
exact composite before live integration.

## 2026-07-31 - Confirm the execution/governance identity boundary

- Read-only source inspection confirmed that `capture_source_tree()` hashes
  only `src/**`, `configs/**`, `pyproject.toml`, and `uv.lock`, with the two
  generated frozen-config publications explicitly excluded. `STATUS.md`,
  `DECISIONS.md`, and the other governance documents are captured by the
  separate `capture_governance_tree()` snapshot and cannot change the
  execution-source root.
- The focused boundary gate passed **17 tests**:
  `.venv\Scripts\python.exe -m pytest -q
  tests/test_execution_source_identity.py tests/test_run_tracking.py -k
  "governance or execution"` returned `17 passed, 26 deselected`.
- A read-only process inventory found no PanNuke confirmatory process, armed
  supervisor action, or supervisor test child. No capsule, T0, Q, E, launcher,
  supervisor arming, or scientific execution was started. Free capacity was
  133.15 GiB, above the sealed 70-GiB minimum.
- The supervisor-v2 typing clone reached zero strict production mypy errors,
  but live-project Ruff policy exposed 29 auto-fixable B009/B010 diagnostics
  in its external synthetic test and both large external files still required
  deterministic formatting. The clone remains unqualified and uninstalled
  until mechanical repair is followed by the complete functional, static,
  compilation, and independent-review gates.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains open at
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - freeze the reviewed cross-component contract

Finish the independent four-phase T0 audit, then overlay its exact final
publisher bytes with strict lifecycle and static-binding v3. In parallel,
finish the request-only Q-v2 controller and fully requalify the mechanically
formatted supervisor clone. Integrate only byte-stable P0=0/P1=0 components;
do not publish Q/E or start science before the combined live gates pass.

## 2026-07-31 - Revoke a T0 freeze that changed after declaration

- Independent root readback rejected the declared T0 reviewer identity
  `6d39bdf2...` / 8,632 bytes because the shared external WIP changed after the
  freeze notice. Independent observers saw successive 9,740-byte and
  9,990-byte candidates; the latter failed collection with a `SyntaxError` in
  the in-progress parent-controller comparison.
- The attempted root command was:
  `$env:PYTHONPATH="<T0-WIP>\src;<project>\src";
  .venv\Scripts\python.exe -B -m pytest -q "<T0-WIP>\tests"
  tests\test_original_confirmatory_technical_authority_v1.py`.
  It stopped during collection. A second collection issue was the duplicate
  test-module basename across the WIP and live trees, so a later independent
  command must use unique test paths or importlib mode. This invocation is
  evidence of unstable bytes, not a passed or failed scientific gate.
- The owner confirmed that post-freeze parent-custody hardening had started to
  prevent a manually invoked reviewer from adopting an abandoned attempt.
  Every prior T0 freeze/hash/audit statement is therefore superseded. The
  repair must pass `py_compile` first, then the complete combined suite,
  Ruff/format/mypy/help, two unchanged rehashes, and a new independent audit.
- The external supervisor-v2 typing/format clone completed owner QA after
  mechanical Ruff repair: source
  `cae0e6ded9863a1dc552dc11299c30676c678c79668d9ae0de73202517008359`
  (868,300 bytes), test
  `4c99c3438168f733581fb9a1c36ac2cfe150fee228760406fd38496c8c503f59`
  (293,790 bytes), **137 tests plus 32 subtests**, strict/configured mypy,
  Ruff check/format, and compilation all passed. It remains external,
  uninstalled, and unarmed while a fresh independent audit runs.
- The request-only Q-v2 refactor reached **53 tests** plus Ruff/format and
  compilation. Its production injection seam is absent, but all release pins
  remain unset and the exact live E/native-handle/IPC factory is intentionally
  unavailable; its CLI remains fail-closed before publication.
- No capsule, T0, Q, E, launcher, supervisor installation/arming, Codex wake,
  or scientific process occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - require an atomic T0 freeze and real downstream factory

Wait for one unchanged T0 publisher/reviewer pair to pass complete QA and two
rehashes, then perform a fresh independent P0/P1 audit. Use only that exact
pair to finish the lifecycle/static combined overlay. In parallel, finish the
source-pinned real E/supervisor factory and independently audit the qualified
supervisor clone. Do not integrate provisional moving bytes.

## 2026-07-31 - Require kernel-continuous T0 reviewer custody

- The live-policy-clean T0 candidate (`a26dde41...` publisher and
  `5af21aa5...` reviewer) independently passed **106 tests** and retained its
  hashes, but an independent audit found one P1. The reviewer authenticates its
  controller parent before the long live pass, yet a hard-killed controller
  can disappear afterwards while the child continues to a CREATE_NEW review
  receipt. The retained `Popen` handle covers normal exception paths, not hard
  parent death.
- That candidate is superseded before publication. The replacement must start
  the reviewer suspended, assign it to a controller-only Windows Job with
  `KILL_ON_JOB_CLOSE`, retain the process/thread/Job handles through
  `WaitForExit`, and only then resume it. The Job handle must never be inherited
  or duplicated into the child. A real synthetic parent-death regression must
  leave the permanent attempt but no review receipt and no retry path.
- The supervisor-v2 clone independently completed P0=0/P1=0 at source
  `cae0e6ded9863a1dc552dc11299c30676c678c79668d9ae0de73202517008359`
  and test
  `4c99c3438168f733581fb9a1c36ac2cfe150fee228760406fd38496c8c503f59`.
  Root re-ran the exact live-project policy: Ruff check passed, both files were
  already formatted, strict source mypy passed, and both hashes were unchanged.
- Two static-v3 provenance contradictions were found and repaired in its
  external WIP: flat `freeze_directory` now equals the nested parent authority,
  and flat execution-source root/manifest plus independent-review receipt now
  equal their nested published-T0 counterparts. STATIC-v3 remains the same
  24-field schema; runner-core was not expanded with duplicate fields.
- The E-custody audit kept Q safely disabled and exposed real integration
  blockers: the controller lacks a full source-pinned suspended-supervisor
  factory, exact downstream schema validation and retained native-handle
  lifecycle; the current supervisor also expects to create
  `job_dir/run_spec.json` after resume while the authority callback would need
  that same file before resume. A separate read-only contract trace and
  external factory WIP are resolving this without inventing scientific fields.
- No live source integration, capsule/T0/Q/E publication, launcher
  materialization, supervisor installation/arming, Codex wake, or scientific
  process occurred. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`;
  M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - close parent death and suspended-spec handoff

Qualify the kernel-bound T0 reviewer replacement. In parallel, finish the
authoritative E-field trace and select the smallest acyclic supervisor launch
spec handoff, including an explicit success-side handle finalizer. Then rerun
the combined lifecycle/static/Q/supervisor contract tests before any live
integration or one-use write.

## 2026-07-31 - Reject post-creation Job assignment for the T0 reviewer

- Implementation review found that stock
  `Popen(CREATE_SUSPENDED)` followed by `AssignProcessToJobObject` still has a
  fatal controller-death gap. CPython closes the original primary-thread
  handle, and a hard death between process creation and Job assignment can
  leave an unbound suspended orphan.
- The T0 replacement therefore uses direct Windows `CreateProcessW` with
  `STARTUPINFOEX` and `PROC_THREAD_ATTRIBUTE_JOB_LIST`, so the reviewer enters
  the controller-only kill-on-close Job atomically at process creation. The
  exact process, primary-thread, and Job handles remain retained. No
  post-creation assignment, fallback, breakaway, or inherited Job handle is
  qualifying.
- The WIP is intentionally non-runnable while this replacement is incomplete.
  No orphan, publication, or scientific process was created.

## 2026-07-31 - Close the E plan ambiguity and select fixed control staging

- A fresh read-only process inventory found no PanNuke primary, confirmatory,
  recovery, armed supervisor, terminal launcher, or `codex exec resume`
  process. C: had 132.84 GiB free. The obsolete
  `aanca-primary-to-option-b` heartbeat was queried through the Codex
  automation API and returned `not_found`; no periodic Codex automation
  exists to compete with the event-driven handoff. Windows Task Scheduler had
  no AANCA/Codex/PanNuke task. The existing Startup entry is the older
  one-shot `recover-all` command, and its installed supervisor had no `jobs`,
  `locks`, or `state` directory.
- The E scientific `plan_sha256` ambiguity is resolved by exact code flow.
  `scientific_request_projection.plan_sha256` is the canonical frozen
  108-cell matrix-plan hash
  `c1993d44...`, and `controls_binding_sha256` is the frozen controls hash
  `7d529...`. The current project `PLAN.md` hash
  `176f0184...` and the Q execution-capsule `plan_sha256` are separate
  identities and must not be substituted for the scientific matrix plan.
- The existing static checkpoint summary is intentionally lossy: it does not
  carry the 180 canonical fit directives required by E. The production E
  factory must therefore reconstruct the fresh projection from source-pinned
  frozen controls, outcome-blind PanNuke inputs and CNN data/split
  fingerprints, compare it to the exact frozen semantic constants and static
  summary/root, and then serialize the complete 180-directive projection. It
  must not read predecessor outcomes or derive the matrix hash from the Q
  capsule field.
- The suspended-spec contradiction is resolved by one fixed external staging
  namespace, `<supervisor_root>/control_staging/<job_id>`, distinct from the
  final `<supervisor_root>/jobs/<job_id>` directory. The final job directory
  remains absent until the resumed supervisor creates it. Staging must use a
  closed CREATE_NEW inventory containing an attempt marker,
  `e_intent.json`, `launch_authorization.json`, the staged supervisor spec,
  and a ready marker, plus only explicitly defined STOP/wake evidence.
  The supervisor's final `run_spec.json` must bind the staged spec bytes,
  hash and physical identity.
- The launch authorization is a separate 44-field supervisor-v2 document in
  the currently qualified schema; it
  is not the historical T0 `technical_authorization_sha256`. It is built only
  after the suspended supervisor identity and deterministic Q/E
  READY/receipt/spec fields exist, then CREATE_NEW-published before the staged
  spec and before resume.
- Startup recovery must scan both `jobs` and the fixed staging inventory.
  Empty, partial, malformed, abandoned, or unmatched staging is a permanent
  CREATE_NEW STOP with zero relaunch/adoption/cleanup and at most one
  diagnosis wake through the pinned exact session. A controller-only factory
  cleanup is insufficient after Windows restart.
- The atomically Job-bound T0 WIP currently parses successfully at source hash
  `7db8e111...`, but that moving checkpoint is not a freeze. The real
  hard-controller-death test has not yet passed. The request-only Q WIP
  closed a pre-gate import-inventory vulnerability and reached 60 focused
  tests at provisional source `ba37a088...`; all final pins remain unset and
  its CLI remains fail-closed. STATIC/lifecycle/E orchestration remains
  provisional after 288 focused passes because final T0, staging and factory
  bytes do not yet exist.
- No live source was integrated; no capsule, T0, Q, E, launch authorization,
  staged spec, final job, launcher materialization, supervisor arming, Codex
  wake, or scientific execution occurred. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - implement and qualify the acyclic staged handoff

Pass the real T0 hard-parent-death regression and freeze one unchanged
publisher/reviewer pair. In parallel, finish the source-pinned fresh-180 E
builder, the exact launch-authorization/staged-spec factory, and supervisor
restart STOP handling. Only then converge STATIC/lifecycle/Q on those exact
bytes, perform an independent P0/P1 audit, and integrate through reviewed
patches before the full live QA and PanNuke gates.

## 2026-07-31 - Disable the legacy direct confirmatory CLI bypass

- Read-only integration audit proved that live
  `python -m histo_audit experiment confirmatory` still loaded
  `execute_confirmatory_study` without a Q/E capsule request and therefore
  selected the generic lifecycle path. That route could bypass the required
  published-T0, Q, E, capsule and supervisor chain.
- The command now stops immediately with
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED`, before lifecycle, gate, dataset,
  cache, executor or run-directory access. The obsolete executor-loading body
  was removed; Typer help and immutable option visibility remain available.
- Focused validation passed:
  `.venv\Scripts\python.exe -m pytest -q tests\test_stage_cli.py` returned
  **8 passed**; Ruff check passed; Ruff format reported both files already
  formatted. The functional command returned recorded exit code **2** and the
  exact fail-closed gate. No run directory or scientific artifact was
  created.
- This is a live execution-source change, so every older source inventory,
  capsule candidate or pre-change source root remains ineligible for the
  future T0/Q/E chain. A fresh inventory and two deterministic capsule builds
  remain mandatory after all remaining integrations.
- No Q/E, supervisor job, wake or scientific process was started. Formal
  status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish the two kernel/restart regressions

Complete the running T0 hard-parent-death suite and the external
`control_staging` supervisor implementation. Then merge the exact fresh-180
E/spec/authorization factory with request-v2 and re-run the combined
component gates before any further live source integration.

## 2026-07-31 - Resolve the pre-ACK final-job causal dependency

- Read-only supervisor inspection proved that Q/E custody cannot acknowledge
  before the supervisor has a final job directory: the sealed receipt path is
  exactly `jobs/<job_id>/q_e_custody_receipt.json`, and the current custody
  lease and ACK bind that path.
- The qualifying ordering therefore permits the resumed supervisor to
  CREATE_NEW-publish only the final job directory, `run_spec.json`, and the
  Q/E custody receipt before ACK. This is a metadata-only state. It does not
  permit process C, scientific execution, outcome access, result publication,
  adoption, cleanup, or retry.
- The external staging inventory remains exactly five CREATE_NEW files in
  order: `staging_attempt.json`, `e_intent.json`,
  `launch_authorization.json`, `supervisor_launch_spec.json`, and
  `staging_ready.json` last. No `manifest.json`, `consumed.json`, alias, STOP,
  or wake record may be added inside that directory.
- Q-v2 owns one closed `control_staging_projection` and its canonical SHA-256.
  The exact projection is repeated in `staging_attempt.json`.
  `staging_ready.json` binds that projection and the first four file
  identities, hashes, and sizes. The final outer `run_spec.json` binds the
  projection and all five staging leaves.
- The canonical supervisor launch specification and its independent launch
  authorization remain the existing exact 44-field schemas. The staging
  projection must not be inserted into either closed payload.
- This decision changes no frozen scientific input and launches nothing.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - qualify unchanged T0 and finish staged handoff

Complete the independent audit and immutable rehash of the frozen T0
candidate. In parallel, finish and independently audit Q-v2, the
fresh-180 E/spec factory, and the five-file supervisor staging/restart
implementation before applying any further live-source patch.

## 2026-07-31 - Remove the Q/attempt/staging hash cycle before publication

- An independent request-v2 audit rejected the provisional derivation in
  which the attempt root consumed final `q_authority_root_sha256` while the
  final Q payload also consumed attempt-derived staging paths. That relation
  is cyclic and cannot be reproduced from canonical bytes.
- Q-v2 now uses an acyclic two-stage derivation. A
  `q_base_authority_root_sha256` is derived from the 11 original static Q
  fields. The attempt root consumes that base root, the job/run/nonce and
  staging projection are then derived, and final
  `q_authority_root_sha256` hashes all 16 unsigned Q fields.
- The serialized Q document does not duplicate a base-projection object. Its
  exact 17-field set is the 11 static fields, the base root, the attempt
  projection/root pair, the staging projection/root pair, and the final Q
  root. Canonicalizers must independently reconstruct the base projection and
  both roots.
- E and terminal custody continue to bind the final Q root. The attempt and
  `staging_attempt.json` bind both the base root and final root without
  treating them as interchangeable.
- No Q byte has been written and no one-use authority has been consumed.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## 2026-07-31 - Freeze a legal outer-Job custody transfer

- The suspended-supervisor factory cannot keep controller-only
  `KILL_ON_JOB_CLOSE` ownership after the controller exits, and it cannot
  place the overnight supervisor in an active-process-limit-one Job because
  that supervisor must later launch the science, verifier, and Codex
  processes.
- The qualifying outer Job keeps `KILL_ON_JOB_CLOSE` set continuously, sets
  no breakaway flags, and has no active-process limit. Singleton/root/project
  locks remain the authority preventing competing jobs.
- After atomic `CreateProcessW` with `JOB_LIST` and exact `HANDLE_LIST`, the
  resumed supervisor validates staging, publishes only the permitted
  metadata-only final-job evidence, completes Q/E custody ACK, and remains
  blocked on its controller pipe.
- Two dedicated bounded anonymous pipes, distinct from the existing Q/E
  stdin/stdout transport, carry one `JOB_CUSTODY_RELEASE` and one
  `JOB_CUSTODY_ACCEPTED`. Their child-side handle values and closed transport
  contract are bound by `staging_ready.json`; both child handles are included
  in the exact `HANDLE_LIST`.
- The controller duplicates the Job handle into the live supervisor as
  non-inheritable, sends one `JOB_CUSTODY_RELEASE`, and waits for exact
  `JOB_CUSTODY_ACCEPTED`. The supervisor revalidates the handle, unchanged
  kill/no-breakaway limits and its own membership, CREATE_NEW-writes the same
  canonical accepted payload to
  `jobs/<job_id>/outer_job_custody_accepted.json`, and only then sends it.
  Science is forbidden before that acceptance. The controller compares pipe
  bytes to a retained receipt readback and closes its source Job handle only
  after the exact match.
- A controller crash before duplication closes the last handle and kills the
  supervisor. A crash after duplication but before release produces pipe EOF;
  the supervisor exits, closes the remote handle, and the Job kills any
  descendant. After acceptance the supervisor owns the Job and may continue
  unattended. The remote handle may never be inherited by science, verifier,
  or Codex children.
- Synthetic real-Windows tests must cover both dedicated-pipe bounds/EOF, all
  three crash cuts, noninheritance, and a nested child launch before this
  factory can qualify. No real process has been launched.

## 2026-07-31 - T0 fresh audits reject two native-handle ownership gaps

- The first remediated T0 candidate passed 118 tests and had zero P0/P1, but a
  fresh audit reproduced a P2 leak when real `threading.Thread.start()` failed
  after the capture object stored the thread but before the native thread
  started. The corrected implementation now detects an unstarted thread and
  closes its raw read HANDLE exactly once.
- A second unchanged-byte audit again found zero P0/P1 and passed the full
  suite, hard-parent-death 5/5, Ruff, format, strict mypy and compile. It then
  reproduced a distinct P2 asynchronous gap after successful
  `CreateProcessW` but before the returned `hProcess`/`hThread` structure was
  adopted by the caller. The Job correctly killed the child, but both native
  handles remained open.
- Both candidates are explicit STOPs and are ineligible for integration. The
  next implementation preallocates the process-information owner in the
  caller, passes it into the atomic create function, and transfers/zeros each
  handle exactly once. A real create-then-interrupt test must prove terminal
  child state and exact closure before another freeze.
- These are bounded, independently reproduced defects rather than repeated
  scientific work. No T0 authority, Q/E, supervisor job, or science process
  was published or launched. Formal status remains
  `PRIMARY_STUDY_COMPLETE`.

## 2026-07-31 - Qualify the disabled Q-v2 WIP and reject two further native-custody assumptions

- The external request-v2 controller WIP passed a fresh independent
  unchanged-byte audit. Its exact checkpoint is
  `q_v2_one_shot_controller.py`
  `7b1d802b99e3bd8feebc22e192654f26504b63d36265e46fc44cb764ca45fe78`
  (129485 bytes) and its test is
  `e7562e89009f411fbd6f61dce7c3d9b168bd0c1d2f3b09a6a9c1b8574bd77ac4`
  (140858 bytes). The exact suite returned **126 passed**; Ruff check,
  Ruff format check and `py_compile` passed. All nine production pins remain
  `None`, and the disabled CLI still exits 2 before reading a request. This
  qualifies only the external WIP contract; it does not authorize live
  integration or consume the one allowed Q replacement-v2 publication.
- The next T0 candidate closed the earlier `PROCESS_INFORMATION` and
  attribute-list return gaps and passed 124 tests, hard-parent-death 5/5,
  strict mypy, Ruff, format, compile, and a 100-launch zero-handle-growth
  check. A fresh opcode-level Windows audit nevertheless reproduced P2
  CALL-to-STORE/UNPACK leaks for the Job, NUL input, and anonymous-pipe
  handles. The exact frozen candidate
  `88e85949fb366088528db89de3658e4a3e298c0cabd3697134f1dd0ee962c887`
  is therefore another explicit STOP and is not integrable.
- A local read-only RAII probe demonstrated a finite remediation for the raw
  HANDLE class: a `ctypes.c_void_p` owner used directly as the native return
  type closed a real Job handle when interrupted at the exact
  CALL-to-STORE opcode, and preallocated owned pipe endpoints closed both
  real pipe handles at the corresponding cut after the API had returned.
  The production WIP must still prove idempotent transfer, no double close,
  attribute-list success/failure ownership, and all declared crash cuts.
- A real Windows factory test rejected the planned controller-to-running-child
  Job-handle injection. The controller could round-trip the newly duplicated
  numeric slot, but the supervisor later observed that same value as a valid
  non-Job handle or `ERROR_INVALID_HANDLE`. This is a P0 ABA/handle-slot
  failure. It failed closed before `JOB_CUSTODY_ACCEPTED`, science, final
  publication, or retry.
- The corrected direction is receiver-initiated duplication after the exact
  Q/E ACK: the supervisor authenticates and retains the already-bound
  controller process, opens it with the exact bounded duplication/query/wait
  rights, duplicates the still-retained controller source Job into its own
  process as non-inheritable, and validates the local Job before ACCEPTED.
  The controller retains its source handle through exact receipt/pipe
  validation. The transfer policy and access/options must be bound in the
  nested staging-ready transport before resume. An independent schema and
  crash-semantics audit is in progress; no WIP bytes are frozen yet.
- The supervisor external-pre-resume validator now has a real synthetic
  checkpoint: one CREATE_SUSPENDED child with retained-handle identity passed,
  while creation-time PID-reuse and exited-target cases failed closed. Nine
  focused staging/recovery tests, Ruff and compile also passed. The full
  supervisor WIP remains unqualified pending the receiver-initiated Job
  contract and full unchanged-byte audit.
- A fresh read-only inventory found no primary, confirmatory, recovery,
  publication, armed-supervisor, or Codex-resume process. No raw PanNuke file,
  frozen preregistration, existing run, Q/E leaf, final supervisor job, or
  scientific artifact was modified. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**, and M9 remains locked.

## Next exact action - close the two finite native-ownership contracts

Finish and independently audit the common owned-HANDLE construction pattern
for T0 and the atomic factory. In parallel, freeze the receiver-initiated
outer-Job RELEASE/ACCEPTED schema, apply it consistently to the authority
builder, factory, and supervisor, and pass the real Windows ABA/crash-cut,
noninheritance, PID-reuse, bounded-pipe and nested-child tests. Only unchanged
zero-material-finding checkpoints may enter the reviewed live allowlist.

## 2026-07-31 - Freeze the three-phase outer-Job custody contract

- An independent read-only audit completed the replacement for the rejected
  remote-handle injection. The only qualifying ordering is now
  `JOB_CUSTODY_RELEASE` -> `JOB_CUSTODY_ACCEPTED` ->
  `JOB_CUSTODY_COMMIT`; science remains forbidden through ACCEPTED and until
  the supervisor has sealed and reread
  `outer_job_custody_committed.json`.
- The frozen closed inventories are: outer Job contract v2 **8 fields**,
  transport v2 **32**, source-to-supervisor duplicate contract **13**,
  controller round-trip contract **13**, RELEASE **17**, transfer preimage
  **10**, ACCEPTED **27**, controller round-trip evidence **21**, COMMIT
  **24**, and COMMITTED receipt **31**. The staging-ready top level remains
  exactly 39 fields; only its nested transport changes.
- The transport uses three separate bounded anonymous pipes: RELEASE
  controller-to-supervisor, ACCEPTED supervisor-to-controller, and COMMIT
  controller-to-supervisor. Every message is one canonical line followed by
  EOF with a 65536-byte inclusive limit. The exact child HANDLE_LIST is now
  six endpoints: Q/E read, Q/E write, RELEASE read, ACCEPTED write, COMMIT
  read, and stderr write.
- Receiver duplication opens the already Q/E-bound controller process with
  exact mask `0x00101040`
  (`PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE`).
  It verifies retained-handle PID, creation, image hash and liveness before
  and after duplication. The controller command SHA is honestly described as
  Q/E-bound; no live PEB/argv readback is claimed.
- After exact ACCEPTED pipe/receipt verification, the controller duplicates
  the supervisor-local Job back to one temporary handle, requires
  `CompareObjectHandles(original_source, temporary) == TRUE`, revalidates Job
  limits and supervisor membership, closes the temporary handle, and then
  closes the original source Job. Only a successful source close permits
  COMMIT. Ambiguous close, controller death, partial/extra/missing line or
  missing EOF produces zero COMMIT and zero science.
- The supervisor writes the exact COMMIT into a CREATE_NEW, sealed, retained
  31-field COMMITTED receipt, revalidates its local Job/membership/flags and
  noninheritance, and only then may create process C. No fourth ACK is
  required. Restart states containing ACCEPTED without COMMITTED, or
  COMMITTED without a qualifying terminal seal, are permanent STOP states
  with no adoption or retry.
- The audit additionally requires owner-on-stack/RAII handling for
  `OpenProcess` and both `DuplicateHandle` output slots, plus real Windows
  crash, ABA/reuse, PID mismatch, CompareObjectHandles=false, three-pipe
  bounds/EOF, source-close ambiguity, nested-child noninheritance and restart
  tests. Authority, factory and supervisor implementations are in progress
  only in external WIP directories.
- No live source, Q/E leaf, supervisor release, existing run, frozen
  preregistration, raw PanNuke input or scientific artifact was changed.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - implement identical schemas and pass the real Windows matrix

Require byte-for-byte schema/key/policy parity between the staging authority,
atomic factory and supervisor. Then run the complete synthetic three-phase
success and crash matrix, freeze exact hashes, and perform independent P0/P1
and native-ownership audits before any reviewed live integration.

## 2026-07-31 - Integrate the independently qualified T0 carrier only

- The final independent exact-byte T0 audit returned `READY` with zero P0,
  P1, or P2 findings. Its integration-scope evidence was 144 passing tests,
  29 focused RAII/attribute/custody tests, hard-parent-death 5/5, a
  100-launch handle soak with delta zero, and an exact native
  `CreateProcessW` return-boundary cut that closed all eight acquired handles
  exactly once. Ruff, format, strict mypy, and compilation also passed in the
  external WIP.
- The live integration used only the frozen six-path allowlist. Four added
  files were rehashed byte-for-byte after `apply_patch`:
  publisher source
  `6BE792901A6A9CD090DB4443FE129BDFFC47B19414A0D96D6234BB88291C5EC4`
  (151328 bytes), reviewer source
  `5AF21AA5831877D449A8DA20123A9E4D69B67C30E86C093CD14DBCB71073919A`
  (9641), publisher tests
  `06D5D0C75E4173E0C23206998D39D2EBACF1002E0FBEC519ACF8316DFB97400B`
  (120566), and reviewer tests
  `E0E2CF7A412A8F57A9266886ACD7D6FC1152229A4AB791D9F174DA761B17BDD5`
  (25193). The other two changes are only the main CLI registration and its
  test. No shim, prototype, carrier, cache, or WIP documentation was copied.
- Live focused QA command:
  `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
  tests\test_original_confirmatory_technical_authority_publication_workflow_v1.py
  tests\test_original_confirmatory_technical_authority_review_producer_v1.py
  tests\test_original_confirmatory_technical_authority_main_cli_v1.py
  tests\test_stage_cli.py` -> **99 passed in 23.71 s**. An earlier invocation
  was terminated by an erroneously short tool timeout before pytest completed;
  it produced no scientific artifact and the complete rerun above passed.
- Scoped `ruff check --no-cache`, `ruff format --check`, and
  `mypy --no-incremental` passed. The new main CLI exposes exactly
  `build-intent`, `review-intent`, `publish`, and `verify`. The functional
  direct `experiment confirmatory` command still exited 2 with
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED`.
- T0 is source-integrated but has not been published. No one-use authority,
  Q, E, supervisor job, Codex wake, or scientific process was launched.
- A fresh independent audit rejected the previously frozen Q-v2 bytes after a
  real Windows `sys.monitoring` cut exposed a `CreateFileW` CALL-to-STORE
  handle leak (process handle count 150 to 151). That Q snapshot therefore has
  an empty live allowlist. A new external replacement WIP is repairing all
  equivalent HANDLE and CRT-descriptor acquisition/transfer boundaries with
  owner-on-stack RAII and crash-cut tests.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - qualify Q replacement and close E/supervisor parity

Finish the new Q-v2 RAII replacement and independently audit unchanged bytes.
In parallel, complete identical authority/factory/supervisor implementations
of the frozen 8/32/13/13/17/10/27/21/24/31 three-phase custody contract.
Only after their synthetic Windows crash, ABA, PID-reuse, pipe-bound,
noninheritance, restart-STOP, and handle-soak matrices pass may the reviewed
allowlists enter live integration and the mandatory full repository gates run.

## 2026-07-31 - Pass the complete live gates after T0 integration

- The first complete live pytest run finished normally with **2260 passed,
  1 skipped, 3 failed in 939.58 s**. Two failures were stale tests that still
  expected the now-superseded direct confirmatory lifecycle/adapter/executor
  path. The third was the closed parent-to-live source-delta fixture, which
  did not yet register the three qualified T0 source modules. These were test
  contract failures, not runtime, data, or scientific failures.
- The two CLI tests now require the stronger current invariant: exit 2 with
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED`, zero lifecycle/gate/adapter/
  executor calls, and zero run-directory creation. The independent
  source-delta fixture now explicitly registers only the three added T0
  modules. Focused rerun:
  `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
  tests\test_confirmatory_cli_inputs.py
  tests\test_resource_bounded_authority_gate.py` -> **31 passed**; scoped Ruff
  and format-check passed.
- The mandatory complete rerun
  `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` passed:
  **2263 passed, 1 skipped in 953.78 s**. The sole skip is the existing
  documented Windows rename/open-file semantic case in
  `test_pannuke_publication_read_toctou.py`.
- `.venv\Scripts\python.exe -B -m ruff check --no-cache .` passed;
  `.venv\Scripts\python.exe -B -m ruff format --check .` reported all
  **204 files** formatted; and
  `.venv\Scripts\python.exe -B -m mypy --no-incremental src` reported no
  issues in **99 source files**.
- The required real-data functional command
  `.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke` passed in 314.4 s with
  `status=valid` and `validation_scope=full_semantic_scan`: 3 folds, 7901
  patches, 22 raw files, 4318 cross-class-overlap pixels in 575 patches,
  10486091 void pixels in 162 patches, and 7164 normal patches. It identified
  1411 overlap-touching instances and excluded exactly those 1411 from both
  primary and confirmatory analysis. It reported zero class arbitration,
  identical primary/confirmatory policy, and `source_masks_modified=false`.
  The idempotent QC selection root is
  `09886588591d9ebb9a725db1022bb0ab8fb94b4bcca419b486e2549b0cc5fd36`.
- External work remains fail-closed. The E/factory six-file custody protocol
  passed an independent strict-type and schema audit, but it is not globally
  integrable until a separate outcome-blind, source-pinned fresh-180 E
  material builder passes. Supervisor candidate `6DB782...` was rejected for
  raw `OpenProcess`/`DuplicateHandle` ownership windows and independent
  strict-type gaps in ATTEMPT33, READY39, spec44, and auth44. Q's native-RAII
  checkpoint passed, but its subsequent typing checkpoint was rejected after
  repository-context mypy exposed three remaining errors. New external
  snapshots are being built; none has entered live execution source.
- No T0 was published; no Q or E was written; no supervisor was armed; and no
  primary, confirmatory, recovery, publication, or Codex-resume process was
  launched. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - close external control-plane candidates without weakening gates

Finish and independently re-audit the Q typing-clean content-addressed
controller, the source-pinned fresh-180 E material builder, and the supervisor
owned-OpenProcess/preowned-DuplicateHandle plus strict-type fixes. Then perform
one final cross-module field-set/active-path/native-ownership audit before any
external release or live integration.

## 2026-07-31 - Integrate the qualified lifecycle/STATIC-v3 carrier

- The independent external audit of
  `AANCA-lifecycle-static-v3-rebase-wip-20260731` returned **PASS** with
  P0/P1/P2 = **0/0/0**. It reconfirmed the closed STATIC-v3 shape of 24
  fields, published-T0 composite shape of 10, nested lifecycle shape of 22,
  all nine flat-to-nested equality checks, and exactly one public
  `verify_live=True` followed by one private unchanged-carrier
  `verify_live=False` check at each protected entrypoint.
- The live integration used only the report's 12-path allowlist. Every live
  path was replaced through `apply_patch` and then rehashed byte-for-byte.
  The six source hashes are `1210c18b...0ae5e`, `32b271de...f3fdc`,
  `700934bd...a9b7e`, `27d02023...79c8d`, `751ea676...db0f`, and
  `deeca995...5434`; the six test hashes are `4578a059...4792f`,
  `e4ec2997...cf96`, `276dd51e...18f9`, `c12a0bda...0ea7`,
  `3fefea37...9238`, and `9232aa35...701`. Their exact sizes and full hashes
  remain recorded in the independently audited `REBASE_REPORT.md`.
- The external-only compatibility test
  `test_static_v3_publisher_candidate_compatibility.py` was deliberately not
  copied. A first focused pytest invocation was killed after five seconds by
  an erroneous tool timeout and made no scientific output. The complete live
  focused rerun over the six changed test modules passed:
  **336 passed in 121.16 s**.
- Scoped `ruff check --no-cache` passed, `ruff format --check` reported all
  12 files formatted, strict mypy reported no issues in the six source files,
  and read-only compilation passed for all 12 files. The functional direct
  command still produced
  `GATED [CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED]` with native process exit
  code 2, before lifecycle, dataset, cache, executor, or run-directory access.
- The first fresh-180 material-builder snapshot remains rejected and external.
  A deep coherent-tamper audit found two P0 defects: 57 transitively imported
  project modules were outside its 14-file pin set with a close-before-import
  TOCTOU window, and a substituted CNN cell ID was accepted after recomputing
  all dependent hashes because the 180-row projection was not cross-linked to
  the exact frozen 36-cell set. It also found two P1 schema/QA weaknesses.
  The successor must fix these; none of the rejected hashes may become a
  production pin.
- No T0 was published, no Q or E was written, no supervisor was armed, and no
  primary, confirmatory, recovery, publication, or Codex-resume process was
  launched. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - finish and cross-audit the external successor chain

Complete the content-addressed release publisher/verifier, release-aware Q,
release/state-separated E/factory, full-closure fresh-180 builder, and
native-RAII supervisor. Require independent unchanged-byte audits and one
cross-module schema/active-path/native-ownership gate before any one-use
publication or scientific process. After the final bounded live integration,
rerun the complete repository gates and real PanNuke validator.

## 2026-07-31 - Pass all mandatory gates after lifecycle/STATIC-v3 integration

- The independent read-only live audit returned **PASS**, P0/P1/P2 =
  **0/0/0**. It found 12/12 byte-identical allowlisted files, no external-only
  compatibility test in live, exact STATIC24/composite10/lifecycle22 parity,
  unchanged authority field inventories, all nine fail-closed equality
  checks, public verification order `[True, False]`, and direct CLI exit 2.
  Its focused evidence was 344 passing tests plus four contract tests.
- The complete repository command
  `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` passed:
  **2301 passed, 1 skipped in 933.10 s**. The sole skip remains the documented
  Windows open-file rename semantic in
  `test_pannuke_publication_read_toctou.py`.
- `.venv\Scripts\python.exe -B -m ruff check --no-cache .` passed;
  `.venv\Scripts\python.exe -B -m ruff format --check .` reported all
  **204 files** formatted; and
  `.venv\Scripts\python.exe -B -m mypy --no-incremental src` reported no
  issues in **99 source files**.
- The functional direct confirmatory command printed
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED`; PowerShell captured its native
  process exit as exactly 2. The required real-data command
  `.venv\Scripts\python.exe -B -m histo_audit data validate-pannuke
  --project-root . --root data\raw\pannuke` passed in 317.6 s with
  `status=valid`, full semantic scan, 3 folds, 7901 patches, 22 raw files,
  4318 overlap pixels in 575 patches, 10486091 void pixels in 162 patches,
  and 1411 overlap-touching instances excluded identically from primary and
  confirmatory analysis. It again reported no class arbitration and
  `source_masks_modified=false`; QC selection SHA remains
  `09886588591d9ebb9a725db1022bb0ab8fb94b4bcca419b486e2549b0cc5fd36`.
- Git ignore readback again proved raw ZIP, NPY, nested manifest Parquet, and
  duplicate-audit NPZ paths ignored. The repository still has no baseline
  commit, so the whole project appears untracked; this is recorded as an
  operational limitation and is not silently repaired during scientific
  gating.
- The release/state-separated E/factory successor passed an independent
  bounded-WIP audit with P0/P1/P2 = **0/0/0**, 62 tests, Ruff, format, strict
  mypy, compilation, and real Windows RAII checks. Its exact closed shapes
  are spec47/auth47/READY42 and its mutable staging/job paths derive only from
  the state root. It remains STOP for publication or execution until final
  parity with the separately qualified release verifier and supervisor.
- No T0 was published, no Q or E was written, no supervisor was armed, and no
  primary, confirmatory, recovery, publication, or Codex-resume process was
  launched. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, and M9 remains locked.

## Next exact action - qualify the immutable external control-plane release

Finish publisher and independent verifier Windows matrices, freeze their exact
bytes, and require a zero-finding independent audit. Then cross-audit the
release-aware Q successor, E/factory, corrected full-closure fresh-180 builder,
and supervisor before the one authorized Q replacement write.

## 2026-07-31 - Pin the pending Codex handoff identity without inventing a wake

- `CODEX_THREAD_ID` is exactly
  `019f703b-661d-7c50-b423-9270657d8d6d`. A read-only session-store check
  found exactly one matching JSONL session file. `codex --version` reports
  `codex-cli 0.130.0`, and `codex exec resume --help` confirms the explicit
  positional `SESSION_ID` plus prompt interface.
- No resume command was executed while this task is active. The existing
  handoff evidence names an older session and therefore remains invalid for
  the current handoff; it will not be rewritten to claim a test that did not
  happen. The exact current-session resume is reserved for one controlled
  short-process handoff after all non-wake supervisor gates pass.
- A read-only Windows inventory found no AANCA or Codex Scheduled Task. The
  event-driven design continues to prohibit scheduled polling, heartbeat
  polling, `--last`, automatic scientific retry, and concurrent long jobs.

## 2026-07-31 - Qualify the release-aware Q replacement without writing Q

- The independent exact-byte audit of
  `AANCA-q-v2-controller-release-state-root-wip-20260731` returned **PASS**,
  P0/P1/P2 = **0/0/0**. Exact qualified bytes are controller source
  `FC83539AE933E749809D4D7C75AB7D2C9D335C1B4837E210EAF01B19BB58A96E`
  (242728), tests
  `C7BF55BE7F005B5B8CC686D6686E8EC04CD3A3A3B71DC230B53E096BD002BAFF`
  (207025), README
  `C6B403B76F8094AAB5E8A045C03A94A0C07AB46F9CAA7AEE1B301A1025F69458`
  (15117), and schema
  `9E0DC89945790E8023E8E90BAD8DA3EEA7C17E5D436E05C485FC90D6A8109697`
  (11897).
- Independent QA passed **155 tests**, repository-context Ruff and format,
  repository mypy, strict mypy, and compilation. The disabled production CLI
  exited 2 before even attempting to read a nonexistent request.
- The audit covered exact publisher-command-8/verifier-command-6 records,
  disjoint release/code/state paths, closed qualification links, two retained
  filesystem passes, ADS/DACL/protected/readonly/reparse checks, publisher
  versus verifier source-SHA and process-instance independence, CREATE_NEW
  transaction evidence, and all false no-science/no-retry flags.
- This qualifies source bytes only. No Q replacement was written, consumed,
  or published. The one user-authorized Q write remains reserved until the
  immutable external release containing these exact bytes is itself
  independently qualified and verified.

## 2026-07-31 - Qualify the full-closure fresh-180 material builder

- The independent unchanged-byte audit of
  `AANCA-fresh-180-e-intent-builder-successor-wip-20260731` returned **PASS**,
  P0/P1/P2 = **0/0/0**. Exact four-file qualification root is
  `93926caa1e5acd96ad45d225645bc2671bb3f0ecc2c038796656af3560cabc3e`.
  Exact files are builder `71212e064cddea8009a4bde1d7d678ff100faffdaee5027af830e7b59140d31e`
  (102710), tests
  `294bf6851967c444ba226f2f7166ba5b49fc63007dc56f440a89b44d3be651b6`
  (37975), schema
  `317204f8c67526d4449b4dfe028a23c94d8c4b4db1a0cc7d602cf27dad915e9e`
  (31502), and README
  `41fcda907bb52cb7b8252816447c5e0a1ff0bd6e19d0d5bdd5ff7812eba7e951`
  (5230).
- The auditor independently reproduced 114 source records with root
  `584fe76cc439a716ec28cb4aefaef60f0b0d5d4c1163e44476c821f091f2f8e5`
  and compact-JSON-plus-LF manifest
  `b313fe678d68fe61470c3921f65f5ace7ccb892e1c07a13936b8c792601403b3`.
  It verified retained-byte loading of 105 Python modules plus the derived
  namespace, zero ambient project imports, default-deny unpinned imports, and
  a final source recapture that stops on drift.
- Exact 108-cell and 36-CNN-cell roots were reproduced; every CNN cell has
  five folds. A coherent five-row cell-ID substitution with all dependent
  hashes recomputed was rejected. Configuration, model, preprocessing, and
  data-split nested schemas also rejected coherently resealed mutations and
  extra fields.
- Independent QA passed **51 tests**, repository Ruff/format, repository and
  strict mypy for source plus tests, in-memory compile, schema references and
  closed-object checks, with zero cache artifacts. The optional external
  `jsonschema` package was unavailable; this is recorded, while the project
  schema tests and an independent structural validator passed.
- The builder performs only outcome-blind retained source/config/cache/label
  capture and contract construction. It has no result discovery, training,
  publication, subprocess, or write path. The rejected first `14379...`
  snapshot remains excluded and is not a production pin.

## 2026-07-31 - Revoke the first release/state E/factory qualification

- A later independent cross-module parity audit found a coherent exact-type
  bypass that the earlier bounded E/factory audit missed. The AUTH47
  canonicalizer accepted `schema_version=2.0`, `max_attempt_count=True`, and
  `automatic_retry_allowed=0`; the SPEC47 canonicalizer accepted
  `schema_version=2.0` and floating-point values for integer timeout/log
  fields after all dependent roots were recomputed.
- The cause is Python mapping equality, where `True == 1`, `False == 0`, and
  `2.0 == 2`, being used before exact type identity was established. This is
  P0 against the frozen strict-JSON-type rule.
- Authority `848DA97D...` and factory `4B4DD632...` are therefore revoked and
  must not be integrated, pinned, published, or used to build E. A separately
  versioned successor must validate exact types for every SPEC47, AUTH47,
  READY42 and nested field before value/root equality, and must pass
  coherently resealed wrong-type-equal mutations plus the full Windows RAII
  matrix.
- This detection occurred before any release, Q write, E write, supervisor
  arming, or science. Q and full-closure fresh-180 source qualifications are
  unaffected; the cross-module STOP operated as designed.

## 2026-07-31 - Reject the first release-tools freeze for incomplete alias denial

- The fresh independent release-tools audit produced two isolated,
  reproducible counterexamples after otherwise valid synthetic publication.
  A plain control-root `active` pointer file and an arbitrary
  `opaque-alias` directory symlink to the exact release were both accepted by
  the verifier, which then falsely attested
  `execution_via_pointer_or_alias_absent_verified=true`.
- The verifier only enumerated selected `current`/`latest` names and `*.lnk`;
  it did not default-deny every unexpected control-root entry or arbitrary
  reparse alias. This is P0 against the no-pointer/no-alias execution
  contract.
- Release-tools hashes `a98087d3...`, `f20dc220...`, `9b19c3b0...`,
  `467cf003...`, `8e13f1b6...`, and `b4d650c3...` are rejected and must not
  be frozen, published, or used for Q. The auditor removed both synthetic
  probes and is completing the remainder of the read-only audit before a
  separately versioned fix.
- The prior 52-test local matrix is retained as evidence but is insufficient;
  the successor must default-deny all non-allowlisted control-root entries
  and prove arbitrary file, directory, junction, symlink, mount-point, and
  shell-link aliases fail closed.
- The same independent audit found a second P0 terminal-contract flaw.
  Publisher SUCCESS was durably written before fallible source-lease and
  singleton-mutex cleanup. A test that performed the real mutex release and
  then reported failure left a complete readonly SUCCESS, raised a publisher
  error, and was nevertheless accepted by the independent verifier.
- A successor must not expose a qualifying SUCCESS while any later cleanup
  can change the command to failure. Its design must either make SUCCESS the
  final fallible operation after all cleanup, or add a separately
  authenticated parent process-exit attestation; merely catching or ignoring
  cleanup failure is not acceptable.
- Causal analysis selected the parent-attestation option. One separately
  qualified waiter will hold the publication singleton, launch publisher and
  independent verifier sequentially, wait on retained process handles without
  polling, require both exit codes to equal zero, and revalidate both terminal
  receipts and the release.
- Only then may the waiter CREATE_NEW, seal and read back one combined
  `release_qualification_attestation` binding both child process/command/
  receipt roots, the release root, waiter identity/qualification, and
  no-science/no-retry flags. That commit, not either child receipt or the
  waiter's future exit, is the terminal trust anchor.

## 2026-07-31 - Require a live Q17/release44 canonicalizer successor

- The completed cross-module audit returned **STOP**, with 2 P0, 2 P1, and 1
  P2 findings. Although the external Q candidate is internally request34,
  pins13, Q11/17 and release44, current live authority/bootstrap still expose
  Q12 and release42 with a single `supervisor_root`.
- Q17 adds `attempt_identity_projection`, `attempt_identity_root_sha256`,
  `control_staging_projection`, `control_staging_projection_sha256`, and
  `q_base_authority_root_sha256`. Release44 replaces stale
  `supervisor_root` with
  `external_control_plane_release_root_sha256`, `supervisor_code_root`, and
  `supervisor_state_root`.
- The qualified external Q delegates canonicalization to the live authority.
  It therefore cannot legally execute against the current live contract even
  though its isolated audit passed. `FC83539A...` remains valid evidence of
  isolated source qualification but is not an integrable or publishable
  production candidate.
- A new external live-authority/bootstrap successor is now being built to
  implement Q17/release44 and recursive exact types while preserving the
  already qualified STATIC24/composite10/lifecycle22 shapes, nine equality
  checks, and public `[True, False]` verification order. In parallel, new
  E/factory and Q successors remove stale aliases and add coherently resealed
  nested type tests.
- Direct confirmatory CLI still exits 2 before lifecycle, dataset, cache,
  executor, or run-directory access. No release, Q, E, supervisor, wake, or
  science transition occurred.
- A pre-freeze parity review also caught an obsolete process-derivation
  vector using `run --spec`. It was superseded before any hash was frozen.
  The authoritative protected launch is the exact 12-element staged bootstrap
  with `--staged-launch-spec` and `--staged-e-intent`, both under the mutable
  state root; direct arbitrary `--spec` is forbidden.
- The corrected semantic checkpoint is process-derivation v2 exact39. It
  separates code and state roots; binds source, launcher, logical/runtime/live
  Python identities; requires `-I -S -B`; binds both staged paths from
  CONTROL_STAGING-v2; constructs exactly the 12-element Python argv and
  complete OS vector; requires PEB/in-process parity; forbids extra argv/cwd
  and launcher execution; and hashes the other 38 typed fields.
- Its rich 10-field process command hash and the factory's compact four-field
  launch-plan hash are intentionally distinct domains. Each is independently
  recomputed, exact program/argv/cwd components are cross-linked, and swapping
  one digest into the other domain must fail. No source hash is frozen until
  bootstrap fixtures and focused QA pass.

## 2026-07-31 - Reject direct final-path commit in the release waiter

- A causal review of the in-progress release-waiter v2 found that its first
  implementation created the authoritative
  `release_qualification_attestation.json` path before later fallible
  write/fsync, sealing, DACL, reopen, close, retained-handle and list-update
  operations. A valid-looking terminal attestation could therefore remain
  while the waiter subsequently exited nonzero. This repeats the class of
  ambiguity that invalidated the first publisher SUCCESS design.
- The in-progress waiter bytes are **not frozen or qualified** and must not be
  published or consumed by Q. No control-plane release, Q, E, supervisor,
  Codex resume, or scientific process was started.
- The required successor protocol prepares and fully validates a
  non-authoritative transaction-local staging leaf first. The single
  no-replace handle-based rename to the exact final path must be the last
  fallible operation; afterward only a preallocated committed-state store and
  immediate process exit are permitted. Fault injection must prove that every
  precommit failure leaves no authoritative final path.
- The release will use a new empty, versioned LOCALAPPDATA control root.
  Historical control-plane evidence will not be deleted, renamed, or
  repurposed. A nonempty legacy root must fail closed.
- Formal project status remains `PRIMARY_STUDY_COMPLETE`. The next command is
  the corrected waiter's full synthetic gate followed by a fresh independent
  read-only audit; the one authorized Q replacement-v2 write remains
  unconsumed.

## 2026-07-31 - Reject E/factory exact10 candidate after custody type bypass

- The E/factory candidate with provisional exact10 inventory digest
  `75fa00e530af5f2f7ba6ee562130c8c2a2fa137244306706085c387a8f3f14bb`
  passed its local 84-test and static QA gates but failed the independent
  adversarial audit. It is unfrozen and **must not be integrated or pinned**.
- The accepted/evidence/commit custody canonicalizers compared retained wire
  handle values with ordinary Python equality before proving exact types. A
  coherently resealed mutation from an integer handle to its equal floating
  value was accepted in three custody records. This is the same
  `int`/`float` equality class that the earlier exact-type hardening was meant
  to eliminate.
- The successor must apply exact wire-handle validation and recursive
  type-aware comparison to every controller-source and supervisor-local handle
  in accepted, evidence, commit and nested committed records. Coherently
  resealed integer-to-float and integer-to-boolean mutations must fail at the
  type predicate, not only at a stale enclosing hash.
- The same audit found a second alias in the exact10 attempt-identity
  canonicalizer: ordinary whole-mapping equality accepted
  `schema_version: true` as equal to integer `1` and silently returned the
  canonical integer. The successor must use a complete exact-type map before
  value equality and must systematically replace every remaining
  type-insensitive expected-mapping/list comparison in both modules.
- No live repository source, Q, E, release, supervisor state, Codex resume or
  scientific artifact was changed by this audit. Formal status remains
  `PRIMARY_STUDY_COMPLETE`.

## 2026-07-31 - Hold live Q17 carrier after independent type-alias probe

- Before any source hash was frozen or integrated, a read-only probe of the
  live-authority successor found the same Python-equality hazard in its Q/E
  custody contract canonicalizer. With the original self-hash retained, it
  accepted and silently normalized `schema_version: true`, an integer timeout
  changed to its equal float, and integer zero changed to `false`.
- Direct deterministic mapping equality also occurs in the Q/E custody
  receipt, ACK and downstream supervisor-spec canonicalizers. The live
  successor and dependent Q/supervisor candidates are therefore on **HOLD**;
  their current bytes and passing local test counts do not qualify them.
- The authority already contains a recursive type-aware JSON comparator. The
  successor must use it for every deterministic mapping/list comparison,
  retain explicit scalar type predicates, and pass coherent bool/int/float
  alias tests across contract, receipt, ACK, spec and other nested projections
  before a new byte inventory is proposed.
- A later independent full-chain probe found that recursive outer equality was
  still insufficient: the nested control-file physical-identity and E-ancestor
  lease canonicalizers compared `schema_version` to integer `1` without first
  proving its type. Both `true` and `1.0` survived a coherently rebuilt
  READY, receipt, ACK and downstream-spec chain. Every nested schema/count/
  size/PID field therefore requires its own exact scalar predicate before the
  corrected carrier can freeze.
- The probe performed no publication or write and read no scientific outcome.
  Frozen `SPEC.md`, `PRE_REGISTRATION.md`, scientific configs and existing
  runs remain unchanged.

## 2026-07-31 - Qualify the system-fixed E/factory exact10 successor

- A fresh independent before/after audit of the corrected external E/factory
  snapshot returned **PASS**, P0/P1/P2 = **0/0/0**. Its exact10 inventory
  digest is
  `35d5dec04d20e33ea233c7eb0c218039ccf0639df81754995b8aee23d713a63f`
  and remained identical before and after all checks.
- Full QA passed **86 tests**, a focused five-test adversarial gate, Ruff
  check/format, strict mypy on all four Python sources, compilation of all four
  sources, and both JSON schemas. README now records the exact four-source
  strict-mypy command.
- The independent matrix rejected 42 coherently resealed wire-handle
  mutations across transport, release, accepted, evidence, commit, committed
  and nested records, covering both equal floats and `True`. Attempt exact10
  rejects boolean schema aliases and invalid retry types.
- Static review found zero whole-mapping ordinary equality paths, zero raw
  wire-handle equality paths and zero integer coercions of parsed wire
  handles. Thirteen factory and nine synthetic-receiver exact handle gates
  cover the custody chain. Rich/compact hash-domain substitution, staged
  launch argv, stale root keyword, RAII, double-start, crash cuts and native
  handle-object comparison tests also passed.
- Exact qualified source hashes are authority
  `4ba53986041a01ef6dc0e4ae65a324411c4cad722e02291d553f42b4379f8a3e`
  and factory
  `61b0683f73285425da8b2320d18256ffe17ba13f311c1be554f3f3f847d4f1d2`.
  This qualifies immutable candidate bytes only; no E, Q, release,
  supervisor, resume or scientific process was created.

## 2026-07-31 - Reject wrapper-import execution of the release waiter

- Independent discovery found a P0 in the moving release-waiter WIP before
  freeze. Its self-command record used the module `__file__` and checked only
  cwd/source equality, so importing the module through `python -` produced an
  apparently pinned waiter source while the actual interpreter argv executed
  stdin wrapper code. Such a wrapper could alter imported module state without
  changing the recorded waiter source hash.
- The production successor must require the exact direct invocation
  `python -I -S -B waiter.py <authorized-tail>`, full `sys.orig_argv`,
  in-process `sys.argv` and live PEB command-line parity, plus retained
  program/source identity before reading authority or mutating the control
  root. `-c`, stdin, import/module and wrapper execution must STOP with zero
  attempt marker and zero child processes.
- Publisher and verifier children require the equivalent direct-shape check
  in addition to their inherited one-use capability and actual retained parent
  identity. Current waiter bytes remain unfrozen and unqualified.
- The same discovery pass found two further release-waiter blockers. First,
  the generic terminal-artifact reader accepted a coherently rooted VERIFIED
  receipt with an integer in a boolean independence field because it did not
  run a receipt-specific recursive canonicalizer. Second, an extra
  control-root file created after the final inventory scan but before the
  terminal rename survived into a successful attestation that still claimed a
  complete default-deny inventory.
- The successor must fully canonicalize every publisher/verifier receipt field
  and crosslink. It must also freeze the exact namespace against new
  write/delete opens under retained directory custody before the final scan,
  then perform a handle-relative no-replace rename using rights acquired
  before the freeze. A race injected at the exact precommit pause must either
  be denied or leave the final authority path absent.

## 2026-07-31 - Hold supervisor recovery proof on nested exact types

- Independent read-only discovery of the moving supervisor WIP found that
  terminal recovery evidence closed only its outer record. The nested
  `supervisor_exit_proof` accepted boolean and floating-point aliases for PID
  and exit code and accepted an unexpected field; a later verifier also used
  ordinary nested mapping equality.
- This is classified P1 because the observed path is durable restart/STOP
  evidence and no automatic relaunch or false scientific SUCCESS path has been
  demonstrated. The supervisor nevertheless remains unfrozen: each prior-exit
  proof variant must have a closed schema, exact integer-or-null predicates and
  recursive verifier comparison, with coherent restart mutation tests.
- No startup recovery, Codex wake, long process or scientific command was
  executed during the probe.

## 2026-07-31 - Require bootstrap-side Q/E custody validation

- Independent discovery found that the moving capsule bootstrap accepted the
  held supervisor run spec from its exact 47-field set and self-computed hash
  without independently canonicalizing the nested Q/E custody contract,
  handoff, receipt and downstream binding. Coherent envelopes containing a
  boolean contract schema and a floating-point nested identity schema passed.
- This is P1 defense in depth because the current supervisor producer rejects
  those aliases, but a self-hash is not proof of authority semantics. The
  bootstrap successor must validate closed nested schemas, exact scalar types,
  deterministic job-directory derivations, roots and Q/E crosslinks before
  any project import or scientific-input read.
- Live carrier freeze and integration remain on HOLD pending independent
  full-envelope rejection tests.

## 2026-07-31 - Current next exact action after converged HOLD audit

- Formal status is exactly `PRIMARY_STUDY_COMPLETE`. Option-B primary recovery
  is complete and consumed; it and the historical finalization-successor path
  must not be re-entered. Original confirmatory has not run and M9 remains
  locked.
- The immediate next command, after the moving waiter fixes above are saved,
  is:
  `.\.venv\Scripts\python.exe -m pytest
  "$env:LOCALAPPDATA\AANCA-control-plane-release-tools-successor-v2-wip-20260731\test_external_control_plane_release_tools_v2.py"
  "$env:LOCALAPPDATA\AANCA-control-plane-release-tools-successor-v2-wip-20260731\test_external_control_plane_release_qualification_waiter_v2.py"
  -q`.
- A passing local command is followed by a fresh unchanged-byte independent
  audit of the release tools. Then the corrected live carrier must pass its
  full focused QA and independent full-envelope type audit; the supervisor
  must pass its closed restart-proof, exact-type, synthetic handoff and
  independent audit gates.
- Only after those three frozen inventories agree may Q be rebuilt against the
  final versioned release root and final authorization/attempt/attestation
  schemas, rerun its full disabled-before-read QA, and receive a fresh
  independent audit.
- Until all of those gates pass: no control-plane publication, Q write, E
  write, supervisor arming, exact-session resume, startup recovery, real long
  process, confirmatory execution or scientific-result read is permitted.

## 2026-07-31 - Integrate and independently qualify the live Q/E carrier

- Integrated the qualified live-carrier allowlist and its one discovered
  dependent call site. The final six-file inventory is 1,615,176 bytes with
  canonical root
  `89A21A001BB9307C7657807753E4FD6381CA0DFB931C7AB3E5FC5C31E1EB5CD7`.
  Exact source hashes are bootstrap `ACAC6291...9774C`, authority
  `CCD80343...C36F0`, terminal `39C894FB...516D5`, bootstrap tests
  `7CF2F714...5A35B`, authority tests `2C99E08B...56F29`, and terminal tests
  `9493AC77...982F`.
- The bootstrap now invokes both closed Q/E validators on the real structural
  and trusted pre-import paths. Coherently resealed boolean/float aliases,
  extra fields and broken crosslinks stop before project import or E claim;
  downstream supervisor `schema_version=2.0` stops before the
  `process_started` read.
- Repository Ruff required one import-only reorder after the external WIP
  handoff. A subsequent mypy run found that the existing terminal caller still
  passed final `run_spec.json` under an obsolete keyword. The repaired caller
  now takes `supervisor_launch_spec_path` and staged `e_intent_path` from the
  already canonical Q `control_staging_projection`; final `run_spec.json`
  remains a separate held and validated artifact. A regression test preserves
  that distinction.
- Executed gates: live allowlist `257 passed`; focused adversarial matrix
  `45 passed`; authority plus terminal `141 passed`; focused dependent
  regression `3 passed`; Ruff and format PASS; configured `mypy src` PASS for
  99 source files; compile PASS. Two unchanged-byte independent audits both
  returned P0/P1/P2 = **0/0/0**.
- The first full repository `pytest -q` attempt reached its 15-minute command
  limit and was terminated without a test verdict. On the second attempt the
  suite completed with `2317 passed, 1 skipped, 24 failed`; all failures were
  in the synthetic technical-authority publication workflow. The shared cause
  was a fixed 2026-07-31 UTC timestamp that had become earlier than the live
  test controller process. The synthetic timeline now derives deterministically
  from that process creation time while preserving intent < attempt < review <
  publication ordering. The complete affected file then passed `78 passed`,
  Ruff and format. A final full repository run remains mandatory after the
  remaining release/Q/supervisor integrations.

## 2026-07-31 - Replace the impossible global namespace-freeze claim

- A real Windows probe proved that changing the destination-parent DACL to
  deny new additions also makes the later same-token rename fail, even when a
  target handle was opened first. Directory share locks likewise block the
  publisher's own rename. The earlier proposed combination was therefore not
  implementable and is not claimed.
- The release successor now constructs one dedicated per-publication
  qualification capsule containing exactly four sealed leaves, applies and
  reads back its protected source-tree policy, and atomically renames the
  complete directory no-replace to `verifications/<publication_id>` as the
  last fallible operation. A final-name collision is permanent STOP. The
  attestation claims exact capsule inventory, not immutability of unrelated
  siblings in the global control root.
- Runtime/schema shapes are exact 17/19/18/13/57. The owner suite passed
  `102 passed`, Ruff, format, strict mypy for five files, compilation and JSON
  schema checks. An independent pass found one P1 cleanup path where a failed
  nonblocking process-status query could raise before the child was ended.
  The causal repair now treats that observation as nonterminal, ends and waits
  through the retained process handle, and closes through nested fallbacks;
  its five focused lifecycle tests pass. Final unchanged-byte independent
  release audit is still running. No release was published.

## 2026-07-31 - Keep the supervisor component qualified but the handoff pin on HOLD

- Supervisor behavior gates passed `174 passed, 194 subtests passed`, the ten
  required synthetic scenarios, Ruff, format, strict mypy and compilation.
  Recovery proof/exception records are now closed and recursively exact.
- Those results qualify component behavior only. The WIP still embeds the old
  session `019faaf3-c547-79e1-b0eb-26e35d214642`; current governing evidence
  requires `019f703b-661d-7c50-b423-9270657d8d6d`. The old successful receipt
  must not be relabelled as a test of the current session, so the supervisor is
  not admitted for final integration or arming.
- After final release and all non-wake parity gates, exactly one controlled
  short-process `codex exec resume` of the current session may create the new
  authentic receipt. Failure or ambiguity is STOP with no second attempt.

## 2026-07-31 - Current next exact action after live integration

- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; original
  confirmatory has not run and M9 remains locked.
- Complete the unchanged-byte release reliability audit of allowlist root
  `864F39E9...9409`. On zero findings, update the Q checkpoint against the
  final capsule-aware release schemas/root and rerun its disabled-before-read
  QA plus one bounded independent audit.
- Then mirror the final release contract into the supervisor, close all
  non-wake parity gates, and perform the single short-process exact-session
  handoff test. Until then there is no production release, Q/E write,
  supervisor arming, startup installation, Codex resume or scientific process.

## 2026-07-31 - Final sealed-directory release tools qualified

- The successor described above completed its final unchanged-byte reliability
  audit. The exact eight-file inventory is 679,648 bytes with canonical root
  `DA9B0EF9353760A1E8DC1D555B34B6A46D7AEF1F81640878A63F3ED17C4A8CC5`.
  Exact hashes are contract `E720328F...F2A7`, README `EEB3C331...AA5`,
  publisher `6A63C819...1756`, waiter `85E3D829...E96E`, verifier
  `59391187...DCC8`, schema `54BFEC30...20D8`, waiter tests
  `525D1A2D...A8F0`, and tools tests `10A7E10D...F468`.
- Owner QA passed `103 passed`; the lifecycle repair passed its focused nine
  tests. Ruff, format, strict mypy, compilation and schema checks passed. The
  independent read-only audit reported P0/P1/P2 = **0/0/0**, identical hashes
  before and after, zero retained child processes and an absent production
  release root.
- This qualifies bytes and protocol only. It did not publish a release, write
  Q or E, arm the supervisor, wake Codex or start science. Earlier release
  roots, including `864F39E9...9409`, remain obsolete HOLD evidence and must
  not be selected.

## 2026-07-31 - Current next exact action after release qualification

- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; original
  confirmatory has not run and M9 remains locked. No primary, confirmatory,
  recovery or AANCA supervisor process is active.
- Freeze the Q replacement-v2 checkpoint against exact release root
  `DA9B0EF9...A8CC5` and exact capsule-aware 17/19/18/13/57 schemas. Then run
  its complete disabled-before-read suite and one unchanged-byte independent
  audit.
- In parallel, mirror the same root and schemas into the supervisor while
  retaining the old-session receipt as invalid/superseded evidence. After Q,
  supervisor, E/factory and fresh180 agree under non-wake tests, perform at
  most one controlled short-process resume of exact current session
  `019f703b-661d-7c50-b423-9270657d8d6d`. Until those gates pass, no
  production publication, Q/E write, supervisor arming or scientific process
  is authorized.

## 2026-07-31 - Extend the live Q/E carrier to scoped release exact48

- Cross-module integration exposed that the previous 44-field
  `supervisor_release` carried only the release root and could not identify one
  publication-specific FINAL2 qualification capsule without discovery. The
  live authority and pre-import bootstrap now carry four additional closed
  fields: publication ID, exact qualification-attestation path, attestation
  file SHA-256 and attestation semantic-root SHA-256.
- Both producers and consumers require `cpr-[0-9a-f]{32}` and the structural
  path `AANCA-control-plane-release-v2/verifications/<publication_id>/`
  `release_qualification_attestation.json`. Both hashes are independently
  typed and enter the inner supervisor-release root and outer 48-field
  contract hash. No directory scan, latest pointer or alias selection was
  introduced.
- Focused live gates passed `269 passed`; the dependent terminal suite passed
  `36 passed`. Ruff, format, configured mypy and compilation passed. Targeted
  coherent-reseal cases for ID, path and both hash fields passed `8 passed`.
- The independent unchanged-byte audit reported P0/P1 = **0/0** and `15
  passed`. The exact four-file inventory is 1,241,380 bytes with canonical
  root `D9963D39D13FF9E66195EAE2DCD78FC033372E52C60170424099A02542973DA0`.
  Exact hashes are bootstrap `D8EBE87D...AB4C`, authority
  `2389C132...CCC`, bootstrap tests `404CF4CC...3FDC` and authority tests
  `BE9890FC...865`.
- The audit recorded one P2 limitation: optional strict mypy over the two test
  modules reports 36 pre-existing dynamic-fixture typing errors. Strict mypy
  over both production files is green, and the configured mandatory `mypy
  src` gate remains green. This does not weaken the exact48 runtime result but
  remains disclosed test typing debt.
- No external attestation was opened by these transport validators: Q and the
  supervisor own the independently sealed readback. No Q/E publication,
  supervisor arm, Codex wake or scientific process occurred.

## 2026-07-31 - Current next exact action after live exact48 qualification

- Complete the moving Q and supervisor mirrors against exact48 and FINAL2,
  then require stable inventories, full non-wake QA and independent audits.
  Update the external E/factory to the same four fields and crosslinks.
- The fresh-180 builder remains a separate outcome-blind scientific-material
  gate and does not consume `supervisor_release`; it requires no exact48 schema
  change. It must still pass final source/readback parity before real E
  construction.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`. Production release,
  the one Q write, E, controlled current-session resume and original
  confirmatory remain unexecuted.

## 2026-07-31 - Hold supervisor admission on retained FINAL2 custody

- A read-only boundary trace confirmed that the moving supervisor correctly
  validates the pinned manifest and the full exact57 -> exact19/exact13/
  exact18/exact17 chain with no-follow, two-pass, fail-closed reads. It also
  exposed a P1 lifetime gap: `_ExternalControlPlaneRootInventoryLease` and
  leaf handles were closed when `_validate_spec` returned, before later Q/E
  custody, prearm and child creation.
- Re-reading exact57 in the bootstrap without retaining handles would merely
  create a second parser and another TOCTOU window. The supervisor successor
  must instead retain one external-release qualification lease covering the
  pinned attestation, manifest, exact four-leaf capsule and every critical
  runtime/source/copy/receipt leaf. It must revalidate immediately before
  process creation and resume, retain custody through terminal completion, and
  fail closed on any identity, byte, DACL, ADS, link or handle change.
- Supervisor admission remains HOLD until race/replace/ACL/ADS/path-alias
  synthetic tests, full non-wake QA and an independent unchanged-byte audit
  qualify this lifetime repair. No process was armed or launched during the
  discovery.

## 2026-07-31 - Hold Q freeze on complete FINAL2 producer semantics

- The independent Q production-parity audit first found and repaired two
  conditions that would reject a valid FINAL2 transaction: the verifier's
  actual release-entry allowlist policy and its JSON-null namespace relative
  paths. Earlier owner work also corrected publication-ID v2 and the exact
  `incoming/<release_root>.<publication_id>` staging path.
- The same audit then found three P1 under-validation gaps. Q parsed release
  authorization and attempt only through generic roots rather than the
  producer's closed exact40/exact24 schemas; child completion was not fully
  cross-compared with command/process representations; and the complete
  production argv was not causally bound through authorization and attempt.
- Q freeze remains HOLD until exact40 -> exact24 -> exact19 -> source receipts
  -> exact57 is recursively closed and coherent-reseal tests reject every
  mismatch. P2 follow-up also requires complete log-leaf seal checks and
  removal or explicit isolation of the unused publication-identity-v1
  constant. The provisional `230 passed` result is not a final qualification.

## 2026-07-31 - Qualify external E/factory exact48

- The staging authority and atomic factory now carry the complete exact48
  supervisor release through exact8 input, SPEC51, AUTH51 and READY46 while
  preserving the separate rich-Q and compact-factory hash domains. The
  publication-specific path is deterministically derived; file and semantic
  attestation roots cannot be swapped, scanned or selected through aliases.
- Full owner QA passed `93 passed`; focused FINAL2 passed `8 passed`; Ruff,
  format, strict mypy for four production sources, compilation and both JSON
  schemas passed. The first full invocation used an unnecessarily long
  synthetic basetemp and hit Windows path length in one fixture; unchanged
  bytes passed all 93 tests under a short fresh TEMP root.
- Independent unchanged-byte audit reported P0/P1/P2 = **0/0/0**. The exact
  ten-file inventory is 437,597 bytes with canonical root
  `E0E214D99CBE8BE21EF0357EC9ADA2CE83C6D355D365896CBB049C4A6AE70E56`.
  Key hashes are authority `C97A15C5...B080`, factory
  `FD2F0392...C771`, authority tests `C5FD3C5E...747B` and factory tests
  `C9C79598...91A6`.
- This qualifies source bytes only. It did not create Q, E, a staging attempt,
  launch authorization, supervisor specification, publication, resume or
  science. Fresh-180 remains a separately qualified input candidate and is not
  silently pinned by this result.

## 2026-07-31 - Reformat and independently qualify fresh-180 material gate

- A fresh read-only check found that the previously passing fresh-180 logic
  still passed `51 passed`, strict mypy and compilation, but its two Python
  files no longer passed the current Ruff formatter. Before any factory pin,
  those two files were mechanically formatted and the full semantic suite was
  rerun unchanged.
- Post-format owner QA passed `51 passed`; Ruff check/format, strict mypy,
  compilation and JSON-schema parsing passed. Independent unchanged-byte audit
  reported P0/P1/P2 = **0/0/0** and again `51 passed`.
- The exact four-file inventory is 179,171 bytes with canonical root
  `F46D9FCA1D91A43E0F28E77A65EA5C0BED4A64D67A2706AEB6B084D17AAF8C25`.
  Exact hashes are builder `735E9159...1AD3`, schema `317204F8...15E9`,
  README `41FCDA90...E951` and tests `09C2C4B9...8D33`.
- The independent audit reconfirmed fresh-only/null predecessor lineage, exact
  180 directives, frozen science hashes, no outcome/result discovery, no
  publication and no retry. This component contains no `supervisor_release`,
  exact48, SPEC51/AUTH51/READY46 or external-control-plane consumer; it only
  supplies outcome-blind scientific material to the separately qualified
  factory.
- No real material build, Q, E, publication, resume or scientific process was
  executed.

## 2026-07-31 - Qualify FINAL2-aware Q checkpoint after producer-parity repairs

- Q now validates publication-ID v2, the producer's exact incoming staging
  path, release-entry allowlist policy and nullable namespace paths. It closes
  exact40 authorization and exact24 attempt, then crosslinks their complete
  production argv, commands and process identities through both exact19 child
  completions, source/copy receipts and exact57. Log leaves require exact
  paths, read-only single-link/no-ADS/non-reparse/protected-DACL seals and
  double observation. The obsolete v1 identity constant is isolated under a
  rejected-legacy name.
- Full owner QA passed `239 passed`; Ruff check/format, project and strict
  mypy, compilation and a real disabled-before-read nonexistent-request probe
  passed. Fresh independent unchanged-byte audit reported P0/P1/P2 =
  **0/0/0** and reconfirmed every gate.
- The exact four-file inventory is 647,981 bytes with canonical root
  `084455B8DB84CC6A8264E4AC8FD5318AE1F53B5193797DBB378B7348E3E341D9`.
  Exact hashes are README `B88E289E...5C2C`, schema
  `CD004EA2...E8DE`, controller `FD36E4DA...5FBF` and tests
  `59BD9ED8...A304`.
- This freezes source behavior, not a Q publication. Production manifest and
  exact57 pins remain intentionally absent/parameterized until the external
  release is published and read back; adding its self-containing release root
  to compiled source would create a hash cycle. No Q file was written.

## 2026-07-31 - Reject pre-final supervisor snapshot on post-wake P0

- The independent retained-custody supervisor audit rejected moving source
  hash `FBE624E4...` despite its local green suite. `_wake_then_revalidate`
  invoked the one-time Codex resume before entering its guarded `try`, so an
  exception after a real resume, such as failure to publish/read back
  `wake_result.json`, could skip the immediate post-wake FINAL2 lease
  revalidation. The outer cleanup would detect drift only after Codex had
  already been awakened.
- This is P0 because wake-once and continuous release custody must remain
  fail-closed on every post-resume exception. The successor must enclose the
  entire wake/result path in `try/finally`, always revalidate FINAL2 after any
  attempted real resume, persist STOP on ambiguous result publication and
  never issue another wake or retry.
- A synthetic test must cover: resume succeeds once, wake-result persistence
  raises, FINAL2 revalidation still executes, STOP is durable and wake count
  remains exactly one. Supervisor remains HOLD pending new hashes, full QA and
  a fresh independent unchanged-byte audit. No real resume occurred.

## 2026-07-31 - Treat Q root 084455b8 as a disabled checkpoint, not production Q

- Read-only release assembly proved that the qualified Q snapshot remains
  intentionally non-executable: all compiled `FINAL_*` source/inventory pins
  are `None`, `_production_release_pins_ready()` is false, and
  `_build_live_authorities_from_request` ends with an unconditional fail-closed
  exception after deleting its provisional lifecycle/Q functions.
- The snapshot also lacks real implementations of the required high-level
  `PublishVerifyAndHandoffE` and `CanonicalizeDownstreamEvidence` closures. The
  independently qualified exact10 E/factory supplies closed staging authority
  and low-level suspended-process/native-handle primitives, but no production
  adapter matching those Q call signatures has yet been admitted.
- Therefore root `084455B8...341D9` remains valuable qualified disabled-source
  evidence but cannot be placed into a production release or used to consume
  the one-Q permission. A final-Q successor must use concrete non-cyclic source
  pins, default-deny imports of the exact release modules, a real high-level
  E/supervisor handoff adapter and native-handle downstream canonicalizer, then
  pass full QA and a new independent audit.
- FINAL2 and Q agree on exactly eight runtime component roles and a 13-entry
  release tree, not nine. `capsule_bootstrap.py` belongs to the separately
  sealed execution capsule and must not be invented as a ninth control-plane
  role without a new contract successor.

## 2026-07-31 - Keep supervisor HOLD on pre-attempt versus post-attempt wake state

- The first post-wake P0 repair introduced a P1 overcorrection: its catch path
  classified an exception before creation of any wake intent/resume attempt as
  permanent post-wake consumption. A valid existing-terminal first wake then
  remained at wake count zero; fresh focused evidence was `1 failed, 21
  passed` plus 20 passing subtests.
- The successor must distinguish phases using durable wake-attempt authority,
  not the presence of a catch. Before any attempt evidence and before calling
  resume, a fully revalidated legal first wake remains possible. Once durable
  attempt evidence exists or a real resume may have occurred, ambiguity is
  consumed STOP with no recovery wake or retry.
- Pre-wake integrity failure must STOP without waking Codex; post-attempt
  failure must STOP without a second wake. Supervisor admission remains HOLD
  pending boundary tests, full green QA and another fresh audit.

## 2026-07-31 - Hold current-session resume while this task is active

- Read-only handoff preflight confirmed exact target session
  `019f703b-661d-7c50-b423-9270657d8d6d`, pinned Codex CLI 0.145.0 at the
  external supervisor path with SHA-256 `83751F15...EB6C`, and exact syntax
  `codex exec resume <SESSION_ID> <PROMPT>`. It also confirmed that neither the
  local help nor official CLI documentation guarantees safe concurrent resume
  of this currently active session.
- The supervisor WIP still pins old session `019faaf3-...` in JSON, source and
  tests. Its authentic receipt proves only that old session and must remain
  immutable/superseded. The old `creation_test` is not bound to the current
  Desktop session's first `session_meta` record and cannot be relabelled.
- The current synthetic success helper exits immediately; its sleep option
  does not delay success. Before any exact-session test, build and qualify one
  event-driven delayed-success/event-gate path plus an honest
  `session_origin/session_presence_preflight` schema. The receipt must bind
  session ID and first metadata record, pinned CLI/path/hash, argv/prompt/cwd/
  environment, PID plus creation time, logs, exit code, no children and
  `attempt_count=1`.
- No `codex exec resume` was executed. A test may occur only at a proven idle
  boundary for this session; ambiguity is STOP and no retry.

## 2026-07-31 - Release assembly dry-run identifies missing formal authorities

- FINAL2 contract, schema, publisher, verifier, Q and supervisor all agree on
  exactly eight runtime roles and five qualification records: Q, E authority,
  E factory, fresh-180 and the supervisor bundle. The corresponding release
  tree has 13 entries. Existing path/SHA/size inventory roots are diagnostic
  provenance and are not automatically valid `qualification_root_sha256`
  authority records.
- Production assembly still needs closed, reviewed builders/receipts for those
  five qualification records; a sealed `python_runtime_identity` external
  dependency bound to exact interpreter path/hash/native identity; and a
  qualified exact-20 release-projection/publish-input dry-run builder. These
  are now being implemented outside the repository with production disabled.
- The other two external dependencies are the execution-source manifest file
  hash and execution-source records canonical root. Release root and manifest
  must remain sealed request-time inputs to Q because compiling either into Q,
  whose own hash belongs to the release projection, creates a cycle.
- The future publication ID uses policy
  `aanca_external_control_plane_publication_identity_v2`, exact control root,
  final release root and one caller-supplied one-use nonce. No nonce,
  authorization, release directory or production manifest was created during
  this dry run.

## 2026-07-31 - Current next exact action after Q qualification

- Complete the independent retained-custody supervisor audit. If it returns
  zero P0/P1, freeze that inventory and cross-audit its exact48/SPEC51/AUTH51/
  READY46 sets against Q and E/factory.
- Then assemble and independently validate the exact immutable external
  release input from the qualified Q, E/factory, fresh-180 and supervisor
  bytes. Only after all non-wake gates agree may the one controlled short
  current-session handoff test occur.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; there is still no
  release, Q, E, resume or original-confirmatory execution.

## 2026-07-31 - Final-Q integration exposed a staged-E/final-job path contradiction

- Direct source inspection confirmed that
  `publish_original_confirmatory_e_intent_once` derives the physical E path as
  `<supervisor_job_dir>/e_intent.json` and opens that final job ancestry, while
  `canonical_original_confirmatory_q_e_custody_ready` and its builder again
  infer the supervisor job directory from `E.parent`. The relevant live source
  locations were lines 13347-13435 and 12080-12237 of
  `src/histo_audit/workflows/original_confirmatory_capsule_authority.py` at the
  time of inspection.
- FINAL2 instead requires physical E staging below
  `<state>/control_staging/<job_id>/e_intent.json`, with the final supervisor job
  directory absent until the suspended child has been validated and resumed.
  The current functions therefore cannot satisfy both contracts and are held
  fail-closed; no adapter is allowed to relabel `E.parent` or pre-create the
  final job directory.
- A separate synthetic-only successor is being built to bind the staged E
  leaf/staging ancestors independently from the future final job and receipt
  paths. Checkpoint Q root `084455B8...341D9` remains unchanged and disabled.
  No Q, E, publication, Codex resume or scientific process was executed.

## 2026-07-31 - Release-tools root da9b0ef9 is disqualified by a pre-open-handle counterexample

- The current eight release-tool files reproducibly form canonical inventory
  root `DA9B0EF9353760A1E8DC1D555B34B6A46D7AEF1F81640878A63F3ED17C4A8CC5`
  with 679,648 bytes. Two earlier per-file size/hash values reported for the
  waiter and its test were stale reporting, although the aggregate root was
  reproducible from the current bytes.
- An independent synthetic fault test then retained a pre-opened
  `GENERIC_WRITE` handle to `publisher_success_receipt.json`. After protected
  DACL installation and after the waiter's final scan, a one-byte `WriteFile`
  and flush succeeded; the waiter still exited 0 and the committed receipt's
  first byte became `0x58`. This is a qualifying P0 commit-window failure.
- The causal path opens staging leaves with read/write/delete sharing, applies
  read-only/DACL protections only later, and performs no final revalidation
  after `rename_buffer_prepared`. A previously granted write handle is not
  revoked by the later ACL change. Therefore `da9b0ef9...` is permanently
  ineligible for publication despite its prior ordinary tests.
- The required repair is isolated in a new successor: create each leaf with
  read-only sharing while retaining the producer's own write handle through
  sealing, plus a deterministic regression proving a pre-open writer cannot
  mutate a success artifact after the final scan. The disqualified directory
  remains unchanged as evidence. No real release or authority was created.

## 2026-07-31 - Supervisor retained-custody candidate passes owner QA after test stabilization

- The production supervisor source stayed byte-identical while the handle-soak
  test was corrected to call `gc.collect()` before both its baseline and final
  measurements. Previous failures showed a lower final handle count, not a
  leak; standalone repetitions were green.
- The new nine-file candidate passed `178 passed` plus 203 subtests in 251.64 s,
  including the two wake-boundary regressions, and its Ruff, format, strict
  mypy and compile gates passed. Because the test file changed, prior inventory
  root `84D6C125...` is superseded and cannot qualify the new bytes.
- Its owner-computed canonical nine-file inventory root is
  `00231CBFF020FD9F656E1EB5C8C557396B453427206E849685E7C6B33B132448`
  over 1,844,943 bytes. The production supervisor source SHA-256 is
  `18A59B3830E109A910FB7210EB6EC8A31C306D0961E16AC09929BE70E45A33FB`.
- A fresh exact inventory and independent unchanged-byte audit remain required.
  The foundation also still carries the superseded session evidence, so it is
  HOLD regardless of owner QA. No production supervisor or resume was armed.
- The fresh unchanged-byte audit subsequently reproduced all 1,844,943 bytes
  and assigned the historical qualification-policy root
  `1FB073478EE24C3244503FE22AB4F7B898F3623ED0303E66C9E14D05E3073D5F`.
  The `00231CBF...` value above is a reproducible owner diagnostic alias using
  different record field names/order, not a byte mismatch.
- Independent results were P0/P1/P2 = 0/0/0, `178 passed` plus 203 subtests in
  257.99 s, focused wake/custody 22 plus 20 subtests, restart/PID/singleton/
  session/no-retry 15 plus 53, three repeated handle-soak passes, and green
  Ruff, format, strict mypy and compile. A retained-handle probe returned
  exactly to baseline after explicit close, so the GC stabilization does not
  mask a native-handle leak.
- This qualifies the foundation code/inventory only. Its old session pin keeps
  it in operational HOLD; the task-complete current-session successor is a
  separate implementation and qualification.

## 2026-07-31 - Exact task-complete event provides a non-polling current-session handoff gate

- The authoritative current-session JSONL is
  `C:/Users/NATAN/.codex/sessions/2026/07/17/rollout-2026-07-17T15-19-33-019f703b-661d-7c50-b423-9270657d8d6d.jsonl`.
  Its first raw `session_meta` record has 41,079 UTF-8 bytes, SHA-256
  `5BDD7B37A5466F264719C7968F054F4966CC911D493CA4AF16DF5971C5426E7D`
  excluding its line delimiter, and binds the exact current session ID,
  `thread_source=user`, `originator=Codex Desktop`, timestamp
  `2026-07-17T13:19:33.810Z`, CLI version `0.145.0-alpha.18` and project cwd.
- An earlier diagnostic value `93089C96...` is invalid: Windows PowerShell 5
  decoded the BOM-less UTF-8 line through its legacy default encoding and then
  re-encoded the resulting text. A shared raw `FileStream` read, strict UTF-8
  decode and independent scan of session files established the authentic
  41,079-byte `5BDD7B37...` identity above. Only raw bytes may enter the receipt.
- Historical readback proved that a root assistant `phase=final_answer` record
  is followed by an `event_msg` with `payload.type=task_complete`, the same
  `turn_id`, the complete last assistant message and terminal timing fields.
  The current active turn ID observed in its own tool records is
  `7edc482c-285f-44e6-adc7-7c9c14eb0556`.
- The current-session supervisor successor will record the arm-time byte offset
  and a one-use marker, wait locally with `ReadDirectoryChangesW`, and require
  a new exact final-answer record followed by exact task-complete for that turn
  and marker. Only the task-complete record establishes the idle boundary for
  one `codex exec resume` attempt. A timeout, malformed append, replacement,
  truncation or ambiguous identity is STOP, never a retry.
- This design uses neither a timer nor model/agent polling. No actual resume,
  supervisor arm or synthetic wake was executed during this discovery.

## 2026-07-31 - Technical and fresh-180 execution-source roots require a closed composite identity

- Final-Q preflight showed that the technical import manifest intentionally
  covers only `src/**`, while the fresh-180 source capture covers `src/**`,
  `configs/**`, `pyproject.toml` and `uv.lock` with the two frozen config paths
  excluded. Passing either root under the other's field deterministically
  fails closed.
- FINAL2 permits exactly three external dependency roles, so no fourth role is
  being added. The production-disabled release-input builder is implementing
  one closed composite `execution_source_identity_v1`: its manifest dependency
  contains explicit technical and fresh-180 subrecords, its records dependency
  binds the canonical root of that ordered pair, and the third dependency
  remains the exact Python runtime identity.
- Each subrecord retains its own path, manifest file hash, records root, scope,
  exclusions and artifact count. Q will crosslink the existing technical pins
  and a distinct nested fresh-180 binding to the composite. exact20, exact4,
  exact3 and the 13-entry release shape remain unchanged. No manifest was
  published and no final source root was frozen.

## 2026-07-31 - Current-session handoff authority is closed without enabling a wake

- The current-session JSONL was read back at 200,466,075 bytes while the first
  draft allowed only 256 MiB. The bounded cap was therefore corrected before
  freeze to 1 GiB and an over-limit regression was added; this is a bounded
  safety limit, not a permanent power or log-setting change.
- Base and attempt authorities now use closed canonical JSON envelopes. They
  bind the exact session origin and native identity, pinned Codex CLI, exact
  `exec resume <session-id> <prompt>` argv, prompt text and hashes, the
  pre-arm offset/prefix, an event-driven `ReadDirectoryChangesW` registration,
  same-turn `final_answer` then terminal `task_complete`, and one-use/no-retry
  semantics. `--last`, a shell, polling and automatic retry are forbidden.
- Synthetic and operational capabilities are non-interchangeable profiles.
  The frozen synthetic successor has capabilities `false/false/true`, seven
  content files plus `FROZEN_INVENTORY.json`, inventory root
  `8B8014FF7509293E845D03FEDFEA7ED24E4615D9A04D13F22E094841D18F710B`,
  42 passing tests and green Ruff, format, strict mypy and compile checks. Its
  concrete synthetic base payload root begins `73E9E553...`; it cannot arm or
  resume Codex. A distinct operational-source successor is under construction
  and must be independently audited before any operational envelope is made.
- No supervisor was armed and no `codex exec resume` was executed.

## 2026-07-31 - Release-tools v3 remains HOLD pending a full-lifecycle regression

- An unchanged-byte audit reproduced eight plain files, 682,886 bytes and
  canonical root
  `0DA52569FC430F7105D6606FA2F54B8049226582FDA2B31686BB79D1662C05E1`.
  Its QA passed with 104 tests plus Ruff, format, strict mypy and compile.
- Source inspection found no remaining ordinary write/delete-share window:
  all four staged leaves are created while the producer retains the only
  write handle and exposes read sharing only. However, the new regression
  exercised a private helper rather than the complete waiter and did not
  reach the post-scan `rename_buffer_prepared` boundary or assert the final
  receipt and waiter exit code. Independent severity is P0/P1/P2 = 0/1/0.
- v3 is therefore not admitted. A separate v4 successor is adding a
  deterministic complete-lifecycle reproduction; v3 remains unchanged.

## 2026-07-31 - Mandatory static gates are green while full live gates run

- `.venv\\Scripts\\python.exe -m ruff check .` returned `All checks passed`.
- `.venv\\Scripts\\python.exe -m ruff format --check .` returned `204 files
  already formatted`.
- `.venv\\Scripts\\python.exe -m mypy src` returned `Success: no issues found
  in 99 source files`.
- Full `pytest -q` and the full semantic PanNuke validator are currently
  running as read-only validation work. They are not recorded as passing until
  their terminal outputs are read back.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%**, original confirmatory has not run and M9 remains locked.

## 2026-07-31 - Full live repository and PanNuke gates pass

- The first full `.venv\\Scripts\\python.exe -m pytest -q` run completed with
  `2352 passed, 1 skipped, 1 failed in 1602.53s`. The sole failure was
  `test_internal_timestamps_bracket_verification_before_review_publication`:
  it fixed review timestamps at 2026-07-31 11:00 UTC while the permanent
  attempt claim is created from the live clock, so the test eventually placed
  review start before its own claim.
- The test now derives its two deterministic clock values from the persisted
  attempt-claim timestamp. It still enforces permanent-attempt verification,
  then review start, live verification, completion and publication in order.
  The changed file passed 11 focused tests and Ruff check/format; a separate
  read-only review found no issue and confirmed that no production check was
  weakened.
- A complete rerun of `.venv\\Scripts\\python.exe -m pytest -q` then passed:
  **2353 passed, 1 skipped in 1538.72s (25:38)**. The single skip is the
  documented POSIX-only rename behavior test; Windows denies the relevant
  delete sharing.
- After the test and documentation change, full static gates were repeated:
  `ruff check .` passed, `ruff format --check .` reported 204 formatted files,
  and `mypy src` reported no issues in 99 source files.
- `.venv\\Scripts\\python.exe -m histo_audit data validate-pannuke
  --project-root . --root data\\raw\\pannuke` returned exit 0 with
  `status=valid` and `validation_scope=full_semantic_scan`: 3 folds, 7,901
  patches and 22 raw files. It reconfirmed 4,318 cross-class-overlap pixels,
  10,486,091 void pixels, 1,411 overlap-touching excluded instances, identical
  primary/confirmatory exclusion, no class arbitration and
  `source_masks_modified=false`. Publication was idempotent.

## 2026-07-31 - Release-input builder and release-tools v4 are independently admitted

- The repaired production-disabled release-input builder is exact4 with
  159,861 bytes and canonical inventory root
  `6DD777F533E8ED3952AF11E60D513123FF2C034FFA446B544E32D7DEA7323552`.
  Independent P0/P1/P2 was 0/0/0; 59 tests, Ruff, format, strict mypy,
  isolated compile and closed-schema/custody checks passed. It accepts neither
  noncanonical inner manifests nor nonexistent crosslinks and never imports or
  executes caller-selected FINAL2 modules. Production remains hard-disabled.
- Release-tools v4 is exact8 with 687,373 bytes and canonical inventory root
  `BDB49545BAA6796C9FE9ADBF40AB56A0CF7CF87CC00EA1540F8F18D911FD1226`.
  Independent P0/P1/P2 was 0/0/0; its full waiter regression reaches
  `rename_buffer_prepared`, proves that an additional writer receives a share
  violation, and binds immutable committed receipt bytes plus waiter exit
  disposition. The full 104-test suite, Ruff with the project config, format,
  strict mypy and compile passed. v3 root `0DA52569...E1` remains historical
  HOLD evidence and was not changed.
- Admission qualifies these disabled component bytes only. It did not publish
  a release and did not create or consume Q or E.

## 2026-07-31 - Attempt-creation authority supersedes the impossible prelaunch attempt

- A cross-component audit proved a temporal cycle in the earlier draft: E is
  immutable before the replacement supervisor runs, while a concrete attempt
  contains watcher registration, live session offset and prefix evidence that
  can exist only after that process runs. E therefore cannot truthfully embed
  that future concrete attempt or its final rendered terminal prompt.
- The versioned operational successor now uses this fixed order:
  Q base12; job and staging projection; one closed exact16
  `codex_handoff_attempt_creation_authority`; final Q exact19 with the creation
  payload hash outside its base preimage; E exact23 with the full creation
  envelope and exact Q base/creation crosslinks; custody COMMITTED; then the
  replacement supervisor arms the event watcher and CREATE_NEW-publishes the
  concrete attempt in
  `<supervisor_state_root>\\jobs\\<job_id>\\codex_handoff_attempt_authority.json`.
  This removes both the temporal cycle and a Q-base hash cycle.
- SPEC52 has `codex=null` and one exact10 `external_codex_handoff` mapping to
  the sealed staged E, creation root, concrete-attempt output and terminal
  handoff receipt. Launch authorization is exact52: the three legacy Codex
  fields are removed, four explicitly named external-handoff roots are added.
  The four legacy release prompt fields are likewise replaced rather than
  relabelled. The runtime replacement occupies the existing exact8
  `option_a_supervisor` role; there is no ninth role or second wake owner.
- The first operational spec candidate is rejected/HOLD: independent review
  found a stale self-hash, forbidden redundant E aliases, an incorrect one-hour
  lifetime spanning a potentially 30-day process, and unclosed terminal and
  wake-intent matrices. The corrected semantics bound only same-turn boundary
  capture to one hour, permit up to the declared 30-day supervised-process
  window after durable capture, and separately bound postterminal wake-intent
  creation and the one Codex process. A fresh atomic spec seal and independent
  audit remain required.
- No operational envelope, Q, E, supervisor arm, Codex resume, publication or
  scientific process was created. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%** and M9 remains locked.
- Next exact action: finish the single-role operational `aanca_supervisor.py`
  successor and corrected closed authority spec, run their full synthetic QA,
  freeze an exact inventory and obtain an unchanged-byte independent 0/0 audit.

## 2026-07-31 - Diagnostic Python runtime identity is reproducible

- The admitted release-input builder was loaded from its exact external source
  under `.venv\\Scripts\\python.exe -I -B` and used read-only to build the
  current runtime identity. It bound CPython 3.12.3 AMD64 at
  `C:/Users/NATAN/Documents/AANCA/.venv/Scripts/python.exe`, 274,712 bytes,
  file SHA-256 `864530D708039551A2C672DDD65E5900FBC08B0981479679723A5B468F8082BC`,
  volume serial `4942234099983559526`, file ID
  `43CE1E00000004000000000000000000`, one link and no reparse point.
- The canonical runtime/dependency root is
  `E9F43C7893A858536FD367A3392DEA3276744D238E209463FF087AD359C4313D`.
  This is diagnostic preflight evidence only: no receipt was published, and the
  same identity must be rebuilt from retained bytes immediately before the
  final release/Q gate.

## 2026-07-31 - Synthetic qualification-receipt builder is independently admitted

- The unchanged external qualification-receipt builder is exact4 with 95,565
  bytes and canonical inventory root
  `17451D819FC03C626D1B738279DAEB8942A7E081CF5B4CBF18650AA920A2743E`.
  Independent severity was P0/P1/P2 = 0/0/0 and the verdict was `ADMIT` only
  for these synthetic-only, production-disabled, non-issuing bytes.
- Its 39 tests, configured Ruff check/format, strict mypy, isolated compile,
  functional inventory CLI, native no-link/no-reparse/no-ADS custody checks,
  closed Draft 2020-12 schema inspection and release-input-builder root
  crosschecks passed. The optional `jsonschema`/`fastjsonschema` package was
  unavailable, so no separate meta-schema library run was performed; the
  built-in schema tests and independent closed-field inspection passed.
- No qualification receipt was issued or persisted, no production switch was
  enabled, and no Q, E, release, supervisor, Codex resume or scientific process
  was created. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%** and M9 remains locked.
- Next exact action: finish and freeze the single operational supervisor plus
  its corrected authority spec and close the Q-to-E environment/staging
  adapter before assembling any immutable release or issuing receipts.

## 2026-07-31 - Resume CLI requires a distinct executable compatibility gate

- The current session's immutable first record identifies its origin CLI as
  `0.145.0-alpha.18`, while the first `codex` on the interactive PowerShell
  path is an npm script wrapper for `codex-cli 0.130.0`. The app-bundled
  WindowsApps executable was readable but returned access denied when invoked
  directly from this process, so it is not a valid unattended program pin.
- The npm package's native `codex.exe` ran directly and reported
  `codex-cli 0.130.0`; its diagnostic size is 235,079,472 bytes and diagnostic
  SHA-256 is
  `280CB1C4E3375D94DBDCBA1A191F4F6ADBF73C293BE1E4F16C74B006662B9C54`.
  Its read-only help confirms the exact
  `exec resume <SESSION_ID> <PROMPT>` shape and that `--last` is optional and
  therefore explicitly forbid-able.
- These are diagnostic observations, not the final operational base pin. The
  runnable executable identity and version must be rebuilt immediately before
  base creation, and one short synthetic current-session resume must prove
  cross-version compatibility before any scientific process is authorized.
  No resume was executed during this check.

## 2026-08-04 - Integration continues; current disk headroom is below the provisional confirmatory margin

- A read-only process check found no AANCA Python/Pythonw process, operational
  supervisor, `codex exec resume`, Q, E or scientific launch. Work therefore
  resumes at the control-plane integration gate rather than process recovery.
- Drive C: currently has 42,526,355,456 free bytes (39.61 GiB). A native
  read-only inventory reports 142,448,179,264 logical bytes below the project
  root: `artifacts` 91.34 GiB, `data` 36.68 GiB and `.venv` 4.50 GiB.
  `artifacts/runs` is dominated by the immutable failed primary and qualifying
  recovery, each about 43.11 GiB; raw PanNuke is about 36.66 GiB.
- No raw file, historical run, seal, authority evidence or cache was deleted,
  compressed, moved or rewritten. Integration and synthetic QA may continue,
  but the final original-confirmatory capacity calculation plus the mandatory
  10-GiB reserve must pass again before launch. Current free space must not be
  represented as a passed capacity gate.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%** and M9 remains locked.

## 2026-08-04 - The exact original-confirmatory capacity gate is 70 GiB

- The final read-only dependency audit corrected the provisional storage
  interpretation above: the frozen original-confirmatory preflight and T0
  verifier require exactly 75,161,927,680 free bytes (70 GiB), not merely the
  estimated active tree plus 10 GiB. The verifier checks current live free
  space fail-closed after validating the capacity receipt.
- A fresh observation found 48,332,210,176 free bytes (45.013 GiB), a shortfall
  of 26,829,717,504 bytes (24.987 GiB). This blocks capsule/T0 admission and a
  later scientific launch, but it does not block external source integration,
  synthetic QA or unchanged-byte audits.
- No storage was reclaimed. Before T0, at least 70 GiB must be observed on the
  exact capsule volume; freeing at least 30 GiB outside immutable AANCA raw/run
  authorities is the current external coordination item.

## 2026-08-04 - The event-driven supervisor and terminal launcher reach qualified HOLD

- The external event-driven supervisor successor is byte-stable and resealed at
  `C:\Users\NATAN\AppData\Local\AANCA-supervisor-v3-external-handoff-working-20260731`.
  Its exact-eight inventory contains 2,050,293 bytes with root
  `0ec92d749ebcfc4010c11d6b3ad2c94aeb3373aadf588f467c6cef68ce9c82a8`;
  the operational-authority payload self-hash is
  `324e71f963485bcf2f17d82ea6271e47b7574e6de81c597e66c095c3758b59f5`.
  Owner QA passed 197 tests plus 167 subtests, Ruff, format, strict mypy,
  compilation, PowerShell parsing and strict JSON/self-hash checks.
- A separate integration-owner unchanged-byte audit reproduced the exact root before
  and after and independently passed the same 197 tests plus 167 subtests in 247.39 s,
  Ruff, format, strict mypy, in-memory compilation and PowerShell parsing. Review of
  the exercised contracts found P0/P1/P2 = 0/0/0. The tests cover event-driven
  process-handle waiting, the nonpersistent `SetThreadExecutionState` request,
  singleton and PID/create-time checks, exact science and separately closed Codex
  environments, exact-17 attempt derivation/readback, terminal disposition matrices,
  restart/PID-reuse `LOST` with exactly one diagnosis wake, active/uncertain
  `AMBIGUOUS` with zero wake, and no automatic scientific retry.
- The production-disabled terminal-client launcher successor is exact four files,
  517,664 bytes, root
  `b42fe549796a07d36fe3f32307703e3b281e1dd66ad6e355959f6a6fd432e2b8`.
  Its owner gate passed 49 tests plus 77 subtests and all static/type/compile checks;
  a fresh unchanged-byte audit returned P0/P1/P2 = 0/0/0 with identical before/after
  hashes. Disabled functional invocations returned exit 125 and created no output.
- The launcher has no legacy placeholder tokens and no Q/E/supervisor/Q-file pins.
  It remains `HOLD` with `PRODUCTION_ARM_ENABLED=False` until the one future
  `original_confirmatory_launcher_upstream_materialization_authority_v1.json` binds
  the final capsule, T0, lifecycle, STATIC-v3, runner, terminal and release bundle.
  The supervisor likewise remains unarmed until the immutable release and one short
  exact-session compatibility test are legal. No `codex exec resume`, Q, E, release,
  supervisor arm or scientific process was executed.

## 2026-08-04 - Reject the first production-qualification issuer candidate

- The first production-disabled one-shot issuer candidate was exact five files,
  103,081 bytes, root
  `97916518222fb461d85a6c7e03209b96281dd08745a923305e0ba42ace3bfb3b`.
  Ordinary QA passed 30 tests, Ruff, format, strict mypy, compilation, schema parsing
  and an inert functional dry-run; a second invocation stopped on collision. It did
  not issue a real receipt or create Q, E, a release or a scientific process.
- Independent unchanged-byte audit returned P0/P1/P2 = **0/3/0** and `REJECT/HOLD`.
  A deterministic barrier changed an output leaf after its individual readback and
  closed handle; the issuer still returned exit 0 even though the actual leaf hash
  differed from its receipt. `_canonical_report` also accepted bool/int/float type
  aliases and validated a normalized rather than exact raw unsigned root. Finally,
  three nested JSON-schema objects were not recursively closed.
- Root `979165...bfb3b` is permanently ineligible. Its bytes remain unchanged as
  diagnostic evidence. A new successor must retain all nine output handles through
  terminal-receipt publication and final set revalidation, enforce exact recursive
  JSON types and raw roots, close every nested schema, add adversarial regressions,
  repeat full QA and obtain another independent unchanged-byte audit. Production
  issuance remains compile-time disabled.

## 2026-08-04 - Read-only storage audit identifies external capacity options

- A read-only audit observed about 48.26 GB free (44.95 GiB), leaving about 25.05 GiB
  below the exact 70-GiB gate. It made no filesystem change. The apparent temporary
  growth during parallel pytest was a transient logical footprint; science must not
  run concurrently with these broad QA suites.
- The largest nonprotected candidates are
  `C:\Users\NATAN\Documents\hostinger migration` at 67.05 GiB, including a 50.61-GiB
  Git/LFS directory, and `C:\Users\NATAN\Downloads` at 40.53 GiB. Six physically
  distinct copies/representations of one Hostinger archive contain about 19.09 GiB
  of potential redundancy, but deletion requires selection of a canonical copy and
  remote/LFS restore verification. Downloads include several large, potentially
  redownloadable installers plus private files whose disposition only the user can
  choose.
- AANCA-like TEMP/test roots and logical UV cache sizes are not treated as safe
  reclaimable capacity: they can contain active synthetic evidence or hardlinks into
  the protected environment. No raw PanNuke file, immutable run, authority, cache,
  external WIP, download, unrelated repository or system power/pagefile setting was
  changed. The capacity gate remains failed.

## 2026-08-04 - Q20 and the staged-E authority pass their single focused parity gate

- The production-disabled Q20 controller remains exact four qualified source files,
  833,784 bytes, root
  `65f444510a345ce63927d53f0fb62291b49a6206d58db892d0a1e4e33416e02c`.
  Its earlier complete gate passed 282 tests plus Ruff, format, project/strict mypy
  and compilation. All production pins remain `None` and Q remains `HOLD`.
- The byte-stable staged-E/Q20/SPEC52 authority source is 708,000 bytes with SHA-256
  `8c09d2abd97ea7a8c250c1d038a6f9c08de6779b6744943aada307add703f3da`.
  Its pure public builder
  `build_original_confirmatory_external_supervisor_spec_payload_v3` accepts sealed
  Q20/E23 plus custody/external-handoff fields and performs no write, authorization,
  ambient discovery, process launch or outcome read. Owner QA passed 127 tests plus
  Ruff, format, strict mypy and compilation.
- Exactly one focused cross-component parity invocation was executed. Its terminal
  output was recovered from the durable session JSONL after the tool-output cell was
  lost; it was not rerun. Result: **3 passed in 0.61 s**, exit 0. It covered the
  golden exact-16 environment, Q20/E23/SPEC52 crosslinks and rejection of Q19/hybrid
  input. Q and authority hashes before/after were identical. No production Q or E
  was written and no process was launched.

## 2026-08-04 - Independent E and release-preassembly audits reject the provisional chain

- A fresh unchanged-byte audit reproduced the provisional E-adapter inventory at
  11 declared files / 556,196 bytes / root
  `50e0d98bb85c764e9da91f689de3c3fdb17c8c806e796b5f08800b51182b7c30`
  and the linked authority at 708,000 bytes / SHA-256
  `8c09d2abd97ea7a8c250c1d038a6f9c08de6779b6744943aada307add703f3da`.
  Its full short-path gate passed 118 tests, Ruff, format, strict mypy, compilation,
  both schemas and the Q20/E23 -> raw51 -> exact8 -> AUTH52 -> SPEC52 parity.
- The same audit nevertheless returned `REJECT/HOLD`, P0/P1/P2 = **1/0/1**. The
  authority's purportedly pure v3 builder transitively read ambient `USERPROFILE`;
  identical sealed Q20/E23 inputs succeeded or failed solely when that variable was
  changed. The successor must derive the user-profile/root relationship from the
  sealed runtime ancestor lease, prove zero ambient reads, recalculate the authority
  hash and repin/retest the adapter. Twenty-one pre-existing cache files outside the
  declared exact11 are a P2 hygiene item and may not enter a release.
- Read-only release preassembly then permanently rejected the repaired-but-still-
  provisional issuer root
  `5b3ddf3fba7b06dba525aab10a996d29a089ba9fc748f96e0f5288f7eb431e27`
  despite its 43 passing tests. Three independent P0 integration contradictions
  remain: qualification output and final empty release-control root were the same
  path; exact-three external dependencies were accepted as ungrounded digest strings
  without physical A/B/composite/runtime receipt custody; and the component-report
  semantics admitted a disabled/unpinned Q rather than requiring a production-capable
  fail-closed final-Q source. No output from either issuer candidate is eligible.
- Release-tools v4 themselves remain admitted and production-enabled behind exact
  one-use authority, but no production waiter was invoked. A legal issuer successor
  must separate qualification input/output, final release control and supervisor
  state roots; physically retain and revalidate the two execution-source scopes,
  composite, runtime receipt and interpreter; and reject Q until every compiled
  upstream `FINAL_*` pin is fixed. Its terminal receipt must bind all of that evidence.
- Final-Q source cannot be frozen from the current Q20 template. It is downstream of
  final capsule/T0, lifecycle/STATIC/runner/terminal evidence, the composite execution
  source, materialized terminal launcher, and the one short exact-session resume gate.
  The control-plane release root/manifest/exact57 remain request-time inputs, avoiding
  a hash cycle. Capacity therefore blocks final T0/final-Q/release, although successor
  source design and the resume-readiness audit may continue.

## 2026-08-04 - Current-session compatibility resume is not yet armed

- A read-only readiness audit confirmed the stable supervisor/source identities,
  current session `019f703b-661d-7c50-b423-9270657d8d6d`, 205,056,849-byte session
  JSONL, 41,079-byte first-record SHA-256 `5BDD7B37...E7D`, and runnable native Codex
  executable SHA-256 `280CB1C4...B9C54` reporting `codex-cli 0.130.0`.
- The one real compatibility resume is **not ready**. No material operational base,
  operational-source inventory, independent audit receipt, qualified base builder or
  compatibility-only launcher exists. The existing synthetic authority has the
  noninterchangeable `false/false/true` capability profile, while the future exact
  operational profile is `true/true/false`. The supervisor's science wake is bound to
  Q/E/SPEC52 and cannot be repurposed for this pre-release compatibility test.
- A new production-disabled external bundle is being built to close only this gap.
  It will bind the exact session/first record/native identity, direct executable and
  closed resume environment, stable supervisor/spec/source/audit pins, one-use
  task-complete watcher, logs/result/readback and permanent no-retry STOP semantics.
  It remains incapable of science/publication and cannot arm or resume while its
  compile-time compatibility switch is false.
- No marker was emitted, watcher armed, operational authority written or
  `codex exec resume` invoked. The future compatibility attempt remains exactly one
  and occurs only after the new bundle passes owner QA and a fresh independent audit.

## 2026-08-04 - Ambient-independent staged-E successors are independently admitted

- The replacement staged-E authority is frozen outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-original-confirmatory-staged-e-authority-q20-spec52-successor-v2-wip-20260804`.
  Its exact-four allowlist contains 908,356 bytes with canonical inventory root
  `53b166f4c437802b8879ecab39281ffb9dd0b7b7d4c4a0252c06975dd3760a39`;
  the authority source is 716,282 bytes with SHA-256
  `b0b68f745e6e9e6ae7e8e83278fb2863959d7601dd24b50c51741d20b10be1ff`.
- The linked Q-to-E production handoff adapter is frozen at
  `C:\Users\NATAN\AppData\Local\AANCA-q-e-production-handoff-adapter-spec52-successor-v2-wip-20260804`.
  Its exact-eleven allowlist contains 556,196 bytes with canonical inventory root
  `422bec399fa93252e61a5ebc16e45723386264100e3cf21fc3cc9299034fd0ed`.
  Neither allowlist contains an undeclared directory or cache file.
- Owner gates passed 127 authority tests and 118 adapter tests plus configured Ruff,
  format, strict mypy, isolated compilation, closed-schema inspection and complete
  Q20/E23/raw51/exact8/AUTH52/SPEC52 cross-parity. A separate unchanged-byte audit
  reproduced both roots before and after, reran all 245 tests and static gates, and
  returned P0/P1/P2 = **0/0/0**, `ADMIT`.
- The independent audit proved that the complete reachable builder closure performs
  zero ambient `USERPROFILE`, `HOME` or `PATH` reads. Canonical output was identical
  with those variables absent and adversarial. The permitted profile relationship is
  derived from the sealed runtime-ancestor anchor, and a coherently rehashed but
  mismatching anchor is rejected fail-closed. Staged E remains distinct from the
  future final job, preserves retained custody through ACK/COMMITTED, and has no retry.
- Admission covers only these unchanged external source bytes. No Q, E, supervisor,
  release, resume or scientific process was created. Final production pins remain
  absent and the chain remains `HOLD` until capacity/T0 and the downstream immutable
  release gates pass. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%** and M9 remains locked.

## 2026-08-04 - Reject issuer v3.1 after unchanged-byte integration audit

- The production-disabled issuer v3.1 candidate remains frozen outside the repository
  at `C:\Users\NATAN\AppData\Local\AANCA-control-plane-production-qualification-issuer-v3-1-successor-wip-20260804`.
  Its exact-five allowlist contains 188,983 bytes with canonical 737-byte inventory
  preimage and root
  `5d05513bfe3ddeaa3939caf8ca311281bb11d046ecb03d41a421b84ffbf591c2`.
  The independent audit reproduced the same count, bytes and root before and after and
  found zero undeclared entries or directories.
- Owner and independent gates both passed 74 tests, configured Ruff check/format,
  strict mypy, isolated/in-memory compilation and Draft-2020-12 schema inspection.
  Release-tools-v4 was independently reconstructed as exact eight files, 687,373
  bytes, zero directories, root
  `bdb49545baa6796c9fe9adbf40ab56a0cf7cf87cc00ea1540f8f18d911fd1226`.
  The 9 non-receipt leaves plus terminal receipt correctly form exact10; the
  post-commit readback does not claim an impossible self-referential root or add an
  eleventh file, and mutation of any of the ten leaves fails closed.
- Passing ordinary gates did not override the independent `REJECT/HOLD` verdict,
  P0/P1/P2 = **2/3/1**. The input, qualification-output and supervisor-state roots are
  nested instead of pairwise ancestry-disjoint; existing directory handles are not
  compared by native volume/file identity. Tool/input containment is checked in only
  one direction. The retained snapshot aliases the caller's mutable request, allowing
  a request-TOCTOU disconnect. Q AST inspection misses non-`Name` bindings such as
  import/exception/pattern captures and dynamic `globals()` writes. Finally, the
  release-tools exact8 regression ignores extra directories even though the current
  retained release-tools root itself is clean.
- The live final release-control prerequisite
  `%LOCALAPPDATA%\AANCA-control-plane-release-v2` is absent and was not created. No
  real qualification receipt, release, Q, E, resume or scientific process was
  created. Issuer root `5d055...591c2` is ineligible and remains unchanged evidence.
- Next exact action: create a new production-disabled issuer v3.2 successor that uses
  a deep canonical request snapshot, enforces bidirectional and native-identity root
  separation, closes every Q binding/dynamic-mutation path and requires exact entry
  cardinality including zero directories/cache; then repeat full QA and obtain a new
  independent unchanged-byte audit.

## 2026-08-04 - Reject issuer v3.2 for one direct import-injection path

- The production-disabled v3.2 candidate is preserved unchanged outside the
  repository at
  `C:\Users\NATAN\AppData\Local\AANCA-control-plane-production-qualification-issuer-v3-2-successor-wip-20260804`.
  Its exact-five allowlist contains 214,338 bytes with canonical 739-byte inventory
  preimage and root
  `dc6d95fa4a3050e79e3bae0100023703356fc49f04e7abcd3cd12d0d73f6d9a7`;
  before/after readback was identical and contained zero directories or caches.
- Owner QA passed 129 tests, configured Ruff check/format, strict mypy, compilation,
  schema and inert functional CLI gates. Production issuance remained compile-time
  disabled and no receipt, release, Q, E, resume or scientific process was created.
- A threat-model-calibrated independent audit returned `REJECT/HOLD`, P0/P1/P2 =
  **0/1/0**. After the thirteen literal `FINAL_*` assignments, the validator accepted
  `from attacker_controlled_module import *`; that is a direct syntactic import
  binding capable of replacing global pins and is in scope under PLAN.md lines
  461-463. All other full gates passed.
- Reflective `sys.modules`/namespace mutation by already trusted exact-hashed source is
  recorded only as the explicit out-of-scope limitation in PLAN.md lines 464-467, not
  as a reason for an unbounded parser-hardening loop. PEP-695 local type parameters do
  not substitute the global runtime pins and are likewise informational.
- Next exact action: preserve v3.2 as rejected evidence and build one minimal v3.3
  successor that rejects every `ImportFrom` wildcard, adds the focused adversarial
  regression, repeats the complete owner gate and receives a fresh independent
  unchanged-byte audit. Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8
  remains **8/10 = 80%** and M9 remains locked.

## 2026-08-04 - Admit the production-disabled issuer v3.3 source

- The v3.3 successor is frozen outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-control-plane-production-qualification-issuer-v3-3-successor-wip-20260804`.
  Its exact-five allowlist contains 216,226 bytes with a canonical 739-byte preimage,
  zero directories/cache and root
  `fcd62ea48cded74e62f6d45377df265e419862839487152d2e71e95bf625037e`.
  Independent before/after reconstruction reproduced the same bytes and root.
- Owner and independent gates each passed 131 tests, configured Ruff check/format,
  strict mypy, isolated compilation and Draft-2020-12 schema validation. The schema
  retains nine root variants, 23/23 recursively closed objects and 83/83 resolved
  references. The linked release-tools source remains exact eight files, 687,373
  bytes, zero directories and root
  `bdb49545baa6796c9fe9adbf40ab56a0cf7cf87cc00ea1540f8f18d911fd1226`.
- The canonical Q fixture passes with 13/13 fixed pins and a wildcard `ImportFrom`
  after those literals now fails closed. A separate threat-model-calibrated audit
  found no other in-scope bypass and returned P0/P1/P2 = **0/0/0**, `ADMIT`.
  Deliberate reflective self-modification by already trusted sealed source remains
  only the explicit out-of-scope limitation in PLAN.md lines 464-467.
- `PRODUCTION_ISSUANCE_COMPILED` remains literal `False`; admission covers only these
  unchanged source bytes. No production qualification, final release, Q, E, resume
  or scientific process was created. Operational state remains `HOLD`, formal status
  remains exactly `PRIMARY_STUDY_COMPLETE`, M8 remains **8/10 = 80%**, and M9 remains
  locked.

## 2026-08-04 - Freeze the compatibility-resume implementation in disabled HOLD

- The compatibility-only bundle is frozen outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-current-session-resume-compatibility-wip-20260804`.
  Its six material leaves have root
  `5b2e29ea0f0c7f1876a17f41cc6e09067452dbe6a88b2855497b6ed2c82bc523`.
  Detached canonical `BUNDLE_INVENTORY.json` is 1,695 bytes with SHA-256
  `773b8bbbe482335bc371fc939c44e47d780a919ab74c8831789b6f5f5fdfafe3`
  and payload SHA-256
  `7036a594dad9b36a4f6cffbc9bc556f757ba850fe2e08c8fb9219f783de208aa`.
  The full exact-seven root is
  `6ce26cbe8e35e53b4e00a8a99f1567b20e61c354ecf59484437369bfe7992ada`,
  with exactly seven ordinary files and zero directories/cache.
- Final owner gates passed 66 tests, Ruff check/format, strict mypy, isolated
  compilation, supervisor exact-eight verification, and disabled functional `run`
  and `recover` invocations. Both functional invocations returned exit 125 before
  state creation. A fresh independent A/B unchanged-byte audit reproduced all roots
  and returned P0/P1/P2 = **0/0/0**, `ADMIT` only for the disabled profile.
- The retained-input TOCTOU found during pre-freeze review is closed: parsed bytes now
  come from the retained no-follow handle with bounded length/hash and a post-read
  live verification; deterministic tests forbid a pathname reopen. Direct wildcard
  import of the three enablement flags also fails closed.
- The literal capability profile remains `False/True/False` for compatibility resume,
  synthetic testing and event-driven arm ACK. Classification is therefore exactly
  `QUALIFY_DISABLED_IMPLEMENTATION / HOLD_ENABLEMENT`; it is not a qualified real
  resume implementation. A separate successor must implement and independently
  verify the watcher-armed acknowledgement before any enablement root is created.
- No marker, arm, resume, Codex wake, Q, E, release or scientific process was run.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**
  and M9 remains locked.

## 2026-08-04 - Reclaim only the exact64 synthetic-QA allowlist and pass 70 GiB narrowly

- Two independent read-only inventories qualified exactly 64 stale synthetic test and
  mypy-cache roots: 11 top-level test roots, seven `%TEMP%` test roots and 46 old
  cache roots. The ordered UTF-8/LF list SHA-256 is
  `beacb20e10fbc9b6ba31c629eae1619961d895879c13676fc534f56442b7a26e`.
  Native no-follow scanning found 28,149,097,836 logical bytes, 28,693,072,480
  unique allocated bytes, zero external hardlink identities, zero enumeration/native
  inconsistencies and zero active Python/pytest or direct process references.
- The cleanup ran once as PowerShell PID 25812, created at 2026-08-04 18:07:41 local.
  A too-early process filter initially missed its very long `EncodedCommand`; native
  process readback then recovered the live PID at 11/64 and no second cleanup was
  launched. A prepared file-backed successor remains dormant with no lock or result
  and was never executed.
- The terminal result is `SUCCESS`, 64/64 deleted. Its SHA-256 is
  `71308c6da9abe0790d3440dcba338a72e21196063ae1a939822aff3545414063`.
  The progress log contains 64 unique paths whose reconstructed ordered path hash is
  exactly the allowlist hash; no allowlisted root remains. Redirected stderr contains
  only PowerShell `Preparing modules for first use` progress CLIXML, not an error.
- Explicit exclusions remained present and untouched: the repository, `.venv`, raw
  PanNuke, `artifacts/runs`, `C:\pt3`, the whole volatile `pytest-of-NATAN`, every
  external authority/WIP and every path outside exact64. Frozen SPEC/PLAN/
  PRE_REGISTRATION hashes and `git diff --check` remain unchanged/clean.
- Immediate readback observed 75,889,000,448 free bytes versus the exact
  75,161,927,680-byte gate: a pass by only 727,072,768 bytes (0.677139 GiB).
  This is sufficient for the gate at that instant but is not durable headroom for a
  broad QA suite. Capacity must be reobserved after further tests and immediately
  before T0/Q/E/science; no scientific authority was consumed.

## 2026-08-04 - T0 readiness audit requires a coherent Q20/E23/SPEC52 carrier first

- A read-only dependency audit found that live
  `src/histo_audit/workflows/original_confirmatory_capsule_authority.py` is 598,757
  bytes with SHA-256
  `2389c132211ef668ee8f57b7b649039e4069700be121e681515fcd2174015ccc`,
  while the admitted ambient-independent successor is 716,282 bytes with SHA-256
  `b0b68f745e6e9e6ae7e8e83278fb2863959d7601dd24b50c51741d20b10be1ff`.
  Live `capsule_bootstrap.py` and capsule terminal still bind the earlier Q/E field
  sets, `codex_terminal_wake_*` and `SUPERVISOR_V2_POLICY`; the admitted authority
  requires Q20/E23/SPEC52, `external_codex_handoff` and `SUPERVISOR_V3_POLICY`.
- Therefore copying only the admitted authority/test bytes into the repository would
  create a contradictory live carrier and is forbidden. The next build is a separate,
  production-disabled external successor for bootstrap plus terminal and focused
  tests. Only after independent admission may its closed allowlist, the admitted
  authority, and tests be integrated together by one owner.
- After coherent integration the focused live gate covers capsule authority, Codex
  handoff authority, terminal and capsule builder tests; it is followed by full
  pytest, Ruff check/format, mypy/compile, expected direct-confirmatory gate failure
  and the full real PanNuke validator. No capsule/T0 is built before that gate.
- `artifacts/execution_capsules`, current T0 request/authority directories and
  `%LOCALAPPDATA%\AANCA-control-plane-release-v2` are absent. Historical lifecycle
  evidence binds source root `1179f917...` and cannot qualify the new T0. Existing
  admitted supervisor/launcher/Q20/E/issuer/release-tools bytes remain reusable only
  in their stated disabled/source roles.
- A separate read-only audit of the surviving `pytest-of-NATAN` root returned an
  empty cleanup allowlist: `pytest-3846` had already been removed by pytest's
  three-session retention, the three surviving numbered roots are current Aug-4
  sessions, and all older synthetic state leaves total only about 8 MiB allocated.
  No deletion was performed. Latest free-space observations remain only about
  0.62 GiB above the exact gate.

## 2026-08-04 - Hold the first arm-ACK successor after independent race review

- The production-disabled arm-ACK successor reached owner QA with **78 passed** plus
  green Ruff check/format, strict mypy, isolated compilation, supervisor exact-eight
  verification and disabled `run`/`recover` exits 125/125. Its transient exact-six
  Python-ordinal root was
  `ab36a0f2f64fe638c82d8705c8897456096b2e81f577aa5e01f67e02f70913af`;
  no bundle inventory was created and the bytes were not frozen.
- A fresh independent unchanged-byte run also passed all 78 tests but returned
  `HOLD`, P0/P1/P2 = **0/2/1 candidate**. A valid pre-existing STOP, and a STOP
  injected after launcher PREPARED readback but before final commit, both allowed a
  final `watcher_armed.json` and a successful return. The in-memory latch did not
  serialize other durable STOP writers. Separately, a valid immediately completed
  boundary was rejected 20/20 times solely because its worker had already exited,
  leaving a correct boundary/no error plus ACK and an erroneous STOP.
- The bounded-cancellation claim also remains unqualified: a synthetic blocking
  `cancel_boundary_wait()` held the caller before the bounded thread join. The stable
  native implementation is expected to cancel quickly, but the generic contract does
  not yet enforce that bound.
- The required successor repair is finite: a shared CREATE_NEW winner token must
  serialize `ABORTED_PRE_ACK` versus `ACK_CLAIMED` before any final receipt rename;
  ACK success then additionally requires fail-if-exists promotion and retained exact
  final readback before signalling the worker. An incomplete claim is STOP-only and
  never retryable. A dead worker is admissible only when a synchronized terminal
  boundary exists and no worker error exists. Cancellation must return within its
  documented bound even if the backend callback blocks.
- No real arm, resume, Codex wake, Q, E, release or scientific process was run.
  Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%**
  and M9 remains locked. Next exact action: complete the three narrow regressions,
  repeat owner QA, then repeat an independent unchanged-byte audit before inventory.

## 2026-08-04 - Admit and freeze the coherent Q20/E23/SPEC52 carrier

- The production-disabled carrier is frozen outside the repository at
  `C:\Users\NATAN\AppData\Local\AANCA-capsule-q20-e23-spec52-carrier-successor-wip-20260804`.
  Its governed exact-18 material root is
  `e9a740aa99ad4818fbc40169316f84f79a7c06ff769dd9a2ac5349160154b462`
  for 18 files and 1,956,988 bytes. The closed roles are exactly nine mapped
  leaves, seven reference-only leaves and two governance files.
- Owner QA passed 339 tests plus configured Ruff check/format, strict mypy and
  isolated compilation. In the independent run 338/339 tests passed under the
  original long basetemp; the sole failure was Windows `MAX_PATH` at a
  262-character generated replacement path. The exact failing node then passed
  1/1 on a verified-empty short basetemp, so every collected node passed on the
  same unchanged governed bytes. This is recorded as a bounded environment-path
  replay, not a scientific or integrity waiver.
- Independent pre-freeze and post-freeze audits both returned P0/P1/P2 =
  **0/0/0**, `ADMIT`. The detached canonical inventory has payload SHA-256
  `589ac740eb913c5d96caa1135631dfc59658dba1db7799a3ba83ecc6f20f2e9b`,
  file SHA-256
  `7ab72a6f2eefbcd60e5960427e808d89bb798c8727779f0e35f91b666ded3471`
  and material inventory root
  `189402ef432e5f3fb9b5cb6e1d6fda5807d505f001d4d901e6d9de73d2a130f2`.
  The full exact-19 seal root is
  `0a76a4c048bda498f8a26847c44fb4a2660e2b3d96d2f66203d0766ec2067be0`
  for 19 files and 1,961,028 bytes.
- All 19 files are read-only ordinary single-link files with zero reparse
  points, named ADS, cache or extra paths. The mapping self-root remains
  `a3b4b13224b75b4eb6066eb7bc6d6e5f3545c39e65361aaf8afb4c72a7630dd8`;
  all nine live repository preimages were re-read after freeze and remain exact.
- No repository integration, Q, E, resume, wake, publication or scientific
  process was performed. The next exact action is one fail-on-drift integration
  of only the nine mapped leaves, followed by focused and full live QA, CLI and
  the real PanNuke validator. Formal status remains exactly
  `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%** and M9 remains locked.

## 2026-08-04 - Reclaim only exact38 old C:\pt3 pytest roots

- Two independent read-only scans qualified the same 38 physical, non-reparse
  synthetic pytest roots from the recorded 2026-07-21 short-basetemp run. Their
  ordered UTF-8/LF allowlist has SHA-256
  `72cdcca58c073ed9cd721239452ef877ab7f0ab8c552086b055e737ac78d00cf`,
  2,217,387,020 logical bytes and about 2.25 GB unique NTFS allocation.
- Both scans found zero active Python/pytest/Ruff/mypy process, zero process
  command-line reference to `C:\pt3`, zero external hardlink or reparse target
  and zero scan error. The one hardlink pair and one nested symlink were wholly
  internal to their candidate roots. Repo, `.venv`, raw PanNuke, real runs,
  authorities, WIPs, all `*current` aliases and every other `C:\pt3` entry were
  excluded.
- The first long inline cleanup command was rejected by tool policy before it
  started; it made no change. A reviewable one-use PowerShell script then repeated
  the exact count/hash, direct-child, ordinary-directory, age, logical-byte and
  process gates before any deletion. Its sole execution returned `SUCCESS`:
  38/38 deleted, zero remaining.
- Free space increased from 75,395,448,832 to 77,646,655,488 bytes, an observed
  delta of 2,251,206,656 bytes. The one-use script was then removed with
  `apply_patch`; no standing cleanup command remains.
- This was synthetic QA-capacity maintenance only. No project source, raw data,
  historical real run, frozen authority or scientific evidence was removed or
  changed. Capacity must still be re-read immediately before every T0/Q/E/science
  boundary.

## 2026-08-04 - Admit and freeze the event-driven arm-ACK successor

- The first semantically complete HOLD snapshot
  `5918e7c855cf6d241f236de258b426558d94e075bea84326d14b7dcef8929c55`
  passed 138 owner tests but was rejected after two deterministic chronology
  bypasses: a year-2000 ACK receipt could precede the 2026 attempt and a
  year-2000 result start could precede the 2026 intent/process-start receipt.
  It changed only because the owner applied those repairs before the auditor's
  final readback; no unchanged-byte admission was claimed.
- The next HOLD root
  `89a10cacf11151bc3f9d2254f2698f6c557f491e8467c62fa13328b2cc70eec3`
  passed 140 owner tests but was rejected P0/P1/P2 = **0/1/1**. Nested worker
  SHA fields accepted 64-digit JSON integers through `str(...)`, and base
  `link_count` accepted JSON `true` as `1`. Both exact-type defects received
  deterministic regressions; no other analogous coercion site was found.
- The repaired material exact-six root is
  `e483c4f99511e5128b76334350cc77c9a424f9e7f789e03f83bae21095630c0a`:
  six files, 299,325 bytes, compatibility source SHA-256
  `18cbf5c9a7e434914de820bc1a2d1fb0d7837b74eaf249b459577de71ef6e15d`
  and test SHA-256
  `06ad806218423b938b98a68bed518616b182480375e00a8fa80de410cb929030`.
  Owner QA passed 143 tests, Ruff check/format, strict mypy, compilation,
  supervisor exact-eight verification and disabled `run`/`recover` exits 125.
- A fresh unchanged-byte audit reproduced every prior bypass and confirmed it
  now fails closed, returning P0/P1/P2 = **0/0/0**, `ADMIT`. After owner-only
  cache removal, the detached inventory payload SHA-256 is
  `26f73489c1316388b32655b318160fbd9a69773a8dcb05b11e5c4a14315bb1a2`,
  inventory file SHA-256 is
  `43b16d9de6c6fd811b513b68df8ac1cb584642915ba2f87179a19571f321984d`
  and the frozen exact-seven seal root is
  `41e7dd4a70d49d3b4800f6f1f344d3f2f74333020fc76c1c2515afec156ef798`
  for seven files and 301,065 bytes. A final independent post-freeze audit again
  returned 0/0/0 `ADMIT` with exact canonical types, no cache/extras/reparse/ADS
  and ordinary read-only `nlink=1` leaves.
- The admitted profile remains production-disabled:
  `compatibility_resume_enabled=false`, `synthetic_stub_testing_enabled=true`,
  `event_driven_arm_ack_implemented=true`,
  `production_materialization_enabled=false`. No operational base, one-use
  authority, real arm, resume, Codex wake, Q, E or science was created.

## 2026-08-04 - Integrate the exact-nine carrier and pass the complete live gate

- The frozen mapping was re-read immediately before integration. All six file
  preimages and three required absences were exact. A first single `apply_patch`
  attempt used standard numbered unified-diff hunk headers; the local parser
  rejected them during verification and changed zero files. A four-line probe
  established the parser's plain-`@@` syntax and was removed. The second single
  531,074-byte `apply_patch` applied all nine leaves together; immediate readback
  matched all nine carrier postimage sizes and SHA-256 values.
- The first focused live gate collected 382 tests: 380 passed and two carrier
  tests failed because their file reads used the external-root package layout
  instead of repository `src`. Only those two reads were changed from
  `SOURCE_ROOT` to `PACKAGE_IMPORT_ROOT`; targeted replay passed 2/2 and the
  complete focused gate then passed **382/382**. A separate static/origin audit
  confirmed exact `__file__` and `__spec__.origin` and returned 0/0/0 `ADMIT`.
  Production bootstrap, authority and terminal remain at carrier SHA-256 values
  `dc523f...`, `b0b68f...` and `89c8dd...`; the formatted live carrier test is
  SHA-256 `a3240a5bbcdd4148a7e43c53cf88b2a1b5af79d50982e0277d566e7ca858f98e`
  and differs from the carrier test only at those two path tokens.
- The first full no-cache command on short basetemp completed with
  **2382 passed, 1 skipped, 5 failed in 1557.96 s**. All five failures used a
  process-global Windows handle-count equality oracle and observed fewer handles
  after cleanup, not more. The exact five passed 5/5 in isolation. Independent
  forward/reverse and controlled-GC probes proved exact owned-close counts
  4/1/2/4/6, a still-valid foreign sentinel, and reproduced a 772-to-768 global
  decrease solely from unrelated cyclic finalizers.
- The test-only repair pre-collects GC and asserts exact unique owned handle
  closures plus a live sentinel instead of process-global equality. Production
  publication code remained byte-identical at SHA-256
  `6be792901a6a9cd090db4443fe129bdffc47b19414a0d96d6234bb88291c5ec4`.
  The repair passed 9/9, its whole file passed 78/78, the two edited test files
  passed 95/95 after formatting, and an independent test audit returned 0/0/0
  `ADMIT`. The second complete no-cache run passed **2387 tests with one expected
  Windows skip and zero failures in 1535.32 s**.
- Exact mandatory style gates now pass: `ruff check .` reports all checks passed
  and `ruff format --check .` reports 206 files formatted. The independently
  admitted authority remains linted but is explicitly excluded only from Ruff
  formatting because changing its bytes would invalidate SHA-256 `b0b68f...`.
  Project-configured `mypy --no-incremental` passes 100 source files and an
  in-memory compile passes 213 Python files. An additional stricter-than-project
  `mypy --strict src` audit is transparently non-green with 27 existing
  dependency/return-any findings across 12 PyTorch/skimage modules; it is not an
  AGENTS.md mandatory gate and no scientific code was altered to hide it.
- The direct command `.venv\Scripts\python.exe -B -m histo_audit experiment
  confirmatory` returned native exit **2** with exact
  `CONFIRMATORY_CAPSULE_AUTHORITY_REQUIRED` before lifecycle, dataset, cache,
  executor or run-directory access.
- Full real-data validation then passed in 595.6 s with `status=valid`, three
  folds, 7,901 patches and 22 raw files. It reports 4,318 cross-class-overlap
  pixels in 575 patches, 10,486,091 void pixels in 162 patches, 1,411
  overlap-touching instances excluded identically from primary and confirmatory,
  zero positive/background conflicts, no class arbitration and
  `source_masks_modified=false`. QC selection SHA-256 remains
  `09886588591d9ebb9a725db1022bb0ab8fb94b4bcca419b486e2549b0cc5fd36`.
- Synthetic QA cleanup was separately fail-closed. Exact-three allowlist
  `95d1992c...` reclaimed 4,461,654,016 observed bytes after first stopping on
  two still-active diagnostic PIDs; exact-seven allowlist `1207e66e...` later
  reclaimed 4,020,150,272 observed bytes. All one-use cleanup scripts were
  removed. Latest free space is 77,232,390,144 bytes and the exact 70-GiB gate
  passes. Raw ZIP/NPY, nested manifest Parquet and duplicate-audit NPZ probes are
  all still ignored by Git. Frozen SPEC/PLAN/PRE_REGISTRATION hashes remain exact
  and `git diff --check` is clean.
- Formal status remains exactly `PRIMARY_STUDY_COMPLETE`; M8 remains
  **8/10 = 80%** and M9 remains locked. The next exact action is a fresh
  execution-source inventory and two byte-reproducible capsule builds, followed
  by capacity-v2/T0/lifecycle/STATIC/launcher qualification. No Q, E, real
  resume, publication or scientific process has yet been run.

## 2026-08-04 - Pass the fresh non-publication two-build capsule gate

- A process readback found no active AANCA, `histo_audit`, confirmatory or
  training command. The pre-absent external root
  `C:\Users\NATAN\AppData\Local\AANCA-capsule-build-live-20260804-v1`
  was then created without changing any prior capsule, authority, run or raw
  PanNuke file. The previously audited non-publishing runner was used at exact
  SHA-256 `5609ed27bc714d33301c0b3e2bd5c17090c837aa7f9c6096b85118cb7a5a459b`.
- `freeze-inventory` exited 0 and wrote one read-only canonical inventory:
  108 entries, 20,440 bytes, file SHA-256
  `c60ba3601b227cd8e8c34f7b8230c946f7d9cd5ef17e28ae5a9f2bf6cd099ca3`,
  source-records root
  `bddcb13b71573a2366c1d8050b5bfef7a8d6b783098277f93f48056c77a83c35`
  and live `capsule_builder.py` SHA-256
  `dd436f2255eb735ffab2db26cf016149512651ae0fc17a8fa7108e0acffbf85e`.
- Two separate fresh Python workers built `build-a` and `build-b`; both native
  exit codes were 0. Both candidates are read-only, 7,593,243 bytes and
  byte-identical at SHA-256
  `e87efdd814fb5916a76eada23e478bd2b13a074ffc3516f882d72eddcc271e90`.
  Their internal manifest SHA-256 is
  `102ae6f53ccbee83b14d9601e5bcc537cc52af5994eab6c23211e6d5893b6d89`
  and internal records root is the exact frozen source root above. Receipts
  are SHA-256 `8d1ff2310b0eecdc11dc24c36cbb60d7ec0db060bbfb2a2c5dbc75085c33cfa8`
  and `75655be3321922d1dc203990bc3aca1c6ecc1c60fb99999df3e58eaee4b925aa`;
  both explicitly record no publication and no scientific execution.
- A separate read-only verifier, without importing the capsule builder,
  recomputed every live-source hash, the records root, whole-file equality,
  canonical manifest, exact ZIP member order and stored bytes, CRCs, fixed
  metadata, local-header contiguity and terminal EOCD. It returned native exit
  0 and `PASS`. PowerShell independently reproduced both whole-file hashes and
  byte equality. The external root contains exactly the inventory, two
  candidates and two receipts; all five leaves are read-only, ordinary and
  single-link. A second unchanged-byte team audit remains in progress before
  downstream admission.
- This is only a non-publication reproducibility gate. Formal status remains
  exactly `PRIMARY_STUDY_COMPLETE`; M8 remains **8/10 = 80%** and M9 remains
  locked. Q, E, resume, capsule publication, supervisor arming, training and
  scientific execution remain absent. The next exact action, after the second
  audit admits these bytes, is the fresh capacity-v2 and acyclic published
  capsule/T0/lifecycle/STATIC/launcher dependency chain.

## 2026-08-04 - Independently admit A/B and hold before capsule publication

- The second unchanged-byte auditor returned **ADMIT, P0/P1/P2 = 0/0/0** for
  the exact five-leaf external build root. It independently recomputed all 108
  live payload hashes, source root `bddcb13b...`, whole-capsule SHA-256
  `e87efdd8...`, the 109-entry fixed ZIP, internal manifest `102ae6f5...`, both
  canonical receipts and ordinary read-only/single-link/no-reparse/no-ADS
  filesystem properties. The production capsule destination, T0 request and T0
  namespace remain absent.
- A live source-capacity readback independently reproduced the schema-v3
  execution-source tree as 114 artifacts, root
  `11e0b8a4af86d251896dcbd72a733b32e958595c015b0e4b25045966c8b9fe6f`,
  canonical LF manifest size 17,448 bytes and file SHA-256
  `952b98279e51508caec712b20b07b7c066fec4312884afcf005bdecce110f8ee`.
  That capture excludes governance documents, so the truthful STATUS/DECISIONS
  append does not change it. The observed free space was 77,454,708,736 bytes,
  a transient pass above the exact 75,161,927,680-byte gate.
- Read-only code tracing found a real missing operational component. The repo
  exposes hardened `publish_capsule_create_new`, but no production command
  emits its exact publisher-process receipt, no distinct implementation emits
  the independent readback receipt, and no production writer emits the final
  capacity-v2/source/T0 input receipts. Test fixtures are not authority and
  those JSON files must not be assembled manually ad hoc.
- The live T0 verifier requires the strict chronology
  `capsule_published < capsule_verified <= capacity_checked <= intent_created`.
  Therefore the next action is to build, synthetically test, freeze and
  independently audit external one-use publisher, distinct readback verifier
  and post-readback capacity writer tools. Adding them to the repository would
  invalidate the just-admitted source inventory and both capsules, so they stay
  outside execution source. Publication, capacity-v2 receipt and T0 remain
  `HOLD` until that tool gate passes; no automatic retry is permitted.

## 2026-08-18 - Complete the reduced presentation MVP

- The owner explicitly replaced the remaining full-project execution target
  with a presentation MVP. `MVP_SCOPE.md` records the reduced boundary without
  changing `SPEC.md`, `PLAN.md` or frozen `PRE_REGISTRATION.md`. Their final
  SHA-256 values remain respectively `9260d7d...e2fd0`,
  `176f018...2357` and `7cd9e1c...473b`.
- The accepted primary source remains the immutable run
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery`: 185/185 required
  cells completed, zero required failures, 37 optional skips, 36 saved
  comparisons, positive `PRIMARY_STUDY_COMPLETE` stage attestation and
  `amended_or_exploratory` disposition. No new training, recovery,
  confirmatory, Q/E, publication, external-validation or session-resume process
  was started.
- Added `histo_audit demo build` and `histo_audit demo verify`. The builder
  reads selected primary and PanNuke QC artifacts only after checksum/seal and
  stage-attestation verification; preserves all 36 comparisons; requires the
  no-class-arbitration and void/background QC policy; refuses overwrite; and
  emits a closed five-file checksum package. Synthetic tests cover successful
  build/CLI, source immutability, overwrite rejection and tamper rejection.
- The real command
  `.venv\Scripts\python.exe -m histo_audit demo build --project-root .
  --run-dir artifacts\runs\20260727T133947.089370Z_pannuke_primary_orphan_recovery
  --qc-bundle reports\pannuke_qc --output-dir artifacts\mvp_demo` returned 0.
  It created `index.html`, `evidence.json`, `README.md`, the bound QC overlay
  and `manifest.json`. The output manifest root is
  `a7728956bbb89d2d63866fbf5f4427e7854e43a4e629c90308d69ff0670e0b36`;
  a fresh `demo verify` returned `status=valid`, five files,
  `DEMO_COMPLETE` presentation status and `PRIMARY_STUDY_COMPLETE` scientific
  status. `artifacts/mvp_demo*/**` is now explicitly Git-ignored.
- Focused QA passed **10/10**. The first full no-cache suite truthfully stopped
  at **2,389 passed, one expected Windows skip and one failure**: the historical
  independent source-delta fixture did not list the new `mvp_demo.py` or the
  already-live `pyproject.toml` delta. Its exact manual allowlist was updated;
  the failed node then passed 1/1. A clean second full run passed
  **2,390 tests, one expected Windows skip, zero failures in 821.92 s**.
- Final mandatory style gates pass: `ruff check .` reports all checks passed and
  `ruff format --check .` reports 208 files already formatted. The final
  functional `demo verify` also passes after these gates.
- The reduced deliverable is now **100% complete** and its presentation stage is
  `DEMO_COMPLETE`. The highest completed scientific-study stage remains
  `PRIMARY_STUDY_COMPLETE`. Original confirmatory (the unfinished portion of
  M8), all of M9/expert review and external validation are explicitly deferred,
  not claimed. Under the original unchanged PLAN, M8 therefore remains 8/10 and
  M9 remains locked; this does not block the separately defined presentation
  MVP.
- Exact next command for presentation or handoff:
  `.venv\Scripts\python.exe -m histo_audit demo verify --output-dir
  artifacts\mvp_demo`, then open `artifacts\mvp_demo\index.html` in a browser.

## 2026-08-18 - Repair scientific-presentation completeness and handoff readiness

- A full read-only project audit recomputed the accepted run integrity root as
  `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`:
  expected and actual roots match, the registry record is present, and there are
  zero missing, added, changed, or erroneous run artifacts. The frozen authority
  hashes remain unchanged: `SPEC.md=9260d7d...e2fd0`,
  `PLAN.md=176f018...2357`, and `PRE_REGISTRATION.md=7cd9e1c...473b`.
- Presentation schema v2 now fails closed unless it can checksum-read and retain:
  all 36 H1/H3/H5/H6/H7 comparisons; all 33,670 H2 subgroup rows with 32,760
  reportable AP estimates; and the registered H4 restoration artifact. The H4
  result is shown prominently and remains adverse to the registered hypothesis:
  audit-guided macro F1 `0.5244310209334606`, random-review mean
  `0.5265870806000477`, difference `-0.0021560596665870513`, and 95% interval
  `[-0.0028586383107464695, -0.0013925186647962915]`.
- The builder now checksum-verifies the registered instance-dependent ranking and
  OOF files for seeds 404, 405, and 406. All three ranking hashes are exactly
  `69766d68a24679b510b3e63e0eb039207119ff1d96953dafe31cfa51f74d276b`
  and all three OOF hashes are exactly
  `730459794225e338834568d4d7bf574b65c38b634726e787d9121f818d702958`.
  The presentation therefore treats them as one deterministic realisation, not
  three independent replications, while retaining every frozen row.
- Holm-adjusted p-values are explicitly labelled one-sided and the saved 95%
  percentile bootstrap intervals remain visible. H6 unavailability, H7 neutrality,
  the adverse H4 result, and the `amended_or_exploratory` disposition are stated in
  the primary view rather than hidden in an appendix.
- The canonical five-file package was rebuilt and verified at manifest root
  `cfabd4dee8f07c1a06de1f3259dde870e3bec642092e9fac4bd1cfac12e0ee75`.
  Browser QA at 1440x900 and 1024x768 found zero console errors, zero document-wide
  horizontal overflow, working expandable evidence tables, and a valid lazy-loaded
  1512x3840 QC image. Superseded candidate/backup packages and temporary browser
  artifacts were removed only after the canonical v2 package passed verification.
- Documentation now marks the 2026-07-17 validation report as historical, records
  completed M5 real-data validation, narrows the locally evidenced CC BY-NC-SA 4.0
  scope to `masks/`, adds the official AQuA DOI `10.52202/075280-3494`, and records
  the H4 and seed limitations in `README.md`, `MVP_SCOPE.md`,
  `ETHICS_AND_LIMITATIONS.md`, and `DECISIONS.md`. The canonical MVP is the only
  generated demo package admitted for Git tracking; licensed raw data, embeddings,
  full runs, and scratch builds remain ignored.
- The first post-repair full suite truthfully stopped at **2,389 passed, one expected
  Windows skip, and one failure** because a non-semantic explanatory docstring in
  `controlled.py` changed a historical source-delta contract. The docstring was
  removed instead of weakening that contract; the failed node then passed 1/1.
  The final full suite on the exact final source passed **2,390 tests, one expected
  Windows skip, zero failures in 877.08 s**.
- Final gates after that suite: `ruff check .` passed; `ruff format --check .`
  reported 208 files already formatted; `mypy src` passed 100 source files;
  `uv lock --check` resolved 83 packages; `uv pip check` found all 64 installed
  packages compatible; and the final real `demo verify` returned `status=valid`.
- Formal status remains `DEMO_COMPLETE` for the presentation and
  `PRIMARY_STUDY_COMPLETE` for the scientific boundary. Confirmatory, expert-review,
  and external-validation claims remain explicitly absent.

## 2026-08-18 - Complete the editorial scrollytelling redesign

- Rebuilt the canonical presentation as an AANCA-specific editorial experience
  combining the supplied Linear-style token system with the centered pacing of the
  owner's reference article. The page now includes a deterministic animated canvas
  hero, fixed progress/navigation chrome, a five-stage scroll-linked method story,
  responsive mobile navigation, reduced-motion and print fallbacks.
- Added code-native, evidence-bound visualisations rather than generated scientific
  imagery: the complete H4 downstream comparison, a forest plot for all 36 frozen
  H1/H3/H5/H6/H7 comparisons, H2 subgroup ranges, a seed-integrity diagram and a
  filterable complete comparison table. The adverse H4 result, unavailable H6 cells,
  neutral H7 result and byte-identical instance-dependent seeds remain prominent.
- Real-browser QA passed at 1440x900 and 390x844. The page has 36 comparison rows,
  two semantic SVG figures and one deterministic canvas; desktop and mobile both
  had document width equal to viewport width. Sticky storytelling advanced the text
  and workflow graphic to the same stage; mobile reduced it to a readable static
  sequence. The H6 filter returned exactly 3/36 rows, expandable evidence and mobile
  navigation worked, `prefers-reduced-motion: reduce` exposed content without motion
  dependence, and the browser console reported zero errors and zero warnings.
- The canonical closed five-file package verifies as `status=valid` with presentation
  status `DEMO_COMPLETE`, scientific status `PRIMARY_STUDY_COMPLETE`, file count 5
  and manifest root
  `2d0d3303548157ad9fada38b11af4b9da9ad784170b15c5b0244e72d4449873e`.
- The final full suite passed **2,390 tests, one expected Windows skip and zero
  failures in 1,032.98 s**. `ruff check .` passed; `ruff format --check .` reported
  208 files already formatted; `mypy src` passed 100 source files; `uv lock --check`
  resolved 83 packages; and `uv pip check` found all 64 installed packages
  compatible. Focused MVP tests also passed 3/3.
- No experiment, training, corruption, annotation or accepted-run artifact was
  changed. Frozen authority hashes remain exactly
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
- Exact presentation command:
  `.venv\Scripts\python.exe -m histo_audit demo verify --output-dir
  artifacts\mvp_demo`, then open `artifacts\mvp_demo\index.html` and reload the
  browser tab.

## 2026-08-19 - Replace the generic hero field with the review-queue workflow

- Replaced the generic Three.js nucleus field with a deterministic explanatory
  sequence tied directly to AANCA: irregular source nucleus contours stay fixed,
  the four-tile AANCA reticle focuses one candidate, five class-signal pulses are
  shown, and only a copied review item follows a visible curve into a numbered
  six-slot expert-review queue. The scene is explicitly labelled
  `Conceptual workflow · not benchmark data`.
- The source animation and generated HTML expose the stable markers
  `threejs-review-queue` and `immutable-source-ranked-review`. Focused MVP tests
  passed **3/3** and the extracted ES-module source passed Node syntax checking.
- Real-browser QA passed at 1440x900 and 390x844. WebGL initialized, the browser
  console reported zero errors and zero warnings, and neither viewport had
  document-wide horizontal overflow. The mobile composition was moved below the
  hero copy after visual inspection. With `prefers-reduced-motion: reduce`, the
  canvas renders the completed queue statically instead of depending on motion.
- The canonical five-file package in `artifacts/mvp_demo` is byte-identical to the
  verified candidate and passes `demo verify` with presentation status
  `DEMO_COMPLETE`, scientific status `PRIMARY_STUDY_COMPLETE`, file count 5 and
  manifest root
  `33b0456c50777760935bbe6587a170667e07babbb4c5755d910b09d1d633cfa7`.
- The full mandatory suite passed **2,390 tests, one expected Windows skip and zero
  failures in 1,110.61 s**. `ruff check .` passed and `ruff format --check .`
  reported 208 files already formatted.
- This was presentation-only work. No experiment, metric, evidence file, source
  annotation, frozen authority or accepted-run artifact was changed. The formal
  stages remain `DEMO_COMPLETE` for the presentation and
  `PRIMARY_STUDY_COMPLETE` for the scientific boundary.
- Exact verification command:
  `.venv\Scripts\python.exe -m histo_audit demo verify --output-dir
  artifacts\mvp_demo`, then open `artifacts\mvp_demo\index.html`.

## 2026-08-19 - Validate the English expert-facing presentation release

- Rebuilt the reader-facing presentation entirely in English. The hero now names
  Natan Smogór as author and records the release date as 18 August 2026. Plain
  explanations introduce the benchmark before the exact design, statistics and
  limitations. Internal completion and disposition codes are absent from visible
  page text but remain unchanged in machine evidence and governance records.
- Applied the supplied Linear-style design constraints: near-black canvas,
  layered neutral surfaces, a single lavender accent, compact Inter/JetBrains Mono
  typography, restrained borders/radii and wider evidence layouts. Replaced the
  overlapping workflow with five separated stages and rewrote comparison labels
  while retaining every raw ID. Tables and forest plots use fixed six-decimal
  formatting, including `-0.000004` rather than wrapped scientific notation.
- Added pinned GSAP 3.15.0, ScrollTrigger 3.15.0 and Three.js 0.185.1 progressive
  enhancements. GSAP drives the reading progress, navigation, reveals, chart marks
  and synchronized method story. Three.js renders the deterministic interactive
  nucleus cloud and scroll-linked transition toward an ordered review queue.
  Static, network-failure, reduced-motion, narrow-screen and print fallbacks retain
  the full scientific narrative.
- Real-browser QA passed at 1440x900 and 390x844. Both widths had zero document-wide
  horizontal overflow; the page contained all 36 comparison rows; GSAP,
  ScrollTrigger and the Three.js renderer loaded; and the console reported zero
  errors and zero warnings. Mobile navigation and the workflow were readable. QA
  found one genuine mobile filter defect caused by a card-layout `display` rule;
  `table tr[hidden]` now remains hidden and the H6 filter visibly returns exactly
  3/36 correct H6 cards.
- Visual evidence is retained under `output/playwright/`, including the full static
  desktop layout, desktop hero/method/results/forest/table details and mobile
  hero/method/table/filter views. The full desktop composition and each detailed
  viewport were visually inspected against `DESIGN-linear.app.md`; no overlap,
  clipped value or unintended horizontal scroll remained.
- The canonical five-file package verifies as `status=valid`, file count 5, with
  manifest root
  `3c1292f04bb1d021b65564a4a7480b3a4d60a1b4c1cf46f2af43df84ed518aa5`.
  Superseded candidate packages, the temporary pre-swap backup and flawed QA
  captures were removed only after canonical verification.
- Focused MVP tests passed **3/3 in 37.22 s**. The final complete suite passed
  **2,390 tests, one expected Windows skip and zero failures in 1,134.57 s**.
  `ruff check .` passed; `ruff format --check .` reported 208 files already
  formatted; `mypy src` passed 100 source files; `uv lock --check` resolved 83
  packages; and `uv pip check` found all 64 installed packages compatible.
- Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No experiment, label, source annotation, accepted-run artifact or scientific
  result changed. Presentation status remains `DEMO_COMPLETE`; the scientific
  boundary remains `PRIMARY_STUDY_COMPLETE`.

## 2026-08-19 - Integrate the explanatory review-queue hero and requalify final bytes

- A concurrent presentation-only update appeared after the first full validation
  had started. It was preserved and inspected rather than overwritten. The update
  replaces the abstract Three.js cloud with a clearer deterministic sequence:
  irregular contours remain in a labelled source patch, one candidate is focused,
  class signals appear, and a copy travels into a six-slot expert-review queue.
  The source contour never moves or disappears, and the canvas explicitly says it
  is a conceptual workflow rather than benchmark data.
- Real-browser revalidation at 1440x900 and 390x844 found zero horizontal overflow,
  zero console errors and zero warnings. The final renderer reports
  `threejs-review-queue` and story identity `immutable-source-ranked-review`; GSAP
  and ScrollTrigger load; all 36 evidence rows remain present; and none of
  `DEMO_COMPLETE`, `PRIMARY_STUDY_COMPLETE` or `amended_or_exploratory` appears in
  visible page text. The H6 filter returns exactly 3/36 H6 rows and no hidden row
  has a computed display other than `none`.
- Reduced-motion QA initially found four non-active method steps still visually
  dimmed. The final CSS now forces all five text steps and every workflow stage to
  full opacity while retaining the static completed Three.js queue. A regression
  assertion binds both fallback rules. Revalidation reports zero hidden reveals,
  zero dim story steps, zero dim workflow stages and zero horizontal overflow.
- Final screenshots are retained at
  `output/playwright/aanca-final-desktop-layout-full.png`,
  `output/playwright/aanca-final-hero-v2.png`,
  `output/playwright/aanca-final-mobile-hero.png` and
  `output/playwright/aanca-final-mobile-filter-h6.png`. The full composition and
  detail views were inspected against `DESIGN-linear.app.md`; the method diagram,
  charts, forest rows and evidence table remain aligned and readable.
- The exact final canonical package verifies as `status=valid`, file count 5, with
  manifest root
  `33b0456c50777760935bbe6587a170667e07babbb4c5755d910b09d1d633cfa7`.
  Focused MVP tests passed **3/3 in 4.01 s**. Because the concurrent update landed
  after the earlier full run had begun, a new full suite was run from a fresh
  collection on the stable final source: **2,390 passed, one expected Windows skip
  and zero failures in 930.14 s**.
- On the same final source, `ruff check .` passed; `ruff format --check .` reported
  208 files already formatted; `mypy src` passed 100 source files;
  `uv lock --check` resolved 83 packages; and `uv pip check` found all 64 installed
  packages compatible. Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No scientific evidence, label, experiment, accepted-run artifact or authority
  changed.

## 2026-08-19 - Publish and validate the professor-facing editorial release

- Rebuilt the English presentation around a centered explanatory narrative. It
  now defines annotation auditing in plain language, records Natan Smogór and the
  18 August 2026 release date, specifies PanNuke/already-segmented nuclei/frozen
  ResNet-18 plus logistic regression/five-fold group-safe OOF/5% review budget,
  and explicitly distinguishes the experimental reference label from biological
  truth.
- Replaced the stacked benchmark explanation with a five-stage cumulative
  serpentine story driven by GSAP ScrollTrigger. Desktop keeps a sticky diagram
  and active text in one frame; mobile and reduced-motion modes show all five
  stages. The header has no progress line, hides while scrolling down, returns on
  slight upward movement and animates the AANCA wordmark from its mark.
- Replaced the former answer-count slogan with `What the study actually learned`,
  kept all H1-H7 outcomes visible without `<details>`, softened H2 to a descriptive
  finding, retained the adverse H4 result and reports 36 preregistered entries as
  33 numeric plus 3 explicitly unavailable. Dense evidence rows stack cleanly on
  mobile and remain filterable.
- Added an always-visible QC preview, complete provenance and citations, a clean
  GitHub repository card, three usage paths and a line-separated footer containing
  authorship, release, scope, responsible-use, evidence, licence and dataset-term
  information. Updated both the root README and package README.
- Real-browser Playwright QA passed at 1440x900 and 390x844. The console reported
  zero errors and zero warnings; document/viewport widths were exactly 1440/1440
  and 390/390. Menu open/close/Escape, the cumulative method stages, H6 filtering
  (3/36 rows), post-filter GSAP refresh and the later `Use` reveal were exercised.
  Visual inspection found no overlap, clipped numeric value or horizontal overflow.
- Selected final screenshots are retained at
  `output/playwright/aanca-professor-release-hero.png`,
  `output/playwright/aanca-professor-release-method-mid.png`,
  `output/playwright/aanca-professor-release-qc.png`,
  `output/playwright/aanca-professor-release-footer.png`,
  `output/playwright/aanca-professor-release-mobile-menu.png` and
  `output/playwright/aanca-professor-release-mobile-use-after-filter.png`.
- The canonical five-file package is byte-identical to the verified candidate and
  passes `demo verify` with `status=valid`, file count 5, presentation status
  `DEMO_COMPLETE`, scientific status `PRIMARY_STUDY_COMPLETE` and manifest root
  `be7366d4034f87119db43958e8c2a0189a3679c5e1abd2863a1044c9d6ffc523`.
- Final validation passed: focused MVP tests **3/3 in 4.79 s**; complete suite
  **2,390 passed, one expected Windows skip and zero failures in 1,089.54 s**;
  `ruff check .` passed; `ruff format --check .` reported 208 files already
  formatted; `mypy src` passed 100 source files; `uv lock --check` resolved 83
  packages; and `uv pip check` found all 64 installed packages compatible.
- Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No experiment, source annotation, result, accepted-run artifact or frozen rule
  changed.

## 2026-08-19 - Optimise delivery, runtime work and first-use verification

- Audited CLI startup, package size, presentation resource loading, browser
  behaviour and the documented first-use path. Warm CLI help and package
  verification were already fast (roughly 30-50 ms), so no speculative import
  rewrite was made. The main usability bottleneck was requiring the full ML
  environment merely to present a static article.
- Added `scripts/present_demo.py`, a Python-standard-library-only launcher that
  verifies the exact closed package, serves it on loopback, opens the browser and
  supports `--no-open`, `--verify-only`, custom binding and automatic free-port
  selection. It passed from isolated mode with `python -I`, demonstrating that it
  does not import the project package or third-party dependencies. The installed
  CLI now also exposes `histo-audit demo serve` with the same verify-before-serve
  behaviour.
- Strengthened both package verifiers to require the manifest record paths to be
  the exact unique payload allowlist. A regression test rewrites a manifest with a
  duplicate path and recomputes its root; verification correctly rejects it before
  trusting the duplicated record.
- Optimised the presentation runtime without changing content or evidence. The
  Three.js loop now pauses outside the hero and while the tab is hidden, resumes on
  return, uses a low-power preference and caps the pixel ratio at 1.5 desktop / 1.25
  compact viewports (or 1 with data saving). The QC image now has explicit intrinsic
  dimensions, lazy asynchronous decoding and low fetch priority; jsDelivr is
  preconnected; table-filter refreshes are coalesced per animation frame.
- Updated the root README, `demo/README.md`, `MVP_SCOPE.md`, the generated package
  README and the article's repository section to make
  `python scripts/present_demo.py` the simplest first-use command while preserving
  the full `uv` workflow for actual software experiments.
- Real-browser Playwright QA passed through the dependency-free server at 1440x900
  and 390x844. Both widths had zero horizontal overflow and zero console errors or
  warnings. The hero state was `running`, changed to `paused` off-screen and resumed
  after returning; reduced motion produced `static`. The H6 filter retained exactly
  3/36 records. Desktop usage cards, the hero, the mobile menu and mobile use section
  were visually inspected with no overlap, clipping or inconsistent spacing.
- The canonical five-file package passes both independent verifiers with
  `status=valid`, file count 5, presentation status `DEMO_COMPLETE`, scientific
  status `PRIMARY_STUDY_COMPLETE` and manifest root
  `1b95f12a167e749f57bb3fded2d82636586c8d018a75423f332242c43d91f49f`.
- Final validation passed: focused tests **13/13 in 6.06 s**; complete suite
  **2,393 passed, one expected Windows skip and zero failures in 1,074.35 s**;
  `ruff check .` passed; `ruff format --check .` reported 209 files already
  formatted; `mypy src` passed 100 source files; `uv lock --check` resolved 83
  packages; and `uv pip check` found all 64 installed packages compatible.
- Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No dataset, source annotation, label, metric, experiment, accepted-run artifact
  or frozen scientific rule changed.

## 2026-08-19 - Complete repository, security, dependency and presentation audit

- Audited the 401 tracked files at Git revision
  `288956737d1693c1461e7538f0e2a0f88836aef8` before modification. The local
  `main`, `origin/main` and remote head were identical; `git fsck --full
  --no-dangling` reported no repository-integrity error. All 76 Markdown files had
  valid in-repository link targets, the article had 32 unique IDs, every internal
  anchor resolved, all local assets resolved after lazy loading, and its UTF-8
  source contained no replacement characters or detected mojibake.
- Completed one formal repository-wide Codex Security Standard scan covering nine
  surfaces: CLI/configuration, archive and numerical-data ingestion, filesystem
  publication, deserialization, subprocesses, the demo server, the browser surface,
  governed capsule control and blinded expert-review packaging. The sealed report
  contains **zero reportable findings** and complete surface coverage. The runtime
  did not permit a delegated independent baseline auditor, which is recorded as a
  variance-reduction limitation rather than hidden.
- Dependency review found one unique PyPI advisory, `PYSEC-2026-3447`, reported
  twice for the same installed `setuptools 81.0.0` distribution. `torch
  2.12.1+cu126` requires `setuptools<82`, so the patched `setuptools>=83` cannot be
  resolved without changing the frozen ML stack. The affected path is Setuptools'
  macOS sdist file-selection logic; AANCA builds with Hatchling and the audited
  presentation verifier uses only the Python standard library. The advisory is
  therefore retained transparently as constrained and unreachable in the verified
  workflow, not silently ignored. `pip-audit` could not map the custom CUDA local
  versions of torch/torchvision to PyPI, which remains an explicit audit limitation.
- Added `.github/workflows/presentation-integrity.yml`. It verifies the exact
  sealed five-file demo on pushes to `main`, pull requests and manual dispatch,
  with read-only repository permission, a five-minute timeout, concurrency
  cancellation, disabled credential persistence and full-SHA pins for
  `actions/checkout` and `actions/setup-python`. Offline `zizmor --pedantic`
  returned **no findings**. The root README now exposes the workflow and explains
  that it complements rather than replaces the full scientific gates.
- Corrected the distributable package author from the generic university-project
  placeholder to `Natan Smogór`; a rebuilt wheel contains the corrected metadata.
  The scientific version and dependency constraints did not change.
- Hardened both verified local presentation servers with deterministic `no-store`,
  CSP frame/object/base restrictions, permissions policy, no-referrer, `nosniff`
  and frame-denial headers, and replaced the Python version banner with `AANCA`.
  Regression coverage checks every header. A real-browser follow-up confirmed that
  pinned GSAP and Three.js still load and the console remains free of errors.
- Playwright QA passed at 1440x900 and 390x844 with zero document-wide horizontal
  overflow. It exercised mobile menu open/close/Escape and focus return, directional
  header hiding/reveal, comparison filtering, the full 36-row evidence table,
  reduced-motion rendering and a blocked-CDN fallback. The no-CDN layout retained
  every reveal and the complete five-stage benchmark; expected network-load errors
  were the only console messages in that deliberately blocked scenario.
- Selected audit captures are retained at
  `output/playwright/aanca-full-layout-audit-20260819.png`,
  `output/playwright/aanca-mobile-layout-audit-20260819.png` and
  `output/playwright/aanca-reduced-motion-audit-20260819.png`. Visual inspection
  found no overlap, clipped value, broken table layout or inconsistent spacing.
- The dependency-free verifier and installed CLI both report `status=valid`, file
  count 5, presentation status `DEMO_COMPLETE`, scientific status
  `PRIMARY_STUDY_COMPLETE` and manifest root
  `1b95f12a167e749f57bb3fded2d82636586c8d018a75423f332242c43d91f49f`.
  The package builds successfully as both sdist and wheel.
- Final validation passed: focused MVP tests **6/6 in 4.82 s**; complete suite
  **2,393 passed, one expected Windows skip and zero failures in 1,089.96 s**;
  `ruff check .` passed; `ruff format --check .` reported 209 files already
  formatted; `mypy src` passed 100 source files; `uv lock --check` resolved 83
  packages; and `uv pip check` found all 64 installed packages compatible.
- Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No dataset, source annotation, label, metric, experiment, accepted-run artifact
  or frozen scientific rule changed.
- Authenticated Git access to `https://github.com/Jaqwilk/AANCA.git` works, but an
  anonymous HTTPS request cannot read the repository and the public web URL returns
  404. Repository visibility or professor access remains an owner-controlled
  external setting and was not changed implicitly by this audit.

## 2026-08-19 - Recoverable workspace and repository cleanup

- Classified the workspace before removal rather than applying a blanket ignored-file
  deletion. The local tree contained approximately 91.373 GiB under `artifacts`,
  36.677 GiB under `data` and 4.501 GiB in `.venv`. Raw PanNuke archives and arrays,
  both 43-GiB primary lineage trees, representation caches, `.venv`, authorities,
  registries, evidence reports and the canonical `artifacts/mvp_demo` package were
  retained. In particular, `git clean -fdX` remains unsafe because ignored paths
  include licensed data and immutable scientific evidence.
- Removed 29 tracked intermediate QA files from the intended repository state:
  superseded English-layout captures, pre-release final captures, review-queue
  animation iterations and the obsolete `mvp_demo_before_review_queue` copy. The
  removed files total 8,735,705 bytes. After the new hygiene policy and attributes
  file are included, the planned repository head contains 377 files and 95,033,593
  bytes instead of 405 files and 103,762,380 bytes: a net reduction of 8,728,787
  bytes without removing any canonical presentation or scientific evidence.
- Retained exactly nine deliberate browser captures: six professor-release views
  and the three latest full/mobile/reduced-motion audit views. `.gitignore` now
  ignores all other `output/playwright` content by default and explicitly allowlists
  only those nine files. `.playwright-cli` is also explicitly classified as local
  cache.
- Moved, rather than permanently deleted, 44 ignored browser-design images, eight
  non-canonical `artifacts/mvp_demo_*` preview directories, one generated figure,
  tool caches, bytecode caches and the 29 tracked intermediate files. The recoverable
  copy is
  `C:\Users\NATAN\AppData\Local\Temp\AANCA-cleanup-backup-20260819T0400` and contains
  572 files / 231,292,255 bytes. Tracked removals also remain recoverable from Git
  history. No transient cache or non-canonical demo preview remains in the workspace.
- Added `.gitattributes` so repository text uses deterministic LF endings across
  platforms while NPY, NPZ, Parquet, PNG and ZIP assets remain binary and
  byte-preserving. Added a README hygiene policy that forbids blanket ignored-file
  cleanup and distinguishes disposable tooling output from retained research state.
- Complete post-cleanup validation passed: **2,393 tests passed, one expected Windows
  skip and zero failures in 1,098.56 s**; Ruff check passed without cache; Ruff format
  check reported 209 files already formatted; mypy passed 100 source files; `uv
  lock --check` resolved 83 packages; and `uv pip check` found all 64 installed
  packages compatible. Both the dependency-free verifier and installed CLI report
  `status=valid`, file count 5 and manifest root
  `1b95f12a167e749f57bb3fded2d82636586c8d018a75423f332242c43d91f49f`.
- All 36 retained Markdown files have valid local link targets. The canonical article
  retains 32 unique IDs, zero duplicate IDs, zero missing static anchors and zero
  missing local assets. `git fsck --full --no-dangling` and `git diff --check` passed.
- Frozen authority hashes remain unchanged:
  `SPEC.md=9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`,
  `PLAN.md=176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`
  and
  `PRE_REGISTRATION.md=7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
  No implementation, dataset, source annotation, label, metric, accepted-run file or
  frozen scientific rule changed.
