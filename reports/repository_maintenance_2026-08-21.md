# Repository maintenance and retention audit — 2026-08-21

## Outcome

The maintained AANCA workspace and public repository were reduced without changing
source annotations, raw datasets, frozen scientific specifications, accepted result
values, or claim boundaries. Cleanup was intentionally conservative: locally removed
material was first moved outside the repository into an exact dated quarantine and
verified. Large superseded run, cache and test classes were then permanently erased
to reclaim disk space; the final deletion boundary is stated below.

No new scientific completion stage is claimed by this engineering maintenance.

## Retention boundary

The following material remains authoritative and was not removed:

- all frozen configurations, preregistrations, amendments, checksum sidecars and
  immutable evidence required to reconstruct scientific decisions;
- raw PanNuke, MoNuSAC, NuCLS and PUMA inputs and their source inventories;
- accepted PanNuke pilot run
  `20260718T143216.354310Z_pannuke_pilot_c7797330e0`;
- accepted recovered primary run
  `20260727T133947.089370Z_pannuke_primary_orphan_recovery`;
- reusable PanNuke, MoNuSAC, PUMA and expanded-development embeddings;
- released NuCLS and MoNuSAC evidence needed by standalone recalculators, plus PUMA
  evidence needed by the project-coupled saved-evidence readback;
- the five-file static demo and its one deliberately retained README hero under
  `docs/assets/`.

The accepted primary and pilot directories were re-hashed after cleanup. Their
integrity roots remained respectively
`8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`
and
`37a9cdc4aab1eb74dc6e86555dfeb96f7682d8bc17bdb0e3a12ec6ab18254666`;
both verifications were valid with no missing, added or changed paths.

## Removed from the maintained repository

The Git cleanup removes only redundant, transient or unconsumed material:

- the tracked `artifacts/qa/` mirror, which duplicated canonical evidence in
  `reports/` and had no maintained consumer;
- nine superseded browser-audit screenshots; the selected release hero was moved
  from generated `output/` into `docs/assets/`;
- orphaned capsule scaffolding (`capsule_builder.py`, `capsule_policy.json` and
  `entry_contract.json`) after its only dispatch consumer had already been retired;
- four machine-local status snapshots (`doctor.json`, `environment_initial.txt`,
  `final_validation.md` and `hardware.json`) whose content is reproducible or stale;
- the unconsumed M7 disk-space preflight snapshot, the superseded preliminary
  external-review note and the disconnected geometric-audit future-work draft;
- the empty notebooks placeholder;
- the redundant presentation-only CI workflow after its package-verification gate
  was moved into the cross-platform scientific workflow.

Historical references to those paths inside immutable run provenance are retained
as historical evidence and were not rewritten.

## Local cleanup and deletion boundary

The first conservative pass moved 4,222 identified files (about 43.78 GiB) to
`C:\Users\NATAN\Documents\AANCA_cleanup_quarantine_20260821`. This inventory
contained 2,840 superseded or interrupted run files (43.47 GiB), generated smoke and
test products, derived caches, the duplicate QA mirror and non-release browser
captures. Later validation products were placed under the same boundary.

Keeping that inventory on `C:` eventually left no writable space. After every target
had been resolved beneath the exact quarantine root, the classified superseded runs,
caches, QA duplication, smoke products and non-release captures were permanently
deleted. The drive reported approximately 43.79 GiB reclaimed. Those deleted items
are not recoverable; all authoritative evidence listed above remains in the active
repository.

The last `mvp_demo_before_author_section/` rollback contained exactly five files and
3,402,366 bytes. After the current package passed both package verifiers and the
desktop, mobile and reduced-motion browser audit, that obsolete copy and the
now-empty quarantine root were permanently removed. No cleanup quarantine remains;
the deleted material is not recoverable.

The active `artifacts/` tree fell from about 92.1 GiB to 48.5 GiB. The remaining
43.3 GiB run footprint is the accepted primary result plus the accepted pilot; the
remaining 5.0 GiB embedding footprint is deliberately reusable. Generated `output/`
is now entirely transient; the 133,735-byte release hero lives under `docs/assets/`.

## Pipeline consolidation

Repeated implementations were replaced by shared maintained primitives:

- one durable NumPy archive writer now serves compressed and intentionally
  uncompressed caches;
- macro-F1 and per-class recall from confusion matrices are defined once in the
  evaluation layer;
- figure publication uses one atomic save helper;
- exact-byte YAML loading and checksum-pinned configuration loading use one config
  authority;
- PUMA, MoNuSAC, autoresearch, smoke, pilot, primary and reporting paths now call
  those shared helpers instead of carrying local copies.

Two specialised NPZ writers remain intentionally independent: frozen cache
publication and create-only confirmatory evidence enforce stronger ownership and
non-overwrite semantics than ordinary derived artifacts. The two standalone
external verifiers also retain independent metric implementations so that they do
not merely call the implementation they are intended to audit.

Across the maintained source-package categories measured before and after this
refactor, Python code decreased from 104,231 to 104,116 lines (net `-115`) even after
adding the shared validation logic. The presentation cleanup then moved the stable
styles out of the generator and removed three complete dead template generations,
unused plotting/rendering helpers and obsolete version suffixes. After adding strict
NuCLS, MoNuSAC and PUMA authority readback, `mvp_demo.py` contains 2,853 nonblank
lines; the complete measured `src/`, `scripts/` and `tests/` Python workspace contains
153,732 nonblank lines, down from 155,697 (net `-1,965`). The stylesheet contains 812
nonblank lines. The goal was lower duplication and fewer failure paths, not artificial
compression of frozen scientific contracts.

## GitHub storage policy

The three 41–78 MB PUMA numeric-evidence archives remain versioned because the
project-coupled saved-evidence readback consumes them. They are stored with Git LFS
instead of ordinary Git objects. Small JSON authorities, reports, protocols and
checksums remain normal Git files and can be reviewed directly.

## Validation

The first targeted run exposed that unquoted YAML dates were parsed as `date` values
by PyYAML but rejected by the new portable config normaliser. Dates and datetimes are
now deterministically normalised to ISO-8601 strings; the two failed cases then
passed. This regression was corrected before the complete suite.

Final gates:

- `uv run ruff check .`: passed;
- `uv run ruff format --check .`: all 215 maintained Python files formatted;
- `uv run pytest`: `1145 passed, 1 skipped` in 837.69 seconds; the skip is the
  documented Windows/POSIX open-file rename difference;
- post-extraction focused suite: `58 passed` in 47.03 seconds;
- wheel build: passed, with `histo_audit/assets/mvp_presentation.css` present in the
  built package;
- deterministic synthetic generation: `generated`, definition
  `791fe34c3bb9042b73badd8209afa1b2b673922e20f2da2da28e9a70d67525b2`;
- synthetic smoke: completed run
  `20260821T111042.663068Z_synthetic_smoke_4d457ebe70`;
- isolated demo verification: `DEMO_COMPLETE`, five files, manifest root
  `1e4e403e08aefc8e9d2e4b18a1b44d24c30c4ab4df106fba45addfd598ca2b4b`;
- project-coupled PUMA evidence readback: all seven internally pre-specified gates
  and all 44 saved convergence checks passed; it did not retrain those models and is
  not third-party validation;
- independent NuCLS feasibility verification: the paired pre/post endpoint remains
  correctly unavailable and the action remains `retain_uncorrected`;
- selected-candidate reconstruction: exact metrics reproduced, `220/220` fits
  converged, no final external test used and no source annotation modified;
- both retained PanNuke run integrity checks: valid;
- workflow YAML parsing and `git diff --check`: passed.

The first post-extraction focused test attempt was interrupted by `ENOSPC` while
pytest was writing its cache. It produced no accepted test result. The classified
quarantine was then removed as described above and the same suite passed all 58
tests.

The independent scripts require the pinned project environment and are therefore
invoked as `uv run python scripts/<name>.py`. Direct system-Python invocations were
rejected with `ModuleNotFoundError` and produced no scientific evidence.

## Final presentation-readiness pass

The final post-cleanup release supersedes the earlier gate counts above:

- full suite: `1147 passed, 1 skipped` in 805.06 seconds;
- Ruff lint and format: passed for 215 maintained Python files;
- mypy: no issues in 103 source files;
- wheel: built, installed in an isolated environment and exposed the complete CLI;
- fresh synthetic run: completed successfully and its temporary products were
  deleted after inspection;
- primary, NuCLS, MoNuSAC, PUMA, selected-candidate and NuCLS-feasibility independent
  verifiers: passed with their registered favourable, adverse and unavailable
  outcomes unchanged;
- current demo: five files, manifest root
  `edbba03401c50eb4a2e0fd2e5a43c744b0726af6c6a6aaed3e5ee3d5c0e29426`;
- browser QA: desktop, iPhone 15 and reduced motion passed without overflow, broken
  images, missing anchors, duplicate IDs, console errors or failed page resources;
- the subtle findings animation was observed, while reduced motion exposed all text
  immediately and rendered the hero canvas statically;
- Markdown link validation and `git diff --check`: passed.
