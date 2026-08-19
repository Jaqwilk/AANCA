# AANCA presentation MVP scope

Date: 2026-08-18

This is a transparent presentation-scope amendment requested by the project
owner. It reduces the deliverable to the smallest reproducible, presentable
research prototype that uses the already completed PanNuke primary study. It
does not amend the scientific analysis, reopen outcome-based choices, or claim
that every milestone in `PLAN.md` has been completed.

## Frozen authority retained

- `SPEC.md`: SHA-256
  `9260d7d00e5a9fe2e9eec0809c3a8b3125aff7cc1d0d35ad1053055bc40e2fd0`.
- `PLAN.md`: SHA-256
  `176f0184f5841a89b8c4746a821d548bb3a1ec8ab59242338b7d65892f552357`.
- `PRE_REGISTRATION.md`: SHA-256
  `7cd9e1cfc38d648ed551ee9835f885c9ab94d45d65b5f8e86064a189d320473b`.
- The accepted primary result remains `amended_or_exploratory` because outcomes
  have already been inspected. No MVP decision changes that disposition.

## MVP deliverable

The MVP consists of:

1. the existing validated PanNuke acquisition and explicit overlap/void QC;
2. the accepted, sealed `PRIMARY_STUDY_COMPLETE` run
   `20260727T133947.089370Z_pannuke_primary_orphan_recovery`;
3. one static, non-diagnostic presentation in `artifacts/mvp_demo`;
4. machine-readable `evidence.json` containing all 36 saved H1/H3/H5/H6/H7
   comparisons, the complete H2 subgroup summary, the registered adverse H4
   downstream result, and the byte-identical instance-dependent seed disclosure,
   without outcome-based selection;
5. a closed five-file output allowlist and checksum verification command;
6. tests proving build, verification, no-overwrite and tamper rejection;
7. a responsive English presentation layer with a deterministic Three.js hero
   that keeps source nucleus contours fixed while copied review evidence enters a
   ranked expert-review queue, GSAP/ScrollTrigger method explanation, data-bound
   charts, a complete filterable evidence table, reduced-motion handling and print
   fallbacks;
8. an explicit byline and professional author profile for Natan Smogór, including
   owner-supplied education details that distinguish current Kozminski University
   study from the completed 80-hour Uniwersytet Młodzieżowy programme, plus the
   release date of 18 August 2026 and a no-endorsement boundary.

Presentation graphics must be derived from verified evidence or from
deterministic, non-scientific interface decoration. They must not introduce
invented benchmark values, histology imagery or new scientific claims.

The presentation may use the `DEMO_COMPLETE` stage after its mandatory gates
pass. Its scientific result boundary remains `PRIMARY_STUDY_COMPLETE`.

## Explicitly deferred

The unchanged original confirmatory study, M9 original-label/expert-review work,
external validation, production capsule/Q/E publication infrastructure and
current-session unattended-resume work are deferred as future work. They are
not prerequisites for this MVP and are not represented as executed. In
particular, the MVP does not claim `CONFIRMATORY_COMPLETE`,
`EXTERNAL_VALIDATION_READY`, or `EXTERNAL_VALIDATION_COMPLETE`.

No further training or scientific retry is authorized or required by this MVP
scope. Deferred code and historical evidence remain preserved.

## Acceptance

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe src
python scripts\present_demo.py --verify-only
.venv\Scripts\python.exe -m histo_audit demo build --project-root . `
  --run-dir artifacts\runs\20260727T133947.089370Z_pannuke_primary_orphan_recovery `
  --qc-bundle reports\pannuke_qc --output-dir artifacts\mvp_demo_rebuild
.venv\Scripts\python.exe -m histo_audit demo verify `
  --output-dir artifacts\mvp_demo_rebuild
```

The build is CREATE-NEW in effect: it refuses to overwrite an existing output
directory. The presentation only reads checksum-verified selected artifacts and
does not modify the run, raw masks, source annotations, frozen files or labels.

## Presentation language

This remains a university research prototype, not a diagnostic system. It may
identify a “potentially inconsistent annotation” and an item “recommended for
expert review”. It must never claim that model disagreement proves a medical
error or that a pathologist was wrong. Machine-oriented completion and analysis
disposition codes remain available in the evidence package and project
documentation, but are omitted from the reader-facing narrative when plain
English communicates the same boundary more clearly.
