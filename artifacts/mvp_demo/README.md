# AANCA presentation MVP

This five-file, read-only article package was generated from checksum-verified
PanNuke primary evidence plus the checked-in NuCLS, MoNuSAC and PUMA result
authorities. The accepted PanNuke run is `20260727T133947.089370Z_pannuke_primary_orphan_recovery`.

From the repository root, verify the complete package and open it locally:

```powershell
python scripts/present_demo.py
```

The launcher uses only the Python standard library. It verifies every packaged file
before serving `127.0.0.1`; it never runs a model or changes data. For verification
without a browser or server:

```powershell
python scripts/present_demo.py --verify-only
```

With the project environment installed, `uv run histo-audit demo serve` and
`uv run histo-audit demo verify` provide the equivalent CLI workflow.

## Current scientific position

- Primary study: `PRIMARY_STUDY_COMPLETE`; its accepted PanNuke analysis remains
  permanently `amended_or_exploratory` and H4 was adverse.
- External evaluation: `EXTERNAL_VALIDATION_COMPLETE`. NuCLS natural multi-rater
  claims were not supported; MoNuSAC controlled retrieval passed but downstream and
  class-safety gates failed; the frozen PUMA controlled confirmation passed all seven
  prospective gates.
- Presentation: `DEMO_COMPLETE`.
- Confirmatory stage: not reached. `CONFIRMATORY_COMPLETE` is not claimed.
- Natural-data action: `retain_uncorrected`.

The positive PUMA result supports transfer under controlled label noise. It does not
show that AANCA detects pathologist errors, discovers biological truth, improves a
real laboratory workflow or is clinically useful. The software never modifies
source annotations automatically; it ranks potentially inconsistent annotations
for qualified expert review.

## Package contents

- `index.html` — responsive English article, including the retained “What the study
  actually learned.” sequence;
- `evidence.json` — sourced primary, external, controlled-confirmation, stress,
  sensitivity, current-action and next-phase summaries;
- `pannuke_mask_qc_overlays.png` — deterministic source-ingestion QC preview;
- `README.md` — this handoff;
- `manifest.json` — SHA-256 allowlist binding every other package file.

The primary evidence retains all 36 registered H1/H3/H5/H6/H7 entries: 33 numeric
results and three explicitly unavailable H6 cells. Displayed intervals, adjusted
p-values, source hashes and external summaries are read from machine-readable
authorities rather than retyped into the page.

## Next phase

The provisional AANCA v2 research phase requires: a new blinded multi-rater natural
reference; nested group-safe measured-utility development; one prospectively frozen
policy; one-shot untouched patient/WSI confirmation; and a prospective multi-site
AANCA-versus-control workflow study. Promotion requires ranking, downstream
confidence intervals, every-class safety and workflow utility to pass together.

Source code, frozen protocols, configs, independent verifiers, evidence and the
complete limitation statement are at <https://github.com/Jaqwilk/AANCA>.

Author: Natan Smogór. Updated: 21 August 2026.
