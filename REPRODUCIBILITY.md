# Reproducibility and evidence boundary

This document distinguishes three different operations that must not be presented
as equivalent: verifying the published article package, reproducing the synthetic
software workflow, and independently recomputing the PanNuke primary study.

## What the public repository can reproduce

After installing Python 3.12 and `uv`, a reviewer can run the complete deterministic
synthetic path on Windows or Linux:

```text
uv sync --dev
uv run histo-audit data generate-synthetic --config configs/smoke.yaml
uv run histo-audit experiment smoke
```

The smoke command writes to `artifacts/smoke_runs` by default. It does not read or
append to `artifacts/runs/registry.csv`, which is a historical workstation ledger
containing absolute Windows paths. A reviewer can also pass an empty directory with
`--runs-root`. Synthetic success validates corruption, group-safe OOF scoring,
ranking, restoration, statistics and artifact plumbing; it is not PanNuke evidence.

The `Scientific software` GitHub Actions workflow is configured to execute lint,
formatting, the complete test suite and this synthetic workflow on both Ubuntu and Windows. Tests
that exercise Windows-native handle custody are explicitly skipped on non-Windows
systems rather than failing during test collection.

## What package verification proves

`python scripts/present_demo.py --verify-only` checks a closed five-file allowlist,
file sizes, SHA-256 identities and consistency of selected evidence fields. It can
detect a changed or incomplete presentation package. It does not run a model,
recompute a bootstrap, reload PanNuke or prove that the upstream analysis was
scientifically correct.

## What is not independently reproducible from GitHub

The accepted primary run occupies approximately 46 GB locally and contains
thousands of files. The public repository does not distribute PanNuke binaries, full per-cell
corruption manifests, OOF probabilities, rankings, checkpoints or the 372 MB
bootstrap array. It therefore cannot independently recompute H1-H7 from the public
checkout alone. The checked-in `artifacts/mvp_demo/evidence.json` is a compact
reporting extract, not a substitute for those upstream arrays.

The retained local primary statistics manifest identifies the principal omitted
outputs:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `primary_statistics.json` | 22,498,321 bytes | `c3685fe9863fd73b1298f0558212cb5267b07c3ce6e4e4f37018dec55c115ac0` |
| `primary_subgroups.csv` | 4,208,358 bytes | `36be649fef067de82cd11b77f508f0a6fe62f649d393a1c9975a4523c24d166e` |
| `primary_bootstrap_evidence.npz` | 372,330,793 bytes | `35f8017cfcc887a1e94498a72e6868481088ce0fac4d8d2369d32504780bafa2` |

A future archival release must publish licence-compatible OOF/ranking and bootstrap
evidence in a durable data repository, with a manifest rooted in the existing
accepted run. Until then the correct claim is “internally traceable and publicly
inspectable summaries,” not “independently reproduced primary results.”

## Public-history disclosure

The timestamped freeze artifact at
`artifacts/preregistrations/20260719T002902.432341Z/git_state.json` records
`commit: null`, `dirty: true` and an untracked source tree. The first public Git
commit is dated 19 August 2026. The July manifests and hashes preserve the identity
and ordering of the local files that were frozen, but the later public history is
not an independent timestamping service and cannot prove that outcomes were unseen.
The accepted analysis is consequently retained as `amended_or_exploratory`.

## Methodological limits that require new evidence

These are open research tasks, not software defects that can be closed by changing
wording or adding tests:

1. Have multiple qualified pathologists blindly review a preregistered top-K sample
   and an equal-sized random sample of naturally occurring, non-injected cases.
2. Report agreement, ambiguity and abstention instead of forcing one expert truth.
3. Repeat splitting at patient or WSI level when verified identifiers are available.
4. Run a prospectively declared sensitivity analysis whose fold allocation uses
   only labels available in a real audit. The accepted benchmark used
   `pre_corruption_label` for fold balance; it was not a model input, but it is
   benchmark-only information.
5. Evaluate an external dataset and a downstream intervention defined before its
   outcomes are inspected.
6. Deposit the full licence-compatible prediction and statistical evidence needed
   to recompute every published comparison.

Until those steps are completed, AANCA is a functioning research prototype and a
synthetic-corruption benchmark. It is not expert-validated, externally validated,
clinical, diagnostic or production-ready.
