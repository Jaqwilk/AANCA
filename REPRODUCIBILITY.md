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
The data-generation command is idempotent: an identical existing package is verified
and reused, while any changed, partial or unexpected artifact fails closed.

The `Scientific software` GitHub Actions workflow first verifies the sealed static
package, then executes lint, formatting, the complete test suite and this synthetic
workflow on both Ubuntu and Windows. Tests that exercise Windows-native handle
custody are explicitly skipped on non-Windows systems rather than failing during
test collection.

## What package verification proves

`python scripts/present_demo.py --verify-only` checks a closed five-file allowlist,
file sizes, SHA-256 identities and consistency of selected evidence fields. It can
detect a changed or incomplete presentation package. It does not run a model,
recompute a bootstrap, reload PanNuke or prove that the upstream analysis was
scientifically correct.

## Independently recalculating the saved primary results

The public GitHub release
[`primary-evidence-v1`](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)
contains all 185 completed-cell rankings and OOF probability arrays, the full
2,000-draw group bootstrap, subgroup table, H4 restoration arrays, frozen controls
and per-cell manifests. The three assets total approximately 2.75 GB and are rooted
in the accepted run's existing SHA-256 identities.

After extracting `aanca-primary-evidence-v1.zip`, run:

```text
uv run python scripts/verify_primary_evidence.py PATH/TO/aanca-primary-evidence-v1
```

The verifier intentionally does not import `histo_audit`. It independently
recalculates the 33 numeric primary comparison bootstrap summaries, intervals,
one-sided p-values and Holm corrections, preserves the three unavailable H6 entries,
and recalculates the adverse H4 macro-F1 result from 100 random-review repetitions.
See [`PUBLIC_EVIDENCE.md`](PUBLIC_EVIDENCE.md) and
[`evidence-release-manifest.json`](evidence-release-manifest.json) for exact asset
sizes, digests and scope.

This is independent recalculation of the saved numeric evidence, not independent
replication of the entire experiment from images. PanNuke images and masks are not
redistributed. Fold checkpoints were not retained in the accepted run, so model
retraining still requires a lawful PanNuke copy and the frozen code/configuration.

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
6. Independently replicate the complete image-to-result run in a separate environment;
   the public release now supports result recalculation and OOF/ranking inspection,
   but not a second independent execution of model training.

Until those steps are completed, AANCA is a functioning research prototype and a
synthetic-corruption benchmark. It is not expert-validated, externally validated,
clinical, diagnostic or production-ready.
