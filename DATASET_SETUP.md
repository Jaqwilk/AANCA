# PanNuke Dataset Acquisition Gate

PanNuke data are used only after provenance, licence/terms, and actual file semantics are verified. The code searches, in order, `PANNUKE_ROOT`, `data/raw/pannuke`, and documented nearby workspace locations. It never silently substitutes an unverified mirror.

## Current local result

As of 2026-08-18, acquisition, schema/semantic validation, overlap/void QC,
manifest construction, and the real-data M5 gate are complete. All three
archives (`fold_1.zip`, `fold_2.zip`, and `fold_3.zip`) and their extracted fold
directories remain under `C:\Users\NATAN\Documents\AANCA\data\raw\pannuke`.
The accepted real-data primary run is recorded in `README.md`; confirmatory,
expert-review, and external-validation work remains deferred. This paragraph
supersedes the historical 2026-07-17 presence-only checkpoint.

The acquisition source is the [official University of Warwick PanNuke
page](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/). The local release
documents explicitly apply CC BY-NC-SA 4.0 to the `masks/` directory and its
contents; they do not locally establish the same licence scope for every image/type
file. Independently, this project restricts all PanNuke use to research and
non-commercial work. Publications and reports using this release must cite both
PanNuke works recorded as `gamper2019pannuke` and `gamper2020pannukeextension` in
`references/references.bib`. The completed schema-v2 acquisition manifest preserves
the exact licence-scope statement, source evidence, and both citation identifiers.
The archive identities reconciled by that manifest are:

| Archive | Reported bytes | Reported local SHA-256 |
| --- | ---: | --- |
| `fold_1.zip` | 700,275,281 | `6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b` |
| `fold_2.zip` | 658,842,552 | `5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07` |
| `fold_3.zip` | 717,969,882 | `c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d` |

These sizes and hashes were supplied by the user as local acquisition evidence;
they are not publisher checksums. The schema-v2 acquisition command independently
recomputed and exactly reconciled them, passed ZIP CRC and safe-path checks, bound
the extracted inventory, and verified an unchanged raw snapshot. The independent
semantic/QC gate subsequently reproduced cross-class overlap and supplied-background
void counts in the saved anomaly-safe QC bundle under `reports/pannuke_qc`; M5 then
passed without modifying the source arrays.

## Immutable local placement and validation policy

1. Use only the provenance-verifiable Warwick release, comply with the locally
   documented CC BY-NC-SA 4.0 terms for `masks/`, retain this project's broader
   research/non-commercial restriction, and provide the required PanNuke citations.
   Do not substitute an unverified mirror merely to unblock execution.
2. Preserve the archives and extracted arrays byte-for-byte. Do not rename,
   reshape, repair, relabel, or write derived values back to the raw release.
3. Treat channel 5 as the supplied background channel, not as a computed complement
   of positive channels 0--4. Pixels assigned to neither a positive class nor the
   supplied background remain unlabeled voids.
4. Count and flag every cross-class overlap pixel without choosing a class. Retain
   every touching nucleus in the manifest with
   `touches_cross_class_overlap`, exclude it from all primary- and
   confirmatory-outcome-eligible analyses before splitting/modelling, and flag the
   containing source patch. Apply the same eligibility mask to every primary and
   required confirmatory scenario.
5. Use no anomaly threshold selected from observed frequency, model performance, or
   final-reference outcomes. Complete per-fold, patch, pixel, instance, eligibility,
   and void counts must be saved.
6. Use the default local path or set `PANNUKE_ROOT` to another legitimately obtained
   immutable copy:

```powershell
$env:PANNUKE_ROOT = (Resolve-Path 'data\raw\pannuke').Path
.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root $env:PANNUKE_ROOT --max-samples-per-fold 100000 --max-overlay-patches 24
.venv\Scripts\python.exe -m histo_audit data build-manifest --project-root . --root $env:PANNUKE_ROOT --batch-rows 4096
.venv\Scripts\python.exe -m histo_audit data audit-duplicates --project-root . --root $env:PANNUKE_ROOT --embedding-device cuda
```

On macOS/Linux:

```bash
export PANNUKE_ROOT=/path/to/verified/PanNuke
.venv/bin/python -m histo_audit data validate-pannuke --project-root . --root "$PANNUKE_ROOT" --max-samples-per-fold 100000 --max-overlay-patches 24
.venv/bin/python -m histo_audit data build-manifest --project-root . --root "$PANNUKE_ROOT" --batch-rows 4096
.venv/bin/python -m histo_audit data audit-duplicates --project-root . --root "$PANNUKE_ROOT" --embedding-device auto
```

The validator must locate every released fold; inspect rather than assume
image/mask/tissue names, shapes, dtypes, channels, and class order; bind its raw
inventory to the already verified acquisition evidence; report complete anomaly
counts; and create representative normal/overlap/void/exclusion overlays. Default validation
artifacts include `reports/pannuke_validation.json`,
`reports/pannuke_validation.md`, `data/manifests/raw_files_sha256.csv`, and
`artifacts/figures/pannuke_overlay_grid.png`. The manifest and duplicate commands
may proceed only after validation succeeds; `experiment pilot` remains gated until
all M5 acceptance checks and the full two-signal duplicate gate pass.

Raw files are never committed. If the official source requires manual licence
acceptance, login, or download, that manual step is a hard gate; no access control
is bypassed. Listing a command here does not assert that it has run; executed
commands and their evidence belong in `STATUS.md`.

## Cross-fold duplicate audit

`data audit-duplicates` is read-only and restricts candidates to pairs from different official folds. It hashes every canonical source patch, confirms exact candidates with array equality, compares deterministic perceptual hashes, and attempts an independent cosine-similarity signal from official frozen torchvision ResNet-18 embeddings. Pretrained weights are never downloaded implicitly: use an existing checksum-validated embedding cache with `--embedding-cache`, or explicitly authorise a download with `--allow-weight-download`. A missing checkpoint is recorded as a blocker and produces no fabricated cosine scores.

The default command writes the required artifacts to:

- `reports/cross_fold_duplicates.md`;
- `artifacts/rankings/cross_fold_duplicate_candidates.csv`;
- `artifacts/provenance/pannuke_patch_hashes.csv`;
- `reports/figures/cross_fold_duplicate_candidates.png`.

The default attempts full-release coverage for both perceptual and embedding comparisons; embeddings run in memory-bounded extraction chunks and retain the encoder's batch/OOM backoff safeguards. The required two-signal near-duplicate gate is complete only when both comparisons cover every validated patch. A user-selected `--max-perceptual-patches` or `--max-embedding-patches` bound is permitted for an exploratory run, but its report remains explicitly incomplete. All candidates remain `review_only`; the command never deletes a patch, changes a fold, or modifies an annotation.

The class-order implementation reference is the archived official [Tissue Image Analytics PanNuke metrics repository at commit `c00014d`](https://github.com/TissueImageAnalytics/PanNuke-metrics/tree/c00014d766ca1be142b81bea19d9ef4315cde65a). It documents positive indices 0–4 as neoplastic, inflammatory, connective/soft tissue, dead, and non-neoplastic epithelial. This does not authorise assuming the downloaded array channel/background layout: the validator still inspects the actual release and fails when semantics are ambiguous.

## NuCLS (optional external validation)

NuCLS is not a substitute for the primary PanNuke benchmark. Before use, verify its licence, multi-rater structure, class definitions, source separation, responsible mapping, and domain differences. Set `NUCLS_ROOT` only for a legitimately obtained local copy.
