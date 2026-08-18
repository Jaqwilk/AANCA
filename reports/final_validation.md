# Historical synthetic validation checkpoint

Date: 2026-07-17 (Europe/Warsaw)

Historical record only. It captures the software-only checkpoint that existed on
the date below and must not be used as the current project-readiness summary. The
current presentation and scientific boundaries are maintained in `README.md`,
`STATUS.md`, and `MVP_SCOPE.md`.

Status: `PIPELINE_COMPLETE`. This validates the deterministic synthetic software pipeline only. It is not a PanNuke, clinical, primary-study, confirmatory, or external-validation result.

## Environment evidence

- Windows 10 Enterprise LTSC, build 19044; CPython 3.12.3 in `.venv`.
- Intel Core i7-13700K; 31.75 GiB RAM.
- NVIDIA GeForce RTX 4070; 12,282 MiB VRAM; driver 551.78.
- PyTorch 2.12.1+cu126; torchvision 0.27.1+cu126; Cleanlab 2.9.0.
- CUDA detected one device. An explicit 512×512 matrix multiplication and backward pass produced finite loss and gradients.
- The project directory was writable; `uv pip check` found all 64 installed packages compatible.
- Complete machine-readable evidence: `reports/doctor.json`.

## Quality gates executed after the final source changes

- `pytest -q`: 111 passed in 47.14 seconds after the final source change.
- `ruff check .`: passed.
- `ruff format --check .`: 63 files already formatted.
- `mypy src`: passed for 48 source files.
- Focused representation/report/workflow tests: passed.
- Current generating source-tree SHA-256: `d55529065c41dd5a65fbdf311f459784221ab2269421b7acaed5f7dd4540720a`.

## Canonical tracked artifacts

| Scenario | Run ID | Status | Integrity | Artifact root SHA-256 | Source matches current |
|---|---|---|---|---|---|
| 10% symmetric controlled corruption | `20260717T162925.902444Z_synthetic_smoke_5573505315` | completed/sealed | valid | `95fdefc840f725c5fadcb15804e1a7252aa907d21d6a4080d334144acff35876` | yes |
| 0% corruption edge case | `20260717T162948.870526Z_synthetic_smoke_zero_corruption_a4d5f87ca0` | completed/sealed | valid | `2ba716349b010c5bc71f8c7a3b509bfd0f7856c6b138ef8fa0d491050e9236bd` | yes |

Independent verification found no missing, added, or changed artifact in either sealed run, and both registry records were present. Strict reconciliation passed for the complete synthetic dataset/source manifest, predictions, corruption rows, rankings, per-sample neighbours, every restoration repeat, final-test probabilities, counts, OOF folds, labels, tissue values, representation identity, configuration, and metrics. Reports contained no prohibited terminology, unsupported non-finite literal, or unsourced numeric result.

The standalone deterministic generator also completed at `data/synthetic/791fe34c3bb9`: 300 samples in 60 groups, generator schema 2, definition SHA-256 `791fe34c3bb9042b73badd8209afa1b2b673922e20f2da2da28e9a70d67525b2`, dataset file SHA-256 `fb01864245d36b9ed96e60c7d4c21b52a9bb69508a1d2923c5a370d923ab0829`, and manifest SHA-256 `8f8496ebcae0814a1f78192414c9ec0b921640f66beb9748f0c2dd05c60d39eb`.

## Scientific-safeguard checks

- Source patch is the grouping unit; no nucleus-level random split exists.
- The canonical synthetic final-reference partition is selected by a fraction of whole source groups; the unused official-fold selector is explicitly `not_applicable` rather than carrying a misleading fold number.
- Every audit-pool sample has exactly one OOF probability vector, and train/held-out group overlap is zero.
- Final-reference groups are disjoint, uncorrupted, and absent from audit fitting/selection.
- `pre_corruption_label`, `observed_label`, and `is_injected_corruption` reconcile exactly with the corruption manifest.
- Random and guided review use identical integer budgets.
- Restoration changes only reviewed injected labels; the final-reference labels remain unchanged.
- The canonical symmetric mechanism correctly records feature-space independence as `not_applicable`.
- The implemented instance-dependent test path uses independently typed morphology-only generator and colour-only auditor features and rejects name-only independence claims.
- At 0% corruption, AP, AUROC, recall, lift, and comparative inference are explicitly undefined; score distributions and false-alert counts are retained instead.
- Subgroup AP is suppressed below 100 samples or 10 injected events.
- Exact target identity is saved and visible as full patch plus contour, target crop plus contour, and target-highlighted context.
- Tissue, class, fold, corruption, PR, review-budget, bootstrap, downstream, subgroup-support, target, neighbour, and controlled error-analysis figures are generated from machine-readable artifacts and linked in both reports where applicable. Figure provenance hashes every input and records selections/ties/transforms.

Manual visual inspection found the target contour/crop/highlight aligned to the same synthetic instance, readable PR and bootstrap axes, complete example metadata, visibly distinct fold-safe neighbour groups, and the expected 0% score/false-alert alternatives. The 0% raw-score panel explicitly warns that method-specific scales are not directly comparable. This is visual software QA, not histopathology validation.

## Real-data and later-stage gate

`reports/doctor.json` records `pannuke_detected: false`. `data validate-pannuke` and `experiment pilot` were both executed and stopped with `GATED [REAL_DATA_UNAVAILABLE]`, exit code 2. Accordingly:

The later stage commands were also executed: `experiment primary` stopped with `GATED [PRIMARY_STUDY_LOCKED]`, exit code 2, and `experiment confirmatory` stopped with `GATED [CONFIRMATORY_LOCKED]`, exit code 2.

- no PanNuke checksum or semantic-validation result exists;
- no actual cross-fold duplicate result exists;
- no PanNuke pilot was run;
- `PRE_REGISTRATION.md` remains DRAFT;
- no primary or confirmatory result exists;
- no original-PanNuke ranking or external-review package exists;
- no genuine expert response exists.

The code includes a fixed PanNuke pilot, preregistration-freeze verification, exploratory original-label audit, blinded-package validation, and full controlled synthetic example grids. Full real primary/confirmatory orchestration and real primary-report grids remain pending later gates and must not be described as completed.

## Exact continuation

```powershell
$env:PANNUKE_ROOT = 'C:\path\to\verified\PanNuke'
.venv\Scripts\python.exe -m histo_audit data validate-pannuke --project-root . --root $env:PANNUKE_ROOT
```

Use only a legitimately obtained, licence-compliant, provenance-verifiable release. No mirror or data-dependent claim is substituted automatically.
