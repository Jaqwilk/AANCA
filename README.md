# Automated Auditing of Nucleus Class Annotations

This repository implements a reproducible university research workflow for detecting and prioritising **potentially inconsistent** nucleus class annotations. The controlled benchmark records intentional label corruptions, evaluates group-safe out-of-fold risk rankings at fixed review budgets, and measures simulated restoration utility. It is not a diagnostic tool and does not declare source annotations medically wrong.

## Current status

The presentation MVP is `DEMO_COMPLETE` as of 2026-08-18. Its scientific
result boundary remains `PRIMARY_STUDY_COMPLETE`: the accepted PanNuke primary
completed 185/185 required cells with zero required failures and saved all 36
statistical comparisons. The result is explicitly
`amended_or_exploratory`; confirmatory, expert review and external validation
are not claimed. Presentation schema v2 includes the complete H2 subgroup
summary, the adverse H4 downstream result, all 36 H1/H3/H5/H6/H7 comparisons,
and an explicit warning that the three registered instance-dependent seeds
produced byte-identical rankings and OOF predictions and are not independent
replications. See `MVP_SCOPE.md` and `STATUS.md` for exact evidence.

Accepted sealed primary:

- `artifacts/runs/20260727T133947.089370Z_pannuke_primary_orphan_recovery`

## Open or reproduce the MVP

Open `artifacts/mvp_demo/index.html` in any browser. The English presentation is
authored by Natan Smogór and dated 18 August 2026. The package is static and does
not need a server or GPU. It uses a restrained Linear-inspired visual system,
compact Inter typography and an interactive Three.js hero in which source
nucleus contours remain fixed while review evidence is copied into a ranked
expert-review queue. The hero is a conceptual workflow, not microscopy or
benchmark data; reduced-motion mode shows its final state without animation.
The page also includes GSAP and ScrollTrigger motion, a five-stage method
narrative, evidence-bound H2/H4/forest charts, the complete filterable comparison
table, mobile navigation and print styles. Every plotted value is rendered from the
checksum-verified evidence package; no benchmark value is illustrative or
generated. Core content and controls remain usable with local font fallbacks if
the network is unavailable; the pinned GSAP 3.15.0 and Three.js 0.185.1 visual
enhancements load from jsDelivr when a network is available.

```powershell
.venv\Scripts\python.exe -m histo_audit demo verify --output-dir artifacts\mvp_demo
```

To reproduce it, choose a new empty output directory because the command never
overwrites an existing package:

```powershell
.venv\Scripts\python.exe -m histo_audit demo build --project-root . `
  --run-dir artifacts\runs\20260727T133947.089370Z_pannuke_primary_orphan_recovery `
  --qc-bundle reports\pannuke_qc --output-dir artifacts\mvp_demo_rebuild
```

## Scientific guardrails

- The minimum split unit is the source patch (`group_id`), never an individual nucleus.
- One official outer fold remains an untouched, uncorrupted final reference test.
- Model-based primary audit scores are computed only from group-safe OOF predictions.
- `pre_corruption_label` and `observed_label` remain permanently separate.
- Injected-corruption results and exploratory original-label rankings are reported separately.
- Every reported experiment metric must come from an immutable run artifact.

Read `SPEC.md`, `PLAN.md`, `STATUS.md`, `PRE_REGISTRATION.md`, and `ETHICS_AND_LIMITATIONS.md` before changing experiments.

## Windows setup (PowerShell)

```powershell
uv sync --dev
.venv\Scripts\python -m histo_audit doctor
.venv\Scripts\python -m histo_audit data generate-synthetic --config configs\smoke.yaml
.venv\Scripts\python -m histo_audit experiment smoke
.venv\Scripts\python -m histo_audit experiment smoke --config configs\smoke_zero.yaml
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy src
```

The checked-in dependency definition selects the current official PyTorch 2.12.1 CUDA 12.6 wheels. NVIDIA documents minor-version compatibility across CUDA 12.x for this driver range; the resolved build is retained only after an actual local CUDA forward/backward test. It does not require a system CUDA Toolkit. If hardware changes, re-run `doctor` and re-resolve the PyTorch index from current official guidance.

## macOS/Linux setup

The CUDA-specific `tool.uv.sources` entry targets this Windows/NVIDIA workstation. On a CPU-only or non-NVIDIA system, create a platform-specific lock using the official PyTorch selector rather than installing a CUDA Toolkit. Commands otherwise use POSIX activation/paths:

```bash
uv sync --dev
.venv/bin/python -m histo_audit doctor
.venv/bin/python -m histo_audit experiment smoke
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

## Required CLI surface

The project exposes `doctor`; synthetic generation; PanNuke
validation/manifest/duplicate inspection; representation extraction; smoke and
pilot experiments; hard-gated primary and confirmatory stage commands;
preregistration freeze; original-label audit; external review-package creation;
report building; and the static `demo build/verify` surface. Deferred commands
remain fail-closed and are not part of the MVP acceptance path.

## Data and results

Dataset binaries, local embeddings, and generated runs are ignored by Git. The
small canonical `artifacts/mvp_demo` package is the deliberate exception so a
reviewer receives the presentation and its checksum manifest. See
`DATASET_SETUP.md` for the acquisition gate. Every run is written under
`artifacts/runs/<run_id>/` and appended to `artifacts/runs/registry.csv`;
completed runs are never silently overwritten.

Current evidence and the exact MVP verification command are in `STATUS.md`.
The PanNuke validator intentionally fails closed if it cannot establish all
three official folds and their actual semantic structure. Dataset licensing
remains separate: local release evidence applies CC BY-NC-SA 4.0 explicitly to
the `masks/` directory, while this project independently restricts all PanNuke
use to non-commercial research and requires both PanNuke citations.
