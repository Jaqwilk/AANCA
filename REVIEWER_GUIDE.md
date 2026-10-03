# Reviewing AANCA v1

Start at **[aancastudy.org/review/](https://aancastudy.org/review/)**. The guide is
readable without an account, installation or JavaScript. No Harvard affiliation,
endorsement or knowledge of an individual professor's review process is implied.

## Read the evidence

Begin with [PROFESSOR_BRIEF.md](PROFESSOR_BRIEF.md), the article's
[findings](https://aancastudy.org/#results) and
[ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md). AANCA ranks potentially
inconsistent nucleus class annotations recommended for expert review; it never
automatically changes source annotations. Natural data remain `retain_uncorrected`.

For methods, use [SPEC.md](SPEC.md), [PRE_REGISTRATION.md](PRE_REGISTRATION.md),
[PLAN.md](PLAN.md) and [DECISIONS.md](DECISIONS.md). Distinguish patch-level from
verified patient/WSI independence, require group-safe OOF predictions, and separate
controlled injected changes from independent-expert disagreement. Check the
outcome-exposure/recovery chronology before interpreting confirmatory claims.
`PRE_REGISTRATION.md` retains its historical pre-freeze state; it is not a living
completion report. Read the accepted execution/recovery records in `STATUS.md` and
`DECISIONS.md` alongside it rather than inferring timing from its current location.
`CONFIRMATORY_COMPLETE` has not been reached.

Use [PUBLIC_EVIDENCE.md](PUBLIC_EVIDENCE.md) to follow individual claims to saved
arrays and source records, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the
scope of each verifier. Retain the adverse PanNuke H4 and NuCLS downstream results
and failed MoNuSAC downstream/class-safety gates. H4's interval is a central 95%
range across 100 random-review repetitions on a fixed reference set, **not a
group-bootstrap confidence interval**. Model disagreement is not adjudicated error.

## Check locally without the research environment

Download the ZIP and its SHA-256 sidecar from
[reviewer-kit-v1](https://github.com/Jaqwilk/AANCA/releases/tag/reviewer-kit-v1).
Compare its digest with the release sidecar, then extract and open `index.html`.
On Windows, use `Get-FileHash aanca-reviewer-kit-v1.zip -Algorithm SHA256`; on
Ubuntu, use `sha256sum aanca-reviewer-kit-v1.zip`. A self-contained manifest detects
internal changes; it does not independently authenticate its publisher.

With Python **3.12**, from the extracted directory (or this repository):

```text
python -I scripts/review_project.py
```

This is read-only, uses the Python standard library, makes no network requests and
checks the thirteen-file article, upstream evidence identities and selected narrative
contracts. In the ZIP it also checks the closed kit inventory. It does not train
models, verify every prose sentence or establish that a medical annotation is wrong.
The kit has no raw histopathology images, pretrained weights or research environment.
Its manifest records the source revision, build-tree state and every included file.

An optional saved-array recalculation needs only NumPy:

```text
python -m pip install -r requirements-reviewer.txt
python -I scripts/review_project.py --numeric --report ../aanca-review.json
```

This independently recalculates the retained NuCLS multi-rater and MoNuSAC numeric
evidence, including matched baselines and group bootstrap gates. It deliberately
retains `not_supported` conclusions. These are project-maintained independent
implementations, not third-party validation. It does not recalculate all RIVA,
MIDOG++, PUMA or independent-pathologist NuCLS results; their separate scopes and
commands are documented in `REPRODUCIBILITY.md`. `--json` prints the full receipt;
failed checks return a nonzero exit code. Store reports outside the immutable kit.

To compare the live article with the downloaded snapshot, explicitly enable network
access with `python -I scripts/review_project.py --online`. An older valid snapshot
can correctly fail equality after a later website release; consult its source
revision rather than treating that as failure of the underlying study.

## Recalculate the primary evidence or exercise the software

The separate [primary-evidence-v1 release](https://github.com/Jaqwilk/AANCA/releases/tag/primary-evidence-v1)
provides the larger PanNuke numeric evidence. The minimum primary-evidence ZIP is
358,518,237 bytes; additional rankings/OOF assets bring the release to about 2.75 GB.
Its expected identities and instructions are in `PUBLIC_EVIDENCE.md` and
`evidence-release-manifest.json`. `scripts/verify_primary_evidence.py` uses NumPy
without importing AANCA; run it on the separately extracted primary evidence.

The complete source repository is needed for software tests and the deterministic
synthetic workflow. Use Python 3.12 and `uv`:

```text
uv sync --dev --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run histo-audit experiment smoke --runs-root artifacts/reviewer-smoke
```

The synthetic run writes only its derived outputs and does not modify source
annotations or the historical accepted-run registry. It is a software demonstration,
not clinical evidence. The full environment is substantially larger than the kit.
CI partitions the full suite into isolated jobs on Ubuntu/Windows, then requires
complete execution receipts; platform/CUDA/weight-cache skips remain explicit.

Retraining the source-image experiment additionally requires lawful dataset access,
frozen configurations and suitable compute. It is different from recalculating saved
predictions and different from replication on new data. Opened final groups cannot
be reused to select a revised model.

## Evaluation terms and scientific responsibility

[LICENSE](LICENSE) permits local copying and execution solely for non-clinical
scientific evaluation, peer review and verification. Other rights are reserved;
dataset/dependency terms are separate. This is not an open-source licence.
[CITATION.cff](CITATION.cff), [CONTRIBUTIONS.md](CONTRIBUTIONS.md) and the
[bibliography](references/references.bib) provide citation and contribution details.
AI assistance supplied no expert labels and is not an independent validator.

This review path was informed by [HMS reproducibility guidance](https://datamanagement.hms.harvard.edu/collect-analyze/reproducibility)
and [HMS access/reuse guidance](https://datamanagement.hms.harvard.edu/access-reuse).
Those sources motivate discoverable documentation and verifiable materials; they
are not a certification of AANCA.
