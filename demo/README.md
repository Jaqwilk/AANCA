# AANCA presentation MVP

The static MVP is generated at `artifacts/mvp_demo/index.html` from the accepted,
sealed primary and the checksum-bound PanNuke QC bundle. The dependency-free
launcher verifies it, serves only on loopback by default, and opens the browser;
it does not execute a model or require the ML environment.

```powershell
python scripts/present_demo.py
python scripts/present_demo.py --verify-only
```

Directly opening `artifacts/mvp_demo/index.html` remains a zero-server fallback.
After installing the full environment, `histo-audit demo serve` and
`histo-audit demo verify` expose the same workflow through the project CLI.

The presentation follows a long-form article layout: the study boundary, method,
metric definitions and findings share a centered 640 px reading column, while
figures widen only when their data require it. All seven registered questions and
their complete answers are retained. Desktop presents them as one shortened pinned
sequence; mobile and reduced-motion layouts keep them in normal document flow.

The reading order is thesis and scope, controlled method, guidance for reading the
metrics, the adverse H4 result, the remaining H1-H7 findings, detailed evidence,
PanNuke QC, experimental-integrity and interpretation limits, reproducibility, and
author context. Narrative paragraphs connect each figure to the question it answers;
plots and tables do not stand alone as dashboard modules.

The 36-entry comparison atlas stays visible. The duplicate numeric table remains
available in one optional audit disclosure and uses a compact keyboard-scrollable
viewport when opened. On narrow screens, each evidence entry becomes a labelled
two-column record.

Study facts and metric definitions use compact editorial rows. The release uses
thin rules and whitespace instead of repeated rounded containers; borders remain
only where they communicate an interactive control, code sample, or image boundary.

Article headings below the opening thesis use the same Inter scale as that thesis:
29 px at desktop, weight 500 and 1.48 line height. Explanatory paragraphs use Inter
at 16 px, weight 400 and 1.74 line height. Small marketing-style kickers were removed
from the article chapters; the command examples remain in compact code boxes.

The article is deliberately minimal. It does not reserve multi-viewport scroll space
for the method diagram or hide ordinary chapters behind entrance animations. The one
intentional presentation sequence is “What the study actually learned.”: on desktop,
its seven findings advance one at a time; mobile and reduced-motion layouts show the
same complete content statically. Section hierarchy is otherwise carried by typography
and compact whitespace; horizontal rules remain only where they help read evidence.
The first animated finding is complete on entry, and direct links align to the start
of the sequence rather than an intermediate question.

The five-stage method is explained once by its complete diagram and a short framing
paragraph. The H4 ledger entry is a reminder of the full result above it, the exact
seed identities and SHA-256 hashes sit in an optional audit disclosure, and the
footer carries one concise responsible-use reminder. The comparison atlas and full
numeric table remain available without repeating their complete content in the
main narrative.

The dependency-free launcher binds request paths explicitly to the verified package
root. This avoids platform-specific path translation failures and makes the printed
root URL serve `index.html` directly.

The package visibly states that it is not diagnostic, preserves the
`amended_or_exploratory` disposition, includes every saved primary comparison,
and identifies only a “potentially inconsistent annotation” or an item
“recommended for expert review”. The later NuCLS multi-rater validation is now
complete and is reported as null/adverse; confirmatory CNN work and newly recruited
blinded expert review remain future work. See `MVP_SCOPE.md` for the original reduced
presentation boundary and `reports/nucls_external_validation_results.md` for the
later external result.
