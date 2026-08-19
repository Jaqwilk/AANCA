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

The article uses a centered 640 px reading column. Its findings sequence contains
only the section title, each registered question, and its complete answer. On
motion-capable desktop browsers, the answer resolves word by word and snaps to a
complete reading state before another gesture can advance by one question. At
900 px and below, with reduced motion, or when the animation CDN is unavailable,
the same seven questions and complete answers are shown as a static sequence
without dead scroll space or hidden evidence.

The full 36-entry comparison atlas and evidence table stay visible without a
disclosure click, but use compact keyboard-scrollable viewports so they do not
dominate the article. On narrow screens, each evidence entry becomes a labelled
two-column record.

The release uses thin rules and whitespace instead of repeated rounded containers.
Borders remain only where they communicate an interactive control, code sample, or
image boundary.

The package visibly states that it is not diagnostic, preserves the
`amended_or_exploratory` disposition, includes every saved primary comparison,
and identifies only a “potentially inconsistent annotation” or an item
“recommended for expert review”. Confirmatory and external validation remain
explicit future work. See `MVP_SCOPE.md` for the reduced acceptance boundary.
