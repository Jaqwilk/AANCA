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

The package visibly states that it is not diagnostic, preserves the
`amended_or_exploratory` disposition, includes every saved primary comparison,
and identifies only a “potentially inconsistent annotation” or an item
“recommended for expert review”. Confirmatory and external validation remain
explicit future work. See `MVP_SCOPE.md` for the reduced acceptance boundary.
