# AANCA presentation MVP

The static MVP is generated at `artifacts/mvp_demo/index.html` from the accepted,
sealed primary and the checksum-bound PanNuke QC bundle. It needs no server and
does not execute a model.

```powershell
.venv\Scripts\python.exe -m histo_audit demo verify --output-dir artifacts\mvp_demo
```

The package visibly states that it is not diagnostic, preserves the
`amended_or_exploratory` disposition, includes every saved primary comparison,
and identifies only a “potentially inconsistent annotation” or an item
“recommended for expert review”. Confirmatory and external validation remain
explicit future work. See `MVP_SCOPE.md` for the reduced acceptance boundary.
