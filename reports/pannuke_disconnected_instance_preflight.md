# PanNuke disconnected-instance preflight

Date: 2026-07-18 (Europe/Warsaw)

This is read-only preflight evidence for the open M5 semantic-validation gate. It
does not modify or adjudicate any source annotation and is not a study outcome.

## Reason for the check

The current validator checks whether every raw `(fold, patch, class, instance_id)`
occupies one connected component. The official local PanNuke release contains raw
instance IDs with multiple components, so treating every such occurrence as fatal
would stop M5 after the already documented overlap/void policy is accepted.

## Full-release result

All positive channels in all 7,901 source patches were inspected from memory-mapped
raw NPY arrays. Connectivity was evaluated independently within each raw class
channel and instance ID.

| Fold | Disconnected with 4-connectivity | Affected patches (4) | Still disconnected with 8-connectivity | Affected patches (8) | Diagonal-only cases |
|---:|---:|---:|---:|---:|---:|
| 1 | 67 | 65 | 38 | 37 | 29 |
| 2 | 63 | 59 | 38 | 36 | 25 |
| 3 | 81 | 77 | 43 | 43 | 38 |
| **Total** | **211** | **201** | **119** | **116** | **92** |

The full scan completed in 26.158 seconds. The 119 eight-connectivity cases show
that the issue cannot be removed merely by changing diagonal-connectivity
convention.

## Gate implication

- Archive identities, CRC/path safety, shapes, dtypes, finite values, and integer
  instance IDs remain separate structural checks.
- A disconnected raw instance ID should not be silently split, merged, repaired,
  or assigned a different class.
- Before the canonical validator runs, the project needs an explicit pre-freeze
  policy that reports these instances and does not mistake a release-level semantic
  annotation property for download corruption.
- The downstream eligibility decision must be fixed without consulting final-fold
  outcomes. `PRE_REGISTRATION.md` already leaves malformed-instance handling for a
  pre-freeze decision after the pilot; therefore M5 can retain and flag these rows,
  while the primary/confirmatory eligibility decision remains fail-closed until it
  is frozen.

This preflight does not itself close M5 and must be reconciled against canonical
validator and nucleus-manifest artifacts.
