# NuCLS supervised-QC pairing feasibility

**Decision:** paired natural pre/post class-label evaluation is unavailable.

## What the official release contains

The official raw SQLite database contains `3995` FOV rows, `131747` annotation-element rows and `1745` preapproved/QC FOVs. The official public page describes 2,168 uncorrected FOVs and 1,744 corrected FOVs. These are quality tiers of different FOVs, not two releases of identical nucleus instances.

## Why the prospective pairing cannot run

The database has one class field (`group`) per stable annotation element and no previous-label, replacement-label, revision-history or paired-state table. Repeated element IDs only arise where the same geometry intersects multiple FOV records; no stable element ID carries two distinct class labels.
There are `438` correction-prefixed final annotations, but their former labels are not retained. Treating the prefix as an error outcome would expose the final QC state to the auditor and would be circular.

## Consequence

AANCA cannot be honestly tested here on annotations later changed during QC. Comparing corrected and uncorrected cohorts would confound annotation quality with different images, patients and class distributions. The pre-registered natural-QC action therefore remains `retain_uncorrected`; no pathologist-error claim is available.
