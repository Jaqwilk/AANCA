# AANCA decisions

This file records current binding decisions. Detailed historical deliberations remain
available in Git history and in tag `pre-audit-simplification-2026-08-20`.

## D001 — Non-diagnostic scope

Status: accepted

AANCA ranks potentially inconsistent annotations for expert review. It never changes
source annotations automatically and never treats model disagreement as proof that a
pathologist or biological reference is wrong.

## D002 — Group-safe partitions

Status: accepted

Every scientific split uses `group_id` at least at source-patch level. Nuclei from one
source group may not cross fitting, validation or held-out partitions. The final
reference fold remains untouched and unavailable for model or method selection.

## D003 — Out-of-fold primary scores

Status: accepted

Primary model-based audit scores use group-safe OOF probabilities. A score produced by
a model fitted on the scored nucleus or its source group is ineligible.

## D004 — Immutable label states

Status: accepted

`pre_corruption_label`, `observed_label`, `is_injected_corruption` and corruption
metadata remain separate in storage and APIs. Restoration creates derived arrays and
does not mutate either source label state.

## D005 — Instance-dependent independence

Status: accepted

Instance-dependent corruption and its evaluated auditor must use independently bound
feature spaces. Any overlap is labelled `circularity_risk` and excluded from
confirmatory interpretation.

## D006 — Accepted analysis disposition

Status: accepted

The July freeze lacks an independent public timestamp and outcomes were encountered
during technical recovery. The accepted primary analysis is therefore permanently
described as `amended_or_exploratory`, not untouched confirmatory evidence.

## D007 — Adverse H4 remains prominent

Status: accepted

The saved H4 result is negative. Audit-guided restoration did not outperform random
review on final-fold macro-F1. It must be shown before favourable ranking results and
must not be reframed as downstream improvement.

## D008 — Public evidence release

Status: implemented

Publish the retained statistical, restoration, ranking and OOF evidence as immutable
GitHub Release assets, anchored by repository and per-cell manifests. Do not publish
raw PanNuke images or masks. State that checkpoints were not retained rather than
inventing or regenerating them after outcome inspection.

Release: `primary-evidence-v1`.

## D009 — Independent statistics verifier

Status: implemented

Maintain a small verifier that does not import the AANCA package. It must check fixed
file identities and independently recalculate all available primary comparisons,
Holm corrections and H4 from the released NumPy arrays.

## D010 — Remove unexecuted governance ceremony

Status: implemented

The original capsule, technical authority, resource-controller and resource-bounded
runner stack was not part of the completed primary evidence and caused most Linux CI
failures. Remove it from active source, CLI and tests. Preserve recoverability in Git
tag `pre-audit-simplification-2026-08-20`.

Retain the scientific contracts: group-safe OOF, frozen inputs, immutable labels,
filesystem readback, primary statistics, restoration, QC and external review-package
validation.

## D011 — Cross-platform CI

Status: accepted

The complete maintained test suite, lint, formatting and synthetic smoke run on both
Ubuntu and Windows. Windows-native handle/WOF checks must be explicitly scoped or
removed with their retired feature; local workstation paths and PanNuke files may not
be required during CI collection.

## D012 — Public CLI scope

Status: accepted

Expose commands that are maintained and can provide a truthful outcome: doctor,
synthetic/data preparation, representations, smoke, pilot, primary, preregistration
freeze, original-label audit, external review packaging, reporting and demo serving.

Do not expose the retired direct confirmatory capsule, lifecycle rehearsal,
resource-bounded sensitivity, historical amendment publication or orphan-recovery
orchestration as if they were supported public workflows.

## D013 — Article structure

Status: accepted

Use a continuous research-article flow: scope and thesis, method, reading guidance,
H4, H1-H7, detailed evidence, QC and limitations, reproducibility, author. Keep the
“What the study actually learned” animation while respecting reduced motion and
maintaining one consistent article typography system.

## D014 — Research limits remain open

Status: accepted

No code change may mark natural-error review, pathologist validation, patient/WSI
independence, clinical utility or external validation complete. These require new,
prospectively gathered evidence and qualified external participants.

## D015 — Repeatable synthetic quick-start

Status: implemented

The deterministic data command may reuse an existing output only after independently
regenerating the expected dataset and checking the complete file set, every array,
the manifest and generation evidence. It must fail without writing when any saved
content differs. This makes the documented quick-start repeatable without weakening
the no-overwrite rule.
