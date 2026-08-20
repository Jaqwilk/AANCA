# External Validation Protocol

**Status:** the blinded PanNuke expert-review path remains protocol-only. A separate
prospectively frozen NuCLS external multi-rater evaluation has now been completed and
published with a null/adverse result.

## Blinded expert-review path

When original PanNuke OOF rankings and authorised image assets are available, build a package with exactly 100 top-ranked and 100 random, non-overlapping annotations unless a dated protocol amendment changes the counts. Mix them with a retained deterministic seed, anonymise review identifiers, and hide selection source, rank, model suggestion, risk score, corruption/reference fields, and any pre-corruption label during initial review.

Every item should show the full source patch, nucleus-centred crop, and exact target contour without encoding class information. Initial response options are:

- annotation supported;
- probably inconsistent;
- ambiguous;
- insufficient context;
- exclude for technical reason.

Preserve each reviewer’s response separately. Do not invent missing responses, collapse disagreement prematurely, or reveal the ranking source before the blinded phase ends. Compare the prespecified combined “probably inconsistent or ambiguous” rate between top-ranked and random sets, with source-group-aware uncertainty where sample counts permit.

## External multi-rater path

NuCLS was evaluated as a separate external study rather than treated as a PanNuke
drop-in benchmark. Its licence, source separation, rater structure and official
three-superclass mapping were frozen before outcome download. Exact official
NP-label/P-truth anchors were paired, and complete TCGA patients were held together.
The endpoint was disagreement with inferred multi-rater pathologist consensus, not
biological truth.

The primary question was whether frozen group-safe risk prioritised NP/P disagreement
above random review at both overall AP and the operational 5% budget. The secondary
question was whether guided correction improved group-held-out macro F1. Neither
combined success rule passed. See
[`nucls_external_validation_results.md`](nucls_external_validation_results.md).

## Completion language

Generating a package permits `EXTERNAL_VALIDATION_READY`; only genuine completed expert or responsible external multi-rater evaluation permits `EXTERNAL_VALIDATION_COMPLETE`. Fixture tests and synthetic labels confer neither status.

The NuCLS execution satisfies the completion definition because it is genuine
multi-rater evidence with a locked analysis and published artifacts. Completion does
not imply a positive result, pathologist error, clinical validity or permission to
change source labels.
