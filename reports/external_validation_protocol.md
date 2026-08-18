# External Validation Protocol

**Status:** protocol implemented/tested with software fixtures; no genuine expert responses or external multi-rater outcomes exist.

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

NuCLS is a candidate rather than an assumed drop-in benchmark. Before analysis, verify its licence, source separation, rater structure, label definitions, and any PanNuke mapping. Do not force incompatible categories. Preserve individual-rater observations and report domain shift (breast-only NuCLS versus multi-tissue PanNuke) separately.

Possible prespecified questions are whether lower estimated label quality correlates with rater disagreement and whether top-ranked samples contain more disagreement than random samples. Rater consensus is an external signal, not guaranteed biological truth.

## Completion language

Generating a package permits `EXTERNAL_VALIDATION_READY`; only genuine completed expert or responsible external multi-rater evaluation permits `EXTERNAL_VALIDATION_COMPLETE`. Fixture tests and synthetic labels confer neither status.
