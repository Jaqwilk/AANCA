# Future geometric-annotation audit protocol

## Status and separation from the current study

This document specifies future interfaces; it reports no executed experiment or result. The current confirmatory question remains class-label consistency for an already segmented, exactly indicated nucleus. Missing-nucleus detection, merged/split-instance detection, and contour-quality auditing are out of scope for that question and must not block the frozen class-label benchmark.

The geometric track must use separate configurations, preregistration amendments, predictions, rankings, metrics, and review fields. A geometric signal must not be presented as proof that a source annotation is wrong and must never automatically modify a source mask. Its output language is “potentially inconsistent geometric annotation” and “recommended for expert review.”

## Shared immutable data contract

Every candidate record should preserve:

- `geometric_candidate_id`, dataset and immutable source-array checksum;
- verified grouping fields, with `group_id` at least the source patch and stronger WSI/patient grouping only when supported;
- official fold, source patch identity, tissue metadata, and pixel coordinate convention;
- source instance IDs and source contours exactly as released;
- proposal-generator name, version, representation, configuration hash, seed, and checkpoint checksum;
- candidate type (`possible_missing`, `possible_merge`, `possible_split`, or `possible_contour_issue`);
- group-safe out-of-fold fold assignment and evidence provenance;
- genuine reviewer responses, kept per reviewer without synthetic responses or premature consensus.

Candidate generation and candidate ranking are distinct. When the same representation generates and evaluates a proposal, the cell is marked `circularity_risk` and excluded from confirmatory claims unless an independent validation design is established. The final reference fold remains unavailable for proposal threshold selection, ranking-method selection, and stopping decisions.

## Interface 1 — possible missing nuclei

### Input and proposal interface

`propose_missing_nuclei(image, source_instance_mask, context) -> candidate regions`

A candidate region must not overlap an existing source instance beyond a frozen tolerance. It records a contour or box, centroid, proposal confidence, nearest source-instance distance, border and artefact flags, and the generator provenance. Proposal confidence is not a class-label probability and cannot establish that a nucleus is truly missing.

### Independent audit interface

`score_missing_candidates(image, source_instance_mask, candidates, reference_context) -> review_risk`

Larger risk means stronger recommendation for expert review. A confirmatory evaluator should use independently generated candidates or an independent representation. Negative controls should include non-nuclear tissue structures and intentionally withheld source instances produced only within development groups.

### Review and evaluation

Blinded response options should include `missing_nucleus_supported`, `not_a_missing_nucleus`, `ambiguous`, `insufficient_context`, and `exclude_technical_reason`. Controlled development metrics may use deliberately withheld instances, but those measure the withholding process rather than naturally missing annotations. Genuine expert validation should report precision and recall at fixed proposal/review budgets, stratified counts, reviewer agreement, and localisation overlap only when a defensible reference contour exists.

## Interface 2 — possible merged instances

### Input and proposal interface

`propose_merged_instances(image, source_instance_mask, instance_id) -> split hypotheses`

Each hypothesis preserves the original contour and supplies two or more non-overlapping child contours whose union is compared with the original. Records include child count, union/intersection residuals, watershed or model evidence, morphology, internal intensity boundaries, neighbouring-instance context, and provenance. The interface does not replace the source instance.

### Independent audit interface

`score_merge_hypotheses(image, original_contour, child_hypotheses, reference_context) -> review_risk`

Controls should include synthetically merged nearby instances in development data, with source groups kept intact. Synthetic adjacency rules and merge morphology must be frozen and reported because easy artificial merges can overstate performance. Confirmatory evaluation requires an independent generator/auditor representation or explicit circularity exclusion.

### Review and evaluation

Blinded options should include `single_instance_supported`, `probable_merge`, `ambiguous`, `insufficient_context`, and `exclude_technical_reason`. Report ranking AP for controlled merges, precision at fixed expert-review budgets, child-count error, and genuine review rates separately. Do not infer that a probable merge implies a class-label error.

## Interface 3 — possible split instances

### Input and proposal interface

`propose_split_instances(image, source_instance_mask, instance_ids) -> merge hypotheses`

A hypothesis links two or more source instances and supplies an immutable proposed union. It records pair or set membership, contour contact/gap distance, colour and texture continuity, combined morphology, proposal confidence, border/artefact flags, and provenance. Candidate enumeration must be spatially bounded so that the number of negative pairs and compute cost are defined before outcome inspection.

### Independent audit interface

`score_split_hypotheses(image, source_contours, proposed_union, reference_context) -> review_risk`

Development controls may deliberately partition a source instance, while preserving the original contour only in hidden controlled-evaluation fields. The partition algorithm, minimum child size, and separation morphology are part of the corruption metadata. Artificial partitions are not assumed to represent natural over-segmentation.

### Review and evaluation

Blinded options should include `separate_instances_supported`, `probable_split_annotation`, `ambiguous`, `insufficient_context`, and `exclude_technical_reason`. Report controlled ranking metrics, fixed-budget yield, pair/set-level localisation, and genuine expert review separately. Prevent duplicate credit when several hypotheses describe the same source-instance set.

## Interface 4 — contour quality

### Input and perturbation interface

`generate_contour_perturbations(source_contour, image, mechanism, severity, seed) -> controlled contours`

Development-only mechanisms may include bounded erosion, dilation, local boundary displacement, vertex simplification, holes, disconnected fragments, and shifts. Perturbations must remain identifiable and preserve `pre_perturbation_contour`, `observed_contour`, mechanism, severity, seed, and configuration hash as separate fields. The untouched final reference fold is never perturbed.

### Audit interface

`score_contour_quality(image, observed_contour, neighbouring_contours, reference_context) -> review_risk`

Potential evidence includes boundary image gradients, target/interior/exterior appearance, shape plausibility, overlap conflicts, border truncation, and representation disagreement. Risk is a ranking signal only. Class labels and any hidden pre-perturbation contour are unavailable to the model.

### Review and evaluation

Blinded options should include `contour_supported`, `probable_under_segmentation`, `probable_over_segmentation`, `probable_displacement`, `ambiguous`, `insufficient_context`, and `exclude_technical_reason`. Controlled metrics may include boundary F-score, Hausdorff distance with an explicitly robust percentile, IoU, and ranking AP by perturbation mechanism/severity. Genuine expert-reviewed contour quality remains a separate validation endpoint.

## Leakage, sampling, and statistics

- Split and bootstrap by verified `group_id`, never by individual instance, contour, pair, or proposal.
- Generate all controlled geometric changes only after the outer development/final split.
- Use group-safe out-of-fold evidence for primary proposal ranking.
- Keep every proposal derived from one source patch in the same fold, including overlapping and duplicate hypotheses.
- Select thresholds, proposal caps, perturbation severity, and stopping rules using development/reference-validation groups only.
- Compare methods at identical expert-review counts. Deduplicate hypotheses under a frozen matching rule before credit assignment.
- Use paired group bootstrap for method differences and retain zero-event cells as not applicable rather than inventing ranking metrics.
- Report candidate-generation recall separately from ranking quality; an auditor cannot recover a missed proposal.

## Staged gates

1. **Interface smoke:** deterministic synthetic patches exercise each proposal type, serialization, exact target display, group-safe splitting, and no source-mask writes.
2. **Controlled pilot:** one frozen mechanism per task estimates runtime and failure modes without inspecting final-fold outcomes.
3. **Protocol freeze:** candidate universe, matching rules, metrics, budgets, seeds, independent feature spaces, and exclusions receive a dated preregistration amendment.
4. **Controlled benchmark:** repeated mechanisms and severities, with candidate-generation and ranking outcomes reported separately.
5. **Blinded expert review:** top-ranked and disjoint random controls are anonymised and deterministically mixed; model suggestions and cohort source remain hidden.
6. **External/multi-rater validation:** proceed only after class definitions, geometric annotation conventions, domain shift, and licence/access are verified.

Failure at any geometric gate does not change the status of the independent class-label audit. The class-label work can advance through its own mandatory gates while this protocol remains future work.

## Required implementation tests before activation

- deterministic candidate IDs and geometry;
- exact preservation of source masks and contours;
- no final-fold perturbation or tuning access;
- no cross-group proposal leakage;
- generator/auditor independence metadata and `circularity_risk` exclusions;
- fixed candidate and review budgets;
- duplicate-hypothesis matching without double credit;
- reviewer package blinding and blank responses;
- explicit empty/degenerate-case behaviour;
- machine-readable provenance for every reported value.

No geometric experiment, expert review, or external validation is claimed complete by this protocol.
