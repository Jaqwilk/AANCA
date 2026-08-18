# Scientific Specification

## Research question and contribution

Can a source-group-safe automated auditing system rank intentionally corrupted nucleus class labels more efficiently than random expert review, and does restoration of the highest-ranked injected corruptions improve downstream nucleus classification?

The contribution is a controlled, comparative, reproducible annotation-auditing workflow for nucleus-level histopathology labels. It combines group-aware out-of-fold (OOF) evidence, complementary risk methods, controlled corruption, fixed review budgets, restoration experiments, independent representation spaces, statistical comparisons, and expert-review-ready explanations. Existing label-error and noisy-label methods are prior art; novelty is not claimed for them.

## Hypotheses

- **H1:** Group-safe OOF methods identify injected corruptions more efficiently than random review.
- **H2:** performance depends on corruption mechanism, nucleus class, and tissue type.
- **H3:** confusion-targeted and instance-dependent corruption are harder than symmetric corruption.
- **H4:** at equal review budget, `audit_guided_restoration` improves downstream macro F1 more than `random_review_restoration`.
- **H5:** a fixed, equal-weight hybrid may improve average ranking performance; this is tested, not assumed.
- **H6:** pathology-specific/self-supervised representations may outperform ImageNet representations; this requires evidence.
- **H7:** explicit target indication may reduce neighbouring-nucleus shortcut risk.

## Scope and terminology

The primary task audits class-label consistency for already segmented nucleus instances. It excludes diagnosis, prognosis, treatment decisions, clinical deployment, automatic source-label modification, missing-nucleus detection, merged/split detection, and contour-quality auditing.

- `pre_corruption_label`: source annotation before intentional project corruption; a quality-controlled reference, not guaranteed biological truth.
- `observed_label`: label exposed to training after optional corruption.
- `is_injected_corruption`: whether this project intentionally changed the source label.
- `restored_label`: controlled-experiment label restored from `observed_label` to `pre_corruption_label` only after simulated review.
- Required experiment names: `uncorrupted_reference_baseline`, `corrupted_observed_baseline`, `random_review_restoration`, `audit_guided_restoration`.

Labels must also retain `corruption_type`, `original_class`, `replacement_class`, `corruption_seed`, generator representation, and configuration hash. The model sees only `observed_label`; hidden reference labels are restricted to controlled evaluation, simulated restoration, reference-label benchmarks, and untouched final evaluation.

## Dataset assumptions and validation

PanNuke is the planned primary dataset, subject to verification from the original publications, official Warwick/Tissue Image Analytics resources, official metrics repository, licence, and downloaded arrays. Exact counts, dimensions, background representation, class order, patient/WSI identifiers, and official-fold independence are not assumed. Raw file hashes and a data manifest are required.

At minimum the inspector must detect folds, images, masks, tissue metadata, shapes, dtypes, ranges, channels/background, instance IDs, positive-class order, malformed items, and representative overlays. Exact and near-duplicate auditing must precede primary split freeze. Near duplicates are flagged, not automatically removed.

Each nucleus manifest includes identity, source paths/index, official fold, patch/group, verified patient/WSI identifiers if available, tissue, class, instance, box, centroid, morphology, border/crop/quality flags, and immutable label/corruption fields. Patient- and WSI-level independence is claimed only if metadata supports it; otherwise reports state that separation is at source-patch level and stronger independence cannot be guaranteed.

## Target representations

Required representations are nucleus-centred context RGB, three-channel target-highlighted RGB with context retained and no class information encoded, and target-specific morphometrics. A compatible confirmatory CNN may use RGB plus a fourth binary target-mask channel. Reports show the full patch, crop, exact contour, nearby nuclei, and target instance ID.

## Split and leakage rules

Official folds form the outer structure. One official fold is an untouched, uncorrupted final reference test; two are development data. Approximately 10% of development patch groups form an uncorrupted reference-validation set; the remaining groups form the audit pool. Corruption occurs only in the audit pool.

Audit OOF predictions use `StratifiedGroupKFold` where feasible, otherwise documented `GroupKFold`. Every audit sample receives exactly one probability vector from a model whose training data excludes that sample and its entire source group. Class ordering is fixed, probabilities sum to one, final-test samples are absent, and train/holdout group identifiers are saved. No nucleus-level random fallback is allowed. Reference validation can support calibration, early stopping, predefined model selection, and pilot debugging; the final test cannot.

## Controlled corruption

Rates are 0%, 5%, 10%, and 20%, with exact counts from a documented rounding rule and no self-replacement.

- `symmetric_random_corruption`: uniform alternative class.
- `confusion_targeted_corruption`: configurable transition matrix, without unverified clinical-realism claims.
- group-conditional corruption: configurable group/tissue rates.
- instance-dependent corruption: hard cases selected in an independently chosen representation, with plausible neighbour alternatives.

Generator/auditor independence is recorded in a matrix. Identical or unverified feature spaces are marked `circularity_risk`, reported separately, and excluded from the primary confirmatory comparison.

## Representations, classifiers, and OOF evidence

Representation families are engineered morphology/colour/HOG/texture features; frozen official ImageNet ResNet-18 (or verified stable equivalent) embeddings; and the first legally accessible, reproducible, hardware-fitting pathology encoder in a frozen priority rule. Unavailable pathology encoders are documented and do not block the ImageNet benchmark.

Primary probabilistic classifiers are multinomial logistic regression on each feature family and a small MLP on frozen embeddings. Only observed development labels determine fitting and class weights. Calibration, if used, is fitted only on reference validation and reported separately. Confirmatory CNN work begins only after the frozen-feature benchmark and uses CUDA/AMP/checkpoint/resume/OOM/early-stopping controls without test tuning.

## Annotation-risk methods

All risk interfaces return larger values for more suspicious annotations. Required methods are self-confidence (`1-P(observed_label)`), clipped negative log-likelihood, alternative-minus-observed prediction margin, predictive entropy as an ambiguity baseline, current stable Cleanlab on OOF probabilities, fold-safe weighted nearest-neighbour disagreement, selected-scenario ensemble disagreement, and a preregistered equal-weight hybrid of percentile-normalised complementary components. Neighbours exclude the sample and all members of its group and come only from valid reference groups. Cleanlab failures remain missing with a recorded blocker; values are never fabricated.

## Metrics and statistics

The controlled positive event is `is_injected_corruption`. Primary ranking metric is average precision/AUPRC; primary review budget is 5%; secondary budgets are 1%, 10%, and 20%. Report AUROC, precision/recall/lift at budget, class/tissue/mechanism/rate AP, reviewed counts, corruptions found, and random expectation. At 0% corruption AP is `not_applicable`; report score distributions/false alerts instead.

Random review uses identical budgets, at least 100 deterministic repetitions when cheap, retained seeds, mean and interval. Paired bootstrap resamples source groups identically for method comparisons, uses at least 2,000 iterations in the primary study unless explicitly infeasible, and reports difference, 95% interval, direction, and probability of positive improvement. Exploratory multiple comparisons use a documented correction such as Holm. Reliable subgroup AP requires at least 100 samples and 10 injected corruptions; otherwise only counts are shown.

## Restoration and downstream utility

Only reviewed injected corruptions are restored to `pre_corruption_label`; unreviewed observations remain unchanged. The four required experiments use identical model, representation, hyperparameters, seeds, review budget, and untouched final reference test. Report accuracy, macro F1 (primary downstream metric), balanced accuracy, per-class precision/recall, confusion matrices, and appropriate calibration metrics. Neutral or adverse effects are retained.

## External-validation distinction

Controlled injected-corruption performance measures the injected process, not naturally occurring errors. Reports separately label the controlled benchmark, exploratory original-label ranking, expert-reviewed validation, and external multi-rater validation. Original-data flags are “low estimated label quality for potential expert review,” never confirmed errors. No expert responses may be simulated.

## Experiment gates

1. Deterministic CPU synthetic smoke: generation, crops/highlighting, corruption, group OOF, audit, metrics, restoration, report, and tests.
2. Real PanNuke pilot when verified data are available: all classes, one outer split, 10% corruption, one seed, ImageNet/logistic, self-confidence/Cleanlab/fold-safe neighbours, restoration, report.
3. Freeze preregistration after pilot and before full primary outcomes.
4. Preregistered frozen-feature primary benchmark.
5. Predefined confirmatory CNN/encoder/target/hybrid/downstream scenarios.
6. Official-fold rotation when feasible.
7. Exploratory original-label audit.
8. Blinded expert-review or responsible multi-rater external-validation package.

## Completion stages

- `INITIALISED`: environment inspected and repository created.
- `PIPELINE_COMPLETE`: deterministic synthetic end-to-end pipeline, report, and tests pass.
- `PILOT_COMPLETE`: verified real PanNuke pilot reaches its report.
- `PRE_REGISTRATION_FROZEN`: final primary definition is cryptographically frozen after the pilot and before primary outcome inspection.
- `PRIMARY_STUDY_COMPLETE`: preregistered frozen-feature benchmark completes.
- `CONFIRMATORY_COMPLETE`: selected CNN/encoder comparisons, ablations, and fold rotations complete as defined.
- `EXTERNAL_VALIDATION_READY`: blinded review or external evaluation package generated.
- `EXTERNAL_VALIDATION_COMPLETE`: genuine expert-reviewed or multi-rater evaluation completed.
- `DEMO_COMPLETE`: optional non-diagnostic presentation demo completed after the primary study.

Synthetic success alone is never described as completion of the entire research project. Every result must originate from saved machine-readable artifacts; omitted or failed experiments and pilot reductions are explicit.
