# Prospective PUMA new-data confirmation protocol

**Freeze date:** 2026-08-21 (Europe/Warsaw)  
**State at freeze:** official record metadata and paper inspected; image and nuclei
archives not downloaded or opened  
**Project:** the existing AANCA system; this is not a replacement project or V2

## Purpose

PUMA is a genuinely new source for this project. This study tests whether the frozen
AANCA candidate transfers to expert-reviewed melanoma images and improves a
downstream classifier after controlled label corruption. It is a compatibility and
controlled-noise confirmation. Because the PUMA release provides final expert-
checked labels rather than paired natural pre/post review labels, it cannot establish
natural pathologist-error detection.

## Frozen data authority

- official Zenodo record: <https://zenodo.org/records/15050523>
- official challenge description: <https://puma.grand-challenge.org/dataset/>
- dataset paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/>
- licence: CC0
- ROI archive: `01_training_dataset_tif_ROIs.zip`, expected MD5
  `1e16d440f0f94156fd5c0ea3f082bd90`
- nuclei archive: `01_training_dataset_geojson_nuclei.zip`, expected MD5
  `1f695c66db9461251c581e850f73c044`

The 14 GB context archive is not required. Hash every downloaded archive before
parsing. Use only the official Zenodo content endpoints.

## Frozen sample split and labels

The public set contains 103 primary and 103 metastatic ROIs, one ROI per case. Treat
the complete official ROI identifier as the case group. Within each source stratum,
sort cases by SHA-256 of `AANCA-PUMA-FINAL-V1|<case_id>` and reserve the first 31 as
the untouched final set. The remaining 72 per stratum form the audit/training set.
No final image, label or derived feature may affect model, threshold, budget,
mapping or candidate selection.

Use the official PUMA three-class benchmark mapping as primary:

- `tumor` = tumor;
- `lymphocyte` = lymphocyte or plasma cell;
- `other` = stroma, vascular endothelium, histiocyte, melanophage, neutrophil,
  apoptotic or epithelium.

The native ten-class result may be reported only as predeclared secondary evidence;
it cannot replace a failed primary endpoint.

## Frozen intervention and evaluation

Transfer candidate hash
`78547a73ef239dab11aee66e8b9b787e84508b82f6ace7bb81dc725f38803ffe`
without PUMA tuning:

- ResNet-18 64+128 pixel multiscale features;
- audit logistic L2 `0.1`, unbalanced;
- fixed hybrid risk with self-confidence weight `0.6`, group-safe k=`31`;
- `balanced_relaxed` queue, budget `5%`;
- intervention `flag_exclude`;
- balanced downstream logistic L2 `0.01`.

On the 144 audit/training cases only, inject deterministic 10% symmetric class
corruption with seeds `26082170`, `26082171`, `26082172`, `26082173`. Permanently
separate final label, observed corrupted label and injection metadata. Produce all
audit scores out of fold with four case-group folds. The corruption generator and
auditor use independent spaces as required by `SPEC.md`.

For each seed compare the AANCA exclusion with the unchanged corrupted training set
and five exact matched-random exclusions at the same budget. Fit all downstream arms
identically and evaluate once on the 62 untouched, uncorrupted final cases.

## Frozen success gates

Use 3,000 whole-case bootstrap replicates. Success requires all of:

- AANCA minus matched-random corruption-retrieval precision lower bound > 0;
- downstream macro-F1 lower bound > unchanged corrupted training;
- downstream macro-F1 lower bound > mean exact matched-random exclusion;
- all four corruption-seed directions versus both controls are positive;
- every primary-class recall lower bound versus unchanged is >= `-0.01`;
- all fits converge and all group, hash, split and final-fold guards pass.

A positive result supports transfer of controlled annotation-noise improvement to
PUMA. It does not support natural-error, clinical, diagnostic or biological-truth
claims, and never permits automatic modification of source annotations.

