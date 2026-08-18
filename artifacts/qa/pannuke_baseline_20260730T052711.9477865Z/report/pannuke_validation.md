# PanNuke local-release validation

**Gate status:** valid for manifest construction after a full streaming semantic scan.

This report inventories the local release and recommends potentially inconsistent annotations for later expert review; it does not modify source annotations.

## Semantic provenance

- Positive channel order: `neoplastic, inflammatory, connective_soft_tissue, dead, non_neoplastic_epithelial`
- Mapping source: https://github.com/TissueImageAnalytics/PanNuke-metrics/tree/c00014d766ca1be142b81bea19d9ef4315cde65a
- Mapping source revision: `c00014d766ca1be142b81bea19d9ef4315cde65a`
- Verification note: Default positive-channel order documented by the official PanNuke metrics repository README at pinned archived commit c00014d766ca1be142b81bea19d9ef4315cde65a; re-verify this evidence before a frozen study.

## Fold arrays

| Fold | Patches | Image shape / dtype / full range | Mask shape / dtype / full range | Supplied background | Overlap pixels / patches | Void pixels / patches | Positive+background pixels / patches | Overlap-touching instances | Disconnected IDs / patches | Tissues |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2656 | `(2656, 256, 256, 3)` / `<f8` / `(0.0, 255.0)` | `(2656, 256, 256, 6)` / `<f8` / `(0.0, 3512.0)` | 5 | 1216 / 194 | 2359296 / 36 | 0 / 0 | 471 | 67 / 65 | 19 |
| 2 | 2523 | `(2523, 256, 256, 3)` / `<f8` / `(0.0, 255.0)` | `(2523, 256, 256, 6)` / `<f8` / `(0.0, 3515.0)` | 5 | 1572 / 190 | 3801371 / 59 | 0 / 0 | 465 | 63 / 59 | 19 |
| 3 | 2722 | `(2722, 256, 256, 3)` / `<f8` / `(0.0, 255.0)` | `(2722, 256, 256, 6)` / `<f8` / `(0.0, 3517.0)` | 5 | 1530 / 191 | 4325424 / 67 | 0 / 0 | 475 | 81 / 77 | 19 |

## Raw provenance

- Hashed raw files (SHA-256): 22
- Semantic scan scope: every patch in every resolved fold
- Released fold IDs discovered: 1, 2, 3
- Expected official fold IDs: 1, 2, 3
- Complete release inventory: true
- Archives retained and never auto-extracted/deleted: 3

## Fixed anomaly-safe mask policy

- The supplied background channel is recorded as supplied; it is not required to be the exact complement of positive-class occupancy.
- Pixels with neither positive nor supplied-background occupancy remain unlabeled (`void`).
- Cross-class-overlap pixels retain every raw channel/instance identity; never arbitrate a class or repair the raw mask.
- Instances touching cross-class overlap are flagged with the shared primary/confirmatory analysis exclusion reason `touches_cross_class_overlap`.
- Raw instance IDs occupying multiple 4-connected components are retained as one raw identity, counted, and flagged; they are never split or repaired. Their primary/confirmatory eligibility must be frozen after the pilot without final-reference outcomes.
- Invalid array shapes, non-finite values, negative IDs, and non-integer-like IDs remain fatal structural errors.
- Release totals: 4318 cross-class-overlap pixels, 10486091 void pixels, 0 positive+background pixels, and 1411 overlap-touching instances.

## Independence limitation

Separation was performed at source-patch level. Patient- and WSI-level independence could not be guaranteed from the released metadata.
Source patch is therefore the mandatory `group_id`; stronger patient/WSI separation must not be claimed without separately verified metadata.
