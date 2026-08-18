# PanNuke mask QC

**QC status:** release-level annotation anomalies recorded; structural gate valid.

This read-only report identifies potentially inconsistent annotations and patches recommended for expert review. It does not adjudicate a class, modify a source mask, or treat model disagreement as proof of annotation error.

## Fixed interpretation

- Positive channels are evaluated independently from the supplied background channel.
- Pixels with neither a positive assignment nor supplied background remain unlabeled/void.
- Cross-class overlap is retained and rendered in one neutral colour; no class arbitration is performed.
- Shared primary/confirmatory overlap-touching exclusion reason: `touches_cross_class_overlap`.
- The same eligibility mask applies to primary and confirmatory analyses.
- Disconnected raw instance IDs are retained and quality-flagged without splitting or repair; their primary/confirmatory eligibility is frozen only after the pilot and without final-reference outcomes.
- Raw source masks modified: `false`.

## Release totals

- Folds / source patches / pixels: 3 / 7,901 / 517,799,936
- Cross-class-overlap pixels / patches: 4,318 / 575
- Unlabeled/void pixels / patches: 10,486,091 / 162
- Positive-and-supplied-background pixels / patches: 0 / 0
- Overlap-touching instances excluded from primary and confirmatory analyses: 1,411
- Disconnected raw instance IDs / affected patches: 211 / 201

## Fold reconciliation

| Fold | Patches | Overlap pixels | Overlap patches | Void pixels | Void patches | Overlap-touching instances | Disconnected IDs | Disconnected patches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2656 | 1216 | 194 | 2359296 | 36 | 471 | 67 | 65 |
| 2 | 2523 | 1572 | 190 | 3801371 | 59 | 465 | 63 | 59 |
| 3 | 2722 | 1530 | 191 | 4325424 | 67 | 475 | 81 | 77 |

## Deterministic anomaly overlay

- Selection strategy: `fixed category round-robin (cross-class overlap, positive+background, void, normal), interleaved by fold then ascending source patch index`
- Selected unique source patches: 24
- Selection SHA-256: `09886588591d9ebb9a725db1022bb0ab8fb94b4bcca419b486e2549b0cc5fd36`
- Patch IDs: `fold_1:patch_81`, `fold_1:patch_1614`, `fold_1:patch_0`, `fold_2:patch_96`, `fold_2:patch_1434`, `fold_2:patch_0`, `fold_3:patch_77`, `fold_3:patch_1356`, `fold_3:patch_0`, `fold_1:patch_86`, `fold_1:patch_1615`, `fold_1:patch_1`, `fold_2:patch_113`, `fold_2:patch_1435`, `fold_2:patch_1`, `fold_3:patch_103`, `fold_3:patch_1506`, `fold_3:patch_1`, `fold_1:patch_119`, `fold_1:patch_1616`, `fold_1:patch_2`, `fold_2:patch_178`, `fold_2:patch_1436`, `fold_2:patch_2`
- Grey pixels denote cross-class overlap without selecting a winning class; amber pixels denote unlabeled/void regions.
