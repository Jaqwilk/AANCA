# Cross-fold duplicate audit

**Duplicate-analysis status:** complete.

This is a patch-level data-integrity audit, not a medical assessment. Every row is a candidate for review only. No patch was deleted, relabelled, or reassigned automatically.

## Coverage and signals

- Source patches with complete SHA-256/perceptual provenance: 7901.
- Cross-fold exact pairs confirmed by SHA-256 and array equality: 0.
- Patches compared by perceptual hash: 7901; threshold: Hamming <= 4.
- Frozen ImageNet ResNet-18 embedding status: `passed`; patches embedded: 7901; threshold: cosine >= 0.995000.
- The perceptual and frozen-embedding signals are methodologically distinct, not claimed to be statistically independent; both consume the same canonical RGB patch.
- Consolidated cross-fold candidate pairs: 121.

## Review artifacts

- Canonical rankings CSV: `C:\Users\NATAN\Documents\AANCA\artifacts\rankings\cross_fold_duplicate_candidates.csv` (SHA-256 `f83bdd1a08d91bc19b50c7e4d12778e0f2a9a2c7e08106252a12baa12049e891`).
- Full per-patch hash provenance: `C:\Users\NATAN\Documents\AANCA\artifacts\provenance\pannuke_patch_hashes.csv` (SHA-256 `d829ba2b6333021f11ca512557a72b0b439b57e4d21a9f2474d4cd56606ffbd1`).
- Candidate-pair visual grid: `C:\Users\NATAN\Documents\AANCA\reports\figures\cross_fold_duplicate_candidates.png` (SHA-256 `15434e11e8bd8801547a99ed63a1bb2593c138009144c50ee33cde1e8b9f067e`).

Ranking is deterministic and lexicographic: confirmed exact equality, number of distinct evidence signals, embedding cosine, then perceptual distance. It is a review order, not a probability of duplication.

## Highest-ranked candidates

| rank | fold/patch A | fold/patch B | exact | pHash distance | cosine | signals | action |
|---:|---|---|:---:|---:|---:|---|---|
| 1 | 1/1434 | 2/1702 | false | 0 |  | perceptual_average_hash | review_only |
| 2 | 1/1434 | 3/1855 | false | 0 |  | perceptual_average_hash | review_only |
| 3 | 2/1689 | 3/1850 | false | 0 |  | perceptual_average_hash | review_only |
| 4 | 2/1689 | 3/1851 | false | 0 |  | perceptual_average_hash | review_only |
| 5 | 2/1700 | 3/1850 | false | 0 |  | perceptual_average_hash | review_only |
| 6 | 2/1700 | 3/1851 | false | 0 |  | perceptual_average_hash | review_only |
| 7 | 2/1702 | 3/1855 | false | 0 |  | perceptual_average_hash | review_only |
| 8 | 1/1434 | 2/1696 | false | 1 |  | perceptual_average_hash | review_only |
| 9 | 1/1973 | 2/1694 | false | 1 |  | perceptual_average_hash | review_only |
| 10 | 1/1975 | 2/1689 | false | 1 |  | perceptual_average_hash | review_only |
| 11 | 1/1975 | 2/1700 | false | 1 |  | perceptual_average_hash | review_only |
| 12 | 1/1975 | 3/1850 | false | 1 |  | perceptual_average_hash | review_only |
| 13 | 1/1975 | 3/1851 | false | 1 |  | perceptual_average_hash | review_only |
| 14 | 2/1661 | 3/1897 | false | 1 |  | perceptual_average_hash | review_only |
| 15 | 2/1661 | 3/2026 | false | 1 |  | perceptual_average_hash | review_only |
| 16 | 2/1689 | 3/1839 | false | 1 |  | perceptual_average_hash | review_only |
| 17 | 2/1692 | 3/1840 | false | 1 |  | perceptual_average_hash | review_only |
| 18 | 2/1696 | 3/1855 | false | 1 |  | perceptual_average_hash | review_only |
| 19 | 2/1700 | 3/1839 | false | 1 |  | perceptual_average_hash | review_only |
| 20 | 2/1921 | 3/1850 | false | 1 |  | perceptual_average_hash | review_only |
| 21 | 2/1921 | 3/1851 | false | 1 |  | perceptual_average_hash | review_only |
| 22 | 1/1088 | 2/2392 | false | 2 |  | perceptual_average_hash | review_only |
| 23 | 1/1933 | 2/1658 | false | 2 |  | perceptual_average_hash | review_only |
| 24 | 1/1933 | 3/1838 | false | 2 |  | perceptual_average_hash | review_only |
| 25 | 1/1936 | 2/1876 | false | 2 |  | perceptual_average_hash | review_only |
| 26 | 1/1936 | 3/1743 | false | 2 |  | perceptual_average_hash | review_only |
| 27 | 1/1946 | 2/1661 | false | 2 |  | perceptual_average_hash | review_only |
| 28 | 1/1975 | 2/1921 | false | 2 |  | perceptual_average_hash | review_only |
| 29 | 1/1975 | 3/1839 | false | 2 |  | perceptual_average_hash | review_only |
| 30 | 2/603 | 3/775 | false | 2 |  | perceptual_average_hash | review_only |
| 31 | 2/1653 | 3/1897 | false | 2 |  | perceptual_average_hash | review_only |
| 32 | 2/1653 | 3/2026 | false | 2 |  | perceptual_average_hash | review_only |
| 33 | 2/1689 | 3/1852 | false | 2 |  | perceptual_average_hash | review_only |
| 34 | 2/1700 | 3/1852 | false | 2 |  | perceptual_average_hash | review_only |
| 35 | 2/1921 | 3/1839 | false | 2 |  | perceptual_average_hash | review_only |
| 36 | 1/596 | 3/1778 | false | 3 |  | perceptual_average_hash | review_only |
| 37 | 1/620 | 2/1702 | false | 3 |  | perceptual_average_hash | review_only |
| 38 | 1/620 | 3/1855 | false | 3 |  | perceptual_average_hash | review_only |
| 39 | 1/1434 | 3/1856 | false | 3 |  | perceptual_average_hash | review_only |
| 40 | 1/1683 | 2/1246 | false | 3 |  | perceptual_average_hash | review_only |
| 41 | 1/1897 | 2/1245 | false | 3 |  | perceptual_average_hash | review_only |
| 42 | 1/1933 | 2/1925 | false | 3 |  | perceptual_average_hash | review_only |
| 43 | 1/1933 | 3/2011 | false | 3 |  | perceptual_average_hash | review_only |
| 44 | 1/1936 | 2/1925 | false | 3 |  | perceptual_average_hash | review_only |
| 45 | 1/1946 | 2/1653 | false | 3 |  | perceptual_average_hash | review_only |
| 46 | 1/1946 | 3/1897 | false | 3 |  | perceptual_average_hash | review_only |
| 47 | 1/1946 | 3/2026 | false | 3 |  | perceptual_average_hash | review_only |
| 48 | 1/1966 | 2/232 | false | 3 |  | perceptual_average_hash | review_only |
| 49 | 1/1967 | 3/775 | false | 3 |  | perceptual_average_hash | review_only |
| 50 | 1/1967 | 3/1849 | false | 3 |  | perceptual_average_hash | review_only |

## Frozen handling policy before the primary study

Candidates remain in the source release and are never automatically deleted. Any conservative group exclusion rule must be fixed before primary outcomes and without consulting final labels. If likely cross-fold duplicates are confirmed by dataset review, report the main predefined analysis and a sensitivity analysis excluding all affected source-patch groups symmetrically.

Independence limitation: Separation was performed at source-patch level. Patient- and WSI-level independence could not be guaranteed from the released metadata.
