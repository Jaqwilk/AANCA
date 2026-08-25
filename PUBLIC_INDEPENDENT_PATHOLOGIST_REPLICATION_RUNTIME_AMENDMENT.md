# Public independent-pathologist replication pre-outcome runtime amendment

Amendment ID: `public_independent_pathologist_replication_runtime_amendment_v1`.

Date: 2026-08-25.

Parent pre-outcome commit:
`3a805ee46f96309642de9aa3d3af25cd9c881aed`.

Parent config SHA-256:
`1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321`.

Outcome inspection before amendment: **false**. No AANCA score had been computed and
no AANCA/reference association had been opened. The downloader resolves every selected
file authority before downloading any image, so no MIDOG++ TIFF was downloaded by the
failed attempt.

## Trigger

The first official Figshare collection query repeated an article while paginating an
unstably ordered public collection. The downloader therefore encountered the same
`472.tiff` authority twice and stopped with
`duplicate MIDOG++ Figshare file authority: 472.tiff` before the first download.

A direct readback found one genuine authority for `472.tiff`: article `22691209`, file
`40283437`, 129,777,570 bytes, supplied/computed MD5
`f30dbace96a7fd65e7a335886e2a2a37`. The failure was pagination duplication, not two
different image contents.

## Frozen correction

The source-discovery layer now:

1. deduplicates collection records by official article ID before requesting details;
2. accepts a repeated filename only when filename, byte size, supplied MD5 and download
   URL are identical;
3. retains the lexicographically smallest `(article_id, file_id)` identity when an
   identical authority is repeated; and
4. fails closed if any immutable content-authority field conflicts.

The dataset, 70-case hash selection, labels, groups, candidate, folds, risks, review
budgets, matched controls, bootstrap, success gates and claim boundary are unchanged.
The parent config and its SHA-256 remain unchanged. This amendment changes only robust
discovery of the already frozen public files.
