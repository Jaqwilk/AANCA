# Public independent-pathologist replication pre-outcome runtime amendment v2

Amendment ID: `public_independent_pathologist_replication_runtime_amendment_v2`.

Date: 2026-08-26.

Parent pre-outcome runtime commit:
`91980cc4e70f54c743b0992427a8833529877412`.

Parent config SHA-256:
`1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321`.

Outcome inspection before amendment: **false**. No AANCA score had been computed and
no AANCA/reference association had been opened. The second downloader attempt again
resolved collection authorities before downloading, so it also downloaded no TIFF.

## Trigger

The v1 amendment correctly rejected conflicts and removed repeated article IDs, but
the official collection endpoint remained unstable across six separate 100-record
pages. On the next request it omitted the authorities for `372.tiff`, `472.tiff` and
`476.tiff`, and execution stopped before download.

The official Figshare OpenAPI document declares a maximum `page_size` of 1,000 for
`GET /collections/{collection_id}/articles`. A single request with `page_size=1000`
returned all 506 unique article IDs in collection `6615571`; the collection therefore
does not require pagination.

## Frozen correction

The source-discovery layer now:

1. requests the collection once with `page_size=1000` and `page=1`;
2. fails closed if 1,000 records are returned or any article ID is duplicated;
3. requests article details with eight workers rather than twenty; and
4. retries HTTP 403, 429 and transient 5xx responses at most five times after the
   initial request, with bounded exponential or `Retry-After` delays no longer than
   eight seconds.

The strict identical-authority check from v1 remains active. The dataset, 70-case
selection, labels, groups, candidate, folds, risks, budgets, matched controls,
bootstrap, success gates and claims remain unchanged. The config and its SHA-256 are
unchanged. This amendment affects only complete, rate-limited discovery of already
frozen public image authorities.
