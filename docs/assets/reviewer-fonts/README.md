# Embedded reviewer typography

Unmodified Latin WOFF2 distributions of the two families already used by the main
AANCA article. Retrieved on 3 October 2026 from Google Fonts; `sources.json` records
the exact official URLs, byte sizes and SHA-256 identities. The adjacent OFL files
retain the original copyright notices and licences. These fonts remain under the
SIL Open Font License 1.1, separately from the project's evaluation licence.

`scripts/build_reviewer_kit.py` embeds the font bytes and complete OFL notices into
each rendered guide. Reading the page makes no font requests, needs no JavaScript
and also works directly from an extracted kit. No font build tool, network fetch or
additional Python dependency is required during packaging or CI.

The published `reviewer-kit-v1` archive is a fixed earlier snapshot. Builds of the
current template are candidates; deployment preserves the published download and
its tracked snapshot identity.
