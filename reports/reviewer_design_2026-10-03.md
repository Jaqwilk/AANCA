# Reviewer guide alignment — 3 October 2026

The owner requested that `https://aancastudy.org/review/` visually match the main
article. D057 bounds this presentation refinement; the scientific authorities,
thirteen-file article and published `reviewer-kit-v1` archive remain unchanged.

The guide now uses the article's four-tile mark, study navigation, exact palette,
Inter/JetBrains Mono, type scale, 1160-pixel rail and thin rules. A compact index
sits beside the introduction on desktop. Section headings frame the reading
column; seven semantic evidence rows stack on phones. Commands retain their exact
text and show the required environment. Negative findings and chronology limits
remain visible, with descriptive source links.

Unmodified Latin WOFF2 brand fonts and their original OFL notices are embedded
into rendered HTML. Their official source URLs and identities are retained in
`docs/assets/reviewer-fonts/sources.json`. This avoids font requests and preserves
the same lettering when a current candidate kit is opened directly from disk.
The fixed published kit keeps its original layout and immutable checksum; live
presentation updates do not silently replace that historical download.

## Executed verification

- Initial browser inspection at 1440, 768, 390 and 320 pixels found no horizontal
  overflow, missing in-page anchors, text-contrast failures or browser exceptions.
  All seven findings and source links remained available.
- Primary action contrast was 4.70:1 normally and 5.52:1 on hover. Body uses
  16-pixel Inter with 27.84-pixel line height. Keyboard focus remained visible;
  native menu and digest disclosure worked without JavaScript.
- One fix batch extended the provenance separator to the full rail. The single
  confirmation inspected desktop and mobile directly from the extracted candidate
  kit, with both embedded fonts loaded and zero external requests. All 47 local
  links, including article navigation fragments, resolve inside the kit.
- White print styling, reduced motion and unobscured anchor destinations passed.
  The detector's line-zero tracking/leading warnings conflict with measured CSS:
  body tracking is -.128 pixels (-.008em), display tracking -.04em, and body leading
  1.74. The display scale intentionally follows the existing brand authority.
- Fresh isolated build/extraction and `review_project.py --numeric --online`
  passed. NumPy readbacks retain both external studies' adverse/not-supported
  conclusions. No model was trained and no source annotation changed.
- `uv run pytest --durations=10 --junitxml=artifacts/qa/reviewer-design-tests.xml`:
  1,259 passed, one native Windows skip, no failures, 605.12 seconds. Ruff check,
  format (233 files) and `mypy src` (106 files) passed. Hosting preflight built
  both outputs, verified the isolated kit and connected over SSH/SFTP.

Exact browser, font-source and scoped-verification evidence is retained in
`reports/reviewer_design_2026-10-03_evidence.json`. Transient captures/logs remain
under `output/playwright` and `artifacts/qa`.

## Publication

The backed-up hosting deployment completed with backup `20261003-222525`. Both
builds, SSH/SFTP connection and health checks passed. Ordinary apex and www HTTP
readbacks match the generated reviewer guide and the independently pinned public
snapshot. Both unchanged thirteen-file articles still match the local manifest.
The review document is 142,205 bytes and needs no additional font or script assets.
The public release checksum remains
`95fbd30be3126e7b987a5454f15a08e31141a645c4a00262efd88268aa33a0dc`.

The existing `Scientific software` workflow runs the reviewer lane, full disjoint
partitions and mandatory complete-coverage aggregation for each new published
revision. The preceding documentation revision passed all seven jobs in
[run 37148633021](https://github.com/Jaqwilk/AANCA/actions/runs/37148633021); that is
prior-revision evidence, not a claim about this refinement's hosted results.
