# AANCA presentation MVP

The package was generated from selected, checksum-verified sources in the accepted
run `20260727T133947.089370Z_pannuke_primary_orphan_recovery`. From the repository root, the recommended presentation
command is:

```powershell
python scripts/present_demo.py
```

This standard-library launcher requires no project dependency installation. It
verifies the closed package before serving it on `127.0.0.1` and opens the article
in the default browser. No model run, dataset, or GPU is required. Use `--no-open`
in headless environments and `--port 0` to select a free port.

For verification without a browser or server, run:

```powershell
python scripts/present_demo.py --verify-only
```

After installing the full research environment, the equivalent commands are
`uv run histo-audit demo serve` and `uv run histo-audit demo verify`.

To share the presentation without repository access, compress and send this entire
directory. The reviewer should extract every file and open `index.html`. Do not send
only `index.html`: the QC image, machine-readable evidence and checksum manifest are
separate files in the same package. Repository links require reviewer access when
the GitHub repository is private.

Author: Natan Smogór. Updated: 20 August 2026.

The responsive presentation is written in English and uses pinned GSAP and
Three.js browser modules for progressive animation. Its evidence, navigation,
tables and scientific interpretation remain available when motion is reduced;
network access is only needed for the optional web fonts, institution logos and
animation libraries.
The WebGL loop pauses while the hero or browser tab is not visible, uses a capped
pixel ratio, and respects reduced-motion and data-saving preferences.

The complete 36-entry comparison atlas remains immediately available inside a
compact, keyboard-scrollable viewport. The duplicate numeric table stays one
native disclosure control away. On narrow screens, each evidence entry becomes a
labelled two-column record without dropping any saved identifier, interval,
adjusted p-value, or unavailable result.

Scientific status: `PRIMARY_STUDY_COMPLETE` and
`EXTERNAL_VALIDATION_COMPLETE`. Presentation status: `DEMO_COMPLETE`. The PanNuke
primary analysis is permanently labelled `amended_or_exploratory`; confirmatory CNN
work and newly recruited blinded expert review were not run. The completed NuCLS
multi-rater validation did not meet its frozen ranking or downstream success rules.

This is a non-diagnostic research prototype. It identifies a potentially
inconsistent annotation and recommends it for expert review; it never modifies
source annotations or claims that a pathologist was wrong.

`evidence.json` contains the sourced H1-H7 summary, the adverse H4 result,
the complete H2 subgroup summary, the byte-identical instance-dependent seed
disclosure, all 36 saved H1/H3/H5/H6/H7 comparisons and the completed null/adverse
NuCLS external-validation summary. P-values shown in the HTML are explicitly
labelled one-sided and Holm-adjusted. `manifest.json` binds every other file in this
package.

Of the 36 preregistered comparison entries, 33 contain numeric results and the
three H6 entries remain explicitly unavailable under the frozen encoder gate.
The public `primary-evidence-v1` GitHub release contains all completed-cell OOF
predictions and rankings, the full bootstrap and H4 restoration arrays. It supports
independent recalculation of the saved comparison statistics; it does not include
raw PanNuke binaries or fold checkpoints that were not retained.
The checked-in NuCLS evidence and independent verifier are documented in
`reports/nucls_external_validation_results.md`; they include derived arrays and
portable source inventories, not raw NuCLS images.
The same bundle is published as GitHub release
`nucls-external-validation-v1` with SHA-256
`e7384e2e8ff6eeab97485dfa3196ddbd261bbe335ebfa572d9f275de402a4d08`.
Source code, setup guidance, specifications, tests, and the complete documentation
map are available at <https://github.com/Jaqwilk/AANCA>.
