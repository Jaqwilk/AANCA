# PUMA realism and clean-label safety stress

This is exploratory post-confirmation evidence. The frozen candidate and its prospective PUMA result were not changed.

| Scenario | AANCA - unchanged macro-F1 (95% CI) | AANCA - matched random | Retrieval advantage | Gates | Failed guard |
| --- | ---: | ---: | ---: | --- | --- |
| clean_labels | +0.004540 [+0.000322, +0.009156] | +0.007768 [+0.003613, +0.012248] | +0.000000 [+0.000000, +0.000000] | FAIL | `other` and lymphocyte recall safety |
| symmetric_1pct | +0.005914 [+0.001819, +0.010259] | +0.008133 [+0.004041, +0.012556] | +0.062470 [+0.048275, +0.078458] | FAIL | `other` and lymphocyte recall safety |
| symmetric_2_5pct | +0.005696 [+0.002245, +0.009483] | +0.007387 [+0.003861, +0.010855] | +0.145585 [+0.115892, +0.177670] | FAIL | `other` recall safety |
| symmetric_5pct | +0.006331 [+0.003427, +0.009647] | +0.008406 [+0.005294, +0.011790] | +0.238365 [+0.192225, +0.285993] | FAIL | `other` recall safety |
| targeted_5pct | +0.007119 [+0.004014, +0.010364] | +0.007803 [+0.004286, +0.011348] | +0.179490 [+0.134066, +0.223610] | FAIL | lymphocyte recall safety |
| targeted_10pct | +0.007294 [+0.004185, +0.010664] | +0.006834 [+0.003346, +0.010614] | +0.269958 [+0.203250, +0.331448] | FAIL | lymphocyte recall safety |
| group_conditional_5pct | +0.005146 [+0.002305, +0.008028] | +0.007933 [+0.004858, +0.011056] | +0.252073 [+0.197037, +0.307615] | FAIL | `other` recall safety |
| group_conditional_10pct | +0.005234 [+0.002299, +0.008236] | +0.008320 [+0.004259, +0.012197] | +0.344212 [+0.277128, +0.405814] | PASS | none |
| instance_geometry_5pct | +0.005876 [+0.001105, +0.011001] | +0.007991 [+0.004194, +0.012145] | +0.117870 [+0.080735, +0.162323] | FAIL | `other` recall safety |

Every scenario had a positive aggregate macro-F1 lower bound against unchanged and
matched-random training. Eight failed only because at least one class-recall lower
bound was below `-0.01`. Clean-label exclusion reduced `other` recall by `-0.013733`,
95% interval `[-0.025390, -0.002789]`; geometry-dependent exclusion reduced it by
`-0.016943`, interval `[-0.028028, -0.005871]`. Directional corruption exposed the
corresponding lymphocyte-recall risk.

The geometry-dependent mechanism deterministically produced the same selected and
replacement label arrays under the four configured seeds. Those four entries must
not be interpreted as independent corruption replicates; their value is the frozen
independent-feature-space stress and whole-group uncertainty.

## Boundary

All corruptions remain controlled. Even broad robustness does not prove that a pathologist was wrong or that the workflow improves clinical practice. Source PUMA annotations were never modified.
