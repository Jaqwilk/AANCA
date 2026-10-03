# AANCA v1 reviewer access and CI optimisation

Date: 3 October 2026. Engineering work; no change of scientific stage.

## Research and scope

The intended reviewer is experienced in medical-school research; their identity
and personal procedure are unknown. The implementation is an inference from
institutional guidance, not a claim about a particular professor or Harvard approval.
[HMS reproducibility guidance](https://datamanagement.hms.harvard.edu/collect-analyze/reproducibility)
supports transparent protocols, versioned code, discoverable documentation and
context. This motivates a short methods-to-evidence reading path and separate
integrity, recalculation and full-experiment instructions.
[HMS access and reuse guidance](https://datamanagement.hms.harvard.edu/access-reuse)
supports documented, identifiable, preserved materials with explicit access and
reuse conditions. This motivates a small offline package, file identities, source
revision and clear evaluation permission. No DOI or independent validator is invented.

[GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
explains that public visibility does not itself grant general reuse rights. On
3 October the owner explicitly selected a narrow non-clinical scientific-evaluation
grant. `LICENSE` permits local copying and execution for evaluation while reserving
the remaining rights; third-party terms are unchanged. This is not an open-source
licence or a grant of clinical use.

## Measured baseline

The successful [run 37139831977](https://github.com/Jaqwilk/AANCA/actions/runs/37139831977)
tested `32032a794dfef155687196b4537a7631529635dc`:

| Platform | Job duration | Pytest step | Dependency installation | Actual pytest outcomes |
| --- | ---: | ---: | ---: | --- |
| Ubuntu | 677 s | 603 s | 36 s | 1,203 passed; 31 skipped |
| Windows | 1,679 s | 1,487 s | 78 s | 1,230 passed; 4 skipped |

Both collected 1,234 cases. The platform difference includes Windows-native file
custody tests; CUDA and uncached official-weight cases also have explicit skips.
The full logs and API readback are retained locally under `artifacts/qa`.

A local checkpoint-completion test took 25.26 s in its call phase. Setting OMP/MKL
thread counts to one produced 24.99 s, which is insufficient evidence of a useful
speed improvement. A cProfile run identified repeated checkpoint reads and tensor
validation. These checks guard frozen evidence and are retained. Profiling follows
the [official pytest timing guidance](https://pytest.org/en/stable/reference/reference.html).

## Execution plan

1. Add deterministic test-case partitioning to isolated CI jobs, with full collection
   receipts, actual outcomes, revision identity, JUnit and slow-test timings. A required
   aggregation job rejects missing, overlapping, failed or inconsistent coverage.
   Unpartitioned `pytest` continues to run the complete suite.
2. Add a dependency-free reviewer check, optional NumPy-only recalculation of the
   retained NuCLS and MoNuSAC evidence, and a deterministic offline reviewer ZIP.
   Include negative conclusions and distinguish saved-array recalculation from
   source-image retraining. Never silently download or train anything.
3. Extend the existing article with a reviewer guide at `/review/`, a prominent
   README entry, direct evidence/protocol links and the limited evaluation permission.
   Preserve the existing identity and all frozen numerical authorities.
4. Run regression tests, all mandatory repository gates and the synthetic CLI;
   exercise the extracted kit outside the repository, on both CI platforms, and
   inspect the browser guide on desktop and mobile.
5. Publish the reviewed code and kit, deploy with a recoverable backup, verify public
   bytes and measure fresh Ubuntu/Windows CI. Report measured wall time separately
   from runner resource usage and scientific efficacy.

Two full-suite jobs per platform spend more runner setup resources to reduce
waiting. This is not a promise of reduced total compute. The quick reviewer lane
uses standard-library checks and does not substitute for the scientific software
suite. Dependency caching is not the measured bottleneck; no large cache is added
without evidence. [GitHub distinguishes caches from workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching),
which are used here to retain execution receipts and test reports.

## Execution evidence

Implemented the standard-library reviewer entry point, explicit NumPy-only optional
readback, deterministic 67-file ZIP, shared online/offline guide, evaluation permission
and complete-suite CI partitions. The archive uses stored members because arrays
and images are already compressed; fixed metadata and exact bytes avoid zlib-version
variation. Its source revision and build-tree state are explicit. All 22 guide-local
links resolve in the extracted kit.

Focused regression checks passed: 39 reviewer/shard/article cases initially, then
29 CI/reviewer/shard cases after adapting the existing workflow assertions. Both
real 25-case partition runs and their coverage aggregation passed. Ruff and format
passed (233 files); mypy passed for 111 files (the `src` invocation checks 106).
Default kit integrity/narrative checks took about 0.1 seconds locally. NumPy-only
NuCLS/MoNuSAC recalculation passed in about 3 seconds, retaining their not-supported
scientific decisions. Synthetic smoke completed in an isolated QA registry.

The initial complete unpartitioned suite produced 1,256 passes, one documented
platform skip and two failures in pre-existing workflow-shape tests, in 586.53 s.
Those tests incorrectly assumed the first checkout was the scientific LFS checkout
and located pytest by its old step name. The tests now parse the workflow's job
structure, require LFS on both full platforms, enforce the Windows type gate, and
require the complete-coverage aggregation. No scientific test or LFS requirement
was removed. The repaired complete suite passed as two isolated local partitions:
619 passes in 319.04 seconds and 640 passes plus one platform skip in 330.52 seconds.
The aggregation verified every one of 1,260 collected cases exactly once, with
1,259 passes and one documented skip. These are local, not hosted CI measurements.
Full logs, JUnit and actual-outcome receipts are retained under `artifacts/qa`.

Bounded browser inspection and one confirmation at 1440/390 pixels found no page
horizontal overflow or missing anchors. Standalone navigation targets are 44 px;
keyboard skip-link focus is visible. Measured body contrast is 13.65:1 and the
primary action is 4.70:1. Print uses dark text on white; the new guide has no scripts
or external font/assets requirement. These are targeted checks, not a formal
accessibility certification. The static detector's inherited Inter warning is
outside this extension's scope; the existing brand was preserved. Its print-hover
contrast finding was fixed in the single accessibility correction batch.

Publication, live-byte equality and fresh hosted Ubuntu/Windows timings remain
pending until all mandatory gates pass.
Scientific status remains `EXTERNAL_VALIDATION_COMPLETE`, presentation status
`DEMO_COMPLETE`, and natural-data action `retain_uncorrected`.
