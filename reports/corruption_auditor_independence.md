# Corruption-Generator / Auditor Independence

**State:** design matrix; primary choices are not frozen. The successful synthetic smoke used symmetric corruption, so feature-space independence was correctly recorded as `not_applicable`.

The primary confirmatory comparison must not select difficult examples and then audit them in the identical feature space. Eligibility below is a design rule, not an outcome claim.

| Corruption generator | Auditor evidence | Primary eligibility | Required record |
|---|---|---:|---|
| Symmetric random sampling | Any OOF probability or representation auditor | Eligible; feature independence not applicable | Mechanism, exact count, seed, `not_applicable` reason |
| Configured class-transition matrix | Any OOF probability or representation auditor | Eligible; feature independence not applicable | Frozen transition matrix and seed; no unverified clinical-realism claim |
| Group/tissue-conditional sampling | Any OOF probability or representation auditor | Eligible; feature independence not applicable | Frozen group-rate rule and seed |
| Target morphometrics / engineered morphology | ImageNet or verified pathology embedding model/NN | Eligible when the deep representation does not ingest the engineered generator vector | Generator/auditor identifiers, preprocessing, checksums, `verified_independent` |
| Target morphometrics / engineered morphology | Nearest neighbour or classifier on the identical engineered vector | Exclude from primary; report separately | `circularity_risk: true` and reason |
| Frozen ImageNet embedding A | Same ImageNet weight/preprocessing embedding A | Exclude from primary; report separately | `circularity_risk: true` and exact weight identifier |
| Frozen ImageNet embedding A | Independently trained pathology encoder B | Eligible only after B’s source, licence, weights, preprocessing, and independence are verified | Both identifiers/checksums and `verified_independent` |
| Pathology encoder A | Different pathology encoder B | Eligible only if training provenance/representations are demonstrably distinct enough for the declared rule | Evidence and frozen decision before outcomes |
| Unknown or incompletely documented generator representation | Any representation-based auditor | Not confirmatory | `unverified`, report separately, exclude from primary claims |

Self-confidence, NLL, margin, entropy, and Cleanlab inherit the representation/model used to generate their OOF probabilities. They are not automatically independent merely because the final risk formula differs. The same generator/auditor identifiers must therefore be compared, and exact equality or unresolved provenance triggers exclusion.

The final primary matrix, including the availability-selected pathology encoder, will be frozen only after a real-data pilot. No primary result may be inspected before that freeze.

## Implemented software evidence

The synthetic implementation exposes distinct typed feature bundles rather than accepting free-form names as proof of independence. Its instance-dependent test path selects examples from five morphology-only generator features and audits them with nine colour-only features; hashes bind the feature values, names, roles, and generator/auditor declarations. Tests reject shared or merely renamed feature evidence and mark unresolved cases `circularity_risk`.

The two canonical runs on 2026-07-17 used `symmetric_random_corruption`, not instance-dependent corruption. Their saved independence status is therefore correctly `not_applicable`; they do not provide empirical evidence for an instance-dependent primary comparison. That comparison remains conditional on a frozen real-data design with independently verified representations.
