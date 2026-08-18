# Preregistration amendment

- Timestamp (UTC): 2026-07-27T17:04:13.080954Z
- Parent authority: C:\Users\NATAN\Documents\AANCA\artifacts\preregistration_amendments\20260727T133947.089370Z
- Parent artifact root: 4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a
- Outcomes inspected: true
- Outcomes inspected at (UTC): 2026-07-27T10:57:07.000000Z
- Reason: Authorize the fixed resource_bounded_confirmatory_v1 sensitivity/feasibility profile using operational resource constraints only after outcome exposure; preserve the sealed recovered primary and original frozen confirmatory definition, and prohibit original-confirmatory, completion-stage, and M9 claims.

## Affected hypotheses

- H1
- H2
- H3
- H4
- H5
- H6
- H7

## Affected analyses

- resource_bounded_confirmatory_sensitivity
- confirmatory_fold_rotation
- confirmatory_ranking_statistics
- confirmatory_restoration_analysis
- confirmatory_model_representation_comparisons
- confirmatory_checkpoint_successor_lineage
- confirmatory_completion_and_m9_eligibility

## Reporting policy

Every affected analysis is amended or exploratory and can never be reported as the original unamended primary analysis.




## Resource-bounded confirmatory execution authority

- Purpose: `resource_bounded_confirmatory_execution`
- Policy: `post_outcome_resource_bounded_confirmatory_execution_v1`
- Historical primary run ID: `20260727T133947.089370Z_pannuke_primary_orphan_recovery`
- Historical primary run directory: `C:\Users\NATAN\Documents\AANCA\artifacts\runs\20260727T133947.089370Z_pannuke_primary_orphan_recovery`
- Historical primary artifact root SHA-256: `8c1c7b277d96889dc4fb45aee282e77e3d351f687990e03e6b57ec5f2313c7e4`
- Historical primary artifact-manifest SHA-256: `9abff1b2f0e745a50b3aa1922d3d725bb629276f6090c32e9b423fea82d0e0ce`
- Historical primary completion evidence SHA-256: `77a02877f9882a608968303e077963abdfbcfabdc997424cfaceadf0df86349c`
- Historical primary execution-gate SHA-256: `7bff23bace66436d191df9f3fe4b92f4db8caf391e019dc9864c650fc1fc0ae7`
- Historical primary stage-attestation record SHA-256: `5af827544502fbdf688a73916ec58b5dac0984c5a682a33ce6dfc97538228871`
- Historical primary stage-attestation verification SHA-256: `e19f17b14a375e8e048e1d2fc3060c19831e4259c8311831d88f75753024c768`
- Historical primary recovery evidence SHA-256: `3f937dac3dc2788131dafc84ce35ae7fda4a24a35bcf437ee14827169b8ccaf1`
- Historical primary recovery authorization SHA-256: `e571ca2eb345439e541e715bae59cc5a19bc75d87a5e7f63878889a0894e2732`
- Historical registration authority: `C:\Users\NATAN\Documents\AANCA\artifacts\preregistration_amendments\20260727T133947.089370Z`
- Historical registration-authority root SHA-256: `4d368d3f49852ecf7678215a5a64c2617067cc0581d353af33460f46ec67f88a`
- Resource profile ID: `resource_bounded_confirmatory_v1`
- Resource confirmatory-config file SHA-256: `783968e8afc132cca0c877aadf953fc68d3c35f606021b3a97ed380478dbad4a`
- Resource confirmatory-config semantic SHA-256: `1c9a41b92dabbeafbb92b1bc8aced158337046fc1d6e056b011f6a27b98e8298`
- Parent execution-source root SHA-256: `ba7fb4c8336c4f9ba138fcda16019dc31bec7e5cc3e8b846e643d6dd0332601b`
- Resource execution-source root SHA-256: `1179f91725a3027c0397e87691774377bbd4ba5469d588390c72b0b88515547b`
- Exact source-delta SHA-256: `7abd9e1627728c4ce89f59cc6162283ec8963468816db6c64849fff1a5ec290e`
- Planned required cells: 24
- Planned CNN cells: 6
- Planned CNN fold checkpoints: 30
- Maximum epochs: 4
- Projected stable run bytes: 12884901888
- Fixed safety margin bytes: 10737418240
- Minimum free bytes before tracker creation: 23622320128
- Maximum active atomic temporary checkpoints: 1
- Minimum total host RAM bytes: 32212254720
- Minimum available host RAM bytes before data loading: 17179869184
- Minimum available host RAM bytes immediately before tracker creation: 12884901888
- CUDA required: true
- Fixed CUDA device index: 0
- Minimum total VRAM bytes: 10737418240
- Minimum free VRAM bytes: 8589934592
- cuDNN required: true
- AMP required: true
- AMP dtype: `float16`
- CUDA smoke input shape: `[1, 3, 224, 224]`
- CUDA smoke forward/backward required: true
- CUDA smoke finite forward/backward required: true
- CUDA smoke maximum peak allocated bytes: 536870912
- Official weight identifier: `ResNet18_Weights.IMAGENET1K_V1`
- Official weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- Implicit weight download allowed: false
- Outcomes inspected: true
- Analysis disposition: `amended_or_exploratory`
- Original confirmatory claim allowed: false
- Study-outcome eligible: false
- Completion stage: null
- Primary rebinding allowed: false
- Primary mutation allowed: false

Exact allowlisted execution-source changes:

- `configs/confirmatory_resource_bounded_amended.yaml` (added)
- `src/histo_audit/cli.py` (modified)
- `src/histo_audit/experiment/__init__.py` (modified)
- `src/histo_audit/experiment/confirmatory_cli_inputs.py` (added)
- `src/histo_audit/experiment/confirmatory_completion.py` (modified)
- `src/histo_audit/experiment/confirmatory_core.py` (modified)
- `src/histo_audit/experiment/confirmatory_runner.py` (modified)
- `src/histo_audit/experiment/resource_bounded_resume.py` (added)
- `src/histo_audit/experiment/resource_bounded_runner.py` (added)
- `src/histo_audit/experiment/study_contracts.py` (modified)
- `src/histo_audit/models/cnn.py` (modified)
- `src/histo_audit/workflows/__init__.py` (modified)
- `src/histo_audit/workflows/lifecycle_qualification.py` (added)
- `src/histo_audit/workflows/preregistration_amendment.py` (modified)
- `src/histo_audit/workflows/study_gates.py` (modified)

This child authority does not alter or replace the historical primary authority. The
historical primary is verified under its direct recovery amendment, while the current
resource profile and live execution source are verified independently under this
child. The child is permanently post-outcome and `amended_or_exploratory`; it may not
support the original confirmatory claim or any completion-stage transition.



## Confirmatory checkpoint-storage policy

- Policy: `single_canonical_checkpoint_copy_v1`
- Scope: `one_checkpoint_per_completed_cnn_cell_oof_fold`
- Canonical relative path: `cells/{cell_id}/checkpoints/fold_{fold_id:02d}.pt`
- Retained copy count: 1
- Link policy: `regular_file_no_symlink_no_junction_no_hardlink`
- Verification policy: `size_sha256_fold_evidence_checkpoint_manifest_postseal_exact_set`
- Scientific effect: `storage_only_no_model_data_split_seed_prediction_metric_restoration_or_estimand_change`

This authority changes storage and reconciliation only. It does not change models,
data, splits, seeds, predictions, metrics, restoration, or estimands. A confirmatory
runner must enforce this policy independently before any eligibility claim.

