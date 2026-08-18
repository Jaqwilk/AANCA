# Preregistration amendment

- Timestamp (UTC): 2026-07-27T13:39:47.089370Z
- Parent authority: C:\Users\NATAN\Documents\AANCA\artifacts\preregistration_amendments\20260719T011146.248393Z
- Parent artifact root: 962ab8b5110d062a314591f6144e0f94bebf68239f9ae8b014e2635eaf42031f
- Outcomes inspected: true
- Outcomes inspected at (UTC): 2026-07-27T10:57:07.000000Z
- Reason: Authorize zero-training recovery of the host-reboot-interrupted unsealed primary after accidental outcome exposure, preserving every frozen scientific method and classifying the recovered analyses as amended_or_exploratory.

## Affected hypotheses

- H1
- H2
- H3
- H4
- H5
- H6
- H7

## Affected analyses

- primary_frozen_feature_benchmark_all_registered_cells
- primary_restoration_analysis
- primary_registered_statistics
- confirmatory_primary_dependency_gate
- confirmatory_single_copy_checkpoint_storage
- primary_orphan_recovery_execution_lineage

## Reporting policy

Every affected analysis is amended or exploratory and can never be reported as the original unamended primary analysis.



## Interrupted-primary recovery authorization

- Policy: `interrupted_unsealed_primary_recovery_v1`
- Source run ID: `20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`
- Source run directory: `C:\Users\NATAN\Documents\AANCA\artifacts\runs\20260719T012712.600409Z_pannuke_primary_frozen_feature_benchmark_f0ed2d1a3f`
- Interruption kind: `host_reboot`
- Host boot timestamp (UTC): `2026-07-27T10:37:04.5000000Z`
- Process absence checked at (UTC): `2026-07-27T11:07:04.4225163Z`
- Source process ID: 20792
- Source process active: false
- Interruption receipt: `C:\Users\NATAN\Documents\AANCA\artifacts\process_observations\20260727T110704.4225163Z_primary_orphan_boot_receipt.json`
- Interruption receipt SHA-256: `2cb53b1efe1ab6441ff9fa6b93c929fb7a5395bd52a20556a3149366fc46d8cc`
- Outcomes inspected: true
- Outcome inspection timestamp (UTC): `2026-07-27T10:57:07.000000Z`
- Analysis disposition: `amended_or_exploratory`
- Scientific method changes: none
- Expected source snapshot root SHA-256: `bc224f73960792a495e03e5039c075c0061ee3d86cce095afb21a91a061cc027`
- Expected filesystem readback root SHA-256: `7ce192b40d241fd2c8b394c1e03905d51a5c654b1373646a5324768a08cc4039`
- Expected restoration readback root SHA-256: `8ad17315d8e90ac8702477a0cf1e7d8a29479eb1282a3055009e164d52713d1d`
- Expected statistics manifest SHA-256: `2d3c8115d371d7bbe55df3a0af83f28a875ec463823c0b07e0086efe1623318f`
- Trust assumption: `trusted_local_process_no_dependency_injection_import_hook_hotpatch_or_concurrent_writer`
- Limitation: `control_flow_and_content_addressed_inheritance_not_fresh_semantic_recomputation`

The source remains an unsealed, ineligible, read-only orphan and may not be repaired,
overwritten, or retroactively sealed. Recovery must verify the interruption receipt and
every expected digest, physically copy only the exact authorized artifact set into one
new run with `retry_of_run_id`, retrain zero cells, and perform no training fallback,
automatic retry, selection, tuning, or scientific-method change. Any recovered analysis
is permanently `amended_or_exploratory`, never the original unamended primary analysis.



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

