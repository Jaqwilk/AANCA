"""Executable core experiment orchestrators."""

from importlib import import_module
from typing import Any

from .confirmatory_completion import (
    ConfirmatoryArtifactReadback,
    ConfirmatoryCellReadback,
    ConfirmatoryFilesystemReadback,
    ConfirmatoryMatrixReconciliation,
    build_confirmatory_completion_evidence,
    read_confirmatory_run_directory,
    reconcile_confirmatory_cell_outcomes,
)
from .confirmatory_core import (
    ConfirmatoryComparisonOperand,
    ConfirmatoryCorruptionInput,
    ConfirmatoryExecutionControls,
    ConfirmatoryFrozenBlocker,
    ConfirmatoryMatrixArtifacts,
    ConfirmatoryPairedComparison,
    ConfirmatoryRotationInputs,
    ConfirmatoryRunnerInputs,
    FrozenFeatureOOFExecution,
    FrozenFeatureProvenance,
    SyntheticConfirmatoryFixtureResult,
    confirmatory_execution_controls_from_frozen_config,
    execute_confirmatory_matrix,
    run_synthetic_confirmatory_contract_fixture,
)
from .confirmatory_statistics import (
    ConfirmatoryStatisticsArtifacts,
    ConfirmatoryStatisticsVerification,
    aggregate_confirmatory_statistics,
    verify_confirmatory_statistics_artifacts,
)
from .pannuke_confirmatory_inputs import (
    ConfirmatoryFrozenFeatureAvailability,
    ConfirmatoryFrozenFeatureCacheSpec,
    ConfirmatoryObservedLabelSet,
    ConfirmatoryPartitionFeature,
    ConfirmatoryPartitionInputs,
    PanNukeConfirmatoryInputs,
    PanNukeConfirmatoryRotationInputs,
    load_pannuke_confirmatory_inputs,
)
from .pannuke_primary_inputs import (
    PanNukePrimaryCachePaths,
    PanNukePrimaryHashExpectations,
    PanNukePrimaryInputError,
    PanNukePrimaryInputsResult,
    RepresentationAvailability,
    build_pannuke_primary_inputs,
    select_stratified_reference_validation_groups,
)
from .pilot import (
    PanNukePilotDevelopmentManifestView,
    PanNukePilotResult,
    build_pannuke_pilot_development_manifest_view,
    reconcile_pilot_audit_evidence,
    run_pannuke_pilot,
)
from .primary_completion import (
    PrimaryFilesystemReadbackEvidence,
    PrimaryMatrixReconciliation,
    PrimaryRestorationReadbackEvidence,
    build_primary_completion_evidence,
    read_primary_filesystem_evidence,
    read_primary_restoration_evidence,
    reconcile_primary_cell_outcomes,
)
from .primary_core import (
    PrimaryCalibrationControls,
    PrimaryCellSelector,
    PrimaryCrossCellComparison,
    PrimaryDownstreamComparison,
    PrimaryExecutionControls,
    PrimaryMatrixArtifacts,
    PrimaryMatrixInputs,
    PrimaryMethodVsRandomComparison,
    PrimaryPairedComparison,
    PrimaryWithinCellComparison,
    SyntheticPrimaryFixtureResult,
    execute_primary_matrix,
    primary_execution_controls_from_frozen_config,
    run_synthetic_primary_integration_fixture,
)
from .primary_recovery import (
    RECOVERY_COPY_POLICY,
    RECOVERY_EVIDENCE_FILENAME,
    RECOVERY_EXPERIMENT_NAME,
    RECOVERY_POLICY,
    RECOVERY_REGISTRATION_STATUS,
    OrphanSourceInspection,
    OrphanSourceSnapshot,
    PrimaryRecoveryError,
    RecoveryArtifact,
    RecoveryAuthorization,
    RecoveryCopyReceipt,
    RecoveryDestinationVerification,
    RecoveryInterruptionEvidence,
    build_primary_recovery_authorization,
    collect_orphan_source_snapshot,
    copy_authorized_orphan_artifacts,
    inspect_orphan_source,
    verify_recovery_destination,
)
from .primary_statistics import (
    PrimaryStatisticsArtifacts,
    PrimaryStatisticsVerification,
    aggregate_primary_statistics,
    verify_primary_statistics_artifacts,
)
from .smoke import SyntheticSmokeResult, run_smoke, run_synthetic_smoke
from .study_contracts import (
    ConfirmatoryMatrixPlan,
    PrimaryMatrixPlan,
    StudyContractError,
    build_confirmatory_matrix_plan,
    build_primary_matrix_plan,
    validate_frozen_confirmatory_config,
    validate_frozen_primary_config,
)

_PRIMARY_RUNNER_EXPORTS = frozenset(
    {
        "PrimaryRunnerDependencies",
        "PrimaryStudyIntegrityError",
        "PrimaryStudyRunnerError",
        "default_primary_cache_paths",
        "derive_primary_cache_hashes",
        "execute_primary_study",
    }
)

_PRIMARY_RECOVERY_RUNNER_EXPORTS = frozenset(
    {
        "PrimaryRecoveryRunnerError",
        "RecoveryDiskPreflight",
        "execute_primary_orphan_recovery",
        "preflight_primary_orphan_recovery",
        "verify_primary_recovery_evidence",
    }
)


def __getattr__(name: str) -> Any:
    """Load the workflow-dependent primary runner only when explicitly requested."""

    if name in _PRIMARY_RUNNER_EXPORTS:
        value = getattr(import_module(f"{__name__}.primary_runner"), name)
        globals()[name] = value
        return value
    if name in _PRIMARY_RECOVERY_RUNNER_EXPORTS:
        value = getattr(import_module(f"{__name__}.primary_recovery_runner"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RECOVERY_COPY_POLICY",
    "RECOVERY_EVIDENCE_FILENAME",
    "RECOVERY_EXPERIMENT_NAME",
    "RECOVERY_POLICY",
    "RECOVERY_REGISTRATION_STATUS",
    "ConfirmatoryArtifactReadback",
    "ConfirmatoryCellReadback",
    "ConfirmatoryComparisonOperand",
    "ConfirmatoryCorruptionInput",
    "ConfirmatoryExecutionControls",
    "ConfirmatoryFilesystemReadback",
    "ConfirmatoryFrozenBlocker",
    "ConfirmatoryFrozenFeatureAvailability",
    "ConfirmatoryFrozenFeatureCacheSpec",
    "ConfirmatoryMatrixArtifacts",
    "ConfirmatoryMatrixPlan",
    "ConfirmatoryMatrixReconciliation",
    "ConfirmatoryObservedLabelSet",
    "ConfirmatoryPairedComparison",
    "ConfirmatoryPartitionFeature",
    "ConfirmatoryPartitionInputs",
    "ConfirmatoryRotationInputs",
    "ConfirmatoryRunnerInputs",
    "ConfirmatoryStatisticsArtifacts",
    "ConfirmatoryStatisticsVerification",
    "FrozenFeatureOOFExecution",
    "FrozenFeatureProvenance",
    "OrphanSourceInspection",
    "OrphanSourceSnapshot",
    "PanNukeConfirmatoryInputs",
    "PanNukeConfirmatoryRotationInputs",
    "PanNukePilotDevelopmentManifestView",
    "PanNukePilotResult",
    "PanNukePrimaryCachePaths",
    "PanNukePrimaryHashExpectations",
    "PanNukePrimaryInputError",
    "PanNukePrimaryInputsResult",
    "PrimaryCalibrationControls",
    "PrimaryCellSelector",
    "PrimaryCrossCellComparison",
    "PrimaryDownstreamComparison",
    "PrimaryExecutionControls",
    "PrimaryFilesystemReadbackEvidence",
    "PrimaryMatrixArtifacts",
    "PrimaryMatrixInputs",
    "PrimaryMatrixPlan",
    "PrimaryMatrixReconciliation",
    "PrimaryMethodVsRandomComparison",
    "PrimaryPairedComparison",
    "PrimaryRecoveryError",
    "PrimaryRecoveryRunnerError",
    "PrimaryRestorationReadbackEvidence",
    "PrimaryRunnerDependencies",
    "PrimaryStatisticsArtifacts",
    "PrimaryStatisticsVerification",
    "PrimaryStudyIntegrityError",
    "PrimaryStudyRunnerError",
    "PrimaryWithinCellComparison",
    "RecoveryArtifact",
    "RecoveryAuthorization",
    "RecoveryCopyReceipt",
    "RecoveryDestinationVerification",
    "RecoveryDiskPreflight",
    "RecoveryInterruptionEvidence",
    "RepresentationAvailability",
    "StudyContractError",
    "SyntheticConfirmatoryFixtureResult",
    "SyntheticPrimaryFixtureResult",
    "SyntheticSmokeResult",
    "aggregate_confirmatory_statistics",
    "aggregate_primary_statistics",
    "build_confirmatory_completion_evidence",
    "build_confirmatory_matrix_plan",
    "build_pannuke_pilot_development_manifest_view",
    "build_pannuke_primary_inputs",
    "build_primary_completion_evidence",
    "build_primary_matrix_plan",
    "build_primary_recovery_authorization",
    "collect_orphan_source_snapshot",
    "confirmatory_execution_controls_from_frozen_config",
    "copy_authorized_orphan_artifacts",
    "default_primary_cache_paths",
    "derive_primary_cache_hashes",
    "execute_confirmatory_matrix",
    "execute_primary_matrix",
    "execute_primary_orphan_recovery",
    "execute_primary_study",
    "inspect_orphan_source",
    "load_pannuke_confirmatory_inputs",
    "preflight_primary_orphan_recovery",
    "primary_execution_controls_from_frozen_config",
    "read_confirmatory_run_directory",
    "read_primary_filesystem_evidence",
    "read_primary_restoration_evidence",
    "reconcile_confirmatory_cell_outcomes",
    "reconcile_pilot_audit_evidence",
    "reconcile_primary_cell_outcomes",
    "run_pannuke_pilot",
    "run_smoke",
    "run_synthetic_confirmatory_contract_fixture",
    "run_synthetic_primary_integration_fixture",
    "run_synthetic_smoke",
    "select_stratified_reference_validation_groups",
    "validate_frozen_confirmatory_config",
    "validate_frozen_primary_config",
    "verify_confirmatory_statistics_artifacts",
    "verify_primary_recovery_evidence",
    "verify_primary_statistics_artifacts",
    "verify_recovery_destination",
]
