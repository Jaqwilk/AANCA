"""Source-group-safe cross-validation and probabilistic models."""

from .image_oof import (
    ConfirmatoryImageOOFFoldEvidence,
    ConfirmatoryImageOOFResult,
    grouped_oof_confirmatory_cnn,
)
from .oof import (
    GroupFold,
    GroupFoldPlan,
    MultinomialLogisticRegression,
    OOFEstimatorFactory,
    OOFFoldEstimatorContext,
    OOFFoldProvenance,
    OOFResult,
    ProbabilisticEstimator,
    grouped_oof_frozen_embedding_mlp,
    grouped_oof_logistic,
    grouped_oof_predict,
    make_group_stratified_fold_plan,
    make_group_stratified_folds,
)

__all__ = [
    "ConfirmatoryImageOOFFoldEvidence",
    "ConfirmatoryImageOOFResult",
    "GroupFold",
    "GroupFoldPlan",
    "MultinomialLogisticRegression",
    "OOFEstimatorFactory",
    "OOFFoldEstimatorContext",
    "OOFFoldProvenance",
    "OOFResult",
    "ProbabilisticEstimator",
    "grouped_oof_confirmatory_cnn",
    "grouped_oof_frozen_embedding_mlp",
    "grouped_oof_logistic",
    "grouped_oof_predict",
    "make_group_stratified_fold_plan",
    "make_group_stratified_folds",
]
