"""Re-run the frozen development winner while recording every optimiser convergence flag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import histo_audit.research.autoresearch as autoresearch_module
from histo_audit.evaluation.review_training import (
    SoftTargetMultinomialLogisticRegression as OriginalSoftModel,
)
from histo_audit.external_validation.monusac import (
    load_frozen_monusac_config,
    prepare_monusac_split,
)
from histo_audit.research.expanded_autoresearch import (
    ExpandedAutoresearchEvaluator,
    build_all_development_partition,
    extract_scaled_resnet18_embeddings,
    load_expanded_autoresearch_config,
    validate_aligned_scales,
)
from histo_audit.research.frozen_candidate import load_frozen_development_candidate
from histo_audit.utils.run_tracking import atomic_write_json, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            f"selected-candidate verification changed {name}: {actual} != {expected}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("configs/aanca_selected_development_candidate.yaml"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/monusac"))
    parser.add_argument(
        "--parent-output",
        type=Path,
        default=Path("artifacts/autoresearch/monusac_aanca_expanded_development_v1_r1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/autoresearch/monusac_aanca_expanded_convergence_v1.json"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidate_path = (root / args.candidate).resolve()
    frozen = load_frozen_development_candidate(candidate_path)
    if frozen.candidate.feature_view != "resnet18_multiscale_64_128":
        raise ValueError("convergence verifier currently supports the frozen multiscale view")
    config, config_sha256 = load_expanded_autoresearch_config(root)
    parent_root = (root / args.parent_output).resolve()
    authority = _load_json(parent_root / "run_authority.json")
    expected_authority = frozen.selection_authority
    amendment_path = root / "configs" / "aanca_autoresearch_full_runtime_amendment.yaml"
    if (
        authority.get("study_id") != expected_authority["parent_study_id"]
        or config_sha256 != expected_authority["parent_config_sha256"]
        or authority.get("partition_sha256") != expected_authority["partition_sha256"]
        or sha256_file(amendment_path) != expected_authority["runtime_amendment_sha256"]
        or sha256_file(parent_root / "run_authority.json")
        != expected_authority["parent_authority_sha256"]
        or sha256_file(parent_root / "results.jsonl") != expected_authority["parent_ledger_sha256"]
    ):
        raise RuntimeError("selected candidate no longer matches the frozen parent authority")

    data_root = (root / args.data_root).resolve()
    monusac_config, _ = load_frozen_monusac_config(root)
    archive = data_root / "monusac_train.zip"
    if sha256_file(archive) != str(monusac_config["authority"]["train_archive_sha256"]):
        raise RuntimeError("official MoNuSAC training archive differs from its authority")
    train_root = data_root / "train" / "MoNuSAC_images_and_annotations"
    exclusions = tuple(str(value) for value in config["data"]["excluded_patients"])
    prepared_64 = prepare_monusac_split(
        train_root, split="train", crop_size=64, excluded_patients=exclusions
    )
    prepared_128 = prepare_monusac_split(
        train_root, split="train", crop_size=128, excluded_patients=exclusions
    )
    validate_aligned_scales(prepared_64, prepared_128)
    embedding_root = root / "artifacts" / "embeddings" / "aanca_expanded"
    resnet_64 = extract_scaled_resnet18_embeddings(
        prepared_64,
        cache_path=embedding_root / "monusac_train_resnet18_context_64.npz",
        scale_px=64,
        device=args.device,
    )
    resnet_128 = extract_scaled_resnet18_embeddings(
        prepared_128,
        cache_path=embedding_root / "monusac_train_resnet18_context_128.npz",
        scale_px=128,
        device=args.device,
    )
    feature_views = {
        "resnet18_multiscale_64_128": np.concatenate((resnet_64, resnet_128), axis=1).astype(
            np.float32, copy=False
        )
    }
    partition = build_all_development_partition(prepared_64, config)
    evaluator = ExpandedAutoresearchEvaluator(
        prepared_64,
        feature_views,
        partition,
        config,
        config_sha256=config_sha256,
    )

    fit_records: list[dict[str, Any]] = []
    original_hard = autoresearch_module.MultinomialLogisticRegression
    original_soft = autoresearch_module.SoftTargetMultinomialLogisticRegression

    class RecordingHardModel(original_hard):  # type: ignore[misc, valid-type]
        def fit(self, features: Any, labels: Any) -> RecordingHardModel:
            super().fit(features, labels)
            fit_records.append(
                {
                    "kind": "hard_label",
                    "training_rows": len(features),
                    "feature_count": int(np.asarray(features).shape[1]),
                    "l2": self.l2,
                    "class_weight_balanced": self.class_weight_balanced,
                    "max_iter": self.max_iter,
                    "converged": self.converged_,
                }
            )
            return self

    class RecordingSoftModel(OriginalSoftModel):
        def fit_soft_targets(
            self,
            features: Any,
            target_probabilities: Any,
            *,
            sample_weight: Any = None,
        ) -> RecordingSoftModel:
            super().fit_soft_targets(
                features,
                target_probabilities,
                sample_weight=sample_weight,
            )
            weights = np.asarray(sample_weight, dtype=np.float64)
            fit_records.append(
                {
                    "kind": "weighted_soft_target",
                    "training_rows": len(features),
                    "feature_count": int(np.asarray(features).shape[1]),
                    "l2": self.l2,
                    "class_weight_balanced": self.class_weight_balanced,
                    "max_iter": self.max_iter,
                    "zero_weight_rows": int(np.sum(weights == 0.0)),
                    "converged": self.converged_,
                }
            )
            return self

    autoresearch_module.MultinomialLogisticRegression = RecordingHardModel  # type: ignore[misc]
    autoresearch_module.SoftTargetMultinomialLogisticRegression = (  # type: ignore[misc]
        RecordingSoftModel
    )
    try:
        result = evaluator.evaluate_full_nested(frozen.candidate)
    finally:
        autoresearch_module.MultinomialLogisticRegression = original_hard  # type: ignore[misc]
        autoresearch_module.SoftTargetMultinomialLogisticRegression = (  # type: ignore[misc]
            original_soft
        )

    evidence = frozen.development_evidence
    downstream = result["downstream"]
    retrieval = result["retrieval"]
    if not isinstance(downstream, dict) or not isinstance(retrieval, dict):
        raise RuntimeError("selected candidate produced incomplete verification evidence")
    for name in (
        "candidate_macro_f1",
        "uncorrected_macro_f1",
        "mean_matched_random_macro_f1",
        "candidate_minus_uncorrected_macro_f1",
        "candidate_minus_matched_random_macro_f1",
    ):
        _assert_close(float(downstream[name]), float(evidence[name]), name)
    _assert_close(
        float(retrieval["candidate_minus_matched_random_precision"]),
        float(evidence["candidate_minus_matched_random_precision"]),
        "candidate_minus_matched_random_precision",
    )

    all_converged = bool(fit_records) and all(bool(record["converged"]) for record in fit_records)
    payload = {
        "schema_version": 1,
        "candidate_sha256": frozen.candidate_sha256,
        "candidate_config_sha256": sha256_file(candidate_path),
        "parent_config_sha256": config_sha256,
        "partition_sha256": partition.partition_sha256,
        "parent_ledger_sha256": sha256_file(parent_root / "results.jsonl"),
        "fit_count": len(fit_records),
        "hard_label_fit_count": sum(record["kind"] == "hard_label" for record in fit_records),
        "weighted_fit_count": sum(
            record["kind"] == "weighted_soft_target" for record in fit_records
        ),
        "all_models_converged": all_converged,
        "fits": fit_records,
        "reproduced_metrics": {
            "candidate_macro_f1": downstream["candidate_macro_f1"],
            "uncorrected_macro_f1": downstream["uncorrected_macro_f1"],
            "mean_matched_random_macro_f1": downstream["mean_matched_random_macro_f1"],
            "candidate_minus_uncorrected_macro_f1": downstream[
                "candidate_minus_uncorrected_macro_f1"
            ],
            "candidate_minus_matched_random_macro_f1": downstream[
                "candidate_minus_matched_random_macro_f1"
            ],
            "candidate_minus_matched_random_precision": retrieval[
                "candidate_minus_matched_random_precision"
            ],
        },
        "final_external_test_used": False,
        "natural_error_detection_evaluated": False,
        "source_annotations_modified": False,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
