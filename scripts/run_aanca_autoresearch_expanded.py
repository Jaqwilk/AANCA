"""Run the fixed expanded AANCA development search without external-test inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from histo_audit.external_validation.monusac import (
    load_frozen_monusac_config,
    prepare_monusac_split,
)
from histo_audit.research.autoresearch import (
    AutoresearchCandidate,
    generate_downstream_candidates,
    generate_ranking_candidates,
    select_full_nested_finalists,
    select_passing_winner,
    select_ranking_finalists,
)
from histo_audit.research.expanded_autoresearch import (
    ExpandedAutoresearchEvaluator,
    build_all_development_partition,
    build_expanded_feature_views,
    extract_phikon_v2_embeddings,
    extract_scaled_resnet18_embeddings,
    load_expanded_autoresearch_config,
    validate_aligned_scales,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _append_record(jsonl_path: Path, tsv_path: Path, record: Mapping[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    with jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not tsv_path.exists():
        with tsv_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "candidate\tstage\tobjective\tstatus\telapsed_seconds\t"
                "downstream_delta\tretrieval_delta\tdescription\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    downstream = record.get("downstream") or {}
    retrieval = record.get("retrieval") or {}
    candidate = record.get("candidate") or {}
    objective = record.get("objective")
    objective_text = "NA" if record.get("status") == "crash" else f"{float(objective or 0.0):.12f}"
    description = ",".join(
        f"{name}={candidate.get(name)}"
        for name in (
            "feature_view",
            "risk_method",
            "neighbour_k",
            "queue_preset",
            "review_budget",
            "intervention",
            "downstream_l2",
            "downstream_class_weight_balanced",
        )
    )
    row = "\t".join(
        (
            str(record.get("candidate_sha256", ""))[:12],
            str(record.get("stage", "")),
            objective_text,
            str(record.get("status", "")),
            f"{float(record.get('elapsed_seconds', 0.0)):.3f}",
            f"{float(downstream.get('candidate_minus_uncorrected_macro_f1', 0.0)):.12f}",
            f"{float(retrieval.get('candidate_minus_matched_random_precision', 0.0)):.12f}",
            description,
        )
    )
    with tsv_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(row + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expanded ledger row {line_number} is not an object")
        records.append(value)
    return records


def _execute_candidates(
    evaluator: ExpandedAutoresearchEvaluator,
    candidates: Sequence[AutoresearchCandidate],
    *,
    stage: str,
    jsonl_path: Path,
    tsv_path: Path,
    existing: list[dict[str, Any]],
    fixed_trial_budget_seconds: float,
) -> list[dict[str, Any]]:
    completed = {
        (str(record.get("stage")), str(record.get("candidate_sha256"))): record
        for record in existing
    }
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        key = (stage, candidate.candidate_sha256)
        if key in completed:
            output.append(completed[key])
            continue
        started = time.perf_counter()
        try:
            if stage == "ranking_screen":
                record = evaluator.evaluate_ranking(candidate)
            elif stage == "downstream_screen":
                record = evaluator.evaluate_downstream_screen(candidate)
            elif stage == "full_nested":
                record = evaluator.evaluate_full_nested(candidate)
            else:
                raise ValueError(f"unsupported expanded search stage: {stage}")
            elapsed = record.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)):
                raise RuntimeError("expanded trial did not report elapsed seconds")
            if float(elapsed) > fixed_trial_budget_seconds:
                record["status"] = "timeout"
                record["all_success_gates_pass"] = False
                record["timeout_reason"] = (
                    "trial completed after the fixed budget and cannot advance"
                )
        except Exception as error:
            record = {
                "schema_version": 1,
                "study_id": evaluator.config["study_id"],
                "stage": stage,
                "candidate": candidate.as_dict(),
                "candidate_sha256": candidate.candidate_sha256,
                "config_sha256": evaluator.config_sha256,
                "partition_sha256": evaluator.partition.partition_sha256,
                "status": "crash",
                "objective": -1.0e308,
                "all_success_gates_pass": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
                "source_annotations_modified": False,
                "final_external_test_used": False,
                "natural_error_detection_evaluated": False,
            }
        _append_record(jsonl_path, tsv_path, record)
        existing.append(record)
        completed[key] = record
        output.append(record)
        if index == 1 or index % 10 == 0 or index == len(candidates):
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "completed": index,
                        "total": len(candidates),
                        "candidate": candidate.short_id,
                        "status": record["status"],
                        "objective": record.get("objective"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return output


def _stage_counts(records: Sequence[Mapping[str, Any]], stage: str) -> dict[str, int]:
    values = [record for record in records if record.get("stage") == stage]
    return {
        "total": len(values),
        **{
            status: sum(record.get("status") == status for record in values)
            for status in (
                "keep",
                "discard",
                "crash",
                "timeout",
            )
        },
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AANCA expanded development autoresearch",
        "",
        f"**Study:** `{summary['study_id']}`  ",
        f"**Executable action:** `{summary['executable_action']}`  ",
        f"**Development disposition:** `{summary['development_disposition']}`  ",
        "**External or previous lockbox outcome used for selection:** `False`",
        "",
        "The study used all eligible official MoNuSAC training patients inside nested",
        "development folds. It has no confirmation set and cannot establish natural",
        "pathologist errors or real clinical/operational utility.",
        "",
        "## Search accounting",
        "",
    ]
    for stage, values in summary["stage_counts"].items():
        lines.append(
            f"- `{stage}`: {values['total']} trials; {values['keep']} keep, "
            f"{values['discard']} discard, {values['crash']} crash, "
            f"{values['timeout']} timeout."
        )
    selected = summary.get("selected_candidate")
    if selected is None:
        lines.extend(
            [
                "",
                "## Selection",
                "",
                "No candidate passed every full nested gate. The fail-closed executable",
                "policy therefore remains `retain_uncorrected`.",
            ]
        )
    else:
        result = summary["selected_full_result"]
        retrieval = result["retrieval"]
        downstream = result["downstream"]
        lines.extend(
            [
                "",
                "## Frozen development candidate",
                "",
                f"Candidate SHA-256: `{summary['selected_candidate_sha256']}`",
                "",
                "```json",
                json.dumps(selected, indent=2, sort_keys=True),
                "```",
                "",
                "## Nested development evidence",
                "",
                "- precision minus matched random: "
                f"`{retrieval['candidate_minus_matched_random_precision']:+.6f}`, "
                f"95% interval `{retrieval['interval_95']}`;",
                "- macro-F1 minus unchanged: "
                f"`{downstream['candidate_minus_uncorrected_macro_f1']:+.6f}`, "
                f"95% interval `{downstream['candidate_minus_uncorrected_interval_95']}`;",
                "- macro-F1 minus matched random: "
                f"`{downstream['candidate_minus_matched_random_macro_f1']:+.6f}`, "
                f"95% interval `{downstream['candidate_minus_matched_random_interval_95']}`;",
                f"- all registered development gates passed: `{result['all_success_gates_pass']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Representation limitation",
            "",
            "Phikon-v2 is restricted to non-commercial research and included TCGA in",
            "pretraining. Since MoNuSAC uses TCGA material, its result is development-only",
            "and cannot be counted as independent confirmation.",
            "",
            "## Claim boundary",
            "",
            "A newly untouched external cohort plus a blinded, multi-rater, multi-site",
            "prospective study remain necessary before any natural-error or real-use claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_train_archive(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"official MoNuSAC training archive is absent: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"official MoNuSAC training SHA-256 differs: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/monusac"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/autoresearch/monusac_aanca_expanded_development_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare pinned representations and evaluator authority without outcomes.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    data_root = (repository_root / args.data_root).resolve()
    output_root = (repository_root / args.output_root).resolve()
    config, config_sha256 = load_expanded_autoresearch_config(repository_root)
    frozen_monusac_config, _ = load_frozen_monusac_config(repository_root)
    _verify_train_archive(
        data_root / "monusac_train.zip",
        str(frozen_monusac_config["authority"]["train_archive_sha256"]),
    )
    train_root = data_root / "train" / "MoNuSAC_images_and_annotations"
    exclusions = tuple(str(value) for value in config["data"]["excluded_patients"])
    prepared_64 = prepare_monusac_split(
        train_root, split="train", crop_size=64, excluded_patients=exclusions
    )
    prepared_128 = prepare_monusac_split(
        train_root, split="train", crop_size=128, excluded_patients=exclusions
    )
    validate_aligned_scales(prepared_64, prepared_128)
    embedding_root = repository_root / "artifacts" / "embeddings" / "aanca_expanded"
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
    phikon, phikon_metadata = extract_phikon_v2_embeddings(
        prepared_128,
        cache_path=embedding_root / "monusac_train_phikon_v2_context_128.npz",
        authority=config["representations"]["phikon_v2"],
        device=args.device,
    )
    feature_views = build_expanded_feature_views(
        prepared_64, prepared_128, resnet_64, resnet_128, phikon
    )
    required_views = tuple(str(value) for value in config["representations"]["required"])
    if set(feature_views) != set(required_views):
        raise RuntimeError("materialised feature views differ from the frozen config")
    partition = build_all_development_partition(prepared_64, config)
    output_root.mkdir(parents=True, exist_ok=True)
    partition_payload = partition.as_dict(prepared_64.manifest["sample_id"].astype(str).tolist())
    partition_path = output_root / "partition.json"
    if partition_path.exists():
        if json.loads(partition_path.read_text(encoding="utf-8")) != partition_payload:
            raise RuntimeError("existing expanded partition differs; refusing to overwrite")
    else:
        atomic_write_json(partition_path, partition_payload)

    authority = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "protocol_sha256": sha256_file(repository_root / "AANCA_AUTORESEARCH_EXPANDED_PROTOCOL.md"),
        "program_sha256": sha256_file(repository_root / "AANCA_AUTORESEARCH_PROGRAM.md"),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "expanded_evaluator_source_sha256": sha256_file(
            repository_root / "src" / "histo_audit" / "research" / "expanded_autoresearch.py"
        ),
        "base_evaluator_source_sha256": sha256_file(
            repository_root / "src" / "histo_audit" / "research" / "autoresearch.py"
        ),
        "evaluation_dependency_sha256": {
            relative: sha256_file(repository_root / relative)
            for relative in (
                "src/histo_audit/auditing/scores.py",
                "src/histo_audit/auditing/strategies.py",
                "src/histo_audit/auditing/two_queue.py",
                "src/histo_audit/corruption/controlled.py",
                "src/histo_audit/cross_validation/oof.py",
                "src/histo_audit/evaluation/restoration.py",
                "src/histo_audit/evaluation/review_training.py",
                "src/histo_audit/external_validation/monusac.py",
                "src/histo_audit/representations/imagenet.py",
                "src/histo_audit/statistics/review.py",
            )
        },
        "concept_source_commit": config["concept_source"]["pinned_commit"],
        "partition_sha256": partition.partition_sha256,
        "train_manifest_sha256": prepared_64.manifest_sha256,
        "train_source_inventory_sha256": prepared_64.source_inventory_sha256,
        "scale_crop_sha256": {
            "64": prepared_64.crops_sha256,
            "128": prepared_128.crops_sha256,
        },
        "feature_views": {
            name: {
                "samples": values.shape[0],
                "dimensions": values.shape[1],
                "dtype": str(values.dtype),
                "sha256": _array_sha256(values),
            }
            for name, values in feature_views.items()
        },
        "phikon_v2_provenance": phikon_metadata,
        "permitted_split": "official_train_only",
        "forbidden_inputs": config["data"]["forbidden_inputs"],
        "previous_internal_lockbox_used_for_selection": False,
        "final_external_test_used": False,
        "natural_error_detection_evaluated": False,
    }
    authority_path = output_root / "run_authority.json"
    if authority_path.exists():
        if json.loads(authority_path.read_text(encoding="utf-8")) != authority:
            raise RuntimeError("existing expanded authority differs; use a new output root")
    else:
        atomic_write_json(authority_path, authority)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "study_id": config["study_id"],
                    "status": "prepared_without_outcome_metrics",
                    "partition_sha256": partition.partition_sha256,
                    "development_groups": len(partition.discovery_groups),
                    "feature_views": {
                        name: list(values.shape) for name, values in feature_views.items()
                    },
                    "output_root": output_root.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    evaluator = ExpandedAutoresearchEvaluator(
        prepared_64,
        feature_views,
        partition,
        config,
        config_sha256=config_sha256,
    )
    jsonl_path = output_root / "results.jsonl"
    tsv_path = output_root / "results.tsv"
    records = _load_records(jsonl_path)
    for record in records:
        if (
            record.get("config_sha256") != config_sha256
            or record.get("partition_sha256") != partition.partition_sha256
        ):
            raise RuntimeError("expanded ledger contains a foreign evaluator identity")
    budget = float(config["successive_halving"]["fixed_trial_budget_seconds"])
    ranking_candidates = generate_ranking_candidates(config, tuple(feature_views))
    ranking_records = _execute_candidates(
        evaluator,
        ranking_candidates,
        stage="ranking_screen",
        jsonl_path=jsonl_path,
        tsv_path=tsv_path,
        existing=records,
        fixed_trial_budget_seconds=budget,
    )
    ranking_finalists = select_ranking_finalists(ranking_records, config)
    downstream_candidates = generate_downstream_candidates(ranking_finalists, config)
    downstream_records = _execute_candidates(
        evaluator,
        downstream_candidates,
        stage="downstream_screen",
        jsonl_path=jsonl_path,
        tsv_path=tsv_path,
        existing=records,
        fixed_trial_budget_seconds=budget,
    )
    full_candidates = select_full_nested_finalists(downstream_records, config)
    full_records = _execute_candidates(
        evaluator,
        full_candidates,
        stage="full_nested",
        jsonl_path=jsonl_path,
        tsv_path=tsv_path,
        existing=records,
        fixed_trial_budget_seconds=budget,
    )
    winner = select_passing_winner(full_records)
    selected_record: dict[str, Any] | None = None
    if winner is not None:
        selected_record = next(
            record
            for record in full_records
            if record.get("candidate_sha256") == winner.candidate_sha256
            and record.get("all_success_gates_pass") is True
        )
        freeze = {
            "schema_version": 1,
            "study_id": config["study_id"],
            "candidate": winner.as_dict(),
            "candidate_sha256": winner.candidate_sha256,
            "config_sha256": config_sha256,
            "partition_sha256": partition.partition_sha256,
            "role": "development_candidate_only",
            "external_confirmation_complete": False,
            "natural_error_detection_proven": False,
            "real_use_superiority_proven": False,
            "automatic_annotation_change_permitted": False,
            "executable_action_until_new_confirmation": "retain_uncorrected",
        }
        freeze_path = output_root / "selected_development_candidate.json"
        if freeze_path.exists():
            if json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
                raise RuntimeError("selected development freeze differs; refusing overwrite")
        else:
            atomic_write_json(freeze_path, freeze)

    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "partition_sha256": partition.partition_sha256,
        "stage_counts": {
            stage: _stage_counts(records, stage)
            for stage in ("ranking_screen", "downstream_screen", "full_nested")
        },
        "selected_candidate": winner.as_dict() if winner else None,
        "selected_candidate_sha256": winner.candidate_sha256 if winner else None,
        "selected_full_result": selected_record,
        "development_disposition": (
            "candidate_frozen_for_genuinely_new_external_validation"
            if winner
            else "no_candidate_passed_all_nested_development_gates"
        ),
        "executable_action": "retain_uncorrected",
        "previous_internal_lockbox_used_for_selection": False,
        "final_external_test_used": False,
        "natural_error_detection_proven": False,
        "real_use_superiority_proven": False,
        "source_annotations_modified": False,
    }
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(output_root / "report.md", _render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
