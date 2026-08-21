"""Run the development-only AANCA autoresearch loop without external-test access."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from histo_audit.external_validation.monusac import (
    extract_monusac_embeddings,
    load_frozen_monusac_config,
    prepare_monusac_split,
)
from histo_audit.research.autoresearch import (
    AutoresearchCandidate,
    AutoresearchEvaluator,
    build_autoresearch_feature_views,
    build_autoresearch_partition,
    generate_downstream_candidates,
    generate_ranking_candidates,
    load_autoresearch_config,
    select_full_nested_finalists,
    select_passing_winner,
    select_ranking_finalists,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_text, sha256_file


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
        )
    )
    row = "\t".join(
        (
            str(record.get("candidate_sha256", ""))[:12],
            str(record.get("stage", "")),
            f"{float(record.get('objective', 0.0)):.12f}",
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
            raise ValueError(f"experiment ledger row {line_number} is not an object")
        records.append(value)
    return records


def _record_key(stage: str, candidate: AutoresearchCandidate) -> tuple[str, str]:
    return stage, candidate.candidate_sha256


def _execute_candidates(
    evaluator: AutoresearchEvaluator,
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
    stage_records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        key = _record_key(stage, candidate)
        if key in completed:
            stage_records.append(completed[key])
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
                raise ValueError(f"unsupported autoresearch stage: {stage}")
            elapsed_seconds = record.get("elapsed_seconds")
            if not isinstance(elapsed_seconds, (int, float)):
                raise RuntimeError("autoresearch trial did not report elapsed seconds")
            if float(elapsed_seconds) > fixed_trial_budget_seconds:
                record["status"] = "timeout"
                record["all_success_gates_pass"] = False
                record["timeout_reason"] = (
                    "trial completed after the fixed development budget; outcome cannot advance"
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
        stage_records.append(record)
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
    return stage_records


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AANCA development autoresearch",
        "",
        f"**Study:** `{summary['study_id']}`  ",
        f"**Executable action:** `{summary['executable_action']}`  ",
        f"**External test used for search:** `{summary['final_external_test_used']}`",
        "",
        "The loop used only official MoNuSAC training patients. It is a controlled,",
        "post-external development search and cannot establish natural pathologist errors",
        "or real clinical utility.",
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
                "No candidate passed every nested development gate. The unchanged model",
                "remains the executable policy and the internal lockbox was not opened.",
            ]
        )
    else:
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
            ]
        )
        lockbox = summary.get("lockbox_result")
        if lockbox is not None:
            downstream = lockbox["downstream"]
            retrieval = lockbox["retrieval"]
            lines.extend(
                [
                    "",
                    "## One-time internal lockbox",
                    "",
                    "- top-K precision minus matched random: "
                    f"`{retrieval['candidate_minus_matched_random_precision']:+.6f}`, "
                    f"95% interval `{retrieval['interval_95']}`;",
                    "- macro-F1 minus unchanged: "
                    f"`{downstream['candidate_minus_uncorrected_macro_f1']:+.6f}`, "
                    f"95% interval "
                    f"`{downstream['candidate_minus_uncorrected_interval_95']}`;",
                    "- macro-F1 minus matched random: "
                    f"`{downstream['candidate_minus_matched_random_macro_f1']:+.6f}`, "
                    f"95% interval "
                    f"`{downstream['candidate_minus_matched_random_interval_95']}`;",
                    f"- all gates passed: `{lockbox['all_success_gates_pass']}`.",
                ]
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Even a favourable internal lockbox result would support only controlled",
            "injected-corruption performance on held-out development patients. A blinded,",
            "multi-rater, multi-site prospective study and a genuinely new external test",
            "remain necessary before any natural-error or real-use superiority claim.",
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
        default=Path("artifacts/autoresearch/monusac_aanca_development_autoresearch_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify training-only inputs and freeze the partition without computing outcomes.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    data_root = (repository_root / args.data_root).resolve()
    output_root = (repository_root / args.output_root).resolve()
    config, config_sha256 = load_autoresearch_config(repository_root)
    frozen_monusac_config, _ = load_frozen_monusac_config(repository_root)
    _verify_train_archive(
        data_root / "monusac_train.zip",
        str(frozen_monusac_config["authority"]["train_archive_sha256"]),
    )
    train_root = data_root / "train" / "MoNuSAC_images_and_annotations"
    prepared = prepare_monusac_split(
        train_root,
        split="train",
        crop_size=64,
        excluded_patients=tuple(str(value) for value in config["data"]["excluded_patients"]),
    )
    embeddings = extract_monusac_embeddings(
        prepared,
        cache_path=(
            repository_root
            / "artifacts"
            / "embeddings"
            / "monusac_current_aanca"
            / "train_resnet18.npz"
        ),
        device=args.device,
    )
    feature_views = build_autoresearch_feature_views(prepared, embeddings)
    partition = build_autoresearch_partition(prepared, config)
    output_root.mkdir(parents=True, exist_ok=True)
    partition_path = output_root / "partition.json"
    partition_payload = partition.as_dict(prepared.manifest["sample_id"].astype(str).tolist())
    if partition_path.exists():
        existing_partition = json.loads(partition_path.read_text(encoding="utf-8"))
        if existing_partition != partition_payload:
            raise RuntimeError("existing autoresearch partition differs; refusing to overwrite")
    else:
        atomic_write_json(partition_path, partition_payload)
    authority = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "protocol_sha256": sha256_file(repository_root / "AANCA_AUTORESEARCH_PROTOCOL.md"),
        "program_sha256": sha256_file(repository_root / "AANCA_AUTORESEARCH_PROGRAM.md"),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_source_sha256": sha256_file(
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
                "src/histo_audit/statistics/review.py",
            )
        },
        "concept_source_commit": config["concept_source"]["pinned_commit"],
        "partition_sha256": partition.partition_sha256,
        "train_manifest_sha256": prepared.manifest_sha256,
        "train_source_inventory_sha256": prepared.source_inventory_sha256,
        "permitted_split": "official_train_only",
        "forbidden_inputs": config["data"]["forbidden_inputs"],
        "feature_views": {
            name: {"samples": values.shape[0], "dimensions": values.shape[1]}
            for name, values in feature_views.items()
        },
        "final_external_test_used": False,
    }
    authority_path = output_root / "run_authority.json"
    if authority_path.exists():
        if json.loads(authority_path.read_text(encoding="utf-8")) != authority:
            raise RuntimeError("existing autoresearch authority differs; use a new output root")
    else:
        atomic_write_json(authority_path, authority)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "study_id": config["study_id"],
                    "status": "prepared_without_outcome_metrics",
                    "partition_sha256": partition.partition_sha256,
                    "discovery_groups": len(partition.discovery_groups),
                    "lockbox_groups": len(partition.lockbox_groups),
                    "output_root": output_root.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    evaluator = AutoresearchEvaluator(
        prepared,
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
            raise RuntimeError("experiment ledger contains a foreign evaluator identity")
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
    freeze_path = output_root / "selected_candidate.freeze.json"
    lockbox_path = output_root / "lockbox_result.json"
    lockbox_result: dict[str, Any] | None = None
    if winner is not None:
        freeze_payload = {
            "schema_version": 1,
            "study_id": config["study_id"],
            "candidate": winner.as_dict(),
            "candidate_sha256": winner.candidate_sha256,
            "config_sha256": config_sha256,
            "partition_sha256": partition.partition_sha256,
            "selection_rule": (
                "all full-nested gates pass; maximise the smaller downstream lower bound, "
                "then point gain, then simplicity"
            ),
            "written_before_internal_lockbox_evaluation": True,
            "final_external_test_used": False,
        }
        if freeze_path.exists():
            existing_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if existing_freeze != freeze_payload:
                raise RuntimeError("selected-candidate freeze differs; lockbox remains closed")
        else:
            atomic_write_json(freeze_path, freeze_payload)
        freeze_sha256 = sha256_file(freeze_path)
        if lockbox_path.exists():
            lockbox_result = json.loads(lockbox_path.read_text(encoding="utf-8"))
            if (
                lockbox_result.get("frozen_candidate_sha256") != winner.candidate_sha256
                or lockbox_result.get("freeze_file_sha256") != freeze_sha256
            ):
                raise RuntimeError("saved lockbox result differs from the frozen candidate")
        else:
            lockbox_result = evaluator.evaluate_lockbox(
                winner,
                frozen_candidate_sha256=winner.candidate_sha256,
                freeze_file_sha256=freeze_sha256,
            )
            atomic_write_json(lockbox_path, lockbox_result)
            _append_record(jsonl_path, tsv_path, lockbox_result)

    stage_counts: dict[str, dict[str, int]] = {}
    for stage in ("ranking_screen", "downstream_screen", "full_nested", "lockbox"):
        stage_values = [
            record for record in _load_records(jsonl_path) if record.get("stage") == stage
        ]
        stage_counts[stage] = {
            "total": len(stage_values),
            **{
                status: sum(record.get("status") == status for record in stage_values)
                for status in ("keep", "discard", "crash", "timeout")
            },
        }
    lockbox_passed = bool(lockbox_result and lockbox_result["all_success_gates_pass"])
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "partition_sha256": partition.partition_sha256,
        "stage_counts": stage_counts,
        "selected_candidate": winner.as_dict() if winner is not None else None,
        "selected_candidate_sha256": winner.candidate_sha256 if winner is not None else None,
        "lockbox_result": lockbox_result,
        "executable_action": (
            "development_candidate_available_but_requires_new_external_validation"
            if lockbox_passed
            else "retain_uncorrected"
        ),
        "final_external_test_used": False,
        "natural_error_detection_proven": False,
        "real_use_superiority_proven": False,
    }
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(output_root / "report.md", _render_report(summary))
    print(
        json.dumps(
            {
                "study_id": summary["study_id"],
                "executable_action": summary["executable_action"],
                "selected_candidate_sha256": summary["selected_candidate_sha256"],
                "lockbox_passed": lockbox_passed,
                "output_root": output_root.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
