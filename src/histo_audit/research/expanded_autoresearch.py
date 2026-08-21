"""Expanded post-lockbox development search without external-test access.

The first autoresearch evaluator remains untouched for provenance.  This module
adds all-development nested folds, multiscale/foundation representations, an
exact-comparator-capacity selector, and trusted reviewed-row weighting.  It has no
loader or argument for MoNuSAC test, NuCLS, or any other final outcome source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from histo_audit.config import load_config_with_file_sha256
from histo_audit.cross_validation.oof import make_group_stratified_fold_plan
from histo_audit.external_validation.monusac import CLASS_ORDER, MoNuSACPreparedData
from histo_audit.representations.imagenet import (
    ResNet18EmbeddingConfig,
    extract_resnet18_embeddings,
    load_embedding_cache,
)
from histo_audit.utils.run_tracking import atomic_write_json, atomic_write_npz, sha256_file

from .autoresearch import (
    AutoresearchCandidate,
    AutoresearchEvaluator,
    AutoresearchPartition,
    _AuditEvidence,
    _crop_statistics,
    _semantic_sha256,
    _validate_training_only,
)


def load_expanded_autoresearch_config(
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load the expanded study config and reject every unsafe claim boundary."""

    path = (
        Path(repository_root).resolve() / "configs" / "aanca_autoresearch_expanded_development.yaml"
    )
    config, digest = load_config_with_file_sha256(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expanded AANCA autoresearch configuration is malformed")
    if config.get("disposition") != "post_lockbox_expanded_development_only_method_search":
        raise ValueError("expanded AANCA autoresearch must remain development-only")
    data = config.get("data")
    partition = config.get("partition")
    boundary = config.get("claim_boundary")
    if not isinstance(data, Mapping) or data.get("permitted_split") != "official_train_only":
        raise ValueError("expanded autoresearch may consume only official MoNuSAC training data")
    if (
        not isinstance(partition, Mapping)
        or partition.get("internal_lockbox_enabled") is not False
        or partition.get("all_official_train_patients_are_development_only") is not True
    ):
        raise ValueError("expanded autoresearch cannot create or reuse an internal lockbox")
    required_false = (
        "natural_error_detection_claim_permitted",
        "pathologist_error_claim_permitted",
        "clinical_or_operational_utility_claim_permitted",
        "automatic_annotation_change_permitted",
        "internal_lockbox_is_external_confirmation",
        "controlled_corruption_result_is_natural_error_evidence",
    )
    if not isinstance(boundary, Mapping) or any(
        boundary.get(name) is not False for name in required_false
    ):
        raise ValueError("expanded autoresearch claim boundary is not fail-closed")
    phikon = config.get("representations", {}).get("phikon_v2", {})
    if (
        phikon.get("revision") != "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd"
        or phikon.get("model_safetensors_sha256")
        != "261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569"
        or phikon.get("model_lfs_object_oid")
        != "941e973207791afdaad9d3b4c4773429ecf6b93095835307d4a83c9c9ba81ce0"
        or phikon.get("permitted_use") != "non_commercial_university_research_only"
        or phikon.get("processor_backend") != "saved_slow_processor_use_fast_false"
        or phikon.get("tcga_pretraining_overlap_limitation") is not True
    ):
        raise ValueError("Phikon-v2 provenance or research-use limitation is not pinned")
    return config, digest


def build_all_development_partition(
    prepared: MoNuSACPreparedData, config: Mapping[str, Any]
) -> AutoresearchPartition:
    """Use every eligible official-train patient inside nested development folds."""

    _validate_training_only(prepared, config)
    labels = prepared.manifest["reference_label"].to_numpy(dtype=np.int64)
    groups = prepared.manifest["group_id"].astype(str).tolist()
    sample_ids = prepared.manifest["sample_id"].astype(str).tolist()
    seed = int(config["partition"]["seed"])
    discovery = np.arange(len(prepared.manifest), dtype=np.int64)
    plan = make_group_stratified_fold_plan(
        labels,
        groups,
        n_splits=int(config["partition"]["discovery_outer_folds"]),
        class_order=tuple(range(len(CLASS_ORDER))),
        seed=seed,
    )
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    coverage = np.zeros(len(discovery), dtype=np.int64)
    for fold in plan.folds:
        train = np.asarray(fold.train_indices, dtype=np.int64)
        validation = np.asarray(fold.holdout_indices, dtype=np.int64)
        if set(groups[index] for index in train).intersection(
            groups[index] for index in validation
        ):
            raise RuntimeError("expanded autoresearch outer patient-group leakage")
        coverage[validation] += 1
        folds.append((train, validation))
    if not np.array_equal(coverage, np.ones(len(discovery), dtype=np.int64)):
        raise RuntimeError("expanded autoresearch folds do not cover development exactly once")
    authority = {
        "study_id": str(config["study_id"]),
        "split_seed": seed,
        "role": "all_official_train_development_no_lockbox",
        "sample_ids": sample_ids,
        "outer_folds": [
            {
                "train": [sample_ids[index] for index in train],
                "validation": [sample_ids[index] for index in validation],
            }
            for train, validation in folds
        ],
    }
    return AutoresearchPartition(
        discovery_indices=discovery,
        lockbox_indices=np.empty(0, dtype=np.int64),
        discovery_outer_folds=tuple(folds),
        discovery_groups=tuple(sorted(set(groups))),
        lockbox_groups=(),
        partition_sha256=_semantic_sha256(authority),
        split_seed=seed,
    )


def validate_aligned_scales(
    context_64: MoNuSACPreparedData, context_128: MoNuSACPreparedData
) -> None:
    """Prove that independently prepared crop scales address identical nuclei."""

    if (
        context_64.split != "train"
        or context_128.split != "train"
        or context_64.manifest_sha256 != context_128.manifest_sha256
        or context_64.source_inventory_sha256 != context_128.source_inventory_sha256
        or not context_64.manifest.equals(context_128.manifest)
        or context_64.crops.shape[1:] != (64, 64, 3)
        or context_128.crops.shape[1:] != (128, 128, 3)
    ):
        raise RuntimeError("multiscale MoNuSAC preparations are not canonically aligned")


def extract_scaled_resnet18_embeddings(
    prepared: MoNuSACPreparedData,
    *,
    cache_path: str | Path,
    scale_px: int,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Extract one scale with crop identity included in persisted provenance."""

    if prepared.crops.shape[1:] != (scale_px, scale_px, 3):
        raise ValueError("prepared crops differ from the requested ResNet-18 scale")
    destination = Path(cache_path).resolve()
    expected_ids = prepared.manifest["sample_id"].astype(str).tolist()
    representation_id = f"monusac_train_resnet18_imagenet1k_v1_context_{scale_px}px_expanded_search"
    if destination.is_file():
        cached = load_embedding_cache(destination)
        cached.validate()
        eligibility = cached.metadata.get("analysis_eligibility")
        if (
            cached.sample_ids.tolist() != expected_ids
            or cached.metadata.get("manifest_sha256") != prepared.manifest_sha256
            or cached.metadata.get("raw_inventory_sha256") != prepared.source_inventory_sha256
            or cached.metadata.get("representation_id") != representation_id
            or not isinstance(eligibility, Mapping)
            or eligibility.get("source_crops_sha256") != prepared.crops_sha256
        ):
            raise RuntimeError("cached scaled ResNet-18 embeddings fail provenance binding")
        return np.asarray(cached.embeddings, dtype=np.float32)
    result = extract_resnet18_embeddings(
        prepared.crops,
        expected_ids,
        config=ResNet18EmbeddingConfig(
            weight_identifier="IMAGENET1K_V1",
            input_variant="rgb",
            device=device,
            batch_size=64,
            minimum_batch_size=1,
            use_amp=True,
            output_dtype="float32",
            allow_weight_download=False,
        ),
        cache_path=destination,
        manifest_sha256=prepared.manifest_sha256,
        raw_inventory_sha256=prepared.source_inventory_sha256,
        representation_id=representation_id,
        analysis_eligibility={
            "study_id": "monusac_aanca_expanded_development_v1",
            "eligible": True,
            "sample_count": len(prepared.manifest),
            "source_context_px": scale_px,
            "source_crops_sha256": prepared.crops_sha256,
            "final_external_test_used": False,
        },
    )
    result.validate()
    return np.asarray(result.embeddings, dtype=np.float32)


def _array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _load_phikon_cache(
    destination: Path,
    prepared: MoNuSACPreparedData,
    authority: Mapping[str, Any],
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    sidecar = destination.with_suffix(f"{destination.suffix}.metadata.json")
    if not destination.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Phikon-v2 cache or provenance sidecar is absent")
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(destination, allow_pickle=False) as payload:
        if set(payload.files) != {"embeddings", "sample_ids", "metadata_json"}:
            raise ValueError("Phikon-v2 cache contains unexpected arrays")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        sample_ids = payload["sample_ids"].astype(str)
        embedded = json.loads(str(payload["metadata_json"].item()))
    expected_ids = prepared.manifest["sample_id"].astype(str).tolist()
    if (
        recorded.get("cache_sha256") != sha256_file(destination)
        or recorded.get("metadata") != embedded
        or sample_ids.tolist() != expected_ids
        or embeddings.shape != (len(prepared.manifest), int(authority["output_dimension"]))
        or not np.isfinite(embeddings).all()
        or embedded.get("embeddings_sha256") != _array_sha256(embeddings)
        or embedded.get("sample_ids_sha256") != _array_sha256(sample_ids)
        or embedded.get("manifest_sha256") != prepared.manifest_sha256
        or embedded.get("raw_inventory_sha256") != prepared.source_inventory_sha256
        or embedded.get("source_crops_sha256") != prepared.crops_sha256
        or embedded.get("model_revision") != authority["revision"]
        or embedded.get("model_safetensors_sha256") != authority["model_safetensors_sha256"]
    ):
        raise RuntimeError("Phikon-v2 cache fails exact data/model provenance verification")
    return embeddings, embedded


def extract_phikon_v2_embeddings(
    prepared: MoNuSACPreparedData,
    *,
    cache_path: str | Path,
    authority: Mapping[str, Any],
    device: str = "auto",
    initial_batch_size: int = 16,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Extract pinned public Phikon-v2 CLS embeddings for research-only use."""

    if prepared.crops.shape[1:] != (
        int(authority["source_context_px"]),
        int(authority["source_context_px"]),
        3,
    ):
        raise ValueError("Phikon-v2 source context differs from the frozen protocol")
    destination = Path(cache_path).resolve()
    if destination.is_file():
        return _load_phikon_cache(destination, prepared, authority)
    if initial_batch_size <= 0:
        raise ValueError("Phikon-v2 batch size must be positive")
    try:
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoImageProcessor,
            AutoModel,
        )
    except ImportError as error:
        raise RuntimeError(
            "Phikon-v2 extraction requires the pinned research dependency group"
        ) from error

    model_id = str(authority["repository"])
    revision = str(authority["revision"])
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "LICENSE.pdf",
            ],
        )
    ).resolve()
    model_path = snapshot / "model.safetensors"
    licence_path = snapshot / "LICENSE.pdf"
    config_path = snapshot / "config.json"
    preprocessing_path = snapshot / "preprocessor_config.json"
    for required in (model_path, licence_path, config_path, preprocessing_path):
        if not required.is_file():
            raise FileNotFoundError(f"pinned Phikon-v2 snapshot lacks {required.name}")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != str(
        authority["model_safetensors_sha256"]
    ) or model_path.stat().st_size != int(authority["model_safetensors_bytes"]):
        raise RuntimeError("downloaded Phikon-v2 weights differ from the frozen authority")

    requested_device = device.strip().lower()
    resolved = torch.device(
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else ("cpu" if requested_device == "auto" else requested_device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Phikon-v2 but is unavailable")
    dtype = torch.float16 if resolved.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=False
    )
    model = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    model.eval().to(device=resolved, dtype=dtype)
    output_dimension = int(authority["output_dimension"])
    embeddings = np.empty((len(prepared.manifest), output_dimension), dtype=np.float32)
    batch_size = initial_batch_size
    cursor = 0
    while cursor < len(prepared.crops):
        stop = min(cursor + batch_size, len(prepared.crops))
        try:
            inputs = processor(
                images=[prepared.crops[index] for index in range(cursor, stop)],
                return_tensors="pt",
            )
            pixel_values = cast(torch.Tensor, inputs["pixel_values"]).to(
                device=resolved, dtype=dtype
            )
            with torch.inference_mode():
                result = model(pixel_values=pixel_values)
            cls = result.last_hidden_state[:, 0, :].float().cpu().numpy()
        except RuntimeError as error:
            is_oom = (
                isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
            )
            if not is_oom or resolved.type != "cuda" or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue
        if cls.shape != (stop - cursor, output_dimension) or not np.isfinite(cls).all():
            raise RuntimeError("Phikon-v2 produced invalid CLS embeddings")
        embeddings[cursor:stop] = cls.astype(np.float32, copy=False)
        cursor = stop
    sample_ids = prepared.manifest["sample_id"].astype(str).to_numpy(dtype=np.str_)
    metadata = {
        "schema_version": 1,
        "study_id": "monusac_aanca_expanded_development_v1",
        "representation_id": "phikon_v2_context_128_pinned_research_only",
        "model_repository": model_id,
        "model_revision": revision,
        "model_safetensors_sha256": model_sha256,
        "model_safetensors_bytes": model_path.stat().st_size,
        "model_config_sha256": sha256_file(config_path),
        "preprocessor_config_sha256": sha256_file(preprocessing_path),
        "processor_backend": str(authority["processor_backend"]),
        "licence": str(authority["licence"]),
        "licence_sha256": sha256_file(licence_path),
        "permitted_use": str(authority["permitted_use"]),
        "tcga_pretraining_overlap_limitation": True,
        "manifest_sha256": prepared.manifest_sha256,
        "raw_inventory_sha256": prepared.source_inventory_sha256,
        "source_crops_sha256": prepared.crops_sha256,
        "source_context_px": int(authority["source_context_px"]),
        "sample_ids_sha256": _array_sha256(sample_ids),
        "embeddings_sha256": _array_sha256(embeddings),
        "output_dimension": output_dimension,
        "device": str(resolved),
        "effective_batch_size": batch_size,
        "labels_consumed": False,
        "frozen": True,
        "source_annotations_modified": False,
        "final_external_test_used": False,
        "package_versions": {
            "numpy": _package_version("numpy"),
            "torch": str(torch.__version__),
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
        },
    }
    atomic_write_npz(
        destination,
        {
            "embeddings": embeddings,
            "sample_ids": sample_ids,
            "metadata_json": np.asarray(
                json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
                dtype=np.str_,
            ),
        },
        compressed=False,
    )
    atomic_write_json(
        destination.with_suffix(f"{destination.suffix}.metadata.json"),
        {"cache_sha256": sha256_file(destination), "metadata": metadata},
    )
    del model
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
    return _load_phikon_cache(destination, prepared, authority)


def build_expanded_feature_views(
    context_64: MoNuSACPreparedData,
    context_128: MoNuSACPreparedData,
    resnet_64: NDArray[np.generic],
    resnet_128: NDArray[np.generic],
    phikon_128: NDArray[np.generic],
) -> dict[str, NDArray[np.float32]]:
    """Construct every preregistered, label-independent expanded feature view."""

    validate_aligned_scales(context_64, context_128)
    count = len(context_64.manifest)
    r64 = np.asarray(resnet_64, dtype=np.float32)
    r128 = np.asarray(resnet_128, dtype=np.float32)
    phikon = np.asarray(phikon_128, dtype=np.float32)
    if (
        r64.shape != (count, 512)
        or r128.shape != (count, 512)
        or phikon.shape != (count, 1024)
        or not all(np.isfinite(values).all() for values in (r64, r128, phikon))
    ):
        raise ValueError("expanded embedding matrices are not finite and canonically aligned")
    stats64 = _crop_statistics(context_64)
    stats128 = _crop_statistics(context_128)
    multiscale = np.concatenate((r64, r128), axis=1).astype(np.float32, copy=False)
    views = {
        "resnet18_context_64": r64,
        "resnet18_context_64_plus_stats": np.concatenate((r64, stats64), axis=1),
        "resnet18_context_128": r128,
        "resnet18_context_128_plus_stats": np.concatenate((r128, stats128), axis=1),
        "resnet18_multiscale_64_128": multiscale,
        "resnet18_multiscale_64_128_plus_stats": np.concatenate(
            (multiscale, stats64, stats128), axis=1
        ),
        "phikon_v2_context_128": phikon,
        "phikon_v2_context_128_plus_stats": np.concatenate((phikon, stats128), axis=1),
        "phikon_v2_resnet18_multiscale": np.concatenate((phikon, multiscale), axis=1),
    }
    return {name: np.asarray(values, dtype=np.float32) for name, values in views.items()}


class ExpandedAutoresearchEvaluator(AutoresearchEvaluator):
    """The fixed expanded evaluator with a feasible exact-random control."""

    def _select(
        self,
        evidence: _AuditEvidence,
        risk: NDArray[np.float64],
        candidate: AutoresearchCandidate,
    ) -> tuple[NDArray[np.int64], dict[str, object]]:
        if (
            self.config["controls"].get(
                "enforce_disjoint_exact_comparator_capacity_before_selection"
            )
            is not True
        ):
            raise RuntimeError("expanded selector requires prospective comparator capacity")
        proposed = np.argmax(evidence.probabilities, axis=1).astype(np.int64)
        fields = tuple(str(value) for value in self.config["controls"]["matched_random_fields"])
        transitions = np.asarray(
            [
                f"{source}->{target}"
                for source, target in zip(evidence.observed, proposed, strict=True)
            ],
            dtype=np.str_,
        )
        values: dict[str, Sequence[str | int]] = {
            "observed_class": evidence.observed.tolist(),
            "organ": [self._organs[index] for index in evidence.full_train_indices],
            "proposed_transition": transitions.tolist(),
        }
        strata = [
            tuple(str(values[field][index]) for field in fields) for index in range(len(risk))
        ]
        totals = Counter(strata)
        capacity = {stratum: count // 2 for stratum, count in totals.items()}
        working = np.asarray(risk, dtype=np.float64).copy()
        if not np.isfinite(working).all():
            raise ValueError("expanded selector received non-finite audit risk")
        floor = float(working.min() - max(1.0, float(np.ptp(working)) + 1.0) * 1.0e6)
        blocked: set[int] = {
            index for index, stratum in enumerate(strata) if capacity[stratum] == 0
        }
        requested = max(1, math.ceil(len(risk) * candidate.review_budget))
        if len(risk) - len(blocked) < requested:
            raise RuntimeError("exact-comparator-capable queue cannot fill the review budget")
        for _ in range(requested + 1):
            working[list(blocked)] = floor
            selected, queue = super()._select(evidence, working, candidate)
            if len(selected) != requested or any(int(index) in blocked for index in selected):
                raise RuntimeError("queue exhausted exact-comparator-capable rows")
            counts = Counter(strata[int(index)] for index in selected)
            violations = {
                stratum: count - capacity[stratum]
                for stratum, count in counts.items()
                if count > capacity[stratum]
            }
            if not violations:
                return selected, {
                    **queue,
                    "exact_comparator_capacity_enforced": True,
                    "comparator_ineligible_rows": sum(
                        totals[stratum] for stratum, value in capacity.items() if value == 0
                    ),
                    "comparator_blocked_rows": len(blocked),
                    "matched_random_fields": list(fields),
                }
            for stratum, excess in violations.items():
                members = [int(index) for index in selected if strata[int(index)] == stratum]
                members.sort(
                    key=lambda index: (
                        working[index],
                        self._sample_ids[int(evidence.full_train_indices[index])],
                    )
                )
                blocked.update(members[:excess])
        raise RuntimeError("exact-comparator-capacity selector did not converge")

    @staticmethod
    def _derive_intervention(
        observed: NDArray[np.int64],
        reference: NDArray[np.int64],
        injected: NDArray[np.bool_],
        selected: NDArray[np.int64],
        policy: str,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
        trusted_weights = {
            "controlled_restore_selected_weight_2": 2.0,
            "controlled_restore_selected_weight_4": 4.0,
        }
        if policy not in trusted_weights:
            return AutoresearchEvaluator._derive_intervention(
                observed, reference, injected, selected, policy
            )
        labels = observed.copy()
        weights = np.ones(len(observed), dtype=np.float64)
        restored = selected[injected[selected]]
        labels[restored] = reference[restored]
        weights[selected] = trusted_weights[policy]
        if not np.array_equal(observed, np.asarray(observed, dtype=np.int64)):
            raise RuntimeError("expanded intervention mutated source observed labels")
        return labels, weights, len(restored)


__all__ = [
    "ExpandedAutoresearchEvaluator",
    "build_all_development_partition",
    "build_expanded_feature_views",
    "extract_phikon_v2_embeddings",
    "extract_scaled_resnet18_embeddings",
    "load_expanded_autoresearch_config",
    "validate_aligned_scales",
]
