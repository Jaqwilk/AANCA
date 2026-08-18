from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histo_audit.pannuke.models import (
    OFFICIAL_METRICS_CLASS_MAPPING,
    DiscoveredFold,
    FoldValidation,
    MaskQCPolicy,
    PanNukeValidationResult,
)
from histo_audit.pannuke.qc import (
    analyse_patch_mask_qc,
    select_anomaly_overlay_patches,
    summarise_fold_mask_qc,
    summarise_global_mask_qc,
)
from histo_audit.pannuke.qc_reporting import (
    OVERLAP_ALPHA,
    MaskQCReportError,
    anomaly_overlay_rgba,
    validate_mask_qc_report_bundle,
    write_mask_qc_report_bundle,
)


def _validation_result(tmp_path: Path) -> PanNukeValidationResult:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    class_names = OFFICIAL_METRICS_CLASS_MAPPING.class_names
    masks = np.zeros((4, 4, 4, 6), dtype=np.uint16)
    masks[..., 5] = 1

    # One cross-class-overlap pixel. Neither channel is selected as the winner.
    masks[0, 0, 0, 0] = 11
    masks[0, 0, 0, 1] = 21
    masks[0, 0, 0, 5] = 0

    # One supplied-background void pixel.
    masks[1, 1, 1, 5] = 0

    # One independently recorded positive+supplied-background pixel.
    masks[2, 2, 2, 2] = 31

    # One ordinary instance with an exact local background complement.
    masks[3, 3, 3, 3] = 41
    masks[3, 3, 3, 5] = 0

    images = np.arange(4 * 4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 4, 3)
    tissues = np.asarray(["a", "b", "c", "d"], dtype="<U1")
    image_path = raw_root / "images.npy"
    mask_path = raw_root / "masks.npy"
    tissue_path = raw_root / "types.npy"
    np.save(image_path, images, allow_pickle=False)
    np.save(mask_path, masks, allow_pickle=False)
    np.save(tissue_path, tissues, allow_pickle=False)

    patch_qc = tuple(
        analyse_patch_mask_qc(
            masks[index],
            fold_id=1,
            patch_index=index,
            class_names=class_names,
            positive_channel_indices=(0, 1, 2, 3, 4),
            background_channel_index=5,
        )
        for index in range(len(masks))
    )
    fold_qc = summarise_fold_mask_qc(1, patch_qc)
    global_qc = summarise_global_mask_qc((fold_qc,))
    discovered = DiscoveredFold(
        fold_id=1,
        image_path=image_path,
        mask_path=mask_path,
        tissue_path=tissue_path,
        image_channel_axis=3,
        mask_channel_axis=3,
    )
    validation = FoldValidation(
        fold_id=1,
        n_patches=4,
        height=4,
        width=4,
        image_shape=images.shape,
        image_dtype=str(images.dtype),
        image_range=(0.0, float(images.max())),
        mask_shape=masks.shape,
        mask_dtype=str(masks.dtype),
        mask_range=(0.0, float(masks.max())),
        tissue_shape=tissues.shape,
        tissue_dtype=str(tissues.dtype),
        tissue_values=tuple(tissues.tolist()),
        positive_channel_indices=(0, 1, 2, 3, 4),
        background_channel_index=5,
        background_channel_candidates=(5,),
        validation_scope="full_semantic_scan",
        full_scan_patch_count=4,
        full_scan_instance_count=4,
        sampled_patch_indices=(0, 1, 2, 3),
        sampled_instance_ids_by_class={},
        overlap_pixel_count_sampled=1,
        malformed_instance_count_sampled=0,
        mask_qc=fold_qc,
    )
    policy = MaskQCPolicy(
        policy_version="pannuke-mask-qc-v1",
        positive_channel_indices=(0, 1, 2, 3, 4),
        background_channel_index_by_fold={"1": 5},
        supplied_background_is_exact_complement_required=False,
        void_definition="no positive class and no supplied background",
        cross_class_overlap_definition="more than one occupied positive class",
        positive_and_background_definition="positive class and supplied background",
        cross_class_overlap_action="retain raw identities; do not arbitrate a class",
        analysis_instance_exclusion_reason="touches_cross_class_overlap",
        applies_identically_to_primary_and_confirmatory=True,
        no_class_arbitration=True,
        source_masks_modified=False,
        release_annotation_anomalies_are_fatal=False,
        structural_invalidity_is_fatal=True,
    )
    selection = select_anomaly_overlay_patches((fold_qc,), max_patches=4)
    return PanNukeValidationResult(
        root=raw_root,
        mapping=OFFICIAL_METRICS_CLASS_MAPPING,
        folds=(discovered,),
        fold_validation=(validation,),
        inventory=(),
        archive_paths=(),
        global_mask_qc=global_qc,
        qc_policy=policy,
        anomaly_overlay_selection=selection,
        expected_fold_ids=(1,),
        release_complete=True,
    )


def _raw_tree_snapshot(root: Path) -> dict[str, tuple[str, int, int, str | None]]:
    snapshot: dict[str, tuple[str, int, int, str | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        snapshot[relative] = (
            "file" if path.is_file() else "directory",
            stat.st_size,
            stat.st_mtime_ns,
            payload_sha256,
        )
    return snapshot


def test_writes_complete_reconciled_bundle_with_neutral_overlay(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    artifacts = write_mask_qc_report_bundle(result, tmp_path / "qc-report", max_overlay_patches=4)

    assert validate_mask_qc_report_bundle(artifacts.bundle_dir) == artifacts
    assert artifacts.patch_row_count == 4
    assert artifacts.instance_row_count == 3
    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    selection = json.loads(artifacts.overlay_selection_path.read_text(encoding="utf-8"))
    assert report["global_mask_qc"]["cross_class_overlap_pixel_count"] == 1
    assert report["global_mask_qc"]["void_pixel_count"] == 1
    assert report["global_mask_qc"]["positive_and_background_pixel_count"] == 1
    assert report["disconnected_instance_qc"] == {
        "affected_patch_count": 0,
        "analysis_eligibility_status": "to_be_frozen_after_pilot_without_final_outcomes",
        "by_fold": [{"affected_patch_count": 0, "fold_id": 1, "instance_count": 0}],
        "connectivity": "4_connected",
        "instance_count": 0,
        "instances_split_or_repaired": False,
        "source_masks_modified": False,
    }
    assert selection["selection_sha256"] == artifacts.selection_sha256
    assert selection["overlay_sha256"] == artifacts.overlay_sha256
    assert selection["rendering"]["cross_class_overlap_colour"] == "#d9d9d9"
    assert selection["rendering"]["positive_class_winner_encoded"] is False
    assert [item["patch_key"] for item in selection["selected_patch_evidence"]] == [
        "fold_1:patch_0",
        "fold_1:patch_2",
        "fold_1:patch_1",
        "fold_1:patch_3",
    ]
    with artifacts.patch_csv_path.open("r", encoding="utf-8", newline="") as handle:
        patch_rows = list(csv.DictReader(handle))
    with artifacts.instance_csv_path.open("r", encoding="utf-8", newline="") as handle:
        instance_rows = list(csv.DictReader(handle))
    assert len(patch_rows) == 4
    assert sum(row["touches_cross_class_overlap"] == "true" for row in instance_rows) == 2
    assert {
        row["analysis_exclusion_reason"]
        for row in instance_rows
        if row["touches_cross_class_overlap"] == "true"
    } == {"touches_cross_class_overlap"}
    for row in instance_rows:
        assert row["primary_eligible"] == row["confirmatory_eligible"]
        assert row["analysis_eligible"] == row["primary_eligible"]
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "potentially inconsistent annotations" in markdown
    assert "recommended for expert review" in markdown
    assert "no class arbitration" in markdown
    with Image.open(artifacts.overlay_path) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0


def test_overlap_layer_is_exactly_neutral_even_when_void_also_true() -> None:
    overlap = np.asarray([[True, False], [False, False]])
    void = np.asarray([[True, True], [False, False]])

    rgba = anomaly_overlay_rgba(overlap, void)

    np.testing.assert_allclose(rgba[0, 0, :3], np.asarray([217, 217, 217]) / 255.0)
    assert rgba[0, 0, 3] == pytest.approx(OVERLAP_ALPHA)
    assert not np.array_equal(rgba[0, 0, :3], rgba[0, 1, :3])
    assert np.count_nonzero(rgba[1, 1]) == 0


def test_identical_rerun_is_idempotent_and_does_not_rewrite(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    first = write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()
    }

    second = write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert second == first
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()
    }
    assert after == before


def test_partial_existing_bundle_fails_closed_without_filling_it(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    destination.mkdir()
    sentinel = destination / "pannuke_mask_qc.json"
    sentinel.write_text("sentinel", encoding="utf-8")

    with pytest.raises(MaskQCReportError, match="partial"):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert sentinel.read_text(encoding="utf-8") == "sentinel"
    assert tuple(destination.iterdir()) == (sentinel,)


def test_active_bundle_publication_lock_rejects_second_writer_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("report construction ran while its publication lock was held")

    monkeypatch.setattr(reporting, "_build_bundle_content", unexpected)
    with (
        reporting.ExclusiveBundlePublicationLock(
            (destination.parent, destination), role="test mask-QC bundle"
        ),
        pytest.raises(FileExistsError, match="publication is active"),
    ):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert not destination.exists()


def test_parent_target_lock_rejects_partially_overlapping_qc_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    first = tmp_path / "qc-a"
    second = tmp_path / "qc-b"

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("report construction ran while its shared parent target was locked")

    monkeypatch.setattr(reporting, "_build_bundle_content", unexpected)
    with (
        reporting.ExclusiveBundlePublicationLock(
            (first.parent, first), role="first mask-QC bundle"
        ),
        pytest.raises(FileExistsError, match="another PanNuke mask-QC bundle"),
    ):
        write_mask_qc_report_bundle(result, second, max_overlay_patches=4)

    assert not first.exists()
    assert not second.exists()


def test_foreign_bundle_created_during_promotion_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    sentinel = destination / "foreign.txt"
    original_publish = reporting.publish_flat_directory_no_overwrite

    def publish_after_foreign_creation(
        staged_directory: str | Path,
        final_destination: str | Path,
        *,
        success_marker_name: str | None = None,
    ) -> list[reporting.PublishedPath]:
        assert Path(final_destination) == destination
        destination.mkdir()
        sentinel.write_text("foreign-owner", encoding="utf-8")
        return original_publish(
            staged_directory,
            final_destination,
            success_marker_name=success_marker_name,
        )

    monkeypatch.setattr(
        reporting,
        "publish_flat_directory_no_overwrite",
        publish_after_foreign_creation,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite publication path"):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert sentinel.read_text(encoding="utf-8") == "foreign-owner"
    assert tuple(destination.iterdir()) == (sentinel,)


def test_final_consistency_failure_preserves_foreign_qc_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    foreign_payload = b"foreign QC report after validation\n"
    original_validate = reporting.validate_mask_qc_report_bundle
    replaced = False

    def replace_after_validation(value: str | Path) -> reporting.MaskQCReportArtifacts:
        nonlocal replaced
        artifacts = original_validate(value)
        if not replaced:
            replaced = True
            artifacts.json_path.unlink()
            artifacts.json_path.write_bytes(foreign_payload)
        return artifacts

    monkeypatch.setattr(reporting, "validate_mask_qc_report_bundle", replace_after_validation)
    with pytest.raises(RuntimeError, match="ownership-safe rollback was incomplete"):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert replaced
    assert (destination / "pannuke_mask_qc.json").read_bytes() == foreign_payload
    assert {path.name for path in destination.iterdir()} == {"pannuke_mask_qc.json"}


def test_broken_bundle_symlink_is_occupied_and_preserved(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    missing_target = tmp_path / "foreign-missing-bundle"
    try:
        destination.symlink_to(missing_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable on this Windows host: {error}")
    original_link_target = destination.readlink()

    with pytest.raises(MaskQCReportError, match="bundle does not exist"):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert destination.is_symlink()
    assert destination.readlink() == original_link_target
    assert not missing_target.exists()


def test_writer_rejects_raw_destinations_before_construction_or_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    raw_root = result.root.resolve()
    before = _raw_tree_snapshot(raw_root)

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("report construction ran before the raw-destination guard")

    monkeypatch.setattr(reporting, "_build_bundle_content", unexpected)
    destinations = (
        tmp_path,
        raw_root,
        raw_root / "nested" / "qc-report",
        tmp_path / "outside-not-created" / ".." / "raw" / "traversal-report",
    )
    for destination in destinations:
        with pytest.raises(MaskQCReportError, match="outside the immutable raw release"):
            write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert _raw_tree_snapshot(raw_root) == before
    assert not (raw_root / "nested").exists()
    assert not (raw_root / "traversal-report").exists()
    assert not (tmp_path / "outside-not-created").exists()


def test_writer_resolves_symlink_into_raw_and_leaves_raw_tree_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import histo_audit.pannuke.qc_reporting as reporting

    result = _validation_result(tmp_path)
    raw_root = result.root.resolve()
    link = tmp_path / "raw-link"
    try:
        link.symlink_to(raw_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable on this Windows host: {error}")
    before = _raw_tree_snapshot(raw_root)

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("report construction ran before the symlink destination guard")

    monkeypatch.setattr(reporting, "_build_bundle_content", unexpected)
    with pytest.raises(MaskQCReportError, match="outside the immutable raw release"):
        write_mask_qc_report_bundle(result, link / "qc-report", max_overlay_patches=4)

    assert _raw_tree_snapshot(raw_root) == before
    assert not (raw_root / "qc-report").exists()


def test_tampered_complete_bundle_is_detected_and_never_overwritten(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    artifacts = write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)
    tampered = artifacts.patch_csv_path.read_bytes() + b"tamper\n"
    artifacts.patch_csv_path.write_bytes(tampered)

    with pytest.raises(MaskQCReportError, match="hash/size differs"):
        validate_mask_qc_report_bundle(destination)
    with pytest.raises(MaskQCReportError, match="hash/size differs"):
        write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)

    assert artifacts.patch_csv_path.read_bytes() == tampered


def test_complete_different_result_is_not_overwritten(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    destination = tmp_path / "qc-report"
    artifacts = write_mask_qc_report_bundle(result, destination, max_overlay_patches=4)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    changed_policy = replace(result.qc_policy, policy_version="pannuke-mask-qc-v2")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_mask_qc_report_bundle(
            replace(result, qc_policy=changed_policy),
            destination,
            max_overlay_patches=4,
        )

    assert artifacts.json_path.read_bytes() == before[artifacts.json_path.name]
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_coordinated_hash_rewrite_cannot_hide_eligibility_tamper(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    artifacts = write_mask_qc_report_bundle(result, tmp_path / "qc-report", max_overlay_patches=4)
    instance_text = artifacts.instance_csv_path.read_text(encoding="utf-8")
    changed = instance_text.replace(
        ",true,false,false,false,touches_cross_class_overlap",
        ",true,false,true,false,touches_cross_class_overlap",
        1,
    )
    assert changed != instance_text
    artifacts.instance_csv_path.write_text(changed, encoding="utf-8", newline="")

    def record(path: Path) -> dict[str, object]:
        content = path.read_bytes()
        return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    report["artifacts"][artifacts.instance_csv_path.name] = record(artifacts.instance_csv_path)
    artifacts.json_path.write_text(
        json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    manifest = json.loads(artifacts.artifact_manifest_path.read_text(encoding="utf-8"))
    manifest["files"][artifacts.instance_csv_path.name] = record(artifacts.instance_csv_path)
    manifest["files"][artifacts.json_path.name] = record(artifacts.json_path)
    artifacts.artifact_manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(MaskQCReportError, match="shared eligibility differs"):
        validate_mask_qc_report_bundle(artifacts.bundle_dir)


def test_cross_level_count_tamper_fails_before_publication(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    tampered_global = replace(
        result.global_mask_qc,
        cross_class_overlap_pixel_count=(result.global_mask_qc.cross_class_overlap_pixel_count + 1),
    )

    with pytest.raises(MaskQCReportError, match="does not reconcile"):
        write_mask_qc_report_bundle(
            replace(result, global_mask_qc=tampered_global),
            tmp_path / "qc-report",
            max_overlay_patches=4,
        )

    assert not (tmp_path / "qc-report").exists()


def test_selected_raw_mask_change_is_detected_by_recomputed_flags(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    fold = result.folds[0]
    changed = np.load(fold.mask_path, allow_pickle=False)
    changed[0, 0, 0, 1] = 0
    np.save(fold.mask_path, changed, allow_pickle=False)

    with pytest.raises(MaskQCReportError, match="raw anomaly masks differ"):
        write_mask_qc_report_bundle(result, tmp_path / "qc-report", max_overlay_patches=4)


def test_selection_sidecar_tamper_is_hash_detected(tmp_path: Path) -> None:
    result = _validation_result(tmp_path)
    artifacts = write_mask_qc_report_bundle(result, tmp_path / "qc-report", max_overlay_patches=4)
    payload = json.loads(artifacts.overlay_selection_path.read_text(encoding="utf-8"))
    payload["rendering"]["positive_class_winner_encoded"] = True
    artifacts.overlay_selection_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MaskQCReportError, match="hash/size differs"):
        validate_mask_qc_report_bundle(artifacts.bundle_dir)
