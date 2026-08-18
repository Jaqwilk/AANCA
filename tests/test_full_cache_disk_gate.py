"""Fail-closed destination-volume gate for the full PanNuke cache build."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import histo_audit.pannuke as pannuke_package
import histo_audit.representations as representations_package
import histo_audit.representations.pannuke as pannuke_module
import histo_audit.representations.pannuke_chunked as chunked_module
from histo_audit.cli import (
    CANONICAL_PANNUKE_VALIDATION_MAX_OVERLAY_PATCHES,
    CANONICAL_PANNUKE_VALIDATION_MAX_SAMPLES_PER_FOLD,
    app,
)
from histo_audit.experiment import representation_independence as independence_module
from histo_audit.experiment.representation_independence import (
    build_pannuke_representation_cache_with_independence,
)
from histo_audit.pannuke import PanNukeValidationResult
from histo_audit.representations.pannuke import (
    FULL_MANIFEST_CACHE_MIN_FREE_BYTES,
    InsufficientFullManifestCacheDiskSpaceError,
    build_pannuke_representation_cache,
    require_full_manifest_cache_disk_space,
)


def _large_manifest_metadata(monkeypatch: pytest.MonkeyPatch, *, rows: int = 10_001) -> None:
    class FakeParquetFile:
        def __init__(self, _path: Path) -> None:
            self.metadata = SimpleNamespace(num_rows=rows)

    monkeypatch.setattr(pannuke_module.pq, "ParquetFile", FakeParquetFile)


def _validation_stub(tmp_path: Path) -> PanNukeValidationResult:
    raw = tmp_path / "raw"
    raw.mkdir()
    return cast(PanNukeValidationResult, SimpleNamespace(root=raw.resolve()))


def test_full_manifest_builder_fails_below_35_gib_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _large_manifest_metadata(monkeypatch)
    output = tmp_path / "cache"
    free_bytes = FULL_MANIFEST_CACHE_MIN_FREE_BYTES - 1
    probes: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        probes.append(Path(path))
        return SimpleNamespace(total=free_bytes * 2, used=free_bytes, free=free_bytes)

    def forbidden_chunked_builder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("chunked allocation must not start below the disk threshold")

    monkeypatch.setattr(pannuke_module.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(
        chunked_module,
        "build_pannuke_representation_cache_chunked",
        forbidden_chunked_builder,
    )

    with pytest.raises(InsufficientFullManifestCacheDiskSpaceError) as caught:
        build_pannuke_representation_cache(
            _validation_stub(tmp_path),
            tmp_path / "full.parquet",
            output,
        )

    check = caught.value.check
    assert check.free_bytes == free_bytes
    assert check.required_free_bytes == FULL_MANIFEST_CACHE_MIN_FREE_BYTES
    assert check.target_output_parent == tmp_path.resolve()
    assert probes == [tmp_path.resolve()]
    assert f"free_bytes={free_bytes}" in str(caught.value)
    assert f"required_free_bytes={FULL_MANIFEST_CACHE_MIN_FREE_BYTES}" in str(caught.value)
    assert not output.exists()
    assert not (tmp_path / ".cache.chunked-resume").exists()


def test_full_manifest_builder_accepts_exact_35_gib_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _large_manifest_metadata(monkeypatch)
    free_bytes = FULL_MANIFEST_CACHE_MIN_FREE_BYTES
    sentinel = object()
    probes: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        probes.append(Path(path))
        return SimpleNamespace(total=free_bytes * 2, used=free_bytes, free=free_bytes)

    def fake_chunked_builder(*_args: Any, **_kwargs: Any) -> object:
        return sentinel

    monkeypatch.setattr(pannuke_module.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(
        chunked_module,
        "build_pannuke_representation_cache_chunked",
        fake_chunked_builder,
    )

    result = build_pannuke_representation_cache(
        _validation_stub(tmp_path),
        tmp_path / "full.parquet",
        tmp_path / "cache",
    )

    assert result is sentinel
    assert probes == [tmp_path.resolve()]


def test_small_and_explicit_sample_id_builds_do_not_consume_full_cache_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_disk_usage(_path: Path) -> None:
        raise AssertionError("smoke and small-fixture paths must not inspect the 35-GiB gate")

    monkeypatch.setattr(pannuke_module.shutil, "disk_usage", forbidden_disk_usage)
    _large_manifest_metadata(monkeypatch, rows=10_000)
    small = require_full_manifest_cache_disk_space(
        tmp_path / "small.parquet",
        tmp_path / "small-cache",
    )
    assert small.check_required is False
    assert small.free_bytes is None

    class ForbiddenParquetFile:
        def __init__(self, _path: Path) -> None:
            raise AssertionError("explicit sample-ID smoke must not inspect full manifest metadata")

    monkeypatch.setattr(pannuke_module.pq, "ParquetFile", ForbiddenParquetFile)
    selected = require_full_manifest_cache_disk_space(
        tmp_path / "full.parquet",
        tmp_path / "sample-cache",
        sample_ids=("sample-a", "sample-b"),
    )
    assert selected.check_required is False
    assert selected.manifest_row_count == 2


def test_explicit_selection_above_10_000_cannot_bypass_disk_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    free_bytes = FULL_MANIFEST_CACHE_MIN_FREE_BYTES - 1
    probes: list[Path] = []

    class ForbiddenParquetFile:
        def __init__(self, _path: Path) -> None:
            raise AssertionError("an explicit selection must use its effective count")

    def disk_usage(path: Path) -> SimpleNamespace:
        probes.append(Path(path))
        return SimpleNamespace(total=free_bytes * 2, used=free_bytes, free=free_bytes)

    monkeypatch.setattr(pannuke_module.pq, "ParquetFile", ForbiddenParquetFile)
    monkeypatch.setattr(pannuke_module.shutil, "disk_usage", disk_usage)

    with pytest.raises(InsufficientFullManifestCacheDiskSpaceError) as caught:
        build_pannuke_representation_cache(
            _validation_stub(tmp_path),
            tmp_path / "full.parquet",
            tmp_path / "selected-cache",
            sample_ids=tuple(f"sample-{index}" for index in range(10_001)),
        )

    assert caught.value.check.manifest_row_count == 10_001
    assert probes == [tmp_path.resolve()]
    assert not (tmp_path / "selected-cache").exists()


def test_cache_plus_independence_gate_runs_before_parent_directories_are_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _large_manifest_metadata(monkeypatch)
    monkeypatch.setattr(
        pannuke_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=FULL_MANIFEST_CACHE_MIN_FREE_BYTES,
            used=1,
            free=FULL_MANIFEST_CACHE_MIN_FREE_BYTES - 1,
        ),
    )
    output = tmp_path / "new-cache-parent" / "cache"
    independence = tmp_path / "new-report-parent" / "independence.json"

    with pytest.raises(InsufficientFullManifestCacheDiskSpaceError):
        build_pannuke_representation_cache_with_independence(
            _validation_stub(tmp_path),
            tmp_path / "full.parquet",
            output,
            independence,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1, 2),
            final_test_fold=3,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            expected_canonical_manifest_sha256="a" * 64,
            expected_analysis_eligible_sample_order_sha256="b" * 64,
            expected_analysis_eligible_sample_count=10_001,
        )

    assert not output.parent.exists()
    assert not independence.parent.exists()


@pytest.mark.parametrize("direction", ("independence_inside_cache", "cache_inside_artifact"))
def test_cache_and_independence_overlap_is_rejected_in_both_directions_without_creation(
    tmp_path: Path,
    direction: str,
) -> None:
    base = tmp_path / "overlap-must-remain-absent"
    if direction == "independence_inside_cache":
        output = base / "cache"
        independence = output / "independence.json"
    else:
        independence = base / "independence.json"
        output = independence / "cache"

    with pytest.raises(ValueError, match="must not contain or overlap"):
        build_pannuke_representation_cache_with_independence(
            _validation_stub(tmp_path),
            tmp_path / "manifest-does-not-need-to-exist.parquet",
            output,
            independence,
            class_order=(0, 1, 2, 3, 4),
            development_official_folds=(1, 2),
            final_test_fold=3,
            reference_validation_fraction_groups=0.1,
            split_seed=223,
            expected_canonical_manifest_sha256="a" * 64,
            expected_analysis_eligible_sample_order_sha256="b" * 64,
            expected_analysis_eligible_sample_count=10_001,
        )

    assert not base.exists()


def test_direct_chunked_entrypoint_cannot_bypass_full_manifest_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _large_manifest_metadata(monkeypatch)
    monkeypatch.setattr(
        pannuke_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=FULL_MANIFEST_CACHE_MIN_FREE_BYTES,
            used=1,
            free=FULL_MANIFEST_CACHE_MIN_FREE_BYTES - 1,
        ),
    )
    output = tmp_path / "direct-chunk-parent" / "cache"

    with pytest.raises(InsufficientFullManifestCacheDiskSpaceError):
        chunked_module.build_pannuke_representation_cache_chunked(
            _validation_stub(tmp_path),
            tmp_path / "full.parquet",
            output,
            chunk_size=4_096,
        )

    assert not output.parent.exists()


def test_cli_gates_full_build_before_heavy_validation_and_reports_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _large_manifest_metadata(monkeypatch)
    free_bytes = FULL_MANIFEST_CACHE_MIN_FREE_BYTES - 1
    monkeypatch.setattr(
        pannuke_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free_bytes * 2, used=free_bytes, free=free_bytes),
    )
    validation_called = False

    def forbidden_validation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal validation_called
        validation_called = True
        raise AssertionError("full PanNuke validation must not run before the disk start gate")

    monkeypatch.setattr(pannuke_package, "validate_pannuke", forbidden_validation)
    data_root = tmp_path / "raw-release"
    data_root.mkdir()
    (data_root / "present.npy").write_bytes(b"local-presence-only")
    output = tmp_path / "derived" / "cache"
    independence = tmp_path / "reports-new" / "independence.json"

    result = CliRunner().invoke(
        app,
        [
            "representations",
            "extract",
            "--project-root",
            str(tmp_path),
            "--data-root",
            str(data_root),
            "--manifest",
            "full.parquet",
            "--output-dir",
            str(output),
            "--independence-output",
            str(independence),
            "--include-context-embeddings",
        ],
    )

    assert result.exit_code == 1, result.output
    assert validation_called is False
    assert f"free_bytes={free_bytes}" in result.output
    assert f"required_free_bytes={FULL_MANIFEST_CACHE_MIN_FREE_BYTES}" in result.output
    assert not output.exists()
    assert not output.parent.exists()
    assert not independence.exists()
    assert not independence.parent.exists()


def test_cli_forwards_exact_primary_analysis_manifest_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_manifest_sha256 = "1" * 64
    sample_order_sha256 = "2" * 64
    sample_count = 188_333
    primary_config = tmp_path / "primary.yaml"
    primary_config.write_text(
        "\n".join(
            (
                "data:",
                "  class_order: [0, 1, 2, 3, 4]",
                "  development_official_folds: [1, 2]",
                "  final_test_fold: 3",
                "  reference_validation_fraction_groups: 0.1",
                "  split_seed: 223",
                "  analysis_manifest_authority:",
                f"    canonical_manifest_sha256: {canonical_manifest_sha256}",
                f"    analysis_eligible_sample_order_sha256: {sample_order_sha256}",
                f"    analysis_eligible_sample_count: {sample_count}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "raw-release"
    data_root.mkdir()
    (data_root / "present.npy").write_bytes(b"presence")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        representations_package,
        "require_full_manifest_cache_disk_space",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pannuke_package,
        "validate_pannuke",
        lambda *_args, **_kwargs: SimpleNamespace(result=SimpleNamespace(root=data_root.resolve())),
    )

    def capture_authority(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise RuntimeError("captured manifest authority")

    monkeypatch.setattr(
        independence_module,
        "build_pannuke_representation_cache_with_independence",
        capture_authority,
    )
    result = CliRunner().invoke(
        app,
        [
            "representations",
            "extract",
            "--project-root",
            str(tmp_path),
            "--data-root",
            str(data_root),
            "--manifest",
            "manifest.parquet",
            "--output-dir",
            "cache",
            "--independence-output",
            "independence.json",
            "--primary-config",
            str(primary_config),
            "--include-context-embeddings",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "captured manifest authority" in result.output
    assert captured["expected_canonical_manifest_sha256"] == canonical_manifest_sha256
    assert captured["expected_analysis_eligible_sample_order_sha256"] == sample_order_sha256
    assert captured["expected_analysis_eligible_sample_count"] == sample_count


def test_cli_uses_canonical_validation_limits_before_representation_cache_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "raw-release"
    data_root.mkdir()
    (data_root / "present.npy").write_bytes(b"presence")
    output = tmp_path / "cache"
    events: list[str] = []
    validation_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(
        representations_package,
        "require_full_manifest_cache_disk_space",
        lambda *_args, **_kwargs: None,
    )

    def capture_validation(*_args: Any, **kwargs: Any) -> Any:
        events.append("validation")
        validation_kwargs.update(kwargs)
        return SimpleNamespace(result=SimpleNamespace(root=data_root.resolve()))

    def stop_at_cache_build(*_args: Any, **_kwargs: Any) -> None:
        events.append("cache_build")
        raise RuntimeError("stop at mocked representation cache build")

    monkeypatch.setattr(pannuke_package, "validate_pannuke", capture_validation)
    monkeypatch.setattr(
        representations_package,
        "build_pannuke_representation_cache",
        stop_at_cache_build,
    )

    result = CliRunner().invoke(
        app,
        [
            "representations",
            "extract",
            "--project-root",
            str(tmp_path),
            "--data-root",
            str(data_root),
            "--manifest",
            "manifest.parquet",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "stop at mocked representation cache build" in result.output
    assert events == ["validation", "cache_build"]
    assert validation_kwargs["max_samples_per_fold"] == (
        CANONICAL_PANNUKE_VALIDATION_MAX_SAMPLES_PER_FOLD
    )
    assert validation_kwargs["max_overlay_patches"] == (
        CANONICAL_PANNUKE_VALIDATION_MAX_OVERLAY_PATCHES
    )
    assert not output.exists()
