from __future__ import annotations

import hashlib
import io
import os
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import capsule_builder
from capsule_builder import CapsuleBuildError, source_inventory


def _regular_member(
    path: Path,
    *,
    relative_path: str = "histo_audit/module.py",
) -> capsule_builder.PayloadMember:
    path.write_bytes(b"VALUE = 1\n")
    return capsule_builder._read_stable_regular_file(
        path,
        relative_path=relative_path,
        role="project_source",
    )


def _byte_build(tmp_path: Path) -> capsule_builder.CapsuleByteBuild:
    member = _regular_member(tmp_path / "module.py")
    inventory = source_inventory((member,))
    return capsule_builder.build_capsule_bytes(
        members=(member,),
        expected_inventory=inventory,
    )


def _publication_path(
    tmp_path: Path,
    build: capsule_builder.CapsuleByteBuild,
    *,
    create_parent: bool = True,
) -> Path:
    parent = tmp_path / "artifacts" / "execution_capsules" / build.sha256
    if create_parent:
        parent.mkdir(parents=True)
    return parent / capsule_builder.CAPSULE_FILENAME


def _read_path(path: Path) -> bytes:
    descriptor = capsule_builder._open_read_no_follow(path)
    try:
        return capsule_builder._read_exact_descriptor(
            descriptor,
            expected_size=int(capsule_builder._lstat(path).st_size),
        )
    finally:
        os.close(descriptor)


def test_create_new_publisher_uses_exact_content_addressed_bytes(
    tmp_path: Path,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)
    result = capsule_builder.publish_capsule_create_new(
        build=build,
        output_path=destination,
    )
    assert result.output_path == destination
    assert result.sha256 == build.sha256
    readback = _read_path(destination)
    assert readback == build.archive_bytes
    assert hashlib.sha256(readback).hexdigest() == build.sha256
    assert capsule_builder._lstat(destination).st_nlink == 1
    assert capsule_builder._is_read_only_file(capsule_builder._lstat(destination))


@pytest.mark.parametrize("preexisting", [b"partial", None])
def test_create_new_publisher_never_adopts_a_preexisting_destination(
    tmp_path: Path,
    preexisting: bytes | None,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)
    payload = build.archive_bytes if preexisting is None else preexisting
    destination.write_bytes(payload)
    before = capsule_builder._lstat(destination)
    with pytest.raises(CapsuleBuildError, match="CREATE_NEW publication failed"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    after = capsule_builder._lstat(destination)
    assert _read_path(destination) == payload
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_create_new_publisher_requires_existing_content_address_parent(
    tmp_path: Path,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build, create_parent=False)
    with pytest.raises((CapsuleBuildError, FileNotFoundError)):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    with pytest.raises(FileNotFoundError):
        capsule_builder._lstat(destination)


def test_create_new_publisher_rejects_symlink_ancestor(tmp_path: Path) -> None:
    build = _byte_build(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    real = tmp_path / "real-execution-capsules"
    (real / build.sha256).mkdir(parents=True)
    alias = artifacts / "execution_capsules"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    destination = alias / build.sha256 / capsule_builder.CAPSULE_FILENAME
    with pytest.raises(CapsuleBuildError, match="link/reparse"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    assert not (real / build.sha256 / capsule_builder.CAPSULE_FILENAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_create_new_publisher_rejects_junction_ancestor(tmp_path: Path) -> None:
    build = _byte_build(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    real = tmp_path / "real-execution-capsules"
    (real / build.sha256).mkdir(parents=True)
    junction = artifacts / "execution_capsules"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(real)],
        capture_output=True,
        check=False,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        timeout=30,
    )
    if completed.returncode != 0:
        pytest.skip(
            "junction creation unavailable: " + completed.stderr.decode(errors="replace").strip()
        )
    destination = junction / build.sha256 / capsule_builder.CAPSULE_FILENAME
    try:
        with pytest.raises(CapsuleBuildError, match="link/reparse"):
            capsule_builder.publish_capsule_create_new(
                build=build,
                output_path=destination,
            )
    finally:
        os.rmdir(junction)


def test_named_stream_detection_leaves_permanent_unadoptable_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)
    original = capsule_builder._named_streams

    def report_injected_stream(path: Path) -> tuple[str, ...]:
        if path == destination:
            return (":injected:$DATA",)
        return original(path)

    monkeypatch.setattr(capsule_builder, "_named_streams", report_injected_stream)
    with pytest.raises(CapsuleBuildError, match="path/handle identity"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    assert _read_path(destination) == build.archive_bytes
    assert capsule_builder._is_read_only_file(capsule_builder._lstat(destination))
    with pytest.raises(CapsuleBuildError, match="CREATE_NEW publication failed"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )


def test_preexisting_hardlink_destination_is_never_adopted(tmp_path: Path) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)
    source = destination.parent / "preexisting.pyz"
    source.write_bytes(build.archive_bytes)
    try:
        os.link(
            capsule_builder._windows_api_path(source) if os.name == "nt" else source,
            capsule_builder._windows_api_path(destination) if os.name == "nt" else destination,
        )
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    with pytest.raises(CapsuleBuildError, match="CREATE_NEW publication failed"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    assert capsule_builder._lstat(destination).st_nlink == 2
    assert _read_path(destination) == build.archive_bytes


def test_same_bytes_replacement_attempt_is_blocked_or_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)
    replacement = destination.parent / "same-bytes-replacement.pyz"
    replacement.write_bytes(build.archive_bytes)
    original = capsule_builder._named_streams
    attempted = False
    replacement_succeeded = False

    def replacement_canary(path: Path) -> tuple[str, ...]:
        nonlocal attempted, replacement_succeeded
        if path == destination and not attempted:
            attempted = True
            try:
                os.replace(
                    capsule_builder._windows_api_path(replacement)
                    if os.name == "nt"
                    else replacement,
                    capsule_builder._windows_api_path(destination)
                    if os.name == "nt"
                    else destination,
                )
            except OSError:
                pass
            else:
                replacement_succeeded = True
        return original(path)

    monkeypatch.setattr(capsule_builder, "_named_streams", replacement_canary)
    if replacement_succeeded:
        pytest.fail("replacement state cannot be true before the canary runs")
    try:
        result = capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    except CapsuleBuildError:
        assert attempted and replacement_succeeded
    else:
        assert attempted and not replacement_succeeded
        assert result.sha256 == build.sha256
        assert _read_path(destination) == build.archive_bytes


def test_exception_after_create_leaves_permanent_partial_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _byte_build(tmp_path)
    destination = _publication_path(tmp_path, build)

    def fail_after_prefix(descriptor: int, payload: bytes) -> None:
        assert os.write(descriptor, payload[:31]) == 31
        os.fsync(descriptor)
        raise RuntimeError("synthetic post-create failure")

    monkeypatch.setattr(capsule_builder, "_write_all", fail_after_prefix)
    with pytest.raises(RuntimeError, match="synthetic post-create failure"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )
    assert _read_path(destination) == build.archive_bytes[:31]
    assert capsule_builder._is_read_only_file(capsule_builder._lstat(destination))
    with pytest.raises(CapsuleBuildError, match="CREATE_NEW publication failed"):
        capsule_builder.publish_capsule_create_new(
            build=build,
            output_path=destination,
        )


def test_two_private_byte_builds_are_exactly_identical(tmp_path: Path) -> None:
    member = _regular_member(tmp_path / "module.py")
    inventory = source_inventory((member,))
    first = capsule_builder.build_capsule_bytes(
        members=(member,),
        expected_inventory=inventory,
    )
    second = capsule_builder.build_capsule_bytes(
        members=(member,),
        expected_inventory=inventory,
    )
    assert first.archive_bytes == second.archive_bytes
    assert first.sha256 == second.sha256
    assert first.internal_manifest_sha256 == second.internal_manifest_sha256


@pytest.mark.parametrize(
    "manifest_alias",
    [
        "aanca_capsule_manifest.json",
        "histo_audit/aanca_capsule_manifest.json/../module.py",
    ],
)
def test_manifest_casefold_alias_or_lexical_alias_is_rejected(
    tmp_path: Path,
    manifest_alias: str,
) -> None:
    member = _regular_member(tmp_path / "module.py")
    with pytest.raises(CapsuleBuildError):
        source_inventory([replace(member, relative_path=manifest_alias)])


def test_role_is_restricted_to_the_exact_emitted_enum(tmp_path: Path) -> None:
    member = _regular_member(tmp_path / "module.py")
    with pytest.raises(CapsuleBuildError, match="role"):
        source_inventory([replace(member, role="valid_but_unassigned_role")])


def test_duplicate_physical_identity_is_rejected(tmp_path: Path) -> None:
    member = _regular_member(tmp_path / "module.py")
    alias_record = replace(member, relative_path="histo_audit/alias.py")
    with pytest.raises(CapsuleBuildError, match="physical identity"):
        source_inventory([member, alias_record])


def test_hardlinked_source_is_rejected_before_read(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    hardlink = tmp_path / "hardlink.py"
    try:
        os.link(source, hardlink)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    with pytest.raises(CapsuleBuildError, match="single-link"):
        capsule_builder._read_stable_regular_file(
            source,
            relative_path="histo_audit/source.py",
            role="project_source",
        )


def test_relative_and_dotdot_source_forms_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    with pytest.raises(CapsuleBuildError, match="absolute"):
        capsule_builder._read_stable_regular_file(
            Path("source.py"),
            relative_path="histo_audit/source.py",
            role="project_source",
        )

    middle = tmp_path / "middle"
    middle.mkdir()
    dotdot_form = middle / ".." / source.name
    with pytest.raises(CapsuleBuildError, match="canonical lexical form"):
        capsule_builder._read_stable_regular_file(
            dotdot_form,
            relative_path="histo_audit/source.py",
            role="project_source",
        )


def test_package_scan_rejects_symlink_root_before_resolve(tmp_path: Path) -> None:
    real_root = tmp_path / "real" / "histo_audit"
    real_root.mkdir(parents=True)
    (real_root / "__init__.py").write_bytes(b"")
    alias_root = tmp_path / "alias" / "histo_audit"
    alias_root.parent.mkdir()
    try:
        alias_root.symlink_to(real_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(CapsuleBuildError, match="link/reparse"):
        capsule_builder._scan_python_paths(alias_root)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_package_scan_rejects_windows_junction_root_before_resolve(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real" / "histo_audit"
    real_root.mkdir(parents=True)
    (real_root / "__init__.py").write_bytes(b"")
    junction_root = tmp_path / "junction" / "histo_audit"
    junction_root.parent.mkdir()
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    completed = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction_root),
            str(real_root),
        ],
        capture_output=True,
        check=False,
        creationflags=creation_flags,
        timeout=30,
    )
    if completed.returncode != 0:
        pytest.skip(
            "junction creation unavailable: " + completed.stderr.decode(errors="replace").strip()
        )
    try:
        with pytest.raises(CapsuleBuildError, match="link/reparse"):
            capsule_builder._scan_python_paths(junction_root)
    finally:
        os.rmdir(junction_root)


def test_source_handle_remains_open_during_replace_canary_and_final_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"ORIGINAL = 1\n")
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(b"REPLACEMENT = 1\n")
    original_open = capsule_builder._open_read_no_follow
    original_named_streams = capsule_builder._named_streams
    descriptor_holder: dict[str, int] = {}
    stream_calls = 0
    replacement_was_blocked = False

    def tracking_open(path: Path) -> int:
        descriptor = original_open(path)
        descriptor_holder["descriptor"] = descriptor
        return descriptor

    def replace_canary(path: Path) -> tuple[str, ...]:
        nonlocal stream_calls, replacement_was_blocked
        stream_calls += 1
        if stream_calls == 2:
            os.fstat(descriptor_holder["descriptor"])
            try:
                os.replace(replacement, path)
            except OSError:
                replacement_was_blocked = True
        return original_named_streams(path)

    monkeypatch.setattr(capsule_builder, "_open_read_no_follow", tracking_open)
    monkeypatch.setattr(capsule_builder, "_named_streams", replace_canary)
    try:
        member = capsule_builder._read_stable_regular_file(
            source,
            relative_path="histo_audit/source.py",
            role="project_source",
        )
    except CapsuleBuildError:
        assert not replacement_was_blocked
    else:
        assert replacement_was_blocked
        assert member.payload == b"ORIGINAL = 1\n"
    with pytest.raises(OSError):
        os.fstat(descriptor_holder["descriptor"])


def test_local_header_tamper_is_rejected_even_when_central_directory_reads(
    tmp_path: Path,
) -> None:
    member = _regular_member(tmp_path / "module.py")
    members = (member,)
    manifest, _, _ = capsule_builder._manifest_bytes(members)
    archive_bytes = capsule_builder._write_archive_bytes(members, manifest)
    capsule_builder._verify_archive_bytes(
        archive_bytes,
        members=members,
        manifest=manifest,
    )

    tampered = bytearray(archive_bytes)
    tampered[10] ^= 1  # local DOS time; central-directory metadata remains intact
    with zipfile.ZipFile(io.BytesIO(tampered)) as archive:
        assert archive.testzip() is None
        assert archive.read(member.relative_path) == member.payload
    with pytest.raises(CapsuleBuildError, match="local header"):
        capsule_builder._verify_archive_bytes(
            bytes(tampered),
            members=members,
            manifest=manifest,
        )
