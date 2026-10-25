"""The output reader never follows links (D-011). Host-only: no containers needed."""

import hashlib
import os
from pathlib import Path

import pytest

from hypora.execution.sandbox.outputs import (
    UnsafeOutputError,
    directory_bytes,
    open_output,
    read_output_bytes,
    remove_unsafe,
    scan_outputs,
)


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    path = tmp_path / "host-secret.txt"
    path.write_text("HOST SECRET", encoding="utf-8")
    return path


@pytest.fixture
def out(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    (root / "nested").mkdir(parents=True)
    (root / "metrics.json").write_text('{"mae": 1.5}', encoding="utf-8")
    (root / "nested" / "preds.csv").write_text("y\n1\n", encoding="utf-8")
    return root


def test_regular_files_are_listed_with_size_and_hash(out: Path) -> None:
    manifest = scan_outputs(out)
    assert manifest.safe
    assert [(f.path, f.size) for f in manifest.files] == [
        ("metrics.json", 12),
        ("nested/preds.csv", 4),
    ]
    assert manifest.files[0].sha256 == hashlib.sha256(b'{"mae": 1.5}').hexdigest()
    assert manifest.total_bytes == directory_bytes(out) == 16


def test_symlinks_are_rejected_and_never_followed(out: Path, secret: Path) -> None:
    (out / "model.bin").symlink_to(secret)  # absolute link to a host file
    (out / "nested" / "rel").symlink_to("../../host-secret.txt")  # relative escape
    (out / "linkdir").symlink_to(secret.parent, target_is_directory=True)  # link to a directory
    manifest = scan_outputs(out)
    assert not manifest.safe
    assert sorted((e.path, e.kind) for e in manifest.rejected) == [
        ("linkdir", "symlink"),
        ("model.bin", "symlink"),
        ("nested/rel", "symlink"),
    ]
    assert all(f.sha256 != hashlib.sha256(b"HOST SECRET").hexdigest() for f in manifest.files)
    assert directory_bytes(out) == 16  # links are not counted (or followed) either


def test_special_files_are_rejected(out: Path) -> None:
    os.mkfifo(out / "pipe")
    assert [(e.path, e.kind) for e in scan_outputs(out).rejected] == [("pipe", "fifo")]


def test_remove_unsafe_unlinks_the_link_not_its_target(out: Path, secret: Path) -> None:
    (out / "model.bin").symlink_to(secret)
    os.mkfifo(out / "nested" / "pipe")
    remove_unsafe(out, scan_outputs(out))
    assert not (out / "model.bin").is_symlink() and not (out / "nested" / "pipe").exists()
    assert secret.read_text(encoding="utf-8") == "HOST SECRET"  # target untouched
    assert scan_outputs(out).safe


def test_open_output_reads_regular_files(out: Path) -> None:
    with open_output(out, "nested/preds.csv") as handle:
        assert handle.read() == b"y\n1\n"
    assert read_output_bytes(out, "metrics.json", max_bytes=100) == b'{"mae": 1.5}'
    with pytest.raises(UnsafeOutputError, match="larger than"):
        read_output_bytes(out, "metrics.json", max_bytes=5)


def test_open_output_refuses_a_final_symlink(out: Path, secret: Path) -> None:
    (out / "model.bin").symlink_to(secret)
    with pytest.raises(UnsafeOutputError):
        open_output(out, "model.bin")


def test_open_output_refuses_a_symlinked_directory_on_the_way(
    out: Path, secret: Path, tmp_path: Path
) -> None:
    (out / "linkdir").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(UnsafeOutputError, match="not a real directory"):
        open_output(out, f"linkdir/{secret.name}")


@pytest.mark.parametrize("bad", ["/etc/passwd", "../host-secret.txt", "nested/../metrics.json", ""])
def test_open_output_refuses_non_plain_paths(out: Path, bad: str) -> None:
    with pytest.raises(UnsafeOutputError):
        open_output(out, bad)


def test_a_symlinked_root_is_refused(out: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(UnsafeOutputError, match="not a real directory"):
        scan_outputs(alias)
