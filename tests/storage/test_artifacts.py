"""Atomic publication, immutable references, and a real process killed mid-write."""

import hashlib
import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import func, select

from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.models import Artifact


def _crash_writer(path: str, root: str, staged) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"partial" * 100000
        staged.set()
        # Wait until the parent kills this process, after at least one real write.
        multiprocessing.Event().wait(60)
        yield b"should never be published"

    ArtifactStore(Database(Path(path)), Path(root)).write("crash", "trial_1", "model", chunks())


def test_atomic_publish_hash_and_idempotent_retry(database: Database, tmp_path: Path) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    artifact = store.write("model_1", "trial_1", "model", [b"hello", b" world"])
    assert artifact.path.read_bytes() == b"hello world"
    assert artifact.sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert store.get("model_1") == artifact
    assert store.write("model_1", "trial_1", "model", [b"hello world"]) == artifact
    assert list(artifact.path.parent.glob(".pending-*")) == []
    with pytest.raises(ValueError, match="different content"):
        store.write("model_1", "trial_1", "model", [b"changed"])
    assert store.get("model_1") == artifact
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Artifact)) == 1


def test_source_exception_leaves_no_partial_artifact(database: Database, tmp_path: Path) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"partial" * 100000
        raise RuntimeError("source interrupted")

    store = ArtifactStore(database, tmp_path / "artifacts")
    with pytest.raises(RuntimeError):
        store.write("interrupted", "trial_1", "model", chunks())
    assert not (store.root / "interrupted/content").exists()
    assert list((store.root / "interrupted").iterdir()) == []
    with pytest.raises(KeyError):
        store.get("interrupted")


def test_killed_writer_never_publishes_partial_file(database: Database, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(database, root)
    good = store.write("existing", "trial_1", "model", [b"already committed"])
    context = multiprocessing.get_context("spawn")
    staged = context.Event()
    process = context.Process(
        target=_crash_writer, args=(str(database.engine.url.database), str(root), staged)
    )
    try:
        process.start()
        assert staged.wait(timeout=20), "child never reached the partial-write boundary"
        assert list((root / "crash").glob(".pending-*"))
        assert not (root / "crash/content").exists()
        process.kill()
        process.join(timeout=10)
        assert process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
    assert not (root / "crash/content").exists()
    with pytest.raises(KeyError):
        store.get("crash")
    assert store.get("existing") == good
    # Retry ignores the abandoned hidden temp file and publishes complete bytes.
    recovered = store.write("crash", "trial_1", "model", [b"complete retry"])
    assert recovered.path.read_bytes() == b"complete retry"


def test_crash_after_replace_recovers_complete_orphan(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    real_replace = os.replace

    def crash_after_replace(source, destination) -> None:
        real_replace(source, destination)
        raise RuntimeError("crash before database commit")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", crash_after_replace)
        with pytest.raises(RuntimeError):
            store.write("orphan", "trial_1", "model", [b"complete"])
    assert (store.root / "orphan/content").read_bytes() == b"complete"
    with pytest.raises(KeyError):
        store.get("orphan")
    result = store.write("orphan", "trial_1", "model", [b"complete"])
    assert store.get("orphan") == result


def test_file_is_fsynced_before_replace(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        calls.append("fsync")
        real_fsync(fd)

    def replace(source, destination) -> None:
        assert source.parent == destination.parent
        calls.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    ArtifactStore(database, tmp_path / "artifacts").write("one", "trial_1", "model", [b"complete"])
    assert calls[:2] == ["fsync", "replace"]
    if os.name == "posix":
        assert calls == ["fsync", "replace", "fsync"]


def test_tampering_and_unknown_trial(database: Database, tmp_path: Path) -> None:
    store = ArtifactStore(database, tmp_path / "artifacts")
    with pytest.raises(KeyError):
        store.write("unknown", "absent", "model", [b"bytes"])
    assert not (store.root / "unknown/content").exists()
    result = store.write("one", "trial_1", "model", [b"bytes"])
    result.path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.get("one")
