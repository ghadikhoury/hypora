"""Immutable local artifacts: same-directory staging, fsync, replace, and hashed DB records."""

import hashlib
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import insert, select

from hypora.storage.database import Database
from hypora.storage.models import Artifact, Trial


@dataclass(frozen=True)
class StoredArtifact:
    """Artifact metadata returned after the file and database record are committed."""

    id: str
    trial_id: str
    path: Path
    sha256: str
    kind: str


def _fsync_directory(directory: Path) -> None:
    # Windows does not support opening directories this way. On POSIX this also persists rename.
    if os.name == "posix":
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class ArtifactStore:
    """Content is immutable per artifact ID; changed content requires a new version/ID."""

    def __init__(self, database: Database, root: Path) -> None:
        self.database = database
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        artifact_id: str,
        trial_id: str,
        kind: str,
        chunks: Iterable[bytes],
    ) -> StoredArtifact:
        """Publish complete bytes atomically, then record their hash under a write transaction.

        A crash before replace leaves only an unreferenced hidden temp file; a crash between
        replace and DB commit can leave a complete orphan. Retrying the same ID/content recovers
        that orphan; readers use committed DB references, never directory listings.
        """
        # IDs become one filename component, never an arbitrary caller-selected path.
        from pydantic import TypeAdapter

        from hypora.core.schemas import Identifier

        TypeAdapter(Identifier).validate_python(artifact_id)
        directory = self.root / artifact_id
        directory.mkdir(exist_ok=True)
        destination = directory / "content"
        fd, filename = tempfile.mkstemp(prefix=".pending-", dir=directory)
        temporary = Path(filename)
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as stream:
                for chunk in chunks:
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            sha256 = digest.hexdigest()
            with self.database.write() as connection:
                if connection.scalar(select(Trial.id).where(Trial.id == trial_id)) is None:
                    raise KeyError(f"unknown trial: {trial_id}")
                existing = (
                    connection.execute(select(Artifact).where(Artifact.id == artifact_id))
                    .mappings()
                    .one_or_none()
                )
                if existing is not None and (
                    existing.trial_id != trial_id
                    or existing.sha256 != sha256
                    or existing.kind != kind
                    or existing.path != str(destination)
                ):
                    raise ValueError("artifact ID already records different content or metadata")
                if destination.exists():
                    with destination.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
                            raise ValueError("existing artifact file has a different hash")
                else:
                    os.replace(temporary, destination)
                    _fsync_directory(directory)
                if existing is None:
                    connection.execute(
                        insert(Artifact).values(
                            id=artifact_id,
                            trial_id=trial_id,
                            kind=kind,
                            sha256=sha256,
                            path=str(destination),
                        )
                    )
            return StoredArtifact(artifact_id, trial_id, destination, sha256, kind)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, artifact_id: str) -> StoredArtifact:
        """Load only a committed reference and verify the file still matches its recorded hash."""
        with self.database.engine.connect() as connection:
            row = connection.execute(select(Artifact).where(Artifact.id == artifact_id)).mappings()
            record = row.one_or_none()
            if record is None:
                raise KeyError(f"unknown artifact: {artifact_id}")
        path = Path(record.path)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record.sha256:
                raise ValueError("artifact hash mismatch")
        return StoredArtifact(record.id, record.trial_id, path, record.sha256, record.kind)
