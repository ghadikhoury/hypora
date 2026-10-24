"""SQLite WAL connections and explicit transactions for independent processes."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool


def _json_dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False)


def create_sqlite_engine(path: Path, *, timeout_seconds: float = 30) -> Engine:
    """Create a file-backed WAL engine with FK checks and a bounded lock wait.

    NullPool keeps connections process-local even if an engine is inherited across a fork.
    Autocommit mode disables sqlite3's implicit transactions; writes use BEGIN IMMEDIATE.
    """
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(path.resolve())),
        connect_args={"timeout": timeout_seconds, "check_same_thread": False},
        poolclass=NullPool,
        json_serializer=_json_dumps,
    )

    @event.listens_for(engine, "connect")
    def configure(connection: sqlite3.Connection, record: object) -> None:
        connection.isolation_level = None
        cursor = connection.cursor()
        try:
            mode = cursor.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise RuntimeError("SQLite WAL mode is required")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=FULL")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def begin(connection: Connection) -> None:
        # Make ORM sessions and multi-statement readers transactional too. Only storage writers
        # request IMMEDIATE, so ordinary readers do not reserve SQLite's single writer slot.
        statement = (
            "BEGIN IMMEDIATE"
            if connection.get_execution_options().get("storage_write")
            else "BEGIN"
        )
        connection.exec_driver_sql(statement)

    return engine


class Database:
    """A database handle; migrations, not create_all, create its tables."""

    def __init__(self, path: Path, *, timeout_seconds: float = 30) -> None:
        self.engine = create_sqlite_engine(path, timeout_seconds=timeout_seconds)

    @contextmanager
    def write(self) -> Iterator[Connection]:
        """Reserve SQLite's single writer before reading any sequence or budget total."""
        with self.engine.connect() as connection:
            connection = connection.execution_options(storage_write=True)
            with connection.begin():
                yield connection

    def close(self) -> None:
        """Dispose this process's database resources."""
        self.engine.dispose()
