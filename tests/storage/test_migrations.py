"""Migration parity, constraints, timestamps, and real transaction rollback."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import CheckConstraint, insert, inspect, select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from hypora.storage.database import Database
from hypora.storage.migrate import migrate, migration_config
from hypora.storage.models import Base, Run, TaskSpecRow


def test_empty_upgrade_matches_models_and_downgrade(tmp_path: Path) -> None:
    database = Database(tmp_path / "empty.sqlite")
    try:
        migrate(database)
        migrate(database)  # Repeat upgrade is harmless.
        with database.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
            assert connection.exec_driver_sql("PRAGMA synchronous").scalar() == 2
            assert set(inspect(connection).get_table_names()) == {
                *Base.metadata.tables,
                "alembic_version",
            }
            context = MigrationContext.configure(
                connection, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(context, Base.metadata) == []
            # Alembic does not diff CHECK constraints. Compare their SQL and names explicitly.
            for name, table in Base.metadata.tables.items():
                reflected = inspect(connection).get_check_constraints(name)
                expected = {
                    str(constraint.name): " ".join(
                        str(
                            constraint.sqltext.compile(
                                dialect=connection.dialect,
                                compile_kwargs={"literal_binds": True, "include_table": False},
                            )
                        ).split()
                    )
                    for constraint in table.constraints
                    if isinstance(constraint, CheckConstraint)
                }
                assert {
                    constraint["name"]: " ".join(constraint["sqltext"].split())
                    for constraint in reflected
                } == expected
                assert inspect(connection).get_pk_constraint(name)["constrained_columns"] == [
                    column.name for column in table.primary_key.columns
                ]
        with database.write() as connection:
            config = migration_config()
            config.attributes["connection"] = connection
            command.downgrade(config, "base")
        assert inspect(database.engine).get_table_names() == ["alembic_version"]
        migrate(database)
        assert "events" in inspect(database.engine).get_table_names()
    finally:
        database.close()


def test_writes_roll_back_and_foreign_keys_are_enforced(database: Database) -> None:
    with pytest.raises(RuntimeError), database.write() as connection:
        connection.execute(insert(Run).values(id="rolled_back", workspace_id="ws_1"))
        raise RuntimeError("abort")
    with database.engine.connect() as connection:
        assert connection.scalar(select(Run.id).where(Run.id == "rolled_back")) is None
    with pytest.raises(IntegrityError), database.write() as connection:
        connection.execute(
            insert(TaskSpecRow).values(id="invalid", run_id="absent", version=1, json={})
        )


def test_document_version_is_unique_per_run(database: Database) -> None:
    with database.write() as connection:
        connection.execute(insert(TaskSpecRow).values(id="one", run_id="run_1", version=1, json={}))
    with pytest.raises(IntegrityError), database.write() as connection:
        connection.execute(insert(TaskSpecRow).values(id="two", run_id="run_1", version=1, json={}))


def test_timestamps_normalize_utc_and_reject_naive(database: Database) -> None:
    stamp = datetime(2026, 10, 23, 10, 0, tzinfo=timezone(timedelta(hours=-4)))
    with database.write() as connection:
        connection.execute(insert(Run).values(id="time", workspace_id="ws_1", created_at=stamp))
    with Session(database.engine) as session:
        row = session.get(Run, "time")
        assert row is not None
        assert row.created_at == stamp.astimezone(UTC)
        assert row.created_at.tzinfo is UTC
        assert row.updated_at.tzinfo is UTC
    with pytest.raises(StatementError, match="timezone-aware"), database.write() as connection:
        connection.execute(
            insert(Run).values(
                id="naive", workspace_id="ws_1", created_at=stamp.replace(tzinfo=None)
            )
        )


def test_orm_session_is_really_transactional(database: Database) -> None:
    with Session(database.engine) as session:
        session.add(Run(id="orm_rollback", workspace_id="ws_1"))
        session.flush()
        session.rollback()
    with Session(database.engine) as session:
        assert session.get(Run, "orm_rollback") is None
