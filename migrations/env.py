"""Alembic environment sharing the runtime WAL/foreign-key/transaction configuration."""

from pathlib import Path

from alembic import context
from sqlalchemy.engine import make_url

from hypora.storage.database import Database
from hypora.storage.models import Base

config = context.config


def run_migrations() -> None:
    """Apply migrations with an injected transaction or a configured SQLite database."""
    if context.is_offline_mode():
        context.configure(
            url=config.get_main_option("sqlalchemy.url"),
            target_metadata=Base.metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    url = make_url(config.get_main_option("sqlalchemy.url"))
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise ValueError("the local runtime requires a file-backed SQLite database")
    database = Database(Path(url.database))
    try:
        with database.write() as connection:
            config.attributes["connection"] = connection
            context.configure(
                connection=connection, target_metadata=Base.metadata, compare_type=True
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        database.close()


run_migrations()
