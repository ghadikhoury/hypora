"""Run the committed Alembic history: uv run python -m hypora.storage.migrate <database>."""

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config

from hypora.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def migration_config() -> Config:
    """Find migration files independently of the caller's working directory."""
    config = Config(str(MIGRATIONS / "alembic.ini"))
    config.set_main_option("script_location", str(MIGRATIONS))
    return config


def migrate(database: Database, revision: str = "head") -> None:
    """Upgrade a database through Alembic, including from an empty file."""
    config = migration_config()
    with database.write() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def main() -> None:
    """Create or upgrade a SQLite database using the committed migration history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    database = Database(args.database)
    try:
        migrate(database)
    finally:
        database.close()


if __name__ == "__main__":
    main()
