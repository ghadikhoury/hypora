"""Each test uses a real file-backed database built only through Alembic."""

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import insert

from hypora.core.schemas import TrialKind
from hypora.storage.database import Database
from hypora.storage.migrate import migrate
from hypora.storage.models import Run, Trial


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """Provide a migrated database with two independent runs and one trial."""
    database = Database(tmp_path / "storage.sqlite")
    migrate(database)
    with database.write() as connection:
        connection.execute(
            insert(Run),
            [
                {
                    "id": "run_1",
                    "workspace_id": "ws_1",
                    "budget_usd": Decimal("2.00"),
                    "budget_compute_s": Decimal("10.00"),
                },
                {
                    "id": "run_2",
                    "workspace_id": "ws_1",
                    "budget_usd": None,
                    "budget_compute_s": None,
                },
            ],
        )
        connection.execute(
            insert(Trial).values(
                id="trial_1",
                run_id="run_1",
                kind=TrialKind.CANDIDATE,
                hypothesis="synthetic storage fixture",
                status="DRAFT",
                dir="runs/run_1/trials/trial_1",
            )
        )
    yield database
    database.close()
