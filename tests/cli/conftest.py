"""Fixtures for the CLI tests: a migrated state database and a seeded run with a known event log.

Event payloads start from the S0.2 schema fixtures (tests/fixtures/schemas/Event/) and get explicit
timestamps, so durations and offsets are exact. Nothing here touches the network or `.env`.
"""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from hypora.core.schemas import AnyEvent, RunStatus
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.models import Run
from hypora.storage.records import create_run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/schemas/Event"
START = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
RUN_ID = "run_cli"
SECRET_ARGUMENT = "ARGUMENT_VALUE_THAT_MUST_NOT_BE_PRINTED"


@pytest.fixture(autouse=True)
def _no_inherited_config_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_") or name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)


def payload(name: str, **changes: Any) -> dict[str, Any]:
    """The payload of the schema fixture `<name>.json`, with top-level fields changed."""
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())["payload"]
    data.update(changes)
    return data


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = Database(tmp_path / "hypora.sqlite")
    migrate(opened)
    yield opened
    opened.close()


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


class Seeder:
    """Appends events to one run at `START + at` seconds."""

    def __init__(self, database: Database, run_id: str = RUN_ID) -> None:
        self.database = database
        self.run_id = run_id
        self.log = EventLog(database)
        create_run(
            database,
            run_id,
            "ws_test",
            budget_usd=Decimal("2.00"),
            budget_compute_s=Decimal("3600"),
            created_at=START,
            updated_at=START,
        )

    def add(self, type: str, role: str, data: dict[str, Any], at: float, **ids: Any) -> AnyEvent:
        return self.log.append(
            self.run_id, type, role, data, created_at=START + timedelta(seconds=at), **ids
        )

    def set_status(self, status: RunStatus) -> None:
        from sqlalchemy import update

        with self.database.write() as connection:
            connection.execute(update(Run).where(Run.id == self.run_id).values(status=status))


def seed_timeline(database: Database) -> list[AnyEvent]:
    """A short run touching every duration rule (see hypora/cli/trace.py)."""
    s = Seeder(database)
    handoff = payload("handoff", role_instance_id="ri_intake")
    handoff["handoff"].update(run_id=RUN_ID, from_role="intake")
    error_result = payload("tool_result_error")
    return [
        s.add(
            "stage_transition",
            "controller",
            {"stage": "INTAKE", "status": "INTAKE", "reason": "run created"},
            0.0,
        ),
        s.add("role_started", "intake", {"role_instance_id": "ri_intake", "tier": "cheap"}, 0.1),
        s.add(
            "model_call",
            "intake",
            payload(
                "model_call",
                latency_ms=900.0,
                input_tokens=500,
                output_tokens=50,
                cost_usd=0.0003,
                rate_limit_wait_ms=2500.0,
                retries=2,
            ),
            1.0,
        ),
        s.add("handoff", "intake", handoff, 1.1),
        s.add(
            "stage_transition",
            "controller",
            {
                "previous_stage": "INTAKE",
                "stage": "PROFILE",
                "previous_status": "INTAKE",
                "status": "INTAKE",
                "reason": "intake_done",
            },
            1.2,
        ),
        s.add(
            "tool_call",
            "single_agent",
            payload(
                "tool_call",
                tool="write_trial_files",
                arguments={"pipeline_py": SECRET_ARGUMENT, "design": {"kind": "baseline"}},
            ),
            2.0,
            trial_id="trl_1",
        ),
        s.add(
            "tool_result",
            "single_agent",
            {
                "call_id": "call_0412",
                "tool": "write_trial_files",
                "status": "ok",
                "latency_ms": 120.0,
            },
            2.2,
            trial_id="trl_1",
        ),
        s.add(
            "job_transition",
            "execution_worker",
            payload(
                "job_transition",
                job_id="job_1",
                trial_id="trl_1",
                expected_prior_state="QUEUED",
                to_state="LEASED",
            ),
            3.0,
            job_id="job_1",
            trial_id="trl_1",
        ),
        s.add(
            "job_transition",
            "execution_worker",
            payload(
                "job_transition",
                job_id="job_1",
                trial_id="trl_1",
                expected_prior_state="LEASED",
                to_state="RUNNING",
            ),
            3.5,
            job_id="job_1",
            trial_id="trl_1",
        ),
        s.add(
            "job_transition",
            "execution_worker",
            payload(
                "job_transition",
                job_id="job_1",
                trial_id="trl_1",
                expected_prior_state="RUNNING",
                to_state="COMPLETED",
            ),
            10.5,
            job_id="job_1",
            trial_id="trl_1",
        ),
        s.add("tool_result", "mcp_server", error_result, 11.0),
        s.add(
            "error",
            "controller",
            {
                "component": "controller",
                "code": "RUN_LIMIT_REACHED",
                "message": "run stopped: max_steps reached (5 of 5)",
            },
            12.0,
        ),
        s.add(
            "stage_transition",
            "controller",
            {
                "previous_stage": "PROFILE",
                "stage": None,
                "previous_status": "INTAKE",
                "status": "FAILED",
                "reason": "fail",
            },
            13.0,
        ),
    ]
