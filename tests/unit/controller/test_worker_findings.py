"""Terminal job failures and scientific narrative survive the worker handoff."""

from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from typing import Any

import pytest

from hypora.core.controller import controller as module
from hypora.core.roles.single_agent import AgentOutcome
from hypora.core.roles.worker import WorkerReport
from hypora.core.schemas import JobState
from hypora.core.schemas.briefs import Brief
from hypora.storage import records


@pytest.mark.anyio
async def test_terminal_failure_and_scientific_questions_reach_c(monkeypatch, tmp_path):
    run: Any = module._Run.__new__(module._Run)
    run.run_id, run.run_dir = "run_review", tmp_path
    run.database, run.dataset = None, tmp_path / "data"
    run.guard = NS(cancel_requested=lambda: False)
    run.controller = NS(
        database_path=tmp_path / "db",
        runs_root=tmp_path,
        run_config=NS(flags=None),
        tool_channel_wrapper=None,
        adapter=None,
        max_agent_turns=5,
    )
    released: list[str] = []
    handoffs: list[Any] = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Channel:
        def __init__(self, client):
            pass

        async def specs(self):
            return []

    @asynccontextmanager
    async def scope(*args):
        yield

    async def agent(*args, **kwargs):
        return AgentOutcome(
            None,
            1,
            0,
            worker_report=WorkerReport(
                summary="Candidate fails on unseen groups",
                confidence=0.7,
                open_questions=["Need revised features"],
            ),
        )

    monkeypatch.setattr(module, "write_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mcp_transport", lambda *args: None)
    monkeypatch.setattr(module, "Client", Client)
    monkeypatch.setattr(module, "McpToolChannel", Channel)
    monkeypatch.setattr(module, "worker_scope", scope)
    monkeypatch.setattr(module, "run_single_agent", agent)
    monkeypatch.setattr(records, "load_trial", lambda *args: NS(batch_id="brf_one"))
    monkeypatch.setattr(module, "list_trials", lambda *args: [NS(trial_id="trl_one")])
    monkeypatch.setattr(
        module,
        "run_jobs",
        lambda *args: [
            NS(trial_id="trl_one", job_id="job_one", status=JobState.FAILED_TERMINAL),
        ],
    )
    run._role_started = lambda *args, **kwargs: "ri_one"
    run._validate_assignment = lambda text: None
    run._cancel = lambda *args: None
    run._handoff = lambda *args, **kwargs: handoffs.append(args[2])
    brief = Brief.model_validate(
        {
            "brief_id": "brf_one",
            "run_id": run.run_id,
            "issued_by": "agent_c",
            "assigned_role": "trial_worker",
            "objective": "Bounded work",
            "output_schema": "TrialFinding",
            "allowed_tools": ["submit_experiment"],
            "end_condition": "Stop after validation",
            "context_refs": [],
            "budget": {"max_tokens": 10000, "max_tool_calls": 5},
        }
    )
    record = NS(version=1, model_dump_json=lambda: "{}")
    finding = await run._worker(brief, NS(release=released.append), "ri_c", record, record, record)
    assert finding.status.value == "failed" and finding.outcome.value == "failed"
    assert finding.error_codes == ["WORKER_FAILED"]
    assert handoffs[0].value == "failed" and released == ["brf_one"]
    assert finding.summary == "Candidate fails on unseen groups"
    assert finding.open_questions == ["Need revised features"] and finding.confidence == 0.7
