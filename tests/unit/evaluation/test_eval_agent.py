"""Agent-system wiring uses scripted/stub collaborators only; no calls or Docker."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hypora.core.controller.controller import RunRequest, RunResult
from hypora.core.controller.states import FAILED
from hypora.core.llm import LLMAdapter
from hypora.core.schemas import ResourceConstraints
from hypora.evaluation.runner.agent import (
    AgentSystem,
    UnsupportedCase,
    bounded_run_config,
    check_supported,
)
from hypora.evaluation.runner.runner import system_input_for
from hypora.evaluation.system import SystemContext
from tests.unit.evaluation.helpers import clean_table
from tests.unit.evaluation.test_eval_runner import plan_at
from tests.unit.evaluation.test_hidden_split import created


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [False, True])
async def test_agent_passes_case_fields_and_preserves_retry_time_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: bool
) -> None:
    plan = plan_at(tmp_path / "plan")
    split, bundle = created(tmp_path / "data", clean_table())
    case = bundle[0]
    system_input = system_input_for(case, split)
    requests: list[RunRequest] = []
    seen: list[dict[str, Any]] = []

    class Controller:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

        async def run(self, request: RunRequest) -> RunResult:
            requests.append(request)
            if crash:
                raise RuntimeError("controlled controller failure")
            return RunResult(
                run_id=request.run_id or "missing",
                state=FAILED,
                final_trial_id=None,
                verified=False,
                report_json=tmp_path / "report.json",
                report_md=tmp_path / "report.md",
                notes=("expected incomplete run",),
                limits_fired=(),
            )

    monkeypatch.setattr("hypora.evaluation.runner.agent.RunController", Controller)
    system = AgentSystem(
        adapter_factory=lambda *_: cast(LLMAdapter, SimpleNamespace(retry_wait_s=3.5))
    )
    context = SystemContext(
        run_id="run_agent_unit",
        runs_root=tmp_path / "runs",
        database=tmp_path / "state.sqlite",
        run_config=plan.run_config,
    )
    if crash:
        with pytest.raises(RuntimeError, match="controlled controller"):
            await system.run(system_input, context)
    else:
        output = await system.run(system_input, context)
        assert not output.completed and output.final_model is None
        assert not any("secondary_metrics" in note for note in output.notes)
    assert system.retry_wait_seconds(context.run_id) == 3.5
    [request] = requests
    assert request.dataset == system_input.dataset
    assert request.budget == case.budget and request.permitted_tools == case.permitted_tools
    assert request.run_id == context.run_id
    assert request.target == case.constraints.target
    assert request.secondary_metrics == case.constraints.secondary_metrics
    assert request.goal.startswith(case.request)
    assert request.visible_constraints == case.constraints.model_dump_json()
    assert "secondary_metrics" in request.goal
    bounded = bounded_run_config(plan.run_config, case.contract_bounds)
    assert seen[0]["run_config"].budgets.job_limits == bounded.budgets.job_limits
    assert (
        bounded.budgets.job_limits.max_memory_gb
        <= case.contract_bounds.max_job_limits.max_memory_gb
    )


def test_unsupported_hard_requirement_refuses_before_model_calls(tmp_path: Path) -> None:
    plan = plan_at(tmp_path / "plan")
    split, bundle = created(tmp_path / "data", clean_table())
    case = bundle[0]
    constraints = case.constraints.model_copy(
        update={"resource_constraints": ResourceConstraints(max_model_size_mb=1)}
    )
    system_input = system_input_for(case, split).model_copy(update={"constraints": constraints})
    context = SystemContext(
        run_id="run_hard_constraint",
        runs_root=tmp_path / "runs",
        database=tmp_path / "state.sqlite",
        run_config=plan.run_config,
    )
    with pytest.raises(UnsupportedCase, match="resource_constraints"):
        check_supported(system_input, context)
