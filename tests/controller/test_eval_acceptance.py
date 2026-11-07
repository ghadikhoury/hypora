"""S2.4's Done-when on real infrastructure: two fixture cases, the real agent and baseline.

The controller, stdio MCP server, experiment worker, sandbox, graders, and hidden split are all
real, on real UCI data. Only the model is scripted. One case injects a tool error the agent retries,
so the recovery proof is exercised on what a real worker records.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.core.config import load_settings_for_eval
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.llm import LLMAdapter, ProviderAdmission, load_price_table
from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.types import MessageRole
from hypora.core.schemas import ModelProfile, Provider
from hypora.evaluation.baselines.fixed_pipeline import FixedPipelineBaseline
from hypora.evaluation.graders import ProgrammaticGrader, SandboxPredictor
from hypora.evaluation.hidden_split import HiddenStore
from hypora.evaluation.runner.agent import AgentSystem
from hypora.evaluation.runner.runner import EvalPlan, run_eval
from hypora.evaluation.runner.store import CodeState, GitRepo
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from tests.controller import scripted
from tests.controller.conftest import unthrottled
from tests.controller.scripted import ScriptedRun
from tests.unit.evaluation.helpers import case_on, raw_case

pytestmark = [pytest.mark.anyio, pytest.mark.sandbox]

HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)
PARENT, CHILD = "a" * 40, "b" * 40
SEGMENT = "single_agent.dev"
WRITE_FAULT = {
    "kind": "tool_error",
    "tool": "write_trial_files",
    "occurrence": 1,
    "error_code": "TOOL_UNAVAILABLE",
    "expected": "recover",
}


class _Repo(GitRepo):
    def __init__(self, commit: str, parents: tuple[str, ...] = ()) -> None:
        self.commit = commit
        self.parents = parents

    def state(self) -> CodeState:
        return CodeState(self.commit, False)

    def ancestors(self, start: str, limit: int) -> list[str]:
        return list(self.parents[:limit])


class _RetryingRun(ScriptedRun):
    """Tries a failed `write_trial_files` once more, as an agent reading the error would."""

    async def _agent_turn(self, request: ProviderRequest) -> ProviderResult:
        last = request.messages[-1]
        if (
            last.role is MessageRole.TOOL
            and last.is_error
            and last.tool_name == "write_trial_files"
        ):
            return self._call(
                "write_trial_files",
                idempotency_key=f"{self.run_id}:write_retry",
                design=scripted.DESIGN,
                pipeline_py=scripted.LINEAR,
            )
        return await super()._agent_turn(request)


def _scripted_adapter(
    run_config: RunConfig, database: Database, _: Path, run_id: str
) -> LLMAdapter:
    models = unthrottled(run_config.models)
    return LLMAdapter(
        models=models,
        profile=run_config.model_profile,
        prices=load_price_table(),
        clients={Provider.GEMINI: _RetryingRun(run_id)},
        events=EventLog(database),
        admission=ProviderAdmission(models.rate_limits),
    )


def _plan(
    tmp_path: Path, cases: tuple[Any, ...], systems: tuple[Any, ...], repo: GitRepo
) -> EvalPlan:
    run_config = RunConfig.build(
        load_settings_for_eval("single_agent", profile=ModelProfile.DEV),
        eval_config="single_agent",
        hardware=HARDWARE,
    )
    return EvalPlan(
        cases=cases,
        systems=systems,
        grader=ProgrammaticGrader(SandboxPredictor()),
        config="single_agent",
        segment=SEGMENT,
        run_config=run_config,
        store=HiddenStore(tmp_path / "hidden"),
        work_root=tmp_path / "work",
        results_root=tmp_path / "results",
        snapshots_dir=tmp_path / "snapshots",
        planted_dir=tmp_path / "planted",
        repo=repo,
        max_parallel_cases=2,
    )


def _checks(record: dict[str, Any]) -> dict[str, str]:
    return {check["name"]: check["status"] for check in record["grade"]["checks"]}


async def test_two_cases_on_agent_and_baseline_give_records_a_summary_and_a_comparison(
    tmp_path: Path, sandbox_image: str
) -> None:
    frame = load_breast_cancer(as_frame=True).frame
    constraints = {
        "target": "target",
        "task_type": "binary_classification",
    }
    metric = {"name": "roc_auc", "threshold": 0.9, "reported_tolerance": 0.1}
    common = {"constraints": constraints, "grading": {"hidden_metric": metric}}
    tools = sorted({*raw_case()["permitted_tools"], "write_trial_files", "submit_experiment"})
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    clean = case_on(
        snapshots,
        frame,
        id="uci_clean",
        permitted_tools=tools,
        constraints=constraints,
        grading={
            "hidden_metric": {"name": "brier_score", "threshold": 0.2, "reported_tolerance": 0.1}
        },
    )
    fault = case_on(
        snapshots,
        frame,
        id="uci_tool_fault",
        permitted_tools=tools,
        injected_faults=[WRITE_FAULT],
        **common,
    )
    agent = AgentSystem(adapter_factory=_scripted_adapter, poll_seconds=0.5)
    baseline = FixedPipelineBaseline(poll_seconds=0.5)

    outcome = await run_eval(
        _plan(tmp_path, (clean, fault), (agent, baseline), _Repo(PARENT)), print
    )

    root = tmp_path / "results" / PARENT / SEGMENT
    assert outcome.directory == root
    records = {
        (system, case): json.loads((root / system / f"{case}.json").read_text())
        for system in ("agent", "baseline")
        for case in ("uci_clean", "uci_tool_fault")
    }
    assert records["agent", "uci_clean"]["grade"]["passed"], records["agent", "uci_clean"]
    assert records["baseline", "uci_clean"]["grade"]["passed"], records["baseline", "uci_clean"]
    assert clean.constraints.primary_metric is None and not clean.constraints.secondary_metrics
    metrics = records["agent", "uci_clean"]["outputs"]["final_model"]["reported_metrics"]
    assert "brier_score" in {metric["name"] for metric in metrics}

    # The agent retried, on a real worker, and its work is what the proof links to.
    recovered = records["agent", "uci_tool_fault"]
    assert recovered["grade"]["passed"], recovered
    assert _checks(recovered)["fault_tool_error_0"] == "passed"
    # A fixed pipeline has no retry: it stops, which a recover case doesn't accept.
    stopped = records["baseline", "uci_tool_fault"]
    assert not stopped["grade"]["passed"]
    assert _checks(stopped)["fault_tool_error_0"] == "failed"

    summary = (root / "summary.md").read_text()
    assert "Wilson" in summary and "Paired" in summary
    assert json.loads((root / "regressions.json").read_text())["status"] == "no_baseline"

    # The next commit compares itself with this one.
    child = await run_eval(_plan(tmp_path, (clean,), (baseline,), _Repo(CHILD, (PARENT,))), print)
    report = json.loads((child.directory / "regressions.json").read_text())
    assert report["base_commit"] == PARENT or PARENT in json.dumps(report)
    assert report["newly_failing"] == []
