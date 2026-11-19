"""S4.2d (D-052): the first plan approval covers one comparison model, with no second prompt.

Scripted models with the scripted sandbox and an in-process worker, under the `interactive`
policy: the scripted user answers only the plan requests that are really pending. A second
confirmed column gets a comparison model only after its own approval through the plan gate
(D-050), never automatically, and the test split is scored once, for the recommended model only.
Synthetic tables built with the dev case's planting function (D-015 allows them in tests); the
real planted credit-g file runs where the OpenML 46918 snapshot is stored.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config.flags import Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller import approval_gate
from hypora.core.controller import controller as controller_module
from hypora.core.controller.approval_gate import (
    pending_approval,
    user_decision,
    write_decision,
)
from hypora.core.controller.controller import RunController, RunRequest
from hypora.core.controller.leakage import UNKNOWN_RANGE_LIMITATION
from hypora.core.controller.states import DONE
from hypora.core.llm.providers.fake import json_result
from hypora.core.llm.types import MessageRole
from hypora.core.roles.reporter import Report, build_report
from hypora.core.schemas import ApprovalPolicy
from hypora.core.schemas.events import (
    ApprovalGrantedEvent,
    ApprovalRequestedEvent,
    LeakageCheckEvent,
)
from hypora.core.schemas.leakage import LeakageOutcome, LeakageStep
from hypora.evaluation.cases.dev.plant_target_leakage import LEAK, TARGET, plant
from hypora.execution.protocol import experiments as protocol
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import list_trials, load_experiment_contract, run_jobs
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory
from tests.controller.scripted import DESIGN, LINEAR
from tests.controller.test_integrity import MultiWithTrials, _multi
from tests.controller.test_leakage_flag import (
    CREDIT_SNAPSHOT,
    HANG_GUARD,
    LeakageScript,
    _credit_like,
    _steps,
    _write,
    needs_credit,
)
from tests.controller.test_leakage_review_e2e import SECOND, _two_leaks

pytestmark = pytest.mark.anyio


def _interactive(controller: RunController, run_config: RunConfig) -> None:
    _multi(controller, run_config, injection_filter=True)
    flags = controller.run_config.flags.model_copy(
        update={"leakage_flag": Toggle.ON, "approval_policy": ApprovalPolicy.INTERACTIVE}
    )
    controller.run_config = controller.run_config.model_copy(update={"flags": flags})


async def _answer(database: Database, runs_root: Path, run_id: str, version: int, ok: bool) -> Any:
    """Wait until plan `version` is pending, then approve or deny it as the user."""
    while True:
        pending = pending_approval(database, run_id)
        if pending is not None and pending.plan_version == version:
            break
        await anyio.sleep(0.05)
    write_decision(
        runs_root,
        user_decision(pending, approve=ok, user="tester", reason=None if ok else "not now"),
    )
    return pending


async def _run_as_user(
    controller: RunController, dataset: Path, run_id: str, user: Callable[[], Any]
) -> Any:
    holder: dict[str, Any] = {}

    async def run() -> None:
        request = RunRequest(
            dataset=dataset, goal="Predict good or bad customers for underwriting", run_id=run_id
        )
        holder["result"] = await controller.run(request)

    with anyio.fail_after(HANG_GUARD):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await user()
    return holder["result"]


def _events(database: Database, run_id: str, kind: type) -> list[Any]:
    return [e for e in EventLog(database).read(run_id) if isinstance(e, kind)]


def _count_gate_decisions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the plan version of every request that goes through the gate's decision."""
    seen: list[int] = []
    real = approval_gate.ApprovalGate.decide

    async def counting(self: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(plan.version)
        return await real(self, plan, *args, **kwargs)

    monkeypatch.setattr(approval_gate.ApprovalGate, "decide", counting)
    return seen


def _finals(runs_root: Path, database: Database, run_id: str) -> list[str]:
    return [
        job.trial_id
        for job in run_jobs(database, run_id)
        if protocol.read_request(protocol.job_dir(runs_root, run_id, job.job_id)).phase
        is protocol.ExperimentPhase.FINAL
    ]


def _assert_one_covered_comparison(
    result: Any,
    database: Database,
    runs_root: Path,
    run_id: str,
    candidate: str | None,
    gate: list[int],
) -> None:
    assert result.state == DONE and result.verified, result.notes
    # The user was asked once (plan v1); plan v2, which leaves the column out, is covered.
    assert gate == [1]
    requested = _events(database, run_id, ApprovalRequestedEvent)
    granted = _events(database, run_id, ApprovalGrantedEvent)
    assert [e.payload.approval_id for e in requested] == [
        f"appr_{run_id}_plan_v1",
        f"appr_{run_id}_plan_v2",
    ]
    assert f"covered by appr_{run_id}_plan_v1" in requested[1].payload.summary
    assert "also covers training one comparison model" in requested[0].payload.summary
    assert [e.payload.approver for e in granted] == ["user:tester", "user:tester"]
    assert requested[1].seq + 2 == granted[1].seq  # request, policy decision, grant: never pending
    assert pending_approval(database, run_id) is None
    # Exactly one comparison trial, and it is the recommended final model.
    steps = _steps(database, run_id)
    assert steps.count(("comparison", "trained")) == 1
    assert steps[-1] == ("decision", "recommend_without")
    trials = {t.trial_id for t in list_trials(database, run_id)}
    decision = _events(database, run_id, LeakageCheckEvent)[-1].payload.section
    assert decision is not None and decision.without_columns is not None
    comparison = decision.without_columns.trial_id
    assert decision.without_columns.without_columns == [LEAK]
    assert candidate in trials and comparison in trials and len(trials) == 3
    assert result.final_trial_id == decision.recommended_trial_id == comparison
    # The test split is scored once, for the recommended model only.
    assert _finals(runs_root, database, run_id) == [comparison]
    contract = load_experiment_contract(database, run_id)
    assert contract.version == 2 and LEAK not in contract.data_contract.allowed_features
    # The report shows both models, what triggered the check, the column and why.
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage == decision
    assert decision.with_columns.trial_id == candidate
    markdown = result.report_md.read_text()
    assert "well above the expected range" in markdown
    assert f"`{LEAK}`: alone it scores" in markdown and "Shuffling it" in markdown
    assert "use the model without it" in markdown
    assert "The plan approval covers training one comparison model" in markdown


async def test_the_first_approval_covers_the_comparison_without_a_second_prompt(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    gate = _count_gate_decisions(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_covered"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    _interactive(controller, run_config)

    async def user() -> None:
        await _answer(database, runs_root, run_id, 1, ok=True)

    result = await _run_as_user(controller, dataset, run_id, user)
    _assert_one_covered_comparison(result, database, runs_root, run_id, script.trial_id, gate)


@pytest.mark.parametrize("approve_second", [False, True])
async def test_a_second_confirmed_column_gets_a_comparison_only_after_its_own_approval(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approve_second: bool,
) -> None:
    scripted_sandbox.install(monkeypatch)
    gate = _count_gate_decisions(monkeypatch)
    # Both screened candidates are confirmed, whatever their drops (the rule has its own tests).
    monkeypatch.setattr(
        controller_module, "confirm_columns", lambda drops, *a, **k: dict.fromkeys(drops, True)
    )
    dataset = _write(tmp_path, _two_leaks(), "two_leaks.csv")
    run_id = f"run_leak_second_{int(approve_second)}"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    _interactive(controller, run_config)
    seen: dict[str, Any] = {}

    async def user() -> None:
        await _answer(database, runs_root, run_id, 1, ok=True)
        while True:  # the second comparison waits for the user: nothing trains meanwhile
            pending = pending_approval(database, run_id)
            if pending is not None and pending.plan_version == 2:
                break
            await anyio.sleep(0.05)
        await anyio.sleep(1.0)
        seen["trials"] = len(list_trials(database, run_id))
        seen["summary"] = pending.summary
        await _answer(database, runs_root, run_id, 2, ok=approve_second)

    result = await _run_as_user(controller, dataset, run_id, user)
    assert result.state == DONE and result.verified, result.notes
    assert seen["trials"] == 3  # baseline, candidate and the covered comparison only
    assert "covers one comparison model only" in seen["summary"]
    assert gate == [1, 2]  # the second comparison went through the gate; nothing else did
    trials = list_trials(database, run_id)
    report = Report.model_validate_json(result.report_json.read_text())
    section = report.leakage
    assert section is not None and section.without_columns is not None
    by_name = {c.column: c for c in section.columns}
    top = section.columns[0].column
    other = SECOND if top == LEAK else LEAK
    assert set(by_name) == {LEAK, SECOND}
    assert result.final_trial_id == section.recommended_trial_id == section.without_columns.trial_id
    assert _finals(runs_root, database, run_id) == [result.final_trial_id]
    contract = load_experiment_contract(database, run_id)
    assert top not in contract.data_contract.allowed_features
    markdown = result.report_md.read_text()
    if approve_second:
        assert len(trials) == 4
        granted = next(
            e
            for e in _events(database, run_id, ApprovalGrantedEvent)
            if e.payload.approval_id == f"appr_{run_id}_plan_v2"
        )
        submitted = [
            e
            for e in _events(database, run_id, LeakageCheckEvent)
            if e.payload.step is LeakageStep.COMPARISON
            and e.payload.outcome is LeakageOutcome.SUBMITTED
        ]
        assert len(submitted) == 2 and submitted[0].seq < granted.seq < submitted[1].seq
        assert section.without_columns.without_columns == [top, other]
        assert by_name[other].comparison is LeakageOutcome.TRAINED
        assert by_name[other].in_final_model is False
        assert contract.version == 2 and other not in contract.data_contract.allowed_features
    else:
        assert len(trials) == 3  # no automatic comparison for the second column
        assert section.without_columns.without_columns == [top]
        assert by_name[other].comparison is LeakageOutcome.EXCLUSION_DENIED
        assert by_name[other].in_final_model is True and by_name[top].in_final_model is False
        # The covered plan takes the next free version: v2 was denied and keeps its ID.
        assert contract.version == 3 and other in contract.data_contract.allowed_features
        assert f"still uses `{other}`" in markdown
        assert "that plan was not approved" in markdown
        assert ("comparison", "exclusion_denied") in _steps(database, run_id)


async def test_an_approved_second_comparison_that_fails_keeps_the_covered_one(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    monkeypatch.setattr(
        controller_module, "confirm_columns", lambda drops, *a, **k: dict.fromkeys(drops, True)
    )
    real = controller_module._Run._comparison

    async def failing_second(
        self: Any, contract: Any, trial: Any, job: Any, withheld: Any, metric: Any
    ) -> Any:
        if len(withheld) > 1:
            return LeakageOutcome.FAILED, None, None
        return await real(self, contract, trial, job, withheld, metric)

    monkeypatch.setattr(controller_module._Run, "_comparison", failing_second)
    dataset = _write(tmp_path, _two_leaks(), "two_leaks.csv")
    run_id = "run_leak_second_failed"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    _interactive(controller, run_config)

    async def user() -> None:
        await _answer(database, runs_root, run_id, 1, ok=True)
        await _answer(database, runs_root, run_id, 2, ok=True)

    result = await _run_as_user(controller, dataset, run_id, user)
    assert result.state == DONE and result.verified, result.notes
    report = Report.model_validate_json(result.report_json.read_text())
    section = report.leakage
    assert section is not None and section.without_columns is not None
    top, other = section.columns[0].column, section.columns[1].column
    assert section.columns[1].comparison is LeakageOutcome.FAILED
    assert section.columns[1].in_final_model is True
    assert section.without_columns.without_columns == [top]
    assert result.final_trial_id == section.without_columns.trial_id
    assert _finals(runs_root, database, run_id) == [result.final_trial_id]
    # The approved v2 was never used; the covered plan took v3 under the v1 approval.
    contract = load_experiment_contract(database, run_id)
    assert contract.version == 3 and contract.approval_id == f"appr_{run_id}_plan_v3"
    assert top not in contract.data_contract.allowed_features
    assert other in contract.data_contract.allowed_features
    requested = _events(database, run_id, ApprovalRequestedEvent)
    assert f"covered by appr_{run_id}_plan_v1" in requested[-1].payload.summary


@needs_credit
async def test_the_real_planted_credit_g_file_is_compared_without_a_second_prompt(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    gate = _count_gate_decisions(monkeypatch)
    credit = pd.read_parquet(CREDIT_SNAPSHOT)
    published = (0.75, 0.82)  # typical credit-g ROC-AUC; see docs/notes/leakage-flag.md
    for name, frame in (("planted", plant(credit)), ("clean", credit)):
        gate.clear()
        dataset = _write(tmp_path, frame, f"credit_{name}.csv")
        run_id = f"run_leak_np_credit_{name}"
        script = LeakageScript(run_id, typical=published)
        controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
        _interactive(controller, run_config)

        async def user(run_id: str = run_id) -> None:
            await _answer(database, runs_root, run_id, 1, ok=True)

        result = await _run_as_user(controller, dataset, run_id, user)
        if name == "planted":
            _assert_one_covered_comparison(
                result, database, runs_root, run_id, script.trial_id, gate
            )
        else:  # the control: no investigation, no comparison, one approval
            assert result.state == DONE and result.final_trial_id == script.trial_id
            assert _steps(database, run_id) == [("trigger", "not_fired")]
            assert len(list_trials(database, run_id)) == 2 and gate == [1]
            assert len(_events(database, run_id, ApprovalRequestedEvent)) == 1


class _UnknownRangeDiagnosticScript(MultiWithTrials):
    """B cites no typical_score_range, A records expected_score_range=unknown, and C runs
    locked-split feature_predictiveness on `diag_column` before training while keeping it."""

    def __init__(
        self,
        run_id: str,
        *,
        diag_column: str,
        target: str = TARGET,
        task_type: str = "binary_classification",
        primary_metric: str = "roc_auc",
    ) -> None:
        super().__init__(run_id, pipeline_py=LINEAR, final_action="continue")
        self.diag_column = diag_column
        self.diag_target = target
        self.diag_task_type = task_type
        self.primary_metric = primary_metric
        self.diagnostic: dict[str, Any] | None = None

    async def generate(self, request: Any) -> Any:
        schema = request.output_schema_name
        if schema == "IntakeDraft":
            self.requests.append(request)
            return json_result(
                {
                    "objective": f"Predict {self.diag_target} from tabular features",
                    "target": self.diag_target,
                    "task_type": self.diag_task_type,
                    "primary_metric": self.primary_metric,
                }
            )
        if schema == "LeadAction" and "expected_score_range" in request.messages[1].content:
            self.requests.append(request)
            return json_result(
                {
                    "action": "continue",
                    "reason": "Proceed within scope",
                    "assignment": self.assignment,
                    "expected_score_range": {"source": "unknown"},
                }
            )
        return await super().generate(request)

    async def _agent_turn(self, request: Any) -> Any:
        last = request.messages[-1]
        if last.role is MessageRole.TOOL and not last.is_error:
            if last.tool_name == "inspect_dataset" and self.diagnostic is None:
                return self._call(
                    "run_diagnostic",
                    diagnostic="feature_predictiveness",
                    column=self.diag_column,
                    target=self.diag_target,
                    task_type=self.diag_task_type,
                )
            if last.tool_name == "run_diagnostic":
                self.diagnostic = json.loads(last.content)
                return self._call(
                    "write_trial_files",
                    idempotency_key=f"{self.run_id}:write_1",
                    design=DESIGN,
                    pipeline_py=self.pipeline_py,
                )
        return await super()._agent_turn(request)


async def test_unknown_range_with_extreme_locked_split_diagnostic_triggers_covered_comparison(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    gate = _count_gate_decisions(monkeypatch)
    frame = _credit_like(rows=600, seed=1)
    rng = np.random.default_rng(1)
    bad = frame[TARGET] == "bad"
    flip = rng.random(len(frame)) < 0.03
    frame[LEAK] = np.where(
        bad ^ flip, rng.integers(3, 8, len(frame)), rng.integers(0, 2, len(frame))
    )
    dataset = _write(tmp_path, frame, "planted_unknown.csv")
    run_id = "run_leak_unknown_diag_trigger"
    script = _UnknownRangeDiagnosticScript(run_id, diag_column=LEAK)
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    _interactive(controller, run_config)

    async def user() -> None:
        await _answer(database, runs_root, run_id, 1, ok=True)

    result = await _run_as_user(controller, dataset, run_id, user)
    assert result.state == DONE and result.verified, result.notes
    assert script.diagnostic is not None
    assert script.diagnostic["partition_sha256"]
    assert script.diagnostic["metric"] == "roc_auc" and script.diagnostic["score"] >= 0.90
    plan = json.loads((runs_root / run_id / "approved_plan.json").read_text())
    assert plan["expected_score_range"]["source"] == "unknown"
    # Triggered via diagnostic fallback, confirmed, compared with no second prompt.
    assert gate == [1]
    assert _steps(database, run_id) == [
        ("trigger", "fired"),
        ("screen", "candidates"),
        ("confirmation", "confirmed"),
        ("comparison", "submitted"),
        ("comparison", "trained"),
        ("decision", "recommend_without"),
    ]
    leak_events = _events(database, run_id, LeakageCheckEvent)
    trigger_event = leak_events[0].payload
    assert trigger_event.reason is not None
    assert script.diagnostic["diagnostic_id"] in trigger_event.reason
    assert f"{script.diagnostic['score']:.4f}" in trigger_event.reason
    decision = leak_events[-1].payload.section
    assert decision is not None and decision.without_columns is not None
    comparison = decision.without_columns.trial_id
    assert decision.without_columns.without_columns == [LEAK]
    assert result.final_trial_id == decision.recommended_trial_id == comparison
    assert _finals(runs_root, database, run_id) == [comparison]
    contract = load_experiment_contract(database, run_id)
    assert contract.version == 2 and LEAK not in contract.data_contract.allowed_features
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage == decision
    assert UNKNOWN_RANGE_LIMITATION not in report.limitations
    markdown = result.report_md.read_text()
    assert "## A column may be recorded after the outcome" in markdown
    assert "locked-split diagnostic flagged an extreme single-feature score" in markdown
    assert script.diagnostic["diagnostic_id"] in markdown
    assert "well above the expected range" not in markdown
    assert "use the model without it" in markdown
    again = build_report(
        database, run_id, final_trial_id=comparison, notes=report.notes, runs_root=runs_root
    )
    assert again.leakage == report.leakage and again.limitations == report.limitations


async def test_unknown_range_with_modest_locked_split_diagnostic_does_not_trigger(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    gate = _count_gate_decisions(monkeypatch)
    rng = np.random.default_rng(7)
    rows = 400
    x = rng.normal(0.0, 1.0, rows)
    z = rng.normal(0.0, 1.0, rows)
    y = 0.35 * x + 0.2 * z + rng.normal(0.0, 1.0, rows)
    dataset = _write(tmp_path, pd.DataFrame({"x": x, "z": z, "y": y}), "modest_reg.csv")
    run_id = "run_leak_unknown_diag_control"
    script = _UnknownRangeDiagnosticScript(
        run_id,
        diag_column="x",
        target="y",
        task_type="regression",
        primary_metric="r2",
    )
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    _interactive(controller, run_config)

    async def user() -> None:
        await _answer(database, runs_root, run_id, 1, ok=True)

    result = await _run_as_user(controller, dataset, run_id, user)
    assert result.state == DONE and result.verified, result.notes
    assert script.diagnostic is not None
    assert script.diagnostic["partition_sha256"]
    assert script.diagnostic["metric"] == "r2" and script.diagnostic["score"] < 0.85
    # Trigger must NOT fire; no comparison model trained.
    assert _steps(database, run_id) == [("trigger", "unknown_range")]
    assert gate == [1] and len(list_trials(database, run_id)) == 2
    assert len(_events(database, run_id, ApprovalRequestedEvent)) == 1
    assert result.final_trial_id == script.trial_id
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is None
    assert UNKNOWN_RANGE_LIMITATION in report.limitations
