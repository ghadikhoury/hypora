"""The `agent` system: the `hypora run` path as an `EvalSystem` (spec §19.2, §10.3; PLAN.md S2.4).

It hands the controller the case's request, dataset, and budget, and reads back what the run
stored. Nothing about the run is assumed: the final model and every metric come from the database,
the artifact store, and `report.json`.

What the agent never receives: `SystemInput.case_id` (a case's ID may name its trap) and the faults
in `SystemContext`, which the channel below applies from outside the agent's view.

Refused rather than run, because the result would look like something it isn't (`UnsupportedCase`
becomes a `harness_error` record):
- a family other than tabular: the controller only runs tabular;
- a fault this path can't inject (only `tool_error`, and `malformed_data`, which is in the data);
The trusted session narrows role permissions to the case's permitted tools, except when the
allowlist defense is deliberately disabled by the comparison config.
"""

import hashlib
import json
import secrets
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import anyio
from pydantic import JsonValue

from hypora.core.config.run_config import RunConfig
from hypora.core.config.settings import BudgetsConfig
from hypora.core.controller.controller import (
    MAX_AGENT_TURNS,
    POLL_SECONDS,
    RunController,
    RunRequest,
    RunResult,
)
from hypora.core.controller.local import WorkerFactory, WorkerProcess
from hypora.core.controller.states import DONE
from hypora.core.llm import LLMAdapter, ProviderAdmission
from hypora.core.llm.sinks import FileModelIOStore
from hypora.core.policy.allowlists import TOOL_TIERS, allowed_tools
from hypora.core.roles.reporter import EarlyStopReport, Report
from hypora.core.roles.single_agent import ToolChannel
from hypora.core.schemas import (
    ActorRole,
    ApprovalPolicy,
    ErrorClass,
    EventType,
    Family,
    JobLimits,
    JobState,
    MetricName,
    ToolError,
    TrialKind,
)
from hypora.core.schemas.events import (
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.evaluation.case import CaseConstraints, ContractBounds, DecisionClass, FaultKind
from hypora.evaluation.case import InjectedFault as Fault
from hypora.evaluation.system import (
    FinalModel,
    ReportedMetric,
    SystemContext,
    SystemInput,
    SystemRun,
)
from hypora.execution.protocol import experiments as protocol
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import (
    list_trials,
    load_experiment_contract,
    run_jobs,
)

AGENT_SYSTEM: Final = "agent"
AGENT_ROLE: Final = ActorRole.SINGLE_AGENT
INJECTABLE_FAULTS: Final = frozenset({FaultKind.TOOL_ERROR, FaultKind.MALFORMED_DATA})
"""Tool errors fire here; malformed data is planted in the dataset. S3.H3 adds the rest."""
UNSUPPORTED_HARD_CONSTRAINTS: Final = (
    "max_quality_degradation",
    "exclusion_rules",
    "named_models",
    "resource_constraints",
)
"""These need later optimization/data/model-selection behavior; refuse instead of omitting them."""

AdapterFactory = Callable[[RunConfig, Database, Path, str], LLMAdapter]
"""Builds a run's adapter from (config, database, runs root, run ID); tests script a provider."""


class UnsupportedCase(Exception):
    """This system can't honestly run this case; the message says what is missing."""


class AgentRunEvidenceError(Exception):
    """A run's stored records don't support what its report says."""


def bounded_run_config(run_config: RunConfig, bounds: ContractBounds) -> RunConfig:
    """The run's config with job limits inside the pre-approved contract (spec §10.3).

    `auto_within_contract` may approve only inside the bounds, so a default above them is lowered
    here, before the controller writes it into the experiment contract.
    """
    wanted, ceiling = run_config.budgets.job_limits, bounds.max_job_limits
    threads = [t for t in (wanted.cpu_threads, ceiling.cpu_threads) if t is not None]
    limits = JobLimits(
        max_wall_seconds=min(wanted.max_wall_seconds, ceiling.max_wall_seconds),
        max_memory_gb=min(wanted.max_memory_gb, ceiling.max_memory_gb),
        gpu=wanted.gpu and ceiling.gpu,
        cpu_threads=min(threads) if threads else None,
    )
    budgets = BudgetsConfig(run=run_config.budgets.run, job_limits=limits)
    return run_config.model_copy(update={"budgets": budgets})


def check_supported(system_input: SystemInput, context: SystemContext) -> None:
    """Raise UnsupportedCase for a case this path can't run as written (module docstring)."""
    if system_input.family is not Family.TABULAR:
        raise UnsupportedCase(
            f"the controller runs tabular cases only, not {system_input.family.value}"
        )
    unsupported = uncarried_constraints(system_input.constraints)
    if unsupported:
        raise UnsupportedCase(
            "Hard constraints are not enforced by this controller yet: " + "; ".join(unsupported)
        )
    faults = {fault.kind for fault in context.injected_faults} - INJECTABLE_FAULTS
    if faults:
        raise UnsupportedCase(
            f"faults {sorted(kind.value for kind in faults)} can't be injected yet (S3.H3); "
            "running without them would grade a fault that never happened"
        )
    if context.run_config.flags.approval_policy is not ApprovalPolicy.AUTO_WITHIN_CONTRACT:
        raise UnsupportedCase("eval runs require auto_within_contract (spec 10.3)")


def uncarried_constraints(constraints: CaseConstraints) -> list[str]:
    """Notes naming each set constraint that couldn't reach this run."""
    defaults = CaseConstraints(target=constraints.target, task_type=constraints.task_type)
    unset = [
        name
        for name in UNSUPPORTED_HARD_CONSTRAINTS
        if getattr(constraints, name) != getattr(defaults, name)
    ]
    return [f"Constraint {name} requires a later implementation." for name in unset]


class FaultLedger:
    """One run's tool-call counts and fired faults, shared by every role session's channel.

    A case's `occurrence` counts the run's calls of the tool, whichever role makes them (the
    grader counts the run's `tool_result` events the same way), so a fault fires once per run,
    not once per session.
    """

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.fired: list[Fault] = []


class FaultInjectingChannel:
    """A role session's tool channel with the case's `tool_error` faults applied (spec §19.2).

    The chosen call never reaches the MCP server, so it has no effect. Its events are written here,
    as the server writes them, under the session's role: the log shows what really happened, and
    the call's permission is checked against that role's allowlist. A call the allowlist doesn't
    cover passes through, so the server refuses it as usual.
    """

    def __init__(
        self,
        inner: ToolChannel,
        *,
        run_id: str,
        events: EventLog,
        faults: tuple[Fault, ...],
        permitted_tools: frozenset[str] | None = None,
        allowlist_enabled: bool = True,
        role: ActorRole = AGENT_ROLE,
        ledger: FaultLedger | None = None,
    ) -> None:
        self.inner = inner
        self.permitted_tools = permitted_tools
        self.allowlist_enabled = allowlist_enabled
        self.run_id = run_id
        self.events = events
        self.role = role
        self.faults = tuple(fault for fault in faults if fault.kind is FaultKind.TOOL_ERROR)
        self.ledger = ledger or FaultLedger()

    @property
    def calls(self) -> Counter[str]:
        """The run's calls per tool so far (shared across sessions)."""
        return self.ledger.calls

    @property
    def fired(self) -> list[Fault]:
        """The faults that have fired in this run (shared across sessions)."""
        return self.ledger.fired

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        """Run one tool call, or fail it if a fault is set for this call of this tool."""
        self.calls[name] += 1
        allowed = (
            {tool.value for tool in allowed_tools(self.role)}
            if self.allowlist_enabled
            else {tool.value for tool in TOOL_TIERS}
        )
        if self.allowlist_enabled and self.permitted_tools is not None:
            allowed &= self.permitted_tools
        for fault in self.faults:
            if (
                fault.tool == name
                and fault.occurrence == self.calls[name]
                and fault not in self.fired
                and name in allowed
            ):
                self.fired.append(fault)
                error = await anyio.to_thread.run_sync(self._record, fault, name)
                return True, json.dumps(error.model_dump(mode="json"))
        return await self.inner.call(name, arguments)

    def _record(self, fault: Fault, name: str) -> ToolError:
        assert fault.error_code is not None  # InjectedFault requires it for a tool_error
        started = time.monotonic()
        call_id = f"call_{secrets.token_hex(8)}"
        error = ToolError(
            code=fault.error_code,
            error_class=ErrorClass.TRANSIENT,
            message=f"{name} failed before it could run.",
            retryable=True,
            suggested_recovery="Try the call again.",
        )
        self._append(
            EventType.TOOL_CALL,
            self.role,
            ToolCallPayload(call_id=call_id, tool=name, session_id=f"sess_{self.run_id}_harness"),
        )
        self._append(
            EventType.POLICY_DECISION,
            ActorRole.POLICY_ENGINE,
            PolicyDecisionPayload(
                call_id=call_id,
                tool=name,
                decision=PolicyOutcome.ALLOW,
                decided_by=PolicyRule.ROLE_ALLOWLIST if self.allowlist_enabled else None,
                reasons=[
                    f"role={self.role.value} and case allow {name}"
                    if self.allowlist_enabled
                    else "defenses.allowlist is off"
                ],
            ),
        )
        self._append(
            EventType.TOOL_RESULT,
            self.role,
            ToolResultPayload(
                call_id=call_id,
                tool=name,
                status=ToolResultStatus.ERROR,
                result_summary={"code": error.code},
                error=error,
                latency_ms=(time.monotonic() - started) * 1000,
            ),
        )
        return error

    def _append(self, kind: EventType, actor: ActorRole, payload: object) -> None:
        self.events.append(self.run_id, kind, actor, payload)

    def fault_notes(self) -> list[str]:
        """What happened to each fault. One that never fired must not look like one survived."""
        notes = []
        for fault in self.faults:
            target = f"{fault.tool} call {fault.occurrence}"
            if fault in self.fired:
                notes.append(f"Injected fault {fault.error_code} fired on {target}.")
            else:
                made = self.calls[str(fault.tool)]
                notes.append(
                    f"Injected fault {fault.error_code} on {target} did NOT fire: the agent made "
                    f"{made} such call(s)."
                )
        return notes


def _decision_class(database: Database, run_id: str, trial_id: str) -> DecisionClass:
    """The decision the final trial's kind records: reusing a model as it is, or training one."""
    kinds = {trial.trial_id: trial.kind for trial in list_trials(database, run_id)}
    if trial_id not in kinds:
        raise AgentRunEvidenceError(f"final trial {trial_id} has no trials row")
    return DecisionClass.REUSE if kinds[trial_id] == TrialKind.EVAL_AS_IS else DecisionClass.TRAIN


def planned_decision(report_json: Path) -> DecisionClass | None:
    """What a run without a verified model set out to do: the report's evidence (D-032). It stays a
    claim the grader recomputes. A run with no plan or no readable report states no decision."""
    try:
        content = report_json.read_bytes()
        try:
            report: Report | EarlyStopReport = Report.model_validate_json(content)
        except ValueError:
            report = EarlyStopReport.model_validate_json(content)
    except (OSError, ValueError):
        return None
    decision = report.evidence.decision if report.evidence else None
    return None if decision is None else DecisionClass(decision.intended)


def _final_job(database: Database, runs_root: Path, run_id: str, trial_id: str) -> str:
    """The completed FINAL job of the trial: the job whose model the report describes."""
    for job in reversed(run_jobs(database, run_id)):
        if job.trial_id != trial_id or job.status is not JobState.COMPLETED:
            continue
        request = protocol.read_request(protocol.job_dir(runs_root, run_id, job.job_id))
        if request.phase is protocol.ExperimentPhase.FINAL:
            return job.job_id
    raise AgentRunEvidenceError(f"final trial {trial_id} has no completed final job")


def final_model_from_records(
    database: Database, runs_root: Path, run_id: str, report_json: Path
) -> FinalModel:
    """The verified model of a run, from stored records, hash-checked where they are read."""
    report = Report.model_validate_json(report_json.read_bytes())
    if report.final is None:
        raise AgentRunEvidenceError("a verified run's report names no final model")
    trial_id = report.final.trial_id
    job_id = _final_job(database, runs_root, run_id, trial_id)
    folder = protocol.job_dir(runs_root, run_id, job_id)
    model = ArtifactStore(database, runs_root / run_id / "artifacts").get(f"{job_id}.model")
    pipeline_path = folder / protocol.TRIAL_SNAPSHOT_DIR / "pipeline.py"
    pipeline_sha256 = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
    if pipeline_sha256 != protocol.read_request(folder).trial_files.get("pipeline.py"):
        raise AgentRunEvidenceError(f"final job {job_id}: pipeline.py changed after the run")
    return FinalModel(
        trial_id=trial_id,
        contract=load_experiment_contract(database, run_id),
        model_path=model.path,
        model_sha256=model.sha256,
        pipeline_path=pipeline_path,
        pipeline_sha256=pipeline_sha256,
        reported_metrics=tuple(
            ReportedMetric(name=MetricName(name), value=value)
            for name, value in report.final.test_metrics.items()
            if value is not None
        ),
    )


class AgentSystem:
    """Runs a case through the controller, exactly as `hypora run` runs a request."""

    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory | None = None,
        worker_factory: WorkerFactory = WorkerProcess,
        poll_seconds: float = POLL_SECONDS,
        max_agent_turns: int = MAX_AGENT_TURNS,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._worker_factory = worker_factory
        self._poll_seconds = poll_seconds
        self._max_agent_turns = max_agent_turns
        self._admission: ProviderAdmission | None = None
        self._retry_waits: dict[str, float] = {}

    @property
    def name(self) -> str:
        """The system's name in result paths."""
        return AGENT_SYSTEM

    def retry_wait_seconds(self, run_id: str) -> float | None:
        """Seconds a finished run spent in retry backoff (`SystemRun` has no place for them)."""
        return self._retry_waits.get(run_id)

    def _adapter(
        self, run_config: RunConfig, database: Database, runs_root: Path, run_id: str
    ) -> LLMAdapter:
        if self._adapter_factory is not None:
            return self._adapter_factory(run_config, database, runs_root, run_id)
        # One admission for every run of this eval: parallel runs share the rate limits, the daily
        # cap, and the concurrency permits, and no new run resets them.
        if self._admission is None:
            self._admission = ProviderAdmission(
                run_config.models.rate_limits, run_config.pools.llm_concurrency
            )
        return LLMAdapter.from_run_config(
            run_config,
            events=EventLog(database),
            admission=self._admission,
            io_store=FileModelIOStore(runs_root),
        )

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        """Run one case's request on its visible dataset and report what the records show."""
        check_supported(system_input, context)
        constraints = system_input.constraints
        run_config = bounded_run_config(context.run_config, system_input.contract_bounds)
        request = RunRequest(
            dataset=system_input.dataset,
            goal=system_input.request
            + "\n\nStructured user constraints:\n"
            + constraints.model_dump_json(),
            visible_constraints=constraints.model_dump_json(),
            secondary_metrics=constraints.secondary_metrics,
            quality_target=constraints.quality_target,
            target=constraints.target,
            task_type=constraints.task_type,
            primary_metric=constraints.primary_metric,
            forbidden_columns=constraints.forbidden_columns,
            budget=system_input.budget,
            run_id=context.run_id,
            permitted_tools=system_input.permitted_tools,
        )
        database = Database(context.database)
        channels: list[FaultInjectingChannel] = []
        ledger = FaultLedger()

        def wrap(inner: ToolChannel, role: ActorRole) -> ToolChannel:
            channel = FaultInjectingChannel(
                inner,
                run_id=context.run_id,
                events=EventLog(database),
                faults=context.injected_faults,
                permitted_tools=frozenset(tool.value for tool in system_input.permitted_tools),
                allowlist_enabled=run_config.flags.defenses.allowlist,
                role=role,
                ledger=ledger,
            )
            channels.append(channel)
            return channel

        adapter: LLMAdapter | None = None
        try:
            migrate(database)
            adapter = self._adapter(run_config, database, context.runs_root, context.run_id)
            controller = RunController(
                database=database,
                runs_root=context.runs_root,
                adapter=adapter,
                run_config=run_config,
                worker_factory=self._worker_factory,
                poll_seconds=self._poll_seconds,
                max_agent_turns=self._max_agent_turns,
                tool_channel_wrapper=wrap,
            )
            started = time.perf_counter()
            result = await controller.run(request)
            wall_clock = time.perf_counter() - started
            return self._system_run(system_input, context, database, result, wall_clock, channels)
        finally:
            if adapter is not None:
                self._retry_waits[context.run_id] = adapter.retry_wait_s
            database.close()

    def _system_run(
        self,
        system_input: SystemInput,
        context: SystemContext,
        database: Database,
        result: RunResult,
        wall_clock: float,
        channels: list[FaultInjectingChannel],
    ) -> SystemRun:
        final_model = None
        decision = None
        notes = list(uncarried_constraints(system_input.constraints))
        if result.verified:
            final_model = final_model_from_records(
                database, context.runs_root, context.run_id, result.report_json
            )
            decision = _decision_class(database, context.run_id, final_model.trial_id)
        else:
            decision = planned_decision(result.report_json)
        notes += result.notes
        if channels:
            notes += channels[-1].fault_notes()
        elif context.injected_faults:  # the run stopped before the agent had a tool channel
            notes.append("Injected faults did NOT fire: the agent never started.")
        return SystemRun(
            case_id=system_input.case_id,
            system=self.name,
            run_id=context.run_id,
            runs_root=context.runs_root,
            database=context.database,
            completed=result.state == DONE,
            wall_clock_seconds=wall_clock,
            decision_class=decision,
            final_model=final_model,
            limits_fired=result.limits_fired,
            notes=tuple(notes),
        )


__all__ = [
    "AGENT_SYSTEM",
    "AdapterFactory",
    "AgentRunEvidenceError",
    "AgentSystem",
    "FaultInjectingChannel",
    "FaultLedger",
    "UnsupportedCase",
    "bounded_run_config",
    "check_supported",
    "final_model_from_records",
    "planned_decision",
    "uncarried_constraints",
]
