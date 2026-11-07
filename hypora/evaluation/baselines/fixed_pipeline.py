"""The fixed-pipeline baseline (spec §19.3; PLAN.md S2.3): the agent's claim to beat automation.

Random split, every column, default HistGradientBoosting, one trial, no model calls. It is meant to
be naive: it never excludes a leak, never keeps a group together, and never respects time order,
and it takes no input that could tell it otherwise (`SystemInput`, not `Case`). Spec §19.2's trap
cases must catch it (S2.5).

Everything except those choices is the agent's path, so the two are graded on the same evidence: a
subclass of the controller's run, with the same state table, limits, event log, private dataset
copy, MCP server, worker, sandbox, harness, verifier, and report. Only the LLM's steps differ:

    Intake         code: the case's structured constraints
    Contract       code: random split, all columns but the forbidden ones, approved by policy
    Agent          two fixed tool calls over MCP: write the template, submit it
    Reporter       no narrative

docs/notes/baseline.md gives the choices and what the baseline deliberately does not do.
"""

import json
import time
from collections.abc import Mapping
from functools import wraps
from importlib import resources
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd  # type: ignore[import-untyped]
from mcp import Client
from pydantic import JsonValue
from sqlalchemy.exc import SQLAlchemyError

from hypora.core.config.run_config import RunConfig
from hypora.core.controller.contract import FOLDS, METRIC_SETS, SEED, TEST_FRACTION
from hypora.core.controller.contract import build_data_contract as default_data_contract
from hypora.core.controller.controller import (
    MCP_READ_TIMEOUT_S,
    POLL_SECONDS,
    RunController,
    RunRequest,
    RunResult,
    _Run,
    select_final,
)
from hypora.core.controller.local import (
    DATASET_ID,
    WorkerFactory,
    WorkerProcess,
    mcp_transport,
    write_session,
)
from hypora.core.controller.states import DONE, RunEvent
from hypora.core.hashing import sha256_hex
from hypora.core.llm.adapter import LLMAdapter
from hypora.core.roles.base import RunLimitReached
from hypora.core.roles.intake import IntakeError, _metric, _task_type
from hypora.core.roles.single_agent import McpToolChannel
from hypora.core.schemas import (
    ActorRole,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovedPlan,
    Budget,
    CostEstimate,
    DataContract,
    DatasetProfile,
    ErrorClass,
    EventType,
    ExperimentContract,
    Family,
    JobLimits,
    JobState,
    MetricName,
    Operation,
    RequestedAction,
    TaskSpecification,
    TaskType,
    ToolName,
    TrialKind,
)
from hypora.core.schemas.contracts import SplitSpec, SplitStrategy
from hypora.core.schemas.events import ErrorPayload
from hypora.core.schemas.strategy import (
    ProposedMeasurement,
    ProposedPhaseBPlan,
    Recommendation,
    RecommendationKind,
)
from hypora.core.schemas.trials import TrialDesign
from hypora.evaluation.case import (
    ContractBounds,
    DecisionClass,
    FaultKind,
    InjectedFault,
    metrics_for,
)
from hypora.evaluation.system import (
    FinalModel,
    ReportedMetric,
    SystemContext,
    SystemInput,
    SystemRun,
)
from hypora.execution.protocol import experiments as protocol
from hypora.mcp_server.errors import ToolFailure
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.session import load_session
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.migrate import migrate
from hypora.storage.records import (
    load_experiment_contract,
    run_jobs,
    save_data_contract,
    save_experiment_contract,
    save_task_spec,
)

SYSTEM_NAME: Final = "baseline"
TEMPLATE: Final = "gradient_boosting/pipeline.py"
"""The default HistGradientBoosting template, unchanged and with no configuration."""

BASELINE_NOTE: Final = (
    "Fixed-pipeline baseline (spec §19.3): random split, every column but the forbidden ones, "
    "default HistGradientBoosting, one trial. Contract built by code, plan approved by policy."
)
NO_NARRATIVE: Final = "No narrative: the fixed-pipeline baseline makes no model calls."

TOOLS: Final = (ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT)
"""The baseline's whole script. It waits on the job records, as the controller does, not by polling
a tool, so it spends two steps of the run's budget."""

_ESTIMATE: Final = CostEstimate(compute_seconds=60.0, assumptions=("flat estimate",))


# --- The fixed choices ----------------------------------------------------------------------------


def reported_metrics_for(
    task: TaskType, primary: MetricName, named: tuple[MetricName, ...]
) -> tuple[MetricName, ...]:
    """Every metric the platform computes for `task`, except the primary: all of them reach the
    report, so the honesty check has a stated value for whichever one a case grades.

    `brier_score` is left out unless a case names it: the harness refuses it, rather than reporting
    it unavailable, on a fold that misses a class.
    """
    allowed = metrics_for(task) - {MetricName.BRIER_SCORE}
    return tuple(m for m in MetricName if m is not primary and (m in allowed or m in named))


def fixed_task_spec(
    system_input: SystemInput, run_id: str, frame: pd.DataFrame
) -> TaskSpecification:
    """The task exactly as the case's structured constraints state it; no model reads the request.

    The primary metric is the case's, or D-019's default for the task when it names none (`roc_auc`
    for binary, `f1_macro` for multiclass, `mae` for regression), the agent's default too. Raises
    IntakeError for what the data can't support.
    """
    constraints = system_input.constraints
    columns = {str(column) for column in frame.columns}
    if constraints.target not in columns:
        raise IntakeError(f"target column {constraints.target!r} is not in the dataset")
    task = _task_type(frame[constraints.target], constraints.task_type, explicit=True)
    named = constraints.primary_metric
    primary = _metric(task, named, explicit=named is not None)
    return TaskSpecification(
        run_id=run_id,
        version=1,
        objective=system_input.request,
        requested_actions=[RequestedAction.TRAIN],
        dataset_ref=f"dataset:{DATASET_ID}",
        family_hint=Family.TABULAR,
        task_type=task,
        target=constraints.target,
        primary_metric=primary,
        secondary_metrics=[m for m in constraints.secondary_metrics if m is not primary],
        quality_target=constraints.quality_target,
        max_quality_degradation=constraints.max_quality_degradation,
        resource_constraints=constraints.resource_constraints,
        forbidden_columns=sorted(set(constraints.forbidden_columns) & columns),
        exclusion_rules=list(constraints.exclusion_rules),
        named_models=list(constraints.named_models),
        budget=system_input.budget,
        approval_policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        allow_gpu=constraints.allow_gpu,
        from_form=["target", "task_type", "forbidden_columns"],
    )


def fixed_data_contract(spec: TaskSpecification, columns: list[str]) -> DataContract:
    """Every column but the target and the forbidden ones, and a random split whatever the data.

    Forbidden columns are the one exception to "all columns": a contract can't include them. Group
    and time columns are ordinary features here, and rows are shuffled across them on purpose.
    """
    return default_data_contract(spec, columns).model_copy(
        update={
            "split": SplitSpec(
                strategy=SplitStrategy.RANDOM,
                test_fraction=TEST_FRACTION,
                validation=f"kfold_{FOLDS}",
            ),
            "risks": ("Fixed-pipeline baseline: no leakage, group, or time-order handling",),
        }
    )


def fixed_plan(spec: TaskSpecification, data: DataContract, budget: Budget) -> ApprovedPlan:
    """The plan the policy approves: one baseline measurement, within `budget`."""
    return ApprovedPlan(
        run_id=spec.run_id,
        version=1,
        recommendation=Recommendation(
            kind=RecommendationKind.TRAIN_TASK_SPECIFIC,
            rationale="The fixed pipeline always trains its one default model.",
        ),
        phase_b_plan=ProposedPhaseBPlan(
            measurements=(
                ProposedMeasurement(
                    option_id="opt_fixed_pipeline",
                    trial_kind=TrialKind.BASELINE,
                    rationale="The score simple automation reaches.",
                    estimated_cost=_ESTIMATE,
                ),
            ),
            notes=BASELINE_NOTE,
        ),
        data_contract_version=data.data_contract_version,
        data_contract_hash=data.content_hash(),
        budget=budget,
        approval_policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
    )


def limits_within(limits: JobLimits, ceiling: JobLimits) -> JobLimits:
    """The run's job limits, lowered to what the case pre-approved (spec §10.3), without a GPU."""
    threads = limits.cpu_threads
    if threads is not None:  # unset means one thread on both sides (`HyporaTools._check_limits`)
        threads = min(threads, ceiling.cpu_threads or 1)
    return JobLimits(
        max_wall_seconds=min(limits.max_wall_seconds, ceiling.max_wall_seconds),
        max_memory_gb=min(limits.max_memory_gb, ceiling.max_memory_gb),
        gpu=False,
        cpu_threads=threads,
    )


def fixed_contract(
    spec: TaskSpecification,
    data: DataContract,
    approval: ApprovalRecord,
    *,
    budget: Budget,
    job_limits: JobLimits,
    bounds: ContractBounds,
) -> ExperimentContract:
    """The frozen terms of the one trial, tied to the approved plan's hash."""
    if spec.primary_metric is None:
        raise ValueError("fixed_task_spec always sets the primary metric")
    return ExperimentContract(
        run_id=spec.run_id,
        version=1,
        approved_plan_hash=approval.artifact_hash,
        approval_id=approval.approval_id,
        family=Family.TABULAR,
        task_type=data.task_type,
        target=data.target,
        data_contract=data,
        primary_metric=spec.primary_metric,
        secondary_metrics=reported_metrics_for(
            data.task_type, spec.primary_metric, tuple(spec.secondary_metrics)
        ),
        quality_threshold=spec.quality_target,
        max_quality_degradation=spec.max_quality_degradation,
        resource_constraints=spec.resource_constraints,
        budget=budget,
        default_job_limits=limits_within(job_limits, bounds.max_job_limits),
        seed=SEED,
    )


_DESIGN: Final = TrialDesign(
    kind=TrialKind.BASELINE,
    operation=Operation.FIT,
    hypothesis="Default HistGradientBoosting on every column with a random split.",
    model_family="hist_gradient_boosting",
    expected_effect="The score simple automation reaches, with no methodology judgment.",
    estimated_cost=_ESTIMATE,
)


def _template() -> str:
    return (
        resources.files("hypora.execution.templates.tabular")
        .joinpath(TEMPLATE)
        .read_text(encoding="utf-8")
    )


def _sort_faults(
    faults: tuple[InjectedFault, ...],
) -> tuple[dict[ToolName, InjectedFault], list[str]]:
    """Which injected faults hit this script's tool calls, and notes on the ones that can't.

    A fixed script calls each tool once, so a `tool_error` hits it only as that tool's first call.
    Malformed data is already in the dataset. No platform code injects the other kinds yet (OOM,
    timeouts, crashes), and a run must not look like it survived a fault it never met.
    """
    hits: dict[ToolName, InjectedFault] = {}
    notes = []
    for fault in faults:
        if fault.kind is FaultKind.MALFORMED_DATA:
            continue
        if fault.kind is not FaultKind.TOOL_ERROR:
            notes.append(f"Injected fault {fault.kind.value} was not applied to this run.")
        elif fault.tool in TOOLS and fault.occurrence == 1:
            assert fault.tool is not None
            hits[fault.tool] = fault
        else:
            notes.append(
                f"Injected tool error on {fault.tool} (call {fault.occurrence}) was never reached: "
                "the fixed pipeline does not make that call."
            )
    return hits, notes


def _error_code(content: str) -> str:
    """A failed tool's error code. Only the code reaches the notes, never its message."""
    try:
        code = json.loads(content).get("code")
    except (ValueError, AttributeError):
        code = None
    return str(code) if code else "unknown"


# --- The run --------------------------------------------------------------------------------------


def _refusal(system_input: SystemInput) -> str | None:
    """Why the baseline can't attempt this case: it has a tabular pipeline and nothing else."""
    if system_input.family is not Family.TABULAR:
        return (
            "The fixed-pipeline baseline runs tabular cases only; this case is "
            f"{system_input.family.value}."
        )
    task = system_input.constraints.task_type
    if task not in METRIC_SETS:
        return f"The fixed-pipeline baseline does not support the task type {task.value}."
    return None


class _BaselineRun(_Run):
    """The controller's run, with code in place of each LLM role."""

    def __init__(
        self,
        controller: RunController,
        request: RunRequest,
        system_input: SystemInput,
        faults: tuple[InjectedFault, ...],
    ) -> None:
        super().__init__(controller, request)
        self.system_input = system_input
        self.family = system_input.family
        self.contract: ExperimentContract | None = None
        self.fault_hits, fault_notes = _sort_faults(faults)
        self.notes += fault_notes

    async def execute(self) -> RunResult:
        """The controller's run; a refused case still leaves a run, its events, and a report, so
        the graders find evidence of the refusal rather than an empty log."""
        refusal = _refusal(self.system_input)
        if refusal is None:
            return await super().execute()
        self._create()
        self._failed(IntakeError(refusal))
        self._advance(RunEvent.FAIL, "refused: the fixed pipeline can't run this case")
        await self._report()
        self._advance(RunEvent.REPORTED_UNVERIFIED, "no verified model")
        return RunResult(
            run_id=self.run_id,
            state=self.state,
            final_trial_id=None,
            verified=False,
            report_json=self.run_dir / "report.json",
            report_md=self.run_dir / "report.md",
            notes=tuple(self.notes),
            limits_fired=(),
        )

    async def _intake(self, frame: pd.DataFrame) -> TaskSpecification:
        spec = fixed_task_spec(self.system_input, self.run_id, frame)
        save_task_spec(self.database, spec)
        ignored = [
            f"{count} {what}"
            for count, what in (
                (len(spec.exclusion_rules), "plain-language exclusion rule(s)"),
                (len(spec.named_models), "named model(s)"),
            )
            if count
        ]
        if ignored:
            self.notes.append(f"Not applied by the fixed pipeline: {', '.join(ignored)}.")
        return spec

    def _contract(self, spec: TaskSpecification, columns: list[str]) -> ExperimentContract:
        data = fixed_data_contract(spec, columns)
        save_data_contract(self.database, self.run_id, data)
        plan = fixed_plan(spec, data, self.budget)
        (self.run_dir / "approved_plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        contract = fixed_contract(
            spec,
            data,
            self._approve(plan, BASELINE_NOTE),
            budget=self.budget,
            job_limits=self.controller.run_config.budgets.job_limits,
            bounds=self.system_input.contract_bounds,
        )
        save_experiment_contract(self.database, contract)
        self.contract = contract
        self.notes.append(BASELINE_NOTE)
        return contract

    async def _experiments(
        self, spec: TaskSpecification, contract: ExperimentContract, profile: DatasetProfile
    ) -> None:
        missing = [tool.value for tool in TOOLS if tool not in self.system_input.permitted_tools]
        if missing:
            self.notes.append(f"No trial: the case does not permit {', '.join(missing)}.")
            return
        trial_id = await self._write_and_submit()
        if trial_id is None:
            return
        self.notes += await self._drain(cancel=False)
        self.notes += self._failed_jobs()
        selection = select_final(self.database, self.run_id, contract.primary_metric, None)
        if selection.trial_id is None or selection.source_job_id is None:
            self.notes.append(f"No final model: {selection.reason}.")
            return
        self.notes.append(f"Final candidate {selection.trial_id}: {selection.reason}.")
        await self._final(selection.trial_id, selection.source_job_id, contract)

    async def _write_and_submit(self) -> str | None:
        """The script, over the same MCP server the agent uses; the new trial, or None if it failed.

        Nothing retries: a fixed pipeline has no recovery, which is what a fault case measures.
        """
        c = self.controller
        session = write_session(
            self.run_dir / "mcp" / "session.json",
            run_id=self.run_id,
            # The only role with an allowlist; its absence of model_call events tells them apart.
            role=ActorRole.SINGLE_AGENT,
            database=c.database_path,
            runs_root=c.runs_root,
            dataset=self.dataset,
            flags=c.run_config.flags,
        )
        log = self.run_dir / "logs" / "mcp-server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as errlog:
            transport = mcp_transport(session, errlog)
            async with Client(transport, read_timeout_seconds=MCP_READ_TIMEOUT_S) as client:
                channel = McpToolChannel(client)
                written = await self._tool(
                    channel,
                    ToolName.WRITE_TRIAL_FILES,
                    {
                        "idempotency_key": f"{self.run_id}:write_fixed_pipeline",
                        "design": _DESIGN.model_dump(mode="json"),
                        "pipeline_py": _template(),
                    },
                )
                if written is None:
                    return None
                trial_id = str(written["trial_id"])
                submitted = await self._tool(
                    channel,
                    ToolName.SUBMIT_EXPERIMENT,
                    {
                        "trial_id": trial_id,
                        "idempotency_key": f"{self.run_id}:submit_fixed_pipeline",
                    },
                )
        return None if submitted is None else trial_id

    async def _tool(
        self, channel: McpToolChannel, tool: ToolName, arguments: Mapping[str, JsonValue]
    ) -> dict[str, Any] | None:
        """One call, under the run's guard. A refused or failed call is a note and None.

        A limit is handled here, as the agent's loop does: raised through the MCP client's task
        group it would arrive wrapped, and look like a crash of the run.
        """
        try:
            self.guard.before_tool_call(tool.value, arguments)
        except RunLimitReached as reached:
            self.notes.append(f"Run stopped: {reached}.")
            return None
        fault = self.fault_hits.get(tool)
        if fault is not None and fault.error_code is not None:
            await self._injected_error(tool, arguments, fault.error_code)
            self.guard.after_tool_call(tool.value)
            self._event(
                EventType.ERROR,
                ErrorPayload(
                    component="fault_injection",
                    code=fault.error_code,
                    message=f"injected tool error on {tool.value}",
                ),
            )
            self.notes.append(f"{tool.value} failed with injected {fault.error_code}; no retry.")
            return None
        is_error, content = await channel.call(tool.value, arguments)
        self.guard.after_tool_call(tool.value)
        if is_error:
            self.notes.append(f"{tool.value} failed ({_error_code(content)}); no retry.")
            return None
        result: dict[str, Any] = json.loads(content)
        return result

    async def _injected_error(
        self, tool: ToolName, arguments: Mapping[str, JsonValue], code: str
    ) -> None:
        """Fail the tool body through the real MCP boundary, retaining its policy and result events.

        The fault is local to this session and call. The original signature and description stay
        intact, so argument checks, role allowlist, result firewall, and event recording still run.
        """
        session = load_session(self.run_dir / "mcp" / "session.json")
        try:
            tools = HyporaTools(session)

            @wraps(getattr(tools, tool.value))
            def fail(**_: Any) -> Any:
                raise ToolFailure(
                    code,
                    ErrorClass.TRANSIENT,
                    f"Injected tool error on {tool.value}.",
                    retryable=False,
                )

            setattr(tools, tool.value, fail)
            await HyporaMCPServer(session, tools=tools).call_tool(tool.value, dict(arguments))
        finally:
            session.close()

    def _failed_jobs(self) -> list[str]:
        """Why a job did not finish, as the worker recorded it; a fixed pipeline does not retry."""
        notes = []
        for job in run_jobs(self.database, self.run_id):
            if job.status is not JobState.COMPLETED:
                folder = protocol.job_dir(self.controller.runs_root, self.run_id, job.job_id)
                outcome = protocol.read_outcome(folder)
                code = outcome.error_code if outcome else None
                notes.append(f"Job {job.job_id} ended {job.status.value} ({code or 'no outcome'}).")
        return notes

    async def _report(self) -> bool:
        report = self._write_facts(NO_NARRATIVE)
        return report is not None and report.outcome == "verified_model"


# --- What the run leaves behind -------------------------------------------------------------------


def final_model_from_report(database: Database, runs_root: Path, run_id: str) -> FinalModel | None:
    """The report's final model with its evidence, or None if the report names none.

    The reported metrics are the numbers the report states; a metric the harness left undefined has
    no value to report. The model and pipeline are the final job's, each re-hashed against what the
    job recorded, so a grader never starts from a file that changed. Raises LookupError, ValueError,
    or OSError on missing or changed evidence.
    """
    report = json.loads((runs_root / run_id / "report.json").read_bytes())
    final = report.get("final")
    if final is None:
        return None
    trial_id = final["trial_id"]
    for job in reversed(run_jobs(database, run_id)):
        folder = protocol.job_dir(runs_root, run_id, job.job_id)
        request = protocol.read_request(folder)
        if job.trial_id == trial_id and request.phase is protocol.ExperimentPhase.FINAL:
            break
    else:
        raise LookupError(f"trial {trial_id} has no final job")
    outcome = protocol.read_outcome(folder)
    if job.status is not JobState.COMPLETED or outcome is None:
        raise LookupError(f"the final job {job.job_id} did not complete")
    model_id = next((a.artifact_id for a in outcome.artifacts if a.kind == "model"), None)
    if model_id is None:
        raise LookupError(f"the final job {job.job_id} stored no model")
    model = ArtifactStore(database, runs_root / run_id / "artifacts").get(model_id)  # re-hashes it
    pipeline = folder / protocol.TRIAL_SNAPSHOT_DIR / "pipeline.py"
    pipeline_sha256 = sha256_hex(pipeline.read_bytes())
    if pipeline_sha256 != request.trial_files["pipeline.py"]:
        raise ValueError("pipeline.py changed after its job was submitted")
    return FinalModel(
        trial_id=trial_id,
        contract=load_experiment_contract(database, run_id),
        model_path=model.path,
        model_sha256=model.sha256,
        pipeline_path=pipeline,
        pipeline_sha256=pipeline_sha256,
        reported_metrics=tuple(
            ReportedMetric(name=MetricName(name), value=value)
            for name, value in final["test_metrics"].items()
            if value is not None
        ),
    )


class FixedPipelineBaseline:
    """Spec §19.3's fixed pipeline as an `EvalSystem`. Failures return a `SystemRun`."""

    def __init__(
        self, *, worker_factory: WorkerFactory = WorkerProcess, poll_seconds: float = POLL_SECONDS
    ) -> None:
        self.worker_factory = worker_factory
        self.poll_seconds = poll_seconds

    @property
    def name(self) -> str:
        """The system's name in result paths."""
        return SYSTEM_NAME

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        """Run one case on its visible dataset; `completed` means a verified final model."""
        started = time.monotonic()
        completed = False
        decision: DecisionClass | None = None
        final: FinalModel | None = None
        limits: tuple[str, ...] = ()
        notes: list[str] = []
        database: Database | None = None
        try:
            context.runs_root.mkdir(parents=True, exist_ok=True)
            context.database.parent.mkdir(parents=True, exist_ok=True)
            database = Database(context.database)
            migrate(database)
            controller = _model_free_controller(
                database,
                context.runs_root,
                context.run_config,
                self.worker_factory,
                self.poll_seconds,
            )
            run = _BaselineRun(
                controller,
                RunRequest(
                    dataset=system_input.dataset,
                    goal=system_input.request,
                    budget=system_input.budget,
                    run_id=context.run_id,
                ),
                system_input,
                context.injected_faults,
            )
            result = await run.execute()
            notes, limits = list(result.notes), result.limits_fired
            decision = None if run.contract is None else DecisionClass.TRAIN
            try:
                final = final_model_from_report(database, controller.runs_root, result.run_id)
            except (LookupError, ValueError, OSError) as exc:
                notes.append(f"The final model's evidence can't be used: {type(exc).__name__}.")
            completed = result.state == DONE and result.verified and final is not None
        except (OSError, SQLAlchemyError) as exc:  # the platform itself: nothing can be recorded
            notes.append(f"The baseline could not run: {type(exc).__name__}.")
        finally:
            if database is not None:
                database.close()
        return SystemRun(
            case_id=system_input.case_id,
            system=SYSTEM_NAME,
            run_id=context.run_id,
            runs_root=context.runs_root,
            database=context.database,
            completed=completed,
            wall_clock_seconds=time.monotonic() - started,
            decision_class=decision,
            final_model=final,
            limits_fired=limits,
            notes=tuple(notes),
        )


def _model_free_controller(
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    worker_factory: WorkerFactory,
    poll_seconds: float,
) -> RunController:
    """A controller with no adapter at all: `_BaselineRun` replaces every step that uses one, and a
    model call could only crash, so "no LLM" holds by construction. The agent's adapter also refuses
    a profile with unset model IDs, which must not stop a run that uses no model."""
    return RunController(
        database=database,
        runs_root=runs_root,
        adapter=cast(LLMAdapter, None),
        run_config=run_config,
        worker_factory=worker_factory,
        poll_seconds=poll_seconds,
    )


__all__ = [
    "BASELINE_NOTE",
    "SYSTEM_NAME",
    "FixedPipelineBaseline",
    "final_model_from_report",
    "fixed_contract",
    "fixed_data_contract",
    "fixed_plan",
    "fixed_task_spec",
    "limits_within",
    "reported_metrics_for",
]
