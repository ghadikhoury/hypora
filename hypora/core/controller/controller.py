"""Controller v1 (spec §5.1, §6.8; PLAN.md S1.5; D-019): one run, from the goal to the report.

The controller is deterministic code. It owns the run's state (the transition table in
`states.py`), its limits (`limits.RunGuard`, checked before every model and tool call), the data
contract and approval (`contract.py`), the choice of the final trial, and the one test-split
evaluation. LLM roles only draft (Intake), experiment (the single agent, over MCP stdio), and
narrate (Reporter).

    INTAKE    Intake drafts the TaskSpecification; code settles every field it can check.
    PROFILE   the dataset profile, from the exact bytes the experiments will read.
    CONTRACT  data contract, plan, the plan hash approved by policy, experiment contract.
    PHASE_B   the experiment worker starts; the single agent runs trials through the MCP
              server; the controller drains unfinished jobs, selects the final trial, and queues
              its `final` job (re-validate, refit, score the test split once, verifier v1).
    REPORT    report.json and report.md from stored records, plus the Reporter's narrative.

A failure at any stage is an `error` event and a `fail` transition into REPORT, so every run
ends with a report that says what happened.
"""

import contextlib
import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Final

import anyio
import pandas as pd  # type: ignore[import-untyped]
from mcp import Client
from pydantic import BaseModel, TypeAdapter

from hypora.core.config.flags import AgentMode, FeatureFlags, Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.contract import (
    ContractError,
    approve_by_policy,
    build_experiment_contract,
    build_plan,
)
from hypora.core.controller.limits import RunGuard
from hypora.core.controller.local import (
    DATASET_ID,
    WorkerFactory,
    WorkerProcess,
    mcp_transport,
    write_session,
)
from hypora.core.controller.methodology import validate_data_proposal
from hypora.core.controller.states import START, RunEvent, RunState, next_state
from hypora.core.llm.adapter import LLMAdapter
from hypora.core.policy.firewall import DataFirewall
from hypora.core.roles.base import USER_CANCEL, RunLimitReached
from hypora.core.roles.intake import IntakeError, IntakeForm, _task_type, run_intake
from hypora.core.roles.methodology import propose_data_contract
from hypora.core.roles.multi import LeadAction, ResearchObservation, draft_plan, main_role_call
from hypora.core.roles.outcome_evidence import derive_outcome_evidence, has_authorized_model_route
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.roles.reporter import (
    EarlyStopReport,
    Report,
    build_report,
    metric_summary,
    render_early_stop,
    render_markdown,
    write_narrative,
)
from hypora.core.roles.research_agent import run_research_agent
from hypora.core.roles.single_agent import (
    AgentOutcome,
    McpToolChannel,
    ToolChannel,
    run_single_agent,
)
from hypora.core.schemas import (
    AUTO_APPROVER,
    ActorRole,
    ApprovalArtifactType,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovedPlan,
    Budget,
    CheckStatus,
    DataContract,
    DatasetProfile,
    ErrorClass,
    EventType,
    ExperimentContract,
    Family,
    Handoff,
    HandoffStatus,
    Identifier,
    JobState,
    MetricDirection,
    MetricName,
    ModelTier,
    Operation,
    Plan,
    PlanStep,
    SubmitExperimentRequest,
    TaskSpecification,
    TaskType,
    ToolName,
)
from hypora.core.schemas.events import (
    ApprovalGrantedPayload,
    ApprovalRequestedPayload,
    ErrorPayload,
    HandoffPayload,
    PlanPayload,
    RoleStartedPayload,
    StageTransitionPayload,
)
from hypora.execution.harness.runner import load_tabular
from hypora.execution.protocol import experiments as protocol
from hypora.mcp_server.server import _redact
from hypora.profiling.tabular import leakage_screen_summary, profile_tabular
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import (
    TERMINAL_STATES,
    InvalidJobTransition,
    JobQueue,
    StaleJobState,
    new_job_id,
)
from hypora.storage.records import (
    RecordNotFound,
    create_run,
    list_trials,
    load_result,
    run_jobs,
    save_approval,
    save_data_contract,
    save_dataset_profile,
    save_experiment_contract,
    save_task_spec,
    set_run_status,
)

MAX_AGENT_TURNS: Final = 60
"""The single agent's own cap on model turns (a role limit, recorded like a run limit)."""
POLL_SECONDS: Final = 1.0
DRAIN_TIMEOUT_S: Final = 900.0
"""How long unfinished jobs may run on after the agent stops before they are cancelled."""
CANCEL_TIMEOUT_S: Final = 120.0
MCP_READ_TIMEOUT_S: Final = 300.0
FINAL_JOB_STAGES: Final = 10
"""Upper bound on the sandbox stages of a final job, for its timeout (folds x 2 + freeze + test)."""
EXAMPLE_PIPELINE: Final = "linear/pipeline.py"


@dataclass(frozen=True)
class RunRequest:
    """What the user asked for: the dataset, the goal, and any structured fields (spec §5.2)."""

    dataset: Path
    goal: str
    target: str | None = None
    task_type: TaskType | None = None
    primary_metric: MetricName | None = None
    forbidden_columns: tuple[str, ...] = ()
    budget: Budget | None = None  # None: the configured run budget (configs/budgets.yaml)
    run_id: str | None = None
    permitted_tools: tuple[ToolName, ...] | None = None
    secondary_metrics: tuple[MetricName, ...] = ()
    quality_target: float | None = None
    visible_constraints: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is not None:
            TypeAdapter(Identifier).validate_python(self.run_id)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    state: RunState
    final_trial_id: str | None
    verified: bool
    report_json: Path
    report_md: Path
    notes: tuple[str, ...]
    limits_fired: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    """The controller's choice of final trial (D-019 3a) and why."""

    trial_id: str | None
    source_job_id: str | None
    reason: str


def new_run_id() -> str:
    """Allocate a path-safe ID for a new run."""
    return f"run_{uuid.uuid4().hex[:12]}"


def stage1_flag_notes(flags: FeatureFlags) -> list[str]:
    """Say so when the run's flags ask for what Stage 1 does not have yet, instead of letting a
    run look like it used them (D-019)."""
    notes = []
    if flags.approval_policy is not ApprovalPolicy.AUTO_WITHIN_CONTRACT:
        notes.append(
            f"flags.approval_policy is {flags.approval_policy.value!r}, but the interactive plan "
            "gate arrives in S3.5; this run's plan was approved by policy (D-019)."
        )
    return notes


def select_final(
    database: Database, run_id: str, metric: MetricName, nominated: str | None
) -> Selection:
    """The agent's nomination if it is a trial of this run with a completed, scored job;
    otherwise the trial with the best mean validation `metric` (computed by code)."""
    latest_completed: dict[str, str] = {}
    for job in run_jobs(database, run_id):  # oldest first: the last completed job wins
        if job.status is JobState.COMPLETED:
            latest_completed[job.trial_id] = job.job_id
    means: dict[str, float] = {}
    for trial_id in latest_completed:
        result = load_result(database, trial_id, "validation")
        mean = metric_summary(result.metrics, metric).mean if result else None
        if mean is not None:
            means[trial_id] = mean
    if nominated is not None and nominated in means:
        return Selection(nominated, latest_completed[nominated], "nominated by the agent")
    if not means:
        return Selection(None, None, "no trial completed validation")
    sign = 1.0 if metric.direction is MetricDirection.MAXIMIZE else -1.0
    best = max(sorted(means), key=lambda trial_id: sign * means[trial_id])
    reason = f"best mean validation {metric.value}"
    if nominated is not None:
        reason += f" (the agent's nomination {nominated} has no completed, scored job)"
    return Selection(best, latest_completed[best], reason)


def queue_final_job(
    queue: JobQueue, runs_root: Path, run_id: str, trial_id: str, source_job_id: str
) -> str:
    """Queue the FINAL job of `trial_id`: the validated job's exact snapshot, re-run by the
    experiment worker with `phase=final` (D-019). Returns the new job ID."""
    source = protocol.job_dir(runs_root, run_id, source_job_id)
    original = protocol.read_request(source)
    snapshot = {
        name: (source / protocol.TRIAL_SNAPSHOT_DIR / name).read_bytes()
        for name in original.trial_files
    }
    key = f"{run_id}:final_{trial_id}"
    request = SubmitExperimentRequest.model_validate(
        {**original.request.model_dump(mode="json"), "idempotency_key": key}
    )
    job_id = new_job_id()
    protocol.publish_job(
        protocol.job_dir(runs_root, run_id, job_id),
        protocol.ExperimentRequest(
            request=request,
            dataset_id=original.dataset_id,
            dataset_path=original.dataset_path,
            dataset_sha256=original.dataset_sha256,
            trial_files=dict(original.trial_files),
            phase=protocol.ExperimentPhase.FINAL,
            source_job_id=source_job_id,
        ),
        snapshot,
    )
    return queue.enqueue(
        trial_id=trial_id,
        operation=Operation.FIT,
        limits=request.limits,
        idempotency_key=key,
        approval_ref=request.approval_ref,
        initiator="controller",
        actor_role=ActorRole.CONTROLLER,
        job_id=job_id,
    )


class LeadStop(Exception):
    """A decided to stop rather than authorize the next phase."""


class RunController:
    """Runs one request at a time against a state database and a runs/ directory."""

    def __init__(
        self,
        *,
        database: Database,
        runs_root: Path,
        adapter: LLMAdapter,
        run_config: RunConfig,
        workspace_id: str = "ws_local",
        max_agent_turns: int = MAX_AGENT_TURNS,
        poll_seconds: float = POLL_SECONDS,
        worker_factory: WorkerFactory = WorkerProcess,
        tool_channel_wrapper: Callable[[ToolChannel], ToolChannel] | None = None,
    ) -> None:
        """`tool_channel_wrapper` is the evaluation harness's fault injection point (spec §19.2):
        it wraps the agent's connection to the MCP server. The agent can't see or reach it."""
        self.worker_factory = worker_factory
        self.tool_channel_wrapper = tool_channel_wrapper
        self.database = database
        path = database.engine.url.database
        if not path:
            raise ValueError("the controller needs a file-backed SQLite database")
        self.database_path = Path(path).resolve()
        self.runs_root = runs_root.resolve()
        self.adapter = adapter
        self.run_config = run_config
        self.workspace_id = workspace_id
        self.max_agent_turns = max_agent_turns
        self.poll_seconds = poll_seconds

    async def run(self, request: RunRequest) -> RunResult:
        """Execute the run to a terminal state; never raises for a failure inside the run."""
        return await _Run(self, request).execute()


@dataclass
class _Run:
    """The mutable state of one run while the controller executes it."""

    controller: RunController
    request: RunRequest
    run_id: str = ""
    state: RunState = START
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        c = self.controller
        self.run_id = self.request.run_id or new_run_id()
        self.database = c.database
        self.events = EventLog(c.database)
        self.queue = JobQueue(c.database)
        self.dataset = self.request.dataset.resolve()
        self.run_dir = c.runs_root / self.run_id
        self.budget = self.request.budget or c.run_config.budgets.run
        self.family = Family.TABULAR
        self.guard = RunGuard(
            run_id=self.run_id, budget=self.budget, database=c.database, runs_root=c.runs_root
        )
        self.worker = c.worker_factory(
            c.database_path,
            c.runs_root,
            self.run_dir / "logs" / "worker.log",
            worker_id=f"worker-{self.run_id}",
            poll_seconds=min(c.poll_seconds, 2.0),
        )
        self.final_trial_id: str | None = None
        self.final_job_completed = False
        self.role_invocations = 0
        self.lead_assignment = ""
        self.research_brief = ""

    # --- the run ----------------------------------------------------------------------------------

    async def execute(self) -> RunResult:
        self._create()
        try:
            source = self.dataset.read_bytes()
            # Every role and worker uses one private copy of the bytes read at run start.
            # Editing the user's source during a model call cannot change this run's data.
            snapshot = self.run_dir / "data" / f"dataset{self.dataset.suffix.lower()}"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            protocol.write_once(snapshot, source)
            self.dataset = snapshot
            self.dataset_fingerprint = hashlib.sha256(source).hexdigest()
            frame = load_tabular(self.dataset, source=source)
            spec = await self._intake(frame)
            self._advance(RunEvent.INTAKE_DONE, f"task spec v{spec.version}: {spec.task_type}")
            profile = self._profile(frame, spec)
            self._advance(RunEvent.PROFILED, f"profile {profile.fingerprint[:12]}")
            if self._multi:
                await self._research_and_decide(spec, profile)
            contract = await self._prepare_data_contract(spec, frame, profile)
            approval_reason = f"plan approved by policy ({AUTO_APPROVER})"
            if self._multi:
                approval_reason += "; A authorized C assignment: " + json.dumps(
                    _redact(self.lead_assignment), ensure_ascii=True, sort_keys=True
                )
            self._advance(RunEvent.APPROVED, approval_reason)
            await self._phase_b(spec, contract, profile)
            self._advance(RunEvent.PHASE_B_DONE, self.notes[-1] if self.notes else "phase B done")
        except LeadStop:
            self._advance(RunEvent.FAIL, "Agent A chose to report")
        except Exception as exc:  # recorded, then the run still reports
            self._failed(exc)
            self._advance(RunEvent.FAIL, f"{self.state.stage} failed: {type(exc).__name__}")
        verified = False
        try:
            verified = await self._report()
        except Exception as exc:
            self._failed(exc)
            self._advance(RunEvent.FAIL, f"report failed: {type(exc).__name__}")
        else:
            if USER_CANCEL in self.guard.fired:
                event, reason = RunEvent.REPORTED_CANCELLED, "cancelled by the user"
            elif verified:
                event, reason = RunEvent.REPORTED_VERIFIED, "verified final model"
            else:
                event, reason = RunEvent.REPORTED_UNVERIFIED, "no verified model"
            self._advance(event, reason)
        return RunResult(
            run_id=self.run_id,
            state=self.state,
            final_trial_id=self.final_trial_id,
            verified=verified,
            report_json=self.run_dir / "report.json",
            report_md=self.run_dir / "report.md",
            notes=tuple(self.notes),
            limits_fired=tuple(self.guard.fired),
        )

    def _create(self) -> None:
        b = self.budget
        create_run(
            self.database,
            self.run_id,
            self.controller.workspace_id,
            family=self.family,
            status=START.status,
            approval_policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
            budget_usd=None if b.max_usd is None else Decimal(str(b.max_usd)),
            budget_compute_s=(
                None if b.max_compute_seconds is None else Decimal(str(b.max_compute_seconds))
            ),
            model_config=json.loads(self.controller.run_config.snapshot_json()),
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._event(
            EventType.STAGE_TRANSITION,
            StageTransitionPayload(stage=START.stage, status=START.status, reason="run created"),
        )

    @property
    def _multi(self) -> bool:
        return self.controller.run_config.flags.agent_mode is AgentMode.MULTI

    async def _main_role(
        self, role: ActorRole, context: str, *, require_assignment: bool = False
    ) -> LeadAction | ResearchObservation:
        instance = self._role_started(role, ModelTier.LEAD)
        try:
            result = await main_role_call(
                self.controller.adapter,
                self.guard,
                run_id=self.run_id,
                role=role,
                context=context,
                assignment_validator=self._validate_assignment if require_assignment else None,
            )
            if isinstance(result, LeadAction):
                if result.action == "continue" and require_assignment and result.assignment is None:
                    raise ValueError("Agent A must assign concrete work to C")
                if result.assignment is not None:
                    self._validate_assignment(result.assignment)
        except Exception as exc:
            self._handoff(instance, role, HandoffStatus.FAILED)
            safe_message = self.controller.adapter.redact_for_log(str(exc))
            if safe_message != str(exc):
                # The original error must not be exposed as an exception chain.
                raise RuntimeError(safe_message) from None
            raise
        cleaned = self.controller.adapter.redact_model_data(result.model_dump(mode="json"))
        result = type(result).model_validate(cleaned)
        output_dir = self.run_dir / "role_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        persisted = result
        if require_assignment and isinstance(result, LeadAction):
            persisted = result.model_copy(update={"reason": "[FILTERED]"})
        (output_dir / f"{instance}.json").write_text(persisted.model_dump_json(), encoding="utf-8")
        self._handoff(instance, role, HandoffStatus.COMPLETE, output_ref=f"role_output:{instance}")
        return result

    def _validate_assignment(self, assignment: str) -> None:
        """Fail closed before recording or forwarding proposed assignment text."""
        if self.controller.adapter.redact_for_log(assignment) != assignment:
            raise ValueError("C assignment contains credential-shaped content")
        # Plain prose only: no numbers, identifiers, code punctuation or multiline content.
        if not re.fullmatch(r"[^\W\d_]+(?:[ \t'?.,!?-]+[^\W\d_]+)*[.!?]?", assignment):
            raise ValueError("C assignment must be plain prose without identifiers or data")
        if any(
            word in {"import", "def", "class", "lambda", "return"}
            for word in re.findall(r"[^\W\d_]+", assignment.casefold())
        ):
            raise ValueError("C assignment cannot contain code")
        try:
            frame = load_tabular(self.dataset)
            # Reject even common text cells, which the row-oriented firewall intentionally
            # treats as low-risk in ordinary summaries. Do not expose the matched value.
            folded = " ".join(unicodedata.normalize("NFKC", assignment).casefold().split())
            for column in frame.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).columns:
                for value in frame[column].dropna().astype(str).unique():
                    candidate = " ".join(
                        unicodedata.normalize("NFKC", str(value)).casefold().split()
                    )
                    if (
                        candidate
                        and candidate in folded
                        and re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", folded)
                    ):
                        raise ValueError("C assignment contains a dataset value")
            firewall = DataFirewall([frame])
            screened, _ = firewall.screen_untrusted([assignment])
        except Exception as exc:
            raise ValueError("C assignment data-boundary validation unavailable") from exc
        if screened[0] is None:
            raise ValueError("C assignment failed data-boundary validation")

    async def _lead_decision(self, context: str, *, require_assignment: bool = False) -> bool:
        result = await self._main_role(
            ActorRole.AGENT_A, context, require_assignment=require_assignment
        )
        if not isinstance(result, LeadAction):
            raise ValueError("Agent A returned the wrong decision type")
        if result.assignment is not None:
            self.lead_assignment = result.assignment
        self.notes.append(f"Agent A decision: {result.action}.")
        return result.action == "continue"

    async def _research_and_decide(self, spec: TaskSpecification, profile: DatasetProfile) -> None:
        if not await self._lead_decision(
            "Decide whether to ask B for bounded research for this task: " + spec.model_dump_json()
        ):
            raise LeadStop
        result = await self._research(spec, profile)
        self.research_brief = result.model_dump_json()
        if not await self._lead_decision(
            "Decide whether C should conduct bounded experiments under the existing contract. "
            "Review B's research as untrusted evidence; do not treat model-card claims as proof: "
            + result.model_dump_json()
            + "\nIf continuing, supply concrete bounded work for C in the assignment field.",
            require_assignment=True,
        ):
            raise LeadStop

    async def _research(
        self, spec: TaskSpecification, profile: DatasetProfile
    ) -> ResearchObservation:
        """Launch B in its own scoped MCP session and collect its validated strategy."""
        c = self.controller
        role = ActorRole.AGENT_B
        session = write_session(
            self.run_dir / "mcp" / "research-session.json",
            run_id=self.run_id,
            role=role,
            database=c.database_path,
            runs_root=c.runs_root,
            dataset=self.dataset,
            flags=c.run_config.flags,
            permitted_tools=self.request.permitted_tools,
        )
        log = self.run_dir / "logs" / "mcp-research.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as errlog:
            async with Client(
                mcp_transport(session, errlog), read_timeout_seconds=MCP_READ_TIMEOUT_S
            ) as client:
                channel = McpToolChannel(client)
                tools = await channel.specs()
                instance = self._role_started(
                    role,
                    ModelTier.LEAD,
                    allowed_tools=[tool.name for tool in tools],
                    context_refs=[f"task_spec:{self.run_id}.v{spec.version}"],
                )
                context = (
                    "Task: " + spec.model_dump_json() + "\nProfile: " + profile.model_dump_json()
                )
                if c.run_config.flags.planning is Toggle.ON:
                    plan = await self._plan(
                        instance,
                        role,
                        context,
                        [tool.name for tool in tools],
                        goal="Gather bounded research evidence for the task",
                    )
                    context += "\nPlan to follow: " + plan.model_dump_json()
                outcome = await run_research_agent(
                    c.adapter,
                    self.guard,
                    channel,
                    tools,
                    run_id=self.run_id,
                    context=context,
                    max_turns=c.max_agent_turns,
                )
        if outcome.observation is None:
            self._handoff(instance, role, HandoffStatus.PARTIAL)
            reason = outcome.error or "Research could not complete."
            self.notes.append(reason)
            return ResearchObservation(summary="Research unavailable.", uncertainties=[reason])
        cleaned = c.adapter.redact_model_data(outcome.observation.model_dump(mode="json"))
        result = ResearchObservation.model_validate(cleaned)
        output_dir = self.run_dir / "role_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{instance}.json").write_text(result.model_dump_json(), encoding="utf-8")
        self._handoff(instance, role, HandoffStatus.COMPLETE, output_ref=f"role_output:{instance}")
        return result

    async def _intake(self, frame: pd.DataFrame) -> TaskSpecification:
        r = self.request
        role = ActorRole.AGENT_A if self._multi else ActorRole.INTAKE
        instance = self._role_started(role, ModelTier.CHEAP)
        try:
            spec = await run_intake(
                self.controller.adapter,
                self.guard,
                run_id=self.run_id,
                form=IntakeForm(
                    goal=r.goal,
                    target=r.target,
                    task_type=r.task_type,
                    primary_metric=r.primary_metric,
                    secondary_metrics=r.secondary_metrics,
                    quality_target=r.quality_target,
                    forbidden_columns=r.forbidden_columns,
                    budget=self.budget,
                ),
                frame=frame,
                dataset_ref=f"dataset:{DATASET_ID}",
                role=role,
            )
        except Exception:
            self._handoff(instance, role, HandoffStatus.FAILED)
            raise
        save_task_spec(self.database, spec)
        self._handoff(
            instance,
            role,
            HandoffStatus.COMPLETE,
            output_ref=f"task_spec:{self.run_id}.v{spec.version}",
            open_questions=list(spec.open_questions),
        )
        return spec

    def _profile(self, frame: pd.DataFrame, spec: TaskSpecification) -> DatasetProfile:
        if spec.target is None or spec.target not in frame:
            raise ContractError("target required before profiling")
        # The pre-lock profile contains no target values or target-derived statistics.
        profile = profile_tabular(
            frame.drop(columns=[spec.target]),
            run_id=self.run_id,
            dataset_ref=f"dataset:{DATASET_ID}",
        )
        profile.fingerprint = self.dataset_fingerprint
        save_dataset_profile(self.database, profile)
        return profile

    def _contract(self, spec: TaskSpecification, data: DataContract) -> ExperimentContract:
        save_data_contract(self.database, self.run_id, data)
        plan = build_plan(
            spec,
            data,
            self.budget,
            note="Evidence-backed methodology; split locked before experiments.",
        )
        (self.run_dir / "approved_plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        approval = self._approve(
            plan, "Data methodology approved; diagnostic partition to be locked."
        )
        contract = build_experiment_contract(
            spec,
            data,
            approval,
            budget=self.budget,
            job_limits=self.controller.run_config.budgets.job_limits,
        )
        save_experiment_contract(self.database, contract)
        self.notes.append(
            "A reviewed C's data methodology; platform locks diagnostic rows."
            if self._multi
            else "Single-agent data methodology selected; platform locks diagnostic rows."
        )
        self.notes += stage1_flag_notes(self.controller.run_config.flags)
        return contract

    async def _propose_methodology(
        self,
        spec: TaskSpecification,
        profile: DatasetProfile,
        *,
        previous: DataContract | None = None,
    ) -> tuple[DataContract, dict[str, tuple[str, str]]]:
        """Collect a bounded C proposal using only case-permitted diagnostics."""
        c = self.controller
        role = ActorRole.AGENT_C if self._multi else ActorRole.SINGLE_AGENT
        permitted = self.request.permitted_tools
        tools = (
            (ToolName.RUN_DIAGNOSTIC,)
            if permitted is None or ToolName.RUN_DIAGNOSTIC in permitted
            else ()
        )
        session = write_session(
            self.run_dir / "mcp" / "methodology-session.json",
            run_id=self.run_id,
            role=role,
            database=c.database_path,
            runs_root=c.runs_root,
            dataset=self.dataset,
            flags=c.run_config.flags,
            permitted_tools=tools,
        )
        log = self.run_dir / "logs" / "methodology-mcp.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        context = (
            "Task specification: "
            + spec.model_dump_json()
            + "\nTarget-free dataset metadata: "
            + profile.model_dump_json()
        )
        if previous is not None:
            context += (
                "\nThe split is now locked. You may run feature_predictiveness only on the "
                "platform's diagnostic-only partition. You may propose feature exclusions with "
                "diagnostic evidence; preserve target, task, split and non-learned fixes exactly."
                "\nCurrent contract: " + previous.model_dump_json()
            )
        else:
            context += "\nNo target-aware diagnostic is available before the split is locked."
        failure: Exception | None = None
        with log.open("a", encoding="utf-8") as errlog:
            transport = mcp_transport(session, errlog)
            async with Client(transport, read_timeout_seconds=MCP_READ_TIMEOUT_S) as client:
                channel: ToolChannel = McpToolChannel(client)
                tool_specs = await McpToolChannel(client).specs()
                if c.tool_channel_wrapper is not None:
                    channel = c.tool_channel_wrapper(channel)
                instance = self._role_started(
                    role, ModelTier.LEAD, allowed_tools=[tool.name for tool in tool_specs]
                )
                try:
                    if c.run_config.flags.planning is Toggle.ON:
                        plan = await self._plan(
                            instance, role, context, [tool.name for tool in tool_specs]
                        )
                        context += "\nPlan to follow: " + plan.model_dump_json()
                    proposal, evidence = await propose_data_contract(
                        c.adapter,
                        self.guard,
                        channel,
                        tool_specs,
                        run_id=self.run_id,
                        role=role,
                        context=context,
                        max_turns=c.max_agent_turns,
                    )
                except Exception as exc:
                    self._handoff(
                        instance,
                        role,
                        HandoffStatus.PARTIAL
                        if isinstance(exc, RunLimitReached)
                        else HandoffStatus.FAILED,
                    )
                    failure = exc
                else:
                    self._handoff(instance, role, HandoffStatus.COMPLETE)
        if failure is not None:
            raise failure
        # Model-generated strings must pass both existing secret and dataset-value boundaries
        # before forwarding to A or persisting any proposed methodology.
        payload = proposal.model_dump(mode="json")
        if c.adapter.redact_model_data(payload) != payload:
            raise ContractError("data proposal failed sensitive-content validation")
        frame = load_tabular(self.dataset)
        if DataFirewall([frame]).check_payload(payload).blocked:
            raise ContractError("data proposal failed data-boundary validation")
        return proposal, evidence

    async def _prepare_data_contract(
        self, spec: TaskSpecification, frame: pd.DataFrame, profile: DatasetProfile
    ) -> ExperimentContract:
        """Choose, freeze, diagnose and reconcile one evidence-backed data methodology."""
        from hypora.execution.harness.partition import (
            create_partition_lock,
            load_verified_partition,
        )

        columns = [str(column) for column in frame.columns]
        draft, evidence = await self._propose_methodology(spec, profile)
        validate_data_proposal(draft, spec, columns, evidence)
        if self._multi and not await self._lead_decision(
            "Review C's proposed data methodology alongside B's evidence. Report on conflict or "
            "ambiguity; otherwise continue. C proposal: "
            + draft.model_dump_json()
            + "\nB evidence: "
            + self.research_brief
        ):
            raise LeadStop
        provisional = self._contract(spec, draft)
        lock_path = self.run_dir / "partition-lock.json"
        create_partition_lock(frame, provisional, self.dataset_fingerprint, lock_path)
        locked = load_verified_partition(frame, provisional, self.dataset_fingerprint, lock_path)
        # Infer or validate task type from D only; never relay values or a parser error.
        try:
            inferred = _task_type(
                locked.prepared.target.iloc[locked.diagnostic], spec.task_type, explicit=True
            )
        except (ValueError, TypeError) as exc:
            raise ContractError(
                "diagnostic partition cannot validate the proposed task type"
            ) from exc
        if inferred is not spec.task_type:
            raise ContractError("diagnostic partition conflicts with the proposed task type")
        revised, more_evidence = await self._propose_methodology(spec, profile, previous=draft)
        evidence.update(more_evidence)
        validate_data_proposal(revised, spec, columns, evidence)
        if (
            revised.target != draft.target
            or revised.task_type != draft.task_type
            or revised.split != draft.split
            or revised.non_learned_fixes != draft.non_learned_fixes
        ):
            raise ContractError(
                "split-affecting methodology changes after diagnostics are forbidden"
            )
        if self._multi and not await self._lead_decision(
            "Reconcile B's evidence and C's locked-split diagnostics. Report on conflict or "
            "ambiguity; otherwise continue. C proposal: "
            + revised.model_dump_json()
            + "\nB evidence: "
            + self.research_brief
        ):
            raise LeadStop
        if revised == draft:
            return provisional
        # Only feature changes are allowed, so the locked rows remain valid. Approve a new
        # immutable plan and contract rather than mutating the original record.
        data = DataContract.model_validate(
            {**revised.model_dump(mode="json"), "data_contract_version": 2}
        )
        load_verified_partition(frame, provisional, self.dataset_fingerprint, lock_path)
        save_data_contract(self.database, self.run_id, data)
        plan = build_plan(
            spec,
            data,
            self.budget,
            note="Evidence-backed methodology; split locked before experiments.",
        ).model_copy(update={"version": 2})
        (self.run_dir / "approved_plan_v2.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        approval = self._approve(plan, "C's evidence-backed feature revision; split locked")
        contract = build_experiment_contract(
            spec,
            data,
            approval,
            budget=self.budget,
            job_limits=self.controller.run_config.budgets.job_limits,
        ).model_copy(update={"version": 2})
        load_verified_partition(frame, contract, self.dataset_fingerprint, lock_path)
        save_experiment_contract(self.database, contract)
        return contract

    def _approve(self, plan: ApprovedPlan, note: str) -> ApprovalRecord:
        """Approve `plan` by policy and record the request, the decision, and both their events."""
        approval = approve_by_policy(plan, reason=note)
        self._event(
            EventType.APPROVAL_REQUESTED,
            ApprovalRequestedPayload(
                approval_id=approval.approval_id,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=approval.artifact_hash,
                policy=approval.policy,
                summary=f"Phase B plan v{plan.version} within budget; {note}",
            ),
        )
        save_approval(self.database, approval)
        self._event(
            EventType.APPROVAL_GRANTED,
            ApprovalGrantedPayload(
                approval_id=approval.approval_id,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=approval.artifact_hash,
                approver=approval.approver,
                policy=approval.policy,
            ),
            actor=ActorRole.POLICY_ENGINE,
        )
        return approval

    async def _phase_b(
        self, spec: TaskSpecification, contract: ExperimentContract, profile: DatasetProfile
    ) -> None:
        try:
            self.worker.start()
            await self._experiments(spec, contract, profile)
        finally:
            try:
                # A transport or controller failure can bypass the normal drain.
                # Cancel before stopping: queued jobs must not outlive their run.
                for job in run_jobs(self.database, self.run_id):
                    if job.status not in TERMINAL_STATES:
                        self._cancel(job.job_id, "controller worker shutdown")
            finally:
                self.worker.stop()

    async def _experiments(
        self, spec: TaskSpecification, contract: ExperimentContract, profile: DatasetProfile
    ) -> None:
        outcome = await self._agent(spec, contract, profile)
        if outcome is None:
            return
        self.notes.append(_describe(outcome))
        self.notes += await self._drain(cancel=outcome.stop is not None)
        nominated = outcome.final.final_trial_id if outcome.final else None
        if self._multi and not await self._lead_decision(
            "Review C's completed work and decide whether to continue to independent final "
            "verification, or report. The platform remains responsible for verification. "
            f"C output: {outcome.final.model_dump_json() if outcome.final else 'unavailable'}. "
            f"Platform evidence: {self._trial_evidence()}. Run notes: {self.notes}"
        ):
            return
        completed = {
            job.trial_id
            for job in run_jobs(self.database, self.run_id)
            if job.status is JobState.COMPLETED
        }
        kinds = {
            trial.kind
            for trial in list_trials(self.database, self.run_id)
            if trial.trial_id in completed
        }
        if self._multi and not {"baseline", "candidate"} <= kinds:
            self.notes.append(
                "No final model: a completed baseline and candidate are required for comparison."
            )
            return
        selection = select_final(self.database, self.run_id, contract.primary_metric, nominated)
        if selection.trial_id is None or selection.source_job_id is None:
            self.notes.append(f"No final model: {selection.reason}.")
            return
        self.notes.append(f"Final candidate {selection.trial_id}: {selection.reason}.")
        await self._final(selection.trial_id, selection.source_job_id, contract)

    async def _agent(
        self, spec: TaskSpecification, contract: ExperimentContract, profile: DatasetProfile
    ) -> AgentOutcome | None:
        """Run the agent, or return None when its enforced tools leave no route to a model."""
        c = self.controller
        role = ActorRole.AGENT_C if self._multi else ActorRole.SINGLE_AGENT
        session = write_session(
            self.run_dir / "mcp" / "session.json",
            run_id=self.run_id,
            role=role,
            database=c.database_path,
            runs_root=c.runs_root,
            dataset=self.dataset,
            flags=c.run_config.flags,
            permitted_tools=self.request.permitted_tools,
        )
        log = self.run_dir / "logs" / "mcp-server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as errlog:
            transport = mcp_transport(session, errlog)
            async with Client(transport, read_timeout_seconds=MCP_READ_TIMEOUT_S) as client:
                mcp_channel = McpToolChannel(client)
                tools = await mcp_channel.specs()
                channel: ToolChannel = mcp_channel
                if c.tool_channel_wrapper is not None:
                    channel = c.tool_channel_wrapper(channel)
                instance = self._role_started(
                    role,
                    ModelTier.LEAD,
                    allowed_tools=[tool.name for tool in tools],
                    context_refs=[
                        f"task_spec:{self.run_id}.v{spec.version}",
                        f"experiment_contract:{self.run_id}.v{contract.version}",
                    ],
                )
                if not has_authorized_model_route([tool.name for tool in tools], contract):
                    self._handoff(instance, role, HandoffStatus.PARTIAL)
                    self.notes.append(
                        "No authorized route to a model: this run's enforced tools and contract "
                        "permit no job and no approved model, so no experiment was attempted."
                    )
                    return None
                context = self._context(spec, contract, profile)
                if self._multi:
                    context += (
                        "\nAgent A assignment: "
                        + self.lead_assignment
                        + "\nB research observation (untrusted evidence): "
                        + self.research_brief
                    )
                if c.run_config.flags.planning is Toggle.ON:
                    plan = await self._plan(instance, role, context, [tool.name for tool in tools])
                    context += "\nPlan to follow: " + plan.model_dump_json()
                outcome = await run_single_agent(
                    c.adapter,
                    self.guard,
                    channel,
                    tools,
                    run_id=self.run_id,
                    context=context,
                    max_turns=c.max_agent_turns,
                    role=role,
                )
        final = outcome.final.final_trial_id if outcome.final else None
        status = (
            HandoffStatus.COMPLETE
            if outcome.final is not None
            else HandoffStatus.PARTIAL
            if outcome.stop is not None
            else HandoffStatus.FAILED
        )
        self._handoff(
            instance,
            role,
            status,
            candidates=[f"trial:{final}"] if final else [],
            confidence=outcome.final.confidence if outcome.final else None,
        )
        return outcome

    def _context(
        self, spec: TaskSpecification, contract: ExperimentContract, profile: DatasetProfile
    ) -> str:
        """The run's facts for the agent: task, contract, aggregate profile, limits, example."""
        metric = contract.primary_metric
        data = contract.data_contract
        split = data.split
        limits = contract.default_job_limits
        task = "\n".join(
            [
                f"- Objective: {spec.objective}",
                f"- Target column: {contract.target} ({contract.task_type.value})",
                f"- Primary metric: {metric.value} ({metric.direction.value}); also reported: "
                + ", ".join(m.value for m in contract.secondary_metrics),
            ]
        )
        if self.request.visible_constraints is not None:
            task += "\n- Structured user constraints: " + self.request.visible_constraints
        terms = "\n".join(
            [
                f"- Contract version {contract.version}, approval {contract.approval_id}",
                f"- Features ({len(data.allowed_features)}): {', '.join(data.allowed_features)}",
                "- Excluded features: "
                + (", ".join(f.name for f in data.excluded_features) or "none"),
                f"- Split: {split.strategy.value}; {split.test_fraction:.0%} of rows are a "
                f"held-out test set you never see; validation folds: {split.validation}",
                f"- Default job limits per stage: {limits.max_wall_seconds} s, "
                f"{limits.max_memory_gb:g} GB memory, {limits.cpu_threads or 1} CPU thread(s)",
            ]
        )
        tabular = profile.tabular
        facts = (
            [
                f"- {tabular.n_rows} rows, {tabular.n_columns} columns, "
                f"{tabular.n_duplicate_rows} duplicate rows",
                f"- Leakage screen: {leakage_screen_summary(tabular)}",
                "- Leakage candidates (not verdicts): "
                + (", ".join(c.column for c in tabular.leakage_candidates) or "none"),
                "- ID-like columns: "
                + (", ".join(c.column for c in tabular.id_like_columns) or "none"),
            ]
            if tabular is not None
            else []
        )
        facts.append("- Call inspect_dataset for per-column types, missing values, and balance.")
        b = self.budget
        caps = [
            f"- Trials: at most {b.max_trials}" if b.max_trials is not None else None,
            f"- Steps (model calls + tool calls): at most {b.max_steps}"
            if b.max_steps is not None
            else None,
            f"- Your model turns: at most {self.controller.max_agent_turns}",
            f"- Wall clock: {b.max_wall_clock_seconds:g} s for the whole run"
            if b.max_wall_clock_seconds is not None
            else None,
        ]
        example = (
            resources.files("hypora.execution.templates.tabular")
            .joinpath(EXAMPLE_PIPELINE)
            .read_text(encoding="utf-8")
        )
        return load_prompt("single_agent_context").render(
            run_id=self.run_id,
            task=task,
            contract=terms,
            profile="\n".join(facts),
            limits="\n".join(cap for cap in caps if cap),
            example=example.strip(),
        )

    async def _drain(self, *, cancel: bool) -> list[str]:
        """Wait for the run's unfinished jobs (bounded), or cancel them when a limit fired."""
        remaining = self.guard.remaining_wall_clock()
        wait = DRAIN_TIMEOUT_S if remaining is None else min(DRAIN_TIMEOUT_S, remaining)
        deadline = anyio.current_time() + wait
        cancelled = False
        notes: list[str] = []
        stop_reason = None
        while True:
            if not cancelled:
                try:
                    self.guard.before_job()
                except RunLimitReached as reached:
                    cancel = True
                    stop_reason = str(reached)
            pending = [
                job
                for job in run_jobs(self.database, self.run_id)
                if job.status not in TERMINAL_STATES
            ]
            if not pending:
                return notes
            now = anyio.current_time()
            stuck = not self.worker.alive()
            if not cancelled and (cancel or stuck or now >= deadline):
                why = (
                    (stop_reason or "a run limit fired")
                    if cancel
                    else ("the drain timed out or the worker exited")
                )
                for job in pending:
                    self._cancel(job.job_id, why)
                notes.append(f"Cancelled {len(pending)} unfinished job(s): {why}.")
                cancelled, deadline = True, now + CANCEL_TIMEOUT_S
            elif cancelled and (stuck or now >= deadline):
                notes.append(f"{len(pending)} job(s) did not confirm cancellation.")
                return notes
            await anyio.sleep(self.controller.poll_seconds)

    def _cancel(self, job_id: str, reason: str) -> None:
        with contextlib.suppress(InvalidJobTransition, StaleJobState):
            self.queue.request_cancel(job_id, "controller", reason, ActorRole.CONTROLLER)

    async def _final(self, trial_id: str, source_job_id: str, contract: ExperimentContract) -> None:
        """Queue and await the final job; record the verifier's verdict as its handoff."""
        try:
            self.guard.before_job()
        except RunLimitReached as reached:
            self.notes.append(f"No final evaluation: {reached}.")
            return
        job_id = queue_final_job(
            self.queue, self.controller.runs_root, self.run_id, trial_id, source_job_id
        )
        self.final_trial_id = trial_id
        timeout = contract.default_job_limits.max_wall_seconds * FINAL_JOB_STAGES
        remaining = self.guard.remaining_wall_clock()
        status = await self._await_job(
            job_id, timeout if remaining is None else min(timeout, remaining)
        )
        self.guard.sync_compute()
        folder = protocol.job_dir(self.controller.runs_root, self.run_id, job_id)
        outcome = protocol.read_outcome(folder)
        verification = outcome.verification if outcome else None
        self.final_job_completed = status is JobState.COMPLETED and outcome is not None
        passed = self.final_job_completed and verification is not None and verification.passed
        if status is not JobState.COMPLETED or outcome is None:
            code = outcome.error_code if outcome else None
            self.notes.append(
                f"The final job {job_id} ended {status.value} ({code or 'no outcome'})."
            )
        else:
            failed = [
                c.check.value
                for c in (verification.checks if verification else [])
                if c.status is not CheckStatus.PASSED and c.status is not CheckStatus.SKIPPED
            ]
            self.notes.append(
                f"Final job {job_id}: test split scored once; verifier v1 "
                + ("passed." if passed else f"failed ({', '.join(failed) or 'no verdict'}).")
            )
        self._handoff(
            f"ver_{self.run_id}",
            ActorRole.VERIFIER,
            HandoffStatus.COMPLETE if passed else HandoffStatus.FAILED,
            candidates=[f"trial:{trial_id}"],
            evidence_refs=[f"job:{job_id}"],
            output_ref=f"result:{trial_id}/test" if outcome and outcome.test_scores else None,
        )

    async def _await_job(self, job_id: str, timeout: float) -> JobState:
        deadline = anyio.current_time() + timeout
        cancelled = False
        while True:
            if not cancelled:
                try:
                    self.guard.before_job()
                except RunLimitReached as reached:
                    self._cancel(job_id, str(reached))
                    self.notes.append(f"Stopped the final job: {reached}.")
                    cancelled = True
                    deadline = anyio.current_time() + CANCEL_TIMEOUT_S
            status = self.queue.get(job_id).status
            if status in TERMINAL_STATES:
                return status
            if anyio.current_time() >= deadline or not self.worker.alive():
                self._cancel(job_id, "the final job timed out or the worker exited")
                return self.queue.get(job_id).status
            await anyio.sleep(self.controller.poll_seconds)

    def _write_facts(self, narrative_note: str | None = None) -> Report | None:
        """Write the report from stored records; None if the run stopped before its contract."""
        json_path, md_path = self.run_dir / "report.json", self.run_dir / "report.md"
        self._notice_cancel()
        final = (
            self.final_trial_id
            if self.final_job_completed and USER_CANCEL not in self.guard.fired
            else None
        )
        try:
            report = build_report(
                self.database,
                self.run_id,
                final_trial_id=final,
                notes=self.notes,
                runs_root=self.controller.runs_root,
            )
        except RecordNotFound:  # stopped before the contract: nothing was measured
            early = EarlyStopReport(
                run_id=self.run_id,
                notes=self.notes,
                evidence=derive_outcome_evidence(
                    self.database, self.controller.runs_root, self.run_id
                ),
            )
            json_path.write_text(early.model_dump_json(indent=2), encoding="utf-8")
            md_path.write_text(render_early_stop(early), encoding="utf-8")
            return None
        report = report.model_copy(update={"narrative_note": narrative_note})
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        return report

    async def _report(self) -> bool:
        """Write report.json and report.md; returns whether the final model was verified."""
        json_path, md_path = self.run_dir / "report.json", self.run_dir / "report.md"
        # The facts first, so a failing narrative still leaves a complete report.
        report = self._write_facts()
        if report is None:
            return False
        role = ActorRole.AGENT_A if self._multi else ActorRole.REPORTER
        instance = self._role_started(role, ModelTier.CHEAP)
        report = await write_narrative(self.controller.adapter, self.guard, report, role=role)
        self._notice_cancel()
        if USER_CANCEL in self.guard.fired:
            report = report.model_copy(
                update={
                    "outcome": "no_verified_model",
                    "final": None,
                    "narrative": None,
                    "narrative_note": "No narrative: cancelled by the user.",
                    "notes": self.notes,
                }
            )
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        self._handoff(
            instance,
            role,
            HandoffStatus.COMPLETE if report.narrative else HandoffStatus.PARTIAL,
            output_ref=f"report:{self.run_id}",
        )
        return report.outcome == "verified_model"

    def _notice_cancel(self) -> None:
        try:
            self.guard.check_cancel()
        except RunLimitReached as reached:
            note = f"Run stopped: {reached}."
            if note not in self.notes:
                self.notes.append(note)

    # --- state, events --------------------------------------------------------------------------

    def _advance(self, event: RunEvent, reason: str) -> None:
        """Apply one transition-table row: the runs row and its stage_transition event commit
        together."""
        previous = self.state
        state = next_state(previous, event)
        payload = StageTransitionPayload(
            previous_stage=previous.stage,
            stage=state.stage,
            previous_status=previous.status,
            status=state.status,
            reason=f"{event.value}: {reason}"[:500],
        )
        with self.database.write() as connection:
            set_run_status(connection, self.run_id, state.status)
            self.events.append(
                self.run_id,
                EventType.STAGE_TRANSITION,
                ActorRole.CONTROLLER,
                payload.model_dump(mode="json"),
                connection=connection,
            )
        self.state = state

    def _failed(self, exc: Exception) -> None:
        """Record a stage failure. Only messages the platform wrote are kept; anything else is
        reduced to its type, so no data value can reach the report or the Reporter."""
        if isinstance(exc, RunLimitReached):  # the guard already recorded the event
            self.notes.append(f"Run stopped: {exc}.")
            return
        own = isinstance(exc, IntakeError | ContractError)
        code = (
            "INTAKE_FAILED"
            if isinstance(exc, IntakeError)
            else "CONTRACT_FAILED"
            if isinstance(exc, ContractError)
            else "CONTROLLER_ERROR"
        )
        message = str(exc)[:300] if own else type(exc).__name__
        stage = self.state.stage.value if self.state.stage else "run"
        self._event(
            EventType.ERROR,
            ErrorPayload(
                component="controller",
                code=code,
                message=f"{stage}: {message}",
                error_class=ErrorClass.MISSING_DATA if own else None,
                retryable=False,
            ),
        )
        self.notes.append(f"The {stage} stage failed: {message}.")

    def _event(
        self, type: EventType, payload: BaseModel, actor: ActorRole = ActorRole.CONTROLLER
    ) -> None:
        self.events.append(self.run_id, type, actor, payload.model_dump(mode="json"))

    def _role_started(
        self,
        role: ActorRole,
        tier: ModelTier,
        *,
        allowed_tools: list[str] | None = None,
        context_refs: list[str] | None = None,
    ) -> str:
        self.role_invocations += 1
        instance = f"ri_{self.run_id}_{role.value}_{self.role_invocations}"
        self.events.append(
            self.run_id,
            EventType.ROLE_STARTED,
            role,
            RoleStartedPayload(
                role_instance_id=instance,
                tier=tier,
                allowed_tools=allowed_tools or [],
                context_refs=context_refs or [],
            ).model_dump(mode="json"),
        )
        return instance

    def _trial_evidence(self) -> str:
        """Return bounded stored validation facts, not another role's conversation."""
        trials = list_trials(self.database, self.run_id)
        facts = []
        for trial in trials[:20]:
            result = load_result(self.database, trial.trial_id, "validation")
            facts.append(
                {
                    "trial_id": trial.trial_id,
                    "kind": trial.kind,
                    "status": trial.status,
                    "hypothesis": trial.hypothesis[:500],
                    "validation": result.metrics if result else None,
                    "verification": result.verification if result else None,
                }
            )
        return json.dumps({"trials": facts, "omitted": max(0, len(trials) - 20)})[:20000]

    async def _plan(
        self,
        instance: str,
        role: ActorRole,
        context: str,
        allowed_tools: list[str],
        *,
        goal: str = "Run a bounded experiment under the approved contract",
    ) -> Plan:
        planner = ActorRole.AGENT_A if self._multi else role
        planner_instance = self._role_started(planner, ModelTier.LEAD) if self._multi else instance
        try:
            draft = await draft_plan(
                self.controller.adapter,
                self.guard,
                run_id=self.run_id,
                role=planner,
                context=context,
                allowed_tools=allowed_tools,
            )
            plan = Plan(
                plan_id=f"plan_{uuid.uuid4().hex[:12]}",
                run_id=self.run_id,
                role=planner,
                goal=goal,
                steps=[
                    PlanStep(
                        step=i,
                        tool=step.tool,
                        purpose=step.purpose,
                        success_check=step.success_check,
                    )
                    for i, step in enumerate(draft.steps, 1)
                ],
            )
            self._event(
                EventType.PLAN, PlanPayload(role_instance_id=planner_instance, plan=plan), planner
            )
        except Exception:
            if self._multi:
                self._handoff(planner_instance, planner, HandoffStatus.FAILED)
            raise
        if self._multi:
            self._handoff(planner_instance, planner, HandoffStatus.COMPLETE)
        return plan

    def _handoff(
        self,
        instance: str,
        role: ActorRole,
        status: HandoffStatus,
        *,
        candidates: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        output_ref: str | None = None,
        confidence: float | None = None,
        open_questions: list[str] | None = None,
    ) -> None:
        handoff = Handoff(
            run_id=self.run_id,
            from_role=role,
            to=ActorRole.CONTROLLER,
            status=status,
            candidates=candidates or [],
            evidence_refs=evidence_refs or [],
            output_ref=output_ref,
            confidence=confidence,
            open_questions=open_questions or [],
        )
        self.events.append(
            self.run_id,
            EventType.HANDOFF,
            role,
            HandoffPayload(role_instance_id=instance, handoff=handoff).model_dump(mode="json"),
        )


def _describe(outcome: AgentOutcome) -> str:
    done = f"{outcome.turns} model turns, {outcome.tool_calls} tool calls"
    if outcome.stop is not None:
        return f"The agent was stopped after {done}: {outcome.stop}."
    if outcome.error is not None:
        return f"The agent's model call failed after {done} ({outcome.error.split(':')[0]})."
    nominated = outcome.final.final_trial_id if outcome.final else None
    return f"The agent finished after {done}; it nominated {nominated or 'no trial'}."


__all__ = [
    "RunController",
    "RunRequest",
    "RunResult",
    "Selection",
    "new_run_id",
    "queue_final_job",
    "select_final",
]
