"""Decision and fault evidence for the run report, from durable platform records only (D-032).

The report's prose can't show that a run decided or handled a fault correctly, and ordinary report
facts don't either. This module states those claims in a form anyone can recompute from the event
log, the job folders, the stored task and contract, and the hash-checked approved plan. No field
comes from an LLM, from benchmark metadata, or from free text.

Only events before the run enters REPORT count, so the Reporter's own events can't change the result
and a later recomputation gives the same answer as the one written into the report.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from hypora.core.schemas import (
    ActorRole,
    AnyEvent,
    ApprovedPlan,
    ExperimentContract,
    HandoffStatus,
    JobState,
    Stage,
    StrictModel,
    TaskSpecification,
    ToolName,
)
from hypora.core.schemas.events import (
    ErrorEvent,
    HandoffEvent,
    JobTransitionEvent,
    RoleStartedEvent,
    StageTransitionEvent,
    ToolResultEvent,
    ToolResultStatus,
)
from hypora.core.schemas.strategy import RecommendationKind
from hypora.execution.protocol import experiments as protocol
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import (
    RecordNotFound,
    load_experiment_contract,
    load_task_spec,
    run_jobs,
)

MODEL_PRODUCING_TOOL: Final = ToolName.SUBMIT_EXPERIMENT.value
"""Every trained or reused model starts from a job, and only this tool queues the agent's jobs."""
CANDIDATE_TOOLS: Final = frozenset(
    {ToolName.WRITE_TRIAL_FILES.value, ToolName.SUBMIT_EXPERIMENT.value}
)
"""The tools a candidate attempt goes through: their failures are failures of that attempt."""
CONTROLLER_COMPONENT: Final = "controller"
AMBIGUOUS_STOP_CODES: Final = frozenset({"RUN_LIMIT_REACHED", "RUN_CANCELLED"})
"""A limit or a user cancel also ends a run, so a fault beside one can't be named as the cause."""
FAILED_STATES: Final = frozenset({JobState.FAILED, JobState.FAILED_TERMINAL})
UNCODED_JOB_FAILURE: Final = "JOB_FAILED"
DATASET_UNREADABLE: Final = "DATASET_UNREADABLE"
"""The controller could not read the dataset's bytes as a table: the one finding that is about the
data itself. A job's `CONTRACT_DATA_ERROR` is not: the worker gives that code to any `ValueError`,
`KeyError`, or `TypeError` in the harness, whatever raised it."""

_CODE: Final = re.compile(r"\b([A-Z][A-Z0-9_]{2,}):")
_TRAIN_PLANS: Final = frozenset(
    {RecommendationKind.TRAIN_TASK_SPECIFIC, RecommendationKind.FINE_TUNE}
)
"""A plan that evaluates first and may fine-tune states no single intent, so it maps to neither."""

IntendedDecision = Literal["train", "reuse", "infeasible"]
FaultOutcome = Literal["recovered", "stopped_safely", "unresolved"]


class DecisionEvidence(StrictModel):
    """What the run was authorized to do, and the records that say so.

    `infeasible` means one narrow thing: the enforced tools and the contract leave no authorized way
    to produce a model on this platform. A failed pipeline or a spent budget never establishes it,
    and it says nothing about whether the task could be learned elsewhere.
    """

    intended: IntendedDecision
    basis: Literal["approved_plan", "no_authorized_model_route"]
    task_spec_ref: str
    contract_ref: str
    approved_plan_hash: str
    capability_event_seq: int | None = None
    enforced_tools: list[str] | None = None
    approved_model_refs: list[str]


class SafeStop(StrictModel):
    """The deterministic reason a run ended without a model, when one can be named."""

    cause: Literal["controller_error", "all_candidates_failed"]
    event_seqs: list[int]


class FaultEvidence(StrictModel):
    """One observed failure and what the platform's records show happened next."""

    source: Literal["tool_result", "job_transition", "controller_error"]
    event_seq: int
    code: str
    tool: str | None = None
    call_id: str | None = None
    job_id: str | None = None
    outcome: FaultOutcome
    recovery_job_id: str | None = None


class OutcomeEvidence(StrictModel):
    """The report's structured account of the decision, the stop, and every observed failure."""

    decision: DecisionEvidence | None = None
    stop: SafeStop | None = None
    faults: list[FaultEvidence]


def _events_before_report(events: Sequence[AnyEvent]) -> list[AnyEvent]:
    for index, event in enumerate(events):
        if isinstance(event, StageTransitionEvent) and event.payload.stage is Stage.REPORT:
            return list(events[:index])
    return list(events)


class _Records:
    """The evidence inputs for one derivation."""

    def __init__(self, database: Database, runs_root: Path, run_id: str) -> None:
        self.database = database
        self.runs_root = runs_root
        self.run_id = run_id
        self.events = _events_before_report(EventLog(database).read(run_id))
        self._requests: dict[str, protocol.ExperimentRequest | None] = {}

    def request(self, job_id: str) -> protocol.ExperimentRequest | None:
        """A job's immutable request; None when the folder has none that parses."""
        if job_id not in self._requests:
            folder = protocol.job_dir(self.runs_root, self.run_id, job_id)
            try:
                self._requests[job_id] = protocol.read_request(folder)
            except (OSError, ValueError):
                self._requests[job_id] = None
        return self._requests[job_id]

    def transitions(self) -> list[JobTransitionEvent]:
        return [event for event in self.events if isinstance(event, JobTransitionEvent)]

    def tool_results(self) -> list[ToolResultEvent]:
        return [event for event in self.events if isinstance(event, ToolResultEvent)]

    def first_started(self, job_id: str) -> int | None:
        return next(
            (
                event.seq
                for event in self.transitions()
                if event.payload.job_id == job_id and event.payload.to_state is JobState.RUNNING
            ),
            None,
        )

    def is_final_job(self, job_id: str) -> bool:
        request = self.request(job_id)
        return request is not None and request.phase is protocol.ExperimentPhase.FINAL

    def final_completion(self) -> tuple[str, int] | None:
        """The FINAL job that completed, with the event that says so."""
        for event in reversed(self.transitions()):
            if event.payload.to_state is JobState.COMPLETED and self.is_final_job(
                event.payload.job_id
            ):
                return event.payload.job_id, event.seq
        return None


# --- The decision -----------------------------------------------------------------------------


def _agent_start(events: Sequence[AnyEvent]) -> RoleStartedEvent | None:
    """The agent's start event, which records the tools its MCP session really served."""
    starts = [
        event
        for event in events
        if isinstance(event, RoleStartedEvent) and event.actor_role is ActorRole.SINGLE_AGENT
    ]
    return starts[-1] if starts else None


def has_authorized_model_route(tools: Sequence[str], contract: ExperimentContract) -> bool:
    """Whether a session serving `tools` under `contract` could lead to a model at all."""
    return MODEL_PRODUCING_TOOL in tools or bool(contract.approved_model_refs)


def _planned_intent(records: _Records, contract: ExperimentContract) -> IntendedDecision | None:
    path = records.runs_root / records.run_id / "approved_plan.json"
    try:
        plan = ApprovedPlan.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None
    if plan.content_hash() != contract.approved_plan_hash:
        return None
    kind = plan.recommendation.kind
    if kind in _TRAIN_PLANS:
        return "train"
    return "reuse" if kind is RecommendationKind.REUSE_PRETRAINED else None


def _decision(records: _Records) -> DecisionEvidence | None:
    try:
        spec: TaskSpecification = load_task_spec(records.database, records.run_id)
        contract = load_experiment_contract(records.database, records.run_id)
    except RecordNotFound:
        return None
    start = _agent_start(records.events)
    tools = None if start is None else sorted(start.payload.allowed_tools)
    jobs = run_jobs(records.database, records.run_id)
    no_route = tools is not None and not jobs and not has_authorized_model_route(tools, contract)
    intended: IntendedDecision | None = (
        "infeasible" if no_route else _planned_intent(records, contract)
    )
    if intended is None:
        return None
    return DecisionEvidence(
        intended=intended,
        basis="no_authorized_model_route" if no_route else "approved_plan",
        task_spec_ref=f"task_spec:{records.run_id}.v{spec.version}",
        contract_ref=f"contract:{contract.hash}",
        approved_plan_hash=contract.approved_plan_hash,
        capability_event_seq=None if start is None else start.seq,
        enforced_tools=tools,
        approved_model_refs=list(contract.approved_model_refs),
    )


# --- The stop ------------------------------------------------------------------------------------


def _agent_finished_by_itself(events: Sequence[AnyEvent]) -> bool:
    """A started agent must have handed back complete: a limit or a dead provider isn't a fault's
    doing. A system with no LLM agent has nothing to check here."""
    started = {
        event.payload.role_instance_id
        for event in events
        if isinstance(event, RoleStartedEvent) and event.actor_role is ActorRole.SINGLE_AGENT
    }
    finished = {
        event.payload.role_instance_id
        for event in events
        if isinstance(event, HandoffEvent)
        and event.payload.handoff.status is HandoffStatus.COMPLETE
    }
    return started <= finished


def _unresolved_tool_failures(results: Sequence[ToolResultEvent]) -> list[int]:
    """Candidate-tool errors nothing overcame: a later success of the same tool resolves one."""
    return [
        result.seq
        for result in results
        if result.payload.status is ToolResultStatus.ERROR
        and not any(
            later.seq > result.seq
            and later.payload.tool == result.payload.tool
            and later.payload.status is ToolResultStatus.OK
            for later in results
        )
    ]


def _all_candidates_failed(records: _Records) -> SafeStop | None:
    """The attempts that ended the run, only when they are all it has left.

    A failure that was overcome is not among them, so it can't be named as the cause of the stop.
    """
    latest: dict[str, JobState] = {}
    for event in records.transitions():
        latest[event.payload.job_id] = event.payload.to_state
    if set(latest.values()) - FAILED_STATES - {JobState.COMPLETED}:
        return None  # cancelled or unfinished work failed for no reason this run can name
    failed_jobs = {job for job, state in latest.items() if state in FAILED_STATES}
    job_failures = [
        event.seq
        for event in records.transitions()
        if event.payload.to_state in FAILED_STATES and event.payload.job_id in failed_jobs
    ]
    failed_sources = set()
    for job in failed_jobs:
        request = records.request(job)
        if request is not None and request.phase is protocol.ExperimentPhase.FINAL:
            failed_sources.add(request.source_job_id)
    # A candidate that worked is usable unless its own final job then failed.
    if any(
        state is JobState.COMPLETED and job not in failed_sources for job, state in latest.items()
    ):
        return None
    results = records.tool_results()
    writes = [r for r in results if r.payload.tool == ToolName.WRITE_TRIAL_FILES.value]
    submits = [r for r in results if r.payload.tool == MODEL_PRODUCING_TOOL]
    written = sum(r.payload.status is ToolResultStatus.OK for r in writes)
    accepted = sum(r.payload.status is ToolResultStatus.OK for r in submits)
    jobs_from_tools = sum(not records.is_final_job(job) for job in latest)
    # A trial written but never tried, or a job with no record of how it ended, did not fail.
    if written > len(submits) or accepted > jobs_from_tools:
        return None
    candidate_results = [*writes, *submits]
    seqs = sorted((*_unresolved_tool_failures(candidate_results), *job_failures))
    return SafeStop(cause="all_candidates_failed", event_seqs=seqs) if seqs else None


def _safe_stop(records: _Records) -> SafeStop | None:
    if records.final_completion() is not None:
        return None
    errors = [event for event in records.events if isinstance(event, ErrorEvent)]
    if any(event.payload.code in AMBIGUOUS_STOP_CODES for event in errors):
        return None
    controller = [e for e in errors if e.payload.component == CONTROLLER_COMPONENT]
    if controller:
        return SafeStop(cause="controller_error", event_seqs=[controller[-1].seq])
    return _all_candidates_failed(records) if _agent_finished_by_itself(records.events) else None


# --- Faults --------------------------------------------------------------------------------------


def _same_intent(failed: protocol.ExperimentRequest, later: protocol.ExperimentRequest) -> bool:
    """A replacement does the failed job's work: same contract, data, operation, pinned model."""
    return (
        failed.request.contract_version == later.request.contract_version
        and failed.dataset_sha256 == later.dataset_sha256
        and failed.request.operation is later.request.operation
        and failed.request.model_ref == later.request.model_ref
    )


def _succeeded_between(
    records: _Records, tool: str, after: int, before: int, **identity: str | None
) -> bool:
    """The tool worked between two points, for this exact trial or job when one is named."""
    return any(
        after < result.seq < before
        and result.payload.tool == tool
        and result.payload.status is ToolResultStatus.OK
        and all(result.payload.result_summary.get(key) == value for key, value in identity.items())
        for result in records.tool_results()
    )


def _ran_to_completion_after(records: _Records, job_id: str, seq: int) -> bool:
    """A start after `seq`, then a completion after that start."""
    transitions = [e for e in records.transitions() if e.payload.job_id == job_id]
    starts = [e.seq for e in transitions if e.payload.to_state is JobState.RUNNING and e.seq > seq]
    return bool(starts) and any(
        e.payload.to_state is JobState.COMPLETED and e.seq > starts[0] for e in transitions
    )


def _retried(records: _Records, failure: JobTransitionEvent) -> bool:
    """The same job really ran again: a later attempt started after the failure, then completed."""
    transitions = [e for e in records.transitions() if e.payload.job_id == failure.payload.job_id]
    starts = [
        e.seq
        for e in transitions
        if e.payload.to_state is JobState.RUNNING
        and e.seq > failure.seq
        and e.payload.attempt > failure.payload.attempt
    ]
    return bool(starts) and any(
        e.payload.to_state is JobState.COMPLETED and e.seq > starts[0] for e in transitions
    )


def _recovered_tool(
    records: _Records, event: ToolResultEvent, final: tuple[str, int]
) -> str | None:
    """The same tool worked again, before and in support of the work behind the final model."""
    tool = event.payload.tool
    final_job, _ = final
    final_started = records.first_started(final_job)
    if final_started is None:
        return None
    if tool not in CANDIDATE_TOOLS:
        return final_job if _succeeded_between(records, tool, event.seq, final_started) else None
    request = records.request(final_job)
    source = None if request is None else request.source_job_id
    replacement = None if source is None else records.request(source)
    started = None if source is None else records.first_started(source)
    if source is None or replacement is None or started is None or started <= event.seq:
        return None
    if not _ran_to_completion_after(records, source, event.seq):
        return None
    # Candidate tools name what they made, so the retry is tied to this very job or its trial.
    identity = (
        {"job_id": source}
        if tool == MODEL_PRODUCING_TOOL
        else {"trial_id": replacement.request.trial_id}
    )
    return source if _succeeded_between(records, tool, event.seq, started, **identity) else None


def _recovered_job(
    records: _Records, event: JobTransitionEvent, final: tuple[str, int]
) -> str | None:
    """A real retry of the job itself, or a replacement of the same work begun after it failed."""
    failed_job = event.payload.job_id
    final_job, _ = final
    if failed_job == final_job:
        return final_job if _retried(records, event) else None
    request = records.request(final_job)
    source = None if request is None else request.source_job_id
    if source is None:
        return None
    failed = records.request(failed_job)
    replacement = records.request(source)
    if failed is None or replacement is None or not _same_intent(failed, replacement):
        return None
    if source == failed_job:
        return source if _retried(records, event) else None
    return source if _ran_to_completion_after(records, source, event.seq) else None


def _fault_outcome(
    event_seq: int, stop: SafeStop | None, *, recovery: str | None, from_controller: bool
) -> FaultOutcome:
    if recovery is not None:
        return "recovered"
    if stop is None or event_seq not in stop.event_seqs:
        return "unresolved"
    expected_cause = "controller_error" if from_controller else "all_candidates_failed"
    return "stopped_safely" if stop.cause == expected_cause else "unresolved"


def _faults(records: _Records, stop: SafeStop | None) -> list[FaultEvidence]:
    final = records.final_completion()
    found = []
    for event in records.events:
        if isinstance(event, ToolResultEvent) and event.payload.status is ToolResultStatus.ERROR:
            payload = event.payload
            completed_later = final is not None and final[1] > event.seq
            recovery = _recovered_tool(records, event, final) if final and completed_later else None
            found.append(
                FaultEvidence(
                    source="tool_result",
                    event_seq=event.seq,
                    code=payload.error.code if payload.error else "TOOL_FAILED",
                    tool=payload.tool,
                    call_id=payload.call_id,
                    outcome=_fault_outcome(
                        event.seq, stop, recovery=recovery, from_controller=False
                    ),
                    recovery_job_id=recovery,
                )
            )
        elif isinstance(event, JobTransitionEvent) and event.payload.to_state in FAILED_STATES:
            completed_later = final is not None and final[1] > event.seq
            recovery = _recovered_job(records, event, final) if final and completed_later else None
            match = _CODE.search(event.payload.reason)
            found.append(
                FaultEvidence(
                    source="job_transition",
                    event_seq=event.seq,
                    code=match[1] if match else UNCODED_JOB_FAILURE,
                    job_id=event.payload.job_id,
                    outcome=_fault_outcome(
                        event.seq, stop, recovery=recovery, from_controller=False
                    ),
                    recovery_job_id=recovery,
                )
            )
        elif isinstance(event, ErrorEvent) and event.payload.component == CONTROLLER_COMPONENT:
            found.append(
                FaultEvidence(
                    source="controller_error",
                    event_seq=event.seq,
                    code=event.payload.code,
                    outcome=_fault_outcome(event.seq, stop, recovery=None, from_controller=True),
                )
            )
    return found


def derive_outcome_evidence(database: Database, runs_root: Path, run_id: str) -> OutcomeEvidence:
    """Compute the run's decision, stop, and fault evidence from its durable records alone."""
    records = _Records(database, runs_root, run_id)
    stop = _safe_stop(records)
    return OutcomeEvidence(decision=_decision(records), stop=stop, faults=_faults(records, stop))


__all__ = [
    "CANDIDATE_TOOLS",
    "DATASET_UNREADABLE",
    "MODEL_PRODUCING_TOOL",
    "DecisionEvidence",
    "FaultEvidence",
    "OutcomeEvidence",
    "SafeStop",
    "derive_outcome_evidence",
    "has_authorized_model_route",
]
