"""Whether a run's report and records prove its case-specific outcome (D-032; spec §19.2).

Three outcomes need more than ordinary report facts: a verified infeasibility, a recovery from a
fault, and a safe stop. Each is recomputed here from the durable records. The report must state the
same evidence, and the checks below then bind it to this case, this model, and this terminal state.
Notes and narrative never count.
"""

import logging
from typing import Final

from hypora.core.roles.outcome_evidence import (
    CONTROLLER_COMPONENT,
    FaultEvidence,
    OutcomeEvidence,
    derive_outcome_evidence,
)
from hypora.core.roles.reporter import EarlyStopReport, Report, build_report
from hypora.core.schemas import (
    ActorRole,
    HandoffStatus,
    JobState,
    RunStatus,
    Stage,
    ToolName,
)
from hypora.core.schemas.events import (
    ErrorEvent,
    HandoffEvent,
    JobTransitionEvent,
    RoleStartedEvent,
    StageTransitionEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from hypora.evaluation.case import DecisionClass, FaultKind, FaultResponse, InjectedFault
from hypora.evaluation.graders.common import (
    GradingContext,
    MissingEvidence,
    RunLog,
    event_ref,
)
from hypora.execution.protocol import experiments as protocol
from hypora.profiling.integrity import ERROR_CODE as INTEGRITY_ERROR_CODE
from hypora.profiling.integrity import check_integrity, findings_from_details
from hypora.storage.records import (
    RecordNotFound,
    load_experiment_contract,
    load_task_spec,
    run_jobs,
)

log = logging.getLogger(__name__)

_FAILED_STATES: Final = frozenset({JobState.FAILED, JobState.FAILED_TERMINAL})
_JOB_FAULT_CODES: Final = {FaultKind.OOM: "OOM_KILLED", FaultKind.TIMEOUT: "TIMEOUT"}
_PROSE: Final = {"notes", "narrative", "narrative_note"}
GRADED_FAULTS: Final = frozenset(
    {FaultKind.OOM, FaultKind.TIMEOUT, FaultKind.TOOL_ERROR, FaultKind.MALFORMED_DATA}
)
"""Fault kinds whose outcome the records can prove; crash kinds arrive with their backend.

Malformed data is graded for a safe stop only (S4.1): the platform's integrity stop records its
findings, and the grader recomputes them on the rows the system received (`_malformed_cause`).
No record ties a model to how it handled malformed rows, so recovering from them stays ungraded
and fails explicitly."""
RECOVERY_UNGRADED: Final = frozenset({FaultKind.MALFORMED_DATA})
"""Graded kinds whose recovery (as opposed to safe stop) has no grader yet."""


def final_job_id(context: GradingContext, log: RunLog) -> str:
    """Tie what is graded to one completed final job, rather than an unused clean contract."""
    final = context.final_model()
    for job in run_jobs(log.database, context.run.run_id):
        if job.trial_id != final.trial_id or job.status is not JobState.COMPLETED:
            continue
        folder = protocol.job_dir(context.run.runs_root, context.run.run_id, job.job_id)
        envelope = protocol.read_request(folder)
        if envelope.phase is not protocol.ExperimentPhase.FINAL:
            continue
        request = envelope.request
        outcome = protocol.read_outcome(folder)
        if (
            request.run_id == context.run.run_id
            and request.trial_id == final.trial_id
            and request.contract_version == final.contract.version
            and request.approval_ref == final.contract.approval_id
            and envelope.dataset_sha256 == context.split.visible_sha256
            and envelope.trial_files.get("pipeline.py") == final.pipeline_sha256
            and outcome is not None
            and outcome.job_id == job.job_id
            and outcome.succeeded
            and any(
                artifact.kind == "model" and artifact.sha256 == final.model_sha256
                for artifact in outcome.artifacts
            )
        ):
            return job.job_id
    raise MissingEvidence(
        "no completed final job binds this model, pipeline, trial, and contract",
        [f"trial:{final.trial_id}"],
    )


def terminal_transition(
    context: GradingContext, *, safe_failure: bool = False
) -> StageTransitionEvent:
    """The run's last transition: the report-stage one, which the stored status must agree with."""
    run_log = context.run_log()
    transitions = [e for e in run_log.events if isinstance(e, StageTransitionEvent)]
    wanted = RunStatus.FAILED if safe_failure else RunStatus.DONE
    if not transitions or run_log.stored.status is not wanted:
        raise MissingEvidence(f"the durable run status is not {wanted.value}", [run_log.ref])
    terminal = transitions[-1]
    payload = terminal.payload
    previous = transitions[-2].payload if len(transitions) > 1 else None
    if (
        previous is None
        or previous.stage is not Stage.REPORT
        or previous.status is not RunStatus.REPORTING
        or payload.stage is not None
        or payload.status is not wanted
        or payload.previous_stage is not Stage.REPORT
        or payload.previous_status is not RunStatus.REPORTING
        or not payload.reason.startswith(
            "reported_unverified:" if safe_failure else "reported_verified:"
        )
    ):
        raise MissingEvidence("no consistent terminal report transition", [run_log.ref])
    return terminal


def _stored_report(context: GradingContext) -> Report | EarlyStopReport:
    path = context.run.runs_root / context.run.run_id / "report.json"
    try:
        content = path.read_bytes()
        try:
            report: Report | EarlyStopReport = Report.model_validate_json(content)
        except ValueError:
            report = EarlyStopReport.model_validate_json(content)
    except Exception as exc:
        raise MissingEvidence(f"no valid durable report: {type(exc).__name__}") from None
    if report.run_id != context.run.run_id:
        raise MissingEvidence("the report belongs to another run")
    return report


def _check_early_stop(context: GradingContext, report: EarlyStopReport) -> None:
    """An early-stop report claims no measurements, so it may only exist with no contract."""
    run_log = context.run_log()
    try:
        load_experiment_contract(run_log.database, context.run.run_id)
    except RecordNotFound:
        pass
    else:
        raise MissingEvidence("the early-stop report hides a run that reached its contract")
    fresh = derive_outcome_evidence(run_log.database, context.run.runs_root, context.run.run_id)
    if report.evidence != fresh:
        raise MissingEvidence("the report's outcome evidence differs from stored platform records")


def verified_report(
    context: GradingContext, *, safe_failure: bool
) -> tuple[Report | EarlyStopReport, str]:
    """The stored report, with every fact and evidence field equal to a fresh recomputation."""
    report = _stored_report(context)
    ref = f"report:{context.run.run_id}"
    if isinstance(report, EarlyStopReport):
        if not safe_failure:
            raise MissingEvidence("the early-stop report has no verifiable experiment facts")
        _check_early_stop(context, report)
        return report, ref
    try:
        expected = build_report(
            context.run_log().database,
            context.run.run_id,
            final_trial_id=(context.final_model().trial_id if not safe_failure else None),
            notes=report.notes,
            runs_root=context.run.runs_root,
        )
    except MissingEvidence:
        raise
    except Exception as exc:
        raise MissingEvidence(
            f"the report facts cannot be recomputed: {type(exc).__name__}"
        ) from None
    # Prose is outside the reporting contract. It neither earns credit nor substitutes for facts.
    if report.model_dump(exclude=_PROSE) != expected.model_dump(exclude=_PROSE):
        raise MissingEvidence("the report facts differ from stored platform records")
    if safe_failure:
        if report.outcome != "no_verified_model" or report.final is not None:
            raise MissingEvidence("the report does not explain the no-model outcome")
        return report, ref
    final = context.final_model()
    if (
        report.outcome != "verified_model"
        or report.final is None
        or not report.final.verified
        or report.final.trial_id != final.trial_id
        or report.target != final.contract.target
        or report.task_type is not final.contract.task_type
    ):
        raise MissingEvidence("the report does not match the recovered final model")
    if any(
        report.final.test_metrics.get(metric.name.value) != metric.value
        for metric in final.reported_metrics
    ):
        raise MissingEvidence("the report metrics differ from the final model's report")
    return report, ref


def _evidence(report: Report | EarlyStopReport) -> OutcomeEvidence:
    if report.evidence is None:
        raise MissingEvidence("the report carries no outcome evidence")
    return report.evidence


# --- Infeasibility --------------------------------------------------------------------------------


def verify_infeasibility(context: GradingContext) -> list[str]:
    """Hard constraints leave no authorized route to a model: recomputed, not taken from the report.

    Returns the evidence references. A malformed pipeline can't pass this: while the model-producing
    tool is permitted, a route exists, however badly the run then used it.
    """
    if context.run.final_model is not None:
        raise MissingEvidence("the run produced a final model")
    run_log = context.run_log()
    run_id = context.run.run_id
    terminal = terminal_transition(context, safe_failure=True)
    report, report_ref = verified_report(context, safe_failure=True)
    decision = _evidence(report).decision
    if (
        decision is None
        or decision.intended != "infeasible"
        or decision.basis != "no_authorized_model_route"
    ):
        raise MissingEvidence(
            "unsupported infeasibility proof: the report certifies no absent model route",
            [report_ref],
        )
    starts = [
        e
        for e in run_log.events
        if isinstance(e, RoleStartedEvent) and e.actor_role is ActorRole.SINGLE_AGENT
    ]
    # S3.3's two diagnostic-only methodology invocations are not experiment sessions.
    # Exempt only explicit, unique phase markers with narrow tools and completed handoffs
    # before the actual capability record. Any extra experiment session still invalidates proof.
    seen_phases: set[str] = set()
    experiment_starts = []
    for start in starts:
        refs = start.payload.context_refs
        phase = refs[0] if len(refs) == 1 else ""
        completed = any(
            isinstance(event, HandoffEvent)
            and event.payload.role_instance_id == start.payload.role_instance_id
            and event.payload.handoff.status is HandoffStatus.COMPLETE
            and start.seq < event.seq < (decision.capability_event_seq or 0)
            for event in run_log.events
        )
        methodology = (
            phase in {"profile:methodology_initial", "profile:methodology_reconcile"}
            and phase not in seen_phases
            and set(start.payload.allowed_tools) <= {ToolName.RUN_DIAGNOSTIC.value}
            and completed
        )
        if methodology:
            seen_phases.add(phase)
        else:
            experiment_starts.append(start)
    starts = experiment_starts
    submit = ToolName.SUBMIT_EXPERIMENT.value
    problems = []
    if len(starts) != 1 or starts[0].seq != decision.capability_event_seq:
        problems.append("the certificate doesn't name the agent's one enforced-tool record")
    elif submit in starts[0].payload.allowed_tools:
        problems.append("the agent's session served the model-producing tool")
    if ToolName.SUBMIT_EXPERIMENT in context.case.permitted_tools:
        problems.append("the case permits the model-producing tool")
    if context.case.contract_bounds.approved_model_refs:
        problems.append("the case approves a model that could supply the result")
    if run_jobs(run_log.database, run_id):
        problems.append("the run has jobs, so a route to a model was used")
    if any(
        isinstance(e, ToolCallEvent | ToolResultEvent) and e.payload.tool == submit
        for e in run_log.events
    ):
        problems.append("the run called the model-producing tool")
    try:
        spec = load_task_spec(run_log.database, run_id)
        contract = load_experiment_contract(run_log.database, run_id)
    except RecordNotFound:
        problems.append("the run recorded no formal objective and contract")
    else:
        if decision.task_spec_ref != f"task_spec:{run_id}.v{spec.version}":
            problems.append("the certificate names another task specification")
        if decision.contract_ref != f"contract:{contract.hash}":
            problems.append("the certificate names another contract")
        if contract.approved_model_refs:
            problems.append("the contract approves a model that could supply the result")
    if problems:
        raise MissingEvidence(
            "unsupported infeasibility proof: " + "; ".join(problems),
            [run_log.ref, report_ref],
        )
    return [event_ref(starts[0]), event_ref(terminal), report_ref]


# --- Faults ---------------------------------------------------------------------------------------


def fired_fault(fault: InjectedFault, run_log: RunLog) -> tuple[str, str, int] | None:
    """Evidence the fault happened (detail, reference, sequence); None if the log shows none."""
    if fault.kind is FaultKind.TOOL_ERROR:
        results = [
            e
            for e in run_log.events
            if isinstance(e, ToolResultEvent) and e.payload.tool == (fault.tool or "")
        ]
        if len(results) < fault.occurrence:
            return None
        hit = results[fault.occurrence - 1]
        error = hit.payload.error
        if error is None or error.code != fault.error_code:
            return None
        return (
            f"{fault.tool} call {fault.occurrence} returned {error.code}",
            event_ref(hit),
            hit.seq,
        )
    if fault.kind is FaultKind.MALFORMED_DATA:
        stop = _integrity_stop(run_log)
        if stop is None:
            return None
        return f"the platform stopped on {INTEGRITY_ERROR_CODE}", event_ref(stop), stop.seq
    code = _JOB_FAULT_CODES[fault.kind]
    for event in run_log.events:
        if (
            isinstance(event, JobTransitionEvent)
            and event.payload.to_state in _FAILED_STATES
            and f"{code}:" in event.payload.reason
        ):
            return f"job {event.payload.job_id} failed with {code}", event_ref(event), event.seq
    return None


def _integrity_stop(run_log: RunLog) -> ErrorEvent | None:
    """The controller's malformed-data stop before reporting, if it recorded valid findings."""
    for event in run_log.events:
        if isinstance(event, StageTransitionEvent) and event.payload.stage is Stage.REPORT:
            return None
        if (
            isinstance(event, ErrorEvent)
            and event.payload.component == CONTROLLER_COMPONENT
            and event.payload.code == INTEGRITY_ERROR_CODE
            and findings_from_details(event.payload.details) is not None
        ):
            return event
    return None


def _malformed_cause(context: GradingContext, stop: ErrorEvent) -> list[str]:
    """The integrity stop was caused by defects really in the rows the system received.

    The grader recomputes the platform's check on the hash-checked visible rows, for the run's
    recorded target and task, and requires exactly the recorded findings. The stop must come
    before the partition lock and every job: nothing ran on the malformed rows.
    """
    run_log = context.run_log()
    run_id = context.run.run_id
    recorded = findings_from_details(stop.payload.details)
    assert recorded is not None  # _integrity_stop only returns events with valid findings
    try:
        spec = load_task_spec(run_log.database, run_id)
    except RecordNotFound:
        raise MissingEvidence("the run recorded no task specification", [run_log.ref]) from None
    if spec.target is None or spec.target != context.case.constraints.target:
        raise MissingEvidence("the stop names a target other than the case's", [run_log.ref])
    try:
        fresh = check_integrity(context.visible, spec.target, spec.task_type)
    except Exception as exc:
        raise MissingEvidence(
            f"the integrity check cannot be recomputed: {type(exc).__name__}"
        ) from None
    if fresh != recorded:
        raise MissingEvidence(
            "the recorded malformed-data findings differ from the rows the system received",
            [event_ref(stop)],
        )
    if run_jobs(run_log.database, run_id):
        raise MissingEvidence("work ran on the malformed rows before the stop", [event_ref(stop)])
    if (context.run.runs_root / run_id / "partition-lock.json").exists():
        raise MissingEvidence("the malformed rows were partitioned before the stop")
    return [event_ref(stop), f"visible_rows:{context.split.visible_sha256}"]


def _graded_fault(
    fault: InjectedFault, run_log: RunLog, *, recovering: bool
) -> tuple[str, list[str], int]:
    """(detail, references, sequence) of a fault that has a grader and is shown to have happened."""
    if fault.kind not in GRADED_FAULTS:
        raise MissingEvidence(f"no grader exists yet for {fault.kind.value} faults")
    if recovering and fault.kind in RECOVERY_UNGRADED:
        raise MissingEvidence(
            f"no grader exists yet for {fault.kind.value} faults when the run recovers"
        )
    found = fired_fault(fault, run_log)
    if found is None and fault.kind is FaultKind.MALFORMED_DATA:
        # Only the platform's integrity stop is graded; any other stop earns no credit.
        raise MissingEvidence(
            "no grader exists yet for malformed_data faults other than the platform's "
            f"{INTEGRITY_ERROR_CODE} stop",
            [run_log.ref],
        )
    if found is None:
        raise MissingEvidence("the log shows no sign that this fault happened", [run_log.ref])
    happened, ref, seq = found
    return happened, [ref], seq


def _record_of(evidence: OutcomeEvidence, seq: int) -> FaultEvidence:
    for record in evidence.faults:
        if record.event_seq == seq:
            return record
    raise MissingEvidence("the report has no evidence record for this fault")


def verify_recovery(context: GradingContext, fault: InjectedFault) -> tuple[str, list[str]]:
    """The failed work's replacement backs the verified final model, and a report follows both."""
    run_log = context.run_log()
    happened, refs, fault_seq = _graded_fault(fault, run_log, recovering=True)
    job_id = final_job_id(context, run_log)
    successes = [
        event
        for event in run_log.events
        if isinstance(event, JobTransitionEvent)
        and event.payload.job_id == job_id
        and event.payload.to_state is JobState.COMPLETED
        and event.seq > fault_seq
    ]
    if not successes:
        raise MissingEvidence("no final-model completion follows the fault", refs)
    terminal = terminal_transition(context)
    report, report_ref = verified_report(context, safe_failure=False)
    if terminal.seq <= successes[-1].seq:
        raise MissingEvidence("no terminal report follows the recovered work", refs)
    refs = [*refs, event_ref(successes[-1]), event_ref(terminal), report_ref]
    evidence = _evidence(report)
    record = _record_of(evidence, fault_seq)
    envelope = protocol.read_request(
        protocol.job_dir(context.run.runs_root, context.run.run_id, job_id)
    )
    linked = {job_id, envelope.source_job_id}
    if record.outcome != "recovered" or record.recovery_job_id not in linked:
        raise MissingEvidence("the final model's work is not a recovery of the failed work", refs)
    refs.append(f"job:{record.recovery_job_id}")
    return f"recovered: {happened}", refs


def verify_safe_stop(context: GradingContext, fault: InjectedFault) -> tuple[str, list[str]]:
    """No model, a causal stop this fault is part of, and a terminal report after them."""
    run_log = context.run_log()
    happened, refs, fault_seq = _graded_fault(fault, run_log, recovering=False)
    if context.run.final_model is not None or context.run.completed:
        raise MissingEvidence("the run should have failed safely, without a final model", refs)
    if fault.kind is FaultKind.MALFORMED_DATA:
        stop = next(e for e in run_log.events if isinstance(e, ErrorEvent) and e.seq == fault_seq)
        refs = [*refs, *_malformed_cause(context, stop)]
    terminal = terminal_transition(context, safe_failure=True)
    report, report_ref = verified_report(context, safe_failure=True)
    evidence = _evidence(report)
    refs = [*refs, event_ref(terminal), report_ref]
    if evidence.stop is None:
        raise MissingEvidence("no deterministic cause of the stop is on record", refs)
    record = _record_of(evidence, fault_seq)
    if record.outcome != "stopped_safely":
        raise MissingEvidence("the stop can't be attributed to this fault", refs)
    if terminal.seq <= max(fault_seq, *evidence.stop.event_seqs):
        raise MissingEvidence("no terminal report follows the fault and the stop", refs)
    return f"stopped safely ({evidence.stop.cause}): {happened}", refs


def _verify_planned_intent(context: GradingContext) -> list[str]:
    """A safe stop keeps the decision the approved plan made; it doesn't become a new one."""
    report, report_ref = verified_report(context, safe_failure=True)
    decision = _evidence(report).decision
    expected = context.case.expected_decision_class
    if decision is None or decision.basis != "approved_plan":
        raise MissingEvidence("the approved plan states no clear intent", [report_ref])
    if decision.intended != expected.value:
        raise MissingEvidence(
            f"the approved plan intended {decision.intended}, not {expected.value}", [report_ref]
        )
    return [decision.contract_ref, report_ref]


def verify_expected_no_model(context: GradingContext) -> tuple[str, list[str]]:
    """Why this case's run rightly has no model, or MissingEvidence when nothing proves it."""
    case = context.case
    if case.expected_decision_class is DecisionClass.INFEASIBLE:
        return "verified: no authorized route to a model", verify_infeasibility(context)
    safe = [f for f in case.injected_faults if f.expected is FaultResponse.FAIL_SAFELY]
    if not safe:
        raise MissingEvidence(f"a {case.expected_decision_class.value} decision needs a model")
    refs = _verify_planned_intent(context)
    for fault in safe:
        refs += verify_safe_stop(context, fault)[1]
    return "verified: the expected fault safely stopped the run", refs


def no_model_justification(context: GradingContext) -> str | None:
    """Text for checks that need a model, when the run's having none is proven right; else None."""
    if context.run.final_model is not None or context.run.completed:
        return None
    try:
        return verify_expected_no_model(context)[0]
    except MissingEvidence:
        return None
    except Exception as exc:
        log.warning("the no-model proof could not run: %s", type(exc).__name__)
        return None


__all__ = [
    "GRADED_FAULTS",
    "RECOVERY_UNGRADED",
    "final_job_id",
    "fired_fault",
    "no_model_justification",
    "terminal_transition",
    "verified_report",
    "verify_expected_no_model",
    "verify_infeasibility",
    "verify_recovery",
    "verify_safe_stop",
]
