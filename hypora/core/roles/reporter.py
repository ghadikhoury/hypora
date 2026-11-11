"""Reporter v1 (spec §5.2 Report; PLAN.md S1.5; D-019): the run report, from DB records only.

Every number in the report comes from stored, platform-computed records: the task spec, the
contract, the profile, each trial's validation results, and the final candidate's test results
and verification verdict. `build_report` assembles those facts deterministically.

The Reporter LLM adds a short narrative (`write_narrative`) but may not write numbers: it refers
to the final model's scores only through placeholders (`{{test.roc_auc}}`,
`{{validation.roc_auc}}`), which `render_markdown` fills from the stored metrics. A validator
rejects any other digit, and the adapter retries once with the errors; a narrative that still
fails is left out, and the report says so.

Result integrity (S3.5, spec §8.6, §14): the verdict and the honesty fields are code-populated.
`verification_status`, `failed_checks` and `skipped_checks` come only from the stored verification
of a completed final job; `limitations`, `resources` and `approval` come from the event log (before
the run enters REPORT, like the outcome evidence), the job folders and the `runs` row. No model
output has a field that can set them, and the agent's narrative is rendered below them. So a later
recomputation (the grader's) gives the same facts as the ones written into the report.
"""

import contextlib
import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message
from hypora.core.roles.base import CallGuard, RunLimitReached
from hypora.core.roles.outcome_evidence import OutcomeEvidence, derive_outcome_evidence
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import (
    AUTO_APPROVER,
    ActorRole,
    AnyEvent,
    ApprovalDecision,
    ApprovalPolicy,
    HandoffStatus,
    JobState,
    MetricDirection,
    MetricName,
    ModelTier,
    Stage,
    StrictModel,
    TaskType,
)
from hypora.core.schemas.common import NonEmptyStr
from hypora.core.schemas.events import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ErrorEvent,
    HandoffEvent,
    JobTransitionEvent,
    ModelCallEvent,
    StageTransitionEvent,
    ToolCallEvent,
)
from hypora.execution.protocol import experiments as protocol
from hypora.profiling.tabular import leakage_screen_summary
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import (
    RecordNotFound,
    list_trials,
    load_dataset_profile,
    load_experiment_contract,
    load_result,
    load_run,
    load_task_spec,
    run_jobs,
)

PLACEHOLDER: Final = re.compile(r"\{\{(test|validation)\.([a-z0-9_]+)\}\}")
_METRIC_WORDS: Final = re.compile(
    r"\b(" + "|".join(sorted((m.value for m in MetricName), key=len, reverse=True)) + r")\b"
)
_KNOWN_METRICS: Final = frozenset(m.value for m in MetricName)

APPROVAL_DEFENSE_OFF_APPROVER: Final = "policy:defense_off"
"""The approver the plan gate records when `defenses.approval` is off (F6 ablation, spec §19.4)."""
USER_APPROVER_PREFIX: Final = "user:"
VERDICT_LINE: Final = "The agent's text below does not change this verdict."
LEARNED_TRANSFORM_LIMITATION: Final = (
    "The learned-transform leakage check is not implemented: transforms fitted inside a pipeline "
    "were not checked for leakage from the evaluation folds or the test split."
)
NOT_RUN_LIMITATION: Final = (
    "No final verification ran, so no model is verified; the causes the platform recorded are "
    "listed below and under decision and faults."
)
NOT_SCREENED_LIMITATION: Final = (
    "Leakage was not screened: the run has no tabular profile to screen features against."
)
TEST_ONCE_LIMITATION: Final = (
    "The test split was scored once, for the final candidate only; every other trial has "
    "validation scores only."
)
FAILED_CHECK_STATUSES: Final = frozenset({"failed", "error"})
FAILED_JOB_STATES: Final = frozenset(
    {JobState.CANCELLED, JobState.FAILED, JobState.FAILED_TERMINAL}
)
MAX_LISTED_JOBS: Final = 10
MAX_REASON_CHARS: Final = 300

VerificationStatus = Literal["passed", "failed", "not_run"]


class ReportNarrative(StrictModel):
    """The Reporter LLM's output: prose with metric placeholders and no digits of its own."""

    summary: NonEmptyStr = Field(max_length=1500)
    observations: list[str] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _no_numbers_of_its_own(self) -> "ReportNarrative":
        for text in (self.summary, *self.observations, *self.limitations):
            for _, metric in PLACEHOLDER.findall(text):
                if metric not in _KNOWN_METRICS:
                    raise ValueError(f"unknown metric in placeholder: {metric!r}")
            bare = _METRIC_WORDS.sub("", PLACEHOLDER.sub("", text))
            if re.search(r"\d", bare):
                raise ValueError(
                    "the narrative must not contain digits: refer to scores only with "
                    "placeholders such as {{test.roc_auc}} and to trials by description"
                )
        return self


class MetricSummary(StrictModel):
    mean: float | None
    std: float | None
    folds: int


class TrialSummary(StrictModel):
    trial_id: str
    kind: str
    hypothesis: str
    status: str
    validation: MetricSummary | None = None  # of the primary metric


class CheckLine(StrictModel):
    check: str
    status: str
    message: str


class FinalModel(StrictModel):
    trial_id: str
    hypothesis: str
    test_metrics: dict[str, float | None]
    validation_means: dict[str, float | None]
    verified: bool
    checks: list[CheckLine]


class ReportApproval(StrictModel):
    """Who decided on the run's plan, from the recorded approval event (spec §10)."""

    approval_id: str
    contract_version: int | None = None  # None in an early-stop report: no contract exists
    artifact_hash: str
    approver: str
    policy: ApprovalPolicy
    decision: ApprovalDecision
    reason: str | None = None


class ResourceUse(StrictModel):
    """What the run used up to the start of reporting (the report's own narrative is excluded).

    Model and tool counts, tokens and cost come from the event log, sandbox compute from the job
    folders' stage logs (what the ledger charges), budgets from the `runs` row.
    """

    usd: float
    llm_tokens: int
    model_calls: int
    tool_calls: int
    compute_seconds: float
    wall_clock_seconds: float
    trials: int
    jobs: int
    budget_usd: float | None = None
    budget_compute_seconds: float | None = None


def _split_checks(checks: Sequence[CheckLine]) -> tuple[list[CheckLine], list[CheckLine]]:
    """The failed (or errored) and the skipped checks, in the verifier's order."""
    failed = [c for c in checks if c.status in FAILED_CHECK_STATUSES]
    skipped = [c for c in checks if c.status == "skipped"]
    return failed, skipped


def _verification_status(final: "FinalModel | None") -> VerificationStatus:
    if final is None:
        return "not_run"
    return "passed" if final.verified else "failed"


class Report(StrictModel):
    """The run report's facts. Numbers only from stored records; text from the controller.

    `outcome`, `verification_status`, `failed_checks` and `skipped_checks` follow the stored
    verification of the final job and nothing else; the validator rejects any other combination.
    """

    run_id: str
    outcome: Literal["verified_model", "no_verified_model"]
    objective: str
    target: str
    task_type: TaskType
    primary_metric: MetricName
    direction: MetricDirection
    secondary_metrics: list[MetricName]
    split: str
    features: int
    excluded_features: list[str]
    leakage_screen: str
    trials: list[TrialSummary]
    final: FinalModel | None = None
    notes: list[str] = Field(default_factory=list)
    narrative: ReportNarrative | None = None
    narrative_note: str | None = None
    sources: list[str] = Field(default_factory=list)
    evidence: OutcomeEvidence | None = None
    verification_status: VerificationStatus = "not_run"
    failed_checks: list[CheckLine] = Field(default_factory=list)
    skipped_checks: list[CheckLine] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    resources: ResourceUse | None = None
    approval: ReportApproval | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_verification(cls, data: Any) -> Any:
        """A report written before S3.5 has no verification fields: derive them from `final`."""
        if not isinstance(data, dict) or "verification_status" in data:
            return data
        raw = data.get("final")
        final = None if raw is None else FinalModel.model_validate(raw)
        failed, skipped = _split_checks(final.checks if final else [])
        return {
            **data,
            "verification_status": _verification_status(final),
            "failed_checks": failed,
            "skipped_checks": skipped,
        }

    @model_validator(mode="after")
    def _verdict_follows_the_stored_verification(self) -> "Report":
        if self.verification_status != _verification_status(self.final):
            raise ValueError("verification_status must follow the final model's stored verdict")
        if (self.outcome == "verified_model") != (self.verification_status == "passed"):
            raise ValueError("only a passed verification yields a verified model")
        failed, skipped = _split_checks(self.final.checks if self.final else [])
        if self.failed_checks != failed or self.skipped_checks != skipped:
            raise ValueError("failed_checks and skipped_checks must list the stored checks")
        return self


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _fold_values(scores: dict[str, Any], metric: MetricName) -> list[float]:
    values = [
        _finite(fold.get("metrics", {}).get(metric.value)) for fold in scores.get("folds", [])
    ]
    return [v for v in values if v is not None]


def metric_summary(scores: dict[str, Any], metric: MetricName) -> MetricSummary:
    """Mean and spread of `metric` over the folds of stored validation scores (metrics.json)."""
    values = _fold_values(scores, metric)
    return MetricSummary(
        mean=statistics.fmean(values) if values else None,
        std=statistics.pstdev(values) if values else None,
        folds=len(values),
    )


@dataclass(frozen=True)
class _Timeline:
    """The run's events before it entered REPORT (all of them while it has not).

    The cut-off makes the derived fields stable: the Reporter's own model call, the transition into
    REPORT and every later event can't change them, so a later recomputation (the grader's) matches
    the stored report whether the facts were built just before or just after that transition.
    """

    before: list[AnyEvent]

    @classmethod
    def read(cls, database: Database, run_id: str) -> "_Timeline":
        events = EventLog(database).read(run_id)
        for index, event in enumerate(events):
            if isinstance(event, StageTransitionEvent) and event.payload.stage is Stage.REPORT:
                return cls(list(events[:index]))
        return cls(list(events))

    @property
    def wall_clock_seconds(self) -> float:
        """Seconds from the run's first event to its last event before reporting."""
        if not self.before:
            return 0.0
        elapsed: timedelta = self.before[-1].created_at - self.before[0].created_at
        return round(elapsed.total_seconds(), 3)


def _cut(text: str | None) -> str | None:
    return None if text is None else text[:MAX_REASON_CHARS]


def _approval(timeline: _Timeline, approval_id: str | None) -> ReportApproval | None:
    """The last recorded decision on `approval_id` (or on any plan, when None)."""
    decision = None
    for event in timeline.before:
        if isinstance(event, ApprovalGrantedEvent | ApprovalDeniedEvent) and (
            approval_id is None or event.payload.approval_id == approval_id
        ):
            decision = event
    if decision is None:
        return None
    payload = decision.payload
    reason = decision.payload.reason if isinstance(decision, ApprovalDeniedEvent) else None
    return ReportApproval(
        approval_id=payload.approval_id,
        artifact_hash=payload.artifact_hash,
        approver=payload.approver,
        policy=payload.policy,
        decision=(
            ApprovalDecision.DENIED
            if isinstance(decision, ApprovalDeniedEvent)
            else ApprovalDecision.APPROVED
        ),
        reason=_cut(reason),
    )


def _resources(
    database: Database, runs_root: Path, run_id: str, timeline: _Timeline
) -> ResourceUse:
    """Resource use up to the start of reporting, from stored records only."""
    calls = [e.payload for e in timeline.before if isinstance(e, ModelCallEvent)]
    tools = sum(isinstance(e, ToolCallEvent) for e in timeline.before)
    jobs = run_jobs(database, run_id)
    compute = sum(
        stage.duration_s
        for job in jobs
        for stage in protocol.read_stage_logs(protocol.job_dir(runs_root, run_id, job.job_id))
    )
    run = load_run(database, run_id)
    return ResourceUse(
        usd=round(sum(c.cost_usd for c in calls), 8),
        llm_tokens=sum(c.input_tokens + c.output_tokens for c in calls),
        model_calls=len(calls),
        tool_calls=tools,
        compute_seconds=round(compute, 6),
        wall_clock_seconds=timeline.wall_clock_seconds,
        trials=len(list_trials(database, run_id)),
        jobs=len(jobs),
        budget_usd=None if run.budget_usd is None else float(run.budget_usd),
        budget_compute_seconds=(
            None if run.budget_compute_s is None else float(run.budget_compute_s)
        ),
    )


def _approval_limitation(approval: ReportApproval | None, *, expected: bool) -> list[str]:
    if approval is None:
        return ["No approval decision is recorded for this run's plan."] if expected else []
    if approval.decision is ApprovalDecision.DENIED:
        reason = (approval.reason or "no reason given").rstrip(".")
        return [f"The plan was denied by {approval.approver}: {reason}."]
    if approval.approver == APPROVAL_DEFENSE_OFF_APPROVER:
        return [
            "The plan was approved with the approval defense switched off "
            "(defenses.approval=false): no bounds check and no human decision ran."
        ]
    if approval.approver.startswith(USER_APPROVER_PREFIX):
        return [f"The plan was approved by the user ({approval.approver})."]
    if approval.approver == AUTO_APPROVER:
        return [
            "The plan was approved by policy (auto_within_contract) within the request's bounds, "
            "without a human decision."
        ]
    return [f"The plan was approved by {approval.approver}, without a human decision."]


def _limitations(
    timeline: _Timeline,
    *,
    verification_status: VerificationStatus,
    screened: bool,
    approval: ReportApproval | None,
    approval_expected: bool,
) -> list[str]:
    """The report's mandatory limitations, written by code in a fixed order (design §2.4)."""
    lines = []
    if verification_status == "not_run":
        lines.append(NOT_RUN_LIMITATION)
    lines.append(LEARNED_TRANSFORM_LIMITATION)
    if not screened:
        lines.append(NOT_SCREENED_LIMITATION)
    job_states: dict[str, JobState] = {}
    for event in timeline.before:
        if isinstance(event, ErrorEvent) and event.payload.code == "RUN_LIMIT_REACHED":
            limit = event.payload.details.get("limit", "unknown")
            message = event.payload.message[:MAX_REASON_CHARS].rstrip(".")
            lines.append(f"The run limit {limit} fired: {message}.")
        elif isinstance(event, ErrorEvent) and event.payload.code == "RUN_CANCELLED":
            lines.append("The user cancelled the run.")
        elif isinstance(event, JobTransitionEvent):
            job_states[event.payload.job_id] = event.payload.to_state
    ended = [(job, state) for job, state in job_states.items() if state in FAILED_JOB_STATES]
    lines += [f"Job {job} ended {state.value}." for job, state in ended[:MAX_LISTED_JOBS]]
    if len(ended) > MAX_LISTED_JOBS:
        lines.append(f"{len(ended) - MAX_LISTED_JOBS} more jobs ended cancelled or failed.")
    for event in timeline.before:
        if not isinstance(event, HandoffEvent):
            continue
        handoff = event.payload.handoff
        if handoff.status is HandoffStatus.COMPLETE:
            continue
        if handoff.from_role is ActorRole.AGENT_B:
            lines.append(
                f"Research was unavailable or incomplete (handoff {handoff.status.value})."
            )
        else:
            lines.append(f"The {handoff.from_role.value} role handed off {handoff.status.value}.")
    lines += _approval_limitation(approval, expected=approval_expected)
    if verification_status != "not_run":
        lines.append(TEST_ONCE_LIMITATION)
    return lines


def build_report(
    database: Database,
    run_id: str,
    *,
    final_trial_id: str | None,
    notes: list[str],
    runs_root: Path | None = None,
) -> Report:
    """Assemble the report's facts from the run's stored records (no model call).

    With `runs_root`, the report also carries the decision and fault evidence (D-032), recomputable
    from the event log, the job folders, and the approved plan, and the run's resource use.
    The verdict fields, limitations and approval are always derived from stored records (S3.5).
    """
    spec = load_task_spec(database, run_id)
    contract = load_experiment_contract(database, run_id)
    profile_id, profile = load_dataset_profile(database, run_id)
    metric = contract.primary_metric
    trials = []
    for trial in list_trials(database, run_id):
        result = load_result(database, trial.trial_id, "validation")
        trials.append(
            TrialSummary(
                trial_id=trial.trial_id,
                kind=trial.kind,
                hypothesis=trial.hypothesis[:300],
                status=trial.status,
                validation=metric_summary(result.metrics, metric) if result else None,
            )
        )
    final = None
    sources = [f"task_spec:{run_id}.v{spec.version}", f"profile:{profile_id}"]
    sources.append(f"experiment_contract:{run_id}.v{contract.version}")
    if final_trial_id is not None:
        test = load_result(database, final_trial_id, "test")
        validation = load_result(database, final_trial_id, "validation")
        if test is not None:
            wanted = [metric, *contract.secondary_metrics]
            checks = test.verification.get("checks", [])
            final = FinalModel(
                trial_id=final_trial_id,
                hypothesis=next((t.hypothesis for t in trials if t.trial_id == final_trial_id), ""),
                test_metrics={
                    m.value: _finite(test.metrics.get("metrics", {}).get(m.value)) for m in wanted
                },
                validation_means={
                    m.value: metric_summary(validation.metrics, m).mean if validation else None
                    for m in wanted
                },
                verified=bool(test.verification.get("passed")),
                checks=[
                    CheckLine(
                        check=str(c.get("check")),
                        status=str(c.get("status")),
                        message=str(c.get("message", ""))[:200],
                    )
                    for c in checks
                ],
            )
            sources.append(f"result:{final_trial_id}/test")
    data = contract.data_contract
    split = data.split
    tabular = profile.tabular
    timeline = _Timeline.read(database, run_id)
    status = _verification_status(final)
    failed, skipped = _split_checks(final.checks if final else [])
    approval = _approval(timeline, contract.approval_id)
    if approval is not None:
        approval = approval.model_copy(update={"contract_version": contract.version})
    return Report(
        run_id=run_id,
        outcome="verified_model" if status == "passed" else "no_verified_model",
        verification_status=status,
        failed_checks=failed,
        skipped_checks=skipped,
        limitations=_limitations(
            timeline,
            verification_status=status,
            screened=tabular is not None,
            approval=approval,
            approval_expected=True,
        ),
        resources=None if runs_root is None else _resources(database, runs_root, run_id, timeline),
        approval=approval,
        objective=spec.objective,
        target=contract.target,
        task_type=contract.task_type,
        primary_metric=metric,
        direction=metric.direction,
        secondary_metrics=list(contract.secondary_metrics),
        split=f"{split.strategy.value}, {split.test_fraction:.0%} test, {split.validation}",
        features=len(data.allowed_features),
        excluded_features=[f.name for f in data.excluded_features],
        leakage_screen=leakage_screen_summary(tabular) if tabular else "leakage not screened",
        trials=trials,
        final=final,
        notes=notes,
        sources=sources,
        evidence=None
        if runs_root is None
        else derive_outcome_evidence(database, runs_root, run_id),
    )


async def write_narrative(
    adapter: LLMAdapter,
    guard: CallGuard,
    report: Report,
    *,
    role: ActorRole = ActorRole.REPORTER,
) -> Report:
    """Add the Reporter LLM's narrative; on any failure the report keeps no narrative."""
    prompt = load_prompt("agent_a_report" if role is ActorRole.AGENT_A else "reporter")
    facts = report.model_dump(
        mode="json", exclude={"narrative", "narrative_note", "sources", "evidence"}
    )
    try:
        guard.before_model_call()
        response = await adapter.complete(
            [Message.system(prompt.text), Message.user(json.dumps(facts, indent=1))],
            tier=ModelTier.CHEAP,
            context=CallContext(run_id=report.run_id, actor_role=role, prompt_ref=prompt.ref),
            output_schema=ReportNarrative,
        )
        guard.after_model_call(response)
    except RunLimitReached as reached:
        return report.model_copy(update={"narrative_note": f"no narrative: {reached}"})
    except LLMError as exc:
        guard.after_model_error(exc)
        note = f"no narrative: the Reporter's output failed ({type(exc).__name__})"
        return report.model_copy(update={"narrative_note": note})
    narrative = response.parsed if isinstance(response.parsed, ReportNarrative) else None
    return report.model_copy(update={"narrative": narrative})


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def fill_placeholders(text: str, report: Report) -> str:
    """Replace {{test.<metric>}} / {{validation.<metric>}} with the final model's stored scores."""

    def value(match: re.Match[str]) -> str:
        if report.final is None:
            return "unavailable"
        table = report.final.test_metrics if match[1] == "test" else report.final.validation_means
        return _number(table.get(match[2]))

    return PLACEHOLDER.sub(value, text)


def render_markdown(report: Report) -> str:
    """The human-readable report. Every number is formatted from `report`'s stored values."""
    lines = [f"# Hypora run report: {report.run_id}", ""]
    outcome = "a verified model" if report.outcome == "verified_model" else "no verified model"
    lines += [f"**Outcome:** {outcome}.", ""]
    # Code-written verdict and limitations come first: the agent's text below can't hide them.
    lines += _render_verification(
        report.verification_status,
        report.failed_checks,
        report.skipped_checks,
        report.limitations,
        report.approval,
        narrated=report.narrative is not None,
    )
    lines += [
        "## Task",
        f"- Objective: {report.objective}",
        f"- Target: `{report.target}` ({report.task_type.value})",
        f"- Primary metric: `{report.primary_metric.value}` ({report.direction.value})",
        "",
    ]
    if report.narrative is not None:
        n = report.narrative
        lines += ["## Summary", fill_placeholders(n.summary, report), ""]
        if n.observations:
            lines += [
                "**Observations**",
                *(f"- {fill_placeholders(o, report)}" for o in n.observations),
            ]
            lines.append("")
        if n.limitations:
            lines += [
                "**Limitations**",
                *(f"- {fill_placeholders(x, report)}" for x in n.limitations),
            ]
            lines.append("")
        lines += [
            "_Written by the agent; every number above is filled in from stored metrics._",
            "",
        ]
    elif report.narrative_note:
        lines += [f"_{report.narrative_note}_", ""]
    lines += ["## Final model"]
    if report.final is None:
        lines += ["No final model: no trial completed validation and verification.", ""]
    else:
        final = report.final
        lines += [
            f"- Trial `{final.trial_id}`: {final.hypothesis}",
            f"- Verified: {'yes' if final.verified else 'NO'}",
            "",
            "| Metric | Test (once) | Validation (mean) |",
            "|---|---|---|",
        ]
        for name, test_value in final.test_metrics.items():
            lines.append(
                f"| {name} | {_number(test_value)} | {_number(final.validation_means.get(name))} |"
            )
        lines += ["", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c.check} | {c.status} | {c.message} |" for c in final.checks]
        lines.append("")
    lines += [
        "## Trials",
        f"| Trial | Kind | Status | Validation {report.primary_metric.value} "
        "(mean ± std, folds) | Hypothesis |",
        "|---|---|---|---|---|",
    ]
    for trial in report.trials:
        v = trial.validation
        score = (
            f"{_number(v.mean)} ± {_number(v.std)} ({v.folds})"
            if v is not None and v.folds
            else "unavailable"
        )
        lines.append(
            f"| `{trial.trial_id}` | {trial.kind} | {trial.status} | {score} | {trial.hypothesis} |"
        )
    lines += [
        "",
        "## Data",
        f"- Split: {report.split}",
        f"- Features used: {report.features}; "
        f"excluded: {', '.join(report.excluded_features) or 'none'}",
        f"- Leakage screen: {report.leakage_screen}",
        "",
    ]
    lines += _render_resources(report.resources)
    lines += _render_evidence(report.evidence)
    if report.notes:
        lines += ["## Run notes", *(f"- {note}" for note in report.notes), ""]
    lines += ["## Sources", *(f"- `{source}`" for source in report.sources), ""]
    return "\n".join(lines)


_STATUS_TEXT: Final = {
    "passed": "passed: every verifier check that ran passed",
    "failed": "FAILED: the final model is not verified",
    "not_run": "NOT RUN: no final verification ran",
}


def _render_verification(
    status: VerificationStatus,
    failed: Sequence[CheckLine],
    skipped: Sequence[CheckLine],
    limitations: Sequence[str],
    approval: ReportApproval | None,
    *,
    narrated: bool,
) -> list[str]:
    """The platform's verdict, failed and skipped checks, and mandatory limitations."""
    lines = ["## Verification and limitations", f"**Verification:** {_STATUS_TEXT[status]}."]
    if status != "passed":
        lines.append(VERDICT_LINE)
    lines.append("")
    if failed:
        lines += ["**Failed checks**", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c.check} | {c.status} | {c.message} |" for c in failed]
        lines.append("")
    if skipped:
        lines += ["**Skipped checks**", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c.check} | {c.status} | {c.message} |" for c in skipped]
        lines.append("")
    written = list(limitations)
    if not narrated:
        written.append("The report has no agent narrative.")
    if written:
        lines += ["**Limitations (written by the platform)**", *(f"- {x}" for x in written), ""]
    if approval is not None:
        version = f" (contract v{approval.contract_version})" if approval.contract_version else ""
        lines += [
            f"**Approval:** {approval.decision.value} by `{approval.approver}` under policy "
            f"`{approval.policy.value}`{version}; approval `{approval.approval_id}`, plan hash "
            f"`{approval.artifact_hash[:12]}`.",
            "",
        ]
    return lines


def _render_resources(resources: ResourceUse | None) -> list[str]:
    if resources is None:
        return []
    r = resources
    usd = f"${r.usd:.4f}" + (f" of ${r.budget_usd:.2f}" if r.budget_usd is not None else "")
    compute = f"{r.compute_seconds:.1f} s" + (
        f" of {r.budget_compute_seconds:.0f} s" if r.budget_compute_seconds is not None else ""
    )
    return [
        "## Resources (up to the start of reporting)",
        f"- Cost: {usd} at list price; LLM tokens: {r.llm_tokens:,}",
        f"- Model calls: {r.model_calls}; tool calls: {r.tool_calls}",
        f"- Sandbox compute: {compute}; wall clock: {r.wall_clock_seconds:.1f} s",
        f"- Trials: {r.trials}; jobs: {r.jobs}",
        "",
    ]


def _render_evidence(evidence: OutcomeEvidence | None) -> list[str]:
    """The platform's account of the decision and of each failure it observed."""
    if evidence is None:
        return []
    lines = ["## Decision and faults"]
    if evidence.decision is not None:
        decision = evidence.decision
        lines.append(f"- Decision: {decision.intended} ({decision.basis})")
    if evidence.stop is not None:
        lines.append(f"- Stop cause: {evidence.stop.cause}")
    for fault in evidence.faults:
        where = fault.tool or fault.job_id or "controller"
        lines.append(f"- Fault {fault.code} at {where}: {fault.outcome}")
    return [*lines, ""] if len(lines) > 1 else []


class EarlyStopReport(StrictModel):
    """The report of a run that stopped before it had a contract (nothing to measure yet)."""

    run_id: str
    outcome: Literal["no_verified_model"] = "no_verified_model"
    notes: list[str]
    evidence: OutcomeEvidence | None = None
    verification_status: Literal["not_run"] = "not_run"
    limitations: list[str] = Field(default_factory=list)
    resources: ResourceUse | None = None
    approval: ReportApproval | None = None


def build_early_stop_report(
    database: Database, run_id: str, *, notes: list[str], runs_root: Path
) -> EarlyStopReport:
    """The report of a run that stopped before its contract, from stored records only.

    Its approval is the last plan decision recorded before reporting: a rejected plan never gets a
    contract, so a user's or the policy's denial is reported here.
    """
    timeline = _Timeline.read(database, run_id)
    approval = _approval(timeline, None)
    screened = False
    with contextlib.suppress(RecordNotFound):  # stopped before profiling: nothing was screened
        screened = load_dataset_profile(database, run_id)[1].tabular is not None
    return EarlyStopReport(
        run_id=run_id,
        notes=notes,
        evidence=derive_outcome_evidence(database, runs_root, run_id),
        limitations=_limitations(
            timeline,
            verification_status="not_run",
            screened=screened,
            approval=approval,
            approval_expected=False,
        ),
        resources=_resources(database, runs_root, run_id, timeline),
        approval=approval,
    )


def render_early_stop(report: EarlyStopReport) -> str:
    """Render the report of a run that stopped before contract creation."""
    lines = [f"# Hypora run report: {report.run_id}", "", "**Outcome:** no verified model.", ""]
    lines += ["The run stopped before its experiment contract was approved.", ""]
    lines += _render_verification(
        report.verification_status, [], [], report.limitations, report.approval, narrated=True
    )
    lines += _render_resources(report.resources)
    lines += _render_evidence(report.evidence)
    lines += ["## Run notes", *(f"- {note}" for note in report.notes), ""]
    return "\n".join(lines)


__all__ = [
    "APPROVAL_DEFENSE_OFF_APPROVER",
    "VERDICT_LINE",
    "CheckLine",
    "EarlyStopReport",
    "FinalModel",
    "RecordNotFound",
    "Report",
    "ReportApproval",
    "ReportNarrative",
    "ResourceUse",
    "build_early_stop_report",
    "build_report",
    "fill_placeholders",
    "metric_summary",
    "render_early_stop",
    "render_markdown",
    "write_narrative",
]
