"""The plan + budget approval gate (spec §10; S3.5 design §2.2).

The controller calls `ApprovalGate.decide` for every plan version, before any Phase B compute:

1. `approval_requested` is recorded for the plan's exact hash, with a summary of what it covers.
2. `defenses.approval` off (an F6 ablation only): granted by policy without the bounds check, with
   the approver `policy:defense_off`. The request summary, the grant reason and a
   `policy_decision` say so; it is never silent.
3. `auto_within_contract`: granted only if `within_bounds` finds nothing outside the case's
   pre-approved bounds; otherwise a DENIED record and `approval_denied` list what is outside.
4. `interactive`: the run waits (no worker process or MCP session is alive yet), polling for the
   decision file `runs/<run>/approvals/<approval_id>.json` every `poll_seconds`, the same idea as
   the `CANCEL` file. Each poll also checks user cancel and the run's wall clock.
5. A decision file is accepted only if it is a valid `ApprovalRecord` for this run, this approval
   ID, a plan, the interactive policy, an approver `user:<name>`, and the hash pending now. A
   decision for another hash is stale: an `error` event `APPROVAL_STALE`, the file is moved aside,
   and the gate keeps waiting. A valid decision is saved (once: the approval ID is the primary
   key) and recorded as `approval_granted` or `approval_denied` by the user.

Cancel and wall-clock expiry raise the guard's own exceptions (`RunCancelled`, `RunLimitReached`,
each recorded once by the guard), so the controller's existing stop path handles them. A denial is
returned, not raised: the controller ends a user rejection CANCELLED and a policy denial FAILED.

The CLI side (`hypora approve`, the inline prompt of `hypora run`) uses `pending_approval`,
`user_decision` and `write_decision` from here, so both ends agree on the file and its checks.
"""

import contextlib
import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

import anyio
from pydantic import BaseModel, ValidationError

from hypora.core.controller.contract import ContractError
from hypora.core.llm.sinks import redact
from hypora.core.policy.approvals import (
    APPROVAL_DEFENSE_OFF_APPROVER,
    ApprovalBounds,
    ApprovalScope,
    plan_scope_hash,
    scope_hash,
    within_bounds,
)
from hypora.core.roles.base import RunLimitReached
from hypora.core.schemas import (
    AUTO_APPROVER,
    ActorRole,
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovedPlan,
    Budget,
    DataContract,
    EventType,
)
from hypora.core.schemas.events import (
    ApprovalDeniedEvent,
    ApprovalDeniedPayload,
    ApprovalGrantedEvent,
    ApprovalGrantedPayload,
    ApprovalRequestedEvent,
    ApprovalRequestedPayload,
    ErrorPayload,
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
)
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.records import save_approval

APPROVALS_DIR: Final = "approvals"
"""`runs/<run>/approvals/`: one decision file per approval ID."""
REJECTED_DIR: Final = "rejected"
"""Stale or invalid decision files are moved here, so a corrected decision can be written."""
MAX_DECISION_BYTES: Final = 64 * 1024
MAX_REASON_CHARS: Final = 500
USER_PREFIX: Final = "user:"
DEFENSE_OFF_NOTE: Final = (
    "defenses.approval is off (F6 ablation): plan granted by policy without the bounds check"
)
COVERED_PREFIX: Final = "covered by "
"""Starts the reason (and so the request summary) of a grant made under an earlier approval."""
GATE_TOOL: Final = "plan_gate"
"""The `tool` named on the gate's `policy_decision` events (the schema requires one)."""

MAX_USER_CHARS: Final = 64
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NAME_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_USER_APPROVER = re.compile(rf"^{USER_PREFIX}[^\x00-\x1f\x7f]{{1,{MAX_USER_CHARS}}}$")


class GateVerdict(StrEnum):
    """How a plan version left the gate."""

    GRANTED = "granted"
    DENIED_BY_USER = "denied_by_user"
    DENIED_BY_POLICY = "denied_by_policy"


@dataclass(frozen=True)
class GateResult:
    """The decision the gate recorded for one plan version."""

    verdict: GateVerdict
    record: ApprovalRecord
    reason: str

    @property
    def granted(self) -> bool:
        """Whether Phase B may start under this approval."""
        return self.verdict is GateVerdict.GRANTED


class GateGuard(Protocol):
    """What the gate needs from the run's guard (`limits.RunGuard` satisfies it)."""

    budget: Budget

    def check_cancel(self) -> None:
        """Raise RunCancelled (recorded once) if the user cancelled the run."""

    def elapsed(self) -> float:
        """Seconds since the run started."""

    def remaining_wall_clock(self) -> float | None:
        """Seconds left before the run's wall-clock limit; None without a limit."""

    def stop(self, limit: str, used: float, maximum: float) -> RunLimitReached:
        """Record that `limit` fired (once) and return the exception to raise."""


# --- the decision file (shared with the CLI) -----------------------------------------------------


class DecisionExists(FileExistsError):
    """A decision was already written for this approval: the first decision wins."""


def approval_id_for(plan: ApprovedPlan) -> str:
    """The approval ID of one plan version (the existing `appr_<run>_plan_v<N>` form)."""
    return f"appr_{plan.run_id}_plan_v{plan.version}"


def artifact_ref_for(plan: ApprovedPlan) -> str:
    """The `artifact_ref` recorded on `approval_requested` for one plan version."""
    return f"approved_plan:{plan.run_id}.v{plan.version}"


def decision_path(runs_root: Path, run_id: str, approval_id: str) -> Path:
    """Where the user's decision on `approval_id` is written."""
    return runs_root / run_id / APPROVALS_DIR / f"{approval_id}.json"


def clean_reason(text: str) -> str:
    """A user's reason as it may be recorded: credential shapes redacted, control characters
    removed, at most `MAX_REASON_CHARS`."""
    redacted = redact(text)
    assert isinstance(redacted, str)
    return _CONTROL.sub(" ", redacted).strip()[:MAX_REASON_CHARS]


def user_decision(
    pending: "PendingApproval",
    *,
    approve: bool,
    user: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> ApprovalRecord:
    """The `ApprovalRecord` a user's decision on `pending` writes. A denial needs a reason."""
    cleaned = clean_reason(reason) if reason else None
    name = _NAME_CONTROL.sub("", user).strip()[:MAX_USER_CHARS] or "unknown"
    return ApprovalRecord(
        approval_id=pending.approval_id,
        run_id=pending.run_id,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=pending.artifact_hash,
        approver=USER_PREFIX + name,
        decision=ApprovalDecision.APPROVED if approve else ApprovalDecision.DENIED,
        policy=ApprovalPolicy.INTERACTIVE,
        reason=cleaned or None,
        created_at=now or datetime.now(UTC),
    )


def write_decision(runs_root: Path, record: ApprovalRecord) -> Path:
    """Write `record` as the decision file, atomically and only once (raises DecisionExists).

    The content goes to a private temporary file first, then is hard-linked into place: a link
    never replaces an existing file, so of two concurrent decisions exactly one lands, and the
    controller never reads a half-written file.
    """
    path = decision_path(runs_root, record.run_id, record.approval_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(indent=2))
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise DecisionExists(
                f"a decision on {record.approval_id} was already written"
            ) from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return path


@dataclass(frozen=True)
class PendingApproval:
    """The latest `approval_requested` of a run that has no decision event yet."""

    run_id: str
    approval_id: str
    artifact_hash: str
    policy: ApprovalPolicy
    summary: str
    artifact_ref: str | None

    @property
    def plan_version(self) -> int | None:
        """The plan version named by the artifact reference, if any."""
        if self.artifact_ref is None:
            return None
        match = re.search(r"\.v(\d+)$", self.artifact_ref)
        return int(match.group(1)) if match else None


def pending_approval(database: Database, run_id: str) -> PendingApproval | None:
    """The run's undecided plan approval request, read from the event log; None if none."""
    events = EventLog(database).read(
        run_id,
        EventFilters(
            types=(
                EventType.APPROVAL_REQUESTED,
                EventType.APPROVAL_GRANTED,
                EventType.APPROVAL_DENIED,
            )
        ),
    )
    decided = {
        event.payload.approval_id
        for event in events
        if isinstance(event, ApprovalGrantedEvent | ApprovalDeniedEvent)
    }
    for event in reversed(events):
        if isinstance(event, ApprovalRequestedEvent):
            payload = event.payload
            if payload.approval_id in decided or payload.artifact_type is not (
                ApprovalArtifactType.PLAN
            ):
                return None
            return PendingApproval(
                run_id=run_id,
                approval_id=payload.approval_id,
                artifact_hash=payload.artifact_hash,
                policy=payload.policy,
                summary=payload.summary,
                artifact_ref=payload.artifact_ref,
            )
    return None


# --- the gate -------------------------------------------------------------------------------------


Sleep = Callable[[float], Awaitable[None]]


class ApprovalGate:
    """Records and decides plan approvals for one run (design §2.2)."""

    def __init__(
        self,
        *,
        database: Database,
        runs_root: Path,
        run_id: str,
        policy: ApprovalPolicy,
        defense_on: bool,
        guard: GateGuard,
        poll_seconds: float = 1.0,
        sleep: Sleep = anyio.sleep,
    ) -> None:
        self.database = database
        self.events = EventLog(database)
        self.runs_root = runs_root
        self.run_id = run_id
        self.policy = policy
        self.defense_on = defense_on
        self.guard = guard
        self.poll_seconds = poll_seconds
        self._sleep = sleep

    async def decide(
        self, plan: ApprovedPlan, scope: ApprovalScope, bounds: ApprovalBounds, *, note: str
    ) -> GateResult:
        """Record the request for `plan`, then grant or deny it under the run's policy.

        Raises ContractError when the plan is not this run's or not bound to `scope`, and the
        guard's RunCancelled / RunLimitReached when the user cancels or the wall clock runs out
        while an interactive approval is pending.
        """
        if plan.run_id != self.run_id or scope.run_id != self.run_id:
            raise ContractError("the plan or its scope belongs to another run")
        if plan_scope_hash(plan) != scope_hash(scope):
            raise ContractError("the plan is not bound to its approval scope")
        approval_id = approval_id_for(plan)
        artifact_hash = plan.content_hash()
        # With the defense off the grant is a policy grant, recorded as such end to end.
        recorded_policy = self.policy if self.defense_on else ApprovalPolicy.AUTO_WITHIN_CONTRACT
        self._event(
            EventType.APPROVAL_REQUESTED,
            ApprovalRequestedPayload(
                approval_id=approval_id,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=artifact_hash,
                artifact_ref=artifact_ref_for(plan),
                policy=recorded_policy,
                summary=self._summary(plan, scope, note),
            ),
        )
        if not self.defense_on:
            self._event(
                EventType.POLICY_DECISION,
                PolicyDecisionPayload(
                    tool=GATE_TOOL,
                    decision=PolicyOutcome.ALLOW,
                    decided_by=PolicyRule.APPROVAL_STATE,
                    reasons=[DEFENSE_OFF_NOTE, f"configured policy: {self.policy.value}"],
                ),
                actor=ActorRole.POLICY_ENGINE,
            )
            return self._record(
                self._policy_record(
                    approval_id,
                    plan,
                    artifact_hash,
                    approved=True,
                    reason=DEFENSE_OFF_NOTE,
                    approver=APPROVAL_DEFENSE_OFF_APPROVER,
                ),
                GateVerdict.GRANTED,
                actor=ActorRole.POLICY_ENGINE,
            )
        if self.policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT:
            outside = within_bounds(scope, plan, bounds)
            if outside:
                reason = "outside the pre-approved contract: " + "; ".join(outside)
                return self._record(
                    self._policy_record(
                        approval_id, plan, artifact_hash, approved=False, reason=reason
                    ),
                    GateVerdict.DENIED_BY_POLICY,
                    actor=ActorRole.POLICY_ENGINE,
                )
            return self._record(
                self._policy_record(approval_id, plan, artifact_hash, approved=True, reason=note),
                GateVerdict.GRANTED,
                actor=ActorRole.POLICY_ENGINE,
            )
        return await self._wait_for_user(approval_id, artifact_hash)

    def covered(
        self,
        plan: ApprovedPlan,
        scope: ApprovalScope,
        *,
        covered_by: ApprovalRecord,
        old_data: DataContract,
        new_data: DataContract,
        note: str,
    ) -> GateResult:
        """Record `plan` as granted under `covered_by`, an earlier grant that already covers it.

        S4.2d (D-052): the plan approval that authorized the final candidate also covers training
        one comparison model without the top confirmed too-good column, so the plan version that
        leaves that column out is recorded without asking again. The gate itself checks the
        cover (`covered_problem`): the covering plan and scope are re-read from the files it
        approved; the new scope equals that scope apart from the data contract hash; the new data
        contract (`new_data`) is the covered one (`old_data`) with exactly one allowed column moved
        to the excluded features; every other plan term is unchanged; and the run has no earlier
        covered grant (one comparison in total). Anything else raises ContractError and records
        nothing. The record carries the covering approval's approver and policy, the request
        summary and a `policy_decision` name the covering approval, and the request, decision and
        grant are appended in one transaction, so no reader ever sees the plan pending (no prompt).
        """
        if plan.run_id != self.run_id or scope.run_id != self.run_id:
            raise ContractError("the plan or its scope belongs to another run")
        if plan_scope_hash(plan) != scope_hash(scope):
            raise ContractError("the plan is not bound to its approval scope")
        if (
            covered_by.run_id != self.run_id
            or covered_by.decision is not ApprovalDecision.APPROVED
            or covered_by.artifact_type is not ApprovalArtifactType.PLAN
        ):
            raise ContractError("the covering approval is not a granted plan approval of this run")
        problem = self.covered_problem(plan, scope, covered_by, old_data, new_data)
        if problem is not None:
            raise ContractError(f"not covered by {covered_by.approval_id}: {problem}")
        approval_id = approval_id_for(plan)
        artifact_hash = plan.content_hash()
        reason = f"{COVERED_PREFIX}{covered_by.approval_id}: {note}"
        record = ApprovalRecord(
            approval_id=approval_id,
            run_id=self.run_id,
            artifact_type=ApprovalArtifactType.PLAN,
            artifact_hash=artifact_hash,
            approver=covered_by.approver,
            decision=ApprovalDecision.APPROVED,
            policy=covered_by.policy,
            reason=reason,
            created_at=datetime.now(UTC),
        )
        save_approval(self.database, record)
        events: list[tuple[EventType, BaseModel, ActorRole]] = [
            (
                EventType.APPROVAL_REQUESTED,
                ApprovalRequestedPayload(
                    approval_id=approval_id,
                    artifact_type=ApprovalArtifactType.PLAN,
                    artifact_hash=artifact_hash,
                    artifact_ref=artifact_ref_for(plan),
                    policy=covered_by.policy,
                    summary=self._summary(plan, scope, reason),
                ),
                ActorRole.CONTROLLER,
            ),
            (
                EventType.POLICY_DECISION,
                PolicyDecisionPayload(
                    tool=GATE_TOOL,
                    decision=PolicyOutcome.ALLOW,
                    decided_by=PolicyRule.APPROVAL_STATE,
                    reasons=[reason[:MAX_REASON_CHARS]],
                ),
                ActorRole.POLICY_ENGINE,
            ),
            (
                EventType.APPROVAL_GRANTED,
                ApprovalGrantedPayload(
                    approval_id=approval_id,
                    artifact_type=ApprovalArtifactType.PLAN,
                    artifact_hash=artifact_hash,
                    approver=record.approver,
                    policy=record.policy,
                ),
                ActorRole.POLICY_ENGINE,
            ),
        ]
        with self.database.write() as connection:
            for type, payload, actor in events:
                self.events.append(
                    self.run_id, type, actor, payload.model_dump(mode="json"), connection=connection
                )
        return GateResult(GateVerdict.GRANTED, record, reason)

    def covered_problem(
        self,
        plan: ApprovedPlan,
        scope: ApprovalScope,
        covered_by: ApprovalRecord,
        old_data: DataContract,
        new_data: DataContract,
    ) -> str | None:
        """Why `covered_by` does not cover `plan` (S4.2d), or None when it does."""
        match = re.fullmatch(rf"appr_{re.escape(self.run_id)}_plan_v(\d+)", covered_by.approval_id)
        if match is None:
            return "the covering approval is not a plan version's approval"
        version = int(match.group(1))
        folder = self.runs_root / self.run_id
        name = "approved_plan.json" if version == 1 else f"approved_plan_v{version}.json"
        try:
            old_plan = ApprovedPlan.model_validate_json((folder / name).read_bytes())
            old_scope = ApprovalScope.model_validate_json(
                (folder / f"approval_scope_v{version}.json").read_bytes()
            )
        except (OSError, ValidationError, ValueError):
            return "the covering plan or scope file is missing or invalid"
        if old_plan.content_hash() != covered_by.artifact_hash:
            return "the covering plan file changed after it was approved"
        if plan_scope_hash(old_plan) != scope_hash(old_scope):
            return "the covering scope file changed after it was approved"
        if plan.version <= old_plan.version:
            return "the plan is not a later version"
        changed = {"data_contract_hash"}
        if scope.model_dump(mode="json", exclude=changed) != old_scope.model_dump(
            mode="json", exclude=changed
        ):
            return "the approval scope differs from the covered one"
        if old_data.content_hash() != old_plan.data_contract_hash:
            return "the covered data contract is not the covering plan's"
        if new_data.content_hash() != plan.data_contract_hash:
            return "the new data contract is not the plan's"
        if new_data.data_contract_version != plan.data_contract_version:
            return "the plan names another data contract version"
        features = {"data_contract_version", "allowed_features", "excluded_features"}
        if new_data.model_dump(mode="json", exclude=features) != old_data.model_dump(
            mode="json", exclude=features
        ):
            return "the data contract changes more than its features"
        removed = [f for f in old_data.allowed_features if f not in new_data.allowed_features]
        added = [e.name for e in new_data.excluded_features[len(old_data.excluded_features) :]]
        if (
            len(removed) != 1
            or list(new_data.allowed_features)
            != [f for f in old_data.allowed_features if f != removed[0]]
            or list(new_data.excluded_features[: len(old_data.excluded_features)])
            != list(old_data.excluded_features)
            or added != removed
        ):
            return "the plan must leave out exactly one allowed column and change nothing else"
        terms = {
            "budget",
            "approval_policy",
            "requested_model_downloads",
            "requested_dependencies",
            "expected_score_range",
        }
        if plan.model_dump(mode="json", include=terms) != old_plan.model_dump(
            mode="json", include=terms
        ):
            return "the plan's terms differ from the covered plan's"
        earlier = EventLog(self.database).read(
            self.run_id, EventFilters(types=(EventType.APPROVAL_REQUESTED,))
        )
        if any(
            isinstance(event, ApprovalRequestedEvent)
            and event.payload.summary.find(f"{COVERED_PREFIX}appr_") >= 0
            for event in earlier
        ):
            return "the run already used its one covered comparison"
        return None

    # --- interactive ------------------------------------------------------------------------------

    async def _wait_for_user(self, approval_id: str, artifact_hash: str) -> GateResult:
        path = decision_path(self.runs_root, self.run_id, approval_id)
        while True:
            self.guard.check_cancel()
            remaining = self.guard.remaining_wall_clock()
            maximum = self.guard.budget.max_wall_clock_seconds
            if remaining is not None and remaining <= 0 and maximum is not None:
                raise self.guard.stop("max_wall_clock_seconds", self.guard.elapsed(), maximum)
            record = self._intake(path, approval_id, artifact_hash)
            if record is not None:
                approved = record.decision is ApprovalDecision.APPROVED
                return self._record(
                    record,
                    GateVerdict.GRANTED if approved else GateVerdict.DENIED_BY_USER,
                    actor=ActorRole.USER,
                )
            delay = self.poll_seconds if remaining is None else min(self.poll_seconds, remaining)
            await self._sleep(max(delay, 0.0))

    def _intake(self, path: Path, approval_id: str, artifact_hash: str) -> ApprovalRecord | None:
        """The decision in `path` if it is valid for the pending hash; otherwise None (a stale or
        invalid file is recorded and moved aside so a corrected decision can be written)."""
        if not path.exists() and not path.is_symlink():
            return None
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("not a regular file")
            content = path.read_bytes()
            if len(content) > MAX_DECISION_BYTES:
                raise ValueError("too large")
            record = ApprovalRecord.model_validate_json(content)
        except (OSError, ValueError, ValidationError):
            self._reject(
                path,
                "APPROVAL_DECISION_INVALID",
                "the decision file is not a valid approval record",
                {"approval_id": approval_id},
            )
            return None
        problems = []
        if record.run_id != self.run_id:
            problems.append("another run")
        if record.approval_id != approval_id:
            problems.append("another approval")
        if record.artifact_type is not ApprovalArtifactType.PLAN:
            problems.append("not a plan decision")
        if record.policy is not ApprovalPolicy.INTERACTIVE:
            problems.append("not an interactive decision")
        if (
            not _USER_APPROVER.fullmatch(record.approver)
            or not record.approver.removeprefix(USER_PREFIX).strip()
        ):
            problems.append("the approver is not a user")
        if problems:
            self._reject(
                path,
                "APPROVAL_DECISION_INVALID",
                "decision ignored: " + ", ".join(problems),
                {"approval_id": approval_id},
            )
            return None
        if record.artifact_hash != artifact_hash:
            self._reject(
                path,
                "APPROVAL_STALE",
                "decision ignored: it names another plan hash than the one pending",
                {
                    "approval_id": approval_id,
                    "pending_hash": artifact_hash,
                    "decision_hash": record.artifact_hash,
                },
            )
            return None
        reason = clean_reason(record.reason) if record.reason else None
        if record.decision is ApprovalDecision.DENIED and not reason:
            reason = "denied by the user"
        return record.model_copy(update={"reason": reason})

    def _reject(self, path: Path, code: str, message: str, details: dict[str, str]) -> None:
        aside = path.parent / REJECTED_DIR / f"{path.stem}.{time.time_ns()}.json"
        aside.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, aside)
        except OSError:
            with contextlib.suppress(OSError):
                path.unlink()
        self._event(
            EventType.ERROR,
            ErrorPayload(
                component="controller",
                code=code,
                message=message,
                retryable=True,
                details=dict(details),
            ),
        )

    # --- records and events -----------------------------------------------------------------------

    def _policy_record(
        self,
        approval_id: str,
        plan: ApprovedPlan,
        artifact_hash: str,
        *,
        approved: bool,
        reason: str,
        approver: str = AUTO_APPROVER,
    ) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=approval_id,
            run_id=plan.run_id,
            artifact_type=ApprovalArtifactType.PLAN,
            artifact_hash=artifact_hash,
            approver=approver,
            decision=ApprovalDecision.APPROVED if approved else ApprovalDecision.DENIED,
            policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
            reason=reason or None,
            created_at=datetime.now(UTC),
        )

    def _record(
        self, record: ApprovalRecord, verdict: GateVerdict, *, actor: ActorRole
    ) -> GateResult:
        """Save the decision (refused if one exists for this ID) and record its event."""
        save_approval(self.database, record)
        if record.decision is ApprovalDecision.APPROVED:
            self._event(
                EventType.APPROVAL_GRANTED,
                ApprovalGrantedPayload(
                    approval_id=record.approval_id,
                    artifact_type=record.artifact_type,
                    artifact_hash=record.artifact_hash,
                    approver=record.approver,
                    policy=record.policy,
                ),
                actor=actor,
            )
        else:
            self._event(
                EventType.APPROVAL_DENIED,
                ApprovalDeniedPayload(
                    approval_id=record.approval_id,
                    artifact_type=record.artifact_type,
                    artifact_hash=record.artifact_hash,
                    approver=record.approver,
                    policy=record.policy,
                    reason=record.reason or "denied",
                ),
                actor=actor,
            )
        return GateResult(verdict, record, record.reason or verdict.value)

    def _summary(self, plan: ApprovedPlan, scope: ApprovalScope, note: str) -> str:
        budget = ", ".join(
            f"{name}={value:g}"
            for name, value in plan.budget.model_dump().items()
            if value is not None
        )
        limits = scope.job_limits
        tools = "; ".join(
            f"{entry.role.value}: {', '.join(t.value for t in entry.tools) or 'none'}"
            for entry in scope.tools
        )
        parts = [
            f"Phase B plan v{plan.version}",
            f"budget {budget or 'unlimited'}",
            f"job limits {limits.max_wall_seconds} s, {limits.max_memory_gb:g} GB, "
            f"gpu {'yes' if limits.gpu else 'no'}, cpu threads {limits.cpu_threads or 1}",
            f"tools {tools or 'none'}",
            f"worker ceiling {scope.max_c_workers}",
            f"network {scope.network}",
            f"scope {scope_hash(scope)[:12]}",
            _range_text(plan),
            note,
        ]
        if not self.defense_on:
            parts.append(f"{DEFENSE_OFF_NOTE} (configured policy {self.policy.value})")
        return "; ".join(part for part in parts if part)

    def _event(
        self, type: EventType, payload: BaseModel, actor: ActorRole = ActorRole.CONTROLLER
    ) -> None:
        self.events.append(self.run_id, type, actor, payload.model_dump(mode="json"))


def _range_text(plan: ApprovedPlan) -> str:
    """The plan's expected score range as the gate shows it (D-050), or nothing when unset."""
    expected = plan.expected_score_range
    if expected is None:
        return ""
    if expected.low is None or expected.high is None:
        return (
            f"expected {expected.metric.value} range unknown; if a locked-split diagnostic flags "
            "an extreme single-feature score, this approval also covers training one comparison "
            "model without the column that explains it and recommending it as the final model"
        )
    return (
        f"expected {expected.metric.value} range {expected.low:g} to {expected.high:g} "
        f"({expected.source.value}); if the final validation score is clearly above it, this "
        "approval also covers training one comparison model without the column that explains "
        "it and recommending it as the final model"
    )


__all__ = [
    "APPROVALS_DIR",
    "COVERED_PREFIX",
    "DEFENSE_OFF_NOTE",
    "ApprovalGate",
    "DecisionExists",
    "GateGuard",
    "GateResult",
    "GateVerdict",
    "PendingApproval",
    "approval_id_for",
    "artifact_ref_for",
    "clean_reason",
    "decision_path",
    "pending_approval",
    "user_decision",
    "write_decision",
]
