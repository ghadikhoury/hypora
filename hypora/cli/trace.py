"""`hypora trace`: a run's event log as a timeline a person can read (spec §16, §21; PLAN.md S1.6).

One row per event: sequence number, time since the run's first event, the acting role, the event
type, a duration, LLM tokens, and a one-line detail. Durations come from the events themselves:

| Event | Duration |
|---|---|
| `model_call`, `tool_result` | the recorded latency |
| `handoff` | since the `role_started` with the same role instance |
| `stage_transition` | time spent in the previous stage |
| `job_transition` | time the job spent in its previous state |

Durations are computed over the whole log before any filter, so a filtered view (only handoffs,
say) still shows how long each role ran. Events hold redacted summaries only (spec §4.3), and the
detail column never shows tool arguments' values, only their names.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from hypora.core.schemas import ActorRole, AnyEvent, EventType
from hypora.core.schemas.events import (
    ApprovalDeniedPayload,
    ApprovalGrantedPayload,
    ApprovalRequestedPayload,
    CallStatus,
    CheckpointPayload,
    ErrorPayload,
    EscalationPayload,
    HandoffPayload,
    JobTransitionPayload,
    ModelCallPayload,
    PlanPayload,
    PolicyDecisionPayload,
    RoleStartedPayload,
    StageTransitionPayload,
    ToolCallPayload,
    ToolResultPayload,
)

FILTER_KEYS: Final = ("type", "role", "trial", "job")


@dataclass(frozen=True)
class TraceRow:
    """One event, ready to print."""

    seq: int
    offset_s: float
    role: str
    type: str
    duration_s: float | None
    tokens: tuple[int, int] | None  # (input, output)
    detail: str
    trial_id: str | None = None
    job_id: str | None = None


@dataclass(frozen=True)
class TraceFilter:
    """Keep rows matching every given key; several values for one key match any of them."""

    types: frozenset[EventType] = frozenset()
    roles: frozenset[ActorRole] = frozenset()
    trials: frozenset[str] = frozenset()
    jobs: frozenset[str] = frozenset()

    def keeps(self, row: TraceRow) -> bool:
        return (
            (not self.types or row.type in self.types)
            and (not self.roles or row.role in self.roles)
            and (not self.trials or row.trial_id in self.trials)
            and (not self.jobs or row.job_id in self.jobs)
        )


def parse_filters(specs: Iterable[str]) -> TraceFilter:
    """Parse `key=value[,value...]` options; raises ValueError naming the valid keys and values."""
    values: dict[str, set[str]] = {key: set() for key in FILTER_KEYS}
    for spec in specs:
        key, sep, raw = spec.partition("=")
        key = key.strip().lower()
        if not sep or key not in values or not raw.strip():
            raise ValueError(
                f"filter {spec!r}: use key=value with key one of {', '.join(FILTER_KEYS)}"
            )
        values[key].update(item.strip() for item in raw.split(",") if item.strip())
    for key, enum in (("type", EventType), ("role", ActorRole)):
        unknown = values[key] - {member.value for member in enum}
        if unknown:
            valid = ", ".join(member.value for member in enum)
            raise ValueError(f"unknown {key} {sorted(unknown)}; valid: {valid}")
    return TraceFilter(
        types=frozenset(EventType(v) for v in values["type"]),
        roles=frozenset(ActorRole(v) for v in values["role"]),
        trials=frozenset(values["trial"]),
        jobs=frozenset(values["job"]),
    )


def _job_id(event: AnyEvent) -> str | None:
    payload = event.payload
    if isinstance(payload, JobTransitionPayload):
        return payload.job_id
    if isinstance(payload, CheckpointPayload):
        return payload.job_id or event.job_id
    return event.job_id


def _trial_id(event: AnyEvent) -> str | None:
    payload = event.payload
    if isinstance(payload, JobTransitionPayload | CheckpointPayload) and payload.trial_id:
        return payload.trial_id
    return event.trial_id


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(event: AnyEvent) -> str:
    """The one-line detail for an event (no tool argument values, no data)."""
    p = event.payload
    if isinstance(p, ModelCallPayload):
        parts = [f"{p.provider.value}/{p.model_id}", f"tier={p.tier.value}"]
        if p.prompt_ref:
            parts.append(p.prompt_ref)
        if p.retries:
            parts.append(f"retries={p.retries}")
        if p.rate_limit_wait_ms >= 100:  # shorter waits are scheduling noise
            parts.append(f"rate-limit wait {p.rate_limit_wait_ms / 1000:.1f}s")
        if p.fallback_from is not None:
            parts.append(f"FALLBACK from {p.fallback_from.model_id} ({p.fallback_from.reason})")
        if p.status is CallStatus.ERROR:
            parts.append(f"ERROR {p.error_code}")
        parts.append(f"${p.cost_usd:.4f}")
        return " ".join(parts)
    if isinstance(p, ToolCallPayload):
        return f"{p.tool}({', '.join(sorted(p.arguments))})"
    if isinstance(p, ToolResultPayload):
        text = f"{p.tool} -> {p.status.value}"
        if p.error is not None:
            text += f" {p.error.code}: {_short(p.error.message, 80)}"
        return text + (" (truncated)" if p.truncated else "")
    if isinstance(p, PolicyDecisionPayload):
        text = f"{p.decision.value} {p.tool}"
        if p.decided_by is not None:
            text += f" by {p.decided_by.value}"
        return text + (f": {_short(p.reasons[0], 80)}" if p.reasons else "")
    if isinstance(p, ApprovalRequestedPayload):
        return f"{p.artifact_type.value} {p.artifact_hash[:12]} ({p.policy.value}): {p.summary}"
    if isinstance(p, ApprovalGrantedPayload):
        return f"{p.artifact_type.value} {p.artifact_hash[:12]} approved by {p.approver}"
    if isinstance(p, ApprovalDeniedPayload):
        return f"{p.artifact_type.value} {p.artifact_hash[:12]} denied by {p.approver}: {p.reason}"
    if isinstance(p, StageTransitionPayload):
        where = f"{p.stage.value}/{p.status.value}" if p.stage else p.status.value
        return f"-> {where}: {p.reason}"
    if isinstance(p, JobTransitionPayload):
        prior = p.expected_prior_state.value if p.expected_prior_state else "new"
        attempt = f" attempt {p.attempt}" if p.attempt else ""
        return f"{p.job_id} {prior} -> {p.to_state.value}{attempt} ({p.initiator}): {p.reason}"
    if isinstance(p, CheckpointPayload):
        return f"{p.kind.value} checkpoint #{p.sequence}" + (f" {p.job_id}" if p.job_id else "")
    if isinstance(p, EscalationPayload):
        return f"{p.escalation_type.value}: {p.summary}"
    if isinstance(p, ErrorPayload):
        return f"{p.code} [{p.component}]: {p.message}"
    if isinstance(p, RoleStartedPayload):
        tier = f" tier={p.tier.value}" if p.tier else ""
        tools = f" tools={len(p.allowed_tools)}" if p.allowed_tools else ""
        return f"{p.role_instance_id}{tier}{tools}"
    if isinstance(p, PlanPayload):
        return f"{p.plan.plan_id} rev {p.plan.revision}: {len(p.plan.steps)} steps"
    if isinstance(p, HandoffPayload):
        h = p.handoff
        text = f"{h.from_role.value} -> {h.to.value} {h.status.value}"
        if h.output_ref:
            text += f" output={h.output_ref}"
        if h.candidates:
            text += f" candidates={','.join(h.candidates)}"
        return text
    return event.type.value  # pragma: no cover - every payload type is handled above


def _seconds(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds())


def trace_rows(events: Sequence[AnyEvent]) -> list[TraceRow]:
    """Every event as a row, in sequence order, with its derived duration."""
    if not events:
        return []
    start = events[0].created_at
    role_starts: dict[str, datetime] = {}
    job_last: dict[str, datetime] = {}
    stage_last: datetime | None = None
    rows = []
    for event in events:
        p = event.payload
        duration: float | None = None
        tokens: tuple[int, int] | None = None
        if isinstance(p, ModelCallPayload):
            duration, tokens = p.latency_ms / 1000, (p.input_tokens, p.output_tokens)
        elif isinstance(p, ToolResultPayload):
            duration = p.latency_ms / 1000
        elif isinstance(p, RoleStartedPayload):
            role_starts[p.role_instance_id] = event.created_at
        elif isinstance(p, HandoffPayload) and p.role_instance_id in role_starts:
            duration = _seconds(event.created_at, role_starts[p.role_instance_id])
        elif isinstance(p, StageTransitionPayload):
            if stage_last is not None:
                duration = _seconds(event.created_at, stage_last)
            stage_last = event.created_at
        elif isinstance(p, JobTransitionPayload):
            if p.job_id in job_last:
                duration = _seconds(event.created_at, job_last[p.job_id])
            job_last[p.job_id] = event.created_at
        rows.append(
            TraceRow(
                seq=event.seq,
                offset_s=_seconds(event.created_at, start),
                role=event.actor_role.value,
                type=event.type.value,
                duration_s=duration,
                tokens=tokens,
                detail=describe(event),
                trial_id=_trial_id(event),
                job_id=_job_id(event),
            )
        )
    return rows


def format_duration(seconds: float | None) -> str:
    """`0.42s`, `12.3s`, `3m05s`, `1h02m`; `-` when there is none."""
    if seconds is None:
        return "-"
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, rest = divmod(round(seconds), 60)
        return f"{minutes}m{rest:02d}s"
    hours, rest = divmod(round(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def format_offset(seconds: float) -> str:
    """Time since the run's first event: `M:SS.s`, or `H:MM:SS` past an hour."""
    if seconds >= 3600:
        hours, within_hour = divmod(int(seconds), 3600)
        return f"{hours}:{within_hour // 60:02d}:{within_hour % 60:02d}"
    minutes, within_minute = divmod(seconds, 60)
    return f"{int(minutes)}:{within_minute:04.1f}"


HEADERS: Final = ("SEQ", "+TIME", "ROLE", "TYPE", "DURATION", "TOKENS IN/OUT", "DETAIL")


def _cells(row: TraceRow) -> tuple[str, ...]:
    tokens = f"{row.tokens[0]:,}/{row.tokens[1]:,}" if row.tokens else "-"
    return (
        str(row.seq),
        format_offset(row.offset_s),
        row.role,
        row.type,
        format_duration(row.duration_s),
        tokens,
        row.detail,
    )


def render_rows(rows: Sequence[TraceRow], width: int | None = None) -> list[str]:
    """Aligned table lines. A line longer than `width` has its detail cut (keeping at least a few
    characters of it); `width=None` never cuts."""
    table = [HEADERS, *(_cells(row) for row in rows)]
    widths = [max(len(cells[i]) for cells in table) for i in range(len(HEADERS) - 1)]
    right = {0, 4}  # numbers and durations align right
    lines = []
    for cells in table:
        fixed = "  ".join(
            cell.rjust(w) if i in right else cell.ljust(w)
            for i, (cell, w) in enumerate(zip(cells[:-1], widths, strict=True))
        )
        line = f"{fixed}  {cells[-1]}"
        if width is not None and len(line) > width:
            line = line[: max(width - 3, len(fixed) + 6)] + "..."
        lines.append(line.rstrip())
    return lines


def summary(rows: Sequence[TraceRow], events: Sequence[AnyEvent]) -> str:
    """Totals over the whole run: events, model calls, tokens, list-price cost, tool results,
    errors, and the time span."""
    calls = [e.payload for e in events if isinstance(e.payload, ModelCallPayload)]
    results = [e.payload for e in events if isinstance(e.payload, ToolResultPayload)]
    errors = sum(1 for e in events if e.type is EventType.ERROR)
    span = _seconds(events[-1].created_at, events[0].created_at) if events else 0.0
    tool_failures = sum(1 for r in results if r.error is not None)
    parts = [
        f"{len(events)} events" + (f" ({len(rows)} shown)" if len(rows) != len(events) else ""),
        f"{len(calls)} model calls ({sum(c.input_tokens for c in calls):,} in / "
        f"{sum(c.output_tokens for c in calls):,} out tokens, "
        f"${sum(c.cost_usd for c in calls):.4f} at list price)",
        f"{len(results)} tool results ({tool_failures} failed)",
        f"{errors} error events",
        f"span {format_duration(span)}",
    ]
    return " | ".join(parts)


__all__ = [
    "FILTER_KEYS",
    "TraceFilter",
    "TraceRow",
    "describe",
    "format_duration",
    "format_offset",
    "parse_filters",
    "render_rows",
    "summary",
    "trace_rows",
]
