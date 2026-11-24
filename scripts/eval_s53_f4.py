#!/usr/bin/env python3
"""Generate F4 (Planning & Error Recovery) evaluation evidence for S5.3.

Compares `planning=on` (written plans before first tool call, backoff retries, replanning,
escalation) vs. `planning=off` (no written plans, no RecoverySession) in `agent_mode=multi` on
15 real fault cases (5 tool_error, 5 timeout, 5 malformed_data) plus a separate 6x3 repeat
sample, reusing the 6 frozen S5.2 `default` primary records and 3 frozen `spambase` repeats for
`planning=on` under a hard $2.20 live spend cap.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import anyio
import numpy as np
from dotenv import load_dotenv

from hypora.core.config import load_settings_for_eval
from hypora.core.config.flags import Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import new_run_id
from hypora.core.hashing import sha256_hex
from hypora.core.llm.credentials import load_credentials
from hypora.core.schemas import EventType, JobState
from hypora.evaluation.case import CASES_DIR, Case, load_case
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore, create_hidden_split
from hypora.evaluation.runner.agent import AgentSystem
from hypora.evaluation.runner.guard import (
    RecordGuard,
    hidden_only_values,
    hidden_row_tokens,
    machine_aliases,
)
from hypora.evaluation.runner.records import (
    CaseRecord,
    Provenance,
    RecordStatus,
    RunKind,
    Timings,
)
from hypora.evaluation.runner.runner import (
    RUNS_DIR,
    STATE_DIR,
    TOOLS_CONTRACT,
    failure_categories,
    inputs_of,
    outputs_of,
    system_input_for,
)
from hypora.evaluation.runner.store import GitRepo
from hypora.evaluation.runner.usage import collect_usage
from hypora.evaluation.stats.core import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_RESAMPLES,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)
from hypora.evaluation.system import SystemContext, SystemInput, SystemRun
from hypora.execution.sandbox.image import tabular_image_tag
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
FROZEN_FINAL_ROOT: Final = Path("/home/ghadi/hypora-final")
FROZEN_COMMIT: Final = "62b4e3fb6016a0abddf70a14152252a62c8aca4a"
FROZEN_DEFAULT_DIR: Final = FROZEN_FINAL_ROOT / "results" / FROZEN_COMMIT / "default"
FROZEN_STATE_DIR: Final = FROZEN_FINAL_ROOT / "runs" / "final-s5.2" / "work" / "state"

WORK_DIR: Final = REPO_ROOT / "runs" / "s53-f4"
OUT_DIR: Final = REPO_ROOT / "results" / "s5.3" / "f4"
CAP_USD: Final = 2.20
PER_RUN_RESERVE_USD: Final = 0.06
MAX_PARALLEL_CASES: Final = 3
PRICE_RETRIEVED_ON: Final = "2026-11-07"
FATAL_ERROR_CODES: Final = frozenset({"PROVIDER_AUTH_FAILED", "QUOTA_EXHAUSTED"})

REUSED_TEST_CASES: Final[tuple[str, ...]] = (
    "test_fault_submit_banknote",
    "test_fault_submit_spambase",
    "test_fault_submit_superconductivity",
    "test_fault_write_retinopathy",
    "test_fault_malformed_landsat",
    "test_fault_malformed_maternal",
)
NEW_F4_CASES: Final[tuple[str, ...]] = (
    "f4_fault_inspect_rice",
    "f4_fault_malformed_wine_quality",
    "f4_fault_malformed_dry_bean",
    "f4_fault_malformed_student_dropout",
    "f4_fault_timeout_banknote",
    "f4_fault_timeout_breast_cancer",
    "f4_fault_timeout_rice",
    "f4_fault_timeout_phishing",
    "f4_fault_timeout_spambase",
)
ALL_15_CASES: Final[tuple[str, ...]] = (*REUSED_TEST_CASES, *NEW_F4_CASES)
REPEAT_CASES: Final[tuple[str, ...]] = (
    "test_fault_submit_spambase",
    "f4_fault_inspect_rice",
    "f4_fault_malformed_wine_quality",
    "f4_fault_malformed_student_dropout",
    "f4_fault_timeout_banknote",
    "f4_fault_timeout_breast_cancer",
)
NEW_REPEAT_CASES_ON: Final[tuple[str, ...]] = tuple(
    cid for cid in REPEAT_CASES if cid != "test_fault_submit_spambase"
)


class SpendCapReached(RuntimeError):
    """Raised before launching a new run once aggregate live spend hits `CAP_USD`."""


class ProviderFatalError(RuntimeError):
    """Raised immediately if any live run hits `PROVIDER_AUTH_FAILED` or `QUOTA_EXHAUSTED`."""


class SpendTracker:
    """Thread-safe aggregate spend tracker across both arms of the F4 collection."""

    def __init__(self, cap_usd: float, journal: Path) -> None:
        self.cap_usd = cap_usd
        self.journal = journal
        self._lock = threading.Lock()
        self._spent_by_run: dict[str, float] = {}
        self._active_runs: set[str] = set()
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "run_settled":
                    self._spent_by_run[str(row["run_id"])] = float(row["actual_spend_usd"])

    def measured_usd(self) -> float:
        with self._lock:
            return sum(self._spent_by_run.values())

    def reserve_run(self, run_id: str) -> None:
        with self._lock:
            projected = (
                sum(self._spent_by_run.values())
                + len(self._active_runs) * PER_RUN_RESERVE_USD
                + PER_RUN_RESERVE_USD
            )
            if projected > self.cap_usd:
                raise SpendCapReached(
                    f"Projected F4 live spend ${projected:.4f} exceeds cap ${self.cap_usd:.2f}"
                )
            self._active_runs.add(run_id)

    def settle_run(self, run_id: str, actual_spend_usd: float) -> None:
        with self._lock:
            self._active_runs.discard(run_id)
            if actual_spend_usd <= 0.0:
                return
            self._spent_by_run[run_id] = actual_spend_usd
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "kind": "run_settled",
                "run_id": run_id,
                "actual_spend_usd": actual_spend_usd,
                "at": datetime.now(UTC).isoformat(),
            }
            with self.journal.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")


class BudgetCappedAgent(AgentSystem):
    """Wraps `AgentSystem` with aggregate F4 budget tracking and safe cancellation."""

    def __init__(self, tracker: SpendTracker) -> None:
        super().__init__()
        self._tracker = tracker

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        self._tracker.reserve_run(context.run_id)
        try:
            return await super().run(system_input, context)
        finally:
            usage = collect_usage(context.database, context.run_id)
            self._tracker.settle_run(context.run_id, usage.usage.actual_spend_usd)


def load_all_15_cases() -> dict[str, Case]:
    """Load all 15 fault cases (6 frozen core test cases + 9 F4 extension cases)."""
    cases: dict[str, Case] = {}
    for cid in REUSED_TEST_CASES:
        cases[cid] = load_case(CASES_DIR / "test" / f"{cid}.yaml")
    for cid in NEW_F4_CASES:
        cases[cid] = load_case(CASES_DIR / "f4_faults" / f"{cid}.yaml")
    return cases


def build_arm_run_config(planning: Toggle) -> RunConfig:
    """Build the eval RunConfig matching S5.2 `default`, toggling only `flags.planning`."""
    settings = load_settings_for_eval("default")
    if settings.flags.planning is not planning:
        flags = settings.flags.model_copy(update={"planning": planning})
        settings = settings.model_copy(update={"flags": flags})
    return RunConfig.build(settings, eval_config="default")


def _clean_record_path(arm_name: str, kind: RunKind, case_id: str, attempt: int) -> Path:
    base = WORK_DIR / arm_name / "clean_records"
    if kind is RunKind.PRIMARY:
        return base / "primary" / f"{case_id}.json"
    return base / "repeats" / case_id / f"{attempt}.json"


def _has_fatal_provider_error(db_path: Path) -> str | None:
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE type IN ('model_call', 'error') ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()
    for (pj,) in rows:
        pl = json.loads(pj)
        for key in ("code", "error_code"):
            val = pl.get(key)
            if val in FATAL_ERROR_CODES:
                return str(val)
        msg = str(pl.get("message", ""))
        for code in FATAL_ERROR_CODES:
            if code in msg:
                return code
    return None


def _provenance_for(run_config: RunConfig) -> Provenance:
    code = GitRepo(REPO_ROOT).state()
    return Provenance(
        commit_sha=code.commit,
        dirty=code.dirty,
        config="default",
        profile=run_config.model_profile,
        flags=run_config.flags,
        run_config_hash=run_config.hash,
        tiers=run_config.active_profile.tiers,
        retry_policy=run_config.models.retry,
        python_version=platform.python_version(),
        sandbox_image=tabular_image_tag(),
        tools_contract_sha256=sha256_hex(TOOLS_CONTRACT.read_bytes()),
    )


async def _run_arm(
    arm_name: str,
    planning: Toggle,
    primary_cases: tuple[Case, ...],
    repeat_cases: tuple[str, ...],
    all_cases: dict[str, Case],
    tracker: SpendTracker,
    *,
    parallel: int = MAX_PARALLEL_CASES,
) -> None:
    """Execute only missing clean primary and repeat slots for one arm, with fail-fast auth."""
    arm_work = WORK_DIR / arm_name / "work"
    runs_root = arm_work.resolve() / RUNS_DIR
    state_root = arm_work.resolve() / STATE_DIR
    runs_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    run_config = build_arm_run_config(planning)
    provenance = _provenance_for(run_config)
    store = HiddenStore()
    grader = ProgrammaticGrader()
    system = BudgetCappedAgent(tracker)
    guard = RecordGuard(
        forbidden_paths=(str(store.root),),
        secrets=load_credentials().secret_values(),
        path_aliases=machine_aliases(runs_root, REPO_ROOT),
        public_fields=frozenset({"inputs", "provenance"}),
    )

    slots: list[tuple[Case, RunKind, int, Path]] = []
    for case in primary_cases:
        dest = _clean_record_path(arm_name, RunKind.PRIMARY, case.id, 1)
        if not dest.is_file():
            slots.append((case, RunKind.PRIMARY, 1, dest))
    for cid in repeat_cases:
        case = all_cases[cid]
        for attempt in range(1, 4):
            dest = _clean_record_path(arm_name, RunKind.REPEAT, cid, attempt)
            if not dest.is_file():
                slots.append((case, RunKind.REPEAT, attempt, dest))

    if not slots:
        print(f"[{arm_name}] All requested slots already exist cleanly; skipping execution.")
        return

    print(f"[{arm_name}] Running {len(slots)} missing slot(s) at parallel={parallel}...")
    prepared_splits: dict[str, tuple[HiddenSplit, frozenset[str], Any, float]] = {}
    for case, _, _, _ in slots:
        if case.id not in prepared_splits:
            t0 = time.perf_counter()
            ds = load_case_dataset(case)
            split = create_hidden_split(case, ds, store, runs_root)
            vals = hidden_only_values(split)
            rows = hidden_row_tokens(split)
            prepared_splits[case.id] = (split, vals, rows, time.perf_counter() - t0)

    limiter = anyio.CapacityLimiter(parallel)
    counter_lock = threading.Lock()
    done_count = 0

    async def run_one(case: Case, kind: RunKind, attempt: int, dest: Path) -> None:
        nonlocal done_count
        async with limiter:
            started_at = datetime.now(UTC)
            split, hidden_vals, hidden_rows, prep_s = prepared_splits[case.id]
            run_id = new_run_id()
            db_path = state_root / f"{run_id}.sqlite"
            t_setup = time.perf_counter()
            db = Database(db_path)
            try:
                migrate(db)
            finally:
                db.close()
            setup_s = prep_s + (time.perf_counter() - t_setup)

            system_input = system_input_for(case, split)
            store.refuse_overlap(db_path, system_input.dataset)
            context = SystemContext(
                run_id=run_id,
                runs_root=runs_root,
                database=db_path,
                run_config=run_config,
                injected_faults=case.injected_faults,
            )
            t_sys = time.perf_counter()
            run = await system.run(system_input, context)
            sys_s = time.perf_counter() - t_sys

            fatal = _has_fatal_provider_error(db_path)
            if fatal is not None:
                raise ProviderFatalError(
                    f"Fatal provider error {fatal} on {arm_name}/{case.id} ({run_id}); "
                    "aborting without writing record."
                )

            t_grade = time.perf_counter()
            grade = await grader.grade(case, split, run)
            grade_s = time.perf_counter() - t_grade

            reported_retry = system.retry_wait_seconds(run_id)
            collected = collect_usage(db_path, run_id, retry_wait_seconds=reported_retry)
            timings = Timings(
                setup_seconds=setup_s,
                system_seconds=sys_s,
                grading_seconds=grade_s,
                rate_limit_wait_seconds=collected.rate_limit_wait_seconds,
                retry_wait_seconds=collected.retry_wait_seconds,
                active_seconds=max(
                    0.0,
                    sys_s
                    - collected.rate_limit_wait_seconds
                    - (collected.retry_wait_seconds or 0.0),
                ),
            )
            record = CaseRecord(
                status=RecordStatus.GRADED,
                run_id=run_id,
                kind=kind,
                attempt=attempt,
                case_id=case.id,
                case_split=case.split,
                system=system.name,
                passed=grade.passed,
                failure_categories=failure_categories(grade),
                provenance=provenance,
                inputs=inputs_of(case, split),
                outputs=outputs_of(run, runs_root),
                grade=grade,
                timings=timings,
                usage=collected.usage,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            cleaned = guard.clean(record.model_dump(mode="json"), hidden_vals, hidden_rows)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
            tmp.replace(dest)
            with counter_lock:
                done_count += 1
                idx = done_count
            tag = "primary" if kind is RunKind.PRIMARY else f"repeat#{attempt}"
            print(
                f"  [{arm_name} {idx}/{len(slots)}] {case.id} ({tag}) "
                f"pass={grade.passed} cost=${collected.usage.actual_spend_usd:.4f} "
                f"wall={sys_s:.1f}s total_live=${tracker.measured_usd():.4f}"
            )

    async with anyio.create_task_group() as tg:
        for case, kind, attempt, dest in slots:
            tg.start_soon(run_one, case, kind, attempt, dest)


@dataclass(frozen=True)
class TraceMetrics:
    plan_events_total: int
    plan_before_first_tool: bool | None
    plan_revisions: int
    recovery_policy_decisions: int
    recovery_retries: int
    recovery_replans: int
    recovery_stops: int
    escalations: int
    timeout_job_failures: int
    recovered_timeout_jobs: list[str]
    events_count: int


def _locate_db(run_id: str, arm_name: str, reused: bool) -> Path | None:
    if reused:
        p = FROZEN_STATE_DIR / f"{run_id}.sqlite"
        if p.is_file():
            return p
    p = WORK_DIR / arm_name / "work" / "state" / f"{run_id}.sqlite"
    return p if p.is_file() else None


def extract_trace_metrics(db_path: Path | None, run_id: str) -> TraceMetrics:
    """Compute F4 planning and recovery counts directly from a run's SQLite event log."""
    if db_path is None or not db_path.is_file():
        return TraceMetrics(0, None, 0, 0, 0, 0, 0, 0, 0, [], 0)
    db = Database(db_path)
    try:
        events = EventLog(db).read(run_id)
    finally:
        db.close()

    first_plan_seq: int | None = None
    first_tool_seq: int | None = None
    plan_events = 0
    plan_revisions = 0
    rec_decisions = 0
    rec_retries = 0
    rec_replans = 0
    rec_stops = 0
    escalations = 0
    timeout_failures = 0
    failed_timeout_jobs: set[str] = set()

    for ev in events:
        if ev.type is EventType.PLAN:
            plan_events += 1
            if first_plan_seq is None:
                first_plan_seq = ev.seq
            plan_obj = getattr(ev.payload, "plan", ev.payload)
            if getattr(plan_obj, "revises_plan_id", None) is not None:
                plan_revisions += 1
        elif ev.type is EventType.TOOL_CALL:
            if first_tool_seq is None:
                first_tool_seq = ev.seq
        elif ev.type is EventType.POLICY_DECISION:
            reasons = getattr(ev.payload, "reasons", [])
            text = " ".join(str(r) for r in reasons)
            decided_by = str(getattr(ev.payload, "decided_by", ""))
            if "recovery=" in text or "idempotency" in decided_by:
                rec_decisions += 1
            if "recovery=retry" in text:
                rec_retries += 1
            elif "recovery=replan" in text:
                rec_replans += 1
            elif "recovery=stop" in text:
                rec_stops += 1
        elif ev.type is EventType.ESCALATION:
            escalations += 1
        elif ev.type is EventType.JOB_TRANSITION:
            reason = str(getattr(ev.payload, "reason", ""))
            to_state = getattr(ev.payload, "to_state", None)
            job_id = str(getattr(ev.payload, "job_id", ""))
            if to_state is JobState.FAILED and "TIMEOUT:" in reason:
                timeout_failures += 1
                failed_timeout_jobs.add(job_id)

    plan_before = None
    if first_tool_seq is not None:
        plan_before = first_plan_seq is not None and first_plan_seq < first_tool_seq
    elif plan_events > 0:
        plan_before = True

    return TraceMetrics(
        plan_events_total=plan_events,
        plan_before_first_tool=plan_before,
        plan_revisions=plan_revisions,
        recovery_policy_decisions=rec_decisions,
        recovery_retries=rec_retries,
        recovery_replans=rec_replans,
        recovery_stops=rec_stops,
        escalations=escalations,
        timeout_job_failures=timeout_failures,
        recovered_timeout_jobs=sorted(failed_timeout_jobs),
        events_count=len(events),
    )


def _fault_kind(case: Case) -> str:
    return case.injected_faults[0].kind.value if case.injected_faults else "none"


def _fault_check_detail(record: CaseRecord) -> str | None:
    if record.grade is None:
        return None
    for chk in record.grade.checks:
        if chk.name.startswith("fault_"):
            return f"{chk.status.value}: {chk.detail}"
    return None


def _hidden_metric_score(record: CaseRecord) -> dict[str, Any] | None:
    if record.grade is None or record.grade.hidden_metric is None:
        return None
    hm = record.grade.hidden_metric
    return {
        "metric": hm.name.value,
        "value": round(float(hm.value), 6),
    }


def _extract_sanitized_trace(
    db_path: Path | None,
    run_id: str,
    label: str,
    case_id: str,
    planning_flag: str,
) -> dict[str, Any] | None:
    if db_path is None or not db_path.is_file():
        return None
    db = Database(db_path)
    try:
        events = EventLog(db).read(run_id)
    finally:
        db.close()

    # Identify failed and completed jobs so every trace shows the full fault-to-recovery arc.
    failed_jobs: set[str] = set()
    completed_jobs: list[str] = []
    err_tools: set[str] = set()
    for ev in events:
        if ev.type is EventType.JOB_TRANSITION:
            jid = str(getattr(ev.payload, "job_id", ""))
            st = getattr(ev.payload, "to_state", None)
            if st in (JobState.FAILED, JobState.FAILED_TERMINAL):
                failed_jobs.add(jid)
            elif st is JobState.COMPLETED:
                completed_jobs.append(jid)
        elif ev.type is EventType.TOOL_RESULT and getattr(ev.payload, "error", None) is not None:
            err_tools.add(str(getattr(ev.payload, "tool", "")))

    key_jobs = failed_jobs | (set(completed_jobs[-2:]) if completed_jobs else set())
    interesting: list[dict[str, Any]] = []
    seen_retry_ok: set[str] = set()
    for ev in events:
        keep = False
        if ev.type in (
            EventType.STAGE_TRANSITION,
            EventType.PLAN,
            EventType.ERROR,
            EventType.ESCALATION,
        ):
            keep = True
        elif ev.type is EventType.JOB_TRANSITION:
            jid = str(getattr(ev.payload, "job_id", ""))
            st = getattr(ev.payload, "to_state", None)
            if jid in key_jobs and st in (
                JobState.QUEUED,
                JobState.LEASED,
                JobState.RUNNING,
                JobState.FAILED,
                JobState.FAILED_TERMINAL,
                JobState.COMPLETED,
            ):
                keep = True
        elif ev.type is EventType.POLICY_DECISION:
            reasons = getattr(ev.payload, "reasons", [])
            if "recovery=" in " ".join(str(r) for r in reasons):
                keep = True
        elif ev.type is EventType.TOOL_RESULT:
            tool = str(getattr(ev.payload, "tool", ""))
            if getattr(ev.payload, "error", None) is not None:
                keep = True
            elif tool in err_tools and tool not in seen_retry_ok:
                seen_retry_ok.add(tool)
                keep = True
        elif ev.type is EventType.TOOL_CALL:
            tool = str(getattr(ev.payload, "tool", ""))
            if tool in err_tools and len(seen_retry_ok) <= len(err_tools):
                keep = True
        if not keep:
            continue
        interesting.append(
            {
                "seq": ev.seq,
                "offset_hint": ev.created_at.isoformat(),
                "role": ev.actor_role.value,
                "type": ev.type.value,
                "summary": _summarize_payload(ev.type, ev.payload),
            }
        )

    return {
        "label": label,
        "case_id": case_id,
        "run_id": run_id,
        "planning_flag": planning_flag,
        "total_events": len(events),
        "key_events": interesting,
    }


def _summarize_payload(etype: EventType, payload: Any) -> str:
    if etype is EventType.STAGE_TRANSITION:
        stage = getattr(payload, "stage", "?")
        reason = getattr(payload, "reason", "")
        return f"stage={stage} reason={reason}"
    if etype is EventType.PLAN:
        plan_obj = getattr(payload, "plan", payload)
        pid = getattr(plan_obj, "plan_id", "?")
        rev = getattr(plan_obj, "revises_plan_id", None)
        steps = getattr(plan_obj, "steps", [])
        goal = str(getattr(plan_obj, "goal", ""))[:90]
        return f"plan_id={pid} revises={rev} steps={len(steps)} {goal}"
    if etype is EventType.TOOL_CALL:
        return f"tool={getattr(payload, 'tool', '?')}"
    if etype is EventType.TOOL_RESULT:
        err = getattr(payload, "error", None)
        tool = getattr(payload, "tool", "?")
        if err is not None:
            code = getattr(err, "code", "?")
            retryable = getattr(err, "retryable", "?")
            return f"tool={tool} ERROR code={code} retryable={retryable}"
        return f"tool={tool} OK"
    if etype is EventType.POLICY_DECISION:
        tool = getattr(payload, "tool", "?")
        dec = getattr(payload, "decision", "?")
        reasons = getattr(payload, "reasons", [])
        return f"tool={tool} decision={dec} reasons={reasons}"
    if etype is EventType.JOB_TRANSITION:
        jid = getattr(payload, "job_id", "?")
        to_st = getattr(payload, "to_state", "?")
        reason = str(getattr(payload, "reason", ""))[:80]
        return f"job={jid} -> {to_st} ({reason})"
    if etype is EventType.ERROR:
        comp = getattr(payload, "component", "?")
        code = getattr(payload, "code", "?")
        msg = str(getattr(payload, "message", ""))[:90]
        return f"component={comp} code={code} msg={msg}"
    if etype is EventType.ESCALATION:
        et = getattr(payload, "escalation_type", "?")
        summ = str(getattr(payload, "summary", ""))[:90]
        return f"type={et} summary={summ}"
    return str(payload)[:100]


def _serialize_record(
    record: CaseRecord,
    case: Case,
    arm_name: str,
    reused: bool,
) -> dict[str, Any]:
    db_path = _locate_db(record.run_id, arm_name, reused)
    tm = extract_trace_metrics(db_path, record.run_id)
    expected = case.injected_faults[0].expected.value if case.injected_faults else "none"
    return {
        "case_id": record.case_id,
        "run_id": record.run_id,
        "fault_kind": _fault_kind(case),
        "expected": expected,
        "reused_from": f"{FROZEN_COMMIT}/default" if reused else None,
        "passed": bool(record.passed),
        "status": record.status.value,
        "failure_categories": list(record.failure_categories),
        "fault_check": _fault_check_detail(record),
        "hidden_score": _hidden_metric_score(record),
        "wall_clock_seconds": round(float(record.timings.system_seconds), 3),
        "cost_usd": round(float(record.usage.actual_spend_usd), 6),
        "model_calls": int(record.usage.model_calls),
        "input_tokens": int(record.usage.input_tokens),
        "output_tokens": int(record.usage.output_tokens),
        "tool_calls": int(record.usage.tool_calls),
        "failed_tool_calls": int(record.usage.failed_tool_calls),
        "plan_events": tm.plan_events_total,
        "plan_before_first_tool": tm.plan_before_first_tool,
        "plan_revisions": tm.plan_revisions,
        "recovery_policy_decisions": tm.recovery_policy_decisions,
        "recovery_retries": tm.recovery_retries,
        "recovery_replans": tm.recovery_replans,
        "recovery_stops": tm.recovery_stops,
        "escalations": tm.escalations,
        "timeout_job_failures": tm.timeout_job_failures,
        "events_count": tm.events_count,
    }


def _rate_block(passes: list[bool]) -> dict[str, Any]:
    n = len(passes)
    k = sum(passes)
    iv = wilson_interval(k, n)
    return {
        "passed": k,
        "total": n,
        "rate": round(k / n, 4) if n else 0.0,
        "wilson_ci_95": [round(iv.low, 4), round(iv.high, 4)],
    }


def _paired_continuous_block(on_vals: list[float], off_vals: list[float]) -> dict[str, Any]:
    """Paired mean difference (on - off) and 10,000-resample percentile bootstrap 95% CI."""
    diffs = np.asarray(on_vals, dtype=float) - np.asarray(off_vals, dtype=float)
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    drawn = rng.integers(0, len(diffs), size=(DEFAULT_RESAMPLES, len(diffs)))
    means = diffs[drawn].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": len(diffs),
        "mean_on": round(float(np.mean(on_vals)), 4),
        "mean_off": round(float(np.mean(off_vals)), 4),
        "mean_difference": round(float(diffs.mean()), 4),
        "bootstrap_ci_95": [round(float(low), 4), round(float(high), 4)],
        "resamples": DEFAULT_RESAMPLES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
    }


def _paired_block(on_outcomes: list[bool], off_outcomes: list[bool]) -> dict[str, Any]:
    tbl = paired_outcome_table(on_outcomes, off_outcomes)
    boot = paired_bootstrap_difference(
        on_outcomes, off_outcomes, resamples=DEFAULT_RESAMPLES, seed=DEFAULT_BOOTSTRAP_SEED
    )
    return {
        "paired_table": {
            "both_pass": tbl.both_pass,
            "planning_on_only": tbl.first_only,
            "planning_off_only": tbl.second_only,
            "both_fail": tbl.both_fail,
        },
        "paired_difference": round(boot.difference, 4),
        "bootstrap_ci_95": [round(boot.interval.low, 4), round(boot.interval.high, 4)],
        "verdict": boot.verdict.value,
        "resamples": DEFAULT_RESAMPLES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = latency_summary([r["wall_clock_seconds"] for r in rows])
    total_cost = sum(r["cost_usd"] for r in rows)
    with_plan = [r for r in rows if r["plan_before_first_tool"] is not None]
    plan_before_ok = sum(1 for r in with_plan if r["plan_before_first_tool"])
    return {
        "overall": _rate_block([r["passed"] for r in rows]),
        "by_fault_kind": {
            kind: _rate_block([r["passed"] for r in rows if r["fault_kind"] == kind])
            for kind in ("tool_error", "timeout", "malformed_data")
        },
        "recovery_cases_only": _rate_block(
            [r["passed"] for r in rows if r["fault_kind"] in ("tool_error", "timeout")]
        ),
        "latency_seconds": {
            "count": lat.count,
            "mean": round(lat.mean, 2),
            "p95": round(lat.p95, 2),
            "method": lat.method,
        },
        "cost_usd": {
            "total": round(total_cost, 6),
            "mean_per_case": round(total_cost / len(rows), 6) if rows else 0.0,
            "price_retrieved_on": PRICE_RETRIEVED_ON,
        },
        "mechanisms": {
            "plan_events_total": sum(r["plan_events"] for r in rows),
            "plan_before_first_tool_count": plan_before_ok,
            "plan_before_first_tool_eligible": len(with_plan),
            "plan_revisions_total": sum(r["plan_revisions"] for r in rows),
            "recovery_policy_decisions_total": sum(r["recovery_policy_decisions"] for r in rows),
            "recovery_retries_total": sum(r["recovery_retries"] for r in rows),
            "recovery_replans_total": sum(r["recovery_replans"] for r in rows),
            "recovery_stops_total": sum(r["recovery_stops"] for r in rows),
            "escalations_total": sum(r["escalations"] for r in rows),
        },
    }


def _load_clean_arm(arm_name: str) -> tuple[dict[str, CaseRecord], dict[str, list[CaseRecord]]]:
    base = WORK_DIR / arm_name / "clean_records"
    primaries: dict[str, CaseRecord] = {}
    repeats: dict[str, list[CaseRecord]] = {}
    pdir = base / "primary"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.json")):
            rec = CaseRecord.model_validate_json(p.read_text(encoding="utf-8"))
            primaries[rec.case_id] = rec
    rdir = base / "repeats"
    if rdir.is_dir():
        for cdir in sorted(rdir.iterdir()):
            if cdir.is_dir():
                repeats[cdir.name] = [
                    CaseRecord.model_validate_json(p.read_text(encoding="utf-8"))
                    for p in sorted(cdir.glob("*.json"))
                ]
    return primaries, repeats


def _collect_and_report(cases: dict[str, Case]) -> tuple[dict[str, Any], str]:
    on_prim, on_rep = _load_clean_arm("planning_on")
    off_prim, off_rep = _load_clean_arm("planning_off")

    reused_primary_ids = set(REUSED_TEST_CASES)
    for cid in REUSED_TEST_CASES:
        rec_path = FROZEN_DEFAULT_DIR / "agent" / f"{cid}.json"
        if rec_path.is_file():
            on_prim[cid] = CaseRecord.model_validate_json(rec_path.read_text(encoding="utf-8"))
    spambase_rep_dir = FROZEN_DEFAULT_DIR / "repeats" / "agent" / "test_fault_submit_spambase"
    if spambase_rep_dir.is_dir():
        on_rep["test_fault_submit_spambase"] = [
            CaseRecord.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(spambase_rep_dir.glob("*.json"))
        ]

    rows_on: list[dict[str, Any]] = []
    rows_off: list[dict[str, Any]] = []
    for cid in ALL_15_CASES:
        if cid not in on_prim:
            raise RuntimeError(f"Missing planning=on primary record for {cid}")
        if cid not in off_prim:
            raise RuntimeError(f"Missing planning=off primary record for {cid}")
        is_reused = cid in reused_primary_ids
        rows_on.append(_serialize_record(on_prim[cid], cases[cid], "planning_on", is_reused))
        rows_off.append(_serialize_record(off_prim[cid], cases[cid], "planning_off", False))

    rep_summary: dict[str, Any] = {}
    for arm_key, rep_map in (("planning_on", on_rep), ("planning_off", off_rep)):
        arm_rep_rows: list[dict[str, Any]] = []
        per_case: dict[str, Any] = {}
        for cid in REPEAT_CASES:
            recs = rep_map.get(cid, [])[:3]
            is_reused = arm_key == "planning_on" and cid == "test_fault_submit_spambase"
            serialized = [_serialize_record(r, cases[cid], arm_key, is_reused) for r in recs]
            arm_rep_rows.extend(serialized)
            per_case[cid] = {
                "runs": len(serialized),
                "passes": sum(1 for x in serialized if x["passed"]),
                "reused_from_s52": is_reused,
                "run_ids": [x["run_id"] for x in serialized],
            }
        lat_r = latency_summary([x["wall_clock_seconds"] for x in arm_rep_rows])
        rep_summary[arm_key] = {
            "total_runs": len(arm_rep_rows),
            "overall": _rate_block([x["passed"] for x in arm_rep_rows]),
            "latency_mean_s": round(lat_r.mean, 2),
            "latency_p95_s": round(lat_r.p95, 2),
            "total_cost_usd": round(sum(x["cost_usd"] for x in arm_rep_rows), 6),
            "per_case": per_case,
        }

    paired_all = _paired_block([r["passed"] for r in rows_on], [r["passed"] for r in rows_off])
    paired_by_kind = {
        kind: _paired_block(
            [r["passed"] for r in rows_on if r["fault_kind"] == kind],
            [r["passed"] for r in rows_off if r["fault_kind"] == kind],
        )
        for kind in ("tool_error", "timeout", "malformed_data")
    }
    paired_recovery = _paired_block(
        [r["passed"] for r in rows_on if r["fault_kind"] in ("tool_error", "timeout")],
        [r["passed"] for r in rows_off if r["fault_kind"] in ("tool_error", "timeout")],
    )

    traces: list[dict[str, Any]] = []
    trace_candidates = [
        ("recovered_tool_error_planning_on", "tool_error", True, "planning_on", rows_on, "on"),
        ("recovered_timeout_planning_on", "timeout", True, "planning_on", rows_on, "on"),
        (
            "safe_stop_malformed_data_planning_on",
            "malformed_data",
            True,
            "planning_on",
            rows_on,
            "on",
        ),
        ("unrecovered_fault_planning_on", None, False, "planning_on", rows_on, "on"),
        ("unrecovered_fault_planning_off", None, False, "planning_off", rows_off, "off"),
    ]
    for label, kind, want_pass, arm_key, arm_rows, flag_str in trace_candidates:
        hit = next(
            (
                r
                for r in arm_rows
                if r["passed"] is want_pass and (kind is None or r["fault_kind"] == kind)
            ),
            None,
        )
        if hit:
            reused_flag = hit["reused_from"] is not None
            tr = _extract_sanitized_trace(
                _locate_db(hit["run_id"], arm_key, reused_flag),
                hit["run_id"],
                label,
                hit["case_id"],
                flag_str,
            )
            if tr:
                traces.append(tr)

    live_cost = (
        sum(r["cost_usd"] for r in rows_on if r["reused_from"] is None)
        + sum(r["cost_usd"] for r in rows_off)
        + rep_summary["planning_off"]["total_cost_usd"]
        + sum(
            _serialize_record(r, cases[cid], "planning_on", False)["cost_usd"]
            for cid in NEW_REPEAT_CASES_ON
            for r in on_rep.get(cid, [])[:3]
        )
    )
    spambase_reps = on_rep.get("test_fault_submit_spambase", [])[:3]
    reused_cost = sum(r["cost_usd"] for r in rows_on if r["reused_from"] is not None) + sum(
        _serialize_record(r, cases["test_fault_submit_spambase"], "planning_on", True)["cost_usd"]
        for r in spambase_reps
    )

    on_sum = _arm_summary(rows_on)
    off_sum = _arm_summary(rows_off)
    evidence: dict[str, Any] = {
        "feature": "F4",
        "title": "Planning & Error Recovery",
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "code_commit_sha": GitRepo(REPO_ROOT).state().commit,
            "frozen_s52_commit": FROZEN_COMMIT,
            "reused_primary_cases": list(REUSED_TEST_CASES),
            "reused_repeat_cases": ["test_fault_submit_spambase"],
            "model_id": "gpt-6-luna",
            "agent_mode": "multi",
            "price_retrieved_on": PRICE_RETRIEVED_ON,
            "cap_usd": CAP_USD,
            "new_live_spend_usd": round(live_cost, 6),
            "reused_s52_spend_usd": round(reused_cost, 6),
        },
        "arms": {
            "planning_on": on_sum,
            "planning_off": off_sum,
        },
        "paired_comparisons": {
            "all_15": paired_all,
            "recovery_only_10": paired_recovery,
            "by_fault_kind": paired_by_kind,
            "latency_seconds": {
                "all_15": _paired_continuous_block(
                    [r["wall_clock_seconds"] for r in rows_on],
                    [r["wall_clock_seconds"] for r in rows_off],
                ),
                "recovery_only_10": _paired_continuous_block(
                    [
                        r["wall_clock_seconds"]
                        for r in rows_on
                        if r["fault_kind"] in ("tool_error", "timeout")
                    ],
                    [
                        r["wall_clock_seconds"]
                        for r in rows_off
                        if r["fault_kind"] in ("tool_error", "timeout")
                    ],
                ),
            },
            "cost_usd": {
                "all_15": _paired_continuous_block(
                    [r["cost_usd"] for r in rows_on],
                    [r["cost_usd"] for r in rows_off],
                ),
                "recovery_only_10": _paired_continuous_block(
                    [
                        r["cost_usd"]
                        for r in rows_on
                        if r["fault_kind"] in ("tool_error", "timeout")
                    ],
                    [
                        r["cost_usd"]
                        for r in rows_off
                        if r["fault_kind"] in ("tool_error", "timeout")
                    ],
                ),
            },
        },
        "per_case_primary": {
            "planning_on": rows_on,
            "planning_off": rows_off,
        },
        "repeat_sample_6x3": rep_summary,
        "representative_traces": traces,
    }

    md = _render_markdown(evidence)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "f4-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "f4-summary.md").write_text(md, encoding="utf-8")
    return evidence, md


def _fmt_rate_ci(block: dict[str, Any]) -> str:
    rate_pct = block["rate"] * 100
    ci = block["wilson_ci_95"]
    return f"{block['passed']}/{block['total']} ({rate_pct:.1f}%, CI {ci})"


def _render_markdown(ev: dict[str, Any]) -> str:
    prov = ev["provenance"]
    on_a = ev["arms"]["planning_on"]
    off_a = ev["arms"]["planning_off"]
    p_all = ev["paired_comparisons"]["all_15"]
    p_rec = ev["paired_comparisons"]["recovery_only_10"]
    short_s52 = prov["frozen_s52_commit"][:7]
    lines = [
        "# F4 Planning & Error Recovery - Evaluation Summary",
        "",
        f"- **Generated:** {ev['generated_at']}",
        f"- **Model:** `{prov['model_id']}` (`agent_mode=multi`)",
        f"- **Frozen S5.2 reused records:** {len(prov['reused_primary_cases'])} primary + 3 repeats"
        f" (`{short_s52}`)",
        f"- **New live F4 spend:** `${prov['new_live_spend_usd']:.5f}` (cap `${CAP_USD:.2f}`)",
        "",
        "## Primary Paired Comparison (15 Fault Cases)",
        "",
        "| Slice | `planning=on` | `planning=off` | Paired Diff | 95% Boot CI | Both/On/Off/Fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    slices = [
        ("All 15 fault cases", on_a["overall"], off_a["overall"], p_all),
        ("Recovery cases (n=10)", on_a["recovery_cases_only"], off_a["recovery_cases_only"], p_rec),
        (
            "Tool errors (n=5)",
            on_a["by_fault_kind"]["tool_error"],
            off_a["by_fault_kind"]["tool_error"],
            ev["paired_comparisons"]["by_fault_kind"]["tool_error"],
        ),
        (
            "Timeouts (n=5)",
            on_a["by_fault_kind"]["timeout"],
            off_a["by_fault_kind"]["timeout"],
            ev["paired_comparisons"]["by_fault_kind"]["timeout"],
        ),
        (
            "Malformed data (n=5)",
            on_a["by_fault_kind"]["malformed_data"],
            off_a["by_fault_kind"]["malformed_data"],
            ev["paired_comparisons"]["by_fault_kind"]["malformed_data"],
        ),
    ]
    for label, s_on, s_off, pr in slices:
        pt = pr["paired_table"]
        both, on_only, off_only, fail = (
            pt["both_pass"],
            pt["planning_on_only"],
            pt["planning_off_only"],
            pt["both_fail"],
        )
        lines.append(
            f"| {label} | {_fmt_rate_ci(s_on)} | {_fmt_rate_ci(s_off)} "
            f"| {pr['paired_difference']:+.3f} | {pr['bootstrap_ci_95']} "
            f"| {both} / {on_only} / {off_only} / {fail} |"
        )
    on_m = on_a["mechanisms"]
    off_m = off_a["mechanisms"]
    on_rec_str = (
        f"{on_m['recovery_policy_decisions_total']} "
        f"({on_m['recovery_retries_total']}/"
        f"{on_m['recovery_replans_total']}/"
        f"{on_m['recovery_stops_total']})"
    )
    on_lat = f"{on_a['latency_seconds']['mean']:.1f} / {on_a['latency_seconds']['p95']:.1f}"
    off_lat = f"{off_a['latency_seconds']['mean']:.1f} / {off_a['latency_seconds']['p95']:.1f}"
    on_cost = f"${on_a['cost_usd']['total']:.5f} / ${on_a['cost_usd']['mean_per_case']:.5f}"
    lat15 = ev["paired_comparisons"]["latency_seconds"]["all_15"]
    lat10 = ev["paired_comparisons"]["latency_seconds"]["recovery_only_10"]
    lat15_s = f"{lat15['mean_difference']:+.1f} CI {lat15['bootstrap_ci_95']}"
    lat10_s = f"{lat10['mean_difference']:+.1f} CI {lat10['bootstrap_ci_95']}"
    off_cost = f"${off_a['cost_usd']['total']:.5f} / ${off_a['cost_usd']['mean_per_case']:.5f}"
    rep_on = ev["repeat_sample_6x3"]["planning_on"]
    rep_off = ev["repeat_sample_6x3"]["planning_off"]
    lines.extend(
        [
            "",
            "## Planning & Recovery Mechanisms + Cost/Latency",
            "",
            "| Metric | `planning=on` | `planning=off` |",
            "|---|---:|---:|",
            f"| Plan before first tool call | {on_m['plan_before_first_tool_count']}/"
            f"{on_m['plan_before_first_tool_eligible']} | "
            f"{off_m['plan_before_first_tool_count']}/"
            f"{off_m['plan_before_first_tool_eligible']} |",
            f"| Total `plan` events | {on_m['plan_events_total']} | {off_m['plan_events_total']} |",
            f"| Plan revisions (`revises_plan_id`) | {on_m['plan_revisions_total']} | "
            f"{off_m['plan_revisions_total']} |",
            f"| Recovery decisions (retry/replan/stop) | {on_rec_str} | "
            f"{off_m['recovery_policy_decisions_total']} |",
            f"| Latency mean / p95 (s) | {on_lat} | {off_lat} |",
            f"| Paired latency delta all-15 (s) | {lat15_s} | - |",
            f"| Paired latency delta recovery-10 (s) | {lat10_s} | - |",
            f"| Total / mean cost (USD) | {on_cost} | {off_cost} |",
            "",
            "## Separate 6x3 Repeat Sample (n=18 per arm)",
            "",
            f"- `planning=on`: {_fmt_rate_ci(rep_on['overall'])}",
            f"- `planning=off`: {_fmt_rate_ci(rep_off['overall'])}",
            "",
            "## Per-Case Results",
            "",
            "| Case | Kind | On | Off | On Plans/Rev | On Cost / Lat | Off Cost / Lat |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    by_off = {r["case_id"]: r for r in ev["per_case_primary"]["planning_off"]}
    for r_on in ev["per_case_primary"]["planning_on"]:
        r_off = by_off[r_on["case_id"]]
        suffix = " (S5.2)" if r_on["reused_from"] else ""
        tag_on = ("PASS" if r_on["passed"] else "FAIL") + suffix
        tag_off = "PASS" if r_off["passed"] else "FAIL"
        lines.append(
            f"| `{r_on['case_id']}` | {r_on['fault_kind']} | {tag_on} | {tag_off} "
            f"| {r_on['plan_events']}/{r_on['plan_revisions']} "
            f"| ${r_on['cost_usd']:.4f} / {r_on['wall_clock_seconds']:.0f}s "
            f"| ${r_off['cost_usd']:.4f} / {r_off['wall_clock_seconds']:.0f}s |"
        )
    lines.append("")
    lines.append("## Representative Sanitized Fault Traces")
    lines.append("")
    for tr in ev["representative_traces"]:
        lines.append(
            f"### {tr['label']} (`{tr['case_id']}`, `{tr['run_id']}`, "
            f"`planning={tr['planning_flag']}`, {tr['total_events']} events)"
        )
        lines.append("")
        lines.append("| Seq | Role | Event | Summary |")
        lines.append("|---:|---|---|---|")
        for item in tr["key_events"]:
            lines.append(
                f"| {item['seq']} | {item['role']} | `{item['type']}` | {item['summary']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    cases = load_all_15_cases()
    all_case_tuple = tuple(cases[cid] for cid in ALL_15_CASES)
    if not args.report_only:
        tracker = SpendTracker(CAP_USD, WORK_DIR / "spend-journal.jsonl")
        new_on_prim = tuple(cases[cid] for cid in NEW_F4_CASES)
        await _run_arm(
            "planning_on",
            Toggle.ON,
            new_on_prim,
            NEW_REPEAT_CASES_ON,
            cases,
            tracker,
            parallel=args.parallel,
        )
        await _run_arm(
            "planning_off",
            Toggle.OFF,
            all_case_tuple,
            REPEAT_CASES,
            cases,
            tracker,
            parallel=args.parallel,
        )
    _evidence, md = _collect_and_report(cases)
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S5.3 F4 planning and recovery evaluation.")
    parser.add_argument(
        "--report-only", action="store_true", help="Only rebuild reports from saved records."
    )
    parser.add_argument(
        "--parallel", type=int, default=MAX_PARALLEL_CASES, help="Max parallel cases."
    )
    args = parser.parse_args()
    anyio.run(async_main, args)


if __name__ == "__main__":
    main()
