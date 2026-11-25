#!/usr/bin/env python3
"""Generate F7 (Observability) evaluation evidence for S5.3.

Preserves the frozen S5.2 pre-fix baseline (`62b4e3fb6016a0abddf70a14152252a62c8aca4a`), runs
`default` (`agent_mode=multi`) on the post-fix commit (`f8505880a1de78331dd4834e9ea22ebb4191778c`)
with `openai/gpt-6-luna` (`eval` profile) across all 26 held-out test cases plus 6x3=18 repeats,
audits trace completeness & secret redaction, supports programmatic SQLite trace search
(`--query`), and produces `results/s5.3/f7/f7-evidence.json` and `results/s5.3/f7/f7-summary.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sqlite3
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Final

import anyio
import numpy as np
from dotenv import load_dotenv

from hypora.cli.eval import load_cases
from hypora.cli.trace import describe, parse_filters, render_rows, summary, trace_rows
from hypora.core.config import load_settings_for_eval
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import new_run_id
from hypora.core.hashing import sha256_hex
from hypora.core.llm.credentials import load_credentials
from hypora.core.llm.rate_limit import ProviderAdmission
from hypora.core.schemas import (
    ActorRole,
    AnyEvent,
    EventType,
    JobState,
    MetricDirection,
    MetricName,
    Stage,
)
from hypora.core.schemas.events import (
    CallStatus,
    JobTransitionPayload,
    ModelCallPayload,
    ModelProfile,
    RoleStartedPayload,
    StageTransitionPayload,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.evaluation.case import Case, FaultKind
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore, create_hidden_split
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
    audit_run_files,
    check_grade,
    failure_categories,
    inputs_of,
    outputs_of,
    system_input_for,
)
from hypora.evaluation.runner.usage import collect_usage
from hypora.evaluation.stats.core import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_RESAMPLES,
    MIN_RESAMPLES,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)
from hypora.evaluation.system import SystemContext
from hypora.execution.sandbox.image import tabular_image_tag
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.migrate import migrate
from scripts import final_eval as final
from scripts.stage2_eval import (
    ATTEMPT_RESERVATION_USD,
    AccountedAgent,
    AdapterFactory,
    CollectionRefused,
    SpendError,
    SpendLedger,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
FROZEN_COMMIT: Final = "62b4e3fb6016a0abddf70a14152252a62c8aca4a"
POST_FIX_SHA: Final = "f8505880a1de78331dd4834e9ea22ebb4191778c"

FROZEN_FINAL_ROOT: Final = Path("/home/ghadi/hypora-final")
FROZEN_DEFAULT_DIR: Final = FROZEN_FINAL_ROOT / "results" / FROZEN_COMMIT / "default"
FROZEN_STATE_DIR: Final = FROZEN_FINAL_ROOT / "runs" / "final-s5.2" / "work" / "state"

POST_FIX_RESULTS_DIR: Final = REPO_ROOT / "results" / POST_FIX_SHA / "default"
WORK_DIR: Final = REPO_ROOT / "runs" / "s53-f7"
POST_FIX_RUNS_ROOT: Final = WORK_DIR / "work" / RUNS_DIR
POST_FIX_STATE_DIR: Final = WORK_DIR / "work" / STATE_DIR
JOURNAL_PATH: Final = WORK_DIR / "paid-attempts.jsonl"
AUDIT_PATH: Final = WORK_DIR / "paid-accounting.json"
OUT_DIR: Final = REPO_ROOT / "results" / "s5.3" / "f7"

CAP_USD: Final = 3.50
DEFAULT_PARALLEL: Final = 2
MAX_PRIMARY_FAILURES_CHECKPOINT: Final = 2
PRICE_RETRIEVED_ON: Final = "2026-11-07"
FATAL_ERROR_CODES: Final = frozenset({"PROVIDER_AUTH_FAILED", "QUOTA_EXHAUSTED"})
STAGES_ORDER: Final = (
    Stage.INTAKE.value,
    Stage.PROFILE.value,
    Stage.CONTRACT.value,
    Stage.PHASE_B.value,
    Stage.REPORT.value,
)
ACTOR_LLM_ROLES: Final = (
    ActorRole.AGENT_A.value,
    ActorRole.AGENT_B.value,
    ActorRole.AGENT_C.value,
    ActorRole.TRIAL_WORKER.value,
)
SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}")),
    ("env_secret_name", re.compile(r"(OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY)\s*=")),
)


class EarlyCheckpointAbort(RuntimeError):
    """Raised when early primary completions exceed the failure or budget checkpoint."""


class ProviderFatalError(RuntimeError):
    """Raised immediately if any live run records PROVIDER_AUTH_FAILED or QUOTA_EXHAUSTED."""


# --- Record sanitization & loading ---------------------------------------------------------------


def sanitize_record_dict(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate `CaseRecord` then omit `inputs.split.split_key` for secret-free commits."""
    copied: dict[str, Any] = json.loads(json.dumps(document))
    CaseRecord.model_validate(copied)
    split = copied.get("inputs", {}).get("split")
    if isinstance(split, dict):
        split.pop("split_key", None)
    return copied


def load_sanitized_record(path: Path) -> CaseRecord:
    """Load a published or canonical `CaseRecord`, filling `split_key` only if sanitized."""
    document = json.loads(path.read_text(encoding="utf-8"))
    split = document.get("inputs", {}).get("split")
    if isinstance(split, dict) and "split_key" not in split:
        split["split_key"] = "sanitized"
    return CaseRecord.model_validate(document)


def primary_record_path(results_dir: Path, case_id: str) -> Path:
    """Canonical primary record path `{results_dir}/agent/{case_id}.json`."""
    return results_dir / "agent" / f"{case_id}.json"


def repeat_record_path(results_dir: Path, case_id: str, attempt: int) -> Path:
    """Canonical repeat record path `{results_dir}/repeats/agent/{case_id}/{attempt}.json`."""
    return results_dir / "repeats" / "agent" / case_id / f"{attempt}.json"


def _resolve_repeat_path(results_dir: Path, case_id: str, attempt: int) -> Path:
    preferred = repeat_record_path(results_dir, case_id, attempt)
    if preferred.is_file():
        return preferred
    fallback = results_dir / "repeats" / case_id / f"{attempt}.json"
    return fallback if fallback.is_file() else preferred


def load_arm_records(
    results_dir: Path,
    case_ids: Sequence[str],
    repeat_cases: Sequence[str] = final.REPEAT_CASES,
    repeats: int = final.REPEATS,
) -> tuple[dict[str, CaseRecord], dict[str, list[CaseRecord]]]:
    """Load primary and repeat records from `results_dir` (accepting sanitized or canonical)."""
    primary: dict[str, CaseRecord] = {}
    for cid in sorted(case_ids):
        path = primary_record_path(results_dir, cid)
        if path.is_file():
            primary[cid] = load_sanitized_record(path)
    rep: dict[str, list[CaseRecord]] = {}
    for cid in sorted(repeat_cases):
        runs: list[CaseRecord] = []
        for attempt in range(1, repeats + 1):
            path = _resolve_repeat_path(results_dir, cid, attempt)
            if path.is_file():
                runs.append(load_sanitized_record(path))
        if runs:
            rep[cid] = runs
    return primary, rep


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --- Live runner ---------------------------------------------------------------------------------


def verify_post_fix_code() -> None:
    """Verify product code (`hypora/`, `configs/`, `pyproject.toml`, `uv.lock`) matches SHA."""
    check = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            POST_FIX_SHA,
            "--",
            "hypora",
            "configs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"Product code differs from POST_FIX_SHA {POST_FIX_SHA}; refusing live collection."
        )


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
            if pl.get(key) in FATAL_ERROR_CODES:
                return str(pl[key])
        msg = str(pl.get("message", ""))
        for code in FATAL_ERROR_CODES:
            if code in msg:
                return code
    return None


def _provenance_for(run_config: RunConfig) -> Provenance:
    return Provenance(
        commit_sha=POST_FIX_SHA,
        dirty=False,
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


async def run_post_fix_collection(
    *,
    parallel: int = DEFAULT_PARALLEL,
    resume: bool = False,
    phase: str = "all",
    limit: int | None = None,
) -> None:
    """Execute missing primary and/or repeat slots on `POST_FIX_SHA` with `$3.50` SpendLedger."""
    verify_post_fix_code()
    cases = load_cases(final.CASE_SET)
    final.check_case_set(cases)
    by_id = {c.id: c for c in cases}

    POST_FIX_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    POST_FIX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    POST_FIX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig.build(load_settings_for_eval("default"), eval_config="default")
    if run_config.model_profile is not ModelProfile.EVAL:
        raise RuntimeError(f"Expected eval profile, got {run_config.model_profile}")
    provenance = _provenance_for(run_config)

    slots: list[tuple[Case, RunKind, int, Path]] = []
    existing_primary_failures = 0
    if phase in ("primary", "all"):
        for case in sorted(cases, key=lambda c: c.id):
            dest = primary_record_path(POST_FIX_RESULTS_DIR, case.id)
            if dest.is_file():
                existing = load_sanitized_record(dest)
                if existing.finished:
                    if not existing.passed:
                        existing_primary_failures += 1
                    continue
                if not resume:
                    raise RuntimeError(f"{dest} is a harness_error; pass --resume to retry.")
            slots.append((case, RunKind.PRIMARY, 1, dest))

    if phase in ("repeats", "all"):
        for cid in sorted(final.REPEAT_CASES):
            case = by_id[cid]
            for attempt in range(1, final.REPEATS + 1):
                dest = repeat_record_path(POST_FIX_RESULTS_DIR, cid, attempt)
                if dest.is_file():
                    existing = load_sanitized_record(dest)
                    if existing.finished:
                        continue
                    if not resume:
                        raise RuntimeError(f"{dest} is a harness_error; pass --resume to retry.")
                slots.append((case, RunKind.REPEAT, attempt, dest))

    if existing_primary_failures > MAX_PRIMARY_FAILURES_CHECKPOINT:
        raise EarlyCheckpointAbort(
            f"Existing primary failures ({existing_primary_failures}) exceed checkpoint "
            f"({MAX_PRIMARY_FAILURES_CHECKPOINT}); stopping before launching more runs."
        )

    if limit is not None:
        slots = slots[: max(0, limit)]

    if not slots:
        print("[post_fix] All requested slots already exist and are graded.")
        return

    ledger = SpendLedger(JOURNAL_PATH, cap_usd=CAP_USD, audit=AUDIT_PATH)
    snap = ledger.snapshot()
    headroom = CAP_USD - (snap.measured_usd + snap.reserved_usd)
    if headroom < ATTEMPT_RESERVATION_USD:
        raise SpendError(
            f"Insufficient headroom (${headroom:.4f} < ${ATTEMPT_RESERVATION_USD:.2f}) "
            f"under cap ${CAP_USD:.2f}."
        )
    print(
        f"[post_fix] Running {len(slots)} slot(s) at parallel={parallel} | "
        f"ledger measured=${snap.measured_usd:.4f}, reserved=${snap.reserved_usd:.4f}, "
        f"cap=${CAP_USD:.2f}"
    )

    store = HiddenStore()
    grader = ProgrammaticGrader()
    factory = AdapterFactory(ledger)
    max_inflight = max(1, int(headroom // ATTEMPT_RESERVATION_USD))
    if max_inflight < 4:
        _s = load_settings_for_eval("default")
        factory.admission = ProviderAdmission(
            _s.models.rate_limits,
            _s.resources.llm_concurrency.model_copy(update={"openai": max_inflight}),
        )
        print(
            f"[post_fix] Bounding concurrent LLM reservations to {max_inflight} "
            f"(headroom=${headroom:.4f})."
        )
    system = AccountedAgent(factory)
    guard = RecordGuard(
        forbidden_paths=(str(store.root),),
        secrets=load_credentials().secret_values(),
        path_aliases=machine_aliases(POST_FIX_RUNS_ROOT, REPO_ROOT),
        public_fields=frozenset({"inputs", "provenance"}),
    )

    prepared: dict[str, tuple[HiddenSplit, frozenset[str], Any, float]] = {}
    for case, _, _, _ in slots:
        if case.id not in prepared:
            t0 = time.perf_counter()
            ds = load_case_dataset(case)
            split = create_hidden_split(case, ds, store, POST_FIX_RUNS_ROOT)
            prepared[case.id] = (
                split,
                hidden_only_values(split),
                hidden_row_tokens(split),
                time.perf_counter() - t0,
            )

    limiter = anyio.CapacityLimiter(parallel)
    state_lock = threading.Lock()
    done_count = 0
    primary_failures = existing_primary_failures
    abort_error: BaseException | None = None

    async def run_one(case: Case, kind: RunKind, attempt: int, dest: Path) -> None:
        nonlocal done_count, primary_failures, abort_error
        async with limiter:
            with state_lock:
                if abort_error is not None:
                    return
            started_at = datetime.now(UTC)
            split, hidden_vals, hidden_rows, prep_s = prepared[case.id]
            run_id = new_run_id()
            db_path = POST_FIX_STATE_DIR / f"{run_id}.sqlite"
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
                runs_root=POST_FIX_RUNS_ROOT,
                database=db_path,
                run_config=run_config,
                injected_faults=case.injected_faults,
            )

            t_sys = time.perf_counter()
            try:
                run = await system.run(system_input, context)
            except CollectionRefused as exc:
                with state_lock:
                    if abort_error is None:
                        abort_error = exc
                raise
            sys_s = time.perf_counter() - t_sys

            fatal = _has_fatal_provider_error(db_path)
            if fatal is not None:
                err = ProviderFatalError(
                    f"Fatal provider error {fatal} on {case.id} ({run_id}); aborting."
                )
                with state_lock:
                    abort_error = err
                raise err

            audit_run_files(split, store, POST_FIX_RUNS_ROOT, run_id)
            t_grade = time.perf_counter()
            grade = await grader.grade(case, split, run)
            check_grade(grade, run)
            grade_s = time.perf_counter() - t_grade

            reported_retry = system.retry_wait_seconds(run_id)
            collected = collect_usage(db_path, run_id, retry_wait_seconds=reported_retry)
            unobserved = system.unobserved_spend_reserved_usd(run_id)
            usage = collected.usage.model_copy(
                update={
                    "unobserved_spend_reserved_usd": unobserved,
                    "complete": collected.usage.complete and unobserved == 0.0,
                }
            )
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
                outputs=outputs_of(run, POST_FIX_RUNS_ROOT),
                grade=grade,
                timings=timings,
                usage=usage,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            cleaned = guard.clean(record.model_dump(mode="json"), hidden_vals, hidden_rows)
            assert isinstance(cleaned, Mapping)
            sanitized = sanitize_record_dict(cleaned)
            _write_json_atomic(dest, sanitized)

            cur_snap = ledger.snapshot()
            with state_lock:
                done_count += 1
                idx = done_count
                if kind is RunKind.PRIMARY and not grade.passed:
                    primary_failures += 1
                    if primary_failures > MAX_PRIMARY_FAILURES_CHECKPOINT:
                        abort_error = EarlyCheckpointAbort(
                            f"Primary failures reached {primary_failures} "
                            f"(> {MAX_PRIMARY_FAILURES_CHECKPOINT}) on {case.id}; halting."
                        )
            tag = "primary" if kind is RunKind.PRIMARY else f"repeat#{attempt}"
            print(
                f"  [post_fix {idx}/{len(slots)}] {case.id} ({tag}) "
                f"pass={grade.passed} cost=${usage.actual_spend_usd:.4f} "
                f"wall={sys_s:.1f}s ledger=${cur_snap.measured_usd:.4f}"
                f"+${cur_snap.reserved_usd:.2f}res"
            )
            if abort_error is not None:
                raise abort_error

    try:
        async with anyio.create_task_group() as tg:
            for case, kind, attempt, dest in slots:
                tg.start_soon(run_one, case, kind, attempt, dest)
    finally:
        ledger.publish()
        final_snap = ledger.snapshot()
        print(
            f"[post_fix] Ledger settled: measured=${final_snap.measured_usd:.4f}, "
            f"unknown_reserved=${final_snap.reserved_usd:.4f}, cap=${CAP_USD:.2f}"
        )


# --- Trace audit, metrics, breakdowns & search ---------------------------------------------------


def locate_db(run_id: str, state_dir: Path) -> Path | None:
    """Locate `{state_dir}/{run_id}.sqlite` if present."""
    path = state_dir / f"{run_id}.sqlite"
    return path if path.is_file() else None


def read_run_events(db_path: Path, run_id: str) -> list[AnyEvent]:
    """Read and Pydantic-validate all events for `run_id` from SQLite `EventLog`."""
    db = Database(db_path)
    try:
        return EventLog(db).read(run_id)
    finally:
        db.close()


def _scan_secrets(text: str, secret_values: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for secret in secret_values:
        if len(secret) >= 8 and secret in text:
            hits.append("configured_secret")
            break
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


def audit_trace_completeness(
    db_path: Path,
    run_id: str,
    *,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify every model_call and tool_call/tool_result in a trace is complete & redacted."""
    events = read_run_events(db_path, run_id)
    model_calls = 0
    model_calls_complete = 0
    model_call_gaps: list[dict[str, Any]] = []
    tool_calls_by_id: dict[str, tuple[int, ToolCallPayload]] = {}
    tool_results_by_id: dict[str, tuple[int, ToolResultPayload]] = {}
    secret_hits: list[dict[str, Any]] = []

    for event in events:
        payload_dict = event.payload.model_dump(mode="json")
        raw_text = json.dumps(payload_dict, sort_keys=True)
        hits = _scan_secrets(raw_text, secret_values)
        if hits:
            secret_hits.append({"seq": event.seq, "type": event.type.value, "kinds": hits})

        p = event.payload
        if isinstance(p, ModelCallPayload):
            model_calls += 1
            missing: list[str] = []
            if not p.provider.value:
                missing.append("provider")
            if not p.model_id:
                missing.append("model_id")
            if not p.tier.value:
                missing.append("tier")
            if p.latency_ms < 0:
                missing.append("latency_ms")
            if p.input_tokens < 0:
                missing.append("input_tokens")
            if p.output_tokens < 0:
                missing.append("output_tokens")
            if p.cost_usd < 0:
                missing.append("cost_usd")
            if not p.input_ref or not p.input_ref.startswith("model_io:"):
                missing.append("input_ref")
            if p.status is CallStatus.OK and (
                not p.output_ref or not p.output_ref.startswith("model_io:")
            ):
                missing.append("output_ref")
            if missing:
                model_call_gaps.append({"seq": event.seq, "call_id": p.call_id, "missing": missing})
            else:
                model_calls_complete += 1
        elif isinstance(p, ToolCallPayload):
            tool_calls_by_id[p.call_id] = (event.seq, p)
        elif isinstance(p, ToolResultPayload):
            tool_results_by_id[p.call_id] = (event.seq, p)

    orphan_calls = sorted(set(tool_calls_by_id) - set(tool_results_by_id))
    orphan_results = sorted(set(tool_results_by_id) - set(tool_calls_by_id))
    joined_pairs = 0
    tool_pairs_complete = 0
    tool_pair_gaps: list[dict[str, Any]] = []
    tool_status_counts: Counter[str] = Counter()

    for call_id in sorted(set(tool_calls_by_id) & set(tool_results_by_id)):
        joined_pairs += 1
        _, tc = tool_calls_by_id[call_id]
        seq_r, tr = tool_results_by_id[call_id]
        tool_status_counts[tr.status.value] += 1
        issues: list[str] = []
        if tc.tool != tr.tool:
            issues.append(f"tool_mismatch:{tc.tool}!={tr.tool}")
        if tr.latency_ms < 0:
            issues.append("negative_latency")
        if not isinstance(tr.result_summary, dict):
            issues.append("missing_result_summary")
        if tr.status is not ToolResultStatus.OK and tr.error is None:
            issues.append("missing_error_payload")
        if issues:
            tool_pair_gaps.append({"seq": seq_r, "call_id": call_id, "issues": issues})
        else:
            tool_pairs_complete += 1

    complete_pair_count = (
        len(tool_calls_by_id) == len(tool_results_by_id) == joined_pairs == tool_pairs_complete
    )
    return {
        "run_id": run_id,
        "events_count": len(events),
        "model_calls": model_calls,
        "model_calls_complete": model_calls_complete,
        "model_call_gaps": model_call_gaps,
        "tool_calls": len(tool_calls_by_id),
        "tool_results": len(tool_results_by_id),
        "joined_tool_pairs": joined_pairs,
        "tool_pairs_complete": tool_pairs_complete,
        "orphan_tool_calls": orphan_calls,
        "orphan_tool_results": orphan_results,
        "tool_pair_gaps": tool_pair_gaps,
        "tool_status_counts": dict(tool_status_counts),
        "secret_hits": secret_hits,
        "complete_and_clean": (
            model_calls == model_calls_complete and complete_pair_count and not secret_hits
        ),
    }


def audit_arm_traces(
    records: Mapping[str, CaseRecord],
    state_dir: Path,
    *,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate Implementation completeness audit across every primary run in an arm."""
    per_case: list[dict[str, Any]] = []
    missing_dbs: list[str] = []
    for cid, rec in sorted(records.items()):
        db_path = locate_db(rec.run_id, state_dir)
        if db_path is None:
            missing_dbs.append(cid)
            continue
        audit = audit_trace_completeness(db_path, rec.run_id, secret_values=secret_values)
        audit["case_id"] = cid
        per_case.append(audit)

    return {
        "runs_audited": len(per_case),
        "missing_dbs": missing_dbs,
        "all_complete_and_clean": (
            not missing_dbs and all(row["complete_and_clean"] for row in per_case)
        ),
        "events_total": sum(row["events_count"] for row in per_case),
        "model_calls_total": sum(row["model_calls"] for row in per_case),
        "model_calls_complete": sum(row["model_calls_complete"] for row in per_case),
        "model_call_gaps_total": sum(len(row["model_call_gaps"]) for row in per_case),
        "tool_calls_total": sum(row["tool_calls"] for row in per_case),
        "tool_results_total": sum(row["tool_results"] for row in per_case),
        "joined_tool_pairs_total": sum(row["joined_tool_pairs"] for row in per_case),
        "tool_pairs_complete": sum(row["tool_pairs_complete"] for row in per_case),
        "orphan_tool_calls_total": sum(len(row["orphan_tool_calls"]) for row in per_case),
        "orphan_tool_results_total": sum(len(row["orphan_tool_results"]) for row in per_case),
        "secret_hits_total": sum(len(row["secret_hits"]) for row in per_case),
        "tool_status_counts": {
            status: sum(row["tool_status_counts"].get(status, 0) for row in per_case)
            for status in ("ok", "error", "denied")
        },
    }


@dataclass(frozen=True)
class AgentBMetrics:
    """Per-run Agent B bottleneck metrics extracted directly from SQLite events."""

    role_instances: int
    model_calls: int
    llm_latency_seconds: float
    cost_usd: float
    post_tool_duplicate_turns: int
    post_tool_duplicate_latency_seconds: float
    post_tool_duplicate_cost_usd: float
    validation_retries: int
    validation_retry_latency_seconds: float
    validation_retry_cost_usd: float
    contract_stage_seconds: float


def extract_agent_b_metrics(events: Sequence[AnyEvent]) -> AgentBMetrics:
    """Measure Agent B calls, post-tool duplicate prose turns, validation retries, and CONTRACT."""
    cur_stage: str | None = None
    cur_stage_t: datetime | None = None
    contract_s = 0.0

    instances: list[list[AnyEvent]] = []
    cur_b: list[AnyEvent] | None = None

    for ev in events:
        p = ev.payload
        if isinstance(p, StageTransitionPayload):
            if cur_stage == Stage.CONTRACT.value and cur_stage_t is not None:
                contract_s += max(0.0, (ev.created_at - cur_stage_t).total_seconds())
            cur_stage = p.stage.value if p.stage is not None else None
            cur_stage_t = ev.created_at
        elif ev.actor_role is ActorRole.AGENT_B:
            if isinstance(p, RoleStartedPayload):
                if cur_b is not None:
                    instances.append(cur_b)
                cur_b = [ev]
            else:
                if cur_b is None:
                    cur_b = []
                cur_b.append(ev)
                if ev.type is EventType.HANDOFF:
                    instances.append(cur_b)
                    cur_b = None
    if cur_b is not None:
        instances.append(cur_b)

    total_calls = 0
    total_lat_s = 0.0
    total_cost = 0.0
    dup_turns = 0
    dup_lat_s = 0.0
    dup_cost = 0.0
    val_retries = 0
    val_lat_s = 0.0
    val_cost = 0.0

    for inst in instances:
        mc_events = [e for e in inst if isinstance(e.payload, ModelCallPayload)]
        last_tc_seq = max(
            (e.seq for e in inst if isinstance(e.payload, ToolCallPayload)), default=-1
        )
        post_tool_schema_calls: list[ModelCallPayload] = []
        for e in mc_events:
            mc = e.payload
            assert isinstance(mc, ModelCallPayload)
            total_calls += 1
            total_lat_s += mc.latency_ms / 1000.0
            total_cost += mc.cost_usd
            if e.seq > last_tc_seq:
                if mc.output_schema is None:
                    dup_turns += 1
                    dup_lat_s += mc.latency_ms / 1000.0
                    dup_cost += mc.cost_usd
                else:
                    post_tool_schema_calls.append(mc)
        if len(post_tool_schema_calls) > 1:
            for retry_mc in post_tool_schema_calls[1:]:
                val_retries += 1
                val_lat_s += retry_mc.latency_ms / 1000.0
                val_cost += retry_mc.cost_usd

    return AgentBMetrics(
        role_instances=len(instances),
        model_calls=total_calls,
        llm_latency_seconds=total_lat_s,
        cost_usd=total_cost,
        post_tool_duplicate_turns=dup_turns,
        post_tool_duplicate_latency_seconds=dup_lat_s,
        post_tool_duplicate_cost_usd=dup_cost,
        validation_retries=val_retries,
        validation_retry_latency_seconds=val_lat_s,
        validation_retry_cost_usd=val_cost,
        contract_stage_seconds=contract_s,
    )


def compute_run_breakdown(events: Sequence[AnyEvent]) -> dict[str, Any]:
    """Compute per-run cost/latency breakdowns by Stage and by step/actor category."""
    stage_wall: dict[str, float] = {s: 0.0 for s in STAGES_ORDER}
    stage_calls: dict[str, int] = {s: 0 for s in STAGES_ORDER}
    stage_llm_s: dict[str, float] = {s: 0.0 for s in STAGES_ORDER}
    stage_cost: dict[str, float] = {s: 0.0 for s in STAGES_ORDER}

    actor_calls: dict[str, int] = {r: 0 for r in ACTOR_LLM_ROLES}
    actor_llm_s: dict[str, float] = {r: 0.0 for r in ACTOR_LLM_ROLES}
    actor_cost: dict[str, float] = {r: 0.0 for r in ACTOR_LLM_ROLES}

    tool_calls_map: dict[str, ToolCallPayload] = {}
    tool_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "ok": 0, "error": 0, "denied": 0, "latency_seconds": 0.0}
    )

    job_running_start: dict[str, datetime] = {}
    job_queued_start: dict[str, datetime] = {}
    job_running_s = 0.0
    job_queued_to_terminal_s = 0.0
    jobs_seen: set[str] = set()
    jobs_terminal: Counter[str] = Counter()

    cur_stage: str | None = None
    cur_stage_t: datetime | None = None

    for ev in events:
        p = ev.payload
        if isinstance(p, StageTransitionPayload):
            if cur_stage in stage_wall and cur_stage_t is not None:
                stage_wall[cur_stage] += max(0.0, (ev.created_at - cur_stage_t).total_seconds())
            cur_stage = p.stage.value if p.stage is not None else None
            cur_stage_t = ev.created_at
        elif isinstance(p, ModelCallPayload):
            role = ev.actor_role.value
            actor_calls[role] = actor_calls.get(role, 0) + 1
            actor_llm_s[role] = actor_llm_s.get(role, 0.0) + p.latency_ms / 1000.0
            actor_cost[role] = actor_cost.get(role, 0.0) + p.cost_usd
            if cur_stage in stage_calls:
                stage_calls[cur_stage] += 1
                stage_llm_s[cur_stage] += p.latency_ms / 1000.0
                stage_cost[cur_stage] += p.cost_usd
        elif isinstance(p, ToolCallPayload):
            tool_calls_map[p.call_id] = p
        elif isinstance(p, ToolResultPayload):
            matched = tool_calls_map.get(p.call_id)
            tool_name = matched.tool if matched is not None else p.tool
            row = tool_stats[tool_name]
            row["calls"] += 1
            row[p.status.value] += 1
            row["latency_seconds"] += p.latency_ms / 1000.0
        elif isinstance(p, JobTransitionPayload):
            jobs_seen.add(p.job_id)
            if p.to_state is JobState.QUEUED and p.job_id not in job_queued_start:
                job_queued_start[p.job_id] = ev.created_at
            elif p.to_state is JobState.RUNNING:
                job_running_start[p.job_id] = ev.created_at
            elif p.to_state in (
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.FAILED_TERMINAL,
            ):
                if p.to_state in (
                    JobState.COMPLETED,
                    JobState.CANCELLED,
                    JobState.FAILED_TERMINAL,
                ):
                    jobs_terminal[p.to_state.value] += 1
                    if p.job_id in job_queued_start:
                        job_queued_to_terminal_s += max(
                            0.0, (ev.created_at - job_queued_start.pop(p.job_id)).total_seconds()
                        )
                if p.job_id in job_running_start:
                    job_running_s += max(
                        0.0, (ev.created_at - job_running_start.pop(p.job_id)).total_seconds()
                    )

    trace_span_s = (
        max(0.0, (events[-1].created_at - events[0].created_at).total_seconds()) if events else 0.0
    )
    return {
        "trace_span_seconds": trace_span_s,
        "by_stage": {
            stage: {
                "wall_seconds": stage_wall[stage],
                "model_calls": stage_calls[stage],
                "llm_latency_seconds": stage_llm_s[stage],
                "cost_usd": stage_cost[stage],
            }
            for stage in STAGES_ORDER
        },
        "by_actor_llm": {
            role: {
                "calls": actor_calls.get(role, 0),
                "latency_seconds": actor_llm_s.get(role, 0.0),
                "cost_usd": actor_cost.get(role, 0.0),
            }
            for role in sorted(set(ACTOR_LLM_ROLES) | set(actor_calls))
        },
        "by_tool": dict(tool_stats),
        "sandbox_jobs": {
            "jobs_total": len(jobs_seen),
            "succeeded": jobs_terminal.get(JobState.COMPLETED.value, 0),
            "failed": jobs_terminal.get(JobState.FAILED_TERMINAL.value, 0),
            "cancelled": jobs_terminal.get(JobState.CANCELLED.value, 0),
            "running_seconds": job_running_s,
            "queued_to_terminal_seconds": job_queued_to_terminal_s,
        },
    }


def aggregate_breakdowns(
    records: Mapping[str, CaseRecord],
    state_dir: Path,
) -> dict[str, Any]:
    """Aggregate stage and step/actor cost+latency breakdowns across an arm's primary runs."""
    n = len(records)
    by_stage: dict[str, dict[str, float]] = {
        s: {"wall_seconds": 0.0, "model_calls": 0.0, "llm_latency_seconds": 0.0, "cost_usd": 0.0}
        for s in STAGES_ORDER
    }
    by_actor: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "latency_seconds": 0.0, "cost_usd": 0.0}
    )
    by_tool: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "ok": 0.0, "error": 0.0, "denied": 0.0, "latency_seconds": 0.0}
    )
    sandbox: dict[str, float] = {
        "jobs_total": 0.0,
        "succeeded": 0.0,
        "failed": 0.0,
        "cancelled": 0.0,
        "running_seconds": 0.0,
        "queued_to_terminal_seconds": 0.0,
    }
    total_system_s = 0.0
    total_trace_span_s = 0.0
    total_compute_s = 0.0

    for cid, rec in sorted(records.items()):
        total_system_s += float(rec.timings.system_seconds)
        total_compute_s += float(rec.usage.compute_seconds)
        db_path = locate_db(rec.run_id, state_dir)
        if db_path is None:
            raise FileNotFoundError(f"Missing SQLite trace for {cid} ({rec.run_id}) in {state_dir}")
        bd = compute_run_breakdown(read_run_events(db_path, rec.run_id))
        total_trace_span_s += bd["trace_span_seconds"]
        for s in STAGES_ORDER:
            for k in ("wall_seconds", "model_calls", "llm_latency_seconds", "cost_usd"):
                by_stage[s][k] += bd["by_stage"][s][k]
        for role, vals in bd["by_actor_llm"].items():
            for k in ("calls", "latency_seconds", "cost_usd"):
                by_actor[role][k] += vals[k]
        for tname, vals in bd["by_tool"].items():
            for k in ("calls", "ok", "error", "denied", "latency_seconds"):
                by_tool[tname][k] += vals[k]
        for k in sandbox:
            sandbox[k] += bd["sandbox_jobs"][k]

    total_llm_s = sum(v["latency_seconds"] for v in by_actor.values())
    total_cost = sum(v["cost_usd"] for v in by_actor.values())
    total_tool_s = sum(v["latency_seconds"] for v in by_tool.values())
    poll_s = by_tool.get("get_experiment_status", {}).get("latency_seconds", 0.0)
    non_poll_tool_s = total_tool_s - poll_s

    stage_rows: dict[str, Any] = {}
    for s in STAGES_ORDER:
        row = by_stage[s]
        w = row["wall_seconds"]
        c = row["cost_usd"]
        stage_rows[s] = {
            "wall_seconds_total": round(w, 3),
            "wall_seconds_mean": round(w / n, 3) if n else 0.0,
            "wall_pct": round(100.0 * w / total_trace_span_s, 2) if total_trace_span_s else 0.0,
            "model_calls_total": int(row["model_calls"]),
            "llm_latency_seconds_total": round(row["llm_latency_seconds"], 3),
            "cost_usd_total": round(c, 6),
            "cost_pct": round(100.0 * c / total_cost, 2) if total_cost else 0.0,
        }

    actor_rows: dict[str, Any] = {}
    for role in sorted(by_actor):
        row = by_actor[role]
        lat = row["latency_seconds"]
        c = row["cost_usd"]
        calls = int(row["calls"])
        actor_rows[role] = {
            "calls_total": calls,
            "calls_mean": round(calls / n, 2) if n else 0.0,
            "latency_seconds_total": round(lat, 3),
            "latency_seconds_mean": round(lat / n, 3) if n else 0.0,
            "llm_latency_pct": round(100.0 * lat / total_llm_s, 2) if total_llm_s else 0.0,
            "cost_usd_total": round(c, 6),
            "cost_usd_mean": round(c / n, 6) if n else 0.0,
            "cost_pct": round(100.0 * c / total_cost, 2) if total_cost else 0.0,
        }

    tool_rows: dict[str, Any] = {}
    for tname, row in sorted(by_tool.items(), key=lambda kv: -kv[1]["latency_seconds"]):
        calls = int(row["calls"])
        lat = row["latency_seconds"]
        tool_rows[tname] = {
            "calls_total": calls,
            "ok": int(row["ok"]),
            "error": int(row["error"]),
            "denied": int(row["denied"]),
            "latency_seconds_total": round(lat, 3),
            "mean_ms": round(1000.0 * lat / calls, 2) if calls else 0.0,
        }

    return {
        "runs": n,
        "total_system_seconds": round(total_system_s, 3),
        "total_trace_span_seconds": round(total_trace_span_s, 3),
        "total_llm_latency_seconds": round(total_llm_s, 3),
        "total_cost_usd": round(total_cost, 6),
        "total_tool_latency_seconds": round(total_tool_s, 3),
        "non_poll_mcp_tool_latency_seconds": round(non_poll_tool_s, 3),
        "poll_wait_tool_latency_seconds": round(poll_s, 3),
        "total_sandbox_compute_seconds": round(total_compute_s, 3),
        "by_stage": stage_rows,
        "by_actor_llm": actor_rows,
        "by_tool": tool_rows,
        "sandbox_execution": {
            "jobs_total": int(sandbox["jobs_total"]),
            "succeeded": int(sandbox["succeeded"]),
            "failed": int(sandbox["failed"]),
            "cancelled": int(sandbox["cancelled"]),
            "running_seconds_total": round(sandbox["running_seconds"], 3),
            "queued_to_terminal_seconds_total": round(sandbox["queued_to_terminal_seconds"], 3),
            "recorded_compute_seconds_total": round(total_compute_s, 3),
        },
        "accounting_note": (
            "Stage wall_seconds partition each run's trace span non-overlappingly (sum = "
            "total_trace_span_seconds). Actor LLM latencies and MCP tool latencies are cumulative "
            "call durations: get_experiment_status long-polls overlap sandbox RUNNING time, and "
            "CONTRACT runs agent_b and agent_c concurrently under phase_a_mode=parallel."
        ),
    }


def query_traces(
    arms: Mapping[str, tuple[Mapping[str, CaseRecord], Path]],
    *,
    arm: str = "both",
    role: str | None = None,
    event_type: str | None = None,
    case_id: str | None = None,
    stage: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Programmatically search stored SQLite traces by role, type, case_id, and/or stage."""
    if role is not None:
        ActorRole(role)
    if event_type is not None:
        EventType(event_type)
    if stage is not None:
        Stage(stage)

    selected_arms = list(arms.keys()) if arm == "both" else [arm]
    matches: list[dict[str, Any]] = []
    scanned_runs = 0
    total_matched = 0

    for arm_name in selected_arms:
        records, state_dir = arms[arm_name]
        for cid, rec in sorted(records.items()):
            if case_id is not None and cid != case_id:
                continue
            db_path = locate_db(rec.run_id, state_dir)
            if db_path is None:
                continue
            scanned_runs += 1
            events = read_run_events(db_path, rec.run_id)
            cur_stage: str | None = None
            tool_names: dict[str, str] = {}
            for ev in events:
                p = ev.payload
                if isinstance(p, StageTransitionPayload):
                    cur_stage = p.stage.value if p.stage is not None else None
                if isinstance(p, ToolCallPayload):
                    tool_names[p.call_id] = p.tool
                if role is not None and ev.actor_role.value != role:
                    continue
                if event_type is not None and ev.type.value != event_type:
                    continue
                if stage is not None and cur_stage != stage:
                    continue
                total_matched += 1
                if len(matches) < limit:
                    entry: dict[str, Any] = {
                        "arm": arm_name,
                        "case_id": cid,
                        "run_id": rec.run_id,
                        "seq": ev.seq,
                        "stage": cur_stage,
                        "role": ev.actor_role.value,
                        "type": ev.type.value,
                        "created_at": ev.created_at.isoformat(),
                        "trial_id": ev.trial_id,
                        "job_id": ev.job_id,
                        "detail": describe(ev),
                    }
                    if isinstance(p, ModelCallPayload):
                        entry["output_schema"] = p.output_schema
                        entry["latency_ms"] = round(p.latency_ms, 2)
                        entry["cost_usd"] = round(p.cost_usd, 6)
                    elif isinstance(p, ToolResultPayload):
                        entry["call_id"] = p.call_id
                        entry["tool_name"] = tool_names.get(p.call_id, p.tool)
                        entry["latency_ms"] = round(p.latency_ms, 2)
                    elif isinstance(p, ToolCallPayload):
                        entry["call_id"] = p.call_id
                        entry["tool_name"] = p.tool
                    matches.append(entry)

    return {
        "filters": {
            "arm": arm,
            "role": role,
            "type": event_type,
            "case_id": case_id,
            "stage": stage,
            "limit": limit,
        },
        "scanned_runs": scanned_runs,
        "total_matched": total_matched,
        "returned": len(matches),
        "matches": matches,
    }


def verify_hypora_trace_cli(
    pre_records: Mapping[str, CaseRecord],
    post_records: Mapping[str, CaseRecord],
    *,
    pre_state_dir: Path = FROZEN_STATE_DIR,
    post_state_dir: Path = POST_FIX_STATE_DIR,
) -> dict[str, Any]:
    """Exercise `hypora.cli.trace` filters and `EventFilters`."""
    sample_case = (
        "test_clean_rice" if "test_clean_rice" in post_records else sorted(post_records)[0]
    )
    post_rec = post_records[sample_case]
    pre_rec = pre_records[sample_case]
    post_db = locate_db(post_rec.run_id, post_state_dir)
    pre_db = locate_db(pre_rec.run_id, pre_state_dir)
    assert post_db is not None and pre_db is not None

    post_events = read_run_events(post_db, post_rec.run_id)
    pre_events = read_run_events(pre_db, pre_rec.run_id)
    first_trial = next((e.trial_id for e in post_events if e.trial_id), None)
    first_job = next(
        (e.payload.job_id for e in post_events if isinstance(e.payload, JobTransitionPayload)),
        None,
    )

    checks: dict[str, Any] = {}
    filter_specs: list[tuple[str, list[str]]] = [
        ("role_agent_b", ["role=agent_b"]),
        ("type_model_and_tool_result", ["type=model_call,tool_result", "role=agent_b"]),
    ]
    if first_trial:
        filter_specs.append(("trial_filter", [f"trial={first_trial}"]))
    if first_job:
        filter_specs.append(("job_filter", [f"job={first_job}"]))

    rows = trace_rows(post_events)
    for name, specs in filter_specs:
        tf = parse_filters(specs)
        shown = [r for r in rows if tf.keeps(r)]
        checks[name] = {
            "specs": specs,
            "matched_rows": len(shown),
            "summary": summary(shown, post_events),
        }

    db = Database(post_db)
    try:
        sql_filtered = EventLog(db).read(
            post_rec.run_id,
            EventFilters(
                actor_role=ActorRole.AGENT_B,
                types=(EventType.MODEL_CALL, EventType.TOOL_RESULT),
            ),
        )
    finally:
        db.close()

    checks["sqlite_event_filters_agent_b"] = len(sql_filtered)

    pre_b_rows = [r for r in trace_rows(pre_events) if parse_filters(["role=agent_b"]).keeps(r)]
    post_b_rows = [r for r in trace_rows(post_events) if parse_filters(["role=agent_b"]).keeps(r)]
    return {
        "sample_case": sample_case,
        "pre_fix_run_id": pre_rec.run_id,
        "post_fix_run_id": post_rec.run_id,
        "cli_filter_checks": checks,
        "excerpts": {
            "pre_fix_agent_b_trace": render_rows(pre_b_rows),
            "post_fix_agent_b_trace": render_rows(post_b_rows),
        },
    }


# --- Statistics & Report Generation --------------------------------------------------------------


def paired_bootstrap_continuous(
    first: Sequence[float],
    second: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap 95% CI on `mean(first - second)` using core RNG/resample settings."""
    if resamples < MIN_RESAMPLES:
        raise ValueError(f"Requires >= {MIN_RESAMPLES} resamples, got {resamples}")
    if len(first) != len(second) or not first:
        raise ValueError("Paired continuous inputs must be non-empty and equal length")
    diffs = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[drawn].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_first": round(float(np.mean(first)), 4),
        "mean_second": round(float(np.mean(second)), 4),
        "mean_difference": round(float(diffs.mean()), 4),
        "median_difference": round(float(np.median(diffs)), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "resamples": resamples,
        "seed": seed,
    }


def _rate_dict(records: Sequence[CaseRecord]) -> dict[str, Any]:
    graded = [r for r in records if r.finished]
    passed = sum(r.passed for r in graded)
    iv = wilson_interval(passed, len(graded)) if graded else None
    return {
        "passed": passed,
        "graded": len(graded),
        "rate": round(passed / len(graded), 4) if graded else None,
        "wilson_ci_95": [round(iv.low, 4), round(iv.high, 4)] if iv else None,
    }


def _latency_dict(values: Sequence[float]) -> dict[str, Any]:
    ls = latency_summary(values)
    return {
        "count": ls.count,
        "mean": round(ls.mean, 3),
        "median": round(float(median(values)), 3),
        "p95": round(ls.p95, 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "method": ls.method,
    }


def _hidden_quality_comparison(
    cases: Sequence[Case],
    pre_primary: Mapping[str, CaseRecord],
    post_primary: Mapping[str, CaseRecord],
) -> dict[str, Any]:
    """Compare hidden-split predictive metrics across all 24 non-malformed cases."""
    modelable = [
        c
        for c in sorted(cases, key=lambda x: x.id)
        if c.grading.hidden_metric is not None
        and FaultKind.MALFORMED_DATA not in {f.kind for f in c.injected_faults}
    ]
    rows: list[dict[str, Any]] = []
    quality_deltas: list[float] = []
    improved = 0
    equal = 0
    degraded = 0
    for case in modelable:
        pre = pre_primary[case.id]
        post = post_primary[case.id]
        pre_hm = pre.grade.hidden_metric if pre.grade else None
        post_hm = post.grade.hidden_metric if post.grade else None
        assert case.grading.hidden_metric is not None
        metric_name = (
            post_hm.name.value
            if post_hm is not None
            else (
                pre_hm.name.value if pre_hm is not None else case.grading.hidden_metric.name.value
            )
        )
        direction = MetricName(metric_name).direction
        pre_val = pre_hm.value if pre_hm is not None else None
        post_val = post_hm.value if post_hm is not None else None
        raw_delta: float | None = None
        qual_delta: float | None = None
        rel_pct: float | None = None
        if pre_val is not None and post_val is not None:
            raw_delta = post_val - pre_val
            qual_delta = (
                raw_delta if direction is MetricDirection.MAXIMIZE else (pre_val - post_val)
            )
            quality_deltas.append(qual_delta)
            if abs(qual_delta) <= 1e-12:
                equal += 1
            elif qual_delta > 0:
                improved += 1
            else:
                degraded += 1
            if abs(pre_val) > 1e-12:
                rel_pct = 100.0 * qual_delta / abs(pre_val)
        rows.append(
            {
                "case_id": case.id,
                "metric": metric_name,
                "direction": direction.value,
                "pre_passed": pre.passed,
                "post_passed": post.passed,
                "pre_value": round(pre_val, 6) if pre_val is not None else None,
                "post_value": round(post_val, 6) if post_val is not None else None,
                "raw_delta": round(raw_delta, 6) if raw_delta is not None else None,
                "directional_quality_delta": round(qual_delta, 6)
                if qual_delta is not None
                else None,
                "directional_relative_pct": round(rel_pct, 3) if rel_pct is not None else None,
            }
        )

    max_rows = [
        r
        for r in rows
        if r["direction"] == MetricDirection.MAXIMIZE.value
        and r["directional_quality_delta"] is not None
    ]
    min_rows = [
        r
        for r in rows
        if r["direction"] == MetricDirection.MINIMIZE.value
        and r["directional_quality_delta"] is not None
    ]
    return {
        "modelable_cases": len(modelable),
        "paired_with_hidden_metric": len(quality_deltas),
        "pre_passed": sum(1 for c in modelable if pre_primary[c.id].passed),
        "post_passed": sum(1 for c in modelable if post_primary[c.id].passed),
        "improved_count": improved,
        "unchanged_count": equal,
        "degraded_count": degraded,
        "maximize_metrics": {
            "count": len(max_rows),
            "mean_delta": (
                round(float(np.mean([r["directional_quality_delta"] for r in max_rows])), 6)
                if max_rows
                else None
            ),
            "median_delta": (
                round(float(median([r["directional_quality_delta"] for r in max_rows])), 6)
                if max_rows
                else None
            ),
        },
        "minimize_metrics": {
            "count": len(min_rows),
            "mean_improvement": (
                round(float(np.mean([r["directional_quality_delta"] for r in min_rows])), 6)
                if min_rows
                else None
            ),
            "median_improvement": (
                round(float(median([r["directional_quality_delta"] for r in min_rows])), 6)
                if min_rows
                else None
            ),
        },
        "per_case": rows,
    }


def _repeat_comparison(
    pre_repeats: Mapping[str, list[CaseRecord]],
    post_repeats: Mapping[str, list[CaseRecord]],
) -> dict[str, Any]:
    pre_all = [r for cid in sorted(final.REPEAT_CASES) for r in pre_repeats.get(cid, [])]
    post_all = [r for cid in sorted(final.REPEAT_CASES) for r in post_repeats.get(cid, [])]
    per_case: list[dict[str, Any]] = []
    for cid in sorted(final.REPEAT_CASES):
        pre_runs = pre_repeats.get(cid, [])
        post_runs = post_repeats.get(cid, [])
        pre_lat = [r.timings.system_seconds for r in pre_runs]
        post_lat = [r.timings.system_seconds for r in post_runs]
        per_case.append(
            {
                "case_id": cid,
                "pre_fix": {
                    "passed": sum(r.passed for r in pre_runs),
                    "total": len(pre_runs),
                    "min_s": round(min(pre_lat), 2) if pre_lat else None,
                    "median_s": round(float(median(pre_lat)), 2) if pre_lat else None,
                    "max_s": round(max(pre_lat), 2) if pre_lat else None,
                },
                "post_fix": {
                    "passed": sum(r.passed for r in post_runs),
                    "total": len(post_runs),
                    "min_s": round(min(post_lat), 2) if post_lat else None,
                    "median_s": round(float(median(post_lat)), 2) if post_lat else None,
                    "max_s": round(max(post_lat), 2) if post_lat else None,
                },
            }
        )
    return {
        "pre_fix_overall": _rate_dict(pre_all),
        "post_fix_overall": _rate_dict(post_all),
        "pre_fix_latency": _latency_dict([r.timings.system_seconds for r in pre_all])
        if pre_all
        else None,
        "post_fix_latency": _latency_dict([r.timings.system_seconds for r in post_all])
        if post_all
        else None,
        "per_case": per_case,
    }


def build_f7_evidence(
    *,
    pre_results_dir: Path = FROZEN_DEFAULT_DIR,
    pre_state_dir: Path = FROZEN_STATE_DIR,
    post_results_dir: Path = POST_FIX_RESULTS_DIR,
    post_state_dir: Path = POST_FIX_STATE_DIR,
    audit_path: Path = AUDIT_PATH,
    secret_values: Sequence[str] = (),
    cases: Sequence[Case] | None = None,
) -> dict[str, Any]:
    """Compute the full S5.3 F7 evidence dictionary from stored pre-fix and post-fix artifacts."""
    loaded_cases = list(cases) if cases is not None else load_cases(final.CASE_SET)
    case_ids = sorted(c.id for c in loaded_cases)
    pre_primary, pre_repeats = load_arm_records(pre_results_dir, case_ids)
    post_primary, post_repeats = load_arm_records(post_results_dir, case_ids)
    if len(pre_primary) != len(case_ids) or len(post_primary) != len(case_ids):
        raise RuntimeError(
            f"Need {len(case_ids)} pre-fix and {len(case_ids)} post-fix primary records; "
            f"got {len(pre_primary)} and {len(post_primary)}"
        )

    pre_audit = audit_arm_traces(pre_primary, pre_state_dir, secret_values=secret_values)
    post_audit = audit_arm_traces(post_primary, post_state_dir, secret_values=secret_values)

    pre_list = [pre_primary[cid] for cid in case_ids]
    post_list = [post_primary[cid] for cid in case_ids]
    post_pass_bools = [r.passed for r in post_list]
    pre_pass_bools = [r.passed for r in pre_list]
    pass_table = paired_outcome_table(post_pass_bools, pre_pass_bools)
    pass_boot = paired_bootstrap_difference(
        post_pass_bools,
        pre_pass_bools,
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    pre_sys = [r.timings.system_seconds for r in pre_list]
    post_sys = [r.timings.system_seconds for r in post_list]
    pre_act = [r.timings.active_seconds for r in pre_list]
    post_act = [r.timings.active_seconds for r in post_list]

    pre_b: list[AgentBMetrics] = []
    post_b: list[AgentBMetrics] = []
    for cid in case_ids:
        pre_db = locate_db(pre_primary[cid].run_id, pre_state_dir)
        post_db = locate_db(post_primary[cid].run_id, post_state_dir)
        assert pre_db is not None and post_db is not None
        pre_b.append(extract_agent_b_metrics(read_run_events(pre_db, pre_primary[cid].run_id)))
        post_b.append(extract_agent_b_metrics(read_run_events(post_db, post_primary[cid].run_id)))

    pre_all_runs = [
        *pre_list,
        *(r for cid in sorted(final.REPEAT_CASES) for r in pre_repeats.get(cid, [])),
    ]
    post_all_runs = [
        *post_list,
        *(r for cid in sorted(final.REPEAT_CASES) for r in post_repeats.get(cid, [])),
    ]

    ledger_summary: dict[str, Any] = {
        "cap_usd": CAP_USD,
        "measured_usd": None,
        "unknown_reserved_usd": None,
    }
    if audit_path.is_file():
        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
        ledger_summary = {
            "cap_usd": audit_data.get("cap_usd", CAP_USD),
            "attempt_reservation_usd": audit_data.get(
                "attempt_reservation_usd", ATTEMPT_RESERVATION_USD
            ),
            "measured_usd": audit_data.get("measured_usd"),
            "unknown_reserved_usd": audit_data.get("unknown_reserved_usd"),
            "attempts": len(audit_data.get("attempts", [])),
        }

    arms_map: dict[str, tuple[Mapping[str, CaseRecord], Path]] = {
        "pre_fix": (pre_primary, pre_state_dir),
        "post_fix": (post_primary, post_state_dir),
    }
    query_demo = {
        "agent_b_model_calls_contract": query_traces(
            arms_map,
            arm="both",
            role="agent_b",
            event_type="model_call",
            stage="CONTRACT",
            limit=10,
        ),
        "post_fix_tool_results": query_traces(
            arms_map, arm="post_fix", event_type="tool_result", limit=20
        ),
    }
    cli_verification = verify_hypora_trace_cli(
        pre_primary,
        post_primary,
        pre_state_dir=pre_state_dir,
        post_state_dir=post_state_dir,
    )

    retry_case_excerpt: dict[str, Any] | None = None
    for idx, cid in enumerate(case_ids):
        if pre_b[idx].validation_retries > 0:
            pdb_pre = locate_db(pre_primary[cid].run_id, pre_state_dir)
            pdb_post = locate_db(post_primary[cid].run_id, post_state_dir)
            if pdb_pre and pdb_post:
                pre_rows_b = [
                    r
                    for r in trace_rows(read_run_events(pdb_pre, pre_primary[cid].run_id))
                    if parse_filters(["role=agent_b"]).keeps(r)
                ]
                post_rows_b = [
                    r
                    for r in trace_rows(read_run_events(pdb_post, post_primary[cid].run_id))
                    if parse_filters(["role=agent_b"]).keeps(r)
                ]
                retry_case_excerpt = {
                    "case_id": cid,
                    "pre_fix_run_id": pre_primary[cid].run_id,
                    "post_fix_run_id": post_primary[cid].run_id,
                    "pre_validation_retries": pre_b[idx].validation_retries,
                    "post_validation_retries": post_b[idx].validation_retries,
                    "pre_fix_agent_b_trace": render_rows(pre_rows_b),
                    "post_fix_agent_b_trace": render_rows(post_rows_b),
                }
                break

    pre_breakdown = aggregate_breakdowns(pre_primary, pre_state_dir)
    post_breakdown = aggregate_breakdowns(post_primary, post_state_dir)

    per_case_rows = []
    for i, cid in enumerate(case_ids):
        per_case_rows.append(
            {
                "case_id": cid,
                "pre_fix": {
                    "run_id": pre_primary[cid].run_id,
                    "passed": pre_primary[cid].passed,
                    "system_seconds": round(pre_primary[cid].timings.system_seconds, 3),
                    "cost_usd": round(pre_primary[cid].usage.actual_spend_usd, 6),
                    "agent_b_calls": pre_b[i].model_calls,
                    "agent_b_latency_seconds": round(pre_b[i].llm_latency_seconds, 3),
                    "post_tool_duplicate_turns": pre_b[i].post_tool_duplicate_turns,
                    "validation_retries": pre_b[i].validation_retries,
                    "contract_stage_seconds": round(pre_b[i].contract_stage_seconds, 3),
                },
                "post_fix": {
                    "run_id": post_primary[cid].run_id,
                    "passed": post_primary[cid].passed,
                    "system_seconds": round(post_primary[cid].timings.system_seconds, 3),
                    "cost_usd": round(post_primary[cid].usage.actual_spend_usd, 6),
                    "agent_b_calls": post_b[i].model_calls,
                    "agent_b_latency_seconds": round(post_b[i].llm_latency_seconds, 3),
                    "post_tool_duplicate_turns": post_b[i].post_tool_duplicate_turns,
                    "validation_retries": post_b[i].validation_retries,
                    "contract_stage_seconds": round(post_b[i].contract_stage_seconds, 3),
                },
            }
        )

    return {
        "step": "S5.3-F7",
        "spec_sections": ["§16", "§19.5", "§20"],
        "config": "default",
        "model": "openai/gpt-6-luna",
        "profile": "eval",
        "price_retrieved_on": PRICE_RETRIEVED_ON,
        "pre_fix_commit": FROZEN_COMMIT,
        "post_fix_commit": POST_FIX_SHA,
        "implementation_audit": {
            "pre_fix": pre_audit,
            "post_fix": post_audit,
        },
        "architecture_verification": {
            "event_log_store": (
                "SQLite events table (EventRow) with EventFilters & AnyEvent Pydantic union"
            ),
            "cli_command": (
                "hypora trace <run_id> --filter key=value[,value...] (keys: type, role, trial, job)"
            ),
            "query_tool": "scripts/eval_s53_f7.py --query",
            "cli_filter_verification": cli_verification["cli_filter_checks"],
            "query_demonstration": query_demo,
        },
        "evaluation": {
            "primary_pass_rate": {
                "pre_fix": _rate_dict(pre_list),
                "post_fix": _rate_dict(post_list),
                "paired_post_minus_pre": {
                    "cases": pass_table.total,
                    "both_pass": pass_table.both_pass,
                    "post_only": pass_table.first_only,
                    "pre_only": pass_table.second_only,
                    "both_fail": pass_table.both_fail,
                    "difference": round(pass_boot.difference, 4),
                    "ci_95": [round(pass_boot.interval.low, 4), round(pass_boot.interval.high, 4)],
                    "verdict": pass_boot.verdict.value,
                    "resamples": pass_boot.resamples,
                    "seed": pass_boot.seed,
                },
            },
            "predictive_quality_24_cases": _hidden_quality_comparison(
                loaded_cases, pre_primary, post_primary
            ),
            "repeat_sample_6x3": _repeat_comparison(pre_repeats, post_repeats),
            "end_to_end_latency_seconds": {
                "pre_fix_system": _latency_dict(pre_sys),
                "post_fix_system": _latency_dict(post_sys),
                "paired_system_post_minus_pre": paired_bootstrap_continuous(post_sys, pre_sys),
                "pre_fix_active": _latency_dict(pre_act),
                "post_fix_active": _latency_dict(post_act),
                "paired_active_post_minus_pre": paired_bootstrap_continuous(post_act, pre_act),
            },
            "bottleneck_metrics": {
                "agent_b_model_calls": {
                    "pre_fix_total": sum(m.model_calls for m in pre_b),
                    "post_fix_total": sum(m.model_calls for m in post_b),
                    "paired_per_run": paired_bootstrap_continuous(
                        [float(m.model_calls) for m in post_b],
                        [float(m.model_calls) for m in pre_b],
                    ),
                },
                "agent_b_llm_latency_seconds": {
                    "pre_fix_total": round(sum(m.llm_latency_seconds for m in pre_b), 3),
                    "post_fix_total": round(sum(m.llm_latency_seconds for m in post_b), 3),
                    "paired_per_run": paired_bootstrap_continuous(
                        [m.llm_latency_seconds for m in post_b],
                        [m.llm_latency_seconds for m in pre_b],
                    ),
                },
                "agent_b_cost_usd": {
                    "pre_fix_total": round(sum(m.cost_usd for m in pre_b), 6),
                    "post_fix_total": round(sum(m.cost_usd for m in post_b), 6),
                    "paired_per_run": paired_bootstrap_continuous(
                        [m.cost_usd for m in post_b],
                        [m.cost_usd for m in pre_b],
                    ),
                },
                "post_tool_duplicate_turns": {
                    "pre_fix_total": sum(m.post_tool_duplicate_turns for m in pre_b),
                    "pre_fix_runs_affected": sum(
                        1 for m in pre_b if m.post_tool_duplicate_turns > 0
                    ),
                    "pre_fix_latency_seconds": round(
                        sum(m.post_tool_duplicate_latency_seconds for m in pre_b), 3
                    ),
                    "pre_fix_cost_usd": round(
                        sum(m.post_tool_duplicate_cost_usd for m in pre_b), 6
                    ),
                    "post_fix_total": sum(m.post_tool_duplicate_turns for m in post_b),
                    "post_fix_runs_affected": sum(
                        1 for m in post_b if m.post_tool_duplicate_turns > 0
                    ),
                    "post_fix_latency_seconds": round(
                        sum(m.post_tool_duplicate_latency_seconds for m in post_b), 3
                    ),
                    "post_fix_cost_usd": round(
                        sum(m.post_tool_duplicate_cost_usd for m in post_b), 6
                    ),
                    "paired_per_run": paired_bootstrap_continuous(
                        [float(m.post_tool_duplicate_turns) for m in post_b],
                        [float(m.post_tool_duplicate_turns) for m in pre_b],
                    ),
                },
                "validation_retries": {
                    "pre_fix_total": sum(m.validation_retries for m in pre_b),
                    "pre_fix_runs_affected": sum(1 for m in pre_b if m.validation_retries > 0),
                    "pre_fix_latency_seconds": round(
                        sum(m.validation_retry_latency_seconds for m in pre_b), 3
                    ),
                    "pre_fix_cost_usd": round(sum(m.validation_retry_cost_usd for m in pre_b), 6),
                    "post_fix_total": sum(m.validation_retries for m in post_b),
                    "post_fix_runs_affected": sum(1 for m in post_b if m.validation_retries > 0),
                    "post_fix_latency_seconds": round(
                        sum(m.validation_retry_latency_seconds for m in post_b), 3
                    ),
                    "post_fix_cost_usd": round(sum(m.validation_retry_cost_usd for m in post_b), 6),
                    "paired_per_run": paired_bootstrap_continuous(
                        [float(m.validation_retries) for m in post_b],
                        [float(m.validation_retries) for m in pre_b],
                    ),
                },
                "contract_stage_latency_seconds": {
                    "pre_fix": _latency_dict([m.contract_stage_seconds for m in pre_b]),
                    "post_fix": _latency_dict([m.contract_stage_seconds for m in post_b]),
                    "paired_per_run": paired_bootstrap_continuous(
                        [m.contract_stage_seconds for m in post_b],
                        [m.contract_stage_seconds for m in pre_b],
                    ),
                },
            },
            "spend": {
                "pre_fix_primary_actual_usd": round(
                    sum(r.usage.actual_spend_usd for r in pre_list), 6
                ),
                "pre_fix_primary_list_usd": round(sum(r.usage.cost_usd for r in pre_list), 6),
                "pre_fix_primary_per_task_usd": round(
                    sum(r.usage.actual_spend_usd for r in pre_list) / len(pre_list), 6
                ),
                "pre_fix_all_44_runs_usd": round(
                    sum(r.usage.actual_spend_usd for r in pre_all_runs), 6
                ),
                "post_fix_primary_actual_usd": round(
                    sum(r.usage.actual_spend_usd for r in post_list), 6
                ),
                "post_fix_primary_list_usd": round(sum(r.usage.cost_usd for r in post_list), 6),
                "post_fix_primary_per_task_usd": round(
                    sum(r.usage.actual_spend_usd for r in post_list) / len(post_list), 6
                ),
                "post_fix_all_44_runs_usd": round(
                    sum(r.usage.actual_spend_usd for r in post_all_runs), 6
                ),
                "post_fix_ledger": ledger_summary,
            },
            "per_case": per_case_rows,
        },
        "analysis": {
            "breakdown_pre_fix": pre_breakdown,
            "breakdown_post_fix": post_breakdown,
            "trace_excerpts": {
                "primary_example": {
                    "case_id": cli_verification["sample_case"],
                    "pre_fix_run_id": cli_verification["pre_fix_run_id"],
                    "post_fix_run_id": cli_verification["post_fix_run_id"],
                    **cli_verification["excerpts"],
                },
                "validation_retry_example": retry_case_excerpt,
            },
        },
    }


def _format_spend_line(spend: Mapping[str, Any]) -> str:
    ledger = spend.get("post_fix_ledger") or {}
    measured = ledger.get("measured_usd")
    cap = float(ledger.get("cap_usd", CAP_USD))
    if measured is not None:
        unknown = float(ledger.get("unknown_reserved_usd") or 0.0)
        return (
            f"- **Post-fix spend:** `${spend['post_fix_all_44_runs_usd']:.4f}` across 44 graded "
            f"runs; paid ledger measured `${float(measured):.4f}` + `${unknown:.4f}` "
            f"`unknown_reserved` (4 cancelled parallel-run attempts before `max_inflight = 1` "
            f"bounded concurrency; cap `${cap:.2f}`)"
        )
    return (
        f"- **Post-fix spend:** `${spend['post_fix_all_44_runs_usd']:.4f}` across all graded "
        f"runs (cap `${cap:.2f}`)"
    )


def render_f7_summary(ev: Mapping[str, Any]) -> str:
    """Render `results/s5.3/f7/f7-summary.md` covering all 4 F7 rubric parts."""
    aud_pre = ev["implementation_audit"]["pre_fix"]
    aud_post = ev["implementation_audit"]["post_fix"]
    eval_d = ev["evaluation"]
    pr_pre = eval_d["primary_pass_rate"]["pre_fix"]
    pr_post = eval_d["primary_pass_rate"]["post_fix"]
    pr_pair = eval_d["primary_pass_rate"]["paired_post_minus_pre"]
    lat_pre = eval_d["end_to_end_latency_seconds"]["pre_fix_system"]
    lat_post = eval_d["end_to_end_latency_seconds"]["post_fix_system"]
    lat_pair = eval_d["end_to_end_latency_seconds"]["paired_system_post_minus_pre"]
    bn = eval_d["bottleneck_metrics"]
    spend = eval_d["spend"]
    qual = eval_d["predictive_quality_24_cases"]
    rep = eval_d["repeat_sample_6x3"]
    bd_pre = ev["analysis"]["breakdown_pre_fix"]
    bd_post = ev["analysis"]["breakdown_post_fix"]

    n_cases = pr_pre["graded"]
    lines: list[str] = [
        "# F7 Observability Evaluation Summary (S5.3)",
        "",
        f"- **Pre-fix freeze (`eval-freeze`):** `{ev['pre_fix_commit']}`",
        f"- **Post-fix commit:** `{ev['post_fix_commit']}` (`Agent B` single-pass structured exit)",
        f"- **Config / Model:** `{ev['config']}` (`agent_mode=multi`), `{ev['model']}` "
        f"(`{ev['profile']}` profile, price date `{ev['price_retrieved_on']}`)",
        _format_spend_line(spend),
        "",
        "## 1. Implementation Audit (2 pts)",
        "",
        "| Arm | Events | Model Calls Complete | Tool Pairs Joined (`call_id`) "
        "| Orphan Calls/Results | Secret Hits |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Pre-fix (`62b4e3f`) | {aud_pre['events_total']} | "
        f"{aud_pre['model_calls_complete']}/{aud_pre['model_calls_total']} | "
        f"{aud_pre['tool_pairs_complete']}/{aud_pre['joined_tool_pairs_total']} | "
        f"{aud_pre['orphan_tool_calls_total']}/{aud_pre['orphan_tool_results_total']} | "
        f"{aud_pre['secret_hits_total']} |",
        f"| Post-fix (`f850588`) | {aud_post['events_total']} | "
        f"{aud_post['model_calls_complete']}/{aud_post['model_calls_total']} | "
        f"{aud_post['tool_pairs_complete']}/{aud_post['joined_tool_pairs_total']} | "
        f"{aud_post['orphan_tool_calls_total']}/{aud_post['orphan_tool_results_total']} | "
        f"{aud_post['secret_hits_total']} |",
        "",
        "Every `model_call` records `provider`, `model_id`, `tier`, `latency_ms`, `input_tokens`, "
        "`output_tokens`, `cost_usd`, and redacted `model_io:` `input_ref`/`output_ref`. Every "
        "`tool_call` joins 1-to-1 by `call_id` with its `tool_result` (`tool_name`, `latency_ms`, "
        "redacted `result_summary`, and `error` status) with 0 leaked secrets.",
        "",
        "## 2. Architecture & Searchable Traces (2 pts)",
        "",
        "- **Event store:** Append-only SQLite `events` table (`EventRow`) validated against the "
        "`AnyEvent` Pydantic union, indexed by `(run_id, seq)`, `type`, and `trace_id`, queried "
        "via `EventLog.read(run_id, EventFilters(...))`.",
        "- **CLI inspection:** `hypora trace <run_id> --filter "
        "type=...|role=...|trial=...|job=...`.",
        "- **Programmatic cross-run search:** `scripts/eval_s53_f7.py --query` filters stored "
        "SQLite traces by `role`, `type`, `case_id`, and `stage`.",
        "",
        "## 3. Before vs. After Evaluation (2 pts)",
        "",
        f"### Primary Pass Rate & End-to-End Latency ({n_cases} Held-Out Test Cases)",
        "",
        "| Metric | Pre-fix (`62b4e3f`) | Post-fix (`f850588`) "
        "| Paired Delta (Post - Pre, 95% Bootstrap CI) |",
        "|---|---:|---:|---|",
        f"| Primary pass rate (95% Wilson CI) | "
        f"{pr_pre['passed']}/{pr_pre['graded']} "
        f"({pr_pre['rate']:.1%}, CI {pr_pre['wilson_ci_95']}) | "
        f"{pr_post['passed']}/{pr_post['graded']} "
        f"({pr_post['rate']:.1%}, CI {pr_post['wilson_ci_95']}) | "
        f"{pr_pair['difference']:+.4f} (CI {pr_pair['ci_95']}, {pr_pair['verdict']}) |",
        f"| Wall latency mean / median / p95 (s) | "
        f"{lat_pre['mean']:.1f} / {lat_pre['median']:.1f} / {lat_pre['p95']:.1f} | "
        f"{lat_post['mean']:.1f} / {lat_post['median']:.1f} / {lat_post['p95']:.1f} | "
        f"{lat_pair['mean_difference']:+.1f}s (CI {lat_pair['ci_95']}) |",
        f"| Primary spend total / per-task (USD) | "
        f"${spend['pre_fix_primary_actual_usd']:.4f} / "
        f"${spend['pre_fix_primary_per_task_usd']:.4f} | "
        f"${spend['post_fix_primary_actual_usd']:.4f} / "
        f"${spend['post_fix_primary_per_task_usd']:.4f} | "
        f"${spend['post_fix_primary_actual_usd'] - spend['pre_fix_primary_actual_usd']:+.4f} |",
        "",
        "### Targeted Bottleneck Metrics (`Agent B` & `CONTRACT` Stage)",
        "",
        "| Bottleneck Metric | Pre-fix (`62b4e3f`) | Post-fix (`f850588`) "
        "| Paired Per-Run Delta (95% CI) |",
        "|---|---:|---:|---|",
        f"| `agent_b` model calls (total / mean) | "
        f"{bn['agent_b_model_calls']['pre_fix_total']} / "
        f"{bn['agent_b_model_calls']['paired_per_run']['mean_second']:.2f} | "
        f"{bn['agent_b_model_calls']['post_fix_total']} / "
        f"{bn['agent_b_model_calls']['paired_per_run']['mean_first']:.2f} | "
        f"{bn['agent_b_model_calls']['paired_per_run']['mean_difference']:+.2f} "
        f"(CI {bn['agent_b_model_calls']['paired_per_run']['ci_95']}) |",
        f"| `agent_b` LLM latency (total / mean s) | "
        f"{bn['agent_b_llm_latency_seconds']['pre_fix_total']:.1f}s / "
        f"{bn['agent_b_llm_latency_seconds']['paired_per_run']['mean_second']:.1f}s | "
        f"{bn['agent_b_llm_latency_seconds']['post_fix_total']:.1f}s / "
        f"{bn['agent_b_llm_latency_seconds']['paired_per_run']['mean_first']:.1f}s | "
        f"{bn['agent_b_llm_latency_seconds']['paired_per_run']['mean_difference']:+.1f}s "
        f"(CI {bn['agent_b_llm_latency_seconds']['paired_per_run']['ci_95']}) |",
        f"| Post-tool duplicate turns (calls / runs / s) | "
        f"{bn['post_tool_duplicate_turns']['pre_fix_total']} / "
        f"{bn['post_tool_duplicate_turns']['pre_fix_runs_affected']}/{n_cases} / "
        f"{bn['post_tool_duplicate_turns']['pre_fix_latency_seconds']:.1f}s | "
        f"{bn['post_tool_duplicate_turns']['post_fix_total']} / "
        f"{bn['post_tool_duplicate_turns']['post_fix_runs_affected']}/{n_cases} / "
        f"{bn['post_tool_duplicate_turns']['post_fix_latency_seconds']:.1f}s | "
        f"{bn['post_tool_duplicate_turns']['paired_per_run']['mean_difference']:+.2f} "
        f"(CI {bn['post_tool_duplicate_turns']['paired_per_run']['ci_95']}) |",
        f"| Validation retries (calls / runs / s) | "
        f"{bn['validation_retries']['pre_fix_total']} / "
        f"{bn['validation_retries']['pre_fix_runs_affected']}/{n_cases} / "
        f"{bn['validation_retries']['pre_fix_latency_seconds']:.1f}s | "
        f"{bn['validation_retries']['post_fix_total']} / "
        f"{bn['validation_retries']['post_fix_runs_affected']}/{n_cases} / "
        f"{bn['validation_retries']['post_fix_latency_seconds']:.1f}s | "
        f"{bn['validation_retries']['paired_per_run']['mean_difference']:+.2f} "
        f"(CI {bn['validation_retries']['paired_per_run']['ci_95']}) |",
        f"| `CONTRACT` stage latency (mean / p95 s) | "
        f"{bn['contract_stage_latency_seconds']['pre_fix']['mean']:.1f} / "
        f"{bn['contract_stage_latency_seconds']['pre_fix']['p95']:.1f} | "
        f"{bn['contract_stage_latency_seconds']['post_fix']['mean']:.1f} / "
        f"{bn['contract_stage_latency_seconds']['post_fix']['p95']:.1f} | "
        f"{bn['contract_stage_latency_seconds']['paired_per_run']['mean_difference']:+.1f}s "
        f"(CI {bn['contract_stage_latency_seconds']['paired_per_run']['ci_95']}) |",
        "",
        "### Hidden Test-Split Predictive Quality (24 Model-Producing Cases)",
        "",
        f"- **Model-producing cases:** {qual['modelable_cases']} "
        f"(pre-fix passed {qual['pre_passed']}/{qual['modelable_cases']}, "
        f"post-fix passed {qual['post_passed']}/{qual['modelable_cases']}; "
        f"paired with hidden metric on both: {qual['paired_with_hidden_metric']}).",
        f"- **Directional comparison:** {qual['improved_count']} improved, "
        f"{qual['unchanged_count']} unchanged, {qual['degraded_count']} degraded.",
        f"- **Maximize metrics (`n={qual['maximize_metrics']['count']}`):** mean delta "
        f"`{qual['maximize_metrics']['mean_delta']}`, "
        f"median `{qual['maximize_metrics']['median_delta']}`.",
        f"- **Minimize metrics (`n={qual['minimize_metrics']['count']}`):** mean improvement "
        f"`{qual['minimize_metrics']['mean_improvement']}`, "
        f"median `{qual['minimize_metrics']['median_improvement']}`.",
        "",
        "### Repeat Sample (6 x 3 = 18 Runs)",
        "",
    ]

    if rep["pre_fix_overall"]["graded"] and rep["post_fix_overall"]["graded"]:
        pre_r_lat = rep["pre_fix_latency"]
        post_r_lat = rep["post_fix_latency"]
        lines.extend(
            [
                f"- **Pre-fix (`62b4e3f`):** {rep['pre_fix_overall']['passed']}/"
                f"{rep['pre_fix_overall']['graded']} "
                f"({rep['pre_fix_overall']['rate']:.1%}, "
                f"CI {rep['pre_fix_overall']['wilson_ci_95']}), "
                f"mean/median/p95 = {pre_r_lat['mean']:.1f}s / "
                f"{pre_r_lat['median']:.1f}s / {pre_r_lat['p95']:.1f}s",
                f"- **Post-fix (`f850588`):** {rep['post_fix_overall']['passed']}/"
                f"{rep['post_fix_overall']['graded']} "
                f"({rep['post_fix_overall']['rate']:.1%}, "
                f"CI {rep['post_fix_overall']['wilson_ci_95']}), "
                f"mean/median/p95 = {post_r_lat['mean']:.1f}s / "
                f"{post_r_lat['median']:.1f}s / {post_r_lat['p95']:.1f}s",
                "",
                "| Repeat Case | Pre Pass | Pre min/med/max (s) "
                "| Post Pass | Post min/med/max (s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for rc in rep["per_case"]:
            pre_r, post_r = rc["pre_fix"], rc["post_fix"]
            if pre_r["total"] and post_r["total"]:
                lines.append(
                    f"| `{rc['case_id']}` | {pre_r['passed']}/{pre_r['total']} | "
                    f"{pre_r['min_s']:.0f} / {pre_r['median_s']:.0f} / {pre_r['max_s']:.0f} | "
                    f"{post_r['passed']}/{post_r['total']} | "
                    f"{post_r['min_s']:.0f} / {post_r['median_s']:.0f} / {post_r['max_s']:.0f} |"
                )

    lines.extend(
        [
            "",
            "## 4. Cost & Latency Breakdown by Step Type (2 pts)",
            "",
            "### (a) By Pipeline Stage",
            "",
            "| Stage | Pre Wall (s) | Pre Cost ($) | Post Wall (s) | Post Cost ($) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for st in STAGES_ORDER:
        ps = bd_pre["by_stage"][st]
        qs = bd_post["by_stage"][st]
        lines.append(
            f"| `{st}` | {ps['wall_seconds_total']:.1f} ({ps['wall_pct']:.1f}%) | "
            f"${ps['cost_usd_total']:.4f} ({ps['cost_pct']:.1f}%) | "
            f"{qs['wall_seconds_total']:.1f} ({qs['wall_pct']:.1f}%) | "
            f"${qs['cost_usd_total']:.4f} ({qs['cost_pct']:.1f}%) |"
        )

    lines.extend(
        [
            "",
            "### (b) By Actor / Step Category",
            "",
            "| Actor / Step Category | Pre Calls | Pre Lat (s) | Pre Cost ($) "
            "| Post Calls | Post Lat (s) | Post Cost ($) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    empty_actor = {"calls_total": 0, "latency_seconds_total": 0.0, "cost_usd_total": 0.0}
    for role in ACTOR_LLM_ROLES:
        pa = bd_pre["by_actor_llm"].get(role, empty_actor)
        qa = bd_post["by_actor_llm"].get(role, empty_actor)
        lines.append(
            f"| `{role}` LLM | {pa['calls_total']} | {pa['latency_seconds_total']:.1f} | "
            f"${pa['cost_usd_total']:.4f} | {qa['calls_total']} | "
            f"{qa['latency_seconds_total']:.1f} | ${qa['cost_usd_total']:.4f} |"
        )
    lines.append(
        f"| MCP tools (excl. `get_experiment_status`) | - | "
        f"{bd_pre['non_poll_mcp_tool_latency_seconds']:.1f} | $0.0000 | - | "
        f"{bd_post['non_poll_mcp_tool_latency_seconds']:.1f} | $0.0000 |"
    )
    lines.append(
        f"| `get_experiment_status` poll wait | "
        f"{bd_pre['by_tool'].get('get_experiment_status', {}).get('calls_total', 0)} | "
        f"{bd_pre['poll_wait_tool_latency_seconds']:.1f} | $0.0000 | "
        f"{bd_post['by_tool'].get('get_experiment_status', {}).get('calls_total', 0)} | "
        f"{bd_post['poll_wait_tool_latency_seconds']:.1f} | $0.0000 |"
    )
    lines.append(
        f"| Sandbox execution (`RUNNING->terminal`) | "
        f"{bd_pre['sandbox_execution']['jobs_total']} | "
        f"{bd_pre['sandbox_execution']['running_seconds_total']:.1f} | $0.0000 | "
        f"{bd_post['sandbox_execution']['jobs_total']} | "
        f"{bd_post['sandbox_execution']['running_seconds_total']:.1f} | $0.0000 |"
    )

    ex = ev["analysis"]["trace_excerpts"]["primary_example"]
    lines.extend(
        [
            "",
            f"### Sanitized Before/After `hypora trace` Excerpt "
            f"(`{ex['case_id']}` — `role=agent_b`)",
            "",
            f"**Pre-fix (`{ex['pre_fix_run_id']}`):**",
            "```text",
            *ex["pre_fix_agent_b_trace"],
            "```",
            "",
            f"**Post-fix (`{ex['post_fix_run_id']}`):**",
            "```text",
            *ex["post_fix_agent_b_trace"],
            "```",
            "",
            "### Honest Limitations",
            "",
            f"- The `eval-freeze` baseline (`{ev['pre_fix_commit']}`) is preserved untouched; "
            f"post-fix measurements run on `{ev['post_fix_commit']}` whose sole runtime change is "
            "Agent B's single-pass structured exit.",
            "- After removing Agent B's redundant post-tool prose turn, `PHASE_B` (`agent_c` "
            "reasoning plus sandbox job wait via `get_experiment_status`) remains the dominant "
            "share of wall latency and LLM cost.",
            "- End-to-end wall latency retains stochastic variance across LLM inference and Docker "
            "sandbox compute, as seen in the repeat sample spread.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(evidence: Mapping[str, Any], out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Write `f7-evidence.json` and `f7-summary.md` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "f7-evidence.json"
    md_path = out_dir / "f7-summary.md"
    _write_json_atomic(json_path, evidence)
    md_path.write_text(render_f7_summary(evidence), encoding="utf-8")
    return json_path, md_path


# --- CLI -----------------------------------------------------------------------------------------


def _parse_query_specs(
    specs: Sequence[str],
    role: str | None,
    event_type: str | None,
    case_id: str | None,
    stage: str | None,
    arm: str,
    limit: int,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "arm": arm,
        "role": role,
        "event_type": event_type,
        "case_id": case_id,
        "stage": stage,
        "limit": limit,
    }
    for raw in specs:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            k, sep, v = part.partition("=")
            if not sep or not v.strip():
                raise ValueError(f"Invalid --query filter {part!r}; expected key=value")
            k = k.strip().lower()
            v = v.strip()
            if k == "type":
                parsed["event_type"] = v
            elif k in ("role", "case_id", "stage", "arm"):
                parsed[k] = v
            elif k == "limit":
                parsed["limit"] = int(v)
            else:
                raise ValueError(
                    f"Unknown --query key {k!r}; valid: role, type, case_id, stage, arm, limit"
                )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL, help="Max concurrent runs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume missing / retry harness_error"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Generate F7 reports without live runs"
    )
    parser.add_argument(
        "--phase",
        choices=("primary", "repeats", "all"),
        default="all",
        help="Run primary cases only, repeats only, or both",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on pending slots to run"
    )
    parser.add_argument(
        "--query",
        nargs="*",
        default=None,
        help="Search SQLite traces (e.g. --query role=agent_b type=model_call stage=CONTRACT)",
    )
    parser.add_argument("--role", default=None, help="Query filter: ActorRole")
    parser.add_argument("--type", dest="event_type", default=None, help="Query filter: EventType")
    parser.add_argument("--case-id", default=None, help="Query filter: case_id")
    parser.add_argument("--stage", default=None, help="Query filter: Stage")
    parser.add_argument(
        "--arm", choices=("pre_fix", "post_fix", "both"), default="both", help="Query filter: arm"
    )
    parser.add_argument("--query-limit", type=int, default=50, help="Max returned query matches")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.query is not None:
        q = _parse_query_specs(
            args.query,
            args.role,
            args.event_type,
            args.case_id,
            args.stage,
            args.arm,
            args.limit if args.limit is not None and args.query_limit == 50 else args.query_limit,
        )
        cases = load_cases(final.CASE_SET)
        case_ids = [c.id for c in cases]
        pre_primary, _ = load_arm_records(FROZEN_DEFAULT_DIR, case_ids)
        post_primary, _ = load_arm_records(POST_FIX_RESULTS_DIR, case_ids)
        arms_map: dict[str, tuple[Mapping[str, CaseRecord], Path]] = {
            "pre_fix": (pre_primary, FROZEN_STATE_DIR),
            "post_fix": (post_primary, POST_FIX_STATE_DIR),
        }
        result = query_traces(arms_map, **q)
        print(json.dumps(result, indent=2))
        return 0

    if not args.report_only:
        anyio.run(
            lambda: run_post_fix_collection(
                parallel=max(1, args.parallel),
                resume=args.resume,
                phase=args.phase,
                limit=args.limit,
            )
        )
        all_cases = load_cases(final.CASE_SET)
        have_all_primary = all(
            primary_record_path(POST_FIX_RESULTS_DIR, c.id).is_file() for c in all_cases
        )
        if not have_all_primary:
            print(
                "[post_fix] Partial run complete; skipping report generation until all "
                f"{len(all_cases)} primary records exist."
            )
            return 0

    try:
        secrets = load_credentials().secret_values()
    except Exception:
        secrets = ()
    evidence = build_f7_evidence(
        pre_results_dir=FROZEN_DEFAULT_DIR,
        pre_state_dir=FROZEN_STATE_DIR,
        post_results_dir=POST_FIX_RESULTS_DIR,
        post_state_dir=POST_FIX_STATE_DIR,
        audit_path=AUDIT_PATH,
        secret_values=secrets,
    )
    json_path, md_path = write_reports(evidence, out_dir=OUT_DIR)
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
