"""Stage 1 gate smoke runs (PLAN.md S1.7; D-016, D-027): two real snapshot tasks, end to end.

    uv run --locked python scripts/stage1_smoke.py run gemini                # free tier
    uv run --locked python scripts/stage1_smoke.py estimate                  # gpt-6-luna dollars
    uv run --locked python scripts/stage1_smoke.py run luna --confirm-paid   # PAID, capped
    uv run --locked python scripts/stage1_smoke.py summary                   # summary.md

Each task runs through the same path as `hypora run` (preflight, controller, MCP server, worker,
sandbox, verifier, reporter) on the `dev` profile, with plan approval by policy. Every run here is
**dev / unrecorded**: a smoke check that the system works, not an evaluation and not a model
selection (D-027). The `eval` profile is never used or changed.

Variants:
- `gemini`: the repo's `dev` profile as configured (gemini-3.5-flash-lite on every tier).
- `luna`: the `dev` profile with every tier mapped to openai/gpt-6-luna **in memory**, for this
  one-off comparison only (D-027). No config file changes. It spends real money, so it needs
  `--confirm-paid`, and the total OpenAI spend recorded in this script's state database is
  capped at `--max-total-usd` (default $1.00, the owner's hard cap): each run's budget is the
  remaining cap minus a safety margin, split over the runs still to do.

State lives in `runs/stage1/` (git-ignored); `hypora status|trace --db runs/stage1/hypora.sqlite`
inspect a run. Per-run records (token counts, cost, event-log size, outcome) and each run's
report are written to `results/stage1/<variant>/`, and `summary` writes `results/stage1/summary.md`.
"""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import anyio

from hypora.cli.run import PreflightError, RunOptions, execute, result_lines
from hypora.cli.status import final_model
from hypora.core.config import ModelsConfig, load_settings
from hypora.core.config.run_config import RunConfig
from hypora.core.config.settings import HyporaSettings
from hypora.core.llm.errors import OUTPUT_SCHEMA_INVALID
from hypora.core.llm.pricing import ModelPrice, load_price_table
from hypora.core.llm.types import Usage
from hypora.core.schemas import ApprovalPolicy, EventType, ModelProfile, ModelTier, Provider
from hypora.core.schemas.events import (
    CallStatus,
    ModelCallPayload,
    StageTransitionPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import (
    RecordNotFound,
    list_runs,
    list_trials,
    load_experiment_contract,
    load_run,
    run_jobs,
)
from hypora.storage.snapshots import SNAPSHOTS_DIR, SnapshotEntry, load_manifest, verify_entry

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
STATE_ROOT: Final = REPO_ROOT / "runs" / "stage1"
RESULTS_ROOT: Final = REPO_ROOT / "results" / "stage1"
RECORD_SCHEMA: Final = "hypora.stage1_smoke/v1"
DEFAULT_CAP_USD: Final = 1.00
SAFETY_MARGIN_USD: Final = 0.05
"""Kept free under the cap: a run's budget is checked before each call, so the call that crosses
it still completes. One gpt-6-luna call at these prompt sizes costs well under a cent."""
MAX_RUN_USD: Final = 0.45
"""Upper bound on one paid run's budget, so the first run cannot use the whole cap."""


@dataclass(frozen=True)
class SmokeTask:
    """One fixture task: a real snapshot (D-015, D-021) and the request made on it."""

    task_id: str
    snapshot: str
    sha256: str
    goal: str
    target: str


TASKS: Final = (
    SmokeTask(
        task_id="credit-g",
        snapshot="openml-46918-credit-g",
        sha256="ef789622335b73bbe9f26f06a5557cec6f29118cb88bb58d82ba234e4a1d5270",
        goal="Predict whether a loan applicant is a good or bad credit risk",
        target="good_or_bad_customer",
    ),
    SmokeTask(
        task_id="fish-toxicity",
        snapshot="openml-46954-qsar-fish-toxicity",
        sha256="3b6dba16ff93c7edb9d63cf23c44be5ea26e76faedbe0d4db9250284f240d98e",
        goal="Predict a chemical's acute toxicity to fish (LC50) from its molecular descriptors",
        target="LC50",
    ),
)


@dataclass(frozen=True)
class Variant:
    """Which model serves every tier of the `dev` profile for a smoke run."""

    name: str
    provider: Provider
    model_id: str
    temperature: float | Literal["not_configurable"]
    paid: bool
    remap: bool
    """True: map every dev tier to this model in memory (D-027). False: the repo's dev profile."""


VARIANTS: Final = {
    "gemini": Variant(
        "gemini", Provider.GEMINI, "gemini-3.5-flash-lite", 1.0, paid=False, remap=False
    ),
    "luna": Variant(
        "luna", Provider.OPENAI, "gpt-6-luna", "not_configurable", paid=True, remap=True
    ),
}


class SmokeError(Exception):
    """The smoke run cannot proceed; the message says why."""


# --- configuration --------------------------------------------------------------------------------


def settings_for(variant: Variant) -> HyporaSettings:
    """Repo settings on the dev profile with policy approval; for `remap`, dev tiers -> variant.

    The eval profile is checked to be byte-for-byte the configured one, so this script can never
    pin, edit, or run it (D-027).
    """
    base = {
        "model_profile": ModelProfile.DEV,
        "flags": {"approval_policy": ApprovalPolicy.AUTO_WITHIN_CONTRACT.value},
    }
    configured = load_settings(overrides=base)
    if not variant.remap:
        settings = configured
    else:
        data = configured.models.model_dump(mode="json")
        tier = {
            "provider": variant.provider.value,
            "model_id": variant.model_id,
            "temperature": variant.temperature,
        }
        data["dev"]["tiers"] = {"lead": tier, "worker": tier, "cheap": tier}
        data["dev"]["description"] = (
            f"S1.7 one-off smoke comparison (D-027): every tier {variant.provider.value}/"
            f"{variant.model_id}; dev/unrecorded, not a model selection."
        )
        models = ModelsConfig.model_validate(data)
        settings = load_settings(overrides={**base, "models": models.model_dump(mode="json")})
    if settings.models.eval != configured.models.eval:
        raise SmokeError("the eval profile changed; this script must never touch it (D-027)")
    tiers = settings.models.dev.tiers
    served = {(tiers.for_tier(t).provider, tiers.for_tier(t).model_id) for t in ModelTier}
    if served != {(variant.provider, variant.model_id)}:
        raise SmokeError(
            f"variant {variant.name!r} expects every dev tier on {variant.provider.value}/"
            f"{variant.model_id}, but the dev profile gives {sorted(served)}"
        )
    return settings


# --- snapshots and spend --------------------------------------------------------------------------


def snapshot_for(task: SmokeTask, directory: Path = SNAPSHOTS_DIR) -> SnapshotEntry:
    """The task's manifest entry, after checking the stored file is the expected snapshot."""
    entry = load_manifest(directory).get(task.snapshot)
    fetch = f"make snapshot SOURCE=openml ID={task.snapshot.split('-')[1]}"
    if entry is None:
        raise SmokeError(f"snapshot {task.snapshot} is not in the manifest; run `{fetch}`")
    if entry.sha256 != task.sha256:
        raise SmokeError(
            f"snapshot {task.snapshot} has SHA-256 {entry.sha256}, expected {task.sha256}: "
            "the source data changed, so these runs would not be comparable"
        )
    problem = verify_entry(entry, directory)
    if problem is not None:
        raise SmokeError(f"snapshot {task.snapshot}: {problem}; re-fetch it with `{fetch}`")
    return entry


def provider_spend_usd(database: Database, provider: Provider) -> float:
    """List-price cost of every model call to `provider` in this state database (all runs)."""
    only_calls = EventFilters(types=(EventType.MODEL_CALL,))
    total = 0.0
    for run in list_runs(database, limit=10_000):
        for event in EventLog(database).read(run.run_id, only_calls):
            payload = event.payload
            if isinstance(payload, ModelCallPayload) and payload.provider is provider:
                total += payload.cost_usd
    return total


def paid_run_budget(spent_usd: float, cap_usd: float, runs_left: int) -> float:
    """The next paid run's budget: the cap minus spend and margin, shared by the runs left."""
    available = cap_usd - SAFETY_MARGIN_USD - spent_usd
    if runs_left < 1 or available <= 0:
        raise SmokeError(
            f"OpenAI spend so far ${spent_usd:.4f} leaves no room under the ${cap_usd:.2f} cap "
            f"(margin ${SAFETY_MARGIN_USD:.2f}); not starting another paid run"
        )
    return round(min(MAX_RUN_USD, available / runs_left), 4)


# --- records --------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def collect_record(
    database: Database,
    runs_root: Path,
    run_id: str,
    *,
    task: SmokeTask,
    snapshot: SnapshotEntry,
    variant: Variant,
    run_config: RunConfig,
    budget_usd: float | None,
    code: dict[str, Any],
) -> dict[str, Any]:
    """Everything S1.7 reports about one run, read back from the event log and the records."""
    events = EventLog(database).read(run_id)
    run = load_run(database, run_id)
    calls = [
        (e.actor_role.value, e.payload) for e in events if isinstance(e.payload, ModelCallPayload)
    ]
    payloads = [payload for _, payload in calls]
    results = [e.payload for e in events if isinstance(e.payload, ToolResultPayload)]
    stages = [e.payload for e in events if isinstance(e.payload, StageTransitionPayload)]
    metric = None
    with contextlib.suppress(RecordNotFound):
        metric = load_experiment_contract(database, run_id).primary_metric.value
    final = final_model(database, run_id, metric)
    by_role: dict[str, dict[str, float]] = {}
    for role, payload in calls:
        row = by_role.setdefault(
            role,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        row["calls"] += 1
        row["input_tokens"] += payload.input_tokens
        row["output_tokens"] += payload.output_tokens
        row["cost_usd"] += payload.cost_usd
    failed_tools = Counter(
        f"{r.tool}:{r.error.code if r.error else r.status.value}"
        for r in results
        if r.status is not ToolResultStatus.OK
    )
    span = (events[-1].created_at - events[0].created_at).total_seconds() if events else 0.0
    trials = list_trials(database, run_id)
    jobs = run_jobs(database, run_id)
    return {
        "schema": RECORD_SCHEMA,
        "step": "S1.7",
        "recorded": False,
        "label": "dev/unrecorded smoke run (D-016, D-027); not an evaluation or model selection",
        "variant": variant.name,
        "profile": run_config.model_profile.value,
        "models": sorted({f"{p.provider.value}/{p.model_id}" for p in payloads}),
        "billing": "paid: list price is what OpenAI bills"
        if variant.paid
        else "free tier: actual spend $0; cost_usd is the paid-tier list price",
        "code": code,
        "config_hash": run_config.hash,
        "task": {"task_id": task.task_id, "goal": task.goal, "target": task.target},
        "dataset": {
            "snapshot": snapshot.name,
            "source": snapshot.source.value,
            "dataset_id": snapshot.dataset_id,
            "title": snapshot.title,
            "version": snapshot.version,
            "license": snapshot.license,
            "source_url": snapshot.source_url,
            "rows": snapshot.rows,
            "columns": snapshot.columns,
            "sha256": snapshot.sha256,
        },
        "run": {
            "run_id": run_id,
            "status": run.status.value,
            "last_stage": stages[-1].stage.value if stages and stages[-1].stage else None,
            "verified": bool(final and final.verified),
            "final_trial_id": final.trial_id if final else None,
            "test_metric": final.metric if final else metric,
            "test_value": final.value if final else None,
            "trials": dict(Counter(t.status for t in trials)),
            "jobs": dict(Counter(j.status.value for j in jobs)),
            "wall_clock_s": round(span, 1),
            "sandbox_compute_s": float(run.spent_compute_s),
            "budget_usd": budget_usd,
        },
        "llm": {
            "calls": len(payloads),
            "calls_failed": sum(p.status is CallStatus.ERROR for p in payloads),
            "schema_retries": sum(p.error_code == OUTPUT_SCHEMA_INVALID for p in payloads),
            "retries": sum(p.retries for p in payloads),
            "input_tokens": sum(p.input_tokens for p in payloads),
            "cached_input_tokens": sum(p.cached_input_tokens for p in payloads),
            "output_tokens": sum(p.output_tokens for p in payloads),
            "cost_usd": round(sum(p.cost_usd for p in payloads), 6),
            "spent_usd_ledger": float(run.spent_usd),
            "latency_s": round(sum(p.latency_ms for p in payloads) / 1000, 1),
            "rate_limit_wait_s": round(sum(p.rate_limit_wait_ms for p in payloads) / 1000, 1),
            "by_role": {
                role: {**row, "cost_usd": round(row["cost_usd"], 6)}
                for role, row in sorted(by_role.items())
            },
            "per_call": [
                {
                    "role": role,
                    "tier": p.tier.value,
                    "input_tokens": p.input_tokens,
                    "cached_input_tokens": p.cached_input_tokens,
                    "output_tokens": p.output_tokens,
                    "status": p.status.value,
                }
                for role, p in calls
            ],
        },
        "tools": {
            "calls": sum(e.type is EventType.TOOL_CALL for e in events),
            "failed": dict(sorted(failed_tools.items())),
        },
        "events": {
            "count": len(events),
            "bytes": sum(len(e.model_dump_json()) for e in events),
            "by_type": dict(sorted(Counter(e.type.value for e in events).items())),
        },
    }


def write_record(record: dict[str, Any], report_md: Path | None, root: Path = RESULTS_ROOT) -> Path:
    """`results/stage1/<variant>/<task>.json`, plus the run's report as `<task>.report.md`."""
    folder = root / record["variant"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['task']['task_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if report_md is not None and report_md.is_file():
        shutil.copyfile(report_md, folder / f"{record['task']['task_id']}.report.md")
    return path


def load_records(variant: str, root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    """The stored records of one variant, in task order."""
    records = []
    for task in TASKS:
        path = root / variant / f"{task.task_id}.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


# --- cost estimates -------------------------------------------------------------------------------


def calls_cost_usd(
    calls: Iterable[dict[str, Any]], price: ModelPrice, output_factor: float
) -> float:
    """What these calls' token counts would cost at `price`, with output tokens scaled.

    Cached tokens are billed at the full input price (no caching assumed: an upper bound).
    """
    total = 0.0
    for call in calls:
        usage = Usage(
            input_tokens=call["input_tokens"],
            output_tokens=round(call["output_tokens"] * output_factor),
        )
        total += price.cost_usd(usage)
    return total


def estimate_lines(records: list[dict[str, Any]], price: ModelPrice) -> list[str]:
    """Per-task and total estimate of the gemini runs' token counts at `price`."""
    if len(records) != len(TASKS):
        raise SmokeError("estimate needs a gemini record for every task; run `run gemini` first")
    lines = [
        f"Estimate for {price.provider.value}/{price.model_id} from the gemini runs' token counts "
        f"(prices from configs/prices.yaml, read {price.retrieved_on}; no prompt caching assumed):",
    ]
    factors = (1.0, 3.0)
    totals = dict.fromkeys(factors, 0.0)
    for record in records:
        calls = record["llm"]["per_call"]
        costs = {f: calls_cost_usd(calls, price, f) for f in factors}
        for factor, cost in costs.items():
            totals[factor] += cost
        lines.append(
            f"  {record['task']['task_id']:<14} {len(calls)} calls, "
            f"{record['llm']['input_tokens']:,} in / {record['llm']['output_tokens']:,} out: "
            f"${costs[1.0]:.4f} (same output) | ${costs[3.0]:.4f} (3x output)"
        )
    lines.append(
        f"  total          ${totals[1.0]:.4f} (same output) | ${totals[3.0]:.4f} (3x output)"
    )
    return lines


# --- commands -------------------------------------------------------------------------------------


async def _run_variant(variant: Variant, cap_usd: float, out: Callable[[str], None]) -> int:
    settings = settings_for(variant)
    run_config = RunConfig.build(settings)
    snapshots = [snapshot_for(task) for task in TASKS]
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = STATE_ROOT / "hypora.sqlite"
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results/stage1")
    code = {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status)}
    failures = 0
    for index, (task, snapshot) in enumerate(zip(TASKS, snapshots, strict=True)):
        budget = None
        if variant.paid:
            database = Database(db_path)
            try:
                migrate(database)
                spent = provider_spend_usd(database, variant.provider)
            finally:
                database.close()
            budget = paid_run_budget(spent, cap_usd, len(TASKS) - index)
            out(
                f"[paid] {variant.provider.value} spend so far ${spent:.4f}; "
                f"this run's budget ${budget:.4f}"
            )
        run_id = f"run_s17_{variant.name}_{task.task_id.replace('-', '_')}_{int(time.time())}"
        options = RunOptions(
            data=SNAPSHOTS_DIR / snapshot.file,
            goal=task.goal,
            target=task.target,
            db=db_path,
            runs_root=STATE_ROOT,
            budget_usd=budget,
            approve=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
            run_id=run_id,
        )
        out(f"== {variant.name} / {task.task_id} ({snapshot.name}, {snapshot.license})")
        result = await execute(options, out=out, run_config=run_config)
        for line in result_lines(result):
            out(line)
        database = Database(db_path)
        try:
            record = collect_record(
                database,
                STATE_ROOT,
                result.run_id,
                task=task,
                snapshot=snapshot,
                variant=variant,
                run_config=run_config,
                budget_usd=budget,
                code=code,
            )
        finally:
            database.close()
        path = write_record(record, result.report_md)
        llm = record["llm"]
        out(
            f"-> {path.relative_to(REPO_ROOT)}: {llm['calls']} model calls, "
            f"{llm['input_tokens']:,} in / {llm['output_tokens']:,} out, ${llm['cost_usd']:.4f}, "
            f"{record['events']['count']} events ({record['events']['bytes']:,} bytes)"
        )
        failures += not result.verified
    return 1 if failures else 0


def summary_markdown(by_variant: dict[str, list[dict[str, Any]]]) -> str:
    """`results/stage1/summary.md`: one row per run, then the per-task comparison."""
    lines = [
        "# Stage 1 gate smoke runs (S1.7)",
        "",
        "Generated by `scripts/stage1_smoke.py summary` from the records in this folder. Every",
        "run is **dev / unrecorded** (D-016, D-027): a check that the system works end to end, not",
        "an evaluation and not a model selection. Costs are list prices; Gemini free-tier runs",
        "cost $0.",
        "",
        "| Variant | Task | Model | Status | Verified | Test metric | Trials | Model calls"
        " | Tokens in / out | Cost (list) | Tool calls / failed | Events (bytes) | Wall clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, records in by_variant.items():
        for r in records:
            run, llm = r["run"], r["llm"]
            metric = (
                f"{run['test_metric']} {run['test_value']:.4f}"
                if run["test_value"] is not None
                else "-"
            )
            lines.append(
                f"| {variant} | {r['task']['task_id']} | {', '.join(r['models'])} | "
                f"{run['status']} | {'yes' if run['verified'] else 'no'} | {metric} | "
                f"{sum(run['trials'].values())} | {llm['calls']} | "
                f"{llm['input_tokens']:,} / {llm['output_tokens']:,} | ${llm['cost_usd']:.4f} | "
                f"{r['tools']['calls']} / {sum(r['tools']['failed'].values())} | "
                f"{r['events']['count']} ({r['events']['bytes']:,}) | {run['wall_clock_s']:.0f} s |"
            )
    lines += ["", "Run IDs (inspect with `hypora trace <id> --db runs/stage1/hypora.sqlite`):", ""]
    for variant, records in by_variant.items():
        for r in records:
            lines.append(f"- {variant} / {r['task']['task_id']}: `{r['run']['run_id']}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="scripts/stage1_smoke.py", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run both tasks on one variant")
    run.add_argument("variant", choices=sorted(VARIANTS))
    run.add_argument("--confirm-paid", action="store_true", help="required for a paid variant")
    run.add_argument("--max-total-usd", type=float, default=DEFAULT_CAP_USD)
    estimate = commands.add_parser("estimate", help="price the gemini runs' tokens for a model")
    estimate.add_argument("--model", default="openai/gpt-6-luna", help="provider/model_id")
    commands.add_parser("summary", help="write results/stage1/summary.md")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            variant = VARIANTS[args.variant]
            if variant.paid and not args.confirm_paid:
                raise SmokeError(f"variant {variant.name!r} spends real money; add --confirm-paid")
            if not 0 < args.max_total_usd <= DEFAULT_CAP_USD:
                raise SmokeError(f"--max-total-usd must be in (0, {DEFAULT_CAP_USD:.2f}]")
            return anyio.run(_run_variant, variant, args.max_total_usd, print)
        if args.command == "estimate":
            provider, _, model_id = args.model.partition("/")
            price = load_price_table().require(Provider(provider), model_id)
            print("\n".join(estimate_lines(load_records("gemini"), price)))
            return 0
        by_variant = {name: load_records(name) for name in VARIANTS}
        path = RESULTS_ROOT / "summary.md"
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text(summary_markdown(by_variant), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0
    except (SmokeError, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
