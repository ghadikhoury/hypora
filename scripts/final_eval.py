"""S5.2 shared final run matrix: the frozen test set, three systems, one $10 paid collection.

PLAN.md S5.2; spec §19.5; the frozen protocol in `docs/notes/test-set-protocol.md`; run book in
`docs/notes/final-eval.md`. The same real controller, worker, sandbox, hidden splits and
programmatic graders as `make eval`, through the same runner, on the `eval-freeze` commit.

Arms (one primary run per case per system, never mixed with the repeat sample):
- `multi`: config `default` (A/B/C), system `agent`, paid.
- `single`: config `single_agent`, system `agent`, paid.
- `baseline`: the fixed pipeline, recorded beside `multi` in the `default` results segment. It makes
  no model call, so it runs outside the paid ledger.
The repeat sample is 3 additional runs per system of the protocol's six cases (`REPEAT_CASES`); the
primary run is not repeat #1.

Spending: one durable ledger (`scripts/stage2_eval.py`, D-054) with a hard $10 cap for every paid
attempt of S5.2. Every actual SDK attempt, retries included, reserves $0.36 before dispatch and is
refused once measured spend plus outstanding reservations plus its own $0.36 would pass $10. The
case budgets ($0.60 each, 88 paid runs) add up to more than the cap, so the runner's full-budget
pre-check cannot apply; the per-attempt reservation is the guarantee (as for S4.2f, D-052). A run
the cap refuses, before it starts or mid-run, is a `harness_error` (not measured), never a graded
failure; `--resume` retries it later with the earlier attempt archived.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import anyio

from hypora.cli.eval import EvalOptions, build_plan, load_cases
from hypora.cli.run import PreflightError
from hypora.core.config import load_eval_config, load_settings_for_eval
from hypora.core.config.run_config import RunConfig
from hypora.core.schemas import EventType, LeakageCheckPayload, ModelProfile, ModelTier
from hypora.core.schemas.leakage import LeakageOutcome, LeakageStep
from hypora.evaluation.baselines.fixed_pipeline import SYSTEM_NAME as BASELINE_SYSTEM
from hypora.evaluation.case import Case
from hypora.evaluation.runner.agent import AGENT_SYSTEM
from hypora.evaluation.runner.records import CaseRecord, RunKind
from hypora.evaluation.runner.runner import (
    EvalOutcome,
    EvalPlan,
    ReportsRetryWaits,
    parallel_limit,
    run_eval,
)
from hypora.evaluation.runner.store import RESULTS_ROOT, EvalRefused, GitRepo, ResultsStore
from hypora.evaluation.runner.usage import collect_usage
from hypora.evaluation.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_RESAMPLES,
    LATENCY_METHOD,
    cost_per_task,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)
from hypora.evaluation.system import EvalSystem, SystemContext, SystemInput, SystemRun
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR, load_manifest
from scripts.readiness_eval import LOCK, current_lock
from scripts.stage2_eval import (
    ATTEMPT_RESERVATION_USD,
    AccountedAgent,
    AdapterFactory,
    CollectionRefused,
    SpendError,
    SpendLedger,
    check_published_history,
)

ROOT: Final = REPO_ROOT
STATE: Final = ROOT / "runs" / "final-s5.2"
"""Git-ignored: the paid journal, the collection lock, and the runs/state databases (`work/`)."""
JOURNAL: Final = STATE / "paid-attempts.jsonl"
WORK_ROOT: Final = STATE / "work"
FINAL_DIR: Final = ROOT / "results" / "final"
AUDIT: Final = FINAL_DIR / "s5.2-paid-accounting.json"
MANIFEST: Final = FINAL_DIR / "s5.2-manifest.json"
ORPHANS: Final = FINAL_DIR / "s5.2-orphans"
SUMMARY_JSON: Final = FINAL_DIR / "s5.2-summary.json"
SUMMARY_MD: Final = FINAL_DIR / "s5.2-summary.md"
CAP_USD: Final = 10.0
"""Owner-approved hard cap for every paid model attempt of S5.2 (its own budget)."""
EXPECTED_RUN_USD: Final = 0.035
"""Recorded S4.2 runs cost about this much each; used only for the printed estimate."""
RUN_MINUTES: Final = (5, 13)
BASELINE_MINUTES: Final = (1, 4)
FREEZE_TAG: Final = "eval-freeze"
CASE_SET: Final = "test"
TEST_CASES: Final = 26
REPEAT_CASES: Final = (
    "test_leak_heart_failure",
    "test_group_concrete",
    "test_time_beijing_pm25",
    "test_imb_churn",
    "test_clean_rice",
    "test_fault_submit_spambase",
)
"""The protocol's repeat sample: one representative case per family (test-set-protocol.md)."""
REPEATS: Final = 3
"""Additional runs per system of each repeat case, reported apart from the primary runs."""
LEAKAGE_TRAP: Final = "target_leakage"
EXCLUSION_OUTCOMES: Final = frozenset(
    {LeakageOutcome.EXCLUDED, LeakageOutcome.EXCLUSION_DENIED, LeakageOutcome.EXCLUSION_FAILED}
)


class Refused(Exception):
    """The collection must not start; nothing was run or spent."""


@dataclass(frozen=True)
class Arm:
    """One system of the comparison: which config's results segment and which runner system."""

    name: str
    config: str
    system: str
    paid: bool


ARMS: Final = {
    "multi": Arm("multi", "default", AGENT_SYSTEM, paid=True),
    "single": Arm("single", "single_agent", AGENT_SYSTEM, paid=True),
    "baseline": Arm("baseline", "default", BASELINE_SYSTEM, paid=False),
}
PAIRS: Final = (("multi", "baseline"), ("single", "baseline"), ("multi", "single"))
"""Paired comparisons the protocol reports, first minus second."""


class Pass(StrEnum):
    """The primary pass (every case once) or the repeat sample (six cases x 3)."""

    PRIMARY = "primary"
    REPEAT = "repeat"


@dataclass(frozen=True)
class Slot:
    """One run the matrix needs."""

    arm: str
    case_id: str
    kind: RunKind
    attempt: int


def planned_slots(arm: Arm, which: Pass, case_ids: Sequence[str]) -> list[Slot]:
    """Every run of one arm's pass, in the order the runner schedules them."""
    if which is Pass.PRIMARY:
        return [Slot(arm.name, case, RunKind.PRIMARY, 1) for case in sorted(case_ids)]
    return [
        Slot(arm.name, case, RunKind.REPEAT, attempt)
        for case in sorted(REPEAT_CASES)
        for attempt in range(1, REPEATS + 1)
    ]


def store_for(arm: Arm, commit: str, results_root: Path = RESULTS_ROOT) -> ResultsStore:
    """The arm's results directory on the frozen commit."""
    return ResultsStore(results_root, commit, arm.config)


def slot_path(store: ResultsStore, arm: Arm, slot: Slot) -> Path:
    """Where the slot's record is written."""
    if slot.kind is RunKind.PRIMARY:
        return store.primary_path(arm.system, slot.case_id)
    return store.repeat_path(arm.system, slot.case_id, slot.attempt)


def slot_state(store: ResultsStore, arm: Arm, slot: Slot) -> str:
    """`graded`, `harness_error` (to be retried with --resume), or `missing`."""
    record = store.read(slot_path(store, arm, slot))
    if record is None:
        return "missing"
    return "graded" if record.finished else "harness_error"


# --- Identity ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """The checked-out commit and the commit the freeze tag names (None when the tag is absent)."""

    head: str
    dirty: bool
    tag: str
    tag_commit: str | None

    @property
    def frozen(self) -> bool:
        """Clean checkout of exactly the tagged commit."""
        return not self.dirty and self.tag_commit == self.head


def tag_commit(tag: str, root: Path = ROOT) -> str | None:
    """The commit a tag points to, or None."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def read_identity(tag: str = FREEZE_TAG, repo: GitRepo | None = None) -> Identity:
    """HEAD, its dirtiness (results/ aside, as the runner counts it), and the tag's commit."""
    git = repo or GitRepo()
    state = git.state()
    return Identity(state.commit, state.dirty, tag, tag_commit(tag, git.root))


def check_identity(identity: Identity) -> None:
    """Recorded S5.2 evidence comes only from a clean checkout of the freeze tag (S5.1)."""
    if identity.tag_commit is None:
        raise Refused(f"tag {identity.tag!r} does not exist here; `git fetch --tags` first")
    if identity.head != identity.tag_commit:
        raise Refused(
            f"HEAD {identity.head[:12]} is not {identity.tag} ({identity.tag_commit[:12]}); "
            f"check out the tag (`git switch --detach {identity.tag}`)"
        )
    if identity.dirty:
        raise Refused("the working tree differs from the freeze commit (results/ aside)")


def check_lock() -> str:
    """The frozen configuration lock still describes the code; returns its SHA-256."""
    committed = json.loads(LOCK.read_text())
    if current_lock() != committed:
        raise Refused(f"{LOCK} does not match the code's prompts, budgets, flags or models")
    return hashlib.sha256(LOCK.read_bytes()).hexdigest()


def check_case_set(cases: Sequence[Case], expected: int = TEST_CASES) -> None:
    """The frozen held-out set: exactly its cases, all test split, with the repeat sample in it."""
    ids = {case.id for case in cases}
    if len(cases) != expected:
        raise Refused(f"the test set has {len(cases)} cases; the frozen protocol has {expected}")
    if any(case.split.value != "test" for case in cases):
        raise Refused("a case in the test set directory is not a test-split case")
    missing = sorted(set(REPEAT_CASES) - ids)
    if missing:
        raise Refused(f"repeat-sample cases are not in the set: {missing}")


def case_set_sha256(cases: Sequence[Case]) -> str:
    """One hash over every case's content hash, by case ID."""
    joined = "\n".join(f"{case.id} {case.content_hash()}" for case in sorted(cases, key=_id))
    return hashlib.sha256(joined.encode()).hexdigest()


def _id(case: Case) -> str:
    return case.id


def config_identity(config: str) -> dict[str, Any]:
    """The eval-profile run configuration a config resolves to, by hash (machine pools included)."""
    settings = load_settings_for_eval(config, profile=ModelProfile.EVAL)
    run_config = RunConfig.build(settings, eval_config=config)
    tiers = run_config.active_profile.tiers
    return {
        "run_config_hash": run_config.hash,
        "agent_mode": run_config.flags.agent_mode.value,
        "profile": run_config.model_profile.value,
        "tiers": tiers.model_dump(mode="json"),
        "fallback_enabled": run_config.active_profile.fallback.enabled,
    }


def manifest_for(identity: Identity, cases: Sequence[Case], lock_sha256: str) -> dict[str, Any]:
    """What every S5.2 record must share; a later collection that differs is refused."""
    return {
        "step": "S5.2",
        "freeze_tag": identity.tag,
        "commit": identity.head,
        "case_set": CASE_SET,
        "case_set_sha256": case_set_sha256(cases),
        "cases": {case.id: case.content_hash() for case in sorted(cases, key=_id)},
        "repeat_cases": sorted(REPEAT_CASES),
        "repeats": REPEATS,
        "readiness_lock_sha256": lock_sha256,
        "configs": {arm.config: config_identity(arm.config) for arm in ARMS.values()},
        "cap_usd": CAP_USD,
        "attempt_reservation_usd": ATTEMPT_RESERVATION_USD,
    }


def record_manifest(manifest: Mapping[str, Any], path: Path = MANIFEST) -> None:
    """Write the manifest once; refuse a collection whose identity differs from the first one."""
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            changed = sorted(
                k for k in {*existing, *manifest} if existing.get(k) != manifest.get(k)
            )
            raise Refused(f"{path} was written for a different collection identity: {changed}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# --- Spending ------------------------------------------------------------------------------------


def headroom_usd(ledger: SpendLedger) -> float:
    """What the cap still admits after measured spend and every outstanding reservation."""
    state = ledger.snapshot()
    return ledger.cap_usd - state.measured_usd - state.reserved_usd


class CapGatedAgent:
    """Refuse a paid run before it starts once the cap can't admit one more attempt.

    The ledger already refuses each attempt past the cap (and `AccountedAgent` then records the run
    as a harness error); this gate only avoids starting runs that could not make a single call.
    """

    def __init__(self, inner: EvalSystem, ledger: SpendLedger) -> None:
        self.inner = inner
        self.ledger = ledger
        self.name = inner.name

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        """Run the agent, or raise `CollectionRefused` (a harness error, D-054) at the cap."""
        if headroom_usd(self.ledger) < ATTEMPT_RESERVATION_USD:
            self.ledger.note_refusal(context.run_id)
            raise CollectionRefused(
                "S5.2 collection cap reached before the run started; case not measured"
            )
        return await self.inner.run(system_input, context)

    def unobserved_spend_reserved_usd(self, run_id: str) -> float:
        """Unknown provider charges the ledger still reserves for this run."""
        return self.ledger.unobserved(run_id)

    def retry_wait_seconds(self, run_id: str) -> float | None:
        """The inner system's retry backoff, when it can say."""
        if isinstance(self.inner, ReportsRetryWaits):
            return self.inner.retry_wait_seconds(run_id)
        return None


def gate_plan(plan: EvalPlan, ledger: SpendLedger, wrap: Callable[[], EvalSystem]) -> EvalPlan:
    """Route the plan's agent through the ledger and drop the full-budget pre-check.

    The runner's `max_total_usd` check refuses when every pending run at its full case budget
    could pass the cap; S5.2's budgets exceed $10, so the ledger's per-attempt reservation is the
    hard limit instead.
    """
    systems = tuple(
        CapGatedAgent(wrap(), ledger) if system.name == AGENT_SYSTEM else system
        for system in plan.systems
    )
    return replace(plan, systems=systems, max_total_usd=None)


@contextmanager
def collection_lock(state: Path = STATE) -> Iterator[None]:
    """Only one S5.2 paid collector at a time (both paid arms share one ledger)."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / "collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused("another S5.2 collector is already running") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


# --- Collection ----------------------------------------------------------------------------------


def eval_options(
    arm: Arm, which: Pass, *, parallel: int, resume: bool, cases: str = CASE_SET
) -> EvalOptions:
    """The runner options of one arm's pass. A repeat pass resumes past its finished primaries."""
    repeat = which is Pass.REPEAT
    return EvalOptions(
        cases=cases,
        config=arm.config,
        systems=(arm.system,),
        profile=ModelProfile.EVAL,
        only=",".join(REPEAT_CASES) if repeat else None,
        parallel=parallel,
        resume=resume or repeat,
        repeats=REPEATS if repeat else 0,
        repeat_cases=REPEAT_CASES if repeat else (),
        max_total_usd=CAP_USD,
        work_root=WORK_ROOT,
    )


def check_pass(store: ResultsStore, arm: Arm, which: Pass, resume: bool) -> None:
    """A repeat pass needs its six primaries graded, and retries only with --resume.

    (A primary pass needs no check here: the runner itself refuses finished records without
    --resume, and replaces only harness errors, archiving them.)
    """
    if which is not Pass.REPEAT:
        return
    primaries = [Slot(arm.name, case, RunKind.PRIMARY, 1) for case in REPEAT_CASES]
    unfinished = [s.case_id for s in primaries if slot_state(store, arm, s) != "graded"]
    if unfinished:
        raise Refused(
            f"{arm.name}: finish the primary pass first; not graded yet: {sorted(unfinished)}"
        )
    started = [
        s for s in planned_slots(arm, which, REPEAT_CASES) if slot_state(store, arm, s) != "missing"
    ]
    if started and not resume:
        raise Refused(
            f"{arm.name}: {len(started)} repeat record(s) exist; pass --resume to keep them and "
            "retry only the unmeasured ones"
        )


async def run_arms(
    plans: Mapping[str, EvalPlan], out: Callable[[str], None] = print
) -> dict[str, EvalOutcome]:
    """Run each arm's plan; several arms run side by side, each in its own results segment."""
    outcomes: dict[str, EvalOutcome] = {}

    async def one(name: str, plan: EvalPlan) -> None:
        outcomes[name] = await run_eval(plan, lambda line: out(f"[{name}] {line}"))

    async with anyio.create_task_group() as group:
        for name, plan in plans.items():
            group.start_soon(one, name, plan)
    return outcomes


def selected_arms(choice: str) -> tuple[Arm, ...]:
    """`both` is the two paid arms side by side; the baseline always runs on its own."""
    if choice == "both":
        return ARMS["multi"], ARMS["single"]
    return (ARMS[choice],)


def collect(arm_choice: str, which: Pass, *, parallel: int, resume: bool, tag: str) -> int:
    """Collect one pass of one arm (or both paid arms) on the freeze commit."""
    arms = selected_arms(arm_choice)
    identity = read_identity(tag)
    check_identity(identity)
    lock_sha = check_lock()
    cases = load_cases(CASE_SET)
    check_case_set(cases)
    manifest = manifest_for(identity, cases, lock_sha)
    record_manifest(manifest)
    for arm in arms:
        check_pass(store_for(arm, identity.head), arm, which, resume)
    share = max(1, parallel // len(arms))

    def plan_of(arm: Arm) -> EvalPlan:
        plan = build_plan(eval_options(arm, which, parallel=share, resume=resume), print)
        if plan.segment != arm.config:
            raise Refused(f"{arm.name}: results segment {plan.segment!r} is not {arm.config!r}")
        if plan.run_config.hash != manifest["configs"][arm.config]["run_config_hash"]:
            raise Refused(f"{arm.name}: run configuration differs from the S5.2 manifest")
        if {case.id for case in plan.cases} - set(manifest["cases"]):
            raise Refused(f"{arm.name}: the plan has cases outside the S5.2 manifest")
        return plan

    if not any(arm.paid for arm in arms):
        outcomes = anyio.run(run_arms, {arm.name: plan_of(arm) for arm in arms})
        return report_outcomes(outcomes)
    with collection_lock():
        ledger = SpendLedger(JOURNAL, cap_usd=CAP_USD, audit=AUDIT)
        check_published_history(ledger)
        left = headroom_usd(ledger)
        if left < ATTEMPT_RESERVATION_USD:
            raise Refused(f"the ${CAP_USD:.2f} S5.2 cap has ${left:.4f} left; no run started")
        print(f"S5.2 ledger: ${left:.4f} of ${CAP_USD:.2f} left before this pass.")
        factory = AdapterFactory(ledger)
        plans = {
            arm.name: gate_plan(plan_of(arm), ledger, lambda: AccountedAgent(factory))
            for arm in arms
        }
        try:
            outcomes = anyio.run(run_arms, plans)
        finally:
            print(f"Paid audit: {ledger.publish()}")
        state = ledger.snapshot()
        print(
            f"S5.2 ledger: measured ${state.measured_usd:.4f}, unknown reserved "
            f"${state.reserved_usd:.4f}, cap ${CAP_USD:.2f}."
        )
    return report_outcomes(outcomes)


def report_outcomes(outcomes: Mapping[str, EvalOutcome]) -> int:
    """Print what each arm recorded; 0 only if every run was graded and recorded."""
    clean = True
    for name, outcome in outcomes.items():
        graded = sum(record.finished for record in outcome.recorded)
        errors = [
            f"{r.case_id}" + (f"#{r.attempt}" if r.kind is RunKind.REPEAT else "")
            for r in outcome.recorded
            if not r.finished
        ]
        print(
            f"[{name}] recorded {len(outcome.recorded)} ({graded} graded), skipped "
            f"{len(outcome.skipped)} finished, unrecorded {len(outcome.unrecorded)}; "
            f"results in {outcome.directory}"
        )
        if errors:
            print(f"[{name}] harness errors (not measured; retry with --resume): {errors}")
        clean = clean and not errors and not outcome.unrecorded
    return 0 if clean else 1


# --- Orphaned receipts (a collector killed mid-run) ----------------------------------------------


def reconcile(arms: Sequence[Arm], commit: str, confirm: bool, out: Callable[[str], None]) -> int:
    """Account for pending receipts a killed collector left, so --resume can continue.

    The runner refuses to start while a receipt has no record, since that run may have spent. In
    S5.2 every paid attempt of that run is already in the ledger (reserved before dispatch; an
    unsettled one stays reserved against the cap forever), so spend is safe. This keeps the run's
    event-log usage and its ledger exposure under `results/final/s5.2-orphans/`, which the summary
    adds to total spend, and removes the receipt; its slot is then simply `missing` and reruns.
    """
    ledger = SpendLedger(JOURNAL, cap_usd=CAP_USD, audit=AUDIT)
    found = 0
    for config in sorted({arm.config for arm in arms}):
        store = ResultsStore(RESULTS_ROOT, commit, config)
        folder = store.directory / "pending"
        if not folder.is_dir():
            continue
        with store.exclusive():
            records = (*store.attempt_records(), *store.repeat_records())
            by_id = {r.run_id: r for r in (*records, *store.primary_records().values())}
            for path in sorted(folder.glob("*.json")):
                receipt = json.loads(path.read_text())
                run_id = receipt["run_id"]
                if run_id in by_id:
                    out(f"{config}: {run_id} already has its record; receipt cleared")
                    if confirm:
                        store.finish_pending(run_id)
                    continue
                found += 1
                attempts = [a for a in ledger.snapshot().attempts if a["run_id"] == run_id]
                usage = collect_usage(WORK_ROOT / "state" / f"{run_id}.sqlite", run_id).usage
                orphan = {
                    "receipt": receipt,
                    "config": config,
                    "ledger_measured_usd": sum(a["actual_usd"] or 0.0 for a in attempts),
                    "ledger_unknown_reserved_usd": sum(
                        a["maximum_usd"] for a in attempts if a["actual_usd"] is None
                    ),
                    "ledger_attempts": len(attempts),
                    "event_log_usage": usage.model_dump(mode="json"),
                }
                out(
                    f"{config}: orphan {receipt['system']}/{receipt['case_id']} {run_id}: "
                    f"ledger ${orphan['ledger_measured_usd']:.4f} measured, "
                    f"${orphan['ledger_unknown_reserved_usd']:.2f} unknown"
                )
                if confirm:
                    ORPHANS.mkdir(parents=True, exist_ok=True)
                    (ORPHANS / f"{run_id}.json").write_text(json.dumps(orphan, indent=2) + "\n")
                    store.finish_pending(run_id)
    if found and not confirm:
        out("Nothing changed. Rerun with --confirm to keep these as orphans and clear receipts.")
    return 0


# --- Dry run -------------------------------------------------------------------------------------


def data_present(case: Case, snapshots_dir: Path = SNAPSHOTS_DIR) -> bool:
    """The case's snapshot (and planted file) are on this machine with the pinned hash."""
    try:
        entry = load_manifest(snapshots_dir).get(case.dataset.snapshot)
    except (OSError, ValueError):
        return False
    if entry is None or entry.sha256 != case.dataset.sha256:
        return False
    if not (snapshots_dir / entry.file).is_file():
        return False
    planted = case.dataset.planted
    return planted is None or (ROOT / "data" / "planted" / planted.file).is_file()


@dataclass(frozen=True)
class PassPlan:
    """One arm's pass against what is on disk."""

    arm: Arm
    which: Pass
    slots: int
    graded: int
    retry: int
    missing: int
    budget_usd: float

    @property
    def pending(self) -> int:
        """Runs this pass would still start."""
        return self.retry + self.missing


def pass_plan(arm: Arm, which: Pass, cases: Sequence[Case], store: ResultsStore) -> PassPlan:
    """Count a pass's slots by state; `budget_usd` is the pending runs' full case budgets."""
    budgets = {case.id: case.budget.max_usd or 0.0 for case in cases}
    states = [
        (slot, slot_state(store, arm, slot))
        for slot in planned_slots(arm, which, [case.id for case in cases])
    ]
    count = Counter(state for _, state in states)
    full = sum(budgets[s.case_id] for s, state in states if state != "graded") if arm.paid else 0.0
    return PassPlan(
        arm, which, len(states), count["graded"], count["harness_error"], count["missing"], full
    )


def dry_run(cases_selector: str, parallel: int, tag: str, out: Callable[[str], None]) -> int:
    """Print the matrix, its cost and time, and the launch order; no Docker, key or provider."""
    identity = read_identity(tag)
    commit = identity.tag_commit or identity.head
    out(f"HEAD {identity.head[:12]} dirty={identity.dirty}; tag {tag}: {identity.tag_commit}")
    out("  freeze identity OK" if identity.frozen else "  NOT the freeze: `collect` would refuse")
    try:
        check_lock()
        out("Readiness lock matches the code.")
    except Refused as exc:
        out(f"Readiness lock: {exc}")
    cases = load_cases(cases_selector)
    try:
        check_case_set(cases)
        out(f"Case set {cases_selector}: {len(cases)} cases, repeat sample present.")
    except Refused as exc:
        out(f"Case set {cases_selector}: {exc}")
    absent = sorted(case.id for case in cases if not data_present(case))
    out(
        f"Case data on this machine: {len(cases) - len(absent)}/{len(cases)}"
        + (f"; missing: {absent}" if absent else "")
    )
    for config in sorted({arm.config for arm in ARMS.values()}):
        info = config_identity(config)
        settings = load_settings_for_eval(config, profile=ModelProfile.EVAL)
        run_config = RunConfig.build(settings, eval_config=config)
        limit = parallel_limit(run_config, load_eval_config(config).max_parallel_cases, parallel)
        models = sorted({f"{t['provider']}/{t['model_id']}" for t in info["tiers"].values()})
        out(
            f"Config {config}: mode {info['agent_mode']}, models {models}, fallback "
            f"{info['fallback_enabled']}, run_config {info['run_config_hash'][:12]}, "
            f"parallel limit here {limit} (asked {parallel})"
        )
    out(f"Records commit: {commit[:12]}")
    plans = [
        pass_plan(arm, which, cases, store_for(arm, commit))
        for which in Pass
        for arm in ARMS.values()
    ]
    out("")
    out("| Arm | Pass | Runs | Graded | Retry | Missing | Paid | Full budgets |")
    out("|---|---|---:|---:|---:|---:|---|---:|")
    for p in plans:
        out(
            f"| {p.arm.name} | {p.which.value} | {p.slots} | {p.graded} | {p.retry} | "
            f"{p.missing} | {'yes' if p.arm.paid else 'no'} | ${p.budget_usd:.2f} |"
        )
    paid_pending = sum(p.pending for p in plans if p.arm.paid)
    free_pending = sum(p.pending for p in plans if not p.arm.paid)
    total_slots = sum(p.slots for p in plans if p.arm.paid)
    full = sum(p.budget_usd for p in plans)
    out("")
    out(f"Paid runs: {total_slots} in the matrix, {paid_pending} still to run.")
    out(f"Baseline runs (no model calls): {free_pending} still to run.")
    out(
        f"Expected paid spend: {paid_pending} x ${EXPECTED_RUN_USD:.3f} = "
        f"${paid_pending * EXPECTED_RUN_USD:.2f}"
    )
    out(
        f"Worst case at full case budgets: ${full:.2f}; hard cap ${CAP_USD:.2f} (ledger, "
        f"${ATTEMPT_RESERVATION_USD:.2f} reserved per attempt): spend can never pass the cap."
    )
    if JOURNAL.exists():
        ledger = SpendLedger(JOURNAL, cap_usd=CAP_USD, audit=AUDIT)
        state = ledger.snapshot()
        out(
            f"Ledger so far: measured ${state.measured_usd:.4f}, unknown reserved "
            f"${state.reserved_usd:.4f}, headroom ${headroom_usd(ledger):.4f}"
        )
    else:
        out(f"Ledger: none yet ({JOURNAL}); full ${CAP_USD:.2f} headroom.")
    low = paid_pending * RUN_MINUTES[0] / max(1, parallel) / 60
    high = paid_pending * RUN_MINUTES[1] / max(1, parallel) / 60
    base_low = free_pending * BASELINE_MINUTES[0] / 60
    base_high = free_pending * BASELINE_MINUTES[1] / 60
    out(
        f"Paid wall time at {parallel} at once: {low:.1f}-{high:.1f} h "
        f"({RUN_MINUTES[0]}-{RUN_MINUTES[1]} min per run); baseline about "
        f"{base_low:.1f}-{base_high:.1f} h at 1 at once."
    )
    return 0


# --- Summary -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRecords:
    """One arm's records on one commit."""

    primary: dict[str, CaseRecord]
    repeats: dict[str, list[CaseRecord]]
    attempts: list[CaseRecord]


def load_arm(arm: Arm, commit: str, results_root: Path = RESULTS_ROOT) -> ArmRecords:
    """Primary, repeat and archived-attempt records of one arm (its system in its segment)."""
    store = store_for(arm, commit, results_root)
    primary = {
        case: record
        for (system, case), record in store.primary_records().items()
        if system == arm.system
    }
    repeats: dict[str, list[CaseRecord]] = {}
    for record in store.repeat_records():
        if record.system == arm.system:
            repeats.setdefault(record.case_id, []).append(record)
    for runs in repeats.values():
        runs.sort(key=lambda r: r.attempt)
    attempts = [r for r in store.attempt_records() if r.system == arm.system]
    return ArmRecords(primary, repeats, attempts)


def _outcome(record: CaseRecord | None) -> str:
    if record is None:
        return "missing"
    if not record.finished:
        return "harness_error"
    return "pass" if record.passed else "fail"


def rate(records: Sequence[CaseRecord]) -> dict[str, Any]:
    """Pass rate over graded runs with its 95% Wilson interval; harness errors are excluded."""
    graded = [r for r in records if r.finished]
    passed = sum(r.passed for r in graded)
    interval = wilson_interval(passed, len(graded)) if graded else None
    return {
        "graded": len(graded),
        "passed": passed,
        "rate": passed / len(graded) if graded else None,
        "wilson_low": interval.low if interval else None,
        "wilson_high": interval.high if interval else None,
        "harness_errors": len(records) - len(graded),
    }


def arm_summary(arm: Arm, records: ArmRecords, case_ids: Sequence[str]) -> dict[str, Any]:
    """Pass rate, latency and cost of one arm's primary runs."""
    primary = [records.primary[c] for c in sorted(case_ids) if c in records.primary]
    graded = [r for r in primary if r.finished]
    every = [*primary, *(r for runs in records.repeats.values() for r in runs), *records.attempts]
    prices = sorted(
        {
            f"{m.provider.value}/{m.model_id} price retrieved {m.price_retrieved_on}"
            for r in every
            for m in r.usage.by_model
        }
    )
    summary: dict[str, Any] = {
        "arm": arm.name,
        "config": arm.config,
        "system": arm.system,
        "pass_rate": rate(primary),
        "harness_error_cases": sorted(r.case_id for r in primary if not r.finished),
        "missing_cases": sorted(set(case_ids) - set(records.primary)),
        "prices": prices,
        "spend_all_runs": {
            "actual_usd": sum(r.usage.actual_spend_usd for r in every),
            "list_usd": sum(r.usage.cost_usd for r in every),
            "unobserved_reserved_usd": sum(r.usage.unobserved_spend_reserved_usd for r in every),
            "usage_complete": all(r.usage.complete for r in every),
            "runs": len(every),
        },
    }
    if graded:
        system = latency_summary([r.timings.system_seconds for r in graded])
        active = latency_summary([r.timings.active_seconds for r in graded])
        actual = cost_per_task([r.usage.actual_spend_usd for r in graded])
        listed = cost_per_task([r.usage.cost_usd for r in graded])
        summary |= {
            "latency_seconds": {
                "mean": system.mean,
                "p95": system.p95,
                "active_mean": active.mean,
                "active_p95": active.p95,
                "method": LATENCY_METHOD,
            },
            "cost_per_graded_task": {
                "actual_usd": actual.per_task_usd,
                "list_usd": listed.per_task_usd,
                "tasks": actual.tasks,
            },
        }
    return summary


def paired(
    first: Mapping[str, CaseRecord], second: Mapping[str, CaseRecord], case_ids: Sequence[str]
) -> dict[str, Any]:
    """Paired outcome table and bootstrap over cases graded for both; the rest listed apart."""
    shared = [
        c
        for c in sorted(case_ids)
        if c in first and c in second and first[c].finished and second[c].finished
    ]
    excluded = sorted(set(case_ids) - set(shared))
    result: dict[str, Any] = {"cases": len(shared), "excluded_cases": excluded}
    if not shared:
        return result
    a = [first[c].passed for c in shared]
    b = [second[c].passed for c in shared]
    table = paired_outcome_table(a, b)
    boot = paired_bootstrap_difference(
        a, b, resamples=DEFAULT_RESAMPLES, seed=DEFAULT_BOOTSTRAP_SEED
    )
    return result | {
        "both_pass": table.both_pass,
        "first_only": table.first_only,
        "second_only": table.second_only,
        "both_fail": table.both_fail,
        "difference": boot.difference,
        "ci_low": boot.interval.low,
        "ci_high": boot.interval.high,
        "resamples": boot.resamples,
        "seed": boot.seed,
        "verdict": boot.verdict.value,
    }


def hidden_metric(record: CaseRecord | None) -> str | None:
    """`name=value` of the hidden-split score, when graded with one."""
    if record is None or record.grade is None or record.grade.hidden_metric is None:
        return None
    metric = record.grade.hidden_metric
    return f"{metric.name}={metric.value:.4g}"


def read_leakage_checks(database: Path, run_id: str) -> list[LeakageCheckPayload] | None:
    """The run's too-good-score investigation steps, or None when its state database is gone."""
    if not database.is_file():
        return None
    db = Database(database)
    try:
        events = EventLog(db).read(run_id, EventFilters(types=(EventType.LEAKAGE_CHECK,)))
    finally:
        db.close()
    return [e.payload for e in events if isinstance(e.payload, LeakageCheckPayload)]


def leakage_view(steps: Sequence[LeakageCheckPayload]) -> dict[str, Any]:
    """Did the too-good-score check fire, which columns did it name, and what did it recommend."""
    fired = any(s.step is LeakageStep.TRIGGER and s.outcome is LeakageOutcome.FIRED for s in steps)
    screened = sorted({c.column for s in steps if s.step is LeakageStep.SCREEN for c in s.columns})
    confirmed = sorted(
        {
            c.column
            for s in steps
            if s.step is LeakageStep.CONFIRMATION
            for c in s.columns
            if c.confirmed
        }
    )
    decisions = [s for s in steps if s.step is LeakageStep.DECISION]
    decision = decisions[-1] if decisions else None
    exclusions = sorted(
        {s.outcome.value for s in steps if s.outcome in EXCLUSION_OUTCOMES}
        | {
            c.comparison.value
            for s in steps
            for c in s.columns
            if c.comparison is not None and c.comparison in EXCLUSION_OUTCOMES
        }
        | {
            s.section.exclusion.value
            for s in decisions
            if s.section is not None and s.section.exclusion is not None
        }
    )
    named = (
        [c.column for c in decision.section.columns]
        if decision is not None and decision.section is not None
        else []
    )
    return {
        "fired": fired,
        "screened": screened,
        "confirmed": confirmed,
        "named": named,
        "decision": decision.outcome.value if decision is not None else None,
        "exclusion": exclusions,
    }


def too_good_report(arms: Mapping[str, ArmRecords], work_root: Path = WORK_ROOT) -> dict[str, Any]:
    """S5.1 asks: how often the check fired on non-leakage cases, the column, the recommendation."""
    rows: list[dict[str, Any]] = []
    checked = unavailable = 0
    for name, records in arms.items():
        if not ARMS[name].paid:
            continue
        runs = [*records.primary.values(), *(r for v in records.repeats.values() for r in v)]
        for record in runs:
            if not record.finished or LEAKAGE_TRAP in record.inputs.trap_kinds:
                continue
            steps = read_leakage_checks(
                work_root / "state" / f"{record.run_id}.sqlite", record.run_id
            )
            if steps is None:
                unavailable += 1
                continue
            checked += 1
            view = leakage_view(steps)
            if view["fired"]:
                rows.append(
                    {
                        "arm": name,
                        "case_id": record.case_id,
                        "kind": record.kind.value,
                        "attempt": record.attempt,
                        "passed": record.passed,
                        **view,
                    }
                )
    return {"runs_checked": checked, "state_unavailable": unavailable, "fired": rows}


def repeat_summary(arm: Arm, records: ArmRecords) -> dict[str, Any]:
    """The repeat sample of one arm, apart from its primary runs."""
    cases = []
    every: list[CaseRecord] = []
    for case in sorted(REPEAT_CASES):
        runs = records.repeats.get(case, [])
        every += runs
        by_attempt = {r.attempt: r for r in runs}
        outcomes = [_outcome(by_attempt.get(n)) for n in range(1, REPEATS + 1)]
        graded = [o for o in outcomes if o in {"pass", "fail"}]
        cases.append(
            {
                "case_id": case,
                "primary": _outcome(records.primary.get(case)),
                "repeats": outcomes,
                "passed": graded.count("pass"),
                "graded": len(graded),
                "hidden_metrics": [hidden_metric(by_attempt.get(n)) for n in range(1, REPEATS + 1)],
            }
        )
    return {"arm": arm.name, "rate": rate(every), "cases": cases}


def orphan_spend(folder: Path = ORPHANS) -> dict[str, float]:
    """Spend of runs whose collector was killed before their record (see `reconcile`)."""
    total = {"runs": 0.0, "ledger_measured_usd": 0.0, "ledger_unknown_reserved_usd": 0.0}
    if not folder.is_dir():
        return total
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        total["runs"] += 1
        total["ledger_measured_usd"] += data["ledger_measured_usd"]
        total["ledger_unknown_reserved_usd"] += data["ledger_unknown_reserved_usd"]
    return total


def build_final_summary(
    commit: str,
    case_ids: Sequence[str],
    *,
    results_root: Path = RESULTS_ROOT,
    work_root: Path = WORK_ROOT,
    audit: Path = AUDIT,
    orphans: Path = ORPHANS,
) -> dict[str, Any]:
    """Every S5.2 number, computed from the records on disk (and the ledger audit)."""
    arms = {name: load_arm(arm, commit, results_root) for name, arm in ARMS.items()}
    identities: dict[str, dict[str, list[str]]] = {}
    for name, records in arms.items():
        every = [
            *records.primary.values(),
            *(r for v in records.repeats.values() for r in v),
            *records.attempts,
        ]
        identities[name] = {
            "commits": sorted({r.provenance.commit_sha for r in every}),
            "run_config_hashes": sorted({r.provenance.run_config_hash for r in every}),
            "agent_modes": sorted({r.provenance.flags.agent_mode.value for r in every}),
            "models": sorted(
                {
                    f"{t.provider.value}/{t.model_id}"
                    for r in every
                    for t in (r.provenance.tiers.for_tier(tier) for tier in ModelTier)
                }
            ),
            "sandbox_images": sorted({r.provenance.sandbox_image for r in every}),
            "tools_contracts": sorted({r.provenance.tools_contract_sha256 for r in every}),
        }
    problems = [
        f"{name}: records from {len(v['commits'])} commits or {len(v['run_config_hashes'])} configs"
        for name, v in identities.items()
        if len(v["commits"]) > 1 or len(v["run_config_hashes"]) > 1
    ]
    failures: dict[str, dict[str, int]] = {}
    for name, records in arms.items():
        counter: Counter[str] = Counter()
        for record in records.primary.values():
            if record.finished and not record.passed:
                counter.update(record.failure_categories)
        failures[name] = dict(sorted(counter.items()))
    per_case = []
    for case in sorted(case_ids):
        row: dict[str, Any] = {"case_id": case}
        for name, records in arms.items():
            found = records.primary.get(case)
            row[name] = {
                "outcome": _outcome(found),
                "hidden_metric": hidden_metric(found),
                "failed_checks": list(found.failure_categories)
                if found is not None and found.finished and not found.passed
                else [],
            }
        per_case.append(row)
    ledger = json.loads(audit.read_text()) if audit.exists() else None
    return {
        "step": "S5.2",
        "commit": commit,
        "cases": len(case_ids),
        "identity": identities,
        "identity_problems": problems,
        "systems": {name: arm_summary(ARMS[name], arms[name], case_ids) for name in ARMS},
        "paired": {
            f"{a}_vs_{b}": {"first": a, "second": b}
            | paired(arms[a].primary, arms[b].primary, case_ids)
            for a, b in PAIRS
        },
        "failure_categories": failures,
        "per_case": per_case,
        "repeats": {name: repeat_summary(ARMS[name], arms[name]) for name in ARMS},
        "too_good_score_on_non_leakage_cases": too_good_report(arms, work_root),
        "ledger": None
        if ledger is None
        else {
            "cap_usd": ledger["cap_usd"],
            "measured_usd": ledger["measured_usd"],
            "unknown_reserved_usd": ledger["unknown_reserved_usd"],
            "attempts": len(ledger["attempts"]),
        },
        "orphans": orphan_spend(orphans),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_final_markdown(summary: Mapping[str, Any]) -> str:
    """The human-readable S5.2 tables (the JSON holds every number)."""
    lines = [
        "# S5.2 final run matrix",
        "",
        f"Commit `{summary['commit']}`, {summary['cases']} held-out cases. Primary runs only in "
        "the rates and paired tables; harness errors (not measured) are excluded from both and "
        "listed. Repeat sample reported separately.",
        "",
    ]
    if summary["identity_problems"]:
        lines += ["**Identity problems:** " + "; ".join(summary["identity_problems"]), ""]
    lines += [
        "## Pass rates (95% Wilson, graded runs)",
        "",
        "| System | Passed / graded | Rate | Wilson 95% | Harness errors | Missing |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, s in summary["systems"].items():
        r = s["pass_rate"]
        interval = (
            "n/a"
            if r["wilson_low"] is None
            else f"{_pct(r['wilson_low'])} - {_pct(r['wilson_high'])}"
        )
        errors = ", ".join(s["harness_error_cases"]) or "-"
        missing = ", ".join(s["missing_cases"]) or "-"
        lines.append(
            f"| {name} | {r['passed']} / {r['graded']} | {_pct(r['rate'])} | {interval} | "
            f"{errors} | {missing} |"
        )
    lines += [
        "",
        f"## Paired comparisons (bootstrap {DEFAULT_RESAMPLES} resamples, seed "
        f"{DEFAULT_BOOTSTRAP_SEED})",
        "",
        "| First vs second | Cases | Both pass | First only | Second only | Both fail | "
        "Difference | 95% CI | Verdict | Excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for p in summary["paired"].values():
        if not p["cases"]:
            lines.append(f"| {p['first']} vs {p['second']} | 0 | | | | | | | | all |")
            continue
        lines.append(
            f"| {p['first']} vs {p['second']} | {p['cases']} | {p['both_pass']} | "
            f"{p['first_only']} | {p['second_only']} | {p['both_fail']} | "
            f"{p['difference']:+.3f} | [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | "
            f"{p['verdict']} | {', '.join(p['excluded_cases']) or '-'} |"
        )
    lines += [
        "",
        "## Cost and latency (graded primary runs)",
        "",
        "| System | Mean s | p95 s (nearest rank) | Actual $/task | List $/task | "
        "All-run actual $ | Unknown reserved $ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summary["systems"].items():
        lat = s.get("latency_seconds")
        cost = s.get("cost_per_graded_task")
        spend = s["spend_all_runs"]
        lines.append(
            f"| {name} | {lat['mean']:.0f} | {lat['p95']:.0f} | "
            if lat
            else f"| {name} | n/a | n/a | "
        )
        lines[-1] += (
            f"{cost['actual_usd']:.4f} | {cost['list_usd']:.4f} | " if cost else "n/a | n/a | "
        ) + f"{spend['actual_usd']:.4f} | {spend['unobserved_reserved_usd']:.2f} |"
    prices = sorted({p for s in summary["systems"].values() for p in s["prices"]})
    if prices:
        lines += ["", "Prices: " + "; ".join(prices) + "."]
    ledger = summary["ledger"]
    if ledger:
        lines += [
            "",
            f"Paid ledger: measured ${ledger['measured_usd']:.4f}, unknown reserved "
            f"${ledger['unknown_reserved_usd']:.2f}, {ledger['attempts']} attempts, cap "
            f"${ledger['cap_usd']:.2f}.",
        ]
    if summary["orphans"]["runs"]:
        o = summary["orphans"]
        lines += [
            "",
            f"Orphaned runs (collector killed before their record): {int(o['runs'])}, ledger "
            f"${o['ledger_measured_usd']:.4f} measured, ${o['ledger_unknown_reserved_usd']:.2f} "
            "unknown.",
        ]
    lines += ["", "## Failure categories (graded primary failures)", ""]
    for name, counts in summary["failure_categories"].items():
        text = ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"
        lines.append(f"- {name}: {text}")
    lines += [
        "",
        "## Per case (primary)",
        "",
        "| Case | " + " | ".join(ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    for row in summary["per_case"]:
        cells = []
        for name in ARMS:
            cell = row[name]
            text = cell["outcome"]
            if cell["hidden_metric"]:
                text += f" ({cell['hidden_metric']})"
            if cell["failed_checks"]:
                text += ": " + ", ".join(cell["failed_checks"])
            cells.append(text)
        lines.append(f"| {row['case_id']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Repeat sample ({len(REPEAT_CASES)} cases x {REPEATS} additional runs)",
        "",
        "| System | Case | Primary | Repeats | Passed / graded |",
        "|---|---|---|---|---:|",
    ]
    for name, rep in summary["repeats"].items():
        for case in rep["cases"]:
            lines.append(
                f"| {name} | {case['case_id']} | {case['primary']} | "
                f"{', '.join(case['repeats'])} | {case['passed']} / {case['graded']} |"
            )
    for name, rep in summary["repeats"].items():
        r = rep["rate"]
        if r["graded"]:
            lines.append(
                f"\n{name} repeat pass rate: {r['passed']} / {r['graded']} ({_pct(r['rate'])}, "
                f"Wilson {_pct(r['wilson_low'])} - {_pct(r['wilson_high'])}); "
                f"{r['harness_errors']} harness errors."
            )
    tg = summary["too_good_score_on_non_leakage_cases"]
    lines += [
        "",
        "## Too-good-score check on non-leakage cases (S5.1 requirement)",
        "",
        f"Graded agent runs checked: {tg['runs_checked']}; fired: {len(tg['fired'])}; state "
        f"database unavailable: {tg['state_unavailable']}.",
    ]
    for row in tg["fired"]:
        named = row["named"] or row["confirmed"] or row["screened"]
        verdict = "pass" if row["passed"] else "fail"
        lines.append(
            f"- {row['arm']} {row['case_id']} ({row['kind']} {row['attempt']}, {verdict}): "
            f"named {named}, decision {row['decision']}, exclusion {row['exclusion'] or 'none'}"
        )
    return "\n".join(lines) + "\n"


def summarize(commit: str | None, tag: str, cases_selector: str) -> int:
    """Write `results/final/s5.2-summary.{json,md}` from the records on the freeze commit."""
    identity = read_identity(tag)
    chosen = commit or identity.tag_commit or identity.head
    cases = load_cases(cases_selector)
    summary = build_final_summary(chosen, [case.id for case in cases])
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    SUMMARY_MD.write_text(render_final_markdown(summary))
    print(f"Wrote {SUMMARY_JSON} and {SUMMARY_MD}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """`plan` (dry run), `collect`, `reconcile`, `spend`, `summarize`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=FREEZE_TAG)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="dry run: matrix, cost, time; no provider/Docker")
    plan.add_argument("--cases", default=CASE_SET, help="test, or a case directory")
    plan.add_argument("--parallel", type=int, default=2)
    run = commands.add_parser("collect")
    run.add_argument("--arm", choices=("multi", "single", "both", "baseline"), required=True)
    run.add_argument("--pass", dest="which", choices=tuple(p.value for p in Pass), required=True)
    run.add_argument("--parallel", type=int, default=2)
    run.add_argument("--resume", action="store_true")
    fix = commands.add_parser("reconcile", help="account for receipts a killed collector left")
    fix.add_argument("--confirm", action="store_true")
    commands.add_parser("spend", help="republish the paid audit")
    summary = commands.add_parser("summarize")
    summary.add_argument("--commit")
    summary.add_argument("--cases", default=CASE_SET)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            return dry_run(args.cases, args.parallel, args.tag, print)
        if args.command == "summarize":
            return summarize(args.commit, args.tag, args.cases)
        if args.command == "spend":
            print(SpendLedger(JOURNAL, cap_usd=CAP_USD, audit=AUDIT).publish())
            return 0
        if args.command == "reconcile":
            identity = read_identity(args.tag)
            check_identity(identity)
            with collection_lock():
                return reconcile(tuple(ARMS.values()), identity.head, args.confirm, print)
        if args.parallel < 1:
            raise Refused("--parallel must be at least 1")
        return collect(
            args.arm, Pass(args.which), parallel=args.parallel, resume=args.resume, tag=args.tag
        )
    except (Refused, SpendError, EvalRefused, PreflightError) as exc:
        print(f"Collection refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
