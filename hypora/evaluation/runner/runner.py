"""The single comparison runner (spec §19.4-19.6; PLAN.md S2.4).

For a case set, a config, and the systems to compare: refuse what must not be recorded, create each
case's hidden split, run every system on every case in parallel up to the configured limit, grade
each run, store one record per run, and write the summary and the regression comparison.

Systems see only `SystemInput`, built field by field in `system_input_for`: trap metadata, grading
rules, and hidden-split settings never reach it. The dataset is always `split.system_dataset()`; the
case's snapshot still holds the hidden rows. docs/notes/eval-runner.md explains the choices.
"""

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import anyio
from pydantic import TypeAdapter, ValidationError

from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import new_run_id
from hypora.core.hashing import sha256_hex
from hypora.core.llm.credentials import load_credentials
from hypora.core.schemas import ApprovalPolicy, Identifier
from hypora.evaluation.case import Case
from hypora.evaluation.case_data import PLANTED_DIR, load_case_dataset
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.hidden_split import (
    HiddenLeakError,
    HiddenSplit,
    HiddenStore,
    create_hidden_split,
)
from hypora.evaluation.runner.guard import (
    RecordGuard,
    RecordLeakError,
    hidden_only_values,
    machine_aliases,
)
from hypora.evaluation.runner.records import (
    HARNESS_ERROR_CATEGORY,
    CaseInputs,
    CaseOutputs,
    CaseRecord,
    FinalModelOutput,
    PendingRun,
    Provenance,
    RecordStatus,
    RunKind,
    SplitSummary,
    Timings,
    Usage,
)
from hypora.evaluation.runner.regression import (
    RegressionReport,
    compare_with_parent,
    regression_lines,
)
from hypora.evaluation.runner.report import build_summary, render_markdown
from hypora.evaluation.runner.store import (
    REGRESSIONS_JSON,
    RESULTS_ROOT,
    SUMMARY_JSON,
    SUMMARY_MD,
    CodeState,
    EvalRefused,
    GitRepo,
    ResultsStore,
)
from hypora.evaluation.runner.usage import collect_usage
from hypora.evaluation.system import (
    CaseGrade,
    CheckStatus,
    EvalSystem,
    Grader,
    SystemContext,
    SystemInput,
    SystemRun,
)
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.image import tabular_image_tag
from hypora.storage.database import Database
from hypora.storage.migrate import migrate
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR

MIN_REPEAT_CASES: Final = 5
MIN_REPEATS: Final = 3
"""Spec §19.5: the repeat sample is at least 5 cases x 3 runs."""
MAX_REPORT_CHARS: Final = 256_000
TOOLS_CONTRACT: Final = REPO_ROOT / "docs" / "mcp" / "tools.json"
RUNS_DIR: Final = "runs"
STATE_DIR: Final = "state"


@runtime_checkable
class ReportsRetryWaits(Protocol):
    """A system that can say how long a run slept in retry backoff (optional)."""

    def retry_wait_seconds(self, run_id: str) -> float | None:
        """Seconds of retry backoff in a finished run, or None if unknown."""
        ...


@dataclass(frozen=True)
class EvalPlan:
    """Everything one eval was asked for, resolved."""

    cases: tuple[Case, ...]
    systems: tuple[EvalSystem, ...]
    grader: Grader
    config: str
    segment: str
    run_config: RunConfig
    store: HiddenStore
    work_root: Path
    """State for the runs: `runs/` and one state database per run. Git-ignored."""
    max_parallel_cases: int = 4
    requested_parallel: int | None = None
    results_root: Path = RESULTS_ROOT
    snapshots_dir: Path = SNAPSHOTS_DIR
    planted_dir: Path = PLANTED_DIR
    repo: GitRepo = field(default_factory=GitRepo)
    resume: bool = False
    allow_dirty: bool = False
    repeat_cases: tuple[str, ...] = ()
    repeats: int = 0
    model_systems: frozenset[str] = frozenset({"agent"})
    """Systems that spend on model calls, for the worst-case spending check."""
    max_total_usd: float | None = None
    run_ids: Callable[[], str] = new_run_id


@dataclass(frozen=True)
class EvalOutcome:
    """What an eval did."""

    directory: Path
    recorded: tuple[CaseRecord, ...]
    skipped: tuple[str, ...]
    unrecorded: tuple[str, ...]
    regression: RegressionReport

    @property
    def clean(self) -> bool:
        """Every run was recorded and graded, and nothing newly fails."""
        return (
            not self.unrecorded
            and all(record.finished for record in self.recorded)
            and not self.regression.newly_failing
        )


@dataclass(frozen=True)
class _Prepared:
    """A case ready to run, or why it isn't. `split` holds references, not rows."""

    case: Case
    split: HiddenSplit | None
    hidden_values: frozenset[str]
    setup_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class _Task:
    prepared: _Prepared
    system: EvalSystem
    kind: RunKind
    attempt: int


def parallel_limit(run_config: RunConfig, configured: int, requested: int | None = None) -> int:
    """How many runs may go at once: the config's maximum, lowered to what the pools can hold.

    Each run has one experiment worker running one job at a time, within `budgets.job_limits`. Model
    calls don't need a slot here: `ProviderAdmission` limits them across all runs.
    """
    job = run_config.budgets.job_limits
    pools = run_config.pools
    by_cpu = pools.cpu_slots // (job.cpu_threads or 1)
    by_memory = int(pools.memory_budget_gb // job.max_memory_gb)
    limit = max(1, min(configured, by_cpu, by_memory))
    return limit if requested is None else max(1, min(limit, requested))


def system_input_for(case: Case, split: HiddenSplit) -> SystemInput:
    """All a system may know about a case. Fields are named one by one, so a new `Case` field can't
    reach a system by default; traps, grading, and the hidden-split settings are left out."""
    return SystemInput(
        case_id=case.id,
        request=case.request,
        constraints=case.constraints,
        family=case.family,
        dataset=split.system_dataset(),
        permitted_tools=case.permitted_tools,
        budget=case.budget,
        contract_bounds=case.contract_bounds,
    )


def audit_run_files(split: HiddenSplit, store: HiddenStore, runs_root: Path, run_id: str) -> None:
    """Check what a run's jobs really read: the visible bytes, from outside the hidden store.

    The runner builds no sandbox job itself, so it has no `SandboxMounts` to pass to `check_mounts`.
    Every job's mounts are made under its job folder and the repository, and `run_eval` has already
    refused a runs folder that overlaps the store; this confirms it from the stored job requests.
    """
    jobs = runs_root / run_id / "jobs"
    if not jobs.is_dir():
        return
    for folder in sorted(jobs.iterdir()):
        if not (folder / protocol.REQUEST_FILE).is_file():
            continue
        store.refuse_overlap(folder)
        request = protocol.read_request(folder)
        store.refuse_overlap(Path(request.dataset_path))
        if request.dataset_sha256 != split.visible_sha256:
            raise HiddenLeakError(f"job {folder.name} read a dataset other than the visible file")


def check_grade(grade: CaseGrade, run: SystemRun) -> None:
    """A grade must be about this run, and its verdict must follow from its checks (spec §19.2)."""
    if (grade.case_id, grade.system, grade.run_id) != (run.case_id, run.system, run.run_id):
        raise ValueError("the grader returned a grade for a different run")
    if any(
        check.status is CheckStatus.PASSED and not check.evidence_refs for check in grade.checks
    ):
        raise ValueError("a passing grader check has no evidence reference")
    statuses = {check.status for check in grade.checks}
    earned = CheckStatus.PASSED in statuses and CheckStatus.FAILED not in statuses
    if grade.passed != earned:
        raise ValueError("the grade's verdict does not follow from its checks")


def failure_categories(grade: CaseGrade) -> tuple[str, ...]:
    """The name of each failed check: what the summary's failure table counts."""
    if grade.passed:
        return ()
    failed = tuple(c.name for c in grade.checks if c.status is CheckStatus.FAILED)
    return failed or ("no_check_passed",)


def _relative(path: Path, root: Path) -> str:
    """A path as a record may show it: under the runs folder, or just the file's name."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _report_text(runs_root: Path, run_id: str) -> str | None:
    path = runs_root / run_id / "report.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_REPORT_CHARS]


def outputs_of(run: SystemRun, runs_root: Path) -> CaseOutputs:
    """A run's outputs for its record, with paths made relative."""
    final = run.final_model
    return CaseOutputs(
        run_id=run.run_id,
        completed=run.completed,
        system_wall_clock_seconds=run.wall_clock_seconds,
        decision_class=run.decision_class,
        limits_fired=run.limits_fired,
        notes=run.notes,
        final_model=None
        if final is None
        else FinalModelOutput(
            trial_id=final.trial_id,
            contract=final.contract,
            model_ref=_relative(final.model_path, runs_root),
            model_sha256=final.model_sha256,
            pipeline_ref=_relative(final.pipeline_path, runs_root),
            pipeline_sha256=final.pipeline_sha256,
            reported_metrics=final.reported_metrics,
        ),
        report_markdown=_report_text(runs_root, run.run_id),
    )


def provenance_of(plan: EvalPlan, code: CodeState) -> Provenance:
    """The versions every record of this eval carries (spec §19.5)."""
    config = plan.run_config
    return Provenance(
        commit_sha=code.commit,
        dirty=code.dirty,
        config=plan.config,
        profile=config.model_profile,
        flags=config.flags,
        run_config_hash=config.hash,
        tiers=config.active_profile.tiers,
        retry_policy=config.models.retry,
        python_version=platform.python_version(),
        sandbox_image=tabular_image_tag(),
        tools_contract_sha256=sha256_hex(TOOLS_CONTRACT.read_bytes()),
    )


def inputs_of(case: Case, split: HiddenSplit | None) -> CaseInputs:
    """What a record states about the case: the request, references, and trap kinds only."""
    return CaseInputs(
        case_sha256=case.content_hash(),
        request=case.request,
        constraints=case.constraints,
        family=case.family,
        permitted_tools=case.permitted_tools,
        budget=case.budget,
        contract_bounds=case.contract_bounds,
        dataset=case.dataset,
        split=None if split is None else SplitSummary.model_validate(split.summary()),
        trap_kinds=tuple(trap.kind for trap in case.traps),
        fault_kinds=tuple(fault.kind for fault in case.injected_faults),
        expected_decision_class=case.expected_decision_class,
    )


class _Eval:
    """The state of one running eval."""

    def __init__(self, plan: EvalPlan, code: CodeState, results: ResultsStore) -> None:
        self.plan = plan
        self.code = code
        self.results = results
        self.runs_root = plan.work_root.resolve() / RUNS_DIR
        self.state_root = plan.work_root.resolve() / STATE_DIR
        self.provenance = provenance_of(plan, code)
        self.guard = RecordGuard(
            forbidden_paths=(str(plan.store.root),),
            secrets=load_credentials().secret_values(),
            path_aliases=machine_aliases(self.runs_root, REPO_ROOT),
        )
        self.recorded: list[CaseRecord] = []
        self.unrecorded: list[str] = []
        self.finished_count = 0

    def prepare(self, case: Case) -> _Prepared:
        """Create the case's hidden split, once, before any system runs on it."""
        started = time.perf_counter()
        try:
            dataset = load_case_dataset(
                case, snapshots_dir=self.plan.snapshots_dir, planted_dir=self.plan.planted_dir
            )
            split = create_hidden_split(case, dataset, self.plan.store, self.runs_root)
            values = hidden_only_values(split)
        except Exception as exc:
            error = self.guard.safe_text(f"{type(exc).__name__}: {exc}")
            return _Prepared(case, None, frozenset(), time.perf_counter() - started, error)
        return _Prepared(case, split, values, time.perf_counter() - started)

    def harness_error(
        self,
        task: _Task,
        run_id: str,
        error: str,
        started_at: datetime,
        timings: Timings,
        usage: Usage,
    ) -> CaseRecord:
        """A run with no grade: not a measurement, replaced by a rerun."""
        prepared = task.prepared
        return CaseRecord(
            status=RecordStatus.HARNESS_ERROR,
            run_id=run_id,
            kind=task.kind,
            attempt=task.attempt,
            case_id=prepared.case.id,
            case_split=prepared.case.split,
            system=task.system.name,
            passed=False,
            failure_categories=(HARNESS_ERROR_CATEGORY,),
            provenance=self.provenance,
            inputs=inputs_of(prepared.case, prepared.split),
            timings=timings,
            usage=usage,
            error=error,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    async def run_task(self, task: _Task) -> None:
        """Run, grade, and record one run. A failure here never stops the other runs."""
        started_at = datetime.now(UTC)
        prepared = task.prepared
        system = task.system
        run_id = self.plan.run_ids()
        database_path = self.state_root / f"{run_id}.sqlite"
        setup_seconds = prepared.setup_seconds
        system_seconds = grading_seconds = 0.0
        usage, rate_wait = Usage(), 0.0
        retry_wait: float | None = 0.0
        record: CaseRecord | None = None
        error = prepared.error
        run: SystemRun | None = None
        grade: CaseGrade | None = None
        outputs: CaseOutputs | None = None
        interrupted: BaseException | None = None
        usage_incomplete = False
        if prepared.split is not None:
            try:
                split = prepared.split
                system_input = system_input_for(prepared.case, split)
                self.plan.store.refuse_overlap(database_path, system_input.dataset)
                setup_started = time.perf_counter()
                try:
                    database = Database(database_path)
                    try:
                        migrate(database)
                    finally:
                        database.close()
                finally:
                    setup_seconds += time.perf_counter() - setup_started
                context = SystemContext(
                    run_id=run_id,
                    runs_root=self.runs_root,
                    database=database_path,
                    run_config=self.plan.run_config,
                    injected_faults=prepared.case.injected_faults,
                )
                self.results.begin_pending(
                    PendingRun(
                        run_id=run_id,
                        case_id=prepared.case.id,
                        system=system.name,
                        kind=task.kind,
                        attempt=task.attempt,
                        case_sha256=prepared.case.content_hash(),
                        run_config_hash=self.plan.run_config.hash,
                        budget_usd=(prepared.case.budget.max_usd or 0.0)
                        if system.name in self.plan.model_systems
                        else 0.0,
                    )
                )
                started = time.perf_counter()
                try:
                    run = await system.run(system_input, context)
                finally:
                    system_seconds = time.perf_counter() - started
                if (run.case_id, run.run_id, run.system) != (
                    prepared.case.id,
                    run_id,
                    system.name,
                ):
                    raise ValueError("the system reported a different case, run, or name")
                audit_run_files(split, self.plan.store, self.runs_root, run_id)
                started = time.perf_counter()
                try:
                    grade = await self.plan.grader.grade(prepared.case, split, run)
                finally:
                    grading_seconds = time.perf_counter() - started
                check_grade(grade, run)
                outputs = outputs_of(run, self.runs_root)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            except BaseException as exc:
                # Cancellation/KeyboardInterrupt must not hide already billed attempts. All
                # accounting and publication below are synchronous (no cancellation checkpoints),
                # and the original interruption is re-raised after preserving the partial record.
                interrupted = exc
                usage_incomplete = True
                error = f"{type(exc).__name__}: run interrupted before its record was complete"
            try:
                reported = (
                    system.retry_wait_seconds(run_id)
                    if isinstance(system, ReportsRetryWaits)
                    else None
                )
                collected = collect_usage(database_path, run_id, retry_wait_seconds=reported)
                usage = collected.usage
                rate_wait, retry_wait = (
                    collected.rate_limit_wait_seconds,
                    collected.retry_wait_seconds,
                )
            except Exception as exc:
                # A grade without its cost and timing is not a complete record.
                error = error or f"could not read the run's usage: {type(exc).__name__}: {exc}"
                usage_incomplete = True
        if usage_incomplete:
            ceiling = (
                prepared.case.budget.max_usd or 0.0
                if system.name in self.plan.model_systems
                else 0.0
            )
            usage = usage.model_copy(
                update={
                    "complete": False,
                    "unobserved_spend_reserved_usd": max(0.0, ceiling - usage.actual_spend_usd),
                }
            )
        timings = Timings(
            setup_seconds=setup_seconds,
            system_seconds=system_seconds,
            grading_seconds=grading_seconds,
            rate_limit_wait_seconds=rate_wait,
            retry_wait_seconds=retry_wait,
            active_seconds=max(0.0, system_seconds - rate_wait - (retry_wait or 0.0)),
        )
        if error is None and run is not None and grade is not None:
            record = CaseRecord(
                status=RecordStatus.GRADED,
                run_id=run_id,
                kind=task.kind,
                attempt=task.attempt,
                case_id=prepared.case.id,
                case_split=prepared.case.split,
                system=system.name,
                passed=grade.passed,
                failure_categories=failure_categories(grade),
                provenance=self.provenance,
                inputs=inputs_of(prepared.case, prepared.split),
                outputs=outputs,
                grade=grade,
                timings=timings,
                usage=usage,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        else:
            text = self.guard.safe_text(error or "the run produced no grade")
            record = self.harness_error(task, run_id, text, started_at, timings, usage)
        try:
            self._store(task, record, started_at)
        finally:
            if interrupted is not None:
                raise interrupted

    def _store(self, task: _Task, record: CaseRecord, started_at: datetime) -> None:
        name = f"{record.system}/{record.case_id}"
        if record.kind is RunKind.REPEAT:
            name += f" repeat {record.attempt}"
        values = task.prepared.hidden_values
        try:
            try:
                document = self.guard.clean(record.model_dump(mode="json"), values)
            except RecordLeakError as exc:
                # The refused content goes nowhere, and the run is reported as not measured.
                record = self.harness_error(
                    task,
                    record.run_id,
                    f"Record refused: {exc}.",
                    started_at,
                    record.timings,
                    record.usage,
                )
                document = self.guard.clean(record.model_dump(mode="json"), values)
            self._check_same_code()
            self.results.write(document, self.results.path_for(record))
        except EvalRefused as exc:
            self.unrecorded.append(f"{name}: {exc}")
            return
        self.recorded.append(CaseRecord.model_validate(document))
        self.results.finish_pending(record.run_id)

    def _check_same_code(self) -> None:
        """A record names one commit, so the code must still be that commit when it is written."""
        state = self.plan.repo.state()
        changed = state.commit != self.code.commit or (state.dirty and not self.code.dirty)
        if changed:
            raise EvalRefused(
                f"the code changed during the eval (commit {state.commit[:12]}, "
                f"dirty={state.dirty}); this result does not describe "
                f"commit {self.code.commit[:12]}"
            )


def _check_plan(plan: EvalPlan) -> None:
    if not plan.cases:
        raise EvalRefused("the case set is empty: nothing to evaluate")
    if not plan.systems:
        raise EvalRefused("no systems to evaluate")
    case_ids = [case.id for case in plan.cases]
    if len(set(case_ids)) != len(case_ids):
        raise EvalRefused("case IDs must be unique within one eval")
    names = [system.name for system in plan.systems]
    try:
        for name in (plan.config, plan.segment, *names):
            TypeAdapter(Identifier).validate_python(name)
    except ValidationError:
        raise EvalRefused("config and system names must be safe result-path identifiers") from None
    if len(set(names)) != len(names):
        raise EvalRefused(f"each system may run once per eval: {names}")
    if plan.run_config.flags.approval_policy is not ApprovalPolicy.AUTO_WITHIN_CONTRACT:
        raise EvalRefused(
            f"config {plan.config!r} must set approval_policy auto_within_contract (spec §10.3)"
        )
    plan.store.refuse_overlap(plan.work_root, plan.results_root)
    if plan.repeats or plan.repeat_cases:
        known = {case.id for case in plan.cases}
        unknown = sorted(set(plan.repeat_cases) - known)
        if unknown:
            raise EvalRefused(f"repeat cases are not in this case set: {unknown}")
        if len(set(plan.repeat_cases)) < MIN_REPEAT_CASES or plan.repeats < MIN_REPEATS:
            raise EvalRefused(
                f"the repeat sample is at least {MIN_REPEAT_CASES} cases x {MIN_REPEATS} runs "
                f"(spec §19.5); asked for {len(set(plan.repeat_cases))} x {plan.repeats}"
            )


def _check_cases(plan: EvalPlan) -> None:
    """A case that doesn't validate never runs: its grade would mean nothing."""
    problems = []
    for case in plan.cases:
        report = validate_case(case, snapshots_dir=plan.snapshots_dir, planted_dir=plan.planted_dir)
        problems += [f"{case.id}: {issue.code}: {issue.message}" for issue in report.issues]
    if problems:
        raise EvalRefused("invalid cases:\n  " + "\n  ".join(problems))


def _pending_runs(
    plan: EvalPlan, results: ResultsStore
) -> tuple[list[tuple[Case, EvalSystem, RunKind, int]], list[str]]:
    """Pending and finished runs. Finished records are refused unless `--resume`."""
    pending: list[tuple[Case, EvalSystem, RunKind, int]] = []
    finished: list[str] = []
    wanted: list[tuple[Case, EvalSystem, RunKind, int, Path]] = []
    for case in sorted(plan.cases, key=lambda c: c.id):
        for system in plan.systems:
            path = results.primary_path(system.name, case.id)
            wanted.append((case, system, RunKind.PRIMARY, 1, path))
    for case in sorted(plan.cases, key=lambda c: c.id):
        if case.id in plan.repeat_cases:
            for system in plan.systems:
                for attempt in range(1, plan.repeats + 1):
                    path = results.repeat_path(system.name, case.id, attempt)
                    wanted.append((case, system, RunKind.REPEAT, attempt, path))
    for case, system, kind, attempt, path in wanted:
        existing = results.read(path)
        if existing is not None and existing.finished and not results.dirty:
            if (
                existing.provenance.run_config_hash != plan.run_config.hash
                or existing.inputs.case_sha256 != case.content_hash()
            ):
                raise EvalRefused(
                    f"{path} is a finished record for different config or case contents; "
                    "it cannot be reused by --resume"
                )
            finished.append(str(path.relative_to(results.directory)))
        else:
            if existing is not None:
                # Refuse an unwritable/conflicting recovery archive before a retry can spend.
                # Publication later repeats this idempotently before replacing the old record.
                results.archive(path, existing)
            pending.append((case, system, kind, attempt))
    if finished and not plan.resume:
        raise EvalRefused(
            f"{len(finished)} finished record(s) exist and are not replaced: "
            f"{', '.join(finished[:5])}{' ...' if len(finished) > 5 else ''}. "
            "Use --resume to keep them and run the rest (one primary run per case per system, "
            "spec §19.5), or delete a record yourself to run that case again."
        )
    return pending, finished


def _check_spend(
    plan: EvalPlan, pending: list[tuple[Case, EvalSystem, RunKind, int]], results: ResultsStore
) -> None:
    """Refuse an eval that could pass the cap if every paid run used its whole budget (trigger 6).
    Only the `eval` profile bills anything: `dev` is the free tier, with fallback disabled."""
    if plan.max_total_usd is None or plan.run_config.model_profile.value != "eval":
        return
    worst = sum(
        case.budget.max_usd or 0.0
        for case, system, _, _ in pending
        if system.name in plan.model_systems
    )
    records = (
        *results.attempt_records(),
        *results.repeat_records(),
        *results.primary_records().values(),
    )
    unique = {record.run_id: record for record in records}
    measured = sum(record.usage.actual_spend_usd for record in unique.values())
    reserved = sum(record.usage.unobserved_spend_reserved_usd for record in unique.values())
    spent = measured + reserved
    if spent + worst > plan.max_total_usd:
        raise EvalRefused(
            f"these runs could spend ${worst:.2f} at their full budgets plus "
            f"${measured:.2f} already spent and ${reserved:.2f} reserved for unknown charges, "
            "over the "
            f"${plan.max_total_usd:.2f} cap. Raise --max-total-usd only with the owner's approval."
        )


async def run_eval(plan: EvalPlan, out: Callable[[str], None] = print) -> EvalOutcome:
    """Run the plan to the end and write its reports; raises EvalRefused before running anything."""
    _check_plan(plan)
    code = plan.repo.state()
    if code.dirty and not plan.allow_dirty:
        raise EvalRefused(
            "the working tree has changes outside results/, so these results would not describe "
            f"commit {code.commit[:12]}. Commit them, or use --allow-dirty to write to the "
            "git-ignored results/raw/ (never a recorded result, never a regression base)."
        )
    results = ResultsStore(plan.results_root, code.commit, plan.segment, dirty=code.dirty)
    _check_cases(plan)
    with results.exclusive():
        results.refuse_unreconciled_pending()
        pending, finished = _pending_runs(plan, results)
        _check_spend(plan, pending, results)
        current = _Eval(plan, code, results)
        current.runs_root.mkdir(parents=True, exist_ok=True)
        current.state_root.mkdir(parents=True, exist_ok=True)
        limit = parallel_limit(plan.run_config, plan.max_parallel_cases, plan.requested_parallel)
        out(
            f"Eval {plan.segment} at {code.commit[:12]}: {len(pending)} run(s) to do, "
            f"{len(finished)} already finished, at most {limit} at once."
        )
        prepared = {
            case.id: current.prepare(case)
            for case in plan.cases
            if any(item[0].id == case.id for item in pending)
        }
        limiter = anyio.CapacityLimiter(limit)

        async def run_one(task: _Task) -> None:
            async with limiter:
                await current.run_task(task)
            current.finished_count += 1
            out(
                f"  [{current.finished_count}/{len(pending)}] "
                f"{task.system.name}/{task.prepared.case.id}"
            )

        async with anyio.create_task_group() as group:
            for case, system, kind, attempt in pending:
                group.start_soon(run_one, _Task(prepared[case.id], system, kind, attempt))

        primary = results.primary_records()
        regression = compare_with_parent(results, code, plan.repo, primary)
        if primary:
            summary = build_summary(
                primary, results.repeat_records(), regression, results.attempt_records()
            )
            results.write_derived(
                {
                    REGRESSIONS_JSON: regression.model_dump_json(indent=2) + "\n",
                    SUMMARY_JSON: summary.model_dump_json(indent=2) + "\n",
                    SUMMARY_MD: render_markdown(summary),
                }
            )
    for line in regression_lines(regression):
        out(line)
    for line in current.unrecorded:
        out(f"  NOT RECORDED  {line}")
    return EvalOutcome(
        directory=results.directory,
        recorded=tuple(current.recorded),
        skipped=tuple(finished),
        unrecorded=tuple(current.unrecorded),
        regression=regression,
    )


__all__ = [
    "MIN_REPEATS",
    "MIN_REPEAT_CASES",
    "EvalOutcome",
    "EvalPlan",
    "ReportsRetryWaits",
    "audit_run_files",
    "check_grade",
    "failure_categories",
    "inputs_of",
    "outputs_of",
    "parallel_limit",
    "provenance_of",
    "run_eval",
    "system_input_for",
]
