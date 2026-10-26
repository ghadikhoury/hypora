"""`hypora run`: check everything a run needs, then run it with live progress (PLAN.md S1.6).

Preflight, before any model call: the dataset file, the configuration, the model profile's
provider key (named, never shown), Docker, the chosen experiment worker, and the sandbox image
(built on first use). A failed check is a `PreflightError` with what to fix, and nothing is
created except the state database. Then the controller (S1.5) runs the request while the
notable events of the run are printed as they are recorded.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import anyio
import docker
from docker.errors import DockerException, ImageNotFound

from hypora.cli.trace import describe, format_offset
from hypora.cli.workers import WorkerMode, compose_worker_problem, worker_factory
from hypora.core.config import ConfigError, load_settings
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import RunController, RunRequest, RunResult, new_run_id
from hypora.core.llm import LLMAdapter, MissingCredentialError, ProviderAdmission
from hypora.core.llm.sinks import FileModelIOStore
from hypora.core.schemas import (
    AnyEvent,
    ApprovalPolicy,
    Budget,
    EventType,
    MetricName,
    TaskType,
)
from hypora.core.schemas.events import ToolResultPayload, ToolResultStatus
from hypora.execution.sandbox.image import ensure_tabular_image, tabular_image_tag
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.migrate import migrate

DATASET_SUFFIXES: Final = (".csv", ".parquet")
PROGRESS_TYPES: Final = frozenset(
    {
        EventType.STAGE_TRANSITION,
        EventType.ROLE_STARTED,
        EventType.HANDOFF,
        EventType.TOOL_CALL,
        EventType.JOB_TRANSITION,
        EventType.APPROVAL_GRANTED,
        EventType.ESCALATION,
        EventType.ERROR,
    }
)
"""What `hypora run` prints while it runs; `hypora trace` shows everything."""


class PreflightError(Exception):
    """A run cannot start; the message says what to fix."""


@dataclass(frozen=True)
class RunOptions:
    """Everything `hypora run` was asked for."""

    data: Path
    goal: str
    db: Path
    runs_root: Path
    target: str | None = None
    task_type: TaskType | None = None
    metric: MetricName | None = None
    forbid: tuple[str, ...] = ()
    budget_usd: float | None = None
    max_trials: int | None = None
    max_wall_clock_s: float | None = None
    approve: ApprovalPolicy | None = None
    run_id: str | None = None
    worker: WorkerMode = WorkerMode.HOST
    quiet: bool = False


class AdapterFactory(Protocol):
    def __call__(self, run_config: RunConfig, database: Database, runs_root: Path) -> LLMAdapter:
        """Build the run's LLM adapter (real provider clients in production)."""
        ...


def provider_adapter(run_config: RunConfig, database: Database, runs_root: Path) -> LLMAdapter:
    """The real adapter: the profile's provider clients, keys from the environment or `.env`."""
    return LLMAdapter.from_run_config(
        run_config,
        events=EventLog(database),
        admission=ProviderAdmission(
            run_config.models.rate_limits, run_config.pools.llm_concurrency
        ),
        io_store=FileModelIOStore(runs_root),
    )


def run_config_for(options: RunOptions) -> RunConfig:
    """The validated configuration, with `--approve` applied; raises PreflightError."""
    if options.approve is ApprovalPolicy.INTERACTIVE:
        raise PreflightError(
            "--approve interactive: the interactive plan gate arrives in S3.D2. Stage 1 runs "
            "approve their plan by policy (D-019); use --approve auto_within_contract or omit it."
        )
    overrides = (
        {"flags": {"approval_policy": options.approve.value}} if options.approve is not None else {}
    )
    try:
        return RunConfig.build(load_settings(overrides=overrides))
    except ConfigError as exc:
        raise PreflightError(str(exc)) from None


def budget_for(options: RunOptions, run_config: RunConfig) -> Budget | None:
    """The configured run budget with the `--budget-usd` / `--max-*` overrides; None if none."""
    updates: dict[str, Any] = {}
    if options.budget_usd is not None:
        updates["max_usd"] = options.budget_usd
    if options.max_trials is not None:
        updates["max_trials"] = options.max_trials
    if options.max_wall_clock_s is not None:
        updates["max_wall_clock_seconds"] = options.max_wall_clock_s
    if not updates:
        return None
    return Budget.model_validate({**run_config.budgets.run.model_dump(), **updates})


def check_dataset(path: Path) -> Path:
    """The dataset as an absolute path; raises PreflightError if it is not a CSV/Parquet file."""
    if not path.is_file():
        raise PreflightError(
            f"no dataset at {path}. Fetch a real snapshot first, e.g. "
            "`make snapshot SOURCE=openml ID=46918`, or pass --data <file.csv|file.parquet>."
        )
    if path.suffix.lower() not in DATASET_SUFFIXES:
        raise PreflightError(f"{path}: the dataset must be a .csv or .parquet file")
    return path.resolve()


def check_docker(client_factory: Callable[[], Any] = docker.from_env) -> Any:
    """A Docker client that answers; raises PreflightError when Docker is not running."""
    try:
        client = client_factory()
        client.ping()
    except DockerException as exc:
        raise PreflightError(
            f"Docker is not reachable ({exc}). Start Docker Desktop (Windows: `docker desktop "
            "start`, with WSL integration on for this distro) or the Docker service, then retry."
        ) from None
    return client


def prepare_sandbox_image(client: Any, out: Callable[[str], None]) -> str:
    """The tabular sandbox image tag, built first when missing (about 2 minutes, once)."""
    tag = tabular_image_tag()
    try:
        client.images.get(tag)
    except ImageNotFound:
        out(f"Building the sandbox image {tag} (first run only; about 2 minutes)...")
    try:
        return ensure_tabular_image(client)
    except DockerException as exc:
        raise PreflightError(f"could not build the sandbox image {tag}: {exc}") from None


@dataclass
class ProgressPrinter:
    """Prints a run's notable events once each, in order, as they appear in the log."""

    events: EventLog
    run_id: str
    out: Callable[[str], None]
    last_seq: int = 0
    start: Any = None
    printed: list[int] = field(default_factory=list)

    def flush(self) -> None:
        """Print every notable event recorded since the last flush."""
        for event in self.events.read(self.run_id, EventFilters(after_seq=self.last_seq)):
            self.last_seq = event.seq
            if self.start is None:
                self.start = event.created_at
            if self._notable(event):
                offset = format_offset((event.created_at - self.start).total_seconds())
                self.out(f"  {offset:>8}  {event.actor_role.value:<14} {describe(event)}")
                self.printed.append(event.seq)

    @staticmethod
    def _notable(event: AnyEvent) -> bool:
        payload = event.payload
        if isinstance(payload, ToolResultPayload):
            return payload.status is not ToolResultStatus.OK
        return event.type in PROGRESS_TYPES


async def run_with_progress(
    controller: RunController,
    request: RunRequest,
    printer: ProgressPrinter | None,
    *,
    poll_seconds: float = 1.0,
) -> RunResult:
    """Run the request; meanwhile print new notable events every `poll_seconds`."""
    if printer is None:
        return await controller.run(request)
    done = anyio.Event()

    async def follow() -> None:
        while not done.is_set():
            printer.flush()
            with anyio.move_on_after(poll_seconds):
                await done.wait()
        printer.flush()

    async with anyio.create_task_group() as group:
        group.start_soon(follow)
        try:
            result = await controller.run(request)
        finally:
            done.set()
    return result


async def execute(
    options: RunOptions,
    *,
    out: Callable[[str], None],
    adapter_factory: AdapterFactory = provider_adapter,
    docker_client: Callable[[], Any] = docker.from_env,
    run_config: RunConfig | None = None,
    poll_seconds: float = 1.0,
) -> RunResult:
    """Preflight, then run the request to a terminal state; raises PreflightError."""
    dataset = check_dataset(options.data)
    config = run_config or run_config_for(options)
    budget = budget_for(options, config)
    run_id = options.run_id or new_run_id()
    request = RunRequest(
        dataset=dataset,
        goal=options.goal,
        target=options.target,
        task_type=options.task_type,
        primary_metric=options.metric,
        forbidden_columns=options.forbid,
        budget=budget,
        run_id=run_id,
    )
    runs_root = options.runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    options.db.parent.mkdir(parents=True, exist_ok=True)
    database = Database(options.db.resolve())
    try:
        migrate(database)
        try:
            adapter = adapter_factory(config, database, runs_root)
        except MissingCredentialError as exc:
            raise PreflightError(str(exc)) from None
        client = check_docker(docker_client)
        if options.worker is WorkerMode.COMPOSE:
            problem = compose_worker_problem(client, options.db, runs_root)
            if problem is not None:
                raise PreflightError(problem)
        prepare_sandbox_image(client, out)
        controller = RunController(
            database=database,
            runs_root=runs_root,
            adapter=adapter,
            run_config=config,
            worker_factory=worker_factory(options.worker),
        )
        out(
            f"Run {run_id}: {config.model_profile.value} profile, "
            f"{options.worker.value} worker, dataset {dataset.name}"
        )
        printer = None if options.quiet else ProgressPrinter(EventLog(database), run_id, out)
        return await run_with_progress(controller, request, printer, poll_seconds=poll_seconds)
    finally:
        database.close()


def result_lines(result: RunResult) -> Sequence[str]:
    """The summary printed when a run ends."""
    verdict = "verified model" if result.verified else "no verified model"
    lines = [
        f"Run {result.run_id} ended {result.state.status.value} ({verdict})",
        f"  final trial  {result.final_trial_id or '-'}",
        f"  limits       {', '.join(result.limits_fired) or 'none fired'}",
        f"  report       {result.report_md}",
    ]
    lines += [f"  note         {note}" for note in result.notes]
    lines.append(f"Next: hypora status {result.run_id} | hypora trace {result.run_id}")
    return lines


__all__ = [
    "AdapterFactory",
    "PreflightError",
    "ProgressPrinter",
    "RunOptions",
    "budget_for",
    "check_dataset",
    "check_docker",
    "execute",
    "provider_adapter",
    "result_lines",
    "run_config_for",
    "run_with_progress",
]
