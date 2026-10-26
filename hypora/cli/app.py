"""The `hypora` command line (spec §21; PLAN.md S1.6): run, status, trace, approve, cancel.

    hypora run --data data/snapshots/openml-46918-credit-g.parquet \\
        --goal "Predict whether a loan applicant is a good or bad credit risk"
    hypora status [RUN_ID]
    hypora trace RUN_ID [--filter type=model_call,tool_call] [--filter role=single_agent]
    hypora cancel RUN_ID
    hypora approve RUN_ID        # placeholder until the interactive plan gate (S3.D2)

The CLI is one client of the runtime (spec principle 11): it builds a `RunRequest` and hands it
to the controller, and it reads state only from the event log and the stored records. State lives
in `runs/hypora.sqlite` and run folders in `runs/` unless --db / --runs-root say otherwise.

Exit codes: 0 success (for `run`: a verified final model); 1 the command ran but did not succeed
(no verified model, unknown run, nothing to cancel); 2 the input or the environment is not ready
(preflight); 130 interrupted.
"""

import shutil
import sys
from pathlib import Path
from typing import Annotated, Final

import anyio
import typer

from hypora.cli.run import PreflightError, RunOptions, execute, result_lines
from hypora.cli.status import TERMINAL, run_status_lines, runs_table_lines
from hypora.cli.trace import parse_filters, render_rows, summary, trace_rows
from hypora.cli.workers import WorkerMode
from hypora.core.controller.controller import RunResult
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.schemas import ApprovalPolicy, MetricName, TaskType
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import RecordNotFound, load_run
from hypora.storage.snapshots import SNAPSHOTS_DIR, load_manifest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT: Final = REPO_ROOT / "runs"
DEFAULT_DB: Final = DEFAULT_RUNS_ROOT / "hypora.sqlite"

app = typer.Typer(
    name="hypora",
    help="Hypora: describe the outcome you need; the agent experiments and reports what works.",
    no_args_is_help=True,
    add_completion=False,
    # Plain tracebacks: Typer's rich tracebacks can print local variables, and a local may hold a
    # provider key while a client is built.
    pretty_exceptions_enable=False,
)

DbOption = Annotated[Path, typer.Option("--db", help="State database (SQLite).")]
RunsRootOption = Annotated[Path, typer.Option("--runs-root", help="Folder of run directories.")]


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _open_existing(db: Path) -> Database:
    """The state database for read-only commands; it must already exist."""
    if not db.is_file():
        raise _fail(f"no state database at {db} (no run has been started with it yet)", 1)
    return Database(db.resolve())


@app.command()
def run(
    data: Annotated[Path, typer.Option("--data", help="Dataset: a .csv or .parquet file.")],
    goal: Annotated[str, typer.Option("--goal", help="The objective, in plain language.")],
    target: Annotated[
        str | None, typer.Option("--target", help="Target column (else inferred from the goal).")
    ] = None,
    task_type: Annotated[TaskType | None, typer.Option("--task-type")] = None,
    metric: Annotated[MetricName | None, typer.Option("--metric", help="Primary metric.")] = None,
    forbid: Annotated[
        list[str] | None, typer.Option("--forbid", help="A column the model must not use.")
    ] = None,
    budget_usd: Annotated[
        float | None, typer.Option("--budget-usd", min=0, help="Run spend cap in USD.")
    ] = None,
    max_trials: Annotated[int | None, typer.Option("--max-trials", min=0)] = None,
    max_wall_clock: Annotated[
        float | None, typer.Option("--max-wall-clock", min=1, help="Run time cap in seconds.")
    ] = None,
    approve: Annotated[
        ApprovalPolicy | None,
        typer.Option("--approve", help="Plan approval; Stage 1 supports auto_within_contract."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Run ID (default: new).")] = None,
    worker: Annotated[
        WorkerMode,
        typer.Option("--worker", help="host: a worker process per run; compose: `make worker-up`."),
    ] = WorkerMode.HOST,
    db: DbOption = DEFAULT_DB,
    runs_root: RunsRootOption = DEFAULT_RUNS_ROOT,
    quiet: Annotated[bool, typer.Option("--quiet", help="Do not print progress.")] = False,
) -> None:
    """Run the agent on a dataset and a goal, end to end, and print where the report is."""
    options = RunOptions(
        data=data,
        goal=goal,
        db=db,
        runs_root=runs_root,
        target=target,
        task_type=task_type,
        metric=metric,
        forbid=tuple(forbid or ()),
        budget_usd=budget_usd,
        max_trials=max_trials,
        max_wall_clock_s=max_wall_clock,
        approve=approve,
        run_id=run_id,
        worker=worker,
        quiet=quiet,
    )
    try:
        result = anyio.run(_execute, options)
    except PreflightError as exc:
        raise _fail(str(exc), 2) from None
    except ValueError as exc:  # e.g. an invalid --run-id
        raise _fail(str(exc), 2) from None
    except KeyboardInterrupt:
        typer.echo(
            "\nInterrupted. The run stopped without its report and keeps its last status until "
            "durable recovery (S3.E3). For a clean stop that still writes the report, use "
            "`hypora cancel <run_id>` from another terminal.",
            err=True,
        )
        raise typer.Exit(130) from None
    for line in result_lines(result):
        typer.echo(line)
    raise typer.Exit(0 if result.verified else 1)


async def _execute(options: RunOptions) -> RunResult:
    return await execute(options, out=typer.echo)


@app.command()
def status(
    run_id: Annotated[
        str | None, typer.Argument(help="A run ID; omit to list recent runs.")
    ] = None,
    db: DbOption = DEFAULT_DB,
    runs_root: RunsRootOption = DEFAULT_RUNS_ROOT,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Runs to list.")] = 20,
) -> None:
    """Show one run's status, spend, trials, jobs, and final model; or list recent runs."""
    database = _open_existing(db)
    try:
        manifest = load_manifest(SNAPSHOTS_DIR)
        if run_id is None:
            lines = runs_table_lines(database, manifest, limit)
        else:
            lines = run_status_lines(database, runs_root.resolve(), run_id, manifest)
    except RecordNotFound:
        raise _fail(f"no run {run_id} in {db}", 1) from None
    finally:
        database.close()
    for line in lines:
        typer.echo(line)


@app.command()
def trace(
    run_id: Annotated[str, typer.Argument(help="The run to show.")],
    filters: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            "-f",
            help="key=value[,value]; keys: type, role, trial, job. Repeat to combine (AND).",
        ),
    ] = None,
    wide: Annotated[bool, typer.Option("--wide", help="Never cut the detail column.")] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Print a run's event log as a timeline: role, type, duration, tokens, detail."""
    try:
        selection = parse_filters(filters or [])
    except ValueError as exc:
        raise _fail(str(exc), 2) from None
    database = _open_existing(db)
    try:
        run_status = load_run(database, run_id).status.value
        events = EventLog(database).read(run_id)
    except RecordNotFound:
        raise _fail(f"no run {run_id} in {db}", 1) from None
    finally:
        database.close()
    rows = trace_rows(events)
    shown = [row for row in rows if selection.keeps(row)]
    width = None if wide or not sys.stdout.isatty() else shutil.get_terminal_size().columns
    if events:
        first, last = events[0].created_at, events[-1].created_at
        typer.echo(
            f"Run {run_id} ({run_status}) | {first:%Y-%m-%d %H:%M:%S} to {last:%H:%M:%S} UTC"
        )
    for line in render_rows(shown, width):
        typer.echo(line)
    typer.echo(summary(shown, events) if events else "No events recorded.")


@app.command()
def approve(run_id: Annotated[str, typer.Argument(help="The run whose plan to approve.")]) -> None:
    """Placeholder: interactive plan approval arrives with the plan gate (S3.D2)."""
    typer.echo(
        f"Interactive approval is not available yet (S3.D2). Stage 1 runs approve their plan by "
        f"policy (D-019); see `hypora trace {run_id} --filter "
        "type=approval_requested,approval_granted`.",
        err=True,
    )
    raise typer.Exit(1)


@app.command()
def cancel(
    run_id: Annotated[str, typer.Argument(help="The run to cancel.")],
    db: DbOption = DEFAULT_DB,
    runs_root: RunsRootOption = DEFAULT_RUNS_ROOT,
) -> None:
    """Ask a running run to stop: it cancels its jobs, writes its report, and ends CANCELLED."""
    database = _open_existing(db)
    try:
        run_status = load_run(database, run_id).status
    except RecordNotFound:
        raise _fail(f"no run {run_id} in {db}", 1) from None
    finally:
        database.close()
    if run_status in TERMINAL:
        raise _fail(f"run {run_id} already ended {run_status.value}; nothing to cancel", 1)
    folder = runs_root.resolve() / run_id
    if not folder.is_dir():  # the controller watches this folder; a stray one would be ignored
        raise _fail(f"no run folder {folder}; pass the --runs-root the run was started with", 1)
    (folder / CANCEL_FILE).write_text("cancel requested with `hypora cancel`\n", encoding="utf-8")
    typer.echo(
        f"Cancel requested for {run_id} ({run_status.value}). Its controller stops before the "
        "next model or tool call, cancels unfinished jobs, writes the report, and ends the run "
        f"CANCELLED. Follow it with `hypora status {run_id}`."
    )


def main() -> None:
    """Console-script entry point."""
    app()


__all__ = ["DEFAULT_DB", "DEFAULT_RUNS_ROOT", "app", "main"]
