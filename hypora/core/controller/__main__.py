"""`python -m hypora.core.controller`: run one dataset and goal end to end (PLAN.md S1.5).

    uv run --locked python -m hypora.core.controller --data data.csv \\
        --goal "Predict whether a tumour is malignant" --target target

Uses the active model profile (`HYPORA_MODEL_PROFILE`, default `dev`) and the provider keys in
`.env`. It needs Docker for the sandbox. The state database and run folders go under `runs/`
(git-ignored) unless --db / --runs-root say otherwise. The user-facing CLI is S1.6; this entry
point is for development and the S1.5 live check.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from hypora.core.config import load_settings
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import RunController, RunRequest
from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.rate_limit import ProviderAdmission
from hypora.core.llm.sinks import FileModelIOStore
from hypora.core.schemas import MetricName, TaskType
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the developer entry point arguments."""
    parser = argparse.ArgumentParser(prog="python -m hypora.core.controller")
    parser.add_argument("--data", type=Path, required=True, help="CSV or Parquet dataset")
    parser.add_argument("--goal", required=True, help="the objective, in plain language")
    parser.add_argument("--target", help="target column (else Intake infers it from the goal)")
    parser.add_argument("--task-type", type=TaskType, choices=list(TaskType), default=None)
    parser.add_argument("--metric", type=MetricName, choices=list(MetricName), default=None)
    parser.add_argument("--forbid", action="append", default=[], help="a column not to use")
    parser.add_argument("--run-id", help="run ID (default: a new one)")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "runs" / "hypora.sqlite")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    """Execute the developer entry point and return its process exit code."""
    settings = load_settings()
    run_config = RunConfig.build(settings)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    database = Database(args.db.resolve())
    try:
        migrate(database)
        adapter = LLMAdapter.from_run_config(
            run_config,
            events=EventLog(database),
            admission=ProviderAdmission(
                run_config.models.rate_limits, run_config.pools.llm_concurrency
            ),
            io_store=FileModelIOStore(args.runs_root.resolve()),
        )
        controller = RunController(
            database=database, runs_root=args.runs_root, adapter=adapter, run_config=run_config
        )
        result = await controller.run(
            RunRequest(
                dataset=args.data,
                goal=args.goal,
                target=args.target,
                task_type=args.task_type,
                primary_metric=args.metric,
                forbidden_columns=tuple(args.forbid),
                run_id=args.run_id,
            )
        )
    finally:
        database.close()
    print(f"run:      {result.run_id}")
    print(f"state:    {result.state}")
    print(f"final:    {result.final_trial_id or '-'} (verified: {result.verified})")
    print(f"limits:   {', '.join(result.limits_fired) or 'none fired'}")
    print(f"report:   {result.report_md}")
    for note in result.notes:
        print(f"  - {note}")
    return 0 if result.verified else 1


def main(argv: list[str] | None = None) -> None:
    """Run the developer entry point from a POSIX terminal."""
    logging.basicConfig(
        stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(run(parse_args(argv))))


if __name__ == "__main__":
    main()
