"""`python -m hypora.execution.worker --db hypora.sqlite --planner package.module:planner`

Runs one execution worker against the state database until interrupted (Ctrl-C / SIGTERM), or,
with `--once`, until nothing is left for it to run. Linux/WSL2 with Docker.
"""

import argparse
import logging
import signal
import threading
from pathlib import Path

from hypora.core.config import detect_resources, load_settings, resolve_pools
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.worker.planner import load_planner
from hypora.execution.worker.worker import ExecutionWorker, default_worker_id
from hypora.storage.database import Database
from hypora.storage.jobs import JobQueue


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hypora.execution.worker")
    parser.add_argument("--db", type=Path, required=True, help="state database (SQLite file)")
    parser.add_argument("--planner", required=True, help="job planner, 'package.module:name'")
    parser.add_argument("--worker-id", default=default_worker_id())
    parser.add_argument("--lease-seconds", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="exit when idle")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = load_settings()
    pools = resolve_pools(
        detect_resources(settings.resources.gpu_detection_timeout_s), settings.resources
    )
    database = Database(args.db)
    worker = ExecutionWorker(
        JobQueue(database, lease_seconds=args.lease_seconds),
        SandboxLauncher(),
        load_planner(args.planner),
        pools,
        worker_id=args.worker_id,
        poll_seconds=args.poll_seconds,
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    logging.getLogger(__name__).info("worker %s started with pools %s", worker.worker_id, pools)
    try:
        worker.run(stop, exit_when_idle=args.once)
    finally:
        database.close()


if __name__ == "__main__":
    main()
