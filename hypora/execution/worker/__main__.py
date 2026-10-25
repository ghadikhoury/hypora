"""`python -m hypora.execution.worker --db hypora.sqlite --planner package.module:planner`

Runs one execution worker against the state database until interrupted (Ctrl-C / SIGTERM), or,
with `--once`, until nothing is left for it to run. Linux/WSL2 with Docker.

The worker process runs with `/` as its working directory and uses absolute paths only. Docker
Desktop's bind mounts can invalidate the working directory of processes that sit on the Windows
drive (docs/notes/dev-environment.md); a worker in `/` never depends on where it was started.
"""

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from hypora.core.config import detect_resources, load_settings, resolve_pools
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.worker.planner import load_planner
from hypora.execution.worker.worker import ExecutionWorker, default_worker_id
from hypora.storage.database import Database
from hypora.storage.jobs import JobQueue

WORKING_DIRECTORY = "/"


def detach_from_start_directory() -> None:
    """Make every relative reference absolute, then move the process to `/`.

    `sys.path` entries are resolved first, so planner and harness imports still work after the
    move. Callers resolve their own paths (the database path, the config directory) before this.
    """
    sys.path[:] = [os.path.abspath(entry or os.curdir) for entry in sys.path]
    os.chdir(WORKING_DIRECTORY)


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

    # Everything that reads a path happens before the move to `/`, with absolute paths.
    settings = load_settings()
    pools = resolve_pools(
        detect_resources(settings.resources.gpu_detection_timeout_s), settings.resources
    )
    database = Database(args.db.resolve())
    worker = ExecutionWorker(
        JobQueue(database, lease_seconds=args.lease_seconds),
        SandboxLauncher(),
        load_planner(args.planner),
        pools,
        worker_id=args.worker_id,
        poll_seconds=args.poll_seconds,
    )
    detach_from_start_directory()
    stop = threading.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, lambda *_: stop.set())
    logging.getLogger(__name__).info(
        "worker %s started in %s with pools %s", worker.worker_id, os.getcwd(), pools
    )
    try:
        worker.run(stop, exit_when_idle=args.once)
    finally:
        for signum, handler in previous.items():  # leave an embedding process as we found it
            signal.signal(signum, handler)
        database.close()


if __name__ == "__main__":
    main()
