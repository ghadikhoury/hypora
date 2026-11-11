"""A local workspace for trying the MCP server in the MCP Inspector (docs/notes/mcp-server.md).

    uv run --locked python -m hypora.mcp_server.dev [--force]

creates `runs/mcp-dev/` (git-ignored): a migrated state database, one run, an approved
experiment contract, and a session file for role `single_agent`. The dataset is real UCI data:
the Breast Cancer Wisconsin (Diagnostic) set as bundled with scikit-learn (569 rows, 30 features,
binary target), so no download is needed (D-015). It is a developer workspace, not an evaluation
case. The production launcher is the controller (S1.5), which uses the same session format.

The session file holds only the hash of the session secret (S3.5, D-044). Unlike the controller,
which never writes the secret, the dev workspace keeps it in `session.secret` (mode 0600) so the
Inspector (`dev_server.py`) and hand-run servers can authenticate.
"""

import argparse
import contextlib
import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.core.config import load_settings
from hypora.core.hashing import canonical_hash
from hypora.core.schemas import (
    ActorRole,
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ExperimentContract,
)
from hypora.execution.harness.partition import create_partition_lock
from hypora.execution.harness.runner import load_tabular
from hypora.mcp_server.session import (
    SESSION_SECRET_ENV,
    DefenseSwitches,
    SessionConfig,
    authenticate,
    new_session_secret,
    secret_sha256,
    write_session_file,
)
from hypora.storage.database import Database
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run, save_approval, save_experiment_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = REPO_ROOT / "runs" / "mcp-dev"
SESSION_FILE = DEV_ROOT / "session.json"
SECRET_FILE_NAME = "session.secret"
"""The dev session's secret, next to its session file, mode 0600 (git-ignored like all of runs/).
The production launcher never writes a secret to disk; this developer workspace does, so the
Inspector and hand-run servers can authenticate."""
RUN_ID = "run_mcp_dev"
APPROVAL_ID = "appr_mcp_dev_plan"
STATUS_WAIT_SECONDS = 5.0
"""Short enough for the MCP Inspector's default 10 s request timeout."""


def dev_contract(run_id: str, features: list[str], approved_plan_hash: str) -> ExperimentContract:
    """A binary-classification contract over the given features (stratified 3-fold CV)."""
    return ExperimentContract.model_validate(
        {
            "run_id": run_id,
            "version": 1,
            "approved_plan_hash": approved_plan_hash,
            "approval_id": APPROVAL_ID,
            "family": "tabular",
            "task_type": "binary_classification",
            "target": "target",
            "data_contract": {
                "data_contract_version": 1,
                "target": "target",
                "task_type": "binary_classification",
                "allowed_features": features,
                "split": {
                    "strategy": "stratified",
                    "test_fraction": 0.2,
                    "validation": "stratified_kfold_3",
                },
            },
            "primary_metric": "roc_auc",
            "secondary_metrics": ["f1_macro", "log_loss"],
            "budget": {"max_usd": 1, "max_compute_seconds": 1800, "max_trials": 10},
            "default_job_limits": {
                "max_wall_seconds": 300,
                "max_memory_gb": 2,
                "gpu": False,
                "cpu_threads": 1,
            },
            "seed": 1990,
        }
    )


def create_dev_workspace(root: Path = DEV_ROOT, *, force: bool = False) -> Path:
    """Create the workspace under `root`; returns the session file's path."""
    if root.exists():
        if not force:
            raise FileExistsError(f"{root} exists; pass --force to recreate it")
        shutil.rmtree(root)
    (root / "data").mkdir(parents=True)
    dataset = root / "data" / "breast_cancer_wisconsin.csv"
    frame = load_breast_cancer(as_frame=True).frame
    frame.to_csv(dataset, index=False)

    database_path = root / "hypora.sqlite"
    database = Database(database_path)
    try:
        migrate(database)
        create_run(database, RUN_ID, "ws_mcp_dev")
        plan_hash = canonical_hash({"dev_plan": RUN_ID, "dataset": dataset.name})
        contract = dev_contract(RUN_ID, [c for c in frame.columns if c != "target"], plan_hash)
        save_experiment_contract(database, contract)
        source = dataset.read_bytes()
        lock_path = root / "runs" / RUN_ID / "partition-lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        create_partition_lock(
            load_tabular(dataset, source=source),
            contract,
            hashlib.sha256(source).hexdigest(),
            lock_path,
        )
        save_approval(
            database,
            ApprovalRecord(
                approval_id=APPROVAL_ID,
                run_id=RUN_ID,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=plan_hash,
                approver="user:dev",
                decision=ApprovalDecision.APPROVED,
                policy=ApprovalPolicy.INTERACTIVE,
                created_at=datetime.now(UTC),
            ),
        )
    finally:
        database.close()

    defenses = load_settings().flags.defenses
    secret = new_session_secret()
    config = SessionConfig(
        session_id="sess_mcp_dev",
        role=ActorRole.SINGLE_AGENT,
        run_id=RUN_ID,
        database=str(database_path),
        runs_root=str(root / "runs"),
        datasets={"ds_main": str(dataset)},
        experiment_dataset="ds_main",
        defenses=DefenseSwitches(
            allowlist=defenses.allowlist,
            firewall=defenses.firewall,
            approval=defenses.approval,
            injection_filter=defenses.injection_filter,
        ),
        status_wait_seconds=STATUS_WAIT_SECONDS,
        secret_sha256=secret_sha256(secret),
    )
    session_file = root / "session.json"
    write_session_file(session_file, config)
    _write_secret(root / SECRET_FILE_NAME, secret)
    return session_file


def _write_secret(path: Path, secret: str) -> None:
    """The dev session's secret, owner-only (0600), for the developer's own launch commands."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secret + "\n")


def dev_secret(session_file: Path = SESSION_FILE) -> str:
    """The secret of the dev session next to `session_file` (the Inspector launch reads it)."""
    return (session_file.parent / SECRET_FILE_NAME).read_text(encoding="utf-8").strip()


def authenticate_dev(session_file: Path = SESSION_FILE) -> SessionConfig:
    """Authenticate a dev server the way `python -m hypora.mcp_server` does: with the secret in
    `HYPORA_MCP_SESSION_SECRET` if set, else the workspace's own owner-only secret file."""
    config = SessionConfig.model_validate_json(session_file.read_bytes())
    secret = os.environ.pop(SESSION_SECRET_ENV, None)
    if secret is None:
        with contextlib.suppress(FileNotFoundError):
            secret = dev_secret(session_file)
    authenticate(config, secret)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hypora.mcp_server.dev")
    parser.add_argument("--force", action="store_true", help="recreate an existing workspace")
    args = parser.parse_args(argv)
    session_file = create_dev_workspace(force=args.force)
    root = session_file.parent
    print(f"Dev workspace: {root}")
    print("Inspector:  uv run --locked mcp dev hypora/mcp_server/dev_server.py")
    print(
        f"Secret:     {root / SECRET_FILE_NAME} (owner-only); a hand-run "
        f"`python -m hypora.mcp_server --session {session_file}` needs it in {SESSION_SECRET_ENV}"
    )
    print(
        "Worker:     uv run --locked python -m hypora.execution.worker.experiments "
        f"--db {root / 'hypora.sqlite'} --runs-root {root / 'runs'}"
    )


if __name__ == "__main__":
    main()
