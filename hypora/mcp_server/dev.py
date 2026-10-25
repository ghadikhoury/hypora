"""A local workspace for trying the MCP server in the MCP Inspector (docs/notes/mcp-server.md).

    uv run --locked python -m hypora.mcp_server.dev [--force]

creates `runs/mcp-dev/` (git-ignored): a migrated state database, one run, an approved
experiment contract, and a session file for role `single_agent`. The dataset is real UCI data:
the Breast Cancer Wisconsin (Diagnostic) set as bundled with scikit-learn (569 rows, 30 features,
binary target), so no download is needed (D-015). It is a developer workspace, not an evaluation
case. The production launcher is the controller (S1.5), which uses the same session format.
"""

import argparse
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
from hypora.mcp_server.session import DefenseSwitches, SessionConfig
from hypora.storage.database import Database
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run, save_approval, save_experiment_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = REPO_ROOT / "runs" / "mcp-dev"
SESSION_FILE = DEV_ROOT / "session.json"
RUN_ID = "run_mcp_dev"
APPROVAL_ID = "appr_mcp_dev_plan"


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
        save_experiment_contract(
            database, dev_contract(RUN_ID, [c for c in frame.columns if c != "target"], plan_hash)
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
    config = SessionConfig(
        session_id="sess_mcp_dev",
        role=ActorRole.SINGLE_AGENT,
        run_id=RUN_ID,
        database=str(database_path),
        runs_root=str(root / "runs"),
        datasets={"ds_main": str(dataset)},
        experiment_dataset="ds_main",
        defenses=DefenseSwitches(allowlist=defenses.allowlist, firewall=defenses.firewall),
    )
    session_file = root / "session.json"
    session_file.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return session_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hypora.mcp_server.dev")
    parser.add_argument("--force", action="store_true", help="recreate an existing workspace")
    args = parser.parse_args(argv)
    session_file = create_dev_workspace(force=args.force)
    root = session_file.parent
    print(f"Dev workspace: {root}")
    print("Inspector:  uv run --locked mcp dev hypora/mcp_server/dev_server.py")
    print(
        "Worker:     uv run --locked python -m hypora.execution.worker.experiments "
        f"--db {root / 'hypora.sqlite'} --runs-root {root / 'runs'}"
    )


if __name__ == "__main__":
    main()
