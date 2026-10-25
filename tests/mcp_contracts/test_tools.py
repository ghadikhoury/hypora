"""Each S1.4 tool through a real MCP client: success paths and every error path (PLAN.md S1.4)."""

import os

import pytest
import yaml  # type: ignore[import-untyped]
from mcp import Client

from hypora.core.policy.firewall import DataFirewall
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.tools import _sandbox_packages
from hypora.storage.records import create_run, load_trial
from tests.mcp_contracts.helpers import (
    DESIGN,
    LINEAR,
    RUN_ID,
    call,
    events,
    finish_job,
    new_trial,
    open_session,
    submit,
)

pytestmark = pytest.mark.anyio


SANDBOX_PANDAS = _sandbox_packages()["pandas"]


def code(content):
    return content["code"]


def last_denial(session):
    """The most recent deny decision (the boundary's firewall 'allow' for the error comes after)."""
    return [
        e.payload for e in events(session, "policy_decision") if e.payload.decision.value == "deny"
    ][-1]


# --- inspect_dataset ------------------------------------------------------------------------------


async def test_inspect_dataset_profiles_the_contracts_dataset_without_rows(client, session):
    error, profile = await call(client, "inspect_dataset")
    assert not error
    assert profile["dataset_ref"] == "dataset:ds_main"  # an ID, never a host path
    tabular = profile["tabular"]
    assert tabular["n_rows"] == 569 and tabular["target"]["column"] == "target"
    frame = session.frame("ds_main")
    assert not DataFirewall([frame]).check_payload(profile).blocked
    again_error, again = await call(client, "inspect_dataset", {"dataset_id": "ds_main"})
    assert not again_error and again == profile  # cached


async def test_inspect_dataset_structural_and_explicit_target(client):
    error, profile = await call(
        client, "inspect_dataset", {"target": "mean radius", "task_type": "regression"}
    )
    assert not error and profile["tabular"]["target"]["column"] == "mean radius"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"dataset_id": "ds_other"}, "DATASET_NOT_FOUND"),
        ({"target": "target"}, "TASK_TYPE_REQUIRED"),
        ({"task_type": "regression"}, "TARGET_REQUIRED"),
        ({"target": "nope", "task_type": "regression"}, "UNKNOWN_COLUMN"),
        ({"target": "target", "task_type": "forecasting"}, "TASK_TYPE_UNSUPPORTED"),
        ({"dataset_id": "../etc"}, "INVALID_ARGUMENTS"),
    ],
)
async def test_inspect_dataset_errors(client, arguments, expected):
    error, content = await call(client, "inspect_dataset", arguments)
    assert error and code(content) == expected


# --- write_trial_files ----------------------------------------------------------------------------


async def test_write_trial_files_creates_a_trial_in_its_own_directory(client, session):
    error, content = await call(
        client,
        "write_trial_files",
        {
            "design": DESIGN,
            "pipeline_py": LINEAR,
            "requirements_lock": f"pandas=={SANDBOX_PANDAS}\n",
        },
    )
    assert not error and content["created"]
    trial_id = content["trial_id"]
    assert content["trial_dir"] == f"runs/{RUN_ID}/trials/{trial_id}"
    assert sorted(content["present"]) == ["pipeline.py", "requirements.lock", "trial.yaml"]
    assert content["submit_defaults"]["approval_ref"] == "appr_mcp_dev_plan"
    directory = session.runs_root / RUN_ID / "trials" / trial_id
    written = yaml.safe_load((directory / "trial.yaml").read_text())
    assert written["trial_id"] == trial_id and written["kind"] == "baseline"
    assert (directory / "pipeline.py").read_text() == LINEAR
    assert load_trial(session.database, trial_id).status == "draft"


async def test_write_trial_files_updates_only_the_files_given(client, session):
    trial_id = await new_trial(client)
    changed = LINEAR + "\n# tuned\n"
    error, content = await call(
        client, "write_trial_files", {"trial_id": trial_id, "pipeline_py": changed}
    )
    assert not error and not content["created"]
    assert [f["name"] for f in content["files"]] == ["pipeline.py"]
    directory = session.runs_root / RUN_ID / "trials" / trial_id
    assert (directory / "pipeline.py").read_text() == changed


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"pipeline_py": LINEAR}, "TRIAL_INCOMPLETE"),
        ({"design": DESIGN, "pipeline_py": "def fit(:\n"}, "PIPELINE_SYNTAX_ERROR"),
        (
            {"design": DESIGN, "pipeline_py": "def fit(train, ctx):\n    pass\n"},
            "PIPELINE_INTERFACE",
        ),
        ({"design": DESIGN, "pipeline_py": "x" * (1024 * 1024 + 1)}, "PIPELINE_TOO_LARGE"),
        ({"trial_id": "trl_missing", "pipeline_py": LINEAR}, "TRIAL_NOT_FOUND"),
        (
            {"design": DESIGN, "pipeline_py": LINEAR, "requirements_lock": "requests==2.32.0"},
            "DEPENDENCY_NOT_APPROVED",
        ),
        (
            {"design": DESIGN, "pipeline_py": LINEAR, "requirements_lock": "pandas==2.0.0"},
            "DEPENDENCY_NOT_APPROVED",
        ),
    ],
)
async def test_write_trial_files_errors(client, arguments, expected):
    error, content = await call(client, "write_trial_files", arguments)
    assert error and code(content) == expected


async def test_unapproved_dependencies_are_a_logged_policy_decision(client, session):
    await call(
        client,
        "write_trial_files",
        {"design": DESIGN, "pipeline_py": LINEAR, "requirements_lock": "torch==2.9.0"},
    )
    assert last_denial(session).decided_by.value == "approval_state"


async def test_an_empty_update_is_refused(client):
    trial_id = await new_trial(client)
    error, content = await call(client, "write_trial_files", {"trial_id": trial_id})
    assert error and code(content) == "NOTHING_TO_WRITE"


async def test_writes_replace_a_planted_link_and_never_follow_it(client, session, tmp_path):
    trial_id = await new_trial(client)
    target = tmp_path / "host_file.txt"
    target.write_text("must not change")
    pipeline = session.runs_root / RUN_ID / "trials" / trial_id / "pipeline.py"
    pipeline.unlink()
    os.symlink(target, pipeline)
    error, _ = await call(
        client, "write_trial_files", {"trial_id": trial_id, "pipeline_py": LINEAR}
    )
    assert not error
    assert target.read_text() == "must not change"
    assert not pipeline.is_symlink() and pipeline.read_text() == LINEAR


async def test_a_linked_trial_directory_is_refused(client, session, tmp_path):
    trial_id = await new_trial(client)
    directory = session.runs_root / RUN_ID / "trials" / trial_id
    outside = tmp_path / "outside"
    outside.mkdir()
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()
    os.symlink(outside, directory)
    error, content = await call(
        client, "write_trial_files", {"trial_id": trial_id, "pipeline_py": LINEAR}
    )
    assert error and code(content) == "UNSAFE_TRIAL_DIR"
    assert list(outside.iterdir()) == []


# --- submit_experiment ----------------------------------------------------------------------------


async def test_submit_freezes_the_trial_and_queues_a_fit_job(client, session):
    trial_id = await new_trial(client)
    error, content = await submit(client, trial_id)
    assert not error and content["state"] == "QUEUED" and not content["deduplicated"]
    job = session.queue.get(content["job_id"])
    assert job.trial_id == trial_id and job.operation.value == "fit"
    folder = session.runs_root / RUN_ID / "jobs" / content["job_id"]
    assert (folder / "trial" / "pipeline.py").read_text() == LINEAR
    await call(client, "write_trial_files", {"trial_id": trial_id, "pipeline_py": LINEAR + "#\n"})
    assert (folder / "trial" / "pipeline.py").read_text() == LINEAR  # the snapshot is frozen
    transitions = [e.payload.to_state.value for e in events(session, "job_transition")]
    assert transitions == ["DRAFT", "VALIDATING", "AWAITING_APPROVAL", "QUEUED"]


async def test_submit_is_idempotent_per_key(client, session):
    trial_id = await new_trial(client)
    _, first = await submit(client, trial_id, key="same")
    error, second = await submit(client, trial_id, key="same")
    assert not error and second["deduplicated"] and second["job_id"] == first["job_id"]
    error, reused = await submit(client, trial_id, key="same", config={"C": 2.0})
    assert error and code(reused) == "IDEMPOTENCY_KEY_REUSED"


async def test_submit_accepts_lower_limits(client, session):
    trial_id = await new_trial(client)
    limits = {"max_wall_seconds": 120, "max_memory_gb": 1, "cpu_threads": 1}
    error, content = await submit(client, trial_id, limits=limits)
    assert not error and content["limits"]["max_wall_seconds"] == 120


@pytest.mark.parametrize(
    ("extra", "expected", "rule"),
    [
        ({"run_id": "run_other"}, "RUN_SCOPE_VIOLATION", "contract_scope"),
        ({"contract_version": 2}, "CONTRACT_VERSION_MISMATCH", None),
        ({"approval_ref": "appr_other"}, "APPROVAL_MISMATCH", "approval_state"),
        ({"operation": "finetune", "model_ref": "org/model@abc"}, "OPERATION_UNSUPPORTED", None),
        ({"family": "vision"}, "OPERATION_UNSUPPORTED", None),
        (
            {"limits": {"max_wall_seconds": 3600, "max_memory_gb": 2}},
            "LIMITS_EXCEED_CONTRACT",
            "contract_scope",
        ),
        (
            {"limits": {"max_wall_seconds": 60, "max_memory_gb": 1, "gpu": True}},
            "LIMITS_EXCEED_CONTRACT",
            "contract_scope",
        ),
    ],
)
async def test_submit_refusals(client, session, extra, expected, rule):
    trial_id = await new_trial(client)
    error, content = await submit(client, trial_id, **extra)
    assert error and code(content) == expected
    if rule:
        assert last_denial(session).decided_by.value == rule
    assert not list((session.runs_root / RUN_ID).glob("jobs/*"))  # nothing was queued


async def test_submit_needs_this_runs_key_and_an_existing_trial(client):
    trial_id = await new_trial(client)
    error, content = await call(
        client, "submit_experiment", {"trial_id": trial_id, "idempotency_key": "run_x:s1"}
    )
    assert error and code(content) == "IDEMPOTENCY_KEY_FORMAT"
    error, content = await submit(client, "trl_nope")
    assert error and code(content) == "TRIAL_NOT_FOUND"
    error, content = await call(client, "submit_experiment", {"trial_id": trial_id})
    assert error and code(content) == "INVALID_ARGUMENTS"  # idempotency_key is required


async def test_submit_without_a_contract(session_file):
    session = open_session(session_file, run_id="run_without_contract")
    create_run(session.database, "run_without_contract", "ws_mcp_dev")
    async with Client(HyporaMCPServer(session)) as client:
        error, content = await call(
            client,
            "write_trial_files",
            {"design": DESIGN, "pipeline_py": LINEAR},
        )
        assert not error and content["submit_defaults"] is None
        error, content = await call(
            client,
            "submit_experiment",
            {"trial_id": content["trial_id"], "idempotency_key": "run_without_contract:s1"},
        )
        assert error and code(content) == "NO_CONTRACT"
    session.close()


async def test_another_runs_trials_and_jobs_are_invisible(client, session, session_file):
    trial_id = await new_trial(client)
    _, submitted = await submit(client, trial_id)
    other = open_session(session_file, run_id="run_other_run", session_id="sess_other")
    create_run(other.database, "run_other_run", "ws_mcp_dev")
    async with Client(HyporaMCPServer(other)) as intruder:
        error, content = await call(
            intruder, "get_experiment_status", {"job_id": submitted["job_id"]}
        )
        assert error and code(content) == "EXPERIMENT_NOT_FOUND"
        error, content = await call(
            intruder, "write_trial_files", {"trial_id": trial_id, "pipeline_py": LINEAR}
        )
        assert error and code(content) == "TRIAL_NOT_FOUND"
    other.close()


# --- status, metrics, logs ------------------------------------------------------------------------


async def test_status_of_a_queued_job(client):
    trial_id = await new_trial(client)
    _, submitted = await submit(client, trial_id)
    error, status = await call(client, "get_experiment_status", {"job_id": submitted["job_id"]})
    assert not error and status["state"] == "QUEUED" and not status["finished"]
    assert status["succeeded"] is None and status["artifacts"] == []
    error, content = await call(client, "get_experiment_metrics", {"job_id": submitted["job_id"]})
    assert error and code(content) == "EXPERIMENT_NOT_FINISHED" and content["retryable"]


async def test_unknown_jobs(client):
    for tool in ("get_experiment_status", "get_experiment_metrics", "get_experiment_logs"):
        error, content = await call(client, tool, {"job_id": "job_unknown"})
        assert error and code(content) == "EXPERIMENT_NOT_FOUND"


GROUPED_SCORES = {
    "folds": [
        {
            "rows": 100,
            "metrics": {
                "roc_auc": 0.97 + 0.01 * fold,
                "f1_macro": 0.9,
                "log_loss": 0.2,
                "pr_auc": None,
                "confusion_matrix": [[40, 2], [3, 55]],
                "per_class": [
                    {"label": 0, "precision": 0.9, "recall": 0.95, "f1": 0.92, "support": 42},
                    {"label": 1, "precision": 0.96, "recall": 0.95, "f1": 0.95, "support": 58},
                ],
            },
            "unavailable": {"pr_auc": "scoring partition does not contain every training class"},
            "per_group": {
                "patient-0042": {"metrics": {"roc_auc": 0.91}, "rows": 30},
                "patient-0077": {"metrics": {"roc_auc": 0.99}, "rows": 70},
            },
        }
        for fold in range(3)
    ]
}


async def test_metrics_of_a_completed_job(client, session):
    trial_id = await new_trial(client)
    _, submitted = await submit(client, trial_id)
    finish_job(session, submitted["job_id"], scores=GROUPED_SCORES)
    error, status = await call(client, "get_experiment_status", {"job_id": submitted["job_id"]})
    assert not error and status["succeeded"] and status["artifacts"][0]["kind"] == "metrics"
    error, metrics = await call(client, "get_experiment_metrics", {"job_id": submitted["job_id"]})
    assert not error
    assert metrics["primary_metric"] == "roc_auc" and metrics["direction"] == "maximize"
    assert metrics["primary"]["folds"] == 3
    assert metrics["primary"]["mean"] == pytest.approx(0.98)
    assert metrics["class_labels"] == ["0", "1"]
    assert metrics["per_class"][1]["support"] == 3 * 58
    groups = metrics["folds"][0]["groups"]
    assert groups == {"n_groups": 2, "mean": pytest.approx(0.95), "worst": 0.91, "best": 0.99}
    assert "patient-0042" not in str(metrics)  # group labels are data values: never returned
    assert metrics["folds"][0]["unavailable"]["pr_auc"].startswith("scoring partition")


async def test_metrics_and_status_of_a_failed_job(client, session):
    trial_id = await new_trial(client)
    _, submitted = await submit(client, trial_id)
    finish_job(
        session,
        submitted["job_id"],
        logs=["Traceback (most recent call last):", "ValueError: boom"],
        error=("STAGE_FAILED", "stage 0 (fit): exit code 1"),
    )
    error, status = await call(client, "get_experiment_status", {"job_id": submitted["job_id"]})
    assert not error and status["finished"] and status["succeeded"] is False
    assert status["state"] == "FAILED_TERMINAL"
    assert status["error"] == {"code": "STAGE_FAILED", "message": "stage 0 (fit): exit code 1"}
    assert status["log_tail"][-1] == "ValueError: boom"
    assert "get_experiment_logs" in status["next_step"]
    error, content = await call(client, "get_experiment_metrics", {"job_id": submitted["job_id"]})
    assert error and code(content) == "EXPERIMENT_FAILED"


async def test_logs_are_paginated(client, session):
    trial_id = await new_trial(client)
    _, submitted = await submit(client, trial_id)
    finish_job(session, submitted["job_id"], logs=[f"epoch {i}" for i in range(7)])
    job = {"job_id": submitted["job_id"]}
    error, first = await call(client, "get_experiment_logs", {**job, "page_size": 3})
    assert not error and first["total_lines"] == 7 and first["next_page"] == 2
    assert [line["text"] for line in first["lines"]] == ["epoch 0", "epoch 1", "epoch 2"]
    _, last = await call(client, "get_experiment_logs", {**job, "page": 3, "page_size": 3})
    assert [line["text"] for line in last["lines"]] == ["epoch 6"] and last["next_page"] is None
    error, content = await call(client, "get_experiment_logs", {**job, "page_size": 500})
    assert error and code(content) == "INVALID_ARGUMENTS"
