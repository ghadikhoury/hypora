"""Offline unit and integration tests for the Hypora HTTP API and SSE server."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient
from typer.testing import CliRunner

from hypora.api.server import create_app, is_loopback_host
from hypora.cli.app import app as cli_app
from hypora.core.controller.approval_gate import decision_path
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.hashing import sha256_hex
from hypora.core.schemas import (
    ActorRole,
    ApprovalArtifactType,
    ApprovalPolicy,
    CostEstimate,
    DatasetModality,
    DatasetProfile,
    EventType,
    JobState,
    MetricName,
    ModelProfile,
    ModelTier,
    Operation,
    RequestedAction,
    RunStatus,
    Stage,
    TaskSpecification,
    TaskType,
    TrialKind,
)
from hypora.core.schemas.events import (
    ApprovalRequestedPayload,
    CallStatus,
    JobTransitionPayload,
    ModelCallPayload,
    Provider,
    StageTransitionPayload,
)
from hypora.core.schemas.task import ColumnProfile, InferredType, TabularProfile
from hypora.core.schemas.trials import TrialFile
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import (
    create_run,
    save_dataset_profile,
    save_task_spec,
    upsert_result,
    upsert_trial,
)
from hypora.storage.snapshots import SnapshotEntry, SnapshotSource, save_entry


def _seed_run(
    db: Database,
    runs_root: Path,
    *,
    run_id: str = "run-api-001",
    status: RunStatus = RunStatus.PHASE_B,
    approval_policy: ApprovalPolicy = ApprovalPolicy.INTERACTIVE,
    with_pending_approval: bool = False,
) -> dict[str, str]:
    create_run(
        db,
        run_id,
        "ws_test",
        status=status.value,
        approval_policy=approval_policy.value,
        budget_usd=Decimal("1.50"),
        spent_usd=Decimal("0.0125"),
        model_config={"model_profile": "dev"},
    )
    events = EventLog(db)
    events.append(
        run_id,
        EventType.STAGE_TRANSITION,
        ActorRole.CONTROLLER,
        StageTransitionPayload(
            previous_stage=None,
            stage=Stage.INTAKE,
            previous_status=None,
            status=RunStatus.INTAKE,
            reason="start",
        ),
    )
    events.append(
        run_id,
        EventType.MODEL_CALL,
        ActorRole.AGENT_A,
        ModelCallPayload(
            call_id="call-001",
            provider=Provider.GEMINI,
            model_id="gemini-2.5-flash",
            profile=ModelProfile.DEV,
            tier=ModelTier.LEAD,
            temperature=0.0,
            status=CallStatus.OK,
            input_tokens=100,
            output_tokens=50,
            latency_ms=250,
            cost_usd=0.001,
        ),
        trial_id="t1",
        job_id="j1",
    )
    events.append(
        run_id,
        EventType.JOB_TRANSITION,
        ActorRole.EXECUTION_WORKER,
        JobTransitionPayload(
            job_id="j1",
            trial_id="t1",
            to_state=JobState.COMPLETED,
            initiator=ActorRole.EXECUTION_WORKER,
            reason="finished",
        ),
        trial_id="t1",
        job_id="j1",
    )
    spec = TaskSpecification(
        run_id=run_id,
        version=1,
        objective="Predict credit risk",
        requested_actions=[RequestedAction.TRAIN],
        target="class",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ROC_AUC,
    )
    save_task_spec(db, spec)

    fp = sha256_hex(b"synthetic-dataset")
    profile = DatasetProfile(
        run_id=run_id,
        dataset_ref="credit-g.parquet",
        fingerprint=fp,
        modality=DatasetModality.TABULAR,
        tabular=TabularProfile(
            n_rows=1000,
            n_columns=1,
            statistics_sample_size=1000,
            columns=[
                ColumnProfile(
                    name="duration",
                    dtype="int64",
                    inferred_type=InferredType.NUMERIC,
                    n_missing=0,
                    missing_fraction=0.0,
                    n_unique=33,
                    unique_fraction=0.033,
                )
            ],
        ),
    )
    save_dataset_profile(db, profile)

    trial_file = TrialFile(
        trial_id="t1",
        kind=TrialKind.BASELINE,
        operation=Operation.FIT,
        hypothesis="Logistic regression baseline",
        model_family="linear",
        expected_effect="Establish baseline",
        estimated_cost=CostEstimate(compute_seconds=60, usd=0.05),
    )
    upsert_trial(db, run_id, trial_file, "trials/t1")
    upsert_result(
        db,
        "t1",
        "val",
        {"metrics": {"roc_auc": 0.78}},
        {"compute_seconds": 1.2},
        verification={"passed": True},
    )
    upsert_result(
        db,
        "t1",
        "test",
        {"metrics": {"roc_auc": 0.77}},
        {"compute_seconds": 0.5},
        verification={"passed": True},
    )

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps({"run_id": run_id, "verified": True}), encoding="utf-8"
    )
    (run_dir / "report.md").write_text("# Report\n\nVerified baseline.", encoding="utf-8")

    artifact_hash = sha256_hex(b"approved-plan-v1")
    approval_id = "appr-v1-001"
    if with_pending_approval:
        (run_dir / "approved_plan_v1.json").write_text(
            json.dumps({"version": 1, "goal": "Predict credit risk"}), encoding="utf-8"
        )
        (run_dir / "approval_scope_v1.json").write_text(
            json.dumps({"max_usd": 1.5}), encoding="utf-8"
        )
        events.append(
            run_id,
            EventType.APPROVAL_REQUESTED,
            ActorRole.AGENT_A,
            ApprovalRequestedPayload(
                approval_id=approval_id,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=artifact_hash,
                policy=approval_policy,
                summary="Baseline + GBDT plan within $1.50",
                artifact_ref="artifact:approved_plan.v1",
            ),
        )
    return {"artifact_hash": artifact_hash, "approval_id": approval_id, "fingerprint": fp}


def test_health_and_fallback_spa(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "hypora.sqlite"
    runs_root = tmp_path / "runs"
    app = create_app(db_path=db_path, runs_root=runs_root, web_dist=tmp_path / "no-dist")
    client = TestClient(app)

    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "hypora-api"
    assert body["db_exists"] is False
    assert body["web_dist_ready"] is False

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")
    assert client.get("/api/health").json()["db_exists"] is True

    html_res = client.get("/")
    assert html_res.status_code == 200
    assert "npm run build" in html_res.text

    missing_api = client.get("/api/nonexistent")
    assert missing_api.status_code == 404


def test_spa_static_serving(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>Hypora SPA</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok');", encoding="utf-8")

    app = create_app(
        db_path=tmp_path / "hypora.sqlite",
        runs_root=tmp_path / "runs",
        web_dist=dist,
    )
    client = TestClient(app)
    assert client.get("/api/health").json()["web_dist_ready"] is True
    assert "Hypora SPA" in client.get("/").text
    assert "Hypora SPA" in client.get("/runs/some-id").text
    assert "console.log" in client.get("/assets/app.js").text
    assert client.get("/api/unknown").status_code == 404


def test_datasets_endpoint(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    app = create_app(
        db_path=tmp_path / "hypora.sqlite",
        runs_root=tmp_path / "runs",
        snapshots_dir=snapshots_dir,
    )
    client = TestClient(app)
    empty = client.get("/api/datasets").json()
    assert empty["manifest_exists"] is False
    assert empty["datasets"] == []

    parquet_path = snapshots_dir / "openml-46918-credit-g.parquet"
    content = b"PAR1-dummy"
    parquet_path.write_bytes(content)
    entry = SnapshotEntry(
        name="credit_g",
        source=SnapshotSource.OPENML,
        dataset_id=46918,
        title="German Credit",
        version="1",
        source_url="https://www.openml.org/d/46918",
        download_url="https://www.openml.org/data/46918",
        license="CC-BY-4.0",
        license_declared="CC-BY-4.0",
        license_evidence="OpenML metadata",
        target="class",
        rows=1000,
        columns=21,
        file="openml-46918-credit-g.parquet",
        sha256=sha256_hex(content),
        bytes=len(content),
        source_sha256=sha256_hex(content),
        retrieved_at=datetime.now(UTC),
    )
    save_entry(entry, directory=snapshots_dir)

    populated = client.get("/api/datasets").json()
    assert populated["manifest_exists"] is True
    assert len(populated["datasets"]) == 1
    assert populated["datasets"][0]["name"] == "credit_g"
    assert populated["datasets"][0]["exists"] is True


def test_runs_list_detail_and_events_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "hypora.sqlite"
    runs_root = tmp_path / "runs"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    migrate(db)
    _seed_run(db, runs_root, run_id="run-100", status=RunStatus.DONE)
    db.close()

    app = create_app(db_path=db_path, runs_root=runs_root)
    client = TestClient(app)

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-100"
    assert runs[0]["final_verified"] is True
    assert runs[0]["final_score"] == 0.77

    assert client.get("/api/runs/unknown-run").status_code == 404
    assert client.get("/api/runs/..bad").status_code == 400

    detail = client.get("/api/runs/run-100").json()
    assert detail["run_id"] == "run-100"
    assert detail["stage"] == "INTAKE"
    assert detail["total_tokens"] == 150
    assert detail["profile"]["tabular"]["n_rows"] == 1000
    assert len(detail["trials"]) == 1
    assert detail["trials"][0]["test_result"]["metrics"]["metrics"]["roc_auc"] == 0.77
    assert detail["report_json"]["verified"] is True
    assert "Verified baseline" in detail["report_md"]

    ev_all = client.get("/api/runs/run-100/events").json()
    assert ev_all["total_events"] == 3
    assert len(ev_all["events"]) == 3

    ev_role = client.get("/api/runs/run-100/events?role=agent_a").json()
    assert len(ev_role["events"]) == 1
    assert ev_role["events"][0]["type"] == "model_call"

    ev_type = client.get("/api/runs/run-100/events?type=job_transition&trial=t1&job=j1").json()
    assert len(ev_type["events"]) == 1
    assert ev_type["events"][0]["job_id"] == "j1"

    ev_after = client.get("/api/runs/run-100/events?after_seq=2").json()
    assert [e["seq"] for e in ev_after["events"]] == [3]

    assert client.get("/api/runs/run-100/events?role=bogus").status_code == 400


def test_extra_state_dir_discovery_is_read_only(tmp_path: Path) -> None:
    primary_db = tmp_path / "primary" / "hypora.sqlite"
    primary_runs = tmp_path / "primary"
    extra_dir = tmp_path / "runs-archive"
    state_dir = extra_dir / "work" / "state"
    hist_runs = extra_dir / "work" / "runs"
    state_dir.mkdir(parents=True)
    hist_db_path = state_dir / "hist-01.sqlite"
    hist_db = Database(hist_db_path)
    migrate(hist_db)
    meta = _seed_run(
        hist_db,
        hist_runs,
        run_id="hist-01",
        status=RunStatus.PHASE_B,
        with_pending_approval=True,
    )
    hist_db.close()

    app = create_app(
        db_path=primary_db,
        runs_root=primary_runs,
        extra_state_dirs=[extra_dir],
    )
    client = TestClient(app)

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "hist-01"
    assert runs[0]["read_only"] is True

    detail = client.get("/api/runs/hist-01").json()
    assert detail["read_only"] is True
    assert detail["report_json"]["run_id"] == "hist-01"

    approve_res = client.post(
        "/api/runs/hist-01/approve",
        json={"approve": True, "hash_prefix": meta["artifact_hash"][:12]},
    )
    assert approve_res.status_code == 409
    assert client.post("/api/runs/hist-01/cancel").status_code == 409


def test_interactive_approval_flow_and_cancel(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "hypora.sqlite"
    runs_root = tmp_path / "runs"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    migrate(db)
    meta = _seed_run(
        db,
        runs_root,
        run_id="run-appr-1",
        status=RunStatus.WAITING_APPROVAL,
        with_pending_approval=True,
    )
    db.close()

    app = create_app(db_path=db_path, runs_root=runs_root)
    client = TestClient(app)

    detail = client.get("/api/runs/run-appr-1").json()
    pending = detail["pending_approval"]
    assert pending is not None
    assert pending["approval_id"] == meta["approval_id"]
    assert pending["plan"]["version"] == 1
    assert pending["scope"]["max_usd"] == 1.5

    # Wrong hash prefix -> 409
    bad_hash = client.post(
        "/api/runs/run-appr-1/approve",
        json={"approve": True, "hash_prefix": "deadbeef"},
    )
    assert bad_hash.status_code == 409

    # Denial without reason -> 400
    no_reason = client.post(
        "/api/runs/run-appr-1/approve",
        json={"approve": False, "hash_prefix": meta["artifact_hash"][:12]},
    )
    assert no_reason.status_code == 400

    # Valid approve -> 200 & atomic file written
    ok = client.post(
        "/api/runs/run-appr-1/approve",
        json={"approve": True, "hash_prefix": meta["artifact_hash"][:12], "user": "alice"},
    )
    assert ok.status_code == 200
    assert ok.json()["decision"] == "approved"
    expected_file = decision_path(runs_root, "run-appr-1", meta["approval_id"])
    assert expected_file.is_file()
    stored_decision = json.loads(expected_file.read_text(encoding="utf-8"))
    assert stored_decision["decision"] == "approved"
    assert stored_decision["approver"] == "user:alice"

    # Second decision -> 409 (already decided)
    second = client.post(
        "/api/runs/run-appr-1/approve",
        json={"approve": False, "hash_prefix": meta["artifact_hash"][:12], "reason": "too late"},
    )
    assert second.status_code == 409

    # Cancel writes CANCEL marker
    cancel = client.post("/api/runs/run-appr-1/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True
    assert (runs_root / "run-appr-1" / CANCEL_FILE).is_file()
    assert client.post("/api/runs/unknown-xyz/cancel").status_code == 404


def test_create_run_validation_and_launch(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b,target\n1,2,0\n3,4,1\n", encoding="utf-8")
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    launched: list[Any] = []

    def fake_launcher(opts: Any) -> None:
        launched.append(opts)

    app = create_app(
        db_path=tmp_path / "runs" / "hypora.sqlite",
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        run_launcher=fake_launcher,
    )
    client = TestClient(app)

    assert client.post("/api/runs", json={"data": "missing.csv", "goal": "test"}).status_code == 400
    assert client.post("/api/runs", json={"data": str(bad_file), "goal": "test"}).status_code == 400
    assert (
        client.post("/api/runs", json={"data": "../outside.csv", "goal": "test"}).status_code == 400
    )

    created = client.post(
        "/api/runs",
        json={
            "data": str(csv_file),
            "goal": "Predict target",
            "target": "target",
            "approve": "auto_within_contract",
        },
    )
    assert created.status_code == 202
    body = created.json()
    assert body["run_id"]
    assert body["approval_policy"] == "auto_within_contract"
    assert len(launched) == 1
    assert launched[0].goal == "Predict target"


def test_eval_summary_real_and_missing(tmp_path: Path) -> None:
    # Real committed repo artifacts
    real_app = create_app(
        db_path=tmp_path / "hypora.sqlite",
        runs_root=tmp_path / "runs",
    )
    real_client = TestClient(real_app)
    res = real_client.get("/api/eval/summary")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data["consolidated_summary"], dict)
    assert isinstance(data["evidence_index"], dict)
    assert isinstance(data["f6_evidence"], dict)
    assert isinstance(data["f7_evidence"], dict)

    # Missing files return null gracefully
    empty_app = create_app(
        db_path=tmp_path / "hypora.sqlite",
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
    )
    empty_client = TestClient(empty_app)
    empty_data = empty_client.get("/api/eval/summary").json()
    assert empty_data == {
        "consolidated_summary": None,
        "evidence_index": None,
        "f6_evidence": None,
        "f7_evidence": None,
    }


def test_sse_stream_closes_on_terminal(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "hypora.sqlite"
    runs_root = tmp_path / "runs"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    migrate(db)
    _seed_run(db, runs_root, run_id="run-sse-1", status=RunStatus.DONE)
    db.close()

    app = create_app(db_path=db_path, runs_root=runs_root, sse_poll_seconds=0.05)
    client = TestClient(app)

    with client.stream("GET", "/api/runs/run-sse-1/events/stream?after_seq=1") as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())

    assert "event: trace_event" in raw
    assert "event: run_terminal" in raw
    assert '"seq": 2' in raw or '"seq":2' in raw
    assert '"seq": 1' not in raw and '"seq":1' not in raw


def test_cli_ui_loopback_guard(monkeypatch: Any, tmp_path: Path) -> None:
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("0.0.0.0") is False

    runner = CliRunner()
    refused = runner.invoke(cli_app, ["ui", "--host", "0.0.0.0"])
    assert refused.exit_code == 2
    assert "non-loopback" in refused.output

    called: dict[str, Any] = {}

    def fake_run(asgi_app: Any, *, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    ok = runner.invoke(
        cli_app,
        ["ui", "--port", "8123", "--db", str(tmp_path / "hypora.sqlite")],
    )
    assert ok.exit_code == 0
    assert called == {"host": "127.0.0.1", "port": 8123}
