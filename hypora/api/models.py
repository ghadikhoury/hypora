"""Pydantic v2 request and response models for the Hypora HTTP API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hypora.core.schemas import ApprovalPolicy, MetricName, TaskType


class HealthResponse(BaseModel):
    """Service health and environment readiness."""

    status: str = "ok"
    service: str = "hypora-api"
    db_path: str
    db_exists: bool
    runs_root: str
    git_sha: str | None = None
    web_dist_ready: bool = False


class DatasetItem(BaseModel):
    """One dataset snapshot from `data/snapshots/manifest.json`."""

    name: str
    source: str
    dataset_id: int
    title: str
    version: str
    license: str
    target: str | None = None
    rows: int
    columns: int
    file: str
    path: str
    exists: bool


class DatasetListResponse(BaseModel):
    """Recorded snapshots available for 1-click run launch."""

    snapshots_dir: str
    manifest_exists: bool
    datasets: list[DatasetItem] = Field(default_factory=list)


class RunListItem(BaseModel):
    """Summary of one run for the runs table."""

    run_id: str
    status: str
    stage: str | None = None
    dataset_label: str
    goal: str | None = None
    model_profile: str | None = None
    approval_policy: str
    budget_usd: float | None = None
    spent_usd: float
    budget_compute_s: float | None = None
    spent_compute_s: float
    pending_approval: bool = False
    final_verified: bool | None = None
    final_metric: str | None = None
    final_score: float | None = None
    source_db: str | None = None
    read_only: bool = False
    created_at: datetime
    updated_at: datetime


class RunListResponse(BaseModel):
    """List of runs across primary and optional read-only state DBs."""

    runs: list[RunListItem] = Field(default_factory=list)


class RunCreateRequest(BaseModel):
    """Payload for `POST /api/runs`."""

    model_config = ConfigDict(extra="forbid")

    data: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    target: str | None = None
    task_type: TaskType | None = None
    metric: MetricName | None = None
    forbid: list[str] = Field(default_factory=list)
    budget_usd: float | None = Field(default=None, ge=0)
    max_trials: int | None = Field(default=None, ge=0)
    max_wall_clock_s: float | None = Field(default=None, ge=1)
    approve: ApprovalPolicy | None = None
    advice_only: bool = False


class RunCreateResponse(BaseModel):
    """Response for `POST /api/runs` (202 Accepted)."""

    run_id: str
    status: str
    approval_policy: str


class PendingApprovalDetail(BaseModel):
    """Pending interactive or policy plan approval details."""

    approval_id: str
    artifact_hash: str
    hash_prefix: str
    policy: str
    summary: str
    plan_version: int | None = None
    artifact_ref: str | None = None
    plan: dict[str, Any] | None = None
    scope: dict[str, Any] | None = None


class TrialMetricSplit(BaseModel):
    """Stored metrics and verification verdict for one trial split."""

    split: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)


class TrialDetail(BaseModel):
    """One trial and its validation/test results."""

    trial_id: str
    kind: str
    hypothesis: str
    status: str
    val_result: TrialMetricSplit | None = None
    test_result: TrialMetricSplit | None = None


class JobDetail(BaseModel):
    """One sandbox job associated with a trial."""

    job_id: str
    trial_id: str
    status: str


class ProfileSummary(BaseModel):
    """Redacted dataset profile aggregates (never raw rows)."""

    record_id: str
    dataset_ref: str
    fingerprint: str
    modality: str
    tabular: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    timeseries: dict[str, Any] | None = None


class RunDetailResponse(BaseModel):
    """Full run inspection payload for `GET /api/runs/{run_id}`."""

    run_id: str
    status: str
    stage: str | None = None
    dataset_label: str
    approval_policy: str
    read_only: bool = False
    source_db: str | None = None
    budget_usd: float | None = None
    spent_usd: float
    budget_compute_s: float | None = None
    spent_compute_s: float
    total_tokens: int = 0
    models_used: list[str] = Field(default_factory=list)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    task_spec: dict[str, Any] | None = None
    profile: ProfileSummary | None = None
    experiment_contract: dict[str, Any] | None = None
    data_contract: dict[str, Any] | None = None
    pending_approval: PendingApprovalDetail | None = None
    trials: list[TrialDetail] = Field(default_factory=list)
    jobs: list[JobDetail] = Field(default_factory=list)
    report_json: dict[str, Any] | None = None
    report_md: str | None = None
    cancel_requested: bool = False


class TraceEventItem(BaseModel):
    """One formatted trace event with its redacted payload."""

    seq: int
    offset_s: float
    created_at: datetime
    role: str
    type: str
    duration_s: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    detail: str
    trial_id: str | None = None
    job_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceEventsResponse(BaseModel):
    """Filtered trace timeline for `GET /api/runs/{run_id}/events`."""

    run_id: str
    status: str
    events: list[TraceEventItem] = Field(default_factory=list)
    total_events: int
    summary: str


class ApprovalRequest(BaseModel):
    """Body for `POST /api/runs/{run_id}/approve`."""

    model_config = ConfigDict(extra="forbid")

    approve: bool
    hash_prefix: str = Field(min_length=1)
    reason: str | None = None
    user: str | None = None


class ApprovalResponse(BaseModel):
    """Response after writing an atomic plan decision."""

    run_id: str
    approval_id: str
    decision: str
    artifact_hash: str
    decision_file: str


class CancelResponse(BaseModel):
    """Response for `POST /api/runs/{run_id}/cancel`."""

    run_id: str
    status: str
    cancel_requested: bool
    cancel_file: str


class EvalSummaryResponse(BaseModel):
    """Committed S5.3/S5.4 evaluation artifacts for the benchmark dashboard."""

    consolidated_summary: dict[str, Any] | None = None
    evidence_index: dict[str, Any] | None = None
    f6_evidence: dict[str, Any] | None = None
    f7_evidence: dict[str, Any] | None = None
