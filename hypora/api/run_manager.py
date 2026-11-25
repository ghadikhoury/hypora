"""Background run lifecycle, state DB discovery, and safe read/control helpers."""

import asyncio
import contextlib
import inspect
import json
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import anyio
from sqlalchemy import select

from hypora.api.models import (
    ApprovalRequest,
    ApprovalResponse,
    CancelResponse,
    DatasetItem,
    DatasetListResponse,
    JobDetail,
    PendingApprovalDetail,
    ProfileSummary,
    RunCreateRequest,
    RunCreateResponse,
    RunDetailResponse,
    RunListItem,
    TraceEventItem,
    TraceEventsResponse,
    TrialDetail,
    TrialMetricSplit,
)
from hypora.cli.approve import ApprovalRefused, check_hash_prefix, plan_file, record_decision
from hypora.cli.run import DATASET_SUFFIXES, RunOptions, execute
from hypora.cli.status import TERMINAL, dataset_label, final_model
from hypora.cli.trace import parse_filters, summary, trace_rows
from hypora.core.config import ConfigError, load_settings
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.approval_gate import PendingApproval, pending_approval
from hypora.core.controller.controller import RunResult, new_run_id
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.schemas import ApprovalPolicy, EventType
from hypora.core.schemas.events import ModelCallPayload, StageTransitionPayload
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.models import DataContractRow
from hypora.storage.records import (
    RecordNotFound,
    StoredRun,
    list_runs,
    list_trials,
    load_dataset_profile,
    load_experiment_contract,
    load_result,
    load_run,
    load_task_spec,
    run_jobs,
)
from hypora.storage.snapshots import MANIFEST_FILE, SNAPSHOTS_DIR, SnapshotManifest, load_manifest

RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

RunLauncher = Callable[[RunOptions], Awaitable[RunResult] | RunResult | None]


class ApiError(Exception):
    """Domain error mapped to an HTTP status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ResolvedRun:
    """A run located in either the primary DB or a read-only extra state DB."""

    db_path: Path
    database: Database
    stored: StoredRun
    run_folder: Path | None
    read_only: bool
    source_db: str


def validate_run_id(run_id: str) -> str:
    """Reject empty or path-traversal run IDs."""
    if not RUN_ID_PATTERN.match(run_id) or ".." in run_id:
        raise ApiError(400, f"invalid run_id: {run_id!r}")
    return run_id


def validate_dataset_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve and validate a .csv or .parquet dataset path without traversal."""
    cleaned = raw_path.strip()
    if not cleaned or "\x00" in cleaned:
        raise ApiError(400, "dataset path must be non-empty")
    candidate = Path(cleaned)
    if ".." in candidate.parts:
        raise ApiError(400, "dataset path must not contain '..' traversal segments")
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    if resolved.suffix.lower() not in DATASET_SUFFIXES:
        raise ApiError(400, f"{cleaned}: dataset must be a .csv or .parquet file")
    if not resolved.is_file():
        raise ApiError(400, f"dataset file not found: {cleaned}")
    return resolved


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (OSError, ValueError):
        return None


def _read_text_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_data_contract(database: Database, run_id: str) -> dict[str, Any] | None:
    statement = (
        select(DataContractRow.document)
        .where(DataContractRow.run_id == run_id)
        .order_by(DataContractRow.version.desc())
        .limit(1)
    )
    with database.engine.connect() as connection:
        document = connection.scalar(statement)
    return dict(document) if isinstance(document, dict) else None


class RunManager:
    """Coordinates live background runs and read-only inspection across state DBs."""

    def __init__(
        self,
        *,
        db_path: Path,
        runs_root: Path,
        extra_state_dirs: Sequence[Path] = (),
        snapshots_dir: Path = SNAPSHOTS_DIR,
        repo_root: Path | None = None,
        run_launcher: RunLauncher | None = None,
    ) -> None:
        self.db_path = db_path.resolve()
        self.runs_root = runs_root.resolve()
        self.extra_state_dirs = tuple(p.resolve() for p in extra_state_dirs)
        self.snapshots_dir = snapshots_dir.resolve()
        self.repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        self._launcher = run_launcher
        self._lock = threading.Lock()
        self._active_threads: dict[str, threading.Thread] = {}
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._launch_errors: dict[str, str] = {}

    def load_snapshot_manifest(self) -> SnapshotManifest:
        """Load the snapshot manifest, returning an empty one if missing."""
        try:
            return load_manifest(self.snapshots_dir)
        except Exception:
            return SnapshotManifest()

    def list_datasets(self) -> DatasetListResponse:
        """List recorded datasets from `data/snapshots/manifest.json`."""
        manifest_path = self.snapshots_dir / MANIFEST_FILE
        manifest_exists = manifest_path.is_file()
        manifest = self.load_snapshot_manifest()
        items = []
        for entry in manifest.snapshots:
            file_path = (self.snapshots_dir / entry.file).resolve()
            items.append(
                DatasetItem(
                    name=entry.name,
                    source=entry.source.value,
                    dataset_id=entry.dataset_id,
                    title=entry.title,
                    version=entry.version,
                    license=entry.license,
                    target=entry.target,
                    rows=entry.rows,
                    columns=entry.columns,
                    file=entry.file,
                    path=str(file_path),
                    exists=file_path.is_file(),
                )
            )
        return DatasetListResponse(
            snapshots_dir=str(self.snapshots_dir),
            manifest_exists=manifest_exists,
            datasets=items,
        )

    def _discover_extra_dbs(self) -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = {self.db_path}
        for directory in self.extra_state_dirs:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.rglob("*.sqlite")):
                resolved = candidate.resolve()
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
        return found

    def _run_folder_for(self, sqlite_path: Path, run_id: str, *, is_primary: bool) -> Path | None:
        if is_primary:
            folder = self.runs_root / run_id
            return folder if folder.is_dir() else None
        candidates = []
        if sqlite_path.parent.name == "state":
            candidates.append(sqlite_path.parent.parent / "runs" / run_id)
            candidates.append(sqlite_path.parent.parent / run_id)
        candidates.append(sqlite_path.parent / run_id)
        candidates.append(self.runs_root / run_id)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def _source_label(self, sqlite_path: Path, *, is_primary: bool) -> str:
        if is_primary:
            return "primary"
        with contextlib.suppress(ValueError):
            return str(sqlite_path.relative_to(self.repo_root))
        return sqlite_path.name

    def resolve_run(self, run_id: str) -> ResolvedRun:
        """Find `run_id` in the primary DB or any discovered extra state DB."""
        validate_run_id(run_id)
        candidates: list[tuple[Path, bool]] = []
        if self.db_path.is_file():
            candidates.append((self.db_path, True))
        for extra in self._discover_extra_dbs():
            candidates.append((extra, False))

        for sqlite_path, is_primary in candidates:
            database = Database(sqlite_path)
            try:
                stored = load_run(database, run_id)
                folder = self._run_folder_for(sqlite_path, run_id, is_primary=is_primary)
                return ResolvedRun(
                    db_path=sqlite_path,
                    database=database,
                    stored=stored,
                    run_folder=folder,
                    read_only=not is_primary,
                    source_db=self._source_label(sqlite_path, is_primary=is_primary),
                )
            except RecordNotFound:
                database.close()
            except Exception:
                database.close()
        raise ApiError(404, f"unknown run: {run_id}")

    def _summarize_run(
        self,
        database: Database,
        stored: StoredRun,
        manifest: SnapshotManifest,
        *,
        source_db: str,
        read_only: bool,
    ) -> RunListItem:
        events = EventLog(database).read(
            stored.run_id, EventFilters(types=(EventType.STAGE_TRANSITION,))
        )
        stages = [e.payload for e in events if isinstance(e.payload, StageTransitionPayload)]
        stage = stages[-1].stage.value if stages and stages[-1].stage else None

        goal: str | None = None
        metric: str | None = None
        try:
            spec = load_task_spec(database, stored.run_id)
            goal = spec.objective
            metric = spec.primary_metric.value if spec.primary_metric else None
        except RecordNotFound:
            pass

        with contextlib.suppress(RecordNotFound, ValueError):
            metric = load_experiment_contract(database, stored.run_id).primary_metric.value

        final = final_model(database, stored.run_id, metric)
        pending = False
        if stored.status not in TERMINAL:
            pending = pending_approval(database, stored.run_id) is not None

        profile_val = stored.model_config.get("model_profile") or stored.model_config.get("profile")
        return RunListItem(
            run_id=stored.run_id,
            status=stored.status.value,
            stage=stage,
            dataset_label=dataset_label(database, stored.run_id, manifest),
            goal=goal,
            model_profile=str(profile_val) if profile_val is not None else None,
            approval_policy=stored.approval_policy,
            budget_usd=float(stored.budget_usd) if stored.budget_usd is not None else None,
            spent_usd=float(stored.spent_usd),
            budget_compute_s=(
                float(stored.budget_compute_s) if stored.budget_compute_s is not None else None
            ),
            spent_compute_s=float(stored.spent_compute_s),
            pending_approval=pending,
            final_verified=final.verified if final is not None else None,
            final_metric=final.metric if final is not None else None,
            final_score=final.value if final is not None else None,
            source_db=source_db,
            read_only=read_only,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
        )

    def list_all_runs(self, limit: int = 100) -> list[RunListItem]:
        """List runs from the primary DB and any discovered extra state DBs."""
        manifest = self.load_snapshot_manifest()
        items: list[RunListItem] = []
        seen_ids: set[str] = set()

        sources: list[tuple[Path, bool]] = []
        if self.db_path.is_file():
            sources.append((self.db_path, True))
        sources.extend((p, False) for p in self._discover_extra_dbs())

        for sqlite_path, is_primary in sources:
            database = Database(sqlite_path)
            try:
                for stored in list_runs(database, limit=limit):
                    if stored.run_id in seen_ids:
                        continue
                    seen_ids.add(stored.run_id)
                    items.append(
                        self._summarize_run(
                            database,
                            stored,
                            manifest,
                            source_db=self._source_label(sqlite_path, is_primary=is_primary),
                            read_only=not is_primary,
                        )
                    )
            except Exception:
                continue
            finally:
                database.close()

        items.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
        return items[:limit]

    def _pending_detail(
        self, database: Database, run_folder: Path | None, pending: PendingApproval
    ) -> PendingApprovalDetail:
        plan_doc: dict[str, Any] | None = None
        scope_doc: dict[str, Any] | None = None
        if run_folder is not None:
            plan_path = plan_file(run_folder.parent, pending)
            if plan_path is not None:
                plan_doc = _read_json_file(plan_path)
            version = pending.plan_version
            if version is not None:
                scope_doc = _read_json_file(run_folder / f"approval_scope_v{version}.json")
        return PendingApprovalDetail(
            approval_id=pending.approval_id,
            artifact_hash=pending.artifact_hash,
            hash_prefix=pending.artifact_hash[:12],
            policy=pending.policy.value,
            summary=pending.summary,
            plan_version=pending.plan_version,
            artifact_ref=pending.artifact_ref,
            plan=plan_doc,
            scope=scope_doc,
        )

    def get_run_detail(self, run_id: str) -> RunDetailResponse:
        """Full detail for one run."""
        resolved = self.resolve_run(run_id)
        db = resolved.database
        try:
            stored = resolved.stored
            manifest = self.load_snapshot_manifest()
            events = EventLog(db).read(run_id)
            stages = [e.payload for e in events if isinstance(e.payload, StageTransitionPayload)]
            calls = [e.payload for e in events if isinstance(e.payload, ModelCallPayload)]
            stage = stages[-1].stage.value if stages and stages[-1].stage else None
            models_used = sorted(
                {f"{c.provider.value}/{c.model_id} ({c.profile.value})" for c in calls}
            )
            total_tokens = sum(c.input_tokens + c.output_tokens for c in calls)

            task_spec: dict[str, Any] | None = None
            with contextlib.suppress(RecordNotFound):
                task_spec = load_task_spec(db, run_id).model_dump(mode="json")

            profile_summary: ProfileSummary | None = None
            with contextlib.suppress(RecordNotFound):
                record_id, profile = load_dataset_profile(db, run_id)
                profile_summary = ProfileSummary(
                    record_id=record_id,
                    dataset_ref=profile.dataset_ref,
                    fingerprint=profile.fingerprint,
                    modality=profile.modality.value,
                    tabular=profile.tabular.model_dump(mode="json") if profile.tabular else None,
                    vision=profile.vision.model_dump(mode="json") if profile.vision else None,
                    timeseries=(
                        profile.timeseries.model_dump(mode="json") if profile.timeseries else None
                    ),
                )

            experiment_contract: dict[str, Any] | None = None
            with contextlib.suppress(RecordNotFound, ValueError):
                experiment_contract = load_experiment_contract(db, run_id).model_dump(mode="json")

            data_contract = _load_data_contract(db, run_id)

            pending_detail: PendingApprovalDetail | None = None
            if stored.status not in TERMINAL:
                pending = pending_approval(db, run_id)
                if pending is not None:
                    pending_detail = self._pending_detail(db, resolved.run_folder, pending)

            trials_out: list[TrialDetail] = []
            for trial in list_trials(db, run_id):
                val_res = load_result(db, trial.trial_id, "val")
                test_res = load_result(db, trial.trial_id, "test")
                trials_out.append(
                    TrialDetail(
                        trial_id=trial.trial_id,
                        kind=trial.kind,
                        hypothesis=trial.hypothesis,
                        status=trial.status,
                        val_result=(
                            TrialMetricSplit(
                                split=val_res.split,
                                metrics=val_res.metrics,
                                verification=val_res.verification,
                            )
                            if val_res is not None
                            else None
                        ),
                        test_result=(
                            TrialMetricSplit(
                                split=test_res.split,
                                metrics=test_res.metrics,
                                verification=test_res.verification,
                            )
                            if test_res is not None
                            else None
                        ),
                    )
                )

            jobs_out = [
                JobDetail(job_id=j.job_id, trial_id=j.trial_id, status=j.status.value)
                for j in run_jobs(db, run_id)
            ]

            report_json = (
                _read_json_file(resolved.run_folder / "report.json")
                if resolved.run_folder is not None
                else None
            )
            report_md = (
                _read_text_file(resolved.run_folder / "report.md")
                if resolved.run_folder is not None
                else None
            )
            cancel_requested = (
                (resolved.run_folder / CANCEL_FILE).is_file()
                if resolved.run_folder is not None
                else False
            )

            return RunDetailResponse(
                run_id=stored.run_id,
                status=stored.status.value,
                stage=stage,
                dataset_label=dataset_label(db, run_id, manifest),
                approval_policy=stored.approval_policy,
                read_only=resolved.read_only,
                source_db=resolved.source_db,
                budget_usd=float(stored.budget_usd) if stored.budget_usd is not None else None,
                spent_usd=float(stored.spent_usd),
                budget_compute_s=(
                    float(stored.budget_compute_s) if stored.budget_compute_s is not None else None
                ),
                spent_compute_s=float(stored.spent_compute_s),
                total_tokens=total_tokens,
                models_used=models_used,
                model_config_snapshot=stored.model_config,
                created_at=stored.created_at,
                updated_at=stored.updated_at,
                task_spec=task_spec,
                profile=profile_summary,
                experiment_contract=experiment_contract,
                data_contract=data_contract,
                pending_approval=pending_detail,
                trials=trials_out,
                jobs=jobs_out,
                report_json=report_json,
                report_md=report_md,
                cancel_requested=cancel_requested,
            )
        finally:
            db.close()

    def get_run_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        role: str | None = None,
        trial: str | None = None,
        job: str | None = None,
        after_seq: int = 0,
        limit: int = 500,
    ) -> TraceEventsResponse:
        """Return formatted `TraceRow` items and redacted event payloads."""
        filter_specs: list[str] = []
        for key, raw in (("type", event_type), ("role", role), ("trial", trial), ("job", job)):
            if raw is not None and raw.strip():
                filter_specs.append(f"{key}={raw.strip()}")
        try:
            selection = parse_filters(filter_specs)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None

        resolved = self.resolve_run(run_id)
        db = resolved.database
        try:
            events = EventLog(db).read(run_id)
            rows = trace_rows(events)
            events_by_seq = {e.seq: e for e in events}
            filtered_rows = [row for row in rows if row.seq > after_seq and selection.keeps(row)][
                :limit
            ]
            items = []
            for row in filtered_rows:
                ev = events_by_seq[row.seq]
                items.append(
                    TraceEventItem(
                        seq=row.seq,
                        offset_s=row.offset_s,
                        created_at=ev.created_at,
                        role=row.role,
                        type=row.type,
                        duration_s=row.duration_s,
                        tokens_in=row.tokens[0] if row.tokens else None,
                        tokens_out=row.tokens[1] if row.tokens else None,
                        detail=row.detail,
                        trial_id=row.trial_id,
                        job_id=row.job_id,
                        payload=ev.payload.model_dump(mode="json"),
                    )
                )
            return TraceEventsResponse(
                run_id=run_id,
                status=resolved.stored.status.value,
                events=items,
                total_events=len(events),
                summary=summary(filtered_rows, events) if events else "No events recorded.",
            )
        finally:
            db.close()

    async def launch_run(self, req: RunCreateRequest) -> RunCreateResponse:
        """Validate inputs and launch a background run."""
        dataset = validate_dataset_path(req.data, self.repo_root)
        goal = req.goal.strip()
        if not goal:
            raise ApiError(400, "goal must not be empty")

        overrides = {"flags": {"approval_policy": req.approve.value}} if req.approve else {}
        try:
            config = RunConfig.build(load_settings(overrides=overrides))
        except ConfigError as exc:
            raise ApiError(400, str(exc)) from None

        run_id = new_run_id()
        options = RunOptions(
            data=dataset,
            goal=goal,
            db=self.db_path,
            runs_root=self.runs_root,
            target=req.target,
            task_type=req.task_type,
            metric=req.metric,
            forbid=tuple(req.forbid),
            budget_usd=req.budget_usd,
            max_trials=req.max_trials,
            max_wall_clock_s=req.max_wall_clock_s,
            approve=req.approve,
            run_id=run_id,
            quiet=True,
            advice_only=req.advice_only,
        )

        if self._launcher is not None:
            result = self._launcher(options)
            if inspect.isawaitable(result):
                task: asyncio.Task[Any] = asyncio.create_task(self._await_launcher(run_id, result))
                with self._lock:
                    self._active_tasks[run_id] = task
        else:
            thread = threading.Thread(
                target=self._run_thread,
                args=(options,),
                name=f"hypora-run-{run_id}",
                daemon=True,
            )
            with self._lock:
                self._active_threads[run_id] = thread
            thread.start()

        return RunCreateResponse(
            run_id=run_id,
            status="pending",
            approval_policy=config.flags.approval_policy.value,
        )

    async def _await_launcher(self, run_id: str, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception as exc:
            with self._lock:
                self._launch_errors[run_id] = str(exc)
        finally:
            with self._lock:
                self._active_tasks.pop(run_id, None)

    def _run_thread(self, options: RunOptions) -> None:
        run_id = options.run_id or ""
        try:
            anyio.run(self._execute_options, options)
        except Exception as exc:
            with self._lock:
                self._launch_errors[run_id] = str(exc)
        finally:
            with self._lock:
                self._active_threads.pop(run_id, None)

    @staticmethod
    async def _execute_options(options: RunOptions) -> RunResult:
        return await execute(options, out=lambda _line: None, terminal=False)

    def approve_run(self, run_id: str, req: ApprovalRequest) -> ApprovalResponse:
        """Validate pending hash prefix and write an atomic approval decision."""
        resolved = self.resolve_run(run_id)
        db = resolved.database
        try:
            if resolved.read_only:
                raise ApiError(409, f"run {run_id} is read-only ({resolved.source_db})")
            if resolved.stored.status in TERMINAL:
                raise ApiError(409, f"run {run_id} already ended {resolved.stored.status.value}")
            pending = pending_approval(db, run_id)
            if pending is None:
                raise ApiError(404, f"run {run_id} has no pending approval")
            if pending.policy is not ApprovalPolicy.INTERACTIVE:
                raise ApiError(
                    409,
                    f"run {run_id} is governed by {pending.policy.value}, not interactive approval",
                )
            if not req.approve and not (req.reason or "").strip():
                raise ApiError(400, "denial requires a non-empty reason")
            try:
                check_hash_prefix(pending, req.hash_prefix)
            except ApprovalRefused as exc:
                raise ApiError(409, str(exc)) from None
            self.runs_root.mkdir(parents=True, exist_ok=True)
            try:
                path = record_decision(
                    self.runs_root,
                    pending,
                    approve=req.approve,
                    reason=req.reason,
                    user=req.user,
                )
            except ApprovalRefused as exc:
                raise ApiError(409, str(exc)) from None
            return ApprovalResponse(
                run_id=run_id,
                approval_id=pending.approval_id,
                decision="approved" if req.approve else "denied",
                artifact_hash=pending.artifact_hash,
                decision_file=str(path),
            )
        finally:
            db.close()

    def cancel_run(self, run_id: str) -> CancelResponse:
        """Write `CANCEL_FILE` to request cancellation of an active run."""
        resolved = self.resolve_run(run_id)
        db = resolved.database
        try:
            if resolved.read_only:
                raise ApiError(409, f"run {run_id} is read-only ({resolved.source_db})")
            cancel_path = self.runs_root / run_id / CANCEL_FILE
            if resolved.stored.status in TERMINAL:
                return CancelResponse(
                    run_id=run_id,
                    status=resolved.stored.status.value,
                    cancel_requested=cancel_path.is_file(),
                    cancel_file=str(cancel_path),
                )
            cancel_path.parent.mkdir(parents=True, exist_ok=True)
            cancel_path.write_text("cancel requested via web API\n", encoding="utf-8")
            return CancelResponse(
                run_id=run_id,
                status=resolved.stored.status.value,
                cancel_requested=True,
                cancel_file=str(cancel_path),
            )
        finally:
            db.close()


__all__ = [
    "ApiError",
    "ResolvedRun",
    "RunLauncher",
    "RunManager",
    "validate_dataset_path",
    "validate_run_id",
]
