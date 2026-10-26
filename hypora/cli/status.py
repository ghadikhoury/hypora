"""`hypora status`: one run's state from its stored records, or a list of recent runs (S1.6).

Everything shown is read from the state database, the run's folder, and the snapshot manifest;
nothing is recomputed. The dataset line matches the run's dataset fingerprint (the SHA-256 of the
bytes the run read) against the snapshot manifest, so a run on a snapshot names its real source.
"""

import contextlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.schemas import EventType, RunStatus
from hypora.core.schemas.events import ModelCallPayload, StageTransitionPayload
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
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
from hypora.storage.snapshots import SnapshotManifest

TERMINAL: Final = frozenset({RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED})


def _utc(value: Any) -> str:
    return f"{value:%Y-%m-%d %H:%M:%S} UTC"


def _counts(values: Sequence[str]) -> str:
    counter = Counter(values)
    return ", ".join(f"{name} {count}" for name, count in sorted(counter.items()))


def dataset_label(database: Database, run_id: str, manifest: SnapshotManifest) -> str:
    """The run's dataset: its snapshot (source, version, license) or its fingerprint."""
    try:
        _, profile = load_dataset_profile(database, run_id)
    except RecordNotFound:
        return "-"
    entry = manifest.by_sha256(profile.fingerprint)
    if entry is None:
        return f"not a recorded snapshot (SHA-256 {profile.fingerprint[:12]})"
    return (
        f"{entry.name}: {entry.source.value} {entry.dataset_id} '{entry.title}' "
        f"({entry.version}), {entry.license}"
    )


@dataclass(frozen=True)
class FinalModel:
    trial_id: str
    verified: bool
    metric: str | None
    value: float | None


def final_model(database: Database, run_id: str, metric: str | None) -> FinalModel | None:
    """The trial whose test split was scored (the controller's final job), if any."""
    for trial in reversed(list_trials(database, run_id)):
        result = load_result(database, trial.trial_id, "test")
        if result is None:
            continue
        scores = result.metrics.get("metrics", {})
        value = scores.get(metric) if metric and isinstance(scores, dict) else None
        return FinalModel(
            trial_id=trial.trial_id,
            verified=bool(result.verification.get("passed")),
            metric=metric,
            value=float(value) if isinstance(value, int | float) else None,
        )
    return None


def _spend(run: StoredRun, tokens: int) -> str:
    usd = f"${run.spent_usd:.4f}" + (f" of ${run.budget_usd:.2f}" if run.budget_usd else "")
    compute = f"{run.spent_compute_s:.1f} s" + (
        f" of {run.budget_compute_s:.0f} s" if run.budget_compute_s else ""
    )
    return f"{usd} at list price | sandbox compute {compute} | {tokens:,} LLM tokens"


def run_status_lines(
    database: Database, runs_root: Path, run_id: str, manifest: SnapshotManifest
) -> list[str]:
    """The `hypora status <run_id>` report; raises RecordNotFound for an unknown run."""
    run = load_run(database, run_id)
    events = EventLog(database).read(
        run_id, EventFilters(types=(EventType.STAGE_TRANSITION, EventType.MODEL_CALL))
    )
    stages = [e.payload for e in events if isinstance(e.payload, StageTransitionPayload)]
    calls = [e.payload for e in events if isinstance(e.payload, ModelCallPayload)]
    stage = stages[-1].stage.value if stages and stages[-1].stage else "-"
    models = sorted({f"{c.provider.value}/{c.model_id} ({c.profile.value})" for c in calls})
    lines = [
        f"Run       {run.run_id}",
        f"Status    {run.status.value} (stage {stage}); created {_utc(run.created_at)}, "
        f"updated {_utc(run.updated_at)}",
        f"Dataset   {dataset_label(database, run_id, manifest)}",
    ]
    metric = None
    try:
        spec = load_task_spec(database, run_id)
        task = spec.task_type.value if spec.task_type else "-"
        metric = spec.primary_metric.value if spec.primary_metric else None
        lines.append(
            f"Task      {task} | target {spec.target or '-'} | metric {metric or '-'}: "
            f"{spec.objective}"
        )
    except RecordNotFound:
        lines.append("Task      - (intake did not finish)")
    with contextlib.suppress(RecordNotFound):  # the contract's metric is the one scored
        metric = load_experiment_contract(database, run_id).primary_metric.value
    lines.append(f"Models    {', '.join(models) or '-'}")
    tokens = sum(c.input_tokens + c.output_tokens for c in calls)
    lines.append(f"Spend     {_spend(run, tokens)}")
    trials = list_trials(database, run_id)
    lines.append(
        f"Trials    {len(trials)}" + (f": {_counts([t.status for t in trials])}" if trials else "")
    )
    jobs = run_jobs(database, run_id)
    lines.append(
        f"Jobs      {len(jobs)}" + (f": {_counts([j.status.value for j in jobs])}" if jobs else "")
    )
    final = final_model(database, run_id, metric)
    if final is not None:
        score = f" | test {final.metric} {final.value:.4f}" if final.value is not None else ""
        verdict = "verified" if final.verified else "NOT verified"
        lines.append(f"Final     {final.trial_id}: {verdict}{score}")
    else:
        lines.append("Final     - (no trial reached the test split)")
    report = runs_root / run_id / "report.md"
    lines.append(f"Report    {report if report.is_file() else '- (not written yet)'}")
    if (runs_root / run_id / CANCEL_FILE).exists():
        state = "honoured" if run.status is RunStatus.CANCELLED else "requested"
        lines.append(f"Cancel    {state} ({runs_root / run_id / CANCEL_FILE})")
    lines.append(f"Trace     hypora trace {run_id}")
    return lines


def runs_table_lines(database: Database, manifest: SnapshotManifest, limit: int) -> list[str]:
    """The `hypora status` list: the newest runs with their status, dataset, and final verdict."""
    runs = list_runs(database, limit)
    if not runs:
        return ['No runs yet. Start one with `hypora run --data <file> --goal "..."`.']
    rows = [("RUN", "STATUS", "CREATED (UTC)", "FINAL", "DATASET")]
    for run in runs:
        final = final_model(database, run.run_id, None)
        verdict = "-" if final is None else "verified" if final.verified else "not verified"
        rows.append(
            (
                run.run_id,
                run.status.value,
                f"{run.created_at:%Y-%m-%d %H:%M:%S}",
                verdict,
                dataset_label(database, run.run_id, manifest).split(":")[0],
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row[:-1], widths, strict=True)) + "  " + row[-1]
        for row in rows
    ]


__all__ = ["TERMINAL", "dataset_label", "final_model", "run_status_lines", "runs_table_lines"]
