"""Reporter v1 (spec §5.2 Report; PLAN.md S1.5; D-019): the run report, from DB records only.

Every number in the report comes from stored, platform-computed records: the task spec, the
contract, the profile, each trial's validation results, and the final candidate's test results
and verification verdict. `build_report` assembles those facts deterministically.

The Reporter LLM adds a short narrative (`write_narrative`) but may not write numbers: it refers
to the final model's scores only through placeholders (`{{test.roc_auc}}`,
`{{validation.roc_auc}}`), which `render_markdown` fills from the stored metrics. A validator
rejects any other digit, and the adapter retries once with the errors; a narrative that still
fails is left out, and the report says so.
"""

import json
import math
import re
import statistics
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message
from hypora.core.roles.base import CallGuard, RunLimitReached
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import (
    ActorRole,
    MetricDirection,
    MetricName,
    ModelTier,
    StrictModel,
    TaskType,
)
from hypora.core.schemas.common import NonEmptyStr
from hypora.profiling.tabular import leakage_screen_summary
from hypora.storage.database import Database
from hypora.storage.records import (
    RecordNotFound,
    list_trials,
    load_dataset_profile,
    load_experiment_contract,
    load_result,
    load_task_spec,
)

PLACEHOLDER: Final = re.compile(r"\{\{(test|validation)\.([a-z0-9_]+)\}\}")
_METRIC_WORDS: Final = re.compile(
    r"\b(" + "|".join(sorted((m.value for m in MetricName), key=len, reverse=True)) + r")\b"
)
_KNOWN_METRICS: Final = frozenset(m.value for m in MetricName)


class ReportNarrative(StrictModel):
    """The Reporter LLM's output: prose with metric placeholders and no digits of its own."""

    summary: NonEmptyStr = Field(max_length=1500)
    observations: list[str] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _no_numbers_of_its_own(self) -> "ReportNarrative":
        for text in (self.summary, *self.observations, *self.limitations):
            for _, metric in PLACEHOLDER.findall(text):
                if metric not in _KNOWN_METRICS:
                    raise ValueError(f"unknown metric in placeholder: {metric!r}")
            bare = _METRIC_WORDS.sub("", PLACEHOLDER.sub("", text))
            if re.search(r"\d", bare):
                raise ValueError(
                    "the narrative must not contain digits: refer to scores only with "
                    "placeholders such as {{test.roc_auc}} and to trials by description"
                )
        return self


class MetricSummary(StrictModel):
    mean: float | None
    std: float | None
    folds: int


class TrialSummary(StrictModel):
    trial_id: str
    kind: str
    hypothesis: str
    status: str
    validation: MetricSummary | None = None  # of the primary metric


class CheckLine(StrictModel):
    check: str
    status: str
    message: str


class FinalModel(StrictModel):
    trial_id: str
    hypothesis: str
    test_metrics: dict[str, float | None]
    validation_means: dict[str, float | None]
    verified: bool
    checks: list[CheckLine]


class Report(StrictModel):
    """The run report's facts. Numbers only from stored records; text from the controller."""

    run_id: str
    outcome: Literal["verified_model", "no_verified_model"]
    objective: str
    target: str
    task_type: TaskType
    primary_metric: MetricName
    direction: MetricDirection
    secondary_metrics: list[MetricName]
    split: str
    features: int
    excluded_features: list[str]
    leakage_screen: str
    trials: list[TrialSummary]
    final: FinalModel | None = None
    notes: list[str] = Field(default_factory=list)
    narrative: ReportNarrative | None = None
    narrative_note: str | None = None
    sources: list[str] = Field(default_factory=list)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _fold_values(scores: dict[str, Any], metric: MetricName) -> list[float]:
    values = [
        _finite(fold.get("metrics", {}).get(metric.value)) for fold in scores.get("folds", [])
    ]
    return [v for v in values if v is not None]


def metric_summary(scores: dict[str, Any], metric: MetricName) -> MetricSummary:
    """Mean and spread of `metric` over the folds of stored validation scores (metrics.json)."""
    values = _fold_values(scores, metric)
    return MetricSummary(
        mean=statistics.fmean(values) if values else None,
        std=statistics.pstdev(values) if values else None,
        folds=len(values),
    )


def build_report(
    database: Database,
    run_id: str,
    *,
    final_trial_id: str | None,
    notes: list[str],
) -> Report:
    """Assemble the report's facts from the run's stored records (no model call)."""
    spec = load_task_spec(database, run_id)
    contract = load_experiment_contract(database, run_id)
    profile_id, profile = load_dataset_profile(database, run_id)
    metric = contract.primary_metric
    trials = []
    for trial in list_trials(database, run_id):
        result = load_result(database, trial.trial_id, "validation")
        trials.append(
            TrialSummary(
                trial_id=trial.trial_id,
                kind=trial.kind,
                hypothesis=trial.hypothesis[:300],
                status=trial.status,
                validation=metric_summary(result.metrics, metric) if result else None,
            )
        )
    final = None
    sources = [f"task_spec:{run_id}.v{spec.version}", f"profile:{profile_id}"]
    sources.append(f"experiment_contract:{run_id}.v{contract.version}")
    if final_trial_id is not None:
        test = load_result(database, final_trial_id, "test")
        validation = load_result(database, final_trial_id, "validation")
        if test is not None:
            wanted = [metric, *contract.secondary_metrics]
            checks = test.verification.get("checks", [])
            final = FinalModel(
                trial_id=final_trial_id,
                hypothesis=next((t.hypothesis for t in trials if t.trial_id == final_trial_id), ""),
                test_metrics={
                    m.value: _finite(test.metrics.get("metrics", {}).get(m.value)) for m in wanted
                },
                validation_means={
                    m.value: metric_summary(validation.metrics, m).mean if validation else None
                    for m in wanted
                },
                verified=bool(test.verification.get("passed")),
                checks=[
                    CheckLine(
                        check=str(c.get("check")),
                        status=str(c.get("status")),
                        message=str(c.get("message", ""))[:200],
                    )
                    for c in checks
                ],
            )
            sources.append(f"result:{final_trial_id}/test")
    data = contract.data_contract
    split = data.split
    tabular = profile.tabular
    return Report(
        run_id=run_id,
        outcome="verified_model" if final is not None and final.verified else "no_verified_model",
        objective=spec.objective,
        target=contract.target,
        task_type=contract.task_type,
        primary_metric=metric,
        direction=metric.direction,
        secondary_metrics=list(contract.secondary_metrics),
        split=f"{split.strategy.value}, {split.test_fraction:.0%} test, {split.validation}",
        features=len(data.allowed_features),
        excluded_features=[f.name for f in data.excluded_features],
        leakage_screen=leakage_screen_summary(tabular) if tabular else "leakage not screened",
        trials=trials,
        final=final,
        notes=notes,
        sources=sources,
    )


async def write_narrative(adapter: LLMAdapter, guard: CallGuard, report: Report) -> Report:
    """Add the Reporter LLM's narrative; on any failure the report keeps no narrative."""
    prompt = load_prompt("reporter")
    facts = report.model_dump(mode="json", exclude={"narrative", "narrative_note", "sources"})
    try:
        guard.before_model_call()
        response = await adapter.complete(
            [Message.system(prompt.text), Message.user(json.dumps(facts, indent=1))],
            tier=ModelTier.CHEAP,
            context=CallContext(
                run_id=report.run_id, actor_role=ActorRole.REPORTER, prompt_ref=prompt.ref
            ),
            output_schema=ReportNarrative,
        )
        guard.after_model_call(response)
    except RunLimitReached as reached:
        return report.model_copy(update={"narrative_note": f"no narrative: {reached}"})
    except LLMError as exc:
        guard.after_model_error(exc)
        note = f"no narrative: the Reporter's output failed ({type(exc).__name__})"
        return report.model_copy(update={"narrative_note": note})
    narrative = response.parsed if isinstance(response.parsed, ReportNarrative) else None
    return report.model_copy(update={"narrative": narrative})


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def fill_placeholders(text: str, report: Report) -> str:
    """Replace {{test.<metric>}} / {{validation.<metric>}} with the final model's stored scores."""

    def value(match: re.Match[str]) -> str:
        if report.final is None:
            return "unavailable"
        table = report.final.test_metrics if match[1] == "test" else report.final.validation_means
        return _number(table.get(match[2]))

    return PLACEHOLDER.sub(value, text)


def render_markdown(report: Report) -> str:
    """The human-readable report. Every number is formatted from `report`'s stored values."""
    lines = [f"# Hypora run report: {report.run_id}", ""]
    outcome = "a verified model" if report.outcome == "verified_model" else "no verified model"
    lines += [f"**Outcome:** {outcome}.", ""]
    lines += [
        "## Task",
        f"- Objective: {report.objective}",
        f"- Target: `{report.target}` ({report.task_type.value})",
        f"- Primary metric: `{report.primary_metric.value}` ({report.direction.value})",
        "",
    ]
    if report.narrative is not None:
        n = report.narrative
        lines += ["## Summary", fill_placeholders(n.summary, report), ""]
        if n.observations:
            lines += [
                "**Observations**",
                *(f"- {fill_placeholders(o, report)}" for o in n.observations),
            ]
            lines.append("")
        if n.limitations:
            lines += [
                "**Limitations**",
                *(f"- {fill_placeholders(x, report)}" for x in n.limitations),
            ]
            lines.append("")
        lines += [
            "_Written by the Reporter; every number above is filled in from stored metrics._",
            "",
        ]
    elif report.narrative_note:
        lines += [f"_{report.narrative_note}_", ""]
    lines += ["## Final model"]
    if report.final is None:
        lines += ["No final model: no trial completed validation and verification.", ""]
    else:
        final = report.final
        lines += [
            f"- Trial `{final.trial_id}`: {final.hypothesis}",
            f"- Verified: {'yes' if final.verified else 'NO'}",
            "",
            "| Metric | Test (once) | Validation (mean) |",
            "|---|---|---|",
        ]
        for name, test_value in final.test_metrics.items():
            lines.append(
                f"| {name} | {_number(test_value)} | {_number(final.validation_means.get(name))} |"
            )
        lines += ["", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c.check} | {c.status} | {c.message} |" for c in final.checks]
        lines.append("")
    lines += [
        "## Trials",
        f"| Trial | Kind | Status | Validation {report.primary_metric.value} "
        "(mean ± std, folds) | Hypothesis |",
        "|---|---|---|---|---|",
    ]
    for trial in report.trials:
        v = trial.validation
        score = (
            f"{_number(v.mean)} ± {_number(v.std)} ({v.folds})"
            if v is not None and v.folds
            else "unavailable"
        )
        lines.append(
            f"| `{trial.trial_id}` | {trial.kind} | {trial.status} | {score} | {trial.hypothesis} |"
        )
    lines += [
        "",
        "## Data",
        f"- Split: {report.split}",
        f"- Features used: {report.features}; "
        f"excluded: {', '.join(report.excluded_features) or 'none'}",
        f"- Leakage screen: {report.leakage_screen}",
        "",
    ]
    if report.notes:
        lines += ["## Run notes", *(f"- {note}" for note in report.notes), ""]
    lines += ["## Sources", *(f"- `{source}`" for source in report.sources), ""]
    return "\n".join(lines)


class EarlyStopReport(StrictModel):
    """The report of a run that stopped before it had a contract (nothing to measure yet)."""

    run_id: str
    outcome: Literal["no_verified_model"] = "no_verified_model"
    notes: list[str]


def render_early_stop(report: EarlyStopReport) -> str:
    lines = [f"# Hypora run report: {report.run_id}", "", "**Outcome:** no verified model.", ""]
    lines += ["The run stopped before its experiment contract was approved.", ""]
    lines += ["## Run notes", *(f"- {note}" for note in report.notes), ""]
    return "\n".join(lines)


__all__ = [
    "EarlyStopReport",
    "FinalModel",
    "RecordNotFound",
    "Report",
    "ReportNarrative",
    "build_report",
    "fill_placeholders",
    "metric_summary",
    "render_early_stop",
    "render_markdown",
    "write_narrative",
]
