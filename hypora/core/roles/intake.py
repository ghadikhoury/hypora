"""Minimal intake (PLAN.md S1.5): one structured-output call drafts the TaskSpecification.

The LLM reads the user's goal and the dataset's column names and types (never rows) and drafts
the task. Code then decides everything it can check: structured form fields win over the draft
(spec §5.2: hard constraints come from structured fields), the target must be a real column,
the task type is corrected from the target's values, and the metric must suit the task. The full
Intake role, with clarifying questions, is S4.1.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message
from hypora.core.roles.base import CallGuard
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import (
    ActorRole,
    ApprovalPolicy,
    Budget,
    Family,
    MetricName,
    ModelTier,
    RequestedAction,
    StrictModel,
    TaskSpecification,
    TaskType,
)
from hypora.core.schemas.common import NonEmptyStr

CLASSIFICATION_METRICS = frozenset(
    {
        MetricName.ACCURACY,
        MetricName.BALANCED_ACCURACY,
        MetricName.PRECISION_MACRO,
        MetricName.RECALL_MACRO,
        MetricName.F1_MACRO,
        MetricName.ROC_AUC,
        MetricName.PR_AUC,
        MetricName.LOG_LOSS,
    }
)
BINARY_ONLY_METRICS = frozenset(
    {MetricName.PRECISION, MetricName.RECALL, MetricName.F1, MetricName.BRIER_SCORE}
)
REGRESSION_METRICS = frozenset({MetricName.MAE, MetricName.RMSE, MetricName.R2})
DEFAULT_METRIC = {
    TaskType.BINARY_CLASSIFICATION: MetricName.ROC_AUC,
    TaskType.MULTICLASS_CLASSIFICATION: MetricName.F1_MACRO,
    TaskType.REGRESSION: MetricName.MAE,
}
"""D-019: the primary metric when the user names none."""

MAX_CLASSES = 50
"""A target with more distinct values than this, if numeric, is treated as regression."""


class IntakeError(ValueError):
    """The goal and data do not determine a task (no usable target)."""


class IntakeDraft(StrictModel):
    """What the Intake LLM returns (its structured output)."""

    objective: NonEmptyStr
    target: str | None = None
    task_type: (
        Literal["binary_classification", "multiclass_classification", "regression"] | None
    ) = None
    primary_metric: MetricName | None = None
    forbidden_columns: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class IntakeForm:
    """Structured fields from the CLI or form: they override the LLM's draft."""

    goal: str
    target: str | None = None
    task_type: TaskType | None = None
    primary_metric: MetricName | None = None
    forbidden_columns: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)


def _columns(frame: pd.DataFrame) -> str:
    return "\n".join(f"- {name} ({dtype})" for name, dtype in frame.dtypes.astype(str).items())


def _task_type(target: pd.Series, hint: TaskType | None) -> TaskType:
    """The task type the target's values allow (code decides; the hint breaks ties)."""
    values = target.dropna()
    distinct = values.nunique()
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if hint is TaskType.REGRESSION and numeric:
        return TaskType.REGRESSION
    if numeric and distinct > MAX_CLASSES:
        return TaskType.REGRESSION
    if distinct < 2:
        raise IntakeError("the target has fewer than two distinct values")
    return TaskType.BINARY_CLASSIFICATION if distinct == 2 else TaskType.MULTICLASS_CLASSIFICATION


def _metric(task: TaskType, wanted: MetricName | None) -> MetricName:
    allowed = (
        REGRESSION_METRICS
        if task is TaskType.REGRESSION
        else CLASSIFICATION_METRICS
        | (BINARY_ONLY_METRICS if task is TaskType.BINARY_CLASSIFICATION else frozenset())
    )
    return wanted if wanted in allowed else DEFAULT_METRIC[task]


async def run_intake(
    adapter: LLMAdapter,
    guard: CallGuard,
    *,
    run_id: str,
    form: IntakeForm,
    frame: pd.DataFrame,
    dataset_ref: str,
) -> TaskSpecification:
    """Draft the task with one model call, then settle every checkable field in code."""
    prompt = load_prompt("intake")
    guard.before_model_call()
    try:
        response = await adapter.complete(
            [
                Message.system(prompt.text),
                Message.user(f"Goal:\n{form.goal}\n\nColumns:\n{_columns(frame)}"),
            ],
            tier=ModelTier.CHEAP,
            context=CallContext(run_id=run_id, actor_role=ActorRole.INTAKE, prompt_ref=prompt.ref),
            output_schema=IntakeDraft,
        )
    except LLMError as exc:
        guard.after_model_error(exc)
        raise
    guard.after_model_call(response)
    draft = response.parsed if isinstance(response.parsed, IntakeDraft) else None
    if draft is None:
        raise IntakeError("the intake model returned no draft")

    columns: Sequence[str] = [str(c) for c in frame.columns]
    from_form = []
    target = form.target or draft.target
    if form.target:
        from_form.append("target")
    if target not in columns:
        raise IntakeError(
            f"no target column: {target!r} is not one of the dataset's columns"
            if target
            else "the goal names no target column; pass one explicitly"
        )
    hint = form.task_type or (TaskType(draft.task_type) if draft.task_type else None)
    if form.task_type:
        from_form.append("task_type")
    task_type = _task_type(frame[target], hint)
    metric = _metric(task_type, form.primary_metric or draft.primary_metric)
    if form.primary_metric:
        from_form.append("primary_metric")
    forbidden = form.forbidden_columns or tuple(draft.forbidden_columns)
    if form.forbidden_columns:
        from_form.append("forbidden_columns")
    unknown = sorted(set(forbidden) - set(columns))
    return TaskSpecification(
        run_id=run_id,
        version=1,
        objective=draft.objective,
        requested_actions=[RequestedAction.TRAIN],
        dataset_ref=dataset_ref,
        family_hint=Family.TABULAR,
        task_type=task_type,
        target=target,
        primary_metric=metric,
        forbidden_columns=sorted(set(forbidden) & set(columns) - {target}),
        budget=form.budget,
        approval_policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        from_form=from_form,
        open_questions=[f"ignored unknown forbidden columns: {unknown}"] if unknown else [],
    )
