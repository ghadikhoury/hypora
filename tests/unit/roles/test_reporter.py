"""Reporter v1: every number comes from stored metrics; the narrative writes none (D-019 4a)."""

from typing import Any

import pytest
from pydantic import ValidationError

from hypora.core.llm.providers.fake import json_result
from hypora.core.roles.reporter import (
    CheckLine,
    FinalModel,
    MetricSummary,
    Report,
    ReportNarrative,
    TrialSummary,
    fill_placeholders,
    render_markdown,
    write_narrative,
)
from hypora.core.schemas import MetricDirection, MetricName, TaskType
from tests.unit.roles.helpers import RecordingGuard, adapter

GOOD: dict[str, Any] = {
    "summary": "The baseline reached {{test.roc_auc}} on the test split ({{validation.roc_auc}} "
    "in validation); its r2-free log_loss was {{test.log_loss}}.",
    "observations": ["Validation and test agree."],
    "limitations": ["Only one trial ran."],
}


def report(**changes: Any) -> Report:
    final = FinalModel(
        trial_id="trl_a",
        hypothesis="A logistic regression baseline",
        test_metrics={"roc_auc": 0.971234, "log_loss": None},
        validation_means={"roc_auc": 0.95, "log_loss": 0.2},
        verified=True,
        checks=[CheckLine(check="metric_recomputation", status="passed", message="ok")],
    )
    values: dict[str, Any] = {
        "run_id": "run_r",
        "outcome": "verified_model",
        "objective": "Predict y",
        "target": "y",
        "task_type": TaskType.BINARY_CLASSIFICATION,
        "primary_metric": MetricName.ROC_AUC,
        "direction": MetricDirection.MAXIMIZE,
        "secondary_metrics": [MetricName.LOG_LOSS],
        "split": "stratified, 20% test, stratified_kfold_3",
        "features": 30,
        "excluded_features": [],
        "leakage_screen": "No leakage candidates flagged among 30 screened features",
        "trials": [
            TrialSummary(
                trial_id="trl_a",
                kind="baseline",
                hypothesis="A logistic regression baseline",
                status="verified",
                validation=MetricSummary(mean=0.95, std=0.01, folds=3),
            )
        ],
        "final": final,
        "sources": ["result:trl_a/test"],
    }
    return Report.model_validate(values | changes)


@pytest.mark.parametrize(
    "text",
    ["It reached 0.97 on test.", "Trial 2 was best.", "{{test.not_a_metric}} looks fine."],
)
def test_the_narrative_may_not_write_numbers_of_its_own(text: str) -> None:
    with pytest.raises(ValidationError):
        ReportNarrative(summary=text)


def test_placeholders_and_metric_names_are_not_numbers() -> None:
    narrative = ReportNarrative.model_validate(GOOD)
    assert narrative.summary.startswith("The baseline")
    ReportNarrative(summary="r2 and f1_macro are reported; {{validation.r2}} is pending.")


def test_placeholders_are_filled_from_the_stored_metrics() -> None:
    filled = fill_placeholders(GOOD["summary"], report())
    assert "reached 0.9712 on the test split (0.9500 in validation)" in filled
    assert "log_loss was unavailable" in filled  # a stored null stays unavailable
    assert fill_placeholders("{{test.roc_auc}}", report(final=None)) == "unavailable"


def test_the_markdown_shows_stored_numbers_and_the_filled_narrative() -> None:
    narrated = report(narrative=ReportNarrative.model_validate(GOOD))
    markdown = render_markdown(narrated)
    assert "| roc_auc | 0.9712 | 0.9500 |" in markdown
    assert "| log_loss | unavailable | 0.2000 |" in markdown
    assert "Verified: yes" in markdown and "{{" not in markdown
    assert "| `trl_a` | baseline | verified | 0.9500 ± 0.0100 (3) |" in markdown


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_write_narrative_keeps_a_valid_narrative() -> None:
    llm, provider = adapter([json_result(GOOD)])
    narrated = await write_narrative(llm, RecordingGuard(), report())
    assert narrated.narrative == ReportNarrative.model_validate(GOOD)
    [request] = provider.requests
    assert "0.971234" in request.messages[-1].content  # the facts it narrates


@pytest.mark.anyio
async def test_a_narrative_that_keeps_writing_numbers_is_left_out() -> None:
    bad: dict[str, object] = {"summary": "It scored 0.97."}
    llm, _ = adapter([json_result(bad), json_result(bad)])  # first answer, then the retry
    guard = RecordingGuard()
    narrated = await write_narrative(llm, guard, report())
    assert narrated.narrative is None and guard.failed_calls == 1
    assert (
        narrated.narrative_note
        == "no narrative: the Reporter's output failed (StructuredOutputError)"
    )
    assert "_no narrative" in render_markdown(narrated)


@pytest.mark.anyio
async def test_a_run_limit_leaves_the_report_without_a_narrative() -> None:
    llm, provider = adapter([])
    narrated = await write_narrative(llm, RecordingGuard(refuse_model_after=0), report())
    assert narrated.narrative is None and provider.requests == []
    assert narrated.narrative_note == "no narrative: max_steps reached (0 of 0)"
