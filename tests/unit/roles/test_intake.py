"""Minimal intake: the LLM drafts, code settles every field it can check (PLAN.md S1.5)."""

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.llm import FakeProvider
from hypora.core.llm.errors import LLMError, ProviderError, ProviderErrorKind
from hypora.core.llm.providers.fake import json_result
from hypora.core.roles.intake import IntakeError, IntakeForm, run_intake
from hypora.core.schemas import MetricName, TaskSpecification, TaskType
from tests.unit.roles.helpers import RecordingGuard, adapter

pytestmark = pytest.mark.anyio
SECRET = "patient_4471_note"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 40, 50, 60] * 15,
            "note": [SECRET, "b", "c", "d"] * 15,
            "outcome": [0, 1, 0, 1] * 15,
            "price": [float(i) for i in range(60)],
        }
    )


async def intake(
    frame: pd.DataFrame, form: IntakeForm, draft: dict[str, object]
) -> tuple[TaskSpecification, FakeProvider, RecordingGuard]:
    llm, provider = adapter([json_result(draft)])
    guard = RecordingGuard()
    spec = await run_intake(
        llm, guard, run_id="run_i", form=form, frame=frame, dataset_ref="dataset:ds_main"
    )
    return spec, provider, guard


async def test_the_draft_fills_what_the_form_leaves_open(frame: pd.DataFrame) -> None:
    spec, provider, guard = await intake(
        frame,
        IntakeForm(goal="Will the patient have the outcome?"),
        {"objective": "Predict the outcome", "target": "outcome", "primary_metric": "accuracy"},
    )
    assert (spec.target, spec.task_type, spec.primary_metric) == (
        "outcome",
        TaskType.BINARY_CLASSIFICATION,
        MetricName.ACCURACY,
    )
    assert spec.from_form == [] and guard.model_calls == 1
    # The model saw the column names and dtypes, never a value.
    [request] = provider.requests
    sent = "\n".join(message.content for message in request.messages)
    assert "- note (" in sent and SECRET not in sent


async def test_form_fields_override_the_draft_and_code_corrects_the_rest(
    frame: pd.DataFrame,
) -> None:
    spec, _, _ = await intake(
        frame,
        IntakeForm(
            goal="Predict the price",
            target="price",
            primary_metric=MetricName.ROC_AUC,  # not a regression metric: the default wins
            forbidden_columns=("note", "price", "ghost"),
        ),
        {
            "objective": "Predict the price",
            "target": "outcome",
            "task_type": "binary_classification",
        },
    )
    assert spec.target == "price" and spec.task_type is TaskType.REGRESSION  # 60 distinct values
    assert spec.primary_metric is MetricName.MAE
    assert spec.forbidden_columns == ["note"]  # the target and unknown columns are dropped
    assert spec.open_questions == ["ignored unknown forbidden columns: ['ghost']"]
    assert spec.from_form == ["target", "primary_metric", "forbidden_columns"]


async def test_a_numeric_target_with_few_values_is_classification(frame: pd.DataFrame) -> None:
    spec, _, _ = await intake(
        frame,
        IntakeForm(goal="x"),
        {"objective": "x", "target": "age", "task_type": "binary_classification"},
    )
    assert spec.task_type is TaskType.MULTICLASS_CLASSIFICATION
    assert spec.primary_metric is MetricName.F1_MACRO


async def test_no_usable_target_is_an_intake_error(frame: pd.DataFrame) -> None:
    with pytest.raises(IntakeError, match="names no target column"):
        await intake(frame, IntakeForm(goal="x"), {"objective": "x"})
    with pytest.raises(IntakeError, match="'nope' is not one of"):
        await intake(frame, IntakeForm(goal="x", target="nope"), {"objective": "x"})


async def test_a_failed_model_call_is_still_accounted(frame: pd.DataFrame) -> None:
    llm, _ = adapter([ProviderError(ProviderErrorKind.AUTHENTICATION, "bad key")])
    guard = RecordingGuard()
    with pytest.raises(LLMError):
        await run_intake(
            llm, guard, run_id="run_i", form=IntakeForm(goal="x"), frame=frame, dataset_ref="d"
        )
    assert guard.failed_calls == 1
