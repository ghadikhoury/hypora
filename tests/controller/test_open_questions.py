"""S4.2b: a refused proposal's open questions are stored screened; nothing else changes.

Scripted multi-agent runs on the breast-cancer CSV. No network or Docker.
"""

import json

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.controller.methodology import (
    OPEN_QUESTIONS_REFUSAL,
    OPEN_QUESTIONS_REFUSED,
    open_question_notes,
)
from hypora.core.llm.providers.fake import json_result
from hypora.core.schemas import ApprovalPolicy
from hypora.core.schemas.events import ErrorEvent, EventType
from hypora.storage.events import EventFilters, EventLog
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti

pytestmark = pytest.mark.anyio
C_PROMPT = "You are Hypora's data and implementation specialist"
FAIR = "Is the mean radius measured before the diagnosis is made?"
CODE = "Should I run this?\n```\nimport os\n```"


class QuestioningC(ScriptedMulti):
    """C's DataContract answers carry the given open questions, one list per answer."""

    def __init__(self, run_id: str, questions: list[list[str]]) -> None:
        super().__init__(run_id)
        self.questions = list(questions)

    async def generate(self, request):
        result = await super().generate(request)
        if request.output_schema_name == "DataContract" and self.questions:
            payload = json.loads(result.text)
            payload["open_questions"] = self.questions.pop(0)
            return json_result(payload)
        return result


def _row_dump(dataset) -> str:
    """One whole dataset row on one line: the data firewall withholds it."""
    row = pd.read_csv(dataset).iloc[0]
    return "Is this row an outlier: " + ", ".join(str(value) for value in row) + "?"


async def _run(make_controller, run_config, dataset, run_id, questions, policy=None):
    """A scripted multi run; `policy` overrides the fixture's `auto_within_contract`. Under
    `interactive` a user is present, so open questions left after C's correction still stop the
    run (S4.2c changes only runs with no user)."""
    script = QuestioningC(run_id, questions)
    controller = make_controller(script, worker_factory=NoWorker)
    update = {"agent_mode": AgentMode.MULTI, "planning": Toggle.OFF}
    if policy is not None:
        update["approval_policy"] = policy
    flags = run_config.flags.model_copy(update=update)
    controller.run_config = run_config.model_copy(update={"flags": flags})
    result = await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id=run_id))
    return script, result


def _refusals(database, run_id) -> list[ErrorEvent]:
    events = EventLog(database).read(run_id, EventFilters(types=(EventType.ERROR,)))
    return [
        e for e in events if isinstance(e, ErrorEvent) and e.payload.code == OPEN_QUESTIONS_REFUSED
    ]


def _not_c(script) -> list[str]:
    """Every text any role other than C read in this run."""
    return [
        message.content
        for request in script.requests
        if C_PROMPT not in request.messages[0].content
        for message in request.messages
    ]


async def test_a_mix_is_stored_screened_and_the_run_stops_as_before(
    make_controller, run_config, dataset, database
):
    dump = _row_dump(dataset)
    script, result = await _run(
        make_controller,
        run_config,
        dataset,
        "run_questions_stop",
        [[FAIR, dump, CODE], [FAIR]],
        policy=ApprovalPolicy.INTERACTIVE,
    )
    refusals = _refusals(database, "run_questions_stop")
    assert [e.payload.details for e in refusals] == [
        {
            "proposal": "initial",
            "attempt": 1,
            "open_questions": [FAIR],
            "open_questions_withheld": 2,
        },
        {
            "proposal": "initial",
            "attempt": 2,
            "open_questions": [FAIR],
            "open_questions_withheld": 0,
        },
    ]
    assert [e.payload.retryable for e in refusals] == [True, False]
    assert all(e.payload.message.endswith(OPEN_QUESTIONS_REFUSAL) for e in refusals)
    # Not a controller error: the graders' safe-stop and fault evidence read the run as before.
    assert all(e.payload.component == "methodology_validator" for e in refusals)
    stored = json.dumps(
        [e.model_dump(mode="json") for e in EventLog(database).read("run_questions_stop")]
    )
    assert "outlier" not in stored and "import os" not in stored
    # The run flow is today's: one correction, then the same stop before any contract.
    proposals = [r for r in script.requests if r.output_schema_name == "DataContract"]
    assert len(proposals) == 2 and not result.verified
    assert any(
        f"CONTRACT stage failed: MethodologyError: {OPEN_QUESTIONS_REFUSAL}" in note
        for note in result.notes
    )
    # Nothing new reaches A or the run notes; the run record gets the stored questions only.
    assert not any(FAIR in text or "outlier" in text for text in _not_c(script))
    assert not any(FAIR in note for note in result.notes)
    notes = open_question_notes(EventLog(database).read("run_questions_stop"))
    assert len(notes) == 2 and "(2 withheld)" in notes[0] and json.dumps([FAIR]) in notes[0]
    assert "outlier" not in "".join(notes)


async def test_a_corrected_refusal_is_recorded_and_the_run_continues(
    make_controller, run_config, dataset, database
):
    script, result = await _run(
        make_controller, run_config, dataset, "run_questions_fixed", [[FAIR]]
    )
    refusals = _refusals(database, "run_questions_fixed")
    assert len(refusals) == 1
    assert refusals[0].payload.details["open_questions"] == [FAIR]
    assert refusals[0].payload.details["open_questions_withheld"] == 0
    assert not any("CONTRACT stage failed" in note for note in result.notes)
    assert not any(FAIR in text for text in _not_c(script))
    assert not any(FAIR in note for note in result.notes)


async def test_only_failing_questions_are_withheld_as_a_count(
    make_controller, run_config, dataset, database
):
    _, _ = await _run(
        make_controller,
        run_config,
        dataset,
        "run_questions_withheld",
        [[_row_dump(dataset)], [CODE]],
        policy=ApprovalPolicy.INTERACTIVE,
    )
    details = [e.payload.details for e in _refusals(database, "run_questions_withheld")]
    assert [(d["open_questions"], d["open_questions_withheld"]) for d in details] == [
        ([], 1),
        ([], 1),
    ]


async def test_a_clean_run_records_no_refusal(make_controller, run_config, dataset, database):
    await _run(make_controller, run_config, dataset, "run_questions_none", [])
    assert _refusals(database, "run_questions_none") == []
    assert open_question_notes(EventLog(database).read("run_questions_none")) == []


async def test_a_row_split_across_lines_is_withheld(make_controller, run_config, dataset, database):
    """Each line holds one value and passes alone; joined (as the eval note shows it) the line
    is a whole row, so the question is screened in its joined form too and withheld."""
    row = pd.read_csv(dataset).iloc[0]
    split = "Are these outliers:\n" + "\n".join(str(value) for value in row)
    await _run(
        make_controller,
        run_config,
        dataset,
        "run_questions_split",
        [[split], [split]],
        policy=ApprovalPolicy.INTERACTIVE,
    )
    details = [e.payload.details for e in _refusals(database, "run_questions_split")]
    assert [(d["open_questions"], d["open_questions_withheld"]) for d in details] == [
        ([], 1),
        ([], 1),
    ]
    notes = open_question_notes(EventLog(database).read("run_questions_split"))
    assert not any("outliers" in note for note in notes)
