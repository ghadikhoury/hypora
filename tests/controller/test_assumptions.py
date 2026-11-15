"""S4.2c: with no user present, C's open questions become recorded conservative assumptions.

Scripted multi-agent runs on a small synthetic credit table (unit-test data only) whose columns
match the D-052 probe's questions. No network or Docker.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.methodology import (
    ASSUMPTION_PREFIX,
    ASSUMPTION_REASON,
    OPEN_QUESTIONS_REFUSAL,
    MethodologyError,
    assume_conservatively,
    columns_named,
    keep_assumed_exclusions,
)
from hypora.core.schemas import ApprovalPolicy, DataContract, TaskType
from hypora.core.schemas.common import EscalationType
from hypora.core.schemas.contracts import ExcludedFeature, SplitSpec, SplitStrategy
from hypora.core.schemas.events import EscalationEvent, EventType
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.records import load_experiment_contract
from tests.controller.test_open_questions import C_PROMPT, _not_c, _refusals, _run

pytestmark = pytest.mark.anyio

# The questions C asked in the approved D-052 probe (run_5f0514ea0c20, dev_target_leakage).
TIMING = (
    "Can the feature owner confirm that collections_referrals_12m reflects information "
    "available at application time?"
)
POLICY = (
    "Do applicable underwriting policies require excluding age_years, personal_status_sex, "
    "foreign_worker, or any associated proxy features?"
)
DRIFT = (
    "Does the intended deployment population differ materially over time or include repeated "
    "applicants?"
)
CODE = "Should I drop this?\n```\nimport os  # duration_months\n```"
SENSITIVE = ["age_years", "personal_status_sex", "foreign_worker"]


@pytest.fixture
def credit(tmp_path: Path) -> Path:
    """A synthetic credit table (unit tests only, PLAN.md §0.4) with the probe's column names."""
    rng = np.random.default_rng(7)
    rows = 240
    frame = pd.DataFrame(
        {
            "duration_months": rng.integers(6, 60, rows),
            "credit_amount": rng.normal(3000, 900, rows).round(2),
            "age_years": rng.integers(19, 75, rows),
            "personal_status_sex": rng.choice(["male_single", "female_married"], rows),
            "foreign_worker": rng.choice(["yes", "no"], rows),
            "collections_referrals_12m": rng.integers(0, 4, rows),
            "target": rng.integers(0, 2, rows),
        }
    )
    path = tmp_path / "data" / "credit.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


def _assumptions(database, run_id) -> list[EscalationEvent]:
    events = EventLog(database).read(run_id, EventFilters(types=(EventType.ESCALATION,)))
    return [
        e
        for e in events
        if isinstance(e, EscalationEvent) and e.payload.summary.startswith(ASSUMPTION_PREFIX)
    ]


def _c_corrections(script) -> list[str]:
    """The correction requests C received after a refused DataContract."""
    return [
        request.messages[-1].content
        for request in script.requests
        if request.output_schema_name == "DataContract"
        and C_PROMPT in request.messages[0].content
        and "The platform refused that DataContract" in request.messages[-1].content
    ]


async def test_no_user_run_continues_with_assumptions_in_the_report(
    make_controller, run_config, credit, database
):
    script, result = await _run(
        make_controller,
        run_config,
        credit,
        "run_assume",
        [[TIMING, POLICY, DRIFT], [TIMING, POLICY, DRIFT]],
    )
    # The run did not stop at CONTRACT: it went on to Phase B and wrote a full report.
    assert not any("CONTRACT stage failed" in note for note in result.notes)
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    # C got its one correction, told that no user can answer; only that refusal is recorded.
    assert [e.payload.details["attempt"] for e in _refusals(database, "run_assume")] == [1]
    corrections = _c_corrections(script)
    assert len(corrections) == 1 and "No user is present in this run" in corrections[0]
    # One escalation event per assumption, with the screened question and its safe answer.
    events = _assumptions(database, "run_assume")
    assert [e.payload.escalation_type for e in events] == [EscalationType.DATA_PROBLEM] * 3
    summaries = [e.payload.summary for e in events]
    assert (
        TIMING in summaries[0]
        and "collections_referrals_12m treated as not available" in (summaries[0])
    )
    assert POLICY in summaries[1] and all(name in summaries[1] for name in SENSITIVE)
    assert DRIFT in summaries[2] and "no feature changed" in summaries[2]
    assert len(events[0].payload.evidence_refs) == 1 and len(events[1].payload.evidence_refs) == 3
    assert events[2].payload.evidence_refs == []
    # The report lists exactly those assumptions (code-written limitations).
    assert [line for line in report["limitations"] if line.startswith(ASSUMPTION_PREFIX)] == (
        summaries
    )
    assert "collections_referrals_12m" in report["excluded_features"]
    counted = "No user is present: C's 3 open question"
    assert any(note.startswith(counted) for note in report["notes"])


async def test_a_doubtful_timing_column_is_never_kept(
    make_controller, run_config, credit, database
):
    await _run(make_controller, run_config, credit, "run_assume_timing", [[TIMING], [TIMING]])
    data = load_experiment_contract(database, "run_assume_timing").data_contract
    assert "collections_referrals_12m" not in data.allowed_features
    (excluded,) = [f for f in data.excluded_features if f.name == "collections_referrals_12m"]
    assert excluded.reason == ASSUMPTION_REASON
    assert str(excluded.evidence).startswith("assumption:")
    assert data.open_questions == ()
    # The other columns are untouched: the assumption changes only what the question names.
    assert set(data.allowed_features) == {
        "duration_months",
        "credit_amount",
        "age_years",
        "personal_status_sex",
        "foreign_worker",
    }


async def test_interactive_run_still_stops_to_ask(make_controller, run_config, credit, database):
    script, result = await _run(
        make_controller,
        run_config,
        credit,
        "run_assume_interactive",
        [[TIMING, DRIFT], [TIMING, DRIFT]],
        policy=ApprovalPolicy.INTERACTIVE,
    )
    assert not result.verified
    assert any(
        f"CONTRACT stage failed: MethodologyError: {OPEN_QUESTIONS_REFUSAL}" in note
        for note in result.notes
    )
    assert _assumptions(database, "run_assume_interactive") == []
    assert len(_refusals(database, "run_assume_interactive")) == 2
    corrections = _c_corrections(script)
    assert len(corrections) == 1 and "No user is present" not in corrections[0]
    assert not any(note.startswith("No user is present") for note in result.notes)


async def test_a_withheld_assumption_is_a_count_only_and_still_conservative(
    make_controller, run_config, credit, database
):
    amounts = pd.read_csv(credit)["credit_amount"].head(12)
    dump = "Are these amounts outliers: " + ", ".join(str(value) for value in amounts) + "?"
    script, result = await _run(
        make_controller,
        run_config,
        credit,
        "run_assume_withheld",
        [[DRIFT], [dump, CODE, DRIFT]],
    )
    events = _assumptions(database, "run_assume_withheld")
    summaries = [e.payload.summary for e in events]
    assert len(summaries) == 3
    withheld = [s for s in summaries if "withheld by the data firewall screen" in s]
    assert len(withheld) == 2 and DRIFT in summaries[2]
    # The code question named duration_months: withheld, yet the column is still excluded.
    assert "duration_months treated as not available" in withheld[1]
    data = load_experiment_contract(database, "run_assume_withheld").data_contract
    assert "duration_months" not in data.allowed_features
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert any("(2 withheld by the data firewall screen)" in note for note in report["notes"])
    # Nothing unscreened is stored, reported or shown to A.
    stored = json.dumps(
        [e.model_dump(mode="json") for e in EventLog(database).read("run_assume_withheld")]
    )
    texts = [
        stored,
        result.report_json.read_text(encoding="utf-8"),
        result.report_md.read_text(encoding="utf-8"),
        *_not_c(script),
    ]
    assert not any("outlier" in text or "import os" in text for text in texts)


async def test_screened_assumptions_are_the_only_question_text_a_sees(
    make_controller, run_config, credit, database
):
    script, _ = await _run(
        make_controller, run_config, credit, "run_assume_a", [[TIMING], [TIMING]]
    )
    seen = _not_c(script)
    # A reviews C's proposal with the question already answered: no open question in it.
    reviews = [text for text in seen if "C proposal: " in text]
    assert reviews and all(TIMING not in text for text in reviews)
    assert all('"open_questions":[]' in text for text in reviews)


# --- platform functions ---------------------------------------------------------------------


def _contract(allowed, excluded=(), questions=()) -> DataContract:
    return DataContract(
        data_contract_version=1,
        target="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        allowed_features=tuple(allowed),
        excluded_features=tuple(excluded),
        split=SplitSpec(
            strategy=SplitStrategy.STRATIFIED, test_fraction=0.2, validation="stratified_kfold_3"
        ),
        open_questions=tuple(questions),
    )


def test_columns_named_ignores_case_and_punctuation_but_needs_whole_words():
    columns = ["age", "collections_referrals_12m", "mean radius", "foreign_worker"]
    assert columns_named("Is the `Collections referrals 12M` field late?", columns) == [
        "collections_referrals_12m"
    ]
    assert columns_named("Is the average AGE known?", columns) == ["age"]
    assert columns_named("What is the average income?", columns) == []
    assert columns_named("Is Mean-Radius measured first?", columns) == ["mean radius"]


def test_assume_conservatively_excludes_only_named_allowed_columns():
    user = ExcludedFeature(name="age_years", reason="user's choice", evidence="task_spec:v1")
    proposal = _contract(
        ["duration_months", "foreign_worker", "collections_referrals_12m"],
        [user],
        [TIMING, POLICY, "Is target recorded after approval?"],
    )
    converted, evidence, assumptions = assume_conservatively(proposal, tag="initial")
    assert converted.allowed_features == ("duration_months",)
    assert converted.open_questions == ()
    assert [a.columns for a in assumptions] == [
        ("collections_referrals_12m",),
        ("foreign_worker",),  # age_years was already excluded by the user
        (),  # the target is never an allowed feature
    ]
    assert evidence == {
        "assumption:initial_1": ("assumption", "collections_referrals_12m"),
        "assumption:initial_2": ("assumption", "foreign_worker"),
    }
    assert converted.excluded_features[0] == user
    assert {f.evidence for f in converted.excluded_features[1:]} == set(evidence)


def test_assume_conservatively_refuses_when_no_feature_would_remain():
    with pytest.raises(MethodologyError, match="no allowed feature remains"):
        assume_conservatively(_contract(["foreign_worker"], questions=[POLICY]), tag="initial")


def test_a_later_proposal_cannot_take_an_assumed_exclusion_back():
    evidence = {"assumption:initial_1": ("assumption", "collections_referrals_12m")}
    later = _contract(["duration_months", "collections_referrals_12m"])
    kept = keep_assumed_exclusions(later, evidence)
    assert kept.allowed_features == ("duration_months",)
    assert kept.excluded_features == (
        ExcludedFeature(
            name="collections_referrals_12m",
            reason=ASSUMPTION_REASON,
            evidence="assumption:initial_1",
        ),
    )
    unchanged = _contract(["duration_months"])
    assert keep_assumed_exclusions(unchanged, evidence) is unchanged
