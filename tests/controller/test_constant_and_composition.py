"""D-060 end to end, scripted models only: the platform's constant reference and its rule, and
the joint target-composition test on A's questioned columns.

The trial runs use the scripted sandbox and an in-process worker (no Docker). Synthetic tables are
unit-test data (D-015)."""

import json
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import (
    CONSTANT_REFERENCE,
    CONSTANT_RULE,
    TARGET_COMPOSITION,
    RunRequest,
)
from hypora.core.llm.providers.fake import json_result
from hypora.storage.database import Database
from hypora.storage.records import list_trials, load_experiment_contract
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.test_request_context import DUMMY, PLATFORM_FACTS, EvidenceC, TrialsWithBest
from tests.controller.test_robustness import CONTINUE, PRE_LOCK, RECONCILE, _errors, _report

pytestmark = pytest.mark.anyio
HANG_GUARD = 300

# A real model whose probabilities are reversed: far worse than a constant on ROC-AUC.
INVERTED = '''"""A logistic regression with reversed probabilities (worse than a constant)."""

import numpy as np
from runtime import Predictions  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


def fit(train, ctx):
    """Fit on the training rows only."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train.X.select_dtypes("number"), train.y)
    return model


def predict(model, X):
    """Reverse the class probabilities."""
    probabilities = model.predict_proba(X.select_dtypes("number"))[:, ::-1]
    classes = tuple(model.classes_)
    values = np.asarray(classes)[probabilities.argmax(axis=1)]
    return Predictions(values, probabilities, classes)
'''


async def _run(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    run_id: str,
    script: Any,
    *,
    worker: Any = scripted_sandbox.ThreadWorker,
    goal: str = "Predict target",
    budget: Any = None,
    **robust: Any,
) -> Any:
    controller = make_controller(script, worker_factory=worker)
    robustness = run_config.flags.robustness.model_copy(update=robust)
    flags = run_config.flags.model_copy(
        update={
            "agent_mode": AgentMode.MULTI,
            "planning": Toggle.OFF,
            "max_c_workers": 0,
            "robustness": robustness,
        }
    )
    controller.run_config = run_config.model_copy(update={"flags": flags})
    with anyio.fail_after(HANG_GUARD):
        return await controller.run(
            RunRequest(dataset=dataset, goal=goal, run_id=run_id, budget=budget)
        )


def _final_candidate(notes: list[str]) -> str:
    (note,) = [n for n in notes if n.startswith("Final candidate ")]
    return note.split()[2].rstrip(":")


# --- the constant reference and its rule ---------------------------------------------------------


async def test_classification_shows_the_reference_but_never_substitutes_a_constant(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_rule"
    script = TrialsWithBest(run_id, pipeline_py=INVERTED)
    result = await _run(make_controller, run_config, dataset, run_id, script)

    # C saw the reference in its context and in the nomination check.
    experiment = [
        r
        for r in script.requests
        if r.output_schema_name is None and "You are Agent C" in r.messages[0].content
    ]
    assert "Platform constant reference (platform fact" in experiment[0].messages[1].content
    (final_request,) = [r for r in script.requests if r.output_schema_name == "FinalAnswer"]
    assert '"constant_reference": {"strategy": "prior"' in final_request.messages[-1].content
    (reference,) = _errors(database, run_id, CONSTANT_REFERENCE)
    assert reference.payload.details["outcome"] == "computed"
    assert reference.payload.details["mean"] == pytest.approx(0.5)

    assert "does not substitute a constant" in experiment[0].messages[1].content

    (rule,) = _errors(database, run_id, CONSTANT_RULE)
    details: Any = rule.payload.details
    assert details["outcome"] == "fired" and details["action"] == "shown_only_classification"
    assert all(not c["beats"] for c in details["comparisons"].values())
    assert details["chosen"] == details["nominated"]
    assert len(list_trials(database, run_id)) == 2  # no constant trial added
    assert not any("Platform constant-reference rule" in n for n in result.notes)
    assert _final_candidate(result.notes) == details["nominated"]
    check = json.loads(
        script.final_review.split("Trial evidence check: ", 1)[1].split(
            ". Candidate code (untrusted, screened): ", 1
        )[0]
    )
    assert "platform_candidate" not in check
    assert all(t["vs_constant_reference"]["beats_reference"] is False for t in check["trials"])


@pytest.fixture
def noisy(tmp_path: Path) -> Path:
    """A skewed regression target the features do not explain (median and mean differ)."""
    rng = np.random.default_rng(7)
    rows = 300
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows).round(3),
            "x2": rng.uniform(0, 10, rows).round(3),
            "y": rng.gamma(1.5, 10.0, rows).round(2),
        }
    )
    path = tmp_path / "data" / "noisy.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


# A featureless model that predicts the training maximum: a poor constant, far worse than the
# median on MAE, and not the reference's constant.
TRAINING_MAX = '''"""Predict the training maximum."""

import numpy as np
from runtime import Predictions  # type: ignore[import-not-found]


def fit(train, ctx):
    """Learn one number from the training labels."""
    return float(np.max(train.y))


def predict(model, X):
    """Predict that number for every row."""
    return Predictions(np.full(len(X), model))
'''
TRAINING_MEDIAN = TRAINING_MAX.replace("maximum", "median").replace("np.max(", "np.median(")


class RegressionTrials(TrialsWithBest):
    """TrialsWithBest on a regression task: the target is `y`, scored by MAE."""

    async def generate(self, request: Any) -> Any:
        if request.output_schema_name == "IntakeDraft":
            return json_result(
                {
                    "objective": "Predict y",
                    "target": "y",
                    "task_type": "regression",
                    "primary_metric": "mae",
                }
            )
        return await super().generate(request)


async def test_regression_no_trial_beats_the_constant_so_the_platform_validates_one(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    noisy: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline is the dummy template's mean (not the MAE reference's median) and the
    candidate a poor constant: the platform validates its own median trial and selects it."""
    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_rule"
    script = RegressionTrials(run_id, pipeline_py=DUMMY, candidate_py=TRAINING_MAX)
    result = await _run(make_controller, run_config, noisy, run_id, script)

    experiment = [
        r
        for r in script.requests
        if r.output_schema_name is None and "You are Agent C" in r.messages[0].content
    ]
    context = experiment[0].messages[1].content
    assert "Platform constant reference (platform fact" in context
    assert "only while a trial slot" in context
    (reference,) = _errors(database, run_id, CONSTANT_REFERENCE)
    assert reference.payload.details["strategy"] == "median"

    (rule,) = _errors(database, run_id, CONSTANT_RULE)
    details: Any = rule.payload.details
    assert details["outcome"] == "fired" and details["action"] == "constant_trial_validated"
    constant = details["constant_trial"]
    assert details["chosen"] == constant and details["nominated"] != constant
    # The mean dummy is not the median reference, so it was not taken as the constant.
    assert not any(c["matches"] for c in details["comparisons"].values())
    # The sandboxed median trial reproduces the in-process reference.
    assert details["constant_trial_mean"] == pytest.approx(details["reference_mean"], abs=1e-5)
    trials = {t.trial_id: t for t in list_trials(database, run_id)}
    assert len(trials) == 3 and trials[constant].kind == "baseline"
    # A reviewed the constant trial's code; the check names it and still labels C's nomination.
    assert "DummyRegressor" in script.final_review
    check = json.loads(
        script.final_review.split("Trial evidence check: ", 1)[1].split(
            ". Candidate code (untrusted, screened): ", 1
        )[0]
    )
    assert check["platform_candidate"] == constant
    nominated = [t["trial_id"] for t in check["trials"] if t.get("nominated_by_c")]
    assert nominated == [details["nominated"]]
    assert any("Platform constant-reference rule" in n for n in result.notes)
    assert _final_candidate(result.notes) == constant


async def test_regression_an_existing_median_trial_is_selected_without_a_new_trial(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    noisy: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_existing"
    script = RegressionTrials(run_id, pipeline_py=TRAINING_MEDIAN, candidate_py=TRAINING_MAX)
    result = await _run(make_controller, run_config, noisy, run_id, script)
    (rule,) = _errors(database, run_id, CONSTANT_RULE)
    details: Any = rule.payload.details
    assert details["outcome"] == "fired" and details["action"] == "existing_constant_trial"
    assert details["chosen"] != details["nominated"]
    assert details["comparisons"][details["chosen"]]["matches"] is True
    assert len(list_trials(database, run_id)) == 2  # no trial added
    assert _final_candidate(result.notes) == details["chosen"]


async def test_regression_without_a_trial_slot_the_nomination_stands(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    noisy: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_no_room"
    script = RegressionTrials(run_id, pipeline_py=DUMMY, candidate_py=TRAINING_MAX)
    budget = run_config.budgets.run.model_copy(update={"max_trials": 2})
    result = await _run(make_controller, run_config, noisy, run_id, script, budget=budget)
    (rule,) = _errors(database, run_id, CONSTANT_RULE)
    details: Any = rule.payload.details
    assert details["outcome"] == "fired" and details["action"] == "no_room"
    assert "trial limit" in str(details["reason"])
    assert details["chosen"] == details["nominated"]
    assert len(list_trials(database, run_id)) == 2
    assert _final_candidate(result.notes) == details["nominated"]


async def test_a_trial_that_beats_the_constant_keeps_cs_nomination(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.controller.scripted import LINEAR

    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_beaten"
    script = TrialsWithBest(run_id, pipeline_py=LINEAR)
    result = await _run(make_controller, run_config, dataset, run_id, script)
    (rule,) = _errors(database, run_id, CONSTANT_RULE)
    details: Any = rule.payload.details
    assert details["outcome"] == "beaten" and details["action"] == "none"
    assert details["chosen"] == details["nominated"] and details["beaten_by"]
    assert len(list_trials(database, run_id)) == 2
    assert _final_candidate(result.notes) == details["nominated"]


async def test_with_the_flag_off_there_is_no_reference_and_no_rule(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_constant_off"
    script = TrialsWithBest(run_id, pipeline_py=INVERTED)
    result = await _run(
        make_controller, run_config, dataset, run_id, script, constant_reference=Toggle.OFF
    )
    assert _errors(database, run_id, CONSTANT_REFERENCE) == []
    assert _errors(database, run_id, CONSTANT_RULE) == []
    assert len(list_trials(database, run_id)) == 2
    assert '"constant_reference": {' not in script.final_review
    assert 'vs_constant_reference": {' not in script.final_review
    assert not any("Platform constant-reference rule" in n for n in result.notes)


# --- the joint target-composition test -----------------------------------------------------------


@pytest.fixture
def summed(tmp_path: Path) -> Path:
    """A count that is exactly the sum of two columns, one of which alone scores low."""
    rng = np.random.default_rng(3)
    rows = 600
    casual = rng.gamma(2.0, 30.0, rows).round()
    registered = (rng.gamma(5.0, 80.0, rows) + 2 * casual).round()
    frame = pd.DataFrame(
        {
            "temp": rng.normal(15, 8, rows).round(2),
            "humidity": rng.uniform(20, 90, rows).round(1),
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
        }
    )
    path = tmp_path / "data" / "summed.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


class RegressionEvidenceC(EvidenceC):
    """EvidenceC on a regression task: the target is `cnt`, scored by MAE."""

    async def generate(self, request: Any) -> Any:
        if request.output_schema_name == "IntakeDraft":
            return json_result(
                {
                    "objective": "Predict the hourly count",
                    "target": "cnt",
                    "task_type": "regression",
                    "primary_metric": "mae",
                }
            )
        return await super().generate(request)


@pytest.mark.parametrize("composition", [Toggle.ON, Toggle.OFF])
async def test_questioned_target_components_are_flagged_and_excluded(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    summed: Path,
    database: Database,
    composition: Toggle,
) -> None:
    run_id = f"run_composition_{composition.value}"
    reason = "`casual` and `registered` may be parts of the target count."
    concern = _report(reason, feature_concern_columns=["casual", "registered"])
    script = RegressionEvidenceC(run_id, {PRE_LOCK: [concern], RECONCILE: [CONTINUE]})
    result = await _run(
        make_controller,
        run_config,
        summed,
        run_id,
        script,
        worker=NoWorker,
        target_composition=composition,
    )
    data = load_experiment_contract(database, run_id).data_contract
    reviews = script.reviews
    facts = json.loads(reviews[1].split(PLATFORM_FACTS, 1)[1].split("\n", 1)[0])
    by_column = {fact["column"]: fact for fact in facts}
    # The single-column diagnostic flags `registered` only, as in the live run.
    assert by_column["registered"]["score"] >= 0.85 > by_column["casual"]["score"]
    assert "registered" not in data.allowed_features
    if composition is Toggle.ON:
        (event,) = _errors(database, run_id, TARGET_COMPOSITION)
        details: Any = event.payload.details
        assert details["outcome"] == "flagged"
        assert details["subset"] == ["casual", "registered"]
        assert details["subset_r2"] >= 0.99 and details["fits"] <= 200
        assert by_column["casual"]["flagged"] and by_column["casual"]["target_component"]
        assert "target_component true" in reviews[1]
        # Excluded citing its own recorded diagnostic, and the reconciliation showed it.
        assert "casual" not in data.allowed_features
        (item,) = [f for f in data.excluded_features if f.name == "casual"]
        assert item.evidence == by_column["casual"]["diagnostic_id"]
        assert any("target-composition" in note for note in result.notes)
    else:
        assert _errors(database, run_id, TARGET_COMPOSITION) == []
        assert not by_column["casual"]["flagged"]
        assert "casual" in data.allowed_features
        assert "target_component" not in reviews[1]
