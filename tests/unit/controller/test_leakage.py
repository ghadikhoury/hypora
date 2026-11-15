"""S4.1b (D-046, D-050): the too-good-score investigation's platform decisions, without models.

Synthetic tables only (D-015 allows them in unit tests). The real planted credit-g file is covered
by `tests/controller/test_leakage_flag.py` where the snapshot is stored.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from hypora.core.config.flags import FeatureFlags, LeakageCheckConfig, Toggle
from hypora.core.controller.leakage import (
    confirm_columns,
    known_at_prediction,
    name_clues,
    permutation_drops,
    resolve_expected_range,
    trigger_check,
)
from hypora.core.roles.multi import ResearchObservationWithRange, TypicalScoreRange
from hypora.core.roles.research_agent import checked_typical_range
from hypora.core.schemas import ApprovedPlan, MetricName, TaskType
from hypora.core.schemas.leakage import ExpectedScoreRange, LeakageOutcome, ScoreRangeSource
from hypora.core.schemas.strategy import Citation
from hypora.execution.protocol.experiments import ExperimentOutcome, ExperimentRequest
from hypora.profiling.diagnostics import screen_candidates, single_column_screen
from tests.unit.schemas.helpers import load

CONFIG = LeakageCheckConfig()
CREDIT_RANGE = ExpectedScoreRange(
    metric=MetricName.ROC_AUC,
    source=ScoreRangeSource.RESEARCH,
    low=0.75,
    high=0.82,
    citation_ref="passage:1",
)


# --- the trigger check ----------------------------------------------------------------------------


def test_the_trigger_fires_on_the_live_planted_score() -> None:
    # run_07fa7fdc10f5 kept the planted column: validation ROC-AUC 0.9909.
    result = trigger_check(CREDIT_RANGE, [0.99, 0.985, 0.995, 0.99, 0.992], CONFIG)
    assert result.outcome is LeakageOutcome.FIRED
    assert result.excess == pytest.approx(0.1704, abs=1e-3)


@pytest.mark.parametrize(
    "folds",
    [
        [0.869],  # clean credit-g, optimistic single holdout (live)
        [0.80, 0.78, 0.83, 0.79, 0.815],  # clean credit-g, 5-fold CV (run_e25bcb422684)
        [0.837],  # random forest
        [0.75],  # inside the range
    ],
)
def test_the_trigger_stays_quiet_on_clean_credit_g_scores(folds: list[float]) -> None:
    assert trigger_check(CREDIT_RANGE, folds, CONFIG).outcome is LeakageOutcome.NOT_FIRED


def test_fold_noise_widens_the_margin() -> None:
    # Mean 0.92 is 0.10 above the high end: enough with steady folds, not with noisy ones.
    steady = trigger_check(CREDIT_RANGE, [0.92, 0.92, 0.92], CONFIG)
    noisy = trigger_check(CREDIT_RANGE, [0.82, 1.0, 0.94], CONFIG)
    assert steady.outcome is LeakageOutcome.FIRED
    assert noisy.outcome is LeakageOutcome.NOT_FIRED


def test_an_unknown_range_never_starts_an_investigation() -> None:
    unknown = ExpectedScoreRange(metric=MetricName.ROC_AUC, source=ScoreRangeSource.UNKNOWN)
    for expected in (unknown, None):
        result = trigger_check(expected, [0.999, 0.999], CONFIG)
        assert result.outcome is LeakageOutcome.UNKNOWN_RANGE


def test_minimized_metrics_are_too_good_below_the_low_end() -> None:
    rmse = ExpectedScoreRange(
        metric=MetricName.RMSE, source=ScoreRangeSource.ESTIMATE, low=50.0, high=60.0
    )
    assert trigger_check(rmse, [5.0, 6.0], CONFIG).outcome is LeakageOutcome.FIRED
    assert trigger_check(rmse, [45.0, 47.0], CONFIG).outcome is LeakageOutcome.NOT_FIRED


def test_ranges_are_validated() -> None:
    estimate, research = ScoreRangeSource.ESTIMATE, ScoreRangeSource.RESEARCH
    with pytest.raises(ValidationError):
        ExpectedScoreRange(metric=MetricName.ROC_AUC, source=estimate, low=0.9, high=0.8)
    with pytest.raises(ValidationError):
        ExpectedScoreRange(metric=MetricName.ACCURACY, source=estimate, low=0.5, high=1.2)
    with pytest.raises(ValidationError):  # research names its source
        ExpectedScoreRange(metric=MetricName.ROC_AUC, source=research, low=0.7, high=0.8)
    with pytest.raises(ValidationError):  # unknown carries no numbers
        ExpectedScoreRange(
            metric=MetricName.ROC_AUC, source=ScoreRangeSource.UNKNOWN, low=0.7, high=0.8
        )


# --- the expected range on the plan ---------------------------------------------------------------


def _ok(_text: str) -> bool:
    return True


def test_a_adopting_research_records_bs_cited_range_verbatim() -> None:
    typical = TypicalScoreRange(
        metric=MetricName.ROC_AUC, low=0.75, high=0.82, citation_ref="passage:1"
    )
    citations = [Citation(ref="passage:1", source_url="https://example.org/credit", revision="1")]
    estimate = SimpleNamespace(source="research", low=0.6, high=0.99, basis=None)
    found = resolve_expected_range(MetricName.ROC_AUC, estimate, typical, citations, _ok)
    assert (found.source, found.low, found.high) == (ScoreRangeSource.RESEARCH, 0.75, 0.82)
    assert found.source_url == "https://example.org/credit"


def test_without_a_usable_research_range_a_records_an_estimate_or_unknown() -> None:
    estimate = SimpleNamespace(source="research", low=0.7, high=0.8, basis="credit risk tasks")
    found = resolve_expected_range(MetricName.ROC_AUC, estimate, None, [], _ok)
    assert found.source is ScoreRangeSource.ESTIMATE and found.basis == "credit risk tasks"
    refused = resolve_expected_range(MetricName.ROC_AUC, estimate, None, [], lambda _t: False)
    assert refused.source is ScoreRangeSource.ESTIMATE and refused.basis is None
    bad = SimpleNamespace(source="estimate", low=0.9, high=0.1, basis=None)
    assert resolve_expected_range(MetricName.ROC_AUC, bad, None, [], _ok).source == "unknown"
    assert resolve_expected_range(MetricName.ROC_AUC, None, None, [], _ok).source == "unknown"
    unknown = SimpleNamespace(source="unknown", low=None, high=None, basis=None)
    assert resolve_expected_range(MetricName.ROC_AUC, unknown, None, [], _ok).source == "unknown"


def test_bs_range_with_an_undeclared_citation_is_dropped_and_said() -> None:
    from tests.controller.test_multi_loop import strategy_fixture

    report = strategy_fixture("run_x")
    report["citations"] = [
        {"ref": "passage:1", "source_url": "https://e.org", "revision": "1", "retrieved_on": None}
    ]
    observation = ResearchObservationWithRange.model_validate(
        {
            "summary": "s",
            "strategy_report": report,
            "typical_score_range": {
                "metric": "roc_auc",
                "low": 0.7,
                "high": 0.8,
                "citation_ref": "passage:9",
            },
        }
    )
    checked = checked_typical_range(observation, MetricName.ROC_AUC)
    assert isinstance(checked, ResearchObservationWithRange)
    assert getattr(checked, "typical_score_range") is None  # noqa: B009
    assert checked.uncertainties[-1].startswith("[Platform] The typical score range was left out")
    cited = getattr(observation, "typical_score_range").model_copy(  # noqa: B009
        update={"citation_ref": "passage:1"}
    )
    kept = checked_typical_range(
        observation.model_copy(update={"typical_score_range": cited}), MetricName.ROC_AUC
    )
    assert getattr(kept, "typical_score_range") == cited  # noqa: B009
    wrong_metric = checked_typical_range(kept, MetricName.ACCURACY)
    assert getattr(wrong_metric, "typical_score_range") is None  # noqa: B009


def test_a_plan_without_a_range_keeps_its_hash_and_one_with_it_changes() -> None:
    data = load("ApprovedPlan/yield")
    plan = ApprovedPlan.model_validate(data)
    assert "expected_score_range" not in plan.model_dump(mode="json")
    assert (
        plan.content_hash()
        == ApprovedPlan.model_validate(json.loads(json.dumps(data))).content_hash()
    )
    ranged = plan.model_copy(update={"expected_score_range": CREDIT_RANGE})
    again = ApprovedPlan.model_validate(ranged.model_dump(mode="json"))
    assert again.expected_score_range == CREDIT_RANGE
    assert again.content_hash() != plan.content_hash()


def test_old_records_still_load() -> None:
    request = {
        "request": {
            "run_id": "run_1",
            "trial_id": "trl_1",
            "operation": "fit",
            "family": "tabular",
            "contract_version": 1,
            "config": {},
            "limits": {"max_wall_seconds": 60, "max_memory_gb": 1.0},
            "approval_ref": "appr_1",
            "idempotency_key": "run_1:k",
        },
        "dataset_id": "ds",
        "dataset_path": "/x.csv",
        "dataset_sha256": "0" * 64,
        "trial_files": {},
    }
    loaded = ExperimentRequest.model_validate(request)
    assert loaded.withheld_columns == [] and loaded.permutation is None
    assert (
        ExperimentOutcome.model_validate({"job_id": "job_1", "succeeded": True}).permutation is None
    )
    # A config snapshot written before the flag existed: the flag takes its default.
    assert FeatureFlags.model_validate({}).leakage_flag is Toggle.ON


# --- the single-column screen ---------------------------------------------------------------------


def _binary(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.choice(["good", "bad"], rows, p=[0.7, 0.3])
    bad = y == "bad"
    return pd.DataFrame(
        {
            "amount": rng.normal(size=rows) + 0.3 * bad,
            "referrals_after": np.where(bad, rng.poisson(2.5, rows), rng.binomial(1, 0.05, rows)),
            "purpose": rng.choice(["car", "tv", "edu"], rows),
            "row_key": [f"id{i}" for i in range(rows)],
            "target": y,
        }
    )


def _scores(frame: pd.DataFrame, task: TaskType, **options: int) -> dict[str, float | None]:
    found = single_column_screen(frame.drop(columns="target"), frame["target"], task, **options)
    return {item.column: item.score for item in found}


def test_a_number_column_that_follows_the_label_scores_high_and_noise_near_chance() -> None:
    scores = _scores(_binary(), TaskType.BINARY_CLASSIFICATION)
    assert scores["referrals_after"] is not None and scores["referrals_after"] > 0.9
    assert scores["purpose"] is not None and scores["purpose"] < 0.6


def test_an_id_like_column_scores_near_chance_not_perfect() -> None:
    scores = _scores(_binary(), TaskType.BINARY_CLASSIFICATION)
    assert scores["row_key"] == pytest.approx(0.5, abs=0.02)


def test_a_category_column_is_scored_out_of_fold() -> None:
    frame = _binary()
    frame["status"] = np.where(frame["target"] == "bad", "collections", "open")
    scores = _scores(frame, TaskType.BINARY_CLASSIFICATION)
    assert scores["status"] == pytest.approx(1.0, abs=0.01)


def test_missing_values_count_as_their_own_category() -> None:
    frame = _binary()
    frame["closed_reason"] = pd.Series("none", index=frame.index).where(frame["target"] != "bad")
    frame["closed_amount"] = np.where(frame["target"] == "bad", np.nan, 1.0)
    scores = _scores(frame, TaskType.BINARY_CLASSIFICATION)
    assert scores["closed_reason"] == pytest.approx(1.0, abs=0.01)
    assert scores["closed_amount"] == pytest.approx(1.0, abs=0.01)


def test_regression_uses_absolute_spearman() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    frame = pd.DataFrame(
        {"echo": -3 * y + 0.01 * rng.normal(size=400), "noise": rng.normal(size=400), "target": y}
    )
    frame["band"] = pd.cut(frame["echo"], 5, labels=list("abcde")).astype(str)
    scores = _scores(frame, TaskType.REGRESSION)
    assert scores["echo"] == pytest.approx(1.0, abs=0.01)
    assert scores["noise"] is not None and scores["noise"] < 0.2
    # Five ordered bands of the echo: out-of-fold band means keep most of the rank order.
    assert scores["band"] is not None and scores["band"] > 0.8


def test_multiclass_averages_one_vs_rest() -> None:
    rng = np.random.default_rng(2)
    y = rng.choice(["a", "b", "c"], 600)
    frame = pd.DataFrame({"code": pd.Series(y).map({"a": 1, "b": 2, "c": 3}), "target": y})
    frame["noise"] = rng.normal(size=600)
    scores = _scores(frame, TaskType.MULTICLASS_CLASSIFICATION)
    assert scores["code"] is not None and scores["code"] > 0.8
    assert scores["noise"] is not None and scores["noise"] < 0.6


def test_the_screen_is_capped_by_a_stratified_sample() -> None:
    frame = _binary(rows=3000)
    capped = _scores(frame, TaskType.BINARY_CLASSIFICATION, max_rows=500)
    assert capped["referrals_after"] is not None and capped["referrals_after"] > 0.9


def test_candidates_stand_far_above_the_rest() -> None:
    found = single_column_screen(
        _binary().drop(columns="target"), _binary()["target"], TaskType.BINARY_CLASSIFICATION
    )
    candidates, rest = screen_candidates(found, min_excess=0.1, gap_ratio=1.5, max_candidates=3)
    assert [c.column for c in candidates] == ["referrals_after"]
    assert rest is not None and rest.column != "referrals_after"
    flat = single_column_screen(
        _binary().drop(columns=["target", "referrals_after"]),
        _binary()["target"],
        TaskType.BINARY_CLASSIFICATION,
    )
    assert screen_candidates(flat, min_excess=0.1, gap_ratio=1.5, max_candidates=3)[0] == []


# --- the permutation confirmation -----------------------------------------------------------------


def _permutation(drops: dict[str, float]) -> dict[str, object]:
    folds = [
        {
            "fold": n,
            "rows": 100,
            "baseline": 0.99,
            "shuffled": {c: [0.99 - d, 0.99 - d] for c, d in drops.items()},
        }
        for n in range(3)
    ]
    return {"metric": "roc_auc", "columns": list(drops), "repeats": 2, "folds": folds}


def test_a_column_is_confirmed_when_shuffling_removes_most_of_the_excess() -> None:
    drops = permutation_drops(_permutation({"leak": 0.2, "checking": 0.01}), MetricName.ROC_AUC)
    assert drops["leak"] == pytest.approx(0.2)
    assert confirm_columns(drops, excess=0.17, noise=0.01, config=CONFIG) == {
        "leak": True,
        "checking": False,
    }


def test_a_small_drop_confirms_nothing() -> None:
    drops = {"a": 0.03, "b": 0.02}
    assert not any(confirm_columns(drops, excess=0.17, noise=0.01, config=CONFIG).values())


def test_a_drop_within_the_fold_noise_confirms_nothing() -> None:
    assert confirm_columns({"a": 0.1}, excess=0.1, noise=0.12, config=CONFIG) == {"a": False}


def test_a_column_must_dominate_the_other_candidates() -> None:
    # "b" is not strong enough itself, but its drop is too close for "a" to be singled out.
    assert confirm_columns({"a": 0.1, "b": 0.07}, excess=0.15, noise=0.0, config=CONFIG) == {
        "a": False,
        "b": False,
    }


def test_at_most_two_columns_are_confirmed() -> None:
    drops = {"a": 0.3, "b": 0.25, "c": 0.2}
    result = confirm_columns(drops, excess=0.2, noise=0.0, config=CONFIG)
    assert [c for c, ok in result.items() if ok] == ["a", "b"]


def test_minimized_metrics_drop_upward() -> None:
    permutation = {
        "columns": ["leak"],
        "folds": [{"fold": 0, "baseline": 5.0, "shuffled": {"leak": [50.0, 52.0]}}],
    }
    assert permutation_drops(permutation, MetricName.RMSE)["leak"] == pytest.approx(46.0)


# --- context: clues and the user's own words ------------------------------------------------------


def test_name_clues_are_context_words() -> None:
    clues = name_clues("collections_referrals_12m")
    assert 'the name contains "collections"' in clues
    assert 'the name contains the time window "12m"' in clues
    assert name_clues("duration_months") == []


@pytest.mark.parametrize(
    ("goal", "known"),
    [
        ("Predict churn. collections_referrals_12m is known at prediction time.", True),
        ("The collections referrals 12m count is available at prediction.", True),
        ("collections_referrals_12m is not known at prediction time.", False),
        ("Predict churn; collections_referrals_12m is not leakage", True),
        ("Predict churn. Everything is known at prediction time.", False),
        ("Predict whether a customer is good or bad.", False),
    ],
)  # more cases, including the PR #50 review's, in test_leakage_review.py
def test_the_user_saying_a_column_is_legitimate_is_recognized_narrowly(
    goal: str, known: bool
) -> None:
    assert known_at_prediction(goal, "collections_referrals_12m") is known


def test_the_flag_has_a_default_and_its_thresholds_validate() -> None:
    assert FeatureFlags().leakage_flag is Toggle.ON
    with pytest.raises(ValidationError):
        LeakageCheckConfig(screen_gap_ratio=1.0)
    with pytest.raises(ValidationError):
        LeakageCheckConfig(max_confirmed=3)
