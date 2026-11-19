"""S4.1b review fixes (PR #50): the report states per column whether the final model uses it, the
"known at prediction time" match is whole-word and doubt-proof, names are bounded and safe in
report.md, nothing model-facing is cut silently, and the verifier's withheld-column guards hold.
"""

import itertools
from datetime import UTC, datetime

import pytest

from hypora.core.config.flags import LeakageCheckConfig
from hypora.core.controller.leakage import (
    DIAGNOSTIC_UNCHECKED_LIMITATION,
    DIAGNOSTIC_UNEXPLAINED_LIMITATION,
    column_outcomes,
    diagnostic_fallback_trigger,
    diagnostic_trigger_check,
    display_name,
    known_at_prediction,
)
from hypora.core.roles.multi import ResearchObservationWithRange
from hypora.core.roles.reporter import _leakage_limitations, _Timeline, render_leakage
from hypora.core.roles.research_agent import checked_typical_range
from hypora.core.schemas import ActorRole, CheckStatus, ErrorClass, MetricName, ToolError
from hypora.core.schemas.events import (
    LeakageCheckEvent,
    LeakageCheckPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.core.schemas.findings import VerificationCheck
from hypora.core.schemas.leakage import (
    LeakageColumnScore,
    LeakageModel,
    LeakageOutcome,
    LeakageSection,
    LeakageStep,
    ScoreRangeSource,
)
from hypora.core.verification.verifier import FinalEvidence, verify_final
from tests.unit.verification.test_verifier import evidence

Out = LeakageOutcome

# --- B2: the user's own words -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "column"),
    [
        ("Our coverage is legitimate.", "age"),
        ("The unpaid balance is known at prediction time.", "paid"),
        ("It's unclear if x is legitimate", "x"),
        ("Nobody knows if x is legitimate", "x"),
        ("I am not sure whether status is not leakage", "status"),
        ("collections_referrals_12m is not known at prediction time.", "collections_referrals_12m"),
        ("Is collections_referrals_12m known at prediction time?", "collections_referrals_12m"),
        ("Maybe status is known at prediction time.", "status"),
        ("status is known at prediction time only for some customers.", "status"),
        ("I think status is legitimate.", "status"),
        ("Everything is known at prediction time.", "status"),
        ("status_code is known at prediction time.", "status"),
        ("status is known at prediction time.", "status_code"),
        ("x is known at prediction time.", "!!!"),
    ],
)
def test_doubt_partial_words_and_other_columns_never_count_as_legitimate(
    goal: str, column: str
) -> None:
    assert known_at_prediction(goal, column) is False


@pytest.mark.parametrize(
    ("goal", "column"),
    [
        ("collections_referrals_12m is known at prediction time.", "collections_referrals_12m"),
        ("Predict churn; collections_referrals_12m is not leakage", "collections_referrals_12m"),
        (
            "The `collections_referrals_12m` column is available at prediction.",
            "collections_referrals_12m",
        ),
        ('"Collections Referrals 12m" is known at prediction time.', "collections_referrals_12m"),
        ("Credit Amount is known before the decision.", "credit amount"),
        ("The 'paid' flag is recorded before the outcome.", "paid"),
        ("Predict churn. age is legitimate. Thanks", "age"),
    ],
)
def test_clear_statements_about_the_exact_column_still_count(goal: str, column: str) -> None:
    assert known_at_prediction(goal, column) is True


# --- B1: per-column truth -------------------------------------------------------------------------


def test_column_outcomes_cover_every_combination() -> None:
    # Two confirmed, neither legitimate: first trained, second ran out of room.
    assert column_outcomes(["a", "b"], [], [Out.TRAINED, Out.BUDGET_EXHAUSTED], Out.EXCLUDED) == {
        "a": (Out.TRAINED, False),
        "b": (Out.BUDGET_EXHAUSTED, True),
    }
    # First comparison failed: the second was never attempted, both stay in.
    assert column_outcomes(["a", "b"], [], [Out.FAILED], None) == {
        "a": (Out.FAILED, True),
        "b": (Out.NOT_ATTEMPTED, True),
    }
    # One legitimate, the other trained and excluded.
    assert column_outcomes(["a", "b"], ["a"], [Out.TRAINED], Out.EXCLUDED) == {
        "a": (Out.SKIPPED_KNOWN_AT_PREDICTION, True),
        "b": (Out.TRAINED, False),
    }
    # Trained but the exclusion plan was denied or failed: the column stays in.
    for exclusion in (Out.EXCLUSION_DENIED, Out.EXCLUSION_FAILED):
        assert column_outcomes(["a"], [], [Out.TRAINED], exclusion) == {"a": (Out.TRAINED, True)}
    assert column_outcomes(["a", "b"], [], [Out.TRAINED, Out.TRAINED], Out.EXCLUDED) == {
        "a": (Out.TRAINED, False),
        "b": (Out.TRAINED, False),
    }


def _section(
    columns: list[str],
    declared: list[str],
    attempts: list[LeakageOutcome],
    exclusion: LeakageOutcome | None,
) -> LeakageSection:
    outcomes = column_outcomes(columns, declared, attempts, exclusion)
    doubtful = [c for c in columns if c not in declared]
    trained = list(itertools.takewhile(lambda o: o is Out.TRAINED, attempts))
    without = (
        LeakageModel(
            trial_id="trl_cmp", without_columns=doubtful[: len(trained)], validation_mean=0.8
        )
        if trained
        else None
    )
    recommended = "trl_cmp" if without is not None and exclusion is Out.EXCLUDED else "trl_cand"
    return LeakageSection(
        metric=MetricName.ROC_AUC,
        expected_low=0.7,
        expected_high=0.8,
        expected_source=ScoreRangeSource.ESTIMATE,
        columns=[
            LeakageColumnScore(
                column=c,
                screen_score=0.9,
                permutation_drop=0.2,
                confirmed=True,
                comparison=outcomes[c][0],
                in_final_model=outcomes[c][1],
            )
            for c in columns
        ],
        with_columns=LeakageModel(trial_id="trl_cand", validation_mean=0.97),
        without_columns=without,
        comparison=Out.TRAINED
        if trained
        else (attempts[0] if attempts else Out.SKIPPED_KNOWN_AT_PREDICTION),
        recommended_trial_id=recommended,
        known_at_prediction=not doubtful,
        exclusion=exclusion if without is not None else None,
    )


CASES = [
    # (columns, declared, attempts, exclusion)
    (["a"], [], [Out.TRAINED], Out.EXCLUDED),
    (["a"], [], [Out.TRAINED], Out.EXCLUSION_DENIED),
    (["a"], [], [Out.TRAINED], Out.EXCLUSION_FAILED),
    (["a"], [], [Out.BUDGET_EXHAUSTED], None),
    (["a"], [], [Out.FAILED], None),
    (["a"], ["a"], [], None),
    (["a", "b"], [], [Out.TRAINED, Out.TRAINED], Out.EXCLUDED),
    (["a", "b"], [], [Out.TRAINED, Out.BUDGET_EXHAUSTED], Out.EXCLUDED),
    (["a", "b"], [], [Out.TRAINED, Out.FAILED], Out.EXCLUDED),
    (["a", "b"], [], [Out.TRAINED, Out.BUDGET_EXHAUSTED], Out.EXCLUSION_DENIED),
    (["a", "b"], [], [Out.FAILED], None),
    (["a", "b"], [], [Out.BUDGET_EXHAUSTED], None),
    (["a", "b"], ["a"], [Out.TRAINED], Out.EXCLUDED),
    (["a", "b"], ["a"], [Out.BUDGET_EXHAUSTED], None),
    (["a", "b"], ["a"], [Out.TRAINED], Out.EXCLUSION_DENIED),
    (["a", "b"], ["a", "b"], [], None),
]


@pytest.mark.parametrize(("columns", "declared", "attempts", "exclusion"), CASES)
def test_the_report_states_for_each_column_whether_the_final_model_uses_it(
    columns: list[str],
    declared: list[str],
    attempts: list[LeakageOutcome],
    exclusion: LeakageOutcome | None,
) -> None:
    section = _section(columns, declared, attempts, exclusion)
    text = "\n".join(render_leakage(section))
    summary = next(line for line in text.splitlines() if line.startswith("Recommended final model"))
    clauses = summary.split("(", 1)[1] if "(" in summary else ""
    still = next((c for c in clauses.split(";") if "still uses" in c), "")
    not_used = next((c for c in clauses.split(";") if "does not use" in c), "")
    for column in section.columns:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"- `{column.column}`"))
        if column.in_final_model:
            assert "STILL USES it" in line and "does not use it" not in line
            assert f"`{column.column}`" in still and f"`{column.column}`" not in not_used
        else:
            assert "The recommended final model does not use it." in line
            assert f"`{column.column}`" in not_used and f"`{column.column}`" not in still
        if column.comparison is Out.SKIPPED_KNOWN_AT_PREDICTION:
            assert "You said it is known when you predict" in line
        if column.comparison is Out.BUDGET_EXHAUSTED:
            assert "left no room" in line
        if column.comparison is Out.NOT_ATTEMPTED:
            assert "comparison before it did not complete" in line
        if column.comparison is Out.FAILED:
            assert "did not complete" in line
    assert ("was denied" in text) is (section.exclusion is Out.EXCLUSION_DENIED)
    assert ("could not be applied" in text) is (section.exclusion is Out.EXCLUSION_FAILED)
    doubtful_used = [
        c.column
        for c in section.columns
        if c.in_final_model and c.comparison is not Out.SKIPPED_KNOWN_AT_PREDICTION
    ]
    assert ("run the task again without it" in text) is bool(doubtful_used)
    # Never "use the model without it" for a model that still uses a flagged column, unless the
    # sentence names exactly what that model leaves out (two columns: "The model without `a`").
    if section.without_columns is not None and len(section.columns) > 1:
        assert "use the model without it" not in text


def test_names_are_bounded_and_safe_in_markdown() -> None:
    long = "x" * 5000
    shown = display_name(long)
    assert len(shown) <= 200 and "more characters omitted by the platform" in shown
    assert display_name("a\nb") == "a b"
    section = _section(["bad`name\nwith newline"], [], [Out.TRAINED], Out.EXCLUDED)
    text = "\n".join(render_leakage(section))
    line = next(ln for ln in text.splitlines() if ln.startswith("- `"))
    assert "`bad'name with newline`" in line
    assert text.count("\n- `") == 1  # one column line, not broken in two


def test_a_dropped_range_never_silently_displaces_bs_uncertainties() -> None:
    from tests.controller.test_multi_loop import strategy_fixture

    report = strategy_fixture("run_x")
    report["citations"] = [{"ref": "passage:1", "source_url": "https://e.org", "revision": "1"}]
    observation = ResearchObservationWithRange.model_validate(
        {
            "summary": "s",
            "uncertainties": [f"u{i}" for i in range(10)],
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
    assert len(checked.uncertainties) == 10
    assert checked.uncertainties[:9] == [f"u{i}" for i in range(9)]
    assert "1 more research uncertainty item(s) not shown" in checked.uncertainties[-1]


# --- the verifier's withheld-column guards ---------------------------------------------------


def _allowed(e: FinalEvidence) -> tuple[CheckStatus, dict[str, object]]:
    checks = verify_final(e).checks
    check = next(c for c in checks if c.check is VerificationCheck.ALLOWED_FEATURES)
    return check.status, dict(check.details)


def test_a_withheld_column_must_be_absent_and_must_be_an_allowed_feature() -> None:
    from dataclasses import replace

    base = evidence()  # allowed features a, b; leak is excluded
    status, details = _allowed(replace(base, feature_columns=("a",), withheld_columns=("b",)))
    assert status is CheckStatus.PASSED and details["withheld_by_platform"] == ["b"]
    status, details = _allowed(replace(base, feature_columns=("a", "b"), withheld_columns=("b",)))
    assert status is CheckStatus.FAILED and details["forbidden_given"] == ["b"]
    # A withheld name that is not an allowed feature is refused, even with the full feature set.
    status, _ = _allowed(replace(base, feature_columns=("a", "b"), withheld_columns=("zzz",)))
    assert status is CheckStatus.FAILED
    status, _ = _allowed(replace(base, feature_columns=("a", "b"), withheld_columns=("leak",)))
    assert status is CheckStatus.FAILED


# --- S4.2j: diagnostic fallback trigger when expected_score_range is unknown ----------------------


def _diag_result(
    *,
    diagnostic_id: str = "diag_abc123",
    diagnostic: str = "feature_predictiveness",
    column: str = "leak_col",
    metric: str = "roc_auc",
    score: float | None = 0.9648,
    partition_sha256: str | None = "a" * 64,
    status: ToolResultStatus = ToolResultStatus.OK,
) -> ToolResultPayload:
    return ToolResultPayload(
        call_id="call_1",
        tool="run_diagnostic",
        status=status,
        result_summary={
            "diagnostic_id": diagnostic_id,
            "diagnostic": diagnostic,
            "column": column,
            "metric": metric,
            "score": score,
            "partition_sha256": partition_sha256,
        },
        error=None
        if status is ToolResultStatus.OK
        else ToolError(
            code="ERR",
            error_class=ErrorClass.INVALID_ARGUMENTS,
            message="failed",
            retryable=False,
        ),
        latency_ms=10.0,
    )


def test_diagnostic_trigger_check_respects_thresholds_lock_and_allowed_features() -> None:
    allowed = ["x", "leak_col", "y"]
    # Extreme roc_auc (>= 0.90) on locked split for kept column fires.
    hit = diagnostic_trigger_check([_diag_result(metric="roc_auc", score=0.90)], allowed)
    assert hit is not None and hit.diagnostic_id == "diag_abc123" and hit.score == 0.90
    # Extreme r2 (>= 0.85) on locked split for kept column fires.
    hit_r2 = diagnostic_trigger_check([_diag_result(metric="r2", score=0.85)], allowed)
    assert hit_r2 is not None and hit_r2.metric == "r2" and hit_r2.threshold == 0.85
    # Modest scores do not fire.
    assert diagnostic_trigger_check([_diag_result(metric="r2", score=0.0856)], allowed) is None
    assert diagnostic_trigger_check([_diag_result(metric="roc_auc", score=0.8999)], allowed) is None
    # Unlocked diagnostic (partition_sha256 is None) never fires.
    assert diagnostic_trigger_check([_diag_result(partition_sha256=None)], allowed) is None
    # Excluded column (not in allowed_features) never fires.
    assert diagnostic_trigger_check([_diag_result(column="dropped_col")], allowed) is None
    # Non-predictiveness diagnostic or failed tool call never fires.
    assert diagnostic_trigger_check([_diag_result(diagnostic="group_overlap")], allowed) is None
    assert diagnostic_trigger_check([_diag_result(status=ToolResultStatus.ERROR)], allowed) is None
    # Highest score wins when multiple extreme diagnostics exist.
    best = diagnostic_trigger_check(
        [
            _diag_result(diagnostic_id="diag_low", score=0.91),
            _diag_result(diagnostic_id="diag_high", score=0.97),
        ],
        allowed,
    )
    assert best is not None and best.diagnostic_id == "diag_high"


def test_diagnostic_fallback_trigger_handles_maximize_minimize_and_empty_folds() -> None:
    config = LeakageCheckConfig()
    max_res = diagnostic_fallback_trigger(MetricName.ROC_AUC, [0.97, 0.98], config)
    assert max_res.outcome is Out.FIRED and max_res.excess == config.score_margin
    min_res = diagnostic_fallback_trigger(MetricName.RMSE, [10.0, 12.0], config)
    assert min_res.outcome is Out.FIRED and min_res.excess == pytest.approx(
        config.relative_margin * 11.0
    )
    empty = diagnostic_fallback_trigger(MetricName.ROC_AUC, [], config)
    assert empty.outcome is Out.UNKNOWN_RANGE


def test_render_leakage_and_limitations_handle_unknown_range_diagnostic_trigger() -> None:
    section = _section(["a"], [], [Out.TRAINED], Out.EXCLUDED).model_copy(
        update={
            "expected_low": 0.0,
            "expected_high": 0.9,
            "expected_source": ScoreRangeSource.UNKNOWN,
            "expected_citation": "locked-split feature_predictiveness diagnostic diag_1 "
            "(roc_auc=0.9648 >= 0.9 on a)",
        }
    )
    text = "\n".join(render_leakage(section))
    assert "locked-split diagnostic flagged an extreme single-feature score" in text
    assert "diag_1" in text and "0.9648" in text
    assert "well above the expected range" not in text
    assert "None" not in text

    now = datetime.now(UTC)
    trigger_event = LeakageCheckEvent(
        run_id="run_x",
        seq=1,
        created_at=now,
        actor_role=ActorRole.CONTROLLER,
        payload=LeakageCheckPayload(
            step=LeakageStep.TRIGGER,
            outcome=Out.FIRED,
            metric=MetricName.ROC_AUC,
            expected_source=ScoreRangeSource.UNKNOWN,
            reason="locked-split diagnostic diag_1",
        ),
    )
    assert _leakage_limitations(_Timeline([trigger_event])) == [DIAGNOSTIC_UNEXPLAINED_LIMITATION]
    unavailable = LeakageCheckEvent(
        run_id="run_x",
        seq=2,
        created_at=now,
        actor_role=ActorRole.CONTROLLER,
        payload=LeakageCheckPayload(
            step=LeakageStep.CONFIRMATION,
            outcome=Out.UNAVAILABLE,
            columns=[LeakageColumnScore(column="a")],
        ),
    )
    assert _leakage_limitations(_Timeline([trigger_event, unavailable])) == [
        DIAGNOSTIC_UNCHECKED_LIMITATION.format(suspects=" (screened candidates: a)")
    ]
