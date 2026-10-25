"""The controller, not the agent, decides which trial is final (D-019 3a)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from hypora.core.controller.controller import select_final
from hypora.core.schemas import MetricName
from hypora.storage.database import Database
from tests.unit.controller.helpers import RUN_ID, add_trial, finished_job, run_database


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = run_database(tmp_path / "hypora.sqlite")
    yield opened
    opened.close()


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def test_the_agents_nomination_wins_when_it_completed(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    add_trial(database, "trl_b")
    finished_job(database, runs, "trl_a", fold_scores=[0.90, 0.92])
    job_b = finished_job(database, runs, "trl_b", fold_scores=[0.80, 0.82])
    chosen = select_final(database, RUN_ID, MetricName.ROC_AUC, "trl_b")
    assert (chosen.trial_id, chosen.source_job_id, chosen.reason) == (
        "trl_b",
        job_b,
        "nominated by the agent",
    )


def test_an_invalid_nomination_falls_back_to_the_best_mean(database: Database, runs: Path) -> None:
    for trial in ("trl_a", "trl_b", "trl_c"):
        add_trial(database, trial)
    job_a = finished_job(database, runs, "trl_a", fold_scores=[0.90, 0.92])
    finished_job(database, runs, "trl_b", fold_scores=[0.80, 0.82])
    finished_job(database, runs, "trl_c", succeeded=False)
    for nominated in ("trl_c", "trl_unknown", None):
        chosen = select_final(database, RUN_ID, MetricName.ROC_AUC, nominated)
        assert (chosen.trial_id, chosen.source_job_id) == ("trl_a", job_a)
        assert chosen.reason.startswith("best mean validation roc_auc")
    assert (
        "trl_c has no completed"
        in select_final(database, RUN_ID, MetricName.ROC_AUC, "trl_c").reason
    )


def test_a_minimized_metric_picks_the_lowest_mean(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    add_trial(database, "trl_b")
    finished_job(database, runs, "trl_a", fold_scores=[3.0, 3.2], metric="mae")
    job_b = finished_job(database, runs, "trl_b", fold_scores=[2.0, 2.4], metric="mae")
    chosen = select_final(database, RUN_ID, MetricName.MAE, None)
    assert (chosen.trial_id, chosen.source_job_id) == ("trl_b", job_b)


def test_no_completed_trial_means_no_final_model(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    finished_job(database, runs, "trl_a", succeeded=False)
    chosen = select_final(database, RUN_ID, MetricName.ROC_AUC, "trl_a")
    assert (chosen.trial_id, chosen.reason) == (None, "no trial completed validation")
