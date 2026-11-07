"""CLI option wiring and refusal are verified with a fake runner; no providers or Docker."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypora.cli.app import app
from hypora.cli.eval import EvalOptions
from hypora.core.schemas import ModelProfile
from hypora.evaluation.runner.regression import RegressionReport
from hypora.evaluation.runner.runner import EvalOutcome
from hypora.evaluation.runner.store import EvalRefused

runner = CliRunner()


def test_eval_options_reach_executor_and_report_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[EvalOptions] = []

    async def execute(options, out):
        seen.append(options)
        out("unit executor")
        return EvalOutcome(
            tmp_path,
            (),
            (),
            (),
            RegressionReport(
                commit="abc",
                config="single_agent.dev",
                status="no_baseline",
                message="No earlier result",
            ),
        )

    monkeypatch.setattr("hypora.cli.eval.execute_eval", execute)
    result = runner.invoke(
        app,
        [
            "eval",
            "--cases",
            "dev",
            "--config",
            "single_agent",
            "--systems",
            "agent,baseline",
            "--profile",
            "dev",
            "--parallel",
            "2",
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "unit executor" in result.output and str(tmp_path) in result.output
    [options] = seen
    assert options.systems == ("agent", "baseline")
    assert options.profile is ModelProfile.DEV and options.parallel == 2 and options.resume


def test_eval_preflight_refusal_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def execute(options, out):
        raise EvalRefused("fixture refusal")

    monkeypatch.setattr("hypora.cli.eval.execute_eval", execute)
    result = runner.invoke(app, ["eval", "--cases", "dev", "--config", "single_agent"])
    assert result.exit_code == 2
    assert "error: fixture refusal" in result.output
