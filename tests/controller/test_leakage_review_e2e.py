"""S4.1b review fixes (PR #50), end to end with scripted models and the scripted sandbox.

- Two confirmed columns, the second comparison out of room: the final model leaves out the first
  column only, and the report says plainly that it still uses the second.
- The exclusion plan denied, or the new contract failing after approval: the run keeps its final
  model (the candidate) under its own approval, and the report says it still uses the column.
- A very long column name cannot make the investigation fail.
Synthetic tables built with the dev case's planting function (D-015 allows them in tests).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config.run_config import RunConfig
from hypora.core.controller import controller as controller_module
from hypora.core.controller.controller import PlanDenied, _Run
from hypora.core.controller.states import DONE
from hypora.core.roles.reporter import Report
from hypora.core.schemas.leakage import LeakageOutcome
from hypora.evaluation.cases.dev.plant_target_leakage import LEAK, TARGET, plant
from hypora.execution.harness import partition
from hypora.storage.database import Database
from hypora.storage.records import load_experiment_contract
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory
from tests.controller.test_leakage_flag import LeakageScript, _credit_like, _run, _steps, _write

pytestmark = pytest.mark.anyio
SECOND = "refund_status"


def _two_leaks() -> pd.DataFrame:
    frame = plant(_credit_like())
    rng = np.random.default_rng(7)
    bad = (frame[TARGET] == "bad").to_numpy()
    refunded = np.where(bad, rng.random(len(frame)) < 0.85, rng.random(len(frame)) < 0.1)
    frame.insert(1, SECOND, np.where(refunded, "refunded", "none"))
    return frame


async def test_two_confirmed_columns_with_the_second_comparison_out_of_room(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    # Both screened candidates are confirmed, whatever their drops (the rule has its own tests).
    monkeypatch.setattr(
        controller_module, "confirm_columns", lambda drops, *a, **k: dict.fromkeys(drops, True)
    )
    dataset = _write(tmp_path, _two_leaks(), "two_leaks.csv")
    run_id = "run_leak_two"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    # Baseline, candidate and one comparison fit; the second comparison has no room.
    result = await _run(controller, run_config, dataset, run_id, max_trials=3)
    assert result.state == DONE and result.verified
    steps = _steps(database, run_id)
    assert ("comparison", "trained") in steps and ("comparison", "budget_exhausted") in steps
    report = Report.model_validate_json(result.report_json.read_text())
    section = report.leakage
    assert section is not None and {c.column for c in section.columns} == {LEAK, SECOND}
    dropped = [c for c in section.columns if c.in_final_model is False]
    kept = [c for c in section.columns if c.in_final_model]
    assert len(dropped) == 1 and len(kept) == 1
    assert dropped[0].comparison is LeakageOutcome.TRAINED
    assert kept[0].comparison is LeakageOutcome.BUDGET_EXHAUSTED
    # The final model's contract leaves out exactly the dropped column, not the kept one.
    contract = load_experiment_contract(database, run_id)
    assert dropped[0].column not in contract.data_contract.allowed_features
    assert kept[0].column in contract.data_contract.allowed_features
    markdown = result.report_md.read_text()
    assert f"still uses `{kept[0].column}`" in markdown
    assert f"does not use `{dropped[0].column}`" in markdown
    assert "use the model without it" not in markdown
    assert f"If `{kept[0].column}` is recorded after the outcome" in markdown
    note = next(n for n in report.notes if n.startswith("Flagged column(s)"))
    assert f"still uses {kept[0].column}" in note and f"leaves out {dropped[0].column}" in note


@pytest.mark.parametrize("failure", ["denied", "broken"])
async def test_an_exclusion_that_does_not_happen_keeps_the_candidate_and_says_so(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    scripted_sandbox.install(monkeypatch)
    if failure == "denied":

        async def denied(self: Any, *args: Any) -> Any:
            raise PlanDenied(SimpleNamespace(reason="denied in the test"))  # type: ignore[arg-type]

        monkeypatch.setattr(_Run, "_excluding_contract", denied)
    else:
        real = partition.load_verified_partition

        def broken(frame: Any, contract: Any, *args: Any) -> Any:
            if contract.version >= 2:  # after the plan that excludes the column is approved
                raise ValueError("partition check failed in the test")
            return real(frame, contract, *args)

        monkeypatch.setattr(partition, "load_verified_partition", broken)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = f"run_leak_exclusion_{failure}"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id)
    # The run keeps a verified final model: the candidate, under its own contract v1.
    assert result.state == DONE and result.verified
    assert result.final_trial_id == script.trial_id
    assert load_experiment_contract(database, run_id).version == 1
    assert _steps(database, run_id)[-1] == ("decision", "recommend_with")
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is not None
    expected = (
        LeakageOutcome.EXCLUSION_DENIED if failure == "denied" else LeakageOutcome.EXCLUSION_FAILED
    )
    assert report.leakage.exclusion is expected
    assert report.leakage.columns[0].in_final_model is True
    markdown = result.report_md.read_text()
    assert f"still uses `{LEAK}`" in markdown and "STILL USES it" in markdown
    assert ("was denied" in markdown) is (failure == "denied")
    assert ("could not be applied" in markdown) is (failure == "broken")


async def test_a_very_long_column_name_is_flagged_not_a_failure(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    long_name = "collections_referrals_after_" + "x" * 2000
    frame = plant(_credit_like()).rename(columns={LEAK: long_name})
    dataset = _write(tmp_path, frame, "long_name.csv")
    run_id = "run_leak_long_name"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id)
    assert result.state == DONE and result.verified
    assert _steps(database, run_id)[-1] == ("decision", "recommend_without")
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is not None
    shown = report.leakage.columns[0].column
    assert len(shown) <= 200 and "more characters omitted by the platform" in shown
    contract = load_experiment_contract(database, run_id)
    assert long_name not in contract.data_contract.allowed_features
