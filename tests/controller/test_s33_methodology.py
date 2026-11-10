"""Scripted S3.3 controller integration on frozen real dev cases; no network or Docker."""

import hashlib
import json

import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result, text_result
from hypora.core.llm.types import MessageRole
from hypora.core.schemas import DataContract, TaskSpecification
from hypora.evaluation.case import CASES_DIR, load_case_set
from hypora.evaluation.cases.dev.calibrate import visible_and_hidden
from hypora.evaluation.cases.dev.plant_grouped_rows import PLANTING as GROUP_PLANTING
from hypora.evaluation.cases.dev.plant_target_leakage import PLANTING as LEAK_PLANTING
from hypora.execution.harness.partition import load_verified_partition
from hypora.execution.harness.runner import load_tabular
from hypora.storage.events import EventLog
from hypora.storage.records import load_experiment_contract
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti
from tests.unit.evaluation.test_dev_cases import _has_snapshot

pytestmark = pytest.mark.anyio
CASES = {case.id: case for case in load_case_set(CASES_DIR / "dev")}
SETTINGS = {
    "dev_target_leakage": ("random", None, "collections_referrals_12m"),
    "dev_grouped_rows": ("group", "patient_id", None),
    "dev_temporal_order": ("temporal", "dteday", "casual"),
}
PLANTINGS = {"dev_target_leakage": LEAK_PLANTING, "dev_grouped_rows": GROUP_PLANTING}


class MethodologyScript(ScriptedMulti):
    """Exercise real diagnostic calls and base decisions on their actual returned IDs."""

    def __init__(self, run_id, strategy, split_key, leakage, *, stop_on_conflict=False):
        super().__init__(run_id)
        self.strategy, self.split_key, self.leakage = strategy, split_key, leakage
        self.stop_on_conflict = stop_on_conflict
        self.change_split = False
        self.ambiguous_target = False
        self.diagnostics = []

    async def generate(self, request):
        schema = request.output_schema_name
        context = request.messages[1].content
        if (
            schema is None
            and "You are Agent B" in request.messages[0].content
            and not request.tools
        ):
            self.requests.append(request)
            return text_result("No research tool is permitted.")
        if schema == "ResearchObservation" and not hasattr(self, "evidence"):
            self.requests.append(request)
            return json_result({"summary": "No research evidence", "uncertainties": []})
        if schema == "IntakeDraft" and self.ambiguous_target:
            self.requests.append(request)
            return json_result(
                {"objective": "Predict outcome", "target": None, "task_type": "regression"}
            )
        if schema == "ResearchObservation" and self.stop_on_conflict:
            self.requests.append(request)
            return json_result({"summary": "Temporal splitting is required", "uncertainties": []})
        if schema == "LeadAction" and self.stop_on_conflict and "Review C's proposed" in context:
            self.requests.append(request)
            assert "Temporal splitting is required" in context
            assert '"strategy":"group"' in context
            return json_result({"action": "report", "reason": "Conflicting evidence"})
        if "You are Hypora's data and implementation specialist" not in request.messages[0].content:
            return await super().generate(request)
        self.requests.append(request)
        if schema == "PlanDraft":
            return json_result(
                {"steps": [{"tool": None, "purpose": "Inspect evidence", "success_check": "Done"}]}
            )
        locked = "Current contract: " in context
        if schema == "DataContract":
            decoder = json.JSONDecoder()
            task = TaskSpecification.model_validate(
                decoder.raw_decode(context.split("Task specification: ", 1)[1])[0]
            )
            profile = decoder.raw_decode(context.split("Target-free dataset metadata: ", 1)[1])[0]
            columns = [item["name"] for item in profile["tabular"]["columns"]] + [task.target]
            if locked:
                payload = decoder.raw_decode(context.split("Current contract: ", 1)[1])[0]
            else:
                payload = build_data_contract(task, columns).model_dump(mode="json")
                if self.strategy == "group":
                    payload["split"] = {
                        "strategy": "group",
                        "group_key": self.split_key,
                        "time_key": None,
                        "test_fraction": 0.2,
                        "validation": "group_kfold_3",
                    }
                elif self.strategy == "temporal":
                    payload["split"] = {
                        "strategy": "temporal",
                        "group_key": None,
                        "time_key": self.split_key,
                        "test_fraction": 0.2,
                        "validation": "walk_forward_3",
                    }
            if locked and self.leakage:
                result = next(
                    item
                    for item in self.diagnostics
                    if item["diagnostic"] == "feature_predictiveness"
                )
                payload["allowed_features"].remove(self.leakage)
                payload["excluded_features"].append(
                    {
                        "name": self.leakage,
                        "reason": "Suspicious predictive feature",
                        "evidence": result["diagnostic_id"],
                    }
                )
            if locked and self.change_split:
                payload["split"] = {
                    "strategy": "random",
                    "group_key": None,
                    "time_key": None,
                    "test_fraction": 0.2,
                    "validation": "kfold_3",
                }
            return json_result(DataContract.model_validate(payload).model_dump(mode="json"))
        if request.messages[-1].role is MessageRole.TOOL:
            result = json.loads(request.messages[-1].content)
            if not request.messages[-1].is_error:
                self.diagnostics.append(result)
            return text_result("Diagnostic reviewed.")
        if not locked and self.split_key is not None:
            diagnostic = "group_overlap" if self.strategy == "group" else "timestamp_order"
            return self._call("run_diagnostic", diagnostic=diagnostic, column=self.split_key)
        if locked and self.leakage:
            task = TaskSpecification.model_validate_json(
                context.split("Task specification: ", 1)[1].split("\n", 1)[0]
            )
            assert task.task_type is not None
            return self._call(
                "run_diagnostic",
                diagnostic="feature_predictiveness",
                column=self.leakage,
                target=task.target,
                task_type=task.task_type.value,
            )
        return text_result("No diagnostic required.")


def _write_visible_case(case, tmp_path):
    if not _has_snapshot(case):
        pytest.skip(f"verified snapshot {case.dataset.snapshot} is not stored")
    planted = tmp_path / "planted"
    if case.id in PLANTINGS:
        PLANTINGS[case.id].write(planted_dir=planted)
    _, visible, _ = visible_and_hidden(case, planted_dir=planted)
    path = tmp_path / f"{case.id}.parquet"
    visible.to_parquet(path, index=False)
    return path


def _configure(make_controller, run_config, script):
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": Toggle.OFF}
    )
    controller.run_config = run_config.model_copy(update={"flags": flags})
    return controller


@pytest.mark.parametrize("case_id", list(SETTINGS))
async def test_real_case_diagnostics_reach_frozen_phase_b_contract(
    case_id, tmp_path, make_controller, run_config, monkeypatch, database, runs_root
):
    from hypora.core.controller.controller import _Run

    case = CASES[case_id]
    dataset = _write_visible_case(case, tmp_path)
    strategy, key, leak = SETTINGS[case_id]
    script = MethodologyScript("run_method", strategy, key, leak)
    controller = _configure(make_controller, run_config, script)
    captured = {}

    async def capture_phase_b(self, spec, contract, profile):
        captured["contract"] = contract
        source = self.dataset.read_bytes()
        captured["locked"] = load_verified_partition(
            load_tabular(self.dataset, source=source),
            contract,
            hashlib.sha256(source).hexdigest(),
            self.run_dir / "partition-lock.json",
        )

    monkeypatch.setattr(_Run, "_phase_b", capture_phase_b)
    await controller.run(
        RunRequest(
            dataset=dataset,
            goal=case.request,
            run_id="run_method",
            target=case.constraints.target,
            task_type=case.constraints.task_type,
            primary_metric=case.constraints.primary_metric,
            permitted_tools=tuple(case.permitted_tools),
        )
    )
    assert "contract" in captured, "controller did not reach Phase B with a frozen contract"
    contract = captured["contract"]
    stored = load_experiment_contract(database, "run_method", contract.version)
    assert stored.content_hash() == contract.content_hash()
    assert contract.data_contract.split.strategy.value == (
        "stratified" if strategy == "random" else strategy
    )
    if key:
        assert key in {
            contract.data_contract.split.group_key,
            contract.data_contract.split.time_key,
        }
        assert any(item["column"] == key for item in script.diagnostics)
    if leak:
        assert leak not in contract.data_contract.allowed_features
        assert any(item.name == leak for item in contract.data_contract.excluded_features)
        assert any(item["diagnostic"] == "feature_predictiveness" for item in script.diagnostics)
    locked = captured["locked"]
    assert set(locked.diagnostic).isdisjoint(locked.prepared.test)
    assert all(set(locked.diagnostic).isdisjoint(f.validation) for f in locked.prepared.folds)
    events = list(EventLog(database).read("run_method"))
    assert any(event.type.value == "tool_call" for event in events)


async def test_a_stops_on_conflicting_methodology(
    tmp_path, make_controller, run_config, database, monkeypatch
):
    from hypora.core.controller.controller import _Run

    dataset = _write_visible_case(CASES["dev_grouped_rows"], tmp_path)
    script = MethodologyScript("run_conflict", "group", "patient_id", None, stop_on_conflict=True)
    controller = _configure(make_controller, run_config, script)
    reached = []

    async def phase_b(*args):
        reached.append(True)

    monkeypatch.setattr(_Run, "_phase_b", phase_b)

    async def conflicting_research(self, spec, profile):
        from hypora.core.roles.multi import ResearchObservation

        return ResearchObservation(summary="Temporal splitting is required", uncertainties=[])

    monkeypatch.setattr(_Run, "_research", conflicting_research)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Conflicting evidence",
            run_id="run_conflict",
            target="total_UPDRS",
            task_type=CASES["dev_grouped_rows"].constraints.task_type,
        )
    )
    assert not reached
    assert any("Agent A decision: report" in note for note in result.notes)
    assert not (controller.runs_root / "run_conflict" / "partition-lock.json").exists()


async def test_ambiguous_target_stops_before_methodology(
    tmp_path, make_controller, run_config, monkeypatch
):
    import pandas as pd  # type: ignore[import-untyped]

    from hypora.core.controller.controller import _Run

    dataset = tmp_path / "ambiguous.csv"
    pd.DataFrame({"x": range(20), "a": range(20), "b": range(20)}).to_csv(dataset, index=False)
    script = MethodologyScript("run_ambiguous", "random", None, None)
    script.ambiguous_target = True
    controller = _configure(make_controller, run_config, script)
    reached = []

    async def phase_b(*args):
        reached.append(True)

    monkeypatch.setattr(_Run, "_phase_b", phase_b)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict one of two targets", run_id="run_ambiguous")
    )
    assert any("no target column" in note for note in result.notes)
    assert not reached
    assert not (controller.runs_root / "run_ambiguous" / "partition-lock.json").exists()


async def test_split_change_after_supervised_diagnostic_fails_closed(
    tmp_path, make_controller, run_config, monkeypatch, database
):
    from hypora.core.controller.controller import _Run

    dataset = _write_visible_case(CASES["dev_temporal_order"], tmp_path)
    script = MethodologyScript("run_changed", "temporal", "dteday", "casual")
    script.change_split = True
    controller = _configure(make_controller, run_config, script)
    reached = []

    async def phase_b(*args):
        reached.append(True)

    monkeypatch.setattr(_Run, "_phase_b", phase_b)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict bike rentals",
            run_id="run_changed",
            target="cnt",
            task_type=CASES["dev_temporal_order"].constraints.task_type,
        )
    )
    assert not reached
    assert any("split-affecting methodology changes" in note for note in result.notes)
    assert any(item["diagnostic"] == "feature_predictiveness" for item in script.diagnostics)
    assert (controller.runs_root / "run_changed" / "partition-lock.json").exists()
    assert load_experiment_contract(database, "run_changed").version == 1
