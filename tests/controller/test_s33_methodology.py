"""Scripted S3.3 methodology on frozen real dev cases: the runs that must stop before Phase B.

The runs that go on to execute a baseline and a candidate under the contract C chose are in
`test_s33_execution.py`, which shares this module's script. No network or Docker.
"""

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
from hypora.storage.records import RecordNotFound, load_experiment_contract
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti
from tests.unit.evaluation.test_dev_cases import _has_snapshot

pytestmark = pytest.mark.anyio
CASES = {case.id: case for case in load_case_set(CASES_DIR / "dev")}
PLANTINGS = {"dev_target_leakage": LEAK_PLANTING, "dev_grouped_rows": GROUP_PLANTING}


class MethodologyScript(ScriptedMulti):
    """Exercise real diagnostic calls and base decisions on their actual returned IDs."""

    def __init__(self, run_id, strategy, split_key, leakage, *, stop_on_conflict=False):
        super().__init__(run_id)
        self.strategy, self.split_key = strategy, split_key
        # One suspicious column or several; each is excluded on its own supervised diagnostic.
        self.leakage = (leakage,) if isinstance(leakage, str) else tuple(leakage or ())
        self.stop_on_conflict = stop_on_conflict
        self.change_split = False
        self.ambiguous_target = False
        self.fixes = []  # non-learned fixes C puts in its first proposal
        self.diagnostics = []

    def _undiagnosed(self):
        seen = {
            item["column"]
            for item in self.diagnostics
            if item["diagnostic"] == "feature_predictiveness"
        }
        return [column for column in self.leakage if column not in seen]

    def _predictiveness(self, context, column):
        task = TaskSpecification.model_validate_json(
            context.split("Task specification: ", 1)[1].split("\n", 1)[0]
        )
        assert task.task_type is not None
        return self._call(
            "run_diagnostic",
            diagnostic="feature_predictiveness",
            column=column,
            target=task.target,
            task_type=task.task_type.value,
        )

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
                if self.fixes:
                    payload["non_learned_fixes"] = self.fixes
            for column in self.leakage if locked else ():
                result = next(
                    item
                    for item in self.diagnostics
                    if item["diagnostic"] == "feature_predictiveness" and item["column"] == column
                )
                payload["allowed_features"].remove(column)
                payload["excluded_features"].append(
                    {
                        "name": column,
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
                if locked and self._undiagnosed():
                    return self._predictiveness(context, self._undiagnosed()[0])
            return text_result("Diagnostic reviewed.")
        if not locked and self.split_key is not None:
            diagnostic = "group_overlap" if self.strategy == "group" else "timestamp_order"
            return self._call("run_diagnostic", diagnostic=diagnostic, column=self.split_key)
        if locked and self.leakage:
            return self._predictiveness(context, self.leakage[0])
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


async def test_a_fix_on_the_target_stops_the_run_before_review_lock_or_contract(
    tmp_path, make_controller, run_config, database, monkeypatch
):
    """Review of PR #37: C's cast of the regression target was accepted, and then truncated the
    labels of every row, held-out ones included. Now the proposal never leaves validation.
    (Every kind of fix is covered where the rule lives: tests/unit/controller/test_methodology.py
    and tests/harness/test_data.py.)"""
    from hypora.core.controller.controller import _Run

    case = CASES["dev_grouped_rows"]
    dataset = _write_visible_case(case, tmp_path)
    script = MethodologyScript("run_label_fix", "group", "patient_id", None)
    script.fixes = [{"cast_dtype": {"column": case.constraints.target, "dtype": "int64"}}]
    controller = _configure(make_controller, run_config, script)
    reached = []

    async def phase_b(*args):
        reached.append(True)

    monkeypatch.setattr(_Run, "_phase_b", phase_b)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal=case.request,
            run_id="run_label_fix",
            target=case.constraints.target,
            task_type=case.constraints.task_type,
            primary_metric=case.constraints.primary_metric,
            permitted_tools=tuple(case.permitted_tools),
        )
    )
    proposals = [r for r in script.requests if r.output_schema_name == "DataContract"]
    assert len(proposals) == 1  # C did propose it; nothing after validation happened
    assert not reached and not result.verified
    assert any("CONTRACT stage failed: MethodologyError" in note for note in result.notes)
    assert not any(
        "Review C's proposed" in r.messages[1].content
        for r in script.requests
        if r.output_schema_name == "LeadAction"
    )
    assert not (controller.runs_root / "run_label_fix" / "partition-lock.json").exists()
    with pytest.raises(RecordNotFound):
        load_experiment_contract(database, "run_label_fix")


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
