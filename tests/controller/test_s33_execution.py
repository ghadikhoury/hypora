"""S3.3 on the real leakage, group and time dev cases, through execution (review of PR #37).

Two things are scripted: the model (`ExecutionScript`) and the container launch
(`scripted_sandbox`). The rest is the run as `hypora run` makes it: C proposes the data contract
from real diagnostics over the MCP server, A reviews it, the platform locks the partition, C
writes and submits a baseline and one justified candidate, and the experiment worker and the
harness run them and the final job under the locked contract.

Each test then reads back the files every stage was actually given, so the contract's exclusions,
its split and the case's limits are checked where trial code runs, not where they were decided.
No network, no Docker, no synthetic data: the cases' own snapshots, or the test skips.
"""

import hashlib
import json

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml  # type: ignore[import-untyped]

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.controller.states import DONE
from hypora.core.llm.providers.fake import json_result, text_result
from hypora.core.llm.types import MessageRole
from hypora.core.roles.reporter import metric_summary
from hypora.core.schemas import JobState, MetricDirection
from hypora.evaluation.case import GroupedRowsTrap, TargetLeakageTrap, TemporalOrderTrap
from hypora.evaluation.runner.agent import bounded_run_config
from hypora.execution.harness.partition import LOCK_FILE, load_verified_partition
from hypora.execution.harness.runner import load_tabular
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.spec import GIB
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.events import EventLog
from hypora.storage.records import list_trials, load_experiment_contract, load_result, run_jobs
from tests.controller import scripted_sandbox
from tests.controller.scripted import LINEAR, TEMPLATES
from tests.controller.test_s33_methodology import CASES, MethodologyScript, _write_visible_case

pytestmark = pytest.mark.anyio
RUN_ID = "run_exec"
CASE_IDS = ("dev_target_leakage", "dev_grouped_rows", "dev_temporal_order")
TRIALS = {
    "baseline": (
        (TEMPLATES / "dummy" / "pipeline.py").read_text(),
        {
            "kind": "baseline",
            "operation": "fit",
            "hypothesis": "A constant prediction is the reference every model must beat",
            "model_family": "dummy",
            "expected_effect": "the reference score on the primary metric",
            "estimated_cost": {"compute_seconds": 10},
        },
    ),
    "candidate": (
        LINEAR,
        {
            "kind": "candidate",
            "operation": "fit",
            "hypothesis": "A regularized linear model on the contract's features generalizes "
            "better than a constant prediction",
            "model_family": "linear",
            "expected_effect": "a better primary metric than the baseline",
            "estimated_cost": {"compute_seconds": 30},
        },
    ),
}


def _choices(case):
    """The methodology the case's own trap definition calls for: strategy, split key, leaks."""
    (trap,) = case.traps
    if isinstance(trap, TargetLeakageTrap):
        return "random", None, trap.columns
    if isinstance(trap, GroupedRowsTrap):
        return "group", trap.group_key, ()
    assert isinstance(trap, TemporalOrderTrap)
    return "temporal", trap.time_key, trap.future_columns


class ExecutionScript(MethodologyScript):
    """C's whole assignment: the methodology, then a baseline and one justified candidate."""

    def __init__(self, case):
        super().__init__(RUN_ID, *_choices(case))
        self.poll_seconds = 0
        self.kind = "baseline"
        self.trials = {}
        self.jobs = {}
        self.metrics = {}

    async def generate(self, request):
        schema = request.output_schema_name
        if schema == "FinalAnswer":
            self.requests.append(request)
            return json_result({"final_trial_id": self.trials.get("candidate"), "confidence": 0.8})
        if schema == "ReportNarrative":
            self.requests.append(request)
            return json_result({"summary": "The candidate was verified on the held-out rows."})
        if schema is None and (
            "You are Agent C" in request.messages[0].content
            or "You are Hypora's machine-learning engineer" in request.messages[0].content
        ):
            self.requests.append(request)
            return self._experiment_turn(request)
        return await super().generate(request)

    def _write(self, kind):
        self.kind = kind
        pipeline, design = TRIALS[kind]
        return self._call(
            "write_trial_files",
            idempotency_key=f"{self.run_id}:write_{kind}",
            design=design,
            pipeline_py=pipeline,
        )

    def _experiment_turn(self, request):
        last = request.messages[-1]
        if last.role is not MessageRole.TOOL:
            return self._call("inspect_dataset")
        if last.is_error:
            return text_result("A tool failed, so the work stops here.")
        result = json.loads(last.content)
        if last.tool_name == "inspect_dataset":
            return self._write("baseline")
        if last.tool_name == "write_trial_files":
            self.trials[self.kind] = result["trial_id"]
            return self._call(
                "submit_experiment",
                trial_id=result["trial_id"],
                idempotency_key=f"{self.run_id}:submit_{self.kind}",
            )
        if last.tool_name == "submit_experiment":
            self.jobs[self.kind] = result["job_id"]
            return self._call("get_experiment_status", job_id=result["job_id"])
        if last.tool_name == "get_experiment_status":
            tool = "get_experiment_metrics" if result["finished"] else "get_experiment_status"
            return self._call(tool, job_id=result["job_id"])
        if last.tool_name == "get_experiment_metrics":
            self.metrics[self.kind] = result
            if self.kind == "baseline":  # the candidate follows the measured baseline
                return self._write("candidate")
        return text_result("The candidate is the stronger trial.")


@pytest.fixture
def stages(monkeypatch):
    return scripted_sandbox.install(monkeypatch)


def _controller(make_controller, run_config, script, case):
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    flags = run_config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": Toggle.OFF}
    )
    # As the eval runner does: job limits inside the case's pre-approved bounds.
    controller.run_config = bounded_run_config(
        run_config.model_copy(update={"flags": flags}), case.contract_bounds
    )
    return controller


def _request(case, dataset, budget=None):
    """The request the eval runner makes of the controller for this case (`AgentSystem.run`)."""
    constraints = case.constraints
    return RunRequest(
        dataset=dataset,
        goal=case.request + "\n\nStructured user constraints:\n" + constraints.model_dump_json(),
        visible_constraints=constraints.model_dump_json(),
        secondary_metrics=constraints.secondary_metrics,
        quality_target=constraints.quality_target,
        target=constraints.target,
        task_type=constraints.task_type,
        primary_metric=constraints.primary_metric,
        forbidden_columns=constraints.forbidden_columns,
        budget=budget or case.budget,
        run_id=RUN_ID,
        permitted_tools=tuple(case.permitted_tools),
    )


def _locked_run(database, runs_root):
    """The run's latest contract and its partition, re-verified against the lock on disk."""
    contract = load_experiment_contract(database, RUN_ID)
    snapshot = runs_root / RUN_ID / "data" / "dataset.parquet"
    source = snapshot.read_bytes()
    locked = load_verified_partition(
        load_tabular(snapshot, source=source),
        contract,
        hashlib.sha256(source).hexdigest(),
        runs_root / RUN_ID / LOCK_FILE,
    )
    return contract, locked.prepared


def _expected_inputs(prepared, *, final):
    """What the harness must mount per stage: (action, {file: (table, positional row IDs)})."""
    features = prepared.features
    labels = pd.DataFrame({"target": prepared.target})

    def fit(train, validation=None):
        tables = {"train_features.parquet": (features, train)}
        tables["train_labels.parquet"] = (labels, train)
        if validation is not None:
            tables["validation_features.parquet"] = (features, validation)
            tables["validation_labels.parquet"] = (labels, validation)
        return "fit", tables

    expected = []
    for fold in prepared.folds:
        expected += [
            fit(fold.train, fold.validation),
            ("predict", {"features.parquet": (features, fold.validation)}),
        ]
    if final:  # refit on development rows only, then test features with no labels at all
        expected += [
            fit(prepared.development),
            ("predict", {"features.parquet": (features, prepared.test)}),
        ]
    return expected


def _assert_stages_got_exactly_the_locked_rows(ran, prepared, contract, *, final):
    expected = _expected_inputs(prepared, final=final)
    assert [stage.action for stage in ran] == [action for action, _ in expected]
    allowed = list(contract.data_contract.allowed_features)
    for stage, (action, tables) in zip(ran, expected, strict=True):
        extra = {"context.json"} | ({"model.joblib"} if action == "predict" else set())
        assert stage.inputs() == set(tables) | extra
        for name, (table, ids) in tables.items():
            mounted = stage.table(name)
            pd.testing.assert_frame_equal(mounted, table.iloc[ids].reset_index(drop=True))
            if name.endswith("features.parquet"):
                assert list(mounted.columns) == allowed


def _boundaries(prepared):
    yield prepared.development, prepared.test
    for fold in prepared.folds:
        yield fold.train, fold.validation


def _mean(database, trial_id, metric):
    result = load_result(database, trial_id, "validation")
    assert result is not None
    mean = metric_summary(result.metrics, metric).mean
    assert mean is not None
    return mean


@pytest.mark.parametrize("case_id", CASE_IDS)
async def test_the_contract_c_chose_is_what_the_baseline_and_candidate_run_under(
    case_id, tmp_path, make_controller, run_config, database, runs_root, stages
):
    case = CASES[case_id]
    dataset = _write_visible_case(case, tmp_path)
    script = ExecutionScript(case)
    controller = _controller(make_controller, run_config, script, case)

    result = await controller.run(_request(case, dataset))

    assert result.state == DONE, result.notes
    contract, prepared = _locked_run(database, runs_root)
    data, split = contract.data_contract, contract.data_contract.split
    frame = prepared.frame

    # The choices: the case's own trap decides what a correct contract is.
    (trap,) = case.traps
    _, key, leaks = _choices(case)
    assert contract.target == case.constraints.target and not data.non_learned_fixes
    assert {item.name for item in data.excluded_features} == set(leaks)
    assert not set(leaks) & set(data.allowed_features)
    for item in data.excluded_features:  # each exclusion cites its own real diagnostic
        assert any(
            found["diagnostic_id"] == item.evidence
            and found["diagnostic"] == "feature_predictiveness"
            and found["column"] == item.name
            for found in script.diagnostics
        )
    if isinstance(trap, TargetLeakageTrap):
        assert split.strategy.value == "stratified"
    else:
        grouped = isinstance(trap, GroupedRowsTrap)
        assert split.strategy.value == ("group" if grouped else "temporal")
        assert key == (split.group_key if grouped else split.time_key)
        assert any(found["column"] == key for found in script.diagnostics)
    # The locked split holds out what the case says must be held out.
    for train, held_out in _boundaries(prepared):
        if split.group_key:
            groups = frame[split.group_key]
            assert set(groups.iloc[train]).isdisjoint(groups.iloc[held_out])
        if split.time_key:
            times = pd.to_datetime(frame[split.time_key])
            assert times.iloc[train].max() < times.iloc[held_out].min()
    assert set(prepared.diagnostic).isdisjoint(prepared.test)
    assert all(set(prepared.diagnostic).isdisjoint(fold.validation) for fold in prepared.folds)

    # Submission: a baseline first, then one candidate justified against it, then the final job.
    trials = {trial.kind: trial for trial in list_trials(database, RUN_ID)}
    assert sorted(trials) == ["baseline", "candidate"]
    assert script.trials == {kind: trial.trial_id for kind, trial in trials.items()}
    assert trials["candidate"].hypothesis == TRIALS["candidate"][1]["hypothesis"]
    order = [
        call.name
        for call in script.tool_calls
        if call.name in ("write_trial_files", "submit_experiment", "get_experiment_metrics")
    ]
    assert order == ["write_trial_files", "submit_experiment", "get_experiment_metrics"] * 2
    jobs = run_jobs(database, RUN_ID)
    assert [job.status for job in jobs] == [JobState.COMPLETED] * 3, result.notes
    baseline, candidate = script.trials["baseline"], script.trials["candidate"]
    assert [job.trial_id for job in jobs] == [baseline, candidate, candidate]
    assert [job.job_id for job in jobs[:2]] == [script.jobs["baseline"], script.jobs["candidate"]]
    metric = contract.primary_metric
    sign = 1 if metric.direction is MetricDirection.MAXIMIZE else -1
    assert sign * _mean(database, candidate, metric) > sign * _mean(database, baseline, metric)
    assert result.verified and result.final_trial_id == candidate
    tested = load_result(database, candidate, "test")
    assert tested is not None and tested.verification["passed"] is True

    # Execution: every stage of every job got exactly the locked rows of the allowed columns.
    folders = [protocol.job_dir(runs_root.resolve(), RUN_ID, job.job_id) for job in jobs]
    for folder, final in zip(folders, (False, False, True), strict=True):
        requested = protocol.read_request(folder).request
        assert requested.contract_version == contract.version
        _assert_stages_got_exactly_the_locked_rows(
            stages.of_job(folder), prepared, contract, final=final
        )
    assert sum(len(stages.of_job(folder)) for folder in folders) == len(stages.stages)
    hidden = {contract.target, *leaks}
    for stage in stages.stages:
        for name in stage.inputs():
            if name.endswith("features.parquet"):
                assert not hidden & set(stage.table(name).columns)
    store = ArtifactStore(database, runs_root.resolve() / RUN_ID / "artifacts")
    manifest = yaml.safe_load(store.get(f"{jobs[-1].job_id}.manifest").path.read_text())
    assert manifest["contract_hash"] == contract.hash
    assert manifest["allowed_features"] == list(data.allowed_features)
    assert manifest["split"] == split.model_dump(mode="json")
    assert manifest["row_ids"]["test"] == prepared.test.tolist()
    if isinstance(trap, TargetLeakageTrap):
        # With the planted column in the features this score is ~1.0: the exclusion took effect.
        assert _mean(database, candidate, metric) < 0.95

    # Limits: the case's budget is the contract's, and no stage ran outside the case's bounds.
    bounds = case.contract_bounds.max_job_limits
    assert contract.budget == case.budget
    assert contract.default_job_limits.max_wall_seconds <= bounds.max_wall_seconds
    for stage in stages.stages:
        assert stage.limits.wall_seconds <= bounds.max_wall_seconds
        assert stage.limits.memory_bytes <= bounds.max_memory_gb * GIB
        assert stage.limits.cpus <= (bounds.cpu_threads or 1)
        assert not stage.limits.gpu
    assert result.limits_fired == ()
    max_trials, max_steps = case.budget.max_trials, case.budget.max_steps
    assert max_trials is not None and max_steps is not None
    assert len(trials) <= max_trials
    events = list(EventLog(database).read(RUN_ID))
    steps = sum(event.type.value in ("model_call", "tool_call") for event in events)
    assert steps <= max_steps
    assert not any(event.type.value == "error" for event in events)


@pytest.mark.parametrize("mode", [AgentMode.MULTI, AgentMode.SINGLE])
async def test_a_trial_budget_too_small_for_a_candidate_stops_it_and_selects_no_model(
    mode, tmp_path, make_controller, run_config, database, runs_root, stages
):
    """The budget binds on the same path: with room for one trial, the candidate is refused before
    it is written, and a baseline alone is never promoted to a final model."""
    case = CASES["dev_target_leakage"]
    dataset = _write_visible_case(case, tmp_path)
    script = ExecutionScript(case)
    controller = _controller(make_controller, run_config, script, case)
    controller.run_config = controller.run_config.model_copy(
        update={"flags": controller.run_config.flags.model_copy(update={"agent_mode": mode})}
    )
    budget = case.budget.model_copy(update={"max_trials": 1})

    result = await controller.run(_request(case, dataset, budget))

    assert "max_trials" in result.limits_fired
    trials = list_trials(database, RUN_ID)
    assert [trial.kind for trial in trials] == ["baseline"]
    jobs = run_jobs(database, RUN_ID)
    assert [job.status for job in jobs] == [JobState.COMPLETED]  # no candidate, no final job
    folder = protocol.job_dir(runs_root.resolve(), RUN_ID, jobs[0].job_id)
    assert len(stages.of_job(folder)) == len(stages.stages) > 0
    assert not result.verified and result.final_trial_id is None
    assert any("baseline and candidate are required" in note for note in result.notes)
    assert load_experiment_contract(database, RUN_ID).budget == budget
