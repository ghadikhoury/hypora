"""Idempotency of the two write tools (spec §11 `{run_id}:{call_id}`, §15.5; PR #15 review).

Serial retries, concurrent retries, and key reuse with different content, for both
`write_trial_files` and `submit_experiment`. The concurrent tests release two calls at the same
instant at the reservation itself, the interleaving the review reproduced.
"""

import errno
import hashlib
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
from sqlalchemy import func, select

from hypora.core.schemas.trials import TrialDesign
from hypora.mcp_server import tools as tools_module
from hypora.mcp_server.errors import ToolFailure
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.idempotency import IdempotencyLedger
from hypora.storage.models import Job, Trial
from tests.mcp_contracts.helpers import DESIGN, LINEAR, RUN_ID, call, fresh_key, new_trial, write

KEY = f"{RUN_ID}:call_retry"


def count(session, model) -> int:
    with session.database.engine.connect() as connection:
        return int(connection.scalar(select(func.count()).select_from(model)))


# --- serial retries (through the MCP client) ------------------------------------------------------


@pytest.mark.anyio
async def test_an_identical_create_retry_returns_the_original_trial(client, session):
    arguments = {"idempotency_key": KEY, "design": DESIGN, "pipeline_py": LINEAR}
    error, first = await call(client, "write_trial_files", arguments)
    error_again, again = await call(client, "write_trial_files", arguments)
    assert not error and not error_again
    assert again == first  # the original result, verbatim
    assert count(session, Trial) == 1


@pytest.mark.anyio
async def test_a_stale_update_retry_does_not_undo_a_later_update(client, session):
    trial_id = await new_trial(client)
    first = {"idempotency_key": KEY, "trial_id": trial_id, "pipeline_py": LINEAR + "# v2\n"}
    await call(client, "write_trial_files", first)
    await write(client, {"trial_id": trial_id, "pipeline_py": LINEAR + "# v3\n"})
    error, replay = await call(client, "write_trial_files", first)  # the late duplicate
    assert not error and replay["files"][0]["name"] == "pipeline.py"
    pipeline = session.runs_root / RUN_ID / "trials" / trial_id / "pipeline.py"
    assert pipeline.read_text().endswith("# v3\n")  # nothing was written again


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changed",
    [
        {"pipeline_py": LINEAR + "# different\n"},
        {"design": {**DESIGN, "hypothesis": "another idea"}},
        {"requirements_lock": ""},
    ],
)
async def test_a_reused_write_key_with_different_content_is_refused(client, session, changed):
    arguments = {"idempotency_key": KEY, "design": DESIGN, "pipeline_py": LINEAR}
    await call(client, "write_trial_files", arguments)
    error, content = await call(client, "write_trial_files", {**arguments, **changed})
    assert error and content["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert count(session, Trial) == 1


@pytest.mark.anyio
async def test_a_key_cannot_move_between_tools(client):
    trial_id = await new_trial(client)
    await call(client, "submit_experiment", {"trial_id": trial_id, "idempotency_key": KEY})
    error, content = await call(
        client,
        "write_trial_files",
        {"idempotency_key": KEY, "design": DESIGN, "pipeline_py": LINEAR},
    )
    assert error and content["code"] == "IDEMPOTENCY_KEY_REUSED"


@pytest.mark.anyio
async def test_write_keys_are_run_scoped(client):
    error, content = await call(
        client,
        "write_trial_files",
        {"idempotency_key": "run_other:w1", "design": DESIGN, "pipeline_py": LINEAR},
    )
    assert error and content["code"] == "IDEMPOTENCY_KEY_FORMAT"


# --- concurrent calls (released together at the reservation) -------------------------------------


def race(monkeypatch, calls: list[Callable[[], Any]]) -> list[Any]:
    """Run `calls` in threads that all reach the idempotency reservation at the same moment."""
    barrier = threading.Barrier(len(calls))
    reserve = IdempotencyLedger.reserve

    def synchronized(self, **kwargs):
        barrier.wait(timeout=30)
        return reserve(self, **kwargs)

    monkeypatch.setattr(IdempotencyLedger, "reserve", synchronized)
    outcomes: list[Any] = [None] * len(calls)

    def run(index: int) -> None:
        try:
            outcomes[index] = calls[index]()
        except ToolFailure as failure:
            outcomes[index] = failure.error

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return outcomes


def trial_for(tools: HyporaTools) -> str:
    return tools.write_trial_files(
        idempotency_key=fresh_key(), design=TrialDesign.model_validate(DESIGN), pipeline_py=LINEAR
    ).trial_id


def test_concurrent_identical_creates_make_one_trial(session, monkeypatch):
    tools = HyporaTools(session)
    design = TrialDesign.model_validate(DESIGN)
    outcomes = race(
        monkeypatch,
        [
            lambda: tools.write_trial_files(idempotency_key=KEY, design=design, pipeline_py=LINEAR)
            for _ in range(2)
        ],
    )
    assert outcomes[0] == outcomes[1] and outcomes[0].trial_id
    assert count(session, Trial) == 1


def test_concurrent_conflicting_writes_accept_exactly_one(session, monkeypatch):
    tools = HyporaTools(session)
    design = TrialDesign.model_validate(DESIGN)
    outcomes = race(
        monkeypatch,
        [
            partial(tools.write_trial_files, idempotency_key=KEY, design=design, pipeline_py=body)
            for body in (LINEAR, LINEAR + "# other\n")
        ],
    )
    codes = sorted(getattr(outcome, "code", "OK") for outcome in outcomes)
    assert codes == ["IDEMPOTENCY_KEY_REUSED", "OK"]
    assert count(session, Trial) == 1


def test_concurrent_identical_submissions_share_one_job(session, monkeypatch):
    tools = HyporaTools(session)
    trial = trial_for(tools)
    outcomes = race(
        monkeypatch,
        [lambda: tools.submit_experiment(trial_id=trial, idempotency_key=KEY) for _ in range(2)],
    )
    assert outcomes[0].job_id == outcomes[1].job_id
    assert sorted(outcome.deduplicated for outcome in outcomes) == [False, True]
    assert count(session, Job) == 1


def test_concurrent_conflicting_submissions_accept_exactly_one(session, monkeypatch):
    """The review's reproduction: same trial and key, configs {"C": 1.0} and {"C": 2.0}."""
    tools = HyporaTools(session)
    trial = trial_for(tools)
    outcomes = race(
        monkeypatch,
        [
            partial(tools.submit_experiment, trial_id=trial, idempotency_key=KEY, config={"C": c})
            for c in (1.0, 2.0)
        ],
    )
    accepted = [o for o in outcomes if not hasattr(o, "code")]
    refused = [o for o in outcomes if hasattr(o, "code")]
    assert len(accepted) == 1 and [o.code for o in refused] == ["IDEMPOTENCY_KEY_REUSED"]
    assert count(session, Job) == 1
    # The one accepted caller's experiment is exactly the one that will run.
    from hypora.execution.protocol import experiments as protocol

    folder = protocol.job_dir(session.runs_root, RUN_ID, accepted[0].job_id)
    stored = protocol.read_request(folder).request.config
    winner = outcomes.index(accepted[0])
    assert stored == {"C": (1.0, 2.0)[winner]}


def test_a_submission_interrupted_before_enqueue_completes_on_retry(session, monkeypatch):
    """A crash between reserving the key and queuing the job: the retry finishes the same plan."""
    tools = HyporaTools(session)
    trial = trial_for(tools)
    original = tools.session.queue.enqueue

    def crash(**kwargs):
        raise RuntimeError("worker died")

    monkeypatch.setattr(tools.session.queue, "enqueue", crash)
    with pytest.raises(RuntimeError):
        tools.submit_experiment(trial_id=trial, idempotency_key=KEY)
    monkeypatch.setattr(tools.session.queue, "enqueue", original)
    retried = tools.submit_experiment(trial_id=trial, idempotency_key=KEY)
    assert retried.state.value == "QUEUED" and count(session, Job) == 1
    assert tools.submit_experiment(trial_id=trial, idempotency_key=KEY).job_id == retried.job_id


def test_a_paused_duplicate_never_replays_over_a_newer_update(session, monkeypatch):
    """The PR #15 re-review's sequence, step for step:

    1. update A reserves key K for pipeline v2 (and pauses);
    2. duplicate A' reserves the same K while A is pending, and pauses after reading "pending";
    3. A finishes and records its result;
    4. update B, under a new key, writes v3;
    5. A' resumes: it must return A's original v2 result and write nothing.
    """
    tools = HyporaTools(session)
    trial = trial_for(tools)
    pipeline = session.runs_root / RUN_ID / "trials" / trial / "pipeline.py"
    v2, v3 = LINEAR + "# v2\n", LINEAR + "# v3\n"
    reserve = IdempotencyLedger.reserve
    reserved = {"A": threading.Event(), "A'": threading.Event()}
    resume = {"A": threading.Event(), "A'": threading.Event()}
    seen: dict[str, Any] = {}

    def paused_after_reserving(self, **kwargs):
        reservation = reserve(self, **kwargs)
        name = threading.current_thread().name
        if name in resume:
            seen[name] = reservation
            reserved[name].set()
            assert resume[name].wait(timeout=30)
        return reservation

    monkeypatch.setattr(IdempotencyLedger, "reserve", paused_after_reserving)
    results: dict[str, Any] = {}

    def update(name: str) -> None:
        results[name] = tools.write_trial_files(idempotency_key=KEY, trial_id=trial, pipeline_py=v2)

    original = threading.Thread(target=update, args=("A",), name="A")
    duplicate = threading.Thread(target=update, args=("A'",), name="A'")
    original.start()
    assert reserved["A"].wait(timeout=30)  # 1
    duplicate.start()
    assert reserved["A'"].wait(timeout=30)  # 2
    assert seen["A'"].result is None and not seen["A'"].fresh  # A' really read "pending"
    resume["A"].set()
    original.join(timeout=30)  # 3
    assert pipeline.read_text() == v2
    tools.write_trial_files(idempotency_key=fresh_key(), trial_id=trial, pipeline_py=v3)  # 4
    assert pipeline.read_text() == v3
    resume["A'"].set()
    duplicate.join(timeout=30)  # 5

    assert pipeline.read_text() == v3  # the newer update was not undone
    assert results["A'"] == results["A"]  # the duplicate returns the original result
    assert [f.name for f in results["A"].files] == ["pipeline.py"]
    assert results["A"].files[0].sha256 == hashlib.sha256(v2.encode()).hexdigest()


def test_an_effect_that_fails_records_nothing_and_its_retry_finishes_the_plan(session, monkeypatch):
    """Crash recovery: a write that dies inside the exclusive window leaves the key pending;
    the identical retry finishes the same plan (same trial ID) and records the result."""
    tools = HyporaTools(session)
    design = TrialDesign.model_validate(DESIGN)
    written = tools_module.write_file

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(tools_module, "write_file", written)
        raise OSError(errno.EIO, "disk went away")

    monkeypatch.setattr(tools_module, "write_file", fail_once)
    with pytest.raises(OSError):
        tools.write_trial_files(idempotency_key=KEY, design=design, pipeline_py=LINEAR)
    assert count(session, Trial) == 0  # the database part rolled back
    retried = tools.write_trial_files(idempotency_key=KEY, design=design, pipeline_py=LINEAR)
    again = tools.write_trial_files(idempotency_key=KEY, design=design, pipeline_py=LINEAR)
    assert retried == again and count(session, Trial) == 1
