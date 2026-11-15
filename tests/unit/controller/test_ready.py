"""Ready work starts promptly; dependencies and cancellation remain bounded."""

import anyio
import pytest

from hypora.core.scheduler.ready import ReadyTask, run_ready


@pytest.mark.anyio
async def test_independent_overlap_and_dependent_waiting():
    slow_started, release_slow, fast_done, child_done = [anyio.Event() for _ in range(4)]

    async def slow():
        slow_started.set()
        await release_slow.wait()

    async def fast():
        await slow_started.wait()
        fast_done.set()

    async def child():
        assert fast_done.is_set()
        assert not release_slow.is_set()
        child_done.set()
        release_slow.set()

    failures = await run_ready(
        {
            "slow": ReadyTask(slow),
            "fast": ReadyTask(fast),
            "child": ReadyTask(child, frozenset({"fast"})),
        },
        2,
    )
    assert not failures and child_done.is_set()


@pytest.mark.anyio
async def test_failure_skips_dependents_but_independent_work_finishes():
    seen = []

    async def fail():
        raise ValueError("expected")

    async def forbidden():
        seen.append("forbidden")

    async def independent():
        seen.append("independent")

    failures = await run_ready(
        {
            "a": ReadyTask(fail),
            "b": ReadyTask(forbidden, frozenset({"a"})),
            "c": ReadyTask(independent),
        },
        2,
    )
    assert set(failures) == {"a", "b"} and seen == ["independent"]


@pytest.mark.anyio
async def test_cancel_on_failure_stops_running_and_waiting_siblings():
    started = anyio.Event()
    cancelled = []

    async def hang():
        started.set()
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            cancelled.append("hang")
            raise

    async def fail():
        await started.wait()
        raise ValueError("expected")

    async def never():
        cancelled.append("never ran")  # must not run

    with anyio.fail_after(10):
        failures = await run_ready(
            {
                "hang": ReadyTask(hang),
                "fail": ReadyTask(fail),
                "child": ReadyTask(never, frozenset({"hang"})),
            },
            3,
            cancel_on_failure=True,
        )
    assert list(failures) == ["fail"] and isinstance(failures["fail"], ValueError)
    assert cancelled == ["hang"]


@pytest.mark.anyio
async def test_cycles_and_unknown_dependencies_rejected():
    async def noop():
        pass

    with pytest.raises(ValueError, match="cyclic"):
        await run_ready({"a": ReadyTask(noop, frozenset({"a"}))}, 1)
    with pytest.raises(ValueError, match="unknown"):
        await run_ready({"a": ReadyTask(noop, frozenset({"missing"}))}, 1)
