"""Bounded, in-memory scheduling of ready tasks; no durable workflow state."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

import anyio


@dataclass(frozen=True)
class ReadyTask:
    """One operation and the named inputs that must complete first."""

    run: Callable[[], Awaitable[None]]
    dependencies: frozenset[str] = frozenset()


async def run_ready(tasks: Mapping[str, ReadyTask], limit: int) -> dict[str, Exception]:
    """Run ready operations up to limit, skipping descendants of failed operations."""
    if limit < 1:
        raise ValueError("concurrency must be positive")
    if any(not task.dependencies <= tasks.keys() for task in tasks.values()):
        raise ValueError("unknown dependency")
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("cyclic dependencies")
        if name in visited:
            return
        visiting.add(name)
        for dep in tasks[name].dependencies:
            visit(dep)
        visiting.remove(name)
        visited.add(name)

    for name in tasks:
        visit(name)
    completed = {name: anyio.Event() for name in tasks}
    failures: dict[str, Exception] = {}
    limiter = anyio.CapacityLimiter(limit)

    async def run(name: str) -> None:
        task = tasks[name]
        try:
            for dependency in task.dependencies:
                await completed[dependency].wait()
            if task.dependencies & failures.keys():
                failures[name] = RuntimeError("dependency failed")
                return
            async with limiter:
                try:
                    await task.run()
                except Exception as exc:
                    failures[name] = exc
        finally:
            completed[name].set()

    async with anyio.create_task_group() as group:
        for name in tasks:
            group.start_soon(run, name)
    return failures
