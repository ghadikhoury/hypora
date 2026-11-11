"""The user's side of the plan gate (spec §10.2; S3.5 design §2.2): `hypora approve` and the
inline prompt of `hypora run --approve interactive`.

Both read the pending request from the event log (`pending_approval`), show the plan and its scope,
and write one decision file with `write_decision` (atomic; the first decision wins). The
controller validates the file again before it acts on it: run, approval ID, policy, a `user:`
approver, and the exact plan hash pending now.
"""

import getpass
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import anyio

from hypora.core.controller.approval_gate import (
    DecisionExists,
    PendingApproval,
    pending_approval,
    user_decision,
    write_decision,
)
from hypora.core.schemas import ApprovalPolicy
from hypora.storage.database import Database

HASH_PREFIX: Final = re.compile(r"^[0-9a-f]{4,64}$")

Answer = tuple[bool, str | None]
"""(approve, reason) from a person; a denial carries its reason."""
Ask = Callable[[PendingApproval], Answer | None]
"""Ask a person about one pending approval; None when they skip (decide later)."""


def current_user() -> str:
    """The OS user name recorded as the approver (`user:<name>`)."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


def plan_file(runs_root: Path, pending: PendingApproval) -> Path | None:
    """The plan file the controller wrote for the pending version, if it exists."""
    folder = runs_root / pending.run_id
    version = pending.plan_version
    names = [f"approved_plan_v{version}.json"] if version else []
    if version in (None, 1):
        names.append("approved_plan.json")
    return next((folder / name for name in names if (folder / name).is_file()), None)


def approval_lines(runs_root: Path, pending: PendingApproval) -> list[str]:
    """What a person is shown before deciding: the request, its exact hash, and its files."""
    version = pending.plan_version
    lines = [
        f"Plan approval pending for run {pending.run_id}",
        f"  approval  {pending.approval_id} (plan v{version or '?'}, {pending.policy.value})",
        f"  hash      {pending.artifact_hash}",
        f"  summary   {pending.summary}",
    ]
    plan = plan_file(runs_root, pending)
    if plan is not None:
        lines.append(f"  plan      {plan}")
    scope = runs_root / pending.run_id / f"approval_scope_v{version}.json"
    if version and scope.is_file():
        lines.append(f"  scope     {scope}")
    return lines


def approve_command(run_id: str, pending: PendingApproval) -> str:
    """The command that decides `pending` from another terminal."""
    return f"hypora approve {run_id} --hash {pending.artifact_hash[:12]}"


class ApprovalRefused(Exception):
    """The decision cannot be written; the message says why."""


def check_hash_prefix(pending: PendingApproval, prefix: str | None) -> None:
    """Refuse when the person approved another plan hash than the one pending now."""
    if prefix is None:
        return
    wanted = prefix.strip().lower()
    if not HASH_PREFIX.match(wanted):
        raise ApprovalRefused("--hash takes 4 to 64 lowercase hex characters of the plan hash")
    if not pending.artifact_hash.startswith(wanted):
        raise ApprovalRefused(
            f"the pending plan hash is {pending.artifact_hash[:12]}, not {wanted}: the plan "
            "changed since you reviewed it. Review it again with `hypora approve "
            f"{pending.run_id}`."
        )


def record_decision(
    runs_root: Path,
    pending: PendingApproval,
    *,
    approve: bool,
    reason: str | None,
    user: str | None = None,
) -> Path:
    """Write the decision file for `pending`; raises ApprovalRefused if one already exists."""
    if not approve and not (reason or "").strip():
        raise ApprovalRefused("a denial needs a reason (--reason TEXT)")
    record = user_decision(pending, approve=approve, user=user or current_user(), reason=reason)
    try:
        return write_decision(runs_root, record)
    except DecisionExists:
        raise ApprovalRefused(
            f"a decision on {pending.approval_id} was already written; the first decision wins"
        ) from None


def terminal_ask(pending: PendingApproval) -> Answer | None:
    """Ask on the terminal: approve, deny (with a reason), or skip."""
    answer = input("Approve this plan? [y]es / [n]o, deny / [s]kip for now: ").strip().lower()
    if answer in ("y", "yes"):
        return True, None
    if answer in ("n", "no"):
        reason = input("Reason for denying: ").strip()
        return False, reason or "denied by the user"
    return None


def is_terminal() -> bool:
    """Whether both standard input and output are a terminal (an inline prompt can be used)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


@dataclass
class InlineApprover:
    """Watches a running interactive run for a pending plan approval and asks once per request.

    In a terminal the question runs in a daemon thread (so a run that ends, or is decided from
    another terminal, never waits on the keyboard); the answer is written only if that request is
    still the pending one. Without a terminal it prints the `hypora approve` command instead.
    """

    database: Database
    runs_root: Path
    run_id: str
    out: Callable[[str], None]
    ask: Ask = terminal_ask
    terminal: bool = False
    seen: set[str] = field(default_factory=set)
    _threads: list[threading.Thread] = field(default_factory=list)

    async def watch(self, done: anyio.Event, poll_seconds: float = 1.0) -> None:
        """Until `done`, offer every new pending interactive approval to the person."""
        while not done.is_set():
            pending = pending_approval(self.database, self.run_id)
            if (
                pending is not None
                and pending.policy is ApprovalPolicy.INTERACTIVE
                and pending.approval_id not in self.seen
            ):
                self.seen.add(pending.approval_id)
                await self._offer(pending, done, poll_seconds)
            with anyio.move_on_after(poll_seconds):
                await done.wait()

    async def _offer(self, pending: PendingApproval, done: anyio.Event, poll: float) -> None:
        for line in approval_lines(self.runs_root, pending):
            self.out(line)
        command = approve_command(self.run_id, pending)
        if not self.terminal:
            self.out(f"Approve with `{command}`, or deny with `{command} --deny --reason TEXT`.")
            return
        answers: list[Answer | None] = []
        thread = threading.Thread(
            target=lambda: answers.append(self.ask(pending)),
            name=f"approve-{pending.approval_id}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()
        while not answers and not done.is_set():
            current = pending_approval(self.database, self.run_id)
            if current is None or current.approval_id != pending.approval_id:
                self.out("This approval was decided elsewhere; the prompt is no longer used.")
                return
            with anyio.move_on_after(poll):
                await done.wait()
        if not answers or done.is_set():
            return
        answer = answers[0]
        if answer is None:
            self.out(f"Skipped. Decide later with `{command}`.")
            return
        current = pending_approval(self.database, self.run_id)
        if current is None or current != pending:
            self.out("This approval is no longer pending; your answer was not recorded.")
            return
        approve, reason = answer
        try:
            path = record_decision(self.runs_root, pending, approve=approve, reason=reason)
        except ApprovalRefused as exc:
            self.out(f"Not recorded: {exc}")
            return
        self.out(f"Decision recorded ({'approved' if approve else 'denied'}): {path}")


__all__ = [
    "Answer",
    "ApprovalRefused",
    "Ask",
    "InlineApprover",
    "approval_lines",
    "approve_command",
    "check_hash_prefix",
    "current_user",
    "is_terminal",
    "plan_file",
    "record_decision",
    "terminal_ask",
]
