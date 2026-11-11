"""`hypora approve`, the `hypora status` approval line, and the inline prompt of `hypora run`
(S3.5 design §2.2). The pending request is seeded in the event log as the gate records it."""

from pathlib import Path
from typing import Any

import anyio
import pytest
from typer.testing import CliRunner

from hypora.cli.app import app
from hypora.cli.approve import InlineApprover
from hypora.cli.status import run_status_lines
from hypora.core.controller.approval_gate import decision_path, pending_approval
from hypora.core.schemas import (
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    EventType,
    RunStatus,
)
from hypora.core.schemas.events import ApprovalGrantedPayload, ApprovalRequestedPayload
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import create_run
from hypora.storage.snapshots import SnapshotManifest

runner = CliRunner()
RUN = "run_gate"
PLAN_HASH = "ab" * 32
APPROVAL = f"appr_{RUN}_plan_v1"


def request_approval(
    database: Database, policy: ApprovalPolicy = ApprovalPolicy.INTERACTIVE, version: int = 1
) -> None:
    EventLog(database).append(
        RUN,
        EventType.APPROVAL_REQUESTED,
        "controller",
        ApprovalRequestedPayload(
            approval_id=f"appr_{RUN}_plan_v{version}",
            artifact_type=ApprovalArtifactType.PLAN,
            artifact_hash=PLAN_HASH,
            artifact_ref=f"approved_plan:{RUN}.v{version}",
            policy=policy,
            summary="Phase B plan v1; budget max_usd=1; worker ceiling 2",
        ),
    )


@pytest.fixture
def waiting(database: Database, runs_root: Path) -> Database:
    """A run parked at the plan gate, with its plan and scope files."""
    create_run(database, RUN, "ws_test", status=RunStatus.WAITING_APPROVAL)
    (runs_root / RUN).mkdir()
    (runs_root / RUN / "approved_plan.json").write_text("{}", encoding="utf-8")
    (runs_root / RUN / "approval_scope_v1.json").write_text("{}", encoding="utf-8")
    request_approval(database)
    return database


def invoke(database: Database, runs_root: Path, *args: str, **kwargs: Any) -> Any:
    db = str(database.engine.url.database)
    return runner.invoke(
        app, ["approve", RUN, *args, "--db", db, "--runs-root", str(runs_root)], **kwargs
    )


def decision(runs_root: Path) -> ApprovalRecord:
    return ApprovalRecord.model_validate_json(decision_path(runs_root, RUN, APPROVAL).read_bytes())


def test_approve_writes_the_users_decision_for_the_pending_hash(
    waiting: Database, runs_root: Path
) -> None:
    result = invoke(waiting, runs_root, "--hash", PLAN_HASH[:8], "--yes")
    assert result.exit_code == 0, result.output
    assert f"hash      {PLAN_HASH}" in result.output
    assert "approval_scope_v1.json" in result.output and "approved_plan.json" in result.output
    record = decision(runs_root)
    assert (record.run_id, record.approval_id, record.artifact_hash) == (RUN, APPROVAL, PLAN_HASH)
    assert record.decision is ApprovalDecision.APPROVED
    assert record.policy is ApprovalPolicy.INTERACTIVE and record.approver.startswith("user:")
    again = invoke(waiting, runs_root, "--yes")
    assert again.exit_code == 1 and "first decision wins" in again.output


def test_deny_needs_a_reason_and_records_it(waiting: Database, runs_root: Path) -> None:
    missing = invoke(waiting, runs_root, "--deny", "--yes")
    assert missing.exit_code == 2 and "--deny needs --reason" in missing.output
    assert not decision_path(runs_root, RUN, APPROVAL).exists()
    denied = invoke(waiting, runs_root, "--deny", "--reason", "budget too high", "--yes")
    assert denied.exit_code == 0, denied.output
    record = decision(runs_root)
    assert record.decision is ApprovalDecision.DENIED and record.reason == "budget too high"


def test_a_hash_other_than_the_pending_one_is_refused(waiting: Database, runs_root: Path) -> None:
    result = invoke(waiting, runs_root, "--hash", "ffff0000", "--yes")
    assert result.exit_code == 1 and "the plan changed since you reviewed it" in result.output
    bad = invoke(waiting, runs_root, "--hash", "not-hex", "--yes")
    assert bad.exit_code == 1 and "hex" in bad.output
    assert not decision_path(runs_root, RUN, APPROVAL).exists()


def test_declining_the_confirmation_records_nothing(waiting: Database, runs_root: Path) -> None:
    result = invoke(waiting, runs_root, input="n\n")
    assert result.exit_code == 1 and "nothing recorded" in result.output
    assert not decision_path(runs_root, RUN, APPROVAL).exists()
    accepted = invoke(waiting, runs_root, input="y\n")
    assert accepted.exit_code == 0 and decision(runs_root).decision is ApprovalDecision.APPROVED


def test_there_must_be_an_interactive_request_pending(database: Database, runs_root: Path) -> None:
    unknown = invoke(database, runs_root, "--yes")
    assert unknown.exit_code == 1 and f"no run {RUN}" in unknown.output
    create_run(database, RUN, "ws_test", status=RunStatus.WAITING_APPROVAL)
    (runs_root / RUN).mkdir()
    nothing = invoke(database, runs_root, "--yes")
    assert nothing.exit_code == 1 and "no plan approval pending" in nothing.output
    request_approval(database, ApprovalPolicy.AUTO_WITHIN_CONTRACT)
    by_policy = invoke(database, runs_root, "--yes")
    assert by_policy.exit_code == 1 and "not by a person" in by_policy.output


def test_a_decided_request_is_no_longer_pending(waiting: Database, runs_root: Path) -> None:
    assert pending_approval(waiting, RUN) is not None
    EventLog(waiting).append(
        RUN,
        EventType.APPROVAL_GRANTED,
        "user",
        ApprovalGrantedPayload(
            approval_id=APPROVAL,
            artifact_type=ApprovalArtifactType.PLAN,
            artifact_hash=PLAN_HASH,
            approver="user:ghadi",
            policy=ApprovalPolicy.INTERACTIVE,
        ),
    )
    assert pending_approval(waiting, RUN) is None
    result = invoke(waiting, runs_root, "--yes")
    assert result.exit_code == 1 and "no plan approval pending" in result.output
    request_approval(waiting, version=2)  # a revised plan asks again
    pending = pending_approval(waiting, RUN)
    assert pending is not None and pending.plan_version == 2


def test_status_shows_the_pending_approval(waiting: Database, runs_root: Path) -> None:
    lines = run_status_lines(waiting, runs_root, RUN, SnapshotManifest())
    [line] = [line for line in lines if line.startswith("Approval")]
    assert APPROVAL in line and PLAN_HASH[:12] in line and f"hypora approve {RUN}" in line


# --- the inline prompt ----------------------------------------------------------------------------


def watch(approver: InlineApprover, until: Any) -> None:
    async def main() -> None:
        done = anyio.Event()

        async def stop_when() -> None:
            with anyio.fail_after(5):
                while not until():
                    await anyio.sleep(0.005)
            done.set()

        async with anyio.create_task_group() as group:
            group.start_soon(stop_when)
            group.start_soon(approver.watch, done, 0.005)

    anyio.run(main)


def test_inline_prompt_writes_the_answer_in_a_terminal(waiting: Database, runs_root: Path) -> None:
    printed: list[str] = []
    asked: list[str] = []

    def ask(pending: Any) -> tuple[bool, str | None]:
        asked.append(pending.approval_id)
        return True, None

    approver = InlineApprover(waiting, runs_root, RUN, printed.append, ask=ask, terminal=True)
    watch(approver, lambda: decision_path(runs_root, RUN, APPROVAL).exists())
    assert asked == [APPROVAL]
    assert decision(runs_root).decision is ApprovalDecision.APPROVED
    assert any(PLAN_HASH in line for line in printed)


def test_without_a_terminal_the_approve_command_is_printed(
    waiting: Database, runs_root: Path
) -> None:
    printed: list[str] = []

    def never(pending: Any) -> None:
        raise AssertionError("no prompt without a terminal")

    approver = InlineApprover(waiting, runs_root, RUN, printed.append, ask=never, terminal=False)
    watch(approver, lambda: any("hypora approve" in line for line in printed))
    assert any(f"hypora approve {RUN} --hash {PLAN_HASH[:12]}" in line for line in printed)
    assert not decision_path(runs_root, RUN, APPROVAL).exists()


def test_a_skipped_prompt_records_nothing(waiting: Database, runs_root: Path) -> None:
    printed: list[str] = []
    approver = InlineApprover(
        waiting, runs_root, RUN, printed.append, ask=lambda pending: None, terminal=True
    )
    watch(approver, lambda: any(line.startswith("Skipped") for line in printed))
    assert not decision_path(runs_root, RUN, APPROVAL).exists()
