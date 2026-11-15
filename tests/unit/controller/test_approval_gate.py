"""The plan approval gate on its own (S3.5 design §2.2): policies, the decision file, staleness,
and safe stops while waiting. No controller, worker or container; the clock is scripted."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from sqlalchemy.exc import IntegrityError

from hypora.core.controller.approval_gate import (
    DEFENSE_OFF_NOTE,
    ApprovalGate,
    DecisionExists,
    GateResult,
    GateVerdict,
    PendingApproval,
    decision_path,
    pending_approval,
    user_decision,
    write_decision,
)
from hypora.core.controller.contract import ContractError, build_data_contract, build_plan
from hypora.core.controller.limits import CANCEL_FILE, RunGuard
from hypora.core.policy.approvals import (
    APPROVAL_DEFENSE_OFF_APPROVER,
    ApprovalBounds,
    ApprovalScope,
    approval_problem,
    bind_scope,
    write_scope,
)
from hypora.core.roles.base import RunCancelled, RunLimitReached
from hypora.core.schemas import (
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovedPlan,
    Budget,
    DataContract,
    EventType,
    ExcludedFeature,
)
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import load_approval, save_approval, save_experiment_contract
from tests.policy.test_approvals import (
    BUDGET,
    COLUMNS,
    contract_for,
    make_bounds,
    make_plan,
    make_scope,
)
from tests.unit.controller.helpers import RUN_ID, run_database, spec

POLL = 0.005


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = run_database(tmp_path / "state.sqlite")
    yield opened
    opened.close()


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    (root / RUN_ID).mkdir(parents=True)
    return root


def gate(
    database: Database,
    runs_root: Path,
    policy: ApprovalPolicy,
    *,
    defense_on: bool = True,
    budget: Budget = BUDGET,
    clock: Clock | None = None,
) -> ApprovalGate:
    guard = RunGuard(
        run_id=RUN_ID,
        budget=budget,
        database=database,
        runs_root=runs_root,
        clock=clock or Clock(),
    )
    return ApprovalGate(
        database=database,
        runs_root=runs_root,
        run_id=RUN_ID,
        policy=policy,
        defense_on=defense_on,
        guard=guard,
        poll_seconds=POLL,
    )


def events(database: Database, *types: EventType) -> list[Any]:
    return [event for event in EventLog(database).read(RUN_ID) if not types or event.type in types]


def decide(
    approval_gate: ApprovalGate,
    plan: ApprovedPlan,
    scope: ApprovalScope,
    bounds: ApprovalBounds | None = None,
    *,
    meanwhile: Any = None,
) -> GateResult:
    """Run the gate; `meanwhile` (async, optional) acts while it waits."""
    result: list[GateResult] = []
    errors: list[Exception] = []

    async def main() -> None:
        async with anyio.create_task_group() as group:
            if meanwhile is not None:
                group.start_soon(meanwhile)
            try:
                with anyio.fail_after(5):
                    result.append(
                        await approval_gate.decide(
                            plan, scope, bounds or make_bounds(), note="Methodology approved."
                        )
                    )
            except Exception as exc:  # re-raised plainly below, not as an exception group
                errors.append(exc)
                group.cancel_scope.cancel()

    anyio.run(main)
    if errors:
        raise errors[0]
    return result[0]


async def until_pending(database: Database) -> Any:
    while (pending := pending_approval(database, RUN_ID)) is None:
        await anyio.sleep(POLL)
    return pending


# --- policies -------------------------------------------------------------------------------------


def test_auto_policy_grants_a_plan_inside_the_bounds(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)
    result = decide(gate(database, runs_root, ApprovalPolicy.AUTO_WITHIN_CONTRACT), plan, scope)
    assert result.granted and result.verdict is GateVerdict.GRANTED
    assert result.record.artifact_hash == plan.content_hash()
    requested, granted = events(database)
    assert (requested.type, granted.type) == (
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_GRANTED,
    )
    assert requested.payload.artifact_ref == f"approved_plan:{RUN_ID}.v1"
    assert "worker ceiling 2" in requested.payload.summary
    assert granted.actor_role.value == "policy_engine"
    contract = contract_for(result.record)
    save_experiment_contract(database, contract)
    assert approval_problem(database, RUN_ID, contract, contract.approval_id) is None
    assert pending_approval(database, RUN_ID) is None


def test_auto_policy_denies_a_plan_outside_the_bounds(database: Database, runs_root: Path) -> None:
    scope = make_scope(max_c_workers=3, exports=True)
    plan = make_plan(scope)
    result = decide(gate(database, runs_root, ApprovalPolicy.AUTO_WITHIN_CONTRACT), plan, scope)
    assert result.verdict is GateVerdict.DENIED_BY_POLICY and not result.granted
    [denied] = events(database, EventType.APPROVAL_DENIED)
    assert denied.payload.approver == "policy:auto_within_contract"
    assert "worker ceiling 3 > 2" in denied.payload.reason
    assert "data export requested" in denied.payload.reason
    assert load_approval(database, result.record.approval_id).decision is ApprovalDecision.DENIED
    assert not events(database, EventType.APPROVAL_GRANTED)


def test_with_the_defense_off_the_grant_skips_bounds_and_says_so(
    database: Database, runs_root: Path
) -> None:
    scope = make_scope(max_c_workers=3)  # outside the bounds: still granted, never silently
    plan = make_plan(scope)
    approval_gate = gate(database, runs_root, ApprovalPolicy.INTERACTIVE, defense_on=False)
    result = decide(approval_gate, plan, scope)
    assert result.granted and result.record.reason == DEFENSE_OFF_NOTE
    assert result.record.approver == APPROVAL_DEFENSE_OFF_APPROVER == "policy:defense_off"
    requested, decision, granted = events(database)
    assert granted.payload.approver == APPROVAL_DEFENSE_OFF_APPROVER
    assert DEFENSE_OFF_NOTE in requested.payload.summary
    assert "configured policy interactive" in requested.payload.summary
    assert decision.type is EventType.POLICY_DECISION
    assert decision.payload.decided_by.value == "approval_state"
    assert granted.payload.policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT
    assert requested.payload.policy is granted.payload.policy  # graders pair them


def test_a_plan_not_bound_to_its_scope_is_refused(database: Database, runs_root: Path) -> None:
    plan = make_plan(make_scope(max_c_workers=1))
    with pytest.raises(ContractError, match="not bound"):
        decide(gate(database, runs_root, ApprovalPolicy.AUTO_WITHIN_CONTRACT), plan, make_scope())
    assert events(database) == []


# --- interactive ----------------------------------------------------------------------------------


def test_an_interactive_approval_waits_for_the_users_decision(
    database: Database, runs_root: Path
) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def user() -> None:
        pending = await until_pending(database)
        assert pending.artifact_hash == plan.content_hash() and pending.plan_version == 1
        await anyio.sleep(5 * POLL)  # the gate keeps polling meanwhile
        write_decision(runs_root, user_decision(pending, approve=True, user="ghadi"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.verdict is GateVerdict.GRANTED
    assert result.record.approver == "user:ghadi"
    granted = events(database, EventType.APPROVAL_GRANTED)[-1]
    assert granted.actor_role.value == "user" and granted.payload.policy.value == "interactive"
    assert pending_approval(database, RUN_ID) is None


def test_a_user_denial_is_recorded_with_its_redacted_reason(
    database: Database, runs_root: Path
) -> None:
    scope = make_scope()
    plan = make_plan(scope)
    key = "sk-" + "A" * 24  # credential-shaped, built at runtime

    async def user() -> None:
        pending = await until_pending(database)
        decision = user_decision(pending, approve=False, user="ghadi", reason=f"too costly {key}")
        write_decision(runs_root, decision)

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.verdict is GateVerdict.DENIED_BY_USER
    [denied] = events(database, EventType.APPROVAL_DENIED)
    assert denied.payload.reason == "too costly [REDACTED]"
    assert key not in json.dumps([e.payload.model_dump(mode="json") for e in events(database)])


def test_a_decision_for_another_hash_is_stale_and_ignored(
    database: Database, runs_root: Path
) -> None:
    scope = make_scope()
    plan = make_plan(scope, version=2)
    old = make_plan(scope)  # v1's hash: what the user reviewed before the revision

    async def user() -> None:
        pending = await until_pending(database)
        stale = user_decision(pending, approve=True, user="ghadi").model_copy(
            update={"artifact_hash": old.content_hash()}
        )
        write_decision(runs_root, stale)
        while not events(database, EventType.ERROR):
            await anyio.sleep(POLL)
        write_decision(runs_root, user_decision(pending, approve=True, user="ghadi"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.granted and result.record.artifact_hash == plan.content_hash()
    [error] = events(database, EventType.ERROR)
    assert error.payload.code == "APPROVAL_STALE"
    assert error.payload.details["decision_hash"] == old.content_hash()
    rejected = list((decision_path(runs_root, RUN_ID, "x").parent / "rejected").iterdir())
    assert len(rejected) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"policy": ApprovalPolicy.AUTO_WITHIN_CONTRACT},
        {"approver": "policy:auto_within_contract"},
        {"approver": "user:"},
        {"approver": "user:   "},
        {"approver": "user:ghadi\nuser:root"},
        {"approver": "user:" + "x" * 65},
        {"run_id": "run_other"},
        {"approval_id": "appr_other_plan_v1"},
        {"artifact_type": ApprovalArtifactType.EXPORT},
    ],
)
def test_a_decision_not_made_by_the_user_for_this_run_is_ignored(
    database: Database, runs_root: Path, change: dict[str, Any]
) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def forger() -> None:
        pending = await until_pending(database)
        forged = user_decision(pending, approve=True, user="ghadi").model_copy(update=change)
        path = decision_path(runs_root, RUN_ID, pending.approval_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(forged.model_dump_json(), encoding="utf-8")
        while not events(database, EventType.ERROR):
            await anyio.sleep(POLL)
        write_decision(runs_root, user_decision(pending, approve=False, user="g", reason="no"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=forger
    )
    assert result.verdict is GateVerdict.DENIED_BY_USER
    assert events(database, EventType.ERROR)[0].payload.code == "APPROVAL_DECISION_INVALID"


def test_a_malformed_decision_file_is_ignored(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def user() -> None:
        pending = await until_pending(database)
        path = decision_path(runs_root, RUN_ID, pending.approval_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("approve please", encoding="utf-8")
        while not events(database, EventType.ERROR):
            await anyio.sleep(POLL)
        write_decision(runs_root, user_decision(pending, approve=True, user="ghadi"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.granted
    assert events(database, EventType.ERROR)[0].payload.code == "APPROVAL_DECISION_INVALID"


def test_cancel_while_waiting_stops_the_run(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def cancel() -> None:
        await until_pending(database)
        (runs_root / RUN_ID / CANCEL_FILE).touch()

    with pytest.raises(RunCancelled):
        decide(gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=cancel)
    [error] = events(database, EventType.ERROR)
    assert error.payload.code == "RUN_CANCELLED"
    assert not events(database, EventType.APPROVAL_GRANTED, EventType.APPROVAL_DENIED)


def test_the_wall_clock_runs_out_while_waiting(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)
    clock = Clock()

    async def time_passes() -> None:
        await until_pending(database)
        clock.now = 601.0

    approval_gate = gate(database, runs_root, ApprovalPolicy.INTERACTIVE, clock=clock)
    with pytest.raises(RunLimitReached) as stopped:
        decide(approval_gate, plan, scope, meanwhile=time_passes)
    assert stopped.value.limit == "max_wall_clock_seconds"
    [error] = events(database, EventType.ERROR)
    assert error.payload.code == "RUN_LIMIT_REACHED"


# --- the decision file and the record -------------------------------------------------------------


def test_the_first_decision_wins(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def user() -> None:
        pending = await until_pending(database)
        path = write_decision(runs_root, user_decision(pending, approve=True, user="a"))
        assert path.stat().st_mode & 0o077 == 0  # private to the user
        with pytest.raises(DecisionExists):
            write_decision(runs_root, user_decision(pending, approve=False, user="b", reason="x"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.granted and result.record.approver == "user:a"
    with pytest.raises(IntegrityError):  # the approvals row is never replaced
        save_approval(
            database, result.record.model_copy(update={"decision": "denied", "reason": "later"})
        )


def test_a_denial_cannot_be_turned_into_a_grant(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)

    async def user() -> None:
        pending = await until_pending(database)
        write_decision(runs_root, user_decision(pending, approve=False, user="g", reason="no"))

    result = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), plan, scope, meanwhile=user
    )
    assert result.verdict is GateVerdict.DENIED_BY_USER and not result.granted
    # Rewriting the decision file afterwards changes nothing: the record is final and the
    # shared check keeps refusing work under this approval.
    path = decision_path(runs_root, RUN_ID, result.record.approval_id)
    path.write_text(
        result.record.model_copy(
            update={"decision": ApprovalDecision.APPROVED, "reason": None}
        ).model_dump_json(),
        encoding="utf-8",
    )
    contract = contract_for(result.record)
    save_experiment_contract(database, contract)
    assert "was denied" in str(approval_problem(database, RUN_ID, contract, contract.approval_id))
    with pytest.raises(IntegrityError):
        save_approval(
            database, result.record.model_copy(update={"decision": ApprovalDecision.APPROVED})
        )
    assert not events(database, EventType.APPROVAL_GRANTED)


def test_user_names_are_cleaned_before_they_are_recorded(database: Database) -> None:
    pending = PendingApproval(
        run_id=RUN_ID,
        approval_id=f"appr_{RUN_ID}_plan_v1",
        artifact_hash="ab" * 32,
        policy=ApprovalPolicy.INTERACTIVE,
        summary="plan",
        artifact_ref=None,
    )
    assert user_decision(pending, approve=True, user="gha\ndi\x00").approver == "user:ghadi"
    assert user_decision(pending, approve=True, user="  ").approver == "user:unknown"
    assert len(user_decision(pending, approve=True, user="x" * 200).approver) == 5 + 64


def test_a_new_plan_version_needs_its_own_decision(database: Database, runs_root: Path) -> None:
    scope = make_scope()
    v1, v2 = make_plan(scope), make_plan(scope, version=2)
    first = decide(gate(database, runs_root, ApprovalPolicy.AUTO_WITHIN_CONTRACT), v1, scope)
    assert first.granted

    async def user() -> None:
        pending = await until_pending(database)
        assert pending.approval_id.endswith("_plan_v2") and pending.plan_version == 2
        write_decision(runs_root, user_decision(pending, approve=True, user="ghadi"))

    second = decide(
        gate(database, runs_root, ApprovalPolicy.INTERACTIVE), v2, scope, meanwhile=user
    )
    assert second.record.approval_id != first.record.approval_id
    assert second.record.artifact_hash == v2.content_hash() != v1.content_hash()


# --- covered by an earlier approval (S4.2d, D-052) ------------------------------------------------


def _without(data: DataContract, columns: list[str], version: int = 2) -> DataContract:
    return DataContract.model_validate(
        {
            **data.model_dump(mode="json", by_alias=True),
            "data_contract_version": version,
            "allowed_features": [f for f in data.allowed_features if f not in columns],
            "excluded_features": [
                *data.excluded_features,
                *(
                    ExcludedFeature(name=c, reason="too good", evidence="leakage_check:x")
                    for c in columns
                ),
            ],
        }
    )


def _covering(database: Database, runs_root: Path, policy: ApprovalPolicy) -> tuple[Any, ...]:
    """Plan v1 granted under `policy` (by the user when interactive), with its files written as
    the controller writes them; returns the gate, the record, the v1 data contract and scope."""
    scope = make_scope()
    v1 = make_plan(scope)
    folder = runs_root / RUN_ID
    (folder / "approved_plan.json").write_text(v1.model_dump_json(indent=2), encoding="utf-8")
    write_scope(folder, 1, scope)

    async def user() -> None:
        pending = await until_pending(database)
        write_decision(runs_root, user_decision(pending, approve=True, user="ghadi"))

    approval_gate = gate(database, runs_root, policy)
    interactive = policy is ApprovalPolicy.INTERACTIVE
    first = decide(approval_gate, v1, scope, meanwhile=user if interactive else None)
    return approval_gate, first.record, build_data_contract(spec(), COLUMNS), scope


def _covered_plan(
    data: DataContract, scope: ApprovalScope, version: int = 2, **scope_changes: Any
) -> tuple[ApprovedPlan, ApprovalScope]:
    new_scope = ApprovalScope.model_validate(
        scope.model_dump() | {"data_contract_hash": data.content_hash()} | scope_changes
    )
    plan = build_plan(spec(), data, BUDGET, note="Too-good-score check.", version=version)
    return bind_scope(plan, new_scope), new_scope


def test_a_covered_plan_is_granted_under_the_earlier_approval_without_a_prompt(
    database: Database, runs_root: Path
) -> None:
    approval_gate, first, old, scope = _covering(database, runs_root, ApprovalPolicy.INTERACTIVE)
    new = _without(old, ["a"])
    v2, new_scope = _covered_plan(new, scope)
    before = len(events(database))
    result = approval_gate.covered(
        v2, new_scope, covered_by=first, old_data=old, new_data=new, note="one comparison"
    )
    assert result.granted and result.record.approval_id == f"appr_{RUN_ID}_plan_v2"
    assert result.record.artifact_hash == v2.content_hash()
    assert result.record.approver == "user:ghadi"
    assert result.record.policy is ApprovalPolicy.INTERACTIVE
    requested, decision, granted = events(database)[before:]
    assert requested.type is EventType.APPROVAL_REQUESTED
    assert f"covered by {first.approval_id}: one comparison" in requested.payload.summary
    assert requested.payload.artifact_ref == f"approved_plan:{RUN_ID}.v2"
    assert decision.type is EventType.POLICY_DECISION
    assert decision.payload.decided_by.value == "approval_state"
    assert granted.type is EventType.APPROVAL_GRANTED
    assert granted.payload.approver == "user:ghadi"
    assert pending_approval(database, RUN_ID) is None  # never pending: no prompt
    assert not decision_path(runs_root, RUN_ID, result.record.approval_id).exists()
    contract = contract_for(result.record)
    save_experiment_contract(database, contract)
    assert approval_problem(database, RUN_ID, contract, contract.approval_id) is None
    # One comparison in total: the cover cannot be used a second time.
    v3, scope3 = _covered_plan(_without(old, ["b"], version=3), scope, version=3)
    with pytest.raises(ContractError, match="already used"):
        approval_gate.covered(
            v3,
            scope3,
            covered_by=first,
            old_data=old,
            new_data=_without(old, ["b"], version=3),
            note="again",
        )


def test_the_gate_refuses_anything_the_earlier_approval_does_not_cover(
    database: Database, runs_root: Path
) -> None:
    approval_gate, first, old, scope = _covering(
        database, runs_root, ApprovalPolicy.AUTO_WITHIN_CONTRACT
    )
    one = _without(old, ["a"])
    # Leaves out "a" and also adds a second exclusion: more than the one covered column.
    two = DataContract.model_validate(
        {
            **one.model_dump(mode="json", by_alias=True),
            "excluded_features": [
                *one.excluded_features,
                ExcludedFeature(name="extra", reason="too good", evidence="leakage_check:x"),
            ],
        }
    )
    v2, new_scope = _covered_plan(one, scope)
    wider_plan, wider_scope = _covered_plan(one, scope, max_c_workers=3)
    both_plan, both_scope = _covered_plan(two, scope)
    denied = first.model_copy(update={"decision": ApprovalDecision.DENIED, "reason": "no"})
    other_run = first.model_copy(update={"run_id": "run_other"})
    cases: list[tuple[Any, ...]] = [
        (v2, new_scope, denied, one, "covering approval"),
        (v2, new_scope, other_run, one, "covering approval"),
        (v2, make_scope(max_c_workers=1), first, one, "not bound"),
        (wider_plan, wider_scope, first, one, "scope differs"),
        (both_plan, both_scope, first, two, "exactly one allowed column"),
        (v2, new_scope, first, two, "not the plan's"),
    ]
    before = len(events(database))
    for plan, plan_scope, record, new, message in cases:
        with pytest.raises(ContractError, match=message):
            approval_gate.covered(
                plan, plan_scope, covered_by=record, old_data=old, new_data=new, note="x"
            )
    # An edited covering plan file no longer covers anything.
    path = runs_root / RUN_ID / "approved_plan.json"
    path.write_text(path.read_text().replace("Methodology.", "Edited."), encoding="utf-8")
    with pytest.raises(ContractError, match="changed after it was approved"):
        approval_gate.covered(v2, new_scope, covered_by=first, old_data=old, new_data=one, note="x")
    assert len(events(database)) == before
