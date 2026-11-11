"""What a plan approval covers, and the one check that an approval is real and current (S3.5).

Spec §10.2: the user approves an exact artifact hash, and execution checks that hash; if the
artifact changed, approval resets. The approved artifact stays the frozen `ApprovedPlan` and its
`content_hash()`. Everything else the approval must cover, but that the plan schema has no field
for, is an `ApprovalScope`: the dataset fingerprint, the data contract hash, budget, job limits,
the tools each role may call, the worker ceiling, the sandbox image (its dependency set), network,
requested dependencies and model downloads, exports, sample sharing, and the LLM providers that see
data summaries. The scope is hashed and its hash is written into the plan's free-text notes as
`scope_sha256=<hex>` before the plan is hashed, so the plan hash covers the scope without a schema
change (D-044, design item 1.2).

- `approval_problem` is the single approval check used on every route that starts work: the
  controller (Phase B start, worker sessions, final job), the MCP server's `submit_experiment`,
  and the experiment worker at lease.
- `within_bounds` is what the `auto_within_contract` policy checks before it approves anything
  (spec §10.3: anything outside the pre-approved contract is denied).
- `bound_plan_problem` re-checks a stored plan against the live scope (recomputed from the values
  the run would actually use), so a plan or scope edited after approval is caught.
"""

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from pydantic import NonNegativeInt, ValidationError

from hypora.core.hashing import canonical_hash
from hypora.core.policy.allowlists import allowed_tools
from hypora.core.schemas import (
    ActorRole,
    ApprovalDecision,
    ApprovedPlan,
    Budget,
    EventType,
    ExperimentContract,
    JobLimits,
    ToolName,
)
from hypora.core.schemas.common import FrozenModel, Identifier, NonEmptyStr, Sha256Hex
from hypora.core.schemas.events import ApprovalRequestedEvent
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.records import RecordNotFound, load_approval, load_experiment_contract

SCOPE_NOTE_PREFIX: Final = "scope_sha256="
"""The plan-notes line that binds an `ApprovedPlan` to its `ApprovalScope`."""

NO_NETWORK: Final = "none"
"""The only network setting a plan can be approved with (spec §14.3)."""

APPROVAL_DEFENSE_OFF_APPROVER: Final = "policy:defense_off"
"""The approver of a plan granted with `defenses.approval` off (F6 ablation only). The reporter
recognises it and adds a mandatory limitation; a person's approver starts with `user:` and the
bounded policy's is `AUTO_APPROVER`."""


class RoleTools(FrozenModel):
    """The tools one role may call in this run: its allowlist intersected with the case's
    permitted tools, sorted so the scope hash does not depend on set order."""

    role: ActorRole
    tools: tuple[ToolName, ...]


class ApprovalScope(FrozenModel):
    """Everything a plan approval covers that `ApprovedPlan` has no field for (design §2.1).

    Local to this module and not a frozen schema (D-042 pattern). Its canonical hash is bound into
    the plan through the `scope_sha256=` notes line.
    """

    run_id: Identifier
    dataset_sha256: Sha256Hex
    data_contract_hash: Sha256Hex
    budget: Budget
    job_limits: JobLimits
    tools: tuple[RoleTools, ...]
    max_c_workers: NonNegativeInt
    parallel_mode: NonEmptyStr
    sandbox_image: NonEmptyStr
    network: NonEmptyStr = NO_NETWORK
    requested_dependencies: tuple[NonEmptyStr, ...] = ()
    requested_model_downloads: tuple[NonEmptyStr, ...] = ()
    exports: bool = False
    share_samples: bool = False
    llm_providers: tuple[NonEmptyStr, ...] = ()


class ApprovalBounds(FrozenModel):
    """What `auto_within_contract` may approve without a person (spec §10.3): the request's own
    budget, the configured job limits, the role allowlists intersected with the case's permitted
    tools, and the configured worker ceiling."""

    budget: Budget
    job_limits: JobLimits
    tools: tuple[RoleTools, ...]
    max_c_workers: NonNegativeInt


def effective_tools(
    roles: Iterable[ActorRole], permitted: Iterable[ToolName] | None
) -> tuple[RoleTools, ...]:
    """Each role's allowlist intersected with `permitted` (None: no case restriction)."""
    allowed = None if permitted is None else frozenset(permitted)
    result = []
    for role in sorted(set(roles), key=lambda r: r.value):
        tools = allowed_tools(role)
        if allowed is not None:
            tools = tools & allowed
        result.append(RoleTools(role=role, tools=tuple(sorted(tools, key=lambda t: t.value))))
    return tuple(result)


def scope_hash(scope: ApprovalScope) -> str:
    """Canonical-JSON SHA-256 of the scope."""
    return canonical_hash(scope.model_dump(mode="json"))


def write_scope(run_dir: Path, version: int, scope: ApprovalScope) -> Path:
    """Write `approval_scope_v<version>.json` (what the user is shown and approves); returns it."""
    path = run_dir / f"approval_scope_v{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scope.model_dump_json(indent=2), encoding="utf-8")
    return path


def bind_scope(plan: ApprovedPlan, scope: ApprovalScope) -> ApprovedPlan:
    """A copy of `plan` whose notes end with the scope's hash line (any old one is replaced).

    Re-validated through the schema, so the result is an ordinary frozen `ApprovedPlan`.
    """
    if scope.run_id != plan.run_id:
        raise ValueError("the scope belongs to another run")
    lines = [
        line
        for line in plan.phase_b_plan.notes.splitlines()
        if not line.startswith(SCOPE_NOTE_PREFIX)
    ]
    lines.append(SCOPE_NOTE_PREFIX + scope_hash(scope))
    document = plan.model_dump(mode="json")
    document["phase_b_plan"]["notes"] = "\n".join(lines)
    return ApprovedPlan.model_validate(document)


def plan_scope_hash(plan: ApprovedPlan) -> str | None:
    """The scope hash bound into `plan`, or None when it has none (or more than one)."""
    values = [
        line.removeprefix(SCOPE_NOTE_PREFIX)
        for line in plan.phase_b_plan.notes.splitlines()
        if line.startswith(SCOPE_NOTE_PREFIX)
    ]
    return values[0] if len(values) == 1 else None


def limits_problems(limits: JobLimits, ceiling: JobLimits) -> list[str]:
    """What in `limits` is above `ceiling` (empty when within it)."""
    over = []
    if limits.max_wall_seconds > ceiling.max_wall_seconds:
        over.append(f"max_wall_seconds {limits.max_wall_seconds} > {ceiling.max_wall_seconds}")
    if limits.max_memory_gb > ceiling.max_memory_gb:
        over.append(f"max_memory_gb {limits.max_memory_gb:g} > {ceiling.max_memory_gb:g}")
    if limits.gpu and not ceiling.gpu:
        over.append("gpu not approved")
    if (limits.cpu_threads or 1) > (ceiling.cpu_threads or 1):
        over.append(f"cpu_threads {limits.cpu_threads} > {ceiling.cpu_threads or 1}")
    return over


def _budget_problems(budget: Budget, bound: Budget) -> list[str]:
    """Budget fields above the bound; an unset (unlimited) field is above any set bound."""
    over = []
    for name in Budget.model_fields:
        value, maximum = getattr(budget, name), getattr(bound, name)
        if maximum is None:
            continue
        if value is None:
            over.append(f"budget {name} is unlimited (bound {maximum:g})")
        elif value > maximum:
            over.append(f"budget {name} {value:g} > {maximum:g}")
    return over


def _tools_problems(scope: tuple[RoleTools, ...], bound: tuple[RoleTools, ...]) -> list[str]:
    allowed: Mapping[ActorRole, frozenset[ToolName]] = {
        entry.role: frozenset(entry.tools) for entry in bound
    }
    over = []
    for entry in scope:
        extra = set(entry.tools) - allowed.get(entry.role, frozenset())
        if extra:
            names = ", ".join(sorted(tool.value for tool in extra))
            over.append(f"tools outside {entry.role.value}'s allowlist: {names}")
    return over


def within_bounds(scope: ApprovalScope, plan: ApprovedPlan, bounds: ApprovalBounds) -> list[str]:
    """Why `plan` (with `scope`) is outside what `auto_within_contract` may approve; empty when
    it is inside. Any requested dependency, model download, network, export or sample sharing
    is outside: those always need a person (spec §10.1)."""
    problems = []
    if scope.run_id != plan.run_id:
        problems.append("the scope belongs to another run")
    if plan_scope_hash(plan) != scope_hash(scope):
        problems.append("the plan is not bound to this scope")
    if plan.data_contract_hash != scope.data_contract_hash:
        problems.append("the plan's data contract differs from the scope's")
    if plan.budget != scope.budget:
        problems.append("the plan's budget differs from the scope's")
    problems += _budget_problems(scope.budget, bounds.budget)
    problems += [
        f"job limits: {item}" for item in limits_problems(scope.job_limits, bounds.job_limits)
    ]
    problems += _tools_problems(scope.tools, bounds.tools)
    if scope.max_c_workers > bounds.max_c_workers:
        problems.append(f"worker ceiling {scope.max_c_workers} > {bounds.max_c_workers}")
    dependencies = (*plan.requested_dependencies, *scope.requested_dependencies)
    if dependencies:
        problems.append(f"new dependencies requested: {', '.join(sorted(set(dependencies)))}")
    downloads = (*plan.requested_model_downloads, *scope.requested_model_downloads)
    if downloads:
        problems.append(f"model downloads requested: {', '.join(sorted(set(downloads)))}")
    if scope.network != NO_NETWORK:
        problems.append(f"network access requested: {scope.network}")
    if scope.exports:
        problems.append("data export requested")
    if scope.share_samples:
        problems.append("sample sharing requested")
    return problems


def approval_problem(
    database: Database,
    run_id: str,
    contract: ExperimentContract,
    approval_ref: str,
    *,
    runs_root: Path | None = None,
    dataset_path: Path | None = None,
    sandbox_image: str | None = None,
) -> str | None:
    """Why `approval_ref` does not authorize work under `contract` on `run_id`; None when it does.

    It requires: the approval record exists, belongs to the same run, is `approved`, names the
    contract's plan hash, is the contract's own approval, and the contract is the run's latest
    version (an approval of plan v1 never authorizes work after a v2 contract exists).
    """
    if contract.run_id != run_id:
        return f"contract v{contract.version} belongs to another run"
    if approval_ref != contract.approval_id:
        return f"{approval_ref!r} is not the approval of contract v{contract.version}"
    try:
        approval = load_approval(database, approval_ref)
    except RecordNotFound:
        return f"no approval {approval_ref!r} is on record"
    if approval.run_id != run_id:
        return f"approval {approval_ref!r} belongs to another run"
    if approval.decision is not ApprovalDecision.APPROVED:
        return f"approval {approval_ref!r} was {approval.decision.value}"
    if approval.artifact_hash != contract.approved_plan_hash:
        return f"approval {approval_ref!r} is for another plan hash"
    try:
        latest = load_experiment_contract(database, run_id)
    except (RecordNotFound, ValueError, ValidationError):
        return "the run's latest contract cannot be loaded"
    if latest.version != contract.version or latest.hash != contract.hash:
        return f"contract v{contract.version} is not the run's latest (v{latest.version})"
    if runs_root is not None:
        return execution_binding_problem(
            database, runs_root, contract, dataset_path=dataset_path, sandbox_image=sandbox_image
        )
    return None


def execution_binding_problem(
    database: Database,
    runs_root: Path,
    contract: ExperimentContract,
    *,
    dataset_path: Path | None,
    sandbox_image: str | None,
) -> str | None:
    """Recheck gate-created plan artifacts at submit and lease; legacy fixture records without
    an artifact-backed approval request retain their original approval-record check."""
    requests = EventLog(database).read(
        contract.run_id, EventFilters(types=(EventType.APPROVAL_REQUESTED,))
    )
    bound = any(
        isinstance(event, ApprovalRequestedEvent)
        and event.payload.approval_id == contract.approval_id
        and event.payload.artifact_ref == f"approved_plan:{contract.run_id}.v{contract.version}"
        for event in requests
    )
    if not bound:
        return None
    folder = runs_root / contract.run_id
    name = (
        "approved_plan.json" if contract.version == 1 else f"approved_plan_v{contract.version}.json"
    )
    try:
        scope = ApprovalScope.model_validate_json(
            (folder / f"approval_scope_v{contract.version}.json").read_bytes()
        )
        live = scope.model_dump(mode="json")
        live["data_contract_hash"] = contract.data_contract.content_hash()
        live["budget"] = contract.budget.model_dump(mode="json")
        live["job_limits"] = contract.default_job_limits.model_dump(mode="json")
        if dataset_path is not None:
            live["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if sandbox_image is not None:
            live["sandbox_image"] = sandbox_image
        current = ApprovalScope.model_validate(live)
    except (OSError, ValidationError, ValueError):
        return "the approved scope or execution dataset is missing or invalid"
    if current.run_id != contract.run_id:
        return "the approved scope belongs to another run"
    return plan_file_problem(folder / name, contract, current)


def bound_plan_problem(
    plan: ApprovedPlan, contract: ExperimentContract, scope: ApprovalScope
) -> str | None:
    """Why the stored `plan` and the live `scope` no longer match what `contract` approved."""
    if plan.content_hash() != contract.approved_plan_hash:
        return "the plan changed after it was approved"
    if plan_scope_hash(plan) != scope_hash(scope):
        return "the approval scope changed after the plan was approved"
    if plan.data_contract_hash != contract.data_contract.content_hash():
        return "the data contract differs from the approved plan's"
    return None


def plan_file_problem(path: Path, contract: ExperimentContract, scope: ApprovalScope) -> str | None:
    """`bound_plan_problem` for the plan file the controller wrote (re-read and re-hashed)."""
    try:
        plan = ApprovedPlan.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError):
        return f"the approved plan file {path.name} is missing or invalid"
    return bound_plan_problem(plan, contract, scope)


__all__ = [
    "APPROVAL_DEFENSE_OFF_APPROVER",
    "NO_NETWORK",
    "SCOPE_NOTE_PREFIX",
    "ApprovalBounds",
    "ApprovalScope",
    "RoleTools",
    "approval_problem",
    "bind_scope",
    "bound_plan_problem",
    "effective_tools",
    "limits_problems",
    "plan_file_problem",
    "plan_scope_hash",
    "scope_hash",
    "within_bounds",
    "write_scope",
]
