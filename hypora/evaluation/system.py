"""What an evaluated system receives and returns, and what grading it produces (spec §19.2, §19.3).

PLAN.md S2.2, S2.3, and S2.4 meet here. The agent and the fixed-pipeline baseline both implement
`EvalSystem`, so the runner can swap them, and `Grader` reads one `SystemRun` from either.

A system gets a `SystemInput`, never a `Case`: a case names its traps, its grading rules, and its
hidden split, none of which a system may see. Injected faults go in `SystemContext`, because the
platform applies them and the system's agent never reads them.
"""

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from hypora.core.config import RunConfig
from hypora.core.schemas import ExperimentContract, ToolName
from hypora.core.schemas.common import (
    Budget,
    Family,
    FrozenRecord,
    Identifier,
    MetricName,
    NonEmptyStr,
    Sha256Hex,
)
from hypora.evaluation.case import (
    Case,
    CaseConstraints,
    ContractBounds,
    DecisionClass,
    InjectedFault,
)
from hypora.evaluation.hidden_split import HiddenSplit


class SystemInput(FrozenRecord):
    """Everything a system may know about a case. `dataset` is `HiddenSplit.system_dataset()`."""

    case_id: Identifier
    request: NonEmptyStr
    constraints: CaseConstraints
    family: Family
    dataset: Path
    permitted_tools: tuple[ToolName, ...]
    budget: Budget
    contract_bounds: ContractBounds


class SystemContext(FrozenRecord):
    """Where a run writes and the settings it runs under; the platform's side of a case."""

    run_id: Identifier
    runs_root: Path
    database: Path
    run_config: RunConfig
    injected_faults: tuple[InjectedFault, ...] = ()


class ReportedMetric(FrozenRecord):
    """A score as the system's final report states it (what the honesty check compares)."""

    name: MetricName
    value: float


class FinalModel(FrozenRecord):
    """The model a run finished with: enough to score it again on rows it never saw."""

    trial_id: Identifier
    contract: ExperimentContract
    model_path: Path
    model_sha256: Sha256Hex
    pipeline_path: Path
    pipeline_sha256: Sha256Hex
    reported_metrics: tuple[ReportedMetric, ...] = ()


class SystemRun(FrozenRecord):
    """One system's run of one case, as the graders and the runner read it."""

    case_id: Identifier
    system: NonEmptyStr
    run_id: Identifier
    runs_root: Path
    database: Path
    completed: bool
    wall_clock_seconds: float
    decision_class: DecisionClass | None = None
    final_model: FinalModel | None = None
    limits_fired: tuple[NonEmptyStr, ...] = ()
    notes: tuple[str, ...] = ()


class CheckStatus(StrEnum):
    """A grader check's result. `NOT_APPLICABLE` never counts as a pass or a failure."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class GraderCheck(FrozenRecord):
    """One programmatic check, with where its evidence is."""

    name: NonEmptyStr
    status: CheckStatus
    detail: str
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class CaseGrade(FrozenRecord):
    """The grade of one run of one case. `passed` means no check failed and at least one passed."""

    case_id: Identifier
    system: NonEmptyStr
    run_id: Identifier
    passed: bool
    hidden_metric: ReportedMetric | None = None
    checks: tuple[GraderCheck, ...]


class EvalSystem(Protocol):
    """A system the runner can evaluate."""

    @property
    def name(self) -> str:
        """The system's name in result paths (`agent`, `baseline`)."""
        ...

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        """Run one case's request on its visible dataset and report what happened."""
        ...


class Grader(Protocol):
    """Grades one finished run against its case. Only the harness holds these inputs."""

    async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
        """Apply every pass criterion of spec §19.2 that the case calls for."""
        ...


__all__ = [
    "CaseGrade",
    "CheckStatus",
    "EvalSystem",
    "FinalModel",
    "Grader",
    "GraderCheck",
    "ReportedMetric",
    "SystemContext",
    "SystemInput",
    "SystemRun",
]
