"""Decision records: Supervisor (spec §6.2), Training Lead batches and decisions (§8.3, §8.4, §8.7).

These are LLM outputs, so they are flat models (an action enum plus the fields that action needs)
rather than nested unions; validators enforce which fields go with which action.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, PositiveInt, model_validator

from hypora.core.schemas.common import (
    Confidence,
    CostEstimate,
    EscalationType,
    Identifier,
    MetricName,
    NonEmptyStr,
    Record,
    RecordRef,
    StrictModel,
    TrialKind,
)
from hypora.core.schemas.trials import ProposedTrial, ShortBudget


class Escalation(StrictModel):
    """A Lead escalation to the Supervisor (spec §6.2, §8.7)."""

    type: EscalationType
    summary: NonEmptyStr
    trial_ids: list[Identifier] = Field(default_factory=list)
    evidence_refs: list[RecordRef] = Field(default_factory=list)


# --- Supervisor (spec §6.2) -----------------------------------------------------------------------


class SupervisorTrigger(StrEnum):
    """The only events on which the controller invokes the Supervisor (spec §6.2)."""

    PHASE_A_JOIN = "phase_a_join"
    ESCALATION = "escalation"
    BUDGET_EXHAUSTED = "budget_exhausted"


class SupervisorAction(StrEnum):
    CONTINUE = "continue"
    REVISIT_DATA = "revisit_data"
    REVISIT_STRATEGY = "revisit_strategy"
    STOP_AND_REPORT = "stop_and_report"
    ASK_USER = "ask_user"


_ACTIONS_WITH_QUESTION = frozenset(
    {SupervisorAction.REVISIT_DATA, SupervisorAction.REVISIT_STRATEGY, SupervisorAction.ASK_USER}
)


class SupervisorDecision(Record):
    """`continue | revisit_data(question) | revisit_strategy(question) | stop_and_report(reason)
    | ask_user(question)` (spec §6.2). The controller executes it; the Supervisor calls no tools."""

    run_id: Identifier
    trigger: SupervisorTrigger
    escalation_type: EscalationType | None = None
    decision: SupervisorAction
    question: str | None = None
    reason: str | None = None
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def _arguments_match_action(self) -> Self:
        if (self.trigger is SupervisorTrigger.ESCALATION) != (self.escalation_type is not None):
            raise ValueError("escalation_type is required for escalation triggers and only them")
        if (self.decision in _ACTIONS_WITH_QUESTION) != bool(self.question):
            raise ValueError(f"question is required for {self.decision.value!r} and only for it")
        if (self.decision is SupervisorAction.STOP_AND_REPORT) != bool(self.reason):
            raise ValueError("reason is required for stop_and_report and only for it")
        return self


# --- Training & Optimization Lead (spec §8.3, §8.4, §8.7) -----------------------------------------


class BreadthSituation(StrEnum):
    """Spec §8.3 adaptive-breadth table."""

    CHOICE_CLEAR = "choice_clear"
    GENUINELY_UNCERTAIN = "genuinely_uncertain"
    USER_NAMED_MODEL = "user_named_model"


class HalvingPlan(StrictModel):
    """Successive halving: K candidates on a short budget, keep the top ones (spec §8.4)."""

    short_budget: ShortBudget
    promote_top_k: PositiveInt
    promotion_metric: MetricName


class TrialBatchProposal(Record):
    """A batch of trials with confidence, a justification for K, and per-trial cost (spec §8.3).

    The controller injects the mandatory baseline and rejects batches over the approved budget.
    """

    batch_id: Identifier
    run_id: Identifier
    contract_version: PositiveInt
    situation: BreadthSituation
    confidence: Confidence
    k_justification: NonEmptyStr
    trials: list[ProposedTrial] = Field(min_length=1)
    halving: HalvingPlan | None = None

    @property
    def total_estimated_cost(self) -> CostEstimate:
        """Sum of the per-trial estimates (computed, not stated by the LLM)."""
        costs = [t.estimated_cost for t in self.trials]
        return CostEstimate(
            compute_seconds=sum(c.compute_seconds for c in costs),
            usd=sum(c.usd for c in costs),
            gpu=any(c.gpu for c in costs),
            assumptions=("sum of per-trial estimates",),
        )

    @model_validator(mode="after")
    def _halving_is_meaningful(self) -> Self:
        if self.halving is not None:
            contenders = [t for t in self.trials if t.kind is not TrialKind.BASELINE]
            if self.halving.promote_top_k >= len(contenders):
                raise ValueError("successive halving must promote fewer trials than it starts")
            # One short budget per round: the batch-level one. A per-trial copy may only repeat it.
            if any(t.short_budget not in (None, self.halving.short_budget) for t in self.trials):
                raise ValueError("a trial's short_budget must match halving.short_budget")
        return self


class LeadAction(StrEnum):
    STOP_SUCCESS = "stop_success"
    PROPOSE_NEXT_BATCH = "propose_next_batch"
    ESCALATE = "escalate"
    STOP_BUDGET = "stop_budget"


class LeadDecision(Record):
    """The Training Lead's decision after each batch (spec §8.7)."""

    run_id: Identifier
    after_batch_id: Identifier
    decision: LeadAction
    rationale: NonEmptyStr
    selected_trial_ids: list[Identifier] = Field(
        default_factory=list, description="Winner, or the Pareto frontier (spec §8.7)."
    )
    next_batch: TrialBatchProposal | None = None
    escalation: Escalation | None = None

    @model_validator(mode="after")
    def _arguments_match_action(self) -> Self:
        if (self.decision is LeadAction.PROPOSE_NEXT_BATCH) != (self.next_batch is not None):
            raise ValueError("next_batch is required for propose_next_batch and only for it")
        if (self.decision is LeadAction.ESCALATE) != (self.escalation is not None):
            raise ValueError("escalation is required for escalate and only for it")
        if self.decision is LeadAction.STOP_SUCCESS and not self.selected_trial_ids:
            raise ValueError("stop_success must name the selected trial(s)")
        return self
