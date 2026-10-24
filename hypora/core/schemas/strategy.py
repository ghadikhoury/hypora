"""Research & Strategy branch output (spec §7.1)."""

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import Field, NonNegativeFloat, model_validator

from hypora.core.schemas.common import (
    Confidence,
    CostEstimate,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    Record,
    RecordRef,
    StrictModel,
    TrialKind,
)


class StrategyPath(StrEnum):
    """The four paths the Research Lead compares (spec §7.1, options A-D)."""

    HOSTED_API = "hosted_api"  # A
    PRETRAINED_AS_IS = "pretrained_as_is"  # B
    FINE_TUNE = "fine_tune"  # C
    TASK_SPECIFIC_TRAINING = "task_specific_training"  # D


class Assessment(StrEnum):
    """How favorable one tradeoff dimension is for an option ('good' = favors the option)."""

    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class TradeoffCell(StrictModel):
    rating: Assessment
    note: str = ""
    evidence_refs: list[RecordRef] = Field(default_factory=list)


class PriceStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class PriceEstimate(StrictModel):
    """A timestamped price, or 'unavailable' (spec §7.1, §13 freshness)."""

    status: PriceStatus
    amount_usd: NonNegativeFloat | None = None
    unit: str | None = Field(None, description="e.g. 'per 1,000 images'.")
    source_ref: RecordRef | None = None
    retrieved_on: date | None = None

    @model_validator(mode="after")
    def _complete_when_available(self) -> Self:
        if self.status is PriceStatus.AVAILABLE:
            if None in (self.amount_usd, self.unit, self.source_ref, self.retrieved_on):
                raise ValueError("an available price needs amount_usd, unit, source, and date")
        elif self.amount_usd is not None:
            raise ValueError("an unavailable price must not carry an amount")
        return self


class TradeoffRow(StrictModel):
    """One option's row of the tradeoff table (spec §7.1, dimensions in the spec's order)."""

    task_fit: TradeoffCell
    evidence_quality: TradeoffCell
    privacy: TradeoffCell
    license: TradeoffCell
    hardware_feasibility: TradeoffCell
    setup_cost: TradeoffCell
    expected_inference_cost: TradeoffCell
    latency_uncertainty: TradeoffCell
    evaluation_risk: TradeoffCell
    license_id: str | None = Field(None, description="SPDX id or license name, if known.")
    price: PriceEstimate | None = Field(None, description="Hosted API price (path A).")


class StrategyOption(StrictModel):
    option_id: Identifier = Field(description="Stable label within the report, e.g. 'B1'.")
    path: StrategyPath
    candidate: NonEmptyStr = Field(
        description="Model ref, service, or model family, e.g. 'microsoft/resnet-18'."
    )
    summary: NonEmptyStr
    tradeoffs: TradeoffRow
    evidence_refs: list[RecordRef] = Field(default_factory=list)


class RecommendationKind(StrEnum):
    NO_MODEL_NEEDED = "no_model_needed"
    USE_HOSTED_API = "use_hosted_api"
    REUSE_PRETRAINED = "reuse_pretrained"
    EVALUATE_PRETRAINED_THEN_FINETUNE_IF_NEEDED = "evaluate_pretrained_then_finetune_if_needed"
    FINE_TUNE = "fine_tune"
    TRAIN_TASK_SPECIFIC = "train_task_specific"
    INFEASIBLE = "infeasible"


class Recommendation(FrozenModel):
    kind: RecommendationKind
    option_ids: tuple[Identifier, ...] = Field(
        (), description="Options this recommendation selects, in order."
    )
    rationale: NonEmptyStr


class ProposedMeasurement(FrozenModel):
    """One candidate the Phase B plan would measure (spec §7.1 'proposed Phase B plan')."""

    option_id: Identifier
    trial_kind: TrialKind
    rationale: NonEmptyStr
    estimated_cost: CostEstimate


class ProposedPhaseBPlan(FrozenModel):
    measurements: tuple[ProposedMeasurement, ...] = Field(min_length=1)
    notes: str = ""

    @property
    def total_estimated_cost(self) -> CostEstimate:
        """Sum of the per-measurement estimates (computed, not stated by the LLM)."""
        costs = [m.estimated_cost for m in self.measurements]
        return CostEstimate(
            compute_seconds=sum(c.compute_seconds for c in costs),
            usd=sum(c.usd for c in costs),
            gpu=any(c.gpu for c in costs),
            assumptions=("sum of per-measurement estimates",),
        )


class Citation(StrictModel):
    ref: RecordRef = Field(description="e.g. 'passage:9', 'catalog:12'.")
    source_url: NonEmptyStr
    title: str | None = None
    revision: str | None = None
    retrieved_on: date | None = None


_NEEDS_NO_PLAN = frozenset({RecommendationKind.NO_MODEL_NEEDED, RecommendationKind.INFEASIBLE})


class StrategyReport(Record):
    """Options A-D compared with evidence, a recommendation, and a Phase B plan (spec §7.1)."""

    run_id: Identifier
    options: list[StrategyOption] = Field(min_length=1)
    recommendation: Recommendation
    proposed_phase_b_plan: ProposedPhaseBPlan | None = None
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        ids = [o.option_id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("option_id values must be unique")
        referenced = set(self.recommendation.option_ids)
        if self.proposed_phase_b_plan is not None:
            referenced |= {m.option_id for m in self.proposed_phase_b_plan.measurements}
        unknown = sorted(referenced - set(ids))
        if unknown:
            raise ValueError(f"unknown option_id references: {unknown}")
        if self.recommendation.kind not in _NEEDS_NO_PLAN and self.proposed_phase_b_plan is None:
            raise ValueError(
                f"recommendation {self.recommendation.kind.value!r} needs a proposed_phase_b_plan"
            )
        return self
