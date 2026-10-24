"""Plans written before acting, and revised after failures (course F4; PLAN.md S3.H1, S3.H2)."""

from typing import Self

from pydantic import Field, PositiveInt, model_validator

from hypora.core.schemas.common import ActorRole, Identifier, NonEmptyStr, Record, StrictModel


class PlanStep(StrictModel):
    step: PositiveInt = Field(description="1-based position in the plan.")
    tool: str | None = Field(
        None,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Tool the step will call, or null for a reasoning/answer step.",
    )
    purpose: NonEmptyStr
    success_check: NonEmptyStr = Field(description="How the role will know the step worked.")
    depends_on: list[PositiveInt] = Field(default_factory=list)


class Plan(Record):
    """An ordered plan emitted before a role's first tool call; replans are new revisions."""

    plan_id: Identifier
    run_id: Identifier
    role: ActorRole
    revision: PositiveInt = 1
    revises: Identifier | None = Field(None, description="plan_id of the plan this replaces.")
    revision_reason: str | None = Field(
        None, description="The failure observation that triggered the replan."
    )
    goal: NonEmptyStr
    steps: list[PlanStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self) -> Self:
        if (self.revision > 1) != (self.revises is not None):
            raise ValueError("revises is required for revisions after the first and only them")
        if self.revises is not None and not self.revision_reason:
            raise ValueError("a revised plan must state its revision_reason")
        numbers = [s.step for s in self.steps]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("steps must be numbered 1..n in order")
        for s in self.steps:
            if any(d >= s.step for d in s.depends_on):
                raise ValueError(f"step {s.step} may only depend on earlier steps")
        return self
