"""Bounded model-facing worker proposals and text-only reports (S3.4)."""

from pydantic import Field

from hypora.core.schemas import StrictModel, ToolName
from hypora.core.schemas.briefs import BriefBudget
from hypora.core.schemas.common import Confidence, RecordRef
from hypora.core.schemas.findings import Claim


class WorkerRequest(StrictModel):
    """C proposes work, never runtime identity or authority."""

    objective: str = Field(min_length=1, max_length=1000)
    end_condition: str = Field(min_length=1, max_length=500)
    allowed_tools: list[ToolName] = Field(min_length=1, max_length=6)
    context_refs: list[RecordRef] = Field(default_factory=list, max_length=5)
    budget: BriefBudget


class WorkerProposal(StrictModel):
    """An empty proposal means C works alone."""

    workers: list[WorkerRequest] = Field(default_factory=list, max_length=4)


class WorkerReport(StrictModel):
    """Worker narrative; the controller supplies trial and outcome facts."""

    summary: str = Field(min_length=1, max_length=2000)
    claims: list[Claim] = Field(default_factory=list, max_length=10)
    confidence: Confidence = 0.0
    open_questions: list[str] = Field(default_factory=list, max_length=10)
