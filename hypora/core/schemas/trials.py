"""Trials: the design shared by proposals, `TrialSpec` records, and `trial.yaml` (spec §8.1)."""

from enum import StrEnum
from typing import Final, Self

from pydantic import Field, JsonValue, PositiveInt, model_validator

from hypora.core.schemas.common import (
    CostEstimate,
    Identifier,
    NonEmptyStr,
    Operation,
    PinnedModelRef,
    Record,
    StrictModel,
    TrialKind,
)

# The trial directory contract (spec §8.1, §6.6): runs/{run_id}/trials/{trial_id}/
TRIAL_DIR_TEMPLATE: Final = "runs/{run_id}/trials/{trial_id}"
TRIAL_INPUT_FILES: Final = ("trial.yaml", "pipeline.py", "requirements.lock")
"""Written by the trial worker via `write_trial_files`."""
TRIAL_OUTPUT_FILES: Final = ("metrics.json", "predictions.parquet", "manifest.yaml")
"""Written by the platform harness, never by trial code."""


class ShortBudget(StrictModel):
    """Reduced budget for a successive-halving round (spec §8.4)."""

    max_epochs: PositiveInt | None = None
    data_fraction: float | None = Field(None, gt=0.0, le=1.0)
    max_wall_seconds: PositiveInt | None = None

    @model_validator(mode="after")
    def _something_is_reduced(self) -> Self:
        if self.max_epochs is None and self.data_fraction is None and self.max_wall_seconds is None:
            raise ValueError("a short budget must reduce epochs, data, or wall time")
        return self


_OPERATION_FOR_KIND = {
    TrialKind.BASELINE: frozenset({Operation.FIT}),
    TrialKind.CANDIDATE: frozenset({Operation.FIT}),
    TrialKind.EVAL_AS_IS: frozenset({Operation.EVAL_AS_IS}),
    TrialKind.FINETUNE: frozenset({Operation.FINETUNE}),
    TrialKind.OPTIMIZATION: frozenset({Operation.OPTIMIZE, Operation.PROFILE}),
}


class TrialDesign(StrictModel):
    """One hypothesis (spec §8.1 `trial.yaml`: hypothesis, model family, config, expected effect,
    estimated cost), plus the operation and model it needs."""

    kind: TrialKind
    operation: Operation
    hypothesis: NonEmptyStr
    model_family: NonEmptyStr = Field(description="e.g. 'hist_gradient_boosting', 'resnet18'.")
    model_ref: PinnedModelRef | None = Field(None, description="Pretrained weights, pinned.")
    parent_trial_id: Identifier | None = Field(None, description="Set for optimization variants.")
    config: dict[str, JsonValue] = Field(default_factory=dict)
    expected_effect: NonEmptyStr
    estimated_cost: CostEstimate
    short_budget: ShortBudget | None = None

    @model_validator(mode="after")
    def _kind_is_consistent(self) -> Self:
        if self.operation not in _OPERATION_FOR_KIND[self.kind]:
            raise ValueError(f"operation {self.operation.value!r} invalid for {self.kind.value!r}")
        needs_model = self.kind in (TrialKind.EVAL_AS_IS, TrialKind.FINETUNE)
        if needs_model and self.model_ref is None:
            raise ValueError(f"{self.kind.value!r} trials need a pinned model_ref")
        if (self.kind is TrialKind.OPTIMIZATION) != (self.parent_trial_id is not None):
            raise ValueError("parent_trial_id is required for optimization trials and only them")
        return self


class ProposedTrial(TrialDesign):
    """A trial inside a Lead's TrialBatchProposal, before the controller assigns IDs."""


class TrialSpec(Record, TrialDesign):
    """A trial as recorded by the controller (`trials` table, spec §15.1)."""

    trial_id: Identifier
    run_id: Identifier
    batch_id: Identifier
    contract_version: PositiveInt

    @property
    def dir(self) -> str:
        """The trial's own directory (the only place its worker may write; spec §6.6)."""
        return TRIAL_DIR_TEMPLATE.format(run_id=self.run_id, trial_id=self.trial_id)

    def to_trial_file(self) -> "TrialFile":
        """The `trial.yaml` content for this trial."""
        design = self.model_dump(include=set(TrialDesign.model_fields))
        return TrialFile.model_validate({"trial_id": self.trial_id, **design})


class TrialFile(Record, TrialDesign):
    """Schema of `trial.yaml` (spec §8.1)."""

    trial_id: Identifier


class OptimizationMethod(StrEnum):
    """Inference optimizations (spec §18.1-18.3, PLAN.md S3.C5)."""

    THREAD_SETTINGS = "thread_settings"
    BATCH_INFERENCE = "batch_inference"
    REDUNDANT_TRANSFORM_REMOVAL = "redundant_transform_removal"
    SERIALIZATION_SIZE = "serialization_size"
    EXPORT_RUNTIME = "export_runtime"
    TORCH_COMPILE = "torch_compile"
    FP16 = "fp16"
    BF16 = "bf16"
    QUANTIZATION = "quantization"
    INPUT_PIPELINE = "input_pipeline"
    SMALLER_ARCHITECTURE = "smaller_architecture"
    CONTEXT_LENGTH = "context_length"
    CROSS_SERIES_BATCHING = "cross_series_batching"


class OptimizationOutcome(StrEnum):
    IMPROVED = "improved"
    NEUTRAL = "neutral"
    REGRESSED = "regressed"
    REJECTED = "rejected"
    FAILED = "failed"
