"""Output models of the MCP tools: typed, bounded summaries (spec §4.3 data firewall, layer 1).

No field can hold per-row data: lists are short and capped, strings are bounded, and every number
is a platform aggregate. `inspect_dataset` returns the approved `DatasetProfile` (S0.2/S1.1)
unchanged. These models are part of the frozen tool contract in docs/mcp/tools.json; they live
here rather than in `hypora/core/schemas/` because only the MCP boundary uses them (D-018).
"""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, StringConstraints

from hypora.core.schemas import (
    Family,
    Identifier,
    JobLimits,
    JobState,
    MetricDirection,
    MetricName,
    Operation,
    Sha256Hex,
    StrictModel,
)

Label = Annotated[str, StringConstraints(max_length=40)]
"""A class label, cut to 40 characters (the D-009 bound)."""

Line = Annotated[str, StringConstraints(max_length=600)]
Note = Annotated[str, StringConstraints(max_length=400)]


class WrittenFile(StrictModel):
    name: Literal["trial.yaml", "pipeline.py", "requirements.lock"]
    sha256: Sha256Hex
    bytes: NonNegativeInt


class SubmitDefaults(StrictModel):
    """The values `submit_experiment` uses when its optional arguments are omitted."""

    run_id: Identifier
    family: Family
    contract_version: PositiveInt
    approval_ref: Identifier
    limits: JobLimits = Field(description="The contract's default (and maximum) job limits.")


class TrialFilesResult(StrictModel):
    trial_id: Identifier = Field(
        description="Pass this to write_trial_files and submit_experiment."
    )
    created: bool = Field(description="True if this call created the trial.")
    trial_dir: Note = Field(description="The trial's own directory, relative to the repository.")
    files: list[WrittenFile] = Field(max_length=3, description="Files this call wrote.")
    present: list[Literal["trial.yaml", "pipeline.py", "requirements.lock"]] = Field(
        max_length=3, description="Files the trial now has."
    )
    submit_defaults: SubmitDefaults | None = Field(
        None, description="Null until the run has an approved experiment contract."
    )
    next_step: Note


class SubmitResult(StrictModel):
    job_id: Identifier = Field(description="Pass this to the get_experiment_* tools.")
    trial_id: Identifier
    state: JobState
    deduplicated: bool = Field(
        description="True if this idempotency_key was already used for this same submission: "
        "the existing job is returned and nothing new is queued."
    )
    operation: Operation
    contract_version: PositiveInt
    limits: JobLimits
    next_step: Note


class ExperimentError(StrictModel):
    code: Annotated[str, StringConstraints(max_length=64)]
    message: Note = Field(description="Platform-written; dataset values are never included.")


class ArtifactRef(StrictModel):
    artifact_id: Identifier
    kind: Literal["metrics", "predictions", "manifest"]
    sha256: Sha256Hex
    bytes: NonNegativeInt


class ExperimentStatus(StrictModel):
    job_id: Identifier
    trial_id: Identifier
    operation: Operation
    state: JobState
    attempts: NonNegativeInt
    finished: bool = Field(description="True once the state can no longer change.")
    succeeded: bool | None = Field(None, description="Null until finished.")
    error: ExperimentError | None = None
    log_tail: list[Line] = Field(
        max_length=20, description="Last log lines, after the data firewall."
    )
    withheld_log_lines: NonNegativeInt = Field(
        0, description="Tail lines withheld because they showed dataset values."
    )
    artifacts: list[ArtifactRef] = Field(max_length=3)
    next_step: Note


class MetricSummary(StrictModel):
    """One metric across validation folds."""

    mean: float
    std: NonNegativeFloat
    min: float
    max: float
    folds: PositiveInt = Field(description="Folds where the metric is defined.")


class GroupSummary(StrictModel):
    """Per-group scores of the primary metric, without group labels (they are data values)."""

    n_groups: NonNegativeInt
    mean: float | None
    worst: float | None
    best: float | None


class FoldScore(StrictModel):
    fold: NonNegativeInt
    rows: NonNegativeInt
    metrics: dict[MetricName, float | None]
    unavailable: dict[MetricName, Note] = Field(
        default_factory=dict, description="Why a metric is undefined on this fold."
    )
    confusion_matrix: list[list[NonNegativeInt]] | None = Field(
        None, max_length=50, description="Rows: true class; columns: predicted (class_labels)."
    )
    groups: GroupSummary | None = None


class ClassScore(StrictModel):
    label: Label
    precision: float
    recall: float
    f1: float
    support: NonNegativeInt = Field(description="Validation rows of this class, all folds.")


class ExperimentMetrics(StrictModel):
    job_id: Identifier
    trial_id: Identifier
    split: Literal["validation"]
    primary_metric: MetricName
    direction: MetricDirection
    primary: MetricSummary | None = Field(None, description="Null if undefined on every fold.")
    summary: dict[MetricName, MetricSummary]
    folds: list[FoldScore] = Field(max_length=20)
    class_labels: list[Label] | None = Field(None, max_length=50)
    per_class: list[ClassScore] = Field(default_factory=list, max_length=50)
    note: Note


class LogLine(StrictModel):
    attempt: NonNegativeInt
    stage: NonNegativeInt
    action: Annotated[str, StringConstraints(max_length=16)]
    line: NonNegativeInt = Field(description="Line number within the stage's log.")
    text: Line


class ExperimentLogs(StrictModel):
    job_id: Identifier
    page: PositiveInt
    page_size: PositiveInt
    total_lines: NonNegativeInt
    lines: list[LogLine] = Field(max_length=200)
    withheld_lines: NonNegativeInt = Field(
        description="Lines on this page withheld because they showed dataset values."
    )
    next_page: PositiveInt | None = None
