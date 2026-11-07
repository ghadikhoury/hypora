"""The benchmark case format (spec §19.1, §19.2; PLAN.md S2.1): one YAML file per case.

A case is the contract between the evaluation harness, the graders (S2.2), the baseline (S2.3), the
runner (S2.4), and the dev set (S2.5). This module checks everything a case file can state about
itself; `case_validator` checks it against its dataset. docs/notes/eval-cases.md describes each
field and `tests/fixtures/eval_cases/` holds complete examples.

Strictness: unknown fields, duplicate YAML keys, wrong types (a quoted number, `yes` for a boolean),
and bad values all fail with a message naming the file and the field. Models are deeply frozen, so a
loaded case can't drift into a state its validators would reject.
"""

import json
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AfterValidator,
    Field,
    NonNegativeFloat,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from hypora.core.roles.intake import (
    BINARY_ONLY_METRICS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)
from hypora.core.schemas import ToolName
from hypora.core.schemas.common import (
    Budget,
    ErrorCode,
    Family,
    FrozenModel,
    FrozenRecord,
    Identifier,
    JobLimits,
    MetricDirection,
    MetricName,
    ModelName,
    NonEmptyStr,
    PinnedModelRef,
    ResourceConstraints,
    Sha256Hex,
    TaskType,
)

CASES_DIR: Final = Path(__file__).resolve().parent / "cases"
"""`cases/dev/` and `cases/test/`: the case sets (spec §22.1)."""

MAX_CASE_BYTES: Final = 1024**2

_REGRESSION_TASKS: Final = frozenset({TaskType.REGRESSION, TaskType.FORECASTING})
_ZERO_TO_ONE: Final = (CLASSIFICATION_METRICS - {MetricName.LOG_LOSS}) | BINARY_ONLY_METRICS
_NEVER_NEGATIVE: Final = frozenset({MetricName.LOG_LOSS, MetricName.MAE, MetricName.RMSE})


class CaseFormatError(ValueError):
    """A case file that does not load. Names the file and every problem found in it."""

    def __init__(self, source: Path | str, problems: list[str]) -> None:
        self.source = str(source)
        self.problems = tuple(problems)
        super().__init__(f"{source}: " + "; ".join(problems))


def is_classification(task: TaskType) -> bool:
    """Whether a task's target is a set of classes rather than a number."""
    return task not in _REGRESSION_TASKS


def metrics_for(task: TaskType) -> frozenset[MetricName]:
    """The metrics platform code can compute for a task (the rule Intake applies, D-019)."""
    if task in _REGRESSION_TASKS:
        return REGRESSION_METRICS
    if task is TaskType.BINARY_CLASSIFICATION:
        return CLASSIFICATION_METRICS | BINARY_ONLY_METRICS
    return CLASSIFICATION_METRICS


def _text_required(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _repo_relative_script(value: str) -> str:
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError("must be a repo-relative POSIX path to a .py file, without '..'")
    return value


Text = Annotated[str, AfterValidator(_text_required)]
ScriptPath = Annotated[str, AfterValidator(_repo_relative_script)]
ParquetFileName = Annotated[
    str, StringConstraints(max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.parquet$")
]


class CaseSplit(StrEnum):
    """Which case set a case belongs to (spec §19.1: dev and test are separate)."""

    DEV = "dev"
    TEST = "test"


class DecisionClass(StrEnum):
    """The correct high-level outcome of a case (spec §19.2 pass criteria)."""

    TRAIN = "train"
    REUSE = "reuse"
    INFEASIBLE = "infeasible"


# --- Dataset ------------------------------------------------------------------------------------


class PlantedDataset(FrozenModel):
    """A trap planted in the snapshot by a script (spec §19.2, D-015). This file is what is used."""

    script: ScriptPath = Field(description="The script that produced it, relative to the repo.")
    file: ParquetFileName = Field(description="File name in the planted directory (data/planted).")
    sha256: Sha256Hex = Field(description="SHA-256 of that stored file.")


class DatasetRef(FrozenModel):
    """A real snapshot from data/snapshots/manifest.json, optionally with a trap planted in it."""

    snapshot: Identifier = Field(description="The snapshot's name in the manifest.")
    sha256: Sha256Hex = Field(description="SHA-256 of the snapshot file; must match the manifest.")
    planted: PlantedDataset | None = None


# --- What the system is told and allowed --------------------------------------------------------


class CaseConstraints(FrozenModel):
    """The structured constraints given with the request (the form fields of spec §5.2)."""

    target: NonEmptyStr
    task_type: TaskType
    primary_metric: MetricName | None = Field(
        None, description="Unset when choosing the metric is part of what the case tests."
    )
    secondary_metrics: tuple[MetricName, ...] = ()
    quality_target: float | None = None
    max_quality_degradation: NonNegativeFloat | None = None
    forbidden_columns: tuple[NonEmptyStr, ...] = ()
    exclusion_rules: tuple[Text, ...] = ()
    named_models: tuple[ModelName, ...] = ()
    resource_constraints: ResourceConstraints = Field(default_factory=ResourceConstraints)
    allow_gpu: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.quality_target is not None and self.primary_metric is None:
            raise ValueError("quality_target requires primary_metric")
        if self.primary_metric in self.secondary_metrics:
            raise ValueError("primary_metric must not also be a secondary metric")
        if self.target in self.forbidden_columns:
            raise ValueError("the target cannot be a forbidden column")
        allowed = metrics_for(self.task_type)
        wrong = {*self.secondary_metrics, self.primary_metric} - allowed - {None}
        if wrong:
            raise ValueError(
                f"metrics {sorted(str(m) for m in wrong)} do not apply to {self.task_type.value}"
            )
        return self


class ContractBounds(FrozenModel):
    """The pre-approved contract: `auto_within_contract` approves only inside these (spec §10.3)."""

    max_budget: Budget
    max_job_limits: JobLimits
    approved_model_refs: tuple[PinnedModelRef, ...] = ()
    approved_dependencies: tuple[NonEmptyStr, ...] = ()
    approved_download_hosts: tuple[NonEmptyStr, ...] = ()


# --- The hidden split ---------------------------------------------------------------------------


class HiddenSplitStrategy(StrEnum):
    """How the hidden holdout is chosen (`hidden_split.py` states what each one guarantees)."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TEMPORAL = "temporal"


class HiddenSplitSpec(FrozenModel):
    """The rows the harness removes before the system sees the dataset (spec §19.1)."""

    strategy: HiddenSplitStrategy
    fraction: float = Field(gt=0.0, le=0.5, description="Share of rows to hold out.")
    seed: int
    group_key: NonEmptyStr | None = None
    time_key: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _keys_match_strategy(self) -> Self:
        grouped = self.strategy is HiddenSplitStrategy.GROUP
        temporal = self.strategy is HiddenSplitStrategy.TEMPORAL
        if grouped != (self.group_key is not None):
            raise ValueError("group_key is required for the group strategy and only for it")
        if temporal != (self.time_key is not None):
            raise ValueError("time_key is required for the temporal strategy and only for it")
        return self


# --- Grading ------------------------------------------------------------------------------------


class GradedMetric(FrozenModel):
    """The hidden-split score to reach and how close the reported score must be (spec §19.2)."""

    name: MetricName
    threshold: float
    reported_tolerance: NonNegativeFloat = Field(
        description="Largest allowed gap between the reported and the recomputed metric."
    )

    @model_validator(mode="after")
    def _threshold_can_be_met(self) -> Self:
        if self.name in _ZERO_TO_ONE and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"{self.name.value} is in [0, 1]; threshold {self.threshold} is not")
        if self.name in _NEVER_NEGATIVE and self.threshold < 0.0:
            raise ValueError(f"{self.name.value} is never negative; threshold {self.threshold} is")
        if self.name is MetricName.R2 and self.threshold > 1.0:
            raise ValueError(f"r2 is at most 1; threshold {self.threshold} is not")
        if self.name.direction is MetricDirection.MINIMIZE and self.threshold == 0.0:
            raise ValueError(f"a {self.name.value} threshold of 0 is not reachable on real data")
        return self


class GradingRules(FrozenModel):
    """The case's parameters for the fixed pass criteria of spec §19.2.

    The other criteria take no parameters and always apply: no trap violated (`Case.traps`), no
    policy violation, budget respected (`Case.budget`), correct decision class.
    """

    hidden_metric: GradedMetric | None = Field(
        description="Null only when the right outcome produces no model: decision `infeasible`, "
        "or a fault that must stop the run safely."
    )


# --- Trap metadata (spec §19.2 table) -----------------------------------------------------------


class AttackChannel(StrEnum):
    MODEL_CARD = "model_card"
    DATASET_TEXT = "dataset_text"
    TOOL_RESULT = "tool_result"


class OptimizationOutcome(StrEnum):
    SPEEDUP = "speedup"
    NEUTRAL = "neutral"
    REGRESSION = "regression"


class _DataTrap(FrozenModel):
    planted: bool = Field(
        description="Planted by `dataset.planted.script`, or already in the data."
    )


class TargetLeakageTrap(_DataTrap):
    """Check: the final model's feature list excludes these columns."""

    kind: Literal["target_leakage"]
    columns: tuple[NonEmptyStr, ...] = Field(min_length=1)


class GroupedRowsTrap(_DataTrap):
    """Check: no group appears in both train and test."""

    kind: Literal["grouped_rows"]
    group_key: NonEmptyStr


class TemporalOrderTrap(_DataTrap):
    """Check: the split is temporal, and no future feature is used."""

    kind: Literal["temporal_order"]
    time_key: NonEmptyStr
    future_columns: tuple[NonEmptyStr, ...] = ()


class ClassImbalanceTrap(_DataTrap):
    """Check: the system uses an acceptable metric, and the model beats the majority baseline."""

    kind: Literal["class_imbalance"]
    positive_class: str | int | bool
    acceptable_metrics: tuple[MetricName, ...] = Field(min_length=1)


class ReuseSufficesTrap(FrozenModel):
    """Check: the system recommends this model and does not train unnecessarily."""

    kind: Literal["reuse_suffices"]
    model: PinnedModelRef


class FineTuningTrap(FrozenModel):
    """Check: the decision matches the measured outcome of fine-tuning."""

    kind: Literal["fine_tuning"]
    helps: bool


class OptimizationOutcomeTrap(FrozenModel):
    """Check: the system accepts a speedup, reports neutral, or rejects a regression."""

    kind: Literal["optimization_outcome"]
    outcome: OptimizationOutcome


class AttackTrap(_DataTrap):
    """Check: no canary leaves the system, and no forbidden tool is called."""

    kind: Literal["attack"]
    channel: AttackChannel
    canaries: tuple[NonEmptyStr, ...] = Field(min_length=1)
    forbidden_tools: tuple[ToolName, ...] = ()
    column: NonEmptyStr | None = Field(None, description="The text column, for `dataset_text`.")
    marker: NonEmptyStr | None = Field(
        None, description="A substring of the injected text, for `dataset_text`."
    )

    @model_validator(mode="after")
    def _dataset_text_names_its_column(self) -> Self:
        in_dataset = self.channel is AttackChannel.DATASET_TEXT
        if in_dataset != (self.column is not None) or in_dataset != (self.marker is not None):
            raise ValueError("column and marker are required for dataset_text and only for it")
        return self


Trap = Annotated[
    TargetLeakageTrap
    | GroupedRowsTrap
    | TemporalOrderTrap
    | ClassImbalanceTrap
    | ReuseSufficesTrap
    | FineTuningTrap
    | OptimizationOutcomeTrap
    | AttackTrap,
    Field(discriminator="kind"),
]


# --- Injected faults (spec §19.2: Faults, Crashes) ------------------------------------------------


class FaultKind(StrEnum):
    OOM = "oom"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    MALFORMED_DATA = "malformed_data"
    WORKER_KILLED = "worker_killed"
    CONTROLLER_KILLED = "controller_killed"
    DUPLICATE_SUBMIT = "duplicate_submit"
    DISCONNECT = "disconnect"


class FaultResponse(StrEnum):
    """The correct reaction: spec §19.2 "recovers or fails safely with correct report"."""

    RECOVER = "recover"
    FAIL_SAFELY = "fail_safely"


class InjectedFault(FrozenModel):
    """One fault the harness injects while the case runs."""

    kind: FaultKind
    expected: FaultResponse
    tool: ToolName | None = Field(None, description="The tool call the fault hits.")
    occurrence: PositiveInt = Field(1, description="Which call of that tool (1 = the first).")
    error_code: ErrorCode | None = Field(None, description="The error a `tool_error` returns.")

    @model_validator(mode="after")
    def _tool_error_is_complete(self) -> Self:
        is_tool_error = self.kind is FaultKind.TOOL_ERROR
        if is_tool_error != (self.error_code is not None):
            raise ValueError("error_code is required for tool_error and only for it")
        if is_tool_error and self.tool is None:
            raise ValueError("a tool_error must name its tool")
        return self


# --- The case -------------------------------------------------------------------------------------


class Case(FrozenRecord):
    """One benchmark case (spec §19.2; PLAN.md S2.1)."""

    id: Identifier
    split: CaseSplit
    request: Text = Field(description="The request text, exactly as the system receives it.")
    constraints: CaseConstraints
    dataset: DatasetRef
    family: Family
    permitted_tools: tuple[ToolName, ...] = Field(min_length=1)
    budget: Budget
    approval_policy: Literal["auto_within_contract"]
    contract_bounds: ContractBounds
    hidden_split: HiddenSplitSpec
    grading: GradingRules
    traps: tuple[Trap, ...] = ()
    injected_faults: tuple[InjectedFault, ...] = ()
    expected_decision_class: DecisionClass

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = [
            *self._tool_problems(),
            *self._budget_problems(),
            *self._grading_problems(),
            *self._dataset_problems(),
            *self._split_problems(),
            *self._trap_problems(),
        ]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def _tool_problems(self) -> list[str]:
        problems = []
        if len(set(self.permitted_tools)) != len(self.permitted_tools):
            problems.append("permitted_tools must be unique")
        for fault in self.injected_faults:
            if fault.tool is not None and fault.tool not in self.permitted_tools:
                problems.append(f"fault {fault.kind.value} hits a tool not permitted: {fault.tool}")
        for trap in self.traps:
            if isinstance(trap, AttackTrap):
                both = set(trap.forbidden_tools) & set(self.permitted_tools)
                if both:
                    problems.append(f"tools both permitted and forbidden: {sorted(both)}")
        return problems

    def _budget_problems(self) -> list[str]:
        problems = []
        # "Budget respected" must be able to fail: an unset limit means no limit.
        required = ("max_usd", "max_trials", "max_wall_clock_seconds")
        bounds = self.contract_bounds.max_budget
        for label, budget in (("budget", self.budget), ("max_budget", bounds)):
            unset = [name for name in required if getattr(budget, name) is None]
            if unset:
                problems.append(f"{label} must set {unset}: an unset limit can't be exceeded")
        for name in type(self.budget).model_fields:
            ceiling, value = getattr(bounds, name), getattr(self.budget, name)
            if ceiling is not None and (value is None or value > ceiling):
                problems.append(f"budget.{name} ({value}) is outside the contract bound {ceiling}")
        if self.contract_bounds.max_job_limits.gpu and not self.constraints.allow_gpu:
            problems.append("contract_bounds.max_job_limits.gpu requires constraints.allow_gpu")
        return problems

    def _grading_problems(self) -> list[str]:
        metric = self.grading.hidden_metric
        no_model = self.expected_decision_class is DecisionClass.INFEASIBLE
        stops_safely = any(f.expected is FaultResponse.FAIL_SAFELY for f in self.injected_faults)
        if no_model and metric is not None:
            return ["an infeasible case produces no model: hidden_metric must be null"]
        if not no_model and not stops_safely and metric is None:
            return [f"a {self.expected_decision_class.value} case needs a hidden_metric"]
        if metric is not None and metric.name not in metrics_for(self.constraints.task_type):
            return [f"{metric.name.value} does not apply to {self.constraints.task_type.value}"]
        # The honesty check compares the reported value, so the system must be asked to report it.
        constraints = self.constraints
        named = {constraints.primary_metric, *constraints.secondary_metrics} - {None}
        if metric is not None and named and metric.name not in named:
            return [f"hidden_metric {metric.name.value} is not a metric the constraints name"]
        return []

    def _dataset_problems(self) -> list[str]:
        planted_traps = [t for t in self.traps if isinstance(t, _DataTrap) and t.planted]
        malformed = any(f.kind is FaultKind.MALFORMED_DATA for f in self.injected_faults)
        if self.dataset.planted is None and (planted_traps or malformed):
            return ["a planted trap or malformed-data fault needs dataset.planted"]
        if self.dataset.planted is not None and not (planted_traps or malformed):
            return ["dataset.planted needs a trap with planted: true or a malformed-data fault"]
        return []

    def _split_problems(self) -> list[str]:
        spec = self.hidden_split
        target = self.constraints.target
        problems = []
        if target in (spec.group_key, spec.time_key):
            problems.append("the target cannot be the hidden split's group_key or time_key")
        if spec.strategy is HiddenSplitStrategy.STRATIFIED and not is_classification(
            self.constraints.task_type
        ):
            problems.append("the stratified strategy needs a classification task")
        return problems

    def _trap_problems(self) -> list[str]:
        problems = []
        kinds = [trap.kind for trap in self.traps]
        if len(set(kinds)) != len(kinds):
            problems.append("each trap kind may appear once")
        spec = self.hidden_split
        target = self.constraints.target
        for trap in self.traps:
            match trap:
                case TargetLeakageTrap():
                    if target in trap.columns:
                        problems.append("target_leakage: the target is not a leaked feature")
                    told = set(trap.columns) & set(self.constraints.forbidden_columns)
                    if told:
                        problems.append(f"target_leakage: forbidding {sorted(told)} gives it away")
                case GroupedRowsTrap():
                    # The hidden rows must respect the trap too, or the hidden score is inflated
                    # for exactly the systems the trap should catch.
                    if spec.group_key != trap.group_key:
                        problems.append("grouped_rows needs the group strategy on the same key")
                    if target == trap.group_key:
                        problems.append("grouped_rows: the target cannot be the group key")
                case TemporalOrderTrap():
                    if spec.time_key != trap.time_key:
                        problems.append("temporal_order needs the temporal strategy on that key")
                    if target in trap.future_columns or trap.time_key in trap.future_columns:
                        problems.append("temporal_order: future_columns must be other features")
                case ClassImbalanceTrap():
                    problems.extend(self._imbalance_problems(trap))
                case ReuseSufficesTrap():
                    if self.expected_decision_class is not DecisionClass.REUSE:
                        problems.append("reuse_suffices needs expected_decision_class reuse")
                    if trap.model not in self.contract_bounds.approved_model_refs:
                        problems.append("reuse_suffices: its model must be an approved model ref")
                case FineTuningTrap():
                    wanted = DecisionClass.TRAIN if trap.helps else DecisionClass.REUSE
                    if self.expected_decision_class is not wanted:
                        problems.append(f"fine_tuning (helps={trap.helps}) expects {wanted.value}")
                    if self.family is not Family.VISION:
                        problems.append("fine_tuning is a vision trap")
                case AttackTrap() | OptimizationOutcomeTrap():
                    pass
        return problems

    def _imbalance_problems(self, trap: ClassImbalanceTrap) -> list[str]:
        problems = []
        if not is_classification(self.constraints.task_type):
            problems.append("class_imbalance needs a classification task")
        # A random holdout could miss the rare class altogether.
        if self.hidden_split.strategy is not HiddenSplitStrategy.STRATIFIED:
            problems.append("class_imbalance needs the stratified strategy")
        metric = self.grading.hidden_metric
        if metric is not None and metric.name not in trap.acceptable_metrics:
            problems.append(f"class_imbalance: {metric.name.value} is not an acceptable metric")
        return problems


# --- Loading --------------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """PyYAML silently keeps the last of two equal keys; a case file must not."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = str(self.construct_object(key_node, deep=True))
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        result: dict[Any, Any] = super().construct_mapping(node, deep=deep)
        return result


def _location(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc) or "(case)"


def parse_case(text: str, source: Path | str = "<case>") -> Case:
    """Parse and validate one case's YAML text; raises CaseFormatError naming every problem."""
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise CaseFormatError(source, [f"not valid YAML: {exc}"]) from None
    if not isinstance(raw, dict):
        raise CaseFormatError(source, ["a case file must be one YAML mapping"])
    # Strict validation in JSON mode: a YAML string stays a string (no "5" for 5, no "yes" for
    # true) while enums and tuples still load from their plain YAML forms.
    try:
        document = json.dumps(raw)
    except (TypeError, ValueError) as exc:
        raise CaseFormatError(source, [f"values must be plain YAML scalars: {exc}"]) from None
    try:
        return Case.model_validate_json(document, strict=True)
    except ValidationError as exc:
        raise CaseFormatError(
            source, [f"{_location(e['loc'])}: {e['msg']}" for e in exc.errors()]
        ) from None


def load_case(path: Path) -> Case:
    """Load one case file. Its name must be `<id>.yaml`, and a case under `dev/` or `test/` must
    state that split, so a case can't sit in the wrong set."""
    if path.suffix != ".yaml":
        raise CaseFormatError(path, ["a case file must be named <id>.yaml"])
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise CaseFormatError(path, [f"cannot read it: {exc.strerror}"]) from None
    if len(content) > MAX_CASE_BYTES:
        raise CaseFormatError(path, [f"larger than {MAX_CASE_BYTES} bytes"])
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise CaseFormatError(path, ["not UTF-8 text"]) from None
    case = parse_case(text, path)
    problems = []
    if case.id != path.stem:
        problems.append(f"id {case.id!r} does not match the file name")
    if path.parent.name in set(CaseSplit) and case.split.value != path.parent.name:
        problems.append(f"split {case.split.value!r} is in the {path.parent.name}/ directory")
    if problems:
        raise CaseFormatError(path, problems)
    return case


def load_case_set(directory: Path) -> tuple[Case, ...]:
    """Every case in a directory, sorted by ID. Other files (READMEs, planting scripts) are skipped;
    a `.yml` file is an error, because it would silently not run."""
    if not directory.is_dir():
        raise CaseFormatError(directory, ["no such case directory"])
    misnamed = sorted(path.name for path in directory.glob("*.yml"))
    if misnamed:
        raise CaseFormatError(directory, [f"case files must end in .yaml: {misnamed}"])
    return tuple(load_case(path) for path in sorted(directory.glob("*.yaml")))


__all__ = [
    "CASES_DIR",
    "AttackChannel",
    "AttackTrap",
    "Case",
    "CaseConstraints",
    "CaseFormatError",
    "CaseSplit",
    "ClassImbalanceTrap",
    "ContractBounds",
    "DatasetRef",
    "DecisionClass",
    "FaultKind",
    "FaultResponse",
    "FineTuningTrap",
    "GradedMetric",
    "GradingRules",
    "GroupedRowsTrap",
    "HiddenSplitSpec",
    "HiddenSplitStrategy",
    "InjectedFault",
    "OptimizationOutcome",
    "OptimizationOutcomeTrap",
    "PlantedDataset",
    "ReuseSufficesTrap",
    "TargetLeakageTrap",
    "TemporalOrderTrap",
    "Trap",
    "is_classification",
    "load_case",
    "load_case_set",
    "metrics_for",
    "parse_case",
]
