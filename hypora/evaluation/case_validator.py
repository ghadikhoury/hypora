"""Case validator (PLAN.md S2.1): a loaded case against its dataset and its trap metadata.

`case.py` checks what a case file states about itself. This checks that it is true of the data: the
dataset hash against the manifest, the columns and values traps refer to, that the hidden split can
be drawn, and that each trap's evidence survives it, since the system only sees the visible rows.
Every problem is reported at once. A dataset that can't be loaded is itself a problem, so a case can
never pass with its data checks skipped.

It checks that a trap's metadata fits the data, not that the trap is effective: S2.5's generation
scripts and its review verify that a planted trap really works (`REVIEW FOCUS`).
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from hypora.core.schemas.common import TaskType
from hypora.evaluation.case import (
    AttackTrap,
    Case,
    CaseFormatError,
    ClassImbalanceTrap,
    GroupedRowsTrap,
    TargetLeakageTrap,
    TemporalOrderTrap,
    is_classification,
    load_case,
)
from hypora.evaluation.case_data import (
    PLANTED_DIR,
    CaseDataset,
    CaseDatasetError,
    load_case_dataset,
)
from hypora.evaluation.hidden_split import (
    HiddenSplitError,
    column_tokens,
    select_hidden_rows,
    time_values,
    value_token,
)
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR

MAX_POSITIVE_SHARE = 0.3
"""A class imbalance trap's positive class is at most this share of the rows. Inclusive, so the
classic 70/30 real datasets (credit-g) qualify."""


@dataclass(frozen=True)
class CaseIssue:
    """One problem: a stable code for tests and tooling, and a message for the case's author."""

    code: str
    message: str


@dataclass(frozen=True)
class CaseReport:
    """The validator's verdict on one case."""

    case_id: str
    issues: tuple[CaseIssue, ...]

    @property
    def ok(self) -> bool:
        """True only if every check ran and passed."""
        return not self.issues


def _target_issues(frame: pd.DataFrame, case: Case) -> list[CaseIssue]:
    target = case.constraints.target
    if target not in frame.columns:
        return [CaseIssue("TARGET_MISSING", f"target column {target!r} is not in the dataset")]
    labels = frame[target].dropna()
    task = case.constraints.task_type
    numeric = pd.api.types.is_numeric_dtype(labels) and not pd.api.types.is_bool_dtype(labels)
    n_classes = len(set(column_tokens(labels)))
    if is_classification(task):
        if n_classes < 2:
            return [CaseIssue("TARGET_TYPE", f"{target!r} has fewer than two classes")]
        if task is TaskType.BINARY_CLASSIFICATION and n_classes != 2:
            return [CaseIssue("TARGET_TYPE", f"{target!r} has {n_classes} classes, not two")]
    elif not numeric:
        return [CaseIssue("TARGET_TYPE", f"{target!r} is not numeric: cannot be {task.value}")]
    return []


def _column_issues(frame: pd.DataFrame, case: Case) -> list[CaseIssue]:
    spec = case.hidden_split
    named: dict[str, tuple[str, ...]] = {
        "constraints.forbidden_columns": case.constraints.forbidden_columns,
        "hidden_split.group_key": (spec.group_key,) if spec.group_key else (),
        "hidden_split.time_key": (spec.time_key,) if spec.time_key else (),
    }
    for trap in case.traps:
        match trap:
            case TargetLeakageTrap():
                named[f"traps.{trap.kind}.columns"] = trap.columns
            case GroupedRowsTrap():
                named[f"traps.{trap.kind}.group_key"] = (trap.group_key,)
            case TemporalOrderTrap():
                named[f"traps.{trap.kind}.columns"] = (trap.time_key, *trap.future_columns)
            case AttackTrap() if trap.column is not None:
                named[f"traps.{trap.kind}.column"] = (trap.column,)
    return [
        CaseIssue("COLUMN_MISSING", f"{field}: column {name!r} is not in the dataset")
        for field, names in named.items()
        for name in names
        if name not in frame.columns
    ]


def _imbalance_issues(frame: pd.DataFrame, case: Case, trap: ClassImbalanceTrap) -> list[CaseIssue]:
    target = case.constraints.target
    if target not in frame.columns:
        return []  # reported as TARGET_MISSING
    counts = Counter(column_tokens(frame[target].dropna()))
    positive = value_token(trap.positive_class)
    if positive not in counts:
        return [
            CaseIssue("TRAP_DATA", f"class_imbalance: no {trap.positive_class!r} in {target!r}")
        ]
    share = counts[positive] / len(frame)
    if counts[positive] != min(counts.values()) or share > MAX_POSITIVE_SHARE:
        return [
            CaseIssue(
                "TRAP_DATA",
                f"class_imbalance: {trap.positive_class!r} is {share:.1%} of the rows; it must be "
                f"the rarest class and at most {MAX_POSITIVE_SHARE:.0%}",
            )
        ]
    return []


def _trap_issues(frame: pd.DataFrame, case: Case) -> list[CaseIssue]:
    issues: list[CaseIssue] = []
    for trap in case.traps:
        match trap:
            case GroupedRowsTrap() if trap.group_key in frame.columns:
                sizes = frame[trap.group_key].value_counts(dropna=True)
                if len(sizes) < 2 or sizes.max() < 2:
                    issues.append(
                        CaseIssue(
                            "TRAP_DATA",
                            f"grouped_rows: {trap.group_key!r} has no repeated groups, so the "
                            "trap is not there",
                        )
                    )
            case TemporalOrderTrap() if trap.time_key in frame.columns:
                try:
                    if time_values(frame[trap.time_key]).nunique() < 2:
                        issues.append(
                            CaseIssue(
                                "TRAP_DATA", f"temporal_order: {trap.time_key!r} never changes"
                            )
                        )
                except HiddenSplitError as exc:
                    issues.append(CaseIssue("TRAP_DATA", f"temporal_order: {exc}"))
            case ClassImbalanceTrap():
                issues.extend(_imbalance_issues(frame, case, trap))
            case AttackTrap() if trap.column is not None and trap.column in frame.columns:
                assert trap.marker is not None  # the model requires it together with `column`
                text = frame[trap.column].dropna().astype(str)
                if not text.str.contains(trap.marker, regex=False).any():
                    issues.append(
                        CaseIssue(
                            "TRAP_DATA",
                            f"attack: no value of {trap.column!r} holds its marker, so nothing "
                            "was injected",
                        )
                    )
    return issues


def _file_issues(case: Case, repo_root: Path) -> list[CaseIssue]:
    planted = case.dataset.planted
    if planted is not None and not (repo_root / planted.script).is_file():
        return [CaseIssue("PLANT_SCRIPT", f"generation script {planted.script} does not exist")]
    return []


def _varies(column: pd.Series) -> bool:
    """Whether a column can tell rows apart. Missing counts as a value: it can leak too."""
    return len(set(column_tokens(column))) >= 2


def _rare_issues(column: pd.Series, trap: ClassImbalanceTrap, where: str) -> list[CaseIssue]:
    tokens = Counter(column_tokens(column))
    count, total = tokens[value_token(trap.positive_class)], sum(tokens.values())
    if count == 0:
        return [
            CaseIssue(
                "TRAP_HOLDOUT", f"class_imbalance: {where} hold no {trap.positive_class!r} at all"
            )
        ]
    # Within one row of the limit: a holdout rounds, and a source exactly at it must not fail.
    if (count - 1) / max(total - 1, 1) > MAX_POSITIVE_SHARE:
        return [
            CaseIssue(
                "TRAP_HOLDOUT",
                f"class_imbalance: {trap.positive_class!r} is {count / total:.1%} of {where}, so "
                f"it isn't rare there (limit {MAX_POSITIVE_SHARE:.0%})",
            )
        ]
    return []


def _holdout_issues(visible: pd.DataFrame, hidden: pd.DataFrame, case: Case) -> list[CaseIssue]:
    """Whether each trap's evidence survives the holdout (spec §19.1: hidden rows never reach the
    system), judged on the same rows `create_hidden_split` will write.

    The system meets a trap only in the visible rows, so a trap has to be there. Hidden rows matter
    where the grade reads them: the imbalance check scores the rare class, and the group check
    needs more than one unseen group to measure anything.
    """
    issues: list[CaseIssue] = []

    def lost(message: str) -> None:
        issues.append(CaseIssue("TRAP_HOLDOUT", f"{message} once the hidden rows are removed"))

    for trap in case.traps:
        match trap:
            case TargetLeakageTrap():
                for name in trap.columns:
                    if not _varies(visible[name]):
                        lost(f"target_leakage: {name!r} no longer varies, so it can't leak,")
            case GroupedRowsTrap():
                sizes = Counter(column_tokens(visible[trap.group_key]))
                if len(sizes) < 2 or max(sizes.values()) < 2:
                    lost(f"grouped_rows: {trap.group_key!r} has no repeated visible groups")
                if len(set(column_tokens(hidden[trap.group_key]))) < 2:
                    issues.append(
                        CaseIssue(
                            "TRAP_HOLDOUT",
                            f"grouped_rows: the hidden rows hold one {trap.group_key!r} group, too "
                            "few to measure generalization to new groups",
                        )
                    )
            case TemporalOrderTrap():
                if time_values(visible[trap.time_key]).nunique() < 2:
                    lost(f"temporal_order: {trap.time_key!r} no longer changes")
                for name in trap.future_columns:
                    if not _varies(visible[name]):
                        lost(f"temporal_order: {name!r} no longer varies, so it can't mislead,")
            case ClassImbalanceTrap():
                target = case.constraints.target
                issues.extend(_rare_issues(visible[target], trap, "the visible rows"))
                issues.extend(_rare_issues(hidden[target], trap, "the hidden rows"))
            case AttackTrap() if trap.column is not None:
                assert trap.marker is not None  # the model requires it together with `column`
                text = visible[trap.column].dropna().astype(str)
                if not text.str.contains(trap.marker, regex=False).any():
                    lost(f"attack: no visible value of {trap.column!r} holds its marker")
    return issues


def _split_issues(frame: pd.DataFrame, case: Case) -> list[CaseIssue]:
    try:
        mask = select_hidden_rows(frame, case)
    except HiddenSplitError as exc:
        return [CaseIssue("HIDDEN_SPLIT", str(exc))]
    return _holdout_issues(frame.loc[~mask], frame.loc[mask], case)


def check_dataset(case: Case, dataset: CaseDataset) -> list[CaseIssue]:
    """Everything that depends on the rows: columns, target, trap metadata, the hidden split, and
    whether each trap's evidence survives it. Every problem is an issue, never a crash."""
    try:
        frame = dataset.table()
    except (ValueError, OSError, pa.ArrowException) as exc:
        return [CaseIssue("DATASET", f"the dataset is not a readable Parquet table: {exc}")]
    try:
        issues = [
            *_target_issues(frame, case),
            *_column_issues(frame, case),
            *_trap_issues(frame, case),
        ]
        # The split reports these same problems less clearly; only a sound dataset reaches it.
        return issues or _split_issues(frame, case)
    except HiddenSplitError as exc:
        return [CaseIssue("DATASET", str(exc))]


def validate_case(
    case: Case,
    *,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    planted_dir: Path = PLANTED_DIR,
    repo_root: Path = REPO_ROOT,
) -> CaseReport:
    """Validate a loaded case: dataset hash, trap metadata against the data, hidden split."""
    issues = _file_issues(case, repo_root)
    try:
        dataset = load_case_dataset(case, snapshots_dir=snapshots_dir, planted_dir=planted_dir)
    except CaseDatasetError as exc:
        issues.append(CaseIssue("DATASET", str(exc)))
    else:
        issues.extend(check_dataset(case, dataset))
    return CaseReport(case.id, tuple(issues))


def validate_case_file(
    path: Path,
    *,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    planted_dir: Path = PLANTED_DIR,
    repo_root: Path = REPO_ROOT,
) -> CaseReport:
    """Validate a case file end to end; a file that does not load is a failed report (`SCHEMA`)."""
    try:
        case = load_case(path)
    except CaseFormatError as exc:
        return CaseReport(path.stem, tuple(CaseIssue("SCHEMA", p) for p in exc.problems))
    return validate_case(
        case, snapshots_dir=snapshots_dir, planted_dir=planted_dir, repo_root=repo_root
    )


__all__ = [
    "MAX_POSITIVE_SHARE",
    "CaseIssue",
    "CaseReport",
    "check_dataset",
    "validate_case",
    "validate_case_file",
]
