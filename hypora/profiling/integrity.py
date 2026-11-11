"""Malformed-data checks the platform must not resolve on its own (spec §19.2 faults; §6.3).

Some defects can't be fixed by methodology without changing what the user asked: a row whose
target is missing, infinite, or (for regression) not a number has no label to learn or score.
Dropping, imputing, or coercing those labels is a label change, which needs the user (spec §6.3),
so the run stops safely before any role sees target-derived data and the report asks for it.

This module is deterministic platform code over the exact rows the run received. The controller
runs it once after profiling; the grader runs the same function on the same rows to prove that
a malformed-data stop was caused by a defect really present in them (D-032; spec §19.2). A
finding names a code, the column, a row count, and the SHA-256 of the affected row positions:
never a value, so it is safe in events and reports.

Feature defects (a numeric column with a stray token, an infinite measurement) are not findings:
a pipeline can handle them inside `fit`, and C's methodology owns them.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import JsonValue, PositiveInt, TypeAdapter

from hypora.core.schemas import Sha256Hex, StrictModel, TaskType

DETAILS_KEY: Final = "integrity_findings"
"""Where an error event's details carry the findings."""
ERROR_CODE: Final = "DATA_MALFORMED"
"""The controller's error code for a stop caused by integrity findings."""


class IntegrityCode(StrEnum):
    """A blocking defect of the target column."""

    TARGET_MISSING = "TARGET_MISSING"  # null, NaN, or blank text
    TARGET_NON_FINITE = "TARGET_NON_FINITE"  # a numeric target of +/-infinity
    TARGET_UNPARSABLE = "TARGET_UNPARSABLE"  # regression target text that is not a number


_TEXT: Final = {
    IntegrityCode.TARGET_MISSING: "missing values",
    IntegrityCode.TARGET_NON_FINITE: "infinite values",
    IntegrityCode.TARGET_UNPARSABLE: "values that are not numbers",
}


class IntegrityFinding(StrictModel):
    """One blocking defect: which check, which column, how many rows, and which rows (hashed)."""

    code: IntegrityCode
    column: str
    rows: PositiveInt
    rows_sha256: Sha256Hex


def rows_digest(positions: Iterable[int]) -> str:
    """The SHA-256 of row positions (0-based, in file order), as a comma-separated list."""
    return hashlib.sha256(",".join(str(int(p)) for p in positions).encode()).hexdigest()


def _finding(code: IntegrityCode, column: str, mask: np.ndarray) -> IntegrityFinding | None:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return None
    return IntegrityFinding(
        code=code, column=column, rows=len(positions), rows_sha256=rows_digest(positions)
    )


def _blank(values: pd.Series) -> np.ndarray:
    """Text cells that are empty or only whitespace (pandas reads them as values, not NaN)."""
    if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
        return np.zeros(len(values), dtype=bool)
    return np.array([isinstance(v, str) and not v.strip() for v in values], dtype=bool)


def check_integrity(
    frame: pd.DataFrame, target: str, task_type: TaskType | None
) -> list[IntegrityFinding]:
    """The blocking defects of `target` in `frame`, in a fixed order; empty when there are none."""
    if target not in frame.columns:
        return []
    values = frame[target].reset_index(drop=True)
    missing = values.isna().to_numpy(dtype=bool) | _blank(values)
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    infinite = np.zeros(len(values), dtype=bool)
    unparsable = np.zeros(len(values), dtype=bool)
    if numeric:
        infinite = np.isinf(values.to_numpy(dtype=float, na_value=np.nan))
    elif task_type is TaskType.REGRESSION:
        parsed = pd.to_numeric(values.where(~missing), errors="coerce")
        unparsable = parsed.isna().to_numpy(dtype=bool) & ~missing
    found = [
        _finding(IntegrityCode.TARGET_MISSING, target, missing),
        _finding(IntegrityCode.TARGET_NON_FINITE, target, infinite),
        _finding(IntegrityCode.TARGET_UNPARSABLE, target, unparsable),
    ]
    return [finding for finding in found if finding is not None]


def finding_details(findings: Sequence[IntegrityFinding]) -> dict[str, JsonValue]:
    """The findings as an error event's `details`."""
    return {DETAILS_KEY: [finding.model_dump(mode="json") for finding in findings]}


_FINDINGS: Final = TypeAdapter(list[IntegrityFinding])


def findings_from_details(details: Mapping[str, JsonValue]) -> list[IntegrityFinding] | None:
    """The findings an error event's details record, or None when they record none validly."""
    try:
        findings = _FINDINGS.validate_python(details.get(DETAILS_KEY))
    except ValueError:
        return None
    return findings or None


def describe_findings(findings: Sequence[IntegrityFinding]) -> str:
    """A value-free sentence for notes and the report: what is wrong and what the user decides."""
    parts = [
        f"the target column {f.column!r} has {f.rows} row(s) with {_TEXT[f.code]} ({f.code.value})"
        for f in findings
    ]
    return (
        "malformed data: "
        + "; ".join(parts)
        + ". Dropping or changing labels needs the user's decision, so nothing was trained"
    )


__all__ = [
    "DETAILS_KEY",
    "ERROR_CODE",
    "IntegrityCode",
    "IntegrityFinding",
    "check_integrity",
    "describe_findings",
    "finding_details",
    "findings_from_details",
    "rows_digest",
]
