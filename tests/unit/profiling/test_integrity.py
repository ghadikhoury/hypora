"""The platform's malformed-target checks (S4.1): value-free, deterministic findings."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import TaskType
from hypora.profiling.integrity import (
    IntegrityCode,
    check_integrity,
    describe_findings,
    finding_details,
    findings_from_details,
    rows_digest,
)


def frame(target: list[object]) -> pd.DataFrame:
    return pd.DataFrame({"x": range(len(target)), "y": target})


def test_clean_targets_have_no_findings() -> None:
    assert check_integrity(frame([0, 1, 0, 1]), "y", TaskType.BINARY_CLASSIFICATION) == []
    assert check_integrity(frame(["a", "b", "a"]), "y", TaskType.BINARY_CLASSIFICATION) == []
    assert check_integrity(frame([1.5, 2.0, -3.0]), "y", TaskType.REGRESSION) == []
    assert check_integrity(frame([0, 1]), "missing_column", None) == []


def test_missing_and_blank_labels_are_found_by_position() -> None:
    (finding,) = check_integrity(frame(["a", None, " ", "b"]), "y", None)
    assert finding.code is IntegrityCode.TARGET_MISSING
    assert finding.rows == 2 and finding.rows_sha256 == rows_digest([1, 2])
    (finding,) = check_integrity(frame([1.0, np.nan, 0.0]), "y", TaskType.BINARY_CLASSIFICATION)
    assert finding.rows == 1 and finding.rows_sha256 == rows_digest([1])


def test_infinite_and_unparsable_regression_targets() -> None:
    codes = [f.code for f in check_integrity(frame([1.0, np.inf, -np.inf]), "y", None)]
    assert codes == [IntegrityCode.TARGET_NON_FINITE]
    found = check_integrity(frame(["1.5", "two", None, "3"]), "y", TaskType.REGRESSION)
    assert [(f.code, f.rows) for f in found] == [
        (IntegrityCode.TARGET_MISSING, 1),
        (IntegrityCode.TARGET_UNPARSABLE, 1),
    ]
    # Text labels are only unparsable when the task needs numbers.
    assert check_integrity(frame(["two", "three"]), "y", TaskType.BINARY_CLASSIFICATION) == []


def test_findings_round_trip_through_event_details_and_carry_no_values() -> None:
    findings = check_integrity(frame(["secret-label", None]), "y", None)
    details = finding_details(findings)
    assert findings_from_details(details) == findings
    assert "secret-label" not in str(details) and "secret-label" not in describe_findings(findings)
    assert findings_from_details({}) is None
    assert findings_from_details({"integrity_findings": []}) is None
    assert findings_from_details({"integrity_findings": [{"code": "NOPE"}]}) is None
