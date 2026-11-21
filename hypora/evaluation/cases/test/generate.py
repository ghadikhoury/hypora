"""Reproduce and validate any test case from its pinned real snapshot.

    uv run --locked python -m hypora.evaluation.cases.test.generate [case_id ...]

Cases with a generation script (`plantings.py`) regenerate their derived Parquet and must match the
SHA-256 in the case file. Every case is then validated against its data, including the hidden
split and whether each trap survives it. Nothing is fetched or substituted here: a missing
snapshot must be obtained with scripts/snapshot_dataset.py (D-015; README.md lists the commands).
"""

import json
import sys
from collections.abc import Sequence

from hypora.evaluation.case import CASES_DIR, Case, load_case_set
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.cases.test.plantings import PLANTINGS


def generate(case: Case) -> dict[str, object]:
    """Regenerate a case's planted data when it has any, then verify its source, hash and trap."""
    planting = PLANTINGS.get(case.id)
    if planting is not None:
        _, digest = planting.write()
        assert case.dataset.planted is not None
        if digest != case.dataset.planted.sha256:
            raise ValueError(f"{case.id}: generated data differs from the recorded hash")
    report = validate_case(case)
    if not report.ok:
        raise ValueError(f"{case.id}: {report.issues}")
    return {"case": case.id, "snapshot": case.dataset.snapshot, "validated": True}


def main(argv: Sequence[str]) -> None:
    """Generate all cases, or exactly the explicitly named cases; unknown names are errors."""
    cases = load_case_set(CASES_DIR / "test")
    wanted = set(argv)
    unknown = wanted - {case.id for case in cases}
    if unknown:
        raise ValueError(f"unknown test cases: {sorted(unknown)}")
    for case in cases:
        if not wanted or case.id in wanted:
            print(json.dumps(generate(case)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
