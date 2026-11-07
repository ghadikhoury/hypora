"""Reproduce and validate any dev case from its pinned real snapshot.

    uv run --locked python -m hypora.evaluation.cases.dev.generate [case_id ...]

The two planted cases regenerate their derived Parquet. Other cases deliberately leave their real
snapshot unchanged: the temporal and imbalance traps are observed properties, the control has no
trap, and the tool fault is declared in YAML and applied only during execution. Never fetch or
substitute data here; missing snapshots must be obtained with scripts/snapshot_dataset.py (D-015).
"""

import json
import sys
from collections.abc import Sequence

from hypora.evaluation.case import CASES_DIR, Case, load_case_set
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.cases.dev import plant_grouped_rows, plant_target_leakage

PLANTINGS = {
    "dev_target_leakage": plant_target_leakage.PLANTING,
    "dev_grouped_rows": plant_grouped_rows.PLANTING,
}


def generate(case: Case) -> dict[str, object]:
    """Regenerate a case's planted data when needed, then verify its source, hash and trap."""
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
    cases = load_case_set(CASES_DIR / "dev")
    wanted = set(argv)
    unknown = wanted - {case.id for case in cases}
    if unknown:
        raise ValueError(f"unknown dev cases: {sorted(unknown)}")
    for case in cases:
        if not wanted or case.id in wanted:
            print(json.dumps(generate(case)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
