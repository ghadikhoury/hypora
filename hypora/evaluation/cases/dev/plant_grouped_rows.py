"""Recover the patients of Parkinsons Telemonitoring (UCI 189), for `dev_grouped_rows`.

The groups are real: 42 patients each recorded about 140 times over six months (the dataset's
documentation). The snapshot drops UCI's identifier column (`subject#`, docs/notes/datasets.md), so
this script recovers it: the file stores each patient's rows together, so a change of age or sex
starts a new patient. That finds 41 of the 42, because two neighbors in the file share both values;
treating them as one patient is the safe direction, since it can only keep rows together.

It also drops `motor_UPDRS`. UCI lists it as a second target, and `total_UPDRS` contains it.
"""

from typing import Final

import pandas as pd  # type: ignore[import-untyped]

from hypora.evaluation.cases.dev.planting import Planting

SNAPSHOT: Final = "uci-189-parkinsons-telemonitoring"
FILE: Final = "dev_grouped_rows.parquet"
TARGET: Final = "total_UPDRS"
GROUP_KEY: Final = "patient_id"
OTHER_TARGET: Final = "motor_UPDRS"


def plant(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with `patient_id` first and the second target gone; rows in their stored order."""
    same_person = frame[["age", "sex"]].eq(frame[["age", "sex"]].shift()).all(axis=1)
    planted = frame.drop(columns=[OTHER_TARGET])
    planted.insert(0, GROUP_KEY, (~same_person).cumsum().astype("int64"))
    return planted


PLANTING: Final = Planting(SNAPSHOT, FILE, plant)

if __name__ == "__main__":
    print(*PLANTING.write(), sep="  ")
