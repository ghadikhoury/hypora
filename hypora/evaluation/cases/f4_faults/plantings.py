"""F4 malformed-data plantings (additional cases complementing the frozen core test set).

uv run --locked python -m hypora.evaluation.cases.f4_faults.plantings
"""

from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.evaluation.cases.dev.planting import Planting

BLANK: Final = ""


def _blank_labels(frame: pd.DataFrame, target: str, count: int, seed: int) -> pd.DataFrame:
    """The target as text with `count` seeded rows blank: an export that lost some labels."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(frame), size=count, replace=False))
    labels = np.array(frame[target].astype(str).tolist(), dtype=object)
    labels[rows] = BLANK
    planted = frame.copy()
    planted[target] = pd.Series(labels, index=frame.index, dtype=object).astype("str")
    return planted


def plant_wine_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 186: 40 of 6,497 `quality` labels blank."""
    return _blank_labels(frame, "quality", 40, seed=186)


def plant_dry_bean(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 602: 50 of 13,611 `Class` labels blank."""
    return _blank_labels(frame, "Class", 50, seed=602)


def plant_student_dropout(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 697: 35 of 4,424 `Target` labels blank."""
    return _blank_labels(frame, "Target", 35, seed=697)


PLANTINGS: Final = {
    "f4_fault_malformed_wine_quality": Planting(
        "uci-186-wine-quality",
        "f4_fault_malformed_wine_quality.parquet",
        plant_wine_quality,
    ),
    "f4_fault_malformed_dry_bean": Planting(
        "uci-602-dry-bean",
        "f4_fault_malformed_dry_bean.parquet",
        plant_dry_bean,
    ),
    "f4_fault_malformed_student_dropout": Planting(
        "uci-697-predict-students-dropout-and-academic-success",
        "f4_fault_malformed_student_dropout.parquet",
        plant_student_dropout,
    ),
}


def main() -> None:
    """Regenerate every planted F4 file and print its path and SHA-256."""
    for planting in PLANTINGS.values():
        path, digest = planting.write()
        print(f"{path}  {digest}")


if __name__ == "__main__":
    main()
