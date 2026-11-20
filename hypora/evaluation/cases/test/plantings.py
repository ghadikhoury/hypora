"""The test set's generation scripts: each reads one verified real snapshot and writes one file.

    uv run --locked python -m hypora.evaluation.cases.test.plantings

Every change here is to real rows and real labels (D-015): recovering a group key the rows already
share, rewriting a timestamp into an unambiguous ISO form, dropping a second target, or, for the two
malformed-data fault cases, blanking a seeded handful of labels. No value is invented except those
blanks. Each function is pure (no clock, no global random state), and the Parquet writer is the
snapshots' own, so the bytes reproduce for given library versions and the case file's SHA-256 can
vouch for them. `data/` is git-ignored, so each machine regenerates the files.
"""

from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.evaluation.cases.dev.planting import Planting


def _group_ids(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """1, 2, ... for each distinct combination of `columns`, numbered by first appearance."""
    return frame.groupby(columns, sort=False, dropna=False).ngroup().astype("int64") + 1


# --- Grouped rows ---------------------------------------------------------------------------------

CONCRETE_COMPONENTS: Final = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
]


def plant_concrete(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 165: the same mixture is tested at several ages; `mix_id` names each mixture.

    A mixture is its seven component amounts (kg per m3). Rows with equal amounts are the same
    recipe cast and crushed at different ages, so a model evaluated on a mixture it trained on (at
    another age) looks better than it is on a new design."""
    planted = frame.copy()
    planted.insert(0, "mix_id", _group_ids(frame, CONCRETE_COMPONENTS))
    return planted


AIRFOIL_SETUP: Final = [
    "attack-angle",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
]


def plant_airfoil(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 291: each wind-tunnel setup is measured over a sweep of frequencies; `setup_id` names it.

    A setup is the airfoil's angle of attack, chord length, free-stream velocity and suction-side
    displacement thickness; the 1503 rows are about 14 frequencies of each of 106 setups."""
    planted = frame.copy()
    planted.insert(0, "setup_id", _group_ids(frame, AIRFOIL_SETUP))
    return planted


REAL_ESTATE_LOCATION: Final = ["X5 latitude", "X6 longitude"]


def plant_real_estate(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 477: several sales share one location (the same building); `location_id` names it."""
    planted = frame.copy()
    planted.insert(0, "location_id", _group_ids(frame, REAL_ESTATE_LOCATION))
    return planted


# --- Temporal order: an unambiguous time key ------------------------------------------------------


def plant_appliances(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 374: the snapshot's `date` lost the space between day and time ('2016-01-1117:00:00').

    The value is rewritten as '2016-01-11 17:00:00'. Every value has that one 18-character form
    (checked here), so nothing else changes."""
    stamps = frame["date"].astype(str)
    if not (stamps.str.len() == 18).all():
        raise ValueError("unexpected date format in the appliances snapshot")
    planted = frame.copy()
    planted["date"] = stamps.str.slice(0, 10) + " " + stamps.str.slice(10)
    return planted


def plant_garment(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 597: `date` is month/day/year ('1/13/2015'); rewritten as ISO '2015-01-13'."""
    planted = frame.copy()
    parsed = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    planted["date"] = parsed.dt.strftime("%Y-%m-%d")
    return planted


TETOUAN_TARGET: Final = "Zone 1 Power Consumption"
TETOUAN_OTHER_ZONES: Final = ["Zone 2  Power Consumption", "Zone 3  Power Consumption"]


def plant_tetouan(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 849: ISO timestamps, and only zone 1's consumption, the target.

    `DateTime` is month/day/year ('1/13/2017 0:10'); it becomes '2017-01-13 00:10:00'. UCI lists all
    three zones as targets; the other two are measured at the same moment as zone 1, so they are
    dropped, as the dev set drops Parkinsons' second target (D-015)."""
    planted = frame.drop(columns=TETOUAN_OTHER_ZONES)
    parsed = pd.to_datetime(frame["DateTime"], format="%m/%d/%Y %H:%M")
    planted["DateTime"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    return planted


# --- Malformed data: blank labels -----------------------------------------------------------------

BLANK: Final = ""


def _blank_labels(frame: pd.DataFrame, target: str, count: int, seed: int) -> pd.DataFrame:
    """The target as text with `count` seeded rows blank: an export that lost some labels.

    A blank label has no class to learn or score, and filling it in would change the user's data,
    so the platform's integrity check (`hypora.profiling.integrity`) stops the run and asks."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(frame), size=count, replace=False))
    labels = np.array(frame[target].astype(str).tolist(), dtype=object)
    labels[rows] = BLANK
    planted = frame.copy()
    planted[target] = pd.Series(labels, index=frame.index, dtype=object).astype("str")
    return planted


def plant_maternal(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 863: 24 of 1014 `RiskLevel` labels blank."""
    return _blank_labels(frame, "RiskLevel", 24, seed=863)


def plant_landsat(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 146: the class codes as text, with 60 of 6435 labels blank."""
    return _blank_labels(frame, "class", 60, seed=146)


PLANTINGS: Final = {
    "test_group_concrete": Planting(
        "uci-165-concrete-compressive-strength", "test_group_concrete.parquet", plant_concrete
    ),
    "test_group_airfoil": Planting(
        "uci-291-airfoil-self-noise", "test_group_airfoil.parquet", plant_airfoil
    ),
    "test_group_real_estate": Planting(
        "uci-477-real-estate-valuation", "test_group_real_estate.parquet", plant_real_estate
    ),
    "test_time_appliances": Planting(
        "uci-374-appliances-energy-prediction", "test_time_appliances.parquet", plant_appliances
    ),
    "test_time_garment": Planting(
        "uci-597-productivity-prediction-of-garment-employees",
        "test_time_garment.parquet",
        plant_garment,
    ),
    "test_time_tetouan_power": Planting(
        "uci-849-power-consumption-of-tetouan-city",
        "test_time_tetouan_power.parquet",
        plant_tetouan,
    ),
    "test_fault_malformed_maternal": Planting(
        "uci-863-maternal-health-risk", "test_fault_malformed_maternal.parquet", plant_maternal
    ),
    "test_fault_malformed_landsat": Planting(
        "uci-146-statlog-landsat-satellite", "test_fault_malformed_landsat.parquet", plant_landsat
    ),
}
"""Case ID -> its generation script. Every other test case uses its snapshot unchanged."""


def main() -> None:
    """Regenerate every planted file of the test set and print its path and SHA-256."""
    for planting in PLANTINGS.values():
        path, digest = planting.write()
        print(f"{path}  {digest}")


if __name__ == "__main__":
    main()
