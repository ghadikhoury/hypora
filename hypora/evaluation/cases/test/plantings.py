"""The test set's generation scripts: each reads one verified real snapshot and writes one file.

    uv run --locked python -m hypora.evaluation.cases.test.plantings

Every change here is to real rows and real labels (D-015): recovering a group key the rows already
share, rewriting a timestamp into an unambiguous ISO form, dropping rows that have no label,
collapsing a source's 3-way survival status into "died", or blanking seeded labels in the two
malformed-data fault cases. Each function is pure (no clock, no global random state). The Parquet
writer is the snapshots' own, so the bytes reproduce for given library versions and the case
file's SHA-256 can vouch for them. `data/` is git-ignored, so each machine regenerates the files.
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


BEIJING_TARGET: Final = "pm2.5"
BEIJING_CLOCK: Final = ["year", "month", "day", "hour"]


def plant_beijing(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 381: hourly PM2.5 at the US embassy in Beijing, 2010-2014.

    The four clock columns become one ISO `timestamp` ('2010-01-02 00:00:00'), placed first. The
    2,067 hours with no PM2.5 reading are dropped: a row without a label can be neither learned nor
    scored (the hidden split refuses missing targets). Nothing else changes."""
    stamps = pd.to_datetime(frame[BEIJING_CLOCK])
    planted = frame.drop(columns=BEIJING_CLOCK)
    planted.insert(0, "timestamp", stamps.dt.strftime("%Y-%m-%d %H:%M:%S"))
    return planted.loc[frame[BEIJING_TARGET].notna()].reset_index(drop=True)


EL_NINO_TARGET: Final = "ss_temp"
EL_NINO_CLOCK: Final = ["year", "month", "day", "date"]


def plant_el_nino(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 122: daily readings of moored buoys in the equatorial Pacific, 1980-1998.

    The clock columns (two-digit `year`, `month`, `day` and the yymmdd `date` integer) become one
    ISO `date` ('1980-03-07'), placed first. Rows with no sea-surface temperature reading, the
    target, are dropped (no label). Nothing else changes."""
    stamps = pd.to_datetime(
        {"year": frame["year"] + 1900, "month": frame["month"], "day": frame["day"]}
    )
    planted = frame.drop(columns=EL_NINO_CLOCK)
    planted.insert(0, "date", stamps.dt.strftime("%Y-%m-%d"))
    return planted.loc[frame[EL_NINO_TARGET].notna()].reset_index(drop=True)


# --- Target leakage -------------------------------------------------------------------------------

CIRRHOSIS_TARGET: Final = "died"


def plant_cirrhosis(frame: pd.DataFrame) -> pd.DataFrame:
    """UCI 878: Mayo Clinic primary biliary cirrhosis trial; `died` replaces the 3-way `Status`.

    `Status` is D (died during follow-up), C (censored: alive at the end of follow-up) or CL
    (censored at liver transplantation). `died` is 1 for D and 0 otherwise, so the outcome is
    "died during the study's follow-up". `N_Days`, the length of that follow-up, stays: it is
    the case's post-outcome column."""
    planted = frame.drop(columns=["Status"])
    planted[CIRRHOSIS_TARGET] = (frame["Status"] == "D").astype("int64")
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
    "test_time_beijing_pm25": Planting(
        "uci-381-beijing-pm2-5", "test_time_beijing_pm25.parquet", plant_beijing
    ),
    "test_time_el_nino": Planting("uci-122-el-nino", "test_time_el_nino.parquet", plant_el_nino),
    "test_leak_cirrhosis": Planting(
        "uci-878-cirrhosis-patient-survival-prediction",
        "test_leak_cirrhosis.parquet",
        plant_cirrhosis,
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
