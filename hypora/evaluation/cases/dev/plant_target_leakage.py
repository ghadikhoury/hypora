"""Plant a post-outcome measurement in credit-g (OpenML 46918), for `dev_target_leakage`.

`collections_referrals_12m` counts referrals to a collections agency in the 12 months after the
credit decision. It is only known once the outcome is, so it can't be used at application time, and
it follows the label closely: most bad customers have at least one referral, few good ones do. The
counts are drawn from a seeded generator: the column is synthetic, the 1000 customers are real
(D-015 allows synthetic data in a script that plants a trap in a real dataset).
"""

from typing import Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.evaluation.cases.dev.planting import Planting

SNAPSHOT: Final = "openml-46918-credit-g"
FILE: Final = "dev_target_leakage.parquet"
TARGET: Final = "good_or_bad_customer"
LEAK: Final = "collections_referrals_12m"
SEED: Final = 46918
BAD_MEAN_REFERRALS: Final = 2.5
GOOD_REFERRAL_RATE: Final = 0.05


def plant(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the leak just before the target, preserving the rows and other columns."""
    rng = np.random.default_rng(SEED)
    bad = (frame[TARGET] == "bad").to_numpy()
    after_bad = rng.poisson(BAD_MEAN_REFERRALS, len(frame))
    after_good = rng.binomial(1, GOOD_REFERRAL_RATE, len(frame))
    planted = frame.copy()
    planted.insert(
        frame.columns.get_loc(TARGET), LEAK, np.where(bad, after_bad, after_good).astype("int64")
    )
    return planted


PLANTING: Final = Planting(SNAPSHOT, FILE, plant)

if __name__ == "__main__":
    print(*PLANTING.write(), sep="  ")
