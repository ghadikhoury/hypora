"""Seeded planted-trap data shared by profiling tests and future Stage 2 case builders."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def make_tabular_traps(
    n_rows: int = 900,
    *,
    n_noise_features: int = 6,
    regression: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
    """Plant an outcome-derived feature, repeated subjects, and a chronological timestamp.

    Each group has ten rows with a common latent outcome. `post_outcome` is derived from
    the target; `subject_id` identifies correlated observations. There are noise features
    to give the median-association heuristic a reference. Total columns = noise count + 6.
    """
    if n_rows < 30 or n_noise_features < 1:
        raise ValueError("use at least 30 rows and one noise feature")
    rng = np.random.default_rng(seed)
    groups = np.arange(n_rows) // 10
    latent: np.ndarray
    target: np.ndarray
    if regression:
        latent = rng.uniform(-5, 5, int(groups.max()) + 1)
        target = latent[groups] + rng.normal(0, 0.03, n_rows)
    else:
        latent = rng.integers(0, 2, int(groups.max()) + 1)
        target = latent[groups]
    data = pd.DataFrame(
        {
            "sample_id": np.arange(n_rows),
            "subject_id": [f"subject_{group:06d}" for group in groups],
            "measured_at": pd.date_range("2025-01-01", periods=n_rows, freq="min").astype(str),
            "post_outcome": target + rng.normal(0, 0.001, n_rows),
        }
    )
    for number in range(n_noise_features):
        data[f"noise_{number}"] = rng.normal(size=n_rows)
    data["treatment"] = rng.choice(["control", "treated"], n_rows)
    data["target"] = target
    return data
