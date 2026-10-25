"""Small trial interface used only inside /harness (spec §8.1). No scoring or test labels."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


@dataclass(frozen=True)
class TrainData:
    """Training rows and optional validation rows. Learned transforms fit on X and y only."""

    X: pd.DataFrame
    y: pd.Series
    X_validation: pd.DataFrame | None = None
    y_validation: pd.Series | None = None


@dataclass(frozen=True)
class TrialContext:
    """Platform-provided seed, task type, and the trial's recorded configuration."""

    seed: int
    task_type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Predictions:
    """Labels/values, plus classification probabilities in the declared classes order."""

    values: np.ndarray
    probabilities: np.ndarray | None = None
    classes: tuple = ()
