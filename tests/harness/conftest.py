"""Synthetic fixtures are test-only, never evaluation/development cases (D-015)."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.schemas import ExperimentContract


@pytest.fixture
def contract_factory():
    """Construct real frozen schemas; no sandbox or contract stubs."""

    def build(strategy="random", validation="holdout", task="regression", fixes=()):
        split = {"strategy": strategy, "test_fraction": 0.2, "validation": validation}
        if strategy in ("group", "group_temporal"):
            split["group_key"] = "group"
        if strategy in ("temporal", "group_temporal"):
            split["time_key"] = "time"
        return ExperimentContract.model_validate(
            {
                "run_id": "run_harness",
                "version": 1,
                "approved_plan_hash": "a" * 64,
                "approval_id": "approval_test",
                "family": "tabular",
                "task_type": task,
                "target": "y",
                "data_contract": {
                    "data_contract_version": 1,
                    "target": "y",
                    "task_type": task,
                    "allowed_features": ["x", "category"],
                    "split": split,
                    "non_learned_fixes": list(fixes),
                    "excluded_features": [
                        {"name": "leak", "reason": "test trap", "evidence": "diag_1"}
                    ],
                },
                "primary_metric": "mae" if task == "regression" else "f1_macro",
                "budget": {
                    "max_usd": 1,
                    "max_compute_seconds": 1800,
                    "max_llm_tokens": 1000,
                    "max_trials": 3,
                    "max_wall_clock_seconds": 1800,
                },
                "default_job_limits": {
                    "max_wall_seconds": 60,
                    "max_memory_gb": 2,
                    "gpu": False,
                    "cpu_threads": 1,
                },
                "seed": 314,
            }
        )

    return build


@pytest.fixture
def frame():
    """Small mixed-type data with grouping, tied timestamps, and an explicitly forbidden feature."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=120)
    data = pd.DataFrame(
        {
            "x": x,
            "category": np.tile(["a", "b", "c"], 40),
            "y": 2 * x + rng.normal(0, 0.1, 120),
            "group": np.repeat(np.arange(12), 10),
            "time": pd.date_range("2026-01-01", periods=60, freq="h").repeat(2),
        }
    )
    data["leak"] = data["y"]
    return data
