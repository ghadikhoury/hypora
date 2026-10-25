"""S1.1: planted traps, privacy bounds, reproducibility, edge cases, and runtime."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from hypora.core.schemas import DatasetProfile, MetricName, TaskType
from hypora.core.schemas.task import IdLikeReason, TabularProfile
from hypora.profiling.tabular import (
    ProfilingConfig,
    leakage_screen_summary,
    profile_csv,
    profile_tabular,
)
from tests.fixtures.tabular import make_tabular_traps


def profile(data: pd.DataFrame, task_type: TaskType | None = None, **kwargs) -> TabularProfile:
    result = profile_tabular(
        data,
        run_id="run_profile",
        dataset_ref="synthetic:fixture",
        target="target" if task_type else None,
        task_type=task_type,
        **kwargs,
    )
    assert result.tabular is not None
    DatasetProfile.model_validate_json(result.model_dump_json())
    return result.tabular


@pytest.mark.parametrize("regression", [False, True])
def test_planted_traps_and_evidence(regression: bool) -> None:
    data = make_tabular_traps(regression=regression)
    kind = TaskType.REGRESSION if regression else TaskType.BINARY_CLASSIFICATION
    result = profile(data, kind)
    leak = next(item for item in result.leakage_candidates if item.column == "post_outcome")
    assert leak.score > leak.threshold
    assert leak.score_metric is (MetricName.R2 if regression else MetricName.ROC_AUC)
    median = np.median([item.score for item in result.feature_associations])
    assert leak.threshold == pytest.approx(median + 0.25)
    assert "3fold" in leak.method and "depth3" in leak.method
    group = next(item for item in result.candidate_group_keys if item.column == "subject_id")
    assert group.n_groups == 90
    assert group.mean_group_size == group.max_group_size == 10
    assert group.target_sharing > group.chance_level + 0.10
    timestamp = next(item for item in result.timestamp_columns if item.column == "measured_at")
    assert timestamp.parse_fraction == 1
    assert timestamp.is_monotonic_increasing
    assert timestamp.median_interval_seconds == 60
    identifier = next(item for item in result.id_like_columns if item.column == "sample_id")
    assert IdLikeReason.MONOTONE_INTEGER in identifier.reasons


def test_full_data_counts_missingness_distributions_and_duplicates() -> None:
    data = pd.DataFrame(
        {
            "amount": [1.0, 3.0, None, 3.0],
            "category": ["a", "b", "a", "b"],
            "target": [0, 1, 1, 1],
        }
    )
    result = profile(data, TaskType.BINARY_CLASSIFICATION)
    assert (result.n_rows, result.n_columns) == (4, 3)
    assert result.n_duplicate_rows == 1 and result.duplicate_fraction == 0.25
    column = result.columns[0]
    assert column.n_missing == 1 and column.missing_fraction == 0.25
    assert column.n_unique == 2 and column.unique_fraction == pytest.approx(2 / 3)
    assert column.numeric is not None
    assert column.numeric.mean == pytest.approx(7 / 3)
    assert result.target is not None
    assert result.target.n_classes == 2 and result.target.imbalance_ratio == 3
    assert sum(item.count for item in result.target.class_counts) == 4
    assert not result.feature_associations  # Too few observations for valid 3-fold CV.


def test_seeded_sampling_and_configurable_threshold() -> None:
    data = make_tabular_traps(n_rows=1200)
    unchanged = data.copy(deep=True)
    config = ProfilingConfig(seed=9, max_sample_rows=600)
    first = profile(data, TaskType.BINARY_CLASSIFICATION, config=config)
    assert first == profile(data, TaskType.BINARY_CLASSIFICATION, config=config)
    assert first.statistics_sample_size == 600
    assert first.n_rows == 1200
    strict = profile(
        data,
        TaskType.BINARY_CLASSIFICATION,
        config=ProfilingConfig(
            seed=9,
            max_sample_rows=600,
            leakage_score_margin=2,
        ),
    )
    assert strict.feature_associations == first.feature_associations
    assert not strict.leakage_candidates
    assert strict.leakage_screen_skipped_reason is None
    assert leakage_screen_summary(strict).startswith("No leakage candidates flagged")
    assert "no leakage found" not in leakage_screen_summary(strict).lower()
    different = profile(
        data,
        TaskType.BINARY_CLASSIFICATION,
        config=ProfilingConfig(
            seed=10,
            max_sample_rows=600,
        ),
    )
    assert different.feature_associations != first.feature_associations
    pd.testing.assert_frame_equal(data, unchanged)


def test_categorical_output_bounds_and_no_free_text() -> None:
    labels = [f"category_{i}_" + "x" * 60 for i in range(8)]
    data = pd.DataFrame(
        {
            "category": labels * 20,
            "notes": [f"private text row number {i} with a unique message" for i in range(160)],
            "target": labels * 20,
        }
    )
    result = profile(data, TaskType.MULTICLASS_CLASSIFICATION)
    serialized = result.model_dump_json()
    assert "private text" not in serialized
    assert "x" * 41 not in serialized
    assert not result.columns[1].top_values
    assert result.target is not None
    assert result.target.n_classes == 8
    for counts in (result.columns[0].top_values, result.target.class_counts):
        assert len(counts) == 5
        assert all(len(item.value) <= 40 and item.count == 20 for item in counts)


def test_truncated_labels_merge_counts() -> None:
    result = profile(pd.DataFrame({"category": ["a" * 40 + "x", "a" * 40 + "y"]}))
    assert len(result.columns[0].top_values) == 1
    assert result.columns[0].top_values[0].count == 2


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ([0] * 12, "fewer than two distinct values"),
        ([0] * 11 + [1], "fewer than three observations"),
        ([None] * 12, "Fewer than six usable target observations"),
        ([0, 1, 0, 1, 0], "Fewer than six usable target observations"),
    ],
)
def test_degenerate_classification_records_skip_reason(target: list, reason: str) -> None:
    result = profile(
        pd.DataFrame({"x": np.arange(len(target)), "target": target}),
        TaskType.BINARY_CLASSIFICATION,
    )
    assert not result.feature_associations and not result.leakage_candidates
    assert result.leakage_screen_skipped_reason is not None
    assert reason in result.leakage_screen_skipped_reason
    restored = TabularProfile.model_validate_json(result.model_dump_json())
    summary = leakage_screen_summary(restored)
    assert summary.startswith("leakage not screened:")
    assert reason in summary and "no leakage found" not in summary.lower()


def test_missing_target_and_legacy_profiles_are_not_reported_clean() -> None:
    result = profile(pd.DataFrame({"x": [1, 2, 3]}))
    assert result.leakage_screen_skipped_reason == "No target was provided."
    legacy = TabularProfile(n_rows=0, n_columns=0, columns=[])
    assert legacy.leakage_screen_skipped_reason is None
    for item in (result, legacy):
        assert leakage_screen_summary(item).startswith("leakage not screened:")


def test_no_features_records_skip_reason() -> None:
    result = profile(pd.DataFrame({"target": [0, 1] * 6}), TaskType.BINARY_CLASSIFICATION)
    assert result.leakage_screen_skipped_reason == "No feature columns are available."
    assert leakage_screen_summary(result).startswith("leakage not screened:")


def test_nonfinite_regression_cv_records_skip_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hypora.profiling.tabular.r2_score", lambda *args, **kwargs: np.nan)
    result = profile(make_tabular_traps(regression=True), TaskType.REGRESSION)
    assert result.leakage_screen_skipped_reason == (
        "No feature has a finite score in all three CV folds."
    )
    assert leakage_screen_summary(result).startswith("leakage not screened:")


def test_completed_screen_report_identifies_candidates() -> None:
    result = profile(make_tabular_traps(), TaskType.BINARY_CLASSIFICATION)
    assert result.leakage_screen_skipped_reason is None
    assert leakage_screen_summary(result).startswith("Leakage candidates flagged:")
    legacy = TabularProfile.model_validate({**result.model_dump(), "feature_associations": []})
    assert "overall screening coverage was not recorded" in leakage_screen_summary(legacy)
    with pytest.raises(ValidationError, match="skipped leakage screen"):
        TabularProfile.model_validate(
            {
                **result.model_dump(),
                "leakage_screen_skipped_reason": "Insufficient targets.",
            }
        )


def test_multiclass_auc_numeric_missing_and_unknown_categories() -> None:
    data = pd.DataFrame(
        {
            "leak": np.tile([0, 1, 2], 100),
            "noise": np.random.default_rng(0).normal(size=300),
            "unique_text": [f"private_{i}" for i in range(300)],
            "missing": [None] * 300,
            "target": np.tile(["a", "b", "c"], 100),
        }
    )
    data.loc[:10, "noise"] = np.inf
    result = profile(data, TaskType.MULTICLASS_CLASSIFICATION)
    assert len(result.feature_associations) == 4
    assert next(item.score for item in result.feature_associations if item.column == "leak") == 1
    assert "private_" not in result.model_dump_json()


def test_regression_nonfinite_and_missing_target() -> None:
    data = make_tabular_traps(regression=True)
    data.loc[:3, "target"] = [np.inf, -np.inf, np.nan, np.nan]
    data["all_missing"] = np.nan
    result = profile(data, TaskType.REGRESSION)
    assert result.target is not None and result.target.n_missing == 2
    assert (
        next(item.score for item in result.feature_associations if item.column == "post_outcome")
        > 0.9
    )
    assert "NaN" not in result.model_dump_json()


def test_uuid_timestamps_and_non_date_categories() -> None:
    data = pd.DataFrame(
        {
            "uuid": [f"00000000-0000-0000-0000-{i:012x}" for i in range(40)],
            "time": [
                *pd.date_range("2025-01-01", periods=38, freq="h").astype(str),
                "2025-01-01",
                "invalid",
            ],
            "category": ["12", "15"] * 20,
        }
    )
    result = profile(data)
    assert IdLikeReason.UUID_PATTERN in result.id_like_columns[0].reasons
    assert len(result.timestamp_columns) == 1
    assert not result.timestamp_columns[0].is_monotonic_increasing
    assert result.timestamp_columns[0].n_duplicate_timestamps == 1


def test_random_grouping_is_not_evidence() -> None:
    rng = np.random.default_rng(20)
    data = pd.DataFrame(
        {"group": np.repeat(np.arange(100), 10), "target": rng.integers(0, 2, 1000)}
    )
    result = profile(data, TaskType.BINARY_CLASSIFICATION)
    assert not result.candidate_group_keys


def test_empty_and_single_row() -> None:
    empty = profile(pd.DataFrame({"x": pd.Series(dtype=float)}))
    assert empty.statistics_sample_size == 0 and empty.duplicate_fraction == 0
    single = profile(pd.DataFrame({"x": [1.0]}))
    assert single.columns[0].numeric is not None and single.columns[0].numeric.std == 0


def test_fingerprint_csv_and_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    data = make_tabular_traps()
    data.to_csv(path, index=False)
    result = profile_csv(path, run_id="run_csv")
    assert result.fingerprint == sha256(path.read_bytes()).hexdigest()
    original = profile_tabular(data, run_id="run_csv", dataset_ref="fixture")
    assert (
        original.fingerprint
        == profile_tabular(
            data.set_axis(np.arange(len(data)) + 5),
            run_id="run_csv",
            dataset_ref="fixture",
        ).fingerprint
    )
    data.loc[0, "noise_0"] += 1
    assert (
        original.fingerprint
        != profile_tabular(
            data,
            run_id="run_csv",
            dataset_ref="fixture",
        ).fingerprint
    )


@pytest.mark.parametrize("size", [-1, 20_001])
def test_schema_sample_bounds(size: int) -> None:
    with pytest.raises(ValidationError):
        TabularProfile(n_rows=0, n_columns=0, columns=[], statistics_sample_size=size)
    assert TabularProfile(n_rows=0, n_columns=0, columns=[]).statistics_sample_size is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sample_rows": 0},
        {"max_sample_rows": 20_001},
        {"seed": -1},
        {"leakage_score_margin": float("nan")},
        {"group_agreement_margin": -1},
    ],
)
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ProfilingConfig(**kwargs)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="provided together"):
        profile_tabular(pd.DataFrame({"target": [1]}), run_id="r", dataset_ref="f", target="target")
    with pytest.raises(ValueError, match="unique"):
        profile(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="existing"):
        profile(pd.DataFrame({"x": [1]}), TaskType.REGRESSION)
    with pytest.raises(ValueError, match="numeric"):
        profile(pd.DataFrame({"target": ["x"]}), TaskType.REGRESSION)


def test_100k_by_50_runtime() -> None:
    data = make_tabular_traps(n_rows=100_000, n_noise_features=44, regression=True)
    started = perf_counter()
    result = profile(data, TaskType.REGRESSION)
    elapsed = perf_counter() - started
    assert result.n_columns == 50 and result.statistics_sample_size == 20_000
    assert len(result.feature_associations) == 49
    assert elapsed < 10, f"100k x 50 profiling took {elapsed:.2f}s (limit 10s)"
