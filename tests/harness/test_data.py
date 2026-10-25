"""Check actual split boundaries and transformations, including adversarial timestamp ties."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.execution.harness.data import apply_contract, prepare_data


@pytest.mark.parametrize(
    "strategy,validation",
    [
        ("random", "holdout"),
        ("stratified", "holdout"),
        ("group", "holdout"),
        ("temporal", "holdout"),
        ("group_temporal", "holdout"),
        ("random", "kfold_3"),
        ("stratified", "stratified_kfold_3"),
        ("group", "group_kfold_3"),
        ("temporal", "walk_forward_3"),
        ("group_temporal", "walk_forward_2"),
    ],
)
def test_splits_are_seeded_disjoint_and_respect_boundaries(
    frame,
    contract_factory,
    strategy,
    validation,
):
    if strategy == "stratified":
        frame["y"] = np.tile([0, 1], 60)
    contract = contract_factory(strategy, validation)
    data = prepare_data(frame, contract)
    again = prepare_data(frame, contract)
    np.testing.assert_array_equal(data.test, again.test)
    assert list(data.features.columns) == ["x", "category"]
    assert set(data.development).isdisjoint(data.test)
    assert set(data.development) | set(data.test) == set(range(len(frame)))
    boundaries = [(data.development, data.test)] + [(f.train, f.validation) for f in data.folds]
    for a, b in boundaries:
        assert set(a).isdisjoint(b)
        assert len(a) and len(b)
        if "group" in strategy:
            assert set(frame.iloc[a]["group"]).isdisjoint(frame.iloc[b]["group"])
        if "temporal" in strategy:
            assert frame.iloc[a]["time"].max() < frame.iloc[b]["time"].min()
    assert len(data.folds) == (1 if validation == "holdout" else int(validation.rsplit("_", 1)[1]))


def test_contract_fixes_are_rowwise_and_do_not_change_source(frame, contract_factory):
    frame.loc[0, "x"] = -999
    frame["temp"] = 32.0
    frame["number"] = "12"
    frame["date"] = "2026-01-01"
    contract = contract_factory(
        fixes=[
            {"set_missing": {"column": "x", "values": [-999]}},
            {"unit_convert": {"column": "temp", "from": "F", "to": "C"}},
            {"cast_dtype": {"column": "number", "dtype": "int64"}},
            {"parse_datetime": "date"},
            {"drop_column": "leak"},
        ]
    )
    fixed = apply_contract(frame, contract)
    assert pd.isna(fixed.loc[0, "x"])
    assert (fixed["temp"] == 0).all()
    assert fixed["number"].dtype == "int64"
    assert str(fixed["date"].dtype).endswith("UTC]")
    assert "leak" not in fixed
    assert frame.loc[0, "x"] == -999
    assert "leak" in frame


def test_unsupported_conversion_is_rejected(frame, contract_factory):
    contract = contract_factory(
        fixes=[{"unit_convert": {"column": "x", "from": "furlong", "to": "s"}}]
    )
    with pytest.raises(ValueError, match="unsupported unit"):
        prepare_data(frame, contract)


def test_missing_targets_and_group_keys_fail(frame, contract_factory):
    frame.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="missing target"):
        prepare_data(frame, contract_factory())
    frame.loc[0, "y"] = 1
    frame.loc[0, "group"] = np.nan
    with pytest.raises(ValueError, match="split key"):
        prepare_data(frame, contract_factory("group"))


def test_interleaved_groups_cannot_silently_become_random_split(frame, contract_factory):
    frame["group"] = np.tile([0, 1], 60)
    with pytest.raises(ValueError, match="no disjoint temporal boundary"):
        prepare_data(frame, contract_factory("group_temporal"))


def test_duplicate_source_index_does_not_corrupt_splits(frame, contract_factory):
    frame.index = [0] * len(frame)
    assert len(prepare_data(frame, contract_factory()).test) == 24


def test_csv_and_parquet_inputs_roundtrip(frame, tmp_path):
    from hypora.execution.harness.runner import load_tabular

    csv = tmp_path / "data.csv"
    parquet = tmp_path / "data.parquet"
    frame.to_csv(csv, index=False)
    frame.to_parquet(parquet, engine="pyarrow", index=False)
    assert load_tabular(csv)["x"].to_numpy() == pytest.approx(frame["x"])
    pd.testing.assert_frame_equal(load_tabular(parquet), frame)
    with pytest.raises(ValueError, match="CSV or Parquet"):
        load_tabular(tmp_path / "data.bin")


def test_pyarrow_version_is_pinned_identically_in_platform_and_sandbox():
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).parents[2]
    dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    platform_pin = next(dep for dep in dependencies if dep.startswith("pyarrow=="))
    for file in (
        "uv.lock",
        "hypora/execution/sandbox/requirements-tabular.in",
        "hypora/execution/sandbox/requirements-tabular.lock",
    ):
        text = (root / file).read_text()
        if file == "uv.lock":
            match = re.search(r'name = "pyarrow"\nversion = "([^"]+)"', text)
            assert match
            version = match.group(1)
            assert platform_pin == "pyarrow==" + version
        else:
            assert platform_pin in text
