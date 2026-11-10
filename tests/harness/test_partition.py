"""Security and determinism tests for a run's locked diagnostic partitions."""

import json
import subprocess
import sys

import numpy as np
import pytest

from hypora.execution.harness.partition import create_partition_lock, load_verified_partition


@pytest.mark.parametrize(
    "strategy,validation",
    [
        ("random", "kfold_3"),
        ("stratified", "stratified_kfold_3"),
        ("group", "group_kfold_3"),
        ("temporal", "walk_forward_3"),
        ("group_temporal", "walk_forward_2"),
    ],
)
def test_diagnostic_rows_never_cross_scoring_boundaries(
    frame, contract_factory, tmp_path, strategy, validation
):
    if strategy == "stratified":
        frame["y"] = np.tile([0, 1], 60)
    contract = contract_factory(strategy, validation)
    lock = create_partition_lock(frame, contract, "a" * 64, tmp_path / "lock")
    prepared = lock.prepared
    diagnostic = set(lock.diagnostic)
    assert diagnostic and diagnostic <= set(prepared.development)
    assert diagnostic.isdisjoint(prepared.test)
    for fold in prepared.folds:
        assert diagnostic <= set(fold.train)
        assert diagnostic.isdisjoint(fold.validation)
        if "group" in strategy:
            assert set(frame.iloc[fold.train]["group"]).isdisjoint(
                frame.iloc[fold.validation]["group"]
            )
        if "temporal" in strategy:
            assert frame.iloc[fold.train]["time"].max() < frame.iloc[fold.validation]["time"].min()
    verified = load_verified_partition(frame, contract, "a" * 64, tmp_path / "lock")
    np.testing.assert_array_equal(verified.diagnostic, lock.diagnostic)


def test_lock_rejects_changed_split_and_tampering_but_accepts_feature_only_revision(
    frame, contract_factory, tmp_path
):
    path = tmp_path / "lock"
    contract = contract_factory("random", "kfold_3")
    original = create_partition_lock(frame, contract, "a" * 64, path)
    data = contract.data_contract.model_copy(update={"allowed_features": ("x",)})
    revision = contract.model_copy(update={"data_contract": data})
    verified = load_verified_partition(frame, revision, "a" * 64, path)
    np.testing.assert_array_equal(original.diagnostic, verified.diagnostic)
    with pytest.raises(ValueError, match="does not match"):
        load_verified_partition(frame, contract.model_copy(update={"seed": 42}), "a" * 64, path)
    with pytest.raises(ValueError, match="does not match"):
        load_verified_partition(frame, contract, "b" * 64, path)
    record = json.loads(path.read_text())
    record["diagnostic"].append(record["test"][0])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        load_verified_partition(frame, contract, "a" * 64, path)


def test_missing_and_linked_lock_are_refused(frame, contract_factory, tmp_path):
    path = tmp_path / "lock"
    contract = contract_factory()
    with pytest.raises(FileNotFoundError):
        load_verified_partition(frame, contract, "a" * 64, path)
    source = tmp_path / "real"
    create_partition_lock(frame, contract, "a" * 64, source)
    path.symlink_to(source)
    with pytest.raises(OSError):
        load_verified_partition(frame, contract, "a" * 64, path)


def test_insufficient_groups_or_time_boundaries_fail_closed(frame, contract_factory, tmp_path):
    frame["group"] = 1
    with pytest.raises(ValueError):
        create_partition_lock(frame, contract_factory("group"), "a" * 64, tmp_path / "group")
    frame["time"] = "2026-01-01"
    with pytest.raises(ValueError):
        create_partition_lock(frame, contract_factory("temporal"), "a" * 64, tmp_path / "time")


def test_other_process_verifies_exact_same_lock(frame, contract_factory, tmp_path):
    contract = contract_factory("random", "kfold_3")
    path = tmp_path / "lock"
    original = create_partition_lock(frame, contract, "a" * 64, path)
    frame.to_parquet(tmp_path / "dataset.parquet")
    (tmp_path / "contract.json").write_text(contract.model_dump_json())
    script = """
import json, sys
from pathlib import Path
import pandas as pd
from hypora.core.schemas import ExperimentContract
from hypora.execution.harness.partition import load_verified_partition
root = Path(sys.argv[1])
frame = pd.read_parquet(root / 'dataset.parquet')
contract = ExperimentContract.model_validate_json((root / 'contract.json').read_text())
lock = load_verified_partition(frame, contract, 'a' * 64, root / 'lock')
print(json.dumps(lock.diagnostic.tolist()))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True
    )
    assert json.loads(child.stdout) == original.diagnostic.tolist()


def test_partition_errors_never_echo_unparseable_source_values(frame, contract_factory, tmp_path):
    frame["time"] = frame["time"].astype(str)
    frame.loc[0, "time"] = "private-source-sentinel"
    with pytest.raises(ValueError) as error:
        create_partition_lock(frame, contract_factory("temporal"), "a" * 64, tmp_path / "lock")
    assert "private-source-sentinel" not in str(error.value)


def test_stratified_validation_refuses_classes_with_too_few_rows(frame, contract_factory, tmp_path):
    frame["y"] = [1] * 8 + [0] * (len(frame) - 8)
    contract = contract_factory("stratified", "stratified_kfold_9")
    with pytest.raises(ValueError, match="partition could not be prepared"):
        create_partition_lock(frame, contract, "a" * 64, tmp_path / "lock")
