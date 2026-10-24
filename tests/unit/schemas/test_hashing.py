"""Canonical JSON and hashing (hypora.core.hashing): deterministic, normalized, strict."""

import hashlib
import math

import pytest

from hypora.core.hashing import canonical_hash, canonical_json, sha256_hex
from hypora.core.schemas import ApprovalRecord, DataContract, ExperimentContract
from tests.unit.schemas.helpers import load


def test_keys_are_sorted_and_whitespace_removed() -> None:
    assert (
        canonical_json({"b": 1, "a": {"d": [1, 2], "c": None}})
        == b'{"a":{"c":null,"d":[1,2]},"b":1}'
    )


def test_key_order_does_not_change_the_hash() -> None:
    assert canonical_hash({"x": 1, "y": 2}) == canonical_hash({"y": 2, "x": 1})


def test_utf8_is_written_unescaped() -> None:
    assert canonical_json({"solvent": "éthanol"}) == '{"solvent":"éthanol"}'.encode()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, b"1"),
        (-0.0, b"0"),
        (0.0, b"0"),
        (0.1, b"0.1"),
        (1e-7, b"1e-07"),
        (2.5e20, b"2.5e+20"),
        (2.0**53 + 2, b"9007199254740994.0"),
        (True, b"true"),
        (3, b"3"),
    ],
)
def test_numbers_are_normalized(value: object, expected: bytes) -> None:
    assert canonical_json(value) == expected


def test_int_and_integral_float_hash_the_same() -> None:
    assert canonical_hash({"test_fraction": 1}) == canonical_hash({"test_fraction": 1.0})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"metrics": [1.0, bad]})


@pytest.mark.parametrize("bad", [{1: "a"}, {"a": {1, 2}}, {"a": b"bytes"}, object()])
def test_non_json_values_are_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_tuples_serialize_like_lists() -> None:
    assert canonical_json((1, "a")) == canonical_json([1, "a"])


def test_hash_is_sha256_of_canonical_bytes() -> None:
    value = {"b": [1.5, "é"], "a": True}
    assert canonical_hash(value) == hashlib.sha256(canonical_json(value)).hexdigest()
    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_golden_hash_is_stable() -> None:
    # Pinned: if this changes, every stored contract hash and approval would silently break.
    # The expected value was computed independently: printf '%s' <bytes> | sha256sum.
    value = {"b": [1.5, "é"], "a": True, "c": 2.0}
    assert canonical_json(value) == '{"a":true,"b":[1.5,"é"],"c":2}'.encode()
    assert canonical_hash(value) == (
        "63d23985ece4e21c029eafe2179dc12dcf4375f9bc5500a2ad606ce4ee385174"
    )


def test_models_hash_through_their_json_dump() -> None:
    contract = DataContract.model_validate(load("DataContract/yield"))
    assert contract.content_hash() == canonical_hash(contract.model_dump(mode="json"))
    assert canonical_json(contract) == canonical_json(contract.model_dump(mode="json"))


def test_golden_data_contract_hash_is_stable() -> None:
    # Pinned so that a pydantic upgrade that changes dumping is caught, not silently re-hashed.
    contract = DataContract.model_validate(load("DataContract/yield"))
    assert contract.content_hash() == load("ApprovedPlan/yield")["data_contract_hash"]
    assert contract.content_hash() == (
        "6f558052d352836236522b32fa67054e750b0406e74089f0a35a9f42e12566cc"
    )


def test_equal_instants_in_different_timezones_hash_equally() -> None:
    record = load("ApprovalRecord/microscopy_network_denied")
    as_utc = {**record, "created_at": "2026-10-23T20:41:05Z"}
    assert ApprovalRecord.model_validate(record).content_hash() == (
        ApprovalRecord.model_validate(as_utc).content_hash()
    )


def test_any_contract_change_changes_its_hash() -> None:
    contract = ExperimentContract.model_validate(load("ExperimentContract/yield"))
    assert contract.hash == contract.content_hash()
    for update in ({"seed": 1235}, {"version": 2}, {"quality_threshold": 5.0}):
        changed = ExperimentContract.model_validate({**contract.model_dump(), **update})
        assert changed.hash != contract.hash, update
