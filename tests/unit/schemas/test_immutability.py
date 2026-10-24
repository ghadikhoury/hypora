"""Frozen records are deeply immutable: nothing reachable from them can change after validation.

`frozen=True` alone is shallow (nested models and lists stay mutable), so these tests check the
whole tree: statically over field annotations, at runtime over the fixtures, and with the exact
mutations a caller might try.
"""

import typing
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hypora.core.schemas import ApprovalRecord, ApprovedPlan, DataContract, ExperimentContract
from tests.unit.schemas.helpers import load

FROZEN_RECORDS: list[type[BaseModel]] = [
    DataContract,
    ApprovedPlan,
    ExperimentContract,
    ApprovalRecord,
]
MUTABLE_CONTAINERS = (list, dict, set)


def _types_in(annotation: Any) -> Iterator[Any]:
    """Yield every type mentioned in an annotation (unwrapping Annotated, unions, generics)."""
    yield annotation
    for arg in typing.get_args(annotation):
        yield from _types_in(arg)


def _reachable_models(root: type[BaseModel]) -> dict[type[BaseModel], list[str]]:
    """Every model class reachable from `root` through field annotations, with the field types."""
    seen: dict[type[BaseModel], list[str]] = {}
    stack = [root]
    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen[model] = []
        for name, field in model.model_fields.items():
            for t in _types_in(field.annotation):
                origin = typing.get_origin(t) or t
                if origin in MUTABLE_CONTAINERS:
                    seen[model].append(f"{name}: {t}")
                if isinstance(t, type) and issubclass(t, BaseModel):
                    stack.append(t)
    return seen


@pytest.mark.parametrize("root", FROZEN_RECORDS, ids=lambda r: r.__name__)
def test_every_reachable_model_is_frozen_with_no_mutable_containers(root: type[BaseModel]) -> None:
    for model, mutable_fields in _reachable_models(root).items():
        assert model.model_config.get("frozen") is True, f"{model.__name__} is not frozen"
        assert not mutable_fields, f"{model.__name__} has mutable fields: {mutable_fields}"


def _walk(value: object, path: str) -> Iterator[tuple[str, object]]:
    yield path, value
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _walk(getattr(value, name), f"{path}.{name}")
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            yield from _walk(item, f"{path}[{i}]")


@pytest.mark.parametrize(
    ("model", "fixture"),
    [
        (DataContract, "DataContract/yield"),
        (DataContract, "DataContract/microscopy"),
        (DataContract, "DataContract/sensor_temporal"),
        (ApprovedPlan, "ApprovedPlan/yield"),
        (ExperimentContract, "ExperimentContract/yield"),
        (ApprovalRecord, "ApprovalRecord/yield_plan_approved"),
    ],
)
def test_validated_instances_contain_no_mutable_values(
    model: type[BaseModel], fixture: str
) -> None:
    record = model.model_validate(load(fixture))
    for path, value in _walk(record, "$"):
        assert not isinstance(value, MUTABLE_CONTAINERS), f"mutable value at {path}"
        if isinstance(value, BaseModel):
            assert type(value).model_config.get("frozen") is True, f"unfrozen model at {path}"


CONTRACT_MUTATIONS: list[tuple[str, Callable[[Any], object]]] = [
    ("top-level field", lambda c: setattr(c, "seed", 7)),
    ("nested list append", lambda c: c.data_contract.allowed_features.append("final_mass")),
    ("nested model field", lambda c: setattr(c.data_contract.split, "test_fraction", 0.9)),
    ("budget field", lambda c: setattr(c.budget, "max_usd", 99.0)),
    ("job limits field", lambda c: setattr(c.default_job_limits, "gpu", True)),
    ("top-level list append", lambda c: c.approved_model_refs.append("x@y")),
    (
        "fix model field",
        lambda c: setattr(c.data_contract.non_learned_fixes[1].unit_convert, "to", "K"),
    ),
]


@pytest.mark.parametrize(
    ("what", "mutate"), CONTRACT_MUTATIONS, ids=[m[0] for m in CONTRACT_MUTATIONS]
)
def test_experiment_contract_rejects_every_mutation(
    what: str, mutate: Callable[[Any], object]
) -> None:
    contract = ExperimentContract.model_validate(load("ExperimentContract/yield"))
    before = contract.hash
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        mutate(contract)
    assert contract.hash == before, f"{what} changed the contract"
    assert "final_mass" not in contract.data_contract.allowed_features


PLAN_MUTATIONS: list[tuple[str, Callable[[Any], object]]] = [
    ("recommendation option list", lambda p: p.recommendation.option_ids.append("D9")),
    (
        "measurement list",
        lambda p: p.phase_b_plan.measurements.append(p.phase_b_plan.measurements[0]),
    ),
    (
        "nested cost estimate",
        lambda p: setattr(p.phase_b_plan.measurements[0].estimated_cost, "compute_seconds", 1.0),
    ),
    ("budget field", lambda p: setattr(p.budget, "max_usd", 99.0)),
]


@pytest.mark.parametrize(("what", "mutate"), PLAN_MUTATIONS, ids=[m[0] for m in PLAN_MUTATIONS])
def test_approved_plan_rejects_every_mutation(what: str, mutate: Callable[[Any], object]) -> None:
    plan = ApprovedPlan.model_validate(load("ApprovedPlan/yield"))
    before = plan.content_hash()
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        mutate(plan)
    assert plan.content_hash() == before, f"{what} changed the approved plan"


def test_list_inputs_are_accepted_and_stored_as_tuples() -> None:
    # JSON arrays still validate (the JSON Schema is unchanged); they become tuples in memory.
    contract = DataContract.model_validate(load("DataContract/yield"))
    assert isinstance(contract.allowed_features, tuple)
    assert contract.model_dump(mode="json")["allowed_features"] == list(contract.allowed_features)


def test_mutation_test_helpers_detect_a_mutable_model() -> None:
    # Guard against a vacuous walker: an ordinary model with a list must be reported.
    class Loose(BaseModel):
        items: list[int]

    assert _reachable_models(Loose)[Loose] == ["items: list[int]"]
    assert Loose.model_config.get("frozen") is not True
