"""Schemas fail loudly: unknown fields, misspelled enums, wrong versions, NaN, mutation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypora.core.schemas import (
    SCHEMAS,
    DataContract,
    EventType,
    MetricDirection,
    MetricName,
    RunStatus,
    TaskSpecification,
    parse_event,
)
from tests.unit.schemas.helpers import fixture_id, fixture_paths, load

ONE_FIXTURE_PER_SCHEMA = sorted({p.parent.name: p for p in fixture_paths()}.values())


@pytest.mark.parametrize("path", ONE_FIXTURE_PER_SCHEMA, ids=fixture_id)
def test_unknown_top_level_field_is_rejected(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        SCHEMAS[path.parent.name].validate_python(data)


@pytest.mark.parametrize("path", ONE_FIXTURE_PER_SCHEMA, ids=fixture_id)
def test_other_schema_versions_are_rejected(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    with pytest.raises(ValidationError, match="schema_version"):
        SCHEMAS[path.parent.name].validate_python(data)


def test_unknown_nested_field_is_rejected() -> None:
    data = load("DataContract/yield")
    data["split"]["shuffle"] = True
    with pytest.raises(ValidationError, match="shuffle"):
        DataContract.model_validate(data)


def test_unknown_payload_field_is_rejected() -> None:
    data = load("Event/model_call")
    data["payload"]["api_key"] = "not-a-real-key"
    with pytest.raises(ValidationError, match="api_key"):
        parse_event(data)


@pytest.mark.parametrize("spelling", ["Regression", "REGRESSION", "regresion"])
def test_enum_values_must_match_the_spec_spelling(spelling: str) -> None:
    data = load("DataContract/yield")
    data["task_type"] = spelling
    with pytest.raises(ValidationError, match="task_type"):
        DataContract.model_validate(data)


def test_nan_is_rejected_in_float_fields() -> None:
    data = load("TaskSpecification/microscopy")
    data["quality_target"] = float("nan")
    with pytest.raises(ValidationError, match="quality_target"):
        TaskSpecification.model_validate(data)


def test_assignment_reruns_cross_field_validators() -> None:
    spec = TaskSpecification.model_validate(load("TaskSpecification/microscopy"))
    with pytest.raises(ValidationError, match="quality_target requires primary_metric"):
        spec.primary_metric = None


def test_event_types_are_the_spec_list_plus_d003() -> None:
    spec_15_1 = [
        "model_call", "tool_call", "tool_result", "policy_decision", "approval_requested",
        "approval_granted", "approval_denied", "stage_transition", "job_transition",
        "checkpoint", "escalation", "error",
    ]  # fmt: skip
    assert [t.value for t in EventType] == [*spec_15_1, "role_started", "plan", "handoff"]


def test_run_status_is_exactly_spec_15_1() -> None:
    assert [s.value for s in RunStatus] == [
        "INTAKE", "PHASE_A", "WAITING_APPROVAL", "PHASE_B", "REPORTING", "DONE", "FAILED",
        "CANCELLED",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("metric", "direction"),
    [
        (MetricName.MAE, MetricDirection.MINIMIZE),
        (MetricName.LOG_LOSS, MetricDirection.MINIMIZE),
        (MetricName.F1_MACRO, MetricDirection.MAXIMIZE),
        (MetricName.R2, MetricDirection.MAXIMIZE),
    ],
)
def test_metric_direction_is_computed(metric: MetricName, direction: MetricDirection) -> None:
    assert metric.direction is direction
