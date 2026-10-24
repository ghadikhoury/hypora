"""Every schema validates its fixtures, round-trips, and the fixtures agree with each other."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypora.core.schemas import (
    EVENT_CLASSES,
    SCHEMAS,
    ApprovalRecord,
    ApprovedPlan,
    DataContract,
    EventType,
    ExperimentContract,
    Handoff,
    Plan,
    PlanEvent,
    ToolError,
    TrialBrief,
    TrialFile,
    TrialSpec,
    VerificationResult,
    parse_event,
)
from tests.unit.schemas.helpers import FIXTURE_ROOT, fixture_id, fixture_paths, load

FIXTURES = fixture_paths()


def test_every_schema_has_fixtures_and_every_fixture_has_a_schema() -> None:
    fixture_dirs = {p.name for p in FIXTURE_ROOT.iterdir() if p.is_dir()}
    assert fixture_dirs == set(SCHEMAS)
    for name in SCHEMAS:
        assert list((FIXTURE_ROOT / name).glob("*.json")), f"{name} has no fixtures"


@pytest.mark.parametrize("path", FIXTURES, ids=fixture_id)
def test_fixture_validates_and_round_trips(path: Path) -> None:
    adapter = SCHEMAS[path.parent.name]
    record = adapter.validate_python(json.loads(path.read_text(encoding="utf-8")))
    dumped = adapter.dump_python(record, mode="json")
    again = adapter.validate_python(dumped)
    assert again == record
    assert again.content_hash() == record.content_hash()
    # The JSON text form round-trips too (what the DB stores).
    assert adapter.validate_json(adapter.dump_json(record)) == record


def test_every_event_type_has_a_fixture_parsed_to_its_class() -> None:
    seen: set[EventType] = set()
    for path in (FIXTURE_ROOT / "Event").glob("*.json"):
        event = parse_event(json.loads(path.read_text(encoding="utf-8")))
        assert type(event) is EVENT_CLASSES[event.type]
        seen.add(event.type)
    assert seen == set(EventType)


def test_fixture_hashes_link_the_yield_run_together() -> None:
    data_contract = DataContract.model_validate(load("DataContract/yield"))
    plan = ApprovedPlan.model_validate(load("ApprovedPlan/yield"))
    contract = ExperimentContract.model_validate(load("ExperimentContract/yield"))
    approval = ApprovalRecord.model_validate(load("ApprovalRecord/yield_plan_approved"))
    verification = VerificationResult.model_validate(load("VerificationResult/yield_passed"))

    assert plan.data_contract_hash == data_contract.content_hash()
    assert contract.data_contract == data_contract
    assert contract.approved_plan_hash == plan.content_hash()
    assert approval.artifact_hash == plan.content_hash()
    assert approval.approval_id == contract.approval_id
    assert verification.contract_hash == contract.hash


def test_trial_records_agree() -> None:
    spec = TrialSpec.model_validate(load("TrialSpec/yield_hgb"))
    assert spec.to_trial_file() == TrialFile.model_validate(load("TrialFile/yield_hgb"))
    assert TrialBrief.model_validate(load("TrialBrief/yield_hgb")).trial == spec
    assert spec.dir == "runs/run_101/trials/trl_0002"


def test_plan_event_carries_the_plan_fixture() -> None:
    event = parse_event(load("Event/plan"))
    assert isinstance(event, PlanEvent)
    assert event.payload.plan == Plan.model_validate(load("Plan/yield_single_agent"))


# --- The spec's own examples validate (spec §6.4, §6.7, §7.2, §12) ------------------------------


def test_spec_7_2_data_contract_example_validates_verbatim() -> None:
    example = load("DataContract/yield")
    del example["schema_version"]  # the spec's YAML predates schema_version; it defaults to 1
    contract = DataContract.model_validate(example)
    dumped = contract.model_dump(mode="json")
    assert dumped["non_learned_fixes"][1] == {
        "unit_convert": {"column": "temp", "from": "F", "to": "C"}
    }
    assert dumped["split"] == {
        "strategy": "group",
        "group_key": "batch_id",
        "time_key": None,
        "test_fraction": 0.2,
        "validation": "group_kfold_5",
    }


def test_spec_6_7_handoff_example_validates() -> None:
    example = load("Handoff/microscopy_research_lead")
    del example["schema_version"]  # otherwise the §6.7 example field for field
    assert Handoff.model_validate(example).recommendation == (
        "evaluate_pretrained_then_finetune_if_needed"
    )


def test_spec_12_error_example_needs_only_error_class_added() -> None:
    spec_example = {
        "is_error": True,
        "code": "INSUFFICIENT_GPU_MEMORY",
        "message": "Requested configuration exceeds available GPU memory.",
        "retryable": False,
        "details": {"available_gb": 8, "estimated_gb": 12},
        "suggested_recovery": "Reduce batch size or select a smaller model.",
    }
    with pytest.raises(ValidationError, match="error_class"):
        ToolError.model_validate(spec_example)
    error = ToolError.model_validate({**copy.deepcopy(spec_example), "error_class": "resource"})
    assert error.is_error is True
