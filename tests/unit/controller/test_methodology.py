"""Validate evidence and user constraints before adopting C's data proposal."""

from typing import Any, get_args

import pytest
from pydantic import TypeAdapter

from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.methodology import MethodologyError, validate_data_proposal
from hypora.core.schemas import DataContract
from hypora.core.schemas.contracts import ExcludedFeature, NonLearnedFix, SplitSpec, SplitStrategy
from tests.unit.controller.helpers import spec

COLUMNS = ["x", "when", "group", "leak", "y"]
LEAK_EVIDENCE = {"diag_1": ("feature_predictiveness", "leak")}


def fixes_on(column: str) -> list[dict[str, Any]]:
    """One fix of every kind the contract schema knows, aimed at `column`."""
    return [
        {"cast_dtype": {"column": column, "dtype": "int64"}},
        {"unit_convert": {"column": column, "from": "g", "to": "kg"}},
        {"set_missing": {"column": column, "values": [0]}},
        {"parse_datetime": column},
        {"drop_column": column},
    ]


def proposed(fixes: list[dict[str, Any]], **changes: Any) -> DataContract:
    """C's proposal over COLUMNS: `leak` excluded on evidence, everything else allowed."""
    base = build_data_contract(spec(), COLUMNS).model_dump(mode="json")
    base["allowed_features"] = ["x", "when", "group"]
    base["excluded_features"] = [{"name": "leak", "reason": "after outcome", "evidence": "diag_1"}]
    return DataContract.model_validate({**base, "non_learned_fixes": fixes, **changes})


def test_valid_proposal_keeps_user_forbidden_features_and_evidence():
    task = spec(forbidden_columns=["leak"])
    proposal = build_data_contract(task, ["x", "group", "leak", "y"])
    validate_data_proposal(proposal, task, ["x", "group", "leak", "y"], {})


def test_non_user_exclusion_requires_observed_diagnostic():
    task = spec()
    base = build_data_contract(task, ["x", "leak", "y"])
    proposed = base.model_copy(
        update={
            "allowed_features": ("x",),
            "excluded_features": (
                ExcludedFeature(name="leak", reason="after outcome", evidence="diag_1"),
            ),
        }
    )
    with pytest.raises(MethodologyError, match="lacks matching diagnostic evidence"):
        validate_data_proposal(proposed, task, ["x", "leak", "y"], {})
    validate_data_proposal(
        proposed, task, ["x", "leak", "y"], {"diag_1": ("feature_predictiveness", "leak")}
    )
    with pytest.raises(MethodologyError, match="matching diagnostic evidence"):
        validate_data_proposal(
            proposed, task, ["x", "leak", "y"], {"diag_1": ("feature_predictiveness", "x")}
        )


def test_unresolved_questions_and_wrong_split_key_fail_closed():
    task = spec()
    proposal = build_data_contract(task, ["x", "y"])
    with pytest.raises(MethodologyError, match="unresolved"):
        validate_data_proposal(
            proposal.model_copy(update={"open_questions": ("which target?",)}),
            task,
            ["x", "y"],
            {},
        )
    split = SplitSpec(
        strategy=SplitStrategy.GROUP, group_key="y", test_fraction=0.2, validation="holdout"
    )
    with pytest.raises(MethodologyError, match="split keys"):
        validate_data_proposal(
            proposal.model_copy(update={"split": split}),
            task,
            ["x", "y"],
            {},
        )


def test_proposal_must_account_for_all_columns():
    task = spec()
    proposal = build_data_contract(task, ["x", "leak", "y"])
    with pytest.raises(MethodologyError, match="every non-target"):
        validate_data_proposal(
            proposal.model_copy(update={"allowed_features": ("x",)}),
            task,
            ["x", "leak", "y"],
            {},
        )


# --- Label integrity: non-learned fixes (review of PR #37) ----------------------------------------


def test_the_fix_examples_cover_every_kind_in_the_schema():
    adapter: TypeAdapter[Any] = TypeAdapter(NonLearnedFix)
    kinds: set[type] = {type(adapter.validate_python(fix)) for fix in fixes_on("x")}
    assert kinds == set(get_args(NonLearnedFix))


@pytest.mark.parametrize("fix", fixes_on("y"), ids=lambda fix: next(iter(fix)))
def test_a_fix_on_the_target_is_refused_whatever_its_kind(fix):
    with pytest.raises(MethodologyError, match="must not change the target column"):
        validate_data_proposal(proposed([fix]), spec(), COLUMNS, LEAK_EVIDENCE)


def test_a_target_fix_is_refused_among_valid_feature_fixes():
    fixes = [*fixes_on("x")[:3], {"cast_dtype": {"column": "y", "dtype": "int64"}}]
    with pytest.raises(MethodologyError, match="must not change the target column"):
        validate_data_proposal(proposed(fixes), spec(), COLUMNS, LEAK_EVIDENCE)


def test_neither_evidence_nor_free_text_in_the_task_authorizes_a_target_fix():
    """The task specification has no field recording a user's decision to transform the labels,
    so nothing C can cite makes a target fix acceptable."""
    task = spec(
        exclusion_rules=["cast y to whole numbers"],
        clarifications=[{"question": "Round y?", "answer": "Yes, cast y to int64."}],
    )
    evidence = {**LEAK_EVIDENCE, "diag_2": ("feature_predictiveness", "y")}
    fix = {"cast_dtype": {"column": "y", "dtype": "int64"}}
    with pytest.raises(MethodologyError, match="must not change the target column"):
        validate_data_proposal(proposed([fix]), task, COLUMNS, evidence)


def test_the_target_cannot_be_dropped_through_the_exclusion_list():
    exclusions = [
        {"name": "leak", "reason": "after outcome", "evidence": "diag_1"},
        {"name": "y", "reason": "drop the labels", "evidence": "diag_2"},
    ]
    evidence = {**LEAK_EVIDENCE, "diag_2": ("feature_predictiveness", "y")}
    with pytest.raises(MethodologyError, match="every non-target"):
        validate_data_proposal(
            proposed([], excluded_features=exclusions), spec(), COLUMNS, evidence
        )


@pytest.mark.parametrize("column", ["x", "leak"])
def test_feature_only_fixes_of_every_kind_are_still_accepted(column):
    """Allowed features, and excluded ones (which no trial ever sees), may be fixed."""
    validate_data_proposal(proposed(fixes_on(column)), spec(), COLUMNS, LEAK_EVIDENCE)


def test_a_fix_on_a_column_the_dataset_lacks_is_refused():
    with pytest.raises(MethodologyError, match="must name dataset columns"):
        validate_data_proposal(
            proposed([{"drop_column": "absent"}]), spec(), COLUMNS, LEAK_EVIDENCE
        )


@pytest.mark.parametrize("fix", fixes_on("group"), ids=lambda fix: next(iter(fix)))
def test_a_fix_on_the_group_key_is_refused(fix):
    split = {
        "strategy": "group",
        "group_key": "group",
        "test_fraction": 0.2,
        "validation": "group_kfold_3",
    }
    evidence = {**LEAK_EVIDENCE, "diag_2": ("group_overlap", "group")}
    validate_data_proposal(proposed([], split=split), spec(), COLUMNS, evidence)
    with pytest.raises(MethodologyError, match="must not change a split key"):
        validate_data_proposal(proposed([fix], split=split), spec(), COLUMNS, evidence)


def test_a_fix_on_the_time_key_is_refused_except_the_split_s_own_parse():
    split = {
        "strategy": "temporal",
        "time_key": "when",
        "test_fraction": 0.2,
        "validation": "walk_forward_3",
    }
    evidence = {**LEAK_EVIDENCE, "diag_2": ("timestamp_order", "when")}
    for fix in fixes_on("when"):
        contract = proposed([fix], split=split)
        if "parse_datetime" in fix:
            validate_data_proposal(contract, spec(), COLUMNS, evidence)
        else:
            with pytest.raises(MethodologyError, match="must not change a split key"):
                validate_data_proposal(contract, spec(), COLUMNS, evidence)
