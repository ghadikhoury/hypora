"""Validate evidence and user constraints before adopting C's data proposal."""

import pytest

from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.methodology import MethodologyError, validate_data_proposal
from hypora.core.schemas.contracts import ExcludedFeature, SplitSpec, SplitStrategy
from tests.unit.controller.helpers import spec


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
