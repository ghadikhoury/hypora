"""Validation of C's evidence-backed data methodology proposals."""

import json
from collections.abc import Collection, Iterable, Mapping
from typing import Final

from hypora.core.schemas import DataContract, TaskSpecification
from hypora.core.schemas.contracts import ParseDatetimeFix
from hypora.core.schemas.events import AnyEvent, ErrorEvent
from hypora.execution.harness.data import fix_column

OPEN_QUESTIONS_REFUSAL: Final = "unresolved methodology questions require escalation"
"""The validator's fixed refusal when C's proposal still lists open questions."""

OPEN_QUESTIONS_REFUSED: Final = "OPEN_QUESTIONS_REFUSED"
"""Error code of the event that stores a refusal's screened open questions (S4.2b)."""


class MethodologyError(ValueError):
    """A proposed methodology is inconsistent with the task or its evidence."""


def open_question_notes(events: Iterable[AnyEvent]) -> list[str]:
    """One run-record note per open-questions refusal, from its event (S4.2b).

    The event already holds only questions that passed the data firewall screen and a count of
    the withheld ones, so this formats stored text and reads nothing else.
    """
    notes = []
    for event in events:
        if not isinstance(event, ErrorEvent) or event.payload.code != OPEN_QUESTIONS_REFUSED:
            continue
        details = event.payload.details
        questions = details.get("open_questions")
        stored = questions if isinstance(questions, list) else []
        notes.append(
            f"C's {details.get('proposal', 'unknown')} data contract proposal (attempt "
            f"{details.get('attempt', '?')}) was refused for open questions; stored after the "
            f"data firewall screen ({details.get('open_questions_withheld', 0)} withheld): "
            + json.dumps(stored, ensure_ascii=False)
        )
    return notes


def _validate_fixes(proposal: DataContract, known: set[str]) -> None:
    """Non-learned fixes transform features; they never touch the labels or the split's keys.

    The harness applies every fix to the whole table before it splits it. A fix on the target
    would therefore rewrite the validation and test labels too (a cast can truncate a regression
    target), and a fix on a split key would change which rows are held out. Diagnostic evidence
    does not justify either, and the task specification records no user decision that could
    authorize one, so both are refused for every kind of fix. The single exception changes
    nothing: re-parsing the time key is the parse the temporal split itself applies.
    """
    split = proposal.split
    for fix in proposal.non_learned_fixes:
        column = fix_column(fix)
        if column not in known:
            raise MethodologyError("non-learned fixes must name dataset columns")
        if column == proposal.target:
            raise MethodologyError("non-learned fixes must not change the target column")
        reparses_time = isinstance(fix, ParseDatetimeFix) and column == split.time_key
        if column in (split.group_key, split.time_key) and not reparses_time:
            raise MethodologyError("non-learned fixes must not change a split key")


def validate_data_proposal(
    proposal: DataContract,
    task: TaskSpecification,
    columns: Collection[str],
    evidence: Mapping[str, tuple[str, str]],
) -> None:
    """Fail closed when a proposed contract changes scope, lacks exclusion evidence, or carries
    a non-learned fix that would rewrite the labels or a split key."""
    known = set(columns)
    if proposal.data_contract_version != 1:
        raise MethodologyError("initial data contract version must be one")
    if proposal.target != task.target or proposal.task_type != task.task_type:
        raise MethodologyError("target and task type must match the task specification")
    if proposal.target not in known:
        raise MethodologyError("target is not present in the dataset")
    allowed = set(proposal.allowed_features)
    excluded = {item.name for item in proposal.excluded_features}
    if len(excluded) != len(proposal.excluded_features):
        raise MethodologyError("excluded features must be unique")
    if allowed | excluded != known - {proposal.target}:
        raise MethodologyError("every non-target column must be accounted for")
    forbidden = set(task.forbidden_columns) & known
    if forbidden & allowed or not forbidden <= excluded:
        raise MethodologyError("the proposal must exclude the user's forbidden columns")
    for item in proposal.excluded_features:
        if item.name in forbidden:
            if item.evidence != f"task_spec:v{task.version}":
                raise MethodologyError("user exclusions must cite the task specification")
        elif item.evidence not in evidence or evidence[item.evidence][1] != item.name:
            raise MethodologyError("a proposed exclusion lacks matching diagnostic evidence")
    for key in (proposal.split.group_key, proposal.split.time_key):
        if key is not None and (key not in known or key == proposal.target):
            raise MethodologyError("split keys must name non-target dataset columns")
    for kind, key in (
        ("group_overlap", proposal.split.group_key),
        ("timestamp_order", proposal.split.time_key),
    ):
        if key is not None and (kind, key) not in evidence.values():
            raise MethodologyError("the proposed split key lacks matching diagnostic evidence")
    _validate_fixes(proposal, known)
    if proposal.open_questions:
        raise MethodologyError(OPEN_QUESTIONS_REFUSAL)
