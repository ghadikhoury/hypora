"""Validation of C's evidence-backed data methodology proposals."""

import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from hypora.core.schemas import DataContract, TaskSpecification
from hypora.core.schemas.contracts import ExcludedFeature, ParseDatetimeFix
from hypora.core.schemas.events import AnyEvent, ErrorEvent
from hypora.execution.harness.data import fix_column

OPEN_QUESTIONS_REFUSAL: Final = "unresolved methodology questions require escalation"
"""The validator's fixed refusal when C's proposal still lists open questions."""

OPEN_QUESTIONS_REFUSED: Final = "OPEN_QUESTIONS_REFUSED"
"""Error code of the event that stores a refusal's screened open questions (S4.2b)."""

ASSUMPTION_KIND: Final = "assumption"
"""Evidence kind (and `RecordRef` prefix) of an exclusion the platform made for an assumption."""

ASSUMPTION_REASON: Final = (
    "assumed not available at prediction time or not permitted: C's open question about it "
    "could not be asked because no user is present (S4.2c)"
)
"""The fixed `reason` of every exclusion made for a conservative assumption."""

NO_FEATURE_LEFT: Final = "no allowed feature remains after the conservative assumptions"
"""The refusal when answering the open questions conservatively would exclude every feature."""

ASSUMPTION_PREFIX: Final = "Assumption (no user present): "
"""Every assumption escalation summary starts with this; the report lists those summaries."""


class MethodologyError(ValueError):
    """A proposed methodology is inconsistent with the task or its evidence."""


@dataclass(frozen=True)
class Assumption:
    """One open question turned into its conservative answer (S4.2c, D-052).

    `columns` are the allowed features the question names, after any column C excluded itself
    without evidence because of it (F1); the platform excludes each of them, cited by the
    matching entry of `refs`. `already` are named columns that were excluded before
    it. A question that names no column changes nothing and is reported as an open risk. The
    question text itself is screened before anything stores it.
    """

    question: str
    columns: tuple[str, ...]
    refs: tuple[str, ...]
    already: tuple[str, ...] = ()


def _words(text: str) -> str:
    """`text` case-folded, with every run of non-alphanumeric characters as one space."""
    return " " + " ".join(re.split(r"[^0-9a-z]+", text.casefold())).strip() + " "


def columns_named(question: str, columns: Iterable[str]) -> list[str]:
    """The `columns` that `question` names, in their given order.

    Matching ignores case and punctuation (`collections_referrals_12m`, `Collections referrals
    12m` and a back-quoted name all match) and needs whole words, so `age` matches "the age of"
    but not "average". A false match only excludes a column: the conservative side.

    A name of `SHORT_NAME` characters or fewer (`a`, `x`, `id`, `no`) would match ordinary words
    in nearly every question and could exclude every feature, so it counts only when written
    exactly and quoted (in backticks or quotes), e.g. `` `id` ``.
    """
    words = _words(question)
    named = []
    for column in columns:
        if len(column.strip()) <= SHORT_NAME:
            quoted = rf"[`'\"]{re.escape(column)}[`'\"]"
            if re.search(quoted, question):
                named.append(column)
        elif _words(column).strip() and _words(column) in words:
            named.append(column)
    return named


SHORT_NAME: Final = 2
"""Column names up to this length match a question only when quoted exactly."""


def unsupported_exclusions(
    proposal: DataContract, task: TaskSpecification, evidence: Mapping[str, tuple[str, str]]
) -> list[str]:
    """The excluded columns `validate_data_proposal` would refuse for lacking evidence.

    The same rule as the validator: an exclusion that is not one of the user's forbidden columns
    must cite a diagnostic returned for that same column. In proposal order.
    """
    forbidden = set(task.forbidden_columns)
    return [
        item.name
        for item in proposal.excluded_features
        if item.name not in forbidden
        and (item.evidence not in evidence or evidence[item.evidence][1] != item.name)
    ]


def assume_conservatively(
    proposal: DataContract,
    *,
    tag: str,
    unsupported: Sequence[str] = (),
    earlier: Sequence[str] = (),
    answer: bool = True,
) -> tuple[DataContract, dict[str, tuple[str, str]], list[Assumption]]:
    """Turn each open question into its conservative answer; platform code, no model call.

    The conservative answer to a question about a column (is it known at prediction time, may it
    be used) is that it is not available: every allowed feature a question names is excluded,
    never kept. The contract comes back with no open questions, the new exclusions cite
    `assumption:<tag>_<n>` evidence (returned for the validator) and nothing else changes.
    Raises MethodologyError when no allowed feature would remain.

    `unsupported` (no user present only, F1): columns C excluded itself without diagnostic
    evidence. One a question names, from this proposal or from `earlier` (C's questions on its
    previous attempt), is the same conservative answer, so its exclusion is recorded as a
    platform assumption (`ASSUMPTION_REASON`, `assumption:` evidence) instead of being refused.
    An unsupported exclusion no question names is left as it is, for the validator to refuse;
    no column is ever allowed by this. An earlier question is reported only when it accounts
    for such an exclusion. With `answer=False` only those exclusions are converted: the
    proposal's own questions stay open and no allowed feature is excluded.
    """
    allowed = list(proposal.allowed_features)
    excluded = list(proposal.excluded_features)
    pending = list(unsupported)  # excluded columns: the schema forbids allowing one too
    evidence: dict[str, tuple[str, str]] = {}
    assumptions = []
    current = list(proposal.open_questions)
    questions = current + [q for q in earlier if q not in current]

    def assume(column: str) -> str:
        ref = f"{ASSUMPTION_KIND}:{tag}_{len(evidence) + 1}"
        evidence[ref] = (ASSUMPTION_KIND, column)
        return ref

    for index, question in enumerate(questions):
        own = index < len(current)
        adopted = columns_named(question, pending)
        named = columns_named(question, allowed) if own and answer else []
        if not own and not adopted:
            continue
        already = [
            column
            for column in columns_named(question, [item.name for item in excluded])
            if column not in adopted
        ]
        refs = []
        for column in adopted:
            ref = assume(column)
            excluded = [
                ExcludedFeature(name=column, reason=ASSUMPTION_REASON, evidence=ref)
                if item.name == column
                else item
                for item in excluded
            ]
            pending.remove(column)
            refs.append(ref)
        for column in named:
            ref = assume(column)
            excluded.append(ExcludedFeature(name=column, reason=ASSUMPTION_REASON, evidence=ref))
            allowed.remove(column)
            refs.append(ref)
        assumptions.append(Assumption(question, (*adopted, *named), tuple(refs), tuple(already)))
    if not allowed:
        raise MethodologyError(NO_FEATURE_LEFT)
    converted = proposal.model_copy(
        update={
            "allowed_features": tuple(allowed),
            "excluded_features": tuple(excluded),
            "open_questions": () if answer else proposal.open_questions,
        }
    )
    return converted, evidence, assumptions


def keep_assumed_exclusions(
    proposal: DataContract, evidence: Mapping[str, tuple[str, str]]
) -> DataContract:
    """`proposal` with every column an earlier assumption excluded still excluded.

    After the split is locked C may change exclusions; a column excluded for a conservative
    assumption stays excluded (cited by the same evidence), whether C moved it back to
    `allowed_features` or omitted it from both lists, so no later proposal can quietly take the
    convenient answer back or drop a settled exclusion. Non-assumed columns are never touched.
    """
    assumed = {column: ref for ref, (kind, column) in evidence.items() if kind == ASSUMPTION_KIND}
    if not assumed:
        return proposal
    excluded = {item.name for item in proposal.excluded_features}
    restore = [column for column in assumed if column not in excluded]
    allowed = tuple(column for column in proposal.allowed_features if column not in assumed)
    if not restore and len(allowed) == len(proposal.allowed_features):
        return proposal
    if not allowed:
        raise MethodologyError(NO_FEATURE_LEFT)
    return proposal.model_copy(
        update={
            "allowed_features": allowed,
            "excluded_features": (
                *proposal.excluded_features,
                *(
                    ExcludedFeature(name=c, reason=ASSUMPTION_REASON, evidence=assumed[c])
                    for c in restore
                ),
            ),
        }
    )


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
