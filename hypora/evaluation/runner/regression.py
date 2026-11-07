"""Regression flags (spec §19.6; PLAN.md S2.4): which cases newly fail against earlier results.

The base is the same config's results on the parent commit. On `main` the parent often has none: a
commit that only adds result files never gets its own eval, and neither does a docs commit. So the
comparison walks back along first-parent history to the nearest commit that has results, and says
how far it went. With no base at all, it says that plainly: no regressions were checked, which is
not the same as none being found.
"""

from collections.abc import Mapping
from typing import Final, Literal, TypedDict

from pydantic import NonNegativeInt

from hypora.core.schemas.common import FrozenModel, FrozenRecord, Identifier, NonEmptyStr
from hypora.evaluation.runner.records import CaseRecord
from hypora.evaluation.runner.store import CodeState, EvalRefused, GitRepo, ResultsStore

ANCESTOR_LIMIT: Final = 200
"""How many first-parent commits to look through for a base."""


class CaseRef(FrozenModel):
    """One case of one system."""

    system: Identifier
    case_id: Identifier


class RegressionReport(FrozenRecord):
    """The comparison, saved as `regressions.json`."""

    commit: NonEmptyStr
    config: NonEmptyStr
    status: Literal["compared", "no_baseline"]
    message: NonEmptyStr
    base_commit: NonEmptyStr | None = None
    commits_back: NonNegativeInt | None = None
    newly_failing: tuple[CaseRef, ...] = ()
    recovered: tuple[CaseRef, ...] = ()
    new_cases: tuple[CaseRef, ...] = ()
    missing_cases: tuple[CaseRef, ...] = ()
    not_comparable: tuple[CaseRef, ...] = ()


class ComparisonGroups(TypedDict):
    """Precisely the outcome groups accepted by RegressionReport."""

    newly_failing: tuple[CaseRef, ...]
    recovered: tuple[CaseRef, ...]
    new_cases: tuple[CaseRef, ...]
    missing_cases: tuple[CaseRef, ...]
    not_comparable: tuple[CaseRef, ...]


def compare_records(
    current: Mapping[tuple[str, str], CaseRecord], base: Mapping[tuple[str, str], CaseRecord]
) -> ComparisonGroups:
    """Sort every (system, case) into what changed between the two sets of primary records.

    Only finished measurements are compared. A run that never got a grade, or a case whose
    definition changed between the commits, is `not_comparable`, not a regression.
    """
    groups: dict[str, list[CaseRef]] = {
        "newly_failing": [],
        "recovered": [],
        "new_cases": [],
        "missing_cases": [],
        "not_comparable": [],
    }
    usable_base = {key: record for key, record in base.items() if record.finished}
    for key in sorted(current):
        ref = CaseRef(system=key[0], case_id=key[1])
        record, earlier = current[key], usable_base.get(key)
        if earlier is None:
            groups["new_cases"].append(ref)
        elif (
            not record.finished
            or record.inputs.case_sha256 != earlier.inputs.case_sha256
            or record.inputs.split != earlier.inputs.split
            or record.provenance.run_config_hash != earlier.provenance.run_config_hash
        ):
            groups["not_comparable"].append(ref)
        elif earlier.passed and not record.passed:
            groups["newly_failing"].append(ref)
        elif record.passed and not earlier.passed:
            groups["recovered"].append(ref)
    for key in sorted(set(usable_base) - set(current)):
        groups["missing_cases"].append(CaseRef(system=key[0], case_id=key[1]))
    return ComparisonGroups(
        newly_failing=tuple(groups["newly_failing"]),
        recovered=tuple(groups["recovered"]),
        new_cases=tuple(groups["new_cases"]),
        missing_cases=tuple(groups["missing_cases"]),
        not_comparable=tuple(groups["not_comparable"]),
    )


def compare_with_parent(
    store: ResultsStore,
    code: CodeState,
    repo: GitRepo,
    current: Mapping[tuple[str, str], CaseRecord],
) -> RegressionReport:
    """Compare `current` with the nearest earlier results of the same config.

    A clean commit starts at its parent. A dirty run starts at HEAD itself, since HEAD is what its
    changes are measured against. Dirty results are never a base: `ResultsStore(dirty=False)` only
    reads committed-commit directories.
    """
    start = code.commit if code.dirty else f"{code.commit}^"
    offset = 0 if code.dirty else 1
    ancestors = repo.ancestors(start, ANCESTOR_LIMIT)
    for distance, commit in enumerate(ancestors, start=offset):
        base_store = ResultsStore(store.root, commit, store.segment)
        try:
            base = base_store.primary_records()
        except EvalRefused as exc:
            return RegressionReport(
                commit=code.commit,
                config=store.segment,
                status="no_baseline",
                message=f"No comparison: {exc}",
                base_commit=commit,
            )
        if not any(record.finished for record in base.values()):
            continue
        groups = compare_records(current, base)
        where = (
            f"the parent commit {commit[:12]}"
            if distance == 1
            else f"{commit[:12]}, {distance} commit(s) back: nearer commits have no results for "
            f"config {store.segment}"
        )
        message = (
            f"Compared with {where}: {len(groups['newly_failing'])} newly failing, "
            f"{len(groups['recovered'])} recovered."
        )
        return RegressionReport(
            commit=code.commit,
            config=store.segment,
            status="compared",
            message=message,
            base_commit=commit,
            commits_back=distance,
            **groups,
        )
    message = (
        f"No regression check: no results for config {store.segment} on the parent of "
        f"{code.commit[:12]} or the {max(len(ancestors) - 1, 0)} commit(s) before it. "
        "These results can be the base for the next commit."
    )
    return RegressionReport(
        commit=code.commit, config=store.segment, status="no_baseline", message=message
    )


def regression_lines(report: RegressionReport) -> list[str]:
    """What the runner prints: the message, then each newly failing case by name."""
    lines = [report.message]
    lines += [f"  NEWLY FAILING  {r.system}/{r.case_id}" for r in report.newly_failing]
    lines += [f"  recovered      {r.system}/{r.case_id}" for r in report.recovered]
    lines += [f"  not comparable {r.system}/{r.case_id}" for r in report.not_comparable]
    return lines


__all__ = [
    "ANCESTOR_LIMIT",
    "CaseRef",
    "RegressionReport",
    "compare_records",
    "compare_with_parent",
    "regression_lines",
]
