"""The programmatic grader (spec §19.2; PLAN.md S2.2): every pass criterion a case calls for.

`CaseGrade.passed` means no check failed and at least one passed. docs/notes/graders.md lists each
check, what it treats as evidence, and what a system must record to be graded.
"""

import anyio.to_thread

from hypora.evaluation.case import Case
from hypora.evaluation.graders.common import GradingContext, failed, guarded
from hypora.evaluation.graders.hidden_score import (
    BEATS_BASELINE,
    HIDDEN_SCORE,
    HONESTY,
    HiddenPredictor,
    HiddenScore,
    check_beats_baseline,
    check_honesty,
    score_hidden,
)
from hypora.evaluation.graders.run_checks import (
    BUDGET,
    COMPLETED,
    DECISION,
    POLICY,
    PROVENANCE,
    check_budget,
    check_completed,
    check_decision,
    check_faults,
    check_policy,
    check_provenance,
)
from hypora.evaluation.graders.sandbox import SandboxPredictor
from hypora.evaluation.graders.traps import check_traps
from hypora.evaluation.hidden_split import HiddenSplit
from hypora.evaluation.system import (
    CaseGrade,
    CheckStatus,
    GraderCheck,
    ReportedMetric,
    SystemRun,
)


def assemble_grade(
    case: Case, run: SystemRun, checks: list[GraderCheck], score: HiddenScore | None
) -> CaseGrade:
    """A grade from its checks: passed only if none failed and at least one passed."""
    statuses = [check.status for check in checks]
    return CaseGrade(
        case_id=case.id,
        system=run.system,
        run_id=run.run_id,
        passed=CheckStatus.FAILED not in statuses and CheckStatus.PASSED in statuses,
        hidden_metric=ReportedMetric(name=score.metric.name, value=score.value) if score else None,
        checks=tuple(checks),
    )


class ProgrammaticGrader:
    """Implements `Grader`. The sandbox and the database are blocking, so it works in a thread."""

    def __init__(self, predictor: HiddenPredictor | None = None) -> None:
        self.predictor = predictor or SandboxPredictor()

    async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
        """Apply every pass criterion of spec §19.2 that the case calls for."""
        return await anyio.to_thread.run_sync(self._grade, case, split, run)

    def _grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
        if case.id != run.case_id or case.id != split.case_id:
            # Don't touch hidden rows or run another case's model for a grade that can't count.
            mismatch = failed(
                "inputs_match",
                f"grading case {case.id} with a run of {run.case_id} and a split of "
                f"{split.case_id}",
            )
            return assemble_grade(case, run, [mismatch], None)
        context = GradingContext(case, split, run)
        try:
            return self._all_checks(context)
        finally:
            context.close()

    def _all_checks(self, context: GradingContext) -> CaseGrade:
        scores: list[HiddenScore] = []

        def hidden() -> GraderCheck:
            check, score = score_hidden(context, self.predictor)
            if score is not None:
                scores.append(score)
            return check

        checks = [
            guarded(COMPLETED, lambda: check_completed(context)),
            guarded(PROVENANCE, lambda: check_provenance(context)),
            guarded(HIDDEN_SCORE, hidden),
        ]
        score = scores[0] if scores else None
        checks += [
            guarded(HONESTY, lambda: check_honesty(context, score)),
            guarded(BEATS_BASELINE, lambda: check_beats_baseline(context, score)),
            *check_traps(context),
            *check_faults(context),
            guarded(POLICY, lambda: check_policy(context)),
            guarded(BUDGET, lambda: check_budget(context)),
            guarded(DECISION, lambda: check_decision(context)),
        ]
        return assemble_grade(context.case, context.run, checks, score)


__all__ = ["ProgrammaticGrader", "assemble_grade"]
