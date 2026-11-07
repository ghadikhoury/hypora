"""`hypora eval` (spec §19.6, §21; PLAN.md S2.4): one case set, one config, the chosen systems.

Preflight, before any model call: the case set, the config and its profile, the systems, the
provider key of a system that calls models (named, never shown), Docker, and the sandbox image. The
runner then refuses a dirty tree, an invalid case, and finished records; docs/notes/eval-runner.md.

Exit codes follow `hypora run`: 0 everything was recorded and nothing newly fails; 1 a run did not
finish or wasn't recorded, or a case newly fails; 2 refused (nothing ran); 130 interrupted.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import anyio
import typer
from pydantic import TypeAdapter, ValidationError

from hypora.cli.run import PreflightError, check_docker, prepare_sandbox_image
from hypora.core.config import ConfigError, load_eval_config, load_settings_for_eval
from hypora.core.config.resources import ResourceConfigError
from hypora.core.config.run_config import RunConfig
from hypora.core.llm import MissingCredentialError
from hypora.core.llm.credentials import load_credentials
from hypora.core.schemas import Identifier, ModelProfile, ModelTier
from hypora.evaluation.baselines.fixed_pipeline import SYSTEM_NAME, FixedPipelineBaseline
from hypora.evaluation.case import (
    CASES_DIR,
    Case,
    CaseFormatError,
    CaseSplit,
    load_case_set,
)
from hypora.evaluation.case_data import PLANTED_DIR
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.hidden_split import HiddenLeakError, HiddenStore
from hypora.evaluation.runner.agent import AGENT_SYSTEM, AgentSystem
from hypora.evaluation.runner.runner import EvalOutcome, EvalPlan, run_eval
from hypora.evaluation.runner.store import RESULTS_ROOT, EvalRefused, results_segment
from hypora.evaluation.system import EvalSystem, Grader
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR

DEFAULT_WORK_ROOT: Final = REPO_ROOT / "runs" / "eval"
DEFAULT_MAX_TOTAL_USD: Final = 5.0
"""PLAN.md §0.3 trigger 6: more than this in one operation needs the owner's approval."""
MODEL_SYSTEMS: Final = frozenset({AGENT_SYSTEM})
"""Systems that call models: they need a key, and they count against the spending cap."""

SYSTEM_FACTORIES: Final[Mapping[str, Callable[[], EvalSystem]]] = {
    AGENT_SYSTEM: AgentSystem,
    SYSTEM_NAME: FixedPipelineBaseline,
}


def default_grader() -> Grader:
    """The programmatic graders (PLAN.md S2.2), which every recorded result is graded by."""
    return ProgrammaticGrader()


def load_cases(selector: str) -> tuple[Case, ...]:
    """`dev` or `test`, or the path of a directory of case files."""
    known = {split.value for split in CaseSplit}
    directory = CASES_DIR / selector if selector in known else Path(selector)
    try:
        cases = load_case_set(directory)
    except CaseFormatError as exc:
        raise PreflightError(str(exc)) from None
    if not cases:
        raise PreflightError(f"no cases in {directory}")
    return cases


def select_cases(cases: tuple[Case, ...], only: str | None) -> tuple[Case, ...]:
    """The cases named by `--only`, or all of them."""
    if only is None:
        return cases
    wanted = _names(only)
    unknown = sorted(set(wanted) - {case.id for case in cases})
    if unknown:
        raise PreflightError(f"--only names cases not in this set: {unknown}")
    return tuple(case for case in cases if case.id in wanted)


def _names(value: str | None) -> tuple[str, ...]:
    return tuple(name.strip() for name in (value or "").split(",") if name.strip())


def build_systems(names: tuple[str, ...]) -> tuple[EvalSystem, ...]:
    """The systems named by `--systems`."""
    if not names:
        raise PreflightError("--systems names no system")
    unknown = sorted(set(names) - set(SYSTEM_FACTORIES))
    if unknown:
        raise PreflightError(f"unknown systems {unknown}; known: {sorted(SYSTEM_FACTORIES)}")
    return tuple(SYSTEM_FACTORIES[name]() for name in names)


def check_credentials(run_config: RunConfig) -> None:
    """The profile's provider keys, so a missing one stops the eval here, once, by variable name."""
    credentials = load_credentials()
    tiers = run_config.active_profile.tiers
    try:
        for tier in ModelTier:
            credentials.api_key(tiers.for_tier(tier).provider)
    except MissingCredentialError as exc:
        raise PreflightError(str(exc)) from None


@dataclass(frozen=True)
class EvalOptions:
    """Everything `hypora eval` was asked for."""

    cases: str
    config: str
    systems: tuple[str, ...]
    profile: ModelProfile | None = None
    only: str | None = None
    parallel: int | None = None
    resume: bool = False
    allow_dirty: bool = False
    repeats: int = 0
    repeat_cases: tuple[str, ...] = ()
    max_total_usd: float = DEFAULT_MAX_TOTAL_USD
    results_root: Path = RESULTS_ROOT
    work_root: Path = DEFAULT_WORK_ROOT
    hidden_root: Path | None = None
    snapshots_dir: Path = SNAPSHOTS_DIR
    planted_dir: Path = PLANTED_DIR


def build_plan(options: EvalOptions, out: Callable[[str], None]) -> EvalPlan:
    """Resolve the options and run the preflight; raises PreflightError saying what to fix."""
    try:
        TypeAdapter(Identifier).validate_python(options.config)
    except ValidationError:
        raise PreflightError(f"{options.config!r} is not a config name") from None
    cases = select_cases(load_cases(options.cases), options.only)
    try:
        config = load_eval_config(options.config)
        settings = load_settings_for_eval(options.config, profile=options.profile)
        run_config = RunConfig.build(settings, eval_config=options.config)
        store = HiddenStore(options.hidden_root.resolve() if options.hidden_root else None)
    except (ConfigError, ResourceConfigError, HiddenLeakError) as exc:
        raise PreflightError(str(exc)) from None
    systems = build_systems(options.systems)
    if MODEL_SYSTEMS & set(options.systems):
        check_credentials(run_config)
    grader = default_grader()
    prepare_sandbox_image(check_docker(), out)
    return EvalPlan(
        cases=cases,
        systems=systems,
        grader=grader,
        config=options.config,
        segment=results_segment(options.config, run_config.model_profile, config.model_profile),
        run_config=run_config,
        store=store,
        work_root=options.work_root,
        max_parallel_cases=config.max_parallel_cases,
        requested_parallel=options.parallel,
        results_root=options.results_root.resolve(),
        snapshots_dir=options.snapshots_dir,
        planted_dir=options.planted_dir,
        resume=options.resume,
        allow_dirty=options.allow_dirty,
        repeat_cases=options.repeat_cases,
        repeats=options.repeats,
        model_systems=MODEL_SYSTEMS,
        max_total_usd=options.max_total_usd,
    )


async def execute_eval(options: EvalOptions, out: Callable[[str], None]) -> EvalOutcome:
    """Preflight, then the runner; raises PreflightError or EvalRefused before anything runs."""
    return await run_eval(build_plan(options, out), out)


def eval_command(
    cases: Annotated[str, typer.Option("--cases", help="dev, test, or a case directory.")],
    config: Annotated[str, typer.Option("--config", help="A name from configs/eval/.")],
    systems: Annotated[
        str, typer.Option("--systems", help="Comma-separated systems to compare.")
    ] = "agent,baseline",
    profile: Annotated[
        ModelProfile | None,
        typer.Option("--profile", help="Model profile (default: the one the config states)."),
    ] = None,
    only: Annotated[
        str | None, typer.Option("--only", help="Comma-separated case IDs: a subset of the set.")
    ] = None,
    parallel: Annotated[
        int | None, typer.Option("--parallel", min=1, help="At most this many runs at once.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Keep finished records and run the rest.")
    ] = False,
    allow_dirty: Annotated[
        bool,
        typer.Option("--allow-dirty", help="Run on a dirty tree, recording to results/raw/ only."),
    ] = False,
    repeats: Annotated[
        int, typer.Option("--repeats", min=0, help="Repeat-sample runs per case (at least 3).")
    ] = 0,
    repeat_cases: Annotated[
        str | None,
        typer.Option("--repeat-cases", help="Comma-separated repeat-sample cases (at least 5)."),
    ] = None,
    max_total_usd: Annotated[
        float, typer.Option("--max-total-usd", min=0, help="Refuse if worst-case spend exceeds it.")
    ] = DEFAULT_MAX_TOTAL_USD,
    results_root: Annotated[Path, typer.Option("--results-root")] = RESULTS_ROOT,
    work_root: Annotated[
        Path, typer.Option("--work-root", help="Runs folder and state databases.")
    ] = DEFAULT_WORK_ROOT,
) -> None:
    """Evaluate systems on a case set; save results, a summary, and regression flags."""
    options = EvalOptions(
        cases=cases,
        config=config,
        systems=_names(systems),
        profile=profile,
        only=only,
        parallel=parallel,
        resume=resume,
        allow_dirty=allow_dirty,
        repeats=repeats,
        repeat_cases=_names(repeat_cases),
        max_total_usd=max_total_usd,
        results_root=results_root,
        work_root=work_root,
    )
    try:
        outcome = anyio.run(execute_eval, options, typer.echo)
    except (PreflightError, EvalRefused) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    except KeyboardInterrupt:
        typer.echo(
            "\nInterrupted: finished records are kept, and no summary was written. Run the same "
            "command with --resume to continue.",
            err=True,
        )
        raise typer.Exit(130) from None
    typer.echo(f"Results: {outcome.directory}")
    raise typer.Exit(0 if outcome.clean else 1)


__all__ = [
    "DEFAULT_MAX_TOTAL_USD",
    "DEFAULT_WORK_ROOT",
    "EvalOptions",
    "build_plan",
    "build_systems",
    "check_credentials",
    "default_grader",
    "eval_command",
    "execute_eval",
    "load_cases",
    "select_cases",
]
