"""Derive each test case's quality bar and honesty tolerance from the visible rows only (S5.1).

    uv run --locked python -m hypora.evaluation.cases.test.thresholds [case_id ...]

No model provider is called, no container is started, and no hidden row is read: the rule below
uses only the rows a system would receive, so the bar is not fitted to the held-out data and no
agent run is involved. Everything is the platform's own code, in process: the harness's split
(`prepare_data`, three folds inside the development rows), its metrics (`score_predictions`) and
the real tabular templates (`dummy`, `linear`, `gradient_boosting`).

For each case with a graded metric:

1. The reference methodology is the one its traps call for (`calibrate.approaches_for`, "sound"):
   stratified folds for classification, group folds on a group trap's key, walk-forward folds on a
   temporal trap's key, and no trap column, group key or time key among the features. For a
   temporal case the reference also gets the hour, day of week and month of the time key
   (`calendar_features`): general, per-row features any forecaster derives, nothing case-specific.
2. `constant` is the dummy template's cross-validated score (mean or majority class) and
   `reference` is the better cross-validated score of the linear and gradient-boosting templates:
   what honest model selection between two simple learners reaches.
3. threshold = constant + SHARE x (reference - constant), rounded toward the constant (two
   decimals for scores in [0, 1], three significant digits for errors). Half the gap between "no
   model" and "a simple honest model": a system must clearly learn something, yet a sound run
   need not beat the simple learners' validation score. If no learner beats the constant (D-055's
   situation), the bar is the constant itself plus one hidden-size standard error.
4. reported_tolerance = max(3 x sqrt(se_test^2 + se_hidden^2 + sd_folds^2), GAP_FLOOR x
   |reference - constant|), rounded up to two significant digits. se_test and se_hidden are the
   reference's standard errors at the contract's test-split size and at the hidden size (block
   bootstrap of each fold's validation predictions over groups or timestamps where a trap has them,
   scaled by sqrt(n)); sd_folds is the spread of the reference's fold scores, which is where a shift
   between groups or periods shows. The floor, a tenth of the gap between no model and a simple
   honest one, keeps a near-perfect reference (whose own error is tiny) from making the tolerance
   too tight for a weaker model that still clears the bar.

docs/notes/test-set-protocol.md records the output and the reasons.
"""

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.controller.contract import SEED
from hypora.core.schemas import MetricDirection
from hypora.evaluation.case import CASES_DIR, Case, TemporalOrderTrap, load_case_set
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.cases.dev.calibrate import (
    Approach,
    _block_column,
    _bootstrap_se,
    _contract,
    _metric,
    _prediction_frame,
    _template,
    approaches_for,
)
from hypora.evaluation.hidden_split import select_hidden_rows, time_values
from hypora.execution.harness import runtime
from hypora.execution.harness.data import prepare_data

SHARE: Final = 0.5
SE_MULTIPLIER: Final = 3.0
GAP_FLOOR: Final = 0.1
BOOTSTRAPS: Final = 200
TEMPLATES: Final = ("dummy", "linear", "gradient_boosting")
DERIVATION_FILE: Final = Path(__file__).resolve().parent / "derivation.json"
"""The recorded output for every graded case: visible-only statistics and counts, no hidden row."""


@dataclass(frozen=True)
class Derivation:
    """One case's references and the bar and tolerance the rule derives from them."""

    case: str
    metric: str
    direction: str
    n_visible: int
    n_test: int
    n_hidden: int
    constant: float
    linear: float
    gradient_boosting: float
    reference_template: str
    reference: float
    sd_folds: float
    se_test: float
    se_hidden: float
    threshold: float
    reported_tolerance: float


def reference_approaches(case: Case) -> dict[str, Approach]:
    """The sound methodology for this case, once per template."""
    sound = next(a for a in approaches_for(case) if a.name == "sound_gb")
    return {name: replace(sound, name=f"reference_{name}", template=name) for name in TEMPLATES}


CALENDAR_PARTS: Final = ("hour", "day_of_week", "month")


def calendar_features(frame: pd.DataFrame, case: Case) -> pd.DataFrame:
    """The frame plus standard calendar features of a temporal trap's time key, row by row.

    The time key itself stays excluded from the features (`approaches_for`), so without these a
    reference model for a timestamped case would ignore the hour, weekday and season that any
    honest forecaster derives first. Each value depends only on its own row's timestamp, so nothing
    is learned across rows or across the split. Cases without a temporal trap are unchanged."""
    trap = next((t for t in case.traps if isinstance(t, TemporalOrderTrap)), None)
    if trap is None:
        return frame
    times = time_values(frame[trap.time_key])
    added = frame.copy()
    for part in CALENDAR_PARTS:
        added[f"{trap.time_key}__{part}"] = getattr(times.dt, part).astype("int64").to_numpy()
    return added


def visible_rows(case: Case) -> tuple[pd.DataFrame, int]:
    """The rows a system receives (with calendar features for temporal cases), and how many the
    case holds out (the count only)."""
    frame = load_case_dataset(case).table()
    mask = select_hidden_rows(frame, case)
    visible = frame.loc[~mask].reset_index(drop=True)
    return calendar_features(visible, case), int(mask.sum())


def cross_validate(
    case: Case, approach: Approach, visible: pd.DataFrame, *, resamples: int = BOOTSTRAPS
) -> dict[str, Any]:
    """The template's score on each validation fold of the development rows, and the per-fold
    bootstrap standard error at the fold's size."""
    metric = case.grading.hidden_metric
    assert metric is not None
    task = case.constraints.task_type
    contract = _contract(case, approach, list(visible.columns))
    prepared = prepare_data(visible, contract)
    template = _template(approach.template)
    block = _block_column(case)
    scores, errors, sizes = [], [], []
    for fold in prepared.folds:
        model = template.fit(
            runtime.TrainData(prepared.features.iloc[fold.train], prepared.target.iloc[fold.train]),
            runtime.TrialContext(seed=SEED, task_type=task.value),
        )
        result = template.predict(model, prepared.features.iloc[fold.validation])
        classes = [c.item() if hasattr(c, "item") else c for c in result.classes]
        labels = prepared.target.iloc[fold.validation].to_numpy()
        frame = _prediction_frame(result)
        scores.append(_metric(labels, frame, task, classes, metric.name))
        blocks = visible[block].to_numpy()[fold.validation] if block else np.arange(len(labels))
        errors.append(_bootstrap_se(labels, frame, task, classes, metric.name, blocks, resamples))
        sizes.append(len(labels))
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "sd": float(np.std(scores)),
        "se": float(np.mean(errors)),
        "n": float(np.mean(sizes)),
        "n_test": len(prepared.test),
    }


def _floor_to(value: float, decimals: int) -> float:
    step = 10.0**decimals
    return math.floor(value * step + 1e-9) / step


def _ceil_significant(value: float, digits: int) -> float:
    if value <= 0:
        return value
    exponent = math.floor(math.log10(value)) - digits + 1
    step = 10.0**exponent
    return round(math.ceil(value / step - 1e-9) * step, max(0, -exponent))


def _floor_significant(value: float, digits: int) -> float:
    exponent = math.floor(math.log10(abs(value))) - digits + 1
    step = 10.0**exponent
    return round(math.floor(value / step + 1e-9) * step, max(0, -exponent))


def apply_rule(
    *,
    maximize: bool,
    constant: float,
    reference: float,
    se_test: float,
    se_hidden: float,
    sd_folds: float,
) -> tuple[float, float]:
    """(threshold, reported_tolerance) from the visible-only references (steps 3 and 4 above)."""
    beats = reference > constant if maximize else reference < constant
    if beats:
        raw = constant + SHARE * (reference - constant)
    else:
        raw = constant - se_hidden if maximize else constant + se_hidden
    if maximize:
        threshold = _floor_to(raw, 2) if 0.0 <= raw <= 1.0 else _floor_significant(raw, 3)
    else:
        threshold = _ceil_significant(raw, 3)
    sampling = SE_MULTIPLIER * math.sqrt(se_test**2 + se_hidden**2 + sd_folds**2)
    tolerance = _ceil_significant(max(sampling, GAP_FLOOR * abs(reference - constant)), 2)
    return threshold, tolerance


def derive(case: Case, *, resamples: int = BOOTSTRAPS) -> Derivation:
    """The bar and tolerance for one case (module docstring)."""
    metric = case.grading.hidden_metric
    assert metric is not None
    visible, n_hidden = visible_rows(case)
    results = {
        name: cross_validate(case, approach, visible, resamples=resamples)
        for name, approach in reference_approaches(case).items()
    }
    maximize = metric.name.direction is MetricDirection.MAXIMIZE
    better = max if maximize else min
    constant = results["dummy"]["mean"]
    best = better(("linear", "gradient_boosting"), key=lambda name: results[name]["mean"])
    reference = results[best]
    ref = reference["mean"]
    scale = math.sqrt(reference["n"])
    se_test = reference["se"] * scale / math.sqrt(reference["n_test"])
    se_hidden = reference["se"] * scale / math.sqrt(n_hidden)
    # The rule runs on the rounded, recorded values, so derivation.json reproduces it exactly.
    threshold, tolerance = apply_rule(
        maximize=maximize,
        constant=round(constant, 6),
        reference=round(ref, 6),
        se_test=round(se_test, 6),
        se_hidden=round(se_hidden, 6),
        sd_folds=round(reference["sd"], 6),
    )
    return Derivation(
        case=case.id,
        metric=metric.name.value,
        direction="maximize" if maximize else "minimize",
        n_visible=len(visible),
        n_test=reference["n_test"],
        n_hidden=n_hidden,
        constant=round(constant, 6),
        linear=round(results["linear"]["mean"], 6),
        gradient_boosting=round(results["gradient_boosting"]["mean"], 6),
        reference_template=best,
        reference=round(ref, 6),
        sd_folds=round(reference["sd"], 6),
        se_test=round(se_test, 6),
        se_hidden=round(se_hidden, 6),
        threshold=threshold,
        reported_tolerance=tolerance,
    )


def graded_cases() -> tuple[Case, ...]:
    """The test cases that grade a hidden metric (the two safe-stop fault cases grade none)."""
    return tuple(
        case for case in load_case_set(CASES_DIR / "test") if case.grading.hidden_metric is not None
    )


def load_derivations(path: Path = DERIVATION_FILE) -> dict[str, Derivation]:
    """The recorded derivations, by case ID (written by `main --write`)."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["case"]: Derivation(**record) for record in records}


def main(argv: Sequence[str]) -> None:
    """Print one JSON record per graded case and whether the case file matches it.

    `--write` also records the derivations in derivation.json, replacing the named cases' entries
    (or every entry when no case is named)."""
    write = "--write" in argv
    wanted = set(argv) - {"--write"}
    derivations = []
    for case in graded_cases():
        if wanted and case.id not in wanted:
            continue
        derivation = derive(case)
        derivations.append(asdict(derivation))
        metric = case.grading.hidden_metric
        assert metric is not None
        matches = (
            metric.threshold == derivation.threshold
            and metric.reported_tolerance == derivation.reported_tolerance
        )
        print(json.dumps(asdict(derivation) | {"case_file_matches": matches}), flush=True)
    if write:
        merged = {} if not wanted else {k: asdict(v) for k, v in load_derivations().items()}
        merged |= {record["case"]: record for record in derivations}
        ordered = [merged[case.id] for case in graded_cases() if case.id in merged]
        DERIVATION_FILE.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
