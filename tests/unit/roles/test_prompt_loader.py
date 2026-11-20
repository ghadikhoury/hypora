"""Prompts are versioned files loaded at runtime, never inlined in code (PLAN.md S1.5)."""

from pathlib import Path

import pytest

from hypora.core.roles.prompt_loader import PROMPTS_DIR, PromptError, load_prompt

ROLES_DIR = PROMPTS_DIR.parent


def test_every_prompt_file_has_a_header_naming_it() -> None:
    names = sorted(path.stem for path in PROMPTS_DIR.glob("*.md"))
    assert names == [
        "agent_a_decide",
        "agent_a_intake",
        "agent_a_report",
        "agent_b",
        "agent_c",
        "agent_c_methodology",
        "agent_c_workers",
        "c_worker",
        "c_worker_final",
        "intake",
        "reporter",
        "role_plan",
        "single_agent",
        "single_agent_context",
        "single_agent_final",
    ]
    for name in names:
        prompt = load_prompt(name)
        assert prompt.ref == f"{name}@v{prompt.version}" and prompt.version >= 1
        assert not prompt.text.startswith("---")


def test_render_fills_every_placeholder_or_refuses() -> None:
    context = load_prompt("single_agent_context")
    values = dict(run_id="run_1", task="t", contract="c", profile="p", limits="l", example="e")
    text = context.render(**values)
    assert "Run ID: run_1" in text and "$" not in text
    del values["example"]
    with pytest.raises(PromptError, match="needs a value for 'example'"):
        context.render(**values)


def test_a_file_without_a_valid_header_is_refused(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("no header\n")
    (tmp_path / "other.md").write_text("---\nprompt: something_else\nversion: 1\n---\ntext\n")
    with pytest.raises(PromptError, match="must start with"):
        load_prompt("bad", tmp_path)
    with pytest.raises(PromptError, match="declares prompt 'something_else'"):
        load_prompt("other", tmp_path)
    with pytest.raises(PromptError, match="no prompt file"):
        load_prompt("missing", tmp_path)


def test_role_code_loads_its_prompts_instead_of_inlining_them() -> None:
    for module in ("intake.py", "single_agent.py", "reporter.py"):
        source = (ROLES_DIR / module).read_text()
        assert "load_prompt(" in source
        assert "You are" not in source and "Return your final answer" not in source


def test_agent_a_decide_includes_methodology_review_guidance() -> None:
    prompt = load_prompt("agent_a_decide")
    assert prompt.version == 7
    lower_text = prompt.text.lower()
    # D-056: report at a methodology review goes back to C; blocker stops at once.
    assert "report sends your reason back to c" in lower_text
    assert "a bounded number of times" in lower_text
    assert "set blocker to true only when no revision of the data contract" in lower_text
    lower = prompt.text.lower()
    assert "explicit training request" in lower
    assert "defensible evaluation" in lower
    assert "advisory" in lower
    assert "duplicate feature rows without a group-id column" in lower
    assert "indivisible chronological blocks" in lower
    # D-058: feature-only concerns, the user's own words, and the trial evidence check.
    assert "feature_concern_columns" in prompt.text
    assert "user's original request" in lower
    assert "use_best_validated_trial" in prompt.text


def test_agent_c_methodology_distinguishes_risks_from_open_questions() -> None:
    prompt = load_prompt("agent_c_methodology")
    assert prompt.version == 8
    lower = prompt.text.lower()
    assert "ordinary input features" in lower and "`risks`" in prompt.text
    assert "`open_questions`" in prompt.text
    assert "post-outcome" in lower
    assert "feature_predictiveness" in prompt.text
    assert "too-good-score check" in lower
    assert "direct target sub-counts" in lower
    assert "at most 1-3 columns" in lower
    # D-058: target-aware diagnostics are refused before the lock; the user's own words.
    assert "refused\nuntil the split is locked" in prompt.text
    assert "user's\noriginal request" in prompt.text


def test_agent_c_includes_imbalanced_classification_guidance() -> None:
    prompt = load_prompt("agent_c")
    assert prompt.version == 11
    lower = prompt.text.lower()
    assert "class imbalance" in lower
    assert "HistGradientBoostingClassifier" in prompt.text
    assert "LightGBM" in prompt.text
    assert "XGBoost" in prompt.text
    assert "threshold calibration" in lower
    assert "inside `fit`" in prompt.text


def test_agent_c_workers_includes_multi_turn_budget_and_small_trial_guidance() -> None:
    prompt = load_prompt("agent_c_workers")
    assert prompt.version == 2
    lower = prompt.text.lower()
    assert "60000" in prompt.text and "80000" in prompt.text
    assert "24000" in prompt.text
    assert "max_trials <= 4" in prompt.text
    assert "mid-draft" in lower


def test_agent_c_includes_draft_recovery_and_mae_regression_guidance() -> None:
    prompt = load_prompt("agent_c")
    assert prompt.version == 11
    lower = prompt.text.lower()
    assert "unsubmitted `draft` trial" in lower
    assert "via its `trial_id`" in prompt.text
    assert "`mae`" in lower
    assert 'loss="absolute_error"' in prompt.text
    assert "l2_regularization" in prompt.text
    assert "min_samples_leaf" in prompt.text
    assert "HuberRegressor" in prompt.text
    assert "Ridge" in prompt.text and "Lasso" in prompt.text
    assert "ExtraTreesRegressor" in prompt.text
    assert "inner cv" in lower


def test_agent_c_includes_step_budget_and_feature_baseline_guidance() -> None:
    prompt = load_prompt("agent_c")
    assert prompt.version == 11
    lower = prompt.text.lower()
    assert "run_diagnostic" in prompt.text
    assert "DummyRegressor" in prompt.text and "DummyClassifier" in prompt.text
    assert "max_trials <= 4" in prompt.text
    assert "~12 run steps" in lower
    assert "remaining steps" in lower


def test_intake_prompts_keep_prediction_time_statements() -> None:
    for name in ("intake", "agent_a_intake"):
        prompt = load_prompt(name)
        assert prompt.version == 2
        lower = prompt.text.lower()
        # D-058: what is available when a prediction is made must survive into the objective.
        assert "which information is available at that moment" in lower
        assert "never drop" in lower


def test_agent_c_names_the_trial_evidence_check() -> None:
    lower = load_prompt("agent_c").text.lower()
    assert "trial evidence check" in lower
    assert "user's original request" in lower
