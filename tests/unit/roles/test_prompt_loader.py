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
    assert prompt.version == 9
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
    assert "a residual risk or a choice between reasonable designs is not a conflict" in lower
    # D-058: feature-only concerns, the user's own words, and the trial evidence check.
    assert "feature_concern_columns" in prompt.text
    assert "user's original request" in lower
    assert "use_best_validated_trial" in prompt.text
    # D-061: a report names a concrete problem, judged against the contract feature facts.
    assert "concrete problem you can name, in the lines shown or the platform evidence" in lower
    assert "when contract feature facts are shown" in lower


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


MODEL_CHOICE_PRINCIPLES = (
    "if the trial budget allows, cover two or three diverse model families",
    "suited to the data's size, feature types and task",
    "train with\n  an objective that matches that metric",
    "when classes are imbalanced, handle it deliberately",
    "consider a nonlinear learner when",
    "chosen from the training rows inside `fit`",
    "never from validation or test labels",
    "small, noisy or grouped, prefer stronger regularization",
    "`fit` splits a group across folds, it can overstate performance on new groups",
    "sandbox compute is a run limit",
    "final job refits",
    "implement the needed loss efficiently",
    "many times slower",
    "prefer a cheaper variant",
)
# D-065: model choice comes from B's research, A's assignment and C's inspection, never from a
# hard-coded list of estimators or hyperparameters, and no rule keyed on a dev-case symptom.
CASE_DERIVED_PATCHES = (
    "HistGradientBoosting",
    "LightGBM",
    "XGBoost",
    "HuberRegressor",
    "Lasso",
    "l2_regularization",
    "min_samples_leaf",
    'loss="absolute_error"',
    "criterion=",
    "max_trials <= 4",
    "~12 run steps",
)


@pytest.mark.parametrize("name", ["agent_c", "single_agent"])
def test_engineer_prompts_carry_the_same_model_choice_principles(name: str) -> None:
    """D-065: C and the single agent share one general principle block (comparable systems)."""
    prompt = load_prompt(name)
    assert prompt.version == (14 if name == "agent_c" else 4)
    lower = prompt.text.lower()
    for principle in MODEL_CHOICE_PRINCIPLES:
        assert principle in lower, principle
    for patch in CASE_DERIVED_PATCHES:
        assert patch not in prompt.text, patch
    assert "the example pipeline is a starting point,\n  not a recommendation" in lower


def test_model_choice_principles_match_between_c_and_single_agent() -> None:
    def block(name: str) -> str:
        text = load_prompt(name).text
        start = text.index("- If the trial budget allows")
        return text[start : text.index("Rules for `pipeline.py`:")]

    assert block("agent_c") == block("single_agent")


def test_agent_c_workers_sizes_budgets_from_the_admission_rule() -> None:
    prompt = load_prompt("agent_c_workers")
    assert prompt.version == 3
    lower = prompt.text.lower()
    # D-065: the budget is the sum of every call; the old fixed 60000-80000 advice stopped
    # most workers before submit_experiment in the S4.2j verify runs.
    assert "sum of all its model calls" in lower
    assert "times the number of calls" in lower
    assert "leave room for later calls to carry the growing conversation" in lower
    assert "write the final report in a separate call" in lower
    assert "60000" not in prompt.text and "max_trials <= 4" not in prompt.text
    assert "worker tools listed in your context" in lower
    assert "denied whole" in lower
    assert "mid-draft" in lower


def test_agent_c_keeps_recovery_constant_and_budget_rules() -> None:
    prompt = load_prompt("agent_c")
    lower = prompt.text.lower()
    assert "unsubmitted `draft` trial" in lower
    assert "via its `trial_id`" in prompt.text
    assert "run_diagnostic" in prompt.text
    assert "DummyRegressor" in prompt.text and "DummyClassifier" in prompt.text
    # D-060: a constant is the supported choice for regression only; never a constant classifier.
    assert "cannot beat a constant baseline" not in prompt.text
    assert "For a regression task, if no trial beats that reference" in prompt.text
    assert "never nominate a constant classifier" in prompt.text
    assert 'DummyClassifier(strategy="prior")' not in prompt.text
    assert "remaining run steps and sandbox\n   compute" in prompt.text
    assert "leave run steps and compute for" in lower
    assert "a slot when a follow-up comparison needs another trial" in lower
    assert "remaining steps" in lower
    # Integrity rules survive the rewrite.
    assert "never print data" in lower
    assert "inside `fit`" in prompt.text
    assert "idempotency_key" in prompt.text
    assert "the contract wins" in lower


def test_research_and_lead_prompts_route_model_choice() -> None:
    """D-065: inter-agent communication, not a fixed list, chooses the model families."""
    b = load_prompt("agent_b")
    assert b.version == 4
    assert "model families worth comparing" in b.text
    assert "the evidence supports them" in b.text
    assert "untrusted evidence" in b.text
    a = load_prompt("agent_a_decide").text
    assert "Choose the kinds of models C should compare from B's research" in a


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
