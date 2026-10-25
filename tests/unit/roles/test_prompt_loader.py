"""Prompts are versioned files loaded at runtime, never inlined in code (PLAN.md S1.5)."""

from pathlib import Path

import pytest

from hypora.core.roles.prompt_loader import PROMPTS_DIR, PromptError, load_prompt

ROLES_DIR = PROMPTS_DIR.parent


def test_every_prompt_file_has_a_header_naming_it() -> None:
    names = sorted(path.stem for path in PROMPTS_DIR.glob("*.md"))
    assert names == [
        "intake",
        "reporter",
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
