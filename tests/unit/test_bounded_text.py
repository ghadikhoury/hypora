"""Explicit, structure-preserving bounds on model-facing text (D-047)."""

import json

from pydantic import JsonValue

from hypora.core.bounded_text import bound_json, shorten
from hypora.core.roles.single_agent import MAX_TOOL_RESULT_CHARS, bound_tool_result


def test_text_that_fits_is_returned_unchanged() -> None:
    assert shorten("short", 10) == "short"
    assert bound_json({"a": [1, 2]}, 100) is not None


def test_a_single_line_is_shortened_at_a_word_with_a_counted_marker() -> None:
    text = "word " * 100
    short = shorten(text, 120)
    assert len(short) <= 120
    assert "more characters omitted by the platform" in short
    kept = short.split(" [... ")[0]
    assert text.startswith(kept) and not kept.endswith("wor")  # a whole word
    count = int(short.split("[... ")[1].split(" ")[0])
    assert count == len(text) - len(kept)


def test_multi_line_text_keeps_whole_lines_and_counts_the_rest() -> None:
    lines = [f"line {i} " + "x" * 30 for i in range(50)]
    short = shorten("\n".join(lines), 400)
    assert len(short) <= 400
    shown = short.splitlines()
    assert shown[:-1] == lines[: len(shown) - 1]  # every shown line is whole
    assert shown[-1].startswith(f"[... {50 - (len(shown) - 1)} more lines")


def test_a_huge_first_line_still_keeps_a_marked_prefix() -> None:
    text = "y " * 2000 + "\nsecond line"
    short = shorten(text, 300)
    assert len(short) <= 300 and short.startswith("y y")
    assert "more characters omitted" in short and "1 more lines" in short


def test_bound_json_keeps_structure_and_counts_removals() -> None:
    value: JsonValue = {"note": "n" * 5000, "items": list(range(500)), "flag": True}
    bounded = bound_json(value, 2000)
    assert bounded is not None and bounded.changed
    text = json.dumps(bounded.value)
    assert len(text) <= 2000
    result = json.loads(text)
    assert result["flag"] is True
    assert result["items"][-1].endswith("more items omitted by the platform for display]")
    assert bounded.omitted_items == 500 - (len(result["items"]) - 1)
    assert bounded.shortened_strings == 1


def test_an_oversized_tool_result_stays_valid_json_and_says_so() -> None:
    lines: list[JsonValue] = [{"text": "z" * 900, "n": i} for i in range(200)]
    text = bound_tool_result({"lines": lines})
    assert len(text) <= MAX_TOOL_RESULT_CHARS
    data = json.loads(text)  # the old cut ended mid-structure, which did not parse
    assert (
        "over the" in data["platform_note"]
        and "list item(s) were left out" in data["platform_note"]
    )
    assert data["result"]["lines"][0] == lines[0]
    assert bound_tool_result({"ok": 1}) == json.dumps({"ok": 1})


def test_json_text_content_keeps_its_structure_too() -> None:
    value = {"rows": ["r" * 100] * 1000}
    data = json.loads(bound_tool_result(json.dumps(value)))
    assert isinstance(data["result"]["rows"], list)
