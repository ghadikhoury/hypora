"""Explicit, structure-preserving bounds for text that reaches a model (D-047).

A live run failed because a 215-character line of valid code was cut to 200 characters without a
mark: the reviewing model saw unmatched parentheses and stopped the run. Every bound on
model-facing text therefore follows two rules:

- Never cut silently. What is removed is replaced by a platform-written marker that says how much
  was removed and that the removal is for display only.
- Keep structure. Multi-line text is shortened at whole-line boundaries, single-line text at a word
  boundary; JSON keeps its shape (long strings are shortened, list items are dropped whole and
  counted), so the result always parses.

These helpers only remove text. Callers screen the full text (data firewall, injection screen,
credential redaction) before bounding it, so a removal can never reveal more than the full text
would have.
"""

import json
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue

MIN_STRING_CHARS: Final = 200
"""The smallest per-string bound `bound_json` shrinks to before it gives up on the value."""
MIN_LIST_ITEMS: Final = 5
"""The smallest per-list bound `bound_json` shrinks to before it gives up on the value."""


def omitted_chars_marker(count: int) -> str:
    """The marker that ends a shortened single-line text."""
    return f" [... {count} more characters omitted by the platform for display]"


def omitted_lines_marker(lines: int, chars: int) -> str:
    """The marker that ends a shortened multi-line text."""
    return f"[... {lines} more lines ({chars} characters) omitted by the platform for display]"


def omitted_items_marker(count: int) -> str:
    """The item that ends a shortened list."""
    return f"[... {count} more items omitted by the platform for display]"


def shorten(text: str, limit: int) -> str:
    """`text` if it fits in `limit` characters; otherwise a marked prefix that fits.

    Multi-line text keeps whole lines; a single line is cut at the last space that leaves room for
    the marker (or hard, for a line with no space). The result is never longer than `limit` when
    `limit` leaves room for a marker (about 80 characters); a smaller limit returns a bare marker.
    """
    if len(text) <= limit:
        return text
    if "\n" in text:
        lines = text.splitlines()
        kept: list[str] = []
        used = 0
        for index, line in enumerate(lines):
            rest = lines[index:]
            marker = omitted_lines_marker(len(rest), sum(len(item) + 1 for item in rest) - 1)
            if used + len(line) + 1 + len(marker) > limit:
                if kept:
                    return "\n".join([*kept, marker])
                # The first line alone is too long: keep a marked prefix of it.
                after = lines[1:]
                tail = (
                    omitted_lines_marker(len(after), sum(len(item) + 1 for item in after) - 1)
                    if after
                    else ""
                )
                head = shorten(line, limit - len(tail) - 1 if tail else limit)
                return f"{head}\n{tail}" if tail else head
            kept.append(line)
            used += len(line) + 1
        return "\n".join(kept)  # pragma: no cover - the text did not fit, so a line must stop it
    room = limit - len(omitted_chars_marker(len(text)))
    if room <= 0:
        return omitted_chars_marker(len(text)).strip()[:limit]
    cut = text.rfind(" ", 0, room + 1)
    if cut < room // 2:
        cut = room
    return text[:cut].rstrip() + omitted_chars_marker(len(text) - len(text[:cut].rstrip()))


@dataclass(frozen=True)
class Bounded:
    """A JSON value bounded to a size, and what was removed to get there."""

    value: JsonValue
    shortened_strings: int = 0
    omitted_items: int = 0

    @property
    def changed(self) -> bool:
        """Whether anything was removed."""
        return bool(self.shortened_strings or self.omitted_items)


def _shrink(value: JsonValue, string_cap: int, list_cap: int, counts: list[int]) -> JsonValue:
    if isinstance(value, str):
        if len(value) > string_cap:
            counts[0] += 1
            return shorten(value, string_cap)
        return value
    if isinstance(value, list):
        kept = [_shrink(item, string_cap, list_cap, counts) for item in value[:list_cap]]
        if len(value) > list_cap:
            counts[1] += len(value) - list_cap
            kept.append(omitted_items_marker(len(value) - list_cap))
        return kept
    if isinstance(value, dict):
        return {key: _shrink(item, string_cap, list_cap, counts) for key, item in value.items()}
    return value


def bound_json(value: JsonValue, max_chars: int) -> Bounded | None:
    """`value` shrunk until its JSON text fits `max_chars`, or None when no bound fits.

    Strings are shortened with `shorten` and lists keep their first items plus one marker item;
    dict keys, numbers and booleans are kept. The per-string and per-list bounds start loose and
    halve until the value fits, so a result with one huge field loses only that field's tail.
    """
    if len(json.dumps(value)) <= max_chars:
        return Bounded(value)
    string_cap, list_cap = max_chars, 1000
    while True:
        counts = [0, 0]
        shrunk = _shrink(value, string_cap, list_cap, counts)
        if len(json.dumps(shrunk)) <= max_chars:
            return Bounded(shrunk, shortened_strings=counts[0], omitted_items=counts[1])
        if string_cap <= MIN_STRING_CHARS and list_cap <= MIN_LIST_ITEMS:
            return None
        string_cap = max(MIN_STRING_CHARS, string_cap // 2)
        list_cap = max(MIN_LIST_ITEMS, list_cap // 2)


__all__ = [
    "Bounded",
    "bound_json",
    "omitted_chars_marker",
    "omitted_items_marker",
    "omitted_lines_marker",
    "shorten",
]
