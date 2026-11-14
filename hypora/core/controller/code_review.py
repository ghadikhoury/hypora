"""The screened, explicitly bounded candidate-code excerpt A reviews before final verification.

Live run `run_b2d5f594bf4b` (D-047) failed because the old excerpt cut every line at 200
characters without a mark. A 215-character line of valid, already-executed code reached A with
unmatched parentheses; A concluded the code could not run and stopped before verification.

What this module guarantees:

- **Screen first, then bound.** Every *full* source line is screened by the data firewall (always)
  and, when `defenses.injection_filter` is on, by the injection screen; credential shapes are then
  redacted. Only afterwards is anything left out for size, so a bound can never cut a data row or
  an instruction down to a fragment the screens would not recognise.
- **Never cut a line.** A line is shown whole or replaced by one platform-written marker naming its
  line number, its length and why it is not shown. When the file exceeds the excerpt budget, a
  contiguous middle block of lines is replaced by one marker; the head and the tail stay.
- **Count everything.** The caller reports how many lines were shown, withheld by each screen,
  and omitted for length or for the budget, so a marker is never mistaken for a code defect.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from hypora.core.policy.injection import INJECTION_MARKER, InjectionCategory, classify_line

MAX_CODE_LINES: Final = 1000
"""Lines shown at most: a PyTorch training script of a few hundred lines fits whole."""
MAX_CODE_LINE_CHARS: Final = 1000
"""A longer line is omitted whole (with a marker), never cut."""
MAX_CODE_EXCERPT_CHARS: Final = 60_000
"""About 15k tokens: the excerpt's total size, markers included."""
CODE_LINE_WITHHELD: Final = "[line withheld by the data firewall]"
HEAD_SHARE: Final = 2 / 3
"""Of the budget, the share the head of an over-budget file gets; the tail gets the rest."""

MARKER_PREFIXES: Final = ("[platform: ", "[line withheld by ")
"""How every platform marker in the excerpt begins."""
SOURCE_LABEL: Final = "(source text) "
"""Prefixed to a source line that begins like a marker (review of PR #47)."""

FirewallScreen = Callable[[Sequence[str]], tuple[list[str | None], dict[str, int]]]
"""`DataFirewall.screen_untrusted`: None marks a withheld line; counts per rule."""


def long_line_marker(number: int, length: int, limit: int) -> str:
    """The marker that replaces one over-long line."""
    return (
        f"[platform: line {number} not shown: {length} characters, over the {limit}-character "
        "display limit; omitted for display only, the platform ran the full line as submitted]"
    )


def budget_marker(first: int, last: int) -> str:
    """The marker that replaces a block of lines left out for the excerpt budget."""
    return (
        f"[platform: lines {first}-{last} not shown: the excerpt's size limit was reached; "
        "omitted for display only, the platform ran them as submitted]"
    )


@dataclass(frozen=True)
class CodeExcerpt:
    """The excerpt and its counts. Every count is over the whole file."""

    text: str
    lines_total: int
    lines_shown: int
    """Source lines shown verbatim (after redaction) in `text`."""
    withheld_by_firewall: int
    withheld_by_injection_filter: int
    omitted_too_long: int
    omitted_for_budget: int
    firewall_counts: dict[str, int] = field(default_factory=dict)
    injection_counts: Counter[InjectionCategory] = field(default_factory=Counter)


def build_code_excerpt(
    lines: Sequence[str],
    *,
    firewall: FirewallScreen,
    injection_filter: bool,
    redact: Callable[[str], str],
    max_lines: int = MAX_CODE_LINES,
    max_line_chars: int = MAX_CODE_LINE_CHARS,
    max_chars: int = MAX_CODE_EXCERPT_CHARS,
) -> CodeExcerpt:
    """Screen every full line, then bound the excerpt at whole-line boundaries (module docstring).

    `firewall` raising is left to the caller (it fails closed: no excerpt at all).
    """
    screened, firewall_counts = firewall(lines)
    injection_counts: Counter[InjectionCategory] = Counter()
    entries: list[str] = []
    verbatim: list[bool] = []
    too_long = 0
    for number, (raw, kept) in enumerate(zip(lines, screened, strict=True), start=1):
        if kept is None:
            entries.append(CODE_LINE_WITHHELD)
            verbatim.append(False)
            continue
        category = classify_line(raw) if injection_filter else None
        if category is not None:
            injection_counts[category] += 1
            entries.append(INJECTION_MARKER)
            verbatim.append(False)
            continue
        text = redact(raw)
        if len(text) > max_line_chars:
            too_long += 1
            entries.append(long_line_marker(number, len(text), max_line_chars))
            verbatim.append(False)
            continue
        if text.startswith(MARKER_PREFIXES):
            # A source line (say, inside a string) that imitates a platform marker is shown
            # labelled as source, so a forged marker cannot pose as the platform's.
            text = SOURCE_LABEL + text
        entries.append(text)
        verbatim.append(True)

    total = sum(len(entry) + 1 for entry in entries)
    if len(entries) <= max_lines and total <= max_chars:
        shown = list(range(len(entries)))
        omitted: tuple[int, int] | None = None
    else:
        shown, omitted = _head_and_tail(entries, max_lines, max_chars)
    parts = [entries[index] for index in shown]
    if omitted is not None:
        first, last = omitted
        head = sum(index < first for index in shown)
        parts.insert(head, budget_marker(first + 1, last + 1))
    return CodeExcerpt(
        text="\n".join(parts),
        lines_total=len(lines),
        lines_shown=sum(verbatim[index] for index in shown),
        withheld_by_firewall=sum(kept is None for kept in screened),
        withheld_by_injection_filter=sum(injection_counts.values()),
        omitted_too_long=too_long,
        omitted_for_budget=0 if omitted is None else omitted[1] - omitted[0] + 1,
        firewall_counts=dict(firewall_counts),
        injection_counts=injection_counts,
    )


def _head_and_tail(
    entries: Sequence[str], max_lines: int, max_chars: int
) -> tuple[list[int], tuple[int, int]]:
    """The indices kept from the start and the end, and the omitted block between them."""
    reserve = len(budget_marker(len(entries), len(entries))) + 1
    budget = max_chars - reserve
    line_budget = max_lines - 1
    head_chars, head_lines = int(budget * HEAD_SHARE), int(line_budget * HEAD_SHARE)
    head: list[int] = []
    used = 0
    for index, entry in enumerate(entries):
        if len(head) >= head_lines or used + len(entry) + 1 > head_chars:
            break
        head.append(index)
        used += len(entry) + 1
    tail: list[int] = []
    start = len(head)
    for index in range(len(entries) - 1, start - 1, -1):
        cost = len(entries[index]) + 1
        if len(head) + len(tail) >= line_budget or used + cost > budget:
            break
        tail.append(index)
        used += cost
    tail.reverse()
    first = len(head)
    last = (tail[0] if tail else len(entries)) - 1
    return [*head, *tail], (first, last)


__all__ = [
    "CODE_LINE_WITHHELD",
    "MAX_CODE_EXCERPT_CHARS",
    "MAX_CODE_LINES",
    "MAX_CODE_LINE_CHARS",
    "CodeExcerpt",
    "budget_marker",
    "build_code_excerpt",
    "long_line_marker",
]
