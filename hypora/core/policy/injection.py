"""Injection screen for untrusted text (S3.5 design §2.5; spec §14; `defenses.injection_filter`).

Web passages, model cards, sandbox logs and the text of a third-party dataset (column names,
category values, class labels) are written by people or programs outside the platform.
They reach an LLM role through a few MCP tool results. A line in them can try to steer that role:
"ignore previous instructions", a fake `system:` turn, "call submit_experiment", "the plan is
approved". This module finds such lines with a small, deterministic set of patterns and replaces
each with a fixed marker. It never returns or records the matched text, only counts per category.

**Where it runs.** At the MCP server's single exit (`screen_tool_result`), on the results of
`SCREENED_TOOLS` only. Output schemas and tool descriptions do not change: the screen edits text
inside existing string fields and leaves dict keys, numbers, booleans and the platform's own
guidance fields (`PLATFORM_KEYS`) alone.

**What it is not.** It is a pattern screen, so a paraphrase, another language or an encoding gets
through. It is one layer: the role allowlists, the approval check on every job, the plain-prose
assignment validator, the data firewall and the sandbox stay mandatory whether this switch is on
or off (`docs/notes/threat-model.md`). With `defenses.injection_filter` off, every screened call
records that the filter was off, like the other defense switches.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from pydantic import JsonValue

from hypora.core.schemas import ToolName
from hypora.core.schemas.events import (
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
)
from hypora.core.schemas.tools import ToolTier

INJECTION_MARKER: Final = "[line withheld by the injection filter: instruction-like untrusted text]"
SHORT_INJECTION_MARKER: Final = "[withheld by the injection filter]"
"""The marker for results whose text fields are length-bounded (`BOUNDED_TOOLS`)."""
SCREENED_TOOLS: Final = frozenset(
    {
        ToolName.GET_MODEL_EVIDENCE,
        ToolName.RETRIEVE_METHODOLOGY,
        ToolName.GET_EXPERIMENT_LOGS,
        ToolName.GET_EXPERIMENT_STATUS,
        ToolName.INSPECT_DATASET,
        ToolName.GET_EXPERIMENT_METRICS,
    }
)
"""The tools whose results carry text written outside the platform (design §2.5). Dataset
profiles and metrics are included because column names, category values and class labels come
from a third-party dataset."""
BOUNDED_TOOLS: Final = frozenset({ToolName.GET_EXPERIMENT_METRICS})
"""Screened tools whose result fields have short length bounds (class labels: 40 characters). A
string with any withheld line is replaced as a whole by `SHORT_INJECTION_MARKER`, so the screened
result still fits its output schema; the counts are the same as for any other tool."""
PLATFORM_KEYS: Final = frozenset({"next_step"})
"""Result fields the platform writes itself (for example "call get_experiment_status again"), left
unscreened so the platform's own guidance is never withheld. Every other string is screened."""
OFF_REASON: Final = "defenses.injection_filter is off"
MAX_SCREEN_DEPTH: Final = 32


class InjectionCategory(StrEnum):
    """Why a line was withheld. Recorded as counts only, never with the text."""

    OVERRIDE = "override"  # tries to replace the role's instructions
    ROLE_MARKER = "role_marker"  # poses as a chat turn or a system message
    TOOL_CALL = "tool_call"  # tells the reader to call a platform tool
    APPROVAL_CLAIM = "approval_claim"  # claims an approval, authorization or permission


_TOOL_NAMES: Final = "|".join(sorted((re.escape(t.value) for t in ToolName), key=len, reverse=True))
# Zero-width space, joiners and direction marks, word joiner, byte-order mark (built from code
# points so the source holds no invisible characters).
_ZERO_WIDTH: Final = re.compile(f"[{chr(0x200B)}-{chr(0x200F)}{chr(0x2060)}{chr(0xFEFF)}]")

_PATTERNS: Final[tuple[tuple[InjectionCategory, re.Pattern[str]], ...]] = (
    (
        InjectionCategory.OVERRIDE,
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b(\s+\w+){0,3}?\s+"
            r"(instructions?|prompts?|rules|directions|guidelines|messages|context)\b"
            r"|\bnew\s+instructions?\s*:"
            r"|\byou\s+are\s+now\b"
            r"|\b(reveal|print|show|repeat)\b(\s+\w+){0,2}?\s+system\s+prompt\b"
        ),
    ),
    (
        InjectionCategory.ROLE_MARKER,
        re.compile(
            r"^\s*(#+\s*)?(system|assistant|developer|user|human)\s*(message|prompt)?\s*:"
            r"|<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>"
            r"|\[/?(inst|sys)\]"
            r"|<<\s*/?sys\s*>>"
            r"|</?\s*(system|assistant|developer)\s*>"
        ),
    ),
    (
        InjectionCategory.TOOL_CALL,
        re.compile(
            r"\b(call|invoke|run|use|execute|trigger)\b(\s+the)?\s+`?(" + _TOOL_NAMES + r")\b"
            r"|\b(" + _TOOL_NAMES + r")\s*\("
        ),
    ),
    (
        InjectionCategory.APPROVAL_CLAIM,
        re.compile(
            r"\b(plan|request|job|run|experiment|export)\s+(is|was|has\s+been)\s+(now\s+)?"
            r"(approved|authori[sz]ed|pre-?approved)\b"
            r"|\bapproval\s+(is\s+|was\s+|has\s+been\s+)?granted\b"
            r"|\bapproval_ref\b"
            r"|\byou\s+(are|have\s+been)\s+(now\s+)?(authori[sz]ed|permitted|allowed)\b"
            r"|\b(approved|authori[sz]ed)\s+by\s+the\s+(user|owner|admin\w*|operator)\b"
        ),
    ),
)


def _normalize(line: str) -> str:
    """Case-fold and drop the tricks that hide a phrase from a plain match."""
    text = unicodedata.normalize("NFKC", line)
    text = _ZERO_WIDTH.sub("", text).casefold()
    return re.sub(r"\s+", " ", text)


def classify_line(line: str) -> InjectionCategory | None:
    """The first category whose pattern matches `line`, or None for an ordinary line."""
    normalized = _normalize(line)
    for category, pattern in _PATTERNS:
        if pattern.search(normalized):
            return category
    return None


def screen_text(text: str) -> tuple[str, Counter[InjectionCategory]]:
    """Replace each instruction-like line of `text` with the marker; return the counts."""
    counts: Counter[InjectionCategory] = Counter()
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        category = classify_line(line)
        if category is not None:
            counts[category] += 1
            ending = line[len(line.rstrip("\r\n")) :]
            lines[index] = INJECTION_MARKER + ending
    return ("".join(lines) if counts else text), counts


@dataclass(frozen=True)
class ScreenResult:
    """A screened JSON value and how many lines of each category were withheld."""

    value: JsonValue
    counts: Mapping[InjectionCategory, int] = field(default_factory=dict)

    @property
    def withheld(self) -> int:
        """Total lines withheld."""
        return sum(self.counts.values())


def screen_value(
    value: JsonValue, *, skip_keys: frozenset[str] = PLATFORM_KEYS, bounded: bool = False
) -> ScreenResult:
    """Screen every string value inside a JSON value; keys, numbers and booleans are unchanged.

    The values of dict keys in `skip_keys` (platform-written fields) are left as they are, at any
    depth. With `bounded`, a string with any withheld line becomes `SHORT_INJECTION_MARKER` as a
    whole (`BOUNDED_TOOLS`). Fails closed: a value nested deeper than `MAX_SCREEN_DEPTH` raises
    ValueError instead of passing unscreened text.
    """
    counts: Counter[InjectionCategory] = Counter()

    def walk(item: JsonValue, depth: int) -> JsonValue:
        if depth > MAX_SCREEN_DEPTH:
            raise ValueError("value too deeply nested to screen")
        if isinstance(item, str):
            screened, found = screen_text(item)
            counts.update(found)
            return SHORT_INJECTION_MARKER if bounded and found else screened
        if isinstance(item, list):
            return [walk(element, depth + 1) for element in item]
        if isinstance(item, dict):
            return {
                key: element if key in skip_keys else walk(element, depth + 1)
                for key, element in item.items()
            }
        return item

    screened = walk(value, 0)
    return ScreenResult(screened, dict(counts))


def injection_policy_decision(
    tool: str,
    *,
    call_id: str | None,
    tier: ToolTier | None,
    enabled: bool,
    result: ScreenResult | None = None,
) -> PolicyDecisionPayload:
    """The `policy_decision` record of one screened call (counts only, never the text).

    Off: allow, no deciding rule, reason `defenses.injection_filter is off`. On with nothing found:
    allow by `injection_filter`. On with hits: deny by `injection_filter`, one reason per category,
    in the same `withheld_lines` shape the data firewall uses for withheld log lines.
    """
    if not enabled:
        return PolicyDecisionPayload(
            call_id=call_id,
            tool=tool,
            tier=tier,
            decision=PolicyOutcome.ALLOW,
            reasons=[OFF_REASON],
        )
    if result is None or not result.withheld:
        return PolicyDecisionPayload(
            call_id=call_id,
            tool=tool,
            tier=tier,
            decision=PolicyOutcome.ALLOW,
            decided_by=PolicyRule.INJECTION_FILTER,
            reasons=["passed"],
        )
    return PolicyDecisionPayload(
        call_id=call_id,
        tool=tool,
        tier=tier,
        decision=PolicyOutcome.DENY,
        decided_by=PolicyRule.INJECTION_FILTER,
        reasons=[
            f"category={category.value} withheld_lines={count}"
            for category, count in sorted(result.counts.items())
            if count
        ],
    )


def screen_tool_result(
    tool: str,
    value: JsonValue,
    *,
    enabled: bool,
    call_id: str | None = None,
    tier: ToolTier | None = None,
) -> tuple[JsonValue, PolicyDecisionPayload | None]:
    """Screen one tool result the way the MCP exit does: the value to return and its record.

    A tool outside `SCREENED_TOOLS` is returned unchanged with no record. With the switch off,
    the value is returned unchanged and the record says the filter was off. With it on, the
    screened value and its allow / deny record are returned.
    """
    if tool not in SCREENED_TOOLS:
        return value, None
    if not enabled:
        return value, injection_policy_decision(tool, call_id=call_id, tier=tier, enabled=False)
    result = screen_value(value, bounded=tool in BOUNDED_TOOLS)
    decision = injection_policy_decision(
        tool, call_id=call_id, tier=tier, enabled=True, result=result
    )
    return result.value, decision


__all__ = [
    "BOUNDED_TOOLS",
    "INJECTION_MARKER",
    "OFF_REASON",
    "PLATFORM_KEYS",
    "SCREENED_TOOLS",
    "SHORT_INJECTION_MARKER",
    "InjectionCategory",
    "ScreenResult",
    "classify_line",
    "injection_policy_decision",
    "screen_text",
    "screen_tool_result",
    "screen_value",
]
