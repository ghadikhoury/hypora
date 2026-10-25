"""Data firewall v1 (spec §4.3; D-018): find raw dataset rows in text that would reach an LLM.

Raw rows never enter LLM context (spec §4.3). Tool outputs are typed, bounded summaries, so the
only place a row can hide is inside a string: a log line, an error message, a label. This module
looks for the dataset's own cells in such text. It knows the data because it runs in platform code
next to the dataset (Mode 1). It never returns or records the matched values, only counts.

**When a cell "appears" in a line of text.** A line is split into tokens on whitespace and the
delimiters pandas, CSV, and Python reprs use (`, ; | ( ) [ ] { } < > = ' "`).
- A numeric cell appears when a numeric token shows it at the token's own precision: the token
  `0.11840` (5 decimals) shows the cell 0.1184, and `13.71` shows 13.7100003. So a pandas
  `df.head()` print, which rounds, is still recognized. Tokens with fewer than
  `min_significant_digits` significant digits (`3`, `42`, `0.5`) carry too little information and
  are ignored.
- A text cell appears when its normalized text (case-folded, whitespace collapsed) equals a run of
  up to `max_ngram` consecutive tokens; a longer text cell is keyed by its first `max_ngram` words.
- Only **distinctive** cells count: the (column, token) pair matches at most `max_rows_per_value`
  rows, the value is not one of the column's `excluded_top_values` most frequent values, and it is
  not the column's minimum or maximum. Profiles legitimately disclose frequent labels and ranges
  (D-009), and a common value proves nothing about which row is being shown.

**Rules.** Text is examined line by line, because a printed row sits on one line.
- `row_on_line`: one line shows `row_cells` (3) cells of the same row (all of them, for a dataset
  with fewer columns).
- `column_on_line`: one line shows `column_cells` (8) cells of the same column, e.g. a printed list.
- `scan_limit`: the text is larger than `max_scan_chars`; fail closed instead of skipping text.
- Untrusted text (sandbox logs, trial error messages) is held to a stricter line rule by
  `screen_untrusted`: `row_cells_untrusted` (2) cells of one row, `column_cells_untrusted` (4)
  cells of one column, or any distinctive text cell (`text_cell`) withholds that line.

`check_payload` applies the boundary rules to every string (and dict key) of a JSON value; the MCP
server runs it on every tool result (`hypora/mcp_server/server.py`). Residual risks are listed in
docs/notes/mcp-server.md: rows made only of common values, cells printed one per line by untrusted
code that the screen did not see, and encoded or transformed values.
"""

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_SPLIT = re.compile(r"[\s,;|()\[\]{}<>=\"'`]+")
_NUMBER = re.compile(r"^[+-]?(?P<int>\d*)(?:\.(?P<frac>\d*))?(?:[eE](?P<exp>[+-]?\d+))?$")


class FirewallRule(StrEnum):
    """Which rule decided (recorded in `policy_decision` events; never the matched values)."""

    ROW_ON_LINE = "row_on_line"
    COLUMN_ON_LINE = "column_on_line"
    TEXT_CELL = "text_cell"
    SCAN_LIMIT = "scan_limit"


@dataclass(frozen=True)
class FirewallConfig:
    """Thresholds of the detector. Defaults are D-018's; tests may tighten or loosen them."""

    row_cells: int = 3
    column_cells: int = 8
    row_cells_untrusted: int = 2
    column_cells_untrusted: int = 4
    max_rows_per_value: int = 3
    excluded_top_values: int = 5
    min_significant_digits: int = 3
    min_text_chars: int = 3
    max_ngram: int = 4
    max_scan_chars: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.row_cells,
            self.column_cells,
            self.row_cells_untrusted,
            self.column_cells_untrusted,
            self.max_rows_per_value,
            self.min_significant_digits,
            self.min_text_chars,
            self.max_ngram,
            self.max_scan_chars,
        )
        if min(values) < 1 or self.excluded_top_values < 0:
            raise ValueError("firewall thresholds must be positive")


@dataclass(frozen=True)
class Verdict:
    """Outcome of a check. `details` holds counts only, safe for events and error messages."""

    blocked: bool
    rule: FirewallRule | None = None
    details: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _NumericToken:
    value: float
    tolerance: float


@dataclass(frozen=True)
class _NumericColumn:
    frame: int
    column: int
    values: np.ndarray  # sorted finite values
    rows: np.ndarray  # row index of each sorted value
    excluded: np.ndarray  # frequent values, min, max


@dataclass
class _LineMatches:
    """Cells shown on one line, as (token, frame, row, column).

    Counts are of distinct tokens: one printed value that happens to match several rows (or the
    same row in several columns) is one appearance, because a real dump prints each cell as its
    own token.
    """

    cells: set[tuple[int, int, int, int]] = field(default_factory=set)
    text_cells: int = 0

    def max_row_cells(self) -> dict[int, int]:
        """Per frame: the most distinct tokens that show cells of one row (in distinct columns)."""
        per_row: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for token, frame, row, column in self.cells:
            per_row[frame, row].setdefault(token, column)
        best: dict[int, int] = defaultdict(int)
        for (frame, _), by_token in per_row.items():
            best[frame] = max(best[frame], len(set(by_token.values())))
        return best

    def max_column_cells(self) -> int:
        """The most distinct tokens that show cells of one column."""
        per_column: dict[tuple[int, int], set[int]] = defaultdict(set)
        for token, frame, _, column in self.cells:
            per_column[frame, column].add(token)
        return max((len(tokens) for tokens in per_column.values()), default=0)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _tokens(line: str) -> list[str]:
    tokens = (part.rstrip(":.").casefold() for part in _SPLIT.split(line))
    return [token for token in tokens if token]


def _numeric_token(token: str, min_significant_digits: int) -> _NumericToken | None:
    """A numeric token with its display precision, or None if it is not informative enough."""
    match = _NUMBER.match(token)
    if match is None or not (match["int"] or match["frac"]):
        return None
    digits = (match["int"] or "") + (match["frac"] or "")
    if len(digits.lstrip("0")) < min_significant_digits:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    decimals = len(match["frac"] or "") - int(match["exp"] or 0)
    tolerance = 0.5 * 10.0 ** (-decimals) * (1 + 1e-9) + 1e-12 * abs(value)
    return _NumericToken(value, tolerance)


class DataFirewall:
    """Indexes the authorized dataset(s) once; then checks text against their cells.

    Pass every frame an LLM-facing string could echo: the raw dataset and the contract-applied
    frame trial code actually sees (after non-learned fixes such as unit conversions).
    """

    def __init__(self, frames: Sequence[pd.DataFrame], config: FirewallConfig | None = None):
        self.config = config or FirewallConfig()
        self._n_columns: list[int] = [len(frame.columns) for frame in frames]
        self._numeric: list[_NumericColumn] = []
        self._text: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        for frame_number, frame in enumerate(frames):
            for column_number, name in enumerate(frame.columns):
                self._index_column(frame_number, column_number, frame[name])

    # --- index ------------------------------------------------------------------------------------

    def _index_column(self, frame: int, column: int, series: pd.Series) -> None:
        series = series.reset_index(drop=True)
        if pd.api.types.is_bool_dtype(series):
            return  # two values: never distinctive
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size == 0:
                return
            order = finite[np.argsort(values[finite], kind="stable")]
            frequent = (
                pd.Series(values[finite]).value_counts().head(self.config.excluded_top_values)
            )
            excluded = np.concatenate(
                [frequent.index.to_numpy(dtype=float), [values[order[0]], values[order[-1]]]]
            )
            self._numeric.append(_NumericColumn(frame, column, values[order], order, excluded))
            return
        present = series.dropna()
        if present.empty:
            return
        frequent = set(present.value_counts().head(self.config.excluded_top_values).index)
        keys: dict[str, list[int]] = defaultdict(list)
        for row, value in present.items():
            if value in frequent:
                continue
            text = _normalize(str(value))
            if len(text) < self.config.min_text_chars:
                continue
            keys[" ".join(text.split(" ")[: self.config.max_ngram])].append(int(row))
        for key, rows in keys.items():
            if len(rows) <= self.config.max_rows_per_value:
                self._text[key].append((frame, column, np.asarray(rows)))

    # --- matching ---------------------------------------------------------------------------------

    def _match_line(self, line: str) -> _LineMatches:
        found = _LineMatches()
        tokens = _tokens(line)
        if not tokens:
            return found
        token_ids: dict[str, int] = {}
        for start in range(len(tokens)):
            for size in range(1, self.config.max_ngram + 1):
                if start + size > len(tokens):
                    break
                key = " ".join(tokens[start : start + size])
                if key in token_ids:
                    continue
                token_id = token_ids[key] = len(token_ids)
                for frame, column, rows in self._text.get(key, ()):
                    found.text_cells += len(rows)
                    found.cells.update((token_id, frame, int(row), column) for row in rows)
        for token in dict.fromkeys(tokens):
            number = _numeric_token(token, self.config.min_significant_digits)
            if number is None:
                continue
            token_id = token_ids[token]
            for numeric in self._numeric:
                values = numeric.values
                low = int(np.searchsorted(values, number.value - number.tolerance, side="left"))
                high = int(np.searchsorted(values, number.value + number.tolerance, side="right"))
                if high == low:
                    continue
                keep = ~np.isin(values[low:high], numeric.excluded)
                rows = numeric.rows[low:high][keep]
                if 0 < rows.size <= self.config.max_rows_per_value:
                    found.cells.update(
                        (token_id, numeric.frame, int(row), numeric.column) for row in rows
                    )
        return found

    def _row_threshold(self, frame: int, cells: int) -> int:
        return max(1, min(cells, self._n_columns[frame]))

    def _row_on_line(self, matches: _LineMatches, cells: int) -> int:
        """The largest row hit on this line if it reaches the threshold, else 0."""
        hits = [
            count
            for frame, count in matches.max_row_cells().items()
            if count >= self._row_threshold(frame, cells)
        ]
        return max(hits, default=0)

    # --- public API -------------------------------------------------------------------------------

    def check_lines(self, lines: Iterable[str]) -> Verdict:
        """Boundary rules over lines of text: block on a row or a column dump on one line."""
        scanned = 0
        chars = 0
        for line in lines:
            chars += len(line)
            if chars > self.config.max_scan_chars:
                return Verdict(True, FirewallRule.SCAN_LIMIT, {"chars": chars, "lines": scanned})
            scanned += 1
            matches = self._match_line(line)
            if row := self._row_on_line(matches, self.config.row_cells):
                return Verdict(
                    True, FirewallRule.ROW_ON_LINE, {"row_cells": row, "line": scanned - 1}
                )
            column = matches.max_column_cells()
            if column >= self.config.column_cells:
                return Verdict(
                    True, FirewallRule.COLUMN_ON_LINE, {"column_cells": column, "line": scanned - 1}
                )
        return Verdict(False, details={"lines": scanned})

    def check_payload(self, payload: object) -> Verdict:
        """Apply the boundary rules to every string and dict key of a JSON-like value."""
        return self.check_lines(line for text in _strings(payload) for line in text.splitlines())

    def screen_untrusted(self, lines: Sequence[str]) -> tuple[list[str | None], dict[str, int]]:
        """Withhold every line of trial-produced text that shows data; None marks a withheld line.

        Returns the screened lines and the count per rule (for the caller's policy event).
        """
        screened: list[str | None] = []
        counts: dict[str, int] = defaultdict(int)
        for line in lines:
            if len(line) > self.config.max_scan_chars:
                counts[FirewallRule.SCAN_LIMIT.value] += 1
                screened.append(None)
                continue
            matches = self._match_line(line)
            rule: FirewallRule | None = None
            if self._row_on_line(matches, self.config.row_cells_untrusted):
                rule = FirewallRule.ROW_ON_LINE
            elif matches.max_column_cells() >= self.config.column_cells_untrusted:
                rule = FirewallRule.COLUMN_ON_LINE
            elif matches.text_cells:
                rule = FirewallRule.TEXT_CELL
            if rule is None:
                screened.append(line)
            else:
                counts[rule.value] += 1
                screened.append(None)
        return screened, dict(counts)


def _strings(value: object) -> Iterator[str]:
    """Every string in a JSON-like value, dict keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)
