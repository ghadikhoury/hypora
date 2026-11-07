"""The last check before a record reaches `results/`, which is committed to a public repository.

No hidden row, hidden-store path, or secret may be in it. A record is built from typed fields, but
some of them are free text (a grader's detail, a run's notes and report), so the finished document
is checked as a whole. A hidden value or store path refuses the record; a credential is replaced;
machine paths become placeholders, so a record names no home directory.
"""

import io
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore[import-untyped]
from pydantic import JsonValue

from hypora.core.llm.sinks import redact
from hypora.evaluation.hidden_split import (
    HIDDEN_FEATURES_FILE,
    HIDDEN_LABELS_FILE,
    HiddenSplit,
)

MIN_VALUE_CHARS: Final = 8
"""Shorter cell values are too likely to appear by chance (a word in a report, a metric)."""


class RecordLeakError(Exception):
    """A record holds something that must never leave the harness; never names the value."""


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)


def _text_paths(value: str, aliases: Mapping[str, str]) -> str:
    for path, alias in aliases.items():
        value = value.replace(path, alias)
    return value


def _replace_paths(value: JsonValue, aliases: Mapping[str, str]) -> JsonValue:
    if isinstance(value, str):
        return _text_paths(value, aliases)
    if isinstance(value, list):
        return [_replace_paths(item, aliases) for item in value]
    if isinstance(value, dict):
        return {
            _text_paths(key, aliases): _replace_paths(item, aliases) for key, item in value.items()
        }
    return value


def _table(content: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(content), engine="pyarrow")


def _cell_texts(frame: pd.DataFrame) -> set[str]:
    """Every cell as text: strings as they are, and floats as the repr a report would print."""
    texts: set[str] = set()
    for name in frame.columns:
        for value in frame[name].astype(object).dropna().unique().tolist():
            if isinstance(value, str):
                texts.add(value)
            elif isinstance(value, (int, float, bool)):
                texts.add(repr(value))
    return texts


def hidden_only_values(split: HiddenSplit) -> frozenset[str]:
    """Distinctive cell values found only in the hidden rows.

    A value that is also visible says nothing, and a short one can match by chance, so neither
    counts. This catches a row dumped into a grader's detail; it can't catch a rephrased value.
    """
    visible = _cell_texts(_table(split.visible_bytes()))
    hidden = _cell_texts(_table(split.hidden_features_bytes())) | _cell_texts(
        _table(split.hidden_labels_bytes())
    )
    distinctive = sorted(v for v in hidden - visible if len(v) >= MIN_VALUE_CHARS)
    return frozenset(distinctive)


@dataclass(frozen=True)
class HiddenRows:
    """Complete hidden-row fingerprints and the column sets of authorized row renderings."""

    tokens: frozenset[tuple[str, ...]] = frozenset()
    field_sets: tuple[frozenset[str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.tokens)


EMPTY_HIDDEN_ROWS: Final = HiddenRows()


def _markdown_values(text: str, fields: frozenset[str]) -> str:
    """Extract values from Markdown field/value pairs without dropping data named like a field."""
    lines = []
    for line in text.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and len(cells) % 2 == 0 and all(name in fields for name in cells[::2]):
            lines.append(", ".join(cells[1::2]))
        else:
            lines.append(line)
    return "\n".join(lines)


_FIELD_NAME = re.compile(r"(?:\"[^\"]*\"|'[^']*'|\w+)\s*[:=]\s*")
_VALUE_TOKEN = re.compile(
    r"[^\W\d_]+(?:[_-][^\W\d_]+)*|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I
)


def _text_tokens(text: str, fields: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Normalize row renderings: CSV, Python/JSON arrays, and named JSON/Python fields.

    Field labels and punctuation aren't cell values. Numbers normalize across integer, float,
    and exponent notation. Words remain tokens, so an isolated common scalar is never a row.
    """
    text = _markdown_values(text, fields) if fields else text
    tokens = []
    for token in _VALUE_TOKEN.findall(_FIELD_NAME.sub("", text)):
        token = token.casefold()
        if token in {"none", "nan", "na", "null"}:
            token = "null"
        else:
            try:
                number = Decimal(token)
                token = str(number.normalize()) if number else "0"
            except DecimalException:
                pass
        tokens.append(token)
    return tuple(tokens)


def _rows(frame: pd.DataFrame) -> set[tuple[str, ...]]:
    return {
        tuple(sorted(tokens))
        for row in frame.itertuples(index=False, name=None)
        if len(tokens := _text_tokens(", ".join(str(cell) for cell in row))) >= 2
    }


def hidden_row_tokens(split: HiddenSplit) -> HiddenRows:
    """All distinctive complete hidden rows, with no dataset-size truncation.

    Include feature-only rows when they have at least two scalar tokens and full rows with the
    label last. Visible rows are excluded to avoid rejecting authorized visible data. Short
    numeric/boolean/string cells are protected as a row, not as isolated common values.
    """
    visible = _table(split.visible_bytes())
    features = _table(split.hidden_features_bytes())
    labels = _table(split.hidden_labels_bytes())
    target = next(column for column in visible.columns if column not in features.columns)
    visible_features = visible[list(features.columns)]
    hidden_full = features.copy()
    hidden_full[target] = labels.iloc[:, 0].to_numpy()
    visible_full = visible[[*features.columns, target]]
    return HiddenRows(
        tokens=frozenset(
            (_rows(features) - _rows(visible_features)) | (_rows(hidden_full) - _rows(visible_full))
        ),
        field_sets=(
            frozenset(str(name) for name in features.columns),
            frozenset(str(name) for name in hidden_full.columns),
            frozenset({*(str(name) for name in features.columns), "target"}),
        ),
    )


def _row_texts(value: JsonValue, field_sets: tuple[frozenset[str], ...]) -> Iterator[str]:
    """Free text and structured rows, without combining unrelated nested metadata."""
    yield from _strings(value)
    yield from _row_containers(value, field_sets)


def _row_containers(value: JsonValue, field_sets: tuple[frozenset[str], ...]) -> Iterator[str]:
    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value):
            yield json.dumps(value)
        for item in value:
            yield from _row_containers(item, field_sets)
    elif isinstance(value, dict):
        if frozenset(value) in field_sets and all(
            not isinstance(item, (dict, list)) for item in value.values()
        ):
            yield json.dumps(value)
        for item in value.values():
            yield from _row_containers(item, field_sets)


def _holds_row(
    text: str, rows: HiddenRows, widths: tuple[int, ...], fields: frozenset[str]
) -> bool:
    tokens = _text_tokens(text, fields)
    return any(
        tuple(sorted(tokens[start : start + width])) in rows.tokens
        for width in widths
        for start in range(len(tokens) - width + 1)
    )


def _holds_any(text: str, values: frozenset[str], lengths: Iterable[int]) -> bool:
    """A set lookup per window and length: fast with many values, unlike one large regex."""
    return any(
        text[start : start + length] in values
        for length in lengths
        for start in range(len(text) - length + 1)
    )


@dataclass(frozen=True)
class RecordGuard:
    """What one eval refuses in its records."""

    forbidden_paths: tuple[str, ...]
    """The hidden store, and anything else no record may name."""
    secrets: tuple[str, ...] = ()
    path_aliases: tuple[tuple[str, str], ...] = ()
    """(machine path, placeholder), most specific first."""

    def clean(
        self,
        document: JsonValue,
        hidden_values: frozenset[str] = frozenset(),
        hidden_rows: HiddenRows = EMPTY_HIDDEN_ROWS,
    ) -> JsonValue:
        """The document ready to store; raises RecordLeakError when it can't be stored at all.

        Checks come first: replacing a home directory would hide a store path under it.
        """
        forbidden = (*self.forbidden_paths, HIDDEN_FEATURES_FILE, HIDDEN_LABELS_FILE)
        lengths = sorted({len(value) for value in hidden_values})
        for text in _strings(document):
            if any(path in text for path in forbidden):
                raise RecordLeakError("the record names the hidden store or its files")
            if lengths and _holds_any(text, hidden_values, lengths):
                raise RecordLeakError("the record holds a value found only in the hidden rows")
        widths = tuple({len(row) for row in hidden_rows.tokens})
        fields = frozenset().union(*hidden_rows.field_sets)
        if hidden_rows and any(
            _holds_row(text, hidden_rows, widths, fields)
            for text in _row_texts(document, hidden_rows.field_sets)
        ):
            raise RecordLeakError("the record holds a complete row found only in the hidden rows")
        return _replace_paths(redact(document, self.secrets), dict(self.path_aliases))

    def safe_text(self, text: str) -> str:
        """An error message fit for a record: store paths, secrets, and machine paths removed."""
        for path in self.forbidden_paths:
            text = text.replace(path, "<hidden-store>")
        cleaned = _replace_paths(redact(text, self.secrets), dict(self.path_aliases))
        assert isinstance(cleaned, str)
        return cleaned


def machine_aliases(runs_root: Path, repo_root: Path) -> tuple[tuple[str, str], ...]:
    """Placeholders for the paths a record might mention, most specific first."""
    return (
        (str(runs_root), "<runs>"),
        (str(repo_root), "<repo>"),
        (str(Path.home()), "~"),
    )


__all__ = [
    "HiddenRows",
    "RecordGuard",
    "RecordLeakError",
    "hidden_only_values",
    "hidden_row_tokens",
    "machine_aliases",
]
