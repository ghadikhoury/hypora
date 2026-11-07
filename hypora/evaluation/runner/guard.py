"""The last check before a record reaches `results/`, which is committed to a public repository.

No hidden row, hidden-store path, or secret may be in it. A record is built from typed fields, but
some of them are free text (a grader's detail, a run's notes and report), so the finished document
is checked as a whole. A hidden value or store path refuses the record; a credential is replaced;
machine paths become placeholders, so a record names no home directory.
"""

import io
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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
MAX_HIDDEN_VALUES: Final = 50_000


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
            elif isinstance(value, float):
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
    return frozenset(distinctive[:MAX_HIDDEN_VALUES])


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

    def clean(self, document: JsonValue, hidden_values: frozenset[str] = frozenset()) -> JsonValue:
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


__all__ = ["RecordGuard", "RecordLeakError", "hidden_only_values", "machine_aliases"]
