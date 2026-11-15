"""The foldings of model-facing text that every screen compares with something (D-048).

The data firewall compares text with dataset cells, the injection screen with instruction
patterns, A's code excerpt with the platform's own markers. Each comparison is only as strong as
its weakest spelling: before D-048 the firewall withheld `patient-0005` but passed the same text
written with full-width digits (U+FF10..U+FF19), which any reader, model or human, reads as the
same value.

Two foldings, because the two sides of a comparison need different strength:

`light_fold` keeps every character that carries meaning and removes only what no reader sees:

- canonical composition (NFC), so a decomposed accent equals the composed one;
- width variants (U+3000 and the U+FF00..U+FFEF block): full-width brackets, commas and letters
  are their ASCII forms;
- invisible characters: format characters (Cf: zero-width spaces and joiners, the soft hyphen,
  direction marks and overrides, word joiners, the byte-order mark, tag characters), the
  combining grapheme joiner, variation selectors and the blank Hangul fillers;
- case (`str.casefold`).

`fold_text` (the full fold) also removes the trivially reversible disguises:

- compatibility forms (NFKD, recomposed at the end): ligatures, superscripts, circled,
  parenthesized and mathematical letters and digits, odd spaces; a parenthesized or dotted digit
  (U+2474..U+249B) is just the digit, so its brackets do not split a token;
- combining marks (Mn, Me): accents, overlay strokes and underlines;
- every decimal digit of any script, and the circled digits NFKD leaves alone, become ASCII digits;
- dashes and the minus sign (category Pd, U+2212) become `-`; look-alike colons become `:`;
- common Cyrillic, Greek and IPA letters that look like Latin ones become those Latin letters.

**ASCII text** is returned only case-folded by both foldings: numbers, metric values, column
names, code and log lines compare exactly as before D-048.

**Which side gets which fold.** The firewall keys a cell under the light fold and checks a line
under both foldings. A disguised copy of a value (full-width digits, Cyrillic letters in a Latin
value) folds back to it. A cell is never folded further than its light form, so an ordinary ASCII
word does not match a non-ASCII cell that would only equal it after folding the cell (`naive`
does not match an accented cell, `max` does not match a Cyrillic cell that looks like it). The
injection screen and the marker check compare the full fold with fixed ASCII patterns.

**What it is not.** A full confusables table (Unicode TR39) is not applied: letters of other
scripts that resemble Latin, or a value spelled out, encoded, reversed or split across lines, are
not folded. Those residual risks are listed in `docs/notes/live-run-hardening.md` (D-048).
"""

import unicodedata
from typing import Final

_DROPPED_CATEGORIES: Final = frozenset({"Cf", "Mn", "Me"})
_DASH: Final = "-"
_MINUS_SIGN: Final = chr(0x2212)

_INVISIBLE: Final = frozenset(
    {
        0x034F,  # combining grapheme joiner
        0x115F,  # Hangul choseong filler
        0x1160,  # Hangul jungseong filler
        0x3164,  # Hangul filler
        0xFFA0,  # half-width Hangul filler
        *range(0xFE00, 0xFE10),  # variation selectors
        *range(0xE0100, 0xE01F0),  # variation selectors supplement
    }
)
"""Characters no reader sees that are not format characters (Cf)."""

_COLONS: Final = frozenset({0x2236, 0xA789, 0x02D0, 0x0589, 0x05C3, 0xFE13, 0xFE55, 0xFF1A})
"""Ratio, modifier letter colon, triangular colon, Armenian and Hebrew colons, presentation and
full-width colons: each becomes `:` in the full fold."""


def _circled_digits() -> dict[int, str]:
    """Number glyphs that NFKD does not decompose: negative and sans-serif circled digits."""
    glyphs: dict[int, str] = {0x24FF: "0", 0x24EA: "0"}
    for start, first in ((0x2776, 1), (0x2780, 1), (0x278A, 1), (0x24F5, 1), (0x24EB, 11)):
        for offset in range(10):
            glyphs[start + offset] = str(first + offset)
    return glyphs


_NUMBER_GLYPHS: Final = _circled_digits()

# Applied after `casefold`, so only lowercase forms are listed (an uppercase look-alike folds to
# its lowercase letter first). Each letter maps to one Latin letter: the one its uppercase or
# lowercase form imitates. Code points only, so the source holds no look-alike characters.
_LOOKALIKES: Final[dict[int, str]] = {
    # Cyrillic small letters
    0x0430: "a",  # a
    0x0432: "b",  # ve (capital looks like B)
    0x0435: "e",  # ie
    0x043A: "k",  # ka
    0x043C: "m",  # em (capital looks like M)
    0x043D: "h",  # en (capital looks like H)
    0x043E: "o",  # o
    0x0440: "p",  # er
    0x0441: "c",  # es
    0x0442: "t",  # te (capital looks like T)
    0x0443: "y",  # u
    0x0445: "x",  # ha
    0x0455: "s",  # dze
    0x0456: "i",  # Byelorussian-Ukrainian i
    0x0458: "j",  # je
    0x04AF: "y",  # straight u
    0x04BB: "h",  # shha
    0x04CF: "l",  # palochka
    0x0501: "d",  # komi de
    0x051B: "q",  # qa
    0x051D: "w",  # we
    # Greek small letters
    0x03B1: "a",  # alpha
    0x03B2: "b",  # beta
    0x03B5: "e",  # epsilon
    0x03B6: "z",  # zeta
    0x03B7: "h",  # eta (capital looks like H)
    0x03B9: "i",  # iota
    0x03BA: "k",  # kappa
    0x03BC: "m",  # mu (capital looks like M)
    0x03BD: "n",  # nu (capital looks like N)
    0x03BF: "o",  # omicron
    0x03C1: "p",  # rho
    0x03C4: "t",  # tau
    0x03C5: "y",  # upsilon (capital looks like Y)
    0x03C7: "x",  # chi
    # Latin and IPA letters outside ASCII that NFKD leaves alone
    0x0131: "i",  # dotless i
    0x0237: "j",  # dotless j
    0x0251: "a",  # alpha (IPA)
    0x0261: "g",  # script g
    0x026A: "i",  # small capital i
    0x0274: "n",  # small capital n
    0x028F: "y",  # small capital y
}


def _is_invisible(char: str) -> bool:
    return ord(char) in _INVISIBLE or unicodedata.category(char) == "Cf"


def _width(char: str) -> str:
    code = ord(char)
    if code == 0x3000 or 0xFF00 <= code <= 0xFFEF:
        return unicodedata.normalize("NFKC", char)
    return char


def light_fold(text: str) -> str:
    """`text` composed (NFC), width variants and invisible characters removed, case-folded.

    Keeps every character a reader sees, so a non-ASCII value is never turned into a different
    ASCII word. ASCII text is only case-folded.
    """
    if text.isascii():
        return text.casefold()
    composed = unicodedata.normalize("NFC", text)
    narrowed = "".join(_width(char) for char in composed)
    visible = "".join(char for char in narrowed if not _is_invisible(char))
    return unicodedata.normalize("NFC", visible.casefold())


def _decompose(char: str) -> str:
    """One character's compatibility decomposition; an enclosed letter or digit loses its frame.

    `(5)` from a parenthesized digit or `1.` from a dotted digit would split a token on the
    tokenizer's delimiters, so the ASCII punctuation is dropped when the decomposition also holds
    a letter or digit. A decomposition without one (full-width brackets) keeps its punctuation.
    """
    if char.isascii():
        return char
    glyph = _NUMBER_GLYPHS.get(ord(char))
    if glyph is not None:
        return glyph
    parts = unicodedata.normalize("NFKD", char)
    if parts != char and any(part.isalnum() for part in parts):
        parts = "".join(
            part for part in parts if part.isalnum() or not part.isascii() or part.isspace()
        )
    return parts


def _fold_char(char: str) -> str:
    if _is_invisible(char):
        return ""
    category = unicodedata.category(char)
    if category in _DROPPED_CATEGORIES:
        return ""
    if category == "Nd":
        return str(unicodedata.decimal(char))
    if category == "Pd" or char == _MINUS_SIGN:
        return _DASH
    if ord(char) in _COLONS:
        return ":"
    return _LOOKALIKES.get(ord(char), char)


def fold_text(text: str) -> str:
    """`text` with every disguise in the module docstring removed, case-folded.

    ASCII text is only case-folded. Line breaks are kept (they are not format characters), so a
    caller may fold a whole text and then split it into lines.
    """
    if text.isascii():
        return text.casefold()
    decomposed = "".join(_decompose(char) for char in text).casefold()
    # casefold can produce new compatibility or combining characters (for example from U+0130),
    # and per-character decomposition skips canonical reordering: decompose the whole once more.
    decomposed = unicodedata.normalize("NFKD", decomposed)
    folded = "".join(_fold_char(char) for char in decomposed)
    return unicodedata.normalize("NFKC", folded)


__all__ = ["fold_text", "light_fold"]
