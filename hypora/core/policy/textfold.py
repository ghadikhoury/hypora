"""One folding of model-facing text for every screen that compares it with something (D-048).

The data firewall compares text with dataset cells, the injection screen with instruction
patterns, A's code excerpt with the platform's own markers. Each comparison is only as strong as
its weakest spelling: before D-048 the firewall withheld `patient-0005` but passed the same text
written with full-width digits (U+FF10..U+FF19), which any reader, model or human, reads as the
same value.

`fold_text` removes the trivially reversible disguises, applied to *both* sides of a comparison
(the firewall folds the cells it indexes and the lines it checks the same way):

- compatibility forms (NFKD, recomposed at the end): full-width and half-width letters, digits
  and punctuation, ligatures, superscripts, circled and mathematical letters, odd spaces;
- invisible format characters (Unicode category Cf): zero-width spaces and joiners, the soft
  hyphen, direction marks and overrides, word joiners, the byte-order mark, tag characters;
- combining marks (Mn, Me): accents, variation selectors, overlay strokes and underlines, so an
  accented or underlined copy of a word compares as the plain word;
- case (`str.casefold`);
- every decimal digit of any script (Arabic-Indic, Devanagari, ...) becomes its ASCII digit;
- dashes and the minus sign (category Pd, U+2212) become `-`;
- Cyrillic and Greek letters that look like Latin ones become those Latin letters.

**False positives do not change for ordinary text.** ASCII text (numbers, metric values,
column names, code, log lines) is returned only case-folded, exactly what the screens did
before. A fold can only make two strings *more* alike, so a screen that matched before still
matches; the new matches are non-ASCII spellings of the same characters.

**What it is not.** A full confusables table (Unicode TR39) is not applied: letters of other
scripts that resemble Latin, or a value spelled out, encoded, reversed or split across lines, are
not folded. Those residual risks are listed in `docs/notes/live-run-hardening.md` (D-048).
"""

import unicodedata
from typing import Final

_DROPPED_CATEGORIES: Final = frozenset({"Cf", "Mn", "Me"})
_DASH: Final = "-"
_MINUS_SIGN: Final = chr(0x2212)

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
    # Latin letters outside ASCII that NFKD leaves alone
    0x0131: "i",  # dotless i
    0x0237: "j",  # dotless j
    0x0261: "g",  # script g
}


def _fold_char(char: str) -> str:
    category = unicodedata.category(char)
    if category in _DROPPED_CATEGORIES:
        return ""
    if category == "Nd":
        return str(unicodedata.decimal(char))
    if category == "Pd" or char == _MINUS_SIGN:
        return _DASH
    return _LOOKALIKES.get(ord(char), char)


def fold_text(text: str) -> str:
    """`text` with the disguises in the module docstring removed, case-folded.

    ASCII text is only case-folded. Line breaks are kept (they are not format characters), so a
    caller may fold a whole text and then split it into lines.
    """
    if text.isascii():
        return text.casefold()
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    # casefold can produce new compatibility or combining characters (for example from U+0130),
    # so decompose once more before dropping marks.
    decomposed = unicodedata.normalize("NFKD", decomposed)
    folded = "".join(_fold_char(char) for char in decomposed)
    return unicodedata.normalize("NFKC", folded)


__all__ = ["fold_text"]
