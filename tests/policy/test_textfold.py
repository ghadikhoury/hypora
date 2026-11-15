"""The foldings the screens share (D-048): disguises go, ASCII text is only case-folded."""

import string

import pytest

from hypora.core.policy.textfold import fold_text, light_fold

ASCII_SAMPLES = [
    string.printable,
    "Iteration 12, loss = 0.4321, val_loss = 0.4510",
    "    ('num', StandardScaler(), numeric),",
    "Trial 3 finished with value: 0.912345 and parameters: {'lr': 0.01234}",
    "mean radius,mean texture,target",
]


@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_ascii_text_is_only_case_folded(text: str) -> None:
    assert fold_text(text) == text.casefold()


@pytest.mark.parametrize(
    ("disguised", "plain"),
    [
        (
            "\uff30\uff41\uff54\uff49\uff45\uff4e\uff54\uff0d\uff10\uff10\uff10\uff15",
            "patient-0005",
        ),
        ("pa\u00adti\u200bent\u2060", "patient"),
        ("Zo\u00eb", "zoe"),
        ("\u0660\u0661\u0662", "012"),
        ("a\u2013b\u2212c\u2011d", "a-b-c-d"),
        ("\u0420\u0410\u0422\u0406\u0415NT", "patient"),  # Cyrillic capitals
        ("\ufb01t", "fit"),  # ligature
        ("x\u00b2", "x2"),  # superscript
    ],
)
def test_disguises_fold_to_the_plain_spelling(disguised: str, plain: str) -> None:
    assert fold_text(disguised) == plain


@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_the_light_fold_also_only_case_folds_ascii(text: str) -> None:
    assert light_fold(text) == text.casefold()


@pytest.mark.parametrize(
    ("text", "light"),
    [
        ("Zo" + chr(0x00EB), "zo" + chr(0x00EB)),  # accents stay
        ("Zoe" + chr(0x0308), "zo" + chr(0x00EB)),  # composed (NFC)
        (chr(0x041C) + chr(0x0410) + chr(0x0425), chr(0x043C) + chr(0x0430) + chr(0x0445)),
        ("x" + chr(0x00B2) + "y", "x" + chr(0x00B2) + "y"),  # superscript stays
        (chr(0x2116) + " 17", chr(0x2116) + " 17"),  # numero sign stays
        (chr(0xFF21) + chr(0xFF08) + "b" + chr(0xFF09), "a(b)"),  # width variants
        ("pa" + chr(0x00AD) + "ti" + chr(0x200B) + "ent" + chr(0x3164), "patient"),  # invisible
    ],
)
def test_the_light_fold_keeps_every_visible_character(text: str, light: str) -> None:
    assert light_fold(text) == light


@pytest.mark.parametrize(
    ("disguised", "plain"),
    [
        ("patient-000" + chr(0x2478), "patient-0005"),  # parenthesized digit loses its brackets
        ("patient-000" + chr(0x248C), "patient-0005"),  # dotted digit loses its dot
        (chr(0x24FF) + chr(0x277A) + chr(0x2789) + chr(0x24EB), "05101" + "1"),  # circled digits
        ("[platform" + chr(0xA789) + " x" + chr(0x2236) + "]", "[platform: x:]"),  # colons
        (chr(0x0251) + chr(0x026A), "ai"),  # IPA look-alikes
        (chr(0xFF08) + "x" + chr(0xFF09), "(x)"),  # full-width brackets keep their punctuation
    ],
)
def test_review_gaps_fold_to_the_plain_spelling(disguised: str, plain: str) -> None:
    assert fold_text(disguised) == plain


def test_folding_is_idempotent_and_keeps_line_breaks() -> None:
    text = "\uff21\u200b\u00e9\n\u0661\u2028x\r\ny"
    once = fold_text(text)
    assert fold_text(once) == once
    assert once.splitlines() == ["ae", "1", "x", "y"]
