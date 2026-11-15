"""The folding every screen shares (D-048): disguises go, ASCII text is only case-folded."""

import string

import pytest

from hypora.core.policy.textfold import fold_text

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


def test_folding_is_idempotent_and_keeps_line_breaks() -> None:
    text = "\uff21\u200b\u00e9\n\u0661\u2028x\r\ny"
    once = fold_text(text)
    assert fold_text(once) == once
    assert once.splitlines() == ["ae", "1", "x", "y"]
