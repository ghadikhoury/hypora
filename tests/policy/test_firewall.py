"""Data firewall v1 (D-018): raw rows in text are recognized; summaries and ordinary logs pass.

Real UCI data (scikit-learn's bundled breast cancer and wine sets) is used wherever the claim is
about realistic data; small synthetic frames only where a test needs a specific planted shape
(D-015 allows synthetic data in unit tests).
"""

import json

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.datasets import (  # type: ignore[import-untyped]
    load_breast_cancer,
    load_iris,
    load_wine,
)

from hypora.core.policy.firewall import DataFirewall, FirewallConfig, FirewallRule
from hypora.core.schemas import TaskType
from hypora.profiling.tabular import profile_tabular


@pytest.fixture(scope="module")
def cancer() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


@pytest.fixture(scope="module")
def cancer_firewall(cancer: pd.DataFrame) -> DataFirewall:
    return DataFirewall([cancer])


def lines(text: str) -> list[str]:
    return text.splitlines()


# --- a deliberate dump is blocked ----------------------------------------------------------------


def test_a_deliberate_df_head_dump_is_blocked(cancer, cancer_firewall):
    verdict = cancer_firewall.check_lines(lines(str(cancer.head())))
    assert verdict.blocked
    assert verdict.rule is FirewallRule.ROW_ON_LINE
    assert verdict.details["row_cells"] >= 3


@pytest.mark.parametrize(
    "render",
    [
        lambda df: df.head().to_csv(index=False),
        lambda df: df.sample(5, random_state=3).to_string(),
        lambda df: str(df.iloc[7].to_dict()),
        lambda df: json.dumps(df.iloc[[3]].to_dict(orient="records")),
        lambda df: df.iloc[:, :8].head(3).to_string(),
        lambda df: str(df.to_numpy()[:2]),
    ],
    ids=["csv", "sample", "row_dict", "records_json", "narrow_table", "numpy"],
)
def test_common_row_renderings_are_blocked(cancer, cancer_firewall, render):
    verdict = cancer_firewall.check_lines(lines(render(cancer)))
    assert verdict.blocked and verdict.rule is FirewallRule.ROW_ON_LINE


def test_a_printed_column_is_blocked(cancer, cancer_firewall):
    text = str(cancer["mean area"].iloc[:12].tolist())
    verdict = cancer_firewall.check_lines([text])
    assert verdict.blocked and verdict.rule is FirewallRule.COLUMN_ON_LINE


def test_rows_inside_a_json_payload_are_found_in_values_and_keys(cancer, cancer_firewall):
    row = " ".join(str(value) for value in cancer.iloc[11].tolist()[:6])
    assert cancer_firewall.check_payload({"logs": [{"text": row}]}).blocked
    assert cancer_firewall.check_payload({"per_group": {row: 1.0}}).blocked


def test_text_values_count_too():
    frame = pd.DataFrame(
        {
            "patient": [f"patient-{i:04d}" for i in range(40)],
            "site": [f"clinic {chr(65 + i % 26)}{i}" for i in range(40)],
            "dose_mg": np.linspace(10.25, 50.75, 40),
            "label": [i % 2 for i in range(40)],
        }
    )
    firewall = DataFirewall([frame])
    row = frame.iloc[5]
    printed = f"{row['patient']} | {row['site']} | {row['dose_mg']:.2f}"
    assert firewall.check_lines([printed]).rule is FirewallRule.ROW_ON_LINE
    screened, counts = firewall.screen_untrusted([f"processing {row['patient']}"])
    assert screened == [None] and counts == {"text_cell": 1}


# --- disguised copies of a value are the value (D-048) -------------------------------------------


def _patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": [f"patient-{i:04d}" for i in range(40)],
            "site": [f"clinic {chr(65 + i % 26)}{i}" for i in range(40)],
            "dose_mg": np.linspace(10.25, 50.75, 40),
            "label": [i % 2 for i in range(40)],
        }
    )


def _fullwidth(text: str) -> str:
    """Every printable ASCII character as its full-width form; spaces as ideographic spaces."""
    return "".join(
        "\u3000" if c == " " else chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text
    )


def _digits(text: str, zero: int) -> str:
    """ASCII digits rewritten in another script's decimal digits (`zero` is its digit 0)."""
    return "".join(chr(zero + int(c)) if c.isdigit() else c for c in text)


DISGUISED_PATIENT = {
    "fullwidth_digits": "patient-\uff10\uff10\uff10\uff15",
    "fullwidth_all": _fullwidth("patient-0005"),
    "arabic_indic_digits": _digits("patient-0005", 0x0660),
    "devanagari_digits": _digits("patient-0005", 0x0966),
    "math_bold_digits": _digits("patient-0005", 0x1D7CE),
    "zero_width_space": "pat\u200bient-00\u200b05",
    "zero_width_joiner": "patient\u200d-0005",
    "soft_hyphen": "pa\u00adtient-0005",
    "word_joiner_and_bom": "\u2060patient-\ufeff0005",
    "direction_override": "\u202epatient-0005\u202c",
    "combining_marks": "pa\u0332tient-00\u03050\u03355",
    "variation_selector": "patient\ufe0f-0005",
    "cyrillic_letters": "p\u0430ti\u0435nt-0005",
    "greek_letters": "pa\u03c4ien\u03c4-0005",
    "non_breaking_hyphen": "patient\u20110005",
    "en_dash": "patient\u20130005",
    "minus_sign": "patient\u22120005",
    "circled_letters": "\u24df\u24d0\u24e3\u24d8\u24d4\u24dd\u24e3-0005",
    "upper_case": "PATIENT-0005",
    # Added after review (PR #48): the reviewer's full caught list and the non-blocking gaps.
    "superscript_digits": "patient-" + "".join(map(chr, (0x2070, 0x2070, 0x2070, 0x2075))),
    "subscript_digits": _digits("patient-0005", 0x2080),
    "math_monospace_digits": _digits("patient-0005", 0x1D7F6),
    "bengali_digits": _digits("patient-0005", 0x09E6),
    "thai_digits": _digits("patient-0005", 0x0E50),
    "zero_width_non_joiner": "pati" + chr(0x200C) + "ent-0005",
    "left_to_right_mark": "patient-" + chr(0x200E) + "0005",
    "bidi_isolates": chr(0x2066) + "patient-0005" + chr(0x2069),
    "tag_characters": "patient" + chr(0xE0041) + "-0005",
    "mongolian_vowel_separator": "pat" + chr(0x180E) + "ient-0005",
    "combining_grapheme_joiner": "pati" + chr(0x034F) + "ent-0005",
    "invisible_times": "patient-00" + chr(0x2062) + "05",
    "variation_selector_supplement": "patient" + chr(0xE0100) + "-0005",
    "hangul_filler": "pa" + chr(0x3164) + "tient-00" + chr(0x3164) + "05",
    "halfwidth_hangul_filler": "patient" + chr(0xFFA0) + "-0005",
    "choseong_jungseong_fillers": "pat" + chr(0x115F) + "ient-" + chr(0x1160) + "0005",
    "parenthesized_digit": "patient-000" + chr(0x2478),
    "dotted_digit": "patient-000" + chr(0x248C),
    "negative_circled_digits": "patient-" + chr(0x24FF) * 3 + chr(0x277A),
    "ipa_alpha": "p" + chr(0x0251) + "tient-0005",
    "small_capital_i": "pat" + chr(0x026A) + "ent-0005",
}


@pytest.mark.parametrize("disguised", DISGUISED_PATIENT.values(), ids=DISGUISED_PATIENT.keys())
def test_a_disguised_text_cell_is_still_withheld(disguised):
    firewall = DataFirewall([_patients()])
    screened, counts = firewall.screen_untrusted([f"processing {disguised} now"])
    assert screened == [None] and counts == {"text_cell": 1}


@pytest.mark.parametrize(
    "disguise",
    [_fullwidth, lambda text: _digits(text, 0x0660), lambda text: text.replace(" ", "\u00a0")],
    ids=["fullwidth", "arabic_indic_digits", "no_break_spaces"],
)
def test_a_disguised_printed_row_is_still_blocked(disguise):
    frame = _patients()
    firewall = DataFirewall([frame])
    row = frame.iloc[5]
    printed = disguise(f"{row['patient']} | {row['site']} | {row['dose_mg']:.2f}")
    assert not printed.isascii()
    assert firewall.check_lines([printed]).rule is FirewallRule.ROW_ON_LINE
    assert firewall.check_payload({"logs": [printed]}).blocked


def test_disguised_numbers_still_count(cancer, cancer_firewall):
    """A row of numeric cells in full-width or Arabic-Indic digits is the row."""
    row = " ".join(str(value) for value in cancer.iloc[11].tolist()[:6])
    for disguised in (_fullwidth(row), _digits(row, 0x0660), row.replace(".", "\uff0e")):
        assert cancer_firewall.check_lines([disguised]).rule is FirewallRule.ROW_ON_LINE
        assert cancer_firewall.screen_untrusted([disguised])[0] == [None]


E_DIAERESIS = chr(0x00EB)
COMBINING_DIAERESIS = chr(0x0308)
FULLWIDTH_AB = chr(0xFF21) + chr(0xFF22)


def test_non_ascii_cells_match_as_written_and_invisibly_disguised():
    """A cell is keyed under the light fold: its verbatim, width-variant, decomposed or
    zero-width-split copy matches; its accent-stripped spelling does not (review of PR #48, B2)."""
    frame = pd.DataFrame(
        {
            "code": [f"{FULLWIDTH_AB}-{i:03d}" for i in range(30)],  # full-width "AB-" + digits
            "name": [f"Zo{E_DIAERESIS} Nakamura-{i}" for i in range(30)],
        }
    )
    firewall = DataFirewall([frame])
    for line in (
        f"{FULLWIDTH_AB}-007",
        "ab-007",  # a width variant is the same characters
        f"Zo{E_DIAERESIS} Nakamura-7",
        f"Zoe{COMBINING_DIAERESIS} Nakamura-7",  # decomposed
        f"Zo{chr(0x200B)}{E_DIAERESIS} Nakamura-7",  # zero-width space inside
    ):
        assert firewall.screen_untrusted([f"row {line}"])[0] == [None], line
    assert firewall.screen_untrusted(["row zoe nakamura-7"])[0] == ["row zoe nakamura-7"]


def _rare_cell_frame(cell: str) -> pd.DataFrame:
    """One rare cell among five frequent fillers (frequent values never count)."""
    fillers = [name for name in ("alpha", "bravo", "charlie", "delta", "echo") for _ in range(6)]
    return pd.DataFrame({"value": [*fillers, cell], "other": list(range(31))})


FULLWIDTH_PARENS = (chr(0xFF08), chr(0xFF09))
B1_CELLS = {
    "cjk_fullwidth_parens": chr(0x6771)
    + chr(0x4EAC)
    + FULLWIDTH_PARENS[0]
    + chr(0x672C)
    + chr(0x793E)
    + FULLWIDTH_PARENS[1],
    "cjk_fullwidth_comma": chr(0x5C71) + chr(0x7530) + chr(0xFF0C) + chr(0x592A) + chr(0x90CE),
    "ascii_parens": "acme(ltd)",
    "ascii_comma": "smith, john",
}


@pytest.mark.parametrize("cell", B1_CELLS.values(), ids=B1_CELLS.keys())
def test_a_verbatim_cell_with_delimiters_inside_is_withheld(cell):
    """Review of PR #48, B1: cells are split with the line tokenizer, so a cell holding full-width
    or ASCII brackets or commas matches its verbatim copy (main withheld the CJK ones)."""
    firewall = DataFirewall([_rare_cell_frame(cell)])
    screened, counts = firewall.screen_untrusted([f"row: {cell}"])
    assert screened == [None] and counts == {"text_cell": 1}


ACUTE_E = chr(0x00E9)
B2_CASES = {
    "naive": ("Na" + chr(0x00EF) + "ve", "fitting naive bayes baseline"),
    "region": ("R" + ACUTE_E + "gion", "region feature dropped"),
    "resume": ("R" + ACUTE_E + "sum" + ACUTE_E, "resume training from checkpoint"),
    "media": ("M" + ACUTE_E + "dia", "loading media files"),
    "senior": ("S" + ACUTE_E + "nior", "senior engineer note"),
    "cyrillic_max": (chr(0x043C) + chr(0x0430) + chr(0x0445), "max: 0.93"),
    "cyrillic_top": (chr(0x0442) + chr(0x043E) + chr(0x0440), "top features: a, b"),
    "cyrillic_bet": (chr(0x0412) + chr(0x0415) + chr(0x0422), "bet size"),
    "superscript": ("x" + chr(0x00B2) + "y", "x2y"),
    "numero": (chr(0x2116) + " 17", "no 17"),
}


@pytest.mark.parametrize(("cell", "line"), B2_CASES.values(), ids=B2_CASES.keys())
def test_an_ascii_line_never_matches_a_non_ascii_cell_only_equal_after_folding(cell, line):
    """Review of PR #48, B2: ordinary ASCII lines pass a dataset of non-ASCII cells, while the
    cell itself, written as in the data, is still withheld."""
    firewall = DataFirewall([_rare_cell_frame(cell)])
    assert firewall.screen_untrusted([line]) == ([line], {})
    assert not firewall.check_lines([line]).blocked
    assert firewall.screen_untrusted([f"row {cell} end"])[0] == [None]


def test_a_dirty_variant_of_a_frequent_value_is_not_distinctive():
    """Review of PR #48, B3: a rare `Yes.` or `>50K.` shares the key of the frequent `yes` or
    `>50K`, so lines naming the frequent value pass, as on main."""
    frequent = [name for name in ("yes", "no", "Male", "Female", ">50K") for _ in range(8)]
    dirty = ["Yes.", "(no)", "'Male'", ">50K.", " Female "]
    frame = pd.DataFrame({"answer": [*frequent, *dirty, "Zebra-17", "Quokka-42"]})
    firewall = DataFirewall([frame])
    ordinary = [
        "yes",
        "answer: yes",
        "sex Male coefficient 0.12",
        "class >50K precision 0.71",
        "label <=50K",
        "Female share 0.33",
    ]
    assert firewall.screen_untrusted(ordinary) == (ordinary, {})
    assert not firewall.check_lines(ordinary).blocked
    # Genuinely rare values in the same column are still distinctive.
    assert firewall.screen_untrusted(["row zebra-17"])[0] == [None]


def test_a_number_in_native_digits_counts_once():
    """Review of PR #48: a number seen by both folds (native digits, then ASCII digits) is one
    printed cell, so column thresholds are not halved."""
    fillers = [100.5 + i for i in range(5) for _ in range(3)]  # the five most frequent values
    x = [*np.linspace(10.125, 49.875, 40), *fillers]
    frame = pd.DataFrame({"x": x, "tag": [f"t{i % 2}" for i in range(len(x))]})
    firewall = DataFirewall([frame])
    values = [f"{value:.3f}" for value in frame["x"].iloc[20:22]]
    for line in (" ".join(values), _digits(" ".join(values), 0x0660)):
        assert firewall.screen_untrusted([line]) == ([line], {}), line
    four = _digits(" ".join(f"{value:.3f}" for value in frame["x"].iloc[20:24]), 0x0660)
    assert firewall.screen_untrusted([four])[1] == {"column_on_line": 1}
    assert not firewall.check_lines([four]).blocked


def test_ordinary_lines_pass_a_dataset_of_non_ascii_cells():
    """False-positive check with French, Russian and Greek cells (not only ASCII ones)."""
    cells = [
        "Fran" + chr(0x00E7) + "ois Dupr" + ACUTE_E,
        chr(0x041C) + chr(0x043E) + chr(0x0441) + chr(0x043A) + chr(0x0432) + chr(0x0430),
        chr(0x0391) + chr(0x03B8) + chr(0x03AE) + chr(0x03BD) + chr(0x03B1),
        "Caf" + ACUTE_E + " Cr" + ACUTE_E + "me",
        "S" + chr(0x00E3) + "o Paulo",
    ]
    fillers = [name for name in ("alpha", "bravo", "charlie", "delta", "echo") for _ in range(6)]
    frame = pd.DataFrame({"city": [*fillers, *cells]})
    firewall = DataFirewall([frame])
    for cell in cells:  # each cell is distinctive: its verbatim copy is withheld
        assert firewall.screen_untrusted([f"row {cell}"])[0] == [None], cell
    ordinary = [
        "fitting naive bayes baseline",
        "francois dupre is not a column; mockba moscow athena cafe creme sao paulo",
        "top features: city, n",
        *training_log(seed=1, n=50),
    ]
    assert firewall.screen_untrusted(ordinary) == (ordinary, {})


def test_ordinary_non_ascii_log_lines_still_pass(cancer_firewall):
    """No new false positive: non-ASCII decoration, other languages and units pass."""
    ordinary = [
        "100%|" + "\u2588" * 10 + "| 5/5 [00:02<00:00, 2.13it/s]",
        "\u00c9poque 3 termin\u00e9e \u2014 perte 0,4312 \u2713",
        "temperature \u22123.5 \u00b0C, \u0394loss = 0.0123",
        "\u8a13\u7df4\u5b8c\u4e86: \u7cbe\u5ea6 0.912",  # Japanese: training done, accuracy
        "fold\u00a02\u2009done in 3.2\u202fs",  # odd spaces
    ]
    screened, counts = cancer_firewall.screen_untrusted(ordinary)
    assert screened == ordinary and counts == {}
    patients = DataFirewall([_patients()])
    assert patients.screen_untrusted(ordinary) == (ordinary, {})


def test_values_after_contract_fixes_are_recognized_when_that_frame_is_indexed():
    """Trial code sees the contract-applied frame (e.g. after F -> C): index it too."""
    rng = np.random.default_rng(4)
    raw = pd.DataFrame(
        {
            "temp_f": rng.uniform(95, 105, 40),
            "pressure_f": rng.uniform(150, 450, 40),
            "flow_f": rng.uniform(60, 90, 40),
        }
    )
    converted = (raw - 32) * 5 / 9
    printed = converted.iloc[10:13].to_string()  # what trial code would print after the fix
    assert not DataFirewall([raw]).check_lines(lines(printed)).blocked
    assert DataFirewall([raw, converted]).check_lines(lines(printed)).blocked


def test_text_beyond_the_scan_limit_fails_closed():
    small = DataFirewall(
        [pd.DataFrame({"a": [1.2345, 2.3456]})], FirewallConfig(max_scan_chars=100)
    )
    verdict = small.check_lines(["x" * 60, "y" * 60])
    assert verdict.blocked and verdict.rule is FirewallRule.SCAN_LIMIT


# --- summaries and ordinary output pass ----------------------------------------------------------


@pytest.mark.parametrize("loader", [load_breast_cancer, load_wine, load_iris])
def test_the_datasets_own_profile_passes(loader):
    """Profiles legitimately disclose frequent labels and ranges (D-009): never a false block."""
    frame = loader(as_frame=True).frame
    task = (
        TaskType.MULTICLASS_CLASSIFICATION
        if frame["target"].nunique() > 2
        else TaskType.BINARY_CLASSIFICATION
    )
    profile = profile_tabular(
        frame, run_id="run_fw", dataset_ref="dataset:x", target="target", task_type=task
    )
    verdict = DataFirewall([frame]).check_payload(profile.model_dump(mode="json"))
    assert not verdict.blocked


def test_describe_output_passes(cancer, cancer_firewall):
    assert not cancer_firewall.check_lines(lines(str(cancer.describe()))).blocked


def training_log(seed: int, n: int = 300) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b, c = rng.uniform(0, 1, 3)
        out.append(f"[{i}] loss {a:.4f} val_loss {b:.4f} acc {c:.3f} lr 0.0123 depth 5")
        out.append(
            f"Trial {i} finished with value: {rng.uniform(0.8, 1):.6f} and parameters: "
            f"{{'lr': {rng.uniform(0, 0.3):.5f}, 'num_leaves': {rng.integers(8, 256)}}}"
        )
    return out


def test_ordinary_training_logs_are_never_blocked_at_the_boundary(cancer_firewall):
    assert not cancer_firewall.check_lines(training_log(seed=0)).blocked


def test_the_stricter_untrusted_screen_rarely_withholds_ordinary_log_lines(cancer_firewall):
    """Measured false-withhold rate on 600 synthetic training-log lines (docs/notes)."""
    screened, _ = cancer_firewall.screen_untrusted(training_log(seed=0))
    withheld = sum(line is None for line in screened) / len(screened)
    assert withheld < 0.05
    wine_screened, _ = DataFirewall([load_wine(as_frame=True).frame]).screen_untrusted(
        training_log(seed=0)
    )
    assert all(line is not None for line in wine_screened)


def test_untrusted_screen_withholds_only_the_lines_that_show_data(cancer, cancer_firewall):
    text = lines(str(cancer.head(3)))
    screened, counts = cancer_firewall.screen_untrusted(
        ["Traceback (most recent call last):", *text, "ValueError: shapes do not match"]
    )
    assert screened[0] == "Traceback (most recent call last):"
    assert screened[-1] == "ValueError: shapes do not match"
    assert screened[1] == text[0]  # the header shows column names, not data
    assert all(line is None for line in screened[2:5])  # the three data rows
    assert counts.get("row_on_line", 0) >= 3


def test_frequent_values_and_column_extremes_never_count(cancer):
    """A line made of the column minimum, maximum, and most frequent value shows no row."""
    column = cancer["mean radius"]
    frequent = column.value_counts().index[0]
    firewall = DataFirewall([cancer])
    line = f"{column.min()} {column.max()} {frequent}"
    assert not firewall.check_lines([line]).blocked
    assert firewall.screen_untrusted([line])[0] == [line]


def test_rows_of_common_values_are_a_documented_gap():
    """Known limitation (docs/notes/mcp-server.md): iris cells are short and frequent, so a
    printed iris row carries no distinctive value and is not recognized."""
    iris = load_iris(as_frame=True).frame
    assert not DataFirewall([iris]).check_lines(lines(str(iris.head()))).blocked


def test_the_firewall_never_reports_matched_values(cancer, cancer_firewall):
    verdict = cancer_firewall.check_lines(lines(str(cancer.head())))
    assert all(isinstance(value, int) for value in verdict.details.values())


def test_invalid_thresholds_are_refused():
    with pytest.raises(ValueError):
        FirewallConfig(row_cells=0)
