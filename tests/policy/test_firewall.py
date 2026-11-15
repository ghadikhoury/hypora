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


def test_non_ascii_cells_match_both_as_written_and_disguised():
    """The index is folded like the lines, so a full-width or accented cell matches itself and
    its plain spelling."""
    frame = pd.DataFrame(
        {
            "code": [f"\uff21\uff22-{i:03d}" for i in range(30)],  # full-width "AB-" + digits
            "name": [f"Zo\u00eb Nakamura-{i}" for i in range(30)],
        }
    )
    firewall = DataFirewall([frame])
    for line in ("\uff21\uff22-007", "ab-007", "Zo\u00eb Nakamura-7", "zoe nakamura-7"):
        assert firewall.screen_untrusted([f"row {line}"])[0] == [None], line


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
