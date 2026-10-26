"""`hypora trace`: rows, durations, filters, rendering, and the command (PLAN.md S1.6)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypora.cli.app import app
from hypora.cli.trace import (
    describe,
    format_duration,
    format_offset,
    parse_filters,
    render_rows,
    summary,
    trace_rows,
)
from hypora.core.schemas import parse_event
from hypora.storage.database import Database
from tests.cli.conftest import FIXTURES, RUN_ID, SECRET_ARGUMENT, seed_timeline

runner = CliRunner()


@pytest.mark.parametrize("fixture", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_every_event_type_has_a_one_line_detail(fixture: Path) -> None:
    event = parse_event(json.loads(fixture.read_text()))
    detail = describe(event)
    assert detail and "\n" not in detail


def test_tool_arguments_are_named_never_shown(database: Database) -> None:
    rows = trace_rows(seed_timeline(database))
    call = next(row for row in rows if row.type == "tool_call")
    assert call.detail == "write_trial_files(design, pipeline_py)"
    assert all(SECRET_ARGUMENT not in row.detail for row in rows)


def test_durations_follow_the_documented_rules(database: Database) -> None:
    rows = {row.seq: row for row in trace_rows(seed_timeline(database))}
    assert rows[3].duration_s == pytest.approx(0.9)  # model call: its latency
    assert rows[3].tokens == (500, 50)
    assert rows[4].duration_s == pytest.approx(1.0)  # handoff: since its role_started
    assert rows[1].duration_s is None and rows[5].duration_s == pytest.approx(1.2)  # stages
    assert rows[7].duration_s == pytest.approx(0.12)  # tool result: its latency
    assert rows[8].duration_s is None  # the job's first transition in this log
    assert rows[9].duration_s == pytest.approx(0.5)
    assert rows[10].duration_s == pytest.approx(7.0)  # time spent RUNNING
    assert rows[13].duration_s == pytest.approx(11.8)  # time spent in PROFILE
    assert rows[13].offset_s == pytest.approx(13.0)
    assert "rate-limit wait 2.5s" in rows[3].detail and "retries=2" in rows[3].detail
    assert rows[11].detail.startswith("submit_experiment -> error INSUFFICIENT_GPU_MEMORY")


def test_filters_select_rows_but_durations_use_the_whole_log(database: Database) -> None:
    rows = trace_rows(seed_timeline(database))
    handoffs = [row for row in rows if parse_filters(["type=handoff"]).keeps(row)]
    assert [row.seq for row in handoffs] == [4]
    assert handoffs[0].duration_s == pytest.approx(1.0)  # its role_started is filtered out
    jobs = parse_filters(["type=job_transition,error", "role=execution_worker", "job=job_1"])
    assert [row.seq for row in rows if jobs.keeps(row)] == [8, 9, 10]
    trial = parse_filters(["trial=trl_1"])
    assert [row.seq for row in rows if trial.keeps(row)] == [6, 7, 8, 9, 10]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("type=nope", "unknown type"),
        ("role=wizard", "unknown role"),
        ("colour=red", "key one of"),
        ("type", "key one of"),
        ("type=", "key one of"),
    ],
)
def test_bad_filters_name_the_valid_choices(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_filters([spec])


def test_formatting_helpers() -> None:
    assert [format_duration(s) for s in (None, 0.4, 12.34, 185, 3725)] == [
        "-",
        "0.40s",
        "12.3s",
        "3m05s",
        "1h02m",
    ]
    assert [format_offset(s) for s in (0, 3.25, 754.0, 3725)] == [
        "0:00.0",
        "0:03.2",
        "12:34.0",
        "1:02:05",
    ]


def test_rendering_aligns_columns_and_cuts_only_the_detail(database: Database) -> None:
    rows = trace_rows(seed_timeline(database))
    lines = render_rows(rows)
    assert lines[0].split() == [
        "SEQ",
        "+TIME",
        "ROLE",
        "TYPE",
        "DURATION",
        "TOKENS",
        "IN/OUT",
        "DETAIL",
    ]
    detail_column = lines[0].index("DETAIL")
    assert all(len(line) >= detail_column for line in lines[1:])
    narrow = render_rows(rows, width=detail_column + 20)
    assert all(len(line) <= detail_column + 20 for line in narrow)
    assert any(line.endswith("...") for line in narrow)


def test_summary_totals_cover_the_whole_run(database: Database) -> None:
    events = seed_timeline(database)
    rows = trace_rows(events)
    text = summary(rows[:2], events)
    assert text.startswith("13 events (2 shown)")
    assert "1 model calls (500 in / 50 out tokens, $0.0003 at list price)" in text
    assert "2 tool results (1 failed)" in text and "1 error events" in text
    assert text.endswith("span 13.0s")


def test_trace_command(database: Database, tmp_path: Path) -> None:
    seed_timeline(database)
    db = str(tmp_path / "hypora.sqlite")
    result = runner.invoke(app, ["trace", RUN_ID, "--db", db])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"Run {RUN_ID} (INTAKE) | 2026-10-25 12:00:00 to 12:00:13 UTC"
    assert len([line for line in lines if "job_transition" in line]) == 3
    assert lines[-1].startswith("13 events |")
    assert SECRET_ARGUMENT not in result.output

    filtered = runner.invoke(app, ["trace", RUN_ID, "--db", db, "-f", "type=handoff"])
    assert filtered.exit_code == 0
    assert "13 events (1 shown)" in filtered.output


def test_trace_command_errors(database: Database, tmp_path: Path) -> None:
    db = str(tmp_path / "hypora.sqlite")
    seed_timeline(database)
    bad = runner.invoke(app, ["trace", RUN_ID, "--db", db, "-f", "type=nope"])
    assert bad.exit_code == 2 and "unknown type" in bad.output
    unknown = runner.invoke(app, ["trace", "run_nope", "--db", db])
    assert unknown.exit_code == 1 and "no run run_nope" in unknown.output
    missing = runner.invoke(app, ["trace", RUN_ID, "--db", str(tmp_path / "none.sqlite")])
    assert missing.exit_code == 1 and "no state database" in missing.output
    assert not (tmp_path / "none.sqlite").exists()  # a read-only command creates nothing
