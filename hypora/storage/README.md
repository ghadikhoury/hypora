# Storage (S0.3)

Implements spec §15.1, §15.5, and §16. No controller, queue executor, approval enforcement,
configuration loader, or evaluation/grading logic is introduced here.

Run from the repository root in WSL/Linux:

```bash
uv run --locked python -m hypora.storage.migrate hypora.sqlite
make check
```

`migrations/0001` is a committed, standalone schema snapshot. It creates all 15 tables from
§15.1, constraints, indexes, and SQLite append-only triggers for `events` and `budget_ledger`.
`eval_cases` and `eval_results` are storage records only; S2 defines their content. No held-out
data or grading rules are implemented or modified. The migration tests upgrade an empty file,
compare the resulting schema with SQLAlchemy metadata, then downgrade and re-upgrade.

## Connections and transactions

`Database(Path(...))` uses SQLite WAL, FULL synchronous writes, foreign-key checks, and a 30-second
busy timeout. Connections are not pooled or shared across processes. Each connection disables
sqlite3's legacy implicit transaction management; SQLAlchemy's begin hook explicitly starts a
transaction, including for ORM sessions. `Database.write()` starts `BEGIN IMMEDIATE` before any
read, so sequence allocation and budget updates are serialized by SQLite across threads and
processes. Readers retain ordinary deferred transactions and can continue reading during a write.
The SQLAlchemy schema uses portable SQL types and non-native enums; this step's runtime is SQLite.

All timestamp columns use `UTCDateTime`: naive inputs fail, aware values normalize to UTC, and
SQLite reads restore the UTC timezone. Engine JSON serialization refuses NaN/Infinity.

## Event log and replay

```python
from pathlib import Path

from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog

database = Database(Path("hypora.sqlite"))
log = EventLog(database)
# The controller must first create the run record.
event = log.append("run_1", "error", "controller", {
    "component": "example", "code": "EXAMPLE", "message": "synthetic observation",
})
tail = log.read("run_1", EventFilters(after_seq=event.seq))
state = log.replay("run_1")
```

`append` validates the complete envelope and JSON payload through S0.2's `parse_event` union before
writing. ORM inserts validate through the same union. Reads validate the stored envelope again.
The database has a composite primary key on `(run_id, seq)`; allocation uses `MAX(seq) + 1` while
holding the write reservation, so failed writes do not consume numbers. Reads support event type,
role, trial/job/trace IDs, UTC time bounds, and sequence bounds. Correlation IDs persist separately
from payloads. Raw SQL update/delete of events and ledger entries fails with an append-only error.
Application code must insert events through `EventLog` or the validated ORM model, rather than raw
SQL inserts, which cannot execute Pydantic validation inside SQLite.

`reduce_event(state, event) -> state` is pure. `replay(state, events)` folds this reducer over an
ordered sequence; `EventLog.replay(run_id)` only fetches events before invoking the fold. The
`REDUCERS` table explicitly covers every approved event type. The minimal projection applies
stage/status, job state, and approval observations; other event types advance the sequence without
projecting their payload. S3.E3 extends the handlers/projection, rather than replacing the fold.
Replay rejects mixed runs, missing events, and out-of-order events. It does not perform side effects,
authorize transitions, or update the database's run projection.

## Budgets

`BudgetLedger.charge(run_id, kind, amount, unit, ref)` atomically inserts an entry and updates the
run total. Units are `usd` and `compute_s`. Decimal conversion rejects negative/nonfinite amounts
and values outside NUMERIC(20, 8)'s range/precision. A retry with the same `(run, kind, unit, ref)`
and amount returns the original sequence; a changed amount fails. `remaining(run_id)` returns
`RemainingBudget(usd, compute_s)`, where `None` means unlimited. Actual overruns are recorded and
produce negative remaining balances: the controller enforces authorization before spending; this
ledger must still record what actually happened. Budget entries are not new event types.

## Artifacts and interruption boundaries

`ArtifactStore(database, root).write(artifact_id, trial_id, kind, chunks)` consumes byte chunks into
a hidden temporary file in the destination directory, flushes and fsyncs it, then publishes it with
`os.replace`. POSIX also fsyncs the directory after replacement. Only after publication is the
artifact row committed with its SHA-256. Artifact IDs name immutable versions; retrying the same
content/metadata is idempotent and changing content requires a new ID. `get` uses only committed
references and verifies the file hash.

A filesystem rename and a SQLite commit cannot be one atomic transaction. Interruption before
rename leaves no final file or artifact reference; abrupt process death can leave an abandoned
hidden temp file. Interruption after rename and before DB commit can leave a complete orphan.
Retrying the same ID/content verifies and registers that orphan. An ordinary source exception
removes its temporary file. Directory listings are never the artifact catalog, and unrelated
committed artifacts are unaffected by a killed writer. This step does not implement general orphan
garbage collection or training checkpoint recovery (S3.E3).

## Acceptance evidence

`hypora/storage/tests/` is collected by `make check` (added to pytest's `testpaths`). Tests cover all
S0.2 event fixtures, invalid writes, UTC normalization, foreign keys, rollback, schema/migration
parity, immutable log rows, pure/incremental replay, exact budget assertions, retry deduplication,
six real thread writers, four spawned process writers, and a spawned artifact writer killed after
writing partial bytes. Artifact tests also exercise fsync-before-replace and the complete-orphan
boundary. Tests use synthetic data and temporary paths only.

References for transaction control and schema parity:
[SQLAlchemy SQLite transactions](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#transactions-with-sqlite-and-the-sqlite3-driver),
[Alembic metadata comparison](https://alembic.sqlalchemy.org/en/latest/api/autogenerate.html#alembic.autogenerate.compare_metadata).
