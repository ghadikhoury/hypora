# C-only workers and dependency-aware scheduling (S3.4)

Spec sections 6.4-6.8 and 9; owner approval D-042.

C makes one structured WorkerProposal before its implementation loop. Empty proposals are valid.
The runtime validates the role, exact context references, tool subset, sensitive text and aggregate
capacity before constructing immutable Brief identities. Only trial workers ship. There is no
worker-spawning MCP tool, even for C; A, B and workers cannot delegate.

Each worker has its own directory, conversation, session ID and MCP process. A trusted brief_id
stamps trials.batch_id on creation and cannot be changed by later edits. Every trial/job lookup
checks ownership; idempotency request hashes include worker scope so siblings cannot recover each
other's creation results. C retains access to the entire run. Worker permissions remain confined
when the optional allowlist defense is disabled.

WorkerReservations holds token, step and trial envelopes atomically, leaving a fixed lead reserve.
WorkerGuard charges actuals to RunGuard once and releases only unused reservations. Worker caps
do not enter run limits_fired. Model attempts reserve a conservative text request envelope and
bounded output before dispatch, including retries/fallbacks; exceeding actual caps blocks success.
The adapter's existing run-level USD/compute and wall-clock controls remain authoritative.

Workers receive only their brief and requested records, never A/B/C transcripts. Findings contain
code-populated trial/job/outcome/result references; text does not supply measured values. Failure,
timeout and cancellation cancel only the worker's jobs, await bounded acknowledgment, close the
MCP session and release the reservation. The controller's phase shutdown remains the backstop.

The ready scheduler is a static in-memory dependency map, with bounded execution and skipped
dependents after failures. It does not persist or replay workflows. By default B finishes before
C drafts methodology. A may justify independent target-free B/C overlap; the runtime waits for
both before A reviews, locks diagnostic rows and reconciles later findings. Worker concurrency
is separate from the unchanged experiment worker/resource-pool concurrency.
