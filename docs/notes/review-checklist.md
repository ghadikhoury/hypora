# Review checklist

Used by authors before opening a PR and by the Reviewer (D-029). These are the kinds of blocking bugs
that review has already caught in this repo. It adds to the step's "Done when" items and spec sections
and does not replace them. Add a line when review catches a new kind of bug.

- **Concurrency and retries:** duplicate, in-flight, or racing calls can't both succeed or replay stale
  state, and a retry with the same key returns the same result (PR #15).
- **Same check on every route:** a guard covers every way in, such as env vars, bulk JSON, case variants,
  redirects, and intermediate hops (PRs #3, #17).
- **Check what you use:** data verified at one point is the data used later, and it can't change in
  between (PR #15).
- **Append-only stays append-only:** no statement can rewrite protected records, including
  `INSERT OR REPLACE` (PR #4).
- **Limits and accounting:** every billed attempt is counted, including retries and failures. Rate
  limits apply at dispatch, and shared limits are shared across runs (PRs #6, #16).
- **Thresholds can be reached:** a rule can fire on the metric's real range and doesn't flag ordinary
  values (PR #7).
- **Real input types:** nullable, categorical, datetime, unorderable, and very long values all work, and
  a value's output doesn't depend on the other rows in its batch (PRs #7, #14, #15).
- **Evidence is real:** verification can't pass with empty or partial evidence, and a limit or failure
  is reported, not hidden (PRs #2, #16).
- **Plan's frozen items:** no unapproved change to a schema, MCP tool, trial interface, or event type
  (triggers 1 and 4).
- **Project rules:** every model call, tool call, state transition, and policy decision emits an event;
  new compared behavior has a feature flag; evaluation and demos use real data.
- **The fix is complete:** sibling code paths have the same fix, and a test fails without it.

- **Hidden metric parity:** systems report every supported task metric without reading the hidden
  grading rule; an undefined required metric fails explicitly rather than silently losing evidence
  (PR #27).
