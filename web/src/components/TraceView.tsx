import React, { useEffect, useMemo, useState } from "react";
import { getRunEvents, subscribeRunEvents } from "../api";
import type { RunListItem, TraceEventItem } from "../types";
import { EventDrawer } from "./EventDrawer";
import { StatusBadge } from "./StatusBadge";

interface TraceViewProps {
  runs: RunListItem[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
}

const ROLES = [
  "agent_a",
  "agent_b",
  "agent_c",
  "trial_worker",
  "controller",
  "policy_engine",
  "verifier",
] as const;

const TERMINAL_STATUSES = new Set([
  "DONE",
  "FAILED",
  "CANCELLED",
  "FAILED_TERMINAL",
]);

function formatOffset(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

export const TraceView: React.FC<TraceViewProps> = ({
  runs,
  selectedRunId,
  onSelectRun,
}) => {
  const [events, setEvents] = useState<TraceEventItem[]>([]);
  const [summary, setSummary] = useState<string>("");
  const [runStatus, setRunStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sseLive, setSseLive] = useState(false);

  const [selectedRoles, setSelectedRoles] = useState<Set<string>>(new Set());
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [trialJobFilter, setTrialJobFilter] = useState<string>("");
  const [search, setSearch] = useState<string>("");
  const [inspectedEvent, setInspectedEvent] = useState<TraceEventItem | null>(
    null
  );

  const selectedRunMeta = useMemo(
    () => runs.find((r) => r.run_id === selectedRunId) ?? null,
    [runs, selectedRunId]
  );

  useEffect(() => {
    if (!selectedRunId) {
      setEvents([]);
      setSummary("");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getRunEvents(selectedRunId, { limit: 2000 })
      .then((res) => {
        if (cancelled) return;
        setEvents(res.events);
        setSummary(res.summary);
        setRunStatus(res.status);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load trace events."
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || !runStatus) return;
    if (TERMINAL_STATUSES.has(runStatus) || selectedRunMeta?.read_only) {
      setSseLive(false);
      return;
    }
    const lastSeq = events.length > 0 ? events[events.length - 1].seq : 0;
    setSseLive(true);
    const unsub = subscribeRunEvents(selectedRunId, lastSeq, {
      onTraceEvent: (ev) => {
        setEvents((prev) => {
          if (prev.some((p) => p.seq === ev.seq)) return prev;
          return [...prev, ev];
        });
      },
      onTerminal: (info) => {
        setRunStatus(info.status);
        setSseLive(false);
      },
      onError: () => {
        setSseLive(false);
      },
    });
    return () => {
      unsub();
      setSseLive(false);
    };
  }, [selectedRunId, runStatus, selectedRunMeta?.read_only]);

  const eventTypes = useMemo(() => {
    const set = new Set<string>();
    for (const ev of events) set.add(ev.type);
    return Array.from(set).sort();
  }, [events]);

  const filteredEvents = useMemo(() => {
    const tj = trialJobFilter.trim().toLowerCase();
    const q = search.trim().toLowerCase();
    return events.filter((ev) => {
      if (selectedRoles.size > 0 && !selectedRoles.has(ev.role)) return false;
      if (typeFilter && ev.type !== typeFilter) return false;
      if (tj) {
        const matchTrial = (ev.trial_id ?? "").toLowerCase().includes(tj);
        const matchJob = (ev.job_id ?? "").toLowerCase().includes(tj);
        if (!matchTrial && !matchJob) return false;
      }
      if (q) {
        const hay =
          `${ev.role} ${ev.type} ${ev.detail} ${ev.trial_id ?? ""} ${ev.job_id ?? ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [events, selectedRoles, typeFilter, trialJobFilter, search]);

  const toggleRole = (role: string) => {
    setSelectedRoles((prev) => {
      const next = new Set(prev);
      if (next.has(role)) next.delete(role);
      else next.add(role);
      return next;
    });
  };

  return (
    <div className="section-stack">
      {/* Run Selector & Stream Status */}
      <div className="card">
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 320 }}>
            <label
              htmlFor="trace-run-select"
              className="field-label"
              style={{ whiteSpace: "nowrap" }}
            >
              Run
            </label>
            <select
              id="trace-run-select"
              className="select mono"
              value={selectedRunId ?? ""}
              onChange={(e) => onSelectRun(e.target.value)}
            >
              {runs.length === 0 && <option value="">No runs available</option>}
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — {r.dataset_label} ({r.status}
                  {r.read_only ? ", read_only" : ""})
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {runStatus && <StatusBadge value={runStatus} />}
            {selectedRunMeta?.read_only && (
              <span className="badge">read_only</span>
            )}
            {sseLive ? (
              <span className="badge badge-cobalt">SSE STREAMING</span>
            ) : (
              <span className="badge">SNAPSHOT</span>
            )}
          </div>
        </div>

        {summary && (
          <div
            className="mono"
            style={{
              marginTop: 10,
              fontSize: 11,
              color: "var(--text-muted)",
              padding: "6px 10px",
              background: "var(--bg-canvas)",
              border: "1px solid var(--border-hairline)",
              borderRadius: 4,
            }}
          >
            {summary}
          </div>
        )}
      </div>

      {/* Filter Bar */}
      <div className="card">
        <div className="section-stack" style={{ gap: 10 }}>
          <div className="chip-row">
            <span className="field-label" style={{ marginRight: 4 }}>
              Roles:
            </span>
            {ROLES.map((r) => (
              <button
                key={r}
                type="button"
                className={`chip ${selectedRoles.has(r) ? "active" : ""}`}
                onClick={() => toggleRole(r)}
              >
                {r}
              </button>
            ))}
            {selectedRoles.size > 0 && (
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setSelectedRoles(new Set())}
              >
                Clear roles
              </button>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "200px 200px minmax(0, 1fr)",
              gap: 10,
            }}
          >
            <select
              className="select mono"
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
            >
              <option value="">All event types ({eventTypes.length})</option>
              {eventTypes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            <input
              className="input mono"
              placeholder="Filter trial_id / job_id..."
              value={trialJobFilter}
              onChange={(e) => setTrialJobFilter(e.target.value)}
            />

            <input
              className="input"
              placeholder="Instant search detail, role, type..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Timeline Table */}
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div
          style={{
            padding: "10px 14px",
            borderBottom: "1px solid var(--border-hairline)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span className="card-title">Event Timeline</span>
          <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
            showing {filteredEvents.length} / {events.length} events
          </span>
        </div>

        {loading ? (
          <div className="empty-state">Loading trace events...</div>
        ) : filteredEvents.length === 0 ? (
          <div className="empty-state">No trace events match the active filters.</div>
        ) : (
          <div className="table-wrap" style={{ maxHeight: "calc(100vh - 340px)" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>SEQ</th>
                  <th>+TIME</th>
                  <th>ROLE</th>
                  <th>TYPE</th>
                  <th style={{ textAlign: "right" }}>DURATION</th>
                  <th style={{ textAlign: "right" }}>TOKENS</th>
                  <th>DETAIL</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.map((ev) => {
                  const isSel = inspectedEvent?.seq === ev.seq;
                  const tok =
                    ev.tokens_in !== null || ev.tokens_out !== null
                      ? `${(ev.tokens_in ?? 0).toLocaleString()}/${(ev.tokens_out ?? 0).toLocaleString()}`
                      : "—";
                  return (
                    <tr
                      key={ev.seq}
                      className={`clickable ${isSel ? "selected-row" : ""}`}
                      onClick={() => setInspectedEvent(ev)}
                    >
                      <td className="mono">{ev.seq}</td>
                      <td className="mono">{formatOffset(ev.offset_s)}</td>
                      <td className="mono">{ev.role}</td>
                      <td>
                        <StatusBadge value={ev.type} />
                      </td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {ev.duration_s !== null
                          ? `${ev.duration_s.toFixed(2)}s`
                          : "—"}
                      </td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {tok}
                      </td>
                      <td className="mono" style={{ fontSize: 11 }}>
                        {ev.detail}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <EventDrawer
        event={inspectedEvent}
        onClose={() => setInspectedEvent(null)}
      />
    </div>
  );
};