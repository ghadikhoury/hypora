import React, { useEffect, useMemo, useState } from "react";
import {
  cancelRun,
  getRunDetail,
  getRunEvents,
  subscribeRunEvents,
} from "../api";
import type {
  RunDetailResponse,
  RunListItem,
  TraceEventItem,
  TrialMetricSplit,
} from "../types";
import { ApprovalGate } from "./ApprovalGate";
import { StatusBadge } from "./StatusBadge";

interface RunsViewProps {
  runs: RunListItem[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  onRefreshRuns: () => void;
  onOpenTrace: (runId: string) => void;
}

const PIPELINE_STAGES = [
  "INTAKE",
  "PROFILE",
  "CONTRACT",
  "PHASE_B",
  "REPORT",
] as const;

const TERMINAL_STATUSES = new Set([
  "DONE",
  "FAILED",
  "CANCELLED",
  "FAILED_TERMINAL",
]);

const ROLE_CHIPS: { key: string; label: string }[] = [
  { key: "agent_a", label: "Agent A" },
  { key: "agent_b", label: "Agent B" },
  { key: "agent_c", label: "Agent C" },
  { key: "trial_worker", label: "Worker" },
  { key: "verifier", label: "Verifier" },
];

function stageIndexFromStatus(status: string, stage: string | null): number {
  const s = (stage ?? status).toUpperCase();
  if (s === "DONE") return 5;
  if (s === "REPORT" || s === "REPORTING") return 4;
  if (s === "PHASE_B") return 3;
  if (s === "CONTRACT" || s === "PHASE_A" || s === "WAITING_APPROVAL") return 2;
  if (s === "PROFILE") return 1;
  return 0;
}

function formatMetricSplit(split: TrialMetricSplit | null): string {
  if (!split) return "—";
  const entries = Object.entries(split.metrics).filter(
    ([, v]) => typeof v === "number"
  );
  if (entries.length === 0) return "—";
  return entries
    .slice(0, 2)
    .map(([k, v]) => `${k}: ${Number(v).toFixed(4)}`)
    .join(" · ");
}

function isSplitVerified(split: TrialMetricSplit | null): boolean | null {
  if (!split) return null;
  const v = split.verification;
  if (typeof v.passed === "boolean") return v.passed;
  if (typeof v.verified === "boolean") return v.verified;
  if (typeof v.status === "string") {
    return v.status.toUpperCase() === "PASSED" || v.status.toUpperCase() === "OK";
  }
  return null;
}

export const RunsView: React.FC<RunsViewProps> = ({
  runs,
  selectedRunId,
  onSelectRun,
  onRefreshRuns,
  onOpenTrace,
}) => {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [showReadOnly, setShowReadOnly] = useState<boolean>(true);

  const [detail, setDetail] = useState<RunDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [liveRoles, setLiveRoles] = useState<Set<string>>(new Set());
  const [liveStage, setLiveStage] = useState<string | null>(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const filteredRuns = useMemo(() => {
    const q = search.trim().toLowerCase();
    return runs.filter((r) => {
      if (!showReadOnly && r.read_only) return false;
      if (statusFilter === "waiting" && !r.pending_approval && r.status !== "WAITING_APPROVAL") {
        return false;
      }
      if (statusFilter === "active" && TERMINAL_STATUSES.has(r.status)) {
        return false;
      }
      if (statusFilter === "done" && r.status !== "DONE") return false;
      if (statusFilter === "failed" && r.status !== "FAILED") return false;
      if (!q) return true;
      return (
        r.run_id.toLowerCase().includes(q) ||
        r.dataset_label.toLowerCase().includes(q) ||
        (r.goal ?? "").toLowerCase().includes(q)
      );
    });
  }, [runs, search, statusFilter, showReadOnly]);

  const loadDetail = async (runId: string) => {
    setLoadingDetail(true);
    setDetailError(null);
    try {
      const [d, evRes] = await Promise.all([
        getRunDetail(runId),
        getRunEvents(runId, { limit: 500 }),
      ]);
      setDetail(d);
      setLiveStage(d.stage);
      const roles = new Set<string>();
      for (const ev of evRes.events) {
        roles.add(ev.role);
      }
      setLiveRoles(roles);
    } catch (err) {
      setDetailError(
        err instanceof Error ? err.message : "Failed to load run details."
      );
    } finally {
      setLoadingDetail(false);
    }
  };

  useEffect(() => {
    if (!selectedRunId) {
      setDetail(null);
      return;
    }
    loadDetail(selectedRunId);
  }, [selectedRunId]);

  // Live SSE updates for non-terminal runs
  useEffect(() => {
    if (!selectedRunId || !detail) return;
    if (TERMINAL_STATUSES.has(detail.status) || detail.read_only) {
      setSseConnected(false);
      return;
    }

    setSseConnected(true);
    const unsub = subscribeRunEvents(selectedRunId, 0, {
      onTraceEvent: (ev: TraceEventItem) => {
        setLiveRoles((prev) => {
          if (prev.has(ev.role)) return prev;
          const next = new Set(prev);
          next.add(ev.role);
          return next;
        });
        const stageFromPayload =
          typeof ev.payload?.stage === "string"
            ? ev.payload.stage
            : typeof ev.payload?.to_stage === "string"
              ? ev.payload.to_stage
              : null;
        if (stageFromPayload) {
          setLiveStage(stageFromPayload);
        }
        if (
          ev.type === "state_transition" ||
          ev.type === "approval_requested" ||
          ev.type === "approval_resolved" ||
          ev.type === "trial_finished"
        ) {
          getRunDetail(selectedRunId)
            .then((fresh) => {
              setDetail(fresh);
              setLiveStage(fresh.stage);
              onRefreshRuns();
            })
            .catch(() => {});
        }
      },
      onTerminal: () => {
        setSseConnected(false);
        getRunDetail(selectedRunId)
          .then((fresh) => {
            setDetail(fresh);
            setLiveStage(fresh.stage);
            onRefreshRuns();
          })
          .catch(() => {});
      },
      onError: () => {
        setSseConnected(false);
      },
    });

    return () => {
      unsub();
      setSseConnected(false);
    };
  }, [selectedRunId, detail?.status, detail?.read_only]);

  const handleCancel = async () => {
    if (!detail) return;
    setCancelling(true);
    try {
      await cancelRun(detail.run_id);
      await loadDetail(detail.run_id);
      onRefreshRuns();
    } catch (err) {
      setDetailError(
        err instanceof Error ? err.message : "Failed to cancel run."
      );
    } finally {
      setCancelling(false);
    }
  };

  const activeStageIdx = detail
    ? stageIndexFromStatus(detail.status, liveStage ?? detail.stage)
    : 0;

  return (
    <div className="runs-layout">
      {/* Master Run List */}
      <div className="runs-list-panel">
        <div className="runs-list-controls">
          <input
            className="input"
            placeholder="Filter by run_id, dataset, goal..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="chip-row">
            {(["all", "active", "waiting", "done", "failed"] as const).map(
              (f) => (
                <button
                  key={f}
                  type="button"
                  className={`chip ${statusFilter === f ? "active" : ""}`}
                  onClick={() => setStatusFilter(f)}
                >
                  {f}
                </button>
              )
            )}
          </div>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 11,
              color: "var(--text-muted)",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={showReadOnly}
              onChange={(e) => setShowReadOnly(e.target.checked)}
            />
            Include historical (read_only) runs
          </label>
        </div>

        <div className="runs-list-scroll">
          {filteredRuns.length === 0 ? (
            <div className="empty-state">No matching runs found.</div>
          ) : (
            filteredRuns.map((r) => (
              <div
                key={r.run_id}
                className={`run-row ${selectedRunId === r.run_id ? "selected" : ""}`}
                onClick={() => onSelectRun(r.run_id)}
              >
                <div className="run-row-top">
                  <span className="mono" style={{ fontWeight: 600, fontSize: 12 }}>
                    {r.run_id}
                  </span>
                  <StatusBadge value={r.status} />
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: "var(--text-ink)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {r.dataset_label}
                </div>
                <div className="run-row-meta">
                  <span className="mono">
                    {r.stage ?? "—"} · ${r.spent_usd.toFixed(4)}
                  </span>
                  <span style={{ display: "flex", gap: 4 }}>
                    {r.pending_approval && (
                      <span className="badge badge-amber">GATE</span>
                    )}
                    {r.read_only && <span className="badge">read_only</span>}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Detail Workspace */}
      <div className="section-stack">
        {detailError && <div className="error-banner">{detailError}</div>}

        {!selectedRunId ? (
          <div className="card empty-state">
            Select a run on the left or launch a new run to inspect its pipeline,
            approval gate, and verification results.
          </div>
        ) : loadingDetail && !detail ? (
          <div className="card empty-state">Loading run {selectedRunId}...</div>
        ) : detail ? (
          <>
            {/* Run Header */}
            <div className="card">
              <div className="card-header">
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="mono" style={{ fontSize: 16, fontWeight: 700 }}>
                      {detail.run_id}
                    </span>
                    <StatusBadge value={detail.status} />
                    {detail.read_only && <span className="badge">read_only</span>}
                    {sseConnected && (
                      <span className="badge badge-cobalt">SSE LIVE</span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                    Dataset: <strong>{detail.dataset_label}</strong> · Policy:{" "}
                    <span className="mono">{detail.approval_policy}</span>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => onOpenTrace(detail.run_id)}
                  >
                    Inspect Trace
                  </button>
                  {!detail.read_only &&
                    !TERMINAL_STATUSES.has(detail.status) && (
                      <button
                        type="button"
                        className="btn btn-danger"
                        disabled={cancelling || detail.cancel_requested}
                        onClick={handleCancel}
                      >
                        {detail.cancel_requested
                          ? "Cancel Requested"
                          : cancelling
                            ? "Cancelling..."
                            : "Cancel Run"}
                      </button>
                    )}
                </div>
              </div>

              <div className="grid-4">
                <div className="kpi-card">
                  <div className="kpi-label">Spend / Budget</div>
                  <div className="kpi-value">
                    ${detail.spent_usd.toFixed(4)}
                  </div>
                  <div className="kpi-sub mono">
                    cap{" "}
                    {detail.budget_usd !== null
                      ? `$${detail.budget_usd.toFixed(2)}`
                      : "unbounded"}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Compute</div>
                  <div className="kpi-value">
                    {detail.spent_compute_s.toFixed(1)}s
                  </div>
                  <div className="kpi-sub mono">
                    cap{" "}
                    {detail.budget_compute_s !== null
                      ? `${detail.budget_compute_s.toFixed(0)}s`
                      : "—"}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Tokens</div>
                  <div className="kpi-value">
                    {detail.total_tokens.toLocaleString()}
                  </div>
                  <div className="kpi-sub mono">
                    {detail.models_used.length > 0
                      ? detail.models_used[0]
                      : "no model calls"}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Trials</div>
                  <div className="kpi-value">{detail.trials.length}</div>
                  <div className="kpi-sub mono">
                    {detail.jobs.length} sandbox jobs
                  </div>
                </div>
              </div>
            </div>

            {/* 5-Stage Pipeline Stepper + Role Handoff Chips */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Pipeline Progression</h3>
                <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  stage: {liveStage ?? detail.stage ?? detail.status}
                </span>
              </div>
              <div className="stepper">
                {PIPELINE_STAGES.map((st, idx) => {
                  let cls = "step-node";
                  let stateLabel = "pending";
                  if (detail.status === "FAILED" && idx === activeStageIdx) {
                    cls += " failed";
                    stateLabel = "failed";
                  } else if (
                    detail.status === "WAITING_APPROVAL" &&
                    st === "CONTRACT"
                  ) {
                    cls += " waiting";
                    stateLabel = "awaiting gate";
                  } else if (
                    detail.status === "DONE" ||
                    idx < activeStageIdx
                  ) {
                    cls += " done";
                    stateLabel = "complete";
                  } else if (idx === activeStageIdx) {
                    cls += " active";
                    stateLabel = "active";
                  }
                  return (
                    <div key={st} className={cls}>
                      <div className="step-label">{st}</div>
                      <div className="step-state">{stateLabel}</div>
                    </div>
                  );
                })}
              </div>

              <div className="role-chips">
                {ROLE_CHIPS.map((rc) => {
                  const seen = liveRoles.has(rc.key);
                  return (
                    <span
                      key={rc.key}
                      className={`badge ${seen ? "badge-cobalt" : ""}`}
                    >
                      {rc.label}
                      {seen ? " ✓" : ""}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Interactive Approval Gate */}
            {detail.pending_approval && (
              <ApprovalGate
                runId={detail.run_id}
                pending={detail.pending_approval}
                dataContract={detail.data_contract}
                experimentContract={detail.experiment_contract}
                readOnly={detail.read_only}
                onResolved={() => {
                  loadDetail(detail.run_id);
                  onRefreshRuns();
                }}
              />
            )}

            {/* Trials & Verification Table */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Trials &amp; Verification</h3>
                <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  platform-computed metrics
                </span>
              </div>
              {detail.trials.length === 0 ? (
                <div className="empty-state">No trials recorded yet.</div>
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Trial ID</th>
                        <th>Kind</th>
                        <th>Status</th>
                        <th>Val Score</th>
                        <th>Test Score</th>
                        <th>Verified</th>
                        <th>Hypothesis</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.trials.map((t) => {
                        const verified =
                          isSplitVerified(t.test_result) ??
                          isSplitVerified(t.val_result);
                        return (
                          <tr key={t.trial_id}>
                            <td className="mono">{t.trial_id}</td>
                            <td className="mono">{t.kind}</td>
                            <td>
                              <StatusBadge value={t.status} />
                            </td>
                            <td className="mono">
                              {formatMetricSplit(t.val_result)}
                            </td>
                            <td className="mono">
                              {formatMetricSplit(t.test_result)}
                            </td>
                            <td>
                              {verified === true ? (
                                <span className="badge badge-forest">VERIFIED</span>
                              ) : verified === false ? (
                                <span className="badge badge-crimson">FAILED</span>
                              ) : (
                                <span className="badge">—</span>
                              )}
                            </td>
                            <td style={{ maxWidth: 320 }}>{t.hypothesis}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Dataset Profile Summary (Aggregates Only) & Final Report */}
            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Dataset Profile (Aggregates Only)</h3>
                  {detail.profile && (
                    <span className="badge">{detail.profile.modality}</span>
                  )}
                </div>
                {!detail.profile ? (
                  <div className="empty-state">Profile not yet recorded.</div>
                ) : (
                  <div className="section-stack" style={{ gap: 10 }}>
                    <div className="kv-list">
                      <span className="kv-key">Record ID</span>
                      <span className="kv-val mono">{detail.profile.record_id}</span>
                      <span className="kv-key">Fingerprint</span>
                      <span className="kv-val mono">
                        {detail.profile.fingerprint}
                      </span>
                    </div>
                    <pre className="code-block">
                      {JSON.stringify(
                        detail.profile.tabular ??
                          detail.profile.timeseries ??
                          detail.profile.vision ??
                          {},
                        null,
                        2
                      )}
                    </pre>
                  </div>
                )}
              </div>

              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Final Report</h3>
                  {detail.report_json && (
                    <span className="badge badge-forest">AVAILABLE</span>
                  )}
                </div>
                {detail.report_md ? (
                  <pre className="code-block" style={{ maxHeight: 360 }}>
                    {detail.report_md}
                  </pre>
                ) : detail.report_json ? (
                  <pre className="code-block" style={{ maxHeight: 360 }}>
                    {JSON.stringify(detail.report_json, null, 2)}
                  </pre>
                ) : (
                  <div className="empty-state">
                    Final report will appear once Phase B and Reporting complete.
                  </div>
                )}
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};