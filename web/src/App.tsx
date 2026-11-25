import React, { useCallback, useEffect, useState } from "react";
import { getHealth, getRuns } from "./api";
import { EvaluationView } from "./components/EvaluationView";
import { NewRunDrawer } from "./components/NewRunDrawer";
import { RunsView } from "./components/RunsView";
import { TraceView } from "./components/TraceView";
import type { HealthResponse, RunListItem } from "./types";

type WorkspaceTab = "runs" | "trace" | "evaluation";

const TAB_META: Record<
  WorkspaceTab,
  { title: string; subtitle: string }
> = {
  runs: {
    title: "Runs & Approval Gate",
    subtitle: "Pipeline progression, interactive plan gate, and platform-verified trials",
  },
  trace: {
    title: "Trace Inspector",
    subtitle: "Append-only event log, role handoffs, and policy audit trail",
  },
  evaluation: {
    title: "Evaluation & Safety Benchmarks",
    subtitle: "Committed S5.3 / S5.4 evidence across 26 held-out cases, F7, and F6",
  },
};

export const App: React.FC = () => {
  const [tab, setTab] = useState<WorkspaceTab>("runs");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [newRunOpen, setNewRunOpen] = useState(false);

  const refreshHealth = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(false);
      })
      .catch(() => {
        setHealthError(true);
      });
  }, []);

  const refreshRuns = useCallback(() => {
    getRuns(100)
      .then((res) => {
        setRuns(res.runs);
        setSelectedRunId((prev) => {
          if (prev && res.runs.some((r) => r.run_id === prev)) return prev;
          return res.runs.length > 0 ? res.runs[0].run_id : null;
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshHealth();
    refreshRuns();
    const id = setInterval(() => {
      refreshHealth();
      refreshRuns();
    }, 15000);
    return () => clearInterval(id);
  }, [refreshHealth, refreshRuns]);

  const waitingCount = runs.filter(
    (r) => r.pending_approval || r.status === "WAITING_APPROVAL"
  ).length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-block">
            <h1 className="wordmark">HYPORA</h1>
            <div className="commit-pill">
              <span>{health?.git_sha ?? "local"}</span>
              <span style={{ color: "var(--border-strong)" }}>|</span>
              <span>eval-freeze 62b4e3f</span>
            </div>
          </div>

          <nav className="nav-list" aria-label="Primary workspaces">
            <button
              type="button"
              className={`nav-item ${tab === "runs" ? "active" : ""}`}
              onClick={() => setTab("runs")}
            >
              <span>Runs &amp; Gate</span>
              <span className="nav-count">
                {waitingCount > 0 ? `${waitingCount} gate` : runs.length}
              </span>
            </button>
            <button
              type="button"
              className={`nav-item ${tab === "trace" ? "active" : ""}`}
              onClick={() => setTab("trace")}
            >
              <span>Trace Inspector</span>
              <span className="nav-count">events</span>
            </button>
            <button
              type="button"
              className={`nav-item ${tab === "evaluation" ? "active" : ""}`}
              onClick={() => setTab("evaluation")}
            >
              <span>Evaluation</span>
              <span className="nav-count">n=26</span>
            </button>
          </nav>
        </div>

        <div className="sidebar-footer">
          <div className="health-row">
            <span
              className={`health-dot ${healthError ? "err" : health ? "ok" : ""}`}
            />
            <span>
              {healthError
                ? "API disconnected"
                : health
                  ? "API connected"
                  : "Connecting..."}
            </span>
          </div>
          <div className="mono" style={{ fontSize: 10, color: "var(--text-faint)" }}>
            {health?.db_exists ? "sqlite ready" : "no active db"}
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <h2 className="topbar-title">{TAB_META[tab].title}</h2>
            <span className="topbar-subtitle">{TAB_META[tab].subtitle}</span>
          </div>
          <div className="topbar-actions">
            <button
              type="button"
              className="btn"
              onClick={() => {
                refreshHealth();
                refreshRuns();
              }}
            >
              Refresh
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setNewRunOpen(true)}
            >
              + New Run
            </button>
          </div>
        </header>

        <main className="workspace-body">
          {tab === "runs" && (
            <RunsView
              runs={runs}
              selectedRunId={selectedRunId}
              onSelectRun={setSelectedRunId}
              onRefreshRuns={refreshRuns}
              onOpenTrace={(runId) => {
                setSelectedRunId(runId);
                setTab("trace");
              }}
            />
          )}

          {tab === "trace" && (
            <TraceView
              runs={runs}
              selectedRunId={selectedRunId}
              onSelectRun={setSelectedRunId}
            />
          )}

          {tab === "evaluation" && <EvaluationView />}
        </main>
      </div>

      <NewRunDrawer
        open={newRunOpen}
        onClose={() => setNewRunOpen(false)}
        onCreated={(runId) => {
          setSelectedRunId(runId);
          setTab("runs");
          refreshRuns();
        }}
      />
    </div>
  );
};