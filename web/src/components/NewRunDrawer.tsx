import React, { useEffect, useState } from "react";
import { createRun, getDatasets } from "../api";
import type {
  ApprovalPolicy,
  DatasetItem,
  MetricName,
  RunCreateRequest,
  TaskType,
} from "../types";

interface NewRunDrawerProps {
  open: boolean;
  onClose: () => void;
  onCreated: (runId: string) => void;
}

const TASK_TYPES: TaskType[] = [
  "binary_classification",
  "multiclass_classification",
  "multilabel_classification",
  "regression",
  "forecasting",
  "window_classification",
];

const METRICS: MetricName[] = [
  "roc_auc",
  "pr_auc",
  "f1",
  "f1_macro",
  "accuracy",
  "balanced_accuracy",
  "log_loss",
  "brier_score",
  "rmse",
  "mae",
  "r2",
  "precision",
  "recall",
  "precision_macro",
  "recall_macro",
];

export const NewRunDrawer: React.FC<NewRunDrawerProps> = ({
  open,
  onClose,
  onCreated,
}) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [sourceMode, setSourceMode] = useState<"snapshot" | "custom">("snapshot");
  const [selectedSnapshot, setSelectedSnapshot] = useState<string>("");
  const [customPath, setCustomPath] = useState<string>("");
  const [goal, setGoal] = useState<string>("");
  const [target, setTarget] = useState<string>("");
  const [taskType, setTaskType] = useState<string>("");
  const [metric, setMetric] = useState<string>("");
  const [budgetUsd, setBudgetUsd] = useState<string>("0.50");
  const [maxTrials, setMaxTrials] = useState<string>("4");
  const [approvePolicy, setApprovePolicy] = useState<ApprovalPolicy>("interactive");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    getDatasets()
      .then((res) => {
        if (cancelled) return;
        const available = res.datasets.filter((d) => d.exists);
        setDatasets(available);
        if (available.length > 0 && !selectedSnapshot) {
          setSelectedSnapshot(available[0].path);
          if (available[0].target && !target) {
            setTarget(available[0].target);
          }
        }
      })
      .catch(() => {
        // non-fatal; user can still enter custom path
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  if (!open) return null;

  const handleSnapshotChange = (path: string) => {
    setSelectedSnapshot(path);
    const found = datasets.find((d) => d.path === path);
    if (found?.target) {
      setTarget(found.target);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const dataPath =
      sourceMode === "snapshot" ? selectedSnapshot.trim() : customPath.trim();
    if (!dataPath) {
      setError("Select a dataset snapshot or enter a .csv / .parquet path.");
      return;
    }
    if (!goal.trim()) {
      setError("Enter a plain-language ML objective.");
      return;
    }

    const payload: RunCreateRequest = {
      data: dataPath,
      goal: goal.trim(),
      approve: approvePolicy,
    };
    if (target.trim()) payload.target = target.trim();
    if (taskType) payload.task_type = taskType as TaskType;
    if (metric) payload.metric = metric as MetricName;
    if (budgetUsd.trim() !== "") {
      const b = Number(budgetUsd);
      if (!Number.isFinite(b) || b < 0) {
        setError("Budget USD must be a non-negative number.");
        return;
      }
      payload.budget_usd = b;
    }
    if (maxTrials.trim() !== "") {
      const m = Number(maxTrials);
      if (!Number.isInteger(m) || m < 0) {
        setError("Max trials must be a non-negative integer.");
        return;
      }
      payload.max_trials = m;
    }

    setSubmitting(true);
    try {
      const created = await createRun(payload);
      onCreated(created.run_id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create run.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Launch New Run</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              POST /api/runs — deterministic split &amp; sandboxed execution
            </div>
          </div>
          <button type="button" className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </div>

        <form onSubmit={handleSubmit} className="drawer-body">
          {error && <div className="error-banner">{error}</div>}

          <div className="field-group">
            <span className="field-label">Dataset Source</span>
            <div className="chip-row">
              <button
                type="button"
                className={`chip ${sourceMode === "snapshot" ? "active" : ""}`}
                onClick={() => setSourceMode("snapshot")}
              >
                Recorded Snapshot ({datasets.length})
              </button>
              <button
                type="button"
                className={`chip ${sourceMode === "custom" ? "active" : ""}`}
                onClick={() => setSourceMode("custom")}
              >
                Custom .csv / .parquet
              </button>
            </div>
          </div>

          {sourceMode === "snapshot" ? (
            <div className="field-group">
              <label className="field-label" htmlFor="snapshot-select">
                Snapshot
              </label>
              <select
                id="snapshot-select"
                className="select"
                value={selectedSnapshot}
                onChange={(e) => handleSnapshotChange(e.target.value)}
              >
                {datasets.length === 0 && (
                  <option value="">No recorded snapshots found</option>
                )}
                {datasets.map((d) => (
                  <option key={d.name} value={d.path}>
                    {d.name} ({d.rows.toLocaleString()} rows × {d.columns} cols)
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="field-group">
              <label className="field-label" htmlFor="custom-path">
                Dataset Path
              </label>
              <input
                id="custom-path"
                className="input mono"
                placeholder="/path/to/dataset.csv or .parquet"
                value={customPath}
                onChange={(e) => setCustomPath(e.target.value)}
              />
            </div>
          )}

          <div className="field-group">
            <label className="field-label" htmlFor="goal-input">
              Plain-Language Goal *
            </label>
            <textarea
              id="goal-input"
              className="textarea"
              placeholder="e.g., Predict customer churn with strong generalization and guard against target leakage."
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
            />
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label className="field-label" htmlFor="target-input">
                Target Column (optional)
              </label>
              <input
                id="target-input"
                className="input mono"
                placeholder="auto-detect"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
              />
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="approval-select">
                Approval Policy
              </label>
              <select
                id="approval-select"
                className="select"
                value={approvePolicy}
                onChange={(e) =>
                  setApprovePolicy(e.target.value as ApprovalPolicy)
                }
              >
                <option value="interactive">interactive (gate before Phase B)</option>
                <option value="auto_within_contract">auto_within_contract</option>
              </select>
            </div>
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label className="field-label" htmlFor="task-select">
                Task Type (optional)
              </label>
              <select
                id="task-select"
                className="select"
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
              >
                <option value="">auto-detect</option>
                {TASK_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="metric-select">
                Primary Metric (optional)
              </label>
              <select
                id="metric-select"
                className="select"
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
              >
                <option value="">auto-detect</option>
                {METRICS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label className="field-label" htmlFor="budget-input">
                Budget USD
              </label>
              <input
                id="budget-input"
                type="number"
                step="0.05"
                min="0"
                className="input mono"
                value={budgetUsd}
                onChange={(e) => setBudgetUsd(e.target.value)}
              />
            </div>

            <div className="field-group">
              <label className="field-label" htmlFor="trials-input">
                Max Trials
              </label>
              <input
                id="trials-input"
                type="number"
                step="1"
                min="0"
                className="input mono"
                value={maxTrials}
                onChange={(e) => setMaxTrials(e.target.value)}
              />
            </div>
          </div>

          <div className="drawer-footer" style={{ padding: "14px 0 0 0" }}>
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting}
            >
              {submitting ? "Launching..." : "Start Run"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};