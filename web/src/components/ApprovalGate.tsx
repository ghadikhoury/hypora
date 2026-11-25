import React, { useState } from "react";
import { approveRun } from "../api";
import type { PendingApprovalDetail } from "../types";

interface ApprovalGateProps {
  runId: string;
  pending: PendingApprovalDetail;
  dataContract: Record<string, unknown> | null;
  experimentContract: Record<string, unknown> | null;
  readOnly: boolean;
  onResolved: () => void;
}

function asStr(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "string") return val;
  if (typeof val === "number" || typeof val === "boolean") return String(val);
  return JSON.stringify(val);
}

export const ApprovalGate: React.FC<ApprovalGateProps> = ({
  runId,
  pending,
  dataContract,
  experimentContract,
  readOnly,
  onResolved,
}) => {
  const [denyMode, setDenyMode] = useState(false);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scope = pending.scope ?? experimentContract ?? {};
  const dc = dataContract ?? {};
  const excludedCols = Array.isArray(dc.excluded_columns)
    ? dc.excluded_columns
    : [];

  const handleDecision = async (approve: boolean) => {
    setError(null);
    if (!approve && !reason.trim()) {
      setError("A denial reason is required.");
      return;
    }
    setSubmitting(true);
    try {
      await approveRun(runId, {
        approve,
        hash_prefix: pending.hash_prefix,
        reason: approve ? null : reason.trim(),
        user: "web-ui",
      });
      onResolved();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to submit approval decision."
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="approval-gate">
      <div className="card-header" style={{ marginBottom: 8 }}>
        <div>
          <div
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "var(--amber)",
              letterSpacing: "0.02em",
            }}
          >
            INTERACTIVE PLAN &amp; BUDGET APPROVAL GATE
          </div>
          <div style={{ fontSize: 12, color: "var(--text-ink)", marginTop: 2 }}>
            {pending.summary}
          </div>
        </div>
        <span className="badge badge-amber">
          plan v{pending.plan_version ?? "?"} · {pending.policy}
        </span>
      </div>

      <div
        className="mono"
        style={{
          fontSize: 11,
          background: "var(--bg-surface)",
          border: "1px solid var(--amber-border)",
          borderRadius: 4,
          padding: "6px 10px",
          wordBreak: "break-all",
        }}
      >
        artifact_hash: {pending.artifact_hash} (prefix:{" "}
        <strong>{pending.hash_prefix}</strong>)
      </div>

      <div className="approval-grid">
        <div className="approval-box">
          <div className="card-title" style={{ marginBottom: 8 }}>
            DataContract
          </div>
          <div className="kv-list">
            <span className="kv-key">Target</span>
            <span className="kv-val mono">{asStr(dc.target_column ?? dc.target)}</span>
            <span className="kv-key">Primary Metric</span>
            <span className="kv-val mono">{asStr(dc.primary_metric)}</span>
            <span className="kv-key">Split Strategy</span>
            <span className="kv-val mono">{asStr(dc.split_strategy)}</span>
            <span className="kv-key">Group Key</span>
            <span className="kv-val mono">{asStr(dc.group_column ?? dc.group_key)}</span>
            <span className="kv-key">Time Key</span>
            <span className="kv-val mono">{asStr(dc.time_column ?? dc.time_key)}</span>
            <span className="kv-key">Excluded Cols</span>
            <span className="kv-val mono">
              {excludedCols.length === 0
                ? "none"
                : excludedCols
                    .map((item) => {
                      if (typeof item === "string") return item;
                      if (typeof item === "object" && item !== null) {
                        const obj = item as Record<string, unknown>;
                        const col = asStr(obj.column ?? obj.name);
                        const ref = obj.evidence_ref ? ` [${asStr(obj.evidence_ref)}]` : "";
                        const rsn = obj.reason ? ` (${asStr(obj.reason)})` : "";
                        return `${col}${rsn}${ref}`;
                      }
                      return asStr(item);
                    })
                    .join("; ")}
            </span>
          </div>
        </div>

        <div className="approval-box">
          <div className="card-title" style={{ marginBottom: 8 }}>
            Budget &amp; Execution Bounds
          </div>
          <div className="kv-list">
            <span className="kv-key">Approval ID</span>
            <span className="kv-val mono">{pending.approval_id}</span>
            <span className="kv-key">Max Spend (USD)</span>
            <span className="kv-val mono">
              {scope.max_usd !== undefined
                ? `$${Number(scope.max_usd).toFixed(2)}`
                : scope.budget_usd !== undefined
                  ? `$${Number(scope.budget_usd).toFixed(2)}`
                  : "—"}
            </span>
            <span className="kv-key">Compute Limit</span>
            <span className="kv-val mono">
              {scope.max_compute_seconds !== undefined
                ? `${scope.max_compute_seconds}s`
                : scope.budget_compute_s !== undefined
                  ? `${scope.budget_compute_s}s`
                  : "—"}
            </span>
            <span className="kv-key">Max Trials</span>
            <span className="kv-val mono">{asStr(scope.max_trials)}</span>
            <span className="kv-key">Artifact Ref</span>
            <span className="kv-val mono">{asStr(pending.artifact_ref)}</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ marginTop: 12 }}>
          {error}
        </div>
      )}

      {!readOnly && (
        <div style={{ marginTop: 12 }}>
          {denyMode ? (
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                className="input"
                placeholder="Required: reason for denying plan..."
                value={reason}
                onChange={(e) => setReason(e.target.value)}
              />
              <button
                type="button"
                className="btn btn-danger"
                disabled={submitting}
                onClick={() => handleDecision(false)}
              >
                {submitting ? "Denying..." : "Confirm Deny"}
              </button>
              <button
                type="button"
                className="btn"
                disabled={submitting}
                onClick={() => {
                  setDenyMode(false);
                  setError(null);
                }}
              >
                Cancel
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className="btn btn-approve"
                disabled={submitting}
                onClick={() => handleDecision(true)}
              >
                {submitting ? "Approving..." : "Approve Plan"}
              </button>
              <button
                type="button"
                className="btn btn-danger"
                disabled={submitting}
                onClick={() => setDenyMode(true)}
              >
                Deny Plan...
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};