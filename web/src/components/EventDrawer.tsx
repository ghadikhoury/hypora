import React, { useState } from "react";
import type { TraceEventItem } from "../types";
import { StatusBadge } from "./StatusBadge";

interface EventDrawerProps {
  event: TraceEventItem | null;
  onClose: () => void;
}

function strOrDash(val: unknown): string {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
    return String(val);
  }
  return JSON.stringify(val);
}

export const EventDrawer: React.FC<EventDrawerProps> = ({ event, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!event) return null;

  const payload = event.payload ?? {};
  const prettyJson = JSON.stringify(payload, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(prettyJson);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable
    }
  };

  const promptRef = payload.prompt_ref ?? payload.prompt_version;
  const costUsd =
    typeof payload.cost_usd === "number" ? payload.cost_usd : null;
  const cacheTokens =
    payload.cached_tokens ?? payload.cache_read_tokens ?? null;
  const policyDecision =
    typeof payload.decision === "string"
      ? payload.decision
      : typeof payload.allowed === "boolean"
        ? payload.allowed
          ? "ALLOWED"
          : "DENIED"
        : null;
  const policyRule = payload.rule ?? payload.policy_rule ?? null;
  const policyReasons = payload.reasons ?? payload.reason ?? null;
  const hasHandoff =
    event.type === "handoff" ||
    Boolean(payload.from_role || payload.to_role || payload.output_ref);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="mono" style={{ fontWeight: 700, fontSize: 14 }}>
                SEQ #{event.seq}
              </span>
              <StatusBadge value={event.type} />
              <span className="badge">{event.role}</span>
            </div>
            <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
              +{event.offset_s.toFixed(2)}s · {event.created_at}
            </div>
          </div>
          <button type="button" className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="drawer-body">
          <div className="card" style={{ padding: 12 }}>
            <div className="card-title" style={{ marginBottom: 8 }}>
              Telemetry &amp; Context
            </div>
            <div className="kv-list">
              <span className="kv-key">Duration</span>
              <span className="kv-val mono">
                {event.duration_s !== null ? `${event.duration_s.toFixed(3)}s` : "—"}
              </span>
              <span className="kv-key">Tokens (In / Out)</span>
              <span className="kv-val mono">
                {event.tokens_in !== null || event.tokens_out !== null
                  ? `${(event.tokens_in ?? 0).toLocaleString()} / ${(event.tokens_out ?? 0).toLocaleString()}`
                  : "—"}
              </span>
              {cacheTokens !== null && (
                <>
                  <span className="kv-key">Cache Tokens</span>
                  <span className="kv-val mono">{strOrDash(cacheTokens)}</span>
                </>
              )}
              {costUsd !== null && (
                <>
                  <span className="kv-key">Cost (USD)</span>
                  <span className="kv-val mono">${costUsd.toFixed(6)}</span>
                </>
              )}
              {promptRef !== undefined && (
                <>
                  <span className="kv-key">Prompt Ref</span>
                  <span className="kv-val mono">{strOrDash(promptRef)}</span>
                </>
              )}
              <span className="kv-key">Trial ID</span>
              <span className="kv-val mono">{strOrDash(event.trial_id)}</span>
              <span className="kv-key">Job ID</span>
              <span className="kv-val mono">{strOrDash(event.job_id)}</span>
              <span className="kv-key">Summary</span>
              <span className="kv-val">{event.detail}</span>
            </div>
          </div>

          {(policyDecision !== null || policyRule !== null || policyReasons !== null) && (
            <div className="card" style={{ padding: 12 }}>
              <div className="card-title" style={{ marginBottom: 8 }}>
                Policy Decision
              </div>
              <div className="kv-list">
                {policyDecision && (
                  <>
                    <span className="kv-key">Verdict</span>
                    <span className="kv-val">
                      <StatusBadge value={policyDecision} />
                    </span>
                  </>
                )}
                {policyRule !== null && (
                  <>
                    <span className="kv-key">Rule</span>
                    <span className="kv-val mono">{strOrDash(policyRule)}</span>
                  </>
                )}
                {policyReasons !== null && (
                  <>
                    <span className="kv-key">Reasons</span>
                    <span className="kv-val mono">{strOrDash(policyReasons)}</span>
                  </>
                )}
              </div>
            </div>
          )}

          {hasHandoff && (
            <div className="card" style={{ padding: 12 }}>
              <div className="card-title" style={{ marginBottom: 8 }}>
                Role Handoff
              </div>
              <div className="kv-list">
                <span className="kv-key">From → To</span>
                <span className="kv-val mono">
                  {strOrDash(payload.from_role)} → {strOrDash(payload.to_role)}
                </span>
                <span className="kv-key">Output Ref</span>
                <span className="kv-val mono">{strOrDash(payload.output_ref)}</span>
                {payload.status !== undefined && (
                  <>
                    <span className="kv-key">Status</span>
                    <span className="kv-val mono">{strOrDash(payload.status)}</span>
                  </>
                )}
              </div>
            </div>
          )}

          <div className="field-group">
            <div className="card-header" style={{ marginBottom: 4 }}>
              <span className="field-label">Redacted Event Payload (JSON)</span>
              <button type="button" className="btn btn-sm" onClick={handleCopy}>
                {copied ? "Copied" : "Copy JSON"}
              </button>
            </div>
            <pre className="code-block">{prettyJson}</pre>
          </div>
        </div>
      </div>
    </div>
  );
};