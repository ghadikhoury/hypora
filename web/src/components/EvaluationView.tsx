import React, { useEffect, useMemo, useState } from "react";
import { getEvalSummary } from "../api";
import type { EvalSummaryResponse } from "../types";
import {
  F6DefenseMatrixChart,
  F7StageComparisonChart,
  PassRateIntervalPlot,
  type F6ConfigRow,
  type PassRateRow,
  type StageDurationRow,
} from "./Charts";
import { StatusBadge } from "./StatusBadge";

type AnyRecord = Record<string, unknown>;

function rec(v: unknown): AnyRecord {
  return typeof v === "object" && v !== null ? (v as AnyRecord) : {};
}

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function str(v: unknown, fallback = "—"): string {
  return typeof v === "string" && v.length > 0 ? v : fallback;
}

function pct(v: unknown): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

function extractWilson(pr: AnyRecord): [number, number] {
  if (Array.isArray(pr.wilson_ci_95) && pr.wilson_ci_95.length >= 2) {
    return [num(pr.wilson_ci_95[0]), num(pr.wilson_ci_95[1])];
  }
  if (Array.isArray(pr.wilson_95_ci) && pr.wilson_95_ci.length >= 2) {
    return [num(pr.wilson_95_ci[0]), num(pr.wilson_95_ci[1])];
  }
  return [num(pr.wilson_low), num(pr.wilson_high)];
}

const STAGE_COLORS: Record<string, string> = {
  INTAKE: "#64748B",
  PROFILE: "#94A3B8",
  CONTRACT: "#B45309",
  PHASE_B: "#1D4ED8",
  REPORT: "#15803D",
};

const STAGE_ORDER = ["INTAKE", "PROFILE", "CONTRACT", "PHASE_B", "REPORT"];

export const EvaluationView: React.FC = () => {
  const [data, setData] = useState<EvalSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [failureSearch, setFailureSearch] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getEvalSummary()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to load evaluation summary."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const consolidated = rec(data?.consolidated_summary);
  const coreTable = rec(consolidated.core_master_table);
  const systems = rec(coreTable.systems);
  const paired = rec(coreTable.paired_comparisons);
  const familyBreakdown = rec(coreTable.family_breakdown_multi_vs_baseline);
  const claimedFeatures = rec(consolidated.claimed_features_master_table);
  const failureCatalog = Array.isArray(consolidated.representative_failure_catalog)
    ? (consolidated.representative_failure_catalog as AnyRecord[])
    : [];

  const passRateRows: PassRateRow[] = useMemo(() => {
    const specs: {
      key: string;
      label: string;
      sublabel: string;
      color: string;
    }[] = [
      {
        key: "multi_post_fix",
        label: "Multi-Agent (post-fix)",
        sublabel: "f850588",
        color: "#15803D",
      },
      {
        key: "multi_pre_fix",
        label: "Multi-Agent (pre-fix)",
        sublabel: "62b4e3f",
        color: "#1D4ED8",
      },
      {
        key: "single",
        label: "Single-Agent",
        sublabel: "62b4e3f",
        color: "#475569",
      },
      {
        key: "baseline",
        label: "Fixed Baseline",
        sublabel: "62b4e3f",
        color: "#B45309",
      },
    ];
    return specs
      .filter((s) => systems[s.key])
      .map((s) => {
        const sys = rec(systems[s.key]);
        const pr = rec(sys.pass_rate);
        const [ciLow, ciHigh] = extractWilson(pr);
        return {
          label: s.label,
          sublabel: s.sublabel,
          passed: num(pr.passed),
          graded: num(pr.graded),
          rate: num(pr.rate),
          ciLow,
          ciHigh,
          color: s.color,
        };
      });
  }, [systems]);

  const f7Evidence = rec(data?.f7_evidence);
  const f7Analysis = rec(f7Evidence.analysis);
  const f7PreBreakdown = rec(f7Analysis.breakdown_pre_fix);
  const f7PostBreakdown = rec(f7Analysis.breakdown_post_fix);
  const f7Bottleneck = rec(rec(f7Analysis.before_vs_after).bottleneck_metrics);

  const stageRows: StageDurationRow[] = useMemo(() => {
    const buildRow = (
      label: string,
      commit: string,
      breakdown: AnyRecord
    ): StageDurationRow | null => {
      const byStage = rec(breakdown.by_stage);
      if (Object.keys(byStage).length === 0) return null;
      const stages = STAGE_ORDER.map((st) => ({
        name: st,
        meanS: num(rec(byStage[st]).wall_seconds_mean),
        color: STAGE_COLORS[st],
      }));
      const totalMeanS = stages.reduce((acc, s) => acc + s.meanS, 0);
      return { label, commit, totalMeanS, stages };
    };
    const r1 = buildRow("Pre-fix (eval-freeze)", "62b4e3f", f7PreBreakdown);
    const r2 = buildRow("Post-fix (single-pass)", "f850588", f7PostBreakdown);
    return [r1, r2].filter((x): x is StageDurationRow => x !== null);
  }, [f7PreBreakdown, f7PostBreakdown]);

  const f6Evidence = rec(data?.f6_evidence);
  const f6Configs = rec(f6Evidence.configs);
  const f6Rows: F6ConfigRow[] = useMemo(() => {
    const order = [
      "all_off",
      "no_allowlist",
      "no_firewall",
      "no_approval",
      "no_sandbox",
      "no_injection_filter",
      "all_on",
    ];
    return order
      .filter((k) => f6Configs[k])
      .map((k) => {
        const overall = rec(rec(f6Configs[k]).overall);
        const [ciLow, ciHigh] = extractWilson(overall);
        return {
          config: k,
          succeeded: num(overall.succeeded),
          total: num(overall.total),
          rate: num(overall.rate),
          ciLow,
          ciHigh,
        };
      });
  }, [f6Configs]);

  const filteredFailures = useMemo(() => {
    const q = failureSearch.trim().toLowerCase();
    if (!q) return failureCatalog;
    return failureCatalog.filter((item) =>
      JSON.stringify(item).toLowerCase().includes(q)
    );
  }, [failureCatalog, failureSearch]);

  if (loading) {
    return (
      <div className="card empty-state">
        Loading committed S5.3 / S5.4 evaluation artifacts...
      </div>
    );
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  if (!data?.consolidated_summary) {
    return (
      <div className="card empty-state">
        Evaluation artifacts not present in this checkout (`results/analysis/consolidated-summary.json` is null).
      </div>
    );
  }

  const multiPre = rec(systems.multi_pre_fix);
  const multiPost = rec(systems.multi_post_fix);
  const singleSys = rec(systems.single);
  const baseSys = rec(systems.baseline);

  const agentBCalls = rec(f7Bottleneck.agent_b_model_calls);
  const agentBLat = rec(f7Bottleneck.agent_b_llm_latency_seconds);
  const dupTurns = rec(f7Bottleneck.post_tool_duplicate_turns);

  return (
    <div className="section-stack">
      {/* Top KPI Summary */}
      <div className="grid-4">
        <div className="kpi-card">
          <div className="kpi-label">Multi-Agent Pass Rate</div>
          <div className="kpi-value">
            {pct(rec(multiPost.pass_rate).rate)}
          </div>
          <div className="kpi-sub mono">
            {num(rec(multiPost.pass_rate).passed)}/{num(rec(multiPost.pass_rate).graded)} held-out cases
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">vs Fixed Baseline</div>
          <div className="kpi-value">
            +{(num(rec(paired.multi_vs_baseline).difference) * 100).toFixed(1)} pp
          </div>
          <div className="kpi-sub mono">
            baseline {pct(rec(baseSys.pass_rate).rate)} · 95% CI [
            {(num(rec(paired.multi_vs_baseline).ci_low) * 100).toFixed(1)},{" "}
            {(num(rec(paired.multi_vs_baseline).ci_high) * 100).toFixed(1)}]
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">F7 Latency Reduction</div>
          <div className="kpi-value">
            {num(rec(multiPost.latency_seconds).mean).toFixed(1)}s
          </div>
          <div className="kpi-sub mono">
            from {num(rec(multiPre.latency_seconds).mean).toFixed(1)}s (
            {(
              num(rec(multiPost.latency_seconds).mean) -
              num(rec(multiPre.latency_seconds).mean)
            ).toFixed(1)}
            s)
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">F6 Safety ASR (all_on)</div>
          <div className="kpi-value">
            {pct(rec(rec(f6Configs.all_on).overall).rate)}
          </div>
          <div className="kpi-sub mono">
            vs {pct(rec(rec(f6Configs.all_off).overall).rate)} all_off (36 attacks)
          </div>
        </div>
      </div>

      {/* Core Comparison (n=26) */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            Core System Comparison (n=26 Held-Out Test Cases)
          </h3>
          <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
            95% Wilson CI · single {pct(rec(singleSys.pass_rate).rate)}
          </span>
        </div>
        <PassRateIntervalPlot rows={passRateRows} />
      </div>

      <div className="grid-2">
        {/* Paired Bootstrap Comparisons */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Paired Bootstrap Comparisons</h3>
            <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
              10,000 resamples · seed 1990
            </span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Comparison</th>
                  <th style={{ textAlign: "right" }}>Delta</th>
                  <th>95% CI</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(paired).map(([key, raw]) => {
                  const p = rec(raw);
                  const ci = Array.isArray(p.ci_95)
                    ? `[${(num(p.ci_95[0]) * 100).toFixed(1)}, ${(num(p.ci_95[1]) * 100).toFixed(1)}] pp`
                    : `[${(num(p.ci_low) * 100).toFixed(1)}, ${(num(p.ci_high) * 100).toFixed(1)}] pp`;
                  return (
                    <tr key={key}>
                      <td className="mono">{key}</td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {(num(p.difference) * 100).toFixed(1)} pp
                      </td>
                      <td className="mono">{ci}</td>
                      <td>
                        <StatusBadge value={str(p.verdict)} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Family Breakdown */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              Family Breakdown (Multi vs Baseline)
            </h3>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Family</th>
                  <th style={{ textAlign: "right" }}>Cases</th>
                  <th style={{ textAlign: "right" }}>Multi Only</th>
                  <th style={{ textAlign: "right" }}>Diff</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(familyBreakdown).map(([fam, raw]) => {
                  const f = rec(raw);
                  return (
                    <tr key={fam}>
                      <td>{fam}</td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {num(f.cases)}
                      </td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {num(f.first_only)}
                      </td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        +{(num(f.difference) * 100).toFixed(0)} pp
                      </td>
                      <td>
                        <StatusBadge value={str(f.verdict)} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* F7 Observability Bottleneck & F6 Safety Matrix */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              F7 Observability Bottleneck (62b4e3f → f850588)
            </h3>
            <span className="badge badge-forest">-147.8s mean</span>
          </div>
          {stageRows.length > 0 && <F7StageComparisonChart rows={stageRows} />}
          <div className="kv-list" style={{ marginTop: 12 }}>
            <span className="kv-key">Agent B Latency</span>
            <span className="kv-val mono">
              {num(rec(agentBLat.paired_per_run).mean_second).toFixed(1)}s →{" "}
              {num(rec(agentBLat.paired_per_run).mean_first).toFixed(1)}s (
              {num(agentBLat.pre_fix_total).toFixed(1)}s →{" "}
              {num(agentBLat.post_fix_total).toFixed(1)}s total)
            </span>
            <span className="kv-key">Agent B Calls</span>
            <span className="kv-val mono">
              {num(agentBCalls.pre_fix_total)} → {num(agentBCalls.post_fix_total)}
            </span>
            <span className="kv-key">Duplicate Turns</span>
            <span className="kv-val mono">
              {num(dupTurns.pre_fix_total)} ({num(dupTurns.pre_fix_runs_affected)}/26 runs) →{" "}
              {num(dupTurns.post_fix_total)} ({num(dupTurns.post_fix_runs_affected)}/26 runs)
            </span>
            <span className="kv-key">Repeat Sample</span>
            <span className="kv-val mono">
              {num(rec(rec(rec(consolidated.f7_repeat_or_core).f7_pre_vs_post).post_fix_overall).passed, 18)}/18
              post-fix vs 17/18 pre-fix
            </span>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              F6 Safety &amp; Security Defense Matrix (36 Attacks × 7 Configs)
            </h3>
            <span className="badge badge-forest">all_on 2.8% ASR</span>
          </div>
          <F6DefenseMatrixChart rows={f6Rows} />
          <div
            style={{
              marginTop: 10,
              fontSize: 11,
              color: "var(--text-muted)",
              padding: "8px 10px",
              background: "var(--amber-bg)",
              border: "1px solid var(--amber-border)",
              borderRadius: 4,
            }}
          >
            <strong>Residual risk note:</strong> <span className="mono">ATK-INJ-10</span>{" "}
            (1/36 = 2.8% ASR in <span className="mono">all_on</span>) bypassed the
            regex injection filter via semantic false-score prose; contained
            structurally because platform code owns official metrics and verdicts.
          </div>
        </div>
      </div>

      {/* Claimed Features (60 Raw Pts / 40 Cap) */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            Claimed Features Master Table (60 Raw Pts / 40 Cap)
          </h3>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Title</th>
                <th style={{ textAlign: "right" }}>Raw Pts</th>
                <th>Key Counts</th>
                <th>Honest Limitations</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(claimedFeatures).map(([fid, raw]) => {
                const f = rec(raw);
                const lims = Array.isArray(f.honest_limitations)
                  ? (f.honest_limitations as string[])
                  : [];
                return (
                  <tr key={fid}>
                    <td className="mono" style={{ fontWeight: 600 }}>
                      {fid}
                    </td>
                    <td>{str(f.title)}</td>
                    <td className="mono" style={{ textAlign: "right" }}>
                      {num(f.raw_points)}
                    </td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {Object.entries(rec(f.counts))
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </td>
                    <td style={{ fontSize: 11, color: "var(--text-muted)" }}>
                      {lims.join(" ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 10-Case Representative Failure Catalog */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            Representative Failure Catalog ({filteredFailures.length} / {failureCatalog.length})
          </h3>
          <input
            className="input"
            style={{ width: 280 }}
            placeholder="Search case_id, cause, evidence path..."
            value={failureSearch}
            onChange={(e) => setFailureSearch(e.target.value)}
          />
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>System / Feature</th>
                <th>Case IDs</th>
                <th>Cause</th>
                <th>Evidence Paths</th>
              </tr>
            </thead>
            <tbody>
              {filteredFailures.map((item) => {
                const cases = Array.isArray(item.case_ids)
                  ? (item.case_ids as string[])
                  : [];
                const paths = Array.isArray(item.evidence_paths)
                  ? (item.evidence_paths as string[])
                  : [];
                return (
                  <tr key={str(item.category_id, str(item.category))}>
                    <td style={{ fontWeight: 600 }}>{str(item.category)}</td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {str(item.system_or_feature)}
                    </td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {cases.join(", ")}
                    </td>
                    <td style={{ maxWidth: 320 }}>{str(item.cause)}</td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {paths.join("\n")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};