import React from "react";

export interface PassRateRow {
  label: string;
  sublabel: string;
  passed: number;
  graded: number;
  rate: number;
  ciLow: number;
  ciHigh: number;
  color: string;
}

export const PassRateIntervalPlot: React.FC<{ rows: PassRateRow[] }> = ({
  rows,
}) => {
  const width = 640;
  const rowH = 42;
  const padL = 175;
  const padR = 125;
  const padT = 18;
  const padB = 28;
  const plotW = width - padL - padR;
  const height = padT + rows.length * rowH + padB;
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="Pass rate and 95% Wilson confidence intervals"
    >
      {/* Vertical gridlines */}
      {ticks.map((t) => {
        const x = padL + t * plotW;
        return (
          <g key={t}>
            <line
              x1={x}
              y1={padT}
              x2={x}
              y2={height - padB}
              stroke="#E4E2DA"
              strokeWidth={1}
            />
            <text
              x={x}
              y={height - 8}
              textAnchor="middle"
              fontSize={10}
              fill="#71717A"
              fontFamily="IBM Plex Mono, monospace"
            >
              {Math.round(t * 100)}%
            </text>
          </g>
        );
      })}

      {/* Rows */}
      {rows.map((r, i) => {
        const cy = padT + i * rowH + rowH / 2;
        const xRate = padL + r.rate * plotW;
        const xLow = padL + r.ciLow * plotW;
        const xHigh = padL + r.ciHigh * plotW;
        return (
          <g key={r.label}>
            <text
              x={padL - 10}
              y={cy - 3}
              textAnchor="end"
              fontSize={12}
              fontWeight={600}
              fill="#18181B"
            >
              {r.label}
            </text>
            <text
              x={padL - 10}
              y={cy + 11}
              textAnchor="end"
              fontSize={10}
              fill="#71717A"
              fontFamily="IBM Plex Mono, monospace"
            >
              {r.sublabel}
            </text>

            {/* Track */}
            <line
              x1={padL}
              y1={cy}
              x2={padL + plotW}
              y2={cy}
              stroke="#F0EFE9"
              strokeWidth={6}
              strokeLinecap="round"
            />

            {/* 95% Wilson CI whisker */}
            <line
              x1={xLow}
              y1={cy}
              x2={xHigh}
              y2={cy}
              stroke={r.color}
              strokeWidth={2.5}
            />
            <line
              x1={xLow}
              y1={cy - 5}
              x2={xLow}
              y2={cy + 5}
              stroke={r.color}
              strokeWidth={2}
            />
            <line
              x1={xHigh}
              y1={cy - 5}
              x2={xHigh}
              y2={cy + 5}
              stroke={r.color}
              strokeWidth={2}
            />

            {/* Point estimate */}
            <circle
              cx={xRate}
              cy={cy}
              r={5}
              fill={r.color}
              stroke="#FFFFFF"
              strokeWidth={1.5}
            />

            {/* Tabular label */}
            <text
              x={padL + plotW + 12}
              y={cy + 4}
              fontSize={11}
              fontWeight={600}
              fill="#18181B"
              fontFamily="IBM Plex Mono, monospace"
            >
              {(r.rate * 100).toFixed(1)}% ({r.passed}/{r.graded})
            </text>
          </g>
        );
      })}
    </svg>
  );
};

export interface StageDurationRow {
  label: string;
  commit: string;
  totalMeanS: number;
  stages: { name: string; meanS: number; color: string }[];
}

export const F7StageComparisonChart: React.FC<{
  rows: StageDurationRow[];
}> = ({ rows }) => {
  const width = 640;
  const rowH = 54;
  const padL = 145;
  const padR = 90;
  const padT = 16;
  const padB = 46;
  const plotW = width - padL - padR;
  const maxVal = Math.max(...rows.map((r) => r.totalMeanS), 700);
  const scaleMax = Math.ceil(maxVal / 100) * 100;
  const height = padT + rows.length * rowH + padB;
  const ticks = [0, 200, 400, 600, scaleMax].filter(
    (v, i, arr) => arr.indexOf(v) === i && v <= scaleMax
  );

  const legendItems = rows[0]?.stages ?? [];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="F7 mean stage duration comparison"
    >
      {ticks.map((t) => {
        const x = padL + (t / scaleMax) * plotW;
        return (
          <g key={t}>
            <line
              x1={x}
              y1={padT}
              x2={x}
              y2={padT + rows.length * rowH}
              stroke="#E4E2DA"
              strokeWidth={1}
            />
            <text
              x={x}
              y={padT + rows.length * rowH + 14}
              textAnchor="middle"
              fontSize={10}
              fill="#71717A"
              fontFamily="IBM Plex Mono, monospace"
            >
              {t}s
            </text>
          </g>
        );
      })}

      {rows.map((r, i) => {
        const y = padT + i * rowH + 12;
        const barH = 22;
        let curX = padL;
        return (
          <g key={r.label}>
            <text
              x={padL - 10}
              y={y + 10}
              textAnchor="end"
              fontSize={12}
              fontWeight={600}
              fill="#18181B"
            >
              {r.label}
            </text>
            <text
              x={padL - 10}
              y={y + 23}
              textAnchor="end"
              fontSize={10}
              fill="#71717A"
              fontFamily="IBM Plex Mono, monospace"
            >
              {r.commit}
            </text>

            {r.stages.map((st) => {
              const w = Math.max(0, (st.meanS / scaleMax) * plotW);
              const rectX = curX;
              curX += w;
              return (
                <rect
                  key={st.name}
                  x={rectX}
                  y={y}
                  width={w}
                  height={barH}
                  fill={st.color}
                >
                  <title>{`${st.name}: ${st.meanS.toFixed(1)}s`}</title>
                </rect>
              );
            })}

            <text
              x={curX + 8}
              y={y + 15}
              fontSize={11}
              fontWeight={600}
              fill="#18181B"
              fontFamily="IBM Plex Mono, monospace"
            >
              {r.totalMeanS.toFixed(1)}s
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <g transform={`translate(${padL}, ${height - 14})`}>
        {legendItems.map((st, idx) => {
          const lx = idx * 96;
          return (
            <g key={st.name} transform={`translate(${lx}, 0)`}>
              <rect x={0} y={-8} width={10} height={10} fill={st.color} rx={2} />
              <text
                x={14}
                y={1}
                fontSize={10}
                fill="#52525B"
                fontFamily="IBM Plex Mono, monospace"
              >
                {st.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
};

export interface F6ConfigRow {
  config: string;
  succeeded: number;
  total: number;
  rate: number;
  ciLow: number;
  ciHigh: number;
}

export const F6DefenseMatrixChart: React.FC<{ rows: F6ConfigRow[] }> = ({
  rows,
}) => {
  const width = 640;
  const rowH = 34;
  const padL = 165;
  const padR = 115;
  const padT = 14;
  const padB = 26;
  const plotW = width - padL - padR;
  const height = padT + rows.length * rowH + padB;
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="F6 attack success rate across defense configurations"
    >
      {ticks.map((t) => {
        const x = padL + t * plotW;
        return (
          <g key={t}>
            <line
              x1={x}
              y1={padT}
              x2={x}
              y2={height - padB}
              stroke="#E4E2DA"
              strokeWidth={1}
            />
            <text
              x={x}
              y={height - 6}
              textAnchor="middle"
              fontSize={10}
              fill="#71717A"
              fontFamily="IBM Plex Mono, monospace"
            >
              {Math.round(t * 100)}%
            </text>
          </g>
        );
      })}

      {rows.map((r, i) => {
        const y = padT + i * rowH + 7;
        const barH = 18;
        const cy = y + barH / 2;
        const barW = r.rate * plotW;
        const xLow = padL + r.ciLow * plotW;
        const xHigh = padL + r.ciHigh * plotW;
        const fill =
          r.config === "all_on" || r.config === "no_injection_filter"
            ? "#15803D"
            : r.config === "all_off"
              ? "#B91C1C"
              : "#B45309";
        return (
          <g key={r.config}>
            <text
              x={padL - 10}
              y={cy + 4}
              textAnchor="end"
              fontSize={11}
              fontWeight={r.config === "all_on" || r.config === "all_off" ? 700 : 500}
              fill="#18181B"
              fontFamily="IBM Plex Mono, monospace"
            >
              {r.config}
            </text>

            <rect
              x={padL}
              y={y}
              width={Math.max(2, barW)}
              height={barH}
              fill={fill}
              opacity={0.85}
              rx={2}
            />

            <line
              x1={xLow}
              y1={cy}
              x2={xHigh}
              y2={cy}
              stroke="#18181B"
              strokeWidth={1.2}
            />
            <line
              x1={xLow}
              y1={cy - 4}
              x2={xLow}
              y2={cy + 4}
              stroke="#18181B"
              strokeWidth={1.2}
            />
            <line
              x1={xHigh}
              y1={cy - 4}
              x2={xHigh}
              y2={cy + 4}
              stroke="#18181B"
              strokeWidth={1.2}
            />

            <text
              x={padL + plotW + 10}
              y={cy + 4}
              fontSize={11}
              fontWeight={600}
              fill="#18181B"
              fontFamily="IBM Plex Mono, monospace"
            >
              {(r.rate * 100).toFixed(1)}% ({r.succeeded}/{r.total})
            </text>
          </g>
        );
      })}
    </svg>
  );
};