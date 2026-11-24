import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { classMeta } from "../../lib/classes.js";
import { pct } from "../../lib/format.js";
import { SERIES, tooltipStyle, useChartTheme } from "./theme.js";

const axisProps = (t) => ({ stroke: t.axis, tick: { fill: t.axis, fontSize: 11 }, tickLine: false, axisLine: { stroke: t.grid } });
const pctTick = (v) => `${Math.round(v * 100)}%`;

/* Reliability diagram: accuracy vs confidence per bin, before and after calibration (temperature or vector scaling). */
export function ReliabilityDiagram({ before, after, height = 260 }) {
  const t = useChartTheme();
  const bins = Array.from({ length: 10 }, (_, i) => {
    const lo = i / 10;
    const find = (arr) => arr?.find((b) => Math.abs(b.lo - lo) < 1e-6);
    const a = find(after);
    const b = find(before);
    return { bin: `${Math.round(lo * 100)}–${Math.round(lo * 100 + 10)}`, mid: lo + 0.05, after: a?.accuracy ?? null, before: b?.accuracy ?? null, count: a?.count ?? 0, conf: a?.confidence ?? null };
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={bins} margin={{ top: 10, right: 12, left: -8, bottom: 4 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="mid" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickFormatter={pctTick} {...axisProps(t)} label={{ value: "confidence", position: "insideBottomRight", offset: -2, fill: t.axis, fontSize: 11 }} />
        <YAxis domain={[0, 1]} tickFormatter={pctTick} {...axisProps(t)} />
        <Tooltip {...tooltipStyle(t)} formatter={(v, n) => [v == null ? "—" : pct(v), n]} labelFormatter={(v) => `confidence ${pct(v - 0.05, 0)}–${pct(v + 0.05, 0)}`} />
        <Legend wrapperStyle={{ fontSize: 12, color: t.axis }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={t.axis} strokeDasharray="4 4" ifOverflow="extendDomain" />
        {before && <Bar dataKey="before" name="before calibration" fill={t.warn} fillOpacity={0.35} barSize={14} />}
        <Bar dataKey="after" name="after calibration" fill={t.accent} barSize={14} radius={[3, 3, 0, 0]} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* Risk–coverage: accuracy on the photos the model answers vs the share it answers, by threshold. */
export function RiskCoverage({ curve, chosen, height = 260 }) {
  const t = useChartTheme();
  const data = (curve || []).filter((r) => r.accuracy_kept != null).map((r) => ({ ...r }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 14, left: -8, bottom: 4 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="coverage" type="number" domain={[0, 1]} reversed tickFormatter={pctTick} {...axisProps(t)} label={{ value: "share answered", position: "insideBottomRight", offset: -2, fill: t.axis, fontSize: 11 }} />
        <YAxis dataKey="accuracy_kept" domain={["dataMin - 0.02", 1]} tickFormatter={pctTick} {...axisProps(t)} />
        <Tooltip {...tooltipStyle(t)} formatter={(v, n) => [pct(v), n === "accuracy_kept" ? "accuracy when answering" : n]} labelFormatter={(v) => `answers ${pct(v)} of photos`} />
        <Line type="monotone" dataKey="accuracy_kept" stroke={t.accent} strokeWidth={2.2} dot={false} isAnimationActive />
        {chosen && <ReferenceDot x={chosen.coverage} y={chosen.accuracy_kept} r={6} fill={t.warn} stroke={t.surface} strokeWidth={2} />}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* Per-class F1 (or recall/precision) for up to 3 models side by side. */
export function PerClassChart({ classes, series, metric = "f1", height = 300 }) {
  const t = useChartTheme();
  const data = classes.map((k) => {
    const row = { key: k, label: classMeta(k).abbr, name: classMeta(k).name };
    series.forEach((s, i) => {
      row[`m${i}`] = s.perClass.find((p) => p.key === k)?.[metric] ?? null;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 8, left: -12, bottom: 4 }} barGap={2}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="label" {...axisProps(t)} />
        <YAxis domain={[0, 1]} tickFormatter={pctTick} {...axisProps(t)} />
        <Tooltip {...tooltipStyle(t)} labelFormatter={(l, p) => p?.[0]?.payload?.name || l} formatter={(v, n) => [v == null ? "—" : pct(v), n]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => (
          <Bar key={s.key} dataKey={`m${i}`} name={s.label} fill={s.color || SERIES[i]} radius={[3, 3, 0, 0]} maxBarSize={26} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Out-of-fold scores of each cross-validation fold per recipe (dots) with the pooled OOF score (bar). */
export function FoldStrip({ recipes, metric = "f1_macro", height }) {
  const t = useChartTheme();
  const points = [];
  recipes.forEach((r, i) =>
    r.per_fold.forEach((f) => points.push({ x: f.value, y: i, fold: f.fold, recipe: r.label })),
  );
  const pooled = recipes.map((r, i) => ({ x: r.pooled, y: i, recipe: r.label }));
  const h = height || recipes.length * 46 + 50;
  const min = Math.max(0, Math.floor((Math.min(...points.map((p) => p.x), ...pooled.map((p) => p.x)) - 0.05) * 20) / 20);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <ScatterChart margin={{ top: 10, right: 16, left: 4, bottom: 8 }}>
        <CartesianGrid stroke={t.grid} horizontal={false} />
        <XAxis type="number" dataKey="x" domain={[min, 1]} tickFormatter={pctTick} {...axisProps(t)} name={metric} />
        <YAxis type="number" dataKey="y" domain={[-0.5, recipes.length - 0.5]} ticks={recipes.map((_, i) => i)} tickFormatter={(i) => recipes[i]?.short || ""} width={150} reversed {...axisProps(t)} tick={{ fill: t.ink, fontSize: 11 }} />
        <ZAxis range={[60, 60]} />
        <Tooltip {...tooltipStyle(t)} formatter={(v, n) => (n === "y" ? [null, null] : [pct(v), "macro-F1"])} labelFormatter={() => ""} content={({ payload }) => {
          const p = payload?.[0]?.payload;
          if (!p) return null;
          return (
            <div style={tooltipStyle(t).contentStyle}>
              <div style={{ fontWeight: 600 }}>{p.recipe}</div>
              <div>{p.fold != null ? `fold ${p.fold}: ` : "pooled out-of-fold: "}{pct(p.x)}</div>
            </div>
          );
        }} />
        <Scatter data={points} fill={t.accent2} fillOpacity={0.55} />
        <Scatter data={pooled} fill={t.accent} shape="diamond" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* Training curves averaged over the 5 folds (held-out macro-F1 and training loss). */
export function TrainingCurves({ curves, height = 260 }) {
  const t = useChartTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={curves} margin={{ top: 10, right: 6, left: -12, bottom: 4 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="epoch" {...axisProps(t)} />
        <YAxis yAxisId="f1" domain={[0, 1]} tickFormatter={pctTick} {...axisProps(t)} />
        <YAxis yAxisId="loss" orientation="right" {...axisProps(t)} />
        <Tooltip {...tooltipStyle(t)} formatter={(v, n) => [n.includes("loss") ? Number(v).toFixed(3) : pct(v), n]} labelFormatter={(e) => `epoch ${e}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line yAxisId="f1" type="monotone" dataKey="val_f1" name="held-out macro-F1" stroke={t.accent} strokeWidth={2.2} dot={false} />
        <Line yAxisId="f1" type="monotone" dataKey="val_acc" name="held-out accuracy" stroke={t.accent2} strokeWidth={1.8} dot={false} />
        <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="training loss" stroke={t.warn} strokeWidth={1.6} strokeDasharray="5 4" dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* CPU latency (log x) vs macro-F1, one dot per model. */
export function LatencyScatter({ points, height = 300 }) {
  const t = useChartTheme();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 16, left: -6, bottom: 8 }}>
        <CartesianGrid stroke={t.grid} />
        <XAxis type="number" dataKey="latency" scale="log" domain={["auto", "auto"]} tickFormatter={(v) => `${Math.round(v)} ms`} {...axisProps(t)} name="CPU latency" />
        <YAxis type="number" dataKey="f1" domain={[0, 1]} tickFormatter={pctTick} {...axisProps(t)} name="macro-F1" />
        <ZAxis range={[90, 90]} />
        <Tooltip {...tooltipStyle(t)} content={({ payload }) => {
          const p = payload?.[0]?.payload;
          if (!p) return null;
          return (
            <div style={tooltipStyle(t).contentStyle}>
              <div style={{ fontWeight: 600 }}>{p.label}</div>
              <div>macro-F1 {pct(p.f1)} · accuracy {pct(p.acc)}</div>
              <div>{Math.round(p.latency)} ms per photo on CPU</div>
            </div>
          );
        }} />
        <Scatter data={points}>
          {points.map((p) => (
            <Cell key={p.key} fill={p.color} stroke={p.highlight ? t.ink : "none"} strokeWidth={2} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* Photos per class (stacked by split). */
export function ClassDistribution({ classes, height = 260 }) {
  const t = useChartTheme();
  const data = classes.map((c) => ({ ...c, label: classMeta(c.key).abbr }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 6, left: -14, bottom: 4 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="label" {...axisProps(t)} />
        <YAxis {...axisProps(t)} />
        <Tooltip {...tooltipStyle(t)} labelFormatter={(l, p) => p?.[0]?.payload?.name || l} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="train" name="train" stackId="s" fill={t.accent} />
        <Bar dataKey="val" name="validation" stackId="s" fill={t.accent2} />
        <Bar dataKey="test" name="test" stackId="s" fill={t.warn} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* Prediction flip rates under harmless changes, old vs new model. */
export function RobustnessChart({ rows, series, height }) {
  const t = useChartTheme();
  const h = height || rows.length * 38 + 60;
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 40, left: 4, bottom: 4 }} barGap={2}>
        <CartesianGrid stroke={t.grid} horizontal={false} />
        <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={pctTick} {...axisProps(t)} />
        <YAxis type="category" dataKey="label" width={170} {...axisProps(t)} tick={{ fill: t.ink, fontSize: 11.5 }} />
        <Tooltip {...tooltipStyle(t)} formatter={(v, n) => [pct(v), n]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color || SERIES[i]} radius={[0, 3, 3, 0]} barSize={12}>
            <LabelList dataKey={s.key} position="right" formatter={(v) => (v == null ? "" : pct(v, 1))} style={{ fill: t.axis, fontSize: 10.5 }} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
