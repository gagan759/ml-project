import { useMemo } from "react";
import { CheckCircle2, CircleHelp, ShieldAlert, Sparkles, TriangleAlert } from "lucide-react";
import { isClassicalModel, servedName } from "../../lib/analytics.js";
import { classMeta } from "../../lib/classes.js";
import { cx, ms, pct } from "../../lib/format.js";
import { Badge } from "../ui/primitives.jsx";

const STATUS_ICON = { Healthy: CheckCircle2, Diseased: ShieldAlert, Uncertain: CircleHelp, "Unusual image": TriangleAlert };
const STATUS_TONE = { Healthy: "leaf", Diseased: "tomato", Uncertain: "sun", "Unusual image": "slate" };

/** Every model's answer for the same photo, with a vote summary. */
export default function CompareTable({ results, trueKey, models }) {
  const meta = useMemo(() => Object.fromEntries((models?.models || []).map((m) => [m.key, m])), [models]);
  const votes = useMemo(() => {
    const tally = {};
    results.forEach((r) => {
      tally[r.prediction.key] = (tally[r.prediction.key] || 0) + 1;
    });
    return Object.entries(tally).sort((a, b) => b[1] - a[1]);
  }, [results]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted">Votes:</span>
        {votes.map(([k, n]) => (
          <span key={k} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-0.5 text-xs">
            <span className="size-2 rounded-full" style={{ background: classMeta(k).color }} />
            {classMeta(k).short} <strong className="tabular">{n}</strong>
          </span>
        ))}
      </div>
      <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-4 py-2.5 font-medium">Model</th>
              <th className="px-4 py-2.5 font-medium">Prediction</th>
              <th className="px-4 py-2.5 text-right font-medium">Confidence</th>
              <th className="px-4 py-2.5 font-medium">Decision</th>
              <th className="px-4 py-2.5 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {results.map((r) => {
              const m = meta[r.model.key];
              const Icon = STATUS_ICON[r.prediction.status];
              const right = trueKey ? r.prediction.key === trueKey : null;
              return (
                <tr key={r.model.key} className="bg-surface">
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-ink">{r.model.name}</span>
                      {m?.generation === "upgrade" && <Sparkles className="size-3.5 text-leaf-600" aria-label="upgraded model" />}
                    </div>
                    <p className="text-xs text-muted">
                      {servedName(m || r.model)} · {isClassicalModel(m) || isClassicalModel(r.model) ? "classical ML" : "deep learning"}
                    </p>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={cx("inline-flex items-center gap-1.5", right === false && "text-tomato-600")}>
                      <span className="size-2 rounded-full" style={{ background: classMeta(r.prediction.key).color }} />
                      {classMeta(r.prediction.key).name}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono tabular">{pct(r.prediction.confidence)}</td>
                  <td className="px-4 py-2.5">
                    <Badge tone={STATUS_TONE[r.prediction.status]} icon={Icon}>
                      {r.prediction.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs tabular text-muted">{ms(r.inference_ms)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
