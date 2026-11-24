import { motion } from "framer-motion";
import { classMeta } from "../../lib/classes.js";
import { cx, pct } from "../../lib/format.js";

/** Calibrated probability of every class the model knows, sorted, with the model's confidence threshold marked. */
export default function ProbabilityBars({ probabilities, threshold, trueKey, compact = false }) {
  const rows = [...(probabilities || [])].sort((a, b) => b.probability - a.probability);
  const top = rows[0]?.key;
  return (
    <ul className={cx("flex flex-col", compact ? "gap-2" : "gap-3")}>
      {rows.map((p, i) => {
        const meta = classMeta(p.key);
        const Icon = meta.icon;
        return (
          <li key={p.key}>
            <div className="mb-1 flex items-center justify-between gap-3 text-sm">
              <span className="flex min-w-0 items-center gap-2">
                <span className="grid size-5 shrink-0 place-items-center rounded-md" style={{ background: `${meta.color}1f`, color: meta.color }}>
                  <Icon className="size-3" aria-hidden />
                </span>
                <span className={cx("truncate", p.key === top ? "font-semibold text-ink" : "text-ink-2")}>{meta.name}</span>
                {trueKey === p.key && <span className="rounded bg-surface-3 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-muted">true</span>}
              </span>
              <span className={cx("font-mono text-[13px] tabular", p.key === top ? "font-semibold text-ink" : "text-muted")}>{pct(p.probability)}</span>
            </div>
            <div className="relative h-2 overflow-hidden rounded-full bg-surface-3">
              <motion.div
                className="h-full rounded-full"
                style={{ background: meta.color, opacity: p.key === top ? 1 : 0.55 }}
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(0.5, p.probability * 100)}%` }}
                transition={{ duration: 0.7, delay: i * 0.04, ease: [0.2, 0.7, 0.2, 1] }}
              />
              {threshold != null && i === 0 && (
                <span className="absolute inset-y-0 w-px bg-ink/60" style={{ left: `${threshold * 100}%` }} title={`Threshold ${pct(threshold, 0)}`} />
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
