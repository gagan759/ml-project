import { useMemo, useState } from "react";
import { classMeta } from "../../lib/classes.js";
import { cx, pct } from "../../lib/format.js";
import { Segmented } from "../ui/primitives.jsx";

/**
 * Interactive confusion matrix. Rows = true class, columns = predicted class.
 * "Row %" shows recall per row (how each true class was predicted); "Count" shows raw photo counts.
 */
export default function ConfusionMatrix({ matrix, classes, title, compact = false, defaultMode = "row" }) {
  const [mode, setMode] = useState(defaultMode);
  const [hover, setHover] = useState(null);
  const data = useMemo(() => {
    const m = (matrix || []).map((r) => r.map(Number));
    const rowSums = m.map((r) => r.reduce((a, b) => a + b, 0));
    const total = rowSums.reduce((a, b) => a + b, 0);
    const correct = m.reduce((a, r, i) => a + (r[i] || 0), 0);
    return { m, rowSums, total, correct };
  }, [matrix]);
  if (!matrix?.length) return null;
  const n = classes.length;
  const cell = (i, j) => {
    const v = data.m[i][j];
    const frac = data.rowSums[i] ? v / data.rowSums[i] : 0;
    return { v, frac };
  };
  const label = (key) => classMeta(key).abbr;

  return (
    <div className="w-full">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          {title && <p className="truncate text-sm font-semibold text-ink">{title}</p>}
          <p className="text-xs text-muted">
            {data.correct}/{data.total} correct · rows = true class, columns = prediction
          </p>
        </div>
        <Segmented
          size="sm"
          label="Matrix values"
          value={mode}
          onChange={setMode}
          options={[
            { value: "row", label: "Row %" },
            { value: "count", label: "Count" },
          ]}
        />
      </div>
      <div className="overflow-x-auto scrollbar-thin">
        <table className="mx-auto border-separate" style={{ borderSpacing: compact ? 2 : 3 }}>
          <thead>
            <tr>
              <th className="p-0" />
              {classes.map((k) => (
                <th key={k} scope="col" className="p-0 pb-1 text-center align-bottom">
                  <span className="inline-block font-mono text-[11px] font-semibold" style={{ color: classMeta(k).color }} title={`Predicted: ${classMeta(k).name}`}>
                    {label(k)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classes.map((rk, i) => (
              <tr key={rk}>
                <th scope="row" className="pr-2 text-right">
                  <span className="font-mono text-[11px] font-semibold" style={{ color: classMeta(rk).color }} title={`True: ${classMeta(rk).name}`}>
                    {label(rk)}
                  </span>
                </th>
                {classes.map((ck, j) => {
                  const { v, frac } = cell(i, j);
                  const diag = i === j;
                  const alpha = Math.min(1, frac * 1.05);
                  const bg = diag ? `rgb(21 131 78 / ${0.08 + alpha * 0.85})` : v ? `rgb(219 58 37 / ${0.1 + alpha * 0.8})` : "transparent";
                  const strong = alpha > 0.5;
                  const active = hover && hover.i === i && hover.j === j;
                  return (
                    <td key={ck} className="p-0">
                      <button
                        type="button"
                        onMouseEnter={() => setHover({ i, j })}
                        onFocus={() => setHover({ i, j })}
                        onMouseLeave={() => setHover(null)}
                        onBlur={() => setHover(null)}
                        className={cx(
                          "grid place-items-center rounded-md border font-mono tabular transition",
                          compact ? "size-9 text-[10px] sm:size-10" : "size-10 text-[11px] sm:size-12 sm:text-xs",
                          v === 0 ? "border-line/60 text-muted/50" : "border-transparent",
                          strong ? "text-white" : "text-ink",
                          active && "ring-2 ring-ink/70",
                        )}
                        style={{ background: bg }}
                        aria-label={`True ${classMeta(rk).name}, predicted ${classMeta(ck).name}: ${v} photos (${pct(frac, 0)} of row)`}
                      >
                        {mode === "count" ? v : v ? Math.round(frac * 100) : "·"}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 min-h-5 text-center text-xs text-muted" aria-live="polite">
        {hover
          ? `${classMeta(classes[hover.i]).name} → predicted ${classMeta(classes[hover.j]).name}: ${data.m[hover.i][hover.j]} of ${data.rowSums[hover.i]} photos (${pct(cell(hover.i, hover.j).frac, 1)})`
          : `${n} classes · hover or focus a cell for details`}
      </p>
    </div>
  );
}
