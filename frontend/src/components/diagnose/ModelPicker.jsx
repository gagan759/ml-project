import { useMemo } from "react";
import { BrainCircuit, Cpu, Layers3, ShieldCheck, Sparkles } from "lucide-react";
import { useData } from "../../context/DataContext.jsx";
import { modelBenefit, servedName } from "../../lib/analytics.js";
import { cx, ms, pct } from "../../lib/format.js";
import { Badge } from "../ui/primitives.jsx";

export const isClassical = (m) => m?.kind === "classical";
export const familyLabel = (m) => (isClassical(m) ? "Classical ML" : "Deep learning");

function ModelBadges({ m }) {
  return (
    <div className="mt-2.5 flex flex-wrap gap-1.5">
      {m.generation === "upgrade" ? <Badge tone="leaf" icon={Sparkles}>Upgraded model</Badge> : <Badge tone="slate">Original study</Badge>}
      <Badge tone="slate" icon={Layers3}>{familyLabel(m)}</Badge>
      {m.test_metrics?.accuracy != null && <Badge tone="sky">test acc {pct(m.test_metrics.accuracy)}</Badge>}
      {m.confidence_threshold != null && <Badge tone="slate" icon={ShieldCheck}>answers at ≥ {pct(m.confidence_threshold, 0)}</Badge>}
      {m.latency_ms != null && <Badge tone="slate" icon={Cpu}>{ms(m.latency_ms)} CPU</Badge>}
    </div>
  );
}

/** The served models — both diagnose the same five conditions: deep learning (default) or classical ML. */
export default function ModelPicker({ model, onModel, disabled }) {
  const { models } = useData();
  const list = useMemo(() => {
    const all = models?.models || [];
    return [...all].sort((a, b) => (b.key === models?.default_model) - (a.key === models?.default_model));
  }, [models]);
  const selected = list.find((m) => m.key === model) || list[0] || null;

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-medium text-ink">Model</p>
      {list.length ? (
        <div role="radiogroup" aria-label="Model" id="model-select" className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {list.map((m) => {
            const on = m.key === selected?.key;
            const Icon = isClassical(m) ? Layers3 : BrainCircuit;
            return (
              <button
                key={m.key}
                type="button"
                role="radio"
                aria-checked={on}
                disabled={disabled}
                onClick={() => onModel(m.key)}
                className={cx(
                  "flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-left transition disabled:opacity-60",
                  on ? "border-leaf-500 bg-leaf-500/5 ring-2 ring-leaf-500/20" : "border-line-strong bg-surface hover:border-leaf-400",
                )}
              >
                <Icon className={cx("mt-0.5 size-4 shrink-0", on ? "text-leaf-600 dark:text-leaf-400" : "text-muted")} aria-hidden />
                <span className="min-w-0">
                  <span className="block text-sm font-medium text-ink">{servedName(m)}</span>
                  <span className="block text-xs text-muted">
                    {familyLabel(m)}
                    {m.key === models?.default_model ? " · recommended" : ""}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="rounded-xl border border-dashed border-line-strong px-3.5 py-2.5 text-sm text-muted">Models load when the server is online</p>
      )}
      {selected && (
        <p className="mt-1 text-sm font-medium leading-relaxed text-ink-2" data-testid="model-benefit">
          {modelBenefit(selected, list.find((m) => m.key !== selected.key))}
        </p>
      )}
      {selected?.description && <p className="text-xs leading-relaxed text-muted" data-testid="model-card">{selected.description}</p>}
      {selected && <ModelBadges m={selected} />}
    </div>
  );
}
