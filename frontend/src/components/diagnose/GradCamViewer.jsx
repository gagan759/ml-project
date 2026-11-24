import { useState } from "react";
import { Columns2, Layers, Scan } from "lucide-react";
import { cx } from "../../lib/format.js";
import { InfoTip, Segmented } from "../ui/primitives.jsx";

/**
 * Grad-CAM viewer. `input` is the 256x256 image the model actually received; `heatmap` is an RGBA PNG whose alpha
 * encodes the Grad-CAM strength, so it can be blended live with an opacity slider. Without a heat map or overlay (the
 * classical ML model has no network to explain) it shows the model input with the server's note instead.
 */
export default function GradCamViewer({ input, heatmap, overlay, note, className = "" }) {
  const [mode, setMode] = useState("overlay");
  const [opacity, setOpacity] = useState(0.7);
  const canBlend = Boolean(input && heatmap);
  if (!canBlend && !overlay) {
    if (!input && !note) return null;
    return (
      <div className={className}>
        <p className="mb-3 text-sm font-semibold text-ink">Grad-CAM</p>
        <div className="flex items-start gap-3 rounded-xl bg-surface-2 p-3">
          {input && <img src={input} alt="Model input after preprocessing" className="size-20 shrink-0 rounded-lg border border-line object-cover" />}
          <p className="text-sm leading-relaxed text-muted">{note || "No heat map is available for this model."}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-ink">
          Grad-CAM
          <InfoTip content="Brighter areas raised the score of the predicted class the most (gradient-weighted activations of the last feature layer). It shows where the network looked, not proof of the cause." />
        </div>
        {canBlend && (
          <Segmented
            size="sm"
            label="Grad-CAM view"
            value={mode}
            onChange={setMode}
            options={[
              { value: "overlay", label: "Overlay", icon: Layers },
              { value: "split", label: "Side by side", icon: Columns2 },
              { value: "input", label: "Model input", icon: Scan },
            ]}
          />
        )}
      </div>

      {canBlend ? (
        <div className={cx("grid gap-3", mode === "split" ? "grid-cols-2" : "grid-cols-1")}>
          {(mode === "split" || mode === "input") && (
            <figure className="overflow-hidden rounded-xl border border-line bg-surface-3">
              <img src={input} alt="Model input after preprocessing" className="aspect-square w-full object-cover" />
              {mode === "split" && <figcaption className="px-2 py-1.5 text-[11px] text-muted">Model input (256×256)</figcaption>}
            </figure>
          )}
          {mode !== "input" && (
            <figure className="relative overflow-hidden rounded-xl border border-line bg-black">
              <img src={input} alt="" className="aspect-square w-full object-cover" />
              <img src={heatmap} alt="Grad-CAM heat map" className="absolute inset-0 aspect-square w-full object-cover" style={{ opacity }} />
              {mode === "split" && <figcaption className="absolute bottom-0 w-full bg-black/50 px-2 py-1.5 text-[11px] text-white/80">Where the network looked</figcaption>}
            </figure>
          )}
        </div>
      ) : (
        <img src={overlay} alt="Grad-CAM overlay" className="aspect-square w-full rounded-xl border border-line object-cover" />
      )}

      {canBlend && mode !== "input" && (
        <label className="mt-3 flex items-center gap-3 text-xs text-muted">
          Heat-map strength
          <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1 accent-leaf-600" aria-label="Heat-map opacity" />
          <span className="w-9 text-right font-mono tabular">{Math.round(opacity * 100)}%</span>
        </label>
      )}
      {note && <p className="mt-2 text-xs text-muted">{note}</p>}
    </div>
  );
}
