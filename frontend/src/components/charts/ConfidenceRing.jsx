import { motion } from "framer-motion";
import { pct } from "../../lib/format.js";

/** Radial gauge of the calibrated confidence, with a tick at the model's threshold. */
export default function ConfidenceRing({ value = 0, threshold, color = "#15834e", size = 132, label = "confidence" }) {
  const stroke = 11;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  const angle = threshold != null ? threshold * 360 - 90 : null;
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - v) }}
          transition={{ duration: 0.9, ease: [0.2, 0.7, 0.2, 1] }}
        />
      </svg>
      {angle != null && (
        <span
          className="absolute h-[14px] w-[3px] rounded-full bg-ink/70"
          style={{
            left: size / 2 - 1.5 + r * Math.cos((angle * Math.PI) / 180),
            top: size / 2 - 7 + r * Math.sin((angle * Math.PI) / 180),
            transform: `rotate(${angle + 90}deg)`,
          }}
          title={`Threshold ${pct(threshold, 0)}`}
        />
      )}
      <div className="absolute text-center">
        <p className="font-display text-2xl font-semibold tabular text-ink">{pct(v, 1)}</p>
        <p className="text-[11px] uppercase tracking-wider text-muted">{label}</p>
      </div>
    </div>
  );
}
