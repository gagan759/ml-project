export const pct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(digits)}%`);
export const num = (v, digits = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(digits));
export const ms = (v) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);
export const ci = (pair, digits = 1) => (pair ? `${pct(pair[0], digits)} – ${pct(pair[1], digits)}` : "—");
export const signedPct = (v, digits = 1) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(100 * v).toFixed(digits)} pts`);

export function compactParams(n) {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

export function bytes(n) {
  if (n == null) return "—";
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} B`;
}

export function timeAgo(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export const cx = (...parts) => parts.filter(Boolean).join(" ");
