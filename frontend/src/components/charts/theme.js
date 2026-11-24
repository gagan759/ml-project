import { useTheme } from "../../context/ThemeContext.jsx";

const LIGHT = { grid: "#e2e8df", axis: "#5f6f66", ink: "#0e1a14", surface: "#ffffff", muted: "#8a988f", accent: "#15834e", accent2: "#0891b2", warn: "#d6760b", bad: "#db3a25", track: "#e8ede5" };
const DARK = { grid: "#1f2c25", axis: "#8b9a92", ink: "#e9f0ec", surface: "#0d1411", muted: "#6b7a72", accent: "#46be7d", accent2: "#22d3ee", warn: "#f8b73e", bad: "#fb7a69", track: "#18241e" };

/** Concrete chart colours for the current theme (SVG attributes do not resolve CSS variables everywhere). */
export function useChartTheme() {
  const { dark } = useTheme();
  return dark ? DARK : LIGHT;
}

export const SERIES = ["#15834e", "#0891b2", "#7c3aed", "#d6760b", "#db2777", "#65a30d", "#2563eb", "#b45309"];

export function tooltipStyle(t) {
  return {
    contentStyle: {
      background: t.surface,
      border: `1px solid ${t.grid}`,
      borderRadius: 12,
      boxShadow: "0 12px 32px -12px rgb(0 0 0 / 0.25)",
      color: t.ink,
      fontSize: 12,
      padding: "8px 10px",
    },
    labelStyle: { color: t.ink, fontWeight: 600, marginBottom: 4 },
    itemStyle: { color: t.ink, padding: 0 },
    cursor: { fill: t.track, opacity: 0.5 },
  };
}
