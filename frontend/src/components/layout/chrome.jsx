import { useEffect, useId } from "react";
import { Link, useLocation } from "react-router-dom";
import { Monitor, Moon, Sun } from "lucide-react";
import { useData } from "../../context/DataContext.jsx";
import { useTheme } from "../../context/ThemeContext.jsx";
import { cx } from "../../lib/format.js";
import { Tip } from "../ui/primitives.jsx";

const SEPALS = [
  [14, 1],
  [166, 1],
  [-28, 0.9],
  [-152, 0.9],
  [90, 0.6],
];

/** The Tomato Care mark (same artwork as assets/brand/mark.svg); ids are per instance so several can share a page. */
export function LogoMark({ className = "size-9" }) {
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <defs>
        <linearGradient id={`${id}t`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#17744a" />
          <stop offset="1" stopColor="#062619" />
        </linearGradient>
        <radialGradient id={`${id}f`} cx="0.36" cy="0.32" r="0.8">
          <stop offset="0" stopColor="#ff967b" />
          <stop offset="0.45" stopColor="#f0543f" />
          <stop offset="1" stopColor="#b3241a" />
        </radialGradient>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#86e8ae" />
          <stop offset="1" stopColor="#2fae6a" />
        </linearGradient>
        <path id={`${id}p`} d="M0 0C3.2-2.6 8.6-2.4 12.2 0 8.6 2.4 3.2 2.6 0 0Z" />
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${id}t)`} />
      <path
        d="M32 21.4C37.8 19.3 46.9 20.1 51.2 27.8c3.8 6.6 2.1 16.6-4.9 22.4C41.5 54.2 22.5 54.2 17.7 50.2 10.7 44.4 9 34.4 12.8 27.8 17.1 20.1 26.2 19.3 32 21.4Z"
        fill={`url(#${id}f)`}
      />
      <ellipse cx="22.6" cy="29.6" rx="4.4" ry="2.4" transform="rotate(-38 22.6 29.6)" fill="#fff" opacity=".32" />
      <g fill={`url(#${id}s)`} stroke="#0b4029" strokeWidth=".8" strokeLinejoin="round">
        {SEPALS.map(([angle, scale]) => (
          <use key={angle} href={`#${id}p`} transform={`translate(32 21.6) rotate(${angle}) scale(${scale})`} />
        ))}
      </g>
      <path d="M32 21.8c-.6-4 .3-7.2 3.6-9.4" fill="none" stroke="#86e8ae" strokeWidth="3" strokeLinecap="round" />
      <path
        d="M14.6 39.6h8.2l2.9-6.1 4.7 11.6 3.9-8.7 2.4 3.2h12.7"
        fill="none"
        stroke="#fff"
        strokeWidth="2.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Logo({ className = "", compact = false }) {
  return (
    <Link to="/" className={cx("group flex items-center gap-2.5", className)} aria-label="Tomato Care home">
      <LogoMark className="size-9 shrink-0 drop-shadow-sm transition-transform duration-300 group-hover:-rotate-6" />
      {!compact && (
        <span className="font-display text-[17px] font-semibold tracking-tight text-ink">
          Tomato<span className="text-leaf-600 dark:text-leaf-400"> Care</span>
        </span>
      )}
    </Link>
  );
}

const THEME_OPTIONS = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
];

export function ThemeToggle({ className = "" }) {
  const { mode, setMode } = useTheme();
  return (
    <div className={cx("inline-flex rounded-lg border border-line bg-surface-2 p-0.5", className)} role="radiogroup" aria-label="Colour theme">
      {THEME_OPTIONS.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={mode === value}
          aria-label={label}
          title={label}
          onClick={() => setMode(value)}
          className={cx("grid size-7 place-items-center rounded-md transition", mode === value ? "bg-surface text-ink shadow-sm" : "text-muted hover:text-ink")}
        >
          <Icon className="size-3.5" aria-hidden />
        </button>
      ))}
    </div>
  );
}

export function ApiStatus({ className = "", showLabel = true, side = "bottom", align = "end" }) {
  const { health } = useData();
  const map = {
    online: { dot: "bg-leaf-500", label: "API online", tip: `Models loaded: ${health.data?.loaded_models?.length ?? 0} · device ${health.data?.device ?? "cpu"}` },
    degraded: { dot: "bg-sun-500", label: "API degraded", tip: "The server is running but the default model did not load." },
    offline: { dot: "bg-tomato-500", label: "API offline", tip: "Read-only mode: analytics come from the bundled snapshot; diagnosis needs the server." },
    checking: { dot: "bg-muted animate-pulse", label: "Connecting…", tip: "Checking the diagnosis server…" },
  }[health.status];
  return (
    <Tip content={map.tip} side={side} align={align} className={className}>
      <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink-2" tabIndex={0}>
        <span className="relative flex size-2">
          {health.status === "online" && <span className="absolute inline-flex size-full animate-ping rounded-full bg-leaf-400 opacity-60" />}
          <span className={cx("relative inline-flex size-2 rounded-full", map.dot)} />
        </span>
        {showLabel && map.label}
      </span>
    </Tip>
  );
}

export function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) {
      const el = document.getElementById(hash.slice(1));
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
    }
    window.scrollTo({ top: 0 });
  }, [pathname, hash]);
  return null;
}

export function useDocumentTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · Tomato Care` : "Tomato Care — Tomato Leaf Disease Diagnosis";
  }, [title]);
}

export function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-8 text-sm text-muted sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
        <div className="flex items-center gap-3">
          <Logo compact />
          <p>
            Tomato Care · a research prototype, <strong className="font-medium text-ink-2">not a substitute for an agronomist</strong>.
          </p>
        </div>
        <p className="max-w-xl md:text-right">
          Images: Tomato-Village dataset (Gehlot et al., <em>Multimedia Systems</em> 2023,{" "}
          <a className="underline decoration-line-strong underline-offset-2 hover:text-ink" href="https://doi.org/10.1007/s00530-023-01158-y" target="_blank" rel="noreferrer">
            doi:10.1007/s00530-023-01158-y
          </a>
          ).
        </p>
      </div>
    </footer>
  );
}
