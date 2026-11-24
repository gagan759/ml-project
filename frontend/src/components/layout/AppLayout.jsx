import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpenText,
  BrainCircuit,
  FileCode2,
  FlaskConical,
  LayoutDashboard,
  LineChart,
  Menu,
  Microscope,
  ScanSearch,
  X,
} from "lucide-react";
import { cx } from "../../lib/format.js";
import { ApiStatus, Logo, ScrollToTop, SiteFooter, ThemeToggle } from "./chrome.jsx";

export const NAV = [
  {
    group: "Workspace",
    items: [
      { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
      { to: "/diagnose", label: "Diagnose", icon: ScanSearch, accent: true },
    ],
  },
  {
    group: "Models",
    items: [
      { to: "/lab", label: "Model lab", icon: FlaskConical },
      { to: "/performance", label: "Performance", icon: LineChart },
      { to: "/explainability", label: "Explainability", icon: BrainCircuit },
    ],
  },
  {
    group: "Knowledge",
    items: [
      { to: "/library", label: "Disease library", icon: BookOpenText },
      { to: "/methodology", label: "Methodology", icon: Microscope },
      { to: "/docs", label: "Documentation", icon: FileCode2 },
    ],
  },
];

function NavItems({ onNavigate }) {
  return (
    <nav className="flex flex-col gap-6" aria-label="Main">
      {NAV.map((section) => (
        <div key={section.group}>
          <p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted/80">{section.group}</p>
          <ul className="flex flex-col gap-0.5">
            {section.items.map(({ to, label, icon: Icon, accent }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cx(
                      "group flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition",
                      isActive ? "bg-leaf-500/10 text-leaf-700 dark:text-leaf-300" : "text-ink-2 hover:bg-surface-2 hover:text-ink",
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <Icon className={cx("size-[18px]", isActive ? "text-leaf-600 dark:text-leaf-400" : "text-muted group-hover:text-ink")} aria-hidden />
                      <span className="flex-1">{label}</span>
                      {accent && !isActive && <span className="rounded-md bg-leaf-500/12 px-1.5 py-0.5 text-[10px] font-semibold text-leaf-700 dark:text-leaf-300">AI</span>}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export default function AppLayout() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <div className="min-h-dvh">
      <ScrollToTop />
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>

      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r border-line bg-surface/80 backdrop-blur-xl lg:flex">
        <div className="flex h-16 items-center px-5">
          <Logo />
        </div>
        <div className="scrollbar-thin flex-1 overflow-y-auto px-3 py-4">
          <NavItems />
        </div>
        <div className="flex items-center justify-between gap-2 border-t border-line px-4 py-3">
          <ApiStatus side="top" align="start" />
          <ThemeToggle />
        </div>
      </aside>

      {/* top bar (mobile + tablet) */}
      <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-line bg-surface/85 px-4 backdrop-blur-xl lg:hidden">
        <Logo />
        <div className="flex items-center gap-2">
          <ApiStatus showLabel={false} />
          <button
            className="grid size-9 place-items-center rounded-lg text-ink hover:bg-surface-2"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
            aria-expanded={open}
          >
            <Menu className="size-5" aria-hidden />
          </button>
        </div>
      </header>

      <AnimatePresence>
        {open && (
          <motion.div className="fixed inset-0 z-50 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <button className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setOpen(false)} aria-label="Close navigation" />
            <motion.aside
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", stiffness: 380, damping: 36 }}
              className="absolute inset-y-0 left-0 flex w-[84%] max-w-xs flex-col bg-surface shadow-pop"
            >
              <div className="flex h-14 items-center justify-between px-4">
                <Logo />
                <button className="grid size-9 place-items-center rounded-lg hover:bg-surface-2" onClick={() => setOpen(false)} aria-label="Close navigation">
                  <X className="size-5" aria-hidden />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto px-3 py-3">
                <NavItems onNavigate={() => setOpen(false)} />
              </div>
              <div className="flex items-center justify-between border-t border-line px-4 py-3">
                <ApiStatus side="top" align="start" />
                <ThemeToggle />
              </div>
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="lg:pl-64">
        <main id="main" className="mx-auto min-h-[calc(100dvh-8rem)] max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
          <Outlet />
        </main>
        <SiteFooter />
      </div>
    </div>
  );
}
