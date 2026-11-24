import { useEffect, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { ArrowRight, Menu, X } from "lucide-react";
import { cx } from "../../lib/format.js";
import { ButtonLink } from "../ui/primitives.jsx";
import { ApiStatus, Logo, ScrollToTop, SiteFooter, ThemeToggle } from "./chrome.jsx";

const LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/lab", label: "Model lab" },
  { to: "/library", label: "Diseases" },
  { to: "/methodology", label: "Methodology" },
  { to: "/docs", label: "Docs" },
];

export default function MarketingLayout() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="min-h-dvh">
      <ScrollToTop />
      <header className={cx("sticky top-0 z-40 transition", scrolled ? "border-b border-line bg-surface/80 backdrop-blur-xl" : "border-b border-transparent")}>
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Logo />
          <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} className="rounded-lg px-3 py-2 text-sm font-medium text-ink-2 transition hover:bg-surface-2 hover:text-ink">
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="hidden items-center gap-3 md:flex">
            <ApiStatus />
            <ThemeToggle />
            <ButtonLink to="/diagnose" size="sm" iconRight={ArrowRight}>
              Diagnose a leaf
            </ButtonLink>
          </div>
          <button className="grid size-9 place-items-center rounded-lg hover:bg-surface-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu" aria-expanded={open}>
            {open ? <X className="size-5" aria-hidden /> : <Menu className="size-5" aria-hidden />}
          </button>
        </div>
        {open && (
          <div className="border-t border-line bg-surface px-4 pb-4 md:hidden">
            <nav className="flex flex-col py-2" aria-label="Mobile">
              {LINKS.map((l) => (
                <Link key={l.to} to={l.to} onClick={() => setOpen(false)} className="rounded-lg px-3 py-2.5 text-[15px] font-medium text-ink-2 hover:bg-surface-2">
                  {l.label}
                </Link>
              ))}
            </nav>
            <div className="flex items-center justify-between gap-3 pt-2">
              <ApiStatus side="top" align="start" />
              <ThemeToggle />
            </div>
            <ButtonLink to="/diagnose" className="mt-4 w-full" iconRight={ArrowRight}>
              Diagnose a leaf
            </ButtonLink>
          </div>
        )}
      </header>
      <main id="main">
        <Outlet />
      </main>
      <SiteFooter />
    </div>
  );
}
